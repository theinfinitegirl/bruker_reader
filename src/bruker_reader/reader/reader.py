from __future__ import annotations  # Support '|' in typing
import os.path
from glob import glob
from struct import Struct
from collections import namedtuple
import numpy as np
import pandas as pd
from typing import Generator

# This reader is heavily inspired (well, essentially cribbed from) the work
# done in the 'rtms' R package:
#   https://cran.r-project.org/package=rtms
# It has been adapted for Python and modified to allow reading of partial
# spectra.
# I have not been able to find documentation of the BAF file format elsewhere,
# so I'm eternally grateful to the 'rtms' authors.

# Lookup table for BAF offsets. Given two bytes (idA, idB) returns the
# type of the offset.
OFFSETS = {
    (0x01, 0x30): "header",
    (0x02, 0x20): "metadata",
    (0x01, 0x10): "spectrum",
    (0x02, 0x10): "maxima"
    }

Offset = namedtuple("Offset", "loc page")


class NamedStruct:
    """ Helper class that combines struct.Struct with a namedtuple"""
    def __init__(self, name, fmt, field_names):
        """Create a named struct.
        'name' and 'field_names' are the arguments to namedtuple.
        'fmt' is a valid struct.struct format string.
        The number of fields must match the number of items returned by
        the 'fmt'
        """
        self.struct = Struct(fmt)
        self.dtype = namedtuple(name, field_names)
        test = self.struct.unpack(b'\x00' * self.struct.size)
        if len(test) != len(self.dtype._fields):
            raise AssertionError(" Wrong number of fields in NamedStruct "
                                 f" {name}. Expected {len(test)}")

    def unpack(self, buff, partial=False):
        if partial:
            return self.dtype._make(self.struct.unpack(buff[:self.size]))
        return self.dtype._make(self.struct.unpack(buff))

    def constant(self, *args):
        """Return namedtuple directly from args, instead of from unpacking
        a buffer. Could be better named.
        """
        return self.dtype(*args)

    @property
    def size(self):
        return self.struct.size

    def read(self, binf):
        return self.unpack(binf.read(self.size))


# NamedStructs for parsing BAF files:

# File vectors
VecHdr = NamedStruct("VecHdr", "<I4sI", "temp delim blksz")
VecFooter = NamedStruct("VecFooter", "<12s", "na")

# Index file:
# In the R Baf reader, they read offsets locations as signed int, so
# they have to do some weird rescaling.
# But really, the offset is just a 32-bit position and a page indicating
# the 2**32 bit chunk.
# We can just represent it as a 64-bit position
IdxOffset = NamedStruct("IdxOffset", "<bb6xI4xQ", "idA idB flag offset")


# Calibration
CalibHdr = NamedStruct("CalibHdr", "<I", "hdrsz")
CalibBlk = NamedStruct("CalibBlk", "<I", "blksz")
CalibMode = NamedStruct("CalibMode", "<8xI", "mode")
CalibAlpha = NamedStruct("CalibAlpha", "<d", "alpha")  # Only if mode==5
CalibRow = NamedStruct("CalibRow", "<dd4xi4xdd",
                       "beta freq_hi sz freq_lo freq_wid")
Calibration = namedtuple("Calibration", ["mode",
                                         "freq_lo",
                                         "freq_hi",
                                         "freq_wid",
                                         "alpha",
                                         "beta",
                                         "size"])

# Spectrum
SpecHdr = NamedStruct("SpecHdr", "<I4xI", "blksz hdrsz")

# Peaks
PeakHdr = NamedStruct("PeakHdr", "<I4xI", "blksz hdrsz")
PeakInfo = NamedStruct("PeakInfo", "<II", "num_peaks offset")


def read_vectors(fname) -> list[bytes]:
    with open(fname, 'rb') as f:
        result = []
        while 1:
            hdr = VecHdr.read(f)
            if hdr.blksz == 0:
                break
            vec_dat = f.read(hdr.blksz - (VecHdr.size + VecFooter.size))
            VecFooter.read(f)
            result.append(vec_dat)
        return result


def read_index_file(fname: str) -> dict[str, int]:
    vecs = read_vectors(fname)
    offsets = {}
    for vec in vecs:
        if vec[1] == 0:
            continue
        off_dat = IdxOffset.unpack(vec)
        off_type = OFFSETS.get((off_dat.idA, off_dat.idB))
        if off_dat is not None:
            offsets[off_type] = off_dat.offset

    return offsets


def glob2(pattern: str, root_dir: str) -> list[str]:
    # Replacement for python 3.10 glob in python 3.8
    pattern = os.path.join(root_dir, pattern)
    results = glob(pattern)
    return [os.path.basename(f) for f in results]


def read_extra_file(fname: str) -> Calibration:
    vecs = read_vectors(fname)
    for vec in vecs:
        if not (vec[0] == 4 and vec[1] == 1):
            # Skip all but calibration data
            continue
        offset = 4
        hdr = CalibHdr.unpack(vec[offset:], partial=True)
        offset += hdr.hdrsz
        blk = CalibBlk.unpack(vec[offset:], partial=True)
        offset += blk.blksz - hdr.hdrsz
        mode = CalibMode.unpack(vec[offset:], partial=True)
        offset += CalibMode.size
        if mode.mode == 5:
            alpha = CalibAlpha.unpack(vec[offset:], partial=True)
        else:
            alpha = CalibAlpha.constant(0.0)
        # In either case, we jump ahead:
        offset += CalibAlpha.size
        row = CalibRow.unpack(vec[offset:], partial=True)

        return Calibration(mode=mode.mode,
                           freq_lo=row.freq_lo,
                           freq_hi=row.freq_hi,
                           freq_wid=row.freq_wid,
                           alpha=alpha.alpha,
                           beta=-row.beta if mode.mode == 4 else row.beta,
                           size=row.sz)
    raise ValueError("Calibration not found in .baf_xtr file")


class BAFReader:
    def __init__(self, dotd_dir: str):
        """Initialize a reader from a Bruker '.d' directory.
        The directory must contain a '.baf', a '.baf_idx', and a '.baf_xtr'
        file in order to be parsed by this reader.
        """
        self.dirname: str = dotd_dir
        dat_file = glob2("*.baf", root_dir=self.dirname)
        if len(dat_file) == 0:
            raise FileNotFoundError("Data (.baf) file not found in directory.")
        self.dat_file = os.path.join(dotd_dir, dat_file[0])
        idx_file = self.dat_file + "_idx"
        xtr_file = self.dat_file + "_xtr"
        
        if not os.path.exists(idx_file):
            raise FileNotFoundError("Index (.baf_idx) file not found in "
                                    "directory.")
        if not os.path.exists(xtr_file):
            raise FileNotFoundError("Calibration (.baf_xtr) file not found in "
                                    "directory.")
        self.offsets = read_index_file(idx_file)
        self.calibration = read_extra_file(xtr_file)
        self._read_spectrum_hdr()
        self._read_peak_hdr()

    def mass2index(self, mass: float) -> np.ndarray:
        c = self.calibration
        # Derived from solving index2mass for mass.
        # More accurate than the 'rtms' solution
        k = ((c.freq_hi / (mass - c.alpha/c.freq_hi)) + c.beta)/c.freq_wid
        return c.size - c.size * k

    def index2mass(self, idx: int | np.ndarray) -> np.ndarray:
        c = self.calibration
        scaled = c.freq_wid * (c.size - idx) / c.size - c.beta
        return c.freq_hi / scaled + c.alpha / c.freq_hi

    def _read_spectrum_hdr(self) -> None:
        with open(self.dat_file, 'rb') as f:
            f.seek(self.offsets["spectrum"])
            hdr = SpecHdr.read(f)
            self.spec_length = (hdr.blksz - hdr.hdrsz) // 4
            self.spec_start = self.offsets["spectrum"] + hdr.hdrsz

    def _read_peak_hdr(self) -> None:
        with open(self.dat_file, 'rb') as f:
            f.seek(self.offsets["maxima"])
            hdr = PeakHdr.read(f)
            f.seek(hdr.hdrsz - PeakHdr.size, os.SEEK_CUR)
            info = PeakInfo.read(f)
            
            self.peak_start = self.offsets["maxima"] + hdr.hdrsz + info.offset
            self.num_peaks = info.num_peaks

    def read_spectrum(self,
                      from_mass: float | None = None,
                      to_mass: float | None = None) -> pd.DataFrame:
        """
        Read spectra and return Pandas dataframe of mass and intensity.
        By default, reads the entire spectrum, but a mass range can be
        specified.
        """
        if from_mass is not None:
            start_idx = int(self.mass2index(from_mass))
            start_idx = max(start_idx, 0)
        else:
            start_idx = 0
        if to_mass is not None:
            end_idx = int(self.mass2index(to_mass))
            end_idx = min(end_idx, self.spec_length)
        else:
            end_idx = self.spec_length
            
        offset = self.spec_start + start_idx * 4
        count = end_idx - start_idx
            
        with open(self.dat_file, 'rb') as f:
            intensity = np.fromfile(f,
                                    offset=offset,
                                    count=count,
                                    dtype='<f'
                                    )
            mass = self.index2mass(np.arange(start_idx, end_idx))
            return pd.DataFrame({"mz": mass, "intensity": intensity})

    def read_peaks(self) -> pd.DataFrame:
        """Read all maxima from the spectrum.
        Returns a Pandas DataFrame of the m/z and intensity of all recorded
        peaks. These are the peaks recorded by the instrument for calibration,
        and only include the top N (2500?) peaks.
        """
        with open(self.dat_file, 'rb') as f:
            indices = np.fromfile(f,
                                  offset=self.peak_start,
                                  count=self.num_peaks,
                                  dtype='<d'
                                  )
            intensity = np.fromfile(f,
                                    count=self.num_peaks,
                                    dtype='<d'
                                    )

            mass = self.index2mass(indices)
            return pd.DataFrame({"mz": mass, "intensity": intensity})


class BAFCache:
    """Convenience class for handling a directory of '.d' files.

    BAFCache scans a given directory for all Bruker '.d' datasets, and allows
    you to access them by their filename sans extension.
    E.g: for files named '/path/to/dot_ds/samp_XXX.d':
    
    >>> bcache = BAFCache("/path/to/dot_ds")
    >>> rdr = bcache['samp_001']

    Readers are cached upon first use (i.e. the file metadata is read and
    stored)
    """
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        baf_files = glob2("*.d", root_dir=root_dir)
        self._sample_lookup = {os.path.splitext(f)[0]:
                               os.path.join(root_dir, f)
                               for f in baf_files}
        self._samples = list(self._sample_lookup.keys())
        self.cache = {}

    def __getitem__(self, key: str | int) -> BAFReader:
        if isinstance(key, int):
            key = self._samples[key]
        if key in self.cache:
            return self.cache[key]
        rdr = BAFReader(self._sample_lookup[key])
        self.cache[key] = rdr
        return rdr

    def __contains__(self, key: str) -> bool:
        return key in self._sample_lookup

    def __iter__(self) -> Generator[BAFReader, None, None]:
        for samp in self._samples:
            yield self[samp]

    @property
    def samples(self) -> list[str]:
        return self._samples[:]

    @property
    def files(self) -> dict[str, str]:
        return self._sample_lookup.copy()
