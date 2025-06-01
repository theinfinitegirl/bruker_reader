# bruker-reader

This is a simple python package for reading FTICR spectra from Bruker BAF files.

It's heavily based off of the R `rtms` package (https://cran.r-project.org/package=rtms), 
who did the heavy lifting of reverse engineering the file format.

## Usage

Usage is simple. Create a reader by pointing it to the `.d` directory:

```
	>>> from bruker_reader import BAFReader
	>>> rdr = BAFReader("/path/to/data/sample.d")
	>>> spec = rdr.read_spectrum()
	... # returns pandas DataFrame with 'mz' and 'intensity' columns
```

You can read partial spectra as well, which can dramatically speed up access:

```
	>>> rdr.read_spectrum(from_mass=451.0, to_mass=509)
```

BAF files also store a list of the N largest peaks (where N is configurable
when setting up a run). Note, this is not necessarily the same as the total
peaks collected after calibrating; but it may still be useful.

```
	>>> rdr.read_peaks()
	# Returns a pandas DataFrame of `mz` and `intensity` values.
```

To ease the process of reading multiple Bruker `.d` files stored in a data dir,
the `BAFCache` class can be used. It creates an easy lookup for spectra files,
as well as caches the metadata for the spectrum upon first use:

```
	>>> from bruker_reader import BAFCache
	# Assuming `/path/to/data` contains `samp1.d`, `samp2.d` ...
	>>> bcache = BAFCache('/path/to/data')

	# Access samples by key
	>>> samp1 = bcache['samp1'].read_spectrum()

	# You can also access by numeric index:
	>>> bcache[12]
	
	# BAFCache has typical iterable behavior:
	>>> if 'samp11' in bcache: ...
	>>> for rdr in bcache: ...
	
	# Getting directory info
	>>> bcache.samples  # Returns a list of sample ids
	>>> bcache.files    # Returns a dict of sample -> filepath
```

