Conversion scripts are based off Thorsten Gluesenkamp's code.

`converter.py` will convert a set of .i3 files to parquet files (in the form of awkward arrays) to be used in ML based reconstruction. Auxiliary files containing metainformation are also created to help with weighting, indexing and dataloading, etc.

The conversion creates a number of .pq files each containing equal number of events. If the total number of events in the .i3 files is not a multiple of number of events per .pq file, then the excess excluded from the conversion. This will of course affect the weighting scheme, which is why the auxiliary files are provided (to compute effective number of i3 files per pq file etc.)

`i3_methods.py` contain functions that extract the actual event and pulse information from the frame.

Pulses are extracted using the `EventFeatureFactory` from `icecube.ml_suite`. The encoding settings are found in `feature_configs.py`.

A minimal example script is provided called `example_convert.py`.