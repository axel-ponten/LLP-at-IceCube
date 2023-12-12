Scripts for simulating and analysing Dark Leptonic Scalars at IceCube.


### Generate MC data

1. The main script for generating LLP MC is SimulateLLP.py. This uses a custom muon propagation segment found in PropagateMuonsLLP.py, which uses a custom I3PropagatorService found in I3PropagatorServicePROPOSAL_LLP.py.

2. millipede_LLP.py runs millipede on the simulated LLP data.
3. The folder condor contains submit scripts and config for the simulation
4. The folder resources contains some PROPOSAL json config scripts with LLP.

### Filtering
Copies from filterscripts in main i3 repo.

1. Simulate online filter (level 1) with SimulateFiltering.py. Added LLPInfo to saved frames in SimulateFilter.py (the default script from i3 repo does not have it, of course).

2. Simulate level 2 using process.py

3. In practice trigger to L2 is done with the script /data/user/axelpo/LLP-data/runL1L2.sh since condor spits out many files and we want to run it on all of them simultaneously.

### filtering-analysis
1. Some scripts to check characteristics of trigger to L2.
2. Includes a sanity check of apparent randomness in L2 filtering. This was found to be due to minbias filter.

### spectrum-at-detector-boundary
1. spectrum-at-detector-boundary contains some scripts characterising spectrum of atmospheric muons at the detector boundary, both for L2 and trigger. This is unrelated to LLP simulation, only used to obtain single muon spectrum at the detector.
