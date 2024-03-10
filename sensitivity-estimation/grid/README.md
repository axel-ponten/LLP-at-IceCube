This directory contains script to compute expected detectable LLP rates in IceCube from the atmospheric muon flux (CORSIKA-in-ice simulation) for a grid of LLPModels. This is used to get sensitivity plots for a particular model over the mass/epsilon parameter space of a given model. If the search is background free, then all grid points with expected signal > 2.3 could be excluded with 90% C.L. if no signal is found in the data.

## Compute llp rates for a grid
*compute_grid_to_hdf.py* computes the expected llp rates using the llpestimation package for an atmospheri muon spectrum given by the MC dataset **CORSIKA-in-ice 20904**. You just need to pass how many CORSIKA files to use (-n) and the name of the resulting .hdf5 file (-o. Note that you don't need the extension, that's done automatically). Will save the keys `["LLPProbabilities", "MMCTrackList", "CorsikaWeightMap", "PolyplopiaPrimary"]`.

```
python compute_grid_to_hdf.py -n 10 -o files/llp_rates_grid
```

@TODO: Currently the grid is hardcoded into the script but will be updated to use a .yml config file instead.

## Weight the grid

Using **simweights** (https://github.com/icecube/simweights) one can weight the event by event llp probabilities. Need number of CORSIKA files used in creation for proper weighting. Create a .csv with rows (mass, eps, llp_rate, LLPModel_unique_id) using

```
python grid_to_csv.py -i files/llp_rates_grid_10_files.hdf5 -n 10 -o llp_rates_grid.csv
```

## Plot the grid

Plot the weighted grid over (mass, eps) using

```
python plot_grid.py -i files/llp_rates_grid_10_files.hdf5 -n 10 -o llp_rates_grid.png -y 10 -m 2.3
```

Or by loading a .csv file made by `grid_to_csv.py` to plot using

```
python plot_grid_from_csv.py -i files/grid_rates.csv -o llp_rates_grid.png -y 10 -m 2.3
```

Multiplies the rates by a number of years *y* (10 years for example) and plots all grid points that have more than *m* expected signals in *y* years.

## Utility script

`weightgrid.py` contains common functions to read .hdf5 files and weight them, used by the other scripts.