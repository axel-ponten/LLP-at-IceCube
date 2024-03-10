""" Script to plot an llpestimation grid.
    Input file should be a .hdf5 file from CORSIKA simulation
    that has been run through the I3LLPProbabilityCalculator module.
    For example the output of compute_grid_to_hdf.py.
    
    Need to input how man CORSIKA files were used for weighting purposes.
    
    Outputs a plot of the grid.
"""
import sys
sys.path.append("..")

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np

import simweights
import weightgrid as wg # custom script in same directory

# Get params from parser
parser = argparse.ArgumentParser(description="create plots from hdf files")
parser.add_argument("-i", "--inputfile", action="store", required = True,
        type=str, dest="inputfile",
        help="Input .hdf5 file with LLPProbabilities frame object.")
parser.add_argument("-o", "--outfile", action="store",
        type=str, default="grid_plot.png", dest="outfile",
        help="Name of plot output.")
parser.add_argument("-n", "--nfiles", action="store",
        type=int, dest="nfiles", required = True,
        help="Number of CORSIKA files used to create hdf5 file.")
parser.add_argument("-y", "--years", action="store",
        type=float, default = 1.0, dest="years", required = False,
        help="How many years of livetime to multiply llp rate by? Default 1.")
parser.add_argument("-v", "--verbose", action="store_true",
        default = False, dest="verbose", required = False,
        help="Print info for each grid point.")
parser.add_argument("-m", "--min-events", action="store",
        type=float, default = None, dest="min-events", required = False,
        help="Minimum # of detectable events to be plotted.")

params = vars(parser.parse_args())  # dict()

####### OPEN FILE #######
hdffile, weights = wg.weight_CORSIKA_hdf5(params["inputfile"], params["nfiles"])

####### WEIGHT GRID #######
masses, epsilons, llp_rates, model_ids = wg.weighted_grid_llp_rate(hdffile, weights)
livetime = params["years"]*3.1536e7 # convert to seconds
signals = [livetime*r for r in llp_rates] # expected n of signals

# remove signals < min_events
if params["min-events"] is not None:
    masses, epsilons, signals = wg.clean_min_events(masses, epsilons, signals, params["min-events"])

# print info on each grid point
if params["verbose"]:
    wg.print_verbose(hdffile, weights)

####### PLOT GRID #######
fig, ax = plt.subplots()

sc = plt.scatter(masses, epsilons, c=np.log10(signals), cmap = 'viridis')
cbar = plt.colorbar(sc)
cbar.set_label("Log10 expected signal")
plt.yscale("log")

plt.title("Expected detectable LLPs in {:.1f} year(s)".format(params["years"]))
plt.ylabel(r'$\epsilon$', fontsize=13)
plt.xlabel(r'$m_\varphi$' + " [GeV]", fontsize=13)

# save grid plot
plt.savefig(params["outfile"], bbox_inches="tight")