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
parser.add_argument("-y", "--years", action="store",
        type=float, default = 1.0, dest="years", required = False,
        help="How many years of livetime to multiply llp rate by? Default 1.")
parser.add_argument("-m", "--min-events", action="store",
        type=float, default = None, dest="min-events", required = False,
        help="Minimum # of detectable events to be plotted.")

params = vars(parser.parse_args())  # dict()

####### OPEN FILE #######
df = pd.read_csv(params["inputfile"])
# remove zero signal points
df = df[df["llp_rate"] > 0]

####### GET GRID #######
masses   = df["mass"]
epsilons = df["epsilon"]
livetime = params["years"]*3.1536e7 # convert years to seconds
signals  = [livetime*r for r in df["llp_rate"]] # expected n of signals

# remove signals < min_events
if params["min-events"] is not None:
    masses, epsilons, signals = wg.clean_min_events(masses, epsilons, signals, params["min-events"])

####### PLOT GRID #######
fig, ax = plt.subplots()

sc = plt.scatter(masses, epsilons, c=np.log10(signals), cmap = 'viridis')
cbar = plt.colorbar(sc)
cbar.set_label("Log10 expected signal")
plt.yscale("log")

modelname = wg.get_name_from_id(df["LLPModel_unique_id"][0]) # for plot title
plt.title("Expected detectable " + modelname + " events in {:.1f} year(s)".format(params["years"]))
plt.ylabel(r'$\epsilon$', fontsize=13)
plt.xlabel(r'$m_\varphi$' + " [GeV]", fontsize=13)
# plt.xlim([0.105, 0.150])
# plt.ylim([1e-6, 1e-2])

# save grid plot
plt.savefig(params["outfile"], bbox_inches="tight")