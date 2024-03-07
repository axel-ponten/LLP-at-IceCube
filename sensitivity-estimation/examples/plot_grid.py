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

from llpestimation import LLPModel, LLPEstimator, LLPMedium, LLPProductionCrossSection
from estimation_utilities import *
import simweights

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

# load the hdf5 file that we just created using pandas
hdffile = pd.HDFStore(params["inputfile"], "r")
nfiles = params["nfiles"]

# instantiate the weighter object by passing the pandas file to it
weighter = simweights.CorsikaWeighter(hdffile, nfiles = nfiles)

# create an object to represent our cosmic-ray primary flux model
flux = simweights.GaisserH4a()

# get the weights by passing the flux to the weighter
weights = weighter.get_weights(flux)

# print some info about the weighting object
print(weighter.tostring(flux))

####### READ IN GRID #######
llp_prob = hdffile.select("LLPProbabilities")
model_ids = llp_prob.keys()[5:] # SKIP first five, like EventID and RunID

def get_mass_eps_from_id(llp_unique_id: str):
    model = LLPModel.from_unique_id(llp_unique_id)
    return model.mass, model.eps

livetime  = params["years"]*3.1536e7 # convert to seconds
masses   = []
epsilons = []
signals = []
for model_id in model_ids:
    mass, eps = get_mass_eps_from_id(model_id)
    weighted_rate = llp_prob[model_id].dot(weights)
    signal = livetime*weighted_rate
    # check if we save the grid point for plotting
    if (params["min-events"] is None) or (signal >= params["min-events"]):
        masses.append(mass)
        epsilons.append(eps)
        signals.append(signal)
    if params["verbose"]:
        print(model_id)
        print("Raw llp probs.", llp_prob[model_id])
        print("Sum of llp probs", llp_prob[model_id].sum())
        print("Weighted llp rate", weighted_rate)


####### PLOT GRID #######
fig, ax = plt.subplots()

sc = plt.scatter(masses, epsilons, c=np.log10(signals), cmap = 'bwr')
cbar = plt.colorbar(sc)
cbar.set_label("Log10 expected signal")
plt.yscale("log")

plt.title("Expected detectable LLPs in {:.1f} year(s)".format(params["years"]))
plt.ylabel(r'$\epsilon$', fontsize=13)
plt.xlabel(r'$m_\varphi$' + " [GeV]", fontsize=13)

# save grid plot
plt.savefig(params["outfile"], bbox_inches="tight")