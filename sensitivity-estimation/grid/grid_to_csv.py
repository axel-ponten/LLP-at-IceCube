""" Script to save an llpestimation grid to csv.
    Input file should be a .hdf5 file from CORSIKA simulation
    that has been run through the I3LLPProbabilityCalculator module.
    For example the output of compute_grid_to_hdf.py.
    
    Need to input how man CORSIKA files were used for weighting purposes.
    
    Outputs a csv with rows: mass, eps, llp_rate, model_id
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

params = vars(parser.parse_args())  # dict()

####### OPEN FILE #######
hdffile, weights = wg.weight_CORSIKA_hdf5(params["inputfile"], params["nfiles"])

####### WEIGHT GRID #######
masses, epsilons, llp_rates, model_ids = wg.weighted_grid_llp_rate(hdffile, weights)

####### SAVE TO CSV #######
df = pd.DataFrame({
    "mass" : masses,
    "epsilon": epsilons,
    "llp_rate": llp_rates,
    "LLPModel_unique_id": model_ids
})
df.to_csv(params["outfile"], index=False)
