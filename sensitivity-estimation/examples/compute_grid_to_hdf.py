""" Compute detectable LLP probabilites for a model grid on a CORSIKA muon spectrum.
Reads in a number of CORSIKA-in-ice files, computes event-by-event LLP probability
for each mass/eps point in the grid, and saves to .hdf5 file.

Uses CORSIKA files from:
/data/sim/IceCube/2020/generated/CORSIKA-in-ice/20904/0198000-0198999/detector/
"""
import sys
sys.path.append("..")

import glob
import argparse
import numpy as np
from itertools import product

import icecube
from icecube import icetray, dataio, hdfwriter
from icecube.icetray import I3Tray

from I3LLPProbabilityCalculator import *
from llpestimation import LLPModel, LLPEstimator, LLPMedium, LLPProductionCrossSection
from estimation_utilities import *

# Get params from parser
parser = argparse.ArgumentParser(description="create hdf files from corsika")
parser.add_argument("-o", "--outputfile", action="store",
        type=str, default="corsika_to_grid", dest="outputfile",
        help="Name of final .hdf5 file. Don't include extension")
parser.add_argument("-n", "--nfiles", action="store",
        type=int, dest="nfiles", required = True,
        help="Number of CORSIKA files used to create hdf5 file.")

params = vars(parser.parse_args())  # dict()
outputfile = params["outputfile"] + "_" + str(params["nfiles"]) + "_files.hdf5"
print(outputfile)
# infiles
filelist = list(glob.glob("/data/sim/IceCube/2020/generated/CORSIKA-in-ice/20904/0198000-0198999/detector/IC86.2020_corsika.020904.198*.i3.zst"),)
filelist = filelist[0:params["nfiles"]] # how many files to use?
print("Number of CORSIKA files used:", len(filelist))
gcdfile = "/data/user/axelpo/LLP-at-IceCube/dark-leptonic-scalar-simulation/resources/GeoCalibDetectorStatus_2021.Run135903.T00S1.Pass2_V1b_Snow211115.i3.gz"

# create LLP models
# @TODO: read in grid points from config file
# @TODO: fix so large number of grid points doesn't crash the program
# masses       = [0.107, 0.108, 0.109, 0.110, 0.111, 0.112, 0.113,
#                 0.114, 0.115, 0.116, 0.117, 0.118, 0.119, 0.120,
#                 0.121, 0.122, 0.123, 0.124, 0.125, 0.126, 0.127,
#                 0.128, 0.129, 0.130, 0.131, 0.132, 0.133, 0.134,
#                 0.135, 0.136, 0.137, 0.138, 0.139, 0.140, 0.142,
#                 0.145, 0.15]
# epsilons     = np.logspace(-2.5, -6, 50)

masses       = [0.107, 0.108, 0.109, 0.110, 0.111, 0.112, 0.113,
                0.114, 0.115, 0.116, 0.117, 0.118, 0.119, 0.120,
                0.121, 0.122, 0.123, 0.124, 0.125]
epsilons     = np.logspace(-2.5, -4, 25)

# create grid
combinations = list(product(masses, epsilons))
#masses       = [item[0] for item in combinations] # flatten
#epsilons     = [item[1] for item in combinations] # flatten

# only keep points that might be interesting
masses = []
epsilons = []
for item in combinations:
    if item[0] > 0.120 and item[1] > 8e-5:
        continue
    masses.append(item[0])
    epsilons.append(item[1])

names        = ["DLS" for _ in masses]
# utility funcs to create LLPModel objects
table_paths  = generate_DLS_WW_oxygen_paths(masses, folder = "../cross_section_tables/")
DLS_models   = generate_DLSModels(masses, epsilons, names, table_paths)
print(masses, epsilons)

# create LLPEstimator
min_gap = 50.0 # minimum detectable LLP gap
DLS_estimator = LLPEstimator(DLS_models, min_gap)

# detector parameters
n_steps = 80

# which frame objects to save to hdf5 file
keys = ["LLPProbabilities",
        "MMCTrackList",
        "CorsikaWeightMap",
        "PolyplopiaPrimary",
       ]

########## Run I3Tray ##########
tray = I3Tray()

tray.Add("I3Reader", FileNameList=filelist)
tray.Add(I3LLPProbabilityCalculator,
         GCDFile = gcdfile,
         llp_estimator = DLS_estimator,
         n_steps = n_steps
)
tray.Add(
    hdfwriter.I3SimHDFWriter,
    keys=keys,
    output=outputfile,
)
tray.Execute()