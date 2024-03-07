import sys
sys.path.append("..")

import glob
import argparse

import icecube
from icecube import icetray, dataio, hdfwriter
from icecube.icetray import I3Tray

from I3LLPProbabilityCalculator import *
from llpestimation import LLPModel, LLPEstimator, LLPMedium, LLPProductionCrossSection
from estimation_utilities import *

# Get params from parser
parser = argparse.ArgumentParser(description="create hdf files from corsika")
parser.add_argument("-o", "--outputfile", action="store",
        type=str, default="", dest="outputfile",
        help="Input .i3 file with LLPProbabilities frame object.")

params = vars(parser.parse_args())  # dict()

# infiles
filelist = list(glob.glob("/data/sim/IceCube/2020/generated/CORSIKA-in-ice/20904/0198000-0198999/detector/IC86.2020_corsika.020904.198*.i3.zst"),)
n_files = 1 # how many files to use?
filelist = filelist[0:n_files]
print("Number of CORSIKA files used:", len(filelist))
gcdfile = "/data/user/axelpo/LLP-at-IceCube/dark-leptonic-scalar-simulation/resources/GeoCalibDetectorStatus_2021.Run135903.T00S1.Pass2_V1b_Snow211115.i3.gz"

# create LLP models
masses   = [0.115, 0.115, 0.13, 0.13]
epsilons = [5e-6, 1e-5, 5e-6, 1e-5]
names    = ["DLS" for _ in masses]
table_paths = generate_DLS_WW_oxygen_paths(masses, folder = "../cross_section_tables/")
DLS_models = generate_DLSModels(masses, epsilons, names, table_paths)

# create LLPEstimator
min_gap = 50.0 # minimum detectable LLP gap
DLS_estimator = LLPEstimator(DLS_models, min_gap)

# detector parameters
n_steps = 50

# icetray paramters
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
#tray.Add(print_LLPInfo, Streams=[icecube.icetray.I3Frame.DAQ])
tray.Add(
    hdfwriter.I3SimHDFWriter,
    keys=keys,
    output=params["outputfile"],
)
tray.Execute()