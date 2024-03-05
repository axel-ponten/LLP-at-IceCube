import sys
sys.path.append("..")

import pandas as pd
import argparse

from llpestimation import LLPModel, LLPEstimator, LLPMedium, LLPProductionCrossSection
from estimation_utilities import *

# Get params from parser
parser = argparse.ArgumentParser(description="create plots from hdf files")
parser.add_argument("-i", "--inputfile", action="store",
        type=str, default="", dest="inputfile",
        help="Input .hdf5 file with LLPProbabilities frame object.")

params = vars(parser.parse_args())  # dict()

# open file
df = pd.read_hdf(params["inputfile"], ["LLPProbabilities", "MMCTrackList"])

print(df)