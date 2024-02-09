import icecube
from icecube import icetray, dataio, dataclasses, hdfwriter
from icecube import MuonGun
from icecube.icetray import I3Tray
import argparse
import glob
import os
import numpy as np

def create_hdf(infiles, outfile, keys):
    tray = I3Tray()
    tray.Add("I3Reader", FileNameList=infiles)
    tray.Add(
        hdfwriter.I3SimHDFWriter,
        keys=keys,
        output=outfile,
    )
    tray.Execute()

def add_args(parser):
    """
    Args:
        parser (argparse.ArgumentParser): the command-line parser
    """
    
    parser.add_argument("-i", "--inputfolder", action="store",
        type=str, default="", dest="inputfolder", required = True,
        help="Input LLP folder")

    parser.add_argument("-k", "--keys", action="store",
        type=str, default="", dest="keys",
        help="Keys to save (use comma separated list for multiple files)")

# Get params from parser
parser = argparse.ArgumentParser(description="create hdf from i3 files")
add_args(parser)
params = vars(parser.parse_args())  # dict()
params['keys'] = params['keys'].split(',')

# create outfile folder and names
clusterID = params["inputfolder"].split('.')[-1][:-1]
outfolder = "clusterID_"+clusterID
os.makedirs(outfolder, exist_ok = True)

# trigger files
trigger_outfile = outfolder + "/trigger.hdf5"
trigger_infiles = list(glob.glob(params["inputfolder"]+"LLPSim*/*.gz"))
create_hdf(trigger_infiles, trigger_outfile, params["keys"])

# L2 file
L2_outfile = outfolder + "/L2.hdf5"
L2_infile = params["inputfolder"] + "L2.i3.gz"
create_hdf([L2_infile], L2_outfile, params["keys"])