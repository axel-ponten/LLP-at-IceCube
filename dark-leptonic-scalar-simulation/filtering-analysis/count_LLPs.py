
import argparse
import icecube
import icecube.icetray
import icecube.dataclasses
from icecube.icetray import I3Tray
from icecube import dataio, icetray



def add_args(parser):
    """
    Args:
        parser (argparse.ArgumentParser): the command-line parser
    """
    
    parser.add_argument("--inputfile", dest="inputfile",
                        default="", type=str, required=True,
                        help="File with LLPs")
     
# Get Params
parser = argparse.ArgumentParser(description="count LLPs")
add_args(parser)
params = vars(parser.parse_args())  # dict()

LLP_count = {}

def count_LLP(frame):
    count = frame["LLPInfo"]["interactions"]
    if count in LLP_count:
        LLP_count[count] += 1
    else:
        LLP_count[count] = 1
    return True

tray = I3Tray()

tray.Add("I3Reader", filenamelist=[params["inputfile"]])
tray.Add(count_LLP, streams=[icetray.I3Frame.DAQ])

tray.Execute()

print("Histogram of LLP interactions for file", params["inputfile"])
print(LLP_count)
