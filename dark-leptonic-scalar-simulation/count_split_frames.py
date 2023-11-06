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
    
    parser.add_argument("-i", dest="inputfile",
                        default="", type=str, required=True,
                        help="File with LLPs")
     
# Get Params
parser = argparse.ArgumentParser(description="count LLPs")
add_args(parser)
params = vars(parser.parse_args())  # dict()

class LLPEventCounter(icetray.I3Module):
    def __init__(self, ctx):
        icetray.I3Module.__init__(self, ctx)
        self.n_physics = 0
        self.n_dict = {}
    def DAQ(self, frame):
        if self.n_physics != 0:
            val = self.n_physics
            if val not in self.n_dict:
                self.n_dict[val] = 1
            else:
                self.n_dict[val] += 1
        self.n_physics = 0
    def Physics(self, frame):
        self.n_physics += 1
    def Finish(self):
        print("Number of physics frames per DAQ frame")
        print(self.n_dict)

tray = I3Tray()

filelist = [s.strip() for s in params["inputfile"].split(",")] 

tray.Add("I3Reader", FilenameList=filelist)
tray.Add(LLPEventCounter)

tray.Execute()

