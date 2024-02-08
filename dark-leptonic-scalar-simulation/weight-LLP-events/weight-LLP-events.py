import icecube
from icecube import icetray, dataio, dataclasses
from icecube import MuonGun
from icecube.icetray import I3Tray
import argparse
import glob
import os

###############################################
def FixMMCTrackListLLP(frame):
    if "MMCTrackListLLP" in frame:
        tracklist_LLP         = frame["MMCTrackListLLP"]
        initial_muon          = max(tracklist_LLP, key = lambda track : track.Ei)
        frame["MMCTrackList"] = icecube.simclasses.I3MMCTrackList([initial_muon])
        return True
    else:
        print("no MMCTrackListLLP in frame!")
        exit()
        return False

def ScaleMuonGunWeight(frame, scale):
    if "MuonWeightUnscaled" in frame:
        frame["MuonWeight"] = dataclasses.I3Double(frame["MuonWeightUnscaled"].value*scale)
    

def harvest_generators(infiles):
    """
    Harvest serialized generator configurations from a set of I3 files.
    """
    from icecube.icetray.i3logging import log_info as log
    generator = None
    for fname in infiles:
        f = dataio.I3File(fname)
        fr = f.pop_frame(icetray.I3Frame.Stream('S'))
        f.close()
        if fr is not None:
            for k in fr.keys():
                v = fr[k]
                if isinstance(v, MuonGun.GenerationProbability):
                    log('%s: found "%s" (%s)' % (fname, k, type(v).__name__), unit="MuonGun")
                    if generator is None:
                        generator = v
                    else:
                        generator += v
    return generator

###############################################


def add_args(parser):
    """
    Args:
        parser (argparse.ArgumentParser): the command-line parser
    """
    
    parser.add_argument("-o", "--output", action="store",
        type=str, default="", dest="outfile", required=True,
        help="Output i3 file")

    parser.add_argument("--model", dest="model",
                        default="Hoerandel5_atmod12_SIBYLL",
                        type=str, required=False,
                        help="primary cosmic-ray flux parametrization")
    
    parser.add_argument("-i", "--inputfile", action="store",
        type=str, default="", dest="infile",
        help="Input i3 file(s)  (use comma separated list for multiple files)")
    
    parser.add_argument("-ws", "--weight-scale", dest="weight-scale",
                        default=1e15, type=float, required=False,
                        help="should be 1e15. number passed to total events in SimulateLLP")
    
    parser.add_argument("-gm", "--generated-muons", dest="generated-muons",
                        type=float, required=True,
                        help="power law spectral index")
    

# Get params from parser
parser = argparse.ArgumentParser(description="Weight LLP file")
add_args(parser)
params = vars(parser.parse_args())  # dict()
params['infile'] = params['infile'].split(',')


############### start weighting ###################
model = MuonGun.load_model(params["model"])

if isinstance(params['infile'],list):
    infiles = params['infile']
else:
    infiles = [params['infile']]
    
generator = harvest_generators(infiles)

# total DAQ frames
DAQ_count = {"count":0} # dont ask me why, but a local float didn't work while a dictionary did
def countDAQ(frame):
    DAQ_count["count"] += 1
    return True
tray = I3Tray()
tray.Add("I3Reader", filenamelist=infiles)
tray.Add(countDAQ, streams=[icetray.I3Frame.DAQ])
tray.Execute()
print("total DAQ frames:",DAQ_count["count"])

print("starting tray")
#icetray.set_log_level(icetray.I3LogLevel.LOG_INFO)
tray = I3Tray()
tray.Add("I3Reader", filenamelist=infiles)
# fix the MMCTrackList of LLP files containing the LLP decay muon
def checkIfDAQ(frame):
    if frame.Stop == icetray.I3Frame.DAQ:
        return True
    else:
        print(frame.Stop)
        return False
#tray.Add(checkIfDAQ) # since passing streams didn't work for copy and delete
tray.Add(lambda frame: frame.Stop == icetray.I3Frame.DAQ) # since passing streams didn't work for copy and delete
tray.AddModule("Copy", "copy", Keys = ["MMCTrackList", "MMCTrackListLLP"], If = lambda f: f.Stop == icetray.I3Frame.DAQ)
tray.AddModule("Delete", "delete", Keys = ["MMCTrackList"], If = lambda f: f.Stop == icetray.I3Frame.DAQ)
tray.Add(FixMMCTrackListLLP, streams=[icetray.I3Frame.DAQ])
# weight and scale since muongun thought we simulated 1e15 muons (see SimulateLLP.py script)
tray.AddModule('I3MuonGun::WeightCalculatorModule', 'MuonWeightUnscaled', Model=model, Generator=generator)
tray.Add(ScaleMuonGunWeight, scale = params["weight-scale"]/(params["generated-muons"] - DAQ_count["count"]), streams=[icetray.I3Frame.DAQ])
tray.Add("I3Writer", filename=params["outfile"])
tray.Execute()



