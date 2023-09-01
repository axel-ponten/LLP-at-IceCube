
import argparse
import json
from icecube.simprod.util import ReadI3Summary, WriteI3Summary
from icecube.simprod.util import CombineHits, DrivingTime, SetGPUEnvironmentVariables
import os.path
from icecube.simprod.util import simprodtray, arguments
from icecube.simprod.util.simprodtray import RunI3Tray
import icecube.icetray
import icecube.dataclasses
from icecube.dataclasses import *
import icecube.dataio
import icecube.phys_services
from icecube.icetray import I3Tray, I3Units
from icecube.simprod.util import BasicCounter
from icecube.simprod.segments import GenerateCosmicRayMuons, PropagateMuons, GenerateNaturalRateMuons
from icecube import clsim
from icecube import polyplopia
from icecube import phys_services
from icecube import PROPOSAL
from icecube import icetray
import icecube
from PropagateMuonsLLP import PropagateMuonsLLP

def add_args(parser):
    """
    Args:
        parser (argparse.ArgumentParser): the command-line parser
    """
    
    arguments.add_seed(parser)
    
    parser.add_argument("--nevents", dest="nevents",
                        type=int, required=True,
                        help="Number of events MuonGun should simulate.")
    
    parser.add_argument("--gcdfile", dest="gcdfile",
                        default="resources/GeoCalibDetectorStatus_2021.Run135903.T00S1.Pass2_V1b_Snow211115.i3.gz", type=str, required=False,
                        help="GCD file.")
    
    parser.add_argument("--outputfile", dest="outputfile",
                        default="", type=str, required=False,
                        help="Output file. Default creates name from PROPOSAL config and nevents.")
    
    parser.add_argument("--PROPOSAL-config-SM", dest="config_SM",
                        default="resources/config_SM.json", type=str, required=False,
                        help="PROPOSAL config file without LLP.")
    
    parser.add_argument("--PROPOSAL-config-LLP", dest="config_LLP",
                        default="resources/config_DLS.json", type=str, required=False,
                        help="PROPOSAL config file with LLP.")
    
    parser.add_argument("--OnlyOneLLP", dest="only_one_LLP",
                        default=False, action="store_true", required=False,
                        help="Switch to Standard Model propagator after LLP production.")

    parser.add_argument("--SaveAllEvents", dest="OnlySaveLLPEvents",
                        default=True, action="store_false", required=False,
                        help="Save all event or only keep frames with LLP production.")
    
    parser.add_argument("--EitherDecayOrProd", dest="both_prod_decay_inside",
                        default=True, action="store_false", required=False,
                        help="Only require LLPs to have EITHER decay or production inside detector, instead of both.")
    
    parser.add_argument("--min_LLP_length", dest="min_LLP_length",
                        default=0, type=float, required=False,
                        help="Manually set minimum length for good LLP events. Default is 0 m.")

def custom_filename(nevents, PROPOSAL_config_LLP):
    """ default filename from PROPOSAL config """
    file = open(PROPOSAL_config_LLP)
    config_json = json.load(file)
    
    llp_multiplier = config_json["global"]["llp_multiplier"]
    mass = config_json["global"]["llp_mass"]
    eps = config_json["global"]["llp_epsilon"]
    simulation_model = config_json["global"]["llp"]
    
    file.close()
    
    return "LLPGun."+simulation_model+".mass-"+str(mass)+".eps-" + str(eps)+".bias-"+str(llp_multiplier)+".nevents-"+str(nevents)+".i3"
    
# Get Params
parser = argparse.ArgumentParser(description="LLPGun script")
add_args(parser)
params = vars(parser.parse_args())  # dict()

if params["outputfile"] == "":
    # default filename
    params["outputfile"] = custom_filename(params["nevents"], params["config_LLP"])

# set up tray
tray = I3Tray()
icetray.set_log_level(icetray.I3LogLevel.LOG_FATAL) # supress warnings from PROPOSAL integration

rand = phys_services.I3GSLRandomService(seed=params['seed'])
tray.context['I3RandomService'] = rand

tray.AddModule("I3InfiniteSource", "TheSource",
               Prefix               = params["gcdfile"],
               Stream               = icecube.icetray.I3Frame.DAQ
              )

tray.AddSegment(GenerateNaturalRateMuons, "muongun",
                NumEvents           = 1e15, # real number of events is set in PropagateMuonsLLP
                GCDFile             = params["gcdfile"],
                mctree_name         = "I3MCTree_preMuonProp",
                flux_model          = "GaisserH4a_atmod12_SIBYLL"
               )

tray.AddSegment(PropagateMuonsLLP,
                "propagator",
                RandomService          = rand,
                InputMCTreeName        = "I3MCTree_preMuonProp",
                OutputMCTreeName       = "I3MCTree",
                PROPOSAL_config_SM     = params["config_SM"],
                PROPOSAL_config_LLP    = params["config_LLP"],
                OnlySaveLLPEvents      = params["OnlySaveLLPEvents"],
                only_one_LLP           = params["only_one_LLP"],
                nevents                = params["nevents"],
                gcdfile                = params["gcdfile"],
                both_prod_decay_inside = params["both_prod_decay_inside"],
                min_LLP_length         = params["min_LLP_length"],
               )

tray.Add("I3Writer", filename=params["outputfile"])

tray.Execute()

