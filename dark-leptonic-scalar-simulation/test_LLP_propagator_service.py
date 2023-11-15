import math
from icecube.simprod.util import ReadI3Summary, WriteI3Summary
from icecube.simprod.util import CombineHits, DrivingTime, SetGPUEnvironmentVariables
import os.path
from icecube.simprod.util import simprodtray, arguments
from icecube.simprod.util.simprodtray import RunI3Tray
import argparse
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

# set up tray
tray = I3Tray()

rand = phys_services.I3GSLRandomService(seed=0)

n_events = 10
filename = "Testing_I3PropagatorService_condor.i3"

icetray.set_log_level(icetray.I3LogLevel.LOG_FATAL) # supress warnings from PROPOSAL integration

tray.context['I3RandomService'] = rand

tray.AddModule("I3InfiniteSource", "TheSource",
               Prefix               = "/data/user/axelpo/LLP-at-IceCube/dark-leptonic-scalar-simulation/resources/GeoCalibDetectorStatus_2021.Run135903.T00S1.Pass2_V1b_Snow211115.i3.gz",
               Stream               = icecube.icetray.I3Frame.DAQ
              )

tray.AddSegment(GenerateNaturalRateMuons, "muongun",
                NumEvents           = n_events,
                GCDFile             = "/data/user/axelpo/LLP-at-IceCube/dark-leptonic-scalar-simulation/resources/GeoCalibDetectorStatus_2021.Run135903.T00S1.Pass2_V1b_Snow211115.i3.gz",
                mctree_name         = "I3MCTree_preMuonProp",
                flux_model          = "GaisserH4a_atmod12_SIBYLL"
               )

tray.AddSegment(PropagateMuonsLLP,
                "propagator",
                RandomService       = rand,
                InputMCTreeName     = "I3MCTree_preMuonProp",
                OutputMCTreeName    = "I3MCTree",
                PROPOSAL_config_SM  = "/data/user/axelpo/LLP-at-IceCube/dark-leptonic-scalar-simulation/resources/config_SM.json",
                PROPOSAL_config_LLP = "/data/user/axelpo/LLP-at-IceCube/dark-leptonic-scalar-simulation/resources/config_DLS.json",
                OnlySaveLLPEvents   = True,
                only_one_LLP        = True,
               )

tray.Add("I3Writer", filename=filename)

tray.Execute()

