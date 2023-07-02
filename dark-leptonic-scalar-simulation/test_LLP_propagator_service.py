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
from I3Tray import I3Tray, I3Units
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

n_events = 1000

tray.context['I3RandomService'] = rand
tray.AddModule("I3InfiniteSource", "TheSource",
               Prefix="../../muon-bkg-study/gcdfile.i3.gz",
               Stream=icecube.icetray.I3Frame.DAQ)

tray.AddSegment(GenerateNaturalRateMuons, "muongun",
                NumEvents=n_events,
                #GCDFile="../../muon-bkg-study/gcdfile.i3.gz",
                GCDFile="../GeoCalibDetectorStatus_2021.Run135903.T00S1.Pass2_V1b_Snow211115.i3.gz",
                mctree_name="I3MCTree_preMuonProp",
                flux_model="GaisserH4a_atmod12_SIBYLL")

tray.AddSegment(PropagateMuonsLLP,
                "propagator",
                RandomService=rand,
                InputMCTreeName="I3MCTree_preMuonProp",
                OutputMCTreeName="I3MCTree",
                PROPOSAL_config_SM="config_SM.json",
                PROPOSAL_config_LLP="config_DLS.json",
                OnlySaveLLPEvents=True,
               )

tray.Add("I3Writer", filename="test.i3")

tray.Execute()

