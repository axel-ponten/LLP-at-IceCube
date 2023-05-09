"""
This tray simulates Dark Leptonic Scalar production from a MuonGun spectrum.
An icetray environment which has the Long Lived Particle modified version of PROPOSAL is needed.
"""

from icecube.simprod.util import ReadI3Summary, WriteI3Summary
from icecube.simprod.util import CombineHits, DrivingTime, SetGPUEnvironmentVariables
import os.path
from icecube.simprod.util import simprodtray, arguments
from icecube.simprod.util.simprodtray import RunI3Tray
import argparse
import icecube.icetray
import icecube.dataclasses
import icecube.dataio
import icecube.phys_services
from I3Tray import I3Tray, I3Units
from icecube.simprod.util import BasicCounter
from icecube.simprod.segments import GenerateCosmicRayMuons, PropagateMuons, GenerateNaturalRateMuons
from icecube import clsim
from icecube import polyplopia
from icecube import phys_services
from icecube import PROPOSAL
from icecube.production_histograms import ProductionHistogramModule
from icecube.production_histograms.histogram_modules.simulation.mctree_primary import I3MCTreePrimaryModule
from icecube.production_histograms.histogram_modules.simulation.mctree import I3MCTreeModule
from icecube.production_histograms.histogram_modules.simulation.mcpe_module import I3MCPEModule


def add_args(parser):
    """
    Args:
        parser (argparse.ArgumentParser): the command-line parser
    """
    arguments.add_gcdfile(parser)
    arguments.add_outputfile(parser)
    arguments.add_summaryfile(parser)
    arguments.add_enablehistogram(parser)
    arguments.add_histogramfilename(parser)

    arguments.add_nproc(parser)
    arguments.add_procnum(parser)
    arguments.add_seed(parser)
    arguments.add_usegslrng(parser)

    arguments.add_nevents(parser)

    arguments.add_icemodellocation(parser)
    arguments.add_icemodel(parser)
    arguments.add_holeiceparametrization(parser)
    arguments.add_oversize(parser)
    arguments.add_efficiency(parser)

    arguments.add_propagatemuons(parser, True)

    arguments.add_photonseriesname(parser)

    arguments.add_gpu(parser)
    arguments.add_usegpus(parser, False)

    parser.add_argument("--model", dest="model",
                        default="Hoerandel5_atmod12_SIBYLL",
                        type=str, required=False,
                        help="primary cosmic-ray flux parametrization")
    parser.add_argument("--gamma", dest="gamma",
                        default=2., type=float, required=False,
                        help="power law spectral index")
    parser.add_argument("--offset", dest="offset",
                        default=700., type=float, required=False,
                        help="power law offset in GeV")
    parser.add_argument("--emin", dest="emin",
                        default=1e4, type=float, required=False,
                        help="mininum generated energy in GeV")
    parser.add_argument("--emax", dest="emax",
                        default=1e7, type=float, required=False,
                        help="maximum generated energy in GeV")
    parser.add_argument("--length", dest="length",
                        default=1600., type=float, required=False,
                        help="cylinder length in m")
    parser.add_argument("--radius", dest="radius",
                        default=800., type=float, required=False,
                        help="cylinder radius in m")
    parser.add_argument("--x", dest="x",
                        default=0., type=float, required=False,
                        help="cylinder x-position in m")
    parser.add_argument("--y", dest="y",
                        default=0., type=float, required=False,
                        help="cylinder y-position in m")
    parser.add_argument("--z", dest="z",
                        default=0., type=float, required=False,
                        help="cylinder z-position in m")
    parser.add_argument("--length-dc", dest="length_dc",
                        default=500., type=float, required=False,
                        help="inner cylinder length in m")
    parser.add_argument("--radius-dc", dest="radius_dc",
                        default=150., type=float, required=False,
                        help="inner cylinder radius in m")
    parser.add_argument("--x-dc", dest="x_dc",
                        default=46.3, type=float, required=False,
                        help="inner cylinder x-position in m")
    parser.add_argument("--y-dc", dest="y_dc",
                        default=-34.9, type=float, required=False,
                        help="inner cylinder y-position in m")
    parser.add_argument("--z-dc", dest="z_dc",
                        default=-300., type=float, required=False,
                        help="inner cylinder z-position in m")
    parser.add_argument("--deepcore", dest="deepcore",
                        default=False, action="store_true", required=False,
                        help="use inner cylinder")
    parser.add_argument("--no-propagate-photons", dest="propagate_photons",
                        default=True, action="store_false", required=False,
                        help="Don't run ClSim.")
    parser.add_argument("--natural-rate", dest="natural_rate",
                        default=False, action="store_true", required=False,
                        help="Sample natural rate muon bundles")
    parser.add_argument("--UseOnlyDeviceNumber", dest="useonlydevicenumber",
                        default=0, type=int, required=False,
                        help="Use only this device.")
    parser.add_argument("--RawPhotonSeriesName", dest="rawphotonseriesname",
                        default=None, type=str, required=False,
                        help="Raw Photon Series Name")
    parser.add_argument("--no-KeepMCTree", dest="keepmctree",
                        default=True, action="store_false", required=False,
                        help='Delete propagated MCTree')




class PropagateWithLLP(icetray.I3Module):
    def __init__(self, ctx):
        icetray.I3Module.__init__(self,ctx)
        
        self.PROPOSAL_config_SM = ""
        self.AddParameter("PROPOSAL_config_SM", "PROPOSAL config without LLP production", self.PROPOSAL_config_SM)
        self.PROPOSAL_config_LLP = ""
        self.AddParameter("PROPOSAL_config_SM", "PROPOSAL config with LLP production", self.PROPOSAL_config_LLP)
        self.force_LLP = False
        self.AddParameter("force_LLP", "Repropagate events that did not produce an LLP", self.force_LLP)
        
    def Configure(self):
        # create muon propagators with and without LLP production
        self.PROPOSAL_config_SM = self.GetParameter("PROPOSAL_config_SM")
        self.PROPOSAL_config_LLP = self.GetParameter("PROPOSAL_config_LLP")
        assert(self.PROPOSAL_config_SM != "")
        assert(self.PROPOSAL_config_LLP != "")
        # TODO is this correct?
        self.propagator_SM = PROPOSAL.I3PropagatorServicePROPOSAL(config_file=self.PROPOSAL_config_SM)
        self.propagator_LLP = PROPOSAL.I3PropagatorServicePROPOSAL(config_file=self.PROPOSAL_config_LLP)
        
        # create cascade propagators
        MaxMuons = 10
        SplitSubPeVCascades = True
        self.cascade_propagator = icecube.cmc.I3CascadeMCService(icecube.phys_services.I3GSLRandomService(1))  # Dummy RNG
        self.cascade_propagator.SetEnergyThresholdSimulation(1*I3Units.PeV)
        if SplitSubPeVCascades:
            self.cascade_propagator.SetThresholdSplit(1*I3Units.TeV)
        else:
            self.cascade_propagator.SetThresholdSplit(1*I3Units.PeV)
        self.cascade_propagator.SetMaxMuons(MaxMuons)
        
    
    def DAQ(self, frame):
        # check here for logic? https://github.com/icecube/icetray/blob/9b7d6278df63fe59ca8ef8f3924a5ebae3ce6137/sim-services/private/sim-services/I3PropagatorModule.cxx
        # reset LLP count
        self.LLPcount = 0
        tree_premuonprop = frame["I3Tree_preMuonProp"]
        self.output_tree = I3MCTree(tree_premuonprop)
        
        # propagate
        primary_children = tree_premuonprop.children(tree_premuonprop.get_head())
        self.RecursivePropagation(primary_children)
        # save all particles which we know how to propagate (no neutrinos etc.)

        if self.force_LLP and self.LLPcount == 0:
            # repropagate
            pass
        
        # TODO: include LLP info in some frame object
        
        # write outputtree
        frame["I3MCTree"] = self.output_tree
        # push frame
        self.PushFrame(frame)
    
    def RecursivePropagation(self, particle_list):
        # somewhere here use tree.append_children(id, ListI3Particle)
        
        propagatable_particles = [13, -13]
        
        for p in particle_list:
            if p.type in propagatable_particles:
                if self.LLPcount > 0:
                    daughters = self.propagator_SM.Propagate(p)
                else:
                    daughters = self.propagator_LLP.Propagate(p)
                self.output_tree.append_children(p.id, daughters)
                self.RecursivePropagation(daughters) # recursive propagation of daughters
        return
        
    def Finish(self):
        pass
        

# Get Params
#parser = argparse.ArgumentParser(description="MuonGunGenerator script")
#add_args(parser)
#params = vars(parser.parse_args())  # dict()

# set up tray
tray = I3Tray()

rand = phys_services.I3GSLRandomService(seed=0)


tray.context['I3RandomService'] = rand
tray.AddModule("I3InfiniteSource", "TheSource",
               Prefix="../../muon-bkg-study/gcdfile.i3.gz",
               Stream=icecube.icetray.I3Frame.DAQ)

tray.AddSegment(GenerateNaturalRateMuons, "muongun",
                NumEvents=10,
                mctree_name="I3MCTree_preMuonProp",
                flux_model="GaisserH4a_atmod12_SIBYLL")

tray.Add(PropagateWithLLP,
         Stream=icecube.icetray.I3Frame.DAQ,
         PROPOSAL_config_SM="config_SM.json",
         PROPOSAL_config_LLP="config_DLS.json",
         force_LLP = True
        )

tray.Add("I3Writer", filename="test.i3")

tray.Execute()

