"""
This tray simulates Dark Leptonic Scalar production from a MuonGun spectrum.
An icetray environment which has the Long Lived Particle modified version of PROPOSAL is needed.
"""

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
from icecube.production_histograms import ProductionHistogramModule
from icecube.production_histograms.histogram_modules.simulation.mctree_primary import I3MCTreePrimaryModule
from icecube.production_histograms.histogram_modules.simulation.mctree import I3MCTreeModule
from icecube.production_histograms.histogram_modules.simulation.mcpe_module import I3MCPEModule
import icecube

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




def WriteLLPInformation(frame):
    LLPinfo = dataclasses.I3MapStringDouble()
    
    LLPinfo["gap_length"] = 0
    LLPinfo["prod_x"] = -9999
    LLPinfo["prod_y"] = -9999
    LLPinfo["prod_y"] = -9999
    
    tree = frame["I3MCTree"]
    
    frame["LLPInformation"] = LLPinfo
    return True
    
    
class PropagateWithLLP(icetray.I3Module):
    def __init__(self, ctx):
        icetray.I3Module.__init__(self,ctx)
        
        self.PROPOSAL_config_SM = ""
        self.AddParameter("PROPOSAL_config_SM", "PROPOSAL config without LLP production", self.PROPOSAL_config_SM)
        self.PROPOSAL_config_LLP = ""
        self.AddParameter("PROPOSAL_config_LLP", "PROPOSAL config with LLP production", self.PROPOSAL_config_LLP)
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
        
        self.force_LLP = self.GetParameter("force_LLP")
        
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
        
        # which particles can be propagated where?
        self.propagatable_particles = [13, -13, 15, -15] # for PROPOSAL
        self.cascade_particles = icecube.sim_services.ShowerParameters.supported_types # for cmc
        
    
    def DAQ(self, frame):
        # reference: https://github.com/icecube/icetray/blob/9b7d6278df63fe59ca8ef8f3924a5ebae3ce6137/sim-services/private/sim-services/I3PropagatorModule.cxx
        
        # reset LLP count
        self.LLPcount = 0
        
        # propagate the primary's children
        tree_premuonprop = frame["I3MCTree_preMuonProp"]
        primary_children = tree_premuonprop.children(tree_premuonprop.get_head())
        #if self.force_LLP:
        #    while self.LLPcount == 0:
        #        self.output_tree = I3MCTree(tree_premuonprop)
        #        self.RecursivePropagation(primary_children)
        #else:
        #    self.output_tree = I3MCTree(tree_premuonprop)
        #    self.RecursivePropagation(primary_children)
        # TODO: include LLP info in some frame object
        self.output_tree = I3MCTree(tree_premuonprop)
        self.RecursivePropagation(primary_children)
        
        # write outputtree
        frame["I3MCTree"] = self.output_tree
        
        # push frame
        if self.force_LLP:
            if self.LLPcount > 0:
                self.PushFrame(frame)
        else:
            self.PushFrame(frame)
    
    def RecursivePropagation(self, particle_list):
        
        for p in particle_list:
            if p.type == 0:
                gap_length = p.length
                production_vertex = p.pos
                self.LLPcount += 1
                print("LLP production at", production_vertex, "with gap length", gap_length)
            if p.type in self.propagatable_particles and math.isnan(p.length):
                if self.LLPcount > 0:
                    daughters = self.propagator_SM.Propagate(p)
                else:
                    daughters = self.propagator_LLP.Propagate(p)
                for d in daughters:
                    # match output of I3PropagatorModule
                    if abs(d.type) > 2000000000:
                        d.length = 0
                self.output_tree.append_children(p.id, daughters)
                self.output_tree.at(p.id).length = p.length #updated length
                self.RecursivePropagation(daughters) # recursive propagation of daughters
            elif p.type in self.cascade_particles and math.isnan(p.length):
                daughters = self.cascade_propagator.Propagate(p)
                self.output_tree.append_children(p.id, daughters)
                self.RecursivePropagation(daughters) # recursive propagation of daughters
        return
        
    def Finish(self):
        pass
        
        

def make_standard_propagators(SplitSubPeVCascades=True,
                              EmitTrackSegments=True,
                              MaxMuons=10,
                              PROPOSAL_config_file="config_SM.json"):
    """
    Set up standard propagators (PROPOSAL for muons and taus, CMC for cascades)

    :param bool SplitSubPeVCascades: Split cascades into segments above 1 TeV. Otherwise, split only above 1 PeV.
    :param bool EmitTrackSegments:   Emit constant-energy track slices in addition to stochastic losses (similar to the output of I3MuonSlicer)
    :param str PROPOSAL_config_file: Path to PROPOSAL config file

    Keyword arguments will be passed to I3PropagatorServicePROPOSAL
    """
    from icecube.icetray import I3Units

    cascade_propagator = icecube.cmc.I3CascadeMCService(
        icecube.phys_services.I3GSLRandomService(1))  # Dummy RNG
    cascade_propagator.SetEnergyThresholdSimulation(1*I3Units.PeV)
    if SplitSubPeVCascades:
        cascade_propagator.SetThresholdSplit(1*I3Units.TeV)
    else:
        cascade_propagator.SetThresholdSplit(1*I3Units.PeV)
    cascade_propagator.SetMaxMuons(MaxMuons)
    muon_propagator = icecube.PROPOSAL.I3PropagatorServicePROPOSAL(
            config_file=PROPOSAL_config_file, slice_tracks=EmitTrackSegments)
    propagator_map =\
        icecube.sim_services.I3ParticleTypePropagatorServiceMap()

    for pt in "MuMinus", "MuPlus", "TauMinus", "TauPlus":
        key = getattr(icecube.dataclasses.I3Particle.ParticleType, pt)
        propagator_map[key] = muon_propagator

    for key in icecube.sim_services.ShowerParameters.supported_types:
        propagator_map[key] = cascade_propagator
    
    return propagator_map

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
                NumEvents=100,
                GCDFile="../../muon-bkg-study/gcdfile.i3.gz",
                mctree_name="I3MCTree_preMuonProp",
                flux_model="GaisserH4a_atmod12_SIBYLL")


#propagators = make_standard_propagators()
#tray.AddModule("I3PropagatorModule", "propagator",
#               PropagatorServices=propagators,
#               RandomService=rand,
#               InputMCTreeName="I3MCTree_preMuonProp",
#               OutputMCTreeName="I3MCTree",
#               RNGStateName="")
#tray.AddSegment(PropagateMuons,
#                "propagator",
#                RandomService=rand,
#                InputMCTreeName="I3MCTree_preMuonProp",
#                OutputMCTreeName="I3MCTree",
#                PROPOSAL_config_file="config_DLS.json",
#                EmitTrackSegments=False,
#               )

tray.Add(PropagateWithLLP,
         PROPOSAL_config_SM="config_SM.json",
         PROPOSAL_config_LLP="config_DLS.json",
         force_LLP = True,
        )


tray.Add("I3Writer", filename="test2.i3")

tray.Execute()

