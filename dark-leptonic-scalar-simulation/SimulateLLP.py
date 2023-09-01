import argparse
import json
import shutil
import os
import os.path
from datetime import datetime
import icecube

from icecube import icetray, dataclasses, simclasses, dataio
import icecube.icetray
import icecube.dataclasses
import icecube.dataio
import icecube.phys_services

from icecube.simprod.util import ReadI3Summary, WriteI3Summary
from icecube.simprod.util import CombineHits, DrivingTime, SetGPUEnvironmentVariables

from icecube.simprod.util import simprodtray, arguments
from icecube.simprod.util.simprodtray import RunI3Tray

from icecube.icetray import I3Tray, I3Units
from icecube.simprod.util import BasicCounter
from icecube.simprod.segments import GenerateCosmicRayMuons, PropagateMuons, GenerateNaturalRateMuons, PPC

from icecube.dataclasses import *
from icecube import clsim
from icecube import polyplopia
from icecube import PROPOSAL
from PropagateMuonsLLP import PropagateMuonsLLP
from icecube.icetray import I3Frame
from icecube.simprod import segments
from icecube import phys_services
from icecube import sim_services
from icecube import vuvuzela
from icecube import DOMLauncher
from icecube import trigger_sim

from icecube.production_histograms import ProductionHistogramModule
from icecube.production_histograms.histogram_modules.simulation.pmt_response import PMTResponseModule
from icecube.production_histograms.histogram_modules.simulation.dom_mainboard_response import InIceResponseModule
from icecube.production_histograms.histogram_modules.simulation.trigger import TriggerModule
from icecube.production_histograms.histograms.simulation.noise_occupancy import NoiseOccupancy
from icecube.production_histograms.histogram_modules.simulation.mctree_primary import I3MCTreePrimaryModule
from icecube.production_histograms.histogram_modules.simulation.mctree import I3MCTreeModule
from icecube.production_histograms.histogram_modules.simulation.mcpe_module import I3MCPEModule

def add_args(parser):
    """
    Args:
        parser (argparse.ArgumentParser): the command-line parser
    """
    
    arguments.add_summaryfile(parser)

    parser.add_argument("--NoHistogram", dest="enablehistogram",
                        default=True, action="store_false", required=False,
                        help='Dont write a SanityChecker histogram file.')

    parser.add_argument("--HistogramFilename", dest="histogramfilename",
                        default="histogram.pkl", type=str, required=False,
                        help='Histogram filename.')
    #arguments.add_enablehistogram(parser) # use default true instead
    #arguments.add_histogramfilename(parser)

    arguments.add_nproc(parser)
    arguments.add_procnum(parser)
    arguments.add_seed(parser)
    arguments.add_usegslrng(parser)
    arguments.add_icemodellocation(parser)
    arguments.add_icemodel(parser)
    arguments.add_holeiceparametrization(parser)
    arguments.add_oversize(parser)
    arguments.add_efficiency(parser)

    arguments.add_propagatemuons(parser, True)

    arguments.add_photonseriesname(parser)

    arguments.add_gpu(parser)
    arguments.add_usegpus(parser, False)

    parser.add_argument("--outputfile", dest="outputfile",
                        default="", type=str, required=False,
                        help="Output file. Default creates name from PROPOSAL config and nevents.")
    
    # directory parameters
    parser.add_argument("--parentdirectory", dest="parentdirectory",
                        default="/data/user/axelpo/LLP-data/",
                        type=str, required=False,
                        help="Directory where to save folder with all output.")
    
    parser.add_argument("--dirname", dest="dirname",
                        default="",
                        type=str, required=False,
                        help="Name of folder with all output.")

    # MuonGun/LLPGun parameters
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
    parser.add_argument("--no-natural-rate", dest="natural_rate",
                        default=True, action="store_false", required=False,
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
    
    parser.add_argument("--nevents", dest="nevents",
                        type=int, required=True,
                        help="Number of events MuonGun should simulate.")
    
    parser.add_argument("--gcdfile", dest="gcdfile",
                        default="/data/user/axelpo/LLP-at-IceCube/dark-leptonic-scalar-simulation/resources/GeoCalibDetectorStatus_2021.Run135903.T00S1.Pass2_V1b_Snow211115.i3.gz", type=str, required=False,
                        help="GCD file.")
    
    parser.add_argument("--PROPOSAL-config-SM", dest="config_SM",
                        default="/data/user/axelpo/LLP-at-IceCube/dark-leptonic-scalar-simulation/resources/config_SM.json", type=str, required=False,
                        help="PROPOSAL config file without LLP.")
    
    parser.add_argument("--PROPOSAL-config-LLP", dest="config_LLP",
                        default=None, type=str, required=False,
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
    
    parser.add_argument("--LLP-model", dest="LLP-model",
                        default=None, type=str, required=False,
                        help="Manually set model of LLP. Default is in config file.")
    
    parser.add_argument("--mass", dest="mass",
                        default=None, type=float, required=False,
                        help="Manually set mass of LLP. Default is in config file.")
    
    parser.add_argument("--eps", dest="eps",
                        default=None, type=float, required=False,
                        help="Manually set epsilon of LLP. Default is in config file.")
    
    parser.add_argument("--bias", dest="bias",
                        default=None, type=float, required=False,
                        help="Manually set bias multiplier of LLP. Default is in config file.")
    
    parser.add_argument("--use-clsim", dest="use-clsim",
                        default=False, action="store_true", required=False,
                        help="Use CLSim instead of PPC.")
    
    parser.add_argument("--no-RunMPHitFilter", dest="runmphitfilter",
                        default=True, action="store_false", required=False,
                        help="Don't run polyplopia's mphitfilter")
    parser.add_argument("--gpulib", dest="gpulib",
                        default="opencl", type=str, required=False,
                        help="set gpu library to load (defaults to cuda)")
    parser.add_argument("--volumecyl", dest="volumecyl",
                        default=False, action="store_true", required=False,
                        help="Don't set volume to regular cylinder (set flag for 300m spacing from the DOMs)")
    parser.add_argument("--MCTreeName", dest="mctreename",
                        default="I3MCTree", type=str, required=False,
                        help="Name of MCTree frame object")
    parser.add_argument("--KeepEmptyEvents", dest="keepemptyevents",
                        default=False, action="store_true", required=False,
                        help="Don't discard events with no MCPEs")
    parser.add_argument("--TempDir", dest="tempdir",
                        default=None, type=str, required=False,
                        help='Temporary working directory with the ice model')
    
    
    # detector parameters
    parser.add_argument("--MCType", dest="mctype",
                        default='corsika_weighted', type=str, required=False,
                        help='Generator particle type')
    parser.add_argument("--UseLinearTree", dest="uselineartree",
                        default=False, action="store_true", required=False,
                        help='Use I3LinearizedMCTree for serialization')
    parser.add_argument("--MCPrescale", dest="mcprescale",
                        default=0, type=int, required=False,
                        help='Prescale for keeping additional Monte Carlo info in the frame')
    parser.add_argument("--IceTop", dest="icetop",
                        default=False, action="store_true", required=False,
                        help='Do IceTop Simulation?')
    parser.add_argument("--noInIce", dest="notinice",
                        default=False, action="store_true", required=False,
                        help='Do not simulate InIce part. If this is set, --IceTop is forced: specifying it is rendundant')
    parser.add_argument("--Genie", dest="genie",
                        default=False, action="store_true", required=False,
                        help='Assume separate Genie MCPEs and BG MCPEs')
    parser.add_argument("--no-FilterTrigger", dest="filtertrigger",
                        default=True, action="store_false", required=False,
                        help="Don't filter untriggered events")
    parser.add_argument("--no-Trigger", dest="trigger",
                        default=True, action="store_false", required=False,
                        help="Don't run trigger simulation")
    parser.add_argument("--LowMem", dest="lowmem",
                        default=False, action="store_true", required=False,
                        help='Low Memory mode')
    parser.add_argument("--no-BeaconLaunches", dest="beaconlaunches",
                        default=True, action="store_false", required=False,
                        help="Don't simulate beacon launches")
    parser.add_argument("--TimeShiftSkipKeys", dest="timeshiftskipkeys",
                        default=[], type=arguments.str_comma_list, required=False,
                        help='Skip keys in the triggersim TimeShifter')
    parser.add_argument("--SampleEfficiency", dest="sampleefficiency",
                        default=0.0, type=float, required=False,
                        help='Resample I3MCPESeriesMap for different efficiency')
    parser.add_argument("--GeneratedEfficiency", dest="generatedefficiency",
                        default=0.0, type=float, required=False,
                        help='Generated efficiency for resampling')
    parser.add_argument("--RunID", dest="runid",
                        default=0, type=int, required=False,
                        help='RunID')
    parser.add_argument("--MCPESeriesName", dest="mcpeseriesname",
                        default='I3MCPESeriesMap', type=str, required=False,
                        help='Name of MCPESeriesMap in frame')
    parser.add_argument("--DetectorName", dest="detectorname",
                        default='IC86', type=str, required=False,
                        help='Name of detector')
    parser.add_argument("--SkipKeys", dest="skipkeys",
                        default=[], type=arguments.str_comma_list, required=False,
                        help='Skip keys for the writer')
    

    

def configure_tray(tray, params, stats, logger):
    """
    Configures the I3Tray instance: adds modules, segments, services, etc.

    Args:
        tray (I3Tray): the IceProd tray instance
        params (dict): command-line arguments (and default values)
                            referenced as dict entries; see add_args()
        stats (dict): dictionary that collects run-time stats
        logger (logging.Logger): the logger for this script
    """
    if params['gpu'] is not None and params['usegpus']:
        SetGPUEnvironmentVariables(params['gpu'])

    tray.AddModule("I3InfiniteSource", "TheSource",
                   Prefix=params['gcdfile'],
                   Stream=icecube.icetray.I3Frame.DAQ)

    ### MUONS WITH MUONGUN ###
    if params['natural_rate']:
        tray.AddSegment(GenerateNaturalRateMuons, "muongun",
                        NumEvents=1e15,
                        mctree_name="I3MCTree_preMuonProp",
                        flux_model="GaisserH4a_atmod12_SIBYLL")
    else:
        # Configure tray segment that actually does stuff.
        tray.AddSegment(GenerateCosmicRayMuons, "muongun",
                        mctree_name="I3MCTree_preMuonProp",
                        num_events=1e15,
                        flux_model=params['model'],
                        gamma_index=params['gamma'],
                        energy_offset=params['offset'],
                        energy_min=params['emin'],
                        energy_max=params['emax'],
                        cylinder_length=params['length'],
                        cylinder_radius=params['radius'],
                        cylinder_x=params['x'],
                        cylinder_y=params['y'],
                        cylinder_z=params['z'],
                        inner_cylinder_length=params['length_dc'],
                        inner_cylinder_radius=params['radius_dc'],
                        inner_cylinder_x=params['x_dc'],
                        inner_cylinder_y=params['y_dc'],
                        inner_cylinder_z=params['z_dc'],
                        use_inner_cylinder=params['deepcore'])

    ### PROPOSAL WITH LLP INTERACTION ###
    tray.AddSegment(PropagateMuonsLLP,
                    "propagator",
                    RandomService          = tray.context["I3RandomService"],
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

    
    if params["use-clsim"]:
        ### PHOTONS WITH CLSIM ###
        tray.AddSegment(clsim.I3CLSimMakeHits, "makeCLSimHits",
                        GCDFile=params['gcdfile'],
                        RandomService=tray.context["I3RandomService"],
                        UseGPUs=params['usegpus'],
                        UseOnlyDeviceNumber=params['useonlydevicenumber'],
                        UseCPUs=not params['usegpus'],
                        IceModelLocation=os.path.join(params['icemodellocation'], params['icemodel']),
                        DOMEfficiency=params['efficiency'],
                        UseGeant4=False,
                        DOMOversizeFactor=params['oversize'],
                        MCTreeName="I3MCTree",
                        MCPESeriesName=params['photonseriesname'],
                        PhotonSeriesName=params['rawphotonseriesname'],
                        HoleIceParameterization=params['holeiceparametrization'])
    else:
        ### PHOTONS WITH PPC
        tray.AddSegment(segments.PPC.PPCTraySegment, "ppc_photons",
                        GPU=params['gpu'],
                        UseGPUs=params['usegpus'],
                        DOMEfficiency=params['efficiency'],
                        DOMOversizeFactor=params['oversize'],
                        IceModelLocation=params['icemodellocation'],
                        HoleIceParameterization=params['holeiceparametrization'],
                        IceModel=params['icemodel'],
                        volumecyl=params['volumecyl'],
                        gpulib=params['gpulib'],
                        InputMCTree=params['mctreename'],
                        keep_empty_events=params['keepemptyevents'],
                        MCPESeriesName=params['photonseriesname'],
                        tempdir=params['tempdir'])

    tray.AddModule("MPHitFilter", "hitfilter",
                   HitOMThreshold=1,
                   RemoveBackgroundOnly=False,
                   I3MCPESeriesMapName=params['photonseriesname'])


    ### DETECTOR ###

    mcprescale = params['nproc']+1 if params['mcprescale']==0 else params['mcprescale']
    tray.AddSegment(segments.DetectorSegment, "detector",
                    gcdfile=params['gcdfile'],
                    mctype=params['mctype'],
                    uselineartree=params['uselineartree'],
                    detector_label=params['detectorname'],
                    runtrigger=params['trigger'],
                    filtertrigger=params['filtertrigger'],
                    stats=stats,
                    inice=not params['notinice'],
                    icetop=params['icetop'] or params['notinice'],
                    genie=params['genie'],
                    prescale=params['mcprescale'],
                    lowmem=params['lowmem'],
                    BeaconLaunches=params['beaconlaunches'],
                    TimeShiftSkipKeys=params['timeshiftskipkeys'],
                    SampleEfficiency=params['sampleefficiency'],
                    GeneratedEfficiency=params['generatedefficiency'],
                    RunID=params['runid'],
                    KeepMCHits=not params['procnum'] % mcprescale,#params['mcprescale'],
                    KeepPropagatedMCTree=not params['procnum'] % mcprescale,#params['mcprescale'],
                    KeepMCPulses=not params['procnum'] % mcprescale)#params['mcprescale'])


    if params['enablehistogram'] and params['histogramfilename']:
        tray.AddModule(ProductionHistogramModule,
                       Histograms=[I3MCTreePrimaryModule,
                                   I3MCTreeModule,
                                   I3MCPEModule,
                                   PMTResponseModule,
                                   InIceResponseModule,
                                   TriggerModule,
                                   NoiseOccupancy],
                       OutputFilename=params['histogramfilename'])
    
    
#############################################
########### SETUP TRAY AND RUN IT ###########
#############################################

# Get Params
parser = argparse.ArgumentParser(description="Full LLP simulation script")
add_args(parser)
params = vars(parser.parse_args())  # dict()

# PROPOSAL parameters
if params["config_LLP"] is None:
    # if no config file then you must pass all LLP parameters manually
    if (params["mass"] is None) or (params["eps"] is None) or (params["bias"] is None) or (params["LLP-model"] is None):
        icetray.logging.log_fatal("If no PROPOSAL LLP config file passed to argparse then you must pass arguments for model, mass, epsilon and bias of the LLP.", "SimulateLLP")
    file = open(params["config_SM"])
    config_json = json.load(file)
    file.close()
    config_json["global"]["llp_enable"]     = True
    config_json["global"]["llp_multiplier"] = params["bias"]
    config_json["global"]["llp_mass"]       = params["mass"]
    config_json["global"]["llp_epsilon"]    = params["eps"]
    config_json["global"]["llp"]            = params["LLP-model"]
else:
    file = open(params["config_LLP"])
    config_json = json.load(file)
    file.close()

# Create directory where to save all output, name contains information about simulation
llp_multiplier   = config_json["global"]["llp_multiplier"]
mass             = config_json["global"]["llp_mass"]
eps              = config_json["global"]["llp_epsilon"]
simulation_model = config_json["global"]["llp"]

if params["dirname"] == "":
    default_directory_name = simulation_model+".mass-"+str(mass)+".eps-" + str(eps)+".nevents-"+str(params["nevents"]) + "_" + datetime.now().strftime("%y%m%d_%H%M%S") 
    params["dirname"] = default_directory_name

directory_path = params["parentdirectory"] + params["dirname"]
if params["natural_rate"]:
    directory_path += "_naturalrate"
if params["both_prod_decay_inside"]:
    directory_path += "_fullycontained"
if bool(params["min_LLP_length"]):
    directory_path += "_macrogaps"
directory_path += "/"

os.makedirs(directory_path)

# copy or save new PROPOSAL config file to path
if params["config_LLP"] is None:
    params["config_LLP"] = directory_path + "config_LLP.json"
    file = open(params["config_LLP"], "w+")
    file.write(json.dumps(config_json))
    file.close()
else:
    shutil.copy(params["config_LLP"], directory_path)

# Use custom output name?
if params["outputfile"] == "":
    # default filename
    params["outputfile"] = "LLPSimulation."+simulation_model+".mass-"+str(mass)+".eps-" + str(eps)+".bias-"+str(llp_multiplier)+".nevents-"+str(params["nevents"])+".i3.gz"

# write LLP selection information to separate logfile
LLPSimulationInfo = {
    "model"                       : simulation_model,
    "mass"                        : mass,
    "epsilon"                     : eps,
    "min_length"                  : params["min_LLP_length"],
    "only_fully_contained_events" : params["both_prod_decay_inside"],
    "only_save_LLP"               : params["OnlySaveLLPEvents"],
    "natural_muon_rate"           : params["natural_rate"],
}    
with open(directory_path + "llp_simulation_info.json", "w+") as LLPFile:
    json.dump(LLPSimulationInfo, LLPFile)

# add directory path to all files
params['outputfile']        = directory_path + params['outputfile']
params['summaryfile']       = directory_path + params['summaryfile']
params['histogramfilename'] = directory_path + params['histogramfilename']
    
# suppress warnings from PROPOSAL LLP integration
icetray.set_log_level(icetray.I3LogLevel.LOG_FATAL)

# Execute Tray
summary = RunI3Tray(params, configure_tray, "MuonGunGenerator",
                    summaryfile=params['summaryfile'],
                    summaryin=icecube.dataclasses.I3MapStringDouble(),
                    outputfile=params['outputfile'],
                    seed=params['seed'],
                    nstreams=params['nproc'],
                    streamnum=params['procnum'],
                    usegslrng=params['usegslrng'])