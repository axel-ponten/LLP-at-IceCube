"""
Tray segments for muon propagation with Long Lived Particle (LLP) interaction. This is a modification of simprod-scripts PropagateMuons.py
"""
import os

import icecube
import icecube.icetray
import icecube.dataclasses
import icecube.phys_services
import icecube.sim_services
import icecube.simclasses
import icecube.cmc
import icecube.PROPOSAL
import json
from I3PropagatorServicePROPOSAL_LLP import I3PropagatorServicePROPOSAL_LLP

@icecube.icetray.traysegment
def PropagateMuonsLLP(tray, name,
                   RandomService=None,
                   CylinderRadius=None,
                   CylinderLength=None,
                   SaveState=True,
                   InputMCTreeName="I3MCTree_preMuonProp",
                   OutputMCTreeName="I3MCTree",
                   PROPOSAL_config_SM="config_SM.json",
                   PROPOSAL_config_LLP="config_DLS.json",
                   OnlySaveLLPEvents=True,
                   **kwargs):
    r"""Propagate muons.

    This segment propagates muons through ice with ``PROPOSAL``; it
    simulates lepton decays and energy losses due to ionization,
    bremsstrahlung, photonuclear interactions, and pair production.
    It also includes Long Lived Particle (LLP) production.

    :param I3RandomService RandomService:
        Random number generator service
    :param float CylinderRadius:
        Radius of the target volume in m
        (this param is now depricated, use the config file in the detector configuration)
    :param float CylinderLength:
        Full height of the target volume in m
        (this param is now depricated, use the config file in the detector configuration)
    :param bool SaveState:
        If set to `True`, store the state of the supplied RNG.
    :param str InputMCTree:
        Name of input :ref:`I3MCTree` frame object
    :param str OutputMCTree:
        Name of output :ref:`I3MCTree` frame object
    :param \**kwargs:
        Additional keyword arguments are passed to
        :func:`icecube.simprod.segments.make_propagator`.

    """
    if CylinderRadius is not None:
        icecube.icetray.logging.log_warn(
            "The CylinderRadius now should be set in the configuration file in the detector configuration")
    if CylinderLength is not None:
        icecube.icetray.logging.log_warn(
            "The CylinderLength now should be set in the configuration file in the detector configuration")
    propagator_map, muon_propagator = make_propagators(tray, PROPOSAL_config_SM, PROPOSAL_config_LLP, **kwargs)

    if SaveState:
        rng_state = InputMCTreeName+"_RNGState"
    else:
        rng_state = ""

    # write simulation information to S frame
    tray.Add(write_simulation_information,
             PROPOSAL_config_LLP = PROPOSAL_config_LLP,
             Streams=[icecube.icetray.I3Frame.Simulation])
    
    # reset the LLP info before each event
    tray.Add(lambda frame : muon_propagator.reset(), 
             Streams=[icecube.icetray.I3Frame.DAQ])

    tray.AddModule("I3PropagatorModule", name+"_propagator",
                   PropagatorServices=propagator_map,
                   RandomService=RandomService,
                   InputMCTreeName=InputMCTreeName,
                   OutputMCTreeName=OutputMCTreeName,
                   RNGStateName=rng_state)

    # write LLP information to frame
    tray.Add(lambda frame : muon_propagator.write_LLPInfo(frame), 
             Streams=[icecube.icetray.I3Frame.DAQ])
    
    if OnlySaveLLPEvents:
        # throw event away if no LLP interaction
        tray.Add(lambda frame : bool(frame["LLPInfo"]["interaction"]), 
                 Streams=[icecube.icetray.I3Frame.DAQ])
    
    # Add empty MMCTrackList objects for events that have none.
    def add_empty_tracklist(frame):
        if "MMCTrackList" not in frame:
            frame["MMCTrackList"] = icecube.simclasses.I3MMCTrackList()
        return True

    tray.AddModule(add_empty_tracklist, name+"_add_empty_tracklist",
                   Streams=[icecube.icetray.I3Frame.DAQ])

    return

def make_propagators(tray,                     
                     PROPOSAL_config_SM,
                     PROPOSAL_config_LLP,
                     SplitSubPeVCascades=True,
                     EmitTrackSegments=True,
                     MaxMuons=10,
                     ):
    """
    Set up propagators (PROPOSAL for muons and taus with LLP interaction, CMC for cascades)

    :param bool SplitSubPeVCascades: Split cascades into segments above 1 TeV. Otherwise, split only above 1 PeV.
    
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
    
    #muon_propagator = icecube.PROPOSAL.I3PropagatorServicePROPOSAL(config_file=PROPOSAL_config_file)
    muon_propagator = I3PropagatorServicePROPOSAL_LLP(PROPOSAL_config_SM, PROPOSAL_config_LLP)
    
    propagator_map = icecube.sim_services.I3ParticleTypePropagatorServiceMap()

    for pt in "MuMinus", "MuPlus", "TauMinus", "TauPlus":
        key = getattr(icecube.dataclasses.I3Particle.ParticleType, pt)
        propagator_map[key] = muon_propagator

    for key in icecube.sim_services.ShowerParameters.supported_types:
        propagator_map[key] = cascade_propagator
    
    return propagator_map, muon_propagator

def write_simulation_information(frame, PROPOSAL_config_LLP):
    """ Write LLP multiplier, mass, epsilon, etc. to S frame """
    if "LLPConfig" not in frame:
        file = open(PROPOSAL_config_LLP)
        config_json = json.load(file)
        
        simulation_info = icecube.dataclasses.I3MapStringDouble()
        simulation_info["llp_multiplier"] = config_json["global"]["llp_multiplier"]
        simulation_info["mass"] = config_json["global"]["llp_mass"]
        simulation_info["epsilon"] = config_json["global"]["llp_epsilon"]
        # TODO: add model to PROPOSAL config and then fix
        #simulation_model = icecube.dataclasses.I3String("DarkLeptonicScalar")
        simulation_model = icecube.dataclasses.I3String(config_json["global"]["llp"])

        frame["LLPConfig"] = simulation_info
        frame["LLPModel"] = simulation_model