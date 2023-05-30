import icecube
import icecube.icetray
from icecube import dataclasses

class I3PropagatorServicePROPOSAL_LLP(icecube._sim_services.I3PropagatorService):
    def __init__(self, config_file_sm = "config_SM.json", config_file_llp = "config_DLS.json"):
        super().__init__()
        self.sm_propagator = icecube.PROPOSAL.I3PropagatorServicePROPOSAL(config_file=config_file_sm)
        self.llp_propagator = icecube.PROPOSAL.I3PropagatorServicePROPOSAL(config_file=config_file_llp)

    def Propagate(self, p, frame):
        if self.llp_counter == 0:
            daughters = self.llp_propagator.Propagate(p)
        else:
            daughters = self.sm_propagator.Propagate(p)
        self.checkLLP(daughters)
        return daughters
    
    def checkLLP(self, daughters):
        """ iterate through daughters to see if LLP happened """
        previous_particle_was_LLP = False
        for d in daughters:
            if previous_particle_was_LLP:
                # if here, then you are at the particle RIGHT after LLP (final state lepton of LLP production)
                parent_energy = self.llp_info["llp_energy"] + d.energy
                self.llp_info["fractional_energy"] = self.llp_info["llp_energy"]/parent_energy*1.0
                previous_particle_was_LLP = False
            if d.type == 0:
                self.llp_counter += 1
                self.llp_info["length"] = d.length
                self.llp_info["prod_x"] = d.pos.x
                self.llp_info["prod_y"] = d.pos.y
                self.llp_info["prod_z"] = d.pos.z
                self.llp_info["azimuth"] = d.dir.azimuth
                self.llp_info["zenith"] = d.dir.zenith
                self.llp_info["llp_energy"] = d.energy
                previous_particle_was_LLP = True

        return
    
    def SetRandomNumberGenerator(self, rand_service):
        self.sm_propagator.SetRandomNumberGenerator(rand_service)
        self.llp_propagator.SetRandomNumberGenerator(rand_service)
        
    def reset(self):
        self.llp_counter = 0
        self.llp_info = dataclasses.I3MapStringDouble()
        
    def write_LLPInfo(self, frame):
        self.llp_info["interaction"] = self.llp_counter
        # default if no  LLP production
        if self.llp_counter == 0:
                self.llp_info["length"] = -1
                self.llp_info["prod_x"] = 9999
                self.llp_info["prod_y"] = 9999
                self.llp_info["prod_z"] = 9999
                self.llp_info["azimuth"] = 9999
                self.llp_info["zenith"] = 9999
                self.llp_info["llp_energy"] = -1
                self.llp_info["fractional_energy"] = -1
        frame["LLPInfo"] = self.llp_info
    