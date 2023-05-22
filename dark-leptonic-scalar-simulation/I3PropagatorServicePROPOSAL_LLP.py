import icecube

class I3PropagatorServicePROPOSAL_LLP(icecube.sim_services.I3PropagatorService):
    def __init__(self, config_sm, config_llp):
        self.sm_propagator = icecube.PROPOSAL.I3PropagatorServicePROPOSAL(config_file=config_sm)
        self.llp_propagator = icecube.PROPOSAL.I3PropagatorServicePROPOSAL(config_file=config_llp)
        self.llp_counter = 0 # to decide whether to use SM or LLP propagator
        
    def Propagate(self, p):
        if self.llp_counter == 0:
            daughters = self.llp_propagator.Propagate(p)
        else:
            daughters = self.sm_propagator.Propagate(p)
        # TODO: check for LLP
        self.checkLLP(daughters)
        return daughters
    
    def checkLLP(self, daughters):
        """ iterate through daughters to see if LLP happened """
        for d in daughters:
            if d.type == 0:
                self.llp_counter += 1
        return