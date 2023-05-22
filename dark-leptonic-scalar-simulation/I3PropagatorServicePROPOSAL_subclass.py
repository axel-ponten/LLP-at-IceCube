import icecube

class I3PropagatorServicePROPOSAL_subclass(icecube._sim_services.I3PropagatorService):
    def __init__(self, config_file):
        super().__init__()
        self.sm_propagator = icecube.PROPOSAL.I3PropagatorServicePROPOSAL(config_file=config_file)
        #self.DiagnosticMap.__init__() # error, this class can't be instatiated from python
        print(dir(self.DiagnosticMap))
        print(help(self.DiagnosticMap))
        self.DiagnosticMap.Put("test", 5)
        #self.DiagnosticMap["test"] = 5
        
    def Propagate(self, p, frame):
        return self.sm_propagator.Propagate(p, frame)
    
    def SetRandomNumberGenerator(self, rand_service):
        self.sm_propagator.SetRandomNumberGenerator(rand_service)
    