import icecube
from icecube import icetray, MuonGun, dataclasses
import numpy as np
from LLPEstimator import *
from estimation_utilities import *

"""
Iterate through a atmospheric muon spectrum and compute event-by-event detectable LLP probability.
"""

class LLPProbabilityCalculator(icetray.I3Module):
    def __init__(self,ctx):
        icetray.I3Module.__init__(self,ctx)
        
        self.gcdFile = ""
        self.AddParameter("GCDFile", "GCD file which defines the in-ice volume", self.gcdFile)
        
        self.n_steps = 100
        self.AddParameter("n_steps", "How many steps for segmented thick target approximation.", self.n_steps)
        
        # @TODO: add energy and zenith dependence on track length parameters.
        self.min_gap = 50.0
        self.AddParameter("min_gap", "Minimum detectable gap length in meters.", self.min_gap)
        
        self.entry_margin = 50.0
        self.AddParameter("entry_margin", "Track margin from entry point in meters.", self.entry_margin)
        
        self.exit_margin = 50.0
        self.AddParameter("exit_margin", "Track margin to exit point in meters.", self.exit_margin)

        self.LLPEstimator = None
        self.AddParameter("llp_estimator", "LLPEstimator object.", self.LLPEstimator)

    def Configure(self):
        self.min_gap      = self.GetParameter("min_gap")
        self.entry_margin = self.GetParameter("entry_margin")
        self.exit_margin  = self.GetParameter("exit_margin")
        self.n_steps      = self.GetParameter("n_steps")
        self.LLPEstimator = self.GetParameter("llp_estimator")
        # create surface for detector volume
        self.gcdFile = self.GetParameter("GCDFile")
        if self.gcdFile != "":
            self.surface = MakeSurface(self.gcdFile, self.padding)
        else:
            print("No GCD file provided, using 1000x500 MuonGun cylinder instead.")
            self.surface = MuonGun.Cylinder(1000,500) # approximate detector volume

    def DAQ(self, frame):
        # @TODO: what if muon stops before exiting detector? maybe its fine cus track.get_energy gives zero?
        
        # muon multiplicity check. @TODO: double check what this does when a neutrino appears
        track_list = MuonGun.Track.harvest(frame['I3MCTree_preMuonProp'], frame['MMCTrackList'])
        if len(track_list) == 1:
            track = track_list[0]
            # Find distance to entrance and exit from sampling volume
            intersections = self.surface.intersection(track.pos, track.dir)
            # trim for margins
            start_length = intersections.first + self.entry_margin
            stop_length  = intersections.second - self.exit_margin
            # check for available space
            if start_length - stop_length <= self.min_gap:
                probabilities = np.zeros(len(self.LLPEstimator.get_LLPModels())) # if no available space, return zero prob
            else:
                # create lists for LLPEstimator
                length_list   = np.linspace(start_length, stop_length, self.n_steps)
                energy_list   = [track.get_energy(l) for l in length_list]
                probabilities = self.LLPEstimator.calc_LLP_probability(length_list, energy_list)
        else:
            probabilities = np.zeros(len(self.LLPEstimator.get_LLPModels())) # if muon bundle, return zero prob
            
        # write probabilities
        # @TODO: use I3Map<string, double> for this, and use unique identifier string
        frame["LLPProbabilities"] = probabilities


    def Finish(self):
        pass
