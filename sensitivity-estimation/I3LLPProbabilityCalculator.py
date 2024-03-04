import icecube
from icecube import icetray, MuonGun, dataclasses, dataio
import numpy as np
from llpestimation import LLPMedium, LLPProductionCrossSection, LLPModel, LLPEstimator
from estimation_utilities import *

"""
Iterate through a atmospheric muon spectrum and compute event-by-event detectable LLP probability.
"""

#Function to read the GCD file and make the extruded polygon which
#defines the edge of the in-ice array
def MakeSurface(gcdName, padding):
    file = dataio.I3File(gcdName, "r")
    frame = file.pop_frame()
    while not "I3Geometry" in frame:
        frame = file.pop_frame()
    geometry = frame["I3Geometry"]
    xyList = []
    zmax = -1e100
    zmin = 1e100
    step = int(len(geometry.omgeo.keys())/10)
    print("Loading the DOM locations from the GCD file")
    for i, key in enumerate(geometry.omgeo.keys()):
        if i % step == 0:
            print( "{0}/{1} = {2}%".format(i,len(geometry.omgeo.keys()), int(round(i/len(geometry.omgeo.keys())*100))))
            
        if key.om in [61, 62, 63, 64] and key.string <= 81: #Remove IT...
            continue

        pos = geometry.omgeo[key].position

        if pos.z > 1500:
            continue

        xyList.append(pos)
        i+=1
    
    return MuonGun.ExtrudedPolygon(xyList, padding)

class I3LLPProbabilityCalculator(icetray.I3Module):
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

        self.padding = 0.0
        self.AddParameter("padding", "padding of gcd boundary.", self.padding)

        self.LLPEstimator = None
        self.AddParameter("llp_estimator", "LLPEstimator object.", self.LLPEstimator)

    def Configure(self):
        self.min_gap      = self.GetParameter("min_gap")
        self.entry_margin = self.GetParameter("entry_margin")
        self.exit_margin  = self.GetParameter("exit_margin")
        self.n_steps      = self.GetParameter("n_steps")
        self.LLPEstimator = self.GetParameter("llp_estimator")
        self.padding      = self.GetParameter("padding")
        # create surface for detector volume
        self.gcdFile = self.GetParameter("GCDFile")
        if self.gcdFile != "":
            self.surface = MakeSurface(self.gcdFile, self.padding)
        else:
            print("No GCD file provided, using 1000x500 MuonGun cylinder instead.")
            self.surface = MuonGun.Cylinder(1000,500) # approximate detector volume

        # predefined list of tuples for events with probabilities that are 0
        self._zero_prob_map = dataclasses.I3MapStringDouble(
            {ID: 0 for ID in self.LLPEstimator.llpmodel_unique_ids}
        )

    def DAQ(self, frame):
        """ Compute total detectable LLP probability for the event using the
        llpestimation package.
        
        Computes probability separately for all models in the LLPEstimator 
        and stores the result as I3MapStringDouble of LLP model ID to probability.

        Detectable events have production and decay vertex inside detector volume,
        and sufficiently long decay gap. Computed through convolution of segmented thin target
        approximation convolved with decay factor (partially integrated decay pdf).

        Only single muons have detectable gaps, so bundles get 0 probability.
        """
        # @TODO: what if muon stops before exiting detector? maybe its fine cus track.get_energy gives zero?
        track_list = MuonGun.Track.harvest(frame['SignalI3MCTree'], frame['MMCTrackList'])
        # only single muons are detectable LLP candidates
        if self.check_single_muon(track_list):
            track = track_list[0]
            ID_probability_map = self.calc_LLP_probability_from_muon(track) # llp prob for single muon
        else:
            ID_probability_map = self._zero_prob_map # if muon bundle, return zero prob
        # write I3MapStringDouble to frame
        frame["LLPProbabilities"] = ID_probability_map

        self.PushFrame(frame)

    def check_single_muon(self, track_list):
        """
        Check that there is one and only one muon in the event.
        """
        counter = 0
        for track in track_list:
            if not track.is_neutrino:
                counter += 1
        if counter == 1:
            return True
        else:
            return False
        
    def calc_LLP_probability_from_muon(self, track) -> dict:
        """ Computes the detectable LLP probability of a given single muon.
        Creates a list of lengths and muon energies inside the detector that
        represents the muon track. Pass this to LLPEstimator.
        """
        # Find distance to entrance and exit from sampling volume
        intersections = self.surface.intersection(track.pos, track.dir)
        # trim for margins
        start_length = intersections.first + self.entry_margin
        stop_length  = intersections.second - self.exit_margin
        # check for available space
        if start_length - stop_length <= self.min_gap:
            ID_probability_map = self._zero_prob_map # if no available space, return zero prob
        else:
            # create ordered length and energy lists for LLPEstimator
            length_list   = np.linspace(start_length, stop_length, self.n_steps)
            energy_list   = [track.get_energy(l) for l in length_list]
            print(energy_list)
            ID_probability_map = dataclasses.I3MapStringDouble(
                self.LLPEstimator.calc_LLP_probability(length_list, energy_list)
            ) # calculate
        return ID_probability_map

    def Finish(self):
        pass
