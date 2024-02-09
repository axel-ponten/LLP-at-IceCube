from I3LLPProbabilityCalculator import *
from LLPEstimator import *
from estimation_utilities import *


# @TODO: implement the LLP calculator on a corsika spectrum
tray = I3Tray()

tray.Add("I3Reader")
tray.Add(I3LLPProbabilityCalculator)
tray.Add("I3Writer")