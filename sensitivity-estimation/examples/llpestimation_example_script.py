"""
Simple example application of llpestimation package.
"""
import sys
sys.path.append("..")

from llpestimation import LLPModel, LLPEstimator, LLPMedium, LLPProductionCrossSection
from estimation_utilities import *
import numpy as np

# create some muons
def generate_atmospheric_muons(n = 10):
    steps = 50
    muons = []
    for _ in range(n):
        length = float(np.random.randint(100,1000)) # random length in detector in meters
        energy = float(np.random.randint(500,2000)) # random energy of muon in GeV
        energy_list = np.linspace(energy, energy*0.5, steps) # unrealistic but whatever
        length_list = np.linspace(0, length, steps) # from 0 to end of detector
        muons.append( (length_list, energy_list) )
    return muons

muons = generate_atmospheric_muons(10)

# create some dark leptonic scalar models (a type of LLP)
masses      = [0.107, 0.110, 0.115, 0.13]
epsilons    = [5e-6, 5e-6, 5e-6, 5e-6]
names       = ["DarkLeptonicScalar" for m in masses]
table_paths = generate_DLS_WW_oxygen_paths(masses, "../cross_section_tables/")
models = generate_DLSModels(masses, epsilons, names, table_paths)

# create LLPEstimator
min_gap = 50.0 # if distance production to decay is too small, not detectable
dls_est = LLPEstimator(models, min_gap) # dark leptonic scalar estimator

# compute probabilities for the muons
probabilities = [dls_est.calc_llp_probability_with_id(muon[0], muon[1]) for muon in muons]
for p, m in zip(probabilities, muons):
    print("Muon length [m] and energy [GeV]:", m[0][-1], m[1][0])
    print("Probabilities:", p)
