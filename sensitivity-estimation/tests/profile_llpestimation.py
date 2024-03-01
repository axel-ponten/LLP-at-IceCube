"""
Profile the llpestimator to see where the bottlenecks are.
"""

import sys
sys.path.append("..")

from llpestimation import LLPModel, LLPEstimator, LLPMedium, LLPProductionCrossSection
from estimation_utilities import *

import timeit
import cProfile
import pstats

masses      = np.arange(0.107, 0.15, 0.001)
# epsilons    = [5e-6, 5e-6, 5e-6, 5e-6, 5e-6, 5e-6]
epsilons    = [1e-5 for m in masses]
names       = ["DarkLeptonicScalar" for m in masses]
table_paths = generate_DLS_WW_oxygen_paths(masses, folder = "../cross_section_tables/")
models = generate_DLSModels(masses, epsilons, names, table_paths)
min_gap = 50.0
est = LLPEstimator(models, min_gap)


# 1 TeV muon losing 300 GeV over 800 m (made up, not following actual energy loss formula)
steps = 100
length_list = np.linspace(0,800,steps)
energy_list = np.linspace(1000,700,steps)

# cProfile for the calculations
print("####### cProfile for probability calculation #######")
with cProfile.Profile() as profile:
    for i in range(1000):
        est.calc_llp_probability(length_list, energy_list)
profile_result = pstats.Stats(profile)
profile_result.sort_stats(pstats.SortKey.TIME)
profile_result.print_stats()
