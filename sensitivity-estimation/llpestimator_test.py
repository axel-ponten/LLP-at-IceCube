from llpestimator import *
from estimation_utilities import *
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import timeit
import cProfile
import pstats

def plot_interpolation(df, interpfunc, mass, eps=1):
    E0array = np.logspace(1,5,1000)
    totcsarray = [interpfunc(energy) for energy in E0array]
    # plot
    plt.figure()
    plt.plot(df["E0"],eps**2*df["totcs"],'k+', label="table entries")
    plt.plot(E0array,totcsarray,'b',label="interpolaton")
    plt.ylabel('$\sigma \; [cm^2]$')
    plt.xlabel('$E_0 \; [GeV]$')
    plt.legend()
    plt.grid()
    plt.xscale("log")
    plt.xlim([10,10000])
    plt.title(mass)
    plt.savefig("LLPEstimator_test_plots/1D_interpolation_"+"{:.3f}".format(mass)+".png")
    plt.show()


############## TEST LLPMedium ##############
print("############## TEST LLPMedium ##############")
n_oxygen = 6.02214076e23 * 0.92 / 18 # number density of oxygen in ice
n_hydrogen = 2*n_oxygen              # number density of hydrogen in ice

oxygen   = LLPMedium("O", n_oxygen, 8, 16)
hydrogen = LLPMedium("H", n_hydrogen, 1, 1)

print("n_oxygen:", oxygen.number_density)
print("n_hydrogen:", hydrogen.number_density)
############## END LLPMedium ##############

############## TEST LLPProductionCrossSection and LLPModel ##############
# @TODO: include hydrogen
print("############## TEST LLPProductionCrossSection ##############")
# parameters for test
name = "DarkLeptonicScalar"
mass = 0.115
eps = 5e-6
tau = calculate_DLS_lifetime(mass, eps)
path_to_table = os.getcwd() + "/cross_section_tables/totcs_WW_m_"+"{:.3f}".format(mass) + ".csv"
print("Parameters:", name, mass, eps, tau, path_to_table)

# CREATE LLPProductionCrossSection
# read in table
df = pd.read_csv(path_to_table, names=["E0", "totcs"])
func_to_xsec = interp1d(df["E0"], eps**2*df["totcs"],kind="quadratic", bounds_error=False,fill_value=(0.0, None))
llp_xsec = LLPProductionCrossSection([func_to_xsec], [oxygen])

print("interactions per cm only using oxygen at 700. GeV", llp_xsec.interactions_per_cm(700.0))
print("interactions per cm only using oxygen at 100000. GeV", llp_xsec.interactions_per_cm(100000.0))
print("interactions per cm only using oxygen at 1e7 GeV", llp_xsec.interactions_per_cm(1e7))
print("interactions per cm only using oxygen at 5. GeV", llp_xsec.interactions_per_cm(5.0))
print("interactions per cm only using oxygen at 0. GeV", llp_xsec.interactions_per_cm(0.0))

# create LLPModel
DLS = LLPModel(name, mass, eps, tau, llp_xsec)
DLS.print_summary()

# print lifetimes at different energies
print("Lifetime at 10 GeV", DLS.get_lifetime(10/mass))
print("at 100 GeV", DLS.get_lifetime(100/mass))
print("at 1000 GeV", DLS.get_lifetime(1000/mass))
# plot
plot_interpolation(df, DLS.llp_xsec.func_tot_xsec_list[0], mass, eps)

# test decay factor
print("test decay factor at E = 500 GeV for:")
print("l1=0, l2=999999999", DLS.decay_factor(0,999999999,500))
print("l1=50, l2=800", DLS.decay_factor(50,800,500))
print("l1=50, l2=75", DLS.decay_factor(50,75,500))
print("l1=50, l2=10", DLS.decay_factor(50,10,500))
print("l1=50, l2=-10", DLS.decay_factor(50,-10,500))

############## END ##############

############## TEST LLPEstimator ##############
# @TODO: fix for new LLPModel structure
print("\n\nTesting LLPEstimator")

masses      = [0.107, 0.110, 0.115, 0.13]
epsilons    = [5e-6, 5e-6, 5e-6, 5e-6]
names       = ["DarkLeptonicScalar" for m in masses]
table_paths = generate_DLS_WW_oxygen_paths(masses)
print("paths", table_paths)

models = generate_DLSModels(masses, epsilons, names, table_paths)
min_gap = 50.0
# test ID's
print("\n\n\nID's for created models")
uniqueID_list = [m.unique_id for m in models]
print(uniqueID_list)
print("\n creating new models from the ID's")
model_from_ID_list = [LLPModel.from_unique_id(ID) for ID in uniqueID_list]
[m.print_summary() for m in model_from_ID_list]
print("\n\n\n")

my_LLPEstimator = LLPEstimator(models, min_gap)
print("Models used:")
[m.print_summary() for m in my_LLPEstimator.llpmodels()]
print("\n\n")

steps = 100
length_list = np.linspace(0,800,steps)
energy_list = np.linspace(1000,700,steps)
print("Lengths and energies")
print(length_list)
print(energy_list)
print("Probabilites")
probabilities = my_LLPEstimator.calc_llp_probability(length_list, energy_list)
print(probabilities)

# @TODO: check that its ordered with llpmodels

# cProfile for the calculations
print("####### cProfile for probability calculation #######")
with cProfile.Profile() as profile:
    for i in range(1000):
        my_LLPEstimator.calc_llp_probability(length_list, energy_list)
profile_result = pstats.Stats(profile)
profile_result.sort_stats(pstats.SortKey.TIME)
profile_result.print_stats()


