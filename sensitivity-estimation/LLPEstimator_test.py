from LLPEstimator import *
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def initialize_cross_section_interpolation(path_to_table: str):
    # read in from tables
    df = pd.read_csv(path_to_table, names=["E0", "totcs"])
    # create interpolation
    return interp1d(df["E0"], df["totcs"],kind="quadratic")

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

############## TEST LLPModel ##############
# parameters for test
name = "DarkLeptonicScalar"
mass = 0.115
eps = 5e-6
tau = calculate_DLS_lifetime(mass, eps)
path_to_table = os.getcwd() + "/cross_section_tables/totcs_WW_m_"+"{:.3f}".format(mass) + ".csv"
print("Parameters:", name, mass, eps, tau, path_to_table)

# read in table
df = pd.read_csv(path_to_table, names=["E0", "totcs"])
unscaled_func = initialize_cross_section_interpolation(path_to_table)
func_to_xsec = lambda energy: eps**2 * unscaled_func(energy)

# create LLPModel
DLS = LLPModel(name, mass, eps, tau, func_to_xsec)
DLS.print_summary()

# print lifetimes at different energies
print("Lifetime at 10 GeV", DLS.get_lifetime(10/mass))
print("at 100 GeV", DLS.get_lifetime(100/mass))
print("at 1000 GeV", DLS.get_lifetime(1000/mass))
# plot
plot_interpolation(df, DLS.func_tot_xsec, mass, eps)
############## END ##############
