"""
Collection of utility functions to implement llpestimation package,
such as creation of dark leptonic scalar models (a type of LLP),
or to generate interpolation functions from the tables.
"""

from llpestimation import LLPModel, LLPEstimator, LLPMedium, LLPProductionCrossSection
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections.abc import Callable

########## Helper functions for DLS ##########
def calculate_DLS_lifetime(mass, eps):
    """ lifetime at first order of Dark Leptonic Scalar. two body decay into e+mu """
    m_e = 0.00051099895000
    m_mu = 0.1056583755
    p1 = np.sqrt( (mass**2 - (m_e + m_mu)**2) * (mass**2 - (m_e - m_mu)**2) ) / (2 * mass) # momentum of electron
    width = eps**2 / (8 * np.pi) * p1 * (1 - (m_e**2 + m_mu**2) / mass**2)
    GeV_to_s = 6.582e-25
    return GeV_to_s * 1 / width

def generate_DLSModels(masses, epsilons, names, table_paths):
    # @TODO: fix for hydrogen
    llpmodel_list = []
    oxygen = get_ice_oxygen()
    for mass, eps, name, path in zip(masses, epsilons, names, table_paths):
        # lifetime
        tau = calculate_DLS_lifetime(mass, eps)
        # tot_xsec function from interpolation tables
        df = pd.read_csv(path, names=["E0", "totcs"])
        func_tot_xsec = interp1d(df["E0"], eps**2*df["totcs"],kind="linear")
        # create LLPProductionCrossSection
        llp_xsec = LLPProductionCrossSection([func_tot_xsec], [oxygen])
        # create new LLPModel
        llpmodel_list.append(LLPModel(name, mass, eps, tau, llp_xsec))
    return llpmodel_list

def generate_DLS_WW_oxygen_paths(masses, folder = None):
    #folder = "/data/user/axelpo/LLP-at-IceCube/sensitivity-estimation/cross_section_tables/"
    import os
    if folder is None:
        folder = os.getcwd() + "/cross_section_tables/"
    paths  = []
    for m in masses:
        m_str = "{:.3f}".format(m)
        if m_str[-1] == "0":
            m_str = m_str[:-1]
        paths.append(folder+"totcs_WW_m_"+m_str+".csv")
    return paths

########## Create south pole ice ##########
def south_pole_ice():
    """
    List of LLPMedium corresponding to IceCube ice.
    """
    n_oxygen = 6.02214076e23 * 0.92 / 18 # number density of oxygen in ice
    n_hydrogen = 2*n_oxygen              # number density of hydrogen in ice
    oxygen  = LLPMedium("O", n_oxygen, 8, 16)
    hydrogen = LLPMedium("H", n_hydrogen, 1, 1)
    return [oxygen, hydrogen]

def get_ice_oxygen():
    n_oxygen = 6.02214076e23 * 0.92 / 18 # number density of oxygen in ice
    oxygen  = LLPMedium("O", n_oxygen, 8, 16)
    return oxygen

########## Plotting ##########
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
    plt.savefig("1D_interpolation_"+"{:.3f}".format(mass)+".png")
    #plt.show()

