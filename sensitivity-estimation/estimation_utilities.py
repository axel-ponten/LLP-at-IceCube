from LLPEstimator import *
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
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
    LLPModel_list = []
    for mass, eps, name, path in zip(masses, epsilons, names, table_paths):
        # lifetime
        tau = calculate_DLS_lifetime(mass, eps)
        # tot_xsec function from interpolation tables
        df = pd.read_csv(path, names=["E0", "totcs"])
        func_tot_xsec = interp1d(df["E0"], eps**2*df["totcs"],kind="quadratic")
        # create new LLPModel
        LLPModel_list.append(LLPModel(name, mass, eps, tau, func_tot_xsec))
    return LLPModel_list

def generate_DLS_WW_oxygen_paths(masses):
    folder = "/data/user/axelpo/LLP-at-IceCube/sensitivity-estimation/cross_section_tables/"
    paths  = []
    for m in masses:
        m_str = "{:.3f}".format(m)
        if m_str[-1] == "0":
            m_str = m_str[:-1]
        paths.append(folder+"totcs_WW_m_"+m_str+".csv")
    return paths
        