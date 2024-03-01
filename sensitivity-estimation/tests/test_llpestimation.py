
import sys
sys.path.append("..")

from llpestimation import LLPModel, LLPEstimator, LLPMedium, LLPProductionCrossSection
from estimation_utilities import *
import os
import numpy as np
import pandas as pd
import pytest

def create_test_DLS_model():
    # parameters for test
    mass = 0.115
    eps = 5e-6
    name = "DarkLeptonicScalar"
    tau = calculate_DLS_lifetime(mass, eps)
    path_to_table = "../cross_section_tables/totcs_WW_m_"+"{:.3f}".format(mass) + ".csv"

    # CREATE LLPProductionCrossSection
    # read in table
    df = pd.read_csv(path_to_table, names=["E0", "totcs"])
    func_to_xsec = interp1d(df["E0"], eps**2*df["totcs"],kind="linear", bounds_error=False,fill_value=(0.0, None))
    n_oxygen = 6.02214076e23 * 0.92 / 18 # number density of oxygen in ice
    oxygen   = LLPMedium("O", n_oxygen, 8, 16)
    llp_xsec = LLPProductionCrossSection([func_to_xsec], [oxygen])

    # create LLPModel
    DLS = LLPModel(name, mass, eps, tau, llp_xsec)
    return DLS

############## TEST LLPMedium ##############
def test_llpmedium():
    n_oxygen = 6.02214076e23 * 0.92 / 18 # number density of oxygen in ice
    n_hydrogen = 2*n_oxygen              # number density of hydrogen in ice

    oxygen   = LLPMedium("O", n_oxygen, 8, 16)
    hydrogen = LLPMedium("H", n_hydrogen, 1, 1)

    assert n_oxygen == oxygen.number_density
    assert n_hydrogen == hydrogen.number_density
############## END LLPMedium ##############

############## TEST LLPProductionCrossSection ##############
def test_llpxsec():
    # parameters for test
    mass = 0.115
    eps = 5e-6
    path_to_table = "../cross_section_tables/totcs_WW_m_"+"{:.3f}".format(mass) + ".csv"
    n_oxygen = 6.02214076e23 * 0.92 / 18 # number density of oxygen in ice
    oxygen   = LLPMedium("O", n_oxygen, 8, 16)

    # CREATE LLPProductionCrossSection
    # read in table
    df = pd.read_csv(path_to_table, names=["E0", "totcs"])
    func_to_xsec = interp1d(df["E0"], eps**2*df["totcs"],kind="linear", bounds_error=False,fill_value=(0.0, None))
    llp_xsec = LLPProductionCrossSection([func_to_xsec], [oxygen])

    # create plot to show interpolation works
    plot_interpolation(df, func_to_xsec, mass, eps)

    # check that function acutally works
    xsec_error = (llp_xsec.func_tot_xsec_list[0](df["E0"]) - eps**2*df["totcs"]) / eps**2*df["totcs"]
    tolerance = 0.001 # 0.1%
    assert np.all(np.absolute(xsec_error) < tolerance)
    assert abs(llp_xsec.interactions_per_cm(700.0) - func_to_xsec(700)*n_oxygen) < 1e-15

    print("interactions per cm at 700. GeV", llp_xsec.interactions_per_cm(700.0))
    print("interactions per cm at 100000. GeV", llp_xsec.interactions_per_cm(100000.0))
    print("interactions per cm at 1e7 GeV", llp_xsec.interactions_per_cm(1e7))
    print("interactions per cm at 5. GeV", llp_xsec.interactions_per_cm(5.0))
    print("interactions per cm at 0. GeV", llp_xsec.interactions_per_cm(0.0))
############## END LLPProductionCrossSection ##############


############## TEST LLPModel ##############
def test_llpmodel_creation():
    # parameters for test
    mass = 0.115
    eps = 5e-6
    name = "DarkLeptonicScalar"
    tau = calculate_DLS_lifetime(mass, eps)
    path_to_table = "../cross_section_tables/totcs_WW_m_"+"{:.3f}".format(mass) + ".csv"

    # CREATE LLPProductionCrossSection
    # read in table
    df = pd.read_csv(path_to_table, names=["E0", "totcs"])
    func_to_xsec = interp1d(df["E0"], eps**2*df["totcs"],kind="linear", bounds_error=False,fill_value=(0.0, None))
    n_oxygen = 6.02214076e23 * 0.92 / 18 # number density of oxygen in ice
    oxygen   = LLPMedium("O", n_oxygen, 8, 16)
    llp_xsec = LLPProductionCrossSection([func_to_xsec], [oxygen])

    # create LLPModel
    DLS = LLPModel(name, mass, eps, tau, llp_xsec)

    assert DLS.mass == mass
    assert DLS.name == name
    assert DLS.eps == eps
    assert DLS.tau == tau
    assert DLS.get_lifetime(1) == tau
    assert DLS.llp_xsec == llp_xsec

def test_llpmodel_decay_factor():
    DLS = create_test_DLS_model()
    assert (DLS.decay_factor(50,800,1000) > 0 and DLS.decay_factor(50,800,1000) <= 1)
    assert DLS.decay_factor(50,50,1000) == 0
    assert DLS.decay_factor(0,np.inf,1000) == 1
    assert DLS.decay_factor(50,10,1000) == 0 # l1 <= l2
    assert DLS.decay_factor(-10,-20,1000) == 0 # no negatives
    assert DLS.decay_factor(-60,20,1000) == 0 # no negatives
    assert DLS.decay_factor(-60,100,1000) == 0 # no negatives

def test_llpmodel_interactions_per_cm():
    DLS = create_test_DLS_model()
    assert DLS.interactions_per_cm(-1) == 0
    assert DLS.interactions_per_cm(0) == 0
    assert DLS.interactions_per_cm(50) < DLS.interactions_per_cm(100)
    assert DLS.interactions_per_cm(1e4) > 0

def test_llpmodel_lifetime():
    DLS = create_test_DLS_model()
    assert DLS.get_lifetime(10) > DLS.get_lifetime(2)
    with pytest.raises(ValueError):
        DLS.get_lifetime(0.1)
    with pytest.raises(ValueError):
        DLS.get_lifetime(-5)
############## END LLPModel ##############

############## TEST LLPEstimator ##############
def create_estimator():
    masses      = [0.107, 0.110, 0.115, 0.13]
    epsilons    = [5e-6, 5e-6, 5e-6, 5e-6]
    names       = ["DarkLeptonicScalar" for m in masses]
    table_paths = generate_DLS_WW_oxygen_paths(masses, folder = "../cross_section_tables/")
    models = generate_DLSModels(masses, epsilons, names, table_paths)
    min_gap = 50.0
    est = LLPEstimator(models, min_gap)
    return est

def test_llpestimator():
    est = create_estimator()

    # check that interactions per cm works
    energy = 500.0
    model_funcs = est.llp_funcs
    probabilities = [interactions(500.0) for interactions, decayfactors in model_funcs]
    print(probabilities)
    assert np.all([probabilities[i] > probabilities[i+1] for i in range(len(probabilities) - 1)]) 

    steps = 100
    length_list = np.linspace(0,800,steps)
    energy_list = np.linspace(1000,700,steps)

    # check all probabilties 
    probabilities = est.calc_llp_probability(length_list, energy_list)
    probabilities_map = est.calc_llp_probability_with_id(length_list, energy_list)
    print(probabilities)
    print(probabilities_map)
    assert np.all(probabilities >= 0.0)

############## END LLPEstimator ##############


