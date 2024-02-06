from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from collections.abc import Callable

class LLPModel():
    """
    Class to hold LLP model parameters and production cross section function.
    """
    def __init__(self, name: str, mass: float, eps: float, tau: float, func_tot_xsec: Callable[[float], float]):
        self.name          = name          # such as DarkLeptonicScalar, etc.
        self.mass          = mass          # in GeV
        self.eps           = eps           # coupling to SM
        self.tau           = tau           # lifetime in s
        self.func_tot_xsec = func_tot_xsec # should return totcs in cm^2, perhaps a interpolation function from a table
        
    def calc_tot_xsec(self, energy: float) -> float:
        """
        Total production cross section in cm^2 per nucleus.
        :param energy: Energy of muon in GeV.
        :return float: Total cross section in cm^2.
        """
        # @TODO: include both hydrogen and oxygen
        return self.func_tot_xsec(energy)

    def get_lifetime(self, gamma = 1.0: float) -> float:
        """
        Lifetime of the LLP model. Can give gamma to get time dilated lifetime.
        :param gamma: lorentz boost of the LLP. often given by E/m.
        :return float: lifetime of the LLP. if no gamma given, then in rest frame.
        """
        return self.tau * gamma

    def decay_factor(self, l1: float, l2: float, energy: float) -> float:
        """
        What is probability to decay between l1 and l2 for an LLP with some energy? Integrate an exponential pdf between l1 and l2.
        :param l1: minimum length before decay. should be same as minimum detectable gap length.
        :param l2: maximum length before decay.
        :param energy: energy of the LLP.
        :return float: between 0-1. the fraction of the decay pdf within length l1 and l2.
        """
        if l1 >= l2:
            # if production point is closer to boundary than minimum gap length, decay factor is zero
            return 0.0
        c_gamma_tau = 299792458.0 * get_lifetime(energy/self.mass) # c * gamma * tau
        prob = np.exp(-l1/c_gamma_tau) - np.exp(-l2/c_gamma_tau) # exp(-l/c*gamma*tau)|^l1_l2
        if prob < 0 or prob > 1:
            raise ValueError("LLP decay probability not bounded between 0 and 1.")
        return prob

    def print_summary(self):
        print("Parameters of model", self.name, ":")
        print("mass", self.mass)
        print("eps", self.eps)
        print("tau", self.tau)
        print("func_tot_xsec", self.func_tot_xsec)
        print("Test func_tot_xsec at 500 GeV", self.calc_tot_xsec(500.0))


class LLPEstimator():
    """
    Class to calculate detectable LLP probability for a list of LLPModels.
    Calculates detectable LLP probability for each model given a muon track (list of ordered length steps and energies).
    """
    def __init__(self, LLPModels: list, min_gap = 50.0: float):
        self.min_gap = min_gap     # what's the shortest detectable LLP gap, in meters
        self.LLPModels = LLPModels # make sure this stay ordered
        self.LLP_xsec_funcs = np.array([m.calc_tot_xsec for m in self.LLPModels])
        
    def calc_LLP_probability(self, length_list: list, energy_list: list) -> list:
        """
        Computes the total detectable LLP probability for a muon track for all models in the LLPEstimator. Convolution of segmented thick target approximation and decay factor.
        :param length_list: List of lengths from 0 (entering detector) to end of detector in meters. Last element should be total length. Should already be trimmed for entry and exit margins.
        :param energy_list: List of energies of the muon from detector entry to exit in GeV. Ordered with length_list.
        :return list: Returns a list of detectable LLP probabilities. Ordered with list of LLPModels.
        """
        if len(length_list) != len(energy_list):
            raise ValueError("length_list and energy_list must contain same number of elements")
            return None
        if min(energy_list) < 0.0:
            raise ValueError("Negative energies not allowed. If muon stopped, give a length and energy list that stops at stopping point.")
            return None

        # @TODO: come up with a more clever way to use numpy functionality to compute the probaiblities for all models as fast as possible

        # numpify input
        length_array = np.asarray(length_list)*100.0 # convert meter to cm
        energy_array = np.asarray(energy_list)       # GeV
        n_steps = len(length_array)

        # parameters for calculations
        track_length = length_list[-1]                                                           # length from entry to exit of detector (already trimmed for exit and entry margins) in cm
        l2_array = np.asarray([track_length - l - self.min_gap for l in length_array])           # from production point to furthest available decay point
        delta_L = np.asarray([length_array[i+1] - length_array[i] for i in range(n_steps) - 1)]) # step length, in cm
        n_oxygen = 6.02214076e23 * 0.92 / 18                                                     # number density of oxygen in ice
        n_hydrogen = 2*n_oxygen                                                                  # number density of hydrogen in ice

        # calculate decay factors and total cross sections
        matrix_decay_factors = np.row_stack([decay_factor(self.min_gap, l2_array, energy_array)])     # 2D matrix with rows corresponding to each model
        matrix_tot_xsec = np.col_stack([n_oxygen*func(energy_array) for func in self.LLP_xsec_funcs]) # 2D matrix with cols corresponding to each model

        # compute probability = sum( delta_L * decay_factor * tot_xsec * number_density )
        probabilities = matrix_decay_factors @ matrix_tot_xsec @ delta_L # NxM*MxN*Nx1 where N is # of models and M is # of length steps
        return probabilities

    def get_LLPModels(self) -> list:
        return self.LLPModels

########## Helper functions ##########
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
        func_tot_xsec_unscaled = interp1d(df["E0"], df["totcs"],kind="quadratic")
        func_tot_xsec          = lambda energy : eps**2 * func_tot_xsec_unscaled(energy)
        # create new LLPModel
        LLPModel_list.append(LLPModel(name, mass, eps, tau, func_tot_xsec))
    return LLPModel_list

def generate_DLS_WW_oxygen_paths(masses):
    folder = "/data/user/axelpo/LLP-at-IceCube/sensitivity-estimation/cross_section_tables/"
    paths  = [folder+"totcs_WW_m_{:.3f}.csv".format(m) for m in masses]
    return paths
        
        
    