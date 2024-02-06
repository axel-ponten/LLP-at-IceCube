from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from collections.abc import Callable

class LLPModel():
    """
    Class to hold LLP model parameters and production cross section function used in LLP estimation.
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

    def get_lifetime(self, gamma: float = 1.0) -> float:
        """
        Lifetime of the LLP model. Can give gamma to get time dilated lifetime.
        :param gamma: lorentz boost of the LLP. often given by E/m.
        :return float: lifetime of the LLP. if no gamma given, then in rest frame.
        """
        return self.tau * gamma

    def decay_factor(self, l1: float, l2: float, energy: float) -> float:
        """
        What is probability to decay between lengths l1 and l2? Integrate exponential pdf between l1 and l2.
        :param l1: Minimum length before decay in cm. Should be same as minimum detectable gap length.
        :param l2: Maximum length before decay in cm.
        :param energy: Energy of the LLP.
        :return float: Between 0-1. Fraction of the decay pdf within length l1 and l2.
        """
        c_gamma_tau = 29979245800.0 * self.get_lifetime(energy/self.mass) # c [cm/s] * gamma * tau
        prob = np.exp(-l1/c_gamma_tau) - np.exp(-l2/c_gamma_tau) # \int^l1_l2 1/c*gamma*tau*exp(-l/c*gamma*tau)
        # check for negative values
        if type(prob) is np.ndarray:
            prob[prob < 0] = 0.0 # if l2 is shorter than l1, return 0
        else:
            if prob < 0.0:
                prob = 0.0
        return prob

    def print_summary(self):
        """
        For testing purposes.
        """
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
    Input expected in meters and GeV. Internally computes with centimeters.
    """
    def __init__(self, LLPModels: list, min_gap_meters: float = 50.0):
        self.min_gap = min_gap_meters*100.0     # what's the shortest detectable LLP gap, in meters. convert to cm
        self.LLPModels = LLPModels # make sure this stay ordered
        self.LLP_funcs = [(m.calc_tot_xsec, m.decay_factor) for m in self.LLPModels]
        
    def calc_LLP_probability(self, length_list: list, energy_list: list) -> list:
        """
        Computes the total detectable LLP probability for a muon track for all models in the LLPEstimator. Convolution of segmented thick target approximation and decay factor.
        :param length_list: List of lengths from 0 to end of detector in meters, already trimmed for entry/exit margins. Last element should be total length.
        :param energy_list: List of energies of the muon from detector entry to exit in GeV. Ordered with length_list.
        :return list: Returns a list of detectable LLP probabilities. Ordered with list of LLPModels.
        """
        if len(length_list) != len(energy_list):
            raise ValueError("length_list and energy_list must contain same number of elements")
            return None
        if min(energy_list) < 0.0:
            raise ValueError("Negative energies not allowed. If muon stopped, give a length and energy list that stops at stopping point.")
            return None

        # @TODO: come up with clever way using numpy functionality to compute the probaiblities faster
        
        # numpify input
        length_array = np.asarray(length_list)*100.0 # convert to cm
        energy_array = np.asarray(energy_list)       # GeV
        track_length = length_array[-1]              # length entry to exit of detector (already trimmed for exit and entry margins) in cm

        # parameters for calculations                         
        l2_array = track_length - length_array - self.min_gap # from production point to furthest available decay point
        delta_L = np.append(np.diff(length_array), 0)         # step length, in cm
        n_oxygen = 6.02214076e23 * 0.92 / 18                  # number density of oxygen in ice
        n_hydrogen = 2*n_oxygen                               # number density of hydrogen in ice
        
        # @TODO: also include hydrogen contribution
        # compute probability = sum( delta_L * decay_factor * number_density * tot_xsec )
        matrix_for_calc = np.row_stack([decay(self.min_gap, l2_array, energy_array)*n_oxygen*xsec(energy_array)
                                             for xsec, decay in self.LLP_funcs]) # 2D matrix with rows corresponding to each model
        probabilities = matrix_for_calc @ delta_L # NxM*Mx1 where N is # of models and M is # of length steps
        return probabilities

    def get_LLPModels(self) -> list:
        return self.LLPModels


        
    