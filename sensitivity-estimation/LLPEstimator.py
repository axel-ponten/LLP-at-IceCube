import numpy as np
from collections.abc import Callable

class LLPMedium():
    def __init__(self, name: str, number_density: float, Z: int, A: int):
        self.name           = name           # e.g. "oxygen" or "hydrogen"
        self.number_density = number_density # per cm^3
        self.Z              = Z              # atomic number
        self.A              = A              # mass number
        
class LLPProductionCrossSection():
    def __init__(self, func_tot_xsec_list: list, medium_list: list):
        self.func_tot_xsec_list = func_tot_xsec_list # tot xsec funcs in cm^2, input energy in units of GeV. type Callable[[float], float]
        self.medium_list        = medium_list        # list of LLPMedium, ordered with func_tot_xsec_list

    def interactions_per_cm(self, energy: float) -> float:
        """
        Total cross section weighted with number density for all elements in medium.
        Different elements have different production cross section and densities so need to consider separately.
        :param energy: Energy of the muon in GeV.
        :return float: Total xsec times num density, units of cm^-1.
        """
        # @TODO: is it unnecessary for each grid point to run m.number_density or is it so small it doesn't matter?
        return sum([xsec(energy)*m.number_density for xsec, m in zip(self.func_tot_xsec_list, self.medium_list)])

class LLPModel():
    """
    Class to hold LLP model parameters and production cross section function used in LLP estimation.
    """
    def __init__(self, name: str, mass: float, eps: float, tau: float, llp_production_xsec: LLPProductionCrossSection):
        self.name                 = name                 # such as DarkLeptonicScalar, etc.
        self.mass                 = mass                 # in GeV
        self.eps                  = eps                  # coupling to SM
        self.tau                  = tau                  # lifetime in s
        self.llp_production_xsec  = llp_production_xsec  # object of LLPProductionCrossSection class
        self.unique_id            = self.get_unique_id() # string with all model information besides xsec function

    def interactions_per_cm(self, energy: float) -> float:
        """
        Total cross section weighted with number density for all elements in medium.
        :param energy: Energy of muon in GeV.
        :return float: Interactions per cm.
        """
        return self.llp_production_xsec.interactions_per_cm(energy)

    def get_lifetime(self, gamma: float = 1.0) -> float:
        """
        Lifetime of the LLP model in lab frame for some gamma.
        :param gamma: Lorentz boost of the LLP. Given by E/m.
        :return float: Lifetime of the LLP in lab frame.
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

    def get_unique_id(self) -> str:
        # @TODO: clever way to compute ID for a model. should contain mass, epsilon, name, lifetime info
        # @TODO: how to include the cross section function?
        """
        Encodes the model in a underscore separated string.
        Used to reconstruct the LLPModel (except cross section function)."
        """
        parameters_str = [self.name, str(self.mass), str(self.eps), str(self.tau)]
        unique_id = "_".join(parameters_str)
        return unique_id

    @classmethod
    def from_unique_id(cls, unique_id: str):
        # @TODO: how to deal with cross section function? don't want to include path to table here, ruins agnosticism of xsec origin
        """
        Returns a new LLPModel object from a unique id.
        """
        parameters_str = unique_id.split("_")
        placeholder_function = lambda x : None
        return cls(parameters_str[0], float(parameters_str[1]), float(parameters_str[2]), float(parameters_str[3]), placeholder_function)
        
    def print_summary(self):
        """
        For testing purposes.
        """
        print("unique ID:", self.unique_id)
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
        self.LLPModel_uniqueIDs = [m.uniqueID for m in self.LLPModels]
        self.LLP_funcs = [(m.interactions_per_cm, m.decay_factor) for m in self.LLPModels]
        # @TODO: add medium with different number densities
        
    def calc_LLP_probability(self, length_list: list, energy_list: list) -> list:
        """
        Computes the total detectable LLP probability for a muon track for all models in the LLPEstimator.
        
        Convolution of segmented thick target approximation and decay factor.

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

        # @TODO: come up with clever way using numpy functionality to compute the probabilities faster
        
        # numpify input
        length_array = np.asarray(length_list)*100.0 # convert to cm
        energy_array = np.asarray(energy_list)       # GeV
        track_length = length_array[-1]              # length entry to exit of detector (already trimmed for exit and entry margins) in cm

        # parameters for calculations                         
        l2_array = track_length - length_array - self.min_gap # from production point to furthest available decay point
        delta_L = np.append(np.diff(length_array), 0)         # step length, in cm
        
        # compute probability = sum( delta_L * decay_factor * number_density * tot_xsec )
        # @TODO: make more readable
        matrix_for_calc = np.row_stack([inter_per_cm(energy_array)*decay(self.min_gap, l2_array, energy_array)
                                             for inter_per_cm, decay in self.LLP_funcs]) # 2D matrix with rows corresponding to each model
        probabilities = matrix_for_calc @ delta_L # NxM*Mx1 where N is # of models and M is # of length steps
        return probabilities

    def calc_LLP_probability_with_ID(self, length_list: list, energy_list: list) -> dict:
        """
        Returns the probabilities calculated in calc_LLP_probability with the LLPModel uniqueID.
        :param length_list: List of lengths from 0 to end of detector in meters, already trimmed for entry/exit margins. Last element should be total length.
        :param energy_list: List of energies of the muon from detector entry to exit in GeV. Ordered with length_list.
        :return dict: Returns a dict of detectable LLP probabilities mapped with LLPModel uniqueID.
        """
        probabilities = self.calc_LLP_probability_with_ID(length_list, energy_list)
        map_ID_probability = {ID: prob for ID, prob in zip(self.LLPModel_uniqueIDs, probabilities)}
        return map_ID_probability

    def get_LLPModels(self) -> list:
        return self.LLPModels
