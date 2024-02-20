"""
@TODO: update docstring
Classes used to estimate detectable LLP event probability.
"""
import numpy as np

class LLPMedium():
    """
    Struct to hold number density of nuclei.
    """
    def __init__(self, name: str, number_density: float, Z: int, A: int):
        self.name           = name           # e.g. "O" or "H"
        self.number_density = number_density # nuclei per cm^3
        self.Z              = Z              # atomic number
        self.A              = A              # mass number

class LLPProductionCrossSection():
    """
    @TODO: add docstring
    """
    def __init__(self, func_tot_xsec_list: list, medium_list: list):
        self.func_tot_xsec_list = func_tot_xsec_list # input GeV energy, returns cm^2
        self.medium_list = medium_list # LLPMediums, ordered with func_tot_xsec_list

    def interactions_per_cm(self, energy: float) -> float:
        """
        Total cross section weighted with number density for all elements in medium.
        $\Sigma^{elem.}_{i} \sigma^{i}_tot(E) \cdot n_{i}$
        :param energy: Energy of the muon in GeV.
        :return float: Total xsec times num density, units of cm^-1.
        """
        # @TODO: is it unnecessary for each grid point to run m.number_density or doesn't matter?
        return sum([xsec(energy)*m.number_density
                    for xsec, m in zip(self.func_tot_xsec_list, self.medium_list)])

class LLPModel():
    """
    LLP model parameters and production cross section function used in LLP estimation.
    """
    def __init__(self,
                 name: str,
                 mass: float,
                 eps: float,
                 tau: float,
                 llp_xsec: LLPProductionCrossSection):
        self.name       = name                 # such as DarkLeptonicScalar, etc.
        self.mass       = mass                 # in GeV
        self.eps        = eps                  # coupling to SM
        self.tau        = tau                  # lifetime in s
        self.llp_xsec   = llp_xsec             # for computing interactions per cm
        self.unique_id  = self.get_unique_id() # string with model info besides xsec function

    def interactions_per_cm(self, energy: float) -> float:
        """
        Total cross section weighted with number density for all elements in medium.
        :param energy: Energy of muon in GeV.
        :return float: Interactions per cm.
        """
        return self.llp_xsec.interactions_per_cm(energy)

    def get_lifetime(self, gamma: float = 1.0) -> float:
        """
        Lifetime of the LLP model in lab frame for some gamma.
        :param gamma: Lorentz boost of the LLP. Given by E/m.
        :return float: Lifetime of the LLP in lab frame.
        """
        return self.tau * gamma

    def decay_factor(self, l1: float, l2: float, energy: float) -> float:
        """
        Probability to decay between lengths l1 and l2.

        Integrate exponential decay pdf between l1 and l2:
        $\int^l1_l2 1/c*gamma*tau*exp(-l/c*gamma*tau)$

        :param l1: Minimum length before decay in cm. Same as minimum detectable gap length.
        :param l2: Maximum length before decay in cm.
        :param energy: Energy of the LLP.
        :return float: Between 0-1. Fraction of the decay pdf within length l1 and l2.
        """
        c_gamma_tau = 29979245800.0 * self.get_lifetime(energy/self.mass) # c [cm/s] * gamma * tau
        prob = np.exp(-l1/c_gamma_tau) - np.exp(-l2/c_gamma_tau)
        # check for negative values
        if isinstance(prob, np.ndarray):
            prob[prob < 0] = 0.0 # if l2 is shorter than l1, return 0
        else:
            prob = max(prob, 0.0)
        return prob

    def get_unique_id(self) -> str:
        # @TODO: how to include the cross section function?
        """W
        Encodes the model in a underscore separated string.
        Used to reconstruct the LLPModel (except cross section function)."
        """
        parameters_str = [self.name, str(self.mass), str(self.eps), str(self.tau)]
        unique_id = "_".join(parameters_str)
        return unique_id

    @classmethod
    def from_unique_id(cls, unique_id: str):
        # @TODO: how to id cross section function? leave out?
        """
        Returns a new LLPModel object from a unique id.
        """
        parameters_str = unique_id.split("_")
        return cls(parameters_str[0],
                   float(parameters_str[1]),
                   float(parameters_str[2]),
                   float(parameters_str[3]),
                   None)

    def print_summary(self):
        """
        For testing purposes.
        """
        print("unique ID:", self.unique_id)
        print("Parameters of model", self.name, ":")
        print("mass", self.mass)
        print("eps", self.eps)
        print("tau", self.tau)
        print("LLPProductionCrossSection", self.llp_xsec)
        print(
            "Test LLPProductionCrossSection at 500 GeV",
            [calc_tot_xsec(500.0) for calc_tot_xsec in self.llp_xsec.func_tot_xsec_list]
        )
        print("Test interaction per cm at 500 GeV", self.interactions_per_cm(500.0))
        print("Decay factor 500 GeV 100 to 800 m", self.decay_factor(100.0, 800.0, 500.0))


class LLPEstimator():
    """
    Class to calculate detectable LLP probability for a list of LLPModels.
    Calculates detectable LLP probability for each model given a
    muon track (list of ordered length steps and energies).
    Input expected in meters and GeV. Internally computes with centimeters.
    """
    def __init__(self, llpmodels: list, min_gap_meters: float = 50.0):
        self.min_gap   = min_gap_meters*100.0 # shortest detectable LLP gap [m -> cm]
        self.llpmodels = llpmodels            # make sure this stay ordered

        # for quicker access later
        self.llpmodel_unique_ids = [m.unique_id for m in self.llpmodels]
        self.llp_funcs = [(m.interactions_per_cm, m.decay_factor) for m in self.llpmodels]

    def calc_llp_probability(self, length_list: list, energy_list: list) -> list:
        """
        Computes the total detectable LLP probability for a muon track.

        Detectable events have production and decay vertex inside detector volume,
        and sufficiently long decay gap. Computed through convolution of segmented thin target
        approximation convolved with decay factor (partially integrated decay pdf).

        Computes probability separately for all models in the LLPEstimator.

        :param length_list: Lengths from 0 to end of detector in m. \
            Trimmed for entry/exit margins. Last element should be total length.

        :param energy_list: Energies of the muon from detector entry to exit in GeV. \
            Ordered with length_list.

        :return list: Returns a list of detectable LLP probabilities. \
            Ordered with list of LLPModels.
        """
        if len(length_list) != len(energy_list):
            raise ValueError("length_list and energy_list \
                             must contain same number of elements")
        if min(energy_list) < 0.0:
            raise ValueError("Negative energies not allowed. \
                             If muon stopped, give a length and \
                             energy list that stops at stopping point.")

        # @TODO: clever way using numpy functionality to compute the probabilities faster

        # numpify input
        length_array = np.asarray(length_list)*100.0 # convert to cm
        energy_array = np.asarray(energy_list)       # GeV
        track_length = length_array[-1]              # from entry to exit of detector

        # parameters for calculations
        l2_array = track_length - length_array - self.min_gap # from prod. vertex to furthest decay vertex
        delta_L = np.append(np.diff(length_array), 0)         # step length, in cm

        # compute probability:
        # sum( delta_L * decay_factor * sum_atoms(atom_number_density * tot_xsec_atom) )
        # @TODO: make more readable
        # 2D matrix with rows = models, cols = thin target approx segments
        matrix_for_calc = np.row_stack(
            [inter_per_cm(energy_array) * decay(self.min_gap, l2_array, energy_array)
                for inter_per_cm, decay in self.llp_funcs]
        )
        probabilities = matrix_for_calc @ delta_L # NxM*Mx1 where N is no. models and M is no. length steps

        # list of probabilities, ordered with self.llpmodels
        return probabilities

    def calc_llp_probability_with_id(self, length_list: list, energy_list: list) -> dict:
        """
        Returns the probabilities calculated in calc_llp_probability mapped to LLPModel unique_id.
        :param length_list: Lengths from 0 to end of detector in m. \
            Trimmed for entry/exit margins. Last element should be total length.

        :param energy_list: Energies of the muon from detector entry to exit in GeV. \
            Ordered with length_list.

        :return dict: Returns a dict of detectable LLP probabilities mapped with LLPModel uniqueID.
        """
        probabilities = self.calc_llp_probability_with_id(length_list, energy_list)
        map_id_probability = dict(zip(self.llpmodel_unique_ids, probabilities))
        return map_id_probability