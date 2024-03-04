"""
Class used to estimate detectable LLP event probability
for a list of LLPModels given a muon represented
by a list of energies along length segments in the detector.
"""
import numpy as np

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

        # trim if muon stops (energy is zero)
        good_points = np.where(energy_array > 0)
        # if only zero muon energy
        if np.all(good_points == False):
            return np.zeros(len(self.llpmodels))
        length_array = length_array[good_points]
        energy_array = energy_array[good_points]

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
        probabilities = self.calc_llp_probability(length_list, energy_list)
        map_id_probability = dict(zip(self.llpmodel_unique_ids, probabilities))
        return map_id_probability