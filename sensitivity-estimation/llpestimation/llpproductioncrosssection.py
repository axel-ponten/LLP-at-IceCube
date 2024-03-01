class LLPProductionCrossSection():
    """
    Class that contains an ordered list of production cross sections
    and medium. Cross section takes GeV as input and outputs cm^2.
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
