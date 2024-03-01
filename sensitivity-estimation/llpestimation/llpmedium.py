class LLPMedium():
    """
    Struct to hold number density of nuclei.
    """
    def __init__(self, name: str, number_density: float, Z: int, A: int):
        self.name           = name           # e.g. "O" or "H"
        self.number_density = number_density # nuclei per cm^3
        self.Z              = Z              # atomic number
        self.A              = A              # mass number