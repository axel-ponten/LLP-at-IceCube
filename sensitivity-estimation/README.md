Estimating the number of detectable LLP events at IceCube from a simulated atmospheric muon flux. Contains scripts to calculate the event by event detectable LLP probability (probability that both production and decay vertex are inside detector volume). Contains interpolation table generation used for total production cross sections. Assumes the LLP takes all the muon energy (justified by the differential cross section peaking at fractional energy x -> 1)

Old naive estimation in naive_estimate_DLS_events.nb, assuming a monochromatic muon flux with zenith = 0 and using the thin target approximation (including a decay factor to contain both LLP vertices in the detector).

## Background
This project is part of a beyond standard model analysis at the IceCube neutrino observatory, which is a cube kilometer large neutrino detector buried 2 km deep in the antarctic ice (https://icecube.wisc.edu/). The detector is built to measure rare cosmic neutrino interactions a few times a year, but also detects the abundant atmospheric muon background at a rate of 3 kHz. The analysis searches for signatures of a hypothesized class of exotic particles called long lived particles (LLPs) in the background atmospheric muon flux. The LLPs could potentially be produced in a bremstrahlung-like process where the muon scatters of an oxygen or hydrogen in the ice. The LLP would then decay after some time, hopefully inside the detector in order to be detectable.

IceCube measures particles through their Cherenkov radiation in the ice. LLPs are dark and do not emit any Cherenkov light. Their production and subsequent decay would therefore leave a gap in the muon track where there is no light emitted. As such, we need both the production and decay vertex of the LLP to be inside the detector volume in order for the event to be detectable.

## What does the package ***llpestimation*** do?

The package's main class ***LLPEstimator*** computes the probability for a given atmospheric muon (represented by its energy along a list of length steps through the detector) to produce an LLP which has both production and decay vertex inside the detector. It does so for a list of LLP models that are defined by their name, mass, coupling, lifetime and production cross section (generated from interpolation tables). The production rate depends on the medium in which the muon travels, particularly the number density of nuclei on which the bremstrahlung-like production scattering takes place.

An example script is provided in the *examples* folder named *llpestimation_example_script.py*. Run it like `python llpestimation_example_script.py`.

#### How does it calculate event expectation rate?
The expected number of events is calculated using a segmented thin target approximation, convolved with a decay factor representing the probability for the produced LLP to decay within the detector volume. A visualization of the process can be found in `resources/Detectable LLP event.pdf`

The thin target approximation is in its general form

$$
N_{int.} = N \cdot dx \cdot n \cdot \sigma
$$

with $dx$ some short length segment, $n$ the number density of scatterers, and $\sigma$ the total cross section per scatterer (nucleus, for example).

Since we have energy loss of the muon travelling through the ice, we segment the thin target approximation and sum contributions from small steps along the muon track. At each step, if an LLP is produced, it must also decay after some minimum reconstructable gap $l_{min}$ but before the end of the detector $l_{max}$. We must also consider contribution from both oxygen and hydrogen. The final formula for probability of an LLP to be produced and decay within the detector volume is then given by

$$
P_{LLP} = \Sigma_{i}^{steps} \left[ f_{decay}(E_i) \Delta{L} \cdot \Sigma_{j}^{O, H} \left[ \sigma_{j}(E_i) \cdot n_j \right]  \right]
$$

where $E_i$ is the energy of the muon at step *i*, $\Delta L$ the step length of the segmentation, $n_j$ the number density of element *j*, and 

$$
f_{decay}(E_i) = e^{\frac{-l_{min}}{c \gamma \tau}} - e^{\frac{-l_{max}}{c \gamma \tau}}
$$

is the decay factor representing the probability to decay between length $l_{min}$ and $l_{max}$.

#### Creating LLPModel objects

An LLPModel consists of a name, mass, epsilon (coupling), lifetime and LLPProductionCrossSection. The LLPProductionCrossSection contains functions that return the total cross section in cm^2 given some energy E. These functions can for example be created through interpolation of the tables in the *cross_section_tables* directory.

The following code snippet could be used to create an list of LLPModel objects.

```python
def generate_DLSModels(masses, epsilons, names, table_paths):
    llpmodel_list = []
    n_oxygen = 6.02214076e23 * 0.92 / 18 # number density of oxygen in ice
    oxygen = LLPMedium("O", n_oxygen, 8, 16)
    for mass, eps, name, path in zip(masses, epsilons, names, table_paths):
        # lifetime
        tau = calculate_DLS_lifetime(mass, eps)
        # tot_xsec function from interpolation tables
        df = pd.read_csv(path, names=["E0", "totcs"])
        func_tot_xsec = interp1d(df["E0"], eps**2*df["totcs"],kind="linear", bounds_error=False,fill_value=(0.0, None))
        # create LLPProductionCrossSection
        llp_xsec = LLPProductionCrossSection([func_tot_xsec], [oxygen])
        # create new LLPModel
        llpmodel_list.append(LLPModel(name, mass, eps, tau, llp_xsec))
    return llpmodel_list
# parameters
masses = [0.115, 0.130] # GeV
epsilons = [1e-4, 1e-5]
names = ["DLS", "DLS"]
table_paths = ["cross_section_tables/totcs_WW_m_0.115.csv", "cross_section_tables/totcs_WW_m_0.13.csv"]
# create LLPModels
my_model_list = generate_DLSModels(masses, epsilons, names, table_paths)
```
#### Utility functions
Accompanying the package is a *estimation_utilities.py* script that helps implement particular LLP models like the dark leptonic scalar, to be used with the LLPEstimator.

#### Tests and profiling of ***llpestimator*** package
In tests folder, there is a test script that you can run with the command line prompt *py.test* and a profiling script that you can run like `python profile_llpestimation.py` (useful for checking bottleneck in calculation).

## Grid calculation
In the *examples* directory there are two python scripts to run I3LLPProbabilityCalculator on some CORSIKA-in-ice simulation for a mass/epsilon grid of LLPModels, and then plot the resulting rates.

This can be run like

```
python compute_grid_to_hdf.py -o test_grid_5_files.hdf5 -n 5
python plot_grid.py -i test_grid_5_files.hdf5 -n 5 -o grid_plot.png -y 10
```
The number of CORSIKA files used is necessary for the weighting of the events.

## Cross section tables
To avoid calculating the exact total cross section for each event, we use interpolation tables to improve the speed of the computation. Since the coupling of the LLP scales the total cross section by $\sigma \rightarrow \epsilon^2 \sigma$ we only compute the tables for $\epsilon = 1$, and then the user can scale the values accordingly before interpolating the points. These tables are produced through mathematica notebooks containing cross section formulas.
