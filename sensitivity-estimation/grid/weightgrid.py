""" Module with function to weight a .hdf5 file
"""
import sys
sys.path.append("..")

import pandas as pd
import simweights
from llpestimation import LLPModel

def weight_CORSIKA_hdf5(inputfile: str, nCORSIKA: int):
    """ Weighs a CORSIKA -> hdf5 file.
    Returns HDFStore object and event weights.
    """
    # load the hdf5 file that we just created using pandas
    hdffile = pd.HDFStore(inputfile, "r")
    
    # instantiate the weighter object by passing the pandas file to it
    weighter = simweights.CorsikaWeighter(hdffile, nfiles = nCORSIKA)
    
    # create an object to represent our cosmic-ray primary flux model
    flux = simweights.GaisserH4a()
    
    # get the weights by passing the flux to the weighter
    weights = weighter.get_weights(flux)
    
    # print some info about the weighting object
    print(weighter.tostring(flux))

    return hdffile, weights

def get_mass_eps_from_id(llp_unique_id: str):
    model = LLPModel.from_unique_id(llp_unique_id)
    return model.mass, model.eps

def weighted_grid_llp_rate(hdffile, weights):
    """ Total weighted llp rate for some grid.
    """
    # open correct columns
    llp_prob = hdffile.select("LLPProbabilities")
    model_ids = llp_prob.keys()[5:] # SKIP first five cols, like EventID and RunID
    
    # do weighted sum of llp probabilities for each model
    masses   = []
    epsilons = []
    rates = []
    for model_id in model_ids:
        mass, eps = get_mass_eps_from_id(model_id)
        weighted_rate = llp_prob[model_id].dot(weights)
        masses.append(mass)
        epsilons.append(eps)
        rates.append(weighted_rate)

    return masses, epsilons, rates, model_ids

def print_verbose(hdffile, weights):
    """ prints output of each grid point.
    """
    # open correct columns
    llp_prob = hdffile.select("LLPProbabilities")
    model_ids = llp_prob.keys()[5:] # SKIP first five cols, like EventID and RunID
    
    # do weighted sum of llp probabilities for each model
    for model_id in model_ids:
        weighted_rate = llp_prob[model_id].dot(weights)
        print(model_id)
        print("Raw llp probs.", llp_prob[model_id])
        print("Sum of llp probs", llp_prob[model_id].sum())
        print("Weighted llp rate", weighted_rate)

def clean_min_events(masses, epsilons, signals, min_events: float):
    """ Remove all grid points with signal < min_events.
    """
    masses_good = []
    epsilons_good = []
    signals_good = []
    for mass, eps, signal in zip(masses, epsilons, signals):
        if signal >= min_events:
            masses_good.append(mass)
            epsilons_good.append(eps)
            signals_good.append(signal)
    return masses_good, epsilons_good, signals_good