import pandas as pd
import numpy as np
import simweights

def print_eventrate(filename, nfiles = 1): 
    n_files = 1
    # load the hdf5 file that we just created using pandas
    hdffile = pd.HDFStore(filename, "r")
    
    # instantiate the weighter object by passing the pandas file to it
    weighter = simweights.CorsikaWeighter(hdffile, nfiles = n_files)
    
    # create an object to represent our cosmic-ray primary flux model
    flux = simweights.GaisserH4a()
    
    # get the weights by passing the flux to the weighter
    weights = weighter.get_weights(flux)
    
    # print some info about the weighting object
    print(weighter.tostring(flux))
    return weighter.tostring(flux)



filename = "/data/user/axelpo/LLP-at-IceCube/dark-leptonic-scalar-simulation/analysis/sanity-check-L2/L2_sanity.hdf5"

flux_dictionary = {
    "trigger"    : print_eventrate("20904_examples/IC86.2020_corsika.020904.000001.hdf5"),
    "L1"         : print_eventrate("L1_sanity.hdf5"),
    "L2"         : print_eventrate("L2_sanity.hdf5"),
    "L2_datasim" : print_eventrate("20904_examples/Level2_IC86.2020_corsika.020904.000001.hdf5"),
}

# print to file
with open("weighting_object.txt", 'w') as outfile:
    for name, flux in flux_dictionary.items():
        outfile.write("\n ----- " + name + " ----- \n")
        outfile.write(flux)