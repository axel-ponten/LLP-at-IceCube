import pandas as pd
import pylab as plt
import simweights
import numpy as np
import math

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))

n_files = 30
# load the hdf5 file that we just created using pandas
hdffile = pd.HDFStore("Muons_in_ice_"+str(n_files)+"files.hdf5", "r")
#hdffile = pd.HDFStore("test.hdf5", "r")

# instantiate the weighter object by passing the pandas file to it
weighter = simweights.CorsikaWeighter(hdffile, nfiles = n_files)

# create an object to represent our cosmic-ray primary flux model
flux = simweights.GaisserH4a()

# get the weights by passing the flux to the weighter
weights = weighter.get_weights(flux)

# print some info about the weighting object
print(weighter.tostring(flux))
# print to file
with open("weighting_object_"+str(n_files)+".txt", 'a') as outfile:
    outfile.write(weighter.tostring(flux))

# create equal spaced bins in log space

# get energy of the primary cosmic-ray from `PolyplopiaPrimary`
primary_energy = weighter.get_column("PolyplopiaPrimary", "energy")
primary_zenith = weighter.get_column("PolyplopiaPrimary", "zenith")

N = weighter.get_column("MuonAtDetectorBoundary", "N")
max_energy = weighter.get_column("MuonAtDetectorBoundary", "HighestMuonEnergyTrack")

energy_N1 = [energy for energy, n in zip(max_energy, N) if n == 1 ]
weights_N1 = [weight for weight, n in zip(weights, N) if n ==1]

# histogram the primary energy with the weights
plt.figure()
plt.hist(N, weights=weights, bins=[0,1,2,3,4,5,6,7,8,9,10])
plt.xlabel("N")
plt.ylabel("freq. [Hz]")
plt.title("Muon multiplicity at detector boundary")
plt.savefig("multiplicity_at_boundary.png")

logbins = np.geomspace(50, max(energy_N1), 50)
mean, std = weighted_avg_and_std(energy_N1, weights_N1)
plt.figure()
plt.hist(energy_N1,  weights = weights_N1, bins = logbins, label = "mean: " + "{:.1f}".format(mean) + "\n" + "std: " + "{:.1f}".format(std))
plt.xlabel("energy [GeV]")
plt.ylabel("freq. [Hz]")
plt.legend()
plt.loglog()
plt.title("Single muon energy at MMC volume boundary")
plt.savefig("highest_single_E_at_boundary.png")

logbins = np.geomspace(50, max(max_energy), 50)
plt.figure()
plt.hist(max_energy,  weights = weights, bins = logbins)
plt.xlabel("energy [GeV]")
plt.ylabel("freq. [Hz]")
plt.loglog()
plt.title("Highest Energy Muon per event at detector boundary")
plt.savefig("highest_E_at_boundary.png")
