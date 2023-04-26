
import pandas as pd
import pylab as plt
import simweights

# load the hdf5 file that we just created using pandas
hdffile = pd.HDFStore("Test_bkg_for_weighting_100files.hdf5", "r")
#hdffile = pd.HDFStore("test.hdf5", "r")

# instantiate the weighter object by passing the pandas file to it
weighter = simweights.CorsikaWeighter(hdffile, nfiles = 100)

# create an object to represent our cosmic-ray primary flux model
flux = simweights.GaisserH4a()

# get the weights by passing the flux to the weighter
weights = weighter.get_weights(flux)

# print some info about the weighting object
print(weighter.tostring(flux))

# create equal spaced bins in log space
bins = plt.geomspace(3e2, 1e7, 50)

# get energy of the primary cosmic-ray from `PolyplopiaPrimary`
primary_energy = weighter.get_column("PolyplopiaPrimary", "energy")
primary_zenith = weighter.get_column("PolyplopiaPrimary", "zenith")
track_gap_probability = weighter.get_column("LLPBackgroundProbability", "NeutrinoInteractionProbability")
track_gap_length = weighter.get_column("LLPBackgroundProbability", "AvailableInteractionLength")

prob_rate = [p*w for p, w in zip(track_gap_probability, weights)]
total_prob_rate =  sum(prob_rate)
atmos_nu_rate = 20e-3 # 20 mHz
print("total probability rate is: ", total_prob_rate, " Hz")
print("seconds in a year ", 365*24*60*60)
print("events per year ", 365*24*60*60*total_prob_rate)
print("atmospheric neutrinos per year ", 365*24*60*60*atmos_nu_rate)

# histogram the primary energy with the weights
plt.figure()
plt.hist(primary_energy, weights=prob_rate, bins=bins)

# make the plot look good
plt.loglog()
plt.xlabel("Primary Energy [GeV]")
plt.ylabel("Event Rate [Hz]")
plt.xlim(bins[0], bins[-1])
plt.savefig("bkg_rate_energy.svg")
plt.show()

# histogram the primary energy with the weights
plt.figure()
plt.hist(primary_zenith, weights=prob_rate, bins=10)

# make the plot look good
plt.xlabel("Primary Zenith")
plt.ylabel("Event Rate [Hz]")
plt.savefig("bkg_rate_zenith.svg")
plt.show()


# histogram the primary energy with the weights
plt.figure()
plt.hist(track_gap_length, weights=weights, bins=70)

# make the plot look good
plt.xlabel("Available track gap length [m]")
plt.ylabel("Event Rate [Hz]")
plt.savefig("bkg_candidate_track_lengths.svg")
plt.show()

plt.figure()
plt.hist2d(track_gap_length, primary_zenith, weights = weights, bins = 50)
plt.xlabel("Available track gap length [m]")
plt.ylabel("Zenith angle")
plt.savefig("bkg_2d_available_length_zenith.svg")