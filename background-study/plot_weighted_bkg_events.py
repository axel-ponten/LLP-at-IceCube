import pandas as pd
import pylab as plt
import simweights

# load the hdf5 file that we just created using pandas
print("reading file ...")
#hdffile = pd.HDFStore("Selected_bkg_candidates_folder_0198000-0198999.hdf5", "r")
hdffile = pd.HDFStore("Selected_bkg_candidates_10files.hdf5", "r")

# instantiate the weighter object by passing the pandas file to it
print("weighting ...")
weighter = simweights.CorsikaWeighter(hdffile, nfiles = 10)

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
track_gap_probability = weighter.get_column("LLPBackgroundProbability", "TotalNeutrinoInteractionProbability")
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
plt.loglog()
plt.xlabel("Primary Energy [GeV]")
plt.ylabel("Event Rate [Hz]")
plt.xlim(bins[0], bins[-1])
plt.savefig("plots/bkg_rate_energy_10files.png")
plt.show()

# histogram the primary energy with the weights
plt.figure()
plt.hist(primary_zenith, weights=prob_rate, bins=10)
plt.xlabel("Primary Zenith")
plt.ylabel("Event Rate [Hz]")
plt.savefig("plots/bkg_rate_zenith_10files.png")
plt.show()


# histogram the primary energy with the weights
plt.figure()
plt.hist(track_gap_length, weights=weights, bins=70)
plt.xlabel("Available track gap length [m]")
plt.ylabel("Event Rate [Hz]")
plt.savefig("plots/bkg_candidate_track_lengths_10files.png")
plt.show()


plt.figure()
plt.hist2d(track_gap_length, primary_zenith, weights = weights, bins = 50)
plt.xlabel("Available track gap length [m]")
plt.ylabel("Zenith angle")
plt.savefig("plots/bkg_2d_available_length_zenith_10files.png")