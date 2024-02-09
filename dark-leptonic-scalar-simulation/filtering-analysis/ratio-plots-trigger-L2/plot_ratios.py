import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import argparse
import tables
import glob


# Get params from parser
parser = argparse.ArgumentParser(description="create plots from hdf files")
parser.add_argument("-i", "--inputfolder", action="store",
        type=str, default="", dest="inputfolder",
        help="Input hdf5 folders. comma separated for many")

parser.add_argument("--all-folders", action="store_true",
        default=False, dest="all-folders",
        help="use all folders in directory for plot")
params = vars(parser.parse_args())  # dict()
params['inputfolder'] = params['inputfolder'].split(',')

if not params["all-folders"] and params["inputfolder"] == "":
    print("Error! Either choose all folders or give list of inputfolders")
    exit()
if params["all-folders"]:
    params["inputfolder"] = glob.glob("clusterID_*")
    print("using all folders", params["inputfolder"])

# read in all the data from the folders
trigger_dfs = []
L2_dfs      = []
clean_MMCTrackList = lambda df : df[df["vector_index"] == 0] 
for folder in params["inputfolder"]:
    # mmctracklist is polluted with LLP decay muon, keep only first
    trigger_df = clean_MMCTrackList(pd.read_hdf(folder+"/trigger.hdf5", "MMCTrackList"))
    L2_df      = clean_MMCTrackList(pd.read_hdf(folder+"/L2.hdf5", "MMCTrackList"))
    trigger_dfs.append(trigger_df)
    L2_dfs.append(L2_df)
    
trigger_df = pd.concat(trigger_dfs)
L2_df = pd.concat(L2_dfs)

########################################
################# PLOT #################
########################################

################ Energy ################
# create bins
logbins = np.geomspace(trigger_df["Ei"].min(), trigger_df["Ei"].max(), 20)

# unweighted counts plot
plt.figure()
plt.hist(trigger_df["Ei"], bins = logbins, alpha=0.5, label="Trigger")
plt.hist(L2_df["Ei"], bins = logbins, alpha=0.5, label = "L2")
plt.legend()
plt.loglog()
plt.xlabel("Energy [GeV]")
plt.ylabel("Unweighted count")
plt.savefig("counts_plot_energy.png")

# ratio plot
hist_trigger, bin_edges = np.histogram(trigger_df["Ei"], bins = logbins)
hist_L2,      bin_edges = np.histogram(L2_df["Ei"], bins = logbins)
ratios = [x/y if y != 0 else 0 for x,y in zip(hist_L2, hist_trigger)]
# Calculate bin centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

plt.figure()
plt.plot(bin_centers, ratios, "o-", color="black")
plt.title("ratio")
plt.xlabel("Energy [GeV]")
plt.ylabel("Ratio L2/Trigger")
plt.xscale("log")
plt.savefig("ratio_plot_energy.png")

# combined ratio and count plot
fig, ax1 = plt.subplots()
ax1.hist(trigger_df["Ei"], bins = logbins, alpha=0.5, label="Trigger", color = "navy")
ax1.hist(L2_df["Ei"], bins = logbins, alpha=0.5, label = "L2", color = "firebrick")
ax1.loglog()
ax1.set_xlabel("Energy [GeV]")
ax1.set_ylabel("Unweighted count")

ax2 = ax1.twinx()
ax2.plot(bin_centers, ratios, "o-", color="black", label="ratio")
ax2.set_ylabel("Ratio L2/Trigger")

# ask matplotlib for the plotted objects and their labels
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
plt.title("LLP survival rate trigger to L2")
plt.savefig("combined_energy.png")


################ Zenith ################
# bins
bins = 10

# unweighted counts plot
plt.figure()
plt.hist(trigger_df["zenith"], bins = bins, alpha=0.5, label="Trigger")
plt.hist(L2_df["zenith"], bins = bins, alpha=0.5, label = "L2")
plt.legend()
plt.xlabel("Zenith [rad]")
plt.ylabel("Unweighted count")
plt.savefig("counts_plot_zenith.png")

# ratio plot
hist_trigger, bin_edges = np.histogram(trigger_df["zenith"], bins = bins)
hist_L2,      bin_edges = np.histogram(L2_df["zenith"], bins = bins)
ratios = [x/y if y != 0 else 0 for x,y in zip(hist_L2, hist_trigger)]
# Calculate bin centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

plt.figure()
plt.plot(bin_centers, ratios, "o-", color="black")
plt.title("ratio")
plt.xlabel("Zenith [rad]")
plt.ylabel("Ratio L2/Trigger")
plt.savefig("ratio_plot_zenith.png")

# combined ratio and count plot
fig, ax1 = plt.subplots()
ax1.hist(trigger_df["zenith"], bins = bins, alpha=0.5, label="Trigger", color = "navy")
ax1.hist(L2_df["zenith"], bins = bins, alpha=0.5, label = "L2", color = "firebrick")
ax1.set_xlabel("Zenith [rad]")
ax1.set_ylabel("Unweighted count")

ax2 = ax1.twinx()
ax2.plot(bin_centers, ratios, "o-", color="black", label="ratio")
ax2.set_ylabel("Ratio L2/Trigger")

# ask matplotlib for the plotted objects and their labels
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
plt.title("LLP survival rate trigger to L2")
plt.savefig("combined_zenith.png")

################ 2D histogram ################

"""
zenith_bins = np.linspace(0,np.pi/2.0,5)
energy_bins = np.geomspace(trigger_df["Ei"].min(), trigger_df["Ei"].max(), 5)
print(zenith_bins)
print(energy_bins)

# Create 2D histogram for both datasets
hist1, xedges, yedges = np.histogram2d(trigger_df["Ei"], trigger_df["zenith"], bins=(energy_bins, zenith_bins))
print(hist1)
hist2, _, _ = np.histogram2d(L2_df["Ei"], L2_df["zenith"],  bins=(energy_bins, zenith_bins))
print(hist2)

# Calculate the ratios
ratios = np.divide(hist2, hist1, out=np.zeros_like(hist1), where=hist1 != 0)
print(ratios)

# Plot the 2D histogram of ratios
plt.figure()
plt.imshow(ratios.T,
           extent=[np.log10(xedges[0]), np.log10(xedges[-1]), yedges[0], yedges[-1]],
           cmap='viridis')
plt.colorbar(label='Ratio')
plt.xscale("log")
plt.title('2D Histogram of Ratios between Bin Counts')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig("test_2d.png")
"""