import pandas as pd
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import corner

def plot_2D_hist(x, y, bins=50, xlabel=None, ylabel=None, title=None, xlog=False, ylog=False):
    plt.figure()
    # if log scale, make bins logarithmic
    if xlog:
        bins_x = np.logspace(np.log10(min(x)), np.log10(max(x)), bins)
        plt.xscale("log")
    else:
        bins_x = bins
    if ylog:
        bins_y = np.logspace(np.log10(min(y)), np.log10(max(y)), bins)
        plt.yscale("log")
    else:
        bins_y = bins
    # create histogram
    plt.hist2d(x, y, bins=[bins_x, bins_y])
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)

def plot_1D_hist(x, bins=50, xlabel=None, xlog=False, title=None, label=None):
    plt.figure()
    # if log scale, make bins logarithmic
    if xlog:
        bins_x = np.logspace(np.log10(min(x)), np.log10(max(x)), bins)
        plt.xscale("log")
    else:
        bins_x = bins
    # create histogram
    plt.hist(x, bins=bins_x, label=label)
    if label is not None:
        plt.legend()
    if xlabel is not None:
        plt.xlabel(xlabel)
    if title is not None:
        plt.title(title)
        
def corner_plot(df):
    
    ##### COORDINATE CORNER PLOT #####
    corner_variables = [
        "prodx_diff",
        "prody_diff",
        "prodz_diff",
        "decayx_diff",
        "decayy_diff",
        "decayz_diff",
    ]

    # Create a list of performance variable values
    performance_values = [df[var] for var in corner_variables]

    # Plot the histogram triangle
    print("Creating corner plot")
    figure = corner.corner(
        np.transpose(performance_values),
        labels=corner_variables,
        show_titles=True,
        title_fmt=".2f",
        plot_contours=True,
        bins=20,
        smooth=True,
        # axes_scale="log",
    )

def performance_histograms(df, log_dict):
    fig, axs = plt.subplots(3, 4, figsize=(12, 12))
    keys = list(log_dict.keys())
    for i in range(3):
        for j in range(4):
            tile_num = i*4 + j
            if tile_num >= len(keys):
                break
            axs[i, j].set_title(keys[tile_num] + ": mean {:.2f}".format(np.mean(df[keys[tile_num]])))
            if keys[tile_num] in log_dict:
                if log_dict[keys[tile_num]]:
                    axs[i, j].set_xscale("log")
                    bins = np.logspace(np.log10(min(df[keys[tile_num]])), np.log10(max(df[keys[tile_num]])), 30)
                else:
                    bins = 30
            axs[i, j].hist(df[keys[tile_num]], bins=bins)
            # axs[i, j].set_xlabel('Units')
            # axs[i, j].set_ylabel('Count')
    plt.tight_layout()


# Create the argument parser
parser = argparse.ArgumentParser(description='Plot some events.')

parser.add_argument("-i", "--input-file", action="store",
    type=str, required=True, dest="input-file",
    help="Input performance csv. Separate multiple files with commas.")

# Parse the arguments
params = vars(parser.parse_args())  # dict()

# split input files. If only one file, split will return a list with one element
filenames = params["input-file"].split(",")

xnames = [
    "muon_energy",
    "muon_zenith",
    "muon_length",
    "total_hits",
]

ynames = [
    "angular_diff",
    "gap_diff",
    "prodx_diff",
    "prody_diff",
    "prodz_diff",
    "decayx_diff",
    "decayy_diff",
    "decayz_diff",
    "prod_diff",
    "decay_diff",
    "MSE"
]

xlog_dict = {
    "muon_energy": True,
    "muon_zenith": False,
    "muon_length": False,
    "total_hits": True,
}

ylog_dict = {
    "angular_diff": True,
    "gap_diff": False,
    "prodx_diff": False,
    "prody_diff": False,
    "prodz_diff": False,
    "decayx_diff": False,
    "decayy_diff": False,
    "decayz_diff": False,
    "prod_diff": False,
    "decay_diff": False,
    "MSE": True,
}

# loop over the files and visualize performance
for filename in filenames:
    # get performance file and dataset name
    print(filename)
    df = pd.read_csv(filename)
    folder = os.path.dirname(filename)
    datasetname = folder.split("/")[-2]

    ##### CORNER PLOT #####
    corner_plot(df)
    plt.savefig(f"{folder}/corner_plot.png")
    
    ##### 1D HISTOGRAMS #####
    # all together
    performance_histograms(df, log_dict=ylog_dict)
    plt.savefig(f"{folder}/performance_histograms.png")
    # separate plots for each variable
    folder_1D = f"{folder}/performance_histograms"
    if not os.path.exists(folder_1D):
        os.mkdir(folder_1D)
    for yname in ynames:
        y = df[yname]
        plot_1D_hist(y, xlabel=yname, xlog=ylog_dict[yname],
                     title = f"{datasetname}",
                     label = f"Mean: {y.mean():.2f}, Std: {y.std():.2f}")
        plt.savefig(f"{folder_1D}/{yname}.png")
        plt.close()

    ##### 2D HISTOGRAMS #####
    spectrumfolder = f"{folder}/input_spectrum"
    if not os.path.exists(spectrumfolder):
        os.mkdir(spectrumfolder)
    # plot zenith vs length
    plot_2D_hist(df["muon_zenith"], df["muon_length"], xlabel="Muon Zenith", ylabel="Muon Length",
                 title=f"{datasetname}",
                 xlog=False, ylog=False)
    plt.savefig(f"{spectrumfolder}/muon_zenith_vs_muon_length.png")
    for xname in xnames:
        # add 1D input spectrum histogram
        plot_1D_hist(df[xname], xlabel=xname, xlog=xlog_dict[xname],
                     title = f"{datasetname}",
                     label = f"Mean: {df[xname].mean():.2f}, Std: {df[xname].std():.2f}")
        plt.savefig(f"{spectrumfolder}/{xname}.png")
        plt.close()
        # create subfolderfor 2D histograms
        folder_2D = f"{folder}/{xname}_2D"
        if not os.path.exists(folder_2D):
            os.mkdir(folder_2D)
        # all performance metrics
        for yname in ynames:
            x = df[xname]
            y = df[yname]
            plot_2D_hist(x, y, xlabel=xname, ylabel=yname,
                         title=f"{datasetname}",
                         xlog=xlog_dict[xname],
                         ylog=ylog_dict[yname])
            plt.savefig(f"{folder}/{xname}_2D/{xname}_vs_{yname}.png")
            plt.close()
    # vs gap length
    folder_gap = f"{folder}/gap_length_2D/"
    if not os.path.exists(folder_gap):
        os.mkdir(folder_gap)
    for yname in ynames:
        x = df["gap_length"]
        y = df[yname]
        plot_2D_hist(x, y, xlabel="Gap Length [m]", ylabel=yname,
                     title=f"{datasetname}",
                     xlog=False, ylog=ylog_dict[yname])
        plt.savefig(f"{folder_gap}gap_vs_{yname}.png")
        plt.close()
