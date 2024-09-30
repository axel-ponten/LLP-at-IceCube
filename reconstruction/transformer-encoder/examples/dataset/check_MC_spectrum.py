""" Create a folder with the results of the training.
    - Training and validation loss plots
    - Model settings
    - Corner plots
    - Performance metrics
    - Example reconstructions
"""

import os
from llp_gap_reco.dataset import LLPDataset, LLPSubset, llp_collate_fn
from llp_gap_reco.encoder import LLPTransformerModel
from llp_gap_reco.training.performance import *
from llp_gap_reco.training.utils import *

from torch.utils.data import DataLoader
import yaml
import torch
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse
import numpy as np
import corner


# Create the argument parser
parser = argparse.ArgumentParser(description='Plot some events.')

# Add the arguments
parser.add_argument('--topfolder', type=str, help='Path to the top folder')
parser.add_argument('--save', action='store_true', help='Save the plots')

# Parse the arguments
args = parser.parse_args()

# Get the values from the arguments
top_folder = args.topfolder
index_file_path = os.path.join(top_folder, "indexfile.pq")
total_index_info = pd.read_parquet(index_file_path)

print(total_index_info)

# make plot folder if doesnt exist
if args.save:
    plot_folder = os.path.join(top_folder, "plots")
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

# histogram muon_energy
plt.figure()
plt.hist(total_index_info["muon_energy"], bins=100)
plt.xlabel("Muon energy [GeV]")
plt.ylabel("Number of events")
plt.yscale("log")

# if save
if args.save:
    plt.savefig(os.path.join(plot_folder, "muon_energy.png"))
else:
    plt.show()

# histogram muon_zenith
plt.figure()
plt.hist(total_index_info["muon_zenith"], bins=100)
plt.xlabel("Muon zenith [rad]")
plt.ylabel("Number of events")
plt.yscale("log")

# if save
if args.save:
    plt.savefig(os.path.join(plot_folder, "muon_zenith.png"))
else:
    plt.show()

# histogram muon_length
plt.figure()
plt.hist(total_index_info["muon_length"], bins=100)
plt.xlabel("Muon length [m]")
plt.ylabel("Number of events")
plt.yscale("log")
# if save
if args.save:
    plt.savefig(os.path.join(plot_folder, "muon_length.png"))
else:
    plt.show()


### CORNER PLOT ###
corner_variables = [
    "muon_energy",
    "muon_zenith",
    "muon_length",
]

# Create a list of performance variable values
performance_values = [total_index_info[var] for var in corner_variables]

# Plot the histogram triangle
print("Creating corner plot")
figure = corner.corner(
    np.transpose(performance_values),
    labels=corner_variables,
    show_titles=True,
    title_fmt=".2f",
    plot_contours=True,
    bins=100,
    smooth=True,
    # axes_scale="log",
)

if args.save:
    figure.savefig(os.path.join(plot_folder, "corner_plot.png"))
else:
    plt.show()