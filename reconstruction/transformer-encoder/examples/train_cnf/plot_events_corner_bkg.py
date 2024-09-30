""" Create a folder with the results of the training.
    - Training and validation loss plots
    - Model settings
    - Corner plots
    - Performance metrics
    - Example reconstructions
"""

import os
from llp_gap_reco.dataset import LLPDataset, UnlabeledLLPDataset, LLPSubset, llp_collate_fn, llp_collate_unlabeled_fn
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
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse
import numpy as np
import corner


# Create the argument parser
parser = argparse.ArgumentParser(description='Plot some events.')

# Add the arguments
parser.add_argument('--topfolder', type=str, help='Path to the top folder')
parser.add_argument('--filenamestart', type=str, default='base', help='Start of the filename for glob')
parser.add_argument('--modelpath', type=str, help='Path to the model file')
parser.add_argument('--flowpath', type=str, help='Path to the flow file')
parser.add_argument('--modelconfig', type=str, help='Path of the model config')
parser.add_argument('--normpath', type=str, help='Path to the normalization arguments file')

# Parse the arguments
args = parser.parse_args()

# Get the values from the arguments
top_folder = args.topfolder
filename_start = args.filenamestart
model_path = args.modelpath
flow_path = args.flowpath
config_path = args.modelconfig
norm_path = args.normpath

# get folder
trainfolder = os.path.dirname(model_path)
folder = f"{trainfolder}/performance/muon_bkg_corner_plots/"
if not os.path.exists(folder):
    os.makedirs(folder)

# create model
model, pdf = create_full_model(config_path, device="cuda")
model.eval()
pdf.eval()
model.load_state_dict(torch.load(model_path))
pdf.load_state_dict(torch.load(flow_path))



# normalizaton args
with open(norm_path, "r") as file:
    normalization_args = yaml.safe_load(file)
# unlabeled or labeled data?

# filepaths
file_paths = glob.glob(top_folder + filename_start + "*.pq")        
dataset = UnlabeledLLPDataset(
    top_folder + "indexfile.pq",
    file_paths,
    top_folder + "feature_indices.yaml",
    normalize_data=True,
    normalization_args=normalization_args,
    device="cuda",
    dtype=torch.float32,
    shuffle_files=True,
)
batch_size = 16
dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=llp_collate_unlabeled_fn)

##### PERFORMANCE #####
# create folder
model_folder = os.path.dirname(model_path)
performance_folder = model_folder + "/performance/"
if not os.path.exists(performance_folder):
    os.mkdir(performance_folder)


##### PLOT EVENT EXAMPLES #####
print("Creating example reconstructions")
# dataloader.dataset.shuffle() # shuffle
# grab one batch
datavecs, datalens, labels = next(iter(dataloader))
samplesize = 1000
samples = []
truth = []
with torch.no_grad():
    nn_output = model(datavecs, datalens)
    nn_output = nn_output.double()
    predictions, pred_std = predict_cnf(model, pdf, datavecs, datalens, samplesize=samplesize)
    for i in range(samplesize):
        target_sample, _, _, _ = pdf.sample(conditional_input=nn_output)
        target_sample = target_sample.cpu()
        sample_gap = distance(target_sample[:,0], target_sample[:,1], target_sample[:,2], target_sample[:,3], target_sample[:,4], target_sample[:,5])
        target_sample = torch.cat([target_sample, sample_gap.unsqueeze(1)], dim=1)
        samples.append(target_sample)
samples = torch.stack(samples, dim = 0)
permuted = samples.permute(1, 2, 0) # event, coordinate, sample

corner_variables = ["prod_x", "prod_y", "prod_z", "decay_x", "decay_y", "decay_z", "gap_len"]
for i, (event_samples, label) in enumerate(zip(permuted, labels)):
    # corner plot
    figure = corner.corner(
        np.transpose(event_samples.numpy()),
        labels=corner_variables,
        show_titles=True,
        title_fmt=".2f",
        plot_contours=True,
        bins=50,
        smooth=True,
        # axes_scale="log",
    )

    # Create a new gridspec for the figure
    gs = gridspec.GridSpec(len(corner_variables), len(corner_variables), figure=figure)

    # Add a new plot in the upper right corner that spans 3x3 tiles
    ax = figure.add_subplot(gs[0:2, -3:], projection='3d')

    # plot_event(datavecs[i], predictions[i], labels[i], normalization_args, ax=ax)
    plot_event(datavecs[i], predictions[i], None, normalization_args, ax=ax)
    plt.savefig(f"{folder}muon_bkg_corner_plot_{i}.png")
    plt.close()