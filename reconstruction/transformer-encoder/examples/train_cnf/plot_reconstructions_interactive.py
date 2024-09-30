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

# create model
model, pdf = create_full_model(config_path, device="cuda")
model.eval()
pdf.eval()
model.load_state_dict(torch.load(model_path))
pdf.load_state_dict(torch.load(flow_path))

# create dataloader
train_dataset, test_dataset = create_split_datasets(top_folder, filename_start, norm_path, split=0.8)
batch_size = 16
dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=llp_collate_fn)

##### PERFORMANCE #####

##### PLOT EVENT EXAMPLES #####
print("Creating example reconstructions")
with open(norm_path, "r") as file:
    normalization_args = yaml.safe_load(file)
# dataloader.dataset.shuffle() # shuffle
# grab one batch
datavecs, datalens, labels = next(iter(dataloader))
# grab one batch manually
title_list = []
for i in range(batch_size):
    data, label = test_dataset.__getitem__(i)
    mu_E, mu_Z, mu_L = test_dataset.get_muon_info_from_idx(i)
    title_list.append("Event {}: E: {:.2f}, Z: {:.2f}, L: {:.2f}".format(i, mu_E, mu_Z, mu_L))
# for corner plot
samplesize = 1000
with torch.no_grad():
    print("Predicting")
    predictions, pred_std = predict_cnf(model, pdf, datavecs, datalens, samplesize=samplesize)
    nn_output = model(datavecs, datalens)
    nn_output = nn_output.double()

# Plot event reconstructions
print("Plotting")
for i in range(len(datavecs)):
    plot_event(datavecs[i], predictions[i], labels[i], normalization_args, title = title_list[i])
    # plt.savefig(f"{folder}event_reconstruction_{i}.png")
    plt.show()
    plt.close()
