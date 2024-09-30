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
batch_size = 128
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=llp_collate_fn)

# fill performance dict
batch_counter = 0
max_batch = -1
print("Starting performance evaluation")
labels_dict = {"prodx": [], "prody": [], "prodz": [], "decayx": [], "decayy": [], "decayz": [], "gap": []}
for datavecs, datalens, labels in dataloader:
    if batch_counter >= max_batch and max_batch != -1:
        break
    if batch_counter % 100 == 0:
        print("Batch", batch_counter, " of ", len(dataloader))
    labels = labels.cpu()
    prodx = labels[0]
    prody = labels[1]
    prodz = labels[2]
    decayx = labels[3]
    decayy = labels[4]
    decayz = labels[5]
    gap_len = distance(prodx, prody, prodz, decayx, decayy, decayz)
    labels_dict["prodx"] += prodx.tolist()
    labels_dict["prody"] += prody.tolist()
    labels_dict["prodz"] += prodz.tolist()
    labels_dict["decayx"] += decayx.tolist()
    labels_dict["decayy"] += decayy.tolist()
    labels_dict["decayz"] += decayz.tolist()
    labels_dict["gap"] += gap_len.tolist()
    batch_counter += 1

# histogram performance
print("Creating histograms")
fig, axs = plt.subplots(3, 3, figsize=(12, 12))
keys = list(labels_dict.keys())
for i in range(3):
    for j in range(3):
        tile_num = i*3 + j
        if tile_num >= len(keys):
            break
        axs[i, j].hist(labels_dict[keys[tile_num]], bins=20)
        axs[i, j].set_title(keys[tile_num] + ": mean {:.2f}".format(np.mean(labels_dict[keys[tile_num]])))
        # axs[i, j].set_xlabel('Units')
        # axs[i, j].set_ylabel('Count')
        
plt.tight_layout()
plt.show()