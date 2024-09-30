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
batch_size = 8
dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=llp_collate_fn)

##### PERFORMANCE #####
# create folder
model_folder = os.path.dirname(model_path)
performance_folder = model_folder + "/performance/"
if not os.path.exists(performance_folder):
    os.mkdir(performance_folder)

# variables
performance_dict = {
    "angular_diff": [],
    "gap_diff": [],
    "prodx_diff": [],
    "prody_diff": [],
    "prodz_diff": [],
    "decayx_diff": [],
    "decayy_diff": [],
    "decayz_diff": [],
    "prod_diff": [],
    "decay_diff": [],
    "MSE": [],
}

# fill performance dict
batch_counter = 0
max_batch = -1
samplesize = 100
print("Starting performance evaluation")
for datavecs, datalens, labels in dataloader:
    if batch_counter % 100 == 0:
        print("Batch", batch_counter, " of ", len(dataloader))
    if batch_counter >= max_batch and max_batch != -1:
        break
    with torch.no_grad():
        predictions, pred_std = predict_cnf(model, pdf, datavecs, datalens, samplesize=samplesize)
        print(datalens)
        print(labels)
        # compute performance criteria
        for p, l, hits in zip(predictions, labels, datalens):
            # unpack pred
            pred_prod_x = p[0].item()
            pred_prod_y = p[1].item()
            pred_prod_z = p[2].item()
            pred_decay_x = p[3].item()
            pred_decay_y = p[4].item()
            pred_decay_z = p[5].item()
            # unpack true
            true_prod_x = l[0].item()
            true_prod_y = l[1].item()
            true_prod_z = l[2].item()
            true_decay_x = l[3].item()
            true_decay_y = l[4].item()
            true_decay_z = l[5].item()
            # calculate performance metrics
            performance_dict["angular_diff"].append(angular_difference_from_points(pred_prod_x, pred_prod_y, pred_prod_z, pred_decay_x, pred_decay_y, pred_decay_z, true_prod_x, true_prod_y, true_prod_z, true_decay_x, true_decay_y, true_decay_z))
            performance_dict["gap_diff"].append(distance(pred_prod_x, pred_prod_y, pred_prod_z, pred_decay_x, pred_decay_y, pred_decay_z) - distance(true_prod_x, true_prod_y, true_prod_z, true_decay_x, true_decay_y, true_decay_z))
            performance_dict["prodx_diff"].append(pred_prod_x - true_prod_x)
            performance_dict["prody_diff"].append(pred_prod_y - true_prod_y)
            performance_dict["prodz_diff"].append(pred_prod_z - true_prod_z)
            performance_dict["decayx_diff"].append(pred_decay_x - true_decay_x)
            performance_dict["decayy_diff"].append(pred_decay_y - true_decay_y)
            performance_dict["decayz_diff"].append(pred_decay_z - true_decay_z)
            performance_dict["prod_diff"].append(distance(pred_prod_x, pred_prod_y, pred_prod_z, true_prod_x, true_prod_y, true_prod_z))
            performance_dict["decay_diff"].append(distance(pred_decay_x, pred_decay_y, pred_decay_z, true_decay_x, true_decay_y, true_decay_z))
            performance_dict["MSE"].append((pred_prod_x - true_prod_x)**2 + (pred_prod_y - true_prod_y)**2 + (pred_prod_z - true_prod_z)**2 + (pred_decay_x - true_decay_x)**2 + (pred_decay_y - true_decay_y)**2 + (pred_decay_z - true_decay_z)**2)
    batch_counter += 1

# save performance dict
df = pd.DataFrame(performance_dict)
print("Saving performance metrics")
df.to_csv(performance_folder + "performance.csv")

# histogram performance
print("Creating histograms")
fig, axs = plt.subplots(3, 4, figsize=(12, 12))
keys = list(performance_dict.keys())
for i in range(3):
    for j in range(4):
        tile_num = i*4 + j
        if tile_num >= len(keys):
            break
        axs[i, j].hist(performance_dict[keys[tile_num]], bins=20)
        axs[i, j].set_title(keys[tile_num] + ": mean {:.2f}".format(np.mean(performance_dict[keys[tile_num]])))
        # axs[i, j].set_xlabel('Units')
        # axs[i, j].set_ylabel('Count')
        
plt.tight_layout()
plt.savefig(performance_folder + "performance_histograms.png")
plt.show()

### CORNER PLOT ###
corner_variables = [
    "prodx_diff",
    "prody_diff",
    "prodz_diff",
    "decayx_diff",
    "decayy_diff",
    "decayz_diff",
]

# Create a list of performance variable values
performance_values = [performance_dict[var] for var in corner_variables]

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
plt.savefig(performance_folder + "corner_plot.png")

##### PLOT EVENT EXAMPLES #####
# print("Creating example reconstructions")
# with open(norm_path, "r") as file:
#     normalization_args = yaml.safe_load(file)
# dataloader.dataset.shuffle() # shuffle
# # grab one batch
# datavecs, datalens, labels = next(iter(dataloader))
# with torch.no_grad():
#     predictions, pred_std = predict_cnf(model, pdf, datavecs, datalens, samplesize=300)
#     nn_output = model(datavecs, datalens)
#     nn_output = nn_output.double()
#     target_sample, base_sample, _, _ = pdf.sample(conditional_input=nn_output, samplesize=300)
#     print(target_sample)
#     print(base_sample)
# for i in range(len(datavecs)):
#     plot_event(datavecs[i], predictions[i], labels[i], normalization_args)
#     plt.savefig(performance_folder + "example_reconstructions_{}.png".format(i))
#     plt.close()
