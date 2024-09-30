""" Takes a network and dataset and saves pdf info to csv. """


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

# get folder
trainfolder = os.path.dirname(model_path)

# create model
model, pdf = create_full_model(config_path, device="cuda")
model.eval()
pdf.eval()
model.load_state_dict(torch.load(model_path))
pdf.load_state_dict(torch.load(flow_path))

pdfdir = dir(pdf)
for x in pdfdir:
    print(x)
# create dataloader
train_dataset, test_dataset = create_split_datasets(top_folder, filename_start, norm_path, split=0.8)
batch_size = 1
dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=llp_collate_fn)



def get_pdf_info(pdf, nn_output):

    return ypred_dict

    # print("Sampling")
    # for i in range(samplesize):
    #     target_sample, _, _, _ = pdf.sample(conditional_input=nn_output)
    #     samples.append(target_sample.cpu())


##### PERFORMANCE #####

# @TODO: can i trust that it's ordered?

# grab one batch
datavecs, datalens, labels = next(iter(dataloader))
# grab one batch manually
datavecs_manual = []
labels_manual = []
title_list = []
# do we just change to iterate over the full dataset?
# still batch it for faster computation
nevents = 500
batch_counter = 0
item_counter = 0
batch_list = []
for i in range(nevents):
    datavecs_manual = []
    labels_manual = []
    if item_counter <= batch_size:
        data, label = test_dataset.__getitem__(i)
        mu_E, mu_Z, mu_L = test_dataset.get_muon_info_from_idx(i)
        #event_id, run_id = test_dataset.get_event_run_info(i) # maybe we can just grab the total index file directly?
        datavecs_manual.append(data)
        labels_manual.append(label)
        title_list.append("Event {}: E: {:.2f}, Z: {:.2f}, L: {:.2f}".format(i, mu_E, mu_Z, mu_L))

# for corner plot
samplesize = 500
with torch.no_grad():
    print("Predicting")
    nn_output = model(datavecs, datalens)
    nn_output = nn_output.double()
    # info
    ypred_dict = pdf.marginal_moments(nn_output,samplesize=300)
    for key, value in ypred_dict.items():
        print(key)
        print(value)
    ypred_dict = pdf.marginal_moments(nn_output,samplesize=3000)
    for key, value in ypred_dict.items():
        print(key)
        print(value)
    ypred_dict = pdf.marginal_moments(nn_output,samplesize=30000)
    for key, value in ypred_dict.items():
        print(key)
        print(value)
    predictions, pred_std = predict_cnf(model, pdf, datavecs, datalens, samplesize=samplesize)
