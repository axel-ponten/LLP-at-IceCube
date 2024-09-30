from llp_gap_reco.dataset import LLPDataset, UnlabeledLLPDataset, LLPSubset, llp_collate_fn, llp_collate_unlabeled_fn
from llp_gap_reco.encoder import LLPTransformerModel
from llp_gap_reco.training.utils import *
from llp_gap_reco.training.performance import plot_event, plot_2x2_fn
import jammy_flows
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


# define a function to plot events
def plot_loop(dataloader, nevents, model, pdf, normalization_args):
    counter = 0
    for datavecs, datalens, labels in dataloader:
        with torch.no_grad():
            pred_mean, pred_std = predict_cnf(model, pdf, datavecs, datalens, samplesize=300)
        for i in range(len(datavecs)):
            if counter == nevents:
                return
            data = datavecs[i]
            label = labels[i]
            prediction = pred_mean[i]
            plot_event(data, prediction, label, normalization_args)
            counter += 1

# define a function to plot events as 2x2
def plot2x2_loop(dataloader, nevents, model, pdf, normalization_args):
    counter = 0
    for datavecs, datalens, labels in dataloader:
        if counter == nevents:
            return
        with torch.no_grad():
            pred_mean, pred_std = predict_cnf(model, pdf, datavecs, datalens, samplesize=300)
        assert len(datavecs) == 4
        plot_2x2_fn(datavecs, pred_mean, labels, normalization_args)
        counter += 1

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Plot some events.')

    # Add the arguments
    parser.add_argument('--topfolder', type=str, help='Path to the top folder')
    parser.add_argument('--filenamestart', type=str, default='base', help='Start of the filename for glob')
    parser.add_argument('--nevents', type=int, default=5, help='Number of events. -1 for all events')
    parser.add_argument('--shuffle', action=argparse.BooleanOptionalAction, default=False, help='Shuffle files')
    parser.add_argument('--predicttrain', action=argparse.BooleanOptionalAction, default=False, help='Predict training set')
    parser.add_argument('--modelpath', type=str, help='Path to the model file')
    parser.add_argument('--flowpath', type=str, help='Path to the flow file')
    parser.add_argument('--modelconfig', type=str, help='Path of the model config')
    parser.add_argument('--normpath', type=str, help='Path to the normalization arguments file')
    parser.add_argument('--plot2x2', action=argparse.BooleanOptionalAction, default=False, help='Plot 2x2 plots')
    parser.add_argument('--unlabeled', action=argparse.BooleanOptionalAction, default=False, help='Unlabeled data?')
    
    # Parse the arguments
    args = parser.parse_args()

    # Get the values from the arguments
    top_folder = args.topfolder
    filename_start = args.filenamestart
    nevents = args.nevents
    shuffle_files = args.shuffle
    predict_train = args.predicttrain
    model_path = args.modelpath
    flow_path = args.flowpath
    config_path = args.modelconfig
    norm_path = args.normpath
    plot_2x2 = args.plot2x2
    is_unlabeled = args.unlabeled

    # create model
    model, pdf = create_full_model(config_path, device="cuda")
    model.eval()
    pdf.eval()
    model.load_state_dict(torch.load(model_path))
    pdf.load_state_dict(torch.load(flow_path))

    # create dataset
    batch_size = 4
    # normalizaton args
    with open(norm_path, "r") as file:
        normalization_args = yaml.safe_load(file)
    # unlabeled or labeled data?
    if is_unlabeled:
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
            shuffle_files=shuffle_files,
        )
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=llp_collate_unlabeled_fn)
    else:
        # labeled data
        train_dataset, test_dataset = create_split_datasets(top_folder, filename_start, norm_path, split=0.8)
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=llp_collate_fn)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=llp_collate_fn)
        # visualize train or test?
        if predict_train:
            dataloader = trainloader
        else:
            dataloader = testloader

    if plot_2x2:
        print("Starting 2x2 plot")
        plot2x2_loop(dataloader, nevents, model, pdf, normalization_args)
    else:
        print("Starting single plot")
        plot_loop(dataloader, nevents, model, pdf, normalization_args)
    plt.show()
