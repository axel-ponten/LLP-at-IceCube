from llp_gap_reco.dataset import LLPDataset, LLPSubset, llp_collate_fn
from llp_gap_reco.encoder import LLPTransformerModel
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

def distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)


def unnormalize_hits(hits, normalization_args):
    """ Normalize the data. x = (x-offset)*scale"""
    # for each feature type ("log_charges", "position", "abs_time", etc.)
    if normalization_args["position"]["scale"] != 1.0:
        hits /= normalization_args["position"]["scale"]
    if normalization_args["position"]["offset"] != 0.0:
        hits += normalization_args["position"]["offset"]
    return hits

def plot_event(data, label, prediction, normalization_args = None):
    # List of 3D positions
    hits = data.squeeze()[:,:3].cpu().numpy()
    charges = data.squeeze()[:,3].cpu().numpy().tolist()
    
    # unnormalize hits for plotting
    if normalization_args is not None:
        hits = unnormalize_hits(hits, normalization_args)
    
    x = [pos[0] for pos in hits]
    y = [pos[1] for pos in hits]
    z = [pos[2] for pos in hits]
    scale = 2
    s = [scale*(10**charge-1) for charge in charges] # total charge from log(1+charge)

    # label
    label = label.cpu().numpy().tolist()
    prod_x = label[0]
    prod_y = label[1]
    prod_z = label[2]
    decay_x = label[3]
    decay_y = label[4]
    decay_z = label[5]

    # prediction
    prediction = prediction.cpu().numpy().tolist()
    pred_prod_x = prediction[0]
    pred_prod_y = prediction[1]
    pred_prod_z = prediction[2]
    pred_decay_x = prediction[3]
    pred_decay_y = prediction[4]
    pred_decay_z = prediction[5]

    # Create a 3D plot
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection='3d')

    # plot hits
    ax.scatter(x, y, z, s=s)
    
    # plot label and draw line between
    ax.scatter(prod_x, prod_y, prod_z, color='red')
    ax.scatter(decay_x, decay_y, decay_z, color='blue')
    label_length = distance(prod_x, prod_y, prod_z, decay_x, decay_y, decay_z)
    ax.plot([prod_x, decay_x], [prod_y, decay_y], [prod_z, decay_z], color='red', label='label: {:.2f} m'.format(label_length))
    
    # plot prediction and draw line between
    ax.scatter(pred_prod_x, pred_prod_y, pred_prod_z, color='red')
    ax.scatter(pred_decay_x, pred_decay_y, pred_decay_z, color='blue')
    pred_length = distance(pred_prod_x, pred_prod_y, pred_prod_z, pred_decay_x, pred_decay_y, pred_decay_z)
    ax.plot([pred_prod_x, pred_decay_x], [pred_prod_y, pred_decay_y], [pred_prod_z, pred_decay_z], color='black', label='pred: {:.2f} m'.format(pred_length))

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Red: Production vertex, Blue: Decay vertex')

    # Set all axes limits
    xmin = -600
    xmax = 600
    ymin = -600
    ymax = 600
    zmin = -600
    zmax = 600
    
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])

    ax.legend()
    
    return



def plot_2x2_fn(data_list, label_list, prediction_list, normalization_args = None):
    if len(data_list) != len(label_list) or len(data_list) != len(prediction_list):
        raise ValueError("data, label and prediction must have the same length")
    if len(data_list) != 4:
        raise ValueError("data must have 4 events")
    
    
    # Create a 3D plot
    fig = plt.figure(figsize=(14,14))  # Set the figure size to 10x8 inches
    plt.title('Red: Production vertex, Blue: Decay vertex')
    plt.subplots_adjust(hspace=0., wspace=0., left=0., right=0.9, top=0.9, bottom=0.)
    # plt.tight_layout()
    # add subplots
    for i, (data, label, prediction) in enumerate(zip(data_list, label_list, prediction_list)):
        ax = fig.add_subplot(221+i, projection='3d')
        
        # List of 3D positions
        hits = data.squeeze()[:,:3].cpu().numpy()
        charges = data.squeeze()[:,3].cpu().numpy().tolist()
        
        # unnormalize hits for plotting
        if normalization_args is not None:
            hits = unnormalize_hits(hits, normalization_args)
        
        x = [pos[0] for pos in hits]
        y = [pos[1] for pos in hits]
        z = [pos[2] for pos in hits]
        scale = 1
        s = [scale*(10**charge-1) for charge in charges] # total charge from log(1+charge)
            
        # label
        label = label.cpu().numpy().tolist()
        prod_x = label[0]
        prod_y = label[1]
        prod_z = label[2]
        decay_x = label[3]
        decay_y = label[4]
        decay_z = label[5]

        # prediction
        prediction = prediction.cpu().numpy().tolist()
        pred_prod_x = prediction[0]
        pred_prod_y = prediction[1]
        pred_prod_z = prediction[2]
        pred_decay_x = prediction[3]
        pred_decay_y = prediction[4]
        pred_decay_z = prediction[5]

        # plot hits
        ax.scatter(x, y, z, s=s)
        
        # plot label and draw line between
        ax.scatter(prod_x, prod_y, prod_z, color='red')
        ax.scatter(decay_x, decay_y, decay_z, color='blue')
        label_length = distance(prod_x, prod_y, prod_z, decay_x, decay_y, decay_z)
        ax.plot([prod_x, decay_x], [prod_y, decay_y], [prod_z, decay_z], color='red', label='label: {:.2f} m'.format(label_length))
        
        # plot prediction and draw line between
        ax.scatter(pred_prod_x, pred_prod_y, pred_prod_z, color='red')
        ax.scatter(pred_decay_x, pred_decay_y, pred_decay_z, color='blue')
        pred_length = distance(pred_prod_x, pred_prod_y, pred_prod_z, pred_decay_x, pred_decay_y, pred_decay_z)
        ax.plot([pred_prod_x, pred_decay_x], [pred_prod_y, pred_decay_y], [pred_prod_z, pred_decay_z], color='black', label='pred: {:.2f} m'.format(pred_length))

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set all axes limits
        xmin = -600
        xmax = 600
        ymin = -600
        ymax = 600
        zmin = -600
        zmax = 600
        
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_zlim([zmin, zmax])

        ax.legend()


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
    parser.add_argument('--modelconfig', type=str, help='Path of the model config')
    parser.add_argument('--normpath', type=str, help='Path to the normalization arguments file')
    parser.add_argument('--plot2x2', action=argparse.BooleanOptionalAction, default=False, help='Plot 2x2 plots')
    
    # Parse the arguments
    args = parser.parse_args()

    # Get the values from the arguments
    top_folder = args.topfolder
    filename_start = args.filenamestart
    nevents = args.nevents
    shuffle_files = args.shuffle
    predict_train = args.predicttrain
    model_path = args.modelpath
    config_path = args.modelconfig
    norm_path = args.normpath
    plot_2x2 = args.plot2x2

    # create model
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    kwargs_dict = config["settings"]
    model = LLPTransformerModel(**kwargs_dict)
    model.to("cuda")
    model.eval()
    model.load_state_dict(torch.load(model_path))

    # input checks
    if nevents < -1:
        raise ValueError("nevents must be -1 or greater")
    # add trailing slash to top folder
    if top_folder[-1] != "/":
        top_folder += "/"

    # filepaths
    index_file_path = top_folder + "indexfile.pq"
    total_index_info = pd.read_parquet(index_file_path)
    feature_indices_file_path = top_folder + "feature_indices.yaml"
    file_paths = glob.glob(top_folder + filename_start + "*.pq")
    
    # normalizaton args
    with open(norm_path, "r") as file:
        normalization_args = yaml.safe_load(file)
    
    # create dataset
    dataset = LLPDataset(
        index_file_path,
        file_paths,
        feature_indices_file_path,
        normalize_data=True,
        normalize_target=False,
        normalization_args=normalization_args,
        device="cuda",
        dtype=torch.float32,
        shuffle_files=shuffle_files,
    )

    # split dataset into train and test
    nfiles = len(file_paths)
    nfiles_train = int(0.8*nfiles)
    nfiles_test = nfiles - nfiles_train
    events_per_file = len(dataset)//nfiles
    train_size = int(nfiles_train*events_per_file)
    test_size = len(dataset) - train_size
    print("#####################")
    print("Dataset info:")
    print("Events per .pq file", events_per_file)
    print("Nfiles train/test", nfiles_train, nfiles_test)
    print("Train size:", train_size)
    print("Test size:", test_size)
    print("Percentage of train data:", train_size/len(dataset)*100.0, "%")
    print("#####################")
    # Created using indices from 0 to train_size.
    train_dataset = LLPSubset(dataset, range(train_size))
    # Created using indices from train_size to train_size + test_size.
    test_dataset = LLPSubset(dataset, range(train_size, train_size + test_size))

    # dataloader
    batch_size = 4
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=llp_collate_fn)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=llp_collate_fn)
    # visualize train or test?
    if predict_train:
        dataloader = trainloader
    else:
        dataloader = testloader

    # define a function to plot events
    def plot_loop(dataloader, nevents, model, normalization_args):
        counter = 0
        for datavecs, datalens, labels in dataloader:
            with torch.no_grad():
                predictions = model(datavecs, datalens)
            for i in range(len(datavecs)):
                if counter == nevents:
                    return
                data = datavecs[i]
                label = labels[i]
                prediction = predictions[i]
                plot_event(data, label, prediction, normalization_args)
                counter += 1
    
    # define a function to plot events as 2x2
    def plot2x2_loop(dataloader, nevents, model, normalization_args):
        counter = 0
        for datavecs, datalens, labels in dataloader:
            if counter == nevents:
                return
            with torch.no_grad():
                predictions = model(datavecs, datalens)
            assert len(datavecs) == 4
            plot_2x2_fn(datavecs, labels, predictions, normalization_args)
            counter += 1
                
    if plot_2x2:
        plot2x2_loop(dataloader, nevents, model, normalization_args)
    else:
        plot_loop(dataloader, nevents, model, normalization_args)
    plt.show()
