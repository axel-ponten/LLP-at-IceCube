from llp_gap_reco.dataset import LLPDataset, llp_collate_fn
from torch.utils.data import DataLoader
import yaml
import torch
import glob
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse

def plot_event(data, label):
    # List of 3D positions
    hits = data.squeeze()[:,:4].cpu().numpy().tolist()
    
    x = [pos[0] for pos in hits]
    y = [pos[1] for pos in hits]
    z = [pos[2] for pos in hits]
    s = [10**pos[3]-1 for pos in hits] # total charge from log(1+charge)
    # label
    label = label.cpu().numpy().tolist()
    prod_x = label[0]
    prod_y = label[1]
    prod_z = label[2]
    decay_x = label[3]
    decay_y = label[4]
    decay_z = label[5]

    # Create a 3D plot
    fig = plt.figure(figsize=(12,12))  # Set the figure size to 10x8 inches
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=s)
    ax.scatter(prod_x, prod_y, prod_z, color='red')
    ax.scatter(decay_x, decay_y, decay_z, color='green')

    # Plot a cylinder
    height = 1000
    radius = 500
    resolution = 100
    theta = np.linspace(0, 2 * np.pi, resolution)
    z_cylinder = np.linspace(-height/2, height/2, resolution)
    theta_grid, z_grid = np.meshgrid(theta, z_cylinder)
    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.1, color='b')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Red = prod, Green = decay')

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

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Plot some events.')

    # Add the arguments
    parser.add_argument('--top_folder', type=str, help='Path to the top folder')
    parser.add_argument('--filename_start', type=str, default='base', help='Start of the filename for glob')
    parser.add_argument('--nevents', type=int, default=5, help='Number of events. -1 for all events')
    parser.add_argument('--shuffle', type=bool, default=False, help='Shuffle the events')
    
    # Parse the arguments
    args = parser.parse_args()

    # Get the values from the arguments
    top_folder = args.top_folder
    filename_start = args.filename_start
    nevents = args.nevents
    shuffle_files = args.shuffle

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

    # create dataset
    dataset = LLPDataset(
        index_file_path,
        file_paths,
        feature_indices_file_path,
        normalize_data=False,
        device="cuda",
        dtype=torch.float32,
        shuffle_files=shuffle_files,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=llp_collate_fn,
    )
    
    # define a function to plot events
    def plot_loop(dataloader, nevents):
        counter = 0
        for datavecs, datalens, labels in dataloader:
            for i in range(len(datavecs)):
                if counter == nevents:
                    return
                data = datavecs[i]
                label = labels[i]
                plot_event(data, label)
                counter += 1
    
    # plot events
    plot_loop(dataloader, nevents)
