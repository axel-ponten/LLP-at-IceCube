import sys
sys.path.append("../")
from llp_gap_reco.dataset import LLPDataset, llp_collate_fn
from torch.utils.data import DataLoader
import yaml
import torch
import glob
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# filepaths
top_folder = "/home/axel/i3/i3-pq-conversion-files/DLS-115-5e-6/"
index_file_path = top_folder + "indexfile.pq"
total_index_info = pd.read_parquet(index_file_path)
feature_indices_file_path = top_folder + "feature_indices.yaml"
file_paths = glob.glob(top_folder + "L2*.pq")

# create dataset
dataset = LLPDataset(
    index_file_path,
    file_paths,
    feature_indices_file_path,
    normalize_data=False,
    device="cuda",
    dtype=torch.float32,
    shuffle_files=True,
)

# get event
for i in range(5):
    data, label = dataset[i]

    # List of 3D positions
    # positions = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    hits = data[:, :4].tolist()
    x = [pos[0] for pos in hits]
    y = [pos[1] for pos in hits]
    z = [pos[2] for pos in hits]
    s = [10**pos[3]-1 for pos in hits]
    # label
    label = label.cpu().numpy().tolist()
    prod_x = label[0]
    prod_y = label[1]
    prod_z = label[2]
    decay_x = label[3]
    decay_y = label[4]
    decay_z = label[5]

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))  # Set the figure size to 10x8 inches
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

    
    # Get the current axes limits
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()

    # Find the largest limit
    largest_limit = max(max(x_limits), max(y_limits), max(z_limits))

    # Set all axes limits to be the largest limit
    ax.set_xlim([-largest_limit, largest_limit])
    ax.set_ylim([-largest_limit, largest_limit])
    ax.set_zlim([-largest_limit, largest_limit])

    # Show the plot
    plt.show()
