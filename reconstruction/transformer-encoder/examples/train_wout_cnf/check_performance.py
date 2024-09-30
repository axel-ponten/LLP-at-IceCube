""" script to check performance of model on test data. """

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
import numpy as np
import corner

def distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

def angular_difference_from_points(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4):
    # calculate line 1
    dx1, dy1, dz1 = calculate_line(x1, y1, z1, x2, y2, z2)
    # calculate line 2
    dx2, dy2, dz2 = calculate_line(x3, y3, z3, x4, y4, z4)
    # find angles
    phi1, theta1 = find_angles(dx1, dy1, dz1)
    phi2, theta2 = find_angles(dx2, dy2, dz2)
    # calculate angular difference
    return angular_difference(phi1, theta1, phi2, theta2)

def calculate_line(x1, y1, z1, x2, y2, z2):
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    return dx, dy, dz

# Find angles phi and theta
def find_angles(dx, dy, dz):
    phi = np.arctan2(dy, dx)
    theta = np.arctan2(np.sqrt(dx**2 + dy**2), dz)
    return phi, theta

def angular_difference(phi1, eta1, phi2, eta2):
    dphi = np.abs(phi1 - phi2)
    if dphi > np.pi:
        dphi = 2*np.pi - dphi
    deta = np.abs(eta1 - eta2)
    return np.sqrt(dphi**2 + deta**2)

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Plot some events.')

    # Add the arguments
    parser.add_argument('--topfolder', type=str, help='Path to the top folder')
    parser.add_argument('--filenamestart', type=str, default='base', help='Start of the filename for glob')
    parser.add_argument('--nevents', type=int, default=5, help='Number of events. -1 for all events')
    parser.add_argument('--predicttrain', action=argparse.BooleanOptionalAction, default=False, help='Predict training set')
    parser.add_argument('--modelpath', type=str, help='Path to the model file')
    parser.add_argument('--modelconfig', type=str, help='Path of the model config')
    parser.add_argument('--normpath', type=str, help='Path to the normalization arguments file')
    
    # Parse the arguments
    args = parser.parse_args()

    # Get the values from the arguments
    top_folder = args.topfolder
    filename_start = args.filenamestart
    nevents = args.nevents
    predict_train = args.predicttrain
    model_path = args.modelpath
    config_path = args.modelconfig
    norm_path = args.normpath

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
        shuffle_files=False,
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
    batch_size = 32
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=llp_collate_fn)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=llp_collate_fn)
    # visualize train or test?
    if predict_train:
        dataloader = trainloader
    else:
        dataloader = testloader

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
    batch_counter = 0
    max_batch = -1
    for datavecs, datalens, labels in dataloader:
        if batch_counter >= max_batch and max_batch != -1:
            break
        with torch.no_grad():
            predictions = model(datavecs, datalens)
            # compute performance criteria
            for p, l in zip(predictions, labels):
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
                # calculate differences
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
    ###
    

# Create a list of all performance variables
performance_variables = [
    # "angular_diff",
    # "gap_diff",
    "prodx_diff",
    "prody_diff",
    "prodz_diff",
    "decayx_diff",
    "decayy_diff",
    "decayz_diff",
    # "prod_diff",
    # "decay_diff",
    # "MSE"
]

# Create a list of performance variable values
performance_values = [performance_dict[var] for var in performance_variables]

# Plot the histogram triangle
figure = corner.corner(
    np.transpose(performance_values),
    labels=performance_variables,
    show_titles=True,
    title_fmt=".2f",
    plot_contours=True,
    bins=20,
    smooth=True,
    # axes_scale="log",
)

# histogram performance
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
plt.show()
