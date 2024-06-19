""" Standard pytorch training loop.
No conditional normalizing flow, only transformer encoder.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import yaml
import glob
import os
import argparse
import time

from llp_gap_reco.encoder import LLPTransformerModel
from llp_gap_reco.dataset import LLPDataset, LLPSubset, llp_collate_fn


###### GET DATASET #######

parser = argparse.ArgumentParser(description='Training arguments')

# Add arguments
parser.add_argument('--topfolder', type=str, help='Path to the top folder')
parser.add_argument('--normpath', type=str, help='Path to the normalization arguments file')
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs')
parser.add_argument('--batchsize', type=int, default=32, help='Batch size')
parser.add_argument('--learningrate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--modelspath', type=str, help='Path to the models directory')
parser.add_argument('--configpath', type=str, help='Path to the config file')
parser.add_argument('--filenamestart', type=str, default="base", help='For glob. How does filename start?')
parser.add_argument('--doplots', type=bool, default=True, help='Plot loss?')

# Parse arguments
args = parser.parse_args()

# Access the arguments
top_folder = args.topfolder
norm_path = args.normpath
n_epochs = args.epochs
batch_size = args.batchsize
learning_rate = args.learningrate
models_path = args.modelspath
config_path = args.configpath
filename_start = args.filenamestart
do_plots = args.doplots

# add trailing slash to top folder and models path
if top_folder[-1] != "/":
    top_folder += "/"
if models_path[-1] != "/":
    models_path += "/"

# create model dir if it does not exist
if not os.path.exists(models_path):
    os.makedirs(models_path)
else:
    # don't lose your precious models!
    if len(os.listdir(models_path)) > 5:
        print("Warning: model directory already exists. Rename it to avoid overwriting models.")
        exit()

# copy model config file to models directory
os.system("cp " + config_path + " " + models_path)

# filepaths
index_file_path = top_folder + "indexfile.pq"
feature_indices_file_path = top_folder + "feature_indices.yaml"
file_paths = glob.glob(top_folder + filename_start + "*.pq")

# normalizaton args
with open(norm_path, "r") as file:
    normalization_args = yaml.safe_load(file)

dataset = LLPDataset(
    index_file_path,
    file_paths,
    feature_indices_file_path,
    normalize_data=True,
    normalize_target=False,
    normalization_args=normalization_args,
    device="cuda",
    dtype=torch.float32,
    shuffle_files=True,
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
print("Batch size:", batch_size)
print("#####################")
# Created using indices from 0 to train_size.
# train_dataset = torch.utils.data.Subset(dataset, range(train_size))
train_dataset = LLPSubset(dataset, range(train_size))
# Created using indices from train_size to train_size + test_size.
# test_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))
test_dataset = LLPSubset(dataset, range(train_size, train_size + test_size))

# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# dataloader
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=llp_collate_fn)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=llp_collate_fn)

####### CREATE MODEL #######
# model settings
with open(config_path, 'r') as stream:
    config = yaml.safe_load(stream)
kwargs_dict = config["settings"]

# create transformer encoder and cond. normalizing flow
model = LLPTransformerModel(**kwargs_dict)
model.to('cuda')
print("Transformer model:", model)
total_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
print("Trainable transformer encoder parameters:", total_params)

############################

####### LOSS AND OPTIMIZER #######
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

####### TRAIN #######
print("Starting training loop with {} epochs".format(n_epochs))
print("#####################################")
train_loss_vals = []
test_loss_vals = []
for epoch in range(n_epochs):
    # shuffle data every epoch
    trainloader.dataset.shuffle()
    model.train()
    train_loss = 0
    start_time = time.time()  # Start the timer for the epoch
    for i, (batch_input, batch_lens, batch_label) in enumerate(trainloader):
        if i%1000 == 0:
            print("Batch", str(i) + "/" + str(train_size//batch_size), " of epoch", epoch + 1, "of", n_epochs, "epochs")
        # reset gradients
        optimizer.zero_grad()
        # propagate input
        nn_output = model(batch_input, batch_lens)
        # compute loss
        loss = criterion(nn_output, batch_label)
        # compute gradient
        loss.backward()
        # update weights
        optimizer.step()
        # add loss
        train_loss += loss.item()
        # print status
        if i%1000 == 0:
            model.eval()
            with torch.no_grad():
                nn_output = model(batch_input, batch_lens)
                print("Some predictions/labels:")
                outputs = nn_output[0:2].tolist()
                labels = batch_label[0:2].tolist()
                for a, b in zip(outputs, labels):
                    print("Pred", a)
                    print("True", b)
                # for name, param in model.named_parameters():
                #     print(name, param.grad.abs().sum())
            model.train()
    ###
    # train loss for the epoch
    train_loss = train_loss*batch_size/train_size  # Calculate the average train loss
    train_loss_vals.append(train_loss)

    # test loss for the epoch
    test_loss = 0.0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for batch_input, batch_lens, batch_label in testloader:
            nn_output = model(batch_input, batch_lens)
            loss = criterion(nn_output, batch_label)
            test_loss += loss.item()

    test_loss = test_loss*batch_size/test_size  # Calculate the average test loss
    test_loss_vals.append(test_loss)
    print("Last test output:", nn_output[0:3])
    print("Last test label:", batch_label[0:3])

    end_time = time.time()  # Stop the timer for the epoch
    epoch_time = end_time - start_time  # Calculate the time taken for the epoch
    print("Epoch", epoch + 1, "in", epoch_time, "s. Train loss", train_loss, ": Test loss", test_loss)
    print("#####################################")
    
    # save losses to csv
    df = pd.DataFrame({"train_loss": train_loss_vals, "test_loss": test_loss_vals})
    df.to_csv(models_path + "loss_no_cnf.csv")
    
    # save model every 5 epochs
    if epoch % 5 == 0:
        model_path = models_path + "model_no_cnf_epoch_{}.pth".format(epoch)
        torch.save(model.state_dict(), model_path)

####### SAVE FINAL MODEL #######
model_path = models_path + "model_no_cnf_final.pth"
torch.save(model.state_dict(), model_path)

####### PLOT LOSS #######
if do_plots:
    plt.figure()
    plt.plot(train_loss_vals, label="train")
    plt.plot(test_loss_vals, label="test")
    plt.legend()
    plt.title("loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.savefig(models_path + "loss_no_cnf.png")
