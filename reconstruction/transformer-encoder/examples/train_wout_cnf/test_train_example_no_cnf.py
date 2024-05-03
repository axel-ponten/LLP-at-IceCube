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

import sys
sys.path.append("../../")
from llp_gap_reco.encoder import LLPTransformerModel
from llp_gap_reco.dataset import LLPDataset, llp_collate_fn

###### GET DATASET #######

# filepaths
top_folder = "/home/axel/i3/i3-pq-conversion-files/DLS-115-5e-6/"
index_file_path = top_folder + "indexfile.pq"
feature_indices_file_path = top_folder + "feature_indices.yaml"
file_paths = glob.glob(top_folder + "L2*.pq")

# normalizaton args
norm_path = "/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/normalization_args.yaml"
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
    shuffle_files=True,
)

# split dataset into train and test

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
print("Train size:", train_size)
print("Test size:", test_size)
# Created using indices from 0 to train_size.
train_dataset = torch.utils.data.Subset(dataset, range(train_size))

# Created using indices from train_size to train_size + test_size.
test_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))

# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# dataloader
batch_size = 32
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=llp_collate_fn)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=llp_collate_fn)
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=llp_collate_fn)

####### CREATE MODEL #######
# model settings
config_path = "../configs/test_settings_no_cnf.yaml"
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
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

####### TRAIN #######
n_epochs = 400
print("Starting training loop with {} epochs".format(n_epochs))
train_loss_vals = []
test_loss_vals = []
for epoch in range(n_epochs):
    model.train()
    train_loss = 0
    for i, (batch_input, batch_lens, batch_label) in enumerate(trainloader):
        if i%1000 == 0:
            print("Batch", i, "of epoch", epoch + 1, "of", n_epochs, "epochs")

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
                print("some output", nn_output[0:3])
                print("some label", batch_label[0:3])
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

    print("Epoch", epoch + 1, "train loss", train_loss, ": test loss", test_loss)

    # save model every 5 epochs
    if epoch % 5 == 0:
        model_path = "models/model_no_cnf_epoch_{}.pth".format(epoch)
        torch.save(model.state_dict(), model_path)

####### SAVE FINAL MODEL #######
model_path = "models/model_no_cnf_final.pth"
torch.save(model.state_dict(), model_path)

# save losses to csv
df = pd.DataFrame({"train_loss": train_loss_vals, "test_loss": test_loss_vals})
df.to_csv("models/loss_no_cnf.csv")

####### PLOT LOSS #######
plt.figure()
plt.plot(train_loss_vals, label="train")
plt.plot(test_loss_vals, label="test")
plt.legend()
plt.title("loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.savefig("models/loss_no_cnf.png")