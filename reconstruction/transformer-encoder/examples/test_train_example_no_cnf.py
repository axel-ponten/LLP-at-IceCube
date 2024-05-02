""" Standard pytorch training loop.
    Example with random input of varying seq. length
    and random labels.
    Just to test that things are working.
    Transformer encoder into a conditional normalizing flow.

    NOTE: it will eventually crash... log prob target will be NaN at some point.
    This is probably because there is absolutely no correlation in the data.

    """

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import yaml


import sys
sys.path.append("../")
from llp_gap_reco.encoder import LLPTransformerModel
from llp_gap_reco.dataset import LLPDataset, llp_collate_fn
import jammy_flows

###### GET DATASET #######

# filepaths
top_folder = "/home/axel/i3/i3-pq-conversion-files/conversion_testing_ground/"
index_file_path = top_folder + "indexfile.pq"
feature_indices_file_path = top_folder + "feature_indices.yaml"
file_paths = [top_folder + "L2test2.000000.pq"]

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
)

def variance_from_covariance(cov_mx):
    return np.sqrt(np.diag(cov_mx))

####### CREATE MODEL #######
# model settings
config_path = "../configs/test_settings_no_cnf.yaml"
with open(config_path, 'r') as stream:
    config = yaml.safe_load(stream)
kwargs_dict = config["settings"]
# dataloader
batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=llp_collate_fn)

# create transformer encoder and cond. normalizing flow
model = LLPTransformerModel(**kwargs_dict)
pdf = jammy_flows.pdf("e6", "gggggt", conditional_input_dim=config["settings"]["output_dim"])
model.to('cuda')
pdf.to('cuda')
print("Transformer model:", model)
print("Flow model:", pdf)
total_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
print("Trainable transformer encoder parameters:", total_params)
print("Trainable flow parameters:", pdf.count_parameters())
############################

####### LOSS AND OPTIMIZER #######
criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.Adam(list(model.parameters()) + list(pdf.parameters()), lr=learning_rate)

####### TRAIN #######
n_epochs = 50
print("Starting training loop with {} epochs".format(n_epochs))
loss_vals = []
for epoch in range(n_epochs):
    for i, (batch_input, batch_lens, batch_label) in enumerate(dataloader):
        print("Batch", i, "of epoch", epoch + 1, "of", n_epochs, "epochs")
        # print(batch_lens, batch_label.shape)
        # [print(vec.shape) for vec in batch_input]
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
    if epoch%1 == 0:
        print("Epoch", epoch + 1, "loss", loss.item())
    loss_vals.append(loss.item())

####### SAVE MODEL #######
model_path = "model.pth"
flow_path  = "flow.pth"
torch.save(model.state_dict(), model_path)
torch.save(pdf.state_dict(), flow_path)

####### PLOT LOSS #######
plt.figure()
plt.plot(loss_vals)
plt.title("loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.savefig("loss.png")