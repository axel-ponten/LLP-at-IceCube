""" Standard pytorch training loop.

    Just to test that things are working.
    Transformer encoder into a conditional normalizing flow.

    """

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import yaml
import glob
import jammy_flows

import sys
sys.path.append("../")
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

# dataloader
batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=llp_collate_fn)

def variance_from_covariance(cov_mx):
    return np.sqrt(np.diag(cov_mx))

####### CREATE MODEL #######
# model settings
config_path = "../configs/test_settings.yaml"
with open(config_path, 'r') as stream:
    config = yaml.safe_load(stream)
kwargs_dict = config["settings"]

# create transformer encoder and cond. normalizing flow
model = LLPTransformerModel(**kwargs_dict)
#sigmoid_layer = nn.Sigmoid()
pdf = jammy_flows.pdf("e6", "gggggt", conditional_input_dim=config["settings"]["output_dim"])

# init for the gaussian flow
# test_label = torch.Tensor([400,400,400, -400,-400,-400]).unsqueeze(0)  
# pdf.init_params(data=test_label)

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
n_epochs = 2
print("Starting training loop with {} epochs".format(n_epochs))
loss_vals = []
for epoch in range(n_epochs):
    for i, (batch_input, batch_lens, batch_label) in enumerate(dataloader):
        print("Batch", i, "of epoch", epoch + 1, "of", n_epochs, "epochs")
        print(batch_lens)
        [print(vec.shape) for vec in batch_input]
        # reset gradients
        optimizer.zero_grad()
        # propagate input
        nn_output = model(batch_input, batch_lens)
        # # check range of output
        # for item in nn_output:
        #     print(torch.min(item), "-", torch.max(item))
        
        #batch_label = batch_label.to('cpu')
        #print(batch_label.device)
        # try:
        log_prob_target, log_prob_base, position_base=pdf(batch_label, conditional_input=nn_output)
        # except:
        #     print("Error in forward pass probably log_prob_target in previous batch had NaN.")
        #     print(nn_output.shape)
        #     print(nn_output)
        # compute loss
        neg_log_loss = -log_prob_target.mean()
        # compute gradient
        neg_log_loss.backward()
        # update weights
        optimizer.step()
        # run test
        # print(nn_output.shape)
        # print(nn_output)
        print("log_prob_target", log_prob_target)
    if epoch%1 == 0:
        print("Epoch", epoch + 1, "loss", neg_log_loss.item())
    loss_vals.append(neg_log_loss.item())

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