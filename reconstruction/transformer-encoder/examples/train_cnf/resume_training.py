""" Standard pytorch training loop. Resume training from a previous state.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import yaml
import glob
import os
import argparse
import time
import jammy_flows

from llp_gap_reco.encoder import LLPTransformerModel
from llp_gap_reco.dataset import LLPDataset, LLPSubset, llp_collate_fn
import llp_gap_reco.training.utils as training_utils


## helper funcs
def variance_from_covariance(cov_mx):
    return np.sqrt(np.diag(cov_mx))

###### GET DATASET #######

parser = argparse.ArgumentParser(description='Training arguments')

# Add arguments
parser.add_argument('--topfolder', type=str, help='Path to the top folder')
parser.add_argument('--normpath', type=str, help='Path to the normalization arguments file')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--batchsize', type=int, default=128, help='Batch size')
parser.add_argument('--learningrate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--modelspath', type=str, help='Path to the models directory')
parser.add_argument('--configpath', type=str, help='Path to the config file')
parser.add_argument('--filenamestart', type=str, default="base", help='For glob. How does filename start?')
parser.add_argument('--doplots', type=bool, default=True, help='Plot loss?')
parser.add_argument('--laststatetransformer', type=str, help='Path to the last state of the transformer')
parser.add_argument('--laststateflow', type=str, help='Path to the last state of the flow')
parser.add_argument('--lastepoch', type=int, default=0, help='Last epoch')

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
last_state_transformer = args.laststatetransformer
last_state_flow = args.laststateflow
last_epoch = args.lastepoch

# add trailing slash to top folder and models path
if top_folder[-1] != "/":
    top_folder += "/"
if models_path[-1] != "/":
    models_path += "/"


# create model dir if it does not exist
if not os.path.exists(models_path):
    print("Model directory doesnt exist!")
    exit()
else:
    # don't lose your precious models!
    if len(os.listdir(models_path)) < 5:
        print("Warning: model directory exists but less than 5 files in it.")
        exit()

###### CREATE DATASET & DATALOADER ######
train_dataset, test_dataset = training_utils.create_split_datasets(top_folder,
                          filename_start,
                          norm_path,
                          split=0.8,
                          shuffle=True
                          )
train_size = len(train_dataset)
test_size = len(test_dataset)
# dataloader
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=llp_collate_fn)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=llp_collate_fn)

####### CREATE MODEL #######
# create transformer encoder and cond. normalizing flow
model, pdf = training_utils.create_full_model(config_path, device="cuda")

# load previous state
model.load_state_dict(torch.load(last_state_transformer))
pdf.load_state_dict(torch.load(last_state_flow))

model.to('cuda')
pdf.to('cuda')
pdf.double()
print("Transformer model:", model)
print("Flow model:", pdf)
total_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
print("Trainable transformer encoder parameters:", total_params)
print("Trainable flow parameters:", pdf.count_parameters())
############################

####### LOSS AND OPTIMIZER #######
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, min_lr=1e-8)

####### TRAIN #######
print("Starting training loop with {} epochs".format(n_epochs))
print("#####################################")
train_loss_vals = []
test_loss_vals = []
for epoch in range(last_epoch, last_epoch + n_epochs):
    # shuffle data every epoch
    trainloader.dataset.shuffle()
    model.train()
    train_loss = 0
    start_time = time.time()  # Start the timer for the epoch
    for i, (batch_input, batch_lens, batch_label) in enumerate(trainloader):
        if i%1000 == 0:
            print("Batch", str(i) + "/" + str(train_size//batch_size), " of epoch", epoch + 1, "of", n_epochs+last_epoch, "epochs")
        try:
            # reset gradients
            optimizer.zero_grad()
            # propagate input
            nn_output = model(batch_input, batch_lens)
            nn_output = nn_output.double() # make double
            # compute loss
            batch_label = batch_label.double()
            log_prob_target, log_prob_base, position_base=pdf(batch_label, conditional_input=nn_output)
            neg_log_loss = -log_prob_target.mean()
            # compute gradient
            neg_log_loss.backward()
            
            # check gradients for NaN
            grads = []
            # transformer
            for param in model.parameters():
                if param.grad == None:
                    continue
                grads.append(param.grad.view(-1))
            # flow
            for param in pdf.parameters():
                if param.grad == None:
                    continue
                grads.append(param.grad.view(-1))
            grads = torch.cat(grads)
            if torch.isnan(grads).any():

                torch.set_printoptions(threshold=10_000)
                print('########################')
                print('nan occured here')
                print('########################')
                
                # model
                for name, param in model.named_parameters():
                    if torch.isnan(param).any():
                        print(name)
                        print(param)
                # flow
                for name, param in pdf.named_parameters():
                    if torch.isnan(param).any():
                        print(name)
                        print(param)
            # if any nan, skip batch
            assert(not torch.isnan(grads).any())
            # update weights
            optimizer.step()
            # add loss
            train_loss += neg_log_loss.item()
        except:
            print("Error in batch", i)
        
    ################# end of epoch #################
    # train loss for the epoch
    train_loss = train_loss*batch_size/train_size  # Calculate the average train loss
    train_loss_vals.append(train_loss)
    
    # test loss for the epoch
    test_loss = 0.0
    model.eval()  # Set the model to evaluation mode
    # pdf.eval()
    with torch.no_grad():
        for batch_input, batch_lens, batch_label in testloader:
            nn_output = model(batch_input, batch_lens)
            nn_output = nn_output.double()
            log_prob_target, log_prob_base, position_base=pdf(batch_label, conditional_input=nn_output)
            neg_log_loss = -log_prob_target.mean()
            test_loss += neg_log_loss.item()
    
    test_loss = test_loss*batch_size/test_size  # Calculate the average test loss
    test_loss_vals.append(test_loss)
    
    # update learning rate?
    scheduler.step(test_loss)
    print("Learning rate:", optimizer.param_groups[0]['lr'])
    
    ##### aux end of epoch #####
    # print example outputs
    print("Last three test labels:", batch_label[0:3])
    y_pred = pdf.marginal_moments(nn_output[0:3],samplesize=300)
    ymean = y_pred["mean_0"]
    ystd  = [variance_from_covariance(y_pred["varlike_0"][i]) for i in range(len(y_pred["mean_0"]))] 
    print("Last three test outputs (mean and std):")
    print(ymean)
    print(ystd)
    
    end_time = time.time()  # Stop the timer for the epoch
    epoch_time = end_time - start_time  # Calculate the time taken for the epoch
    print("Epoch", epoch + 1, "in", epoch_time, "s. Train loss", train_loss, ": Test loss", test_loss)
    print("#####################################")
    
    # save losses to csv
    df = pd.DataFrame({"train_loss": train_loss_vals, "test_loss": test_loss_vals})
    df.to_csv(models_path + "loss_resumed.csv")

    ####### PLOT LOSS #######
    if do_plots:
        plt.figure()
        plt.plot(train_loss_vals, label="train")
        plt.plot(test_loss_vals, label="test")
        plt.legend()
        plt.title("loss: batch size {} lr {}".format(batch_size, learning_rate))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.savefig(models_path + "loss_resumed.png")

    # save model every 5 epochs
    if epoch % 5 == 0:
        model_path = models_path + "model_epoch_{}.pth".format(epoch)
        flow_path  = models_path + "flow_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_path)
    torch.save(pdf.state_dict(), flow_path)

####### SAVE FINAL MODEL #######
model_path = models_path + "model_resumed_final.pth"
flow_path  = models_path + "flow_resumed_final.pth"
torch.save(model.state_dict(), model_path)
torch.save(pdf.state_dict(), flow_path)

# save losses to csv
df = pd.DataFrame({"train_loss": train_loss_vals, "test_loss": test_loss_vals})
df.to_csv(models_path + "loss_resumed.csv")

####### PLOT LOSS #######
if do_plots:
    plt.figure()
    plt.plot(train_loss_vals, label="train")
    plt.plot(test_loss_vals, label="test")
    plt.legend()
    plt.title("loss: batch size {} lr {}".format(batch_size, learning_rate))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.savefig(models_path + "loss_resumed.png")

