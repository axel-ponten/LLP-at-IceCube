import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import argparse

from llp_gap_reco.dataset import llp_collate_fn
import llp_gap_reco.training.utils as training_utils
from llp_gap_reco.training.train import Trainer

###### ARGPARSE ######
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
parser.add_argument('--gradclip', type=float, default=2000., help='Gradient clipping value')

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
grad_clip_val = args.gradclip

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

###### CREATE DATASET & DATALOADER ######
train_dataset, test_dataset = training_utils.create_split_datasets(top_folder,
                          filename_start,
                          norm_path,
                          split=0.8,
                          shuffle=True
                          )
# dataloader
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                         collate_fn=llp_collate_fn)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         collate_fn=llp_collate_fn)

####### CREATE MODEL #######
# create transformer encoder and cond. normalizing flow
model, pdf = training_utils.create_full_model(config_path, device="cuda")
# init for the gaussianization flow
training_utils.init_pdf_target_space(pdf, train_dataset, n_init=1000, device="cuda")

# print information
print("Transformer model:", model)
print("Flow model:", pdf)
total_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
print("Trainable transformer encoder parameters:", total_params)
print("Flow parameters:", pdf.count_parameters())
###

########## TRAIN ##########
# create trainer
trainer = Trainer(model, pdf, device="cuda")
# optimizer and scheduler
optimizer = torch.optim.Adam(list(model.parameters()) + list(pdf.parameters()), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, min_lr=1e-10)
# train
print("Grad clip value:", grad_clip_val)
trainer.train(trainloader, testloader, n_epochs, optimizer, scheduler, models_path,
              start_epoch=0, verbose=True, save_freq=5, grad_clip=grad_clip_val)