""" This module contains utility functions for training the model. """
from llp_gap_reco.encoder import LLPTransformerModel
from llp_gap_reco.dataset import LLPDataset, LLPSubset
import yaml
import glob
import torch
import jammy_flows
import numpy as np

class EarlyStopper:
    """
    Class for implementing early stopping during training.

    Attributes:
        patience (int): The number of epochs to wait before stopping if the validation loss does not improve.
        percent_tolerance (float): The percentage tolerance of the running average loss for determining if the validation loss has improved.

    Methods:
        early_stop(validation_loss): Checks if the validation loss has stopped improving and returns True if early stopping criteria is met, False otherwise.
    """

    def __init__(self, patience=15, percent_tolerance=0.0):
        self.patience = patience
        self.percent_tolerance = percent_tolerance # of running avg loss

        self.counter = 0
        self.min_validation_loss = float('inf')
        # for running average of validation loss
        self.running_average = []
        self.average_length = 10

    def early_stop(self, validation_loss):
        """
        Checks if the validation loss has stopped improving and returns True if early stopping criteria is met, False otherwise.

        Args:
            validation_loss (float): The current validation loss.

        Returns:
            bool: True if early stopping criteria is met, False otherwise.
        """
        if self.percent_tolerance > 0.0:
            # running average of the last few validation losses
            if len(self.running_average) >= self.average_length:
                self.running_average.pop(0)
            self.running_average.append(validation_loss)
            tolerance = self.percent_tolerance * np.mean(self.running_average)
        else:
            tolerance = 0.0
        
        # new minimum?
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        # counter increment?
        elif validation_loss > (self.min_validation_loss + tolerance):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def predict_cnf(model, pdf, datavecs, datalens, samplesize=300):
    def variance_from_covariance(cov_mx):
        return np.sqrt(np.diag(cov_mx))
    with torch.no_grad():
        nn_output = model(datavecs, datalens)
        nn_output = nn_output.double()
        pred = pdf.marginal_moments(nn_output,samplesize=samplesize)
        # mean and std
        pred_mean = pred["mean_0"]
        pred_std  = [variance_from_covariance(pred["varlike_0"][i]) for i in range(len(pred_mean))] 
    return pred_mean, pred_std

def create_full_model(config_path, device="cuda"):
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    model = create_transformer(config_path, device)
    pdf = create_cnf(config_path, device=device)
    return model, pdf

def create_transformer(config_path, device="cuda"):
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    kwargs_dict = config["settings"]
    model = LLPTransformerModel(**kwargs_dict)
    model.to(device)
    return model

def create_cnf(config_path, device="cuda"):
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    inputdim=config["settings"]["output_dim"]
    flow_str=config["flow"]["flow_str"]
    opt_dict = get_opt_dict() # for cnf
    # create cnf
    pdf = jammy_flows.pdf("e6", flow_str,
                      conditional_input_dim=inputdim,
                      options_overwrite=opt_dict,
                      )
    pdf.double()
    pdf.to(device)
    return pdf

def get_opt_dict():
    """ for cnf """
    opt_dict=dict()
    opt_dict["t"]=dict()
    opt_dict["t"]["cov_type"]="full"
    opt_dict["g"]=dict()
    opt_dict["g"]["fit_normalization"]=0
    opt_dict["g"]["upper_bound_for_widths"]=1.0
    opt_dict["g"]["lower_bound_for_widths"]=0.01
    return opt_dict

def init_pdf_target_space(pdf, dataset, n_init=200, device="cuda"):
    """ initialize the pdf target space. needed since labels might be much different than 1. """
    initialization_labels = torch.stack([dataset[i][1] for i in range(n_init)], dim=0).to('cpu')
    pdf.init_params(data=initialization_labels)
    pdf.to(device)

def create_split_datasets(top_folder,
                          filename_start,
                          norm_path,
                          split=0.8,
                          shuffle=False,
                          ):
    # normalizaton args
    with open(norm_path, "r") as file:
        normalization_args = yaml.safe_load(file)

    # filepaths
    index_file_path = top_folder + "indexfile.pq"
    feature_indices_file_path = top_folder + "feature_indices.yaml"
    file_paths = glob.glob(top_folder + filename_start + "*.pq")
    
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
        shuffle_files=shuffle,
    )

    # split dataset into train and test
    nfiles = len(file_paths)
    nfiles_train = int(split*nfiles)
    nfiles_test = nfiles - nfiles_train
    events_per_file = len(dataset)//nfiles
    train_size = int(nfiles_train*events_per_file)
    test_size = len(dataset) - train_size
    print("#####################")
    print("Dataset info:")
    print("Total events", len(dataset))
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
    
    return train_dataset, test_dataset