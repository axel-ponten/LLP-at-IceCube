""" Dataset class for LLP data.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import yaml
import numpy as np

#import awkward as ak

# @TODO: use this instead? https://pytorch.org/data/main/generated/torchdata.datapipes.iter.ParquetDataFrameLoader.html
class LLPDataset(Dataset):
    """
    Dataset class for loading and preprocessing LLP data.
    
    Args:
        index_file_path (str): Path to the file containing event index information.
        file_paths (list): List of .pq file paths containing the data.
        feature_indices_file_path (str): Path to the file containing feature indices.
        normalize (bool, optional): Flag indicating whether to normalize the data. Defaults to True.
        normalization_args (dict, optional): Dictionary containing normalization arguments for each feature type. 
            Defaults to None.
        device (str, optional): Device to be used for torch. Should be either "cpu" or "cuda". Defaults to None.
        dtype (torch.dtype, optional): Data type for torch. Defaults to None.
    """
    def __init__(self, index_file_path, file_paths, feature_indices_file_path,
                 normalize=True, normalization_args=None, device=None, dtype=None):
        # file with event index info
        self.index_file_path = index_file_path
        self.total_index_info = pd.read_parquet(index_file_path)

        # data files
        self.file_paths = file_paths
        self.num_events = len(self.total_index_info)

        # feature indices in data vector. dictionary with keys as feature types and values as indices
        self.feature_indices = yaml.safe_load(open(feature_indices_file_path, "r"))

        # dictionary {feature types : relevant normalization args}
        self.normalize = normalize
        if self.normalize:
            assert normalization_args is not None
            # check that all features are covered
            for feature_type in self.feature_indices.keys():
                assert feature_type in normalization_args
        self.normalization_args = normalization_args

        # only load one file to memory at a time
        self.current_load_file_index = -1

        # dtype device for torch
        self.device = device
        if self.device is not None:
            assert self.device in ["cpu", "cuda"]
        self.dtype = dtype
        

    def __len__(self):
        return self.num_events

    def __getitem__(self, idx):
        # get index info row
        index_row = self.total_index_info.iloc[idx]
        file_index = index_row["file_index"]
        index_within_file = index_row["index_within_file"]
        
        # is data already loaded? if not, load new file into memory
        if self.current_load_file_index != file_index:
            self.load_file(file_index)
        
        # get data and label
        data = self.df_file.iloc[index_within_file]["data_encoded"] # 1d arr of obj, need to stack
        label = torch.tensor([self.df_file.iloc[index_within_file]["llp_prod_x"],
                              self.df_file.iloc[index_within_file]["llp_prod_y"],
                              self.df_file.iloc[index_within_file]["llp_prod_z"],
                              self.df_file.iloc[index_within_file]["llp_decay_x"],
                              self.df_file.iloc[index_within_file]["llp_decay_y"],
                              self.df_file.iloc[index_within_file]["llp_decay_z"]],
                              dtype=self.dtype, device=self.device)
        
        # transform from 1D np arr of obj to 2D torch.tensor
        data = self.transform_data(data)
        
        # normalize
        if self.normalize:
            # @TODO: normalize file when loading?
            data = self.normalize_data(data)
            label = self.normalize_target(label) 
        
        return data, label
    
    def transform_data(self, data):
        """ Add and modify information to the data."""
        data = np.stack(data, axis=0)   
        data = torch.tensor(data, dtype=self.dtype, device=self.device)
        return data

    def load_file(self, file_index):
        """ Load the file at file_index into memory. """
        file_path = self.file_paths[file_index]
        self.df_file = pd.read_parquet(file_path)
        self.current_load_file_index = file_index

    def normalize_data(self, data):
        """ Normalize the data. x = (x-offset)*scale"""
        # for each feature type ("log_charges", "position", "abs_time", etc.)
        for feature_type, indices in self.feature_indices.items():
            if self.normalization_args[feature_type]["offset"] != 0.0:
                data[:, indices] -= self.normalization_args[feature_type]["offset"]
            if self.normalization_args[feature_type]["scale"] != 1.0:
                data[:, indices] *= self.normalization_args[feature_type]["scale"]
        return data

    def normalize_target(self, target):
        """ Scale the target (gap vertices), same as position """
        if self.normalization_args["position"]["scale"] != 1.0:
            target *= self.normalization_args["position"]["scale"]
        return target


def llp_collate_fn(batch):
    """ Custom collate function for LLP data to match transformer input. """
    datavecs = [item[0].unsqueeze(0) for item in batch]
    datalens = torch.tensor([item.shape[1] for item in datavecs], device=datavecs[0].device)
    label = [item[1] for item in batch]
    label = torch.stack(label, dim=0)
    return datavecs, datalens, label