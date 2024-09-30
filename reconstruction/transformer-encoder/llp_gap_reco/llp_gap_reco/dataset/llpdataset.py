""" Dataset class for LLP data.
"""

import torch
from torch.utils.data import Dataset, Subset
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
        shuffle_files (bool, optional): Flag indicating whether to shuffle the files. Defaults to False.
    """
    def __init__(self, index_file_path, file_paths, feature_indices_file_path,
                 normalize_data=True, normalize_target=False, normalization_args=None,
                 device=None, dtype=None,
                 shuffle_files=False,
                 dataset_type="training"):
        # file with event index info
        self.index_file_path = index_file_path
        self.total_index_info = pd.read_parquet(index_file_path)

        # data files
        self.file_paths = sorted(file_paths) # make filelist match index file
        self.num_events = len(self.total_index_info)

        # feature indices in data vector. dictionary with keys as feature types and values as indices
        self.feature_indices = yaml.safe_load(open(feature_indices_file_path, "r"))

        # dictionary {feature types : relevant normalization args}
        self.normalize_data = normalize_data
        if self.normalize_data:
            assert normalization_args is not None
            # check that all features are covered
            for feature_type in self.feature_indices.keys():
                assert feature_type in normalization_args
        self.normalize_target = normalize_target
        self.normalization_args = normalization_args

        # only load one file to memory at a time
        self.current_load_file_index = -1

        # shuffle events within a file?
        print("INFO! Using dataset", self.__class__.__name__, ": Use init attr `shuffle_files` and set shuffle=False in Dataloader.")
        self.shuffle_files = shuffle_files
        if self.shuffle_files:
            self.shuffle()
        # dtype device for torch
        self.device = device
        if self.device is not None:
            assert self.device in ["cpu", "cuda"]
        self.dtype = dtype
        
        # dataset type
        self.dataset_type = dataset_type

    def __len__(self):
        return self.num_events

    def __getitem__(self, idx):
        # get index info row
        index_row = self.total_index_info.iloc[idx]
        file_index = index_row["file_index"]
        index_within_file = index_row["index_within_file"]
        # convert to int for indexing
        file_index = int(file_index)
        index_within_file = int(index_within_file)
        
        # is data already loaded? if not, load new file into memory
        if self.current_load_file_index != file_index:
            self.load_file(file_index)
            # assert that you opened the right file
            assert self.df_file.iloc[index_within_file]["event_id"] == index_row["event_id"] , "Event ID mismatch. Are filepaths sorted?"
            assert self.df_file.iloc[index_within_file]["run_id"]   == index_row["run_id"] , "Run ID mismatch. Are filepaths sorted?"
        
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
        if self.normalize_data:
            # @TODO: normalize file when loading?
            data = self._normalize_data(data)
        if self.normalize_target:
            label = self._normalize_target(label) 
        
        return data, label

    def get_data_and_info(self, idx):
        data, label = self.__getitem__(idx)
        muon_energy, muon_zenith, muon_length = self.get_muon_info_from_idx(idx)
        return data, label, muon_energy, muon_zenith, muon_length

    def get_event_run_info(self, idx):
        """ Returns event id and run id of given an idx. """
        row = self.total_index_info.iloc[idx]
        return row["event_id"], row["run_id"]

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

    def _normalize_data(self, data):
        """ Normalize the data. x = (x-offset)*scale"""
        # for each feature type ("log_charges", "position", "abs_time", etc.)
        for feature_type, indices in self.feature_indices.items():
            if self.normalization_args[feature_type]["offset"] != 0.0:
                data[:, indices] -= self.normalization_args[feature_type]["offset"]
            if self.normalization_args[feature_type]["scale"] != 1.0:
                data[:, indices] *= self.normalization_args[feature_type]["scale"]
        return data

    def _normalize_target(self, target):
        """ Scale the target (gap vertices), same as position """
        if self.normalization_args["position"]["scale"] != 1.0:
            target *= self.normalization_args["position"]["scale"]
        return target

    def shuffle(self):
        """ Shuffle the files. """
        # shuffle events with same file index
        if self.shuffle_files:
            # shuffle events with same file index
            grouped_index_info = self.total_index_info.groupby("file_index")
            self.total_index_info = grouped_index_info.sample(frac=1).reset_index(drop=True)
        else:
            print("INFO! Called LLPDataset.shuffle() but shuffle_files is False. No shuffling done.")
            
    def get_muon_info(self, event_id, run_id):
        """ Get muon energy, zenith and length from event_id and run_id. """
        # get row with event_id and run_id
        row = self.total_index_info[(self.total_index_info["event_id"] == event_id) & (self.total_index_info["run_id"] == run_id)]
        return row["muon_energy"], row["muon_zenith"], row["muon_length"]
    
    def get_muon_info_from_idx(self, idx):
        """ Get muon energy, zenith and length from index. """
        row = self.total_index_info.iloc[idx]
        return row["muon_energy"], row["muon_zenith"], row["muon_length"]

class UnlabeledLLPDataset(LLPDataset):
    """
    Dataset class for loading and preprocessing unlabeled data.
    """
    def __getitem__(self, idx):
        # get index info row
        index_row = self.total_index_info.iloc[idx]
        file_index = index_row["file_index"]
        index_within_file = index_row["index_within_file"]
        # convert to int for indexing
        file_index = int(file_index)
        index_within_file = int(index_within_file)
        
        # is data already loaded? if not, load new file into memory
        if self.current_load_file_index != file_index:
            self.load_file(file_index)
            # assert that you opened the right file
            assert self.df_file.iloc[index_within_file]["event_id"] == index_row["event_id"] , "Event ID mismatch. Are filepaths sorted?"
            assert self.df_file.iloc[index_within_file]["run_id"]   == index_row["run_id"] , "Run ID mismatch. Are filepaths sorted?"
        
        # get data
        data = self.df_file.iloc[index_within_file]["data_encoded"] # 1d arr of obj, need to stack
        
        # transform from 1D np arr of obj to 2D torch.tensor
        data = self.transform_data(data)
        
        # normalize
        if self.normalize_data:
            data = self._normalize_data(data)
        
        return data

class LLPSubset(Subset):
    def __init__(self, llpdataset: LLPDataset, indices) -> None:
        super().__init__(llpdataset, indices)
    
    def shuffle(self):
        self.dataset.shuffle()
        
    def get_muon_info_from_idx(self, idx):
        true_index = self.indices[idx]
        return self.dataset.get_muon_info_from_idx(true_index)
    
    def get_data_and_info(self, idx):
        true_index = self.indices[idx]
        return self.dataset.get_data_and_info(true_index)

###### COLLATE FUNCTION FOR DATALOADER ######
def llp_collate_fn(batch):
    """ Custom collate function for LLP data to match transformer input. """
    datavecs = [item[0].unsqueeze(0) for item in batch]
    datalens = torch.tensor([item.shape[1] for item in datavecs], device=datavecs[0].device)
    label = [item[1] for item in batch]
    label = torch.stack(label, dim=0)
    return datavecs, datalens, label

def llp_collate_unlabeled_fn(batch):
    """ Custom collate function for unlabeled data to match transformer input. """
    datavecs = [item.unsqueeze(0) for item in batch]
    datalens = torch.tensor([item.shape[1] for item in datavecs], device=datavecs[0].device)
    label = [None for item in batch]
    return datavecs, datalens, label