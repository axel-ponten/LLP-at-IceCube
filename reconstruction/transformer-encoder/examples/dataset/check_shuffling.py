import sys
sys.path.append("../")
from llp_gap_reco.dataset import LLPDataset, llp_collate_fn
from torch.utils.data import DataLoader
import yaml
import torch
import glob
import pandas as pd

# filepaths
top_folder = "/home/axel/i3/i3-pq-conversion-files/DLS-115-5e-6/"
index_file_path = top_folder + "indexfile.pq"
total_index_info = pd.read_parquet(index_file_path)
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
    normalization_args=normalization_args,
    device="cuda",
    dtype=torch.float32,
    shuffle_files=False,
)

dataset_shuffle = LLPDataset(
    index_file_path,
    file_paths,
    feature_indices_file_path,
    normalize_data=True,
    normalization_args=normalization_args,
    device="cuda",
    dtype=torch.float32,
    shuffle_files=True,
)

##### INDEX FILE #####
print("Total index info:")
print(total_index_info)
##### TEST DATASET ######
# not shuffled
idx = 0
index_row = dataset.total_index_info.iloc[idx]
file_index = index_row["file_index"]
index_within_file = index_row["index_within_file"]
data, label = dataset[idx]
event = dataset.df_file.iloc[index_row["index_within_file"]]
print("\n##### Not shuffled #####:")
print("Index row")
print(index_row)
print("label", label)
print(event)

# shuffled
idx = 0
index_row = dataset_shuffle.total_index_info.iloc[idx]
file_index = index_row["file_index"]
index_within_file = index_row["index_within_file"]
data, label = dataset_shuffle[idx]
event = dataset_shuffle.df_file.iloc[index_row["index_within_file"]]
print("\n##### Shuffled #####:")
print("Index row")
print(index_row)
print("label", label)
print(event)

##### TEST USING DATALOADER ######
print("\n\n##### TEST USING DATALOADER #####")
def test_dataloader(shuffle = False):
    print("\nNew dataloader")
    dataset = LLPDataset(
        index_file_path,
        file_paths,
        feature_indices_file_path,
        normalize_data=True,
        normalization_args=normalization_args,
        device="cuda",
        dtype=torch.float32,
        shuffle_files=shuffle,
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=llp_collate_fn)
    # Iterate over the dataloader
    for i, batch in enumerate(dataloader):
        datavecs, datalens, labels = batch
        print("batch", i, "datalens", datalens)
        if i > 0:
            break
        
print("##### Not shuffled (twice instantiated, should be same) #####")
test_dataloader(shuffle=False)
test_dataloader(shuffle=False)

print("\n\n##### Shuffled (twice instantiated, should be different) #####")
test_dataloader(shuffle=True)
test_dataloader(shuffle=True)

#### CHECK THAT WE DONT MIX EVENTS ####
print("\n\n##### CHECK THAT WE DONT MIX INPUT/TARGET #####")
# create dataset
dataset = LLPDataset(
    index_file_path,
    file_paths,
    feature_indices_file_path,
    normalize_data=True,
    normalization_args=normalization_args,
    device="cuda",
    dtype=torch.float32,
    shuffle_files=False,
)

dataset_shuffle = LLPDataset(
    index_file_path,
    file_paths,
    feature_indices_file_path,
    normalize_data=True,
    normalization_args=normalization_args,
    device="cuda",
    dtype=torch.float32,
    shuffle_files=True,
)

# find row matching event id and run id
event_id = 77
run_id = 777700112


index_row = dataset.total_index_info[(dataset.total_index_info["event_id"] == event_id) & (dataset.total_index_info["run_id"] == run_id)]
index_row_shuffle = dataset_shuffle.total_index_info[(dataset_shuffle.total_index_info["event_id"] == event_id) & (dataset_shuffle.total_index_info["run_id"] == run_id)]
print("Index row")
print(index_row)
print("Index row shuffled")
print(index_row_shuffle)

print("Event from unshuffled")
data, label = dataset[index_row.index.item()]
print("label", label)
print("data", data[:3])

print("Event from shuffled")
data, label = dataset_shuffle[index_row_shuffle.index.item()]
print("label", label)
print("data", data[:3])