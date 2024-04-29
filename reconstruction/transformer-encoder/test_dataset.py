from llpdataset import LLPDataset, llp_collate_fn
import yaml
import torch

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
    normalize=True,
    normalization_args=normalization_args,
    device="cuda",
    dtype=torch.float32,
)


##### TEST DATASET ######
print(dataset)

for i in range(len(dataset)):
    data, label = dataset[i]
    print(i, "label", label)
    print("pos", data[:, [0,1,2]])
    if i > 3:
        break

##### TEST USING DATALOADER ######
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=llp_collate_fn)
# Iterate over the dataloader
for i, batch in enumerate(dataloader):
    datavecs, datalens, labels = batch
    # Perform operations on the batch
    # ...
    print("batch", i)
    print("datavecs", datavecs)
    print("datalens", datalens)
    if i > 3:
        break
