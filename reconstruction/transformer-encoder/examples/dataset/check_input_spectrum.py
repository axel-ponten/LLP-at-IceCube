from llp_gap_reco.dataset import LLPDataset, llp_collate_fn
import yaml
import torch
import glob
import matplotlib.pyplot as plt

# filepaths
top_folder = "/home/axel/i3/i3-pq-conversion-files/DarkLeptonicScalar.mass-110.eps-1e-5.nevents-50000_ene_1e3_2e5_gap_100_240503.208637138/"
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
    normalization_args=normalization_args,
    device="cuda",
    dtype=torch.float32,
    shuffle_files=False,
)

# feature indices
print("Feature indices:")
# open yaml file with feature indices
with open(feature_indices_file_path, "r") as file:
    feature_indices = yaml.safe_load(file)

# save all the inputs
n_events = len(dataset)
n_features = dataset[0][0].shape[1]
print(f"Number of events: {n_events}")
print(f"Number of features: {n_features}")
n_events = 10
inputs = [[] for i in range(n_features)]
for i in range(n_events):
    data, label = dataset[i]
    for k in range(data.shape[0]):
        for j in range(n_features):
            inputs[j].append(data[k,j].item())
    if i%500 == 0:
        print(f"Processed {i} events")
    if i > n_events:
        break

# plot
fig, axs = plt.subplots(4, 4, figsize=(12, 12))

def get_name_from_index(index, feature_indices):
    name = ""
    for key, val in feature_indices.items():
        if type(val) == int:
            if val == index:
                name = key
                break
        else:
            if index in val:
                name = key
                break
    return name

for i in range(4):
    for j in range(4):
        tile_num = i*4 + j
        if tile_num >= n_features:
            break
        feature_type = get_name_from_index(tile_num, feature_indices)
        axs[i, j].hist(inputs[i*4+j], bins=10)
        axs[i, j].set_title(f'Input {i*4+j+1}')
        axs[i, j].set_xlabel(feature_type)
        axs[i, j].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig("input_spectrum.png")

