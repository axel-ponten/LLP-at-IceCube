@TODO: update

## Transformer encoder
Transformer model in pytorch originally written by Thorsten Gluesenkamp, and modified by Axel Ponten.

Input is DOM summary statistics with variable sequence length (how many DOMs were hit). Since each batch then will have variable sequence length, we use the xformers package which can do batched multihead attention on GPUs with variable seq. length.

To reconstruct gaps with event-by-event uncertainty we use a conditonal normalizing flow implemented in the jammy_flows package. We encode the input data with a transformer which we then plug into a 6D gaussian conditional normalizing flow (prod_x, prod_y, prod_z, dec_x, dec_y, dec_z).

## Table of Contents

- [Transformer encoder](#transformer-encoder)
- [Table of Contents](#table-of-contents)
- [Requirements](#requirements)
- [llp\_gap\_repo package](#llp_gap_repo-package)
  - [Encoder](#encoder)
  - [Dataset](#dataset)
- [Usage](#usage)

## Requirements

Requires pytorch, jammy_flows and xformers. Create venv and run:

```ssh
pip install -r llp_gap_reco/requirements.txt
```

## llp_gap_repo package

### Encoder
The package `llp_gap_reco` contains a transformer encoder imported as `from llp_gap_reco.encoder import LLPTransformerModel`. You need a config yaml file to create the model found in `configs/`. Example of creating a transformer encoder:

```python
import yaml
from llp_gap_reco.encoder import LLPTransformerModel

####### CREATE MODEL #######
# model settings
config_path = "configs/default_settings.yaml"
with open(config_path, 'r') as stream:
    config = yaml.safe_load(stream)
kwargs_dict = config["settings"]

# create transformer encoder
model = LLPTransformerModel(**kwargs_dict)
```

### Dataset
The package also contains a pytorch custom dataset with a custom collate_fn (for creating batches for `LLPTransformerModel`) imported as `from llp_gap_reco.dataset import LLPDataset, llp_collate_fn`. Normalization settings need to be provided, and are found in e.g. `configs/normalization_args.yaml`. Index file and feature indices needs to be provided. These are produced together with the .i3 to .pq conversion.

To improve performance, the **shuffling of events** is only done within each .pq files. We only want to open a .pq file once, so when shuffling is set to true (using flag `shuffle_files`), we scramble the events within the file using `self.df_file = self.df_file.sample(frac=1).reset_index(drop=True)`. ***This means that you should set shuffle = False in the dataloader!***. Otherwise you will most likely open and close a file with every event.

Example of creating dataset and dataloader:

```python
from torch.utils.data import DataLoader
import yaml
from llp_gap_reco.dataset import LLPDataset, llp_collate_fn

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
    shuffle_files=True,
)

# dataloader
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=llp_collate_fn)
```

## Usage

Some examples are found in `examples/`