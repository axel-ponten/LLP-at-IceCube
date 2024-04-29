"""This script demonstrates the forward of a custom Transformer model.
It tests:

1. CustomTransformerEncoderLayer
2. CustomTransformerEncoder
3. Full model

The script generates test input, sets the model to evaluation mode, and runs the input through the model.
The output and its shape are printed for each test case.
"""

import torch
import numpy as np
from torchviz import make_dot
import yaml

import llp_gap_reco.encoder.transformer_encoder as mh_attention_encoder_axel

############################################
############## Helper functions ############
############################################
def read_config(config_path): 
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

def generate_input(batch_size, seq_length, input_dim):
    # Generate test input
    dummy_input = torch.randn((batch_size, seq_length, input_dim), dtype=torch.float32)
    return dummy_input

def test_model(model, input, input2=None):
    # Set the model to evaluation mode
    model.eval()

    # Move the model and input to the same device
    model = model.to('cuda')
    input = [x.to('cuda') for x in input] if isinstance(input, list) else input.to('cuda')

    # Run the input through the model
    if input2 is not None:
        input2 = input2.to('cuda')
        output = model(input, input2)
    else:
        output = model(input)

    return output

############################################
###### CustomTransformerEncoderLayer #######
############################################
print("testing CustomTransformerEncoderLayer:")
layer = mh_attention_encoder_axel.CustomTransformerEncoderLayer(40,
                                                                1,
                                                                device='cuda',
                                                                dtype=torch.float32,)

print(layer)
print(test_model(layer, generate_input(3,5,40)))


############################################
######## CustomTransformerEncoder ##########
############################################
print("testing CustomTransformerEncoder:")

# settings
encoder_layer_args = [40, 1] # embed dim, num heads
encoder_layer_kwargs = {"attn_package": "xformers"}
num_layers = 2
absolute_input_dim = 10 # @TODO: unused for now?

# create the encoder
encoder = mh_attention_encoder_axel.CustomTransformerEncoder(encoder_layer_args, 
                 encoder_layer_kwargs,
                 num_layers, 
                 absolute_input_dim)

print(encoder)
print(test_model(encoder, generate_input(3,5,40)))

############################################
###### Full Model w/out norm. flow #########
############################################
print("testing full model:")

# model settings
config_path = "configs/test_settings.yaml"
config = read_config(config_path)
kwargs_dict = config["settings"]
#kwargs_dict = {"input_dim": 20, "output_dim": 128, "io_mlp_hidden_dims": "128"}

# create model
model = mh_attention_encoder_axel.LLPTransformerModel(**kwargs_dict)
print(model)

# inputs
batch_size = 8
datavecs = [generate_input(1,np.random.randint(5,20),config["settings"]["input_dim"]) for i in range(1,batch_size+1)]
datalens = torch.Tensor([vec.shape[1] for vec in datavecs])
print("datavecs", datavecs)
print("datalens", datalens)
print("shapes", [vec.shape for vec in datavecs])
output = test_model(model, datavecs, datalens)
print("output", output)
print("output shape", output.shape)
