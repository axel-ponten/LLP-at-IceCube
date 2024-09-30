""" open saved models and check if gradients are zero """
import glob
import yaml
import torch

import jammy_flows
from llp_gap_reco.encoder import LLPTransformerModel
import argparse

# paths to saved models
parser = argparse.ArgumentParser(description='Check saved models')
parser.add_argument('--transformer', type=str, help='Path to transformer model')
parser.add_argument('--flow', type=str, help='Path to flow model')
parser.add_argument('--config', type=str, help='Path to config file')

args = parser.parse_args()

# Use the arguments in your code
transformer_path = args.transformer
flow_path = args.flow
config_path = args.config

# create transformer
with open(config_path, 'r') as stream:
    config = yaml.safe_load(stream)
kwargs_dict = config["settings"]
model = LLPTransformerModel(**kwargs_dict)
model.to("cuda")
model.eval()
print(model)

model.load_state_dict(torch.load(transformer_path))

# create flow
pdf = jammy_flows.pdf("e6", "gggggt", conditional_input_dim=kwargs_dict["output_dim"])
pdf.load_state_dict(torch.load(flow_path))

# get grad again
batch_size = 32
datavecs = [torch.randn((1, 50, kwargs_dict["input_dim"]), device="cuda") for _ in range(batch_size)]
datalens = torch.Tensor([vec.shape[1] for vec in datavecs]).to("cuda")
nn_output = model(datavecs, datalens)
print(nn_output)
criterion = torch.nn.MSELoss()
loss = criterion(nn_output, torch.randn((batch_size, kwargs_dict["output_dim"]), device="cuda"))
loss.backward()

# load state into model
with torch.no_grad():
    try:
        print("Transformer model:")
        for name, param in model.named_parameters():
            n_weights = param.numel()
            weight_sum = param.abs().sum()
            grad_sum = param.grad.abs().sum()
            print(name, n_weights, "{:.2f}, {:.2f} : {:.2f}, {:.2f}".format(weight_sum.item(), weight_sum.item()/n_weights, grad_sum.item(), grad_sum.item()/n_weights))
        
        print("\n\n######\nFlow model:")
        for name, param in pdf.named_parameters():
            n_weights = param.numel()
            weight_sum = param.abs().sum()
            grad_sum = param.grad.abs().sum()
            print(name, n_weights, "{:.2f}, {:.2f} : {:.2f}, {:.2f}".format(weight_sum.item(), weight_sum.item()/n_weights, grad_sum.item(), grad_sum.item()/n_weights))
    except Exception as e:
        print("An error occurred:", str(e))