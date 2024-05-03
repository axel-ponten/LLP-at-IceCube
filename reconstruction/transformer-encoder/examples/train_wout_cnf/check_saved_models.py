""" open saved models and check if gradients are zero """
import glob
import yaml
import torch

from llp_gap_reco.encoder import LLPTransformerModel

# paths to saved models
model_paths = sorted(glob.glob('models/*.pth'))

# create model
config_path = "../../configs/test_settings_no_cnf.yaml"
with open(config_path, 'r') as stream:
    config = yaml.safe_load(stream)
kwargs_dict = config["settings"]
model = LLPTransformerModel(**kwargs_dict)
model.to("cuda")
model.eval()
print(model)

# get grad again
batch_size = 32
datavecs = [torch.randn((1, 50, 16), device="cuda") for _ in range(batch_size)]
datalens = torch.Tensor([vec.shape[1] for vec in datavecs]).to("cuda")
nn_output = model(datavecs, datalens)
print(nn_output)
criterion = torch.nn.MSELoss()
loss = criterion(nn_output, torch.randn((batch_size, 6), device="cuda"))
loss.backward()

# load state into model
with torch.no_grad():
    for i, model_path in enumerate(model_paths):
        print("Model path:", model_path)
        model.load_state_dict(torch.load(model_path))
        try:
            for name, param in model.named_parameters():
                n_weights = param.numel()
                weight_sum = param.abs().sum()
                grad_sum = param.grad.abs().sum()
                print(name, n_weights, "{:.2f}, {:.2f} : {:.2f}, {:.2f}".format(weight_sum.item(), weight_sum.item()/n_weights, grad_sum.item(), grad_sum.item()/n_weights))
        except Exception as e:
            print("An error occurred:", str(e))
        print("\n\n")
        if i > 10:
            break