""" Standard pytorch training loop.
    Example with random input of varying seq. length
    and random labels.
    Just to test that things are working.
    Transformer encoder into a conditional normalizing flow.

    NOTE: it will eventually crash... log prob target will be NaN at some point.
    This is probably because there is absolutely no correlation in the data.

    """

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml

from llp_gap_reco.encoder import LLPTransformerModel
import jammy_flows

###### GENERATE DATA #######
def generate_dummy_input(batch_size, seq_length, input_dim):
    # Generate test input
    dummy_input = torch.randn((batch_size, seq_length, input_dim), dtype=torch.float32)
    return dummy_input

def generate_dummy_labels(batch_size):
    # labels: prod_x, prod_y, prod_z, dec_x, dec_y, dec_z
    dummy_labels = torch.randn((batch_size, 6), dtype=torch.float32)
    return dummy_labels

def forward_model(model, pdf, datavecs):
    # create datalens
    datalens = torch.Tensor([vec.shape[1] for vec in datavecs])

    # # Set the model to evaluation mode
    # model.eval()
    # pdf.eval()

    # Move the model and input to the same device
    model = model.to('cuda')
    pdf = pdf.to('cuda')
    datavecs = [x.to('cuda') for x in datavecs] if isinstance(datavecs, list) else datavecs.to('cuda')
    datalens = datalens.to('cuda')

    # Run the input through the model
    output = model(datavecs, datalens)
    y_pred = pdf.marginal_moments(output,samplesize=300)

    return y_pred

def variance_from_covariance(cov_mx):
    return np.sqrt(np.diag(cov_mx))

####### CREATE MODEL #######
# model settings
config_path = "configs/test_settings.yaml"
with open(config_path, 'r') as stream:
    config = yaml.safe_load(stream)
kwargs_dict = config["settings"]

# create transformer encoder and cond. normalizing flow
model = LLPTransformerModel(**kwargs_dict)
pdf = jammy_flows.pdf("e6", "gggggt", conditional_input_dim=config["settings"]["output_dim"])
model.to('cuda')
pdf.to('cuda')
print("Transformer model:", model)
print("Flow model:", pdf)
total_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
print("Trainable transformer encoder parameters:", total_params)
print("Trainable flow parameters:", pdf.count_parameters())
############################

############ FAKE TRAINING DATA ############
n_batches = 10
batch_size = 8
input_dim = config["settings"]["input_dim"]
train_loader = []
for i in range(n_batches):
    # generate
    datavecs = [generate_dummy_input(1,np.random.randint(12, 5000), input_dim) for _ in range(batch_size)]
    datalens = torch.Tensor([vec.shape[1] for vec in datavecs])
    datalabels = generate_dummy_labels(batch_size)
    # send to cuda
    datavecs = [x.to('cuda') for x in datavecs] if isinstance(datavecs, list) else datavecs.to('cuda')
    datalens = datalens.to('cuda')
    datalabels = datalabels.to('cuda')
    # append
    train_loader.append((datavecs, datalens, datalabels))

####### LOSS AND OPTIMIZER #######
criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.Adam(list(model.parameters()) + list(pdf.parameters()), lr=learning_rate)

####### TRAIN #######
n_epochs = 2
print("Starting training loop with {} epochs".format(n_epochs))
loss_vals = []
for epoch in range(n_epochs):
    for i, (batch_input, batch_lens, batch_label) in enumerate(train_loader):
        print("Batch", i, "of epoch", epoch + 1, "of", n_epochs, "epochs")
        # reset gradients
        optimizer.zero_grad()
        # propagate input
        nn_output = model(batch_input, batch_lens)
        try:
            log_prob_target, log_prob_base, position_base=pdf(batch_label, conditional_input=nn_output)
        except:
            print("Error in forward pass probably log_prob_target in previous batch had NaN.")
            print(nn_output.shape)
            print(nn_output)
        # compute loss
        neg_log_loss = -log_prob_target.mean()
        # compute gradient
        neg_log_loss.backward()
        # update weights
        optimizer.step()
        # run test
        # print(nn_output.shape)
        # print(nn_output)
        # print("log_prob_target", log_prob_target)
    if epoch%1 == 0:
        print("Epoch", epoch + 1, "loss", neg_log_loss.item())
    loss_vals.append(neg_log_loss.item())

####### SAVE MODEL #######
model_path = "model.pth"
flow_path  = "flow.pth"
torch.save(model.state_dict(), model_path)
torch.save(pdf.state_dict(), flow_path)

####### PLOT LOSS #######
plt.figure()
plt.plot(loss_vals)
plt.title("loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.savefig("loss.png")