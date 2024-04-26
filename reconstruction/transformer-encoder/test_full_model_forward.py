import numpy as np
import torch
import yaml

from mh_attention_encoder_axel import LLPTransformerModel
import jammy_flows

###### funcs #######
def generate_dummy_input(batch_size, seq_length, input_dim):
    # Generate test input
    dummy_input = torch.randn((batch_size, seq_length, input_dim), dtype=torch.float32)
    return dummy_input

def forward_model(model, pdf, datavecs):
    # create datalens
    datalens = torch.Tensor([vec.shape[1] for vec in datavecs])

    # Set the model to evaluation mode
    model.eval()
    pdf.eval()

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

if __name__ == "__main__":
    ####### CREATE MODEL AND PASS FAKE DATA #######

    # model settings
    config_path = "configs/test_settings.yaml"
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    kwargs_dict = config["settings"]

    # create transformer encoder and cond. normalizing flow
    model = LLPTransformerModel(**kwargs_dict)
    pdf = jammy_flows.pdf("e6", "gggggt", conditional_input_dim=config["settings"]["output_dim"])

    print("Transformer model:", model)
    print("Flow model:", pdf)
    total_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("Trainable transformer encoder parameters:", total_params)
    print("Trainable flow parameters:", pdf.count_parameters())

    print("######################################")
    # inputs
    batch_size = 8
    input_dim = config["settings"]["input_dim"]
    datavecs = [generate_dummy_input(1,np.random.randint(12, 5000), input_dim) for i in range(batch_size)]

    # calculate output
    print("Test forward fake data through full model:")
    y_pred = forward_model(model, pdf, datavecs)

    # look at output
    print("\n\nOutput keys from jammy_flows pdf.marginal_moments:")
    [print(key, y_pred[key].shape) for key in y_pred.keys()]

    print("\n\nEvent by event output (mean and std):\n")
    labels = ["prod_x", "prod_y", "prod_z", "dec_x", "dec_y", "dec_z"]
    for i in range(len(y_pred["mean_0"])):
        print("event #",i,"input",datavecs[i].shape)
        ymean = y_pred["mean_0"][i]
        ystd  = variance_from_covariance(y_pred["varlike_0"][i])
        [print(labels[i] + f"  {y:.4f} +- {y_sig:.4f}") for i, (y, y_sig) in enumerate(zip(ymean, ystd))]
