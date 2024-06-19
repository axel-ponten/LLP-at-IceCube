import matplotlib.pyplot as plt

import pandas as pd
import os

os.chdir(os.path.dirname(__file__))
# Load the data
loss = pd.read_csv("loss_no_cnf.csv")
train_loss_vals = loss["train_loss"]
test_loss_vals = loss["test_loss"]

plt.figure()
plt.plot(train_loss_vals, label="train")
plt.plot(test_loss_vals, label="test")
plt.legend()
plt.title("loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.savefig("loss_no_cnf.png")