""" example script to read the .pq converted i3 file
"""

import awkward as ak
import pandas as pd
import numpy as np
import yaml
import argparse

# Specify the path to your Parquet file
parser = argparse.ArgumentParser(description='Example script to read the .pq converted i3 file')
parser.add_argument('-i', dest="inputfile", type=str, help='Path to the Parquet file')
args = parser.parse_args()
parquet_file_path = args.inputfile
print("Parquet file:", parquet_file_path)

####### READ WITH PANDAS #######
print("\n####### READ WITH PANDAS #######")
# Read the Parquet file
df = pd.read_parquet(parquet_file_path)

print("shape", df.shape)
print("type", type(df))
print("\ncolumns")
for col in df:
    print(col)

print("\nexample dataencoded:")
print("shape", df["data_encoded"][0].shape)
print("shape after np.stack()", np.stack(df["data_encoded"][0], axis=0).shape)

print("\nexample llp prod xyz:")
print(df["llp_prod_x"][0],df["llp_prod_y"][0],df["llp_prod_z"][0])

print("example llp decay xyz:")
print(df["llp_decay_x"][0],df["llp_decay_y"][0],df["llp_decay_z"][0])

print("example llp gap:")
print(df["llp_gap_length"][0])

####### READ WITH AWKWARD #######
# https://awkward-array.org/doc/main/reference/generated/ak.Array.html
print("\n\n####### READ WITH AWKWARD #######")
ak_arr = ak.from_parquet(parquet_file_path)
print("ak_arr", ak_arr)
print("type", type(ak_arr))
print("ak_arr[0]", ak_arr[0])

print("\nkeys (ak_arr.fields)", ak_arr.fields)

data_encoded = ak_arr["data_encoded"]
print("\nexample dataencoded:", data_encoded)
print("ak.num", ak.num(data_encoded))
print("ak.count", ak.count(data_encoded))
print("type", type(data_encoded))
first_event = data_encoded[0]
print("first event of data_encoded", first_event)
print("type", type(first_event))
print("ak.num(first_event)", ak.num(first_event))
print("ak.count(first_event)", ak.count(first_event))