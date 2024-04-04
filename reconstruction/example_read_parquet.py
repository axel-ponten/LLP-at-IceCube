""" example script to read the .pq converted i3 file
"""

import pyarrow.parquet as pq

# Specify the path to your Parquet file
parquet_file_path = "/data/user/axelpo/conversion_testing_ground/L2test2.000000.pq"

# Read the Parquet file
table = pq.read_table(parquet_file_path)

# Convert the table to a pandas DataFrame
df = table.to_pandas()
print(df)
print("columns")
for col in df:
    print(col)

print("example dataencoded:")
print(df["data_encoded"][0])

print("example llp prod xyz:")
print(df["llp_prod_x"][0],df["llp_prod_y"][0],df["llp_prod_z"][0])

print("example llp decay xyz:")
print(df["llp_decay_x"][0],df["llp_decay_y"][0],df["llp_decay_z"][0])

print("example llp gap:")
print(df["llp_gap_length"][0])