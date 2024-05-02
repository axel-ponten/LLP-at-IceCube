import yaml
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Example script')

parser.add_argument('--index_file_path', type=str, default='~/i3/i3-pq-conversion-files/conversion_testing_ground/indexfile.pq', help='Path to the index file')
parser.add_argument('--feature_indices_file_path', type=str, default='/home/axel/i3/i3-pq-conversion-files/conversion_testing_ground/feature_indices.yaml', help='Path to the feature indices file')

args = parser.parse_args()

index_file_path = args.index_file_path
feature_indices_file_path = args.feature_indices_file_path

# test index file
print("\nIndex file:")
total_index_info = pd.read_parquet(index_file_path)
print(total_index_info)

# load feature indices yaml file
print("\n Feature indices file (yaml):")
print(feature_indices_file_path)
feature_indices = yaml.load(open(feature_indices_file_path, "r"), Loader=yaml.FullLoader)
print(feature_indices)