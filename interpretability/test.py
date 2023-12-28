from helpers import EntropyLog, entropy, evaluate, attn_entropy_get, extract_entropy_scores, build_table, format_table_to_dataframe
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/home/wdwatson2/projects/CAT-Transformer/model')
from testingModel import CATTransformer, MyFTTransformer, Combined_Dataset, train, test, EarlyStopping

device_in_use = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device_in_use)



with open('/home/wdwatson2/projects/CAT-Transformer/interpretability/entropylog.pkl', 'rb') as file:
    entropylog = pickle.load(file)


# Extract entropy scores for all classes
all_entropy_scores = extract_entropy_scores(entropylog)

# Print or use the extracted entropy scores
for entry in all_entropy_scores:
    print(f"Model: {entry['Model']}, Layers: {entry['Layers']}, Dataset: {entry['Dataset']}, "
          f"Split: {entry['Split']}, Class: {entry['Class']}, Entropy: {entry['Entropy']}")


# Build the table from entropy_log
entropy_table = build_table(entropylog)

# Format the table into a pandas DataFrame
formatted_df = format_table_to_dataframe(entropy_table)

# Print the DataFrame
print(formatted_df)

# formatted_df.to_csv('income_table.csv')