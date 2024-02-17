from helpers import Log
import pandas as pd
import pickle

with open('/home/wdwatson2/projects/CAT-Transformer/embed_experiments/gaussian_determine/performance_log.pkl', 'rb') as file:
    performance_log = pickle.load(file)

# Create DataFrame
df = performance_log.create_dataframe()

# Export to CSV
df.to_csv('/home/wdwatson2/projects/CAT-Transformer/embed_experiments/gaussian_determine/output.csv', index=False)

