import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helpers import plot_entropy_vs_layers, plot_entropy_vs_layers_classes
import pickle

df = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\interpretability\results.csv')

datasets = ['Higgs', 'Income', 'Jannis']
classes = ['class_0', 'class_1', 'class_2', 'class_3']
models = ['CAT', 'FT']
# for dataset in datasets:
#     for cls in classes:
#         plot_entropy_vs_layers(df, cls, dataset, 'test')

for dataset in datasets:
    for model in models:
        plot_entropy_vs_layers_classes(df, model, dataset)









