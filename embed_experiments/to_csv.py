from EvaluationLog import EvaluationLog
import torch
import pandas as pd
import pickle

# import sys
# # sys.path.insert(0, '/home/wdwatson2/projects/CAT-Transformer/model')
# sys.path.insert(0, r'C:\Users\smbm2\projects\CAT-Transformer\model')
# from testingModel import CATTransformer, MyFTTransformer, Combined_Dataset, train, test, EarlyStopping

device_in_use = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device_in_use)

with open(r'C:\Users\smbm2\projects\CAT-Transformer\embed_experiments\evaluation_log.pkl', 'rb') as file:
    log = pickle.load(file)

df_best_test_accuracy = log.get_best_test_accuracy()

df_best_test_accuracy.to_csv(r'C:\Users\smbm2\projects\CAT-Transformer\embed_experiments\results.csv', index=False)








