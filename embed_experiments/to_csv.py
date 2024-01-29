from EvaluationLog import EvaluationLog
import torch
import pandas as pd
import pickle
import numpy as np

# import sys
# # sys.path.insert(0, '/home/wdwatson2/projects/CAT-Transformer/model')
# sys.path.insert(0, r'C:\Users\smbm2\projects\CAT-Transformer\model')
# from testingModel import CATTransformer, MyFTTransformer, Combined_Dataset, train, test, EarlyStopping

device_in_use = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device_in_use)

with open(r'C:\Users\smbm2\projects\CAT-Transformer\embed_experiments\evaluation_log.pkl', 'rb') as file:
    evaluation_log = pickle.load(file)

def convert_to_dataframe(evaluation_log):
    data = []

    for model_name, model_data in evaluation_log.log.items():
        for embedding_technique, embedding_data in model_data.items():
            for dataset_name, dataset_data in embedding_data.items():
                acc_values = []
                rmse_values = []

                for trial_number, trial_data in dataset_data.items():
                    trial_acc_values = evaluation_log.get_metric_values(model_name, embedding_technique, dataset_name, trial_number, 'Test Acc')
                    trial_rmse_values = evaluation_log.get_metric_values(model_name, embedding_technique, dataset_name, trial_number, 'Test RMSE')

                    # Use the last value of each acc and rmse list
                    acc_value = trial_acc_values[-1] if trial_acc_values[-1] is not None else np.nan
                    rmse_value = trial_rmse_values[-1] if trial_rmse_values[-1] is not None else np.nan

                    acc_values.append(acc_value)
                    rmse_values.append(rmse_value)

                print(f"{model_name}, {embedding_technique}, {dataset_name}: acc_values={acc_values}, rmse_values={rmse_values}")

                # Check if the lists are not empty before calculating the mean
                mean_acc = np.mean(acc_values) if acc_values else np.nan
                mean_rmse = np.mean(rmse_values) if rmse_values else np.nan

                data.append({
                    'Model': model_name,
                    'Embedding Technique': embedding_technique,
                    'Dataset': dataset_name,
                    'Mean Test Acc': mean_acc,
                    'Mean Test RMSE': mean_rmse
                })

    df = pd.DataFrame(data)
    return df

df = convert_to_dataframe(evaluation_log)


df.to_csv(r'C:\Users\smbm2\projects\CAT-Transformer\embed_experiments\results.csv', index=False)








