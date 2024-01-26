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
    evaluation_log = pickle.load(file)

# Create an empty DataFrame
columns = ['Model', 'Embedding Technique', 'Dataset', 'Mean Test Acc', 'Std Test Acc', 'Mean Test RMSE', 'Std Test RMSE']
result_df = pd.DataFrame(columns=columns)

# Iterate over the data in your EvaluationLog
for model_name in evaluation_log.log:
    for embedding_technique in evaluation_log.log[model_name]:
        for dataset_name in evaluation_log.log[model_name][embedding_technique]:
            trial_numbers = list(evaluation_log.log[model_name][embedding_technique][dataset_name].keys())

            # Calculate mean and std for Test Acc
            test_acc_values = []
            for trial_number in trial_numbers:
                values = evaluation_log.get_metric_values(model_name, embedding_technique, dataset_name, trial_number, 'Test Acc')[-1]
                if values:
                    test_acc_values.append(values)
            
            mean_test_acc = pd.Series(test_acc_values).mean()
            std_test_acc = pd.Series(test_acc_values).std()

            # Calculate mean and std for Test RMSE
            test_rmse_values = []
            for trial_number in trial_numbers:
                values = evaluation_log.get_metric_values(model_name, embedding_technique, dataset_name, trial_number, 'Test RMSE')[-1]
                if values:
                    test_rmse_values.append(values)
            
            mean_test_rmse = pd.Series(test_rmse_values).mean()
            std_test_rmse = pd.Series(test_rmse_values).std()

            # Append the results to the DataFrame
            result_df = result_df.append({
                'Model': model_name,
                'Embedding Technique': embedding_technique,
                'Dataset': dataset_name,
                'Mean Test Acc': mean_test_acc,
                'Std Test Acc': std_test_acc,
                'Mean Test RMSE': mean_test_rmse,
                'Std Test RMSE': std_test_rmse
            }, ignore_index=True)


result_df.to_csv(r'C:\Users\smbm2\projects\CAT-Transformer\embed_experiments\results.csv', index=False)








