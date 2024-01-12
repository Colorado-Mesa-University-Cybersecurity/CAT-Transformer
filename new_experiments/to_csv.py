import pandas as pd
import pickle
import numpy as np

# Assuming you've already loaded your data into performance_log
with open('/home/wdwatson2/projects/CAT-Transformer/new_experiments/performance_log.pkl', 'rb') as file:
    performance_log = pickle.load(file)

# with open(r'C:\Users\smbm2\projects\CAT-Transformer\new_experiments\performance_log.pkl', 'rb') as file:
#     performance_log = pickle.load(file)

# Initialize lists to store metrics
classification_metrics = ['Test Accuracy', 'Test F1']
regression_metrics = ['Test RMSE']

results = {'Model': [], 'Dataset': [], 'Metric': [], 'Mean': [], 'Std Dev': []}

for model_name in performance_log.log:
    for dataset_name in performance_log.log[model_name]:
        for metric_name in performance_log.log[model_name][dataset_name]:
            values = []
            if metric_name in classification_metrics:
                # Classification metrics (Accuracy, F1)
                for trial in performance_log.log[model_name][dataset_name][metric_name]:
                    trial_values = performance_log.get_metric_values(model_name, dataset_name, metric_name, trial)
                    if trial_values:
                        values.append(trial_values[-1])  # Retrieve last value for each trial
                
                if values:
                    mean = np.mean(values)
                    std_dev = np.std(values)
                    results['Model'].append(model_name)
                    results['Dataset'].append(dataset_name)
                    results['Metric'].append(metric_name)
                    results['Mean'].append(mean)
                    results['Std Dev'].append(std_dev)
                    
            elif metric_name in regression_metrics:
                # Regression metrics (RMSE)
                for trial in performance_log.log[model_name][dataset_name][metric_name]:
                    trial_values = performance_log.get_metric_values(model_name, dataset_name, metric_name, trial)
                    if trial_values:
                        values.append(trial_values[-1])  # Retrieve last value for each trial
                
                if values:
                    mean = np.mean(values)
                    std_dev = np.std(values)
                    results['Model'].append(model_name)
                    results['Dataset'].append(dataset_name)
                    results['Metric'].append(metric_name)
                    results['Mean'].append(mean)
                    results['Std Dev'].append(std_dev)

# Create DataFrame from the results dictionary
df = pd.DataFrame(results)

# Saving the DataFrame to a CSV file
df.to_csv('/home/wdwatson2/projects/CAT-Transformer/new_experiments/performance_metrics.csv', index=False)

# df.to_csv(r'C:\Users\smbm2\projects\CAT-Transformer\new_experiments\performance_metrics.csv', index=False)

