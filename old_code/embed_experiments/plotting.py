import matplotlib.pyplot as plt
from EvaluationLog import EvaluationLog, plot_test_accuracies, plot_loss, plot_test_rmses
import pickle

# Load the object
# with open('/home/wdwatson2/projects/CAT-Transformer/embed_experiments/evaluation_log.pkl', 'rb') as file:
#     evaluation_log = pickle.load(file)

with open(r'C:\Users\smbm2\projects\CAT-Transformer\embed_experiments\evaluation_log.pkl', 'rb') as file:
    evaluation_log = pickle.load(file)

# dataset = 'Higgs'

models = ["CAT", "FT"]
embedding_techniques = ["ConstantPL", "PL", "ExpFF", "L"]
metrics = ["Train Loss", "Test Loss", "Train Acc", "Test Acc"]
datasets = ['Helena', 'Higgs', 'Income', 'California', 'Covertype']

trial_num = 2

for dataset in datasets:
    plot_loss(evaluation_log, "CAT", dataset, trial_num)

for dataset in datasets:
    plot_test_accuracies(evaluation_log, "CAT",dataset, trial_num)

for dataset in datasets:
    plot_test_rmses(evaluation_log, "CAT",dataset, trial_num)