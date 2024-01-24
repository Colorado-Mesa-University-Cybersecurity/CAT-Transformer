import matplotlib.pyplot as plt
from EvaluationLog import EvaluationLog, plot_train_losses, plot_train_accuracies, plot_test_accuracies, plot_two_accuracies, plot_loss
import pickle

# Load the object
# with open('/home/wdwatson2/projects/CAT-Transformer/embed_experiments/evaluation_log.pkl', 'rb') as file:
#     evaluation_log = pickle.load(file)

with open(r'C:\Users\smbm2\projects\CAT-Transformer\embed_experiments\evaluation_log.pkl', 'rb') as file:
    evaluation_log = pickle.load(file)

# dataset = 'Higgs'

models = ["CAT", "FT"]
embedding_techniques = ["ConstantPL", "PL", "Exp", "L"]
metrics = ["Train Loss", "Test Loss", "Train Acc", "Test Acc"]
datasets = ['Helena', 'Higgs', 'Income', 'California', 'Covertype']

for dataset in datasets:
    plot_loss(evaluation_log, "CAT", dataset)

# plot_two_accuracies(evaluation_log, models, dataset, "ConstantPL")

# # Plot train losses for each model
# for model in models:
#     plot_train_losses(evaluation_log, model, dataset)

# # Plot train losses for each model
# for model in models:
#     plot_train_accuracies(evaluation_log, model,dataset)

# # Plot train losses for each model
# for model in models:
#     plot_test_losses(evaluation_log, model,dataset)

# Plot train losses for each model

for dataset in datasets:
    plot_test_accuracies(evaluation_log, "CAT",dataset)