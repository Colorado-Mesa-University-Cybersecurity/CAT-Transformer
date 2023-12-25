import matplotlib.pyplot as plt
from EvaluationLog import EvaluationLog, plot_train_losses, plot_train_accuracies, plot_test_accuracies, plot_test_losses, plot_two_accuracies
import pickle

# Load the object
with open('/home/wdwatson2/projects/CAT-Transformer/cat_vs_ft/evaluation_log.pkl', 'rb') as file:
    evaluation_log = pickle.load(file)

dataset = 'Aloi'

models = ["CAT", "FT"]
embedding_techniques = ["ConstantPL", "PL", "Exp", "L"]
metrics = ["Train Loss", "Test Loss", "Train Acc", "Test Acc"]

plot_two_accuracies(evaluation_log, models, dataset, "ConstantPL")

# Plot train losses for each model
for model in models:
    plot_train_losses(evaluation_log, model, dataset)

# Plot train losses for each model
for model in models:
    plot_train_accuracies(evaluation_log, model,dataset)

# Plot train losses for each model
for model in models:
    plot_test_losses(evaluation_log, model,dataset)

# Plot train losses for each model
for model in models:
    plot_test_accuracies(evaluation_log, model,dataset)