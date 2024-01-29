import pandas as pd

class EvaluationLog:
    def __init__(self):
        self.log = {}  # Dictionary to store data: {model_name: {embedding_technique: {dataset_name: {trial_number: {metric_name: [values]}}}}}

    def add_model(self, model_name):
        if model_name not in self.log:
            self.log[model_name] = {}

    def add_embedding_technique(self, model_name, embedding_technique):
        if model_name not in self.log:
            self.add_model(model_name)
        if embedding_technique not in self.log[model_name]:
            self.log[model_name][embedding_technique] = {}

    def add_dataset(self, model_name, embedding_technique, dataset_name):
        if model_name not in self.log:
            self.add_model(model_name)
        if embedding_technique not in self.log[model_name]:
            self.add_embedding_technique(model_name, embedding_technique)
        if dataset_name not in self.log[model_name][embedding_technique]:
            self.log[model_name][embedding_technique][dataset_name] = {}

    def add_metric(self, model_name, embedding_technique, dataset_name, trial_number, metric_name, values):
        if model_name not in self.log:
            self.add_model(model_name)
        if embedding_technique not in self.log[model_name]:
            self.add_embedding_technique(model_name, embedding_technique)
        if dataset_name not in self.log[model_name][embedding_technique]:
            self.add_dataset(model_name, embedding_technique, dataset_name)
        if trial_number not in self.log[model_name][embedding_technique][dataset_name]:
            self.log[model_name][embedding_technique][dataset_name][trial_number] = {}
        if metric_name not in self.log[model_name][embedding_technique][dataset_name][trial_number]:
            self.log[model_name][embedding_technique][dataset_name][trial_number][metric_name] = []
        self.log[model_name][embedding_technique][dataset_name][trial_number][metric_name].extend(values)

    def get_metric_values(self, model_name, embedding_technique, dataset_name, trial_number, metric_name):
        if model_name in self.log and embedding_technique in self.log[model_name] \
                and dataset_name in self.log[model_name][embedding_technique] \
                and trial_number in self.log[model_name][embedding_technique][dataset_name] \
                and metric_name in self.log[model_name][embedding_technique][dataset_name][trial_number]:
            return self.log[model_name][embedding_technique][dataset_name][trial_number][metric_name]
        else:
            return [None]

    def add_new_dataset(self, dataset_name):
        for model in self.log:
            for embedding in self.log[model]:
                self.add_dataset(model, embedding, dataset_name)

    def add_metric_for_trial(self, model_name, embedding_technique, dataset_name, trial_number, metric_name, values):
        self.add_metric(model_name, embedding_technique, dataset_name, trial_number, metric_name, values)


import matplotlib.pyplot as plt

models = ["CAT", "FT"]
embedding_techniques = ["ConstantPL", "PL", "ExpFF", "L"]
new_labels = ["RFF - L", "Periodic - L", "Log-linear - L", "L"]
metrics = ["Train Loss", "Test Loss", "Train Acc", "Test Acc", "Train RMSE", "Test RMSE"]
ieee_colors = ['#ca0020', '#0571b0', '#92c5de','#f4a582']
# ieee_colors = ['#a1dab4', '#2c7fb8', '#41b6c4','#253494']

def plot_test_accuracies(evaluation_log, model_name, dataset_name, trial_number=0):
    plt.figure(figsize=(10, 6))
    for i, (embedding, name) in enumerate(zip(embedding_techniques, new_labels)):
        test_accuracies = evaluation_log.get_metric_values(model_name, embedding, dataset_name, trial_number, "Test Acc")
        if test_accuracies:
            plt.plot(range(len(test_accuracies)), test_accuracies, label=f"{name}", color=ieee_colors[i % len(ieee_colors)])
    plt.title(f"{model_name} Test Accuracies for Different Embedding Schemes - {dataset_name} (Trial {trial_number + 1})")
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_test_rmses(evaluation_log, model_name, dataset_name, trial_number=0):
    plt.figure(figsize=(10, 6))
    for i, (embedding, name) in enumerate(zip(embedding_techniques, new_labels)):
        test_accuracies = evaluation_log.get_metric_values(model_name, embedding, dataset_name, trial_number, "Test RMSE")
        if test_accuracies:
            plt.plot(range(len(test_accuracies)), test_accuracies, label=f"{name}", color=ieee_colors[i % len(ieee_colors)])
    plt.title(f"{model_name} Test RMSE for Different Embedding Schemes - {dataset_name} (Trial {trial_number + 1})")
    plt.xlabel("Epochs")
    plt.ylabel("Test RMSE")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_loss(evaluation_log, model_name, dataset_name, trial_number=0):
    plt.figure(figsize=(12, 8))
    
    for i, (embedding, name) in enumerate(zip(embedding_techniques, new_labels)):
        train_losses = evaluation_log.get_metric_values(model_name, embedding, dataset_name, trial_number, "Train Loss")
        test_losses = evaluation_log.get_metric_values(model_name, embedding, dataset_name, trial_number, "Test Loss")
        
        if train_losses:
            plt.plot(range(len(train_losses)), train_losses, label=f"{name} - Train", color=ieee_colors[i % len(ieee_colors)], linewidth=2)
            if test_losses:
                plt.plot(range(len(test_losses)), test_losses, label=f"{name} - Test", color=ieee_colors[i % len(ieee_colors)], linestyle='dotted', linewidth=2)

    plt.title(f"{model_name} Train and Test Losses for Different Embedding Schemes - {dataset_name} (Trial {trial_number + 1})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

