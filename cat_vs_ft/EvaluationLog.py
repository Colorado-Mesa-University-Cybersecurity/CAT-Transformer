

class EvaluationLog:
    def __init__(self):
        self.log = {}  # Dictionary to store data: {model_name: {embedding_technique: {dataset_name: {metric_name: [values]}}}}

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

    def add_metric(self, model_name, embedding_technique, dataset_name, metric_name, values):
        if model_name not in self.log:
            self.add_model(model_name)
        if embedding_technique not in self.log[model_name]:
            self.add_embedding_technique(model_name, embedding_technique)
        if dataset_name not in self.log[model_name][embedding_technique]:
            self.add_dataset(model_name, embedding_technique, dataset_name)
        if metric_name not in self.log[model_name][embedding_technique][dataset_name]:
            self.log[model_name][embedding_technique][dataset_name][metric_name] = []
        self.log[model_name][embedding_technique][dataset_name][metric_name].extend(values)

    def get_metric_values(self, model_name, embedding_technique, dataset_name, metric_name):
        if model_name in self.log and embedding_technique in self.log[model_name] \
                and dataset_name in self.log[model_name][embedding_technique] \
                and metric_name in self.log[model_name][embedding_technique][dataset_name]:
            return self.log[model_name][embedding_technique][dataset_name][metric_name]
        else:
            return None

    def add_new_dataset(self, dataset_name):
        for model in self.log:
            for embedding in self.log[model]:
                self.add_dataset(model, embedding, dataset_name)

    def add_metric_for_dataset(self, model_name, embedding_technique, dataset_name, metric_name, values):
        if model_name in self.log and embedding_technique in self.log[model_name] \
                and dataset_name in self.log[model_name][embedding_technique]:
            if metric_name not in self.log[model_name][embedding_technique][dataset_name]:
                self.log[model_name][embedding_technique][dataset_name][metric_name] = []
            self.log[model_name][embedding_technique][dataset_name][metric_name].extend(values)

import matplotlib.pyplot as plt

models = ["CAT", "FT"]
embedding_techniques = ["ConstantPL", "PL", "Exp", "L"]
metrics = ["Train Loss", "Test Loss", "Train Acc", "Test Acc"]

def plot_train_losses(evaluation_log, model_name, dataset_name):
    plt.figure(figsize=(10, 6))
    for embedding in embedding_techniques:
        train_losses = evaluation_log.get_metric_values(model_name, embedding, dataset_name, "Train Loss")
        if train_losses:
            plt.plot(range(len(train_losses)), train_losses, label=f"{embedding}")
    plt.title(f"{model_name} Train Losses for Different Embedding Schemes")
    plt.xlabel("Epochs")
    plt.ylabel("Train Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_train_accuracies(evaluation_log, model_name, dataset_name):
    plt.figure(figsize=(10, 6))
    for embedding in embedding_techniques:
        train_losses = evaluation_log.get_metric_values(model_name, embedding, dataset_name, "Train Acc")
        if train_losses:
            plt.plot(range(len(train_losses)), train_losses, label=f"{embedding}")
    plt.title(f"{model_name} Train Accuracies for Different Embedding Schemes")
    plt.xlabel("Epochs")
    plt.ylabel("Train Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_test_losses(evaluation_log, model_name, dataset_name):
    plt.figure(figsize=(10, 6))
    for embedding in embedding_techniques:
        train_losses = evaluation_log.get_metric_values(model_name, embedding, dataset_name, "Test Loss")
        if train_losses:
            plt.plot(range(len(train_losses)), train_losses, label=f"{embedding}")
    plt.title(f"{model_name} Test Losses for Different Embedding Schemes")
    plt.xlabel("Epochs")
    plt.ylabel("Test Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_test_accuracies(evaluation_log, model_name, dataset_name):
    plt.figure(figsize=(10, 6))
    for embedding in embedding_techniques:
        train_losses = evaluation_log.get_metric_values(model_name, embedding, dataset_name, "Test Acc")
        if train_losses:
            plt.plot(range(len(train_losses)), train_losses, label=f"{embedding}")
    plt.title(f"{model_name} Test Accuracies for Different Embedding Schemes")
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()