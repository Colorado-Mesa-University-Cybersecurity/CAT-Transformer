import pickle
import matplotlib.pyplot as plt
import numpy as np

class AttentionScoresLog:
    def __init__(self):
        self.log = {}  # {model_name: {num_layers: {dataset_name: {'train_scores': [], 'test_scores': []}}}}

    def add_model_layers(self, model_name, num_layers):
        if model_name not in self.log:
            self.log[model_name] = {}
        if num_layers not in self.log[model_name]:
            self.log[model_name][num_layers] = {}

    def add_dataset(self, model_name, num_layers, dataset_name):
        if model_name in self.log and num_layers in self.log[model_name]:
            if dataset_name not in self.log[model_name][num_layers]:
                self.log[model_name][num_layers][dataset_name] = {'train_scores': [], 'test_scores': []}

    def add_attention_scores(self, model_name, num_layers, dataset_name, train_scores, test_scores):
        if model_name in self.log and num_layers in self.log[model_name] \
                and dataset_name in self.log[model_name][num_layers]:
            self.log[model_name][num_layers][dataset_name]['train_scores'].append(train_scores)
            self.log[model_name][num_layers][dataset_name]['test_scores'].append(test_scores)

    def get_attention_scores(self, model_name, num_layers, dataset_name):
        if model_name in self.log and num_layers in self.log[model_name] \
                and dataset_name in self.log[model_name][num_layers]:
            return self.log[model_name][num_layers][dataset_name]['train_scores'], \
                   self.log[model_name][num_layers][dataset_name]['test_scores']
        else:
            return None, None

    def add_new_dataset(self, dataset_name):
        for model in self.log:
            for layers in self.log[model]:
                self.add_dataset(model, layers, dataset_name)

    def save_log_as_pickle(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.log, file)

    @classmethod
    def load_log_from_pickle(cls, filename):
        instance = cls()
        with open(filename, 'rb') as file:
            instance.log = pickle.load(file)
        return instance



models = ["CAT", "FT"]
embedding_techniques = ["ConstantPL", "PL", "Exp", "L"]
metrics = ["Train Loss", "Test Loss", "Train Acc", "Test Acc"]

def plot_two_accuracies(evaluation_log, model_name, dataset_name, embedding):
    plt.figure(figsize=(10, 6))
    for model in model_name:
        test_acc = evaluation_log.get_metric_values(model, embedding, dataset_name, "Test Acc")
        if test_acc:
            plt.plot(range(len(test_acc)), test_acc, label=f"{model}")
    plt.title(f"{model_name} Test Accuracies")
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

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