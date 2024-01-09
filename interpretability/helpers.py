import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch 
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, '/home/wdwatson2/projects/CAT-Transformer/model')
# sys.path.insert(0, r'C:\Users\smbm2\projects\CAT-Transformer\model')
from testingModel import Combined_Dataset

class TrainingAttnScoresLog:
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

class EntropyLog:
    def __init__(self):
        self.data = {}  # Initialize the data structure

    def add_entry(self, model, dataset, split, class_name, layers, attn_list, entropy):
        if model not in self.data:
            self.data[model] = {'Layers': {}}
        if layers not in self.data[model]['Layers']:
            self.data[model]['Layers'][layers] = {}
        if dataset not in self.data[model]['Layers'][layers]:
            self.data[model]['Layers'][layers][dataset] = {}
        if split not in self.data[model]['Layers'][layers][dataset]:
            self.data[model]['Layers'][layers][dataset][split] = {}
        if class_name not in self.data[model]['Layers'][layers][dataset][split]:
            self.data[model]['Layers'][layers][dataset][split][class_name] = {
                'Attention': attn_list,
                'Entropy': entropy
            }

    def get_entropy(self, model, layers, dataset, split, class_name):
        if model in self.data and 'Layers' in self.data[model] and layers in self.data[model]['Layers'] \
                and dataset in self.data[model]['Layers'][layers] \
                and split in self.data[model]['Layers'][layers][dataset] \
                and class_name in self.data[model]['Layers'][layers][dataset][split]:
            return self.data[model]['Layers'][layers][dataset][split][class_name]['Entropy']
        else:
            return None

    def get_data(self):
        return self.data

def entropy(distribution):
    probabilities = distribution / np.sum(distribution)  # Normalize probabilities
    entropy_val = -np.sum(probabilities * np.log2(probabilities + 1e-12))  # Add small value to avoid log(0)
    return entropy_val

def evaluate(model, dataloader, device_in_use):
    model.eval()  # Set model to evaluation mode
    accuracies = []
    attentions = []

    with torch.no_grad():
        for (cat_x, cont_x, labels) in dataloader:
            cat_x,cont_x,labels=cat_x.to(device_in_use),cont_x.to(device_in_use),labels.to(device_in_use)

            predictions, attention = model(cat_x, cont_x)

            _, predicted = torch.max(predictions, 1)
            accuracy = (predicted == labels).sum().item() / labels.size(0)
            accuracies.append(accuracy)

            attentions.append(attention.cpu().numpy()) 

    avg_accuracy = np.mean(accuracies)
    all_attentions = np.concatenate(attentions, axis=0) if attentions else None

    return avg_accuracy, all_attentions

def attn_entropy_get(log:EntropyLog, trained_model, model_name, num_layers, dataset_name, df_train:pd.DataFrame, df_test:pd.DataFrame, target, cat_columns, num_columns, device_in_use):
    classes = np.unique(df_train[target])

    for x in classes:
        class_train_samples = df_train.loc[df_train[target] == x]
        class_test_samples = df_test.loc[df_test[target] == x]

        class_train_dataset = Combined_Dataset(class_train_samples, cat_columns=cat_columns, num_columns=num_columns, task1_column=target)
        class_test_dataset = Combined_Dataset(class_test_samples, cat_columns=cat_columns, num_columns=num_columns, task1_column=target)

        # class_train_dataloader = DataLoader(class_train_dataset, batch_size=len(class_train_dataset))
        # class_test_dataloader = DataLoader(class_test_dataset, batch_size=len(class_test_dataset))

        class_train_dataloader = DataLoader(class_train_dataset, batch_size=512)
        class_test_dataloader = DataLoader(class_test_dataset, batch_size=512)
             
        train_acc, train_attn = evaluate(trained_model, class_train_dataloader, device_in_use)
        train_attn = train_attn.mean(0)
        test_acc, test_attn = evaluate(trained_model, class_test_dataloader, device_in_use)
        test_attn = test_attn.mean(0)

        log.add_entry(model_name, dataset_name, "train", "class_"+str(x) , num_layers, train_attn, entropy(train_attn))
        log.add_entry(model_name, dataset_name, "test", "class_"+str(x) , num_layers, test_attn, entropy(test_attn))

def extract_entropy_scores(entropy_log):
    entropy_scores = []

    for model, model_data in entropy_log.get_data().items():
        for layers, layer_data in model_data.get('Layers', {}).items():
            for dataset, dataset_data in layer_data.items():
                for split, split_data in dataset_data.items():
                    for class_name, class_data in split_data.items():
                        entropy = class_data.get('Entropy')
                        if entropy is not None:
                            entropy_scores.append({
                                'Model': model,
                                'Layers': layers,
                                'Dataset': dataset,
                                'Split': split,
                                'Class': class_name,
                                'Entropy': entropy
                            })

    return entropy_scores

def format_table_to_dataframe(table):
    headers = ["Income", "5", "10", "15", "20"]
    data = []

    for class_name, class_data in table.items():
        for layers, layers_data in class_data['Income'].items():
            for split, split_data in layers_data.items():
                for dataset, dataset_data in split_data.items():
                    for model, entropy in dataset_data.items():
                        row = {
                            'Class': class_name,
                            'Layers': layers,
                            'Split': split,
                            'Dataset': dataset,
                            'Model': model,
                            'Entropy': entropy
                        }
                        data.append(row)

    df = pd.DataFrame(data)
    return df

def build_table(entropy_log):
    table = {}

    for model, model_data in entropy_log.get_data().items():
        if 'Layers' in model_data:
            for layers, layer_data in model_data['Layers'].items():
                for dataset, dataset_data in layer_data.items():
                    for split, split_data in dataset_data.items():
                        for class_name, class_data in split_data.items():
                            if class_name not in table:
                                table[class_name] = {'Income': {}}

                            if layers not in table[class_name]['Income']:
                                table[class_name]['Income'][layers] = {}

                            if split not in table[class_name]['Income'][layers]:
                                table[class_name]['Income'][layers][split] = {}

                            if dataset not in table[class_name]['Income'][layers][split]:
                                table[class_name]['Income'][layers][split][dataset] = {}

                            table[class_name]['Income'][layers][split][dataset][model] = class_data['Entropy']

    return table