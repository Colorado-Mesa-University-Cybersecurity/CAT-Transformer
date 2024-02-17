import pandas as pd

class Log:
    def __init__(self):
        self.log={}
    
    def add_embedding_technique(self, embedding_technique):
        if embedding_technique not in self.log:
            self.log[embedding_technique] = {}
    def add_dataset(self,embedding_technique, dataset_name):
        if embedding_technique not in self.log:
            self.add_embedding_technique(embedding_technique)
        if dataset_name not in self.log[embedding_technique]:
            self.log[embedding_technique][dataset_name] = {}
    def add_initialization(self, embedding_technique, dataset_name, base):
        if embedding_technique not in self.log:
            self.add_embedding_technique(embedding_technique)
        if dataset_name not in self.log[embedding_technique]:
            self.add_dataset(embedding_technique, dataset_name)
        if base not in self.log[embedding_technique][dataset_name]:
            self.log[embedding_technique][dataset_name][base] = {}
    def add_metric(self, embedding_technique, dataset_name, base, metric_name, values):
        if embedding_technique not in self.log:
            self.add_embedding_technique(embedding_technique)
        if dataset_name not in self.log[embedding_technique]:
            self.add_dataset(embedding_technique, dataset_name)
        if base not in self.log[embedding_technique][dataset_name]:
            self.add_initialization(embedding_technique, dataset_name, base)
        if metric_name not in self.log[embedding_technique][dataset_name][base]:
            self.log[embedding_technique][dataset_name][base][metric_name] = []
        self.log[embedding_technique][dataset_name][base][metric_name].extend(values)
    def get_metric_values(self, embedding_technique, dataset_name, base, metric_name):
        if embedding_technique in self.log and dataset_name in self.log[embedding_technique] \
                and base in self.log[embedding_technique][dataset_name] \
                and metric_name in self.log[embedding_technique][dataset_name][base]:
            return self.log[embedding_technique][dataset_name][base][metric_name]
        else:
            return None
    def add_new_dataset(self, dataset_name):
        for embedding in self.log:
            self.add_dataset(embedding, dataset_name)

    def create_dataframe(self):
        data = {'Embedding Technique': [], 'Dataset': [], 'Base': [], 'Test Accuracy': [], 'Test F1': []}

        for embedding_technique, datasets in self.log.items():
            for dataset_name, bases in datasets.items():
                for base, metrics in bases.items():
                    accuracy_values = self.get_metric_values(embedding_technique, dataset_name, base, 'Test Accuracy')
                    f1_values = self.get_metric_values(embedding_technique, dataset_name, base, 'Test F1')

                    if accuracy_values and f1_values:
                        last_accuracy = accuracy_values[-1]
                        last_f1 = f1_values[-1]

                        data['Embedding Technique'].append(embedding_technique)
                        data['Dataset'].append(dataset_name)
                        data['Base'].append(base)
                        data['Test Accuracy'].append(last_accuracy)
                        data['Test F1'].append(last_f1)

        df = pd.DataFrame(data)
        return df