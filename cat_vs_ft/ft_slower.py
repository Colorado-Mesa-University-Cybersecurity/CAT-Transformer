import sys
# sys.path.insert(0, '/home/cscadmin/CyberResearch/CAT-Transformer/model')
sys.path.insert(0, r'C:\Users\smbm2\projects\CAT-Transformer\model')
# sys.path.insert(0, '/home/warin/projects/CAT-Transformer/model')
from testingModel import CATTransformer, Combined_Dataset, train, test, count_parameters
from testingModel import MyFTTransformer
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import pickle
device_in_use = 'cuda'


#Get Helena

df_train = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\helena\train.csv')
df_test = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\helena\test.csv')
df_val = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\helena\validation.csv') #READ FROM RIGHT SPOT

# df_train = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/helena/train.csv')
# df_test = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/helena/test.csv')
# df_val = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/helena/validation.csv')

# df_train = pd.read_csv('/home/warin/projects/CAT-Transformer/datasets/helena/train.csv')
# df_test = pd.read_csv('/home/warin/projects/CAT-Transformer/datasets/helena/test.csv')
# df_val = pd.read_csv('/home/warin/projects/CAT-Transformer/datasets/helena/validation.csv')


# df_train.columns
cont_columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27']
target = ['class']

#CHECKING TO MAKE SURE YOUR LIST IS CORRECT (NO NEED TO TOUCH)
yourlist = cont_columns + target
yourlist.sort()
oglist = list(df_train.columns)
oglist.sort()

assert(yourlist == oglist), "You may of spelled feature name wrong or you forgot to put one of them in the list"

target_classes = [max(len(df_train[target].value_counts()), len(df_val[target].value_counts()),len(df_test[target].value_counts()))]
print("target classes",target_classes)
# Create a StandardScaler and fit it to the cont features
scaler = StandardScaler()
scaler.fit(df_train[cont_columns])

# Transform the training, test, and validation datasets
df_train[cont_columns] = scaler.transform(df_train[cont_columns])
df_test[cont_columns] = scaler.transform(df_test[cont_columns])
df_val[cont_columns] = scaler.transform(df_val[cont_columns])

#Wrapping in Dataset
train_dataset = Combined_Dataset(df_train, cat_columns=[], num_columns=cont_columns, task1_column='class')
val_dataset = Combined_Dataset(df_val, cat_columns=[], num_columns=cont_columns, task1_column='class')
test_dataset = Combined_Dataset(df_test, cat_columns=[], num_columns=cont_columns, task1_column='class')

#This is a hyperparameter that is not tuned. Maybe mess with what makes sense here
batch_size = 256

# Wrapping with DataLoader for easy batch extraction
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


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


evaluation_log = EvaluationLog()

# Adding models, datasets, and metrics
models = ["CAT", "FT"]
embedding_techniques = ["ConstantPL", "PL", "Exp", "L"]
datasets = ["Helena"]
metrics = ["Train Loss", "Test Loss", "Train Acc", "Test Acc"]

for model in models:
    for embedding in embedding_techniques:
        for dataset in datasets:
            evaluation_log.add_dataset(model, embedding, dataset)


########################################################################################################################################################################
#Linear
#CAT
cat_model = CATTransformer(embedding='Linear',
                           n_cont=len(cont_columns),
                       cat_feat=[], 
                       targets_classes=target_classes,
                       embed_size=160
                       ).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=cat_model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_acc = train(regression_on=False, 
                                   dataloader=train_dataloader, 
                                   model=cat_model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc = test(regression_on=False,
                               dataloader=test_dataloader,
                               model=cat_model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_accuracies_1.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies_1.append(test_acc)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
    test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

best_index = test_accuracies_1.index(max(test_accuracies_1))
print(f"Best accuracy {test_accuracies_1[best_index]}")

evaluation_log.add_metric("CAT", "L","Helena", "Train Loss", train_losses)
evaluation_log.add_metric("CAT", "L","Helena", "Test Loss", test_losses)
evaluation_log.add_metric("CAT", "L","Helena", "Train Acc", train_accuracies_1)
evaluation_log.add_metric("CAT", "L","Helena", "Test Acc", test_accuracies_1)


#FT
ft_model = MyFTTransformer(embedding='Linear',
                            n_cont=len(cont_columns),
                       cat_feat=[], 
                       targets_classes=target_classes,
                       embed_size=160
                       ).to(device_in_use)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=ft_model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_acc = train(regression_on=False, 
                                   dataloader=train_dataloader, 
                                   model=ft_model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc = test(regression_on=False,
                               dataloader=test_dataloader,
                               model=ft_model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_accuracies_1.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies_1.append(test_acc)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
    test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

best_index = test_accuracies_1.index(max(test_accuracies_1))
print(f"Best accuracy {test_accuracies_1[best_index]}")

evaluation_log.add_metric("FT", "L","Helena", "Train Loss", train_losses)
evaluation_log.add_metric("FT", "L","Helena", "Test Loss", test_losses)
evaluation_log.add_metric("FT", "L","Helena", "Train Acc", train_accuracies_1)
evaluation_log.add_metric("FT", "L","Helena", "Test Acc", test_accuracies_1)


####################################################################################################################################################################################################################################

# PL
#CAT
cat_model = CATTransformer(embedding='PL',
                           n_cont=len(cont_columns),
                       cat_feat=[], 
                       targets_classes=target_classes,
                       embed_size=160
                       ).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=cat_model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_acc = train(regression_on=False, 
                                   dataloader=train_dataloader, 
                                   model=cat_model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc = test(regression_on=False,
                               dataloader=test_dataloader,
                               model=cat_model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_accuracies_1.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies_1.append(test_acc)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
    test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

best_index = test_accuracies_1.index(max(test_accuracies_1))
print(f"Best accuracy {test_accuracies_1[best_index]}")

evaluation_log.add_metric("CAT", "PL","Helena", "Train Loss", train_losses)
evaluation_log.add_metric("CAT", "PL","Helena", "Test Loss", test_losses)
evaluation_log.add_metric("CAT", "PL","Helena", "Train Acc", train_accuracies_1)
evaluation_log.add_metric("CAT", "PL","Helena", "Test Acc", test_accuracies_1)


#FT
ft_model = MyFTTransformer(embedding='PL',
                            n_cont=len(cont_columns),
                       cat_feat=[], 
                       targets_classes=target_classes,
                       embed_size=160
                       ).to(device_in_use)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=ft_model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_acc = train(regression_on=False, 
                                   dataloader=train_dataloader, 
                                   model=ft_model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc = test(regression_on=False,
                               dataloader=test_dataloader,
                               model=ft_model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_accuracies_1.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies_1.append(test_acc)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
    test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

best_index = test_accuracies_1.index(max(test_accuracies_1))
print(f"Best accuracy {test_accuracies_1[best_index]}")

evaluation_log.add_metric("FT", "PL","Helena", "Train Loss", train_losses)
evaluation_log.add_metric("FT", "PL","Helena", "Test Loss", test_losses)
evaluation_log.add_metric("FT", "PL","Helena", "Train Acc", train_accuracies_1)
evaluation_log.add_metric("FT", "PL","Helena", "Test Acc", test_accuracies_1)

####################################################################################################################################################################################################################################

# ConstantPL
#CAT
cat_model = CATTransformer(embedding='ConstantPL',
                           n_cont=len(cont_columns),
                       cat_feat=[], 
                       targets_classes=target_classes,
                       embed_size=160
                       ).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=cat_model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_acc = train(regression_on=False, 
                                   dataloader=train_dataloader, 
                                   model=cat_model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc = test(regression_on=False,
                               dataloader=test_dataloader,
                               model=cat_model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_accuracies_1.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies_1.append(test_acc)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
    test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

best_index = test_accuracies_1.index(max(test_accuracies_1))
print(f"Best accuracy {test_accuracies_1[best_index]}")

evaluation_log.add_metric("CAT", "ConstantPL","Helena", "Train Loss", train_losses)
evaluation_log.add_metric("CAT", "ConstantPL","Helena", "Test Loss", test_losses)
evaluation_log.add_metric("CAT", "ConstantPL","Helena", "Train Acc", train_accuracies_1)
evaluation_log.add_metric("CAT", "ConstantPL","Helena", "Test Acc", test_accuracies_1)


#FT
ft_model = MyFTTransformer(embedding='ConstantPL',
                            n_cont=len(cont_columns),
                       cat_feat=[], 
                       targets_classes=target_classes,
                       embed_size=160
                       ).to(device_in_use)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=ft_model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_acc = train(regression_on=False, 
                                   dataloader=train_dataloader, 
                                   model=ft_model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc = test(regression_on=False,
                               dataloader=test_dataloader,
                               model=ft_model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_accuracies_1.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies_1.append(test_acc)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
    test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

best_index = test_accuracies_1.index(max(test_accuracies_1))
print(f"Best accuracy {test_accuracies_1[best_index]}")

evaluation_log.add_metric("FT", "ConstantPL","Helena", "Train Loss", train_losses)
evaluation_log.add_metric("FT", "ConstantPL","Helena", "Test Loss", test_losses)
evaluation_log.add_metric("FT", "ConstantPL","Helena", "Train Acc", train_accuracies_1)
evaluation_log.add_metric("FT", "ConstantPL","Helena", "Test Acc", test_accuracies_1)


####################################################################################################################################################################################################################################

# EXP
#CAT
cat_model = CATTransformer(embedding='Exp',
                           n_cont=len(cont_columns),
                       cat_feat=[], 
                       targets_classes=target_classes,
                       embed_size=160
                       ).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=cat_model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_acc = train(regression_on=False, 
                                   dataloader=train_dataloader, 
                                   model=cat_model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc = test(regression_on=False,
                               dataloader=test_dataloader,
                               model=cat_model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_accuracies_1.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies_1.append(test_acc)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
    test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

best_index = test_accuracies_1.index(max(test_accuracies_1))
print(f"Best accuracy {test_accuracies_1[best_index]}")

evaluation_log.add_metric("CAT", "Exp","Helena", "Train Loss", train_losses)
evaluation_log.add_metric("CAT", "Exp","Helena", "Test Loss", test_losses)
evaluation_log.add_metric("CAT", "Exp","Helena", "Train Acc", train_accuracies_1)
evaluation_log.add_metric("CAT", "Exp","Helena", "Test Acc", test_accuracies_1)


#FT
ft_model = MyFTTransformer(embedding='Exp',
                            n_cont=len(cont_columns),
                       cat_feat=[], 
                       targets_classes=target_classes,
                       embed_size=160
                       ).to(device_in_use)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=ft_model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_acc = train(regression_on=False, 
                                   dataloader=train_dataloader, 
                                   model=ft_model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc = test(regression_on=False,
                               dataloader=test_dataloader,
                               model=ft_model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_accuracies_1.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies_1.append(test_acc)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
    test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

best_index = test_accuracies_1.index(max(test_accuracies_1))
print(f"Best accuracy {test_accuracies_1[best_index]}")

evaluation_log.add_metric("FT", "Exp","Helena", "Train Loss", train_losses)
evaluation_log.add_metric("FT", "Exp","Helena", "Test Loss", test_losses)
evaluation_log.add_metric("FT", "Exp","Helena", "Train Acc", train_accuracies_1)
evaluation_log.add_metric("FT", "Exp","Helena", "Test Acc", test_accuracies_1)

with open(r'C:\Users\smbm2\projects\CAT-Transformer\cat_vs_ft\evaluation_log.pkl', 'wb') as file:
    pickle.dump(evaluation_log, file)









