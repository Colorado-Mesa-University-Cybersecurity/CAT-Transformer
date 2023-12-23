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
from EvaluationLog import EvaluationLog
device_in_use = 'cuda'

#GET Aloi

# df_train = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/aloi/train.csv')
# df_test = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/aloi/test.csv')
# df_val = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/aloi/validation.csv') #READ FROM RIGHT SPOT

df_train = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\aloi\train.csv')
df_test = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\aloi\test.csv')
df_val = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\aloi\validation.csv') #READ FROM RIGHT

cont_columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', 
                '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', 
                '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', 
                '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', 
                '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', 
                '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', 
                '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', 
                '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', 
                '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', 
                '124', '125', '126', '127']
target = ['target']

#CHECKING TO MAKE SURE YOUR LIST IS CORRECT (NO NEED TO TOUCH)
yourlist = cont_columns + target
yourlist.sort()
oglist = list(df_train.columns)
oglist.sort()

assert(yourlist == oglist), "You may of spelled feature name wrong or you forgot to put on of them in the list"

target_classes = [max(len(df_train[target].value_counts()), len(df_val[target].value_counts()),len(df_test[target].value_counts()))]
print(target_classes)
# Create a StandardScaler and fit it to the cont features
scaler = StandardScaler()
scaler.fit(df_train[cont_columns])

# Transform the training, test, and validation datasets
df_train[cont_columns] = scaler.transform(df_train[cont_columns])
df_test[cont_columns] = scaler.transform(df_test[cont_columns])
df_val[cont_columns] = scaler.transform(df_val[cont_columns])

#Wrapping in Dataset
train_dataset = Combined_Dataset(df_train, cat_columns=[], num_columns=cont_columns, task1_column='target')
val_dataset = Combined_Dataset(df_val, cat_columns=[], num_columns=cont_columns, task1_column='target')
test_dataset = Combined_Dataset(df_test, cat_columns=[], num_columns=cont_columns, task1_column='target')

#This is a hyperparameter that is not tuned. Maybe mess with what makes sense here
batch_size = 256

# Wrapping with DataLoader for easy batch extraction
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Load the object
# with open('/home/cscadmin/CyberResearch/CAT-Transformer/cat_vs_ft/evaluation_log.pkl', 'rb') as file:
#     evaluation_log = pickle.load(file)

with open(r'C:\Users\smbm2\projects\CAT-Transformer\cat_vs_ft\evaluation_log.pkl', 'rb') as file:
    evaluation_log = pickle.load(file)

# Adding models, datasets, and metrics
models = ["CAT", "FT"]
embedding_techniques = ["ConstantPL", "PL", "Exp", "L"]
datasets = ["Helena", "Covertype"]
metrics = ["Train Loss", "Test Loss", "Train Acc", "Test Acc"]

evaluation_log.add_new_dataset("Aloi")


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

epochs = 150

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

evaluation_log.add_metric("CAT", "L","Aloi", "Train Loss", train_losses)
evaluation_log.add_metric("CAT", "L","Aloi", "Test Loss", test_losses)
evaluation_log.add_metric("CAT", "L","Aloi", "Train Acc", train_accuracies_1)
evaluation_log.add_metric("CAT", "L","Aloi", "Test Acc", test_accuracies_1)


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

epochs = 150

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

evaluation_log.add_metric("FT", "L","Aloi", "Train Loss", train_losses)
evaluation_log.add_metric("FT", "L","Aloi", "Test Loss", test_losses)
evaluation_log.add_metric("FT", "L","Aloi", "Train Acc", train_accuracies_1)
evaluation_log.add_metric("FT", "L","Aloi", "Test Acc", test_accuracies_1)


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

epochs = 150

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

evaluation_log.add_metric("CAT", "PL","Aloi", "Train Loss", train_losses)
evaluation_log.add_metric("CAT", "PL","Aloi", "Test Loss", test_losses)
evaluation_log.add_metric("CAT", "PL","Aloi", "Train Acc", train_accuracies_1)
evaluation_log.add_metric("CAT", "PL","Aloi", "Test Acc", test_accuracies_1)


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

epochs = 150

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

evaluation_log.add_metric("FT", "PL","Aloi", "Train Loss", train_losses)
evaluation_log.add_metric("FT", "PL","Aloi", "Test Loss", test_losses)
evaluation_log.add_metric("FT", "PL","Aloi", "Train Acc", train_accuracies_1)
evaluation_log.add_metric("FT", "PL","Aloi", "Test Acc", test_accuracies_1)

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

epochs = 150

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

evaluation_log.add_metric("CAT", "ConstantPL","Aloi", "Train Loss", train_losses)
evaluation_log.add_metric("CAT", "ConstantPL","Aloi", "Test Loss", test_losses)
evaluation_log.add_metric("CAT", "ConstantPL","Aloi", "Train Acc", train_accuracies_1)
evaluation_log.add_metric("CAT", "ConstantPL","Aloi", "Test Acc", test_accuracies_1)


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

epochs = 150

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

evaluation_log.add_metric("FT", "ConstantPL","Aloi", "Train Loss", train_losses)
evaluation_log.add_metric("FT", "ConstantPL","Aloi", "Test Loss", test_losses)
evaluation_log.add_metric("FT", "ConstantPL","Aloi", "Train Acc", train_accuracies_1)
evaluation_log.add_metric("FT", "ConstantPL","Aloi", "Test Acc", test_accuracies_1)


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

epochs = 150

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

evaluation_log.add_metric("CAT", "Exp","Aloi", "Train Loss", train_losses)
evaluation_log.add_metric("CAT", "Exp","Aloi", "Test Loss", test_losses)
evaluation_log.add_metric("CAT", "Exp","Aloi", "Train Acc", train_accuracies_1)
evaluation_log.add_metric("CAT", "Exp","Aloi", "Test Acc", test_accuracies_1)


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

epochs = 150

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

evaluation_log.add_metric("FT", "Exp","Aloi", "Train Loss", train_losses)
evaluation_log.add_metric("FT", "Exp","Aloi", "Test Loss", test_losses)
evaluation_log.add_metric("FT", "Exp","Aloi", "Train Acc", train_accuracies_1)
evaluation_log.add_metric("FT", "Exp","Aloi", "Test Acc", test_accuracies_1)

with open(r'C:\Users\smbm2\projects\CAT-Transformer\cat_vs_ft\evaluation_log.pkl', 'wb') as file:
    pickle.dump(evaluation_log, file)

# with open('/home/cscadmin/CyberResearch/CAT-Transformer/cat_vs_ft/evaluation_log.pkl', 'wb') as file:
#     pickle.dump(evaluation_log, file)









