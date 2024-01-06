from helpers import EntropyLog, entropy, evaluate, attn_entropy_get
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/home/wdwatson2/projects/CAT-Transformer/model')
from testingModel import CATTransformer, MyFTTransformer, Combined_Dataset, train, test, EarlyStopping

device_in_use = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device_in_use)

# Load log

with open('/home/wdwatson2/projects/CAT-Transformer/interpretability/entropylog.pkl', 'rb') as file:
    entropylog = pickle.load(file)


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
cat_columns = []

#CHECKING TO MAKE SURE YOUR LIST IS CORRECT (NO NEED TO TOUCH)
yourlist = cont_columns + target
yourlist.sort()
oglist = list(df_train.columns)
oglist.sort()

assert(yourlist == oglist), "You may of spelled feature name wrong or you forgot to put on of them in the list"

cat_features = ()

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

######################################################################################################################################################

#1 layer models

model_cat = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features,
                       targets_classes=target_classes,
                       get_attn=True,
                       num_layers=1).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_cat.parameters(), lr = 0.0001)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 200

for t in range(epochs):
    train_loss, train_acc, attn= train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_cat, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, attn = test(regression_on=False,
                               get_attn=True,
                               dataloader=test_dataloader,
                               model=model_cat,
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

    early_stopping(test_acc)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break


#And a FT

model_ft = MyFTTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features,
                       targets_classes=target_classes,
                       get_attn=True,
                       num_layers=1).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_ft.parameters(), lr = 0.0001)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 200

for t in range(epochs):
    train_loss, train_acc, attn = train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_ft, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, attn = test(regression_on=False,
                               get_attn=True,
                               dataloader=test_dataloader,
                               model=model_ft,
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

    early_stopping(test_acc)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break

attn_entropy_get(entropylog, model_cat, "CAT", 1, "Aloi", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 1, "Aloi", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

data = entropylog.get_data()
print(data)

######################################################################################################################################################

#2 layer models

model_cat = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features,
                       targets_classes=target_classes,
                       get_attn=True,
                       num_layers=2).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_cat.parameters(), lr = 0.0001)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 200

for t in range(epochs):
    train_loss, train_acc, attn= train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_cat, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, attn = test(regression_on=False,
                               get_attn=True,
                               dataloader=test_dataloader,
                               model=model_cat,
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

    early_stopping(test_acc)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break


#And a FT

model_ft = MyFTTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features,
                       targets_classes=target_classes,
                       get_attn=True,
                       num_layers=2).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_ft.parameters(), lr = 0.0001)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 200

for t in range(epochs):
    train_loss, train_acc, attn = train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_ft, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, attn = test(regression_on=False,
                               get_attn=True,
                               dataloader=test_dataloader,
                               model=model_ft,
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

    early_stopping(test_acc)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break

attn_entropy_get(entropylog, model_cat, "CAT", 2, "Aloi", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 2, "Aloi", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

data = entropylog.get_data()
print(data)

###########################################################################################################################################################################################

# 3 layers

model_cat = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features,
                       targets_classes=target_classes,
                       get_attn=True,
                       num_layers=3).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_cat.parameters(), lr = 0.0001)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 200

for t in range(epochs):
    train_loss, train_acc, attn= train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_cat, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, attn = test(regression_on=False,
                               get_attn=True,
                               dataloader=test_dataloader,
                               model=model_cat,
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

    early_stopping(test_acc)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break


#And a FT

model_ft = MyFTTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features,
                       targets_classes=target_classes,
                       get_attn=True,
                       num_layers=3).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_ft.parameters(), lr = 0.0001)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 200

for t in range(epochs):
    train_loss, train_acc, attn = train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_ft, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, attn = test(regression_on=False,
                               get_attn=True,
                               dataloader=test_dataloader,
                               model=model_ft,
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

    early_stopping(test_acc)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break

attn_entropy_get(entropylog, model_cat, "CAT", 3, "Aloi", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 3, "Aloi", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

data = entropylog.get_data()
print(data)

###########################################################################################################################################################################################

# 4 layers

model_cat = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features,
                       targets_classes=target_classes,
                       get_attn=True,
                       num_layers=4).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_cat.parameters(), lr = 0.0001)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 200

for t in range(epochs):
    train_loss, train_acc, attn= train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_cat, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, attn = test(regression_on=False,
                               get_attn=True,
                               dataloader=test_dataloader,
                               model=model_cat,
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

    early_stopping(test_acc)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break


#And a FT

model_ft = MyFTTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features,
                       targets_classes=target_classes,
                       get_attn=True,
                       num_layers=4).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_ft.parameters(), lr = 0.0001)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 200

for t in range(epochs):
    train_loss, train_acc, attn = train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_ft, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, attn = test(regression_on=False,
                               get_attn=True,
                               dataloader=test_dataloader,
                               model=model_ft,
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

    early_stopping(test_acc)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break

attn_entropy_get(entropylog, model_cat, "CAT", 4, "Aloi", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 4, "Aloi", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

data = entropylog.get_data()
print(data)

###########################################################################################################################################################################################

# 5 layers

model_cat = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features,
                       targets_classes=target_classes,
                       get_attn=True,
                       num_layers=5).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_cat.parameters(), lr = 0.0001)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 200

for t in range(epochs):
    train_loss, train_acc, attn= train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_cat, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, attn = test(regression_on=False,
                               get_attn=True,
                               dataloader=test_dataloader,
                               model=model_cat,
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

    early_stopping(test_acc)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break


#And a FT

model_ft = MyFTTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features,
                       targets_classes=target_classes,
                       get_attn=True,
                       num_layers=5).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_ft.parameters(), lr = 0.0001)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 200

for t in range(epochs):
    train_loss, train_acc, attn = train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_ft, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, attn = test(regression_on=False,
                               get_attn=True,
                               dataloader=test_dataloader,
                               model=model_ft,
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

    early_stopping(test_acc)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break

attn_entropy_get(entropylog, model_cat, "CAT", 5, "Aloi", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 5, "Aloi", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

data = entropylog.get_data()
print(data)

###########################################################################################################################################################################################

# 6 layers

model_cat = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features,
                       targets_classes=target_classes,
                       get_attn=True,
                       num_layers=6).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_cat.parameters(), lr = 0.0001)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 200

for t in range(epochs):
    train_loss, train_acc, attn= train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_cat, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, attn = test(regression_on=False,
                               get_attn=True,
                               dataloader=test_dataloader,
                               model=model_cat,
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

    early_stopping(test_acc)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break


#And a FT

model_ft = MyFTTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features,
                       targets_classes=target_classes,
                       get_attn=True,
                       num_layers=6).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_ft.parameters(), lr = 0.0001)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 200

for t in range(epochs):
    train_loss, train_acc, attn = train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_ft, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, attn = test(regression_on=False,
                               get_attn=True,
                               dataloader=test_dataloader,
                               model=model_ft,
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

    early_stopping(test_acc)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break

attn_entropy_get(entropylog, model_cat, "CAT", 6, "Aloi", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 6, "Aloi", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

data = entropylog.get_data()
print(data)

###########################################################################################################################################################################################

# 7 layers

model_cat = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features,
                       targets_classes=target_classes,
                       get_attn=True,
                       num_layers=7).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_cat.parameters(), lr = 0.0001)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 200

for t in range(epochs):
    train_loss, train_acc, attn= train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_cat, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, attn = test(regression_on=False,
                               get_attn=True,
                               dataloader=test_dataloader,
                               model=model_cat,
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

    early_stopping(test_acc)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break


#And a FT

model_ft = MyFTTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features,
                       targets_classes=target_classes,
                       get_attn=True,
                       num_layers=7).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_ft.parameters(), lr = 0.0001)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 200

for t in range(epochs):
    train_loss, train_acc, attn = train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_ft, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, attn = test(regression_on=False,
                               get_attn=True,
                               dataloader=test_dataloader,
                               model=model_ft,
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

    early_stopping(test_acc)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break

attn_entropy_get(entropylog, model_cat, "CAT", 7, "Aloi", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 7, "Aloi", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

data = entropylog.get_data()
print(data)

###########################################################################################################################################################################################

# 8 layers

model_cat = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features,
                       targets_classes=target_classes,
                       get_attn=True,
                       num_layers=8).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_cat.parameters(), lr = 0.0001)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 200

for t in range(epochs):
    train_loss, train_acc, attn= train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_cat, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, attn = test(regression_on=False,
                               get_attn=True,
                               dataloader=test_dataloader,
                               model=model_cat,
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

    early_stopping(test_acc)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break


#And a FT

model_ft = MyFTTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features,
                       targets_classes=target_classes,
                       get_attn=True,
                       num_layers=8).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_ft.parameters(), lr = 0.0001)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 200

for t in range(epochs):
    train_loss, train_acc, attn = train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_ft, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, attn = test(regression_on=False,
                               get_attn=True,
                               dataloader=test_dataloader,
                               model=model_ft,
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

    early_stopping(test_acc)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break

attn_entropy_get(entropylog, model_cat, "CAT", 8, "Aloi", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 8, "Aloi", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

data = entropylog.get_data()
print(data)

###########################################################################################################################################################################################

# 9 layers

model_cat = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features,
                       targets_classes=target_classes,
                       get_attn=True,
                       num_layers=9).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_cat.parameters(), lr = 0.0001)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 200

for t in range(epochs):
    train_loss, train_acc, attn= train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_cat, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, attn = test(regression_on=False,
                               get_attn=True,
                               dataloader=test_dataloader,
                               model=model_cat,
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

    early_stopping(test_acc)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break


#And a FT

model_ft = MyFTTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features,
                       targets_classes=target_classes,
                       get_attn=True,
                       num_layers=9).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_ft.parameters(), lr = 0.0001)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 200

for t in range(epochs):
    train_loss, train_acc, attn = train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_ft, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, attn = test(regression_on=False,
                               get_attn=True,
                               dataloader=test_dataloader,
                               model=model_ft,
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

    early_stopping(test_acc)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break

attn_entropy_get(entropylog, model_cat, "CAT", 9, "Aloi", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 9, "Aloi", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

data = entropylog.get_data()
print(data)

###############################################################################################################################################################################

# 10 layers

model_cat = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features,
                       targets_classes=target_classes,
                       get_attn=True,
                       num_layers=10).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_cat.parameters(), lr = 0.0001)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 250

for t in range(epochs):
    train_loss, train_acc, attn= train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_cat, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, attn = test(regression_on=False,
                               get_attn=True,
                               dataloader=test_dataloader,
                               model=model_cat,
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

    early_stopping(test_acc)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break


#And a FT

model_ft = MyFTTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features,
                       targets_classes=target_classes,
                       get_attn=True,
                       num_layers=10).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_ft.parameters(), lr = 0.0001)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 250

for t in range(epochs):
    train_loss, train_acc, attn = train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_ft, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, attn = test(regression_on=False,
                               get_attn=True,
                               dataloader=test_dataloader,
                               model=model_ft,
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

    early_stopping(test_acc)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break

attn_entropy_get(entropylog, model_cat, "CAT", 10, "Aloi",df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 10, "Aloi",df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

data = entropylog.get_data()
print(data)

###############################################################################################################################################################################

# 15 layers

model_cat = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features,
                       targets_classes=target_classes,
                       get_attn=True,
                       num_layers=15).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_cat.parameters(), lr = 0.0001)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 250

for t in range(epochs):
    train_loss, train_acc, attn= train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_cat, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, attn = test(regression_on=False,
                               get_attn=True,
                               dataloader=test_dataloader,
                               model=model_cat,
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

    early_stopping(test_acc)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break


#And a FT

model_ft = MyFTTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features,
                       targets_classes=target_classes,
                       get_attn=True,
                       num_layers=15).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_ft.parameters(), lr = 0.0001)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 250

for t in range(epochs):
    train_loss, train_acc, attn = train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_ft, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, attn = test(regression_on=False,
                               get_attn=True,
                               dataloader=test_dataloader,
                               model=model_ft,
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

    early_stopping(test_acc)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break

attn_entropy_get(entropylog, model_cat, "CAT", 15, "Aloi",df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 15, "Aloi",df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

data = entropylog.get_data()
print(data)

#####################################################################################################################################################################################

#20 layers

model_cat = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features,
                       targets_classes=target_classes,
                       get_attn=True,
                       num_layers=20).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_cat.parameters(), lr = 0.0001)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 250

for t in range(epochs):
    train_loss, train_acc, attn= train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_cat, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, attn = test(regression_on=False,
                               get_attn=True,
                               dataloader=test_dataloader,
                               model=model_cat,
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

    early_stopping(test_acc)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break


#And a FT

model_ft = MyFTTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features,
                       targets_classes=target_classes,
                       get_attn=True,
                       num_layers=20).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_ft.parameters(), lr = 0.0001)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 250

for t in range(epochs):
    train_loss, train_acc, attn = train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_ft, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, attn = test(regression_on=False,
                               get_attn=True,
                               dataloader=test_dataloader,
                               model=model_ft,
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

    early_stopping(test_acc)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break

attn_entropy_get(entropylog, model_cat, "CAT", 20, "Aloi",df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 20, "Aloi",df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

data = entropylog.get_data()
print(data)


with open('/home/wdwatson2/projects/CAT-Transformer/interpretability/entropylog.pkl', 'wb') as file:
    pickle.dump(entropylog, file)













