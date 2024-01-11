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

# Jannis

df_train = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/jannis/train.csv')
df_test = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/jannis/test.csv')
df_val = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/jannis/validation.csv') 

cont_columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30',
       'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40',
       'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50',
       'V51', 'V52', 'V53', 'V54']
cat_columns = []
target = ['class']

#CHECKING TO MAKE SURE YOUR LIST IS CORRECT (NO NEED TO TOUCH)
yourlist = cont_columns + cat_columns+target
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
train_dataset = Combined_Dataset(df_train, cat_columns=cat_columns, num_columns=cont_columns, task1_column=target[0])
val_dataset = Combined_Dataset(df_val, cat_columns=cat_columns, num_columns=cont_columns, task1_column=target[0])
test_dataset = Combined_Dataset(df_test, cat_columns=cat_columns, num_columns=cont_columns, task1_column=target[0])

batch_size = 256

# Wrapping with DataLoader for easy batch extraction
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

######################################################################################################################################################

#10 layer models

model_cat = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features,
                       targets_classes=target_classes,
                       get_attn=True,
                       num_layers=10).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_cat.parameters(), lr = 0.0005)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 800

for t in range(epochs):
    train_loss, train_acc, train_f1, attn= train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_cat, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, test_f1, attn = test(regression_on=False,
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
optimizer = torch.optim.Adam(params=model_ft.parameters(), lr = 0.0005)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 800

for t in range(epochs):
    train_loss, train_acc, train_f1, attn = train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_ft, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, test_f1, attn = test(regression_on=False,
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

attn_entropy_get(entropylog, model_cat, "CAT", 10, "Jannis", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 10, "Jannis", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

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
optimizer = torch.optim.Adam(params=model_cat.parameters(), lr = 0.0005)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 800

for t in range(epochs):
    train_loss, train_acc, train_f1, attn= train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_cat, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, test_f1, attn = test(regression_on=False,
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
optimizer = torch.optim.Adam(params=model_ft.parameters(), lr = 0.0005)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 800

for t in range(epochs):
    train_loss, train_acc, train_f1, attn = train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_ft, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, test_f1, attn = test(regression_on=False,
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

attn_entropy_get(entropylog, model_cat, "CAT", 5, "Jannis" ,df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 5, "Jannis" ,df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

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
optimizer = torch.optim.Adam(params=model_cat.parameters(), lr = 0.0005)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 800

for t in range(epochs):
    train_loss, train_acc, train_f1, attn= train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_cat, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, test_f1, attn = test(regression_on=False,
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
optimizer = torch.optim.Adam(params=model_ft.parameters(), lr = 0.0005)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 800

for t in range(epochs):
    train_loss, train_acc, train_f1, attn = train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_ft, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, test_f1, attn = test(regression_on=False,
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

attn_entropy_get(entropylog, model_cat, "CAT", 15, "Jannis" ,df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 15, "Jannis" ,df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

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
optimizer = torch.optim.Adam(params=model_cat.parameters(), lr = 0.0005)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 800

for t in range(epochs):
    train_loss, train_acc, train_f1, attn= train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_cat, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, test_f1, attn = test(regression_on=False,
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
optimizer = torch.optim.Adam(params=model_ft.parameters(), lr = 0.0005)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 800

for t in range(epochs):
    train_loss, train_acc, train_f1, attn = train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_ft, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, test_f1, attn = test(regression_on=False,
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

attn_entropy_get(entropylog, model_cat, "CAT", 20, "Jannis" ,df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 20, "Jannis" ,df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

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
optimizer = torch.optim.Adam(params=model_cat.parameters(), lr = 0.0005)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 800

for t in range(epochs):
    train_loss, train_acc, train_f1, attn= train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_cat, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, test_f1, attn = test(regression_on=False,
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
optimizer = torch.optim.Adam(params=model_ft.parameters(), lr = 0.0005)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 800

for t in range(epochs):
    train_loss, train_acc, train_f1, attn = train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_ft, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, test_f1, attn = test(regression_on=False,
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

attn_entropy_get(entropylog, model_cat, "CAT", 2, "Jannis", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 2, "Jannis", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

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
optimizer = torch.optim.Adam(params=model_cat.parameters(), lr = 0.0005)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 800

for t in range(epochs):
    train_loss, train_acc, train_f1, attn= train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_cat, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, test_f1, attn = test(regression_on=False,
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
optimizer = torch.optim.Adam(params=model_ft.parameters(), lr = 0.0005)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 800

for t in range(epochs):
    train_loss, train_acc, train_f1, attn = train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_ft, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, test_f1, attn = test(regression_on=False,
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

attn_entropy_get(entropylog, model_cat, "CAT", 4, "Jannis", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 4, "Jannis", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

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
optimizer = torch.optim.Adam(params=model_cat.parameters(), lr = 0.0005)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 800

for t in range(epochs):
    train_loss, train_acc, train_f1, attn= train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_cat, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, test_f1, attn = test(regression_on=False,
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
optimizer = torch.optim.Adam(params=model_ft.parameters(), lr = 0.0005)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 800

for t in range(epochs):
    train_loss, train_acc, train_f1, attn = train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_ft, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, test_f1, attn = test(regression_on=False,
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

attn_entropy_get(entropylog, model_cat, "CAT", 6, "Jannis", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 6, "Jannis", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

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
optimizer = torch.optim.Adam(params=model_cat.parameters(), lr = 0.0005)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 800

for t in range(epochs):
    train_loss, train_acc, train_f1, attn= train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_cat, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, test_f1, attn = test(regression_on=False,
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
optimizer = torch.optim.Adam(params=model_ft.parameters(), lr = 0.0005)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 800

for t in range(epochs):
    train_loss, train_acc, train_f1, attn = train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_ft, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, test_f1, attn = test(regression_on=False,
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

attn_entropy_get(entropylog, model_cat, "CAT", 7, "Jannis", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 7, "Jannis", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

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
optimizer = torch.optim.Adam(params=model_cat.parameters(), lr = 0.0005)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 800

for t in range(epochs):
    train_loss, train_acc, train_f1, attn= train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_cat, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, test_f1, attn = test(regression_on=False,
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
optimizer = torch.optim.Adam(params=model_ft.parameters(), lr = 0.0005)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 800

for t in range(epochs):
    train_loss, train_acc, train_f1, attn = train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_ft, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, test_f1, attn = test(regression_on=False,
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

attn_entropy_get(entropylog, model_cat, "CAT", 8, "Jannis", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 8, "Jannis", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

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
optimizer = torch.optim.Adam(params=model_cat.parameters(), lr = 0.0005)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 800

for t in range(epochs):
    train_loss, train_acc, train_f1, attn= train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_cat, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, test_f1, attn = test(regression_on=False,
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
optimizer = torch.optim.Adam(params=model_ft.parameters(), lr = 0.0005)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 800

for t in range(epochs):
    train_loss, train_acc, train_f1, attn = train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_ft, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, test_f1, attn = test(regression_on=False,
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

attn_entropy_get(entropylog, model_cat, "CAT", 9, "Jannis", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 9, "Jannis", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

data = entropylog.get_data()
print(data)


#####################################################################################################################################################

# 1 layer models

model_cat = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features,
                       targets_classes=target_classes,
                       get_attn=True,
                       num_layers=1).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_cat.parameters(), lr = 0.0005)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 800

for t in range(epochs):
    train_loss, train_acc, train_f1, attn= train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_cat, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, test_f1, attn = test(regression_on=False,
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
optimizer = torch.optim.Adam(params=model_ft.parameters(), lr = 0.0005)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 800

for t in range(epochs):
    train_loss, train_acc, train_f1, attn = train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_ft, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, test_f1, attn = test(regression_on=False,
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

attn_entropy_get(entropylog, model_cat, "CAT", 1, "Jannis", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 1, "Jannis", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

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
optimizer = torch.optim.Adam(params=model_cat.parameters(), lr = 0.0005)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 800

for t in range(epochs):
    train_loss, train_acc, train_f1, attn= train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_cat, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, test_f1, attn = test(regression_on=False,
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
optimizer = torch.optim.Adam(params=model_ft.parameters(), lr = 0.0005)

early_stopping = EarlyStopping(patience=10, verbose=True)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 800

for t in range(epochs):
    train_loss, train_acc, train_f1, attn = train(regression_on=False, 
                                  get_attn=True,
                                   dataloader=train_dataloader, 
                                   model=model_ft, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc, test_f1, attn = test(regression_on=False,
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

attn_entropy_get(entropylog, model_cat, "CAT", 3, "Jannis" ,df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 3, "Jannis" ,df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

data = entropylog.get_data()
print(data)


with open('/home/wdwatson2/projects/CAT-Transformer/interpretability/entropylog.pkl', 'wb') as file:
    pickle.dump(entropylog, file)