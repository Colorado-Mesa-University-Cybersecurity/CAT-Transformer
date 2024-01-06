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

# Covertype

df_train = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/covertype/train.csv')
df_test = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/covertype/test.csv')
df_val = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/covertype/validation.csv') 

cont_columns = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
       'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',
       'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
       'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
       'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10',
       'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14',
       'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18',
       'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22',
       'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',
       'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
       'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
       'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38',
       'Soil_Type39', 'Soil_Type40']
cat_columns = []
target = ['Cover_Type']

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

attn_entropy_get(entropylog, model_cat, "CAT", 2, "Covertype", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 2, "Covertype", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

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

attn_entropy_get(entropylog, model_cat, "CAT", 4, "Covertype", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 4, "Covertype", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

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

attn_entropy_get(entropylog, model_cat, "CAT", 6, "Covertype", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 6, "Covertype", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

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

attn_entropy_get(entropylog, model_cat, "CAT", 7, "Covertype", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 7, "Covertype", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

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

attn_entropy_get(entropylog, model_cat, "CAT", 8, "Covertype", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 8, "Covertype", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

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

attn_entropy_get(entropylog, model_cat, "CAT", 9, "Covertype", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)
attn_entropy_get(entropylog, model_ft, "FT", 9, "Covertype", df_train, df_test, target[0], cat_columns, cont_columns, device_in_use)

data = entropylog.get_data()
print(data)


with open('/home/wdwatson2/projects/CAT-Transformer/interpretability/entropylog.pkl', 'wb') as file:
    pickle.dump(entropylog, file)