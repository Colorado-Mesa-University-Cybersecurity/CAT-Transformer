import sys
sys.path.insert(0, '/home/cscadmin/CyberResearch/CAT-Transformer/model')
# sys.path.insert(0, r'C:\Users\smbm2\projects\CAT-Transformer\model')
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

#GET California

df_train = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/california/train.csv')
df_test = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/california/test.csv')
df_val = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/california/validation.csv') #READ FROM RIGHT SPOT

# df_train = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\california\train.csv')
# df_test = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\california\test.csv')
# df_val = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\california\validation.csv') #READ FROM RIGHT SPOT

# df_train = pd.read_csv('/home/warin/projects/CAT-Transformer/datasets/california_sklearn/train.csv')
# df_test = pd.read_csv('/home/warin/projects/CAT-Transformer/datasets/california_sklearn/test.csv')
# df_val = pd.read_csv('/home/warin/projects/CAT-Transformer/datasets/california_sklearn/validation.csv')


cont_columns = [ 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
       'Latitude', 'Longitude']
target = ['MedInc']

#CHECKING TO MAKE SURE YOUR LIST IS CORRECT (NO NEED TO TOUCH)
yourlist = cont_columns + target
yourlist.sort()
oglist = list(df_train.columns)
oglist.sort()

assert(yourlist == oglist), "You may of spelled feature name wrong or you forgot to put on of them in the list"

target_classes = [min(len(df_train[target].value_counts()), len(df_val[target].value_counts()),len(df_test[target].value_counts()))]
print(target_classes)
# Create a StandardScaler and fit it to the cont features
scaler = StandardScaler()
scaler.fit(df_train[cont_columns])

# Transform the training, test, and validation datasets
df_train[cont_columns] = scaler.transform(df_train[cont_columns])
df_test[cont_columns] = scaler.transform(df_test[cont_columns])
df_val[cont_columns] = scaler.transform(df_val[cont_columns])

#Wrapping in Dataset
train_dataset = Combined_Dataset(df_train, cat_columns=[], num_columns=cont_columns, task1_column='MedInc')
val_dataset = Combined_Dataset(df_val, cat_columns=[], num_columns=cont_columns, task1_column='MedInc')
test_dataset = Combined_Dataset(df_test, cat_columns=[], num_columns=cont_columns, task1_column='MedInc')

#This is a hyperparameter that is not tuned. Maybe mess with what makes sense here
batch_size = 256

# Wrapping with DataLoader for easy batch extraction
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Load the object
with open('/home/cscadmin/CyberResearch/CAT-Transformer/cat_vs_ft/evaluation_log.pkl', 'rb') as file:
    evaluation_log = pickle.load(file)

# with open(r'C:\Users\smbm2\projects\CAT-Transformer\cat_vs_ft\evaluation_log.pkl', 'rb') as file:
#     evaluation_log = pickle.load(file)

# Adding models, datasets, and metrics
models = ["CAT", "FT"]
embedding_techniques = ["ConstantPL", "PL", "Exp", "L"]
datasets = ["Helena", "Covertype", "Higgs", "Income"]
metrics = ["Train Loss", "Test Loss", "Train Acc", "Test Acc"]

evaluation_log.add_new_dataset("California")


########################################################################################################################################################################
#Linear
#CAT
cat_model = CATTransformer(embedding='Linear',
                           n_cont=len(cont_columns),
                       cat_feat=[], 
                       targets_classes=target_classes,
                       embed_size=160,
                       regression_on=True
                       ).to(device_in_use)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(params=cat_model.parameters(), lr=0.0001)

train_losses = []
train_rmse_1 = [] 
test_losses = []
test_rmse_1 = [] 

epochs = 150

for t in range(epochs):
    train_loss, train_rmse = train(regression_on=True, 
                                   dataloader=train_dataloader, 
                                   model=cat_model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_rmse = test(regression_on=True,
                               dataloader=test_dataloader,
                               model=cat_model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_rmse_1.append(train_rmse)
    test_losses.append(test_loss)
    test_rmse_1.append(test_rmse)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, RMSE {(train_rmse)}"
    test_metrics = f"Test: Loss {(test_loss)}, RMSE {(test_rmse)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

best_index = test_rmse_1.index(min(test_rmse_1))
print(f"Best RMSE {test_rmse_1[best_index]}")

evaluation_log.add_metric("CAT", "L","California", "Train Loss", train_losses)
evaluation_log.add_metric("CAT", "L","California", "Test Loss", test_losses)
evaluation_log.add_metric("CAT", "L","California", "Train Acc", train_rmse_1)
evaluation_log.add_metric("CAT", "L","California", "Test Acc", test_rmse_1)


#FT
ft_model = MyFTTransformer(embedding='Linear',
                            n_cont=len(cont_columns),
                       cat_feat=[], 
                       targets_classes=target_classes,
                       embed_size=160,
                       regression_on=True
                       ).to(device_in_use)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(params=ft_model.parameters(), lr=0.0001)

train_losses = []
train_rmse_1 = [] 
test_losses = []
test_rmse_1 = [] 

epochs = 150

for t in range(epochs):
    train_loss, train_rmse = train(regression_on=True, 
                                   dataloader=train_dataloader, 
                                   model=ft_model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_rmse = test(regression_on=True,
                               dataloader=test_dataloader,
                               model=ft_model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_rmse_1.append(train_rmse)
    test_losses.append(test_loss)
    test_rmse_1.append(test_rmse)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, RMSE {(train_rmse)}"
    test_metrics = f"Test: Loss {(test_loss)}, RMSE {(test_rmse)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

best_index = test_rmse_1.index(min(test_rmse_1))
print(f"Best RMSE {test_rmse_1[best_index]}")

evaluation_log.add_metric("FT", "L","California", "Train Loss", train_losses)
evaluation_log.add_metric("FT", "L","California", "Test Loss", test_losses)
evaluation_log.add_metric("FT", "L","California", "Train Acc", train_rmse_1)
evaluation_log.add_metric("FT", "L","California", "Test Acc", test_rmse_1)


####################################################################################################################################################################################################################################

# PL
#CAT
cat_model = CATTransformer(embedding='PL',
                           n_cont=len(cont_columns),
                       cat_feat=[], 
                       targets_classes=target_classes,
                       embed_size=160,
                       regression_on=True
                       ).to(device_in_use)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(params=cat_model.parameters(), lr=0.0001)

train_losses = []
train_rmse_1 = [] 
test_losses = []
test_rmse_1 = [] 

epochs = 150

for t in range(epochs):
    train_loss, train_rmse = train(regression_on=True, 
                                   dataloader=train_dataloader, 
                                   model=cat_model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_rmse = test(regression_on=True,
                               dataloader=test_dataloader,
                               model=cat_model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_rmse_1.append(train_rmse)
    test_losses.append(test_loss)
    test_rmse_1.append(test_rmse)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, RMSE {(train_rmse)}"
    test_metrics = f"Test: Loss {(test_loss)}, RMSE {(test_rmse)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

best_index = test_rmse_1.index(min(test_rmse_1))
print(f"Best RMSE {test_rmse_1[best_index]}")

evaluation_log.add_metric("CAT", "PL","California", "Train Loss", train_losses)
evaluation_log.add_metric("CAT", "PL","California", "Test Loss", test_losses)
evaluation_log.add_metric("CAT", "PL","California", "Train Acc", train_rmse_1)
evaluation_log.add_metric("CAT", "PL","California", "Test Acc", test_rmse_1)


#FT
ft_model = MyFTTransformer(embedding='PL',
                            n_cont=len(cont_columns),
                       cat_feat=[], 
                       targets_classes=target_classes,
                       embed_size=160,
                       regression_on=True
                       ).to(device_in_use)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(params=ft_model.parameters(), lr=0.0001)

train_losses = []
train_rmse_1 = [] 
test_losses = []
test_rmse_1 = [] 

epochs = 150

for t in range(epochs):
    train_loss, train_rmse = train(regression_on=True, 
                                   dataloader=train_dataloader, 
                                   model=ft_model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_rmse = test(regression_on=True,
                               dataloader=test_dataloader,
                               model=ft_model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_rmse_1.append(train_rmse)
    test_losses.append(test_loss)
    test_rmse_1.append(test_rmse)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, RMSE {(train_rmse)}"
    test_metrics = f"Test: Loss {(test_loss)}, RMSE {(test_rmse)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

best_index = test_rmse_1.index(min(test_rmse_1))
print(f"Best RMSE {test_rmse_1[best_index]}")

evaluation_log.add_metric("FT", "PL","California", "Train Loss", train_losses)
evaluation_log.add_metric("FT", "PL","California", "Test Loss", test_losses)
evaluation_log.add_metric("FT", "PL","California", "Train Acc", train_rmse_1)
evaluation_log.add_metric("FT", "PL","California", "Test Acc", test_rmse_1)

####################################################################################################################################################################################################################################

# ConstantPL
#CAT
cat_model = CATTransformer(embedding='ConstantPL',
                           n_cont=len(cont_columns),
                       cat_feat=[], 
                       targets_classes=target_classes,
                       embed_size=160,
                       regression_on=True
                       ).to(device_in_use)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(params=cat_model.parameters(), lr=0.0001)

train_losses = []
train_rmse_1 = [] 
test_losses = []
test_rmse_1 = [] 

epochs = 150

for t in range(epochs):
    train_loss, train_rmse = train(regression_on=True, 
                                   dataloader=train_dataloader, 
                                   model=cat_model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_rmse = test(regression_on=True,
                               dataloader=test_dataloader,
                               model=cat_model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_rmse_1.append(train_rmse)
    test_losses.append(test_loss)
    test_rmse_1.append(test_rmse)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, RMSE {(train_rmse)}"
    test_metrics = f"Test: Loss {(test_loss)}, RMSE {(test_rmse)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

best_index = test_rmse_1.index(min(test_rmse_1))
print(f"Best RMSE {test_rmse_1[best_index]}")

evaluation_log.add_metric("CAT", "ConstantPL","California", "Train Loss", train_losses)
evaluation_log.add_metric("CAT", "ConstantPL","California", "Test Loss", test_losses)
evaluation_log.add_metric("CAT", "ConstantPL","California", "Train Acc", train_rmse_1)
evaluation_log.add_metric("CAT", "ConstantPL","California", "Test Acc", test_rmse_1)


#FT
ft_model = MyFTTransformer(embedding='ConstantPL',
                            n_cont=len(cont_columns),
                       cat_feat=[], 
                       targets_classes=target_classes,
                       embed_size=160,
                       regression_on=True
                       ).to(device_in_use)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(params=ft_model.parameters(), lr=0.0001)

train_losses = []
train_rmse_1 = [] 
test_losses = []
test_rmse_1 = [] 

epochs = 150

for t in range(epochs):
    train_loss, train_rmse = train(regression_on=True, 
                                   dataloader=train_dataloader, 
                                   model=ft_model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_rmse = test(regression_on=True,
                               dataloader=test_dataloader,
                               model=ft_model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_rmse_1.append(train_rmse)
    test_losses.append(test_loss)
    test_rmse_1.append(test_rmse)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, RMSE {(train_rmse)}"
    test_metrics = f"Test: Loss {(test_loss)}, RMSE {(test_rmse)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

best_index = test_rmse_1.index(min(test_rmse_1))
print(f"Best RMSE {test_rmse_1[best_index]}")

evaluation_log.add_metric("FT", "ConstantPL","California", "Train Loss", train_losses)
evaluation_log.add_metric("FT", "ConstantPL","California", "Test Loss", test_losses)
evaluation_log.add_metric("FT", "ConstantPL","California", "Train Acc", train_rmse_1)
evaluation_log.add_metric("FT", "ConstantPL","California", "Test Acc", test_rmse_1)


####################################################################################################################################################################################################################################

# EXP
#CAT
cat_model = CATTransformer(embedding='Exp',
                           n_cont=len(cont_columns),
                       cat_feat=[], 
                       targets_classes=target_classes,
                       embed_size=160,
                       regression_on=True
                       ).to(device_in_use)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(params=cat_model.parameters(), lr=0.0001)

train_losses = []
train_rmse_1 = [] 
test_losses = []
test_rmse_1 = [] 

epochs = 150

for t in range(epochs):
    train_loss, train_rmse = train(regression_on=True, 
                                   dataloader=train_dataloader, 
                                   model=cat_model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_rmse = test(regression_on=True,
                               dataloader=test_dataloader,
                               model=cat_model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_rmse_1.append(train_rmse)
    test_losses.append(test_loss)
    test_rmse_1.append(test_rmse)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, RMSE {(train_rmse)}"
    test_metrics = f"Test: Loss {(test_loss)}, RMSE {(test_rmse)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

best_index = test_rmse_1.index(min(test_rmse_1))
print(f"Best RMSE {test_rmse_1[best_index]}")

evaluation_log.add_metric("CAT", "Exp","California", "Train Loss", train_losses)
evaluation_log.add_metric("CAT", "Exp","California", "Test Loss", test_losses)
evaluation_log.add_metric("CAT", "Exp","California", "Train Acc", train_rmse_1)
evaluation_log.add_metric("CAT", "Exp","California", "Test Acc", test_rmse_1)


#FT
ft_model = MyFTTransformer(embedding='Exp',
                            n_cont=len(cont_columns),
                       cat_feat=[], 
                       targets_classes=target_classes,
                       embed_size=160,
                       regression_on=True
                       ).to(device_in_use)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(params=ft_model.parameters(), lr=0.0001)

train_losses = []
train_rmse_1 = [] 
test_losses = []
test_rmse_1 = [] 

epochs = 150

for t in range(epochs):
    train_loss, train_rmse = train(regression_on=True, 
                                   dataloader=train_dataloader, 
                                   model=ft_model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_rmse = test(regression_on=True,
                               dataloader=test_dataloader,
                               model=ft_model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_rmse_1.append(train_rmse)
    test_losses.append(test_loss)
    test_rmse_1.append(test_rmse)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, RMSE {(train_rmse)}"
    test_metrics = f"Test: Loss {(test_loss)}, RMSE {(test_rmse)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

best_index = test_rmse_1.index(min(test_rmse_1))
print(f"Best RMSE {test_rmse_1[best_index]}")

evaluation_log.add_metric("FT", "Exp","California", "Train Loss", train_losses)
evaluation_log.add_metric("FT", "Exp","California", "Test Loss", test_losses)
evaluation_log.add_metric("FT", "Exp","California", "Train Acc", train_rmse_1)
evaluation_log.add_metric("FT", "Exp","California", "Test Acc", test_rmse_1)

# with open(r'C:\Users\smbm2\projects\CAT-Transformer\cat_vs_ft\evaluation_log.pkl', 'wb') as file:
#     pickle.dump(evaluation_log, file)

with open('/home/cscadmin/CyberResearch/CAT-Transformer/cat_vs_ft/evaluation_log.pkl', 'wb') as file:
    pickle.dump(evaluation_log, file)








