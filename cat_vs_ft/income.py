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

#GET INCOME

df_train = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/income/train.csv')
df_test = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/income/test.csv')
df_val = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/income/validation.csv') #READ FROM RIGHT SPOT

# df_train = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\income\train.csv')
# df_test = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\income\test.csv')
# df_val = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\income\validation.csv') #READ FROM RIGHT SPOT

cont_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
       'hours-per-week']
cat_columns = ['workclass', 'education', 'marital-status', 'occupation',
       'relationship', 'race', 'sex', 'native-country']
target = ['income']

#CHECKING TO MAKE SURE YOUR LIST IS CORRECT (NO NEED TO TOUCH)
yourlist = cont_columns + cat_columns+target
yourlist.sort()
oglist = list(df_train.columns)
oglist.sort()

assert(yourlist == oglist), "You may of spelled feature name wrong or you forgot to put on of them in the list"

cat_features = (10,16,7,16,6,5,2,43)

target_classes = [max(len(df_train[target].value_counts()), len(df_val[target].value_counts()),len(df_test[target].value_counts()))]

# Create a StandardScaler and fit it to the cont features
scaler = StandardScaler()
scaler.fit(df_train[cont_columns])

# Transform the training, test, and validation datasets
df_train[cont_columns] = scaler.transform(df_train[cont_columns])
df_test[cont_columns] = scaler.transform(df_test[cont_columns])
df_val[cont_columns] = scaler.transform(df_val[cont_columns])

#Wrapping in Dataset
train_dataset = Combined_Dataset(df_train, cat_columns=cat_columns, num_columns=cont_columns, task1_column='income')
val_dataset = Combined_Dataset(df_val, cat_columns=cat_columns, num_columns=cont_columns, task1_column='income')
test_dataset = Combined_Dataset(df_test, cat_columns=cat_columns, num_columns=cont_columns, task1_column='income')

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

evaluation_log.add_new_dataset("Income")


########################################################################################################################################################################
#Linear
#CAT
cat_model = CATTransformer(embedding='Linear',
                           n_cont=len(cont_columns),
                       cat_feat=cat_columns, 
                       targets_classes=target_classes,
                       embed_size=160
                       ).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=cat_model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 1

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

evaluation_log.add_metric("CAT", "L","Income", "Train Loss", train_losses)
evaluation_log.add_metric("CAT", "L","Income", "Test Loss", test_losses)
evaluation_log.add_metric("CAT", "L","Income", "Train Acc", train_accuracies_1)
evaluation_log.add_metric("CAT", "L","Income", "Test Acc", test_accuracies_1)


#FT
ft_model = MyFTTransformer(embedding='Linear',
                            n_cont=len(cont_columns),
                       cat_feat=cat_columns, 
                       targets_classes=target_classes,
                       embed_size=160
                       ).to(device_in_use)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=ft_model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 1

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

evaluation_log.add_metric("FT", "L","Income", "Train Loss", train_losses)
evaluation_log.add_metric("FT", "L","Income", "Test Loss", test_losses)
evaluation_log.add_metric("FT", "L","Income", "Train Acc", train_accuracies_1)
evaluation_log.add_metric("FT", "L","Income", "Test Acc", test_accuracies_1)


####################################################################################################################################################################################################################################

# PL
#CAT
cat_model = CATTransformer(embedding='PL',
                           n_cont=len(cont_columns),
                       cat_feat=cat_columns, 
                       targets_classes=target_classes,
                       embed_size=160
                       ).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=cat_model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 1

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

evaluation_log.add_metric("CAT", "PL","Income", "Train Loss", train_losses)
evaluation_log.add_metric("CAT", "PL","Income", "Test Loss", test_losses)
evaluation_log.add_metric("CAT", "PL","Income", "Train Acc", train_accuracies_1)
evaluation_log.add_metric("CAT", "PL","Income", "Test Acc", test_accuracies_1)


#FT
ft_model = MyFTTransformer(embedding='PL',
                            n_cont=len(cont_columns),
                       cat_feat=cat_columns, 
                       targets_classes=target_classes,
                       embed_size=160
                       ).to(device_in_use)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=ft_model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 1

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

evaluation_log.add_metric("FT", "PL","Income", "Train Loss", train_losses)
evaluation_log.add_metric("FT", "PL","Income", "Test Loss", test_losses)
evaluation_log.add_metric("FT", "PL","Income", "Train Acc", train_accuracies_1)
evaluation_log.add_metric("FT", "PL","Income", "Test Acc", test_accuracies_1)

####################################################################################################################################################################################################################################

# ConstantPL
#CAT
cat_model = CATTransformer(embedding='ConstantPL',
                           n_cont=len(cont_columns),
                       cat_feat=cat_columns, 
                       targets_classes=target_classes,
                       embed_size=160
                       ).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=cat_model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 1

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

evaluation_log.add_metric("CAT", "ConstantPL","Income", "Train Loss", train_losses)
evaluation_log.add_metric("CAT", "ConstantPL","Income", "Test Loss", test_losses)
evaluation_log.add_metric("CAT", "ConstantPL","Income", "Train Acc", train_accuracies_1)
evaluation_log.add_metric("CAT", "ConstantPL","Income", "Test Acc", test_accuracies_1)


#FT
ft_model = MyFTTransformer(embedding='ConstantPL',
                            n_cont=len(cont_columns),
                       cat_feat=cat_columns, 
                       targets_classes=target_classes,
                       embed_size=160
                       ).to(device_in_use)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=ft_model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 1

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

evaluation_log.add_metric("FT", "ConstantPL","Income", "Train Loss", train_losses)
evaluation_log.add_metric("FT", "ConstantPL","Income", "Test Loss", test_losses)
evaluation_log.add_metric("FT", "ConstantPL","Income", "Train Acc", train_accuracies_1)
evaluation_log.add_metric("FT", "ConstantPL","Income", "Test Acc", test_accuracies_1)


####################################################################################################################################################################################################################################

# EXP
#CAT
cat_model = CATTransformer(embedding='Exp',
                           n_cont=len(cont_columns),
                       cat_feat=cat_columns, 
                       targets_classes=target_classes,
                       embed_size=160
                       ).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=cat_model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 1

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

evaluation_log.add_metric("CAT", "Exp","Income", "Train Loss", train_losses)
evaluation_log.add_metric("CAT", "Exp","Income", "Test Loss", test_losses)
evaluation_log.add_metric("CAT", "Exp","Income", "Train Acc", train_accuracies_1)
evaluation_log.add_metric("CAT", "Exp","Income", "Test Acc", test_accuracies_1)


#FT
ft_model = MyFTTransformer(embedding='Exp',
                            n_cont=len(cont_columns),
                       cat_feat=cat_columns, 
                       targets_classes=target_classes,
                       embed_size=160
                       ).to(device_in_use)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=ft_model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 

epochs = 1

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

evaluation_log.add_metric("FT", "Exp","Income", "Train Loss", train_losses)
evaluation_log.add_metric("FT", "Exp","Income", "Test Loss", test_losses)
evaluation_log.add_metric("FT", "Exp","Income", "Train Acc", train_accuracies_1)
evaluation_log.add_metric("FT", "Exp","Income", "Test Acc", test_accuracies_1)

# with open(r'C:\Users\smbm2\projects\CAT-Transformer\cat_vs_ft\evaluation_log.pkl', 'wb') as file:
#     pickle.dump(evaluation_log, file)

with open('/home/cscadmin/CyberResearch/CAT-Transformer/cat_vs_ft/evaluation_log.pkl', 'wb') as file:
    pickle.dump(evaluation_log, file)









