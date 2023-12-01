import sys
sys.path.insert(0, r'C:\Users\smbm2\projects\CAT-Transformer\model')
from model_embeddings import CATTransformer, Combined_Dataset, train, test
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
device_in_use = 'cuda'

##########################################################################################################################################################
# HELENA FIRST
df_train = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\helena\train.csv')
df_test = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\helena\test.csv')
df_val = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\helena\validation.csv') #READ FROM RIGHT SPOT

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
#------------------------------------------------------------------------------------------------------------------------------------------------
#Normal - Mixed - Trainable
print("HELENA\n", file=open("log.txt", 'a'))
print("Normal - Mixed - Trainable\n",file=open("log.txt", 'a'))
model = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=[], #pass cat_feat an emtpy list if dataset contains no cat features
                       targets_classes=target_classes,
                       trainable=True,
                       initialization="normal",
                       linear_on=True,
                       mixed_on=True
                       ).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 
test_f1_scores = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_acc = train(regression_on=False, 
                                   dataloader=train_dataloader, 
                                   model=model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc = test(regression_on=False,
                               dataloader=test_dataloader,
                               model=model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_accuracies_1.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies_1.append(test_acc)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
    test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}",file=open("log.txt", 'a'))

best_index = test_accuracies_1.index(max(test_accuracies_1))
print(f"Best accuracy {test_accuracies_1[best_index]}\n",file=open("log.txt", 'a'))

#------------------------------------------------------------------------------------------------------------------------------------------------
#Normal - Mixed - Not Trainable
print("Normal - Mixed - Not Trainable\n",file=open("log.txt", 'a'))
model = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=[], #pass cat_feat an emtpy list if dataset contains no cat features
                       targets_classes=target_classes,
                       trainable=False,
                       initialization="normal",
                       linear_on=True,
                       mixed_on=True
                       ).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 
test_f1_scores = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_acc = train(regression_on=False, 
                                   dataloader=train_dataloader, 
                                   model=model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc = test(regression_on=False,
                               dataloader=test_dataloader,
                               model=model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_accuracies_1.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies_1.append(test_acc)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
    test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}",file=open("log.txt", 'a'))

best_index = test_accuracies_1.index(max(test_accuracies_1))
print(f"Best accuracy {test_accuracies_1[best_index]}\n",file=open("log.txt", 'a'))


#------------------------------------------------------------------------------------------------------------------------------------------------
#Normal - Not Mixed - Trainable
print("Normal - Not Mixed - Trainable\n",file=open("log.txt", 'a'))
model = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=[], #pass cat_feat an emtpy list if dataset contains no cat features
                       targets_classes=target_classes,
                       trainable=True,
                       initialization="normal",
                       linear_on=True,
                       mixed_on=False
                       ).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 
test_f1_scores = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_acc = train(regression_on=False, 
                                   dataloader=train_dataloader, 
                                   model=model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc = test(regression_on=False,
                               dataloader=test_dataloader,
                               model=model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_accuracies_1.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies_1.append(test_acc)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
    test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}",file=open("log.txt", 'a'))

best_index = test_accuracies_1.index(max(test_accuracies_1))
print(f"Best accuracy {test_accuracies_1[best_index]}\n",file=open("log.txt", 'a'))

#-----------------------------------------------------------------------------------------------------------------------------------------------

#Normal - Not Mixed - Not Trainable
print("Normal - Not Mixed - Not Trainable\n",file=open("log.txt", 'a'))
model = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=[], #pass cat_feat an emtpy list if dataset contains no cat features
                       targets_classes=target_classes,
                       trainable=False,
                       initialization="normal",
                       linear_on=True,
                       mixed_on=False
                       ).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 
test_f1_scores = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_acc = train(regression_on=False, 
                                   dataloader=train_dataloader, 
                                   model=model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc = test(regression_on=False,
                               dataloader=test_dataloader,
                               model=model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_accuracies_1.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies_1.append(test_acc)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
    test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}",file=open("log.txt", 'a'))

best_index = test_accuracies_1.index(max(test_accuracies_1))
print(f"Best accuracy {test_accuracies_1[best_index]}\n",file=open("log.txt", 'a'))

#-----------------------------------------------------------------------------------------------------------------------------------------------

#Exp - Trainable
print("Exp - Trainable\n",file=open("log.txt", 'a'))
model = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=[], #pass cat_feat an emtpy list if dataset contains no cat features
                       targets_classes=target_classes,
                       trainable=True,
                       initialization="log-linear",
                       linear_on=True,
                       mixed_on=True
                       ).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 
test_f1_scores = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_acc = train(regression_on=False, 
                                   dataloader=train_dataloader, 
                                   model=model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc = test(regression_on=False,
                               dataloader=test_dataloader,
                               model=model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_accuracies_1.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies_1.append(test_acc)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
    test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}",file=open("log.txt", 'a'))

best_index = test_accuracies_1.index(max(test_accuracies_1))
print(f"Best accuracy {test_accuracies_1[best_index]}\n",file=open("log.txt", 'a'))


#-----------------------------------------------------------------------------------------------------------------------------------------------

#Exp - Not Trainable
print("Exp - Not Trainable\n",file=open("log.txt", 'a'))
model = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=[], #pass cat_feat an emtpy list if dataset contains no cat features
                       targets_classes=target_classes,
                       trainable=False,
                       initialization="log-linear",
                       linear_on=True,
                       mixed_on=True
                       ).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 
test_f1_scores = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_acc = train(regression_on=False, 
                                   dataloader=train_dataloader, 
                                   model=model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc = test(regression_on=False,
                               dataloader=test_dataloader,
                               model=model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_accuracies_1.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies_1.append(test_acc)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
    test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}",file=open("log.txt", 'a'))

best_index = test_accuracies_1.index(max(test_accuracies_1))
print(f"Best accuracy {test_accuracies_1[best_index]}\n",file=open("log.txt", 'a'))

####################################################################################################################################################################
#INCOME
# df_train = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/income/train.csv')
# df_test = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/income/test.csv')
# df_val = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/income/validation.csv') #READ FROM RIGHT SPOT

df_train = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\income\train.csv')
df_test = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\income\test.csv')
df_val = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\income\validation.csv') #READ FROM RIGHT SPOT

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
print("target classes", target_classes)
# Create a StandardScaler and fit it to the cont features
scaler = StandardScaler()
scaler.fit(df_train[cont_columns])

# Transform the training, test, and validation datasets
df_train[cont_columns] = scaler.transform(df_train[cont_columns])
df_test[cont_columns] = scaler.transform(df_test[cont_columns])
df_val[cont_columns] = scaler.transform(df_val[cont_columns])


#Wrapping in Dataset
train_dataset = Combined_Dataset(df_train, cat_columns, cont_columns, 'income')
val_dataset = Combined_Dataset(df_val, cat_columns, cont_columns, 'income')
test_dataset = Combined_Dataset(df_test, cat_columns, cont_columns, 'income')

#This is a hyperparameter that is not tuned. Maybe mess with what makes sense here
#Also try looking to see what other papers have done
batch_size = 256

# Wrapping with DataLoader for easy batch extraction
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#------------------------------------------------------------------------------------------------------------------------------------------------------------

#Normal - Mixed - Trainable
print("\nINCOME\n", file=open("log.txt", 'a'))
print("Normal - Mixed - Trainable\n",file=open("log.txt", 'a'))
model = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features, #pass cat_feat an emtpy list if dataset contains no cat features
                       targets_classes=target_classes,
                       trainable=True,
                       initialization="normal",
                       linear_on=True,
                       mixed_on=True
                       ).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 
test_f1_scores = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_acc = train(regression_on=False, 
                                   dataloader=train_dataloader, 
                                   model=model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc = test(regression_on=False,
                               dataloader=test_dataloader,
                               model=model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_accuracies_1.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies_1.append(test_acc)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
    test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}",file=open("log.txt", 'a'))

best_index = test_accuracies_1.index(max(test_accuracies_1))
print(f"Best accuracy {test_accuracies_1[best_index]}\n",file=open("log.txt", 'a'))

#------------------------------------------------------------------------------------------------------------------------------------------------
#Normal - Mixed - Not Trainable
print("Normal - Mixed - Not Trainable\n",file=open("log.txt", 'a'))
model = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features, #pass cat_feat an emtpy list if dataset contains no cat features
                       targets_classes=target_classes,
                       trainable=False,
                       initialization="normal",
                       linear_on=True,
                       mixed_on=True
                       ).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 
test_f1_scores = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_acc = train(regression_on=False, 
                                   dataloader=train_dataloader, 
                                   model=model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc = test(regression_on=False,
                               dataloader=test_dataloader,
                               model=model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_accuracies_1.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies_1.append(test_acc)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
    test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}",file=open("log.txt", 'a'))

best_index = test_accuracies_1.index(max(test_accuracies_1))
print(f"Best accuracy {test_accuracies_1[best_index]}\n",file=open("log.txt", 'a'))


#------------------------------------------------------------------------------------------------------------------------------------------------
#Normal - Not Mixed - Trainable
print("Normal - Not Mixed - Trainable\n",file=open("log.txt", 'a'))
model = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features, #pass cat_feat an emtpy list if dataset contains no cat features
                       targets_classes=target_classes,
                       trainable=True,
                       initialization="normal",
                       linear_on=True,
                       mixed_on=False
                       ).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 
test_f1_scores = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_acc = train(regression_on=False, 
                                   dataloader=train_dataloader, 
                                   model=model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc = test(regression_on=False,
                               dataloader=test_dataloader,
                               model=model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_accuracies_1.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies_1.append(test_acc)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
    test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}",file=open("log.txt", 'a'))

best_index = test_accuracies_1.index(max(test_accuracies_1))
print(f"Best accuracy {test_accuracies_1[best_index]}\n",file=open("log.txt", 'a'))

#-----------------------------------------------------------------------------------------------------------------------------------------------

#Normal - Not Mixed - Not Trainable
print("Normal - Not Mixed - Not Trainable\n",file=open("log.txt", 'a'))
model = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features, #pass cat_feat an emtpy list if dataset contains no cat features
                       targets_classes=target_classes,
                       trainable=False,
                       initialization="normal",
                       linear_on=True,
                       mixed_on=False
                       ).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 
test_f1_scores = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_acc = train(regression_on=False, 
                                   dataloader=train_dataloader, 
                                   model=model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc = test(regression_on=False,
                               dataloader=test_dataloader,
                               model=model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_accuracies_1.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies_1.append(test_acc)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
    test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}",file=open("log.txt", 'a'))

best_index = test_accuracies_1.index(max(test_accuracies_1))
print(f"Best accuracy {test_accuracies_1[best_index]}\n",file=open("log.txt", 'a'))

#-----------------------------------------------------------------------------------------------------------------------------------------------

#Exp - Trainable
print("Exp - Trainable\n",file=open("log.txt", 'a'))
model = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features, #pass cat_feat an emtpy list if dataset contains no cat features
                       targets_classes=target_classes,
                       trainable=True,
                       initialization="log-linear",
                       linear_on=True,
                       mixed_on=True
                       ).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 
test_f1_scores = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_acc = train(regression_on=False, 
                                   dataloader=train_dataloader, 
                                   model=model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc = test(regression_on=False,
                               dataloader=test_dataloader,
                               model=model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_accuracies_1.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies_1.append(test_acc)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
    test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}",file=open("log.txt", 'a'))

best_index = test_accuracies_1.index(max(test_accuracies_1))
print(f"Best accuracy {test_accuracies_1[best_index]}\n",file=open("log.txt", 'a'))


#-----------------------------------------------------------------------------------------------------------------------------------------------

#Exp - Not Trainable
print("Exp - Not Trainable\n",file=open("log.txt", 'a'))
model = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=cat_features, #pass cat_feat an emtpy list if dataset contains no cat features
                       targets_classes=target_classes,
                       trainable=False,
                       initialization="log-linear",
                       linear_on=True,
                       mixed_on=True
                       ).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 
test_f1_scores = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_acc = train(regression_on=False, 
                                   dataloader=train_dataloader, 
                                   model=model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc = test(regression_on=False,
                               dataloader=test_dataloader,
                               model=model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_accuracies_1.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies_1.append(test_acc)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
    test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}",file=open("log.txt", 'a'))

best_index = test_accuracies_1.index(max(test_accuracies_1))
print(f"Best accuracy {test_accuracies_1[best_index]}\n",file=open("log.txt", 'a'))

###########################################################################################################################################################
#CALIFORNIA_SKLEARN

df_train = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\california\train.csv')
df_test = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\california\test.csv')
df_val = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\california\validation.csv') #READ FROM RIGHT SPOT

cont_columns = [ 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
       'Latitude', 'Longitude']
target = ['MedInc']

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
train_dataset = Combined_Dataset(df_train, [], cont_columns, 'MedInc')
val_dataset = Combined_Dataset(df_val, [], cont_columns, 'MedInc')
test_dataset = Combined_Dataset(df_test, [], cont_columns, 'MedInc')

#This is a hyperparameter that is not tuned. Maybe mess with what makes sense here
#Also try looking to see what other papers have done
batch_size = 256

# Wrapping with DataLoader for easy batch extraction
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#----------------------------------------------------------------------------------------------------------------------------------------------

#Normal - Mixed - Trainable
print("\nCALIFORNIA_SKLEARN\n", file=open("log.txt", 'a'))
print("Normal - Mixed - Trainable\n",file=open("log.txt", 'a'))
model = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=[], #pass cat_feat an emtpy list if dataset contains no cat features
                       targets_classes=target_classes,
                       trainable=True,
                       initialization="normal",
                       linear_on=True,
                       mixed_on=True,
                       regression_on=True
                       ).to(device_in_use)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

train_losses = []
train_rmses = [] 
test_losses = []
test_rmses = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_rmse = train(regression_on=True, 
                                   dataloader=train_dataloader, 
                                   model=model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_rmse = test(regression_on=True,
                               dataloader=test_dataloader,
                               model=model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_rmses.append(train_rmse)
    test_losses.append(test_loss)
    test_rmses.append(test_rmse)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, RMSE {(train_rmse)}"
    test_metrics = f"Test: Loss {(test_loss)}, RMSE {(test_rmse)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}",file=open("log.txt", 'a'))

best_index = test_rmses.index(min(test_rmses))
print(f"Best RMSE {test_rmses[best_index]}\n",file=open("log.txt", 'a'))

#------------------------------------------------------------------------------------------------------------------------------------------------
#Normal - Mixed - Not Trainable
print("Normal - Mixed - Not Trainable\n",file=open("log.txt", 'a'))
model = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=[], #pass cat_feat an emtpy list if dataset contains no cat features
                       targets_classes=target_classes,
                       trainable=False,
                       initialization="normal",
                       linear_on=True,
                       mixed_on=True,
                       regression_on=True
                       ).to(device_in_use)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

train_losses = []
train_rmses = [] 
test_losses = []
test_rmses = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_rmse = train(regression_on=True, 
                                   dataloader=train_dataloader, 
                                   model=model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_rmse = test(regression_on=True,
                               dataloader=test_dataloader,
                               model=model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_rmses.append(train_rmse)
    test_losses.append(test_loss)
    test_rmses.append(test_rmse)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, RMSE {(train_rmse)}"
    test_metrics = f"Test: Loss {(test_loss)}, RMSE {(test_rmse)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}",file=open("log.txt", 'a'))

best_index = test_rmses.index(min(test_rmses))
print(f"Best RMSE {test_rmses[best_index]}\n",file=open("log.txt", 'a'))


#------------------------------------------------------------------------------------------------------------------------------------------------
#Normal - Not Mixed - Trainable
print("Normal - Not Mixed - Trainable\n",file=open("log.txt", 'a'))
model = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=[], #pass cat_feat an emtpy list if dataset contains no cat features
                       targets_classes=target_classes,
                       trainable=True,
                       initialization="normal",
                       linear_on=True,
                       mixed_on=False,
                       regression_on=True
                       ).to(device_in_use)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

train_losses = []
train_rmses = [] 
test_losses = []
test_rmses = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_rmse = train(regression_on=True, 
                                   dataloader=train_dataloader, 
                                   model=model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_rmse = test(regression_on=True,
                               dataloader=test_dataloader,
                               model=model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_rmses.append(train_rmse)
    test_losses.append(test_loss)
    test_rmses.append(test_rmse)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, RMSE {(train_rmse)}"
    test_metrics = f"Test: Loss {(test_loss)}, RMSE {(test_rmse)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}",file=open("log.txt", 'a'))

best_index = test_rmses.index(min(test_rmses))
print(f"Best RMSE {test_rmses[best_index]}\n",file=open("log.txt", 'a'))

#-----------------------------------------------------------------------------------------------------------------------------------------------

#Normal - Not Mixed - Not Trainable
print("Normal - Not Mixed - Not Trainable\n",file=open("log.txt", 'a'))
model = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=[], #pass cat_feat an emtpy list if dataset contains no cat features
                       targets_classes=target_classes,
                       trainable=False,
                       initialization="normal",
                       linear_on=True,
                       mixed_on=False,
                       regression_on=True
                       ).to(device_in_use)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

train_losses = []
train_rmses = [] 
test_losses = []
test_rmses = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_rmse = train(regression_on=True, 
                                   dataloader=train_dataloader, 
                                   model=model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_rmse = test(regression_on=True,
                               dataloader=test_dataloader,
                               model=model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_rmses.append(train_rmse)
    test_losses.append(test_loss)
    test_rmses.append(test_rmse)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, RMSE {(train_rmse)}"
    test_metrics = f"Test: Loss {(test_loss)}, RMSE {(test_rmse)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}",file=open("log.txt", 'a'))

best_index = test_rmses.index(min(test_rmses))
print(f"Best RMSE {test_rmses[best_index]}\n",file=open("log.txt", 'a'))

#-----------------------------------------------------------------------------------------------------------------------------------------------

#Exp - Trainable
print("Exp - Trainable\n",file=open("log.txt", 'a'))
model = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=[], #pass cat_feat an emtpy list if dataset contains no cat features
                       targets_classes=target_classes,
                       trainable=True,
                       initialization="log-linear",
                       linear_on=True,
                       mixed_on=True,
                       regression_on=True
                       ).to(device_in_use)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

train_losses = []
train_rmses = [] 
test_losses = []
test_rmses = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_rmse = train(regression_on=True, 
                                   dataloader=train_dataloader, 
                                   model=model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_rmse = test(regression_on=True,
                               dataloader=test_dataloader,
                               model=model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_rmses.append(train_rmse)
    test_losses.append(test_loss)
    test_rmses.append(test_rmse)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, RMSE {(train_rmse)}"
    test_metrics = f"Test: Loss {(test_loss)}, RMSE {(test_rmse)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}",file=open("log.txt", 'a'))

best_index = test_rmses.index(min(test_rmses))
print(f"Best RMSE {test_rmses[best_index]}\n",file=open("log.txt", 'a'))


#-----------------------------------------------------------------------------------------------------------------------------------------------

#Exp - Not Trainable
print("Exp - Not Trainable\n",file=open("log.txt", 'a'))
model = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=[], #pass cat_feat an emtpy list if dataset contains no cat features
                       targets_classes=target_classes,
                       trainable=False,
                       initialization="log-linear",
                       linear_on=True,
                       mixed_on=True,
                       regression_on=True
                       ).to(device_in_use)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

train_losses = []
train_rmses = [] 
test_losses = []
test_rmses = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_rmse = train(regression_on=True, 
                                   dataloader=train_dataloader, 
                                   model=model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_rmse = test(regression_on=True,
                               dataloader=test_dataloader,
                               model=model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_rmses.append(train_rmse)
    test_losses.append(test_loss)
    test_rmses.append(test_rmse)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, RMSE {(train_rmse)}"
    test_metrics = f"Test: Loss {(test_loss)}, RMSE {(test_rmse)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}",file=open("log.txt", 'a'))

best_index = test_rmses.index(min(test_rmses))
print(f"Best RMSE {test_rmses[best_index]}\n",file=open("log.txt", 'a'))

#########################################################################################################################################################
#HIGGS
# df_train = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/higgs/train.csv')
# df_test = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/higgs/test.csv')
# df_val = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/higgs/validation.csv') #READ FROM RIGHT SPOT

df_train = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\higgs\train.csv')
df_test = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\higgs\test.csv')
df_val = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\higgs\validation.csv') #READ FROM RIGHT SPOT

cont_columns = ['lepton_pT', 'lepton_eta', 'lepton_phi',
       'missing_energy_magnitude', 'missing_energy_phi', 'jet1pt', 'jet1eta',
       'jet1phi', 'jet1b-tag', 'jet2pt', 'jet2eta', 'jet2phi', 'jet2b-tag',
       'jet3pt', 'jet3eta', 'jet3phi', 'jet3b-tag', 'jet4pt', 'jet4eta',
       'jet4phi', 'jet4b-tag', 'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb',
       'm_wbb', 'm_wwbb']
target = ['class']

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
train_dataset = Combined_Dataset(df_train, [],  cont_columns, 'class')
val_dataset = Combined_Dataset(df_val,[], cont_columns, 'class')
test_dataset = Combined_Dataset(df_test,[], cont_columns, 'class')

#This is a hyperparameter that is not tuned. Maybe mess with what makes sense here
#Also try looking to see what other papers have done
batch_size = 256

# Wrapping with DataLoader for easy batch extraction
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#-------------------------------------------------------------------------------------------------------------------------------------------

#Normal - Mixed - Trainable
print("HIGGS\n", file=open("log.txt", 'a'))
print("Normal - Mixed - Trainable\n",file=open("log.txt", 'a'))
model = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=[], #pass cat_feat an emtpy list if dataset contains no cat features
                       targets_classes=target_classes,
                       trainable=True,
                       initialization="normal",
                       linear_on=True,
                       mixed_on=True
                       ).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 
test_f1_scores = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_acc = train(regression_on=False, 
                                   dataloader=train_dataloader, 
                                   model=model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc = test(regression_on=False,
                               dataloader=test_dataloader,
                               model=model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_accuracies_1.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies_1.append(test_acc)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
    test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}",file=open("log.txt", 'a'))

best_index = test_accuracies_1.index(max(test_accuracies_1))
print(f"Best accuracy {test_accuracies_1[best_index]}\n",file=open("log.txt", 'a'))

#------------------------------------------------------------------------------------------------------------------------------------------------
#Normal - Mixed - Not Trainable
print("Normal - Mixed - Not Trainable\n",file=open("log.txt", 'a'))
model = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=[], #pass cat_feat an emtpy list if dataset contains no cat features
                       targets_classes=target_classes,
                       trainable=False,
                       initialization="normal",
                       linear_on=True,
                       mixed_on=True
                       ).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 
test_f1_scores = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_acc = train(regression_on=False, 
                                   dataloader=train_dataloader, 
                                   model=model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc = test(regression_on=False,
                               dataloader=test_dataloader,
                               model=model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_accuracies_1.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies_1.append(test_acc)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
    test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}",file=open("log.txt", 'a'))

best_index = test_accuracies_1.index(max(test_accuracies_1))
print(f"Best accuracy {test_accuracies_1[best_index]}\n",file=open("log.txt", 'a'))


#------------------------------------------------------------------------------------------------------------------------------------------------
#Normal - Not Mixed - Trainable
print("Normal - Not Mixed - Trainable\n",file=open("log.txt", 'a'))
model = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=[], #pass cat_feat an emtpy list if dataset contains no cat features
                       targets_classes=target_classes,
                       trainable=True,
                       initialization="normal",
                       linear_on=True,
                       mixed_on=False
                       ).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 
test_f1_scores = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_acc = train(regression_on=False, 
                                   dataloader=train_dataloader, 
                                   model=model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc = test(regression_on=False,
                               dataloader=test_dataloader,
                               model=model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_accuracies_1.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies_1.append(test_acc)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
    test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}",file=open("log.txt", 'a'))

best_index = test_accuracies_1.index(max(test_accuracies_1))
print(f"Best accuracy {test_accuracies_1[best_index]}\n",file=open("log.txt", 'a'))

#-----------------------------------------------------------------------------------------------------------------------------------------------

#Normal - Not Mixed - Not Trainable
print("Normal - Not Mixed - Not Trainable\n",file=open("log.txt", 'a'))
model = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=[], #pass cat_feat an emtpy list if dataset contains no cat features
                       targets_classes=target_classes,
                       trainable=False,
                       initialization="normal",
                       linear_on=True,
                       mixed_on=False
                       ).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 
test_f1_scores = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_acc = train(regression_on=False, 
                                   dataloader=train_dataloader, 
                                   model=model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc = test(regression_on=False,
                               dataloader=test_dataloader,
                               model=model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_accuracies_1.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies_1.append(test_acc)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
    test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}",file=open("log.txt", 'a'))

best_index = test_accuracies_1.index(max(test_accuracies_1))
print(f"Best accuracy {test_accuracies_1[best_index]}\n",file=open("log.txt", 'a'))

#-----------------------------------------------------------------------------------------------------------------------------------------------

#Exp - Trainable
print("Exp - Trainable\n",file=open("log.txt", 'a'))
model = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=[], #pass cat_feat an emtpy list if dataset contains no cat features
                       targets_classes=target_classes,
                       trainable=True,
                       initialization="log-linear",
                       linear_on=True,
                       mixed_on=True
                       ).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 
test_f1_scores = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_acc = train(regression_on=False, 
                                   dataloader=train_dataloader, 
                                   model=model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc = test(regression_on=False,
                               dataloader=test_dataloader,
                               model=model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_accuracies_1.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies_1.append(test_acc)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
    test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}",file=open("log.txt", 'a'))

best_index = test_accuracies_1.index(max(test_accuracies_1))
print(f"Best accuracy {test_accuracies_1[best_index]}\n",file=open("log.txt", 'a'))


#-----------------------------------------------------------------------------------------------------------------------------------------------

#Exp - Not Trainable
print("Exp - Not Trainable\n",file=open("log.txt", 'a'))
model = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=[], #pass cat_feat an emtpy list if dataset contains no cat features
                       targets_classes=target_classes,
                       trainable=False,
                       initialization="log-linear",
                       linear_on=True,
                       mixed_on=True
                       ).to(device_in_use)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

train_losses = []
train_accuracies_1 = [] 
test_losses = []
test_accuracies_1 = [] 
test_f1_scores = [] 

epochs = 100

for t in range(epochs):
    train_loss, train_acc = train(regression_on=False, 
                                   dataloader=train_dataloader, 
                                   model=model, 
                                   loss_function=loss_function, 
                                   optimizer=optimizer, 
                                   device_in_use=device_in_use)
    test_loss, test_acc = test(regression_on=False,
                               dataloader=test_dataloader,
                               model=model,
                               loss_function=loss_function,
                               device_in_use=device_in_use)
    train_losses.append(train_loss)
    train_accuracies_1.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies_1.append(test_acc)

    epoch_str = f"Epoch [{t+1:2}/{epochs}]"
    train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
    test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
    print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}",file=open("log.txt", 'a'))

best_index = test_accuracies_1.index(max(test_accuracies_1))
print(f"Best accuracy {test_accuracies_1[best_index]}\n",file=open("log.txt", 'a'))