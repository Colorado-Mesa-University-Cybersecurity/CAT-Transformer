import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
from rtdl_revisiting_models import FTTransformer

from helpers import Log

import sys
sys.path.insert(0, '/home/wdwatson2/projects/CAT-Transformer/model')
from testingModel import CATTransformer, MyFTTransformer, Combined_Dataset, train, test, EarlyStopping, count_parameters
import for_rtdl

device_in_use = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device_in_use)

performance_log = Log()
performance_log.add_embedding_technique('Gaussian')
performance_log.add_embedding_technique('Log-Linear')

bases = [0.001, 0.1, 0.5, 1, 4]

##########################################################################################################################################################################################

performance_log.add_new_dataset('Income')

#income
df_train = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/income/train.csv')
df_test = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/income/test.csv')
df_val = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/income/validation.csv') 

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

assert(yourlist == oglist), "You may of spelled feature name wrong or you forgot to put one of them in the list"

cat_features = (10,16,7,16,6,5,2,43)

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


for base in bases:
    #CAT
    cat_model = CATTransformer(n_cont=len(cont_columns),
                        cat_feat=cat_features,
                        targets_classes=target_classes,
                        get_attn=False,
                        embedding='ConstantPL',
                        alpha=base
                        ).to(device_in_use)

    optimizer = torch.optim.Adam(params=cat_model.parameters(), lr=0.0005)
    loss_function = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(patience=10, verbose=True)

    train_losses = []
    train_accuracies_1 = [] 
    train_f1s = []
    test_losses = []
    test_accuracies_1 = [] 
    test_f1s = []

    epochs = 800

    for t in range(epochs):
        train_loss, train_acc, train_f1= train(regression_on=False, 
                                    get_attn=False,
                                    dataloader=train_dataloader, 
                                    model=cat_model, 
                                    loss_function=loss_function, 
                                    optimizer=optimizer, 
                                    device_in_use=device_in_use)
        test_loss, test_acc, test_f1= test(regression_on=False,
                                get_attn=False,
                                dataloader=test_dataloader,
                                model=cat_model,
                                loss_function=loss_function,
                                device_in_use=device_in_use)
        train_losses.append(train_loss)
        train_accuracies_1.append(train_acc)
        train_f1s.append(train_f1)
        test_losses.append(test_loss)
        test_accuracies_1.append(test_acc)
        test_f1s.append(test_f1)

        epoch_str = f"Epoch [{t+1:2}/{epochs}]"
        train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}, F1 {(train_f1)}"
        test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}, F1 {(test_f1)}"
        print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

        early_stopping(test_acc)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

    performance_log.add_metric('Gaussian', 'Income', base, 'Test Accuracy', test_accuracies_1)
    performance_log.add_metric('Gaussian', 'Income', base, 'Test F1', test_f1s)
    performance_log.add_metric('Gaussian', 'Income', base, 'Train Loss', train_losses)
    performance_log.add_metric('Gaussian', 'Income', base, 'Test Loss', test_losses)

    cat_model = CATTransformer(n_cont=len(cont_columns),
                        cat_feat=cat_features,
                        targets_classes=target_classes,
                        get_attn=False,
                        embedding='ExpFF',
                        alpha=base
                        ).to(device_in_use)

    optimizer = torch.optim.Adam(params=cat_model.parameters(), lr=0.0005)
    loss_function = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(patience=10, verbose=True)

    train_losses = []
    train_accuracies_1 = [] 
    train_f1s = []
    test_losses = []
    test_accuracies_1 = [] 
    test_f1s = []

    epochs = 800

    for t in range(epochs):
        train_loss, train_acc, train_f1= train(regression_on=False, 
                                    get_attn=False,
                                    dataloader=train_dataloader, 
                                    model=cat_model, 
                                    loss_function=loss_function, 
                                    optimizer=optimizer, 
                                    device_in_use=device_in_use)
        test_loss, test_acc, test_f1= test(regression_on=False,
                                get_attn=False,
                                dataloader=test_dataloader,
                                model=cat_model,
                                loss_function=loss_function,
                                device_in_use=device_in_use)
        train_losses.append(train_loss)
        train_accuracies_1.append(train_acc)
        train_f1s.append(train_f1)
        test_losses.append(test_loss)
        test_accuracies_1.append(test_acc)
        test_f1s.append(test_f1)

        epoch_str = f"Epoch [{t+1:2}/{epochs}]"
        train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}, F1 {(train_f1)}"
        test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}, F1 {(test_f1)}"
        print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

        early_stopping(test_acc)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

    performance_log.add_metric('Log-Linear', 'Income', base, 'Test Accuracy', test_accuracies_1)
    performance_log.add_metric('Log-Linear', 'Income', base, 'Test F1', test_f1s)
    performance_log.add_metric('Log-Linear', 'Income', base, 'Train Loss', train_losses)
    performance_log.add_metric('Log-Linear', 'Income', base, 'Test Loss', test_losses)

###############################################################################################################################################################################################
performance_log.add_new_dataset('Higgs')

# Higgs

df_train = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/higgs/train.csv')
df_test = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/higgs/test.csv')
df_val = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/higgs/validation.csv') 

cont_columns = ['lepton_pT', 'lepton_eta', 'lepton_phi',
       'missing_energy_magnitude', 'missing_energy_phi', 'jet1pt', 'jet1eta',
       'jet1phi', 'jet1b-tag', 'jet2pt', 'jet2eta', 'jet2phi', 'jet2b-tag',
       'jet3pt', 'jet3eta', 'jet3phi', 'jet3b-tag', 'jet4pt', 'jet4eta',
       'jet4phi', 'jet4b-tag', 'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb',
       'm_wbb', 'm_wwbb']
target = ['class']
cat_columns = []


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


for base in bases:
    #CAT
    cat_model = CATTransformer(n_cont=len(cont_columns),
                        cat_feat=cat_features,
                        targets_classes=target_classes,
                        get_attn=False,
                        embedding='ConstantPL',
                        alpha=base
                        ).to(device_in_use)

    optimizer = torch.optim.Adam(params=cat_model.parameters(), lr=0.0005)
    loss_function = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(patience=10, verbose=True)

    train_losses = []
    train_accuracies_1 = [] 
    train_f1s = []
    test_losses = []
    test_accuracies_1 = [] 
    test_f1s = []

    epochs = 800

    for t in range(epochs):
        train_loss, train_acc, train_f1= train(regression_on=False, 
                                    get_attn=False,
                                    dataloader=train_dataloader, 
                                    model=cat_model, 
                                    loss_function=loss_function, 
                                    optimizer=optimizer, 
                                    device_in_use=device_in_use)
        test_loss, test_acc, test_f1= test(regression_on=False,
                                get_attn=False,
                                dataloader=test_dataloader,
                                model=cat_model,
                                loss_function=loss_function,
                                device_in_use=device_in_use)
        train_losses.append(train_loss)
        train_accuracies_1.append(train_acc)
        train_f1s.append(train_f1)
        test_losses.append(test_loss)
        test_accuracies_1.append(test_acc)
        test_f1s.append(test_f1)

        epoch_str = f"Epoch [{t+1:2}/{epochs}]"
        train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}, F1 {(train_f1)}"
        test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}, F1 {(test_f1)}"
        print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

        early_stopping(test_acc)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

    performance_log.add_metric('Gaussian', 'Higgs', base, 'Test Accuracy', test_accuracies_1)
    performance_log.add_metric('Gaussian', 'Higgs', base, 'Test F1', test_f1s)
    performance_log.add_metric('Gaussian', 'Higgs', base, 'Train Loss', train_losses)
    performance_log.add_metric('Gaussian', 'Higgs', base, 'Test Loss', test_losses)

    cat_model = CATTransformer(n_cont=len(cont_columns),
                        cat_feat=cat_features,
                        targets_classes=target_classes,
                        get_attn=False,
                        embedding='ExpFF',
                        alpha=base
                        ).to(device_in_use)

    optimizer = torch.optim.Adam(params=cat_model.parameters(), lr=0.0005)
    loss_function = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(patience=10, verbose=True)

    train_losses = []
    train_accuracies_1 = [] 
    train_f1s = []
    test_losses = []
    test_accuracies_1 = [] 
    test_f1s = []

    epochs = 800

    for t in range(epochs):
        train_loss, train_acc, train_f1= train(regression_on=False, 
                                    get_attn=False,
                                    dataloader=train_dataloader, 
                                    model=cat_model, 
                                    loss_function=loss_function, 
                                    optimizer=optimizer, 
                                    device_in_use=device_in_use)
        test_loss, test_acc, test_f1= test(regression_on=False,
                                get_attn=False,
                                dataloader=test_dataloader,
                                model=cat_model,
                                loss_function=loss_function,
                                device_in_use=device_in_use)
        train_losses.append(train_loss)
        train_accuracies_1.append(train_acc)
        train_f1s.append(train_f1)
        test_losses.append(test_loss)
        test_accuracies_1.append(test_acc)
        test_f1s.append(test_f1)

        epoch_str = f"Epoch [{t+1:2}/{epochs}]"
        train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}, F1 {(train_f1)}"
        test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}, F1 {(test_f1)}"
        print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

        early_stopping(test_acc)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

    performance_log.add_metric('Log-Linear', 'Higgs', base, 'Test Accuracy', test_accuracies_1)
    performance_log.add_metric('Log-Linear', 'Higgs', base, 'Test F1', test_f1s)
    performance_log.add_metric('Log-Linear', 'Higgs', base, 'Train Loss', train_losses)
    performance_log.add_metric('Log-Linear', 'Higgs', base, 'Test Loss', test_losses)

##################################################################################################################################################################################################

performance_log.add_new_dataset('Helena')
#Get Helena

# df_train = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\helena\train.csv')
# df_test = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\helena\test.csv')
# df_val = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\helena\validation.csv') #READ FROM RIGHT SPOT

# df_train = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/helena/train.csv')
# df_test = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/helena/test.csv')
# df_val = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/helena/validation.csv')

df_train = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/helena/train.csv')
df_test = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/helena/test.csv')
df_val = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/helena/validation.csv')


# df_train.columns
cont_columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27']
target = ['class']
cat_columns = []

#CHECKING TO MAKE SURE YOUR LIST IS CORRECT (NO NEED TO TOUCH)
yourlist = cont_columns + target
yourlist.sort()
oglist = list(df_train.columns)
oglist.sort()

cat_features = ()

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

for base in bases:
    #CAT
    cat_model = CATTransformer(n_cont=len(cont_columns),
                        cat_feat=cat_features,
                        targets_classes=target_classes,
                        get_attn=False,
                        embedding='ConstantPL',
                        alpha=base
                        ).to(device_in_use)

    optimizer = torch.optim.Adam(params=cat_model.parameters(), lr=0.0005)
    loss_function = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(patience=10, verbose=True)

    train_losses = []
    train_accuracies_1 = [] 
    train_f1s = []
    test_losses = []
    test_accuracies_1 = [] 
    test_f1s = []

    epochs = 800

    for t in range(epochs):
        train_loss, train_acc, train_f1= train(regression_on=False, 
                                    get_attn=False,
                                    dataloader=train_dataloader, 
                                    model=cat_model, 
                                    loss_function=loss_function, 
                                    optimizer=optimizer, 
                                    device_in_use=device_in_use)
        test_loss, test_acc, test_f1= test(regression_on=False,
                                get_attn=False,
                                dataloader=test_dataloader,
                                model=cat_model,
                                loss_function=loss_function,
                                device_in_use=device_in_use)
        train_losses.append(train_loss)
        train_accuracies_1.append(train_acc)
        train_f1s.append(train_f1)
        test_losses.append(test_loss)
        test_accuracies_1.append(test_acc)
        test_f1s.append(test_f1)

        epoch_str = f"Epoch [{t+1:2}/{epochs}]"
        train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}, F1 {(train_f1)}"
        test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}, F1 {(test_f1)}"
        print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

        early_stopping(test_acc)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

    performance_log.add_metric('Gaussian', 'Helena', base, 'Test Accuracy', test_accuracies_1)
    performance_log.add_metric('Gaussian', 'Helena', base, 'Test F1', test_f1s)
    performance_log.add_metric('Gaussian', 'Helena', base, 'Train Loss', train_losses)
    performance_log.add_metric('Gaussian', 'Helena', base, 'Test Loss', test_losses)

    cat_model = CATTransformer(n_cont=len(cont_columns),
                        cat_feat=cat_features,
                        targets_classes=target_classes,
                        get_attn=False,
                        embedding='ExpFF',
                        alpha=base
                        ).to(device_in_use)

    optimizer = torch.optim.Adam(params=cat_model.parameters(), lr=0.0005)
    loss_function = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(patience=10, verbose=True)

    train_losses = []
    train_accuracies_1 = [] 
    train_f1s = []
    test_losses = []
    test_accuracies_1 = [] 
    test_f1s = []

    epochs = 800

    for t in range(epochs):
        train_loss, train_acc, train_f1= train(regression_on=False, 
                                    get_attn=False,
                                    dataloader=train_dataloader, 
                                    model=cat_model, 
                                    loss_function=loss_function, 
                                    optimizer=optimizer, 
                                    device_in_use=device_in_use)
        test_loss, test_acc, test_f1= test(regression_on=False,
                                get_attn=False,
                                dataloader=test_dataloader,
                                model=cat_model,
                                loss_function=loss_function,
                                device_in_use=device_in_use)
        train_losses.append(train_loss)
        train_accuracies_1.append(train_acc)
        train_f1s.append(train_f1)
        test_losses.append(test_loss)
        test_accuracies_1.append(test_acc)
        test_f1s.append(test_f1)

        epoch_str = f"Epoch [{t+1:2}/{epochs}]"
        train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}, F1 {(train_f1)}"
        test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}, F1 {(test_f1)}"
        print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

        early_stopping(test_acc)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

    performance_log.add_metric('Log-Linear', 'Helena', base, 'Test Accuracy', test_accuracies_1)
    performance_log.add_metric('Log-Linear', 'Helena', base, 'Test F1', test_f1s)
    performance_log.add_metric('Log-Linear', 'Helena', base, 'Train Loss', train_losses)
    performance_log.add_metric('Log-Linear', 'Helena', base, 'Test Loss', test_losses)


with open('/home/wdwatson2/projects/CAT-Transformer/embed_experiments/gaussian_determine/performance_log.pkl', 'wb') as file:
    pickle.dump(performance_log, file)



