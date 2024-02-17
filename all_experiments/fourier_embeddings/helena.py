#RUN FIRST

import sys
sys.path.insert(0, '/home/cscadmin/CyberResearch/CAT-Transformer/model')
# sys.path.insert(0, r'C:\Users\smbm2\projects\CAT-Transformer\model')
# sys.path.insert(0, '/home/warin/projects/CAT-Transformer/model')
from testingModel import CATTransformer, Combined_Dataset, train, test, count_parameters
from testingModel import MyFTTransformer, EarlyStopping

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

device_in_use = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device_in_use)

#Get Helena

# df_train = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\helena\train.csv')
# df_test = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\helena\test.csv')
# df_val = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\helena\validation.csv') #READ FROM RIGHT SPOT

df_train = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/helena/train.csv')
df_test = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/helena/test.csv')
df_val = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/helena/validation.csv')

# df_train = pd.read_csv('/home/warin/projects/CAT-Transformer/datasets/helena/train.csv')
# df_test = pd.read_csv('/home/warin/projects/CAT-Transformer/datasets/helena/test.csv')
# df_val = pd.read_csv('/home/warin/projects/CAT-Transformer/datasets/helena/validation.csv')


# df_train.columns
cont_columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27']
cat_columns = []
target = ['class']

#CHECKING TO MAKE SURE YOUR LIST IS CORRECT (NO NEED TO TOUCH)
yourlist = cont_columns + target
yourlist.sort()
oglist = list(df_train.columns)
oglist.sort()

assert(yourlist == oglist), "You may of spelled feature name wrong or you forgot to put one of them in the list"

target_classes = [max(len(df_train[target].value_counts()), len(df_val[target].value_counts()),len(df_test[target].value_counts()))]
print("target classes",target_classes)

cat_features = ()

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

#This is a hyperparameter that is not tuned. Maybe mess with what makes sense here
batch_size = 256

# Wrapping with DataLoader for easy batch extraction
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

evaluation_log = EvaluationLog()

# Adding models, datasets, and metrics
models = ["CAT", "FT"]
embedding_techniques = ["ConstantPL", "PL", "ExpFF", "L"]
datasets = ["Helena"]
metrics = ["Train Loss", "Test Loss", "Train Acc", "Test Acc", "Test RMSE", "Train RMSE"]

for model in models:
    for embedding in embedding_techniques:
        for dataset in datasets:
            evaluation_log.add_dataset(model, embedding, dataset)


########################################################################################################################################################################
#Linear
#CAT
            

for trial_num in range(3):
    cat_model = CATTransformer(embedding='L',
                            n_cont=len(cont_columns),
                            cat_feat=cat_features,
                            targets_classes=target_classes).to(device_in_use)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=cat_model.parameters(), lr=0.0005)

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

    evaluation_log.add_metric("CAT", "L","Helena", trial_num,"Train Loss", train_losses)
    evaluation_log.add_metric("CAT", "L","Helena", trial_num,"Test Loss", test_losses)
    evaluation_log.add_metric("CAT", "L","Helena", trial_num,"Train Acc", train_accuracies_1)
    evaluation_log.add_metric("CAT", "L","Helena", trial_num,"Test Acc", test_accuracies_1)


    #FT
    ft_model = MyFTTransformer(embedding='L',
                                n_cont=len(cont_columns),
                        cat_feat=cat_features, 
                        targets_classes=target_classes
                        ).to(device_in_use)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=ft_model.parameters(), lr=0.0005)

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
                                    model=ft_model, 
                                    loss_function=loss_function, 
                                    optimizer=optimizer, 
                                    device_in_use=device_in_use)
        test_loss, test_acc, test_f1= test(regression_on=False,
                                get_attn=False,
                                dataloader=test_dataloader,
                                model=ft_model,
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

    evaluation_log.add_metric("FT", "L","Helena", trial_num,"Train Loss", train_losses)
    evaluation_log.add_metric("FT", "L","Helena", trial_num,"Test Loss", test_losses)
    evaluation_log.add_metric("FT", "L","Helena", trial_num,"Train Acc", train_accuracies_1)
    evaluation_log.add_metric("FT", "L","Helena", trial_num,"Test Acc", test_accuracies_1)


####################################################################################################################################################################################################################################

# PL
#CAT

for trial_num in range(3):
    cat_model = CATTransformer(embedding='PL',
                            n_cont=len(cont_columns),
                            cat_feat=cat_features,
                            targets_classes=target_classes).to(device_in_use)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=cat_model.parameters(), lr=0.0005)

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

    evaluation_log.add_metric("CAT", "PL","Helena", trial_num,"Train Loss", train_losses)
    evaluation_log.add_metric("CAT", "PL","Helena", trial_num,"Test Loss", test_losses)
    evaluation_log.add_metric("CAT", "PL","Helena", trial_num,"Train Acc", train_accuracies_1)
    evaluation_log.add_metric("CAT", "PL","Helena", trial_num,"Test Acc", test_accuracies_1)


    #FT
    ft_model = MyFTTransformer(embedding='PL',
                                n_cont=len(cont_columns),
                        cat_feat=cat_features, 
                        targets_classes=target_classes
                        ).to(device_in_use)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=ft_model.parameters(), lr=0.0005)

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
                                    model=ft_model, 
                                    loss_function=loss_function, 
                                    optimizer=optimizer, 
                                    device_in_use=device_in_use)
        test_loss, test_acc, test_f1= test(regression_on=False,
                                get_attn=False,
                                dataloader=test_dataloader,
                                model=ft_model,
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

    evaluation_log.add_metric("FT", "PL","Helena", trial_num,"Train Loss", train_losses)
    evaluation_log.add_metric("FT", "PL","Helena", trial_num,"Test Loss", test_losses)
    evaluation_log.add_metric("FT", "PL","Helena", trial_num,"Train Acc", train_accuracies_1)
    evaluation_log.add_metric("FT", "PL","Helena", trial_num,"Test Acc", test_accuracies_1)

####################################################################################################################################################################################################################################

# ConstantPL
#CAT

for trial_num in range(3):
    cat_model = CATTransformer(embedding='ConstantPL',
                            n_cont=len(cont_columns),
                            cat_feat=cat_features,
                            targets_classes=target_classes).to(device_in_use)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=cat_model.parameters(), lr=0.0005)

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

    evaluation_log.add_metric("CAT", "ConstantPL","Helena", trial_num,"Train Loss", train_losses)
    evaluation_log.add_metric("CAT", "ConstantPL","Helena", trial_num,"Test Loss", test_losses)
    evaluation_log.add_metric("CAT", "ConstantPL","Helena", trial_num,"Train Acc", train_accuracies_1)
    evaluation_log.add_metric("CAT", "ConstantPL","Helena", trial_num,"Test Acc", test_accuracies_1)


    #FT
    ft_model = MyFTTransformer(embedding='ConstantPL',
                                n_cont=len(cont_columns),
                        cat_feat=cat_features, 
                        targets_classes=target_classes
                        ).to(device_in_use)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=ft_model.parameters(), lr=0.0005)

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
                                    model=ft_model, 
                                    loss_function=loss_function, 
                                    optimizer=optimizer, 
                                    device_in_use=device_in_use)
        test_loss, test_acc, test_f1= test(regression_on=False,
                                get_attn=False,
                                dataloader=test_dataloader,
                                model=ft_model,
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

    evaluation_log.add_metric("FT", "ConstantPL","Helena", trial_num,"Train Loss", train_losses)
    evaluation_log.add_metric("FT", "ConstantPL","Helena", trial_num,"Test Loss", test_losses)
    evaluation_log.add_metric("FT", "ConstantPL","Helena", trial_num,"Train Acc", train_accuracies_1)
    evaluation_log.add_metric("FT", "ConstantPL","Helena", trial_num,"Test Acc", test_accuracies_1)

####################################################################################################################################################################################################################################

# EXP
#CAT

for trial_num in range(3):
    cat_model = CATTransformer(embedding='ExpFF',
                            n_cont=len(cont_columns),
                            cat_feat=cat_features,
                            targets_classes=target_classes).to(device_in_use)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=cat_model.parameters(), lr=0.0005)

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

    evaluation_log.add_metric("CAT", "ExpFF","Helena", trial_num,"Train Loss", train_losses)
    evaluation_log.add_metric("CAT", "ExpFF","Helena", trial_num,"Test Loss", test_losses)
    evaluation_log.add_metric("CAT", "ExpFF","Helena", trial_num,"Train Acc", train_accuracies_1)
    evaluation_log.add_metric("CAT", "ExpFF","Helena", trial_num,"Test Acc", test_accuracies_1)


    #FT
    ft_model = MyFTTransformer(embedding='ExpFF',
                                n_cont=len(cont_columns),
                        cat_feat=cat_features, 
                        targets_classes=target_classes
                        ).to(device_in_use)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=ft_model.parameters(), lr=0.0005)

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
                                    model=ft_model, 
                                    loss_function=loss_function, 
                                    optimizer=optimizer, 
                                    device_in_use=device_in_use)
        test_loss, test_acc, test_f1= test(regression_on=False,
                                get_attn=False,
                                dataloader=test_dataloader,
                                model=ft_model,
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

    evaluation_log.add_metric("FT", "ExpFF","Helena", trial_num,"Train Loss", train_losses)
    evaluation_log.add_metric("FT", "ExpFF","Helena", trial_num,"Test Loss", test_losses)
    evaluation_log.add_metric("FT", "ExpFF","Helena", trial_num,"Train Acc", train_accuracies_1)
    evaluation_log.add_metric("FT", "ExpFF","Helena", trial_num,"Test Acc", test_accuracies_1)

# with open(r'C:\Users\smbm2\projects\CAT-Transformer\cat_vs_ft\evaluation_log.pkl', 'wb') as file:
#     pickle.dump(evaluation_log, file)

with open('/home/cscadmin/CyberResearch/CAT-Transformer/embed_experiments/evaluation_log.pkl', 'wb') as file:
    pickle.dump(evaluation_log, file)









