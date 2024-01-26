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

#GET COVERTYPE

df_train = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/covertype/train.csv')
df_test = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/covertype/test.csv')
df_val = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/covertype/validation.csv') #READ FROM RIGHT SPOT

# df_train = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\covertype\train.csv')
# df_test = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\covertype\test.csv')
# df_val = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\covertype\validation.csv') #READ FROM RIGHT

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
target = ['Cover_Type']
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
train_dataset = Combined_Dataset(df_train, cat_columns=[], num_columns=cont_columns, task1_column='Cover_Type')
val_dataset = Combined_Dataset(df_val, cat_columns=[], num_columns=cont_columns, task1_column='Cover_Type')
test_dataset = Combined_Dataset(df_test, cat_columns=[], num_columns=cont_columns, task1_column='Cover_Type')

#This is a hyperparameter that is not tuned. Maybe mess with what makes sense here
batch_size = 256

# Wrapping with DataLoader for easy batch extraction
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Load the object
with open('/home/cscadmin/CyberResearch/CAT-Transformer/embed_experiments/evaluation_log.pkl', 'rb') as file:
    evaluation_log = pickle.load(file)

# Adding models, datasets, and metrics
models = ["CAT", "FT"]
embedding_techniques = ["ConstantPL", "PL", "Exp", "L"]
datasets = ["Helena", "Covertype"]
metrics = ["Train Loss", "Test Loss", "Train Acc", "Test Acc"]

evaluation_log.add_new_dataset("Covertype")


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

    evaluation_log.add_metric("CAT", "L","Covertype", trial_num,"Train Loss", train_losses)
    evaluation_log.add_metric("CAT", "L","Covertype", trial_num,"Test Loss", test_losses)
    evaluation_log.add_metric("CAT", "L","Covertype", trial_num,"Train Acc", train_accuracies_1)
    evaluation_log.add_metric("CAT", "L","Covertype", trial_num,"Test Acc", test_accuracies_1)


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

    evaluation_log.add_metric("FT", "L","Covertype", trial_num,"Train Loss", train_losses)
    evaluation_log.add_metric("FT", "L","Covertype", trial_num,"Test Loss", test_losses)
    evaluation_log.add_metric("FT", "L","Covertype", trial_num,"Train Acc", train_accuracies_1)
    evaluation_log.add_metric("FT", "L","Covertype", trial_num,"Test Acc", test_accuracies_1)


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

    evaluation_log.add_metric("CAT", "PL","Covertype", trial_num,"Train Loss", train_losses)
    evaluation_log.add_metric("CAT", "PL","Covertype", trial_num,"Test Loss", test_losses)
    evaluation_log.add_metric("CAT", "PL","Covertype", trial_num,"Train Acc", train_accuracies_1)
    evaluation_log.add_metric("CAT", "PL","Covertype", trial_num,"Test Acc", test_accuracies_1)


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

    evaluation_log.add_metric("FT", "PL","Covertype", trial_num,"Train Loss", train_losses)
    evaluation_log.add_metric("FT", "PL","Covertype", trial_num,"Test Loss", test_losses)
    evaluation_log.add_metric("FT", "PL","Covertype", trial_num,"Train Acc", train_accuracies_1)
    evaluation_log.add_metric("FT", "PL","Covertype", trial_num,"Test Acc", test_accuracies_1)

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

    evaluation_log.add_metric("CAT", "ConstantPL","Covertype", trial_num,"Train Loss", train_losses)
    evaluation_log.add_metric("CAT", "ConstantPL","Covertype", trial_num,"Test Loss", test_losses)
    evaluation_log.add_metric("CAT", "ConstantPL","Covertype", trial_num,"Train Acc", train_accuracies_1)
    evaluation_log.add_metric("CAT", "ConstantPL","Covertype", trial_num,"Test Acc", test_accuracies_1)


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

    evaluation_log.add_metric("FT", "ConstantPL","Covertype", trial_num,"Train Loss", train_losses)
    evaluation_log.add_metric("FT", "ConstantPL","Covertype", trial_num,"Test Loss", test_losses)
    evaluation_log.add_metric("FT", "ConstantPL","Covertype", trial_num,"Train Acc", train_accuracies_1)
    evaluation_log.add_metric("FT", "ConstantPL","Covertype", trial_num,"Test Acc", test_accuracies_1)

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

    evaluation_log.add_metric("CAT", "ExpFF","Covertype", trial_num,"Train Loss", train_losses)
    evaluation_log.add_metric("CAT", "ExpFF","Covertype", trial_num,"Test Loss", test_losses)
    evaluation_log.add_metric("CAT", "ExpFF","Covertype", trial_num,"Train Acc", train_accuracies_1)
    evaluation_log.add_metric("CAT", "ExpFF","Covertype", trial_num,"Test Acc", test_accuracies_1)


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

    evaluation_log.add_metric("FT", "ExpFF","Covertype", trial_num,"Train Loss", train_losses)
    evaluation_log.add_metric("FT", "ExpFF","Covertype", trial_num,"Test Loss", test_losses)
    evaluation_log.add_metric("FT", "ExpFF","Covertype", trial_num,"Train Acc", train_accuracies_1)
    evaluation_log.add_metric("FT", "ExpFF","Covertype", trial_num,"Test Acc", test_accuracies_1)

# with open(r'C:\Users\smbm2\projects\CAT-Transformer\cat_vs_ft\evaluation_log.pkl', 'wb') as file:
#     pickle.dump(evaluation_log, file)

with open('/home/cscadmin/CyberResearch/CAT-Transformer/embed_experiments/evaluation_log.pkl', 'wb') as file:
    pickle.dump(evaluation_log, file)









