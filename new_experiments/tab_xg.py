import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score

from helpers import ModelPerformanceLog

import sys
sys.path.insert(0, '/home/wdwatson2/projects/CAT-Transformer/model')
from testingModel import Combined_Dataset, train, test, EarlyStopping, count_parameters, rmse
from tab_transformer_pytorch import TabTransformer

device_in_use = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device_in_use)

# with open('/home/wdwatson2/projects/CAT-Transformer/new_experiments/performance_log.pkl', 'rb') as file:
#     performance_log = pickle.load(file)

# ##########################################################################################################################################################################################

# #income
# df_train = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/income/train.csv')
# df_test = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/income/test.csv')
# df_val = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/income/validation.csv') 

# cont_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
#        'hours-per-week']
# cat_columns = ['workclass', 'education', 'marital-status', 'occupation',
#        'relationship', 'race', 'sex', 'native-country']
# target = ['income']

# #CHECKING TO MAKE SURE YOUR LIST IS CORRECT (NO NEED TO TOUCH)
# yourlist = cont_columns + cat_columns+target
# yourlist.sort()
# oglist = list(df_train.columns)
# oglist.sort()

# assert(yourlist == oglist), "You may of spelled feature name wrong or you forgot to put one of them in the list"

# cat_features = (10,16,7,16,6,5,2,43)

# target_classes = [max(len(df_train[target].value_counts()), len(df_val[target].value_counts()),len(df_test[target].value_counts()))]
# print(target_classes)
# # Create a StandardScaler and fit it to the cont features
# scaler = StandardScaler()
# scaler.fit(df_train[cont_columns])

# # Transform the training, test, and validation datasets
# df_train[cont_columns] = scaler.transform(df_train[cont_columns])
# df_test[cont_columns] = scaler.transform(df_test[cont_columns])
# df_val[cont_columns] = scaler.transform(df_val[cont_columns])

# #Wrapping in Dataset
# train_dataset = Combined_Dataset(df_train, cat_columns=cat_columns, num_columns=cont_columns, task1_column=target[0])
# val_dataset = Combined_Dataset(df_val, cat_columns=cat_columns, num_columns=cont_columns, task1_column=target[0])
# test_dataset = Combined_Dataset(df_test, cat_columns=cat_columns, num_columns=cont_columns, task1_column=target[0])

# batch_size = 256

# # Wrapping with DataLoader for easy batch extraction
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# #For xgboost
# X_train = df_train.drop(target[0], axis=1)
# y_train = df_train[target[0]]

# X_test = df_test.drop(target[0], axis=1)
# y_test = df_test[target[0]]


# for trial_num in range(3):
#     #TAB
#     tab_model = TabTransformer(categories = cat_features,      # tuple containing the number of unique values within each category
#                                 num_continuous = len(cont_columns),                # number of continuous values
#                                 dim = 32,                           # dimension, paper set at 32
#                                 dim_out = target_classes[0],               
#                                 depth = 6,                          # depth, paper recommended 6
#                                 heads = 8,                          # heads, paper recommends 8
#                                 attn_dropout = 0.1,                 # post-attention dropout
#                                 ff_dropout = 0.1,                   # feed forward dropout
#                                 mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
#                                 mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc))
#                                 ).to(device_in_use)

#     optimizer = torch.optim.AdamW(params=tab_model.parameters()) #no default lr or weight decay was given in the paper so I will use the default for AdamW
#     loss_function = nn.CrossEntropyLoss()

#     early_stopping = EarlyStopping(patience=10, verbose=True)

#     train_losses = []
#     train_accuracies_1 = [] 
#     train_f1s = []
#     test_losses = []
#     test_accuracies_1 = [] 
#     test_f1s = []

#     epochs = 800

#     for t in range(epochs):
#         train_loss, train_acc, train_f1= train(regression_on=False, 
#                                     get_attn=False,
#                                     dataloader=train_dataloader, 
#                                     model=tab_model, 
#                                     loss_function=loss_function, 
#                                     optimizer=optimizer, 
#                                     device_in_use=device_in_use)
#         test_loss, test_acc, test_f1= test(regression_on=False,
#                                 get_attn=False,
#                                 dataloader=test_dataloader,
#                                 model=tab_model,
#                                 loss_function=loss_function,
#                                 device_in_use=device_in_use)
#         train_losses.append(train_loss)
#         train_accuracies_1.append(train_acc)
#         train_f1s.append(train_f1)
#         test_losses.append(test_loss)
#         test_accuracies_1.append(test_acc)
#         test_f1s.append(test_f1)

#         epoch_str = f"Epoch [{t+1:2}/{epochs}]"
#         train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}, F1 {(train_f1)}"
#         test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}, F1 {(test_f1)}"
#         print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

#         early_stopping(test_acc)
        
#         if early_stopping.early_stop:
#             print("Early stopping")
#             break
    
#     print(trial_num)

#     performance_log.add_metric('TAB', 'Income', 'Test Accuracy', test_accuracies_1, trial=trial_num)
#     performance_log.add_metric('TAB', 'Income', 'Train Accuracy', train_accuracies_1, trial=trial_num)
#     performance_log.add_metric('TAB', 'Income', 'Test F1', test_f1s, trial=trial_num)
#     performance_log.add_metric('TAB', 'Income', 'Train Loss', train_losses, trial=trial_num)
#     performance_log.add_metric('TAB', 'Income', 'Test Loss', test_losses, trial=trial_num)


#     #XGBoost
#     xg_model = xgb.XGBClassifier()

#     xg_model.fit(X_train, y_train)

#     test_accuracies_1 = [] 
#     test_f1s = []

#     yhat = xg_model.predict(X_test)
#     acc = accuracy_score(y_test, yhat)
#     f1 = f1_score(y_test, yhat, average='weighted')

#     print(f"Accuracy: {acc}, F1: {f1}")

#     test_accuracies_1.append(acc)
#     test_f1s.append(f1)

#     performance_log.add_metric('XGBoost', 'Income', 'Test Accuracy', test_accuracies_1, trial=trial_num)
#     performance_log.add_metric('XGBoost', 'Income', 'Test F1', test_f1s, trial=trial_num)


# ###############################################################################################################################################################################################

# df_train = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/higgs/train.csv')
# df_test = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/higgs/test.csv')
# df_val = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/higgs/validation.csv') 

# cont_columns = ['lepton_pT', 'lepton_eta', 'lepton_phi',
#        'missing_energy_magnitude', 'missing_energy_phi', 'jet1pt', 'jet1eta',
#        'jet1phi', 'jet1b-tag', 'jet2pt', 'jet2eta', 'jet2phi', 'jet2b-tag',
#        'jet3pt', 'jet3eta', 'jet3phi', 'jet3b-tag', 'jet4pt', 'jet4eta',
#        'jet4phi', 'jet4b-tag', 'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb',
#        'm_wbb', 'm_wwbb']
# target = ['class']
# cat_columns = []


# #CHECKING TO MAKE SURE YOUR LIST IS CORRECT (NO NEED TO TOUCH)
# yourlist = cont_columns + cat_columns+target
# yourlist.sort()
# oglist = list(df_train.columns)
# oglist.sort()

# assert(yourlist == oglist), "You may of spelled feature name wrong or you forgot to put on of them in the list"

# cat_features = ()

# target_classes = [max(len(df_train[target].value_counts()), len(df_val[target].value_counts()),len(df_test[target].value_counts()))]
# print(target_classes)
# # Create a StandardScaler and fit it to the cont features
# scaler = StandardScaler()
# scaler.fit(df_train[cont_columns])

# # Transform the training, test, and validation datasets
# df_train[cont_columns] = scaler.transform(df_train[cont_columns])
# df_test[cont_columns] = scaler.transform(df_test[cont_columns])
# df_val[cont_columns] = scaler.transform(df_val[cont_columns])

# #Wrapping in Dataset
# train_dataset = Combined_Dataset(df_train, cat_columns=cat_columns, num_columns=cont_columns, task1_column=target[0])
# val_dataset = Combined_Dataset(df_val, cat_columns=cat_columns, num_columns=cont_columns, task1_column=target[0])
# test_dataset = Combined_Dataset(df_test, cat_columns=cat_columns, num_columns=cont_columns, task1_column=target[0])

# batch_size = 256

# # Wrapping with DataLoader for easy batch extraction
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# #For xgboost
# X_train = df_train.drop(target[0], axis=1)
# y_train = df_train[target[0]]

# X_test = df_test.drop(target[0], axis=1)
# y_test = df_test[target[0]]


# for trial_num in range(3):
#     #TAB
#     tab_model = TabTransformer(categories = cat_features,      # tuple containing the number of unique values within each category
#                                 num_continuous = len(cont_columns),                # number of continuous values
#                                 dim = 32,                           # dimension, paper set at 32
#                                 dim_out = target_classes[0],               
#                                 depth = 6,                          # depth, paper recommended 6
#                                 heads = 8,                          # heads, paper recommends 8
#                                 attn_dropout = 0.1,                 # post-attention dropout
#                                 ff_dropout = 0.1,                   # feed forward dropout
#                                 mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
#                                 mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc))
#                                 ).to(device_in_use)

#     optimizer = torch.optim.AdamW(params=tab_model.parameters()) #no default lr or weight decay was given in the paper so I will use the default for AdamW
#     loss_function = nn.CrossEntropyLoss()

#     early_stopping = EarlyStopping(patience=10, verbose=True)

#     train_losses = []
#     train_accuracies_1 = [] 
#     train_f1s = []
#     test_losses = []
#     test_accuracies_1 = [] 
#     test_f1s = []

#     epochs = 800

#     for t in range(epochs):
#         train_loss, train_acc, train_f1= train(regression_on=False, 
#                                     get_attn=False,
#                                     dataloader=train_dataloader, 
#                                     model=tab_model, 
#                                     loss_function=loss_function, 
#                                     optimizer=optimizer, 
#                                     device_in_use=device_in_use)
#         test_loss, test_acc, test_f1= test(regression_on=False,
#                                 get_attn=False,
#                                 dataloader=test_dataloader,
#                                 model=tab_model,
#                                 loss_function=loss_function,
#                                 device_in_use=device_in_use)
#         train_losses.append(train_loss)
#         train_accuracies_1.append(train_acc)
#         train_f1s.append(train_f1)
#         test_losses.append(test_loss)
#         test_accuracies_1.append(test_acc)
#         test_f1s.append(test_f1)

#         epoch_str = f"Epoch [{t+1:2}/{epochs}]"
#         train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}, F1 {(train_f1)}"
#         test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}, F1 {(test_f1)}"
#         print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

#         early_stopping(test_acc)
        
#         if early_stopping.early_stop:
#             print("Early stopping")
#             break
    
#     print(trial_num)

#     performance_log.add_metric('TAB', 'Higgs', 'Test Accuracy', test_accuracies_1, trial=trial_num)
#     performance_log.add_metric('TAB', 'Higgs', 'Train Accuracy', train_accuracies_1, trial=trial_num)
#     performance_log.add_metric('TAB', 'Higgs', 'Test F1', test_f1s, trial=trial_num)
#     performance_log.add_metric('TAB', 'Higgs', 'Train Loss', train_losses, trial=trial_num)
#     performance_log.add_metric('TAB', 'Higgs', 'Test Loss', test_losses, trial=trial_num)


#     #XGBoost
#     xg_model = xgb.XGBClassifier()

#     xg_model.fit(X_train, y_train)

#     test_accuracies_1 = [] 
#     test_f1s = []

#     yhat = xg_model.predict(X_test)
#     acc = accuracy_score(y_test, yhat)
#     f1 = f1_score(y_test, yhat, average='weighted')

#     print(f"Accuracy: {acc}, F1: {f1}")

#     test_accuracies_1.append(acc)
#     test_f1s.append(f1)

#     performance_log.add_metric('XGBoost', 'Higgs', 'Test Accuracy', test_accuracies_1, trial=trial_num)
#     performance_log.add_metric('XGBoost', 'Higgs', 'Test F1', test_f1s, trial=trial_num)


# ##################################################################################################################################################################################################

# #Get Helena

# # df_train = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\helena\train.csv')
# # df_test = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\helena\test.csv')
# # df_val = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\helena\validation.csv') #READ FROM RIGHT SPOT

# # df_train = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/helena/train.csv')
# # df_test = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/helena/test.csv')
# # df_val = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/helena/validation.csv')

# df_train = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/helena/train.csv')
# df_test = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/helena/test.csv')
# df_val = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/helena/validation.csv')


# # df_train.columns
# cont_columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
#        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
#        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27']
# target = ['class']
# cat_columns = []

# #CHECKING TO MAKE SURE YOUR LIST IS CORRECT (NO NEED TO TOUCH)
# yourlist = cont_columns + target
# yourlist.sort()
# oglist = list(df_train.columns)
# oglist.sort()

# cat_features = ()

# assert(yourlist == oglist), "You may of spelled feature name wrong or you forgot to put one of them in the list"

# target_classes = [max(len(df_train[target].value_counts()), len(df_val[target].value_counts()),len(df_test[target].value_counts()))]
# print("target classes",target_classes)
# # Create a StandardScaler and fit it to the cont features
# scaler = StandardScaler()
# scaler.fit(df_train[cont_columns])

# # Transform the training, test, and validation datasets
# df_train[cont_columns] = scaler.transform(df_train[cont_columns])
# df_test[cont_columns] = scaler.transform(df_test[cont_columns])
# df_val[cont_columns] = scaler.transform(df_val[cont_columns])

# #Wrapping in Dataset
# train_dataset = Combined_Dataset(df_train, cat_columns=[], num_columns=cont_columns, task1_column='class')
# val_dataset = Combined_Dataset(df_val, cat_columns=[], num_columns=cont_columns, task1_column='class')
# test_dataset = Combined_Dataset(df_test, cat_columns=[], num_columns=cont_columns, task1_column='class')

# #This is a hyperparameter that is not tuned. Maybe mess with what makes sense here
# batch_size = 256

# # Wrapping with DataLoader for easy batch extraction
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# #For xgboost
# X_train = df_train.drop(target[0], axis=1)
# y_train = df_train[target[0]]

# X_test = df_test.drop(target[0], axis=1)
# y_test = df_test[target[0]]


# for trial_num in range(3):
#     #TAB
#     tab_model = TabTransformer(categories = cat_features,      # tuple containing the number of unique values within each category
#                                 num_continuous = len(cont_columns),                # number of continuous values
#                                 dim = 32,                           # dimension, paper set at 32
#                                 dim_out = target_classes[0],               
#                                 depth = 6,                          # depth, paper recommended 6
#                                 heads = 8,                          # heads, paper recommends 8
#                                 attn_dropout = 0.1,                 # post-attention dropout
#                                 ff_dropout = 0.1,                   # feed forward dropout
#                                 mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
#                                 mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc))
#                                 ).to(device_in_use)

#     optimizer = torch.optim.AdamW(params=tab_model.parameters()) #no default lr or weight decay was given in the paper so I will use the default for AdamW
#     loss_function = nn.CrossEntropyLoss()

#     early_stopping = EarlyStopping(patience=10, verbose=True)

#     train_losses = []
#     train_accuracies_1 = [] 
#     train_f1s = []
#     test_losses = []
#     test_accuracies_1 = [] 
#     test_f1s = []

#     epochs = 800

#     for t in range(epochs):
#         train_loss, train_acc, train_f1= train(regression_on=False, 
#                                     get_attn=False,
#                                     dataloader=train_dataloader, 
#                                     model=tab_model, 
#                                     loss_function=loss_function, 
#                                     optimizer=optimizer, 
#                                     device_in_use=device_in_use)
#         test_loss, test_acc, test_f1= test(regression_on=False,
#                                 get_attn=False,
#                                 dataloader=test_dataloader,
#                                 model=tab_model,
#                                 loss_function=loss_function,
#                                 device_in_use=device_in_use)
#         train_losses.append(train_loss)
#         train_accuracies_1.append(train_acc)
#         train_f1s.append(train_f1)
#         test_losses.append(test_loss)
#         test_accuracies_1.append(test_acc)
#         test_f1s.append(test_f1)

#         epoch_str = f"Epoch [{t+1:2}/{epochs}]"
#         train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}, F1 {(train_f1)}"
#         test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}, F1 {(test_f1)}"
#         print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

#         early_stopping(test_acc)
        
#         if early_stopping.early_stop:
#             print("Early stopping")
#             break
    
#     print(trial_num)

#     performance_log.add_metric('TAB', 'Helena', 'Test Accuracy', test_accuracies_1, trial=trial_num)
#     performance_log.add_metric('TAB', 'Helena', 'Train Accuracy', train_accuracies_1, trial=trial_num)
#     performance_log.add_metric('TAB', 'Helena', 'Test F1', test_f1s, trial=trial_num)
#     performance_log.add_metric('TAB', 'Helena', 'Train Loss', train_losses, trial=trial_num)
#     performance_log.add_metric('TAB', 'Helena', 'Test Loss', test_losses, trial=trial_num)


#     #XGBoost
#     xg_model = xgb.XGBClassifier()

#     xg_model.fit(X_train, y_train)

#     test_accuracies_1 = [] 
#     test_f1s = []

#     yhat = xg_model.predict(X_test)
#     acc = accuracy_score(y_test, yhat)
#     f1 = f1_score(y_test, yhat, average='weighted')

#     print(f"Accuracy: {acc}, F1: {f1}")

#     test_accuracies_1.append(acc)
#     test_f1s.append(f1)

#     performance_log.add_metric('XGBoost', 'Helena', 'Test Accuracy', test_accuracies_1, trial=trial_num)
#     performance_log.add_metric('XGBoost', 'Helena', 'Test F1', test_f1s, trial=trial_num)

# ##############################################################################################################################################################################################

# # Covertype

# df_train = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/covertype/train.csv')
# df_test = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/covertype/test.csv')
# df_val = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/covertype/validation.csv') 

# cont_columns = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
#        'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
#        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
#        'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',
#        'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
#        'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
#        'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10',
#        'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14',
#        'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18',
#        'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22',
#        'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',
#        'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
#        'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
#        'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38',
#        'Soil_Type39', 'Soil_Type40']
# cat_columns = []
# target = ['Cover_Type']

# #CHECKING TO MAKE SURE YOUR LIST IS CORRECT (NO NEED TO TOUCH)
# yourlist = cont_columns + cat_columns+target
# yourlist.sort()
# oglist = list(df_train.columns)
# oglist.sort()

# assert(yourlist == oglist), "You may of spelled feature name wrong or you forgot to put on of them in the list"

# cat_features = ()

# target_classes = [max(len(df_train[target].value_counts()), len(df_val[target].value_counts()),len(df_test[target].value_counts()))]
# print(target_classes)
# # Create a StandardScaler and fit it to the cont features
# scaler = StandardScaler()
# scaler.fit(df_train[cont_columns])

# # Transform the training, test, and validation datasets
# df_train[cont_columns] = scaler.transform(df_train[cont_columns])
# df_test[cont_columns] = scaler.transform(df_test[cont_columns])
# df_val[cont_columns] = scaler.transform(df_val[cont_columns])

# #Wrapping in Dataset
# train_dataset = Combined_Dataset(df_train, cat_columns=cat_columns, num_columns=cont_columns, task1_column=target[0])
# val_dataset = Combined_Dataset(df_val, cat_columns=cat_columns, num_columns=cont_columns, task1_column=target[0])
# test_dataset = Combined_Dataset(df_test, cat_columns=cat_columns, num_columns=cont_columns, task1_column=target[0])

# batch_size = 256

# # Wrapping with DataLoader for easy batch extraction
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# #For xgboost
# X_train = df_train.drop(target[0], axis=1)
# y_train = df_train[target[0]]

# X_test = df_test.drop(target[0], axis=1)
# y_test = df_test[target[0]]


# for trial_num in range(3):
#     #TAB
#     tab_model = TabTransformer(categories = cat_features,      # tuple containing the number of unique values within each category
#                                 num_continuous = len(cont_columns),                # number of continuous values
#                                 dim = 32,                           # dimension, paper set at 32
#                                 dim_out = target_classes[0],               
#                                 depth = 6,                          # depth, paper recommended 6
#                                 heads = 8,                          # heads, paper recommends 8
#                                 attn_dropout = 0.1,                 # post-attention dropout
#                                 ff_dropout = 0.1,                   # feed forward dropout
#                                 mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
#                                 mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc))
#                                 ).to(device_in_use)

#     optimizer = torch.optim.AdamW(params=tab_model.parameters()) #no default lr or weight decay was given in the paper so I will use the default for AdamW
#     loss_function = nn.CrossEntropyLoss()

#     early_stopping = EarlyStopping(patience=10, verbose=True)

#     train_losses = []
#     train_accuracies_1 = [] 
#     train_f1s = []
#     test_losses = []
#     test_accuracies_1 = [] 
#     test_f1s = []

#     epochs = 800

#     for t in range(epochs):
#         train_loss, train_acc, train_f1= train(regression_on=False, 
#                                     get_attn=False,
#                                     dataloader=train_dataloader, 
#                                     model=tab_model, 
#                                     loss_function=loss_function, 
#                                     optimizer=optimizer, 
#                                     device_in_use=device_in_use)
#         test_loss, test_acc, test_f1= test(regression_on=False,
#                                 get_attn=False,
#                                 dataloader=test_dataloader,
#                                 model=tab_model,
#                                 loss_function=loss_function,
#                                 device_in_use=device_in_use)
#         train_losses.append(train_loss)
#         train_accuracies_1.append(train_acc)
#         train_f1s.append(train_f1)
#         test_losses.append(test_loss)
#         test_accuracies_1.append(test_acc)
#         test_f1s.append(test_f1)

#         epoch_str = f"Epoch [{t+1:2}/{epochs}]"
#         train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}, F1 {(train_f1)}"
#         test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}, F1 {(test_f1)}"
#         print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

#         early_stopping(test_acc)
        
#         if early_stopping.early_stop:
#             print("Early stopping")
#             break
    
#     print(trial_num)

#     performance_log.add_metric('TAB', 'Covertype', 'Test Accuracy', test_accuracies_1, trial=trial_num)
#     performance_log.add_metric('TAB', 'Covertype', 'Train Accuracy', train_accuracies_1, trial=trial_num)
#     performance_log.add_metric('TAB', 'Covertype', 'Test F1', test_f1s, trial=trial_num)
#     performance_log.add_metric('TAB', 'Covertype', 'Train Loss', train_losses, trial=trial_num)
#     performance_log.add_metric('TAB', 'Covertype', 'Test Loss', test_losses, trial=trial_num)


#     #XGBoost
#     xg_model = xgb.XGBClassifier()

#     xg_model.fit(X_train, y_train)

#     test_accuracies_1 = [] 
#     test_f1s = []

#     yhat = xg_model.predict(X_test)
#     acc = accuracy_score(y_test, yhat)
#     f1 = f1_score(y_test, yhat, average='weighted')

#     print(f"Accuracy: {acc}, F1: {f1}")

#     test_accuracies_1.append(acc)
#     test_f1s.append(f1)

#     performance_log.add_metric('XGBoost', 'Covertype', 'Test Accuracy', test_accuracies_1, trial=trial_num)
#     performance_log.add_metric('XGBoost', 'Covertype', 'Test F1', test_f1s, trial=trial_num)

# #############################################################################################################################################################################

# #GET Aloi

# # df_train = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/aloi/train.csv')
# # df_test = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/aloi/test.csv')
# # df_val = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/aloi/validation.csv') #READ FROM RIGHT SPOT

# # df_train = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\aloi\train.csv')
# # df_test = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\aloi\test.csv')
# # df_val = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\aloi\validation.csv') #READ FROM RIGHT

# df_train = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/aloi/train.csv')
# df_test = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/aloi/test.csv')
# df_val = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/aloi/validation.csv') 

# cont_columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', 
#                 '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', 
#                 '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', 
#                 '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', 
#                 '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', 
#                 '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', 
#                 '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', 
#                 '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', 
#                 '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', 
#                 '124', '125', '126', '127']
# target = ['target']
# cat_columns = []

# #CHECKING TO MAKE SURE YOUR LIST IS CORRECT (NO NEED TO TOUCH)
# yourlist = cont_columns + target
# yourlist.sort()
# oglist = list(df_train.columns)
# oglist.sort()

# cat_features=()

# assert(yourlist == oglist), "You may of spelled feature name wrong or you forgot to put on of them in the list"

# target_classes = [max(len(df_train[target].value_counts()), len(df_val[target].value_counts()),len(df_test[target].value_counts()))]
# print(target_classes)
# # Create a StandardScaler and fit it to the cont features
# scaler = StandardScaler()
# scaler.fit(df_train[cont_columns])

# # Transform the training, test, and validation datasets
# df_train[cont_columns] = scaler.transform(df_train[cont_columns])
# df_test[cont_columns] = scaler.transform(df_test[cont_columns])
# df_val[cont_columns] = scaler.transform(df_val[cont_columns])

# #Wrapping in Dataset
# train_dataset = Combined_Dataset(df_train, cat_columns=cat_columns, num_columns=cont_columns, task1_column=target[0])
# val_dataset = Combined_Dataset(df_val, cat_columns=cat_columns, num_columns=cont_columns, task1_column=target[0])
# test_dataset = Combined_Dataset(df_test, cat_columns=cat_columns, num_columns=cont_columns, task1_column=target[0])

# batch_size = 256

# # Wrapping with DataLoader for easy batch extraction
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# #For xgboost
# X_train = df_train.drop(target[0], axis=1)
# y_train = df_train[target[0]]

# X_test = df_test.drop(target[0], axis=1)
# y_test = df_test[target[0]]


# for trial_num in range(3):
#     #TAB
#     tab_model = TabTransformer(categories = cat_features,      # tuple containing the number of unique values within each category
#                                 num_continuous = len(cont_columns),                # number of continuous values
#                                 dim = 32,                           # dimension, paper set at 32
#                                 dim_out = target_classes[0],               
#                                 depth = 6,                          # depth, paper recommended 6
#                                 heads = 8,                          # heads, paper recommends 8
#                                 attn_dropout = 0.1,                 # post-attention dropout
#                                 ff_dropout = 0.1,                   # feed forward dropout
#                                 mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
#                                 mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc))
#                                 ).to(device_in_use)

#     optimizer = torch.optim.AdamW(params=tab_model.parameters()) #no default lr or weight decay was given in the paper so I will use the default for AdamW
#     loss_function = nn.CrossEntropyLoss()

#     early_stopping = EarlyStopping(patience=10, verbose=True)

#     train_losses = []
#     train_accuracies_1 = [] 
#     train_f1s = []
#     test_losses = []
#     test_accuracies_1 = [] 
#     test_f1s = []

#     epochs = 800

#     for t in range(epochs):
#         train_loss, train_acc, train_f1= train(regression_on=False, 
#                                     get_attn=False,
#                                     dataloader=train_dataloader, 
#                                     model=tab_model, 
#                                     loss_function=loss_function, 
#                                     optimizer=optimizer, 
#                                     device_in_use=device_in_use)
#         test_loss, test_acc, test_f1= test(regression_on=False,
#                                 get_attn=False,
#                                 dataloader=test_dataloader,
#                                 model=tab_model,
#                                 loss_function=loss_function,
#                                 device_in_use=device_in_use)
#         train_losses.append(train_loss)
#         train_accuracies_1.append(train_acc)
#         train_f1s.append(train_f1)
#         test_losses.append(test_loss)
#         test_accuracies_1.append(test_acc)
#         test_f1s.append(test_f1)

#         epoch_str = f"Epoch [{t+1:2}/{epochs}]"
#         train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}, F1 {(train_f1)}"
#         test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}, F1 {(test_f1)}"
#         print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

#         early_stopping(test_acc)
        
#         if early_stopping.early_stop:
#             print("Early stopping")
#             break
    
#     print(trial_num)

#     performance_log.add_metric('TAB', 'Aloi', 'Test Accuracy', test_accuracies_1, trial=trial_num)
#     performance_log.add_metric('TAB', 'Aloi', 'Train Accuracy', train_accuracies_1, trial=trial_num)
#     performance_log.add_metric('TAB', 'Aloi', 'Test F1', test_f1s, trial=trial_num)
#     performance_log.add_metric('TAB', 'Aloi', 'Train Loss', train_losses, trial=trial_num)
#     performance_log.add_metric('TAB', 'Aloi', 'Test Loss', test_losses, trial=trial_num)


#     #XGBoost
#     xg_model = xgb.XGBClassifier()

#     xg_model.fit(X_train, y_train)

#     test_accuracies_1 = [] 
#     test_f1s = []

#     yhat = xg_model.predict(X_test)
#     acc = accuracy_score(y_test, yhat)
#     f1 = f1_score(y_test, yhat, average='weighted')

#     print(f"Accuracy: {acc}, F1: {f1}")

#     test_accuracies_1.append(acc)
#     test_f1s.append(f1)

#     performance_log.add_metric('XGBoost', 'Aloi', 'Test Accuracy', test_accuracies_1, trial=trial_num)
#     performance_log.add_metric('XGBoost', 'Aloi', 'Test F1', test_f1s, trial=trial_num)

# ##################################################################################################################################################################################

# #GET California

# # df_train = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/california/train.csv')
# # df_test = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/california/test.csv')
# # df_val = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/california/validation.csv') #READ FROM RIGHT SPOT

# # df_train = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\california\train.csv')
# # df_test = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\california\test.csv')
# # df_val = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\california\validation.csv') #READ FROM RIGHT SPOT

# df_train = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/california/train.csv')
# df_test = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/california/test.csv')
# df_val = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/california/validation.csv')

# cont_columns = [ 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
#        'Latitude', 'Longitude']
# target = ['MedInc']
# cat_columns=[]

# #CHECKING TO MAKE SURE YOUR LIST IS CORRECT (NO NEED TO TOUCH)
# yourlist = cont_columns + target
# yourlist.sort()
# oglist = list(df_train.columns)
# oglist.sort()

# cat_features=()

# assert(yourlist == oglist), "You may of spelled feature name wrong or you forgot to put on of them in the list"

# target_classes = [max(len(df_train[target].value_counts()), len(df_val[target].value_counts()),len(df_test[target].value_counts()))]
# print(target_classes)
# # Create a StandardScaler and fit it to the cont features
# scaler = StandardScaler()
# scaler.fit(df_train[cont_columns])

# # Transform the training, test, and validation datasets
# df_train[cont_columns] = scaler.transform(df_train[cont_columns])
# df_test[cont_columns] = scaler.transform(df_test[cont_columns])
# df_val[cont_columns] = scaler.transform(df_val[cont_columns])

# #Wrapping in Dataset
# train_dataset = Combined_Dataset(df_train, cat_columns=cat_columns, num_columns=cont_columns, task1_column=target[0])
# val_dataset = Combined_Dataset(df_val, cat_columns=cat_columns, num_columns=cont_columns, task1_column=target[0])
# test_dataset = Combined_Dataset(df_test, cat_columns=cat_columns, num_columns=cont_columns, task1_column=target[0])

# batch_size = 256

# # Wrapping with DataLoader for easy batch extraction
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# #For xgboost
# X_train = df_train.drop(target[0], axis=1)
# y_train = df_train[target[0]]

# X_test = df_test.drop(target[0], axis=1)
# y_test = df_test[target[0]]

# for trial_num in range(3):
#     #TAB
#     tab_model = TabTransformer(categories = cat_features,      # tuple containing the number of unique values within each category
#                                 num_continuous = len(cont_columns),                # number of continuous values
#                                 dim = 32,                           # dimension, paper set at 32
#                                 dim_out = 1,               
#                                 depth = 6,                          # depth, paper recommended 6
#                                 heads = 8,                          # heads, paper recommends 8
#                                 attn_dropout = 0.1,                 # post-attention dropout
#                                 ff_dropout = 0.1,                   # feed forward dropout
#                                 mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
#                                 mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc))
#                                 ).to(device_in_use)

#     optimizer = torch.optim.AdamW(params=tab_model.parameters()) #no default lr or weight decay was given in the paper so I will use the default for AdamW
#     loss_function = nn.MSELoss()

#     early_stopping = EarlyStopping(patience=16, verbose=True, mode='min')

#     train_losses = []
#     train_rmse_1 = [] 
#     test_losses = []
#     test_rmse_1 = [] 

#     epochs = 800

#     for t in range(epochs):
#         train_loss, train_rmse = train(regression_on=True, 
#                                     get_attn=False,
#                                     dataloader=train_dataloader, 
#                                     model=tab_model, 
#                                     loss_function=loss_function, 
#                                     optimizer=optimizer, 
#                                     device_in_use=device_in_use)
#         test_loss, test_rmse = test(regression_on=True, 
#                                     get_attn=False,
#                                     dataloader=train_dataloader, 
#                                     model=tab_model, 
#                                     loss_function=loss_function, 
#                                     device_in_use=device_in_use)
#         train_losses.append(train_loss)
#         train_rmse_1.append(train_rmse)
#         test_losses.append(test_loss)
#         test_rmse_1.append(test_rmse)

#         epoch_str = f"Epoch [{t+1:2}/{epochs}]"
#         train_metrics = f"Train: Loss {(train_loss)}, RMSE {(train_rmse)}"
#         test_metrics = f"Test: Loss {(test_loss)}, RMSE {(test_rmse)}"
#         print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

#         early_stopping(test_rmse)
        
#         if early_stopping.early_stop:
#             print("Early stopping")
#             break
    
#     print(trial_num)

#     performance_log.add_metric('TAB', 'California', 'Test RMSE', test_rmse_1, trial=trial_num)
#     performance_log.add_metric('TAB', 'California', 'Train RMSE', train_rmse_1, trial=trial_num)
#     performance_log.add_metric('TAB', 'California', 'Train Loss', train_losses, trial=trial_num)
#     performance_log.add_metric('TAB', 'California', 'Test Loss', test_losses, trial=trial_num)


#     #XGBoost
#     xg_model = xgb.XGBRegressor()

#     xg_model.fit(X_train, y_train)

#     test_rmse_1 = [] 

#     yhat = xg_model.predict(X_test)
#     rmse_xg = rmse(y_test, yhat)

#     print(f"RMSE: {rmse_xg}")

#     test_rmse_1.append(rmse_xg)

#     performance_log.add_metric('XGBoost', 'California', 'Test RMSE', test_rmse_1, trial=trial_num)

# #############################################################################################################################################################################

# #GET Jannis

# # df_train = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/jannis/train.csv')
# # df_test = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/jannis/test.csv')
# # df_val = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/jannis/validation.csv') #READ FROM RIGHT SPOT

# # df_train = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\jannis\train.csv')
# # df_test = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\jannis\test.csv')
# # df_val = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\jannis\validation.csv') #READ FROM RIGHT

# df_train = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/jannis/train.csv')
# df_test = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/jannis/test.csv')
# df_val = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/jannis/validation.csv') 

# cont_columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
#        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
#        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30',
#        'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40',
#        'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50',
#        'V51', 'V52', 'V53', 'V54']
# cat_columns = []
# target = ['class']

# #CHECKING TO MAKE SURE YOUR LIST IS CORRECT (NO NEED TO TOUCH)
# yourlist = cont_columns + target
# yourlist.sort()
# oglist = list(df_train.columns)
# oglist.sort()

# cat_features=()

# assert(yourlist == oglist), "You may of spelled feature name wrong or you forgot to put on of them in the list"

# target_classes = [max(len(df_train[target].value_counts()), len(df_val[target].value_counts()),len(df_test[target].value_counts()))]
# print(target_classes)
# # Create a StandardScaler and fit it to the cont features
# scaler = StandardScaler()
# scaler.fit(df_train[cont_columns])

# # Transform the training, test, and validation datasets
# df_train[cont_columns] = scaler.transform(df_train[cont_columns])
# df_test[cont_columns] = scaler.transform(df_test[cont_columns])
# df_val[cont_columns] = scaler.transform(df_val[cont_columns])

# #Wrapping in Dataset
# train_dataset = Combined_Dataset(df_train, cat_columns=cat_columns, num_columns=cont_columns, task1_column=target[0])
# val_dataset = Combined_Dataset(df_val, cat_columns=cat_columns, num_columns=cont_columns, task1_column=target[0])
# test_dataset = Combined_Dataset(df_test, cat_columns=cat_columns, num_columns=cont_columns, task1_column=target[0])

# batch_size = 256

# # Wrapping with DataLoader for easy batch extraction
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# #For xgboost
# X_train = df_train.drop(target[0], axis=1)
# y_train = df_train[target[0]]

# X_test = df_test.drop(target[0], axis=1)
# y_test = df_test[target[0]]


# for trial_num in range(3):
#     #TAB
#     tab_model = TabTransformer(categories = cat_features,      # tuple containing the number of unique values within each category
#                                 num_continuous = len(cont_columns),                # number of continuous values
#                                 dim = 32,                           # dimension, paper set at 32
#                                 dim_out = target_classes[0],               
#                                 depth = 6,                          # depth, paper recommended 6
#                                 heads = 8,                          # heads, paper recommends 8
#                                 attn_dropout = 0.1,                 # post-attention dropout
#                                 ff_dropout = 0.1,                   # feed forward dropout
#                                 mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
#                                 mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc))
#                                 ).to(device_in_use)

#     optimizer = torch.optim.AdamW(params=tab_model.parameters()) #no default lr or weight decay was given in the paper so I will use the default for AdamW
#     loss_function = nn.CrossEntropyLoss()

#     early_stopping = EarlyStopping(patience=10, verbose=True)

#     train_losses = []
#     train_accuracies_1 = [] 
#     train_f1s = []
#     test_losses = []
#     test_accuracies_1 = [] 
#     test_f1s = []

#     epochs = 800

#     for t in range(epochs):
#         train_loss, train_acc, train_f1= train(regression_on=False, 
#                                     get_attn=False,
#                                     dataloader=train_dataloader, 
#                                     model=tab_model, 
#                                     loss_function=loss_function, 
#                                     optimizer=optimizer, 
#                                     device_in_use=device_in_use)
#         test_loss, test_acc, test_f1= test(regression_on=False,
#                                 get_attn=False,
#                                 dataloader=test_dataloader,
#                                 model=tab_model,
#                                 loss_function=loss_function,
#                                 device_in_use=device_in_use)
#         train_losses.append(train_loss)
#         train_accuracies_1.append(train_acc)
#         train_f1s.append(train_f1)
#         test_losses.append(test_loss)
#         test_accuracies_1.append(test_acc)
#         test_f1s.append(test_f1)

#         epoch_str = f"Epoch [{t+1:2}/{epochs}]"
#         train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}, F1 {(train_f1)}"
#         test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}, F1 {(test_f1)}"
#         print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

#         early_stopping(test_acc)
        
#         if early_stopping.early_stop:
#             print("Early stopping")
#             break
    
#     print(trial_num)

#     performance_log.add_metric('TAB', 'Jannis', 'Test Accuracy', test_accuracies_1, trial=trial_num)
#     performance_log.add_metric('TAB', 'Jannis', 'Train Accuracy', train_accuracies_1, trial=trial_num)
#     performance_log.add_metric('TAB', 'Jannis', 'Test F1', test_f1s, trial=trial_num)
#     performance_log.add_metric('TAB', 'Jannis', 'Train Loss', train_losses, trial=trial_num)
#     performance_log.add_metric('TAB', 'Jannis', 'Test Loss', test_losses, trial=trial_num)


#     #XGBoost
#     xg_model = xgb.XGBClassifier()

#     xg_model.fit(X_train, y_train)

#     test_accuracies_1 = [] 
#     test_f1s = []

#     yhat = xg_model.predict(X_test)
#     acc = accuracy_score(y_test, yhat)
#     f1 = f1_score(y_test, yhat, average='weighted')

#     print(f"Accuracy: {acc}, F1: {f1}")

#     test_accuracies_1.append(acc)
#     test_f1s.append(f1)

#     performance_log.add_metric('XGBoost', 'Jannis', 'Test Accuracy', test_accuracies_1, trial=trial_num)
#     performance_log.add_metric('XGBoost', 'Jannis', 'Test F1', test_f1s, trial=trial_num)



####################################################################################################################################################################################
    
#Wine

with open('/home/wdwatson2/projects/CAT-Transformer/new_experiments/performance_log.pkl', 'rb') as file:
    performance_log = pickle.load(file)

df_train = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/wine/train.csv')
df_test = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/wine/test.csv')
df_val = pd.read_csv('/home/wdwatson2/projects/CAT-Transformer/datasets/wine/validation.csv')

cont_columns = ['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar',
       'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide', 'density',
       'pH', 'sulphates', 'alcohol']
target = ['quality']
cat_columns=[]

#CHECKING TO MAKE SURE YOUR LIST IS CORRECT (NO NEED TO TOUCH)
yourlist = cont_columns + target
yourlist.sort()
oglist = list(df_train.columns)
oglist.sort()

cat_features=[]

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
train_dataset = Combined_Dataset(df_train, cat_columns=cat_columns, num_columns=cont_columns, task1_column=target[0])
val_dataset = Combined_Dataset(df_val, cat_columns=cat_columns, num_columns=cont_columns, task1_column=target[0])
test_dataset = Combined_Dataset(df_test, cat_columns=cat_columns, num_columns=cont_columns, task1_column=target[0])

batch_size = 256

# Wrapping with DataLoader for easy batch extraction
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#For xgboost
X_train = df_train.drop(target[0], axis=1)
y_train = df_train[target[0]]

X_test = df_test.drop(target[0], axis=1)
y_test = df_test[target[0]]

for trial_num in range(3):
    #TAB
    tab_model = TabTransformer(categories = cat_features,      # tuple containing the number of unique values within each category
                                num_continuous = len(cont_columns),                # number of continuous values
                                dim = 32,                           # dimension, paper set at 32
                                dim_out = 1,               
                                depth = 6,                          # depth, paper recommended 6
                                heads = 8,                          # heads, paper recommends 8
                                attn_dropout = 0.1,                 # post-attention dropout
                                ff_dropout = 0.1,                   # feed forward dropout
                                mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
                                mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc))
                                ).to(device_in_use)

    optimizer = torch.optim.AdamW(params=tab_model.parameters()) #no default lr or weight decay was given in the paper so I will use the default for AdamW
    loss_function = nn.MSELoss()

    early_stopping = EarlyStopping(patience=10, verbose=True, mode='min')

    train_losses = []
    train_rmse_1 = [] 
    test_losses = []
    test_rmse_1 = [] 

    epochs = 800

    for t in range(epochs):
        train_loss, train_rmse = train(regression_on=True, 
                                    get_attn=False,
                                    dataloader=train_dataloader, 
                                    model=tab_model, 
                                    loss_function=loss_function, 
                                    optimizer=optimizer, 
                                    device_in_use=device_in_use)
        test_loss, test_rmse = test(regression_on=True, 
                                    get_attn=False,
                                    dataloader=train_dataloader, 
                                    model=tab_model, 
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

        early_stopping(test_rmse)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    print(trial_num)

    performance_log.add_metric('TAB', 'Wine', 'Test RMSE', test_rmse_1, trial=trial_num)
    performance_log.add_metric('TAB', 'Wine', 'Train RMSE', train_rmse_1, trial=trial_num)
    performance_log.add_metric('TAB', 'Wine', 'Train Loss', train_losses, trial=trial_num)
    performance_log.add_metric('TAB', 'Wine', 'Test Loss', test_losses, trial=trial_num)


    #XGBoost
    xg_model = xgb.XGBRegressor()

    xg_model.fit(X_train, y_train)

    test_rmse_1 = [] 

    yhat = xg_model.predict(X_test)
    rmse_xg = rmse(y_test, yhat)

    print(f"RMSE: {rmse_xg}")

    test_rmse_1.append(rmse_xg)

    performance_log.add_metric('XGBoost', 'Wine', 'Test RMSE', test_rmse_1, trial=trial_num)

with open('/home/wdwatson2/projects/CAT-Transformer/new_experiments/performance_log.pkl', 'wb') as file:
    pickle.dump(performance_log, file)