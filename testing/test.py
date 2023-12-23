import sys
# sys.path.insert(0, '/home/cscadmin/CyberResearch/CAT-Transformer/model')
sys.path.insert(0, r'C:\Users\smbm2\projects\CAT-Transformer\model')
# sys.path.insert(0, '/home/warin/projects/CAT-Transformer/model')
from updatedModel import CATTransformer, Combined_Dataset, train, test, count_parameters
from tab_transformer_pytorch import FTTransformer
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

device_in_use = 'cuda'

#HELENA
df_train = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\helena\train.csv')
df_test = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\helena\test.csv')
df_val = pd.read_csv(r'C:\Users\smbm2\projects\CAT-Transformer\datasets\helena\validation.csv') #READ FROM RIGHT SPOT

# df_train = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/helena/train.csv')
# df_test = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/helena/test.csv')
# df_val = pd.read_csv('/home/cscadmin/CyberResearch/CAT-Transformer/datasets/helena/validation.csv')

# df_train = pd.read_csv('/home/warin/projects/CAT-Transformer/datasets/helena/train.csv')
# df_test = pd.read_csv('/home/warin/projects/CAT-Transformer/datasets/helena/test.csv')
# df_val = pd.read_csv('/home/warin/projects/CAT-Transformer/datasets/helena/validation.csv')


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


model = CATTransformer(n_cont=len(cont_columns),
                       cat_feat=[], #pass cat_feat an emtpy list if dataset contains no cat features
                       targets_classes=target_classes,
                       embed_size=160
                       ).to(device_in_use)

print("CAT", count_parameters(model))

model2 = FTTransformer(categories=(),
                       num_continuous=len(cont_columns),
                       dim=160,
                       depth=1,
                       heads=5,
                       dim_out=target_classes[0]
                       ).to(device_in_use)

print("FT", count_parameters(model2))

# loss_function = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

# train_losses = []
# train_accuracies_1 = [] 
# test_losses = []
# test_accuracies_1 = [] 
# test_f1_scores = [] 

# epochs = 200

# for t in range(epochs):
#     train_loss, train_acc = train(regression_on=False, 
#                                    dataloader=train_dataloader, 
#                                    model=model, 
#                                    loss_function=loss_function, 
#                                    optimizer=optimizer, 
#                                    device_in_use=device_in_use)
#     test_loss, test_acc = test(regression_on=False,
#                                dataloader=test_dataloader,
#                                model=model,
#                                loss_function=loss_function,
#                                device_in_use=device_in_use)
#     train_losses.append(train_loss)
#     train_accuracies_1.append(train_acc)
#     test_losses.append(test_loss)
#     test_accuracies_1.append(test_acc)

#     epoch_str = f"Epoch [{t+1:2}/{epochs}]"
#     train_metrics = f"Train: Loss {(train_loss)}, Accuracy {(train_acc)}"
#     test_metrics = f"Test: Loss {(test_loss)}, Accuracy {(test_acc)}"
#     print(f"{epoch_str:15} | {train_metrics:65} | {test_metrics:65}")

# best_index = test_accuracies_1.index(max(test_accuracies_1))
# print(f"Best accuracy {test_accuracies_1[best_index]}\n")

# # Plotting the loss curves
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 2, 1)
# plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
# plt.plot(range(1, epochs+1), [l for l in test_losses], label='Test Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Test Loss Curve')
# plt.legend()

# # Plotting the accuracy curves
# plt.subplot(1, 2, 2)
# plt.plot(range(1, epochs+1), train_accuracies_1, label='Train Accuracy')
# plt.plot(range(1, epochs+1), test_accuracies_1, label='Test Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Training and Test Accuracy Curve')
# plt.legend()