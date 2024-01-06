import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

def rmse(y_true, y_pred):
    # Calculate the squared differences
    squared_diff = (y_true - y_pred)**2

    # Calculate the mean of the squared differences
    mean_squared_diff = torch.mean(squared_diff)

    # Calculate the square root to obtain RMSE
    rmse = torch.sqrt(mean_squared_diff)

    return rmse.item()  # Convert to a Python float

#Training and Testing Loops for Different Cases
def train(get_attn, regression_on, dataloader, model, loss_function, optimizer, device_in_use):
    model.train()

    total_loss=0
    total_correct_1 = 0
    total_samples_1 = 0
    all_targets_1 = []
    all_predictions_1 = []

    attention_ovr_batch = []

    total_rmse = 0

    if not regression_on:
        for (cat_x, cont_x, labels) in dataloader:
            cat_x,cont_x,labels=cat_x.to(device_in_use),cont_x.to(device_in_use),labels.to(device_in_use)

            predictions = model(cat_x, cont_x)

            loss = loss_function(predictions, labels.long())
            total_loss+=loss.item()

            #computing accuracy
            y_pred_softmax_1 = torch.softmax(predictions, dim=1)
            _, y_pred_labels_1 = torch.max(y_pred_softmax_1, dim=1)
            total_correct_1 += (y_pred_labels_1 == labels).sum().item()
            total_samples_1 += labels.size(0)
            all_targets_1.extend(labels.cpu().numpy())
            all_predictions_1.extend(y_pred_labels_1.cpu().numpy())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.empty_cache()

        avg_loss = total_loss/len(dataloader)
        accuracy = total_correct_1 / total_samples_1

        if get_attn:
            return avg_loss, accuracy, attention_ovr_batch
        else:
            return avg_loss, accuracy
    
    else:
        for (cat_x, cont_x, labels) in dataloader:
            cat_x,cont_x,labels=cat_x.to(device_in_use),cont_x.to(device_in_use),labels.to(device_in_use)

            if get_attn:
                predictions, attention = model(cat_x, cont_x)
                attention_ovr_batch.append(attention.detach().cpu().numpy())
            else:
                predictions = model(cat_x, cont_x)

            loss = loss_function(predictions, labels.unsqueeze(1))
            total_loss+=loss.item()

            rmse_value = rmse(labels.unsqueeze(1), predictions)
            total_rmse+=rmse_value

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.empty_cache()

        avg_loss = total_loss/len(dataloader)
        avg_rmse = total_rmse/len(dataloader)

        if get_attn:
            return avg_loss, accuracy, attention_ovr_batch
        else:
            return avg_loss, accuracy

def test(get_attn, regression_on, dataloader, model, loss_function, device_in_use):
    model.eval()

    total_loss=0
    total_correct_1 = 0
    total_samples_1 = 0
    all_targets_1 = []
    all_predictions_1 = []

    attention_ovr_batch = []

    total_rmse = 0

    if not regression_on:
        with torch.no_grad():
            for (cat_x, cont_x, labels) in dataloader:
                cat_x,cont_x,labels=cat_x.to(device_in_use),cont_x.to(device_in_use),labels.to(device_in_use)

                if get_attn:
                    predictions, attention = model(cat_x, cont_x)
                    attention_ovr_batch.append(attention.detach().cpu().numpy())
                else:
                    predictions = model(cat_x, cont_x)

                loss = loss_function(predictions, labels.long())
                total_loss+=loss.item()

                #computing accuracy
                y_pred_softmax_1 = torch.softmax(predictions, dim=1)
                _, y_pred_labels_1 = torch.max(y_pred_softmax_1, dim=1)
                total_correct_1 += (y_pred_labels_1 == labels).sum().item()
                total_samples_1 += labels.size(0)
                all_targets_1.extend(labels.cpu().numpy())
                all_predictions_1.extend(y_pred_labels_1.cpu().numpy())

                torch.cuda.empty_cache()

            avg_loss = total_loss/len(dataloader)
            accuracy = total_correct_1 / total_samples_1

            if get_attn:
                return avg_loss, accuracy, attention_ovr_batch
            else:
                return avg_loss, accuracy
    
    else:
        with torch.no_grad():
            for (cat_x, cont_x, labels) in dataloader:
                cat_x,cont_x,labels=cat_x.to(device_in_use),cont_x.to(device_in_use),labels.to(device_in_use)

                if get_attn:
                    predictions, attention = model(cat_x, cont_x)
                    attention_ovr_batch.append(attention.detach().cpu().numpy())
                else:
                    predictions = model(cat_x, cont_x)

                loss = loss_function(predictions, labels.unsqueeze(1))
                total_loss+=loss.item()

                rmse_value = rmse(labels.unsqueeze(1), predictions)
                total_rmse+=rmse_value

                torch.cuda.empty_cache()

            avg_loss = total_loss/len(dataloader)
            avg_rmse = total_rmse/len(dataloader)

            if get_attn:
                return avg_loss, accuracy, attention_ovr_batch
            else:
                return avg_loss, accuracy
       