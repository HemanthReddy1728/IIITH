# !pip install conllu
# !gdown 1LuAgqwOTsRIcYysARoYMkaHGBKy3rkBm
# !gdown 1tRS2vreMs3qV6qNLHkD384L6XLB7y0_U
# !gdown 1aRrilCwIpJZowOkg_Meo7L4c9B_QjLoU
# !gdown 1yJdlUdv8cOVdnZNit4VrkPcrbd-N8FkW
# !gdown 11yci296NmMco-hnfH0Y1nWYOVLJI_pUW
# !gdown 1X--IPgjQlxQLZhGJk241dWHgGfmixEE5
# !gdown 10M530qW7v8Xwscbu4iQ28BSmGqAHkClo
# !gdown 1yXwdLzXCW9cxdFu1De7imKqiT21rFkIh
# !gdown 1gUEIp-_wPvQiQf5PhajpxXjW2sAgjqdR
# !gdown 1w3T3Ca1OPQdjekhmBzSDqhQ0c-HDJJpT

import time
import os
import sys
import json
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import conllu
from conllu import parse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, confusion_matrix

from lstm_accuracy import accuracy
from lstm_getLossDataset import getLossDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
sentenceLens = {}

def train(model, data, optimizer, criterion, valDat, maxPat, MODEL_NAME):
    epoch_loss = 0
    epoch = 0
    prevLoss = 10000000
    prevValLoss = 10000000
    train_accuracy_list = []
    train_loss_list = []
    es_patience = maxPat
    valid_loss_list = []
    valid_accuracy_list = []
    model.train()
    dataL = DataLoader(data, batch_size=32, shuffle=True)
    lossDec = True
    model.train_data = data
    while lossDec:
        epoch_loss = 0
        for i, (x, y) in enumerate(dataL):
            optimizer.zero_grad()
            x = x.to(model.device)

            y = y.to(model.device)

            output = model(x)

            y = y.view(-1)
            output = output.view(-1, output.shape[-1])

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()


        validationLoss = getLossDataset(valDat, model)
        print(f"Validation loss: {validationLoss}")
        if validationLoss > prevValLoss:
            print("Validation loss increased")
            if es_patience > 0:
                es_patience -= 1
            else:  # early stopping
                print("Early stopping")
                # model = torch.load(f"{MODEL_NAME}.pt")
                model.load_state_dict(torch.load(f"{MODEL_NAME}.pt"))
                lossDec = False
        else:
            # torch.save(model, f"{MODEL_NAME}.pt")
            torch.save(model.state_dict(), f"{MODEL_NAME}.pt")
            es_patience = maxPat
        prevValLoss = validationLoss

        model.train()
        if epoch_loss / len(dataL) > prevLoss:
            lossDec = False
        prevLoss = epoch_loss / len(dataL)

        print(f"Epoch {epoch + 1} loss: {epoch_loss / len(dataL)}")
        epoch += 1
        train_accuracy_list.append(accuracy(model, data))
        train_loss_list.append(epoch_loss / len(dataL))
        valid_loss_list.append(validationLoss.item())
        valid_accuracy_list.append(accuracy(model, valDat))
        # print(train_loss_list, valid_loss_list, train_accuracy_list, valid_accuracy_list)

    plt.plot(train_accuracy_list, c="red", label="Train Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    plt.plot(train_loss_list, c="blue", label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    plt.plot(valid_loss_list, c="green", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    plt.plot(valid_accuracy_list, c="yellow", label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    return train_loss_list, valid_loss_list, train_accuracy_list, valid_accuracy_list
