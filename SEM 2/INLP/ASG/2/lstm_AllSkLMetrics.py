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

device = "cuda" if torch.cuda.is_available() else "cpu"
sentenceLens = {}

def AllSkLMetrics(model, data):
    model.eval()
    y_true = []
    y_pred = []

    for i, (x, y) in enumerate(DataLoader(data, batch_size=32, shuffle=True)):
        x = x.to(model.device)
        y = y.to(model.device)

        output = model(x)

        y = y.view(-1)
        output = output.view(-1, output.shape[-1])
        suitableIndex = [i for i in range(x.view(-1).size(0)) if x.view(-1)[i] != model.train_data.PADIndex]
        _, predicted = torch.max(output, 1)
        y_masked = y[suitableIndex]
        y_pred_masked = predicted[suitableIndex]
        y_true.extend(y_masked.tolist())
        y_pred.extend(y_pred_masked.tolist())

    y_trueTag = [model.train_data.tagIndex2Word[i] for i in y_true]
    y_predTag = [model.train_data.tagIndex2Word[i] for i in y_pred]

    # Calculate metrics
    sk_accuracy = accuracy_score(y_trueTag, y_predTag)
    classification_rep = classification_report(y_trueTag, y_predTag)
    recall_micro = recall_score(y_trueTag, y_predTag, average='micro')
    recall_macro = recall_score(y_trueTag, y_predTag, average='macro')
    f1_micro = f1_score(y_trueTag, y_predTag, average='micro')
    f1_macro = f1_score(y_trueTag, y_predTag, average='macro')
    confusion_mat = confusion_matrix(y_trueTag, y_predTag)

    return sk_accuracy, classification_rep, recall_micro, recall_macro, f1_micro, f1_macro, confusion_mat