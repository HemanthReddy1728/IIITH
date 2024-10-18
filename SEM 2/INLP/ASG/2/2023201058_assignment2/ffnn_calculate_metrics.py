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

def calculate_metrics(model, dataloader, unique_pos_tag_rev_dict):
    model.eval()
    predictions = []
    true_labels = []
    prediction_ids = []
    true_label_ids = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            prediction_ids.extend(predicted.tolist())
            true_label_ids.extend(labels.tolist())

    predictions = [unique_pos_tag_rev_dict[i] for i in prediction_ids]
    true_labels = [unique_pos_tag_rev_dict[i] for i in true_label_ids]

    sk_accuracy = accuracy_score(true_labels, predictions)
    classification_rep = classification_report(true_labels, predictions)
    recall_micro = recall_score(true_labels, predictions, average='micro')
    recall_macro = recall_score(true_labels, predictions, average='macro')
    f1_micro = f1_score(true_labels, predictions, average='micro')
    f1_macro = f1_score(true_labels, predictions, average='macro')
    confusion_mat = confusion_matrix(true_labels, predictions)

    return sk_accuracy, classification_rep, recall_micro, recall_macro, f1_micro, f1_macro, confusion_mat