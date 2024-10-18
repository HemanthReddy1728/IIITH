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

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, vocab_size, outputSize):
        super(LSTM, self).__init__()  # call the init function of the parent class
        self.device = device
        self.num_layers = num_layers  # number of LSTM layers
        self.hidden_size = hidden_size  # size of LSTM hidden state
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)  # LSTM layer
        self.decoder = nn.Linear(hidden_size, outputSize)  # linear layer to map the hidden state to output classes
        self.train_data = None

        self.embedLayer = nn.Embedding(vocab_size, input_size)

        self.to(self.device)

    def forward(self, x, state=None):
        # Set initial states for the LSTM layer or use the states passed from the previous time step
        embeddings = self.embedLayer(x)

        # Forward propagate through the LSTM layer
        out, _ = self.lstm(embeddings)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        return self.decoder(out)




