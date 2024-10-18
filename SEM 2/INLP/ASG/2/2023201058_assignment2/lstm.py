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

from lstm_customdataset import CustomDataset
from lstm_lstmnn import LSTM
# from lstm_getLossDataset import getLossDataset
from lstm_train import train
from lstm_accuracy import accuracy
from lstm_AllSkLMetrics import AllSkLMetrics

device = "cuda" if torch.cuda.is_available() else "cpu"
sentenceLens = {}

mod_num_list = [1, 2, 3]

for mod_num in mod_num_list:
    sentenceLens = {}
    if mod_num == 1:
        lay = 64
        lea_rt = 0.01
        patience = 1
        test_name = "Test1"
    elif mod_num == 2:
        lay = 128
        lea_rt = 0.025
        patience = 4
        test_name = "Test2"
    elif mod_num == 3:    
        lay = 32
        lea_rt = 0.001
        patience = 4
        test_name = "Test3"

    inputT = open('en_atis-ud-train.conllu', 'r', encoding='utf-8')
    data = parse(inputT.read())
    val = open('en_atis-ud-dev.conllu', 'r', encoding='utf-8')
    valData = parse(val.read())
    tags = []
    sentences = []
    for sentence in data:
        unitTag = []
        unitSentence = []
        for token in sentence:
            unitTag.append(token["upostag"])
            unitSentence.append(token["form"])

        tags.append(unitTag)
        sentences.append(unitSentence)

    trainData = CustomDataset(sentences.copy(), tags.copy())
    trainData.LowFrequencyWordRemover(2)

    tags = []
    sentences = []
    for sentence in valData:
        unitTag = []
        unitSentence = []
        for token in sentence:
            if token["upostag"] == 'SYM':
                continue
            unitTag.append(token["upostag"])
            unitSentence.append(token["form"])

        tags.append(unitTag)
        sentences.append(unitSentence)

    valData = CustomDataset(sentences.copy(), tags.copy())


    if os.path.exists(f"lstm_model_{mod_num}.pt"):
        print("Loading model...")
        model = torch.load(f"lstm_model_{mod_num}.pt")
        valData.OOV_Handler(model.train_data.vocabSet, model.train_data.vocab, model.train_data.tagVocabSet, model.train_data.tagVocab)
    else:
        print("Training model...")
        # def __init__(self, input_size, hidden_size, num_layers, vocab_size, outputSize):
        model = LSTM(lay, lay, 1, len(trainData.vocab), len(trainData.tagVocab))
        model.train_data = trainData
        valData.OOV_Handler(model.train_data.vocabSet, model.train_data.vocab, model.train_data.tagVocabSet, model.train_data.tagVocab)
        optimizer = torch.optim.Adam(model.parameters(), lr=lea_rt)
        criterion = nn.CrossEntropyLoss()
        # def train(model, data, optimizer, criterion, valDat, maxPat, special1, special2):
        train_loss_list, valid_loss_list, train_accuracy_list, valid_accuracy_list = train(model, trainData, optimizer, criterion, valData, patience, test_name)
        torch.save(model, f"lstm_model_{mod_num}.pt")



    if os.path.exists(f"lstm_model_{mod_num}.pt"):
        print("Loading model...")
        model = torch.load(f"lstm_model_{mod_num}.pt")
        model.eval()
        test = open('en_atis-ud-test.conllu', 'r', encoding='utf-8')
        testData = conllu.parse(test.read())
        tags = []
        sentences = []
        for sentence in testData:
            unitTag = []
            unitSentence = []
            for token in sentence:
                unitTag.append(token["upostag"])
                unitSentence.append(token["form"])

            tags.append(unitTag)
            sentences.append(unitSentence)

        testData = CustomDataset(sentences.copy(), tags.copy())
        testData.OOV_Handler(model.train_data.vocabSet, model.train_data.vocab, model.train_data.tagVocabSet, model.train_data.tagVocab)
        print(f"Training Accuracy: {accuracy(model, model.train_data)}")

        # print(f"Validation Accuracy: {accuracy(model, valData)}")

        # print(f"Testing Accuracy: {accuracy(model, testData)}")

        valData_sk_accuracy, valData_classification_rep, valData_recall_micro, valData_recall_macro, valData_f1_micro, valData_f1_macro, valData_confusion_mat = AllSkLMetrics(model, valData)

        testData_sk_accuracy, testData_classification_rep, testData_recall_micro, testData_recall_macro, testData_f1_micro, testData_f1_macro, testData_confusion_mat = AllSkLMetrics(model, testData)

        print(f"Metrics of val set: ")
        # print(f"{accuracy_dev=}")
        print(f"{valData_sk_accuracy=}")
        print(f"valData_classification_rep = ")
        print(valData_classification_rep)
        print(f"{valData_recall_micro=}")
        print(f"{valData_recall_macro=}")
        print(f"{valData_f1_micro=}")
        print(f"{valData_f1_macro=}")
        print(f"valData_confusion_mat = ")
        print(valData_confusion_mat)
        print()

        print(f"Metrics of test set: ")
        # print(f"{accuracy_test=}")
        print(f"{testData_sk_accuracy=}")
        print(f"testData_classification_rep = ")
        print(testData_classification_rep)
        print(f"{testData_recall_micro=}")
        print(f"{testData_recall_macro=}")
        print(f"{testData_f1_micro=}")
        print(f"{testData_f1_macro=}")
        print(f"testData_confusion_mat = ")
        print(testData_confusion_mat)
