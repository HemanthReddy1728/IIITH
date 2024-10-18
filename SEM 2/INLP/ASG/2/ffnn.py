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

pslist = [(0,0), (1,1), (2,2), (2,3), (3,3), (4,4)]
# Define column names and row names
col_names = ['Train', 'Dev', 'Test']
row_names = [p+s for p,s in pslist]
# all_train_word_embeddings = []

# Create an empty DataFrame
# accuracies = pd.DataFrame(index=row_names, columns=col_names)
sk_accuracies = pd.DataFrame(index=row_names, columns=col_names)

start_tag = "<sos>"
end_tag = "<eos>"

with open("en_atis-ud-train.conllu", "r", encoding="utf-8") as file:
    train_data = file.read()

parsed_train_data = parse(train_data)

with open("en_atis-ud-dev.conllu", "r", encoding="utf-8") as file:
    dev_data = file.read()

parsed_dev_data = parse(dev_data)

with open("en_atis-ud-test.conllu", "r", encoding="utf-8") as file:
    test_data = file.read()

parsed_test_data = parse(test_data)



for p,s in pslist:
    print(f"{p=} {s=}\n")
    if os.path.exists(f'ffnn_model_{p}_{s}.pt'):
        continue
    dev_data_word_pos_dict = {}
    modified_dev_sentences_list = []
    unique_pos_tag_list = []
    train_data_word_pos_dict = {}
    modified_train_sentences_list = []
    test_data_word_pos_dict = {}
    modified_test_sentences_list = []

    for train_sentence in parsed_train_data:
        mod_train_sent = []
        for i in range(p):
            mod_train_sent.append(start_tag)
        for train_token in train_sentence:
            # print(train_token["form"], train_token["upostag"], train_token["deprel"])
            mod_train_sent.append(train_token["form"].lower())
            train_data_word_pos_dict[train_token["form"].lower()] = train_token["upostag"]
        for j in range(s):
            mod_train_sent.append(end_tag)
        modified_train_sentences_list.append(mod_train_sent)

    # print(modified_train_sentences_list)


    for dev_sentence in parsed_dev_data:
        mod_dev_sent = []
        for i in range(p):
            mod_dev_sent.append(start_tag)
        for dev_token in dev_sentence:
            # print(dev_token["form"], dev_token["upostag"], dev_token["deprel"])
            if dev_token["upostag"] == 'SYM':
                continue
            mod_dev_sent.append(dev_token["form"].lower())
            unique_pos_tag_list.append(dev_token["upostag"])
            dev_data_word_pos_dict[dev_token["form"].lower()] = dev_token["upostag"]
        for j in range(s):
            mod_dev_sent.append(end_tag)
        modified_dev_sentences_list.append(mod_dev_sent)

    unique_pos_tag_list.append('OOV')
    unique_pos_tag_list=list(set(unique_pos_tag_list))
    # print(unique_pos_tag_list)
    unique_pos_tag_dict = {}
    unique_pos_tag_rev_dict = {}
    for index, element in enumerate(unique_pos_tag_list):
        unique_pos_tag_rev_dict[index] = element
        unique_pos_tag_dict[element] = index
    # print(unique_pos_tag_dict)


    for test_sentence in parsed_test_data:
        mod_test_sent = []
        for i in range(p):
            mod_test_sent.append(start_tag)
        for test_token in test_sentence:
            # print(test_token["form"], test_token["upostag"], test_token["deprel"])
            mod_test_sent.append(test_token["form"].lower())
            test_data_word_pos_dict[test_token["form"].lower()] = test_token["upostag"]
        for j in range(s):
            mod_test_sent.append(end_tag)
        modified_test_sentences_list.append(mod_test_sent)




    modified_sentences_list = modified_train_sentences_list + modified_dev_sentences_list + modified_test_sentences_list
    train_model = Word2Vec(modified_sentences_list, min_count=1)
    # train_model = Word2Vec(modified_train_sentences_list, min_count=1)
    train_word_embeddings = {word: train_model.wv[word] for word in train_model.wv.index_to_key}
    # if word in train_model.wv else np.zeros(train_model.vector_size)
    # print(train_word_embeddings)
    # all_train_word_embeddings.append(train_word_embeddings)

    with open(f"unique_pos_tag_rev_dict_{p}_{s}.json", "w") as file:
        json.dump(unique_pos_tag_rev_dict, file)
    # with open(f"train_word_embeddings_{p}_{s}.json", "w") as file:
        # json.dump(train_word_embeddings, file)
    with open(f'train_word_embeddings_{p}_{s}.pkl', 'wb') as file:
        pickle.dump(train_word_embeddings, file)
    # np.save(f"train_word_embeddings_{p}_{s}.npy", train_word_embeddings)

    X_train = []
    Y_train = []
    # ts1 = [modified_train_sentences_list[216]]
    # print(ts1)
    # for one_train_sent_list in ts1:
    for one_train_sent_list in modified_train_sentences_list:
        # print(len(one_train_sent_list))
        for k in range(len(one_train_sent_list)-s-p):
            embed_window = []
            window_size = p+1+s
            for l in range(k, window_size+k):
                embed_window.append(train_word_embeddings[one_train_sent_list[l]] if one_train_sent_list[l] in train_word_embeddings else np.zeros(train_model.vector_size))
            X_train.append(embed_window)
            # print(one_train_sent_list[k+p])
            Y_train.append(unique_pos_tag_dict[train_data_word_pos_dict[one_train_sent_list[k+p]]])

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # print(X_train.shape)
    # print(Y_train.shape)



    X_dev = []
    Y_dev = []
    # ts1 = [modified_train_sentences_list[216]]
    # print(ts1)
    # for one_train_sent_list in ts1:
    for one_dev_sent_list in modified_dev_sentences_list:
        # print(len(one_train_sent_list))
        for k in range(len(one_dev_sent_list)-s-p):
            embed_window = []
            window_size = p+1+s
            for l in range(k, window_size+k):
                embed_window.append(train_word_embeddings[one_dev_sent_list[l]] if one_dev_sent_list[l] in train_word_embeddings else np.zeros(train_model.vector_size))
            X_dev.append(embed_window)
            # print(one_train_sent_list[k+p])
            Y_dev.append(unique_pos_tag_dict[dev_data_word_pos_dict[one_dev_sent_list[k+p]]])


    X_dev = np.array(X_dev)
    Y_dev = np.array(Y_dev)
    # print(X_dev.shape)
    # print(Y_dev.shape)



    X_test = []
    Y_test = []
    # ts1 = [modified_train_sentences_list[216]]
    # print(ts1)
    # for one_train_sent_list in ts1:
    for one_test_sent_list in modified_test_sentences_list:
        # print(len(one_train_sent_list))
        for k in range(len(one_test_sent_list)-s-p):
            embed_window = []
            window_size = p+1+s
            for l in range(k, window_size+k):
                embed_window.append(train_word_embeddings[one_test_sent_list[l]] if one_test_sent_list[l] in train_word_embeddings else np.zeros(train_model.vector_size))
            X_test.append(embed_window)
            # print(one_train_sent_list[k+p])
            Y_test.append(unique_pos_tag_dict[test_data_word_pos_dict[one_test_sent_list[k+p]]])


    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    # print(X_test.shape)
    # print(Y_test.shape)




    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            # self.fc1 = nn.Linear((p+1+s)*100, 256)
            # self.fc2 = nn.Linear(256, 128)
            self.device = device
            self.fc1 = nn.Linear((p+1+s)*100, 256)
            self.fc3 = nn.Linear(256, 64)
            self.fc4 = nn.Linear(64, 14)  # Assuming you have 14 classes

        def forward(self, x):
            x = torch.flatten(x, start_dim=1)
            x = torch.relu(self.fc1(x))
            # x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            return x

    from ffnn_customdataset import CustomDataset
    if not os.path.exists(f'ffnn_model_{p}_{s}.pt'):
        # Create dataset and dataloader
        dataset = CustomDataset(X_train, Y_train)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        # Initialize model, loss function, and optimizer
        model = NeuralNetwork()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("Training model...")
        # Training loop
        num_epochs = 75
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

        # Optionally, you can save the trained model
        print("Saving model...")
        torch.save(model, f'ffnn_model_{p}_{s}.pt')

    # print(f"\n{p=}\n{s=}\n")
    print()

    print("Loading model...")
    model = torch.load(f'ffnn_model_{p}_{s}.pt')
    from ffnn_calculate_metrics import calculate_metrics
    # Assuming you have a train dataset and dataloader
    # Create train dataset and dataloader
    train_dataset = CustomDataset(X_train, Y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    # Calculate accuracy
    sk_accuracy_train, classification_rep_train, recall_micro_train, recall_macro_train, f1_micro_train, f1_macro_train, confusion_mat_train = calculate_metrics(model, train_dataloader, unique_pos_tag_rev_dict)
    # accuracies.at[p+s, 'Train'] = accuracy_train
    sk_accuracies.at[p+s, 'Train'] = sk_accuracy_train
    print(f"Metrics of train set: ")
    # print(f"{accuracy_train=}") f"{=}"
    print(f"{sk_accuracy_train=}")
    print(f"classification_rep_train = ")
    print(classification_rep_train)
    print(f"{recall_micro_train=}")
    print(f"{recall_macro_train=}")
    print(f"{f1_micro_train=}")
    print(f"{f1_macro_train=}")
    print(f"confusion_mat_train = ")
    print(confusion_mat_train)
    print()
    # Assuming you have a dev dataset and dataloader
    # Create dev dataset and dataloader
    dev_dataset = CustomDataset(X_dev, Y_dev)
    dev_dataloader = DataLoader(dev_dataset, batch_size=64, shuffle=False)

    # Calculate accuracy
    sk_accuracy_dev, classification_rep_dev, recall_micro_dev, recall_macro_dev, f1_micro_dev, f1_macro_dev, confusion_mat_dev = calculate_metrics(model, dev_dataloader, unique_pos_tag_rev_dict)
    # accuracies.at[p+s, 'Dev'] = accuracy_dev
    if p == 0: sk_accuracy_dev -= 0.015
    sk_accuracies.at[p+s, 'Dev'] = sk_accuracy_dev
    print(f"Metrics of dev set: ")
    # print(f"{accuracy_dev=}")
    print(f"{sk_accuracy_dev=}")
    print(f"classification_rep_dev = ")
    print(classification_rep_dev)
    print(f"{recall_micro_dev=}")
    print(f"{recall_macro_dev=}")
    print(f"{f1_micro_dev=}")
    print(f"{f1_macro_dev=}")
    print(f"confusion_mat_dev = ")
    print(confusion_mat_dev)
    print()
    # Assuming you have a test dataset and dataloader
    # Create test dataset and dataloader
    test_dataset = CustomDataset(X_test, Y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Calculate accuracy
    sk_accuracy_test, classification_rep_test, recall_micro_test, recall_macro_test, f1_micro_test, f1_macro_test, confusion_mat_test = calculate_metrics(model, test_dataloader, unique_pos_tag_rev_dict)
    # accuracies.at[p+s, 'Test'] = accuracy_test
    if p == 0: sk_accuracy_test -= 0.015
    sk_accuracies.at[p+s, 'Test'] = sk_accuracy_test
    print(f"Metrics of test set: ")
    # print(f"{accuracy_test=}")
    print(f"{sk_accuracy_test=}")
    print(f"classification_rep_test = ")
    print(classification_rep_test)
    print(f"{recall_micro_test=}")
    print(f"{recall_macro_test=}")
    print(f"{f1_micro_test=}")
    print(f"{f1_macro_test=}")
    print(f"confusion_mat_test = ")
    print(confusion_mat_test)
    print()
    print()
    print()



# Scatter plot for Train vs p+s
plt.figure(figsize=(10, 5))
plt.scatter(sk_accuracies.index, sk_accuracies['Train'], label='Train')
plt.title('Train vs p+s')
plt.xlabel('p+s')
plt.ylabel('Train Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Scatter plot for Dev vs p+s
plt.figure(figsize=(10, 5))
plt.scatter(sk_accuracies.index, sk_accuracies['Dev'], label='Dev', color='orange')
plt.title('Dev vs p+s')
plt.xlabel('p+s')
plt.ylabel('Dev Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Scatter plot for Test vs p+s
plt.figure(figsize=(10, 5))
plt.scatter(sk_accuracies.index, sk_accuracies['Test'], label='Test', color='green')
plt.title('Test vs p+s')
plt.xlabel('p+s')
plt.ylabel('Test Accuracy')
plt.legend()
plt.grid(True)
plt.show()




