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
from ffnn_neuralnetwork import NeuralNetwork
from lstm_lstmnn import LSTM

device = "cuda" if torch.cuda.is_available() else "cpu"
sentenceLens = {}

# def main():
if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ['-f', '-r']:
        print("Usage: python pos_tagger.py {-f | -r}")
        sys.exit(1)

    tagger_option = sys.argv[1]

    # tagger_option = str(input())

    if tagger_option == '-f':
        # from ffnn_neuralnetwork import NeuralNetwork
        sentenceLens = {}
    
        p = s = 1
        if os.path.exists("ffnn_model_1_1.pt"):
            window_size = p+1+s
            start_tag = "<sos>"
            end_tag = "<eos>"

            print("Loading model...")
            model = torch.load("ffnn_model_1_1.pt")
            model.eval()

            with open(f"unique_pos_tag_rev_dict_{p}_{s}.json", "r") as file:
                unique_pos_tag_rev_dict = json.load(file)
            # unique_pos_tag_rev_dict = {"0": "VERB", "1": "INTJ", "2": "PROPN", "3": "AUX", "4": "DET", "5": "OOV", "6": "PRON", "7": "ADP", "8": "NUM", "9": "PART", "10": "CCONJ", "11": "ADJ", "12": "ADV", "13": "NOUN"}
            print(unique_pos_tag_rev_dict)
            with open(f'train_word_embeddings_{p}_{s}.pkl', 'rb') as f:
                train_word_embeddings = pickle.load(f)


            while True:
                # Y_sent = []
                # train_word_embeddings = all_train_word_embeddings[1]
                sent = str(input("input sentence: "))
                if sent == "0":
                    break
                one_sent_list = sent.lower().split()
                # orig = sent.copy()
                # for test_sentence in parsed_test_data:
                mod_one_sent = []
                for i in range(p):
                    mod_one_sent.append(start_tag)
                for one_token in one_sent_list:
                    # print(test_token["form"], test_token["upostag"], test_token["deprel"])
                    mod_one_sent.append(one_token.lower())
                    # one_data_word_pos_dict[one_token["form"].lower()] = one_token["upostag"]
                for j in range(s):
                    mod_one_sent.append(end_tag)
                # modified_one_sentences_list.append(mod_one_sent)
                # print(mod_one_sent)



                # modified_sentences_list = modified_train_sentences_list + modified_dev_sentences_list + modified_test_sentences_list
                # one_sent_model = Word2Vec([mod_one_sent], min_count=1)
                # train_model = Word2Vec(modified_train_sentences_list, min_count=1)
                # one_sent_word_embeddings = {word: one_sent_model.wv[word] for word in one_sent_model.wv.index_to_key}

                X_sent = []
                for k in range(len(mod_one_sent)-s-p):
                    embed_window = []
                    for l in range(k, window_size+k):
                        # print(l) one_sent_model.vector_size
                        embed_window.append(train_word_embeddings[mod_one_sent[l]] if mod_one_sent[l] in train_word_embeddings else np.zeros(100))
                    X_sent.append(embed_window)
                    # print(one_train_sent_list[k+p])
                    # Y_sent.append(unique_pos_tag_dict[dev_data_word_pos_dict[one_dev_sent_list[k+p]]])
                # for i in range(len(sent)):
                #     sent[i] = sent[i].lower()
                X_sent = np.array(X_sent)
                # Y_test = np.array(Y_test)
                # print(X_sent.shape)

                # print(str(one_sent_list[1]))
                # print(Y_test.shape)
                # print(len(embed_window))

                # when is the flight to denver

                with torch.no_grad():
                    for i in range(len(one_sent_list)):
                        # print(train_word_embeddings[one_sent_list[i]])
                    # print(torch.max(torch.nn.functional.softmax(model(torch.tensor(X_sent)), dim=1),1))
                        output = model(torch.Tensor(X_sent[i]).unsqueeze(0))
                        # print(f"{output=}")
                        mod_output = str(torch.argmax(output, dim=1).item())
                        # print(f"{mod_output=}")
                        print(f"{one_sent_list[i]}\t{unique_pos_tag_rev_dict[mod_output]}")

    elif tagger_option == '-r':
        sentenceLens = {}
            
        if os.path.exists("lstm_model_1.pt"):
            print("Loading model...")
            model = torch.load("lstm_model_1.pt")
            while True:
                sent = str(input("input sentence: ")).split()
                # print(sent)
                if sent == ['0']:
                    break
                orig = sent.copy()

                for i in range(len(sent)):
                    sent[i] = sent[i].lower()

                # convert to index f"{=}"
                for i in range(len(sent)):
                    if sent[i] in model.train_data.vocab:
                        sent[i] = model.train_data.Word2Index[sent[i]]
                    else:
                        sent[i] = model.train_data.Word2Index["<OOV>"]

                sent = torch.tensor(sent).to(model.device)
                # print(f"{sent=}")
                output = model(sent)
                # print(f"{output=}")
                # softmax and output tag
                mod_output = torch.nn.functional.softmax(output, dim=1)
                # print(f"{mod_output=}")
                something, predicted = torch.max(mod_output, 1)
                # print(f"{something=}")
                # print(f"{predicted=}")

                for i in range(len(sent)):
                    print(f"{orig[i]}\t{model.train_data.tagIndex2Word[predicted[i].item()].upper()}")
                # when is the flight to denver



# if __name__ == "__main__":
    # main()