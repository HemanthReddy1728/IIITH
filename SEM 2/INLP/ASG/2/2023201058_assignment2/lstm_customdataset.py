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

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, tags):
        self.sentences = sentences
        self.tags = tags

        self.device = device
        self.cutoff = 40


        self.sentences = [[token.lower() for token in sentence] for sentence in self.sentences]
        self.tags = [[token.lower() for token in sentence] for sentence in self.tags]


        self.sentences = [list(zip(sentence, tag)) for sentence, tag in zip(self.sentences, self.tags)]

        self.Ssentences = [sentence for sentence in self.sentences if self.cutoff > len(sentence) > 0]
        self.Lsentences = [sentence for sentence in self.sentences if len(sentence) >= self.cutoff]

        self.sentences = []
        for sentence in self.Lsentences:
            half = len(sentence) // 2
            self.sentences.append(sentence[:half])
            self.sentences.append(sentence[half:])
        self.sentences += self.Ssentences
        # print(self.sentences)

        self.sentences = [[("<SOS>", "<SOT>")] + sentence + [("<EOS>", "<EOT>")] for sentence in self.sentences if len(sentence) <= self.cutoff]

        # print(self.sentences)
        # print(len(self.sentences))

        self.vocab = set()
        self.tagVocab = set()

        # add PADding token
        self.vocab.add("<PAD>")
        self.tagVocab.add("<PAD>")
        self.maxSentSize = 0

        for sentence in self.sentences:
            if len(sentence) not in sentenceLens:
                sentenceLens[len(sentence)] = 1
            else:
                sentenceLens[len(sentence)] += 1

            for token in sentence:
                self.vocab.add(token[0])
                self.tagVocab.add(token[1])

            if len(sentence) > self.maxSentSize:
                self.maxSentSize = len(sentence)

        self.vocab = list(self.vocab)
        self.tagVocab = list(self.tagVocab)

        # add OOV tag
        if "<OOV>" not in self.vocab:
            self.vocab.append("<OOV>")

        if "<OOV>" not in self.tagVocab:
            self.tagVocab.append("<OOV>")

        if "<SOS>" not in self.vocab:
            self.vocab.append("<SOS>")

        if "<SOT>" not in self.tagVocab:
            self.tagVocab.append("<SOT>")

        if "<EOS>" not in self.vocab:
            self.vocab.append("<EOS>")

        if "<EOT>" not in self.tagVocab:
            self.tagVocab.append("<EOT>")

        self.vocabSet = set(self.vocab)
        self.Word2Index = {w: i for i, w in enumerate(self.vocab)}
        self.index2Word = {i: w for i, w in enumerate(self.vocab)}

        # print(self.index2Word)

        self.tagVocabSet = set(self.tagVocab)
        self.tagWord2Index = {w: i for i, w in enumerate(self.tagVocab)}
        self.tagIndex2Word = {i: w for i, w in enumerate(self.tagVocab)}

        # PAD each sentence to 40
        for i in range(len(self.sentences)):
            # print(type(self.sentences[i]))
            self.sentences[i] = self.sentences[i] + [("<PAD>", "<PAD>")] * (self.maxSentSize - len(self.sentences[i]))

        self.sentencesIndex = torch.tensor([[self.Word2Index[token[0]] for token in sentence] for sentence in self.sentences], device=self.device)
        self.tagSentencesIndex = torch.tensor([[self.tagWord2Index[token[1]] for token in sentence] for sentence in self.sentences], device=self.device)
        self.PADIndex = self.Word2Index["<PAD>"]
        self.tagPadIndex = self.tagWord2Index["<PAD>"]

        self.SOSIndex = self.Word2Index["<SOS>"]
        self.SOTIndex = self.tagWord2Index["<SOT>"]

        self.EOSIndex = self.Word2Index["<EOS>"]
        self.EOTIndex = self.tagWord2Index["<EOT>"]

    def __len__(self):
        return len(self.sentencesIndex)

    def __getitem__(self, index):
        # sentence, last word
        return self.sentencesIndex[index], self.tagSentencesIndex[index]

    def LowFrequencyWordRemover(self, threshold):
        # remove words with frequency less than threshold
        freq = {}
        for sentence in self.sentences:
            for token in sentence:
                if token[0] not in freq:
                    freq[token[0]] = 1
                else:
                    freq[token[0]] += 1

        for i in range(len(self.sentences)):
            for j in range(len(self.sentences[i])):
                if freq[self.sentences[i][j][0]] <= threshold:
                    t = list(self.sentences[i][j])
                    if t[0] in self.vocab:
                        self.vocab.remove(t[0])
                    if t[0] in self.vocabSet:
                        self.vocabSet.remove(t[0])
                    t[0] = "<OOV>"
                    self.sentences[i][j] = tuple(t)

        self.Word2Index = {w: i for i, w in enumerate(self.vocab)}
        self.index2Word = {i: w for i, w in enumerate(self.vocab)}
        self.sentencesIndex = torch.tensor([[self.Word2Index[token[0]] for token in sentence] for sentence in self.sentences], device=self.device)

        self.PADIndex = self.Word2Index["<PAD>"]
        self.tagPadIndex = self.tagWord2Index["<PAD>"]

        self.SOSIndex = self.Word2Index["<SOS>"]
        self.SOTIndex = self.tagWord2Index["<SOT>"]

        self.EOSIndex = self.Word2Index["<EOS>"]
        self.EOTIndex = self.tagWord2Index["<EOT>"]


    def OOV_Handler(self, vocab_set, vocab, tagVocab_set, tagVocab):
        for i in range(len(self.sentences)):
            for j in range(len(self.sentences[i])):
                if self.sentences[i][j][0] not in vocab_set:
                    # remove from vocab and vocab set
                    if self.sentences[i][j][0] in self.vocab:
                        self.vocab.remove(self.sentences[i][j][0])
                    if self.sentences[i][j][0] in self.vocabSet:
                        self.vocabSet.remove(self.sentences[i][j][0])
                    t = list(self.sentences[i][j])
                    t[0] = "<OOV>"
                    self.sentences[i][j] = tuple(t)
        self.Word2Index = {w: i for i, w in enumerate(vocab)}
        self.index2Word = {i: w for i, w in enumerate(vocab)}
        self.sentencesIndex = torch.tensor([[self.Word2Index[token[0]] for token in sentence] for sentence in self.sentences], device=self.device)

        for i in range(len(self.sentences)):
            for j in range(len(self.sentences[i])):
                if self.sentences[i][j][1] not in tagVocab_set:
                    # remove from vocab and vocab set
                    if self.sentences[i][j][1] in self.tagVocab:
                        self.tagVocab.remove(self.sentences[i][j][1])
                    if self.sentences[i][j][1] in self.tagVocabSet:
                        self.tagVocabSet.remove(self.sentences[i][j][1])
                    t = list(self.sentences[i][j])
                    t[1] = "<OOV>"
                    self.sentences[i][j] = tuple(t)

        self.tagWord2Index = {w: i for i, w in enumerate(tagVocab)}
        self.tagIndex2Word = {i: w for i, w in enumerate(tagVocab)}
        self.tagSentencesIndex = torch.tensor([[self.tagWord2Index[token[1]] for token in sentence] for sentence in self.sentences], device=self.device)

        self.PADIndex = self.Word2Index["<PAD>"]
        self.tagPadIndex = self.tagWord2Index["<PAD>"]

        self.SOSIndex = self.Word2Index["<SOS>"]
        self.SOTIndex = self.tagWord2Index["<SOT>"]

        self.EOSIndex = self.Word2Index["<EOS>"]
        self.EOTIndex = self.tagWord2Index["<EOT>"]