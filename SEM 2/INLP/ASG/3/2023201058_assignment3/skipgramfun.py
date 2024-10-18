import math, nltk, random, html, re, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from itertools import chain
# from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
# from torch.nn.functional import sigmoid, logsigmoid
# import torch.nn.functional as F
# from torch.autograd import Variable
# from torchtext.data import get_tokenizer
# from torchtext.vocab import Vocab, build_vocab_from_iterator
# from torch.nn.utils import clip_grad_norm_

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score, confusion_matrix
# from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def clean_text(text):
    text = re.sub(r"([a-zA-Z]+)n[\'’]t", r"\1 not", text)
    text = re.sub(r"([iI])[\'’]m", r"\1 am", text)
    text = re.sub(r"([iI])[\'’]ll", r"\1 will", text)
    text = re.sub(r"[^a-zA-Z0-9\:\$\-\,\%\.\?\!]+", " ", text)
    text = html.unescape(text)
    text = re.sub(r"_(.*?)_", r"\1", text)
    text = re.sub(r'(\w+)-(\w+)', r'\1\2', text)
    return text


# def tokenize_text(text, symbols, lemmatizer):
def tokenize_text(text, stop_words, lemmatizer):
    tokens = word_tokenize(text)
    # tokens = [token.lower() for token in tokens]
    tokens = [token.lower() for token in tokens if token.isalpha()]  # Remove non-alphabetic tokens
    tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    # tokens = [token for token in tokens if token not in symbols]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize tokens
    return tokens


def tokenize_text2(text, word_to_index, stop_words, lemmatizer):
    tokens = word_tokenize(text)
    # tokens = [token.lower() for token in tokens]
    tokens = [token.lower() for token in tokens if token.isalpha()]  # Remove non-alphabetic tokens
    tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    tokens = [token if token in word_to_index else 'unk' for token in tokens]
    # Remove stopwords
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize tokens
    return tokens



# flatten = lambda l: [item for sublist in l for item in sublist]
def flatten(nested_list):
    return list(chain.from_iterable(nested_list))

def getBatch(batch_size, train_data):
    # random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex: eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch

    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch

# def prepare_sequence(seq, word2index):
    # idxs = list(map(lambda w: word2index[w] if word2index.get(w) is not None else word2index['unk'], seq))
    # return Variable(LongTensor(idxs))

# def prepare_word(word, word2index):
    # return Variable(LongTensor([word2index[word]]) if word2index.get(word) is not None else LongTensor([word2index['unk']]))


# def prepare_word(word, word2index):
    # if word in word2index:
    #     idx = word2index[word]
    # else:
    #     idx = word2index['unk']
    # return torch.tensor([idx], dtype=torch.long, device='cuda' if torch.cuda.is_available() else 'cpu')
    # idx = [word2index[word] if word in word2index else word2index['unk']]
    # return torch.tensor(idx, dtype=torch.long, device='cuda' if torch.cuda.is_available() else 'cpu')

def prepare_sequence(seq, word2index):
    idxs = [word2index[w] if w in word2index else word2index['unk'] for w in seq]
    return torch.tensor(idxs, dtype=torch.double) #, device='cuda' if torch.cuda.is_available() else 'cpu')


def negative_sampling(targets, unigram_table, NEGSAMP, word2index):
    batch_size = targets.size(0)
    neg_samples = []
    for i in range(batch_size):
        nsample = []
        # target_index = targets[i].data.tolist()[0]
        target_index = targets[i].data.cpu().tolist()[0] if torch.cuda.is_available() else targets[i].data.tolist()[0]
        while len(nsample) < NEGSAMP: # num of sampling
            neg = random.choice(unigram_table)
            if word2index[neg] == target_index:
                continue
            nsample.append(neg)
        neg_samples.append(prepare_sequence(nsample, word2index).view(1, -1))
    return torch.cat(neg_samples)




class SkipgramNegSampling(nn.Module):
    def __init__(self, vocab_size, projection_dim):
        super(SkipgramNegSampling, self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, projection_dim) # center embedding
        self.embedding_u = nn.Embedding(vocab_size, projection_dim) # out embedding
        self.logsigmoid = nn.LogSigmoid()
                
        initrange = np.sqrt(2 / (vocab_size + projection_dim)) # Xavier init
        self.embedding_v.weight.data.uniform_(-initrange, initrange) # init
        # self.embedding_u.weight.data.uniform_(-0.0, 0.0) # init
        self.embedding_u.weight.data.uniform_(-initrange, initrange) # init
        
    def forward(self, center_words, target_words, negative_words):
        center_embeds = self.embedding_v(center_words) # B x 1 x D
        target_embeds = self.embedding_u(target_words) # B x 1 x D
        negative_embeds = self.embedding_u(negative_words) # B x K x D
        
        positive_score = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2) # Bx1
        # negative_score = torch.sum(negative_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2), 1).view(negs.size(0), -1) # BxK -> Bx1
        negative_scores = negative_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2) # BxK -> Bx1

        loss = self.logsigmoid(positive_score) + torch.sum(self.logsigmoid(-negative_scores).view(negative_words.size(0), -1), 1)
        return -torch.mean(loss)
    
    def prediction(self, inputs):
        embeds = self.embedding_v(inputs)
        return embeds
    


class SentenceDataset(Dataset):
    def __init__(self, sentences, labels, word2index, word_embeddings):
        self.sentences = sentences
        self.labels = labels
        self.word2index = word2index
        self.word_embeddings = word_embeddings

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx, word2index, word_embeddings):
        sentence = self.sentences[idx]
        # print(text)
        label = self.labels[idx]
        # Convert text to embedding
        embedding = torch.tensor(np.array([self.word_embeddings[word] if word in word2index else word_embeddings['unk'] for word in sentence]), dtype=torch.double)
        # tokens = [token if token in word_to_index else 'unk' for token in tokens]
        return embedding, label
    

def collate_fn(batch):
    inputs, labels = zip(*batch)  # Unzip the batch into inputs and labels
    inputs = pad_sequence(inputs, batch_first=True)  # Pad the sequences
    labels = torch.tensor(labels, dtype=torch.double)  # Convert labels to a tensor
    return inputs, labels


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Assuming x is a padded sequence of embeddings
        lstm_out, (ht, ct) = self.lstm(x)
        return self.fc(ht[-1])



def AllSkLMetrics(true_labels, predicted_labels, DatasetType, WINDOW_SIZE, array):
    accuracy = accuracy_score(true_labels, predicted_labels)
    array.append(accuracy)
    # test_accuracies.append(test_accuracy)
    print(f'Accuracy on the {DatasetType} set: {accuracy} for window_size = {WINDOW_SIZE}')

    # Additional Metrics
    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels))

    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, predicted_labels))

    # Micro and Macro Recall
    micro_recall = recall_score(true_labels, predicted_labels, average='micro')
    macro_recall = recall_score(true_labels, predicted_labels, average='macro')
    print(f'Micro Recall: {micro_recall}')
    print(f'Macro Recall: {macro_recall}')

    # Micro and Macro F1 Score
    micro_f1 = f1_score(true_labels, predicted_labels, average='micro')
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro')
    print(f'Micro F1 Score: {micro_f1}')
    print(f'Macro F1 Score: {macro_f1}')
    print()
    print()
    print()