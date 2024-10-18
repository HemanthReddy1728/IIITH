import numpy as np
import pandas as pd
import re, html, subprocess
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score, confusion_matrix

# from torchtext.data import get_tokenizer
# from torchtext.vocab import Vocab, build_vocab_from_iterator
# from torchtext.data.utils import get_tokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    text = re.sub(r"([a-zA-Z]+)n[\'’]t", r"\1 not", text)
    text = re.sub(r"([iI])[\'’]m", r"\1 am", text)
    text = re.sub(r"([iI])[\'’]ll", r"\1 will", text)
    text = re.sub(r"[^a-zA-Z0-9\:\$\-\,\%\.\?\!]+", " ", text)
    text = html.unescape(text)
    text = re.sub(r"_(.*?)_", r"\1", text)
    text = re.sub(r'(\w+)-(\w+)', r'\1\2', text)
    return text

# def tokenize_text(text, stop_words, lemmatizer):
def tokenize_text(text, symbols, lemmatizer):
    tokens = word_tokenize(text)
    # tokens = [token.lower() for token in tokens]
    tokens = [token.lower() for token in tokens if token.isalpha()]  # Remove non-alphabetic tokens
    # tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    tokens = [token for token in tokens if token not in symbols]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize tokens
    return tokens


def tokenize_text2(text, word_to_index, symbols, lemmatizer):
    tokens = word_tokenize(text)
    # tokens = [token.lower() for token in tokens]
    tokens = [token.lower() for token in tokens if token.isalpha()]  # Remove non-alphabetic tokens
    # tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    tokens = [token for token in tokens if token not in symbols]
    # Remove stopwords
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize tokens
    tokens = [token if token in word_to_index else 'unk' for token in tokens]
    return tokens


class SentenceDataset(Dataset):
    def __init__(self, sentences, labels, word_to_index, word_embeddings):
        self.sentences = sentences
        self.labels = labels
        self.word_to_index = word_to_index
        self.word_embeddings = word_embeddings

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx, word_to_index, word_embeddings):
        sentence = self.sentences[idx]
        # print(text)
        label = self.labels[idx]
        # Convert text to embedding
        embedding = torch.tensor(np.array([self.word_embeddings[word] if word in word_to_index else word_embeddings['unk'] for word in sentence]), dtype=torch.float)
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