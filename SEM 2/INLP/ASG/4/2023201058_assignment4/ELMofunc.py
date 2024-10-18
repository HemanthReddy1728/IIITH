import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
# from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from ELMo import dimension, device
# dimension = 100
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import string
import nltk, re, html, subprocess
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from IPython.display import FileLink
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score, confusion_matrix

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
    tokens = [token.lower() for token in tokens if token.isalpha()]  # Remove non-alphabetic tokens
    tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    # tokens = [token for token in tokens if token not in symbols]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize tokens
    # tokens = [token.lower() for token in tokens]
    return tokens


def create_vocab(tokenized_dataset, freq_threshold=2):
    # Create a dictionary to hold the word frequency counts
    word_counts = {}

    # Count the frequency of each word in the dataset
    for example in tokenized_dataset:
        for word in example['text']:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1

    # Create a vocabulary of words that occur at least freq_threshold times
    vocab = {'<PAD>': 0, '<UNK>': 1}
    index = 2
    for word, count in word_counts.items():
        if count >= freq_threshold:
            vocab[word] = index
            index += 1

    return vocab


def token2index_dataset(tokenized_dataset, vocab):
    # Convert each text to a list of indices using the vocabulary
    indexed_dataset = []
    for example in tokenized_dataset:
        indexed_text = [vocab.get(word, 1) for word in example['text']]
        indexed_dataset.append({'text': indexed_text, 'label': example['label']})

    return indexed_dataset


def pad_texts(texts, max_length, pad_index=0):
    padded_texts = []
    for text in texts:
        if len(text) >= max_length:
            padded_texts.append(text[:max_length])
        else:
            num_padding = max_length - len(text)
            new_text = text + [pad_index] * num_padding
            padded_texts.append(new_text)
    return padded_texts


class DatasetConstructor(Dataset):
    def __init__(self, data, labels):
        self.data = data
        # self.labels = [0 if label < 0.5 else 1 for label in labels]
        self.labels = labels
        # self.backward_data = [text[:-1] for text in self.data]
        # self.forward_data = [text[1:] for text in self.data]
        self.forward_data = [text for text in self.data]
        self.backward_data = [text[::-1] for text in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index]) ,torch.tensor(self.forward_data[index]), torch.tensor(self.backward_data[index]), torch.tensor(self.labels[index])
    

class ELMo(nn.Module):
    def __init__(self, vocab_size, weights_matrix, hidden_size=dimension, num_layers=2, dropout=0.36):
        super(ELMo, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, dimension)
        self.embedding.weight.data.copy_(torch.FloatTensor(weights_matrix))
        self.embedding.weight.requires_grad = True

        # LSTM layers
        self.lstm_forward = nn.LSTM(dimension, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.lstm_backward = nn.LSTM(dimension, hidden_size, num_layers, dropout=dropout, batch_first=True)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x_forward, x_backward):
        # Embedding layer
        x_forward = self.embedding(x_forward)
        x_backward = self.embedding(x_backward)

        # print("1 ", x_backward.shape)

        # LSTM layers
        out_forward, _ = self.lstm_forward(x_backward)
        # print("2 ", out_forward.shape)
        out_backward, _ = self.lstm_backward(out_forward)
        # print("3 ", out_backward.shape)

        out = self.fc(out_backward)

        return out
    

# def accuracy(outputs, labels):
#     _, preds = torch.max(outputs, dim=1)
#     return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def train_loop_EE(model, criterion, optimizer, train_loader, valid_loader, num_epochs, vocab, BVA, BTA):
    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_valid_acc = BVA  # Variable to store the best validation accuracy
    best_train_acc = BTA
    best_model_path = 'best_bilstm.pt'  # Path to save the best model

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_total_accuracy = 0.0, 0.0

        for x, x_forward, x_backward, labels in tqdm(train_loader):
            x_forward, x_backward = x_forward.to(device), x_backward.to(device)
            optimizer.zero_grad()
            outputs = model(x_forward, x_backward).view(-1, len(vocab))
            target = x_forward.view(-1).to(device)
            loss = criterion(outputs, target)
            # acc = accuracy(outputs, target)
            _, preds = torch.max(outputs, dim=1)
            acc = torch.tensor(torch.sum(preds == target).item() / len(preds))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_total_accuracy += acc.item()

        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_total_accuracy / len(train_loader))

        model.eval()
        valid_loss, valid_total_accuracy = 0.0, 0.0
        with torch.no_grad():
            for x, x_forward, x_backward, labels in tqdm(valid_loader):
                x_forward, x_backward = x_forward.to(device), x_backward.to(device)
                outputs = model(x_forward, x_backward).view(-1, len(vocab))
                target = x_forward.view(-1).to(device)
                loss = criterion(outputs, target)
                # acc = accuracy(outputs, target)
                _, preds = torch.max(outputs, dim=1)
                acc = torch.tensor(torch.sum(preds == target).item() / len(preds))
                valid_loss += loss.item()
                valid_total_accuracy += acc.item()

        valid_losses.append(valid_loss / len(valid_loader))
        valid_accuracies.append(valid_total_accuracy / len(valid_loader))

        
        # Check if the current validation accuracy is the best, save the model if it is
        if valid_accuracies[-1] >= best_valid_acc:
            if valid_accuracies[-1] > best_valid_acc:
                best_valid_acc = valid_accuracies[-1]
                torch.save(model.state_dict(), best_model_path)
                # print(f"Saved new best model with validation accuracy: {best_valid_acc}")
                # Print training and validation results
                print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_losses[-1]}, Validation Loss: {valid_losses[-1]}, Training Accuracy: {train_accuracies[-1]}, Validation Accuracy: {valid_accuracies[-1]} (Saved new best model)')
            elif train_accuracies[-1] > best_train_acc:
                best_train_acc = train_accuracies[-1]
                torch.save(model.state_dict(), best_model_path)
                # print(f"Saved new best model with validation accuracy: {best_valid_acc}")
                # Print training and validation results
                print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_losses[-1]}, Validation Loss: {valid_losses[-1]}, Training Accuracy: {train_accuracies[-1]}, Validation Accuracy: {valid_accuracies[-1]} (Saved new best model)')                
            else:
                # Print training and validation results
                print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_losses[-1]}, Validation Loss: {valid_losses[-1]}, Training Accuracy: {train_accuracies[-1]}, Validation Accuracy: {valid_accuracies[-1]}')
        else:
            # Print training and validation results
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_losses[-1]}, Validation Loss: {valid_losses[-1]}, Training Accuracy: {train_accuracies[-1]}, Validation Accuracy: {valid_accuracies[-1]}')


    return train_losses, valid_losses, train_accuracies, valid_accuracies, best_valid_acc, best_train_acc


def PreTrainingTask(elmo_model, criterion, optimizer, train_loader, valid_loader, num_epochs, vocab, BVA1, BTA1):
    # Train the ELMo model
    train_losses, valid_losses, train_accuracies, valid_accuracies, BVA1, BTA1 = train_loop_EE(elmo_model, criterion, optimizer, train_loader, valid_loader, num_epochs, vocab, BVA1, BTA1)
    print()

    # Plot the training and validation loss
    plt.plot(train_losses, label='Training loss')
    plt.plot(valid_losses, label='Validation loss')
    plt.legend()
    plt.show()

    # Plot the training and validation accuracies
    plt.plot(train_accuracies, label='Training accuracy')
    plt.plot(valid_accuracies, label='Validation accuracy')
    plt.legend()
    plt.show()

    # save the model
    # torch.save(elmo_model.state_dict(), 'elmo_model.pt')

    # elmo_model.load_state_dict(torch.load('elmo_model1.pt'))
    elmo_model.load_state_dict(torch.load('best_bilstm.pt'))#, map_location=torch.device('cpu')))

    
    return elmo_model, BVA1, BTA1


from ELMo import elmo_lstmb, elmo_lstmf

# AG_News Analysis Model
class AG_News_Analysis(nn.Module):
    def __init__(self, vocab_size, elmo_embeddings, embedding_dim=dimension, hidden_size=dimension, num_layers=2, dropout=0.36, output_dim=4):
        super(AG_News_Analysis, self).__init__()
        # Embedding layer
        # self.embedding = nn.Embedding.from_pretrained(torch.tensor(elmo_embeddings))
        self.embedding = nn.Embedding.from_pretrained(elmo_embeddings.clone().detach())
        self.embedding.weight.requires_grad = True
        self.weightage = nn.Parameter(torch.FloatTensor([0.333, 0.333, 0.333]))

        # LSTM layers
        self.lstm1 = elmo_lstmf
        self.lstm2 = elmo_lstmb

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Fully connected layer
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # Embedding layer
        x = self.embedding(x)
        embed1 = self.fc1(x)
        embed2, _ = self.lstm1(x)
        embed3, _ = self.lstm2(embed2)
        embed = torch.stack([embed1, embed2, embed3], dim=1)
        embed = torch.sum(embed * self.weightage.view(1, -1, 1, 1), dim=1)  # perform weighted sum along dim=1

        # take max and dropout
        out = torch.max(embed, dim=1)[0]
        out = self.dropout(out)

        # Fully connected layer
        out = self.fc2(out)

        return out
    
    
# 4.1 Trainable λs
class AG_News_Analysis_Flow(nn.Module):
    def __init__(self, vocab_size, elmo_embeddings, embedding_dim=dimension, hidden_size=dimension, num_layers=2, dropout=0.36, output_dim=4):
        super(AG_News_Analysis_Flow, self).__init__()
        # self.embedding = nn.Embedding.from_pretrained(torch.tensor(elmo_embeddings), freeze=False)
        # self.embedding = nn.Embedding.from_pretrained(torch.tensor(elmo_embeddings))
        self.embedding = nn.Embedding.from_pretrained(elmo_embeddings.clone().detach())
        self.lstm1 = elmo_lstmf
        self.lstm2 = elmo_lstmb
        self.dropout = nn.Dropout(p=dropout)
        # Fully connected layer
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)

        # λ weights, initialized and trainable
        self.weightage = nn.Parameter(torch.rand(3))

    def forward(self, x):
        x = self.embedding(x)
        e1 = self.fc1(x)
        e2, _ = self.lstm1(x)
        e3, _ = self.lstm2(e2)
        E_hat = torch.stack([e1, e2, e3], dim=1)
        # E_hat = torch.sum(E_hat * self.weightage.unsqueeze(0).unsqueeze(-1), dim=1)
        weights_normalized = F.softmax(self.weightage, dim=0)
        E_hat = torch.sum(E_hat * weights_normalized.view(1, -1, 1, 1), dim=1)  # perform weighted sum along dim=1
        out = torch.max(E_hat, dim=1)[0]
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# 4.2 Frozen λs
class AG_News_Analysis_Froz(nn.Module):
    def __init__(self, vocab_size, elmo_embeddings, embedding_dim=dimension, hidden_size=dimension, num_layers=2, dropout=0.36, output_dim=4):
        super(AG_News_Analysis_Froz, self).__init__()
        # self.embedding = nn.Embedding.from_pretrained(torch.tensor(elmo_embeddings), freeze=False)
        # self.embedding = nn.Embedding.from_pretrained(torch.tensor(elmo_embeddings))
        self.embedding = nn.Embedding.from_pretrained(elmo_embeddings.clone().detach())
        self.lstm1 = elmo_lstmf
        self.lstm2 = elmo_lstmb
        self.dropout = nn.Dropout(p=dropout)
        # Fully connected layer
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)

        # λ weights, initialized but not trainable
        self.weightage = torch.rand(3, requires_grad=False).to(device)

    def forward(self, x):
        x = self.embedding(x)
        e1 = self.fc1(x)
        e2, _ = self.lstm1(x)
        e3, _ = self.lstm2(e2)
        E_hat = torch.stack([e1, e2, e3], dim=1).to(device)
        # E_hat = torch.sum(E_hat * self.weightage.unsqueeze(0).unsqueeze(-1), dim=1)
        weights_normalized = F.softmax(self.weightage, dim=0)
        E_hat = torch.sum(E_hat * weights_normalized.view(1, -1, 1, 1), dim=1)  # perform weighted sum along dim=1
        out = torch.max(E_hat, dim=1)[0]
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# 4.3 Learnable Function
class AG_News_Analysis_LNNF(nn.Module):
    def __init__(self, vocab_size, elmo_embeddings, embedding_dim=dimension, hidden_size=dimension, num_layers=2, dropout=0.36, output_dim=4):
        super(AG_News_Analysis_LNNF, self).__init__()
        # self.embedding = nn.Embedding.from_pretrained(torch.tensor(elmo_embeddings), freeze=False)
        # self.embedding = nn.Embedding.from_pretrained(torch.tensor(elmo_embeddings))
        self.embedding = nn.Embedding.from_pretrained(elmo_embeddings.clone().detach())
        self.lstm1 = elmo_lstmf
        self.lstm2 = elmo_lstmb
        self.dropout = nn.Dropout(p=dropout)
        # Fully connected layer
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)

        # Learnable function
        self.combine = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),  # Combining 3 layers into 1
            # nn.ReLU(),
            # nn.Softmax(dim=-1))
            # nn.Linear(hidden_size, hidden_size),  # Additional layer to reshape if necessary
            nn.Softmax(dim=1)  # Ensuring softmax normalization on the combined output            
        )
        
    def forward(self, x):
        x = self.embedding(x)
        e1 = self.fc1(x)
        e2, _ = self.lstm1(x)
        e3, _ = self.lstm2(e2)
        # E_hat = torch.cat([e1, e2, e3], dim=1)
        E_hat = torch.cat([e1, e2, e3], dim=2)
        E_hat = self.combine(E_hat)
        # E_hat2 = torch.stack([e1, e2, e3], dim=1).to(device)
        # print(E_hat.shape, comb.shape)
        # E_hat = torch.sum(E_hat2 * nn.Parameter(comb).view(1, -1, 1, 1), dim=1)  # perform weighted sum along dim=1
        out = torch.max(E_hat, dim=1)[0]
        out = self.dropout(out)
        out = self.fc2(out)
        return out


# TRAIN LOOP
# Define the training loop
def train_loop_AGNA(model, criterion, optimizer, train_loader, valid_loader, num_epochs, modelName, BVA, BTA):
    train_losses = []
    valid_losses = []
    train_acc = []
    valid_acc = []
    max_valid_acc = BVA  # Initialize the max validation accuracy
    max_train_acc = BTA
    for epoch in range(num_epochs):
        # Train the model
        model.train()
        train_loss = 0.0
        train_correct = 0
        for x, x_forward, x_backward, labels in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(x.to(device)).to(device)
            labels = labels.to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (torch.max(outputs, 1)[1] == labels).sum().item()
    
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_acc_percentage = train_correct / len(train_loader)
        train_acc.append(train_acc_percentage)

        # Evaluate the model
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        with torch.no_grad():
            for x, x_forward, x_backward, labels in tqdm(valid_loader):
                outputs = model(x.to(device)).to(device)
                labels = labels.to(device)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                valid_correct += (torch.max(outputs, 1)[1] == labels).sum().item()
        
        valid_loss /= len(valid_loader)
        valid_losses.append(valid_loss)
        valid_acc_percentage = valid_correct / len(valid_loader)
        valid_acc.append(valid_acc_percentage)

        # print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}, Validation Loss: {valid_loss}, Training Accuracy: {train_acc_percentage}, Validation Accuracy: {valid_acc_percentage}')
        
        # Check if the current validation accuracy is the best we've seen so far
        if valid_acc[-1] >= max_valid_acc:
            if valid_acc[-1] > max_valid_acc:
                max_valid_acc = valid_acc[-1]
                # Save model
                torch.save(model.state_dict(), f'best_{modelName}.pt')
                # print(f"Saved new best model with Validation Accuracy: {max_valid_acc}")
                print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}, Validation Loss: {valid_loss}, Training Accuracy: {train_acc_percentage}, Validation Accuracy: {valid_acc_percentage} (Saved new best model)')
            elif train_acc[-1] > max_train_acc:
                max_train_acc = train_acc[-1]
                # Save model
                torch.save(model.state_dict(), f'best_{modelName}.pt')
                # print(f"Saved new best model with Validation Accuracy: {max_valid_acc}")
                print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}, Validation Loss: {valid_loss}, Training Accuracy: {train_acc_percentage}, Validation Accuracy: {valid_acc_percentage} (Saved new best model)')            
            else:
                print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}, Validation Loss: {valid_loss}, Training Accuracy: {train_acc_percentage}, Validation Accuracy: {valid_acc_percentage}')
        else:
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}, Validation Loss: {valid_loss}, Training Accuracy: {train_acc_percentage}, Validation Accuracy: {valid_acc_percentage}')
        
    return train_losses, valid_losses, train_acc, valid_acc, max_valid_acc, max_train_acc


def AllSkLMetrics(model, DatasetType, data_loader):
    correct_pred, num_examples = 0, 0
    true_labels = []
    predicted_labels = []
    # test_accuracies = []
    model.eval()
    for x, _, _, labels in tqdm(data_loader):
        outputs = model(x.to(device)).to(device)
        labels = labels.to(device)
        # predicted_labels = torch.argmax(outputs, 1)
        _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())
        num_examples += labels.size(0)
        correct_pred += (predicted == labels).sum()
    test_accuracy = correct_pred.float()/num_examples
    accuracy = accuracy_score(true_labels, predicted_labels)
    # array.append(accuracy)
    # test_accuracies.append(test_accuracy)
    print(f'Accuracy on the {DatasetType} set: {accuracy}')#' for window_size = {WINDOW_SIZE}')

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

# plot accuracy
# def accuracy():
#     correct_pred, num_examples = 0, 0
#     true_labels = []
#     predicted_labels = []
# #     test_accuracies = []
#     model.eval()
#     for x, _, _, labels in tqdm(data_loader):
#         outputs = model(x.to(device)).to(device)
#         labels = labels.to(device)
# #         predicted_labels = torch.argmax(outputs, 1)
#         _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
#         true_labels.extend(labels.cpu().numpy())
#         predicted_labels.extend(predicted.cpu().numpy())
#         num_examples += labels.size(0)
#         correct_pred += (predicted == labels).sum()
#     test_accuracy = correct_pred.float()/num_examples

def DownStreamTask(model, criterion, optimizer, train_loader, valid_loader, test_loader, num_epochs, modelName, BVA2, BTA2):
    # Train the AG_News Analysis model
    train_losses, valid_losses, train_acc, valid_acc, BVA2, BTA2 = train_loop_AGNA(model, criterion, optimizer, train_loader, valid_loader, num_epochs, modelName, BVA2, BTA2)
    print()
    # Plot the training and validation loss
    plt.plot(train_losses, label='Training loss')
    plt.plot(valid_losses, label='Validation loss')
    plt.legend()
    plt.show()

    # Plot the training and validation accuracy
    plt.plot(train_acc, label='Training accuracy')
    plt.plot(valid_acc, label='Validation accuracy')
    plt.legend()
    plt.show()

    # save the model
    # torch.save(ag_news_model.state_dict(), 'best_ag_news_model.pt')

    model.load_state_dict(torch.load(f'best_{modelName}.pt')) #, map_location=torch.device('cpu')))
    print()
    AllSkLMetrics(model, 'valid', valid_loader)
    print()
    AllSkLMetrics(model, 'test', test_loader) 
    
    return model, BVA2, BTA2

