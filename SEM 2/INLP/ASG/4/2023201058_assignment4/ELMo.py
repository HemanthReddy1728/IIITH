from ELMofunc import *

# !pip install gdown datasets
# !gdown 1wjStFMnaA0csj5-roEIQXU4z6Pbdc3Pn
# glove_vectors = torchtext.vocab.Vectors(name='glove.6B.100d.txt')
dimension = 100 
glove_vectors = torchtext.vocab.GloVe(name='6B', dim=dimension)

dataset = load_dataset("ag_news", trust_remote_code=True)
punctuations = set(string.punctuation)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

stop_words = set(stopwords.words('english'))
symbols = sorted(['.', ',', '?', '-', ':', ';', '$', '%', '!', 's'])
stop_words.update(symbols)
lemmatizer = WordNetLemmatizer()

test = []
lnt3 = []
for example in dataset["test"]:
    text = example["text"]
    # tokens = word_tokenize(text.lower())
    # tokens = [(token) for token in tokens if token not in stop_words and token not in punctuations and token.isalnum()]
    tokens = tokenize_text(clean_text(text), stop_words, lemmatizer)
    test.append({"text": tokens, "label": example["label"]})
    lnt3.append(len(tokens))

pretrain = []
for example in dataset["train"]:
    text = example["text"]
    # tokens = word_tokenize(text.lower())
    # tokens = [(token) for token in tokens if token not in stop_words and token not in punctuations and token.isalnum()]
    tokens = tokenize_text(clean_text(text), stop_words, lemmatizer)
    pretrain.append({"text": tokens, "label": example["label"]})

# Create the vocabulary
vocab = create_vocab(pretrain)

# Assuming 'data' is your list of dictionaries as described
tokens = [d['text'] for d in pretrain]  # Extracting the text data
labels = [d['label'] for d in pretrain]  # Extracting the labels


# Split the data into training and validation sets
train_tokens, val_tokens, train_labels, val_labels = train_test_split(tokens, labels, test_size=0.075, stratify=labels, random_state=42)

# If you need the dictionaries in the split data:
train = []
lnt1 = []
for tokenL, label in zip(train_tokens, train_labels):
    train.append({'text': tokenL, 'label': label})
    lnt1.append(len(tokenL))
    
val = []
lnt2 = []
for tokenL, label in zip(val_tokens, val_labels):
    val.append({'text': tokenL, 'label': label})
    lnt2.append(len(tokenL))

# print(len(train), len(val), len(test))
# print(len(lnt1), len(lnt2), len(lnt3))

# MAX_LENGTH = 119
# lnt = lnt1+lnt2+lnt3
MAX_LENGTH = max(lnt1+lnt2+lnt3)
print(MAX_LENGTH)




# Convert the tokenized datasets to indices
# create data loaders
# batch_size = 32

train_data = token2index_dataset(train, vocab)
train_data_labels = [example['label'] for example in train_data]
train_data_texts = [example['text'] for example in train_data]
train_data_pad_texts = pad_texts(train_data_texts, MAX_LENGTH)
train_dataset = DatasetConstructor(train_data_pad_texts, train_data_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

valid_data = token2index_dataset(val, vocab)
valid_data_labels = [example['label'] for example in valid_data]
valid_data_texts = [example['text'] for example in valid_data]
valid_data_pad_texts = pad_texts(valid_data_texts, MAX_LENGTH)
valid_dataset = DatasetConstructor(valid_data_pad_texts, valid_data_labels)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

test_data = token2index_dataset(test, vocab)
test_data_labels = [example['label'] for example in test_data]
test_data_texts = [example['text'] for example in test_data]
test_data_pad_texts = pad_texts(test_data_texts, MAX_LENGTH)
test_dataset = DatasetConstructor(test_data_pad_texts, test_data_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)




# Create a weight matrix for words in the training dataset
weights_matrix = torch.zeros((len(vocab), dimension))
words_found = 0
for i, word in enumerate(vocab.keys()):
    try:
        weights_matrix[i] = glove_vectors[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = torch.zeros(dimension)


# Instantiate the ELMo model
elmo_model = ELMo(len(vocab), weights_matrix).to(device)
print(elmo_model)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(elmo_model.parameters(), lr=1e-6)
# optimizer = torch.optim.SGD(elmo_model.parameters(), lr=0.01, momentum=0.9)
elmo_embeddings = None
num_epochs = 25
BVA1 = 0.0
BTA1 = 0.0


optimizer = torch.optim.Adam(elmo_model.parameters(), lr=1e-2)
elmo_model, BVA1, BTA1 = PreTrainingTask(elmo_model, criterion, optimizer, train_loader, valid_loader, num_epochs, vocab, BVA1, BTA1)

optimizer = torch.optim.Adam(elmo_model.parameters(), lr=1e-3)
elmo_model, BVA1, BTA1 = PreTrainingTask(elmo_model, criterion, optimizer, train_loader, valid_loader, num_epochs, vocab, BVA1, BTA1)

optimizer = torch.optim.Adam(elmo_model.parameters(), lr=1e-4)
elmo_model, BVA1, BTA1 = PreTrainingTask(elmo_model, criterion, optimizer, train_loader, valid_loader, num_epochs, vocab, BVA1, BTA1)

optimizer = torch.optim.Adam(elmo_model.parameters(), lr=1e-5)
elmo_model, BVA1, BTA1 = PreTrainingTask(elmo_model, criterion, optimizer, train_loader, valid_loader, num_epochs, vocab, BVA1, BTA1)

optimizer = torch.optim.Adam(elmo_model.parameters(), lr=1e-6)
elmo_model, BVA1, BTA1 = PreTrainingTask(elmo_model, criterion, optimizer, train_loader, valid_loader, num_epochs, vocab, BVA1, BTA1)

# for name, param in elmo_model.named_parameters():
    # if param.requires_grad:
        # print(name, param.data, param.shape)

elmo_embeddings = list(elmo_model.parameters())[0] #.to(device).detach().numpy()
# print(elmo_embeddings.shape)
# word = "<UNK>"
# word_index = vocab[word]
# print(elmo_embeddings[word_index])
torch.save(elmo_embeddings, 'bilstm_elmo_embeddings.pt')

elmo_lstmf = elmo_model.lstm_forward
elmo_lstmb = elmo_model.lstm_backward



# save the model
# torch.save(ag_news_model.state_dict(), 'best_ag_news_model.pt')

    # model.load_state_dict(torch.load(f'best_{modelName}.pt'))#, map_location=torch.device('cpu')))
    # print()
    # AllSkLMetrics(model, 'valid', valid_loader)
    # print()
    # AllSkLMetrics(model, 'test', test_loader) 