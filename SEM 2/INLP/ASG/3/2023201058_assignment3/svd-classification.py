from svdfun import *

# Read the data
train_df = pd.read_csv("train.csv")
corpus_train = train_df["Description"].to_numpy()
# tokenizer = get_tokenizer("basic_english")
# stop_words = set(stopwords.words('english'))
# stop_words.update(['.', ',', '?', '-', ':', '$', '%', '!'])
symbols = sorted(['.', ',', '?', '-', ':', ';', '$', '%', '!', 's'])
lemmatizer = WordNetLemmatizer()




tokens_train = []
for sent in corpus_train:
    tokens_train.append(tokenize_text(clean_text(sent), symbols, lemmatizer))
    # tokens_train.append(sent)

train_df['Description'] = [' '.join(tokens) for tokens in tokens_train]

test_df = pd.read_csv("test.csv")
corpus_test = test_df["Description"].to_numpy()
tokens_test = []
for sent in corpus_test:
    tokens_test.append(tokenize_text(clean_text(sent), symbols, lemmatizer))
    # tokens_test.append(sent)

test_df['Description'] = [' '.join(tokens) for tokens in tokens_test]

# from sklearn.preprocessing import LabelEncoder
# Convert class labels to numeric
# LabEnc = LabelEncoder()
# train_df['Class Index'] = LabEnc.fit_transform(train_df['Class Index'])
# test_df['Class Index'] = LabEnc.fit_transform(test_df['Class Index'])

# corpus_train = train_df["Description"].to_numpy()
# corpus_test = test_df["Description"].to_numpy()
# # Build Vocabulary
# MIN_WORD_FREQUENCY = 1
# vocab_train = build_vocab_from_iterator(tokens_train, min_freq=MIN_WORD_FREQUENCY, specials=["<unk>"])
# vocab_train.set_default_index(vocab_train["<unk>"])
# vocab_test = build_vocab_from_iterator(tokens_test, min_freq=MIN_WORD_FREQUENCY, specials=["<unk>"])
# vocab_test.set_default_index(vocab_test["<unk>"])

# print(f"Total sentences in train text: {len(tokens_train)}")
# print(f"Unique train words: {len(vocab_train)}")
# print(f"Total sentences in test text: {len(tokens_test)}")
# print(f"Unique test words: {len(vocab_test)}")

# with open("vocab_train.pkl", "wb") as f:
    # pickle.dump(vocab_train, f)
# with open("vocab_test.pkl", "wb") as f:
    # pickle.dump(vocab_test, f)

# Extract descriptions
descriptions = train_df['Description'].tolist()


# Tokenize the descriptions
tokenized_descriptions = [description.split() for description in descriptions]

# Define vocabulary
vocabulary = sorted(set(word for description in tokenized_descriptions for word in description))
word_to_index = {word: idx for idx, word in enumerate(vocabulary)}
vocab_size = len(vocabulary)
# Convert vocabulary to list for indexing
vocab_list = list(vocabulary)

train_df['CleanDescriptionTokens'] = tokenized_descriptions



corpus_test2 = test_df["Description"].to_numpy()
tokens_test2 = []
for sent in corpus_test2:
    tokens_test2.append(tokenize_text2(sent, word_to_index))
    # tokens_test.append(sent)

test_df['Description'] = [' '.join(tokens) for tokens in tokens_test2]


tokenized_descriptions_test = test_df['Description'].tolist()
test_df['CleanDescriptionTokens'] = [description.split() for description in tokenized_descriptions_test]



window_sizes = [1,2,3,4,5]
val_accuracies = []
test_accuracies = []
k = 300  # Number of dimensions for word vectors

for WINDOW_SIZE in window_sizes:
    print(f'Window Size = {WINDOW_SIZE}')
    # Access all word vectors
    word_embeddings = torch.load(f'svd-word-vectors-w={WINDOW_SIZE}.pt')
    # word_embeddings



    # Convert class labels to numeric and # Split data into training and validation sets
    LabEnc = LabelEncoder()
    train_df['Class Index'] = LabEnc.fit_transform(train_df['Class Index'])
    train_data, val_data = train_test_split(train_df, test_size=0.1, random_state=42)
    test_df['Class Index'] = LabEnc.fit_transform(test_df['Class Index'])
    test_data = test_df

    model = LSTMClassifier(k, 512, len(LabEnc.classes_))
    # Assuming the use of a GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    model.to(device)



    # print(word_embeddings['explosion'])
    # text = train_data['CleanDescriptionTokens'].tolist()[0]
    # print(torch.tensor([word_embeddings[word] for word in text]))

    # Create dataset instances and # DataLoader instances
    train_dataset = SentenceDataset(train_data['CleanDescriptionTokens'].tolist(), train_data['Class Index'].tolist(), word_embeddings)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    val_dataset = SentenceDataset(val_data['CleanDescriptionTokens'].tolist(), val_data['Class Index'].tolist(), word_embeddings)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    test_dataset = SentenceDataset(test_data['CleanDescriptionTokens'].tolist(), test_data['Class Index'].tolist(), word_embeddings)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


    # Set the model in training mode
    num_epochs = 5
    best_val_accuracy = 0
    best_model = None
    print("DL")
    for epoch in range(num_epochs):
        print(epoch+1, " train")
        model.train()
        running_loss = 0.0
        for descriptions, labels in tqdm(train_loader):
            descriptions, labels = descriptions.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(descriptions)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        print("eval")
        model.eval()
        correct = 0
        total = 0

        # Disable gradient calculation during validation
        with torch.no_grad():
            for descriptions, labels in tqdm(val_loader):  # Using train_loader for validation too
                descriptions, labels = descriptions.to(device), labels.to(device)
                outputs = model(descriptions)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss/len(train_loader)}, Validation Accuracy: {val_accuracy}")
        
        # Save the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model.state_dict()
            # Save the trained model
            torch.save(best_model, f'svd-classification-model-w={WINDOW_SIZE}.pt')
            
        # Set the model back in training mode
        model.train()
        
    print()
    print("SL")

    # Load the saved model state dictionary
    model.load_state_dict(torch.load(f'svd-classification-model-w={WINDOW_SIZE}.pt'))
    model.to(device)
    print()

    # Evaluation on the val set
    print("val")
    model.eval()
    correct = 0
    total = 0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for descriptions, labels in tqdm(val_loader):
            descriptions, labels = descriptions.to(device), labels.to(device)
            outputs = model(descriptions)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # test_accuracy = correct / total
    # test_accuracies.append(test_accuracy)
    # print(f'Accuracy on the test set: {test_accuracy} for window_size = {window_size}')
    AllSkLMetrics(true_labels, predicted_labels, 'val', WINDOW_SIZE, val_accuracies)



    # Evaluation on the test set
    print("test")
    model.eval()
    correct = 0
    total = 0
    true_labels = []
    predicted_labels = []


    with torch.no_grad():
        for descriptions, labels in tqdm(test_loader):
            descriptions, labels = descriptions.to(device), labels.to(device)
            outputs = model(descriptions)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # test_accuracy = correct / total
    # test_accuracies.append(test_accuracy)
    # print(f'Accuracy on the test set: {test_accuracy} for window_size = {window_size}')
    AllSkLMetrics(true_labels, predicted_labels, 'test', WINDOW_SIZE, test_accuracies)



# Plot
plt.plot(window_sizes, val_accuracies, marker='o', label='Validation Accuracies')
plt.plot(window_sizes, test_accuracies, marker='o', label='Test Accuracies')
plt.xlabel('Window Sizes')
plt.ylabel('Accuracies')
plt.title('Validation and Test Accuracies vs Window Sizes')
plt.legend()
plt.grid(True)
plt.show()  
