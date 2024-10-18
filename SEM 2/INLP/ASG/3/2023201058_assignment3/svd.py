from svdfun import *

# Read the data
train_df = pd.read_csv("train.csv")
corpus_train = train_df["Description"].to_numpy()
# tokenizer = get_tokenizer("basic_english")
stop_words = set(stopwords.words('english'))
# stop_words.update(['.', ',', '?', '-', ':', '$', '%', '!'])
symbols = sorted(['.', ',', '?', '-', ':', ';', '$', '%', '!', 's'])
stop_words.update(symbols)
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
    tokens_test2.append(tokenize_text2(sent, word_to_index, symbols, lemmatizer))
    # tokens_test.append(sent)

test_df['Description'] = [' '.join(tokens) for tokens in tokens_test2]


tokenized_descriptions_test = test_df['Description'].tolist()
test_df['CleanDescriptionTokens'] = [description.split() for description in tokenized_descriptions_test]







window_sizes = [1,2,3,4,5]
k = 300  # Number of dimensions for word vectors
# Build co-occurrence matrix
# co_occurrence_matrix = np.zeros((vocab_size, vocab_size))



for WINDOW_SIZE in window_sizes:
    print(f'Window Size = {WINDOW_SIZE}')
    co_occurrence_matrix = lil_matrix((vocab_size, vocab_size), dtype=np.float64)
    for description in tokenized_descriptions:
        for i, word in enumerate(description):
            center_index = word_to_index[word]
            for j in range(max(0, i - WINDOW_SIZE), min(len(description), i + WINDOW_SIZE + 1)):
                if j != i:
                    # context_word = description[j]
                    context_index = word_to_index.get(description[j])
                    if context_index is not None:
                        co_occurrence_matrix[center_index,context_index] += 1
                        # co_occurrence_matrix[center_index][context_index] += 1

    print("COM")

    svd = TruncatedSVD(n_components=k)
    word_vectors = svd.fit_transform(co_occurrence_matrix)
    print("SVD")
    # Create a dictionary with word embeddings
    word_embeddings = {word: word_vectors[word_to_index[word]] for word in vocabulary}
    word_embeddings['unk'] = np.zeros(k)
    # Save the word embeddings dictionary as a PyTorch file
    torch.save(word_embeddings, f'svd-word-vectors-w={WINDOW_SIZE}.pt')



