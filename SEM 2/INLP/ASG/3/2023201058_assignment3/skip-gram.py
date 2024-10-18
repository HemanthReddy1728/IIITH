from skipgramfun import *

# random.seed(1024)
# USE_CUDA = torch.cuda.is_available()
# gpus = [0]
#torch.cuda.set_device(gpus[0])

# FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
# ByteTensor = torch.cuda.ByteTensor if torch.cuda.is_available() else torch.ByteTensor

# Read the data
train_df = pd.read_csv("train.csv")
corpus_train = train_df["Description"].to_numpy()
# tokenizer = get_tokenizer("basic_english")
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '?', '-', ':', ';', '$', '%', '!', 's'])
# symbols = sorted(['.', ',', '?', '-', ':', ';', '$', '%', '!', 's'])
lemmatizer = WordNetLemmatizer()


tokens_train = []
for sent in corpus_train:
    # tokens_train.append(tokenize_text(clean_text(sent), symbols, lemmatizer))
    tokens_train.append(tokenize_text(clean_text(sent), stop_words, lemmatizer))
    # tokens_train.append(sent)

train_df['Description'] = [' '.join(tokens) for tokens in tokens_train]

test_df = pd.read_csv("test.csv")
corpus_test = test_df["Description"].to_numpy()
tokens_test = []
for sent in corpus_test:
    # tokens_test.append(tokenize_text(clean_text(sent), symbols, lemmatizer))
    tokens_test.append(tokenize_text(clean_text(sent), stop_words, lemmatizer))
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
    tokens_test2.append(tokenize_text2(sent, vocabulary, stop_words, lemmatizer))
    # tokens_test.append(sent)

test_df['Description'] = [' '.join(tokens) for tokens in tokens_test2]

tokenized_descriptions_test = test_df['Description'].tolist()
test_df['CleanDescriptionTokens'] = [description.split() for description in tokenized_descriptions_test]



tokenized_descriptions_sub = tokenized_descriptions[:70000]
word_count = Counter(flatten(tokenized_descriptions_sub))
MIN_COUNT = 1
exclude = []
for w, c in word_count.items():
    if c < MIN_COUNT:
        exclude.append(w)

vocab = list(set(flatten(tokenized_descriptions_sub)) - set(exclude))

word2index = {}
for vo in vocab:
    if word2index.get(vo) is None:
        word2index[vo] = len(word2index)

index2word = {v : k for k, v in word2index.items()}




window_sizes = [4,3,2,1]
for WINDOW_SIZE in window_sizes:
    print(f'Window Size = {WINDOW_SIZE}')
    print("TD")
    windows =  flatten([list(nltk.ngrams(['<DUMMY>'] * WINDOW_SIZE + c + ['<DUMMY>'] * WINDOW_SIZE, WINDOW_SIZE * 2 + 1)) for c in tokenized_descriptions_sub])
    print(len(windows))
    sample_train_data = []

    for window in tqdm(windows):
        for i in range(WINDOW_SIZE * 2 + 1):
            if window[i] in exclude or window[WINDOW_SIZE] in exclude or i == WINDOW_SIZE or window[i] == '<DUMMY>':
                continue # min_count
            # if i == WINDOW_SIZE or window[i] == '<DUMMY>':
                # continue
            # sample_train_data.append((prepare_word(window[WINDOW_SIZE], word2index).view(1, -1), prepare_word(window[i], word2index).view(1, -1)))
            sample_train_data.append((prepare_sequence([window[WINDOW_SIZE]], word2index).view(1, -1), prepare_sequence([window[i]], word2index).view(1, -1)))
    print(len(sample_train_data))

    del windows
    
    Z = 0.001
    num_total_words = sum([c for w, c in word_count.items() if w not in exclude])
    unigram_table = []

    for vo in vocab:
        unigram_table.extend([vo] * int(((word_count[vo]/num_total_words)**0.75)/Z))
    print(len(vocab), len(unigram_table))

    EMBEDDING_SIZE = 300
    BATCH_SIZE = 32
    EPOCH = 5
    NEGSAMP = 5 # Num of Negative Sampling
    losses = []
    # sgnsmodel = SkipgramNegSampling(len(word2index), EMBEDDING_SIZE)
    # if torch.cuda.is_available():
        # sgnsmodel = sgnsmodel.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sgnsmodel = SkipgramNegSampling(len(word2index), EMBEDDING_SIZE).to(device)
    optimizer = optim.Adam(sgnsmodel.parameters(), lr=0.001)

    for epoch in tqdm(range(EPOCH)):
        print(epoch+1, " TrainEmbed")
        for i,batch in enumerate(getBatch(BATCH_SIZE, sample_train_data)):

            inputs, targets = zip(*batch)

            inputs = torch.cat(inputs) # B x 1
            targets = torch.cat(targets) # B x 1
            negs = negative_sampling(targets, unigram_table, NEGSAMP, word2index)
            sgnsmodel.zero_grad()

            loss = sgnsmodel(inputs, targets, negs)

            loss.backward()
            optimizer.step()

            losses.append(loss.data.tolist())
        # if epoch % 10 == 0:
        print(f"Epoch : {epoch+1}, mean_loss : {np.mean(losses)}")
        # losses = []
    

    del sample_train_data

    # Create a dictionary with word embeddings

    # k = EMBEDDING_SIZE
    word_embeddings = {word : sgnsmodel.embedding_v.weight.data[word2index[word]] for word in vocab}
    word_embeddings['unk'] = np.zeros(EMBEDDING_SIZE)
    # Save the word embeddings dictionary as a PyTorch file
    torch.save(word_embeddings, f'skip-gram-word-vectors-w={WINDOW_SIZE}.pt')
    print()
    print()



