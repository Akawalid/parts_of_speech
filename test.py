#%% 
import os 
from conllu import parse_incr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# %% [markdown]
# I. Data extraction.  


#All treebanks are located in the file "Data/ud-treebanks-v2.17". 
# First we parse the files to extract list of sentences. Each sentence is a list of elements of the form (form, upos). 
# For a given word, "form" is its basic form (e.g for eaten, it would be eat)


# %%
def load_sentences(filepath):
    """Loads a .conllu file and returns a list of sentences. 
    Each sentence is a list of elements (words) of the form (word, upos)
    """

    sentences = []
    with open(filepath, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            sentence = [(t["form"], t["upos"]) for t in tokenlist]
            sentences.append(sentence)
    return sentences

# %%
DATA = "data/raw/allzip/ud-treebanks-v2.17"


train_path_english = os.path.join(DATA, "UD_English-EWT", "en_ewt-ud-train.conllu")
valid_path_english = os.path.join(DATA, "UD_English-EWT", "en_ewt-ud-dev.conllu")
test_path_english = os.path.join(DATA, "UD_English-EWT", "en_ewt-ud-test.conllu")

#extracting a list of sentences
train_sentences = load_sentences(train_path_english)
dev_sentences   = load_sentences(valid_path_english)
test_sentences  = load_sentences(test_path_english)


# %% [markdown]
#Now we need to prepare the data for training. Classic models normally expect :
#- X : list of words (or features derived from words) /list of list words (each sublist being a sentence)
#- y : list of labels (UPOS)

#Therefore we need to transform our list of sentences


# %%
def transform_sentences(sentences, flat=False):
    """Builds dataset (X and y) from a list of sentences.
    Each sentence is a list of (form, upos) tuples.

    Args:
        sentences: list of sentences (each sentence = list of (word, tag))

    Returns:
        X: list of words if flat=True else list of list of words (each list inside the main one
        representing a sentence)
        y: list of UPOS tags (or list of list of tags if flat=False)
    """

    if flat:
        X, y = [], []
        for sentence in sentences:
            for word, tag in sentence:
                X.append(word)
                y.append(tag)
        return X, y

    else:
        X = [[word for word, _ in sentence] for sentence in sentences]
        y = [[tag for _, tag in sentence] for sentence in sentences]
        return X, y
    

# %%
# converting all sentences to X, y form
X_train, y_train = transform_sentences(train_sentences, flat=False)
X_dev, y_dev = transform_sentences(dev_sentences, flat=False)
X_test, y_test = transform_sentences(test_sentences, flat=False)


# %%
# checking that our transformation algorithm is working
for i in range(10):
    print(X_train[i], y_train[i])


# %%
#building a vocabulary
UNK = "<UNK>"
PAD = "<PAD>"

def build_vocab(X_train):
    word_vocab = {"<PAD>": 0, "<UNK>": 1}
    for sentence in X_train:
        for w in sentence:
            if w not in word_vocab:
                word_vocab[w] = len(word_vocab)
    return word_vocab

word_vocab = build_vocab(X_train)

# %%
def encode(X, word_vocab):
    return [[word_vocab.get(w, word_vocab[UNK]) for w in sentence] for sentence in X]


# %%
label_encoder = LabelEncoder()
label_encoder.fit([tag for sentence in y_train for tag in sentence])


# %%
# Data encoding
X_train_enc = encode(X_train, word_vocab)
y_train_enc = [label_encoder.transform(sentence).tolist() for sentence in y_train]
X_dev_enc = encode(X_dev, word_vocab)
y_dev_enc = [label_encoder.transform(sentence).tolist() for sentence in y_dev]


# %%
# Building DataLoaders

class POSDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
            len(self.X[i]),
        )
    

def collate_fn(batch):
    word_ids, tag_ids, lengths = zip(*batch)
    word_ids = torch.nn.utils.rnn.pad_sequence(word_ids, batch_first=True, padding_value=0)
    tag_ids = torch.nn.utils.rnn.pad_sequence(tag_ids, batch_first=True, padding_value=-100)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return word_ids, tag_ids, lengths

train_dataset = POSDataset(X_train_enc, y_train_enc)
train_loader = DataLoader(train_dataset, batch_size = 64, shuffle=True, collate_fn=collate_fn)
dev_dataset = POSDataset(X_dev_enc, y_dev_enc)
dev_loader = DataLoader(dev_dataset, batch_size = 64, shuffle=True, collate_fn=collate_fn)


#%%
# Model

class BiLSTMPOSTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=1
        )
        self.output_layer = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, word_ids):
        #word_ids is of shape (batch, seq_len)
        emb = self.embedding(word_ids)
        post_lstm, _ = self.lstm(emb)
        logits = self.output_layer(post_lstm)
        return logits
# %%
#Model Training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(word_vocab)
n_classes = len(label_encoder.classes_)
emb_dim = 128
hidden_dim = 128

model = BiLSTMPOSTagger(vocab_size, emb_dim, hidden_dim, n_classes, padding_idx=0).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=-100) #ignoring padding index

def training(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    for word_ids, tag_ids, lengths in loader:
        word_ids, tag_ids = word_ids.to(device), tag_ids.to(device)
        logits = model(word_ids)
        loss = criterion(logits.view(-1, n_classes), tag_ids.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for word_ids, tag_ids, lengths in loader:
            word_ids, tag_ids = word_ids.to(device), tag_ids.to(device)
            logits = model(word_ids)
            pred = logits.argmax(dim=-1)
            mask = tag_ids != -100
            correct += (pred[mask] == tag_ids[mask]).sum().item()
            total += mask.sum().item()
    return correct / total if total else 0.0


num_epochs = 10
for epoch in range(num_epochs):
    loss = training(model, train_loader, optimizer, criterion, device)
    acc = evaluate(model, dev_loader, device)
    print(f"Epoch {epoch+1}/{num_epochs}  loss={loss:.4f}  dev_acc={acc:.4f}")
# %%
