#%% 
import os 
from conllu import parse_incr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np


# %% [markdown]
# I. Data extraction.  


#All treebanks are located in the file "Data/ud-treebanks-v2.17". 
# First we parse the files to extract list of sentences. Each sentence is a list of elements of the form (form, upos). 
# For a given word, "form" is its basic form (e.g for eaten, it would be eat)

# %%
def load_sentences(path):
    """Loads a .conllu file and returns a list of sentences. 
    Each sentence is a tuple of lists of the form (words, upos tags)
    We skip the multi-token words since they don't play any role in Pos tagging
    """
    sentences = []

    with open(path, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            words = []
            pos_tags = []

            for token in tokenlist:
                if isinstance(token["id"], int):  # skip MWTs
                    words.append(token["form"].lower())
                    pos_tags.append(token["upos"])

            sentences.append((words, pos_tags))

    return sentences

# %%
PAD = "<PAD>"
UNK = "<UNK>"

# %%
from collections import Counter

def build_word_vocab(sentences, min_freq=1):
    counter = Counter()

    for words, _ in sentences:
        counter.update(words)

    word2idx = {PAD: 0, UNK: 1}
    idx = 2

    for word, freq in counter.items():
        if freq >= min_freq:
            word2idx[word] = idx
            idx += 1

    return word2idx

# %%
def build_pos_vocab(sentences):
    tags = set()

    for _, pos_tags in sentences:
        tags.update(pos_tags)

    tag2idx = {PAD: 0}
    idx = 1

    for tag in sorted(tags):
        tag2idx[tag] = idx
        idx += 1

    return tag2idx

# %%
def encode_sentences(sentences, word2idx, tag2idx):
    """
        It's better to encode the words manually rather than usingf LabelEncoder of Sklearn
        because the latter can change the encoding of the words from train to dev to test 
        datasets.
    """
    encoded = []

    for words, tags in sentences:
        word_ids = [
            word2idx.get(w, word2idx[UNK]) for w in words
        ]
        tag_ids = [
            tag2idx[t] for t in tags
        ]
        encoded.append((word_ids, tag_ids))

    return encoded

# %%
import torch

def pad_sequences(batch):
    """
    batch: list of (word_ids, tag_ids)
    it takes the longest setence in the batch and padds the remaining sentences with 
    the label <PAD> to match the length of the longest sentece.
    """
    batch_size = len(batch)
    lengths = torch.tensor([len(x[0]) for x in batch])

    max_len = max(lengths)

    words_padded = torch.zeros(batch_size, max_len, dtype=torch.long)
    tags_padded = torch.zeros(batch_size, max_len, dtype=torch.long)

    for i, (words, tags) in enumerate(batch):
        words_padded[i, :len(words)] = torch.tensor(words)
        tags_padded[i, :len(tags)] = torch.tensor(tags)

    return words_padded, tags_padded, lengths

# %%
from torch.utils.data import Dataset

class POSDataset(Dataset):
    def __init__(self, encoded_sentences):
        self.data = encoded_sentences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# %%
DATA = "data/raw/allzip/ud-treebanks-v2.17"


train_path_english = os.path.join(DATA, "UD_English-EWT", "en_ewt-ud-train.conllu")
valid_path_english = os.path.join(DATA, "UD_English-EWT", "en_ewt-ud-dev.conllu")
test_path_english = os.path.join(DATA, "UD_English-EWT", "en_ewt-ud-test.conllu")

#extracting a list of sentences
train_sentences = load_sentences(train_path_english)
dev_sentences   = load_sentences(valid_path_english)
test_sentences  = load_sentences(test_path_english)

# Build vocabs
word2idx = build_word_vocab(train_sentences, min_freq=2)
tag2idx  = build_pos_vocab(train_sentences)

# %% [markdown]
#Now we need to prepare the data for training. Classic models normally expect :
#- X : list of words (or features derived from words) /list of list words (each sublist being a sentence)
#- y : list of labels (UPOS)

#Therefore we need to transform our list of sentences

# Encode
train_encoded = encode_sentences(train_sentences, word2idx, tag2idx)
dev_encoded   = encode_sentences(dev_sentences, word2idx, tag2idx)
test_encoded  = encode_sentences(test_sentences, word2idx, tag2idx)
# Dataset
train_dataset = POSDataset(train_encoded)
dev_dataset   = POSDataset(dev_encoded)
test_dataset   = POSDataset(test_encoded)

# %%
from torch.utils.data import DataLoader

def collate_fn(batch):
    return pad_sequences(batch)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)
"""
DataLoader creates an iterator, 
in the main loop it follows these instructions:
    Pick `batch_size=32` random indices
    Call __getitem__ `batch_size=32`  times
    Put the results into a list
    Pass that list to collate_fn
    Yield padded tensors
"""
