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
DATA = "Data/ud-treebanks-v2.17"


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
X_train, y_train = transform_sentences(train_sentences, flat=True)
X_dev, y_dev = transform_sentences(dev_sentences, flat=True)
X_test, y_test = transform_sentences(test_sentences, flat=True)


# %%
# checking that our transformation algorithm is working
for i in range(10):
    print(X_train[i], y_train[i])


# %%
#building a vocabulary
UNK = "<UNK>"
word_vocab = {UNK: 0}
for w in X_train:
    if w not in word_vocab:
        word_vocab[w] = len(word_vocab)


# %%
def encode(X_train):
    return [word_vocab.get(w, word_vocab[UNK]) for w in X_train]


# %%
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
y_train_enc = label_encoder.transform(y_train)


# %%
# Training
X_train_enc = np.array(encode(X_train)).reshape(-1, 1)
clf = LogisticRegression(max_iter=500, random_state=42)
clf.fit(X_train_enc, y_train_enc)

# %%
#predictions and evaluation

X_dev_enc = np.array(encode(X_dev)).reshape(-1, 1)
y_dev_enc = label_encoder.transform(y_dev)
print("Accuracy (dev):", clf.score(X_dev_enc, y_dev_enc))
# %%
