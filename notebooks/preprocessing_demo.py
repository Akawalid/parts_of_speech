# %% [markdown]
# # UD POS Preprocessing Demo
# 
# This notebook demonstrates the preprocessing pipeline from `src/data/preprocessing.py`.  
# We will:
# 1. Load UD English sentences
# 2. Build vocabularies
# 3. Encode sentences
# 4. Show padded batches ready for BiLSTM

# %%
# Adjust Python path if running from root
import sys
from dotenv import load_dotenv
import os

sys.path.append("/src")

# %%
from preprocessing import (
    load_sentences,
    build_word_vocab,
    build_pos_vocab,
    encode_sentences,
    POSDataset,
    create_dataloader,
    PAD, UNK
)

# %% [markdown]
# ## 1️⃣ Load English UD Data
load_dotenv()
DATA_PATH = os.getenv("UD_DATA_PATH")
TRAIN_FILE = os.path.join(DATA_PATH, "en_ewt-ud-train.conllu")
DEV_FILE   = os.path.join(DATA_PATH, "en_ewt-ud-dev.conllu")
TEST_FILE  = os.path.join(DATA_PATH, "en_ewt-ud-test.conllu")

# %%
train_sentences = load_sentences(TRAIN_FILE)
dev_sentences   = load_sentences(DEV_FILE)
test_sentences  = load_sentences(TEST_FILE)

print(f"Loaded {len(train_sentences)} training sentences")
print(f"Example sentence 0: {train_sentences[0]}")

# %% [markdown]
# ## 2️⃣ Build Vocabularies

# %%
word2idx = build_word_vocab(train_sentences, min_freq=2)
tag2idx  = build_pos_vocab(train_sentences)

print(f"Vocabulary size: {len(word2idx)}")
print(f"POS tag set: {tag2idx}")

# %% [markdown]
# ## 3️⃣ Encode Sentences

# %%
train_encoded = encode_sentences(train_sentences, word2idx, tag2idx)
dev_encoded   = encode_sentences(dev_sentences, word2idx, tag2idx)

# Show encoded first sentence
print("First training sentence (word ids, tag ids):")
print(train_encoded[0])

# %% [markdown]
# ## 4️⃣ Create Dataset & DataLoader

# %%
train_dataset = POSDataset(train_encoded)
train_loader = create_dataloader(train_dataset, batch_size=4)

# %% [markdown]
# ## 5️⃣ Inspect a Batch

# %%
for words_padded, tags_padded, lengths in train_loader:
    print("Words (padded):")
    print(words_padded)
    print("Tags (padded):")
    print(tags_padded)
    print("Lengths:")
    print(lengths)
    break  # show only first batch

# %% [markdown]
# ## 6️⃣ Decode IDs Back to Words/Tags (optional)

# %%
idx2word = {idx: w for w, idx in word2idx.items()}
idx2tag  = {idx: t for t, idx in tag2idx.items()}

first_sentence_words = [idx2word[i.item()] for i in words_padded[0, :lengths[0]]]
first_sentence_tags  = [idx2tag[i.item()] for i in tags_padded[0, :lengths[0]]]

print("First sentence in batch (decoded):")
print("Words:", first_sentence_words)
print("Tags: ", first_sentence_tags)

# %%
