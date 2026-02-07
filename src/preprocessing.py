"""
Preprocessing for POS tagging on Universal Dependencies (English).

This module:
- Loads UD .conllu files
- Skips Multi-Word Tokens (MWTs)
- Builds vocabularies
- Encodes sentences
- Pads batches for BiLSTM models
- Provides PyTorch Dataset and DataLoader helpers
"""

import os
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from conllu import parse_incr
from dotenv import load_dotenv
# ---------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------

PAD = "<PAD>"
UNK = "<UNK>"

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_sentences(path):
    """
    Load a UD .conllu file.

    Returns:
        sentences: List of tuples (words, pos_tags)
            - words: list[str]
            - pos_tags: list[str]
    Notes:
        - Skips Multi-Word Tokens (MWTs)
        - Lowercases words
    """
    sentences = []

    with open(path, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            words = []
            pos_tags = []

            for token in tokenlist:
                # Keep only syntactic tokens (ignore MWTs)
                if isinstance(token["id"], int):
                    words.append(token["form"].lower())
                    pos_tags.append(token["upos"])

            sentences.append((words, pos_tags))

    return sentences

# ---------------------------------------------------------------------
# Vocabulary building
# ---------------------------------------------------------------------

def build_word_vocab(sentences, min_freq=1):
    """
    Build a word-to-index vocabulary from training sentences.

    Args:
        sentences: list of (words, tags)
        min_freq: minimum frequency to keep a word

    Returns:
        word2idx: dict[str, int]
    """
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


def build_pos_vocab(sentences):
    """
    Build a POS-tag-to-index vocabulary.

    Args:
        sentences: list of (words, tags)

    Returns:
        tag2idx: dict[str, int]
    """
    tags = set()

    for _, pos_tags in sentences:
        tags.update(pos_tags)

    tag2idx = {PAD: 0}
    idx = 1

    for tag in sorted(tags):
        tag2idx[tag] = idx
        idx += 1

    return tag2idx

# ---------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------

def encode_sentences(sentences, word2idx, tag2idx):
    """
    Encode sentences as lists of word IDs and tag IDs.

    Notes:
        - Manual encoding is preferred over sklearn LabelEncoder because it 
        ensures stability across train/dev/test splits
    """
    encoded = []

    for words, tags in sentences:
        word_ids = [word2idx.get(w, word2idx[UNK]) for w in words]
        tag_ids = [tag2idx[t] for t in tags]
        encoded.append((word_ids, tag_ids))

    return encoded

# ---------------------------------------------------------------------
# Padding (used by DataLoader)
# ---------------------------------------------------------------------

def pad_sequences(batch):
    """
    Pad a batch of variable-length sentences.

    Args:
        batch: list of (word_ids, tag_ids)

    Returns:
        words_padded: LongTensor (B, T)
        tags_padded:  LongTensor (B, T)
        lengths:      LongTensor (B)
    """
    batch_size = len(batch)
    lengths = torch.tensor([len(words) for words, _ in batch], dtype=torch.long)
    max_len = lengths.max().item()

    words_padded = torch.zeros(batch_size, max_len, dtype=torch.long)
    tags_padded = torch.zeros(batch_size, max_len, dtype=torch.long)

    for i, (words, tags) in enumerate(batch):
        words_padded[i, :len(words)] = torch.tensor(words)
        tags_padded[i, :len(tags)] = torch.tensor(tags)

    return words_padded, tags_padded, lengths

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class POSDataset(Dataset):
    """
    PyTorch Dataset for POS tagging.
    Each item is a (word_ids, tag_ids) pair.
    """

    def __init__(self, encoded_sentences):
        self.data = encoded_sentences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ---------------------------------------------------------------------
# DataLoader helper
# ---------------------------------------------------------------------

def create_dataloader(dataset, batch_size=32, shuffle=True):
    """
    Create a DataLoader with proper padding.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=pad_sequences
    )