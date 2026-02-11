"""
Preprocessing for POS tagging on Universal Dependencies (Multi-lingual).

This module:
- Loads UD .conllu files
- Skips Multi-Word Tokens (MWTs)
- Builds vocabularies (word, POS tag, character)
- Encodes sentences (words and characters)
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
CHAR_PAD = "<CHR_PAD>"

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


def build_char_vocab(sentences):
    """
    Build a character-to-index vocabulary.
    
    Args:
        sentences: list of (words, tags)
    
    Returns:
        char2idx: dict[str, int]
    """
    chars = set()
    
    for words, _ in sentences:
        for word in words:
            chars.update(word)
    
    char2idx = {CHAR_PAD: 0}
    idx = 1
    
    for char in sorted(chars):
        char2idx[char] = idx
        idx += 1
    
    return char2idx

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


def encode_sentences_with_chars(sentences, word2idx, tag2idx, char2idx, max_word_length=20):
    """
    Encode sentences with both word IDs and character IDs.
    
    Args:
        sentences: list of (words, tags)
        word2idx: word to index mapping
        tag2idx: tag to index mapping
        char2idx: character to index mapping
        max_word_length: maximum characters per word
    
    Returns:
        encoded: list of (word_ids, tag_ids, char_ids)
            - word_ids: list[int]
            - tag_ids: list[int]
            - char_ids: list[list[int]]
    """
    encoded = []
    
    for words, tags in sentences:
        word_ids = [word2idx.get(w, word2idx[UNK]) for w in words]
        tag_ids = [tag2idx[t] for t in tags]
        
        # Encode characters for each word
        char_ids_list = []
        for word in words:
            char_ids = [char2idx.get(c, 1) for c in word[:max_word_length]]  # Truncate to max length
            # Pad characters to fixed length
            char_ids += [0] * (max_word_length - len(char_ids))
            char_ids_list.append(char_ids)
        
        encoded.append((word_ids, tag_ids, char_ids_list))
    
    return encoded

# ---------------------------------------------------------------------
# Padding (used by DataLoader)
# ---------------------------------------------------------------------

def pad_sequences(batch):
    """
    Pad a batch of variable-length sentences.

    Args:
        batch: list of (word_ids, tag_ids) or (word_ids, tag_ids, char_ids)

    Returns:
        words_padded: LongTensor (B, T)
        tags_padded:  LongTensor (B, T)
        lengths:      LongTensor (B)
        char_ids_padded: LongTensor (B, T, max_word_length) or None
    """
    batch_size = len(batch)
    has_chars = len(batch[0]) == 3
    
    lengths = torch.tensor([len(words) for words, _, *_ in batch], dtype=torch.long)
    max_len = lengths.max().item()

    words_padded = torch.zeros(batch_size, max_len, dtype=torch.long)
    tags_padded = torch.zeros(batch_size, max_len, dtype=torch.long)
    
    char_ids_padded = None
    if has_chars:
        max_word_length = len(batch[0][2][0]) if batch[0][2] else 20
        char_ids_padded = torch.zeros(batch_size, max_len, max_word_length, dtype=torch.long)

    for i, item in enumerate(batch):
        if has_chars:
            words, tags, chars = item
        else:
            words, tags = item
            chars = None
        
        words_padded[i, :len(words)] = torch.tensor(words)
        tags_padded[i, :len(tags)] = torch.tensor(tags)
        
        if char_ids_padded is not None and chars:
            for j, char_seq in enumerate(chars):
                char_ids_padded[i, j, :] = torch.tensor(char_seq)

    if char_ids_padded is not None:
        return words_padded, tags_padded, lengths, char_ids_padded
    else:
        return words_padded, tags_padded, lengths

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class POSDataset(Dataset):
    """
    PyTorch Dataset for POS tagging.
    Each item is a (word_ids, tag_ids) pair or (word_ids, tag_ids, char_ids) triple.
    """

    def __init__(self, encoded_sentences):
        self.data = encoded_sentences
        self.has_chars = len(encoded_sentences[0]) == 3 if encoded_sentences else False

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
