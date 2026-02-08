# src/data.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from preprocessing import (
    load_sentences,
    build_word_vocab,
    build_pos_vocab,
    encode_sentences,
    POSDataset,
    pad_sequences,
)

def load_data_and_dataloaders(data_path, batch_size=4, min_freq=2):
    """
    Load the UD dataset, build vocabularies, encode sentences, 
    create datasets and dataloaders.

    Returns:
        train_loader, dev_loader, test_loader, word2idx, tag2idx
    """
    # Paths
    train_file = os.path.join(data_path, "en_ewt-ud-train.conllu")
    dev_file   = os.path.join(data_path, "en_ewt-ud-dev.conllu")
    test_file  = os.path.join(data_path, "en_ewt-ud-test.conllu")

    # Load sentences
    train_sentences = load_sentences(train_file)
    dev_sentences   = load_sentences(dev_file)
    test_sentences  = load_sentences(test_file)

    # Build vocabularies
    word2idx = build_word_vocab(train_sentences, min_freq=min_freq)
    tag2idx  = build_pos_vocab(train_sentences)

    # Encode sentences
    train_encoded = encode_sentences(train_sentences, word2idx, tag2idx)
    dev_encoded   = encode_sentences(dev_sentences, word2idx, tag2idx)
    test_encoded  = encode_sentences(test_sentences, word2idx, tag2idx)

    # Create datasets
    train_dataset = POSDataset(train_encoded)
    dev_dataset   = POSDataset(dev_encoded)
    test_dataset  = POSDataset(test_encoded)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_sequences)
    dev_loader   = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=pad_sequences)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, collate_fn=pad_sequences)

    return train_loader, dev_loader, test_loader, word2idx, tag2idx


# Optional: function to build vocab and label encoder separately
def build_vocab_and_labels(sentences, min_freq=2):
    """
    Helper to create word2idx and tag2idx dictionaries
    """
    word2idx = build_word_vocab(sentences, min_freq=min_freq)
    tag2idx  = build_pos_vocab(sentences)
    return word2idx, tag2idx
