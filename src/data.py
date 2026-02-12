import os
import torch
from torch.utils.data import Dataset, DataLoader
from preprocessing import (
    load_sentences,
    build_word_vocab,
    build_pos_vocab,
    build_char_vocab,
    encode_sentences_with_chars,
    POSDataset,
    pad_sequences,
)
from language_manager import LanguageManager
import config


def load_data_and_dataloaders(data_path, language='en', batch_size=config.BATCH_SIZE, min_freq=config.MIN_WORD_FREQ, use_chars=True):
    lang_manager = LanguageManager(data_path)
    
    file_paths = lang_manager.get_file_paths(language)
    
    print(f"Loading {config.LANGUAGES[language]['name']} dataset...")
    train_sentences = load_sentences(file_paths['train'])
    dev_sentences = load_sentences(file_paths['dev'])
    test_sentences = load_sentences(file_paths['test'])
    
    print(f"  Train sentences: {len(train_sentences)}")
    print(f"  Dev sentences: {len(dev_sentences)}")
    print(f"  Test sentences: {len(test_sentences)}")
    
    print("Building vocabularies...")
    word2idx = build_word_vocab(train_sentences, min_freq=min_freq)
    tag2idx = build_pos_vocab(train_sentences)
    char2idx = build_char_vocab(train_sentences) if use_chars else None
    
    print(f"  Vocabulary size: {len(word2idx)}")
    print(f"  POS tags: {len(tag2idx)}")
    if char2idx:
        print(f"  Character vocabulary: {len(char2idx)}")
    
    if use_chars:
        train_encoded = encode_sentences_with_chars(train_sentences, word2idx, tag2idx, char2idx)
        dev_encoded = encode_sentences_with_chars(dev_sentences, word2idx, tag2idx, char2idx)
        test_encoded = encode_sentences_with_chars(test_sentences, word2idx, tag2idx, char2idx)
    else:
        from preprocessing import encode_sentences
        train_encoded = encode_sentences(train_sentences, word2idx, tag2idx)
        dev_encoded = encode_sentences(dev_sentences, word2idx, tag2idx)
        test_encoded = encode_sentences(test_sentences, word2idx, tag2idx)
    
    train_dataset = POSDataset(train_encoded)
    dev_dataset = POSDataset(dev_encoded)
    test_dataset = POSDataset(test_encoded)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_sequences)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=pad_sequences)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=pad_sequences)
    
    lang_manager.cache_vocab(language, word2idx, tag2idx, char2idx)
    
    return train_loader, dev_loader, test_loader, word2idx, tag2idx, char2idx


def load_data_for_languages(data_path, languages=None, batch_size=config.BATCH_SIZE, use_chars=True):
    if languages is None:
        languages = list(config.LANGUAGES.keys())
    
    data_dict = {}
    
    for lang in languages:
        try:
            train_loader, dev_loader, test_loader, word2idx, tag2idx, char2idx = load_data_and_dataloaders(
                data_path, language=lang, batch_size=batch_size, use_chars=use_chars
            )
            data_dict[lang] = {
                'train_loader': train_loader,
                'dev_loader': dev_loader,
                'test_loader': test_loader,
                'word2idx': word2idx,
                'tag2idx': tag2idx,
                'char2idx': char2idx,
            }
            print(f"Successfully loaded {config.LANGUAGES[lang]['name']}")
        except FileNotFoundError as e:
            print(f"Failed to load {lang}: {e}")
    
    return data_dict
