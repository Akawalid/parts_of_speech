# Data Setup Instructions

## Installing Requirements

First, install all required packages:

```bash
pip install -r requirements.txt
```

## Data Path Configuration

**Important:** The dataset is not included in the repository due to its size (7GB). Since each team member stores the data in a different location, we need to configure the path individually to ensure the project runs correctly during grading.

### Setup Steps

1. Create a `.env` file in the project root directory
2. Add your local data path to the file

### Example

If your data is located at `/home/bagga/Desktop/pos_ud_project/data/raw/allzip/ud-treebanks-v2.17/UD_English-EWT`, your `.env` file should contain:

```
UD_DATA_PATH=/home/bagga/Desktop/pos_ud_project/data/raw/allzip/ud-treebanks-v2.17/UD_English-EWT
```

### For the Instructor

When testing this project, please:
1. Download the UD Treebanks v2.17 dataset from: 
`https://lindat.mff.cuni.cz/repository/items/b4fcb1e0-f4b2-4939-80f5-baeafda9e5c0`
2. Extract it to your preferred location
3. Create a `.env` file in the project root
4. Set `UD_DATA_PATH` to point to your `UD_English-EWT` directory

This approach ensures path compatibility across different systems without hardcoding absolute paths in the source code.

# Project Architecture

This document describes the main modules of the POS tagging project and their responsibilities.

---

## 1. run.py (Entry point of the project)
**Role:** calls the function `main.py` of `src/train.py`.

## 1. `src/preprocessing.py`

**Role:** Prepare data for training.

### Responsibilities
- Read `.conllu` files from the UD corpus
- Clean and transform data
- Create vocabularies for words and POS tags
- Encode into numerical identifiers
- Handle padding for batches

### Main Functions

| Function | Description |
|----------|-------------|
| `load_sentences(path)` | Reads a UD `.conllu` file and returns a list of tuples `(words, tags)` |
| `build_word_vocab(sentences, min_freq=1)` | Creates a word → index dictionary |
| `build_pos_vocab(sentences)` | Creates a POS tag → index dictionary |
| `encode_sentences(sentences, word2idx, tag2idx)` | Transforms words and tags into numerical identifiers |
| `pad_sequences(batch)` | Pads a batch of sentences to the same length |
| `POSDataset(Dataset)` | PyTorch Dataset for encoded sentences |

---

## 2. `src/data.py`

**Role:** Interface for loading data and creating DataLoaders for training.

### Main Functions

| Function | Description |
|----------|-------------|
| `load_data_and_dataloaders(data_path, batch_size=4, min_freq=2)` | Loads train/dev/test sets, builds vocabularies, encodes sentences, and returns DataLoaders with padding |
| `build_vocab_and_labels(sentences, min_freq=2)` | Separately creates word and tag dictionaries if needed |

---

## 3. `src/model.py`

**Role:** Define the BiLSTM model for POS tagging and training/evaluation functions.

### Main Classes and Functions

| Class / Function | Description |
|------------------|-------------|
| `BiLSTMPOSTagger(nn.Module)` | BiLSTM model with embedding layer, bidirectional LSTM, and linear layer for POS classification |
| `train_epoch(model, loader, optimizer, criterion, device, n_classes)` | Trains the model for one epoch |
| `train_model(model, train_loader, dev_loader, optimizer, criterion, device, n_classes, num_epochs)` | Trains over multiple epochs and displays performance on dev set |
| `evaluate(model, loader, device)` | Evaluates model accuracy on a DataLoader |

---

## 4. `src/train.py`

**Role:** Main training script.

### Responsibilities
- Load DataLoaders
- Create model and optimizer
- Train the model and evaluate on test set

### Main Functions

| Function | Description |
|----------|-------------|
| `main()` | Entry point for training. Loads data, creates model and DataLoaders, trains and evaluates |

---

## Workflow Summary

```
Data (.conllu files)
    ↓
preprocessing.py (load & encode)
    ↓
data.py (create DataLoaders)
    ↓
model.py (BiLSTM architecture)
    ↓
train.py (training & evaluation)
    ↓
Trained Model
```

## What comes next (ordred wrt to priority)
- Adapt the model to be multilingual.
- put the hyperparameters inside the config file.
- Optimize the hyperparameters using DeepHyper to improve the performance.
- Orgnize better the functions by creating utilities files (for data, training, model?...).