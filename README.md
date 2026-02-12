# Multilingual Part-of-Speech Tagger

## What Does This Project Do?

This project teaches a computer to automatically identify what **part of speech** every word is in a sentence.

**Part of speech** means the grammatical role of a word:
- **Noun**: person, place, thing (cat, Paris, happiness)
- **Verb**: action or state (run, is, seems)
- **Adjective**: describes a noun (red, tall, happy)
- **Adverb**: modifies a verb (quickly, very, never)
- **Preposition**: shows relationships (in, on, under)
- And many more...

### Example

For the sentence: *"The quick brown fox jumps over the fence"*

The system should identify:
```
The       -> Determiner
quick     -> Adjective
brown     -> Adjective
fox       -> Noun
jumps     -> Verb
over      -> Preposition
the       -> Determiner
fence     -> Noun
```

## Why Is This Important?

Part-of-speech tagging is the **foundation** for many language technologies:
- Machine translation (translating between languages)
- Named entity recognition (finding names of people, places, companies)
- Sentiment analysis (understanding if text is positive or negative)
- Grammar checking and spell correction
- Search engines understanding what you mean
- Voice assistants understanding voice commands

---

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

If your data is located at `/home/bagga/Desktop/pos_ud_project/data/raw/allzip/ud-treebanks-v2.17`, your `.env` file should contain:

```
UD_DATA_PATH=/home/bagga/Desktop/pos_ud_project/data/raw/allzip/ud-treebanks-v2.17
```

### For the Instructor

When testing this project, please:
1. Download the UD Treebanks v2.17 dataset from: 
`https://lindat.mff.cuni.cz/repository/items/b4fcb1e0-f4b2-4939-80f5-baeafda9e5c0`
2. Extract it to your preferred location
3. Create a `.env` file in the project root
4. Set `UD_DATA_PATH` to point to your extracted treebanks directory

This approach ensures path compatibility across different systems without hardcoding absolute paths in the source code.

---

# Project Structure

```
project/
├── src/
│   ├── model.py              ← Neural network architecture (BiLSTM + Attention)
│   ├── train.py              ← Training loop for single language
│   ├── data.py               ← Data loading and DataLoader creation
│   ├── preprocessing.py      ← Parsing, tokenization, vocabulary building
│   ├── evaluate.py           ← Metrics computation (Accuracy, F1, Precision, Recall)
│   ├── config.py             ← All hyperparameters and language definitions
│   └── language_manager.py   ← Multi-language dataset handling
│
├── run.py                    ← Train for ONE language
├── train_all_languages.py    ← Train for ALL languages (with metrics logging)
│
├── results/                  ← Training outputs (auto-generated)
│   └── 20260212_143022/      ← Timestamped run folder
│       └── training_summary.json  ← Metrics and statistics (JSON)
│
├── checkpoints/              ← Intermediate model saves during training
├── requirements.txt          ← Python dependencies
├── .env                      ← Local configuration (UD_DATA_PATH)
└── README.md                 ← This file
```

---

# Project Architecture

This document describes the main modules and their responsibilities.

## Module Overview

### 1. `src/preprocessing.py`

**Role:** Prepare data for training.

#### Responsibilities
- Read `.conllu` files from the UD corpus (Universal Dependencies)
- Clean and transform raw data
- Create vocabularies for words, POS tags, and characters
- Encode sentences into numerical identifiers
- Handle padding for batch processing

#### Main Functions

| Function | Description |
|----------|-------------|
| `load_sentences(path)` | Reads a UD `.conllu` file and returns list of `(words, tags)` tuples |
| `build_word_vocab(sentences, min_freq=2)` | Creates word → index mapping, filters by frequency |
| `build_pos_vocab(sentences)` | Creates POS tag → index mapping |
| `build_char_vocab(sentences)` | Creates character → index mapping for CharCNN |
| `encode_sentences_with_chars(sentences, word2idx, tag2idx, char2idx)` | Encodes words and characters as numerical IDs |
| `pad_sequences(batch)` | Pads batch to same length for BiLSTM |
| `POSDataset(Dataset)` | PyTorch Dataset wrapper for encoded sentences |

---

### 2. `src/data.py`

**Role:** Load data and create PyTorch DataLoaders for training/evaluation.

#### Main Functions

| Function | Description |
|----------|-------------|
| `load_data_and_dataloaders(data_path, language='en', batch_size=32)` | Loads train/dev/test sets, builds vocabularies, returns DataLoaders |
| `load_data_for_languages(data_path, languages=['en','fr','de','es','zh'])` | Loads data for multiple languages simultaneously |

---

### 3. `src/model.py`

**Role:** Define the neural network architecture for POS tagging.

#### Main Classes

| Class | Description |
|-------|-------------|
| `CharCNN` | Extracts character-level features using convolutional filters (kernel sizes 3, 4, 5) |
| `MultiHeadAttention` | Multi-head attention mechanism (4 heads) for token weighting |
| `CRFLayer` | Conditional Random Field for sequence-level tagging (optional) |
| `SOTABiLSTMPOSTagger` | Complete model: CharCNN + Word Embeddings + BiLSTM (2 layers) + Attention |

#### Architecture Flow

```
Input: (batch_size, seq_length)
    ↓
Word Embedding (256-dim) + CharCNN (75-dim) = 331-dim
    ↓
LayerNorm + Dropout
    ↓
BiLSTM (2 layers, bidirectional, 256 hidden) → 512-dim
    ↓
MultiHeadAttention (4 heads) → 512-dim
    ↓
Dropout
    ↓
Dense Layer → num_tags
    ↓
Output: Logits (batch_size, seq_length, num_tags)
```

---

### 4. `src/train.py`

**Role:** Main training loop for a single language.

#### Responsibilities
- Load data using `data.py`
- Create model and optimizer (Adam)
- Perform training with early stopping
- Evaluate on test set
- Save best model checkpoint and metrics

#### Key Functions

| Function | Description |
|----------|-------------|
| `main(language='en', num_epochs=20, learning_rate=0.001)` | Entry point: loads data, trains model, evaluates |
| Returns: Dict with `test_accuracy`, `test_f1`, `test_precision`, `test_recall`, `per_tag_f1` |

---

### 5. `src/evaluate.py`

**Role:** Compute performance metrics.

#### Main Classes

| Class | Method | Description |
|-------|--------|-------------|
| `POSEvaluator` | `evaluate(model, loader, device)` | Computes accuracy, precision, recall, F1 (macro), per-tag F1 |
| `POSEvaluator` | `_per_tag_f1()` | Returns F1 score for each POS tag |

#### Metrics Computed
- **Accuracy**: % of correct predictions
- **Precision**: "When predicting TAG, how often correct?" (macro average)
- **Recall**: "Out of all actual TAGs, how many found?" (macro average)
- **F1**: Harmonic mean of precision and recall
- **Per-tag F1**: Individual F1 for each POS tag

---

### 6. `src/config.py`

**Role:** Centralized configuration for all hyperparameters and language definitions.

#### Language Configuration
```python
LANGUAGES = {
    'en': {'treebank': 'UD_English-EWT', 'name': 'English', ...},
    'fr': {'treebank': 'UD_French-GSD', 'name': 'French', ...},
    'de': {'treebank': 'UD_German-GSD', 'name': 'German', ...},
    'es': {'treebank': 'UD_Spanish-AnCora', 'name': 'Spanish', ...},
    'zh': {'treebank': 'UD_Chinese-GSD', 'name': 'Chinese', ...},
}
```

#### Model Hyperparameters
| Parameter | Value | Meaning |
|-----------|-------|---------|
| `EMBEDDING_DIM` | 256 | Word embedding dimension |
| `CHAR_EMBEDDING_DIM` | 50 | Character embedding dimension |
| `HIDDEN_DIM` | 256 | LSTM hidden state dimension |
| `NUM_LSTM_LAYERS` | 2 | Number of stacked LSTM layers |
| `BIDIRECTIONAL` | True | Use bidirectional LSTM |
| `USE_ATTENTION` | True | Include attention layer |
| `DROPOUT_RATE` | 0.4 | Dropout probability |
| `BATCH_SIZE` | 32 | Training batch size |
| `LEARNING_RATE` | 0.001 | Adam optimizer learning rate |
| `NUM_EPOCHS` | 20 | Maximum training epochs |
| `EARLY_STOPPING_PATIENCE` | 5 | Stop if no improvement for N epochs |

---

### 7. `src/language_manager.py`

**Role:** Handle multi-language dataset paths and caching.

#### Responsibilities
- Map language codes to UD treebank directories
- Cache vocabularies per language
- Load cached vocabularies if available

---

## Entry Points

### 8. `run.py`

Trains a **single language**:

```bash
python run.py en          # Train English
python run.py fr          # Train French
python run.py de es zh    # Train German, Spanish, Chinese
```

---

### 9. `train_all_languages.py`

Trains **all languages** with result logging:

```bash
python train_all_languages.py
```

**Output Structure:**
```
results/20260212_143022/
├── training_summary.json       ← All metrics in JSON
│   {
│       "languages": {
│           "en": {
│               "status": "completed",
│               "metrics": {
│                   "test_accuracy": 0.9634,
│                   "test_f1": 0.9505,
│                   ...
│               }
│           },
│           ...
│       }
│   }
├── results_visualization.png   ← Plots (auto-generated)
├── training_results.csv        ← CSV export
└── models/                     ← Trained models
```

---

## Training Workflow

```
Data (.conllu files)
    ↓
[preprocessing.py] Load & encode
    ↓
[data.py] Create DataLoaders
    ↓
[model.py] Build BiLSTM + Attention
    ↓
[train.py] Training loop
    ↓
[evaluate.py] Compute metrics
    ↓
[train_all_languages.py] Log results (JSON)
    ↓
results/20260212_143022/
├── training_summary.json
└── models/
```

---

## How Does It Work?

### The Brain: A Deep Neural Network

The system uses an artificial neural network with multiple specialized layers:

1. **Character-Level Analysis Layer (CharCNN)**
   - Looks at each word letter-by-letter
   - Extracts patterns like prefixes and suffixes (e.g., "-ing" suggests a verb form)
   - Uses small sliding windows (sizes 3, 4, 5 letters) to detect patterns
   - Returns a summary of "what this word looks like"

2. **Word Embedding Layer**
   - Converts each word to a mathematical representation (256 numbers)
   - Similar words get similar representations
   - Learned from training data: learns that "dog" and "cat" are similar

3. **BiLSTM Layer (The Memory)**
   - **BiLSTM** = "Bidirectional Long Short-Term Memory"
   - Reads the sentence in both directions (forwards AND backwards)
   - Can access context: knows words before and after the current word
   - Remembers important information across multiple words
   - 2 layers deep = enhanced pattern recognition

4. **Attention Layer (The Focus)**
   - Decides which words are most important for classification
   - "When tagging this word, I should pay special attention to THESE other words"
   - Uses multi-head attention (4 parallel attention mechanisms)

5. **Classification Layer**
   - Makes the final decision: "This word is a [NOUN/VERB/ADJ/...]"
   - Uses probability scores: 95% sure it's a Noun, 4% Verb, 1% other

### Training Process

The system learns through **supervised learning**:

1. **Start**: Random, untrained network (like a blank slate)
2. **Show examples**: Feed it sentences with correct answers
3. **It guesses**: The network makes predictions
4. **Compare**: Check how wrong it was
5. **Improve**: Adjust all internal parameters to be more correct
6. **Repeat**: Do this thousands of times until accurate

**Training data** comes from the **Universal Dependencies** project—professionally annotated sentences in multiple languages.

## Supported Languages

The system works with 5 languages:
- 🇬🇧 English (46,570 training sentences)
- 🇫🇷 French (14,555 training sentences)
- 🇩🇪 German (10,076 training sentences)
- 🇪🇸 Spanish (13,123 training sentences)
- 🇨🇳 Chinese (4,996 training sentences)

## How to Use

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env file with data path
echo "UD_DATA_PATH=/path/to/ud-treebanks" > .env

# 3. Train (option A: single language)
python run.py en

# 3. Train (option B: all languages)
python train_all_languages.py

# 4. Visualize results
python visualize_results.py
```

### Train on One Language

```bash
python run.py en          # English
python run.py fr          # French
python run.py de          # German
python run.py es          # Spanish
python run.py zh          # Chinese
```

### Train on All Languages

```bash
python train_all_languages.py
```

Automatically trains all 5 languages and saves metrics with timestamp.

## Understanding the Results

### Key Metrics

After training, you'll see metrics like:

- **Accuracy**: Percentage of words correctly tagged
  - 95% accuracy = 95 out of 100 words labeled correctly

- **Precision**: "When the model predicts NOUN, how often is it right?"
  - High precision = few false positives

- **Recall**: "Out of all actual NOUNs, how many did we find?"
  - High recall = few missed cases

- **F1 Score**: Balanced combination of precision and recall
  - Single number 0-1 (higher is better)
  - Most important metric for imbalanced datasets

### Example Output

```
Epoch 1/20
Train Loss:  0.8234
Dev Acc:     92.34%
Dev F1:      0.9145
Dev Prec:    0.9201
Dev Recall:  0.9087
```

**What this means**: After one pass through all training data, the model correctly tags 92% of test words, with an F1 score of 0.91 (very good).

### Per-Tag Performance

The system also shows which POS tags are hardest:
- Common tags (NOUN, VERB) usually 95%+ accuracy
- Rare tags (rare particle types) might be 70-80% accurate
- Uncommon words struggle more than common ones

## Further Improvements

Possible enhancements to the system (ordered by priority):

1. **Optimize hyperparameters using DeepHyper** - Automated hyperparameter tuning
2. **Organize functions better** - Create utility files for data, training, model
3. **Automatic Language Detection followed by POS Tagging** -Develop a chatbot that automatically identifies the language of the input text and then performs part-of-speech (POS) tagging accordingly.
