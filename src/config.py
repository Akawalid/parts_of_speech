LANGUAGES = {
    'en': {
        'treebank': 'UD_English-EWT',
        'name': 'English',
        'prefix': 'en_ewt-ud',
        'iso_code': 'eng'
    },
    'fr': {
        'treebank': 'UD_French-GSD',
        'name': 'French',
        'prefix': 'fr_gsd-ud',
        'iso_code': 'fra'
    },
    'de': {
        'treebank': 'UD_German-GSD',
        'name': 'German',
        'prefix': 'de_gsd-ud',
        'iso_code': 'deu'
    },
    'es': {
        'treebank': 'UD_Spanish-AnCora',
        'name': 'Spanish',
        'prefix': 'es_ancora-ud',
        'iso_code': 'spa'
    },
    'zh': {
        'treebank': 'UD_Chinese-GSD',
        'name': 'Chinese',
        'prefix': 'zh_gsd-ud',
        'iso_code': 'zho'
    },
}

DEFAULT_LANGUAGE = 'en'

EMBEDDING_DIM = 256  
CHAR_EMBEDDING_DIM = 50  

CHAR_KERNEL_SIZES = [3, 4, 5]
CHAR_NUM_FILTERS = 25

HIDDEN_DIM = 256  
NUM_LSTM_LAYERS = 2
BIDIRECTIONAL = True

USE_ATTENTION = True
ATTENTION_DIM = 256

USE_CRF = False

DROPOUT_RATE = 0.4
CHAR_DROPOUT_RATE = 0.3
LAYER_NORM = True


BATCH_SIZE =32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP = 1.0

NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
EVAL_EVERY_N_BATCHES = 100

MIN_WORD_FREQ = 2
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

MODELS_DIR = "models"
CHECKPOINTS_DIR = "checkpoints"
LOGS_DIR = "logs"

MODEL_NAME_TEMPLATE = "{language}_pos_tagger_epoch{epoch}.pt"
VOCAB_NAME_TEMPLATE = "{language}_vocabs.pkl"
