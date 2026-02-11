import os
import pickle
from typing import Dict, Tuple, Optional
from config import LANGUAGES


class LanguageManager:
    def __init__(self, base_data_path: str):
        self.base_path = base_data_path
        self.loaded_vocabs: Dict[str, Dict] = {}  # Cache vocabulary for each language
        self.language_paths: Dict[str, str] = {}  # Cache computed paths
        
    def get_language_path(self, language: str) -> str:
        if language not in LANGUAGES:
            raise ValueError(f"Language '{language}' not supported. Available: {list(LANGUAGES.keys())}")
        
        if language in self.language_paths:
            return self.language_paths[language]
        
        treebank = LANGUAGES[language]['treebank']
        path = os.path.join(self.base_path, treebank)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Language treebank directory not found: {path}")
        
        self.language_paths[language] = path
        return path
    
    def get_file_paths(self, language: str) -> Dict[str, str]:
        lang_path = self.get_language_path(language)
        prefix = LANGUAGES[language]['prefix']
        
        return {
            'train': os.path.join(lang_path, f"{prefix}-train.conllu"),
            'dev': os.path.join(lang_path, f"{prefix}-dev.conllu"),
            'test': os.path.join(lang_path, f"{prefix}-test.conllu"),
        }
    
    def cache_vocab(self, language: str, word2idx: Dict, tag2idx: Dict, char2idx: Dict) -> None:
        self.loaded_vocabs[language] = {
            'word2idx': word2idx,
            'tag2idx': tag2idx,
            'char2idx': char2idx,
        }
    
    def get_cached_vocab(self, language: str) -> Optional[Dict]:
        return self.loaded_vocabs.get(language)
    
    def save_vocab(self, language: str, word2idx: Dict, tag2idx: Dict, char2idx: Dict, save_path: str) -> None:
        data = {
            'language': language,
            'word2idx': word2idx,
            'tag2idx': tag2idx,
            'char2idx': char2idx,
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load_vocab(self, save_path: str) -> Optional[str]:
        if not os.path.exists(save_path):
            return None
        
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
        
        language = data['language']
        self.loaded_vocabs[language] = {
            'word2idx': data['word2idx'],
            'tag2idx': data['tag2idx'],
            'char2idx': data['char2idx'],
        }
        
        return language
    
    @staticmethod
    def is_language_available(language: str) -> bool:
        return language in LANGUAGES
    
    @staticmethod
    def get_language_name(language: str) -> str:
        if language not in LANGUAGES:
            return "Unknown"
        return LANGUAGES[language]['name']
    
    @staticmethod
    def get_all_languages() -> Dict[str, str]:
        return {code: LANGUAGES[code]['name'] for code in LANGUAGES}
