import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.train import main

if __name__ == "__main__":
    language = sys.argv[1] if len(sys.argv) > 1 else 'en'
    main(language=language)
