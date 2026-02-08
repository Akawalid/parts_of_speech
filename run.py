# run.py
# %%
"""
Entry point for training the POS tagging model.
Simply calls the main() function from src.train.
"""

import sys
import os

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.train import main

if __name__ == "__main__":
    main()
