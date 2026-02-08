# src/train.py
import os
import torch
from dotenv import load_dotenv
from src.model import BiLSTMPOSTagger, train_model, evaluate
from src.data import load_data_and_dataloaders

def main():
    # Load environment
    load_dotenv()
    DATA_PATH = os.getenv("UD_DATA_PATH")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data and dataloaders
    train_loader, dev_loader, test_loader, word2idx, tag2idx = load_data_and_dataloaders(DATA_PATH)
    
    # Build model
    vocab_size = len(word2idx)
    n_classes  = len(tag2idx)
    model = BiLSTMPOSTagger(vocab_size, 128, 128, n_classes).to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    # Train
    train_model(model, train_loader, dev_loader, optimizer, criterion, device, n_classes, num_epochs=10)
    
    # Evaluate
    test_acc = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.4f}")