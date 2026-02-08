# run.py
import torch
from src.model import BiLSTMPOSTagger, train_model
from src.data import get_dataloaders, build_vocab_and_labels

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
train_loader, dev_loader, word_vocab, label_encoder = get_dataloaders()

# Model
vocab_size = len(word_vocab)
n_classes = len(label_encoder.classes_)
emb_dim = 128
hidden_dim = 128

model = BiLSTMPOSTagger(vocab_size, emb_dim, hidden_dim, n_classes).to(device)

# Optimizer & loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

# Train
train_model(model, train_loader, dev_loader, optimizer, criterion, device, n_classes, num_epochs=10)
