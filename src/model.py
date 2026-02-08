# src/model.py

import torch
import torch.nn as nn

class BiLSTMPOSTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=1
        )
        self.output_layer = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, word_ids):
        """
        word_ids: Tensor of shape (batch_size, seq_len)
        """
        emb = self.embedding(word_ids)
        post_lstm, _ = self.lstm(emb)
        logits = self.output_layer(post_lstm)
        return logits


# -----------------------------
# Training / Evaluation helpers
# -----------------------------

def train_epoch(model, loader, optimizer, criterion, device, n_classes):
    model.train()
    epoch_loss = 0.0
    for word_ids, tag_ids, lengths in loader:
        word_ids, tag_ids = word_ids.to(device), tag_ids.to(device)
        logits = model(word_ids)
        loss = criterion(logits.view(-1, n_classes), tag_ids.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for word_ids, tag_ids, lengths in loader:
            word_ids, tag_ids = word_ids.to(device), tag_ids.to(device)
            logits = model(word_ids)
            pred = logits.argmax(dim=-1)
            mask = tag_ids != -100  # ignore padding
            correct += (pred[mask] == tag_ids[mask]).sum().item()
            total += mask.sum().item()
    return correct / total if total else 0.0


def train_model(
    model, train_loader, dev_loader, optimizer, criterion, device, n_classes, num_epochs=10
):
    """
    Train the model and print dev accuracy after each epoch.
    """
    for epoch in range(num_epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, device, n_classes)
        acc = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}  loss={loss:.4f}  dev_acc={acc:.4f}")
