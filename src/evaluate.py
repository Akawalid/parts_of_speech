import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


class POSEvaluator:    
    def __init__(self, idx2tag: Dict[int, str]):
        self.idx2tag = idx2tag
        self.tag2idx = {v: k for k, v in idx2tag.items()}
    
    def evaluate(self, model, loader, device, use_crf=False) -> Dict:
        model.eval()
        all_preds = []
        all_true = []
        all_lengths = []
        
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 4:
                    word_ids, tag_ids, lengths, char_ids = batch
                    char_ids = char_ids.to(device)
                else:
                    word_ids, tag_ids, lengths = batch
                    char_ids = None
                
                word_ids = word_ids.to(device)
                tag_ids = tag_ids.to(device)
                
                if char_ids is not None:
                    logits = model(word_ids, char_ids, lengths)
                else:
                    logits = model(word_ids, lengths=lengths)
                
                if use_crf and hasattr(model, 'crf') and model.crf is not None:
                    preds = model.crf.decode(logits)
                    preds = torch.tensor(preds, device=device)
                else:
                    preds = torch.argmax(logits, dim=-1)
                
                for i, length in enumerate(lengths):
                    length = length.item()
                    all_preds.extend(preds[i, :length].cpu().tolist())
                    all_true.extend(tag_ids[i, :length].cpu().tolist())
        
        metrics = {
            'accuracy': self._accuracy(all_true, all_preds),
            'precision': self._precision(all_true, all_preds),
            'recall': self._recall(all_true, all_preds),
            'f1': self._f1(all_true, all_preds),
            'per_tag_f1': self._per_tag_f1(all_true, all_preds),
        }
        
        return metrics
    
    def _accuracy(self, true: List[int], pred: List[int]) -> float:
        if len(true) == 0:
            return 0.0
        correct = sum(1 for t, p in zip(true, pred) if t == p)
        return correct / len(true)
    
    def _precision(self, true: List[int], pred: List[int]) -> float:
        if len(true) == 0:
            return 0.0
        return precision_score(true, pred, average='macro', zero_division=0)
    
    def _recall(self, true: List[int], pred: List[int]) -> float:
        if len(true) == 0:
            return 0.0
        return recall_score(true, pred, average='macro', zero_division=0)
    
    def _f1(self, true: List[int], pred: List[int]) -> float:
        if len(true) == 0:
            return 0.0
        return f1_score(true, pred, average='macro', zero_division=0)
    
    def _per_tag_f1(self, true: List[int], pred: List[int]) -> Dict[str, float]:
        per_tag_f1 = {}
        for tag_idx, tag_name in self.idx2tag.items():
            if tag_idx == 0:
                continue
            try:
                f1 = f1_score(
                    [1 if t == tag_idx else 0 for t in true],
                    [1 if p == tag_idx else 0 for p in pred],
                    average='binary',
                    zero_division=0
                )
                per_tag_f1[tag_name] = f1
            except:
                pass
        return per_tag_f1


def train_and_evaluate_epoch(model, train_loader, dev_loader, optimizer, device, criterion=None, use_crf=False):
    model.train()
    train_loss = 0.0
    
    for batch in train_loader:
        if len(batch) == 4:
            word_ids, tag_ids, lengths, char_ids = batch
            char_ids = char_ids.to(device)
        else:
            word_ids, tag_ids, lengths = batch
            char_ids = None
        
        word_ids = word_ids.to(device)
        tag_ids = tag_ids.to(device)
        
        if char_ids is not None:
            logits = model(word_ids, char_ids, lengths)
        else:
            logits = model(word_ids, lengths=lengths)
        
        if use_crf and hasattr(model, 'crf') and model.crf is not None:
            mask = (tag_ids != 0).float()
            loss = model.crf(logits, tag_ids, mask).mean()
        else:
            logits_flat = logits.view(-1, logits.shape[-1])
            tag_ids_flat = tag_ids.view(-1)
            loss = criterion(logits_flat, tag_ids_flat)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    evaluator = POSEvaluator({v: k for k, v in model.idx2tag.items()} if hasattr(model, 'idx2tag') else {i: str(i) for i in range(model.num_tags)})
    dev_metrics = evaluator.evaluate(model, dev_loader, device, use_crf=use_crf)
    
    return train_loss, dev_metrics
