"""
Multi-lingual POS tagging training script with SOTA BiLSTM + Attention + CRF.
"""

import os
import torch
from dotenv import load_dotenv
from model import create_model
from data import load_data_and_dataloaders
from evaluate import POSEvaluator, train_and_evaluate_epoch
import config


def main(language='en', num_epochs=config.NUM_EPOCHS, learning_rate=config.LEARNING_RATE):
    """
    Main training function for multi-lingual POS tagging.
    
    Args:
        language: Language code (e.g., 'en', 'fr', 'de')
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
    
    Returns:
        dict: Training history and best metrics
    """
    load_dotenv()
    DATA_PATH = os.getenv("UD_DATA_PATH")
    
    if not DATA_PATH:
        raise ValueError("UD_DATA_PATH environment variable not set. Please create a .env file.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training on {config.LANGUAGES[language]['name']} data")
    print()
    
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    train_loader, dev_loader, test_loader, word2idx, tag2idx, char2idx = load_data_and_dataloaders(
        DATA_PATH, language=language, batch_size=config.BATCH_SIZE, use_chars=True
    )
    print()
    
    print("=" * 60)
    print("BUILDING MODEL")
    print("=" * 60)
    vocab_size = len(word2idx)
    char_vocab_size = len(char2idx) if char2idx else 100
    num_tags = len(tag2idx)
    
    model = create_model(vocab_size, char_vocab_size, num_tags, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY)
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    best_dev_f1 = 0.0
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    
    training_history = {
        'epochs': [],
        'train_loss': [],
        'dev_accuracy': [],
        'dev_precision': [],
        'dev_recall': [],
        'dev_f1': []
    }
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)
        
        train_loss, dev_metrics = train_and_evaluate_epoch(
            model, train_loader, dev_loader, optimizer, device, 
            criterion=criterion, use_crf=config.USE_CRF
        )
        
        dev_f1 = dev_metrics['f1']
        dev_acc = dev_metrics['accuracy']
        
        training_history['epochs'].append(epoch + 1)
        training_history['train_loss'].append(float(train_loss))
        training_history['dev_accuracy'].append(float(dev_acc))
        training_history['dev_precision'].append(float(dev_metrics['precision']))
        training_history['dev_recall'].append(float(dev_metrics['recall']))
        training_history['dev_f1'].append(float(dev_f1))
        
        print(f"Train Loss:  {train_loss:.4f}")
        print(f"Dev Acc:     {dev_acc:.4f}")
        print(f"Dev F1:      {dev_f1:.4f}")
        print(f"Dev Prec:    {dev_metrics['precision']:.4f}")
        print(f"Dev Recall:  {dev_metrics['recall']:.4f}")
        
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            patience_counter = 0
            print(f"New best model found (F1: {best_dev_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping after {config.EARLY_STOPPING_PATIENCE} epochs without improvement")
                break
    
    model_path = os.path.join(config.MODELS_DIR, f"best_model_{language}.pt")
    torch.save(best_model_state, model_path)
    print(f"\nBest model saved to {model_path} (F1: {best_dev_f1:.4f})")
    
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    
    model.load_state_dict(best_model_state)
    evaluator = POSEvaluator({v: k for k, v in tag2idx.items()})
    test_metrics = evaluator.evaluate(model, test_loader, device, use_crf=config.USE_CRF)
    
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1:        {test_metrics['f1']:.4f}")
    
    print("\nPer-tag F1 scores:")
    for tag, f1 in sorted(test_metrics['per_tag_f1'].items(), key=lambda x: x[1], reverse=True):
        if tag != '<PAD>':
            print(f"  {tag:12s}: {f1:.4f}")
    
    return {
        'training_history': training_history,
        'best_epoch': best_epoch,
        'best_dev_f1': float(best_dev_f1),
        'test_metrics': {
            'test_accuracy': float(test_metrics['accuracy']),
            'test_precision': float(test_metrics['precision']),
            'test_recall': float(test_metrics['recall']),
            'test_f1': float(test_metrics['f1']),
            'per_tag_f1': {k: float(v) for k, v in test_metrics['per_tag_f1'].items()}
        }
    }


if __name__ == "__main__":
    import sys
    
    language = sys.argv[1] if len(sys.argv) > 1 else 'en'
    main(language=language)