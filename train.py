import os
import torch
from dotenv import load_dotenv
from model import create_model
from data import load_data_and_dataloaders
from evaluate import POSEvaluator, train_and_evaluate_epoch
import config


def main(language='en', num_epochs=config.NUM_EPOCHS, learning_rate=config.LEARNING_RATE):
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
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)

        train_loss, dev_metrics = train_and_evaluate_epoch(
            model, train_loader, dev_loader, optimizer, device, 
            criterion=criterion, use_crf=config.USE_CRF
        )
        
        dev_f1 = dev_metrics['f1']
        dev_acc = dev_metrics['accuracy']
        
        print(f"Train Loss:  {train_loss:.4f}")
        print(f"Dev Acc:     {dev_acc:.4f}")
        print(f"Dev F1:      {dev_f1:.4f}")
        print(f"Dev Prec:    {dev_metrics['precision']:.4f}")
        print(f"Dev Recall:  {dev_metrics['recall']:.4f}")
        
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            patience_counter = 0
            
            model_path = os.path.join(
                config.MODELS_DIR,
                config.MODEL_NAME_TEMPLATE.format(language=language, epoch=epoch+1)
            )
            torch.save(model.state_dict(), model_path)
            print(f"✓ Model saved to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\n⚠ Early stopping after {config.EARLY_STOPPING_PATIENCE} epochs without improvement")
                break
    
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    
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


if __name__ == "__main__":
    import sys
    
    language = sys.argv[1] if len(sys.argv) > 1 else 'en'
    main(language=language)
