import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
import torch

import config
from train import main as train_single_language


class MultiLanguageTrainer:    
    def __init__(self):
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.results_dir / self.timestamp
        self.run_dir.mkdir(exist_ok=True)
        
        self.metrics_dir = self.run_dir / "metrics"
        self.logs_dir = self.run_dir / "logs"

        self.metrics_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        
        self.summary_file = self.run_dir / "training_summary.json"
        self.results = {
            'timestamp': self.timestamp,
            'languages': {},
            'summary': {}
        }
        
        print(f"\nResults directory: {self.run_dir}")
    
    def train_language(self, language: str) -> dict:
        print(f"\n{'='*70}")
        print(f"Training on {config.LANGUAGES[language]['name'].upper()}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        try:
            metrics = train_single_language(language=language)
            
            elapsed_time = time.time() - start_time
            
            result = {
                'status': 'completed',
                'language': language,
                'language_name': config.LANGUAGES[language]['name'],
                'elapsed_time_seconds': elapsed_time,
                'elapsed_time_minutes': round(elapsed_time / 60, 2),
                'timestamp': datetime.now().isoformat()
            }
            
            if metrics:
                result['training_history'] = metrics['training_history']
                result['best_epoch'] = metrics['best_epoch']
                result['best_dev_f1'] = metrics['best_dev_f1']
                result['test_metrics'] = metrics['test_metrics']
            
            print(f"\n{language.upper()} training completed in {elapsed_time/60:.2f} minutes")
            if metrics:
                print(f"  Best epoch:    {metrics['best_epoch']}")
                print(f"  Best dev F1:   {metrics['best_dev_f1']:.4f}")
                print(f"  Test Accuracy: {metrics['test_metrics']['test_accuracy']:.4f}")
                print(f"  Test F1:       {metrics['test_metrics']['test_f1']:.4f}")
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            
            result = {
                'status': 'failed',
                'language': language,
                'language_name': config.LANGUAGES[language]['name'],
                'error': str(e),
                'elapsed_time_seconds': elapsed_time,
                'elapsed_time_minutes': round(elapsed_time / 60, 2),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"\n{language.upper()} training FAILED: {e}")
        
        return result
    
    def train_all(self, languages=None):
        if languages is None:
            languages = list(config.LANGUAGES.keys())
        
        print(f"\nStarting multi-language training run")
        print(f"Timestamp: {self.timestamp}")
        print(f"Languages: {', '.join(languages)}")
        print(f"Results saved to: {self.run_dir}")
        
        for lang in languages:
            if lang not in config.LANGUAGES:
                print(f"WARNING: Language '{lang}' not supported. Skipping.")
                continue
            
            result = self.train_language(lang)
            self.results['languages'][lang] = result
        
        self._generate_summary()
        self._save_results()
    
    def _generate_summary(self):
        results = self.results['languages']
        
        successful = [r for r in results.values() if r['status'] == 'completed']
        failed = [r for r in results.values() if r['status'] == 'failed']
        
        total_time = sum(r.get('elapsed_time_seconds', 0) for r in results.values())
        
        avg_metrics = {}
        if successful:
            metrics_keys = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
            for key in metrics_keys:
                values = [r['test_metrics'][key] for r in successful if 'test_metrics' in r]
                if values:
                    avg_metrics[f'avg_{key}'] = round(sum(values) / len(values), 4)
        
        self.results['summary'] = {
            'total_languages': len(results),
            'successful_trainings': len(successful),
            'failed_trainings': len(failed),
            'total_training_time_seconds': total_time,
            'total_training_time_hours': round(total_time / 3600, 2),
            'successful_languages': [r['language'] for r in successful],
            'failed_languages': [r['language'] for r in failed],
            **avg_metrics
        }
    
    def _save_results(self):
        with open(self.summary_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        for lang, result in self.results['languages'].items():
            if result['status'] == 'completed':
                metrics_data = {
                    'language': lang,
                    'language_name': result['language_name'],
                    'training_history': result.get('training_history', {}),
                    'best_epoch': result.get('best_epoch'),
                    'best_dev_f1': result.get('best_dev_f1'),
                    'test_metrics': result.get('test_metrics', {})
                }
                
                metrics_file = self.metrics_dir / f"{lang}_metrics.json"
                with open(metrics_file, 'w') as f:
                    json.dump(metrics_data, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"TRAINING SUMMARY")
        print(f"{'='*70}")
        print(f"Total languages trained: {self.results['summary']['total_languages']}")
        print(f"Successful: {self.results['summary']['successful_trainings']}")
        print(f"Failed: {self.results['summary']['failed_trainings']}")
        print(f"Total time: {self.results['summary']['total_training_time_hours']} hours")
        
        if 'avg_test_f1' in self.results['summary']:
            print(f"\nAverage metrics across languages:")
            print(f"  Accuracy:  {self.results['summary']['avg_test_accuracy']:.4f}")
            print(f"  Precision: {self.results['summary']['avg_test_precision']:.4f}")
            print(f"  Recall:    {self.results['summary']['avg_test_recall']:.4f}")
            print(f"  F1:        {self.results['summary']['avg_test_f1']:.4f}")
        
        print(f"\nSuccessful: {', '.join(self.results['summary']['successful_languages'])}")
        if self.results['summary']['failed_languages']:
            print(f"Failed: {', '.join(self.results['summary']['failed_languages'])}")
        print(f"\nResults saved to: {self.summary_file}")
        print(f"{'='*70}\n")


def main():
    load_dotenv()
    DATA_PATH = os.getenv("UD_DATA_PATH")
    
    if not DATA_PATH:
        raise ValueError("UD_DATA_PATH environment variable not set. Please create a .env file.")
    
    if len(sys.argv) > 1:
        languages = sys.argv[1:]
    else:
        languages = list(config.LANGUAGES.keys())
    
    trainer = MultiLanguageTrainer()
    
    trainer.train_all(languages)


if __name__ == "__main__":
    main()