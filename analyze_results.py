import json
import sys
from pathlib import Path
from datetime import datetime
import statistics


class ResultsAnalyzer:    
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        
        if not self.results_dir.exists():
            print("No results directory found. Train first!")
            sys.exit(1)
    
    def list_all_runs(self):
        runs = sorted([d for d in self.results_dir.iterdir() if d.is_dir()])
        
        if not runs:
            print("No training runs found.")
            return
        
        print(f"\nAvailable Training Runs (Total: {len(runs)})")
        print("=" * 70)
        
        for i, run_dir in enumerate(runs, 1):
            summary_file = run_dir / "training_summary.json"
            
            if summary_file.exists():
                with open(summary_file) as f:
                    data = json.load(f)
                    timestamp = data.get('timestamp', 'unknown')
                    successful = data['summary'].get('successful_trainings', 0)
                    failed = data['summary'].get('failed_trainings', 0)
                    total_hours = data['summary'].get('total_training_time_hours', 0)
                    
                    print(f"\n{i}. {run_dir.name}")
                    print(f"  {timestamp}")
                    print(f"  Successful: {successful} | ❌ Failed: {failed}")
                    print(f"    Total time: {total_hours} hours")
        
        print("\n" + "=" * 70)
    
    def analyze_run(self, run_id=None):
        if run_id is None:
            runs = sorted([d for d in self.results_dir.iterdir() if d.is_dir()])
            if not runs:
                print("No training runs found.")
                return
            run_dir = runs[-1]
        else:
            run_dir = self.results_dir / run_id
            if not run_dir.exists():
                print(f"Run '{run_id}' not found.")
                return
        
        summary_file = run_dir / "training_summary.json"
        if not summary_file.exists():
            print(f"Summary file not found in {run_dir}")
            return
        
        with open(summary_file) as f:
            results = json.load(f)
        
        print(f"\n{'='*70}")
        print(f"TRAINING RUN ANALYSIS")
        print(f"{'='*70}")
        print(f"Run ID: {run_dir.name}")
        print(f"Timestamp: {results['timestamp']}")
        
        summary = results['summary']
        print(f"\n{'─'*70}")
        print(f"SUMMARY")
        print(f"{'─'*70}")
        print(f"Total languages:        {summary['total_languages']}")
        print(f"Successful trainings:   {summary['successful_trainings']}")
        print(f"Failed trainings:       {summary['failed_trainings']}")
        print(f"Total training time:    {summary['total_training_time_hours']} hours")
        
        # Display per-language results
        print(f"\n{'─'*70}")
        print(f"PER-LANGUAGE RESULTS")
        print(f"{'─'*70}")
        
        languages = results['languages']
        
        for lang, result in sorted(languages.items()):
            status_icon = "" if result['status'] == 'completed' else "❌"
            lang_name = result.get('language_name', lang)
            
            print(f"\n{status_icon} {lang_name.upper()} ({lang})")
            print(f"   Status:  {result['status']}")
            print(f"   Time:    {result.get('elapsed_time_minutes', 'N/A')} minutes")
            
            if result['status'] == 'failed':
                print(f"   Error:   {result.get('error', 'Unknown error')}")
        
        times = [r.get('elapsed_time_minutes', 0) for r in languages.values() if r['status'] == 'completed']
        
        if times:
            print(f"\n{'─'*70}")
            print(f" TIME STATISTICS (completed trainings only)")
            print(f"{'─'*70}")
            print(f"Average time:  {statistics.mean(times):.2f} minutes")
            print(f"Min time:      {min(times):.2f} minutes")
            print(f"Max time:      {max(times):.2f} minutes")
            print(f"Median time:   {statistics.median(times):.2f} minutes")
        
        # File structure
        print(f"\n{'─'*70}")
        print(f"OUTPUT STRUCTURE")
        print(f"{'─'*70}")
        print(f"Results directory: {run_dir}")
        
        # List subdirectories
        logs_dir = run_dir / "logs"
        models_dir = run_dir / "models"
        
        if logs_dir.exists():
            print(f"  Logs: {len(list(logs_dir.glob('*')))} files")
        if models_dir.exists():
            print(f"  Models: {len(list(models_dir.glob('*')))} files")
        
        print(f"  Summary: {summary_file}")
        
        print(f"\n{'='*70}\n")
    
    def compare_runs(self, run_ids=None):
        if run_ids is None:
            run_dirs = sorted([d for d in self.results_dir.iterdir() if d.is_dir()])
        else:
            run_dirs = [self.results_dir / rid for rid in run_ids]
        
        if not run_dirs:
            print(" No runs found.")
            return
        
        print(f"\n{'='*70}")
        print(f" COMPARING {len(run_dirs)} TRAINING RUNS")
        print(f"{'='*70}\n")
        
        all_results = {}
        for run_dir in run_dirs:
            summary_file = run_dir / "training_summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    all_results[run_dir.name] = json.load(f)
        
        print(f"{'Run ID':<20} {'Successful':<15} {'Failed':<15} {'Total Hours':<15}")
        print("─" * 70)
        
        for run_id, results in sorted(all_results.items()):
            summary = results['summary']
            successful = summary.get('successful_trainings', 0)
            failed = summary.get('failed_trainings', 0)
            hours = summary.get('total_training_time_hours', 0)
            
            print(f"{run_id:<20} {successful:<15} {failed:<15} {hours:<15.2f}")
        
        print("=" * 70 + "\n")


def main():
    analyzer = ResultsAnalyzer()
    
    if len(sys.argv) < 2:
        analyzer.analyze_run()
    elif sys.argv[1] == "--all":
        analyzer.list_all_runs()
    elif sys.argv[1] == "--compare":
        run_ids = sys.argv[2:] if len(sys.argv) > 2 else None
        analyzer.compare_runs(run_ids)
    else:
        analyzer.analyze_run(sys.argv[1])


if __name__ == "__main__":
    main()
