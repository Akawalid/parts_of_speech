import json
import csv
import sys
from pathlib import Path


class MetricsExporter:
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        
        if not self.results_dir.exists():
            print(" No results directory found.")
            sys.exit(1)
    
    def export_run(self, run_id=None):
        if run_id is None:
            runs = sorted([d for d in self.results_dir.iterdir() if d.is_dir()])
            if not runs:
                print("No runs found.")
                return
            run_dir = runs[-1]
        else:
            run_dir = self.results_dir / run_id
            if not run_dir.exists():
                print(f"Run '{run_id}' not found.")
                return
        
        summary_file = run_dir / "training_summary.json"
        if not summary_file.exists():
            print(f"Summary not found in {run_dir}")
            return
        
        with open(summary_file) as f:
            results = json.load(f)
        
        csv_file = run_dir / "training_results.csv"
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            writer.writerow([
                'Language',
                'Status',
                'Time (minutes)',
                'Time (seconds)',
                'Timestamp',
                'Error'
            ])
            
            for lang, result in sorted(results['languages'].items()):
                writer.writerow([
                    result.get('language_name', lang),
                    result['status'],
                    result.get('elapsed_time_minutes', 'N/A'),
                    result.get('elapsed_time_seconds', 'N/A'),
                    result.get('timestamp', 'N/A'),
                    result.get('error', '')
                ])
        
        print(f" Exported to: {csv_file}")
    
    def export_all_runs(self):
        runs = sorted([d for d in self.results_dir.iterdir() if d.is_dir()])
        
        if not runs:
            print("No runs found.")
            return
        
        csv_file = self.results_dir / "all_runs_comparison.csv"
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Run ID',
                'Total Languages',
                'Successful',
                'Failed',
                'Total Hours',
                'Timestamp'
            ])
            
            for run_dir in runs:
                summary_file = run_dir / "training_summary.json"
                if summary_file.exists():
                    with open(summary_file) as sf:
                        results = json.load(sf)
                        summary = results['summary']
                        
                        writer.writerow([
                            run_dir.name,
                            summary.get('total_languages', 0),
                            summary.get('successful_trainings', 0),
                            summary.get('failed_trainings', 0),
                            summary.get('total_training_time_hours', 0),
                            results.get('timestamp', 'N/A')
                        ])
        
        print(f"Exported all runs to: {csv_file}")


def main():
    exporter = MetricsExporter()
    
    if len(sys.argv) < 2:
        exporter.export_run()
    elif sys.argv[1] == "--all":
        exporter.export_all_runs()
    else:
        exporter.export_run(sys.argv[1])


if __name__ == "__main__":
    main()
