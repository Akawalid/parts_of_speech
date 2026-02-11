import json
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta


class TrainingMonitor:    
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
    
    def get_recent_run(self):
        if not self.results_dir.exists():
            return None
        
        runs = sorted([d for d in self.results_dir.iterdir() if d.is_dir()])
        return runs[-1] if runs else None
    
    def get_run_status(self, run_dir):
        summary_file = run_dir / "training_summary.json"
        
        if not summary_file.exists():
            return None
        
        with open(summary_file) as f:
            return json.load(f)
    
    def get_elapsed_time(self, run_dir):
        if not run_dir.exists():
            return None
        
        try:
            run_time = datetime.strptime(run_dir.name, "%Y%m%d_%H%M%S")
            elapsed = datetime.now() - run_time
            return elapsed
        except:
            return None
    
    def display_status(self, run_dir=None):
        if run_dir is None:
            run_dir = self.get_recent_run()
        
        if run_dir is None:
            print("No training runs found.")
            return
        
        if isinstance(run_dir, str):
            run_dir = Path(run_dir)
        
        status = self.get_status(run_dir)
        elapsed = self.get_elapsed_time(run_dir)
        
        print("\n" + "="*70)
        print("TRAINING MONITOR DASHBOARD")
        print("="*70)
        print(f"Run: {run_dir.name}")
        print(f"Elapsed: {elapsed if elapsed else 'N/A'}")
        
        if status is None:
            print("Training in progress (summary not yet generated)...")
        else:
            self._display_summary(status)
        
        print("="*70 + "\n")
    
    def _display_summary(self, status):
        summary = status.get('summary', {})
        languages = status.get('languages', {})
        
        print(f"\n PROGRESS:")
        print(f"   Total languages:     {summary.get('total_languages', 0)}")
        print(f"   Completed:        {summary.get('successful_trainings', 0)}")
        print(f"   Failed:           {summary.get('failed_trainings', 0)}")
        print(f"   In progress:      {summary.get('total_languages', 0) - summary.get('successful_trainings', 0) - summary.get('failed_trainings', 0)}")
        
        print(f"\nLANGUAGE STATUS:")
        for lang, result in sorted(languages.items()):
            status_icon = "" if result['status'] == 'completed' else "" if result['status'] == 'in_progress' else ""
            time_display = f"{result.get('elapsed_time_minutes', 'N/A')} min" if 'elapsed_time_minutes' in result else 'Running...'
            
            lang_name = result.get('language_name', lang).upper()
            print(f"   {status_icon} {lang_name:<10} {time_display}")
        
        total_hours = summary.get('total_training_time_hours', 0)
        print(f"\n TOTAL TIME: {total_hours} hours")
    
    def live_monitor(self, update_interval=30):
        run_dir = self.get_recent_run()
        
        if run_dir is None:
            print("No training runs found. Start training first!")
            return
        
        print("Live Monitoring Started")
        print(f"Updates every {update_interval} seconds (Ctrl+C to stop)")
        
        try:
            while True:
                self.display_status(run_dir)
                time.sleep(update_interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")


def main():
    monitor = TrainingMonitor()
    
    if len(sys.argv) < 2:
        monitor.live_monitor()
    elif sys.argv[1] == "--once":
        monitor.display_status()
    else:
        monitor.display_status(sys.argv[1])


if __name__ == "__main__":
    main()
