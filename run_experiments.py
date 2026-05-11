"""
Phase 8 Step 2 — Rigorous Experimental Framework
Run from the my_ids/ directory:
    python run_experiments.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from experiments.experiment_runner import ExperimentRunner

if __name__ == "__main__":
    runner = ExperimentRunner(
        results_path=os.path.join(HERE, "results.csv"),
        ground_truth_path=os.path.join(HERE, "ground_truth.csv"),
        output_dir=os.path.join(HERE, "experiments", "plots"),
    )
    runner.run_all()
