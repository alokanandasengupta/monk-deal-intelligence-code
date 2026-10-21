"""Trains and validates the prediction model (grouped-by-creator folds, calibrated 80% ranges). Writes outputs/pitch_model.joblib and outputs/model_results.json."""
import os, runpy
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
runpy.run_path(os.path.join(ROOT, "src", "pitch_model.py"), run_name="__main__")
