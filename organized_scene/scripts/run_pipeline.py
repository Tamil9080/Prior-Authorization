import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"d:\cts hackthon\new1\organized_scene")
TRAIN_SCRIPT = ROOT / "scripts" / "preprocess_and_train.py"
PREDICT_SCRIPT = ROOT / "scripts" / "predict_decision.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the model and score a CSV in one command.")
    parser.add_argument("input_csv", help="CSV file to train on and score")
    parser.add_argument("--output", default=str(ROOT / "predictions" / "pipeline_predictions.csv"), help="Prediction output path")
    args = parser.parse_args()

    train_command = [sys.executable, str(TRAIN_SCRIPT), "--source", args.input_csv]
    predict_command = [sys.executable, str(PREDICT_SCRIPT), args.input_csv, "--output", args.output]

    print("Training model...")
    train_result = subprocess.run(train_command, check=False)
    if train_result.returncode != 0:
        return train_result.returncode

    print("Scoring input CSV...")
    predict_result = subprocess.run(predict_command, check=False)
    return predict_result.returncode


if __name__ == "__main__":
    raise SystemExit(main())