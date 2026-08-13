# Organized Scene Pipeline

This folder contains a preprocessing and RandomForest training pipeline for the prior-authorization export dataset.

## Files
- `scripts/preprocess_and_train.py`: preprocesses `export.csv`, derives `decision` from `Approval rate`, splits train/test, trains a RandomForest model, and saves artifacts.
- `scripts/predict_decision.py`: scores a CSV or single-row JSON file.
- `scripts/predict_interactive.py`: prints a terminal input form and predicts from typed values.
- `scripts/run_pipeline.py`: trains and scores a CSV in one command.
- `scripts/compare_with_original.py`: compares predictions against the original label.
- `processed/`: cleaned data output.
- `models/`: trained model and metrics.
- `predictions/`: saved prediction outputs.
- `raw/`: place raw inputs here if you want to copy them into the organized layout.

## Run
From the workspace root:

```powershell
python organized_scene\scripts\preprocess_and_train.py
```

By default, `decision` is derived as `approved` when `Approval rate >= 50%` and `denied` otherwise.

## Score Another CSV

After training, score a different CSV with the saved model:

```powershell
python organized_scene\scripts\predict_decision.py path\to\other_file.csv
```

The predictions are saved to `organized_scene\predictions\predicted_decisions.csv` unless you pass `--output`. The scorer also accepts a single-row JSON file.

## Train and Score in One Command

If you want to train and immediately score the same CSV in one step:

```powershell
python organized_scene\scripts\run_pipeline.py path\to\your_file.csv --output organized_scene\predictions\your_predictions.csv
```

## Predict From One JSON Input

If you want to test one record manually, use a JSON file like [input_sample.json](input_sample.json):

```powershell
python organized_scene\scripts\predict_from_json.py organized_scene\input_sample.json
```

This works with the current `export.csv` feature schema. The `Patient_Age_At_Request` style fields you showed earlier are not compatible with this trained model yet and would need a separate dataset/schema.

## Compare Predictions With the Original Label

To remove the label from the input features, predict, and compare against the original label:

```powershell
python organized_scene\scripts\compare_with_original.py export.csv --output organized_scene\predictions\comparison.csv
```

If `decision` is not present, the script derives the original label from `Approval rate` using the same 50% rule as training.

## Interactive Input

To type values directly in the terminal and predict:

```powershell
python organized_scene\scripts\predict_interactive.py
```

The script prints an input section, lets you type each field, and then shows `predicted_decision` plus class probabilities.
