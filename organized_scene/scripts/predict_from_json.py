import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(r"d:\cts hackthon\new1\organized_scene")
MODEL_PATH = ROOT / "models" / "random_forest_model.joblib"


def align_input(df: pd.DataFrame, expected_features: list[str]) -> pd.DataFrame:
    aligned = df.copy()

    for column in expected_features:
        if column not in aligned.columns:
            aligned[column] = pd.NA

    return aligned[expected_features]


def normalize_numeric_columns(df: pd.DataFrame, model) -> pd.DataFrame:
    numeric_features = []
    for name, transformer, columns in model.named_steps["preprocessor"].transformers_:
        if name == "num":
            numeric_features = list(columns)
            break

    normalized = df.copy()
    for column in numeric_features:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(
                normalized[column].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict decision from a single JSON input row.")
    parser.add_argument("input_json", help="Path to a JSON file containing one record")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Path to the trained model")
    args = parser.parse_args()

    model = joblib.load(Path(args.model))
    input_path = Path(args.input_json)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("Input JSON must be a single object with field names and values.")

    row = pd.DataFrame([data])
    expected_features = list(model.named_steps["preprocessor"].feature_names_in_)
    if "decision" in expected_features:
        expected_features = [column for column in expected_features if column != "decision"]

    aligned = align_input(row, expected_features)
    aligned = normalize_numeric_columns(aligned, model)

    prediction = model.predict(aligned)[0]
    output = {"predicted_decision": prediction}

    if hasattr(model.named_steps["classifier"], "predict_proba"):
        probabilities = model.predict_proba(aligned)[0]
        classes = list(model.named_steps["classifier"].classes_)
        output["probabilities"] = {str(class_name): float(probabilities[index]) for index, class_name in enumerate(classes)}

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())