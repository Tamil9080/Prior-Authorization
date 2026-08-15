import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "random_forest_model.joblib"
DEFAULT_OUTPUT = ROOT / "predictions" / "predicted_decisions.csv"


def get_expected_features(model) -> list[str]:
    expected_features = list(model.named_steps["preprocessor"].feature_names_in_)
    if "decision" in expected_features:
        expected_features = [column for column in expected_features if column != "decision"]
    return expected_features


def align_input(df: pd.DataFrame, expected_features: list[str]) -> pd.DataFrame:
    aligned = df.copy()

    for column in expected_features:
        if column not in aligned.columns:
            aligned[column] = pd.NA

    aligned = aligned[expected_features]
    return aligned


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


def load_input_file(input_path: Path) -> pd.DataFrame:
    if input_path.suffix.lower() == ".json":
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise SystemExit("JSON input must contain one object with field names and values.")
        return pd.DataFrame([data])

    return pd.read_csv(input_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict decision labels for a CSV file.")
    parser.add_argument("input_csv", help="Path to the CSV file to score")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Where to save predictions")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Path to the trained model")
    args = parser.parse_args()

    model_path = Path(args.model)
    input_path = Path(args.input_csv)
    output_path = Path(args.output)

    model = joblib.load(model_path)
    df = load_input_file(input_path)

    expected_features = get_expected_features(model)
    missing_required = [column for column in expected_features if column not in df.columns]
    if len(missing_required) == len(expected_features):
        raise SystemExit(
            "Input file does not match the trained model schema. "
            "This model expects the export.csv columns, not the clinical PA testing fields."
        )

    aligned = align_input(df, expected_features)
    aligned = normalize_numeric_columns(aligned, model)
    predictions = model.predict(aligned)

    result = df.copy()
    result["predicted_decision"] = predictions

    if hasattr(model.named_steps["classifier"], "predict_proba"):
        probabilities = model.predict_proba(aligned)
        classes = list(model.named_steps["classifier"].classes_)
        for index, class_name in enumerate(classes):
            result[f"prob_{class_name}"] = probabilities[:, index]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Saved predictions to {output_path}")
    print(result[["predicted_decision"]].head())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())