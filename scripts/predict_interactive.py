import argparse
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "random_forest_model.joblib"


def get_expected_features(model) -> list[str]:
    expected_features = list(model.named_steps["preprocessor"].feature_names_in_)
    if "decision" in expected_features:
        expected_features = [column for column in expected_features if column != "decision"]
    return expected_features


def get_numeric_features(model) -> list[str]:
    numeric_features = []
    for name, transformer, columns in model.named_steps["preprocessor"].transformers_:
        if name == "num":
            numeric_features = list(columns)
            break
    return numeric_features


def prompt_for_row(expected_features: list[str], numeric_features: list[str]) -> pd.DataFrame:
    print("\n=== Input Section ===")
    print("Press Enter to leave a field blank and let the model impute it.\n")

    row = {}
    for column in expected_features:
        label = f"{column}"
        if column in numeric_features:
            label += " [number]"
        else:
            label += " [text]"

        value = input(f"{label}: ").strip()
        row[column] = value if value else None

    return pd.DataFrame([row])


def normalize_numeric_columns(df: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    normalized = df.copy()
    for column in numeric_features:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(
                normalized[column].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactively type values and predict decision.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Path to the trained model")
    args = parser.parse_args()

    model = joblib.load(Path(args.model))
    expected_features = get_expected_features(model)
    numeric_features = get_numeric_features(model)

    row = prompt_for_row(expected_features, numeric_features)
    row = normalize_numeric_columns(row, numeric_features)
    row = row.where(pd.notna(row), None)

    prediction = model.predict(row)[0]
    print("\n=== Prediction ===")
    print(f"predicted_decision: {prediction}")

    if hasattr(model.named_steps["classifier"], "predict_proba"):
        probabilities = model.predict_proba(row)[0]
        classes = list(model.named_steps["classifier"].classes_)
        print("probabilities:")
        for index, class_name in enumerate(classes):
            print(f"  {class_name}: {probabilities[index]:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
