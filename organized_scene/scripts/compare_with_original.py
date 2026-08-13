import argparse
import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


ROOT = Path(r"d:\cts hackthon\new1\organized_scene")
MODEL_PATH = ROOT / "models" / "random_forest_model.joblib"


def parse_rate(value: object) -> float:
    text = str(value).strip()
    if not text or text.upper() == "NA":
        return float("nan")
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return float("nan")
    number = float(match.group(0))
    return number / 100.0 if "%" in text else number


def derive_original_label(df: pd.DataFrame, threshold: float) -> pd.Series:
    if "decision" in df.columns and df["decision"].fillna("").astype(str).str.strip().ne("").any():
        return df["decision"].fillna("").astype(str).str.strip()

    if "Approval rate" in df.columns:
        approval_rate = df["Approval rate"].map(parse_rate)
        if approval_rate.notna().sum() == 0:
            raise SystemExit("Could not derive original labels because Approval rate is empty.")
        return approval_rate.apply(lambda value: "approved" if value >= threshold else "denied")

    raise SystemExit("The input file must have either `decision` or `Approval rate` to compare against.")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()

    text_columns = [
        "Carrier",
        "Service category",
        "Request",
        "Code type",
        "Code",
        "Description of service",
        "Drug name",
        "Drug brand names",
    ]
    numeric_columns = [
        "Year",
        "Number of requests per code",
        "Expedited - Avg response time",
        "Standard - Avg response time",
        "Extenuating circumstances - Avg response time",
        "Expedited - Number of requests",
        "Standard - Number of requests",
        "Extenuating circumstances - Number of requests",
    ]

    for column in text_columns:
        if column in features.columns:
            features[column] = features[column].fillna("missing").astype(str)

    for column in numeric_columns:
        if column in features.columns:
            features[column] = pd.to_numeric(
                features[column].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )

    for column in ["Approval rate", "Initially denied then approved - approval rate"]:
        if column in features.columns:
            features = features.drop(columns=[column])

    if "Index" in features.columns:
        features = features.drop(columns=["Index"])

    return features


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
    parser = argparse.ArgumentParser(description="Compare model predictions with the original label.")
    parser.add_argument("input_csv", help="CSV file containing the original label or approval rate")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Path to the trained model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Approval rate threshold")
    parser.add_argument("--output", default="", help="Optional path to save row-level comparison CSV")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    model = joblib.load(Path(args.model))
    df = pd.read_csv(input_path)

    original = derive_original_label(df, args.threshold)
    features = build_features(df)

    expected_features = list(model.named_steps["preprocessor"].feature_names_in_)
    if "decision" in expected_features:
        expected_features = [column for column in expected_features if column != "decision"]

    aligned = align_input(features, expected_features)
    aligned = normalize_numeric_columns(aligned, model)

    predicted = pd.Series(model.predict(aligned), index=df.index, name="predicted_decision")

    comparison = df.copy()
    comparison["original_decision"] = original
    comparison["predicted_decision"] = predicted
    comparison["match"] = comparison["original_decision"].astype(str).str.strip() == comparison["predicted_decision"].astype(str).str.strip()

    accuracy = accuracy_score(comparison["original_decision"], comparison["predicted_decision"])
    report = classification_report(comparison["original_decision"], comparison["predicted_decision"], output_dict=True)
    matrix = confusion_matrix(comparison["original_decision"], comparison["predicted_decision"]).tolist()

    summary = {
        "accuracy": accuracy,
        "confusion_matrix": matrix,
        "classification_report": report,
        "rows": int(len(comparison)),
        "matches": int(comparison["match"].sum()),
        "mismatches": int((~comparison["match"]).sum()),
    }

    print(json.dumps(summary, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(output_path, index=False)
        print(f"Saved row-level comparison to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())