import argparse
import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = ROOT / "raw" / "export.csv"
PROCESSED_DATA = ROOT / "processed" / "preprocessed_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "random_forest_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


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
    for column in text_columns:
        if column in features.columns:
            features[column] = features[column].fillna("missing").astype(str)

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


def derive_decision(df: pd.DataFrame) -> pd.Series:
    if "decision" in df.columns and df["decision"].fillna("").astype(str).str.strip().ne("").any():
        return df["decision"].fillna("").astype(str).str.strip()

    if "Approval rate" not in df.columns:
        raise SystemExit("No `decision` column and no `Approval rate` column available to derive labels.")

    def parse_rate(value: object) -> float:
        if pd.isna(value):
            return float("nan")
        text = str(value).strip()
        if not text or text.upper() == "NA":
            return float("nan")
        match = re.search(r"[-+]?\d*\.?\d+", text)
        if not match:
            return float("nan")
        number = float(match.group(0))
        return number / 100.0 if "%" in text else number

    approval_rate = df["Approval rate"].map(parse_rate)
    if approval_rate.notna().sum() == 0:
        raise SystemExit("Approval rate values are empty, so `decision` cannot be derived.")

    decision = approval_rate.apply(lambda value: "approved" if value >= 0.5 else "denied")
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess data and train a RandomForest model.")
    parser.add_argument("--target", default="decision", help="Target column to predict")
    parser.add_argument("--source", default=str(RAW_DATA), help="Input CSV file")
    parser.add_argument(
        "--approval-threshold",
        type=float,
        default=0.5,
        help="Threshold used to derive approved vs denied from Approval rate",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    source_path = Path(args.source)
    df = pd.read_csv(source_path)

    if args.target == "decision":
        if "decision" in df.columns and df["decision"].fillna("").astype(str).str.strip().ne("").any():
            target = df["decision"].fillna("").astype(str).str.strip()
        elif "Approval rate" in df.columns:
            approval_rate = df["Approval rate"].astype(str).str.strip()

            def parse_rate(value: str) -> float:
                if not value or value.upper() == "NA":
                    return float("nan")
                match = re.search(r"[-+]?\d*\.?\d+", value)
                if not match:
                    return float("nan")
                number = float(match.group(0))
                return number / 100.0 if "%" in value else number

            approval_rate_numeric = approval_rate.map(parse_rate)
            if approval_rate_numeric.notna().sum() == 0:
                raise SystemExit("Approval rate values are empty, so `decision` cannot be derived.")
            target = approval_rate_numeric.apply(
                lambda value: "approved" if value >= args.approval_threshold else "denied"
            )
        else:
            raise SystemExit(
                "Neither `decision` nor `Approval rate` exists in the source file, so labels cannot be created."
            )
    else:
        if args.target not in df.columns:
            raise SystemExit(f"Target column '{args.target}' not found in {source_path}")
        target = df[args.target].fillna("").astype(str).str.strip()

    if target.eq("").all():
        raise SystemExit("Target column is empty in the current dataset.")

    processed = build_features(df)
    processed["decision"] = target.values
    processed.to_csv(PROCESSED_DATA, index=False)

    y = target
    X = processed.drop(columns=[args.target], errors="ignore")

    numeric_features = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    categorical_features = [col for col in X.columns if col not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=args.random_state,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved processed data to {PROCESSED_DATA}")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())