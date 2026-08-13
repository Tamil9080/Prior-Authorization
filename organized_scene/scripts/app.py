import os
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

app = Flask(__name__, static_folder="../static", static_url_path="")

ROOT = Path(r"d:\cts hackthon\new1\organized_scene")
EXPORT_MODEL_PATH = ROOT / "models" / "random_forest_model.joblib"
CLINICAL_MODEL_PATH = ROOT / "models" / "clinical_rf_model.joblib"
CLINICAL_DATA_PATH = ROOT / "raw" / "clinical_pa_test_data.csv"

# Global model place holders
export_model = None
clinical_model = None

def get_expected_export_features(model) -> list[str]:
    expected = list(model.named_steps["preprocessor"].feature_names_in_)
    if "decision" in expected:
        expected = [c for c in expected if c != "decision"]
    return expected

def normalize_export_numeric_columns(df: pd.DataFrame, model) -> pd.DataFrame:
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

def train_clinical_model():
    """Train clinical PA model from CSV if not exists, and save to models/clinical_rf_model.joblib"""
    if CLINICAL_MODEL_PATH.exists():
        print(f"Loading existing clinical model from {CLINICAL_MODEL_PATH}")
        return joblib.load(CLINICAL_MODEL_PATH)
    
    print("Training clinical RF model...")
    df = pd.read_csv(CLINICAL_DATA_PATH)
    
    text_cols = ['diagnosis', 'requested_service', 'clinical_history', 'previous_treatments', 'medications', 'lab_results']
    cat_cols = ['diagnosis_code', 'procedure_code', 'provider_specialty', 'urgency', 'policy_id']
    num_cols = ['age']
    
    for col in text_cols:
        df[col] = df[col].fillna("None").astype(str)
    for col in cat_cols:
        df[col] = df[col].fillna("missing").astype(str)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ] + [
            (f'text_{col}', TfidfVectorizer(max_features=500), col) for col in text_cols
        ]
    )
    
    X = df[num_cols + cat_cols + text_cols]
    y_reas = df['decision_reason']
    
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=300))
    ])
    
    clf.fit(X, y_reas)
    CLINICAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, CLINICAL_MODEL_PATH)
    print(f"Clinical model trained and saved to {CLINICAL_MODEL_PATH}")
    return clf

# Load models on startup
try:
    if EXPORT_MODEL_PATH.exists():
        export_model = joblib.load(EXPORT_MODEL_PATH)
        print("Loaded export model.")
    else:
        print("Export model not found! Run training pipeline first.")
except Exception as e:
    print(f"Error loading export model: {e}")

try:
    clinical_model = train_clinical_model()
except Exception as e:
    print(f"Error loading/training clinical model: {e}")


@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/predict_export", methods=["POST"])
def predict_export():
    if not export_model:
        return jsonify({"error": "Export model is not loaded."}), 500
    
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided."}), 400
        
        row = pd.DataFrame([data])
        expected = get_expected_export_features(export_model)
        
        # align
        for col in expected:
            if col not in row.columns:
                row[col] = pd.NA
        row = row[expected]
        row = normalize_export_numeric_columns(row, export_model)
        row = row.where(pd.notna(row), None)
        
        pred = export_model.predict(row)[0]
        probs = {}
        if hasattr(export_model.named_steps["classifier"], "predict_proba"):
            probabilities = export_model.predict_proba(row)[0]
            classes = list(export_model.named_steps["classifier"].classes_)
            probs = {str(c): float(probabilities[i]) for i, c in enumerate(classes)}
        
        return jsonify({
            "predicted_decision": str(pred),
            "probabilities": probs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict_clinical", methods=["POST"])
def predict_clinical():
    if not clinical_model:
        return jsonify({"error": "Clinical model is not loaded."}), 500
    
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided."}), 400
        
        text_cols = ['diagnosis', 'requested_service', 'clinical_history', 'previous_treatments', 'medications', 'lab_results']
        cat_cols = ['diagnosis_code', 'procedure_code', 'provider_specialty', 'urgency', 'policy_id']
        num_cols = ['age']
        
        # Build user row df
        row_dict = {}
        for col in num_cols:
            row_dict[col] = pd.to_numeric(data.get(col), errors='coerce') if data.get(col) is not None else None
        for col in cat_cols:
            row_dict[col] = str(data.get(col, 'missing'))
        for col in text_cols:
            row_dict[col] = str(data.get(col, 'None'))
            
        row = pd.DataFrame([row_dict])
        
        reason = clinical_model.predict(row)[0]
        
        # Map reason to meets_criteria and decision using the deterministic mapping discovered
        meets_criteria = "Partial"
        decision = "Denied"
        
        if "Clinical criteria met" in reason:
            meets_criteria = "Yes"
            decision = "Approved"
        elif "clinical criteria not met" in reason:
            meets_criteria = "No"
            decision = "Denied"
        elif "Cannot render decision" in reason:
            meets_criteria = "Insufficient Information"
            decision = "Pending Additional Information"
        elif "Approved with conditions" in reason or "Approved on the basis of clinical urgency" in reason:
            meets_criteria = "Partial"
            decision = "Approved"
        
        probs = {}
        if hasattr(clinical_model.named_steps["classifier"], "predict_proba"):
            probabilities = clinical_model.predict_proba(row)[0]
            classes = list(clinical_model.named_steps["classifier"].classes_)
            probs = {str(c): float(probabilities[i]) for i, c in enumerate(classes)}
        
        return jsonify({
            "decision": decision,
            "meets_criteria": meets_criteria,
            "decision_reason": reason,
            "probabilities": probs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
