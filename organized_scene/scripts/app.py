import os
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import pypdf
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
REVIEW_MODEL_PATH = ROOT / "models" / "review_rf_model.joblib"

CLINICAL_DATA_PATH = ROOT / "raw" / "clinical_pa_test_data.csv"
REVIEW_DATA_PATH = Path(r"d:\cts hackthon\new1\clinical_pa_training_data_with_review.csv")

# Global model place holders
export_model = None
clinical_model = None
review_model = None

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

def train_clinical_model(force_retrain=False):
    """Train clinical PA model from CSV, and save to models/clinical_rf_model.joblib"""
    if CLINICAL_MODEL_PATH.exists() and not force_retrain:
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
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1))
    ])
    
    clf.fit(X, y_reas)
    CLINICAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, CLINICAL_MODEL_PATH)
    print(f"Clinical model trained and saved to {CLINICAL_MODEL_PATH}")
    return clf

def train_review_model(force_retrain=False):
    """Train review PA model from CSV, and save to models/review_rf_model.joblib"""
    if REVIEW_MODEL_PATH.exists() and not force_retrain:
        print(f"Loading existing review model from {REVIEW_MODEL_PATH}")
        return joblib.load(REVIEW_MODEL_PATH)
    
    print("Training review RF model on combined datasets...")
    df1 = pd.read_csv(REVIEW_DATA_PATH)
    df2 = pd.read_csv(Path(r"d:\cts hackthon\new1\rule_based_training_data.csv"))
    
    # Align text columns for rule based training data
    df2['clinical_history'] = df2['feature_summary'].fillna("None").astype(str)
    df2['previous_treatments'] = df2['feature_summary'].fillna("None").astype(str)
    df2['medications'] = "None"
    df2['lab_results'] = "None"
    
    df = pd.concat([df1, df2], ignore_index=True)
    df.loc[df['meets_criteria'] == 'Partial', 'decision'] = 'In Review'
    
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
    y = df[['decision', 'meets_criteria', 'decision_reason']]
    
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=50, n_jobs=-1))
    ])
    
    clf.fit(X, y)
    REVIEW_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, REVIEW_MODEL_PATH)
    print(f"Review model trained and saved to {REVIEW_MODEL_PATH}")
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

try:
    review_model = train_review_model()
except Exception as e:
    print(f"Error loading/training review model: {e}")


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

def check_policy_coverage(data):
    """
    Check if the requested service or diagnosis is present in the clinical policy manual.
    Returns: (is_covered, message)
    """
    pdf_path = Path(r"d:\cts hackthon\new1\clinical_policy_manual.pdf")
    if not pdf_path.exists():
        return True, ""
    
    try:
        reader = pypdf.PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() or ""
        
        proc_code = str(data.get("procedure_code", "")).strip()
        diag_code = str(data.get("diagnosis_code", "")).strip()
        req_service = str(data.get("requested_service", "")).strip()
        diag = str(data.get("diagnosis", "")).strip()
        
        # Check if the procedure code or diagnosis code is mentioned in the manual
        covered = False
        if proc_code and proc_code.lower() in full_text.lower():
            covered = True
        elif diag_code and diag_code.lower() in full_text.lower():
            covered = True
        elif req_service and req_service.lower() in full_text.lower():
            covered = True
        elif diag and diag.lower() in full_text.lower():
            covered = True
            
        if not covered:
            return False, f"Denied: Requested service '{req_service}' (Procedure Code: {proc_code}) or diagnosis '{diag}' (ICD-10 Code: {diag_code}) is not present in the clinical policy manual and is not covered by the health plan."
            
        return True, ""
    except Exception as e:
        print(f"Error reading clinical policy manual: {e}")
        return True, ""

def check_member_handbook_rules(data):
    """
    Verify the request plan configuration against the Member Handbook.
    Returns: (is_valid, decision, meets_criteria, reason, timeframe)
    """
    pdf_path = Path(r"d:\cts hackthon\new1\member_handbook.pdf")
    if not pdf_path.exists():
        # Fallback to hardcoded parsing if PDF is missing
        handbook_text = "HMO In-network only Yes, from PCP. EPO In-network only No. POS Yes, from PCP. PPO In and out-of-network No."
    else:
        try:
            reader = pypdf.PdfReader(pdf_path)
            handbook_text = ""
            for page in reader.pages:
                handbook_text += page.extract_text() or ""
        except Exception as e:
            print(f"Error reading member handbook PDF: {e}")
            handbook_text = ""

    plan_type = str(data.get("plan_type", "PPO")).strip().upper()
    network_status = str(data.get("network_status", "In-Network")).strip().lower()
    pcp_referral = str(data.get("pcp_referral", "Yes")).strip().lower()
    urgency = str(data.get("urgency", "Routine")).strip().lower()

    # Determine timeframe based on urgency rules in handbook
    timeframe = "Within 5-7 business days" # default Routine
    if "urgent" in urgency:
        timeframe = "Within 24-72 hours"
    elif "emergent" in urgency:
        timeframe = "Immediate / concurrent with care"

    # Verify plan type rules
    # Check HMO rules
    if plan_type == "HMO":
        # HMO covers In-network only and requires PCP referral
        if "out-of-network" in network_status:
            return False, "Denied", "No", "Denied: HMO plans cover in-network services only. Out-of-network care is not covered.", timeframe
        if pcp_referral != "yes" and "referral" in handbook_text.lower():
            return False, "Denied", "No", "Denied: HMO plans require a referral from your Primary Care Provider (PCP) before obtaining specialist or diagnostic services.", timeframe

    # Check EPO rules
    elif plan_type == "EPO":
        # EPO covers In-network only, no referral required
        if "out-of-network" in network_status:
            return False, "Denied", "No", "Denied: EPO plans cover in-network services only. Out-of-network care is not covered.", timeframe

    # Check POS rules
    elif plan_type == "POS":
        # POS covers in and out of network, but requires referral
        if pcp_referral != "yes" and "referral" in handbook_text.lower():
            return False, "Denied", "No", "Denied: POS plans require a referral from your Primary Care Provider (PCP) for coverage.", timeframe

    return True, "Approved", "Yes", "", timeframe

@app.route("/api/predict_clinical", methods=["POST"])
def predict_clinical():
    if not clinical_model:
        return jsonify({"error": "Clinical model is not loaded."}), 500
    
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided."}), 400
        
        # RAG Coverage Check first!
        covered, msg = check_policy_coverage(data)
        if not covered:
            return jsonify({
                "decision": "Denied",
                "meets_criteria": "No",
                "decision_reason": msg,
                "probabilities": {msg: 1.0},
                "timeframe": "N/A"
            })
        
        # Member Handbook check!
        handbook_ok, h_decision, h_meets, h_reason, timeframe = check_member_handbook_rules(data)
        if not handbook_ok:
            return jsonify({
                "decision": h_decision,
                "meets_criteria": h_meets,
                "decision_reason": h_reason,
                "probabilities": {h_reason: 1.0},
                "timeframe": timeframe
            })
        
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
        decision = "In Review"
        
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
            decision = "In Review"
        
        probs = {}
        if hasattr(clinical_model.named_steps["classifier"], "predict_proba"):
            probabilities = clinical_model.predict_proba(row)[0]
            classes = list(clinical_model.named_steps["classifier"].classes_)
            probs = {str(c): float(probabilities[i]) for i, c in enumerate(classes)}
        
        return jsonify({
            "decision": decision,
            "meets_criteria": meets_criteria,
            "decision_reason": reason,
            "probabilities": probs,
            "timeframe": timeframe,
            "patient_name": str(data.get("patient_name", ""))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict_review", methods=["POST"])
def predict_review():
    if not review_model:
        return jsonify({"error": "Review model is not loaded."}), 500
    
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided."}), 400
        
        # RAG Coverage Check first!
        covered, msg = check_policy_coverage(data)
        if not covered:
            return jsonify({
                "decision": "Denied",
                "meets_criteria": "No",
                "decision_reason": msg,
                "probabilities": {msg: 1.0},
                "timeframe": "N/A"
            })
        
        # Member Handbook check!
        handbook_ok, h_decision, h_meets, h_reason, timeframe = check_member_handbook_rules(data)
        if not handbook_ok:
            return jsonify({
                "decision": h_decision,
                "meets_criteria": h_meets,
                "decision_reason": h_reason,
                "probabilities": {h_reason: 1.0},
                "timeframe": timeframe
            })
        
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
        
        pred = review_model.predict(row)[0]
        decision = str(pred[0])
        meets_criteria = str(pred[1])
        reason = str(pred[2])
        
        probs = {}
        if hasattr(review_model.named_steps["classifier"], "predict_proba"):
            probabilities = review_model.predict_proba(row)
            classes = list(review_model.named_steps["classifier"].classes_[0])
            probs = {str(c): float(probabilities[0][0][i]) for i, c in enumerate(classes)}
        
        return jsonify({
            "decision": decision,
            "meets_criteria": meets_criteria,
            "decision_reason": reason,
            "probabilities": probs,
            "timeframe": timeframe,
            "patient_name": str(data.get("patient_name", ""))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
