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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

app = Flask(__name__, static_folder="../static", static_url_path="")

script_dir = Path(__file__).resolve().parent
ROOT = script_dir.parent  # organized_scene
PROJECT_ROOT = ROOT.parent  # workspace root

EXPORT_MODEL_PATH = ROOT / "models" / "random_forest_model.joblib"
CLINICAL_MODEL_PATH = ROOT / "models" / "clinical_rf_model.joblib"
REVIEW_MODEL_PATH = ROOT / "models" / "review_rf_model.joblib"

CLINICAL_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "pa" / "clinical_pa_test_data.csv"
REVIEW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "pa" / "clinical_pa_training_data_with_review.csv"
RULE_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "pa" / "rule_based_training_data.csv"

# Global model placeholders
export_model = None
clinical_model = None
review_model = None

# Cached texts/rules
CLINICAL_POLICY_TEXT = ""
MEMBER_HANDBOOK_TEXT = ""
CLINICAL_POLICY_RULES = None

def load_policy_resources():
    """Load policy rules and handbook PDF texts into memory at startup."""
    global CLINICAL_POLICY_TEXT, MEMBER_HANDBOOK_TEXT, CLINICAL_POLICY_RULES
    
    # 1. Load clinical_policy_rules.json
    rules_json_path = PROJECT_ROOT / "clinical_policy_rules.json"
    if rules_json_path.exists():
        try:
            with open(rules_json_path, "r", encoding="utf-8") as f:
                CLINICAL_POLICY_RULES = json.load(f)
            print("Loaded clinical_policy_rules.json into memory.")
        except Exception as e:
            print(f"Error loading clinical_policy_rules.json: {e}")
            
    # 2. Parse clinical_policy_manual.pdf (fallback/caching)
    pdf_policy_path = PROJECT_ROOT / "data" / "raw" / "cms_mcd" / "clinical_policy_manual.pdf"
    if pdf_policy_path.exists():
        try:
            reader = pypdf.PdfReader(pdf_policy_path)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() or ""
            CLINICAL_POLICY_TEXT = full_text
            print("Loaded and cached clinical_policy_manual.pdf text.")
        except Exception as e:
            print(f"Error caching clinical policy manual PDF: {e}")
            
    # 3. Parse member_handbook.pdf
    pdf_handbook_path = PROJECT_ROOT / "data" / "raw" / "cms_mcd" / "member_handbook.pdf"
    if pdf_handbook_path.exists():
        try:
            reader = pypdf.PdfReader(pdf_handbook_path)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() or ""
            MEMBER_HANDBOOK_TEXT = full_text
            print("Loaded and cached member_handbook.pdf text.")
        except Exception as e:
            print(f"Error caching member handbook PDF: {e}")

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

def populate_mock_clinical_fields(df):
    """
    Generate realistic clinical fields (medications/lab_results) for the rule-based dataset 
    to prevent target leakage where the classifier trivially identifies dataset source.
    """
    df_copy = df.copy()
    
    # Define mapping by policy number
    policy_mappings = {
        'MHI-ORTHO-014': ('NSAIDs (Ibuprofen 600mg), Acetaminophen', 'Not applicable'),
        'MHI-RHEUM-007': ('Methotrexate 15mg/week, Folic acid', 'ESR: 45 mm/hr (elevated), RF: positive'),
        'MHI-ENDO-021': ('Insulin Glargine 20 units daily, Aspart 5 units mealtime', 'HbA1c: 8.5%'),
        'MHI-NEURO-009': ('Propranolol 80mg daily, Sumatriptan PRN', 'Not applicable'),
        'MHI-PULM-011': ('None reported', 'Sleep study AHI: 18'),
        'MHI-PSYCH-018': ('Sertraline 100mg, Bupropion 150mg', 'TSH: 2.1 mIU/L'),
        'MHI-CARD-003': ('Aspirin 81mg, Atorvastatin 40mg', 'Troponin: negative'),
        'MHI-ORTHO-022': ('Meloxicam 15mg daily, Tramadol 50mg PRN', 'X-ray: Kellgren-Lawrence Grade 3'),
        'MHI-PSYCH-005': ('Methylphenidate 20mg', 'Not applicable'),
        'MHI-GI-016': ('Mesalamine 2.4g daily, Azathioprine 100mg', 'CRP: 15 mg/L')
    }
    
    service_mappings = {
        'lumbar': ('NSAIDs (Ibuprofen 600mg), Acetaminophen', 'Not applicable'),
        'adalimumab': ('Methotrexate 15mg/week, Folic acid', 'ESR: 45 mm/hr (elevated), RF: positive'),
        'glucose': ('Insulin Glargine 20 units daily, Aspart 5 units mealtime', 'HbA1c: 8.5%'),
        'erenumab': ('Propranolol 80mg daily, Sumatriptan PRN', 'Not applicable'),
        'cpap': ('None reported', 'Sleep study AHI: 18'),
        'magnetic': ('Sertraline 100mg, Bupropion 150mg', 'TSH: 2.1 mIU/L'),
        'cardiac': ('Aspirin 81mg, Atorvastatin 40mg', 'Troponin: negative'),
        'knee': ('Meloxicam 15mg daily, Tramadol 50mg PRN', 'X-ray: Kellgren-Lawrence Grade 3'),
        'lisdexamfetamine': ('Methylphenidate 20mg', 'Not applicable'),
        'infliximab': ('Mesalamine 2.4g daily, Azathioprine 100mg', 'CRP: 15 mg/L')
    }

    medications = []
    lab_results = []
    
    for _, row in df_copy.iterrows():
        pol = str(row.get('policy_number', '')).strip()
        svc = str(row.get('requested_service', '')).lower()
        
        meds_val, labs_val = 'None', 'None'
        if pol in policy_mappings:
            meds_val, labs_val = policy_mappings[pol]
        else:
            for keyword, (m, l) in service_mappings.items():
                if keyword in svc:
                    meds_val, labs_val = m, l
                    break
        medications.append(meds_val)
        lab_results.append(labs_val)
        
    df_copy['medications'] = medications
    df_copy['lab_results'] = lab_results
    return df_copy

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
    y = df['decision'].fillna("missing").astype(str).str.strip()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )
    
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, class_weight="balanced", n_jobs=-1))
    ])
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    
    CLINICAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, CLINICAL_MODEL_PATH)
    
    metrics_path = CLINICAL_MODEL_PATH.parent / "clinical_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    
    print(f"Clinical model trained and saved to {CLINICAL_MODEL_PATH}")
    print(f"Clinical test set accuracy: {metrics['accuracy']:.4f}")
    return clf

def train_review_model(force_retrain=False):
    """Train review PA model from CSV, and save to models/review_rf_model.joblib"""
    if REVIEW_MODEL_PATH.exists() and not force_retrain:
        print(f"Loading existing review model from {REVIEW_MODEL_PATH}")
        return joblib.load(REVIEW_MODEL_PATH)
    
    print("Training review RF model on combined datasets...")
    df1 = pd.read_csv(REVIEW_DATA_PATH)
    df2 = pd.read_csv(RULE_DATA_PATH)
    
    # Align text columns for rule based training data
    df2['clinical_history'] = df2['feature_summary'].fillna("None").astype(str)
    df2['previous_treatments'] = df2['feature_summary'].fillna("None").astype(str)
    
    # Fix dataset leakage by populating clinical fields with realistic values
    df2 = populate_mock_clinical_fields(df2)
    
    df = pd.concat([df1, df2], ignore_index=True)
    
    # Removed label corruption overwrite: df.loc[df['meets_criteria'] == 'Partial', 'decision'] = 'In Review'
    
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
    y = df['decision'].fillna("missing").astype(str).str.strip()
    y_full = df[['decision', 'meets_criteria', 'decision_reason']]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )
    
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=50, class_weight="balanced", n_jobs=-1))
    ])
    
    clf.fit(X_train, y_train)
    y_pred_dec = clf.predict(X_test)
    
    # Evaluate decision
    dec_acc = accuracy_score(y_test, y_pred_dec)
    dec_report = classification_report(y_test, y_pred_dec, output_dict=True)
    
    # Run the rule engine on the test set to evaluate meets_criteria and decision_reason
    test_indices = X_test.index
    y_true_all = y_full.loc[test_indices]
    
    y_pred_mc = []
    y_pred_reas = []
    
    import rule_engine
    for idx in test_indices:
        row_dict = df.loc[idx].to_dict()
        pred_dec = y_pred_dec[list(test_indices).index(idx)]
        mc, reas = rule_engine.derive_criteria_and_reason(row_dict, pred_dec)
        y_pred_mc.append(mc)
        y_pred_reas.append(reas)
        
    y_true_mc = y_true_all['meets_criteria'].fillna("missing").astype(str)
    y_true_reas = y_true_all['decision_reason'].fillna("missing").astype(str)
    
    mc_acc = accuracy_score(y_true_mc, y_pred_mc)
    reas_acc = accuracy_score(y_true_reas, y_pred_reas)
    
    metrics = {
        "decision": {
            "accuracy": dec_acc,
            "confusion_matrix": confusion_matrix(y_test, y_pred_dec).tolist(),
            "classification_report": dec_report
        },
        "meets_criteria": {
            "accuracy": mc_acc,
            "classification_report": classification_report(y_true_mc, y_pred_mc, output_dict=True)
        },
        "decision_reason": {
            "accuracy": reas_acc,
            "classification_report": classification_report(y_true_reas, y_pred_reas, output_dict=True)
        },
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    
    REVIEW_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, REVIEW_MODEL_PATH)
    
    metrics_path = REVIEW_MODEL_PATH.parent / "review_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    
    print(f"Review model trained and saved to {REVIEW_MODEL_PATH}")
    print(f"Review test set accuracy - Decision: {dec_acc:.4f}, Meets Criteria: {mc_acc:.4f}, Reason: {reas_acc:.4f}")
    return clf

# Load PDF resources and policy rules into memory
try:
    load_policy_resources()
except Exception as e:
    print(f"Error caching policy/handbook resources: {e}")

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
    if not CLINICAL_POLICY_RULES:
        # Fallback to cached PDF text if JSON is not available
        if CLINICAL_POLICY_TEXT:
            proc_code = str(data.get("procedure_code", "")).strip()
            diag_code = str(data.get("diagnosis_code", "")).strip()
            req_service = str(data.get("requested_service", "")).strip()
            diag = str(data.get("diagnosis", "")).strip()
            
            covered = False
            if proc_code and proc_code.lower() in CLINICAL_POLICY_TEXT.lower():
                covered = True
            elif diag_code and diag_code.lower() in CLINICAL_POLICY_TEXT.lower():
                covered = True
            elif req_service and req_service.lower() in CLINICAL_POLICY_TEXT.lower():
                covered = True
            elif diag and diag.lower() in CLINICAL_POLICY_TEXT.lower():
                covered = True
                
            if not covered:
                return False, f"Denied: Requested service '{req_service}' (Procedure Code: {proc_code}) or diagnosis '{diag}' (ICD-10 Code: {diag_code}) is not present in the clinical policy manual and is not covered by the health plan."
        return True, ""
    
    proc_code = str(data.get("procedure_code", "")).strip().upper()
    diag_code = str(data.get("diagnosis_code", "")).strip().upper()
    
    # Check if either procedure_code or diagnosis_code matches any policy
    covered = False
    for policy in CLINICAL_POLICY_RULES.get("policies", []):
        covered_procs = [c.upper() for c in policy.get("procedure_codes", [])]
        covered_diags = [c.upper() for c in policy.get("diagnosis_codes", [])]
        if (proc_code and proc_code in covered_procs) or (diag_code and diag_code in covered_diags):
            covered = True
            break
            
    if not covered:
        req_service = str(data.get("requested_service", "")).strip()
        diag = str(data.get("diagnosis", "")).strip()
        return False, f"Denied: Requested service '{req_service}' (Procedure Code: {proc_code}) or diagnosis '{diag}' (ICD-10 Code: {diag_code}) is not present in the clinical policy manual and is not covered by the health plan."
        
    return True, ""

def check_member_handbook_rules(data):
    """
    Verify the request plan configuration against the Member Handbook.
    Returns: (is_valid, decision, meets_criteria, reason, timeframe)
    """
    handbook_text = MEMBER_HANDBOOK_TEXT
    if not handbook_text:
        # Fallback to hardcoded parsing if handbook text is missing
        handbook_text = "HMO In-network only Yes, from PCP. EPO In-network only No. POS Yes, from PCP. PPO In and out-of-network No."

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
    if plan_type == "HMO":
        if "out-of-network" in network_status:
            return False, "Denied", "No", "Denied: HMO plans cover in-network services only. Out-of-network care is not covered.", timeframe
        if pcp_referral != "yes" and "referral" in handbook_text.lower():
            return False, "Denied", "No", "Denied: HMO plans require a referral from your Primary Care Provider (PCP) before obtaining specialist or diagnostic services.", timeframe

    elif plan_type == "EPO":
        if "out-of-network" in network_status:
            return False, "Denied", "No", "Denied: EPO plans cover in-network services only. Out-of-network care is not covered.", timeframe

    elif plan_type == "POS":
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
        
        # Classifier predicts decision directly (Approved / Denied / Pending / In Review)
        pred_decision = clinical_model.predict(row)[0]
        
        # Derives meets_criteria and decision_reason using deterministic rule engine
        import rule_engine
        meets_criteria, reason = rule_engine.derive_criteria_and_reason(data, pred_decision)
        
        probs = {}
        if hasattr(clinical_model.named_steps["classifier"], "predict_proba"):
            probabilities = clinical_model.predict_proba(row)[0]
            classes = list(clinical_model.named_steps["classifier"].classes_)
            probs = {str(c): float(probabilities[i]) for i, c in enumerate(classes)}
        
        return jsonify({
            "decision": pred_decision,
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
        
        # Classifier predicts decision directly (Approved / Denied / Pending / In Review)
        pred_decision = review_model.predict(row)[0]
        
        # Derives meets_criteria and decision_reason using deterministic rule engine
        import rule_engine
        meets_criteria, reason = rule_engine.derive_criteria_and_reason(data, pred_decision)
        
        probs = {}
        if hasattr(review_model.named_steps["classifier"], "predict_proba"):
            probabilities = review_model.predict_proba(row)[0]
            classes = list(review_model.named_steps["classifier"].classes_)
            probs = {str(c): float(probabilities[i]) for i, c in enumerate(classes)}
        
        return jsonify({
            "decision": pred_decision,
            "meets_criteria": meets_criteria,
            "decision_reason": reason,
            "probabilities": probs,
            "timeframe": timeframe,
            "patient_name": str(data.get("patient_name", ""))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # NOTE: debug=True must be changed to False before any production deployment.
    # Werkzeug debug mode allows remote code execution if exposed.
    app.run(host="127.0.0.1", port=5000, debug=True)
