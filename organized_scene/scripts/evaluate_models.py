import os
import sys
import re
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(r"d:\cts hackthon\new1\organized_scene")
EXPORT_MODEL_PATH = ROOT / "models" / "random_forest_model.joblib"
CLINICAL_MODEL_PATH = ROOT / "models" / "clinical_rf_model.joblib"
REVIEW_MODEL_PATH = ROOT / "models" / "review_rf_model.joblib"

EXPORT_DATA_PATH = ROOT / "raw" / "export.csv"
CLINICAL_DATA_PATH = ROOT / "raw" / "clinical_pa_test_data.csv"
REVIEW_DATA_PATH = Path(r"d:\cts hackthon\new1\clinical_pa_training_data_with_review.csv")
RULE_DATA_PATH = Path(r"d:\cts hackthon\new1\rule_based_training_data.csv")

def parse_rate(value: object) -> float:
    text = str(value).strip()
    if not text or text.upper() == "NA":
        return float("nan")
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return float("nan")
    number = float(match.group(0))
    return number / 100.0 if "%" in text else number

def predict_in_batches(model, X, batch_size=500):
    predictions = []
    for i in range(0, len(X), batch_size):
        chunk = X.iloc[i:i+batch_size]
        pred = model.predict(chunk)
        predictions.append(pred)
    if len(predictions[0].shape) > 1:
        return np.vstack(predictions)
    else:
        return np.concatenate(predictions)

def evaluate_export_model():
    print("Evaluating Export Model...")
    model = joblib.load(EXPORT_MODEL_PATH)
    df = pd.read_csv(EXPORT_DATA_PATH)
    
    # Derive target decision
    if "decision" in df.columns and df["decision"].fillna("").astype(str).str.strip().ne("").any():
        y_true = df["decision"].fillna("").astype(str).str.strip()
    else:
        approval_rate = df["Approval rate"].map(parse_rate)
        y_true = approval_rate.apply(lambda v: "approved" if v >= 0.5 else "denied")
        
    expected = list(model.named_steps["preprocessor"].feature_names_in_)
    if "decision" in expected:
        expected = [c for c in expected if c != "decision"]
        
    # Align and clean inputs
    features = df.copy()
    text_columns = ["Carrier", "Service category", "Request", "Code type", "Code", "Description of service", "Drug name", "Drug brand names"]
    numeric_columns = ["Year", "Number of requests per code", "Expedited - Avg response time", "Standard - Avg response time", "Extenuating circumstances - Avg response time", "Expedited - Number of requests", "Standard - Number of requests", "Extenuating circumstances - Number of requests"]
    
    for c in text_columns:
        if c in features.columns:
            features[c] = features[c].fillna("missing").astype(str)
    for c in numeric_columns:
        if c in features.columns:
            features[c] = pd.to_numeric(features[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
            
    # Keep only expected
    aligned = features.reindex(columns=expected)
    
    # Normalize numeric columns using model preprocessor spec
    numeric_features = []
    for name, transformer, columns in model.named_steps["preprocessor"].transformers_:
        if name == "num":
            numeric_features = list(columns)
            break
    for c in numeric_features:
        if c in aligned.columns:
            aligned[c] = pd.to_numeric(aligned[c], errors="coerce").fillna(0)
            
    y_pred = predict_in_batches(model, aligned)
    acc = accuracy_score(y_true, y_pred)
    rep = classification_report(y_true, y_pred, output_dict=True)
    return acc, rep

def evaluate_clinical_model():
    print("Evaluating Clinical Model...")
    model = joblib.load(CLINICAL_MODEL_PATH)
    df = pd.read_csv(CLINICAL_DATA_PATH)
    
    text_cols = ['diagnosis', 'requested_service', 'clinical_history', 'previous_treatments', 'medications', 'lab_results']
    cat_cols = ['diagnosis_code', 'procedure_code', 'provider_specialty', 'urgency', 'policy_id']
    num_cols = ['age']
    
    for col in text_cols:
        df[col] = df[col].fillna("None").astype(str)
    for col in cat_cols:
        df[col] = df[col].fillna("missing").astype(str)
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(df['age'].median() if not df['age'].isnull().all() else 45)
    
    X = df[num_cols + cat_cols + text_cols]
    y_true = df['decision_reason']
    
    y_pred = predict_in_batches(model, X)
    acc = accuracy_score(y_true, y_pred)
    rep = classification_report(y_true, y_pred, output_dict=True)
    return acc, rep

def evaluate_review_model():
    print("Evaluating Review Model...")
    model = joblib.load(REVIEW_MODEL_PATH)
    
    df1 = pd.read_csv(REVIEW_DATA_PATH)
    df2 = pd.read_csv(RULE_DATA_PATH)
    
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
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(df['age'].median() if not df['age'].isnull().all() else 45)
    
    X = df[num_cols + cat_cols + text_cols]
    y_true = df[['decision', 'meets_criteria', 'decision_reason']]
    
    y_pred = predict_in_batches(model, X)
    
    results = {}
    for i, col in enumerate(['decision', 'meets_criteria', 'decision_reason']):
        acc = accuracy_score(y_true[col].astype(str), y_pred[:, i].astype(str))
        rep = classification_report(y_true[col].astype(str), y_pred[:, i].astype(str), output_dict=True)
        results[col] = (acc, rep)
        
    return results

def main():
    try:
        exp_acc, exp_rep = evaluate_export_model()
        clin_acc, clin_rep = evaluate_clinical_model()
        rev_results = evaluate_review_model()
        
        report_path = Path(r"C:\Users\hp laptop\.gemini\antigravity-ide\brain\542b742b-1fc9-4552-8ca6-40352744ad6f\evaluation_report.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Model Evaluation Report\n\n")
            f.write("This report summarizes the performance metrics of the trained Random Forest classifiers in the Prior Authorization platform.\n\n")
            
            f.write("## 1. Aggregate Carrier-Level Model\n")
            f.write(f"* **Accuracy**: {exp_acc:.4f}\n")
            f.write(f"* **Macro Precision**: {exp_rep['macro avg']['precision']:.4f}\n")
            f.write(f"* **Macro Recall**: {exp_rep['macro avg']['recall']:.4f}\n")
            f.write(f"* **Macro F1-Score**: {exp_rep['macro avg']['f1-score']:.4f}\n\n")
            
            f.write("## 2. Clinical Patient-Level Model\n")
            f.write(f"* **Accuracy**: {clin_acc:.4f}\n")
            f.write(f"* **Macro Precision**: {clin_rep['macro avg']['precision']:.4f}\n")
            f.write(f"* **Macro Recall**: {clin_rep['macro avg']['recall']:.4f}\n")
            f.write(f"* **Macro F1-Score**: {clin_rep['macro avg']['f1-score']:.4f}\n\n")
            
            f.write("## 3. Review Decision Model (Multi-Output)\n")
            for col, (acc, rep) in rev_results.items():
                f.write(f"### Target: `{col}`\n")
                f.write(f"* **Accuracy**: {acc:.4f}\n")
                f.write(f"* **Macro Precision**: {rep['macro avg']['precision']:.4f}\n")
                f.write(f"* **Macro Recall**: {rep['macro avg']['recall']:.4f}\n")
                f.write(f"* **Macro F1-Score**: {rep['macro avg']['f1-score']:.4f}\n\n")
                
        print(f"Model evaluation complete. Summary saved to {report_path}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during model evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
