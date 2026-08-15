import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, precision_recall_curve, auc
)

# Paths
script_dir = Path(__file__).resolve().parent
PROJECT_ROOT = script_dir.parent

# Set up sys.path to import from app/
sys.path.append(str(PROJECT_ROOT / "app"))

from policy import policy_service
from rules import rule_evaluator

# Load models and raw data
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest" / "clinical_rf_model.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "pa" / "clinical_pa_test_data.csv"
REPORT_PATH = PROJECT_ROOT / "organized_scene" / "predictions" / "triage_evaluation_report.md"

def main():
    if not MODEL_PATH.exists():
        print(f"Error: Model file {MODEL_PATH} does not exist. Run app startup training first!")
        return
        
    if not DATA_PATH.exists():
        print(f"Error: Data file {DATA_PATH} does not exist.")
        return
        
    import joblib
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    
    # Preprocess text and categorical columns
    text_cols = ['diagnosis', 'requested_service', 'clinical_history', 'previous_treatments', 'medications', 'lab_results']
    cat_cols = ['diagnosis_code', 'procedure_code', 'provider_specialty', 'urgency', 'policy_id']
    num_cols = ['age']
    
    for col in text_cols:
        df[col] = df[col].fillna("None").astype(str)
    for col in cat_cols:
        df[col] = df[col].fillna("missing").astype(str)
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(df['age'].median() if not df['age'].isnull().all() else 45)
    
    X = df[num_cols + cat_cols + text_cols]
    y_true_decision = df['decision'].fillna("missing").astype(str).str.strip()
    
    print(f"Running triage evaluation on {len(df)} test rows...")
    
    # Batch predict to avoid loops overhead
    print("Running batch machine learning predictions...")
    y_pred_all = model.predict(X)
    
    y_prob_all = []
    if hasattr(model.named_steps["classifier"], "predict_proba"):
        probabilities_all = model.predict_proba(X)
        classes = list(model.named_steps["classifier"].classes_)
        if "Approved" in classes:
            approved_idx = classes.index("Approved")
            y_prob_all = probabilities_all[:, approved_idx]
        else:
            y_prob_all = probabilities_all[:, 0]
    else:
        y_prob_all = np.array([0.5] * len(X))
        
    y_pred_triage_dec = []
    
    # Reset index to iterate safely
    df = df.reset_index(drop=True)
    
    print("Evaluating policy rules...")
    # Run predictions row by row to evaluate rules
    for idx in range(len(df)):
        row_dict = df.loc[idx].to_dict()
        pred = y_pred_all[idx]
        prob = y_prob_all[idx]
        
        # 2. Match Policy
        policy = policy_service.match_policy(
            service_code=row_dict.get("procedure_code"),
            diagnosis_code=row_dict.get("diagnosis_code"),
            service_description=row_dict.get("requested_service"),
            diagnosis=row_dict.get("diagnosis")
        )
        
        # 3. Evaluate Policy rules
        if not policy:
            final_rec = "PEND"
        else:
            rules_eval = rule_evaluator.evaluate_rules(row_dict, policy)
            statuses = [r["status"] for r in rules_eval]
            
            if "NOT_MET" in statuses:
                final_rec = "Denied"
            elif "INSUFFICIENT" in statuses:
                final_rec = "Pending Additional Information"
            else:
                final_rec = "Approved" if pred.upper() == "APPROVED" else "Denied"
                
        y_pred_triage_dec.append(final_rec)
        
    y_pred_triage_dec = np.array(y_pred_triage_dec)
    y_prob_all = np.array(y_prob_all)
    
    # Convert decisions to string type for comparisons
    y_true_str = y_true_decision.values.astype(str)
    
    # Multiclass Metrics (Triage vs. Ground Truth)
    acc = accuracy_score(y_true_str, y_pred_triage_dec)
    precision_macro = precision_score(y_true_str, y_pred_triage_dec, average='macro', zero_division=0)
    recall_macro = recall_score(y_true_str, y_pred_triage_dec, average='macro', zero_division=0)
    f1_macro = f1_score(y_true_str, y_pred_triage_dec, average='macro', zero_division=0)
    
    classes_labels = sorted(list(set(y_true_str) | set(y_pred_triage_dec)))
    cm = confusion_matrix(y_true_str, y_pred_triage_dec, labels=classes_labels)
    
    # Binary metrics (Approved vs Non-Approved)
    y_true_binary = (y_true_str == "Approved").astype(int)
    y_pred_binary = (y_pred_triage_dec == "Approved").astype(int)
    
    # Compute ROC-AUC and PR-AUC using prediction probability
    roc_auc = roc_auc_score(y_true_binary, y_prob_all)
    p_prec, p_rec, _ = precision_recall_curve(y_true_binary, y_prob_all)
    pr_auc = auc(p_rec, p_prec)
    
    # Calculate False Approvals and False Denials
    false_approvals = int(np.sum((y_pred_binary == 1) & (y_true_binary == 0)))
    false_denials = int(np.sum((y_pred_binary == 0) & (y_true_binary == 1)))
    
    print("\n=== Triage Engine Performance ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1-Score: {f1_macro:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"False Approvals: {false_approvals}")
    print(f"False Denials: {false_denials}")
    
    # Save md report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_labels = [str(lbl) for lbl in classes_labels]
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Triage Engine Evaluation Report\n\n")
        f.write("This report evaluates the Prior Authorization Triage and Policy Companion engine (Random Forest prediction combined with CMS Local Coverage Determination rules).\n\n")
        
        f.write("## 1. Overall Performance Metrics\n")
        f.write(f"* **Accuracy**: {acc:.4f}\n")
        f.write(f"* **Macro Precision**: {precision_macro:.4f}\n")
        f.write(f"* **Macro Recall**: {recall_macro:.4f}\n")
        f.write(f"* **Macro F1-Score**: {f1_macro:.4f}\n")
        f.write(f"* **Approved Binary ROC-AUC**: {roc_auc:.4f}\n")
        f.write(f"* **Approved Binary PR-AUC**: {pr_auc:.4f}\n\n")
        
        f.write("## 2. Safety Audit Metrics\n")
        f.write(f"* **False Approvals (Critical Risk)**: {false_approvals}\n")
        f.write("  *Description: The system recommended APPROVE, but the clinical record should have been Denied or Pended.*\n")
        f.write(f"* **False Denials (Friction Risk)**: {false_denials}\n")
        f.write("  *Description: The system recommended Denied or Pended, but the clinical record was Approved.*\n\n")
        
        f.write("## 3. Confusion Matrix\n")
        f.write(f"Labels: `{clean_labels}`\n\n")
        f.write("```\n")
        f.write(str(cm) + "\n")
        f.write("```\n")
        
    print(f"\nEvaluation complete. Portable report saved to {REPORT_PATH}")

if __name__ == "__main__":
    main()
