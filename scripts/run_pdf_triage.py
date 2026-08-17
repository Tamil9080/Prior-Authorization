import sys
import os
import json
import argparse
from pathlib import Path
import pandas as pd
import joblib

# Absolute project root path
ROOT = Path("d:/cts hackthon/new1")
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "app"))

from utils.pdf_parser import extract_text_from_pdf, parse_clinical_details
from policy.policy_service import match_policy
from rules.rule_evaluator import evaluate_rules
from services.rag_service import rag_engine

def run_pdf_triage(pdf_path: str, output_json: str = None):
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"Error: PDF file '{pdf_path}' does not exist.")
        sys.exit(1)
        
    print(f"[{pdf_file.name}] Ingesting and extracting text...")
    with open(pdf_file, "rb") as f:
        text = extract_text_from_pdf(f)
        
    if not text:
        print("Error: Could not extract text from PDF.")
        sys.exit(1)
        
    print("Parsing clinical parameters...")
    details = parse_clinical_details(text)
    
    # Save/display JSON
    print("\n--- Extracted JSON Parameters ---")
    print(json.dumps(details, indent=2))
    
    if output_json:
        out_path = Path(output_json)
        out_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
        print(f"Saved extracted parameters to: {out_path}")
        
    # Load ML Model
    print("\n--- Running Machine Learning Triage ---")
    model_path = ROOT / "models" / "random_forest" / "clinical_rf_model.joblib"
    ml_prediction = "UNKNOWN"
    ml_confidence = 0.5
    
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            # Match app features dictionary
            features_dict = {
                "age": details["age"],
                "diagnosis": details["diagnosis"],
                "diagnosis_code": details["diagnosis_code"],
                "requested_service": details["service_description"],
                "procedure_code": details["service_code"],
                "clinical_history": details["clinical_history"],
                "previous_treatments": details["previous_treatment"],
                "medications": details.get("medications", "None documented"),
                "lab_results": details["lab_results"],
                "provider_specialty": details["provider_specialty"],
                "urgency": details["urgency"],
                "policy_id": "POL100001" # Default fallback
            }
            
            row = pd.DataFrame([features_dict])
            text_cols = ['diagnosis', 'requested_service', 'clinical_history', 'previous_treatments', 'medications', 'lab_results']
            cat_cols = ['diagnosis_code', 'procedure_code', 'provider_specialty', 'urgency', 'policy_id']
            num_cols = ['age']
            
            row_dict = {}
            for col in num_cols:
                row_dict[col] = pd.to_numeric(row[col].iloc[0], errors='coerce')
            for col in cat_cols:
                row_dict[col] = str(row[col].iloc[0])
            for col in text_cols:
                row_dict[col] = str(row[col].iloc[0])
                
            aligned_row = pd.DataFrame([row_dict])
            pred = model.predict(aligned_row)[0]
            ml_prediction = "APPROVED" if pred.upper() == "APPROVED" else "DENIED"
            
            if hasattr(model.named_steps["classifier"], "predict_proba"):
                probabilities = model.predict_proba(aligned_row)[0]
                classes = list(model.named_steps["classifier"].classes_)
                probs = {str(c).upper(): float(probabilities[i]) for i, c in enumerate(classes)}
                ml_confidence = float(probs.get(ml_prediction, 0.5))
                
            print(f"ML Model Prediction: {ml_prediction} (Confidence: {ml_confidence:.2%})")
        except Exception as e:
            print(f"Error executing ML prediction: {e}")
    else:
        print(f"Clinical model not found at {model_path}. Skipping ML prediction.")

    # Match coverage policy
    print("\n--- Matching Payer Coverage Policy ---")
    insurer = details["insurer_or_payer"]
    policy = match_policy(
        service_code=details["service_code"],
        diagnosis_code=details["diagnosis_code"],
        service_description=details["service_description"],
        diagnosis=details["diagnosis"],
        insurer_or_payer=insurer
    )
    
    policy_found = policy is not None
    if policy_found:
        print(f"Matched Policy ID: {policy.get('policy_id')}")
        print(f"Policy Title: {policy.get('policy_title')}")
        print(f"Policy Payer: {policy.get('insurer_or_payer')}")
    else:
        print("No matching coverage policy found in database.")

    # Rule checks
    print("\n--- Evaluating Clinical Policy Rules ---")
    final_recommendation = "PEND"
    reason = "Additional documentation is required."
    
    if not policy_found:
        final_recommendation = "PEND"
        reason = "No covered policy matches this code. Escalated to utilization review."
    else:
        rules_eval = evaluate_rules(features_dict, policy)
        print("Evaluated Rules:")
        for r in rules_eval:
            print(f"  * Rule '{r['rule']}': {r['status']} - {r['evidence']}")
            
        statuses = [r["status"] for r in rules_eval]
        if "NOT_MET" in statuses:
            final_recommendation = "DENY"
            denied_rules = [r["rule"] for r in rules_eval if r["status"] == "NOT_MET"]
            reason = f"Coverage criteria not met. Failed rules: {', '.join(denied_rules)}."
        elif "INSUFFICIENT" in statuses:
            final_recommendation = "PEND"
            missing_rules = [r["rule"] for r in rules_eval if r["status"] == "INSUFFICIENT"]
            reason = f"Required information is missing: {', '.join(missing_rules)}."
        else:
            final_recommendation = "APPROVE"
            reason = "All policy criteria met. Prior authorization recommended for approval."

    # Citations
    print("\n--- Retrieving Clinical Citations ---")
    try:
        # Initialize RAG first
        rag_engine.initialize()
        rag_query = f"{details['service_description']} {details['diagnosis_code']} {details['service_code']} necessity criteria"
        citations = rag_engine.retrieve(rag_query, top_k=2)
        for i, c in enumerate(citations):
            print(f"Citation {i+1} [{c['source']}]:")
            print(f"  Snippet: {c['text'][:120]}...")
    except Exception as e:
        print(f"Error getting RAG citations: {e}")

    # Summary
    print("\n==================================================")
    print("                TRIAGE RESOLUTION SUMMARY         ")
    print("==================================================")
    print(f"Patient Name         : {details['patient_name']}")
    print(f"Patient ID           : {details['patient_id']}")
    print(f"Procedure Code       : {details['service_code']}")
    print(f"Diagnosis Code       : {details['diagnosis_code']}")
    print(f"ML Recommendation    : {ml_prediction}")
    print(f"Final Triage Decision: {final_recommendation}")
    print(f"Reason Summary       : {reason}")
    print("==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Prior-Auth Triage directly on a clinical PDF.")
    parser.add_argument("pdf", help="Path to patient clinical record PDF")
    parser.add_argument("--output", default=None, help="Save parsed details to this JSON file path")
    args = parser.parse_args()
    
    run_pdf_triage(args.pdf, args.output)
