import sys
import uuid
import pandas as pd
from pathlib import Path
from flask import Blueprint, request, jsonify

# Add app directory to path if needed
sys.path.append(str(Path(__file__).resolve().parent.parent))

from policy import policy_service
from rules import rule_evaluator
from services.rag_service import rag_engine
from utils import audit_logger

check_blueprint = Blueprint("check", __name__)

# We will import the trained models from the main app context or reload them locally
clinical_model = None

def load_local_model():
    global clinical_model
    if clinical_model is not None:
        return clinical_model
        
    # Attempt to load from models/random_forest/
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    model_path = PROJECT_ROOT / "models" / "random_forest" / "clinical_rf_model.joblib"
    if model_path.exists():
        try:
            import joblib
            clinical_model = joblib.load(model_path)
            print("check.py: Loaded clinical model from random_forest directory.")
        except Exception as e:
            print(f"check.py: Error loading model: {e}")
    return clinical_model

@check_blueprint.route("/api/prior-authorization/check", methods=["POST"])
def check_prior_authorization():
    model = load_local_model()
    
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided."}), 400
            
        patient_id = str(data.get("patient_id", f"PAT{uuid.uuid4().hex[:6].upper()}"))
        age = data.get("age", 45)
        diag_code = str(data.get("diagnosis_code", "")).strip().upper()
        service_code = str(data.get("service_code", "")).strip().upper()
        service_desc = str(data.get("service_description", "")).strip()
        prev_treatment = str(data.get("previous_treatment", "")).strip()
        treatment_dur = data.get("treatment_duration", 0)
        
        # Build features for Random Forest classifier
        # RF model expects: age, diagnosis, diagnosis_code, requested_service, procedure_code, clinical_history, previous_treatments, medications, lab_results, provider_specialty, urgency, policy_id
        clinical_history = str(data.get("clinical_history", f"{age}-year-old patient with diagnosis code {diag_code}. Symptom duration {treatment_dur} weeks."))
        meds = str(data.get("medications", "None documented"))
        labs = str(data.get("lab_results", "None documented"))
        provider_specialty = str(data.get("provider_specialty", "General Medicine"))
        urgency = str(data.get("urgency", "Routine"))
        policy_id = str(data.get("policy_id", "POL100001"))
        diagnosis_text = str(data.get("diagnosis", service_desc))
        
        features_dict = {
            "age": age,
            "diagnosis": diagnosis_text,
            "diagnosis_code": diag_code,
            "requested_service": service_desc,
            "procedure_code": service_code,
            "clinical_history": clinical_history,
            "previous_treatments": prev_treatment,
            "medications": meds,
            "lab_results": labs,
            "provider_specialty": provider_specialty,
            "urgency": urgency,
            "policy_id": policy_id
        }
        
        # 1. Run Machine Learning Model (Prediction)
        ml_prediction = "DENIED"
        ml_confidence = 0.5
        probs = {}
        
        if model:
            row = pd.DataFrame([features_dict])
            # Align features
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
        
        # 2. Retrieve CMS Coverage Policy
        policy = policy_service.match_policy(
            service_code=service_code,
            diagnosis_code=diag_code,
            service_description=service_desc,
            diagnosis=diagnosis_text
        )
        
        policy_found = policy is not None
        p_id = policy.get("policy_id", "N/A") if policy_found else "N/A"
        p_title = policy.get("policy_title", "N/A") if policy_found else "N/A"
        p_source = "CMS MCD" if policy_found else "N/A"
        
        # 3. Rule-by-rule evaluation
        rules_eval = rule_evaluator.evaluate_rules(features_dict, policy)
        
        # 4. Final Decision Logic
        # All required rules MET -> APPROVE (or match ML prediction if more conservative)
        # Any rule NOT_MET -> DENY
        # Any rule INSUFFICIENT -> PEND
        final_recommendation = "PEND"
        reason = "Additional documentation is required."
        
        if not policy_found:
            final_recommendation = "PEND"
            reason = "No covered policy matches this code. Escalated to utilization review."
        else:
            statuses = [r["status"] for r in rules_eval]
            
            if "NOT_MET" in statuses:
                final_recommendation = "DENY"
                # Find the rule that caused the denial
                denied_rules = [r["rule"] for r in rules_eval if r["status"] == "NOT_MET"]
                reason = f"Coverage criteria not met. Failed rules: {', '.join(denied_rules)}."
            elif "INSUFFICIENT" in statuses:
                final_recommendation = "PEND"
                missing_rules = [r["rule"] for r in rules_eval if r["status"] == "INSUFFICIENT"]
                reason = f"Required information is missing: {', '.join(missing_rules)}."
            else:
                final_recommendation = "APPROVE"
                reason = "All policy criteria met. Prior authorization recommended for approval."
                
        # 5. Retrieve RAG Citations
        rag_query = f"{service_desc} {diag_code} {service_code} medical necessity criteria"
        citations = rag_engine.retrieve(rag_query, top_k=3)
        
        # 6. Generate request ID and write Audit Log
        request_id = f"REQ{uuid.uuid4().hex[:6].upper()}"
        audit_entry = audit_logger.log_triage_request(
            request_id=request_id,
            patient_id=patient_id,
            ml_prediction=ml_prediction,
            ml_confidence=ml_confidence,
            policy_id=p_id,
            rules_evaluated=[r["rule"] for r in rules_eval],
            rule_results={r["rule"]: r["status"] for r in rules_eval},
            final_recommendation=final_recommendation
        )
        
        return jsonify({
            "request_id": request_id,
            "patient_id": patient_id,
            "ml_prediction": ml_prediction,
            "ml_confidence": ml_confidence,
            "policy_found": policy_found,
            "policy_id": p_id,
            "policy_title": p_title,
            "policy_source": p_source,
            "rules": rules_eval,
            "final_recommendation": final_recommendation,
            "reason": reason,
            "evidence_chunks": citations,
            "timeframe": "Within 5-7 business days" if urgency.lower() == "routine" else "Within 24-72 hours"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
