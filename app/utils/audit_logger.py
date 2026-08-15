import json
import os
from datetime import datetime
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AUDIT_FILE = PROJECT_ROOT / "data" / "processed" / "pa" / "audit_log.jsonl"

def log_triage_request(
    request_id,
    patient_id,
    ml_prediction,
    ml_confidence,
    policy_id,
    rules_evaluated,
    rule_results,
    final_recommendation
):
    """
    Log an initial prior authorization triage check request to the audit trail.
    """
    # Create directory if it doesn't exist
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    log_entry = {
        "request_id": request_id,
        "patient_id": patient_id,
        "request_timestamp": datetime.utcnow().isoformat() + "Z",
        "model_version": "RandomForest_v2.0",
        "ml_prediction": ml_prediction,
        "ml_confidence": float(ml_confidence),
        "policy_id": policy_id,
        "policy_version": "CMS_MCD_2026",
        "rules_evaluated": rules_evaluated,
        "rule_results": rule_results,
        "final_recommendation": final_recommendation,
        "human_review": False,
        "final_decision": final_recommendation
    }
    
    with open(AUDIT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"Logged request {request_id} to audit trail.")
    return log_entry

def log_reviewer_decision(
    request_id,
    reviewer_decision,
    review_reason
):
    """
    Record a human reviewer's final decision override for an existing request.
    Updates the matching log entry in-place or appends an override record.
    """
    if not AUDIT_FILE.exists():
        print(f"Warning: Audit log does not exist yet. Cannot update decision.")
        return None
        
    entries = []
    updated = False
    updated_entry = None
    
    # Read all lines
    with open(AUDIT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("request_id") == request_id:
                # Update in-place
                entry["human_review"] = True
                entry["reviewer_decision"] = reviewer_decision
                entry["final_decision"] = reviewer_decision
                entry["review_reason"] = review_reason
                entry["review_timestamp"] = datetime.utcnow().isoformat() + "Z"
                updated_entry = entry
                updated = True
            entries.append(entry)
            
    # Write back if updated
    if updated:
        with open(AUDIT_FILE, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        print(f"Updated request {request_id} with reviewer decision: {reviewer_decision}.")
        return updated_entry
    else:
        print(f"Warning: Request ID {request_id} not found in audit log.")
        return None
