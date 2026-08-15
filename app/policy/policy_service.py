import os
import pandas as pd
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "coverage_rules" / "coverage_rules.csv"
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "coverage_rules" / "coverage_rules.parquet"

_policy_df = None

def get_policy_database():
    """
    Load the policy database into memory. Attempts Parquet first, then falls back to CSV.
    """
    global _policy_df
    if _policy_df is not None:
        return _policy_df
        
    if PARQUET_PATH.exists():
        try:
            _policy_df = pd.read_parquet(PARQUET_PATH)
            print("Loaded policy database from Parquet format.")
            return _policy_df
        except Exception as e:
            print(f"Error reading policy Parquet: {e}. Falling back to CSV.")
            
    if CSV_PATH.exists():
        try:
            _policy_df = pd.read_csv(CSV_PATH)
            print("Loaded policy database from CSV format.")
            return _policy_df
        except Exception as e:
            print(f"Error reading policy CSV: {e}")
            
    print("Warning: Policy database files not found. Creating empty fallback DataFrame.")
    _policy_df = pd.DataFrame(columns=[
        "policy_id", "insurer_or_payer", "policy_type", "policy_title", 
        "service_code", "service_description", "diagnosis_code", "coverage_status", 
        "prior_authorization_required", "coverage_rule", "medical_necessity", 
        "documentation_required", "age_requirement", "frequency_limit", 
        "quantity_limit", "limitations", "exclusions", "effective_date", "source_url"
    ])
    return _policy_df

def match_policy(
    service_code=None, 
    diagnosis_code=None, 
    service_description=None, 
    diagnosis=None,
    policy_title=None,
    insurer_or_payer=None
):
    """
    Query the policy database based on the request parameters.
    Uses a weighted scoring system to find the most relevant policy.
    Returns the best matching policy dict, or None if no match is found.
    """
    df = get_policy_database()
    if df.empty:
        return None
        
    best_policy = None
    best_score = 0
    
    service_code = str(service_code).strip().upper() if service_code else ""
    diagnosis_code = str(diagnosis_code).strip().upper() if diagnosis_code else ""
    service_desc = str(service_description).strip().lower() if service_description else ""
    diag_desc = str(diagnosis).strip().lower() if diagnosis else ""
    pol_title = str(policy_title).strip().lower() if policy_title else ""
    payer = str(insurer_or_payer).strip().lower() if insurer_or_payer else ""
    
    for _, row in df.iterrows():
        score = 0
        
        # 1. Exact match on service code (CPT / HCPCS)
        row_svc_code = str(row.get("service_code", "")).strip().upper()
        if service_code and row_svc_code == service_code:
            score += 10
            
        # 2. Match on diagnosis code (ICD-10 list)
        row_diag_codes = str(row.get("diagnosis_code", "")).strip().upper()
        if diagnosis_code:
            # Check if diagnosis code is present in comma/semicolon-separated list
            code_list = [c.strip() for c in row_diag_codes.replace(";", ",").split(",")]
            if diagnosis_code in code_list:
                score += 5
            elif diagnosis_code[:3] in [c[:3] for c in code_list]:  # Match category level (e.g. M17)
                score += 2
                
        # 3. Substring match on service description / title
        row_svc_desc = str(row.get("service_description", "")).strip().lower()
        row_title = str(row.get("policy_title", "")).strip().lower()
        if service_desc:
            if service_desc in row_svc_desc:
                score += 3
            if service_desc in row_title:
                score += 2
                
        # 4. Substring match on diagnosis text
        row_rule = str(row.get("coverage_rule", "")).strip().lower()
        if diag_desc:
            if diag_desc in row_title:
                score += 2
            if diag_desc in row_rule:
                score += 1
                
        # 5. Policy title search
        if pol_title and pol_title in row_title:
            score += 4
            
        # 6. Payer filter (highest priority weight to guarantee selected insurer is matched)
        row_payer = str(row.get("insurer_or_payer", "")).strip().lower()
        if payer:
            clean_payer = payer.replace("(cms)", "").strip()
            clean_row_payer = row_payer.replace("(cms)", "").strip()
            if clean_payer in clean_row_payer or clean_row_payer in clean_payer:
                score += 30
            
        # Update best match
        if score > best_score:
            best_score = score
            best_policy = row.to_dict()
            best_policy["relevance_score"] = float(score)
            
    if best_score > 0:
        return best_policy
    return None
