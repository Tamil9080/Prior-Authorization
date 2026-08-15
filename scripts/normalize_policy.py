import json
from pathlib import Path
import pandas as pd

# Paths
script_dir = Path(__file__).resolve().parent
PROJECT_ROOT = script_dir.parent
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "cms_mcd" / "cms_mcd_raw.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "coverage_rules"

def main():
    if not RAW_FILE.exists():
        print(f"Error: Raw file {RAW_FILE} does not exist. Run ingest_cms.py first!")
        return

    with open(RAW_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    normalized_records = []
    for item in raw_data:
        # Standardize structure and mapping to target columns
        record = {
            "policy_id": item.get("policy_id"),
            "insurer_or_payer": "Medicare (CMS)",
            "policy_type": item.get("policy_type"),
            "policy_title": item.get("policy_title"),
            "service_code": item.get("service_code"),
            "service_description": item.get("service_description"),
            "diagnosis_code": item.get("diagnosis_code"),
            "coverage_status": item.get("coverage_status"),
            "prior_authorization_required": item.get("prior_authorization_required"),
            "coverage_rule": item.get("medical_necessity_rule"),
            "medical_necessity": item.get("medical_necessity_rule"),
            "documentation_required": item.get("documentation_required"),
            "age_requirement": "No age restriction",
            "frequency_limit": item.get("frequency_limit", "None"),
            "quantity_limit": item.get("quantity_limit", "None"),
            "limitations": item.get("limitations", "None"),
            "exclusions": item.get("exclusions", "None"),
            "effective_date": item.get("effective_date"),
            "source_url": item.get("source_url")
        }
        normalized_records.append(record)

    df = pd.DataFrame(normalized_records)
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    csv_file = PROCESSED_DIR / "coverage_rules.csv"
    df.to_csv(csv_file, index=False)
    print(f"Normalized policy database written to {csv_file}")
    
    # Save as Parquet (requires pyarrow or fastparquet)
    parquet_file = PROCESSED_DIR / "coverage_rules.parquet"
    try:
        df.to_parquet(parquet_file, index=False, engine='pyarrow')
        print(f"Normalized policy database written to {parquet_file}")
    except ImportError:
        # Attempt fallback to fastparquet or report issue
        try:
            df.to_parquet(parquet_file, index=False, engine='fastparquet')
            print(f"Normalized policy database written to {parquet_file}")
        except Exception as e:
            print(f"Warning: Could not save as Parquet due to missing libraries: {e}")
            print("Please run `pip install pyarrow` to enable Parquet storage.")

if __name__ == "__main__":
    main()
