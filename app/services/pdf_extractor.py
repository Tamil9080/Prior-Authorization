import io
import re
from pathlib import Path
import pypdf

REQUIRED_FIELDS = {
    "patient_id": ["Patient ID", "Member ID"],
    "patient_name": ["Patient Name", "Name", "Member Name"],
    "age": ["Age"],
    "diagnosis": ["Diagnosis", "Diagnosis Description", "Assessment"],
    "diagnosis_code": ["ICD-10 Code", "ICD-10", "Diagnosis Code", "ICD Code"],
    "requested_service": ["Requested Service", "Service Description", "Procedure", "Service"],
    "procedure_code": ["CPT/HCPCS Code", "CPT Code", "HCPCS Code", "Procedure Code", "Service Code"],
    "provider_specialty": ["Provider Specialty", "Specialty"],
    "previous_treatments": ["Previous Treatment", "Previous Treatments", "Prior Therapy", "Conservative Management"],
    "treatment_duration": ["Treatment Duration", "Duration of Treatment", "Trial Duration", "Duration"]
}

def parse_duration_text(text: str) -> float | None:
    """Regex-extracts a number + unit (day/week/month/year) and converts to weeks."""
    if not text:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*(day|week|month|year)s?\b", text.lower())
    if match:
        val = float(match.group(1))
        unit = match.group(2)
        if "day" in unit:
            return val / 7.0
        elif "week" in unit:
            return val
        elif "month" in unit:
            return val * 4.333
        elif "year" in unit:
            return val * 52.0
            
    # Handle descriptive terms if no number
    if "several years" in text.lower() or "years" in text.lower():
        return 104.0
    if "several months" in text.lower():
        return 16.0
    if "several weeks" in text.lower() or "several" in text.lower():
        return 6.0
        
    return None

def extract_patient_fields(pdf_path_or_bytes) -> dict:
    """
    Extracts patient fields from path or bytes of a clinical record PDF.
    Returns: dict with 'extracted_fields' (dict) and 'extraction_warnings' (list).
    """
    text = ""
    if isinstance(pdf_path_or_bytes, (str, Path)):
        try:
            with open(pdf_path_or_bytes, "rb") as f:
                text = _extract_text_from_stream(f)
        except Exception as e:
            return {
                "extracted_fields": {},
                "extraction_warnings": [f"Failed to open PDF file: {str(e)}"]
            }
    elif isinstance(pdf_path_or_bytes, bytes):
        text = _extract_text_from_stream(io.BytesIO(pdf_path_or_bytes))
    else:
        # Assume it's a file stream
        text = _extract_text_from_stream(pdf_path_or_bytes)
        
    if not text:
        return {
            "extracted_fields": {},
            "extraction_warnings": ["Could not extract any text from the PDF file (possibly empty or scanned image)."]
        }
        
    extracted = {}
    
    # 1. Search for structured label:value pairs
    for field_key, labels in REQUIRED_FIELDS.items():
        found = False
        for label in labels:
            pattern = rf"(?:^|\n|\r)\s*{re.escape(label)}\s*:\s*(.*?)(?:\n|\r|\Z)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = match.group(1).strip()
                if val:
                    extracted[field_key] = val
                    found = True
                    break
        # Special case: Age conversion to integer
        if found and field_key == "age":
            try:
                extracted["age"] = int(extracted["age"])
            except ValueError:
                pass
                
    # 2. Fall back to heuristic parsers for missing fields
    extracted = _fallback_heuristics(text, extracted)
    
    # 3. Standardize and cross-populate fields
    # Previous treatments
    prev = extracted.get("previous_treatments") or ""
    extracted["previous_treatment"] = prev
    extracted["previous_treatments"] = prev
    
    # Procedure/Service codes
    code = extracted.get("procedure_code") or ""
    extracted["procedure_code"] = code
    extracted["service_code"] = code
    
    # Service Description
    desc = extracted.get("requested_service") or ""
    extracted["requested_service"] = desc
    extracted["service_description"] = desc
    
    # Treatment duration conversion to weeks
    dur_text = extracted.get("treatment_duration")
    if dur_text:
        weeks = parse_duration_text(dur_text)
        if weeks is not None:
            extracted["treatment_duration_weeks"] = round(weeks, 2)
        else:
            extracted["treatment_duration_weeks"] = None
    else:
        extracted["treatment_duration_weeks"] = None
        
    # Urgency, Plan, Network, Referral, and Insurer Defaults
    if "urgency" not in extracted:
        extracted["urgency"] = "Routine"
    if "plan_type" not in extracted:
        plan_type = "PPO"
        for plan in ["PPO", "HMO", "EPO", "POS"]:
            if re.search(rf"\b{plan}\b", text):
                plan_type = plan
                break
        extracted["plan_type"] = plan_type
    if "network_status" not in extracted:
        network_status = "In-Network"
        if re.search(r"\b(out[- ]of[- ]network|oon|non[- ]participating|non[- ]network)\b", text, re.IGNORECASE):
            network_status = "Out-of-Network"
        extracted["network_status"] = network_status
    if "pcp_referral" not in extracted:
        pcp_referral = "No"
        if re.search(r"\b(referral on file|pcp referral|referred by pcp|pcp referral status\s*:\s*yes)\b", text, re.IGNORECASE):
            pcp_referral = "Yes"
        extracted["pcp_referral"] = pcp_referral
    if "insurer_or_payer" not in extracted:
        extracted["insurer_or_payer"] = "Medicare (CMS)"
        
    # Clinical History / lab defaults if not found
    if "clinical_history" not in extracted:
        # Fall back to first 300 chars of text
        extracted["clinical_history"] = text[:300].strip()
    if "lab_results" not in extracted:
        extracted["lab_results"] = "None documented"
        
    # 4. Generate warnings for completely missing required fields
    warnings = []
    for field_key in REQUIRED_FIELDS.keys():
        if field_key not in extracted or extracted[field_key] is None or extracted[field_key] == "":
            warnings.append(field_key)
            
    return {
        "extracted_fields": extracted,
        "extraction_warnings": warnings
    }

def _extract_text_from_stream(stream) -> str:
    try:
        reader = pypdf.PdfReader(stream)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading PDF stream: {e}")
        return ""

def _fallback_heuristics(text: str, current_fields: dict) -> dict:
    updated = dict(current_fields)
    
    # Age fallback
    if "age" not in updated:
        age_match = re.search(r"\bage\s*:\s*(\d{1,3})\b", text, re.IGNORECASE)
        if not age_match:
            age_match = re.search(r"\b(\d{1,3})\s*(?:year\s*-\s*old|years?\s*old|yo)\b", text, re.IGNORECASE)
        if age_match:
            try:
                updated["age"] = int(age_match.group(1))
            except ValueError:
                pass
            
    # Diagnosis code fallback (ICD-10)
    if "diagnosis_code" not in updated:
        diag_matches = re.findall(r"\b([A-Z]\d{2}(?:\.\d{1,3})?)\b", text)
        excluded_prefixes = ["E", "J", "G"]
        for code in diag_matches:
            if code[0] not in excluded_prefixes:
                updated["diagnosis_code"] = code
                break
                
    # Procedure code fallback (CPT/HCPCS)
    if "procedure_code" not in updated:
        hcpcs_match = re.search(r"\b(E0601|E2103|J3032|J1745|J0135|J8499|G0105|G0121|G0104|G0328|S0285)\b", text, re.IGNORECASE)
        if hcpcs_match:
            updated["procedure_code"] = hcpcs_match.group(1).upper()
        else:
            cpts = re.findall(r"\b(\d{5})\b", text)
            supported_cpts = ["72148", "27447", "90867", "93458", "45378", "45380", "45381", "45384", "45385", "45388", "99245", "99417"]
            for c in cpts:
                if c in supported_cpts:
                    updated["procedure_code"] = c
                    break
            if "procedure_code" not in updated and cpts:
                for c in cpts:
                    if not (c.startswith("19") or c.startswith("20")):
                        updated["procedure_code"] = c
                        break
                        
    # Provider specialty fallback
    if "provider_specialty" not in updated:
        specialties_map = {
            "orthopedic": "Orthopedics",
            "ortho": "Orthopedics",
            "rheumatology": "Rheumatology",
            "rheum": "Rheumatology",
            "endocrin": "Endocrinology",
            "pulmon": "Pulmonology",
            "psychiat": "Psychiatry",
            "cardio": "Cardiology",
            "gastro": "Gastroenterology",
            "neuro": "Neurology"
        }
        for kw, val in specialties_map.items():
            if re.search(rf"\b{kw}", text, re.IGNORECASE):
                updated["provider_specialty"] = val
                break
                
    # Insurer fallback
    if "insurer_or_payer" not in updated:
        payers_map = {
            "unitedhealthcare": "UnitedHealthcare",
            "uhc": "UnitedHealthcare",
            "aetna": "Aetna",
            "cigna": "Cigna",
            "humana": "Humana",
            "blue cross": "Blue Cross Blue Shield",
            "bcbs": "Blue Cross Blue Shield",
            "medicare": "Medicare (CMS)",
            "cms": "Medicare (CMS)"
        }
        for kw, val in payers_map.items():
            if re.search(rf"\b{kw}\b", text, re.IGNORECASE):
                updated["insurer_or_payer"] = val
                break

    # Diagnosis fallback
    if "diagnosis" not in updated:
        diag_match = re.search(r"(?:diagnosis|dx|assessment|impression)\s*:?\s*([A-Za-z0-9\s,\.-]+)", text, re.IGNORECASE)
        if diag_match:
            updated["diagnosis"] = diag_match.group(1).strip()

    # Requested service fallback
    if "requested_service" not in updated:
        desc_match = re.search(r"(?:requested service|procedure|service|ordered)\s*:?\s*(.*?)(?:\n|\r|\Z)", text, re.IGNORECASE)
        if desc_match:
            updated["requested_service"] = desc_match.group(1).strip()

    # Previous treatment fallback
    if "previous_treatments" not in updated:
        treat_keywords = ["physical therapy", "pt", "nsaid", "ibuprofen", "naproxen", "meloxicam", "failed", "trial", "corticosteroid", "injection", "sertraline", "venlafaxine", "psychotherapy", "cbt"]
        found_treatments = []
        for line in text.split('\n'):
            for kw in treat_keywords:
                if kw in line.lower() and line.strip() not in found_treatments:
                    found_treatments.append(line.strip())
                    break
        if found_treatments:
            updated["previous_treatments"] = "; ".join(found_treatments)[:250]

    # Treatment duration fallback
    if "treatment_duration" not in updated:
        duration_match = re.search(r"\b(\d+(?:\.\d+)?\s*(?:day|week|month|year)s?\b)", text, re.IGNORECASE)
        if duration_match:
            updated["treatment_duration"] = duration_match.group(1)

    return updated
