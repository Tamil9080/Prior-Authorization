import sys
import os
import json
import requests
from pathlib import Path

# Setup paths
ROOT = Path("d:/cts hackthon/new1")
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "app"))

from reportlab.pdfgen import canvas
from services.pdf_extractor import extract_patient_fields

def generate_test_pdf(filename: str):
    """Generate a structured label:value PDF for testing."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    c = canvas.Canvas(filename)
    
    # Write lines of text in label:value format
    text_lines = [
        "Patient ID: SYN-PT-1007",
        "Patient Name: Eleanor Vance",
        "Age: 41",
        "Diagnosis: Lumbar radiculopathy",
        "ICD-10 Code: M54.16",
        "CPT/HCPCS Code: 72148",
        "Previous Treatment: Physical Therapy",
        "Treatment Duration: 50 days"
    ]
    
    y = 750
    for line in text_lines:
        c.drawString(100, y, line)
        y -= 25
        
    c.save()
    print(f"Generated test PDF at: {filename}")

def test_extraction(pdf_path: str):
    print("\n--- Testing PDF Extractor Service ---")
    result = extract_patient_fields(pdf_path)
    fields = result.get("extracted_fields", {})
    warnings = result.get("extraction_warnings", [])
    
    print("Extracted fields:")
    print(json.dumps(fields, indent=2))
    print("Warnings:", warnings)
    
    # Assertions
    assert fields.get("patient_id") == "SYN-PT-1007", f"Expected patient_id to be SYN-PT-1007, got {fields.get('patient_id')}"
    assert fields.get("patient_name") == "Eleanor Vance", "Patient name extraction failed"
    assert fields.get("age") == 41, f"Expected age to be integer 41, got {fields.get('age')} ({type(fields.get('age'))})"
    assert fields.get("diagnosis_code") == "M54.16", "Diagnosis code extraction failed"
    assert fields.get("service_code") == "72148", "CPT/HCPCS code extraction failed"
    assert "Physical Therapy" in fields.get("previous_treatments", ""), "Previous treatments extraction failed"
    
    # Check numeric conversion: 50 days / 7 = 7.14 weeks
    dur_weeks = fields.get("treatment_duration_weeks")
    assert dur_weeks is not None, "Treatment duration weeks should not be None"
    assert 7.1 <= dur_weeks <= 7.2, f"Expected duration weeks to be ~7.14, got {dur_weeks}"
    
    # Verify standardizer populated both previous_treatment and previous_treatments
    assert fields.get("previous_treatment") == fields.get("previous_treatments"), "Inconsistent previous treatment keys"
    
    print("All extractor service assertions passed successfully!")

def test_api_endpoint(pdf_path: str):
    print("\n--- Testing HTTP POST /api/prior-authorization/check-pdf ---")
    url = "http://127.0.0.1:5000/api/prior-authorization/check-pdf"
    
    try:
        with open(pdf_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)
            
        print(f"Status Code: {response.status_code}")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        print("Response Keys:", list(data.keys()))
        
        extracted = data.get("extracted_fields", {})
        warnings = data.get("extraction_warnings", [])
        decision = data.get("decision_result", {})
        
        print("\nExtracted details via API:")
        print(f"  Patient: {extracted.get('patient_name')} ({extracted.get('patient_id')})")
        print(f"  Age: {extracted.get('age')}")
        print(f"  Procedure Code: {extracted.get('service_code')}")
        print(f"  ICD-10 Code: {extracted.get('diagnosis_code')}")
        print(f"  Parsed Duration (Weeks): {extracted.get('treatment_duration_weeks')}")
        
        print("\nTriage Decision:")
        print(f"  Final Recommendation: {decision.get('final_recommendation')}")
        print(f"  ML Classifier Prediction: {decision.get('ml_prediction')} (Confidence: {decision.get('ml_confidence')})")
        print(f"  Matched Policy ID: {decision.get('policy_id')}")
        print(f"  Reason: {decision.get('reason')}")
        
        # Verify the decision outcome
        # Since Lumbar Spine MRI (72148) matches policy 38429, and patient has >= 6 weeks duration (7.14)
        # and has conservative management (PT), the rules should MET and resolve to APPROVE
        assert decision.get("final_recommendation") == "APPROVE", f"Expected decision APPROVE, got {decision.get('final_recommendation')}"
        
        print("\nAll HTTP API assertions passed successfully!")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to local Flask server. Is it running at http://127.0.0.1:5000?")
        sys.exit(1)

if __name__ == "__main__":
    pdf_file = str(ROOT / "scratch" / "sample_patient_record.pdf")
    generate_test_pdf(pdf_file)
    test_extraction(pdf_file)
    test_api_endpoint(pdf_file)
    print("\n=============================================")
    print("      ALL TESTS PASSED SUCCESSFULLY!         ")
    print("=============================================")
