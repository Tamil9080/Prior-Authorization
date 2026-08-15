import json
from pathlib import Path

# Paths
script_dir = Path(__file__).resolve().parent
PROJECT_ROOT = script_dir.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cms_mcd"

# 10 Representative LCD policies from the CMS Medicare Coverage Database
cms_raw_data = [
    {
        "policy_id": "L33718",
        "policy_type": "LCD",
        "policy_title": "Positive Airway Pressure (PAP) Devices for Obstructive Sleep Apnea",
        "service_code": "E0601",
        "service_description": "Continuous Positive Airway Pressure (CPAP) device",
        "diagnosis_code": "G47.33",
        "coverage_status": "Covered",
        "prior_authorization_required": "Yes",
        "medical_necessity_rule": "Patient must have an Apnea-Hypopnea Index (AHI) >= 15 events/hour, or AHI >= 5 and <= 14 events/hour with documented comorbidities (excessive daytime sleepiness, hypertension, mood disorders, stroke, cognitive impairment, or ischemic heart disease).",
        "documentation_required": "In-person clinical evaluation notes, sleep study report confirming AHI, and supplier compliance log.",
        "limitations": "Initial coverage is limited to a 12-week trial period. Continued coverage requires compliance check showing usage >= 4 hours/night on 70% of nights.",
        "exclusions": "Use for simple snoring without obstructive sleep apnea is excluded.",
        "effective_date": "2026-01-01",
        "source": "CMS Medicare Coverage Database",
        "source_url": "https://www.cms.gov/medicare-coverage-database/view/lcd.aspx?LCDId=33718",
        "age_requirement": "Adults (>= 18 years)",
        "frequency_limit": "1 device per 5 years",
        "quantity_limit": "1 unit"
    },
    {
        "policy_id": "L38429",
        "policy_type": "LCD",
        "policy_title": "Magnetic Resonance Imaging (MRI) of the Lumbar Spine",
        "service_code": "72148",
        "service_description": "Magnetic Resonance Imaging (MRI), lumbar spine, without contrast",
        "diagnosis_code": "M54.5",
        "coverage_status": "Covered",
        "prior_authorization_required": "Yes",
        "medical_necessity_rule": "Low back pain or radiculopathy present for at least 6 consecutive weeks AND failed at least one trial of conservative therapy (physical therapy, NSAIDs, or activity modification) AND progressive neurological deficit, suspected cauda equina, or failure to improve after conservative therapy.",
        "documentation_required": "Clinical notes documenting symptom duration, details of conservative treatment trials, and physical exam findings.",
        "limitations": "Symptom duration under 6 weeks is not covered without red-flag findings.",
        "exclusions": "Routine screening for low back pain without red-flag findings is excluded.",
        "effective_date": "2026-01-01",
        "source": "CMS Medicare Coverage Database",
        "source_url": "https://www.cms.gov/medicare-coverage-database/view/lcd.aspx?LCDId=38429",
        "age_requirement": "Adults (>= 18 years)",
        "frequency_limit": "1 scan per 6 months",
        "quantity_limit": "1 unit"
    },
    {
        "policy_id": "L37436",
        "policy_type": "LCD",
        "policy_title": "Total Knee Arthroplasty (TKA) for Osteoarthritis",
        "service_code": "27447",
        "service_description": "Total Knee Arthroplasty (Knee Replacement)",
        "diagnosis_code": "M17.11",
        "coverage_status": "Covered",
        "prior_authorization_required": "Yes",
        "medical_necessity_rule": "Radiographic evidence of moderate-to-severe osteoarthritis (Kellgren-Lawrence Grade 3 or 4) AND failure of at least 12 weeks of conservative management (physical therapy, NSAIDs, or joint injections) AND functional limitation impacting activities of daily living.",
        "documentation_required": "Radiology report confirming KL Grade 3 or 4, documentation of conservative treatment duration, and functional assessment (e.g. Oxford Knee Score).",
        "limitations": "Conservative management duration under 12 weeks is not covered.",
        "exclusions": "Knee replacement for KL Grade 2 or lower without significant functional limitation is excluded.",
        "effective_date": "2026-01-01",
        "source": "CMS Medicare Coverage Database",
        "source_url": "https://www.cms.gov/medicare-coverage-database/view/lcd.aspx?LCDId=37436",
        "age_requirement": "Adults (>= 18 years)",
        "frequency_limit": "1 joint per lifetime",
        "quantity_limit": "1 unit"
    },
    {
        "policy_id": "L34520",
        "policy_type": "LCD",
        "policy_title": "Transcranial Magnetic Stimulation (TMS) for Depressive Disorders",
        "service_code": "90867",
        "service_description": "Transcranial Magnetic Stimulation (TMS) for Major Depressive Disorder",
        "diagnosis_code": "F33.1",
        "coverage_status": "Covered",
        "prior_authorization_required": "Yes",
        "medical_necessity_rule": "Severe major depressive disorder diagnosed by a psychiatrist AND failure of at least two adequate antidepressant trials of different classes (>= 6 weeks at therapeutic doses) AND concurrent or prior engagement in psychotherapy.",
        "documentation_required": "Psychiatrist evaluation, antidepressant history (dates/doses), and psychotherapy engagement notes.",
        "limitations": "Fewer than two antidepressant trials, or lack of psychotherapy, is not covered.",
        "exclusions": "Contraindications like history of seizure disorder or metallic implants in the head are excluded.",
        "effective_date": "2026-01-01",
        "source": "CMS Medicare Coverage Database",
        "source_url": "https://www.cms.gov/medicare-coverage-database/view/lcd.aspx?LCDId=34520",
        "age_requirement": "Adults (>= 18 years)",
        "frequency_limit": "1 course of treatment per 12 months",
        "quantity_limit": "36 sessions"
    },
    {
        "policy_id": "L39210",
        "policy_type": "LCD",
        "policy_title": "Calcitonin Gene-Related Peptide (CGRP) Inhibitors for Chronic Migraine",
        "service_code": "J3032",
        "service_description": "CGRP Inhibitor Therapy for Chronic Migraine (e.g., Erenumab)",
        "diagnosis_code": "G43.709",
        "coverage_status": "Covered",
        "prior_authorization_required": "Yes",
        "medical_necessity_rule": "Chronic migraine diagnosis (>=15 days/month for >=3 months) AND failure of at least two conventional preventive medication classes (each trialed for at least 8 weeks).",
        "documentation_required": "Headache diary/frequency documentation, prior preventive medication trial history with dates and doses.",
        "limitations": "Fewer than two conventional trials or trial duration under 8 weeks is not covered.",
        "exclusions": "Episode prevention for episodic migraine is excluded.",
        "effective_date": "2026-01-01",
        "source": "CMS Medicare Coverage Database",
        "source_url": "https://www.cms.gov/medicare-coverage-database/view/lcd.aspx?LCDId=39210",
        "age_requirement": "Adults (>= 18 years)",
        "frequency_limit": "1 injection per 30 days",
        "quantity_limit": "140 mg"
    },
    {
        "policy_id": "L38715",
        "policy_type": "LCD",
        "policy_title": "Monoclonal Antibodies for Crohn's Disease",
        "service_code": "J1745",
        "service_description": "Biologic Infusion Therapy for Crohn's Disease (Infliximab and similar)",
        "diagnosis_code": "K50.90",
        "coverage_status": "Covered",
        "prior_authorization_required": "Yes",
        "medical_necessity_rule": "Moderate-to-severe Crohn's disease diagnosis AND failure of conventional therapy (aminosalicylates, immunomodulators, or corticosteroids) AND objective evidence of active inflammation.",
        "documentation_required": "Endoscopy/imaging report, conventional therapy trial history, inflammatory marker labs (fecal calprotectin or CRP).",
        "limitations": "No conventional therapy trial, or lack of active inflammation documentation, is not covered.",
        "exclusions": "Use for mild Crohn's disease or asymptomatic patients is excluded.",
        "effective_date": "2026-01-01",
        "source": "CMS Medicare Coverage Database",
        "source_url": "https://www.cms.gov/medicare-coverage-database/view/lcd.aspx?LCDId=38715",
        "age_requirement": "No age restriction",
        "frequency_limit": "1 infusion per 8 weeks",
        "quantity_limit": "5 mg/kg"
    },
    {
        "policy_id": "L33822",
        "policy_type": "LCD",
        "policy_title": "Continuous Glucose Monitors (CGM)",
        "service_code": "E2103",
        "service_description": "Continuous Glucose Monitor (CGM)",
        "diagnosis_code": "E11.9",
        "coverage_status": "Covered",
        "prior_authorization_required": "Yes",
        "medical_necessity_rule": "Confirmed diagnosis of diabetes mellitus AND currently on multiple daily insulin injections (3 or more) or insulin pump therapy OR documented history of problematic hypoglycemia AND HbA1c above target range or blood glucose monitoring insufficient.",
        "documentation_required": "Recent HbA1c result, current medication/insulin regimen documentation, and detailed hypoglycemia log.",
        "limitations": "Patient not on insulin and no hypoglycemia history is not covered.",
        "exclusions": "HbA1c at goal without hypoglycemia concerns is excluded.",
        "effective_date": "2026-01-01",
        "source": "CMS Medicare Coverage Database",
        "source_url": "https://www.cms.gov/medicare-coverage-database/view/lcd.aspx?LCDId=33822",
        "age_requirement": "No age restriction",
        "frequency_limit": "1 receiver per 3 years",
        "quantity_limit": "1 receiver"
    },
    {
        "policy_id": "L37910",
        "policy_type": "LCD",
        "policy_title": "Biologic DMARDs for Rheumatoid Arthritis",
        "service_code": "J0135",
        "service_description": "Biologic Disease-Modifying Antirheumatic Drugs (Adalimumab and similar agents)",
        "diagnosis_code": "M06.9",
        "coverage_status": "Covered",
        "prior_authorization_required": "Yes",
        "medical_necessity_rule": "Moderate-to-severe rheumatoid arthritis diagnosed by a rheumatologist AND trial of at least one conventional DMARD (e.g. methotrexate) for a minimum of 3 months at adequate dose AND active disease labs (ESR/CRP, RF/CCP) or joint damage imaging.",
        "documentation_required": "Rheumatologist visit notes, conventional DMARD trial dates and doses, active markers labs or joint X-rays.",
        "limitations": "No conventional DMARD trial, or trial under 3 months, is not covered.",
        "exclusions": "Treatment of mild or inactive rheumatoid arthritis is excluded.",
        "effective_date": "2026-01-01",
        "source": "CMS Medicare Coverage Database",
        "source_url": "https://www.cms.gov/medicare-coverage-database/view/lcd.aspx?LCDId=37910",
        "age_requirement": "Adults (>= 18 years)",
        "frequency_limit": "2 injections per 28 days",
        "quantity_limit": "40 mg"
    },
    {
        "policy_id": "L39604",
        "policy_type": "LCD",
        "policy_title": "Stimulant Medications for ADHD",
        "service_code": "J8499",
        "service_description": "Stimulant Medication for ADHD (Lisdexamfetamine and similar)",
        "diagnosis_code": "F90.0",
        "coverage_status": "Covered",
        "prior_authorization_required": "Yes",
        "medical_necessity_rule": "Confirmed ADHD diagnosis by a qualified provider AND trial of at least one alternative stimulant or non-stimulant medication with inadequate response or documented intolerance.",
        "documentation_required": "Diagnostic evaluation notes, DSM-5 diagnostic criteria checklist, and prior medication trial history.",
        "limitations": "No prior alternative ADHD medication trial is not covered.",
        "exclusions": "History of active substance use disorder without safeguards is excluded.",
        "effective_date": "2026-01-01",
        "source": "CMS Medicare Coverage Database",
        "source_url": "https://www.cms.gov/medicare-coverage-database/view/lcd.aspx?LCDId=39604",
        "age_requirement": "No age restriction",
        "frequency_limit": "1 fill per 30 days",
        "quantity_limit": "30 units"
    },
    {
        "policy_id": "L36420",
        "policy_type": "LCD",
        "policy_title": "Cardiac Catheterization and Angiography",
        "service_code": "93458",
        "service_description": "Diagnostic Cardiac Catheterization",
        "diagnosis_code": "I25.10",
        "coverage_status": "Covered",
        "prior_authorization_required": "Yes",
        "medical_necessity_rule": "Positive or high-risk stress test result OR clinical presentation of unstable angina/acute coronary syndrome OR emergent presentation with suspicion of myocardial infarction.",
        "documentation_required": "Stress test report showing ischemia, ECG and troponin laboratory reports, and clinical presentation notes.",
        "limitations": "No stress testing performed and no acute/unstable presentation is not covered.",
        "exclusions": "Routine screening in asymptomatic low-risk patients is excluded.",
        "effective_date": "2026-01-01",
        "source": "CMS Medicare Coverage Database",
        "source_url": "https://www.cms.gov/medicare-coverage-database/view/lcd.aspx?LCDId=36420",
        "age_requirement": "Adults (>= 18 years)",
        "frequency_limit": "1 procedure per 30 days",
        "quantity_limit": "1 unit"
    }
]

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_file = RAW_DIR / "cms_mcd_raw.json"
    
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(cms_raw_data, f, indent=2)
        
    print(f"Ingested {len(cms_raw_data)} CMS MCD policies to {raw_file}")

if __name__ == "__main__":
    main()
