from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cms_mcd"

guides = {
    "unitedhealthcare_policy_manual.txt": (
        "UnitedHealthcare Commercial Coverage Guideline: Total Knee Arthroplasty (TKA).\n\n"
        "Coverage Criteria:\n"
        "1. Radiographic evidence of moderate-to-severe osteoarthritis (Kellgren-Lawrence Grade 3 or 4).\n"
        "2. Failure of at least 12 weeks of conservative therapy. Crucially, the conservative management trial must consist of formal, supervised physical therapy (supervised PT) directed by a licensed therapist. Self-directed home exercise programs or general activity modification are not considered adequate trials under UHC terms.\n"
        "3. Significant functional joint limitation impacting activities of daily living (e.g. transfers, walking, stairs).\n"
        "4. Exclusion: Joint replacement for early-stage OA (KL Grade 1 or 2) is considered not medically necessary.\n\n"
        "UnitedHealthcare Sleep Therapy Guideline: Continuous Positive Airway Pressure (CPAP).\n\n"
        "UHC covers CPAP for obstructive sleep apnea when an approved sleep study (polysomnogram or home sleep test) confirms an Apnea-Hypopnea Index (AHI) of 15 or more events per hour, or AHI between 5 and 14 with documented sleep-related symptoms (daytime sleepiness, mood disorders, hypertension, or ischemic heart disease)."
    ),
    "aetna_policy_manual.txt": (
        "Aetna Clinical Policy Bulletin (CPB) 0702: Transcranial Magnetic Stimulation (TMS).\n\n"
        "Aetna considers repetitive transcranial magnetic stimulation (rTMS) medically necessary for major depressive disorder (MDD) when all the following criteria are met:\n"
        "1. Confirmed diagnosis of severe major depressive disorder (recurrent or single episode).\n"
        "2. Evaluation and request submitted by a board-certified psychiatrist. A detailed specialist consultation note from psychiatry must be documented within 30 days prior to the authorization request.\n"
        "3. Failure of, or intolerance to, at least two antidepressant medications from different pharmacological classes trialed for at least 6 weeks at therapeutic doses.\n"
        "4. Active or prior engagement in formal psychotherapy.\n\n"
        "Aetna Policy Bulletin: MRI Lumbar Spine.\n\n"
        "Aetna covers non-contrast lumbar spine MRI (CPT 72148) only after at least 6 consecutive weeks of conservative therapy (NSAIDs and physical therapy) have failed, unless red-flag symptoms are documented (e.g. cauda equina syndrome, severe focal neuro deficit, or high-velocity trauma)."
    ),
    "cigna_policy_manual.txt": (
        "Cigna Medical Coverage Policy: Lumbar Spine Imaging.\n\n"
        "Cigna considers magnetic resonance imaging (MRI) of the lumbar spine (CPT 72148) medically necessary when:\n"
        "1. Low back pain or radicular pain has persisted for a minimum of 6 consecutive weeks.\n"
        "2. A pain intensity rating score using a validated scale (e.g., Visual Analog Scale - VAS) of 6 or higher is documented in the clinical history notes.\n"
        "3. Failure of at least one conservative therapy trial (physical therapy or NSAIDs).\n"
        "4. Exclusion: Routine imaging for uncomplicated low back pain (VAS < 6) is not covered."
    ),
    "humana_policy_manual.txt": (
        "Humana National Coverage Policy: Positive Airway Pressure (PAP) Devices.\n\n"
        "Humana covers CPAP (HCPCS E0601) for sleep apnea when:\n"
        "1. The ordering practitioner has completed the Humana Pre-Service Prior Authorization Checklist (PA Checklist) confirming symptom evaluation.\n"
        "2. Sleep study (facility-based or home-based) documents an AHI >= 15 events/hour, or AHI of 5-14 with documented co-existing conditions (excessive daytime sleepiness, cognitive deficits, or cardiovascular disease).\n"
        "3. Exclusion: Devices ordered without a completed Humana PA Checklist will be pended for additional information."
    ),
    "bcbs_policy_manual.txt": (
        "Blue Cross Blue Shield Association Medical Policy: Knee Reconstruction.\n\n"
        "BCBS covers Total Knee Arthroplasty (CPT 27447) when the following criteria are documented:\n"
        "1. Moderate-to-severe joint narrowing or osteoarthritis (KL Grade 3 or 4).\n"
        "2. Documented functional impairment score using a validated clinical assessment scale. Validated scales include the Oxford Knee Score (OKS), the Western Ontario and McMaster Universities Osteoarthritis Index (WOMAC), or the Knee Injury and Osteoarthritis Outcome Score (KOOS).\n"
        "3. Inadequate response to at least 12 weeks of non-surgical conservative treatment (injections, physical therapy, or weight loss program).\n"
        "4. Exclusion: Joint replacement without a documented functional impairment score (WOMAC, OKS, or KOOS) will be pended."
    )
}

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename, content in guides.items():
        filepath = RAW_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created commercial guide: {filepath}")

if __name__ == "__main__":
    main()
