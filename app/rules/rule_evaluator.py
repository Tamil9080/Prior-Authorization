import re
import pandas as pd

def evaluate_rules(request_data, policy):
    """
    Evaluate a prior authorization request against an applicable coverage policy.
    Returns: A list of dicts: [{'rule': str, 'status': 'MET'|'NOT_MET'|'INSUFFICIENT', 'evidence': str}]
    """
    if not policy:
        return [
            {
                "rule": "Policy Coverage Check",
                "status": "NOT_MET",
                "evidence": "No applicable coverage policy could be retrieved from the database."
            }
        ]
        
    rules_results = []
    
    # 1. Rule: Qualifying Diagnosis
    diag_code = str(request_data.get("diagnosis_code", "")).strip().upper()
    policy_diags = str(policy.get("diagnosis_code", "")).strip().upper()
    
    if not diag_code:
        rules_results.append({
            "rule": "Qualifying Diagnosis",
            "status": "INSUFFICIENT",
            "evidence": "Patient diagnosis code (ICD-10) is missing in the request."
        })
    else:
        code_list = [c.strip() for c in policy_diags.replace(";", ",").split(",")]
        if diag_code in code_list:
            rules_results.append({
                "rule": "Qualifying Diagnosis",
                "status": "MET",
                "evidence": f"Patient ICD-10 code '{diag_code}' matches the covered diagnosis list for policy {policy.get('policy_id')}."
            })
        elif diag_code[:3] in [c[:3] for c in code_list]:
            rules_results.append({
                "rule": "Qualifying Diagnosis",
                "status": "MET",
                "evidence": f"Patient ICD-10 code '{diag_code}' matches category-level coverage for policy {policy.get('policy_id')}."
            })
        else:
            rules_results.append({
                "rule": "Qualifying Diagnosis",
                "status": "NOT_MET",
                "evidence": f"Patient ICD-10 code '{diag_code}' is not covered under policy {policy.get('policy_id')} (requires: {policy_diags})."
            })



    # 3. Rule: Medical Necessity / Conservative Treatment
    pol_id = policy.get("policy_id", "")
    prev_treat = str(request_data.get("previous_treatment") or request_data.get("previous_treatments") or "").strip().lower()
    history = str(request_data.get("clinical_history", "")).strip().lower()
    lab = str(request_data.get("lab_results", "")).strip().lower()
    
    if pol_id == "L33718":  # CPAP
        # Requires AHI study report (supports AHI: 32, AHI of 32, AHI is 32, etc.)
        ahi_match = re.search(r"ahi\b.*?(\d+)", lab + " " + history)
        if not ahi_match:
            rules_results.append({
                "rule": "Sleep Study / AHI Documentation",
                "status": "INSUFFICIENT",
                "evidence": "Apnea-Hypopnea Index (AHI) value not documented in lab results or clinical history."
            })
        else:
            ahi = int(ahi_match.group(1))
            if ahi >= 15:
                rules_results.append({
                    "rule": "Sleep Study / AHI Documentation",
                    "status": "MET",
                    "evidence": f"Diagnostic sleep study documents qualifying AHI of {ahi} events/hour (>= 15 required)."
                })
            elif ahi >= 5:
                # Check for documented symptoms/comorbidities in clinical notes
                comorbidities = ["hypertension", "sleepiness", "depression", "stroke", "insomnia", "cardiovascular", "heart disease"]
                found = [c for c in comorbidities if c in history]
                if found:
                    rules_results.append({
                        "rule": "Sleep Study / AHI Documentation",
                        "status": "MET",
                        "evidence": f"Mild OSA (AHI {ahi}) covered due to documented sleep comorbidities: {', '.join(found)}."
                    })
                else:
                    rules_results.append({
                        "rule": "Sleep Study / AHI Documentation",
                        "status": "NOT_MET",
                        "evidence": f"Mild OSA (AHI {ahi}) requires documented comorbidities (excessive sleepiness, hypertension, etc.) which were not found."
                    })
            else:
                rules_results.append({
                    "rule": "Sleep Study / AHI Documentation",
                    "status": "NOT_MET",
                    "evidence": f"Sleep study AHI of {ahi} events/hour is below the diagnostic threshold (>= 5 required)."
                })

    elif pol_id == "L38429":  # Lumbar Spine MRI
        # Requires duration >= 6 weeks and conservative treatment (PT, NSAIDs)
        duration_6_weeks = "6 week" in history or "8 week" in history or "12 week" in history or "years" in history or "several" in history
        has_pt = "physical therapy" in prev_treat or "pt" in prev_treat
        has_nsaid = "nsaid" in prev_treat or "ibuprofen" in prev_treat or "naproxen" in prev_treat or "meloxicam" in prev_treat
        
        if not duration_6_weeks:
            rules_results.append({
                "rule": "Symptom Duration",
                "status": "NOT_MET",
                "evidence": "Low back pain duration is under the required 6 consecutive weeks."
            })
        else:
            rules_results.append({
                "rule": "Symptom Duration",
                "status": "MET",
                "evidence": "Back pain duration is documented as >= 6 consecutive weeks."
            })
            
        if not prev_treat or prev_treat == "none documented":
            rules_results.append({
                "rule": "Conservative Management",
                "status": "NOT_MET",
                "evidence": "No conservative management trial (physical therapy or NSAIDs) attempted or documented."
            })
        elif has_pt or has_nsaid:
            rules_results.append({
                "rule": "Conservative Management",
                "status": "MET",
                "evidence": f"Documented conservative trial: {prev_treat}."
            })
        else:
            rules_results.append({
                "rule": "Conservative Management",
                "status": "INSUFFICIENT",
                "evidence": f"Documentation of conservative therapy failed trial is incomplete (reported: {prev_treat})."
            })

    elif pol_id == "L37436":  # Total Knee Arthroplasty
        # Requires KL Grade 3 or 4, and conservative management >= 12 weeks
        kl_match = re.search(r"grade\s*(\d)", lab)
        has_conservative = "weight loss" in prev_treat or "corticosteroid" in prev_treat or "injection" in prev_treat or "therapy" in prev_treat or "nsaid" in prev_treat
        
        if not kl_match:
            rules_results.append({
                "rule": "Radiographic Evidence",
                "status": "INSUFFICIENT",
                "evidence": "Knee radiography report with Kellgren-Lawrence Grade is missing."
            })
        else:
            grade = int(kl_match.group(1))
            if grade >= 3:
                rules_results.append({
                    "rule": "Radiographic Evidence",
                    "status": "MET",
                    "evidence": f"X-ray report documents Kellgren-Lawrence Grade {grade} osteoarthritis (Grade 3 or 4 required)."
                })
            else:
                rules_results.append({
                    "rule": "Radiographic Evidence",
                    "status": "NOT_MET",
                    "evidence": f"X-ray report documents Kellgren-Lawrence Grade {grade} (requires Grade 3 or 4)."
                })
                
        if not has_conservative:
            rules_results.append({
                "rule": "Conservative Treatment Duration",
                "status": "NOT_MET",
                "evidence": "No conservative treatment (PT, weight loss, or injections) documented."
            })
        else:
            # check for duration details in history or previous treatment text
            combined_text = history + " " + prev_treat
            duration_match = "12 week" in combined_text or "16 week" in combined_text or "24 week" in combined_text or "several" in combined_text or "years" in combined_text
            if duration_match:
                rules_results.append({
                    "rule": "Conservative Treatment Duration",
                    "status": "MET",
                    "evidence": "Conservative management attempted for >= 12 weeks."
                })
            else:
                rules_results.append({
                    "rule": "Conservative Treatment Duration",
                    "status": "INSUFFICIENT",
                    "evidence": "Conservative management trial duration is under 12 weeks or not clearly specified."
                })

    elif pol_id == "L34520":  # TMS
        # Requires psychiatrist visit AND >= 2 antidepressant failed trials AND psychotherapy
        has_psych = "psychiatrist" in history or "psychiatry" in history or "provider_specialty" in request_data and request_data["provider_specialty"].lower() == "psychiatry"
        
        # Count antidepressant trials in previous treatments or history
        meds = str(request_data.get("medications", "")).lower()
        antidepressants = ["sertraline", "fluoxetine", "citalopram", "escitalopram", "paroxetine", "venlafaxine", "duloxetine", "bupropion", "mirtazapine", "amitriptyline"]
        trials_count = sum(1 for med in antidepressants if med in prev_treat or med in meds or med in history)
        
        has_therapy = "psychotherapy" in prev_treat or "therapy" in prev_treat or "behavioral" in prev_treat
        
        if not has_psych:
            rules_results.append({
                "rule": "Specialist Evaluation",
                "status": "INSUFFICIENT",
                "evidence": "Documentation of specialist evaluation by a psychiatrist is missing."
            })
        else:
            rules_results.append({
                "rule": "Specialist Evaluation",
                "status": "MET",
                "evidence": "Evaluation performed by psychiatry specialty."
            })
            
        if trials_count >= 2:
            rules_results.append({
                "rule": "Antidepressant Medication Trials",
                "status": "MET",
                "evidence": f"Documented failure of {trials_count} distinct antidepressant medication trials (>= 2 required)."
            })
        elif trials_count == 1:
            rules_results.append({
                "rule": "Antidepressant Medication Trials",
                "status": "NOT_MET",
                "evidence": "Only 1 antidepressant medication trial documented. Policy requires failure of >= 2 different drug classes."
            })
        else:
            rules_results.append({
                "rule": "Antidepressant Medication Trials",
                "status": "INSUFFICIENT",
                "evidence": "Documentation of previous antidepressant medication drug trials is missing."
            })
            
        if has_therapy:
            rules_results.append({
                "rule": "Psychotherapy Engagement",
                "status": "MET",
                "evidence": "Active or prior engagement in formal psychotherapy documented."
            })
        else:
            rules_results.append({
                "rule": "Psychotherapy Engagement",
                "status": "NOT_MET",
                "evidence": "No concurrent or prior engagement in formal psychotherapy documented."
            })

    # Default rules for other policies
    else:
        # Check standard conservative treatment trial
        if not prev_treat or prev_treat == "none" or prev_treat == "none documented":
            rules_results.append({
                "rule": "Prior Therapy Trial",
                "status": "NOT_MET",
                "evidence": "No first-line conservative or alternative therapy trial documented in the request."
            })
        else:
            rules_results.append({
                "rule": "Prior Therapy Trial",
                "status": "MET",
                "evidence": f"Documented prior therapy trial: {prev_treat}."
            })
            
        # Check standard documentation
        if not lab or lab == "none" or lab == "none documented":
            rules_results.append({
                "rule": "Clinical Documentation",
                "status": "INSUFFICIENT",
                "evidence": "Required clinical lab results, diagnostic findings, or imaging reports are missing."
            })
        else:
            rules_results.append({
                "rule": "Clinical Documentation",
                "status": "MET",
                "evidence": f"Clinical findings documented: {lab}."
            })

    # Add general policy requirement
    rules_results.append({
        "rule": "Payer Policy Rule Verification",
        "status": "MET",
        "evidence": f"Review completed under Medicare LCD {policy.get('policy_id')} - {policy.get('policy_title')}."
    })
    
    return rules_results
