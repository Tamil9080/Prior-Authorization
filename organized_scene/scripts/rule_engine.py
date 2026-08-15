import json
import pandas as pd

def derive_criteria_and_reason(row, decision):
    """
    Deterministic rule engine that maps row features and the predicted decision 
    to meets_criteria and decision_reason.
    """
    # Detect if we have structured_features indicating the rule-based dataset
    pol_num = row.get('policy_number', None)
    s_feat = row.get('structured_features', None)
    
    is_rule_based = pd.notna(pol_num) or pd.notna(s_feat)
    
    if is_rule_based:
        # Load structured features dict
        feats = {}
        if pd.notna(s_feat):
            if isinstance(s_feat, str):
                try:
                    feats = json.loads(s_feat)
                except Exception:
                    pass
            elif isinstance(s_feat, dict):
                feats = s_feat
        
        policy = str(pol_num).strip() if pd.notna(pol_num) else ""
        if not policy:
            # Fallback policy mapping by requested service
            svc = str(row.get('requested_service', '')).lower()
            if 'mri' in svc or 'lumbar' in svc:
                policy = 'MHI-ORTHO-014'
            elif 'adalimumab' in svc or 'humira' in svc:
                policy = 'MHI-RHEUM-007'
            elif 'glucose' in svc or 'cgm' in svc:
                policy = 'MHI-ENDO-021'
            elif 'erenumab' in svc or 'aimovig' in svc:
                policy = 'MHI-NEURO-009'
            elif 'cpap' in svc or 'sleep apnea' in svc:
                policy = 'MHI-PULM-011'
            elif 'tms' in svc or 'transcranial' in svc:
                policy = 'MHI-PSYCH-018'
            elif 'catheterization' in svc or 'cardiac' in svc:
                policy = 'MHI-CARD-003'
            elif 'arthroplasty' in svc or 'knee replacement' in svc:
                policy = 'MHI-ORTHO-022'
            elif 'lisdexamfetamine' in svc or 'vyvanse' in svc:
                policy = 'MHI-PSYCH-005'
            elif 'infliximab' in svc or 'remicade' in svc:
                policy = 'MHI-GI-016'
        
        # Apply rules per policy
        if policy == 'MHI-NEURO-009':
            if decision == 'Approved':
                return 'Yes', 'Approved: At least two preventive medication classes failed at adequate duration.'
            elif decision == 'Denied':
                return 'No', 'Denied: Fewer than two preventive medication classes trialed for >=8 weeks each.'
            else:
                return 'Insufficient Information', 'Cannot render decision: required clinical documentation incomplete. Resubmit with complete documentation.'
                
        elif policy == 'MHI-ORTHO-014':
            if decision == 'Approved':
                return 'Yes', 'Approved: Symptom duration and conservative treatment / red-flag criteria met.'
            elif decision == 'Denied':
                if feats.get('symptom_duration_weeks', 0) < 6 and not feats.get('red_flag_findings', False):
                    return 'No', 'Denied: Symptom duration under 6 weeks with no red-flag findings.'
                else:
                    return 'No', 'Denied: No conservative treatment attempted or documented.'
            else:
                return 'Insufficient Information', 'Cannot render decision: required clinical documentation incomplete. Resubmit with complete documentation.'
                
        elif policy == 'MHI-ORTHO-022':
            if decision == 'Approved':
                return 'Yes', 'Approved: Imaging grade, conservative treatment duration, and functional limitation all documented.'
            elif decision == 'Denied':
                if feats.get('kl_grade', 0) < 3:
                    return 'No', 'Denied: Kellgren-Lawrence grade below 3 without significant functional limitation.'
                else:
                    return 'Partial', 'Denied: criteria only partially met. Imaging supports need, but conservative management under 12 weeks.'
            else:
                if not feats.get('documentation_complete', True):
                    return 'Insufficient Information', 'Cannot render decision: required clinical documentation incomplete. Resubmit with complete documentation.'
                else:
                    return 'Insufficient Information', 'Cannot render decision: functional assessment (e.g., oxford knee score) not documented. Resubmit with complete documentation.'
                    
        elif policy == 'MHI-PSYCH-005':
            if decision == 'Approved':
                return 'Yes', 'Approved: Alternative medication trial with inadequate response documented.'
            elif decision == 'Denied':
                return 'No', 'Denied: No alternative stimulant/non-stimulant medication trial documented.'
            else:
                return 'Insufficient Information', 'Cannot render decision: required clinical documentation incomplete. Resubmit with complete documentation.'
                
        elif policy == 'MHI-PSYCH-018':
            if decision == 'Approved':
                return 'Yes', 'Approved: Medication trial history and psychotherapy engagement both documented.'
            elif decision == 'Denied':
                if feats.get('num_antidepressant_trials', 0) >= 2 and feats.get('antidepressant_trials_min_6wk', False) and not feats.get('psychotherapy_engaged', True):
                    return 'Partial', 'Denied: criteria only partially met. Medication trial criteria met, but no psychotherapy engagement documented.'
                else:
                    return 'No', 'Denied: Fewer than two antidepressant trials at adequate duration.'
            else:
                return 'Insufficient Information', 'Cannot render decision: required clinical documentation incomplete. Resubmit with complete documentation.'
                
        elif policy == 'MHI-PULM-011':
            if decision == 'Approved':
                return 'Yes', 'Approved: AHI meets threshold for CPAP therapy.'
            elif decision == 'Denied':
                return 'No', 'Denied: AHI below qualifying threshold for the documented symptom profile.'
            else:
                if not feats.get('documentation_complete', True):
                    return 'Insufficient Information', 'Cannot render decision: required clinical documentation incomplete. Resubmit with complete documentation.'
                else:
                    return 'Insufficient Information', 'Cannot render decision: sleep study report not on file. Resubmit with complete documentation.'
                    
        elif policy == 'MHI-RHEUM-007':
            if decision == 'Approved':
                return 'Yes', 'Approved: DMARD trial duration and supportive labs/imaging both met.'
            elif decision == 'Denied':
                if feats.get('dmard_trial_months', 0) >= 3 and not feats.get('labs_supportive', True) and not feats.get('imaging_joint_damage', True):
                    return 'Partial', 'Denied: criteria only partially met. DMARD trial met, but no supportive labs or imaging on file.'
                else:
                    return 'No', 'Denied: Conventional DMARD trial under 3 months (or not documented).'
            else:
                return 'Insufficient Information', 'Cannot render decision: required clinical documentation incomplete. Resubmit with complete documentation.'
                
        elif policy == 'MHI-CARD-003':
            if decision == 'Approved':
                return 'Yes', 'Approved: Positive stress test, unstable presentation, or emergent MI suspicion present.'
            elif decision == 'Denied':
                return 'No', 'Denied: No qualifying stress test result or acute/unstable presentation documented.'
            else:
                return 'Insufficient Information', 'Cannot render decision: required clinical documentation incomplete. Resubmit with complete documentation.'
                
        elif policy == 'MHI-ENDO-021':
            if decision == 'Approved':
                return 'Yes', 'Approved: Insulin/pump therapy or hypoglycemia history, plus glycemic control issue.'
            elif decision == 'Denied':
                if (feats.get('on_insulin_or_pump', False) or feats.get('hypoglycemia_history', False)) and not feats.get('hba1c_above_target', True) and not feats.get('insufficient_smbg', True):
                    return 'Partial', 'Denied: criteria only partially met. On qualifying therapy, but HbA1c at goal without other concerns.'
                else:
                    return 'No', 'Denied: Not on insulin/pump therapy and no hypoglycemia history documented.'
            else:
                return 'Insufficient Information', 'Cannot render decision: required clinical documentation incomplete. Resubmit with complete documentation.'
                
        elif policy == 'MHI-GI-016':
            if decision == 'Approved':
                return 'Yes', 'Approved: Conventional therapy failure and objective inflammation evidence both documented.'
            elif decision == 'Denied':
                if feats.get('conventional_therapy_tried', False) and not feats.get('objective_inflammation_evidence', True):
                    return 'Partial', 'Denied: criteria only partially met. Conventional therapy trial documented, but no objective inflammation evidence.'
                else:
                    return 'No', 'Denied: No conventional therapy trial documented.'
            else:
                return 'Insufficient Information', 'Cannot render decision: required clinical documentation incomplete. Resubmit with complete documentation.'
        
        # Fallback if policy not found but rule-based
        if decision == 'Approved':
            return 'Yes', 'Approved: policy criteria met.'
        elif decision == 'Denied':
            return 'No', 'Denied: policy criteria not met.'
        else:
            return 'Insufficient Information', 'Cannot render decision: required clinical documentation incomplete.'

    else:
        # Clinical PA datasets
        urg = str(row.get('urgency', 'Routine')).strip()
        prev = str(row.get('previous_treatments', '')).strip()
        has_multiple = ';' in prev
        
        if decision == 'Approved':
            if not has_multiple or prev.lower() == 'none documented':
                mc = 'Partial'
                if urg == 'Emergent':
                    reas = 'Approved on the basis of clinical urgency despite partial criteria documentation.'
                else:
                    reas = 'Approved with conditions: partial criteria met, service authorized with monitoring requirement.'
            else:
                mc = 'Yes'
                reas = 'Clinical criteria met: documented conservative treatment history and supporting findings justify the requested service.'
        elif decision == 'Denied':
            if not has_multiple or prev.lower() == 'none documented':
                mc = 'No'
                reas = 'Denied: clinical criteria not met. Conservative/first-line treatment not attempted or documented.'
            else:
                mc = 'Partial'
                reas = 'Denied: criteria only partially met; step therapy or additional documentation required before approval.'
        elif decision == 'Pending Additional Information':
            mc = 'Insufficient Information'
            missing = str(row.get('missing_information', ''))
            if pd.isna(row.get('missing_information', None)) or not missing.strip():
                # Fallback to check other fields if missing_information is empty
                missing = "Clinical notes incomplete"
                
            if 'lab results' in missing.lower():
                reas = 'Cannot render decision: recent lab results not provided. Resubmit with complete documentation.'
            elif 'consultation' in missing.lower() or 'specialist' in missing.lower():
                reas = 'Cannot render decision: specialist consultation notes missing. Resubmit with complete documentation.'
            elif 'failed conservative' in missing.lower() or 'failed' in missing.lower():
                reas = 'Cannot render decision: documentation of failed conservative therapy missing. Resubmit with complete documentation.'
            elif 'imaging' in missing.lower() or 'x-ray' in missing.lower():
                reas = 'Cannot render decision: imaging report not attached. Resubmit with complete documentation.'
            else:
                reas = 'Cannot render decision: clinical notes incomplete. Resubmit with complete documentation.'
        elif decision == 'In Review':
            mc = 'Under Review'
            if urg in ['Urgent', 'Emergent']:
                reas = 'Urgent request queued for expedited review within 24-72 hours.'
            else:
                reas = 'Request received with complete documentation; assigned to clinical reviewer, decision pending.'
        else:
            mc = 'Under Review'
            reas = 'Submitted within standard review window; no action taken yet.'
            
        return mc, reas
