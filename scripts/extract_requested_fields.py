import os
import pandas as pd
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE = str(PROJECT_ROOT / "synthea_csv" / "csv")
OUT = str(PROJECT_ROOT / "extracted_requested_fields.csv")
REF_DATE = pd.to_datetime('2021-11-01')

def read_csv(name):
    path = os.path.join(BASE, name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, dtype=str, low_memory=False)

patients = read_csv('patients.csv')
conditions = read_csv('conditions.csv')
procedures = read_csv('procedures.csv')
meds = read_csv('medications.csv')
encounters = read_csv('encounters.csv')
providers = read_csv('providers.csv')
observations = read_csv('observations.csv')

# Prepare patients
patients = patients.rename(columns={c: c.strip() for c in patients.columns})
patients['Id'] = patients['Id'].astype(str)
if 'BIRTHDATE' in patients.columns:
    patients['BIRTHDATE'] = pd.to_datetime(patients['BIRTHDATE'], errors='coerce')
    age_series = ((REF_DATE - patients['BIRTHDATE']).dt.days / 365.25)
    patients['age'] = pd.to_numeric(age_series, errors='coerce')
else:
    patients['age'] = pd.NA

# Helper: agg unique joined
def agg_unique(df, group_col, value_col):
    if df is None or value_col not in df.columns:
        return pd.Series()
    agg = df.groupby(group_col)[value_col].apply(lambda s: '; '.join(pd.unique(s.dropna()))).rename(value_col)
    return agg

result = pd.DataFrame()
result['patient_id'] = patients['Id']
result = result.set_index('patient_id')
result['age'] = patients.set_index('Id')['age']

# Diagnoses
if conditions is not None:
    conditions.columns = [c.strip() for c in conditions.columns]
    diag_desc = agg_unique(conditions, 'PATIENT', 'DESCRIPTION')
    diag_code = agg_unique(conditions, 'PATIENT', 'CODE')
    result['diagnosis'] = diag_desc
    result['diagnosis_code'] = diag_code
else:
    result['diagnosis'] = pd.NA
    result['diagnosis_code'] = pd.NA

# Procedures -> requested_service, procedure_code, previous_treatments, clinical_history
if procedures is not None:
    procedures.columns = [c.strip() for c in procedures.columns]
    proc_desc = agg_unique(procedures, 'PATIENT', 'DESCRIPTION')
    proc_code = agg_unique(procedures, 'PATIENT', 'CODE')
    result['requested_service'] = proc_desc
    result['procedure_code'] = proc_code
    result['previous_treatments'] = proc_desc
else:
    result['requested_service'] = pd.NA
    result['procedure_code'] = pd.NA
    result['previous_treatments'] = pd.NA

# clinical_history: use diagnoses + procedures
result['clinical_history'] = result[['diagnosis', 'previous_treatments']].apply(
    lambda row: '; '.join([x for x in [str(row['diagnosis']), str(row['previous_treatments'])] if x and x != 'nan']), axis=1
)

# medications
if meds is not None:
    meds.columns = [c.strip() for c in meds.columns]
    med_desc = agg_unique(meds, 'PATIENT', 'DESCRIPTION')
    result['medications'] = med_desc
else:
    result['medications'] = pd.NA

# lab_results: from observations (take DESCRIPTION:VALUE for first 20 per patient)
if observations is not None:
    observations.columns = [c.strip() for c in observations.columns]
    if 'PATIENT' in observations.columns:
        def make_lab_series(grp):
            pieces = []
            for _, r in grp.head(20).iterrows():
                desc = r.get('DESCRIPTION', '')
                val = r.get('VALUE', '')
                if pd.isna(desc):
                    continue
                if pd.isna(val) or val == 'nan':
                    pieces.append(desc)
                else:
                    pieces.append(f"{desc}: {val}")
            return '; '.join(pieces)
        lab = observations.groupby('PATIENT').apply(make_lab_series).rename('lab_results')
        result['lab_results'] = lab
    else:
        result['lab_results'] = pd.NA
else:
    result['lab_results'] = pd.NA

# provider_specialty: use encounters -> provider -> providers.SPECIALITY (most frequent)
if encounters is not None and providers is not None:
    encounters.columns = [c.strip() for c in encounters.columns]
    providers.columns = [c.strip() for c in providers.columns]
    prov_map = providers.set_index('Id')['SPECIALITY'].to_dict()
    def map_specialty(grp):
        provs = grp['PROVIDER'].dropna().map(prov_map).dropna()
        if len(provs)==0:
            return pd.NA
        return provs.mode().iat[0]
    spec = encounters.groupby('PATIENT').apply(map_specialty).rename('provider_specialty')
    result['provider_specialty'] = spec
else:
    result['provider_specialty'] = pd.NA

# urgency: aggregate encounter classes
if encounters is not None:
    if 'ENCOUNTERCLASS' in encounters.columns:
        urg = agg_unique(encounters, 'PATIENT', 'ENCOUNTERCLASS')
        result['urgency'] = urg
    else:
        result['urgency'] = pd.NA
else:
    result['urgency'] = pd.NA

# policy_id, meets_criteria, missing_information, decision, decision_reason, request_id -> not in dataset
for c in ['policy_id','meets_criteria','missing_information','decision','decision_reason','request_id']:
    result[c] = pd.NA

# Ensure requested column order
cols = ['request_id','patient_id','age','diagnosis','diagnosis_code','requested_service','procedure_code','clinical_history','previous_treatments','medications','lab_results','provider_specialty','urgency','policy_id','meets_criteria','missing_information','decision','decision_reason']
# patient_id currently index -> move to column
result = result.reset_index().rename(columns={'index':'patient_id'})
# Some columns may be missing in result; add them
for c in cols:
    if c not in result.columns:
        result[c] = pd.NA

result = result[cols]

result.to_csv(OUT, index=False)
print(f"Wrote {OUT} with {len(result)} rows")
