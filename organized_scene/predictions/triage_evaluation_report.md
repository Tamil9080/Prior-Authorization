# Triage Engine Evaluation Report

This report evaluates the Prior Authorization Triage and Policy Companion engine (Random Forest prediction combined with CMS Local Coverage Determination rules).

## 1. Overall Performance Metrics
* **Accuracy**: 0.5680
* **Macro Precision**: 0.5419
* **Macro Recall**: 0.5452
* **Macro F1-Score**: 0.4900
* **Approved Binary ROC-AUC**: 0.9755
* **Approved Binary PR-AUC**: 0.9718

## 2. Safety Audit Metrics
* **False Approvals (Critical Risk)**: 32
  *Description: The system recommended APPROVE, but the clinical record should have been Denied or Pended.*
* **False Denials (Friction Risk)**: 188
  *Description: The system recommended Denied or Pended, but the clinical record was Approved.*

## 3. Confusion Matrix
Labels: `['Approved', 'Denied', 'Pending Additional Information']`

```
[[304 137  51]
 [ 12 235  14]
 [ 20 198  29]]
```
