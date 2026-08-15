# Model Evaluation Report

This report summarizes the performance metrics of the trained Random Forest classifiers in the Prior Authorization platform.

## 1. Aggregate Carrier-Level Model
* **Accuracy**: 0.9733
* **Macro Precision**: 0.9807
* **Macro Recall**: 0.9537
* **Macro F1-Score**: 0.9660

## 2. Clinical Patient-Level Model (Target Leakage Fixed)
* **Accuracy**: 0.9010
* **Macro Precision**: 0.9052
* **Macro Recall**: 0.8883
* **Macro F1-Score**: 0.8950

## 3. Review Decision Model (Target Leakage & Merge Leakage Fixed)
### Target: `decision`
* **Accuracy**: 0.9223
* **Macro Precision**: 0.9242
* **Macro Recall**: 0.8881
* **Macro F1-Score**: 0.9042

### Target: `meets_criteria`
* **Accuracy**: 0.8694
* **Macro Precision**: 0.8854
* **Macro Recall**: 0.8400
* **Macro F1-Score**: 0.8545

### Target: `decision_reason`
* **Accuracy**: 0.8148
* **Macro Precision**: 0.7810
* **Macro Recall**: 0.8096
* **Macro F1-Score**: 0.7789

