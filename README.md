# Employee Attrition Analysis

A workforce-analytics project combining exploratory analysis, a Random Forest classification workflow, and a Power BI report.

## Objective
Explore patterns associated with employee attrition and build a classification model that can help identify variables associated with employees leaving.

## Analysis workflow
The Python script:
- loads an employee dataset from `Employee_Cleaned.xls`;
- visualizes attrition counts;
- compares attrition rates by city and payment tier;
- examines age distributions for employees who stayed versus left;
- one-hot encodes categorical variables;
- creates a train/test split;
- trains a `RandomForestClassifier`;
- calculates feature importance and model accuracy.

## Repository contents
- `Employee Attrition Analysis.py` — Python analysis and modeling.
- `Employee Data Analysis.pbix` — Power BI report.

## Reproducibility limitation
The script expects `Employee_Cleaned.xls`, but that dataset is not currently committed. The repository therefore cannot reproduce the analysis from a fresh clone as-is.

## Methodology note
The current train/test split does not specify `stratify=y`, and the README does not claim a model accuracy because the committed source file does not contain the executed output. A future revision should add class-balance checks, precision/recall/F1/ROC-AUC, cross-validation, and leakage review before treating the model as decision support.

## Responsible use
Feature importance indicates model association, not causation. Workforce models should be reviewed for fairness, privacy, and appropriate human oversight and should not be used as an automated employment decision system.

## Next improvements
Add the authoritative dataset or documented acquisition instructions, dependency management, saved dashboard screenshots, reproducible outputs, a data dictionary, and clearer separation between exploratory analytics and predictive modeling.

## Skills demonstrated
Python · pandas · scikit-learn · Random Forest · Workforce Analytics · Power BI · Data Visualization

## Author
Martin Ngare
