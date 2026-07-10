# NHANES Diabetes Risk Prediction

A diabetes prediction pipeline built from raw CDC health data, handling everything from download to cleaning to labeling to model training with no pre-cleaned dataset involved.

## Pipeline

```
data.py  →  clean_and_merge.py  →  model.py
(scrape/download    (merge, clean,      (train + evaluate
NHANES .xpt files)   derive Outcome)     SVC classifier)
```

**1. `data.py`** — scrapes CDC's NHANES data page and downloads `.xpt` files for a given component/cycle.
```bash
python data.py --component Laboratory --cycle 2021-2023 --filter GLU --load
```
Pulled from the 2021-2023 cycle: `DEMO_L` (age/sex), `GLU_L` (glucose), `GHB_L` (HbA1c), `INS_L` (insulin), `BMX_L` (BMI), `BPXO_L` (blood pressure), `DIQ_L` (self-reported diagnosis), `RHQ_L` (pregnancy proxy).

**2. `clean_and_merge.py`** — merges all files on respondent ID, maps to Pima-style columns, derives the `Outcome` label, and outputs `nhanes_Clean_style.csv`.
```bash
python clean_and_merge.py
```
Label = diabetic if self-reported diagnosis OR HbA1c ≥ 6.5%. **Glucose is deliberately excluded from the label** even though it's a valid ADA criterion, because it's also a model feature — using it for both would let the model trivially "predict" its own label. Adults only (Age ≥ 18); implausible values nulled rather than dropped.

**3. `model.py`** — imputes missing values, scales features, trains a linear-kernel SVM, reports accuracy, and demos single-record inference.
```bash
python model.py
```

## Results

| Metric | Value |
|---|---|
| Training accuracy | 88.1%|
| Test accuracy | 88.2% |

*(Earlier run showed ~92%, but that included Glucose in the label logic. Numbers will be updated after rerunning the corrected pipeline.)*

## Known limitations

- **Pregnancies** is a binary "ever pregnant" proxy, not a lifetime count (NHANES dropped the count question this cycle).
- **BloodPressure** is systolic (mean of 3 readings), not diastolic like the original Pima column.
- **Insulin** has ~50% missingness (NHANES only samples a subset); currently median-imputed.
- Scaler/imputer are still fit on the full dataset before the train/test split (mild leakage on means/medians, not labels) — next fix.

## Next steps

- Fit imputer/scaler on training split only
- Try `HistGradientBoostingClassifier` (handles NaN natively)
- Add a missingness indicator for Insulin
- Deploy via FastAPI; add SHAP for interpretability

## Tech stack

`requests`, `BeautifulSoup` · `pandas`, `numpy` · `scikit-learn` (SVC, StandardScaler, SimpleImputer) · CDC NHANES 2021-2023 (public domain)
