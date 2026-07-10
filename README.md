# NHANES Diabetes Risk Prediction

A diabetes classification pipeline built on real CDC NHANES (National Health and Nutrition Examination Survey) data, structured to mirror the well-known Pima Indians Diabetes schema. Unlike the original Pima dataset, which is a fixed, decades-old sample, this project pulls, cleans, and labels current NHANES cycle data end-to-end, from raw government lab files to a trained classifier.

## Why this project

Most public "diabetes prediction" tutorials reuse the same static Pima Indians CSV. This project instead builds the equivalent dataset from scratch using the current NHANES 2021-2023 cycle, which means handling real-world problems the Pima CSV already has "cleaned away": missing lab values, self-report inconsistencies, unit mismatches across files, and label definitions that have to be built rather than assumed. The goal was to demonstrate a full pipeline, not just fit a model on a tidy CSV.

## Pipeline overview

The project is split into three scripts that run in sequence:

```
data.py           -> clean_and_merge.py    -> model.py
(download raw      (merge, clean, label,     (train + evaluate
 NHANES .xpt        derive Outcome)           SVC classifier)
 files)
```

### 1. `data.py` — data acquisition

Scrapes the CDC NHANES data page (`wwwn.cdc.gov/Nchs/Nhanes/search/datapage.aspx`) for a given survey component and cycle, finds all `.xpt` (SAS transport) file links, and downloads them locally.

```bash
python data.py --component Laboratory --cycle 2021-2023 --filter GLU --load
```

- `--component`: one of Demographics, Dietary, Examination, Laboratory, Questionnaire, LimitedAccess
- `--cycle`: NHANES survey cycle, e.g. `2021-2023`
- `--filter`: optional substring to only download matching files
- `--load`: optionally load each file into pandas and print its shape, as a sanity check

Files are saved to `nhanes_raw/<cycle>/<component>/`. Already-downloaded files are skipped on rerun.

For this project, the following NHANES files were pulled from the **2021-2023** cycle:

| File | Component | Contents used |
|---|---|---|
| `DEMO_L` | Demographics | Age, sex |
| `GLU_L` | Laboratory | Fasting plasma glucose |
| `GHB_L` | Laboratory | HbA1c (used for labeling only) |
| `INS_L` | Laboratory | Insulin |
| `BMX_L` | Examination | BMI |
| `BPXO_L` | Examination | Oscillometric blood pressure readings |
| `DIQ_L` | Questionnaire | Self-reported diabetes diagnosis |
| `RHQ_L` | Questionnaire | Reproductive health (pregnancy proxy) |

### 2. `clean_and_merge.py` — cleaning, merging, and labeling

Loads each raw `.xpt` file, extracts the relevant columns, merges everything on the shared respondent ID (`SEQN`), and produces a single clean CSV: `nhanes_Clean_style.csv`.

```bash
python clean_and_merge.py
```

**Column mapping (NHANES → Pima-style schema):**

| Output column | Source | Notes |
|---|---|---|
| `Pregnancies` | `RHQ131` ("ever pregnant?") | Binary proxy (1/0), not a lifetime count — see limitations below |
| `Glucose` | `GLU_L.LBXGLU` | Fasting plasma glucose |
| `BloodPressure` | `BPXO_L.BPXOSY1/2/3` (mean) | **Systolic**, not diastolic — see limitations below |
| `Insulin` | `INS_L.LBXIN` | |
| `BMI` | `BMX_L.BMXBMI` | |
| `Age` | `DEMO_L.RIDAGEYR` | |
| `Outcome` | Derived, see below | Binary diabetes label |

**Label derivation (`Outcome`):**

A respondent is labeled diabetic (`1`) if:
- They self-report a prior diabetes diagnosis (`DIQ010 == 1`), **or**
- Their HbA1c is ≥ 6.5% (the standard ADA diagnostic threshold)

Respondents with neither self-report nor HbA1c available are dropped (no diagnostic signal). Everyone else is labeled `0`.

**Important design decision — avoiding label leakage:** Glucose is deliberately excluded from the label logic, even though a fasting glucose ≥ 126 mg/dL is also a clinically valid ADA diagnostic criterion. This is because Glucose is one of the six model features. Using it to define the label as well would mean the model could "predict" diabetes simply by rediscovering the threshold used to build its own target variable, artificially inflating accuracy without learning anything predictive. HbA1c is used for labeling but deliberately **not** included as a feature, for the same reason.

**Other cleaning steps:**
- Age filtered to adults only (`Age >= 18`), since Pregnancies/BMI-based diabetes risk isn't meaningful for children and the Pima population this schema mirrors was adult-only.
- Range sanity checks null out biologically implausible values (e.g. Glucose outside 30-700 mg/dL, BMI outside 10-90).
- Out-of-range or missing values are set to `NaN` rather than dropped, so missingness is handled explicitly downstream (in `model.py`) instead of silently losing rows.

### 3. `model.py` — training and evaluation

```bash
python model.py
```

Steps:
1. Loads `nhanes_Clean_style.csv`
2. Imputes missing values:
   - `Pregnancies`: filled with `0`. NaN here means the respondent was male (the `RHQ` questionnaire is administered to females only), not missing data, so `0` is the correct fill rather than a statistical guess.
   - `Glucose`, `BloodPressure`, `Insulin`, `BMI`: median imputation
3. Standardizes all features with `StandardScaler`
4. Splits into train/test (80/20, stratified on Outcome)
5. Trains a linear-kernel SVM (`sklearn.svm.SVC(kernel='linear')`)
6. Reports training and test accuracy
7. Demonstrates single-record inference on a manually specified input

## Results

| Metric | Value |
|---|---|
| Training accuracy | ~92.5%* |
| Test accuracy | ~92.3%* |

\*From the version of the pipeline that still included Glucose in the label logic. After removing Glucose from `label_row()` to fix label leakage, these numbers are expected to drop — a lower, leakage-free accuracy is the more honest and defensible result and should be reported once the corrected pipeline is rerun.

## Known limitations

- **Pregnancies is a binary proxy, not a count.** The original NHANES question that asked for lifetime pregnancy count (`RHQ160`) was dropped from the questionnaire in this cycle. `RHQ131` ("have you ever been pregnant") is used instead, converted to 1/0. This does not carry the same information as the original Pima `Pregnancies` count and should be described as a proxy, not treated as equivalent.
- **BloodPressure is systolic, not diastolic.** The original Pima `BloodPressure` column is diastolic blood pressure. This project uses the mean of three systolic oscillometric readings (`BPXOSY1/2/3`) instead, since that was the available measurement in `BPXO_L`. The column is named `BloodPressure` for schema consistency with Pima, but the underlying clinical measurement differs.
- **Insulin has substantial missingness** (roughly half of rows in early exploration). NHANES only samples insulin for a subset of participants. Median imputation is used, but this is a coarse fix for a feature that's missing at this scale; a missingness indicator or dropping the feature are alternatives worth exploring.
- **Scaler and imputer are currently fit on the full dataset before the train/test split**, which is a mild form of data leakage (affects computed means/medians, not labels). For a stricter methodology, these should be fit on the training split only and applied via `.transform()` to the test split.
- **Linear-kernel SVM** was used for simplicity/interpretability; no hyperparameter tuning or comparison against other model families (e.g. gradient boosting, random forest) has been done yet.

## Possible next steps

- Refit imputer/scaler on training data only to remove the remaining leakage
- Compare against `HistGradientBoostingClassifier` (handles NaN natively, no imputation needed)
- Add a missingness indicator feature for Insulin instead of pure median imputation
- Package into a FastAPI endpoint for deployment
- Add SHAP-based feature importance for interpretability

## Tech stack

- **Data acquisition**: `requests`, `BeautifulSoup` (web scraping CDC data page)
- **Data processing**: `pandas`, `numpy`
- **Modeling**: `scikit-learn` (`SVC`, `StandardScaler`, `SimpleImputer`, `train_test_split`)
- **Source data**: CDC NHANES 2021-2023 cycle, public domain
