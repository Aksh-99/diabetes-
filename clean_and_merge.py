"""
NHANES 2021-2023 Diabetes Dataset - Clean & Merge
---------------------------------------------------
Loads the 5 downloaded .xpt files, cleans each one, merges them on SEQN,
derives a diabetes label from ADA clinical criteria, and writes one CSV.

Expects this folder structure (created by nhanes_downloader.py):
    nhanes_raw/2021-2023/Demographics/DEMO_L.xpt
    nhanes_raw/2021-2023/Laboratory/GLU_L.xpt
    nhanes_raw/2021-2023/Laboratory/GHB_L.xpt
    nhanes_raw/2021-2023/Examination/BMX_L.xpt
    nhanes_raw/2021-2023/Questionnaire/DIQ_L.xpt

Usage:
    python clean_and_merge.py

Output:
    nhanes_diabetes_clean.csv
"""

import pandas as pd
import os

ROOT = "nhanes_raw/2021-2023"

PATHS = {
    "DEMO": f"{ROOT}/Demographics/DEMO_L.xpt",
    "GLU":  f"{ROOT}/Laboratory/GLU_L.xpt",
    "GHB":  f"{ROOT}/Laboratory/GHB_L.xpt",
    "BMX":  f"{ROOT}/Examination/BMX_L.xpt",
    "DIQ":  f"{ROOT}/Questionnaire/DIQ_L.xpt",
}


def load(name, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {name} file at {path}. Did the download finish?")
    df = pd.read_sas(path, format="xport")
    print(f"[loaded] {name}: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def main():
    demo = load("DEMO", PATHS["DEMO"])
    glu  = load("GLU",  PATHS["GLU"])
    ghb  = load("GHB",  PATHS["GHB"])
    bmx  = load("BMX",  PATHS["BMX"])
    diq  = load("DIQ",  PATHS["DIQ"])

    # ---- 1. Demographics: keep + clean ----
    demo = demo[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3"]].copy()
    demo = demo.rename(columns={
        "RIDAGEYR": "age",
        "RIAGENDR": "sex",       # 1 = Male, 2 = Female
        "RIDRETH3": "race_eth",
    })
    # NHANES caps reported age at 80 ("80 years of age and older") - keep as-is
    # but flag it so downstream modeling can treat it as censored if needed.
    demo["age_topcoded"] = (demo["age"] == 80)

    # ---- 2. Fasting glucose ----
    # LBXGLU = Fasting Glucose (mg/dL). Only measured on a MEC fasting subsample,
    # so many SEQNs will have NaN here - that's expected, not an error.
    glu = glu[["SEQN", "LBXGLU"]].copy()
    glu = glu.rename(columns={"LBXGLU": "fasting_glucose"})

    # ---- 3. HbA1c ----
    ghb = ghb[["SEQN", "LBXGH"]].copy()
    ghb = ghb.rename(columns={"LBXGH": "hba1c"})

    # ---- 4. BMI / body measures ----
    bmx = bmx[["SEQN", "BMXBMI", "BMXWT", "BMXHT", "BMXWAIST"]].copy()
    bmx = bmx.rename(columns={
        "BMXBMI": "bmi",
        "BMXWT": "weight_kg",
        "BMXHT": "height_cm",
        "BMXWAIST": "waist_cm",
    })

    # ---- 5. Diabetes questionnaire ----
    # DIQ010 codes: 1=Yes, 2=No, 3=Borderline/Prediabetes, 7=Refused, 9=Don't know
    diq = diq[["SEQN", "DIQ010"]].copy()
    diq = diq.rename(columns={"DIQ010": "self_report_raw"})
    # Recode refused/don't know as missing rather than a real answer
    diq["self_report_raw"] = diq["self_report_raw"].replace({7: None, 9: None})

    # ---- Merge everything on SEQN ----
    df = demo.merge(glu, on="SEQN", how="left") \
              .merge(ghb, on="SEQN", how="left") \
              .merge(bmx, on="SEQN", how="left") \
              .merge(diq, on="SEQN", how="left")

    print(f"\n[merged] {df.shape[0]} rows, {df.shape[1]} cols")

    # ---- Sanity-check ranges and clear out biologically impossible values ----
    # (NHANES rarely has these, but worth guarding against corrupt/miscoded rows)
    df.loc[(df["bmi"] < 10) | (df["bmi"] > 90), "bmi"] = None
    df.loc[(df["fasting_glucose"] < 30) | (df["fasting_glucose"] > 700), "fasting_glucose"] = None
    df.loc[(df["hba1c"] < 2) | (df["hba1c"] > 20), "hba1c"] = None

    # ---- Derive diabetes label using ADA criteria ----
    # Positive if: self-reported diagnosis (1) OR HbA1c >= 6.5% OR fasting glucose >= 126 mg/dL
    # Only assign 0 if we have at least one real signal and none of them are positive.
    def label_row(row):
        if row["self_report_raw"] == 1:
            return 1
        if pd.notna(row["hba1c"]) and row["hba1c"] >= 6.5:
            return 1
        if pd.notna(row["fasting_glucose"]) and row["fasting_glucose"] >= 126:
            return 1
        if pd.isna(row["self_report_raw"]) and pd.isna(row["hba1c"]) and pd.isna(row["fasting_glucose"]):
            return None  # no signal at all - can't label this person
        return 0

    df["diabetic"] = df.apply(label_row, axis=1)

    # ---- Also keep a 3-class prediabetes-aware label, since DIQ010==3 means "borderline" ----
    def label_row_3class(row):
        if row["self_report_raw"] == 1 or (pd.notna(row["hba1c"]) and row["hba1c"] >= 6.5) or \
           (pd.notna(row["fasting_glucose"]) and row["fasting_glucose"] >= 126):
            return "diabetic"
        if row["self_report_raw"] == 3 or (pd.notna(row["hba1c"]) and 5.7 <= row["hba1c"] < 6.5) or \
           (pd.notna(row["fasting_glucose"]) and 100 <= row["fasting_glucose"] < 126):
            return "prediabetic"
        if pd.isna(row["self_report_raw"]) and pd.isna(row["hba1c"]) and pd.isna(row["fasting_glucose"]):
            return None
        return "no_diabetes"

    df["diabetes_status"] = df.apply(label_row_3class, axis=1)

    # ---- Drop rows with zero diagnostic signal (can't be labeled at all) ----
    before = len(df)
    df_labeled = df.dropna(subset=["diabetic"]).copy()
    print(f"[filter] Dropped {before - len(df_labeled)} rows with no diagnostic signal "
          f"(no self-report, no HbA1c, no fasting glucose)")

    # ---- Save both versions: full merged (for reference) and labeled-only (for modeling) ----
    df.to_csv("nhanes_diabetes_full.csv", index=False)
    df_labeled.to_csv("nhanes_diabetes_clean.csv", index=False)

    print(f"\nSaved:")
    print(f"  nhanes_diabetes_full.csv    -> {df.shape[0]} rows (includes unlabeled)")
    print(f"  nhanes_diabetes_clean.csv   -> {df_labeled.shape[0]} rows (ready for modeling)")
    print(f"\nBinary label distribution:\n{df_labeled['diabetic'].value_counts()}")
    print(f"\n3-class label distribution:\n{df_labeled['diabetes_status'].value_counts()}")
    print(f"\nMissingness by column (in clean dataset):")
    print(df_labeled.isna().mean().sort_values(ascending=False).round(3))


if __name__ == "__main__":
    main()