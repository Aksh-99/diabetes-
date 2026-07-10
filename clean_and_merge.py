import pandas as pd
import os

ROOT = "nhanes_raw/2021-2023"

PATHS = {
    "DEMO":    f"{ROOT}/Demographics/DEMO_L.xpt",
    "GLU":     f"{ROOT}/Laboratory/GLU_L.xpt",
    "GHB":     f"{ROOT}/Laboratory/GHB_L.xpt",
    "INS":     f"{ROOT}/Laboratory/INS_L.xpt",
    "BMX":     f"{ROOT}/Examination/BMX_L.xpt",
    "BPXO":    f"{ROOT}/Examination/BPXO_L.xpt",
    "DIQ":     f"{ROOT}/Questionnaire/DIQ_L.xpt",
    "RHQ":     f"{ROOT}/Questionnaire/RHQ_L.xpt",
}


def load(name, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {name} file at {path}.")
    df = pd.read_sas(path, format="xport")
    print(f"[loaded] {name}: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def main():
    dfs = {name: load(name, path) for name, path in PATHS.items()}

    # Demographics
    demo = dfs["DEMO"][["SEQN", "RIDAGEYR", "RIAGENDR"]].rename(
        columns={"RIDAGEYR": "Age", "RIAGENDR": "sex"}
    )

    # Glucose
    glu = dfs["GLU"][["SEQN", "LBXGLU"]].rename(columns={"LBXGLU": "Glucose"})

    # HbA1c (used for Outcome label only, not a model feature)
    ghb = dfs["GHB"][["SEQN", "LBXGH"]].rename(columns={"LBXGH": "hba1c"})

    # Insulin
    ins = dfs["INS"][["SEQN", "LBXIN"]].rename(columns={"LBXIN": "Insulin"})

    # BMI
    bmx = dfs["BMX"][["SEQN", "BMXBMI"]].rename(columns={"BMXBMI": "BMI"})

    # Blood Pressure: average available systolic readings (BPXOSY1/2/3).
    bpxo = dfs["BPXO"][["SEQN", "BPXOSY1", "BPXOSY2", "BPXOSY3"]].copy()
    bpxo["BloodPressure"] = bpxo[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1, skipna=True)
    bpxo = bpxo[["SEQN", "BloodPressure"]]

    # Diabetes self-report (for Outcome label)
    diq = dfs["DIQ"][["SEQN", "DIQ010"]].rename(columns={"DIQ010": "self_report_raw"})
    diq["self_report_raw"] = diq["self_report_raw"].replace({7: None, 9: None})

    # Pregnancies: RHQ131 "Ever been pregnant?" (1=Yes, 2=No) 
    if "RHQ131" in dfs["RHQ"].columns:
        rhq = dfs["RHQ"][["SEQN", "RHQ131"]].rename(columns={"RHQ131": "pregnancies_raw"})
        rhq["pregnancies_raw"] = rhq["pregnancies_raw"].replace({7: None, 9: None})
        rhq["Pregnancies"] = rhq["pregnancies_raw"].map({1: 1, 2: 0})
        rhq = rhq[["SEQN", "Pregnancies"]]
    else:
        print("[warn] RHQ131 not found - check columns:", dfs["RHQ"].columns.tolist())
        rhq = dfs["RHQ"][["SEQN"]].copy()
        rhq["Pregnancies"] = None

    # Merge everything on SEQN
    df = demo.merge(glu, on="SEQN", how="left") \
              .merge(ghb, on="SEQN", how="left") \
              .merge(ins, on="SEQN", how="left") \
              .merge(bmx, on="SEQN", how="left") \
              .merge(bpxo, on="SEQN", how="left") \
              .merge(diq, on="SEQN", how="left") \
              .merge(rhq, on="SEQN", how="left")

    print(f"\n[merged] {df.shape[0]} rows, {df.shape[1]} cols")

    # Adult-only filter
    before_age = len(df)
    df = df[df["Age"] >= 18].copy()
    print(f"[filter] Dropped {before_age - len(df)} rows for Age < 18")

    # Range sanity checks
    checks = {
        "Glucose": (30, 700), "BMI": (10, 90), "Insulin": (0, 900),
        "BloodPressure": (40, 250), "Age": (0, 100),
    }
    for col, (lo, hi) in checks.items():
        bad = (df[col] < lo) | (df[col] > hi)
        if bad.sum() > 0:
            print(f"  [clip] {col}: nulled {bad.sum()} out-of-range value(s)")
        df.loc[bad, col] = None

    # Outcome label (self-report + HbA1c only) 
    def label_row(row):
        if row["self_report_raw"] == 1:
            return 1
        if pd.notna(row["hba1c"]) and row["hba1c"] >= 6.5:
            return 1
        if pd.isna(row["self_report_raw"]) and pd.isna(row["hba1c"]):
            return None
        return 0

    df["Outcome"] = df.apply(label_row, axis=1)

    # Drop rows with no diagnostic signal at all
    before = len(df)
    df = df.dropna(subset=["Outcome"]).copy()
    print(f"[filter] Dropped {before - len(df)} rows with no diagnostic signal")

    # Final column order matching Clean style
    final_cols = ["Pregnancies", "Glucose", "BloodPressure", "Insulin", "BMI", "Age", "Outcome"]
    out = df[final_cols].copy()

    n_missing_preg = out["Pregnancies"].isna().sum()
    n_male = (df["sex"] == 1).sum()
    print(f"\n[note] Pregnancies is NaN for {n_missing_preg} rows "
          f"(RHQ is females-only; {n_male} rows in this dataset are male and will always be NaN here).")
    print("[note] Consider: (a) restricting to female-only rows to match Clean's original population, "
          "or (b) filling male rows with 0 since males cannot be pregnant - "
          "this script does NEITHER automatically, so decide before modeling.")

    out.to_csv("nhanes_Clean_style.csv", index=False)
    print(f"\nSaved: nhanes_Clean_style.csv -> {out.shape[0]} rows, {out.shape[1]} cols")
    print(f"\nOutcome distribution:\n{out['Outcome'].value_counts()}")
    print(f"\nMissingness by column:")
    print(out.isna().mean().sort_values(ascending=False).round(3))


if __name__ == "__main__":
    main()