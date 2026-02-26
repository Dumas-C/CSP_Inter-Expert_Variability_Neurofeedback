# -*- coding: utf-8 -*-
"""
Expert influence on CSP selection:
- Familiarity (3-level ordered) -> linear trend only
- Background (2 groups, STRICT BCI keyword) -> t-test + Mann–Whitney + assumptions
- Experience (2 groups: <5y vs >=5y) -> t-test + Mann–Whitney + assumptions

Stats only (no figures).

Copyright (c) 2026 Paris Brain Institute.
Created: Feb 2026
Author: Cassandra Dumas
"""

import os
import numpy as np
import pandas as pd

from scipy.stats import shapiro, levene, linregress, ttest_ind, mannwhitneyu



# ==========================================================
# Shared helpers
# ==========================================================
SUBJECT_ORDER = [3, 18, 7, 22, 12, 5, 24, 11, 20, 6, 16, 14, 23, 8, 19, 2, 17, 15, 13, 21]
CSP_CHOICES_LETTERS = ["A", "B", "C", "D", "E", "F"]

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

def _load_questionnaire(csv_name="CSP_Selection_Answers.csv"):
    """Load CSV and return (data, df_conf, df_ch, fam_col, background_col, exp_col)."""
    data = pd.read_csv(os.path.join(project_root, "Data", csv_name))
    subj = list(SUBJECT_ORDER)

    conf_cols = [c for c in data.columns if "confident" in c.lower()]
    choice_cols = [c for c in data.columns if "[Choice]" in c]

    fam_col = next(c for c in data.columns if "familiar" in c.lower())
    background_col = next(c for c in data.columns if "describe" in c.lower())
    exp_col = next(c for c in data.columns if "experience" in c.lower())

    n = min(len(subj), len(conf_cols), len(choice_cols))
    if n == 0:
        raise RuntimeError("No aligned columns found (confident / [Choice]) or SUBJECT_ORDER empty.")

    df_conf = pd.DataFrame({f"Subject {subj[i]}": data[conf_cols[i]].values for i in range(n)})
    df_ch = pd.DataFrame({f"Subject {subj[i]}": data[choice_cols[i]].astype(str).values for i in range(n)})

    cols = sorted(df_ch.columns, key=lambda s: int(s.split()[1]))
    df_conf, df_ch = df_conf[cols], df_ch[cols]

    return data, df_conf, df_ch, fam_col, background_col, exp_col


def _expert_level_metrics(df_conf: pd.DataFrame, df_ch: pd.DataFrame) -> pd.DataFrame:
    """Compute expert-level mean confidence + individual agreement (vs per-subject consensus)."""
    out = pd.DataFrame(index=df_conf.index)
    out["Mean_Confidence"] = df_conf.mean(axis=1)

    consensus = df_ch.mode(axis=0).iloc[0]
    out["Individual_Agreement"] = (df_ch == consensus).mean(axis=1)

    return out


def _print_assumptions_two_groups(df: pd.DataFrame, group_col: str, groups, value_col: str):
    """Shapiro per group (if n>=3) + Levene (median-centered)."""
    print(f"\n--- Assumptions: {value_col} ---")
    arrays = []
    for g in groups:
        vals = df.loc[df[group_col] == g, value_col].dropna().to_numpy(dtype=float)
        arrays.append(vals)
        if len(vals) >= 3:
            W, p = shapiro(vals)
            print(f"Shapiro {g}: W={W:.3f}, p={p:.3f} (n={len(vals)})")
        else:
            print(f"Shapiro {g}: not tested (n={len(vals)} < 3)")
    if all(len(a) > 0 for a in arrays) and len(arrays) >= 2:
        stat, p = levene(*arrays, center="median")
        print(f"Levene (median-centered): stat={stat:.3f}, p={p:.3f}")
    else:
        print("Levene: not tested (empty group)")


def _cohens_d_independent(x: pd.Series, y: pd.Series):
    """Cohen's d with pooled SD for independent samples (NaN-safe)."""
    x = x.dropna()
    y = y.dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = np.sqrt(((len(x) - 1) * vx + (len(y) - 1) * vy) / (len(x) + len(y) - 2))
    if pooled <= 0 or np.isnan(pooled):
        return np.nan
    return float((x.mean() - y.mean()) / pooled)


def _linear_trend(values: pd.Series, ordered_cat: pd.Series):
    """
    Linear trend via simple regression: values ~ code(1..K).
    Returns slope, r, p, n.
    """
    if not hasattr(ordered_cat, "cat"):
        raise TypeError("ordered_cat must be a pandas Categorical Series (df[col] as categorical).")
    codes = ordered_cat.cat.codes.astype(float) + 1.0  # 1..K, NaN -> code=0
    m = (ordered_cat.cat.codes >= 0) & values.notna()
    x = codes[m].to_numpy(dtype=float)
    y = values[m].to_numpy(dtype=float)
    if len(y) < 3:
        return np.nan, np.nan, np.nan, int(len(y))
    res = linregress(x, y)
    return float(res.slope), float(res.rvalue), float(res.pvalue), int(len(y))


# ==========================================================
# Main
# ==========================================================
def Expert_Influence_Analysis(csv_name="CSP_Selection_Answers.csv"):
    """
    Run the requested reduced analyses (stats only):
    - Familiarity: 3-level linear trend (Mean Confidence + Individual Agreement)
    - Background: STRICT 2 groups (BCI keyword only) with t-test + MWU
    - Experience: 2 groups (<5y vs >=5y) with t-test + MWU
    """
    data, df_conf, df_ch, fam_col, background_col, exp_col = _load_questionnaire(csv_name=csv_name)
    base = _expert_level_metrics(df_conf, df_ch)

    # ==========================================================
    # Familiarity (3 levels) - linear trend only
    # ==========================================================
    fam_raw = data[fam_col].astype(str).str.strip()

    fam_no = "No familiarity"
    fam_lim = "Limited familiarity (general knowledge or occasional use)"
    fam_mod = "Moderate familiarity (regular use or solid theoretical understanding)"
    fam_high = "High familiarity (extensive use in research or applied BCI/neurofeedback)"
    fam3_merge = "Moderate + High familiarity"
    g3 = [fam_no, fam_lim, fam3_merge]

    fam3 = fam_raw.replace({fam_mod: fam3_merge, fam_high: fam3_merge})
    fam3 = pd.Categorical(fam3, categories=g3, ordered=True)

    df_fam = base.copy()
    df_fam["Fam3"] = fam3

    print("\n==============================")
    print(" FAMILIARITY (3 LEVELS) - LINEAR TREND ONLY")
    print("==============================")
    print("Levels:", " | ".join(["None", "Limited", "Moderate/High"]))
    print("\nGroup counts:")
    print(pd.Series(df_fam["Fam3"]).value_counts(dropna=False).reindex(g3))

    slope_c, r_c, p_c, n_c = _linear_trend(df_fam["Mean_Confidence"], df_fam["Fam3"])
    slope_a, r_a, p_a, n_a = _linear_trend(df_fam["Individual_Agreement"], df_fam["Fam3"])

    print("\nLinear trend (values ~ familiarity code 1..3):")
    print(f"Mean Confidence:      slope = {slope_c:.3f}, r = {r_c:.3f}, p = {p_c:.3f} (n={n_c})")
    print(f"Individual Agreement: slope = {slope_a:.3f}, r = {r_a:.3f}, p = {p_a:.3f} (n={n_a})")

    # ==========================================================
    # Background (STRICT) - 2 groups only
    # ==========================================================
    bg_raw = data[background_col].astype(str).str.strip()
    bg = bg_raw.str.lower()

    is_bci_strict = (
        bg.str.contains(r"\bbci\b", regex=True, na=False)
        | bg.str.contains(r"brain[-\s]?computer[-\s]?interface", regex=True, na=False)
    )

    df_bg = base.copy()
    df_bg["Background_raw"] = bg_raw
    df_bg["Background_2_strict"] = np.where(is_bci_strict, "BCI", "NON BCI")
    g2_bg = ["BCI", "NON BCI"]
    df_bg["Background_2_strict"] = pd.Categorical(df_bg["Background_2_strict"], categories=g2_bg, ordered=True)

    print("\n==============================")
    print(" BACKGROUND (STRICT) - 2 GROUPS")
    print(" BCI only if: 'BCI' or 'brain-computer interface'")
    print("==============================")
    print("\nGroup counts:")
    print(df_bg["Background_2_strict"].value_counts(dropna=False).reindex(g2_bg))

    # Assumptions
    _print_assumptions_two_groups(df_bg, "Background_2_strict", g2_bg, "Mean_Confidence")
    _print_assumptions_two_groups(df_bg, "Background_2_strict", g2_bg, "Individual_Agreement")

    # Tests: Mean Confidence
    x = df_bg.loc[df_bg["Background_2_strict"] == "BCI", "Mean_Confidence"].dropna()
    y = df_bg.loc[df_bg["Background_2_strict"] == "NON BCI", "Mean_Confidence"].dropna()
    if len(x) > 0 and len(y) > 0:
        t, p = ttest_ind(x, y, equal_var=True)
        d = _cohens_d_independent(x, y)
        U, pU = mannwhitneyu(x.to_numpy(float), y.to_numpy(float), alternative="two-sided")
        print("\nMean Confidence:")
        print(f"t-test (equal_var=True): t = {float(t):.3f}, p = {float(p):.3f}, d = {d:.3f}")
        print(f"Mann–Whitney (two-sided): U = {float(U):.3f}, p = {float(pU):.3f}")
    else:
        print("\nMean Confidence: skipped (empty group).")

    # Tests: Individual Agreement
    x = df_bg.loc[df_bg["Background_2_strict"] == "BCI", "Individual_Agreement"].dropna()
    y = df_bg.loc[df_bg["Background_2_strict"] == "NON BCI", "Individual_Agreement"].dropna()
    if len(x) > 0 and len(y) > 0:
        t, p = ttest_ind(x, y, equal_var=True)
        d = _cohens_d_independent(x, y)
        U, pU = mannwhitneyu(x.to_numpy(float), y.to_numpy(float), alternative="two-sided")
        print("\nIndividual Agreement:")
        print(f"t-test (equal_var=True): t = {float(t):.3f}, p = {float(p):.3f}, d = {d:.3f}")
        print(f"Mann–Whitney (two-sided): U = {float(U):.3f}, p = {float(pU):.3f}")
    else:
        print("\nIndividual Agreement: skipped (empty group).")

    # ==========================================================
    # Experience - 2 groups only (<5y vs >=5y)
    # ==========================================================
    exp_raw = data[exp_col].astype(str).str.strip()

    df_exp = base.copy()
    df_exp["Experience_raw"] = exp_raw
    df_exp["Experience_2"] = np.where(df_exp["Experience_raw"] == "< 5 years", "< 5 years", ">= 5 years")
    g2_exp = ["< 5 years", ">= 5 years"]
    df_exp["Experience_2"] = pd.Categorical(df_exp["Experience_2"], categories=g2_exp, ordered=True)

    print("\n==============================")
    print(" EXPERIENCE - 2 GROUPS")
    print(" Groups: < 5 years vs >= 5 years")
    print("==============================")
    print("\nGroup counts:")
    print(df_exp["Experience_2"].value_counts(dropna=False).reindex(g2_exp))

    # Assumptions
    _print_assumptions_two_groups(df_exp, "Experience_2", g2_exp, "Mean_Confidence")
    _print_assumptions_two_groups(df_exp, "Experience_2", g2_exp, "Individual_Agreement")

    # Tests: Mean Confidence
    x = df_exp.loc[df_exp["Experience_2"] == "< 5 years", "Mean_Confidence"].dropna()
    y = df_exp.loc[df_exp["Experience_2"] == ">= 5 years", "Mean_Confidence"].dropna()
    if len(x) > 0 and len(y) > 0:
        t, p = ttest_ind(x, y, equal_var=True)
        d = _cohens_d_independent(x, y)
        U, pU = mannwhitneyu(x.to_numpy(float), y.to_numpy(float), alternative="two-sided")
        print("\nMean Confidence:")
        print(f"t-test (equal_var=True): t = {float(t):.3f}, p = {float(p):.3f}, d = {d:.3f}")
        print(f"Mann–Whitney (two-sided): U = {float(U):.3f}, p = {float(pU):.3f}")
    else:
        print("\nMean Confidence: skipped (empty group).")

    # Tests: Individual Agreement
    x = df_exp.loc[df_exp["Experience_2"] == "< 5 years", "Individual_Agreement"].dropna()
    y = df_exp.loc[df_exp["Experience_2"] == ">= 5 years", "Individual_Agreement"].dropna()
    if len(x) > 0 and len(y) > 0:
        t, p = ttest_ind(x, y, equal_var=True)
        d = _cohens_d_independent(x, y)
        U, pU = mannwhitneyu(x.to_numpy(float), y.to_numpy(float), alternative="two-sided")
        print("\nIndividual Agreement:")
        print(f"t-test (equal_var=True): t = {float(t):.3f}, p = {float(p):.3f}, d = {d:.3f}")
        print(f"Mann–Whitney (two-sided): U = {float(U):.3f}, p = {float(pU):.3f}")
    else:
        print("\nIndividual Agreement: skipped (empty group).")