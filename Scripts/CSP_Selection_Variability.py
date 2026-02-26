# -*- coding: utf-8 -*-
"""
Inter-expert CSP selection variability (STATS ONLY)

What is kept:
- Load questionnaire choices
- Load CSP objects (patterns_)
- Compute per-subject mean topo + SD topo across experts' selections (Option B)
- Compute per-expert mean topo + SD topo across subjects' selections (Option B)
- ELECTRODE-WISE SD summaries (median + IQR) + per-subject / per-expert Top-N prints
- Channel-wise summaries for Fp1, Fp2, C3 (mean ± SD)
- No figures, no ERD, no saving

Copyright (c) 2026 Paris Brain Institute.
Created: Feb 2026
Author: Cassandra Dumas
"""

import os
import numpy as np
import pandas as pd
import mne
import joblib

# ==========================================================
# Shared helpers
# ==========================================================
SUBJECT_ORDER = [3, 18, 7, 22, 12, 5, 24, 11, 20, 6, 16, 14, 23, 8, 19, 2, 17, 15, 13, 21]
CSP_CHOICES_LETTERS = ["A", "B", "C", "D", "E", "F"]

CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'C3', 'Cz', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2'] 

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

csp_root = r"C:\Users\cassandra.dumas\OneDrive - ICM\Documents\PHD\CSP_Neurofeedback\BETAPARK\Data\_CSP_"


def CSP_Variability_Stats(TOP_N=10):
    # -------------------------
    # Paths
    # -------------------------
    csv_path = os.path.join(project_root, "Data", "CSP_Selection_Answers.csv")
    csp_path = os.path.join(csp_root, "_HAND_CALIBRATION_", "mutual_info")

    letter_to_idx = {k: i for i, k in enumerate("ABCDEF")}

    # -------------------------
    # Load questionnaire
    # -------------------------
    data = pd.read_csv(csv_path)
    choice_cols = [c for c in data.columns if "[Choice]" in c]
    subj_nums = list(SUBJECT_ORDER)

    df_choices = pd.DataFrame(
        {
            f"Subject {subj_nums[i]}": data[c].astype(str).values
            for i, c in enumerate(choice_cols[: len(subj_nums)])
        }
    ).reindex(
        sorted([f"Subject {s}" for s in subj_nums], key=lambda x: int(x.split()[1])),
        axis=1,
    )

    cols = df_choices.columns.tolist()
    subj_ids = [f"sub-S{int(c.split()[1]):03d}" for c in cols]

    # -------------------------
    # MNE info (for channel names + indices only)
    # -------------------------
    info = mne.create_info(
        CHANNELS,
        sfreq=1000.0,
        ch_types=["eeg"] * len(CHANNELS),
    )
    info.set_montage('easycap-M1', on_missing="ignore")
    ch_names = info["ch_names"]

    if "C3" not in ch_names:
        raise RuntimeError("Channel 'C3' not found in Channels.BETAPARK_STUDY.")
    c3_idx = ch_names.index("C3")

    # -------------------------
    # Load CSP objects
    # -------------------------
    csp_by_sub = {}
    for sid in subj_ids:
        obj = joblib.load(os.path.join(csp_path, f"CSP_{sid}.joblib"))
        csp = obj if hasattr(obj, "patterns_") else (obj.get("csp") if isinstance(obj, dict) and "csp" in obj else obj.get("CSP"))
        if csp is None or not hasattr(csp, "patterns_"):
            raise RuntimeError(f"Could not find CSP patterns_ for {sid} in loaded object.")
        csp_by_sub[sid] = csp

    # -------------------------
    # Compute mean/std per subject (across experts' selected CSPs)
    # -------------------------
    mean_by_sub, std_by_sub = {}, {}

    for col, sid in zip(cols, subj_ids):
        # map experts' choices A..F -> 0..5
        idx = []
        for x in df_choices[col].astype(str).values:
            if x not in letter_to_idx:
                continue  # ignore invalid/missing
            idx.append(letter_to_idx[x])
        idx = np.asarray(idx, dtype=int)

        if idx.size == 0:
            mean_by_sub[sid] = np.full(len(ch_names), np.nan, dtype=float)
            std_by_sub[sid] = np.full(len(ch_names), np.nan, dtype=float)
            continue

        topos = csp_by_sub[sid].patterns_[idx, :].copy()

        # normalize each topo
        topos /= (np.linalg.norm(topos, axis=1, keepdims=True) + 1e-12)

        # sign convention: enforce C3 negative
        topos[topos[:, c3_idx] > 0] *= -1

        mean_by_sub[sid] = np.nanmean(topos, axis=0)
        std_by_sub[sid] = np.nanstd(topos, axis=0)

    # -------------------------
    # Compute mean/std per expert (across subjects)
    # -------------------------
    mean_by_exp, std_by_exp = {}, {}
    n_experts = df_choices.shape[0]

    for e in range(n_experts):
        exp_id = f"expert-{e+1:02d}"
        exp_topos = []

        for j, sid in enumerate(subj_ids):
            choice = str(df_choices.iat[e, j])
            if choice not in letter_to_idx:
                continue

            csp_idx = letter_to_idx[choice]
            topo = csp_by_sub[sid].patterns_[csp_idx, :].copy()

            topo /= (np.linalg.norm(topo) + 1e-12)
            if topo[c3_idx] > 0:
                topo *= -1

            exp_topos.append(topo)

        if len(exp_topos) == 0:
            mean_by_exp[exp_id] = np.full(len(ch_names), np.nan, dtype=float)
            std_by_exp[exp_id] = np.full(len(ch_names), np.nan, dtype=float)
            continue

        exp_topos = np.asarray(exp_topos, dtype=float)
        mean_by_exp[exp_id] = np.nanmean(exp_topos, axis=0)
        std_by_exp[exp_id] = np.nanstd(exp_topos, axis=0)

    # =========================================================
    # ELECTRODE-WISE PRINTS (MEDIAN + IQR)
    # =========================================================
    def iqr(x):
        return np.nanpercentile(x, 75) - np.nanpercentile(x, 25)

    print("\n======================================")
    print("CSP VARIABILITY (STATS ONLY)")
    print("======================================")
    print(f"Subjects: {len(subj_ids)} | Experts: {n_experts} | Channels: {len(ch_names)}")
    print(f"Top-N electrodes: {TOP_N}")

    # -------- Subject-level: per subject top electrodes
    print("\n======================================")
    print("ELECTRODE SD — SUBJECT LEVEL (TOP)")
    print("======================================")

    sub_sd_mat = np.vstack([std_by_sub[sid] for sid in subj_ids])  # (n_subjects, n_channels)

    for sid in subj_ids:
        sd_topo = std_by_sub[sid]
        if np.all(np.isnan(sd_topo)):
            print(f"\n{sid} | SD unavailable (no valid choices).")
            continue
        order = np.argsort(sd_topo)[::-1]
        print(f"\n{sid} | Top-{TOP_N} electrodes by SD")
        for k in order[:TOP_N]:
            print(f"  {ch_names[k]:<4} | SD = {sd_topo[k]:.4f}")

    # -------- Subject-level: global per electrode median + IQR across subjects
    med_sd = np.nanmedian(sub_sd_mat, axis=0)
    iqr_sd = np.apply_along_axis(iqr, 0, sub_sd_mat)

    order_global = np.argsort(med_sd)[::-1]
    print("\n--- SUBJECT LEVEL SUMMARY (PER ELECTRODE: MEDIAN ± IQR across subjects) ---")
    print(f"Top-{TOP_N} electrodes by MEDIAN SD:")
    for k in order_global[:TOP_N]:
        print(f"  {ch_names[k]:<4} | median SD = {med_sd[k]:.4f} | IQR = {iqr_sd[k]:.4f}")

    order_low = np.argsort(med_sd)
    print(f"\nBottom-{TOP_N} electrodes by MEDIAN SD (most stable):")
    for k in order_low[:TOP_N]:
        print(f"  {ch_names[k]:<4} | median SD = {med_sd[k]:.4f} | IQR = {iqr_sd[k]:.4f}")

    # -------- Expert-level
    print("\n======================================")
    print("ELECTRODE SD — EXPERT LEVEL (TOP)")
    print("======================================")

    exp_ids = list(std_by_exp.keys())
    exp_sd_mat = np.vstack([std_by_exp[eid] for eid in exp_ids])  # (n_experts, n_channels)

    for eid in exp_ids:
        sd_topo = std_by_exp[eid]
        if np.all(np.isnan(sd_topo)):
            print(f"\n{eid} | SD unavailable (no valid choices).")
            continue
        order = np.argsort(sd_topo)[::-1]
        print(f"\n{eid} | Top-{TOP_N} electrodes by SD")
        for k in order[:TOP_N]:
            print(f"  {ch_names[k]:<4} | SD = {sd_topo[k]:.4f}")

    med_sd_exp = np.nanmedian(exp_sd_mat, axis=0)
    iqr_sd_exp = np.apply_along_axis(iqr, 0, exp_sd_mat)

    order_global_exp = np.argsort(med_sd_exp)[::-1]
    print("\n--- EXPERT LEVEL SUMMARY (PER ELECTRODE: MEDIAN ± IQR across experts) ---")
    print(f"Top-{TOP_N} electrodes by MEDIAN SD:")
    for k in order_global_exp[:TOP_N]:
        print(f"  {ch_names[k]:<4} | median SD = {med_sd_exp[k]:.4f} | IQR = {iqr_sd_exp[k]:.4f}")

    # =========================================================
    # CHANNEL-WISE SUMMARY: Fp1, Fp2 vs C3 (MEAN ± SD across subjects)
    # =========================================================
    targets = ["Fp1", "Fp2", "C3"]
    present = [ch for ch in targets if ch in ch_names]
    missing = [ch for ch in targets if ch not in ch_names]
    if missing:
        print(f"\n[WARNING] Missing channels: {missing}")

    idx_map = {ch: ch_names.index(ch) for ch in present}

    vals_sub = {
        ch: np.array([std_by_sub[sid][idx_map[ch]] for sid in subj_ids], dtype=float)
        for ch in present
    }

    print("\n--- SUBJECT LEVEL | Inter-expert SD at channels (mean ± SD across subjects) ---")
    for ch in present:
        x = vals_sub[ch]
        print(f"{ch:<3}: mean = {np.nanmean(x):.4f} | SD = {np.nanstd(x, ddof=1):.4f} (n={np.sum(~np.isnan(x))})")

    if "Fp1" in present and "Fp2" in present:
        frontal_mean = 0.5 * (vals_sub["Fp1"] + vals_sub["Fp2"])
        print(
            f"\nFp( mean of Fp1&Fp2 ): mean = {np.nanmean(frontal_mean):.4f} | "
            f"SD = {np.nanstd(frontal_mean, ddof=1):.4f} (n={np.sum(~np.isnan(frontal_mean))})"
        )

    if "C3" in present and "Fp1" in present and "Fp2" in present:
        diff_fp1 = vals_sub["Fp1"] - vals_sub["C3"]
        diff_fp2 = vals_sub["Fp2"] - vals_sub["C3"]
        diff_fp = frontal_mean - vals_sub["C3"]
        print("\nDifferences (Fp - C3) | mean ± SD across subjects")
        print(f"Fp1-C3: mean = {np.nanmean(diff_fp1):.4f} | SD = {np.nanstd(diff_fp1, ddof=1):.4f}")
        print(f"Fp2-C3: mean = {np.nanmean(diff_fp2):.4f} | SD = {np.nanstd(diff_fp2, ddof=1):.4f}")
        print(f"Fp -C3: mean = {np.nanmean(diff_fp):.4f} | SD = {np.nanstd(diff_fp, ddof=1):.4f}")

    # =========================================================
    # CHANNEL-WISE SUMMARY: Fp1, Fp2 vs C3 (MEAN ± SD across experts)
    # =========================================================
    vals_exp = {
        ch: np.array([std_by_exp[eid][idx_map[ch]] for eid in exp_ids], dtype=float)
        for ch in present
    }

    print("\n--- EXPERT LEVEL | Across-subject SD at channels (mean ± SD across experts) ---")
    for ch in present:
        x = vals_exp[ch]
        print(f"{ch:<3}: mean = {np.nanmean(x):.4f} | SD = {np.nanstd(x, ddof=1):.4f} (n={np.sum(~np.isnan(x))})")