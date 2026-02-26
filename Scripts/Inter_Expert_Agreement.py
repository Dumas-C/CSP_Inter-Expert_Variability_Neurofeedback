# -*- coding: utf-8 -*-
"""
Inter-expert agreement on CSP selection (stats-only).

- Per-subject consensus metrics (modal agreement, normalized entropy)
- Global Fleiss' kappa (+ bootstrap CI)
- Rank-based analyses (global + subject-level)
- Multiple-comparisons control (FDR-BH)

Copyright (c) 2026 Paris Brain Institute.
Author: Cassandra Dumas
"""

import os
import numpy as np
import pandas as pd

from numpy.random import default_rng
from scipy.stats import entropy, spearmanr, ttest_1samp, wilcoxon
from statsmodels.stats.inter_rater import fleiss_kappa
from statsmodels.stats.multitest import multipletests

# ==========================================================
# Helpers
# ==========================================================
SUBJECT_ORDER = [3, 18, 7, 22, 12, 5, 24, 11, 20, 6, 16, 14, 23, 8, 19, 2, 17, 15, 13, 21]
CSP_CHOICES_LETTERS = ["A", "B", "C", "D", "E", "F"]
    
def bootstrap_fleiss_kappa(counts_matrix, n_boot=10000, alpha=0.05, seed=42):
    """Bootstrap CI for Fleiss' kappa by resampling items (rows)."""
    rng = default_rng(seed)
    counts_matrix = np.asarray(counts_matrix)
    n_items = counts_matrix.shape[0]

    kappas = np.empty(n_boot, dtype=float)
    kappa_obs = float(fleiss_kappa(counts_matrix))

    for b in range(n_boot):
        idx = rng.integers(0, n_items, size=n_items)
        sample = counts_matrix[idx, :]
        kappas[b] = fleiss_kappa(sample)

    ci_low = float(np.quantile(kappas, alpha / 2))
    ci_high = float(np.quantile(kappas, 1 - alpha / 2))
    return kappa_obs, ci_low, ci_high


def descriptive_stats(series):
    """Return mean ± sd for a pandas Series."""
    series = pd.Series(series).dropna()
    return float(series.mean()), float(series.std(ddof=1))


# ==========================================================
# Main
# ==========================================================
def inter_expert_agreement_stats(boot_n=10000, boot_seed=42, alpha=0.05, fdr_alpha=0.05):
    # ------------------------------------------------------
    # Load questionnaire
    # ------------------------------------------------------
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    csv_path = os.path.join(project_root, "Data", "CSP_Selection_Answers.csv")
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

    letters = list(CSP_CHOICES_LETTERS) if CSP_CHOICES_LETTERS else list("ABCDEF")
    
    if letters is None or len(letters) == 0:
        letters = list("ABCDEF")

    df_choices = df_choices.where(df_choices.isin(letters), np.nan)

    # ------------------------------------------------------
    # Agreement metrics
    # ------------------------------------------------------
    count_table = df_choices.apply(pd.Series.value_counts).fillna(0).T
    count_table = count_table.reindex(columns=letters, fill_value=0)

    probs = df_choices.apply(lambda c: c.value_counts(normalize=True), axis=0).fillna(0)
    probs = probs.reindex(index=letters, fill_value=0)

    percent_agreement = probs.max(axis=0)
    entropy_bits = probs.apply(lambda p: entropy(p, base=2), axis=0)
    entropy_norm = entropy_bits / np.log2(len(letters))

    # ------------------------------------------------------
    # Descriptive stats
    # ------------------------------------------------------
    agree_mean, agree_sd = descriptive_stats(percent_agreement)
    ent_mean, ent_sd = descriptive_stats(entropy_norm)

    # ------------------------------------------------------
    # Fleiss' kappa + bootstrap CI
    # ------------------------------------------------------
    kappa_obs = float(fleiss_kappa(count_table.to_numpy()))
    kappa_obs, kappa_ci_low, kappa_ci_high = bootstrap_fleiss_kappa(
        count_table.to_numpy(),
        n_boot=boot_n,
        alpha=alpha,
        seed=boot_seed,
    )

    # ------------------------------------------------------
    # Global rank–selection association
    # ------------------------------------------------------
    total_counts = count_table.sum(axis=0).to_numpy(dtype=float)
    total_props = total_counts / total_counts.sum()

    ranks = np.arange(1, len(letters) + 1)
    rho_global, p_global = spearmanr(ranks, total_props)

    # ------------------------------------------------------
    # Subject-level rank analysis
    # ------------------------------------------------------
    letter_to_rank = {L: i + 1 for i, L in enumerate(letters)}
    df_ranks = df_choices.replace(letter_to_rank)

    mean_rank_per_subject = df_ranks.mean(axis=0).dropna()
    mid_rank = (len(letters) + 1) / 2

    t_stat, p_t = ttest_1samp(mean_rank_per_subject.values, mid_rank)
    w_stat, p_w = wilcoxon(mean_rank_per_subject.values - mid_rank)

    # ------------------------------------------------------
    # Rank ↔ Consensus
    # ------------------------------------------------------
    agreement_aligned = percent_agreement.loc[mean_rank_per_subject.index]
    rho_agree_rank, p_agree_rank = spearmanr(
        mean_rank_per_subject.values,
        agreement_aligned.values,
    )

    # ------------------------------------------------------
    # FDR correction
    # ------------------------------------------------------
    pvals = np.array([p_global, p_t, p_w, p_agree_rank])
    test_names = [
        "Global rank–selection (Spearman)",
        "Mean rank vs mid-rank (t-test)",
        "Mean rank vs mid-rank (Wilcoxon)",
        "Rank–consensus (Spearman)",
    ]

    reject, qvals, _, _ = multipletests(pvals, alpha=fdr_alpha, method="fdr_bh")

    # ======================================================
    # PRINTS
    # ======================================================
    print("\n==============================")
    print(" INTER-EXPERT AGREEMENT STATS")
    print("==============================")

    print("\nAgreement (modal proportion across subjects):")
    print(f"Mean ± SD = {agree_mean:.3f} ± {agree_sd:.3f}")

    print("\nNormalized entropy:")
    print(f"Mean ± SD = {ent_mean:.3f} ± {ent_sd:.3f}")

    print("\nFLEISS' KAPPA (GLOBAL):")
    print(f"kappa = {kappa_obs:.3f}")
    print(f"{int((1-alpha)*100)}% bootstrap CI = [{kappa_ci_low:.3f}, {kappa_ci_high:.3f}]")

    print("\nGLOBAL RANK–SELECTION ASSOCIATION:")
    print(f"Spearman rho = {rho_global:.3f}, p = {p_global:.4f}")

    print("\nSUBJECT-LEVEL RANK ANALYSIS:")
    print(f"Mean selected rank = {mean_rank_per_subject.mean():.3f}")
    print(f"Reference mid-rank = {mid_rank:.1f}")
    print(f"t-test: t = {t_stat:.3f}, p = {p_t:.4f}")
    print(f"Wilcoxon: W = {w_stat:.3f}, p = {p_w:.4f}")

    print("\nRANK ↔ CONSENSUS:")
    print(f"Spearman rho = {rho_agree_rank:.3f}, p = {p_agree_rank:.4f}")

    print("\nFDR-BH CORRECTION:")
    for name, pval, qval, rej in zip(test_names, pvals, qvals, reject):
        flag = " (FDR<0.05)" if rej else ""
        print(f"{name}: p={pval:.4f}, q={qval:.4f}{flag}")