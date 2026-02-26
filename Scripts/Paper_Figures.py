# -*- coding: utf-8 -*-
"""
Paper Figures — figure generation only (no stats, no prints except saved paths).

Creates and saves figures into:
    <project_root>/Figures/

Assumption:
- This script is located in <project_root>/Scripts/ (or one level under project root).
  project_root is inferred via __file__.

Figures included:
1) Final summary (Agreement + Confidence) for:
   - Background (BCI strict: BCI vs NON BCI)
   - Experience (<5 vs >=5)
   - CSP Familiarity (3 groups)
2) Figure 1 (paper): Heatmap choices (A) + Agreement/Entropy (B), large
3) Grid selected: mean & SD topomaps for 2 subjects + 2 experts

Copyright (c) 2026 Paris Brain Institute.
Author: Cassandra Dumas
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

import mne
import joblib

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
from matplotlib.lines import Line2D
from scipy.stats import entropy


# ==========================================================
# Project paths
# ==========================================================
def _project_root_from_file():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(base_dir)


def _paper_figures_dir():
    root = _project_root_from_file()
    out_dir = os.path.join(root, "Figures")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ==========================================================
# Shared helpers
# ==========================================================
SUBJECT_ORDER = [3, 18, 7, 22, 12, 5, 24, 11, 20, 6, 16, 14, 23, 8, 19, 2, 17, 15, 13, 21]
CSP_CHOICES_LETTERS = ["A", "B", "C", "D", "E", "F"]

CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'C3', 'Cz', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2'] 

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

csp_root = r"C:\Users\cassandra.dumas\OneDrive - ICM\Documents\PHD\CSP_Neurofeedback\BETAPARK\Data\_CSP_"


def _load_questionnaire_aligned(csv_name="CSP_Selection_Answers.csv"):
    data = pd.read_csv(os.path.join(project_root, "Data", csv_name))
    subj = list(SUBJECT_ORDER)

    conf_cols = [c for c in data.columns if "confident" in c.lower()]
    choice_cols = [c for c in data.columns if "[Choice]" in c]

    background_col = next(c for c in data.columns if "describe" in c.lower())
    exp_col = next(c for c in data.columns if "experience" in c.lower())
    fam_col = next(c for c in data.columns if "familiar" in c.lower())

    n = min(len(subj), len(conf_cols), len(choice_cols))
    if n == 0:
        raise RuntimeError("No aligned columns found (confident / [Choice]) or SUBJECT_ORDER empty.")

    df_conf = pd.DataFrame({f"Subject {subj[i]}": data[conf_cols[i]].values for i in range(n)})
    df_ch = pd.DataFrame({f"Subject {subj[i]}": data[choice_cols[i]].astype(str).values for i in range(n)})

    cols = sorted(df_ch.columns, key=lambda s: int(s.split()[1]))
    df_conf, df_ch = df_conf[cols], df_ch[cols]

    return data, df_conf, df_ch, background_col, exp_col, fam_col


def _expert_metrics(df_conf: pd.DataFrame, df_ch: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=df_conf.index)
    df["Mean_Confidence"] = df_conf.mean(axis=1)
    consensus = df_ch.mode(axis=0).iloc[0]
    df["Individual_Agreement"] = (df_ch == consensus).mean(axis=1)
    return df

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

# ==========================================================
# FIGURE: Final Summary (Agreement + Confidence)
# ==========================================================
def make_final_confidence_agreement_summary_figure(
    save_name="final_summary_agreement_confidence_strict",
    background_mode="strict",
    p_conf_bg=0.009,
    p_ag_bg=0.022,
    csv_name="CSP_Selection_Answers.csv",
):
    """
    background_mode:
      - "strict": BCI only if "BCI" or "brain-computer interface"
      - "broad": + neurofeedback/NF
    p_conf_bg / p_ag_bg:
      p-values to annotate on Background comparison only (straight line + stars)
    """
    out_dir = _paper_figures_dir()

    data, df_conf, df_ch, background_col, exp_col, fam_col = _load_questionnaire_aligned(csv_name)
    base = _expert_metrics(df_conf, df_ch)

    df = pd.DataFrame({
        "Background_raw": data[background_col].astype(str).str.strip(),
        "Experience_raw": data[exp_col].astype(str).str.strip(),
        "Familiarity_raw": data[fam_col].astype(str).str.strip(),
        "Mean_Confidence": base["Mean_Confidence"].values,
        "Individual_Agreement": base["Individual_Agreement"].values,
    })

    # ---- Background (2 groups)
    bg = df["Background_raw"].astype(str).str.lower()
    if background_mode.lower().strip() == "broad":
        is_bci = (
            bg.str.contains(r"\bbci\b", regex=True, na=False)
            | bg.str.contains(r"brain[-\s]?computer[-\s]?interface", regex=True, na=False)
            | bg.str.contains(r"neuro[-\s]?feedback", regex=True, na=False)
            | bg.str.contains(r"\bnf\b", regex=True, na=False)
        )
    else:
        is_bci = (
            bg.str.contains(r"\bbci\b", regex=True, na=False)
            | bg.str.contains(r"brain[-\s]?computer[-\s]?interface", regex=True, na=False)
        )
    df["Background_2"] = np.where(is_bci, "BCI", "NON BCI")
    bg_groups = ["BCI", "NON BCI"]

    # ---- Experience (2 groups)
    df["Experience_2"] = np.where(df["Experience_raw"] == "< 5 years", "< 5 years", ">= 5 years")
    exp_groups = ["< 5 years", ">= 5 years"]

    # ---- Familiarity (3 groups)
    fam_no = "No familiarity"
    fam_lim = "Limited familiarity (general knowledge or occasional use)"
    fam_mod = "Moderate familiarity (regular use or solid theoretical understanding)"
    fam_high = "High familiarity (extensive use in research or applied BCI/neurofeedback)"
    fam3_merge = "Moderate + High familiarity"
    df["Fam3"] = df["Familiarity_raw"].replace({fam_mod: fam3_merge, fam_high: fam3_merge})
    fam_groups = [fam_no, fam_lim, fam3_merge]

    # ---- Layout positions
    x_bg = [1, 2]
    x_exp = [3, 4]
    x_fam = [5, 6, 7]
    all_positions = x_bg + x_exp + x_fam
    separators = [2.5, 4.5]
    xlabels = ["BCI", "NON BCI", "< 5 years", "≥ 5 years", "None", "Limited", "Moderate/High"]

    agree_sets = (
        [df.loc[df["Background_2"] == g, "Individual_Agreement"].dropna().to_numpy(float) for g in bg_groups]
        + [df.loc[df["Experience_2"] == g, "Individual_Agreement"].dropna().to_numpy(float) for g in exp_groups]
        + [df.loc[df["Fam3"] == g, "Individual_Agreement"].dropna().to_numpy(float) for g in fam_groups]
    )
    conf_sets = (
        [df.loc[df["Background_2"] == g, "Mean_Confidence"].dropna().to_numpy(float) for g in bg_groups]
        + [df.loc[df["Experience_2"] == g, "Mean_Confidence"].dropna().to_numpy(float) for g in exp_groups]
        + [df.loc[df["Fam3"] == g, "Mean_Confidence"].dropna().to_numpy(float) for g in fam_groups]
    )

    empty_agree = [len(v) == 0 for v in agree_sets]
    empty_conf = [len(v) == 0 for v in conf_sets]
    agree_sets_safe = [v if len(v) > 0 else np.array([0.0], dtype=float) for v in agree_sets]
    conf_sets_safe = [v if len(v) > 0 else np.array([np.nan], dtype=float) for v in conf_sets]

    rng = np.random.default_rng(42)
    cmap_bg = cm.get_cmap("Greens")
    cmap_exp = cm.get_cmap("Oranges")
    cmap_fam = cm.get_cmap("Blues")

    pal_bg = [cmap_bg(v) for v in np.linspace(0.45, 0.75, 2)]
    pal_exp = [cmap_exp(v) for v in np.linspace(0.45, 0.75, 2)]
    pal_fam = [cmap_fam(v) for v in np.linspace(0.45, 0.80, 3)]
    pal_all = pal_bg + pal_exp + pal_fam

    fig = plt.figure(figsize=(12.0, 6.0), dpi=150)
    ax_ag = fig.add_subplot(2, 1, 1)
    ax_cf = fig.add_subplot(2, 1, 2)

    x_left, x_right = 0.55, 7.45
    for ax in (ax_ag, ax_cf):
        ax.set_xlim(x_left, x_right)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for s in separators:
            ax.axvline(s, linestyle=":", linewidth=1.2, color="black", alpha=0.7)

    # Agreement: violin + jitter
    vp = ax_ag.violinplot(
        agree_sets_safe,
        positions=all_positions,
        showmedians=True,
        showextrema=False,
        widths=0.85,
    )
    for i, body in enumerate(vp["bodies"]):
        body.set(facecolor=pal_all[i], edgecolor="black", alpha=0.60, linewidth=0.8)
        if empty_agree[i]:
            body.set(alpha=0.0)
    if "cmedians" in vp:
        vp["cmedians"].set(color="black", linewidth=1.2)

    for i, x in enumerate(all_positions):
        if empty_agree[i]:
            continue
        y = agree_sets[i]
        ax_ag.scatter(
            x + rng.uniform(-0.10, 0.10, len(y)),
            y,
            s=20,
            facecolors="white",
            edgecolors="black",
            linewidths=0.7,
            zorder=3,
        )

    ax_ag.set(ylabel="Individual agreement", ylim=(0, 1), yticks=np.linspace(0, 1, 6))
    ax_ag.set_xticks(all_positions)
    ax_ag.set_xticklabels(xlabels, fontsize=10)
    ax_ag.tick_params(axis="x", labelbottom=True)

    # Confidence: boxplot + jitter
    bp = ax_cf.boxplot(
        conf_sets_safe,
        positions=all_positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set(facecolor=pal_all[i], edgecolor="black", alpha=0.75, linewidth=0.9)
        if empty_conf[i]:
            patch.set(alpha=0.0)
    for k in ("whiskers", "caps", "medians"):
        for artist in bp[k]:
            artist.set(color="black", linewidth=1.0)

    for i, x in enumerate(all_positions):
        if empty_conf[i]:
            continue
        y = conf_sets[i]
        ax_cf.scatter(
            x + rng.uniform(-0.12, 0.12, len(y)),
            y,
            s=20,
            facecolors="white",
            edgecolors="black",
            linewidths=0.7,
            zorder=3,
        )

    ax_cf.set(ylabel="Mean confidence", ylim=(1, 4), yticks=[1, 2, 3, 4])
    ax_cf.set_xticks(all_positions)
    ax_cf.set_xticklabels(xlabels, fontsize=10)

    # Block labels below x-axis on bottom row
    label_y = -0.12
    ax_cf.text(1.5, label_y, "Background", ha="center", va="top",
               fontsize=11, transform=ax_cf.get_xaxis_transform())
    ax_cf.text(3.5, label_y, "Experience", ha="center", va="top",
               fontsize=11, transform=ax_cf.get_xaxis_transform())
    ax_cf.text(6.0, label_y, "CSP Familiarity", ha="center", va="top",
               fontsize=11, transform=ax_cf.get_xaxis_transform())
    fig.subplots_adjust(bottom=0.18)

    # Significance on Background only (straight line + stars)
    def _stars(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    star_conf = _stars(p_conf_bg)
    star_ag = _stars(p_ag_bg)

    y_line_ag = 0.96
    ax_ag.plot([1, 2], [y_line_ag, y_line_ag], lw=1.5, c="black")
    ax_ag.text(1.5, y_line_ag + 0.02, star_ag, ha="center", va="bottom", fontsize=12)

    y_line_cf = 3.85
    ax_cf.plot([1, 2], [y_line_cf, y_line_cf], lw=1.5, c="black")
    ax_cf.text(1.5, y_line_cf + 0.05, star_conf, ha="center", va="bottom", fontsize=12)

    # Panel labels
    panel_x = -0.052
    panel_y = 1.09
    ax_ag.text(panel_x, panel_y, "A", transform=ax_ag.transAxes,
               fontsize=16, fontweight="bold", ha="left", va="top")
    ax_cf.text(panel_x, panel_y, "B", transform=ax_cf.transAxes,
               fontsize=16, fontweight="bold", ha="left", va="top")

    mode = background_mode.lower().strip()
    out_path = os.path.join(out_dir, f"{save_name}_{mode}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_path


# ==========================================================
# FIGURE 1: Heatmap choices + Agreement/Entropy (Large)
# ==========================================================
def make_figure1_heatmap_and_agreement_entropy_large(
    save_name="Figure1_LARGE_CSPChoices_AgreementEntropy",
    csv_name="CSP_Selection_Answers.csv",
):
    out_dir = _paper_figures_dir()

    data = pd.read_csv(os.path.join(project_root, "Data", csv_name))
    subj = list(SUBJECT_ORDER)
    choice_cols = [c for c in data.columns if "[Choice]" in c]
    background_col = next(c for c in data.columns if "describe" in c.lower())

    n = min(len(subj), len(choice_cols))
    df_choice = pd.DataFrame(
        {f"Subject {subj[i]}": data[choice_cols[i]].astype(str).values for i in range(n)}
    ).reindex(
        sorted([f"Subject {s}" for s in subj[:n]], key=lambda x: int(x.split()[1])),
        axis=1,
    )

    letters = list("ABCDEF")
    df_choice = df_choice.where(df_choice.isin(letters), np.nan).replace({k: i for i, k in enumerate(letters)})

    # Subject order by consensus strength (max proportion), tie-break: valid desc, sid asc
    valid = df_choice.notna().sum(axis=0).astype(int)
    max_count = df_choice.apply(
        lambda col: col.value_counts(dropna=True).max() if col.notna().any() else 0, axis=0
    ).astype(int)
    max_prop = (max_count / valid.replace(0, np.nan)).fillna(0.0).round(6)
    subj_id = pd.Index(df_choice.columns).to_series().apply(lambda s: int(str(s).split()[1]))

    order_subjects = (
        pd.DataFrame({"max_prop": max_prop, "valid": valid, "sid": subj_id.astype(int)}, index=df_choice.columns)
        .sort_values(["max_prop", "valid", "sid"], ascending=[False, False, True])
        .index.tolist()
    )
    df_choice = df_choice[order_subjects]

    # Expert sorting: STRICT BCI vs NON BCI
    bg_raw = data[background_col].astype(str).str.strip().str.lower().iloc[: df_choice.shape[0]]
    is_bci = (
        bg_raw.str.contains(r"\bbci\b", regex=True, na=False)
        | bg_raw.str.contains(r"brain[-\s]?computer[-\s]?interface", regex=True, na=False)
    )
    bg_group = np.where(is_bci.values, "BCI", "NON BCI")
    order_experts_bg = (
        list(pd.Index(df_choice.index)[np.where(bg_group == "BCI")[0]])
        + list(pd.Index(df_choice.index)[np.where(bg_group == "NON BCI")[0]])
    )
    df_choice = df_choice.loc[order_experts_bg]

    n_bci = int((bg_group == "BCI").sum())
    bg_bounds = [0, n_bci, df_choice.shape[0]]
    bg_labels = ["BCI Experts", "NON BCI Experts"]

    # Agreement + entropy (same order)
    inv = {i: L for i, L in enumerate(letters)}
    df_choices_letters = df_choice.replace(inv)

    probs = df_choices_letters.apply(lambda c: c.value_counts(normalize=True), axis=0).fillna(0)
    probs = probs.reindex(index=letters, fill_value=0)

    percent_agreement = probs.max(axis=0)
    entropy_bits = probs.apply(lambda p: entropy(p, base=2), axis=0)
    entropy_norm = entropy_bits / np.log2(len(letters)) if len(letters) > 1 else entropy_bits * 0.0

    # Colormap: choices A–F (Blues)
    base_cmap = plt.get_cmap("Blues")
    colors_choice = base_cmap(np.linspace(0.35, 0.95, 6))
    cmap_choice = ListedColormap(colors_choice)
    norm_choice = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap_choice.N)

    n_exp, n_sub = df_choice.shape

    fig = plt.figure(figsize=(16, 5), dpi=300)
    fig.subplots_adjust(bottom=0.15, top=0.92, left=0.05, right=0.95)

    gs_main = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.07)
    gs_left = gs_main[0].subgridspec(1, 2, width_ratios=[1.0, 0.25], wspace=0.01)

    axA = fig.add_subplot(gs_left[0, 0])
    axLegA = fig.add_subplot(gs_left[0, 1])
    axB = fig.add_subplot(gs_main[0, 1])
    axLegA.axis("off")

    # Panel A: Heatmap
    axA.imshow(df_choice.values, cmap=cmap_choice, norm=norm_choice, interpolation="nearest", aspect="auto")
    axA.set_title("A", loc="left", fontweight="bold", fontsize=20, x=-0.14, y=1.0012)
    axA.set_xlabel("Subjects", fontsize=15, labelpad=10)

    axA.set_xticks(np.arange(n_sub))
    axA.set_yticks(np.arange(n_exp))
    axA.set_xticklabels([f"S{int(s.split()[1]):02d}" for s in df_choice.columns], rotation=90, fontsize=11)
    axA.set_yticklabels([f"E{int(i)+1:02d}" for i in df_choice.index], fontsize=11)

    for sp in axA.spines.values():
        sp.set_visible(False)

    axA.set_xticks(np.arange(-0.5, n_sub, 1), minor=True)
    axA.set_yticks(np.arange(-0.5, n_exp, 1), minor=True)
    axA.grid(which="minor", color=(1, 1, 1, 0.25), linewidth=0.6)
    axA.tick_params(which="minor", bottom=False, left=False)
    axA.tick_params(axis="y", pad=2)

    # Separator between BCI and NON BCI
    if 0 < bg_bounds[1] < n_exp:
        axA.hlines(bg_bounds[1] - 0.5, -0.5, n_sub - 0.5, colors=(1, 1, 1, 0.55), linewidth=1.2)

    # Bracket outside
    trans = blended_transform_factory(axA.transAxes, axA.transData)
    brace_x = -0.09
    brace_cap = 0.06

    for gi in range(2):
        y0 = bg_bounds[gi]
        y1 = bg_bounds[gi + 1]
        if y1 <= y0:
            continue
        axA.plot([brace_x, brace_x], [y0 - 0.5, y1 - 0.5], transform=trans, color="grey", lw=1.6, clip_on=False)
        axA.plot([brace_x, brace_x + brace_cap], [y0 - 0.5, y0 - 0.5], transform=trans, color="grey", lw=1.6, clip_on=False)
        axA.plot([brace_x, brace_x + brace_cap], [y1 - 0.5, y1 - 0.5], transform=trans, color="grey", lw=1.6, clip_on=False)
        axA.text(
            brace_x - 0.04,
            (y0 + y1 - 1) / 2,
            bg_labels[gi],
            transform=trans,
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
        )

    # Legend A
    axLegA.legend(
        handles=[Patch(facecolor=colors_choice[i], edgecolor="none", label=letters[i]) for i in range(6)],
        title="Choice",
        frameon=False,
        loc="center left",
        fontsize=12,
        title_fontsize=13,
        handlelength=1.2,
        labelspacing=0.8,
    )

    # Panel B: Agreement + Entropy
    order = df_choice.columns.tolist()
    x = np.arange(len(order))
    pa = percent_agreement.loc[order].values
    en = entropy_norm.loc[order].values

    axB.set_title("B", loc="left", fontweight="bold", fontsize=20, x=-0.085, y=1.0012)
    axB.spines["top"].set_visible(False)
    axB.grid(True, axis="y", alpha=0.25)
    axB.set_axisbelow(True)

    axB.plot(
        x, pa,
        color="black", linestyle="-", linewidth=2.5,
        marker="o", markersize=5, markerfacecolor="white", markeredgewidth=1.2,
        label="Agreement",
    )

    axB.set_ylabel("Agreement", fontsize=15)
    axB.set_ylim(0, 1)
    axB.set_yticks(np.linspace(0, 1, 6))
    axB.tick_params(axis="y", labelsize=12)

    axB.set_xticks(x)
    axB.set_xticklabels([f"S{int(s.split()[1]):02d}" for s in order], rotation=90, ha="center", fontsize=12)
    axB.set_xlabel("Subjects", fontsize=15, labelpad=9)

    axB2 = axB.twinx()
    axB2.spines["top"].set_visible(False)
    axB2.plot(
        x, en,
        color="black", linestyle="--", linewidth=2.5,
        marker="s", markersize=5, markerfacecolor="white", markeredgewidth=1.2,
        label="Entropy",
    )
    axB2.set_ylabel("Entropy", fontsize=15)
    axB2.set_ylim(0, 1)
    axB2.set_yticks(np.linspace(0, 1, 6))
    axB2.spines["right"].set_linestyle("--")
    axB2.tick_params(axis="y", labelsize=12)

    legend_handles = [
        Line2D([], [], color="black", linestyle="-", linewidth=2.5,
               marker="o", markersize=6, markerfacecolor="white",
               markeredgecolor="black", label="Agreement"),
        Line2D([], [], color="black", linestyle="--", linewidth=2.5,
               marker="s", markersize=6, markerfacecolor="white",
               markeredgecolor="black", markeredgewidth=2, label="Entropy"),
    ]
    axB.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.75, 0.20),
        bbox_transform=axB.transAxes,
        frameon=False,
        fontsize=12,
        handlelength=3.2,
    )

    out_png = os.path.join(out_dir, f"{save_name}.png")
    fig.savefig(out_png, dpi=300)  # keep layout stable
    plt.close(fig)

    return out_png


# ==========================================================
# FIGURE: Grid selected (2 subjects + 2 experts)
# ==========================================================
def make_grid_selected_csp_mean_sd(
    subjects=(3, 21),
    experts=(10, 21),
    save_name=None,
    csv_name="CSP_Selection_Answers.csv",
):
    out_dir = _paper_figures_dir()

    if save_name is None:
        save_name = f"GroupedGrid_Sub{subjects[0]}_Sub{subjects[1]}_Exp{experts[0]}_Exp{experts[1]}"

    csv_path = os.path.join(project_root, "Data", csv_name)
    csp_path = os.path.join(csp_root, "_HAND_CALIBRATION_", "mutual_info")

    letter_to_idx = {k: i for i, k in enumerate("ABCDEF")}
    cmap_csp = discrete_cmap(10, "PiYG")
    cmap_sd = discrete_cmap(5, "Greys")

    # Load questionnaire
    data = pd.read_csv(csv_path)
    choice_cols = [c for c in data.columns if "[Choice]" in c]
    subj_nums = list(SUBJECT_ORDER)

    df_choices = pd.DataFrame({
        f"Subject {subj_nums[i]}": data[c].astype(str).values
        for i, c in enumerate(choice_cols[:len(subj_nums)])
    }).reindex(
        sorted([f"Subject {s}" for s in subj_nums], key=lambda x: int(x.split()[1])),
        axis=1,
    )

    cols = df_choices.columns.tolist()
    subj_ids = [f"sub-S{int(c.split()[1]):03d}" for c in cols]

    # MNE info
    info = mne.create_info(
        CHANNELS,
        sfreq=1000.0,
        ch_types=["eeg"] * len(CHANNELS),
    )
    info.set_montage('easycap-M1', on_missing="ignore")
    if "C3" not in CHANNELS:
        raise RuntimeError("Channel C3 not found in Channels.BETAPARK_STUDY.")
    c3_idx = CHANNELS.index("C3")

    # Load CSP objects
    csp_by_sub = {}
    for sid in subj_ids:
        obj = joblib.load(os.path.join(csp_path, f"CSP_{sid}.joblib"))
        csp_by_sub[sid] = obj if hasattr(obj, "patterns_") else (obj["csp"] if "csp" in obj else obj["CSP"])

    # Compute mean/std per subject
    mean_by_sub, std_by_sub = {}, {}
    for col, sid in zip(cols, subj_ids):
        idx = np.fromiter((letter_to_idx[x] for x in df_choices[col].values), dtype=int)
        topos = csp_by_sub[sid].patterns_[idx, :].copy()
        topos /= (np.linalg.norm(topos, axis=1, keepdims=True) + 1e-12)
        topos[topos[:, c3_idx] > 0] *= -1
        mean_by_sub[sid] = topos.mean(0)
        std_by_sub[sid] = topos.std(0)

    # Compute mean/std per expert
    mean_by_exp, std_by_exp = {}, {}
    for e in range(df_choices.shape[0]):
        exp_id = f"expert-{e+1:02d}"
        exp_topos = []
        for j, sid in enumerate(subj_ids):
            csp_idx = letter_to_idx[df_choices.iat[e, j]]
            topo = csp_by_sub[sid].patterns_[csp_idx, :].copy()
            topo /= (np.linalg.norm(topo) + 1e-12)
            if topo[c3_idx] > 0:
                topo *= -1
            exp_topos.append(topo)
        exp_topos = np.asarray(exp_topos)
        mean_by_exp[exp_id] = exp_topos.mean(0)
        std_by_exp[exp_id] = exp_topos.std(0)

    # Select keys
    sub_keys = [f"sub-S{s:03d}" for s in subjects]
    exp_keys = [f"expert-{e:02d}" for e in experts]

    # Figure layout (kept as your compact design)
    fig = plt.figure(figsize=(3.5, 2.5))

    gs = gridspec.GridSpec(
        7, 5,
        width_ratios=[1, 0.04, 0.3, 1, 0.04],
        height_ratios=[0.4, 1, 1, 0.1, 0.4, 1, 1],
        left=0.02,
        right=0.88,
        top=1.0,
        bottom=0.02,
        wspace=0,
        hspace=0.05,
    )

    axes = {}
    for i in range(7):
        axes[(i, 0)] = fig.add_subplot(gs[i, 0])  # Mean
        axes[(i, 3)] = fig.add_subplot(gs[i, 3])  # SD
        for col in [0, 3]:
            axes[(i, col)].set_aspect("equal")
            axes[(i, col)].set_xticks([])
            axes[(i, col)].set_yticks([])
            axes[(i, col)].set_frame_on(False)

    # shift SD column left (as in your code)
    for i in range(7):
        ax = axes[(i, 3)]
        pos = ax.get_position()
        shift = -0.085
        ax.set_position([pos.x0 + shift, pos.y0, pos.width, pos.height])

    fig.text(0.05, 0.95, "A) Subjects", fontsize=6, fontweight="bold")
    fig.text(0.05, 0.45, "B) Experts", fontsize=6, fontweight="bold")
    fig.text(0.07, 0.87, f"S{subjects[0]:02d}", fontsize=5)
    fig.text(0.07, 0.67, f"S{subjects[1]:02d}", fontsize=5)
    fig.text(0.07, 0.37, f"E{experts[0]:02d}", fontsize=5)
    fig.text(0.07, 0.17, f"E{experts[1]:02d}", fontsize=5)

    row_mapping = {0: 1, 1: 2, 2: 5, 3: 6}
    row_items = [sub_keys[0], sub_keys[1], exp_keys[0], exp_keys[1]]

    mean_subject_img = None
    sd_subject_img = None
    mean_expert_img = None
    sd_expert_img = None

    # NOTE: vlim are fixed here (as your code) for consistent visual comparison
    for idx, key in enumerate(row_items):
        r = row_mapping[idx]

        if key.startswith("sub-"):
            mean_topo = mean_by_sub[key]
            std_topo = std_by_sub[key]
        else:
            mean_topo = mean_by_exp[key]
            std_topo = std_by_exp[key]

        im_mean, _ = mne.viz.plot_topomap(
            mean_topo, info,
            axes=axes[(r, 0)],
            cmap=cmap_csp,
            vlim=(-0.5, 0.5),
            show=False,
            contours=0,
        )

        im_sd, _ = mne.viz.plot_topomap(
            std_topo, info,
            axes=axes[(r, 3)],
            cmap=cmap_sd,
            vlim=(0, 0.4),
            show=False,
            contours=0,
        )

        if idx < 2:
            mean_subject_img = im_mean
            sd_subject_img = im_sd
        else:
            mean_expert_img = im_mean
            sd_expert_img = im_sd

    # Colorbars
    for gs_loc, img in [
        (gs[1:3, 1], mean_subject_img),
        (gs[1:3, 4], sd_subject_img),
        (gs[5:7, 1], mean_expert_img),
        (gs[5:7, 4], sd_expert_img),
    ]:
        pos = gs_loc.get_position(fig)
        center_y = pos.y0 + pos.height / 2
        new_height = pos.height * 0.7
        new_y0 = center_y - new_height / 2

        if pos.x0 > 0.4:  # SD bars (right)
            shift = -0.15
            label = "SD"
        else:
            shift = -0.075
            label = "Mean"

        cax_new = fig.add_axes([pos.x0 + shift, new_y0, pos.width, new_height])
        cb = fig.colorbar(img, cax=cax_new)
        cb.ax.tick_params(labelsize=5, pad=0.8)

        pos_cb = cax_new.get_position()
        x = pos_cb.x1 + (0.07 if label == "SD" else 0.09)
        y = pos_cb.y0 + pos_cb.height / 2
        fig.text(x, y, label, fontsize=7, rotation=90, va="center", ha="left")

    out_path = os.path.join(out_dir, f"{save_name}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return out_path


# ==========================================================
# One-call wrapper (optional, call it from your main)
# ==========================================================
def Make_Paper_Figures():
    """
    Call this from your main pipeline.
    Returns a dict of saved paths.
    """
    saved = {}

    saved["final_summary_strict"] = make_final_confidence_agreement_summary_figure(
        save_name="final_summary_agreement_confidence",
        background_mode="strict",
        p_conf_bg=0.009,
        p_ag_bg=0.022,
    )

    png1, pdf1 = make_figure1_heatmap_and_agreement_entropy_large(
        save_name="Figure1_LARGE_CSPChoices_AgreementEntropy"
    )
    saved["figure1_png"] = png1
    saved["figure1_pdf"] = pdf1

    saved["grid_selected"] = make_grid_selected_csp_mean_sd(
        subjects=(3, 21),
        experts=(10, 21),
        save_name="GroupedGrid_Sub03_Sub21_Exp10_Exp21",
    )

    return saved