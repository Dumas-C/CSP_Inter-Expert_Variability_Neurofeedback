# -*- coding: utf-8 -*-
"""
ADDITIONAL PAPER FIGURES
- CSP variability topomaps (subjects: ERD + Mean/SD, experts: Mean/SD)
- Heatmaps (choices + description) with shared subject order (consensus strength)

Saves ONLY figures into:
    <project_root>/Figures/Additionnal/

No stats prints.

Copyright (c) 2026 Paris Brain Institute.
Author: Cassandra Dumas
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne
import joblib
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory


# ==========================================================
# Shared helpers
# ==========================================================
SUBJECT_ORDER = [3, 18, 7, 22, 12, 5, 24, 11, 20, 6, 16, 14, 23, 8, 19, 2, 17, 15, 13, 21]
CSP_CHOICES_LETTERS = ["A", "B", "C", "D", "E", "F"]

CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'C3', 'Cz', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2'] 

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

csp_root = r"C:\Users\cassandra.dumas\OneDrive - ICM\Documents\PHD\CSP_Neurofeedback\BETAPARK\Data\_CSP_"
motor_physiology_root = r"C:\Users\cassandra.dumas\OneDrive - ICM\Documents\PHD\CSP_Neurofeedback\BETAPARK\Data\_MOTOR_NEUROPHYSIOLOGY_"

def _project_root_from_file():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(base_dir)


def _additional_fig_dir():
    out_dir = os.path.join(_project_root_from_file(), "Figures", "Additionnal")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

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
# 1) CSP variability topomaps (subjects + experts)
# ==========================================================
def Make_CSP_Variability_Topomaps(csv_name="CSP_Selection_Answers.csv"):
    """
    Saves:
      - per subject: ERD + Mean(local) + SD(global)
      - per expert : Mean(local) + SD(global)
    into <project_root>/Figures/Additionnal/CSP_Variability/
    """
    out_root = os.path.join(_additional_fig_dir(), "CSP_Variability")
    out_sub = os.path.join(out_root, "Subjects")
    out_exp = os.path.join(out_root, "Experts")
    os.makedirs(out_sub, exist_ok=True)
    os.makedirs(out_exp, exist_ok=True)

    # Paths / params
    csv_path = os.path.join(project_root, "Data", csv_name)
    csp_path = os.path.join(csp_root, "_HAND_CALIBRATION_", "mutual_info")
    tfr_root = os.path.join(motor_physiology_root, "_HAND_CALIBRATION_", "_TFR_")

    letter_to_idx = {k: i for i, k in enumerate("ABCDEF")}
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "figure.dpi": 300})

    cmap_csp = discrete_cmap(10, "PiYG")
    cmap_sd = discrete_cmap(5, "Greys")
    cmap_erd = discrete_cmap(13, "RdBu_r")

    erd_tmin, erd_tmax = 0, 8
    erd_fmin, erd_fmax = 8, 30

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
        csp_by_sub[sid] = obj if hasattr(obj, "patterns_") else (
            obj["csp"] if "csp" in obj else obj["CSP"]
        )

    mean_by_sub, std_by_sub = {}, {}
    mean_by_exp, std_by_exp = {}, {}

    # Per subject
    for col, sid in zip(cols, subj_ids):
        idx = np.fromiter((letter_to_idx[x] for x in df_choices[col].values), dtype=int)
        topos = csp_by_sub[sid].patterns_[idx, :].copy()

        topos /= (np.linalg.norm(topos, axis=1, keepdims=True) + 1e-12)
        topos[topos[:, c3_idx] > 0] *= -1

        mean_by_sub[sid] = topos.mean(0)
        std_by_sub[sid] = topos.std(0)

    # Per expert
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

    # Global SD scales
    global_sdmax_sub = max(np.max(std_by_sub[sid]) for sid in subj_ids)
    global_sdmax_exp = max(np.max(std_by_exp[e]) for e in std_by_exp)
    global_sdmax_sub = max(global_sdmax_sub, 1e-12)
    global_sdmax_exp = max(global_sdmax_exp, 1e-12)

    # -------------------------
    # Plot subjects
    # -------------------------
    for sid in subj_ids:
        mean_topo = mean_by_sub[sid]
        std_topo = std_by_sub[sid]
        local_vlim = max(np.max(np.abs(mean_topo)), 1e-12)

        # ERD (local)
        has_erd = True
        try:
            tfr_path = os.path.join(tfr_root, sid, f"TFR_Hand_Calibration_{sid}-tfr.h5")
            tfr_obj = mne.time_frequency.read_tfrs(tfr_path)
            tfr = tfr_obj[0] if isinstance(tfr_obj, list) else tfr_obj
            tfr = tfr.copy().crop(tmin=erd_tmin, tmax=erd_tmax, fmin=erd_fmin, fmax=erd_fmax)

            erd_topo = np.mean(tfr.data, axis=(1, 2))

            if tfr.ch_names != info["ch_names"]:
                ch_to_idx = {ch: i for i, ch in enumerate(tfr.ch_names)}
                erd_topo = np.array([erd_topo[ch_to_idx[ch]] for ch in info["ch_names"]])

            erd_vmax = max(np.max(np.abs(erd_topo)), 1e-12)
            erd_vlim = (-erd_vmax, erd_vmax)
        except Exception:
            has_erd = False

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"{sid} — Inter-expert CSP selection", y=1.02, fontweight="bold")

        # ERD
        if has_erd:
            im0, _ = mne.viz.plot_topomap(
                erd_topo, info, axes=axes[0],
                cmap=cmap_erd, vlim=erd_vlim, show=False, contours=0
            )
            axes[0].set_title(f"ERD ({erd_fmin}-{erd_fmax} Hz)")
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        else:
            axes[0].axis("off")

        # Mean (LOCAL)
        im1, _ = mne.viz.plot_topomap(
            mean_topo, info, axes=axes[1],
            cmap=cmap_csp, vlim=(-local_vlim, local_vlim),
            show=False, contours=0
        )
        axes[1].set_title("Mean CSP (local)")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # SD (GLOBAL)
        im2, _ = mne.viz.plot_topomap(
            std_topo, info, axes=axes[2],
            cmap=cmap_sd, vlim=(0, global_sdmax_sub),
            show=False, contours=0
        )
        axes[2].set_title(f"SD CSP (global max {global_sdmax_sub:.2f})")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        fig.savefig(os.path.join(out_sub, f"{sid}_ERD_MeanLocal_SDGlobal.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # -------------------------
    # Plot experts
    # -------------------------
    for exp_id in mean_by_exp:
        mean_topo = mean_by_exp[exp_id]
        std_topo = std_by_exp[exp_id]
        local_vlim = max(np.max(np.abs(mean_topo)), 1e-12)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"{exp_id} — Across-subject CSP selection", y=1.02, fontweight="bold")

        im1, _ = mne.viz.plot_topomap(
            mean_topo, info, axes=axes[0],
            cmap=cmap_csp, vlim=(-local_vlim, local_vlim),
            show=False, contours=0
        )
        axes[0].set_title("Mean CSP (local)")
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        im2, _ = mne.viz.plot_topomap(
            std_topo, info, axes=axes[1],
            cmap=cmap_sd, vlim=(0, global_sdmax_exp),
            show=False, contours=0
        )
        axes[1].set_title(f"SD CSP (global max {global_sdmax_exp:.2f})")
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        fig.savefig(os.path.join(out_exp, f"{exp_id}_MeanLocal_SDGlobal.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    return out_root


# ==========================================================
# 2) Heatmaps: choices + description, shared subject order
# ==========================================================
def Make_Heatmaps_CSP_Choices_And_Description(csv_name="CSP_Selection_Answers.csv"):
    """
    Saves 4 figures into <project_root>/Figures/Additionnal/Heatmaps/
      1) choices sorted by background
      2) choices sorted by familiarity
      3) description sorted by background
      4) description sorted by familiarity
    """
    out_dir = os.path.join(_additional_fig_dir(), "Heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    data = pd.read_csv(os.path.join(project_root, "Data", csv_name))
    subj = list(SUBJECT_ORDER)

    choice_cols = [c for c in data.columns if "[Choice]" in c]
    descr_cols = [c for c in data.columns if "describe this selection?" in c.lower()]

    background_col = next(c for c in data.columns if "describe" in c.lower())
    fam_col = next(c for c in data.columns if "familiar" in c.lower())

    n = min(len(subj), len(choice_cols), len(descr_cols))
    if n == 0:
        raise RuntimeError("No aligned columns found for choices/description.")

    # Choices df
    df_choice = pd.DataFrame(
        {f"Subject {subj[i]}": data[choice_cols[i]].astype(str).values for i in range(n)}
    ).reindex(
        sorted([f"Subject {s}" for s in subj[:n]], key=lambda x: int(x.split()[1])),
        axis=1,
    )

    letters = list("ABCDEF")
    df_choice = df_choice.where(df_choice.isin(letters), np.nan).replace({k: i for i, k in enumerate(letters)})

    # Subject order by consensus strength
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

    # Description df (same subject order)
    df_descr = pd.DataFrame(
        {f"Subject {subj[i]}": data[descr_cols[i]].values for i in range(n)}
    ).reindex(
        sorted([f"Subject {s}" for s in subj[:n]], key=lambda x: int(x.split()[1])),
        axis=1,
    )

    df_descr = df_descr.apply(lambda col: col.astype(str).str.strip().str.lower())
    df_descr = df_descr.applymap(
        lambda x: 0 if "clear choice" in x
        else 1 if "several plausible" in x
        else 2 if "no suitable" in x
        else np.nan
    )
    df_descr = df_descr[order_subjects]

    # Sorting keys
    bg_raw = data[background_col].astype(str).str.strip().str.lower().iloc[: df_choice.shape[0]]
    fam_raw = data[fam_col].astype(str).str.strip().iloc[: df_choice.shape[0]]

    # Strict background
    is_bci = (
        bg_raw.str.contains(r"\bbci\b", regex=True, na=False)
        | bg_raw.str.contains(r"brain[-\s]?computer[-\s]?interface", regex=True, na=False)
    )
    bg_group = np.where(is_bci.values, "BCI", "NON BCI")

    # Familiarity labels
    fam_no = "No familiarity"
    fam_lim = "Limited familiarity (general knowledge or occasional use)"
    fam_mod = "Moderate familiarity (regular use or solid theoretical understanding)"
    fam_high = "High familiarity (extensive use in research or applied BCI/neurofeedback)"

    fam_rank = fam_raw.replace({fam_high: 0, fam_mod: 1, fam_lim: 2, fam_no: 3})
    fam_rank = pd.to_numeric(fam_rank, errors="coerce").fillna(99).astype(int)

    expert_ids = pd.Series(np.arange(df_choice.shape[0]), index=df_choice.index)

    # Orders
    order_experts_bg = (
        list(pd.Index(df_choice.index)[np.where(bg_group == "BCI")[0]])
        + list(pd.Index(df_choice.index)[np.where(bg_group == "NON BCI")[0]])
    )
    order_experts_fam = (
        pd.DataFrame({"rank": fam_rank.values, "eid": expert_ids.values}, index=df_choice.index)
        .sort_values(["rank", "eid"], ascending=[True, True])
        .index.tolist()
    )

    # Reindexed matrices
    df_choice_bg = df_choice.loc[order_experts_bg]
    df_descr_bg = df_descr.loc[order_experts_bg]
    df_choice_fam = df_choice.loc[order_experts_fam]
    df_descr_fam = df_descr.loc[order_experts_fam]

    # Bounds for brackets
    n_exp = df_choice.shape[0]
    n_bci = int((bg_group == "BCI").sum())
    bg_bounds = [0, n_bci, n_exp]
    bg_labels = ["BCI", "NON BCI"]

    fam_groups_ordered = [fam_high, fam_mod, fam_lim, fam_no]
    fam_short = ["High", "Moderate", "Limited", "None"]
    fam_raw_sorted = fam_raw.loc[order_experts_fam].reset_index(drop=True)
    counts_fam = [int((fam_raw_sorted == lab).sum()) for lab in fam_groups_ordered]
    fam_bounds = [0]
    for c in counts_fam:
        fam_bounds.append(fam_bounds[-1] + c)
    fam_bounds[-1] = n_exp

    # Colormaps
    base_cmap = plt.get_cmap("Blues")
    colors_choice = base_cmap(np.linspace(0.35, 0.95, 6))
    cmap_choice = ListedColormap(colors_choice)
    norm_choice = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap_choice.N)

    palette_descr = ["#8FD08A", "#FFBE6A", "#F08A8A"]
    cmap_descr = ListedColormap(palette_descr)
    norm_descr = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_descr.N)
    labels_descr = ["Clear choice", "Several plausible options", "No suitable CSP"]

    # Styling
    cell = 0.46
    tick_fs = 7
    legend_fs = 11
    legend_title_fs = 12

    brace_x = -0.055
    brace_cap = 0.020
    left_margin = 0.10
    def_fig_h_pad = 1.6

    # ==========================================================
    # 4 figures
    # ==========================================================
    def _plot_heatmap(mat, cmap, norm, legend_handles, legend_title, bounds, bound_labels, out_name):
        n_exp_, n_sub_ = mat.shape
        fig, ax = plt.subplots(figsize=(n_sub_ * cell + 2.8, n_exp_ * cell + def_fig_h_pad), dpi=300)
        fig.subplots_adjust(left=left_margin)

        ax.imshow(mat.values, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")

        ax.set_xticks(np.arange(n_sub_))
        ax.set_yticks(np.arange(n_exp_))
        ax.set_xticklabels([f"S{int(s.split()[1]):02d}" for s in mat.columns], rotation=90, fontsize=tick_fs)
        ax.set_yticklabels([f"E{int(i)+1:02d}" for i in mat.index], fontsize=tick_fs)
        ax.set(xlabel="Subjects", ylabel="Experts")

        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(np.arange(-0.5, n_sub_, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_exp_, 1), minor=True)
        ax.grid(which="minor", color=(1, 1, 1, 0.25), linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)

        ax.legend(
            handles=legend_handles,
            title=legend_title,
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=legend_fs,
            title_fontsize=legend_title_fs,
            handlelength=1.2,
            labelspacing=0.6,
        )

        trans = blended_transform_factory(ax.transAxes, ax.transData)

        # separators
        for b in bounds[1:-1]:
            if 0 < b < n_exp_:
                ax.hlines(b - 0.5, -0.5, n_sub_ - 0.5, colors=(1, 1, 1, 0.55), linewidth=1.2)

        # brackets + labels
        for gi in range(len(bound_labels)):
            y0 = bounds[gi]
            y1 = bounds[gi + 1]
            if y1 <= y0:
                continue
            ax.plot([brace_x, brace_x], [y0 - 0.5, y1 - 0.5], transform=trans, color="black", lw=1.6, clip_on=False)
            ax.plot([brace_x, brace_x + brace_cap], [y0 - 0.5, y0 - 0.5], transform=trans, color="black", lw=1.6, clip_on=False)
            ax.plot([brace_x, brace_x + brace_cap], [y1 - 0.5, y1 - 0.5], transform=trans, color="black", lw=1.6, clip_on=False)
            ax.text(
                brace_x - 0.030,
                (y0 + y1 - 1) / 2,
                bound_labels[gi],
                transform=trans,
                rotation=90,
                ha="center",
                va="center",
                fontsize=12,
            )

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, out_name), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 1) choices by background
    _plot_heatmap(
        df_choice_bg,
        cmap_choice,
        norm_choice,
        legend_handles=[Patch(facecolor=colors_choice[i], edgecolor="none", label=letters[i]) for i in range(6)],
        legend_title="Choice",
        bounds=bg_bounds,
        bound_labels=bg_labels,
        out_name="heatmap_choices_sorted_by_background.png",
    )

    # 2) choices by familiarity
    _plot_heatmap(
        df_choice_fam,
        cmap_choice,
        norm_choice,
        legend_handles=[Patch(facecolor=colors_choice[i], edgecolor="none", label=letters[i]) for i in range(6)],
        legend_title="Choice",
        bounds=fam_bounds,
        bound_labels=fam_short,
        out_name="heatmap_choices_sorted_by_familiarity.png",
    )

    # 3) description by background
    _plot_heatmap(
        df_descr_bg,
        cmap_descr,
        norm_descr,
        legend_handles=[Patch(facecolor=palette_descr[i], edgecolor="none", label=labels_descr[i]) for i in range(3)],
        legend_title="Description",
        bounds=bg_bounds,
        bound_labels=bg_labels,
        out_name="heatmap_description_sorted_by_background.png",
    )

    # 4) description by familiarity
    _plot_heatmap(
        df_descr_fam,
        cmap_descr,
        norm_descr,
        legend_handles=[Patch(facecolor=palette_descr[i], edgecolor="none", label=labels_descr[i]) for i in range(3)],
        legend_title="Description",
        bounds=fam_bounds,
        bound_labels=fam_short,
        out_name="heatmap_description_sorted_by_familiarity.png",
    )

    return out_dir


# ==========================================================
# Main entry point to call from your main script
# ==========================================================
def Make_Additional_Figures():
    """
    Generates all "Additionnal" figures and returns their output folders.
    """
    outputs = {}
    outputs["csp_variability"] = Make_CSP_Variability_Topomaps()
    outputs["heatmaps"] = Make_Heatmaps_CSP_Choices_And_Description()
    return outputs