from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def _finish(fig, path: str | Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_dataset_figure(monthly: pd.DataFrame, top_actions: pd.DataFrame, top_services: pd.DataFrame,
                        session_sizes: np.ndarray, path: str | Path, dpi: int = 600) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.plot(pd.to_datetime(monthly["Month"]), monthly["Events"])
    ax.set_title("(a) Monthly CloudTrail event volume", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month", fontsize=11); ax.set_ylabel("Events", fontsize=11); ax.tick_params(labelsize=10)
    ax = axes[0, 1]
    d = top_actions.head(10).sort_values("Count")
    ax.barh(d["Name"], d["Count"])
    ax.set_title("(b) Most frequent API actions", fontsize=13, fontweight="bold"); ax.tick_params(labelsize=10)
    ax = axes[1, 0]
    d = top_services.head(10).sort_values("Count")
    ax.barh(d["Name"], d["Count"])
    ax.set_title("(c) Most frequent AWS services", fontsize=13, fontweight="bold"); ax.tick_params(labelsize=10)
    ax = axes[1, 1]
    bins = np.logspace(0, np.log10(max(float(np.max(session_sizes)), 2)), 30)
    ax.hist(session_sizes, bins=bins)
    ax.set_xscale("log")
    ax.set_title("(d) Session-size distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Events per session", fontsize=11); ax.set_ylabel("Sessions", fontsize=11); ax.tick_params(labelsize=10)
    _finish(fig, path, dpi)


def save_benchmark_figure(benchmark: pd.DataFrame, per_family: pd.DataFrame, path: str | Path, dpi: int = 600) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    display = benchmark.head(10).copy()
    for ax, metric, title in [
        (axes[0, 0], "PR_AUC_mean", "(a) Controlled-anomaly PR-AUC"),
        (axes[0, 1], "ROC_AUC_mean", "(b) Controlled-anomaly ROC-AUC"),
        (axes[1, 0], "F1_mean", "(c) F1 at label-free threshold"),
    ]:
        d = display.sort_values(metric)
        ax.barh(d["Model"], d[metric])
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.tick_params(labelsize=9)
        ax.set_xlim(left=0)
    ax = axes[1, 1]
    pivot = per_family.pivot(index="Model", columns="Corruption", values="mean")
    image = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)), [x.replace("_", " ") for x in pivot.columns], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=9)
    ax.set_title("(d) Recall by controlled anomaly family", fontsize=13, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Recall")
    _finish(fig, path, dpi)


def save_ablation_figure(ablation: pd.DataFrame, seed_metrics: pd.DataFrame, gate_summary: pd.DataFrame,
                         reference_fused: np.ndarray | None, anomaly_fused: np.ndarray | None,
                         path: str | Path, dpi: int = 600) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    d = ablation.sort_values("PR_AUC_mean")
    ax.barh(d["Model"], d["PR_AUC_mean"], xerr=d["PR_AUC_sd"])
    ax.set_title("(a) Ablation PR-AUC", fontsize=13, fontweight="bold"); ax.tick_params(labelsize=9)
    ax = axes[0, 1]
    for name, group in seed_metrics[seed_metrics["Model"].str.contains("CloudTrace|Ablation")].groupby("Model"):
        ax.scatter(group["FPR"], group["Recall"], label=name, s=28)
    ax.set_xlabel("Reference false-positive rate", fontsize=11); ax.set_ylabel("Controlled-anomaly recall", fontsize=11)
    ax.set_title("(b) Threshold behavior across seeds", fontsize=13, fontweight="bold"); ax.tick_params(labelsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1, 0]
    if len(gate_summary):
        g = gate_summary.groupby("View")["Mean_weight"].agg(["mean", "std"]).reindex(["event", "relational", "temporal"]).dropna()
        ax.bar(g.index, g["mean"], yerr=g["std"])
    ax.set_ylim(0, 1); ax.set_ylabel("Mean gate weight", fontsize=11)
    ax.set_title("(c) Learned view-gate weights", fontsize=13, fontweight="bold"); ax.tick_params(labelsize=10)
    ax = axes[1, 1]
    if reference_fused is not None and anomaly_fused is not None and len(reference_fused) and len(anomaly_fused):
        n = min(len(reference_fused), 2500); m = min(len(anomaly_fused), 2500)
        all_z = np.vstack([reference_fused[:n], anomaly_fused[:m]])
        proj = PCA(n_components=2, random_state=0).fit_transform(all_z)
        ax.scatter(proj[:n, 0], proj[:n, 1], s=7, alpha=0.35, label="Reference")
        ax.scatter(proj[n:, 0], proj[n:, 1], s=7, alpha=0.35, label="Controlled anomaly")
        ax.legend(fontsize=9, frameon=False)
    ax.set_xlabel("PC1", fontsize=11); ax.set_ylabel("PC2", fontsize=11)
    ax.set_title("(d) Fused latent representation", fontsize=13, fontweight="bold"); ax.tick_params(labelsize=10)
    _finish(fig, path, dpi)


def save_drift_figure(drift: pd.DataFrame, thresholds: pd.DataFrame, validation_scores: np.ndarray,
                      controlled_scores: np.ndarray | None, main_threshold: float,
                      path: str | Path, dpi: int = 600) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    months = pd.to_datetime(drift["Month"])
    ax = axes[0, 0]
    ax.plot(months, drift["Score_median"], marker="o", label="Median")
    ax.plot(months, drift["Score_p95"], marker="o", label="95th percentile")
    ax.set_title("(a) Future-period anomaly-score drift", fontsize=13, fontweight="bold"); ax.legend(frameon=False, fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=9); ax.tick_params(axis="y", labelsize=10)
    ax = axes[0, 1]
    ax.plot(months, 1000 * drift["Static_alert_rate"], marker="o", label="Static")
    ax.plot(months, 1000 * drift["Adaptive_alert_rate"], marker="o", label="Adaptive")
    ax.set_ylabel("Alerts per 1,000 sessions", fontsize=11)
    ax.set_title("(b) Monthly alert burden", fontsize=13, fontweight="bold"); ax.legend(frameon=False, fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=9); ax.tick_params(axis="y", labelsize=10)
    ax = axes[1, 0]
    ax.plot(thresholds["Target_alerts_per_1000"], thresholds["Observed_alerts_per_1000"], marker="o")
    mx = max(thresholds["Observed_alerts_per_1000"].max(), thresholds["Target_alerts_per_1000"].max())
    ax.plot([0, mx], [0, mx], linestyle="--")
    ax.set_xlabel("Target alerts per 1,000", fontsize=11); ax.set_ylabel("Observed future alerts per 1,000", fontsize=11)
    ax.set_title("(c) EVT alert-budget calibration", fontsize=13, fontweight="bold"); ax.tick_params(labelsize=10)
    ax = axes[1, 1]
    ax.hist(validation_scores, bins=60, density=True, alpha=0.55, label="Validation reference")
    if controlled_scores is not None:
        ax.hist(controlled_scores, bins=60, density=True, alpha=0.45, label="Controlled anomaly")
    ax.axvline(main_threshold, linestyle="--", linewidth=1.5, label="EVT threshold")
    ax.set_title("(d) Score distributions and threshold", fontsize=13, fontweight="bold")
    ax.set_xlabel("Anomaly score", fontsize=11); ax.set_ylabel("Density", fontsize=11); ax.tick_params(labelsize=10)
    ax.legend(fontsize=8, frameon=False)
    _finish(fig, path, dpi)


def save_computation_figure(training: pd.DataFrame, inference: pd.DataFrame, unseen: pd.DataFrame,
                            path: str | Path, dpi: int = 600) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ax = axes[0]
    d = training.groupby("Model")["Train_seconds"].mean().sort_values()
    ax.barh(d.index, d.values); ax.set_xlabel("Mean training time (s)", fontsize=10)
    ax.set_title("(a) Training cost", fontsize=12, fontweight="bold"); ax.tick_params(labelsize=8)
    ax = axes[1]
    row = inference.iloc[0]
    ax.bar(["Parameters", "FLOPs/session"], [row["Parameters"], row["Approx_linear_FLOPs_per_session"]])
    ax.set_yscale("log"); ax.set_title("(b) Model size and compute", fontsize=12, fontweight="bold"); ax.tick_params(labelsize=9)
    ax = axes[2]
    ax.bar(unseen["Subset"], 1000 * unseen["Alert_rate"])
    ax.set_ylabel("Candidate alerts per 1,000", fontsize=10)
    ax.set_title("(c) Seen vs unseen identities", fontsize=12, fontweight="bold"); ax.tick_params(axis="x", rotation=20, labelsize=9)
    _finish(fig, path, dpi)
