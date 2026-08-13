from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .baselines import FrequencyRarityBaseline, PCAReconstructionBaseline, IsolationForestBaseline, OneClassSVMBaseline
from .config import ExperimentConfig
from .corruptions import make_controlled_benchmark
from .data import load_cloudtrail_csv, sessionize, chronological_split, attach_split
from .evt import fit_evt
from .features import fit_feature_artifacts, build_session_features, fit_scaler, transform_features, view_indices
from .metrics import binary_metrics, summarize_runs, exact_sign_flip_test
from .model import CloudTraceMVAD, training_loss, parameter_count, approximate_linear_flops
from .scoring import score_model, RobustComponentCalibrator
from .utils import ensure_dir, json_dump, set_seed, stable_hash
from .plotting import save_dataset_figure, save_benchmark_figure, save_ablation_figure, save_drift_figure, save_computation_figure


def _device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _views_tensor(x: np.ndarray, indices: dict[str, list[int]], names: list[str], device: torch.device):
    return [torch.as_tensor(x[:, indices[n]], dtype=torch.float32, device=device) for n in names]


def train_mvad(x_train, x_val, indices, view_names, cfg, seed, device):
    set_seed(seed)
    dims = [len(indices[n]) for n in view_names]
    model = CloudTraceMVAD(dims, cfg.hidden_dim, cfg.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    train_views_np = [x_train[:, indices[n]].astype(np.float32) for n in view_names]
    val_views = _views_tensor(x_val, indices, view_names, device)
    dataset = TensorDataset(*[torch.from_numpy(v) for v in train_views_np])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, generator=generator)
    best_state, best_val, wait = None, float("inf"), 0
    history = []
    started = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train(); running = []
        for batch in loader:
            clean = [b.to(device) for b in batch]
            optimizer.zero_grad(set_to_none=True)
            loss, parts = training_loss(model, clean, cfg.mask_probability, cfg.gaussian_noise,
                                        cfg.agreement_weight, cfg.variance_weight)
            loss.backward(); optimizer.step(); running.append(parts["loss"])
        model.eval()
        with torch.no_grad():
            out = model(val_views)
            val_loss = torch.stack([torch.mean((p - t) ** 2) for p, t in zip(out["reconstructions"], val_views)]).mean().item()
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(running)), "val_reconstruction": val_loss})
        if val_loss < best_val - 1e-7:
            best_val = val_loss; wait = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    elapsed = time.perf_counter() - started
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history), elapsed


def _neural_scores(model, x, indices, view_names, device, calibrator=None):
    views = _views_tensor(x, indices, view_names, device)
    components, gate, fused = score_model(model, views)
    if calibrator is None:
        return components, gate, fused, None
    return components, gate, fused, calibrator.score(components)


def run_experiment(input_path: str | Path, cfg: ExperimentConfig, output_dir: str | Path, device_name: str | None = None):
    output = ensure_dir(output_dir)
    tables = ensure_dir(output / "tables"); models = ensure_dir(output / "models"); logs = ensure_dir(output / "logs"); figures = ensure_dir(output / "figures")
    device = _device(device_name)

    events, schema, audit = load_cloudtrail_csv(input_path)
    events, sessions = sessionize(events, cfg.session_gap_minutes)
    sessions = chronological_split(sessions, cfg.split_train, cfg.split_val)
    events = attach_split(events, sessions)
    audit["sessions"] = int(len(sessions))
    json_dump({"input": str(input_path), "schema": schema.__dict__, "audit": audit}, logs / "data_source_audit.json")

    train_events = events[events["split"] == "train"].copy()
    artifacts = fit_feature_artifacts(train_events, cfg.top_actions, cfg.top_services)
    feature_frame = build_session_features(events, sessions, artifacts)
    scaler = fit_scaler(feature_frame[feature_frame["split"] == "train"], artifacts.feature_columns)
    x_all = transform_features(feature_frame, scaler, artifacts.feature_columns)
    indices = view_indices(artifacts)
    split_masks = {s: feature_frame["split"].eq(s).to_numpy() for s in ["train", "validation", "test"]}
    split_x = {s: x_all[m] for s, m in split_masks.items()}
    split_f = {s: feature_frame.loc[m].reset_index(drop=True) for s, m in split_masks.items()}
    joblib.dump(artifacts, models / "feature_engineering_artifacts.joblib")
    joblib.dump(scaler, models / "robust_scaler.joblib")

    monthly = events.assign(Month=events["_event_time"].dt.to_period("M").astype(str)).groupby("Month").size().reset_index(name="Events")
    top_actions = events["_event_name"].value_counts().reset_index(); top_actions.columns = ["Name", "Count"]
    top_services = events["_event_source"].value_counts().reset_index(); top_services.columns = ["Name", "Count"]
    monthly.to_csv(tables / "Supplementary_monthly_event_counts.csv", index=False)
    top_actions.to_csv(tables / "Supplementary_top_API_actions.csv", index=False)
    top_services.to_csv(tables / "Supplementary_top_AWS_services.csv", index=False)
    q = feature_frame[feature_frame["split"] == "train"][artifacts.feature_columns].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    q.columns = [f"q{int(100*x):02d}" for x in q.columns]
    q.reset_index(names="Feature").to_csv(tables / "Supplementary_training_feature_quantiles.csv", index=False)
    save_dataset_figure(monthly, top_actions, top_services, sessions["n_events"].to_numpy(), figures / "Figure_3_dataset_results.png", cfg.figure_dpi)

    audit_table = pd.DataFrame({"Property": [
        "Raw CSV rows", "Invalid/missing timestamps removed", "Duplicate eventID rows removed", "Curated unique events",
        "Duplicate rate (%)", "Sessions", "Unique identities", "Unique source IP values", "Unique user agents",
        "Unique API actions", "Unique AWS services", "Unique AWS regions", "Start time", "End time"],
        "Value": [audit["raw_rows"], audit["invalid_timestamps_removed"], audit["duplicates_removed"], audit["curated_rows"],
                  audit["duplicate_rate_percent"], audit["sessions"], audit["unique_identities"], audit["unique_source_ips"],
                  audit["unique_user_agents"], audit["unique_api_actions"], audit["unique_aws_services"], audit["unique_regions"],
                  audit["start_time"], audit["end_time"]]})
    audit_table.to_csv(tables / "Table_1_dataset_audit.csv", index=False)

    split_rows = []
    for s in ["test", "train", "validation"]:
        sf = split_f[s]
        ev = events[events["split"] == s]
        durations = (sf["end_time"] - sf["start_time"]).dt.total_seconds()
        split_rows.append({"split": s, "sessions": len(sf), "events": len(ev), "identities": sf["identity"].nunique(),
                           "median_events": sf["n_events"].median(), "p95_events": sf["n_events"].quantile(.95),
                           "median_duration_sec": durations.median()})
    pd.DataFrame(split_rows).to_csv(tables / "Table_2_chronological_splits.csv", index=False)
    pd.DataFrame([(v, f) for v, cols in [("Event/content", artifacts.event_features), ("Temporal/sequence", artifacts.temporal_features),
                                         ("Relational/identity graph", artifacts.relational_features)] for f in cols],
                 columns=["View", "Feature"]).to_csv(tables / "Table_3_feature_dictionary.csv", index=False)

    all_seed_rows, per_family_rows, gate_rows, histories, training_profiles = [], [], [], [], []
    full_seed42_checkpoint = None
    seed42_reference_fused = None
    seed42_anomaly_fused = None
    seed42_validation_scores = None
    seed42_controlled_scores = None
    models_to_run = ["CloudTrace-MVAD (Full)", "Ablation: No event", "Ablation: No temporal", "Ablation: No relational", "Ablation: Event only"]
    view_map = {
        "CloudTrace-MVAD (Full)": ["event", "temporal", "relational"],
        "Ablation: No event": ["temporal", "relational"],
        "Ablation: No temporal": ["event", "relational"],
        "Ablation: No relational": ["event", "temporal"],
        "Ablation: Event only": ["event"],
    }

    for seed in cfg.seeds:
        ref = split_f["test"].iloc[: cfg.max_test_reference].copy()
        bench = make_controlled_benchmark(ref, cfg.controlled_corruptions, seed)
        x_bench = transform_features(bench, scaler, artifacts.feature_columns)
        y = bench["controlled_label"].to_numpy(int)
        x_train = split_x["train"][: cfg.max_train_sessions]
        x_val = split_x["validation"][: cfg.max_val_sessions]

        # Classical baselines
        rarity_names = ["action_rarity_mean", "action_rarity_max", "service_rarity_mean", "service_rarity_max",
                        "region_rarity_mean", "region_rarity_max", "error_rarity_mean", "ua_rarity_mean", "ip_rarity_mean"]
        rarity_idx = [artifacts.feature_columns.index(c) for c in rarity_names]
        classical = {
            "Frequency/Rarity": FrequencyRarityBaseline(rarity_idx),
            "PCA Reconstruction": PCAReconstructionBaseline(),
            "Isolation Forest": IsolationForestBaseline(seed),
            "One-Class SVM": OneClassSVMBaseline(nu=max(cfg.operational_alert_rate, 0.005)),
        }
        for name, baseline in classical.items():
            xt = x_train
            if name == "One-Class SVM" and len(xt) > cfg.max_ocsvm_train:
                rng = np.random.default_rng(seed)
                xt = xt[rng.choice(len(xt), cfg.max_ocsvm_train, replace=False)]
            started = time.perf_counter(); baseline.fit(xt); train_seconds = time.perf_counter() - started
            val_scores = baseline.score_samples(x_val)
            threshold = float(np.quantile(val_scores, 1.0 - cfg.operational_alert_rate))
            scores = baseline.score_samples(x_bench)
            met = binary_metrics(y, scores, threshold)
            all_seed_rows.append({"Model": name, "Seed": seed, **met})
            training_profiles.append({"Model": name, "Seed": seed, "Parameters": np.nan, "Approx_FLOPs": np.nan,
                                      "Train_seconds": train_seconds, "Peak_VRAM_MB": 0.0})
            for family in cfg.controlled_corruptions:
                m = bench["corruption_family"].eq(family).to_numpy()
                per_family_rows.append({"Model": name, "Seed": seed, "Corruption": family,
                                        "Recall": float(np.mean(scores[m] >= threshold))})

        # Denoising autoencoder baseline implemented as one-view MVAD over all features.
        all_idx = {"all": list(range(len(artifacts.feature_columns)))}
        dae, hist, train_seconds = train_mvad(x_train, x_val, all_idx, ["all"], cfg, seed, device)
        cval, _, _, _ = _neural_scores(dae, x_val, all_idx, ["all"], device)
        # one reconstruction component plus disagreement; score only reconstruction
        dae_val_scores = cval[:, 0]
        dae_threshold = float(np.quantile(dae_val_scores, 1.0 - cfg.operational_alert_rate))
        cb, _, _, _ = _neural_scores(dae, x_bench, all_idx, ["all"], device)
        dae_scores = cb[:, 0]
        met = binary_metrics(y, dae_scores, dae_threshold)
        all_seed_rows.append({"Model": "Denoising Autoencoder", "Seed": seed, **met})
        training_profiles.append({"Model": "Denoising Autoencoder", "Seed": seed, "Parameters": parameter_count(dae),
                                  "Approx_FLOPs": approximate_linear_flops(dae), "Train_seconds": train_seconds, "Peak_VRAM_MB": np.nan})
        for _, r in hist.iterrows(): histories.append({"Model": "Denoising Autoencoder", "Seed": seed, **r.to_dict()})
        for family in cfg.controlled_corruptions:
            m = bench["corruption_family"].eq(family).to_numpy()
            per_family_rows.append({"Model": "Denoising Autoencoder", "Seed": seed, "Corruption": family,
                                    "Recall": float(np.mean(dae_scores[m] >= dae_threshold))})

        for model_name in models_to_run:
            names = view_map[model_name]
            model, hist, train_seconds = train_mvad(x_train, x_val, indices, names, cfg, seed, device)
            val_components, val_gate, _, _ = _neural_scores(model, x_val, indices, names, device)
            if len(names) == 3:
                weights = np.array([2, 2, 2, 1], dtype=float) / 7.0
            else:
                weights = np.ones(val_components.shape[1], dtype=float)
                # disagreement receives half the reconstruction weight when present
                if val_components.shape[1] > 1:
                    weights[-1] = 0.5
                weights /= weights.sum()
            calibrator = RobustComponentCalibrator.fit(val_components, weights)
            val_scores = calibrator.score(val_components)
            evt = fit_evt(val_scores, cfg.evt_tail_quantile)
            threshold = evt.threshold(cfg.operational_alert_rate)
            bench_components, bench_gate, _, _ = _neural_scores(model, x_bench, indices, names, device)
            scores = calibrator.score(bench_components)
            met = binary_metrics(y, scores, threshold)
            all_seed_rows.append({"Model": model_name, "Seed": seed, **met})
            training_profiles.append({"Model": model_name, "Seed": seed, "Parameters": parameter_count(model),
                                      "Approx_FLOPs": approximate_linear_flops(model), "Train_seconds": train_seconds,
                                      "Peak_VRAM_MB": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0})
            for _, r in hist.iterrows(): histories.append({"Model": model_name, "Seed": seed, **r.to_dict()})
            for j, name in enumerate(names):
                gate_rows.append({"Model": model_name, "Seed": seed, "View": name, "Mean_weight": float(bench_gate[:, j].mean()),
                                  "Std_weight": float(bench_gate[:, j].std())})
            for family in cfg.controlled_corruptions:
                m = bench["corruption_family"].eq(family).to_numpy()
                per_family_rows.append({"Model": model_name, "Seed": seed, "Corruption": family,
                                        "Recall": float(np.mean(scores[m] >= threshold))})
            if model_name == "CloudTrace-MVAD (Full)" and seed == 42:
                clean_n = len(ref)
                _, _, seed42_reference_fused, _ = _neural_scores(model, x_bench[:clean_n], indices, names, device)
                _, _, seed42_anomaly_fused, _ = _neural_scores(model, x_bench[clean_n:], indices, names, device)
                seed42_validation_scores = val_scores.copy()
                seed42_controlled_scores = scores[clean_n:].copy()
                full_seed42_checkpoint = {
                    "state_dict": model.state_dict(), "view_names": names, "calibrator": calibrator.to_dict(),
                    "feature_columns": artifacts.feature_columns, "view_indices": indices, "config": cfg.__dict__,
                    "evt_metadata": evt.to_dict(),
                }

    seed_df = pd.DataFrame(all_seed_rows)
    seed_df.to_csv(tables / "Supplementary_all_seed_metrics.csv", index=False)
    benchmark = summarize_runs(seed_df)
    benchmark = benchmark.sort_values("PR_AUC_mean", ascending=False)
    benchmark.to_csv(tables / "Table_4_controlled_anomaly_benchmark.csv", index=False)
    per = pd.DataFrame(per_family_rows)
    per_summary = per.groupby(["Model", "Corruption"])["Recall"].agg(["mean", "std", "count"]).reset_index()
    per_summary["Recall_mean_SD"] = per_summary.apply(lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1)
    per_summary.to_csv(tables / "Table_5_per_corruption_detection.csv", index=False)
    benchmark[benchmark["Model"].str.contains("CloudTrace|Ablation")].to_csv(tables / "Table_6_ablation_study.csv", index=False)
    pd.DataFrame(training_profiles).to_csv(tables / "Supplementary_training_profiles_all_seeds.csv", index=False)
    pd.DataFrame(histories).to_csv(tables / "Supplementary_training_histories.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(tables / "Supplementary_gate_weights_all_seeds.csv", index=False)

    # Paired comparisons
    stat_rows = []
    full = seed_df[seed_df["Model"] == "CloudTrace-MVAD (Full)"].set_index("Seed")
    for comp in seed_df["Model"].drop_duplicates():
        if comp == "CloudTrace-MVAD (Full)": continue
        other = seed_df[seed_df["Model"] == comp].set_index("Seed")
        common = full.index.intersection(other.index)
        for metric in ["PR_AUC", "ROC_AUC", "F1"]:
            diff = full.loc[common, metric].to_numpy() - other.loc[common, metric].to_numpy()
            effect, p = exact_sign_flip_test(diff)
            stat_rows.append({"Comparator": comp, "Metric": metric, "Mean_difference": diff.mean(),
                              "Paired_effect_size": effect, "Exact_sign_flip_p": p})
    pd.DataFrame(stat_rows).to_csv(tables / "Table_8_statistical_comparisons.csv", index=False)
    per_summary_for_plot = per_summary[["Model", "Corruption", "mean"]].copy()
    save_benchmark_figure(benchmark, per_summary_for_plot, figures / "Figure_4_controlled_anomaly_performance.png", cfg.figure_dpi)
    ablation = benchmark[benchmark["Model"].str.contains("CloudTrace|Ablation")].copy()
    save_ablation_figure(ablation, seed_df, pd.DataFrame(gate_rows), seed42_reference_fused, seed42_anomaly_fused,
                         figures / "Figure_5_ablation_and_representation.png", cfg.figure_dpi)

    # Operational full-model analysis using seed-42 checkpoint.
    if full_seed42_checkpoint is not None:
        torch.save(full_seed42_checkpoint, models / "CloudTrace_MVAD_full_seed42.pt")
        model = CloudTraceMVAD([len(indices[n]) for n in ["event", "temporal", "relational"]], cfg.hidden_dim, cfg.latent_dim).to(device)
        model.load_state_dict(full_seed42_checkpoint["state_dict"])
        cal = RobustComponentCalibrator(**{k: np.asarray(v) for k, v in full_seed42_checkpoint["calibrator"].items()})
        val_comp, _, _, _ = _neural_scores(model, split_x["validation"], indices, ["event", "temporal", "relational"], device)
        val_scores = cal.score(val_comp)
        evt = fit_evt(val_scores, cfg.evt_tail_quantile)
        test_comp, test_gate, test_fused, _ = _neural_scores(model, split_x["test"], indices, ["event", "temporal", "relational"], device)
        test_scores = cal.score(test_comp)
        threshold_rows = []
        for rate in [0.001, 0.005, 0.01]:
            threshold = evt.threshold(rate)
            observed = float(np.mean(test_scores >= threshold))
            threshold_rows.append({"Target_alert_rate": rate, "Target_alerts_per_1000": 1000*rate, "Threshold": threshold,
                                   "Observed_test_rate": observed, "Observed_alerts_per_1000": 1000*observed, "Method": "GPD-EVT"})
        pd.DataFrame(threshold_rows).to_csv(tables / "Table_9_EVT_operational_thresholds.csv", index=False)
        main_threshold = evt.threshold(cfg.operational_alert_rate)
        tf = split_f["test"].copy(); tf["score"] = test_scores
        train_ids = set(split_f["train"]["identity"].astype(str))
        tf["seen_identity"] = tf["identity"].astype(str).isin(train_ids)
        unseen_rows = []
        for label, mask in [("Seen identity", tf["seen_identity"]), ("Unseen identity", ~tf["seen_identity"])]:
            g = tf[mask]
            unseen_rows.append({"Subset": label, "Sessions": len(g), "Mean_score": g["score"].mean(),
                                "Alert_rate": np.mean(g["score"] >= main_threshold) if len(g) else np.nan})
        pd.DataFrame(unseen_rows).to_csv(tables / "Table_11_unseen_identity_candidate_anomaly_analysis.csv", index=False)
        candidates = tf.nlargest(cfg.candidate_anomalies, "score").copy()
        candidates["identity_hash"] = candidates["identity"].map(stable_hash)
        candidates[["session_id", "start_time", "end_time", "n_events", "identity_hash", "score"]].to_csv(
            tables / "Table_12_top_candidate_anomalies_unlabelled.csv", index=False)

        # Confidence-gated monthly adaptation.
        base_center = float(np.median(val_scores))
        base_mad = float(np.median(np.abs(val_scores - base_center)))
        base_scale = max(1.4826 * base_mad, 1e-8)
        target_z = (main_threshold - base_center) / base_scale
        current_center, current_scale = base_center, base_scale
        drift_rows = []
        tf["Month"] = pd.to_datetime(tf["start_time"], utc=True).dt.to_period("M").astype(str)
        for month, group in tf.groupby("Month", sort=True):
            vals = group["score"].to_numpy(float)
            adaptive_threshold = float(current_center + target_z * current_scale)
            static_rate = float(np.mean(vals >= main_threshold))
            adaptive_rate = float(np.mean(vals >= adaptive_threshold))
            low_risk = vals[vals < adaptive_threshold]
            if len(low_risk) >= 20:
                month_center = float(np.median(low_risk))
                month_scale = max(float(1.4826 * np.median(np.abs(low_risk - month_center))), 1e-8)
                current_center = 0.8 * current_center + 0.2 * month_center
                current_scale = 0.8 * current_scale + 0.2 * month_scale
            drift_rows.append({"Month": month, "Sessions": len(group), "Score_median": float(np.median(vals)),
                               "Score_p95": float(np.quantile(vals, .95)), "Static_threshold": main_threshold,
                               "Static_alert_rate": static_rate, "Adaptive_threshold": adaptive_threshold,
                               "Adaptive_alert_rate": adaptive_rate, "Updated_center": current_center, "Updated_scale": current_scale})
        drift_df = pd.DataFrame(drift_rows)
        drift_df.to_csv(tables / "Table_10_future_period_drift_and_alerts.csv", index=False)

        # Inference computational profile.
        batch_n = min(len(split_x["test"]), 4096)
        sample_x = split_x["test"][:batch_n]
        sample_views = _views_tensor(sample_x, indices, ["event", "temporal", "relational"], device)
        for _ in range(10):
            score_model(model, sample_views)
        if device.type == "cuda": torch.cuda.synchronize(device)
        timings = []
        for _ in range(30):
            t0 = time.perf_counter(); score_model(model, sample_views)
            if device.type == "cuda": torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - t0)
        total_s = float(np.mean(timings))
        per_session_ms = 1000.0 * total_s / max(batch_n, 1)
        throughput = batch_n / max(total_s, 1e-12)
        import psutil, os
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        inference_df = pd.DataFrame([{
            "Model": "CloudTrace-MVAD (Full)", "Parameters": parameter_count(model),
            "FP32_model_size_MB": parameter_count(model) * 4 / 1024**2,
            "Approx_linear_FLOPs_per_session": approximate_linear_flops(model),
            "Inference_ms_per_session": per_session_ms, "Throughput_sessions_per_sec": throughput,
            "Latency_run_p95_ms": 1000 * float(np.quantile(timings, .95)), "Current_process_peak_RSS_MB": rss,
            "Device": str(device)}])
        inference_df.to_csv(tables / "Table_13_inference_computational_profile.csv", index=False)
        architecture_df = pd.DataFrame([
            ["Input", "Three session views", "Event=47, Temporal=21, Relational=12"],
            ["View encoder", "Linear-GELU-LayerNorm-Linear-LayerNorm", f"hidden={cfg.hidden_dim}, latent={cfg.latent_dim}"],
            ["Fusion", "Softmax gated weighted sum", "One learned weight per view and session"],
            ["Decoders", "One decoder per view", "latent-hidden-original view dimension"],
            ["Training objective", "Denoising reconstruction + agreement + variance guard", f"mask={cfg.mask_probability}, noise={cfg.gaussian_noise}"],
            ["Optimizer", "AdamW", f"lr={cfg.learning_rate}, weight_decay={cfg.weight_decay}"],
            ["Runs", "Independent random seeds", str(tuple(cfg.seeds))],
            ["Threshold", "Generalized Pareto EVT", f"tail q={cfg.evt_tail_quantile}; alert rate={cfg.operational_alert_rate}"],
        ], columns=["Component", "Specification", "Value"])
        architecture_df.to_csv(tables / "Table_14_model_architecture_and_hyperparameters.csv", index=False)
        training_df = pd.DataFrame(training_profiles)
        grouped_training = training_df.groupby("Model", as_index=False).agg(Parameters=("Parameters", "mean"), Approx_FLOPs=("Approx_FLOPs", "mean"),
            Train_seconds_mean=("Train_seconds", "mean"), Train_seconds_sd=("Train_seconds", "std"), Peak_VRAM_MB=("Peak_VRAM_MB", "max"))
        grouped_training.to_csv(tables / "Table_7_training_computational_profile.csv", index=False)
        unseen_df = pd.DataFrame(unseen_rows)
        save_drift_figure(drift_df, pd.DataFrame(threshold_rows), seed42_validation_scores if seed42_validation_scores is not None else val_scores,
                          seed42_controlled_scores, main_threshold, figures / "Figure_6_operational_drift_analysis.png", cfg.figure_dpi)
        save_computation_figure(training_df, inference_df, unseen_df, figures / "Figure_7_computation_and_unseen_actors.png", cfg.figure_dpi)

    manifest = {"config": cfg.__dict__, "source": str(input_path), "audit": audit,
                "features": artifacts.feature_columns, "views": indices, "device": str(device)}
    json_dump(manifest, output / "run_manifest.json")
    return output
