from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from cloudtrace_mvad.data import load_cloudtrail_csv, sessionize, chronological_split, attach_split
from cloudtrace_mvad.features import build_session_features, transform_features, runtime_artifacts_from_archive
from cloudtrace_mvad.model import CloudTraceMVAD
from cloudtrace_mvad.scoring import RobustComponentCalibrator, score_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the future chronological split with the archived seed-42 model")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "reference_outputs" / "models"
    archived_artifacts = joblib.load(model_dir / "feature_engineering_artifacts.joblib")
    scaler = joblib.load(model_dir / "robust_scaler.joblib")
    ckpt = torch.load(model_dir / "CloudTrace_MVAD_full_seed42.pt", map_location="cpu", weights_only=False)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    events, _, _ = load_cloudtrail_csv(args.input)
    events, sessions = sessionize(events, int(ckpt["config"]["session_gap_minutes"]))
    sessions = chronological_split(sessions, float(ckpt["config"]["split_train"]), float(ckpt["config"]["split_val"]))
    events = attach_split(events, sessions)
    artifacts = runtime_artifacts_from_archive(archived_artifacts, events[events["split"] == "train"].copy())
    features = build_session_features(events, sessions, artifacts)
    future = features[features["split"] == "test"].reset_index(drop=True)
    x = transform_features(future, scaler, ckpt["feature_columns"])
    names = list(ckpt["view_names"]); indices = ckpt["view_indices"]
    model = CloudTraceMVAD([len(indices[n]) for n in names], int(ckpt["config"]["hidden_dim"]), int(ckpt["config"]["latent_dim"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    views = [torch.as_tensor(x[:, indices[n]], dtype=torch.float32, device=device) for n in names]
    components, gate, _ = score_model(model, views)
    cal = RobustComponentCalibrator(
        median=np.asarray(ckpt["calibrator"]["median"]),
        scale=np.asarray(ckpt["calibrator"]["scale"]),
        weights=np.asarray(ckpt["calibrator"]["weights"]),
    )
    scores = cal.score(components)
    result = future[["session_id", "identity", "start_time", "end_time", "n_events"]].copy()
    result["score"] = scores
    for j, name in enumerate(names): result[f"gate_{name}"] = gate[:, j]
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} session scores to {args.output}")


if __name__ == "__main__":
    main()
