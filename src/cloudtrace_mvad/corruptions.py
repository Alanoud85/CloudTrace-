from __future__ import annotations

import numpy as np
import pandas as pd


def _raise_floor(series: pd.Series, floor: float) -> pd.Series:
    return np.maximum(series.to_numpy(float), floor)


def apply_corruption(frame: pd.DataFrame, family: str, params: dict[str, float]) -> pd.DataFrame:
    out = frame.copy()
    if family == "burst":
        out["log_n_events"] = out["log_n_events"] + params["log_n_events_shift"]
        out["events_per_min"] = out["events_per_min"] * params["events_per_min_factor"]
        for col in ["gap_mean", "gap_median", "gap_std", "gap_max"]:
            out[col] = out[col] * params["gap_factor"]
        out["burst_ratio"] = _raise_floor(out["burst_ratio"], params["burst_ratio_floor"])
    elif family == "context_hijack":
        for col in ["identity_ip_novel_rate", "identity_ua_novel_rate", "identity_region_novel_rate"]:
            out[col] = _raise_floor(out[col], params["novelty_floor"])
        for col in ["unique_ip_ratio", "unique_ua_ratio", "unique_region_ratio"]:
            out[col] = _raise_floor(out[col], params["unique_context_floor"])
        for col in ["ua_rarity_mean", "ua_rarity_max", "ip_rarity_mean", "ip_rarity_max"]:
            out[col] = out[col] + params["rarity_shift"]
    elif family == "api_sequence":
        out["action_switch_rate"] = _raise_floor(out["action_switch_rate"], params["action_switch_floor"])
        out["transition_surprise_mean"] = out["transition_surprise_mean"] + params["transition_mean_shift"]
        out["transition_surprise_max"] = out["transition_surprise_max"] + params["transition_max_shift"]
    elif family == "failure_storm":
        out["error_rate"] = _raise_floor(out["error_rate"], params["error_rate_floor"])
        out["error_rarity_mean"] = out["error_rarity_mean"] + params["error_rarity_shift"]
        out["events_per_min"] = out["events_per_min"] * params["events_per_min_factor"]
    elif family == "region_service_switch":
        out["region_switch_rate"] = _raise_floor(out["region_switch_rate"], params["region_switch_floor"])
        out["service_switch_rate"] = _raise_floor(out["service_switch_rate"], params["service_switch_floor"])
        out["unique_region_ratio"] = _raise_floor(out["unique_region_ratio"], params["unique_region_floor"])
        out["unique_service_ratio"] = _raise_floor(out["unique_service_ratio"], params["unique_service_floor"])
        out["identity_region_novel_rate"] = _raise_floor(out["identity_region_novel_rate"], params["novelty_floor"])
        out["identity_service_novel_rate"] = _raise_floor(out["identity_service_novel_rate"], params["novelty_floor"])
    else:
        raise ValueError(f"Unknown corruption family: {family}")
    return out


def make_controlled_benchmark(
    reference: pd.DataFrame,
    families: dict[str, dict[str, float]],
    seed: int,
) -> pd.DataFrame:
    if not families:
        raise ValueError("At least one corruption family is required")
    rng = np.random.default_rng(seed)
    family_names = list(families)
    order = np.arange(len(reference))
    rng.shuffle(order)
    assigned = np.array([family_names[i % len(family_names)] for i in range(len(reference))], dtype=object)
    family_for_row = np.empty(len(reference), dtype=object)
    family_for_row[order] = assigned
    corrupted_parts = []
    for family in family_names:
        mask = family_for_row == family
        part = apply_corruption(reference.loc[mask].copy(), family, families[family])
        part["controlled_label"] = 1
        part["corruption_family"] = family
        corrupted_parts.append(part)
    clean = reference.copy()
    clean["controlled_label"] = 0
    clean["corruption_family"] = "reference"
    corrupted = pd.concat(corrupted_parts, ignore_index=True)
    return pd.concat([clean, corrupted], ignore_index=True)
