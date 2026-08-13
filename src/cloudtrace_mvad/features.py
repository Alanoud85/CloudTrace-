from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


EVENT_BASE = [
    "error_rate", "unique_event_ratio", "unique_service_ratio",
    "action_rarity_mean", "action_rarity_max", "service_rarity_mean", "service_rarity_max",
    "region_rarity_mean", "region_rarity_max", "error_rarity_mean",
]
TEMPORAL = [
    "log_n_events", "log_duration", "events_per_min", "gap_mean", "gap_median", "gap_std", "gap_max",
    "burst_ratio", "nighttime_ratio", "weekend_ratio", "action_switch_rate", "service_switch_rate",
    "region_switch_rate", "ip_switch_rate", "ua_switch_rate", "transition_surprise_mean",
    "transition_surprise_max", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
RELATIONAL = [
    "unique_region_ratio", "unique_ip_ratio", "unique_ua_ratio", "identity_ip_novel_rate",
    "identity_ua_novel_rate", "identity_region_novel_rate", "identity_service_novel_rate",
    "identity_action_novel_rate", "ua_rarity_mean", "ua_rarity_max", "ip_rarity_mean", "ip_rarity_max",
]


@dataclass
class FrequencyModel:
    probabilities: dict[str, dict[str, float]]
    floors: dict[str, float]
    transitions: dict[str, float]
    transition_floor: float

    def rarity(self, column: str, values: Iterable[str]) -> np.ndarray:
        probs = self.probabilities[column]
        floor = self.floors[column]
        return -np.log(np.array([probs.get(str(v), floor) for v in values], dtype=float))

    def transition_surprise(self, actions: list[str]) -> np.ndarray:
        if not actions:
            return np.zeros(1, dtype=float)
        pairs = [f"<START>→{actions[0]}"] + [f"{a}→{b}" for a, b in zip(actions[:-1], actions[1:])]
        return -np.log(np.array([self.transitions.get(p, self.transition_floor) for p in pairs], dtype=float))


@dataclass
class IdentityReference:
    values: dict[str, dict[str, set[str]]]

    def novelty_rate(self, identity: str, relation: str, observed: Iterable[str]) -> float:
        obs = list(map(str, observed))
        if not obs:
            return 0.0
        known = self.values.get(identity, {}).get(relation)
        if known is None:
            return 1.0
        return float(np.mean([v not in known for v in obs]))


@dataclass
class FeatureArtifacts:
    frequency: FrequencyModel
    identity_reference: IdentityReference
    top_actions: list[str]
    top_services: list[str]
    feature_columns: list[str]
    event_features: list[str]
    temporal_features: list[str]
    relational_features: list[str]


def _fit_probability(values: pd.Series) -> tuple[dict[str, float], float]:
    v = values.fillna("<MISSING>").astype(str)
    counts = v.value_counts(dropna=False)
    n = max(int(counts.sum()), 1)
    probs = (counts / n).to_dict()
    return {str(k): float(x) for k, x in probs.items()}, 1.0 / n


def fit_frequency_model(train_events: pd.DataFrame) -> FrequencyModel:
    mapping = {
        "eventName": "_event_name", "eventSource": "_event_source", "awsRegion": "_region",
        "errorCode": "_error_code", "userAgent": "_user_agent", "sourceIPAddress": "_source_ip",
    }
    probabilities, floors = {}, {}
    for name, col in mapping.items():
        probabilities[name], floors[name] = _fit_probability(train_events[col])
    transition_counts: Counter[str] = Counter()
    total = 0
    for _, group in train_events.groupby("session_id", sort=False):
        actions = group.sort_values("_event_time")["_event_name"].astype(str).tolist()
        if not actions:
            continue
        pairs = [f"<START>→{actions[0]}"] + [f"{a}→{b}" for a, b in zip(actions[:-1], actions[1:])]
        transition_counts.update(pairs)
        total += len(pairs)
    total = max(total, 1)
    transitions = {k: v / total for k, v in transition_counts.items()}
    return FrequencyModel(probabilities, floors, transitions, 1.0 / total)


def fit_identity_reference(train_events: pd.DataFrame) -> IdentityReference:
    rel_cols = {
        "ip": "_source_ip", "ua": "_user_agent", "region": "_region",
        "service": "_event_source", "action": "_event_name",
    }
    values: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for identity, group in train_events.groupby("_identity", sort=False):
        for rel, col in rel_cols.items():
            values[str(identity)][rel].update(group[col].astype(str).unique().tolist())
    return IdentityReference({k: dict(v) for k, v in values.items()})


def fit_feature_artifacts(train_events: pd.DataFrame, top_actions: int, top_services: int) -> FeatureArtifacts:
    frequency = fit_frequency_model(train_events)
    identity_reference = fit_identity_reference(train_events)
    actions = train_events["_event_name"].value_counts().head(top_actions).index.astype(str).tolist()
    services = train_events["_event_source"].value_counts().head(top_services).index.astype(str).tolist()
    event_features = EVENT_BASE + [f"action_prop_{x}" for x in actions] + [f"service_prop_{x}" for x in services]
    columns = event_features + TEMPORAL + RELATIONAL
    return FeatureArtifacts(
        frequency=frequency,
        identity_reference=identity_reference,
        top_actions=actions,
        top_services=services,
        feature_columns=columns,
        event_features=event_features,
        temporal_features=list(TEMPORAL),
        relational_features=list(RELATIONAL),
    )


def _switch_rate(values: list[str]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.mean(np.array(values[1:], dtype=object) != np.array(values[:-1], dtype=object)))


def _safe_rarity(freq: FrequencyModel, name: str, values: list[str]) -> tuple[float, float]:
    r = freq.rarity(name, values)
    return float(np.mean(r)), float(np.max(r))


def build_session_features(events: pd.DataFrame, sessions: pd.DataFrame, artifacts: FeatureArtifacts) -> pd.DataFrame:
    rows = []
    freq = artifacts.frequency
    ref = artifacts.identity_reference
    grouped = events.groupby("session_id", sort=False)
    session_meta = sessions.set_index("session_id")
    for session_id, group in grouped:
        group = group.sort_values("_event_time", kind="mergesort")
        if session_id not in session_meta.index:
            continue
        identity = str(group["_identity"].iloc[0])
        actions = group["_event_name"].astype(str).tolist()
        services = group["_event_source"].astype(str).tolist()
        regions = group["_region"].astype(str).tolist()
        ips = group["_source_ip"].astype(str).tolist()
        uas = group["_user_agent"].astype(str).tolist()
        errors = group["_error_code"].astype(str).tolist()
        times = group["_event_time"]
        n = len(group)
        duration = float((times.iloc[-1] - times.iloc[0]).total_seconds()) if n > 1 else 0.0
        gaps = times.diff().dt.total_seconds().dropna().to_numpy(dtype=float)
        if gaps.size == 0:
            gaps = np.zeros(1, dtype=float)
        action_r_mean, action_r_max = _safe_rarity(freq, "eventName", actions)
        service_r_mean, service_r_max = _safe_rarity(freq, "eventSource", services)
        region_r_mean, region_r_max = _safe_rarity(freq, "awsRegion", regions)
        error_r = freq.rarity("errorCode", errors)
        ua_r_mean, ua_r_max = _safe_rarity(freq, "userAgent", uas)
        ip_r_mean, ip_r_max = _safe_rarity(freq, "sourceIPAddress", ips)
        trans = freq.transition_surprise(actions)
        hours = times.dt.hour.to_numpy(dtype=float)
        dows = times.dt.dayofweek.to_numpy(dtype=float)
        row = {
            "session_id": session_id,
            "identity": identity,
            "start_time": times.iloc[0],
            "end_time": times.iloc[-1],
            "n_events": n,
            "error_rate": float(np.mean(np.array(errors) != "<NONE>")),
            "unique_event_ratio": float(len(set(actions)) / n),
            "unique_service_ratio": float(len(set(services)) / n),
            "action_rarity_mean": action_r_mean,
            "action_rarity_max": action_r_max,
            "service_rarity_mean": service_r_mean,
            "service_rarity_max": service_r_max,
            "region_rarity_mean": region_r_mean,
            "region_rarity_max": region_r_max,
            "error_rarity_mean": float(np.mean(error_r)),
            "log_n_events": float(np.log1p(n)),
            "log_duration": float(np.log1p(duration)),
            "events_per_min": float(n / max(duration / 60.0, 1.0)),
            "gap_mean": float(np.mean(gaps)),
            "gap_median": float(np.median(gaps)),
            "gap_std": float(np.std(gaps)),
            "gap_max": float(np.max(gaps)),
            "burst_ratio": float(np.mean(gaps <= 60.0)) if n > 1 else 0.0,
            "nighttime_ratio": float(np.mean((hours < 6) | (hours >= 22))),
            "weekend_ratio": float(np.mean(dows >= 5)),
            "action_switch_rate": _switch_rate(actions),
            "service_switch_rate": _switch_rate(services),
            "region_switch_rate": _switch_rate(regions),
            "ip_switch_rate": _switch_rate(ips),
            "ua_switch_rate": _switch_rate(uas),
            "transition_surprise_mean": float(np.mean(trans)),
            "transition_surprise_max": float(np.max(trans)),
            "hour_sin": float(np.mean(np.sin(2 * np.pi * hours / 24.0))),
            "hour_cos": float(np.mean(np.cos(2 * np.pi * hours / 24.0))),
            "dow_sin": float(np.mean(np.sin(2 * np.pi * dows / 7.0))),
            "dow_cos": float(np.mean(np.cos(2 * np.pi * dows / 7.0))),
            "unique_region_ratio": float(len(set(regions)) / n),
            "unique_ip_ratio": float(len(set(ips)) / n),
            "unique_ua_ratio": float(len(set(uas)) / n),
            "identity_ip_novel_rate": ref.novelty_rate(identity, "ip", ips),
            "identity_ua_novel_rate": ref.novelty_rate(identity, "ua", uas),
            "identity_region_novel_rate": ref.novelty_rate(identity, "region", regions),
            "identity_service_novel_rate": ref.novelty_rate(identity, "service", services),
            "identity_action_novel_rate": ref.novelty_rate(identity, "action", actions),
            "ua_rarity_mean": ua_r_mean,
            "ua_rarity_max": ua_r_max,
            "ip_rarity_mean": ip_r_mean,
            "ip_rarity_max": ip_r_max,
        }
        action_counts = pd.Series(actions).value_counts()
        service_counts = pd.Series(services).value_counts()
        for action in artifacts.top_actions:
            row[f"action_prop_{action}"] = float(action_counts.get(action, 0) / n)
        for service in artifacts.top_services:
            row[f"service_prop_{service}"] = float(service_counts.get(service, 0) / n)
        meta = session_meta.loc[session_id]
        row["split"] = meta["split"]
        rows.append(row)
    out = pd.DataFrame(rows)
    for col in artifacts.feature_columns:
        if col not in out:
            out[col] = 0.0
    return out[["session_id", "identity", "start_time", "end_time", "n_events", "split"] + artifacts.feature_columns]


def fit_scaler(train_features: pd.DataFrame, columns: list[str]) -> RobustScaler:
    scaler = RobustScaler(quantile_range=(5, 95))
    scaler.fit(train_features[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float))
    return scaler


def transform_features(frame: pd.DataFrame, scaler: RobustScaler, columns: list[str]) -> np.ndarray:
    x = frame[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    return scaler.transform(x).astype(np.float32)


def view_indices(artifacts: FeatureArtifacts) -> dict[str, list[int]]:
    lookup = {c: i for i, c in enumerate(artifacts.feature_columns)}
    return {
        "event": [lookup[c] for c in artifacts.event_features],
        "temporal": [lookup[c] for c in artifacts.temporal_features],
        "relational": [lookup[c] for c in artifacts.relational_features],
    }


def runtime_artifacts_from_archive(archive: dict, train_events: pd.DataFrame) -> FeatureArtifacts:
    """Convert the tracked preprocessing dictionary into runtime feature artifacts."""
    freq_raw = archive["frequency_artifacts"]
    probabilities = {
        name: {str(k): float(v) for k, v in freq_raw[name]["probabilities"].items()}
        for name in ["eventName", "eventSource", "awsRegion", "errorCode", "userAgent", "sourceIPAddress"]
    }
    floors = {
        name: float(freq_raw[name]["floor"])
        for name in ["eventName", "eventSource", "awsRegion", "errorCode", "userAgent", "sourceIPAddress"]
    }
    frequency = FrequencyModel(
        probabilities=probabilities,
        floors=floors,
        transitions={str(k): float(v) for k, v in freq_raw["transition"]["probabilities"].items()},
        transition_floor=float(freq_raw["transition"]["floor"]),
    )
    return FeatureArtifacts(
        frequency=frequency,
        identity_reference=fit_identity_reference(train_events),
        top_actions=list(archive["top_actions"]),
        top_services=list(archive["top_services"]),
        feature_columns=list(archive["feature_columns"]),
        event_features=list(archive["event_features"]),
        temporal_features=list(archive["temporal_features"]),
        relational_features=list(archive["relational_features"]),
    )
