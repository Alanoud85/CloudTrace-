from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


COLUMN_ALIASES = {
    "event_time": ["eventTime", "eventtime", "event_time", "timestamp", "EventTime"],
    "event_id": ["eventID", "eventId", "eventid", "event_id", "EventId"],
    "event_name": ["eventName", "eventname", "event_name", "EventName"],
    "event_source": ["eventSource", "eventsource", "event_source", "EventSource"],
    "region": ["awsRegion", "awsregion", "region", "aws_region"],
    "source_ip": ["sourceIPAddress", "sourceipaddress", "source_ip", "sourceIpAddress"],
    "user_agent": ["userAgent", "useragent", "user_agent"],
    "error_code": ["errorCode", "errorcode", "error_code"],
}

IDENTITY_ALIASES = [
    "userIdentity.principalId", "userIdentity_principalId", "userIdentityprincipalId",
    "principalId", "principalid", "userIdentity.arn", "userIdentity_arn",
    "userIdentityarn", "arn", "userIdentity.userName", "userIdentity_userName",
    "userIdentityuserName", "userName", "username", "userIdentity.accessKeyId",
    "userIdentity_accessKeyId", "userIdentityaccessKeyId", "accessKeyId", "accesskeyid",
]


@dataclass
class Schema:
    event_time: str
    event_id: str
    event_name: str
    event_source: str
    region: str
    source_ip: str
    user_agent: str
    error_code: str | None
    identity_columns: list[str]


def _first_existing(columns: Iterable[str], aliases: list[str], required: bool = True) -> str | None:
    lookup = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias in columns:
            return alias
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    if required:
        raise KeyError(f"None of the expected columns were found: {aliases}")
    return None


def detect_schema(df: pd.DataFrame) -> Schema:
    cols = list(df.columns)
    resolved = {
        key: _first_existing(cols, aliases, required=(key != "error_code"))
        for key, aliases in COLUMN_ALIASES.items()
    }
    identity = [c for c in IDENTITY_ALIASES if c in cols]
    if not identity:
        lower = {c.lower(): c for c in cols}
        identity = [lower[c.lower()] for c in IDENTITY_ALIASES if c.lower() in lower]
    identity = list(dict.fromkeys(identity))
    if not identity:
        identity = [resolved["source_ip"]]
    return Schema(identity_columns=identity, **resolved)


def load_cloudtrail_csv(path: str | Path) -> tuple[pd.DataFrame, Schema, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    schema = detect_schema(df)
    raw_rows = len(df)
    df["_event_time"] = pd.to_datetime(df[schema.event_time], utc=True, errors="coerce")
    invalid_time = int(df["_event_time"].isna().sum())
    df = df.loc[df["_event_time"].notna()].copy()
    before_dedup = len(df)
    df = df.drop_duplicates(subset=[schema.event_id], keep="first").copy()
    duplicates_removed = before_dedup - len(df)
    for semantic, col in {
        "event_name": schema.event_name,
        "event_source": schema.event_source,
        "region": schema.region,
        "source_ip": schema.source_ip,
        "user_agent": schema.user_agent,
    }.items():
        df[f"_{semantic}"] = df[col].fillna("<MISSING>").astype(str)
    if schema.error_code is None:
        df["_error_code"] = "<NONE>"
    else:
        error = df[schema.error_code]
        df["_error_code"] = error.where(error.notna() & error.astype(str).str.len().gt(0), "<NONE>").astype(str)
    df["_identity"] = build_identity_key(df, schema.identity_columns, schema.source_ip)
    df = df.sort_values(["_identity", "_event_time", schema.event_id], kind="mergesort").reset_index(drop=True)
    audit = {
        "raw_rows": int(raw_rows),
        "invalid_timestamps_removed": invalid_time,
        "duplicates_removed": int(duplicates_removed),
        "curated_rows": int(len(df)),
        "duplicate_rate_percent": float(100.0 * duplicates_removed / max(before_dedup, 1)),
        "start_time": df["_event_time"].min(),
        "end_time": df["_event_time"].max(),
        "unique_identities": int(df["_identity"].nunique()),
        "unique_source_ips": int(df["_source_ip"].nunique()),
        "unique_user_agents": int(df["_user_agent"].nunique()),
        "unique_api_actions": int(df["_event_name"].nunique()),
        "unique_aws_services": int(df["_event_source"].nunique()),
        "unique_regions": int(df["_region"].nunique()),
    }
    return df, schema, audit


def build_identity_key(df: pd.DataFrame, identity_columns: list[str], source_ip_col: str) -> pd.Series:
    out = pd.Series("", index=df.index, dtype="object")
    for col in identity_columns:
        vals = df[col].fillna("").astype(str).str.strip()
        valid = vals.ne("") & ~vals.str.lower().isin({"nan", "none", "null", "<missing>"})
        out = out.mask(out.eq("") & valid, vals)
    fallback = df[source_ip_col].fillna("<MISSING_IP>").astype(str)
    out = out.mask(out.eq(""), "ip:" + fallback)
    return out


def sessionize(df: pd.DataFrame, gap_minutes: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    gaps = work.groupby("_identity", sort=False)["_event_time"].diff().dt.total_seconds()
    new_session = gaps.isna() | gaps.gt(gap_minutes * 60)
    work["_session_seq"] = new_session.groupby(work["_identity"]).cumsum().astype("int64") - 1
    work["session_id"] = work["_identity"].astype(str) + "::" + work["_session_seq"].astype(str)
    sessions = work.groupby("session_id", sort=False).agg(
        identity=("_identity", "first"),
        start_time=("_event_time", "min"),
        end_time=("_event_time", "max"),
        n_events=("_event_time", "size"),
    ).reset_index()
    sessions["duration_sec"] = (sessions["end_time"] - sessions["start_time"]).dt.total_seconds()
    sessions = sessions.sort_values(["start_time", "session_id"], kind="mergesort").reset_index(drop=True)
    return work, sessions


def chronological_split(sessions: pd.DataFrame, train_frac: float, val_frac: float) -> pd.DataFrame:
    n = len(sessions)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    labels = np.full(n, "test", dtype=object)
    labels[:n_train] = "train"
    labels[n_train:n_train + n_val] = "validation"
    out = sessions.copy()
    out["split"] = labels
    return out


def attach_split(events: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    mapping = sessions.set_index("session_id")["split"]
    out = events.copy()
    out["split"] = out["session_id"].map(mapping)
    return out
