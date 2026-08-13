from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    session_gap_minutes: int = 30
    split_train: float = 0.70
    split_val: float = 0.15
    split_test: float = 0.15
    top_actions: int = 25
    top_services: int = 12
    max_train_sessions: int = 120000
    max_val_sessions: int = 30000
    max_test_reference: int = 15000
    max_ocsvm_train: int = 12000
    batch_size: int = 1024
    epochs: int = 18
    patience: int = 4
    learning_rate: float = 8e-4
    weight_decay: float = 1e-5
    latent_dim: int = 32
    hidden_dim: int = 96
    mask_probability: float = 0.10
    gaussian_noise: float = 0.03
    agreement_weight: float = 0.10
    variance_weight: float = 0.02
    seeds: list[int] = field(default_factory=lambda: [42, 52, 62, 72, 82])
    evt_tail_quantile: float = 0.98
    operational_alert_rate: float = 0.005
    candidate_anomalies: int = 100
    figure_dpi: int = 600
    controlled_corruptions: dict[str, dict[str, float]] = field(default_factory=dict)

    def validate(self) -> None:
        total = self.split_train + self.split_val + self.split_test
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"Split fractions must sum to 1.0, got {total}")
        if not 0.5 < self.evt_tail_quantile < 1.0:
            raise ValueError("evt_tail_quantile must be between 0.5 and 1.0")
        if not 0 < self.operational_alert_rate < 1:
            raise ValueError("operational_alert_rate must be between 0 and 1")


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    cfg = ExperimentConfig(**raw)
    cfg.validate()
    return cfg
