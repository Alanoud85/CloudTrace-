from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch


@dataclass
class RobustComponentCalibrator:
    median: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    @classmethod
    def fit(cls, components: np.ndarray, weights: np.ndarray | None = None) -> "RobustComponentCalibrator":
        median = np.median(components, axis=0)
        mad = np.median(np.abs(components - median), axis=0)
        scale = np.maximum(1.4826 * mad, 1e-8)
        if weights is None:
            if components.shape[1] == 4:
                weights = np.array([2, 2, 2, 1], dtype=float) / 7.0
            else:
                weights = np.ones(components.shape[1], dtype=float) / components.shape[1]
        return cls(median=median, scale=scale, weights=np.asarray(weights, dtype=float))

    def transform(self, components: np.ndarray) -> np.ndarray:
        return (components - self.median) / self.scale

    def score(self, components: np.ndarray) -> np.ndarray:
        return self.transform(components) @ self.weights

    def to_dict(self) -> dict:
        return {"median": self.median, "scale": self.scale, "weights": self.weights}


@torch.no_grad()
def score_model(model, views: list[torch.Tensor]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    out = model(views)
    recon_errors = []
    for pred, target in zip(out["reconstructions"], views):
        recon_errors.append(torch.mean((pred - target) ** 2, dim=1))
    if len(out["latents"]) > 1:
        pairs = [torch.mean((a - b) ** 2, dim=1) for a, b in combinations(out["latents"], 2)]
        disagreement = torch.stack(pairs, dim=1).mean(dim=1)
    else:
        disagreement = torch.zeros_like(recon_errors[0])
    components = torch.stack(recon_errors + [disagreement], dim=1).cpu().numpy()
    gate = out["gate"].cpu().numpy()
    fused = out["fused"].cpu().numpy()
    return components, gate, fused
