from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import genpareto


@dataclass
class EVTModel:
    tail_quantile: float
    u: float
    shape: float
    scale: float
    tail_n: int
    n_reference: int

    def threshold(self, alert_rate: float) -> float:
        tail_fraction = 1.0 - self.tail_quantile
        if alert_rate >= tail_fraction:
            return float(np.quantile([self.u], 0.5))
        conditional_survival = alert_rate / tail_fraction
        q = 1.0 - conditional_survival
        excess = genpareto.ppf(q, self.shape, loc=0.0, scale=self.scale)
        return float(self.u + excess)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def fit_evt(scores: np.ndarray, tail_quantile: float = 0.98) -> EVTModel:
    x = np.asarray(scores, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 100:
        raise ValueError("At least 100 finite validation scores are required for EVT fitting")
    u = float(np.quantile(x, tail_quantile))
    excesses = x[x > u] - u
    if len(excesses) < 20:
        raise ValueError("Insufficient upper-tail observations for GPD fitting")
    shape, _, scale = genpareto.fit(excesses, floc=0.0)
    return EVTModel(tail_quantile, u, float(shape), float(scale), int(len(excesses)), int(len(x)))
