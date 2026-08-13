import pandas as pd

from cloudtrace_mvad.corruptions import apply_corruption


def base_frame():
    return pd.DataFrame({
        "log_n_events": [1.0], "events_per_min": [1.0], "gap_mean": [10.0], "gap_median": [10.0], "gap_std": [1.0], "gap_max": [12.0], "burst_ratio": [0.1],
        "identity_ip_novel_rate": [0.0], "identity_ua_novel_rate": [0.0], "identity_region_novel_rate": [0.0], "identity_service_novel_rate": [0.0],
        "unique_ip_ratio": [0.1], "unique_ua_ratio": [0.1], "unique_region_ratio": [0.1], "unique_service_ratio": [0.1],
        "ua_rarity_mean": [1.0], "ua_rarity_max": [2.0], "ip_rarity_mean": [1.0], "ip_rarity_max": [2.0],
        "action_switch_rate": [0.1], "transition_surprise_mean": [1.0], "transition_surprise_max": [2.0],
        "error_rate": [0.0], "error_rarity_mean": [1.0], "region_switch_rate": [0.1], "service_switch_rate": [0.1],
    })


def test_burst_corruption_changes_intensity():
    params = {"log_n_events_shift": 1.0, "events_per_min_factor": 4.0, "gap_factor": 0.2, "burst_ratio_floor": 0.8}
    out = apply_corruption(base_frame(), "burst", params)
    assert out.loc[0, "events_per_min"] == 4.0
    assert out.loc[0, "gap_mean"] == 2.0
    assert out.loc[0, "burst_ratio"] == 0.8
