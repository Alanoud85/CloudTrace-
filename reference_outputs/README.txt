
CloudTrace-MVAD output package
==============================

This run uses only nineteenFeaturesDf.csv. The transformed dec12_18features.csv file is deliberately excluded.

Main contribution implemented:
1. eventID deduplication and chronological whole-session partitioning;
2. event/content, temporal, and identity-relation views;
3. gated self-supervised multi-view denoising reconstruction;
4. controlled anomaly families without human labels;
5. generalized-Pareto EVT thresholds and confidence-gated drift adaptation;
6. five-seed statistics, ablation, operational alert analysis, and computational profiling.

Interpretation warning:
The original dataset has no event-level benign/malicious ground truth. The controlled corruptions are evaluation constructs, and alerts on unmodified future data are candidate anomalies rather than confirmed attacks.
