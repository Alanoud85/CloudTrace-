from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1] / "reference_outputs" / "tables"
audit = pd.read_csv(root / "Table_1_dataset_audit.csv")
bench = pd.read_csv(root / "Table_4_controlled_anomaly_benchmark.csv")
evt = pd.read_csv(root / "Table_9_EVT_operational_thresholds.csv")
unseen = pd.read_csv(root / "Table_11_unseen_identity_candidate_anomaly_analysis.csv")

full = bench.loc[bench["Model"] == "CloudTrace-MVAD (Full)"].iloc[0]
iso = bench.loc[bench["Model"] == "Isolation Forest"].iloc[0]
main = evt.loc[evt["Target_alert_rate"].sub(0.005).abs().idxmin()]
seen = unseen.loc[unseen["Subset"] == "Seen identity"].iloc[0]
new = unseen.loc[unseen["Subset"] == "Unseen identity"].iloc[0]

print("CloudTrace-MVAD archived reference run")
print("=" * 39)
for _, row in audit.iterrows():
    if row["Property"] in {"Raw CSV rows", "Duplicate eventID rows removed", "Curated unique events", "Sessions"}:
        print(f"{row['Property']}: {row['Value']}")
print(f"CloudTrace-MVAD PR-AUC: {full['PR_AUC']}")
print(f"CloudTrace-MVAD ROC-AUC: {full['ROC_AUC']}")
print(f"Isolation Forest PR-AUC: {iso['PR_AUC']}")
print(f"Main EVT threshold: {main['Threshold']:.4f}")
print(f"Future candidate alerts per 1000: {main['Observed_alerts_per_1000']:.2f}")
print(f"Unseen/seen alert-rate ratio: {new['Alert_rate'] / seen['Alert_rate']:.2f}")
