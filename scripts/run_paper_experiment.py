from __future__ import annotations

import argparse

from cloudtrace_mvad.config import load_config
from cloudtrace_mvad.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CloudTrace-MVAD paper experiment")
    parser.add_argument("--input", required=True, help="Path to nineteenFeaturesDf.csv")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--output", default="outputs/paper_run")
    parser.add_argument("--device", default=None, help="cpu, cuda, or an explicit torch device")
    args = parser.parse_args()
    cfg = load_config(args.config)
    path = run_experiment(args.input, cfg, args.output, args.device)
    print(f"Experiment outputs written to {path}")


if __name__ == "__main__":
    main()
