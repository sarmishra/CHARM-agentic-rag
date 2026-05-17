#!/usr/bin/env python3
"""
scripts/run_ablation.py

Reproduces the component ablation study (Table 5 in the paper).
Runs six configurations on HotpotQA and prints CDR, FPR, and LO/s.

Usage:
    python scripts/run_ablation.py \
        --dataset hotpotqa \
        --input data/hotpotqa_retrieval_injected.jsonl \
        --output results/ablation/ \
        --n_runs 5
"""

import argparse
import json
import os
import time
from pathlib import Path

from charm.framework import CHARMFramework
from evaluation.metrics import compute_all_metrics


ABLATION_CONFIGS = {
    "SFV Only":         {"sfv_w": 1.0, "csct_w": 0.0, "cpm_w": 0.0},
    "CSCT Only":        {"sfv_w": 0.0, "csct_w": 1.0, "cpm_w": 0.0},
    "CPM Only":         {"sfv_w": 0.0, "csct_w": 0.0, "cpm_w": 1.0},
    "SFV + CSCT":       {"sfv_w": 0.5, "csct_w": 0.5, "cpm_w": 0.0},
    "SFV + CSCT + CPM": {"sfv_w": 0.4, "csct_w": 0.4, "cpm_w": 0.2},
    "Full CHARM":       {"sfv_w": 0.4, "csct_w": 0.4, "cpm_w": 0.2},
}


def run_ablation_config(
    config_name: str,
    weights: dict,
    trajectories: list[dict],
    pipeline_fn,
    n_runs: int = 5,
) -> dict:
    """Run a single ablation configuration n_runs times and average results."""
    all_predictions = []
    all_latencies = []

    for run in range(n_runs):
        charm = CHARMFramework(
            sfv_threshold=0.72,
            csct_drift_threshold=0.18,
            cpm_temperature=1.4,
            crt_threshold=0.55,
            crt_weights=(
                weights["sfv_w"],
                weights["csct_w"],
                weights["cpm_w"],
            ),
        )

        run_predictions = []
        run_latencies = []

        for traj in trajectories:
            t_start = time.perf_counter()
            result = charm.run_pipeline(
                query=traj["query"],
                pipeline_fn=pipeline_fn,
            )
            t_end = time.perf_counter()

            run_latencies.extend(result.latency_per_stage_ms)
            run_predictions.append({
                "is_cascade": traj.get("is_cascade", False),
                "cascade_detected": result.cascade_detected,
                "detection_stage": result.cascade_depth,
                "injection_stage": traj.get("injection_stage"),
                "final_output": result.final_output,
                "ground_truth": traj.get("ground_truth_answer", ""),
                "resolved_successfully": result.resolved_successfully,
            })

        all_predictions.extend(run_predictions)
        all_latencies.append(run_latencies)

    metrics = compute_all_metrics(
        predictions_charm=all_predictions,
        predictions_none=[],  # EPR not needed for ablation
        latency_records=all_latencies,
        system_name=config_name,
        dataset_name="hotpotqa",
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Injected trajectory JSONL file")
    parser.add_argument("--output", default="results/ablation/",
                        help="Output directory")
    parser.add_argument("--n_runs", type=int, default=5,
                        help="Number of runs per config for std computation")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    with open(args.input) as f:
        trajectories = [json.loads(line) for line in f]

    # Import your pipeline function here
    # from your_pipeline import pipeline_fn
    # For testing, use a dummy pipeline:
    from evaluation.agentic_wrapper import create_pipeline_fn
    pipeline_fn = create_pipeline_fn(model="gpt-4o")

    print("\n" + "="*65)
    print("CHARM Component Ablation Study — Table 5 (Paper)")
    print("="*65)
    print(f"{'Configuration':<25} {'CDR':>8} {'FPR':>8} {'LO/s':>12}")
    print("-"*65)

    results = []
    for config_name, weights in ABLATION_CONFIGS.items():
        metrics = run_ablation_config(
            config_name, weights, trajectories, pipeline_fn, args.n_runs
        )
        results.append(metrics)
        print(
            f"{config_name:<25} "
            f"{metrics.cdr*100:>7.1f}% "
            f"{metrics.fpr*100:>7.1f}% "
            f"{metrics.lo_s_ms:>8.0f}±{metrics.lo_s_std:.0f}ms"
        )

        # Save individual result
        out_file = Path(args.output) / f"{config_name.replace(' ', '_')}.json"
        with open(out_file, "w") as f:
            json.dump({
                "config": config_name,
                "cdr": metrics.cdr,
                "fpr": metrics.fpr,
                "epr": metrics.epr,
                "lo_s_ms": metrics.lo_s_ms,
                "lo_s_std": metrics.lo_s_std,
            }, f, indent=2)

    print("="*65)
    print(f"\nResults saved to {args.output}")
    print("\nCopy these numbers into Table 5 (tab:ablation) in the LaTeX paper.")


if __name__ == "__main__":
    main()
