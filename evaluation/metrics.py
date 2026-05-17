"""
evaluation/metrics.py

All six evaluation metrics from the paper:
  CDR  — Cascade Detection Rate
  FPR  — False Positive Rate
  EPR  — Error Propagation Reduction (Equation 2 from paper)
  MSR  — Mitigation Success Rate
  CDD  — Cascade Depth at Detection (Novel metric)
  LO/s — Latency Overhead per Stage
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class EvaluationResult:
    """Holds all six paper metrics for one system on one dataset."""
    system_name: str
    dataset_name: str
    cdr: float          # Cascade Detection Rate (↑)
    fpr: float          # False Positive Rate (↓)
    epr: float          # Error Propagation Reduction (↑)
    msr: Optional[float]  # Mitigation Success Rate (↑); None if no mitigation
    cdd: Optional[float]  # Cascade Depth at Detection (↓); None if no detection
    lo_s_ms: float      # Latency Overhead per Stage in ms
    lo_s_std: float     # Standard deviation of LO/s across 5 runs

    def __str__(self) -> str:
        msr_str = f"{self.msr*100:.1f}%" if self.msr is not None else "N/A"
        cdd_str = f"{self.cdd:.1f}" if self.cdd is not None else "N/A"
        return (
            f"[{self.system_name} on {self.dataset_name}]\n"
            f"  CDR:  {self.cdr*100:.1f}%\n"
            f"  FPR:  {self.fpr*100:.1f}%\n"
            f"  EPR:  {self.epr*100:.1f}%\n"
            f"  MSR:  {msr_str}\n"
            f"  CDD:  {cdd_str}\n"
            f"  LO/s: {self.lo_s_ms:.0f} ± {self.lo_s_std:.0f} ms"
        )


def compute_cdr(
    predictions: list[dict],
    injection_stage_window: int = 1,
) -> float:
    """
    Cascade Detection Rate (CDR).

    Percentage of injected cascades identified before final output.
    A detection is a true positive (TP) if the CRT flags an anomaly
    at stage s_j where j ≤ s_inject + injection_stage_window.

    Parameters
    ----------
    predictions : list[dict]
        Each dict must contain:
          - 'is_cascade': bool — whether this trajectory has an injected cascade
          - 'cascade_detected': bool — whether CHARM flagged it
          - 'detection_stage': int or None — stage at which CHARM flagged
          - 'injection_stage': int or None — stage where cascade was injected
    injection_stage_window : int
        Number of stages after injection within which detection counts as TP.

    Returns
    -------
    float : CDR in [0, 1]
    """
    cascades = [p for p in predictions if p["is_cascade"]]
    if not cascades:
        return 0.0

    true_positives = 0
    for p in cascades:
        if not p["cascade_detected"]:
            continue
        det_stage = p.get("detection_stage")
        inj_stage = p.get("injection_stage")
        if det_stage is not None and inj_stage is not None:
            if det_stage <= inj_stage + injection_stage_window:
                true_positives += 1
        else:
            true_positives += 1  # No stage info — count as detected

    return true_positives / len(cascades)


def compute_fpr(predictions: list[dict]) -> float:
    """
    False Positive Rate (FPR).

    Percentage of grounded (non-cascade) trajectories incorrectly flagged.

    Parameters
    ----------
    predictions : list[dict]
        Each dict must contain 'is_cascade' and 'cascade_detected'.
    """
    clean = [p for p in predictions if not p["is_cascade"]]
    if not clean:
        return 0.0
    false_positives = sum(1 for p in clean if p["cascade_detected"])
    return false_positives / len(clean)


def compute_epr(
    predictions_charm: list[dict],
    predictions_none: list[dict],
) -> float:
    """
    Error Propagation Reduction (EPR). — Equation 2 in the paper.

    EPR = 1 - (EM_CHARM / EM_None)

    where EM is the exact-match error rate on the final pipeline output
    across all injected trajectories.

    Parameters
    ----------
    predictions_charm : list[dict]
        Predictions from CHARM system. Each dict needs 'is_cascade',
        'final_output', 'ground_truth'.
    predictions_none : list[dict]
        Predictions from No-Detection baseline (same trajectories).
    """
    def error_rate(preds):
        cascades = [p for p in preds if p["is_cascade"]]
        if not cascades:
            return 0.0
        errors = sum(
            1 for p in cascades
            if p["final_output"].strip().lower()
            != p["ground_truth"].strip().lower()
        )
        return errors / len(cascades)

    em_charm = error_rate(predictions_charm)
    em_none = error_rate(predictions_none)

    if em_none == 0:
        return 0.0
    return 1.0 - (em_charm / em_none)


def compute_msr(predictions: list[dict]) -> Optional[float]:
    """
    Mitigation Success Rate (MSR).

    Percentage of detected cascades that were successfully resolved
    by the mitigation pattern.

    Parameters
    ----------
    predictions : list[dict]
        Each dict needs 'cascade_detected' and 'resolved_successfully'.
    """
    detected = [p for p in predictions if p.get("cascade_detected")]
    if not detected:
        return None
    resolved = sum(1 for p in detected if p.get("resolved_successfully"))
    return resolved / len(detected)


def compute_cdd(predictions: list[dict]) -> Optional[float]:
    """
    Cascade Depth at Detection (CDD). — Novel metric.

    Average pipeline stage at which a cascade is detected.
    Lower values indicate earlier intervention.

    Parameters
    ----------
    predictions : list[dict]
        Each dict needs 'cascade_detected' and 'detection_stage'.
    """
    detected = [
        p for p in predictions
        if p.get("cascade_detected") and p.get("detection_stage") is not None
    ]
    if not detected:
        return None
    return float(np.mean([p["detection_stage"] for p in detected]))


def compute_lo_s(latency_records: list[list[float]]) -> tuple[float, float]:
    """
    Latency Overhead per Stage (LO/s).

    Average wall-clock time added by CHARM components at each individual
    pipeline stage, averaged over all stages and all runs.

    Parameters
    ----------
    latency_records : list[list[float]]
        Each inner list contains per-stage latencies (ms) for one run.
        Should contain 5 runs for the paper's reported ±std.

    Returns
    -------
    mean_ms : float
        Mean per-stage overhead in milliseconds.
    std_ms : float
        Standard deviation across runs.
    """
    all_stage_latencies = [
        lat for run in latency_records for lat in run
    ]
    mean_ms = float(np.mean(all_stage_latencies))
    # Std over run-level means (5 runs as per paper)
    run_means = [np.mean(run) for run in latency_records]
    std_ms = float(np.std(run_means))
    return mean_ms, std_ms


def compute_all_metrics(
    predictions_charm: list[dict],
    predictions_none: list[dict],
    latency_records: list[list[float]],
    system_name: str = "CHARM",
    dataset_name: str = "unknown",
) -> EvaluationResult:
    """
    Compute all six paper metrics in one call.

    Parameters
    ----------
    predictions_charm : list[dict]
        CHARM system outputs. Each dict requires:
          is_cascade, cascade_detected, detection_stage, injection_stage,
          final_output, ground_truth, resolved_successfully
    predictions_none : list[dict]
        No-detection baseline outputs (same trajectories).
    latency_records : list[list[float]]
        Per-stage latency lists from 5 independent runs.
    """
    cdr = compute_cdr(predictions_charm)
    fpr = compute_fpr(predictions_charm)
    epr = compute_epr(predictions_charm, predictions_none)
    msr = compute_msr(predictions_charm)
    cdd = compute_cdd(predictions_charm)
    lo_s_mean, lo_s_std = compute_lo_s(latency_records)

    return EvaluationResult(
        system_name=system_name,
        dataset_name=dataset_name,
        cdr=cdr,
        fpr=fpr,
        epr=epr,
        msr=msr,
        cdd=cdd,
        lo_s_ms=lo_s_mean,
        lo_s_std=lo_s_std,
    )
