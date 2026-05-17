"""
charm/framework.py
Main CHARMFramework class — orchestrates SFV, CSCT, CPM, and CRT.
"""

from __future__ import annotations
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from .sfv import StageLevelFactVerifier
from .csct import CrossStageConsistencyTracker
from .cpm import ConfidencePropagationMonitor
from .crt import CascadeResolutionTrigger

logger = logging.getLogger(__name__)


@dataclass
class StageOutput:
    """Output produced by a single pipeline stage."""
    stage_id: int
    stage_name: str
    context: str
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class CHARMResult:
    """Full result from a CHARM-monitored pipeline run."""
    final_output: str
    cascade_detected: bool
    cascade_type: Optional[str]
    cascade_depth: Optional[int]       # CDD: stage at which cascade was caught
    mitigation_applied: Optional[str]
    stage_outputs: list[StageOutput]
    sfv_scores: list[float]
    csct_scores: list[float]
    cpm_scores: list[float]
    crt_signal: float
    latency_per_stage_ms: list[float]  # LO/s per stage
    resolved_successfully: bool = False


class CHARMFramework:
    """
    Cascading Hallucination Aware Resolution and Mitigation (CHARM).

    Operates as a parallel observation and enforcement layer alongside
    a standard agentic RAG pipeline. Comprises four components:
      - SFV: Stage-Level Fact Verifier
      - CSCT: Cross-Stage Consistency Tracker
      - CPM: Confidence Propagation Monitor
      - CRT: Cascade Resolution Trigger

    Parameters
    ----------
    sfv_threshold : float
        NLI entailment threshold for SFV (default 0.72, paper-calibrated).
    csct_drift_threshold : float
        Cosine similarity drift threshold for CSCT (default 0.18).
    cpm_temperature : float
        Temperature scaling parameter for CPM calibration (default 1.4).
    crt_threshold : float
        Cascade trigger threshold for CRT (default 0.55).
    crt_weights : tuple[float, float, float]
        Weighted voting scheme (SFV, CSCT, CPM). Must sum to 1.0.
    """

    def __init__(
        self,
        sfv_threshold: float = 0.72,
        csct_drift_threshold: float = 0.18,
        cpm_temperature: float = 1.4,
        crt_threshold: float = 0.55,
        crt_weights: tuple = (0.4, 0.4, 0.2),
    ):
        if abs(sum(crt_weights) - 1.0) > 1e-6:
            raise ValueError("crt_weights must sum to 1.0")

        self.sfv = StageLevelFactVerifier(threshold=sfv_threshold)
        self.csct = CrossStageConsistencyTracker(
            drift_threshold=csct_drift_threshold
        )
        self.cpm = ConfidencePropagationMonitor(temperature=cpm_temperature)
        self.crt = CascadeResolutionTrigger(
            threshold=crt_threshold,
            weights=crt_weights
        )

        self._initial_evidence: Optional[str] = None

    def monitor_stage(
        self,
        stage_output: StageOutput,
        ground_truth: Optional[str] = None,
    ) -> tuple[float, bool]:
        """
        Monitor a single pipeline stage. Returns (crt_signal, cascade_flag).

        Parameters
        ----------
        stage_output : StageOutput
            The output produced by the current pipeline stage.
        ground_truth : str, optional
            Ground truth for offline evaluation. Not available in production.

        Returns
        -------
        crt_signal : float
            Aggregated anomaly score from CRT [0, 1].
        cascade_flagged : bool
            True if the CRT threshold is exceeded.
        """
        # Store initial retrieved evidence for SFV grounding
        if stage_output.stage_id == 1:
            self._initial_evidence = stage_output.context

        # Run the three monitoring components
        sfv_score = self.sfv.score(
            stage_output.context,
            self._initial_evidence or stage_output.context
        )
        csct_score = self.csct.score(stage_output.context)
        cpm_score = self.cpm.score(stage_output.confidence)

        # Aggregate via CRT
        crt_signal, flagged = self.crt.aggregate(sfv_score, csct_score, cpm_score)

        return crt_signal, flagged

    def run_pipeline(
        self,
        query: str,
        pipeline_fn,
        mitigation_strategy: str = "CRR",
    ) -> CHARMResult:
        """
        Run a full agentic pipeline under CHARM monitoring.

        Parameters
        ----------
        query : str
            The user query to process.
        pipeline_fn : callable
            A function that accepts (query, stage_id) and returns StageOutput.
            This is the agentic pipeline being monitored.
        mitigation_strategy : str
            Which mitigation to apply on cascade detection.
            One of: 'CRR', 'SCT', 'PVA', 'PRR', 'HITL'.

        Returns
        -------
        CHARMResult
        """
        from mitigation import get_mitigation

        self._reset()
        stage_outputs = []
        sfv_scores, csct_scores, cpm_scores = [], [], []
        latencies = []
        cascade_detected = False
        cascade_stage = None
        cascade_type = None
        mitigation_applied = None
        resolved = False
        final_output = ""

        stage_id = 1
        max_stages = 5  # Standard 5-stage agentic pipeline

        while stage_id <= max_stages:
            t_start = time.perf_counter()

            # Execute pipeline stage
            stage_output = pipeline_fn(query=query, stage_id=stage_id,
                                       history=stage_outputs)
            stage_outputs.append(stage_output)

            # CHARM monitoring
            sfv_s = self.sfv.score(
                stage_output.context,
                self._initial_evidence or stage_output.context
            )
            csct_s = self.csct.score(stage_output.context)
            cpm_s = self.cpm.score(stage_output.confidence)
            crt_signal, flagged = self.crt.aggregate(sfv_s, csct_s, cpm_s)

            sfv_scores.append(sfv_s)
            csct_scores.append(csct_s)
            cpm_scores.append(cpm_s)

            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000)

            if stage_id == 1:
                self._initial_evidence = stage_output.context

            if flagged:
                cascade_detected = True
                cascade_stage = stage_id
                cascade_type = self.crt.infer_cascade_type(
                    sfv_s, csct_s, cpm_s
                )

                logger.warning(
                    f"Cascade flagged at stage {stage_id} | "
                    f"type={cascade_type} | CRT={crt_signal:.3f}"
                )

                # Apply mitigation
                mitigation = get_mitigation(mitigation_strategy)
                mitigation_result = mitigation.apply(
                    query=query,
                    stage_id=stage_id,
                    stage_outputs=stage_outputs,
                    cascade_type=cascade_type,
                )
                resolved = mitigation_result.success
                mitigation_applied = mitigation_strategy

                if resolved:
                    # Re-execute from the corrected state
                    stage_outputs = mitigation_result.corrected_stages
                    self._reset()
                    # Continue from corrected state
                    stage_id = mitigation_result.restart_stage
                    continue
                else:
                    logger.error("Mitigation failed. Halting pipeline.")
                    break

            if stage_id == max_stages:
                final_output = stage_output.context
            stage_id += 1

        return CHARMResult(
            final_output=final_output,
            cascade_detected=cascade_detected,
            cascade_type=cascade_type,
            cascade_depth=cascade_stage,
            mitigation_applied=mitigation_applied,
            stage_outputs=stage_outputs,
            sfv_scores=sfv_scores,
            csct_scores=csct_scores,
            cpm_scores=cpm_scores,
            crt_signal=crt_signal if stage_outputs else 0.0,
            latency_per_stage_ms=latencies,
            resolved_successfully=resolved,
        )

    def _reset(self):
        """Reset stateful components between pipeline runs."""
        self._initial_evidence = None
        self.csct.reset()
        self.cpm.reset()
