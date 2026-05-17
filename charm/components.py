"""
charm/csct.py — Cross-Stage Consistency Tracker (CSCT)

Tracks semantic trajectory across all pipeline stages using Sentence-BERT
embeddings and cosine similarity drift detection.
"""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-mpnet-base-v2"


class CrossStageConsistencyTracker:
    """
    CSCT: Cross-Stage Consistency Tracker.

    Maintains a running consistency check across all pipeline stages
    using Sentence-BERT embedding-based cosine similarity to track
    semantic trajectories and flag contradictions or anomalous shifts.

    Parameters
    ----------
    drift_threshold : float
        Maximum allowable cosine similarity drop between consecutive stages.
        Default 0.18 (paper-calibrated).
    model_name : str
        Sentence-BERT model identifier.
    """

    def __init__(
        self,
        drift_threshold: float = 0.18,
        model_name: str = _DEFAULT_MODEL,
    ):
        self.drift_threshold = drift_threshold
        logger.info(f"Loading CSCT model: {model_name}")
        self._model = SentenceTransformer(model_name)
        self._embeddings: list[np.ndarray] = []

    def score(self, stage_context: str) -> float:
        """
        Compute CSCT anomaly score for the current stage.

        Parameters
        ----------
        stage_context : str
            The context output of the current stage.

        Returns
        -------
        float
            Anomaly score in [0, 1]. High values indicate semantic drift
            exceeding the drift threshold.
        """
        embedding = self._model.encode([stage_context])[0]
        self._embeddings.append(embedding)

        if len(self._embeddings) < 2:
            return 0.0  # No drift possible on first stage

        prev_emb = self._embeddings[-2].reshape(1, -1)
        curr_emb = self._embeddings[-1].reshape(1, -1)
        similarity = float(cosine_similarity(prev_emb, curr_emb)[0][0])

        # Drift = 1 - similarity; anomaly if drift > threshold
        drift = 1.0 - similarity
        anomaly = max(0.0, drift - self.drift_threshold) / (1.0 - self.drift_threshold)

        if drift > self.drift_threshold:
            logger.debug(
                f"CSCT flagged: drift={drift:.3f} > δ={self.drift_threshold}"
            )

        return min(1.0, anomaly)

    def reset(self):
        """Clear embedding history between pipeline runs."""
        self._embeddings = []


# ---------------------------------------------------------------------------

"""
charm/cpm.py — Confidence Propagation Monitor (CPM)

Tracks model self-reported confidence scores across stages using Bayesian
confidence updating and temperature scaling calibration.
"""

import logging
import numpy as np
from scipy.special import softmax

logger = logging.getLogger(__name__)


class ConfidencePropagationMonitor:
    """
    CPM: Confidence Propagation Monitor.

    Tracks the model's self-reported confidence scores across stages.
    Applies temperature scaling calibration and detects unwarranted
    confidence inflation—where the agent becomes increasingly certain
    despite semantic drift detected by CSCT.

    Parameters
    ----------
    temperature : float
        Temperature scaling parameter T. Calibrated at T=1.4 on
        held-out trajectories (see paper Section 6.1).
    inflation_threshold : float
        Minimum confidence increase per stage to flag as anomalous.
        Default 0.15.
    """

    def __init__(
        self,
        temperature: float = 1.4,
        inflation_threshold: float = 0.15,
    ):
        self.temperature = temperature
        self.inflation_threshold = inflation_threshold
        self._confidence_history: list[float] = []

    def calibrate(self, raw_confidence: float) -> float:
        """Apply temperature scaling to raw LLM confidence score."""
        # Temperature scaling: p_calibrated = softmax(logit / T)
        # For a single probability p, logit = log(p / (1-p))
        p = max(1e-6, min(1 - 1e-6, raw_confidence))
        logit = np.log(p / (1 - p))
        scaled = 1.0 / (1.0 + np.exp(-logit / self.temperature))
        return float(scaled)

    def score(self, raw_confidence: float) -> float:
        """
        Compute CPM anomaly score for the current stage.

        Parameters
        ----------
        raw_confidence : float
            Self-reported confidence from the LLM at this stage [0, 1].

        Returns
        -------
        float
            Anomaly score in [0, 1]. High values indicate unwarranted
            confidence inflation.
        """
        calibrated = self.calibrate(raw_confidence)
        self._confidence_history.append(calibrated)

        if len(self._confidence_history) < 2:
            return 0.0

        prev_conf = self._confidence_history[-2]
        curr_conf = self._confidence_history[-1]
        inflation = curr_conf - prev_conf

        # Anomaly if confidence is rising despite no new strong evidence
        if inflation > self.inflation_threshold:
            anomaly = min(1.0, inflation / (1.0 - self.inflation_threshold))
            logger.debug(
                f"CPM flagged: inflation={inflation:.3f} > "
                f"threshold={self.inflation_threshold}"
            )
            return anomaly

        return 0.0

    def reset(self):
        """Clear confidence history between pipeline runs."""
        self._confidence_history = []


# ---------------------------------------------------------------------------

"""
charm/crt.py — Cascade Resolution Trigger (CRT)

Aggregates signals from SFV, CSCT, and CPM using weighted voting.
Halts the pipeline when the aggregated score exceeds the cascade
detection threshold θ, and infers the cascade type.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Cascade type inference thresholds
_CASCADE_TYPE_MAP = {
    "retrieval":            {"sfv": 0.6, "csct": 0.3, "cpm": 0.1},
    "inference":            {"sfv": 0.5, "csct": 0.5, "cpm": 0.2},
    "context_poisoning":    {"sfv": 0.3, "csct": 0.7, "cpm": 0.1},
    "confidence_inflation": {"sfv": 0.1, "csct": 0.2, "cpm": 0.7},
}


class CascadeResolutionTrigger:
    """
    CRT: Cascade Resolution Trigger.

    Aggregates anomaly signals from SFV, CSCT, and CPM using a configurable
    weighted voting scheme. When the aggregated score exceeds threshold θ,
    the pipeline is halted and an appropriate mitigation strategy is invoked.

    Parameters
    ----------
    threshold : float
        Cascade detection threshold θ (default 0.55, paper-calibrated via
        grid search over validation splits).
    weights : tuple[float, float, float]
        Weights for (SFV, CSCT, CPM). Must sum to 1.0.
        Default (0.4, 0.4, 0.2) reflects lower reliability of CPM.
    """

    def __init__(
        self,
        threshold: float = 0.55,
        weights: tuple = (0.4, 0.4, 0.2),
    ):
        self.threshold = threshold
        self.w_sfv, self.w_csct, self.w_cpm = weights

    def aggregate(
        self,
        sfv_score: float,
        csct_score: float,
        cpm_score: float,
    ) -> tuple[float, bool]:
        """
        Aggregate component anomaly scores and determine if cascade is triggered.

        Parameters
        ----------
        sfv_score : float
            Anomaly score from SFV [0, 1].
        csct_score : float
            Anomaly score from CSCT [0, 1].
        cpm_score : float
            Anomaly score from CPM [0, 1].

        Returns
        -------
        crt_signal : float
            Weighted aggregate anomaly score [0, 1].
        flagged : bool
            True if crt_signal exceeds θ.
        """
        crt_signal = (
            self.w_sfv * sfv_score
            + self.w_csct * csct_score
            + self.w_cpm * cpm_score
        )

        flagged = crt_signal >= self.threshold

        if flagged:
            logger.warning(
                f"CRT triggered: signal={crt_signal:.3f} ≥ θ={self.threshold} "
                f"(SFV={sfv_score:.3f}, CSCT={csct_score:.3f}, CPM={cpm_score:.3f})"
            )

        return crt_signal, flagged

    def infer_cascade_type(
        self,
        sfv_score: float,
        csct_score: float,
        cpm_score: float,
    ) -> str:
        """
        Infer the most likely cascade type from component signal pattern.

        Returns one of: 'retrieval', 'inference', 'context_poisoning',
        'confidence_inflation'.
        """
        scores = {"sfv": sfv_score, "csct": csct_score, "cpm": cpm_score}
        best_type = "retrieval"
        best_match = -1.0

        for cascade_type, profile in _CASCADE_TYPE_MAP.items():
            # Cosine-like match between observed and profile
            match = sum(
                profile[k] * scores[k] for k in scores
            )
            if match > best_match:
                best_match = match
                best_type = cascade_type

        return best_type
