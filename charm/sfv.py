"""
charm/sfv.py — Stage-Level Fact Verifier (SFV)

Uses cross-encoder NLI (DeBERTa-v3-base) to verify each stage output
against the initially retrieved evidence. Returns an anomaly score
where high values indicate low entailment (potential cascade).
"""

import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-base"
# NLI label indices: 0=contradiction, 1=neutral, 2=entailment
_ENTAILMENT_IDX = 2


class StageLevelFactVerifier:
    """
    SFV: Stage-Level Fact Verifier.

    Checks each intermediate stage output against the initially retrieved
    evidence before passing it to the subsequent stage, using cross-encoder
    NLI entailment scoring.

    Parameters
    ----------
    threshold : float
        Entailment threshold τ. Outputs below this score trigger anomaly
        flag. Default 0.72 (paper-calibrated on validation splits).
    model_name : str
        HuggingFace cross-encoder model identifier.
    """

    def __init__(
        self,
        threshold: float = 0.72,
        model_name: str = _DEFAULT_MODEL,
    ):
        self.threshold = threshold
        logger.info(f"Loading SFV model: {model_name}")
        self._model = CrossEncoder(model_name, num_labels=3)

    def score(self, stage_context: str, initial_evidence: str) -> float:
        """
        Compute SFV anomaly score for a stage output.

        Parameters
        ----------
        stage_context : str
            The context output of the current stage.
        initial_evidence : str
            The initially retrieved evidence (stage 1 output), used as the
            grounding reference for all subsequent stages.

        Returns
        -------
        float
            Anomaly score in [0, 1]. Low values = well-grounded.
            High values = potential cascade (entailment score below threshold).
        """
        if not initial_evidence or not stage_context:
            return 0.0

        logits = self._model.predict(
            [(initial_evidence, stage_context)],
            apply_softmax=True
        )
        entailment_prob = float(logits[0][_ENTAILMENT_IDX])

        # Anomaly score: 1 - entailment (high = anomalous)
        anomaly = 1.0 - entailment_prob
        is_flagged = entailment_prob < self.threshold

        if is_flagged:
            logger.debug(
                f"SFV flagged: entailment={entailment_prob:.3f} < τ={self.threshold}"
            )

        return anomaly
