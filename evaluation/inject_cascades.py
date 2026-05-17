"""
evaluation/inject_cascades.py

Four-method cascade injection protocol as described in Section 6.2 of the paper.

Injection methods:
  1. Retrieval Cascade   — Replace top-1 doc with counterfactual (GPT-4o)
  2. Inference Cascade   — Prepend misleading cue at stage 2
  3. Context Poisoning   — Insert adversarial embedding-proximal passage
  4. Confidence Inflation — Strip hedging language from stage outputs
"""

from __future__ import annotations
import json
import random
import logging
import re
from pathlib import Path
from openai import OpenAI

logger = logging.getLogger(__name__)

_HEDGING_PATTERNS = [
    r"\bpossibly\b", r"\bperhaps\b", r"\bmight\b", r"\bmay be\b",
    r"\bit is possible that\b", r"\buncertain\b", r"\bapparently\b",
    r"\bseems to\b", r"\bit appears\b", r"\blikely\b",
]


class CascadeInjector:
    """
    Implements the four-method cascade injection protocol from the paper.

    Parameters
    ----------
    openai_client : OpenAI
        Initialized OpenAI client for counterfactual generation.
    random_seed : int
        Seed for reproducibility.
    """

    def __init__(self, openai_client: OpenAI, random_seed: int = 42):
        self.client = openai_client
        random.seed(random_seed)

    # ------------------------------------------------------------------
    # Method 1: Retrieval Cascade Injection
    # ------------------------------------------------------------------

    def inject_retrieval_cascade(
        self,
        trajectory: dict,
        target_stage: int = 1,
    ) -> dict:
        """
        Replace the top-1 retrieved document with a GPT-4o-generated
        counterfactual that is semantically proximate but factually incorrect.

        Used for: HotpotQA
        Detectable by: SFV (source-output semantic divergence at stage 1)

        Parameters
        ----------
        trajectory : dict
            A trajectory dict with 'stages' list.
        target_stage : int
            The retrieval stage to corrupt (default 1).

        Returns
        -------
        dict : Modified trajectory with injection metadata.
        """
        traj = trajectory.copy()
        original_doc = traj["stages"][target_stage - 1]["retrieved_document"]
        query = traj["query"]

        counterfactual = self._generate_counterfactual(original_doc, query)

        traj["stages"][target_stage - 1]["retrieved_document"] = counterfactual
        traj["stages"][target_stage - 1]["context_output"] = counterfactual
        traj["is_cascade"] = True
        traj["cascade_type"] = "retrieval"
        traj["injection_stage"] = target_stage
        traj["original_document"] = original_doc

        logger.info(f"Retrieval cascade injected at stage {target_stage}")
        return traj

    def _generate_counterfactual(self, document: str, query: str) -> str:
        """Use GPT-4o to generate a semantically proximate but factually wrong doc."""
        prompt = (
            f"The following document is relevant to this query: '{query}'\n\n"
            f"Document:\n{document}\n\n"
            f"Rewrite the document so that it appears factually credible and "
            f"is on the same topic, but changes at least one key fact "
            f"(a date, a number, a proper noun, or a causal claim) to be "
            f"incorrect. Keep the writing style identical. "
            f"Return ONLY the rewritten document, no preamble."
        )
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Method 2: Inference Cascade Injection
    # ------------------------------------------------------------------

    def inject_inference_cascade(
        self,
        trajectory: dict,
        target_stage: int = 2,
    ) -> dict:
        """
        Prepend a misleading reasoning cue to the intermediate context
        at stage 2, corrupting inference while leaving retrieval clean.

        Used for: MuSiQue
        Detectable by: CSCT (entailment drop between evidence and conclusion)

        Parameters
        ----------
        trajectory : dict
            A trajectory dict with 'stages' list.
        target_stage : int
            The reasoning stage to corrupt (default 2).
        """
        traj = trajectory.copy()
        original_context = traj["stages"][target_stage - 1]["context_output"]
        query = traj["query"]

        misleading_cue = self._generate_misleading_cue(original_context, query)
        corrupted_context = misleading_cue + "\n\n" + original_context

        traj["stages"][target_stage - 1]["context_output"] = corrupted_context
        traj["is_cascade"] = True
        traj["cascade_type"] = "inference"
        traj["injection_stage"] = target_stage
        traj["original_context"] = original_context

        logger.info(f"Inference cascade injected at stage {target_stage}")
        return traj

    def _generate_misleading_cue(self, context: str, query: str) -> str:
        """Generate a misleading reasoning premise."""
        prompt = (
            f"Query: {query}\nContext: {context[:500]}\n\n"
            f"Write a single sentence (max 30 words) that sounds like a "
            f"logical reasoning step but leads to an incorrect conclusion "
            f"about the query. It must be plausible but factually wrong. "
            f"Return ONLY the sentence."
        )
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return f"Note: {response.choices[0].message.content.strip()}"

    # ------------------------------------------------------------------
    # Method 3: Context Poisoning Injection
    # ------------------------------------------------------------------

    def inject_context_poisoning(
        self,
        trajectory: dict,
        target_stage: int = 1,
    ) -> dict:
        """
        Insert an adversarial passage into the knowledge base using an
        embedding-proximal approach, ensuring the poisoned document passes
        retrieval relevance filtering.

        Used for: 2WikiMultiHopQA
        Detectable by: CSCT (anomalous semantic shift between stages)

        Note: Full adversarial embedding optimization requires access to the
        retriever's embedding model. This implementation uses GPT-4o to generate
        a semantically proximate adversarial passage as an approximation.
        """
        traj = trajectory.copy()
        query = traj["query"]
        original_context = traj["stages"][target_stage - 1]["context_output"]

        adversarial_passage = self._generate_adversarial_passage(
            query, original_context
        )

        # Prepend adversarial passage — simulates successful retrieval injection
        traj["stages"][target_stage - 1]["context_output"] = (
            adversarial_passage + "\n\n" + original_context
        )
        traj["is_cascade"] = True
        traj["cascade_type"] = "context_poisoning"
        traj["injection_stage"] = target_stage
        traj["adversarial_passage"] = adversarial_passage

        logger.info(f"Context poisoning injected at stage {target_stage}")
        return traj

    def _generate_adversarial_passage(self, query: str, context: str) -> str:
        """Generate embedding-proximal adversarial passage."""
        prompt = (
            f"Query: {query}\nContext excerpt: {context[:300]}\n\n"
            f"Write a short paragraph (50-80 words) that:\n"
            f"1. Uses similar vocabulary and style to the context above\n"
            f"2. Appears related to the query\n"
            f"3. Contains a subtle but critical factual error about the topic\n"
            f"4. Would be retrieved by a semantic search for the query\n"
            f"Return ONLY the paragraph."
        )
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Method 4: Confidence Inflation Injection
    # ------------------------------------------------------------------

    def inject_confidence_inflation(
        self,
        trajectory: dict,
        target_stage: int = 2,
    ) -> dict:
        """
        Strip hedging language from stage outputs to simulate false certainty
        propagation. Causes downstream stages to treat uncertain facts as definite.

        Used for: Custom Adversarial Set (all cascade types)
        Detectable by: CPM (confidence score increase despite semantic drift)
        """
        traj = trajectory.copy()
        original_context = traj["stages"][target_stage - 1]["context_output"]

        # Strip all hedging language patterns
        stripped = original_context
        for pattern in _HEDGING_PATTERNS:
            stripped = re.sub(pattern, "", stripped, flags=re.IGNORECASE)
        # Clean up double spaces
        stripped = re.sub(r"  +", " ", stripped).strip()

        # Artificially inflate confidence score for this stage
        traj["stages"][target_stage - 1]["context_output"] = stripped
        traj["stages"][target_stage - 1]["confidence"] = min(
            1.0,
            traj["stages"][target_stage - 1].get("confidence", 0.5) + 0.35
        )
        traj["is_cascade"] = True
        traj["cascade_type"] = "confidence_inflation"
        traj["injection_stage"] = target_stage
        traj["original_context"] = original_context

        logger.info(f"Confidence inflation injected at stage {target_stage}")
        return traj

    # ------------------------------------------------------------------
    # Batch injection
    # ------------------------------------------------------------------

    def inject_dataset(
        self,
        trajectories: list[dict],
        cascade_type: str,
        n_samples: int = 500,
    ) -> list[dict]:
        """
        Apply cascade injection to a dataset sample.

        Parameters
        ----------
        trajectories : list[dict]
            Input trajectory dicts (clean).
        cascade_type : str
            One of: 'retrieval', 'inference', 'context_poisoning',
            'confidence_inflation'.
        n_samples : int
            Number of trajectories to inject.
        """
        method_map = {
            "retrieval":            self.inject_retrieval_cascade,
            "inference":            self.inject_inference_cascade,
            "context_poisoning":    self.inject_context_poisoning,
            "confidence_inflation": self.inject_confidence_inflation,
        }

        if cascade_type not in method_map:
            raise ValueError(
                f"Unknown cascade type '{cascade_type}'. "
                f"Choose from: {list(method_map.keys())}"
            )

        sample = random.sample(trajectories, min(n_samples, len(trajectories)))
        injected = []
        for traj in sample:
            try:
                injected.append(method_map[cascade_type](traj))
            except Exception as e:
                logger.warning(f"Injection failed for trajectory: {e}")
                continue

        logger.info(
            f"Injected {len(injected)} '{cascade_type}' cascades "
            f"from {len(trajectories)} trajectories."
        )
        return injected


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inject cascades into dataset")
    parser.add_argument("--input", required=True, help="Input JSONL trajectory file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument(
        "--cascade_type", required=True,
        choices=["retrieval", "inference", "context_poisoning",
                 "confidence_inflation"]
    )
    parser.add_argument("--n_samples", type=int, default=500)
    args = parser.parse_args()

    import os
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    injector = CascadeInjector(client)

    with open(args.input) as f:
        trajectories = [json.loads(line) for line in f]

    injected = injector.inject_dataset(
        trajectories, args.cascade_type, args.n_samples
    )

    with open(args.output, "w") as f:
        for traj in injected:
            f.write(json.dumps(traj) + "\n")

    print(f"Written {len(injected)} injected trajectories to {args.output}")
