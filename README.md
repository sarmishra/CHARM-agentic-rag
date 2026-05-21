# CHARM: Cascading Hallucination Aware Resolution and Mitigation

**Official implementation for the paper:**
> *Cascading Hallucination in Agentic RAG Pipelines: The CHARM Framework for Detection and Mitigation in Multi-Step Reasoning Systems*
> Saroj Mishra — [Venue, Year]

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)

---

## Overview

Multi-step agentic RAG pipelines are vulnerable to **cascading hallucination** — a failure mode where errors introduced at early pipeline stages propagate and amplify across successive reasoning steps, producing confident but factually incorrect final outputs.

CHARM is a modular, retrofit-friendly framework that detects and interrupts these cascades **without requiring architectural replacement** of the underlying pipeline.

### Four Detection Components

| Component | Acronym | Role | Cascade Types |
|---|---|---|---|
| Stage-Level Fact Verifier | SFV | NLI entailment scoring at each stage | Retrieval, Inference |
| Cross-Stage Consistency Tracker | CSCT | Embedding drift monitoring across trajectory | Inference, Context Poisoning |
| Confidence Propagation Monitor | CPM | Bayesian confidence anomaly detection | Confidence Inflation |
| Cascade Resolution Trigger | CRT | Aggregates signals; halts pipeline; triggers mitigation | All Types |

### Four Mitigation Patterns

| Pattern | Acronym | Mechanism | Overhead |
|---|---|---|---|
| Cascade Re-Retrieval | CRR | Fresh retrieval with refined query | Medium |
| Staged Confidence Thresholding | SCT | Stage-gate confidence checks | Low |
| Parallel Verification Agent | PVA | Independent parallel verification agent | High |
| Pipeline Rollback and Re-Execution | PRR | Rollback to last clean stage | Highest |

---

## Repository Structure

```
CHARM/
├── charm/                          # Core CHARM framework
│   ├── __init__.py
│   ├── framework.py                # Main CHARMFramework class
│   ├── sfv.py                      # Stage-Level Fact Verifier
│   ├── csct.py                     # Cross-Stage Consistency Tracker
│   ├── cpm.py                      # Confidence Propagation Monitor
│   └── crt.py                      # Cascade Resolution Trigger
│
├── mitigation/                     # Mitigation patterns
│   ├── __init__.py
│   ├── crr.py                      # M1: Cascade Re-Retrieval
│   ├── sct.py                      # M2: Staged Confidence Thresholding
│   ├── pva.py                      # M3: Parallel Verification Agent
│   └── prr.py                      # M4: Pipeline Rollback and Re-Execution
│
├── evaluation/                     # Evaluation harness
│   ├── __init__.py
│   ├── agentic_wrapper.py          # Agentic trajectory wrapper
│   ├── inject_cascades.py          # Four-method cascade injection protocol
│   ├── metrics.py                  # CDR, FPR, EPR, MSR, CDD, LO/s
│   └── run_evaluation.py           # Main evaluation script
│
├── data/
│   ├── adversarial/                # 200 annotated adversarial trajectories
│   │   ├── retrieval_cascades.jsonl
│   │   ├── inference_cascades.jsonl
│   │   ├── context_poisoning_cascades.jsonl
│   │   └── confidence_inflation_cascades.jsonl
│   └── README.md
│
├── configs/
│   ├── default.yaml                # Default CHARM configuration
│   ├── high_throughput.yaml        # Optimized for low latency (M2 only)
│   └── enterprise.yaml             # Maximum reliability (M3 + M4)
│
├── scripts/
│   ├── run_ablation.sh             # Reproduce ablation study (Table 5)
│   ├── run_baselines.sh            # Run all baseline comparisons
│   └── run_full_eval.sh            # Full evaluation across all datasets
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/sarmishra/CHARM-agentic-rag.git
cd CHARM-agentic-rag
pip install -r requirements.txt
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

---

## Quick Start

### Basic Usage: Wrap an Existing Pipeline

```python
from charm import CHARMFramework

# Initialize CHARM with default configuration
charm = CHARMFramework(
    sfv_threshold=0.72,       # NLI entailment threshold
    csct_drift_threshold=0.18, # Cosine similarity drift threshold
    cpm_temperature=1.4,       # Confidence calibration temperature
    crt_threshold=0.55,        # Cascade trigger threshold
    crt_weights=(0.4, 0.4, 0.2)  # SFV, CSCT, CPM weights
)

# Wrap your existing pipeline execution
pipeline_stages = ["query_formulation", "retrieval", "reasoning",
                   "tool_use", "synthesis"]

result = charm.run_pipeline(
    query="What is the GDP of the country that hosted the 2020 Olympics?",
    pipeline_stages=pipeline_stages,
    mitigation_strategy="CRR"  # M1: Cascade Re-Retrieval
)

print(result.final_output)
print(f"Cascade detected: {result.cascade_detected}")
print(f"Detection stage: {result.cascade_depth}")
print(f"Mitigation applied: {result.mitigation_applied}")
```

### Running the Full Evaluation

```bash
# Reproduce main results (Table 3 in the paper)
python evaluation/run_evaluation.py \
    --datasets hotpotqa musique 2wikimultihopqa adversarial \
    --baselines none selfcheckgpt ragas self_correction ever ircot \
    --model gpt-4o \
    --output_dir results/

# Reproduce ablation study (Table 5 in the paper)
bash scripts/run_ablation.sh

# Run all baselines
bash scripts/run_baselines.sh
```

---

## Configuration

### Default Configuration (`configs/default.yaml`)

```yaml
backbone_llm: gpt-4o
llm_temperature: 0.0
retriever: faiss
embedding_model: text-embedding-3-small

charm:
  sfv:
    model: cross-encoder/nli-deberta-v3-base
    threshold: 0.72
  csct:
    model: all-mpnet-base-v2
    drift_threshold: 0.18
  cpm:
    calibration: temperature_scaling
    temperature: 1.4
    fallback: nli_entailment_proxy
    calibration_samples: 500
  crt:
    threshold: 0.55
    weights:
      sfv: 0.4
      csct: 0.4
      cpm: 0.2

mitigation:
  default: CRR
  high_confidence_cascade: HITL  # Route to human review
```

---

## Reproducing Paper Results

### Step 1: Download Datasets

```bash
# HotpotQA
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json \
     -O data/hotpotqa_dev.json

# MuSiQue (download from https://github.com/StonyBrookNLP/musique)
# 2WikiMultiHopQA (download from https://github.com/Alab-NII/2wikimultihop)
```

### Step 2: Inject Cascades

```bash
python evaluation/inject_cascades.py \
    --dataset hotpotqa \
    --cascade_type retrieval \
    --input data/hotpotqa_dev.json \
    --output data/hotpotqa_retrieval_injected.jsonl \
    --n_samples 500
```

### Step 3: Run Evaluation

```bash
python evaluation/run_evaluation.py \
    --input data/hotpotqa_retrieval_injected.jsonl \
    --system charm \
    --config configs/default.yaml \
    --output results/hotpotqa_charm.json
```

### Step 4: Run Ablation

```bash
bash scripts/run_ablation.sh --dataset hotpotqa \
    --output results/ablation/
```

---

## Cascade Injection Protocol

The injection protocol creates four cascade types as described in the paper:

| Type | Method | Dataset |
|---|---|---|
| Retrieval | Top-1 doc replaced with counterfactual (GPT-4o generated) | HotpotQA |
| Inference | Misleading reasoning cue prepended at stage 2 | MuSiQue |
| Context Poisoning | Embedding-proximal adversarial passage inserted | 2WikiMultiHopQA |
| Confidence Inflation | Hedging language stripped from stage outputs | All (Adversarial Set) |

---

## Dataset

The `data/adversarial/` directory contains the 200 annotated adversarial
trajectories used in the paper, with the following fields:

```json
{
  "trajectory_id": "adv_001",
  "cascade_type": "retrieval",
  "injection_stage": 1,
  "query": "...",
  "stages": [
    {
      "stage_id": 1,
      "stage_name": "retrieval",
      "context_output": "...",
      "is_injected": true,
      "ground_truth_error_magnitude": 0.73
    }
  ],
  "ground_truth_answer": "...",
  "final_output_without_charm": "...",
  "cascade_detected_at_stage": null
}
```

---

## Citation

If you use CHARM in your research, please cite:

```bibtex
@article{mishra2025charm,
  title   = {Cascading Hallucination in Agentic {RAG} Pipelines: The {CHARM} 
             Framework for Detection and Mitigation in Multi-Step Reasoning Systems},
  author  = {Mishra, Saroj},
  journal = {[Venue]},
  year    = {2025}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
