# Context-Aware Kitsune IDS

Context-Aware Kitsune IDS is a research implementation of an intrusion-detection pipeline that extends Kitsune-style online anomaly detection with contextual risk assessment, adaptive decision thresholds, attack classification, and actionable counterfactual explanations.

## Research motivation

Network anomaly scores alone do not express operational risk. The same score can merit a different response depending on traffic context, learned behavioural profiles, and the consequences of false positives or false negatives. This project studies a context-aware decision pipeline that turns anomaly signals into explainable, policy-relevant actions.

## Architecture overview

```text
PCAP traffic
    |
    v
Kitsune feature extraction and anomaly score
    |
    v
Context features + traffic profile + risk engine
    |
    v
Adaptive thresholding
    |
    v
Context-aware decision layer
    |---------------------------|
    v                           v
normal traffic          attack classification
                                |
                                v
                    counterfactual explanations
```

The anomaly detector is evaluated before classification and explanation. Classification and counterfactual generation are reserved for traffic that the decision layer identifies as anomalous.

## Features

- **Kitsune-based anomaly detection** using the bundled KitNET implementation.
- **Adaptive thresholding** for changing traffic conditions and feedback-aware calibration.
- **Context-aware decision layer** that combines anomaly, contextual, and profile-derived risk signals.
- **Attack classification** for suspicious traffic after anomaly gating.
- **Counterfactual explanations** that identify changes associated with a different decision outcome.

## Repository layout

- `main_ids.py` — primary IDS pipeline entry point.
- `kitsune_wrapper.py`, `feature_extractor.py` — Kitsune integration and feature handling.
- `adaptive_threshold.py`, `risk_engine.py`, `context_features.py` — adaptive/contextual risk components.
- `decision_layer/` — severity and response policy logic.
- `counterfactual_engine/`, `explain_pipeline.py` — explanation components.
- `feedback/` — feedback, drift, and adaptive-learning utilities.
- `evaluation/`, `experiments/` — reproducible evaluation and experiment code.
- `paper/` — manuscript sources and paper figures.
- `profiles/` — example traffic profiles.

## Installation

Use Python 3.10 or newer and create an isolated environment outside version control.

```bash
git clone https://github.com/benjari2004-hash/Context-Aware-Kitsune-IDS.git
cd Context-Aware-Kitsune-IDS
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the IDS against a local packet capture:

```bash
python main_ids.py --pcap /path/to/capture.pcap --mode NORMAL
```

Available operating modes are `NORMAL`, `SENSITIVE`, and `STRICT`. See `python main_ids.py --help` for threshold, profile, and feedback options.

Run the UNSW-NB15 evaluation with a locally obtained CSV:

```bash
python evaluation/run_unsw_experiment.py \
  --csv /path/to/UNSW_NB15_training-set.csv \
  --results evaluation/results
```

Generated metrics, plots, logs, and result files are intentionally ignored by Git.

## Datasets

Datasets and packet captures are not distributed with this repository. Obtain them directly from their original providers, store them outside the repository or in an ignored local directory, and pass their paths through the relevant command-line options. The evaluation modules include loaders and preprocessing code, but no raw traffic, CSV, archive, or capture data is committed.

## Experimental evaluation

The `evaluation/` package supports reproducible UNSW-NB15 experiments, baseline comparisons, preprocessing, metrics, and experiment logging. The `experiments/` package contains scripts for feedback behaviour, false-positive injection, ablation, drift, and counterfactual-quality studies. Run outputs are generated locally and excluded from version control so that the repository contains code, profiles, and paper sources rather than derived artefacts.

## Research use

This repository is intended for research and evaluation. Validate configurations, thresholds, and response policies in a controlled environment before using any result in an operational security workflow.
