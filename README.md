# model_regression_detection
Model Regression Detection

# Model Regression Detection System (MRDS)

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](#)

An enterprise-grade, automated **Model Regression Detection System** for machine learning models and LLM/GenAI applications. **MRDS** continuously monitors, tests, and evaluates candidate models against baseline metrics, catching performance drops, data drift, semantic degradation, and latency spikes before they impact production.

---

## 📋 Table of Contents
- [Architecture](#-architecture)
- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Usage & Examples](#-usage--examples)
  - [1. Tabular & Classical ML Quality Gate](#1-tabular--classical-ml-quality-gate)
  - [2. GenAI & LLM Output Regression Testing](#2-genai--llm-output-regression-testing)
  - [3. CI/CD Pipeline Integration](#3-cicd-pipeline-integration)
- [Supported Frameworks & Ecosystem](#-supported-frameworks--ecosystem)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🏗 Architecture

The system evaluates candidate models against reference baseline models or golden datasets across a four-stage verification pipeline:

1. Data Ingestion
   - Baseline datasets
   - Candidate predictions
   - Production traces

2. Metrics Engine
   - Drift detection (PSI / KS)
   - Model performance checks (F1 / MSE / accuracy)
   - LLM-as-a-Judge evaluation

3. Quality Gates
   - Static thresholds
   - Dynamic baselines
   - Pass/fail decisions

4. Action & Alerting
   - Fail CI/CD builds
   - Trigger auto-rollback
   - Notify Slack / PagerDuty



---

## ✨ Key Features

* **Statistical Drift Detection**: Automatically compute Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests to detect feature and target drift.
* **Classical ML Evaluation**: Regression metrics ($\text{MSE}$, $\text{MAPE}$, $R^2$) and classification metrics ($F_1$-score, Precision/Recall, ROC-AUC) regression tracking.
* **LLM & RAG Quality Verification**: Semantic embedding similarity checks, factual alignment, context recall, and automated LLM-as-a-Judge evaluations.
* **Production Operational Metrics**: Track token usage, p95/p99 latency regression, JSON schema adherence, and API error rates.
* **CI/CD Native**: Direct integration with Pytest, GitHub Actions, GitLab CI, and MLflow model registries.

---

## 📁 Repository Structure

```text
model-regression-detection/
├── .github/
│   └── workflows/
│       └── model_ci.yml         # CI/CD pipeline definition for automated testing
├── configs/
│   └── threshold_config.yaml    # Quality gate thresholds and metric definitions
├── data/
│   ├── baseline_preds.csv       # Sample baseline prediction artifacts
│   └── candidate_preds.csv      # Sample candidate model outputs
├── src/
│   ├── metrics/
│   │   ├── classical.py         # Statistical & tabular ML metrics (PSI, KS, MSE, F1)
│   │   └── llm_eval.py          # GenAI evaluators (Embedding distance, Judge scoring)
│   ├── quality_gate.py          # Decision engine comparing candidate vs baseline
│   └── utils.py                 # Data loading and report generation
├── tests/
│   └── test_regression_gate.py  # Pytest suite for CI/CD model verification
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── main.py                      # CLI entry point for evaluation
└── LICENSE                      # Project license
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Git

### Installation

```bash
git clone https://github.com/your-username/model-regression-detection.git
cd model-regression-detection
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
# On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Define acceptable quality gate thresholds inside `configs/threshold_config.yaml`:

```yaml
classical_ml:
  max_mse_increase_pct: 5.0      # Fail if Candidate MSE is >5% higher than Baseline
  min_f1_score_drop: 0.02        # Fail if Candidate F1 drops by >0.02
  ks_p_value_alpha: 0.05          # Fail if distribution drift p-value < 0.05
  max_psi_threshold: 0.2           # Alert if PSI >= 0.2

llm_eval:
  min_semantic_similarity: 0.85   # Minimum cosine similarity between outputs
  max_latency_p95_ms: 1200        # Max allowable 95th percentile latency in ms
  max_hallucination_rate: 0.03    # Max allowable hallucination rate threshold
```

---

## 💻 Usage & Examples

### 1. Tabular & Classical ML Quality Gate

Evaluate candidate model predictions against baseline predictions programmatically:

```python
from src.quality_gate import ModelQualityGate

# Initialize Quality Gate with config
gate = ModelQualityGate(config_path="configs/threshold_config.yaml")

# Run assessment on baseline vs candidate datasets
report = gate.evaluate_tabular_model(
    baseline_path="data/baseline_preds.csv",
    candidate_path="data/candidate_preds.csv",
    target_col="actual_value",
    pred_col="predicted_value"
)

# Output evaluation verdict
if report.passed:
    print("✅ Candidate model PASSED all regression checks.")
else:
    print(f"❌ Regression Detected! Failures: {report.failure_reasons}")
```

### 2. GenAI & LLM Output Regression Testing

Run semantic similarity and LLM-as-a-Judge evaluations on candidate prompt/model responses:

```python
from src.metrics.llm_eval import evaluate_llm_regression

baseline_responses = [
    "To reset your password, navigate to Settings > Security > Reset Password."
]
candidate_responses = [
    "You can change your password by going to the Settings page and selecting Security."
]

metrics = evaluate_llm_regression(
    baseline_outputs=baseline_responses,
    candidate_outputs=candidate_responses,
    embedding_model="text-embedding-3-small"
)

print(f"Semantic Similarity Score: {metrics['cosine_similarity']:.4f}")
print(f"Passes Quality Threshold: {metrics['is_acceptable']}")
```

### 3. CI/CD Pipeline Integration

Execute the test suite via `pytest` to block automated deployments on detected regressions:

```bash
pytest tests/test_regression_gate.py --verbose
```

Example GitHub Actions workflow:

```yaml
name: Model Regression Detection Pipeline

on:
  push:
    branches: [ main, release/* ]
  pull_request:
    branches: [ main ]

jobs:
  evaluate-regression:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run Model Quality Gate
        run: |
          pytest tests/test_regression_gate.py
```

---

## 🛠 Supported Frameworks & Ecosystem

This system integrates with leading tools across the ML and LLM monitoring ecosystem:

| Category | Supported Tools / Frameworks |
| --- | --- |
| MLOps & Tracking | MLflow, Weights & Biases, ClearML |
| Drift & Testing | Evidently AI, Deepchecks, whylogs |
| LLM & RAG Evaluation | RAGAS, TruLens, Promptfoo |
| CI/CD & Deployment | GitHub Actions, GitLab CI, Argo Workflows |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/NewMetric`
3. Commit your changes: `git commit -m 'Add new metric evaluator'`
4. Push to the branch: `git push origin feature/NewMetric`
5. Open a Pull Request.

Please ensure all existing test suites pass before submitting a PR:

```bash
pytest
```

---

## 📜 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
















