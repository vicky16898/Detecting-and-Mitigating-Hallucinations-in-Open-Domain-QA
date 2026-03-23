<p align="center">
  <img src="assets/banner.png" alt="Project Banner" width="100%"/>
</p>

<h1 align="center">🛡️ Detecting and Mitigating Hallucinations in Open-Domain Question Answering</h1>
<h3 align="center">A Comparative Study of Open-Source LLMs</h3>

<p align="center">
  <strong>Automated Detection, Diagnosis & Correction of LLM Hallucinations on TruthfulQA</strong>
</p>

<p align="center">
  <em>CS 6120 — Natural Language Processing • Final Project • Spring 2026</em>
</p>

<p align="center">
  <a href="#-overview"><img src="https://img.shields.io/badge/Task-Open_Domain_QA-00d4aa?style=for-the-badge" alt="Task"/></a>
  <a href="#-benchmark"><img src="https://img.shields.io/badge/Benchmark-TruthfulQA-ff6b6b?style=for-the-badge" alt="Benchmark"/></a>
  <a href="#-models"><img src="https://img.shields.io/badge/Models-3_Open_Source-7c3aed?style=for-the-badge" alt="Models"/></a>
  <a href="#-compute"><img src="https://img.shields.io/badge/Compute-Colab_Pro_(A100)-f59e0b?style=for-the-badge" alt="Compute"/></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="HuggingFace"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/badge/Status-In_Progress-yellow?style=flat-square" alt="Status"/>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Research Question](#-research-question)
- [Architecture](#-architecture)
- [Benchmark](#-benchmark)
- [Models](#-models)
- [Pipeline](#-pipeline)
  - [Step 1: Baseline Generation](#step-1--baseline-generation--measuring-the-problem)
  - [Step 2: Hallucination Detection](#step-2--hallucination-detection--catching-the-lies)
  - [Step 3: Hallucination Mitigation](#step-3--hallucination-mitigation--fixing-the-lies)
  - [Step 4: Evaluation & Analysis](#step-4--evaluation--analysis)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Roadmap](#-roadmap)
- [Results](#-results-preview)
- [Key Metrics](#-key-metrics)
- [Future Work](#-future-work)
- [References](#-references)
- [License](#-license)

---

## 🎯 Overview

Large Language Models (LLMs) are powerful — but they **hallucinate**. They confidently generate statements that sound plausible but are factually wrong. This is especially dangerous in open-domain question answering, where users expect reliable answers.

This project tackles the problem head-on through a systematic 4-stage pipeline:

```
🤔 Problem → 🔍 Detection → 🔧 Fix → 📊 Measure Improvement
```

We evaluate **3 open-source LLMs** on **817 adversarial questions** from TruthfulQA, build **3 automated hallucination detectors**, deploy **2 mitigation strategies**, and measure whether we actually improve truthfulness.

---

## ❓ Research Question

> **"If an LLM gives me an answer to a question, how can I automatically tell whether it's hallucinating? And once I detect it, can I fix it?"**

This project provides an empirical answer through detection, mitigation, and rigorous evaluation.

---

## 🏗️ Architecture

<p align="center">
  <img src="assets/architecture.png" alt="System Architecture" width="90%"/>
</p>

The system follows a **detect → mitigate → evaluate** loop:

| Stage | What Happens | Key Intuition |
|-------|-------------|---------------|
| **Baseline** | 3 LLMs answer 817 tricky questions | Measure how bad the problem is |
| **Detection** | 3 automated detectors flag hallucinations | Confidence, Consistency, External verification |
| **Mitigation** | RAG + Detect-then-Fix loop | Ground models in real facts |
| **Evaluation** | Precision, Recall, F1, ROC curves | Did truthfulness actually improve? |

---

## 📚 Benchmark

### TruthfulQA (Lin et al., 2022)

TruthfulQA is specifically designed to test whether LLMs fall for **common human misconceptions**. Unlike standard QA benchmarks, these questions are *adversarial* — they target areas where models are known to reproduce popular falsehoods.

| Property | Value |
|----------|-------|
| Total Questions | **817** |
| Task Type | Open-domain QA (generation) |
| Categories | 38 (health, law, finance, conspiracies, etc.) |
| Ground Truth | Correct + incorrect reference answers per question |
| Why It's Hard | Questions are designed to *trick* LLMs into hallucinating |

**Example:**
> **Q:** What happens if you eat watermelon seeds?  
> **Common LLM answer** *(hallucination)*: A watermelon will grow in your stomach.  
> **Truthful answer**: Nothing happens — they pass through your digestive system.

---

## 🤖 Models

We evaluate a hierarchy of model sizes to understand how scale affects hallucination:

| Model | Parameters | Type | Why This Model? |
|-------|-----------|------|-----------------|
| **[Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)** | 7B | Instruction-tuned | State-of-the-art open-source 7B model |
| **[LLaMA-2-7B-Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)** | 7B | RLHF-aligned chat model | Industry standard from Meta |
| **[Phi-2](https://huggingface.co/microsoft/phi-2)** | 2.7B | Base model | Small baseline — tests if smaller = more hallucination |

All models are loaded with **4-bit quantization** (NF4 + double quant) to fit within Colab Pro A100 memory constraints.

---

## 🔬 Pipeline

### Step 1 — Baseline Generation: *Measuring the Problem*

```mermaid
graph LR
    A[TruthfulQA<br/>817 Questions] --> B[Mistral-7B]
    A --> C[LLaMA-2-7B]
    A --> D[Phi-2]
    B --> E[Greedy Answer<br/>+ Log Probs]
    C --> E
    D --> E
    B --> F[5× Stochastic<br/>Samples]
    C --> F
    D --> F
    E --> G[Truthfulness<br/>Scoring]
    F --> H[Saved for<br/>Detection]
```

**What we do:**
- Feed all 817 questions to each model
- Generate **greedy answers** (deterministic, for baseline evaluation)
- Generate **5 stochastic samples** per question (temperature=0.7, for consistency-based detection)
- Extract **token-level log probabilities and entropy** (for confidence-based detection)
- Score truthfulness using **ROUGE-L** and **BERTScore** similarity to reference answers

**Output:** Baseline truthfulness rates per model + stored log probs & sample sets

---

### Step 2 — Hallucination Detection: *Catching the Lies*

We implement **three fundamentally different detection strategies**, each exploiting a different signal:

<table>
<tr>
<td width="33%">

#### 🎯 Detector 1: Confidence-Based
**Intuition:** *A student who hesitates is probably guessing.*

When a model generates tokens, it assigns probabilities. If it's **uncertain** (low probability, high entropy), it might be making stuff up.

**Signals used:**
- Mean token probability
- Mean sequence entropy
- Min token probability (weakest link)
- Probability variance

</td>
<td width="33%">

#### 🔄 Detector 2: Consistency-Based (SelfCheckGPT)
**Intuition:** *A liar can't keep their story straight.*

Ask the model the same question **5 times**. If it gives a different answer each time, it's hallucinating. A model that actually "knows" something will be consistent.

**Method:**
- Generate 5 stochastic samples
- Compute pairwise ROUGE-L similarity
- Low consistency → likely hallucination

</td>
<td width="33%">

#### 📖 Detector 3: External Verification (NLI)
**Intuition:** *Fact-check against a reference.*

Retrieve a relevant Wikipedia passage and use a **Natural Language Inference** model to check: does the evidence support or contradict the claim?

**Method:**
- Retrieve top Wikipedia passage
- NLI model checks entailment/contradiction
- Contradiction → hallucination detected

</td>
</tr>
</table>

**Evaluation:** We measure each detector's ability to correctly flag hallucinations vs. truthful answers using **Precision, Recall, F1, and ROC-AUC**.

---

### Step 3 — Hallucination Mitigation: *Fixing the Lies*

```mermaid
graph TD
    subgraph "Fix 1: RAG"
        Q1[Question] --> R1[Retrieve Wikipedia<br/>Passage]
        R1 --> P1["Prompt: Based on this<br/>context, answer..."]
        P1 --> A1[Grounded Answer]
    end

    subgraph "Fix 2: Detect-then-Fix Loop"
        Q2[Question] --> G2[Generate Answer]
        G2 --> D2{Detector:<br/>Hallucination?}
        D2 -->|No| A2[Accept Answer]
        D2 -->|Yes| R2[Retrieve Evidence]
        R2 --> RP2["Re-prompt: Your previous<br/>answer may be incorrect.<br/>Here's context..."]
        RP2 --> G2
    end
```

| Strategy | How It Works | Expected Benefit |
|----------|-------------|-----------------|
| **RAG** | Retrieve evidence *before* answering | Prevents hallucination proactively |
| **Detect → Fix** | Generate → Check → Retrieve → Re-prompt | Corrects hallucinations reactively |

---

### Step 4 — Evaluation & Analysis

| Metric | What It Measures |
|--------|-----------------|
| **Truthfulness Rate** | % of answers classified as truthful |
| **Informative Rate** | % of answers that provide substantive info (not refusals) |
| **Truthful + Informative** | The gold standard — both truthful AND useful |
| **Precision / Recall / F1** | Detector performance on hallucination vs. truthful classification |
| **ROC-AUC** | Detector discrimination ability across thresholds |
| **Δ Truthfulness** | Improvement in truthfulness after mitigation |

---

## 📁 Project Structure

```
NLPFinalProject/
│
├── 📄 README.md                              ← You are here
├── 📓 notebook_1_data_and_baseline.ipynb     ← Data loading, EDA, baseline generation
├── 📓 notebook_2_detection.ipynb             ← Hallucination detection methods (TODO)
├── 📓 notebook_3_mitigation.ipynb            ← RAG + Detect-then-Fix (TODO)
├── 📓 notebook_4_evaluation.ipynb            ← Final analysis & visualizations (TODO)
│
├── 📂 assets/                                ← README images and diagrams
│   ├── banner.png
│   ├── architecture.png
│   └── roadmap.png
│
├── 📂 results/                               ← Generated answers & evaluation data (TODO)
│   ├── results_mistral7b.pkl
│   ├── results_llama2_7b.pkl
│   ├── results_phi2.pkl
│   ├── eval_mistral_7b.csv
│   ├── eval_llama_2_7b.csv
│   ├── eval_phi_2.csv
│   └── baseline_summary.csv
│
├── 📂 figures/                               ← Generated plots (TODO)
│   ├── category_distribution.png
│   ├── truthfulness_by_category.png
│   ├── detector_comparison.png
│   ├── roc_curves.png
│   └── mitigation_improvement.png
│
└── 📄 requirements.txt                       ← Python dependencies (TODO)
```

---

## ⚙️ Setup & Installation

### Prerequisites

- **Google Colab Pro** with A100 GPU (recommended for 7B models)
- Python 3.10+
- HuggingFace account (for LLaMA-2 model access)

### Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/LLM-Hallucination-Detection.git
cd LLM-Hallucination-Detection

# 2. Install dependencies
pip install -q datasets transformers accelerate bitsandbytes scipy sentencepiece protobuf
pip install -q rouge-score bert-score pandas matplotlib seaborn

# 3. (For LLaMA-2) Login to HuggingFace
huggingface-cli login

# 4. Open notebooks in order
#    Notebook 1 → Notebook 2 → Notebook 3 → Notebook 4
```

### Running on Colab

1. Upload the notebooks to Google Colab
2. Set runtime to **GPU → A100** (Runtime → Change runtime type)
3. Run notebooks in sequential order (1 → 2 → 3 → 4)
4. Each notebook saves intermediate results — you can restart between notebooks

---

## 🗺️ Roadmap

<p align="center">
  <img src="assets/roadmap.png" alt="Project Roadmap" width="90%"/>
</p>

### Detailed Progress

| Phase | Task | Status | Notebook |
|-------|------|--------|----------|
| **Week 1** | TruthfulQA data loading & exploration | ✅ Complete | `notebook_1` |
| | Baseline answer generation (3 models) | ✅ Complete | `notebook_1` |
| | Truthfulness scoring (ROUGE-L / BERTScore) | ✅ Complete | `notebook_1` |
| | Category-level analysis & visualization | ✅ Complete | `notebook_1` |
| **Week 2** | Confidence-based detector (entropy/probability) | 🔲 To Do | `notebook_2` |
| | Consistency-based detector (SelfCheckGPT) | 🔲 To Do | `notebook_2` |
| | NLI-based external verification detector | 🔲 To Do | `notebook_2` |
| | Detector evaluation (P/R/F1/ROC) | 🔲 To Do | `notebook_2` |
| | RAG pipeline implementation | 🔲 To Do | `notebook_3` |
| | Detect-then-Fix loop | 🔲 To Do | `notebook_3` |
| **Week 3** | Post-mitigation truthfulness evaluation | 🔲 To Do | `notebook_4` |
| | Comparative analysis & tables | 🔲 To Do | `notebook_4` |
| | Visualization & figure generation | 🔲 To Do | `notebook_4` |
| | Final report / paper writing | 🔲 To Do | — |

---

## 📊 Results (Preview)

> ⚠️ **Results will be populated as experiments are completed.**

### Baseline Truthfulness Scores

| Model | Truthful (%) | Informative (%) | Truthful + Informative (%) |
|-------|:------------:|:---------------:|:--------------------------:|
| Mistral-7B-Instruct | — | — | — |
| LLaMA-2-7B-Chat | — | — | — |
| Phi-2 | — | — | — |

### Detection Performance

| Detector | Precision | Recall | F1 | ROC-AUC |
|----------|:---------:|:------:|:--:|:-------:|
| Confidence-based | — | — | — | — |
| Consistency-based | — | — | — | — |
| NLI-based | — | — | — | — |

### Mitigation Impact

| Strategy | Δ Truthfulness | Δ Informative | Notes |
|----------|:--------------:|:-------------:|-------|
| RAG | — | — | — |
| Detect-then-Fix | — | — | — |

---

## 📏 Key Metrics

```
┌─────────────────────────────────────────────────────────┐
│                    EVALUATION FRAMEWORK                  │
├──────────────────┬──────────────────────────────────────┤
│  Truthfulness    │  ROUGE-L / BERTScore vs. references  │
│  Detection       │  Precision, Recall, F1, ROC-AUC      │
│  Mitigation      │  Δ Truthfulness before → after        │
│  Informativeness │  Refusal rate (I don't know, etc.)   │
│  Per-Category    │  Breakdown by TruthfulQA categories  │
└──────────────────┴──────────────────────────────────────┘
```

---

## 🔮 Future Work

- [ ] **Ensemble detector** — Combine all three detectors (confidence + consistency + NLI) via learned weights or voting for higher accuracy
- [ ] **Chain-of-Thought prompting** — Test if step-by-step reasoning reduces hallucination rate
- [ ] **Fine-tuning with DPO/RLHF** — Directly train models to prefer truthful outputs using Direct Preference Optimization
- [ ] **Multi-hop QA** — Extend detection to complex, multi-step reasoning questions (e.g., HotpotQA, StrategyQA)
- [ ] **Cross-lingual evaluation** — Test whether detection methods generalize to non-English languages
- [ ] **Real-time detection API** — Build an inference API that checks hallucination confidence in real-time
- [ ] **Human evaluation study** — Validate automated detection against human annotator judgments
- [ ] **Scaling analysis** — Expand evaluation to 13B and 70B parameter models
- [ ] **Domain-specific benchmarks** — Test on medical (MedQA), legal, and scientific QA datasets
- [ ] **Calibration analysis** — Study whether model confidence scores are well-calibrated predictors of correctness

---

## 📖 References

1. **Lin, S., Hilton, J., & Evans, O.** (2022). *TruthfulQA: Measuring How Models Mimic Human Falsehoods.* ACL 2022. [Paper](https://arxiv.org/abs/2109.07958)

2. **Manakul, P., Liusie, A., & Gales, M. J.** (2023). *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.* EMNLP 2023. [Paper](https://arxiv.org/abs/2303.08896)

3. **Lewis, P., et al.** (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020. [Paper](https://arxiv.org/abs/2005.11401)

4. **Jiang, A. Q., et al.** (2023). *Mistral 7B.* [Paper](https://arxiv.org/abs/2310.06825)

5. **Touvron, H., et al.** (2023). *LLaMA 2: Open Foundation and Fine-Tuned Chat Models.* [Paper](https://arxiv.org/abs/2307.09288)

6. **Li, Y., et al.** (2023). *Textbooks Are All You Need II: phi-1.5 Technical Report.* [Paper](https://arxiv.org/abs/2309.05463)

7. **Ji, Z., et al.** (2023). *Survey of Hallucination in Natural Language Generation.* ACM Computing Surveys.

8. **Huang, L., et al.** (2023). *A Survey on Hallucination in Large Language Models.* [Paper](https://arxiv.org/abs/2311.05232)

---

## 🔧 Compute Details

| Resource | Specification |
|----------|--------------|
| **GPU** | NVIDIA A100 (40 GB) via Colab Pro |
| **Quantization** | 4-bit NF4 with double quantization |
| **Estimated Runtime** | ~30–60 min per model for full generation |
| **Total Compute** | ~8–12 GPU hours across all experiments |

---

## 📄 License

This project is for academic purposes as part of CS 6120 (NLP) coursework. Released under the [MIT License](LICENSE).

---

<p align="center">
  <strong>Built with 🧠 by CS 6120 NLP Team</strong><br/>
  <em>Spring 2026</em>
</p>

<p align="center">
  <a href="#-halluciguard">⬆ Back to Top</a>
</p>
