# 📘 Project Playbook: Hallucination Detection Enhancement (2026)

This document is the definitive guide for the **MIND Framework Upgrade**. It is designed for Google Colab and focuses on our implementation of **Llama 3.1 8B** and **Multi-Layer Latent Dynamics**.

---

## 🧐 The Project Thesis (For Submission)

**Goal:** To detect hallucinations in real-time by modeling the *internal state transitions* of an LLM, specifically targeting the new **Llama 3.1 8B** architecture.

### Research Novelty
1.  **Modern LLM Support**: Fully integrated Llama 3.1 (base & instruct) with a universal `find_answer_start` algorithm for robust chat-template parsing.
2.  **Multi-Layer Latent Dynamics**: We moved beyond simple layer averaging (MIND 2024) to a **3-channel feature strategy**:
    - **Concatenation**: Capturing discrete signals from 5 specific depths (Layers 1, 8, 16, 24, 32).
    - **Inter-Layer Deltas**: We compute the vector difference between layer pairs (e.g., `L32 - L24`). A high delta suggests high internal disagreement—a precursor to hallucination.
    - **Sequence Mean-Pooling**: Tracking the high-level semantic "confidence" across the generated text.
3.  **GPT-4 Automated Labeling**: Implemented an automated factual-labeling pipeline using the OpenAI Batch API, eliminating the need for manual human annotation.

---

## 🔑 Authentication Setup

### 1. HuggingFace (Model Access)
Llama 3.1 is a **gated model**. 
1. Request access at [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B).
2. Create a **'Read'** token under **Settings -> Access Tokens**.

### 2. OpenAI (Automated Labeling)
1. Generate an API key at [OpenAI Platform](https://platform.openai.com/api-keys).

### 🛡️ Secure Setup in Colab
1. In Colab, click the **Key icon** (🔑) in the left sidebar.
2. Add `HF_TOKEN` and `OPENAI_API_KEY` as secrets and enable "Notebook access".

---

## 🚀 Execution Steps (Google Colab)

### 1. Setup & Clone
```python
# Clone from the public repository (vicky-testing branch)
!git clone -b vicky-testing https://github.com/vicky16898/Detecting-and-Mitigating-Hallucinations-in-Open-Domain-QA.git
%cd Detecting-and-Mitigating-Hallucinations-in-Open-Domain-QA
!pip install -r requirements.txt
!python -m spacy download en_core_web_sm

# Load tokens
from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')
os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
```

### 2. Train the Hallucination Detector
```bash
# A. Generate Wiki training samples
!python src/generate_data.py --model_family llama3base --model_type 8 --gpu 0

# B. Extract our 40k-dim Multi-Layer features
!python src/generate_hd.py --model_family llama3base --model_type 8 --strategy multi_layer --gpu 0

# C. Train the MLP Classifier
!python src/train.py --model_name llama3base8b --strategy multi_layer --device cuda:0
```

### 3. Automated HELM Evaluation
```bash
# A. Generate Evaluation responses
!python src/generate_helm_data.py --model_family llama3base --model_type 8 --gpu 0

# B. Submit for GPT-4 Auto-Labeling
!python src/label_helm_data.py --model_name llama3base8b --mode submit
# Once batch status is 'completed' (check in OpenAI Dashboard or via --mode status):
!python src/label_helm_data.py --model_name llama3base8b --mode fetch

# C. Get Final AUC/Accuracy Scores
!python src/generate_hd_for_helm.py --model_family llama3base --model_type 8 --strategy multi_layer --gpu 0
!python src/detection_score.py --strategy multi_layer --gpu 0
```
