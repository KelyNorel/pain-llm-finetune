# Pain LLM Fine-tuning

Local, HIPAA-compliant evaluation of small LLMs for chronic pain clinical scoring.

## Overview

Clinical monitoring of chronic pain currently relies on validated questionnaires 
administered at sparse intervals, or on large proprietary LLMs that process sensitive 
patient data via external APIs. This project evaluates whether small, locally-run LLMs 
can replicate the clinical scoring of Llama 405B from naturalistic patient narratives — 
without requiring condition-specific content and without sending data outside the clinic.

All processing runs fully local on Apple M4 — no data leaves the machine (HIPAA-compliant).
The 405B scores serve as ground truth, validated against clinical questionnaires 
in a submitted manuscript (Norel et al., under review).

## Data
Transcript data from University of Rochester (not included — HIPAA).
CLBP cohort only (n=67 participants).
- `data/transcripts/general/` — general interview section (TXT) — participants not discussing their condition
- `data/transcripts/dx/` — diagnostic section (TXT) — condition-specific content
- `data/transcripts/full/` — full interview (TXT, general + dx combined)

**Note:** Transcripts exceeding 2000 words are truncated before scoring. 
Longest transcript observed: 3156 words (subject 1248).

## Environment
- Apple M4, 24GB unified memory
- MLX — fine-tuning framework, optimized for Apple Silicon
- Ollama — local model serving
- ChromaDB — vector store for RAG

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/ingest.py  # build RAG knowledge base from literature PDFs
```

## Pipeline

### Scripts
| Script | Description |
|--------|-------------|
| `src/llm_scorer.py` | Baseline scoring via Ollama (2 prompts, 9 metrics) |
| `src/llm_scorer_rag.py` | RAG scoring — 1 prompt per metric with clinical literature context |
| `src/run_scoring.py` | Batch baseline scoring over all subjects |
| `src/run_scoring_rag.py` | Batch RAG scoring |
| `src/run_scoring_finetuned.py` | Batch scoring with fine-tuned model (MLX) |
| `src/compare_all.py` | Compare all conditions vs 405B ground truth (general section) |
| `src/compare_all_dx.py` | Compare baseline and RAG vs 405B ground truth (dx section) |
| `src/ingest.py` | Ingest clinical literature PDFs into ChromaDB |
| `src/prepare_finetune_data.py` | Prepare JSONL dataset for LoRA fine-tuning |

### Metrics (0.0–10.0, higher = worse)
Nine clinical metrics extracted per transcript, matching the published pipeline:
Physical_Pain, Emotional_Pain, Depression, poor_QoL, Anxiety, Catastrophizing, 
Rumination, Narrative_Fragmentation, Agency_Deficit

### RAG Knowledge Base
- v1: 19 papers (psychology, rumination, narrative, agency, chronic pain)
- v2: 30 papers (v1 + 11 pain-specific papers: pain assessment, biomarkers, NLP in pain)

---

## Results

### General Section: Agreement between Llama 3.2 3B and Llama 405B
Spearman r — higher values indicate stronger agreement with 405B ground truth  
(n=67 CLBP participants, general interview — participants not discussing their condition)

| Metric | Baseline | RAG-v1 | RAG-v2 | FT-v1 | FT-v2 | FT-v3 | FT-v4 |
|--------|----------|--------|--------|-------|-------|-------|-------|
| Physical_Pain | 0.369** | 0.367** | 0.345** | nan | -0.068 | nan | nan |
| Emotional_Pain | 0.343** | 0.290* | 0.274* | 0.261* | 0.152 | 0.261* | nan |
| Depression | 0.396*** | 0.375** | 0.328** | 0.386** | 0.354** | 0.464*** | nan |
| poor_QoL | 0.309* | 0.559*** | 0.559*** | -0.148 | 0.022 | 0.182 | nan |
| Anxiety | 0.302* | 0.425*** | 0.412*** | 0.293* | 0.108 | 0.067 | nan |
| Catastrophizing | 0.212 | 0.263* | 0.263* | 0.137 | 0.153 | -0.041 | nan |
| Rumination | 0.311* | 0.349** | 0.349** | -0.035 | 0.117 | -0.035 | nan |
| Narrative_Fragmentation | 0.340** | 0.347** | 0.347** | 0.202 | -0.028 | 0.268* | nan |
| Agency_Deficit | 0.306* | 0.326** | 0.326** | 0.158 | 0.130 | 0.138 | nan |

*p<0.05, **p<0.01, ***p<0.001  
nan = model output was constant (no variance), Spearman undefined  
RAG+FT not shown: model collapsed to constant output — fine-tuning overfit overrides RAG context

---

### Diagnostic Section (dx): Agreement between Llama 3.2 3B and Llama 405B
Spearman r — higher values indicate stronger agreement with 405B ground truth  
(n=67 CLBP participants, condition-specific interview section)  
Shown as validation only — condition-specific content is not available in real-world monitoring.

| Metric | Baseline | RAG-v1 |
|--------|----------|--------|
| Physical_Pain | 0.682*** | 0.600*** |
| Emotional_Pain | 0.599*** | 0.578*** |
| Depression | 0.572*** | 0.518*** |
| poor_QoL | 0.632*** | 0.534*** |
| Anxiety | 0.445*** | 0.491*** |
| Catastrophizing | 0.425*** | 0.082 |
| Rumination | 0.549*** | 0.527*** |
| Narrative_Fragmentation | 0.261* | 0.451*** |
| Agency_Deficit | 0.328** | 0.345** |

*p<0.05, **p<0.01, ***p<0.001  
Strong dx results confirm the model understands the clinical constructs — 
the challenge in general section is transcript content, not model capability.

---

## Key Findings

**1. Small LLMs with RAG capture clinical signal from non-condition-specific narratives**
The primary contribution: using only general conversation (not asking about symptoms), 
RAG + 3B model shows significant agreement with 405B ground truth on 8/9 metrics.
This enables scalable, low-burden longitudinal monitoring without condition-specific interviews.

**2. RAG improves over baseline in general section**
Retrieving relevant clinical literature as context improves agreement with 405B:
- poor_QoL: 0.309 → 0.559*** (largest improvement)
- Anxiety: 0.302 → 0.425***
- Catastrophizing: 0.212 (ns) → 0.263* (became significant)

**3. Knowledge base size beyond a threshold does not further improve results**
Expanding from 19 to 30 papers (RAG-v2) yields no improvement — the limiting factor 
is transcript content specificity, not knowledge base size.

**4. Fine-tuning with n=67 fails regardless of strategy**
All fine-tuning versions (v1–v4) produce constant or near-constant outputs for most metrics.
LoRA fine-tuning teaches surface memorization, not clinical inference.
The gap between 3B and 405B in Physical_Pain from non-specific narratives reflects 
a fundamental model capacity difference — not a training data or hyperparameter problem.
A larger base model (e.g. 70B) or substantially more labeled data would be needed.

**5. Concrete constructs are better captured than abstract cognitive ones**
In the general section, metrics with more explicit linguistic expression show stronger 
agreement with 405B (Physical_Pain r=0.369**, Depression r=0.396***), while abstract 
cognitive constructs requiring inference show weaker agreement (Catastrophizing r=0.212ns, 
Agency_Deficit r=0.306*). This pattern is consistent with the cognitive/emotional clustering 
identified in the original 405B analysis — small models capture emotional signal more readily 
than cognitive-evaluative constructs.

**6. dx section as validation**
Strong baseline results in dx (r~0.5-0.7) confirm the 3B model understands the 
clinical constructs — when content is explicit, no RAG needed. The challenge in the 
general section is transcript content specificity, not model capability.

---

## Fine-tuning Details

| Version | Dataset | Ground truth | Layers | LR | Val loss (start→end) | Notes |
|---------|---------|-------------|--------|-----|----------------------|-------|
| v1 | 67 (general) | 405b_general | 8 | 1e-5 | 2.792→2.298 | baseline FT, ordered split |
| v2 | 201 (general+full+dx) | 405b_general | 8 | 1e-5 | 2.964→2.521 | full duplicates general+dx |
| v3 | 134 (general+dx) | 405b_general | 8 | 1e-5 | 2.836→2.385 | wrong GT for dx |
| v4 | 134 (general+dx) | 405b_general + 405b_dx | 4 | 5e-6 | 2.986→2.565 (min 2.405 @iter150) | correct GT, fewer layers, lower LR |

**Method:** LoRA via MLX  
**Peak memory:** 6.6 GB (v1) / 14.0 GB (v2, v3) / 9.8 GB (v4) of 24 GB available

---

## Conclusions

**RAG-v1 > Baseline >> Fine-tuning (n=67) — for general section**

- With RAG, a small local LLM captures meaningful clinical signal from non-condition-specific 
  narratives — sufficient for longitudinal monitoring without large models or external APIs
- RAG helps when transcript content is non-specific; knowledge base size saturates quickly
- Fine-tuning with n=67 consistently fails — memorization instead of generalization
- Scaling to a larger model (70B) running on dedicated GPU hardware (e.g. Lambda workstation) 
  is the most promising path to closing the gap with 405B
- All processing is fully local and HIPAA-compliant (Apple M4, 24GB)

---

## Next Steps
1. Test with larger local model (Llama 70B) on dedicated GPU hardware (e.g. Lambda workstation)
