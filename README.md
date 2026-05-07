# Pain LLM Fine-tuning

Local, HIPAA-compliant fine-tuning of LLMs on chronic pain interview data.

## Data
Clinical and transcript data from University of Rochester (not included).
- `data/clinical/` — clinical variables (CSV)
- `data/transcripts/full/` — full interviews (TXT)
- `data/transcripts/general/` — general section (TXT)
- `data/transcripts/dx/` — diagnostic section (patients only, TXT)

## Environment
- Apple M4, 24GB unified memory
- MLX for fine-tuning (Apple Silicon optimized)
- Ollama for local model serving

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Results

### Baseline: Small Model (Llama 3.2 3B) vs Large Model (Llama 405B)

Spearman correlations between 3B and 405B scores on CLBP cohort (n=67):

| Metric | r | p-value |
|--------|---|---------|
| Physical_Pain | 0.369 | 0.002 ** |
| Emotional_Pain | 0.343 | 0.005 ** |
| Depression | 0.396 | 0.001 *** |
| poor_QoL | 0.309 | 0.011 * |
| Anxiety | 0.302 | 0.013 * |
| Catastrophizing | 0.212 | 0.085 |
| Rumination | 0.311 | 0.010 * |
| Narrative_Fragmentation | 0.340 | 0.005 ** |
| Agency_Deficit | 0.306 | 0.012 * |

**Hypothesis:** Fine-tuning improves small LLM replication of large model clinical 
scoring, particularly for cognitively complex constructs (Catastrophizing, Agency Deficit).

## Fine-tuning (LoRA)

**Method:** LoRA (Low-Rank Adaptation) via MLX
- Base model: Llama 3.2 3B Instruct (4-bit quantized)
- Trainable parameters: 0.108% (3.47M / 3,212M)
- Layers: last 8 transformer layers (q_proj, v_proj)
- Iterations: 100, batch size: 1, learning rate: 1e-5
- Train/val split: 80/20 (53 train, 14 val)
- Peak memory: 6.66 GB (of 24 GB available, M4)

**Training loss:**
| Iter | Train Loss | Val Loss |
|------|-----------|---------|
| 1    | -         | 2.792   |
| 10   | 2.712     | -       |
| 20   | 2.408     | -       |
| 30   | 2.108     | -       |
| 40   | 2.054     | -       |
| 50   | 1.952     | -       |
| 60   | 2.059     | -       |
| 70   | 1.873     | -       |
| 80   | 1.866     | -       |
| 90   | 1.915     | -       |
| 100  | 2.027     | 2.298   |

**Notes:**
- Val loss dropped from 2.792 → 2.298 (improvement: 0.494)
- Train/val gap of 0.271 suggests moderate overfitting (expected with n=67)
- Split was ordered by Study ID, not randomized — to fix in next iteration
- Adapters saved to models/lora_adapters/adapters.safetensors

## Comparison: Baseline vs RAG vs Fine-tuned (Spearman r vs 405B ground truth)

| Metric | Baseline | RAG | Fine-tuned |
|--------|----------|-----|------------|
| Physical_Pain | 0.369** | 0.367** | nan |
| Emotional_Pain | 0.343** | 0.290* | 0.261* |
| Depression | 0.396*** | 0.375** | 0.386** |
| poor_QoL | 0.309* | 0.559*** | -0.148 |
| Anxiety | 0.302* | 0.425*** | 0.293* |
| Catastrophizing | 0.212 | 0.263* | 0.137 |
| Rumination | 0.311* | 0.349** | -0.035 |
| Narrative_Fragmentation | 0.340** | 0.347** | 0.202 |
| Agency_Deficit | 0.306* | 0.326** | 0.158 |

*p<0.05, **p<0.01, ***p<0.001

## Key Findings

- **RAG > Baseline** on most metrics, especially poor_QoL (0.309→0.559) and Anxiety (0.302→0.425)
- **RAG made Catastrophizing significant** (0.212ns→0.263*) — clinical literature helps cognitively complex constructs
- **Fine-tuning with n=67 overfit severely** — Physical_Pain constant (7.2) for all subjects
- **Next steps:** Data augmentation (full + general + dx transcripts) to increase training set ~3x before fine-tuning

## Conclusions

RAG with domain-specific clinical literature outperforms vanilla prompting for most metrics.
Fine-tuning requires more data — augmentation with all transcript types planned for v2.
All processing is fully local and HIPAA-compliant (Apple M4, 24GB).

## Fine-tuning v2 (LoRA with data augmentation)

**Changes vs v1:**
- Dataset: 201 examples (3x — general + full + dx transcripts)
- Shuffled with random.seed(42) — randomized train/val split
- Iters: 200, batch-size: 2
- Note: transcripts >2048 tokens truncated (max observed: 4985 tokens)

**Training:**
- Val loss: 2.964 → 2.521
- Train/val gap: 0.094 (vs 0.271 in v1) — less overfit ✅
- Peak memory: 14.0 GB

## Final Comparison (Spearman r vs 405B ground truth, n=67)

| Metric | Baseline | RAG | FT-v1 | FT-v2 |
|--------|----------|-----|-------|-------|
| Physical_Pain | 0.369** | 0.367** | nan | -0.068 |
| Emotional_Pain | 0.343** | 0.290* | 0.261* | 0.152 |
| Depression | 0.396*** | 0.375** | 0.386** | 0.354** |
| poor_QoL | 0.309* | 0.559*** | -0.148 | 0.022 |
| Anxiety | 0.302* | 0.425*** | 0.293* | 0.108 |
| Catastrophizing | 0.212 | 0.263* | 0.137 | 0.153 |
| Rumination | 0.311* | 0.349** | -0.035 | 0.117 |
| Narrative_Fragmentation | 0.340** | 0.347** | 0.202 | -0.028 |
| Agency_Deficit | 0.306* | 0.326** | 0.158 | 0.130 |

*p<0.05, **p<0.01, ***p<0.001

## Overall Conclusion

**RAG > Baseline > FT-v1 > FT-v2**

- RAG with domain-specific clinical literature is the most effective approach
- Fine-tuning with n=67 (even with 3x augmentation) leads to overfitting
- Key insight: for small LLMs in specialized clinical domains, retrieval-augmented generation outperforms parameter fine-tuning when labeled data is scarce
- Next: RAG + Fine-tuned combination to be tested

## Fine-tuning v3 (LoRA — general + dx only, no duplication)

**Changes vs v2:**
- Removed `full` transcripts (full = general + dx, was duplicating data)
- Dataset: 134 examples (general + dx only)
- Train/val: 107/27, shuffled with random.seed(42)

**Training:**
- Val loss: 2.836 → 2.385 (best val loss across all versions)
- Train/val gap: 0.304
- Peak memory: 14.0 GB

## Updated Comparison (Spearman r vs 405B ground truth, n=67)

| Metric | Baseline | RAG | FT-v1 | FT-v2 | FT-v3 |
|--------|----------|-----|-------|-------|-------|
| Physical_Pain | 0.369** | 0.367** | nan | -0.068 | nan |
| Emotional_Pain | 0.343** | 0.290* | 0.261* | 0.152 | 0.261* |
| Depression | 0.396*** | 0.375** | 0.386** | 0.354** | 0.464*** |
| poor_QoL | 0.309* | 0.559*** | -0.148 | 0.022 | 0.182 |
| Anxiety | 0.302* | 0.425*** | 0.293* | 0.108 | 0.067 |
| Catastrophizing | 0.212 | 0.263* | 0.137 | 0.153 | -0.041 |
| Rumination | 0.311* | 0.349** | -0.035 | 0.117 | -0.035 |
| Narrative_Fragmentation | 0.340** | 0.347** | 0.202 | -0.028 | 0.268* |
| Agency_Deficit | 0.306* | 0.326** | 0.158 | 0.130 | 0.138 |

## Key Insight — FT-v3
- Depression: 0.464*** — best across ALL conditions including RAG
- Narrative_Fragmentation: 0.268* — recovered significance
- Physical_Pain still constant (7.2) — model memorizes most common value
- RAG still wins overall, but FT-v3 wins on Depression specifically
- Suggests RAG + FT-v3 combination may be optimal
