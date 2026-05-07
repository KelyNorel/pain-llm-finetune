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
