import json
import pandas as pd
from pathlib import Path

TRANSCRIPTS_DIR = Path("data/transcripts/general")
SCORES_405B = Path("data/processed/llm_scores_clbp_405b.csv")
OUTPUT_DIR = Path("data/processed")

def prepare_dataset():
    df = pd.read_csv(SCORES_405B)
    print(f"Loaded {len(df)} subjects from 405B scores")
    
    METRICS = ["Physical_Pain", "Emotional_Pain", "Depression", "poor_QoL",
               "Anxiety", "Catastrophizing", "Rumination", 
               "Narrative_Fragmentation", "Agency_Deficit"]
    
    examples = []
    skipped = 0
    
    for _, row in df.iterrows():
        study_id = int(row["study_id"])
        txt_path = TRANSCRIPTS_DIR / f"{study_id}_CLBP.txt"
        
        if not txt_path.exists():
            print(f"  Missing transcript: {study_id}")
            skipped += 1
            continue
        
        transcript = txt_path.read_text(encoding="utf-8").strip()
        
        # Scores del 405B como ground truth
        scores = {m: round(float(row[m]), 1) for m in METRICS if m in row}
        
        # Formato instruction/response para MLX
        example = {
            "text": f"""You are a knowledgeable psychiatrist evaluating a patient interview. Score these 9 clinical metrics from 0.0 to 10.0 (higher = worse):
Physical_Pain, Emotional_Pain, Depression, poor_QoL, Anxiety, Catastrophizing, Rumination, Narrative_Fragmentation, Agency_Deficit

Transcript:
{transcript}

Response:
{json.dumps(scores)}"""
        }
        examples.append(example)
    
    print(f"Prepared {len(examples)} examples ({skipped} skipped)")
    
    # Split train/val (80/20)
    split = int(len(examples) * 0.8)
    train = examples[:split]
    val = examples[split:]
    
    # Guardar JSONL
    for split_name, data in [("train", train), ("valid", val)]:
        path = OUTPUT_DIR / f"finetune_{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
        print(f"Saved {len(data)} examples to {path}")

if __name__ == "__main__":
    prepare_dataset()