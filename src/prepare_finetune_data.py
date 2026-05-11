import json
import random
import pandas as pd
from pathlib import Path

TRANSCRIPTS = {
    "general": Path("data/transcripts/general"),
    "dx": Path("data/transcripts/dx")
}

SCORES = {
    "general": Path("data/processed/llm_scores_clbp_405b.csv"),
    "dx": Path("data/processed/llm_scores_clbp_dx_405b.csv")
}

OUTPUT_DIR = Path("data/processed")

def prepare_dataset():
    METRICS = ["Physical_Pain", "Emotional_Pain", "Depression", "poor_QoL",
               "Anxiety", "Catastrophizing", "Rumination",
               "Narrative_Fragmentation", "Agency_Deficit"]

    examples = []

    for transcript_type, transcript_dir in TRANSCRIPTS.items():
        df = pd.read_csv(SCORES[transcript_type])
        print(f"Loaded {len(df)} subjects for {transcript_type}")

        for _, row in df.iterrows():
            study_id = int(row["study_id"])
            scores = {m: round(float(row[m]), 1) for m in METRICS if m in row}

            txt_path = transcript_dir / f"{study_id}_CLBP.txt"
            if not txt_path.exists():
                print(f"  Missing {transcript_type}: {study_id}")
                continue

            transcript = txt_path.read_text(encoding="utf-8").strip()
            words = transcript.split()
            if len(words) > 2000:
                transcript = ' '.join(words[:2000])

            example = {
                "text": f"""You are a knowledgeable psychiatrist evaluating a patient interview. Score these 9 clinical metrics from 0.0 to 10.0 (higher = worse):
Physical_Pain, Emotional_Pain, Depression, poor_QoL, Anxiety, Catastrophizing, Rumination, Narrative_Fragmentation, Agency_Deficit

Transcript:
{transcript}

Response:
{json.dumps(scores)}"""
            }
            examples.append(example)

    print(f"\nTotal: {len(examples)} examples")

    random.seed(42)
    random.shuffle(examples)

    split = int(len(examples) * 0.8)
    train = examples[:split]
    val = examples[split:]

    for split_name, data in [("train", train), ("valid", val)]:
        path = OUTPUT_DIR / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
        print(f"Saved {len(data)} examples to {path}")

if __name__ == "__main__":
    prepare_dataset()