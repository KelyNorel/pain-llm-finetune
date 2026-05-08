import time
import pandas as pd
from pathlib import Path
from llm_scorer_rag import score_transcript_rag

TRANSCRIPTS_DIR = Path("data/transcripts/general")
#TRANSCRIPTS_DIR = Path("data/transcripts/dx")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(exist_ok=True)

def extract_study_id(filename: str) -> str:
    return filename.replace("_CLBP.txt", "")

def run_batch():
    start_total = time.time()
    txt_files = sorted(TRANSCRIPTS_DIR.glob("*_CLBP.txt"))
    print(f"Found {len(txt_files)} transcripts\n")

    results = []

    for i, txt_path in enumerate(txt_files):
        study_id = extract_study_id(txt_path.name)
        print(f"Processing {study_id}... ({i+1}/{len(txt_files)})")
        start = time.time()

        try:
            text = txt_path.read_text(encoding="utf-8")
            scores = score_transcript_rag(text)
            scores["study_id"] = int(study_id)
            results.append(scores)
            elapsed = time.time() - start
            print(f"  ✓ Done ({elapsed:.1f}s)\n")

        except Exception as e:
            print(f"  ERROR: {e}\n")

    total = time.time() - start_total
    print(f"\nTotal time: {total:.1f}s ({total/60:.1f} min)")
    print(f"Average per subject: {total/len(results):.1f}s")

    df = pd.DataFrame(results)
    cols = ["study_id", "Physical_Pain", "Emotional_Pain", "Depression",
            "poor_QoL", "Anxiety", "Catastrophizing", "Rumination",
            "Narrative_Fragmentation", "Agency_Deficit"]
    df = df[cols]

    output_path = OUTPUT_DIR / "llm_scores_clbp_rag_v2.csv"
    #output_path = OUTPUT_DIR / "llm_scores_clbp_dx_rag.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return df

if __name__ == "__main__":
    df = run_batch()
    print(df.describe())