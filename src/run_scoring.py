import json
import pandas as pd
from pathlib import Path
from llm_scorer import score_transcript
import time

# Paths
TRANSCRIPTS_DIR = Path("data/transcripts/general")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(exist_ok=True)

def extract_study_id(filename: str) -> str:
    """Extrae Study ID del nombre del archivo. Ej: 1202_CLBP.txt -> 1202"""
    return filename.replace("_CLBP.txt", "")

def run_batch():
    start_total = time.time()
    txt_files = sorted(TRANSCRIPTS_DIR.glob("*_CLBP.txt"))
    print(f"Found {len(txt_files)} transcripts\n")
    
    results = []
    
    for txt_path in txt_files:
    #for i, txt_path in enumerate(txt_files):
    #    if i >= 3:
    #        break
        study_id = extract_study_id(txt_path.name)
        print(f"Processing {study_id}...")
        
        try:
            text = txt_path.read_text(encoding="utf-8")
            scores = score_transcript(text)
            scores["study_id"] = int(study_id)
            results.append(scores)
            print(f"  ✓ Done\n")
            
        except Exception as e:
            print(f"  ERROR: {e}\n")

    total = time.time() - start_total
    print(f"\nTotal time: {total:.1f}s ({total/60:.1f} min)")
    print(f"Average per subject: {total/len(results):.1f}s")
    # Guardar resultados
    df = pd.DataFrame(results)
    # Reordenar columnas
    cols = ["study_id", "Physical_Pain", "Emotional_Pain", "Depression", 
            "poor_QoL", "Anxiety", "Catastrophizing", "Rumination", 
            "Narrative_Fragmentation", "Agency_Deficit"]
    df = df[cols]
    
    output_path = OUTPUT_DIR / "llm_scores_clbp_small_test3.csv"
    df.to_csv(output_path, index=False)
    print(f"\nDone! {len(df)} subjects scored.")
    print(f"Saved to {output_path}")
    return df

if __name__ == "__main__":
    df = run_batch()
    print(df.describe())