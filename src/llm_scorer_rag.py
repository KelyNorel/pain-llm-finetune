import json
import requests
import chromadb
import time
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"
CHROMA_DIR = Path("data/chroma_db")

# Cargar ChromaDB
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_collection("clinical_literature")

METRICS_CONFIG = {
    "Physical_Pain": {
        "query": "physical pain chronic back pain intensity assessment",
        "description": "Rate the level of physical pain. 0.0 = no pain, 10.0 = most severe pain."
    },
    "Emotional_Pain": {
        "query": "emotional pain negative affect chronic pain",
        "description": "Rate the level of emotional pain. 0.0 = no emotional pain, 10.0 = most severe."
    },
    "Depression": {
        "query": "depression symptoms assessment chronic pain",
        "description": "Rate depression level. 0.0 = no depression, 10.0 = most severe."
    },
    "poor_QoL": {
        "query": "quality of life chronic pain functional impact",
        "description": "Rate quality of life. 0.0 = best QoL, 10.0 = poorest QoL."
    },
    "Anxiety": {
        "query": "anxiety symptoms chronic pain assessment",
        "description": "Rate anxiety level. 0.0 = no anxiety, 10.0 = most severe."
    },
    "Catastrophizing": {
        "query": "catastrophizing chronic pain cognitive assessment",
        "description": "Rate catastrophizing. 0.0 = no catastrophizing, 10.0 = maximum."
    },
    "Rumination": {
        "query": "rumination repetitive thinking chronic pain depression",
        "description": "Rate rumination. 0.0 = no rumination, 10.0 = maximum."
    },
    "Narrative_Fragmentation": {
        "query": "narrative coherence fragmentation autobiographical memory",
        "description": "Rate narrative fragmentation. 0.0 = highly coherent, 10.0 = severely fragmented."
    },
    "Agency_Deficit": {
        "query": "sense of agency personal control locus of control depression",
        "description": "Rate agency deficit. 0.0 = strong agency, 10.0 = complete powerlessness."
    }
}

def retrieve_context(query: str, n_results: int = 2) -> str:
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    context = ""
    for chunk, source in zip(chunks, sources):
        context += f"[{source}]\n{chunk}\n\n"
    return context

def score_single_metric(transcript: str, metric: str, config: dict) -> float:
    """Score una sola métrica con contexto RAG específico."""
    start = time.time()
    context = retrieve_context(config["query"])

    prompt = f"""You are a knowledgeable psychiatrist. Use this clinical literature as reference:

{context}

Based on the patient interview transcript below, provide a single score for:
{metric}: {config["description"]}

Respond ONLY with a JSON object like this: {{"{metric}": 5.3}}
One decimal place. Think deeply.

Transcript:
{transcript}"""

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0}
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        raw = response.json()["response"].strip()
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        score = json.loads(raw)
        value = float(list(score.values())[0])
        elapsed = time.time() - start
        print(f"  {metric}: {value} ({elapsed:.1f}s)")
        return value
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR {metric}: {e} ({elapsed:.1f}s)")
        return None

def score_transcript_rag(transcript_text: str) -> dict:
     # Truncar si es muy largo
    words = transcript_text.split()
    if len(words) > 2000:
        transcript_text = ' '.join(words[:2000])
        print(f"  Truncated to 2000 words")
    """Score los 9 metrics con RAG, un prompt por métrica."""
    results = {}
    for metric, config in METRICS_CONFIG.items():
        value = score_single_metric(transcript_text, metric, config)
        if value is not None:
            results[metric] = value
    return results

if __name__ == "__main__":
    test = "I have been dealing with back pain for years. It affects everything I do. Some days I can barely get out of bed."
    test_neutral = "I enjoy gardening and spending time with my family. Life is generally good."
    print("Testing RAG scorer (9 prompts)...\n")
    start_total = time.time()
    result = score_transcript_rag(test_neutral)
    total = time.time() - start_total
    print(f"\nFinal: {result}")
    print(f"Total time: {total:.1f}s ({total/9:.1f}s per metric)")