import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

PROMPT_1 = """You are a knowledgeable psychiatrist tasked with evaluating the emotional state of a patient based on an interview transcript. Carefully analyze the transcript and provide the following scores as decimal values (to one decimal place) on a scale from 0.0 to 10.0, formatted exactly as specified:

1. Physical_Pain: Rate the level of physical pain separately on a scale from 0.0 to 10.0, where 0.0 indicates no physical pain and 10.0 indicates the most severe level of physical pain.
2. Emotional_Pain: Rate the level of emotional pain separately on a scale from 0.0 to 10.0, where 0.0 indicates no emotional pain and 10.0 indicates the most severe emotional pain possible.
3. Depression: Rate the patient's level of depression on a scale from 0.0 to 10.0, where 0.0 indicates no depression and 10.0 indicates the most severe depression possible.
4. QoL: Rate the patient's quality of life on a scale from 0.0 to 10.0, where 0.0 indicates the best quality of life and 10.0 indicates the poorest quality of life.
5. Anxiety: Rate the patient's anxiety on a scale from 0.0 to 10.0, where 0.0 indicates no anxiety and 10.0 indicates the most severe anxiety possible.

Provide only a valid JSON response with decimal values to one decimal place. The JSON should follow this exact structure:
{"Physical_Pain": [decimal value], "Emotional_Pain": [decimal value], "Depression": [decimal value], "poor_QoL": [decimal value], "Anxiety": [decimal value]}

Think deeply."""

PROMPT_2 = """You are a knowledgeable psychiatrist tasked with evaluating the emotional state of a patient based on an interview transcript. Carefully analyze the transcript and provide the following scores as decimal values (to one decimal place) on a scale from 0.0 to 10.0, formatted exactly as specified:

6. Catastrophizing: Rate the participant's catastrophic thinking patterns on a scale from 0.0 to 10.0, where 0.0 indicates no catastrophizing (realistic assessment of potential outcomes, balanced perspective on challenges, appropriate evaluation of risks) and 10.0 indicates maximum catastrophizing (consistently predicting worst-case scenarios, dramatic overestimation of negative possibilities, persistent expectation of disaster across multiple situations, inability to consider moderate or positive outcomes).
7. Rumination: Rate the participant's level of rumination on a scale from 0.0 to 10.0, where 0.0 indicates no rumination (fluid, forward-moving thought patterns, varied topics, solution-focused thinking) and 10.0 indicates maximum rumination (persistent repetitive thinking about negative experiences, circular thought patterns, inability to move past specific concerns, excessive focus on problems without progress toward resolution, repetitive analysis of past events or worries).
8. Narrative_Fragmentation: Rate the participant's overall narrative coherence on a scale from 0.0 to 10.0, where 0.0 indicates highly coherent narrative structure (clear chronological flow, appropriate time anchoring, consistent timeline, logical cause-and-effect reasoning, appropriate causal connections) and 10.0 indicates severely fragmented narrative (no clear sequence, major temporal gaps, contradictory timeline, no logical causal connections, magical thinking).
9. Agency_Deficit: Rate the participant's sense of personal agency and control on a scale from 0.0 to 10.0, where 0.0 indicates strong sense of personal agency (balanced internal-external attribution, recognizes personal responsibility and ability to influence outcomes, appropriate internal locus of control) and 10.0 indicates complete powerlessness (exclusively external attribution, victim stance, everything controlled by fate/others/circumstances, no sense of personal influence).

Provide only a valid JSON response with decimal values to one decimal place. The JSON should follow this exact structure:
{"Catastrophizing": [decimal value], "Rumination": [decimal value], "Narrative_Fragmentation": [decimal value], "Agency_Deficit": [decimal value]}

Think deeply."""


def score_transcript(transcript_text: str) -> dict:
    """Score a transcript using both prompts. Returns merged dict of 9 metrics."""
    
    results = {}
    
    for prompt_template, prompt_num in [(PROMPT_1, 1), (PROMPT_2, 2)]:
        full_prompt = f"{prompt_template}\n\nTranscript:\n{transcript_text}"
        
        payload = {
            "model": MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0}
        }
        
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
            raw = response.json()["response"].strip()
            
            # Limpiar si viene con markdown
            if "```" in raw:
                raw = raw.split("```")[1].replace("json", "").strip()
            
            scores = json.loads(raw)
            results.update(scores)
            print(f"  Prompt {prompt_num} OK: {scores}")
            
        except Exception as e:
            print(f"  ERROR en prompt {prompt_num}: {e}")
            print(f"  Raw response: {raw if 'raw' in locals() else 'N/A'}")
    
    return results


if __name__ == "__main__":
    # Test rápido con texto dummy
    test_text = "I have been dealing with back pain for years. It affects everything I do. Some days I can barely get out of bed."
    print("Testing scorer...")
    result = score_transcript(test_text)
    print(f"\nFinal result: {result}")