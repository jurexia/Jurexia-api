import os
from pathlib import Path
from google import genai
from google.genai import types

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CACHE_CORPUS_DIR = r"C:\Users\jdmju\.gemini\antigravity\playground\obsidian-expanse\jurexia-api-git\cache_corpus"

def count_tokens():
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not set")
        return

    client = genai.Client(api_key=GEMINI_API_KEY)
    corpus_dir = Path(CACHE_CORPUS_DIR)
    
    files = sorted(corpus_dir.glob("*.txt"))
    total_tokens = 0
    
    print(f"Checking token count for {len(files)} files in {CACHE_CORPUS_DIR}...")
    
    all_text = ""
    for f in files:
        text = f.read_text(encoding="utf-8")
        # Estimate per file
        res = client.models.count_tokens(
            model="gemini-1.5-flash", # Use a common model for counting
            contents=[text]
        )
        print(f" - {f.name}: {res.total_tokens} tokens")
        total_tokens += res.total_tokens
        all_text += text

    # Final combined count (system instruction + all texts)
    final_res = client.models.count_tokens(
        model="gemini-1.5-flash",
        contents=[all_text]
    )
    print("="*30)
    print(f"Total tokens in corpus: {final_res.total_tokens}")
    print(f"Remaining for 1,048,576 limit: {1048576 - final_res.total_tokens}")
    print(f"Remaining for 2,097,152 limit: {2097152 - final_res.total_tokens}")

if __name__ == "__main__":
    count_tokens()
