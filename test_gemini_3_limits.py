
import os
import yaml
from google import genai
from google.genai import types

# Load from specific cloudrun file
yaml_path = r"C:\Users\jdmju\.gemini\antigravity\playground\obsidian-expanse\jurexia-api-git\.env.cloudrun.yaml"

with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

api_key = config.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

print(f"DEBUG: Using API key starting with {api_key[:10]}...")

try:
    print("\n--- MODEL TOKEN LIMITS ---")
    for model in client.models.list():
        # Check for Flash and Pro models
        if "flash" in model.name or "pro" in model.name:
            print(f"Model: {model.name}")
            print(f"  Input Token Limit: {model.input_token_limit}")

    # TEST: Try to count tokens with a large simulated request or look at gemini-3 specifically
    print("\n--- GEMINI 3 SPECIFIC CHECK ---")
    # Attempting to check the limit for a potential 2M token input count (simulated)
    # The API might report it in the model metadata.
    
except Exception as e:
    print(f"Error: {e}")
