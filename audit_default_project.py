"""Check for active Gemini caches in the Default Gemini Project (gen-lang-client-0132141051)"""
from google import genai
import os
import yaml

# Load the API key from the same config
yaml_path = r"C:\Users\jdmju\.gemini\antigravity\playground\obsidian-expanse\jurexia-api-git\.env.cloudrun.yaml"
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

API_KEY = config.get('GEMINI_API_KEY', '')
if not API_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env.cloudrun.yaml")
    exit(1)

print(f"Checking caches in Default Gemini Project via API KEY...")
client = genai.Client(api_key=API_KEY)

try:
    caches = list(client.caches.list())
    print(f"Found {len(caches)} caches via API KEY.")
    for c in caches:
        print(f" - Name: {c.name}")
        print(f"   Display: {c.display_name}")
        print(f"   Model: {c.model}")
        print(f"   Created: {c.create_time}")
        print(f"   Expire: {c.expire_time}")
        print(f"   Token count: {c.usage_metadata}")
        print()
except Exception as e:
    print(f"Error: {e}")

# Also check tuned models
print("\n--- Checking for tuned models ---")
try:
    tuned_models = list(client.tunings.list())
    print(f"Found {len(tuned_models)} tuning jobs.")
    for t in tuned_models:
        print(f" - {t}")
except Exception as e:
    print(f"Error listing tuned models: {e}")
