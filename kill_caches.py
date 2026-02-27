"""
EMERGENCY: Delete ALL active Gemini caches to stop charges.
Non-interactive - deletes everything automatically.
"""
import os
import sys
import yaml

sys.stdout.reconfigure(encoding='utf-8')

yaml_path = r"C:\Users\jdmju\.gemini\antigravity\playground\obsidian-expanse\jurexia-api-git\.env.cloudrun.yaml"
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

API_KEY = config.get('GEMINI_API_KEY', '')
if not API_KEY:
    print("ERROR: GEMINI_API_KEY not found")
    exit(1)

from google import genai

client = genai.Client(api_key=API_KEY)

print("DELETING ALL GEMINI CACHES...")

caches = list(client.caches.list())
print(f"Found {len(caches)} caches to delete.\n")

deleted = 0
failed = 0
for i, cache in enumerate(caches):
    try:
        client.caches.delete(name=cache.name)
        deleted += 1
        if (i + 1) % 10 == 0:
            print(f"  Deleted {i+1}/{len(caches)}...")
    except Exception as e:
        failed += 1
        print(f"  FAILED [{i+1}]: {cache.name} - {e}")

print(f"\nDone. Deleted: {deleted}, Failed: {failed}")

# Verify
remaining = list(client.caches.list())
print(f"Remaining caches: {len(remaining)}")
if len(remaining) == 0:
    print("ALL CACHES CLEARED. Hourly charges stopped.")
