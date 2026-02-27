from google import genai
import sys
import os
import yaml

region = sys.argv[1] if len(sys.argv) > 1 else "us-central1"

# Load Vertex config
yaml_path = r"C:\Users\jdmju\.gemini\antigravity\playground\obsidian-expanse\jurexia-api-git\.env.cloudrun.yaml"
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

PROJECT = config.get('GCP_PROJECT', 'gen-lang-client-0981303295')

print(f"Checking VERTEX AI caches in {PROJECT} ({region})...")

client = genai.Client(
    vertexai=True,
    project=PROJECT,
    location=region
)

try:
    caches = list(client.caches.list())
    if not caches:
        print(f"Found 0 caches in {region}.")
    else:
        print(f"Found {len(caches)} Vertex AI caches in {region}.")
        for c in caches:
            print(f" - {c.name}: {c.display_name} (Created: {c.create_time})")
            # client.caches.delete(name=c.name) # Uncomment to delete
except Exception as e:
    print(f"Error in {region}: {e}")
