from google import genai
import os
import yaml

# Load Vertex config
yaml_path = r"C:\Users\jdmju\.gemini\antigravity\playground\obsidian-expanse\jurexia-api-git\.env.cloudrun.yaml"
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

PROJECT = config.get('GCP_PROJECT', 'gen-lang-client-0981303295')
LOCATION = config.get('GCP_LOCATION', 'us-central1')

print(f"Checking VERTEX AI caches in {PROJECT} ({LOCATION})...")

client = genai.Client(
    vertexai=True,
    project=PROJECT,
    location=LOCATION
)

try:
    caches = list(client.caches.list())
    print(f"Found {len(caches)} Vertex AI caches.")
    for c in caches:
        print(f" - {c.name}: {c.display_name} (Created: {c.create_time})")
except Exception as e:
    print(f"Error listing Vertex AI caches: {e}")
