
import os
import yaml
import json
from google import genai

yaml_path = r"C:\Users\jdmju\.gemini\antigravity\playground\obsidian-expanse\jurexia-api-git\.env.cloudrun.yaml"
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

client = genai.Client(api_key=config['GEMINI_API_KEY'])

models = []
for m in client.models.list():
    models.append(m.model_dump())

with open('full_models_list.json', 'w') as f:
    json.dump(models, f, indent=2)

print(f"Saved {len(models)} models to full_models_list.json")
