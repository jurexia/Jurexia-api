
import os
import yaml
import requests
import json

yaml_path = r"C:\Users\jdmju\.gemini\antigravity\playground\obsidian-expanse\jurexia-api-git\.env.cloudrun.yaml"
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

api_key = config['GEMINI_API_KEY']
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    with open('all_rest_models.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data.get('models', []))} models to all_rest_models.json")
else:
    print(f"Error {response.status_code}: {response.text}")
