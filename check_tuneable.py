
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
    tuneable = []
    for m in data.get('models', []):
        methods = m.get('supportedGenerationMethods', [])
        if 'createTunedModel' in methods:
            tuneable.append({
                'name': m['name'],
                'methods': methods
            })
    
    with open('tuneable_models.json', 'w') as f:
        json.dump(tuneable, f, indent=2)
    
    print(f"Found {len(tuneable)} tuneable models:")
    for m in tuneable:
        print(f"- {m['name']}")
else:
    print(f"Error {response.status_code}: {response.text}")
