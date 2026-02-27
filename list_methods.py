
from google import genai
import yaml

yaml_path = r"C:\Users\jdmju\.gemini\antigravity\playground\obsidian-expanse\jurexia-api-git\.env.cloudrun.yaml"
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

client = genai.Client(api_key=config['GEMINI_API_KEY'])

print("Methods in client.tuned_models:")
print([m for m in dir(client.tuned_models) if not m.startswith('_')])

print("\nMethods in client.models:")
print([m for m in dir(client.models) if not m.startswith('_')])
