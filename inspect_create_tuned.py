
from google import genai
import inspect
import yaml

yaml_path = r"C:\Users\jdmju\.gemini\antigravity\playground\obsidian-expanse\jurexia-api-git\.env.cloudrun.yaml"
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

client = genai.Client(api_key=config['GEMINI_API_KEY'])

print("Signature of create_tuned_model:")
sig = inspect.signature(client.models.create_tuned_model)
print(sig)
