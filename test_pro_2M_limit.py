
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

model_to_test = "models/gemini-3-pro-preview"
amt = 2100000
text = "derecho constitucional mexicano legal " * (amt // 5)
count_res = client.models.count_tokens(model=model_to_test, contents=[text])
actual_tokens = count_res.total_tokens

print(f"Testing PRO model with {actual_tokens:,} tokens...")

try:
    cache = client.caches.create(
        model=model_to_test,
        config=types.CreateCachedContentConfig(
            display_name=f"pro-test",
            contents=[types.Content(role="user", parts=[types.Part(text=text)])],
            ttl="60s",
        )
    )
    print(f"SUCCESS! Gemini 3 Pro supports {actual_tokens:,} tokens in cache.")
    client.caches.delete(name=cache.name)
except Exception as e:
    print(f"FAILED: {e}")
