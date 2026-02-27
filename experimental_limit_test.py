
import os
import yaml
import time
from google import genai
from google.genai import types

# Load from specific cloudrun file
yaml_path = r"C:\Users\jdmju\.gemini\antigravity\playground\obsidian-expanse\jurexia-api-git\.env.cloudrun.yaml"

with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

api_key = config.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Test content: ~1,100,000 tokens to exceed the 1,048,576 limit
num_tokens = 1100000
test_text = "lorem ipsum legal " * (num_tokens // 4) 

print(f"Attempting to create a cache of approx {num_tokens} tokens...")
print(f"Text length: {len(test_text):,} characters.")

model_to_test = "models/gemini-3-flash-preview"

try:
    cache = client.caches.create(
        model=model_to_test,
        config=types.CreateCachedContentConfig(
            display_name="limit-test",
            contents=[types.Content(role="user", parts=[types.Part(text=test_text)])],
            ttl="300s", # 5 minutes
        )
    )
    print(f"SUCCESS! Cache created: {cache.name}")
    print(f"This means the 1M limit is NOT a hard cap for caching in {model_to_test}.")
    # Cleanup immediately
    client.caches.delete(name=cache.name)
    print("Cache deleted.")
except Exception as e:
    print(f"FAILED: {e}")
    error_msg = str(e).lower()
    if "limit" in error_msg or "token" in error_msg or "too large" in error_msg:
        print("Confirmed: The limit is indeed around 1M tokens.")
    else:
        print("Encountered an unexpected error, check the message above.")
