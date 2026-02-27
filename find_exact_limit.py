
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

model_to_test = "models/gemini-3-flash-preview"

# Test targeted amounts
test_amounts = [1000000, 1048576, 1100000, 1200000, 1300000, 1500000]

for amt in test_amounts:
    # Use a dummy text block
    text = "articulo constitucional mejico legal juridico ley " * (amt // 7)
    
    # Calculate exact tokens first
    count_res = client.models.count_tokens(model=model_to_test, contents=[text])
    actual_tokens = count_res.total_tokens
    
    print(f"Testing Actual tokens: {actual_tokens:,}...")
    
    try:
        cache = client.caches.create(
            model=model_to_test,
            config=types.CreateCachedContentConfig(
                display_name=f"test-{actual_tokens}",
                contents=[types.Content(role="user", parts=[types.Part(text=text)])],
                ttl="60s",
            )
        )
        print(f"  SUCCESS at {actual_tokens:,}")
        client.caches.delete(name=cache.name)
    except Exception as e:
        print(f"  FAILED at {actual_tokens:,}: {e}")
        # break # stop once we find the limit
