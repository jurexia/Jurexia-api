
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load env from parent directory
env_path = os.path.abspath(os.path.join(os.getcwd(), "..", ".env"))
print(f"Loading env from: {env_path}")
load_dotenv(env_path)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

print(f"DEBUG: GEMINI_API_KEY is {'SET' if api_key else 'MISSING'}")

if not api_key:
    # Try one level more just in case
    env_path = os.path.abspath(os.path.join(os.getcwd(), ".env"))
    load_dotenv(env_path)
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

try:
    models_to_check = ["gemini-3.1-flash", "gemini-3.1-pro", "gemini-3-flash", "gemini-3-pro", "gemini-1.5-pro"]
    
    print("\n--- MODEL CAPABILITIES ---")
    for model in client.models.list():
        m_name = model.name.lower()
        if any(x in m_name for x in models_to_check) or "flash" in m_name or "pro" in m_name:
            print(f"Model ID: {model.name}")
            print(f"  Input Token Limit: {model.input_token_limit}")
            print(f"  Context Window: {getattr(model, 'context_window', 'N/A')}")
            
    # Try a fake cache creation check (will fail but might give info on limit in error)
    print("\n--- CACHE LIMIT TEST ---")
    try:
        # Just info gathering
        print("Checking gemini-3.1-flash-preview specifically...")
    except Exception as e:
        print(f"Cache test error: {e}")

except Exception as e:
    print(f"Error: {e}")
