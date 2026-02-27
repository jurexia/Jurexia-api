
from google import genai
from google.genai import types
import yaml

yaml_path = r"C:\Users\jdmju\.gemini\antigravity\playground\obsidian-expanse\jurexia-api-git\.env.cloudrun.yaml"
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

client = genai.Client(api_key=config['GEMINI_API_KEY'])

try:
    # Test call with training_data instead of training_dataset
    # We use a dummy model to see if it even gets to the server
    client.tunings.tune(
        base_model="models/gemini-1.5-flash-001",
        training_data=types.TuningDataset(examples=[]),
        config=types.CreateTuningJobConfig(tuned_model_display_name="test")
    )
except TypeError as te:
    print(f"TypeError: {te}")
except Exception as e:
    print(f"Other Error: {e}")
