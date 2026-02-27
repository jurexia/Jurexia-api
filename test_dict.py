
from google import genai
import yaml

yaml_path = r"C:\Users\jdmju\.gemini\antigravity\playground\obsidian-expanse\jurexia-api-git\.env.cloudrun.yaml"
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

client = genai.Client(api_key=config['GEMINI_API_KEY'])

examples = [
    {"text_input": "Hola", "output": "Hola, ¿en qué puedo ayudarte?"},
    {"text_input": "Quién eres", "output": "Soy un asistente legal."}
]

try:
    print("Intentando tuning con diccionarios...")
    operation = client.tunings.tune(
        base_model="models/gemini-1.0-pro-001",
        training_dataset={'examples': examples},
        config={'tuned_model_display_name': 'test-tuning-delete-me'}
    )
    print(f"Éxito: {operation.name}")
except Exception as e:
    print(f"Error: {e}")
