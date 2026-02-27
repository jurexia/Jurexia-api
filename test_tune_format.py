
import os
import yaml
from google import genai
from google.genai import types

# 1. Configuracion
yaml_path = r"C:\Users\jdmju\.gemini\antigravity\playground\obsidian-expanse\jurexia-api-git\.env.cloudrun.yaml"
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

client = genai.Client(api_key=config['GEMINI_API_KEY'])

def test_tune_format():
    try:
        # Probamos con el formato mas simple posible
        print("Probando client.tunings.tune con 1 ejemplo...")
        tuning_job = client.tunings.tune(
            base_model="models/gemini-1.5-flash",
            training_dataset=types.TuningDataset(
                examples=[
                    types.TuningExample(
                        text_input="Hola",
                        output="Hola, ¿en qué puedo ayudarte?"
                    )
                ]
            ),
            config=types.CreateTuningJobConfig(
                tuned_model_display_name="test-format-tuning"
            )
        )
        print(f"Exito: {tuning_job.name}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_tune_format()
