
import os
import yaml
import json
from google import genai
from google.genai import types

# 1. Configuracion
yaml_path = r"C:\Users\jdmju\.gemini\antigravity\playground\obsidian-expanse\jurexia-api-git\.env.cloudrun.yaml"
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

client = genai.Client(api_key=config['GEMINI_API_KEY'])

DATASET_FILE = "tuning_dataset_amparo_v1.jsonl"
BASE_MODEL = "models/gemini-1.5-flash-001" 
TUNED_MODEL_NAME = "iurexia-redactor-v1-amparo"

def start_tuning():
    if not os.path.exists(DATASET_FILE):
        print(f"Error: No se encuentra {DATASET_FILE}. Ejecuta primero prepare_tuning_dataset.py")
        return

    print(f"Subiendo dataset y lanzando tuning en Google AI Studio...")
    print(f"Base: {BASE_MODEL}")
    print(f"Nombre objetivo: {TUNED_MODEL_NAME}")

    try:
        # Cargamos los ejemplos y los convertimos al formato TuningExample
        examples = []
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                user_text = ""
                model_text = ""
                # El archivo original tiene el formato de Chat (contents -> role/parts)
                for msg in data["contents"]:
                    if msg["role"] == "user":
                        user_text = msg["parts"][0]["text"]
                    elif msg["role"] == "model":
                        model_text = msg["parts"][0]["text"]
                
                if user_text and model_text:
                    examples.append(
                        types.TuningExample(
                            text_input=user_text,
                            output=model_text
                        )
                    )

        print(f"Cargados {len(examples)} ejemplos. Iniciando Job via client.tunings.tune...")
        
        # Lanzamiento usando el metodo correcto de google-genai v1.x
        tuning_job = client.tunings.tune(
            base_model=BASE_MODEL,
            training_dataset=types.TuningDataset(
                examples=examples
            ),
            config=types.CreateTuningJobConfig(
                tuned_model_display_name=TUNED_MODEL_NAME,
                epoch_count=1,
                batch_size=4,
                learning_rate=0.001
            )
        )
        
        print(f"✅ ¡Tuning iniciado con éxito!")
        print(f"Nombre del Job: {tuning_job.name}")
        print(f"Estado inicial: {tuning_job.state}")
        print("\nPuedes monitorear el progreso en https://aistudio.google.com/app/tuned_models")
        
    except Exception as e:
        print(f"❌ Error al iniciar el tuning: {e}")

if __name__ == "__main__":
    start_tuning()
