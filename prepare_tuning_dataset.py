
import os
import yaml
import json
from qdrant_client import QdrantClient
from google import genai
from google.genai import types

# 1. Configuracion de Entorno
yaml_path = r"C:\Users\jdmju\.gemini\antigravity\playground\obsidian-expanse\jurexia-api-git\.env.cloudrun.yaml"
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

qdrant_client = QdrantClient(url=config['QDRANT_URL'], api_key=config['QDRANT_API_KEY'])
gemini_client = genai.Client(api_key=config['GEMINI_API_KEY'])

COLLECTION_NAME = "sentencias_amparo_directo"
LIMIT_SAMPLES = 200
# Formato optimizado para OpenAI (Conversacional)
OUTPUT_FILE = "dataset_openai_redactor.jsonl"
MAX_CHARS = 100000 # Opcional: ampliar para OpenAI, pero mantenemos algo razonable
SYSTEM_PROMPT = "Eres un Secretario de Estudio y Cuenta de la Suprema Corte de Justicia de la Nación. Redacta la resolución o proyecto de sentencia basándote en la siguiente instrucción y empleando un lenguaje técnico, preciso, exhaustivo y estructurado."

def generate_instruction(text_chunk, hierarchy):
    """Usa Gemini para generar una instruccion de usuario basada en el texto legal."""
    prompt = f"""
    ERES UN EXPERTO EN PROMPT ENGINEERING PARA IA LEGAL. 
    Tu tarea es leer este fragmento de una sentencia de Amparo Directo en Mexico y redactar una INSTRUCCION tecnica.
    
    FRAGMENTO LEGAL: {text_chunk[:2000]} # Solo enviamos muestra para el prompt
    JERARQUIA: {hierarchy}
    
    REGLAS:
    1. La instruccion debe ser profesional (Secretario de Acuerdos).
    2. Debe ser especifica al contenido legal del fragmento.
    3. Responde UNICAMENTE con la instruccion de texto plano.
    """
    try:
        response = gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Redacta un fragmento de Amparo Directo: {hierarchy}"

def main():
    print(f"Iniciando preparacion para OpenAI ({COLLECTION_NAME})...")
    
    res = qdrant_client.scroll(collection_name=COLLECTION_NAME, limit=LIMIT_SAMPLES)
    points = res[0]
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, p in enumerate(points):
            payload = p.payload
            text_output = payload.get("texto", "")
            hierarchy = payload.get("jerarquia_txt", "Amparo Directo")
            
            if not text_output or len(text_output) < 100:
                continue
            
            # Truncar para cumplir con limites de AI Studio
            if len(text_output) > MAX_CHARS:
                text_output = text_output[:MAX_CHARS] + "... [truncado]"

            user_input = generate_instruction(text_output, hierarchy)
            
            # Formato estándar OpenAI Conversational Tuning
            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": text_output}
                ]
            }
            
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            if (i + 1) % 20 == 0:
                print(f"Procesados {i+1}/{LIMIT_SAMPLES}...")

    print(f"Dataset listo: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
