import os
import mimetypes
from supabase import create_client, Client
import yaml
from pathlib import Path

# Configuración
BASE_DIR = r"C:\Respaldo_Iurexia_PDFs"
BUCKET_NAME = "legal-docs"

def upload_pdfs():
    # Intentar cargar config desde .env.cloudrun.yaml
    yaml_path = Path(r"c:\Users\jdmju\.gemini\antigravity\playground\obsidian-expanse\jurexia-api-git\.env.cloudrun.yaml")
    
    if not yaml_path.exists():
        print(f"❌ Error: {yaml_path} no encontrado")
        return

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    
    url = config.get("SUPABASE_URL")
    # Intentamos SERVICE_KEY o SERVICE_ROLE_KEY
    key = config.get("SUPABASE_SERVICE_KEY") or config.get("SUPABASE_SERVICE_ROLE_KEY")
    
    if not url or not key:
        print("❌ Error: SUPABASE_URL o SUPABASE_SERVICE_KEY no encontrados en .env.cloudrun.yaml")
        return

    supabase: Client = create_client(url, key)
    
    files_uploaded = 0
    errors = 0

    print(f"🚀 Iniciando subida a Supabase Bucket: {BUCKET_NAME}")

    # Verificar si el bucket existe, si no, lo intentamos crear (esto puede fallar por permisos)
    try:
        supabase.storage.get_bucket(BUCKET_NAME)
        print(f"✅ Bucket '{BUCKET_NAME}' detectado.")
    except Exception:
        print(f"⚠️ Bucket '{BUCKET_NAME}' no encontrado. Intentando crear...")
        try:
            supabase.storage.create_bucket(BUCKET_NAME, options={"public": True})
            print(f"✅ Bucket '{BUCKET_NAME}' creado como Público.")
        except Exception as e:
            print(f"❌ No se pudo crear el bucket: {e}. Asegúrate de crearlo manualmente en el dashboard de Supabase.")
            return

    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.lower().endswith(".pdf"):
                local_path = os.path.join(root, file)
                # Crear el path relativo para el storage (ej: Queretaro/Leyes/archivo.pdf)
                relative_path = os.path.relpath(local_path, BASE_DIR).replace("\\", "/")
                
                mime_type, _ = mimetypes.guess_type(local_path)
                if not mime_type:
                    mime_type = "application/pdf"

                try:
                    with open(local_path, "rb") as f:
                        # Subir archivo (force true para sobrescribir si ya existe)
                        # Nota: Supabase storage upload tiene un límite por defecto de 50MB por archivo.
                        # Nuestros archivos son de ~70MB el total de 160, así que cada uno es pequeño.
                        supabase.storage.from_(BUCKET_NAME).upload(
                            path=relative_path,
                            file=f,
                            file_options={"cache-control": "3600", "upsert": "true", "content-type": mime_type}
                        )
                        print(f"✅ Subido: {relative_path}")
                        files_uploaded += 1
                except Exception as e:
                    # Si el error es que ya existe (duplicate), no incrementamos errores si usamos upsert o es esperado
                    # Pero el wrapper de python a veces da error aun con upsert si no se maneja bien
                    print(f"❌ Error subiendo {relative_path}: {e}")
                    errors += 1

    print(f"\n✨ Resumen: {files_uploaded} archivos subidos con éxito. {errors} errores.")

if __name__ == "__main__":
    upload_pdfs()
