
import os
import sys
from google.cloud import storage
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel
import vertexai

# CONFIGURATION
SCRIP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ID = "gen-lang-client-0981303295" # Deduced from gcloud
LOCATION = "us-central1"
BUCKET_NAME = f"jurexia-tuning-{PROJECT_ID}"
DATASET_FILE = os.path.join(SCRIP_DIR, "tuning_dataset_vertex.jsonl")
BASE_MODEL = "gemini-1.5-flash-002"
DISPLAY_NAME = "jurexia_redactor_amparo_v1"

def start_tuning():
    print(f"--- Vertex AI Tuning Job Launcher ---")
    print(f"Project: {PROJECT_ID}")
    print(f"Location: {LOCATION}")
    
    try:
        # 1. Initialize SDKs
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        storage_client = storage.Client(project=PROJECT_ID)
        
        # 2. Handle Bucket
        bucket = storage_client.bucket(BUCKET_NAME)
        if not bucket.exists():
            print(f"Creating bucket {BUCKET_NAME}...")
            bucket = storage_client.create_bucket(BUCKET_NAME, location=LOCATION)
        else:
            print(f"Using existing bucket {BUCKET_NAME}")
            
        # 3. Upload Dataset
        local_file_path = DATASET_FILE
        file_name = os.path.basename(local_file_path)
        blob = bucket.blob(file_name)
        print(f"Uploading {local_file_path} to gs://{BUCKET_NAME}/{file_name}...")
        blob.upload_from_filename(local_file_path)
        dataset_uri = f"gs://{BUCKET_NAME}/{file_name}"
        print(f"Dataset URI: {dataset_uri}")
        
        # 4. Launch Tuning Job
        print(f"Launching tuning job for {BASE_MODEL}...")
        
        # In Vertex AI Python SDK, tuning is often done via sft.train
        from vertexai.tuning import sft
        
        sft_job = sft.train(
            source_model=BASE_MODEL,
            train_dataset=dataset_uri,
            tuned_model_display_name=DISPLAY_NAME,
            epochs=4,
            learning_rate_multiplier=1.0,
        )
        
        print(f"Tuning job started successfully!")
        print(f"Job Name: {sft_job.name}")
        print(f"Tuned Model Resource: {sft_job.tuned_model_name}")
        print(f"You can monitor the progress in the Google Cloud Console under Vertex AI -> Training.")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("\nPossible solutions:")
        print("1. Run: gcloud auth application-default login")
        print(f"2. Ensure Vertex AI API and Cloud Storage API are enabled in project {PROJECT_ID}")
        print("3. Ensure you have the 'Vertex AI User' and 'Storage Admin' roles.")

if __name__ == "__main__":
    start_tuning()
