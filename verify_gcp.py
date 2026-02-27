
import os
from google.cloud import storage
from google.cloud import aiplatform

def verify():
    try:
        # Check Project ID
        project_id = os.environ.get('GCP_PROJECT_ID', 'gen-lang-client-0981303295')
        print(f"Testing with Project ID: {project_id}")
        
        # Try to list buckets
        storage_client = storage.Client(project=project_id)
        buckets = list(storage_client.list_buckets(max_results=5))
        print("Successfully connected to GCS. Found buckets:")
        for bucket in buckets:
            print(f" - {bucket.name}")
            
        if not buckets:
            print("No buckets found in this project.")
            
        # Try to init AI Platform
        aiplatform.init(project=project_id, location='us-central1')
        print("Successfully initialized AI Platform (Vertex AI).")
        
    except Exception as e:
        print(f"Error during verification: {e}")

if __name__ == "__main__":
    verify()
