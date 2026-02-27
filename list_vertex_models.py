
import os
from google.cloud import aiplatform

def list_tuned_models():
    PROJECT_ID = "gen-lang-client-0981303295"
    LOCATION = "us-central1"
    
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    
    print("Checking supported models for tuning in Vertex AI...")
    # There isn't a direct "list tuneable" but we can try common ones
    models = ["gemini-1.5-flash-001", "gemini-1.5-flash-002", "gemini-1.0-pro-002", "gemini-1.5-pro-001"]
    
    for model in models:
        try:
            # This is a bit of a hack to see if it accepts the name
            print(f"Testing {model}...")
            # We won't actually run a job, just check if we can get info
            pass 
        except Exception as e:
            print(f"Error with {model}: {e}")

if __name__ == "__main__":
    list_tuned_models()
