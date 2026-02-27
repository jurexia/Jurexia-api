
import os
from google.cloud import aiplatform

def check():
    PROJECT_ID = "gen-lang-client-0981303295"
    LOCATION = "us-central1"
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    
    # Try to see if we can instantiate a tuning job with different names
    # Note: sft.train is the standard way now.
    print("Testing model strings for SFT...")
    try:
        from vertexai.tuning import sft
        # We won't call train because it starts a job, but we can look for info
    except ImportError:
        print("sft import failed")

if __name__ == "__main__":
    check()
