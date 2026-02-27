
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

# We know the project ID from earlier
project_id = "gen-lang-client-0981303295"
location = "us-central1"

client = genai.Client(
    vertexai=True,
    project=project_id,
    location=location
)

print("Listing models available on Vertex AI...")
try:
    for model in client.models.list():
        print(f" - {model.name}")
except Exception as e:
    print(f"Error listing models: {e}")
