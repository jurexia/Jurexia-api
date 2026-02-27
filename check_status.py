
import yaml
from google import genai
import json

yaml_path = r"C:\Users\jdmju\.gemini\antigravity\playground\obsidian-expanse\jurexia-api-git\.env.cloudrun.yaml"
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

client = genai.Client(api_key=config['GEMINI_API_KEY'])




def check_status():
    print("Fetching active tuning jobs...")
    try:
        # En google-genai v1.x se usa client.tunings.list()
        jobs = client.tunings.list()
        found = False
        for job in jobs:
            print(f"Job Name: {job.name}")
            print(f"State: {job.state}")
            print(f"Base Model: {job.base_model}")
            print(f"Created: {job.create_time}")
            print("-" * 20)
            found = True
        
        if not found:
            print("No active tuning jobs found.")
                
    except Exception as e:
        print(f"Error checking status: {e}")




if __name__ == "__main__":
    check_status()
