"""
Iurexia Fine-Tuning Launcher — Upload JSONL + Start Fine-tuning Job
"""
import os
import sys
from openai import OpenAI

JSONL_PATH = os.path.join(os.path.dirname(__file__), "ft_data", "training_data.jsonl")

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # 1. Upload training file
    print("📤 Uploading training data to OpenAI...")
    with open(JSONL_PATH, "rb") as f:
        upload = client.files.create(file=f, purpose="fine-tune")
    print(f"   ✅ File uploaded: {upload.id} ({upload.bytes:,} bytes)")

    # 2. Launch fine-tuning job
    print("\n🚀 Launching fine-tuning job...")
    job = client.fine_tuning.jobs.create(
        training_file=upload.id,
        model="gpt-4o-mini-2024-07-18",
        suffix="iurexia-redactor-v1",
        hyperparameters={
            "n_epochs": 3,
        },
    )
    print(f"   ✅ Job created: {job.id}")
    print(f"   📊 Status: {job.status}")
    print(f"   🏷️  Model will be: ft:gpt-4o-mini-2024-07-18:*:iurexia-redactor-v1:*")
    print(f"\n   Monitor at: https://platform.openai.com/finetune/{job.id}")
    print(f"   Or run: python -c \"from openai import OpenAI; c=OpenAI(); j=c.fine_tuning.jobs.retrieve('{job.id}'); print(j.status, j.fine_tuned_model)\"")

if __name__ == "__main__":
    main()
