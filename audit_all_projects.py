"""
Check for active Gemini caches in ALL projects using ADC (Application Default Credentials).
This checks the Default Gemini Project (gen-lang-client-0132141051) which still has
generativelanguage.googleapis.com enabled.
"""
import subprocess
import json

projects = [
    "gen-lang-client-0132141051",
    "gen-lang-client-0981303295",
    "irexia",
    "iurexia"
]

print("=" * 70)
print("COMPREHENSIVE GCP GEMINI AUDIT — ALL PROJECTS")
print("=" * 70)

for proj in projects:
    print(f"\n{'='*50}")
    print(f"PROJECT: {proj}")
    print(f"{'='*50}")
    
    # Check enabled APIs
    try:
        result = subprocess.run(
            ["gcloud", "services", "list", "--enabled", f"--project={proj}", 
             "--filter=name:generativelanguage OR name:aiplatform",
             "--format=value(name)"],
            capture_output=True, text=True, timeout=30
        )
        apis = result.stdout.strip()
        if apis:
            print(f"  ACTIVE APIs: {apis}")
        else:
            print(f"  No Gemini/Vertex APIs enabled ✅")
    except Exception as e:
        print(f"  Error listing APIs: {e}")

    # Check Cloud Run services
    try:
        result = subprocess.run(
            ["gcloud", "run", "services", "list", f"--project={proj}", 
             "--format=value(name,region)"],
            capture_output=True, text=True, timeout=30
        )
        services = result.stdout.strip()
        if services:
            print(f"  Cloud Run services: {services}")
        else:
            print(f"  No Cloud Run services ✅")
    except Exception as e:
        print(f"  Error listing services: {e}")

    # Check Compute Engine instances
    try:
        result = subprocess.run(
            ["gcloud", "compute", "instances", "list", f"--project={proj}",
             "--format=value(name,zone,status)"],
            capture_output=True, text=True, timeout=30
        )
        instances = result.stdout.strip()
        if instances:
            print(f"  🔴 COMPUTE INSTANCES: {instances}")
        else:
            print(f"  No Compute instances ✅")
    except Exception as e:
        print(f"  Error listing instances: {e}")

    # Check Cloud Functions
    try:
        result = subprocess.run(
            ["gcloud", "functions", "list", f"--project={proj}",
             "--format=value(name,status)"],
            capture_output=True, text=True, timeout=30
        )
        functions = result.stdout.strip()
        if functions:
            print(f"  🔴 CLOUD FUNCTIONS: {functions}")
        else:
            print(f"  No Cloud Functions ✅")
    except Exception as e:
        pass  # API may not be enabled

    # Check API keys
    try:
        result = subprocess.run(
            ["gcloud", "services", "api-keys", "list", f"--project={proj}",
             "--format=table(name,displayName,restrictions.apiTargets)"],
            capture_output=True, text=True, timeout=30
        )
        keys = result.stdout.strip()
        if keys:
            print(f"  API KEYS:\n{keys}")
    except Exception as e:
        pass

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
