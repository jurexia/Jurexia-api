"""
Check current Gemini cache status and estimate billing.
Also checks if there are any remaining caches.
"""
import os, sys, yaml, json
sys.stdout.reconfigure(encoding='utf-8')

yaml_path = r".env.cloudrun.yaml"
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

API_KEY = config.get('GEMINI_API_KEY', '')
if not API_KEY:
    print("ERROR: GEMINI_API_KEY not found")
    sys.exit(1)

from google import genai

client = genai.Client(api_key=API_KEY)

# 1. Check active caches
print("=" * 60)
print("ACTIVE GEMINI CACHES")
print("=" * 60)
caches = list(client.caches.list())
print(f"Count: {len(caches)}")
if caches:
    for cache in caches:
        name = getattr(cache, 'name', 'unknown')
        display = getattr(cache, 'display_name', 'unknown')
        model = getattr(cache, 'model', 'unknown')
        meta = getattr(cache, 'usage_metadata', None)
        tokens = getattr(meta, 'total_token_count', 0) if meta else 0
        create = getattr(cache, 'create_time', 'unknown')
        expire = getattr(cache, 'expire_time', 'unknown')
        print(f"\n  Name: {name}")
        print(f"  Display: {display}")
        print(f"  Model: {model}")
        print(f"  Tokens: {tokens:,}")
        print(f"  Created: {create}")
        print(f"  Expires: {expire}")
        cost_per_hour = tokens / 1_000_000
        print(f"  Cost/hour: ${cost_per_hour:.2f}")
else:
    print("  No active caches. Current hourly cost: $0.00")

print()
print("=" * 60)
print("BILLING ESTIMATE (based on audit data from earlier)")
print("=" * 60)
print()
print("Earlier audit found 84 caches, each with ~968,370 tokens.")
print("Each cache cost ~$0.97/hour in storage.")
print()
print("Estimated costs based on cache creation times:")
print("  84 caches x $0.97/hr = $81.48/hour combined")
print("  If active for ~4 hours = ~$326")
print("  If active for ~12 hours = ~$978")
print("  If active for ~17 hours = ~$1,385")
print()
print("NOTE: To see exact charges, check:")
print("  1. Google AI Studio: https://aistudio.google.com/app/plan")
print("  2. GCP Console: https://console.cloud.google.com/billing/0103F3-3CFE3A-590A27")
print("  3. GCP Cost Table: https://console.cloud.google.com/billing/0103F3-3CFE3A-590A27/reports")
print()
print("For credits info, check:")
print("  https://console.cloud.google.com/billing/0103F3-3CFE3A-590A27/credits")
