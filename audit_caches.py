"""
List all active Gemini caches (read-only, no deletion).
Shows cache details for billing analysis.
"""
import os
import sys
import yaml
from datetime import datetime, timezone

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

yaml_path = r"C:\Users\jdmju\.gemini\antigravity\playground\obsidian-expanse\jurexia-api-git\.env.cloudrun.yaml"
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

API_KEY = config.get('GEMINI_API_KEY', '')
if not API_KEY:
    print("ERROR: GEMINI_API_KEY not found")
    exit(1)

from google import genai

client = genai.Client(api_key=API_KEY)

print("=" * 70)
print("  GEMINI CACHE AUDIT - Active Caches")
print("=" * 70)

caches = list(client.caches.list())

if not caches:
    print("\nNo active caches found. No storage charges accruing.")
    exit(0)

now = datetime.now(timezone.utc)
total_cost = 0

for i, cache in enumerate(caches):
    name = cache.name
    display = getattr(cache, 'display_name', 'N/A')
    model = getattr(cache, 'model', 'N/A')
    
    # Token count
    usage = getattr(cache, 'usage_metadata', None)
    tokens = 0
    if usage:
        tokens = getattr(usage, 'total_token_count', 0)
    
    create_time = getattr(cache, 'create_time', None)
    expire_time = getattr(cache, 'expire_time', None)
    
    # Calculate hours alive
    hours_alive = 0
    if create_time:
        if isinstance(create_time, str):
            ct = datetime.fromisoformat(create_time.replace('Z', '+00:00'))
        else:
            ct = create_time if create_time.tzinfo else create_time.replace(tzinfo=timezone.utc)
        hours_alive = (now - ct).total_seconds() / 3600
    
    # Cost estimate (storage only)
    hourly_cost = (tokens / 1_000_000) * 1.0  # $1/M tokens/hour for Flash
    total_cache_cost = hourly_cost * hours_alive
    total_cost += total_cache_cost
    
    print(f"\n  Cache [{i+1}]")
    print(f"  Name:         {name}")
    print(f"  Display Name: {display}")
    print(f"  Model:        {model}")
    print(f"  Tokens:       {tokens:,}")
    print(f"  Created:      {create_time}")
    print(f"  Expires:      {expire_time}")
    print(f"  Hours Alive:  {hours_alive:.1f}h")
    print(f"  Cost/hour:    ${hourly_cost:.2f}")
    print(f"  Est. Cost:    ${total_cache_cost:.2f}")

print(f"\n{'=' * 70}")
print(f"  TOTAL ESTIMATED STORAGE COST: ${total_cost:.2f}")
print(f"  (Does not include input/output token charges)")
print(f"{'=' * 70}")
