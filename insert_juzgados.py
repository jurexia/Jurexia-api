"""Insert juzgados data into Supabase using the anon key + RLS insert policy."""
import json
import requests

SUPABASE_URL = "https://ukcuzhwmmfwvcedvhfll.supabase.co"
ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVrY3V6aHdtbWZ3dmNlZHZoZmxsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzAxNTIyOTQsImV4cCI6MjA4NTcyODI5NH0.agVcqH3VByMqxcLH2ErIsxHpN23nJklzcm36W_svTy0"

with open("juzgados_distrito.json", "r", encoding="utf-8") as f:
    data = json.load(f)

headers = {
    "apikey": ANON_KEY,
    "Authorization": f"Bearer {ANON_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}

# Fix ciudad to just be the estado
for d in data:
    d["ciudad"] = d["estado"]
    d.pop("titular", None)
    d.pop("cv_x", None)

# Insert in batches
batch_size = 50
errors = 0
total_inserted = 0
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/juzgados_distrito",
        headers=headers,
        json=batch,
        timeout=30,
    )
    if resp.status_code in (200, 201):
        total_inserted += len(batch)
        print(f"  ✅ Batch {i//batch_size + 1}: {len(batch)} rows (total: {total_inserted})")
    else:
        print(f"  ❌ Batch {i//batch_size + 1}: {resp.status_code} — {resp.text[:300]}")
        errors += 1

if errors:
    print(f"\n⚠️  {errors} batch(es) failed")
else:
    print(f"\n✅ All {total_inserted} rows inserted successfully!")
