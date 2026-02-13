#!/usr/bin/env python3
"""
Diagnostic 2: Check what fields are present/missing per point.
Focus on 'registro' field specifically.
"""
import os
from qdrant_client import QdrantClient

QDRANT_URL = os.environ.get("QDRANT_URL", "https://d6766dbb-cf4c-40a2-a636-78060cc09ccc.us-east4-0.gcp.cloud.qdrant.io")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4hZwbdZT6esMLx7hjHCi79hD5gLpEAVphmuNGYB3A0Y")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

COLLECTION = "jurisprudencia_nacional"

# Categorize ALL points by their payload key sets
key_sets = {}
total = 0
offset = None

for batch in range(50):
    result = client.scroll(
        collection_name=COLLECTION,
        limit=500,
        offset=offset,
        with_payload=True,
    )
    points, next_offset = result
    for p in points:
        total += 1
        payload = p.payload or {}
        ks = frozenset(payload.keys())
        if ks not in key_sets:
            key_sets[ks] = {"count": 0, "example_id": p.id, "example_payload": {}}
        key_sets[ks]["count"] += 1
        if key_sets[ks]["count"] == 1:
            # Store a simplified example
            key_sets[ks]["example_payload"] = {
                k: (str(v)[:80] if isinstance(v, str) else v) 
                for k, v in payload.items() 
                if k != "texto"  # skip texto (too long)
            }
            # Add texto preview
            key_sets[ks]["example_payload"]["texto_preview"] = (payload.get("texto") or "")[:150]
    
    if next_offset is None:
        break
    offset = next_offset

print(f"Total points scanned: {total:,}")
print(f"Distinct payload schemas: {len(key_sets)}")

for i, (ks, info) in enumerate(sorted(key_sets.items(), key=lambda x: -x[1]["count"]), 1):
    print(f"\n{'='*70}")
    print(f"SCHEMA {i}: {info['count']:,} points ({info['count']*100/total:.1f}%)")
    print(f"Fields: {sorted(ks)}")
    print(f"Example ID: {info['example_id']}")
    for k, v in sorted(info['example_payload'].items()):
        print(f"  {k}: {v}")
