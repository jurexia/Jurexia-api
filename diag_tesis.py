#!/usr/bin/env python3
"""
Diagnostic: Compare tesis WITH and WITHOUT metadata in jurisprudencia_nacional.
Samples points to identify differences between ingestion batches.
"""
import os
from qdrant_client import QdrantClient

QDRANT_URL = os.environ.get("QDRANT_URL", "https://d6766dbb-cf4c-40a2-a636-78060cc09ccc.us-east4-0.gcp.cloud.qdrant.io")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4hZwbdZT6esMLx7hjHCi79hD5gLpEAVphmuNGYB3A0Y")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

COLLECTION = "jurisprudencia_nacional"

# Get collection info
info = client.get_collection(COLLECTION)
print(f"Collection: {COLLECTION}")
print(f"Total points: {info.points_count:,}")

# Sample points and classify by metadata presence
with_meta = []
without_meta = []
all_keys = set()
offset = None

for batch in range(40):
    result = client.scroll(
        collection_name=COLLECTION,
        limit=500,
        offset=offset,
        with_payload=True,
    )
    points, next_offset = result
    for p in points:
        payload = p.payload or {}
        all_keys.update(payload.keys())
        
        # Check for localization metadata
        has_registro = bool(payload.get("registro"))
        has_instancia = bool(payload.get("instancia"))
        has_materia = bool(payload.get("materia"))
        has_tesis = bool(payload.get("tesis"))
        has_tipo = bool(payload.get("tipo"))
        
        has_localization = has_registro or has_instancia or has_materia or has_tesis or has_tipo
        
        entry = {
            "id": p.id,
            "keys": sorted(payload.keys()),
            "origen": payload.get("origen", "N/A"),
            "ref": payload.get("ref", "N/A"),
            "registro": payload.get("registro"),
            "instancia": payload.get("instancia"),
            "materia": payload.get("materia"),
            "tesis": payload.get("tesis"),
            "tipo": payload.get("tipo"),
            "texto_preview": (payload.get("texto") or "")[:100],
        }
        
        if has_localization:
            with_meta.append(entry)
        else:
            without_meta.append(entry)
    
    if next_offset is None:
        break
    offset = next_offset

print(f"\nAll payload keys across collection: {sorted(all_keys)}")
print(f"\nPoints WITH metadata: {len(with_meta):,}")
print(f"Points WITHOUT metadata: {len(without_meta):,}")

# Show examples of each
print("\n" + "=" * 70)
print("  EXAMPLES WITH METADATA (first 3)")
print("=" * 70)
for entry in with_meta[:3]:
    print(f"\n  ID: {entry['id']}")
    print(f"  Keys: {entry['keys']}")
    print(f"  origen: {entry['origen']}")
    print(f"  ref: {entry['ref']}")
    print(f"  registro: {entry['registro']}")
    print(f"  instancia: {entry['instancia']}")
    print(f"  materia: {entry['materia']}")
    print(f"  tesis: {entry['tesis']}")
    print(f"  tipo: {entry['tipo']}")
    print(f"  texto: {entry['texto_preview']}...")

print("\n" + "=" * 70)
print("  EXAMPLES WITHOUT METADATA (first 3)")
print("=" * 70)
for entry in without_meta[:3]:
    print(f"\n  ID: {entry['id']}")
    print(f"  Keys: {entry['keys']}")
    print(f"  origen: {entry['origen']}")
    print(f"  ref: {entry['ref']}")
    print(f"  texto: {entry['texto_preview']}...")

# Analyze 'origen' distribution
print("\n" + "=" * 70)
print("  ORIGEN DISTRIBUTION")
print("=" * 70)
origen_with = {}
origen_without = {}
for e in with_meta:
    o = e["origen"]
    origen_with[o] = origen_with.get(o, 0) + 1
for e in without_meta:
    o = e["origen"]
    origen_without[o] = origen_without.get(o, 0) + 1

print("\n  WITH metadata by origen:")
for o in sorted(origen_with, key=origen_with.get, reverse=True)[:15]:
    print(f"    {origen_with[o]:>5} | {o}")

print("\n  WITHOUT metadata by origen:")
for o in sorted(origen_without, key=origen_without.get, reverse=True)[:15]:
    print(f"    {origen_without[o]:>5} | {o}")
