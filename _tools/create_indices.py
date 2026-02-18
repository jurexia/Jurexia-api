#!/usr/bin/env python3
"""Create payload indices on leyes_estatales after collection recreation."""
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

QDRANT_URL = "https://d6766dbb-cf4c-40a2-a636-78060cc09ccc.us-east4-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4hZwbdZT6esMLx7hjHCi79hD5gLpEAVphmuNGYB3A0Y"

c = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Create keyword indices for all filterable fields
KEYWORD_FIELDS = ["entidad", "origen", "tipo_codigo", "jurisdiccion", "categoria", "ref"]

for field in KEYWORD_FIELDS:
    print(f"Creating keyword index for '{field}'...")
    c.create_payload_index(
        collection_name="leyes_estatales",
        field_name=field,
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print(f"  ✅ {field}")

# Also create a text index for 'texto' (useful for full-text search)
print("Creating text index for 'texto'...")
c.create_payload_index(
    collection_name="leyes_estatales",
    field_name="texto",
    field_schema=PayloadSchemaType.TEXT,
)
print("  ✅ texto")

print("\nAll indices created!")

# Verify
info = c.get_collection("leyes_estatales")
indexed_fields = info.payload_schema
print(f"\nIndexed fields: {list(indexed_fields.keys())}")
