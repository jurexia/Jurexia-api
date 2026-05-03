"""Check Qdrant collections and counts"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient

QDRANT_URL = os.environ.get("QDRANT_URL",
    "https://d6766dbb-cf4c-40a2-a636-78060cc09ccc.us-east4-0.gcp.cloud.qdrant.io")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4hZwbdZT6esMLx7hjHCi79hD5gLpEAVphmuNGYB3A0Y")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)

print("=" * 60)
print("QDRANT COLLECTIONS STATUS")
print("=" * 60)

collections = client.get_collections().collections
for c in sorted(collections, key=lambda x: x.name):
    try:
        count = client.count(c.name).count
        print(f"  {c.name:40s}  {count:>8,} pts")
    except Exception as e:
        print(f"  {c.name:40s}  ERROR: {e}")

print("=" * 60)

# Check specifically for SCJN
scjn_names = [c.name for c in collections if 'scjn' in c.name.lower()]
if scjn_names:
    print(f"\nSCJN collections found: {scjn_names}")
else:
    print("\n⚠️  NO SCJN COLLECTION FOUND IN QDRANT")

# Also check sentencias_holdings for SCJN-circuito points
print("\nChecking sentencias_holdings for circuito='SCJN' points...")
try:
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    result = client.count(
        collection_name="sentencias_holdings",
        count_filter=Filter(must=[
            FieldCondition(key="circuito", match=MatchValue(value="SCJN"))
        ])
    )
    print(f"  sentencias_holdings with circuito='SCJN': {result.count:,} pts")
except Exception as e:
    print(f"  Error checking SCJN in sentencias_holdings: {e}")

# Check by known circuits
print("\nBreakdown by circuito in sentencias_holdings:")
for circ in ["1", "2", "4", "22", "SCJN"]:
    try:
        result = client.count(
            collection_name="sentencias_holdings",
            count_filter=Filter(must=[
                FieldCondition(key="circuito", match=MatchValue(value=circ))
            ])
        )
        print(f"  circuito={circ:6s}: {result.count:>8,} pts")
    except Exception as e:
        print(f"  circuito={circ:6s}: ERROR: {e}")
