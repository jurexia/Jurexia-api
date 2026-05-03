"""
SCJN JSON Quality Audit
========================
Verifica que los 26,726 JSONs tengan la información necesaria para:
1. Búsqueda semántica (holding de calidad)
2. Render de tarjeta (expediente, sala, materia, sentido)
3. Visualización de PDF (pdf_url o datos para construirlo)
"""
import json, os, sys, statistics
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent / "data" / "sentencias_scjn"

print("=" * 70)
print("  SCJN JSON QUALITY AUDIT")
print("=" * 70)

# ── 1. Count and structure ─────────────────────────────────────────────
all_jsons = list(DATA_DIR.rglob("*.json"))
# Exclude checkpoint/meta files
all_jsons = [j for j in all_jsons if not j.name.startswith("_")]
print(f"\nTotal JSONs found: {len(all_jsons):,}")

# Show directory structure
print("\nDirectory structure:")
for sala_dir in sorted(DATA_DIR.iterdir()):
    if sala_dir.is_dir() and not sala_dir.name.startswith("_"):
        tipos = [d.name for d in sala_dir.iterdir() if d.is_dir()]
        count = sum(1 for _ in sala_dir.rglob("*.json"))
        print(f"  {sala_dir.name}/ ({count:,} JSONs)")
        for t in sorted(tipos):
            tc = sum(1 for _ in (sala_dir / t).rglob("*.json"))
            print(f"    {t}/ ({tc:,})")

# ── 2. Sample 5 JSONs to show full structure ───────────────────────────
print("\n" + "=" * 70)
print("  SAMPLE JSON KEYS (first 3)")
print("=" * 70)
samples = all_jsons[:3]
for s in samples:
    data = json.loads(s.read_text(encoding="utf-8"))
    print(f"\n  FILE: {s.relative_to(DATA_DIR)}")
    print(f"  KEYS: {list(data.keys())}")
    # Show key field values
    for key in ["asunto_id", "expediente", "sala", "tipo_asunto", "sentido", 
                 "materia", "tema_juridico", "ministro_ponente", "fecha_sentencia",
                 "anio", "is_ef_eligible", "pdf_url", "url_pdf", "gcs_url"]:
        val = data.get(key, "<<MISSING>>")
        if isinstance(val, str) and len(val) > 100:
            val = val[:100] + "..."
        print(f"    {key}: {val}")
    # Holding preview
    h = data.get("holding", "")
    print(f"    holding: [{len(h)} chars] {h[:150]}...")

# ── 3. Field completeness analysis ────────────────────────────────────
print("\n" + "=" * 70)
print("  FIELD COMPLETENESS (all JSONs)")
print("=" * 70)

CRITICAL_FIELDS = [
    "asunto_id", "expediente", "sala", "tipo_asunto", "sentido",
    "materia", "holding", "tema_juridico", "fecha_sentencia",
    "ministro_ponente", "anio", "is_ef_eligible", "estudio_de_fondo",
]
PDF_FIELDS = ["pdf_url", "url_pdf", "gcs_url", "pdf_gcs", "archivo_pdf"]

field_present = Counter()
field_nonempty = Counter()
holding_lengths = []
ef_lengths = []
salas = Counter()
materias = Counter()
sentidos = Counter()
tipos = Counter()
anios = Counter()
has_any_pdf_field = 0

for jf in all_jsons:
    try:
        data = json.loads(jf.read_text(encoding="utf-8"))
    except:
        continue
    
    for f in CRITICAL_FIELDS:
        if f in data:
            field_present[f] += 1
            v = data[f]
            if v and str(v).strip() and str(v).strip().lower() not in ("none", "null", ""):
                field_nonempty[f] += 1
    
    # PDF fields
    pdf_found = False
    for pf in PDF_FIELDS:
        if pf in data:
            field_present[pf] += 1
            if data[pf]:
                field_nonempty[pf] += 1
                pdf_found = True
    if pdf_found:
        has_any_pdf_field += 1
    
    # Stats
    h = data.get("holding", "")
    if h:
        holding_lengths.append(len(h))
    ef = data.get("estudio_de_fondo", "")
    if ef:
        ef_lengths.append(len(ef))
    if data.get("sala"):
        salas[data["sala"]] += 1
    if data.get("materia"):
        materias[data["materia"]] += 1
    if data.get("sentido"):
        sentidos[data["sentido"]] += 1
    if data.get("tipo_asunto"):
        tipos[data["tipo_asunto"]] += 1
    if data.get("anio"):
        anios[data["anio"]] += 1

total = len(all_jsons)
print(f"\n{'Field':<30s} {'Present':>10s} {'Non-empty':>10s} {'%':>6s}")
print("-" * 60)
for f in CRITICAL_FIELDS + PDF_FIELDS:
    p = field_present.get(f, 0)
    ne = field_nonempty.get(f, 0)
    pct = f"{ne/total*100:.1f}%" if total else "—"
    marker = " ⚠" if ne < total * 0.5 and f in ["holding", "expediente", "sala"] else ""
    print(f"  {f:<28s} {p:>8,} {ne:>10,} {pct:>6s}{marker}")

print(f"\n  Has ANY pdf field:          {has_any_pdf_field:>8,}  ({has_any_pdf_field/total*100:.1f}%)")

# ── 4. Holding quality ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  HOLDING QUALITY")
print("=" * 70)
if holding_lengths:
    print(f"  Total with holding:    {len(holding_lengths):,} / {total:,} ({len(holding_lengths)/total*100:.1f}%)")
    print(f"  Mean length:           {statistics.mean(holding_lengths):.0f} chars")
    print(f"  Median:                {statistics.median(holding_lengths):.0f} chars")
    print(f"  Min:                   {min(holding_lengths)} chars")
    print(f"  Max:                   {max(holding_lengths):,} chars")
    print(f"  < 50 chars (bad):      {sum(1 for l in holding_lengths if l < 50):,}")
    print(f"  < 100 chars (weak):    {sum(1 for l in holding_lengths if l < 100):,}")
    print(f"  > 200 chars (good):    {sum(1 for l in holding_lengths if l > 200):,}")
    print(f"  > 500 chars (strong):  {sum(1 for l in holding_lengths if l > 500):,}")

# ── 5. EF quality ───────────────────────────────────────────────────────
print("\n  Estudio de Fondo:")
if ef_lengths:
    print(f"  Total with EF:         {len(ef_lengths):,} / {total:,} ({len(ef_lengths)/total*100:.1f}%)")
    print(f"  Mean length:           {statistics.mean(ef_lengths):.0f} chars")
else:
    print(f"  Total with EF:         0")

# ── 6. Distribution breakdown ──────────────────────────────────────────
print("\n" + "=" * 70)
print("  DISTRIBUTION BREAKDOWN")
print("=" * 70)

print("\n  Salas:")
for s, c in salas.most_common():
    print(f"    {s:<30s} {c:>6,}")

print("\n  Materias (top 10):")
for m, c in materias.most_common(10):
    print(f"    {m:<30s} {c:>6,}")

print("\n  Sentidos (top 10):")
for s, c in sentidos.most_common(10):
    print(f"    {s:<30s} {c:>6,}")

print("\n  Tipos de asunto (top 10):")
for t, c in tipos.most_common(10):
    print(f"    {t:<30s} {c:>6,}")

print("\n  Anios (top 10):")
for a, c in sorted(anios.items(), key=lambda x: -x[1])[:10]:
    print(f"    {a:<10} {c:>6,}")

# ── 7. GCS PDF availability check ──────────────────────────────────────
print("\n" + "=" * 70)
print("  GCS PDF URL RECONSTRUCTION TEST")
print("=" * 70)

# Check the GCS upload progress file
gcs_progress = Path(__file__).parent / "_tools" / ".gcs_upload_progress_scjn.json"
if gcs_progress.exists():
    gcs_data = json.loads(gcs_progress.read_text(encoding="utf-8"))
    if isinstance(gcs_data, dict):
        uploaded_ids = gcs_data.get("uploaded", gcs_data.get("done", []))
        if isinstance(uploaded_ids, list):
            print(f"  GCS uploaded IDs: {len(uploaded_ids):,}")
        elif isinstance(gcs_data, list):
            print(f"  GCS uploaded IDs: {len(gcs_data):,}")
        else:
            print(f"  GCS progress keys: {list(gcs_data.keys())[:10]}")
    elif isinstance(gcs_data, list):
        print(f"  GCS uploaded IDs: {len(gcs_data):,}")
else:
    print("  ⚠ No GCS upload progress file found")

# Check what the ingest script uses for pdf_url
print("\n  Checking ingest_scjn_qdrant.py for pdf_url construction...")
ingest_file = Path(__file__).parent / "_tools" / "ingest_scjn_qdrant.py"
if ingest_file.exists():
    content = ingest_file.read_text(encoding="utf-8")
    if "pdf_url" in content:
        for line in content.split("\n"):
            if "pdf_url" in line.lower():
                print(f"    {line.strip()}")
    else:
        print("    ⚠ NO pdf_url field in ingest script!")
else:
    print("    ⚠ ingest_scjn_qdrant.py not found")

# ── 8. Sample asunto_ids vs GCS ────────────────────────────────────────
print("\n  Sample asunto_ids from JSONs:")
sample_ids = []
for jf in all_jsons[:10]:
    data = json.loads(jf.read_text(encoding="utf-8"))
    aid = data.get("asunto_id", "")
    print(f"    {jf.parent.parent.name}/{jf.parent.name}/{jf.stem} -> asunto_id={aid}")
    if aid:
        sample_ids.append(str(aid))

print("\n" + "=" * 70)
print("  VERDICT")
print("=" * 70)

issues = []
if len(holding_lengths) < total * 0.8:
    issues.append(f"Only {len(holding_lengths)/total*100:.0f}% have holdings")
if has_any_pdf_field == 0:
    issues.append("NO pdf_url field in any JSON — need to construct from asunto_id + GCS base")
if sum(1 for l in holding_lengths if l < 100) > total * 0.2:
    issues.append(f"{sum(1 for l in holding_lengths if l<100)/total*100:.0f}% have very short holdings")

if issues:
    print("  ISSUES FOUND:")
    for i in issues:
        print(f"    - {i}")
else:
    print("  ALL CHECKS PASSED")
