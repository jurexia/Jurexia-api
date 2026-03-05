"""
extract_mercantil_corpus.py — Iurexia PDF → Clean TXT Converter
================================================================
Downloads 4 mercantile-law PDFs from Supabase Storage and converts
them into clean, article-structured .txt files for the Gemini
context cache corpus.

Output files:
  cache_corpus_mercantil/
    01_codigo_comercio.txt
    02_ley_titulos_credito.txt
    03_ley_sociedades_mercantiles.txt
    04_ley_contrato_seguro.txt

Cleaning rules:
  ✅ Keeps: full article text, chapter/title/book structure headings
  ❌ Removes: transitorios, derogated articles, page numbers, headers/footers,
     DOF reform dates, editorial noise, blank lines

Usage:
  1. Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env (or env vars)
  2. pip install pymupdf supabase python-dotenv
  3. python scripts/extract_mercantil_corpus.py

  OR if you have local PDF files:
  python scripts/extract_mercantil_corpus.py --local path/to/pdfs/
"""

import os
import re
import sys
import argparse
import unicodedata
from pathlib import Path
from typing import Optional

# ── PDF extraction library ──────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
except ImportError:
    print("❌ PyMuPDF not installed. Run: pip install pymupdf")
    sys.exit(1)

# ── Config ──────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "cache_corpus_mercantil"

# Map of output filename → Supabase storage path (adjust paths as needed)
# These are BEST GUESSES — user may need to adjust the storage paths
LAWS = {
    "01_codigo_comercio.txt": {
        "display_name": "Código de Comercio",
        "supabase_path": "Federal/Codigos/Codigo_de_Comercio.pdf",
        "alt_paths": [
            "Federal/Codigo_de_Comercio.pdf",
            "Codigos/Codigo_de_Comercio.pdf",
            "leyes-federales/Codigo_de_Comercio.pdf",
        ],
    },
    "02_ley_titulos_credito.txt": {
        "display_name": "Ley General de Títulos y Operaciones de Crédito",
        "supabase_path": "Federal/Leyes/LGTOC.pdf",
        "alt_paths": [
            "Federal/Ley_General_de_Titulos_y_Operaciones_de_Credito.pdf",
            "Federal/Leyes/Ley_General_Titulos_Operaciones_Credito.pdf",
            "leyes-federales/LGTOC.pdf",
        ],
    },
    "03_ley_sociedades_mercantiles.txt": {
        "display_name": "Ley General de Sociedades Mercantiles",
        "supabase_path": "Federal/Leyes/LGSM.pdf",
        "alt_paths": [
            "Federal/Ley_General_de_Sociedades_Mercantiles.pdf",
            "Federal/Leyes/Ley_General_Sociedades_Mercantiles.pdf",
            "leyes-federales/LGSM.pdf",
        ],
    },
    "04_ley_contrato_seguro.txt": {
        "display_name": "Ley sobre el Contrato de Seguro",
        "supabase_path": "Federal/Leyes/LCS.pdf",
        "alt_paths": [
            "Federal/Ley_sobre_el_Contrato_de_Seguro.pdf",
            "Federal/Leyes/Ley_Contrato_Seguro.pdf",
            "leyes-federales/LCS.pdf",
        ],
    },
}

# ── Cleaning Patterns ──────────────────────────────────────────────────────

# Patterns to REMOVE entirely
REMOVE_PATTERNS = [
    # Page numbers (standalone)
    re.compile(r"^\s*\d{1,4}\s*$"),
    # DOF headers/footers
    re.compile(r"^.*Diario Oficial de la Federación.*$", re.IGNORECASE),
    re.compile(r"^.*D\.?\s*O\.?\s*F\.?.*\d{1,2}\s+de\s+\w+\s+de\s+\d{4}.*$", re.IGNORECASE),
    # Reform/publication dates in parentheses
    re.compile(r"^\s*\(Reformad[oa].*D\.?O\.?F\.?.*\d{4}.*\)\s*$", re.IGNORECASE),
    re.compile(r"^\s*\(Adicionad[oa].*D\.?O\.?F\.?.*\d{4}.*\)\s*$", re.IGNORECASE),
    re.compile(r"^\s*\(Derogad[oa].*D\.?O\.?F\.?.*\d{4}.*\)\s*$", re.IGNORECASE),
    # "Última reforma publicada..."
    re.compile(r"^.*[ÚU]ltima reforma publicada.*$", re.IGNORECASE),
    # "Nueva Ley publicada..."
    re.compile(r"^.*Nueva Ley publicada.*$", re.IGNORECASE),
    # "Ley publicada en el..."
    re.compile(r"^.*Ley publicada en el.*$", re.IGNORECASE),
    # CÁMARA DE DIPUTADOS headers
    re.compile(r"^.*C[ÁA]MARA DE DIPUTADOS.*$", re.IGNORECASE),
    re.compile(r"^.*H\.\s*CONGRESO DE LA UNI[ÓO]N.*$", re.IGNORECASE),
    re.compile(r"^.*Secretar[ií]a General.*$", re.IGNORECASE),
    re.compile(r"^.*Secretar[ií]a de Servicios Parlamentarios.*$", re.IGNORECASE),
    re.compile(r"^.*Direcci[óo]n General de Servicios de Documentaci[óo]n.*$", re.IGNORECASE),
    # Repeated copyright/page markers
    re.compile(r"^\s*\d+\s+de\s+\d+\s*$"),
    # Empty parenthetical reform notes inline (will be handled separately)
]

# Pattern to detect TRANSITORIOS section
TRANSITORIOS_START = re.compile(
    r"^\s*(TRANSITORIOS?|ART[IÍ]CULOS?\s+TRANSITORIOS?)\s*$",
    re.IGNORECASE,
)

# Pattern to detect derogated articles
DEROGATED_PATTERN = re.compile(
    r"^\s*Art[íi]culo\s+\d+[\w\s]*[-–—\.]\s*(Se\s+deroga|Derogado)\s*\.?\s*$",
    re.IGNORECASE,
)

# Structural headings to KEEP (books, titles, chapters, sections)
STRUCTURE_HEADINGS = re.compile(
    r"^\s*(LIBRO\s+\w+|T[IÍ]TULO\s+\w+|CAP[IÍ]TULO\s+\w+|SECCI[OÓ]N\s+\w+)\b",
    re.IGNORECASE,
)

# Article start pattern
ARTICLE_PATTERN = re.compile(
    r"^\s*Art[íi]culo\s+(\d+[\w\s]*?)[\.\-–—]",
    re.IGNORECASE,
)

# Inline reform annotations to strip (e.g., "(Reformado, D.O.F. 25 de mayo de 2000)")
INLINE_REFORM = re.compile(
    r"\s*\((Reformad[oa]|Adicionad[oa]|Derogad[oa]|Fe de erratas|Nota del editor)[^)]*\)",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Normalize unicode characters and whitespace."""
    # Normalize unicode
    text = unicodedata.normalize("NFC", text)
    # Fix common PDF extraction artifacts
    text = text.replace("\u00ad", "")  # soft hyphens
    text = text.replace("\ufeff", "")  # BOM
    text = text.replace("\u200b", "")  # zero-width space
    # Normalize quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    # Normalize dashes
    text = text.replace("\u2013", "–").replace("\u2014", "—")
    return text


def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    full_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        full_text.append(text)
    doc.close()
    return "\n".join(full_text)


def clean_law_text(raw_text: str, law_name: str) -> str:
    """
    Clean extracted PDF text according to rules:
    1. Keep article text with full structure (chapters, titles, books)
    2. Remove transitorios
    3. Remove derogated articles
    4. Remove page numbers, headers/footers, DOF reform dates
    5. Strip inline reform annotations
    6. Collapse excessive whitespace
    """
    lines = raw_text.split("\n")
    cleaned_lines = []
    in_transitorios = False
    in_derogated = False
    current_article = ""
    skip_until_next_article = False

    for line in lines:
        line = normalize_text(line)
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            # Keep one blank line between sections
            if cleaned_lines and cleaned_lines[-1].strip():
                cleaned_lines.append("")
            continue

        # Check for TRANSITORIOS section — stop processing
        if TRANSITORIOS_START.match(stripped):
            in_transitorios = True
            continue

        if in_transitorios:
            # Check if we hit a new structural section after transitorios
            # (some PDFs have multiple reform transitorios followed by more content)
            if STRUCTURE_HEADINGS.match(stripped) and not re.match(
                r"^\s*(TRANSITORIOS?|ART[IÍ]CULO)", stripped, re.IGNORECASE
            ):
                in_transitorios = False
                # Fall through to process this line
            else:
                continue

        # Check for derogated articles — skip them entirely
        if DEROGATED_PATTERN.match(stripped):
            skip_until_next_article = True
            continue

        if skip_until_next_article:
            # Resume when we hit the next article or structural heading
            if ARTICLE_PATTERN.match(stripped) or STRUCTURE_HEADINGS.match(stripped):
                skip_until_next_article = False
            else:
                continue

        # Check removal patterns
        should_remove = False
        for pattern in REMOVE_PATTERNS:
            if pattern.match(stripped):
                should_remove = True
                break
        if should_remove:
            continue

        # Strip inline reform annotations
        cleaned_line = INLINE_REFORM.sub("", stripped)

        # Skip if line became empty after stripping
        if not cleaned_line.strip():
            continue

        cleaned_lines.append(cleaned_line)

    # Post-processing: collapse multiple blank lines
    result_lines = []
    prev_blank = False
    for line in cleaned_lines:
        if not line.strip():
            if not prev_blank:
                result_lines.append("")
                prev_blank = True
        else:
            result_lines.append(line)
            prev_blank = False

    # Add header
    header = f"{'='*60}\n{law_name}\n{'='*60}\n"
    result = header + "\n".join(result_lines).strip()

    return result


def download_from_supabase(law_config: dict, bucket: str = "legal-docs") -> Optional[bytes]:
    """Download a PDF from Supabase Storage. Tries primary path, then alternates."""
    from dotenv import load_dotenv
    load_dotenv()

    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY", "")

    if not supabase_url or not supabase_key:
        print("⚠️  SUPABASE_URL or SUPABASE_SERVICE_KEY not set")
        return None

    # Try direct public URL first
    import urllib.request
    import urllib.error

    paths_to_try = [law_config["supabase_path"]] + law_config.get("alt_paths", [])

    for path in paths_to_try:
        # Try public URL
        public_url = f"{supabase_url}/storage/v1/object/public/{bucket}/{path}"
        try:
            print(f"  Trying: {public_url}")
            req = urllib.request.Request(public_url)
            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status == 200:
                    data = response.read()
                    print(f"  ✅ Downloaded ({len(data):,} bytes)")
                    return data
        except urllib.error.HTTPError as e:
            print(f"  ❌ HTTP {e.code}: {path}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"  ❌ Could not download from any path")
    return None


def list_supabase_bucket(bucket: str = "legal-docs") -> list:
    """List files in a Supabase Storage bucket to find the right paths."""
    from dotenv import load_dotenv
    load_dotenv()

    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY", "")

    if not supabase_url or not supabase_key:
        print("⚠️  SUPABASE_URL or SUPABASE_SERVICE_KEY not set")
        return []

    import urllib.request
    import json

    # List root of bucket
    api_url = f"{supabase_url}/storage/v1/object/list/{bucket}"
    headers = {
        "Authorization": f"Bearer {supabase_key}",
        "apikey": supabase_key,
        "Content-Type": "application/json",
    }

    try:
        data = json.dumps({"prefix": "", "limit": 1000, "offset": 0}).encode()
        req = urllib.request.Request(api_url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
            return result
    except Exception as e:
        print(f"Error listing bucket: {e}")
        return []


def process_local_pdfs(pdf_dir: str):
    """Process PDF files from a local directory."""
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        print(f"❌ Directory not found: {pdf_dir}")
        sys.exit(1)

    pdfs = sorted(pdf_path.glob("*.pdf"))
    if not pdfs:
        print(f"❌ No PDF files found in {pdf_dir}")
        sys.exit(1)

    print(f"Found {len(pdfs)} PDF files in {pdf_dir}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for pdf in pdfs:
        print(f"\n{'─'*50}")
        print(f"Processing: {pdf.name}")

        # Extract text
        raw_text = extract_pdf_text(str(pdf))
        print(f"  Raw text: {len(raw_text):,} chars")

        # Determine law name from filename
        law_name = pdf.stem.replace("_", " ").title()

        # Clean text
        cleaned = clean_law_text(raw_text, law_name)
        tokens_est = len(cleaned) // 4
        print(f"  Cleaned: {len(cleaned):,} chars (~{tokens_est:,} tokens)")

        # Determine output filename
        output_name = None
        for key, config in LAWS.items():
            # Try to match by name similarity
            if any(part.lower() in pdf.stem.lower()
                   for part in config["display_name"].lower().split()):
                output_name = key
                break

        if not output_name:
            # Fallback: use the PDF filename
            output_name = pdf.stem.lower().replace(" ", "_") + ".txt"

        output_path = OUTPUT_DIR / output_name
        output_path.write_text(cleaned, encoding="utf-8")
        print(f"  ✅ Saved: {output_path}")


def process_from_supabase():
    """Download PDFs from Supabase and process them."""
    import tempfile

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # First, try to list the bucket to find the correct paths
    print("🔍 Listing Supabase bucket 'legal-docs' to find PDF paths...\n")
    files = list_supabase_bucket()

    if files:
        print("Files/folders found in bucket root:")
        for f in files[:30]:
            name = f.get("name", "?")
            is_folder = f.get("id") is None
            print(f"  {'📁' if is_folder else '📄'} {name}")
        print()

    for output_name, config in LAWS.items():
        print(f"\n{'─'*50}")
        print(f"📥 {config['display_name']}")

        # Download PDF
        pdf_data = download_from_supabase(config)

        if not pdf_data:
            print(f"  ⚠️  Skipped — could not download. Check path in script.")
            continue

        # Save to temp file and extract
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_data)
            tmp_path = tmp.name

        try:
            raw_text = extract_pdf_text(tmp_path)
            print(f"  Raw text: {len(raw_text):,} chars")

            cleaned = clean_law_text(raw_text, config["display_name"])
            tokens_est = len(cleaned) // 4
            print(f"  Cleaned: {len(cleaned):,} chars (~{tokens_est:,} tokens)")

            output_path = OUTPUT_DIR / output_name
            output_path.write_text(cleaned, encoding="utf-8")
            print(f"  ✅ Saved: {output_path}")
        finally:
            os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(
        description="Extract and clean Mexican mercantile law PDFs for Genio Mercantil corpus"
    )
    parser.add_argument(
        "--local",
        type=str,
        help="Path to directory with local PDF files (skip Supabase download)",
    )
    parser.add_argument(
        "--list-bucket",
        action="store_true",
        help="Only list files in the Supabase bucket (for debugging paths)",
    )
    args = parser.parse_args()

    print("═" * 60)
    print("  IUREXIA — Mercantil Corpus Extractor")
    print("═" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    if args.list_bucket:
        files = list_supabase_bucket()
        if files:
            for f in files:
                name = f.get("name", "?")
                is_folder = f.get("id") is None
                size = f.get("metadata", {}).get("size", "")
                print(f"  {'📁' if is_folder else '📄'} {name}  {f'({size} bytes)' if size else ''}")
        return

    if args.local:
        process_local_pdfs(args.local)
    else:
        process_from_supabase()

    print(f"\n{'═'*60}")
    print(f"  ✅ Done! Corpus files saved to: {OUTPUT_DIR}")
    print()

    # Show summary
    if OUTPUT_DIR.exists():
        total_chars = 0
        for f in sorted(OUTPUT_DIR.glob("*.txt")):
            chars = len(f.read_text(encoding="utf-8"))
            tokens = chars // 4
            total_chars += chars
            print(f"  {f.name}: {chars:,} chars (~{tokens:,} tokens)")

        total_tokens = total_chars // 4
        print(f"\n  Total: {total_chars:,} chars (~{total_tokens:,} tokens)")
        print(f"  Estimated cache cost: ${total_tokens / 1_000_000:.2f}/hour")


if __name__ == "__main__":
    main()
