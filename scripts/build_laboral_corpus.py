"""
build_laboral_corpus.py - Iurexia Laboral Corpus Builder
=========================================================
Reads LFT, LSS, LFTSE, Ley INFONAVIT, and extracts Art. 123 CPEUM.
Applies cleaning and outputs token-optimized TXT files.

Sources:
  - C:\Proyectos\LEYES_LIMPIAS_TXT\LEYES_FEDERALES\
  - cache_corpus\CPEUM.txt (Art. 123 only)

Target: <= 450,000 tokens
"""

import re
import sys
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# -- Paths --
SOURCE_DIR = Path(r"C:\Proyectos\LEYES_LIMPIAS_TXT\LEYES_FEDERALES")
API_DIR = Path(__file__).resolve().parent.parent
# Trimmed CPEUM only has Arts 1-30 + 103-107; use full version from tmp
CPEUM_PATH = Path(r"C:\tmp\CPEUM_full_utf8.txt")
OUTPUT_DIR = API_DIR / "cache_corpus_laboral"

# -- Law definitions --
LAWS = [
    {
        "output": "01_ley_federal_trabajo.txt",
        "source": "LEY Federal del Trabajo.txt",
        "type": "full",
    },
    {
        "output": "02_ley_seguro_social.txt",
        "source": "LEY del Seguro Social.txt",
        "type": "full",
    },
    {
        "output": "03_ley_trabajadores_estado.txt",
        "source": "Trabajadores al Servicio del Estado",
        "type": "full",
    },
    {
        "output": "04_ley_infonavit.txt",
        "source": "Instituto del Fondo Nacional de la Vivienda para los Trabajadores",
        "type": "full",
    },
    {
        "output": "05_art123_cpeum.txt",
        "source": "CPEUM",  # special handling
        "type": "cpeum_articles",
        "articles": (123, 123),  # single article but both apartados A and B
    },
]

# -- Cleaning patterns --

TRANSITORIOS_START = re.compile(
    r"^(\s*#+\s*)?(-{2,}\s*)?(TRANSITORIOS?|ART[IÍ]CULOS?\s+TRANSITORIOS?)\s*(-{2,})?\s*$",
    re.IGNORECASE,
)

DEROGATED_ARTICLE = re.compile(
    r"^####\s+Art[ií]culo\s+\d+[\w\s\-\.]*\s*[-\.]\s*\(?\s*[Ss]e\s+derog[aó]\s*\)?\s*\.?\s*$",
    re.IGNORECASE,
)

DEROGATED_STANDALONE = re.compile(
    r"^\s*\(?\s*[Ss]e\s+derog[aó]\s*\)?\s*\.?\s*$",
    re.IGNORECASE,
)

INLINE_REFORM = re.compile(
    r"\s*\(?\s*(Reformad[oa]|Adicionad[oa]|Derogad[oa]|Fe\s+de\s+erratas|"
    r"Nota\s+del\s+editor|Art[ií]culo\s+reformado|Fracci[oó]n\s+reformad[oa]|"
    r"P[aá]rrafo\s+reformado|P[aá]rrafo\s+adicionado|Inciso\s+reformado|"
    r"Fracci[oó]n\s+adicionad[oa]|Fracci[oó]n\s+derogad[oa]|"
    r"P[aá]rrafo\s+derogado|Art[ií]culo\s+adicionado|Art[ií]culo\s+derogado|"
    r"Secci[oó]n\s+adicionad[oa]|Cap[ií]tulo\s+adicionado|T[ií]tulo\s+adicionado|"
    r"Denominaci[oó]n\s+del|Numeral\s+reformado|Numeral\s+adicionado|"
    r"[UÚ]ltimo\s+p[aá]rrafo)[^)]*\)?\s*",
    re.IGNORECASE,
)

STANDALONE_REFORM = re.compile(
    r"^\s*(Reformad[oa]|Adicionad[oa]|Derogad[oa]|Fe\s+de\s+erratas|"
    r"Art[ií]culo\s+reformado|Fracci[oó]n\s+reformad[oa]|"
    r"P[aá]rrafo\s+(reformado|adicionado|derogado)|Inciso\s+reformado|"
    r"Fracci[oó]n\s+(adicionad[oa]|derogad[oa])|"
    r"Art[ií]culo\s+(adicionado|derogado|con\s+fracciones)|"
    r"Secci[oó]n\s+(adicionad[oa]|derogad[oa]|reformad[oa])|"
    r"Cap[ií]tulo\s+(adicionado|derogado|reformado)|"
    r"T[ií]tulo\s+(derogado|adicionado|reformado)|"
    r"Denominaci[oó]n\s+del|Cantidades\s+actualizada|[UÚu]ltima\s+reforma|"
    r"Numeral\s+(reformado|adicionado)|"
    r"Definici[oó]n\s+(reformad[oa]|adicionad[oa]))\s*(DOF|publicad[oa])?\s*",
    re.IGNORECASE,
)

DOF_LINE = re.compile(
    r"^\s*(DOF\s+\d|Diario\s+Oficial|El\s+Presidente\s+de\s+la\s+Rep[uú]blica|"
    r"ART[IÍ]CULO\s+(PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO)\.\-\s*Se\s+expide|"
    r"Publicad[oa]\s+en\s|Nueva\s+Ley\s+publicada|Ley\s+publicada|"
    r"C[oó]digo\s+publicado|Ley\s+abrogada|Que\s+en\s+virtud|"
    r"Disposici[oó]n\s+derogatoria|"
    r"ha\s+servido\s+dirigirme)",
    re.IGNORECASE,
)

PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$")

DEROGATED_HEADER = re.compile(
    r"^#+\s*(T[ií]tulo|Cap[ií]tulo|Secci[oó]n|Art[ií]culo)\s+(derogad[oa]|reformad[oa])\s+DOF",
    re.IGNORECASE,
)


def clean_law(text: str) -> str:
    """Apply standard cleaning to a law text."""
    lines = text.splitlines()
    cleaned = []
    derogated_count = 0
    skip_derogated = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # TRANSITORIOS - stop
        if TRANSITORIOS_START.search(stripped):
            print(f"  Cutting TRANSITORIOS at line {i+1}")
            break

        # DOF lines
        if DOF_LINE.match(stripped):
            continue

        # Derogated article headers
        if DEROGATED_ARTICLE.match(stripped):
            derogated_count += 1
            skip_derogated = True
            continue

        # Derogated headers
        if DEROGATED_HEADER.match(stripped):
            continue

        # Standalone derogation
        if DEROGATED_STANDALONE.match(stripped):
            skip_derogated = False
            continue

        # Standalone reform annotations
        if STANDALONE_REFORM.match(stripped):
            continue

        # Page numbers
        if PAGE_NUMBER.match(stripped):
            continue

        # Reset derogated block on new header
        if stripped.startswith("#") or stripped.startswith("SECCI"):
            skip_derogated = False

        if skip_derogated and not stripped:
            continue

        skip_derogated = False

        # Inline reform annotations
        line_cleaned = INLINE_REFORM.sub("", line).rstrip()
        cleaned.append(line_cleaned)

    if derogated_count > 0:
        print(f"  Removed {derogated_count} derogated articles")

    # Collapse blank lines
    result = []
    prev_blank = False
    for line in cleaned:
        if not line.strip():
            if prev_blank:
                continue
            prev_blank = True
        else:
            prev_blank = False
        result.append(line)

    return "\n".join(result).strip()


def extract_cpeum_articles(cpeum_text: str, start_art: int, end_art: int) -> str:
    """Extract a range of articles (inclusive) from the CPEUM text."""
    lines = cpeum_text.splitlines()
    header = "# CONSTITUCIÓN POLÍTICA DE LOS ESTADOS UNIDOS MEXICANOS\n"
    header += f"## Artículo {start_art} — Materia Laboral (Apartados A y B)\n\n"

    # Pattern to detect any article header (CPEUM uses "Art. N.-" format)
    art_header = re.compile(
        r"^(?:#+\s*)?Art(?:[ií]culo|\.)?\s+(\d+)\s*[\.\-]",
        re.IGNORECASE,
    )

    collecting = False
    article_lines = []

    for line in lines:
        stripped = line.strip()
        m = art_header.match(stripped)
        if m:
            art_num = int(m.group(1))
            if start_art <= art_num <= end_art:
                collecting = True
            elif art_num > end_art:
                break
            else:
                collecting = False

        if collecting:
            article_lines.append(line)

    if not article_lines:
        print(f"  WARNING: Articles {start_art}-{end_art} not found in CPEUM!")
        return ""

    # Clean inline reform annotations
    cleaned = []
    for line in article_lines:
        line_c = INLINE_REFORM.sub("", line).rstrip()
        if not STANDALONE_REFORM.match(line_c.strip()):
            cleaned.append(line_c)

    return header + "\n".join(cleaned).strip()


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def find_source_file(name: str) -> Path:
    """Find a source file by partial name match."""
    # Try exact match first
    exact = SOURCE_DIR / name
    if exact.exists():
        return exact
    # Fuzzy match
    name_lower = name.lower()
    for f in SOURCE_DIR.iterdir():
        if name_lower in f.name.lower():
            return f
    raise FileNotFoundError(f"Cannot find: {name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_chars = 0
    total_tokens = 0
    results = []

    print("=" * 70)
    print("Iurexia - Laboral Corpus Builder")
    print("=" * 70)

    for law in LAWS:
        out_name = law["output"]
        out_path = OUTPUT_DIR / out_name

        if law["type"] == "cpeum_articles":
            # Special: extract article range from CPEUM
            start_art, end_art = law["articles"]
            print(f"\nExtracting: Art. {start_art} CPEUM")
            cpeum_text = CPEUM_PATH.read_text(encoding="utf-8")
            raw_chars = len(cpeum_text)
            print(f"  CPEUM source: {raw_chars:,} chars")

            cleaned = extract_cpeum_articles(cpeum_text, start_art, end_art)
            clean_chars = len(cleaned)
            clean_tokens = estimate_tokens(cleaned)
            print(f"  Extracted: {clean_chars:,} chars (~{clean_tokens:,} tokens)")

            out_path.write_text(cleaned, encoding="utf-8")
            print(f"  Saved: {out_name}")

            total_chars += clean_chars
            total_tokens += clean_tokens
            results.append((out_name, clean_chars, clean_tokens, 0))
            continue

        # Standard law processing
        src_path = find_source_file(law["source"])
        print(f"\nProcessing: {src_path.name}")
        raw = src_path.read_text(encoding="utf-8")
        raw_chars = len(raw)
        raw_tokens = estimate_tokens(raw)
        print(f"  Raw: {raw_chars:,} chars (~{raw_tokens:,} tokens)")

        cleaned = clean_law(raw)
        clean_chars = len(cleaned)
        clean_tokens = estimate_tokens(cleaned)
        reduction = (1 - clean_chars / raw_chars) * 100 if raw_chars else 0

        print(f"  Clean: {clean_chars:,} chars (~{clean_tokens:,} tokens) [{reduction:.1f}% reduction]")

        out_path.write_text(cleaned, encoding="utf-8")
        print(f"  Saved: {out_name}")

        total_chars += clean_chars
        total_tokens += clean_tokens
        results.append((out_name, clean_chars, clean_tokens, reduction))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'File':<55} {'Chars':>10} {'Tokens':>10} {'Cut':>6}")
    print("-" * 82)
    for name, chars, tokens, cut in results:
        print(f"{name:<55} {chars:>10,} {tokens:>10,} {cut:>5.1f}%")
    print("-" * 82)
    print(f"{'TOTAL':<55} {total_chars:>10,} {total_tokens:>10,}")
    print()

    MAX_TOKENS = 450_000
    if total_tokens > MAX_TOKENS:
        print(f"WARNING: {total_tokens:,} tokens EXCEEDS limit of {MAX_TOKENS:,}")
        print(f"  Need to cut {total_tokens - MAX_TOKENS:,} more tokens")
    else:
        remaining = MAX_TOKENS - total_tokens
        print(f"WITHIN BUDGET: {total_tokens:,} / {MAX_TOKENS:,} tokens ({remaining:,} remaining)")


if __name__ == "__main__":
    main()
