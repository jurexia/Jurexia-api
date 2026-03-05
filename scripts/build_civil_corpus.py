"""
build_civil_corpus.py - Iurexia Civil Corpus Builder
======================================================
Reads CCF and CNPCF from LEYES_LIMPIAS_TXT, applies cleaning,
and outputs token-optimized TXT files for the Gemini context cache.

Strategy:
  - CNPCF: Full content, only clean DOF/transitorios/reform annotations
  - CCF: Aggressive trim — remove derogated articles, condense empty "Se deroga" runs

Target: <= 300,000 tokens total (~1,200,000 chars / 4 chars per token)
"""

import re
import sys
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# -- Source & Output --
SOURCE_DIR = Path(r"C:\Proyectos\LEYES_LIMPIAS_TXT\LEYES_FEDERALES")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "cache_corpus_civil"

LAWS = {
    "01_codigo_civil_federal.txt": {
        "source": "Código Civil Federal.txt",
        "aggressive_trim": True,
    },
    "02_codigo_nacional_procedimientos_civiles_familiares.txt": {
        "source": "CÓDIGO Nacional de Procedimientos Civiles y Familiares.txt",
        "aggressive_trim": False,
    },
}

# -- Patterns --

TRANSITORIOS_START = re.compile(
    r"^(\s*#+\s*)?(TRANSITORIOS?|ART[IÍ]CULOS?\s+TRANSITORIOS?)",
    re.IGNORECASE,
)

# Derogated article: line is just "#### Artículo N.- (Se deroga)." or similar
DEROGATED_ARTICLE = re.compile(
    r"^####\s+Art[ií]culo\s+\d+[\w\s\-\.]*\s*[-\.]\s*\(?\s*[Ss]e\s+derog[aó]\s*\)?\s*\.?\s*$",
    re.IGNORECASE,
)

# Standalone derogation line
DEROGATED_STANDALONE = re.compile(
    r"^\s*\(?\s*[Ss]e\s+derog[aó]\s*\)?\s*\.?\s*$",
    re.IGNORECASE,
)

# Inline reform annotations
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

# Standalone reform annotation lines
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

# DOF/publication date lines
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

# For CCF: detect runs of consecutive derogated articles
CCF_ARTICLE_HEADER = re.compile(
    r"^####\s+Art[ií]culo\s+(\d+)",
    re.IGNORECASE,
)


def clean_law(text: str, law_name: str, aggressive_trim: bool = False) -> str:
    """Apply cleaning rules to a law text."""
    lines = text.splitlines()
    cleaned = []
    skip_derogated = False
    derogated_count = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # 1. TRANSITORIOS - stop
        if TRANSITORIOS_START.search(stripped):
            print(f"  Cutting TRANSITORIOS at line {i+1}")
            break

        # 2. DOF lines
        if DOF_LINE.match(stripped):
            continue

        # 3. Derogated article headers (aggressive trim only)
        if aggressive_trim and DEROGATED_ARTICLE.match(stripped):
            derogated_count += 1
            skip_derogated = True
            continue

        # 4. Derogated headers
        if DEROGATED_HEADER.match(stripped):
            continue

        # 5. Standalone derogation
        if DEROGATED_STANDALONE.match(stripped):
            skip_derogated = False
            continue

        # 6. Standalone reform annotations
        if STANDALONE_REFORM.match(stripped):
            continue

        # 7. Page numbers
        if PAGE_NUMBER.match(stripped):
            continue

        # 8. Reset derogated block on new header/article
        if stripped.startswith("#") or stripped.startswith("SECCI"):
            if skip_derogated and derogated_count > 0:
                pass  # We already skipped; don't add summary for derogated
            skip_derogated = False

        if skip_derogated and not stripped:
            continue

        skip_derogated = False

        # 9. Inline reform annotations
        line_cleaned = INLINE_REFORM.sub("", line)
        line_cleaned = line_cleaned.rstrip()

        cleaned.append(line_cleaned)

    if aggressive_trim and derogated_count > 0:
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


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_chars = 0
    total_tokens = 0
    results = []

    print("=" * 70)
    print("Iurexia - Civil Corpus Builder")
    print("=" * 70)

    for out_name, config in LAWS.items():
        src_name = config["source"]
        aggressive = config["aggressive_trim"]
        src_path = SOURCE_DIR / src_name
        out_path = OUTPUT_DIR / out_name

        if not src_path.exists():
            print(f"\n  NOT FOUND: {src_path}")
            # Try alt encoding
            for f in SOURCE_DIR.iterdir():
                if f.name.lower().replace("ó","o").replace("í","i") == src_name.lower().replace("ó","o").replace("í","i"):
                    src_path = f
                    break
            if not src_path.exists():
                print("  Available files:")
                for f in sorted(SOURCE_DIR.glob("*.txt")):
                    print(f"    {f.name}")
                sys.exit(1)

        print(f"\nProcessing: {src_name}")
        raw = src_path.read_text(encoding="utf-8")
        raw_chars = len(raw)
        raw_tokens = estimate_tokens(raw)
        print(f"  Raw: {raw_chars:,} chars (~{raw_tokens:,} tokens)")
        if aggressive:
            print(f"  Mode: AGGRESSIVE TRIM (derogated articles removed)")
        else:
            print(f"  Mode: STANDARD CLEAN (full content preserved)")

        cleaned = clean_law(raw, src_name, aggressive_trim=aggressive)
        clean_chars = len(cleaned)
        clean_tokens = estimate_tokens(cleaned)
        reduction = (1 - clean_chars / raw_chars) * 100 if raw_chars else 0

        print(f"  Clean: {clean_chars:,} chars (~{clean_tokens:,} tokens) [{reduction:.1f}% reduction]")

        out_path.write_text(cleaned, encoding="utf-8")
        print(f"  Saved: {out_path.name}")

        total_chars += clean_chars
        total_tokens += clean_tokens
        results.append((out_name, clean_chars, clean_tokens, reduction))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'File':<60} {'Chars':>10} {'Tokens':>10} {'Cut':>6}")
    print("-" * 86)
    for name, chars, tokens, cut in results:
        print(f"{name:<60} {chars:>10,} {tokens:>10,} {cut:>5.1f}%")
    print("-" * 86)
    print(f"{'TOTAL':<60} {total_chars:>10,} {total_tokens:>10,}")
    print()

    MAX_TOKENS = 450_000
    if total_tokens > MAX_TOKENS:
        print(f"WARNING: {total_tokens:,} tokens EXCEEDS limit of {MAX_TOKENS:,}")
        print(f"  Need to cut {total_tokens - MAX_TOKENS:,} more tokens")
        print(f"  Consider trimming CCF further (remove Libro 3 Sucesiones?)")
    else:
        remaining = MAX_TOKENS - total_tokens
        print(f"WITHIN BUDGET: {total_tokens:,} / {MAX_TOKENS:,} tokens ({remaining:,} remaining)")


if __name__ == "__main__":
    main()
