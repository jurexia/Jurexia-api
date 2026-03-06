"""
build_agrario_corpus.py - Iurexia Agrario Corpus Builder
=========================================================
Reads Ley Agraria, Ley Orgánica de los Tribunales Agrarios,
and extracts Art. 27 from the CPEUM.

Sources:
  - C:\Proyectos\LEYES_LIMPIAS_TXT\LEYES_FEDERALES\
  - C:\tmp\CPEUM_full_utf8.txt (for Art. 27)

Target: <= 500,000 tokens
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
CPEUM_PATH = Path(r"C:\tmp\CPEUM_full_utf8.txt")
API_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = API_DIR / "cache_corpus_agrario"

# -- Law definitions --
LAWS = [
    {
        "output": "01_ley_agraria.txt",
        "source": "Ley Agraria.txt",
        "type": "full",
    },
    {
        "output": "02_ley_organica_tribunales_agrarios.txt",
        "source": "LEY Orgánica de los Tribunales Agrarios.txt",
        "type": "full",
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

        if TRANSITORIOS_START.search(stripped):
            print(f"  Cutting TRANSITORIOS at line {i+1}")
            break

        if DOF_LINE.match(stripped):
            continue

        if DEROGATED_ARTICLE.match(stripped):
            derogated_count += 1
            skip_derogated = True
            continue

        if DEROGATED_HEADER.match(stripped):
            continue

        if DEROGATED_STANDALONE.match(stripped):
            skip_derogated = False
            continue

        if STANDALONE_REFORM.match(stripped):
            continue

        if PAGE_NUMBER.match(stripped):
            continue

        if stripped.startswith("#") or stripped.startswith("SECCI"):
            skip_derogated = False

        if skip_derogated and not stripped:
            continue

        skip_derogated = False

        line_cleaned = INLINE_REFORM.sub("", line).rstrip()
        cleaned.append(line_cleaned)

    if derogated_count > 0:
        print(f"  Removed {derogated_count} derogated articles")

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


def extract_cpeum_article(text: str, article_num: int) -> str:
    """Extract a specific article from the CPEUM."""
    lines = text.splitlines()
    # Match patterns like "#### Artículo 27" or "Art. 27.-"
    art_pattern = re.compile(
        rf"^(####\s+)?Art[ií]culo\s+{article_num}\b|^Art\.\s*{article_num}\.\-",
        re.IGNORECASE
    )
    next_art_pattern = re.compile(
        rf"^(####\s+)?Art[ií]culo\s+{article_num + 1}\b|^Art\.\s*{article_num + 1}\.\-",
        re.IGNORECASE
    )

    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if start_idx is None and art_pattern.match(line.strip()):
            start_idx = i
        elif start_idx is not None and next_art_pattern.match(line.strip()):
            end_idx = i
            break

    if start_idx is not None:
        if end_idx is None:
            end_idx = min(start_idx + 500, len(lines))
        extracted = "\n".join(lines[start_idx:end_idx]).strip()
        print(f"  Extracted Art. {article_num}: {len(extracted):,} chars (~{len(extracted)//4:,} tokens)")
        return extracted
    else:
        print(f"  WARNING: Art. {article_num} NOT FOUND in CPEUM")
        return ""


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def find_source_file(name: str) -> Path:
    exact = SOURCE_DIR / name
    if exact.exists():
        return exact
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
    print("Iurexia - Agrario Corpus Builder")
    print("=" * 70)

    # Process laws
    for law in LAWS:
        out_name = law["output"]
        out_path = OUTPUT_DIR / out_name

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

    # Extract Art. 27 from CPEUM
    print(f"\nProcessing: CPEUM Art. 27")
    cpeum_text = CPEUM_PATH.read_text(encoding="utf-8")
    art27 = extract_cpeum_article(cpeum_text, 27)
    if art27:
        art27_cleaned = clean_law(art27)
        out_name = "03_cpeum_art27.txt"
        out_path = OUTPUT_DIR / out_name

        header = "# CONSTITUCIÓN POLÍTICA DE LOS ESTADOS UNIDOS MEXICANOS\n"
        header += "## Artículo 27 — Fundamento del régimen agrario\n\n"
        final = header + art27_cleaned

        clean_chars = len(final)
        clean_tokens = estimate_tokens(final)
        out_path.write_text(final, encoding="utf-8")
        print(f"  Saved: {out_name} ({clean_chars:,} chars, ~{clean_tokens:,} tokens)")

        total_chars += clean_chars
        total_tokens += clean_tokens
        results.append((out_name, clean_chars, clean_tokens, 0.0))

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

    MAX_TOKENS = 500_000
    if total_tokens > MAX_TOKENS:
        print(f"WARNING: {total_tokens:,} tokens EXCEEDS limit of {MAX_TOKENS:,}")
    else:
        remaining = MAX_TOKENS - total_tokens
        print(f"WITHIN BUDGET: {total_tokens:,} / {MAX_TOKENS:,} tokens ({remaining:,} remaining)")


if __name__ == "__main__":
    main()
