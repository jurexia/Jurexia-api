"""
build_fiscal_corpus.py - Iurexia Fiscal Corpus Builder
=======================================================
Reads CFF, LISR, LIVA, Ley Federal de Procedimiento Contencioso Administrativo.
Applies cleaning and outputs token-optimized TXT files.

Sources:
  - C:\Proyectos\LEYES_LIMPIAS_TXT\LEYES_FEDERALES\

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
API_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = API_DIR / "cache_corpus_fiscal"

# -- Law definitions --
LAWS = [
    {
        "output": "01_codigo_fiscal_federacion.txt",
        "source": "Código Fiscal de la Federación.txt",
        "type": "full",
    },
    {
        "output": "02_ley_impuesto_renta.txt",
        "source": "LEY del Impuesto sobre la Renta.txt",
        "type": "lisr_trimmed",  # remove Título VI (REFIPRES) + VII (estímulos)
    },
    {
        "output": "03_ley_iva.txt",
        "source": "LEY DEL IMPUESTO AL VALOR AGREGADO.txt",
        "type": "full",
    },
    {
        "output": "04_ley_procedimiento_contencioso.txt",
        "source": "LEY Federal de Procedimiento Contencioso Administrativo.txt",
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


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def trim_lisr(text: str) -> str:
    """Remove Título VI (REFIPRES) and Título VII (Estímulos Fiscales) from LISR.
    These are ultra-specialized sections that consume ~56K tokens."""
    lines = text.splitlines()
    titulo_vi_re = re.compile(r"^##\s*T[IÍ]TULO\s+VI\b", re.IGNORECASE)
    result = []
    skipping = False
    skipped_chars = 0

    for line in lines:
        stripped = line.strip()
        if titulo_vi_re.match(stripped):
            skipping = True
            # Add a marker so users know content was trimmed
            result.append("")
            result.append("[TÍTULOS VI y VII OMITIDOS — REFIPRES y Estímulos Fiscales]")
            result.append("")
            print("  Trimming: Título VI (REFIPRES) + Título VII (Estímulos Fiscales)")

        if skipping:
            skipped_chars += len(line)
            continue

        result.append(line)

    print(f"  Trimmed: {skipped_chars:,} chars (~{skipped_chars//4:,} tokens)")
    return "\n".join(result)


def find_source_file(name: str) -> Path:
    """Find a source file by partial name match."""
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
    print("Iurexia - Fiscal Corpus Builder")
    print("=" * 70)

    for law in LAWS:
        out_name = law["output"]
        out_path = OUTPUT_DIR / out_name

        src_path = find_source_file(law["source"])
        print(f"\nProcessing: {src_path.name}")
        raw = src_path.read_text(encoding="utf-8")
        raw_chars = len(raw)
        raw_tokens = estimate_tokens(raw)
        print(f"  Raw: {raw_chars:,} chars (~{raw_tokens:,} tokens)")

        # Apply LISR-specific trimming before standard cleaning
        if law["type"] == "lisr_trimmed":
            raw = trim_lisr(raw)

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

    MAX_TOKENS = 500_000
    if total_tokens > MAX_TOKENS:
        print(f"WARNING: {total_tokens:,} tokens EXCEEDS limit of {MAX_TOKENS:,}")
        print(f"  Need to cut {total_tokens - MAX_TOKENS:,} more tokens")
    else:
        remaining = MAX_TOKENS - total_tokens
        print(f"WITHIN BUDGET: {total_tokens:,} / {MAX_TOKENS:,} tokens ({remaining:,} remaining)")


if __name__ == "__main__":
    main()
