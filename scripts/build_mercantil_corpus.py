"""
build_mercantil_corpus.py - Iurexia Mercantil Corpus Builder v2
================================================================
Reads 5 federal mercantile laws from LEYES_LIMPIAS_TXT, applies aggressive
cleaning, and outputs token-optimized TXT files for the Gemini context cache.

Cleaning rules:
  1. Remove TRANSITORIOS sections (all text after TRANSITORIOS header)
  2. Remove derogated articles: "(Se deroga)" or only contain derogation text
  3. Remove DOF reform annotations (inline and standalone)
  4. Remove page numbers, headers/footers
  5. Collapse excessive blank lines (max 1)
  6. Keep Libro / Titulo / Capitulo / Seccion structure headers
  7. Keep article text with full content
  8. LISF: Only keep Titulos 1,2,6,7 (definitions, operations, procedures, prohibitions)

Target: <= 300,000 tokens total (~1,200,000 chars / 4 chars per token)
"""

import re
import sys
from pathlib import Path

# Fix encoding on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# -- Source & Output --
SOURCE_DIR = Path(r"C:\Proyectos\LEYES_LIMPIAS_TXT\LEYES_FEDERALES")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "cache_corpus_mercantil"

# Map: output filename -> source filename
LAWS = {
    "01_codigo_comercio.txt": "CODIGO de Comercio.txt",
    "02_ley_titulos_credito.txt": "LEY GENERAL DE TITULOS Y OPERACIONES DE CREDITO.txt",
    "03_ley_sociedades_mercantiles.txt": "LEY General de Sociedades Mercantiles.txt",
    "04_ley_contrato_seguro.txt": "LEY Sobre el Contrato de Seguro.txt",
    "05_ley_instituciones_seguros_fianzas.txt": "LEY de Instituciones de Seguros y de Fianzas.txt",
}

# -- LISF: Titles to KEEP (all others are cut) --
# 1 = Disposiciones Preliminares (definitions)
# 2 = De las Instituciones (operations & ramos)
# 6 = De los Procedimientos (claims, cobro de fianza)
# 7 = De las Prohibiciones
LISF_KEEP_TITLES = {"PRIMERO", "SEGUNDO", "SEXTO", "SEPTIMO", "SÉPTIMO"}

# -- Patterns to STRIP --

TRANSITORIOS_START = re.compile(
    r"^(\s*#+\s*)?(TRANSITORIOS?|ART[II]CULOS?\s+TRANSITORIOS?|---\s*TRANSITORIOS?\s*---)"
    r"|^(\s*#+\s*)?ARTICULOS\s+TRANSITORIOS\s+DE\s+DECRETOS",
    re.IGNORECASE,
)

DEROGATED_FULL = re.compile(
    r"^####\s+Art[ii]culo\s+\d+[\w\s]*[-.]?\s*\(?\s*Se\s+derog[ao]\s*\)?\s*\.?\s*$",
    re.IGNORECASE,
)

DEROGATED_STANDALONE = re.compile(
    r"^\s*\(?\s*Se\s+derog[ao]\s*\)?\s*\.?\s*$",
    re.IGNORECASE,
)

INLINE_REFORM = re.compile(
    r"\s*\(?\s*(Reformad[oa]|Adicionad[oa]|Derogad[oa]|Fe\s+de\s+erratas|"
    r"Nota\s+del\s+editor|Art[ii]culo\s+reformado|Fracci[oó]n\s+reformad[oa]|"
    r"P[aá]rrafo\s+reformado|P[aá]rrafo\s+adicionado|Inciso\s+reformado|"
    r"Fracci[oó]n\s+adicionad[oa]|Fracci[oó]n\s+derogad[oa]|"
    r"P[aá]rrafo\s+derogado|Art[ii]culo\s+adicionado|Art[ii]culo\s+derogado|"
    r"Secci[oó]n\s+adicionad[oa]|Cap[ii]tulo\s+adicionado|T[ii]tulo\s+adicionado|"
    r"Denominaci[oó]n\s+del|Numeral\s+reformado|Numeral\s+adicionado|"
    r"[UÚ]ltimo\s+p[aá]rrafo)[^)]*\)?\s*",
    re.IGNORECASE,
)

STANDALONE_REFORM = re.compile(
    r"^\s*(Reformad[oa]|Adicionad[oa]|Derogad[oa]|Fe\s+de\s+erratas|"
    r"Art[ií]culo\s+reformado|Fracci[oó]n\s+reformad[oa]|"
    r"P[aá]rrafo\s+reformado|P[aá]rrafo\s+adicionado|Inciso\s+reformado|"
    r"P[aá]rrafo\s+(Quinto|Cuarto|Tercero|Segundo|Sexto|S[eé]ptimo|Octavo|Noveno|D[eé]cimo)\.?-?\s*Se\s+derog|"
    r"Fracci[oó]n\s+adicionad[oa]|Fracci[oó]n\s+derogad[oa]|"
    r"P[aá]rrafo\s+derogado|Art[ií]culo\s+adicionado|Art[ií]culo\s+derogado|"
    r"Art[ií]culo\s+con\s+fracciones|Secci[oó]n\s+adicionad[oa]|Cap[ií]tulo\s+adicionado|"
    r"T[ií]tulo\s+(derogado|adicionado|reformado)|Cap[ií]tulo\s+(derogado|reformado)|"
    r"Secci[oó]n\s+(derogad[oa]|reformad[oa])|"
    r"Denominaci[oó]n\s+del|Cantidades\s+actualizada|[UÚ]ltima\s+reforma|"
    r"Ultima\s+reforma|Numeral\s+reformado|Numeral\s+adicionado|"
    r"Definici[oó]n\s+(reformad[oa]|adicionad[oa])|"
    r"SECCI[OÓ]N\s+[UÚ]NICA|Secci[oó]n\s+[uú]nica)\s*(DOF|publicad[oa])?\s*",
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

# Pattern to detect TITULO headers in LISF
TITULO_HEADER = re.compile(
    r"^##\s+T[IÍ]TULO\s+(\S+)",
    re.IGNORECASE,
)


def extract_lisf_titles(text: str) -> str:
    """For the LISF, extract only the titles we want to keep."""
    lines = text.splitlines()
    result = []
    current_title = None
    keeping = False
    
    # Add the law name header
    for i, line in enumerate(lines):
        stripped = line.strip()
        m = TITULO_HEADER.match(stripped)
        if m:
            title_num = m.group(1).upper()
            # Normalize accented characters
            title_num = title_num.replace("É", "E").replace("Í", "I")
            if title_num in LISF_KEEP_TITLES:
                keeping = True
                current_title = title_num
                print(f"  [KEEP] TITULO {title_num}")
            else:
                if keeping:
                    # We were keeping, now hit a title we don't want
                    pass
                keeping = False
                current_title = title_num
                print(f"  [CUT]  TITULO {title_num}")
        
        if keeping:
            result.append(line)
        elif i < 10:
            # Always keep the first few lines (law name)
            result.append(line)
    
    return "\n".join(result)


def clean_law(text: str, law_name: str, is_lisf: bool = False) -> str:
    """Apply all cleaning rules to a law text."""
    
    # For LISF, first extract only the titles we want
    if is_lisf:
        print("  Applying LISF title filter...")
        text = extract_lisf_titles(text)
    
    lines = text.splitlines()
    cleaned = []
    skip_derogated_block = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # 1. TRANSITORIOS - stop here
        if TRANSITORIOS_START.search(stripped):
            print(f"  Cutting TRANSITORIOS at line {i+1}")
            break
        
        # 2. Skip standalone DOF annotations
        if DOF_LINE.match(stripped):
            continue
        
        # 3. Skip derogated article headers
        if DEROGATED_FULL.match(stripped):
            skip_derogated_block = True
            continue
        
        # 4. Skip derogated headers
        if DEROGATED_HEADER.match(stripped):
            continue
        
        # 5. Skip standalone derogation lines
        if DEROGATED_STANDALONE.match(stripped):
            skip_derogated_block = False
            continue
        
        # 6. Skip standalone reform annotation lines
        if STANDALONE_REFORM.match(stripped):
            continue
        
        # 7. Skip page numbers
        if PAGE_NUMBER.match(stripped):
            continue
        
        # 8. A new article or structural header resets derogated block skip
        if stripped.startswith("#") or stripped.startswith("SECCI") or stripped.startswith("Secci"):
            skip_derogated_block = False
        
        if skip_derogated_block and not stripped:
            continue
        
        skip_derogated_block = False
        
        # 9. Clean inline reform annotations
        line_cleaned = INLINE_REFORM.sub("", line)
        
        # 10. Remove trailing whitespace
        line_cleaned = line_cleaned.rstrip()
        
        cleaned.append(line_cleaned)
    
    # Collapse excessive blank lines (max 1 consecutive)
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
    
    text_out = "\n".join(result).strip()
    return text_out


def estimate_tokens(text: str) -> int:
    """Rough token estimate: chars / 4."""
    return len(text) // 4


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total_chars = 0
    total_tokens = 0
    results = []
    
    print("=" * 70)
    print("Iurexia - Mercantil Corpus Builder v2")
    print("=" * 70)
    
    for out_name, src_name in LAWS.items():
        src_path = SOURCE_DIR / src_name
        out_path = OUTPUT_DIR / out_name
        
        if not src_path.exists():
            # Try with accent variations
            alt_name = src_name.replace("CODIGO", "CÓDIGO")
            alt_path = SOURCE_DIR / alt_name
            if alt_path.exists():
                src_path = alt_path
            else:
                print(f"\n  NOT FOUND: {src_path}")
                print(f"  Also tried: {alt_path}")
                # List what's available
                print("  Available files:")
                for f in sorted(SOURCE_DIR.glob("*.txt")):
                    print(f"    {f.name}")
                sys.exit(1)
        
        is_lisf = "instituciones" in out_name.lower()
        
        print(f"\nProcessing: {src_name}")
        raw = src_path.read_text(encoding="utf-8")
        raw_chars = len(raw)
        raw_tokens = estimate_tokens(raw)
        print(f"  Raw: {raw_chars:,} chars (~{raw_tokens:,} tokens)")
        
        cleaned = clean_law(raw, src_name, is_lisf=is_lisf)
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
    print(f"{'File':<50} {'Chars':>10} {'Tokens':>10} {'Cut':>6}")
    print("-" * 76)
    for name, chars, tokens, cut in results:
        print(f"{name:<50} {chars:>10,} {tokens:>10,} {cut:>5.1f}%")
    print("-" * 76)
    print(f"{'TOTAL':<50} {total_chars:>10,} {total_tokens:>10,}")
    print()
    
    MAX_TOKENS = 300_000
    if total_tokens > MAX_TOKENS:
        print(f"WARNING: {total_tokens:,} tokens EXCEEDS limit of {MAX_TOKENS:,}")
        print(f"  Need to cut {total_tokens - MAX_TOKENS:,} more tokens")
    else:
        remaining = MAX_TOKENS - total_tokens
        print(f"WITHIN BUDGET: {total_tokens:,} / {MAX_TOKENS:,} tokens ({remaining:,} remaining)")


if __name__ == "__main__":
    main()
