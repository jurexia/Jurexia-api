"""
Iurexia Fine-Tuning Data Extractor — Redactor Sentencia v2
============================================================
Extracts text from curated sentencia PDFs and redacción DOC/DOCX files,
then generates OpenAI fine-tuning JSONL.

Training Strategy:
  - 12 redacción examples → teach WRITING STYLE (high-level judicial prose)
  - 25 structure examples → teach SENTENCE STRUCTURE (by resolution type)

Output: scripts/ft_data/training_data.jsonl
"""

import json
import os
import sys
import re
from pathlib import Path

# ── Extraction libraries ──
try:
    import pdfplumber
except ImportError:
    print("ERROR: pip install pdfplumber")
    sys.exit(1)

try:
    import docx
except ImportError:
    print("ERROR: pip install python-docx")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# CURATED SELECTION — Best examples per type
# ═══════════════════════════════════════════════════════════════════════════════

REDACCION_DIR = r"C:\Users\jdmju\Downloads\Manual\Redacción de alto nivel"
SENTENCIAS_BASE = r"C:\Proyectos\EJEMPLOS DE SENTENCIAS INGESTAR"

# All 12 redacción examples (keep ALL)
# These will be auto-discovered from the directory

# Curated structure examples — selected for diversity of sentido/tema
CURATED_STRUCTURE = {
    "amparo_directo": [
        # Concede
        "AMPARA Y PROTEGE.pdf",
        "Ampara al quejoso principal, la sentencia reclamada es incongruente (en su vertiente interna y externa.pdf",
        "concede el amparo.pdf",
        # Niega
        "SE NIEGA EL AMPARO A LA PARTE QUEJOSA (ACTOR), PUES NO DESESTIMA LAS CONSIDERACIONES TORALES QUE SUSTENTAN EL FALLO –JUICIO ORDINARIO CIVIL SOBRE DIVORCIO Y LIQUIDACIÓN DE SOCIEDAD CONYUGAL-.pdf",
        "Juicio ordinario civil. Prescripción positiva y reivindicación. Se niega el amparo, ante la ineficacia de los conceptos de violación formulados.pdf",
        # Sobresee
        "SE SOBRESEE EN EL PRESENTE JUICIO.pdf",
        "SOBRESEIMIENTO DEL JUICIO DE AMPARO ANTE LA FALTA DE PUBLICACIÓN DE EDICTOS.pdf",
        # Perspectiva de género
        "Divorcio incausado. Pensión compensatoria. Son infundados en un aspecto y por otra fundados los conceptos de violación, en relación al amparo adhesivo son inoperantes los argumentos formulados.pdf",
    ],
    "amparo_revision": [
        # Confirma concesión
        "Se confirma la sentencia recurrida y se concede el amparo solicitado, porque contrario a lo indicado.pdf",
        "En la materia de la revisión, se confirma la concesión. El artículo 152, fracción I, de la Ley de Hacienda del Estado (en la porción reclamada), viola los principios de equidad y proporcionalidad tributaria, por dos ra.pdf",
        # Confirma negativa
        "SE CONFIRMA LA NEGATIVA DEL AMPARO, PUES EL INMUEBLE QUE PRETENDE EL REVISIONISTA SEA PARTE DE LA SOCIEDAD CONYUGAL FUE ADQUIRIDO POR LA TERCERA INTERESADA (ACTORA) DURANTE LA VIGENCIA DEL CONCUBINATO, AL PRONUNCIARSE SOBRE EL QUANTUM DE LA PENSIÓN COMPENSA.pdf",
        # Revoca
        "Revoca y concede. No se actualiza la causa de improcedencia relativa al consentimiento tácito, por lo que se levanta el sobreseimiento y se reasume jurisdicción. Los conceptos de violación son en una parte ineficaces y.pdf",
        "Reasunción de jurisdicción con motivo de la revocación de la sentencia recurrida debido a que el juez.pdf",
        # Constitucionalidad
        "SE REVOCA Y SE CONCEDE EL AMPARO. LOS ARTÍCULOS 83 BIS-8 AL 83 BIS-13 DE LA LEY DE HACIENDA DEL ESTADO DE QUERÉTARO INTEGRAN UN SISTEMA NORMATIVO QUE REGULA EL IMPUESTO POR LA EMISIÓN DE GASES A LA ATMÓSFERA. LA CIRCUN.pdf",
        # Sobreseimiento
        "SE CONFIRMA EL SOBRESEIMIENTO DECRETADO. SON INFUNDADOS LOS AGRAVIOS, PUES DE LAS PROBANZAS QUE EXHIBIÓ LA PARTE QUEJOSA NO SE ADVIERTE CUÁNDO REALIZÓ EL PAGO DE DERECHOS DE INSCRIPCIÓN QUE COMBATE, PUES SE OBSERVA QUE.pdf",
    ],
    "queja": [
        # Fundada
        "CONVIVENCIAS DE INFANTES CON SU PROGENITOR EN VACACIONES Y DÍAS ESPECIALES. IMPROCEDENCIA DEL AMPARO POR NO AGOTARSE EL PRINCIPIO DE DEFINITIVIDAD. AGRAVIOS FUNDADOS.pdf",
        "Desechamiento de demanda de amparo, con motivo de que en contra del acto reclamado procedía recurso de reposición. Queja fundada. Imposible reparación.pdf",
        # Infundada
        "INFUNDADA.QUEJA VS AUTO QUE NEGÓ LA SUSPENSIÓN SOLICITADA POR EL QUEJOSO PARA QUE SE SUSPENDIERAN CONVIVENCIAS.pdf",
        "SE DECLARA INFUNDADO EL RECURSO DE QUEJA, PUES EXISTE CRITERIO REITERADO DE LA SUPREMA CORTE DE JUSTICIA DE LA NACIÓN QUE LA FALTA DE INTERÉS JURÍDICO NO CONSTITUYE UNA CAUSA MANIFIESTA E INDUDABLE DE IMPROCEDENCIA PAR.pdf",
        # Sin materia
        "QUEDA SIN MATERIA EL RECURSO.pdf",
    ],
    "revision_fiscal": [
        # Procedente/fundada
        "Recurso de revisión fiscal es fundado; revoca sentencia.pdf",
        "Procedimiento de presunción de inexistencia de operaciones. si la autoridad fiscal lo inicia porque detectó que un contribuyente dio efectos fiscales a comprobantes emitidos por otro cuyos datos están inscritos en el l.pdf",
        # Improcedente
        "Revisión fiscal improcedente. No se surte alguna de las hipótesis que prevé el artículo 63 de la Ley Federal de Procedimiento Contencioso Administrativo. El acto impugnado en el juicio de nulidad consistió en la resolu.pdf",
        "SE DESECHA REVISIÓN FISCAL.pdf",
        # Inoperante
        "DEVOLUCIÓN DE IMPUESTO AL VALOR AGREGADO.AGRAVIOS INOPERANTES E INEFICACES.pdf",
    ],
}

# ═══════════════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_pdf(path: str, max_chars: int = 80_000) -> str:
    """Extract text from a PDF file using pdfplumber."""
    text_parts = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                if sum(len(p) for p in text_parts) > max_chars:
                    break
    except Exception as e:
        print(f"  ⚠️ Error extracting {Path(path).name}: {e}")
        return ""
    return "\n".join(text_parts)[:max_chars]


def extract_docx(path: str, max_chars: int = 80_000) -> str:
    """Extract text from a DOCX file."""
    try:
        doc = docx.Document(path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return text[:max_chars]
    except Exception as e:
        print(f"  ⚠️ Error extracting {Path(path).name}: {e}")
        return ""


def extract_doc(path: str, max_chars: int = 80_000) -> str:
    """Extract text from old .doc files via textract or fallback."""
    try:
        # Try converting with python-docx (sometimes works for .doc)
        return extract_docx(path, max_chars)
    except:
        print(f"  ⚠️ Cannot extract .doc: {Path(path).name} (try converting to .docx)")
        return ""


def extract_text(path: str) -> str:
    """Extract text from any supported file."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return extract_pdf(path)
    elif ext == ".docx":
        return extract_docx(path)
    elif ext == ".doc":
        return extract_doc(path)
    elif ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()[:80_000]
    else:
        print(f"  ⚠️ Unsupported format: {ext}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_estudio_de_fondo(text: str) -> str:
    """
    Try to extract the 'estudio de fondo' section from a sentencia.
    Falls back to the full text if section markers aren't found.
    """
    # Common markers for estudio de fondo sections
    start_markers = [
        r"(?i)ESTUDIO\s+DE\s+FONDO",
        r"(?i)CONSIDERANDO\s*:?\s*\n",
        r"(?i)(?:QUINTO|SEXTO|SÉPTIMO|CUARTO)\s*[.\-–]\s*(?:Estudio|Análisis|Fondo)",
        r"(?i)análisis\s+de\s+(?:los\s+)?(?:conceptos?\s+de\s+violación|agravios)",
    ]
    end_markers = [
        r"(?i)(?:POR\s+LO\s+(?:EXPUESTO|ANTES\s+EXPUESTO|ANTERIORMENTE))",
        r"(?i)R\s*E\s*S\s*U\s*E\s*L\s*V\s*E",
        r"(?i)RESOLUTIVOS?\s*:?\s*\n",
        r"(?i)(?:PRIMERO|ÚNICO)\s*[.\-–]\s*(?:Se\s+(?:concede|niega|sobresee|ampara))",
    ]

    best_start = None
    for marker in start_markers:
        match = re.search(marker, text)
        if match:
            if best_start is None or match.start() < best_start:
                best_start = match.start()
            break

    best_end = None
    if best_start is not None:
        for marker in end_markers:
            for match in re.finditer(marker, text[best_start:]):
                pos = best_start + match.start()
                if pos > best_start + 500:  # At least 500 chars of content
                    best_end = pos
                    break
            if best_end:
                break

    if best_start is not None:
        section = text[best_start : best_end] if best_end else text[best_start:]
        if len(section) > 500:
            return section.strip()

    # Fallback: return middle 60% of the text (skip headers/footers)
    total = len(text)
    return text[int(total * 0.2) : int(total * 0.8)].strip()


def classify_sentido(text: str, tipo: str) -> str:
    """Classify the sentido (outcome) of a sentencia from its text."""
    text_lower = text.lower()

    if tipo == "amparo_directo":
        if "se concede" in text_lower or "ampara y protege" in text_lower:
            return "conceder"
        elif "se niega" in text_lower or "no ampara" in text_lower:
            return "negar"
        elif "se sobresee" in text_lower or "sobreseimiento" in text_lower:
            return "sobreseer"
        return "negar"  # default

    elif tipo == "amparo_revision":
        if "se revoca" in text_lower and "se concede" in text_lower:
            return "revocar_y_conceder"
        elif "se confirma" in text_lower and "se concede" in text_lower:
            return "confirmar_concesion"
        elif "se confirma" in text_lower and ("se niega" in text_lower or "negativa" in text_lower):
            return "confirmar_negativa"
        elif "se revoca" in text_lower:
            return "revocar"
        elif "se confirma" in text_lower:
            return "confirmar"
        elif "sin materia" in text_lower:
            return "sin_materia"
        return "confirmar"

    elif tipo == "queja":
        if "fundad" in text_lower and "infundad" not in text_lower:
            return "fundada"
        elif "infundad" in text_lower:
            return "infundada"
        elif "sin materia" in text_lower:
            return "sin_materia"
        return "infundada"

    elif tipo == "revision_fiscal":
        if "fundad" in text_lower and "infundad" not in text_lower:
            return "fundado"
        elif "improcedente" in text_lower or "se desecha" in text_lower:
            return "improcedente"
        elif "infundad" in text_lower or "inoperant" in text_lower:
            return "infundado"
        return "infundado"

    return "sin_clasificar"


# ═══════════════════════════════════════════════════════════════════════════════
# JSONL GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_REDACCION = (
    "Eres un redactor judicial de élite de un Tribunal Colegiado de Circuito mexicano. "
    "Tu función es redactar estudios de fondo de sentencias con precisión técnica, "
    "prosa jurídica de alto nivel, estructura lógica impecable, y fundamentación rigurosa. "
    "Utilizas lenguaje judicial formal, citas artículos con su número exacto y ley de origen, "
    "y referencias a tesis y jurisprudencias aplicables cuando están disponibles. "
    "Adaptas la estructura de la sentencia al tipo de resolución: amparo directo, "
    "amparo en revisión, recurso de queja, o revisión fiscal."
)

TIPO_LABELS = {
    "amparo_directo": "Amparo Directo",
    "amparo_revision": "Amparo en Revisión",
    "queja": "Recurso de Queja",
    "revision_fiscal": "Revisión Fiscal",
}


def build_training_example_structure(text: str, tipo: str, filename: str) -> dict:
    """Build a fine-tuning example for sentence STRUCTURE learning."""
    estudio = extract_estudio_de_fondo(text)
    sentido = classify_sentido(text, tipo)

    # Truncate if too long (OpenAI recommends <16K tokens per example for mini)
    max_assistant = 12_000  # chars ≈ 3K tokens
    if len(estudio) > max_assistant:
        estudio = estudio[:max_assistant] + "\n[...texto truncado por límite de entrenamiento...]"

    user_prompt = (
        f"TIPO DE RESOLUCIÓN: {TIPO_LABELS.get(tipo, tipo)}\n"
        f"SENTIDO PROPUESTO: {sentido}\n"
        f"CASO: {filename[:200]}\n\n"
        f"Redacta el estudio de fondo para esta resolución."
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_REDACCION},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": estudio},
        ]
    }


def build_training_example_redaccion(text: str, filename: str) -> dict:
    """Build a fine-tuning example for WRITING STYLE learning."""
    # Use the full text (or estudio de fondo) as the model output
    estudio = extract_estudio_de_fondo(text)
    max_assistant = 15_000
    if len(estudio) > max_assistant:
        estudio = estudio[:max_assistant] + "\n[...texto truncado...]"

    user_prompt = (
        f"EJEMPLO DE REDACCIÓN DE ALTO NIVEL: {filename[:200]}\n\n"
        f"Estudia el estilo de redacción jurídica de este engrose y replica su calidad "
        f"de argumentación, estructura lógica y prosa judicial."
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_REDACCION},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": estudio},
        ]
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    output_dir = Path(__file__).parent / "ft_data"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "training_data.jsonl"

    training_examples = []
    stats = {"redaccion": 0, "structure": {}, "errors": 0, "total_chars": 0}

    # ── 1. REDACCIÓN DE ALTO NIVEL (12 — all) ──────────────────────────────
    print("=" * 70)
    print("📚 FASE 1: Extrayendo ejemplos de redacción de alto nivel...")
    print("=" * 70)

    redaccion_dir = Path(REDACCION_DIR)
    if redaccion_dir.exists():
        for f in sorted(redaccion_dir.iterdir()):
            if f.suffix.lower() in (".pdf", ".doc", ".docx", ".txt"):
                print(f"  📄 {f.name[:80]}...")
                text = extract_text(str(f))
                if text and len(text) > 500:
                    example = build_training_example_redaccion(text, f.stem)
                    training_examples.append(example)
                    stats["redaccion"] += 1
                    stats["total_chars"] += len(text)
                    print(f"     ✅ {len(text):,} chars extracted")
                else:
                    stats["errors"] += 1
                    print(f"     ❌ Insufficient text ({len(text) if text else 0} chars)")
    else:
        print(f"  ❌ Directory not found: {REDACCION_DIR}")

    # ── 2. STRUCTURE EXAMPLES (25 curated) ─────────────────────────────────
    print("\n" + "=" * 70)
    print("🏛️  FASE 2: Extrayendo ejemplos de estructura por tipo...")
    print("=" * 70)

    tipo_dirs = {
        "amparo_directo": "Amparos Directos",
        "amparo_revision": "Amparo en Revisión",
        "queja": "Quejas",
        "revision_fiscal": "Revisión Fiscal",
    }

    for tipo, filenames in CURATED_STRUCTURE.items():
        subdir = Path(SENTENCIAS_BASE) / tipo_dirs[tipo]
        stats["structure"][tipo] = 0

        print(f"\n  📁 {TIPO_LABELS.get(tipo, tipo)} ({len(filenames)} seleccionados)")

        for fname in filenames:
            fpath = subdir / fname
            if fpath.exists():
                print(f"    📄 {fname[:70]}...")
                text = extract_text(str(fpath))
                if text and len(text) > 500:
                    example = build_training_example_structure(text, tipo, fpath.stem)
                    training_examples.append(example)
                    stats["structure"][tipo] += 1
                    stats["total_chars"] += len(text)
                    print(f"       ✅ {len(text):,} chars | sentido: {classify_sentido(text, tipo)}")
                else:
                    stats["errors"] += 1
                    print(f"       ❌ Insufficient text")
            else:
                stats["errors"] += 1
                print(f"    ❌ NOT FOUND: {fname[:60]}")

    # ── 3. WRITE JSONL ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("💾 Escribiendo JSONL...")
    print("=" * 70)

    with open(output_file, "w", encoding="utf-8") as f:
        for ex in training_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # ── 4. STATS ───────────────────────────────────────────────────────────
    total_tokens_est = stats["total_chars"] // 4  # rough estimate
    cost_est = (total_tokens_est * 3) / 1_000_000 * 3  # 3 epochs

    print(f"\n{'='*70}")
    print(f"📊 RESULTADOS")
    print(f"{'='*70}")
    print(f"  Redacción de alto nivel: {stats['redaccion']} ejemplos")
    for tipo, count in stats["structure"].items():
        print(f"  {TIPO_LABELS.get(tipo, tipo):25s}: {count} ejemplos")
    print(f"  {'─'*40}")
    print(f"  Total ejemplos:          {len(training_examples)}")
    print(f"  Total caracteres:        {stats['total_chars']:,}")
    print(f"  Tokens estimados:        ~{total_tokens_est:,}")
    print(f"  Costo estimado (3 eps):  ~${cost_est:.2f} USD")
    print(f"  Errores:                 {stats['errors']}")
    print(f"\n  📁 Output: {output_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
