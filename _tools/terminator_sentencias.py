#!/usr/bin/env python3
"""
terminator_sentencias.py — Terminator para Sentencias de Tribunales Colegiados

Limpia y estructura el texto extraído de PDFs de sentencias judiciales.
A diferencia de terminator_leyes.py (que divide por artículos), las sentencias
son documentos narrativos largos que se dividen por secciones judiciales.

Pipeline:
  1. limpieza_headers()          → Remueve basura de headers/footers de tribunales
  2. costura_inteligente()       → Une líneas cortadas por columnas del PDF
  3. split_por_secciones()       → Separa en RESULTANDOS, CONSIDERANDOS, RESOLUTIVOS
  4. sub_split_considerandos()   → Divide cada CONSIDERANDO individual
  5. chunk_seccion()             → Aplica chunking con overlap a secciones largas

Uso:
    from terminator_sentencias import procesar_sentencia
    chunks = procesar_sentencia(texto_raw, archivo_origen, tipo_sentencia)
"""

import re
from dataclasses import dataclass
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

MIN_CHUNK_LEN = 80          # Mínimo de caracteres para un chunk válido
MAX_CHUNK_CHARS = 5000      # Máximo por chunk (~1250 tokens)
OVERLAP_CHARS = 500         # Overlap para secciones largas


# ══════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ChunkSentencia:
    """Un chunk procesado y limpio de una sentencia, listo para embedding."""
    texto: str                        # Texto completo y limpio del chunk
    seccion: str                      # "encabezado", "resultandos", "considerandos", "resolutivos", "general"
    considerando_num: str             # "PRIMERO", "SEGUNDO", etc. (si aplica)
    archivo_origen: str               # Nombre del archivo PDF
    tipo_sentencia: str               # "amparo_directo", "amparo_revision", etc.
    chunk_index: int = 0              # Sub-chunk index para secciones largas
    jerarquia_txt: str = ""           # "Amparo Directo > CONSIDERANDOS > QUINTO"


# ══════════════════════════════════════════════════════════════════════
# PASO 1: LIMPIEZA DE HEADERS/FOOTERS
# ══════════════════════════════════════════════════════════════════════

_BASURA_PATTERNS = [
    # Headers de tribunales
    r'PODER\s+JUDICIAL\s+DE\s+LA\s+FEDERACI[ÓO]N',
    r'TRIBUNAL\s+COLEGIADO\s+(?:DE\s+CIRCUITO\s+)?EN\s+MATERIA',
    r'CONSEJO\s+DE\s+LA\s+JUDICATURA\s+FEDERAL',
    r'SUPREMA\s+CORTE\s+DE\s+JUSTICIA\s+DE\s+LA\s+NACI[ÓO]N',
    
    # Paginación
    r'Pág(?:ina)?\.?\s*\d+\s*(?:de\s*\d+)?',
    r'Página\s*\d+\s*de\s*\d+',
    r'[-–—]\s*\d+\s*[-–—]',
    r'^\s*\d{1,3}\s*$',                      # Números sueltos de página
    
    # URLs y marcas
    r'https?://\S+',
    r'www\.\S+',
    
    # Sellos y marcas de agua
    r'FIRMA\s+ELECTR[ÓO]NICA',
    r'Firmado\s+electr[óo]nicamente',
    r'CERTIFICADO\s+DIGITAL',
    r'Este\s+documento\s+(?:es|fue)\s+(?:generado|firmado)',
    r'Documento\s+firmado\s+electr[óo]nicamente',
    
    # Marcas de fojas
    r'^\s*(?:Foja|Fj\.?|Fs?\.)\s*\d+\s*(?:(?:frente|vuelta|fv?\.)\s*)?$',
]

_BASURA_COMPILED = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _BASURA_PATTERNS]


def limpieza_headers(texto: str) -> str:
    """Elimina headers, footers, y basura intrusiva de PDFs de tribunales."""
    for pattern in _BASURA_COMPILED:
        texto = pattern.sub(' ', texto)
    
    # Colapsar whitespace excesivo
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    texto = re.sub(r'[ \t]{2,}', ' ', texto)
    
    return texto.strip()


# ══════════════════════════════════════════════════════════════════════
# PASO 2: COSTURA INTELIGENTE
# ══════════════════════════════════════════════════════════════════════

def costura_inteligente(texto: str) -> str:
    """
    Une líneas cortadas por el formato del PDF.
    Respeta boundaries de secciones judiciales.
    """
    # 1. Unir palabras con guión al final de línea
    texto = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', texto)
    
    # 2. Unir líneas que no terminan en puntuación terminal
    # PERO respetar boundaries de secciones
    texto = re.sub(
        r'(?<![.;:!?\)])\n'
        r'(?!\s*(?:'
        r'R\s*E\s*S\s*U\s*L\s*T\s*A\s*N\s*D\s*O'
        r'|C\s*O\s*N\s*S\s*I\s*D\s*E\s*R\s*A\s*N\s*D\s*O'
        r'|P\s*U\s*N\s*T\s*O\s*S?\s*R\s*E\s*S\s*O\s*L\s*U\s*T\s*I\s*V\s*O'
        r'|RESULTANDOS?\b'
        r'|CONSIDERANDOS?\b'
        r'|PUNTOS?\s*RESOLUTIVOS?\b'
        r'|PRIMERO\b|SEGUNDO\b|TERCERO\b|CUARTO\b|QUINTO\b'
        r'|SEXTO\b|S[ÉE]PTIMO\b|OCTAVO\b|NOVENO\b|D[ÉE]CIMO\b'
        r'|UND[ÉE]CIMO\b|DUO?D[ÉE]CIMO\b'
        r'|[IVX]+\.\s'
        r'|[a-z]\)\s'
        r'))',
        ' ',
        texto,
        flags=re.IGNORECASE
    )
    
    # 3. Limpiar espacios múltiples
    texto = re.sub(r'[ \t]{2,}', ' ', texto)
    texto = re.sub(r' +\.', '.', texto)
    
    return texto.strip()


# ══════════════════════════════════════════════════════════════════════
# PASO 3: SPLIT POR SECCIONES JUDICIALES
# ══════════════════════════════════════════════════════════════════════

# Regex for section boundaries -- support both normal and spaced-out formatting
_RESULTANDOS_RE = re.compile(
    r'(?:^|\n)\s*(?:R\s*E\s*S\s*U\s*L\s*T\s*A\s*N\s*D\s*O|RESULTANDOS?)\s*:?\s*\n',
    re.IGNORECASE | re.MULTILINE
)

_CONSIDERANDOS_RE = re.compile(
    r'(?:^|\n)\s*(?:C\s*O\s*N\s*S\s*I\s*D\s*E\s*R\s*A\s*N\s*D\s*O|CONSIDERANDOS?)\s*:?\s*\n',
    re.IGNORECASE | re.MULTILINE
)

_RESOLUTIVOS_RE = re.compile(
    r'(?:^|\n)\s*(?:P\s*U\s*N\s*T\s*O\s*S?\s*R\s*E\s*S\s*O\s*L\s*U\s*T\s*I\s*V\s*O|PUNTOS?\s*RESOLUTIVOS?)\s*:?\s*\n',
    re.IGNORECASE | re.MULTILINE
)


# Ordinal numbers used in CONSIDERANDOS
_ORDINAL_RE = re.compile(
    r'(?:^|\n)\s*(PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|'
    r'S[ÉE]PTIMO|OCTAVO|NOVENO|D[ÉE]CIMO|'
    r'UND[ÉE]CIMO|DUO?D[ÉE]CIMO|DECIMO?\s*TERCERO|'
    r'DECIMO?\s*CUARTO|DECIMO?\s*QUINTO|DECIMO?\s*SEXTO|'
    r'DECIMO?\s*S[ÉE]PTIMO|DECIMO?\s*OCTAVO|DECIMO?\s*NOVENO|'
    r'VIG[ÉE]SIMO)'
    r'\s*[\.\-\:]',
    re.IGNORECASE | re.MULTILINE
)


def _find_section_boundaries(texto: str) -> dict:
    """
    Find the start positions of each major section.
    Returns dict with keys: 'encabezado', 'resultandos', 'considerandos', 'resolutivos'
    Each value is (start, end) tuple or None.
    """
    sections = {}
    
    # Find all matches
    res_match = _RESULTANDOS_RE.search(texto)
    con_match = _CONSIDERANDOS_RE.search(texto)
    pts_match = _RESOLUTIVOS_RE.search(texto)
    
    # Build ordered list of boundaries
    boundaries = []
    if res_match:
        boundaries.append(('resultandos', res_match.end()))
    if con_match:
        boundaries.append(('considerandos', con_match.end()))
    if pts_match:
        boundaries.append(('resolutivos', pts_match.end()))
    
    # Sort by position
    boundaries.sort(key=lambda x: x[1])
    
    # Extract sections
    # Encabezado: everything before first section
    first_start = boundaries[0][1] if boundaries else len(texto)
    first_header_start = min(
        (m.start() for m in [res_match, con_match, pts_match] if m),
        default=len(texto)
    )
    if first_header_start > 0:
        sections['encabezado'] = (0, first_header_start)
    
    for i, (name, start) in enumerate(boundaries):
        if i + 1 < len(boundaries):
            # Find the header match for the NEXT section to know where this one ends
            next_name = boundaries[i + 1][0]
            if next_name == 'resultandos' and res_match:
                end = res_match.start()
            elif next_name == 'considerandos' and con_match:
                end = con_match.start()
            elif next_name == 'resolutivos' and pts_match:
                end = pts_match.start()
            else:
                end = boundaries[i + 1][1]
        else:
            end = len(texto)
        sections[name] = (start, end)
    
    return sections


def split_por_secciones(texto: str) -> dict:
    """
    Split sentencia text into major judicial sections.
    Returns dict mapping section name -> text content.
    """
    boundaries = _find_section_boundaries(texto)
    
    sections = {}
    for name, (start, end) in boundaries.items():
        content = texto[start:end].strip()
        if content and len(content) >= MIN_CHUNK_LEN:
            sections[name] = content
    
    # Fallback: if no sections detected, treat entire text as "general"
    if not sections:
        sections['general'] = texto.strip()
    
    return sections


def sub_split_considerandos(texto_considerandos: str) -> list[tuple[str, str]]:
    """
    Split CONSIDERANDOS section into individual numbered considerandos.
    Returns list of (ordinal_name, text) tuples.
    """
    matches = list(_ORDINAL_RE.finditer(texto_considerandos))
    
    if not matches:
        return [("GENERAL", texto_considerandos)]
    
    parts = []
    for i, match in enumerate(matches):
        ordinal = match.group(1).strip().upper()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(texto_considerandos)
        
        part_text = texto_considerandos[start:end].strip()
        if part_text and len(part_text) >= MIN_CHUNK_LEN:
            parts.append((ordinal, part_text))
    
    # If there's text before the first ordinal, include it
    if matches[0].start() > MIN_CHUNK_LEN:
        preamble = texto_considerandos[:matches[0].start()].strip()
        if preamble:
            parts.insert(0, ("PREÁMBULO", preamble))
    
    return parts


# ══════════════════════════════════════════════════════════════════════
# PASO 4: CHUNKING CON OVERLAP
# ══════════════════════════════════════════════════════════════════════

def _split_long_text(texto: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Subdivide texto largo en partes con overlap."""
    if len(texto) <= max_chars:
        return [texto]
    
    parts = []
    start = 0
    while start < len(texto):
        end = start + max_chars
        if end >= len(texto):
            parts.append(texto[start:])
            break
        
        # Find best split point
        split_point = texto.rfind('\n\n', start + max_chars // 2, end)
        if split_point == -1:
            split_point = texto.rfind('. ', start + max_chars // 2, end)
        if split_point == -1:
            split_point = texto.rfind(' ', start + max_chars // 2, end)
        if split_point == -1:
            split_point = end
        else:
            split_point += 1
        
        parts.append(texto[start:split_point])
        start = split_point - OVERLAP_CHARS
    
    return parts


# ══════════════════════════════════════════════════════════════════════
# PIPELINE COMPLETO
# ══════════════════════════════════════════════════════════════════════

def procesar_sentencia(
    texto_raw: str, 
    archivo_origen: str, 
    tipo_sentencia: str
) -> list[ChunkSentencia]:
    """
    Pipeline completo: texto crudo de PDF → chunks limpios de sentencia.
    
    Args:
        texto_raw: Texto extraído del PDF
        archivo_origen: Nombre del archivo PDF
        tipo_sentencia: "amparo_directo", "amparo_revision", etc.
    
    Returns:
        Lista de ChunkSentencia listos para embedding
    """
    # Normalizar line endings
    texto = texto_raw.replace('\r\n', '\n').replace('\r', '\n')
    
    # Paso 1: Limpiar headers/footers
    texto = limpieza_headers(texto)
    
    # Paso 2: Costura inteligente
    texto = costura_inteligente(texto)
    
    if len(texto) < MIN_CHUNK_LEN:
        return []
    
    # Paso 3: Split por secciones
    secciones = split_por_secciones(texto)
    
    chunks: list[ChunkSentencia] = []
    tipo_label = tipo_sentencia.replace('_', ' ').title()
    
    for seccion_nombre, seccion_texto in secciones.items():
        if seccion_nombre == 'considerandos':
            # Sub-split considerandos by ordinal number
            sub_parts = sub_split_considerandos(seccion_texto)
            for ordinal, part_text in sub_parts:
                jerarquia = f"{tipo_label} > CONSIDERANDOS > {ordinal}"
                sub_chunks = _split_long_text(part_text)
                for i, sub_text in enumerate(sub_chunks):
                    if len(sub_text.strip()) >= MIN_CHUNK_LEN:
                        chunks.append(ChunkSentencia(
                            texto=sub_text.strip(),
                            seccion="considerandos",
                            considerando_num=ordinal,
                            archivo_origen=archivo_origen,
                            tipo_sentencia=tipo_sentencia,
                            chunk_index=i,
                            jerarquia_txt=jerarquia,
                        ))
        else:
            # Other sections: just chunk by size
            jerarquia = f"{tipo_label} > {seccion_nombre.upper()}"
            sub_chunks = _split_long_text(seccion_texto)
            for i, sub_text in enumerate(sub_chunks):
                if len(sub_text.strip()) >= MIN_CHUNK_LEN:
                    chunks.append(ChunkSentencia(
                        texto=sub_text.strip(),
                        seccion=seccion_nombre,
                        considerando_num="",
                        archivo_origen=archivo_origen,
                        tipo_sentencia=tipo_sentencia,
                        chunk_index=i,
                        jerarquia_txt=jerarquia,
                    ))
    
    return chunks


# ══════════════════════════════════════════════════════════════════════
# DIAGNÓSTICO
# ══════════════════════════════════════════════════════════════════════

def diagnostico(chunks: list[ChunkSentencia]) -> dict:
    """Genera estadísticas de los chunks procesados."""
    if not chunks:
        return {"total": 0}
    
    lens = [len(c.texto) for c in chunks]
    secciones = {}
    for c in chunks:
        secciones[c.seccion] = secciones.get(c.seccion, 0) + 1
    
    return {
        "total_chunks": len(chunks),
        "chars_min": min(lens),
        "chars_max": max(lens),
        "chars_avg": sum(lens) // len(lens),
        "chars_total": sum(lens),
        "por_seccion": secciones,
    }


if __name__ == "__main__":
    import sys
    
    # Quick test with a file argument
    if len(sys.argv) > 1:
        import pymupdf
        filepath = sys.argv[1]
        doc = pymupdf.open(filepath)
        text = "\n".join(page.get_text("text") for page in doc)
        doc.close()
        
        chunks = procesar_sentencia(text, filepath, "amparo_directo")
        stats = diagnostico(chunks)
        
        print(f"\n📊 DIAGNÓSTICO:")
        for k, v in stats.items():
            print(f"   {k}: {v}")
        
        print(f"\n📄 CHUNKS:")
        for c in chunks[:10]:
            preview = c.texto[:100].replace('\n', ' ')
            print(f"   [{c.seccion}] {c.considerando_num or '-'} ({len(c.texto)} chars)")
            print(f"      {c.jerarquia_txt}")
            print(f"      {preview}...")
    else:
        print("Uso: python terminator_sentencias.py <archivo.pdf>")
