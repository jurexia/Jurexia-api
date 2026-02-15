#!/usr/bin/env python3
"""
terminator_leyes.py — Terminator para Leyes Estatales Mexicanas

Limpia y estructura el texto extraído de PDFs de leyes estatales.
Produce artículos completos con texto limpio y jerarquía preservada.

Pipeline:
  1. limpieza_headers()       → Remueve basura de headers/footers de legislaturas
  2. costura_inteligente()    → Une líneas cortadas por columnas del PDF
  3. split_por_articulos()    → Separa en artículos completos
  4. enriquecer_jerarquia()   → Extrae Título, Capítulo, Sección del contexto
  5. validar_articulo()       → Verifica integridad del artículo

Uso:
    from terminator_leyes import procesar_ley
    articulos = procesar_ley(texto_raw, nombre_ley)
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

MIN_ARTICLE_LEN = 40         # Mínimo de caracteres para un artículo válido
MAX_CHUNK_CHARS = 5000       # Máximo por chunk (~1250 tokens)
OVERLAP_CHARS = 400          # Overlap para artículos largos


# ══════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ArticuloLimpio:
    """Un artículo procesado y limpio, listo para embedding."""
    texto: str                    # Texto completo y limpio del artículo
    ref: str                      # "Art. 15", "Art. 15 Bis", "Transitorios"
    origen: str                   # Nombre completo de la ley
    titulo: str = ""              # "TÍTULO PRIMERO" (jerarquía)
    capitulo: str = ""            # "CAPÍTULO II" (jerarquía)
    seccion: str = ""             # "SECCIÓN TERCERA" (jerarquía)
    jerarquia_txt: str = ""       # "Ley X > Título I > Cap. II > Art. 15"
    chunk_index: int = 0          # Sub-chunk para artículos largos
    es_transitorio: bool = False  # Artículos transitorios


# ══════════════════════════════════════════════════════════════════════
# PASO 1: LIMPIEZA DE HEADERS/FOOTERS
# ══════════════════════════════════════════════════════════════════════

# Patrones de basura comunes en PDFs de legislaturas estatales
_BASURA_PATTERNS = [
    # Headers de páginas de legislaturas
    r'Periódico\s+Oficial\s+"?La\s+Sombra\s+de\s+Arteaga"?',
    r'PERIÓDICO\s+OFICIAL\s+DEL?\s+GOBIERNO',
    r'Gaceta\s+Oficial\s+del?\s+',
    r'Gaceta\s+Parlamentaria',
    r'DIARIO\s+OFICIAL',
    r'La\s+Sombra\s+de\s+Arteaga',
    
    # Paginación
    r'Pág(?:ina)?\.?\s*\d+\s*(?:de\s*\d+)?',
    r'Página\s*\d+\s*de\s*\d+',
    r'[-–—]\s*\d+\s*[-–—]',                  # — 15 —
    r'^\s*\d+\s*$',                           # Números sueltos (no artículos)
    
    # Fechas de publicación (en headers)
    r'\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|'
    r'septiembre|octubre|noviembre|diciembre)\s+de\s+\d{4}',
    
    # URLs y marcas web
    r'https?://\S+',
    r'www\.\S+',
    
    # Headers repetitivos de legislaturas
    r'(?:LX|LI|LII|LIII|LIV|LV|LVI|LVII|LVIII|LIX|LXI|LXII|LXIII|LXIV|LXV)\s*LEGISLATURA',
    r'H\.\s*CONGRESO\s+DEL\s+ESTADO',
    r'PODER\s+LEGISLATIVO\s+DEL\s+ESTADO',
    
    # Marcas de agua y sellos
    r'ÚLTIMA\s+REFORMA\s+PUBLICADA\s+EN\s+EL\s+P\.?\s*O\.?',
    r'Fe\s+de\s+erratas?\s+(?:publicada|DOF)',
    r'Nota\s+de\s+Editor:.*',
]

_BASURA_COMPILED = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _BASURA_PATTERNS]


def limpieza_headers(texto: str) -> str:
    """
    Elimina headers, footers, y basura intrusiva de PDFs de legislaturas.
    Preserva la estructura del contenido legal.
    """
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
    Une líneas cortadas por el formato de columnas del PDF.
    
    Problema: Los PDFs de legislaturas cortan líneas por ancho de columna,
    produciendo oraciones fragmentadas:
        "Los ciudadanos que come-"
        "tan infracciones serán"
    
    Solución: Unir líneas que no terminan en puntuación, respetando
    los límites de artículos, incisos y fracciones.
    """
    # 1. Unir palabras con guión al final de línea (hifenación)
    texto = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', texto)
    
    # 2. Unir líneas que no terminan en puntuación terminal
    # PERO respetar:
    #   - Artículos (Artículo N.)
    #   - Fracciones (I., II., III., etc.)
    #   - Incisos (a), b), c), etc.)
    #   - Numerales (1., 2., 3., etc.)
    #   - Títulos/Capítulos en MAYÚSCULAS
    texto = re.sub(
        r'(?<![\.;:!?\)])\n'              # Línea NO termina en puntuación
        r'(?!\s*(?:'                        # La siguiente línea NO empieza con:
        r'Art[ií]culo\s+'                   #   Artículo
        r'|[IVX]+\.\s'                      #   Fracciones romanas (I., II., etc.)
        r'|[a-z]\)\s'                       #   Incisos (a), b), etc.)
        r'|\d+\.\s'                         #   Numerales (1., 2., etc.)
        r'|TÍTULO\s'                        #   Títulos
        r'|CAPITULO\s|CAPÍTULO\s'           #   Capítulos
        r'|SECCIÓN\s|SECCION\s'             #   Secciones
        r'|LIBRO\s'                         #   Libros
        r'|TRANSITORI'                      #   Transitorios
        r'))',
        ' ',
        texto,
        flags=re.IGNORECASE
    )
    
    # 3. Limpiar espacios múltiples (pero NO eliminar \n\n que son párrafos)
    texto = re.sub(r'[ \t]{2,}', ' ', texto)
    texto = re.sub(r' +\.', '.', texto)
    
    return texto.strip()


# ══════════════════════════════════════════════════════════════════════
# PASO 3: SPLIT POR ARTÍCULOS
# ══════════════════════════════════════════════════════════════════════

# Regex para detectar el inicio de un artículo
_ART_BOUNDARY = re.compile(
    r'(?:^|\n)\s*'                                      # Inicio de línea
    r'(Art[ií]culo\s+\d+[\w]*'                          # "Artículo 15", "Artículo 15A"
    r'(?:\s+(?:BIS|TER|QUÁTER|QUATER|QUINQUIES))?'     # Sufijos opcionales
    r')\s*[\.\-\s]',                                     # Seguido de período, guión o espacio
    re.IGNORECASE | re.MULTILINE
)

# Regex para extraer la referencia de un artículo
_ART_REF = re.compile(
    r'Art[ií]culo\s+(\d+[\w]*(?:\s+(?:BIS|TER|QUÁTER|QUATER|QUINQUIES))?)',
    re.IGNORECASE
)

# Regex para detectar Transitorios
_TRANSITORIOS = re.compile(
    r'(?:^|\n)\s*(TRANSITORIOS?)\s*\n',
    re.IGNORECASE | re.MULTILINE
)

# Regex para detectar encabezados de estructura
_TITULO = re.compile(
    r'(?:^|\n)\s*(TÍTULO\s+[IVXLCDM\d]+(?:\s+[A-ZÁÉÍÓÚÑ,\s]+)?)',
    re.IGNORECASE | re.MULTILINE
)
_CAPITULO = re.compile(
    r'(?:^|\n)\s*(CAP[ÍI]TULO\s+[IVXLCDM\d]+(?:\s+[A-ZÁÉÍÓÚÑ,\s]+)?)',
    re.IGNORECASE | re.MULTILINE
)
_SECCION = re.compile(
    r'(?:^|\n)\s*(SECCI[ÓO]N\s+[IVXLCDM\d]+(?:\s+[A-ZÁÉÍÓÚÑ,\s]+)?)',
    re.IGNORECASE | re.MULTILINE
)


def _extraer_jerarquia(texto_previo: str, titulo_actual: str, 
                        capitulo_actual: str, seccion_actual: str):
    """
    Extrae la jerarquía actual (Título, Capítulo, Sección) del texto
    que precede a un artículo.
    """
    # Buscar el último Título, Capítulo y Sección en el texto previo
    for m in _TITULO.finditer(texto_previo):
        titulo_actual = m.group(1).strip()
        capitulo_actual = ""      # Reset capítulo al cambiar de título
        seccion_actual = ""       # Reset sección al cambiar de título
    
    for m in _CAPITULO.finditer(texto_previo):
        capitulo_actual = m.group(1).strip()
        seccion_actual = ""       # Reset sección al cambiar de capítulo
    
    for m in _SECCION.finditer(texto_previo):
        seccion_actual = m.group(1).strip()
    
    return titulo_actual, capitulo_actual, seccion_actual


def _construir_jerarquia_txt(origen: str, titulo: str, capitulo: str, 
                              seccion: str, ref: str) -> str:
    """Construye la cadena de jerarquía textual."""
    parts = [origen]
    if titulo:
        parts.append(titulo)
    if capitulo:
        parts.append(capitulo)
    if seccion:
        parts.append(seccion)
    parts.append(ref)
    return " > ".join(parts)


def split_por_articulos(texto: str, origen: str) -> list[ArticuloLimpio]:
    """
    Separa el texto limpio en artículos individuales completos.
    
    Estrategia:
    1. Detecta boundaries de artículos usando regex
    2. Cada artículo incluye todo el texto hasta el siguiente artículo
    3. Preserva la jerarquía (Título, Capítulo, Sección) por contexto
    4. Los artículos largos se sub-dividen con overlap
    5. Preámbulo y Transitorios se manejan como chunks especiales
    """
    if not texto.strip():
        return []
    
    articulos: list[ArticuloLimpio] = []
    
    # Encontrar todas las posiciones de artículos
    matches = list(_ART_BOUNDARY.finditer(texto))
    
    if not matches:
        # No hay artículos detectados → chunk como texto fijo
        return _chunk_texto_fijo(texto, origen, "Sección General")
    
    # Estado de jerarquía
    titulo_actual = ""
    capitulo_actual = ""
    seccion_actual = ""
    
    # Procesar preámbulo (texto antes del primer artículo)
    preambulo = texto[:matches[0].start()].strip()
    if preambulo and len(preambulo) > MIN_ARTICLE_LEN:
        # Extraer jerarquía del preámbulo
        titulo_actual, capitulo_actual, seccion_actual = _extraer_jerarquia(
            preambulo, titulo_actual, capitulo_actual, seccion_actual
        )
        articulos.extend(_chunk_texto_fijo(preambulo, origen, "Preámbulo"))
    
    # Procesar cada artículo
    for idx, match in enumerate(matches):
        # Texto del artículo = desde este match hasta el siguiente (o fin del texto)
        art_start = match.start()
        if idx + 1 < len(matches):
            art_end = matches[idx + 1].start()
        else:
            art_end = len(texto)
        
        art_text = texto[art_start:art_end].strip()
        
        if len(art_text) < MIN_ARTICLE_LEN:
            continue
        
        # Extraer referencia del artículo
        ref_match = _ART_REF.search(match.group(1))
        ref = f"Art. {ref_match.group(1)}" if ref_match else match.group(1).strip()[:30]
        
        # Actualizar jerarquía basada en el texto previo
        # (Headers de Título/Capítulo/Sección que estaban ANTES de este artículo)
        titulo_actual, capitulo_actual, seccion_actual = _extraer_jerarquia(
            art_text, titulo_actual, capitulo_actual, seccion_actual
        )
        
        jerarquia = _construir_jerarquia_txt(
            origen, titulo_actual, capitulo_actual, seccion_actual, ref
        )
        
        # Verificar si hay Transitorios en este artículo (para el último)
        trans_match = _TRANSITORIOS.search(art_text)
        if trans_match and idx == len(matches) - 1:
            # Separar contenido del artículo y transitorios
            art_only = art_text[:trans_match.start()].strip()
            trans_text = art_text[trans_match.start():].strip()
            
            if art_only and len(art_only) >= MIN_ARTICLE_LEN:
                articulos.extend(_chunk_articulo(
                    art_only, ref, origen, titulo_actual, 
                    capitulo_actual, seccion_actual, jerarquia
                ))
            
            if trans_text and len(trans_text) >= MIN_ARTICLE_LEN:
                trans_jer = _construir_jerarquia_txt(
                    origen, "", "", "", "Transitorios"
                )
                articulos.extend(_chunk_articulo(
                    trans_text, "Transitorios", origen, 
                    "", "", "", trans_jer, es_transitorio=True
                ))
            continue
        
        # Chunk normal del artículo
        articulos.extend(_chunk_articulo(
            art_text, ref, origen, titulo_actual, 
            capitulo_actual, seccion_actual, jerarquia
        ))
    
    return articulos


def _chunk_articulo(texto: str, ref: str, origen: str, 
                     titulo: str, capitulo: str, seccion: str,
                     jerarquia: str, es_transitorio: bool = False) -> list[ArticuloLimpio]:
    """
    Convierte un artículo en uno o más chunks.
    Si el artículo excede MAX_CHUNK_CHARS, lo subdivide con overlap.
    """
    if len(texto) <= MAX_CHUNK_CHARS:
        return [ArticuloLimpio(
            texto=texto,
            ref=ref,
            origen=origen,
            titulo=titulo,
            capitulo=capitulo,
            seccion=seccion,
            jerarquia_txt=jerarquia,
            chunk_index=0,
            es_transitorio=es_transitorio,
        )]
    
    # Subdividir artículos largos
    parts = _split_long_text(texto)
    return [
        ArticuloLimpio(
            texto=part,
            ref=ref,
            origen=origen,
            titulo=titulo,
            capitulo=capitulo,
            seccion=seccion,
            jerarquia_txt=jerarquia,
            chunk_index=i,
            es_transitorio=es_transitorio,
        )
        for i, part in enumerate(parts)
    ]


def _chunk_texto_fijo(texto: str, origen: str, ref: str) -> list[ArticuloLimpio]:
    """Chunk fallback para texto sin artículos (preámbulos, exposiciones de motivos)."""
    parts = _split_long_text(texto, max_chars=3500)
    return [
        ArticuloLimpio(
            texto=part.strip(),
            ref=ref if len(parts) == 1 else f"{ref} ({i+1})",
            origen=origen,
            jerarquia_txt=f"{origen} > {ref}",
            chunk_index=i,
        )
        for i, part in enumerate(parts) if len(part.strip()) >= MIN_ARTICLE_LEN
    ]


def _split_long_text(texto: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Subdivide texto largo en partes con overlap, priorizando cortes en párrafos."""
    if len(texto) <= max_chars:
        return [texto]
    
    parts = []
    start = 0
    while start < len(texto):
        end = start + max_chars
        if end >= len(texto):
            parts.append(texto[start:])
            break
        
        # Buscar el mejor punto de corte (prioridad: párrafo > oración > palabra)
        split_point = texto.rfind('\n\n', start + max_chars // 2, end)
        if split_point == -1:
            split_point = texto.rfind('. ', start + max_chars // 2, end)
        if split_point == -1:
            split_point = texto.rfind(' ', start + max_chars // 2, end)
        if split_point == -1:
            split_point = end
        else:
            split_point += 1  # Incluir delimitador
        
        parts.append(texto[start:split_point])
        start = split_point - OVERLAP_CHARS  # Overlap
    
    return parts


# ══════════════════════════════════════════════════════════════════════
# PASO 4: PIPELINE COMPLETO
# ══════════════════════════════════════════════════════════════════════

def procesar_ley(texto_raw: str, nombre_ley: str) -> list[ArticuloLimpio]:
    """
    Pipeline completo: texto crudo de PDF → artículos limpios.
    
    Args:
        texto_raw: Texto extraído del PDF (sin procesar)
        nombre_ley: Nombre oficial de la ley
    
    Returns:
        Lista de ArticuloLimpio listos para embedding
    """
    # Normalizar line endings
    texto = texto_raw.replace('\r\n', '\n').replace('\r', '\n')
    
    # Paso 1: Limpiar headers/footers
    texto = limpieza_headers(texto)
    
    # Paso 2: Costura inteligente (unir líneas quebradas)
    texto = costura_inteligente(texto)
    
    # Paso 3: Split por artículos
    articulos = split_por_articulos(texto, nombre_ley)
    
    return articulos


# ══════════════════════════════════════════════════════════════════════
# DIAGNÓSTICO (para testing)
# ══════════════════════════════════════════════════════════════════════

def diagnostico(articulos: list[ArticuloLimpio]) -> dict:
    """Genera estadísticas de los artículos procesados."""
    if not articulos:
        return {"total": 0}
    
    lens = [len(a.texto) for a in articulos]
    refs = set(a.ref for a in articulos)
    transitorios = sum(1 for a in articulos if a.es_transitorio)
    
    return {
        "total": len(articulos),
        "articulos_unicos": len(refs),
        "transitorios": transitorios,
        "chars_min": min(lens),
        "chars_max": max(lens),
        "chars_avg": sum(lens) // len(lens),
        "con_jerarquia": sum(1 for a in articulos if a.titulo),
    }


if __name__ == "__main__":
    # Quick test with a sample
    sample = """
    TÍTULO PRIMERO
    DISPOSICIONES GENERALES
    
    CAPÍTULO I
    DEL OBJETO Y DEFINICIONES
    
    Artículo 1. La presente Ley es de orden público e interés social y tiene por 
    objeto regular las relaciones entre los particulares y el Estado en materia de
    infracciones cívicas.
    
    Artículo 2. Para los efectos de la presen-
    te Ley, se entenderá por:
    I. Autoridad: Los servidores públicos;
    II. Infracción: Toda conducta que atente contra el orden público;
    III. Sanción: La multa o arresto impuestos.
    
    CAPÍTULO II
    DE LAS INFRACCIONES
    
    Artículo 3. Son faltas cívicas las siguientes conductas:
    a) Orinar o defecar en la vía pública;
    b) Consumir bebidas alcohólicas en la vía pública;
    c) Alterar el orden público con ruidos excesivos.
    
    Artículo 4. Las sanciones por las infracciones previstas en el artículo anterior
    serán de 11 a 20 veces la Unidad de Medida y Actualización vigente.
    
    TRANSITORIOS
    
    PRIMERO. La presente Ley entrará en vigor al día siguiente de su publicación.
    SEGUNDO. Se derogan todas las disposiciones que se opongan a esta Ley.
    """
    
    articulos = procesar_ley(sample, "Ley de Cultura Cívica de Querétaro")
    stats = diagnostico(articulos)
    print(f"\n📊 DIAGNÓSTICO:")
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    print(f"\n📄 ARTÍCULOS:")
    for a in articulos:
        preview = a.texto[:100].replace('\n', ' ')
        print(f"   [{a.ref}] ({len(a.texto)} chars) {a.jerarquia_txt}")
        print(f"      {preview}...")
