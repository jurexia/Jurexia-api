"""LA VARA, MEDIDA. Qué tan bueno es un considerando de fondo.

David puso el listón donde tenía que ponerlo: «nuestro redactor debe ser mejor
que el secretario que redactó esos proyectos (yo)». Un objetivo así no se
cumple por opinión, así que esto mide lo que se puede contar.

LAS CINCO MEDIDAS SALEN DE LAS DEBILIDADES DE SUS PROPIOS ENGROSES, medidas
sobre los cinco que entregó:

  · DENSIDAD. De 3,798 palabras del estudio de la queja 233/2025, 2,260 —el
    59%— son transcripción literal de una ejecutoria de la Suprema Corte, y lo
    que sigue parafrasea lo mismo: el razonamiento propio cabe en 600. En la
    revisión fiscal 6/2025 la transcripción es el 53%; en el amparo directo
    642/2024, el 48% es relato del acto y de los conceptos. Un estudio que
    transcribe la mitad no razona la mitad.

  · EXHAUSTIVIDAD. La queja 143/2026 tiene tres agravios y el tribunal contesta
    uno sin decir nada de los otros dos. Eso es omisión de estudio, y es el
    reproche que se gana en revisión.

  · CONGRUENCIA INTERNA. El resolutivo del ARA 17/2025 remite «en términos del
    considerando séptimo» y el engrose no tiene séptimo considerando.

  · CONGRUENCIA DEL FALLO. El considerando de procedencia de la queja 143/2026
    dice «El presente recurso resulta IMPROCEDENTE» y el asunto se resolvió
    FUNDADO, revocando y ordenando admitir.

  · PROMESA CUMPLIDA. Los cinco abren con «es innecesario transcribir el
    contenido de la sentencia y los agravios», y tres de ellos transcriben
    después media sentencia. Prometer y no cumplir en el mismo documento es
    peor que no prometer.

NO MIDE EL CRITERIO, y no puede. Si el sentido es el correcto lo dice el
tribunal, no una regla. Esto mide la FACTURA: si el documento contesta todo lo
que se le planteó, si razona en vez de copiar, y si dice en un sitio lo mismo
que en otro.
"""

from __future__ import annotations

import re
import unicodedata

# El bloque que se estudia empieza donde el documento deja de narrar.
# EL RÓTULO DEL FONDO NO ES SIEMPRE «Estudio». En el ARA 17/2025 el
# considerando de fondo se llama «QUINTO. Cuestiones a resolver» y el que
# rotula «Estudio» no existe; buscar sólo esa palabra devolvía cero palabras y
# hacía pasar por perfecto el engrose que más defectos tiene. Se buscan los
# rótulos que el corpus usa para el fondo, y si ninguno aparece, se toma desde
# el último considerando hasta el resuelve, que es donde vive.
_RX_ESTUDIO = re.compile(
    r"\n\s*(?:OCTAVO|S[ÉE]PTIMO|SEXTO|QUINTO|CUARTO)\.\s*"
    r"(?:Estudio|Soluci[óo]n|Cuestiones\s+a\s+resolver|An[áa]lisis|Fondo)"
    r"[^\n]*\n", re.I)
_RX_SOLUCION = re.compile(r"\n\s*Soluci[óo]n\s*\n", re.I)


_RX_EXISTE = re.compile(
    r"\n\s*(PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|S[ÉE]PTIMO|OCTAVO|"
    r"NOVENO|D[ÉE]CIMO)\.", re.I)


def _norm(x: str) -> str:
    y = unicodedata.normalize("NFKD", (x or "").lower())
    return "".join(c for c in y if not unicodedata.combining(c))


def estudio_de(texto: str) -> str:
    """El considerando de fondo, aislado."""
    m = _RX_ESTUDIO.search(texto or "") or _RX_SOLUCION.search(texto or "")
    if not m:
        # Último recurso: desde el considerando más avanzado hasta el resuelve.
        todos = list(_RX_EXISTE.finditer(texto or ""))
        if not todos:
            return ""
        m = todos[-1]
    cuerpo = texto[m.end():]
    # EL FONDO NO TERMINA EN EL RESUELVE cuando después hay un considerando
    # más —«SEXTO. Revisión adhesiva queda sin materia»—: ése también es fondo
    # y cortarlo antes regalaba 223 palabras donde hay mil quinientas.
    fin = re.search(r"\n\s*(?:Por lo expuesto|R\s*E\s*S\s*U\s*E\s*L\s*V\s*E)", cuerpo)
    return cuerpo[:fin.start()] if fin else cuerpo


# ── 1. DENSIDAD ────────────────────────────────────────────────────────────
# Lo transcrito es lo entrecomillado y los bloques en cursiva; aquí, sobre
# texto plano, lo que va entre comillas de cualquier familia. Se mide el peso
# de la cita, no su existencia: citar es necesario, vivir de la cita no.
_RX_CITA = re.compile(r"[«\"“]([^»\"”]{80,})[»\"”]")


def densidad(estudio: str) -> dict:
    pal = len(re.findall(r"\w+", estudio or ""))
    citado = sum(len(re.findall(r"\w+", m.group(1)))
                 for m in _RX_CITA.finditer(estudio or ""))
    return {"palabras": pal, "transcritas": citado,
            "propias": pal - citado,
            "pct_propio": round((pal - citado) / pal, 3) if pal else 0.0}


# ── 2. EXHAUSTIVIDAD ───────────────────────────────────────────────────────
_ORD = ("primer", "segundo", "tercer", "cuarto", "quinto", "sexto",
        "séptimo", "septimo", "octavo", "noveno", "décimo", "decimo")
_RX_ANUNCIA = re.compile(
    r"\b(?:en el|el)\s+(" + "|".join(_ORD) + r")\s+"
    r"(?:concepto|agravio|motivo|planteamiento)", re.I)
_RX_CALIFICA = re.compile(
    r"\b(?:es|son|resulta[n]?|deviene[n]?)\s+"
    r"(fundad[oa]s?|infundad[oa]s?|inoperantes?|ineficaces?|inatendibles?|"
    r"innecesari[oa]s?)\b", re.I)


# EL ESTUDIO AGRUPADO ES LEGÍTIMO, y contarlo como omisión era un fallo de la
# medida, no del documento. El engrose del ARA 17/2025 —el mejor de los cinco
# en este punto— no contesta agravio por agravio: hace «un solo movimiento de
# reencuadre que los hace caer todos a la vez». Lo que hay que exigir no es una
# calificación por planteamiento, sino que NINGUNO quede sin respuesta: o se le
# contesta, o se dice expresamente que se estudia con otro.
_RX_AGRUPA = re.compile(
    r"(?:se\s+)?(?:examinar|analizar|estudiar)[áa]n?\s+"
    r"(?:de\s+manera\s+)?conjunta(?:mente)?|"
    r"en\s+(?:un|el)\s+(?:solo\s+)?(?:bloque|apartado)[^.]{0,60}"
    r"(?:agravios?|conceptos?)|"
    r"por\s+(?:su\s+)?(?:estrecha\s+)?relaci[óo]n[^.]{0,40}"
    r"(?:se\s+)?(?:estudian|analizan|examinan)", re.I)


def exhaustividad(estudio: str) -> dict:
    """¿Queda algún planteamiento sin respuesta?"""
    anunciados = {_norm(m.group(1)).rstrip("o")
                  for m in _RX_ANUNCIA.finditer(estudio or "")}
    calificados = len(_RX_CALIFICA.findall(estudio or ""))
    agrupa = bool(_RX_AGRUPA.search(estudio or ""))
    # Con estudio agrupado basta UNA calificación que los cubra; sin él, hace
    # falta al menos una por planteamiento anunciado.
    ok = (calificados >= 1) if agrupa else (calificados >= len(anunciados))
    return {"planteamientos_anunciados": len(anunciados),
            "calificaciones_emitidas": calificados,
            "estudio_agrupado": agrupa,
            "contesta_todo": ok if anunciados else None}


# ── 3. CONGRUENCIA INTERNA ─────────────────────────────────────────────────
_RX_REMITE = re.compile(
    r"considerando\s+(primer[oa]|segundo|tercer[oa]|cuart[oa]|quint[oa]|"
    r"sext[oa]|s[ée]ptim[oa]|octav[oa]|noven[oa]|d[ée]cim[oa])", re.I)
def remisiones_rotas(texto: str) -> list:
    """Los «en términos del considerando séptimo» que no existen."""
    hay = {_norm(m.group(1)).rstrip("o") for m in _RX_EXISTE.finditer(texto or "")}
    fuera = []
    for m in _RX_REMITE.finditer(texto or ""):
        o = _norm(m.group(1)).rstrip("oa")
        if o not in {x.rstrip("oa") for x in hay}:
            fuera.append(m.group(0))
    return sorted(set(fuera))


# ── 4. CONGRUENCIA DEL FALLO ───────────────────────────────────────────────
def procedencia_contradice(texto: str) -> str:
    """«resulta improcedente» en un asunto que se resolvió por el fondo."""
    m = re.search(r"Procedencia\.(.{0,320})", texto or "", re.S)
    if not m:
        return ""
    if re.search(r"\bimprocedente\b", m.group(1), re.I) and \
       re.search(r"\b(?:es|son)\s+fundad|se\s+revoca|ampara\s+y\s+protege",
                 texto or "", re.I):
        return " ".join(m.group(1).split())[:150]
    return ""


# ── 5. PROMESA CUMPLIDA ────────────────────────────────────────────────────
def promesa_rota(texto: str) -> dict:
    """Prometió no transcribir y transcribió."""
    promete = bool(re.search(r"innecesari[oa]\s+(?:su\s+)?tra[ns]scri", texto or "", re.I))
    est = estudio_de(texto)
    d = densidad(est)
    return {"prometio": promete,
            "pct_transcrito_despues": round(1 - d["pct_propio"], 3) if d["palabras"] else 0.0,
            "rota": promete and d["palabras"] > 300 and d["pct_propio"] < 0.70}


def medir(texto: str) -> dict:
    est = estudio_de(texto)
    return {"densidad": densidad(est),
            "exhaustividad": exhaustividad(est),
            "remisiones_rotas": remisiones_rotas(texto),
            "procedencia_contradice": procedencia_contradice(texto),
            "promesa": promesa_rota(texto)}


# ── 6. LA CIFRA DEL ACERVO NO SE ESCRIBE ───────────────────────────────────
# Trampa nueva, de este cambio: la predicción entra al prompt como «EL ACERVO:
# Conceder (82% de 50 sentencias)» y un modelo que lee una cifra en su contexto
# tiende a citarla. Una sentencia que dice «el 82% de los tribunales concede»
# es impublicable: el criterio no se vota.
# EL `\b` DESPUÉS DEL `%` NO CASA NUNCA: el porcentaje no es carácter de
# palabra, así que la frontera exige uno detrás y detrás hay un espacio. Se
# quita, y con él el falso «no lo caza» que me devolvió mi propia prueba.
_RX_ESTADISTICA = re.compile(
    r"[^.]{0,90}\b\d{1,3}\s*(?:%|por\s+ciento)[^.]{0,60}"
    r"(?:tribunal|colegiad|sentencia|acervo|precedent)[^.]{0,60}\.|"
    r"[^.]{0,90}(?:tribunal|colegiad|acervo|precedent)[^.]{0,50}"
    r"\b\d{1,3}\s*(?:%|por\s+ciento)[^.]{0,60}\.|"
    r"[^.]{0,70}\bla\s+mayor[íi]a\s+de\s+los\s+(?:tribunales|colegiados)"
    r"[^.]{0,80}\.", re.I)


def estadistica_en_el_texto(texto: str) -> list:
    """Las frases que citan la jurimetría como si fuera argumento."""
    return sorted({" ".join(m.group(0).split())
                   for m in _RX_ESTADISTICA.finditer(texto or "")})


# ── 7. EL MISMO PÁRRAFO DOS VECES ──────────────────────────────────────────
# Medido en el ADC 642/2024: tres párrafos copiados palabra por palabra dentro
# del mismo considerando, ochenta líneas después. Cero criterio jurídico, cero
# coste: n-gramas de quince palabras repetidos dentro del estudio.
def duplicacion_interna(estudio: str, n: int = 15) -> list:
    pal = re.findall(r"\w+", (estudio or "").lower())
    if len(pal) < n * 3:
        return []
    vistos: dict = {}
    # Se separa por n, no por 3n: dos párrafos idénticos consecutivos son el
    # mismo defecto que dos separados por ochenta líneas, y exigir tres veces
    # la ventana dejaba fuera justo el caso que motivó la medida.
    crudos: list = []
    for i in range(len(pal) - n):
        g = " ".join(pal[i:i + n])
        if g in vistos and i - vistos[g] >= n:
            crudos.append((i, g))
        else:
            vistos.setdefault(g, i)
    # LOS SOLAPADOS CUENTAN UNA VEZ. Un párrafo repetido produce decenas de
    # n-gramas consecutivos; contarlos todos exagera el defecto. Se agrupan por
    # su POSICIÓN en el texto —que es lo que los hace el mismo hallazgo— y no
    # por su índice en la lista, que era el error de la primera versión: con él
    # dos repeticiones lejanas se fundían en una y una sola se contaba entera.
    fuera, ultimo = [], -10 ** 9
    for i, g in crudos:
        if i - ultimo > n:
            fuera.append(g)
            ultimo = i
    return fuera[:6]
