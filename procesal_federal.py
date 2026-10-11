"""NORMATIVA PROCESAL FEDERAL — una fuente propia para el procedimiento (11-oct-2026).

David: «implementa una estrategia perfecta para recuperación de fuentes y
respuestas procesales en penal, civil y familiar, laboral, agrario y
administrativo… que, cruzada con la ley sustantiva local, dé resultados muy
precisos».

POR QUÉ UNA COLECCIÓN APARTE Y NO `leyes_federales`. Los códigos procesales ya
estaban ahí, pero como un trozo más entre 164 leyes y sin nada que diga de qué
trata cada artículo en términos de PROCEDIMIENTO. La pregunta procesal típica
—«¿en qué plazo apelo el auto que desecha pruebas en un juicio familiar?»— tiene
tres ejes que el coseno solo no separa: la MATERIA (familiar ≠ penal ≠ laboral),
la FIGURA (recurso, plazo, prueba) y el TEXTO EXACTO del plazo. Aquí cada
artículo se guarda etiquetado con los tres, de su texto vigente (PDF de la
Cámara de Diputados, octubre de 2026, copia en GCS para el visor).

LA ESTRATEGIA DE RECUPERACIÓN (`buscar`):
1. `detectar` decide si la consulta es procesal, de qué materia(s) y qué
   figuras pide. Sin latencia: expresiones regulares sobre el texto
   normalizado. La materia que ya detectó el chat entra como pista.
2. Artículos citados («art. 471 del CNPP») → recuperación EXACTA por ley y
   número, puntuación 1.0.
3. Dos búsquedas densas en paralelo: por materia, y por materia + figura.
4. Reordenación: coseno + coincidencia de figuras + coincidencia léxica (los
   plazos y los nombres de los recursos se piden con palabras exactas) + un
   empujón si piden plazo y el artículo lo fija.
5. Expansión por remisión: los mejores arrastran los artículos de la MISMA ley
   a los que remiten («en los términos del artículo 52») —el plazo suele estar
   en el remitido—.
6. Un artículo, una entrada: de cada artículo queda su mejor trozo.

El chat los recibe con lugar garantizado (ver main.py, «PROCESAL FEDERAL») y con
una instrucción: fundar vía, plazo y recurso en estos artículos y cruzarlos con
la ley sustantiva local que ya viene en el contexto.
"""
from __future__ import annotations

import asyncio
import re
import unicodedata
from typing import Iterable, Optional

COLECCION = "procesal_federal"

# abreviatura → (nombre oficial, materia procesal, rango de artículos a ingerir)
# Rango None = la ley completa. Sólo se acota donde la ley mezcla lo sustantivo
# con lo procesal: la LFT (derecho procesal del trabajo y sus autoridades), el
# Código de Comercio (Libro Quinto, juicios mercantiles), el CFF (Título V,
# procedimientos administrativos) y la Ley Agraria (procuración y justicia).
LEYES = {
    "CNPP":   ("Código Nacional de Procedimientos Penales", "penal", None),
    "LNEP":   ("Ley Nacional de Ejecución Penal", "penal", None),
    "CNPCF":  ("Código Nacional de Procedimientos Civiles y Familiares", "civil_familiar", None),
    "CFPC":   ("Código Federal de Procedimientos Civiles", "civil_familiar", None),
    "CCom":   ("Código de Comercio", "mercantil", (1049, 1500)),
    "LFT":    ("Ley Federal del Trabajo", "laboral", (523, 1010)),
    "LFTSE":  ("Ley Federal de los Trabajadores al Servicio del Estado", "laboral", None),
    "LAgra":  ("Ley Agraria", "agrario", (134, 200)),
    "LOTA":   ("Ley Orgánica de los Tribunales Agrarios", "agrario", None),
    "LFPCA":  ("Ley Federal de Procedimiento Contencioso Administrativo", "administrativo", None),
    "LOTFJA": ("Ley Orgánica del Tribunal Federal de Justicia Administrativa", "administrativo", None),
    "LFPA":   ("Ley Federal de Procedimiento Administrativo", "administrativo", None),
    "CFF":    ("Código Fiscal de la Federación", "administrativo", (116, 196)),
    "LAmp":   ("Ley de Amparo", "amparo", None),
    "LOPJF":  ("Ley Orgánica del Poder Judicial de la Federación", "organica", None),
    "LGMASC": ("Ley General de Mecanismos Alternativos de Solución de Controversias", "masc", None),
}

MATERIAS_ETIQUETA = {
    "penal": "penal", "civil_familiar": "civil y familiar", "mercantil": "mercantil",
    "laboral": "laboral", "agrario": "agrario", "administrativo": "administrativo y fiscal",
    "amparo": "amparo", "organica": "orgánica (competencia federal)", "masc": "mecanismos alternativos",
}


def normalizar(t: str) -> str:
    t = unicodedata.normalize("NFKD", t or "")
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", t.lower())


# ── LAS FIGURAS ───────────────────────────────────────────────────────────────
# Las mismas expresiones etiquetan los artículos al ingerir y leen la consulta:
# si difirieran, la búsqueda por figura pediría etiquetas que nadie puso.
_F = {
    "plazo": r"\b(plazos?|terminos?|dentro de (los|las)? ?\w+ (dias|horas)|dias habiles|horas siguientes|cuantos dias|a mas tardar)\b",
    "recurso": r"\b(recursos?|apelacion|apelar|revocacion|reposicion|queja|reclamacion|inconformidad|casacion|impugna\w*)\b",
    "notificacion": r"\b(notifica\w*|citacion|cedulas?|estrados|edictos?|buzon)\b",
    "emplazamiento": r"\bemplaza\w*",
    "prueba": r"\b(pruebas?|probatori\w*|testimonial|testigos?|pericial|peritos?|confesional|documentales?|inspeccion judicial|desahogo|ofrecimiento|carga de la prueba)\b",
    "audiencia": r"\baudiencias?\b",
    "competencia": r"\b(competen\w*|incompetencia|jurisdiccion|ante quien|que juzgado|que tribunal)\b",
    "medida_cautelar": r"\b(medidas? cautelar\w*|providencias? precautori\w*|embargo|arraigo|prision preventiva|medidas? de proteccion|medidas? provisional\w*)\b",
    "suspension": r"\bsuspension\b",
    "ejecucion": r"\b(ejecucion|ejecutar|cumplimiento de (la )?(sentencia|ejecutoria|laudo)|via de apremio|remate|adjudicacion)\b",
    "incidente": r"\bincident\w*",
    "demanda": r"\b(demanda|escrito inicial|ampliacion de (la )?demanda|promover)\b",
    "contestacion": r"\b(contestacion|contestar|reconvencion|excepciones?|allanamiento)\b",
    "sentencia": r"\b(sentencias?|resolucion definitiva|laudo|ejecutoria)\b",
    "caducidad": r"\b(caducidad|preclusion|precluye|sobreseimiento|sobresee\w*|desistimiento)\b",
    "nulidad_actuaciones": r"\bnulidad de (actuaciones|notificaciones|lo actuado)\b",
    "apremio": r"\b(medidas de apremio|correcciones disciplinarias|multas?|costas|gastos y costas)\b",
    "impedimento": r"\b(impedimentos?|excusas?|excusarse|recusacion|recusar)\b",
    "acumulacion": r"\b(acumulacion|conexidad|litispendencia)\b",
    "conciliacion": r"\b(conciliacion|conciliatori\w*|mediacion|convenio|acuerdo reparatorio|mecanismos alternativos)\b",
    "improcedencia": r"\b(improcedencia|improcedente|desecha\w*|inadmisib\w*)\b",
    "via": r"\b(juicio oral|juicio ordinario|juicio especial|juicio ejecutivo|juicio sumario|procedimiento especial|procedimiento abreviado|procedimiento ordinario|via (ordinaria|especial|sumaria|ejecutiva|oral))\b",
}
FIGURAS = {k: re.compile(v) for k, v in _F.items()}

_NUM = (r"(?:\d+|un[oa]?|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|"
        r"quince|dieciseis|diecisiete|dieciocho|diecinueve|veinte|veinticinco|treinta|cuarenta(?: y (?:cinco|ocho))?|"
        r"sesenta|setenta y dos|noventa|ciento veinte|ciento ochenta)")
RE_PLAZO = re.compile(rf"\b{_NUM}\s+(?:dias?|horas|meses|mes|anos?)(?:\s+(?:habiles|naturales|siguientes))?\b")
RE_REMISION = re.compile(r"\bart(?:iculo|\.)s?\s+(\d{1,4})(?:\s*(?:,|y|o)\s*(\d{1,4}))?(?:\s*(?:,|y|o)\s*(\d{1,4}))?")
_OTRA_LEY = re.compile(r"^\W{0,3}(?:\w+\s+){0,3}(de la|del)\s+(constitucion|ley|codigo|reglamento)")


def figuras_de(texto_norm: str) -> list[str]:
    return [k for k, rx in FIGURAS.items() if rx.search(texto_norm)]


def plazos_de(texto_norm: str, tope: int = 6) -> list[str]:
    vistos = []
    for m in RE_PLAZO.finditer(texto_norm):
        p = m.group(0).strip()
        if p not in vistos:
            vistos.append(p)
        if len(vistos) >= tope:
            break
    return vistos


def remisiones_de(texto_norm: str, propio: Optional[int], tope: int = 6) -> list[int]:
    """Artículos de la MISMA ley a los que remite. «el artículo 14 de la
    Constitución» no cuenta: la cola que nombra otra ley lo descarta."""
    fuera = []
    for m in RE_REMISION.finditer(texto_norm):
        if _OTRA_LEY.match(texto_norm[m.end():m.end() + 60]):
            continue
        for g in m.groups():
            if g:
                n = int(g)
                if n != propio and n not in fuera:
                    fuera.append(n)
        if len(fuera) >= tope:
            break
    return fuera[:tope]


# ── LA CONSULTA ───────────────────────────────────────────────────────────────
_SENAL = re.compile(
    r"\b(plazo|termino|cuantos dias|en que tiempo|recurso|apela\w*|impugna\w*|revoca\w*|queja|revision|amparo|"
    r"demanda\w*|contesta\w*|emplaza\w*|notifica\w*|audiencia|prueba|ofrec\w*|desahog\w*|incidente|ejecucion|"
    r"embargo|cautelar|suspension|competen\w*|juzgado|tribunal|sentencia|auto|acuerdo|juicio|procedimiento|"
    r"tramit\w*|caducidad|preclu\w*|sobresee\w*|sobreseimiento|desist\w*|exhorto|edicto|perito|testig\w*|"
    r"vinculacion|imputacion|carpeta de investigacion|denuncia|querella|conciliacion|laudo|nulidad|via|"
    r"requisitos|como se promueve|como promuevo|que procede|procede)\b")

_MATERIA_DE_CONSULTA = {
    "penal": r"\b(penal|imputad\w*|imputacion|vinculacion|ministerio publico|fiscalia|carpeta de investigacion|victima|ofendid\w*|delito|prision preventiva|juez de control|tribunal de enjuiciamiento|procedimiento abreviado|suspension condicional|acusacion|querella|denuncia|sentenciad\w*|libertad anticipada|preliberacion\w*|juez de ejecucion)\b",
    "civil_familiar": r"\b(civil|familiar|divorcio|alimentos|pension alimenticia|guarda|custodia|convivencia|patria potestad|adopcion|filiacion|paternidad|sucesion|testament\w*|intestado|arrendamiento|desahucio|usucapion|prescripcion positiva|reivindicatori\w*|danos y perjuicios|interdiccion|violencia familiar|rectificacion de acta|cnpcf|hipotecari\w*)\b",
    "mercantil": r"\b(mercantil|pagare|letra de cambio|titulos? de credito|cheque|ejecutivo mercantil|oral mercantil|comerciante|sociedad mercantil|credito bancario|codigo de comercio)\b",
    "laboral": r"\b(laboral|despid\w*|trabajador\w*|patron\w*|salario\w*|salarios caidos|reinstalacion|indemnizacion constitucional|tribunal laboral|centro de conciliacion|conciliacion prejudicial|sindicat\w*|huelga|riesgo de trabajo|burocrat\w*|tribunal federal de conciliacion|lft)\b",
    "agrario": r"\b(agrari\w*|ejid\w*|comuner\w*|parcela\w*|tierras comunales|asamblea ejidal|procuraduria agraria|tribunal unitario|tribunal superior agrario|registro agrario)\b",
    "administrativo": r"\b(administrativ\w*|juicio de nulidad|contencioso|tfja|tribunal federal de justicia administrativa|sat|credito fiscal|fiscal|multa|clausura|recurso de revocacion|infonavit|imss|visita domiciliaria|procedimiento administrativo de ejecucion|cofepris|profeco|licitacion|autoridad fiscal|cff)\b",
    "amparo": r"\b(amparo|quejoso|autoridad responsable|acto reclamado|tercero interesado|juez de distrito|tribunal colegiado|ejecutoria|suspension provisional|suspension definitiva)\b",
    "masc": r"\b(mediacion|mecanismos alternativos|facilitador|acuerdo reparatorio|justicia restaurativa)\b",
}
_MATERIA_RX = {k: re.compile(v) for k, v in _MATERIA_DE_CONSULTA.items()}

# La materia que ya detectó el chat, traducida a la nuestra.
_PISTA = {
    "penal": "penal", "procesal_penal": "penal", "civil": "civil_familiar", "familiar": "civil_familiar",
    "procesal_civil": "civil_familiar", "mercantil": "mercantil", "laboral": "laboral", "agrario": "agrario",
    "administrativo": "administrativo", "fiscal": "administrativo", "amparo": "amparo",
}

# «art. 471 del CNPP», «artículo 17 de la Ley de Amparo»
_ALIAS_LEY = [
    ("CNPP", r"cnpp|codigo nacional de procedimientos penales"),
    ("CNPCF", r"cnpcf|codigo nacional de procedimientos civiles"),
    ("CFPC", r"cfpc|codigo federal de procedimientos civiles"),
    ("LAmp", r"ley de amparo"),
    ("LFT", r"lft|ley federal del trabajo"),
    ("LFTSE", r"lftse|ley federal de los trabajadores al servicio del estado"),
    ("LFPCA", r"lfpca|ley federal de procedimiento contencioso"),
    ("LFPA", r"lfpa|ley federal de procedimiento administrativo"),
    ("CFF", r"cff|codigo fiscal de la federacion"),
    ("CCom", r"codigo de comercio"),
    ("LAgra", r"ley agraria"),
    ("LNEP", r"ley nacional de ejecucion penal"),
    ("LOPJF", r"ley organica del poder judicial de la federacion"),
    ("LOTFJA", r"ley organica del tribunal federal de justicia administrativa"),
    ("LOTA", r"ley organica de los tribunales agrarios"),
]
_RE_CITA = re.compile(r"\bart(?:iculo|\.)?s?\s*(\d{1,4})\s*(?:bis|ter)?\s*(?:,|\s)*(?:de la|del|de)\s+(" +
                      "|".join(p for _, p in _ALIAS_LEY) + r")")


# EL FOCO: lo que la pregunta pide, no todo lo que menciona. «¿Qué recurso
# procede contra el auto que desecha pruebas?» nombra pruebas, pero pide un
# RECURSO; sin foco, los artículos de pruebas ganaban por mencionar más cosas.
_FOCO = [
    ("recurso", r"\b(que recurso|recurso procede|recurso que procede|como (se )?impugn\w*|medio de impugnacion|es apelable|se puede apelar|procede (la )?(apelacion|revocacion|queja|revision))\b"),
    ("plazo", r"\b(plazo|termino|cuantos dias|en que tiempo|cuanto tiempo|hasta cuando|a partir de cuando)\b"),
    ("competencia", r"\b(ante quien|que juez|que juzgado|que tribunal|competente|competencia)\b"),
    ("prueba", r"\b(que pruebas|como (se )?prueb\w*|ofrecer pruebas|carga de la prueba)\b"),
    ("medida_cautelar", r"\b(medidas? cautelar\w*|embargo precautorio|providencia precautoria)\b"),
    ("suspension", r"\b(suspension provisional|suspension definitiva|suspender el acto)\b"),
]
_FOCO_RX = [(k, re.compile(v)) for k, v in _FOCO]
# Leyes de ámbito especial que sólo deben ganar cuando la consulta lo pide.
_ESPECIAL = {
    "LFTSE": re.compile(r"\b(burocrat\w*|servidor(es)? public\w*|al servicio del estado|dependencia|tfca|tribunal federal de conciliacion)\b"),
    "CFPC": re.compile(r"\b(codigo federal de procedimientos civiles|cfpc|federacion como parte|juicio federal)\b"),
}


def detectar(consulta: str, pistas: Optional[Iterable[str]] = None) -> dict:
    q = normalizar(consulta)
    citas = []
    for m in _RE_CITA.finditer(q):
        for abrev, patron in _ALIAS_LEY:
            if re.fullmatch(patron, m.group(2)):
                citas.append((abrev, int(m.group(1))))
                break
    procesal = bool(_SENAL.search(q)) or bool(citas)
    materias = [k for k, rx in _MATERIA_RX.items() if rx.search(q)]
    for p in pistas or []:
        m = _PISTA.get(str(p or "").lower())
        if m and m not in materias:
            materias.append(m)
    figuras = figuras_de(q)
    if "competencia" in figuras and "organica" not in materias:
        materias.append("organica")
    if materias and "amparo" not in materias and re.search(r"\bamparo\b", q):
        materias.append("amparo")
    # Una pregunta sustantiva («¿qué pena tiene el fraude procesal?») no es
    # procesal aunque diga «procesal»; tampoco «¿qué es un contrato?».
    fuerte = bool(citas) or len(figuras) >= 2 or bool(re.search(
        r"\b(plazo|termino|cuantos dias|recurso|apela\w*|emplaza\w*|vinculacion|imputacion|"
        r"amparo|incidente|suspension|que procede|via)\b", q))
    foco = [k for k, rx in _FOCO_RX if rx.search(q)]
    return {"procesal": procesal and (fuerte or bool(figuras)), "fuerte": fuerte,
            "materias": materias, "figuras": figuras, "foco": foco, "citas": citas, "q": q}


def _lexico(q: str, texto_norm: str) -> float:
    pal = {w for w in re.findall(r"[a-z]{5,}", q)} - {"cuantos", "cuales", "cuando", "donde", "puedo", "tengo",
                                                     "sobre", "contra", "porque", "debe", "deben", "dentro"}
    if not pal:
        return 0.0
    return sum(1 for w in pal if w in texto_norm) / len(pal)


async def buscar(qdrant, consulta: str, vector, pistas=None, limite: int = 8) -> tuple[dict, list]:
    """(detección, [(id, puntuación, payload)]) — vacío si la consulta no es procesal."""
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

    det = detectar(consulta, pistas)
    if not det["procesal"] or not vector:
        return det, []

    async def _q(filtro, n):
        r = qdrant.query_points(collection_name=COLECCION, query=vector, using="dense",
                                query_filter=filtro, limit=n, with_payload=True)
        if asyncio.iscoroutine(r):
            r = await r
        return list(getattr(r, "points", None) or [])

    async def _exactos():
        fuera = []
        for abrev, art in det["citas"][:4]:
            r = qdrant.scroll(collection_name=COLECCION, limit=3, with_payload=True, scroll_filter=Filter(must=[
                FieldCondition(key="abreviatura", match=MatchValue(value=abrev)),
                FieldCondition(key="articulo_num", match=MatchValue(value=art))]))
            if asyncio.iscoroutine(r):
                r = await r
            fuera += list(r[0] if isinstance(r, tuple) else r)
        return fuera

    mats = det["materias"]
    f_mat = Filter(must=[FieldCondition(key="materia_procesal", match=MatchAny(any=mats))]) if mats else None
    tareas = [_q(f_mat, 40), _exactos()]
    figs = det["foco"] or [f for f in det["figuras"] if f != "via"] or det["figuras"]
    for grupo_fig in ([[f] for f in det["foco"][:2]] if len(det["foco"]) > 1 else ([figs] if figs else [])):
        must = [FieldCondition(key="figuras", match=MatchAny(any=grupo_fig))]
        if mats:
            must.append(FieldCondition(key="materia_procesal", match=MatchAny(any=mats)))
        tareas.append(_q(Filter(must=must), 20))
    res = await asyncio.gather(*tareas, return_exceptions=True)
    densos = [p for i, grupo in enumerate(res) if i != 1 and isinstance(grupo, list) for p in grupo]
    exactos = res[1] if isinstance(res[1], list) else []

    puntos = {}
    for p in densos:
        pl = p.payload or {}
        s = float(p.score or 0)
        comunes = len(set(det["figuras"]) & set(pl.get("figuras") or []))
        s += min(0.12, 0.05 * comunes)
        s += 0.06 * _lexico(det["q"], pl.get("texto_norm") or normalizar(pl.get("texto_raw", "")))
        if "plazo" in det["figuras"] and pl.get("plazos"):
            s += 0.04
        if set(det["foco"]) & set(pl.get("figuras") or []):
            s += 0.06
        rx_esp = _ESPECIAL.get(pl.get("abreviatura"))
        if rx_esp is not None and not rx_esp.search(det["q"]):
            s -= 0.05
        if mats and pl.get("materia_procesal") == mats[0]:
            s += 0.02
        if str(p.id) not in puntos or puntos[str(p.id)][1] < s:
            puntos[str(p.id)] = (str(p.id), s, pl)
    for p in exactos:
        puntos[str(p.id)] = (str(p.id), 1.0, p.payload or {})

    # Un artículo, una entrada: su mejor trozo.
    por_art = {}
    for t in sorted(puntos.values(), key=lambda x: x[1], reverse=True):
        llave = (t[2].get("abreviatura"), t[2].get("articulo_num"), t[2].get("ref"))
        por_art.setdefault(llave, t)
    orden = sorted(por_art.values(), key=lambda x: x[1], reverse=True)
    # Con dos focos («¿ante qué juez y en qué plazo?») cada uno se lleva sus
    # dos mejores; si no, el más fuerte llena la lista y el otro no aparece.
    mejores = []
    if len(det["foco"]) > 1:
        for f in det["foco"]:
            mejores += [t for t in orden if f in (t[2].get("figuras") or []) and t not in mejores][:2]
    mejores += [t for t in orden if t not in mejores]
    mejores = sorted(mejores[:limite], key=lambda x: x[1], reverse=True)

    # Expansión por remisión: lo que remite a otro artículo de la misma ley
    # suele dejar ahí el plazo o el requisito.
    vistos = {(t[2].get("abreviatura"), t[2].get("articulo_num")) for t in mejores}
    extra = []
    for _id, s, pl in mejores[:3]:
        refs = [n for n in (pl.get("remisiones") or [])
                if (pl.get("abreviatura"), n) not in vistos][:2]
        if not refs:
            continue
        try:
            r = qdrant.scroll(collection_name=COLECCION, limit=len(refs) * 2, with_payload=True,
                              scroll_filter=Filter(must=[
                                  FieldCondition(key="abreviatura", match=MatchValue(value=pl.get("abreviatura"))),
                                  FieldCondition(key="articulo_num", match=MatchAny(any=refs))]))
            if asyncio.iscoroutine(r):
                r = await r
            for q in (r[0] if isinstance(r, tuple) else r):
                k = ((q.payload or {}).get("abreviatura"), (q.payload or {}).get("articulo_num"))
                if k not in vistos and (q.payload or {}).get("chunk_index", 0) == 0:
                    vistos.add(k)
                    extra.append((str(q.id), s - 0.05, q.payload or {}))
        except Exception:
            continue
    return det, mejores + extra[:3]


def instruccion(leyes_presentes: Iterable[str]) -> str:
    por_nombre = {v[0]: v[1] for v in LEYES.values()}
    vistas = []
    for n in leyes_presentes or []:
        m = por_nombre.get(n)
        if m and m not in vistas:
            vistas.append(m)
    mats = ", ".join(MATERIAS_ETIQUETA.get(m, m) for m in vistas) or "la que corresponda"
    return (
        "NORMATIVA PROCESAL FEDERAL (fuente <documento tipo=\"LEY_PROCESAL_FEDERAL\">, texto vigente a octubre "
        f"de 2026; materia: {mats}). Funda la VÍA, el PLAZO (cuéntalo: hábiles o naturales, desde cuándo corre) "
        "y el RECURSO en esos artículos y cítalos por número. Cruza lo procesal con la ley sustantiva local "
        "que también viene en el contexto: la sustantiva dice qué se reclama; la procesal, cómo, ante quién y "
        "en qué plazo. En civil y familiar, el Código Nacional de Procedimientos Civiles y Familiares entra en "
        "vigor por entidad según su declaratoria: si en el contexto viene el código de procedimientos local "
        "de la entidad del usuario, advierte cuál rige y no los mezcles. Si un plazo no está en estos "
        "artículos, dilo; no lo calcules de memoria.")
