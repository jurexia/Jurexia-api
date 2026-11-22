# -*- coding: utf-8 -*-
"""EL VOCABULARIO CANÓNICO DE LA JURIMETRÍA (22-nov-2026).

David: «vamos a organizar esto para que la jurimetría sea efectiva por circuito (y estado de la república);
tenemos que tener datos limpios y bien ordenados para que sea precisa».

El censo de `sentencias_holdings` (205,139 sentencias, `redactor-sentencias/oaj/censo_holdings.py`) encontró:
  · tres grafías del tribunal: «8TCC_ADM» (circuitos 1-6), el nombre completo (16) y «1TCC»/«TCC_PENAL» (22);
  · 23 grafías de materia («Trabajo» y «laboral», «Civil» y «civil», «fiscal»…), 40 de tipo de asunto y 30 de
    sentido; la web mandaba «administrativo» y lo guardado es «administrativa»: 0 resultados;
  · «confirma» contado siempre como NIEGA y «modifica» como SOBRESEE: en un recurso confirmar es que NO prospera
    lo que pidió quien recurrió, sea quejoso o autoridad, y modificar es que prospera en parte.

Este módulo es la ÚNICA tabla de equivalencias. La usan la escritura de los campos `j_*` en Qdrant
(`redactor-sentencias/oaj/jurimetria/canon_qdrant.py`), `/api/jurimetria` y, por copia, SIAPJF. Es puro: sin red,
sin Qdrant, para probarlo solo (`test_jurimetria_canon.py`).

Dos preguntas distintas, dos campos:
  · `j_prospera`: ¿prosperó lo que pidió QUIEN PROMOVIÓ este asunto? En amparo directo, el quejoso; en un recurso,
    el recurrente (sea quien sea). Sale del sentido solo: si / no / parcial / sin_fondo.
  · `j_quejoso`: ¿cómo quedó el QUEJOSO (o el particular, en revisión fiscal)? gana / gana_en_parte / pierde /
    sin_fondo. En amparo directo es lo mismo que lo anterior; en revisión y queja depende de quién recurrió y sale
    de la síntesis oficial de la OAJ («Confirma y Ampara») o de la lectura de los resolutivos.
"""
from __future__ import annotations

import datetime as _dt
import re
import unicodedata

VERSION = 1
# Toma de protesta de las magistraturas electas (reforma judicial de 2024): las sentencias de antes y de después
# no son de los mismos tribunales aunque se llamen igual.
CORTE_REFORMA = "2025-09-01"


def plano(s) -> str:
    """Minúsculas, sin acentos, «_» como espacio y espacios colapsados."""
    t = unicodedata.normalize("NFKD", str(s or ""))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower().replace("_", " ")
    return " ".join(t.split())


def _nulo(s) -> bool:
    return plano(s) in ("", "null", "none", "nan", "n/d", "nd", "∅")


# ── circuito ↔ estado ─────────────────────────────────────────────────────────────────────────────────────────
# Cada entidad federativa tiene su circuito (el 32.º, Colima, desde 2013). Los centros auxiliares no son circuito.
ESTADOS = {
    "1": "Ciudad de México", "2": "Estado de México", "3": "Jalisco", "4": "Nuevo León", "5": "Sonora",
    "6": "Puebla", "7": "Veracruz", "8": "Coahuila", "9": "San Luis Potosí", "10": "Tabasco", "11": "Michoacán",
    "12": "Sinaloa", "13": "Oaxaca", "14": "Yucatán", "15": "Baja California", "16": "Guanajuato",
    "17": "Chihuahua", "18": "Morelos", "19": "Tamaulipas", "20": "Chiapas", "21": "Guerrero", "22": "Querétaro",
    "23": "Zacatecas", "24": "Nayarit", "25": "Durango", "26": "Baja California Sur", "27": "Quintana Roo",
    "28": "Tlaxcala", "29": "Hidalgo", "30": "Aguascalientes", "31": "Campeche", "32": "Colima",
}
_ALIAS_ESTADO = {plano(v): k for k, v in ESTADOS.items()}
_ALIAS_ESTADO.update({
    "cdmx": "1", "ciudad de mexico": "1", "distrito federal": "1", "df": "1",
    "mexico": "2", "edomex": "2", "estado de mexico": "2",
    "coahuila de zaragoza": "8", "michoacan de ocampo": "11", "veracruz de ignacio de la llave": "7",
    "queretaro de arteaga": "22", "bc": "15", "bcs": "26",
})


def circuito_de_estado(estado) -> str | None:
    """«CIUDAD_DE_MEXICO», «Nuevo León», «edomex» → «1», «4», «2». None si no se reconoce."""
    return _ALIAS_ESTADO.get(plano(estado))


def estado_de_circuito(circuito) -> str | None:
    return ESTADOS.get(str(circuito or "").strip().lstrip("0") or "?")


def circuito_de_consulta(circuito=None, estado=None) -> str | None:
    """Lo que manda la web: el número de circuito («22», «022») o el estado. Gana el circuito si viene."""
    c = re.sub(r"\D", "", str(circuito or "")).lstrip("0")
    if c and c in ESTADOS:
        return c
    return circuito_de_estado(estado) if estado else None


# ── ordinales de los nombres oficiales ────────────────────────────────────────────────────────────────────────
_UNIDADES = {"primer": 1, "primero": 1, "segundo": 2, "tercer": 3, "tercero": 3, "cuarto": 4, "quinto": 5,
             "sexto": 6, "septimo": 7, "octavo": 8, "noveno": 9}
_DECENAS = {"decimo": 10, "vigesimo": 20, "trigesimo": 30}
_ESPECIALES = {"undecimo": 11, "duodecimo": 12}


def _partir_compuesto(w: str) -> list:
    """«decimosexto» → [decimo, sexto]; «decimoctavo» → [decimo, octavo]; lo demás, igual."""
    for d in _DECENAS:
        if w.startswith(d) and len(w) > len(d):
            r = w[len(d):]
            if r in _UNIDADES:
                return [d, r]
            if "o" + r in _UNIDADES:
                return [d, "o" + r]
    return [w]


def numero_ordinal(texto) -> int | None:
    """«Décimo Primer» → 11, «Decimosexto» → 16, «Vigésimo Segundo» → 22, «Trigésimo» → 30, «Primer» → 1;
    None si no es ordinal."""
    p = [x for w in plano(texto).split() for x in _partir_compuesto(w)]
    if not p:
        return None
    if len(p) == 1 and p[0] in _ESPECIALES:
        return _ESPECIALES[p[0]]
    n, i = 0, 0
    if p[0] in _DECENAS:
        n, i = _DECENAS[p[0]], 1
    if i < len(p) and p[i] in _UNIDADES:
        n += _UNIDADES[p[i]]
        i += 1
    return n if n and i == len(p) else None


_NOMBRE_ORDINAL = {1: "Primer", 2: "Segundo", 3: "Tercer", 4: "Cuarto", 5: "Quinto", 6: "Sexto", 7: "Séptimo",
                   8: "Octavo", 9: "Noveno", 10: "Décimo", 20: "Vigésimo", 30: "Trigésimo"}


def nombre_ordinal(n: int) -> str:
    """11 → «Décimo Primer», 22 → «Vigésimo Segundo» (así se nombran los tribunales y los circuitos)."""
    if n in _NOMBRE_ORDINAL:
        return _NOMBRE_ORDINAL[n]
    d, u = (n // 10) * 10, n % 10
    if d in _NOMBRE_ORDINAL and u in _NOMBRE_ORDINAL:
        return f"{_NOMBRE_ORDINAL[d]} {_NOMBRE_ORDINAL[u]}"
    return ""


# ── tribunal ──────────────────────────────────────────────────────────────────────────────────────────────────
_MATERIA_ORGANO = {"administrativa": "ADM", "civil": "CIV", "penal": "PEN", "trabajo": "LAB", "laboral": "LAB",
                   "mixta": "MIX"}
_CLAVE_MATERIA = {"ADM": "ADM", "ADMINISTRATIVA": "ADM", "CIV": "CIV", "CIVIL": "CIV", "PEN": "PEN",
                  "PENAL": "PEN", "LAB": "LAB", "TRA": "LAB", "TRABAJO": "LAB", "LABORAL": "LAB"}
_NOMBRE_MATERIA = {"ADM": "Administrativa", "CIV": "Civil", "PEN": "Penal", "LAB": "de Trabajo", "MIX": "Mixta"}
# El circuito 22 escribe sus claves sin materia o sin ordinal; es el catálogo de `fase_espejo.TRIBUNALES_22`
# y de SIAPJF (`motor/juridico/metodo/precedentes.py`).
_ORGANOS_22 = {
    "1TCC": "Primer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito",
    "2TCC": "Segundo Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito",
    "3TCC": "Tercer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito",
    "TCC_PENAL": "Tribunal Colegiado en Materias Penal y Administrativa del Vigésimo Segundo Circuito",
    "TCC_ADM": "Tribunal Colegiado en Materias Administrativa y de Trabajo del Vigésimo Segundo Circuito",
}
_RX_CABEZA = re.compile(r"^(?P<ord>(?:[a-z]+ ){0,2}?)tribunal(?:es)? colegiados?\b")
_RX_MATERIAS = re.compile(r"\bmaterias? (?P<mat>.+?)(?= del? (?:[a-z]+ ){0,2}?[a-z]+ circuito\b| con (?:residencia|sede) en\b|$)")
_RX_CIRC = re.compile(r"\bdel? (?P<circ>(?:[a-z]+ ){0,2}?[a-z]+) circuito\b")
_RX_CLAVE = re.compile(r"^(\d{1,2})?TCC(?:_([A-Z]+))?$")


def tribunal_de_nombre(nombre, circuito=None) -> dict | None:
    """Del nombre oficial al tribunal canónico:
    «Tercer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito, con residencia en
    Querétaro, Querétaro» → {"id": "22:3:ADM-CIV", "n": 3, "circuito": "22", "materias": "ADM-CIV",
    "nombre": «… del Vigésimo Segundo Circuito», "sede": «Querétaro, Querétaro»}. None si no es un nombre."""
    if _nulo(nombre):
        return None
    original = " ".join(str(nombre).split())
    p = " ".join(plano(original).replace(",", " ").replace(".", " ").split())
    m = _RX_CABEZA.match(p)
    if not m:
        return None
    n = numero_ordinal(m.group("ord")) if m.group("ord").strip() else 0
    if n is None:
        return None
    mats = []
    mm = _RX_MATERIAS.search(p)
    for w in (mm.group("mat").split() if mm else []):
        if w in _MATERIA_ORGANO and _MATERIA_ORGANO[w] not in mats:
            mats.append(_MATERIA_ORGANO[w])
    c = None
    mc = _RX_CIRC.search(p)
    if mc:
        k = numero_ordinal(mc.group("circ"))
        c = str(k) if k and str(k) in ESTADOS else None
    if c is None and circuito is not None and str(circuito).strip() in ESTADOS:
        c = str(circuito).strip()
    partes = re.split(r"[\s,]+con\s+(?:residencia|sede)\s+en\s+", original, maxsplit=1, flags=re.I)
    corto = partes[0].strip(" ,;.")
    sede = partes[1].strip(" ,;.") if len(partes) > 1 else None
    mats = sorted(mats)          # «Penal y Administrativa» y «Administrativa y Penal» son la misma clave
    return {"id": f"{c or 'x'}:{n}:{'-'.join(mats) or 'SIN'}", "n": n, "circuito": c,
            "materias": "-".join(mats) or None, "nombre": corto, "sede": sede}


def tribunal_de_clave(clave, circuito) -> dict | None:
    """«8TCC_ADM» + "1" → Octavo Tribunal Colegiado en Materia Administrativa del Primer Circuito; las cinco claves
    del 22 por su catálogo. None si la clave no se entiende (no se adivina)."""
    if _nulo(clave):
        return None
    c = str(circuito or "").strip()
    k = str(clave).upper().replace(" ", "")
    if c == "22" and k in _ORGANOS_22:
        return tribunal_de_nombre(_ORGANOS_22[k], c)
    m = _RX_CLAVE.match(k)
    if not m or c not in ESTADOS:
        return None
    n = int(m.group(1)) if m.group(1) else 0
    mat = _CLAVE_MATERIA.get(m.group(2) or "", None) if m.group(2) else None
    if m.group(2) and not mat:
        return None
    partes = [nombre_ordinal(n) if n else "", "Tribunal Colegiado",
              f"en Materia {_NOMBRE_MATERIA[mat]}" if mat else "", f"del {nombre_ordinal(int(c))} Circuito"]
    nombre = " ".join(x for x in partes if x)
    return {"id": f"{c}:{n}:{mat or 'SIN'}", "n": n, "circuito": c, "materias": mat, "nombre": nombre, "sede": None}


def tribunal(clave, circuito, completo=None) -> dict | None:
    """El tribunal de un punto del acervo: `tribunal_completo`, el `tribunal` si ya es un nombre, o la clave."""
    return (tribunal_de_nombre(completo, circuito) or tribunal_de_nombre(clave, circuito)
            or tribunal_de_clave(clave, circuito))


# ── tipo de asunto ────────────────────────────────────────────────────────────────────────────────────────────
TIPOS = ("amparo_directo", "amparo_revision", "revision_incidental", "queja", "revision_fiscal",
         "revision_contencioso", "reclamacion", "inconformidad", "conflicto_competencial", "impedimento",
         "inejecucion", "repeticion_acto", "otro")
NOMBRE_TIPO = {"amparo_directo": "amparo directo", "amparo_revision": "amparo en revisión",
               "revision_incidental": "revisión incidental (suspensión)", "queja": "queja",
               "revision_fiscal": "revisión fiscal", "revision_contencioso": "revisión contencioso administrativa",
               "reclamacion": "reclamación", "inconformidad": "inconformidad",
               "conflicto_competencial": "conflicto competencial", "impedimento": "impedimento",
               "inejecucion": "incidente de inejecución", "repeticion_acto": "repetición del acto reclamado",
               "otro": "otro"}
RECURSOS = {"amparo_revision", "revision_incidental", "queja", "revision_fiscal", "revision_contencioso",
            "reclamacion", "inconformidad", "impedimento", "inejecucion", "repeticion_acto"}


def tipo_canon(t) -> str | None:
    p = plano(t)
    if not p or p in ("null", "none"):
        return None
    if "directo" in p:
        return "amparo_directo"
    if "fiscal" in p:
        return "revision_fiscal"
    if "contencios" in p:
        return "revision_contencioso"
    if "incidente en revision" in p or "revision incidental" in p:
        return "revision_incidental"
    if "inejecuci" in p:
        return "inejecucion"
    if "repetici" in p:
        return "repeticion_acto"
    if "revision" in p:
        return "amparo_revision"
    if "queja" in p:
        return "queja"
    if "reclamaci" in p:
        return "reclamacion"
    if "inconformidad" in p:
        return "inconformidad"
    if "competenc" in p:
        return "conflicto_competencial"
    if "impediment" in p:
        return "impedimento"
    return "otro"


def tipo_de_consulta(t) -> str | None:
    """Lo que escribe la web o el secretario («Amparo directo», «AD», «revisión fiscal») → tipo canónico."""
    p = plano(t)
    atajos = {"ad": "amparo_directo", "ar": "amparo_revision", "ri": "revision_incidental", "q": "queja",
              "rf": "revision_fiscal", "rec": "reclamacion", "inc": "inconformidad", "cc": "conflicto_competencial"}
    if p in atajos:
        return atajos[p]
    if p in TIPOS:
        return p
    c = tipo_canon(p)
    return None if c in (None, "otro") else c


# ── materia ───────────────────────────────────────────────────────────────────────────────────────────────────
MATERIAS = ("administrativa", "civil", "laboral", "penal", "constitucional", "agraria", "comun", "mixta")
_MATERIA = {
    "administrativa": ("administrativa", None), "administrativo": ("administrativa", None),
    "fiscal": ("administrativa", "fiscal"), "administrativa | fiscal": ("administrativa", "fiscal"),
    "administrativa | civil": ("mixta", None),
    "civil": ("civil", None), "mercantil": ("civil", "mercantil"), "familiar": ("civil", "familiar"),
    "laboral": ("laboral", None), "trabajo": ("laboral", None), "labor": ("laboral", None),
    "penal": ("penal", None), "constitucional": ("constitucional", None), "agraria": ("agraria", None),
    "comun": ("comun", None), "mixta": ("mixta", None),
}


def materia_canon(m) -> tuple[str | None, str | None]:
    """(materia, submateria): «Trabajo» → (laboral, None); «fiscal» → (administrativa, fiscal);
    «mercantil» → (civil, mercantil). (None, None) si no se sabe."""
    return _MATERIA.get(plano(m), (None, None))


def materia_de_consulta(m) -> str | None:
    """«administrativo» (la web), «Laboral», «fiscal» → la materia canónica por la que se filtra."""
    return materia_canon(m)[0]


# ── sentido ───────────────────────────────────────────────────────────────────────────────────────────────────
SENTIDOS = ("concede", "concede_parcial", "niega", "sobresee", "confirma", "revoca", "modifica", "fundado",
            "infundado", "inoperante", "desecha", "sin_materia", "incompetencia", "competencia", "otro")
_SENTIDO = {
    "concede": "concede", "parcialmente concede": "concede_parcial", "niega": "niega", "sobresee": "sobresee",
    "confirma": "confirma", "revoca": "revoca", "modifica": "modifica",
    "fundado": "fundado", "fundada": "fundado", "esencialmente fundado": "fundado", "procedente": "fundado",
    "infundado": "infundado", "infundada": "infundado", "inoperante": "inoperante",
    "desecha": "desecha", "improcedente": "desecha",
    "sin materia": "sin_materia",
    "incompetencia": "incompetencia", "incompetente": "incompetencia", "declina": "incompetencia",
    "declina competencia": "incompetencia", "remite": "incompetencia",
    "resuelve competencia": "competencia", "resuelve": "competencia",
    "otro": "otro", "devuelve": "otro", "inexistente": "otro",
}


def sentido_canon(s) -> str | None:
    """El sentido guardado en la lista cerrada; None si el acervo no lo sabe (vacío, «null»)."""
    if _nulo(s):
        return None
    return _SENTIDO.get(plano(s), "otro")


_SIN_FONDO = {"sobresee", "desecha", "sin_materia", "incompetencia"}


def prospera(tipo, sentido) -> str | None:
    """¿Prosperó lo que pidió quien promovió ESTE asunto? si / no / parcial / sin_fondo; None si el sentido no
    basta para saberlo (p. ej. «concede» en una revisión: depende de quién recurrió)."""
    t, s = tipo, sentido
    if s is None or t is None:
        return None
    if t == "amparo_directo":
        return {"concede": "si", "concede_parcial": "parcial", "niega": "no"}.get(s) or (
            "sin_fondo" if s in _SIN_FONDO else None)
    if t in RECURSOS:
        r = {"confirma": "no", "revoca": "si", "modifica": "parcial", "fundado": "si", "infundado": "no",
             "inoperante": "no", "desecha": "sin_fondo", "sin_materia": "sin_fondo",
             "incompetencia": "sin_fondo"}.get(s)
        return r
    return None


def quejoso(tipo, sentido, prospera_=None) -> str | None:
    """¿Cómo quedó el quejoso (en revisión fiscal, el particular)? gana / gana_en_parte / pierde / sin_fondo; None si
    hace falta saber quién recurrió (revisión y queja con «confirma»/«revoca»)."""
    t, s = tipo, sentido
    if s is None or t is None:
        return None
    if t in ("amparo_directo", "amparo_revision", "revision_incidental"):
        r = {"concede": "gana", "concede_parcial": "gana_en_parte", "niega": "pierde"}.get(s)
        if r:
            return r
        if t == "amparo_directo" and s in _SIN_FONDO:
            return "sin_fondo"
        if s == "sobresee":
            return "sin_fondo"
        return None
    if t in ("revision_fiscal", "revision_contencioso"):
        # Recurre siempre la autoridad: que no prospere su recurso es que gana el particular.
        return {"no": "gana", "si": "pierde", "parcial": "gana_en_parte", "sin_fondo": "sin_fondo"}.get(
            prospera_ if prospera_ is not None else prospera(t, s))
    return None


# ── la síntesis oficial de la OAJ («Confirma y Ampara», «1. Revoca. 2. Sobresee.») ─────────────────────────────
_RX_AMPARA = re.compile(r"(?<!no )(?<!no se )\b(ampara|concede el amparo|se concede)\b")
_RX_NO_AMPARA = re.compile(r"\b(no ampara|no se ampara|niega el amparo|se niega)\b")
_RX_SOBRESEE = re.compile(r"\bsobrese")


def quejoso_de_sintesis(s) -> str | None:
    """De la síntesis oficial a cómo quedó el quejoso. «Confirma y Ampara» → gana; «1. Sobresee. 2. Ampara.» →
    gana_en_parte; «Confirma y No ampara» → pierde; «Revoca y Sobresee» → sin_fondo; None si no lo dice."""
    p = plano(s)
    if not p:
        return None
    amp, noamp, sob = bool(_RX_AMPARA.search(p)), bool(_RX_NO_AMPARA.search(p)), bool(_RX_SOBRESEE.search(p))
    if amp and (noamp or sob):
        return "gana_en_parte"
    if amp:
        return "gana"
    if noamp:
        return "pierde" if not sob else "pierde"
    if sob:
        return "sin_fondo"
    return None


def decision_de_sintesis(s) -> str | None:
    """La decisión del colegiado en la síntesis oficial (confirma / revoca / modifica / fundado / infundado /
    sin_materia / desecha); None si no la dice o dice dos."""
    p = plano(s)
    hall = set()
    for rx, d in ((r"\bconfirm", "confirma"), (r"\brevoc", "revoca"), (r"\bmodific", "modifica"),
                  (r"\b(?<!in)fundad", "fundado"), (r"\binfundad", "infundado"), (r"\bsin materia", "sin_materia"),
                  (r"\bdesech", "desecha"), (r"\binoperant", "inoperante")):
        if re.search(rx, p):
            hall.add(d)
    if "infundado" in hall and "fundado" in hall:
        hall.discard("fundado")
    return hall.pop() if len(hall) == 1 else None


# ── fecha y periodo ───────────────────────────────────────────────────────────────────────────────────────────
def fecha_iso(f, hoy: _dt.date | None = None) -> str | None:
    """«2024-07-08» o «08-07-2024» → «2024-07-08»; None si no es una fecha real entre 2000 y hoy."""
    s = str(f or "").strip()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s) or None
    if m:
        a, me, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        m = re.match(r"^(\d{2})[-/](\d{2})[-/](\d{4})", s)
        if not m:
            return None
        d, me, a = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        x = _dt.date(a, me, d)
    except ValueError:
        return None
    if x < _dt.date(2000, 1, 1) or x > (hoy or _dt.date.today()):
        return None
    return x.isoformat()


def periodo(fecha) -> str | None:
    """«anterior» o «nueva_integracion» (desde el 1-sep-2025); None sin fecha."""
    f = fecha_iso(fecha)
    if not f:
        return None
    return "nueva_integracion" if f >= CORTE_REFORMA else "anterior"


def ponente_canon(nombre) -> str | None:
    """«Mgdo. JUAN  Bonilla Pizano» → «Juan Bonilla Pizano»; None si vacío."""
    if _nulo(nombre):
        return None
    t = re.sub(r"(?i)^\s*(magistrad[oa]s?|mgd[oa]\.?|ponente:?|lic\.?|dr\.?|dra\.?)\s+", "", str(nombre)).strip(" .,;")
    t = " ".join(t.split())
    return " ".join(w.capitalize() if w.isupper() or w.islower() else w for w in t.split()) or None


# ── el punto completo ─────────────────────────────────────────────────────────────────────────────────────────
def campos(payload: dict, oficial: dict | None = None) -> dict:
    """Los campos `j_*` de un punto de `sentencias_holdings`. `oficial` (opcional) es la fila del listado de la
    OAJ del mismo NEUN: {organo, fecha, titular, sintesis, tipo, materia}. Lo oficial gana a lo leído."""
    p = payload or {}
    o = oficial or {}
    circ = str(p.get("circuito") or "").strip() or None
    trib = tribunal_de_nombre(o.get("organo"), circ) if o.get("organo") else None
    trib = trib or tribunal(p.get("tribunal"), circ, p.get("tribunal_completo"))
    tipo = tipo_canon(p.get("tipo_asunto")) or tipo_canon(o.get("tipo"))
    mat, sub = materia_canon(p.get("materia"))
    if not mat and o.get("materia"):
        mat, sub = materia_canon(o.get("materia"))
    sen = sentido_canon(p.get("sentido"))
    pro = prospera(tipo, sen)
    que = quejoso(tipo, sen, pro)
    sint = " ".join(str(o.get("sintesis") or "").split())[:240] or None
    if sint and tipo in ("amparo_revision", "revision_incidental", "queja"):
        que = quejoso_de_sintesis(sint) or que
        if pro is None:
            pro = prospera(tipo, decision_de_sintesis(sint))
    f_of, f_le = fecha_iso(o.get("fecha")), fecha_iso(p.get("fecha_sentencia"))
    fecha = f_of or f_le
    pon = ponente_canon(o.get("titular")) or ponente_canon(p.get("magistrado_ponente"))
    llave = p.get("neun") or (re.search(r"_(\d{6,})$", str(p.get("file_id") or "")) or [None, None])[1] \
        or p.get("file_id")
    out = {
        "j_v": VERSION,
        "j_estado": estado_de_circuito(circ),
        "j_tribunal": trib["id"] if trib else None,
        "j_tribunal_nombre": trib["nombre"] if trib else None,
        "j_tipo": tipo,
        "j_materia": mat,
        "j_submateria": sub,
        "j_sentido": sen,
        "j_prospera": pro,
        "j_quejoso": que,
        "j_fecha": f"{fecha}T00:00:00Z" if fecha else None,
        "j_anio": int(fecha[:4]) if fecha else None,
        "j_periodo": periodo(fecha),
        "j_fecha_fuente": "oaj" if f_of else "lectura" if f_le else None,
        "j_ponente": pon,
        "j_sentencia": f"{circ}:{llave}" if llave else None,
        "j_sintesis_oficial": sint,
    }
    if trib and trib.get("sede"):
        out["j_sede"] = trib["sede"]
    return out


# Los índices de Qdrant de estos campos (modo estricto: sin índice no se puede filtrar).
INDICES = {"j_v": "integer", "j_estado": "keyword", "j_tribunal": "keyword", "j_tipo": "keyword",
           "j_materia": "keyword", "j_submateria": "keyword", "j_sentido": "keyword", "j_prospera": "keyword",
           "j_quejoso": "keyword", "j_fecha": "datetime", "j_anio": "integer", "j_periodo": "keyword",
           "j_ponente": "keyword", "j_sentencia": "keyword"}
