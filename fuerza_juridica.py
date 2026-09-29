# -*- coding: utf-8 -*-
"""LA FUERZA JURÍDICA DE UNA AUTORIDAD, EN UN SOLO SITIO (rediseño del taller,
punto 3, 29-sep-2026).

POR QUÉ. David: «La misma fuente puede llegar con distinta fuerza a quien decide
y a quien redacta». `fase6_rag._tesis_de` ponía `obligatoria = bool(vincula)`
—OBLIGATORIA para unas 5,900 jurisprudencias de colegiados y de Plenos de
Circuito que a un colegiado no lo obligan (art. 217, párr. tercero, LA)— y así
llegaba al prompt de la propuesta, del estudio, del plan, de la recalificación
y al documento; la tarjeta y la deliberación, en cambio, usaban
`fuerza_para_colegiado`. Aquí vive esa regla —movida de `tarjeta_decision`,
que la re-exporta— y todo lo demás la lee.

DOS PREGUNTAS SEPARADAS (David): «¿Esta autoridad vincula al tribunal? ¿La
regla que contiene gobierna este supuesto? La primera no resuelve la segunda.»
  · `fuerza_de` / `anotar` contestan la PRIMERA respecto del órgano que
    resuelve (su circuito, su región, su designación) y del régimen temporal
    que se puede leer de la tesis (época, tipo, vigencia aparte).
  · La SEGUNDA no la contesta el código: `aplicabilidad` queda «no_evaluada»
    salvo que alguien la haya medido —la lectura de la deliberación
    («resuelve», «distinguible», «ajena») o, para las sentencias propias de la
    OAJ, el nivel calibrado («mismo_problema», «posible»)—. Que una tesis
    obligue nunca se escribe como que gobierne el caso.

LAS SENTENCIAS DEL PROPIO TRIBUNAL (espejo y OAJ) no son jurisprudencia: no
vinculan; apartarse pide razón. `de_sentencia_propia` les da su rótulo y lleva
su nivel como aplicabilidad medida.
"""
from __future__ import annotations

import re
import unicodedata


def _get(x, k, d=None):
    if isinstance(x, dict):
        return x.get(k, d)
    return getattr(x, k, d)


def _txt(x) -> str:
    return str(x or "").strip()


def _plano(t) -> str:
    t = unicodedata.normalize("NFD", str(t or "").upper())
    return "".join(ch for ch in t if unicodedata.category(ch) != "Mn")


# ═══ LA FUERZA DE UN CRITERIO PARA ESTE TRIBUNAL ═════════════════════════════
#
# POR QUÉ NO SE USA `obligatoria`. `fase6_rag._tesis_de` la calcula como
# `bool(vincula)`, y eso rotula OBLIGATORIA a unas 5,900 jurisprudencias de
# colegiados y de Plenos de Circuito (medido en lector2_potencia, 28-sep-2026).
# El artículo 217, párrafo tercero, de la Ley de Amparo dice lo contrario para
# quien lee la tarjeta: la jurisprudencia de un colegiado obliga a los órganos
# de su circuito CON EXCEPCIÓN de los plenos regionales y de los tribunales
# colegiados. Un juez que aplique «lo que obliga manda» con ese rótulo
# decidiría mal. Nació (28-sep) sólo para la tarjeta y la deliberación; desde
# el rediseño del 29-sep-2026 (David, punto 3: «la misma fuente puede llegar
# con distinta fuerza a quien decide y a quien redacta») es la ÚNICA regla, y
# la leen la consulta, la propuesta, el plan, el estudio y el documento.
#
# LA REGIÓN DE CADA CIRCUITO, MEDIDA (no recordada). En el volcado local del
# Semanario (redactor-sentencias/rag/tesis.jsonl, 26-ago-2026) hay 946 tesis de
# Plenos Regionales; 861 traen la región en la clave («PR.A.C.CN.», «.CS.»).
# Se contó qué circuitos nombra cada una: el Vigésimo Segundo sale 17 veces,
# todas Centro-Norte. El Primero sale en las dos (159 CN, 107 CS: el circuito
# se reparte por materia) y por eso queda sin región: no se afirma.
REGION_DEL_CIRCUITO = {
    2: "CN", 3: "CS", 4: "CN", 5: "CN", 6: "CS", 7: "CS", 8: "CN", 9: "CN",
    10: "CS", 11: "CS", 12: "CN", 13: "CS", 14: "CS", 15: "CN", 16: "CN",
    17: "CN", 18: "CS", 19: "CN", 20: "CS", 21: "CS", 22: "CN", 23: "CN",
    24: "CN", 25: "CN", 26: "CN", 27: "CS", 28: "CN", 29: "CS", 30: "CN",
    31: "CS", 32: "CS",
}
_NOMBRE_REGION = {"CN": "Centro-Norte", "CS": "Centro-Sur"}

_UNIDADES = {"PRIMER": 1, "PRIMERO": 1, "SEGUNDO": 2, "TERCER": 3, "TERCERO": 3,
             "CUARTO": 4, "QUINTO": 5, "SEXTO": 6, "SEPTIMO": 7, "OCTAVO": 8,
             "NOVENO": 9}
_DECENAS = {"DECIMO": 10, "VIGESIMO": 20, "TRIGESIMO": 30}
_ROMANOS = ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))


def _ordinal(palabras: str) -> int:
    """«Vigésimo Segundo» → 22; «Tercer» → 3; 0 si no es un ordinal."""
    n = 0
    for w in _plano(palabras).split():
        if w in _DECENAS:
            n += _DECENAS[w]
        elif w in _UNIDADES:
            n += _UNIDADES[w]
        else:
            return 0
    return n


def _romano(n: int) -> str:
    out = ""
    for v, s in _ROMANOS:
        while n >= v:
            out += s
            n -= v
    return out


_RX_CIRCUITO = re.compile(
    r"\b((?:(?:DECIMO|VIGESIMO|TRIGESIMO)\s+)?(?:PRIMERO?|SEGUNDO|TERCERO?|CUARTO|QUINTO|"
    r"SEXTO|SEPTIMO|OCTAVO|NOVENO)|DECIMO|VIGESIMO|TRIGESIMO)\s+CIRCUITO\b")
_RX_TRIBUNAL = re.compile(
    r"^\s*((?:(?:DECIMO|VIGESIMO)\s+)?(?:PRIMERO?|SEGUNDO|TERCERO?|CUARTO|QUINTO|SEXTO|"
    r"SEPTIMO|OCTAVO|NOVENO)|DECIMO|VIGESIMO)\s+TRIBUNAL\s+COLEGIADO\b")
_LETRA_MATERIA = (("ADMINISTRATIVA", "A"), ("CIVIL", "C"), ("PENAL", "P"),
                  ("TRABAJO", "T"), ("LABORAL", "T"))


def circuito_de(tribunal: str) -> int:
    """El número del circuito de «Tercer Tribunal Colegiado … del Vigésimo
    Segundo Circuito»; 0 si no se lee."""
    m = _RX_CIRCUITO.search(_plano(tribunal))
    return _ordinal(m.group(1)) if m else 0


def designacion_de(tribunal: str) -> str:
    """El prefijo con que el Semanario rotula las tesis de ESTE tribunal:
    «XXII.3o.A.C.» para el Tercer Tribunal Colegiado en Materias
    Administrativa y Civil del Vigésimo Segundo Circuito. "" si no se lee."""
    t = _plano(tribunal)
    circ = circuito_de(tribunal)
    m = _RX_TRIBUNAL.search(t)
    if not circ or not m:
        return ""
    n = _ordinal(m.group(1))
    if not n:
        return ""
    letras = ""
    mm = re.search(r"\bEN\s+MATERIAS?\s+(.+?)\s+DEL?\s", t)
    if mm:
        letras = "".join(f"{l}." for w, l in _LETRA_MATERIA if w in mm.group(1))
    return f"{_romano(circ)}.{n}o.{letras}"


_RX_CLAVE_EN_TEXTO = re.compile(r"\[TESIS:\s*([^\]]+)\]")
# LA CLAVE EN LA LOCALIZACIÓN (de `deliberacion.py`, unificada aquí el
# 28-sep-2026): el payload del taller no guarda `numero_tesis`, pero el
# Semanario a veces la escribe en la localización («…; Tesis: PR.A.C.CN. J/7 K
# (12a.)»). Tres formas: la del Pleno Regional, la de un colegiado con su
# circuito en romanos y la de la Corte («1a./J. 5/2020»).
# LA JURISPRUDENCIA DE UN COLEGIADO («XXII.3o.A.C. J/2 K (11a.)») lleva la «J/»
# DESPUÉS de la designación y separada por un espacio; el patrón sólo admitía
# la aislada («XXII.3o.A.C.5 K»), así que la jurisprudencia del propio tribunal
# no se reconocía y salía «orienta» (revisión del 28-sep-2026, AR 631/2025).
_RX_CLAVE_EN_LOCALIZACION = re.compile(
    r"(PR\.(?:[A-Z]{1,3}\.)+\s*J/\S+(?:\s+[A-Z]{1,2})?\s*\(\d{1,2}a\.\)"
    r"|\b(?:[IVXL]+|\([IVX]+\s+Regi[oó]n\))\.\S+(?:\s+J/\S+)?\s+[A-Z]{1,2}\s*\(\d{1,2}a\.\)"
    r"|\b\d?[a-zA-Z]{0,2}\./J\.\s*\d+/\d{2,4}(?:\s*\(\d{1,2}a\.\))?)")


def clave_de_tesis(t: dict) -> str:
    """La clave de la tesis («1a./J. 67/2014 (10a.)», «PR.A.C.CN. J/7 K»), de
    donde esté: un campo propio, el encabezado del texto («[TESIS: …]») o la
    localización. El material del taller casi nunca la trae: `_tesis_de` no la
    guarda (ver las dudas de `fuerza_para_colegiado`). «» si no consta: nunca
    se inventa."""
    for k in ("clave", "clave_tesis", "numero_tesis", "tesis"):
        v = _txt(_get(t, k))
        if v:
            return v
    m = _RX_CLAVE_EN_TEXTO.search(str(_get(t, "texto") or "")[:400])
    if m:
        return m.group(1).strip()
    m = _RX_CLAVE_EN_LOCALIZACION.search(str(_get(t, "localizacion") or ""))
    return m.group(1).strip() if m else ""


def region_de_clave(clave: str) -> str:
    """«PR.A.C.CN. J/7 K (12a.)» → «CN». La región va en el último tramo de la
    sigla del Pleno Regional. «» si la clave no es de un Pleno Regional."""
    m = re.match(r"\s*PR\.((?:[A-Z]{1,3}\.)+)", str(clave or "").upper())
    if not m:
        return ""
    tramos = [x for x in m.group(1).split(".") if x]
    return tramos[-1] if tramos and tramos[-1] in _NOMBRE_REGION else ""


def _tipo(t: dict) -> str:
    """«jurisprudencia» | «aislada» | «precedente» | "". La clave también lo
    dice: «/J.» o «J/» es jurisprudencia aunque el campo `tipo` falte."""
    x = _plano(_get(t, "tipo"))
    loc = _plano(_get(t, "localizacion"))
    clave = _plano(clave_de_tesis(t)).replace(" ", "")
    if "PRECEDENTE" in x:
        return "precedente"
    if "JURISPRUDENCIA" in x or loc.startswith("[J]") or "/J." in clave or "J/" in clave:
        return "jurisprudencia"
    if "AISLADA" in x or loc.startswith("[TA]"):
        return "aislada"
    return ""


def _organo(t: dict) -> str:
    """«scjn» | «pleno_regional» | «pleno_circuito» | «colegiado» | ""."""
    inst = _plano(_get(t, "instancia"))
    loc = _plano(_get(t, "localizacion"))
    clave = _plano(clave_de_tesis(t))
    # EL ORDEN IMPORTA: «PLENOS DE CIRCUITO» y «PLENOS REGIONALES» contienen
    # «PLENO», y el Pleno a secas es el de la Corte.
    if "REGIONAL" in inst or clave.startswith("PR."):
        return "pleno_regional"
    if ("PLENOS DE CIRCUITO" in inst or "PLENO DE CIRCUITO" in inst or "PLENO EN MATERIA" in inst
            or clave.startswith("PC.")):
        return "pleno_circuito"
    if ("SALA" in inst or inst == "PLENO" or "SUPREMA CORTE" in inst
            or re.search(r";\s*(PLENO|1A\. SALA|2A\. SALA|3A\. SALA|4A\. SALA)\s*;", loc)):
        return "scjn"
    if "COLEGIAD" in inst or inst.startswith("TRIBUNALES") or "T.C.C." in loc:
        return "colegiado"
    return ""


def _region_de_tesis(t: dict) -> str:
    r = region_de_clave(clave_de_tesis(t))
    if r:
        return r
    c = _plano(clave_de_tesis(t)) + " " + _plano(_get(t, "instancia"))
    if ".CN." in c or "CENTRO-NORTE" in c or "CENTRO NORTE" in c:
        return "CN"
    if ".CS." in c or "CENTRO-SUR" in c or "CENTRO SUR" in c:
        return "CS"
    return ""


def _regiones_configuradas() -> dict:
    """{circuito: región} de `DELIBERACION_REGIONES` («22:CN,1:CS»): la
    corrección a mano de la tabla medida, si algún día el Consejo mueve un
    circuito de región. Vacío por omisión. Se lee en cada llamada."""
    import os
    out = {}
    for par in (os.getenv("DELIBERACION_REGIONES", "") or "").split(","):
        if ":" in par:
            c, r = par.split(":", 1)
            try:
                n = int(c.strip())
            except ValueError:
                continue
            if r.strip().upper() in _NOMBRE_REGION:
                out[n] = r.strip().upper()
    return out


def region_del_circuito(circuito: int) -> str:
    """La región del Pleno Regional de un circuito: la configurada a mano
    (`DELIBERACION_REGIONES`) o la medida (`REGION_DEL_CIRCUITO`); «» si no
    se sabe."""
    if not circuito:
        return ""
    return _regiones_configuradas().get(circuito) or REGION_DEL_CIRCUITO.get(circuito, "")


# LOS RÓTULOS DEL PROPIO TRIBUNAL (revisión del 28-sep-2026, AR 631/2025). El
# artículo 228 de la Ley de Amparo habla de las propias JURISPRUDENCIAS: el
# tribunal queda vinculado y, para apartarse, tiene que dar argumentos
# suficientes. Estaba al revés: la jurisprudencia propia no se reconocía (salía
# «orienta») y la aislada propia y las sentencias del espejo llevaban el 228.
FUERZA_JURISPRUDENCIA_PROPIA = ("jurisprudencia propia: vincula a este tribunal; apartarse "
                                "exige argumentos suficientes (art. 228)")
FUERZA_AISLADA_PROPIA = "precedente propio (tesis aislada): no es jurisprudencia; apartarse pide razón"
FUERZA_SENTENCIA_PROPIA = ("precedente propio (sentencia de este tribunal): no es jurisprudencia "
                           "ni voto; apartarse pide razón")


def fuerza_para_colegiado(tesis: dict, tribunal: str = "", circuito: int = 0, *,
                          region: str | None = None, clave_propia: str = "") -> dict:
    """La fuerza de un criterio ANTE UN TRIBUNAL COLEGIADO, calculada por
    código: {"fuerza": "obliga"|"orienta"|"pleno_circuito"|"precedente_propio",
    "fuerza_texto": "…"} y, si depende de un dato que no consta, "por_confirmar":
    True. `tribunal`: el nombre del tribunal que resuelve (de ahí salen su
    circuito, su región y su designación); `circuito` lo fuerza; `region`
    («CN» | «CS») fuerza la del circuito; `clave_propia` («XXII.3o.A.C.») la
    designación del propio tribunal.

    UNA SOLA DEFINICIÓN (integración del 28-sep-2026, AR 631/2025): la tarjeta
    y la deliberación escribieron cada una la suya en paralelo. Ésta es la
    única; `deliberacion.py` la importa. De la de la deliberación se quedan la
    clave leída de la localización, el Pleno «en materia» como Pleno de
    Circuito, la región y la clave propia explícitas, la corrección por
    `DELIBERACION_REGIONES` y la marca `por_confirmar`; de la de la tarjeta, la
    región MEDIDA de cada circuito, la designación leída del nombre del
    tribunal y el cuidado de no confundir «XXII.3o.A.C.» con «XXII.3o.A.C.P.».

    LA REGLA (SPEC C, 28-sep-2026):
      · Suprema Corte, Pleno o Salas: jurisprudencia o precedente obligatorio
        (arts. 217, párr. primero, 222 y 223 LA) → «obliga»; aislada → «orienta».
      · Pleno Regional DE LA REGIÓN DEL CIRCUITO, jurisprudencia (art. 217,
        párr. segundo) → «obliga»; de otra región o aislada → «orienta».
      · Pleno de Circuito (extinto) → «pleno_circuito», rótulo propio, SIN
        afirmar si obliga.
      · Jurisprudencia de colegiado → «orienta (art. 217, párr. tercero)».
      · Aislada → «orienta».
      · Tesis del propio tribunal → «precedente_propio». El artículo 228 sólo
        se rotula en su JURISPRUDENCIA: la vincula y apartarse exige
        argumentos suficientes. La aislada propia no es jurisprudencia y no se
        le pone ese artículo (revisión del 28-sep-2026, AR 631/2025).

    DUDAS PARA DAVID (se dejan escritas porque cambian el rótulo, no el orden):
      1. PLENOS DE CIRCUITO. Se extinguieron con la reforma de 2021; si su
         jurisprudencia sigue obligando a los colegiados de su circuito
         mientras no la sustituya un Pleno Regional es cosa de los
         transitorios. Aquí no se afirma: sale «Pleno de Circuito».
      2. LA REGIÓN DE UNA TESIS DE PLENO REGIONAL casi nunca se puede leer del
         material: `_tesis_de` no guarda la clave («PR.A.C.CN. J/7 K») y la
         instancia dice «Plenos Regionales» a secas. Sin la región, la
         jurisprudencia de un Pleno Regional sale «orienta» con la nota de que
         obligaría si es de la región: se prefiere quedarse corto a rotular
         «obliga» lo que quizá no obliga. Guardar la clave en `_tesis_de`
         cerraría esta duda.
      3. EL PRECEDENTE PROPIO también necesita la clave (el prefijo
         «XXII.3o.A.C.»); sin ella, una tesis del propio tribunal sale como
         cualquier otra de colegiado. Los precedentes propios que sí llegan
         son las filas del espejo (`tu_tribunal`), que no son tesis.
      4. LA JURISPRUDENCIA DE LA CORTE ANTERIOR A 2013 se rotula «obliga»
         (sexto transitorio de la Ley de Amparo: sigue en vigor en lo que no
         se oponga). Si perdió vigencia lo dice el sello, no este rótulo.
      5. LA AISLADA DE LA CORTE DE LA UNDÉCIMA ÉPOCA que en realidad es
         precedente obligatorio (arts. 222 y 223) se reconoce sólo si el
         Semanario la publicó como jurisprudencia: se lee del `tipo`.
    """
    tesis = tesis if isinstance(tesis, dict) else {}
    organo = _organo(tesis)
    tipo = _tipo(tesis)
    obligatorio = tipo in ("jurisprudencia", "precedente")
    circ = int(circuito or 0) or circuito_de(tribunal)
    clave = _plano(clave_de_tesis(tesis)).replace(" ", "")
    desig = _plano(clave_propia or (designacion_de(tribunal) if tribunal else "")).replace(" ", "")

    if organo in ("colegiado", "") and desig and clave.startswith(desig):
        # «XXII.3o.» no puede casar con «XXII.3o.A.C.» de OTRO tribunal ni
        # «XXII.3o.A.C.» con «XXII.3o.A.C.P.»: tras la designación viene un
        # número, la «J/» o el fin.
        resto = clave[len(desig):]
        if not resto or not resto[0].isalpha() or resto.startswith("J/"):
            if tipo == "jurisprudencia":
                return {"fuerza": "precedente_propio", "fuerza_texto": FUERZA_JURISPRUDENCIA_PROPIA}
            return {"fuerza": "precedente_propio", "fuerza_texto": FUERZA_AISLADA_PROPIA}
    if organo == "scjn":
        if obligatorio:
            return {"fuerza": "obliga",
                    "fuerza_texto": "obliga (art. 217, párr. primero"
                                    + ("; arts. 222 y 223" if tipo == "precedente" else "") + ")"}
        return {"fuerza": "orienta", "fuerza_texto": "orienta: aislada de la Suprema Corte"}
    if organo == "pleno_regional":
        if not obligatorio:
            return {"fuerza": "orienta", "fuerza_texto": "orienta: aislada de un Pleno Regional"}
        reg_t = _region_de_tesis(tesis)
        reg_c = (region or "").strip().upper() or region_del_circuito(circ)
        if reg_t and reg_c and reg_t == reg_c:
            return {"fuerza": "obliga",
                    "fuerza_texto": f"obliga: Pleno Regional {_NOMBRE_REGION.get(reg_c, reg_c)}, "
                                    f"región de este circuito (art. 217, párr. segundo)"}
        if reg_t and reg_c:
            return {"fuerza": "orienta",
                    "fuerza_texto": f"orienta: Pleno Regional de otra región "
                                    f"({_NOMBRE_REGION.get(reg_t, reg_t)})"}
        # SIN LA REGIÓN DE LA TESIS O SIN LA DEL CIRCUITO no se afirma que
        # obligue: se queda corto y lo dice (`por_confirmar`, que la
        # deliberación lee para no dejar que el juez decida por «lo que obliga»).
        return {"fuerza": "orienta", "por_confirmar": True,
                "fuerza_texto": "orienta: Pleno Regional sin región identificada; si es de la "
                                "región de este circuito, obliga (art. 217, párr. segundo)"}
    if organo == "pleno_circuito":
        return {"fuerza": "pleno_circuito", "fuerza_texto": "Pleno de Circuito"}
    if organo == "colegiado" and obligatorio:
        return {"fuerza": "orienta", "fuerza_texto": "orienta (art. 217, párr. tercero)"}
    return {"fuerza": "orienta", "fuerza_texto": "orienta"}




# ═══ LA API COMÚN (rediseño, punto 3) ════════════════════════════════════════
#
# Todo lo que decide o imprime la fuerza de una autoridad pasa por aquí:
#   · `fuerza_de(t, tribunal)`   → la regla de arriba + `vincula` (True | False |
#                                  None = no se puede afirmar);
#   · `anotar(tesis, tribunal)`  → lo escribe en cada tesis, idempotente;
#   · `rotulo(t)`                → la frase para un prompt o un documento;
#   · `orden(t)`                 → la clave para ordenar: lo que vincula primero;
#   · `de_sentencia_propia(f)`   → una fila del espejo del propio tribunal.
#
# `obligatoria` SE CONSERVA como campo, porque lo leen el orden de la búsqueda,
# el plan, la recalificación y el documento; pero ya no significa «jurisprudencia
# en general» (`vincula` del Semanario, que ahora se guarda en `vincula_origen`)
# sino «VINCULA A ESTE TRIBUNAL». Sin tribunal conocido se calcula igual para
# cualquier colegiado: la de colegiado orienta, la de la Corte obliga, la de un
# Pleno Regional sin región queda por confirmar y NO se rotula obligatoria.

_RX_EPOCA = re.compile(r"\b(\d{1,2})a\.?\s*[ÉE]poca", re.I)


def epoca_de(t: dict) -> int:
    """La época del Semanario leída de la localización («11a. Época»); 0 si no
    consta. Es el régimen temporal que la tesis dice de sí misma."""
    m = _RX_EPOCA.search(_txt(_get(t, "localizacion")))
    return int(m.group(1)) if m else 0


def vincula_de(f: dict):
    """True si el criterio vincula al tribunal, False si sólo orienta, None si
    no se puede afirmar (Pleno de Circuito, región por confirmar)."""
    if not isinstance(f, dict):
        return None
    if f.get("por_confirmar") or f.get("fuerza") == "pleno_circuito":
        return None
    if f.get("fuerza") == "obliga":
        return True
    if f.get("fuerza") == "precedente_propio" and f.get("fuerza_texto") == FUERZA_JURISPRUDENCIA_PROPIA:
        return True
    return False


def fuerza_de(t: dict, tribunal: str = "", circuito: int = 0, *,
              region: str | None = None, clave_propia: str = "") -> dict:
    """{fuerza, fuerza_texto, por_confirmar?, vincula} de una TESIS para el
    tribunal que resuelve. Sin tribunal, la regla general de un colegiado."""
    f = dict(fuerza_para_colegiado(t, tribunal, circuito, region=region,
                                   clave_propia=clave_propia))
    f["vincula"] = vincula_de(f)
    return f


def anotar(tesis, tribunal: str = "", circuito: int = 0, *,
           region: str | None = None, clave_propia: str = "") -> list:
    """Escribe la fuerza en cada tesis de la lista (en su sitio) y la devuelve.

    Idempotente: si ya se anotó para ESTE tribunal no se recalcula; si se anotó
    sin tribunal y ahora se sabe cuál es, se recalcula (la jurisprudencia
    propia y la región sólo se conocen con él). `aplicabilidad` —la segunda
    pregunta— se crea «no_evaluada» y nunca se pisa si alguien ya la midió.
    """
    para = _txt(tribunal) or _txt(clave_propia)
    for t in (tesis or []):
        if not isinstance(t, dict):
            continue
        if t.get("_fuerza_para") == para and "fuerza" in t and "vincula_al_tribunal" in t:
            continue
        if "vincula_origen" not in t:
            # El dato crudo del Semanario («es jurisprudencia»), que antes se
            # llamaba `obligatoria`; se conserva para quien lo necesite.
            t["vincula_origen"] = t.get("vincula") if "vincula" in t else t.get("obligatoria")
        f = fuerza_de(t, tribunal, circuito, region=region, clave_propia=clave_propia)
        t["fuerza"] = f["fuerza"]
        t["fuerza_texto"] = f["fuerza_texto"]
        t["por_confirmar"] = bool(f.get("por_confirmar"))
        t["vincula_al_tribunal"] = f["vincula"]
        t["obligatoria"] = f["vincula"] is True
        t["_fuerza_para"] = para
        if not isinstance(t.get("aplicabilidad"), dict):
            t["aplicabilidad"] = {"estado": "no_evaluada"}
    return tesis


def _leida(t: dict) -> tuple:
    """(vincula, fuerza_texto) de una tesis SIN tocarla: la anotada, o la
    calculada sin tribunal si nadie la anotó (un material guardado antes del
    29-sep trae `obligatoria = vincula` del Semanario y no se le cree)."""
    if "vincula_al_tribunal" in t:
        return t.get("vincula_al_tribunal"), _txt(t.get("fuerza_texto"))
    f = fuerza_de(t)
    return f["vincula"], _txt(f["fuerza_texto"])


def vincula(t: dict):
    """True | False | None: ¿vincula al tribunal? Sin modificar la tesis."""
    return _leida(t)[0] if isinstance(t, dict) else False


def rotulo(t: dict) -> str:
    """La frase de la fuerza para un prompt o un documento. Si la tesis no se
    anotó, se calcula sin tribunal (nunca se lee `vincula` a pelo)."""
    if not isinstance(t, dict):
        return "orientadora"
    v, txt = _leida(t)
    if v is True:
        return f"OBLIGATORIA para este tribunal — {txt}"
    if v is None:
        return f"fuerza POR CONFIRMAR — {txt}"
    return f"orientadora — {txt}"


def orden(t: dict) -> int:
    """0 si vincula, 1 si está por confirmar, 2 si orienta."""
    if not isinstance(t, dict):
        return 2
    v = _leida(t)[0]
    return 0 if v is True else (1 if v is None else 2)


def de_sentencia_propia(fila: dict) -> dict:
    """Una sentencia del propio tribunal (espejo viejo u OAJ): no es
    jurisprudencia, no vincula, apartarse pide razón. La segunda pregunta, si
    se midió, es el nivel calibrado de la OAJ («mismo_problema» ≥85%,
    «posible» 50-84%); el espejo viejo no la mide."""
    fila = fila if isinstance(fila, dict) else {}
    nivel = _txt(fila.get("nivel"))
    aplic = ({"estado": nivel, "fuente": "oaj_calibrada", "similitud": fila.get("similitud"),
              "cota_inferior": bool(fila.get("cota_inferior"))}
             if nivel in ("mismo_problema", "posible") else {"estado": "no_evaluada"})
    return {"fuerza": "precedente_propio", "fuerza_texto": FUERZA_SENTENCIA_PROPIA,
            "vincula": False, "aplicabilidad": aplic}
