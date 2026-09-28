"""LA TARJETA «EL PROBLEMA PRINCIPAL Y SU SOLUCIÓN» — sin modelo, 0 USD.

POR QUÉ EXISTE. AR 631/2025, 28-sep-2026. David, al ver el retroceso del
redactor: «lo más importante es plantearle al secretario cuál es el problema
principal y cuáles son los secundarios. Y preguntarle cómo resolverías tú este
problema jurídico. Yo te propongo aplicar esta interpretación o esta
jurisprudencia (…) ¿o quieres resolver en sentido opuesto? dándole la
alternativa de solución sustentada también (…) y nada más un botón que me
permita ir a resolver con mi criterio».

LO QUE YA HABÍA Y NADIE ENSEÑABA. La propuesta (fase 5, formato 2) ya escribe
para el asunto entero la vía del motor y la contraria —`global.alternativa`,
con sentido, razón, efecto y apoyos—, por qué el principal es el principal
(`global.contexto.tema_principal`), por dónde se cae (`en_contra`), la vía
protectora, las constancias y, por cada accesorio, su suerte condicional en
las dos vías (`checklist`). La pantalla enseñaba un CONTADOR de apoyos y
escondía la vía contraria detrás de «Ver por qué». Aquí se arma la tarjeta con
eso, sin pedirle nada a ningún modelo:

  · el principal, con la jerarquía de la fase 3 (o la del secretario) y el
    aviso si el motor tomó otro problema como el que decide;
  · las dos vías con el MISMO peso —«te propongo» y «¿o al revés?»—, cada una
    con su desenlace CALCULADO POR CÓDIGO (tipos_asunto / fase_rama: en el 631
    la vía contraria revoca una concesión y eso obliga a reasumir jurisdicción
    y estudiar los conceptos que el juez no estudió, art. 93, fr. VI) y sus
    apoyos HIDRATADOS contra el acervo del asunto: registro, rubro literal,
    instancia, vigencia y la FUERZA para ESTE tribunal (`fuerza_para_colegiado`);
  · los secundarios resueltos solos en cada vía por el MISMO árbol que decide
    al generar (`arbol_decision.reparto_para_pantalla`, dos pasadas);
  · un estado «claro | reñido | no_alcanza» con sus razones, determinista y sin
    porcentajes.

LO QUE NO HACE, Y POR QUÉ.
  · NO DECIDE. El motor acierta 50% contra 24 engroses reales (banco Kingston)
    y «siempre niega» acierta 54%; su «confianza alta» no predijo el acierto
    (4 de 8, 2 de 6). La tarjeta no enseña la confianza del modelo ni un
    porcentaje, y sólo rotula «recomendada» cuando las señales deterministas lo
    sostienen (ver `_estado`).
  · NO CITA LO QUE NO CONSTA. Un registro que no está en el acervo del asunto
    no se pinta como cita (va a `avisos`); una tesis abandonada o sustituida no
    entra en «te propongo aplicar»; las pistas de internet nunca se citan.
  · NO VOTA. El espejo del propio tribunal y la jurimetría se enseñan como
    precedentes, no se cuentan (fase_espejo.py, «lo que esto no es»).
  · NO IMPORTA main NI TOCA LA BASE. El endpoint (`GET /taller/tarjeta`) pone
    las lecturas y la marca «tarjeta» alrededor; aquí todo es puro y se prueba
    sin el arranque de la app (el arranque borra las cachés de Gemini).

LA DELIBERACIÓN (SPEC C2, detrás de bandera). Si la marca «deliberacion»
existe, sus dos vías se proyectan sobre `vias.propuesta` (la que recomienda el
juez) y `vias.opuesta` (la otra), cada una con su cadena; los apoyos se
vuelven a verificar aquí y la fuerza se recalcula por código. Sin ella, todo
sale de la propuesta formato 2. Ver `_payload_deliberacion` para la forma que
se lee (tolerante: el módulo `deliberacion.py` se escribe en paralelo).

CONTRATO: contrato_tarjeta.md (formato 1). Todos los campos pueden faltar o
venir null: la pantalla los tolera.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import re
import unicodedata

FORMATO = 1

ESTADOS = ("claro", "reñido", "no_alcanza")

# Lo que la pantalla pone en el hueco de los resolutivos cuando el desenlace
# depende de un estudio que aún no se hace: el hueco a la vista, como en los
# adelantos de David («Se ********** la sentencia recurrida»).
HUECO = "**********"

# Una tesis que PERDIÓ su vigencia por completo no entra en «te propongo
# aplicar» (vigencia_tesis: 558 con pérdida expresa). Es la misma frontera de
# `vigencia_tesis.linea_visible`: todo lo que no es corrección de texto
# «perdió vigencia», también la MODIFICADA (48 en el índice). La aclarada y la
# de texto sustituido no cambian el criterio —cambian el texto que se cita— y
# la que la perdió EN PARTE puede seguir sirviendo en la otra parte: ésas se
# enseñan con su etiqueta, no se quitan.
_PERDIO_VIGENCIA = frozenset(("abandonada", "interrumpida", "sustituida",
                              "superada", "sin_efectos", "modificada"))

_RX_REGISTRO = re.compile(r"\b(\d{6,7})\b")


# ═══ UTILIDADES ══════════════════════════════════════════════════════════════

def _get(x, k, d=None):
    if isinstance(x, dict):
        return x.get(k, d)
    return getattr(x, k, d)


def _txt(x) -> str:
    return str(x or "").strip()


def _plano(t) -> str:
    t = unicodedata.normalize("NFD", str(t or "").upper())
    return "".join(ch for ch in t if unicodedata.category(ch) != "Mn")


def _preg(p) -> str:
    return _txt(p.get("pregunta") if isinstance(p, dict) else p)


def _prospera(s: str) -> bool:
    try:
        import tipos_asunto as _ta
        return _ta.prospera(s or "")
    except Exception:                                   # pragma: no cover
        s = (s or "").lower()
        return "fundad" in s and "insuficien" not in s


def _clave_problema(t) -> str:
    try:
        import arbol_decision as _ad
        return _ad.clave_problema(t)
    except Exception:                                   # pragma: no cover
        return str(t or "")[:400]


def _mismo_tema(a: str, b: str) -> bool:
    """¿El mismo problema, escrito distinto? La reserva de la fase 5 (medida en
    la revisión 410/2026); si no carga, sólo la igualdad del recorte."""
    if _clave_problema(a).strip().lower() == _clave_problema(b).strip().lower():
        return True
    try:
        import fase5_propuesta as _f5
        return bool(_f5._mismo_tema(a, b))
    except Exception:                                   # pragma: no cover
        return False


def _lista(material, campo: str) -> list:
    """`material.tesis` / `material["tesis"]`: el Material de la sesión o su
    forma guardada en la fila, lo que llegue."""
    xs = _get(material, campo, None) if material is not None else None
    return [x for x in (xs or []) if isinstance(x, dict)]


def _limpio(v):
    """Ida y vuelta por JSON: la tarjeta va a la fila y a la pantalla."""
    try:
        return json.loads(json.dumps(v, ensure_ascii=False, default=str))
    except Exception:                                   # pragma: no cover
        return None


# ═══ LA FUERZA DE UN CRITERIO PARA ESTE TRIBUNAL ═════════════════════════════
#
# POR QUÉ NO SE USA `obligatoria`. `fase6_rag._tesis_de` la calcula como
# `bool(vincula)`, y eso rotula OBLIGATORIA a unas 5,900 jurisprudencias de
# colegiados y de Plenos de Circuito (medido en lector2_potencia, 28-sep-2026).
# El artículo 217, párrafo tercero, de la Ley de Amparo dice lo contrario para
# quien lee la tarjeta: la jurisprudencia de un colegiado obliga a los órganos
# de su circuito CON EXCEPCIÓN de los plenos regionales y de los tribunales
# colegiados. Un juez que aplique «lo que obliga manda» con ese rótulo
# decidiría mal. Aquí la fuerza se calcula por código y SÓLO para la tarjeta y
# la deliberación: el rótulo del estudio no cambia (es decisión de David).
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


def clave_de_tesis(t: dict) -> str:
    """La clave de la tesis («1a./J. 67/2014 (10a.)», «PR.A.C.CN. J/7 K»), de
    donde esté. El material del taller casi nunca la trae: `_tesis_de` no la
    guarda (ver las dudas de `fuerza_para_colegiado`)."""
    for k in ("clave", "clave_tesis", "numero_tesis", "tesis"):
        v = _txt(_get(t, k))
        if v:
            return v
    m = _RX_CLAVE_EN_TEXTO.search(str(_get(t, "texto") or "")[:400])
    return m.group(1).strip() if m else ""


def _tipo(t: dict) -> str:
    """«jurisprudencia» | «aislada» | «precedente» | ""."""
    x = _plano(_get(t, "tipo"))
    loc = _plano(_get(t, "localizacion"))
    if "PRECEDENTE" in x:
        return "precedente"
    if "JURISPRUDENCIA" in x or loc.startswith("[J]"):
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
    if "PLENOS DE CIRCUITO" in inst or "PLENO DE CIRCUITO" in inst or clave.startswith("PC."):
        return "pleno_circuito"
    if ("SALA" in inst or inst == "PLENO" or "SUPREMA CORTE" in inst
            or re.search(r";\s*(PLENO|1A\. SALA|2A\. SALA|3A\. SALA|4A\. SALA)\s*;", loc)):
        return "scjn"
    if "COLEGIAD" in inst or inst.startswith("TRIBUNALES") or "T.C.C." in loc:
        return "colegiado"
    return ""


def _region_de_tesis(t: dict) -> str:
    c = _plano(clave_de_tesis(t)) + " " + _plano(_get(t, "instancia"))
    if ".CN." in c or "CENTRO-NORTE" in c or "CENTRO NORTE" in c:
        return "CN"
    if ".CS." in c or "CENTRO-SUR" in c or "CENTRO SUR" in c:
        return "CS"
    return ""


def fuerza_para_colegiado(tesis: dict, tribunal: str = "", circuito: int = 0) -> dict:
    """La fuerza de un criterio ANTE UN TRIBUNAL COLEGIADO, calculada por
    código: {"fuerza": "obliga"|"orienta"|"pleno_circuito"|"precedente_propio",
    "fuerza_texto": "…"}. `tribunal`: el nombre del tribunal que resuelve (de
    ahí salen su circuito, su región y su designación); `circuito` lo fuerza.

    LA REGLA (SPEC C, 28-sep-2026):
      · Suprema Corte, Pleno o Salas: jurisprudencia o precedente obligatorio
        (arts. 217, párr. primero, 222 y 223 LA) → «obliga»; aislada → «orienta».
      · Pleno Regional DE LA REGIÓN DEL CIRCUITO, jurisprudencia (art. 217,
        párr. segundo) → «obliga»; de otra región o aislada → «orienta».
      · Pleno de Circuito (extinto) → «pleno_circuito», rótulo propio, SIN
        afirmar si obliga.
      · Jurisprudencia de colegiado → «orienta (art. 217, párr. tercero)».
      · Aislada → «orienta».
      · Tesis del propio tribunal → «precedente propio (art. 228)».

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
    organo = _organo(tesis)
    tipo = _tipo(tesis)
    obligatorio = tipo in ("jurisprudencia", "precedente")
    circ = circuito or circuito_de(tribunal)
    clave = _plano(clave_de_tesis(tesis)).replace(" ", "")
    desig = _plano(designacion_de(tribunal)).replace(" ", "") if tribunal else ""

    if organo == "colegiado" and desig and clave.startswith(desig):
        # «XXII.3o.» no puede casar con «XXII.3o.A.C.» de OTRO tribunal ni
        # «XXII.3o.A.C.» con «XXII.3o.A.C.P.»: tras la designación viene un
        # número, la «J/» o el fin.
        resto = clave[len(desig):]
        if not resto or not resto[0].isalpha() or resto.startswith("J/"):
            return {"fuerza": "precedente_propio",
                    "fuerza_texto": "precedente propio (art. 228)"}
    if organo == "scjn":
        if obligatorio:
            return {"fuerza": "obliga",
                    "fuerza_texto": "obliga (art. 217, párr. primero)"}
        return {"fuerza": "orienta", "fuerza_texto": "orienta: aislada de la Suprema Corte"}
    if organo == "pleno_regional":
        if not obligatorio:
            return {"fuerza": "orienta", "fuerza_texto": "orienta: aislada de un Pleno Regional"}
        reg_t = _region_de_tesis(tesis)
        reg_c = REGION_DEL_CIRCUITO.get(circ, "")
        if reg_t and reg_c and reg_t == reg_c:
            return {"fuerza": "obliga",
                    "fuerza_texto": f"obliga: Pleno Regional {_NOMBRE_REGION[reg_c]}, "
                                    f"región de este circuito (art. 217, párr. segundo)"}
        if reg_t and reg_c:
            return {"fuerza": "orienta",
                    "fuerza_texto": f"orienta: Pleno Regional de otra región "
                                    f"({_NOMBRE_REGION.get(reg_t, reg_t)})"}
        return {"fuerza": "orienta",
                "fuerza_texto": "orienta: Pleno Regional sin región identificada; si es de la "
                                "región de este circuito, obliga (art. 217, párr. segundo)"}
    if organo == "pleno_circuito":
        return {"fuerza": "pleno_circuito", "fuerza_texto": "Pleno de Circuito"}
    if organo == "colegiado" and obligatorio:
        return {"fuerza": "orienta", "fuerza_texto": "orienta (art. 217, párr. tercero)"}
    return {"fuerza": "orienta", "fuerza_texto": "orienta"}


# ═══ LOS APOYOS, HIDRATADOS Y VERIFICADOS ═══════════════════════════════════

def _vigencia(t: dict) -> dict | None:
    """El sello guardado con la tesis o, si no lo trae (filas anteriores al
    25-sep-2026 lo traen null), el del índice local."""
    v = _get(t, "vigencia")
    if isinstance(v, dict) and v.get("estado"):
        return v
    try:
        import vigencia_tesis as _vt
        return _vt.de(_get(t, "registro"))
    except Exception:                                   # pragma: no cover
        return None


def _etiqueta_vigencia(v) -> str | None:
    if not v:
        return None
    try:
        import vigencia_tesis as _vt
        return _vt.etiqueta(v) or None
    except Exception:                                   # pragma: no cover
        return str(v.get("estado") or "") or None


def perdio_vigencia(t: dict) -> bool:
    v = _vigencia(t)
    return bool(v and str(v.get("estado") or "") in _PERDIO_VIGENCIA and not v.get("parcial"))


def catalogo_de(material) -> dict:
    """{registro: tesis} del acervo del asunto: el catálogo cerrado contra el
    que se verifica cada cita."""
    out = {}
    for t in _lista(material, "tesis"):
        reg = _txt(t.get("registro"))
        if reg and reg not in out:
            out[reg] = t
    return out


def apoyo_de_tesis(t: dict, tribunal: str = "", circuito: int = 0) -> dict:
    """Un APOYO del contrato a partir de una tesis DEL ACERVO. El rubro es el
    literal del acervo, nunca el que escriba un modelo."""
    f = fuerza_para_colegiado(t, tribunal, circuito)
    return {"registro": _txt(t.get("registro")), "rubro": _txt(t.get("rubro")),
            "instancia": _txt(t.get("instancia")), "tipo": _tipo(t) or None,
            "fuerza": f["fuerza"], "fuerza_texto": f["fuerza_texto"],
            "vigencia": _etiqueta_vigencia(_vigencia(t)),
            "de_internet": bool(t.get("de_internet")), "en_acervo": True, "norma": None}


def _apoyo_norma(texto: str) -> dict:
    return {"registro": None, "rubro": None, "instancia": None, "tipo": None,
            "fuerza": None, "fuerza_texto": None, "vigencia": None,
            "de_internet": False, "en_acervo": None, "norma": texto}


def hidratar_apoyos(apoyos, catalogo: dict, tribunal: str = "", circuito: int = 0,
                    via: str = "") -> tuple:
    """(apoyos del contrato, avisos). Cada apoyo del motor —«2026918», «el
    registro 170378», «art. 49 CPC Qro», o {"registro": …} de la
    deliberación— se verifica contra el catálogo del asunto:
      · con registro y en el acervo → la tesis, con rubro literal, fuerza y
        vigencia; si perdió la vigencia por completo, NO entra (va al aviso);
      · con registro que no está en el acervo → NO se pinta (va al aviso);
      · sin registro → una norma.
    """
    fuera, avisos, vistos = [], [], set()
    _de = f" de {via}" if via else ""
    for a in (apoyos or []):
        if isinstance(a, dict):
            reg = _txt(a.get("registro"))
            m = _RX_REGISTRO.search(reg)
            reg = m.group(1) if m else ""
            txt = reg or _txt(a.get("norma") or a.get("precepto"))
            if not txt and _txt(a.get("id")):
                # Un identificador del catálogo de la deliberación sin su
                # registro no se puede comprobar aquí: no se pinta.
                avisos.append(f"Un apoyo{_de} llegó sólo con su identificador "
                              f"({_txt(a.get('id'))}), sin registro: no se pinta como cita.")
                continue
        else:
            txt = _txt(a)
            m = _RX_REGISTRO.search(txt)
            reg = m.group(1) if m else ""
        if not txt:
            continue
        if reg:
            if reg in vistos:
                continue
            vistos.add(reg)
            t = catalogo.get(reg)
            if t is None:
                avisos.append(f"El registro {reg}{_de} no está en el acervo del asunto: no se "
                              f"propone aplicarlo hasta comprobarlo en el Semanario.")
                continue
            if perdio_vigencia(t):
                avisos.append(f"El registro {reg}{_de} ({_txt(t.get('rubro'))[:90]}) perdió su "
                              f"vigencia —{_etiqueta_vigencia(_vigencia(t))}— y no se propone "
                              f"aplicarlo.")
                continue
            fuera.append(apoyo_de_tesis(t, tribunal, circuito))
        else:
            k = _plano(txt)
            if k in vistos:
                continue
            vistos.add(k)
            fuera.append(_apoyo_norma(txt))
    return fuera, avisos


def registros_sueltos(texto: str, catalogo: dict) -> list:
    """Las cifras de registro que un texto libre nombra y el acervo del asunto
    no tiene (para avisar: la razón del motor viaja tal cual al resolver)."""
    return [r for r in dict.fromkeys(_RX_REGISTRO.findall(str(texto or ""))) if r not in catalogo]


# ═══ EL DESENLACE, POR CÓDIGO ═══════════════════════════════════════════════
#
# En el 631 el proyecto «revocó para volver a conceder» (CONTINUAR §3): la rama
# se dedujo contando verbos en el PDF. Con el resolutivo del juzgado leído
# (577c700) y la rama de `tipos_asunto`, la tarjeta dice ANTES de generar qué
# resolutivos saldrían en cada vía. Sin nombres propios: el encargo del 631
# guardaba a la recurrente como «quejoso» (SPEC B, punto 4) y un nombre
# equivocado en un resolutivo previsto es peor que «la parte quejosa».
_GENERICOS = {"quejoso": "la parte quejosa",
              "responsable_originaria": "la autoridad responsable",
              "expediente": "de origen", "HUECO": HUECO}


def _rellenar(p: str) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: _GENERICOS.get(m.group(1), HUECO), p)


def desenlace_de(tipo_asunto: str, resolvio_a_quo: str, sentido: str) -> tuple:
    """(puntos resolutivos previstos, nota | None) de un sentido, por código.

    AMPARO EN REVISIÓN QUE REVOCA UNA CONCESIÓN (el 631, vía contraria): se
    revoca y, antes de negar, el tribunal reasume jurisdicción y estudia los
    conceptos de violación que el juez no estudió (art. 93, fr. VI, de la Ley
    de Amparo); sólo si caen, se niega. Sin eso, «revoca y niega» es
    prematuro, y la nota lo dice.
    """
    try:
        import tipos_asunto as _ta
    except Exception:                                   # pragma: no cover
        return [], None
    tipo = _ta.normalizar(tipo_asunto or "") or "amparo_directo"
    s = _txt(sentido).lower().replace(" ", "_")
    if not s:
        return [], None
    pros = _ta.prospera(s)
    if tipo == "amparo_revision":
        a = _txt(resolvio_a_quo).lower()
        rama = _ta.rama_revision(a, s)
        info = _ta.RAMAS_REVISION.get(rama) or {}
        puntos = [_rellenar(p) for p in (info.get("puntos") or [])]
        nota = info.get("aviso") or None
        if rama == "revoca_fondo_niega" and a in ("concede", "sobresee_concede"):
            nota = ("Al revocar la concesión, el tribunal reasume jurisdicción y estudia los "
                    "conceptos de violación que el Juzgado no estudió (art. 93, fr. VI, de la "
                    "Ley de Amparo); sólo si caen, se niega. Si alguno prospera, se concede "
                    "por esa razón.")
        elif rama.startswith("revoca_sobreseimiento"):
            # El sentido del AMPARO sale del estudio de los conceptos, que el
            # tribunal hace por primera vez: no se supone ninguno.
            puntos = [puntos[0], f"SEGUNDO. La Justicia de la Unión {HUECO} a la parte quejosa, "
                                 f"según resulte del estudio de los conceptos de violación."] \
                if puntos else []
            nota = ("Levantado el sobreseimiento, el tribunal asume jurisdicción y estudia los "
                    f"conceptos de violación por primera vez ({info.get('fundamento', '')}); el "
                    "sentido del amparo sale de ese estudio.")
        return puntos, nota
    if tipo == "queja":
        if s == "sin_materia":
            return ["ÚNICO. Se declara sin materia el recurso de queja."], None
        return [f"ÚNICO. Es {'fundado' if pros else 'infundado'} el recurso de queja."], None
    if tipo == "revision_fiscal":
        return [f"ÚNICO. Se {'revoca' if pros else 'confirma'} la sentencia recurrida."], None
    if pros:
        return ["ÚNICO. La Justicia de la Unión ampara y protege a la parte quejosa, contra el "
                "acto reclamado a la autoridad responsable, para los efectos precisados en el "
                "último considerando de esta ejecutoria."], None
    return ["ÚNICO. La Justicia de la Unión no ampara ni protege a la parte quejosa, contra el "
            "acto reclamado a la autoridad responsable."], None


# ═══ EL PRINCIPAL ════════════════════════════════════════════════════════════

def indice_principal(problemas: list) -> tuple:
    """(índice 0-based, de dónde sale la jerarquía). Un solo principal:
    /taller/problema lo impone. Sin marca, el primero, como el árbol."""
    for i, p in enumerate(problemas or []):
        if isinstance(p, dict) and _txt(p.get("jerarquia")).lower() == "principal":
            return i, ("secretario" if p.get("jerarquia_por_secretario") else "fase3")
    return 0, "por_omision"


def _numero_motor(glob: dict, problemas: list) -> int:
    """El problema que el MOTOR tomó como el que decide (1-based), o 0. Lo
    dice en la lista (`papel: principal`) y en `problema_que_decide`; la
    propuesta no ve la jerarquía de la fase 3 y nadie lo conciliaba."""
    for c in (glob.get("checklist") or []):
        if isinstance(c, dict) and _txt(c.get("papel")).lower() == "principal":
            try:
                n = int(c.get("numero"))
                if 1 <= n <= len(problemas):
                    return n
            except (TypeError, ValueError):
                pass
    pqd = _txt(glob.get("problema_que_decide"))
    if pqd:
        for i, p in enumerate(problemas or []):
            if _clave_problema(_preg(p)).lower() == _clave_problema(pqd).lower():
                return i + 1
        # EL MISMO TEMA, SÓLO SI NO ES AMBIGUO: en el 631 las dos preguntas
        # comparten media frase y el umbral las daba por iguales; afirmar una
        # discrepancia (o negarla) con eso sería elegir en silencio.
        casan = [i for i, p in enumerate(problemas or []) if _mismo_tema(_preg(p), pqd)]
        if len(casan) == 1:
            return casan[0] + 1
    return 0


def _propuesta_de(propuestas: list, pregunta: str, i: int) -> dict:
    """La propuesta del motor para ese problema: por el texto (así las alinea
    `emparejar`) y, si no casa, por la posición."""
    k = _clave_problema(pregunta).lower()
    for p in (propuestas or []):
        if _clave_problema(_get(p, "problema", "")).lower() == k:
            return p if isinstance(p, dict) else dict(vars(p))
    if 0 <= i < len(propuestas or []):
        p = propuestas[i]
        return p if isinstance(p, dict) else dict(vars(p))
    return {}


def _contraste_de(contraste: list, numero: int) -> dict | None:
    for c in (contraste or []):
        if isinstance(c, dict):
            try:
                if int(c.get("numero")) == numero:
                    return {k: c.get(k) for k in ("razon_toral", "la_combate", "sobrevive",
                                                  "veredicto_previo")}
            except (TypeError, ValueError):
                continue
    return None


# ═══ LOS SECUNDARIOS, POR EL ÁRBOL, EN LAS DOS VÍAS ═════════════════════════

def _correr_arbol(problemas: list, p_idx: int, sentido_via: str, sentido_motor: str,
                  propuestas_motor: list, checklist: list, tipo_asunto: str) -> dict:
    """{clave_problema: criterio repartido} con el principal en `sentido_via`.

    Es la misma regla que aplicará el resolver (`arbol_decision`), sin modelo:
    presupuesto() con la cita literal, la guarda procesal de los arts. 74-V,
    174 y 189, el tema distinto y el mayor beneficio. Los accesorios llevan lo
    que el motor propuso sólo si esta vía es la del motor; en la otra van
    vacíos y sin tocar —el camino `_de_la_maquina`—, y el árbol marca
    `recalificar` donde la calificación escrita para la otra vía no sirve.
    """
    try:
        import arbol_decision as _ad
    except Exception:                                   # pragma: no cover
        return {}
    via_motor = bool(sentido_motor) and _prospera(sentido_via) == _prospera(sentido_motor)
    criterios = []
    for i, p in enumerate(problemas):
        preg = _preg(p)
        if i == p_idx:
            criterios.append({"problema": preg, "sentido": sentido_via, "razonamiento": "",
                              "jerarquia": "principal", "tocado": False})
            continue
        s = r = ""
        if via_motor:
            pm = _propuesta_de(propuestas_motor, preg, i)
            s, r = _txt(pm.get("sentido")), _txt(pm.get("razon"))
        criterios.append({"problema": preg, "sentido": s, "razonamiento": r,
                          "jerarquia": "accesorio", "tocado": False})
    try:
        res = _ad.reparto_para_pantalla(
            problemas, criterios, checklist, propuestas_motor,
            sentido_motor=sentido_motor, tipo_asunto=tipo_asunto)
    except Exception as e:                              # pragma: no cover
        print(f"   ⚠️ TARJETA: el árbol no se pudo correr: {type(e).__name__}: {str(e)[:120]}")
        return {}
    return {_clave_problema(c.get("problema")): c for c in (res.get("criterios") or [])}


_DE_ARBOL = {"principal": "principal", "tuya": "secretario", "recalificada": "motor"}


def _suerte_en_via(c: dict | None, pros: bool, entrada: dict, sentido_motor: str,
                   pm: dict, rel_declarada: str, delib: tuple | None) -> dict | None:
    """La SUERTE del contrato para un accesorio en una vía.

    Lo que el árbol deja «por recalificar» (la vía contraria a la del motor) no
    se enseña vacío: se enseña la suerte que el motor escribió PARA ESA VÍA
    (`arbol_decision._suerte`), o la del abogado de esa vía si hay
    deliberación, rotulada «previsto». Al elegir la vía, /taller/recalificar la
    regenera con la premisa del secretario, como hoy.
    """
    if c is None:
        return None
    de_a = _txt(c.get("de"))
    recal = bool(c.get("recalificar"))
    out = {"sentido": _txt(c.get("sentido")), "de": _DE_ARBOL.get(de_a, "arbol"),
           "por_que": _txt(c.get("por_que")), "relacion": _txt(c.get("relacion")),
           "guarda": _txt(c.get("guarda")) or None, "recalificar": recal,
           "previsto": False, "de_arbol": de_a}
    if not recal:
        return out
    s2 = r2 = ""
    if delib and delib[0]:
        s2, r2 = delib
    else:
        try:
            import arbol_decision as _ad
            s2, r2 = _ad._suerte(entrada or {}, "si_prospera" if pros else "si_no_prospera",
                                 sentido_motor)
        except Exception:                               # pragma: no cover
            s2, r2 = "", ""
    if s2:
        out.update(sentido=s2, de="motor", previsto=True,
                   por_que=r2 or "la suerte que el motor escribió para esta vía")
    elif not rel_declarada and _txt(pm.get("sentido_propio") or pm.get("sentido")):
        # SIN RELACIÓN DECLARADA CON EL PRINCIPAL, lo que el motor le propuso
        # es su calificación por sus propios méritos: se enseña como previsto.
        out.update(sentido=_txt(pm.get("sentido_propio") or pm.get("sentido")), de="motor",
                   previsto=True,
                   por_que=("la calificación que el motor le propuso por su cuenta; al elegir "
                            "esta vía se vuelve a calificar con tu premisa"))
    else:
        out.update(sentido="", previsto=False,
                   por_que=("sin suerte escrita para esta vía: al elegirla, el motor lo vuelve "
                            "a calificar con tu premisa"))
    return out


# ═══ LA DELIBERACIÓN, SI EXISTE (SPEC C2) ════════════════════════════════════

def _payload_deliberacion(marca) -> dict | None:
    """El resultado de la deliberación dentro de su marca, sea cual sea el
    envoltorio: {huella, estado: "listo", resultado|respuesta|deliberacion:
    {…}} o el objeto directo. Se reconoce por sus `vias` (A = el principal
    prospera, B = no prospera, lector2_potencia §0). None si no hay."""
    if not isinstance(marca, dict):
        return None
    if _txt(marca.get("estado")) in ("en_curso", "fallo", "apagada"):
        return None
    for k in ("resultado", "respuesta", "deliberacion"):
        v = marca.get(k)
        if isinstance(v, dict) and isinstance(v.get("vias"), dict):
            return v
    if isinstance(marca.get("vias"), dict):
        return marca
    return None


def _estado_juez(d: dict) -> str:
    for v in (d.get("estado_juez"), (d.get("juez") or {}).get("estado")
              if isinstance(d.get("juez"), dict) else None, d.get("estado")):
        v = _txt(v).lower().replace("renido", "reñido")
        if v in ESTADOS:
            return v
    return ""


def _suertes_delib(v: dict) -> dict:
    """{numero: (sentido, razón)} de los secundarios que escribió el abogado
    de esa vía, sólo si traen una calificación que el árbol reconoce."""
    out = {}
    try:
        import arbol_decision as _ad
    except Exception:                                   # pragma: no cover
        return out
    for x in (v.get("secundarios") or []):
        if not isinstance(x, dict):
            continue
        try:
            n = int(x.get("numero"))
        except (TypeError, ValueError):
            continue
        s = _ad._sentido_valido(x.get("sentido"))
        if not s:
            s, r = _ad._leer_suerte(_txt(x.get("suerte")))
        else:
            r = ""
        if s:
            out[n] = (s, _txt(x.get("razon")) or r)
    return out


# ═══ EL GANCHO DE LOS CONCEPTOS OMITIDOS (SPEC B, otra rama) ═════════════════
#
# Otro implementador escribe la función pura que dice si, al revocar una
# concesión, hacen falta los conceptos de violación que el juez no estudió y
# si los tenemos (art. 93, fr. VI). Aquí sólo se engancha: import perezoso, y
# si no existe (o falla) el campo sale null. Se llama con los argumentos que
# su firma pida de este fondo común, para no depender de cómo la nombre.
GANCHOS_OMITIDOS = (("fase_rama", "conceptos_omitidos"),
                    ("tipos_asunto", "conceptos_omitidos"),
                    ("conceptos_omitidos", "conceptos_omitidos"))


def _conceptos_omitidos(fondo: dict):
    fn = None
    for mod, nombre in GANCHOS_OMITIDOS:
        try:
            m = __import__(mod)
        except Exception:
            continue
        fn = getattr(m, nombre, None)
        if callable(fn):
            break
        fn = None
    if fn is None:
        return None
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        if any(p.kind == p.VAR_KEYWORD for p in params.values()):
            res = fn(**fondo)
        else:
            kw = {k: fondo[k] for k in params if k in fondo}
            if kw:
                res = fn(**kw)
            elif len(params) == 1:
                res = fn(fondo)
            else:
                return None
    except Exception as e:
        print(f"   ⚠️ TARJETA: conceptos omitidos no disponibles: {type(e).__name__}")
        return None
    if not isinstance(res, dict):
        return None
    out = dict(res)
    out.setdefault("hacen_falta", bool(res.get("hacen_falta")))
    out.setdefault("por_que", _txt(res.get("por_que")) or None)
    out.setdefault("tenemos", bool(res.get("tenemos")))
    return _limpio(out)


# ═══ EL ESTADO, DETERMINISTA ═════════════════════════════════════════════════

def _vigentes_que_obligan(apoyos: list) -> set:
    return {a["registro"] for a in (apoyos or [])
            if a.get("registro") and a.get("fuerza") == "obliga"}


def _estado(prop: dict | None, opu: dict | None, contraste: dict | None,
            discrepa: dict | None, indispensables: list, hay_global: bool) -> tuple:
    """(estado, razones) SIN deliberación. Nunca un porcentaje.

    «no_alcanza»: no hay sentido para el asunto, falta una constancia que el
      motor declaró indispensable, o ninguna vía tiene un criterio del acervo
      verificado y vigente.
    «claro»: DOS señales independientes coinciden y ninguna la contradice:
      (1) el contraste CIERRA el punto en la dirección de la propuesta —el
          agravio no combate la razón toral, o el fallo sobrevive por otra
          consideración, y la propuesta no lo hace prosperar—, y
      (2) la propuesta se apoya en un criterio que OBLIGA a este tribunal que
          la contraria no invoca, y la contraria no tiene uno propio.
    «reñido»: todo lo demás.

    POR QUÉ TAN EXIGENTE. Kingston-24: el motor acierta 50% contra 54% de
    «siempre niega», y su error es sesgado a CONCEDER (en 6 de 15 asuntos
    estables concedió donde el engrose negó por infundados). Una propuesta que
    hace prosperar el agravio nunca sale «claro» sin deliberación: el
    contraste sólo puede decir «hay que examinarlo», no «tiene razón». Y
    contar criterios no basta: en el 631 la propuesta traía cuatro
    jurisprudencias de la Corte sobre cosa juzgada en genérico (168958,
    170353…) y la figura que decidía era otra (causahabiencia procesal).
    """
    razones = []
    if not hay_global:
        return "no_alcanza", ["El motor no propuso un sentido para todo el asunto: "
                              "la propuesta que se ve es la del problema principal."]
    if indispensables:
        razones.append("Falta una constancia que el motor declaró indispensable: "
                       + "; ".join(indispensables[:3]) + ".")
    en_acervo = [a for v in (prop, opu) if v for a in (v.get("apoyos") or [])
                 if a.get("en_acervo")]
    if not en_acervo:
        razones.append("Ninguna de las dos vías se apoya en un criterio del acervo verificado "
                       "y vigente.")
    if razones:
        return "no_alcanza", razones

    pros = bool(prop and prop.get("prospera"))
    ob_p = _vigentes_que_obligan((prop or {}).get("apoyos"))
    ob_o = _vigentes_que_obligan((opu or {}).get("apoyos"))
    solo_p, solo_o = ob_p - ob_o, ob_o - ob_p

    cierra = None                  # True: a favor de la propuesta; False: en contra
    if contraste:
        v = _txt(contraste.get("veredicto_previo")).lower()
        if v in ("inoperante", "fundado_pero_insuficiente"):
            cierra = not pros
            if v == "inoperante":
                razones.append("El contraste dice que el planteamiento no combate la razón toral "
                               "del fallo" + ("; la propuesta lo desestima." if not pros else
                                               ", y la propuesta lo hace prosperar."))
            else:
                razones.append("El contraste dice que el fallo se sostiene por otra "
                               "consideración" + ("; la propuesta lo desestima." if not pros else
                                                   ", y la propuesta lo hace prosperar."))
        else:
            razones.append("El contraste no cierra el punto: el planteamiento combate la razón "
                           "toral y hay que decidir si tiene razón.")
    else:
        razones.append("No hay contraste del principal.")

    if solo_p and not solo_o:
        razones.append("La propuesta invoca criterios que obligan a este tribunal y la contraria "
                       "no: " + ", ".join(sorted(solo_p)[:4]) + ".")
    elif solo_o and not solo_p:
        razones.append("La vía contraria invoca criterios que obligan a este tribunal y la "
                       "propuesta no: " + ", ".join(sorted(solo_o)[:4]) + ".")
    elif ob_p or ob_o:
        razones.append("Las dos vías invocan criterios que obligan a este tribunal"
                       + (f" ({', '.join(sorted(ob_p & ob_o)[:3])} en las dos)"
                          if ob_p & ob_o else "") + ": ese escalón no las separa.")
    else:
        razones.append("Ninguna vía invoca un criterio que obligue a este tribunal.")
    if discrepa:
        razones.append("El motor tomó como problema que decide otro distinto del principal.")
    if opu is None:
        razones.append("El motor no encontró cómo sostener la vía contraria con el acervo.")

    claro = (cierra is True and bool(solo_p) and not solo_o and not discrepa)
    if cierra is False or not claro:
        return "reñido", razones
    return "claro", razones


# ═══ LAS VÍAS ════════════════════════════════════════════════════════════════

def _via(sentido: str, razon: str, efecto: str, apoyos_crudos, catalogo: dict,
         rama_info: dict, protectora: dict | None, avisos: list, etiqueta: str,
         interpretacion=None, cadena=None, objecion=None) -> dict:
    tribunal = _txt(rama_info.get("tribunal"))
    circuito = int(rama_info.get("circuito") or 0) or circuito_de(tribunal)
    apoyos, av = hidratar_apoyos(apoyos_crudos, catalogo, tribunal, circuito, etiqueta)
    avisos.extend(av)
    sueltos = registros_sueltos(razon, catalogo)
    if sueltos:
        avisos.append(f"La razón de {etiqueta} nombra registros que no están en el acervo del "
                      f"asunto ({', '.join(sueltos[:4])}): no se tienen por citados.")
    puntos, nota = desenlace_de(rama_info.get("tipo_asunto", ""),
                                rama_info.get("resolvio_a_quo", ""), sentido)
    vp = None
    if isinstance(protectora, dict) and _txt(protectora.get("sentido")) \
            and _prospera(protectora["sentido"]) == _prospera(sentido):
        vp = _limpio(protectora)
        # LOS APOYOS DE LA VÍA PROTECTORA NO LOS VERIFICABA NADIE (`revisar`
        # sólo mira la global y la alternativa). En el 631 traía dos registros
        # que no están en el acervo del asunto. Se quedan los comprobados, con
        # la misma forma de hoy —registros en texto— para la pantalla que ya
        # la pinta, y el detalle aparte.
        ap_vp, av_vp = hidratar_apoyos(protectora.get("apoyos"), catalogo, tribunal, circuito,
                                       "la vía protectora")
        avisos.extend(av_vp)
        vp["apoyos"] = [a["registro"] or a["norma"] for a in ap_vp]
        vp["apoyos_detalle"] = ap_vp
    return {"sentido": _txt(sentido), "prospera": _prospera(sentido), "razon": _txt(razon),
            "efecto": _txt(efecto), "desenlace": puntos, "desenlace_nota": nota,
            "interpretacion": _txt(interpretacion) or None,
            "cadena": _limpio(cadena) if isinstance(cadena, dict) else None,
            "objecion": objecion if (isinstance(objecion, dict)
                                     and (objecion.get("de_la_otra_via")
                                          or objecion.get("respuesta"))) else None,
            "apoyos": apoyos, "via_protectora": vp}


def _via_de_deliberacion(v: dict, glob: dict, catalogo: dict, rama_info: dict,
                         avisos: list, etiqueta: str) -> dict:
    """Una vía del abogado de la deliberación, verificada aquí otra vez: los
    apoyos salen del catálogo del asunto (el modelo sólo nombra ids), la fuerza
    y la vigencia por código y el desenlace por `tipos_asunto`."""
    sentido = _txt(v.get("sentido"))
    cadena = v.get("cadena") if isinstance(v.get("cadena"), dict) else None
    razon = _txt(v.get("razon")) or _txt((cadena or {}).get("conclusion"))
    alt = glob.get("alternativa") if isinstance(glob.get("alternativa"), dict) else {}
    efecto = _txt(v.get("efecto"))
    if not efecto:
        if _txt(glob.get("sentido")) and _prospera(glob["sentido"]) == _prospera(sentido):
            efecto = _txt(glob.get("efecto"))
        elif _txt(alt.get("sentido")) and _prospera(alt["sentido"]) == _prospera(sentido):
            efecto = _txt(alt.get("efecto"))
    obj = v.get("objecion") if isinstance(v.get("objecion"), dict) else {
        "de_la_otra_via": _txt(v.get("objecion_de_la_otra_via")) or None,
        "respuesta": _txt(v.get("respuesta")) or None}
    return _via(sentido, razon, efecto, v.get("propongo_aplicar") or v.get("apoyos") or [],
                catalogo, rama_info, glob.get("via_protectora"), avisos, etiqueta,
                interpretacion=v.get("interpretacion") or (cadena or {}).get("regla"),
                cadena=cadena, objecion=obj)


# ═══ EL PROPIO TRIBUNAL Y LA LÍNEA DE LA CORTE ═══════════════════════════════

def _es_del_principal(texto: str, propias: list, ajenas: list) -> bool:
    """¿Esta entrada del espejo es del principal? El espejo guarda la pregunta
    tal como estaba al consultar: se casa por igualdad con la pregunta o con
    la que el secretario corrigió (`pregunta_original`). Sólo si no es
    literalmente la de otro problema, se admite el mismo tema escrito
    distinto: en el 631 las dos preguntas comparten «la sustitución de la
    parte actora» y el umbral de `_mismo_tema` las daba por la misma."""
    k = _clave_problema(texto).strip().lower()
    if not k:
        return True
    if any(k == _clave_problema(p).strip().lower() for p in propias if p):
        return True
    if any(k == _clave_problema(p).strip().lower() for p in ajenas if p):
        return False
    return any(_mismo_tema(texto, p) for p in propias if p)


def _tu_tribunal(espejo: list, propias: list, ajenas: list = ()) -> list:
    """Las filas del espejo del principal. Tolera las de la OAJ (nivel,
    calificación, razón, similitud calibrada en %, NEUN; rama precedentes-oaj,
    sin mezclar) y las del espejo de hoy (`score` es un coseno, no una
    similitud: no se enseña como tal)."""
    filas = []
    for e in (espejo or []):
        if not isinstance(e, dict):
            continue
        if "filas" in e:
            if not _es_del_principal(_txt(e.get("problema")), propias, ajenas):
                continue
            filas.extend(f for f in (e.get("filas") or []) if isinstance(f, dict))
        elif e.get("expediente") and _es_del_principal(_txt(e.get("problema")), propias, ajenas):
            filas.append(e)
    out = []
    for f in filas:
        sim = f.get("similitud")
        try:
            sim = round(float(sim) / 100.0, 2) if sim is not None and float(sim) > 1 else (
                float(sim) if sim is not None else None)
        except (TypeError, ValueError):
            sim = None
        out.append({"expediente": _txt(f.get("expediente")), "fecha": _txt(f.get("fecha")),
                    "sentido": _txt(f.get("sentido")),
                    "calificacion": _txt(f.get("calificacion")) or None,
                    "razon": _txt(f.get("razon")) or None, "similitud": sim,
                    "nivel": _txt(f.get("nivel")) or None,
                    "neun": _txt(f.get("neun")) or None,
                    "tipo_asunto": _txt(f.get("tipo_asunto")) or None,
                    "tema": _txt(f.get("tema")) or None,
                    "pdf_url": _txt(f.get("pdf_url")) or None})
    return out


def _linea_corte(material, internet, tribunal: str, circuito: int) -> dict:
    conf = [apoyo_de_tesis(t, tribunal, circuito) for t in _lista(material, "tesis")
            if t.get("de_internet")]
    pistas = [_txt(p) for p in ((internet or {}).get("pistas") or []) if _txt(p)]
    return {"confirmadas": conf, "pistas": pistas}


# ═══ ARMAR ═══════════════════════════════════════════════════════════════════

def vacia(estado_calculo: str, huella: str = "") -> dict:
    """La tarjeta cuando todavía no hay qué enseñar."""
    return {"formato": FORMATO, "estado_calculo": estado_calculo, "huella": huella,
            "principal": None, "vias": {"propuesta": None, "opuesta": None},
            "recomendada": None, "estado": None, "estado_por_que": [], "secundarios": [],
            "independientes": [], "que_la_cambiaria": None, "tu_tribunal": [],
            "linea_corte": {"confirmadas": [], "pistas": []}, "deliberacion": None,
            "conceptos_omitidos": None, "avisos": []}


def armar(propuesta_guardada, material, problemas_fase3, contraste=None, espejo=None,
          internet=None, rama_info=None, deliberacion=None, *, propuestas_motor=None) -> dict:
    """La tarjeta (formato 1 de contrato_tarjeta.md), sin modelo.

    `propuesta_guardada`: la respuesta de /taller/proponer (formato 2):
      global, propuestas, contraste. `material`: el Material de la sesión o su
      forma guardada (tesis, espejo). `problemas_fase3`: los de la fase 3, en
      su orden. `contraste`: la lista por planteamiento (si None, la de la
      propuesta). `espejo`: si None, el del material. `internet`: ses/marca
      {pistas, resumen, …}. `rama_info`: {tipo_asunto, resolvio_a_quo,
      tribunal, circuito?, huella?, conceptos_violacion?, necesita_conceptos?}.
      `deliberacion`: la marca «deliberacion» (SPEC C2) o None.
      `propuestas_motor`: las propuestas con `sentido_propio`/`razon_propia`
      (lo que el motor propuso antes del árbol; viajan en la columna
      `propuestas` de la fila); si None, las de la respuesta.
    """
    resp = propuesta_guardada if isinstance(propuesta_guardada, dict) else {}
    rama_info = dict(rama_info or {})
    problemas = [p if isinstance(p, dict) else {"pregunta": _txt(p)}
                 for p in (problemas_fase3 or [])]
    huella = _txt(rama_info.get("huella") or resp.get("huella"))
    if not problemas:
        return vacia("sin_propuesta", huella)
    glob = resp.get("global") if isinstance(resp.get("global"), dict) else {}
    props = [dict(p) if isinstance(p, dict) else dict(vars(p))
             for p in (resp.get("propuestas") or [])]
    # LO QUE EL MOTOR PROPUSO ANTES DEL ÁRBOL (`sentido_propio`) viaja en la
    # columna `propuestas` de la fila, no en la respuesta: la guarda procesal
    # lo necesita para devolverle a una procesal su calificación.
    pm_lista = [dict(p) if isinstance(p, dict) else dict(vars(p))
                for p in (propuestas_motor or [])] or [dict(p) for p in props]
    for p in pm_lista:
        p.setdefault("sentido_propio", "")
        p.setdefault("razon_propia", "")
    props = props or pm_lista
    contraste = contraste if contraste is not None else (resp.get("contraste") or [])
    espejo = espejo if espejo is not None else _lista(material, "espejo")
    tribunal = _txt(rama_info.get("tribunal"))
    circuito = int(rama_info.get("circuito") or 0) or circuito_de(tribunal)
    tipo_asunto = _txt(rama_info.get("tipo_asunto"))
    catalogo = catalogo_de(material)
    avisos: list = []

    # ── EL PRINCIPAL ──
    p_idx, jer_de = indice_principal(problemas)
    p_num = p_idx + 1
    pral = problemas[p_idx]
    p_preg = _preg(pral)
    pm_pral = _propuesta_de(props, p_preg, p_idx)
    ctx = glob.get("contexto") if isinstance(glob.get("contexto"), dict) else {}
    n_mot = _numero_motor(glob, problemas) if glob else 0
    discrepa = None
    if n_mot and n_mot != p_num:
        discrepa = {"numero_motor": n_mot,
                    "nota": (f"El motor tomó como problema que decide el {n_mot} "
                             f"(«{_preg(problemas[n_mot - 1])[:140]}»), no el {p_num} que "
                             f"{'marcaste tú' if jer_de == 'secretario' else 'marcó la fase 3'} "
                             f"como principal. La tarjeta sigue la jerarquía; el porqué del "
                             f"principal que escribió el motor puede hablar del otro.")}
    pred = pm_pral.get("prediccion") if isinstance(pm_pral.get("prediccion"), dict) else {}
    principal = {"numero": p_num, "pregunta": p_preg, "clase": _txt(pral.get("clase")) or None,
                 "jerarquia_de": jer_de,
                 "por_que_principal": _txt(ctx.get("tema_principal")) or None,
                 "discrepa_motor": discrepa,
                 "contraste": _contraste_de(contraste, p_num),
                 "prediccion": ({"frase": _txt(pred.get("frase")), "n": pred.get("n")}
                                if _txt(pred.get("frase")) else None)}

    # ── LAS VÍAS ──
    hay_global = bool(glob and glob.get("alcanza", True) is not False and _txt(glob.get("sentido")))
    sentido_motor = _txt(glob.get("sentido")) if hay_global else ""
    delib = _payload_deliberacion(deliberacion)
    alt = glob.get("alternativa") if isinstance(glob.get("alternativa"), dict) else {}
    recomendada = None
    estado_juez = ""
    suertes_delib = {"propuesta": {}, "opuesta": {}}
    delib_ctx = None
    via_p = via_o = None
    if delib:
        vias_d = {str(k).upper(): v for k, v in (delib.get("vias") or {}).items()
                  if isinstance(v, dict) and _txt(v.get("sentido"))}
        rec = _txt(delib.get("recomendada")).upper()
        if rec not in vias_d:
            # Sin recomendada (reñido), la columna de la izquierda es la vía
            # que coincide con la del motor, para que «la propuesta» siga
            # queriendo decir lo mismo; la pantalla no rotula ninguna.
            rec = next((k for k, v in vias_d.items()
                        if _prospera(v["sentido"]) == _prospera(sentido_motor)),
                       next(iter(vias_d), ""))
        otra = next((k for k in vias_d if k != rec), "")
        if rec:
            via_p = _via_de_deliberacion(vias_d[rec], glob, catalogo, rama_info, avisos,
                                         "la vía propuesta")
            via_o = (_via_de_deliberacion(vias_d[otra], glob, catalogo, rama_info, avisos,
                                          "la vía contraria") if otra else None)
            suertes_delib["propuesta"] = _suertes_delib(vias_d[rec])
            if otra:
                suertes_delib["opuesta"] = _suertes_delib(vias_d[otra])
            estado_juez = _estado_juez(delib)
            pr_d = delib.get("principal") if isinstance(delib.get("principal"), dict) else delib
            delib_ctx = {"origen": "deliberacion",
                         "pregunta_decisiva": _txt(pr_d.get("pregunta_decisiva")) or None,
                         "figura": _txt(pr_d.get("figura")) or None,
                         "proposicion_toral": (_limpio(pr_d.get("proposicion_toral"))
                                               if isinstance(pr_d.get("proposicion_toral"), dict)
                                               else None)}
            for a in (delib.get("avisos") or []):
                if _txt(a):
                    avisos.append(_txt(a))
        else:
            delib = None
    if not (delib and delib_ctx):
        delib = None
        if hay_global:
            via_p = _via(sentido_motor, _txt(glob.get("razon")), _txt(glob.get("efecto")),
                         glob.get("apoyos"), catalogo, rama_info, glob.get("via_protectora"),
                         avisos, "la vía propuesta", interpretacion=glob.get("interpretacion"))
            if _txt(alt.get("sentido")) and _prospera(alt["sentido"]) != _prospera(sentido_motor):
                via_o = _via(_txt(alt.get("sentido")), _txt(alt.get("razon")),
                             _txt(alt.get("efecto")), alt.get("apoyos"), catalogo, rama_info,
                             glob.get("via_protectora"), avisos, "la vía contraria",
                             interpretacion=alt.get("interpretacion"))
                if not via_o["razon"]:
                    avisos.append("La vía contraria llegó sin razón: «Redactar el criterio de "
                                  "esta vía» la pide al motor sólo con tu clic.")
            else:
                via_o = None
                if _txt(alt.get("sentido")):
                    avisos.append("La alternativa del motor va en el mismo sentido que su "
                                  "propuesta: no es una vía contraria y no se enseña como tal.")
        else:
            # SIN PROPUESTA GLOBAL: el principal con su propuesta por problema,
            # sin columna contraria; la ventana manual se abre sola.
            via_p = _via(_txt(pm_pral.get("sentido")), _txt(pm_pral.get("razon")), "",
                         pm_pral.get("apoyos"), catalogo, rama_info, None, avisos,
                         "la propuesta del principal")
            via_o = None
            if via_p["sentido"]:
                avisos.append("Sin sentido para todo el asunto: el desenlace previsto sale sólo "
                              "de la propuesta del problema principal.")
    sentido_ref = sentido_motor or _txt(pm_pral.get("sentido"))

    # ── LOS SECUNDARIOS, POR EL ÁRBOL, EN CADA VÍA ──
    checklist = list(glob.get("checklist") or []) if hay_global else []
    if delib and isinstance(delib.get("checklist"), list):
        checklist = list(delib["checklist"])
    rep_p = (_correr_arbol(problemas, p_idx, via_p["sentido"], sentido_ref, pm_lista,
                           checklist, tipo_asunto) if via_p and via_p["sentido"] else {})
    rep_o = (_correr_arbol(problemas, p_idx, via_o["sentido"], sentido_ref, pm_lista,
                           checklist, tipo_asunto) if via_o and via_o["sentido"] else {})
    try:
        import arbol_decision as _ad
    except Exception:                                   # pragma: no cover
        _ad = None
    secundarios, independientes = [], []
    for i, p in enumerate(problemas):
        if i == p_idx:
            continue
        preg = _preg(p)
        k = _clave_problema(preg)
        entrada = _ad.entrada_de(checklist, i + 1, preg) if _ad else {}
        rel = _ad.relacion_de(p, p_num, entrada) if _ad else ""
        pm = _propuesta_de(pm_lista, preg, i)
        cp, co = rep_p.get(k), rep_o.get(k)
        s_p = _suerte_en_via(cp, bool(via_p and via_p["prospera"]), entrada, sentido_ref, pm,
                             rel, suertes_delib["propuesta"].get(i + 1))
        s_o = _suerte_en_via(co, bool(via_o and via_o["prospera"]), entrada, sentido_ref, pm,
                             rel, suertes_delib["opuesta"].get(i + 1))
        # EL TEMA DISTINTO NO ES UN SECUNDARIO: el árbol lo dejó «distinto» en
        # las vías que corrió (la suerte escrita para una vía manda sobre la
        # etiqueta, 93/2026). Va aparte, con su propuesta propia.
        vias_corridas = [x for x in (cp, co) if x is not None]
        if rel == "distinto" and vias_corridas and all(_txt(x.get("de")) == "distinto"
                                                       for x in vias_corridas):
            pr_i = _propuesta_de(props, preg, i)
            ap, av = hidratar_apoyos(pr_i.get("apoyos"), catalogo, tribunal, circuito,
                                     f"el problema {i + 1}")
            avisos.extend(av)
            independientes.append({"numero": i + 1, "pregunta": preg,
                                   "propuesta": {"sentido": _txt(pr_i.get("sentido")),
                                                 "razon": _txt(pr_i.get("razon")),
                                                 "apoyos": ap}})
            continue
        relacion = ("presupone" if any(_txt((x or {}).get("relacion")) == "presupone"
                                       for x in (s_p, s_o))
                    else "depende" if rel in ("depende", "distinto") else "autonoma")
        secundarios.append({"numero": i + 1, "pregunta": preg,
                            "clase": _txt(p.get("clase")) or None, "relacion": relacion,
                            "en_propuesta": s_p, "en_opuesta": s_o})

    # ── LO QUE LA CAMBIARÍA ──
    vp = glob.get("via_protectora") if isinstance(glob.get("via_protectora"), dict) else {}
    indispensables = [_txt(c.get("que")) for c in (glob.get("constancias") or [])
                      if isinstance(c, dict) and c.get("indispensable") is True and _txt(c.get("que"))]
    crux = None
    if delib and isinstance(delib.get("crux"), dict):
        crux = {k: (_txt(delib["crux"].get(k)) or None) for k in ("que", "si_cambia", "constancia")}
    que_cambiaria = {"en_contra": _txt(glob.get("en_contra")) or None, "crux": crux,
                     "constancias_indispensables": indispensables,
                     "limite_protector": _txt(vp.get("limite")) or None}

    # ── EL ESTADO ──
    estado, por_que = _estado(via_p, via_o, principal["contraste"], discrepa, indispensables,
                              hay_global or bool(delib))
    if delib and estado_juez:
        # EL JUEZ MANDA, con dos frenos que no se discuten: sin constancia
        # indispensable o sin un solo apoyo verificado, no alcanza aunque el
        # juez diga otra cosa (la verificación E es de código).
        duros = estado == "no_alcanza"
        por_que = ([f"Deliberación: el juez la marcó «{estado_juez}»."] + por_que)
        estado = "no_alcanza" if duros else estado_juez
    recomendada = "propuesta" if estado == "claro" else None

    # ── LOS CONCEPTOS QUE EL JUEZ NO ESTUDIÓ (gancho, SPEC B) ──
    via_rev = next((v for v in (via_p, via_o) if v and v.get("prospera")), None)
    omitidos = None
    if via_rev is not None:
        omitidos = _conceptos_omitidos({
            "tipo_asunto": tipo_asunto, "resolvio_a_quo": _txt(rama_info.get("resolvio_a_quo")),
            "sentido": via_rev["sentido"], "sentido_global": sentido_motor,
            "conceptos_violacion": rama_info.get("conceptos_violacion"),
            "problemas": problemas, "contraste": contraste, "propuesta": resp,
            "rama_info": rama_info})

    tarjeta = {
        "formato": FORMATO, "estado_calculo": "listo", "huella": huella,
        "principal": principal,
        "vias": {"propuesta": via_p, "opuesta": via_o},
        "recomendada": recomendada, "estado": estado, "estado_por_que": por_que,
        "secundarios": secundarios, "independientes": independientes,
        "que_la_cambiaria": que_cambiaria,
        "tu_tribunal": _tu_tribunal(
            espejo, [p_preg, _txt(pral.get("pregunta_original"))],
            [x for j, q in enumerate(problemas) if j != p_idx
             for x in (_preg(q), _txt(q.get("pregunta_original")))]),
        "linea_corte": _linea_corte(material, internet, tribunal, circuito),
        "deliberacion": delib_ctx if delib else None,
        "conceptos_omitidos": omitidos,
        "avisos": list(dict.fromkeys(a for a in avisos if a)),
    }
    return _limpio(tarjeta)


def clave_de(tarjeta: dict) -> str:
    """Identifica una tarjeta armada: si no cambió, la marca no se reescribe."""
    base = json.dumps(tarjeta, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:20]


# ═══ QUÉ MARCAS DE LA FILA SE USAN ══════════════════════════════════════════

def elegir_marcas(fila: dict, huella: str, propuestas_fila=None, ahora=None) -> dict:
    """De las marcas de la fila —propuesta, global_propuesta, contraste,
    deliberacion—, las de ESTE adelanto (misma huella). Devuelve
    {"estado_calculo", "propuesta", "contraste", "deliberacion"}.

    La global es la ÚLTIMA que vio la pantalla (`global_propuesta`, 26-sep-2026)
    y manda sobre la de la propuesta calculada sola: si el secretario pidió la
    propuesta con contexto, ésa no queda en la marca «propuesta». Lo demás
    (propuestas por problema, contraste) sale de la marca «propuesta»; si no
    está, de la columna `propuestas` de la fila y del contraste adelantado.
    """
    import time as _time
    fila = fila if isinstance(fila, dict) else {}
    pm = fila.get("propuesta") if isinstance(fila.get("propuesta"), dict) else {}
    gp = fila.get("global_propuesta") if isinstance(fila.get("global_propuesta"), dict) else {}
    ct = fila.get("contraste") if isinstance(fila.get("contraste"), dict) else {}
    dl = fila.get("deliberacion") if isinstance(fila.get("deliberacion"), dict) else {}
    resp = {}
    if pm.get("huella") == huella and pm.get("estado") == "listo" \
            and isinstance(pm.get("respuesta"), dict) and pm["respuesta"].get("formato") == 2:
        resp = dict(pm["respuesta"])
    g = gp.get("global") if (gp.get("huella") == huella and isinstance(gp.get("global"), dict)) \
        else None
    if g is not None:
        resp["global"] = g
    if not resp:
        en_curso = pm.get("huella") == huella and pm.get("estado") == "en_curso"
        if en_curso:
            try:
                import taller_estado as _te
                vivo = not _te.abandonada(pm, (lambda: ahora) if ahora else _time.time)
            except Exception:                           # pragma: no cover
                vivo = True
            if vivo:
                return {"estado_calculo": "calculando", "propuesta": None,
                        "contraste": None, "deliberacion": None}
        return {"estado_calculo": "sin_propuesta", "propuesta": None, "contraste": None,
                "deliberacion": None}
    if not resp.get("propuestas") and propuestas_fila:
        resp["propuestas"] = [p for p in propuestas_fila if isinstance(p, dict)]
    contraste = resp.get("contraste")
    if not contraste and ct.get("huella") == huella and ct.get("estado") == "listo":
        contraste = list(ct.get("items") or [])
    return {"estado_calculo": "listo", "propuesta": resp, "contraste": contraste or [],
            "deliberacion": dl if dl.get("huella") == huella else None}
