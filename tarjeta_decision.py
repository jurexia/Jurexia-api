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
    las lecturas alrededor; aquí todo es puro y se prueba sin el arranque de la
    app (el arranque borra las cachés de Gemini). Desde la revisión del
    28-sep-2026 la tarjeta NO se guarda como marca: se reescribía el `estado`
    entero sin control de versión y podía pisar otras marcas.

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
# Vive en `fuerza_juridica.py` desde el rediseño del 29-sep-2026 (punto 3: una
# sola regla para quien decide y quien redacta). Se re-exporta con los mismos
# nombres: la deliberación y las pruebas los importan de aquí.
from fuerza_juridica import (  # noqa: E402,F401
    FUERZA_AISLADA_PROPIA, FUERZA_JURISPRUDENCIA_PROPIA, FUERZA_SENTENCIA_PROPIA,
    REGION_DEL_CIRCUITO, _NOMBRE_REGION, _organo, _region_de_tesis, _regiones_configuradas,
    _tipo, circuito_de, clave_de_tesis, designacion_de, fuerza_para_colegiado,
    region_de_clave, region_del_circuito)


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
    import fuerza_juridica as _fj
    return _fj.sello_perdio_vigencia(_vigencia(t))


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
            **({"por_confirmar": True} if f.get("por_confirmar") else {}),
            "vigencia": _etiqueta_vigencia(_vigencia(t)),
            "de_internet": bool(t.get("de_internet")), "en_acervo": True, "norma": None}


def _apoyo_norma(texto: str, en_acervo: bool | None = None) -> dict:
    return {"registro": None, "rubro": None, "instancia": None, "tipo": None,
            "fuerza": None, "fuerza_texto": None, "vigencia": None,
            "de_internet": False, "en_acervo": en_acervo, "norma": texto}


# ═══ LA NORMA, VERIFICADA CONTRA LOS PRECEPTOS DEL MATERIAL ═════════════════
# Revisión del 28-sep-2026 (AR 631/2025): una vía fundada sólo en la ley salía
# «no alcanza» («ninguna de las dos vías se apoya en un criterio del acervo
# verificado»), porque toda norma llegaba con `en_acervo: None`, también la del
# catálogo de la deliberación, que su código ya había verificado. Y ese freno
# es «duro»: anulaba al juez. La ley es el primer fundamento. Una norma cuenta
# como verificada si vino del catálogo de la deliberación o si su artículo y su
# ley casan con un precepto del material; si no, sigue sin contar.
_LEY_ABREVIADA = {
    "CPC": "PROCESAL CIVIL", "CPCF": "FEDERAL PROCEDIMIENTOS CIVILES",
    "CFPC": "FEDERAL PROCEDIMIENTOS CIVILES", "CC": "CIVIL", "CCF": "CIVIL FEDERAL",
    "CPEUM": "CONSTITUCION", "CFF": "FISCAL FEDERACION",
    "CNPP": "NACIONAL PROCEDIMIENTOS PENALES",
    "CNPCF": "NACIONAL PROCEDIMIENTOS CIVILES FAMILIARES"}
_LEY_SIN_SEÑA = {"CODIGO", "ESTADO", "ESTADOS", "LIBRE", "SOBERANO", "PARA", "LOCAL",
                 "UNIDOS", "MEXICANOS", "POLITICA", "ARTICULO", "ARTICULOS", "FRACCION",
                 "PARRAFO", "DICHO", "MISMO", "VIGENTE"}
_LEY_PROCESAL = {"PROCESAL", "PROCEDIMIENTO"}


def _raiz_ley(w: str) -> str:
    if len(w) > 5 and w.endswith("ES"):
        return w[:-2]
    if len(w) > 5 and w.endswith("S"):
        return w[:-1]
    return w


def _palabras_de_ley(texto: str) -> set:
    out = set()
    for w in re.findall(r"[A-ZÑ]+", _plano(texto)):
        if w in _LEY_ABREVIADA:
            out |= set(_LEY_ABREVIADA[w].split())
        elif len(w) >= 4 and w not in _LEY_SIN_SEÑA:
            out.add(_raiz_ley(w))
    return out


def norma_en_material(texto: str, normas) -> bool:
    """¿«art. 49 CPC Qro», «artículo 2294 del Código Civil local»… es un
    precepto del material (`material.normas`: cuerpo_legal, articulo)? Casa
    el número del artículo y al menos una palabra de la ley, y la ley procesal
    no se confunde con la sustantiva («Código Civil» ≠ «Código Procesal
    Civil»). Sin ley escrita no se verifica: se prefiere quedarse corto."""
    t = str(texto or "")
    i = re.search(r"\bart", t, re.I)
    if not i:
        return False
    nums = set(re.findall(r"\b(\d{1,4})\b", t[i.start():]))
    palabras_t = _palabras_de_ley(re.sub(r"\d+", " ", t))
    if not nums or not palabras_t:
        return False
    for n in (normas or []):
        if not isinstance(n, dict) or _txt(n.get("articulo")) not in nums:
            continue
        palabras_n = _palabras_de_ley(_txt(n.get("cuerpo_legal") or n.get("ley")))
        if bool(palabras_n & _LEY_PROCESAL) != bool(palabras_t & _LEY_PROCESAL):
            continue
        if (palabras_n & palabras_t) - _LEY_PROCESAL:
            return True
    return False


def _apoyo_precedente_propio(texto: str, fuerza_texto: str = "") -> dict:
    """Un asunto ya resuelto por ESTE tribunal que la deliberación propone
    aplicar (su catálogo lo rotula «O»). No es tesis ni tiene registro: se
    enseña como texto —la pantalla pinta así lo que no tiene registro— con su
    fuerza, y nunca como voto (fase_espejo.py, «lo que esto no es»)."""
    return {"registro": None, "rubro": None, "instancia": None, "tipo": None,
            "fuerza": "precedente_propio",
            "fuerza_texto": fuerza_texto or FUERZA_SENTENCIA_PROPIA,
            "vigencia": None, "de_internet": False, "en_acervo": None,
            "norma": f"{texto} de este tribunal (precedente propio)",
            "precedente_propio": texto}


def hidratar_apoyos(apoyos, catalogo: dict, tribunal: str = "", circuito: int = 0,
                    via: str = "", normas=None) -> tuple:
    """(apoyos del contrato, avisos). Cada apoyo del motor —«2026918», «el
    registro 170378», «art. 49 CPC Qro», o {"registro": …} de la
    deliberación— se verifica contra el catálogo del asunto:
      · con registro y en el acervo → la tesis, con rubro literal, fuerza y
        vigencia; si perdió la vigencia por completo, NO entra (va al aviso);
      · con registro que no está en el acervo → NO se pinta (va al aviso);
      · sin registro → una norma: `en_acervo` True si vino del catálogo de la
        deliberación (su código ya la verificó) o si casa con un precepto de
        `normas` (el material); si no, None (`norma_en_material`).
    """
    fuera, avisos, vistos = [], [], set()
    _de = f" de {via}" if via else ""
    for a in (apoyos or []):
        norma_verificada = False
        if isinstance(a, dict):
            norma_verificada = a.get("en_acervo") is True and not _txt(a.get("registro"))
            reg = _txt(a.get("registro"))
            m = _RX_REGISTRO.search(reg)
            reg = m.group(1) if m else ""
            # EL PRECEDENTE PROPIO DE LA DELIBERACIÓN (su «O»): sin registro y
            # sin norma. Antes caía en «llegó sólo con su identificador» y se
            # tiraba; es un asunto de este tribunal que ya estaba en su
            # catálogo cerrado, verificado allí contra el espejo.
            if not reg and _txt(a.get("precedente_propio")):
                k = "O:" + _plano(a["precedente_propio"])
                if k not in vistos:
                    vistos.add(k)
                    fuera.append(_apoyo_precedente_propio(_txt(a["precedente_propio"]),
                                                          _txt(a.get("fuerza_texto"))))
                continue
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
            fuera.append(_apoyo_norma(
                txt, True if (norma_verificada or norma_en_material(txt, normas)) else None))
    return fuera, avisos


# Una cantidad no es un registro: «$600000», «600000 pesos». Sin esto, la razón
# que habla de una condena se leería como una cita fuera del acervo.
_RX_REGISTRO_SUELTO = re.compile(r"(?<![$\d.,])\b(\d{6,7})\b(?!\s*(?:pesos|m\.?n\.?|m2|metros))",
                                 re.I)


def registros_sueltos(texto: str, catalogo: dict) -> list:
    """Las cifras de registro que un texto libre nombra y el acervo del asunto
    no tiene (para avisar: la razón del motor viaja tal cual al resolver)."""
    return [r for r in dict.fromkeys(_RX_REGISTRO_SUELTO.findall(str(texto or "")))
            if r not in catalogo]


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


def desenlace_de(tipo_asunto: str, resolvio_a_quo: str, sentido: str, *,
                 quien_recurre: str = "", sobresee_ademas: bool = False,
                 resolutivo_recurrida: str = "", procedencia: bool = False) -> tuple:
    """(puntos resolutivos previstos, nota | None) de un sentido, por código.

    AMPARO EN REVISIÓN QUE REVOCA UNA CONCESIÓN (el 631, vía contraria): se
    revoca y, antes de negar, el tribunal reasume jurisdicción y estudia los
    conceptos de violación que el juez no estudió (art. 93, fr. VI, de la Ley
    de Amparo); sólo si caen, se niega. Sin eso, «revoca y niega» es
    prematuro, y la nota lo dice.

    LA MISMA REGLA QUE EL DOCUMENTO (integración del 28-sep-2026): cuándo se
    reasume lo dice `tipos_asunto.reasuncion` —no se reasume si recurre la
    propia quejosa— y los puntos salen de `tipos_asunto.puntos_reasuncion`,
    los mismos con que se compone el resolutivo: el verbo del amparo va en
    HUECO porque depende de un estudio que aún no se hace, y si la recurrida
    también sobreseyó, la revocación se acota a la materia de la revisión. Con
    el resolutivo del juzgado (`resolutivo_recurrida`, la fuente de verdad de
    577c700) los puntos nombran a quien él amparó; sin él, «la parte quejosa».
    La deliberación usa esta misma función (`deliberacion.consecuencia_de`).

    REVISIÓN DEL 28-sep-2026 (AR 631/2025), tres desenlaces que salían mal:
      · RECURRE LA QUEJOSA Y GANA contra una concesión: salía «no ampara ni
        protege» a quien pedía más. Ahora la sentencia que corresponde la ampara
        (fr. V) y el verbo sobre la recurrida —revocar o modificar— va en hueco
        hasta que el estudio diga el alcance del vicio;
      · PROSPERA LA IMPROCEDENCIA (`procedencia`, el principal es de clase
        «procedencia»): se revoca y se sobresee, sin reasumir (fr. II);
      · LA VÍA QUE CONFIRMA UNA SENTENCIA MIXTA, con la autoridad o la tercera
        recurriendo: el sobreseimiento que nadie impugnó no es materia de la
        revisión —no se vuelve a decretar—, y la confirmación se acota a ella,
        como la revocación en la otra vía, con los nombres del resolutivo.
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
        q = _txt(quien_recurre).lower()
        rama = _ta.rama_revision(a, s, quien_recurre=q, procedencia=bool(procedencia))
        info = _ta.RAMAS_REVISION.get(rama) or {}
        puntos = [_rellenar(p) for p in (info.get("puntos") or [])]
        nota = info.get("aviso") or None
        res = _txt(resolutivo_recurrida)
        reas = _ta.reasuncion(a, s, quien_recurre=q, procedencia=bool(procedencia))
        # ¿LA SENTENCIA ERA MIXTA Y LA RECURRE QUIEN NO PIDIÓ EL AMPARO? Entonces
        # el sobreseimiento no es materia de la revisión, en las dos vías.
        mixta_ajena = (bool(sobresee_ademas) or a == "sobresee_concede") and q != "quejoso"
        confirma = ((rama == "confirma_concede" and (res or mixta_ajena))
                    or (rama == "confirma_sobresee_concede" and mixta_ajena))
        if confirma:
            puntos = [_rellenar(p) for p in _ta.puntos_confirma_concede(res, parcial=mixta_ajena)]
            if mixta_ajena:
                nota = ("La sentencia recurrida también sobreseyó respecto de algún acto: si nadie "
                        "lo impugnó, ese sobreseimiento queda firme y no es materia de la revisión; "
                        "la confirmación se acota a ella.")
        elif rama == "revoca_fondo_concede" and q == "quejoso" and a in ("concede",):
            # LA QUEJOSA RECURRE SU CONCESIÓN Y GANA: no se le niega lo que ya
            # tenía (fr. V). Ver `tipos_asunto.puntos_quejosa_mejora`.
            puntos = [_rellenar(p) for p in _ta.puntos_quejosa_mejora(res)]
            nota = ("Recurre la propia quejosa contra una concesión: si su agravio prospera, no se "
                    "le niega en el fondo el amparo que ya tenía —su recurso no puede empeorarle "
                    "la situación—; se revoca o se modifica la sentencia, según el alcance del "
                    "vicio, y la que corresponde la ampara con el alcance o los efectos que "
                    "resulten del estudio (art. 93, fr. V, de la Ley de Amparo). Una causa de "
                    "improcedencia, que se examina de oficio, es lo único que llevaría a sobreseer.")
        elif rama == "revoca_sobresee":
            nota = ("Prospera la improcedencia que alegó quien recurre: se revoca y se sobresee "
                    f"({info.get('fundamento', '')}). No se reasume jurisdicción ni se estudian "
                    "conceptos de violación: la fracción VI del artículo 93 es para los agravios "
                    "de fondo.")
        elif rama == "revoca_fondo_niega" and reas == "concesion":
            # LO FIRME, CON LA REGLA DEL DOCUMENTO (AR 631/2025, al generar en
            # pantalla): si recurre quien no es la quejosa, el sobreseimiento de
            # la mixta nadie lo impugnó y va PRIMERO, como en el circuito.
            _firme_t = _ta.sobreseimiento_firme(bool(sobresee_ademas) or a == "sobresee_concede", q)
            puntos = [_rellenar(p) for p in _ta.puntos_reasuncion(
                "", res, firme=_firme_t, parcial=bool(sobresee_ademas))]
            nota = ("Al revocar la concesión, el tribunal reasume jurisdicción y estudia los "
                    "conceptos de violación que el Juzgado no estudió (art. 93, fr. VI, de la "
                    "Ley de Amparo); sólo si caen, se niega. Si alguno prospera, se concede "
                    "por esa razón.")
            if _firme_t:
                nota += (" La sentencia recurrida también sobreseyó y quien recurre no es la "
                         "quejosa: ese sobreseimiento no es materia de la revisión y queda firme "
                         "(salvo que la quejosa se haya adherido para combatirlo).")
            elif sobresee_ademas:
                nota += (" La sentencia recurrida también sobreseyó: si nadie lo impugnó, ese "
                         "sobreseimiento queda firme.")
        elif rama == "revoca_fondo_niega" and res:
            # SIN REASUNCIÓN: revocar la concesión niega lo que ella concedió,
            # con las palabras del juzgado (577c700). Desde la revisión del
            # 28-sep-2026 la quejosa recurrente ya no llega aquí (su recurso
            # fundado la ampara, arriba); queda como respaldo.
            puntos = [_rellenar(p) for p in (_ta.puntos_revoca_concesion(res) or info.get("puntos") or [])]
        elif rama.startswith("revoca_sobreseimiento"):
            # El sentido del AMPARO sale del estudio de los conceptos, que el
            # tribunal hace por primera vez: no se supone ninguno.
            mixta_propia = a == "sobresee_concede" and q == "quejoso"
            primero = _ta.REVOCA_PARCIAL if mixta_propia else (puntos[0] if puntos else "")
            puntos = [primero, f"SEGUNDO. La Justicia de la Unión {HUECO} a la parte quejosa, "
                               f"según resulte del estudio de los conceptos de violación."] \
                if primero else []
            nota = ("Levantado el sobreseimiento, el tribunal asume jurisdicción y estudia los "
                    f"conceptos de violación por primera vez ({info.get('fundamento', '')}); el "
                    "sentido del amparo sale de ese estudio.")
            if mixta_propia:
                # LA QUEJOSA RECURRE UNA SENTENCIA MIXTA: combate el
                # sobreseimiento; la concesión por los demás actos es suya y
                # nadie la recurrió.
                nota += (" La concesión por los demás actos no es materia de su recurso y queda "
                         "firme: la revocación se acota a la materia de la revisión.")
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
    de esa vía, sólo si traen una calificación que el árbol reconoce.

    LA FORMA REAL DE LA DELIBERACIÓN (integración del 28-sep-2026): el
    documento de `deliberacion.deliberar` guarda lo que escribió cada abogado
    en `secundarios_escritos`, con la suerte como {sentido, razon}; el
    contrato provisional que se leía aquí decía `secundarios` con la suerte en
    texto, y con el documento real la tarjeta no veía ninguna. Se leen las dos
    formas."""
    out = {}
    try:
        import arbol_decision as _ad
    except Exception:                                   # pragma: no cover
        return out
    for x in list(v.get("secundarios_escritos") or []) + list(v.get("secundarios") or []):
        if not isinstance(x, dict):
            continue
        try:
            n = int(x.get("numero"))
        except (TypeError, ValueError):
            continue
        if n in out:
            continue
        su = x.get("suerte")
        if isinstance(su, dict):
            s, r = _ad._sentido_valido(su.get("sentido")), _txt(su.get("razon"))
        else:
            s = _ad._sentido_valido(x.get("sentido"))
            r = ""
            if not s:
                s, r = _ad._leer_suerte(_txt(su))
        if s:
            out[n] = (s, _txt(x.get("razon")) or r)
    return out


def _catalogo_de_la_deliberacion(delib: dict) -> dict:
    """{registro: tesis} del catálogo cerrado de la deliberación. Sus tesis
    salieron de la búsqueda sobre la pregunta decisiva —el acervo, no el
    modelo— y pasaron allí su verificación (vigencia, lectura, cupo); muchas no
    están en el material del adelanto, que se buscó con la pregunta del
    agravio. Sin sumarlas, la tarjeta las tiraba como «fuera del acervo del
    asunto» y la vía del abogado se quedaba sin apoyos. El documento las
    guarda sin su texto; para la tarjeta basta el rubro literal."""
    out = {}
    for e in ((delib or {}).get("catalogo") or {}).values():
        if isinstance(e, dict) and e.get("clase") == "tesis" and _txt(e.get("registro")):
            out.setdefault(_txt(e["registro"]), {
                "registro": _txt(e["registro"]), "rubro": _txt(e.get("rubro")),
                "instancia": _txt(e.get("instancia")), "tipo": _txt(e.get("tipo")),
                "clave": _txt(e.get("clave")), "de_internet": bool(e.get("de_internet"))})
    return out


def _normas_de_la_deliberacion(delib: dict) -> list:
    """Los preceptos del catálogo cerrado de la deliberación, con la forma de
    `material.normas` (cuerpo_legal, articulo), para verificar las normas que
    nombran las vías (revisión del 28-sep-2026)."""
    return [{"cuerpo_legal": _txt(e.get("ley")), "articulo": _txt(e.get("articulo"))}
            for e in ((delib or {}).get("catalogo") or {}).values()
            if isinstance(e, dict) and e.get("clase") == "norma" and _txt(e.get("articulo"))]


# ═══ LOS CONCEPTOS QUE EL JUEZ NO ESTUDIÓ (SPEC B) ════════════════════════════
#
# La función de SPEC B —`fase_rama.conceptos_omitidos`, pura, sin modelo— dice
# si, al prosperar el recurso, el tribunal tiene que estudiar conceptos de
# violación que nadie estudió (art. 93, frs. I, V y VI) y si ya los tenemos
# (los aportó el secretario, o están en la demanda de las constancias o en la
# recurrida). Hasta la integración (28-sep-2026) aquí había un gancho perezoso
# que la buscaba por nombre y la llamaba con los argumentos que su firma
# pidiera; con la función real, ese gancho le pasaba el sentido DEL MOTOR, y en
# el 631 —motor «infundado», la vía que revoca es la contraria— contestaba que
# no hacía falta nada. Ahora se la llama directamente con el sentido de la vía
# que prospera, lo que hizo el juzgado, quién recurre y las fases donde buscar.


def _conceptos_omitidos(rama_info: dict, sentido_via: str, fases=None):
    import fase_rama as _fr
    res = _fr.conceptos_omitidos(rama_info, sentido_via, fases)
    return _limpio(res) if isinstance(res, dict) else None


# ═══ EL ESTADO, DETERMINISTA ═════════════════════════════════════════════════

def _vigentes_que_obligan(apoyos: list) -> set:
    return {a["registro"] for a in (apoyos or [])
            if a.get("registro") and a.get("fuerza") == "obliga"}


# CÓMO SE LLAMAN LAS DOS COLUMNAS EN PANTALLA (revisión del 28-sep-2026, AR
# 631/2025). Con «claro» son la propuesta y la contraria; con «reñido» o «no
# alcanza» la pantalla las rotula «Vía A» y «Vía B» y promete que ninguna se
# recomienda: las razones del estado no pueden hablar entonces de «la
# propuesta», que en pantalla no existe.
NOMBRES_CLARO = ("la propuesta", "la vía contraria")
NOMBRES_NEUTROS = ("la vía A", "la vía B")


def _may(t: str) -> str:
    return t[:1].upper() + t[1:]


def _estado(prop: dict | None, opu: dict | None, contraste: dict | None,
            discrepa: dict | None, indispensables: list, hay_global: bool,
            nombres: tuple = NOMBRES_CLARO) -> tuple:
    """(estado, razones) SIN deliberación. Nunca un porcentaje. `nombres`:
    cómo se llaman en pantalla la columna de la izquierda y la de la derecha
    (`NOMBRES_CLARO` | `NOMBRES_NEUTROS`); `armar` vuelve a pedir las razones
    con los neutros cuando el estado final no es «claro».

    «no_alcanza»: no hay sentido para el asunto, falta una constancia que el
      motor declaró indispensable, o ninguna vía se apoya en un criterio del
      acervo verificado y vigente ni en un precepto verificado (la ley es el
      primer fundamento: `hidratar_apoyos`).
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
    P, O = nombres
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
                       "y vigente ni en un precepto verificado del material.")
    if razones:
        return "no_alcanza", razones

    pros = bool(prop and prop.get("prospera"))
    ob_p = _vigentes_que_obligan((prop or {}).get("apoyos"))
    ob_o = _vigentes_que_obligan((opu or {}).get("apoyos"))
    solo_p, solo_o = ob_p - ob_o, ob_o - ob_p
    comunes = sorted(ob_p & ob_o)
    _comparten = (f"; las dos comparten {', '.join(comunes[:3])}" if comunes else "")

    cierra = None                  # True: a favor de la propuesta; False: en contra
    if contraste:
        v = _txt(contraste.get("veredicto_previo")).lower()
        if v in ("inoperante", "fundado_pero_insuficiente"):
            cierra = not pros
            if v == "inoperante":
                razones.append("El contraste dice que el planteamiento no combate la razón toral "
                               "del fallo" + (f"; {P} lo desestima." if not pros else
                                               f", y {P} lo hace prosperar."))
            else:
                razones.append("El contraste dice que el fallo se sostiene por otra "
                               "consideración" + (f"; {P} lo desestima." if not pros else
                                                   f", y {P} lo hace prosperar."))
        else:
            razones.append("El contraste no cierra el punto: el planteamiento combate la razón "
                           "toral y hay que decidir si tiene razón.")
    else:
        razones.append("No hay contraste del principal.")

    # QUÉ INVOCA CADA UNA, SIN «LA OTRA NO» (revisión del 28-sep-2026): contar
    # como diferencia lo que la otra vía no repite, cuando la otra también
    # invoca un criterio que obliga, empujaba hacia una columna.
    if solo_p and not solo_o:
        razones.append(f"{_may(P)} invoca además criterios que obligan a este tribunal: "
                       + ", ".join(sorted(solo_p)[:4]) + _comparten + ".")
    elif solo_o and not solo_p:
        razones.append(f"{_may(O)} invoca además criterios que obligan a este tribunal: "
                       + ", ".join(sorted(solo_o)[:4]) + _comparten + ".")
    elif solo_p and solo_o:
        razones.append(f"Cada vía invoca criterios que obligan a este tribunal ({P}: "
                       + ", ".join(sorted(solo_p)[:3]) + f"; {O}: " + ", ".join(sorted(solo_o)[:3])
                       + _comparten + "): ese escalón no las separa.")
    elif comunes:
        razones.append("Las dos vías invocan los mismos criterios que obligan a este tribunal ("
                       + ", ".join(comunes[:3]) + "): ese escalón no las separa.")
    else:
        razones.append("Ninguna vía invoca un criterio que obligue a este tribunal.")
    if discrepa:
        razones.append("El motor tomó como problema que decide otro distinto del principal.")
    if opu is None:
        razones.append(f"El motor no encontró cómo sostener {O} con el acervo.")

    claro = (cierra is True and bool(solo_p) and not solo_o and not discrepa)
    if cierra is False or not claro:
        return "reñido", razones
    return "claro", razones


# ═══ LAS VÍAS ════════════════════════════════════════════════════════════════

def _procedencia(rama_info: dict) -> bool:
    """¿El principal —el agravio que prospera en la vía que lo hace
    prosperar— es de procedencia (improcedencia o sobreseimiento)?"""
    return (bool(rama_info.get("procedencia"))
            or _txt(rama_info.get("clase_principal")).lower() == "procedencia")


def _favorece(sentido: str, rama_info: dict):
    """¿Esta vía le da la razón a quien reclama el derecho? True | False | None
    (no consta), con el CARÁCTER de quien recurre (`dialogo_constitucional`,
    SPEC B): si recurre la tercera interesada y prospera, pierde la quejosa."""
    try:
        import dialogo_constitucional as _dc
        import tipos_asunto as _ta
        tipo = _ta.normalizar(rama_info.get("tipo_asunto", "") or "")
        return _dc.favorece_a_la_persona(
            sentido, tipo, _txt(rama_info.get("recurrente")),
            tipo in ("amparo_revision", "queja", "revision_fiscal"),
            papel=_txt(rama_info.get("quien_recurre")))
    except Exception:                                   # pragma: no cover
        return None


# El efecto que escribió el motor y que es un REENVÍO: choca con el art. 93, fr.
# VI (el tribunal reasume jurisdicción, no devuelve el asunto).
_RX_EFECTO_REENVIO = re.compile(
    r"\bnuev[oa]\s+(?:decisi[óo]n|resoluci[óo]n|sentencia|pronunciamiento)|\breenv[íi]|"
    r"\bdevolver\b|\bdevuelv|\bdicte\s+otra\b|\bemita\s+otra\b|\breponer\b|\breposici[óo]n\b|"
    r"\bpara\s+que\s+(?:el\s+)?(?:juzgado|juez|a\s+quo)\b", re.I)


def _via(sentido: str, razon: str, efecto: str, apoyos_crudos, catalogo: dict,
         rama_info: dict, protectora: dict | None, avisos: list, etiqueta: str,
         interpretacion=None, cadena=None, objecion=None, normas=None) -> dict:
    tribunal = _txt(rama_info.get("tribunal"))
    circuito = int(rama_info.get("circuito") or 0) or circuito_de(tribunal)
    apoyos, av = hidratar_apoyos(apoyos_crudos, catalogo, tribunal, circuito, etiqueta, normas)
    avisos.extend(av)
    sueltos = registros_sueltos(razon, catalogo)
    if sueltos:
        avisos.append(f"La razón de {etiqueta} nombra registros que no están en el acervo del "
                      f"asunto ({', '.join(sueltos[:4])}): no se tienen por citados.")
    a_quo = rama_info.get("resolvio_a_quo") or rama_info.get("que_hizo") or ""
    puntos, nota = desenlace_de(rama_info.get("tipo_asunto", ""), a_quo,
                                sentido, quien_recurre=_txt(rama_info.get("quien_recurre")),
                                sobresee_ademas=bool(rama_info.get("sobresee_ademas")),
                                resolutivo_recurrida=_txt(rama_info.get("resolutivo_recurrida")),
                                procedencia=_procedencia(rama_info))
    # EL EFECTO DEL MOTOR QUE CONTRADICE EL DESENLACE (revisión del 28-sep-2026,
    # AR 631/2025: en la 462 la nota decía «reasume jurisdicción» y la línea de
    # debajo, «exigiría una nueva decisión», un reenvío). Si esta vía revoca una
    # concesión, la consecuencia es la calculada por código; el efecto que
    # habla de devolver o de una nueva decisión no se enseña como consecuencia:
    # se guarda como lo que escribió el motor y se avisa.
    efecto_motor = _txt(efecto)
    try:
        import tipos_asunto as _ta
        reas = (_ta.reasuncion(_txt(a_quo).lower(), _txt(sentido).lower().replace(" ", "_"),
                               quien_recurre=_txt(rama_info.get("quien_recurre")),
                               procedencia=_procedencia(rama_info))
                if _ta.normalizar(rama_info.get("tipo_asunto", "") or "") == "amparo_revision"
                else "")
    except Exception:                                   # pragma: no cover
        reas = ""
    efecto_visible = efecto_motor
    if reas == "concesion" and efecto_motor and _RX_EFECTO_REENVIO.search(efecto_motor):
        efecto_visible = ""
        avisos.append(f"El efecto que escribió el motor para {etiqueta} habla de devolver el "
                      f"asunto o de una nueva decisión, y eso choca con el artículo 93, fracción "
                      f"VI, de la Ley de Amparo: al revocar una concesión el tribunal reasume "
                      f"jurisdicción, sin reenvío. No se enseña como consecuencia de esa vía; la "
                      f"consecuencia es el desenlace calculado.")
    vp = None
    if isinstance(protectora, dict) and _txt(protectora.get("sentido")) \
            and _prospera(protectora["sentido"]) == _prospera(sentido):
        # SÓLO A FAVOR DE LA PERSONA (revisión del 28-sep-2026; SPEC B §4): una
        # propuesta guardada antes del arreglo del papel del recurrente podía
        # colgar la lectura protectora de la vía que, recurriendo la tercera
        # interesada, le quita el amparo a la quejosa (tarjeta real de la 462).
        # Se cuelga si la vía favorece a quien reclama el derecho o no consta;
        # si consta que no, va al aviso.
        if _favorece(sentido, rama_info) is False:
            avisos.append(f"La propuesta guardada ofrecía una lectura protectora (pro persona o "
                          f"interpretación conforme) en {etiqueta}, que no favorece a quien reclama "
                          f"el derecho: no se enseña. Esas lecturas operan sólo a favor de la "
                          f"persona.")
        else:
            vp = _limpio(protectora)
            # LOS APOYOS DE LA VÍA PROTECTORA NO LOS VERIFICABA NADIE (`revisar`
            # sólo mira la global y la alternativa). En el 631 traía dos
            # registros que no están en el acervo del asunto. Se quedan los
            # comprobados, con la misma forma de hoy —registros en texto— para
            # la pantalla que ya la pinta, y el detalle aparte.
            ap_vp, av_vp = hidratar_apoyos(protectora.get("apoyos"), catalogo, tribunal,
                                           circuito, "la vía protectora", normas)
            avisos.extend(av_vp)
            vp["apoyos"] = [a["registro"] or a["norma"] for a in ap_vp]
            vp["apoyos_detalle"] = ap_vp
    return {"sentido": _txt(sentido), "prospera": _prospera(sentido), "razon": _txt(razon),
            "efecto": efecto_visible,
            **({"efecto_motor": efecto_motor} if efecto_visible != efecto_motor else {}),
            "desenlace": puntos, "desenlace_nota": nota,
            "interpretacion": _txt(interpretacion) or None,
            "cadena": _limpio(cadena) if isinstance(cadena, dict) else None,
            "objecion": objecion if (isinstance(objecion, dict)
                                     and (objecion.get("de_la_otra_via")
                                          or objecion.get("respuesta"))) else None,
            "apoyos": apoyos, "via_protectora": vp}


def _soluciones_para_tarjeta(delib: dict, id_propuesta, id_contraria) -> list:
    """La lista de soluciones de la deliberación, resumida para la pantalla.
    [] si la deliberación no las trae (sin la bandera, como siempre)."""
    sols = [s for s in (delib or {}).get("soluciones") or [] if isinstance(s, dict)]
    if not sols:
        return []
    vias = (delib or {}).get("vias") or {}
    por_sol = {(v or {}).get("solucion"): k for k, v in vias.items() if isinstance(v, dict)}
    fallas = {}
    for p in ((delib or {}).get("juez") or {}).get("pasadas") or []:
        for via, f in ((p or {}).get("fallas") or {}).items():
            if isinstance(f, dict) and f.get("falla") and via not in fallas:
                fallas[via] = f
    out = []
    for s in sols:
        rv = s.get("revision") or {}
        f = fallas.get(por_sol.get(s.get("id")))
        out.append({
            "id": s.get("id"), "prospera": bool(s.get("prospera")), "sentido": _txt(s.get("sentido")),
            "tipo_efecto": _txt(s.get("tipo_efecto")), "rama": _txt(s.get("rama")),
            "desenlace": [_txt(x) for x in (s.get("desenlace") or [])][:3],
            "resumen": _txt(s.get("resumen")) or _txt((s.get("conclusion") or {}).get("texto")),
            "revision": {"estado": _txt(rv.get("estado")) or "sin_revisar",
                         "avisos": [_txt(a) for a in (rv.get("avisos") or [])][:3]},
            "falla": ({"que": _txt(f.get("falla")), "fatal": bool(f.get("fatal"))} if f else None),
            "sostenible": s.get("sostenible", True) is not False,
            "papel": ("propuesta" if s.get("id") == id_propuesta
                      else "contraria" if s.get("id") == id_contraria else None),
        })
    return out


def _via_de_deliberacion(v: dict, glob: dict, catalogo: dict, rama_info: dict,
                         avisos: list, etiqueta: str, normas=None) -> dict:
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
                cadena=cadena, objecion=obj, normas=normas)


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
                    "pdf_url": _txt(f.get("pdf_url")) or None,
                    # LO QUE LA FILA DE LA OAJ TRAE Y AQUÍ SE TIRABA (29-sep-2026):
                    # sin la cota, «57% o más» se leía como 57 exacto; sin la
                    # fuente, una coincidencia por el TEMA del asunto pasaba por
                    # una del planteamiento; sin la pregunta no hay qué comparar;
                    # sin el enlace no hay cómo abrirla. Las del espejo viejo no
                    # los traen y salen en blanco.
                    "cota_inferior": bool(f.get("cota_inferior")),
                    "fuente": _txt(f.get("fuente")) or None,
                    "pregunta": _txt(f.get("pregunta")) or None,
                    "autoridad": _txt(f.get("autoridad")) or None,
                    "enlace_oaj": _txt(f.get("enlace_oaj")) or None})
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
            "conceptos_omitidos": None, "ficha": None, "avisos": []}


def _provisional(material, recomendada, estado, por_que) -> dict:
    """LA CONSULTA PROVISIONAL NO RECOMIENDA (rediseño, etapa 2, 29-sep-2026).
    Si venció la espera de la pregunta decisiva y falta la búsqueda de su
    figura, la tarjeta enseña las dos vías igual —el secretario puede elegir
    una o pedir al motor que redacte—, pero no rotula ninguna como
    recomendada: la recomendación dependería de un análisis que no se hizo."""
    ce = _get(material, "consulta_estado") or {}
    try:
        import contexto_taller as _ct_p
        _on = _ct_p.rediseno("consulta_provisional")
    except Exception:
        _on = False
    if _on and isinstance(ce, dict) and ce.get("estado") == "provisional":
        falta = ", ".join(ce.get("faltan") or []) or "parte de la consulta"
        return {"recomendada": None, "estado": "no_alcanza",
                "estado_por_que": [f"Consulta PROVISIONAL: falta {falta} (la pregunta decisiva no "
                                   f"llegó a tiempo). Se completa sola al volver a proponer."]
                                  + list(por_que or [])}
    return {"recomendada": recomendada, "estado": estado, "estado_por_que": por_que}


def armar(propuesta_guardada, material, problemas_fase3, contraste=None, espejo=None,
          internet=None, rama_info=None, deliberacion=None, *, propuestas_motor=None,
          fases=None, decisiva=None) -> dict:
    """La tarjeta (formato 1 de contrato_tarjeta.md), sin modelo.

    `propuesta_guardada`: la respuesta de /taller/proponer (formato 2):
      global, propuestas, contraste. `material`: el Material de la sesión o su
      forma guardada (tesis, espejo). `problemas_fase3`: los de la fase 3, en
      su orden. `contraste`: la lista por planteamiento (si None, la de la
      propuesta). `espejo`: si None, el del material. `internet`: ses/marca
      {pistas, resumen, …}. `rama_info`: lo de `redactor_adelanto.info_de_rama`
      —tipo_asunto, que_hizo, quien_recurre, conceptos_violacion (el TEXTO que
      aportó el secretario), sobresee_ademas— más tribunal, circuito?,
      resolvio_a_quo?, resolutivo_recurrida?, huella?, necesita_conceptos?,
      ficha? (FICHA del contrato, de `ficha_procesal.para_tarjeta`).
      `deliberacion`: la marca «deliberacion» (SPEC C2) o None.
      `propuestas_motor`: las propuestas con `sentido_propio`/`razon_propia`
      (lo que el motor propuso antes del árbol; viajan en la columna
      `propuestas` de la fila); si None, las de la respuesta.
      `fases`: las del adelanto, donde `fase_rama.conceptos_omitidos` busca
      la demanda entre las constancias o la recurrida que los transcribe.
      `decisiva`: el documento de `pregunta_decisiva` (SPEC E3); si None, el
      que viaja con el material. `principal.pregunta` es la cuestión decisiva
      y `principal.pregunta_recurrida`, la de la fase 3, aparte.
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
    normas = _lista(material, "normas")
    avisos: list = []

    # ── EL PRINCIPAL ──
    p_idx, jer_de = indice_principal(problemas)
    p_num = p_idx + 1
    pral = problemas[p_idx]
    p_preg = _preg(pral)
    # LA CLASE DEL PRINCIPAL DECIDE SI, AL PROSPERAR, SE SOBRESEE (revisión del
    # 28-sep-2026): un agravio de procedencia de la autoridad o de la tercera
    # no reasume nada (art. 93, fr. II). Manda el principal de esta tarjeta.
    if _txt(pral.get("clase")):
        rama_info["clase_principal"] = _txt(pral.get("clase")).lower()
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
    # LA JURIMETRÍA NO VA EN LA TARJETA (revisión del 28-sep-2026, AR 631/2025).
    # `prediccion.frase` es «Infundado (75% de 12 sentencias del acervo)» y la
    # pantalla la pintaba encima de las dos columnas «del mismo peso»: un
    # porcentaje que inclina hacia una vía, con un motor que acierta 50% contra
    # el 54% de «siempre niega» (sesgo de automatización medido). La tarjeta no
    # enseña porcentajes. Si el dato vuelve —sin cifra relativa, rotulado como
    # dato que no decide o plegado tras un clic— lo decide David (SPEC C
    # mencionaba la predicción). Sigue en la propuesta, donde estaba.
    # LA PREGUNTA DEL ENCABEZADO ES LA QUE DECIDE (SPEC E3, AR 631/2025): la
    # recurrida preguntó por la cosa juzgada y lo que decide es si el
    # adquirente puede sustituirse en la ejecución. La de la fase 3 va aparte
    # («así lo planteó la recurrida»); `p_preg` sigue emparejando propuesta,
    # espejo y secundarios, que se guardaron con ella.
    try:
        import pregunta_decisiva as _pd_t
        _dec_t = _pd_t.vigente(decisiva if decisiva is not None else _get(material, "decisiva"),
                               problemas)
        _enc = _pd_t.para_tarjeta(_dec_t, p_preg)
    except Exception:
        _enc = {"pregunta": p_preg, "pregunta_recurrida": None, "figura": None}
    principal = {"numero": p_num, "pregunta": _enc["pregunta"],
                 "pregunta_recurrida": _enc["pregunta_recurrida"], "figura": _enc["figura"],
                 "clase": _txt(pral.get("clase")) or None,
                 "jerarquia_de": jer_de,
                 "por_que_principal": _txt(ctx.get("tema_principal")) or None,
                 "discrepa_motor": discrepa,
                 "contraste": _contraste_de(contraste, p_num),
                 "prediccion": None}

    # ── LAS VÍAS ──
    hay_global = bool(glob and glob.get("alcanza", True) is not False and _txt(glob.get("sentido")))
    sentido_motor = _txt(glob.get("sentido")) if hay_global else ""
    delib = _payload_deliberacion(deliberacion)
    if delib:
        for reg, t in _catalogo_de_la_deliberacion(delib).items():
            catalogo.setdefault(reg, t)
        normas = list(normas) + _normas_de_la_deliberacion(delib)
    alt = glob.get("alternativa") if isinstance(glob.get("alternativa"), dict) else {}
    recomendada = None
    estado_juez = ""
    suertes_delib = {"propuesta": {}, "opuesta": {}}
    delib_ctx = None
    via_p = via_o = None
    if delib:
        vias_d = {str(k).upper(): v for k, v in (delib.get("vias") or {}).items()
                  if isinstance(v, dict) and _txt(v.get("sentido"))}
        # LA COLUMNA DE LA IZQUIERDA: la recomendada; si es reñido, la vía en
        # que coincidieron las dos pasadas del juez (`inclinacion`, que el
        # documento real guarda aparte y sólo sirve para el ORDEN, nunca como
        # recomendación); si tampoco, la que coincide con la del motor.
        rec = _txt(delib.get("recomendada")).upper() or _txt(delib.get("inclinacion")).upper()
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
                                         "la vía propuesta", normas)
            via_o = (_via_de_deliberacion(vias_d[otra], glob, catalogo, rama_info, avisos,
                                          "la vía contraria", normas) if otra else None)
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
            # LAS SOLUCIONES POSIBLES (rediseño, etapa 3): sólo si la
            # deliberación las trae (bandera «soluciones_por_desenlace»). Las
            # dos columnas siguen siendo A y B; esto es la lista completa, con
            # la revisión por código y la falla que vio el juez en cada una.
            _sols = _soluciones_para_tarjeta(delib, vias_d.get(rec, {}).get("solucion"),
                                             vias_d.get(otra, {}).get("solucion") if otra else None)
            if _sols:
                delib_ctx["soluciones"] = _sols
        else:
            delib = None
    if not (delib and delib_ctx):
        delib = None
        if hay_global:
            via_p = _via(sentido_motor, _txt(glob.get("razon")), _txt(glob.get("efecto")),
                         glob.get("apoyos"), catalogo, rama_info, glob.get("via_protectora"),
                         avisos, "la vía propuesta", interpretacion=glob.get("interpretacion"),
                         normas=normas)
            if _txt(alt.get("sentido")) and _prospera(alt["sentido"]) != _prospera(sentido_motor):
                via_o = _via(_txt(alt.get("sentido")), _txt(alt.get("razon")),
                             _txt(alt.get("efecto")), alt.get("apoyos"), catalogo, rama_info,
                             glob.get("via_protectora"), avisos, "la vía contraria",
                             interpretacion=alt.get("interpretacion"), normas=normas)
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
                         "la propuesta del principal", normas=normas)
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
                                     f"el problema {i + 1}", normas)
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
    _args_estado = (via_p, via_o, principal["contraste"], discrepa, indispensables,
                    hay_global or bool(delib))
    estado, por_que = _estado(*_args_estado)
    if delib and estado_juez:
        # EL JUEZ MANDA, con dos frenos que no se discuten: sin constancia
        # indispensable o sin un solo apoyo verificado, no alcanza aunque el
        # juez diga otra cosa (la verificación E es de código).
        duros = estado == "no_alcanza"
        estado = "no_alcanza" if duros else estado_juez
    if estado != "claro":
        # CON LOS NOMBRES DE LA PANTALLA: sin «claro» las columnas son la vía A
        # y la vía B y ninguna se rotula propuesta (revisión del 28-sep-2026).
        por_que = _estado(*_args_estado, nombres=NOMBRES_NEUTROS)[1]
    if delib and estado_juez:
        por_que = [f"Deliberación: el juez la marcó «{estado_juez}»."] + por_que
    recomendada = "propuesta" if estado == "claro" else None

    # ── LOS CONCEPTOS QUE EL JUEZ NO ESTUDIÓ (SPEC B) ──
    # Con el sentido de LA VÍA QUE PROSPERA, sea la propuesta o la contraria:
    # en el 631 el motor propuso «infundado» y la vía que revoca es la otra.
    via_rev = next((v for v in (via_p, via_o) if v and v.get("prospera")), None)
    omitidos = None
    if via_rev is not None:
        _ri = dict(rama_info)
        _ri.setdefault("que_hizo", _txt(rama_info.get("resolvio_a_quo")))
        # `conceptos_violacion` es el TEXTO que aportó el secretario: un
        # booleano se leería como «los aportó» (str(True)).
        if not isinstance(_ri.get("conceptos_violacion"), str):
            _ri["conceptos_violacion"] = ""
        omitidos = _conceptos_omitidos(_ri, via_rev["sentido"], fases)
        # «POR CONFIRMAR» (revisión del 28-sep-2026): la recurrida no dice que
        # quedaran conceptos sin estudiar. La pantalla no bloquea (lee
        # `hacen_falta === true`), pero el secretario tiene que saberlo.
        if isinstance(omitidos, dict) and omitidos.get("hacen_falta") == "por_confirmar":
            avisos.append(_txt(omitidos.get("por_que")))
    # NO CONSTA QUIÉN RECURRE (revisión del 28-sep-2026, AR 631/2025): sin prueba
    # de que recurra la quejosa, `papel_del_recurrente` ya no contesta
    # «quejoso»; la fr. VI se aplica y el diálogo no toma dirección. Se dice.
    if (_txt(tipo_asunto).lower() == "amparo_revision" and "quien_recurre" in rama_info
            and not _txt(rama_info.get("quien_recurre"))):
        avisos.append("No consta quién recurre —la quejosa, la tercera interesada o la "
                      "autoridad—: el desenlace se calcula como si no fuera la quejosa (se "
                      "reasume jurisdicción si se revoca una concesión) y la vía que favorece a "
                      "la persona no se da por sabida. Escribe quién recurre en el encargo.")

    tarjeta = {
        "formato": FORMATO, "estado_calculo": "listo", "huella": huella,
        "principal": principal,
        "vias": {"propuesta": via_p, "opuesta": via_o},
        **_provisional(material, recomendada, estado, por_que),
        "secundarios": secundarios, "independientes": independientes,
        "que_la_cambiaria": que_cambiaria,
        "tu_tribunal": _tu_tribunal(
            espejo, [p_preg, _txt(pral.get("pregunta_original"))],
            [x for j, q in enumerate(problemas) if j != p_idx
             for x in (_preg(q), _txt(q.get("pregunta_original")))]),
        "linea_corte": _linea_corte(material, internet, tribunal, circuito),
        "deliberacion": delib_ctx if delib else None,
        "conceptos_omitidos": omitidos,
        # LA FICHA PROCESAL (SPEC_E2, 28-sep-2026; contrato FICHA): quién
        # promovió, quién recurre y con qué carácter, qué resolvió el juzgado
        # por acto, qué es materia y qué quedó firme. La arma quien llama con
        # `ficha_procesal.para_tarjeta` y viaja en `rama_info`; la pantalla la
        # enseña en una línea. None si no llegó.
        "ficha": rama_info.get("ficha") if isinstance(rama_info.get("ficha"), dict)
                 and rama_info.get("ficha") else None,
        "avisos": list(dict.fromkeys(a for a in avisos if a)),
    }
    return _limpio(tarjeta)


def clave_de(tarjeta: dict) -> str:
    """Identifica una tarjeta armada (para el registro del servidor y las
    pruebas: la misma entrada da la misma tarjeta)."""
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
