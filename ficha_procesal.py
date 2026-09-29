"""LA FICHA PROCESAL DEL ASUNTO: QUIÉN ES QUIÉN Y QUÉ SE REVISA, UNA VEZ Y POR CÓDIGO.

SPEC_E, sección E2 (28-sep-2026). David, tras leer el proyecto intermedio del
AR 631/2025: «se impugna una sentencia de un juzgado de distrito, no de la
Magistrada que emitió el acto reclamado. El quejoso y el recurrente pueden no
necesariamente coincidir y el taller presupone que sí. […] hay que entender
cómo funciona el amparo».

POR QUÉ EXISTE. Cada pieza del taller adivinaba los papeles por su cuenta: la
propuesta con «la recurrente» a secas, la deliberación con el nombre tecleado,
la síntesis con dos renglones propios, el compositor con cinco llamadas
sueltas (`_quejoso_del_amparo`, `_recurrente_de`, `papel_del_recurrente`,
`_organo_recurrido` y el tercero por su lado). En el 631 eso produjo, en el
mismo proyecto, «QUEJOSA Y RECURRENTE: IMPULSORA…», la legitimación por el
art. 6o., una síntesis en la que «la parte quejosa adquirió el inmueble» y un
órgano recurrido que era la Magistrada del acto reclamado.

QUÉ HACE. Arma UNA ficha con lo que ya se leyó —encargo, fases, ficha de
partes, resolutivo del juzgado— reutilizando las deducciones que ya existen
(no las repite): `redactor_adelanto.papel_del_recurrente`, `_quejoso_del_amparo`,
`_recurrente_de`, `_organo_recurrido`, `fase_rama.que_hizo_el_juzgado`,
`fase_rama.sobreseyo_ademas`, `tipos_asunto.rama_revision` y
`tipos_asunto.reasuncion`. Cada dato lleva su FUENTE y lo que no consta se
dice en `avisos`: un dato inventado en la carátula se firma.

Y SE ENTREGA COMO DATOS. `bloque()` la escribe como renglones «rótulo: valor»
para los prompts de la propuesta, el contraste, la deliberación, el estudio,
la síntesis y la estructura; `linea()` la resume en un renglón para la
tarjeta. Nunca frases modelo: en un prompt, lo que se escribe entero se copia
entero (cuatro veces en este proyecto).

Función pura: sin modelo, sin red, sin base. Nunca lanza desde `de_resultado`.
"""
from __future__ import annotations

import re

FORMATO = 1

# El carácter procesal de quien recurre, con las figuras de la Ley de Amparo.
CARACTER = {
    "quejoso": "parte quejosa",
    "tercero": "parte tercera interesada",
    "autoridad": "autoridad",
}

_NOMBRE_TIPO = {
    "amparo_directo": "amparo directo",
    "amparo_revision": "amparo en revisión",
    "queja": "recurso de queja",
    "revision_fiscal": "revisión fiscal",
}

_ETIQUETA_QUE = {"sobresee": "sobreseimiento", "concede": "concesión", "niega": "negativa"}

# Los ordinales que ABREN un punto resolutivo. En mayúsculas y sin `re.I`: en
# minúsculas («el considerando primero.») son prosa, no un punto.
_RX_ORDINAL = re.compile(
    r"\b(PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|S[ÉE]PTIMO|OCTAVO|[ÚU]NICO)\s*\.\s+")
# Sobre qué recae el punto: lo que sigue a «respecto de», «en contra de» o
# «contra», hasta la remisión a los considerandos o los efectos.
_RX_OBJETO = re.compile(
    r"\b(?:respecto\s+(?:de(?:l)?|a)|en\s+contra\s+de(?:l)?|contra)\s+(?P<o>.+?)"
    r"(?=,?\s+(?:por\s+(?:los|las)\s+(?:motivos|razones|fundamentos|consideraciones)|"
    r"para\s+(?:los|el)\s+efecto|en\s+t[ée]rminos\s+de|conforme\s+a\s+lo|"
    r"precisad[oa]s?\s+en\s+el)|\.\s*$|$)", re.I)
# La autoridad a la que se atribuye el acto, dentro del objeto del punto.
_RX_AUTORIDAD = re.compile(
    r"(?:atribuid[oa]s?|reclamad[oa]s?|imputad[oa]s?)\s+(?:a\s+(?:la|el|los|las)\s+|al\s+|a\s+)"
    r"(?P<a>[A-ZÁÉÍÓÚÑ].+?)(?=,|\s+consistente|\s+y\s+que|\s*$)"
    r"|\bdel?\s+(?P<b>(?:Juzgado|Juez|Jueza|Sala|Magistrad[oa]|Tribunal|Presidente|Presidenta|"
    r"Director|Directora|Secretar[íi]o|Ayuntamiento|Congreso|Gobernador|Titular|Jefe|"
    r"Administrador|Procurador|Fiscal|Instituto|Comisi[óo]n|Junta|Registr|Notari)[^,;]*?)"
    r"(?=,|\s+consistente|\s*$)")
# ¿Consta una revisión adhesiva? Sólo con el nombre de la figura, nunca con la
# palabra suelta «adhesión».
_RX_ADHESIVA = re.compile(r"\b(?:revisi[óo]n|recurso)\s+(?:de\s+revisi[óo]n\s+)?adhesiv[oa]\b"
                          r"|\badhiri[óo]\s+al\s+recurso\b|\badhesi[óo]n\s+al\s+recurso\b", re.I)
_RX_ADHESIVO_AD = re.compile(r"\bamparo\s+adhesivo\b|\badhiri[óo]\s+al\s+amparo\b", re.I)


def _txt(x) -> str:
    return " ".join(str(x or "").split())


# El cargo delante del órgano: «Magistrada de la Sala Civil Uno» y «Sala Civil
# Uno del Tribunal Superior» son la misma responsable (el 631 las traía así, una
# del formulario y otra del resolutivo del juzgado).
_RX_CARGO = re.compile(r"^(?:(?:el|la|los|las)\s+)?(?:c\.\s+)?(?:magistrad[oa]s?|titular|juez|jueza|"
                       r"presidente|presidenta|integrantes)(?:\s+(?:presidente|presidenta|"
                       r"unitari[oa]|numerari[oa]))?\s+(?:de\s+)?(?:(?:el|la|los|las)\s+)?", re.I)


def _misma_autoridad(a: str, b: str) -> bool:
    """¿Dos maneras de nombrar a la misma autoridad? Igualdad, una dentro de
    otra, o lo mismo quitado el cargo de su titular."""
    import redactor_adelanto as _ra
    if _ra._mismo_nombre(a, b):
        return True
    a2, b2 = _RX_CARGO.sub("", _txt(a)), _RX_CARGO.sub("", _txt(b))
    return len(a2.split()) >= 2 and len(b2.split()) >= 2 and _ra._mismo_nombre(a2, b2)


def _normalizar_tipo(tipo: str) -> str:
    try:
        import tipos_asunto as _ta
        return _ta.normalizar(tipo or "")
    except Exception:
        return (tipo or "").strip().lower()


def puntos_resolutivos(seccion: str) -> list:
    """[{ordinal, texto, que, objeto, autoridad}] de la sección resolutiva del
    juzgado, un punto por ordinal. `que` es lo que dice ESE punto con las
    fórmulas del resolutivo (`fase_rama.que_dice_el_resolutivo`); los puntos
    de trámite (notifíquese, archívese) no dicen nada y no entran."""
    import fase_rama as _fr
    t = _txt(seccion)
    if not t:
        return []
    marcas = list(_RX_ORDINAL.finditer(t))
    trozos = []
    if marcas:
        for i, m in enumerate(marcas):
            fin = marcas[i + 1].start() if i + 1 < len(marcas) else len(t)
            trozos.append((m.group(1).upper(), t[m.end():fin].strip()))
    else:
        trozos.append(("", t))
    fuera = []
    for ordinal, cuerpo in trozos:
        que = _fr.que_dice_el_resolutivo(cuerpo)
        if not que:
            continue
        mo = _RX_OBJETO.search(cuerpo)
        objeto = _txt(mo.group("o")).rstrip(" .,;") if mo else ""
        ma = _RX_AUTORIDAD.search(objeto) if objeto else None
        autoridad = _txt((ma.group("a") or ma.group("b")) if ma else "").rstrip(" .,;")
        # «sobresee_concede» en UN punto: dos resoluciones, dos renglones.
        for q in (que.split("_") if "_" in que else [que]):
            fuera.append({"ordinal": ordinal, "que": q, "objeto": objeto[:300],
                          "autoridad": autoridad[:200], "texto": cuerpo[:600]})
    return fuera


def _perjudica(que: str, papel: str):
    """¿Ese punto perjudica a quien recurre? True | False | None (no se sabe).
    El sobreseimiento y la negativa perjudican a la quejosa; la concesión, a la
    autoridad y a la tercera interesada."""
    if papel == "quejoso":
        return que in ("sobresee", "niega")
    if papel in ("tercero", "autoridad"):
        return que == "concede"
    return None


def _rige_93(que: str, papel: str, sobresee_ademas: bool) -> str:
    """La fracción del art. 93 que rige el estudio, por lo que resolvió el
    juzgado y por quién recurre. La VI es la de la autoridad o la tercera
    interesada contra una concesión (2a./J. 113/2007, registro 171925: «sin
    importar quién interponga el recurso» cuando se revoca; sólo la quejosa la
    excluye, `tipos_asunto.reasuncion`)."""
    import tipos_asunto as _ta
    q = (que or "").strip().lower()
    if q in ("concede", "sobresee_concede") and papel != "quejoso":
        return _ta.FUNDAMENTO_REASUNCION["concesion"]
    if q in ("sobresee",) and papel in ("quejoso", ""):
        return _ta.FUNDAMENTO_REASUNCION["sobreseimiento"]
    if q in ("niega", "sobresee_niega", "concede", "sobresee_concede") and papel == "quejoso":
        return ("artículo 93, fracciones I y V, de la Ley de Amparo"
                if q.startswith("sobresee") else "artículo 93, fracción V, de la Ley de Amparo")
    return ""


def _resultado_rama(rama: str, reas: str) -> str:
    """Qué supone cada rama, como dato (no como frase del proyecto)."""
    return {
        "confirma_niega": "confirma; subsiste la negativa",
        "confirma_concede": "confirma; subsiste la concesión",
        "confirma_sobresee": "confirma; subsiste el sobreseimiento",
        "confirma_sobresee_niega": "confirma; subsisten el sobreseimiento y la negativa",
        "confirma_sobresee_concede": "confirma; subsisten el sobreseimiento y la concesión",
        "revoca_sobreseimiento_concede": "revoca el sobreseimiento; reasume jurisdicción y "
                                         "estudia los conceptos de violación; concede o niega "
                                         "según ese estudio",
        "revoca_sobreseimiento_niega": "revoca el sobreseimiento; reasume jurisdicción y "
                                       "estudia los conceptos de violación; concede o niega "
                                       "según ese estudio",
        "revoca_fondo_niega": ("revoca la concesión; reasume jurisdicción sobre los conceptos "
                               "de violación que el juzgado no estudió; niega si ninguno "
                               "prospera, concede por razón distinta si alguno prospera")
                              if reas == "concesion" else "revoca la concesión; niega",
        "revoca_fondo_concede": "revoca la negativa; concede",
        "modifica_efectos": "modifica sólo los efectos de la concesión",
        "repone_procedimiento": "revoca y ordena reponer el procedimiento",
        "sin_materia": "sin materia",
        "sin_determinar": "no calculable: no consta qué resolvió el juzgado",
    }.get(rama, rama or "no calculable")


def _desenlace_revision(que: str, papel: str, sobresee_ademas: bool, firme: list) -> dict:
    import tipos_asunto as _ta
    fuera = {}
    for clave, sentido in (("si_prospera", "fundado"), ("si_no_prospera", "infundado")):
        rama = _ta.rama_revision(que, sentido)
        reas = _ta.reasuncion(que, sentido, quien_recurre=papel)
        fund = (_ta.FUNDAMENTO_REASUNCION.get(reas)
                or (_ta.RAMAS_REVISION.get(rama) or {}).get("fundamento") or "")
        d = {"rama": rama, "reasuncion": reas, "fundamento": fund,
             "resultado": _resultado_rama(rama, reas)}
        if clave == "si_prospera" and firme and rama.startswith("revoca"):
            d["alcance"] = "sólo en la materia de la revisión; lo firme no se toca"
        fuera[clave] = d
    return fuera


def armar(encargo, fases=None, partes=None, *, acto: str = "", declarado: str = "") -> dict:
    """La ficha procesal del asunto (formato 1). Ver el docstring del módulo.

    `encargo`: el `redactor_adelanto.Encargo` (o algo con sus atributos).
    `fases`: las del adelanto (resolutivo, fuentes, resolvio_a_quo…).
    `partes`: la `fase_partes.Partes` leída de la sentencia, o None.
    `acto`: el texto de la resolución recurrida o reclamada; si falta, la
    primera fuente de las fases. `declarado`: lo que el motor dijo que resolvió
    el juzgado al proponer (última fuente, como en `que_hizo_el_juzgado`)."""
    import fase_rama as _fr
    import redactor_adelanto as _ra
    e = encargo
    tipo = _normalizar_tipo(getattr(e, "tipo_asunto", "") or "amparo_directo")
    es_recurso = bool(getattr(e, "es_recurso", False))
    avisos: list = []
    fuentes = list(getattr(fases, "fuentes", None) or []) + ["", ""]
    acto = acto or str(fuentes[0] or "")
    res = _ra._resolutivo_del_a_quo(fases, acto) if es_recurso else ""

    # ── LA QUEJOSA: quien pidió el amparo (el resolutivo del juzgado manda) ──
    tecleado = _txt(getattr(e, "quejoso", ""))
    leido = _txt(getattr(partes, "quejoso", ""))
    quejosa = _txt(_ra._quejoso_del_amparo(e, partes, res))
    del_res = _txt(_fr.quejoso_del_resolutivo(res)) if res else ""
    if del_res and _ra._misma_parte(quejosa, del_res):
        f_q = "resolutivo del juzgado"
    elif leido and _ra._misma_parte(quejosa, leido):
        f_q = "ficha de partes"
    else:
        f_q = "formulario"
    if not quejosa:
        avisos.append("No consta quién promovió el amparo.")

    # ── QUIÉN RECURRE Y EN QUÉ CARÁCTER ──
    papel = _ra.papel_del_recurrente(e, partes, res) if es_recurso else ""
    aparte = _txt(_ra._recurrente_de(e, partes, res)) if es_recurso else ""
    recurrente = {}
    if es_recurso:
        nombre_r = aparte or (quejosa if papel == "quejoso" else "")
        recurrente = {"nombre": nombre_r, "aparte": aparte, "papel": papel,
                      "caracter": CARACTER.get(papel, ""),
                      "fuente": ("formulario (recurrente)" if _txt(getattr(e, "recurrente", ""))
                                 else "formulario, separado de la quejosa por el resolutivo"
                                 if aparte else "formulario")}
        if not nombre_r:
            avisos.append("No consta el nombre de quien interpuso el recurso.")

    # ── TERCEROS INTERESADOS ──
    terceros = []
    _t_raw = _txt(getattr(partes, "tercero_interesado", ""))
    if _t_raw:
        terceros.append({"nombre": _t_raw, "fuente": "ficha de partes"})
    elif papel == "tercero" and aparte:
        # La recurrente que no es la quejosa ni una autoridad es la tercera
        # interesada (art. 5o., fr. III): el 631.
        terceros.append({"nombre": aparte, "fuente": "recurrente que no es quejosa ni autoridad"})

    # ── EL ÓRGANO DE LA RESOLUCIÓN RECURRIDA (en la revisión, el Juzgado) ──
    organo = {}
    if es_recurso:
        n_o = _txt(_ra._organo_recurrido(e, partes, acto))
        f_o = ("ficha de partes" if n_o and _ra._mismo_nombre(n_o, _txt(getattr(partes, "autoridad_responsable", "")))
               else "encabezado de la sentencia recurrida" if n_o else "")
        organo = {"nombre": n_o, "fuente": f_o}
        if not n_o:
            avisos.append("No consta el órgano que dictó la resolución recurrida.")

    # ── QUÉ RESOLVIÓ, POR PUNTO, CON SU FUENTE ──
    resolvio = {}
    materia, firme = [], []
    responsables = []
    if tipo == "amparo_revision":
        que = _fr.que_hizo_el_juzgado(fases, declarado) if fases is not None else \
            _fr.resolvio_segun_resolutivos(acto)
        if fases is not None and _fr.que_dice_el_resolutivo(str(getattr(fases, "resolutivo_recurrida", "") or "")):
            f_que = "punto resolutivo del juzgado"
        elif fases is not None and str(getattr(fases, "resolvio_a_quo", "") or "").strip().lower() in _fr._VALIDOS_A_QUO:
            f_que = "lectura del PDF de la sentencia recurrida"
        elif (declarado or "").strip():
            f_que = "lo declarado al proponer"
        else:
            f_que = "antecedentes" if que else ""
        sob_ad = bool(fases is not None and _fr.sobreseyo_ademas(fases, declarado))
        seccion = _fr.seccion_resolutiva(acto) or res
        puntos = puntos_resolutivos(seccion)
        if not puntos and res:
            puntos = puntos_resolutivos(res)
        resolvio = {"que": que or "", "fuente": f_que, "sobresee_ademas": sob_ad,
                    "puntos": [{k: p[k] for k in ("ordinal", "que", "objeto")} for p in puntos]}
        if not que:
            avisos.append("No consta qué resolvió el juzgado: ni su resolutivo ni la lectura "
                          "del PDF lo dicen.")
        elif not puntos:
            avisos.append("No se pudieron leer los puntos resolutivos uno por uno: lo resuelto "
                          "sale de " + (f_que or "otra fuente") + ".")
        for p in puntos:
            if p["autoridad"]:
                if not any(_misma_autoridad(p["autoridad"], r["autoridad"]) for r in responsables):
                    responsables.append({"autoridad": p["autoridad"], "acto": p["objeto"],
                                         "resolvio": p["que"],
                                         "fuente": f"punto {p['ordinal'] or 'resolutivo'} del juzgado"})
            et = f"{_ETIQUETA_QUE.get(p['que'], p['que'])}" + (
                f" (punto {p['ordinal']})" if p["ordinal"] else "") + (
                f": {p['objeto']}" if p["objeto"] else "")
            dano = _perjudica(p["que"], papel)
            if dano is True:
                materia.append(et)
            elif dano is False:
                firme.append(et)
        if not puntos and que:
            # Sin puntos legibles, lo que hizo el juzgado en bloque.
            for q in (que.split("_") if "_" in que else [que]):
                dano = _perjudica(q, papel)
                et = _ETIQUETA_QUE.get(q, q)
                (materia if dano is True else firme if dano is False else []).append(et)
        # NADA LE PERJUDICA Y AUN ASÍ RECURRE: la quejosa que recurre una
        # concesión combate sus términos o sus efectos. Eso no queda firme: es
        # la materia de la revisión.
        if firme and not materia:
            materia = [x + " (sus términos o efectos)" for x in firme]
            firme = []
        if sob_ad and papel in ("tercero", "autoridad") and not any(
                x.startswith("sobreseimiento") for x in firme):
            firme.append("sobreseimiento respecto de otro acto")
    # La autoridad del formulario, si los puntos no la nombran (en la revisión
    # es la del acto reclamado, nunca el juzgado).
    _resp_f = _txt(getattr(e, "responsable", "")) or _txt(getattr(partes, "autoridad_responsable", ""))
    if _resp_f and not any(_misma_autoridad(_resp_f, r["autoridad"]) for r in responsables) \
            and not (organo.get("nombre") and _ra._mismo_nombre(_resp_f, organo["nombre"])):
        if tipo != "amparo_revision" or not re.search(r"juzgado\s+\w*\s*de\s+distrito", _resp_f, re.I):
            _acto_f = ""
            if tipo == "amparo_directo":
                _exp = _txt(getattr(fases, "expediente_origen", ""))
                _fec = _txt(getattr(fases, "fecha_origen", ""))
                _acto_f = "resolución reclamada" + (f" de {_fec}" if _fec else "") + (
                    f", expediente {_exp}" if _exp else "")
            responsables.append({"autoridad": _resp_f, "acto": _acto_f, "resolvio": "",
                                 "fuente": "formulario" if _txt(getattr(e, "responsable", ""))
                                 else "ficha de partes"})
    if not responsables and tipo in ("amparo_directo", "amparo_revision"):
        avisos.append("No consta la autoridad responsable.")
    if tipo == "amparo_directo" and not terceros:
        avisos.append("No consta el tercero interesado.")

    # ── ¿HAY ADHESIVO? ──
    _donde_ad = ""
    _rx_ad = _RX_ADHESIVO_AD if tipo == "amparo_directo" else _RX_ADHESIVA
    for nombre, texto in (("escrito", fuentes[1]),
                          ("resumen de lo que se combate", getattr(fases, "resumen_conceptos", "")),
                          ("antecedentes", getattr(fases, "antecedentes", ""))):
        if _rx_ad.search(str(texto or "") if not isinstance(texto, (list, tuple))
                         else "\n".join(map(str, texto))):
            _donde_ad = nombre
            break
    adhesivo = {"consta": bool(_donde_ad), "donde": _donde_ad}

    # ── EL ART. 93 Y EL DESENLACE, POR CÓDIGO ──
    art_93 = None
    desenlace = {}
    if tipo == "amparo_revision":
        que = resolvio.get("que", "")
        rige = _rige_93(que, papel, resolvio.get("sobresee_ademas", False))
        des = _desenlace_revision(que, papel, resolvio.get("sobresee_ademas", False), firme)
        art_93 = {"rige": rige, **des}
        desenlace = des
        if not rige:
            avisos.append("No se pudo fijar la fracción del artículo 93 que rige: falta lo que "
                          "resolvió el juzgado o el carácter de quien recurre.")
    elif tipo == "amparo_directo":
        desenlace = {"si_prospera": {"resultado": "concede"},
                     "si_no_prospera": {"resultado": "niega"}}

    return {"formato": FORMATO, "tipo_asunto": tipo, "es_recurso": es_recurso,
            "quejosa": {"nombre": quejosa, "fuente": f_q if quejosa else "",
                        "tecleado": tecleado},
            "responsables": responsables, "terceros": terceros,
            "organo_recurrido": organo, "resolvio": resolvio,
            "recurrente": recurrente, "adhesivo": adhesivo,
            "materia_revision": materia, "firme": firme,
            "art_93": art_93, "desenlace": desenlace, "avisos": avisos}


def de_resultado(r, declarado: str = "") -> dict:
    """La ficha de una sesión del taller (`Resultado`): encargo, fases y partes.
    Nunca lanza: si algo falla, una ficha vacía con el aviso."""
    try:
        e = getattr(r, "encargo", None)
        if e is None:
            return {}
        return armar(e, getattr(r, "fases", None), getattr(r, "partes", None),
                     declarado=declarado)
    except Exception as ex:                          # pragma: no cover - defensivo
        print(f"   ⚠️ FICHA PROCESAL: no se pudo armar: {type(ex).__name__}: {str(ex)[:120]}")
        return {}


def _renglon(rotulo: str, valor: str, fuente: str = "") -> str:
    v = _txt(valor)
    if not v:
        return ""
    return f"  {rotulo}: {v}" + (f" [fuente: {fuente}]" if fuente else "")


def bloque(ficha: dict) -> str:
    """La ficha como bloque de DATOS para un prompt: renglones «rótulo: valor»
    con su fuente. «» si no hay ficha. Sin instrucciones ni frases que copiar."""
    f = ficha or {}
    if not f.get("tipo_asunto"):
        return ""
    tipo = f["tipo_asunto"]
    L = ["LA FICHA PROCESAL DEL ASUNTO (datos del expediente fijados por código, con su "
         "fuente; donde un dato no consta, se dice):",
         _renglon("Tipo de asunto", _NOMBRE_TIPO.get(tipo, tipo))]
    q = f.get("quejosa") or {}
    L.append(_renglon("Parte quejosa (promovió el amparo)" if tipo != "revision_fiscal"
                      else "Parte actora en el juicio de nulidad", q.get("nombre"), q.get("fuente")))
    for r in f.get("responsables") or []:
        v = r.get("autoridad", "") + (f" — acto: {r['acto']}" if r.get("acto") else "")
        L.append(_renglon("Autoridad responsable", v, r.get("fuente")))
    for t in f.get("terceros") or []:
        L.append(_renglon("Tercero interesado", t.get("nombre"), t.get("fuente")))
    o = f.get("organo_recurrido") or {}
    if f.get("es_recurso"):
        L.append(_renglon("Órgano que dictó la resolución recurrida",
                          o.get("nombre") or "no consta", o.get("fuente")))
    rv = f.get("resolvio") or {}
    if rv:
        pts = "; ".join(
            (f"{p['ordinal']}: " if p.get("ordinal") else "")
            + _ETIQUETA_QUE.get(p.get("que"), p.get("que", ""))
            + (f" — {p['objeto']}" if p.get("objeto") else "")
            for p in rv.get("puntos") or [])
        L.append(_renglon("Lo que resolvió el juzgado", pts or rv.get("que") or "no consta",
                          rv.get("fuente") if not pts else "sus puntos resolutivos"))
    rc = f.get("recurrente") or {}
    if f.get("es_recurso"):
        L.append(_renglon("Recurrente", (rc.get("nombre") or "no consta")
                          + (f" ({rc['caracter']})" if rc.get("caracter") else ""), rc.get("fuente")))
        ad = f.get("adhesivo") or {}
        L.append(_renglon("Revisión adhesiva" if tipo == "amparo_revision" else "Adhesión",
                          f"consta ({ad.get('donde')})" if ad.get("consta") else "no consta"))
    elif tipo == "amparo_directo":
        ad = f.get("adhesivo") or {}
        L.append(_renglon("Amparo adhesivo",
                          f"consta ({ad.get('donde')})" if ad.get("consta") else "no consta"))
    if f.get("materia_revision"):
        L.append(_renglon("Materia de la revisión (lo que perjudica a quien recurre)",
                          "; ".join(f["materia_revision"])))
    if f.get("firme"):
        L.append(_renglon("No impugnado por quien recurre (firme, salvo que otra parte lo "
                          "combata)", "; ".join(f["firme"])))
    a = f.get("art_93") or {}
    if a:
        L.append(_renglon("Artículo 93 de la Ley de Amparo que rige", a.get("rige") or "no consta"))
    d = f.get("desenlace") or {}
    for clave, rot in (("si_prospera", "Si el recurso prospera" if f.get("es_recurso")
                        else "Si algún concepto prospera"),
                       ("si_no_prospera", "Si no prospera" if f.get("es_recurso")
                        else "Si ninguno prospera")):
        x = d.get(clave) or {}
        if x.get("resultado"):
            v = x["resultado"] + (f" · {x['alcance']}" if x.get("alcance") else "") + (
                f" · {x['fundamento']}" if x.get("fundamento") else "")
            L.append(_renglon(rot, v, "código (tipos_asunto)" if x.get("rama") else ""))
    for av in f.get("avisos") or []:
        L.append(_renglon("No consta", av))
    return "\n".join(x for x in L if x) + "\n"


# El carácter con las palabras del contrato de la tarjeta (FICHA).
_CARACTER_TARJETA = {"quejoso": "quejosa", "tercero": "tercera interesada",
                     "autoridad": "autoridad responsable"}


def para_tarjeta(ficha: dict):
    """La ficha con la forma FICHA del contrato de la tarjeta
    (contrato_tarjeta.md): {tipo, quejosa, responsables, terceros, recurrida,
    recurrente, materia, avisos, linea}. None si no hay ficha."""
    f = ficha or {}
    if not f.get("tipo_asunto"):
        return None
    rv = f.get("resolvio") or {}
    o = (f.get("organo_recurrido") or {}).get("nombre") or ""
    recurrida = None
    if f.get("es_recurso"):
        resol = [{"acto": p.get("objeto") or "", "sentido": p.get("que") or ""}
                 for p in rv.get("puntos") or []]
        if not resol and rv.get("que"):
            resol = [{"acto": "", "sentido": q} for q in str(rv["que"]).split("_")]
        recurrida = {"organo": o, "resolvio": resol} if (o or resol) else None
    rc = f.get("recurrente") or {}
    recurrente = ({"quien": rc.get("nombre") or "",
                   "caracter": _CARACTER_TARJETA.get(rc.get("papel") or "", "")}
                  if f.get("es_recurso") and (rc.get("nombre") or rc.get("papel")) else None)
    materia = "; ".join(f.get("materia_revision") or [])
    if f.get("firme"):
        materia = (materia + " · " if materia else "") + "firme: " + "; ".join(f["firme"])
    return {"tipo": f["tipo_asunto"],
            "quejosa": (f.get("quejosa") or {}).get("nombre") or "",
            "responsables": [{"autoridad": r.get("autoridad") or "", "acto": r.get("acto") or ""}
                             for r in f.get("responsables") or []],
            "terceros": [t.get("nombre") for t in f.get("terceros") or [] if t.get("nombre")],
            "recurrida": recurrida, "recurrente": recurrente, "materia": materia,
            "avisos": list(f.get("avisos") or []), "linea": linea(f)}


def linea(ficha: dict) -> str:
    """La ficha en UN renglón, para la tarjeta. «» si no hay ficha."""
    f = ficha or {}
    tipo = f.get("tipo_asunto")
    if not tipo:
        return ""
    partes = [_NOMBRE_TIPO.get(tipo, tipo)]
    q = (f.get("quejosa") or {}).get("nombre")
    if q:
        partes.append(f"quejosa: {q}")
    if f.get("es_recurso"):
        rc = f.get("recurrente") or {}
        if rc.get("nombre") and rc.get("papel") != "quejoso":
            partes.append(f"recurre: {rc['nombre']} ({rc.get('caracter') or 'carácter no consta'})")
        elif rc.get("papel") == "quejoso":
            partes.append("recurre la quejosa")
        o = (f.get("organo_recurrido") or {}).get("nombre")
        rv = f.get("resolvio") or {}
        _que = " y ".join(dict.fromkeys(
            _ETIQUETA_QUE.get(p.get("que"), p.get("que")) for p in rv.get("puntos") or [])) \
            or _ETIQUETA_QUE.get(rv.get("que"), rv.get("que") or "")
        if o or _que:
            partes.append("recurrida: " + (o or "órgano no consta") + (f" ({_que})" if _que else ""))
        if f.get("materia_revision"):
            partes.append("materia: " + ", ".join(x.split(":")[0] for x in f["materia_revision"]))
        if f.get("firme"):
            partes.append("firme: " + ", ".join(x.split(":")[0] for x in f["firme"]))
        rige = (f.get("art_93") or {}).get("rige") or ""
        m = re.search(r"fracci(?:ón|ones)\s+([IVX]+(?:\s+y\s+[IVX]+)?)", rige)
        if m:
            partes.append(f"art. 93, fr. {m.group(1)}")
    else:
        rs = f.get("responsables") or []
        if rs:
            partes.append(f"responsable: {rs[0].get('autoridad')}")
        ts = f.get("terceros") or []
        if ts:
            partes.append(f"tercero: {ts[0].get('nombre')}")
    if f.get("avisos"):
        partes.append(f"{len(f['avisos'])} dato(s) no constan")
    return " · ".join(partes)
