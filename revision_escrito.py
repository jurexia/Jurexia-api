"""LA REVISIÓN ANTES DE PRESENTAR — 28-sep-2026.

Lo que el abogado revisa a ojo antes de firmar y que se puede comprobar sin
modelo: los requisitos que la ley pide al escrito, los huecos que quedaron,
las plantillas sin llenar, el cierre y las frases rotas. Sin red, sin costo
y en tiempo lineal: corre sobre lo que está en la hoja, con cada clic.

LO QUE NO HACE, A PROPÓSITO: no calcula el plazo. Un cómputo necesita la
fecha de notificación, la regla de surtimiento y el calendario de inhábiles
del órgano; hecho a medias, un plazo equivocado es peor que ninguno. Si el
escrito no dice cuándo se notificó el acto, lo avisa, y eso es todo.

Cada hallazgo es {nivel, que, fundamento, donde}:
  · «falta»:  el escrito así se previene o se desecha, o lleva algo sin llenar;
  · «revise»: puede estar bien, pero conviene mirarlo;
  · «bien»:   un requisito que sí está (para que la lista diga qué se revisó).
"""

from __future__ import annotations

import re
import unicodedata
import linter_juridico

TOPE_TEXTO = 300_000


def _plano(texto: str) -> str:
    t = unicodedata.normalize("NFD", texto or "")
    t = "".join(c for c in t if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"[ \t]+", " ", t)


def _hallazgo(nivel: str, que: str, fundamento: str = "", donde: str = "") -> dict:
    return {"nivel": nivel, "que": que, "fundamento": fundamento, "donde": donde}


# ── ¿Qué escrito es? ─────────────────────────────────────────────────────
# Por lo que dice, en el orden en que se descartan: una resolución no es un
# escrito de parte, un amparo directo va ante un colegiado contra una
# sentencia definitiva, un recurso tiene agravios, una demanda ordinaria
# tiene prestaciones.
def tipo_de_escrito(llano: str) -> str:
    if re.search(r"\b(?:se resuelve|resolutivos?|considerandos?)\b", llano) and \
            not re.search(r"\bprotesto\b", llano):
        return "resolucion"
    amparo = re.search(r"\b(?:demanda de amparo|juicio de amparo|amparo (?:directo|indirecto)|"
                       r"conceptos? de violacion)\b", llano)
    if amparo:
        if re.search(r"\bamparo directo\b", llano) or (
                re.search(r"\btribunal colegiado\b", llano)
                and re.search(r"\b(?:sentencia definitiva|laudo|resolucion que puso fin)\b", llano)):
            return "amparo_directo"
        return "amparo_indirecto"
    if re.search(r"\bagravios?\b", llano):
        return "recurso"
    if re.search(r"\bprestaciones\b", llano):
        return "demanda"
    return "otro"


_NOMBRES_TIPO = {
    "amparo_indirecto": "demanda de amparo indirecto",
    "amparo_directo": "demanda de amparo directo",
    "recurso": "recurso",
    "demanda": "demanda",
    "resolucion": "resolución",
    "otro": "escrito",
}

# Los requisitos de la demanda de amparo: cada uno con lo que lo delata en el
# texto. Se busca el rastro, no la redacción: «tercero interesado» basta para
# saber que se ocupó de él, y el abogado juzga si lo hizo bien.
_REQUISITOS_108 = [
    (r"\bdomicilio\b", "El domicilio del quejoso o de quien promueve en su nombre",
     "artículo 108, fracción I, de la Ley de Amparo", "revise"),
    (r"\btercer[oa]s? interesad[oa]s?\b",
     "El tercero interesado, o la manifestación bajo protesta de que no lo conoce",
     "artículo 108, fracción II, de la Ley de Amparo", "falta"),
    (r"\bautoridad(?:es)? responsables?\b", "La autoridad o autoridades responsables",
     "artículo 108, fracción III, de la Ley de Amparo", "falta"),
    (r"\b(?:actos? reclamados?|norma general|omision reclamada|se reclama)\b",
     "El acto, la omisión o la norma que se reclama de cada autoridad",
     "artículo 108, fracción IV, de la Ley de Amparo", "falta"),
    (r"\bbajo protesta de decir verdad\b",
     "Los antecedentes «bajo protesta de decir verdad»: sin la manifestación, el juez previene",
     "artículo 108, fracción V, de la Ley de Amparo", "falta"),
    (r"\b(?:preceptos? (?:constitucionales|violados)|derechos? humanos? (?:violados?|que se estiman)|"
     r"articulos? \d{1,3}\w* (?:constitucional|de la constitucion))",
     "Los preceptos que contienen los derechos humanos cuya violación se reclama",
     "artículo 108, fracción VI, de la Ley de Amparo", "falta"),
    (r"\bconceptos? de violacion\b", "Los conceptos de violación",
     "artículo 108, fracción VIII, de la Ley de Amparo", "falta"),
]

_REQUISITOS_175 = [
    (r"\bdomicilio\b", "El domicilio del quejoso y de quien promueve en su nombre",
     "artículo 175, fracción I, de la Ley de Amparo", "revise"),
    (r"\btercer[oa]s? interesad[oa]s?\b", "El nombre y el domicilio del tercero interesado",
     "artículo 175, fracción II, de la Ley de Amparo", "falta"),
    (r"\bautoridad(?:es)? responsables?\b", "La autoridad responsable",
     "artículo 175, fracción III, de la Ley de Amparo", "falta"),
    (r"\b(?:actos? reclamados?|sentencia definitiva|laudo|resolucion que (?:puso|pone) fin)\b",
     "El acto reclamado: la sentencia definitiva, el laudo o la resolución que puso fin al juicio",
     "artículo 175, fracción IV, de la Ley de Amparo", "falta"),
    (r"\bnotific\w*\b[^.\n]{0,80}\b\d{1,2} de (?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
     r"septiembre|setiembre|octubre|noviembre|diciembre)\b|\btuvo conocimiento\b",
     "La fecha en que se notificó el acto reclamado o en que se tuvo conocimiento de él",
     "artículo 175, fracción V, de la Ley de Amparo", "falta"),
    (r"\b(?:preceptos? (?:constitucionales|violados)|derechos? humanos? (?:violados?|que se estiman)|"
     r"articulos? \d{1,3}\w* (?:constitucional|de la constitucion))",
     "Los preceptos que contienen los derechos humanos cuya violación se reclama",
     "artículo 175, fracción VI, de la Ley de Amparo", "falta"),
    (r"\bconceptos? de violacion\b", "Los conceptos de violación",
     "artículo 175, fracción VII, de la Ley de Amparo", "falta"),
]

_MESES = r"(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)"
# Lo que dejan las plantillas y los modelos cuando no llenan: corchetes con
# indicaciones y equis. «PARTE QUEJOSA» como nombre lo caza el linter, que
# sabe distinguirlo de «la parte quejosa sostiene…», que es prosa.
_PLANTILLA = re.compile(
    r"\[(?:nombre|ciudad|fecha|domicilio|numero|n[uú]mero|expediente|juzgado|tipo|materia|"
    r"actor|demandado|quejos[oa]|autoridad|monto|cantidad|x{1,20})[^\[\]\n]{0,60}\]"
    r"|\bX{4,}\b|_{3,} de _{3,}",
    re.I,
)
_DATO_PENDIENTE = re.compile(r"\[[ \t]{0,8}DATO[ \t]{1,8}PENDIENTE[ \t]{0,8}:[ \t]{0,8}([^\[\]\n]{1,200})\]", re.I)


def revisar_escrito(texto: str) -> dict:
    """La revisión del escrito que está en la hoja. Ver el docstring del módulo."""
    t = (texto or "")[:TOPE_TEXTO]
    llano = _plano(t)
    tipo = tipo_de_escrito(llano)
    hallazgos: list = []

    # 1. Lo que quedó sin llenar.
    pendientes = [m.group(1).strip() for m in _DATO_PENDIENTE.finditer(t)]
    if pendientes:
        unicos = list(dict.fromkeys(pendientes))
        hallazgos.append(_hallazgo(
            "falta", f"Quedan {len(pendientes)} {'dato pendiente' if len(pendientes) == 1 else 'datos pendientes'}: "
                     + "; ".join(unicos[:6]) + ("…" if len(unicos) > 6 else ""),
            donde=f"[DATO PENDIENTE: {unicos[0]}]"))
    plantillas = list(_PLANTILLA.finditer(t))
    if plantillas:
        m = plantillas[0]
        hallazgos.append(_hallazgo(
            "falta", f"{len(plantillas)} {'marca' if len(plantillas) == 1 else 'marcas'} de plantilla sin llenar "
                     f"(«{m.group(0)[:40]}»)", donde=m.group(0)))

    # 2. Los requisitos que la ley pide a la demanda de amparo.
    requisitos = {"amparo_indirecto": _REQUISITOS_108, "amparo_directo": _REQUISITOS_175}.get(tipo)
    for patron, que, fundamento, nivel in requisitos or []:
        if re.search(patron, llano):
            hallazgos.append(_hallazgo("bien", que, fundamento))
        else:
            hallazgos.append(_hallazgo(nivel, que, fundamento))
    if tipo == "amparo_indirecto":
        if not re.search(r"\bcopias?\b", llano):
            hallazgos.append(_hallazgo(
                "revise", "Las copias de traslado: una para cada parte y dos para el incidente de suspensión, "
                          "si se pide", "artículo 110 de la Ley de Amparo"))
        if not re.search(r"\b(?:notific\w*|tuvo conocimiento|tuve conocimiento|se hizo sabedor|"
                         r"me ostento sabedor)\b", llano):
            hallazgos.append(_hallazgo(
                "revise", "No encuentro cuándo se notificó o se conoció el acto: sin esa fecha no se puede "
                          "comprobar que la demanda está en plazo",
                "artículo 17 de la Ley de Amparo"))
    if tipo == "amparo_directo" and not re.search(r"\bcopias?\b", llano):
        hallazgos.append(_hallazgo(
            "revise", "Las copias: una para el expediente de la autoridad responsable y una para cada parte",
            "artículo 177 de la Ley de Amparo"))

    # 3. El recurso y la demanda ordinaria: lo mínimo de su esqueleto.
    if tipo == "recurso":
        hallazgos.append(_hallazgo(
            "bien" if re.search(r"\b(?:resolucion|sentencia|auto|acuerdo) (?:recurrida|impugnada|que se combate)\b",
                                llano) else "revise",
            "Qué resolución se combate, con su fecha y el órgano que la dictó"))
    if tipo == "demanda":
        for patron, que in ((r"\bhechos\b", "Los hechos, numerados"),
                            (r"\b(?:derecho|fundamentos? de derecho|consideraciones de derecho)\b",
                             "El capítulo de derecho: el fundamento de cada prestación"),
                            (r"\bpruebas?\b", "Las pruebas, relacionadas con los hechos")):
            hallazgos.append(_hallazgo("bien" if re.search(patron, llano) else "falta", que))

    # 4. El cierre de un escrito de parte.
    if tipo != "resolucion":
        cierre = re.search(r"\b(?:protesto|protestamos|atentamente|respetuosamente)\b", llano)
        hallazgos.append(_hallazgo("bien" if cierre else "revise",
                                   "El cierre: «PROTESTO LO NECESARIO» o su equivalente"))
        fecha = re.search(r"\ba \d{1,2} de " + _MESES + r" de(?:l)? (?:\d{4}|dos mil)", llano)
        hallazgos.append(_hallazgo("bien" if fecha else "revise", "El lugar y la fecha del escrito"))
        if not re.search(r"_{8,}|\bfirma\b|\bfirmado\b", llano):
            hallazgos.append(_hallazgo("revise", "El renglón de la firma y el nombre de quien firma"))

    # 5. Frases rotas, comillas huérfanas, puntuación imposible. El linter se
    # escribió para los proyectos de sentencia, que escribe el modelo: sus
    # `\s+[,;]` y `\s*\.` son cuadráticos con una racha larga de blancos, y
    # aquí el texto lo trae el abogado. Con los blancos colapsados a uno, cada
    # uno cuesta un carácter (300 mil hostiles, menos de un cuarto de segundo).
    for que, donde in linter_juridico.revisar(re.sub(r"\s+", " ", t))[:8]:
        hallazgos.append(_hallazgo("revise", que[:1].upper() + que[1:], donde=donde[:140]))

    orden = {"falta": 0, "revise": 1, "bien": 2}
    hallazgos.sort(key=lambda h: orden.get(h["nivel"], 3))
    return {
        "tipo": tipo,
        "tipo_nombre": _NOMBRES_TIPO[tipo],
        "hallazgos": hallazgos,
        "faltan": sum(h["nivel"] == "falta" for h in hallazgos),
        "revisar": sum(h["nivel"] == "revise" for h in hallazgos),
    }
