# -*- coding: utf-8 -*-
"""DE DÓNDE VIENE LA SENTENCIA RECLAMADA EN AMPARO DIRECTO (30-sep-2026).

David, sobre un proyecto que generó en soporte@ (AD 323/2025, un juicio oral
mercantil sobre el artículo 71 de la Ley Federal de Protección al Consumidor):
«Siempre, en amparo directo, partimos de la base de que hay una sala. Es decir,
una segunda instancia. Pero no siempre es así. La segunda instancia ocurre sólo
si hubo recurso de apelación […]. La autoridad responsable era el propio juez
de oralidad mercantil. En segundo lugar, la sentencia había sido dictada en
cumplimiento.»

El proyecto decía «la Sala responsable sí…», «la Sala atendió…», «la Sala no
eliminó…» de un JUZGADO. La causa era una sola línea: `tipos_asunto.SUJETOS`
fijaba «la Sala» como el órgano de TODO amparo directo, y de ahí bebían una
docena de prompts. Aquí se lee, del nombre de la responsable y de los autos:

  · QUÉ CLASE DE ÓRGANO dictó lo reclamado (juez, sala de alzada, sala del
    TFJA, junta, tribunal laboral) y, con eso, CÓMO SE LE NOMBRA en la prosa;
  · si hubo SEGUNDA INSTANCIA («alzada») o el juicio terminó en la primera
    («unica»: el oral mercantil —art. 1390 Bis del Código de Comercio—, un
    juicio de cuantía menor, un laudo, una sentencia de nulidad);
  · y si la sentencia reclamada se dictó EN CUMPLIMIENTO de una ejecutoria de
    amparo: cuál, y qué dijo que se hiciera (sus efectos), que es de donde sale
    qué quedó vinculado y qué se resolvió con libertad de jurisdicción.

LA REGLA DE SIEMPRE EN ESTE TALLER: ante la duda, vacío. Un hueco se ve y el
secretario lo corrige en «El asunto, en corto»; un origen equivocado se firma.
La clase de la responsable manda sobre el texto: si lo reclamado lo dictó un
juez, no hubo alzada aunque los autos cuenten una apelación de otra etapa.
"""
from __future__ import annotations

import re
import unicodedata

# ═══ LA CLASE DEL ÓRGANO ════════════════════════════════════════════════════


def _plano(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "").lower())
    return " ".join("".join(c for c in s if not unicodedata.combining(c)).split())


# El orden importa: lo más específico primero. Una «Sala Regional … del Tribunal
# Federal de Justicia Administrativa» es sala, pero de ÚNICA instancia (el
# juicio de nulidad); una «Sala Civil del Tribunal Superior de Justicia»
# resuelve la apelación.
#
# MEDIDO SOBRE LOS 4,697 AMPAROS DIRECTOS DEL 3er TCC XXII (30-sep-2026): la
# SALA SUPERIOR y las SECCIONES del Tribunal de Justicia Administrativa DEL
# ESTADO resuelven la REVISIÓN (179 asuntos, segunda instancia), mientras que
# la Sala Superior del TFJA y sus Salas Regionales resuelven el juicio de
# nulidad en única instancia (655). El Tribunal Unitario de Circuito es
# alzada (apelación federal); el Unitario AGRARIO, única instancia (145).
_CLASES = (
    ("sala_alzada", re.compile(r"\b(sala superior|seccion de la sala superior)\b.*\btribunal de justicia "
                               r"administrativa del estado\b|\bseccion de la sala superior\b")),
    ("sala_tfja", re.compile(r"\bsala\b.*\btribunal (federal )?de justicia administrativa\b|"
                             r"\bsala (regional|especializada|superior)\b")),
    ("sala_alzada", re.compile(r"\bsala\b")),
    ("tribunal_agrario", re.compile(r"\btribunal unitario agrario\b")),
    ("tribunal_alzada", re.compile(r"\btribunal (colegiado )?de (alzada|apelacion)\b|\bcolegiado de apelacion\b|"
                                   r"\btribunal unitario\b")),
    ("junta", re.compile(r"\bjunta\b.*\bconciliacion\b")),
    ("tribunal_laboral", re.compile(r"\btribunal\b.*\b(laboral|trabajo|conciliacion y arbitraje)\b")),
    ("jueza", re.compile(r"^jueza\b|\bla jueza\b")),
    ("juez", re.compile(r"\bjuez\b|\bjuzgado\b")),
)


def clase_de_organo(nombre: str) -> str:
    """«juez», «jueza», «sala_alzada», «sala_tfja», «tribunal_alzada», «junta»,
    «tribunal_laboral», «tribunal_agrario» o «» si el nombre no lo dice."""
    p = _plano(nombre)
    if not p:
        return ""
    for clase, rx in _CLASES:
        if rx.search(p):
            return clase
    return ""


# Cómo se nombra en la prosa, en el mismo orden que `tipos_asunto.SUJETOS`:
# (sujeto corto, con «responsable», genérico, abreviado). Así lo escribe el
# Tercer Tribunal Colegiado del XXII Circuito en sus amparos directos contra
# jueces de oralidad: «el Juez responsable», «la autoridad responsable».
SUJETOS_ORGANO = {
    "sala_alzada": ("la Sala", "la Sala responsable", "la autoridad responsable", "la responsable"),
    "sala_tfja": ("la Sala", "la Sala responsable", "la autoridad responsable", "la responsable"),
    "tribunal_alzada": ("el Tribunal de Alzada", "el Tribunal responsable", "la autoridad responsable",
                        "la responsable"),
    "juez": ("el Juez", "el Juez responsable", "la autoridad responsable", "la responsable"),
    "jueza": ("la Jueza", "la Jueza responsable", "la autoridad responsable", "la responsable"),
    "junta": ("la Junta", "la Junta responsable", "la autoridad responsable", "la responsable"),
    "tribunal_laboral": ("el Tribunal", "el Tribunal responsable", "la autoridad responsable",
                         "la responsable"),
    "tribunal_agrario": ("el Tribunal Unitario Agrario", "el Tribunal responsable",
                         "la autoridad responsable", "la responsable"),
}
# Si no consta quién es: una fórmula que no afirma nada. «la Sala» por omisión
# era justo el error.
SUJETOS_NEUTROS = ("la autoridad responsable", "la autoridad responsable",
                   "la responsable", "la responsable")

_UNICA = {"sala_tfja", "juez", "jueza", "junta", "tribunal_laboral", "tribunal_agrario"}
_ALZADA = {"sala_alzada", "tribunal_alzada"}


# ═══ LA INSTANCIA ═══════════════════════════════════════════════════════════

# La vía de un juicio que no admite apelación contra su sentencia. Sólo se usa
# cuando la clase del órgano no lo resolvió (nombre vacío o ilegible).
_VIA_UNICA = re.compile(
    r"juicio\s+oral\s+mercantil|v[íi]a\s+oral\s+mercantil|oralidad\s+mercantil|"
    r"1390\s+bis|no\s+proceder[áa]\s+recurso\s+ordinario|"
    r"\blaudo\b|juicio\s+(?:contencioso\s+administrativo|de\s+nulidad)", re.I)


def instancia_de(clase: str, texto: str = "") -> str:
    """«unica», «alzada» o «». La clase manda; el texto sólo si no hay clase."""
    if clase in _UNICA:
        return "unica"
    if clase in _ALZADA:
        return "alzada"
    s = " ".join(str(texto or "").split())
    if not s:
        return ""
    import fase_origen as _fo
    if _fo._RX_HAY_APELACION.search(s):
        return "alzada"
    if _VIA_UNICA.search(s):
        return "unica"
    return ""


# ═══ LA SENTENCIA DICTADA EN CUMPLIMIENTO ═══════════════════════════════════

# «en cumplimiento a la ejecutoria», «en cumplimiento de la ejecutoria dictada en
# el juicio de amparo directo civil 590/2023», «en acatamiento al fallo
# protector». Exige el OBJETO del cumplimiento —ejecutoria, sentencia de amparo,
# fallo protector— porque «en cumplimiento del contrato» o «en cumplimiento de la
# sentencia» (la de origen, en ejecución) son otra cosa.
_RX_CUMPL = re.compile(
    r"en\s+(?:cumplimiento|acatamiento)\s+(?:a|de|al|del)\s+(?:la\s+|lo\s+resuelto\s+en\s+la\s+|"
    r"(?:esa|dicha|la\s+citada|la\s+referida|la\s+mencionada)\s+)?"
    r"(?:ejecutoria|sentencia\s+de\s+amparo|resoluci[óo]n\s+de\s+amparo|fallo\s+protector|"
    r"concesi[óo]n\s+(?:del\s+|de\s+)?amparo|protecci[óo]n\s+constitucional)", re.I)

# El número de la ejecutoria, cerca de la mención: «amparo directo civil
# 590/2023», «A.D.C. 590/2023», «ADC 590/2023», «juicio de amparo 590/2023».
_RX_EJECUTORIA = re.compile(
    r"(?:(?:juicio\s+de\s+)?amparo\s+(?:directo|indirecto|en\s+revisi[óo]n)?\s*"
    r"(?:civil|mercantil|administrativo|laboral|penal|familiar|agrario)?\s*"
    r"(?:n[úu]mero\s+)?|\bA\.?\s?D\.?\s?C\.?\s+|\bA\.?\s?D\.?\s+)"
    r"(\d{1,5}\s*/\s*\d{4})", re.I)

# Dónde empiezan los efectos transcritos de esa ejecutoria.
_RX_EFECTOS = re.compile(
    r"para\s+(?:el\s+)?efecto\s+de\s+que|para\s+los\s+(?:siguientes\s+)?efectos|"
    r"efectos\s+(?:de\s+la\s+concesi[óo]n|del\s+amparo|de\s+la\s+protecci[óo]n)|"
    r"los\s+efectos\s+(?:del\s+amparo\s+)?(?:son|fueron|consisten)", re.I)


def _nombre_ejecutoria(ventana: str) -> str:
    m = _RX_EJECUTORIA.search(ventana)
    if not m:
        return ""
    frase = " ".join(m.group(0).split())
    num = re.sub(r"\s+", "", m.group(1))
    tipo = "amparo directo"
    fp = _plano(frase)
    if "indirecto" in fp:
        tipo = "amparo indirecto"
    elif "revision" in fp:
        tipo = "amparo en revisión"
    materia = next((m_ for m_ in ("civil", "mercantil", "administrativo", "laboral", "penal", "familiar",
                                  "agrario") if m_ in fp), "")
    if re.match(r"a\.?\s?d\.?\s?c", fp):
        materia = materia or "civil"
    return " ".join(x for x in (tipo, materia, num) if x)


# LAS TRAMPAS MEDIDAS en los 193 casos del 3er TCC (30-sep-2026):
#   · «dar cumplimiento a la ejecutoria de fecha … dictada dentro de los autos
#     del toca»: esa «ejecutoria» es la sentencia de APELACIÓN firme, no un
#     amparo → la ejecutoria sola exige «amparo» cerca;
#   · lo que la quejosa «refiere/aduce/afirma» que se dictó en cumplimiento;
#   · «en cumplimiento al Acuerdo General…».
_RX_ALEGA = re.compile(r"(?:refiere|aduce|afirma|sostiene|alega|argumenta)[^.]{0,40}$", re.I)
_RX_AMPARO_CERCA = re.compile(r"amparo|protector|protecci[óo]n\s+constitucional|concesi[óo]n", re.I)


#   · y, las más, TESIS TRANSCRITAS y cumplimientos FUTUROS: «…en el nuevo acto
#     que emita en cumplimiento a la ejecutoria de amparo», «los motivos que la
#     sala aduzca en cumplimiento…». Calibrado sobre el corpus (precisión 0.52
#     con la sola frase): se exige además un NÚMERO de amparo cerca, un verbo
#     de emisión en PASADO y que no lo preceda un subjuntivo o un futuro.
#     Medido sobre el primer 45% de los 4,697 AD (donde van los antecedentes):
#     0.52 → 0.77 de precisión contra la referencia del corpus, con 0.66 de
#     cobertura; leídos a mano, casi todos los «falsos positivos» que quedan
#     SON sentencias en cumplimiento que la referencia no contó («37. Sentencia
#     reclamada. En cumplimiento al fallo protector, la Sala responsable dictó
#     la resolución de…»). El secretario lo confirma o lo corrige en pantalla.
_RX_NUMERO_CERCA = re.compile(r"\b\d{1,5}\s*/\s*\d{4}\b")
_RX_DICTADO = re.compile(
    r"\b(?:dict[óo]|emiti[óo]|pronunci[óo]|resolvi[óo]|se\s+dej[óo]\s+insubsistente|dej[óo]\s+insubsistente|"
    r"dejando\s+insubsistente|(?:fue|ha\s+sido|es)\s+(?:dictad|emitid|pronunciad)[oa]|"
    r"(?:nueva|otra)\s+(?:sentencia|resoluci[óo]n)|la\s+sentencia\s+reclamada|el\s+acto\s+reclamado|"
    r"dictad[oa]\s+en\s+cumplimiento|emitid[oa]\s+en\s+cumplimiento|"
    # Como abre la propia sentencia de cumplimiento: «En cumplimiento a la
    # ejecutoria dictada en el amparo directo… en la que se concedió el amparo
    # para el efecto de que…», «…se dicta la presente resolución».
    r"se\s+(?:dicta|emite|pronuncia)|procede\s+a\s+dictar|concedi[óo]\s+(?:el\s+amparo|la\s+protecci)|"
    r"para\s+(?:el\s+|los\s+)?efectos?\s+(?:de\s+que|siguientes))", re.I)
_RX_EMISION_TRAS = re.compile(
    r"\b(?:emiti[óo]|dict[óo]|pronunci[óo]|resolvi[óo]|se\s+dict[óo]|se\s+emiti[óo]|dej[óo]\s+insubsistente|"
    r"se\s+dej[óo]\s+insubsistente|constituye\s+el\s+acto\s+reclamado)\b", re.I)
_RX_NO_PASADO = re.compile(
    r"(?:que\s+(?:emita|dicte|pronuncie|resuelva)|aduzca|deber[áa]|podr[áa]|habr[áa]\s+de|a\s+fin\s+de)[^.]{0,85}$",
    re.I)


def _es_cumplimiento(s: str, m) -> bool:
    if "ejecutoria" in m.group(0).lower():
        tras = s[m.end():m.end() + 220]
        # Con demostrativo («esa ejecutoria») el amparo se nombró ANTES.
        cerca = tras if not re.search(r"(?:esa|dicha|citada|referida|mencionada)\s+ejecutoria", m.group(0), re.I) \
            else s[max(0, m.start() - 300):m.end() + 220]
        if not _RX_AMPARO_CERCA.search(cerca) or re.search(r"\btoca\b", tras[:120], re.I) and not \
                re.search(r"amparo", tras[:120], re.I):
            return False
    antes = s[max(0, m.start() - 90):m.start()]
    if _RX_ALEGA.search(antes) or _RX_NO_PASADO.search(antes):
        return False
    # LO QUE PRUEBA QUE SE DICTÓ: un verbo de emisión en pasado justo después
    # («En cumplimiento al fallo protector, la Sala responsable emitió la
    # sentencia de…»), o bien el número de la ejecutoria y la emisión en la
    # ventana. Una tesis transcrita no trae ni lo uno ni lo otro.
    if _RX_EMISION_TRAS.search(s[m.end():m.end() + 200]):
        return True
    ventana = s[max(0, m.start() - 250):m.end() + 250]
    if not _RX_NUMERO_CERCA.search(s[m.start():m.end() + 250]) and not _RX_NUMERO_CERCA.search(
            s[max(0, m.start() - 200):m.start()]):
        return False
    return bool(_RX_DICTADO.search(ventana))


def cumplimiento_de(*textos: str) -> dict:
    """{consta, ejecutoria, fragmento, efectos} leídos de los textos dados, en
    orden de preferencia (antecedentes, resumen del acto, el acto mismo).

    `efectos` es el tramo donde el acto transcribe lo que la ejecutoria mandó,
    si lo transcribe; vacío si no. Sin él no se puede saber qué quedó vinculado:
    se pide como constancia, no se supone."""
    out = {"consta": False, "ejecutoria": "", "fragmento": "", "efectos": ""}
    for texto in textos:
        s = " ".join(str(texto or "").split())
        if not s:
            continue
        m = next((x for x in _RX_CUMPL.finditer(s) if _es_cumplimiento(s, x)), None)
        if not m:
            continue
        a, b = max(0, m.start() - 250), min(len(s), m.end() + 350)
        ventana = s[a:b]
        out["consta"] = True
        out["fragmento"] = out["fragmento"] or ventana.strip()
        out["ejecutoria"] = out["ejecutoria"] or _nombre_ejecutoria(s[m.start():m.end() + 350]) \
            or _nombre_ejecutoria(ventana)
        if not out["efectos"]:
            e = _RX_EFECTOS.search(s, max(0, m.start() - 400))
            if e and e.start() - m.end() < 6000:
                out["efectos"] = s[e.start():e.start() + 3000].strip()
    return out


# ═══ TODO JUNTO ═════════════════════════════════════════════════════════════


def origen(responsable: str = "", *textos: str, tipo_asunto: str = "amparo_directo",
           manual: dict | None = None) -> dict:
    """El origen del acto reclamado. `textos`: antecedentes, resumen del acto y
    el acto mismo (en ese orden de confianza). `manual` es lo que el secretario
    corrigió en pantalla y manda sobre lo leído.

    {clase, instancia, sujetos, lo_resuelto, cumplimiento, fuente}"""
    import fase_origen as _fo
    clase = clase_de_organo(responsable)
    juntos = " ".join(str(t or "") for t in textos[:2])
    inst = instancia_de(clase, juntos)
    out = {
        "clase": clase,
        "instancia": inst,
        "sujetos": list(SUJETOS_ORGANO.get(clase, SUJETOS_NEUTROS)),
        "lo_resuelto": _fo.lo_resuelto(juntos, tipo_asunto) if inst != "unica"
        else (_fo.lo_resuelto(juntos, tipo_asunto) or "ese juicio"),
        "cumplimiento": cumplimiento_de(*textos),
        "fuente": "leido",
    }
    if isinstance(manual, dict) and manual:
        if manual.get("instancia") in ("unica", "alzada"):
            out["instancia"] = manual["instancia"]
        elif manual.get("instancia") == "no_consta":
            # Él dijo que NO CONSTA: no se supone ni una ni otra, aunque la
            # lectura creyera saberlo (antes «» volvía a lo leído).
            out["instancia"] = ""
        if isinstance(manual.get("cumplimiento"), bool):
            out["cumplimiento"] = dict(out["cumplimiento"], consta=manual["cumplimiento"])
        for k in ("ejecutoria", "efectos"):
            if str(manual.get(k) or "").strip():
                out["cumplimiento"][k] = str(manual[k]).strip()
        # EL SOBRESEIMIENTO POR CUMPLIMIENTO SÓLO LO CONFIRMA ÉL: quitar el
        # estudio de fondo no lo decide una lectura (ver `cumplimiento_ejecutoria`).
        if manual.get("sobreseer") is True:
            out["sobreseer_confirmado"] = True
        out["fuente"] = "secretario"
    # Con una sola instancia, «ese recurso de apelación» no puede ser lo
    # resuelto (después de lo manual: la corrección también cuenta).
    if out["instancia"] == "unica" and "apelaci" in out["lo_resuelto"]:
        out["lo_resuelto"] = "ese juicio"
    return out


def aviso(o: dict) -> str:
    """Una línea para «El asunto, en corto» y para los prompts."""
    if not isinstance(o, dict):
        return ""
    partes = []
    if o.get("instancia") == "unica":
        partes.append("dictada en ÚNICA instancia (no hubo apelación)")
    elif o.get("instancia") == "alzada":
        partes.append("dictada en segunda instancia (resolvió un recurso de apelación)")
    c = o.get("cumplimiento") or {}
    if c.get("consta"):
        partes.append("en CUMPLIMIENTO de la ejecutoria del " + (c.get("ejecutoria") or "amparo anterior"))
    return "La sentencia reclamada fue " + " y ".join(partes) + "." if partes else ""
