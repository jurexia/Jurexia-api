# -*- coding: utf-8 -*-
"""La ley aplicable la decide la MATERIA, no el lugar que se nombra (28-sep-2026).

POR QUÉ EXISTE
--------------
Seis consultas de prueba para el paquete de anuncios, escritas como las
escribiría alguien que no es abogado, todas «en Querétaro» y con el esfuerzo
Básico. Cuatro salieron con la ley equivocada, y las cuatro por la misma causa:
el nombre del estado arrastraba su legislación a la materia federal, y la
legislación federal a la materia del estado.

  · pagaré            → juicio ejecutivo mercantil fundado en el Código de
                        Procedimientos Civiles y en el Código Civil del estado.
                        Lo rige el Código de Comercio.
  · despido           → demanda directa al tribunal laboral, sin la conciliación
                        prejudicial obligatoria (art. 684-B LFT), y los veinte
                        días por año del art. 50 sumados a la indemnización del 48.
  · contrato de renta → el Código Civil Federal «aplicable supletoriamente» al del
                        estado, quince veces. No es supletorio de ningún código
                        estatal.
  · pensión           → «vía de juicio ordinario civil», cuando el código procesal
                        del estado la manda sumaria, y el Código Nacional citado
                        como «supletorio».

Dos de las cuatro las ordenaba el propio sistema: la inyección por estado decía
«Las leyes federales (Código Civil Federal, etc.) son SUPLETORIAS», y el prompt de
redacción pedía entrelazar «legislación federal, estatal…» en cinco a ocho fuentes.

Este módulo no busca nada ni llama a ningún modelo. Decide la materia con
señales de alta precisión en el texto, con el dictamen del Estratega como
respaldo, y redacta las reglas. Vale para las 32 entidades: el nombre del estado
sólo se interpola donde decide la AUTORIDAD, nunca donde decidiría la ley.

Si una señal no es segura, el módulo calla: sin materia no hay regla dinámica, y
el escrito queda con las reglas generales del prompt, que son condicionales.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Optional, Set

MATERIAS = ("mercantil", "laboral", "familiar", "civil")
FEDERALES = frozenset({"mercantil", "laboral"})
LOCALES = frozenset({"familiar", "civil"})


def _plano(texto) -> str:
    """Minúsculas y sin acentos: las señales se escriben una sola vez."""
    t = unicodedata.normalize("NFD", str(texto or "").lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def _re(*patrones: str) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patrones))


# ── SEÑALES ───────────────────────────────────────────────────────────────────
# Sólo las que no admiten otra lectura. «Cheque» se quedó fuera: el librado sin
# fondos también es delito en varios códigos penales. «Renta» sola tampoco: es
# el impuesto sobre la renta. «Sociedad anónima» tampoco: casi siempre es el
# patrón de un despido. Lo dudoso lo decide el Estratega, o nadie.
_SENALES = {
    "mercantil": _re(
        r"\bpagare(s)?\b",
        r"\bletras? de cambio\b",
        r"\btitulos? de credito\b",
        r"\baccion cambiaria\b",
        # No «mercantil» a secas: «la clausura de un establecimiento
        # mercantil» es derecho administrativo (comparar.py lo trae).
        r"\b(juicio|via|materia|accion|controversia|contrato|operacion|acto)s? "
        r"(ejecutiv[oa] |oral |ordinari[oa] )?mercantil(es)?\b",
        r"\bactos? de comercio\b",
        r"\bcodigo de comercio\b",
        r"\barrendamiento financiero\b",
    ),
    "laboral": _re(
        r"\bme (corrieron|despidieron|dieron de baja|liquidaron)\b",
        r"\bdespid(o|os|ieron|ida|ido)\b",
        r"\bfiniquito\b",
        r"\b(mi|su|el|la) (patron|patrona|empleador|empleadora)\b",
        r"\brelacion (laboral|de trabajo)\b",
        r"\bsalarios? (caidos|vencidos)\b",
        r"\breinstala(cion|r|ran)\b",
        r"\bcentros? (federal )?de conciliacion\b",
        r"\btribunal(es)? laboral(es)?\b",
        r"\bley federal del trabajo\b",
        r"\brescision\b.{0,40}\b(laboral|trabajador|trabajo)\b",
        r"\b(trabajador|trabajadora)\b.{0,60}\b(falto|faltas|rescision|despido|renuncia)\b",
        r"\b(aguinaldo|prima vacacional|prima de antiguedad|horas extras?)\b",
    ),
    "familiar": _re(
        r"\bpension (alimenticia|de alimentos)\b",
        r"\b(pension|alimentos|manutencion)\b.{0,80}\b(hij[oa]s?|menor(es)?|nin[oa]s?|bebe)\b",
        r"\b(hij[oa]s?|menor(es)?|nin[oa]s?|bebe)\b.{0,80}\b(pension|alimentos|manutencion)\b",
        r"\b(deudor|acreedor)(a|es)? alimentari",
        r"\bdivorci",
        r"\bguarda y custodia\b",
        r"\bpatria potestad\b",
        r"\bregimen de (visitas|convivencia)\b",
        r"\bviolencia familiar\b",
    ),
    "civil": _re(
        r"\barrendamiento\b",
        r"\b(arrendador|arrendatari[oa]|inquilin[oa]s?)\b",
        r"\brent(ar|o|a|e|arle|arla)\b.{0,40}\b(casa|departamento|local|inmueble|terreno|cuarto|bodega|oficina|propiedad)\b",
        r"\bdesahucio\b",
        r"\b(usucapion|prescripcion adquisitiva)\b",
        r"\bcompraventa\b.{0,40}\b(casa|departamento|inmueble|terreno|predio|lote)\b",
        r"\bservidumbre\b",
        r"\bcopropiedad\b",
    ),
}

# El que presta servicios al Estado no se rige por la Ley Federal del Trabajo:
# lo rige su ley burocrática, federal o local (arts. 116-VI y 123-B CPEUM), y a
# la policía, el derecho administrativo (123-B-XIII). Con cualquiera de estas
# señales el módulo no afirma la LFT.
_PUBLICO = _re(
    r"\bburocrat",
    r"\bservidor(a|es)? public",
    r"\bal servicio del estado\b",
    r"\b(trabajador|trabajadora|empleado|empleada)(es|s)? (del|de un|de la) (estado|gobierno|municipio|ayuntamiento|dependencia)\b",
    r"\bayuntamiento\b",
    r"\bgobierno (del estado|estatal|municipal|federal)\b",
    r"\bpolic(ia|ias)\b",
    r"\bseguridad publica\b",
    r"\bissste\b",
    r"\bpoder judicial\b",
    r"\bmagisterio\b",
    r"\bsecretaria de educacion\b",
)
# En un amparo manda la Ley de Amparo y la jerarquía constitucional, sea cual
# sea la materia del acto: el módulo calla y deja el bloque de siempre. Si no
# callara, un amparo contra una sentencia mercantil recibiría la orden de
# «fundar la acción y la vía en el Código de Comercio».
_AMPARO = re.compile(r"\bamparo\b")

# «Arrendamiento financiero» es mercantil (LGTOC) aunque lleve la palabra.
_ARR_FINANCIERO = re.compile(r"\barrendamiento financiero\b|\barrendadora financiera\b")


def materias_en_texto(texto) -> Set[str]:
    """Las materias que el texto acusa sin lugar a dudas."""
    t = _plano(texto)
    halladas = {m for m, patron in _SENALES.items() if patron.search(t)}
    if "civil" in halladas and _ARR_FINANCIERO.search(t):
        # Sólo si «civil» salió por el arrendamiento: se vuelve a mirar sin él.
        if not _SENALES["civil"].search(_ARR_FINANCIERO.sub(" ", t)):
            halladas.discard("civil")
    return halladas


def es_servicio_publico(texto) -> bool:
    return bool(_PUBLICO.search(_plano(texto)))


def _normal(m: Optional[str]) -> Optional[str]:
    m = _plano(m).strip()
    return m if m in MATERIAS else None


def materia(texto, estratega: Optional[str] = None, elegida: Optional[str] = None) -> Optional[str]:
    """La materia que decide la ley aplicable, o None si no hay certeza.

    Orden: la materia que el usuario ELIGIÓ en la interfaz manda, como en el
    resto del sistema (si eligió otra que no es de las cuatro, se calla); luego
    las señales del texto, y el Estratega sólo desempata o suple. Con amparo no
    se afirma ninguna; lo laboral con señal de servicio público, tampoco.
    """
    if _AMPARO.search(_plano(texto)):
        return None
    if elegida and str(elegida).strip():
        m = _normal(elegida)
    else:
        halladas = materias_en_texto(texto)
        del_estratega = _normal(estratega)
        if len(halladas) == 1:
            m = next(iter(halladas))
        elif len(halladas) > 1:
            m = del_estratega if del_estratega in halladas else None
        else:
            m = del_estratega
    if m == "laboral" and es_servicio_publico(texto):
        return None
    return m


def es_federal(m: Optional[str]) -> bool:
    return m in FEDERALES


def es_local(m: Optional[str]) -> bool:
    return m in LOCALES


# ── EL CÓDIGO NACIONAL, ENTIDAD POR ENTIDAD ───────────────────────────────────
# El Código Nacional de Procedimientos Civiles y Familiares entra en cada
# entidad por declaratoria de su congreso, por fases (distrito, materia, tipo de
# procedimiento), y rige sólo los asuntos que empiezan después; el plazo legal
# vence el 1 de abril de 2027. Una regla general no puede decir cuál de los dos
# códigos rige: en la CDMX el nacional rige ya todo lo nuevo, y en Querétaro
# apenas un distrito. Aquí sólo entran entidades VERIFICADAS, con su fuente; las
# demás reciben la regla general, que no toma partido.
#
# REVISAR antes del 1-abr-2027 y con cada declaratoria nueva. Verificado el
# 28-sep-2026.
VIGENCIA_CNPCF = {
    "CDMX": {
        "nombre": "la Ciudad de México",
        "alcance": "plena",
        "texto": ("rige todos los asuntos civiles y familiares que se inician desde el 15 de noviembre "
                  "de 2025 (entró por fases desde el 1 de diciembre de 2024)"),
        "fuente": "DOF, Declaratoria de Vigencia del CNPCF en la Ciudad de México (SIDOF 5737237)",
    },
    "QUERETARO": {
        "nombre": "Querétaro",
        "alcance": "parcial",
        "texto": ("rige desde el 10 de septiembre de 2026 sólo en el Distrito Judicial de San Juan del "
                  "Río y para determinados procedimientos; en los demás distritos los asuntos se "
                  "tramitan conforme al Código de Procedimientos Civiles del Estado de Querétaro"),
        "excepcion": r"\bsan juan del rio\b",
        "fuente": ("Poder Judicial del Estado de Querétaro: primera etapa de la justicia civil y "
                   "familiar oral en San Juan del Río (10-sep-2026); la legislatura la aplazó a septiembre"),
    },
}
_ALIAS_ENTIDAD = {"CIUDAD_DE_MEXICO": "CDMX", "DISTRITO_FEDERAL": "CDMX"}
_CNPCF = re.compile(r"\bcodigo nacional de procedimientos civiles\b")
_PIDE_CNPCF = re.compile(r"\bcodigo nacional\b|\boral(es)?\b")


def vigencia_cnpcf(estado) -> Optional[dict]:
    """Lo verificado sobre el Código Nacional en esa entidad, o None."""
    clave = _plano(estado).upper().strip().replace(" ", "_")
    return VIGENCIA_CNPCF.get(_ALIAS_ENTIDAD.get(clave, clave))


def instruccion_procesal(m: Optional[str], estado) -> Optional[str]:
    """Qué código procesal rige un asunto local en esa entidad, si está verificado."""
    if not es_local(m):
        return None
    v = vigencia_cnpcf(estado)
    if not v:
        return None
    lugar = v["nombre"]
    cabeza = ("CÓDIGO PROCESAL APLICABLE (criterio interno: aplícalo sin explicarlo ni mencionarlo).\n"
              f"En {lugar}, el Código Nacional de Procedimientos Civiles y Familiares {v['texto']}. ")
    if v["alcance"] == "plena":
        return cabeza + ("Un asunto que se inicia ahora se funda en él —vía, procedimiento y medidas—, no en "
                         "el código procesal anterior de la entidad, que sólo rige los asuntos iniciados "
                         "antes de esa fecha. No mezcles artículos de los dos.")
    return cabeza + ("Salvo que el usuario diga que su asunto es de ese distrito, funda la vía, el "
                     "procedimiento y las medidas sólo en el código procesal del estado y no cites el "
                     "Código Nacional.")


# ── LO QUE NO RIGE NO ENTRA AL CONTEXTO ───────────────────────────────────────
# La regla del prompt no bastó sola. En la primera corrida con el arreglo, el
# contrato de renta aún citó dos veces el Código Civil Federal («aplicable
# supletoriamente a la materia civil local»): sus quince artículos eran la ÚNICA
# ley federal del contexto. Y el despido se escribió para un burócrata del
# gobierno del estado porque los seis documentos locales que sobrevivieron al
# freno eran de la ley de los trabajadores del estado, más dos de la federal del
# apartado B. Es la lección de fuentes_elegidas.py: lo que el modelo tiene
# delante lo usa; el filtro va antes que la instrucción.
_SUPLETORIOS_AJENOS = re.compile(
    r"\bcodigo civil federal\b|\bcodigo federal de procedimientos civiles\b")
_APARTADO_B = re.compile(
    r"trabajadores al servicio del estado|fraccion xiii bis del apartado b"
    r"|remuneraciones de los servidores publicos")
_PIDE_FEDERAL = re.compile(r"\bfederal(es)?\b")


def es_ley_ajena(m: Optional[str], silo: Optional[str], origen: Optional[str], texto,
                 estado=None) -> bool:
    """True si el documento es una ley que no rige un asunto de la materia `m`.

    · laboral (patrón particular: `materia` ya descartó el servicio público):
      fuera la ley del estado y la federal del apartado B;
    · mercantil: fuera la ley del estado;
    · civil y familiar: fuera el Código Civil Federal y el Código Federal de
      Procedimientos Civiles, salvo que el usuario hable de lo federal; y el
      Código Nacional donde su vigencia es PARCIAL (VIGENCIA_CNPCF), salvo que
      el usuario lo nombre, hable de oralidad o del distrito donde ya rige.
    La jurisprudencia, la Constitución y los tratados no se tocan nunca.
    """
    s = (silo or "").strip().lower()
    if not s.startswith("leyes_"):
        return False
    o = _plano(origen)
    local = s != "leyes_federales"
    if m == "laboral":
        return local or bool(_APARTADO_B.search(o))
    if m == "mercantil":
        return local
    if m in LOCALES and not local:
        t = _plano(texto)
        if _SUPLETORIOS_AJENOS.search(o):
            return not _PIDE_FEDERAL.search(t)
        if _CNPCF.search(o):
            v = vigencia_cnpcf(estado)
            if v and v["alcance"] == "parcial":
                excepcion = v.get("excepcion")
                return not (_PIDE_CNPCF.search(t) or (excepcion and re.search(excepcion, t)))
    return False


# ── LA REGLA DINÁMICA (materia federal con un estado nombrado) ───────────────
# Sustituye, para estas dos materias, al bloque genérico de «fuero federal
# detectado», que ordena NO mencionar el estado: en un escrito, el estado es
# justamente el que dice ante qué juez o centro se presenta. Lo que no se dice
# es por qué no se citan sus leyes.
def instruccion(m: Optional[str], entidad: Optional[str]) -> Optional[str]:
    """El bloque para el prompt cuando la materia es federal, o None."""
    if not es_federal(m):
        return None
    entidad = (entidad or "").strip()
    lugar = entidad or "el estado"
    del_lugar = f"de {entidad}" if entidad else "del estado"
    if m == "mercantil":
        return (
            "LEY APLICABLE — MATERIA MERCANTIL (criterio interno: aplícalo sin explicarlo ni mencionarlo).\n"
            f"Que el asunto ocurra en {lugar} decide ante qué juez se presenta, no qué ley lo rige: el acto "
            "y el título son mercantiles y se rigen por el Código de Comercio y las leyes mercantiles "
            f"especiales, también cuando conoce un juez {del_lugar}.\n"
            "1. Funda la acción, la vía y cada etapa del procedimiento en el Código de Comercio y, si hay "
            "títulos de crédito, en la Ley General de Títulos y Operaciones de Crédito.\n"
            "2. Lo que esas leyes no regulen se suple como ordena el propio Código de Comercio: en el fondo, "
            "con el Código Civil Federal (artículo 2); en el procedimiento, con la ley que señala su "
            "artículo 1054.\n"
            f"3. El código civil y el código de procedimientos civiles {del_lugar} NO fundan nada en este "
            "asunto: si aparecen en el contexto, no los cites. Del lugar sólo se toma la autoridad ante la "
            "que se presenta el escrito.\n"
            "4. Transcribe los artículos federales y la jurisprudencia con su [Doc ID: uuid]."
        )
    return (
        "LEY APLICABLE — MATERIA LABORAL (criterio interno: aplícalo sin explicarlo ni mencionarlo).\n"
        f"Que el asunto ocurra en {lugar} decide ante qué Centro de Conciliación y qué tribunal se acude, "
        "no qué ley rige: la relación de trabajo se rige por la Ley Federal del Trabajo, también ante las "
        f"autoridades {del_lugar}.\n"
        "1. El patrón es particular salvo que el usuario diga que trabaja para el gobierno o para una "
        f"dependencia pública: no apliques leyes burocráticas, federales ni {del_lugar}, ni ley civil o "
        "procesal del estado, aunque aparezcan en el contexto. Si pide lo que tiene que presentar, "
        "entrégale el escrito de parte, no un estudio ni una resolución.\n"
        "2. Antes de demandar es obligatoria la conciliación prejudicial (artículos 684-A y siguientes), "
        "salvo los conflictos que exceptúa el artículo 685 Ter. Se solicita ante el Centro de Conciliación "
        f"{del_lugar}, o ante el Centro Federal de Conciliación y Registro Laboral si la empresa o la rama "
        "son de jurisdicción federal (artículo 527). Si el usuario no dice que ya concilió o que ya tiene "
        "la constancia de no conciliación, el escrito que redactas es la solicitud de conciliación "
        "(artículo 684-C). Su destinatario es el Centro de Conciliación y así se escribe el encabezado: la "
        "solicitud no se dirige a ningún juez ni tribunal, porque ninguno conoce de ella. La demanda ante "
        "el Tribunal Laboral va después y se acompaña de esa constancia (artículo 872, apartado B, "
        "fracción I).\n"
        "3. El plazo para reclamar un despido es de dos meses desde la separación, y la solicitud de "
        "conciliación lo suspende (artículo 518).\n"
        "4. Cada prestación se reclama con el artículo que la otorga para ese supuesto. En el despido "
        "injustificado la acción es de reinstalación o de indemnización de tres meses de salario, con "
        "salarios vencidos hasta por doce meses e intereses (artículo 48). La indemnización de veinte días "
        "por año del artículo 50 sólo procede en los supuestos en que la ley remite a él —como el patrón "
        "eximido de reinstalar (artículo 49) o la rescisión por causa imputable al patrón (artículo 52)—; "
        "no se suma por sí sola a la del artículo 48.\n"
        "5. Transcribe los artículos de la Ley Federal del Trabajo y la jurisprudencia con su [Doc ID: uuid]."
    )


# ── EL ARTÍCULO QUE FIJA LA VÍA ───────────────────────────────────────────────
# El escrito de la pensión dijo «vía ordinaria» porque el artículo del código
# procesal que manda sumarios los alimentos nunca llegó al contexto: la consulta
# habla de un papá que dejó de pagar, no de vías. Esta consulta sale a buscarlo
# con el vocabulario con el que los códigos lo escriben —sumario, especial,
# controversia del orden familiar—, que es distinto en cada entidad.
_CONCEPTOS_DE_VIA = (
    ("alimentos", _re(r"\bpension\b", r"\balimentos\b", r"\bmanutencion\b", r"\balimentari")),
    ("arrendamiento", _re(r"\barrendamiento\b", r"\b(arrendador|arrendatari[oa]|inquilin[oa]s?)\b",
                          r"\brent(ar|o|a|e|arle|arla)\b", r"\bdesahucio\b")),
    ("divorcio", _re(r"\bdivorci")),
    ("guarda y custodia", _re(r"\bguarda y custodia\b", r"\bcustodia\b", r"\bpatria potestad\b",
                              r"\bregimen de (visitas|convivencia)\b", r"\bconvivencias?\b")),
)


def conceptos_de_via(texto, m: Optional[str]) -> list:
    """Los conceptos cuya vía hay que fundar, sólo en materia local."""
    if not es_local(m):
        return []
    t = _plano(texto)
    return [c for c, patron in _CONCEPTOS_DE_VIA if patron.search(t)]


def consulta_de_via(conceptos: Iterable[str]) -> Optional[str]:
    """El texto con que se busca en el código procesal el artículo de la vía."""
    cs = [c for c in conceptos if c]
    if not cs:
        return None
    que = ", ".join(cs[:-1]) + (" y " if len(cs) > 1 else "") + cs[-1]
    # «Vía oral familiar» es como lo dice el Código Nacional (art. 663) y
    # «se tramitarán oralmente», códigos como el de Sonora.
    return (f"Se tramitarán sumariamente, oralmente o en vía especial los juicios de {que}; "
            f"vía sumaria, vía oral familiar, juicio especial o controversia del orden familiar sobre {que}")


# ── LAS REGLAS DEL PROMPT DE REDACCIÓN ────────────────────────────────────────
# Generales, para las 32 entidades y las tres potencias de redacción. Sin
# ejemplos que se puedan copiar: describen qué hacer, no el texto que sale.
REGLAS_REDACCION = """
────────────────────────────────────────────────────────────────
 LEY APLICABLE: LA DECIDE LA MATERIA, NO EL LUGAR
────────────────────────────────────────────────────────────────

- El lugar que menciona el usuario decide ante qué autoridad se presenta el escrito; la MATERIA decide qué ley lo rige. Antes de fundar, fija la materia y cita sólo las leyes que la rigen. Una ley que no rige el caso no se cita aunque esté en el contexto, y ninguna se nombra «supletoria» si la ley que la necesitaría no la llama.
- MERCANTIL (actos de comercio, títulos de crédito, sociedades): se rige por el Código de Comercio y las leyes mercantiles especiales, también cuando conoce un juez del estado. Lo que esas leyes no regulen se suple como ordena el propio Código de Comercio: en el fondo, con el Código Civil Federal (artículo 2); en el procedimiento, con la ley que señala su artículo 1054. El código civil y el código de procedimientos civiles del estado no fundan la acción, la vía ni el procedimiento mercantil.
- LABORAL (relaciones de trabajo del apartado A del artículo 123 constitucional): se rige por la Ley Federal del Trabajo en todo el país, también ante los centros y tribunales de los estados. Antes de demandar es obligatoria la conciliación prejudicial ante el Centro de Conciliación competente (artículos 684-A y siguientes), salvo los conflictos que exceptúa el artículo 685 Ter; la solicitud se dirige a ese Centro, no a un juez, y la demanda va después y se acompaña de la constancia de no conciliación (artículo 872, apartado B, fracción I). Cada prestación se reclama con el artículo que la otorga para ese supuesto: la indemnización de veinte días por año del artículo 50 sólo procede en los supuestos en que la ley remite a él —como el patrón eximido de reinstalar (artículo 49) o la rescisión por causa imputable al patrón (artículo 52)— y no se suma por sí sola a la del artículo 48. Si el usuario no dice que trabaja para una dependencia o entidad pública, el patrón es particular y rige la Ley Federal del Trabajo; las leyes burocráticas sólo rigen a quien trabaja para el Estado.
- CIVIL Y FAMILIAR (fuero común): se rigen por el código civil o familiar del estado y por su código procesal. El Código Civil Federal y el Código Federal de Procedimientos Civiles no son supletorios de los códigos de un estado: no los cites como fundamento ni como supletorios de la ley local. Lo federal entra sólo cuando rige por sí mismo: la Constitución, los tratados y las leyes generales aplicables.
- UN SOLO CÓDIGO PROCESAL. El Código Nacional de Procedimientos Civiles y Familiares entra en cada estado por declaratoria, a menudo por distritos, por materias y por tipos de procedimiento, y rige sólo los asuntos que se inician después; donde rige sustituye al código procesal del estado, y donde no rige no se cita: nunca es supletorio de él. Funda la vía, el procedimiento y las medidas en un solo código, el que rija para ese asunto en ese lugar; si el contexto indica cuál rige, síguelo, y no mezcles artículos de los dos.
- LA VÍA SE FUNDA, NO SE SUPONE. La vía la fija el artículo del código procesal que la establece para esa pretensión: localízalo en el contexto y cítalo en el proemio. La vía ordinaria es residual: antes de elegirla, comprueba que el código no prevea para esa pretensión una vía sumaria, especial o de controversia familiar.
"""
