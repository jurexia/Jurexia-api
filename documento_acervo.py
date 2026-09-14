# -*- coding: utf-8 -*-
"""El prompt del análisis de documentos, y la consulta con que busca en el acervo.

HASTA EL 14-SEP-2026 ESTA RUTA NO TENÍA ACERVO. `/analyze-document` recibía el
documento y nada más; el prompt lo decía con todas las letras —«NO tienes
acervo que consultar»— y prohibía citar tesis y números de artículo que no
estuvieran en el documento. Medido sobre 25 días: 1.441 respuestas, 11 con
[Doc ID], el 73% con números de artículo sin ninguna ancla y el 35% con tesis
sin ancla. La regla en prosa frenaba poco (del 74% al 66%), porque una regla
no sustituye a la fuente. Por esa puerta entraron los diez artículos «de
Puebla» que eran federales, las tesis inexistentes de Edgar Mata y el «no
está tipificado» del 200 bis de Baja California Sur, que sí lo estaba.

Ahora la ruta busca en el acervo —misma maquinaria que el chat, filtrada por
la entidad del abogado— y el prompt cambia de «no tienes acervo» a «cita sólo
del acervo». Las dos variantes viven aquí:

  · `prompt_documento(con_acervo=False)` es el texto de siempre, sin tocar:
    es lo que se usa cuando el acervo no responde o no devuelve nada, porque
    un acervo caído no puede volverse una invitación a inventar.
  · `prompt_documento(con_acervo=True)` trae las reglas de cita del chat que
    importan aquí: Doc ID obligatorio, transcripción literal, prohibido mudar
    un artículo de una ley a otra, y qué decir cuando el fundamento no está.

`consulta_para_acervo` arma la búsqueda: la instrucción del abogado, los
ordenamientos que el documento invoca y su arranque, que es donde viven el
tipo de asunto, las partes y el fundamento. Es una función pura para poder
probarla sin red.
"""
import re
from collections import Counter
from typing import List

_BASE = """Eres Iurexia, un asistente jurídico de alto nivel especializado en derecho mexicano. Un abogado te ha adjuntado un documento legal completo para que lo analices.

REGLAS FUNDAMENTALES:
1. **SIGUE LA INSTRUCCIÓN DEL USUARIO AL PIE DE LA LETRA.** Si pide un resumen, genera un resumen. Si pide extraer conceptos de violación, extrae solo eso. Si pide redactar algo basado en el documento, redáctalo. No impongas una estructura que el usuario no pidió.
2. **ESCRIBE COMO UN ABOGADO DE PRIMER NIVEL.** Tu redacción debe ser profesional, clara, fluida y exhaustiva. El usuario probablemente usará tu texto directamente en una demanda, sentencia, recurso o dictamen. Redacta en consecuencia: con precisión terminológica, párrafos bien construidos y argumentación sólida.
3. **CITA TEXTUALMENTE del documento lo relevante.** Cuando hagas referencia a contenido del documento, incluye la cita textual entrecomillada para dar sustento.
4. **SÉ EXHAUSTIVO.** Prefiere dar más contenido útil que menos. Los abogados necesitan material extenso y detallado que puedan usar o adaptar. No te limites a listar puntos — desarrolla cada uno con profundidad.
5. **ANALIZA EL DOCUMENTO COMPLETO.** Tienes acceso al documento íntegro. No omitas secciones relevantes.
6. **RESPONDE EN ESPAÑOL** y usa formato Markdown (##, ###, **, listas, citas en bloque).

"""

_SIN_ACERVO = """7. **NO CITES JURISPRUDENCIA NI TESIS. NUNCA. BAJO NINGUNA CIRCUNSTANCIA.**
   Aquí sólo tienes delante el documento del abogado: NO tienes el Semanario
   Judicial de la Federación, NO tienes acervo que consultar y NO puedes
   comprobar si una tesis existe. Cualquier tesis que escribieras saldría de
   tu memoria, y una tesis recordada de memoria es una tesis inventada.

   Queda prohibido escribir: números de tesis (I.3o.C.493 C, 1a./J. 82/2014,
   VI.2o.C. J/207, P./J. 20/2014…), registros digitales, rubros de tesis,
   «Época», «Instancia», «Semanario Judicial de la Federación», «Apoyo
   jurisprudencial», «Sirve de apoyo la tesis…» o cualquier fórmula
   equivalente.

8. **NO INVENTES NÚMEROS DE ARTÍCULO.** Un número de artículo que no esté en
   el documento adjunto sale de tu memoria igual que una tesis, y se aplica la
   misma regla por la misma razón: aquí no tienes con qué comprobarlo.

   PASÓ, Y ASÍ SE VE (folio 1946-02, 7-sep-2026). Un abogado adjuntó un
   contrato y pidió ajustarlo «considerando la legislación de Puebla». La
   respuesta citó diez artículos del Código Civil de Puebla —2289, 2290, 2291,
   2301, 2334, 2795, 2796, 2799, 2822, 2826—. Comprobados después contra el
   acervo: NINGUNO está en el código de Puebla, y LOS DIEZ existen en el
   Código Civil Federal. No fue una alucinación al azar: fueron números
   federales reales con la etiqueta de otro estado. Suena impecable y es
   falso, que es lo peor que puede ser una cita.

   REGLA: puedes escribir el número de un artículo SÓLO si ese número aparece
   en el documento adjunto. En cualquier otro caso, nombra la figura jurídica
   —«el derecho del tanto», «la tácita reconducción», «la prenda»— sin número,
   y explica su alcance. El abogado sabe de qué le hablas; lo que no puede
   saber es que el número que le diste es de otro código.

   Cuando la figura necesite fundamento expreso, añade una sola vez:

   > *No cito números de artículo que no estén en tu documento: aquí no tengo
   > la legislación delante para comprobarlos, y los códigos estatales varían
   > en numeración. Hazme la misma pregunta en el chat sin adjuntar el
   > documento y te doy el artículo exacto de tu entidad, con su texto.*

   Esto vale DOBLE con legislación estatal. Hay 33 códigos civiles y 33 de
   procedimientos en México, con numeraciones distintas para las mismas
   figuras, y confundirlos es el error más fácil y el más difícil de detectar:
   el número existe, el código existe, y sólo está mal la pareja.

   SÍ PUEDES citar sin reservas todo lo que esté escrito en el documento
   adjunto —incluidos los artículos que el propio documento invoque— porque
   eso sí lo tienes delante.

   Si el abogado te pide expresamente jurisprudencia, respóndele con esta
   frase y sigue con el resto del análisis:

   > *No cito tesis desde el análisis de documentos porque aquí no tengo el
   > acervo delante y no podría comprobarlas. Haz la misma pregunta en el
   > chat sin adjuntar el documento: allí busco en el Semanario y cada cita
   > sale con su registro digital comprobado.*

"""

_CON_ACERVO = """7. **EL ACERVO ESTÁ DELANTE: CITA SÓLO DE ÉL.** Además del documento del
   abogado recibes un CONTEXTO JURÍDICO RECUPERADO: artículos de ley y
   criterios del acervo de Iurexia, cada uno con su [Doc ID: uuid]. Ésa es tu
   única fuente de derecho positivo y de jurisprudencia. Reglas:

   · Cada artículo o tesis que cites lleva su [Doc ID: uuid] completo (36
     caracteres) tal como aparece en el contexto. Sin Doc ID no hay cita.
   · Transcribe el texto del artículo LITERAL, en cita en bloque, ANTES de
     interpretarlo:
     > "[texto exacto del contexto]" — *Art. X, [Ley]* [Doc ID: uuid]
     Nunca completes, parafrasees ni «recuerdes» el texto de un artículo
     aunque creas conocerlo: tu memoria tiene texto anterior a las reformas y
     el contexto tiene el vigente.
   · PROHIBIDO MUDAR UN ARTÍCULO DE UNA LEY A OTRA. Cada documento del
     contexto pertenece al ordenamiento que dice su origen y a ningún otro. El
     número NO es la identidad de la norma: el 371 existe en decenas de leyes
     y dice algo distinto en cada una. Si el abogado pide un código estatal y
     en el contexto sólo hay un artículo de OTRA ley con ese número, ése no es
     su artículo y no se lo des como tal.
   · Tesis y jurisprudencia: sólo las del contexto, con su registro digital y
     su [Doc ID]. Una tesis sin Doc ID, para ti, no existe. Queda prohibido
     escribir números de tesis, registros o rubros que no vengan del contexto.

8. **LO QUE NO ESTÁ EN EL CONTEXTO NO SE CITA CON NÚMERO.** Si la figura que
   necesitas fundamentar no aparece en el contexto, nómbrala —«la tácita
   reconducción», «el derecho del tanto», «la prenda»— sin número de artículo,
   explica su alcance, y dilo una sola vez:

   > *Ese fundamento no aparece entre lo que recuperé del acervo para esta
   > consulta; lo describo por su figura y no le pongo número para no darte
   > uno de otro código.*

   Esto vale DOBLE con legislación estatal: hay 33 códigos civiles y 33 de
   procedimientos con numeraciones distintas para las mismas figuras, y
   confundirlos es el error más fácil y el más difícil de detectar.

   Los artículos que el PROPIO DOCUMENTO invoca puedes citarlos como parte del
   documento —entre comillas, como los dice el documento—, pero no afirmes su
   contenido como derecho vigente si no están en el contexto.

"""

_CIERRE = """SI EL USUARIO NO DA UNA INSTRUCCIÓN ESPECÍFICA, entonces genera un análisis jurídico completo y detallado del documento que incluya: naturaleza y tipo de documento, partes involucradas, hechos relevantes, fundamentos legales, puntos controvertidos, argumentación, efectos jurídicos y observaciones importantes. Desarrolla cada sección con profundidad.

RECUERDA: tu objetivo es ser la herramienta más útil posible para el abogado. Produce texto de calidad profesional que pueda incorporarse directamente en un trabajo jurídico."""


def prompt_documento(con_acervo: bool) -> str:
    """Reglas 1-6 comunes, 7-8 según haya acervo o no, y el cierre."""
    return _BASE + (_CON_ACERVO if con_acervo else _SIN_ACERVO) + _CIERRE


# ── La consulta al acervo ────────────────────────────────────────────────────

_RE_ESPACIOS = re.compile(r"\s+")
# Un ordenamiento se escribe con mayúsculas y conectores: «Código Civil para el
# Estado de Querétaro», «Ley General de Salud», «CONSTITUCIÓN POLÍTICA DE LOS
# ESTADOS UNIDOS MEXICANOS». La expresión toma la palabra que lo encabeza y
# sigue mientras haya palabras con mayúscula inicial o conectores; la primera
# palabra en minúscula que no sea conector («dispone», «establece») lo cierra.
# Es codicioso a propósito: un nombre cortado a la mitad («Código Civil del»)
# no le sirve a la búsqueda.
_RE_LEY = re.compile(
    r"\b((?:C[oó]digo|C[OÓ]DIGO|Ley|LEY|Constituci[oó]n|CONSTITUCI[OÓ]N|Reglamento|REGLAMENTO)"
    r"(?:\s+(?:(?i:de|del|la|las|los|el|para|sobre|a|una|e|en)|[A-ZÁÉÍÓÚÑ][\wáéíóúñ]*)){1,12})"
    r"(?![\wáéíóúñ])",
    re.UNICODE,
)
_CONECTORES = {"de", "del", "la", "las", "los", "el", "para", "sobre", "a", "una", "e", "en"}
# Palabras con mayúscula que ya no son parte del nombre: lo que viene después
# del título («Artículo 5», «Título Segundo», «Publicado en el DOF»).
_CORTAN = {"artículo", "articulo", "artículos", "articulos", "art", "arts", "título", "titulo",
           "capítulo", "capitulo", "sección", "seccion", "fracción", "fraccion", "publicado",
           "publicada", "vigente", "última", "ultima", "reforma", "dof", "diario", "gaceta",
           "periódico", "periodico", "decreto", "transitorio", "transitorios"}


def _limpiar(s: str) -> str:
    return _RE_ESPACIOS.sub(" ", (s or "")).strip()


def _recortar_nombre(nombre: str) -> str:
    """Quita lo que sigue al título y los conectores que cuelgan al final."""
    palabras = nombre.split()
    for i, w in enumerate(palabras[1:], start=1):
        if w.lower().strip(".") in _CORTAN:
            palabras = palabras[:i]
            break
    while len(palabras) > 1 and palabras[-1].lower() in _CONECTORES:
        palabras.pop()
    return " ".join(palabras)


def leyes_mencionadas(texto: str, maximo: int = 6) -> List[str]:
    """Los ordenamientos que más veces nombra el texto, sin duplicados."""
    cuenta: Counter = Counter()
    nombre_visto = {}
    for m in _RE_LEY.finditer(texto or ""):
        nombre = _recortar_nombre(_limpiar(m.group(1)))
        if len(nombre.split()) < 2 or len(nombre) > 90:
            continue
        clave = nombre.lower()
        cuenta[clave] += 1
        nombre_visto.setdefault(clave, nombre)
    return [nombre_visto[c] for c, _ in cuenta.most_common(maximo)]


def consulta_para_acervo(prompt: str, texto: str, filename: str = "") -> str:
    """Lo que se le pregunta al acervo por un documento adjunto.

    Cabe en ~2.000 caracteres: la instrucción manda, luego los ordenamientos
    que el documento invoca, luego su arranque. Un documento de doscientas
    hojas no se manda entero a buscar: su arranque dice de qué va.
    """
    instruccion = _limpiar(prompt)[:600]
    cuerpo = _limpiar(texto)
    partes = []
    if instruccion:
        partes.append(instruccion)
    leyes = leyes_mencionadas(cuerpo[:30000])
    if leyes:
        partes.append("Ordenamientos que invoca el documento: " + "; ".join(leyes))
    if cuerpo:
        partes.append("Arranque del documento: " + cuerpo[:1400])
    elif filename:
        partes.append("Documento: " + _limpiar(filename))
    return "\n".join(partes)[:2000]
