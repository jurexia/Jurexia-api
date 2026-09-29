"""La síntesis del proyecto, redactada como tesis del Semanario.

David: «fíjate cómo al final viene una síntesis del asunto elaborada a manera
de tesis del semanario judicial de la federación, es decir, con un título, un
apartado de hechos, un apartado de criterio o propuesta de resolución y un
apartado de justificación. En lugar de poner los dos nombres hasta abajo del
proyecto del secretario y del magistrado, eso no sirve, hay que generar esta
portada del proyecto que trae la síntesis».

QUÉ SE MIDIÓ ANTES DE ESCRIBIR ESTO. 1,360 documentos de la carpeta del taller.
La forma que él describe —título en versales, «Hechos:», «Criterio jurídico:»,
«Justificación:»— aparece en 81, y en 23 de ellos AL FINAL del documento (por
encima del 85% de su extensión), siempre con el título en versales delante.
Extensiones medianas de esos 23:

    título .......... 236 caracteres
    Hechos .......... 815
    Criterio ........ 571
    Justificación ... 1,148

Hay una segunda forma en su corpus —«CONTEXTO / PROPUESTA / JUSTIFICACIÓN», 4
documentos— que NO es la que pidió: él nombró los cuatro campos del Semanario,
título incluido. Se implementa la que describió.

EL REGISTRO ES DE PROYECTO, NO DE TESIS PUBLICADA. En el ADC 821/2025 el
criterio empieza «Se propone determinar que…», no «Se determina que…». Un
proyecto propone; lo que se publica ya resolvió. Esa diferencia de una palabra
es la que distingue un documento que va a sesión de uno que salió de ella.
"""
from __future__ import annotations
import json
import os
import re

MODELO_SINTESIS = os.getenv("MODELO_SINTESIS", "gpt-5.6-luna")

_RX_JSON = re.compile(r"\{.*\}", re.S)

_PROMPT = """Eres secretario de un tribunal colegiado de circuito. Acabas de
terminar un proyecto de sentencia y tienes que redactar la SÍNTESIS que va en
la portada, con la forma de una tesis del Semanario Judicial de la Federación.

EL ASUNTO
Tipo: {tipo}
Expediente: {expediente}
{rotulo_promovente}: {quejoso}{papeles}
Sentido que se propone: {sentido}

EL PROYECTO, tal como quedó redactado:
──────────────────────────────────────────
{estudio}
──────────────────────────────────────────

DEVUELVE UN JSON con exactamente estas cuatro claves:

"titulo": el rubro, EN MAYÚSCULAS, sin punto final. Enuncia la figura jurídica
  y lo que se resuelve sobre ella. Empieza por el concepto —«OSCURIDAD DE LA
  DEMANDA. NO SE CONFIGURA EN EL JUICIO DE…»—, con el punto que separa el tema
  del criterio. Alrededor de 240 caracteres. NO menciones el número de
  expediente ni los nombres de las partes.

"hechos": qué pasó, en pasado y en tercera persona, sin nombres propios de
  particulares: a cada quien por su figura EN ESTE ASUNTO{figuras}. Es el
  antecedente que hace comprensible el criterio, no el resumen del expediente.
  Alrededor de 800 caracteres, un solo párrafo.

"criterio": la propuesta de resolución. EMPIEZA CON «Se propone determinar
  que» —es un proyecto, todavía no se resuelve—, y enuncia la regla en
  abstracto, aplicable a otros casos iguales. Alrededor de 570 caracteres, un
  solo párrafo.

"justificacion": por qué. Los preceptos y las razones que sostienen el
  criterio, encadenadas. Alrededor de 1,150 caracteres, un solo párrafo.

REGLAS QUE NO SE ROMPEN
- Todo sale del proyecto que tienes arriba. No añadas razones que no estén.
- Ninguna cita de tesis por su rubro completo ni por su número de registro: la
  síntesis se lee de corrido.
- Nada de «el presente asunto», «en el caso que nos ocupa», «se advierte que».
- Si el proyecto no resuelve un criterio generalizable —porque el recurso se
  desechó, quedó sin materia o se declaró extemporáneo— devuelve "titulo": ""
  y deja los demás campos vacíos. Vale más ninguna síntesis que una inventada.

Sólo el JSON, sin texto alrededor."""


def prompt(*, tipo_asunto: str = "", expediente: str = "", quejoso: str = "",
           sentido: str = "", estudio: str = "", recurrente: str = "",
           papel_recurrente: str = "", organo: str = "", ficha: str = "") -> str:
    """El prompt de la síntesis, con los papeles de ESTE asunto como datos.

    `ficha` (SPEC_E2, 28-sep-2026): el bloque de `ficha_procesal.bloque`. Si
    llega, sustituye a los renglones sueltos de recurrente y órgano: la
    síntesis del 631 contó que «la parte quejosa adquirió el inmueble» porque
    no sabía quién era quién, y la ficha lo dice con su fuente —quejosa,
    recurrente y su carácter, órgano recurrido, qué resolvió y qué es materia
    de la revisión—."""
    try:
        import tipos_asunto as _ta_s
        _t = _ta_s.normalizar(tipo_asunto or "")
    except Exception:
        _t = (tipo_asunto or "").strip().lower()
    papeles = ""
    if _t in ("amparo_revision", "queja") and (recurrente or "").strip():
        _car = _CARACTER.get((papel_recurrente or "").strip().lower(), "")
        papeles += (f"\nParte recurrente: {recurrente.strip()}"
                    + (f" ({_car})" if _car else ""))
    if _t == "amparo_revision" and (organo or "").strip():
        papeles += f"\nÓrgano que dictó la sentencia recurrida: {organo.strip()}"
    if (ficha or "").strip():
        papeles = "\n" + ficha.rstrip()
    return _PROMPT.format(
        tipo=tipo_asunto or "no consta",
        expediente=expediente or "no consta",
        quejoso=quejoso or "no consta",
        rotulo_promovente=("Parte quejosa (promovió el amparo)" if _t == "amparo_revision"
                           else "Parte promovente"),
        papeles=papeles,
        figuras=_FIGURAS.get(_t, _FIGURAS["amparo_directo"]),
        sentido=sentido or "no consta",
        estudio=(estudio or "")[:60000])


def _limpio(x) -> str:
    """Una cadena de una línea, sin el rótulo repetido ni comillas sueltas."""
    t = re.sub(r"\s+", " ", str(x or "")).strip()
    # El modelo a veces repite la etiqueta dentro del valor.
    t = re.sub(r"^(?:hechos|criterio\s+jur[íi]dico|criterio|justificaci[óo]n)\s*:\s*",
               "", t, flags=re.I)
    return t.strip(' "“”')


# LAS FIGURAS DE ESTE ASUNTO (28-sep-2026, AR 631/2025). El prompt daba como
# ejemplo las del amparo directo —«la parte quejosa», «la autoridad
# responsable»— y la «parte promovente» era lo tecleado: en una revisión que
# interpuso la tercera interesada, la síntesis contó que «la parte quejosa
# adquirió el inmueble» (fue la recurrente) y que «la autoridad responsable
# concedió el amparo» (fue el Juzgado de Distrito). Ahora se le dice quién es
# quién, como dato, con la figura que le toca a cada uno.
_FIGURAS = {
    "amparo_directo": " —la parte quejosa, la autoridad responsable, la parte "
                      "tercera interesada—",
    "amparo_revision": " —la parte quejosa (quien promovió el amparo), la parte "
                       "recurrente (quien interpuso la revisión, si no es la "
                       "quejosa), el Juzgado de Distrito (quien dictó la "
                       "sentencia recurrida) y la autoridad responsable (quien "
                       "dictó el acto reclamado)—",
    "queja": " —la parte recurrente y el órgano que dictó el auto recurrido—",
    "revision_fiscal": " —la autoridad recurrente, la parte actora y la Sala "
                       "que dictó la sentencia recurrida—",
}
_CARACTER = {"tercero": "parte tercera interesada", "autoridad": "autoridad responsable",
             "quejoso": "la propia parte quejosa"}


async def sintetizar(cliente, *, tipo_asunto: str = "", expediente: str = "",
                     quejoso: str = "", sentido: str = "",
                     estudio: str = "", recurrente: str = "",
                     papel_recurrente: str = "", organo: str = "", ficha: str = "") -> dict:
    """{titulo, hechos, criterio, justificacion}; vacío si no procede.

    NUNCA REVIENTA EL PROYECTO. La síntesis es la última página; si el modelo
    falla, se entrega el documento sin ella. Un proyecto sin portada sirve; un
    proyecto que no se generó, no.
    """
    if cliente is None or not (estudio or "").strip():
        return {}
    try:
        import llamada_modelo as _lm
        r = await _lm.crear(
            cliente, model=MODELO_SINTESIS, max_completion_tokens=6000,
            reasoning_effort="medium",
            messages=[{"role": "user", "content": prompt(
                tipo_asunto=tipo_asunto, expediente=expediente, quejoso=quejoso,
                sentido=sentido, estudio=estudio, recurrente=recurrente,
                papel_recurrente=papel_recurrente, organo=organo, ficha=ficha)}])
        m = _RX_JSON.search((r.choices[0].message.content or "").strip())
        d = json.loads(m.group(0)) if m else {}
    except Exception:
        return {}
    if not isinstance(d, dict):
        return {}
    titulo = _limpio(d.get("titulo")).rstrip(".").upper()
    # SIN TÍTULO NO HAY SÍNTESIS. Es la salida que el propio prompt ofrece para
    # los asuntos sin criterio generalizable, y también lo que queda cuando el
    # modelo devolvió algo que no sirve.
    if len(titulo) < 30:
        return {}
    out = {"titulo": titulo,
           "hechos": _limpio(d.get("hechos")),
           "criterio": _limpio(d.get("criterio")),
           "justificacion": _limpio(d.get("justificacion"))}
    if not (out["hechos"] and out["criterio"] and out["justificacion"]):
        return {}
    # EL REGISTRO DE PROYECTO. Si el modelo escribió «Se determina que» —el
    # tiempo de la tesis ya publicada— se corrige aquí: es una palabra, y es
    # la que dice si el documento va a sesión o salió de ella.
    out["criterio"] = re.sub(r"^se\s+determina\b", "Se propone determinar",
                             out["criterio"], flags=re.I)
    return out
