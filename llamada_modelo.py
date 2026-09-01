"""LLAMAR AL MODELO SIN QUE UN PARÁMETRO TUMBE LA FASE ENTERA.

Los modelos de razonamiento no aceptan `temperature`. Cuando el taller cambió
de motor, las dos fases que la fijaban —la lectura del expediente y la
propuesta de sentido— empezaron a recibir:

    Unsupported value: 'temperature' does not support 0 with this model.
    Only the default (1) value is supported.

Y ahí se vio lo caro que sale capturar una excepción sin mirarla. En
`/taller/proponer` el error salía a la cara como un 500, que al menos se ve.
Pero en la fase 3 estaba dentro de un `try` que apunta el aviso y sigue:

    except Exception as e:
        f.avisos.append(f"No se pudieron derivar los problemas jurídicos: {e}")

De modo que el adelanto devolvía 200, el documento salía bien de forma, y los
PROBLEMAS JURÍDICOS iban vacíos. Sin ellos no hay consulta al acervo que
apuntar, ni propuesta que calificar, ni estudio que ordenar: el corazón del
taller llevaba semanas apagado y el único síntoma era un aviso entre catorce.

LA REGLA: se quita EL PARÁMETRO que el servidor rechaza, no la llamada. Es la
diferencia entre perder el determinismo —que es un lujo— y perder la fase, que
es el producto.
"""

from __future__ import annotations

import re

# «Unsupported value: 'temperature' does not support 0», «Unsupported
# parameter: 'seed'», «'top_p' is not supported with this model».
_RX_PARAM = re.compile(
    r"[Uu]nsupported (?:value|parameter)[^']*'([a-z_]+)'|"
    r"'([a-z_]+)'\s+is not supported", re.I)

# Sólo se quitan los de MUESTREO. Si el servidor rechaza `messages` o `model`,
# eso no es un lujo prescindible: es la llamada, y tiene que reventar.
_PRESCINDIBLES = {"temperature", "seed", "top_p", "frequency_penalty",
                  "presence_penalty", "logprobs"}


def parametro_rechazado(exc: Exception) -> str:
    m = _RX_PARAM.search(str(exc) or "")
    if not m:
        return ""
    p = (m.group(1) or m.group(2) or "").strip().lower()
    return p if p in _PRESCINDIBLES else ""


async def crear(cliente, **kw):
    """`chat.completions.create`, quitando lo que el modelo no admita."""
    quitados = []
    for _ in range(4):
        try:
            r = await cliente.chat.completions.create(**kw)
            if quitados:
                print(f"   ℹ️ el modelo no admite {', '.join(quitados)}: se "
                      f"llamó sin ellos (se pierde determinismo, no la fase)")
            return r
        except Exception as exc:
            p = parametro_rechazado(exc)
            if not p or p not in kw:
                raise
            kw.pop(p, None)
            quitados.append(p)
    raise RuntimeError("El modelo rechazó demasiados parámetros seguidos.")
