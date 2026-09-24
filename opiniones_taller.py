"""LA OPINIÓN DEL SECRETARIO SOBRE CADA PROYECTO.

POR QUÉ EXISTE. David (24-sep-2026): «debemos implementar un auditor para
verificar la experiencia de todos los usuarios para aprovechar la calidad de
sus sentencias. Al término de cada proyecto abrir un cuadro de texto con
formato visual profesional para que el usuario escriba sus puntos de vista y
aspectos a mejorar en el taller y, particularmente, en la calidad de las
sentencias que entrega».

QUÉ SE PREGUNTA, Y POR QUÉ ASÍ. Un cuadro de texto libre, solo, se contesta
poco y no se puede sumar: diez opiniones en prosa no dicen si el taller
mejora. Así que la prosa va, pero acompañada de tres cosas que SÍ se suman:

  · la calificación de la sentencia, de 1 a 5;
  · CUÁNTO TUVO QUE CORREGIR para poder firmarla —nada, poco, mucho, la
    rehízo—, que es la cifra que de verdad mide a un redactor: la meta del
    método del secretario son 2-3 horas por 10 páginas, y un proyecto que se
    rehace no ahorró ninguna;
  · los aspectos de la sentencia, uno por uno —sentido, fundamentación,
    citas, redacción, estructura, efectos, cómputo del plazo—, cada uno
    «bien» o «a mejorar».

Y UNA FOTO DEL PROYECTO AL OPINAR: su tipo, su sentido y los avisos que el
propio pipeline le puso. Es lo que permite al auditor cruzar lo que dijo la
persona con lo que dijo la máquina —si el secretario marca «citas: a mejorar»
en proyectos donde el barrido no avisó de nada, la verificación tiene un
hueco—.

Este módulo no toca el modelo ni la base: valida y arma la fila. Guardarla es
del endpoint, que es quien tiene la sesión.
"""
from __future__ import annotations

# Los aspectos de la sentencia que se califican uno por uno. El orden es el de
# lectura de un proyecto. La clave es estable —el auditor suma por ella—; la
# etiqueta es lo que ve el secretario.
ASPECTOS = (
    ("sentido", "El sentido de la resolución"),
    ("fundamentacion", "Fundamentación y argumentación"),
    ("citas", "Citas de ley y jurisprudencia"),
    ("redaccion", "Redacción y estilo"),
    ("estructura", "Estructura y forma del proyecto"),
    ("efectos", "Efectos y puntos resolutivos"),
    ("computo", "Cómputo del plazo y procedencia"),
)
_CLAVES = {k for k, _ in ASPECTOS}
VALORES_ASPECTO = ("bien", "mejorar")
CORRECCION = ("nada", "poco", "mucho", "rehecho")
TOPE_TEXTO = 6000


def _texto(x) -> str:
    """Se conservan los saltos de párrafo: el secretario escribe en párrafos."""
    return str(x or "").replace("\r", "").strip()[:TOPE_TEXTO]


def limpiar(calificacion=None, correccion="", aspectos=None,
            sobre_sentencia="", sobre_taller="") -> tuple:
    """(fila_parcial, errores). La fila trae sólo lo que el secretario dijo."""
    errores = []
    cal = None
    if calificacion not in (None, "", 0, "0"):
        try:
            cal = int(calificacion)
        except (TypeError, ValueError):
            cal = None
        if cal is None or not 1 <= cal <= 5:
            errores.append("La calificación va de 1 a 5.")
            cal = None
    cor = str(correccion or "").strip().lower()
    if cor and cor not in CORRECCION:
        errores.append("La corrección es «nada», «poco», «mucho» o «rehecho».")
        cor = ""
    asp = {}
    for k, v in (aspectos or {}).items():
        k = str(k or "").strip().lower()
        v = str(v or "").strip().lower()
        if k in _CLAVES and v in VALORES_ASPECTO:
            asp[k] = v
    s_sent, s_tall = _texto(sobre_sentencia), _texto(sobre_taller)
    if cal is None and not cor and not asp and not s_sent and not s_tall:
        errores.append("La opinión llegó vacía: no hay nada que guardar.")
    return ({"calificacion": cal, "correccion": cor or None, "aspectos": asp,
             "sobre_sentencia": s_sent or None, "sobre_taller": s_tall or None},
            errores)


def foto_del_proyecto(estado: dict, version: int = 0) -> dict:
    """{version, tipo_asunto, materia, sentido, avisos_n, avisos} de la versión
    pedida —o de la última si `version` es 0—, sacada del `estado` guardado."""
    est = estado if isinstance(estado, dict) else {}
    enc = est.get("encargo") if isinstance(est.get("encargo"), dict) else {}
    pila = [p for p in (est.get("proyectos") or []) if isinstance(p, dict)]
    ficha = None
    if version:
        ficha = next((p for p in pila if int(p.get("version") or 0) == int(version)), None)
    if ficha is None:
        ficha = est.get("proyecto") if isinstance(est.get("proyecto"), dict) else (pila[0] if pila else {})
    ficha = ficha or {}
    # EL SENTIDO: el global si lo hubo; si no, el del principal; si no, el primero.
    sentido = str(ficha.get("sentido_global") or "").strip()
    if not sentido:
        crits = [c for c in (ficha.get("criterios") or []) if isinstance(c, dict)]
        pral = next((c for c in crits if str(c.get("jerarquia") or "") == "principal"),
                    crits[0] if crits else {})
        sentido = str(pral.get("sentido") or "").strip()
    avisos = [str(a) for a in (ficha.get("avisos") or [])][:40]
    return {"version": int(ficha.get("version") or version or 0),
            "tipo_asunto": str(enc.get("tipo_asunto") or "")[:40] or None,
            "materia": str(enc.get("materia") or "")[:40] or None,
            "sentido": sentido[:40] or None,
            "avisos_n": len(avisos), "avisos": avisos}
