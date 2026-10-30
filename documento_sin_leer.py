"""EL CHAT NO RESPONDE COMO SI TUVIERA UN DOCUMENTO QUE NO SE LEYÓ.   (29-oct-2026)

Cuando el análisis de un adjunto falla —se perdió el contacto, se cortó la
conexión, no se extrajo el texto—, la conversación guarda igual el mensaje
«📄 **Documento adjunto:** nombre», pero sin el texto leído (el bloque
DOCUMENTO_INICIO/FIN sólo entra si el servidor alcanzó a devolverlo). En 30
días, 48 de esas fallas en la web siguieron con una pregunta al chat SIN volver
a adjuntar —«analizando dicho documento, la cláusula cuarta…», «dame el
proyecto como te lo pedí valorando todo», «te envié el auto recurrido»—, y el
modelo, que ve el nombre del archivo y la petición, contesta como si lo
hubiera leído: el 26-oct un proyecto Platinum «estudió» agravios que nunca se
leyeron (folio 3640-01), y el 3-oct se contestó sobre «la cláusula cuarta» de
un contrato cuyo texto no viajaba en la conversación.

Aquí se detectan esos adjuntos sin texto y se le dice al modelo, en una
instrucción aparte, que no tiene su contenido. La app móvil nunca guarda el
texto leído, así que la instrucción también es cierta ahí: lo que Iurexia
respondió sobre el documento sí está en la conversación; su texto, no.
"""
from __future__ import annotations

import re
from typing import Iterable, List

_ADJUNTO = re.compile(r"📄\s*\*\*Documento adjunto:\*\*\s*([^\n]+)")
_LEIDO = "DOCUMENTO_INICIO"


def adjuntos_sin_texto(mensajes: Iterable) -> List[str]:
    """Los nombres de los adjuntos de la conversación cuyo texto no viaja en
    ella, en orden y sin repetir. Un nombre que más tarde sí se leyó (se volvió
    a adjuntar y esta vez entró) no cuenta."""
    sin, con = [], set()
    for m in mensajes:
        if (getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)) != "user":
            continue
        texto = getattr(m, "content", None)
        if texto is None and isinstance(m, dict):
            texto = m.get("content")
        texto = str(texto or "")
        nombres = [n.strip()[:160] for n in _ADJUNTO.findall(texto)]
        if not nombres:
            continue
        if _LEIDO in texto:
            con.update(nombres)
        else:
            sin.extend(n for n in nombres if n not in sin)
    return [n for n in sin if n not in con]


def nota_documentos_sin_leer(mensajes: Iterable) -> str:
    """La instrucción para el modelo, o "" si todos los adjuntos viajan con su
    texto."""
    nombres = adjuntos_sin_texto(mensajes)
    if not nombres:
        return ""
    lista = ", ".join(f"«{n}»" for n in nombres[:8]) + (" y otros" if len(nombres) > 8 else "")
    return (
        "DOCUMENTOS CUYO TEXTO NO TIENES\n"
        f"En esta conversación el abogado adjuntó {lista}, pero el texto de "
        f"{'ese documento' if len(nombres) == 1 else 'esos documentos'} NO está en la conversación: "
        "no se pudo leer o no llegó. No describas, no cites y no analices su contenido, ni lo deduzcas "
        "de su nombre o de lo hablado. Sí puedes apoyarte en lo que Iurexia ya respondió sobre él más "
        "arriba. Si lo que pide el abogado depende de ese texto, dilo en la primera línea de tu respuesta "
        "y pídele que lo vuelva a adjuntar (o que pegue el fragmento que necesita)."
    )
