"""LO QUE PASA DEL TOPE DE HOJAS SE LEE EN PARTE, NO SE RECHAZA.   (29-oct-2026, David)

Un documento con más hojas de las que lee el plan se rechazaba con un 413 y
nada más: 53 rechazos a 32 usuarios de pago en 30 días, y hoy un abogado que
acababa de pasar a Pro recibió tres seguidos (201, 201 y 101 hojas contra su
tope de 100). «Divídalo en partes» pide una herramienta que muchos no tienen
a mano, y el abogado se queda sin nada.

Desde el 25-oct el gratuito ya leía una muestra de 20 hojas. Ahora todos los
planes: se leen las primeras N hojas que permite el plan y se dice claramente.

  · LA PRIMERA LÍNEA LA ESCRIBE EL SERVIDOR, no el modelo: así sale siempre,
    dice siempre lo mismo y queda en la conversación tal cual. Al modelo se le
    dice qué hojas recibe y que no lo repita.
  · La pantalla recibe `limite_plan` y ofrece las hojas siguientes con un
    botón: el mismo archivo con `desde_hoja`, sin que el abogado tenga que
    partir el PDF.

Lo que vale para todos, también el gratuito: sólo PDF (de un DOCX no se sabe
el número de hojas; ahí manda el tope de caracteres, como siempre) y los
administradores leen entero.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

# ── CUÁNTAS HOJAS LEE CADA PLAN ──────────────────────────────────────────
#
# Medido sobre 365 documentos reales subidos entre el 8 y el 16-ago-2026: la
# mediana tiene 15 páginas, el percentil 90 llega a 64 y el mayor visto fue
# de 522. El 47 % entra escaneado y hay que pasarlo por OCR, que es la parte
# cara: por eso el tope se mide en HOJAS y no en caracteres — es lo que de
# verdad cuesta, y además es lo que un abogado sabe contar.
#
#   20 hojas  → el 62 % de los documentos (una promoción corta)
#   50 hojas  → el 85 %
#  100 hojas  → el 93 %
#  600 hojas  → todo lo medido, con margen sobre el máximo de 522
PAGINAS_POR_PLAN: Dict[str, int] = {
    "gratuito": 20,
    "basico_monthly": 50, "basico_annual": 50,
    "pro_monthly": 100, "pro_annual": 100,
    "platinum_monthly": 600, "platinum_annual": 600,
    "ultra_secretarios": 600,
}
# Un plan que no está en la tabla (uno nuevo, o un nombre raro) lee como
# antes del 16-ago: 30 hojas.
TOPE_DESCONOCIDO = 30
PLAN_QUE_CUBRE = [(50, "Básico"), (100, "Pro"), (600, "Platinum")]

_NOMBRES = (("platinum", "Platinum"), ("ultra", "Ultra Secretarios"), ("pro", "Pro"),
            ("basico", "Básico"))


def tope_del_plan(subscription_type: Optional[str]) -> int:
    return PAGINAS_POR_PLAN.get(subscription_type or "gratuito", TOPE_DESCONOCIDO)


def nombre_del_plan(subscription_type: Optional[str]) -> str:
    """Como lo lee el abogado: «Pro», «Básico»… y «gratuito» en minúscula,
    que es como se nombra en la pantalla."""
    s = (subscription_type or "").strip().lower()
    for prefijo, nombre in _NOMBRES:
        if s.startswith(prefijo):
            return nombre
    return "gratuito"


def plan_necesario(paginas: int) -> Optional[str]:
    """El plan más barato que sí leería un documento de este tamaño."""
    for tope, nombre in PLAN_QUE_CUBRE:
        if paginas <= tope:
            return nombre
    return None


def plan_que_lee_mas(paginas: int, tope_actual: int, gratuito: bool = False) -> Optional[str]:
    """El plan que se le sugiere: el más barato que lo lee entero y, si
    ninguno lo lee entero, el que más lee; nunca uno que lea lo mismo o menos
    que el actual. Al gratuito, lo mismo que su tarjeta desde el 25-oct
    (David): Pro o Platinum si lo leen entero, y nada si ninguno lo lee."""
    if gratuito:
        return "Pro" if paginas <= 100 else ("Platinum" if paginas <= 600 else None)
    for tope, nombre in PLAN_QUE_CUBRE:
        if paginas <= tope and tope > tope_actual:
            return nombre
    tope, nombre = PLAN_QUE_CUBRE[-1]
    return nombre if paginas > tope and tope > tope_actual else None


def desde_valido(desde: Optional[str]) -> Optional[int]:
    """La hoja desde la que se pidió leer (`desde_hoja`), o None si no se pidió
    o no es un número: un valor raro se ignora y se lee desde el principio."""
    try:
        n = int(str(desde).strip())
    except (TypeError, ValueError):
        return None
    return n if n >= 1 else None


def tramo(total: int, tope: int, desde: Optional[int] = None) -> Optional[Tuple[int, int]]:
    """(primera, última) — 1-based e incluidas — de las hojas que se leen, o
    None si el documento se lee entero tal cual llegó.

    Sin `desde`: las primeras `tope` si el documento pasa del tope. Con
    `desde`: desde esa hoja, hasta `tope` hojas. `desde` fuera del documento
    es un error de quien llama (ver ValueError)."""
    if total <= 0 or tope <= 0:
        return None
    if desde and desde > 1:
        if desde > total:
            raise ValueError(f"El documento tiene {total} hojas: no hay hoja {desde}.")
        return desde, min(total, desde + tope - 1)
    if total > tope:
        return 1, tope
    return None


def _hojas(primera: int, ultima: int) -> str:
    return f"la hoja {primera}" if primera == ultima else f"las hojas {primera} a {ultima}"


def primera_linea(subscription_type: Optional[str], tope: int, total: int,
                  primera: int, ultima: int) -> str:
    """La línea con la que empieza la respuesta. En cursiva y con su párrafo:
    se lee como nota, no como parte del análisis."""
    plan = nombre_del_plan(subscription_type)
    if primera == 1:
        leido = f"Leí las primeras {ultima} de las {total} hojas de su documento"
    else:
        leido = f"Leí {_hojas(primera, ultima)} de las {total} de su documento"
    if plan == "gratuito":
        # La frase que el modelo escribía desde el 25-oct, ahora fija.
        linea = f"{leido} (el plan gratuito lee hasta {tope})."
    else:
        linea = f"{leido}: su plan {plan} lee hasta {tope} por documento."
        resto = []
        if ultima < total:
            resto.append(f"adjunte {_hojas(ultima + 1, total)} por separado")
            mayor = plan_que_lee_mas(total, tope)
            if mayor:
                resto.append(f"pase a {mayor} (hasta {_tope_nombre(mayor)})")
        if resto:
            linea += " Para el resto, " + " o ".join(resto) + "."
    return f"*{linea}*\n\n"


def _tope_nombre(nombre: str) -> int:
    return next((t for t, n in PLAN_QUE_CUBRE if n == nombre), 0)


def nota_para_el_modelo(tope: int, total: int, primera: int, ultima: int) -> str:
    """Va pegada al encabezado del documento en el mensaje del usuario."""
    if primera == 1:
        recibe = f"sólo recibes las primeras {ultima}"
    else:
        recibe = (f"sólo recibes {_hojas(primera, ultima)} (la primera página de este texto es "
                  f"la hoja {primera} del original: si citas una hoja, usa la numeración original)")
    return (
        f"\n\n[NOTA: El documento tiene {total} páginas y el plan del usuario lee hasta {tope} por "
        f"documento: {recibe}. Ya se le avisó al usuario en una línea aparte, antes de tu "
        f"respuesta: no lo repitas ni hables de planes o precios. Basa el análisis sólo en esas "
        f"páginas y no supongas el contenido de las demás; si lo que se pide depende de ellas, dilo.]"
    )


def aviso(subscription_type: Optional[str], tope: int, total: int,
          primera: int, ultima: int) -> dict:
    """El evento `limite_plan` para la pantalla. `leidas`, `total` y `plan`
    son los de la muestra del gratuito (25-oct); lo demás es para la tarjeta
    de los planes de pago y el botón de las hojas siguientes."""
    actual = nombre_del_plan(subscription_type)
    return {
        "leidas": ultima - primera + 1,
        "total": total,
        "plan": plan_que_lee_mas(total, tope, gratuito=(actual == "gratuito")),
        "plan_actual": actual,
        "tope": tope,
        "desde": primera,
        "hasta": ultima,
        "siguiente": ultima + 1 if ultima < total else None,
    }
