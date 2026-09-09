"""Qué es cada PDF que llega de SISE, leyéndolo.

David: «los pdf (el primero en cada asunto) regularmente tiene todo lo
necesario, pero también está el auto de admisión y el auto de turno que son
indispensables para verificar datos como la presentación, los terceros
interesados, el magistrado ponente».

POR QUÉ SE CLASIFICA LEYENDO Y NO POR LA POSICIÓN EN LA TABLA. La admisión y el
turno son actuaciones distintas y no siempre en el mismo orden; y la promoción
que abre el asunto trae unas veces la sentencia recurrida y otras no. Medido en
la revisión fiscal 91/2025: sus 78 páginas de recurso NO contienen la
recurrida —ni su número de expediente, ni «VISTOS los autos», ni «RESUELVE»—.
«El primero trae todo» es *regularmente*, no siempre, y proyectar sin la
recurrida por haberlo supuesto es peor que decir que falta.
"""
from __future__ import annotations
import re

# ── LO QUE SE PUDO CALIBRAR CONTRA DOCUMENTOS REALES ──────────────────────
# La sentencia de la Sala y el escrito de revisión del 91/2025, los dos
# pasados por Azure. Las marcas de abajo separan esos dos sin ambigüedad.
_SENTENCIA = (
    (re.compile(r"VISTOS\s+los\s+autos", re.I), 3),
    (re.compile(r"R\s*E\s*S\s*U\s*E\s*L\s*V\s*E\s*:?", re.I), 3),
    (re.compile(r"\bsentencia\s+definitiva\b", re.I), 2),
    (re.compile(r"\bC\s*O\s*N\s*S\s*I\s*D\s*E\s*R\s*A\s*N\s*D\s*O", re.I), 2),
    (re.compile(r"\bpor\s+lo\s+expuesto\s+y\s+fundado\b", re.I), 2),
)
_PROMOCION = (
    (re.compile(r"\bse\s+interpone\s+recurso\b", re.I), 3),
    (re.compile(r"\binterpongo\s+recurso\b", re.I), 3),
    # AMPARO DIRECTO. Las de arriba son de RECURSO, y con ellas una demanda de
    # amparo no puntuaba fuerte: la del ADC 536/2025 empataba 6-6 con
    # «sentencia» y salía «desconocido». Éstas son las fórmulas con que se pide
    # el amparo, y no aparecen en una sentencia que lo resuelve.
    (re.compile(r"\bsolicit[oa]\s+el\s+amparo\b", re.I), 3),
    (re.compile(r"\bdemando\s+el\s+amparo\b", re.I), 3),
    (re.compile(r"\bbajo\s+protesta\s+de\s+decir\s+verdad\b", re.I), 3),
    (re.compile(r"\bA\s*G\s*R\s*A\s*V\s*I\s*O\s*S\b", re.I), 2),
    (re.compile(r"\bprimer\s+agravio\b", re.I), 2),
    (re.compile(r"\bconceptos?\s+de\s+violaci[óo]n\b", re.I), 2),
    (re.compile(r"\bdemanda\s+de\s+amparo\b", re.I), 1),
    (re.compile(r"\bautoridad\s+responsable\b", re.I), 1),
)

# ── LO QUE TODAVÍA NO SE HA CALIBRADO ─────────────────────────────────────
# No hay ningún auto de admisión ni de turno en el corpus del tribunal ni en
# las descargas: estas fórmulas son las usuales, pero NO están comprobadas
# contra documentos reales. Hasta que lleguen dos o tres por la extensión, un
# acuerdo que no case con nada se devuelve como «acuerdo» a secas, que es
# honesto, en vez de adivinar cuál es.
_ADMISION = (
    (re.compile(r"\bse\s+admite\s+a\s+tr[áa]mite\b", re.I), 3),
    (re.compile(r"\bt[ée]ngase\s+por\s+admitid", re.I), 3),
    (re.compile(r"\bse\s+admite\s+el\s+recurso\b", re.I), 3),
    (re.compile(r"\bt[ée]ngase\s+por\s+interpuesto\b", re.I), 2),
    (re.compile(r"\btercer[oa]s?\s+interesad", re.I), 1),
)
_TURNO = (
    (re.compile(r"\bt[úu]rnese\b", re.I), 3),
    (re.compile(r"\bse\s+turna\s+(?:el\s+)?(?:asunto|expediente)\b", re.I), 3),
    (re.compile(r"\bpara\s+la\s+elaboraci[óo]n\s+del\s+proyecto\b", re.I), 2),
    (re.compile(r"\bmagistrad[oa]\s+ponente\b", re.I), 1),
)
_NOTIFICACION = (
    (re.compile(r"\bconstancia\s+de\s+notificaci[óo]n\b", re.I), 3),
    (re.compile(r"\bnotificaci[óo]n\s+(?:electr[óo]nica|personal|por\s+lista)\b", re.I), 2),
    (re.compile(r"\bse\s+notific[óa]\b", re.I), 1),
)

_TIPOS = (("sentencia_recurrida", _SENTENCIA), ("promocion", _PROMOCION),
          ("auto_admision", _ADMISION), ("auto_turno", _TURNO),
          ("notificacion", _NOTIFICACION))

# Un documento corto no es una sentencia por mucho que diga «considerando».
MINIMO_SENTENCIA = 8000


def puntuar(texto: str) -> dict:
    """Cuánto se parece el texto a cada tipo. Para poder auditar la decisión."""
    t = " ".join((texto or "").split())
    fuera = {}
    for nombre, marcas in _TIPOS:
        fuera[nombre] = sum(peso for rx, peso in marcas if rx.search(t))
    if len(t) < MINIMO_SENTENCIA:
        fuera["sentencia_recurrida"] = max(0, fuera["sentencia_recurrida"] - 3)
    # CITAR UNA SENTENCIA NO ES SERLO.
    #
    # Una demanda de amparo directo TRANSCRIBE la sentencia que combate: es lo
    # normal, y por eso puntuaba a la vez como escrito y como sentencia. La del
    # ADC 536/2025 —85 páginas sobre una pericial declarada desierta— empataba
    # 6 contra 6 y la regla del empate la devolvía como «desconocido»: el
    # taller se quedaba sin saber cuál era el escrito y avisaba, además, de que
    # faltaba la recurrida.
    #
    # Lo que separa una sentencia de verdad de su transcripción son las marcas
    # FUERTES —«VISTOS los autos», «RESUELVE»—, no «CONSIDERANDO» ni «sentencia
    # definitiva», que es lo que se cita. Sin ninguna marca fuerte, lo demás es
    # cita, y se descuenta. Es el mismo criterio que ya usaba
    # `trae_la_recurrida`, que exigía las fuertes y sólo ellas.
    if not any(rx.search(t) for rx, peso in _SENTENCIA if peso >= 3):
        fuera["sentencia_recurrida"] = max(0, fuera["sentencia_recurrida"] - 4)
    return fuera


def clasificar(texto: str) -> tuple:
    """(tipo, confianza). «acuerdo» cuando no se distingue cuál es."""
    p = puntuar(texto)
    if not any(p.values()):
        return "desconocido", 0
    mejor = max(p, key=lambda k: p[k])
    seg = sorted(p.values(), reverse=True)
    # SI DOS TIPOS EMPATAN, NO SE ELIGE. Un auto que menciona al ponente y
    # también admite el recurso puntúa en los dos, y decidirlo a suertes es lo
    # que hace que el proyecto salga con el magistrado equivocado.
    if len(seg) > 1 and seg[0] == seg[1]:
        if {"auto_admision", "auto_turno"} & {k for k in p if p[k] == seg[0]}:
            return "acuerdo", seg[0]
        return "desconocido", seg[0]
    return mejor, p[mejor]


def trae_la_recurrida(texto_promocion: str) -> bool:
    """¿El escrito que abre el asunto lleva dentro la sentencia recurrida?

    Se exige la marca fuerte —«VISTOS los autos» o el «RESUELVE»—, no que se
    hable de la sentencia: un recurso habla de ella en cada página.
    """
    t = " ".join((texto_promocion or "").split())
    return bool(re.search(r"VISTOS\s+los\s+autos", t, re.I)
                and re.search(r"R\s*E\s*S\s*U\s*E\s*L\s*V\s*E", t, re.I))
