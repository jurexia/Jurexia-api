"""LO QUE SE VE AL LEER: FRASES ROTAS, COMILLAS HUÉRFANAS Y NOMBRES GENÉRICOS.

David enumera los síntomas exactos, tomados de proyectos reales:

    «...cuyo texto. Del precepto...»              frase cortada
    «...debe contener , así como y.»              puntuación doble, conjunción sola
    «...consecuencia automática d la jurisprudencia...»  preposición amputada
    «registro 174962, no resulta aplicable...»    fragmento sin sujeto
    un « sin su »                                  comilla huérfana
    «PARTE QUEJOSA», «PARTE PROMOVENTE»            el marcador en vez del nombre

═══════════════════════════════════════════════════════════════════════════
LO QUE ESTE LINTER TIENE PROHIBIDO CAZAR
═══════════════════════════════════════════════════════════════════════════
Este proyecto lleva tres rondas aprendiendo que un detector que acusa a lo
bueno es peor que no tenerlo, así que la lista de lo que NO debe saltar se
escribió ANTES que los patrones, sacada de los engroses reales:

  · las abreviaturas: «art.», «fr.», «frs.», «Reg.», «pág.», «núm.», «2a./J.»,
    «P./J.», «I.4o.A.», que llevan punto y siguen en minúscula;
  · los incisos «a)», «b)», y los romanos sueltos de una fracción: «IV.»;
  · las enumeraciones con «y,» y con «, y»: «los artículos 74, 76, y 79»;
  · el cierre de una transcripción parcial: «[…]» y «(…)»;
  · «d.» como abreviatura no existe en español jurídico, pero «D.O.F.» sí;
  · las cifras con punto y coma: «$847,738.77;».

Y una regla de método: cada patrón trae aquí sus casos de control, y si un
control falla, se quita el patrón. No se afina hasta que pase.
"""

from __future__ import annotations

import re

# ── 1. PUNTUACIÓN IMPOSIBLE ────────────────────────────────────────────────
# Dos signos seguidos que no forman ninguna construcción del español: «,,»,
# «.,», « ,», «;.». Se exige el espacio ANTES de la coma porque «847,738» es
# una cifra y «74, 76» es una enumeración.
# «Qro.,» y «6o.,» y «art.,» son ABREVIATURAS seguidas de coma, y en español
# jurídico se escriben así todo el rato. Cazarlas hacía que el linter acusara a
# tres de los cinco engroses reales por escribir bien.
_ABREV_ANTES = (r"(?<!\bQro)(?<!\bMéx)(?<!\bC\.P)(?<!\bD\.F)(?<!\bS\.A)"
                r"(?<!\bart)(?<!\bfr)(?<!\bp[áa]g)(?<!\bn[úu]m)"
                r"(?<![0-9][oa])(?<![IVXLC])")

_PUNTUACION = [
    (_ABREV_ANTES + r"[.,;:]\s*[,;](?!\s*\d)",
     "dos signos de puntuación seguidos"),
    (r"\s+[,;](?=\s)", "coma o punto y coma precedidos de espacio"),
    (r"\(\s*\)|«\s*»|“\s*”", "un paréntesis o unas comillas vacías"),
]

# ── 2. PREPOSICIÓN AMPUTADA ────────────────────────────────────────────────
# «d la», «e el», «n el»: una letra suelta que era una preposición. Se exige
# que vaya seguida de artículo, que es donde se ve; una «y» o una «o» sueltas
# son conjunciones legítimas.
_AMPUTADA = re.compile(
    r"\b([bcdfghjklmnpqrstvwxz])\s+(?:el|la|los|las|un|una)\b", re.I)

# ── 3. COMILLA HUÉRFANA ────────────────────────────────────────────────────
# Se cuentan, no se buscan: un « sin su » no se ve mirando una ventana.

# ── 4. EL MARCADOR EN VEZ DEL NOMBRE ───────────────────────────────────────
# «la parte quejosa» en minúsculas, dentro de la prosa, es CORRECTO en amparo
# directo: lo dice el catálogo. Lo que no puede aparecer es el marcador en
# VERSALES —que es donde va el nombre— ni en un punto resolutivo.
_GENERICO = re.compile(
    r"\b(PARTE\s+QUEJOSA|PARTE\s+PROMOVENTE|PARTE\s+RECURRENTE|"
    r"NOMBRE\s+DEL\s+QUEJOSO|AUTORIDAD\s+RESPONSABLE\s*:?\s*$)\b")
# EL RESOLUTIVO EMPIEZA DESPUÉS DEL «R E S U E L V E». Sin ese ancla, el
# patrón cazaba «TERCERO. Resolución recurrida y agravios de la parte
# recurrente», que es un RÓTULO de considerando perfectamente correcto —lo dice
# el catálogo— y acusaba al engrose por escribirlo.
_RX_RESOLUTIVO = re.compile(
    r"R\s*E\s*S\s*U\s*E\s*L\s*V\s*E.{0,2500}", re.S | re.I)

# LO QUE ESTE LINTER NO SABE CAZAR, y lo digo en vez de inventar un patrón:
# «El precepto cuyo texto. Del precepto transcrito deriva la regla.» El punto
# cae tras un SUSTANTIVO —«texto»—, así que no hay palabra imposible que
# buscar: hace falta saber que la oración no tiene verbo, y eso pide analizar
# la frase, no mirar una ventana. Un patrón aproximado acusaría a los títulos y
# a las enumeraciones, que es peor que no tenerlo.
#
# ── 6. LA FRASE QUE SE CORTA A MITAD ───────────────────────────────────────
# «El precepto cuyo texto. Del precepto transcrito deriva la regla.» El punto
# cae después de una palabra que EXIGE continuación: un relativo, una
# preposición, una conjunción. No hace falta gramática fina —basta la lista de
# palabras que jamás terminan una oración en español.
_COLGADA = re.compile(
    r"\b(cuy[oa]s?|del?\s+cual(?:es)?|en\s+(?:el|la|los|las)\s+(?:que|cual)|"
    r"as[íi]\s+como|mediante|conforme|respecto|acorde|"
    r"de|en|por|para|con|sin|sobre|entre|hacia|desde|hasta|y|o|e|u|que|"
    r"porque|pues|aunque|si|cuando|donde|seg[úu]n)\s*\.", re.I)

# ── 5. FRASE SIN VERBO tras un punto ───────────────────────────────────────
# «registro 174962, no resulta aplicable...» empieza sin sujeto porque el punto
# anterior cortó la frase. Se busca una oración que ARRANCA en minúscula, que
# es la marca inequívoca —salvo abreviatura, que se excluye.
_ABREV = (r"art|arts|fr|frs|reg|p[áa]g|n[úu]m|vol|cfr|op|cit|etc|"
          r"[a-z]?\d*[ao]|[IVXLC]+|Sr|Sra|Lic|Mtro|Dr|D\.O\.F|inc")
_MINUSCULA = re.compile(
    r"(?<![.]\s)(?<!\[…\])\.\s+([a-záéíóúñ]{3,})",
)


def _sin_abreviatura(texto: str, m) -> bool:
    antes = texto[max(0, m.start() - 18):m.start()]
    return not re.search(rf"\b(?:{_ABREV})$", antes, re.I)


def revisar(texto: str) -> list:
    """Los defectos de sintaxis del documento. [(qué, dónde)]"""
    t = texto or ""
    fuera: list = []

    for patron, que in _PUNTUACION:
        for m in re.finditer(patron, t):
            ctx = " ".join(t[max(0, m.start() - 55):m.end() + 45].split())
            fuera.append((que, ctx))

    for m in _AMPUTADA.finditer(t):
        ctx = " ".join(t[max(0, m.start() - 45):m.end() + 45].split())
        fuera.append((f"preposición amputada: «{m.group(0)}»", ctx))

    for a, b, nombre in (("«", "»", "angulares"), ("“", "”", "tipográficas")):
        if t.count(a) != t.count(b):
            fuera.append((f"comillas {nombre} sin cerrar: "
                          f"{t.count(a)} abren y {t.count(b)} cierran", ""))

    # El marcador genérico: en versales, y en los resolutivos.
    for m in _GENERICO.finditer(t):
        ctx = " ".join(t[max(0, m.start() - 55):m.end() + 45].split())
        fuera.append((f"el marcador en vez del nombre: «{m.group(0)}»", ctx))
    for m in _RX_RESOLUTIVO.finditer(t):
        if re.search(r"(?:ampara|protege|sobresee|desecha|confirma|revoca|"
                     r"modifica)[^.]{0,60}\bparte\s+"
                     r"(?:quejosa|promovente|recurrente)\b", m.group(0), re.I):
            fuera.append(("el resolutivo nombra a «la parte» en vez de a la "
                          "persona", " ".join(m.group(0).split())[:150]))

    for m in _COLGADA.finditer(t):
        ctx = " ".join(t[max(0, m.start() - 55):m.end() + 45].split())
        fuera.append((f"la frase se corta en «{m.group(1)}», que exige "
                      f"continuación", ctx))

    fuera += [(f"orden contradictoria: {q}", d) for q, d in oximorones(t)]

    # LA VARIABLE QUE NO SE SUSTITUYÓ. El catálogo interpola con `{quejoso}`,
    # `{responsable}`, `{numero}`. Si una plantilla llega al documento sin
    # rellenar, la llave se ve —y lo que se ve, alguien lo firma—.
    for m in _RX_LLAVE.finditer(t):
        ctx = " ".join(t[max(0, m.start() - 55):m.end() + 45].split())
        fuera.append((f"variable sin sustituir: «{m.group(0)}»", ctx))

    # EL COMODÍN DE ASTERISCOS. `banco.py` lo usa a propósito —«se ve, y lo que
    # se ve no se firma sin mirarlo»— y ésa es la intención correcta. Pero
    # nadie lo estaba CONTANDO al final, así que se veía en el documento sin
    # aparecer en los avisos: quien recibía el proyecto tenía que tropezárselo.
    n_hueco = len(_RX_HUECO.findall(t))
    if n_hueco:
        m = _RX_HUECO.search(t)
        fuera.append((f"quedan {n_hueco} comodines sin resolver: son datos que "
                      f"el expediente tiene y el formulario no pidió",
                      " ".join(t[max(0, m.start() - 70):m.end() + 60].split())))

    for m in _MINUSCULA.finditer(t):
        if _sin_abreviatura(t, m):
            ctx = " ".join(t[max(0, m.start() - 60):m.end() + 40].split())
            fuera.append(("frase que empieza en minúscula: el punto anterior "
                          "cortó la oración", ctx))

    # Un mismo defecto repetido cansa; se entregan los seis primeros de cada
    # clase, que es lo que un secretario revisa de una sentada.
    vistos: dict = {}
    limpio: list = []
    for que, ctx in fuera:
        clave = que.split(":")[0]
        vistos[clave] = vistos.get(clave, 0) + 1
        if vistos[clave] <= 6:
            limpio.append((que, ctx))
    return limpio


# Sólo las llaves con un nombre de variable dentro: `{quejoso}`, `{numero_2}`.
# No se caza `{` a secas porque en una transcripción de ley puede aparecer.
_RX_LLAVE = re.compile(r"\{[a-z_][a-z0-9_]{2,30}\}")
_RX_HUECO = re.compile(r"\*{4,}")


# ── 7. LA FÓRMULA QUE SE CONTRADICE A SÍ MISMA ─────────────────────────────
# David: «fórmulas oximorónicas del tipo "reitere con plenitud de
# jurisdicción"». No es un capricho de estilo: son dos destinos opuestos del
# asunto y no pueden convivir en la misma orden.
#
#   PLENITUD DE JURISDICCIÓN es lo que hace ESTE tribunal cuando, levantado el
#   sobreseimiento o revocada la sentencia, resuelve él mismo lo que el a quo
#   no resolvió. Es asumir la decisión, no delegarla.
#
#   REITERAR, REPONER o DEJAR INSUBSISTENTE es lo que se ordena a OTRO órgano.
#   Es devolver la decisión, no asumirla.
#
# Mandar «reitere con plenitud de jurisdicción» ordena a la responsable que
# haga lo que por definición hace quien no recibe órdenes. Quien lo lea no sabe
# quién decide.
_OXIMORONES = [
    (r"(?:reitere|reitera|reiterando|dicte|emita|deje\s+insubsistente|"
     r"repon(?:ga|iendo)|devu[ée]lva\w*)[^.;]{0,80}"
     r"plenitud\s+de\s+jurisdicci[óo]n",
     "se ordena a otro órgano actuar «con plenitud de jurisdicción», que es "
     "lo que hace quien resuelve por sí mismo: o se asume la decisión o se "
     "devuelve, no las dos"),
    (r"plenitud\s+de\s+jurisdicci[óo]n[^.;]{0,80}"
     r"(?:reitere|reitera|dicte|emita|deje\s+insubsistente|repon(?:ga|iendo))",
     "se invoca la plenitud de jurisdicción y en la misma frase se ordena a "
     "otro que dicte: son destinos opuestos del asunto"),
    (r"se\s+confirma[^.;]{0,60}y\s+se\s+revoca|"
     r"se\s+revoca[^.;]{0,60}y\s+se\s+confirma",
     "confirmar y revocar lo mismo en el mismo punto"),
    (r"se\s+sobresee[^.;]{0,80}(?:ampara\s+y\s+protege|se\s+concede\s+el\s+amparo)",
     "sobreseer y amparar sobre el mismo acto: el sobreseimiento impide "
     "entrar al fondo"),
]
_OXIMORONES = [(re.compile(p, re.I | re.S), q) for p, q in _OXIMORONES]


def oximorones(texto: str) -> list:
    """Órdenes que se anulan entre sí. [(qué, dónde)]"""
    t = texto or ""
    fuera = []
    for rx, porque in _OXIMORONES:
        m = rx.search(t)
        if m:
            fuera.append((porque, " ".join(m.group(0).split())[:150]))
    return fuera
