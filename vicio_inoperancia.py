"""EL VICIO CONCRETO DE LA INOPERANCIA (2-oct-2026).

David: «en la cita de jurisprudencias sobre inoperancia siempre es la misma;
el sistema es muy recurrente con jurisprudencias iguales cuando tenemos un
acervo basto».

La causa no era el acervo sino la PREGUNTA. Las tres puertas por las que entra
una tesis de inoperancia preguntaban por el género:

  · la fase 3 marcaba el impedimento siempre con el motivo «inoperancia», así
    que la pregunta sintética del adelanto era «¿Inoperancia: …?» y su
    traducción a rubro caía en «AGRAVIOS INOPERANTES» a secas;
  · /taller/razonar buscaba con un texto FIJO por calificativa más 180
    caracteres del problema: el vector lo dominaba el texto fijo y las cuatro
    tesis salían prácticamente iguales en cualquier asunto;
  · el prompt v1 mandaba citar una tesis de inoperancia aunque el material no
    trajera ninguna, y el modelo ponía la célebre que recuerda.

Pero un secretario no cita «la tesis de la inoperancia»: cita la del VICIO —que
el agravio es novedoso, que parte de una premisa falsa, que reitera sin
combatir—, y vicios distintos tienen jurisprudencia distinta. La variedad que
pide David sale de ahí, no del azar: nunca se cambia una obligatoria de la
Corte por una orientadora sólo por variar.

Aquí vive lo que comparten las puertas: el catálogo de vicios (las mismas
claves tipificadas de `plan_estudio.RAZONES`), cómo se deduce el vicio de un
texto, y los fragmentos de prompt que cambian con la bandera
`inoperancia_por_vicio`. Con la bandera apagada, cada fragmento devuelve
EXACTAMENTE el texto de antes (lo comprueba test_inoperancia_vicio.py).
"""
from __future__ import annotations

import re
import unicodedata

# ═══ EL CATÁLOGO ═════════════════════════════════════════════════════════════
# Las razones de inoperancia tipificadas del plan (plan_estudio.RAZONES y
# _RAZONES_DE_INOPERANTE) más las calificativas que no son «inoperante» pero se
# fundan igual con tesis de técnica. La descripción dice QUÉ es el vicio, para
# que la fase 3 elija la clave y la pregunta sintética lo nombre; no es un
# ejemplo de redacción.
VICIOS = {
    "no_combate": "no combate la consideración toral que sostiene lo resuelto",
    "ataca_accesoria": ("ataca una consideración accesoria o dada a mayor abundamiento y "
                        "deja intocada la que rige el fallo"),
    "generico": "es genérico o dogmático: afirma la ilegalidad sin razonar por qué",
    "reitera_sin_combatir": ("reitera lo planteado en la instancia anterior sin controvertir lo "
                             "que se le respondió"),
    "falsa_premisa": ("parte de una premisa falsa: atribuye al acto lo que no dice o da por "
                      "ocurrido un hecho que no consta"),
    "novedoso": "plantea una cuestión que no se hizo valer ante el órgano que resolvió",
    "cosa_juzgada_amparo_previo": "versa sobre lo ya decidido en un juicio de amparo anterior",
    "procesal_171_172": ("alega una violación procesal que no se preparó como lo exigen los "
                         "artículos 171 y 172 de la Ley de Amparo"),
    "adhesivo_fuera_182": ("en el amparo adhesivo, plantea lo que el artículo 182 de la Ley de "
                           "Amparo no autoriza"),
    "deriva_de_desestimado": "depende de otro planteamiento ya desestimado y corre su suerte",
    "fundado_insuficiente": ("tiene razón, pero no alcanza para variar el sentido porque "
                             "subsisten otras consideraciones que lo sostienen"),
    # Las calificativas con tesis de técnica propias (las de antes).
    "inatendible": ("no puede atenderse por cómo o cuándo se plantea: es ajeno a la litis o "
                    "se dirige contra lo consentido"),
    "ineficaz": "aun fundado, no trasciende al resultado del fallo",
    "innecesario": "su estudio es innecesario porque ya se alcanzó el mayor beneficio",
}

# Las de FORMA: la suplencia confirmada (trabajador, menor, reo) las prohíbe
# (plan_estudio.RAZONES, «forma»). Traerles tesis reintroduciría el error.
VICIOS_DE_FORMA = frozenset(("no_combate", "ataca_accesoria", "generico",
                             "reitera_sin_combatir"))

# Las calificativas del criterio que piden tesis de técnica. «inoperante» se
# abre en sus vicios; las demás SON su propio vicio.
CALIFICATIVAS = ("inoperante", "inatendible", "ineficaz", "innecesario",
                 "fundado_insuficiente")

# Sin saber cuál, el vicio de la inoperancia por omisión: el más frecuente y el
# que menos afirma (no combatir lo que rige no presupone ningún hecho).
VICIO_POR_OMISION = "no_combate"


def activa() -> bool:
    """¿Rige la bandera `inoperancia_por_vicio` en esta petición? Nunca lanza."""
    try:
        import contexto_taller as _ct
        return bool(_ct.rediseno("inoperancia_por_vicio"))
    except Exception:                                   # pragma: no cover
        return False


def _llano(x: str) -> str:
    x = unicodedata.normalize("NFKD", (x or "").lower())
    return " ".join("".join(c for c in x if not unicodedata.combining(c)).split())


# EL ORDEN IMPORTA: lo específico antes que lo general. «Reitera sin combatir»
# contiene «no combate»; «fundado pero insuficiente» no es «no combate» aunque
# diga que subsisten consideraciones.
#
# ANCLADAS, NO SUELTAS (revisión adversarial, 3-oct-2026). La razón que llega
# aquí casi nunca son las dos líneas del secretario: es el párrafo del motor
# (60-160 palabras) o la razón global del modo acervo, que citan fojas, tocas,
# claves de tesis y artículos de cualquier ley. Con números sueltos, «foja
# 171», «el toca 172/2025», «2a./J. 172/2010» o «artículo 182 del Código
# Fiscal» daban la violación procesal no preparada o el adhesivo; «cuyo rubro
# y texto se reproducen» o «como lo ha reiterado la Corte», la reiteración; y
# la excepción de cosa juzgada del juicio natural (tema de FONDO frecuente en
# lo civil), el amparo previo. Ahora el número sólo cuenta junto a «Ley de
# Amparo», la reiteración tiene que recaer sobre los conceptos, los agravios o
# lo planteado, la cosa juzgada tiene que ser la de un amparo y el adhesivo
# tiene que venir con su límite.
_PISTAS = (
    ("cosa_juzgada_amparo_previo", r"amparo (?:anterior|previo)|ejecutoria (?:de amparo )?(?:anterior|previa)"
                                   r"|ya (?:fue|se) (?:decidid|resuelt)\w* en (?:un|el|otro) (?:juicio de )?amparo"
                                   r"|cosa juzgada[^.;]{0,60}\bamparo\b|\bamparo\b[^.;]{0,60}cosa juzgada"),
    ("procesal_171_172", r"\bart(?:iculos?|s?\.)\s*17[12]\b[^.;]{0,60}ley de amparo"
                         r"|\b17[12]\s+(?:y\s+17[12]\s+)?de la ley de amparo"
                         r"|preparacion de la violacion"
                         r"|violacion\w* (?:procesal\w*|del procedimiento|al procedimiento)[^.;]{0,60}"
                         r"no (?:se )?(?:hab\w* )?prepar\w*"
                         r"|no (?:se )?prepar\w*[^.;]{0,40}violacion\w* (?:procesal|del procedimiento)"),
    ("adhesivo_fuera_182", r"adhesiv\w*[^.;]{0,80}(?:\b182\b|no (?:lo )?(?:autoriza|permite|procede)|exced\w*"
                           r"|fuera de (?:lo|los)|ajen\w*)"
                           r"|\b182\b[^.;]{0,40}ley de amparo"),
    ("deriva_de_desestimado", r"(?:deriva|depende|hace\w* depender)\w* de (?:otros?|aquel\w*|los ya|las ya|lo ya)\b"
                              r"|corre\w* (?:la )?(?:misma )?suerte|(?:de|a) (?:los|las|lo) ya desestimad"),
    ("fundado_insuficiente", r"fundad\w* pero (?:insuficien|inoperan|ineficaz)|aun (?:cuando|siendo|si) (?:fuera |resultara )?fundad"
                             r"|no trasciende|insuficiente\w* para (?:revocar|modificar|conceder|variar)"),
    ("novedoso", r"novedos\w*|no (?:se )?(?:hizo|hicieron|planteo|plantearon|adujo|adujeron|formulo) valer"
                 r"|no (?:fue|fueron) (?:planteado|aducido|propuesto|formulado)\w*|no form\w* parte de la litis"
                 r"|cuestion\w* no (?:aducid|planteadas?|propuest)\w*"),
    ("falsa_premisa", r"premisa\w* (?:falsa|inexacta|equivocada|erronea|incorrecta)|falsa\w* premisa"
                      r"|parte de (?:un|una) (?:supuesto|premisa)"),
    ("reitera_sin_combatir", r"(?:reiter|reproduc|repit|repet)\w*[^.;]{0,30}?"
                             r"(?:conceptos|argumentos|agravios|planteamientos|alegatos"
                             r"|lo (?:ya )?(?:planteado|alegado|expuesto|dicho|hecho valer|aducido))"
                             r"|transcrib\w* (?:los|lo) (?:conceptos|argumentos)"),
    ("ataca_accesoria", r"accesori\w*|mayor abundamiento|obiter|secundari\w*|ex abundantia"),
    ("generico", r"generic\w*|dogmatic\w*|\bvag[oa]s?\b|abstract\w*|sin (?:precisar|razonar|explicar)"
                 r"|meras? afirmacion"),
    ("no_combate", r"no (?:se )?(?:combat|controvi|ataca|impugn|desvirt)\w*|intocad\w*|toral|consideracion\w* que (?:sustenta|rige)"),
)
_PISTAS_RX = tuple((k, re.compile(p)) for k, p in _PISTAS)


def vicio_de_texto(texto: str) -> str:
    """La clave del vicio que el texto nombra, o «» si no nombra ninguno. Sólo
    expresiones regulares: barato, determinista y igual en los dos workers."""
    t = _llano(texto)
    if not t:
        return ""
    for clave, rx in _PISTAS_RX:
        if rx.search(t):
            return clave
    return ""


def vicio_de(sentido: str, texto: str = "", vicio: str = "") -> str:
    """El vicio de un criterio. Para «inoperante»: el declarado si es del
    catálogo; si no, el que nombra su razón; si no, la inoperancia a secas
    (`no_combate`). Las otras calificativas de técnica (inatendible, ineficaz,
    innecesario, fundado pero insuficiente) SON su vicio. «» si la
    calificativa no es de técnica (fundado, infundado, sin materia): para ésas
    basta el material del caso.

    LA CALIFICATIVA NO SE DESPLAZA (revisión adversarial, 3-oct-2026): antes un
    vicio sacado del texto de la razón ganaba aun sobre «ineficaz» —«artículo
    172 del Código» la volvía la violación procesal no preparada—. Lo que el
    secretario calificó como ineficaz se funda con tesis de la ineficacia."""
    s = (sentido or "").strip().lower().replace(" ", "_")
    if s not in CALIFICATIVAS:
        return ""
    if s != "inoperante":
        return s
    v = (vicio or "").strip().lower()
    if v in VICIOS:
        return v
    return vicio_de_texto(texto) or VICIO_POR_OMISION


# ═══ LA FASE 3: EL MOTIVO DEL IMPEDIMENTO ════════════════════════════════════
_IMPEDIMENTO_ANTES = ('Si adviertes un impedimento técnico que llevaría a inoperancia, ponlo en\n'
                      '"impedimento" como {"motivo": "inoperancia", "explicacion": "..."}.')


# Los vicios que sólo existen en una vía: la preparación de la violación
# procesal (arts. 171-172) y el adhesivo (art. 182) son del amparo directo; la
# reiteración de lo dicho en la instancia anterior, de los recursos
# (plan_estudio.RAZONES, «solo_recursos»).
_SOLO_DIRECTO = ("procesal_171_172", "adhesivo_fuera_182")
_SOLO_RECURSOS = ("reitera_sin_combatir",)


def cabe_en_la_via(vicio: str, tipo_asunto: str = "") -> bool:
    """¿Este vicio existe en esta vía? La misma partición que `regla_impedimento`:
    la violación procesal no preparada y el adhesivo, sólo en el amparo
    directo; la reiteración de la instancia, sólo en los recursos. Sin tipo
    conocido, sí (3-oct-2026: el vicio DEDUCIDO de la razón no pasaba por aquí
    y en una revisión fiscal se traían tesis del adhesivo)."""
    try:
        import tipos_asunto as _ta
        t = _ta.normalizar(tipo_asunto or "") or (tipo_asunto or "")
    except Exception:                                   # pragma: no cover
        t = tipo_asunto or ""
    if not t:
        return True
    if t == "amparo_directo":
        return vicio not in _SOLO_RECURSOS
    return vicio not in _SOLO_DIRECTO


def regla_impedimento(tipo_asunto: str = "") -> str:
    """El renglón del impedimento del prompt de problemas (fases123). Con la
    bandera, pide además la clave del vicio, DESCRITA —cada clave con lo que
    significa—, sin un ejemplo que copiar, y sólo las que caben en la vía.
    «motivo» sigue valiendo «inoperancia»: la pantalla lo pinta («posible
    inoperancia»)."""
    if not activa():
        return _IMPEDIMENTO_ANTES
    fuera = ("ineficaz", "innecesario", "inatendible")
    if tipo_asunto == "amparo_directo":
        fuera += _SOLO_RECURSOS
    elif tipo_asunto:
        fuera += _SOLO_DIRECTO
    claves = [k for k in VICIOS if k not in fuera]
    lista = "\n".join(f"  · {k}: {VICIOS[k]}" for k in claves)
    return ('Si adviertes un impedimento técnico que llevaría a inoperancia, ponlo en\n'
            '"impedimento" como un objeto con tres campos: "motivo", que vale siempre\n'
            '"inoperancia"; "vicio", la clave del vicio concreto, una de éstas:\n'
            f'{lista}\n'
            'y "explicacion", en qué consiste aquí ese vicio. Si ninguna clave describe\n'
            'lo que adviertes, deja "vicio" en null y explícalo.')


# ═══ EL ADELANTO: LA PREGUNTA SINTÉTICA ══════════════════════════════════════
def pregunta_sintetica(clave: str, pre: str, x: dict) -> str:
    """La pregunta que el impedimento (o el apoyo) añade a la consulta del
    acervo. Sin la bandera, la de antes: «¿Inoperancia: …?». Con ella, el
    impedimento nombra su vicio —el declarado o el que dice la explicación—:
    así la traducción a rubro no cae en el género «AGRAVIOS INOPERANTES»."""
    base = f"¿{str(x.get('motivo') or pre).capitalize()}: {x['explicacion']}?"
    if clave != "impedimento" or not activa():
        return base
    v = str(x.get("vicio") or "").strip().lower()
    if v not in VICIOS:
        v = vicio_de_texto(str(x.get("explicacion") or ""))
    if not v:
        return base
    return f"¿Inoperancia porque el planteamiento {VICIOS[v]}: {x['explicacion']}?"


# ═══ EL ESTUDIO: LA REGLA DE LA INOPERANCIA Y EL EJEMPLO DE LA ANALOGÍA ══════
_REGLA_V1_ANTES = """    planteamiento, SE CITA UNA TESIS SOBRE LA INOPERANCIA de que se trate
    —hay criterios para cada tipo: no combatir la razón toral, ser novedoso,
    partir de premisa falsa, versar sobre cuestión firme— y SE APLICA a este
    caso con el mismo diálogo jurídico que el resto: la regla del criterio,
    los hechos de aquí, y por qué encajan.

    La tesis hace el trabajo pesado, y por eso el apartado es corto. La
    inoperancia se razona con autoridad, no se desarrolla con párrafos: un
    apartado de inoperancia más largo que el del tema principal delata que no
    se supo dónde estaba el asunto."""


def regla_inoperante_v1(q1: str) -> str:
    """La regla del apartado inoperante del prompt v1. Con la bandera: la tesis
    sale del MATERIAL y es la del vicio de ese planteamiento; sin ella, se
    razona sin cita. La v1 mandaba citar aunque no hubiera ninguna, y el modelo
    ponía la que recuerda."""
    if not activa():
        return _REGLA_V1_ANTES
    return f"""    planteamiento, se dice cuál es su vicio —no combatir la razón toral, ser
    novedoso, partir de premisa falsa, reiterar sin controvertir, versar sobre
    cuestión firme…— y SE APLICA a este caso con el mismo diálogo jurídico que
    el resto: la regla, los hechos de aquí, y por qué encajan.

    LA TESIS DE LA INOPERANCIA SALE DEL MATERIAL, Y ES LA DE ESE VICIO: entre
    las tesis de la técnica, la que trata el mismo vicio que este {q1}. Dos
    planteamientos con vicios distintos llevan tesis distintas; el mismo vicio
    repetido remite a la ya citada. Si el material no trae ninguna sobre ese vicio, la
    inoperancia se razona sin cita: NUNCA se cita de memoria una tesis que no
    está en el material.

    La razón hace el trabajo pesado, y por eso el apartado es corto. La
    inoperancia no se desarrolla con párrafos: un apartado de inoperancia más
    largo que el del tema principal delata que no se supo dónde estaba el
    asunto."""


_REGLA_V4_ANTES = """    · INOPERANTE: de uno a tres párrafos que dicen qué consideración deja sin
      combatir, o por qué no puede examinarse, con su razón. La tesis sobre la
      inoperancia se cita sólo si hace falta para sostenerla."""


def regla_inoperante_v4() -> str:
    """El renglón INOPERANTE de la medida de los apartados (v2-v4)."""
    if not activa():
        return _REGLA_V4_ANTES
    return """    · INOPERANTE: de uno a tres párrafos que dicen qué consideración deja sin
      combatir, o por qué no puede examinarse, con su razón. La tesis sobre la
      inoperancia se cita sólo si hace falta para sostenerla, y sale del
      MATERIAL: la de la técnica que trata el vicio de ESTE argumento (no
      combatir, novedad, premisa falsa, reiteración…). Vicios distintos, tesis
      distintas. Si el material no trae una sobre ese vicio, se razona sin
      cita; nunca de memoria."""


_ANALOGIA_ANTES = """        «Sustenta esa consideración, por analogía, la jurisprudencia 2a./J.
         58/2010 de la Segunda Sala de la Suprema Corte de Justicia de la
         Nación, de registro …, de rubro y texto siguientes:»"""


def ejemplo_analogia() -> str:
    """La fórmula de la cita por analogía. Llevaba una clave REAL —la 2a./J.
    58/2010— y un ejemplo en el prompt se copia literal (medido tres veces):
    con la bandera, la forma se describe sin clave."""
    if not activa():
        return _ANALOGIA_ANTES
    return """        «Sustenta esa consideración, por analogía, la jurisprudencia
         [su clave] de [el órgano que la emitió], de registro [su número],
         de rubro y texto siguientes:»"""


_CLAVE_ANTES = "La clave —«2a./J. 58/2010»— no lo sustituye"


def clave_no_sustituye() -> str:
    """«La clave no sustituye al registro», sin una clave real que copiar."""
    if not activa():
        return _CLAVE_ANTES
    return "La clave de la tesis no lo sustituye"
