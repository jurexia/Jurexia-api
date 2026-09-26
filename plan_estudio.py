# -*- coding: utf-8 -*-
"""EL PLAN DEL ESTUDIO: quién contesta qué, dónde se expone cada premisa y en
qué orden, decidido ANTES de redactar (Paso 2b + 3 de la propuesta del
estudio, aprobada por David el 26-sep-2026; `diag/w2_final.md` §3 y §4 y el
contrato común `diag/contrato_paso2.md`).

POR QUÉ EXISTE. Medido el 26-sep-2026 en 8 casos × 2 corridas, con un
localizador ciego y 675 citas verificadas: la v2 del prompt reduce la Solución
a la mitad y repite menos, pero contesta con razón propia sólo el 73 % de los
argumentos autónomos (la v1, el 79 %) y DUPLICA las omisiones graves (20
frente a 10). Al acortar sin saber qué argumentos hay, funde los que traen un
dato propio en una respuesta global. Pedirle «no repitas» o «contesta todo» no
lo arregla: ninguna etapa decidía qué premisa es común a varios argumentos ni
qué trae cada uno de suyo. Aquí se decide, con datos, y el estudio lo recibe
como guion.

LO QUE NO HACE, Y NO SE NEGOCIA
  · NO DECIDE EL SENTIDO DE NINGÚN PROBLEMA. El de cada problema es el que
    fijó el secretario y no se toca. DENTRO de él, cada argumento lleva su
    propia calificación razonada (plan-4, 26-sep-2026: el ADC 642/2024 v4
    sacó «fundado» el argumento de la reconvención porque el plan le ponía la
    etiqueta de su problema, y la Sala sí la había examinado), con la regla de
    coherencia que se comprueba aquí: problema que prospera ⇒ al menos un
    argumento lo funda; problema que no prospera ⇒ ningún argumento queda
    fundado sin decir por qué no alcanza (va como «fundado pero
    insuficiente»); problema que no se estudia ⇒ sus argumentos, tampoco. Lo
    que sí sería cambiar el sentido de un PROBLEMA sale como PROPUESTA
    visible en el panel y no se aplica sin su clic. No se usa
    `fase6_estudio._misma_direccion`: trata como neutro todo sentido fuera de
    sus dos listas (w2_final §1.5, L1), justo el agujero por el que un plan
    podía cambiar la calificación sin que nada lo detectara.
  · NO REDACTA REGLAS. Una premisa son sus fuentes y sus anclas; la regla la
    escribe el estudio a partir de la razón del secretario.
  · NO INVENTA DATOS. La cita de cada segmento, la de su dato, la de cada
    proposición y la del rastro de cada premisa se verifican palabra por
    palabra; lo que no se encuentra se borra (V0 g).

LAS PIEZAS (firmas del contrato; lo que se añade va sólo como keyword):
  clave()     identifica un plan: adelanto + criterio + contexto + suplencia.
  planear()   la llamada al planificador: el modelo de las fases con
              razonamiento medio y JSON estricto.
  reparar()   lo que el código corrige solo: la etiqueta que no cabe en el
              sentido de su problema (sin estudio dentro de un problema que
              se estudia → la del problema; fundado dentro de uno que no
              prospera → fundado pero insuficiente; un problema que prospera
              sin argumento que lo funde → todos a la del problema y la otra,
              a propuestas); `reitera` con ancla propia → `desarrolla`;
              datos sin verificar, borrados. Y lo organizativo que V0
              rechazaba (26-sep-2026): la unidad que mezcla, partida; la
              premisa sin rastro, retirada; la razón que no cabe con la
              etiqueta, la de la etiqueta (la otra, dicha en el aviso: no
              es una propuesta de calificación); la cita
              que nadie encontró, fuera; referencias rotas, limpias; el orden
              del art. 189, cuando se puede sin decidir. Cada cosa, dicha.
  validar()   V0: lo que obliga a rehacer el plan. [] = válido.
  preparar()  planear → reparar → validar; un reintento con la lista de lo
              que falló; si vuelve a fallar, NO hay plan (el estudio va con la
              v3) y se dice.
  vista()     el GUION por forma (estándar o moderna): datos, sin frases.
  bloque()    el guion con la descripción de sus rótulos, tal como entra al
              prompt del estudio (v4).
  fila_*()    el estado en `taller_sesiones.plan`, en funciones puras, para
              el compare-and-set de main.py (gunicorn -w 2: nada de esto vive
              en memoria).

LECCIÓN QUE NO SE NEGOCIA: en un prompt van DESCRIPCIONES y DATOS, nunca
frases modelo (un ejemplo escrito en el prompt se firma literal; medido tres
veces en este proyecto). Ni el prompt del planificador ni el guion traen una
sola frase para copiar.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
import unicodedata

# ═══ CONSTANTES ═════════════════════════════════════════════════════════════

# Sube cuando cambie el esquema, el prompt o una regla de V0: un plan hecho con
# otra versión no se reutiliza (entra en la clave).
# plan-3: reparar la organización (partir unidades, retirar premisas sin rastro…).
# plan-4: la calificación por argumento dentro del sentido del problema (V0 b
# deja de exigir igualdad y exige coherencia): un plan-3 traía la etiqueta del
# problema en cada argumento y no se reutiliza.
PLAN_VERSION = "plan-4"

# EL MODELO DE LAS FASES, con razonamiento MEDIO (propuesta §3.6: «Se mide
# ESFUERZO_PLAN=medium contra high»). Se lee al llamar, no al importar, para
# que una prueba pueda cambiarlo. El razonamiento del ESTUDIO no se toca: ése
# vive en fase6_estudio y sigue en «high».
ESFUERZO_PLAN = os.getenv("ESFUERZO_PLAN", "medium")
# Cupo de salida: razonamiento medio + un JSON que en el 642 lleva ~20
# segmentos. Si vuelve vacía o cortada se pide una vez con el doble.
TOPE_SALIDA = int(os.getenv("PLAN_TOPE_SALIDA", "16000"))

# CUATRO CORRIDAS POR SESIÓN, contando todas (pedido de la pantalla, precálculo
# y resolver). Una corrida es un pedido: el planificador y, si V0 lo rechaza,
# su único reintento. Coste [H]: ~0.02 USD por llamada (w2_final §3.6).
TOPE_CORRIDAS = int(os.getenv("PLAN_TOPE_CORRIDAS", "4"))

# LO QUE EL RESOLVER ESPERA COMO MUCHO, dentro de la tarea del flujo y nunca
# antes de las cabeceras (la pasarela corta a los ~280 s). Si vence, el estudio
# se escribe SIN plan (v3) y se dice.
ESPERA_RESOLVER_S = float(os.getenv("PLAN_ESPERA_S", "120"))

# Una corrida «en curso» sin latido en este tiempo es de un worker que murió
# (despliegue, SIGTERM): se trata como abandonada. MENOR QUE LA ESPERA DEL
# RESOLVER (revisión adversarial de la integración, 26-sep-2026): con 150 s de
# umbral y 120 de espera, el resolver nunca veía muerta la corrida de un worker
# caído y el estudio iba sin plan; ahora late cada 20 s (`main._taller_plan_
# correr`), se da por muerta con tres latidos perdidos y el resolver la relanza
# dentro de su ventana.
LATIDO_S = 20.0
PLAN_ABANDONADO_S = 3 * LATIDO_S

# Tramos del escrito que ve el planificador: tope TOTAL y, dentro, un tope
# POR CONCEPTO (cabeza y cola) para que el recorte nunca se lleve el final de
# un concepto —que es donde suelen ir el precedente invocado o la petición de
# suplencia (escrito_642: 641-661 y 720-723)—.
TRAMOS_TOPE_TOTAL = 48000
TRAMO_TOPE_MIN = 2500

# Mínimo de palabras para que una cita cuente como verificada. Por debajo, una
# «cita» casa por azar en cualquier escrito largo.
MIN_PALABRAS_CITA_SEGMENTO = 6
MIN_PALABRAS_CITA = 4

# ═══ CATÁLOGOS CERRADOS (w2_final §4.2); el validador rechaza lo demás ══════

TRATAMIENTOS = ("aplica", "remite", "desarrolla", "residual", "no_se_estudia",
                "no_se_expresa_art79")
# Los que se contestan dentro de una unidad (con su premisa).
_EN_UNIDAD = ("aplica", "remite", "desarrolla")
VICIOS = ("procedencia", "procesal", "forma", "omision", "fondo")
DIFERENCIAS = ("hecho", "prueba", "norma", "precedente", "procesal", "consecuencia")
CARACTERES = ("toral", "accesoria", "obiter", "consecuencia")
FUENTES_DATO = ("escrito", "reclamada", "constancia", "antecedentes")

# Las tres clases a las que se reduce toda calificación. Es el «mapa cerrado»
# de V0 (b): prospera / no prospera / no se estudia, con `tipos_asunto.prospera`
# como única fuente de si prospera.
PROSPERA, NO_PROSPERA, NO_SE_ESTUDIA = "prospera", "no_prospera", "no_se_estudia"
_SENTIDOS_NO_SE_ESTUDIA = {"innecesario", "sin_materia", "cae_con_principal",
                           "no_se_estudia", "adhesivo_sin_materia"}

# Las 17 razones tipificadas: la clase que implican y, cuando la calificación
# es una de las «finas», cuáles admite. «forma» = inoperancia por cómo se
# planteó el argumento: la que la suplencia confirmada prohíbe (V0 k).
RAZONES = {
    "fondo_desestimado":           {"clase": NO_PROSPERA},
    "omision_inexistente":         {"clase": NO_PROSPERA},
    "fundado":                     {"clase": PROSPERA},
    "esencialmente_fundado":       {"clase": PROSPERA},
    "fundado_insuficiente":        {"clase": NO_PROSPERA},
    "no_combate":                  {"clase": NO_PROSPERA, "forma": True, "con_p": True},
    "ataca_accesoria":             {"clase": NO_PROSPERA, "forma": True, "con_p": True},
    "generico":                    {"clase": NO_PROSPERA, "forma": True},
    "reitera_sin_combatir":        {"clase": NO_PROSPERA, "forma": True, "solo_recursos": True},
    "falsa_premisa":               {"clase": NO_PROSPERA},
    "novedoso":                    {"clase": NO_PROSPERA},
    "cosa_juzgada_amparo_previo":  {"clase": NO_PROSPERA, "con_p": True},
    "procesal_171_172":            {"clase": NO_PROSPERA},
    "adhesivo_fuera_182":          {"clase": NO_PROSPERA},
    "innecesario_mayor_beneficio": {"clase": NO_SE_ESTUDIA},
    "cae_con_principal":           {"clase": NO_SE_ESTUDIA, "nunca_procesal": True},
    "sin_materia":                 {"clase": NO_SE_ESTUDIA, "nunca_procesal": True},
    "adhesivo_sin_materia":        {"clase": NO_SE_ESTUDIA, "nunca_procesal": True},
}
# Calificaciones finas: la razón tiene que ser una de éstas.
_RAZONES_DE_INFUNDADO = {"fondo_desestimado", "omision_inexistente"}
_RAZONES_DE_INOPERANTE = {"no_combate", "ataca_accesoria", "generico",
                          "reitera_sin_combatir", "falsa_premisa", "novedoso",
                          "cosa_juzgada_amparo_previo", "procesal_171_172",
                          "adhesivo_fuera_182", "fundado_insuficiente"}
_RAZONES_FORMA = {k for k, v in RAZONES.items() if v.get("forma")}

# Qué dice cada razón, para el prompt del planificador (descripciones, no
# frases: lección del ejemplo que se firma literal).
_DESCRIBE_RAZON = {
    "fondo_desestimado": "el argumento se examina y no tiene razón en el fondo",
    "omision_inexistente": "alega una omisión de la responsable que no existe; el dato dice dónde se pronunció",
    "fundado": "tiene razón y prospera",
    "esencialmente_fundado": "tiene razón en lo sustancial y prospera; hay que acotar en qué parte",
    "fundado_insuficiente": "tiene razón pero no alcanza: sobrevive otra proposición suficiente (nombrarla en ataca)",
    "no_combate": "no combate la proposición que sostiene lo resuelto (ataca = la proposición que deja sin combatir)",
    "ataca_accesoria": "ataca una proposición accesoria y deja firme la toral (ataca = la accesoria)",
    "generico": "no se entiende qué combate ni por qué",
    "reitera_sin_combatir": "repite lo alegado en la instancia sin combatir la respuesta (sólo en recursos)",
    "falsa_premisa": "parte de un hecho o una lectura del acto que no es cierta",
    "novedoso": "plantea algo que no se hizo valer cuando debía",
    "cosa_juzgada_amparo_previo": "discute lo que una ejecutoria de amparo previa ya vinculó (ataca = la proposición vinculada)",
    "procesal_171_172": "violación procesal no preparada o sin trascendencia al fallo (arts. 171 y 172 LA)",
    "adhesivo_fuera_182": "el adhesivo no fortalece el fallo ni ataca un punto decisorio que perjudique al adherente (art. 182 LA)",
    "innecesario_mayor_beneficio": "violación procesal que no se estudia porque una concesión de fondo da mayor beneficio (art. 189 LA)",
    "cae_con_principal": "queda sin materia porque depende de un principal que ya decide el asunto (nunca en una procesal)",
    "sin_materia": "lo que atacaba dejó de existir (nunca en una procesal)",
    "adhesivo_sin_materia": "el principal no prospera y el adhesivo se queda sin materia",
}
_DESCRIBE_TRAT = {
    "aplica": "se contesta con una premisa de su unidad, con su dato propio si lo trae",
    "remite": "ya contestado por la premisa de otra unidad expuesta antes; se remite a ella con la proposición que decide lo suyo",
    "desarrolla": "trae algo que ninguna premisa contesta (su diferencia): se construye sólo eso",
    "residual": "argumento menor: una o dos frases con su calificación y su razón",
    "no_se_estudia": "no se estudia, con su razón (cae_con_principal, sin_materia, innecesario_mayor_beneficio, adhesivo_sin_materia)",
    # PENÚLTIMO párrafo del art. 79 LA («En estos casos solo se expresará en
    # las sentencias cuando la suplencia derive de un beneficio»), verificado
    # contra el texto vigente el 26-sep-2026; el último es otro: la suplencia
    # por violaciones procesales o formales sólo opera sin vicio de fondo.
    "no_se_expresa_art79": "segmento suplido que no trae beneficio: no se expresa en la sentencia (art. 79, penúltimo párrafo); sólo para segmentos S",
}
# Qué es cada vicio (el orden del art. 189 depende de él): descripciones, sin
# frases que copiar.
_DESCRIBE_VICIO = {
    "procedencia": "ataca la procedencia del juicio o del recurso, o lo que la condiciona",
    "procesal": "violación a las leyes del procedimiento cometida durante el juicio (artículos 171 y 172 de la Ley de Amparo)",
    "forma": "vicio formal de la resolución misma —fundamentación, motivación, congruencia— sin discutir lo decidido",
    "omision": "la responsable dejó de pronunciarse sobre algo que se le planteó",
    "fondo": "discute lo decidido: los hechos, la valoración de las pruebas o la interpretación y aplicación de la norma",
}

# ═══ NORMALIZACIÓN ═════════════════════════════════════════════════════════

def _sin_acentos(t: str) -> str:
    t = unicodedata.normalize("NFKD", str(t or ""))
    return "".join(c for c in t if not unicodedata.combining(c))


def _ws(t) -> str:
    return re.sub(r"\s+", " ", str(t or "")).strip()


def norm_sentido(s) -> str:
    """La calificación como clave del catálogo: «Fundado pero insuficiente» →
    «fundado_insuficiente». Igual para lo que manda la pantalla y lo que
    devuelve el modelo, que es lo que permite la IGUALDAD de V0 (b)."""
    t = _sin_acentos(str(s or "")).strip().lower()
    t = re.sub(r"[\s\-]+", "_", t)
    alias = {"fundado_pero_insuficiente": "fundado_insuficiente",
             "fundados_pero_insuficientes": "fundado_insuficiente",
             "innecesarios": "innecesario", "sin_materia_": "sin_materia",
             "infundados": "infundado", "fundados": "fundado",
             "inoperantes": "inoperante", "ineficaces": "ineficaz"}
    return alias.get(t, t)


def clase_sentido(s) -> str:
    """prospera | no_prospera | no_se_estudia | «» (sin sentido)."""
    t = norm_sentido(s)
    if not t:
        return ""
    if t in _SENTIDOS_NO_SE_ESTUDIA:
        return NO_SE_ESTUDIA
    import tipos_asunto as _ta
    return PROSPERA if _ta.prospera(t) else NO_PROSPERA


# ═══ LA CALIFICACIÓN DE CADA ARGUMENTO DENTRO DEL SENTIDO DE SU PROBLEMA ═════
# (plan-4, 26-sep-2026). David: «que el proyecto mismo sea inteligente (se le
# denomina congruencia interna)». Hasta plan-3 la etiqueta de cada argumento
# era, por igualdad, el sentido de su problema; en el ADC 642/2024 v4 eso hizo
# «fundado» el argumento de la reconvención —la Sala SÍ la examinó— porque su
# problema era fundado, cuando la v3, sin plan, lo calificó bien como
# infundado dentro de un problema fundado. Dentro de un problema fundado caben
# argumentos infundados o inoperantes: el sentido del PROBLEMA es del
# secretario y no se toca; la calificación de cada argumento la razona el plan
# (y el estudio) y tiene que casar con él:
#   · problema que prospera   → al menos un argumento lo funda;
#   · problema que no prospera → ningún argumento queda fundado sin decir por
#     qué no alcanza: ése es el «fundado pero insuficiente»;
#   · problema que no se estudia → sus argumentos, tampoco (misma etiqueta);
#   · y dentro de un problema que se estudia, ningún argumento se declara sin
#     estudio (la misma regla que `exhaustivo` vigila en el texto).
def calificacion_conocida(s) -> bool:
    """¿Es una calificación del catálogo (o una de las de no estudiar)?"""
    t = norm_sentido(s)
    if not t:
        return False
    import tipos_asunto as _ta
    return t in _ta.CALIFICACIONES or t in _SENTIDOS_NO_SE_ESTUDIA \
        or t in ("parcialmente_fundado", "sustancialmente_fundado", "fundado_suplido")


def etiqueta_fuera(etiqueta, fijado) -> str:
    """Por qué la calificación de un argumento no cabe dentro del sentido
    `fijado` de su problema, o «» si cabe. Sin sentido fijado no hay nada que
    casar (eso es el pendiente «sentido»)."""
    et, fij = norm_sentido(etiqueta), norm_sentido(fijado)
    if not fij:
        return ""
    cf = clase_sentido(fij)
    if cf == NO_SE_ESTUDIA:
        return "" if et == fij else (
            f"distinta del sentido fijado «{fij}»: un problema que no se estudia no califica "
            f"sus argumentos; si crees que debe estudiarse, va a propuestas")
    if not et:
        return f"vacía dentro de un problema que se estudia («{fij}»)"
    if not calificacion_conocida(et):
        return "no es una calificación del catálogo"
    ce = clase_sentido(et)
    if ce == NO_SE_ESTUDIA:
        return (f"declara sin estudio un argumento de un problema que se estudia («{fij}»); "
                f"dentro de él cada argumento se califica en el fondo")
    if cf == NO_PROSPERA and ce == PROSPERA:
        return (f"prospera dentro de un problema que no prospera («{fij}»): el argumento que "
                f"tiene razón y no alcanza es fundado_insuficiente")
    return ""


def problemas_sin_quien_los_funde(segmentos: list, probs: list) -> list:
    """Los problemas que prosperan y en los que NINGÚN argumento con sentido
    lleva una calificación que prospera: la regla de coherencia al revés."""
    fuera = []
    for p in probs or []:
        if clase_sentido(p.get("sentido")) != PROSPERA:
            continue
        suyos = [s for s in segmentos or [] if s.get("problema_id") == p.get("id")
                 and s.get("pendiente") != "sentido" and s.get("trat") != "no_se_expresa_art79"]
        if suyos and not any(clase_sentido(s.get("etiqueta")) == PROSPERA for s in suyos):
            fuera.append(p["id"])
    return fuera


_RX_ID = re.compile(r"^\s*(AD|[CASP]|U|M)\s*(\d+)\s*(?:\.?\s*([a-z]{1,3}))?\s*$", re.I)


def norm_id(x) -> str:
    """«c1.A», «C1a», « C1.a » → «C1.a»; «ad2.b» → «AD2.b». Vacío si no es un
    identificador: un id mal escrito no puede casar con el inventario por
    accidente."""
    m = _RX_ID.match(str(x or ""))
    if not m:
        return ""
    pre, n, letra = m.group(1).upper(), m.group(2), (m.group(3) or "").lower()
    return f"{pre}{n}.{letra}" if letra else f"{pre}{n}"


def _pid(x) -> str | None:
    """«P2», «p2», «(P2)» → «P2»; None si no hay."""
    m = re.search(r"\bP\s*(\d+)\b", str(x or ""), re.I)
    return f"P{m.group(1)}" if m else None


def _enum(x, opciones, por_omision=None):
    t = _sin_acentos(str(x or "")).strip().lower().replace(" ", "_")
    return t if t in opciones else por_omision


def _razon(x) -> tuple[str, str | None]:
    """(razón del catálogo, proposición que nombra) — «no_combate(P2)» →
    («no_combate», «P2»). Razón desconocida = «» (V0 la rechaza)."""
    t = _sin_acentos(str(x or "")).strip().lower()
    p = _pid(t)
    base = re.sub(r"\(.*?\)", "", t).strip().replace(" ", "_")
    return (base if base in RAZONES else ""), p


# ═══ VERIFICACIÓN LITERAL (V0 g) ═══════════════════════════════════════════
# «Palabra por palabra»: se comparan SECUENCIAS DE PALABRAS, sin mayúsculas,
# acentos ni puntuación. Así una cita buena no se pierde por unas comillas o
# un salto de línea del PDF, y una cita inventada no pasa porque se parezca.

def _palabras(t: str) -> list[str]:
    t = str(t or "").replace("\u00ad", "")
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)      # guion de fin de línea
    t = _sin_acentos(t).lower()
    return re.findall(r"\w+", t)


def _renglones_repuestos(t: str) -> str:
    """El texto con los renglones de ancho fijo del PDF repuestos, con EL
    MISMO normalizador que usa el inventario para sacar sus citas
    (`inventario.normalizar_escrito`: «posesi⏎ones» → «posesiones»). Sin el
    inventario, el texto tal cual."""
    try:
        import inventario as _inv
        return _inv.normalizar_escrito(t)[0]
    except Exception:
        return t


class Texto:
    """Un texto listo para buscar citas en él muchas veces.

    DOS LECTURAS DEL MISMO TEXTO, y una cita vale si casa con cualquiera
    (revisión del 26-sep-2026). El escrito llega del PDF a renglón fijo, con
    palabras partidas sin guion al final del renglón. El inventario saca sus
    citas del escrito con los renglones repuestos; el planificador copia del
    texto crudo que ve en su prompt. Medido en las 8 sesiones del banco: con
    sólo la lectura cruda, V0 aceptaba 56 de las 101 citas que el inventario
    verifica (1 de 7 en el 640/2024), y el plan habría fallado por «citas sin
    verificar» que sí están en el escrito. Las dos lecturas son el escrito
    palabra por palabra; ninguna admite una palabra que no esté."""

    def __init__(self, *textos: str):
        crudos = [str(x or "") for x in textos]
        self.plano = " " + " ".join(_palabras("\n".join(crudos))) + " "
        rep = " " + " ".join(_palabras("\n".join(_renglones_repuestos(x) for x in crudos))) + " "
        self.planos = [self.plano] + ([rep] if rep != self.plano else [])

    def __bool__(self):
        return len(self.plano) > 2

    def contiene(self, cita: str, minimo: int = MIN_PALABRAS_CITA) -> bool:
        """¿La cita está, palabra por palabra, en alguna de las dos lecturas?
        Admite elisiones («…», «...»): cada tramo tiene que estar, en orden."""
        if not self or not str(cita or "").strip():
            return False
        tramos = [w for w in (_palabras(fr) for fr in
                              re.split(r"\.\.\.|…|\[\s*\.{3}\s*\]", str(cita))) if w]
        if not tramos or sum(len(w) for w in tramos) < minimo:
            return False
        for plano in self.planos:
            pos, bien = 0, True
            for w in tramos:
                aguja = " " + " ".join(w) + " "
                i = plano.find(aguja, pos)
                if i < 0:
                    bien = False
                    break
                pos = i + len(aguja) - 1
            if bien:
                return True
        return False


# ═══ ANCLAS DURAS (las del inventario, y un respaldo para la razón) ═════════

_RX_REG = re.compile(r"(?:registro(?:\s+digital)?|reg\.?)\s*(?:n[úu]m(?:ero)?\.?\s*)?:?\s*(\d{6,7})\b", re.I)
_RX_ART = re.compile(r"\bart[íi]culos?\s+(\d{1,4})(?:\s*(?:bis|ter))?", re.I)


def anclas_de_razon(texto: str) -> dict:
    """Registros y artículos que el secretario escribió en su razón (V0 h):
    tienen que figurar en las fuentes de la premisa de su unidad. Se busca
    aquí y no en `inventario.anclas_duras` porque ésta excluye los datos del
    propio asunto y aquí interesa todo lo que él citó."""
    t = str(texto or "")
    return {"registros": sorted(set(_RX_REG.findall(t))),
            "articulos": sorted(set(_RX_ART.findall(t)), key=lambda x: int(x))}


def tipo_de_ancla(a: str) -> str:
    """La diferencia que trae un ancla propia (V0 f): «art …» → norma;
    registro, tesis o expediente → precedente; cifra o fecha → hecho."""
    t = _sin_acentos(str(a or "")).lower().strip()
    if t.startswith(("art", "ley", "cod")):
        return "norma"
    if t.startswith(("reg", "tesis", "exp", "amparo", "juicio", "toca")):
        return "precedente"
    return "hecho"


def _anclas_seg(seg: dict) -> set:
    return {_ws(_sin_acentos(a)).lower() for a in (seg.get("anclas") or []) if str(a).strip()}


# ═══ LO QUE EL PLAN LEE DEL ASUNTO ═════════════════════════════════════════

class PlanNoDisponible(Exception):
    """Falta algo sin lo cual no hay plan (el inventario, el escrito…). El
    resolver escribe entonces sin plan (v3) y lo dice."""


def segmentos_de(fases, escrito: str = "", es_recurso: bool = False) -> list[dict]:
    """El inventario de la pieza INVENTARIO (`inventario.segmentos`, contrato
    del Paso 2). Sin él no hay plan: el plan se enlaza al piso por ID y no
    se inventa un inventario propio aquí —dos inventarios que discrepan son
    peor que ninguno—."""
    try:
        import inventario as _inv
    except ImportError as ex:
        raise PlanNoDisponible("falta el inventario de argumentos (inventario.py)") from ex
    segs = _inv.segmentos(fases, escrito or _escrito_de(fases), es_recurso) or []
    return [normalizar_segmento_piso(s) for s in segs if isinstance(s, dict)]


def normalizar_segmento_piso(s: dict) -> dict:
    return {"id": norm_id(s.get("id")), "concepto": _int(s.get("concepto")),
            "parrafo": _int(s.get("parrafo")), "texto": _ws(s.get("texto"))[:900],
            "cita": _ws(s.get("cita"))[:600],
            "anclas": [str(a) for a in (s.get("anclas") or []) if str(a).strip()][:20],
            # El número de concepto lo puso el inventario por el orden de
            # párrafos, no el escrito (`inventario.segmentos`): el guion no lo
            # escribe. No entra en la clave del plan (`huella_entradas`).
            "concepto_inferido": bool(s.get("concepto_inferido"))}


def _int(x, por_omision=0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return por_omision


def _fuentes(fases) -> list:
    return list(getattr(fases, "fuentes", None) or []) + ["", ""]


def _escrito_de(fases) -> str:
    return str(_fuentes(fases)[1] or "")


def _acto_de(fases) -> str:
    return str(_fuentes(fases)[0] or "")


def problemas_del_criterio(crit: list, fases) -> list[dict]:
    """Los problemas de la fase 3, cada uno con el criterio que el secretario
    le fijó. `id` es su número (1…n). El criterio se casa por la pregunta
    —normalizada, o la original si él la retocó con el lápiz— y, si no, por
    tema (`fase5_propuesta._mismo_tema`), con aviso; lo que no casa se queda
    sin criterio: mejor un «sin sentido» visible que un sentido ajeno."""
    probs = [p if isinstance(p, dict) else {"pregunta": str(p)}
             for p in (getattr(fases, "problemas", None) or [])]
    if not probs and getattr(fases, "problema_global", ""):
        probs = [{"pregunta": fases.problema_global, "jerarquia": "principal"}]
    try:
        import fases123_pipeline as _f123
        _cubre = _f123.cubre_de
    except Exception:                                   # pragma: no cover
        def _cubre(p):
            return sorted({_int(x) for x in (p.get("cubre") or []) if _int(x)})
    libres = list(crit or [])
    fuera = []
    for i, p in enumerate(probs, 1):
        preg = str(p.get("pregunta") or "")
        # Y POR LA CLAVE RECORTADA DEL ÁRBOL: el criterio armado recorta el
        # problema a 400 caracteres y la fase 3 lo trae entero; sin esto un
        # planteamiento largo sólo casaba «por tema», con aviso.
        claves = {_clave_texto(preg), _clave_texto(p.get("pregunta_original")),
                  _clave_texto(_clave_problema(preg)),
                  _clave_texto(_clave_problema(p.get("pregunta_original")))} - {""}
        elegido, por_tema = None, False
        for c in libres:
            if _clave_texto(getattr(c, "problema", "")) in claves:
                elegido = c
                break
        if elegido is None:
            try:
                import fase5_propuesta as _f5
                for c in libres:
                    if _f5._mismo_tema(str(getattr(c, "problema", "")), preg):
                        elegido, por_tema = c, True
                        break
            except Exception:
                pass
        if elegido is not None:
            libres.remove(elegido)
        fuera.append({
            "id": i, "pregunta": preg,
            "clase": str(p.get("clase") or "fondo").strip().lower(),
            "jerarquia": str(getattr(elegido, "jerarquia", "") or p.get("jerarquia")
                             or ("principal" if i == 1 else "accesorio")).strip().lower(),
            "depende_de": _int(p.get("depende_de"), 0) or None,
            "cubre": _cubre(p),
            "sentido": norm_sentido(getattr(elegido, "sentido", "")) if elegido else "",
            "razon": str(getattr(elegido, "razonamiento", "") or "") if elegido else "",
            "grupo": str(getattr(elegido, "grupo", "") or "").strip() if elegido else "",
            "por_tema": por_tema,
            # SIN CALIFICAR TRAS EL CAMBIO DE SENTIDO (26-sep-2026): el
            # criterio está, pero vacío —el árbol lo tumbó y la recalificación
            # no llegó—. No es uno que el secretario dejó sin decidir: se
            # desarrolla con el material y va primero en ADVERTENCIAS.
            "sin_calificar": bool(elegido is not None
                                  and not norm_sentido(getattr(elegido, "sentido", ""))),
        })
    # Un criterio que no casó con ningún problema (el camino de un solo
    # sentido, o una pregunta reescrita del todo) se añade como problema
    # propio: su sentido no puede perderse por no encontrar pareja.
    for c in libres:
        fuera.append({"id": len(fuera) + 1, "pregunta": str(getattr(c, "problema", "")),
                      "clase": "fondo", "jerarquia": str(getattr(c, "jerarquia", "") or "accesorio"),
                      "depende_de": None, "cubre": [],
                      "sentido": norm_sentido(getattr(c, "sentido", "")),
                      "razon": str(getattr(c, "razonamiento", "") or ""),
                      "grupo": str(getattr(c, "grupo", "") or "").strip(),
                      "por_tema": False, "sin_pareja": True})
    return fuera


def _clave_texto(x) -> str:
    return re.sub(r"[^a-z0-9 ]", "", _sin_acentos(_ws(x)).lower()).strip()


def _clave_problema(t) -> str:
    """La clave recortada con que el árbol compara problemas (un solo sitio:
    `arbol_decision.clave_problema`)."""
    try:
        import arbol_decision as _ad
        return _ad.clave_problema(t)
    except Exception:                                   # pragma: no cover
        return str(t or "")[:400]


def n_planteamientos(fases) -> int:
    c = getattr(fases, "conteo", None) or {}
    return _int(c.get("n")) if str(c.get("estado")) == "contado" else 0


def problemas_permitidos(seg: dict, probs: list[dict], n: int = 2) -> list[int]:
    """V0 (c): el problema de un segmento lo fija el CÓDIGO, no el modelo: el
    que en su «cubre» incluye el concepto del segmento. Con n<2 el «concepto»
    del inventario es el orden del párrafo y no un planteamiento, y el «cubre»
    que quede es el de la plantilla vieja (L11: 13 sesiones de concepto único
    lo traían lleno): vale cualquiera. Un concepto huérfano —ningún «cubre» lo
    trae— también."""
    con_cubre = [p for p in probs if p.get("cubre")]
    todos = [p["id"] for p in probs]
    if not con_cubre or n < 2:
        return todos
    k = _int(seg.get("concepto"))
    hits = [p["id"] for p in probs if k and k in (p.get("cubre") or [])]
    return hits or todos


def indice_material(material, fases=None) -> dict:
    """El índice del material CON EL MISMO RECORTE QUE VERÁ EL ESTUDIO
    (`fase6_estudio._bloque_material` tras `_litis_y_material`): las diez
    primeras tesis que no son de la técnica más las de la técnica, y los doce
    primeros preceptos que sobreviven al filtro de la litis. Las tesis de
    MÉTODO se quitan: sólo sirven al peldaño del diálogo constitucional y no
    pueden ser fuente de una premisa. V0 (h) exige que toda fuente esté aquí."""
    import fase6_estudio as _f6
    tesis_m = list(getattr(material, "tesis", None) or [])
    _tec = [t for t in tesis_m if isinstance(t, dict) and t.get("tecnica")]
    _sel = [t for t in tesis_m if isinstance(t, dict) and not t.get("tecnica")][:_f6.MAX_TESIS_PROMPT]
    tesis = [t for t in _sel + _tec if not t.get("metodo")]
    normas_m = [n for n in (getattr(material, "normas", None) or []) if isinstance(n, dict)]
    if fases is not None:
        try:
            import litis_normativa as _ln
            normas_m, _ = _ln.filtrar_normas(normas_m, _ln.leyes_de_la_litis(fases))
        except Exception:
            pass
    normas = normas_m[:_f6.MAX_NORMAS_PROMPT]
    return {
        "tesis": [{"id": f"T{i}", "registro": str(t.get("registro") or "").strip(),
                   "rubro": _ws(t.get("rubro"))[:400],
                   "obligatoria": bool(t.get("obligatoria")),
                   "texto": str(t.get("texto") or "")[:4000]}
                  for i, t in enumerate(tesis, 1)],
        "normas": [{"id": f"N{j}", "cuerpo_legal": _ws(n.get("cuerpo_legal"))[:200],
                    "articulo": _ws(n.get("articulo"))[:20],
                    "texto": str(n.get("texto") or "")[:4000]}
                   for j, n in enumerate(normas, 1)],
    }


def huella_indice(indice: dict) -> str:
    base = [[t["registro"] for t in indice.get("tesis") or []],
            [f"{n['cuerpo_legal']}|{n['articulo']}" for n in indice.get("normas") or []]]
    return hashlib.sha1(json.dumps(base, ensure_ascii=False).encode()).hexdigest()[:12]


def huella_entradas(r, material, conceptos_violacion: str = "", segs: list = None) -> str:
    """La parte de la clave que no es el criterio (w2_final §3.2): el adelanto
    —problemas y resúmenes, la misma huella del contraste—, el escrito, el
    índice del material que verá el estudio, los conceptos de violación del
    recurso que reasume y el inventario. Si cualquiera cambia, el plan viejo
    no sirve."""
    import taller_estado as _te
    fases = getattr(r, "fases", None)
    e = getattr(r, "encargo", None)
    base = {
        "adelanto": _te.huella_contraste(r),
        "escrito": hashlib.sha1(_escrito_de(fases).encode("utf-8", "ignore")).hexdigest()[:12],
        "indice": huella_indice(indice_material(material, fases)),
        "cv": hashlib.sha1(_ws(conceptos_violacion).encode()).hexdigest()[:12],
        "tipo": str(getattr(e, "tipo_asunto", "") or ""),
        "segs": [[s.get("id"), s.get("concepto"), sorted(s.get("anclas") or [])]
                 for s in (segs or [])],
    }
    return hashlib.sha1(json.dumps(base, ensure_ascii=False, sort_keys=True)
                        .encode()).hexdigest()[:16]


def suplencia_de_clave(suplencia) -> dict:
    """Sólo la CONFIRMADA cambia el plan (V0 k). La propuesta sin confirmar no
    entra en la clave: si no, cada vez que la pantalla la enseña cambiaría."""
    try:
        import suplencia as _sp
        if _sp.confirmada(suplencia):
            return {"fraccion": suplencia.get("fraccion"),
                    "a_favor_de": _ws(suplencia.get("a_favor_de"))[:200]}
    except Exception:
        pass
    return {}


def clave(crit, fases_huella: str, contexto: str, suplencia, formato: str = "",
          variante: str = "v4") -> str:
    """LA CLAVE DE UN PLAN. Lleva la RAZÓN literal del secretario: usar un plan
    hecho con otra razón es el riesgo jurídico mayor (w2_final §3.7, la
    objeción que se descartó), y el coste de invalidar se controla con el
    antirrebote de la pantalla y el tope de corridas.

    `formato` se acepta por el contrato pero NO entra: el plan es el mismo en
    la estándar y en la moderna —sólo cambia el renderizado (`vista`)—, y
    meterlo obligaría a planear otra vez al cambiar de forma (§4.6, §5.5)."""
    base = {
        "v": PLAN_VERSION, "var": str(variante or ""),
        "fases": str(fases_huella or ""),
        "crit": [{"problema": _clave_texto(getattr(c, "problema", "")),
                  "sentido": norm_sentido(getattr(c, "sentido", "")),
                  "razon": _ws(getattr(c, "razonamiento", "")),
                  "jerarquia": str(getattr(c, "jerarquia", "") or "").strip().lower(),
                  "grupo": str(getattr(c, "grupo", "") or "").strip()}
                 for c in (crit or [])],
        "contexto": hashlib.sha1(_ws(contexto).encode()).hexdigest()[:12],
        "suplencia": suplencia_de_clave(suplencia),
    }
    return hashlib.sha1(json.dumps(base, ensure_ascii=False, sort_keys=True)
                        .encode()).hexdigest()[:20]


# ═══ EL PROMPT DEL PLANIFICADOR ═════════════════════════════════════════════

def _tramos_del_escrito(fases, tope_total: int = TRAMOS_TOPE_TOTAL) -> list[tuple[str, str]]:
    """El escrito literal, por concepto. Con la cadena del contador
    (`conteo.tramos`, offsets sobre `fuentes[1]`) se corta por concepto; sin
    ella, el escrito entero. Cada tramo con su tope, cabeza y cola, para que
    nunca se pierda el final."""
    esc = _escrito_de(fases)
    if not esc.strip():
        return []
    conteo = getattr(fases, "conteo", None) or {}
    tramos = []
    if str(conteo.get("estado")) == "contado" and conteo.get("tramos"):
        for k, t in enumerate(conteo.get("tramos") or [], 1):
            try:
                a, b = int(t[0]), int(t[1])
            except Exception:
                continue
            tramos.append((str(k), esc[a:b]))
    if not tramos:
        tramos = [("escrito", esc)]
    cupo = max(TRAMO_TOPE_MIN, tope_total // max(1, len(tramos)))
    fuera = []
    for k, t in tramos:
        t = t.strip()
        if len(t) > cupo:
            cab = int(cupo * 0.6)
            t = t[:cab] + "\n[…]\n" + t[-(cupo - cab):]
        fuera.append((k, t))
    return fuera


def _bloque_problemas(probs: list[dict]) -> str:
    lineas = []
    for p in probs:
        s = p["sentido"] or "SIN SENTIDO FIJADO"
        lineas.append(
            f"PROBLEMA {p['id']} · clase: {p['clase']} · jerarquía: {p['jerarquia']}"
            + (f" · depende de: {p['depende_de']}" if p.get("depende_de") else "")
            + (f" · cubre conceptos: {', '.join(str(x) for x in p['cubre'])}" if p.get("cubre") else "")
            + (f" · grupo del secretario: {p['grupo']}" if p.get("grupo") else "")
            + f"\n  pregunta: {p['pregunta']}"
            + f"\n  sentido fijado por el secretario: {s}"
            + f"\n  razón del secretario (literal): {p['razon'] or '—'}")
    return "\n".join(lineas)


def _bloque_segmentos(segs: list[dict], probs: list[dict], n: int = 2) -> str:
    lineas = []
    for s in segs:
        perm = problemas_permitidos(s, probs, n)
        lineas.append(
            f"{s['id']} · concepto {s['concepto'] or '—'}"
            # Dato: ese número es el orden del párrafo, no el del escrito.
            + (" (orden del párrafo: el resumen no numera los conceptos)"
               if s.get("concepto_inferido") else "")
            + " · problema(s) posible(s): "
            f"{', '.join(str(x) for x in perm)}"
            + (f" · anclas: {' | '.join(s['anclas'])}" if s.get("anclas") else " · anclas: —")
            + f"\n  resumen: {s['texto']}"
            + (f"\n  cita literal del escrito: «{s['cita']}»" if s.get("cita") else
               "\n  cita literal del escrito: (no encontrada: pon tú 10 a 40 palabras literales del tramo en «cita»)"))
    return "\n".join(lineas)


def _bloque_indice(ind: dict) -> str:
    p = []
    for t in ind.get("tesis") or []:
        p.append(f"{t['id']} · registro {t['registro']} · "
                 f"{'obligatoria' if t['obligatoria'] else 'orientadora'} · {t['rubro']}\n"
                 f"   {_ws(t['texto'])[:500]}")
    for n in ind.get("normas") or []:
        p.append(f"{n['id']} · {n['cuerpo_legal']} — art. {n['articulo']}\n"
                 f"   {_ws(n['texto'])[:400]}")
    return "\n".join(p) or "(el material no trae tesis ni preceptos)"


def prompt_plan(*, tipo_asunto: str, probs: list[dict], segs: list[dict],
                resumen_acto: str, tramos: list, indice: dict, contexto: str = "",
                conceptos_violacion: str = "", contraste=None, checklist=None,
                suplencia=None, faltas: list = None, n: int = 2) -> str:
    """El prompt del planificador: DESCRIPCIONES de lo que decide, el esquema
    con TIPOS y los DATOS del asunto. Ni una frase para copiar."""
    import tipos_asunto as _ta
    voc = _ta.vocabulario_de(tipo_asunto or "amparo_directo")
    q1 = voc["combate_singular"]
    organo = _ta.sujetos_de(tipo_asunto or "amparo_directo")["organo"][0]
    _ad = _ta.normalizar(tipo_asunto) == "amparo_directo"
    # LA REGLA PROCESAL DEL AMPARO DIRECTO (arts. 74-V, 174 y 189) sólo donde
    # rige, como en el árbol (`violacion_procesal.guarda_aplica`): en un recurso
    # el planificador la aplicaba y el guion citaba el 189 (revisión
    # adversarial de la integración, 26-sep-2026).
    _regla_proc = (
        "  · Las VIOLACIONES PROCESALES se deciden todas (artículos 74, fracción V, y 174 de la "
        "Ley de Amparo): un segmento de vicio procesal nunca va a cae_con_principal, sin_materia ni "
        "no_se_estudia; la única que puede quedar sin estudio es la de innecesario_mayor_beneficio, "
        "cuando una concesión de fondo da mayor beneficio (artículo 189).\n"
        if _guarda_procesal(tipo_asunto) else "")
    sup = suplencia if isinstance(suplencia, dict) else {}
    try:
        import suplencia as _sp
        sup_conf = _sp.confirmada(sup)
    except Exception:
        sup_conf = False
    razones = "\n".join(f"  {k}: {v} → implica {RAZONES[k]['clase'].replace('_', ' ')}"
                        for k, v in _DESCRIBE_RAZON.items())
    trats = "\n".join(f"  {k}: {v}" for k, v in _DESCRIBE_TRAT.items())
    vicios = "\n".join(f"  {k}: {v}" for k, v in _DESCRIBE_VICIO.items())
    escrito = "\n\n".join(
        (f"── {q1} {k} ──\n{t}" if k != "escrito" else t) for k, t in tramos
    ) or "(no está el escrito)"
    _faltas = ""
    if faltas:
        _faltas = ("\nLO QUE EL CÓDIGO RECHAZÓ DE TU PLAN ANTERIOR (corrígelo; lo demás "
                   "puede quedarse):\n" + "\n".join(f"  - {f}" for f in faltas[:40]) + "\n")
    _sup = "no hay suplencia confirmada"
    if sup_conf:
        _sup = (f"CONFIRMADA por el secretario: fracción {sup.get('fraccion')} del artículo 79 "
                f"de la Ley de Amparo, a favor de {sup.get('a_favor_de') or 'la parte promovente'}")
    elif sup.get("fraccion"):
        _sup = f"propuesta sin confirmar (fracción {sup.get('fraccion')}): no cambia nada"
    _contraste = ""
    if contraste:
        _contraste = ("\nEL CONTRASTE DEL MOTOR (razón toral de cada planteamiento y si el "
                      f"{q1} la combate; es lectura, no decisión):\n"
                      + json.dumps(contraste, ensure_ascii=False)[:6000] + "\n")
    _check = ""
    if checklist:
        _check = ("\nLA LISTA DE COMPROBACIÓN DEL MOTOR (temas que la respuesta debe cubrir):\n"
                  + json.dumps(checklist, ensure_ascii=False)[:4000] + "\n")
    _cv = (f"\nLOS CONCEPTOS DE VIOLACIÓN QUE SE ESTUDIAN AL REASUMIR JURISDICCIÓN:\n"
           f"{str(conceptos_violacion)[:12000]}\n") if str(conceptos_violacion or "").strip() else ""
    _ctx = (f"\nCONTEXTO QUE APORTÓ EL SECRETARIO (autos y constancias; de aquí salen, con su "
            f"cita, las razones R1…Rn de una resolución procesal):\n{str(contexto)[:12000]}\n"
            if str(contexto or "").strip() else "")
    _orden_ad = (
        "  · En el amparo directo, el fondo antes que el procedimiento y la forma (artículo 189\n"
        "    de la Ley de Amparo); el orden sólo se invierte si estudiar primero una violación\n"
        "    procesal o formal da mayor beneficio, y entonces orden.por_que dice en qué consiste\n"
        "    ese beneficio.\n") if _ad else (
        "  · En un recurso, el orden que fije su técnica.\n")

    return f"""Organizas el estudio de fondo de un proyecto de resolución de un Tribunal Colegiado de Circuito ({voc['nombre']}). NO decides el sentido: el secretario ya lo fijó para cada problema y está abajo. NO redactas: la prosa la escribe otro paso con tu plan como guion. Decides, con datos, cómo se organiza la respuesta a cada argumento, para que ninguno quede sin respuesta propia y ninguna premisa se exponga dos veces.

QUÉ DECIDES
1. PROPOSICIONES DEL ACTO (P1…Pn): cada consideración de {organo} que sostiene lo resuelto y que algún segmento ataca, o que sostiene el fallo por sí sola. Con su carácter, su relación con las demás, su fuente y una cita literal que la muestre, tomada de lo que resolvió {organo} o del contexto.
2. CADA SEGMENTO DEL INVENTARIO, TODOS y con su identificador tal cual: la proposición que ataca, el vicio que alega, si sólo reitera otro segmento, el dato propio que trae, la razón tipificada que lo decide, cómo se trata y qué diferencia obliga a desarrollarlo.
3. PREMISAS (M1…Mn): la regla que decide una o varias proposiciones, dicha SÓLO por sus fuentes (identificadores T/N del índice, o registros y artículos que el secretario escribió en su razón) y de dos a cuatro anclas (términos sueltos). NO escribas la regla. Cada premisa lleva su rastro: de dónde sale —la razón del secretario o el material— y una cita literal de ese rastro.
4. UNIDADES (U1…Un): los segmentos que atacan la MISMA proposición con la MISMA razón y el MISMO vicio. El tema común no basta: dos segmentos que hablan de lo mismo pero atacan proposiciones distintas, o caen por razones distintas, o uno alega omisión y otro fondo, van en unidades distintas. Si el secretario marcó un grupo, los problemas de ese grupo se estudian juntos.
5. EL ORDEN DEL ESTUDIO: el orden de la lista de unidades.

REGLAS QUE EL CÓDIGO COMPRUEBA (si no se cumplen, tu plan se rechaza)
  · El SENTIDO DE CADA PROBLEMA lo fijó el secretario y no se toca. etiqueta = la calificación de ESE segmento dentro de su problema, por lo que él mismo plantea, y coherente con el sentido del problema: si el problema prospera, al menos uno de sus segmentos lo funda y los demás pueden ser infundados o inoperantes; si no prospera, ninguno queda fundado —el que tiene razón y no alcanza es fundado_insuficiente—; si el problema no se estudia (innecesario, sin materia), todos sus segmentos llevan ese mismo sentido. Dentro de un problema que se estudia, ningún segmento se declara sin estudio.
  · propuestas: SÓLO cuando crees que el sentido de un PROBLEMA debería ser otro. seg = uno de sus segmentos; a = la calificación que propones para el problema entero; por_que = la razón. La calificación distinta de un segmento dentro de su problema no es una propuesta: va en su etiqueta.
  · problema_id: uno de los problemas posibles que el inventario indica para ese segmento.
  · Todo segmento que se contesta (aplica, remite, desarrolla) está en UNA unidad, y sólo en una; los residuales y los que no se estudian pueden quedar fuera de las unidades.
  · razon: del catálogo, compatible con la etiqueta (la tabla dice qué implica cada una).
  · reitera: sólo si TODAS las anclas del segmento están entre las del segmento reiterado. Un segmento con un ancla propia (un artículo, un registro, un expediente, una cifra o una fecha que el otro no trae) no reitera: trat «desarrolla», con su diferencia.
  · Las citas —de dato, de proposición y de rastro— son COPIA LITERAL, palabra por palabra, del texto que dice su fuente. Lo que el código no encuentre se borra.
{_regla_proc}  · Suplencia: {_sup}. Con suplencia confirmada, ningún segmento de la parte favorecida se declara inoperante por cómo se planteó (no_combate, ataca_accesoria, generico, reitera_sin_combatir).
  · cosa_juzgada_amparo_previo sólo contra una proposición vinculada por una ejecutoria de amparo que conste en lo que resolvió {organo}, en el contexto o en el material.
  · Un segmento cuyo problema tiene sentido pero cuya razón del secretario NO contesta lo propio del segmento (su dato, su precepto, su precedente): pendiente «razon» y trat «desarrolla». No inventes esa razón.
  · Un segmento cuyo problema no tiene sentido fijado: pendiente «sentido» y etiqueta vacía.
  · Orden. orden.criterio «promovente» = los apartados siguen el orden del escrito; «prelacion» = siguen el orden de tu lista de unidades. Con cualquiera de los dos, el orden que resulta cumple esto:
  · La procedencia primero, y cada accesorio después de su principal.
{_orden_ad}
CATÁLOGOS
razon (nombre: qué significa → qué implica):
{razones}
trat:
{trats}
vicio (la clase de violación que alega el segmento):
{vicios}
diferencia (lo que obliga a desarrollar): hecho | prueba | norma | precedente | procesal | consecuencia
carácter de una proposición: toral | accesoria | obiter | consecuencia
relación de una proposición: necesaria | suficiente | dependiente_de:Pk
fuente de una proposición: reclamada | recurrida | resolucion_procesal(Ri) | ejecutoria
fuente de un dato: escrito | reclamada | constancia | antecedentes

ESQUEMA DE LA RESPUESTA — un objeto JSON y nada más; los valores son los tipos:
{{
 "proposiciones": [{{"id": "P<n>", "dice": "<30 palabras como máximo>", "caracter": "<carácter>", "relacion": "<relación>", "fuente": "<fuente>", "cita": "<literal>", "vinculada_por_ejecutoria": <true|false>}}],
 "segmentos": [{{"id": "<id del inventario>", "problema_id": <número>, "vicio": "<vicio>", "ataca": "P<n>" | null, "reitera": "<id>" | null, "dato": {{"texto": "<qué dato>", "cita": "<literal, 40 palabras como máximo>", "fuente": "<fuente de un dato>"}} | null, "etiqueta": "<la calificación de este segmento>", "razon": "<razon>", "trat": "<trat>", "diferencia": "<diferencia>" | null, "pendiente": null | "sentido" | "razon", "cita": "<sólo si el inventario no trae cita: 10 a 40 palabras literales del escrito>", "sostiene": "<25 palabras como máximo, para la pantalla>"}}],
 "premisas": [{{"id": "M<n>", "responde_a": ["P<n>"], "fuentes": {{"tesis": ["T<n>" | "<registro>"], "normas": ["N<n>" | "<artículo y ley>"]}}, "anclas": ["<término>"], "rastro": "razon" | "material", "rastro_cita": "<literal de la razón o del texto de la fuente>"}}],
 "unidades": [{{"id": "U<n>", "problemas": [<número>], "segmentos": ["<id>"], "premisa": "M<n>" | null, "objecion": {{"de": "<quién la plantea>", "anclas": ["<término>"]}} | null}}],
 "propuestas": [{{"seg": "<id>", "de": "<el sentido fijado del problema>", "a": "<la calificación que propones para el problema>", "por_que": "<razón breve>"}}],
 "avisos_al_secretario": ["<texto breve>"],
 "orden": {{"criterio": "promovente" | "prelacion", "por_que": "<razón breve>"}}
}}
{_faltas}
═══ DATOS DEL ASUNTO ═══

LOS PROBLEMAS Y EL CRITERIO DEL SECRETARIO:
{_bloque_problemas(probs)}

SUPLENCIA DE LA QUEJA: {_sup}.

EL INVENTARIO DE SEGMENTOS (cada argumento del escrito, con su identificador):
{_bloque_segmentos(segs, probs, n)}

LO QUE RESOLVIÓ {organo.upper()} (resumen):
{str(resumen_acto or '')[:12000]}

EL ESCRITO, LITERAL, POR {q1.upper()}:
{escrito}
{_contraste}{_check}{_cv}{_ctx}
ÍNDICE DEL MATERIAL (lo único citable como fuente, además de lo que el secretario escribió en su razón):
{_bloque_indice(indice)}
"""


# ═══ LA LLAMADA ═════════════════════════════════════════════════════════════

def _modelo() -> str:
    m = os.getenv("MODELO_PLAN", "").strip()
    if m:
        return m
    import fases123_pipeline as _f123
    return _f123.MODELO_FASES


async def _llamar(cliente, prompt: str) -> str:
    """Una llamada con JSON estricto, por `llamada_modelo.crear` (el respaldo
    ante la cola de latencia y la retirada de parámetros de muestreo). Si
    vuelve vacía o cortada, otra vez con el doble de cupo: el razonamiento se
    MANTIENE —planear sin pensar es justo lo que no se quiere—."""
    import llamada_modelo as _lm
    kw = dict(model=_modelo(), messages=[{"role": "user", "content": prompt}],
              max_completion_tokens=TOPE_SALIDA,
              response_format={"type": "json_object"})
    if ESFUERZO_PLAN:
        kw["reasoning_effort"] = ESFUERZO_PLAN
    r = await _lm.crear(cliente, **kw)
    txt = (r.choices[0].message.content or "").strip()
    fin = str(getattr(r.choices[0], "finish_reason", "") or "").lower()
    if not txt or fin == "length":
        kw["max_completion_tokens"] = TOPE_SALIDA * 2
        r = await _lm.crear(cliente, **kw)
        txt = (r.choices[0].message.content or "").strip()
    return txt


def _json_de(txt: str) -> dict:
    try:
        d = json.loads(txt)
        return d if isinstance(d, dict) else {}
    except Exception:
        m = re.search(r"\{.*\}", str(txt or ""), re.S)
        if m:
            try:
                d = json.loads(m.group(0))
                return d if isinstance(d, dict) else {}
            except Exception:
                return {}
    return {}


def _palabras_max(t, n: int) -> str:
    """Recorta a `n` palabras. Las anclas son TÉRMINOS y la proposición cabe en
    treinta palabras: un campo largo del planificador sería una regla redactada
    colándose al guion, que es justo lo que el plan no hace."""
    w = _ws(t).split(" ")
    return " ".join(w[:n]) if len(w) > n else _ws(t)


def normalizar(crudo: dict) -> dict:
    """Lo que devolvió el modelo, con los tipos del esquema. No corrige nada
    de fondo —eso es `reparar`— sólo la forma: ids, enumeraciones, listas y
    topes de palabras."""
    d = crudo if isinstance(crudo, dict) else {}

    def _lista(x):
        return [y for y in (x if isinstance(x, list) else []) if isinstance(y, dict)]

    segs = []
    for s in _lista(d.get("segmentos")):
        rz, rp = _razon(s.get("razon"))
        dato = s.get("dato") if isinstance(s.get("dato"), dict) else None
        if dato is not None:
            dato = {"texto": _ws(dato.get("texto"))[:300], "cita": _ws(dato.get("cita"))[:500],
                    "fuente": _enum(dato.get("fuente"), FUENTES_DATO, "escrito")}
            if not dato["cita"] and not dato["texto"]:
                dato = None
        pend = _enum(s.get("pendiente"), ("sentido", "razon"))
        segs.append({
            "id": norm_id(s.get("id")),
            "problema_id": _int(s.get("problema_id")) or None,
            "vicio": _enum(s.get("vicio"), VICIOS, "fondo"),
            "ataca": _pid(s.get("ataca")),
            "reitera": norm_id(s.get("reitera")) or None,
            "dato": dato,
            "etiqueta": norm_sentido(s.get("etiqueta")),
            "razon": rz, "razon_crudo": _ws(s.get("razon"))[:60],
            "razon_p": rp,
            "trat": _enum(s.get("trat"), TRATAMIENTOS, ""),
            "diferencia": _enum(s.get("diferencia"), DIFERENCIAS),
            "pendiente": pend,
            "cita_plan": _ws(s.get("cita"))[:600],
            "sostiene": _palabras_max(s.get("sostiene"), 25)[:220],
        })
    props = []
    for p in _lista(d.get("proposiciones")):
        rel = _sin_acentos(str(p.get("relacion") or "")).strip().lower()
        dep = _pid(rel) if rel.startswith("dependiente") else None
        props.append({
            "id": _pid(p.get("id")) or "",
            "dice": _palabras_max(p.get("dice"), 30)[:300],
            "caracter": _enum(p.get("caracter"), CARACTERES, "toral"),
            "relacion": (f"dependiente_de:{dep}" if dep else
                         _enum(rel, ("necesaria", "suficiente"), "necesaria")),
            "fuente": _ws(p.get("fuente"))[:60],
            "cita": _ws(p.get("cita"))[:600],
            "vinculada_por_ejecutoria": bool(p.get("vinculada_por_ejecutoria") is True
                                             or str(p.get("vinculada_por_ejecutoria")).lower() in ("true", "1", "si", "sí")),
        })
    prem = []
    for m in _lista(d.get("premisas")):
        f = m.get("fuentes") if isinstance(m.get("fuentes"), dict) else {}
        prem.append({
            "id": norm_id(m.get("id")),
            "responde_a": [x for x in (_pid(y) for y in (m.get("responde_a") or [])) if x],
            "fuentes": {"tesis": [_ws(x)[:120] for x in (f.get("tesis") or []) if _ws(x)][:4],
                        "normas": [_ws(x)[:160] for x in (f.get("normas") or []) if _ws(x)][:6]},
            "anclas": [_palabras_max(x, 6)[:60] for x in (m.get("anclas") or []) if _ws(x)][:6],
            "rastro": _enum(m.get("rastro"), ("razon", "material"), ""),
            "rastro_cita": _ws(m.get("rastro_cita"))[:600],
        })
    unis = []
    for u in _lista(d.get("unidades")):
        ob = u.get("objecion") if isinstance(u.get("objecion"), dict) else None
        unis.append({
            "id": norm_id(u.get("id")),
            "problemas": [_int(x) for x in (u.get("problemas") or []) if _int(x)],
            "segmentos": [x for x in (norm_id(y) for y in (u.get("segmentos") or [])) if x],
            "premisa": norm_id(u.get("premisa")) or None,
            "objecion": ({"de": _ws(ob.get("de"))[:120],
                          "anclas": [_palabras_max(x, 6)[:60] for x in (ob.get("anclas") or []) if _ws(x)][:6]}
                         if ob else None),
        })
    props_ = [{"seg": norm_id(x.get("seg")), "de": norm_sentido(x.get("de")),
               "a": norm_sentido(x.get("a")), "por_que": _ws(x.get("por_que"))[:400]}
              for x in _lista(d.get("propuestas"))]
    orden = d.get("orden") if isinstance(d.get("orden"), dict) else {}
    return {
        "segmentos": segs, "proposiciones": props, "premisas": prem, "unidades": unis,
        "propuestas": [x for x in props_ if x["seg"]],
        "avisos_al_secretario": [_ws(x)[:400] for x in (d.get("avisos_al_secretario") or [])
                                 if isinstance(x, str) and _ws(x)][:12],
        "orden": {"criterio": _enum(orden.get("criterio"), ("promovente", "prelacion"), "promovente"),
                  "por_que": _ws(orden.get("por_que"))[:400]},
    }


async def planear(cliente, r, crit, material, contexto: str = "", suplencia=None, *,
                  segs: list = None, contraste=None, faltas: list = None,
                  conceptos_violacion: str = "", checklist=None) -> dict:
    """UNA llamada al planificador. Devuelve el plan con los tipos del esquema,
    sin reparar ni validar (eso es `preparar`). Nunca redacta nada que llegue
    al documento: su salida sólo se lee como datos."""
    fases = r.fases
    e = getattr(r, "encargo", None)
    es_recurso = bool(getattr(e, "es_recurso", False))
    if segs is None:
        segs = segmentos_de(fases, _escrito_de(fases), es_recurso)
    tipo = str(getattr(e, "tipo_asunto", "") or "amparo_directo")
    probs = problemas_del_criterio(crit, fases)
    glob = getattr(e, "propuesta_global", None) or {}
    prompt = prompt_plan(
        tipo_asunto=tipo, probs=probs, segs=segs,
        resumen_acto=getattr(fases, "resumen_acto", ""),
        tramos=_tramos_del_escrito(fases), indice=indice_material(material, fases),
        contexto=contexto, conceptos_violacion=conceptos_violacion,
        contraste=contraste,
        checklist=(checklist if checklist is not None else
                   (glob.get("checklist") if isinstance(glob, dict) else None)),
        suplencia=suplencia, faltas=faltas, n=n_planteamientos(fases))
    txt = await _llamar(cliente, prompt)
    plan = normalizar(_json_de(txt))
    plan.update({"version": PLAN_VERSION, "tipo_asunto": tipo,
                 "suplencia": dict(suplencia) if isinstance(suplencia, dict) else {}})
    return plan


# ═══ REPARAR: LO QUE EL CÓDIGO CORRIGE SOLO ═════════════════════════════════

class _Ctx:
    """Lo que V0 necesita del asunto, calculado una vez."""

    def __init__(self, crit, segs, fases, material, contexto, suplencia=None, tocados=None):
        self.probs = problemas_del_criterio(crit, fases)
        self.n = n_planteamientos(fases)
        self.por_pid = {p["id"]: p for p in self.probs}
        self.segs = [s for s in (segs or []) if s.get("id")]
        self.seg_piso = {s["id"]: s for s in self.segs}
        self.escrito = Texto(_escrito_de(fases), "")
        self.fuentes_dato = Texto(_escrito_de(fases), _acto_de(fases),
                                  getattr(fases, "antecedentes", "") or "",
                                  getattr(fases, "autos", "") or "", contexto or "")
        self.fuentes_p = Texto(_acto_de(fases), contexto or "",
                               getattr(fases, "autos", "") or "",
                               getattr(fases, "resumen_acto", "") or "")
        self.indice = indice_material(material, fases)
        self.razones = {p["id"]: Texto(p["razon"]) for p in self.probs}
        # Por la clave recortada del árbol: los tocados llegan enteros.
        self.tocados = {_clave_texto(_clave_problema(t)) for t in (tocados or [])}
        self.suplencia = suplencia if isinstance(suplencia, dict) else {}
        self.hay_escrito = bool(self.escrito)
        # LA ÚNICA EXCEPCIÓN DEL ART. 189 es una concesión DE FONDO (Decisión
        # 1 de David). Un problema de procedencia que prospera no lo es: antes
        # contaba (`!= "procesal"`) y dejaba pasar una procesal sin estudiar.
        self.alguno_prospera_fondo = any(
            clase_sentido(p["sentido"]) == PROSPERA and p["clase"] == "fondo"
            for p in self.probs)
        self.alguno_prospera = any(clase_sentido(p["sentido"]) == PROSPERA for p in self.probs)

    def tocado(self, pid) -> bool:
        p = self.por_pid.get(pid)
        return bool(p and _clave_texto(_clave_problema(p["pregunta"])) in self.tocados)

    def sup_confirmada(self) -> bool:
        try:
            import suplencia as _sp
            return _sp.confirmada(self.suplencia)
        except Exception:
            return False

    def resolver_tesis(self, x: str) -> str | None:
        """«T3», «2014643», «registro 2014643» → el registro, si está en el
        índice o en la razón del secretario."""
        x = _ws(x)
        m = re.fullmatch(r"T\s*(\d+)", x, re.I)
        if m:
            for t in self.indice["tesis"]:
                if t["id"].upper() == f"T{m.group(1)}":
                    return t["registro"] or None
            return None
        reg = re.search(r"\b(\d{6,7})\b", x)
        if not reg:
            return None
        reg = reg.group(1)
        if any(t["registro"] == reg for t in self.indice["tesis"]):
            return reg
        if any(reg in anclas_de_razon(p["razon"])["registros"] for p in self.probs):
            return reg
        return None

    def resolver_norma(self, x: str) -> str | None:
        """«N2», «art. 84 CPC Qro» → «Art. 84 — <cuerpo legal>», si está en el
        índice o si el secretario lo citó en su razón."""
        x = _ws(x)
        m = re.fullmatch(r"N\s*(\d+)", x, re.I)
        if m:
            for n in self.indice["normas"]:
                if n["id"].upper() == f"N{m.group(1)}":
                    return f"Art. {n['articulo']} — {n['cuerpo_legal']}"
            return None
        a = re.search(r"(\d{1,4})", x)
        if not a:
            return None
        num = a.group(1)
        cand = [n for n in self.indice["normas"]
                if re.match(rf"^{num}\b", _ws(n["articulo"]))]
        if len(cand) == 1:
            return f"Art. {cand[0]['articulo']} — {cand[0]['cuerpo_legal']}"
        if len(cand) > 1:
            px = set(_palabras(x)) - {num, "art", "articulo", "de", "la", "del", "ley", "codigo"}
            for n in cand:
                if px & set(_palabras(n["cuerpo_legal"])):
                    return f"Art. {n['articulo']} — {n['cuerpo_legal']}"
            return f"Art. {cand[0]['articulo']} — {cand[0]['cuerpo_legal']}"
        if any(num in anclas_de_razon(p["razon"])["articulos"] for p in self.probs):
            return x[:160]
        return None

    def texto_de_fuentes(self, prem: dict) -> Texto:
        partes = []
        for x in prem["fuentes"]["tesis"]:
            reg = self.resolver_tesis(x)
            for t in self.indice["tesis"]:
                if reg and t["registro"] == reg:
                    partes += [t["rubro"], t["texto"]]
        for x in prem["fuentes"]["normas"]:
            m = re.search(r"(\d{1,4})", x)
            for n in self.indice["normas"]:
                if m and re.match(rf"^{m.group(1)}\b", _ws(n["articulo"])):
                    partes.append(n["texto"])
        return Texto(*partes)


def reparar(plan: dict, crit, segs, fases, material, contexto: str = "", suplencia=None, *,
            tocados=None) -> tuple[dict, list[str]]:
    """Lo que el código corrige sin volver a preguntar, y lo que dice al
    hacerlo. Devuelve (plan reparado, avisos). No toca el sentido: lo IMPONE
    (la etiqueta es la del criterio) y lo demás que el modelo afinó va a
    propuestas."""
    plan = copy.deepcopy(plan if isinstance(plan, dict) else {})
    if suplencia is None:
        suplencia = plan.get("suplencia") or {}
    cx = _Ctx(crit, segs, fases, material, contexto, suplencia, tocados)
    avisos: list[str] = []
    _del_modelo = list(plan.get("avisos_al_secretario") or [])
    plan["problemas"] = [{"id": p["id"], "pregunta": p["pregunta"], "sentido": p["sentido"],
                          "clase": p["clase"], "jerarquia": p["jerarquia"], "grupo": p["grupo"],
                          "sin_calificar": bool(p.get("sin_calificar"))}
                         for p in cx.probs]
    plan.setdefault("propuestas", [])
    plan.setdefault("avisos_al_secretario", [])
    segmentos = [s for s in (plan.get("segmentos") or []) if s.get("id")]
    # REFERENCIAS QUE NO CASAN (26-sep-2026, primer caso real): un segmento
    # repetido o uno que no está en el inventario no es un argumento más; se
    # quita la copia o lo inventado y se dice. Nunca un argumento del escrito:
    # el que FALTA sigue siendo motivo de rehacer el plan (V0 a).
    segmentos = _limpiar_segmentos(segmentos, cx, plan)
    por_id = {}
    for s in segmentos:
        por_id.setdefault(s["id"], s)

    # Los datos del piso mandan sobre lo que el modelo diga de ellos.
    sin_cita = []
    for s in segmentos:
        piso = cx.seg_piso.get(s["id"])
        if piso:
            s["concepto"] = piso["concepto"]
            s["concepto_inferido"] = bool(piso.get("concepto_inferido"))
            s["anclas"] = list(piso["anclas"])
            s["resumen"] = piso["texto"]
            s["cita"] = piso["cita"] if cx.escrito.contiene(piso["cita"], MIN_PALABRAS_CITA_SEGMENTO) else ""
        else:
            s.setdefault("anclas", [])
            s.setdefault("cita", "")
        # (g) Sin cita del piso: la del planificador, si está literal.
        if not s.get("cita") and s.get("cita_plan"):
            if cx.escrito.contiene(s["cita_plan"], MIN_PALABRAS_CITA_SEGMENTO):
                s["cita"] = s["cita_plan"]
            else:
                avisos.append(f"{s['id']}: la cita que propuso el planificador no está en el escrito; se borró")
        # (g) NI EL INVENTARIO NI EL PLANIFICADOR LA ENCONTRARON: el segmento es
        # del inventario (del código, no del modelo) y el estudio lo contesta
        # con plan o sin él, así que tirar el plan no lo ancla a nada; sólo
        # pierde la organización. Medido el 26-sep-2026: en el 93/2026 los dos
        # primeros intentos, que en lo demás pasaban V0, se rechazaron sólo por
        # esto. Queda sin cita, como el dato que no se verifica, y se dice.
        s["sin_cita"] = bool(cx.hay_escrito and not s["id"].startswith("S")
                             and s["id"] in cx.seg_piso and not s.get("cita"))
        if s["sin_cita"]:
            sin_cita.append(s["id"])
    if sin_cita:
        plan["avisos_al_secretario"].append(
            f"sin cita literal del escrito: {', '.join(sin_cita[:12])} (ni el inventario ni el "
            f"planificador la encontraron palabra por palabra); se estudian igual y el mapa los "
            f"muestra sin cita: compruébalos en el escrito")

    # (c) El problema lo fija el código.
    for s in segmentos:
        perm = problemas_permitidos(cx.seg_piso.get(s["id"]) or s, cx.probs, cx.n)
        if len(perm) == 1 and s.get("problema_id") != perm[0]:
            s["problema_id"] = perm[0]
        elif s.get("problema_id") not in perm:
            s["problema_id"] = perm[0] if perm else None
            avisos.append(f"{s['id']}: el problema que eligió el planificador no cubre su "
                          f"{'concepto' if s.get('concepto') else 'argumento'}; va al problema {s['problema_id']}")

    # (b) LA CALIFICACIÓN DE CADA ARGUMENTO, DENTRO DEL SENTIDO DE SU PROBLEMA
    # (plan-4; ver `etiqueta_fuera`). El sentido del PROBLEMA es el del
    # secretario y no se toca; la etiqueta del argumento se queda si cabe en
    # él. Lo que no cabe se corrige sin decidir nada de fondo, y se dice:
    #   · en un problema que NO SE ESTUDIA, la del problema; la otra, a
    #     propuestas (proponer estudiarlo es proponer otro sentido del problema);
    #   · FUNDADO dentro de un problema que no prospera → fundado pero
    #     insuficiente: es la calificación que dice que tiene razón y no alcanza;
    #   · vacía, fuera del catálogo o SIN ESTUDIO dentro de uno que se estudia →
    #     la del problema.
    # Salvo que el secretario haya tocado ese problema a mano: entonces ni se
    # propone.
    for s in segmentos:
        p = cx.por_pid.get(s.get("problema_id"))
        fijado = p["sentido"] if p else ""
        if not fijado:
            s["etiqueta"] = ""
            s["pendiente"] = "sentido"
            continue
        if s.get("pendiente") == "sentido":
            s["pendiente"] = None
        et = norm_sentido(s.get("etiqueta"))
        motivo = etiqueta_fuera(et, fijado)
        if not motivo:
            s["etiqueta"] = et
            continue
        if clase_sentido(fijado) == NO_SE_ESTUDIA:
            if et and calificacion_conocida(et) and not cx.tocado(p["id"]) \
                    and not any(x["seg"] == s["id"] for x in plan["propuestas"]):
                plan["propuestas"].append({"seg": s["id"], "de": fijado, "a": et,
                                           "por_que": "calificación que sugirió el planificador"})
            s["etiqueta"] = fijado
        elif clase_sentido(fijado) == NO_PROSPERA and clase_sentido(et) == PROSPERA \
                and calificacion_conocida(et):
            s["etiqueta"] = "fundado_insuficiente"
            avisos.append(f"{s['id']}: el planificador lo calificó «{et}» dentro de un problema "
                          f"«{fijado}»; va como fundado pero insuficiente: el estudio dice por qué "
                          f"no alcanza")
        else:
            if et:
                avisos.append(f"{s['id']}: «{et}» no cabe dentro de un problema «{fijado}»; lleva "
                              f"la calificación del problema")
            s["etiqueta"] = fijado
    # EL PROBLEMA QUE PROSPERA Y QUE NINGÚN ARGUMENTO FUNDA: las etiquetas del
    # planificador niegan el sentido del secretario. No se elige a cuál darle
    # la razón —eso sería decidir—: todos vuelven a la del problema, como en
    # plan-3, y lo que el planificador opinaba va a propuestas, que es un
    # cambio de sentido del PROBLEMA y sólo lo aplica el secretario.
    for pid in problemas_sin_quien_los_funde(segmentos, cx.probs):
        p = cx.por_pid[pid]
        suyos = [s for s in segmentos if s.get("problema_id") == pid
                 and s.get("pendiente") != "sentido"]
        cuenta: dict = {}
        for s in suyos:
            cuenta[s["etiqueta"]] = cuenta.get(s["etiqueta"], 0) + 1
        otra = max(cuenta, key=lambda k: cuenta[k]) if cuenta else ""
        for s in suyos:
            s["etiqueta"] = p["sentido"]
        if otra and not cx.tocado(pid) \
                and not any((por_id.get(x.get("seg")) or {}).get("problema_id") == pid
                            for x in plan["propuestas"]):
            plan["propuestas"].append({"seg": suyos[0]["id"], "de": p["sentido"], "a": otra,
                                       "por_que": "según el planificador, ninguno de los argumentos "
                                                  "de este problema lo funda"})
        avisos.append(f"problema {pid}: el planificador no dejó ningún argumento que lo funde; "
                      f"todos llevan «{p['sentido']}», el sentido que fijaste")
    # UNA PROPUESTA ES UNA CALIFICACIÓN (26-sep-2026): el planificador del
    # 642/2024 propuso «fundado → fondo_desestimado» y «→ no_combate», que son
    # RAZONES; el botón del panel pondría eso como sentido. Se traduce a la
    # calificación que la razón implica y la razón queda dicha en el porqué.
    for x in plan["propuestas"]:
        if x.get("a") in RAZONES and _implica(x["a"]) not in ("", x["a"]):
            x["por_que"] = _ws(f"({x['a']}) {x.get('por_que') or ''}")[:400]
            x["a"] = _implica(x["a"])

    # UNA PROPUESTA ES DEL PROBLEMA, NO DEL ARGUMENTO (plan-4): el botón del
    # panel cambia la calificación de todo el problema. Así que «de» es el
    # sentido del problema, lo que ya es ese sentido no se propone, dos
    # propuestas iguales para el mismo problema son una, y la que sólo repite
    # la etiqueta que ya lleva su argumento habla del ARGUMENTO —ya está dicha
    # en su etiqueta— y no se ofrece como cambio del problema.
    def _fijado_de(sid):
        p_ = cx.por_pid.get((por_id.get(sid) or {}).get("problema_id"))
        return p_["sentido"] if p_ else ""
    _vistas, _props = set(), []
    for x in plan["propuestas"]:
        if not (x.get("seg") in por_id and x.get("a") and calificacion_conocida(x["a"])):
            continue
        _pid = por_id[x["seg"]].get("problema_id")
        if not _fijado_de(x["seg"]) or x["a"] == _fijado_de(x["seg"]) or cx.tocado(_pid) \
                or x["a"] == (por_id[x["seg"]].get("etiqueta") or "") \
                or (_pid, x["a"]) in _vistas:
            continue
        _vistas.add((_pid, x["a"]))
        x["de"] = _fijado_de(x["seg"])
        _props.append(x)
    plan["propuestas"] = _props

    # (f) `reitera` con ancla propia → desarrolla, con la diferencia del ancla.
    for s in segmentos:
        t = s.get("reitera")
        if not t:
            continue
        otro = cx.seg_piso.get(t) or por_id.get(t)
        if otro is None or t == s["id"]:
            s["reitera"] = None
            continue
        propias = _anclas_seg(cx.seg_piso.get(s["id"]) or s) - _anclas_seg(otro)
        if propias:
            s["reitera"] = None
            if s.get("trat") in ("remite", "aplica", "residual", ""):
                s["trat"] = "desarrolla"
            s["diferencia"] = s.get("diferencia") or tipo_de_ancla(sorted(propias)[0])
            avisos.append(f"{s['id']}: no reitera a {t} porque trae {', '.join(sorted(propias)[:3])}; se desarrolla")

    # (g) Datos verificados palabra por palabra; lo que no, se borra.
    for s in segmentos:
        d = s.get("dato")
        if d and not cx.fuentes_dato.contiene(d.get("cita", ""), MIN_PALABRAS_CITA):
            s["dato"] = None
            avisos.append(f"{s['id']}: el dato que propuso el planificador no está en el expediente; "
                          f"se borró y el argumento se contesta sin él")
    for pr in plan.get("proposiciones") or []:
        if pr.get("cita") and not cx.fuentes_p.contiene(pr["cita"], MIN_PALABRAS_CITA):
            pr["cita"] = ""
            avisos.append(f"{pr['id']}: la cita de la proposición no está en el acto ni en el contexto; se borró")

    # (h) Fuentes de las premisas: del índice o de la razón del secretario.
    for m in plan.get("premisas") or []:
        tes, nor = [], []
        for x in m["fuentes"]["tesis"]:
            reg = cx.resolver_tesis(x)
            if reg and reg not in tes:
                tes.append(reg)
            elif not reg:
                avisos.append(f"{m['id']}: «{x[:40]}» no está en el material ni en la razón; se quitó de sus fuentes")
        for x in m["fuentes"]["normas"]:
            n = cx.resolver_norma(x)
            if n and n not in nor:
                nor.append(n)
            elif not n:
                avisos.append(f"{m['id']}: «{x[:40]}» no está en el material ni en la razón; se quitó de sus fuentes")
        m["fuentes"] = {"tesis": tes, "normas": nor}
        # (i) El rastro, con su cita literal.
        if m.get("rastro_cita"):
            en_razon = any(cx.razones[p["id"]].contiene(m["rastro_cita"]) for p in cx.probs)
            en_mat = cx.texto_de_fuentes(m).contiene(m["rastro_cita"])
            if m.get("rastro") == "razon" and not en_razon and en_mat:
                m["rastro"] = "material"
            elif m.get("rastro") == "material" and not en_mat and en_razon:
                m["rastro"] = "razon"
            elif not (en_razon or en_mat):
                # PALABRAS DE MÁS EN LOS BORDES (revisión adversarial,
                # 26-sep-2026): en el 640/2024 el rastro de M2 era el rubro
                # de su propia tesis (T7, registro 187149) con un artículo
                # añadido delante, y V0 retiraba una premisa buena. Se deja
                # sólo lo que está palabra por palabra; nada se añade.
                _rec, _donde = _recorte_literal(
                    m["rastro_cita"], [cx.texto_de_fuentes(m)] + [cx.razones[p["id"]] for p in cx.probs])
                if _rec:
                    m["rastro"] = "material" if _donde == 0 else "razon"
                    m["rastro_cita"] = _rec
                    avisos.append(f"{m['id']}: su rastro traía palabras de más en los bordes; se dejó sólo "
                                  f"lo que está palabra por palabra en {'su fuente' if _donde == 0 else 'tu razón'}")
                else:
                    m["rastro_cita"] = ""
    # (h, segunda mitad) Lo que el secretario citó en su razón figura en la
    # premisa de la unidad de su problema.
    prem_por_id = {m["id"]: m for m in plan.get("premisas") or []}
    for u in plan.get("unidades") or []:
        m = prem_por_id.get(u.get("premisa"))
        if not m:
            continue
        pids = set(u.get("problemas") or []) | {
            (por_id.get(x) or {}).get("problema_id") for x in u.get("segmentos") or []}
        for pid in pids:
            p = cx.por_pid.get(pid)
            if not p:
                continue
            an = anclas_de_razon(p["razon"])
            for reg in an["registros"]:
                if reg not in m["fuentes"]["tesis"]:
                    m["fuentes"]["tesis"].append(reg)
            for art in an["articulos"]:
                if not any(re.search(rf"\b{art}\b", n) for n in m["fuentes"]["normas"]):
                    n = cx.resolver_norma(f"art. {art}")
                    m["fuentes"]["normas"].append(n or f"art. {art} (razón del secretario)")

    # (k) Suplidos sin beneficio: al mapa, no a la sentencia (art. 79,
    # PENÚLTIMO párrafo, LA).
    for s in segmentos:
        if s["id"].startswith("S") and clase_sentido(s.get("etiqueta")) != PROSPERA:
            s["trat"] = "no_se_expresa_art79"
    # Y el ÚLTIMO párrafo del art. 79: la suplencia por violaciones procesales
    # o formales sólo opera si en el acto no hay vicio de fondo. Si el criterio
    # concede por fondo y además por una procesal o formal suplida, el plan no
    # lo cambia (es el sentido), pero lo dice.
    if cx.alguno_prospera_fondo:
        for s in segmentos:
            if s["id"].startswith("S") and s.get("vicio") in ("procesal", "forma") \
                    and clase_sentido(s.get("etiqueta")) == PROSPERA:
                _a = (f"{s['id']}: se suple una violación procesal o formal y hay un vicio de fondo "
                      f"que prospera (art. 79, último párrafo, de la Ley de Amparo)")
                if _a not in plan["avisos_al_secretario"]:
                    plan["avisos_al_secretario"].append(_a)

    # LO QUE EL CRITERIO NO MANDA ESTUDIAR, NO SE ESTUDIA: si su problema dice
    # «innecesario» o «sin materia», el argumento va a una frase con su razón,
    # no a una unidad con su premisa. Es organización, no sentido: la etiqueta
    # ya es la del criterio. Tampoco se le pide una razón que no se usará.
    for s in segmentos:
        if s.get("etiqueta") and clase_sentido(s["etiqueta"]) == NO_SE_ESTUDIA:
            if s.get("trat") in _EN_UNIDAD + ("",):
                avisos.append(f"{s['id']}: su criterio dice que no se estudia; va a no_se_estudia")
                s["trat"] = "no_se_estudia"
            if s.get("pendiente") == "razon":
                s["pendiente"] = None
    # (m) El adhesivo sigue la suerte del principal (art. 182 LA): si nada
    # prospera y el criterio aun así le da una calificación de estudio, el plan
    # no la cambia; lo dice.
    if not cx.alguno_prospera:
        for s in segmentos:
            if s["id"].startswith("AD") and s.get("etiqueta") \
                    and clase_sentido(s["etiqueta"]) != NO_SE_ESTUDIA:
                _a = (f"{s['id']}: el principal no prospera y tu criterio estudia el adhesivo; sigue la "
                      f"suerte del principal (art. 182 de la Ley de Amparo)")
                if _a not in plan["avisos_al_secretario"]:
                    plan["avisos_al_secretario"].append(_a)

    # Decisión 6: el pendiente de razón se desarrolla.
    for s in segmentos:
        if s.get("pendiente") == "razon" and s.get("trat") in ("aplica", "remite", "residual", ""):
            s["trat"] = "desarrolla"

    # (p) Proposición suficiente que nadie ataca: aviso, NUNCA reclasificación
    # (cambiaría el sentido, que no es del plan).
    atacadas = {s.get("ataca") for s in segmentos} | {s.get("razon_p") for s in segmentos}
    for pr in plan.get("proposiciones") or []:
        if pr.get("relacion") == "suficiente" and pr.get("id") not in atacadas:
            _a = (f"{pr['id']} («{pr['dice'][:90]}») sostiene lo resuelto por sí sola y ningún "
                  f"argumento la ataca")
            if _a not in plan["avisos_al_secretario"]:
                plan["avisos_al_secretario"].append(_a)
    # (e) El grupo del secretario junta proposiciones distintas: se avisa.
    for u in plan.get("unidades") or []:
        us = [por_id[x] for x in u.get("segmentos") or [] if x in por_id]
        ps = {s.get("ataca") for s in us if s.get("ataca")}
        if len(ps) > 1 and _grupo_comun(u, us, cx):
            _a = (f"{u['id']}: el grupo que marcaste junta argumentos que atacan "
                  f"{', '.join(sorted(ps))}; se estudian en un apartado con respuestas separadas por proposición")
            if _a not in plan["avisos_al_secretario"]:
                plan["avisos_al_secretario"].append(_a)
    # (j, la parte que no es del plan) Una procesal que el CRITERIO deja sin
    # estudiar sin concesión de fondo de mayor beneficio: el plan no cambia el
    # sentido, pero lo dice arriba (arts. 74-V, 174 y 189 LA).
    for s in segmentos:
        if _guarda_procesal(plan.get("tipo_asunto", "")) and _es_procesal(s, cx) \
                and clase_sentido(s.get("etiqueta")) == NO_SE_ESTUDIA \
                and not cx.alguno_prospera_fondo and not s["id"].startswith("AD"):
            _a = (f"{s['id']}: tu criterio deja sin estudiar una violación procesal sin una concesión "
                  f"de fondo de mayor beneficio (arts. 74, fracción V, 174 y 189 de la Ley de Amparo)")
            if _a not in plan["avisos_al_secretario"]:
                plan["avisos_al_secretario"].append(_a)

    plan["segmentos"] = segmentos
    plan["suplencia"] = dict(suplencia) if isinstance(suplencia, dict) else {}
    plan["n_planteamientos"] = cx.n
    # LO ORGANIZATIVO QUE V0 RECHAZABA Y EL CÓDIGO PUEDE CORREGIR SOLO.
    _reparar_organizacion(plan, cx)
    # LO QUE DICE EL CÓDIGO VA PRIMERO: la ficha guarda doce avisos del plan y
    # una corrección del código no puede quedarse fuera por los del modelo.
    _av = plan["avisos_al_secretario"]
    plan["avisos_al_secretario"] = [a for a in _av if a not in _del_modelo] + \
                                   [a for a in _av if a in _del_modelo]
    return plan, avisos


# ═══ REPARAR LA ORGANIZACIÓN (26-sep-2026, tras el primer caso real) ════════
# El ADC 642/2024 se quedó sin plan dos veces por «U9: mezcla vicios» y «M4:
# premisa sin rastro verificado». Las dos reglas son correctas (una unidad es
# la MISMA proposición, la MISMA razón y el MISMO vicio; una premisa necesita
# rastro verificado), pero tirar el plan entero no lo era: el código puede
# partir la unidad y retirar la premisa sin decidir nada de fondo, como V0 (g)
# ya borra el dato que no se verifica. Medido en 8 casos × 2 corridas con el
# código de producción: 8 de 16 corridas se quedaban sin plan, y en las 8 lo
# que tumbó el segundo intento era organizativo o una falsa alarma de V0; la
# única falla de contenido apareció en un primer intento. Aquí va sólo lo que es
# ORGANIZACIÓN; lo de contenido (un argumento que falta, una procesal que se
# deja sin estudiar sin el art. 189, una inoperancia sin razón de
# inoperancia) sigue siendo motivo de rehacer el plan. Nada de esto toca la
# etiqueta de un segmento ni quita un argumento del escrito, y cada
# corrección se dice en `avisos_al_secretario`.

def _limpiar_segmentos(segmentos: list, cx: "_Ctx", plan: dict) -> list:
    """Una copia repetida de un segmento, fuera (se queda la primera); un
    identificador DEL ESPACIO DEL INVENTARIO (C o A: los únicos que numera
    `inventario.segmentos`) que no está en él, fuera: es un argumento que el
    planificador numeró mal o inventó. Un AD o un S no se toca: pueden venir
    de otro escrito o de la suplencia, y V0 (a) decide. El que FALTA no se
    inventa aquí: V0 (a) lo sigue rechazando."""
    fuera, vistos, dup, ajenos = [], set(), [], []
    for s in segmentos:
        sid = s["id"]
        if sid in vistos:
            dup.append(sid)
            continue
        if cx.seg_piso and sid not in cx.seg_piso and re.match(r"^[CA]\d", sid):
            ajenos.append(sid)
            continue
        vistos.add(sid)
        fuera.append(s)
    if dup:
        plan["avisos_al_secretario"].append(
            f"el planificador repitió {', '.join(sorted(set(dup))[:8])}: se tomó la primera vez")
    if ajenos:
        plan["avisos_al_secretario"].append(
            f"el planificador nombró segmentos que no están en el inventario "
            f"({', '.join(ajenos[:8])}): se quitaron")
    return fuera


def _contestado(s: dict) -> bool:
    """¿Se contesta dentro de una unidad? (lo que V0 e mira)."""
    return s.get("trat") in _EN_UNIDAD and s.get("pendiente") != "sentido"


def _implica(rz: str) -> str:
    """La calificación que implica una razón del catálogo (para la propuesta
    que deja ver lo que el planificador opinaba)."""
    if rz in _RAZONES_DE_INFUNDADO:
        return "infundado"
    if rz == "fundado_insuficiente":
        return "fundado_insuficiente"
    if rz in _RAZONES_DE_INOPERANTE:
        return "inoperante"
    if rz in ("fundado", "esencialmente_fundado"):
        return rz
    if rz in ("sin_materia", "adhesivo_sin_materia"):
        return "sin_materia"
    if rz in RAZONES:
        return "innecesario"
    return ""


def _falla_razon(rz: str, et: str) -> str:
    """Por qué la razón `rz` no cabe con la etiqueta `et` (V0 d), o «»."""
    if not rz or not et:
        return ""
    if RAZONES[rz]["clase"] != clase_sentido(et):
        return f"la razón {rz} no es compatible con «{et}»"
    if et == "infundado" and rz not in _RAZONES_DE_INFUNDADO:
        return f"«infundado» sólo admite {', '.join(sorted(_RAZONES_DE_INFUNDADO))}"
    if et == "inoperante" and rz not in _RAZONES_DE_INOPERANTE:
        return f"«inoperante» exige una razón de inoperancia, no {rz}"
    if et == "fundado_insuficiente" and rz != "fundado_insuficiente":
        return "«fundado pero insuficiente» exige la razón fundado_insuficiente"
    return ""


def _razon_de_la_etiqueta(s: dict, et: str, cx: "_Ctx") -> str:
    """LA ÚNICA razón del catálogo que la etiqueta deja, o «» si deja varias.
    «fundado» → fundado; «infundado» → omision_inexistente si el vicio es de
    omisión y fondo_desestimado si no; «fundado pero insuficiente» → la suya.
    Lo que no se estudia, por lo que el CRITERIO dice de él: sin materia, el
    adhesivo sin materia, la procesal que una concesión de fondo vuelve
    innecesaria (art. 189) y, lo demás, lo que cae con el principal (el
    árbol escribe «Dado el sentido del estudio del problema principal, queda
    sin materia…»). La procesal que el criterio deja sin estudiar SIN
    concesión de fondo no tiene razón que la cubra: se queda como está y la
    avisa `reparar` (arts. 74-V, 174 y 189). «Inoperante» deja diez: no se
    elige ninguna; eso lo decide el reintento."""
    et = norm_sentido(et)
    cl = clase_sentido(et)
    if cl == PROSPERA:
        return "esencialmente_fundado" if et == "esencialmente_fundado" else "fundado"
    if et == "infundado":
        return "omision_inexistente" if s.get("vicio") == "omision" else "fondo_desestimado"
    if et == "fundado_insuficiente":
        return "fundado_insuficiente"
    if cl == NO_SE_ESTUDIA:
        sid = s.get("id") or ""
        if sid.startswith("AD") and not cx.alguno_prospera:
            return "adhesivo_sin_materia"
        procesal = _guarda_procesal(getattr(cx, "tipo_asunto", "")) and _es_procesal(s, cx)
        if procesal:
            return "innecesario_mayor_beneficio" if cx.alguno_prospera_fondo else ""
        if et == "adhesivo_sin_materia":
            return "adhesivo_sin_materia"
        if et == "sin_materia":
            return "sin_materia"
        return "cae_con_principal"
    return ""


def _premisa_con_rastro(m: dict, cx: "_Ctx") -> bool:
    """V0 (i), en un solo sitio: el rastro dice de dónde sale la premisa y su
    cita está, palabra por palabra, ahí."""
    cita = m.get("rastro_cita") or ""
    if m.get("rastro") == "razon":
        return any(cx.razones[p["id"]].contiene(cita) for p in cx.probs)
    if m.get("rastro") == "material":
        return cx.texto_de_fuentes(m).contiene(cita)
    return False


def _recorte_literal(cita: str, textos: list, max_fuera: int = 3) -> tuple[str, int]:
    """La cita SIN hasta `max_fuera` palabras de sus bordes, si así está
    palabra por palabra en alguno de `textos` (objetos `Texto`) y conserva al
    menos tres cuartas partes de sus palabras y nunca menos de ocho. Devuelve
    (lo que queda, índice del texto donde está) o («», -1). Sólo QUITA: lo que
    queda es copia literal, y lo quitado deja de presentarse como cita."""
    toks = str(cita or "").split()
    n = len(toks)
    minimo = max(8, -(-3 * n // 4))
    for fuera in range(1, max_fuera + 1):
        for a in range(fuera + 1):
            w = toks[a:n - (fuera - a)]
            if len(w) < minimo:
                continue
            c = " ".join(w)
            for i, t in enumerate(textos):
                if t.contiene(c):
                    return c, i
    return "", -1


def _aviso(plan: dict, texto: str) -> None:
    if texto not in plan["avisos_al_secretario"]:
        plan["avisos_al_secretario"].append(texto)


def _reparar_organizacion(plan: dict, cx: "_Ctx") -> None:
    """Las correcciones organizativas, EN SITIO y en este orden (cada una
    deja lo que la siguiente necesita): referencias, razones, premisas sin
    rastro, unidades que mezclan y orden."""
    cx.tipo_asunto = plan.get("tipo_asunto", "")
    segmentos = plan.get("segmentos") or []
    por_id = {s["id"]: s for s in segmentos}
    props = {p.get("id") for p in plan.get("proposiciones") or [] if p.get("id")}

    # 1 · REFERENCIAS A IDENTIFICADORES QUE NO EXISTEN, limpias.
    quitadas = []
    for s in segmentos:
        if s.get("ataca") and s["ataca"] not in props:
            quitadas.append(f"{s['id']}→{s['ataca']}")
            s["ataca"] = None
        if s.get("razon_p") and s["razon_p"] not in props:
            s["razon_p"] = None
    prems = [m for m in plan.get("premisas") or [] if m.get("id")]
    for m in prems:
        _r = [pk for pk in m.get("responde_a") or [] if pk in props]
        if len(_r) != len(m.get("responde_a") or []):
            quitadas.append(f"{m['id']}→{', '.join(pk for pk in m['responde_a'] if pk not in props)}")
            m["responde_a"] = _r
    ids_m = {m["id"] for m in prems}
    en_unidad: dict = {}
    unis = []
    rotas_u = []                  # unidades cuya premisa no existe: como las retiradas (paso 3)
    for u in plan.get("unidades") or []:
        lista = []
        for x in u.get("segmentos") or []:
            if x not in por_id:
                quitadas.append(f"{u.get('id')}→{x}")
            elif x in en_unidad:
                quitadas.append(f"{x} también en {u.get('id')} (se queda en {en_unidad[x]})")
            elif x not in lista:
                lista.append(x)
                en_unidad[x] = u.get("id")
        u["segmentos"] = lista
        if u.get("premisa") and u["premisa"] not in ids_m:
            quitadas.append(f"{u.get('id')}→{u['premisa']}")
            u["premisa"] = None
            rotas_u.append(u)
        if lista:
            unis.append(u)
    plan["unidades"] = unis
    if quitadas:
        _aviso(plan, "referencias del plan a lo que no existe, quitadas: " + "; ".join(quitadas[:10]))

    # 2 · LA RAZÓN QUE NO CABE CON LA ETIQUETA. La etiqueta es la del criterio
    # (V0 b) y no se toca; en lugar de la razón que la contradice va la única
    # que la etiqueta deja, y la que había se dice en el aviso. Si deja varias
    # («inoperante»), no se elige: el reintento. Medido: 30 rechazos en 6 de
    # las 16 corridas (4 de los 8 casos), y el reintento no los corregía
    # (43/2025: diez en el primer intento, cuatro en el segundo). Un segmento
    # pendiente de razón (Decisión 6) no la necesita.
    # ESA RAZÓN NO ES UNA PROPUESTA (revisión adversarial, 26-sep-2026): en
    # los 47 casos medidos el planificador había puesto como etiqueta el
    # sentido del secretario y no propuso nada para ese argumento; en los
    # problemas «fundado» (43, 103, 642, 93) lo que el propio planificador dice
    # que el argumento sostiene es fundado: usaba fondo_desestimado u
    # omision_inexistente para decir de QUÉ trata el argumento (fondo,
    # omisión), no cómo se califica. Convertirla en propuesta ponía en el panel
    # un botón para invertir el sentido que nadie había propuesto. Las
    # afinaciones del planificador salen de su etiqueta o de sus propuestas.
    ajustadas, puestas = [], []
    for s in segmentos:
        et = s.get("etiqueta") or ""
        if not et or s.get("pendiente") == "sentido":
            continue
        rz = s.get("razon") or ""
        if not rz and s.get("pendiente") == "razon":
            continue
        if rz and not _falla_razon(rz, et):
            continue
        nueva = _razon_de_la_etiqueta(s, et, cx)
        if not nueva or nueva == rz:
            continue
        if rz:
            ajustadas.append(f"{s['id']} ({rz}→{nueva})")
        else:
            puestas.append(f"{s['id']} ({nueva})")
        s["razon"] = nueva
        if not RAZONES[nueva].get("con_p"):
            s["razon_p"] = None
    if ajustadas:
        _aviso(plan, "razones que no cabían con tu sentido, ajustadas a él (antes de la flecha, la "
                     "que había puesto el planificador; no cambia la calificación): "
                     + ", ".join(ajustadas[:12]))
    if puestas:
        _aviso(plan, "razones que el planificador dejó vacías, puestas por tu sentido: "
                     + ", ".join(puestas[:12]))

    # 3 · PREMISA SIN RASTRO VERIFICADO (V0 i): se retira. Sus unidades quedan
    # sin premisa y lo que se contestaba con ella (aplica, remite) se
    # desarrolla, desde la razón del secretario. Medido: de las cuatro que V0
    # rechazó, tres citaban como rastro el resumen del ACTO (lo que resolvió
    # la responsable), que no es regla verificada que exponer; la cuarta era
    # el rubro de su tesis con una palabra de más delante, y ésa ya no llega
    # aquí: `reparar` (i) deja sólo lo literal (`_recorte_literal`).
    def _sin_premisa(unis_m: list) -> list:
        segs_m = []
        for u in unis_m:
            u["premisa"] = None
            for x in u["segmentos"]:
                s = por_id[x]
                if _contestado(s):
                    if s.get("trat") in ("aplica", "remite"):
                        s["trat"] = "desarrolla"
                    s["sin_premisa"] = True
                    segs_m.append(x)
        return segs_m

    retiradas = [m for m in plan.get("premisas") or [] if not _premisa_con_rastro(m, cx)]
    if retiradas:
        fuera = {m["id"] for m in retiradas}
        plan["premisas"] = [m for m in plan.get("premisas") or [] if m["id"] not in fuera]
        for m in retiradas:
            unis_m = [u for u in plan["unidades"] if u.get("premisa") == m["id"]]
            segs_m = _sin_premisa(unis_m)
            _aviso(plan, f"{m['id']} se retiró: su rastro no está palabra por palabra en tu razón ni "
                         f"en el material"
                         + (f"; {', '.join(u['id'] for u in unis_m)} queda sin premisa expuesta y se "
                            f"desarrolla desde tu razón ({', '.join(segs_m[:10])})" if unis_m else ""))
    # LA UNIDAD QUE NOMBRABA UNA PREMISA INEXISTENTE (paso 1) queda igual que
    # la de una retirada (revisión adversarial, 26-sep-2026): si no, sus
    # argumentos salían en el guion como APLICA sin premisa que aplicar.
    rotas_u = [u for u in rotas_u if u in plan["unidades"]]
    if rotas_u:
        segs_m = _sin_premisa(rotas_u)
        if segs_m:
            _aviso(plan, f"{', '.join(u['id'] for u in rotas_u)} nombraba una premisa que no existe: "
                         f"queda sin premisa expuesta y se desarrolla desde tu razón ({', '.join(segs_m[:10])})")

    # 4 · UNIDAD QUE MEZCLA proposición, vicio o razón (V0 e): se parte. Salvo
    # el grupo del secretario, que manda. Cada parte conserva la premisa (el
    # guion la expone una vez y las demás la aplican); la objeción se queda
    # en la primera. Un segmento sin proposición o pendiente de razón no
    # separa: va con los de su vicio.
    nuevas_u = []
    ultimo = max([_int(re.sub(r"\D", "", str(u.get("id") or "")) or 0) for u in plan["unidades"]] + [0])
    for u in plan["unidades"]:
        us = [por_id[x] for x in u["segmentos"]]
        conts = [s for s in us if _contestado(s)]
        partes: list = []
        for s in conts:
            a = s.get("ataca") or None
            v = s.get("vicio")
            r = s.get("razon") or None           # pendiente de razón: no separa
            for g in partes:
                if g["v"] == v and (a is None or g["a"] is None or g["a"] == a) \
                        and (r is None or g["r"] is None or g["r"] == r):
                    g["segs"].append(s["id"])
                    g["a"] = g["a"] or a
                    g["r"] = g["r"] or r
                    break
            else:
                partes.append({"a": a, "v": v, "r": r, "segs": [s["id"]]})
        if len(partes) < 2 or _grupo_comun(u, conts, cx):
            nuevas_u.append(u)
            continue
        resto = [x for x in u["segmentos"] if x not in {y for g in partes for y in g["segs"]}]
        ids = []
        for i, g in enumerate(partes):
            if i == 0:
                uid = u["id"]
                segs_g = g["segs"] + resto
            else:
                ultimo += 1
                uid = f"U{ultimo}"
                segs_g = g["segs"]
            pids = sorted({por_id[x].get("problema_id") for x in segs_g if por_id[x].get("problema_id")})
            nuevas_u.append({"id": uid, "problemas": pids or list(u.get("problemas") or []),
                             "segmentos": segs_g, "premisa": u.get("premisa"),
                             "objecion": u.get("objecion") if i == 0 else None})
            ids.append(f"{uid} ({', '.join(g['segs'][:6])})")
        que = []
        if len({g["a"] for g in partes if g["a"]}) > 1:
            que.append("proposiciones")
        if len({g["v"] for g in partes}) > 1:
            que.append("vicios")
        if len({g["r"] for g in partes if g["r"]}) > 1:
            que.append("razones")
        _aviso(plan, f"{u['id']} juntaba argumentos con {' y '.join(que) or 'rasgos'} distintos; se partió en "
                     + " · ".join(ids) + (f", con la misma premisa {u['premisa']}" if u.get("premisa") else ""))
    plan["unidades"] = nuevas_u

    # 5 · EL ORDEN (V0 l), cuando el código puede cumplirlo sin decidir nada:
    # la procedencia primero y, en el amparo directo sin mayor beneficio
    # dicho, el fondo antes que el procedimiento y la forma (art. 189). Si con
    # el orden del escrito no se puede, el de prelación, que es lo que el
    # prompt ya pide. Si ni así, se deja como estaba: lo decide el reintento.
    _reordenar(plan, cx)


def _reordenar(plan: dict, cx: "_Ctx") -> None:
    por_id = {s["id"]: s for s in plan.get("segmentos") or []}
    if not _faltas_de_orden(plan, cx, por_id):
        return
    import tipos_asunto as _ta
    _ad = _ta.normalizar(plan.get("tipo_asunto", "")) in ("", "amparo_directo")
    por_que = _sin_acentos((plan.get("orden") or {}).get("por_que") or "").lower()
    unis0 = list(plan.get("unidades") or [])
    orden0 = dict(plan.get("orden") or {})

    def _peso(u):
        v = {por_id[x].get("vicio") for x in u.get("segmentos") or [] if x in por_id}
        if "procedencia" in v:
            return 0
        if _ad and "beneficio" not in por_que and v & _VICIOS_TRAS_FONDO_189 \
                and not v & _VICIOS_FONDO_189:
            return 2
        return 1

    unis = sorted(unis0, key=_peso)
    # Cada accesorio después de su principal: la primera unidad de un
    # accesorio no puede ir antes que la primera de su principal.
    principales = [p["id"] for p in cx.probs if p["jerarquia"] == "principal"]
    for _ in range(len(unis)):
        primero: dict = {}
        for i, u in enumerate(unis):
            for x in u.get("segmentos") or []:
                primero.setdefault((por_id.get(x) or {}).get("problema_id"), i)
        mover = None
        for p in cx.probs:
            dep = p.get("depende_de") or (principales[0] if principales and p["jerarquia"] == "accesorio"
                                         and p["id"] not in principales else None)
            if dep and p["id"] in primero and dep in primero and primero[p["id"]] < primero[dep]:
                mover = (primero[p["id"]], primero[dep])
                break
        if not mover:
            break
        u = unis.pop(mover[0])
        unis.insert(mover[1], u)
    cambiado = [u["id"] for u in unis] != [u["id"] for u in unis0]
    plan["unidades"] = unis
    if cambiado and not _faltas_de_orden(plan, cx, por_id):
        _aviso(plan, "el orden de las unidades se ajustó a la ley (procedencia primero; en el amparo "
                     "directo, el fondo antes que el procedimiento y la forma, art. 189): "
                     + ", ".join(u["id"] for u in unis))
        return
    if (plan.get("orden") or {}).get("criterio") != "prelacion":
        plan["orden"] = dict(orden0, criterio="prelacion")
        if not _faltas_de_orden(plan, cx, por_id):
            _aviso(plan, "el orden del escrito no cumplía la prelación de la ley; el estudio sigue el "
                         "de las unidades: " + ", ".join(u["id"] for u in unis))
            return
    plan["unidades"] = unis0
    plan["orden"] = orden0


def _grupo_comun(u: dict, us: list, cx: _Ctx) -> bool:
    pids = set(u.get("problemas") or []) | {s.get("problema_id") for s in us}
    grupos = {(cx.por_pid.get(p) or {}).get("grupo") for p in pids if p}
    return len(grupos) == 1 and bool(next(iter(grupos)))


def _guarda_procesal(tipo_asunto: str = "") -> bool:
    """¿Rigen los arts. 74-V, 174 y 189? La MISMA pregunta que el árbol
    (`violacion_procesal.guarda_aplica`): amparo directo, o sin tipo."""
    try:
        import violacion_procesal as _vp
        return bool(_vp.guarda_aplica(tipo_asunto or ""))
    except Exception:                                   # pragma: no cover
        return True


def _es_procesal(s: dict, cx: _Ctx) -> bool:
    p = cx.por_pid.get(s.get("problema_id")) or {}
    return s.get("vicio") == "procesal" or p.get("clase") == "procesal"


# ═══ V0: LO QUE OBLIGA A REHACER EL PLAN ════════════════════════════════════

def validar(plan: dict, crit, segs, fases, material, contexto: str = "", *,
            suplencia=None, tocados=None) -> list[str]:
    """V0 (w2_final §4.3). Devuelve la lista de lo que falla, en castellano
    llano porque vuelve al planificador en el reintento; [] = válido. Es
    determinista y no llama a nadie."""
    plan = plan if isinstance(plan, dict) else {}
    if suplencia is None:
        suplencia = plan.get("suplencia") or {}
    cx = _Ctx(crit, segs, fases, material, contexto, suplencia, tocados)
    f: list[str] = []
    segmentos = [s for s in (plan.get("segmentos") or [])]
    ids = [s.get("id") for s in segmentos]
    por_id = {s.get("id"): s for s in segmentos if s.get("id")}
    props = {p.get("id"): p for p in (plan.get("proposiciones") or []) if p.get("id")}
    prems = {m.get("id"): m for m in (plan.get("premisas") or []) if m.get("id")}
    unis = [u for u in (plan.get("unidades") or [])]

    if not segmentos:
        return ["el plan no trae segmentos"]
    # (a) Todo el piso, una vez; nada inventado; todo colocado.
    faltan = [s["id"] for s in cx.segs if s["id"] not in por_id]
    if faltan:
        f.append(f"faltan segmentos del inventario: {', '.join(faltan[:20])}")
    dup = sorted({x for x in ids if ids.count(x) > 1 and x})
    if dup:
        f.append(f"segmentos repetidos: {', '.join(dup)}")
    ajenos = [x for x in ids if x and x not in cx.seg_piso and not x.startswith("S")]
    if ajenos:
        f.append(f"segmentos que no están en el inventario: {', '.join(ajenos[:10])}")
    if any(not x for x in ids):
        f.append("hay segmentos sin identificador válido")
    en_unidad = {}
    for u in unis:
        for x in u.get("segmentos") or []:
            if x in en_unidad:
                f.append(f"{x} está en dos unidades ({en_unidad[x]} y {u.get('id')})")
            en_unidad[x] = u.get("id")
            if x not in por_id:
                f.append(f"la unidad {u.get('id')} nombra un segmento que no existe: {x}")
    for s in segmentos:
        sid = s.get("id") or "?"
        if s.get("trat") not in TRATAMIENTOS:
            f.append(f"{sid}: trat fuera del catálogo")
        elif s.get("trat") in _EN_UNIDAD and sid not in en_unidad and not s.get("pendiente"):
            f.append(f"{sid}: se contesta ({s['trat']}) pero no está en ninguna unidad")

    for s in segmentos:
        sid = s.get("id") or "?"
        p = cx.por_pid.get(s.get("problema_id"))
        # (c)
        perm = problemas_permitidos(cx.seg_piso.get(sid) or s, cx.probs, cx.n)
        if s.get("problema_id") not in perm:
            f.append(f"{sid}: problema_id {s.get('problema_id')} no es uno de sus problemas posibles ({perm})")
        # (b) COHERENCIA CON EL SENTIDO DEL PROBLEMA (plan-4), no igualdad: la
        # etiqueta es la calificación del argumento y tiene que caber en el
        # sentido que el secretario fijó para su problema (`etiqueta_fuera`).
        fijado = (p or {}).get("sentido", "")
        if fijado and s.get("pendiente") != "sentido":
            _fe = etiqueta_fuera(s.get("etiqueta"), fijado)
            if _fe:
                f.append(f"{sid}: etiqueta «{s.get('etiqueta')}» {_fe}")
        if not fijado and s.get("pendiente") != "sentido":
            f.append(f"{sid}: su problema no tiene sentido fijado; va con pendiente «sentido»")
        # (d) La razón es del catálogo y coherente con la etiqueta. EL PENDIENTE
        # DE RAZÓN NO LA LLEVA (falsa alarma medida el 26-sep-2026: el prompt
        # le dice al planificador «no inventes esa razón» y V0 rechazaba la
        # razón vacía: 14 rechazos en 6 de 16 corridas, 5 de los 8 casos): lo
        # contesta el estudio con el material o el secretario en el panel
        # (Decisión 6).
        rz = s.get("razon") or ""
        if not rz:
            if s.get("pendiente") not in ("sentido", "razon"):
                f.append(f"{sid}: razón fuera del catálogo ({s.get('razon_crudo') or 'vacía'})")
        else:
            _fr = _falla_razon(rz, s.get("etiqueta") or "")
            if _fr:
                f.append(f"{sid}: {_fr}")
            if RAZONES[rz].get("solo_recursos") and not _es_recurso(plan):
                f.append(f"{sid}: {rz} sólo cabe en recursos")
            if RAZONES[rz].get("con_p") and not (s.get("ataca") or s.get("razon_p")):
                f.append(f"{sid}: {rz} tiene que nombrar la proposición (ataca)")
        # (f)
        t = s.get("reitera")
        if t:
            otro = cx.seg_piso.get(t) or por_id.get(t)
            if otro is None:
                f.append(f"{sid}: reitera a {t}, que no existe")
            else:
                # Las anclas son las del PISO (código), no lo que diga el plan.
                _propias = _anclas_seg(cx.seg_piso.get(sid) or s) - _anclas_seg(otro)
                if _propias:
                    f.append(f"{sid}: no puede reiterar a {t}: trae anclas propias "
                             f"({', '.join(sorted(_propias)[:3])}); va «desarrolla»")
        # Lo que obliga a desarrollar: su diferencia, o que el código le
        # retiró la premisa por no tener rastro (`_reparar_organizacion`).
        if s.get("trat") == "desarrolla" and not s.get("diferencia") and not s.get("pendiente") \
                and not s.get("sin_premisa"):
            f.append(f"{sid}: desarrolla sin decir qué diferencia lo obliga")
        # (g) La cita literal del escrito. Los suplidos (S) no la tienen: no
        # salen del escrito sino de la suplencia. Tampoco el segmento del
        # inventario cuya cita no encontró nadie: `reparar` lo marca y lo dice.
        if cx.hay_escrito and not sid.startswith("S") and not s.get("sin_cita"):
            cita = s.get("cita") or ""
            if not cita:
                piso = (cx.seg_piso.get(sid) or {}).get("cita", "")
                cita = piso if cx.escrito.contiene(piso, MIN_PALABRAS_CITA_SEGMENTO) else s.get("cita_plan", "")
            if not cx.escrito.contiene(cita, MIN_PALABRAS_CITA_SEGMENTO):
                f.append(f"{sid}: sin cita literal verificada del escrito (pon 10 a 40 palabras literales en «cita»)")
        d = s.get("dato")
        if d and not cx.fuentes_dato.contiene(d.get("cita", ""), MIN_PALABRAS_CITA):
            f.append(f"{sid}: la cita del dato no está en el expediente")
        if s.get("ataca") and s["ataca"] not in props:
            f.append(f"{sid}: ataca {s['ataca']}, que no está entre las proposiciones")
        # (j) PROCESALES: todas se deciden (arts. 74-V y 174 LA). La única que
        # queda sin estudio es la que una concesión de fondo con mayor
        # beneficio vuelve innecesaria (art. 189; Decisión 1 de David). Si es
        # el CRITERIO el que la deja sin estudiar sin esa concesión, el plan
        # no puede arreglarlo —no decide el sentido—: `reparar` lo avisa
        # arriba y aquí no se rechaza, porque rehacer el plan no lo cambiaría.
        # El adhesivo, con el principal que no prospera, va por (m).
        # Sólo donde rige (amparo directo; sin tipo, rige), como en el árbol.
        if _guarda_procesal(plan.get("tipo_asunto", "")) and _es_procesal(s, cx) \
                and not (sid.startswith("AD") and not cx.alguno_prospera):
            _no_estudia = s.get("trat") == "no_se_estudia" or bool(
                rz and RAZONES[rz]["clase"] == NO_SE_ESTUDIA)
            if clase_sentido(s.get("etiqueta")) != NO_SE_ESTUDIA:
                if _no_estudia:
                    f.append(f"{sid}: una violación procesal no puede quedar sin estudio "
                             f"({rz or s.get('trat')}); arts. 74, fracción V, y 174 de la Ley de Amparo")
            elif cx.alguno_prospera_fondo and rz != "innecesario_mayor_beneficio":
                f.append(f"{sid}: una violación procesal sólo deja de estudiarse con "
                         f"innecesario_mayor_beneficio (art. 189), nunca con {rz or s.get('trat')}")
        # (k) Suplencia confirmada: sin inoperancia de forma a su favor.
        if cx.sup_confirmada() and rz in _RAZONES_FORMA and _de_la_parte_favorecida(sid, cx):
            f.append(f"{sid}: con la suplencia confirmada no se declara inoperante por cómo se planteó ({rz})")
        if cx.sup_confirmada() and rz == "procesal_171_172" and _exime_171(cx) \
                and _de_la_parte_favorecida(sid, cx):
            f.append(f"{sid}: el sujeto de esta suplencia está exento de preparar la violación (art. 171)")
        # (o) Cosa juzgada sólo contra lo vinculado.
        if rz == "cosa_juzgada_amparo_previo":
            pk = s.get("razon_p") or s.get("ataca")
            if not (pk in props and props[pk].get("vinculada_por_ejecutoria")):
                f.append(f"{sid}: cosa juzgada sólo contra una proposición vinculada por la ejecutoria")
        # Decisión 6.
        if s.get("pendiente") == "razon" and s.get("trat") != "desarrolla":
            f.append(f"{sid}: pendiente de razón va «desarrolla»")
        # EL TRATAMIENTO NO PUEDE BORRAR UN ARGUMENTO (revisión del 26-sep-2026).
        # El guion no imprime lo que va a `no_se_expresa_art79` y reduce a una
        # frase lo que va a `no_se_estudia`: si el plan pone ahí un concepto
        # que el criterio DECIDE, el argumento desaparece del estudio sin que
        # nadie lo haya resuelto así (art. 74, fracción II, de la Ley de
        # Amparo: el análisis de TODOS los conceptos).
        if s.get("trat") == "no_se_expresa_art79" and not sid.startswith("S"):
            f.append(f"{sid}: no_se_expresa_art79 es sólo para lo suplido (S); un argumento del "
                     f"escrito siempre se contesta")
        if s.get("trat") == "no_se_estudia" and fijado and clase_sentido(fijado) != NO_SE_ESTUDIA:
            f.append(f"{sid}: el criterio lo decide («{fijado}»): no puede ir a no_se_estudia")
        # (m) Adhesivo sin materia si el principal no prospera. Sólo se exige
        # cuando el criterio de su problema ya dice que no se estudia: si dice
        # otra cosa, el plan no puede cambiarlo (lo avisa `reparar`) y exigirlo
        # aquí haría imposible cualquier plan.
        if sid.startswith("AD") and not cx.alguno_prospera and rz != "adhesivo_sin_materia" \
                and clase_sentido(s.get("etiqueta")) == NO_SE_ESTUDIA:
            f.append(f"{sid}: si el principal no prospera, el adhesivo queda sin materia (adhesivo_sin_materia)")

    # (b, la otra mitad) UN PROBLEMA QUE PROSPERA LO FUNDA AL MENOS UNO DE SUS
    # ARGUMENTOS: si todos van en contra, las etiquetas niegan el sentido del
    # secretario (`reparar` lo devuelve a su sentido y lo propone).
    for pid in problemas_sin_quien_los_funde(segmentos, cx.probs):
        f.append(f"problema {pid}: el secretario lo fijó «{cx.por_pid[pid]['sentido']}» y ninguna "
                 f"etiqueta de sus segmentos lo funda; al menos uno lleva una calificación que "
                 f"prospera (si crees que el problema no prospera, va a propuestas)")

    # (e) Una unidad no mezcla proposición, vicio ni razón (salvo el grupo del
    # secretario, que manda).
    for u in unis:
        us = [por_id[x] for x in u.get("segmentos") or [] if x in por_id]
        us = [s for s in us if s.get("trat") in _EN_UNIDAD and s.get("pendiente") != "sentido"]
        if not us:
            continue
        mezcla = []
        if len({s.get("ataca") for s in us if s.get("ataca")}) > 1:
            mezcla.append("proposiciones")
        if len({s.get("vicio") for s in us}) > 1:
            mezcla.append("vicios")
        # El pendiente de razón no trae razón: no «mezcla» (falsa alarma
        # medida: 722/2025, una unidad de fondo_desestimado con su pendiente).
        if len({s.get("razon") for s in us if s.get("razon")}) > 1:
            mezcla.append("razones")
        if mezcla and not _grupo_comun(u, us, cx):
            f.append(f"{u.get('id')}: mezcla {', '.join(mezcla)} distintas; son unidades distintas")
        if u.get("premisa") and u["premisa"] not in prems:
            f.append(f"{u.get('id')}: su premisa {u['premisa']} no existe")

    # (h)(i) Premisas: fuentes del índice o de la razón, y rastro con cita.
    for m in prems.values():
        for x in m["fuentes"]["tesis"]:
            if not cx.resolver_tesis(x):
                f.append(f"{m['id']}: la fuente «{x[:40]}» no está en el índice ni en la razón del secretario")
        for x in m["fuentes"]["normas"]:
            if not cx.resolver_norma(x):
                f.append(f"{m['id']}: la fuente «{x[:40]}» no está en el índice ni en la razón del secretario")
        if not _premisa_con_rastro(m, cx):
            f.append(f"{m['id']}: premisa sin rastro verificado en la razón del secretario ni en sus fuentes")
        for pk in m.get("responde_a") or []:
            if pk not in props:
                f.append(f"{m['id']}: responde a {pk}, que no existe")

    # Proposiciones: citas verificadas.
    for pr in props.values():
        if pr.get("cita") and not cx.fuentes_p.contiene(pr["cita"], MIN_PALABRAS_CITA):
            f.append(f"{pr['id']}: la cita no está en el acto ni en el contexto")

    # (l) Orden.
    f += _faltas_de_orden(plan, cx, por_id)
    return f


def _es_recurso(plan: dict) -> bool:
    import tipos_asunto as _ta
    return _ta.normalizar(plan.get("tipo_asunto", "")) not in ("", "amparo_directo")


def _de_la_parte_favorecida(sid: str, cx: _Ctx) -> bool:
    """La suplencia se confirma a favor de quien promueve casi siempre; si el
    secretario la confirmó a favor del adherente, son los AD."""
    quien = _sin_acentos(str(cx.suplencia.get("a_favor_de") or "")).lower()
    if "adher" in quien:
        return sid.startswith("AD")
    return not sid.startswith("AD")


def _exime_171(cx: _Ctx) -> bool:
    try:
        import suplencia as _sp
        return bool(_sp._POR_ID.get(cx.suplencia.get("fraccion"), {}).get("exime_171"))
    except Exception:
        return False


# ART. 189 DE LA LEY DE AMPARO (verificado contra el texto vigente, DOF
# 13-03-2025): «se privilegiará el estudio de los conceptos de violación de
# FONDO por encima de los de PROCEDIMIENTO Y FORMA, a menos que invertir el
# orden redunde en un mayor beneficio». Hasta la revisión del 26-sep-2026 el
# código contaba la «forma» del lado del fondo, al revés que la ley. La
# «omisión» (la responsable no se pronunció) no va en ninguna de las dos
# listas: según lo omitido es fondo o forma, y el código no puede saberlo.
_VICIOS_FONDO_189 = {"fondo"}
_VICIOS_TRAS_FONDO_189 = {"procesal", "forma"}


def _secuencias_de_orden(plan: dict, cx: _Ctx, por_id: dict) -> list[tuple]:
    """Los órdenes en que el guion PRESENTA el estudio, que es lo que hay que
    revisar —no una lista que nadie imprime—:
      · el de las unidades: la moderna ordena sus problemas por ahí, y la
        estándar también cuando el orden es de prelación;
      · con el orden «promovente», el de los apartados de la estándar, que
        siguen el número de concepto del escrito (`_apartados_estandar`).
    El plan no sabe con qué forma se escribirá (la forma no entra en la
    clave), así que V0 revisa las dos. Hasta el 26-sep-2026 sólo miraba las
    unidades, y una estándar con la procesal en el primer concepto salía antes
    que el fondo aunque V0 la diera por buena."""
    unis = plan.get("unidades") or []
    seqs = [("unidades", [[por_id[x] for x in u.get("segmentos") or [] if x in por_id]
                          for u in unis])]
    if (plan.get("orden") or {}).get("criterio") != "prelacion":
        # El concepto de cada segmento es el del PISO: lo que diga el plan de
        # él no cuenta (y un plan sin reparar puede no traerlo).
        _p = dict(plan)
        _p["segmentos"] = [dict(s, concepto=(cx.seg_piso.get(s.get("id")) or s).get("concepto") or 0)
                           for s in plan.get("segmentos") or [] if s.get("id")]
        _por = {s["id"]: s for s in _p["segmentos"]}
        aps = _apartados_estandar(_p, _por)
        seqs.append(("apartados", [[s for s in ap["segmentos"]
                                    if s.get("trat") != "no_se_expresa_art79"
                                    and s.get("pendiente") != "sentido"] for ap in aps]))
    return seqs


def _faltas_de_orden(plan: dict, cx: _Ctx, por_id: dict) -> list[str]:
    f = []
    import tipos_asunto as _ta
    _ad = _ta.normalizar(plan.get("tipo_asunto", "")) in ("", "amparo_directo")
    por_que = _sin_acentos((plan.get("orden") or {}).get("por_que") or "").lower()
    principales = [p["id"] for p in cx.probs if p["jerarquia"] == "principal"]
    for que, seq in _secuencias_de_orden(plan, cx, por_id):
        # Cómo se dice en el reintento: el planificador tiene que saber QUÉ
        # orden falló y cómo arreglarlo.
        donde = ("en la lista de unidades" if que == "unidades" else
                 "con orden «promovente» (los apartados de la estándar siguen el número de "
                 "concepto del escrito; si ese orden no cumple, usa «prelacion» y ordena las unidades)")
        primero_de_problema, clase_u = {}, []
        for i, us in enumerate(seq):
            for s in us:
                primero_de_problema.setdefault(s.get("problema_id"), i)
            clase_u.append({s.get("vicio") for s in us})
        # La procedencia primero.
        proc = [i for i, v in enumerate(clase_u) if "procedencia" in v]
        otras = [i for i, v in enumerate(clase_u) if v and "procedencia" not in v]
        if proc and otras and max(proc) > min(otras):
            f.append(f"la procedencia va antes que cualquier otra cosa, {donde}")
        # Cada accesorio después de su principal.
        for p in cx.probs:
            dep = p.get("depende_de") or (principales[0] if principales and p["jerarquia"] == "accesorio"
                                         and p["id"] not in principales else None)
            if dep and p["id"] in primero_de_problema and dep in primero_de_problema \
                    and primero_de_problema[p["id"]] < primero_de_problema[dep]:
                f.append(f"el problema {p['id']} (accesorio) se estudia antes que el {dep}, del que "
                         f"depende, {donde}")
        # En el amparo directo, el fondo antes que el procedimiento y la forma
        # salvo mayor beneficio DICHO (art. 189; Decisión 1 de David: la única
        # excepción es el mayor beneficio, y orden.por_que dice en qué consiste).
        if _ad:
            tras = [i for i, v in enumerate(clase_u) if v & _VICIOS_TRAS_FONDO_189]
            fondo = [i for i, v in enumerate(clase_u) if v & _VICIOS_FONDO_189]
            if tras and fondo and min(tras) < min(fondo) and "beneficio" not in por_que:
                f.append(f"una violación procesal o formal va antes que el fondo sin decir en "
                         f"orden.por_que el mayor beneficio (art. 189 de la Ley de Amparo), {donde}")
    return f


# ═══ PREPARAR: LA CORRIDA COMPLETA ═════════════════════════════════════════

async def preparar(cliente, r, crit, material, contexto: str = "", suplencia=None, *,
                   segs: list = None, contraste=None, tocados=None,
                   conceptos_violacion: str = "", checklist=None) -> tuple:
    """planear → reparar → validar; si V0 falla, UN reintento con la lista de lo
    que falló; si vuelve a fallar, (None, avisos, info): el estudio se escribe
    sin plan (v3) y se dice. Devuelve (plan | None, avisos, info)."""
    fases = r.fases
    if segs is None:
        segs = segmentos_de(fases, _escrito_de(fases),
                            bool(getattr(getattr(r, "encargo", None), "es_recurso", False)))
    if not segs:
        raise PlanNoDisponible("el inventario no trae ningún segmento")
    if not _escrito_de(fases).strip():
        raise PlanNoDisponible("no está el escrito: sin él no hay cita que verificar")
    info = {"intentos": 0, "faltas": []}
    faltas: list = []
    for _ in range(2):
        info["intentos"] += 1
        crudo = await planear(cliente, r, crit, material, contexto, suplencia, segs=segs,
                              contraste=contraste, faltas=faltas,
                              conceptos_violacion=conceptos_violacion, checklist=checklist)
        plan, avisos = reparar(crudo, crit, segs, fases, material, contexto, suplencia,
                               tocados=tocados)
        faltas = validar(plan, crit, segs, fases, material, contexto, suplencia=suplencia,
                         tocados=tocados)
        info["faltas"].append(list(faltas))
        if not faltas:
            return plan, avisos, info
    return None, [f"el plan del estudio no pasó la validación dos veces ({len(faltas)} fallas; "
                  f"la primera: {faltas[0][:160] if faltas else '—'})"], info


def aplicar_razones(plan: dict, razones_segmento: dict) -> dict:
    """DECISIÓN 6 DE DAVID (opción a, 26-sep-2026): el panel pide la razón que
    falta antes de generar. La que el secretario escribió para un segmento
    «pendiente: razon» entra al guion como SUYA y deja de estar pendiente; la
    que no escribió se queda pendiente y el estudio la desarrolla con el
    material y la pone PRIMERO en ADVERTENCIAS. Determinista: no gasta una
    corrida del planificador."""
    plan = copy.deepcopy(plan or {})
    razones = {norm_id(k): _ws(v)[:2000] for k, v in (razones_segmento or {}).items()
               if norm_id(k) and _ws(v)}
    for s in plan.get("segmentos") or []:
        txt = razones.get(s.get("id"))
        if txt and s.get("pendiente") == "razon":
            s["razon_secretario"] = txt
            s["pendiente"] = None
    return plan


def leer_razones_segmento(crudo) -> dict:
    """El campo `razones_segmento` del formulario: JSON {"C3.e": "texto"}. Un
    JSON roto se trata como vacío (no tumba la generación)."""
    d = crudo
    if not isinstance(d, dict):
        try:
            d = json.loads(str(crudo or "").strip() or "{}")
        except Exception:
            return {}
    if not isinstance(d, dict):
        return {}
    return {norm_id(k): _ws(v)[:2000] for k, v in list(d.items())[:60]
            if norm_id(k) and isinstance(v, str) and _ws(v)}


# ═══ LA VISTA: EL GUION POR FORMA ══════════════════════════════════════════
# Presupuesto de extensión por función [H: provisional, sin calibrar; se mide
# contra la Solución de los 24 Kingston antes de volverlo aviso]. Es un TECHO
# por apartado, no una meta: la meta de extensión fue la causa F1 de la
# repetición (w2_final §1.3).
_PALABRAS = {"expone": 220, "aplica_con_dato": 90, "aplica": 45, "remite": 50,
             "desarrolla": 240, "residual": 45, "no_se_estudia": 35, "pendiente": 200,
             "no_se_expresa_art79": 0}


def _q1(plan: dict) -> str:
    import tipos_asunto as _ta
    return _ta.vocabulario_de(plan.get("tipo_asunto") or "amparo_directo")["combate_singular"]


def _apartados_estandar(plan: dict, por_id: dict) -> list[dict]:
    """Apartados de la estándar: por CONCEPTO (la fórmula del oficio), en el
    orden del estudio. Dos conceptos se juntan en un apartado sólo si todos
    sus segmentos contestados están en la MISMA unidad, o si el secretario los
    agrupó: juntar por tema es lo que la propuesta prohíbe."""
    unis = plan.get("unidades") or []
    u_de = {}
    for i, u in enumerate(unis):
        for x in u.get("segmentos") or []:
            u_de.setdefault(x, i)
    por_concepto: dict = {}
    orden_c = []
    for s in plan.get("segmentos") or []:
        k = s.get("concepto") or 0
        if k not in por_concepto:
            por_concepto[k] = []
            orden_c.append(k)
        por_concepto[k].append(s)
    if (plan.get("orden") or {}).get("criterio") == "prelacion":
        def _peso(k):
            ix = [u_de[s["id"]] for s in por_concepto[k] if s["id"] in u_de]
            return (min(ix) if ix else 10 ** 6, k)
        orden_c.sort(key=_peso)
    else:
        orden_c.sort(key=lambda k: (k == 0, k))
    grupos_p = {p["id"]: p.get("grupo") for p in plan.get("problemas") or []}
    apartados: list[dict] = []
    for k in orden_c:
        segs_k = por_concepto[k]
        unis_k = {u_de[s["id"]] for s in segs_k if s["id"] in u_de and s.get("trat") in _EN_UNIDAD}
        grupos_k = {grupos_p.get(s.get("problema_id")) for s in segs_k} - {"", None}
        if apartados:
            prev = apartados[-1]
            misma_unidad = len(unis_k) == 1 and prev["unidades"] == unis_k
            mismo_grupo = len(grupos_k) == 1 and prev["grupos"] == grupos_k
            if misma_unidad or mismo_grupo:
                prev["conceptos"].append(k)
                prev["segmentos"] += segs_k
                continue
        apartados.append({"conceptos": [k], "segmentos": list(segs_k),
                          "unidades": unis_k, "grupos": grupos_k})
    return apartados


def _linea_seg(s: dict, trat: str, extra: str = "", sentidos: dict = None) -> str:
    et = s.get("etiqueta") or "sin sentido fijado"
    partes = [f"{trat.upper().replace('_', ' ')} {s['id']}"]
    if s.get("etiqueta"):
        # LA ETIQUETA DEL ARGUMENTO QUE NO ES LA DE SU PROBLEMA SE DICE CON SU
        # PROBLEMA (plan-4): «infundado dentro del problema 1, fundado» es lo
        # que el estudio tiene que escribir sin volverlo el sentido de todo.
        _fp = (sentidos or {}).get(s.get("problema_id"), "")
        partes.append(f"etiqueta: {et}" + (f" (dentro del problema {s.get('problema_id')}, {_fp})"
                                             if _fp and _fp != et else ""))
    if s.get("razon"):
        partes.append(f"razón: {s['razon']}" + (f"({s['razon_p']})" if s.get("razon_p") else ""))
    if s.get("ataca"):
        partes.append(f"ataca {s['ataca']}")
    if s.get("diferencia") and trat == "desarrolla":
        partes.append(f"diferencia: {s['diferencia']}")
    if s.get("sin_premisa") and trat == "desarrolla":
        partes.append(_RX_SIN_PREMISA)
    if s.get("dato"):
        d = s["dato"]
        partes.append(f"dato ({d.get('fuente')}): «{d.get('cita')}»")
    if s.get("razon_secretario"):
        partes.append(f"razón del secretario para este argumento: «{s['razon_secretario']}»")
    if extra:
        partes.append(extra)
    return "  " + " · ".join(partes)


def _linea_premisa(m: dict, props: dict, uid: str = "") -> str:
    # La proposición, por su síntesis y sin comillas (no es cita del acto).
    resp = ", ".join(f"{pk} (en síntesis: {(props.get(pk) or {}).get('dice', '')})"
                     for pk in m.get("responde_a") or [])
    fu = [f"registro {x}" for x in m["fuentes"]["tesis"]] + list(m["fuentes"]["normas"])
    return ("  EXPONE " + m["id"]
            + (f" · unidad {uid}" if uid else "")
            + (f" · responde a {resp}" if resp else "")
            + (f" · fuentes: {'; '.join(fu)}" if fu else " · fuentes: la razón del secretario")
            + (f" · anclas: {' | '.join(m.get('anclas') or [])}" if m.get("anclas") else ""))


def vista(plan: dict, formato: str = "estandar") -> str:
    """EL GUION. Datos, no prosa: identificadores, etiquetas, razones
    tipificadas, fuentes, anclas y citas verificadas. Nada que el estudio
    pueda copiar como frase hecha (lección del ejemplo que se firma literal).

    ESTÁNDAR: un apartado por concepto —o por los que se juntan de verdad—, en
    el orden del estudio. MODERNA: un apartado por PROBLEMA del secretario, con
    su pregunta, y las unidades como párrafos dentro (w2_final §4.6, §5.5).
    Cada premisa lleva EXPONE una sola vez, en el primer apartado que la usa;
    después se aplica o se remite a ese apartado."""
    import formato_sentencia as _fs
    plan = plan or {}
    moderna = _fs.normalizar(formato) == _fs.MODERNA
    q1 = _q1(plan)
    varios = int(plan.get("n_planteamientos") or 0) >= 2
    segs = plan.get("segmentos") or []
    por_id = {s["id"]: s for s in segs if s.get("id")}
    props = {p["id"]: p for p in plan.get("proposiciones") or [] if p.get("id")}
    prems = {m["id"]: m for m in plan.get("premisas") or [] if m.get("id")}
    unis = plan.get("unidades") or []
    u_de = {}
    for u in unis:
        for x in u.get("segmentos") or []:
            u_de.setdefault(x, u)
    L = ["GUION DEL ESTUDIO — organiza; el sentido es el del criterio del secretario; la prosa es tuya",
         f"FORMA: {'moderna: un apartado por problema, con su pregunta' if moderna else 'estándar: un apartado por ' + q1}"]
    o = plan.get("orden") or {}
    L.append(f"ORDEN: {'prelación lógica' if o.get('criterio') == 'prelacion' else 'el del escrito'}"
             + (f" · por qué: {o['por_que']}" if o.get("por_que") else ""))
    # EL SENTIDO DE CADA PROBLEMA, del secretario (plan-4): la etiqueta de cada
    # argumento es la suya DENTRO de él, y el estudio tiene que ver las dos.
    _sent = [f"{p['id']} {p['sentido']}" for p in plan.get("problemas") or [] if p.get("sentido")]
    if _sent:
        L.append("SENTIDO DE CADA PROBLEMA (del secretario; no se toca): problema "
                 + " · problema ".join(_sent))
    if props:
        # LO QUE DICE CADA PROPOSICIÓN ES LA PARÁFRASIS DEL PLANIFICADOR, no
        # palabras del acto: sin comillas angulares, que en una sentencia dicen
        # «literal» (revisión adversarial de la integración, 26-sep-2026).
        L.append("PROPOSICIONES DEL ACTO (en síntesis del planificador; no son palabras del acto):")
        for p in props.values():
            L.append(f"  {p['id']} · en síntesis: {p['dice']} · {p['caracter']} · {p['relacion']}"
                     + (" · vinculada por ejecutoria" if p.get("vinculada_por_ejecutoria") else ""))
    expuesta: dict = {}          # premisa → apartado donde se expone
    vista_u: dict = {}           # unidad → primer apartado donde aparece
    apartado_de: dict = {}       # segmento → apartado donde se contesta
    objecion_puesta: set = set()
    presupuesto: list[str] = []
    _sentidos_p = {p.get("id"): p.get("sentido") for p in plan.get("problemas") or [] if p.get("sentido")}

    def _render_segs(n_ap: int, lista: list, salida: list) -> int:
        palabras = 0
        # LA OBJECIÓN, UNA VEZ: tras el último argumento de su unidad en el
        # apartado donde la unidad se contesta por primera vez, que es donde
        # pesa (después de la razón decisoria).
        ultimo_de_u = {}
        for s in lista:
            u = u_de.get(s["id"])
            if u is not None and (s.get("trat") or "aplica") in _EN_UNIDAD:
                ultimo_de_u[u.get("id")] = s["id"]
        for s in lista:
            u = u_de.get(s["id"])
            trat = s.get("trat") or "aplica"
            if s.get("pendiente") == "sentido":
                continue                               # van abajo, SIN SENTIDO
            apartado_de.setdefault(s["id"], n_ap)
            extra = []
            if u is not None and trat in _EN_UNIDAD:
                uid = u.get("id")
                m = prems.get(u.get("premisa"))
                vista_u.setdefault(uid, n_ap)
                # ¿DÓNDE VIVE LA RESPUESTA A LA QUE SE REMITE O SE APLICA? La
                # premisa de su unidad se expone UNA vez, en el primer apartado
                # que la usa —sea de esta unidad o de otra que comparte premisa
                # (revisión del 26-sep-2026: una remisión a la premisa que
                # expuso OTRA unidad salía como «aplica» sin destino, y el
                # estudio la volvía a exponer)—. Sin premisa, el destino es el
                # apartado del segmento que reitera o el primero de su unidad.
                if m and m["id"] not in expuesta:
                    salida.append(_linea_premisa(m, props, uid))
                    expuesta[m["id"]] = n_ap
                    palabras += _PALABRAS["expone"]
                if m:
                    dest = expuesta[m["id"]]
                else:
                    dest = apartado_de.get(s.get("reitera") or "") or vista_u[uid]
                if trat == "remite" and dest < n_ap:
                    salida.append(_linea_seg(s, "remite", (
                        f"unidad {uid} → apartado {dest}" + (f" ({m['id']})" if m else "")
                        + (f" · proposición: {', '.join(m.get('responde_a') or [])}"
                           if m and m.get("responde_a") else "")), _sentidos_p))
                    palabras += _PALABRAS["remite"]
                    continue
                if trat == "remite":
                    trat = "aplica"                    # no se remite hacia adelante
                extra.append(f"unidad {uid}")
                if m and dest != n_ap:
                    extra.append(f"premisa {m['id']} ya expuesta en el apartado {dest}: "
                                 f"no se expone otra vez")
                if u.get("objecion") and uid not in objecion_puesta \
                        and vista_u.get(uid) == n_ap and ultimo_de_u.get(uid) == s["id"]:
                    objecion_puesta.add(uid)
                    ob = u["objecion"]
                    extra.append(f"objeción aquí, una vez: {ob.get('de')}" + (
                        f" ({' | '.join(ob.get('anclas') or [])})" if ob.get("anclas") else ""))
            if s.get("pendiente") == "razon":
                salida.append(_linea_seg(s, "desarrolla", " · ".join(extra + ["PENDIENTE DE RAZÓN (ver abajo)"]),
                                         _sentidos_p))
                palabras += _PALABRAS["pendiente"]
                continue
            salida.append(_linea_seg(s, trat, " · ".join(extra), _sentidos_p))
            palabras += _PALABRAS.get("aplica_con_dato" if trat == "aplica" and s.get("dato") else trat, 45)
        return palabras

    if not moderna:
        for n, ap in enumerate(_apartados_estandar(plan, por_id), 1):
            vivos = [s for s in ap["segmentos"] if s.get("trat") != "no_se_expresa_art79"
                     and s.get("pendiente") != "sentido"]
            if not vivos:
                continue
            ets = []
            for s in vivos:
                if s.get("etiqueta") and s["etiqueta"] not in ets:
                    ets.append(s["etiqueta"])
            con = [str(k) for k in ap["conceptos"] if k]
            # UN NÚMERO DE CONCEPTO QUE NO ES DEL ESCRITO NO SE ESCRIBE: sin
            # ordinales en el resumen y sin una cuenta que case, el concepto es
            # el orden de párrafos (`inventario`, `concepto_inferido`), y
            # «abre: concepto de violación 3» ordenaba nombrar en la prosa un
            # tercer concepto que la demanda no tiene (comprobación de la
            # revisión adversarial de la integración, 26-sep-2026). Se abre por
            # los argumentos, como cuando hay uno solo.
            _inferido = any(x.get("concepto_inferido") for x in ap["segmentos"])
            abre = (f"{q1} {' y '.join(con)}" if varios and con and not _inferido
                    else "argumentos " + ", ".join(s["id"] for s in vivos))
            L.append(f"APARTADO {n} · abre: {abre}"
                     + (f" · etiqueta: {'; '.join(ets)}" if ets else "")
                     + (f" · grupo del secretario {next(iter(ap['grupos']))}" if len(ap["grupos"]) == 1 else ""))
            w = _render_segs(n, vivos, L)
            presupuesto.append(f"ap.{n} ≤ {max(120, w)}")
    else:
        probs = plan.get("problemas") or []
        orden_p = []
        for u in unis:
            for x in u.get("segmentos") or []:
                pid = (por_id.get(x) or {}).get("problema_id")
                if pid and pid not in orden_p:
                    orden_p.append(pid)
        for p in probs:
            if p["id"] not in orden_p:
                orden_p.append(p["id"])
        pregunta = {p["id"]: p for p in probs}
        n = 0
        for pid in orden_p:
            lista = [s for s in segs if s.get("problema_id") == pid
                     and s.get("trat") != "no_se_expresa_art79" and s.get("pendiente") != "sentido"]
            if not lista:
                continue
            # Dentro del problema, por unidad (cada unidad, un párrafo), en el
            # orden de las unidades.
            lista.sort(key=lambda s: (unis.index(u_de[s["id"]]) if s["id"] in u_de else 10 ** 6))
            n += 1
            pr = pregunta.get(pid) or {}
            L.append(f"APARTADO {n} · problema {pid}: «{pr.get('pregunta', '')}» · etiqueta: "
                     f"{pr.get('sentido') or 'sin sentido fijado'}")
            w = _render_segs(n, lista, L)
            presupuesto.append(f"ap.{n} ≤ {max(120, w)}")

    pend = [s for s in segs if s.get("pendiente") == "razon"]
    if pend:
        L.append("PENDIENTE DE RAZÓN: " + " · ".join(
            f"{s['id']} ({s.get('diferencia') or 'lo propio del argumento'})" for s in pend)
            + " → desarróllalo con el material sin atribuirle al secretario una razón que no dio; "
              "va PRIMERO en ADVERTENCIAS")
    # LOS SIN CALIFICAR TRAS EL CAMBIO DE SENTIDO (26-sep-2026) no son los que
    # el secretario dejó sin decidir: se desarrollan (ver `bloque`).
    _sinc = {p.get("id") for p in plan.get("problemas") or [] if p.get("sin_calificar")}
    sinc = [s for s in segs if s.get("pendiente") == "sentido" and s.get("problema_id") in _sinc]
    sin = [s for s in segs if s.get("pendiente") == "sentido" and s.get("problema_id") not in _sinc]
    if sinc:
        L.append("SIN CALIFICAR TRAS EL CAMBIO DE SENTIDO: " + ", ".join(s["id"] for s in sinc)
                 + " → desarróllalo con el material y la premisa del principal; va PRIMERO "
                   "en ADVERTENCIAS")
    if sin:
        L.append("SIN SENTIDO FIJADO: " + ", ".join(s["id"] for s in sin)
                 + " → no lo califiques; va en ADVERTENCIAS")
    # LAS UNIDADES QUE PROSPERAN: de ellas, y sólo de ellas, salen los efectos
    # (w2_final §4.4, V6), y cada efecto lleva la marca ⟦U⟧ de la suya (§4.7).
    # Sin esta lista el estudio no tenía cómo saber qué identificador poner.
    prosperan = []
    for u in unis:
        _us = [por_id[x] for x in u.get("segmentos") or [] if x in por_id
               and por_id[x].get("trat") in _EN_UNIDAD and por_id[x].get("pendiente") != "sentido"]
        if _us and any(clase_sentido(x.get("etiqueta")) == PROSPERA for x in _us):
            prosperan.append(f"{u.get('id')} ({', '.join(x['id'] for x in _us)})")
    if prosperan:
        L.append("UNIDADES QUE PROSPERAN (de ellas salen los efectos): " + " · ".join(prosperan))
    # QUÉ SIGNIFICA CADA RAZÓN USADA: su descripción del catálogo, para que el
    # estudio sepa qué respuesta toca sin tener que adivinar un identificador.
    usadas = []
    for s in segs:
        if s.get("razon") and s["razon"] not in usadas:
            usadas.append(s["razon"])
    if usadas:
        L.append("RAZONES (tipo de respuesta; los nombres no se escriben): " + " · ".join(
            f"{k} = {_DESCRIBE_RAZON.get(k, '')}" for k in usadas))
    sup = [s for s in segs if s.get("trat") == "no_se_expresa_art79"]
    if sup:
        L.append("SUPLIDOS SIN BENEFICIO (no se expresan, art. 79, penúltimo párrafo): "
                 + ", ".join(s["id"] for s in sup))
    # SIN TECHO POR APARTADO (26-sep-2026, David: la brevedad no es el
    # objetivo). El presupuesto por función (_PALABRAS) era provisional y sin
    # calibrar, y un «aplica ≤ 45» o «remite ≤ 50» es justo lo que empuja a
    # contestar en genérico el argumento que trae su dato: la v4 contestó con
    # respuesta propia el 79 % de los decisivos, frente al 98 % de la v1. Se
    # sigue calculando por apartado —sirve para medir— pero no entra al guion:
    # la medida es que cada argumento con dato propio reciba la suya.
    return "\n".join(L)


# SIN PREMISA VERIFICADA (26-sep-2026): la premisa de su unidad se retiró
# porque su rastro no estaba en la razón del secretario ni en el material
# (`_reparar_organizacion`). Como el rótulo de abajo, su descripción sólo entra
# cuando el guion lo trae: la v4 de siempre no cambia.
_RX_SIN_PREMISA = "sin premisa verificada"
_SIN_PREMISA_DESC = """
- «sin premisa verificada»: el plan no encontró, palabra por palabra, de dónde
  sale la regla que decide esa unidad. Contéstala desde la razón del
  secretario; si hace falta una regla, sólo con una fuente del material que la
  diga, y sin atribuírsela a él. Los argumentos de una misma unidad comparten
  esa respuesta: se construye una vez, en el primero, y en los demás sólo lo
  propio de cada uno."""

# SIN CALIFICAR TRAS EL CAMBIO DE SENTIDO (26-sep-2026): su descripción sólo
# entra cuando el guion trae el rótulo, para que la v4 de siempre no cambie.
_RX_SIN_CALIFICAR = "SIN CALIFICAR TRAS EL CAMBIO DE SENTIDO:"
_SIN_CALIFICAR_DESC = """
- SIN CALIFICAR TRAS EL CAMBIO DE SENTIDO: el secretario resolvió el principal
  al revés de lo que propuso el motor y la calificación de esos argumentos se
  retiró sin que llegara la nueva. Contéstalos en un apartado propio, en el
  orden de su problema, por lo que ellos mismos plantean y con el sentido y la
  razón del principal como hechos dados; la calificación la propones tú, sin
  presentarla como del secretario, y van PRIMERO en ADVERTENCIAS diciendo que
  debe confirmarla él."""


def bloque(guion: str) -> str:
    """El guion, con lo que significa cada rótulo, tal como entra al prompt
    del estudio en la v4. Descripciones de función; las únicas palabras entre
    comillas son rótulos que NO van a la sentencia."""
    if not str(guion or "").strip():
        return ""
    return f"""
═══════════════════════════════════════════════════════════════════════
EL GUION DEL ESTUDIO — manda la organización
═══════════════════════════════════════════════════════════════════════
Lo que sigue organiza tu estudio: qué apartados hay y en qué orden, qué
argumentos contesta cada uno, dónde se expone cada premisa una sola vez y
dónde sólo se aplica o se remite. Se armó sobre el criterio del secretario y
se verificó contra el expediente; el sentido de cada argumento es el de su
criterio y el guion no lo cambia. Donde este prompt te deja decidir el orden
o los grupos, manda el guion.
- Un apartado por cada APARTADO del guion, en su orden. En la estándar, abre
  nombrando los argumentos que contesta; en la moderna, con la pregunta del
  problema.
- EXPONE: aquí, y sólo aquí, se construye esa premisa —la regla con su fuente
  dentro de la frase, su límite y su apoyo—. Fuentes y anclas dicen de dónde
  sale, no son texto para copiar.
- APLICA: el argumento se contesta con la premisa ya expuesta; si trae dato,
  el dato aparece en su respuesta.
- DESARROLLA: el argumento trae algo que ninguna premisa contesta (su
  diferencia): se dice qué trae de distinto y se construye sólo eso.
- REMITE: se contesta nombrando el apartado donde se expuso la premisa y la
  proposición que decide lo distintivo de este argumento; una remisión que
  sólo manda a lo ya dicho lo deja sin respuesta.
- RESIDUAL: una o dos frases con su calificación y su razón. NO SE ESTUDIA:
  una frase que dice por qué.
- etiqueta: la calificación de ESE argumento. El sentido de cada problema es
  el que fijó el secretario (SENTIDO DE CADA PROBLEMA) y no se toca; dentro de
  él, cada argumento se contesta con su etiqueta: en un problema que prospera
  caben argumentos infundados o inoperantes junto al que lo funda, y en uno
  que no prospera el argumento que tiene razón y no alcanza va como fundado
  pero insuficiente, diciendo por qué no alcanza. Cada apartado abre con la
  calificación que resulta de sus argumentos —si difieren, cuál corresponde a
  cada parte— antes de demostrarla.
- «objeción aquí, una vez»: la objeción seria se contesta en ese punto y en
  ningún otro.
- «razón del secretario para este argumento»: es la razón que decide ese
  argumento; se desarrolla a partir de ella.
- PENDIENTE DE RAZÓN: el criterio no contesta lo propio de ese argumento.
  Contéstalo con el material, sin presentar tu respuesta como razón del
  secretario, y enuméralo PRIMERO en ADVERTENCIAS diciendo que esa respuesta
  debe revisarla él.
- SIN SENTIDO FIJADO: no lo califiques; dilo en ADVERTENCIAS.{_SIN_CALIFICAR_DESC if _RX_SIN_CALIFICAR in guion else ""}{_SIN_PREMISA_DESC if _RX_SIN_PREMISA in guion else ""}
- EXTENSIÓN: la pide lo que cada apartado contesta. Cada argumento con dato
  propio recibe su respuesta, aunque eso alargue el apartado; lo que no se
  hace es repetir. El único techo es el del estudio entero, y no es una meta.
- MARCAS DEL PLAN, además de las de los argumentos y con su mismo formato: el
  párrafo donde construyes una premisa que el guion dice EXPONE lleva en su
  marca el identificador M de esa premisa (junto a los de los argumentos que
  contesta, si los hay); y cada efecto que se sigue de una de las UNIDADES QUE
  PROSPERAN empieza con la marca de esa unidad, su identificador U. Con guion,
  esto sustituye a lo que la regla de las marcas dice del párrafo que expone
  una premisa y de los efectos. Son internas, como las demás.
- Los rótulos y los identificadores del guion (APARTADO, EXPONE, APLICA, C1.a,
  P2, M1, U1…) no se escriben en la sentencia, salvo dentro de las marcas
  entre ⟦ ⟧.
- Si el guion te parece equivocado en algo, síguelo igual y explícalo en el
  último párrafo de ADVERTENCIAS, que empieza con el rótulo
  «DESVIACIONES DEL GUION»; nunca en el cuerpo del estudio.

{guion}
"""


def para_ficha(plan: dict, estado: str = "usado", clave_: str = "", avisos=None) -> dict:
    """Lo que la ficha del proyecto y el evento «listo» guardan del plan: lo
    que la pestaña «Mapa del estudio» necesita (segmento → apartado →
    etiqueta → razón, con la cita) y nada más pesado."""
    if not plan:
        return {"estado": estado, "clave": clave_, "avisos": list(avisos or [])[:12]}

    # RECORTADO (revisión del 26-sep-2026): la ficha se apila con cada proyecto
    # en `estado`, que ya lleva el acto y el escrito. La cita del segmento
    # mide 10-40 palabras por contrato y la razón del secretario puede ser
    # larga; a la pestaña le basta su arranque. El plan entero sigue en la
    # columna `plan`.
    def _corta(v, n):
        return " ".join(str(v).split()[:n]) if isinstance(v, str) else v

    _topes = {"cita": 40, "razon_secretario": 60}
    return {
        "estado": estado, "clave": clave_ or plan.get("clave", ""),
        "version": plan.get("version", ""),
        "segmentos": [{k: _corta(s.get(k), _topes[k]) if k in _topes else s.get(k)
                       for k in ("id", "problema_id", "concepto", "etiqueta", "razon",
                                 "trat", "reitera", "diferencia", "pendiente", "cita",
                                 "razon_secretario", "sin_cita", "sin_premisa")
                       if s.get(k) not in (None, "", []) and s.get(k) is not False}
                      for s in plan.get("segmentos") or []][:120],
        "unidades": [{"id": u.get("id"), "segmentos": u.get("segmentos"), "premisa": u.get("premisa")}
                     for u in plan.get("unidades") or []],
        "premisas": [{"id": m.get("id"), "fuentes": m.get("fuentes"), "anclas": m.get("anclas")}
                     for m in plan.get("premisas") or []],
        "propuestas": list(plan.get("propuestas") or [])[:20],
        "avisos": list(avisos or [])[:12] + list(plan.get("avisos_al_secretario") or [])[:12],
    }


# ═══ EL ESTADO EN LA FILA (taller_sesiones.plan), EN FUNCIONES PURAS ═════════
# Regla de la casa ([[estado-entre-workers]]): con gunicorn -w 2 nada que deba
# sobrevivir a una petición vive en memoria. La columna `plan` guarda:
#   {rev, version, huella, corridas, pedido_clave,
#    planes: {clave: {estado, desde, latido, plan, avisos, segundos}}}
# `rev` es el turno del compare-and-set: main.py escribe SÓLO si la fila
# sigue en el `rev` que leyó (si no, relee y vuelve a aplicar la función).
# Cada clave tiene su casilla, así que un plan viejo que termina último no
# pisa al nuevo: escribe en la suya.

_MAX_CASILLAS = 6


def _doc_vacio(huella: str) -> dict:
    return {"rev": 0, "version": PLAN_VERSION, "huella": huella, "corridas": 0,
            "pedido_clave": "", "planes": {}}


def _doc(doc, huella: str) -> dict:
    """El documento de la fila, o uno nuevo si es de otro adelanto (el tope
    de corridas y los planes son por adelanto: rehacerlo empieza de cero)."""
    d = copy.deepcopy(doc) if isinstance(doc, dict) else None
    if not d or d.get("huella") != huella:
        nuevo = _doc_vacio(huella)
        nuevo["rev"] = int((d or {}).get("rev") or 0)
        return nuevo
    d.setdefault("planes", {})
    d.setdefault("corridas", 0)
    return d


def casilla_abandonada(c: dict, ahora: float) -> bool:
    if not isinstance(c, dict) or c.get("estado") != "en_curso":
        return False
    try:
        ultimo = float(c.get("latido") or c.get("desde") or 0)
    except (TypeError, ValueError):
        ultimo = 0.0
    return ahora - ultimo > PLAN_ABANDONADO_S


def fila_pedir(doc, clave_: str, huella: str, ahora: float,
               tope: int = TOPE_CORRIDAS) -> tuple:
    """(doc nuevo, decisión). Decisión: «listo» · «en_curso» (ya corre esa
    clave) · «fallo» (ya se calculó y falló: nunca se recalcula una clave ya
    calculada) · «tope» (cuatro corridas en esta sesión) · «lanzar» (se
    reservó la corrida: quien llama la corre)."""
    d = _doc(doc, huella)
    c = d["planes"].get(clave_)
    d["pedido_clave"] = clave_
    if isinstance(c, dict):
        if c.get("estado") == "listo":
            return d, "listo"
        if c.get("estado") == "fallo":
            return d, "fallo"
        if c.get("estado") == "en_curso" and not casilla_abandonada(c, ahora):
            return d, "en_curso"
    if int(d.get("corridas") or 0) >= tope:
        return d, "tope"
    d["corridas"] = int(d.get("corridas") or 0) + 1
    d["planes"][clave_] = {"estado": "en_curso", "desde": ahora, "latido": ahora}
    _podar(d, clave_)
    return d, "lanzar"


def fila_latido(doc, clave_: str, huella: str, ahora: float):
    d = _doc(doc, huella)
    c = d["planes"].get(clave_)
    if not isinstance(c, dict) or c.get("estado") != "en_curso":
        return None, "nada"
    c["latido"] = ahora
    return d, "latido"


def fila_resultado(doc, clave_: str, huella: str, plan, avisos, segundos: float,
                   ahora: float, reintentable: bool = False) -> tuple:
    """Guarda el resultado de SU clave. Si la fila ya es de otro adelanto, no
    escribe nada (lo calculado es de otro asunto).

    «fallo» es lo que V0 rechazó dos veces: determinista, no se recalcula.
    Un error del proveedor (red, cuota, un 500) es otra cosa: se guarda como
    «error» y el pedido siguiente lo vuelve a intentar —cuenta corrida—, para
    que un tropiezo pasajero no deje sin plan ese criterio toda la sesión."""
    if not isinstance(doc, dict) or doc.get("huella") != huella:
        return None, "otro_adelanto"
    d = _doc(doc, huella)
    estado = "listo" if plan else ("error" if reintentable else "fallo")
    d["planes"][clave_] = {"estado": estado, "plan": plan or None,
                           "avisos": list(avisos or [])[:20], "segundos": round(segundos, 1),
                           "hecho": ahora}
    _podar(d, clave_)
    return d, estado


def _podar(d: dict, conservar: str) -> None:
    planes = d.get("planes") or {}
    if len(planes) <= _MAX_CASILLAS:
        return
    viejas = sorted((k for k in planes if k != conservar),
                    key=lambda k: float((planes[k] or {}).get("hecho") or (planes[k] or {}).get("desde") or 0))
    for k in viejas[:len(planes) - _MAX_CASILLAS]:
        planes.pop(k, None)


def estado_para_pantalla(doc, huella: str, ahora: float, clave_: str = "") -> dict:
    """Lo que devuelve GET /taller/plan: {estado, clave, plan, avisos}. Sin
    fila, o de otro adelanto: «sin_plan». Una corrida sin latido: «fallo»."""
    if not isinstance(doc, dict) or doc.get("huella") != huella:
        return {"estado": "sin_plan", "clave": "", "plan": None, "avisos": []}
    k = clave_ or str(doc.get("pedido_clave") or "")
    c = (doc.get("planes") or {}).get(k)
    if not isinstance(c, dict):
        return {"estado": "sin_plan", "clave": k, "plan": None, "avisos": []}
    est = c.get("estado")
    if est == "en_curso" and casilla_abandonada(c, ahora):
        est = "fallo"
    if est == "error":                  # la pantalla deja de esperar; el
        est = "fallo"                   # próximo pedido lo reintenta
    return {"estado": est if est in ("listo", "en_curso", "fallo") else "sin_plan",
            "clave": k, "plan": c.get("plan") if est == "listo" else None,
            "avisos": list(c.get("avisos") or []),
            "corridas": int(doc.get("corridas") or 0), "tope": TOPE_CORRIDAS}
