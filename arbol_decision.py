"""LA SUERTE DE LOS ACCESORIOS LA DICTA EL PRINCIPAL — en los tres modos.

POR QUÉ EXISTE. Amparo directo 93/2026, 22-sep-2026. David: «la lógica del
planteamiento principal no domina la suerte de los planteamientos secundarios
en automático (…) si cambio de sentido o el sentido de la resolución principal
es uno, los accesorios caen por su propio peso cuando tienen estrecha relación
o cuando a ningún fin práctico produce su análisis».

LO QUE HABÍA. La sustracción de materia vivía en `modos_decision.repartir` y
sólo en el modo GLOBAL, sólo cuando el principal es fundado. En el modo por
problema —el que se usa al corregir un tema— cada planteamiento era una isla.
Y `depende_de`, que la fase 3 calcula para cada problema, no lo leía nadie.

LA REGLA, en las dos direcciones:

  · EL PRINCIPAL PROSPERA y alcanza → los accesorios que dependen de él quedan
    SIN MATERIA («innecesario»): se dice y se dice por qué.
  · EL PRINCIPAL NO PROSPERA → los accesorios que descansaban en su premisa
    caen con él: son INOPERANTES, porque parten de lo que ya se desestimó
    («la Sala debía estudiar los alegatos contra el crédito» presupone que el
    crédito estaba en la litis, y eso acaba de negarse).

LAS DOS DIRECCIONES NO SON LA MISMA (26-sep-2026, paso 2 del plan que aprobó
David). «Depende» se escribía para las dos y se aplicaba a las dos, y no son
simétricas: que un accesorio quede sin materia si el principal PROSPERA no
dice nada de si cae cuando el principal NO prospera. Medido en el ADC
722/2025, cuatro corridas, v1 y v2 por igual: el principal —¿la falta de
inscripción registral hacía improcedente la acción?— se desestimó, y el
segundo concepto (la condena comprendió conceptos distintos de los reclamados
y pactados: congruencia) y el tercero (costas de segunda instancia: art. 136,
agravios fundados, sin temeridad) salieron «inoperantes porque descansan en
la premisa» registral, que ninguno de los dos contiene. Si la acción fuera
improcedente, la condena y las costas caerían, y por eso la fase 3 escribió
`depende_de`; pero con la acción procedente, su causa de pedir es OTRA y el
engrose real los contestó uno por uno. Con la calificación del motor en la
mano —infundados, con su razón—, el árbol la tiraba y el estudio los
despachaba en un párrafo sin tesis: omisión de estudio, la falta más cara.

LA REGLA DESDE ENTONCES, para la vía en que el principal NO prospera: un
accesorio CAE con el principal sólo si su argumento DA POR CIERTA la premisa
desestimada, y eso se VERIFICA (`presupuesto`): el motor lo declara en la
lista (`presupone`, con las palabras del propio planteamiento donde la da por
cierta), esas palabras están en lo que la fase 3 resumió que se combate en ese
problema, comparten un ancla con el principal, y el accesorio no plantea nada
más por su cuenta (`causa_propia`). Si falta cualquiera de las cuatro cosas,
se estudia con su calificación PROPIA —la que el motor le propuso sabiendo
que el principal no prosperaba, o la que escribió para esta vía— y se le dice
al secretario. Ni `depende_de`, ni `relacion: depende`, ni una suerte
«inoperante» sin la cita bastan: son la palabra del motor, no la prueba. Y una
suerte «infundado» para esta vía es una calificación de fondo: se estudia con
su razón, no se le pega la fórmula de la caída (se la pegaba, y el estudio la
despachaba igual).

Y LOS ESCAPES, que son los de siempre y uno nuevo:
  · lo que el secretario marcó a mano (`tocado`) no se toca;
  · el tema DISTINTO —el motor lo declaró ajeno a la suerte del principal—;
  · el que pide MAYOR BENEFICIO que lo concedido en el principal;
  · el que no depende de nada: se estudia por su cuenta.

LA GUARDA PROCESAL (26-sep-2026, decisión 1 de David). En amparo directo una
VIOLACIÓN PROCESAL nunca queda sin materia, innecesaria ni caída con el
principal: los artículos 74, fracción V, y 174 de la Ley de Amparo mandan
decidirlas todas. La ÚNICA excepción es que un problema de FONDO prospere con
mayor beneficio que la reposición (artículo 189): entonces la procesal queda
innecesaria por mayor beneficio y su razón lo dice con el 189. Si lo que
prospera es una procesal, las demás procesales se deciden igual y el fondo que
depende del principal queda sin materia; el que no declara dependencia no se
toca: se le señala al secretario. Si el principal NO prospera, la procesal se
decide con la calificación que el motor le escribió para esa vía, razonada y
sin la fórmula de la caída; si no le escribió ninguna, conserva la suya y se
avisa. Las piezas de la regla viven en `violacion_procesal`, que también usa
`modos_decision.repartir`.

DE DÓNDE SALE LA DEPENDENCIA. De tres fuentes, en este orden: la suerte
condicional que la fase 5 escribe por tema (`si_prospera` / `si_no_prospera`,
con su razón), la `relacion` que declara, y el `depende_de` de la fase 3. No
hay heurística de texto: una palabra dentro de una pregunta no prueba
dependencia, y este proyecto ya sabe qué pasa con las heurísticas de una
palabra. Lo único que se lee con palabras es la CITA que el motor aporta para
la caída, y sólo para comprobar que existe donde dice: nunca para deducir que
cae.

Un solo módulo, una sola regla, aplicada en tres sitios: al proponer (para que
la pantalla ya enseñe la suerte), en /taller/reparto (cuando el secretario
cambia el principal) y al resolver (para que el documento la lleve).
"""
from __future__ import annotations

INNECESARIO = "innecesario"
INOPERANTE = "inoperante"
# Donde `main._taller_armar_criterio` corta el texto del problema. Lo que llega
# recortado se casa con su problema de la fase 3 por esos primeros caracteres.
CORTE_PROBLEMA = 400


def clave_problema(t) -> str:
    """LA CLAVE CON QUE SE COMPARA UN PROBLEMA CON OTRO, en todas las puertas
    (revisión adversarial de la integración, 26-sep-2026). El criterio armado
    (`main._taller_armar_criterio`) recorta el problema a `CORTE_PROBLEMA`;
    /taller/reparto, las propuestas guardadas, los tocados y la fase 3 lo traen
    entero. Comparar el entero con el recortado fallaba en silencio con los
    planteamientos largos —16 de 157 sesiones reales tienen alguno—: el accesorio
    que el secretario marcó a mano dejaba de ser suyo y el motor lo recalificaba,
    y el principal largo perdía su propuesta y su «alcanza». Por eso TODA
    comparación de problemas pasa por aquí: el árbol, la recalificación, la
    pantalla de /taller/recalificar y el desenlace. Un solo sitio, un solo
    corte."""
    return str(t or "")[:CORTE_PROBLEMA]

# Lo que un accesorio pide y que NO se declara innecesario aunque el principal
# prospere: da más de lo concedido. Es la misma lista de modos_decision, que
# sigue siendo su dueño; se lee de ahí para que no haya dos.
try:
    from modos_decision import _pide_mas as pide_mas   # noqa: F401
except Exception:                                      # pragma: no cover
    def pide_mas(problema: str) -> bool:
        return False

_SENTIDOS = ("fundado", "esencialmente_fundado", "sustancialmente_fundado",
             "parcialmente_fundado", "fundado_insuficiente", "infundado",
             "inoperante", "inatendible", "ineficaz", "sin_materia", INNECESARIO)


def _get(x, k, d=""):
    if isinstance(x, dict):
        return x.get(k, d)
    return getattr(x, k, d)


def _set(x, k, v):
    if isinstance(x, dict):
        x[k] = v
    else:
        setattr(x, k, v)


def _texto(p) -> str:
    return p if isinstance(p, str) else str((p or {}).get("pregunta") or p)


def _prospera(s: str) -> bool:
    try:
        import tipos_asunto as _ta
        return _ta.prospera(s)
    except Exception:
        s = (s or "").lower()
        return "fundad" in s and "insuficien" not in s


def _sentido_valido(s: str) -> str:
    s = str(s or "").strip().lower().replace(" ", "_")
    if s in ("sin_materia", "queda_sin_materia"):
        return INNECESARIO
    return s if s in _SENTIDOS else ""


def _vp():
    """El módulo de la violación procesal, o None si no carga. Sin él la
    guarda se apaga en vez de tumbar el reparto: es lo que pasaba ayer."""
    try:
        import violacion_procesal as _m
        return _m
    except Exception:                                   # pragma: no cover
        return None


def _decide(s: str) -> bool:
    """¿Esta calificación DECIDE el planteamiento? «innecesario» y «sin
    materia» no deciden: dicen por qué no se entra."""
    s = str(s or "").strip().lower().replace(" ", "_")
    return bool(s) and s not in (INNECESARIO, "sin_materia", "queda_sin_materia")


# El arranque con que este módulo escribe la caída con el principal. El
# estudio (`fase6_estudio`, al armar el criterio) lo reconoce por ESTE texto y
# lo redacta como «cae con el principal: no se estudia de fondo». Una procesal
# que vuelve de la pantalla con él —un reparto anterior al 26-sep— saldría sin
# decidir aunque su calificación diga «inoperante».
CAE_CON_PRINCIPAL = "Descansa en la premisa que se desestimó"
# Y el de la sustracción de materia, que escriben este módulo y
# `modos_decision.repartir`.
SIN_MATERIA = "Dado el sentido del estudio del problema principal, queda sin materia"


def _cae_con_principal(c) -> bool:
    """¿Viene calificado como caído con el principal? La fórmula Y una de las
    calificaciones con que el árbol la escribe; con otra calificación la
    fórmula es un resto, no una caída."""
    return (str(_get(c, "razonamiento", "") or "").startswith(CAE_CON_PRINCIPAL)
            and str(_get(c, "sentido", "") or "").strip().lower()
            in (INOPERANTE, "infundado", "ineficaz", "inatendible"))


# ═══ LA SUERTE CONDICIONAL, LEÍDA DE LA LISTA DE COMPROBACIÓN ══════════════

def entrada_de(checklist: list, numero: int, problema: str = "") -> dict:
    """La entrada de la lista de comprobación del problema `numero` (1..n)."""
    for c in (checklist or []):
        if not isinstance(c, dict):
            continue
        try:
            if int(c.get("numero")) == numero:
                return c
        except (TypeError, ValueError):
            pass
    if problema:
        _p = problema.strip().lower()[:60]
        for c in (checklist or []):
            if isinstance(c, dict) and _p and _p in str(c.get("tema") or "").lower():
                return c
    return {}


import re as _re

# «Fundado en lo conducente: …», «Infundado: al no formar parte…»,
# «Inoperante, porque…», «Queda sin materia»: el sentido va al principio de la
# suerte escrita en prosa, y el resto es la razón.
_RX_SUERTE = _re.compile(
    r"^\W*(?:(?:es|resulta|ser[íi]a|quedar[íi]a|queda)\s+)?"
    r"((?:esencialmente|sustancialmente|parcialmente)\s+fundad[oa]s?|"
    r"fundad[oa]s?\s+(?:pero|aunque)\s+insuficiente|"
    r"fundad[oa]s?|infundad[oa]s?|inoperantes?|ineficaz|ineficaces|inatendibles?|"
    r"innecesari[oa]s?|sin\s+materia)\b\s*(?:en\s+lo\s+conducente)?[\s:,.;—–-]*(.*)$",
    _re.I | _re.S)


def _leer_suerte(texto: str) -> tuple:
    """(sentido, razón) leídos de una suerte escrita en prosa; ('', '') si no
    empieza por una calificación."""
    m = _RX_SUERTE.match(str(texto or "").strip())
    if not m:
        return "", ""
    cab = m.group(1).lower()
    cab = _re.sub(r"\s+", " ", cab)
    if "insuficiente" in cab:
        s = "fundado_insuficiente"
    elif cab.startswith(("esencialmente", "sustancialmente", "parcialmente")):
        s = cab.split()[0] + "_fundado"
    elif cab.startswith("fundad"):
        s = "fundado"
    elif cab.startswith("infundad"):
        s = "infundado"
    elif cab.startswith("inoperante"):
        s = INOPERANTE
    elif cab.startswith("ineficac"):
        s = "ineficaz"
    elif cab.startswith("inatendible"):
        s = "inatendible"
    else:
        s = INNECESARIO
    return s, m.group(2).strip()


def _suerte(entrada: dict, clave: str, sentido_motor: str = "") -> tuple:
    """(sentido, razón) de `si_prospera` / `si_no_prospera`.

    Primero lo estructurado que la fase 5 escribe desde hoy. Para las listas
    de antes —`con_propuesta` / `con_alternativa`, en prosa y referidas a la
    vía del motor— se traduce: si la propuesta del motor prosperaba,
    `con_propuesta` es la suerte «si prospera» y `con_alternativa` la otra;
    si no, al revés. Medido en el 93/2026: el motor marcó el tema 2 como
    DISTINTO y a la vez escribió que, en la vía contraria, «al no formar parte
    de la litis el crédito fiscal, los alegatos no podían ampliarla»: su
    suerte depende del principal aunque la etiqueta diga lo contrario. La
    suerte escrita es más concreta que la etiqueta, y manda.
    """
    s = entrada.get(clave)
    if isinstance(s, dict):
        return _sentido_valido(s.get("sentido")), str(s.get("razon") or "").strip()
    if isinstance(s, str) and s.strip():
        return _leer_suerte(s)
    if not sentido_motor:
        return "", ""
    motor_prospera = _prospera(sentido_motor)
    quiero_prospera = (clave == "si_prospera")
    fuente = "con_propuesta" if (motor_prospera == quiero_prospera) else "con_alternativa"
    txt = str(entrada.get(fuente) or "").strip()
    if not txt or txt.upper().startswith("SIN DETERMINAR"):
        return "", ""
    return _leer_suerte(txt)


def relacion_de(problema, principal_num: int, entrada: dict) -> str:
    """'distinto' | 'depende' | '' (no declarada)."""
    if entrada.get("tema_distinto"):
        return "distinto"
    rel = str(entrada.get("relacion") or "").strip().lower()
    if rel in ("distinto", "independiente"):
        return "distinto"
    if rel in ("depende", "dependiente", "accesorio"):
        return "depende"
    if isinstance(problema, dict):
        dep = problema.get("depende_de")
        try:
            if dep is not None and int(dep) == int(principal_num):
                return "depende"
        except (TypeError, ValueError):
            pass
    return ""


# ═══ ¿SU ARGUMENTO DA POR CIERTA LA PREMISA DEL PRINCIPAL? ════════════════
#
# La prueba de que un accesorio cae con el principal (26-sep-2026). No se
# deduce: el motor la DECLARA en la lista de comprobación y aquí sólo se
# comprueba que lo declarado existe. La fase 5 escribe, para cada accesorio
# cuyo argumento da por cierta la premisa del principal,
#
#     "presupone": {"premisa": "…", "cita": "…", "causa_propia": null | "…"}
#
# y `cita` son palabras literales de «Se combate diciendo» de ESE problema
# —el `combate` de la fase 3, que es lo que el motor tiene delante y lo que
# llega aquí con el problema—. Se verifica que:
#   1. la cita tiene cuerpo (cuatro palabras o más);
#   2. está, tal cual, en lo que se combate en ese problema (o en su pregunta);
#   3. comparte al menos un ancla con el principal (su pregunta, lo que se
#      combate en él o lo que resolvió el órgano): si no, las palabras citadas
#      son del accesorio, pero la premisa no es la del principal;
#   4. no hay `causa_propia`: si el accesorio plantea además algo por su
#      cuenta, eso se contesta aunque la premisa caiga, y el problema se
#      estudia (la parte que descansa en la premisa se contesta remitiendo a
#      ella: es la técnica que David aprobó para una razón mixta).
#
# Las anclas son un filtro de coherencia sobre lo que el motor ya citó, no un
# detector: sin `presupone` no hay caída, compartan las palabras que compartan.

_CAIDA = (INOPERANTE, "ineficaz", "inatendible")

# Palabras que cualquier problema del taller comparte con cualquier otro: no
# anclan una premisa. Vocabulario del oficio y de las partes, no del asunto.
_VACIAS = frozenset("""
a al ante bajo con contra de del desde durante en entre hacia hasta mediante
para por segun sin sobre tras el la los las lo un una unos unas y e o u ni que
se su sus es son fue fueron ser sido era esta este esto estas estos esa ese eso
esas esos dicha dicho dichos dichas cual cuales quien cuyo cuya misma mismo
mismos mismas tambien ya le les otra otro otros otras no si como mas pero aun
porque pues tanto cuando donde mientras aunque pese dado dada cada ambos ambas
todo toda todos todas solo sola asi
debia debio debe deben podia puede tenia tuvo hacer hizo sostiene afirma alega
aduce considera estima estimo resolvio declaro determino confirmo modifico
parte partes quejosa quejoso actora actor demandada demandado recurrente
tercero interesado responsable autoridad sala tribunal juez jueza juzgado
organo instancia sentencia resolucion reclamada recurrida acto actos concepto
conceptos violacion violaciones agravio agravios argumento argumentos
planteamiento planteamientos omitio omitir omision analizar analizo analisis
examinar examino estudio estudiar valorar valoro valoracion considerar
considero debido debida indebida indebidamente correcta correctamente legal
legalidad ilegal fundamentacion motivacion fundo motivo fundada motivada
articulo articulos constitucional constitucionales constitucion ley codigo
pruebas prueba derecho derechos procedencia improcedente procedente
respecto relacion relativo relativa conforme caso manera forma modo termino
terminos virtud cuenta principio principios estado juicio procedimiento
""".split())


def _plano(t) -> str:
    """Minúsculas, sin acentos, sin puntuación: para comparar lo citado con
    lo resumido sin que una tilde o una coma lo impidan."""
    import unicodedata
    t = unicodedata.normalize("NFD", str(t or "").lower())
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    return " ".join(_re.findall(r"[a-z0-9ñ]+", t))


def _anclas(t) -> set:
    return {w for w in _plano(t).split()
            if len(w) >= 4 and not w.isdigit() and w not in _VACIAS}


def _textos_de(p, *claves) -> str:
    if isinstance(p, dict):
        return " ".join(str(p.get(k) or "") for k in claves)
    return str(p or "")


def _vacio(x) -> bool:
    return (x is None or (isinstance(x, str) and _plano(x) in
            ("", "null", "none", "ninguna", "ninguno", "no", "nada", "n a")))


def presupuesto(entrada: dict, accesorio, principal) -> dict:
    """¿El accesorio da por cierta la premisa del principal, y consta?

    `entrada`: su entrada de la lista de comprobación; `accesorio` y
    `principal`: los dicts de la fase 3 (o sus preguntas). Devuelve
    {"verificado": bool, "motivo": str, "cita": str, "premisa": str,
     "causa_propia": str}. `motivo` dice por qué no, para el aviso y para
    medir: "sin_declarar", "causa_propia", "cita_corta", "cita_extensa",
    "cita_no_consta",
    "cita_ajena_al_principal"; "" cuando se verifica.
    """
    pre = (entrada or {}).get("presupone")
    if isinstance(pre, str) and not _vacio(pre):
        pre = {"cita": pre}
    out = {"verificado": False, "motivo": "sin_declarar", "cita": "",
           "premisa": "", "causa_propia": ""}
    if not isinstance(pre, dict):
        return out
    cita = str(pre.get("cita") or "").strip().strip("«»“”\"'").strip()
    out["cita"] = cita
    out["premisa"] = str(pre.get("premisa") or "").strip()
    causa = pre.get("causa_propia")
    if not _vacio(causa):
        out["causa_propia"] = str(causa).strip()
        out["motivo"] = "causa_propia"
        return out
    pc = _plano(cita)
    if len(pc.split()) < 4:
        out["motivo"] = "cita_corta"
        return out
    # NI EL PLANTEAMIENTO ENTERO (integración, 26-sep-2026): medido sobre 12
    # asuntos, las dos citas de `presupone` que dio el motor eran el párrafo
    # entero de lo que se combate. Eso «consta» siempre y no prueba dónde da por
    # cierta la premisa: la prueba es un pasaje, no el todo.
    _n_cit = len(pc.split())
    _n_com = max((len(_plano(_textos_de(accesorio, k)).split()) for k in ("combate", "pregunta")),
                 default=0)
    if _n_cit > 60 or (_n_com >= 15 and _n_cit >= 0.8 * _n_com):
        out["motivo"] = "cita_extensa"
        return out
    # PALABRAS ENTERAS Y DENTRO DE UN SOLO TEXTO (revisión del 26-sep-2026):
    # sin los espacios de los bordes, «scindida es la obligada principal»
    # constaba dentro de «la escindida es la obligada principal», y con los
    # dos campos pegados una cita podía empezar en lo que se combate y acabar
    # en la pregunta. La cita empieza y acaba donde empieza y acaba una
    # palabra, y está entera en uno de los dos.
    if not any(f" {pc} " in f" {_plano(_textos_de(accesorio, k))} "
               for k in ("combate", "pregunta")):
        out["motivo"] = "cita_no_consta"
        return out
    if not (_anclas(cita) & _anclas(_textos_de(principal, "pregunta", "combate", "resolvio"))):
        out["motivo"] = "cita_ajena_al_principal"
        return out
    out["verificado"] = True
    out["motivo"] = ""
    return out


# ═══ LA REGLA ══════════════════════════════════════════════════════════════

def aplicar(problemas: list, criterios: list, checklist: list = None,
            propuestas: list = None, tocados: set = None,
            sentido_motor: str = "", tipo_asunto: str = "",
            recalificadas: dict = None, huella_adelanto: str = "",
            global_dictado: bool = False, huella_premisa: str = "") -> tuple:
    """Ajusta EN SITIO el sentido de los accesorios que siguen al principal.

    `problemas`: los dicts de la fase 3, en su orden (pregunta, jerarquia,
    depende_de, clase). `criterios`: dicts o `Criterio` con problema/sentido/
    razonamiento/jerarquia; los que el secretario marcó llevan `tocado` True
    o están en `tocados`. `checklist`: la lista de la fase 5. `propuestas`:
    lo que propuso el motor (para `alcanza` y, en la guarda procesal, para
    devolverle a una procesal la calificación que el motor le dio).
    `tipo_asunto`: la guarda procesal es del amparo directo (sin tipo, rige).

    EL CAMBIO DE SENTIDO (David, 26-sep-2026: «si cambio sentido hay que
    tumbar y regenerar con la premisa del cambio de sentido»). Si el principal
    va por la vía CONTRARIA a la que propuso el motor, los accesorios
    relacionados que el secretario no tocó y cuya calificación sería la de la
    otra vía se TUMBAN —sentido y razón vacíos— y su `detalle` lleva
    `"recalificar": True`, `"de": "por_recalificar"` y `"clave_recalificar"`
    (ver `recalificar.py`). `recalificadas`: lo que el motor recalificó con la
    premisa —una casilla {"clave", "resultados"} o la rama entera de la fila,
    {clave: casilla}—; se aplica SÓLO la de la misma clave (`de:
    "recalificada"`). `huella_adelanto` entra en esa clave, y `huella_premisa`
    (`recalificar.huella_premisa`: la suplencia confirmada y el contexto que ve
    el prompt de la recalificación) también. `global_dictado`:
    el sentido del asunto lo dictó el secretario en el modo global; su brocha
    es su palabra y no se recalifica nada.

    Devuelve (avisos, detalle): `detalle` es {problema: {"de": ..., "por_que":
    ...}} para que la pantalla diga de quién es cada calificación. Cuando
    actúa la guarda procesal, la entrada lleva además "guarda": "procesal"
    (se decide) o "mayor_beneficio_189" (la única excepción).
    """
    avisos: list = []
    detalle: dict = {}
    if not criterios:
        return avisos, detalle
    # TODA COMPARACIÓN DE PROBLEMAS, POR LA MISMA CLAVE RECORTADA
    # (`clave_problema`). El criterio armado recorta el problema a 400
    # caracteres (`main._taller_armar_criterio`) y /taller/reparto, las
    # propuestas guardadas, los tocados y la fase 3 no. Antes sólo `por_texto`
    # casaba por el recorte (81fe4a0): el accesorio largo que el secretario
    # marcó no era «suyo» en los gemelos —se tumbaba y lo recalificaba el
    # motor, la máquina cambiando su sentido—, y el principal largo perdía su
    # propuesta (`_pr_p`, `_motor_de`) y su «alcanza» (revisión adversarial de
    # la integración, 26-sep-2026). Las claves de `detalle` siguen siendo el
    # texto de cada criterio tal como llegó: cada puerta lee el suyo.
    _kp = clave_problema
    _tocados = {_kp(t) for t in (tocados or [])}
    for c in criterios:
        if _get(c, "tocado", False):
            _tocados.add(_kp(_get(c, "problema", "")))

    textos = [_texto(p) for p in (problemas or [])]
    por_texto: dict = {}
    for _i, (_t3, _p3) in enumerate(zip(textos, problemas or [])):
        por_texto.setdefault(_kp(_t3), (_i + 1, _p3))

    principal = next((c for c in criterios
                      if str(_get(c, "jerarquia", "")).lower() == "principal"), None)
    if principal is None:
        # Sin marca de jerarquía, el primero de la fase 3 es el principal.
        principal = next((c for c in criterios
                          if textos and _kp(_get(c, "problema", "")) == _kp(textos[0])),
                         criterios[0])
    p_txt = str(_get(principal, "problema", ""))
    p_num = por_texto.get(_kp(p_txt), (1, None))[0]
    p_sent = str(_get(principal, "sentido", "")).strip().lower()
    if not p_sent:
        return avisos, detalle
    pros = _prospera(p_sent)
    detalle[p_txt] = {"de": "principal", "por_que": ""}

    # ¿ALCANZA? Un principal fundado que sólo repone el procedimiento no deja
    # sin materia lo que pide el fondo. La propuesta lo dice cuando lo sabe.
    alcanza = True
    for pr in (propuestas or []):
        if _kp(_get(pr, "problema", "")) == _kp(p_txt) and _get(pr, "alcanza", True) is False:
            alcanza = False
    if pros and not alcanza:
        avisos.append(
            "NO SE APLICÓ LA SUSTRACCIÓN DE MATERIA: el motor no pudo afirmar "
            "que lo fundado del principal alcance para resolver el asunto. Los "
            "accesorios se estudian.")

    # ═══ LA GUARDA PROCESAL (26-sep-2026) ══════════════════════════════════
    # David aprobó alinear el reparto con los artículos 74-V, 174 y 189: una
    # violación procesal se DECIDE siempre, y sólo deja de estudiarse cuando un
    # problema de FONDO prospera con mayor beneficio que la reposición. Antes
    # de hoy el árbol declaraba innecesaria la segunda procesal de un asunto
    # con el principal fundado, igual que cualquier accesorio: quedaba sin
    # decidir y nada lo decía (falla L2 del diagnóstico).
    #
    # La clase la da la fase 3 (`clase`); sin ella, el vocabulario de lo que se
    # combate, con la misma función que usa el resto del taller.
    _m = _vp()
    guarda = bool(_m and _m.guarda_aplica(tipo_asunto))
    _p_ref = por_texto.get(_kp(p_txt), (0, None))[1] or p_txt

    def _procesal(txt: str, pd) -> bool:
        return guarda and _m.clase_de(pd if pd is not None else txt) == "procesal"

    p_proc = _procesal(p_txt, por_texto.get(_kp(p_txt), (0, None))[1])
    # LA ÚNICA PUERTA PARA NO DECIDIR UNA PROCESAL: el principal es de fondo y
    # prospera entero. Si la concesión es para efectos que dan menos que
    # reponer, eso no lo sabe el código: lo sabe el secretario, y se le avisa.
    mb_fondo = bool(guarda and pros and alcanza
                    and _m.concesion_de_fondo_con_mayor_beneficio(_p_ref, p_sent, alcanza))
    # LO QUE EL MOTOR PROPUSO PARA ESTE PROBLEMA, POR SUS PROPIOS MÉRITOS.
    # Revisión del 26-sep-2026: la propuesta pasa por este mismo árbol y main
    # SOBRESCRIBE la propuesta guardada con lo que el árbol decide. Una
    # procesal que la excepción del 189 dejó «innecesaria» se guardaba así, y
    # si después el secretario cambiaba el principal, aquí ya no había qué
    # devolverle: quedaba sin calificar y el estudio la escribía «sin
    # materia» (se reprodujo con un fondo fundado que pasa a infundado). main
    # guarda ahora lo que el motor propuso ANTES del árbol en `sentido_propio`
    # y `razon_propia`; se prefiere eso, y sólo si no decide, lo guardado.
    def _propia(pr) -> tuple:
        for _ks, _kr in (("sentido_propio", "razon_propia"), ("sentido", "razon")):
            _s = str(_get(pr, _ks, "") or "").strip().lower().replace(" ", "_")
            _r = str(_get(pr, _kr, "") or "").strip()
            # Una propuesta guardada antes de hoy puede traer la caída que el
            # árbol le escribió («inoperante» + «Descansa en la premisa…»): eso
            # no es decidirla, es otra vez sacarla.
            if _decide(_s) and not _r.startswith(CAE_CON_PRINCIPAL):
                return _s, _r
        return "", ""

    _motor_de = {_kp(_get(pr, "problema", "")): _propia(pr) for pr in (propuestas or [])}
    por_189: list = []

    # ¿El principal va por la MISMA vía que propuso el motor? Entonces lo que el
    # motor propuso para cada problema es la calificación de esta vía.
    _misma_via = bool(sentido_motor) and _prospera(str(sentido_motor)) == pros
    # LO MISMO, MIRANDO LO QUE EL MOTOR PROPUSO PARA EL PRINCIPAL (26-sep-2026).
    # `sentido_motor` es el sentido GLOBAL, y no siempre llega: el banco y las
    # sesiones cuyo global vivía sólo en memoria resuelven sin él, y el ADC
    # 722/2025 salió así —sin global y con el principal donde el motor lo
    # propuso—. Lo que el motor propuso para el problema principal dice en qué
    # vía calificó los demás. Sólo lo usa la regla de la caída; la guarda
    # procesal sigue con `_misma_via`, como estaba.
    _pr_p = next((pr for pr in (propuestas or [])
                  if _kp(_get(pr, "problema", "")) == _kp(p_txt)), None) or {}
    _s_mot_p = (str(_get(_pr_p, "sentido_propio", "") or "")
                or str(_get(_pr_p, "sentido", "") or ""))
    _via_motor = (_prospera(_s_mot_p) == pros) if _decide(_s_mot_p) else _misma_via
    _via_conocida = _decide(_s_mot_p) or bool(sentido_motor)
    _p_ref_f3 = por_texto.get(_kp(p_txt), (0, None))[1] or p_txt
    estudiados: list = []          # relacionados con el principal que NO caen

    # ═══ EL CAMBIO DE SENTIDO: QUIÉN SE RECALIFICA (26-sep-2026) ═══════════
    # David: «si cambio sentido hay que tumbar y regenerar con la premisa del
    # cambio de sentido». Sólo cuando consta que el principal va por la vía
    # CONTRARIA a la del motor; si vuelve a la del motor, vale lo que el motor
    # propuso y nada se recalifica. Y nunca en el modo global dictado: el
    # sentido que el secretario dictó para todo el asunto es su palabra.
    #
    # QUIÉN ENTRA NO DEPENDE DE LO QUE TRAE: se decide por la regla (la
    # relación con el principal, la suerte escrita, la guarda), no por el
    # sentido que llega. Lo que llega de un accesorio no tocado lo escribió la
    # máquina —la propuesta de la otra vía, un reparto anterior, o la
    # recalificación que la pantalla devuelve—, y si decidiera quién entra, la
    # clave cambiaría en cuanto la pantalla devolviera lo recalificado y la
    # recalificación guardada no se encontraría nunca.
    # Sin el módulo que regenera no se tumba: se conserva y se avisa, como
    # antes (tumbar sin poder regenerar dejaría los accesorios sin calificar
    # y sin aviso).
    _cambio_via = bool(_via_conocida and not _via_motor and not global_dictado
                       and _recalificar_mod() is not None)
    _cand: dict = {}               # problema → {"procesal": bool}
    # LA RAZÓN QUE ÉL TECLEÓ SIN ELEGIR SENTIDO es su palabra aunque la pantalla
    # no lo marque tocado. Si ese accesorio se recalifica, se tumba el sentido
    # pero su razón se queda: el motor la recibe como dato y sólo le pone el
    # sentido coherente con ella (la razón es suya, el sentido lo pide él).
    _razon_suya = {str(_get(c, "problema", "")): str(_get(c, "razonamiento", "") or "")
                   for c in criterios
                   if c is not principal and _kp(_get(c, "problema", "")) not in _tocados
                   and not str(_get(c, "sentido", "") or "").strip()
                   and str(_get(c, "razonamiento", "") or "").strip()}

    def _recal_ok(t: str) -> bool:
        return _cambio_via

    # LO QUE LA PANTALLA DEVUELVE VACÍO Y ÉL NO TOCÓ, CON EL PRINCIPAL EN LA
    # VÍA DEL MOTOR, es un tumbado de una vuelta anterior en la otra vía
    # (26-sep-2026): el principal volvió y vale lo que el motor propuso. Una
    # razón que él tecleó se queda.
    if _via_conocida and _via_motor:
        for c in criterios:
            if c is principal or str(_get(c, "jerarquia", "")).lower() == "principal":
                continue
            _t0 = str(_get(c, "problema", ""))
            if _kp(_t0) in _tocados or str(_get(c, "sentido", "") or "").strip():
                continue
            _s0, _r0 = _motor_de.get(_kp(_t0), ("", ""))
            if _s0:
                _set(c, "sentido", _s0)
                if not str(_get(c, "razonamiento", "") or "").strip():
                    _set(c, "razonamiento", _r0 or "")

    def _conservar(c, t: str) -> str:
        """Lo que se hace con una procesal que el árbol habría sacado: se
        queda con su calificación. Si la que trae no decide —llegó
        «innecesario» de otra pasada, o «inoperante» con la fórmula de caer
        con el principal—, se le devuelve la que el motor le propuso, con su
        razón; si tampoco hay, se queda como está y el aviso lo pide.

        Y SI EL PRINCIPAL VUELVE A LA VÍA DEL MOTOR, manda lo que el motor
        propuso (revisión del 26-sep-2026). La que trae puede ser la que se le
        escribió para la vía contraria en un reparto anterior: con el principal
        a infundado la pericial quedaba «inoperante, se ofreció en la
        ampliación precluida», y al volverlo a fundado lo conservaba —la
        ampliación admitida y su pericial inoperante por haberse ofrecido en
        ella—."""
        s_act = str(_get(c, "sentido", "")).strip().lower().replace(" ", "_")
        _cae = _cae_con_principal(c)
        s_mot, r_mot = _motor_de.get(_kp(t), ("", ""))
        _otra = bool(_misma_via and s_mot and _decide(s_act) and s_act != s_mot)
        if _decide(s_act) and not _cae and not _otra:
            return s_act
        # La razón de «queda sin materia», de «innecesario» (también la del
        # 189) o de «descansa en la premisa desestimada» sostenía NO decidirla,
        # y la de otra calificación sostenía ésa: no se dejan pegadas. El
        # estudio (fase6) reconoce la caída por su arranque y la escribiría
        # como tal. Otra razón —la que el secretario tecleó antes de elegir
        # sentido— no se toca.
        _raz = str(_get(c, "razonamiento", "") or "")
        if _cae or _otra or "sin materia" in _raz.lower() or "innecesari" in _raz.lower():
            _set(c, "razonamiento", "")
            _raz = ""
        if s_mot:
            _set(c, "sentido", s_mot)
            if r_mot and not _raz.strip():
                _set(c, "razonamiento", r_mot)
            return s_mot
        return s_act

    def _aviso_procesal(t: str, s_kept: str, cae_con: bool, sugerida: str = "",
                        razon_sug: str = "") -> str:
        if not cae_con:
            # El mismo texto que el reparto global: en modo global corren
            # seguidos sobre el mismo problema y así no se lee dos veces.
            # «Principal procesal» quiere decir que PROSPERA: con uno
            # infundado el aviso decía «aunque el principal prospere».
            return _m.aviso_se_decide(t, s_kept, bool(p_proc and pros))
        _cal = (f"Se estudia con su calificación: {s_kept.replace('_', ' ')}."
                if _decide(s_kept) else
                "ESTÁ SIN CALIFICAR: califícala tú antes de generar.")
        _mot = (f" Para esta vía el motor le había escrito «{sugerida.replace('_', ' ')}»"
                + (f": {razon_sug[:160]}" if razon_sug else "")
                + ". Revisa que su calificación siga siendo la tuya."
                if sugerida else "")
        return (f"«{t[:90]}» es una VIOLACIÓN PROCESAL y NO se declaró caída con "
                f"el principal: los artículos 74, fracción V, y 174 de la Ley de "
                f"Amparo mandan decidirla por lo que ella misma plantea. {_cal}{_mot}")

    caen, siguen = 0, 0
    for c in criterios:
        if c is principal:
            continue
        t = str(_get(c, "problema", ""))
        if str(_get(c, "jerarquia", "")).lower() == "principal":
            continue
        num, pdict = por_texto.get(_kp(t), (0, None))
        entrada = entrada_de(checklist, num, t)
        rel = relacion_de(pdict, p_num, entrada)
        suyo = _kp(t) in _tocados
        proc = _procesal(t, pdict)

        if suyo:
            detalle[t] = {"de": "tuya", "por_que": "lo calificaste tú; manda sobre la suerte del principal"}
            # Si su marca ya no casa con lo que el principal implica, se avisa,
            # no se pisa.
            _esp, _ = _suerte(entrada, "si_prospera" if pros else "si_no_prospera", sentido_motor)
            if _esp and _esp != str(_get(c, "sentido", "")).lower():
                avisos.append(
                    f"«{t[:70]}» lo calificaste {str(_get(c, 'sentido', '')).replace('_', ' ')}; "
                    f"con el principal {p_sent.replace('_', ' ')} la suerte que el motor "
                    f"le había escrito era {_esp.replace('_', ' ')}. Se respeta tu marca.")
            # SU MARCA MANDA, PERO UNA PROCESAL SIN DECIDIR SE LE DICE. Si él
            # mismo la puso innecesaria sin que el fondo prospere con mayor
            # beneficio, el proyecto dejaría una violación procesal sin
            # decidir: no se pisa, se avisa.
            if proc and not _decide(str(_get(c, "sentido", ""))) \
                    and str(_get(c, "sentido", "")).strip() and not mb_fondo:
                avisos.append(
                    f"«{t[:90]}» la marcaste {str(_get(c, 'sentido', '')).replace('_', ' ')} "
                    f"y es una VIOLACIÓN PROCESAL: los artículos 74, fracción V, y 174 de "
                    f"la Ley de Amparo mandan decidirla, y sólo puede quedar innecesaria "
                    f"si un problema de fondo prospera con mayor beneficio que la "
                    f"reposición (artículo 189). Se respeta tu marca; revisa si es lo "
                    f"que quieres.")
            continue
        # LA SUERTE ESCRITA MANDA SOBRE LA ETIQUETA «DISTINTO». Si el motor dijo
        # que el tema es distinto pero escribió que en esta vía «al no formar
        # parte de la litis… no podían», lo concreto es la suerte.
        _s_via, _ = _suerte(entrada, "si_prospera" if pros else "si_no_prospera", sentido_motor)
        if rel == "distinto" and not _s_via:
            if not proc and _cae_con_principal(c):
                # Un resto de otra pasada: el tema distinto no cae con nadie.
                _calificacion_propia(c, "", "", _via_motor, _motor_de.get(_kp(t), ("", "")))
            detalle[t] = {"de": "distinto", "por_que": "tema distinto: no cuelga del principal y se estudia aparte"}
            siguen += 1
            continue
        if pros and pide_mas(t):
            detalle[t] = {"de": "mayor_beneficio", "por_que": "pide más que lo concedido en el principal: se estudia"}
            avisos.append(
                f"NO SE DECLARÓ INNECESARIO «{t[:90]}»: pide algo que da MÁS que lo "
                f"concedido en el principal. Declararlo innecesario sería negarlo "
                f"sin decirlo. Se estudia.")
            if _recal_ok(t):
                # Se estudia, pero lo que trae se escribió con el principal sin
                # prosperar: se recalifica con la premisa. EL MOTIVO VIAJA
                # (revisión adversarial, 26-sep-2026): la recalificación no
                # puede declarar «innecesario» lo que esta regla manda estudiar
                # —sería negar sin estudio el de mayor beneficio (art. 189)—.
                _cand[t] = {"procesal": proc, "se_estudia": "mayor_beneficio"}
            continue

        if pros:
            if not alcanza:
                if _recal_ok(t):
                    # Lo fundado no alcanza y se estudia; su calificación se
                    # escribió con el principal sin prosperar. El motivo viaja:
                    # la recalificación no puede dejarlo sin materia.
                    _cand[t] = {"procesal": proc, "se_estudia": "no_alcanza"}
                continue
            s, razon = _suerte(entrada, "si_prospera", sentido_motor)
            if not s and rel == "depende":
                s = INNECESARIO
            if s == INNECESARIO and proc:
                # ── LA GUARDA, CUANDO EL PRINCIPAL PROSPERA ──
                if mb_fondo:
                    # La única excepción, y dicha con su artículo.
                    _set(c, "sentido", INNECESARIO)
                    _set(c, "razonamiento", _m.RAZON_MAYOR_BENEFICIO)
                    detalle[t] = {"de": "principal", "guarda": "mayor_beneficio_189",
                                  "por_que": "violación procesal innecesaria por mayor beneficio "
                                             "(art. 189): el fondo que prospera da más que reponer"}
                    por_189.append(t)
                else:
                    _k = _conservar(c, t)
                    detalle[t] = {"de": "propio", "guarda": "procesal",
                                  "por_que": "violación procesal: los arts. 74-V y 174 mandan "
                                             "decidirla; no queda sin materia"}
                    if _recal_ok(t):
                        # Se decide, pero con lo que el motor escribió con el
                        # principal sin prosperar: se recalifica con la premisa.
                        _cand[t] = {"procesal": True}
                    else:
                        avisos.append(_aviso_procesal(t, _k, cae_con=False))
                    siguen += 1
                continue
            if s == INNECESARIO:
                _set(c, "sentido", INNECESARIO)
                _set(c, "razonamiento",
                     "Dado el sentido del estudio del problema principal, queda sin "
                     "materia el análisis de este planteamiento"
                     + (f": {razon}" if razon else "."))
                detalle[t] = {"de": "principal", "por_que": razon or "queda sin materia al prosperar el principal"}
                caen += 1
            elif s:
                # El motor escribió otra suerte para esta vía (p. ej. «fundado en
                # lo conducente»): se respeta, con su razón.
                _set(c, "sentido", s)
                if razon:
                    _set(c, "razonamiento", razon)
                detalle[t] = {"de": "principal", "por_que": razon}
                if _recal_ok(t) and not razon.strip():
                    # Escrita para esta vía pero sin su razón: se recalifica.
                    _cand[t] = {"procesal": proc}
                siguen += 1
            else:
                if not proc and _cae_con_principal(c):
                    # Llegó caída de un reparto con el principal en la otra
                    # vía; con éste prosperando, esa razón ya no sostiene nada.
                    _calificacion_propia(c, "", "", _via_motor, _motor_de.get(_kp(t), ("", "")))
                detalle[t] = {"de": "propio", "por_que": "no consta que dependa del principal: se estudia por su cuenta"}
                if _recal_ok(t):
                    # No queda sin materia y lo que trae se escribió con el
                    # principal sin prosperar: se recalifica con la premisa.
                    _cand[t] = {"procesal": proc}
                siguen += 1
        else:
            s, razon = _suerte(entrada, "si_no_prospera", sentido_motor)
            s_escrita = s
            if not s and rel == "depende":
                s = INOPERANTE
            if s in (INOPERANTE, "infundado", "ineficaz", "inatendible", INNECESARIO) and proc:
                # ── LA GUARDA, CUANDO EL PRINCIPAL NO PROSPERA ──
                # «Descansa en la premisa desestimada, de modo que su estudio no
                # produciría ningún fin práctico» es no decidirla. Una procesal
                # se decide por lo que ella plantea.
                #
                # PERO SI EL MOTOR LE ESCRIBIÓ UNA CALIFICACIÓN PARA ESTA VÍA,
                # ésa es la que decide (revisión del 26-sep-2026). Conservar la
                # que trae es conservar la de la vía CONTRARIA: la pericial
                # ofrecida en una ampliación que se acaba de tener por bien
                # precluida salía «fundado», con reposición para admitirla. Es
                # el mismo fallo que dio origen a este módulo en el 93/2026 —el
                # accesorio conservó el tratamiento escrito para el sentido
                # contrario—. Se aplica como CALIFICACIÓN razonada, sin la
                # fórmula de la caída: el estudio la decide y la razona, no la
                # despacha como «cae con el principal».
                if (_entrada_tiene_suerte(entrada, sentido_motor) and _decide(s)
                        and not str(razon or "").startswith(CAE_CON_PRINCIPAL)):
                    _set(c, "sentido", s)
                    _set(c, "razonamiento", razon or "")
                    detalle[t] = {"de": "principal", "guarda": "procesal",
                                  "por_que": "violación procesal: se decide con la calificación "
                                             "que el motor le escribió para esta vía, razonada; "
                                             "no se declara caída (arts. 74-V y 174)"}
                    avisos.append(
                        f"«{t[:90]}» es una VIOLACIÓN PROCESAL y NO se declaró caída con el "
                        f"principal: los artículos 74, fracción V, y 174 de la Ley de Amparo "
                        f"mandan decidirla. Con el principal {p_sent.replace('_', ' ')} se "
                        f"decide {s.replace('_', ' ')}, que es lo que el motor le escribió "
                        f"para esta vía" + (f": {razon[:160]}" if razon else "")
                        + ". El estudio lo razona como calificación suya. Revisa que sea la tuya.")
                    siguen += 1
                    continue
                _k = _conservar(c, t)
                detalle[t] = {"de": "propio", "guarda": "procesal",
                              "por_que": "violación procesal: se decide por sí misma "
                                         "(arts. 74-V y 174); no cae con el principal"}
                if _recal_ok(t):
                    # Lo que conserva es lo que el motor le escribió para la
                    # otra vía, sin suerte escrita para ésta: se recalifica.
                    _cand[t] = {"procesal": True}
                else:
                    avisos.append(_aviso_procesal(
                        t, _k, cae_con=True,
                        sugerida=(s if _entrada_tiene_suerte(entrada, sentido_motor) else ""),
                        razon_sug=razon))
                siguen += 1
                continue
            if proc:
                # Una procesal que la guarda no tomó —sin suerte que la saque,
                # o con una calificación de fondo para esta vía—: como estaba.
                # Sus restos de otra pasada los recoge el cierre de la guarda.
                if s:
                    _set(c, "sentido", s)
                    if razon:
                        _set(c, "razonamiento", razon)
                    detalle[t] = {"de": "principal", "por_que": razon}
                else:
                    detalle[t] = {"de": "propio", "por_que": "no consta que dependa del principal: se estudia por su cuenta"}
                siguen += 1
                continue
            # ── ¿CAE CON EL PRINCIPAL? SÓLO SI LO PRESUPONE, Y CONSTA ──
            # (26-sep-2026; ver el encabezado del módulo y `presupuesto`.)
            ev = presupuesto(entrada, pdict if pdict is not None else t, _p_ref_f3)
            if ev["verificado"] and (not s_escrita or s_escrita in _CAIDA
                                     or s_escrita == INNECESARIO):
                s_c = s_escrita if s_escrita in _CAIDA else INOPERANTE
                _r = razon if (razon and s_escrita in _CAIDA
                               and not razon.startswith(CAE_CON_PRINCIPAL)) else ""
                _r = _r or ev["premisa"]
                _set(c, "sentido", s_c)
                _set(c, "razonamiento",
                     "Descansa en la premisa que se desestimó al resolver el "
                     "problema principal"
                     + (f": {_r}" if _r else
                        ", de modo que su estudio no produciría ningún fin práctico."))
                detalle[t] = {"de": "principal", "relacion": "presupone", "cita": ev["cita"],
                              "por_que": (_r or "cae con el principal: su argumento da por "
                                          "cierta la premisa desestimada")
                              + f" («{ev['cita'][:120]}»)"}
                caen += 1
            elif s_escrita or rel == "depende" or _cae_con_principal(c):
                # RELACIONADO CON EL PRINCIPAL, PERO SU CAUSA DE PEDIR ES SUYA
                # (o no consta que no lo sea): se estudia con su calificación.
                _k, _de, _org = _calificacion_propia(c, s_escrita, razon, _via_motor,
                                                     _motor_de.get(_kp(t), ("", "")))
                detalle[t] = {"de": _de, "relacion": ("mixta" if ev["motivo"] == "causa_propia"
                                                       else "autonoma"),
                              "motivo": ev["motivo"], "origen": _org,
                              "por_que": _por_que_estudia(ev, _k)}
                # CON EL PRINCIPAL EN LA OTRA VÍA, se recalifica todo el que no
                # trae la suerte que el motor escribió PARA ESTA VÍA con su
                # razón: la de la otra vía, la que trae (que escribió la
                # máquina), un resto o nada.
                if _recal_ok(t) and not (
                        _org == "via" and str(razon or "").strip()
                        and not str(razon).startswith(CAE_CON_PRINCIPAL)):
                    _cand[t] = {"procesal": False}
                else:
                    estudiados.append((t, _k, _org, ev, s_escrita))
                siguen += 1
            else:
                detalle[t] = {"de": "propio", "por_que": "no consta que dependa del principal: se estudia por su cuenta"}
                siguen += 1

    # ── NINGUNA PROCESAL SALE SIN DECIDIR, VENGA DE DONDE VENGA ──────────
    # El árbol ya no las saca; pero pueden LLEGAR sacadas: de un reparto
    # anterior, de la propuesta, de una sesión de antes del 26-sep. Las que el
    # árbol dejó en paz (tema distinto, sin dependencia, lo fundado no alcanza)
    # conservaban ese «innecesario» sin que nada lo mirara.
    if guarda:
        for c in criterios:
            if c is principal or str(_get(c, "jerarquia", "")).lower() == "principal":
                continue
            t = str(_get(c, "problema", ""))
            if _kp(t) in _tocados or not _procesal(t, por_texto.get(_kp(t), (0, None))[1]):
                continue
            s_act = str(_get(c, "sentido", "")).strip().lower()
            _cae = _cae_con_principal(c)
            if not s_act or (_decide(s_act) and not _cae):
                continue
            if (detalle.get(t) or {}).get("guarda") == "procesal":
                continue                   # ya se avisó dentro del recorrido
            if _cae:
                # Llegó «caída con el principal»: se decide por sí misma.
                _k = _conservar(c, t)
                detalle[t] = {"de": "propio", "guarda": "procesal",
                              "por_que": "violación procesal: se decide por sí misma "
                                         "(arts. 74-V y 174); no cae con el principal"}
                if t not in _cand:
                    avisos.append(_aviso_procesal(t, _k, cae_con=True))
                continue
            if mb_fondo:
                # «sin materia» se escribe como tal en el estudio, que sólo
                # reconoce «innecesario»; la razón es la del 189.
                _set(c, "sentido", INNECESARIO)
                if "189" not in str(_get(c, "razonamiento", "") or ""):
                    _set(c, "razonamiento", _m.RAZON_MAYOR_BENEFICIO)
                if (detalle.get(t) or {}).get("guarda") != "mayor_beneficio_189":
                    detalle[t] = {"de": "principal", "guarda": "mayor_beneficio_189",
                                  "por_que": "violación procesal innecesaria por mayor beneficio "
                                             "(art. 189): el fondo que prospera da más que reponer"}
                    por_189.append(t)
                # Innecesaria por la regla (art. 189): no se recalifica.
                _cand.pop(t, None)
                continue
            _k = _conservar(c, t)
            detalle[t] = {"de": "propio", "guarda": "procesal",
                          "por_que": "violación procesal: los arts. 74-V y 174 mandan "
                                     "decidirla; no queda sin materia"}
            if t not in _cand:
                avisos.append(_aviso_procesal(t, _k, cae_con=False))

    # ═══ TUMBAR Y, SI YA ESTÁ, APLICAR LO RECALIFICADO (26-sep-2026) ═══════
    # Ver arriba (`_cambio_via`) y `recalificar.py`. Lo que se tumba se queda
    # SIN sentido ni razón: la calificación de la otra vía no se usa ni se le
    # enseña al modelo que recalifica. Si ya hay una recalificación guardada
    # con la MISMA clave —el mismo principal, el mismo sentido, la misma razón
    # del secretario, los mismos tumbados, el mismo adelanto—, se aplica.
    _recal = [str(_get(c, "problema", "")) for c in criterios
              if str(_get(c, "problema", "")) in _cand and _kp(_get(c, "problema", "")) not in _tocados]
    if _recal:
        _rc = _recalificar_mod()
        if _rc is not None:
            _aplicadas, _faltan = _tumbar_y_aplicar(
                _rc, criterios, _recal, _cand, detalle, principal, p_txt, p_sent, pros,
                recalificadas, huella_adelanto, tipo_asunto,
                {t: r for t, r in _razon_suya.items() if t in _recal},
                huella_premisa)
            if _faltan:
                avisos.append(
                    f"SE RECALIFICAN CON TU PREMISA {len(_faltan)} planteamiento(s): con el "
                    f"principal {p_sent.replace('_', ' ')} —la vía contraria a la que propuso el "
                    f"motor—, la calificación que el motor les había escrito suponía el principal "
                    f"al revés y no se usa. El motor los vuelve a calificar con tu sentido y tu "
                    f"razón como premisa: " + " · ".join(f"«{x[:80]}»" for x in _faltan[:6])
                    + ". Si calificas tú alguno, manda tu marca.")
            if _aplicadas:
                avisos.append(
                    f"RECALIFICADOS CON TU PREMISA {len(_aplicadas)} planteamiento(s): con el "
                    f"principal {p_sent.replace('_', ' ')}, el motor los volvió a calificar "
                    f"partiendo de tu sentido y tu razón: "
                    + " · ".join(f"«{x[:80]}» ({s.replace('_', ' ') or 'sin calificar'})"
                                 + (" —prospera: el asunto prosperaría por él aunque el principal "
                                    "no prospere—" if (not pros and _prospera(s)) else "")
                                 for x, s in _aplicadas[:6])
                    + ". Revisa que sean los tuyos; si marcas alguno, manda tu marca.")

    if por_189:
        # El mismo texto que el reparto global, que corre antes en esa vía.
        avisos.append(_m.aviso_mayor_beneficio(por_189, p_sent))
    # EL FONDO, CON LA REPOSICIÓN. Si lo que prospera es una procesal, la
    # sentencia reclamada se deja insubsistente y el fondo queda sin materia;
    # el árbol lo aplica a lo que depende del principal, pero no inventa
    # dependencias donde la fase 5 no las escribió. Lo que quedó «por su
    # cuenta» se le señala al secretario, que es quien sabe si da más.
    if guarda and pros and alcanza and p_proc:
        _fondo_vivo = [
            str(_get(c, "problema", "")) for c in criterios
            if c is not principal
            and str(_get(c, "jerarquia", "")).lower() != "principal"
            and _kp(_get(c, "problema", "")) not in _tocados
            and not _procesal(str(_get(c, "problema", "")),
                              por_texto.get(_kp(_get(c, "problema", "")), (0, None))[1])
            and (detalle.get(str(_get(c, "problema", ""))) or {}).get("de") in ("propio", "distinto",
                                                                                "recalificada")
            and _decide(str(_get(c, "sentido", "")))]
        if _fondo_vivo:
            avisos.append(
                "EL PRINCIPAL ES UNA VIOLACIÓN PROCESAL QUE PROSPERA: la reposición "
                "deja insubsistente la sentencia reclamada y el fondo queda sin "
                "materia, salvo lo que dé más que reponer. Siguen calificados por su "
                "cuenta: " + " · ".join(f"«{x[:80]}»" for x in _fondo_vivo[:4])
                + ". Si no piden más que la reposición, márcalos innecesarios.")

    if caen:
        if pros:
            avisos.append(
                f"SUSTRACCIÓN DE MATERIA aplicada a {caen} planteamiento(s): al "
                f"resultar {p_sent.replace('_', ' ')} el principal, su estudio "
                f"queda sin materia. El proyecto lo DICE, no lo calla.")
        else:
            avisos.append(
                f"{caen} planteamiento(s) accesorio(s) CAEN CON EL PRINCIPAL: al "
                f"resultar {p_sent.replace('_', ' ')}, su argumento da por cierta "
                f"una premisa desestimada y se declaran inoperantes con esa razón. "
                f"Si alguno merece estudio propio, márcalo tú.")
    if estudiados:
        # LO QUE NO SE TIRÓ, DICHO CUANDO HAY ALGO QUE REVISAR (26-sep-2026).
        # El accesorio que se estudia con la calificación que el motor le dio
        # para esta misma vía no necesita aviso: es un problema más, y la
        # pantalla ya dice por qué en `por_que`. Se avisa cuando el secretario
        # tiene algo que mirar: el motor escribió que caía y no mostró dónde
        # da por cierta la premisa, plantea algo propio además de la premisa,
        # su calificación se escribió con el principal en la otra vía, o se
        # quedó sin calificar.
        _partes = []
        for _t, _k, _org, _ev, _s_esc in estudiados:
            _nota = ""
            if _ev["motivo"] == "causa_propia":
                _nota = f" —plantea además algo propio: {_ev['causa_propia'][:110]}—"
            elif _s_esc in _CAIDA and _org == "via":
                _nota = (" —el motor escribió para esta vía que caía, sin mostrar dónde da "
                         "por cierta la premisa: el estudio lo razona como calificación suya—")
            elif _s_esc in _CAIDA:
                _nota = " —el motor escribió que caía, sin mostrar dónde da por cierta la premisa—"
            # «La otra vía» sólo cuando se sabe: el motor propuso el principal
            # al revés y lo que el accesorio lleva es lo que el motor le propuso
            # entonces. Un «infundado» que dictó el secretario para todo el
            # asunto (modo global) no es de la otra vía, aunque no se escribiera
            # para éste.
            _de_otra = _org == "motor_otra_via" or (
                _org == "actual" and _via_conocida and not _via_motor
                and _k and _k == _motor_de.get(_kp(_t), ("", ""))[0])
            if _de_otra and _via_conocida:
                _nota += (" —calificación escrita con el principal en la otra vía: revisa que "
                          "siga siendo la tuya—")
            elif _de_otra:
                # Revisión del 26-sep-2026: sin el sentido del motor para el
                # principal ni el global no se sabe en qué vía la escribió, y
                # decir «la otra» sería afirmar lo que no consta.
                _nota += (" —es la que el motor le propuso, sin que conste en qué vía la "
                          "escribió: revisa que sea la tuya—")
            # LO QUE CAMBIA EL DESENLACE, DICHO (revisión del 26-sep-2026).
            # Medido en el banco Kingston con el principal desestimado: en los
            # cinco accesorios que quedaron «fundado» con la calificación de la
            # otra vía, el engrose real los declaró infundados o inoperantes.
            # Tumbarlos sería la omisión de estudio que esta regla corrige, y
            # rebajarlos sería inventarles calificación; lo que no puede pasar
            # es que el asunto prospere por uno de ellos sin que el secretario
            # lo haya visto.
            if _de_otra and _k and _prospera(_k):
                _nota += (" —y como prospera, el asunto prosperaría por él aunque el principal "
                          "no prospere—")
            if not _decide(_k):
                # «innecesario» o «sin materia» que llegaron de otra pasada no
                # califican nada: con el principal que no prospera, nada lo deja
                # sin materia (antes sólo se avisaba si llegaba vacío).
                _nota += " —SIN CALIFICAR: califícalo tú antes de generar—"
            elif _org == "resto":
                _nota += (" —llegó declarado caído con el principal y no hay otra calificación "
                          "para él: se estudia con ésa, sin la fórmula de la caída; revisa que "
                          "sea la tuya—")
            if _nota:
                _partes.append(f"«{_t[:80]}» ({_k.replace('_', ' ') if _k else 'sin calificar'})"
                               + _nota)
        if _partes:
            avisos.append(
                f"NO SE DECLARARON CAÍDOS CON EL PRINCIPAL {len(_partes)} planteamiento(s) "
                f"relacionados con él: con el principal {p_sent.replace('_', ' ')}, cae sólo "
                f"el argumento que, entero, da por cierta la premisa desestimada, y de éstos no "
                f"consta. "
                f"Se estudian con su calificación: " + " · ".join(_partes)
                + ". Si alguno sólo tiene sentido con esa premisa, márcalo inoperante tú.")
    # EL ASUNTO QUE PROSPERA POR UN ACCESORIO, DICHO SIEMPRE (integración,
    # 26-sep-2026). Medido sobre 16 engroses con el principal desestimado: en
    # 6 un accesorio que el secretario no tocó quedaba «fundado» —cuatro con la
    # calificación del motor en la misma vía, dos con la suerte escrita para
    # ella— y sólo se avisaba si venía de la otra vía. Que el asunto prospere
    # por un accesorio puede ser justo lo correcto; lo que no puede pasar es
    # que ocurra sin que él lo lea. Lo que él marcó es su decisión: no se avisa.
    if not pros:
        _dicho = " ".join(avisos)
        _por_acc = [str(_get(c, "problema", "")) for c in criterios
                    if c is not principal
                    and str(_get(c, "jerarquia", "")).lower() != "principal"
                    and _kp(_get(c, "problema", "")) not in _tocados
                    and _prospera(str(_get(c, "sentido", "")))]
        _por_acc = [t for t in _por_acc if f"«{t[:80]}»" not in _dicho]
        if _por_acc:
            avisos.append(
                f"CON EL PRINCIPAL {p_sent.replace('_', ' ').upper()}, EL ASUNTO PROSPERARÍA "
                f"POR {'UN ACCESORIO' if len(_por_acc) == 1 else 'ACCESORIOS'} que no "
                f"calificaste tú: " + " · ".join(f"«{t[:80]}»" for t in _por_acc[:4])
                + ". Esa calificación es la que propuso el motor; confírmala o márcala tú "
                  "antes de generar.")
    return avisos, detalle


def _recalificar_mod():
    """El módulo de la recalificación, o None si no carga: sin él, el árbol
    se comporta como antes (conserva y avisa) en vez de tumbar sin poder
    regenerar."""
    try:
        import recalificar as _rc
        return _rc
    except Exception:                                   # pragma: no cover
        return None


def _tumbar_y_aplicar(_rc, criterios: list, recal: list, cand: dict, detalle: dict,
                      principal, p_txt: str, p_sent: str, pros: bool,
                      recalificadas, huella_adelanto: str, tipo_asunto: str,
                      razon_suya: dict = None, huella_premisa: str = "") -> tuple:
    """Tumba los de `recal` y aplica la recalificación guardada de la misma
    clave. Devuelve ([(problema, sentido) aplicados], [problemas pendientes]).

    Lo recalificado se escribe como el árbol escribe lo suyo: una calificación
    de fondo con su razón; la caída verificada con la fórmula de la caída (el
    estudio la reconoce por ella); «innecesario» con la de la sustracción.
    `razon_suya`: la razón que el secretario tecleó sin elegir sentido; se
    queda y de lo recalificado sólo se toma el sentido. No entra en la clave
    (la pantalla la devuelve después con el sentido): viaja guardada con el
    resultado y se vuelve a poner desde ahí."""
    razon_suya = dict(razon_suya or {})
    k = _rc.clave(p_txt, p_sent, str(_get(principal, "razonamiento", "") or ""), recal,
                  huella_adelanto, tipo_asunto, huella_premisa)
    # La premisa sola: si él pisó alguno de los tumbados, lo que el motor ya
    # recalificó para los demás con esta misma premisa sigue valiendo.
    kp = _rc.clave_premisa(p_txt, p_sent, str(_get(principal, "razonamiento", "") or ""),
                           huella_adelanto, tipo_asunto, huella_premisa)
    casilla = _rc.casilla_de(recalificadas, k, kp, recal)
    res = dict((casilla or {}).get("resultados") or {})
    _res_k: dict = {}
    for _x, _v in res.items():
        _res_k.setdefault(clave_problema(_x), _v)
    por_c = {str(_get(c, "problema", "")): c for c in criterios}
    aplicadas, faltan = [], []
    for t in recal:
        c = por_c[t]
        prev = detalle.get(t) or {}
        base = {x: prev[x] for x in ("guarda", "relacion") if prev.get(x)}
        base.update({"clave_recalificar": k, "premisa_recalificar": kp, "principal": p_txt,
                     "procesal": bool((cand.get(t) or {}).get("procesal"))})
        # POR QUÉ EL ÁRBOL LO MANDA ESTUDIAR aunque el principal prospere
        # («mayor_beneficio», «no_alcanza»): la recalificación lo recibe y no
        # puede declararlo innecesario.
        _se_est = str((cand.get(t) or {}).get("se_estudia") or "")
        if _se_est:
            base["se_estudia"] = _se_est
        _suya = str(razon_suya.get(t) or "").strip()
        if _suya:
            base["razon_suya"] = True
        # TUMBAR: ni el sentido ni la razón de la otra vía se quedan. La razón
        # que tecleó él, sí.
        _set(c, "sentido", "")
        _set(c, "razonamiento", _suya)
        # Lo guardado lleva el problema como lo armó el criterio (recortado a
        # 400); /taller/reparto lo trae entero: se casa por `clave_problema`.
        rt = res.get(t) if isinstance(res.get(t), dict) else (
            _res_k.get(clave_problema(t)) if isinstance(_res_k.get(clave_problema(t)), dict)
            else None)
        s_rt = _sentido_valido((rt or {}).get("sentido"))
        r_rt = str((rt or {}).get("razon") or "").strip()
        if s_rt == INNECESARIO and _se_est:
            # DEFENSA: lo que esta regla manda estudiar no se deja sin materia
            # aunque lo diga una recalificación (la validación ya lo rechaza;
            # esto cubre una casilla que se colara). Se queda por recalificar.
            rt = None
        if rt and s_rt and r_rt:
            # La suya: la que llega ahora sin sentido o, si la pantalla ya la
            # devuelve con el sentido recalificado, la que se guardó con él.
            _suya = _suya or str(rt.get("razon_suya") or "").strip()
            if _suya:
                base["razon_suya"] = True
            pre = rt.get("presupone") if isinstance(rt.get("presupone"), dict) else None
            # CON SU RAZÓN, de lo recalificado sólo se toma el sentido, también
            # en la caída y en la sustracción: la fórmula sustituiría lo que él
            # escribió (revisión adversarial, 26-sep-2026).
            if pre and rt.get("verificado") is True and not base["procesal"] and not pros:
                _pq = str(pre.get("por_que") or "").strip() or r_rt
                _set(c, "sentido", s_rt if s_rt in _CAIDA else INOPERANTE)
                _set(c, "razonamiento", _suya or
                     f"{CAE_CON_PRINCIPAL} al resolver el problema principal: {_pq}")
                base.update({"relacion": "presupone", "cita": str(pre.get("cita") or "")})
                _pant = (f"cae con tu premisa{', con tu razón' if _suya else ': ' + _pq[:200]} "
                         f"(«{str(pre.get('cita') or '')[:120]}»)")
            elif s_rt == INNECESARIO:
                _set(c, "sentido", INNECESARIO)
                _set(c, "razonamiento", _suya or
                     f"{SIN_MATERIA} el análisis de este planteamiento: {r_rt}")
                _pant = "innecesario, con tu razón" if _suya else r_rt
            else:
                _set(c, "sentido", s_rt)
                _set(c, "razonamiento", _suya or r_rt)
                _pant = (f"{s_rt.replace('_', ' ')}, con tu razón" if _suya else r_rt)
            detalle[t] = dict(base, de="recalificada", recalificar=False, recalificada=True,
                              por_que=f"recalificada por el motor con tu premisa: {_pant[:240]}")
            aplicadas.append((t, str(_get(c, "sentido", "") or "")))
        else:
            detalle[t] = dict(base, de="por_recalificar", recalificar=True,
                              por_que=(f"con el principal {p_sent.replace('_', ' ')} —la vía contraria "
                                       f"a la que propuso el motor— la calificación que traía se "
                                       f"escribió para la otra vía y no se usa: se recalifica con tu "
                                       f"premisa"))
            faltan.append(t)
    return aplicadas, faltan


def _calificacion_propia(c, s_escrita: str, razon_escrita: str, via_motor: bool,
                         motor: tuple) -> tuple:
    """La calificación con que se estudia un accesorio relacionado con el
    principal que NO cae con él (26-sep-2026). Ajusta `c` en sitio y devuelve
    (sentido, de, origen).

    El orden, y por qué:
      1. Si el motor calificó los problemas con el principal en ESTA vía, su
         propuesta para este problema es la calificación de fondo que hizo
         sabiendo que el principal no prosperaba («la calificación propia que
         el motor le propuso para su problema»). En el ADC 722/2025 eran dos
         «infundado» con su razón, y el árbol los tiraba. Salvo que traiga un
         sentido sin razón distinto del del motor: es el global que dictó el
         secretario, y manda.
      2. Si no, lo que el motor escribió para esta vía en la lista, aplicado
         como calificación razonada —sin la fórmula de la caída—, que es lo
         que la guarda procesal ya hacía con una procesal.
      3. Si no, la que trae, salvo que sea un resto de otra pasada (la caída o
         el «sin materia»).
      4. Si no, la que el motor le propuso, aunque sea de la otra vía: mejor
         que dejarla sin calificar, y el aviso lo dice.
      5. Si no hay nada, la que trae, aunque sea un resto, y el aviso pide que
         la califique o la revise.
    La razón que sostenía otro sentido no se queda pegada al nuevo; la que el
    secretario tecleó para el mismo sentido, sí.
    """
    s_act = str(_get(c, "sentido", "") or "").strip().lower().replace(" ", "_")
    r_act = str(_get(c, "razonamiento", "") or "")
    # UN RESTO DE OTRA PASADA se reconoce por las fórmulas que escriben este
    # módulo y `modos_decision`, no por una palabra suelta: una razón de fondo
    # puede decir «innecesario» con todo derecho.
    sobra = (r_act.startswith(CAE_CON_PRINCIPAL) or r_act.startswith(SIN_MATERIA)
             or s_act in (INNECESARIO, "sin_materia", "queda_sin_materia"))
    s_mot, r_mot = motor or ("", "")
    _r_esc = "" if str(razon_escrita or "").startswith(CAE_CON_PRINCIPAL) else str(razon_escrita or "")
    _act_ok = _decide(s_act) and not sobra
    if via_motor and s_mot and _act_ok and not r_act.strip() and s_act != s_mot:
        # UNA CALIFICACIÓN SIN RAZÓN QUE NADIE ESCRIBIÓ PARA ÉL es el sentido
        # que el secretario dictó para todo el asunto (modo global: la brocha
        # llega sin razón). Su palabra va antes que la del motor —«lo que debe
        # dársele mayor peso es a la palabra del secretario»—; lo que el motor
        # propuso con razón, en cambio, sí sustituye a la calificación de la
        # otra vía que un reparto anterior dejó (ida y vuelta del principal).
        s_n, r_n, de, org = s_act, "", "propio", "actual"
    elif via_motor and s_mot:
        s_n, r_n, de, org = s_mot, r_mot, "propio", "motor"
    elif s_escrita and _decide(s_escrita):
        s_n, r_n, de, org = s_escrita, _r_esc, "principal", "via"
    elif _act_ok:
        s_n, r_n, de, org = s_act, r_act, "propio", "actual"
    elif s_mot:
        s_n, r_n, de, org = s_mot, r_mot, "propio", "motor_otra_via"
    else:
        # Sin nada más, se queda con lo que trae —como la guarda procesal, que
        # no inventa calificación— y el aviso lo dice: «resto» si lo que trae
        # es la caída de otra pasada (revisión del 26-sep-2026: salía sin
        # aviso ninguno), «sin_calificar» si no califica.
        s_n, r_n, de = s_act, "", "propio"
        org = "resto" if (sobra and _decide(s_act)) else "sin_calificar"
    if sobra:
        _set(c, "razonamiento", "")
        r_act = ""
    if s_n and s_n != s_act:
        _set(c, "sentido", s_n)
        _set(c, "razonamiento", r_n or "")
    elif r_n and not r_act.strip():
        _set(c, "razonamiento", r_n)
    return s_n, de, org


def _por_que_estudia(ev: dict, sentido: str) -> str:
    _s = sentido.replace("_", " ") if sentido else "sin calificar"
    if ev.get("motivo") == "causa_propia":
        return (f"parte de su argumento descansa en la premisa desestimada, pero plantea "
                f"además algo propio: se estudia con su calificación ({_s})")
    return (f"se relaciona con el principal, pero no consta que su argumento dé por cierta "
            f"la premisa desestimada: se estudia con su calificación ({_s})")


def _entrada_tiene_suerte(entrada: dict, sentido_motor: str = "") -> bool:
    """¿El motor escribió una suerte para la vía «no prospera»? (para no
    presentarle al secretario como «lo que escribió el motor» lo que en
    realidad dedujo el árbol de un `depende_de`)."""
    return bool(_suerte(entrada or {}, "si_no_prospera", sentido_motor)[0])


def reparto_para_pantalla(problemas: list, criterios: list, checklist: list = None,
                          propuestas: list = None, sentido_motor: str = "",
                          tipo_asunto: str = "", recalificadas: dict = None,
                          huella_adelanto: str = "", huella_premisa: str = "") -> dict:
    """Lo que devuelve /taller/reparto: los criterios ya ajustados y de quién
    es cada uno. Trabaja sobre COPIAS: la pantalla decide qué hace con ello.
    Los que se tumban por el cambio de sentido vuelven vacíos, con `de:
    "por_recalificar"` y `recalificar: true`; si ya hay recalificación
    guardada con la misma clave (`recalificadas`), vuelven con ella y `de:
    "recalificada"`. Sin llamar a ningún modelo."""
    copia = [dict(c) if isinstance(c, dict) else {
        "problema": _get(c, "problema", ""), "sentido": _get(c, "sentido", ""),
        "razonamiento": _get(c, "razonamiento", ""),
        "jerarquia": _get(c, "jerarquia", "accesorio"),
        "tocado": bool(_get(c, "tocado", False))} for c in (criterios or [])]
    avisos, detalle = aplicar(problemas, copia, checklist, propuestas,
                              sentido_motor=sentido_motor, tipo_asunto=tipo_asunto,
                              recalificadas=recalificadas, huella_adelanto=huella_adelanto,
                              huella_premisa=huella_premisa)
    for c in copia:
        _pantalla(c, detalle.get(str(c.get("problema", "")), {}))
    return {"criterios": copia, "avisos": avisos}


def _pantalla(c: dict, d: dict) -> dict:
    """Los campos que la pantalla lee de cada criterio repartido (en sitio)."""
    c["de"] = d.get("de", "")
    c["por_que"] = d.get("por_que", "")
    # «procesal» (se decide) o «mayor_beneficio_189»: la pantalla de hoy
    # pinta `de` y `por_que`; esto viaja para la que lo distinga.
    c["guarda"] = d.get("guarda", "")
    # «presupone» (cae con el principal, con su cita), «autonoma» o
    # «mixta» (relacionado con el principal, pero se estudia): 26-sep-2026.
    # Viaja para la pantalla que lo distinga; la de hoy pinta `por_que`.
    c["relacion"] = d.get("relacion", "")
    # EL CAMBIO DE SENTIDO (26-sep-2026): tumbado y pendiente de
    # recalificar, o ya recalificado con la premisa del secretario.
    c["recalificar"] = bool(d.get("recalificar"))
    c["recalificada"] = bool(d.get("recalificada"))
    return c
