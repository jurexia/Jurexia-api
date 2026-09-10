"""LOS TRES MODOS DE DECIDIR EL SENTIDO.

David: «la interfaz ahora deberá ofrecer la posibilidad de una solución a
partir de holdings y jurimetría, posibilitar al secretario a introducir un
sentido global del proyecto (que el motor debe tomar en cuenta si lo hace) o
permitirle seguir la línea de resolución por conflicto como actualmente
funciona (pero con una redacción clara de los problemas jurídicos)».

Son tres formas de llegar al MISMO sitio —una lista de `Criterio`, uno por
problema— y por eso conviene que estén juntas y no repartidas por el endpoint:
lo que cambia es de dónde sale el sentido, no qué se hace con él.

    ACERVO       lo propone la máquina con los holdings y la jurimetría, y el
                 secretario acepta o corrige. Es lo que ya hacía /taller/proponer,
                 ahora con la predicción de CADA problema al lado.
    GLOBAL       el secretario dicta un sentido para el proyecto entero. La
                 máquina lo reparte y aplica la sustracción de materia.
    POR PROBLEMA uno por uno, como hasta hoy.

═══════════════════════════════════════════════════════════════════════════
LA SUSTRACCIÓN DE MATERIA, Y POR QUÉ NO ES AUTOMÁTICA DEL TODO
═══════════════════════════════════════════════════════════════════════════
David: «Si el secretario elige Solución Global sobre el P1 como fundado, el
sistema automáticamente etiqueta los demás como innecesarios por sustracción
de materia».

La regla es correcta y es la que evita el vicio contrario —contestar cinco
planteamientos cuando el primero ya resolvió el asunto—, pero tiene un límite
que hay que respetar o se convierte en omisión de estudio:

  · SÓLO opera cuando lo fundado ALCANZA. Un agravio fundado que sólo lleva a
    reponer el procedimiento no vuelve innecesario el que pide el fondo con
    mayor beneficio: ahí hay que estudiar los dos. Por eso `alcanza` viaja
    desde la propuesta y aquí se respeta.

  · NUNCA sobre un planteamiento de MAYOR BENEFICIO. Si un accesorio pide algo
    que da más que lo concedido —la nulidad lisa y llana frente a la reposición—
    declararlo innecesario es negarle al quejoso lo que pidió sin decirlo.

  · Y SE ESCRIBE, no se calla. La fórmula del corpus —medida en el ARA 17/2025—
    dice «Dado el sentido de la revisión principal… queda sin materia»: el
    documento explica por qué no estudia, que es lo contrario de omitir.
"""

from __future__ import annotations

ACERVO = "acervo"
GLOBAL = "global"
POR_PROBLEMA = "por_problema"

# El sentido que se le pone a lo que ya no hace falta estudiar. No es una
# calificación del planteamiento —no se dice que sea infundado— sino una razón
# para no entrar: por eso tiene nombre propio.
INNECESARIO = "innecesario"

# Los sentidos que RESUELVEN a favor de quien promueve. Sólo uno de éstos en el
# principal puede volver innecesarios los accesorios.
_ALCANZAN = {"fundado", "fundado_suplido", "concede"}

# Lo que un accesorio pide y que NO se puede declarar innecesario aunque el
# principal prospere: da más de lo que el principal concede.
# «DE FONDO» ESTABA AQUÍ Y HABRÍA MATADO LA FUNCIÓN. Casi todo problema
# jurídico que redacta la fase 3 contiene esas dos palabras —«¿la responsable
# resolvió el fondo del asunto?»—, así que la sustracción de materia, que es
# justamente lo que David pidió por su nombre, no se habría aplicado NUNCA y
# el aviso habría mentido sobre el motivo. Es la trampa de siempre: una
# heurística de una palabra dentro de un texto que la contiene por otra razón.
#
# Lo que queda son figuras que sólo aparecen cuando de verdad se pide más:
# nadie escribe «nulidad lisa y llana» ni «cosa juzgada» de pasada.
_MAYOR_BENEFICIO = (
    "mayor beneficio", "lisa y llana", "nulidad lisa",
    "prescripción", "prescripcion", "caducidad", "cosa juzgada",
    "improcedencia del juicio", "sobreseimiento",
)


def _pide_mas(problema: str) -> bool:
    t = (problema or "").lower()
    return any(x in t for x in _MAYOR_BENEFICIO)


def repartir(problemas: list, modo: str, sentido_global: str = "",
             propuestas: list = None, calificaciones: dict = None,
             global_dictado: bool = False) -> tuple:
    """(lista de {problema, sentido, razonamiento, jerarquia}, avisos).

    `problemas` son los dicts de la fase 3; `propuestas`, lo que sugirió el
    motor; `calificaciones`, lo que el secretario marcó por problema.
    """
    avisos: list = []
    props = {p.get("problema", ""): p for p in (propuestas or [])
             if isinstance(p, dict)}
    califs = calificaciones or {}

    def _texto(p):
        return p if isinstance(p, str) else str((p or {}).get("pregunta") or p)

    def _jer(p, i):
        if isinstance(p, dict) and p.get("jerarquia"):
            return str(p["jerarquia"]).strip().lower()
        return "principal" if i == 0 else "accesorio"

    fuera = []
    for i, p in enumerate(problemas or []):
        t = _texto(p)
        jer = _jer(p, i)
        if modo == POR_PROBLEMA:
            c = califs.get(t) or {}
            sentido = str(c.get("sentido") or "").strip().lower()
            razon = str(c.get("razonamiento") or "")
        elif modo == GLOBAL:
            # ── LO QUE EL SECRETARIO TOCÓ, MANDA ──────────────────────────
            #
            # El sentido global es un RELLENO, no una orden sobre cada tema.
            # David pidió que un concepto de violación —el de la pericial
            # declarada desierta— se calificara INFUNDADO, lo marcó, y el
            # proyecto salió FUNDADO: la pantalla se había puesto sola en modo
            # global con el sentido del MODELO y tiraba en silencio lo que él
            # había marcado.
            #
            # Aquí se cierra por el lado del servidor: el global rellena los
            # problemas que nadie tocó, y donde hay calificación expresa gana
            # ésa. Sus palabras: «lo que debe dársele mayor peso es a la
            # palabra del secretario, no a la automatización del sistema».
            c = califs.get(t) or {}
            _suyo = str(c.get("sentido") or "").strip().lower()
            # EL ORDEN DEL RELLENO. Manda lo que él marcó; si no marcó nada,
            # vale más la propuesta que el motor hizo PARA ESE PROBLEMA que el
            # sentido global, que es una brocha gorda.
            #
            # Se vio probando el ADC 536/2025: el secretario cambió UN concepto
            # y el otro perdió su calificación propia —el motor lo había
            # propuesto infundado— porque el global lo aplastaba con «fundado».
            # Cambiar una cosa no puede rehacer las demás.
            _prop = str((props.get(t) or {}).get("sentido") or "").strip().lower()
            _glob = (sentido_global or "").strip().lower()
            # ── EL ORDEN, Y POR QUÉ IMPORTA CUÁL ─────────────────────────
            #
            # Manda siempre lo que él marcó para ESE problema. Después depende
            # de QUIÉN puso el sentido global:
            #
            #  · si lo DICTÓ él —eligió «infundado» y «global» a propósito—,
            #    es una orden sobre el asunto entero y va por delante de lo que
            #    el motor propuso para cada problema.
            #  · si lo puso la pantalla al llegar la propuesta, no es su
            #    palabra sino un eco del motor, y entonces vale más la
            #    propuesta concreta de cada problema que ese eco.
            #
            # Sin esa distinción, la primera versión de este arreglo dejó el
            # global por debajo SIEMPRE, y David dictó «infundado global» y
            # recibió un proyecto que amparaba: cada problema se quedó con lo
            # que el motor había propuesto y su orden no llegó a ningún sitio.
            # ── Y CUÁNDO EL GLOBAL DICTADO PASA POR DELANTE ──────────────
            #
            # Faltaba un caso, y se vio conduciendo el 91/2025 en pantalla:
            # el secretario califica los temas uno a uno, cambia de idea, se
            # pasa a la vía global y dicta INFUNDADO —y la tarjeta seguía
            # diciendo inoperante, fundado, fundado. Sus marcas por tema le
            # ganaban al global que acababa de dictar. Es literalmente su queja:
            # «le indiqué con botones que lo declarara infundado, que el sentido
            # fuera global, y lo hizo fundado».
            #
            # En la vía global la pantalla ya NO muestra las pastillas por tema,
            # así que una marca por tema presente aquí sólo puede ser un resto
            # de la otra vía: es lo viejo, y el global dictado es lo último que
            # él dijo. Manda lo último.
            #
            # El eco sigue por debajo: si el sentido global lo puso la pantalla
            # al llegar la propuesta (global_dictado=False) no es su palabra, y
            # entonces la marca por tema conserva la preferencia. Ésa es la
            # lección del ADC 536/2025 y se queda intacta.
            _pisado = bool(global_dictado and _glob and _suyo and _suyo != _glob)
            if global_dictado and _glob:
                sentido = _glob
            elif _suyo:
                sentido = _suyo
            else:
                sentido = _prop or _glob
            # LA RAZÓN NO PUEDE SOBREVIVIR AL SENTIDO QUE LA SOSTENÍA. Si el
            # global pisa una marca suya, su razón argumentaba lo contrario:
            # dejarla pegada al sentido nuevo es fabricar una incongruencia.
            razon = "" if _pisado else (
                str(c.get("razonamiento") or "")
                or (str((props.get(t) or {}).get("razon") or "")
                    if not _suyo and not (global_dictado and _glob) else ""))
            if _pisado:
                avisos.append(
                    f"«{t[:70]}» lo habías marcado {_suyo.replace('_', ' ')} "
                    f"tema por tema, y se resuelve "
                    f"{_glob.replace('_', ' ')} porque después dictaste ese "
                    f"sentido para todo el asunto. Tu razón de aquel momento no "
                    f"se usa: sostenía lo contrario.")
            elif _suyo and _suyo != (sentido_global or "").strip().lower():
                avisos.append(
                    f"«{t[:70]}» se resuelve {_suyo.replace('_', ' ')} porque "
                    f"así lo marcaste, no {(sentido_global or '').replace('_', ' ')} "
                    f"como el resto del asunto.")
        else:
            pr = props.get(t) or {}
            sentido = str(pr.get("sentido") or "").strip().lower()
            razon = str(pr.get("razon") or "")
        fuera.append({"problema": t, "sentido": sentido,
                      "razonamiento": razon, "jerarquia": jer})

    if modo != GLOBAL or not fuera:
        return fuera, avisos

    # ── LA SUSTRACCIÓN DE MATERIA ──────────────────────────────────────────
    principal = next((x for x in fuera if x["jerarquia"] == "principal"), fuera[0])
    if principal["sentido"] not in _ALCANZAN:
        return fuera, avisos
    # NI SIQUIERA LA SUSTRACCIÓN DE MATERIA pisa lo que el secretario marcó. Si
    # dijo que un planteamiento es infundado, se estudia y se declara infundado,
    # aunque el principal prospere y el resto quede sin materia: quien decide si
    # un tema merece respuesta propia es él.
    # Y si el global lo dictó él, no hay marcas por tema que exceptuar: las que
    # hubiera son de la otra vía y acaban de quedar pisadas arriba.
    _suyos = set() if global_dictado and (sentido_global or "").strip() else {
        str(k) for k, v in (califs or {}).items()
        if str((v or {}).get("sentido") or "").strip()}

    # ¿ALCANZA? La propuesta lo dice cuando el material da para saberlo.
    pr_principal = props.get(principal["problema"]) or {}
    if pr_principal.get("alcanza") is False:
        avisos.append(
            "NO SE APLICÓ LA SUSTRACCIÓN DE MATERIA: el motor no pudo afirmar "
            "que lo fundado del problema principal alcance para resolver el "
            "asunto. Los accesorios se estudian.")
        return fuera, avisos

    tocados = 0
    for x in fuera:
        if x is principal or x["jerarquia"] == "principal":
            continue
        if x["problema"] in _suyos:
            avisos.append(
                f"«{x['problema'][:70]}» NO se declaró innecesario: lo "
                f"calificaste tú, y eso manda sobre la sustracción de materia.")
            continue
        if _pide_mas(x["problema"]):
            avisos.append(
                f"NO SE DECLARÓ INNECESARIO «{x['problema'][:90]}»: pide algo "
                f"que da MÁS que lo concedido en el principal. Declararlo "
                f"innecesario sería negarlo sin decirlo. Se estudia.")
            continue
        x["sentido"] = INNECESARIO
        x["razonamiento"] = (
            "Dado el sentido del estudio del problema principal, queda sin "
            "materia el análisis de este planteamiento.")
        tocados += 1
    if tocados:
        avisos.append(
            f"SUSTRACCIÓN DE MATERIA aplicada a {tocados} planteamiento(s): al "
            f"resultar {principal['sentido']} el principal, su estudio se "
            f"vuelve innecesario. El proyecto lo DICE, no lo calla.")
    return fuera, avisos
