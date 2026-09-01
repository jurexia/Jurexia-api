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
             propuestas: list = None, calificaciones: dict = None) -> tuple:
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
            sentido = (sentido_global or "").strip().lower()
            razon = ""
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
