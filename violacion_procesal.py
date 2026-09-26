"""LA VIOLACIÓN PROCESAL SE EXAMINA SOBRE LA RESOLUCIÓN QUE LA DECIDIÓ.

POR QUÉ EXISTE. Amparo directo 93/2026, 22-sep-2026. La quejosa reclamaba, como
violación procesal, que la Sala tuvo por precluido su derecho a ampliar la
demanda; contra ese acuerdo interpuso reclamación y la interlocutoria lo
confirmó. David pegó en la ventana del taller LOS MOTIVOS de esa interlocutoria
—vía sumaria, cinco días del artículo 58-6, del 5 al 12 de agosto, escrito del
14—. El proyecto citó la jurisprudencia correcta (188621: la Sala debe otorgar
expresamente el plazo) y despachó la reclamación en un párrafo: «esa
consideración no supera el vicio advertido». Abordó el tema desde una
perspectiva equivocada —una teoría de «ambigüedad entre dos regímenes»— en vez
de confrontar, una por una, las razones de la resolución que decidió la
violación.

LA CAUSA. Lo que el secretario aporta viaja bajo un rótulo genérico —«DOCUMENTO
APORTADO POR EL SECRETARIO — cítalo por lo que dice»— y el modelo lo trata como
un papel más. Y la razón toral que el contraste calcula sale SIEMPRE de la
sentencia definitiva («la actora no ejerció su derecho»), que en una violación
procesal no es la razón toral: ésa vive en el acuerdo recurrido y en la
interlocutoria que lo confirmó.

LA REGLA. En amparo directo la violación procesal se examina sobre la
actuación que la causó y sobre la resolución del recurso ordinario que la
confirmó (artículos 171 y 172 de la Ley de Amparo). El estudio enuncia las
razones de ESA resolución y las confronta con la ley y la jurisprudencia; la
calificación del planteamiento sale de ese contraste, no de la regla
abstracta.

Se usa en TRES sitios: al aportar (para decir qué es lo que llegó), en la
propuesta (para que el motor proponga con la razón toral correcta) y en el
estudio (para que se escriba como lo que es).
"""
from __future__ import annotations

import re

# Clases de lo que el secretario aporta.
RESOLUCION_PROCESAL = "resolucion_procesal"   # la interlocutoria / el acuerdo recurrido
CONSTANCIA = "constancia"                     # un contrato, un acta, un peritaje…
OTRO = "otro"

# ═══ LO QUE ES UNA RESOLUCIÓN DEL INCIDENTE, Y LO QUE SÓLO LA MENCIONA ═════
#
# CALIBRADO sobre los dos documentos reales del 93/2026: la sentencia
# definitiva (75 mil caracteres) y la demanda de amparo (62 mil). Las dos
# MENCIONAN la reclamación, el auto recurrido y la preclusión —la demanda la
# relata, la sentencia la recoge— y una primera versión, por señales sueltas,
# las clasificaba a las dos como «la resolución que decidió la violación».
# Eso habría mandado al modelo tratar la demanda de la quejosa como la razón
# toral a confrontar.
#
# Lo que distingue a la resolución del incidente no es que hable de él: es
# que lo DECIDE, en voz del órgano, y que trata de él con densidad —una
# interlocutoria de diez mil caracteres nombra la reclamación y el auto
# recurrido decenas de veces; una demanda de sesenta mil, cinco—. Y un
# escrito de parte se reconoce por su voz: «vengo a», «solicito», «conceptos
# de violación», «PRESENTE».
_RX_DECIDE_INCIDENTE = re.compile(
    r"(?:es|son|resulta|resultan|se\s+declara|se\s+declar[óo]|declar[óo]|se\s+estim[óo]|"
    r"estim[óo]|se\s+consider[óo]|consider[óo]|se\s+calific[óo]|calific[óo]|resolvi[óo]|"
    r"se\s+resuelve|se\s+resolvi[óo])\s+(?:que\s+)?(?:era|es|resulta|resultaba)?\s*"
    r"(?:infundad|fundad|improcedent|inoperant|inatendibl)\w*\s+"
    r"(?:el\s+|la\s+|dicho\s+|dicha\s+)?(?:recurso|reclamaci[óo]n|incidente|queja|revocaci[óo]n)|"
    r"(?:recurso\s+de\s+)?reclamaci[óo]n\s+(?:interpuest\w+\s+)?(?:[^.]{0,80}?)\b"
    r"(?:es|resulta|se\s+declar[óoa]|fue\s+declarad[oa]|se\s+estim[óoa])\s+"
    r"(?:infundad|fundad|improcedent|inoperant)\w*|"
    r"se\s+confirma\s+(?:el\s+|la\s+)?(?:auto|acuerdo|prove[íi]do|resoluci[óo]n)\s+"
    r"(?:recurrid|impugnad|reclamad|de\s+fecha|de\s+\w+\s+de)|"
    r"(?:[úu]nico|primero)\W{0,4}(?:es|resulta|se\s+declara)\s+(?:infundad|fundad|improcedent)\w+|"
    r"(?:motivos?|razones|consideraciones)\s+por\s+(?:los|las)\s+(?:cuales|que)\s+"
    r"(?:se\s+)?(?:consider|declar|estim|resolvi)\w+\s+(?:infundad|fundad|improcedent|extempor)\w*|"
    # EL ACUERDO MISMO, en voz del instructor: «se tiene por precluido», «se
    # desecha la ampliación por extemporánea», «no ha lugar a admitir».
    r"se\s+(?:tiene|tuvo)\s+por\s+precluid\w+|"
    r"se\s+desech\w+\s+(?:la\s+|el\s+)?(?:ampliaci[óo]n|prueba|recurso|demanda|promoci[óo]n)|"
    r"no\s+ha\s+lugar\s+a\s+(?:admitir|tener|acordar)", re.I)
_RX_TEMA_INCIDENTE = re.compile(
    r"reclamaci[óo]n|recurso|incidente|(?:auto|acuerdo|prove[íi]do)\s+(?:recurrid|impugnad|reclamad)\w*|"
    r"preclu\w+|ampliaci[óo]n\s+de\s+(?:la\s+)?demanda|extempor[áa]ne\w+|desech\w+", re.I)
_RX_ESCRITO_DE_PARTE = re.compile(
    r"conceptos?\s+de\s+violaci[óo]n|vengo\s+a|solicit[oa]\b|\bPRESENTE\b|bajo\s+protesta|"
    r"en\s+mi\s+car[áa]cter|H\.\s+TRIBUNAL|se[ñn]alando\s+como\s+domicilio|"
    r"personalidad\s+que\s+tengo|el\s+suscrito|la\s+suscrita", re.I)
_RX_SENTENCIA_DEFINITIVA = re.compile(
    r"sentencia\s+definitiva|se\s+declara\s+la\s+nulidad|R\s*E\s*S\s*U\s*E\s*L\s*V\s*E|"
    r"conceptos?\s+de\s+impugnaci[óo]n", re.I)

# Un contrato, un acta, un peritaje: constancia, no resolución.
_RX_CONSTANCIA = re.compile(
    r"contrato\s+colectivo|cl[áa]usula\s+\d|acta\s+(?:de|circunstanciada)|dictamen\s+pericial|"
    r"peritaje|reglamento\s+interior\s+de\s+trabajo|convenio\b", re.I)


def clasificar(texto: str) -> tuple:
    """(clase, señales). `clase` ∈ {resolucion_procesal, constancia, otro}."""
    t = " ".join(str(texto or "").split())
    if not t:
        return OTRO, []
    senales = []
    n_dec = len(_RX_DECIDE_INCIDENTE.findall(t))
    n_tema = len(_RX_TEMA_INCIDENTE.findall(t))
    dens = n_tema * 1000.0 / max(len(t), 1)
    n_parte = len(_RX_ESCRITO_DE_PARTE.findall(t))
    n_def = len(_RX_SENTENCIA_DEFINITIVA.findall(t))
    if n_dec:
        senales.append(f"decide×{n_dec}")
    senales.append(f"tema {dens:.1f}/1k")
    if n_parte >= 3:
        senales.append(f"escrito de parte×{n_parte}")
    if n_def >= 3:
        senales.append(f"sentencia definitiva×{n_def}")
    # LA RESOLUCIÓN DEL INCIDENTE: decide sobre él, trata de él con densidad,
    # y no es un escrito de parte ni la sentencia definitiva.
    if n_dec >= 1 and dens >= 1.0 and n_parte < 3 and n_def < 3:
        return RESOLUCION_PROCESAL, senales
    if _RX_CONSTANCIA.search(t) and n_dec == 0:
        return CONSTANCIA, senales + ["constancia"]
    return OTRO, senales


# ═══ ¿CUÁL DE LOS PROBLEMAS ES LA VIOLACIÓN PROCESAL? ══════════════════════
#
# La fase 3 ahora lo declara en `clase` («procesal»). Para las sesiones
# anteriores, y como respaldo, el vocabulario de lo que se combate: la
# pregunta del 93/2026 no dice «violación procesal» por ningún lado —dice
# «¿debió admitir la ampliación de demanda…?»— y la marca vieja («violaci» +
# «procesal» en la pregunta) no la veía.
_RX_PROBLEMA_PROCESAL = re.compile(
    r"violaci[óo]n(?:es)?\s+(?:al\s+)?proce(?:sal|dimiento)|reposici[óo]n\s+del\s+procedimiento|"
    r"ampliaci[óo]n\s+de\s+(?:la\s+)?demanda|preclu\w+|desech\w+\s+(?:la\s+|el\s+)?(?:prueba|ampliaci[óo]n|recurso|demanda)|"
    r"(?:falta|ilegal(?:idad)?)\s+de\s+emplazamiento|indebid\w+\s+emplazamiento|"
    r"(?:admisi[óo]n|desahogo)\s+de\s+(?:la\s+|una\s+)?prueba|recurso\s+de\s+reclamaci[óo]n|"
    r"no\s+(?:se\s+)?(?:admiti[óo]|desahog[óo]|tuvo\s+por\s+(?:ofrecid|presentad)\w+)", re.I)


def es_problema_procesal(p) -> bool:
    """¿Este problema (dict de la fase 3 o texto) es una violación procesal?"""
    if isinstance(p, dict):
        if str(p.get("clase") or "").strip().lower() == "procesal":
            return True
        if str(p.get("clase") or "").strip().lower() in ("fondo", "procedencia"):
            return False
        t = " ".join(str(p.get(k) or "") for k in ("pregunta", "combate", "resolvio"))
    else:
        t = str(p or "")
    return bool(_RX_PROBLEMA_PROCESAL.search(t))


def hay(problemas: list, criterios: list = None, contexto: str = "") -> bool:
    """¿Este asunto trae una violación procesal que estudiar?"""
    for p in (problemas or []):
        if es_problema_procesal(p):
            return True
    for c in (criterios or []):
        if es_problema_procesal(str(getattr(c, "problema", "") or
                                    (c.get("problema") if isinstance(c, dict) else ""))):
            return True
    return clasificar(contexto)[0] == RESOLUCION_PROCESAL


# ═══ CÓMO ENTRA AL PROMPT ══════════════════════════════════════════════════

def rotulo(clase: str) -> str:
    if clase == RESOLUCION_PROCESAL:
        return "LA RESOLUCIÓN QUE DECIDIÓ LA VIOLACIÓN PROCESAL — consta en autos"
    if clase == CONSTANCIA:
        return "CONSTANCIA DE AUTOS QUE EL ACERVO NO TIENE"
    return "DOCUMENTO APORTADO POR EL SECRETARIO — no estaba en el acervo"


def instrucciones(clase: str, para: str = "estudio") -> str:
    """Lo que el modelo tiene que hacer con lo aportado, según lo que sea.

    `para` ∈ {propuesta, estudio}: en la propuesta se decide el sentido con la
    razón toral correcta; en el estudio se escribe la confrontación.
    """
    if clase != RESOLUCION_PROCESAL:
        return ("Lo trae quien tiene el expediente delante. Cítalo por lo que dice,\n"
                "como constancia de autos; NO le inventes un registro ni lo trates\n"
                "como jurisprudencia.")
    base = (
        "En amparo directo la violación procesal se examina sobre la actuación\n"
        "que la causó y sobre la resolución del recurso ordinario que la confirmó\n"
        "(artículos 171 y 172 de la Ley de Amparo). LAS RAZONES DE ESTA RESOLUCIÓN\n"
        "SON LA RAZÓN TORAL del planteamiento procesal —no las de la sentencia\n"
        "definitiva, que sólo recogió su resultado—.\n")
    if para == "propuesta":
        return base + (
            "Para el planteamiento que la combate: toma como `razon_toral` lo que\n"
            "ESTA resolución sostuvo, enfrenta cada una de sus razones con la ley y\n"
            "la jurisprudencia del material, y propón el sentido según resistan o\n"
            "no. No propongas a partir de la regla abstracta: propón a partir de si\n"
            "las razones concretas de esta resolución se sostienen.")
    # ── LA TÉCNICA, REESCRITA EL 26-SEP-2026 (decisión 2 de David) ──────
    # Decía «ENUNCIA, una por una» y «ESTÁ PROHIBIDO despacharla en un
    # párrafo». El estudio del 93/2026 lo cumplió a la letra y escribió cuatro
    # párrafos, uno por razón, contestados con la misma frase (estandar2,
    # l. 104-109): la orden fabricaba la repetición que luego se le reprochaba.
    # Lo que se quería evitar —contestar con una conclusión que no nombra las
    # razones de la resolución— sigue prohibido; lo que cambia es que las que
    # caen por la misma respuesta se contestan juntas, la dependiente cae con
    # la otra salvo que un efecto la presuponga, y la mixta se parte.
    # Se describe la FUNCIÓN y no se dan frases: un ejemplo escrito aquí se
    # copia literal al proyecto (tres veces medido). Por eso también salió la
    # frase entre comillas que se daba como lo que NO había que escribir.
    # La misma regla, con las mismas cuatro piezas, está en
    # `tipos_asunto.TECNICA_RESOLUCION["directo_violacion_procesal"]`.
    return base + (
        "TÉCNICA OBLIGADA para el planteamiento que la combate:\n"
        "  1. IDENTIFICA las razones en que esta resolución se sostiene (qué\n"
        "     plazo aplicó, desde cuándo lo contó, con qué fundamento, por qué\n"
        "     tuvo por no exigible lo que la parte reclama). Ninguna se queda\n"
        "     sin respuesta.\n"
        "  2. CONFRONTA cada razón con la ley que rige la vía y con la\n"
        "     jurisprudencia obligatoria: di si esa razón resiste o cae, y por\n"
        "     qué. Si un criterio obligatorio dispone lo contrario de lo que la\n"
        "     resolución sostuvo, la razón cae y se dice con el registro delante.\n"
        "  3. LAS QUE CAEN POR LA MISMA RESPUESTA SE CONTESTAN JUNTAS,\n"
        "     nombrándolas a todas: una razón no pide desarrollo propio si lo que\n"
        "     la derriba es lo mismo que derriba a otra.\n"
        "  4. LA RAZÓN QUE DEPENDE DE OTRA CAE CON ELLA y basta decirlo,\n"
        "     nombrándola, SALVO QUE UN EFECTO DE LA CONCESIÓN LA PRESUPONGA —la\n"
        "     oportunidad, el cómputo o la cuantía con que la responsable tendrá\n"
        "     que actuar al reponer—: entonces se desarrolla con el dato del\n"
        "     expediente, o, si el material no lo trae, se dice en ADVERTENCIAS.\n"
        "  5. LA RAZÓN MIXTA SE PARTE: si una misma razón une afirmaciones que se\n"
        "     contestan de modo distinto —una regla de derecho y un hecho del\n"
        "     expediente, por ejemplo—, cada parte recibe su respuesta.\n"
        "  6. LA CALIFICACIÓN SALE DE ESE CONTRASTE: es fundado si cae al menos\n"
        "     una razón de la que dependía el resultado; infundado si todas\n"
        "     resisten. Y trasciende al fallo sólo si la actuación privó a la\n"
        "     parte de una defensa que podía cambiar el resultado —dilo.\n"
        "  7. NO BASTA una conclusión que no nombre las razones de la resolución,\n"
        "     ni razonar sobre una regla general sin bajar a lo que esta\n"
        "     resolución dijo. Cítala como «la interlocutoria de… que resolvió el\n"
        "     recurso de reclamación» o «el acuerdo de…», nunca como\n"
        "     «documento aportado».")


def bloque(contexto: str, para: str = "estudio", tope: int = 20000,
           recortar=None) -> str:
    """El bloque del prompt, con el rótulo y la técnica que le corresponden."""
    c = (contexto or "").strip()
    if not c:
        return ""
    clase, _ = clasificar(c)
    cuerpo = recortar(c, tope) if recortar else c[:tope]
    raya = "═" * 63
    return (f"\n{raya}\n{rotulo(clase)}\n{raya}\n"
            f"{instrucciones(clase, para)}\n\n{cuerpo}\n")
