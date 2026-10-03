# -*- coding: utf-8 -*-
"""EL EXAMINADOR: UN SEGUNDO RAZONADOR QUE ELIGE ENTRE LAS DOS VÍAS
(David, 2-oct-2026).

«El tribunal sólo sirve como guía en casos de precedentes claros, pero el motor
puede aprovechar su capacidad de razonar una respuesta en aplicación de una
jurisprudencia aplicable al caso, una aplicable temáticamente o, incluso, una
por analogía. El motor no está casado con los precedentes del tribunal (…) lo
que resuelve es la inteligencia en el razonamiento jurídico, por eso buscamos
el agente».

POR QUÉ HACE FALTA. Medido en el banco Kingston con el oro auditado sobre los
resolutivos (21 amparos directos): la propuesta del motor elegía conceder en el
71-83% de los asuntos que el tribunal NEGÓ, y acertaba 8 de 21. Pero en los 21
el motor YA había escrito la vía contraria «como si la defendiera»: el fallo no
era ver la solución, era ESCOGER (diagnóstico del sesgo v2, 29-sep).

QUÉ HACE. Otro modelo, de otra familia (Gemini 3.8 Flash con razonamiento
alto: es el que David vio detectar los errores del proyecto), recibe la
resolución y el escrito ÍNTEGROS, el análisis neutral, el material del acervo
y las DOS vías ya escritas, sin saber cuál propuso el motor (el orden A/B sale
del número del asunto, para que la posición no sesgue). Las examina con el
mismo rigor, por las fallas que el diagnóstico encontró en las concesiones
equivocadas —razón autónoma sin combatir, técnica antes que el fondo, premisa
de hecho no acreditada, requisito legal desplazado por una lectura protectora,
falta de trascendencia, jurisprudencia que no aplica, accesorio que arrastra—
y la suplencia de la queja cuando opera, y da su PROBABILIDAD RAZONADA de que
los conceptos o agravios prosperen. Esa probabilidad decide el lado (regla del
50.01%).

MEDIDO (2-oct-2026, sólo lectura de lo guardado, sin regenerar): el examinador
acertó 15 de 21 (71%) contra 8 de 21 del motor, sin ninguna estadística del
tribunal en el prompt. Cuesta ~45 s y unos 4 centavos de dólar por asunto
(≈55 mil tokens de entrada, 8 mil de razonamiento).

LO QUE NO HACE. No escribe razones nuevas (las dos vías ya están escritas), no
cambia lo que decida el secretario (corre sobre la propuesta, antes) y nunca
bloquea: si falla o vence, la propuesta se queda con el lado del motor.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time

MODELO = os.getenv("EXAMINADOR_MODELO", "gemini-3.8-flash")
TOPE_S = float(os.getenv("EXAMINADOR_TOPE_S", "150") or 150)
PENSAMIENTO = (os.getenv("EXAMINADOR_PENSAMIENTO", "high") or "high").strip().lower()
MAX_ACTO = 320_000
MAX_ESCRITO = 220_000
# LO QUE VIO EL MOTOR Y EL EXAMINADOR NO (revisión adversarial, 3-oct-2026):
# las constancias de autos, lo aportado con «Aportar y proponer», las
# respuestas del secretario y el bloque de la ejecutoria (todo armado por
# `main._con_autos`) más la ficha procesal. Tope PROPIO, para que unos autos
# enormes no se coman el presupuesto de la resolución y del escrito.
MAX_AUTOS = int(os.getenv("EXAMINADOR_MAX_AUTOS", "120000") or 120_000)
MAX_FICHA = 20_000
MAX_SALIDA = 24_000
VERSION = "examinador-1"

_cliente = None


def _gemini():
    """El cliente de la API de Gemini (clave GEMINI_API_KEY), uno por proceso."""
    global _cliente
    if _cliente is None:
        from google import genai
        clave = (os.getenv("GEMINI_API_KEY") or "").strip()
        if not clave:
            raise RuntimeError("sin GEMINI_API_KEY")
        _cliente = genai.Client(api_key=clave)
    return _cliente


def _t(x, n: int = 0) -> str:
    s = " ".join(str(x or "").split())
    return s[:n] if n else s


def _reg(x) -> str:
    return re.sub(r"\D", "", str(x or ""))


def prospera(sentido) -> bool | None:
    import probabilidad_sentido as _ps
    v = _ps.prospera(sentido)
    return None if v is None else bool(v)


# ═══ EL PROMPT ═══════════════════════════════════════════════════════════════

def _bloque_analisis(a) -> str:
    if not isinstance(a, dict) or not (a.get("razones") or a.get("hechos")):
        return "(sin análisis previo)"
    L = [f"CUESTIÓN CENTRAL: {_t(a.get('cuestion_central'))}", "RAZONES DE LO RESUELTO:"]
    for r in (a.get("razones") or [])[:20]:
        L.append(f"  {r.get('id')} [{r.get('relacion')}] {_t(r.get('afirma'), 400)} → {_t(r.get('conclusion'), 300)}"
                 + (f" (cita: «{_t(r.get('cita'), 300)}»)" if r.get("cita") else ""))
    L.append("ARGUMENTOS DEL ESCRITO (qué razón atacan):")
    for x in (a.get("argumentos") or [])[:40]:
        L.append(f"  {x.get('segmento')}: pide {_t(x.get('pide'), 250)} · porque {_t(x.get('por_que'), 250)}"
                 f" · ataca {', '.join(x.get('combate') or []) or 'ninguna'}")
    L.append("HECHOS (quién los afirma y su condición probatoria):")
    for h in (a.get("hechos") or [])[:35]:
        L.append(f"  {h.get('id')} {_t(h.get('que'), 250)} — afirma: {h.get('afirma')} — {h.get('condicion')}")
    for p in (a.get("premisas") or [])[:15]:
        try:
            import analisis_litis as _al_e
            _estado_p = _al_e.estado_de_premisa(p)
        except Exception:
            _estado_p = str(p.get("el_acto") or "")
        L.append(f"  PREMISA: la parte afirma {_t(p.get('afirma_la_parte'), 250)} — la resolución: "
                 f"{_estado_p}" + (f" — respuesta del secretario: {p.get('respuesta')}" if p.get("respuesta") else ""))
    if a.get("autonomas_sin_combatir"):
        L.append("RAZONES AUTÓNOMAS QUE NINGÚN ARGUMENTO ATACA (cálculo del análisis): "
                 + ", ".join(a["autonomas_sin_combatir"]))
    return "\n".join(L)


def _bloque_via(nombre: str, v: dict, tesis_por_reg: dict) -> str:
    L = [f"VÍA {nombre}: los conceptos/agravios son {str(v.get('sentido') or '').upper()}",
         f"  Razón: {_t(v.get('razon'))}", f"  Qué pasa con lo demás: {_t(v.get('efecto'))}"]
    for r in [str(x) for x in (v.get("apoyos") or [])][:6]:
        t = tesis_por_reg.get(_reg(r)) or {}
        L.append(f"  Apoyo {r}: {_t(t.get('rubro'), 300) or '(no está en el acervo)'}")
    return "\n".join(L)


def _bloque_tesis(tesis: list, citadas: set) -> str:
    orden = sorted([t for t in tesis if isinstance(t, dict)],
                   key=lambda t: (_reg(t.get("registro")) not in citadas, not t.get("obligatoria")))
    L = []
    for t in orden[:30]:
        reg = _reg(t.get("registro"))
        L.append(f"- [{reg}] {t.get('instancia') or ''} {t.get('clave_tesis') or ''} · "
                 f"{'JURISPRUDENCIA' if t.get('obligatoria') else 'tesis aislada'}\n  {_t(t.get('rubro'), 350)}"
                 + (f"\n  Texto: {_t(t.get('texto'), 1200)}" if reg in citadas else ""))
    return "\n".join(L) or "(sin tesis)"


def _bloque_normas(normas: list) -> str:
    return "\n".join(f"- {n.get('cuerpo_legal')}, art. {n.get('articulo')}: {_t(n.get('texto'), 700)}"
                     for n in (normas or [])[:25] if isinstance(n, dict)) or "(sin normas)"


def _sin_amparo(tipo: str) -> bool:
    """¿El asunto no es juicio de amparo (revisión fiscal)? Ahí no hay artículo
    79 y el estricto derecho es un hecho de la ley, no una presunción."""
    try:
        import tipos_asunto as _ta
        t = _ta.normalizar(tipo or "") or ""
    except Exception:
        t = ""
    t = t or "_".join(str(tipo or "").lower().replace("ó", "o").split())
    return t == "revision_fiscal"


def _rotulo(fr: str) -> str:
    try:
        import suplencia as _sp
        return _sp.rotulo(fr) or fr
    except Exception:
        return fr


def _regla_suplencia(suplencia: dict | None, tipo: str = "") -> str:
    """La regla 8 del prompt: la suplencia de la queja, DICHA COMO HECHO SÓLO SI
    EL SECRETARIO LA CONFIRMÓ (revisión adversarial, 3-oct-2026).

    Antes se le daba al examinador `suplencia.proponer_de(r)` —la propuesta
    determinista, que el secretario aún no ha visto— como si fuera la decisión:
    una VII inferida de lo que la parte AFIRMA de sí misma («soy de escasos
    recursos») llegaba como «OPERA la suplencia», y un «ninguna» por no
    encontrar patrón llegaba como «rige el estricto derecho» en un familiar
    con un menor que los patrones no vieron. La suplencia es un umbral que
    decide él (`suplencia.py`): sin su confirmación es un INDICIO, ni se
    presume ni se excluye. `suplencia["confirmada"] is True` es la marca: la
    pone main sólo con la del secretario (`r.encargo.suplencia`)."""
    sp = suplencia if isinstance(suplencia, dict) else {}
    fr = str(sp.get("fraccion") or "").strip()
    cola = ("\n   Donde opera, la deficiencia del planteamiento o que no se combata una consideración no lo vuelve\n"
            "   inoperante: el tribunal examina de oficio la legalidad de lo resuelto en perjuicio de esa parte. Pero la\n"
            "   suplencia no crea pruebas ni releva de probar los hechos que a esa parte le tocaba acreditar. Donde no\n"
            "   opera, sólo se examina lo que se planteó y como se planteó.")
    cab = "\n8. La suplencia de la queja (artículo 79 de la Ley de Amparo). "
    if _sin_amparo(tipo):
        # No es juicio de amparo: no hay 79 que decidir (`suplencia.proponer`).
        return (cab + "Este asunto no es un juicio de amparo: el artículo 79 no rige y se examina en "
                "estricto derecho, sólo lo que se planteó y como se planteó.")
    if not fr:
        return ""
    if sp.get("confirmada") is True:
        if fr != "ninguna":
            dato = (f"El secretario CONFIRMÓ que en este asunto OPERA la suplencia de la queja: "
                    f"{sp.get('rotulo') or _rotulo(fr)}, a favor de {sp.get('a_favor_de') or 'quien promueve'}.")
        else:
            dato = ("El secretario CONFIRMÓ que en este asunto NO opera la suplencia de la queja: rige el "
                    "estricto derecho.")
        return cab + dato + cola
    # SIN CONFIRMAR: la propuesta automática, sus alternativas y lo que pide
    # la parte, como indicios. «Ninguna» automática NO es «estricto derecho».
    ind = []
    if fr != "ninguna":
        ind.append(f"la propuesta automática apunta a la {sp.get('rotulo') or _rotulo(fr)}"
                   + (f" ({_t(sp.get('porque'), 300)})" if sp.get("porque") else ""))
    else:
        ind.append("la propuesta automática no encontró en el texto ningún supuesto (eso NO afirma el "
                   "estricto derecho: la I y la VI no se leen desde el texto, y la II o la VII dependen de "
                   "hechos que el patrón puede no ver)")
    alts = [a for a in (sp.get("alternativas") or []) if isinstance(a, dict) and a.get("fraccion")]
    if alts:
        ind.append("también podría operar la " + "; la ".join(
            f"{a.get('rotulo') or _rotulo(str(a.get('fraccion')))}"
            + (f" ({_t(a.get('porque'), 200)})" if a.get("porque") else "") for a in alts[:4]))
    ped = str(sp.get("pedida") or "").strip()
    if ped and ped != "ninguna":
        ind.append(f"la parte la pide expresamente ({_rotulo(ped)}); que la pida no la acredita")
    return (cab + "El secretario TODAVÍA NO HA DECIDIDO si opera la suplencia de la queja: no la "
            "presumas ni la excluyas. Indicios, no hechos del asunto: " + "; ".join(ind) + ".\n"
            "   Si la suerte de una vía depende de que opere o no la suplencia, dilo en «razon_decisiva» y en\n"
            "   «que_lo_cambiaria» en lugar de darla por cierta." + cola)


def _recortar(texto: str, n: int) -> str:
    """Recorta por EL MEDIO, no por la cola: `_con_autos` pone delante las
    respuestas del secretario y detrás de los autos lo aportado y el bloque
    de la ejecutoria; cortar por la cola los perdería con autos largos."""
    texto = str(texto or "")
    if len(texto) <= n:
        return texto
    cabeza = int(n * 0.6)
    cola = n - cabeza
    return (texto[:cabeza] + f"\n[… se omiten {len(texto) - n} caracteres de en medio …]\n"
            + texto[-cola:])


def _bloque_autos(autos: str, ficha: str) -> str:
    """El bloque de lo que vio el motor y no está en la resolución ni en el
    escrito («» si no hay nada). Va ANTES de la resolución."""
    autos, ficha = str(autos or "").strip(), str(ficha or "").strip()
    if not (autos or ficha):
        return ""
    partes = ["CONSTANCIAS DE AUTOS, LO APORTADO AL EXPEDIENTE, LAS RESPUESTAS DEL SECRETARIO Y LA FICHA "
              "PROCESAL (lo mismo que tuvo delante quien escribió las dos vías; las premisas de hecho se "
              "verifican también contra esto, y lo que aquí consta no se tiene por «no acreditado»):"]
    if ficha:
        partes.append("FICHA PROCESAL:\n<<<\n" + _recortar(ficha, MAX_FICHA) + "\n>>>")
    if autos:
        partes.append("<<<\n" + _recortar(autos, MAX_AUTOS) + "\n>>>")
    return "\n".join(partes) + "\n\n"


def prompt(tipo: str, problemas: list, via_a: dict, via_b: dict, analisis, tesis: list, normas: list,
           acto: str, escrito: str, suplencia: dict | None = None, autos: str = "", ficha: str = "") -> str:
    """El prompt del examinador. `autos` es el contexto que armó `main._con_autos`
    (constancias, lo aportado, respuestas del secretario, la ejecutoria) y
    `ficha` la ficha procesal: lo que vio el motor al escribir las vías
    (revisión adversarial, 3-oct-2026). Sin ellos, el prompt es el de antes."""
    bloque_autos = _bloque_autos(autos, ficha)
    # LA REGLA 3 VERIFICABA LAS PREMISAS SÓLO CONTRA LA RESOLUCIÓN: la
    # constancia de notificación que el secretario aportó no existía para el
    # examinador, que tachaba «premisa no acreditada» lo que obraba en autos.
    regla3_autos = (" También consta lo que obra en el bloque de constancias de autos, lo aportado y las\n"
                    "   respuestas del secretario: contra eso también se verifica." if bloque_autos else "")
    tesis_por_reg = {_reg(t.get("registro")): t for t in (tesis or []) if isinstance(t, dict)}
    citadas = {_reg(x) for v in (via_a, via_b) for x in (v.get("apoyos") or [])}
    probs = "\n".join(f"  {i}. [{p.get('jerarquia', '')}] {_t(p.get('pregunta'), 300)}"
                      for i, p in enumerate(problemas or [], 1) if isinstance(p, dict)) or "  (sin problemas)"
    tipo_txt = (tipo or "amparo directo").replace("_", " ")
    return f"""Eres magistrado de un Tribunal Colegiado de Circuito. Tienes delante un {tipo_txt} y DOS
proyectos de solución opuestos, escritos por un secretario que no eres tú. No sabes cuál prefiere. Tu trabajo
es decidir, con razonamiento jurídico, cuál se sostiene, y decir con qué probabilidad los conceptos de violación
o agravios PROSPERAN (es decir, que el tribunal conceda o revoque).

CÓMO SE EXAMINA (en este orden, y con el mismo rigor para las dos vías):
1. La razón toral de lo resuelto: qué consideraciones sostienen el fallo y cuáles bastan SOLAS. Si una razón
   autónoma no es combatida por ningún concepto o agravio, el fallo se sostiene por ella aunque lo demás prospere.
2. La técnica antes que el fondo: un planteamiento novedoso, que no combate la consideración, que reitera lo
   ya dicho sin atacar la respuesta, genérico o dogmático, o que parte de una premisa falsa, es inoperante.
3. Las premisas de hecho: lo que la parte AFIRMA en su escrito no es un hecho del asunto. Se verifica contra lo
   que la resolución tuvo por acreditado, lo que valoró y lo que consta en ella. Si la resolución lo desmiente o
   no lo tiene por probado y la carga era de quien lo afirma, la premisa no está acreditada.{regla3_autos}
4. El derecho aplicable: el requisito legal expreso manda. Una lectura protectora, pro persona o de interpretación
   conforme no suprime un requisito que la ley exige ni sustituye la prueba que faltó.
5. La trascendencia: que la parte tenga un punto no basta. Una omisión, un error de valoración o un vicio formal
   sólo prospera si cambia el resultado; si lo resuelto se sostiene igual, es fundado pero inoperante o
   insuficiente. «Que vuelva a valorar» exige mostrar que la nueva valoración puede llevar a otro sentido.
6. La jurisprudencia: para cada criterio que invocan las vías, di si aplica DIRECTAMENTE al supuesto, si aplica
   por el TEMA, si aplica por ANALOGÍA o si NO APLICA (y por qué). Una jurisprudencia obligatoria que resuelve el
   supuesto pesa más que una tesis aislada; una que sólo comparte palabras no resuelve nada.
7. Los accesorios no arrastran: un planteamiento accesorio débil no hace prosperar el asunto si el principal no
   prospera, salvo que por sí solo dé un beneficio propio.{_regla_suplencia(suplencia, tipo)}

Al final, tu probabilidad de que los conceptos o agravios prosperen (de 0 a 1). Es tu juicio jurídico sobre ESTE
asunto, no una estadística. Si dudas, comprométete igual con el lado que te parezca más probable: el secretario
necesita una propuesta, no una abstención.

PROBLEMAS JURÍDICOS PLANTEADOS:
{probs}

{_bloque_via('A', via_a, tesis_por_reg)}

{_bloque_via('B', via_b, tesis_por_reg)}

ANÁLISIS NEUTRAL PREVIO DE LA LITIS (hecho por otro modelo; verifícalo, no lo des por cierto):
{_bloque_analisis(analisis)}

JURISPRUDENCIA DEL ACERVO (las citadas por las vías llevan su texto):
{_bloque_tesis(tesis or [], citadas)}

NORMAS DEL ACERVO:
{_bloque_normas(normas)}

{bloque_autos}RESOLUCIÓN RECLAMADA O RECURRIDA, ÍNTEGRA:
<<<
{str(acto or '')[:MAX_ACTO]}
>>>

ESCRITO DE CONCEPTOS DE VIOLACIÓN O AGRAVIOS, ÍNTEGRO:
<<<
{str(escrito or '')[:MAX_ESCRITO]}
>>>

En «razon_decisiva» y «que_lo_cambiaria» nombra cada vía POR SU SENTIDO («la que declara fundados los
conceptos», «la que los niega por inoperantes»), NUNCA por su letra: quien lo lee no ve las letras A y B.
«p_prospera» va de 0 a 1 (0.35, no 35) y tiene que ir con el «lado» que eliges.

Devuelve SÓLO un JSON con esta forma:
{{"razones_torales": [{{"afirma": "<qué sostiene el fallo>", "autonoma": true, "combatida": true}}],
 "examen": {{
   "A": {{"fallas": [{{"gravedad": "fatal|seria|menor", "tipo": "autonoma_sin_combatir|tecnica|premisa_no_acreditada|derecho_aplicable|trascendencia|jurisprudencia_no_aplica|otra", "explicacion": "<una o dos frases>", "evidencia": "<pasaje literal breve de la resolución o del escrito>"}}],
          "jurisprudencia": [{{"registro": "<registro>", "aplica": "directa|tematica|analogia|no_aplica", "por_que": "<una frase>"}}],
          "solidez": <0 a 10>}},
   "B": {{"fallas": [], "jurisprudencia": [], "solidez": <0 a 10>}}}},
 "p_prospera": <0 a 1>,
 "lado": "A|B",
 "razon_decisiva": "<la razón que decide, tres o cuatro renglones>",
 "que_lo_cambiaria": "<qué dato o criterio movería la decisión, un renglón>"}}"""


# ═══ LA LLAMADA Y SU LECTURA ═════════════════════════════════════════════════

_RX_JSON = re.compile(r"\{.*\}", re.S)


def leer(crudo: str) -> dict | None:
    """El JSON del examinador, o None si no se puede leer."""
    m = _RX_JSON.search(crudo or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    return d if isinstance(d, dict) else None


def propuesta_es_a(numero: str) -> bool:
    """¿Va la propuesta del motor como vía A? Sale del número del asunto: fijo
    para el mismo asunto (reproducible) y repartido entre asuntos (la posición
    no empuja siempre al mismo lado)."""
    return int(hashlib.sha1(str(numero or "").encode("utf-8")).hexdigest(), 16) % 2 == 0


def normalizar_p(x) -> float | None:
    """`p_prospera` a la escala de 0 a 1 (revisión adversarial, 3-oct-2026).

    El modelo puede devolver el porcentaje (35 por 0.35: el mismo JSON le pide
    `solidez` de 0 a 10). Antes se recortaba a [0, 1] y un 35 se volvía 1.0:
    100 % a CONCEDER, el sesgo que el examinador vino a corregir. Entre 1 y 100
    se lee como porcentaje; por encima de 100, o negativo, o no numérico, no
    hay número (None) y decide la letra."""
    try:
        p = float(x)
    except (TypeError, ValueError):
        return None
    if p != p or p < 0:  # NaN o negativo
        return None
    if p > 100:
        return None
    if p > 1:
        p = p / 100.0
    return round(p, 4)


# LAS LETRAS A/B NO LAS VE EL SECRETARIO (revisión adversarial, 3-oct-2026): el
# orden sale del hash del número y es relativo a la propuesta del motor ANTES
# del volteo; la pantalla dice «Te propongo» y «¿O resolverías en sentido
# opuesto?». Una razón que dice «la vía B se sostiene» se lee al revés en la
# mitad de los asuntos. Se sustituyen por código, además de pedírselo al
# modelo. Sólo con un sustantivo o artículo delante: una «A» suelta es la
# preposición, y «apartado B» del artículo 123 es derecho, no una vía.
_ARTICULO = {"la": "la", "el": "la", "del": "de la", "al": "a la", "una": "una", "esta": "esta",
             "esa": "esa", "otra": "otra"}
_RX_AMBAS = re.compile(r"\b(?:(las|Las)\s+)?(v[íi]as|V[íi]as|proyectos|Proyectos|opciones|Opciones)\s+"
                       r"[«\"']?A[»\"']?\s+y\s+[«\"']?B[»\"']?(?![\w])")
_RX_LETRA = re.compile(
    r"(?<![\w])(?:(la|La|el|El|del|Del|al|Al|una|Una|esta|Esta|esa|Esa|otra|Otra)\s+)?"
    r"(?:(v[íi]a|V[íi]a|lado|Lado|proyecto|Proyecto|opci[óo]n|Opci[óo]n|postura|Postura|soluci[óo]n|Soluci[óo]n)\s+)?"
    r"([«\"'(]?)([AB])([»\"')]?)(?![\w])(?!\s+qu(?:o|em)\b)")  # «el A quo» es latín, no una vía


def limpiar_letras(texto: str, propuesta_a: bool, via_ganadora: str) -> str:
    """«vía A / vía B» → «la vía que se propone» / «la vía contraria», según
    qué vía ganó el examen (`via_ganadora`: «propuesta» | «alternativa», en
    términos del motor) y qué letra llevaba la propuesta del motor."""
    if not texto:
        return texto or ""

    def _nombre(letra: str) -> str:
        via_letra = "propuesta" if (letra == "A") == bool(propuesta_a) else "alternativa"
        return "vía que se propone" if via_letra == via_ganadora else "vía contraria"

    def _ambas(m):
        return ("Las" if (m.group(1) or "").startswith("L") else "las") + " dos vías"

    def _una(m):
        art, sust, abre, letra, cierra = m.groups()
        if not (art or sust):
            # «A» suelta: la preposición, no una vía. Sólo entre comillas o
            # paréntesis («B», (A)) es inequívocamente la letra de una vía.
            if not (abre and cierra):
                return m.group(0)
            nombre = f"la {_nombre(letra)}"
            return f"({nombre})" if abre == "(" else nombre
        mayus = (art or sust or "")[:1].isupper()
        a = _ARTICULO.get((art or "la").lower(), "la")
        a = a[:1].upper() + a[1:] if mayus else a
        return f"{a} {_nombre(letra)}"

    texto = _RX_AMBAS.sub(_ambas, texto)
    return _RX_LETRA.sub(_una, texto)


def decidir(d: dict, propuesta_a: bool, via_propuesta: dict, via_alternativa: dict) -> dict | None:
    """De la salida del modelo a la decisión: qué vía gana, con qué probabilidad
    de que prospere. La probabilidad manda (regla del 50.01%) cuando va con la
    letra; si no la trae o no se lee, manda el lado que eligió. Ninguna de las
    dos → None.

    SI EL NÚMERO CONTRADICE LA LETRA (revisión adversarial, 3-oct-2026), manda
    la LETRA y el número se descarta (`p_prospera` None, `incoherente` True).
    Por qué la letra y no el motor: la letra es la elección explícita y es la
    que razonan `razon_decisiva` y el examen de cada vía (escritos por letra);
    el número es el campo que se equivoca de escala o de sentido. Dejar que
    mande el motor tiraría un examen cuyo razonamiento sí es coherente."""
    if not isinstance(d, dict):
        return None
    lado = str(d.get("lado") or "").strip().upper()
    via_lado = None
    if lado in ("A", "B"):
        via_lado = "propuesta" if (lado == "A") == propuesta_a else "alternativa"
    p = normalizar_p(d.get("p_prospera"))
    sp = prospera(via_propuesta.get("sentido"))
    sa = prospera(via_alternativa.get("sentido"))
    via_p = None
    if p is not None and sp is not None and sa is not None and sp != sa and p != 0.5:
        quiere = p > 0.5
        via_p = "propuesta" if sp == quiere else "alternativa"
    incoherente = bool(via_p and via_lado and via_p != via_lado)
    if incoherente:
        print(f"   ⚖️ EXAMINADOR: p_prospera={p} contradice la letra «{lado}»; manda la letra, sin número")
        via, p = via_lado, None
    elif via_p:
        via = via_p
    elif via_lado:
        via = via_lado
    else:
        return None
    sv = sp if via == "propuesta" else sa
    ex = d.get("examen") if isinstance(d.get("examen"), dict) else {}
    ex_p = ex.get("A" if propuesta_a else "B") or {}
    ex_a = ex.get("B" if propuesta_a else "A") or {}

    def _limpio(x, n):
        return limpiar_letras(_t(x, n), propuesta_a, via)

    def _fallas(lst):
        out = []
        for f in [f for f in (lst or []) if isinstance(f, dict)][:8]:
            f = dict(f)
            if f.get("explicacion"):
                f["explicacion"] = limpiar_letras(str(f["explicacion"]), propuesta_a, via)
            out.append(f)
        return out

    return {"via": via, "p_prospera": p, "lado": ("prospera" if sv else "no_prospera") if sv is not None else None,
            "razon_decisiva": _limpio(d.get("razon_decisiva"), 1200),
            "que_lo_cambiaria": _limpio(d.get("que_lo_cambiaria"), 400),
            "fallas": {"propuesta": _fallas(ex_p.get("fallas")),
                       "alternativa": _fallas(ex_a.get("fallas"))},
            "jurisprudencia": {"propuesta": [j for j in (ex_p.get("jurisprudencia") or []) if isinstance(j, dict)][:8],
                               "alternativa": [j for j in (ex_a.get("jurisprudencia") or []) if isinstance(j, dict)][:8]},
            "solidez": {"propuesta": ex_p.get("solidez"), "alternativa": ex_a.get("solidez")},
            "coincide_con_su_lado": (via_lado is None or via_lado == via),
            "incoherente": incoherente}


async def examinar(numero: str, tipo: str, problemas: list, via_propuesta: dict, via_alternativa: dict,
                   analisis, tesis: list, normas: list, acto: str, escrito: str,
                   suplencia: dict | None = None, tope_s: float | None = None,
                   autos: str = "", ficha: str = "") -> dict | None:
    """La decisión del examinador entre las dos vías, o None (sin clave, sin
    las dos vías, falla o vence). Nunca lanza. `autos` y `ficha`: lo que vio el
    motor (ver `prompt`)."""
    t0 = time.time()
    try:
        if not (str(via_propuesta.get("sentido") or "").strip() and str(via_alternativa.get("sentido") or "").strip()):
            return None
        if prospera(via_propuesta.get("sentido")) == prospera(via_alternativa.get("sentido")):
            # Las dos vías van al mismo lado: no hay nada que escoger.
            return None
        a_es_p = propuesta_es_a(numero)
        va, vb = (via_propuesta, via_alternativa) if a_es_p else (via_alternativa, via_propuesta)
        texto = prompt(tipo, problemas, va, vb, analisis, tesis, normas, acto, escrito, suplencia,
                       autos=autos, ficha=ficha)
        from google.genai import types as gt
        cfg = gt.GenerateContentConfig(
            temperature=0.2, response_mime_type="application/json", max_output_tokens=MAX_SALIDA,
            thinking_config=gt.ThinkingConfig(thinking_level=PENSAMIENTO))
        tope = float(tope_s or TOPE_S)
        d, uso = None, None
        for intento in range(2):
            restante = tope - (time.time() - t0)
            if restante < 15:
                break
            try:
                r = await asyncio.wait_for(
                    _gemini().aio.models.generate_content(model=MODELO, contents=texto, config=cfg),
                    timeout=restante)
            except asyncio.TimeoutError:
                print(f"   ⚖️ EXAMINADOR: venció el tope de {tope:.0f} s")
                return None
            d = leer(getattr(r, "text", "") or "")
            uso = getattr(r, "usage_metadata", None)
            if d is not None:
                break
            print(f"   ⚖️ EXAMINADOR: respuesta ilegible (intento {intento + 1})")
        dec = decidir(d, a_es_p, via_propuesta, via_alternativa) if d else None
        if dec is None:
            return None
        dec.update({"modelo": MODELO, "version": VERSION, "segundos": round(time.time() - t0, 1),
                    "tokens": {"entrada": getattr(uso, "prompt_token_count", None),
                               "razonamiento": getattr(uso, "thoughts_token_count", None),
                               "salida": getattr(uso, "candidates_token_count", None)}})
        print(f"   ⚖️ EXAMINADOR ({MODELO}): gana la {dec['via']} · p(prospera)={dec['p_prospera']} · "
              f"{dec['segundos']} s · tokens {dec['tokens']}")
        return dec
    except Exception as ex:
        print(f"   ⚖️ EXAMINADOR: falló ({type(ex).__name__}: {str(ex)[:160]}); se queda la propuesta del motor")
        return None


def explicacion(dec: dict, sentido: str, volteada: bool, sentido_motor: str) -> str:
    """La frase que lee el secretario: el lado, la probabilidad y la razón."""
    p = dec.get("p_prospera")
    lado_p = (p if dec.get("lado") == "prospera" else (1 - p)) if isinstance(p, (int, float)) else None
    pct = f", con una probabilidad del {int(round(lado_p * 100))}%" if lado_p is not None else ""
    base = f"Se propone «{str(sentido).replace('_', ' ')}»{pct}, según el examen de las dos vías"
    if volteada and sentido_motor:
        base += (f" (el motor se inclinaba por «{str(sentido_motor).replace('_', ' ')}»; su razón queda "
                 f"como vía contraria)")
    razon = dec.get("razon_decisiva") or ""
    return (base + ". " + razon).strip()
