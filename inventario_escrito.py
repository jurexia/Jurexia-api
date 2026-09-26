# -*- coding: utf-8 -*-
"""LOS ARGUMENTOS LEÍDOS DEL ESCRITO — lo que el resumen de la fase 2 no trae.

POR QUÉ EXISTE. El inventario de la v3/v4 (`inventario.segmentos`) sale del
RESUMEN de conceptos de la fase 2. Medido el 26-sep-2026 con el localizador a
ciego (7 asuntos, 104 argumentos leídos del ESCRITO, emparejados a mano con el
inventario): 12 de las 27 omisiones graves de las corridas v3/v4 eran
argumentos que NO ESTÁN en el inventario porque el resumen no los trae —la
testimonial de cargo aleccionada y la falsedad de la actora sobre el hijo del
103/2025, el oficio mal pedido al RAN del 263/2025, la reiteración de la carga
de la prueba en el primer concepto del 43/2025—. Ninguna comprobación sobre
las marcas puede ver lo que no está en la lista. La v1 no los pierde porque
escribe largo con el escrito a la vista.

QUÉ HACE. Una llamada al modelo de las fases, con razonamiento (alto por
omisión: ver `ESFUERZO`), lee el escrito ENTERO —con los renglones del PDF repuestos— y
devuelve sus argumentos: una razón de ilegalidad con su dato propio, su
concepto, una cita literal y los renglones del inventario que ya la recogen.
Luego, SIN MODELO:
  · la cita se VERIFICA palabra por palabra contra el escrito (sin mayúsculas,
    acentos ni puntuación, como V0); de lo que el modelo copió se queda el
    tramo literal más largo, si tiene de 8 a 40 palabras y es al menos la
    mitad de lo copiado. Lo que no se verifica se descarta;
  · la FUSIÓN conserva el piso tal cual —mismos ids, mismo orden— y añade como
    segmentos nuevos del mismo concepto, con los ids que siguen la serie, los
    argumentos que el modelo no casó con ningún renglón de ese concepto y cuya
    cita no pisa la de un segmento del piso (ni la de otro nuevo) del mismo
    concepto.

POR QUÉ EL MODELO DICE QUÉ RENGLÓN LO RECOGE, Y NO SÓLO EL CÓDIGO. Se midió
primero la fusión sin modelo, con los 104 argumentos del localizador haciendo
de «extraídos» y el emparejamiento a mano como verdad (sim_offline, 26-sep):
el parecido léxico ponderado con el renglón que de verdad los recoge va de
0.00 a 0.72 (mediana ~0.35) y el de los 11 que no están, de 0.10 a 0.34; la
distancia entre citas, de 0 a 44,000 caracteres para los recogidos. Un resumen
parafrasea y una lectura parafrasea otra vez: por palabras no se separan. El
código se queda con lo que sí puede comprobar —la cita, el concepto, que un
id exista y sea del mismo concepto, y el solape de pasajes— y el modelo con
el juicio de si un renglón ya enuncia esa razón con su dato.

CALIBRACIÓN (26-sep-2026, invesc-3, esfuerzo alto; 15 corridas en los 8 asuntos
del banco, 103/2025, 263/2025 y 43/2025 repetidos; cada segmento nuevo leído a
mano contra el localizador ciego):
  · EXHAUSTIVIDAD. De los 104 argumentos del localizador, el piso trae 93; el
    piso con la lectura, 101. De los 12 pares de omisión grave que no estaban
    en el inventario se recuperan 11 en TODAS las repeticiones (la testimonial
    aleccionada y la falsedad sobre el hijo del 103/2025, el oficio al RAN del
    263/2025). No sale nunca la reiteración de la carga de la prueba en el
    primer concepto del 43/2025 (0 de 3), ni dos invocaciones genéricas.
  · PRECISIÓN. 22 segmentos nuevos en la primera pasada: 8 son argumentos que
    el piso no trae, 1 otra faceta de uno de ellos, 8 el mismo argumento que
    un renglón del piso con el dato propio que ese renglón no nombra (el
    contrato, el folio real, la hipótesis del criterio de costas) y 5 duplican
    sin dato nuevo. Ninguno inventado: toda cita es literal. Lo que más
    duplica es un escrito largo con renglones del piso genéricos (43/2025: 4 de
    9, y 0 de 3 en otra repetición) o conceptos que sólo son un rótulo.
  · TAMAÑO. El bloque del inventario pasa de 55,706 a 65,745 caracteres en los
    8 asuntos (+18 %); el que más, +52 % (43/2025). Sin lectura, idéntico.
  · COSTE Y TIEMPO. 0.013-0.022 USD por asunto (media 0.018) y 82-168 s
    (mediana 120), en segundo plano desde que termina el adelanto.
  · ESTABILIDAD. Los graves del 103/2025 salen en sus 4 corridas y el del
    263/2025 en sus 3; el número de nuevos varía (103: 7/7/5/6; 263: 1/2/4;
    43: 9/7/3). La temperatura 0 no sirve: el modelo con razonamiento la
    rechaza.

LO QUE NO HACE: decidir sentidos, calificar, ni cambiar un segmento del piso.
El inventario ORGANIZA. Sólo para la v3 y la v4: la v1 no lo ve nunca.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import os
import re
import time
from collections import Counter

import inventario as _inv

# ═══ CONSTANTES ═════════════════════════════════════════════════════════════
# Sube cuando cambie el prompt, la llamada, la verificación o la fusión: lo
# guardado con otra versión no se reutiliza (entra en la huella).
VERSION = "invesc-3"

# Razonamiento ALTO por omisión, medido (26-sep-2026, 103/2025, mismo prompt):
# con «medium» el modelo razonó 485 tokens, leyó 19 argumentos y NO separó ni
# la testimonial de cargo aleccionada ni la falsedad sobre el hijo —las dos
# omisiones graves que motivan esto—; con «high» razonó ~9,000, leyó 43 y dio
# las dos. Cuesta ~0.02 USD y ~2 minutos, y corre en segundo plano. Nunca
# «none»: leer sin razonar es lo que ya hace la fase 2, y ahí se pierden.
ESFUERZO = os.getenv("INV_ESCRITO_ESFUERZO", "high")
TOPE_SALIDA = int(os.getenv("INV_ESCRITO_TOPE_SALIDA", "24000"))
# El escrito que ve el modelo, en caracteres. El más largo del banco (43/2025)
# son 132,000; el tope deja pasar los de doscientas páginas enteros.
TOPE_ESCRITO = int(os.getenv("INV_ESCRITO_TOPE_CARACTERES", "320000"))

CITA_MIN, CITA_MAX = 8, 40
# Del texto que el modelo presenta como cita, la parte literal tiene que ser al
# menos esta proporción: si copió ocho palabras y parafraseó treinta, no copió.
PROPORCION_LITERAL = 0.5
# Dos citas del mismo concepto que comparten esta proporción de las palabras
# de la más corta son el mismo pasaje.
SOLAPE_MISMO_PASAJE = 0.5
# EL BLOQUE NO SE DISPARA: como mucho se añaden estos segmentos por asunto
# (y nunca más que el piso entero). Lo que sobra se registra, no se añade.
MAX_NUEVOS = int(os.getenv("INV_ESCRITO_MAX_NUEVOS", "12"))
TEXTO_MAX = 60
DATO_MAX = 20
# LO QUE EL MODELO DICE QUE UN RENGLÓN DEL PISO YA RECOGE SE COMPRUEBA: tiene
# que copiar de ese renglón las palabras que nombran el dato del argumento
# (`PALABRAS_MIN` seguidas, literales), y esas palabras tienen que compartir con
# el argumento leído raíces con peso —lo raras que son en el escrito,
# `inventario._Escrito.peso`— de al menos `PESO_DATO`. Medido en la primera
# corrida del 103/2025 (invesc-1): sin esta comprobación el modelo casó con un
# renglón del piso los 27 argumentos que leyó, entre ellos la testimonial de
# cargo aleccionada, que ningún renglón nombra (compartía con el renglón sólo
# «testigo» y «expresan»). Sin comprobar, un «ya está» del modelo no vale.
PALABRAS_MIN = 2
# 2.5 y no 3.0: con 3.0 caían reclamos buenos cuyo dato es una palabra que el
# escrito repite («exhaustiva» en el 263/2025 pesa 2.8; «declaraciones» en el
# 103/2025, 2.9); los que deben caer quedan muy por debajo («la testimonial»
# frente a la testimonial aleccionada, 1.6). Medido en invesc-2, esfuerzo alto.
PESO_DATO = 2.5

# El estado en la fila (`taller_sesiones.plan`, rama «inventario»).
LATIDO_S = 20.0
ABANDONADA_S = 3 * LATIDO_S
TOPE_CORRIDAS = int(os.getenv("INV_ESCRITO_TOPE_CORRIDAS", "3"))
# Lo que se espera, como mucho, a una extracción en curso o recién lanzada:
# dentro de la tarea del resolver (el estudio espera además al plan, 120 s),
# en /taller/plan/pedir (una petición: corto) y en el plan adelantado (suelto).
# Con esfuerzo alto la lectura tardó de 82 a 168 s en las 15 corridas de la
# calibración (mediana 120; los 168 son el 263/2025): 180 deja que la que se
# calcula dentro de la tarea (una sesión de antes del despliegue) llegue. Lo
# normal es que ya esté: empieza al terminar el adelanto.
ESPERA_RESOLVER_S = float(os.getenv("INV_ESCRITO_ESPERA_S", "180"))
ESPERA_PEDIDO_S = float(os.getenv("INV_ESCRITO_ESPERA_PEDIDO_S", "15"))
ESPERA_ADELANTADO_S = float(os.getenv("INV_ESCRITO_ESPERA_ADELANTADO_S", "300"))


def interruptor() -> str:
    """`INVENTARIO_ESCRITO`: «off» apaga la lectura (el inventario vuelve a ser
    el piso, sin más) · «casa» (por omisión) la precalcula para las cuentas de
    casa y para quien tenga la v3/v4 como variante global de su tipo · «todos»
    la precalcula para todos. Al generar se usa SÓLO con la v3 y la v4."""
    m = (os.getenv("INVENTARIO_ESCRITO", "casa") or "casa").strip().lower()
    if m in ("off", "0", "no", "apagado"):
        return "off"
    return "todos" if m in ("todos", "sombra", "on", "1") else "casa"


def aplica_variante(variante: str) -> bool:
    return str(variante or "").strip().lower() in ("v3", "v4")


# ═══ EL ESCRITO, POR PALABRAS ═══════════════════════════════════════════════

def _fuentes(fases) -> list:
    return list(_inv._get(fases, "fuentes", None) or []) + ["", ""]


def escrito_de(fases) -> str:
    return str(_fuentes(fases)[1] or "")


class Lectura:
    """El escrito normalizado (`inventario.normalizar_escrito`), partido en
    palabras sin mayúsculas ni acentos, con la posición de cada una. Sirve
    para verificar una cita palabra por palabra y devolver el pasaje tal como
    está en el escrito."""

    def __init__(self, escrito: str):
        self.norm, _ = _inv.normalizar_escrito(escrito or "")
        pl = _inv._plano(self.norm)
        self.toks = [(m.start(), m.end()) for m in re.finditer(r"\w+", pl)]
        palabras = [pl[a:b] for a, b in self.toks]
        self.cadena = " " + " ".join(palabras) + " "
        self.ini = []
        p = 1
        for w in palabras:
            self.ini.append(p)
            p += len(w) + 1

    @staticmethod
    def palabras(texto: str) -> list:
        t = str(texto or "").replace("­", "")
        return re.findall(r"\w+", _inv._plano(_inv._colapsar(t)))

    def _buscar(self, ws: list) -> int:
        """Índice de la palabra del escrito donde empieza la secuencia, o -1."""
        if not ws:
            return -1
        i = self.cadena.find(" " + " ".join(ws) + " ")
        if i < 0:
            return -1
        return bisect.bisect_left(self.ini, i + 1)

    def localizar(self, cita: str) -> tuple:
        """(primera palabra, última palabra, largo del tramo literal, palabras
        de la cita). El tramo es el más largo de la cita que está seguido en el
        escrito; (-1, -1, 0, n) si ninguno llega a `CITA_MIN`."""
        ws = self.palabras(cita)
        mejor = (-1, -1, 0)
        i = 0
        while i + CITA_MIN <= len(ws):
            j = i + CITA_MIN
            if self._buscar(ws[i:j]) < 0:
                i += 1
                continue
            while j < len(ws) and self._buscar(ws[i:j + 1]) >= 0:
                j += 1
            k = self._buscar(ws[i:j])
            if j - i > mejor[2]:
                mejor = (k, k + (j - i) - 1, j - i)
            i = j
        return mejor[0], mejor[1], mejor[2], len(ws)

    def pasaje(self, a: int, b: int) -> str:
        """El texto del escrito de la palabra `a` a la `b`, tal como está."""
        return self.norm[self.toks[a][0]:self.toks[b][1]].strip()


def texto_legible(escrito: str) -> str:
    """Lo que lee el modelo: el escrito con los renglones del PDF repuestos y
    los saltos que no parten palabra (ver `inventario.normalizar_escrito`)."""
    t, _ = _inv.normalizar_escrito(escrito or "", saltos=True)
    return re.sub(r"\n{2,}", "\n", t)


# ═══ LA HUELLA ═══════════════════════════════════════════════════════════════

def huella(fases, escrito: str = "", es_recurso: bool = False, piso: list = None) -> str:
    """Lo que identifica una lectura: la versión, el escrito, el conteo y el
    piso (los ids con lo que dicen: `en_inventario` los nombra). Depende sólo
    del adelanto; nunca del criterio."""
    esc = escrito or escrito_de(fases)
    if piso is None:
        piso = _inv.segmentos(fases, esc, es_recurso) or []
    conteo = _inv._get(fases, "conteo", None) or {}
    base = {"v": VERSION,
            "escrito": hashlib.sha1(esc.encode("utf-8", "ignore")).hexdigest()[:16],
            "conteo": [str(conteo.get("estado") or ""), int(conteo.get("n") or 0)],
            "rec": bool(es_recurso),
            "piso": [[s.get("id"), s.get("concepto"), s.get("texto")] for s in piso]}
    return hashlib.sha1(json.dumps(base, ensure_ascii=False, sort_keys=True)
                        .encode()).hexdigest()[:16]


# ═══ EL PROMPT ═══════════════════════════════════════════════════════════════
# Lección de la casa, medida tres veces: lo que va en un prompt como ejemplo se
# copia literal. Aquí no hay ni una frase para imitar ni un dato de un asunto
# real: sólo la descripción de qué es un argumento y de cada campo.

def _vocabulario(tipo: str, es_recurso: bool) -> tuple:
    """(q plural, q1 singular, nombre del escrito)."""
    try:
        import tipos_asunto as _ta
        voc = _ta.vocabulario_de(tipo or ("amparo_revision" if es_recurso else "amparo_directo"))
        q1 = voc["combate_singular"]
    except Exception:
        q1 = "agravio" if es_recurso else "concepto de violación"
    q = "agravios" if es_recurso else "conceptos de violación"
    if q1.startswith("agravio"):
        q = "agravios"
    escrito = "un escrito de agravios" if q == "agravios" else "una demanda de amparo"
    return q, q1, escrito


def _bloque_piso(piso: list, q1: str) -> str:
    """Los renglones del piso, ENTEROS (≤ 80 palabras): el modelo copia de
    ellos las palabras que nombran el dato y el código las busca ahí."""
    if not piso:
        return "(el resumen no dejó renglones: el inventario está vacío)"
    fuera = []
    for s in piso:
        fuera.append(f"{s.get('id')} · {q1} {s.get('concepto')} · "
                     f"{_inv._colapsar(str(s.get('texto') or ''))}")
    return "\n".join(fuera)


def prompt(fases, escrito: str, piso: list, es_recurso: bool = False,
           tipo: str = "") -> str:
    q, q1, nombre = _vocabulario(tipo, es_recurso)
    legible = texto_legible(escrito)
    recorte = ""
    if len(legible) > TOPE_ESCRITO:
        cab = int(TOPE_ESCRITO * 0.7)
        legible = legible[:cab] + "\n[…]\n" + legible[-(TOPE_ESCRITO - cab):]
        recorte = (f"\nEl escrito es muy largo y se muestra recortado por en medio "
                   f"(marca «[…]»): trabaja con lo que ves.")
    inferido = any(s.get("concepto_inferido") for s in piso or [])
    numeracion = (
        f"el número del {q1} en que la parte lo plantea, con la misma numeración "
        f"de los renglones del inventario de abajo"
        + (" (ahí el número es el orden de los párrafos del resumen: úsalo igual)"
           if inferido else "")
        + f"; si el escrito tiene un {q1} que el inventario no numera, sigue la "
        f"numeración del escrito")
    return f"""Vas a leer completo un {nombre} y a levantar el inventario de sus argumentos: cada razón por la que la parte sostiene que el acto que combate es ilegal. El inventario sirve para que quien redacte la sentencia no deje ninguno sin contestar. No decides si tienen razón.

QUÉ ES UN ARGUMENTO
- Una razón de ilegalidad con su dato propio: el hecho, la prueba, la constancia o el testigo, el precepto con su ley, la tesis o el precedente, o la consecuencia que se pide, que la distingue de las demás razones.
- Dos razones con datos propios distintos son dos argumentos aunque estén en el mismo párrafo o en el mismo {q1}.
- La razón que el escrito repite en otro {q1} también es argumento de ese otro {q1}.
- Lo que sólo sostiene una razón ya dicha —la tesis transcrita en su apoyo, la doctrina, la misma idea con otras palabras— forma parte de ese argumento; no es otro.
- La consecuencia que se pide por una razón (reponer el procedimiento, revocar, que se ordene algo) va con esa razón como su dato; es argumento aparte sólo si el escrito la pide por una razón que no ha dado antes.
- La invocación genérica de preceptos o derechos, sin razón propia, es argumento sólo si el escrito le dedica un {q1} o un apartado propio.
- Cuenta también la razón de ilegalidad formulada fuera del capítulo de {q} (en los antecedentes o en el petitorio) si el escrito la presenta como motivo de agravio o pide una consecuencia por ella; va con el {q1} con que se relaciona.
- No son argumentos: los antecedentes narrados que no se hacen valer como motivo, la transcripción de la sentencia reclamada o de tesis, los datos formales de la demanda (autoridades, acto, fechas de notificación), la procedencia, la oportunidad ni la suspensión.

LO QUE DEVUELVES: un objeto JSON {{"argumentos": [ ... ]}}, un elemento por argumento, en el orden en que aparecen en el escrito, con estos campos:
- "concepto": {numeracion}.
- "argumento": la razón, en tercera persona y en presente, con su dato propio; no más de {TEXTO_MAX} palabras.
- "dato": sólo el dato propio que la distingue (nombres, constancia, artículo con su ley, número de registro, cifra, fecha, consecuencia pedida); no más de {DATO_MAX} palabras.
- "cita": de {CITA_MIN} a {CITA_MAX} palabras SEGUIDAS copiadas literalmente del escrito, del pasaje donde la parte plantea esa razón —no de la transcripción de la sentencia ni de una tesis—. Cópialas como están, con sus errores de escritura y de OCR: el sistema comprueba cada cita contra el escrito y descarta el argumento cuya cita no esté.
- "en_inventario": los renglones del inventario de abajo, DEL MISMO {q1}, que ya enuncian esta razón con su dato propio, cada uno como {{"id": su identificador, "palabras": las palabras de ESE renglón, copiadas tal cual y seguidas, que nombran el dato propio de este argumento}}. Si no puedes copiar del renglón las palabras que nombran ese dato, el renglón no lo recoge aunque trate el mismo tema o el mismo tipo de prueba: déjalo fuera. Un renglón de otro {q1} tampoco lo recoge. Lista vacía si ninguno lo recoge. El sistema comprueba esas palabras contra el renglón.
Recorre el escrito entero y devuelve todos los argumentos, estén o no en el inventario.{recorte}

EL INVENTARIO QUE YA EXISTE (sacado del resumen del adelanto; identificador · {q1} · lo que dice el resumen):
{_bloque_piso(piso, q1)}

EL ESCRITO:
{legible}
"""


# ═══ LA LLAMADA ═════════════════════════════════════════════════════════════

def _modelo() -> str:
    m = os.getenv("MODELO_INV_ESCRITO", "").strip()
    if m:
        return m
    import fases123_pipeline as _f123
    return _f123.MODELO_FASES


def _uso(r) -> dict:
    try:
        u = getattr(r, "usage", None)
        det = getattr(u, "completion_tokens_details", None)
        return {"entrada": int(getattr(u, "prompt_tokens", 0) or 0),
                "salida": int(getattr(u, "completion_tokens", 0) or 0),
                "razonamiento": int(getattr(det, "reasoning_tokens", 0) or 0)}
    except Exception:
        return {}


# Precio del modelo de las fases (gpt-5.6-luna por la API de OpenAI), USD por
# millón de tokens: el mismo que documenta `fases123_pipeline`. Sólo para medir.
PRECIO_ENTRADA = float(os.getenv("INV_ESCRITO_PRECIO_ENTRADA", "0.20"))
PRECIO_SALIDA = float(os.getenv("INV_ESCRITO_PRECIO_SALIDA", "1.20"))


def coste(uso: dict) -> float:
    return round((uso.get("entrada", 0) * PRECIO_ENTRADA
                  + uso.get("salida", 0) * PRECIO_SALIDA) / 1e6, 5)


async def _llamar(cliente, texto: str) -> tuple:
    """(respuesta, uso sumado). JSON estricto, por `llamada_modelo.crear`. Si
    vuelve vacía o cortada, otra vez con el doble de cupo y EL MISMO
    razonamiento: leer sin pensar es lo que ya hace la fase 2.

    SIN temperatura ni semilla, a propósito. Se probó ponerlas como las fases
    (26-sep-2026, 43/2025 y 103/2025): el modelo con razonamiento rechaza
    `temperature` y `llamada_modelo` repite la petición sin ellas. No compran
    determinismo y cuestan una petición rechazada por lectura. La estabilidad
    se midió repitiendo la lectura (ver la tabla de calibración)."""
    import llamada_modelo as _lm
    kw = dict(model=_modelo(), messages=[{"role": "user", "content": texto}],
              max_completion_tokens=TOPE_SALIDA,
              response_format={"type": "json_object"})
    if ESFUERZO:
        kw["reasoning_effort"] = ESFUERZO
    r = await _lm.crear(cliente, **kw)
    uso = _uso(r)
    txt = (r.choices[0].message.content or "").strip()
    fin = str(getattr(r.choices[0], "finish_reason", "") or "").lower()
    if not txt or fin == "length":
        kw["max_completion_tokens"] = TOPE_SALIDA * 2
        r = await _lm.crear(cliente, **kw)
        u2 = _uso(r)
        uso = {k: uso.get(k, 0) + u2.get(k, 0) for k in set(uso) | set(u2)}
        uso["llamadas"] = 2
        txt = (r.choices[0].message.content or "").strip()
    else:
        uso["llamadas"] = 1
    return txt, uso


def leer_json(txt: str) -> list:
    """La lista de argumentos crudos, o [] si no hay JSON legible."""
    d = None
    try:
        d = json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", str(txt or ""), re.S)
        if m:
            try:
                d = json.loads(m.group(0))
            except Exception:
                d = None
    if isinstance(d, dict):
        d = d.get("argumentos")
    return [x for x in (d or []) if isinstance(x, dict)] if isinstance(d, list) else []


# ═══ LA VERIFICACIÓN (sin modelo) ════════════════════════════════════════════

def _entero(x):
    try:
        return int(str(x).strip().split()[0].rstrip(".º°"))
    except Exception:
        m = re.search(r"\d+", str(x or ""))
        if m:
            return int(m.group(0))
        o = _inv._ORD.get(_inv._plano(str(x or "")).strip())
        return o if o and o > 0 else None


def _palabras_max(t: str, n: int) -> str:
    ps = _inv._colapsar(str(t or "")).split(" ")
    return " ".join(ps[:n]) if len(ps) > n else " ".join(ps)


def conceptos_validos(piso: list, conteo: dict) -> set:
    """Los números de concepto que un argumento leído puede llevar: los del
    piso y los que contó el contador, y los que queden entre medias (un resumen
    que se salta el segundo no quita el segundo al escrito)."""
    vistos = {int(s.get("concepto") or 0) for s in piso or []} - {0}
    n = int((conteo or {}).get("n") or 0) if str((conteo or {}).get("estado")) == "contado" else 0
    tope = max(list(vistos) + [n, 0])
    return set(range(1, tope + 1)) if tope else set()


def _concepto_por_tramo(conteo: dict, lec: Lectura, pos_norm: int, escrito_pos: list):
    """El concepto cuyo tramo del contador contiene la cita, o None."""
    if str((conteo or {}).get("estado")) != "contado":
        return None
    try:
        crudo = escrito_pos[pos_norm]
    except Exception:
        return None
    tramos = conteo.get("tramos") or []
    valores = conteo.get("valores") or list(range(1, len(tramos) + 1))
    for v, tr in zip(valores, tramos):
        try:
            if int(tr[0]) <= crudo < int(tr[1]):
                return int(v)
        except Exception:
            continue
    return None


def verificar(crudos: list, escrito: str, piso: list, conteo: dict = None) -> tuple:
    """([argumentos verificados], Counter de descartes). Cada verificado:
    {concepto, texto, dato, cita (canónica, del escrito), en_inventario (ids
    del piso DEL MISMO concepto que existen), palabras [a, b]}."""
    lec = Lectura(escrito)
    _, pos_crudo = _inv.normalizar_escrito(escrito or "")
    validos = conceptos_validos(piso, conteo or {})
    ids_piso = {_inv_id(s.get("id")): int(s.get("concepto") or 0) for s in piso or []}
    texto_piso = {_inv_id(s.get("id")): str(s.get("texto") or "") for s in piso or []}
    # El peso de cada raíz: lo rara que es en ESTE escrito (la misma medida
    # con que el piso ancla sus citas).
    peso = _inv._Escrito(escrito or "").peso
    fuera, desc = [], Counter()
    for x in crudos or []:
        texto = _palabras_max(x.get("argumento") or x.get("texto") or "", TEXTO_MAX)
        if len(texto.split()) < 4:
            desc["sin argumento"] += 1
            continue
        a, b, n, total = lec.localizar(str(x.get("cita") or ""))
        if a < 0 or n < CITA_MIN:
            desc["cita no verificada"] += 1
            continue
        if n < PROPORCION_LITERAL * total:
            desc["cita parafraseada"] += 1
            continue
        if n > CITA_MAX:
            b = a + CITA_MAX - 1
        c = _entero(x.get("concepto"))
        if validos and c not in validos:
            c2 = _concepto_por_tramo(conteo or {}, lec, lec.toks[a][0], pos_crudo)
            if c2 in validos:
                c = c2
            else:
                desc["concepto fuera del escrito"] += 1
                continue
        if not c or c < 1:
            desc["sin concepto"] += 1
            continue
        dato = _palabras_max(x.get("dato") or "", DATO_MAX)
        cita = lec.pasaje(a, b)
        en_inv, dichos = [], []
        for i in (x.get("en_inventario") or []):
            k, pal = _reclamo(i)
            dichos.append({"id": k or str(i)[:12], "palabras": pal[:300]})
            if not k or k in en_inv:
                continue
            if ids_piso.get(k) != c:
                desc["«ya está» en otro concepto o id que no existe"] += 1
                continue
            ok, motivo = recoge(texto_piso[k], pal, texto + " " + dato + " " + cita, peso)
            if ok:
                en_inv.append(k)
            else:
                desc[f"«ya está» sin comprobar ({motivo})"] += 1
        fuera.append({"concepto": c, "texto": texto, "dato": dato,
                      "cita": cita, "palabras": [a, b],
                      "en_inventario": en_inv,
                      # Lo que el modelo dijo, se aceptara o no: para medir.
                      "en_inventario_modelo": dichos[:6]})
    return fuera, desc


# Las siglas de tres letras son dato («NIP», «RAN», «CFF»), y las raíces del
# inventario (`inventario._raices`) las dejan fuera por cortas. Medido en el
# 43/2025 (invesc-3): el renglón que nombra «el NIP» no podía recoger el
# argumento del NIP. Fuera de las vacías, una palabra de tres letras cuenta; su
# peso, el de una palabra que el escrito casi no usa (no está en su índice).
_CORTAS_VACIAS = set("""
que los las del por con una sus son fue han hay mas sin sea ser asi dos era
han les nos ese esa eso uno tal ley art fr fracc num pag cfr etc sea via
""".split())


def _raices_dato(texto: str) -> set:
    base = _inv._raices(texto)
    for w in re.findall(r"[a-z0-9ñ]+", _inv._plano(texto or "")):
        if len(w) == 3 and not w.isdigit() and w not in _CORTAS_VACIAS \
                and w not in _inv._VACIAS:
            base.add(w)
    return base


def _reclamo(x) -> tuple:
    """(id, palabras) de un elemento de «en_inventario»: {"id", "palabras"} o,
    si el modelo devolvió sólo el id, (id, «»)."""
    if isinstance(x, dict):
        return _inv_id(x.get("id")), _inv._colapsar(str(x.get("palabras") or ""))
    return _inv_id(x), ""


def recoge(texto_renglon: str, palabras: str, argumento: str, peso) -> tuple:
    """(¿el renglón recoge el argumento?, motivo). Las palabras que el modelo
    copió del renglón tienen que estar en él, seguidas (`PALABRAS_MIN` como
    mínimo; de una copia imperfecta vale el tramo literal más largo), y
    compartir con el argumento leído raíces de peso `PESO_DATO` o más: el dato,
    no el tema."""
    ws = Lectura.palabras(palabras)
    if len(ws) < PALABRAS_MIN:
        return False, "sin palabras del renglón"
    ren = " " + " ".join(Lectura.palabras(texto_renglon)) + " "
    mejor = []
    for i in range(len(ws)):
        j = i + PALABRAS_MIN
        if j > len(ws) or (" " + " ".join(ws[i:j]) + " ") not in ren:
            continue
        while j < len(ws) and (" " + " ".join(ws[i:j + 1]) + " ") in ren:
            j += 1
        if j - i > len(mejor):
            mejor = ws[i:j]
    if not mejor:
        return False, "palabras que no están en el renglón"
    comunes = _raices_dato(" ".join(mejor)) & _raices_dato(argumento)
    if sum(peso(r) for r in comunes) < PESO_DATO:
        return False, "palabras que no nombran el dato"
    return True, ""


def _inv_id(x) -> str:
    """«c1.A», «C1a», «(C1.a)» → «C1.a»; «» si no es un id."""
    m = re.search(r"\b([CA])\s*(\d+)\s*\.?\s*([a-z]{1,2})\b", str(x or ""), re.I)
    if not m:
        return ""
    return f"{m.group(1).upper()}{m.group(2)}.{m.group(3).lower()}"


# ═══ LA FUSIÓN (sin modelo, pura) ════════════════════════════════════════════

def _solape(a: tuple, b: tuple) -> float:
    """Proporción de palabras compartidas por dos tramos [a0, a1] y [b0, b1],
    sobre el más corto."""
    if not a or not b:
        return 0.0
    ini, fin = max(a[0], b[0]), min(a[1], b[1])
    if fin < ini:
        return 0.0
    return (fin - ini + 1) / max(1, min(a[1] - a[0], b[1] - b[0]) + 1)


def fusionar(piso: list, extraidos: list, escrito: str, es_recurso: bool = False) -> tuple:
    """(segmentos, informe). El piso, intacto y en su orden; detrás de cada
    concepto, los argumentos leídos que no recoge, con los ids que siguen la
    serie de ese concepto. Nunca lanza: si algo falla, el piso."""
    piso = [dict(s) for s in (piso or [])]
    informe = {"leidos": len(extraidos or []), "por_modelo": 0, "por_pasaje": 0,
               "repetidos": 0, "nuevos": 0, "sobrantes": 0}
    if not extraidos:
        return piso, informe
    try:
        lec = Lectura(escrito)
        pref = "A" if es_recurso else "C"
        # Los tramos del piso, en palabras del escrito.
        tramo_piso = {}
        for s in piso:
            if s.get("cita"):
                a, b, n, _ = lec.localizar(s["cita"])
                if a >= 0:
                    tramo_piso[s["id"]] = (a, b)
        inferido = any(s.get("concepto_inferido") for s in piso)
        propias = _inv.anclas_del_asunto(escrito or "")
        letras = Counter(int(s.get("concepto") or 0) for s in piso)
        nuevos = []
        for x in sorted(extraidos, key=lambda y: (int(y.get("concepto") or 0),
                                                  (y.get("palabras") or [0])[0])):
            c = int(x.get("concepto") or 0)
            if not c:
                continue
            if x.get("en_inventario"):
                informe["por_modelo"] += 1
                continue
            a, b = (x.get("palabras") or [None, None])[:2]
            if a is None:
                a, b, n, _ = lec.localizar(x.get("cita") or "")
                if a < 0:
                    continue
            tr = (int(a), int(b))
            if any(_solape(tr, tramo_piso.get(s["id"])) >= SOLAPE_MISMO_PASAJE
                   for s in piso if int(s.get("concepto") or 0) == c):
                informe["por_pasaje"] += 1
                continue
            if any(_solape(tr, y["_tramo"]) >= SOLAPE_MISMO_PASAJE
                   for y in nuevos if y["concepto"] == c):
                informe["repetidos"] += 1
                continue
            if len(nuevos) >= min(MAX_NUEVOS, max(1, len(piso))):
                informe["sobrantes"] += 1
                continue
            sid = f"{pref}{c}.{_inv._letra(letras[c])}"
            letras[c] += 1
            texto = _inv._recortar(str(x.get("texto") or ""))
            dato = str(x.get("dato") or "")
            nuevos.append({
                "id": sid, "concepto": c, "parrafo": None, "texto": texto,
                "cita": str(x.get("cita") or ""),
                "anclas": _inv.anclas_duras(texto + " \n " + dato + " \n " + str(x.get("cita") or ""),
                                            excluir=propias),
                "pagina": "", "parecido": None,
                "concepto_inferido": inferido,
                # De dónde sale: lo leyó el modelo en el escrito, no el resumen.
                "origen": "escrito", "dato": dato, "_tramo": tr})
        informe["nuevos"] = len(nuevos)
        for y in nuevos:
            y.pop("_tramo", None)
        # Cada nuevo, detrás del último segmento de su concepto (o donde toca
        # por número si el piso no tiene ese concepto).
        fuera = []
        por_c = {}
        for y in nuevos:
            por_c.setdefault(y["concepto"], []).append(y)
        conceptos_piso = [int(s.get("concepto") or 0) for s in piso]
        for i, s in enumerate(piso):
            fuera.append(s)
            c = conceptos_piso[i]
            siguiente = conceptos_piso[i + 1] if i + 1 < len(piso) else None
            if siguiente != c:
                # Antes de cerrar este concepto, los nuevos de conceptos que el
                # piso no tiene y van antes del siguiente.
                fuera.extend(por_c.pop(c, []))
                for k in sorted(list(por_c)):
                    if k < (siguiente if siguiente is not None else 10 ** 6) and k not in conceptos_piso:
                        fuera.extend(por_c.pop(k))
        for k in sorted(por_c):
            fuera.extend(por_c[k])
        return fuera, informe
    except Exception as ex:
        informe["error"] = type(ex).__name__
        return piso, informe


# ═══ LA EXTRACCIÓN ENTERA ═══════════════════════════════════════════════════

def _preparar(fases, esc: str, es_recurso: bool, tipo: str) -> tuple:
    """(piso, huella, prompt | «»), sin modelo."""
    piso = _inv.segmentos(fases, esc, es_recurso) or []
    h = huella(fases, esc, es_recurso, piso)
    texto = prompt(fases, esc, piso, es_recurso, tipo) if (esc.strip() and piso) else ""
    return piso, h, texto


async def extraer(cliente, fases, escrito: str = "", es_recurso: bool = False,
                  tipo: str = "") -> dict:
    """{estado, huella, argumentos, descartes, uso, coste, segundos, avisos}.
    «listo» con la lista verificada (puede ir vacía: el escrito no trae nada
    que el piso no tenga); «vacio» si no hay escrito o piso; «error» si el
    modelo falló (se puede reintentar). Nunca lanza."""
    import asyncio
    t0 = time.time()
    esc = escrito or escrito_de(fases)
    base = {"huella": "", "version": VERSION, "argumentos": [], "descartes": {},
            "uso": {}, "coste": 0.0, "avisos": []}
    try:
        # Lo que no es la llamada, FUERA DEL BUCLE DE EVENTOS: el piso y la
        # lectura normalizada de un escrito de doscientas páginas tardan
        # décimas, y en ese bucle corre el latido del flujo de otra pantalla.
        piso, h, texto = await asyncio.to_thread(_preparar, fases, esc, es_recurso, tipo)
    except Exception as ex:
        base.update(estado="error", segundos=round(time.time() - t0, 1),
                    avisos=[f"no se pudo preparar la lectura ({type(ex).__name__})"])
        return base
    base["huella"] = h
    if not esc.strip() or not piso:
        base.update(estado="vacio", segundos=round(time.time() - t0, 1),
                    avisos=["sin escrito o sin inventario del resumen: queda el piso"])
        return base
    try:
        txt, uso = await _llamar(cliente, texto)
    except Exception as ex:
        base.update(estado="error", segundos=round(time.time() - t0, 1),
                    avisos=[f"la lectura del escrito falló ({type(ex).__name__})"])
        return base
    crudos = leer_json(txt)
    conteo = _inv._get(fases, "conteo", None) or {}
    verif, desc = await asyncio.to_thread(verificar, crudos, esc, piso, conteo)
    base.update(estado="listo", argumentos=verif, descartes=dict(desc), uso=uso,
                coste=coste(uso), crudos=len(crudos),
                segundos=round(time.time() - t0, 1))
    if not crudos:
        base["estado"] = "error"
        base["avisos"] = ["el modelo no devolvió argumentos legibles"]
    return base


def segmentos(fases, escrito: str = "", es_recurso: bool = False, extraidos=None) -> list:
    """El inventario de la v3/v4: el piso y, si hay lectura, lo que añade."""
    return _inv.segmentos(fases, escrito, es_recurso, extraidos=extraidos)


# ═══ EL ESTADO EN LA FILA (taller_sesiones.plan, rama «inventario») ══════════
# Funciones PURAS: las aplica `main._taller_plan_cas`, el compare-and-set por
# `rev` del plan. La base es `plan_estudio._doc`: un documento de otro adelanto
# se rehace entero (el plan, las recalificaciones y la lectura de otro adelanto
# no sirven). gunicorn -w 2: nada de esto vive en memoria.

def _base(doc, huella_adelanto: str) -> dict:
    import plan_estudio as _pe
    d = _pe._doc(doc, huella_adelanto)
    if not isinstance(d.get("inventario"), dict):
        d["inventario"] = {}
    return d


def abandonada(c: dict, ahora: float) -> bool:
    if not isinstance(c, dict) or c.get("estado") != "en_curso":
        return False
    try:
        ultimo = float(c.get("latido") or c.get("desde") or 0)
    except (TypeError, ValueError):
        ultimo = 0.0
    return ahora - ultimo > ABANDONADA_S


def _casilla(doc, huella_adelanto: str, h: str):
    if not isinstance(doc, dict) or doc.get("huella") != huella_adelanto:
        return None
    c = doc.get("inventario")
    if isinstance(c, dict) and c.get("huella") == h:
        return c
    return None


def fila_pedir(doc, huella_adelanto: str, h: str, ahora: float,
               tope: int = TOPE_CORRIDAS) -> tuple:
    """(doc nuevo | None, decisión): «listo» · «vacio» · «en_curso» (se
    espera, no se duplica) · «tope» · «lanzar» (se reservó la corrida: quien
    llama la corre). Un «error» del proveedor o una corrida abandonada se
    vuelven a lanzar, y cuentan."""
    c = _casilla(doc, huella_adelanto, h)
    if c:
        if c.get("estado") in ("listo", "vacio"):
            return None, c["estado"]
        if c.get("estado") == "en_curso" and not abandonada(c, ahora):
            return None, "en_curso"
    d = _base(doc, huella_adelanto)
    prev = d["inventario"] if d["inventario"].get("huella") == h else {}
    corridas = int(prev.get("corridas") or 0)
    if corridas >= tope:
        return None, "tope"
    d["inventario"] = {"huella": h, "version": VERSION, "estado": "en_curso",
                       "desde": ahora, "latido": ahora, "corridas": corridas + 1}
    return d, "lanzar"


def fila_latido(doc, huella_adelanto: str, h: str, ahora: float) -> tuple:
    c = _casilla(doc, huella_adelanto, h)
    if not c or c.get("estado") != "en_curso":
        return None, "nada"
    d = _base(doc, huella_adelanto)
    d["inventario"]["latido"] = ahora
    return d, "latido"


def fila_resultado(doc, huella_adelanto: str, h: str, salida: dict, ahora: float) -> tuple:
    """Guarda la lectura. Si la fila ya es de otro adelanto o de otra huella,
    no escribe nada (lo leído es de otro asunto)."""
    if not isinstance(doc, dict) or doc.get("huella") != huella_adelanto:
        return None, "otro_adelanto"
    d = _base(doc, huella_adelanto)
    prev = d["inventario"]
    if prev and prev.get("huella") not in (None, h):
        return None, "otra_huella"
    est = str((salida or {}).get("estado") or "error")
    d["inventario"] = {
        "huella": h, "version": VERSION,
        "estado": est if est in ("listo", "vacio", "error") else "error",
        "argumentos": list((salida or {}).get("argumentos") or [])[:80],
        "descartes": dict((salida or {}).get("descartes") or {}),
        "uso": dict((salida or {}).get("uso") or {}),
        "coste": float((salida or {}).get("coste") or 0.0),
        "segundos": round(float((salida or {}).get("segundos") or 0), 1),
        "avisos": list((salida or {}).get("avisos") or [])[:10],
        "corridas": int(prev.get("corridas") or 1), "hecho": ahora}
    return d, d["inventario"]["estado"]


def guardado(doc, huella_adelanto: str, h: str, ahora: float) -> tuple:
    """(estado, argumentos) de la lectura guardada para esta huella: «listo»
    con su lista, «vacio», «en_curso», «fallo» (error o abandonada) o
    «ninguno»."""
    c = _casilla(doc, huella_adelanto, h)
    if not c:
        return "ninguno", []
    est = c.get("estado")
    if est == "en_curso":
        return ("fallo", []) if abandonada(c, ahora) else ("en_curso", [])
    if est in ("listo", "vacio"):
        return est, list(c.get("argumentos") or [])
    return "fallo", []
