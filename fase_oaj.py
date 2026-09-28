# -*- coding: utf-8 -*-
"""EL ESPEJO, DESDE EL ÍNDICE DE LA OAJ — los precedentes del propio tribunal,
planteamiento por planteamiento.

La tarjeta «Su propio tribunal» nació sobre el acervo viejo (`fase_espejo.py`,
léase entero antes de tocar esto): 3,585 sentencias de UN tribunal, con 2025
casi vacío, un umbral de coseno a pelo y una cobertura medida del 9%. Esta es
la fuente nueva: `oaj_precedentes`, el índice de las sentencias PÚBLICAS de los
tribunales del circuito en los cuatro tipos que el taller proyecta, y con los
planteamientos de las sentencias ya leídas desmontados uno a uno —pregunta, qué
se combatía, qué se resolvió, cómo se calificó y por qué—.

LO QUE CAMBIA RESPECTO DEL ESPEJO VIEJO, Y POR QUÉ
=================================================
 1. SE BUSCA PLANTEAMIENTO CONTRA PLANTEAMIENTO. El espejo viejo buscaba la
    pregunta del caso contra trozos de estudio de fondo, y entre las dos formas
    había una décima de coseno que era todo el margen. Aquí el punto indexado
    tiene la MISMA forma que el problema de la fase 3, así que la consulta es
    «pregunta combate resolvio» y no la pregunta sola: así se midió que casa
    mejor, y así se midió LA TABLA. Por eso un problema que no trae las tres
    piezas no consulta: su coseno pasado por una tabla medida con otra forma
    de consulta daría un porcentaje que ninguna medición respalda.

 2. EL PORCENTAJE NO ES EL COSENO. Un coseno de 0.62 no significa «62%» de
    nada, y cada tipo de asunto tiene su propia distribución. El coseno se
    convierte con una tabla de calibración isotónica (`oaj_calibracion.json`,
    versionada junto a este módulo) y sólo se enseña lo que la tabla pone en
    0.85 o más CON AL MENOS TRES PARES DETRÁS. SIN TABLA NO HAY PORCENTAJE: si
    el JSON no existe, no trae el tipo, o no trae un corte fiable, se devuelve
    una lista vacía y el taller cae al espejo viejo. Inventar un porcentaje
    sería de la misma especie que la tesis inventada.

 3. SÓLO DONDE SE CALIBRÓ. La tabla del 28-sep-2026 se midió dentro del 3TCC;
    entre tribunales el estilo del tema cambia y el corte puede moverse (lo
    dice el propio informe de calibración). Un «90%» en el Primer Tribunal
    calculado con la tabla del Tercero no sale de ninguna medición: fuera de
    los órganos calibrados, se calla.

 4. SIN PISO DE FILAS. El espejo viejo exigía tres sentencias porque su coseno
    crudo no distinguía el punto del vecino; una coincidencia calibrada al 85%
    vale sola.

 5. EL RESPALDO POR TEMA. Si ningún planteamiento pasa, se prueba con los
    puntos `clase=asunto`, cuyo vector es el de «tipo. tema» de la sentencia,
    con su propia tabla. La fila lo dice (`fuente: "tema"`): no es lo mismo
    coincidir en el punto que coincidir en la materia del asunto.

LO QUE SIGUE SIN SER, Y POR LAS MISMAS RAZONES
==============================================
Tampoco aquí se cuenta, se acusa ni se traduce el sentido a cubos. Las tres
mediciones que mataron al «contradictor» están en `fase_espejo.py` y valen
igual para este índice: el artículo 224 de la Ley de Amparo exige unanimidad y
el índice no trae votación; la calificación del planteamiento y el sentido del
resolutivo no son la misma escala. Aquí, además, el `sentido` de la fila es el
del resolutivo de la SENTENCIA y la coincidencia es con UNO de sus
planteamientos: uno calificado infundado dentro de una sentencia que concedió
por otro concepto se contaría como «concede». Por eso esta fuente no lleva
renglón de resumen: se enseñan los precedentes, su calificación literal y su
razón, y compara el secretario.

LA FORMA DEL JSON DE CALIBRACIÓN
================================
La escribe `redactor-sentencias/oaj/calibracion/calibrar_openai.py`:

    {
      "planteamiento": {"Amparo Directo": [[cos_min, cos_max, prob, n], ...]},
      "asunto":        {"Revisión Fiscal": [[cos_min, cos_max, prob, n], ...]},
      "umbral_85":     {"planteamiento": {...},
                        "asunto": {"Revisión Fiscal": 0.7612,
                                   "Amparo Directo": null}},
      "organos":       ["<nombre OAJ exacto>" | "3TCC", ...]      (opcional)
    }

`n` es cuántos pares calificados sostienen el tramo. Un tramo sin `n` (tres
columnas) se lee, pero no se sabe cuánto lo sostiene y NO cuenta. `umbral_85`
presente y `null` es el calibrador diciendo «aquí no hay corte fiable»: se
calla, no se deduce otro de la tabla. Si trae número, sólo puede ENDURECER el
corte (ver `_corte`). `organos` son los tribunales en que se midió; si falta,
vale sólo el 3TCC, que es donde se midió la del 28-sep-2026.

NO LO LLENA ESTE MÓDULO. La colección y la tabla las produce otro proceso; aquí
sólo se leen. Mientras la tabla no exista, este módulo calla siempre y el
taller sigue exactamente como estaba.
"""

import inspect
import json
import math
import os
import re

COLECCION = "oaj_precedentes"
VECTOR = "dense"            # text-embedding-3-small, 1536, coseno: embed_leyes

# ── LO QUE SE ENSEÑA ─────────────────────────────────────────────────────────
PROB_MINIMA = 0.85
# Un tramo de la isotónica sostenido por UN par da «1.0» con un solo ejemplo.
# El calibrador sólo admite como corte un tramo con tres o más pares; aquí se
# aplica la misma regla, también para leer el porcentaje.
PARES_MINIMOS = 3
# Ni con la tabla entera a favor se afirma certeza: «100% de similitud» es una
# promesa que ningún banco de un centenar de pares sostiene.
TOPE_VISIBLE = 99
MOSTRADAS = 6
# Se piden más de las que se muestran porque una sentencia trae varios
# planteamientos y varios del mismo asunto entran juntos: se queda el mejor de
# cada NEUN y con treinta suele haber seis distintos.
PEDIDOS_PLANTEAMIENTO = 30
PEDIDOS_ASUNTO = 12
# El embebedor topa en 8,192 tokens y el error llega como RetryError envuelto
# en BadRequestError, que aquí se leería como «no hay precedentes». Un
# planteamiento normal son unos cientos de caracteres; esto sólo corta el
# «combate» desmesurado.
MAX_CONSULTA = 12000

# No hay enlace profundo por NEUN en el Buscador de la OAJ: se enlaza el
# buscador y la pantalla ofrece el NEUN para copiarlo y pegarlo ahí.
ENLACE_OAJ = "https://ejusticia.cjf.gob.mx/BuscadorSISE/"

# LA COBERTURA, DICHA COMO ES. No «todas las sentencias»: (1) el filtro es de
# igualdad con el nombre ACTUAL del órgano, y lo que la OAJ publica con la
# denominación anterior —«Tercer Tribunal Colegiado del Vigésimo Segundo
# Circuito (01/08/2006 - 13/03/2016)», entre otros— no entra; (2) los
# planteamientos sólo existen para las sentencias ya leídas, y el resto se
# compara por su tema. Una tarjeta que dijera «todas» invitaría a leer la
# ausencia como «el tribunal nunca resolvió esto», que el índice no sostiene.
NOTA_COBERTURA = (
    "Índice de la OAJ, con el nombre actual del tribunal: lo que publicó con "
    "su denominación anterior no entra. Los planteamientos se comparan con los "
    "de las sentencias ya leídas y, si ninguno coincide, con el tema de los "
    "asuntos. Que un asunto no salga aquí no quiere decir que el tribunal no "
    "lo haya resuelto.")

RUTA_CALIBRACION = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "oaj_calibracion.json")

# ── LOS TIPOS, CON LA GRAFÍA DE LA OAJ ───────────────────────────────────────
# «Amparo en revisión» va con minúscula porque así lo escribe la OAJ, y el
# filtro es de palabra clave: «Amparo en Revisión», que es como lo guarda el
# acervo viejo, devolvería cero puntos sin error ninguno.
TIPOS_OAJ = {
    "amparo_directo": "Amparo Directo",
    "amparo_revision": "Amparo en revisión",
    "queja": "Queja",
    "revision_fiscal": "Revisión Fiscal",
}

# ── LOS ÓRGANOS DEL CIRCUITO 22, CON SU NOMBRE OAJ ACTUAL ────────────────────
# Las claves son las de `fase_espejo.resolver_tribunal`, que ya sabe leer el
# nombre que el secretario escribió en la ficha (y ya cazó el «Segundo» del
# Vigésimo Segundo Circuito leído como ordinal del tribunal). Aquí sólo se
# añade el nombre con el que la OAJ los indexa, que lleva la residencia.
#
# Van escritos enteros y no como TRIBUNALES_22 + sufijo: son el valor exacto
# de un filtro de palabra clave, y si un día el nombre para LEER cambia, el de
# FILTRAR no tiene por qué cambiar con él.
#
# Fuera del circuito 22 no hay mapa y la fuente OAJ calla, igual que el espejo
# viejo: imprimir el nombre de un tribunal deducido es imprimir uno que no es.
_RESIDENCIA = ", con residencia en Querétaro, Querétaro"
ORGANOS_OAJ = {
    "1TCC": "Primer Tribunal Colegiado en Materias Administrativa y Civil "
            "del Vigésimo Segundo Circuito" + _RESIDENCIA,
    "2TCC": "Segundo Tribunal Colegiado en Materias Administrativa y Civil "
            "del Vigésimo Segundo Circuito" + _RESIDENCIA,
    "3TCC": "Tercer Tribunal Colegiado en Materias Administrativa y Civil "
            "del Vigésimo Segundo Circuito" + _RESIDENCIA,
    "TCC_ADM": "Tribunal Colegiado en Materias Administrativa y de Trabajo "
               "del Vigésimo Segundo Circuito" + _RESIDENCIA,
    "TCC_PENAL": "Tribunal Colegiado en Materias Penal y Administrativa "
                 "del Vigésimo Segundo Circuito" + _RESIDENCIA,
}

# DÓNDE SE MIDIÓ LA TABLA, cuando el JSON no lo dice. La del 28-sep-2026 declara
# «3er TCC XXII» en su `fuente` y no trae la lista `organos`; mientras no la
# traiga, sólo el 3TCC habla.
ORGANOS_CALIBRADOS_POR_DEFECTO = ("3TCC",)

# «… del Vigésimo Segundo Circuito» o «… del XXII Circuito», que
# `fase_precedente.circuito_de` no sabe leer en romanos.
_RX_22 = re.compile(r"\bvig[eé]simo\s+segundo\s+circuito\b|\bxxii\s+circuito\b",
                    re.I)


def del_circuito_22(tribunal: str, circuito: str = "", ciudad: str = "") -> bool:
    """¿Consta que el tribunal que redacta es del Vigésimo Segundo Circuito?

    QUE CONSTE, NO QUE NO CONSTE LO CONTRARIO. `fase_espejo.resolver_tribunal`
    acepta el circuito vacío como si fuera el 22, y `circuito_de` devuelve
    vacío con cualquier cosa que no sepa leer: «… del Decimoquinto Circuito»,
    «… del 7o. Circuito», o un nombre sin cláusula de circuito. Así, un Primer
    Tribunal de Mexicali se resolvía al 1TCC de Querétaro y la tarjeta «Su
    propio tribunal» le enseñaba sentencias ajenas con porcentaje encima. El
    taller lo usan secretarios de toda la república; aquí se exige que conste.

    · Un tribunal AUXILIAR nunca es el propio tribunal de estos precedentes:
      resuelve asuntos de otros, con otros nombres.
    · Si el nombre trae «circuito» y no es el 22, no lo es.
    · Si el nombre no dice circuito alguno, sólo la residencia en Querétaro
      —del nombre o de la ciudad de la ficha— lo acredita.
    """
    t = " ".join(str(tribunal or "").split()).lower()
    if not t or "auxiliar" in t:
        return False
    if str(circuito or "").strip() == "22" or _RX_22.search(t):
        return True
    if "circuito" in t:
        return False
    lugar = t + " " + str(ciudad or "").lower()
    return "querétaro" in lugar or "queretaro" in lugar


def organo_de(tribunal: str, circuito: str = "", ciudad: str = "") -> tuple:
    """(clave, nombre OAJ) del tribunal que redacta. (None, '') si no consta."""
    if not del_circuito_22(tribunal, circuito, ciudad):
        return None, ""
    try:
        import fase_espejo as fe
        clave, _ = fe.resolver_tribunal(tribunal, "22")
    except Exception:
        return None, ""
    if not clave or clave not in ORGANOS_OAJ:
        return None, ""
    return clave, ORGANOS_OAJ[clave]


def tipo_oaj(tipo_taller: str) -> str:
    """«amparo_revision» → «Amparo en revisión». Vacío si no es de los cuatro."""
    t = str(tipo_taller or "").strip()
    if t in TIPOS_OAJ:
        return TIPOS_OAJ[t]
    try:
        import tipos_asunto as _ta
        return TIPOS_OAJ.get(_ta.normalizar(t), "")
    except Exception:
        return ""


def texto_consulta(problema) -> str:
    """El texto que se embebe: «pregunta combate resolvio», o vacío.

    Es la forma con que se midió la tabla (`t_plant` del calibrador) y la
    misma con que se indexaron los planteamientos. VACÍO SI FALTA UNA PIEZA:
    la pregunta sola —el problema global, las preguntas sintéticas de
    inoperancia o de sustento, una sesión vieja guardada como texto— tiene
    otra distribución de coseno, y pasarla por esta tabla daría un porcentaje
    «calibrado» que ninguna medición respalda. Esa consulta calla.
    """
    if not isinstance(problema, dict):
        return ""
    partes = [" ".join(str(problema.get(k) or "").split())
              for k in ("pregunta", "combate", "resolvio")]
    if not all(partes):
        return ""
    return " ".join(partes)[:MAX_CONSULTA]


# ═══════════════════════════════════════════════════════════════════════════
# LA CALIBRACIÓN
# ═══════════════════════════════════════════════════════════════════════════
_CACHE = {}
_AVISADOS = set()


def _avisar_una_vez(clave, texto: str):
    """Un callar que se decide por el dato se dice en el log, pero una vez.

    El silencio de esta fuente es a propósito, y por eso mismo es peligroso:
    si la tabla viene con otra forma, o sin corte, la tarjeta vuelve al acervo
    viejo y nada lo distingue de «no hubo coincidencias». Se imprime una vez
    por tabla y motivo, no en cada consulta.
    """
    if clave in _AVISADOS:
        return
    _AVISADOS.add(clave)
    print(f"   ⚠️ precedentes OAJ: {texto}")


def cargar_calibracion(ruta: str = "") -> dict:
    """El JSON de calibración, o {} si no está o no se puede leer.

    Se relee cuando cambia en disco (por su fecha de modificación): recalibrar
    no debe exigir reiniciar el servicio para que la tabla nueva cuente.
    """
    ruta = ruta or RUTA_CALIBRACION
    try:
        mt = os.path.getmtime(ruta)
    except OSError:
        return {}
    k = (ruta, mt)
    if k in _CACHE:
        return _CACHE[k]
    try:
        with open(ruta, encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception as e:
        print(f"   ⚠️ precedentes OAJ: calibración ilegible ({ruta}): {e}")
        d = {}
    if not isinstance(d, dict):
        d = {}
    _CACHE.clear()
    _CACHE[k] = d
    return d


def _numero(v):
    """float finito, o None. Un bool no es un número aunque Python lo crea."""
    if isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def tabla_de(cal: dict, clase: str, tipo: str):
    """[(cos_min, cos_max, prob, n)] ordenada, o None. `n` None si no viene.

    UNA TABLA A MEDIO LEER ES UNA TABLA INVENTADA. Si un solo tramo no trae
    tres o cuatro números, o la probabilidad sale de [0, 1], o el tramo está al
    revés, o `n` no es un entero no negativo, la tabla entera se descarta, esa
    clase calla y el log lo dice: así se detectó que la primera versión de
    este lector descartaba en silencio toda tabla real, que viene con `n`.
    """
    crudo = ((cal or {}).get(clase) or {})
    crudo = crudo.get(tipo) if isinstance(crudo, dict) else None
    if not isinstance(crudo, list) or not crudo:
        return None
    tramos = []
    for x in crudo:
        motivo = ""
        if not (isinstance(x, (list, tuple)) and len(x) in (3, 4)):
            motivo = "un tramo no trae 3 ni 4 columnas"
        else:
            cmin, cmax, p = (_numero(v) for v in x[:3])
            n = _numero(x[3]) if len(x) == 4 else None
            if None in (cmin, cmax, p):
                motivo = "un tramo trae algo que no es número"
            elif not (0.0 <= p <= 1.0) or cmin > cmax:
                motivo = "un tramo trae probabilidad fuera de [0, 1] o está al revés"
            elif len(x) == 4 and (n is None or n < 0 or not n.is_integer()):
                motivo = "un tramo trae un número de pares que no es entero"
        if motivo:
            _avisar_una_vez(("forma", id(cal), clase, tipo),
                            f"tabla {clase}/{tipo} descartada: {motivo}; "
                            f"esta clase calla")
            return None
        tramos.append((cmin, cmax, p, int(n) if n is not None else None))
    tramos.sort(key=lambda t: t[0])
    return tramos


def _sostenidos(tabla):
    """Los tramos con pares suficientes detrás. Sin `n`, no se sabe: fuera."""
    return [t for t in (tabla or []) if t[3] is not None and t[3] >= PARES_MINIMOS]


def probabilidad(tabla, coseno: float):
    """La probabilidad calibrada de ese coseno, o None.

    Es la función escalonada de la regresión isotónica, leída SÓLO sobre los
    tramos sostenidos por `PARES_MINIMOS` pares o más: manda el último de ellos
    cuyo inicio queda por debajo del coseno. Así,

    · un coseno que cae en un hueco entre tramos recibe la probabilidad del
      tramo de ABAJO, que en una tabla monótona es la menor: el hueco se
      resuelve por el lado de callar;
    · un tramo de un solo par —el «1.0» de un único ejemplo— no pone
      porcentaje: se lee el tramo sostenido de debajo;
    · por encima del último tramo sostenido se recorta a él, que es lo que
      hace la isotónica fuera de rango; por debajo del primero no hay dato y no
      se extrapola: None.
    """
    p = None
    for cmin, _cmax, prob, _n in _sostenidos(tabla):
        if coseno >= cmin:
            p = prob
        else:
            break
    return p


def _corte(cal: dict, clase: str, tipo: str, tabla):
    """El coseno a partir del cual la tabla puede dar 0.85. None si nunca.

    Es el inicio del primer tramo SOSTENIDO que llega a 0.85 —la misma regla
    con que el calibrador escribe `umbral_85`—: por debajo de él ningún coseno
    puede pasar, así que se le pide a Qdrant que ni lo mande.

    `umbral_85` del JSON, para esa clase y tipo:
      · ausente → manda la tabla;
      · con número → se toma el MÁS ALTO de los dos: si las dos anotaciones
        discrepan, callar es el fallo barato;
      · presente y `null` → el calibrador dejó escrito que ahí no hay corte
        fiable del 85%. None, y se calla. Ignorarlo y deducir un corte de la
        tabla fue lo que hacía la primera versión, y con la tabla real ponía
        «100% de similitud» en el amparo directo sobre un tramo de un par.
    """
    cortes = [cmin for cmin, _c, p, _n in _sostenidos(tabla) if p >= PROB_MINIMA]
    if not cortes:
        return None
    corte = min(cortes)
    umbrales = ((cal or {}).get("umbral_85") or {})
    umbrales = umbrales.get(clase) if isinstance(umbrales, dict) else None
    if isinstance(umbrales, dict) and tipo in umbrales:
        declarado = _numero(umbrales.get(tipo))
        if declarado is None:
            return None
        corte = max(corte, declarado)
    return corte


def organos_calibrados(cal: dict) -> set:
    """Los nombres OAJ de los tribunales en que se midió la tabla.

    Admite el nombre exacto o la clave del taller («3TCC»). Si el JSON no trae
    la lista, vale sólo `ORGANOS_CALIBRADOS_POR_DEFECTO`.
    """
    crudo = (cal or {}).get("organos")
    if not isinstance(crudo, list):
        crudo = list(ORGANOS_CALIBRADOS_POR_DEFECTO)
    fuera = set()
    for x in crudo:
        x = str(x or "").strip()
        if x:
            fuera.add(ORGANOS_OAJ.get(x, x))
    return fuera


# ═══════════════════════════════════════════════════════════════════════════
# LA BÚSQUEDA
# ═══════════════════════════════════════════════════════════════════════════
async def _esperar(r):
    return await r if inspect.isawaitable(r) else r


async def _buscar(qdrant, vector, clase, tipo, organo, limite, corte) -> list:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    # TIPO Y ÓRGANO VAN EN EL FILTRO, no se comprueban después. Un precedente
    # de otro tribunal en la tarjeta «Su propio tribunal» es exactamente el
    # error que esta tarjeta no puede cometer, y una queja junto a un directo
    # no es el mismo punto aunque la pregunta se parezca.
    debe = [FieldCondition(key="clase", match=MatchValue(value=clase)),
            FieldCondition(key="tipo", match=MatchValue(value=tipo)),
            FieldCondition(key="organo", match=MatchValue(value=organo))]
    r = await _esperar(qdrant.query_points(
        collection_name=COLECCION, query=vector, using=VECTOR,
        query_filter=Filter(must=debe), limit=limite,
        score_threshold=corte, with_payload=True))
    return list(getattr(r, "points", None) or [])


def _neun(pl: dict):
    """El NEUN como entero, o None.

    Acepta 12345, «12345», 12345.0 y «12345.0»: una columna de pandas con algún
    NaN guarda los enteros como float, y rechazar «12345.0» apagaría la fuente
    entera sin una sola excepción en el log. No acepta bool ni un float con
    decimales, que no son un número de expediente.
    """
    v = (pl or {}).get("neun")
    if isinstance(v, bool):
        return None
    x = _numero(str(v).strip()) if v is not None else None
    if x is None or not x.is_integer():
        return None
    n = int(x)
    return n if n > 0 else None


def _txt(pl: dict, k: str) -> str:
    v = str((pl or {}).get(k) or "").strip()
    # La cadena "null" pasa cualquier comprobación de verdad y se imprimiría
    # tal cual; ya pasó con las fechas del acervo viejo.
    return "" if v.lower() in ("null", "none") else v


def _filas(puntos, tabla, corte, fuente: str) -> list:
    """Lo mejor de cada NEUN, calibrado, filtrado y en el formato de la tarjeta."""
    # UN PLANTEAMIENTO NO ES UNA SENTENCIA. Tres planteamientos del mismo asunto
    # entrarían como tres precedentes y son uno: se queda el de mejor coseno.
    mejor = {}
    sin_neun = 0
    for p in puntos:
        pl = getattr(p, "payload", None) or {}
        n = _neun(pl)
        if n is None:
            # Sin NEUN no hay con qué buscarla en la OAJ: una cita que nadie
            # puede comprobar no se enseña.
            sin_neun += 1
            continue
        sc = float(getattr(p, "score", 0.0) or 0.0)
        if n not in mejor or sc > mejor[n][0]:
            mejor[n] = (sc, pl)
    if puntos and not mejor:
        # Qdrant devolvió puntos y NINGUNO se puede citar: eso no es «no hay
        # precedentes», es un índice con el NEUN mal guardado.
        _avisar_una_vez(("neun", fuente),
                        f"{sin_neun} puntos ({fuente}) sin NEUN válido; "
                        f"la fuente calla")

    filas = []
    for n, (sc, pl) in sorted(mejor.items(), key=lambda kv: -kv[1][0]):
        # EL CORTE SE COMPRUEBA AQUÍ TAMBIÉN, no sólo en Qdrant: el
        # `score_threshold` es una petición al servidor, y el porcentaje es
        # una afirmación nuestra.
        if corte is None or sc < corte:
            continue
        prob = probabilidad(tabla, sc)
        if prob is None or prob < PROB_MINIMA:
            continue
        es_pl = fuente == "planteamiento"
        filas.append({
            "tipo_asunto": _txt(pl, "tipo"),
            "expediente": _txt(pl, "alias"),
            "fecha": _txt(pl, "fecha"),
            "sentido": _txt(pl, "sentido"),
            "tema": _txt(pl, "tema"),
            "score": round(sc, 3),
            # La OAJ no da PDF por enlace directo; el campo se conserva vacío
            # para que la fila tenga la misma forma que las del espejo viejo.
            "pdf_url": "",
            # Hacia abajo: 0.869 es «86%», nunca «87%». El porcentaje ya es
            # una afirmación; no se redondea a favor. Y con tope: ver arriba.
            "similitud": min(int(prob * 100 + 1e-9), TOPE_VISIBLE),
            "fuente": fuente,
            "pregunta": _txt(pl, "pregunta") if es_pl else "",
            "razon": _txt(pl, "razon") if es_pl else "",
            "calificacion": _txt(pl, "calificacion") if es_pl else "",
            "autoridad": _txt(pl, "autoridad") if es_pl else "",
            "neun": n,
            "enlace_oaj": ENLACE_OAJ,
        })
        if len(filas) >= MOSTRADAS:
            break
    return filas


async def precedentes_oaj(qdrant, embed, problema, tipo_taller: str,
                          organo_oaj: str) -> list:
    """Los precedentes del propio tribunal para UN planteamiento. Hasta seis.

    `problema` es el dict de la fase 3 (pregunta, combate, resolvio); cualquier
    otra cosa —una cadena, un dict sin alguna de las tres— calla, porque la
    tabla sólo vale para la forma con que se midió (ver `texto_consulta`).
    `tipo_taller` es la clave del taller (amparo_directo, amparo_revision,
    queja, revision_fiscal) y `organo_oaj` el nombre exacto con el que la OAJ
    indexa al tribunal (ver `organo_de`).

    Lista vacía en cualquier duda: sin tabla, sin tipo, sin órgano, órgano no
    calibrado, sin nada que pase del 85%, o con cualquier error. El taller cae
    entonces al espejo viejo; callar aquí es el fallo barato y nunca tumba la
    sentencia.
    """
    try:
        consulta = texto_consulta(problema)
        tipo = tipo_oaj(tipo_taller)
        organo = str(organo_oaj or "").strip()
        if not (qdrant and embed and consulta and tipo and organo):
            return []

        cal = cargar_calibracion()
        if not cal:
            return []
        if organo not in organos_calibrados(cal):
            _avisar_una_vez(("organo", id(cal), organo),
                            f"la tabla no se midió en «{organo[:70]}…»; "
                            f"ahí la fuente OAJ calla")
            return []
        t_pl = tabla_de(cal, "planteamiento", tipo)
        t_as = tabla_de(cal, "asunto", tipo)
        corte_pl = _corte(cal, "planteamiento", tipo, t_pl)
        corte_as = _corte(cal, "asunto", tipo, t_as)
        # Sin corte para ninguna de las dos clases no se embebe siquiera: el
        # porcentaje no se puede decir y no se gasta la llamada.
        if corte_pl is None and corte_as is None:
            _avisar_una_vez(("sin_corte", id(cal), tipo),
                            f"sin corte fiable al 85% para {tipo}; "
                            f"la fuente OAJ calla en ese tipo")
            return []

        try:
            vector = await embed(consulta)
        except Exception as e:
            print(f"   ⚠️ precedentes OAJ: no se pudo embeber el problema: {e}")
            return []

        if corte_pl is not None:
            puntos = await _buscar(qdrant, vector, "planteamiento", tipo,
                                   organo, PEDIDOS_PLANTEAMIENTO, corte_pl)
            filas = _filas(puntos, t_pl, corte_pl, "planteamiento")
            if filas:
                return filas

        # EL RESPALDO. Mismo vector: el de la consulta del planteamiento contra
        # el vector de «tipo. tema» del asunto. Así se midió la tabla de
        # `asunto` (pares_reales: planteamiento real del taller contra el tema
        # de la OAJ); con otra forma de consulta su porcentaje no significaría
        # lo que dice.
        if corte_as is not None:
            puntos = await _buscar(qdrant, vector, "asunto", tipo,
                                   organo, PEDIDOS_ASUNTO, corte_as)
            return _filas(puntos, t_as, corte_as, "tema")
        return []
    except Exception as e:
        print(f"   ⚠️ precedentes OAJ: {e}")
        return []
