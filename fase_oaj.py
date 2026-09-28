# -*- coding: utf-8 -*-
"""EL ESPEJO, DESDE EL ÍNDICE DE LA OAJ — los precedentes del propio tribunal,
planteamiento por planteamiento.

La tarjeta «Su propio tribunal» nació sobre el acervo viejo (`fase_espejo.py`,
léase entero antes de tocar esto): 3,585 sentencias de UN tribunal, con 2025
casi vacío, un umbral de coseno a pelo y una cobertura medida del 9%. Esta es
la fuente nueva: `oaj_precedentes`, el índice de las sentencias PÚBLICAS de los
tribunales del circuito en los cuatro tipos que el taller proyecta, y con los
planteamientos de cada sentencia desmontados uno a uno —pregunta, qué se
combatía, qué se resolvió, cómo se calificó y por qué—.

LO QUE CAMBIA RESPECTO DEL ESPEJO VIEJO, Y POR QUÉ
=================================================
 1. SE BUSCA PLANTEAMIENTO CONTRA PLANTEAMIENTO. El espejo viejo buscaba la
    pregunta del caso contra trozos de estudio de fondo, y entre las dos formas
    había una décima de coseno que era todo el margen. Aquí el punto indexado
    tiene la MISMA forma que el problema de la fase 3, así que la consulta es
    «pregunta combate resolvio» y no la pregunta sola: así se midió que casa
    mejor contra los planteamientos.

 2. EL PORCENTAJE NO ES EL COSENO. Un coseno de 0.62 no significa «62%» de
    nada, y cada tipo de asunto tiene su propia distribución. El coseno se
    convierte con una tabla de calibración isotónica (`oaj_calibracion.json`,
    versionada junto a este módulo) y sólo se enseña lo que la tabla pone en
    0.85 o más. SIN TABLA NO HAY PORCENTAJE: si el JSON no existe, o no trae el
    tipo, se devuelve una lista vacía y el taller cae al espejo viejo. Inventar
    un porcentaje sería de la misma especie que la tesis inventada.

 3. SIN PISO DE FILAS. El espejo viejo exigía tres sentencias porque su coseno
    crudo no distinguía el punto del vecino; una coincidencia calibrada al 85%
    vale sola.

 4. EL RESPALDO POR TEMA. Si ningún planteamiento pasa, se prueba con los
    puntos `clase=asunto`, cuyo vector es el del TEMA de la sentencia, con su
    propia tabla. La fila lo dice (`fuente: "tema"`): no es lo mismo coincidir
    en el punto que coincidir en la materia del asunto.

LO QUE SIGUE SIN SER, Y POR LAS MISMAS RAZONES
==============================================
Tampoco aquí se cuenta, se acusa ni se traduce el sentido a cubos. Las tres
mediciones que mataron al «contradictor» están en `fase_espejo.py` y valen
igual para este índice: el artículo 224 de la Ley de Amparo exige unanimidad y
el índice no trae votación; la calificación del planteamiento y el sentido del
resolutivo no son la misma escala. Se enseñan los precedentes, su calificación
literal y su razón, y compara el secretario.

LA FORMA DEL JSON DE CALIBRACIÓN
================================
    {
      "planteamiento": {"Amparo Directo": [[cos_min, cos_max, prob], ...], ...},
      "asunto":        {"Amparo Directo": [[cos_min, cos_max, prob], ...], ...},
      "umbral_85":     {"planteamiento": {"Amparo Directo": 0.71, ...},
                        "asunto": {...}}
    }

Las claves de tipo son las de la OAJ, con su grafía exacta. El corte que manda
es el de la tabla; `umbral_85`, si viene, sólo puede ENDURECERLO (ver `_corte`).

NO LO LLENA ESTE MÓDULO. La colección y la tabla las produce otro proceso; aquí
sólo se leen. Mientras la tabla no exista, este módulo calla siempre y el
taller sigue exactamente como estaba.
"""

import inspect
import json
import os
import re
import unicodedata

COLECCION = "oaj_precedentes"
VECTOR = "dense"            # text-embedding-3-small, 1536, coseno: embed_leyes

# ── LO QUE SE ENSEÑA ─────────────────────────────────────────────────────────
PROB_MINIMA = 0.85
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

NOTA_COBERTURA = ("Índice de la OAJ: todas las sentencias públicas del "
                  "tribunal de los cuatro tipos.")

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


def organo_de(tribunal: str, circuito: str = "") -> tuple:
    """(clave, nombre OAJ) del tribunal que redacta. (None, '') si no se sabe."""
    try:
        import fase_espejo as fe
        clave, _ = fe.resolver_tribunal(tribunal, circuito)
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
    """El texto que se embebe: «pregunta combate resolvio», en ese orden.

    Así se midió que casa mejor contra los planteamientos indexados, que tienen
    esa misma forma. Si llega sólo la pregunta —un problema viejo guardado como
    texto—, se busca con ella.
    """
    if isinstance(problema, dict):
        partes = [str(problema.get(k) or "").strip()
                  for k in ("pregunta", "combate", "resolvio")]
        t = " ".join(p for p in partes if p)
    else:
        t = str(problema or "").strip()
    return " ".join(t.split())[:MAX_CONSULTA]


# ═══════════════════════════════════════════════════════════════════════════
# LA CALIBRACIÓN
# ═══════════════════════════════════════════════════════════════════════════
_CACHE = {}


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


def tabla_de(cal: dict, clase: str, tipo: str):
    """[(cos_min, cos_max, prob)] ordenada, o None.

    UNA TABLA A MEDIO LEER ES UNA TABLA INVENTADA. Si un solo tramo no trae tres
    números, o la probabilidad sale de [0, 1], o el tramo está al revés, la
    tabla entera se descarta y esa clase calla.
    """
    crudo = ((cal or {}).get(clase) or {})
    crudo = crudo.get(tipo) if isinstance(crudo, dict) else None
    if not isinstance(crudo, list) or not crudo:
        return None
    tramos = []
    for x in crudo:
        if not (isinstance(x, (list, tuple)) and len(x) == 3):
            return None
        try:
            cmin, cmax, p = (float(v) for v in x)
        except (TypeError, ValueError):
            return None
        if not (0.0 <= p <= 1.0) or cmin > cmax:
            return None
        tramos.append((cmin, cmax, p))
    tramos.sort(key=lambda t: t[0])
    return tramos


def probabilidad(tabla, coseno: float):
    """La probabilidad calibrada de ese coseno, o None.

    Es la función escalonada de la regresión isotónica: manda el último tramo
    cuyo inicio queda por debajo del coseno. Así, un coseno que cae en un hueco
    entre dos tramos recibe la probabilidad del tramo de ABAJO, que en una
    tabla monótona es la menor: el hueco se resuelve por el lado de callar.

    Por encima del último tramo se recorta al último, que es lo que hace la
    isotónica fuera de rango. Por debajo del primero no hay dato y no se
    extrapola: None.
    """
    if not tabla:
        return None
    p = None
    for cmin, _cmax, prob in tabla:
        if coseno >= cmin:
            p = prob
        else:
            break
    return p


def _corte(cal: dict, clase: str, tipo: str, tabla):
    """El coseno a partir del cual la tabla puede dar 0.85. None si nunca.

    Es el inicio del primer tramo que llega a 0.85: por debajo de él ningún
    coseno puede pasar, así que se le pide a Qdrant que ni lo mande. Si el JSON
    trae además `umbral_85` para esa clase y tipo, se toma el MÁS ALTO de los
    dos: si las dos anotaciones discrepan, callar es el fallo barato.
    """
    cortes = [cmin for cmin, _c, p in (tabla or []) if p >= PROB_MINIMA]
    if not cortes:
        return None
    corte = min(cortes)
    try:
        declarado = float((((cal or {}).get("umbral_85") or {})
                           .get(clase) or {}).get(tipo))
        corte = max(corte, declarado)
    except (AttributeError, TypeError, ValueError):
        pass
    return corte


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
    try:
        n = int(str((pl or {}).get("neun")).strip())
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _txt(pl: dict, k: str) -> str:
    v = str((pl or {}).get(k) or "").strip()
    # La cadena "null" pasa cualquier comprobación de verdad y se imprimiría
    # tal cual; ya pasó con las fechas del acervo viejo.
    return "" if v.lower() in ("null", "none") else v


def _filas(puntos, tabla, fuente: str) -> list:
    """Lo mejor de cada NEUN, calibrado, filtrado y en el formato de la tarjeta."""
    # UN PLANTEAMIENTO NO ES UNA SENTENCIA. Tres planteamientos del mismo asunto
    # entrarían como tres precedentes y son uno: se queda el de mejor coseno.
    mejor = {}
    for p in puntos:
        pl = getattr(p, "payload", None) or {}
        n = _neun(pl)
        if n is None:
            # Sin NEUN no hay con qué buscarla en la OAJ: una cita que nadie
            # puede comprobar no se enseña.
            continue
        sc = float(getattr(p, "score", 0.0) or 0.0)
        if n not in mejor or sc > mejor[n][0]:
            mejor[n] = (sc, pl)

    filas = []
    for n, (sc, pl) in sorted(mejor.items(), key=lambda kv: -kv[1][0]):
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
            # una afirmación; no se redondea a favor.
            "similitud": int(prob * 100 + 1e-9),
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

    `problema` es el dict de la fase 3 (pregunta, combate, resolvio) o, en
    sesiones viejas, la pregunta sola. `tipo_taller` es la clave del taller
    (amparo_directo, amparo_revision, queja, revision_fiscal) y `organo_oaj` el
    nombre exacto con el que la OAJ indexa al tribunal (ver `organo_de`).

    Lista vacía en cualquier duda: sin tabla, sin tipo, sin órgano, sin nada
    que pase del 85%, o con cualquier error. El taller cae entonces al espejo
    viejo; callar aquí es el fallo barato y nunca tumba la sentencia.
    """
    try:
        consulta = texto_consulta(problema)
        tipo = tipo_oaj(tipo_taller)
        organo = str(organo_oaj or "").strip()
        if not (qdrant and embed and consulta and tipo and organo):
            return []

        cal = cargar_calibracion()
        t_pl = tabla_de(cal, "planteamiento", tipo)
        t_as = tabla_de(cal, "asunto", tipo)
        corte_pl = _corte(cal, "planteamiento", tipo, t_pl)
        corte_as = _corte(cal, "asunto", tipo, t_as)
        # Sin tabla para ninguna de las dos clases no se embebe siquiera: el
        # porcentaje no se puede decir y no se gasta la llamada.
        if corte_pl is None and corte_as is None:
            return []

        try:
            vector = await embed(consulta)
        except Exception as e:
            print(f"   ⚠️ precedentes OAJ: no se pudo embeber el problema: {e}")
            return []

        if corte_pl is not None:
            puntos = await _buscar(qdrant, vector, "planteamiento", tipo,
                                   organo, PEDIDOS_PLANTEAMIENTO, corte_pl)
            filas = _filas(puntos, t_pl, "planteamiento")
            if filas:
                return filas

        # EL RESPALDO. Mismo vector: el de la consulta del planteamiento contra
        # el vector del tema del asunto. La tabla de `asunto` tiene que haberse
        # medido con esa misma forma de consulta, o su porcentaje no significa
        # lo que dice.
        if corte_as is not None:
            puntos = await _buscar(qdrant, vector, "asunto", tipo,
                                   organo, PEDIDOS_ASUNTO, corte_as)
            return _filas(puntos, t_as, "tema")
        return []
    except Exception as e:
        print(f"   ⚠️ precedentes OAJ: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════
# EL RENGLÓN DE RESUMEN
# ═══════════════════════════════════════════════════════════════════════════
def _etiqueta(tema: str) -> str:
    """«Improcedencia del juicio» → «improcedencia_del_juicio»."""
    x = unicodedata.normalize("NFKD", str(tema or "").strip().lower())
    x = "".join(c for c in x if not unicodedata.combining(c))
    return re.sub(r"[\s\-]+", "_", x)


def resumen(filas: list) -> str:
    """El mismo renglón que el espejo viejo, con dos guardas más.

    · El `tema` de la OAJ es prosa y no etiqueta: «Inoperancia de los
      conceptos…» con mayúscula no casaría con la guarda de las etiquetas que
      ya contienen el resultado (`fase_espejo._TAUTOLOGICAS`). Se pasa a la
      forma de etiqueta antes de mirarla.
    · El `sentido` puede venir vacío en el índice. «1 niega» sobre tres filas
      se leería como la cuenta de las tres; si falta en alguna, no se resume.
    """
    if not filas or any(not str(f.get("sentido") or "").strip() for f in filas):
        return ""
    try:
        import fase_espejo as fe
        return fe.resumen([dict(f, tema=_etiqueta(f.get("tema")),
                                tipo_asunto=str(f.get("tipo_asunto") or ""))
                           for f in filas])
    except Exception:
        return ""
