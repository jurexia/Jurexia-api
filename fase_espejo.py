# -*- coding: utf-8 -*-
"""EL ESPEJO DEL PROPIO TRIBUNAL — seis sentencias suyas que puede abrir.

David preguntó qué implementar para superar a un secretario. La respuesta
medida: no se le supera en criterio, se le supera en MEMORIA. Un secretario
recuerda lo que él mismo trabajó; el acervo tiene 3,585 sentencias del Tercer
Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo
Circuito, de 2019 a 2026, incluidas las de las ponencias que no son la suya y
las anteriores a su llegada.

LO QUE ESTO NO ES, Y POR QUÉ.
=============================
La primera versión era un CONTRADICTOR: «su tribunal ya resolvió esto siete
veces al revés». Tres mediciones la mataron, y las tres se dejan escritas aquí
para que nadie la resucite sin volver a medir.

 1. JURÍDICAMENTE IBA LARGA. El artículo 217, párrafo tercero, de la Ley de
    Amparo EXCLUYE a los tribunales colegiados de la obligatoriedad de la
    jurisprudencia de colegiados; la autovinculación viene del 228, párrafo
    segundo. Y para que un criterio reiterado sea jurisprudencia, el artículo
    224 exige UNANIMIDAD. No hay campo de votación en ningún punto del acervo
    —recorrido el inventario completo de claves de los 6,379 holdings del
    3TCC—, así que el presupuesto legal es inverificable SIEMPRE. Una tarjeta
    que le anuncie a un magistrado el deber de interrumpir jurisprudencia es de
    la misma especie que la tesis inventada.

 2. EL DATO NO SOSTIENE EL CONTEO. `tema_juridico` no es una taxonomía: 5,024
    etiquetas distintas para 6,379 holdings del 3TCC, y sólo 194 temas (3.9%)
    llegan a tres expedientes. Contar por igualdad de etiqueta calla en nueve
    de cada diez asuntos; contar sobre el top-K de un vector cuenta vecinos de
    coseno y no sentencias del punto.

 3. EL AVISO MÁS APARATOSO SALDRÍA AL REVÉS. El grupo mayor del 3TCC
    —inconstitucionalidad_sistema_normativo_tributario, 44 holdings, 44 de 44
    «confirma»— tiene 20 casos (45%) en que el juez de distrito había CONCEDIDO
    el amparo: el tribunal confirmó una concesión. Traducir «confirma» a «niega»
    —que es lo que hace `_CUBO` en fase_precedente.py— habría producido
    exactamente la acusación contraria a la verdad.

Así que aquí no se cuenta, no se acusa y no se traduce el sentido a cubos. Se
enseñan seis sentencias propias con su fecha, su sentido LITERAL y su enlace, y
compara el secretario. Es menos de lo que se prometió y es lo que el dato
aguanta.

EL UMBRAL, MEDIDO
=================
Banco de 24 preguntas redactadas como las escribiría un secretario —no con el
texto de la etiqueta, que es hacer trampa—: trece sobre puntos que el 3TCC sí
ha trabajado y once sobre puntos que no.

    umbral 0.73 · piso de 3 filas  →  habla en 13/13 · ruido 0/11

El margen es el número que hay que vigilar: la positiva más baja marca 0.766 y
el control más alto 0.699. El umbral queda a 0.036 por debajo de la primera y a
0.031 por encima del segundo. `test_espejo_propio.py` falla si esa distancia
baja de 0.03 al reingerir el acervo.

Una de las «controles» resultó no serlo: la guarda y custodia compartida marcó
0.771 y la comprobación enseñó que el 3TCC tiene 26 temas de custodia y 166
sobre menores. Es materia suya y la recuperación acertó; el error era de quien
armó el banco.
"""

import inspect
import re

COL_HOLDINGS = "sentencias_holdings"


def col_estudios(circuito) -> str:
    """El acervo TROCEADO del circuito. Ver la nota del umbral."""
    return "sentencias_ef_c%s" % str(circuito or "22").strip()


# ── EL UMBRAL Y EL PISO, Y CÓMO SE LLEGÓ A ELLOS ─────────────────────────────
#
# LA PRIMERA CALIBRACIÓN ESTABA MAL Y ASÍ SE DESCUBRIÓ. Se midió con un banco de
# 24 preguntas escritas a mano: 13/13 y 0/11 contra `sentencias_holdings` con
# umbral 0.73. Parecía limpio. Luego se probó contra los planteamientos REALES
# que la fase 3 produce —23 de seis asuntos del propio secretario— y el espejo
# habló en CERO de 23.
#
# El motivo: el banco se escribió a partir de los temas mayores del acervo, o
# sea preguntando lo que el corpus sabía contestar, y en lenguaje de doctrina
# («¿son inoperantes los conceptos que reiteran los agravios?»). El pipeline
# entrega preguntas del CASO: «¿La Sala examinó las excepciones conforme a los
# términos en que fueron planteadas?». Entre las dos formas hay una décima de
# coseno, y la décima era todo el margen.
#
# Se probaron dos rescates y los dos se midieron antes de creerlos:
#   · reformular a rubro con `fase6_rag.consulta_conceptual` → BAJA la
#     puntuación (0.698 → 0.656). Ese truco sirve para el vector `rubro` de la
#     jurisprudencia, no para holdings escritos en prosa de sentencia.
#   · reformular a holding declarativo → sube (mediana 0.634 → 0.677) pero
#     sigue sin llegar: 0 de 23 alcanzan 0.73.
#
# LO QUE SÍ FUNCIONÓ fue cambiar de acervo. `sentencias_ef_c22` es el estudio de
# fondo TROCEADO —75,846 puntos, 12,037 del 3TCC— frente a 6,379 holdings: doce
# veces más texto y a nivel de argumento, que es como está escrita la pregunta.
# Medido sobre los mismos 23 planteamientos reales y 5 controles:
#
#     reales   min 0.561 · mediana 0.664 · max 0.738
#     controles min 0.439 · mediana 0.583 · max 0.648
#     con umbral 0.70 → habla en 6 de 23 · ruido 0 de 5
#
# SEIS DE VEINTITRÉS ES LA COBERTURA REAL, y se dice en voz alta: el espejo
# habla en uno de cada cuatro planteamientos. Las distribuciones se SOLAPAN
# —el peor real (0.561) está por debajo del mejor control (0.648)—, así que no
# existe umbral que hable en todos sin mentir en alguno. Ante esa disyuntiva se
# elige callar, porque el fallo de callar es que el secretario no vea una
# sentencia suya, y el de hablar es meterle en pantalla expedientes que no van
# del punto.
UMBRAL = 0.70
PISO_FILAS = 3          # con menos de tres, la tarjeta no se enseña
PEDIDAS = 24            # se piden veinticuatro trozos y se quedan seis sentencias
MOSTRADAS = 6

# ── LOS TRIBUNALES DEL CIRCUITO 22 ───────────────────────────────────────────
# `tribunal_completo` está vacío en 6,379 de 6,379 holdings del 3TCC, así que
# el nombre largo no se lee del acervo: se resuelve aquí. Los nombres son los
# mismos que main.py ya usa en `_CIRCUIT_TRIBUNALES`.
#
# FUERA DEL CIRCUITO 22 NO HAY MAPA Y EL ESPEJO CALLA. Las claves de otros
# circuitos llevan sufijo de materia —3TCC_ADM, 3TCC_CIV, 3TCC_LAB— y «3TCC» a
# secas sólo existe en el 22: adivinar la correspondencia sería imprimir el
# nombre de un tribunal que no es. Ampliarlo exige comprobar cada circuito con
# un facet, no deducirlo.
TRIBUNALES_22 = {
    "1TCC": "Primer Tribunal Colegiado en Materias Administrativa y Civil "
            "del Vigésimo Segundo Circuito",
    "2TCC": "Segundo Tribunal Colegiado en Materias Administrativa y Civil "
            "del Vigésimo Segundo Circuito",
    "3TCC": "Tercer Tribunal Colegiado en Materias Administrativa y Civil "
            "del Vigésimo Segundo Circuito",
    "TCC_PENAL": "Tribunal Colegiado en Materias Penal y Administrativa "
                 "del Vigésimo Segundo Circuito",
    "TCC_ADM": "Tribunal Colegiado en Materias Administrativa y de Trabajo "
               "del Vigésimo Segundo Circuito",
}

_ORD_TRIB = (
    (r"\bprimer", "1TCC"), (r"\bsegundo", "2TCC"), (r"\btercer", "3TCC"),
)

# «… del Vigésimo Segundo Circuito». Todo lo que va de aquí en adelante nombra
# al CIRCUITO y no al tribunal, y sus ordinales no cuentan.
_RX_CLAUSULA_CIRCUITO = re.compile(
    r"\bdel\s+(?:[a-záéíóúñ]+\s+){1,3}?circuito")


def resolver_tribunal(nombre: str, circuito: str = "") -> tuple:
    """(clave, nombre_largo) del tribunal que redacta. (None, '') si no se sabe.

    Se lee del nombre que el secretario escribió en la ficha. Si no es del
    circuito 22, se devuelve None y el espejo no se enseña: ver arriba.
    """
    t = str(nombre or "")
    if not t.strip():
        return None, ""
    if str(circuito or "").strip() not in ("", "22"):
        return None, ""
    plano = re.sub(r"\s+", " ", t).lower()
    # El de materias PENAL y el de TRABAJO no llevan ordinal: son únicos.
    if re.search(r"penal", plano):
        return "TCC_PENAL", TRIBUNALES_22["TCC_PENAL"]
    if re.search(r"trabajo|laboral", plano):
        return "TCC_ADM", TRIBUNALES_22["TCC_ADM"]
    # EL ORDINAL DEL TRIBUNAL, NO EL DEL CIRCUITO. Cazado en la prueba de humo:
    # «Tercer Tribunal Colegiado … del VIGÉSIMO SEGUNDO Circuito» devolvía 2TCC,
    # porque el «segundo» del circuito casaba antes que el «tercer» del
    # tribunal. Habría impreso el nombre de un tribunal ajeno junto a seis
    # expedientes ajenos, que es justo lo que esta tarjeta no puede hacer.
    # Se corta la frase del circuito antes de buscar el ordinal.
    cabeza = _RX_CLAUSULA_CIRCUITO.split(plano)[0]
    for patron, clave in _ORD_TRIB:
        if re.search(patron, cabeza):
            return clave, TRIBUNALES_22[clave]
    return None, ""


# ── LA COBERTURA, DICHA EN VOZ ALTA ──────────────────────────────────────────
# El acervo propio tiene un agujero en 2025: 185 sentencias del 3TCC frente a
# 1,183 de 2024. Si el tribunal cambió de criterio ese año, el espejo enseña el
# viejo. Se escribe DENTRO de la tarjeta, no en una nota al pie, y por eso
# tampoco se ordena ni se presenta nada como «lo último que dijo el tribunal».
NOTA_COBERTURA = (
    "El acervo propio va de 2019 a 2026 y 2025 está incompleto: 185 sentencias "
    "frente a 1,183 de 2024. Si el tribunal cambió de criterio ese año, aquí no "
    "aparece."
)

# ── LAS ETIQUETAS QUE YA CONTIENEN EL RESULTADO ──────────────────────────────
# `tema_juridico` se asigna DESPUÉS de resolver, así que hay familias enteras
# —inoperancia_, improcedencia_, desechamiento_— en las que el sentido de las
# sentencias recuperadas es tautológico: todas niegan porque la etiqueta ES la
# negativa. Las filas se enseñan igual (son sentencias suyas del punto y sirven
# para leerlas), pero el renglón que resume los sentidos se calla: ahí no mide
# criterio, mide la definición de la categoría.
_TAUTOLOGICAS = re.compile(
    r"^(?:inoperanc|improcedenc|procedenc|desechamiento|desistimiento|"
    r"sobreseimiento|caducidad|extemporane)|_sin_materia|sin_materia")

# ── EL VOCABULARIO DEL SENTIDO ───────────────────────────────────────────────
# Crudo del 3TCC: 'confirma' 2,414 pero también 'confirmar' 7, 'confirmar
# sobreseimiento' 2, 'confirmar_sobreseimiento' 1; 'sin_materia' 18 contra 'sin
# materia' 16. Se normaliza SÓLO para contar en el renglón de resumen; en cada
# fila se imprime la palabra que el acervo guardó.
def _norma(s: str) -> str:
    x = re.sub(r"[\s\-]+", "_", str(s or "").strip().lower())
    if x.startswith("confirmar") or x.startswith("confirma"):
        return "confirma"
    if x.startswith("revoca"):
        return "revoca"
    if x.startswith("modifica"):
        return "modifica"
    if x.startswith("sobresee") or x.startswith("sobresei"):
        return "sobresee"
    if x.startswith("desecha"):
        return "desecha"
    if x.startswith("sin_materia"):
        return "sin_materia"
    if x.startswith("declina"):
        return "declina_competencia"
    if "ampar" in x and "conced" in x:
        return "concede"
    return x


_RX_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-"
                      r"[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


async def _esperar(r):
    return await r if inspect.isawaitable(r) else r


async def espejo(qdrant, embed, problema: str, tribunal_key: str,
                 circuito: str = "22") -> list:
    """Las sentencias del PROPIO tribunal más cercanas a este punto.

    Devuelve hasta seis filas {tipo_asunto, expediente, fecha, sentido, tema,
    score, pdf_url}. Lista vacía si no hay al menos `PISO_FILAS`: un espejo que
    siempre opina no es un espejo, y aquí callar es el fallo barato.

    DOS PASOS, Y CADA UNO CONTRA LA COLECCIÓN QUE LE TOCA:
      1. se BUSCA en el estudio de fondo troceado del circuito, que es donde el
         texto está a nivel de argumento y se parece a como está escrita la
         pregunta (ver la nota del umbral);
      2. se CITA desde `sentencias_holdings`, que es donde viven el tipo de
         asunto y la fecha. Sin esos dos la cita no se puede comprobar: «44/2021»
         a secas son CUATRO sentencias distintas del 3TCC —Amparo Directo,
         Queja, Revisión Fiscal y Amparo en Revisión— con cuatro fechas y cuatro
         PDF.

    SIN FILTRO DE MATERIA, a propósito. El tribunal ya es un filtro más estrecho,
    y `GRAFIAS` de fase_precedente.py no contempla los 937 holdings 'mercantil'
    ni los 150 'fiscal' del 3TCC: filtrar por materia haría que el espejo dijera
    «su tribunal nunca ha visto esto» sobre un pagaré teniendo 937 asuntos
    mercantiles propios.
    """
    if not (qdrant and tribunal_key and (problema or "").strip()):
        return []
    try:
        vector = await embed(problema)
    except Exception as e:
        print(f"   ⚠️ espejo del tribunal: no se pudo embeber el problema: {e}")
        return []

    from qdrant_client.models import FieldCondition, Filter, MatchValue

    # SÓLO POR TRIBUNAL. La clave «3TCC» a secas existe únicamente en el
    # circuito 22 —en los demás llevan sufijo de materia, 3TCC_ADM, 3TCC_CIV—,
    # así que el filtro de circuito sobra aquí; y en esta colección `circuito`
    # viaja unas veces como número y otras como texto, que es un 400 esperando.
    debe = [FieldCondition(key="tribunal", match=MatchValue(value=tribunal_key))]
    try:
        r = await _esperar(qdrant.query_points(
            collection_name=col_estudios(circuito), query=vector, using="dense",
            query_filter=Filter(must=debe), limit=PEDIDAS,
            score_threshold=UMBRAL, with_payload=True))
        puntos = getattr(r, "points", None) or []
    except Exception as e:
        print(f"   ⚠️ espejo del tribunal: {e}")
        return []
    if not puntos:
        return []

    # UN TROZO NO ES UNA SENTENCIA. El estudio va troceado y varios trozos del
    # mismo asunto entran juntos; se queda el mejor de cada sentencia.
    mejor = {}
    for p in puntos:
        pl = p.payload or {}
        hid = str(pl.get("holding_id") or "")
        exp = str(pl.get("expediente") or "")
        clave = hid or exp
        if not clave:
            continue
        sc = float(getattr(p, "score", 0.0) or 0.0)
        if clave not in mejor or sc > mejor[clave][0]:
            mejor[clave] = (sc, pl)
    orden = sorted(mejor.items(), key=lambda kv: -kv[1][0])[:MOSTRADAS]
    if len(orden) < PISO_FILAS:
        return []

    # EL TIPO DE ASUNTO Y LA FECHA, DE DONDE ESTÁN. El acervo troceado no los
    # trae; el holding sí. Si la lectura falla, la fila se queda sin ellos y se
    # imprime lo que se sabe: nunca se inventan.
    ids = [k for k, _ in orden if _RX_UUID.match(k or "")]
    datos = {}
    if ids:
        try:
            recs = await _esperar(qdrant.retrieve(
                collection_name=COL_HOLDINGS, ids=ids, with_payload=True))
            for rec in (recs or []):
                datos[str(rec.id)] = rec.payload or {}
        except Exception as e:
            print(f"   ⚠️ espejo: no se pudo leer la cita del holding: {e}")

    filas = []
    for clave, (sc, pl) in orden:
        h = datos.get(clave, {})
        fecha = str(h.get("fecha_sentencia") or "")
        # 5 holdings del 3TCC traen la CADENA "null", que pasa cualquier
        # comprobación de verdad/falsedad y se imprimiría tal cual.
        if fecha.strip().lower() in ("null", "none"):
            fecha = ""
        filas.append({
            "tipo_asunto": str(h.get("tipo_asunto") or "").strip(),
            "expediente": str(pl.get("expediente") or h.get("expediente") or "").strip(),
            "fecha": fecha,
            "sentido": str(pl.get("sentido") or h.get("sentido") or "").strip(),
            "tema": str(pl.get("tema_juridico") or h.get("tema_juridico") or "").strip(),
            "score": round(sc, 3),
            "pdf_url": str(pl.get("pdf_url") or h.get("pdf_url") or "").strip(),
        })
    filas = [f for f in filas if f["expediente"]]
    if len(filas) < PISO_FILAS:
        return []
    return filas


def resumen(filas: list) -> str:
    """El renglón que describe las seis filas. NO compara con el proyecto.

    Aquí estuvo la tentación y aquí se resiste: la calificación del proyecto es
    del planteamiento —fundado, infundado— y el `sentido` del acervo es del
    resolutivo —concede, niega, confirma—. No son la misma escala. Cruzarlas es
    lo que produjo el error del 45% en el grupo tributario, donde «confirma»
    confirmaba una CONCESIÓN. Se describe lo que hay y compara el secretario,
    que es de quien es el criterio.
    """
    if not filas:
        return ""
    tipos = {f["tipo_asunto"] for f in filas if f["tipo_asunto"]}
    if len(tipos) != 1:
        # Sobreseer en un amparo directo y confirmar en una queja no son el
        # mismo sentido ni son contrarios: mezclados, cualquier recuento miente.
        return ""
    if any(_TAUTOLOGICAS.search(f["tema"] or "") for f in filas):
        return ""
    cuenta = {}
    for f in filas:
        v = _norma(f["sentido"])
        if v:
            cuenta[v] = cuenta.get(v, 0) + 1
    if not cuenta:
        return ""
    tipo = tipos.pop()
    partes = ", ".join(f"{n} «{k.replace('_', ' ')}»"
                       for k, n in sorted(cuenta.items(), key=lambda x: -x[1]))
    return (f"De estos {len(filas)} asuntos propios, todos {tipo}: {partes}. "
            f"Compárelo usted con el sentido que propone.")
