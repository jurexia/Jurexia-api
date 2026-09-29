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
    0.50 o más CON AL MENOS TRES PARES DETRÁS (ver «Los dos niveles», abajo).
    SIN TABLA NO HAY PORCENTAJE: si el JSON no existe, no trae el tipo, o no
    trae un corte fiable, se devuelve una lista vacía y el taller cae al espejo
    viejo. Inventar un porcentaje sería de la misma especie que la tesis
    inventada.

 3. SÓLO DONDE SE CALIBRÓ. La tabla del 28-sep-2026 se midió dentro del 3TCC;
    entre tribunales el estilo del tema cambia y el corte puede moverse (lo
    dice el propio informe de calibración). Un «90%» en el Primer Tribunal
    calculado con la tabla del Tercero no sale de ninguna medición: fuera de
    los órganos calibrados, se calla.

 4. SIN PISO DE FILAS. El espejo viejo exigía tres sentencias porque su coseno
    crudo no distinguía el punto del vecino; una coincidencia calibrada vale
    sola.

 5. EL RESPALDO POR TEMA. Si ningún planteamiento llega al 85%, se prueba con
    los puntos `clase=asunto`, cuyo vector es el de «tipo. tema» de la
    sentencia, con su propia tabla. La fila lo dice (`fuente: "tema"`): no es
    lo mismo coincidir en el punto que coincidir en la materia del asunto.

LOS DOS NIVELES (David, 28-sep-2026: «adelante con la opción 1 + 2»)
====================================================================
La tabla del 28-sep no tiene, en ningún tipo, un tramo de planteamiento
sostenido por tres pares que llegue al 85%: los tramos altos dan entre 0.43 y
0.67. Con un solo nivel la tarjeta callaba casi siempre, y callaba también lo
que la medición sí respalda: «hay una probabilidad de 57% de que éste sea el
mismo problema». Por eso cada fila lleva `nivel`:

 · «mismo_problema»: probabilidad calibrada de 0.85 o más. Lo de siempre;
   hasta seis. Incluye el respaldo por tema, que sigue exigiendo 0.85.
 · «posible»: entre 0.50 y 0.85, hasta tres, y SÓLO por planteamiento. El tema
   del asunto coincide con demasiadas sentencias para que medio acierto diga
   algo, y un «posible» por tema sería un posible de un posible.

Las dos con su probabilidad REAL de la tabla, hacia abajo y con tope en 99: un
«posible» se enseña como lo que es, no se sube de nivel ni se le redondea a
favor. Los dos cortes se sacan de la tabla con la MISMA regla (primer tramo
sostenido que llega a la probabilidad) y ninguno está escrito aquí: al
recalibrar, lo que la tabla nueva diga manda.

EXACTA O COTA INFERIOR (`cota_inferior` en la fila). La tabla sólo MIDE un
número dentro de un tramo sostenido. Un coseno que cae en el hueco entre dos
tramos, o por encima del último sostenido, recibe el número del tramo de abajo:
la isotónica es monótona, así que la probabilidad verdadera es ésa o más, pero
no es la que la tabla midió en ese coseno. Con la tabla del 28-sep pasa con las
coincidencias MÁS altas del amparo directo: de 0.8147 a 0.9157 sólo hay tramos
de un par, y un 0.85 sale «57%» por el tramo de debajo. La fila lo dice y la
pantalla escribe «57% o más», en vez de hacer pasar la cota por la medida.

Una sentencia sale una sola vez: si un planteamiento suyo es «mismo problema»,
no vuelve a salir como «posible» por otro; si coincidió por tema al 85%, su
planteamiento al 57% no se repite abajo.

Y los posibles son LOS SIGUIENTES, no unos cualesquiera: si un planteamiento
que la tabla pone en 85% o más se queda fuera porque el JSON endureció o anuló
el corte de arriba, los posibles de ese planteamiento callan. Todos tendrían un
coseno menor que el escondido, y el secretario vería como mejor candidato uno
más débil sin saber que hay otro por encima.

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
      "umbral_50":     {"planteamiento": {"Queja": null, ...}}   (opcional)
      "organos":       ["<nombre OAJ exacto>" | "3TCC", ...]      (opcional)
    }

`n` es cuántos pares calificados sostienen el tramo. Un tramo sin `n` (tres
columnas) se lee, pero no se sabe cuánto lo sostiene y NO cuenta. `umbral_85`
presente y `null` es el calibrador diciendo «aquí no hay corte fiable»: se
calla, no se deduce otro de la tabla. Si trae número, sólo puede ENDURECER el
corte (ver `_corte`). `umbral_50` es lo mismo para el nivel «posible»: la
tabla del 28-sep no lo trae y manda la tabla; un número sólo lo endurece.

EL `null` CALLA EN LOS TRES PISOS. `"umbral_50": null` apaga el nivel entero,
`{"planteamiento": null}` lo apaga en esa clase y `{"planteamiento":
{"Queja": null}}` en ese tipo; igual con `umbral_85`. Es la forma natural de
escribir «aquí no», y la primera versión sólo la oía en el piso del tipo: con
`null` en la raíz el nivel seguía hablando. Un valor que no es número, objeto
ni `null` tampoco se adivina: calla ese nivel y lo dice el log.

EL CALIBRADOR NO CONSERVA `umbral_50`. `calibrar_openai.py` reescribe el JSON
entero y no lo escribe: al recalibrar se pierde cualquier `umbral_50` puesto a
mano. Si se anota uno, hay que volver a ponerlo después de cada calibración.
`organos` son los tribunales en que se midió; si falta, vale sólo el 3TCC, que
es donde se midió la del 28-sep-2026.

NO LO LLENA ESTE MÓDULO. La colección y la tabla las produce otro proceso; aquí
sólo se leen. Mientras la tabla no exista, este módulo calla siempre y el
taller sigue exactamente como estaba.

EL ORDEN DE DESPLIEGUE: PRIMERO EL FRONT
========================================
El front de `main` no conoce `nivel`, `similitud` ni el NEUN: pinta cada fila
como «una sentencia propia» más, bajo «las sentencias suyas más cercanas a cada
planteamiento». Un «posible» del 50% se vería igual que una del 95%, y por la
regla de `_espejo_propio` («si la OAJ habla en alguno, la tarjeta es suya
entera») reemplazaría al espejo viejo en todos los planteamientos. El front de
la rama `precedentes-oaj` ya aguanta este API y el anterior (una fila sin
`nivel` va arriba; una sin `similitud`, como del acervo viejo). Así que:

 1. el front sale a producción ANTES que este API, o a la vez;
 2. `oaj_calibracion.json` se versiona junto a este módulo DESPUÉS de lo
    anterior: es lo que enciende la fuente (sin él, calla siempre).

Si ese orden no se puede garantizar, `OAJ_POSIBLES=0` en el servicio calla el
nivel «posible» sin tocar el código, y se quita cuando el front esté arriba.
Render no relee las variables sin reiniciar el servicio.
"""

import inspect
import json
import math
import os
import re

COLECCION = "oaj_precedentes"
VECTOR = "dense"            # text-embedding-3-small, 1536, coseno: embed_leyes

# ── LO QUE SE ENSEÑA ─────────────────────────────────────────────────────────
# Los dos niveles de la tarjeta (ver «Los dos niveles» arriba). Son los pisos
# de PROBABILIDAD, no de coseno: el coseno que les corresponde sale de la
# tabla de cada tipo, y el JSON sólo puede subirlo.
PROB_MINIMA = 0.85          # «mismo problema»
PROB_POSIBLE = 0.50         # «posible precedente»
NIVEL_MISMO = "mismo_problema"
NIVEL_POSIBLE = "posible"
# Un tramo de la isotónica sostenido por UN par da «1.0» con un solo ejemplo.
# El calibrador sólo admite como corte un tramo con tres o más pares; aquí se
# aplica la misma regla, también para leer el porcentaje.
PARES_MINIMOS = 3
# Ni con la tabla entera a favor se afirma certeza: «100% de similitud» es una
# promesa que ningún banco de un centenar de pares sostiene.
TOPE_VISIBLE = 99
MOSTRADAS = 6
# Los posibles van DEBAJO y son menos: tres renglones que hay que revisar a
# mano ya son trabajo; seis serían una lista que nadie termina.
MOSTRADAS_POSIBLES = 3
# La clase del JSON con la tabla del PRIMER resultado. Ver «El primero no es
# como los demás».
CLASE_PRIMERO = "planteamiento_r1"


def posibles_activos() -> bool:
    """¿Habla el nivel «posible»? Interruptor `OAJ_POSIBLES`, encendido si no
    se dice nada.

    Es la reversa sin despliegue (ver «El orden de despliegue»): si el API
    llega a producción antes que el front que sabe pintar los dos niveles,
    `OAJ_POSIBLES=0` calla el nivel de abajo y el de arriba sigue igual. Se lee
    en cada consulta, no al importar; aun así Render sólo ve el cambio tras
    reiniciar. Cualquier valor que no sea un «sí» claro lo apaga: una errata
    en la variable no debe encender lo que se quiso callar.
    """
    return os.getenv("OAJ_POSIBLES", "1").strip().lower() in ("1", "true", "si", "sí")
# Se piden más de las que se muestran porque una sentencia trae varios
# planteamientos y varios del mismo asunto entran juntos: se queda el mejor de
# cada NEUN. Con treinta solía haber seis distintos; con los dos niveles hacen
# falta nueve, y se piden en proporción. Es la misma búsqueda, no otra.
PEDIDOS_PLANTEAMIENTO = 45
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
    "de las sentencias ya leídas y, si ninguno es el mismo problema, con el "
    "tema de los asuntos. Que un asunto no salga aquí no quiere decir que el "
    "tribunal no lo haya resuelto.")

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


def lectura(tabla, coseno: float):
    """(probabilidad, exacta) de ese coseno; (None, False) si no hay dato.

    Es la función escalonada de la regresión isotónica, leída SÓLO sobre los
    tramos sostenidos por `PARES_MINIMOS` pares o más: manda el último de ellos
    cuyo inicio queda por debajo del coseno. `exacta` es True sólo si el coseno
    cae DENTRO de ese tramo, entre su inicio y su final: ahí la tabla midió ese
    número. Fuera, el número es una COTA INFERIOR:

    · un coseno que cae en un hueco entre tramos sostenidos recibe la
      probabilidad del tramo de ABAJO, que en una tabla monótona es la menor;
    · un tramo de uno o dos pares —el «1.0» de un único ejemplo— no pone
      porcentaje: se lee el tramo sostenido de debajo;
    · por encima del último tramo sostenido se da el suyo. La isotónica sí
      tiene datos ahí (de un par), pero ninguno sostenido: lo único que se
      puede afirmar es «ése o más».

    Por debajo del primer tramo sostenido no hay dato y no se extrapola.

    El final del tramo viene redondeado a cuatro decimales, así que un coseno
    en la cuarta cifra del borde puede salir como cota sin serlo. Es el lado
    bueno del error: «57% o más» también es verdad cuando es 57%.
    """
    p, exacta = None, False
    for cmin, cmax, prob, _n in _sostenidos(tabla):
        if coseno >= cmin:
            p, exacta = prob, coseno <= cmax
        else:
            break
    return p, exacta


def probabilidad(tabla, coseno: float):
    """La probabilidad calibrada de ese coseno, o None. Ver `lectura`: es su
    número, sin decir si es exacto o cota inferior."""
    return lectura(tabla, coseno)[0]


def _corte(cal: dict, clase: str, tipo: str, tabla, prob: float = PROB_MINIMA,
           clave: str = "umbral_85"):
    """El coseno a partir del cual la tabla puede dar `prob`. None si nunca.

    Es el inicio del primer tramo SOSTENIDO que llega a `prob` —la misma regla
    con que el calibrador escribe `umbral_85`—: por debajo de él ningún coseno
    puede pasar, así que se le pide a Qdrant que ni lo mande. El corte del
    nivel «posible» es esta misma función con 0.50 y `umbral_50`: una sola
    regla para los dos niveles, para que no puedan leer la tabla distinto.

    `umbral_85` (o `umbral_50`) del JSON, para esa clase y tipo:
      · ausente → manda la tabla;
      · con número → se toma el MÁS ALTO de los dos: si las dos anotaciones
        discrepan, callar es el fallo barato;
      · presente y `null` → el calibrador dejó escrito que ahí no hay corte
        fiable. None, y se calla. Ignorarlo y deducir un corte de la tabla fue
        lo que hacía la primera versión, y con la tabla real ponía «100% de
        similitud» en el amparo directo sobre un tramo de un par.

    El `null` vale en los tres pisos —la clave entera, la clase y el tipo—, y
    un valor de forma desconocida en cualquiera de ellos calla con aviso. La
    versión anterior convertía el `null` de la raíz o de la clase en «no dice
    nada» (`get(...) or {}`), y el nivel que se quiso apagar seguía hablando.
    """
    cortes = [cmin for cmin, _c, p, _n in _sostenidos(tabla) if p >= prob]
    if not cortes:
        return None
    corte = min(cortes)
    cal = cal if isinstance(cal, dict) else {}
    if clave not in cal:
        return corte
    # Piso por piso: la clave, la clase, el tipo. En cada uno, ausente sigue
    # bajando, `null` calla, y lo que no se sabe leer calla y se dice.
    nodo, ruta = cal[clave], clave
    for piso in (clase, tipo):
        if nodo is None:
            return None
        if not isinstance(nodo, dict):
            _avisar_una_vez(("umbral_forma", id(cal), ruta),
                            f"`{ruta}` no es un objeto ni null; el nivel que "
                            f"gobierna calla")
            return None
        if piso not in nodo:
            return corte
        nodo, ruta = nodo[piso], f"{ruta}.{piso}"
    if nodo is None:
        return None
    declarado = _numero(nodo)
    if declarado is None:
        _avisar_una_vez(("umbral_forma", id(cal), ruta),
                        f"`{ruta}` no es un número ni null; ese nivel calla")
        return None
    return max(corte, declarado)


def _corte_posible(cal: dict, clase: str, tipo: str, tabla):
    """El corte del nivel «posible»: 0.50, con `umbral_50`. Ver `_corte`."""
    return _corte(cal, clase, tipo, tabla, PROB_POSIBLE, "umbral_50")


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


def _mejores(puntos, fuente: str) -> list:
    """[(coseno, NEUN, payload)]: el mejor punto de cada NEUN, de mayor a menor.

    UN PLANTEAMIENTO NO ES UNA SENTENCIA. Tres planteamientos del mismo asunto
    entrarían como tres precedentes y son uno: se queda el de mejor coseno. Se
    hace ANTES de repartir por nivel, y así una sentencia cae en un solo nivel,
    el de su mejor planteamiento: nunca sale arriba por uno y abajo por otro.
    """
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
    return [(sc, n, pl) for n, (sc, pl)
            in sorted(mejor.items(), key=lambda kv: -kv[1][0])]


def _fila(sc: float, n: int, pl: dict, prob: float, fuente: str,
          nivel: str, exacta: bool = True) -> dict:
    """Una fila en el formato de la tarjeta."""
    es_pl = fuente == "planteamiento"
    return {
        "tipo_asunto": _txt(pl, "tipo"),
        "expediente": _txt(pl, "alias"),
        "fecha": _txt(pl, "fecha"),
        "sentido": _txt(pl, "sentido"),
        "tema": _txt(pl, "tema"),
        "score": round(sc, 3),
        # La OAJ no da PDF por enlace directo; el campo se conserva vacío
        # para que la fila tenga la misma forma que las del espejo viejo.
        "pdf_url": "",
        # Hacia abajo: 0.869 es «86%», nunca «87%», y 0.571 es «57%». El
        # porcentaje ya es una afirmación; no se redondea a favor. Y con tope:
        # ver arriba. Vale igual para los dos niveles: un «posible» lleva su
        # número real, no uno que lo acerque al nivel de arriba.
        "similitud": min(int(prob * 100 + 1e-9), TOPE_VISIBLE),
        # True cuando el coseno no cae dentro de un tramo sostenido y el número
        # es el del tramo de abajo: la pantalla dice «57% o más» (ver
        # «Exacta o cota inferior» arriba).
        "cota_inferior": not exacta,
        "fuente": fuente,
        "nivel": nivel,
        "pregunta": _txt(pl, "pregunta") if es_pl else "",
        "razon": _txt(pl, "razon") if es_pl else "",
        "calificacion": _txt(pl, "calificacion") if es_pl else "",
        "autoridad": _txt(pl, "autoridad") if es_pl else "",
        "neun": n,
        "enlace_oaj": ENLACE_OAJ,
    }


def _repartir(mejores, tabla, fuente: str, corte_mismo, corte_posible,
              excluir=frozenset(), primero=None):
    """(mismo problema, posibles, escondidas): cada NEUN calibrado y puesto en
    su nivel.

    `primero` es (tabla, corte_mismo, corte_posible) del PRIMER resultado —la
    sentencia más cercana, ya sin el propio asunto— o None. Si viene, esa
    sentencia se lee con su tabla y sus cortes y las demás con los de siempre
    (ver «El primero no es como los demás»). El rango es la posición en
    `mejores`, ANTES de `excluir`: si la primera ya salió por tema, la segunda
    sigue siendo la segunda y se lee con la tabla de las demás.

    `corte_posible` None apaga el nivel de abajo (el respaldo por tema lo
    llama así: por tema sólo se enseña lo del 85%). `excluir` son NEUN que ya
    salieron en otro nivel; se saltan ANTES de contar el tope, para que el
    hueco lo ocupe la siguiente y no quede un renglón menos.

    EL CORTE SE COMPRUEBA AQUÍ TAMBIÉN, no sólo en Qdrant: el
    `score_threshold` es una petición al servidor, y el porcentaje es una
    afirmación nuestra. Y cada nivel con SU corte, porque a Qdrant se le pidió
    desde el más bajo de los dos.

    `escondidas` cuenta las sentencias que la tabla pone en 85% o más y el
    corte de arriba dejó fuera. Eso sólo pasa si el JSON lo endureció o lo
    anuló (el corte de la tabla es, por construcción, el inicio del primer
    tramo sostenido que llega al 85%), y entonces LOS POSIBLES CALLAN: todos
    tienen un coseno menor que el escondido, y enseñarlos sería presentar
    como mejor candidato a uno más débil, sin decir que hay otro por encima.
    """
    mismas, posibles = [], []
    escondidas = 0
    for pos, (sc, n, pl) in enumerate(mejores):
        if n in excluir:
            continue
        if pos == 0 and primero is not None and primero[0]:
            tabla_i, c_mismo, c_posible = primero
        else:
            tabla_i, c_mismo, c_posible = tabla, corte_mismo, corte_posible
        prob, exacta = lectura(tabla_i, sc)
        if prob is None:
            continue
        if prob >= PROB_MINIMA:
            # Del 85% para arriba es «mismo problema» o nada. Si el JSON
            # endureció o anuló ese corte y este coseno no lo pasa, NO baja a
            # «posible»: su probabilidad no está entre 0.50 y 0.85, y
            # enseñarla ahí sería ponerle otro número. Y pasada la sexta, la
            # séptima tampoco baja: el nivel de abajo no es un cajón de sobras.
            if c_mismo is None or sc < c_mismo:
                escondidas += 1
            elif len(mismas) < MOSTRADAS:
                mismas.append(_fila(sc, n, pl, prob, fuente, NIVEL_MISMO,
                                    exacta))
        elif prob >= PROB_POSIBLE:
            if (c_posible is not None and sc >= c_posible
                    and len(posibles) < MOSTRADAS_POSIBLES):
                posibles.append(_fila(sc, n, pl, prob, fuente, NIVEL_POSIBLE,
                                      exacta))
    if escondidas:
        posibles = []
    return mismas, posibles, escondidas


def numero_expediente(x):
    """(número, año) de un expediente escrito como sea, o None.

    «17/2025», «17-2025», «3._ARA_17-2025» y «AR 17 / 2025» dan (17, "2025").
    Sirve para reconocer el PROPIO asunto entre sus precedentes: si ya está
    resuelto y publicado en la OAJ (un reproceso, un asunto de prueba), su
    sentencia sale arriba como precedente de sí misma. Medido el 28-sep en la
    prueba de humo: el AR 17/2025 y el AD 704/2022 se citaban a sí mismos. El
    tipo no se compara porque la búsqueda ya filtra por tipo.
    """
    m = re.search(r"(\d{1,5})\s*[/\-]\s*(\d{4})", str(x or ""))
    return (int(m.group(1)), m.group(2)) if m else None


def _sin_el_propio(mejores, expediente_propio) -> list:
    """`mejores` sin las sentencias del propio asunto. ANTES de contar rangos:
    si el propio asunto fuera el primero, el segundo se leería con la tabla de
    las demás y perdería el nivel que la del primero le da."""
    propio = numero_expediente(expediente_propio)
    if not propio:
        return list(mejores)
    return [m for m in mejores if numero_expediente((m[2] or {}).get("alias")) != propio]


async def precedentes_oaj(qdrant, embed, problema, tipo_taller: str,
                          organo_oaj: str, expediente_propio: str = "") -> list:
    """Los precedentes del propio tribunal para UN planteamiento.

    Hasta seis de nivel «mismo_problema» y, detrás, hasta tres «posible»; cada
    fila dice el suyo en `nivel` (ver «Los dos niveles»). Primero van todas
    las del nivel de arriba.

    `problema` es el dict de la fase 3 (pregunta, combate, resolvio); cualquier
    otra cosa —una cadena, un dict sin alguna de las tres— calla, porque la
    tabla sólo vale para la forma con que se midió (ver `texto_consulta`).
    `tipo_taller` es la clave del taller (amparo_directo, amparo_revision,
    queja, revision_fiscal) y `organo_oaj` el nombre exacto con el que la OAJ
    indexa al tribunal (ver `organo_de`). `expediente_propio` es el número del
    asunto que se proyecta: su propia sentencia, si ya está publicada, no es
    su precedente y no cuenta para el rango (ver `numero_expediente`).

    Lista vacía en cualquier duda: sin tabla, sin tipo, sin órgano, órgano no
    calibrado, sin nada que llegue al 50% (o al 85% por tema), o con cualquier
    error. El taller cae entonces al espejo viejo; callar aquí es el fallo
    barato y nunca tumba la sentencia.
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
        posible_pl = _corte_posible(cal, "planteamiento", tipo, t_pl)
        # EL PRIMERO, CON SU TABLA (ver «El primero no es como los demás»).
        # Sin ella, el primero se lee con la de todos, como antes.
        t_r1 = tabla_de(cal, CLASE_PRIMERO, tipo)
        corte_r1 = _corte(cal, CLASE_PRIMERO, tipo, t_r1) if t_r1 else None
        posible_r1 = _corte_posible(cal, CLASE_PRIMERO, tipo, t_r1) if t_r1 else None
        if (posible_pl is not None or posible_r1 is not None) and not posibles_activos():
            # La reversa sin despliegue (ver «El orden de despliegue»). Se
            # apaga ANTES de decidir si se embebe: si sólo hablaba el nivel de
            # abajo, no se paga una llamada cuyo resultado no se enseña.
            _avisar_una_vez(("posibles_apagados",),
                            "OAJ_POSIBLES apagado: el nivel «posible» calla")
            posible_pl = posible_r1 = None
        # Por tema sólo el nivel de arriba: no se calcula corte «posible».
        corte_as = _corte(cal, "asunto", tipo, t_as)
        # Sin ningún corte no se embebe siquiera: el porcentaje no se puede
        # decir y no se gasta la llamada.
        if (corte_pl is None and posible_pl is None and corte_r1 is None
                and posible_r1 is None and corte_as is None):
            _avisar_una_vez(("sin_corte", id(cal), tipo),
                            f"sin corte fiable (ni al 85% ni al 50% por "
                            f"planteamiento, ni al 85% por tema) para {tipo}; "
                            f"la fuente OAJ calla en ese tipo")
            return []

        try:
            vector = await embed(consulta)
        except Exception as e:
            print(f"   ⚠️ precedentes OAJ: no se pudo embeber el problema: {e}")
            return []

        mejores_pl = []
        primero = (t_r1, corte_r1, posible_r1) if t_r1 else None
        cortes_pl = [c for c in (corte_pl, posible_pl, corte_r1, posible_r1)
                     if c is not None]
        if cortes_pl:
            # UNA SOLA BÚSQUEDA PARA LOS DOS NIVELES —y para el primero y los
            # demás—, desde el corte más bajo que aplique, y luego se reparte.
            # Dos búsquedas devolverían las mismas sentencias arriba y abajo y
            # habría que casarlas después. Pedir desde el corte más bajo no
            # cambia el orden: el primero sigue siendo el de mayor coseno.
            puntos = await _buscar(qdrant, vector, "planteamiento", tipo,
                                   organo, PEDIDOS_PLANTEAMIENTO, min(cortes_pl))
            mejores_pl = _sin_el_propio(_mejores(puntos, "planteamiento"),
                                        expediente_propio)
        mismas, posibles, escondidas = _repartir(
            mejores_pl, t_pl, "planteamiento", corte_pl, posible_pl,
            primero=primero)
        if escondidas:
            _avisar_una_vez(("escondidas", id(cal), tipo),
                            f"{tipo}: un planteamiento que la tabla pone en 85% "
                            f"o más no pasa el `umbral_85` del JSON; ahí los "
                            f"posibles callan para no enseñar uno más débil "
                            f"como el mejor")

        # EL RESPALDO, cuando ningún planteamiento llega al 85%, aunque haya
        # posibles: un tema al 90% dice más que un planteamiento al 57%, y los
        # posibles no son el nivel de arriba. Mismo vector: el de la consulta
        # del planteamiento contra el vector de «tipo. tema» del asunto. Así se
        # midió la tabla de `asunto` (pares_reales: planteamiento real del
        # taller contra el tema de la OAJ); con otra forma de consulta su
        # porcentaje no significaría lo que dice.
        if not mismas and corte_as is not None:
            # SU PROPIA RED. Antes de los dos niveles esta búsqueda sólo corría
            # cuando no había nada; ahora corre también con posibles ya
            # calculados, y una falla suya (un timeout de Qdrant) no puede
            # llevarse lo que la primera búsqueda sí respondió: se enseña lo
            # del planteamiento y el tema se da por no encontrado.
            try:
                puntos = await _buscar(qdrant, vector, "asunto", tipo,
                                       organo, PEDIDOS_ASUNTO, corte_as)
                por_tema, _, _ = _repartir(
                    _sin_el_propio(_mejores(puntos, "tema"), expediente_propio),
                    t_as, "tema", corte_as, None)
            except Exception as e:
                print(f"   ⚠️ precedentes OAJ: falló el respaldo por tema "
                      f"({e}); se enseña lo del planteamiento")
                por_tema = []
            if por_tema:
                mismas = por_tema
                # UNA SENTENCIA, UNA VEZ. Si su tema coincidió al 85%, su
                # planteamiento al 57% no se repite abajo, y su lugar entre
                # los tres lo toma el siguiente posible. Con los MISMOS cortes
                # de la primera pasada, para que «escondidas» se cuente con la
                # misma regla: los posibles sólo reviven si la sentencia
                # escondida es justo la que salió arriba por tema, que ya está
                # a la vista.
                _, posibles, _ = _repartir(mejores_pl, t_pl, "planteamiento",
                                           corte_pl, posible_pl,
                                           excluir={f["neun"] for f in mismas},
                                           primero=primero)
        return mismas + posibles
    except Exception as e:
        print(f"   ⚠️ precedentes OAJ: {e}")
        return []
