"""LAS DOS FORMAS DE LA SENTENCIA — la estándar y la moderna.

POR QUÉ EXISTE. David, 25-sep-2026: «una sentencia de Tribunal Colegiado de
Circuito, aunque debe ser exhaustiva, no necesariamente debe ser tan extensa.
Creo que podemos sintetizar más (sin dejar de referirnos a hechos, sentencia,
agravios o conceptos) cada aspecto de lo que entrega el producto terminado.
Para esto vamos a dar dos opciones de sentencia (la actual se queda como es) y
"Generar sentencia en versión moderna" (…) una sentencia que atiende al
problema jurídico central de forma exhaustiva, con argumentación de alto nivel,
pero prescinde de información irrelevante. En cambio, la sentencia con el
formato estándar, vamos a prescindir de generar problemas jurídicos, sino
establecer el formato estándar. Es decir, "Sobre el primer concepto de
violación, en el que la quejosa sostiene…. Se considera infundado. Lo
anterior…", mientras que el formato moderno empieza con la interrogante y
enseguida se responde. Fundamental que el formato moderno, aunque con menos
texto, aborde todos los puntos planteados en el recurso o demanda de amparo.
Pero resuma o prescinda de resumir consideraciones que no serán materia de
estudio».

LO QUE HABÍA. El estudio se escribía con la pregunta de cada problema jurídico
como rótulo —«1. ¿La Sala debió admitir la ampliación…?» / «Sí. …»—: ésa es
ya la forma moderna. Medido en el ADC 93/2026 v8: 8,463 palabras, el 72 % en
el considerando de estudio (6,113), y dentro de él 1,600 de resumen de la
sentencia reclamada —casi todo sobre competencia y firma, que nadie combatía—
y 1,400 de resumen de los conceptos antes de la primera línea de solución.

CÓMO QUEDA:
  · ESTÁNDAR — la extensión y los resúmenes completos de siempre, pero el
    estudio va CONCEPTO POR CONCEPTO, en el orden del promovente, con la
    fórmula del oficio. Los problemas jurídicos se siguen calculando: deciden
    el sentido en la pantalla y guían el estudio; lo que ya no hacen es
    rotularlo.
  · MODERNA — la pregunta y enseguida la respuesta, con un objetivo de
    palabras menor; los antecedentes y los dos resúmenes se CONDENSAN a lo que
    es materia de estudio, en una llamada barata que corre EN PARALELO al
    estudio (no alarga la espera). Si el resumen condensado pierde un concepto,
    se usa el completo: la síntesis nunca paga con exhaustividad.

EN LAS DOS se comprueba al final que cada concepto o agravio numerado recibe
respuesta en el estudio (`sin_contestar`), calibrado sobre los engroses reales
del banco Kingston para no acusar a los buenos.
"""
from __future__ import annotations

import json
import re
import unicodedata

ESTANDAR = "estandar"
MODERNA = "moderna"
FORMATOS = (ESTANDAR, MODERNA)


def normalizar(x) -> str:
    t = unicodedata.normalize("NFKD", str(x or "").strip().lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return MODERNA if t.startswith("modern") else ESTANDAR


def rotulo(formato: str) -> str:
    return "versión moderna" if normalizar(formato) == MODERNA else "formato estándar"


# ═══════════════════════════════════════════════════════════════════════════
# CUÁNTO SE ESCRIBE
# ═══════════════════════════════════════════════════════════════════════════
# La estándar conserva la medida del corpus (fase6.PALABRAS_ESTUDIO = 3733).
# La moderna se mide por lo que HAY QUE DECIDIR: un problema que cae con el
# principal o queda sin materia se despacha en un párrafo y no suma. Con el
# 93/2026 —un principal y un accesorio sin materia— da 1,600, frente a las
# ~3,000 que ocupó la solución en v8.
MODERNA_BASE = 1000
MODERNA_POR_PROBLEMA = 400
MODERNA_MIN = 1600
MODERNA_MAX = 3000


def _vivo(c) -> bool:
    s = str(getattr(c, "sentido", "") or "").lower()
    r = str(getattr(c, "razonamiento", "") or "")
    return s != "innecesario" and not r.startswith("Descansa en la premisa que se desestimó")


def palabras_moderna(criterios: list) -> int:
    vivos = sum(1 for c in (criterios or []) if _vivo(c)) or 1
    return max(MODERNA_MIN, min(MODERNA_MAX, MODERNA_BASE + MODERNA_POR_PROBLEMA * vivos))


# ═══ LA EXTENSIÓN SIN PRESIÓN (p2-congruencia, 26-sep-2026) ════════════════
# David, tras medir v1/v3/v4: acepta que el estudio crezca de ~1,500 a
# 2,000–2,500 palabras si eso da la exhaustividad; la brevedad NO es el
# objetivo. La v3/v4 contestó en genérico más argumentos que la v1 (decisivos
# con respuesta propia 98 / 90 / 79 %) con la Solución a la mitad.
#   · EL TECHO DE LA MODERNA EN LA FAMILIA v2 es alto: una vez y media su
#     medida —el punto donde `fase6_estudio.revisar` ya dice «salió larga»—.
#     Antes el prompt le daba de techo la medida misma, y eso es una meta.
#   · LA REFERENCIA DE LA ESTÁNDAR EN LA FAMILIA v2: 2,000 a 2,500 palabras,
#     más cuantos más problemas vivos. Es lo que usa `revisar` para «se quedó
#     corto» (con el 45 %, como la v1), nunca una cifra en el prompt.
MODERNA_TECHO_FACTOR = 1.5
ESTANDAR_REFERENCIA_MIN = 2000
ESTANDAR_REFERENCIA_MAX = 2500
ESTANDAR_REFERENCIA_BASE = 1500
ESTANDAR_REFERENCIA_POR_PROBLEMA = 250


def techo_moderna(criterios: list) -> int:
    """El techo alto de la moderna (familia v2): no es meta."""
    return round(MODERNA_TECHO_FACTOR * palabras_moderna(criterios))


def referencia_estandar(criterios: list) -> int:
    """La referencia de la estándar (familia v2), de 2,000 a 2,500 según los
    problemas vivos: la que mide si el estudio se quedó corto."""
    vivos = sum(1 for c in (criterios or []) if _vivo(c)) or 1
    return max(ESTANDAR_REFERENCIA_MIN, min(ESTANDAR_REFERENCIA_MAX,
               ESTANDAR_REFERENCIA_BASE + ESTANDAR_REFERENCIA_POR_PROBLEMA * vivos))


# ═══════════════════════════════════════════════════════════════════════════
# LO QUE CAMBIA EN EL PROMPT DEL ESTUDIO
# ═══════════════════════════════════════════════════════════════════════════
def forma_del_estudio(formato: str, q: str, q1: str, parte: str, calif: str,
                      palabras: int, variante: str = "v1") -> str:
    """El bloque de FORMA que decide cómo se ordena y se abre cada apartado.

    Sustituye a las cuatro instrucciones que antes obligaban a la pregunta en
    los dos casos —la de FORMA, la de viñetas, la de «tu texto empieza» y la
    del criterio— para que no convivan dos órdenes contrarias en el prompt.

    `variante` es la del prompt del estudio (ver `fase6_estudio.VARIANTES`):
    la v2 cambia la unidad de la estándar y quita el ejemplo de la moderna.
    """
    if str(variante or "").strip().lower() == "v2":
        return _forma_del_estudio_v2(formato, q, q1, parte, calif, palabras)
    # SIN ALZADA NO HAY «SALA» EN EL EJEMPLO (30-sep-2026): un ejemplo del
    # prompt se copia literal, y en un juicio oral mercantil la responsable es
    # el juez. Sólo si la instancia consta como única; si no, como siempre.
    try:
        import tipos_asunto as _ta_fs
        _quien = "La autoridad responsable" if _ta_fs.instancia_actual() == "unica" else "La Sala responsable"
    except Exception:
        _quien = "La Sala responsable"
    if normalizar(formato) == MODERNA:
        return f"""
FORMATO: VERSIÓN MODERNA — la pregunta y enseguida la respuesta.
- ABRE con la calificación general en una frase («Los {q} son {calif}.») y
  ACTO SEGUIDO el primer problema, numerado y como pregunta, SOLA en su
  párrafo y terminada en «?». El párrafo siguiente EMPIEZA POR LA RESPUESTA
  —«Sí.», «No.», «No lo estaba.»— y en esa misma frase NOMBRA el {q1} o los
  {q} que contesta y su calificación: «No. El primer {q1} es infundado,
  porque…». Así:

      1. ¿{_quien} estaba obligada a precisar la unidad de
      cuantificación aplicable a cada periodo?

      Sí lo estaba. El segundo {q1} es fundado, porque el artículo 50 de la
      Ley Federal de Procedimiento…

  Nombrar el {q1} no es adorno: es lo que permite comprobar, leyendo, que
  ninguno se quedó sin respuesta.
- ES LA VERSIÓN CORTA, NO LA INCOMPLETA. Todos los planteamientos se
  contestan —uno olvidado es un amparo de vuelta—, pero se escribe con
  economía: alrededor de {palabras} palabras EN TOTAL.
    · EL PROBLEMA CENTRAL se razona con argumentación de alto nivel y a
      fondo: la regla, su fuente, la aplicación a estos hechos y la objeción
      seria refutada. Nada más. Cada párrafo avanza; ninguno repite.
    · UNO O DOS CRITERIOS por problema, los que de verdad deciden, no una
      batería. Un criterio que sólo refuerza lo ya fundado sobra aquí.
    · LO ACCESORIO, EN UN PÁRRAFO: qué se alegó, su calificación y por qué.
    · FUERA lo que no decide: el recuento de antecedentes, lo que la
      responsable resolvió y nadie combate, los «no pasa inadvertido» sobre
      objeciones que nadie planteó, y la paráfrasis de la tesis recién citada.
{_remision_antecedentes()}
"""
    return f"""
FORMATO: ESTÁNDAR — concepto por concepto, con la fórmula del oficio.
- NO HAY PREGUNTAS NI RÓTULOS NUMERADOS. Los problemas jurídicos que vienen
  abajo son tu guía para decidir y ordenar; NO se escriben en la sentencia.
  Ninguna línea del estudio empieza por «¿» ni por «1.», «2.».
- ABRE con la calificación general en una frase («Los {q} son {calif}.») y,
  si reagrupas o alteras el orden, el anuncio de método con el artículo 76 de
  la Ley de Amparo. Después, UN APARTADO POR {q1.upper()}, en el orden en que
  los formuló {parte}, y cada uno ABRE ASÍ, en este orden:
    1) el {q1} por su ordinal y lo que en él se sostiene, en su versión más
       fuerte: «Sobre el primer {q1}, en el que {parte} sostiene que…»;
    2) su calificación, en la frase siguiente: «Se considera infundado.»,
       «Es fundado.», «Resulta inoperante.»;
    3) y la demostración, que arranca con «Lo anterior, porque…», «Lo
       anterior es así, ya que…» o «Lo anterior se estima así, toda vez
       que…», y recorre la premisa normativa, la aplicación al caso y la
       conclusión.
  LA CALIFICACIÓN DEL PASO 4 SE ADELANTA a la frase que sigue al
  planteamiento; el apartado CIERRA con lo que se sigue de ella.
  Varía las entradas —«Sobre el segundo…», «En relación con el tercer…»,
  «Por lo que hace al cuarto…»— y no copies la fórmula letra por letra en
  todos; lo que no varía es el orden: planteamiento, calificación, «Lo
  anterior…».
- SI UN {q1.upper()} PLANTEA VARIOS PROBLEMAS con sentidos distintos, se dice
  en su apertura —«es fundado en una parte e infundado en otra»— y se
  contesta cada parte bajo el mismo apartado, en párrafos sucesivos.
- SI VARIOS {q} SE ESTUDIAN JUNTOS, el apartado abre nombrándolos todos
  —«Sobre los {q} segundo y tercero, en los que {parte} sostiene…»— con una
  sola calificación.
- LOS QUE QUEDAN SIN MATERIA O CAEN CON EL PRINCIPAL abren igual y se
  despachan en un párrafo: «Sobre el segundo {q1}, en el que… Su estudio
  resulta innecesario, dado que…».
"""


# ═══ LA FORMA EN LA v2 DEL PROMPT (26-sep-2026) ═══════════════════════════
# Lo que cambia, por la propuesta del estudio que David aprobó:
#   · ESTÁNDAR. La unidad deja de ser el concepto con su premisa propia —«un
#     apartado por concepto» era una de las cinco órdenes que fabricaban la
#     repetición (C1)— y pasa a ser la consideración atacada con la razón que
#     la decide, nombrando los conceptos que entran. El apartado de un solo
#     concepto sigue abriendo con la fórmula de David. La calificación va una
#     vez, al abrir (fila 16): el apartado ya no «cierra» con ella. La
#     apertura dice qué consideración se ataca en no más de treinta palabras:
#     el sistema gastaba 59-84 en re-contar el concepto y el engrose, 27
#     (fila 4). Lo accesorio se mide con la tabla única del prompt (fila 13).
#   · MODERNA. Sale el ejemplo de la unidad de cuantificación y el artículo 50
#     de la LFPCA (fila 14b: un ejemplo con materia se copia), la objeción
#     deja de pedirse aquí (fila 8: se pedía desde cuatro sitios) y la medida
#     es un techo. Cada apartado sigue siendo un problema con su pregunta.
def _forma_del_estudio_v2(formato: str, q: str, q1: str, parte: str, calif: str,
                          palabras: int) -> str:
    # LA APERTURA YA NO ES UNA FRASE HECHA (revisión adversarial de la fase E,
    # 28-sep-2026): «ABRE con… («Los agravios son en parte fundados y en parte
    # innecesarios.»)» llegaba al prompt v4 del AR 631/2025 ya concordada con
    # sus calificaciones, y se copiaba literal. La calificación va como dato.
    # La v1 (`forma_del_estudio` sin variante) queda como estaba: congelada.
    if normalizar(formato) == MODERNA:
        return f"""
FORMATO: VERSIÓN MODERNA — la pregunta y enseguida la respuesta.
- ABRE con la calificación general de los {q}, en una frase tuya (dato: {calif};
  no es una fórmula que copiar) y
  ACTO SEGUIDO el primer problema, numerado y como pregunta, SOLA en su
  párrafo y terminada en «?». El párrafo siguiente EMPIEZA POR LA RESPUESTA
  —«Sí.», «No.»— y enseguida, EN ESE MISMO PÁRRAFO, NOMBRA el {q1} o los {q}
  que contesta y su calificación, antes de demostrarla: la respuesta no va
  sola en su renglón. Nombrar el {q1} no es adorno: es lo que permite
  comprobar, leyendo, que ninguno se quedó sin respuesta.
- ES LA VERSIÓN CORTA, NO LA INCOMPLETA. Todos los planteamientos se
  contestan —uno olvidado es un amparo de vuelta—. Lo que se condensa es lo
  que no decide —lo de la lista FUERA—, nunca la respuesta de un argumento
  que trae un dato propio. El techo de {palabras} palabras que fija la
  extensión es alto y no es una meta.
    · EL PROBLEMA CENTRAL se razona a fondo y con argumentación de alto nivel:
      la regla, su fuente y la aplicación a estos hechos. Cada párrafo avanza;
      ninguno repite.
    · UNO O DOS CRITERIOS por problema, los que de verdad deciden, no una
      batería. Un criterio que sólo refuerza lo ya fundado sobra.
    · LO ACCESORIO, con la medida única de la extensión.
    · FUERA lo que no decide: el recuento de antecedentes, lo que la
      responsable resolvió y nadie combate, los «no pasa inadvertido» sobre
      objeciones que nadie planteó, y la paráfrasis de la tesis recién citada.
{_remision_antecedentes()}
"""
    return f"""
FORMATO: ESTÁNDAR — por la consideración que se ataca, con la fórmula del oficio.
- NO HAY PREGUNTAS NI RÓTULOS NUMERADOS. Los problemas jurídicos que vienen
  abajo son tu guía para decidir y ordenar; NO se escriben en la sentencia.
  Ninguna línea del estudio empieza por «¿» ni por «1.», «2.».
- ABRE con la calificación general de los {q}, en una frase tuya (dato: {calif};
  no es una fórmula que copiar) y,
  si reagrupas o alteras el orden, el anuncio de método con el artículo 76 de
  la Ley de Amparo.
- LA UNIDAD DEL ESTUDIO ES LA CONSIDERACIÓN QUE SE ATACA Y LA RAZÓN QUE LA
  DECIDE, NO EL {q1.upper()}. Los {q} que atacan la misma consideración y caen
  por la misma razón se estudian en UN apartado que los nombra a todos por su
  ordinal; el que ataca otra consideración, o cae por otra razón, lleva el
  suyo. Un {q1} que trae dos argumentos de distinta suerte se parte: cada
  parte con su respuesta y su calificación.
- CADA APARTADO ABRE ASÍ, en este orden:
    1) si es de un solo {q1}, con la fórmula del oficio: «Sobre el primer {q1},
       en el que {parte} sostiene que…», diciendo en no más de treinta
       palabras QUÉ CONSIDERACIÓN ataca —no el argumento entero: el resumen de
       arriba ya lo expuso—; si junta varios, «Sobre los {q} segundo y
       tercero, en los que…», y dice qué los une;
    2) su calificación, en la frase siguiente Y EN EL MISMO PÁRRAFO —el
       párrafo que abre el apartado termina con ella—: «Se considera
       infundado.», «Es fundado.», «Resulta inoperante.». Nunca en un renglón
       aparte ni después de la demostración;
    3) y la demostración, en el párrafo siguiente, que arranca con «Lo
       anterior, porque…», «Lo anterior es así, ya que…» o «Lo anterior se
       estima así, toda vez que…». Ese «Lo anterior» es la calificación que se
       acaba de decir: sin ella antes, no tiene a qué referirse.
  Varía las entradas —«Sobre el segundo…», «En relación con el tercer…»,
  «Por lo que hace al cuarto…»— y no copies la fórmula letra por letra en
  todos; lo que no varía es el orden: planteamiento, calificación, «Lo
  anterior…».
- LA CALIFICACIÓN VA UNA VEZ, AL ABRIR. El apartado no la repite al final:
  cierra con la consecuencia y, si el siguiente depende de él, con el puente.
  Si dentro un argumento tiene una calificación distinta, se dice en su
  párrafo, antes de demostrarla.
- LO QUE QUEDA SIN MATERIA O CAE CON EL PRINCIPAL se nombra por su ordinal y
  se despacha con la medida única de la extensión.
"""


# LA COHERENCIA DE CADA ARGUMENTO CON EL SENTIDO DE SU PROBLEMA (familia v2;
# la misma regla que `plan_estudio.etiqueta_fuera` comprueba en el plan). Es
# una descripción: ni una frase que copiar.
_COHERENCIA = [
    "DENTRO DE ESE SENTIDO, CADA ARGUMENTO LLEVA SU PROPIA CALIFICACIÓN,",
    "razonada por lo que él mismo plantea, y coherente con la del problema: si",
    "el problema prospera, al menos uno de sus argumentos lo funda y los demás",
    "pueden ser infundados o inoperantes; si no prospera, ninguno queda fundado",
    "sin decir por qué no alcanza —ése es el fundado pero insuficiente—. La",
    "calificación distinta de un argumento se dice en su párrafo, antes de",
    "demostrarla, y no cambia la del problema."]


def forma_del_criterio(formato: str, q1: str = "concepto de violación",
                       parte: str = "la parte quejosa",
                       variante: str = "v1") -> list:
    """Las líneas del bloque del criterio que dicen cómo se abre cada problema.

    `q1` y `parte` salen del vocabulario del tipo de asunto: un ejemplo con
    «la quejosa» escrito a mano acabó firmado en una revisión fiscal, donde
    quien recurre es la autoridad."""
    _v2 = str(variante or "").strip().lower() == "v2"
    if _v2 and normalizar(formato) != MODERNA:
        # EN LA v2 DE LA ESTÁNDAR el criterio ya no reparte «{q1} por {q1}»:
        # dice la calificación y la razón, y el estudio agrupa por la
        # consideración atacada (ver `_forma_del_estudio_v2`).
        # LA CALIFICACIÓN POR ARGUMENTO (p2-congruencia, 26-sep-2026): «cada
        # {q1} recibe la calificación que le da el problema» hacía heredar al
        # argumento la del problema —la reconvención del ADC 642/2024 salió
        # «fundada» en la v4—. El sentido del problema no se toca; el del
        # argumento, dentro de él, se razona (`_COHERENCIA`).
        return [
            "LOS PROBLEMAS DE ABAJO NO SE ESCRIBEN: son el CRITERIO con que se",
            "califica cada planteamiento. El SENTIDO de cada problema lo fijó el",
            "secretario y no se cambia: cada apartado abre con la calificación que",
            "le da el problema que lo resuelve (la línea CUBRE dice cuáles son), y",
            "la RAZÓN DEL SECRETARIO es la que decide: los que caen por la misma",
            "razón contra la misma consideración van en un solo apartado.",
            ""] + _COHERENCIA + [""]
    if normalizar(formato) == MODERNA:
        return [
            "CADA PROBLEMA ABRE CON SU PREGUNTA, LITERAL Y EN SU PROPIA LÍNEA.",
            "El problema va escrito abajo como pregunta: cópiala tal cual como",
            "rótulo del apartado, numerada, y contéstala en el párrafo siguiente,",
            "que empieza por la respuesta y nombra el planteamiento que contesta.",
            "",
            "Y NO ESCRIBAS LA PREGUNTA DOS VECES ni la parafrasees en el cuerpo:",
            "el apartado ya la lleva, y repetirla es lo que hace kilométricas a",
            "las sentencias. Cada apartado dice lo suyo y sólo lo suyo.", ""] + (
            ["EL SENTIDO de cada problema lo fijó el secretario y no se cambia:",
             "la respuesta abre con él."] + _COHERENCIA + [""] if _v2 else [])
    return [
        "LOS PROBLEMAS DE ABAJO NO SE ESCRIBEN: son el CRITERIO con que se",
        f"califica cada planteamiento. El estudio va {q1} por {q1} —«Sobre",
        f"el primer {q1}, en el que {parte} sostiene… Se considera infundado.",
        "Lo anterior…»— y cada uno recibe la calificación que le da el problema",
        "que lo resuelve (la línea CUBRE dice cuáles son).", ""]


# ═══════════════════════════════════════════════════════════════════════════
# QUÉ PLANTEAMIENTOS CUBRE CADA CRITERIO
# ═══════════════════════════════════════════════════════════════════════════
def _norm(x: str) -> str:
    t = unicodedata.normalize("NFKD", str(x or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", t)


def _palabras(x: str) -> set:
    return {w for w in _norm(x).split() if len(w) > 4}


def cubre_de(criterio, problemas: list) -> list:
    """Los números de planteamiento del reparto de la fase 3 para ese criterio.

    El criterio lleva la pregunta; la fase 3 la guardó con su «cubre». Se casan
    por palabras con peso, porque el secretario pudo retocar la pregunta con el
    lápiz. Sin pareja segura, lista vacía: mejor no decir nada que atribuir un
    concepto a otro problema."""
    k = _palabras(getattr(criterio, "problema", "") or "")
    if not k:
        return []
    mejor, puntos = None, 0.0
    for p in (problemas or []):
        if not isinstance(p, dict):
            continue
        kp = _palabras(p.get("pregunta") or "")
        if not kp:
            continue
        s = len(k & kp) / max(1, min(len(k), len(kp)))
        if s > puntos:
            mejor, puntos = p, s
    if mejor is None or puntos < 0.6:
        return []
    fuera = []
    for x in (mejor.get("cubre") or []):
        try:
            fuera.append(int(x))
        except Exception:
            pass
    return sorted(set(fuera))


_ORDINAL_TXT = {1: "primero", 2: "segundo", 3: "tercero", 4: "cuarto",
                5: "quinto", 6: "sexto", 7: "séptimo", 8: "octavo",
                9: "noveno", 10: "décimo"}


def ordinal(n: int) -> str:
    """Pospuesto: «los conceptos primero y tercero»."""
    return _ORDINAL_TXT.get(int(n), f"{n}.º")


def ordinal_antepuesto(n: int) -> str:
    """Antepuesto, con su apócope: «el primer concepto», «el tercer agravio»."""
    return {1: "primer", 3: "tercer"}.get(int(n)) or ordinal(n)


def cubre_en_texto(numeros: list, q1: str) -> str:
    """«el primer concepto de violación» / «los conceptos de violación
    primero y tercero»."""
    ns = [int(x) for x in (numeros or [])]
    if not ns:
        return ""
    if len(ns) == 1:
        return f"el {ordinal_antepuesto(ns[0])} {q1}"
    plural = re.sub(r"^(\w+)", lambda m: m.group(1) + "s", q1, count=1)
    lista = [ordinal(x) for x in ns]
    return f"los {plural} {', '.join(lista[:-1])} y {lista[-1]}"


# ═══════════════════════════════════════════════════════════════════════════
# ¿CADA PLANTEAMIENTO RECIBIÓ RESPUESTA?
# ═══════════════════════════════════════════════════════════════════════════
# Se busca el ORDINAL junto a «concepto» o «agravio», que es como los nombran
# los engroses reales. Medido en los 14 estudios del banco Kingston que traen
# escrito y estudio: «En su primer concepto de violación…» (8), «Primer
# concepto» (10), «Análisis del tercer concepto» (3), «Respecto al segundo y
# tercer conceptos» y «Finalmente, en el cuarto…». Un ordinal a menos de
# `_VENTANA` caracteres de la palabra, dentro de la misma frase, cuenta.
_RX_ORD = {
    1: r"primer[oa]?|1[°º]", 2: r"segund[oa]s?|2[°º]", 3: r"tercer[oa]?s?|3[°º]",
    4: r"cuart[oa]s?|4[°º]", 5: r"quint[oa]s?|5[°º]", 6: r"sext[oa]s?|6[°º]",
    7: r"s[ée]ptim[oa]s?|7[°º]", 8: r"octav[oa]s?|8[°º]", 9: r"noven[oa]s?|9[°º]",
    10: r"d[ée]cim[oa]s?|10[°º]", 11: r"und[ée]cim[oa]s?|d[ée]cimo\s*primer[oa]?",
    12: r"duod[ée]cim[oa]s?|d[ée]cimo\s*segund[oa]",
}
# «motivo» NO: «por los fundamentos y motivos… considerando sexto» del
# ADA 704/2022 contaba como un sexto planteamiento.
_RX_PALABRA = re.compile(r"\b(?:concepto|agravio)s?\b", re.I)
_VENTANA = 90
# Estudio conjunto de TODOS: basta que se declare.
_RX_TODOS_JUNTOS = re.compile(
    r"(?:analiza|estudia|examina)[a-z]*\s+(?:de\s+manera\s+)?(?:conjunta|en\s+su\s+conjunto)"
    r"|(?:conceptos|agravios)[^.]{0,60}(?:en\s+su\s+conjunto|conjuntamente)"
    r"|(?:[úu]nico|todos\s+los)\s+(?:concepto|agravio)", re.I)


def _frase_alrededor(texto: str, i: int, j: int) -> str:
    ini = max(texto.rfind(".", 0, i), texto.rfind("\n", 0, i), i - _VENTANA)
    fin_p = texto.find(".", j)
    fin_n = texto.find("\n", j)
    fin = min(x for x in (fin_p if fin_p >= 0 else len(texto),
                          fin_n if fin_n >= 0 else len(texto), j + _VENTANA))
    return texto[max(0, ini):fin]


def nombrados(texto: str) -> set:
    """Los números de planteamiento que el texto nombra por su ordinal."""
    t = texto or ""
    vistos = set()
    for m in _RX_PALABRA.finditer(t):
        tramo = _frase_alrededor(t, m.start(), m.end())
        for n, rx in _RX_ORD.items():
            if re.search(r"\b(?:" + rx + r")\b", tramo, re.I):
                vistos.add(n)
    return vistos


def sin_contestar(estudio: str, n: int) -> list:
    """Los planteamientos 1..n que el estudio no nombra.

    Con n < 2 no hay nada que comprobar por ordinal: el único se contesta o no,
    y eso lo mide ya `calidad_estudio.exhaustividad`. Si el estudio declara que
    los analiza todos conjuntamente, tampoco: eso es legítimo."""
    try:
        n = int(n or 0)
    except Exception:
        n = 0
    if n < 2 or not (estudio or "").strip():
        return []
    if _RX_TODOS_JUNTOS.search(estudio):
        return []
    hay = {x for x in nombrados(estudio) if x <= n}
    # SÓLO SI EL ESTUDIO NOMBRA POR ORDINAL. Los dos formatos lo exigen, y un
    # estudio que no nombra ninguno contesta de otra manera —«los anteriores
    # conceptos de violación son fundados», ADA 704/2022—: acusarlo sería
    # acusar un engrose firmado. Lo que se busca es el hueco en una serie
    # que sí se numera.
    if not hay:
        return []
    faltan = [i for i in range(1, n + 1) if i not in hay]
    # UNA RESPUESTA SIN ORDINAL TAMBIÉN ES RESPUESTA. El ADC 640/2024 contesta
    # su primer concepto como «resulta inoperante el concepto de violación
    # del quejoso en cuanto a la supuesta ilegalidad del dictamen» y sólo al
    # segundo lo nombra. Cada frase así —el planteamiento en singular, con su
    # calificación y sin ordinal— cubre a uno de los que faltan. «Este
    # planteamiento» no cuenta: no dice de cuál se habla (93/2026 v5).
    if faltan and len(_respuestas_sin_ordinal(estudio)) >= len(faltan):
        return []
    return faltan


_RX_SINGULAR = re.compile(
    r"\b(?:el|este|dicho|ese|aquel)\s+(?:concepto\s+de\s+violaci[óo]n|agravio)\b", re.I)
_RX_CALIFICA_FRASE = re.compile(
    r"\b(?:fundad|infundad|inoperant|inefica[cz]|inatendibl|innecesari|desestim|"
    r"sin\s+materia)", re.I)


def _respuestas_sin_ordinal(texto: str) -> list:
    fuera = []
    for frase in re.split(r"(?<=[.;])\s+|\n+", texto or ""):
        if (_RX_SINGULAR.search(frase) and _RX_CALIFICA_FRASE.search(frase)
                and not any(re.search(r"\b(?:" + rx + r")\b", frase, re.I)
                            for rx in _RX_ORD.values())):
            fuera.append(frase)
    return fuera


def aviso_sin_contestar(faltan: list, es_recurso: bool) -> str:
    if not faltan:
        return ""
    q1 = "agravio" if es_recurso else "concepto de violación"
    que = cubre_en_texto(faltan, q1).upper()
    return (f"EL ESTUDIO NO NOMBRA {que}: no se ve "
            f"dónde se le contesta. Si se estudió junto con otro, que el "
            f"apartado lo diga por su ordinal; si no se estudió, es omisión y "
            f"hay que añadirlo antes de listar.")


# ═══════════════════════════════════════════════════════════════════════════
# LA SÍNTESIS DE LA VERSIÓN MODERNA
# ═══════════════════════════════════════════════════════════════════════════
def _antecedentes_en_prosa() -> bool:
    """¿Rige la bandera `antecedentes_en_prosa` en esta petición?"""
    try:
        import contexto_taller as _ct_ant
        return _ct_ant.rediseno("antecedentes_en_prosa")
    except Exception:
        return False


def _remision_antecedentes() -> str:
    """La regla de no remitir a los antecedentes por número, en el estudio moderno.

    SIN ENUMERAR (2-oct-2026, David: «debemos quitar la enumeración de
    antecedentes»). La razón de siempre —«se resumen de nuevo y su numeración
    cambia»— suponía que había números; con la bandera ya no los hay, y la
    regla se queda con su razón nueva y sin el ejemplo de la remisión, que un
    ejemplo en el prompt se copia. Sin la bandera, el texto de siempre.
    """
    if _antecedentes_en_prosa():
        return ("- NO REMITAS A LOS ANTECEDENTES POR UN NÚMERO: no llevan ninguno, van en\n"
                "  prosa. Nómbralos por su fecha o su contenido.")
    return ("- NO REMITAS A LOS ANTECEDENTES POR SU NÚMERO («como se dijo en el antecedente\n"
            "  7»): en esta versión se resumen de nuevo y su numeración cambia. Nómbralos\n"
            "  por su fecha o su contenido.")


def prompt_sintesis(antecedentes: str, resumen_acto: str, resumen_conceptos: str,
                    problemas: list, criterios: list, organo: str, q: str,
                    n_planteamientos: int = 0) -> str:
    combatido = []
    for i, p in enumerate(problemas or [], 1):
        if isinstance(p, dict):
            linea = f"{i}. {p.get('pregunta', '')}"
            if p.get("resolvio"):
                linea += f"\n   lo que resolvió {organo} sobre esto: {p['resolvio']}"
            combatido.append(linea)
    sentidos = "\n".join(f"· {getattr(c, 'problema', '')} → {str(getattr(c, 'sentido', '')).upper()}"
                         for c in (criterios or []))
    n_txt = (f"El escrito trae {n_planteamientos} {q}: los {n_planteamientos} "
             f"tienen que aparecer, cada uno con su ordinal." if n_planteamientos >= 2 else "")
    # EN PROSA, SIN NÚMEROS (2-oct-2026, bandera `antecedentes_en_prosa`). La
    # numeración de la moderna la ponía el MODELO, por la regla 4 y por el
    # ejemplo del JSON («1. …», «2. …»), que se copia tal cual. Con la bandera,
    # la regla pide prosa y el ejemplo es de párrafos, como los otros dos.
    if _antecedentes_en_prosa():
        _regla_ante = ("4. ANTECEDENTES: los hechos que hacen falta para entender la solución,\n"
                       "   en prosa encadenada y sin numerar —ni números de orden, ni viñetas, ni\n"
                       "   incisos—, un párrafo por etapa, con sus fechas; los trámites que no\n"
                       "   inciden en nada se quitan. El último hecho sigue siendo en qué paró.")
        _molde_ante = '["párrafo", "párrafo"]'
    else:
        _regla_ante = ("4. ANTECEDENTES: los hechos que hacen falta para entender la solución,\n"
                       "   numerados, con sus fechas; los trámites que no inciden en nada se quitan.")
        _molde_ante = '["1. …", "2. …"]'
    return f"""Eres el secretario de un Tribunal Colegiado de Circuito. Tienes tres
apartados ya escritos de un proyecto y hay que CONDENSARLOS para la versión
moderna de la sentencia: la que atiende a fondo el problema jurídico central y
prescinde de lo irrelevante.

LO QUE ES MATERIA DE ESTUDIO (los problemas jurídicos del asunto):
{chr(10).join(combatido) or "(no se fijaron)"}

EL SENTIDO QUE SE DECIDIÓ:
{sentidos or "(sin criterio)"}

REGLAS, en este orden de importancia:
1. NO INVENTES NADA. Sólo condensas: cada dato —fecha, cantidad, nombre,
   número de expediente, artículo, tesis— sale del texto de abajo y se copia
   tal cual. Si dudas, quítalo antes que reescribirlo.
2. {q.upper()}: TODOS, sin excepción, cada uno con su ordinal en el orden del
   escrito —«En el primer…», «En el segundo…»—, en UN párrafo cada uno (dos
   si el que decide el asunto lo necesita), diciendo lo esencial de lo que
   sostiene. Lo que alega y no incide en ningún problema se resume en media
   frase; las tesis que invoca se nombran sólo si son la base de su
   planteamiento. {n_txt}
3. LO QUE RESOLVIÓ {organo.upper()}: sólo las consideraciones que se combaten
   —las que tocan los problemas de arriba— con el detalle que haga falta para
   entender la respuesta. Lo que resolvió y NADIE combate se dice en UNA frase
   al final («En lo demás, … consideraciones que no se controvierten.») o se
   omite si no importa para decidir.
{_regla_ante}
5. Registro de sentencia, tercera persona, sin viñetas ni Markdown, sin
   rótulos, sin comentarios sobre lo que estás haciendo.
6. Cada apartado condensado, a lo sumo la MITAD de largo que el original.

═══ ANTECEDENTES ═══
{antecedentes}

═══ LO QUE RESOLVIÓ {organo.upper()} ═══
{resumen_acto}

═══ {q.upper()} ═══
{resumen_conceptos}

Devuelve JSON y nada más:
{{"antecedentes": {_molde_ante},
  "resolvio": ["párrafo", "párrafo"],
  "planteamientos": ["párrafo", "párrafo"]}}"""


# Por debajo de esto un apartado no dice nada, sea cual sea el original.
MINIMO_PALABRAS = 40


def _lista(x) -> list:
    if isinstance(x, str):
        x = [p for p in x.split("\n")]
    return [str(p).strip() for p in (x or []) if str(p).strip()]


def _pal(xs: list) -> int:
    return sum(len(str(p).split()) for p in xs)


def validar_sintesis(crudo: dict, antecedentes: list, acto: list, conceptos: list,
                     n_planteamientos: int = 0) -> tuple:
    """Lo que se queda de la síntesis y por qué se rechaza lo que no.

    Cada apartado se decide solo: condensado si es más corto y no pierde nada;
    el original si no. Devuelve ({antecedentes, acto, conceptos}, notas)."""
    notas = []
    fuera = {"antecedentes": antecedentes, "acto": acto, "conceptos": conceptos}
    if not isinstance(crudo, dict):
        return fuera, ["la síntesis no llegó: van los resúmenes completos"]
    pares = (("antecedentes", "antecedentes", antecedentes),
             ("acto", "resolvio", acto),
             ("conceptos", "planteamientos", conceptos))
    for clave, campo, original in pares:
        nuevo = _lista(crudo.get(campo))
        if not nuevo or not original:
            continue
        if _pal(nuevo) >= 0.9 * _pal(original):
            notas.append(f"{clave}: la síntesis no acortó; se queda el original")
            continue
        # EL SUELO ES ABSOLUTO, NO UNA PROPORCIÓN. Era «menos del 15 % del
        # original» y rechazó justo la síntesis buena: en el ADC 93/2026 casi
        # toda la sentencia —competencia, firma, nulidad— no la combate nadie,
        # y condensarla a una décima parte es lo que David pidió («resuma o
        # prescinda de resumir consideraciones que no serán materia de
        # estudio»). v10 salió con las 1,530 palabras de siempre por eso.
        if _pal(nuevo) < MINIMO_PALABRAS:
            notas.append(f"{clave}: la síntesis se quedó en casi nada; se queda el original")
            continue
        if clave == "conceptos" and int(n_planteamientos or 0) >= 2:
            faltan = [i for i in range(1, int(n_planteamientos) + 1)
                      if i not in nombrados("\n".join(nuevo))]
            if faltan:
                notas.append(f"conceptos: la síntesis perdió el {', '.join(map(str, faltan))}; "
                             f"se queda el original")
                continue
        fuera[clave] = nuevo
    return fuera, notas


def leer_json(crudo: str) -> dict:
    m = re.search(r"\{.*\}", crudo or "", re.S)
    try:
        return json.loads(m.group(0) if m else (crudo or ""))
    except Exception:
        return {}
