# -*- coding: utf-8 -*-
"""EL SUPERVISOR DEL PROYECTO: una segunda lectura que corrige, no que avisa.

POR QUÉ EXISTE. David, 2-oct-2026: «el arreglo definitivo sería que el motor
supere al secretario con un agente supervisor del proceso… un LLM o agente
que, sin tanto costo, revise el proyecto y ajuste sus errores (cuando meto el
proyecto a gemini 3.8 flash razonamiento extendido detecta errores y propone
mejorar la redacción, que sigue siendo a veces muy extensa)». Y el punto 6:
«nunca dar por hecho que lo que se dice en los recursos o conceptos de
violación es cierto».

QUÉ HABÍA. Del estudio sólo CORREGÍAN texto los arreglos sin modelo (rótulos,
marcas, congruencia de aperturas, litis) y una llamada con modelo, la
reparación dirigida de `exhaustivo.reparar`, que sólo AÑADE lo que falta.
Todo lo demás —linter, fechas, calidad, duplicación, barrido— SÓLO AVISA, y
el secretario recibía 19 avisos sobre defectos que nadie arreglaba.

QUÉ HACE. Lee el ESTUDIO ya escrito —con sus marcas ⟦…⟧— junto con lo que lo
ancla: el criterio del secretario (inmutable), lo que la resolución reclamada
o recurrida tuvo por acreditado, lo que alega la parte, el catálogo cerrado de
registros y normas del material, y los hallazgos de los controles
deterministas que hoy sólo avisan. Devuelve PARCHES POR PÁRRAFO NUMERADO
(reemplazar, condensar, eliminar). El código los pasa por guardas y aplica
sólo los que pasan; los demás se cuentan en «descartadas». Si la llamada falla
o vence, el estudio queda COMO ESTABA. Después, `_terminar` vuelve a correr
todos sus controles sobre el texto corregido: el supervisor no se valida a sí
mismo.

EL MOTOR, OTRA FAMILIA. El estudio lo escribe gpt-5.6-luna; un revisor de la
misma familia comparte sus puntos ciegos. Lo que David vio funcionar a mano es
Gemini Flash con razonamiento extendido: `gemini-3.8-flash` existe en la API
de AI Studio (verificado el 2-oct-2026 con models.list: 1M de entrada, 65k de
salida, thinking). Se llama por el SDK google-genai con GEMINI_API_KEY, como
el OCR del taller. Si no hay clave o el SDK no abre, cae a gpt-5.6-luna por el
`cliente` del resolver (el mismo chat_client): un supervisor de la misma
familia es peor que uno de otra, pero mejor que ninguno, y el coste de luna es
conocido. Si Gemini VENCE no se reintenta con luna: sumaría otro tope entero a
un resolver que ya roza los cinco minutos.

LAS GUARDAS (todas por código; un parche que falle una se descarta):
  · no cambia la dirección de ninguna calificación (`congruencia.direcciones`)
    ni el conjunto de calificativos (infundado ≠ inoperante; 3-oct-2026),
    salvo la incongruencia que endereza (`_endereza`: por el plan, o un
    «fundado» dentro de un criterio todo «contra»);
  · no cambia el desenlace del párrafo (conceder/negar, revocar/confirmar/
    modificar, reponer, «para efectos», sobreseer) y, si la suma de parches
    cambiara lo que el resolutivo lee del estudio entero, se revierte todo;
  · no quita la respuesta a un argumento marcado (`marcas.rastros`);
  · no quita ni cambia los registros del párrafo; no introduce registros ni
    números de tesis/expediente («82/2013») fuera del material ∪ el párrafo
    original ∪ las fuentes del asunto, ni registros de tesis sin vigencia;
    un rubro nuevo sólo si es, entero, el de una tesis del material;
  · no introduce artículos fuera del párrafo original ∪ las normas del material;
  · no introduce fechas ni transcripciones entrecomilladas que no consten;
  · conserva las marcas ⟦…⟧ (se le enseña el párrafo SIN ellas y el código las
    vuelve a poner delante: el mapa de marcas cuenta párrafos, no posiciones);
  · el párrafo resultante tiene ≥ 6 palabras (el compositor tira los más
    cortos, documento_generado.py ~3808);
  · «eliminar» sólo en párrafos sin marcas, sin citas y sin calificación;
  · no alarga un párrafo más de un 20% salvo error jurídico o incongruencia;
  · no toca los rubros transcritos, los encabezados ni los rótulos internos
    del guion (párrafos FIJOS), y un párrafo que lleva un rubro entre
    comillas lo conserva literal;
  · no pierde el ordinal del concepto que contesta, ni el número de la orden
    de efectos;
  · no escribe frases de herramienta («material proporcionado»…).

DETRÁS DE LA BANDERA `supervisor_proyecto` (contexto_taller.BANDERAS_REDISENO;
«casa» por omisión). Apagada, nadie llama a este módulo: el resolver, el meta
y los prompts quedan idénticos.

LOS ANTECEDENTES (3-oct-2026). David: «el revisor agente (gemini) concentra
su capacidad en antecedentes y estudio de fondo» —lo procesal sale ya por
construcción (resultandos_por_tipo)—. Con la bandera `procedencia_por_tipo`
la MISMA llamada revisa también el considerando de antecedentes: sus párrafos
van numerados aparte («A1…», nunca mezclados con los «P» del estudio) y sus
parches pasan sus propias guardas (`guardas_antecedente`): ninguna fecha,
número, nombre ni artículo que no esté en el párrafo o en las fuentes (el
acto, el escrito, las constancias); no cambia lo que cada instancia resolvió
ni lo que tuvo por acreditado; no alarga más de un 20%; seis palabras como
mínimo; «eliminar» sólo la repetición pura. El trámite en este tribunal no se
le enseña nunca. Los antecedentes corregidos llegan al proyecto por
`material.sintesis`, que es de donde `_terminar` los toma.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
import unicodedata

# ═══ EL MOTOR Y SUS TOPES ═══════════════════════════════════════════════════
SUPERVISOR_MODELO = os.getenv("SUPERVISOR_MODELO", "gemini-3.8-flash")
# EL RESPALDO, por el chat_client del resolver (OpenAI directo). Razonamiento
# alto: la regla de David de no bajar el razonamiento de lo que escribe el
# estudio vale también para lo que lo reescribe.
SUPERVISOR_RESPALDO = os.getenv("SUPERVISOR_RESPALDO", "gpt-5.6-luna")
SUPERVISOR_RESPALDO_ESFUERZO = os.getenv("SUPERVISOR_RESPALDO_ESFUERZO", "high")
# EL RAZONAMIENTO, EXTENDIDO PERO CON TECHO (medido el 2-oct-2026 sobre el AR
# 631/2025, 12,430 tokens de entrada). Con thinking_level «high»,
# gemini-3.8-flash piensa HASTA AGOTAR la salida: 38,401 de 40,000 tokens en
# 137 s con un techo, 62,914 de 65,536 en 213 s con el otro, y en los dos la
# respuesta se cortó a media lista (se rescató lo entero). Los dos dieron las
# mismas correcciones en los párrafos que alcanzaron: pensar el doble no
# corrigió más, sólo llegó menos lejos. Un presupuesto fijo (thinking_budget,
# que el modelo acepta: comprobado el mismo día) deja pensar ~24 mil tokens —
# más que un «medium»— y reserva la salida para los parches: ~95 s a los
# ~290 tok/s medidos. «high» o «medium» → thinking_level; un número →
# thinking_budget. No es bajar el razonamiento de nadie: el supervisor es nuevo
# y su techo es el que lo deja terminar.
#
# PERO EL PRESUPUESTO NO SE RESPETA (medido el mismo día en el resolver real):
# con thinking_budget=24576 el modelo pensó 46,270 tokens, se comió los 150 s
# del tope y en el ADA 103/2025 entregó 0 correcciones. Sobre el mismo estudio
# del 631: «medium» 56 s y 13 correcciones (2,897 → 2,473 palabras), 8192 53 s
# y 8, «low» 4 s y 3, «high» 137-213 s y 16. «medium» corrige casi lo mismo que
# «high» en la cuarta parte del tiempo, y el nivel sí lo respeta.
SUPERVISOR_RAZONAMIENTO = os.getenv("SUPERVISOR_RAZONAMIENTO", "medium")
# TEMPERATURA BAJA: corrige, no crea. Google recomienda 1.0 para la familia 3
# (por debajo, dice, puede entrar en bucles); si se ve que se repite, se sube
# desde Render sin desplegar.
SUPERVISOR_TEMPERATURA = float(os.getenv("SUPERVISOR_TEMPERATURA", "0.3"))
# EL TOPE Y LA SALIDA. La salida va al máximo del modelo (65,536; el
# razonamiento cuenta dentro) para que los parches nunca se queden sin sitio.
# El tope, 120 s: el doble de lo medido con «medium» (56 s). Con «high» hay
# que subirlo a 240 (se midieron 213 s).
SUPERVISOR_TOPE_S = float(os.getenv("SUPERVISOR_TOPE_S", "120"))
SUPERVISOR_MAX_SALIDA = int(os.getenv("SUPERVISOR_MAX_SALIDA", "65536"))
# EL INTERRUPTOR DE EMERGENCIA, aparte de la bandera: si un día el proveedor
# falla en cadena, se apaga para todos desde Render sin tocar las banderas.
SUPERVISOR_ACTIVO = os.getenv("SUPERVISOR_PROYECTO_ACTIVO", "1").strip().lower() not in ("0", "no", "false")
# EL PRECIO, si se sabe (USD por millón). En el código no hay precio publicado
# de gemini-3.8-flash; sin él, el informe no inventa un coste.
_PRECIO_ENTRADA = os.getenv("SUPERVISOR_PRECIO_ENTRADA_USD", "")
_PRECIO_SALIDA = os.getenv("SUPERVISOR_PRECIO_SALIDA_USD", "")
# Tras un fallo rápido de Gemini, el respaldo sólo se intenta si queda tiempo
# para una llamada de verdad.
_MIN_RESPALDO_S = 45.0

# LA EXTENSIÓN QUE DAVID ACEPTA (p2-congruencia, 26-sep-2026): 2,000 a 2,500
# palabras de estudio. La brevedad no es objetivo si cuesta exhaustividad.
TECHO_PALABRAS = 2500

TIPOS = ("error_juridico", "incongruencia", "hecho_no_acreditado", "cita",
         "repeticion", "extension", "redaccion")
ACCIONES = ("reemplazar", "condensar", "eliminar")
_TIPOS_QUE_ALARGAN = ("error_juridico", "incongruencia")
MIN_PALABRAS = 6
CRECIMIENTO_MAX = 1.20
MAX_PARCHES = 60
# Fracción de párrafos que se puede eliminar en una pasada: más es reescribir.
MAX_ELIMINAR = 0.25
# Si lo aplicado dejara el estudio por debajo de esta fracción, se revierte
# todo: un supervisor no se come medio estudio.
PISO_PALABRAS = 0.60


def activo() -> bool:
    """¿Rige la bandera del supervisor en esta petición? El interruptor de
    emergencia lo mira `en_el_resolver`, que entonces deja el informe en
    «apagado» (contrato D) sin llamar a nadie."""
    try:
        import contexto_taller as _ct
        return _ct.rediseno("supervisor_proyecto")
    except Exception:
        return False


def rige_antecedentes() -> bool:
    """¿Revisa también los antecedentes? Bandera `procedencia_por_tipo`.

    SÓLO SI LA BANDERA ESTÁ DADA DE ALTA (3-oct-2026): `rediseno` con un nombre
    que no está en BANDERAS_REDISENO cae a «casa» y encendería esto para las
    cuentas de prueba antes de que el cableado registre la bandera. Mientras
    no exista, False."""
    try:
        import contexto_taller as _ct
        if "procedencia_por_tipo" not in (getattr(_ct, "BANDERAS_REDISENO", None) or {}):
            return False
        _f = getattr(_ct, "rige", None) or _ct.rediseno
        return bool(_f("procedencia_por_tipo"))
    except Exception:
        return False


# ═══ LAS FRASES DE HERRAMIENTA ══════════════════════════════════════════════
# Lo que un tribunal no escribe: el estudio del AR 631/2025 dijo varias veces
# «no obra entre las constancias remitidas para este estudio» y «no obra en el
# material proporcionado». Es el vocabulario del encargo que recibe el modelo,
# no el de una sentencia. Lista mínima y literal, para no acusar prosa buena;
# el paquete «preguntas» trae la suya y se fusionarán.
FRASES_HERRAMIENTA = (
    r"material\s+(?:proporcionad|disponible|aportad|remitid|suministrad|recibid)\w*",
    # «en el material» suelto; seguido de su adjetivo lo caza la de arriba.
    r"\ben\s+el\s+material\b(?!\s+(?:probatori|de\s+(?:prueba|construcci)|proporcionad|disponible|"
    r"aportad|remitid|suministrad|recibid))",
    r"constancias\s+(?:remitidas|proporcionadas|disponibles|aportadas)\s+para\s+(?:este|el\s+presente)\s+estudio",
    r"remitid\w+\s+para\s+(?:este|el\s+presente)\s+estudio",
    r"(?:aportad|remitid|proporcionad|suministrad)\w*\s+al\s+material\b",
    r"documentos?\s+(?:proporcionad|suministrad)\w*",
    r"informaci[óo]n\s+(?:proporcionad|disponible|suministrad)\w*",
    r"\binsumos?\b",
    r"\bel\s+usuario\b",
    r"(?:seg[úu]n|conforme\s+a)\s+(?:el|los)\s+(?:contexto|datos)\s+(?:proporcionad|recibid)\w*",
)
_RX_HERRAMIENTA = re.compile("|".join(f"(?:{p})" for p in FRASES_HERRAMIENTA), re.I)


def frases_de_herramienta(texto: str) -> list:
    """[(frase, contexto)] de cada frase de herramienta del texto."""
    t = texto or ""
    fuera = []
    for m in _RX_HERRAMIENTA.finditer(t):
        ctx = " ".join(t[max(0, m.start() - 60):m.end() + 50].split())
        fuera.append((m.group(0), ctx))
    return fuera


# ═══ LOS PÁRRAFOS NUMERADOS ═════════════════════════════════════════════════
_RX_ENCABEZADO = re.compile(
    r"^\s*(?:PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|S[ÉE]PTIMO|OCTAVO|NOVENO|D[ÉE]CIMO)\.\s*"
    r"(?:Estudio|Efectos)\b", re.I)
_RX_CITA_COMILLAS = re.compile(r"[«“\"]([^«»“”\"]{12,})[»”\"]")
_RX_REG = re.compile(r"\b(\d{6,7})\b")
_RX_NUM_ANIO = re.compile(r"\b(\d{1,5}/\d{4})\b")
_RX_ARTS = re.compile(
    r"\bart(?:[íi]culos?|s?\.)\s+((?:\d{1,4}\s*(?:º|°|o\.|bis|ter|quater)?\s*(?:,|y|e|-|–)?\s*)+)", re.I)
_RX_Y_NUM_DE_LEY = re.compile(
    r"\b(?:y|e)\s+(\d{1,4}(?:\s*(?:,|y|e)\s*\d{1,4})*)\s+(?:de|del)\s+(?:la\s+|el\s+)?"
    r"(?:constituci[óo]n|c[óo]digo|ley|reglamento|convenci[óo]n)", re.I)
_RX_PREFIJO_ORDEN = re.compile(r"^\s*(\d{1,2}\s*[.)]|[a-z]\))\s+")
_RX_ID_INTERNO = re.compile(r"\b(?:AD|[CASMU])\d{1,3}\.[a-z]{1,2}\b")


def _palabras(t: str) -> int:
    return len((t or "").split())


def _plano(t: str) -> str:
    y = unicodedata.normalize("NFKD", str(t or "").lower())
    y = "".join(c for c in y if not unicodedata.combining(c))
    return " ".join(re.sub(r"[«»“”\"'’‘]", " ", y).split())


def _es_rubro(t: str) -> bool:
    try:
        import congruencia as _cg
        return _cg._es_rubro(t)
    except Exception:
        letras = [c for c in (t or "")[:120] if c.isalpha()]
        return bool(letras) and sum(c.isupper() for c in letras) / len(letras) > 0.6


def _fijo(texto: str) -> str:
    """Por qué un párrafo no se toca («» si se puede corregir)."""
    t = (texto or "").strip()
    if _RX_ENCABEZADO.match(t):
        return "encabezado"
    if _palabras(t) < MIN_PALABRAS:
        return "rótulo"
    # LO QUE QUEDA FUERA DE LAS COMILLAS decide: un rubro o un precepto
    # transcrito solo en su párrafo es fijo; «Sirve de apoyo la tesis de
    # rubro «…»» se puede corregir, pero la guarda le exige el rubro literal.
    fuera_comillas = _RX_CITA_COMILLAS.sub(" ", t)
    if _palabras(fuera_comillas) < 3:
        return "rubro" if _es_rubro(t) else "transcripción"
    if _es_rubro(fuera_comillas):
        return "rubro"
    return ""


# LOS RÓTULOS DEL GUION (revisión adversarial, 3-oct-2026). El supervisor corre
# ANTES de `redactor_adelanto.limpiar_rotulos_del_guion`, que es quien lleva
# «DESVIACIONES DEL GUION: …» a ADVERTENCIAS y tira «APLICA C1.a», «APARTADO 2 ·»…
# Esos renglones no son prosa del estudio: si el supervisor los reescribe en
# prosa, el limpiador ya no los reconoce y la nota interna llega al .docx; si
# los elimina, el secretario pierde la explicación de la desviación. Son FIJOS.
# Los patrones se toman del limpiador mismo para no divergir; la copia de abajo
# sólo rige si no se puede importar (y una prueba vigila que sean iguales).
_RX_ROTULO_GUION_COPIA = re.compile(
    r"^[ \t>*#\-]*(?:"
    r"GUION DEL ESTUDIO\b"
    r"|APARTADO \d+ ·"
    r"|(?:APLICA|REMITE|DESARROLLA|RESIDUAL|NO SE ESTUDIA) (?:AD|[CAS])\d+\.[a-z]{1,2}\b"
    r"|EXPONE M\d+\b"
    r"|JERARQUÍA DEL PROBLEMA \d+"
    r"|(?:INNECESARIOS POR SUFICIENCIA|CAEN POR DERIVAR) (?:AD|[CAS])\d+\.[a-z]{1,2}\b"
    r")")
_RX_DESVIACIONES_COPIA = re.compile(r"^[ \t>*#\-]*DESVIACIONES DEL GUION\b")


def _rotulo_del_guion(*renglones) -> bool:
    """¿Alguno de los renglones empieza por un rótulo interno del guion?"""
    try:
        import redactor_adelanto as _ra
        rxs = (_ra._RX_ROTULO_GUION, _ra._RX_DESVIACIONES)
    except Exception:
        rxs = (_RX_ROTULO_GUION_COPIA, _RX_DESVIACIONES_COPIA)
    return any(rx.match(r or "") for r in renglones for rx in rxs)


def bloques(estudio: str) -> list:
    """Los párrafos del estudio como los cuenta el mapa de marcas: un renglón
    no vacío con texto es un párrafo; una marca sola en su renglón vale para
    el párrafo siguiente. [{n, inicio, linea, ids, marcas, texto, fijo}]."""
    import marcas as _mc
    lineas = (estudio or "").split("\n")
    fuera, pend_ids, ini_pend = [], [], None
    for k, ln in enumerate(lineas):
        if not ln.strip():
            continue
        ms = _mc.marcas_en(ln)
        limpio = _mc.sin_marcas(ln).strip()
        if not limpio:
            if ms:
                pend_ids.extend(x for _, _, v in ms for x in v)
                ini_pend = k if ini_pend is None else ini_pend
            continue
        resto = ln
        for a, b, _ in sorted(ms, reverse=True):
            resto = resto[:a] + resto[b:]
        fijo = "rótulo del guion" if _rotulo_del_guion(ln, limpio) else _fijo(limpio)
        # UNA MARCA A MEDIAS no se puede reponer intacta: el párrafo no se toca.
        if not fijo and ("⟦" in resto or "⟧" in resto):
            fijo = "marca a medias"
        fuera.append({"n": len(fuera) + 1, "inicio": ini_pend if ini_pend is not None else k,
                      "linea": k, "ids": pend_ids + [x for _, _, v in ms for x in v],
                      "marcas": [ln[a:b] for a, b, _ in ms], "texto": limpio, "fijo": fijo})
        pend_ids, ini_pend = [], None
    return fuera


# ═══ LAS CITAS DE UN TEXTO ══════════════════════════════════════════════════
def registros_de(t: str) -> set:
    return set(_RX_REG.findall(t or ""))


def numeros_con_anio(t: str) -> set:
    return set(_RX_NUM_ANIO.findall(t or ""))


def articulos_de(t: str) -> set:
    """Los números de artículo citados («artículos 49, 57 y 279», «y 2284 y
    2294 del Código Civil»). Sólo números: la guarda compara qué se cita, y
    nombrar mal la ley lo caza después `litis.sanear` y el barrido."""
    fuera = set()
    for m in _RX_ARTS.finditer(t or ""):
        fuera |= set(re.findall(r"\d{1,4}", m.group(1)))
    for m in _RX_Y_NUM_DE_LEY.finditer(t or ""):
        fuera |= set(re.findall(r"\d{1,4}", m.group(1)))
    return fuera


def _rubros_entre_comillas(t: str) -> list:
    return [" ".join(m.group(1).split()) for m in _RX_CITA_COMILLAS.finditer(t or "")
            if _es_rubro(m.group(1))]


# LO QUE SÓLO AFIRMA LA PARTE (punto 6 de David). Medido en la prueba real
# del AR 631/2025: al condensar «En el escrito de agravios se reproduce que la
# primera rescindió el contrato…», el supervisor dejó «la primera rescindió el
# contrato…» —el hecho de la parte, dado por cierto—. Si el párrafo atribuía
# algo a alguien, el texto nuevo tiene que seguir atribuyéndolo.
_RX_ATRIBUCION_EXTRA = re.compile(
    r"\bse\s+reproduce\b|\bseg[úu]n\s+(?:la|el|lo|los|las|su|sus)\b|\ba\s+decir\s+de\b|"
    r"\bescrito\s+de\s+(?:agravios|demanda|expresi[óo]n)|"
    r"\ben\s+(?:sus|su|los|el|la)\s+(?:agravios?|conceptos?\s+de\s+violaci[óo]n|demanda|recurso)\b|"
    r"\bla\s+(?:afirmaci[óo]n|alegaci[óo]n|aseveraci[óo]n)\b|\baserto\b", re.I)
# LOS VERBOS QUE SÓLO DICE UNA PARTE cuentan siempre; los que también usa el
# tribunal («sostiene», «refiere», «pretende», «considera que»…) sólo con la
# parte nombrada justo antes. Calibrado en la segunda corrida del 631: «la
# razón que sostiene la concesión», «la inconformidad se refiere al fondo» y
# «quien pretende la sustitución» no atribuyen nada, y con la lista amplia de
# `congruencia` tres condensaciones buenas se descartaban.
_RX_VERBO_FUERTE = re.compile(
    r"\b(?:aduce[n]?|adujo|adujeron|alega[n]?|aleg[óo]|alegaron|arguye[n]?|argument[óo]|esgrime[n]?|"
    r"asevera[n]?|asever[óo]|se\s+duele[n]?|se\s+doli[óo])\b", re.I)
_RX_PARTE_Y_VERBO = re.compile(
    r"\b(?:recurrentes?|quejos[ao]s?|inconformes?|promovente|incidentista|tercer[oa]\s+interesad[oa]|"
    r"la\s+parte(?!\s+considerativa)|las\s+partes)\b[^.;]{0,60}?\b(?:sostien[e]n?|sostuv(?:o|ieron)|"
    r"afirma[n]?|afirm[óo]|refiere[n]?|refiri[óo]|se[ñn]ala[n]?|se[ñn]al[óo]|manifiesta[n]?|"
    r"manifest[óo]|plantea[n]?|plante[óo]|atribuye[n]?|invoca[n]?|expresa[n]?|dice[n]?|"
    r"considera[n]?|estima[n]?|pretende[n]?|reclama[n]?|insiste[n]?)\b", re.I)


_RX_DISTANCIA = re.compile(
    r"\b(?:alegad|pretendid|supuest|aducid)[oa]s?\b|"
    r"\b(?:invocad|planteado|sostenid|afirmad)[oa]s?\s+por\s+(?:la|el|las|los)\s+"
    r"(?:parte|recurrente|quejos|inconforme|promovente|incidentista)", re.I)
_VACIAS_ATR = {"que", "porque", "sobre", "entre", "desde", "hasta", "contra", "mediante",
               "cuando", "donde", "aunque", "tambien", "ademas", "dicha", "dicho", "esta",
               "este", "estos", "estas", "otras", "otros", "parte", "recurrente"}


def _todas_las_atribuciones(t: str) -> list:
    return [m for rx in (_RX_ATRIBUCION_EXTRA, _RX_VERBO_FUERTE, _RX_PARTE_Y_VERBO, _RX_DISTANCIA)
            for m in rx.finditer(t or "")]


def atribuciones(t: str) -> int:
    """Cuántas marcas de atribución trae el texto: «en el escrito de agravios
    se reproduce», «según la», «la afirmación de», «la alegada violación», un
    verbo que sólo dice una parte, o uno ambiguo con la parte nombrada delante."""
    return len(_todas_las_atribuciones(t))


def _raiz(w: str) -> str:
    return w[:6]


def hechos_atribuidos(t: str) -> list:
    """Lo que el texto pone en boca de alguien: por cada atribución, las
    palabras de contenido de lo que sigue hasta el fin de la frase. Sólo las
    que dicen algo (4 o más palabras de contenido): «la recurrente sostiene
    que se alteró la cosa juzgada» es el rótulo del agravio, no un hecho."""
    fuera = []
    for m in _todas_las_atribuciones(t):
        cola = re.split(r"[.;:]", (t or "")[m.end():m.end() + 260], 1)[0]
        pal = {_raiz(w) for w in re.findall(r"\w{5,}", _plano(cola)) if w not in _VACIAS_ATR}
        if len(pal) >= 4:
            fuera.append(pal)
    return fuera


def da_por_cierto(viejo: str, nuevo: str) -> bool:
    """¿El texto nuevo afirma sin atribuir lo que el viejo atribuía? Sólo si el
    nuevo ya no atribuye nada y repite al menos el 40% de lo atribuido (y tres
    palabras). El 40%, medido en las dos corridas del 631: una condensación
    suelta la mitad de las palabras, y la que dio por cierto lo de la parte
    conservaba 5 de 12."""
    if atribuciones(nuevo):
        return False
    raices = {_raiz(w) for w in re.findall(r"\w{5,}", _plano(nuevo))}
    return any(len(h & raices) >= 3 and len(h & raices) / len(h) >= 0.4
               for h in hechos_atribuidos(viejo))


def _direcciones(t: str) -> set:
    try:
        import congruencia as _cg
        return _cg.direcciones([t])
    except Exception:
        return set()


def _ordinales(t: str) -> set:
    try:
        import congruencia as _cg
        return _cg.ordinales(t)
    except Exception:
        return set()


def _fechas(t: str) -> set:
    try:
        import fechas_en_autos as _fe
        return set(_fe.fechas_de(t or "").keys())
    except Exception:
        return set()


def _get(o, k, defecto=None):
    if isinstance(o, dict):
        return o.get(k, defecto)
    return getattr(o, k, defecto)


# ═══ EL CALIFICATIVO, NO SÓLO SU DIRECCIÓN (3-oct-2026) ════════════════════
# La revisión adversarial lo ejecutó: `congruencia.direcciones` mete en la
# misma clase «contra» a infundado, inoperante, ineficaz, inatendible y
# «fundado pero insuficiente», así que un parche que cambiaba «se considera
# infundado» por «se considera inoperante» pasaba las guardas con cualquier
# tipo. No es un matiz: el inoperante no entra al fondo. El prompt promete que
# «ninguna corrección cambia una calificación» y ésta es la guarda que lo
# cumple. Se lee sólo en las frases que califican por cuenta de este tribunal
# (`congruencia.califica`), sin los rubros: el mismo filtro de `direcciones`.
CALIFICATIVOS = ("fundado", "infundado", "inoperante", "ineficaz", "inatendible",
                 "fundado_insuficiente", "innecesario", "sin_materia")
_CALIF_CONTRA = {"infundado", "inoperante", "ineficaz", "inatendible", "fundado_insuficiente"}
_RX_FUND_INSUF = re.compile(r"\bfundad\w*\s*,?\s*(?:pero|aunque)\s+(?:\w+\s+)?insuficien\w*", re.I)
_RX_CALIFICATIVO = (
    ("infundado", re.compile(r"\binfundad[oa]s?\b", re.I)),
    ("inoperante", re.compile(r"\binoperan\w*", re.I)),
    ("ineficaz", re.compile(r"\binefica\w*", re.I)),
    ("inatendible", re.compile(r"\binatendib\w*", re.I)),
    ("innecesario", re.compile(r"\binnecesari[oa]s?\b", re.I)),
    ("sin_materia", re.compile(r"\bsin\s+materia\b", re.I)),
)
# «fundado», con o sin «esencialmente/parcialmente/sustancialmente»; «no es
# fundado» es un infundado dicho de otro modo. «Le asiste la razón» es un
# fundado y «no le asiste»/«carece de razón», un infundado: si no se contaran,
# «no le asiste la razón» → «es inoperante» pasaría por igual.
_RX_FUNDADO = re.compile(r"\bfundad[oa]s?\b", re.I)
_RX_ASISTE = re.compile(r"\b(?:le\s+|les\s+)?asiste\s+(?:la\s+)?raz[óo]n|\btiene\w*\s+(?:la\s+)?raz[óo]n", re.I)
_RX_CARECE = re.compile(r"\bcarece\w*\s+de\s+raz[óo]n", re.I)
_RX_NO_ANTES = re.compile(r"\bno\s+(?:\w+\s+){0,2}$", re.I)


def calificativos(t: str) -> set:
    """El conjunto de calificativos propios del texto (de `CALIFICATIVOS`)."""
    fuera = set()
    try:
        import congruencia as _cg
    except Exception:
        return fuera
    for fr in _cg.frases(t or ""):
        if _cg._es_rubro(fr) or not _cg.califica(fr):
            continue
        s = _cg._RX_NO_CALIF.sub(" ", fr)
        if _RX_FUND_INSUF.search(s):
            fuera.add("fundado_insuficiente")
            s = _RX_FUND_INSUF.sub(" ", s)
        for clave, rx in _RX_CALIFICATIVO:
            if rx.search(s):
                fuera.add(clave)
        for rx, si, no in ((_RX_FUNDADO, "fundado", "infundado"), (_RX_ASISTE, "fundado", "infundado")):
            for m in rx.finditer(s):
                fuera.add(no if _RX_NO_ANTES.search(s[max(0, m.start() - 24):m.start()]) else si)
        if _RX_CARECE.search(s):
            fuera.add("infundado")
    return fuera


def _familia(etiqueta: str) -> str:
    """La etiqueta del plan o el sentido del criterio, en el vocabulario de
    `CALIFICATIVOS`: todo lo que prospera es «fundado» (el texto no distingue
    «esencialmente» de «parcialmente»)."""
    s = re.sub(r"\s+", "_", str(etiqueta or "").strip().lower())
    if not s:
        return ""
    if s in CALIFICATIVOS and s != "fundado":
        return s
    try:
        import tipos_asunto as _ta
        if _ta.prospera(s):
            return "fundado"
    except Exception:
        pass
    return s if s in CALIFICATIVOS else ""


# ═══ EL DESENLACE DEL PÁRRAFO (3-oct-2026) ══════════════════════════════════
# LO GRAVE DE LA REVISIÓN ADVERSARIAL. En una revisión en plenitud (art. 93,
# frs. V y VI) el cierre del estudio dice «procede revocar la sentencia
# recurrida y negar el amparo» y NO lleva palabra de calificación. Un parche
# que cambiaba «negar» por «conceder» pasaba todas las guardas, y después
# `fase_rama.sentido_en_plenitud`, `_rama_de` y documento_generado leían el
# segundo resolutivo de ese texto: el motor invertía el sentido que fijó el
# secretario y el proyecto salía coherente con el texto alterado. Ahora un
# parche no puede cambiar ninguna señal de desenlace del párrafo —conceder o
# negar, revocar, confirmar o modificar, reponer, «para efectos», sobreseer o
# levantar el sobreseimiento—, ni negar una que afirmaba («no procede revocar»).
_RX_DESENLACE = (
    ("concede", re.compile(r"\bconced(?:e|en|er|erse|ida|ido|i[óo])\s+(?:el\s+)?(?:amparo|la\s+protecci[óo]n)"
                           r"|\bampara\s+y\s+protege", re.I)),
    ("niega", re.compile(r"\b(?:niega|niegan|neg(?:ar|arse|ada|ado|[óo]))\s+(?:el\s+)?(?:amparo|la\s+protecci[óo]n)"
                         r"|\bno\s+ampara\s+ni\s+protege", re.I)),
    # REVOCAR Y CONFIRMAR, SÓLO CON SU OBJETO o con el verbo del tribunal
    # delante: «lo que confirma que la prueba…» no es un desenlace, y acusarlo
    # descartaría condensaciones buenas.
    ("revoca", re.compile(r"\bse\s+revoca\b(?!\s+(?:el\s+)?sobreseimiento)"
                          r"|\brevoca[rn]?(?:se|la|lo)?\s+(?:la\s+|el\s+)?(?:sentencia|resoluci[óo]n|fallo|"
                          r"interlocutoria|determinaci[óo]n|auto\b|recurrid)"
                          r"|\b(?:procede|debe|deber[áa]|lo\s+procedente\s+es)\s+revocar", re.I)),
    ("confirma", re.compile(r"\bse\s+confirma\b(?!\s+que\b)"
                            r"|\bconfirma[rn]?(?:se|la|lo)?\s+(?:la\s+|el\s+)?(?:sentencia|resoluci[óo]n|fallo|"
                            r"interlocutoria|determinaci[óo]n|auto\b|recurrid)"
                            r"|\b(?:procede|debe|deber[áa]|lo\s+procedente\s+es)\s+confirmar", re.I)),
    ("modifica", re.compile(r"\b(?:se\s+modifica|modificar(?:se|la|lo)?)\s+(?:la\s+)?(?:sentencia|resoluci[óo]n|fallo)", re.I)),
    ("repone", re.compile(r"\breponer\w*\s+(?:el\s+)?procedimiento|\breposici[óo]n\s+del\s+procedimiento", re.I)),
    ("para_efectos", re.compile(r"\b(?:[úu]nicamente|s[óo]lo|solamente)\s+para\s+(?:los\s+)?efectos\b"
                                r"|\b(?:amparo|protecci[óo]n)\s+(?:\w+\s+){0,6}?para\s+(?:los\s+)?efectos\b"
                                r"|\bpara\s+(?:los\s+)?efectos\s+(?:precisados|que\s+se\s+precisan|siguientes)\b", re.I)),
    ("sobresee", re.compile(r"\bse\s+sobresee\b|\bsobrese(?:er|erse|y[óo]|[ií]do)\b"
                            r"|\bdecret\w+\s+el\s+sobreseimiento", re.I)),
    ("levanta_sobreseimiento", re.compile(r"\b(?:levant|revoc)\w*\s+(?:el\s+)?sobreseimiento", re.I)),
)


def desenlace_de(t: str) -> set:
    """Las señales de desenlace del texto, con «no_» delante si van negadas,
    más lo que de él leen `fase_rama` (plenitud, violación procesal, sólo los
    efectos) y `desenlace._dichos` (revoca/confirma por cuenta del tribunal)."""
    tt = " ".join((t or "").split())
    fuera = set()
    for clave, rx in _RX_DESENLACE:
        for m in rx.finditer(tt):
            neg = _RX_NO_ANTES.search(tt[max(0, m.start() - 30):m.start()])
            fuera.add(("no_" if neg else "") + clave)
    try:
        import fase_rama as _fr
        s = _fr.sentido_en_plenitud(tt)
        if s:
            fuera.add("plenitud_" + s)
        if _fr.hay_violacion_procesal(tt):
            fuera.add("violacion_procesal")
        if _fr.solo_los_efectos(tt):
            fuera.add("solo_los_efectos")
    except Exception:
        pass
    try:
        import desenlace as _ds
        for prospera, _ in _ds._dichos(tt):
            fuera.add("dicho_revoca" if prospera else "dicho_confirma")
    except Exception:
        pass
    return fuera


def desenlace_global(estudio: str) -> tuple:
    """Lo que de TODO el estudio leen el resolutivo y la rama: el sentido en
    plenitud, si hay violación procesal, si sólo son los efectos y el último
    dicho revoca/confirma. La red de seguridad de `supervisar`: si cambia
    entre el estudio de entrada y el corregido, se revierte todo."""
    tt = " ".join((estudio or "").split())
    plen = proc = efe = ""
    try:
        import fase_rama as _fr
        plen, proc, efe = (_fr.sentido_en_plenitud(tt), _fr.hay_violacion_procesal(tt),
                           _fr.solo_los_efectos(tt))
    except Exception:
        pass
    ult = None
    try:
        import desenlace as _ds
        d = _ds._dichos(tt)
        ult = d[-1][0] if d else None
    except Exception:
        pass
    return (plen, proc, efe, ult)


# ═══ LO QUE ANCLA LAS GUARDAS ═══════════════════════════════════════════════
class Anclas:
    """Lo que un parche puede citar sin inventar: el material y las fuentes."""

    def __init__(self, material=None, fuentes=None, criterios=None, plan=None):
        tesis = list(_get(material, "tesis", None) or [])
        normas = list(_get(material, "normas", None) or [])
        self.registros = set()
        self.num_anio = set()
        # LOS RUBROS ENTEROS DEL MATERIAL y los registros de tesis que PERDIERON
        # VIGENCIA (3-oct-2026): un rubro nuevo sólo entra si es, completo, el
        # de una tesis del material, y un registro nuevo nunca es el de una
        # jurisprudencia abandonada (f6.revisar, que lo avisaría, corre antes).
        self.rubros = set()
        self.sin_vigencia = set()
        texto_tesis = []
        for t in tesis:
            if not isinstance(t, dict):
                continue
            regs = registros_de(str(t.get("registro") or ""))
            self.registros |= regs
            if t.get("rubro"):
                self.rubros.add(_plano(t.get("rubro")).rstrip(". "))
            try:
                import fuerza_juridica as _fj
                if _fj.sello_perdio_vigencia(t.get("vigencia")):
                    self.sin_vigencia |= regs
            except Exception:
                pass
            for k in ("clave", "tesis", "rubro"):
                texto_tesis.append(str(t.get(k) or ""))
        for x in texto_tesis:
            self.num_anio |= numeros_con_anio(x)
        self.articulos = set()
        for n in normas:
            if isinstance(n, dict):
                self.articulos |= set(re.findall(r"\d{1,4}", str(n.get("articulo") or "")))
        fuentes = [str(x or "") for x in (fuentes or []) if x]
        self.fuentes_plano = _plano("\n".join(fuentes))
        self.fechas = set()
        for f in fuentes:
            self.fechas |= _fechas(f)
        for f in fuentes:
            self.num_anio |= numeros_con_anio(f)
        # LA DIRECCIÓN DEL CRITERIO: si todos los problemas van en la misma,
        # un párrafo que califica al revés es la incongruencia que se corrige.
        self.dir_criterio = set()
        try:
            import exhaustivo as _ex
            for c in criterios or []:
                d = _ex._direccion(str(_get(c, "sentido") or ""))
                if d:
                    self.dir_criterio.add(d)
        except Exception:
            pass
        # LAS ETIQUETAS DEL PLAN (v4): {id del argumento: su calificación dentro
        # de su problema}. Es lo único que dice qué calificación corresponde a
        # un párrafo concreto: el sentido del problema no basta, porque dentro
        # de un problema fundado cabe un argumento infundado (plan_estudio).
        try:
            import exhaustivo as _ex2
            self.etiquetas = dict(_ex2.etiquetas_del_plan(plan) or {})
        except Exception:
            self.etiquetas = {}
        # EL INVENTARIO de argumentos, para medir que un párrafo condensado
        # siga contestando cada argumento que marca (`marcas.rastros`).
        self.inventario = [s for s in list(_get(material, "inventario", None) or [])
                           if isinstance(s, dict) and s.get("id")]


# ═══ LAS GUARDAS ════════════════════════════════════════════════════════════
def _endereza(bloque: dict, tipo: str, d_v: set, d_n: set, c_v: set, c_n: set,
              anclas: Anclas) -> bool:
    """¿El cambio de calificación es la incongruencia que se corrige?

    LA EXCEPCIÓN ACOTADA (revisión adversarial, 3-oct-2026). Antes bastaba que
    todo el criterio fuera en un solo sentido y el párrafo calificara al revés.
    Pero con un criterio todo «fundado», el plan v4 etiqueta como infundados o
    inoperantes argumentos DENTRO del problema fundado (plan_estudio: «si el
    problema prospera, al menos uno lo funda y los demás pueden ser infundados
    o inoperantes»); el estudio, siguiendo el plan, escribe «Es infundado…», y
    Gemini lo volteaba a «fundado» con tipo incongruencia. Se eligió:

      1 · ATRIBUIR POR EL PLAN cuando se puede —el párrafo marca argumentos y
          todos tienen etiqueta—: el texto nuevo tiene que decir exactamente
          los calificativos y la dirección de esas etiquetas, y el viejo no.
          Es lo único que permite también la corrección de calificativo en la
          misma dirección (el estudio escribió «infundado» donde el plan dice
          «inoperante»).
      2 · SIN PLAN QUE LO ATRIBUYA, sólo el volteo a «contra» cuando TODO el
          criterio es «contra»: es el único sentido en que el plan prohíbe
          argumentos de la dirección opuesta dentro del problema («si no
          prospera, ninguno queda fundado»), así que un «fundado» ahí es
          incongruente sin necesidad de saber a qué argumento responde. Con un
          criterio todo «favor» no se puede saber y el parche se descarta.

    Y siempre con tipo error_juridico o incongruencia: «redacción» no corrige
    calificaciones."""
    if tipo not in _TIPOS_QUE_ALARGAN or not c_n:
        return False
    ids = list(dict.fromkeys(bloque.get("ids") or []))
    if ids and anclas.etiquetas and all(i in anclas.etiquetas for i in ids):
        fams = {_familia(anclas.etiquetas[i]) for i in ids} - {""}
        if not fams:
            return False
        dirs = {"favor" if f == "fundado" else "contra" for f in fams if f in _CALIF_CONTRA | {"fundado"}}
        return c_n == fams and d_n == dirs and (c_v != fams or d_v != dirs)
    return (anclas.dir_criterio == {"contra"} and "favor" in d_v and d_n == {"contra"}
            and c_n <= _CALIF_CONTRA)


def _pierde_respuesta(bloque: dict, viejo: str, nuevo: str, anclas: Anclas) -> str:
    """La id de un argumento marcado que tenía rastro en el párrafo viejo y no
    en el nuevo, o «». Las ids sin rastro en el viejo no se pueden comprobar
    y pasan. El idf de `rastros` se calcula sobre el inventario ENTERO."""
    ids = set(bloque.get("ids") or [])
    if not ids or not anclas.inventario:
        return ""
    try:
        import marcas as _mc
        rv = _mc.rastros(anclas.inventario, viejo)
        rn = _mc.rastros(anclas.inventario, nuevo)
        um = _mc.UMBRAL_RASTRO
    except Exception:
        return ""
    for i in sorted(ids):
        if i not in rv:
            continue
        habia = rv[i][0] or rv[i][1] >= um
        queda = bool(rn.get(i)) and (rn[i][0] or rn[i][1] >= um)
        if habia and not queda:
            return i
    return ""


def guardas(parche: dict, bloque: dict, anclas: Anclas) -> str:
    """El motivo para descartar un parche, o «» si pasa."""
    if bloque is None:
        return "párrafo inexistente"
    if bloque.get("fijo"):
        return f"párrafo fijo ({bloque['fijo']})"
    accion, tipo = parche.get("accion"), parche.get("tipo")
    viejo = bloque["texto"]
    if accion == "eliminar":
        if bloque.get("ids"):
            return "eliminar un párrafo que contesta argumentos (lleva marcas)"
        if registros_de(viejo) or articulos_de(viejo) or numeros_con_anio(viejo):
            return "eliminar un párrafo con citas"
        if _direcciones(viejo):
            return "eliminar un párrafo que califica"
        if _rubros_entre_comillas(viejo):
            return "eliminar un párrafo con un rubro"
        # Un cierre «procede revocar … y negar el amparo» no lleva palabra de
        # calificación ni cita: sin esta guarda se podía eliminar (3-oct-2026).
        if desenlace_de(viejo):
            return "eliminar un párrafo que dicta el desenlace"
        return ""
    nuevo = parche.get("texto") or ""
    if "⟦" in nuevo or "⟧" in nuevo or "[[" in nuevo:
        return "marca dentro del texto"
    if _RX_ID_INTERNO.search(nuevo) and not _RX_ID_INTERNO.search(viejo):
        return "identificador interno en el texto"
    n_v, n_n = _palabras(viejo), _palabras(nuevo)
    if n_n < MIN_PALABRAS:
        return f"párrafo de menos de {MIN_PALABRAS} palabras ({n_n})"
    if accion == "condensar" and n_n >= n_v:
        return "condensar sin acortar"
    if n_n > n_v * CRECIMIENTO_MAX and tipo not in _TIPOS_QUE_ALARGAN:
        return f"alarga el párrafo más de un 20% ({n_v} → {n_n} palabras)"
    # LA CALIFICACIÓN: misma dirección Y mismos calificativos (3-oct-2026:
    # infundado → inoperante pasaba con la dirección sola). La única excepción
    # es la incongruencia que ENDEREZA un párrafo que califica distinto de lo
    # que le corresponde; ver `_endereza`.
    d_v, d_n = _direcciones(viejo), _direcciones(nuevo)
    c_v, c_n = calificativos(viejo), calificativos(nuevo)
    if (d_v != d_n or c_v != c_n) and not _endereza(bloque, tipo, d_v, d_n, c_v, c_n, anclas):
        if d_v != d_n:
            return f"cambia la calificación ({'/'.join(sorted(d_v)) or 'ninguna'} → " \
                   f"{'/'.join(sorted(d_n)) or 'ninguna'})"
        return f"cambia el calificativo ({'/'.join(sorted(c_v)) or 'ninguno'} → " \
               f"{'/'.join(sorted(c_n)) or 'ninguno'})"
    # EL DESENLACE (lo grave de la revisión del 3-oct-2026): ninguna señal de
    # conceder/negar, revocar/confirmar/modificar, reponer, «para efectos» o
    # sobreseer puede aparecer, desaparecer ni negarse. El criterio es del
    # secretario; el supervisor corrige la prosa, no el resolutivo.
    ds_v, ds_n = desenlace_de(viejo), desenlace_de(nuevo)
    if ds_v != ds_n:
        return f"cambia el desenlace ({'/'.join(sorted(ds_v)) or 'ninguno'} → " \
               f"{'/'.join(sorted(ds_n)) or 'ninguno'})"
    # LA RESPUESTA A CADA ARGUMENTO MARCADO (3-oct-2026): condensar no puede
    # quitar lo que contesta a uno de los argumentos que el párrafo marca. El
    # mapa seguiría dándolo por contestado —la marca se repone delante— y el
    # control V1 no avisaría. Se mide con el rastro de `marcas.verificar`: un
    # argumento que tenía rastro en el párrafo viejo tiene que conservarlo.
    perdido = _pierde_respuesta(bloque, viejo, nuevo, anclas)
    if perdido:
        return f"quita la respuesta a {perdido}"
    # LOS REGISTROS DEL PÁRRAFO NO SE QUITAN NI SE CAMBIAN (3-oct-2026): una
    # cita cambiada por el supervisor elude los controles de citas, que ya
    # corrieron. Si la cita no sostiene lo que se dice, se corrige la frase.
    reg_quitados = registros_de(viejo) - registros_de(nuevo)
    if reg_quitados:
        return f"quita o cambia un registro del párrafo ({', '.join(sorted(reg_quitados)[:3])})"
    reg_nuevos = registros_de(nuevo) - registros_de(viejo) - anclas.registros
    if reg_nuevos:
        return f"registro fuera del material ({', '.join(sorted(reg_nuevos)[:3])})"
    reg_caducos = (registros_de(nuevo) - registros_de(viejo)) & anclas.sin_vigencia
    if reg_caducos:
        return f"cita una tesis que perdió vigencia ({', '.join(sorted(reg_caducos)[:3])})"
    na_nuevos = numeros_con_anio(nuevo) - numeros_con_anio(viejo) - anclas.num_anio
    if na_nuevos:
        return f"número de tesis o expediente que no consta ({', '.join(sorted(na_nuevos)[:3])})"
    art_nuevos = articulos_de(nuevo) - articulos_de(viejo) - anclas.articulos
    if art_nuevos:
        return f"artículo fuera del párrafo y de las normas ({', '.join(sorted(art_nuevos)[:3])})"
    f_nuevas = _fechas(nuevo) - _fechas(viejo) - anclas.fechas
    if f_nuevas:
        return "fecha que no consta"
    for r in _rubros_entre_comillas(viejo):
        if _plano(r) not in _plano(nuevo):
            return "altera o quita un rubro transcrito"
    # UN RUBRO NUEVO (texto en mayúsculas entre comillas) sólo entra si ya
    # estaba en el párrafo o es, ENTERO, el de una tesis del material: la
    # comprobación de transcripciones de abajo salta los rubros, y el catálogo
    # se le enseña al modelo; un rubro truncado («… […]»), alterado o de otra
    # tesis pegado a un registro real pasaba (3-oct-2026).
    for r in _rubros_entre_comillas(nuevo):
        rp = _plano(r).rstrip(". ")
        if rp not in _plano(viejo) and rp not in anclas.rubros:
            return "rubro entre comillas que no es, entero, el de una tesis del material"
    pv = _plano(viejo)
    for m in _RX_CITA_COMILLAS.finditer(nuevo):
        q = m.group(1)
        if _palabras(q) >= 6 and not _es_rubro(q):
            qp = _plano(q)
            if qp not in pv and qp not in anclas.fuentes_plano:
                return "transcripción entrecomillada que no consta"
    if frases_de_herramienta(nuevo):
        return "frase de herramienta"
    if da_por_cierto(viejo, nuevo):
        return "borra la atribución a la parte: da por cierto lo que sólo se afirma"
    o_v, o_n = _ordinales(viejo), _ordinales(nuevo)
    if not o_v <= o_n:
        return "pierde el ordinal del concepto que contesta"
    if not o_n <= o_v and tipo not in _TIPOS_QUE_ALARGAN:
        return "nombra un concepto que el párrafo no contestaba"
    pre_v = _RX_PREFIJO_ORDEN.match(viejo)
    if pre_v:
        pre_n = _RX_PREFIJO_ORDEN.match(nuevo)
        if not pre_n or re.sub(r"\s", "", pre_n.group(1)) != re.sub(r"\s", "", pre_v.group(1)):
            return "pierde el número de la orden"
    return ""


# ═══ LOS ANTECEDENTES (3-oct-2026) ══════════════════════════════════════════
# David, 3-oct-2026: «el revisor agente (gemini) concentra su capacidad en
# antecedentes y estudio de fondo». Hasta hoy los parches sólo alcanzaban el
# estudio, y los antecedentes —que escribe el modelo de las fases, a veces con
# la fecha o el órgano equivocados, o dando por cierto lo que sólo afirmó una
# parte— llegaban al proyecto sin segunda lectura. Aquí van sus párrafos, sus
# anclas y sus guardas, APARTE de las del estudio: lo que se mide en un
# antecedente (lo que resolvió cada instancia, sus fechas, sus nombres) no es
# lo que se mide en un párrafo del estudio (la calificación, las citas).
ANT_CRECIMIENTO_MAX = 1.20
# Lo que tiene que repetirse de un párrafo en OTRO para que borrarlo no pierda
# nada: el 80% de sus palabras de contenido y TODOS sus datos.
ANT_UMBRAL_REPETICION = 0.80
# El extracto del acto que ve el revisor (`extracto_del_acto`): hasta
# EXTRACTO_CABEZA caracteres de los pasajes donde constan los datos de los
# antecedentes, y EXTRACTO_COLA de los puntos resolutivos. El prompt sigue
# corto; las guardas cotejan contra el texto ENTERO.
EXTRACTO_CABEZA = 7000
EXTRACTO_COLA = 3000
# La síntesis moderna corre a la par del estudio y casi siempre ya terminó;
# si no, no se la espera más que esto (los antecedentes se quedan sin revisar).
ESPERA_SINTESIS_S = 30.0

# EL TRÁMITE EN ESTE TRIBUNAL NO ES UN ANTECEDENTE: lo escriben los
# resultandos por construcción (resultandos_por_tipo) y al revisor no se le
# enseña. Si el redactor de las fases lo metió en los antecedentes, el
# párrafo queda FIJO y fuera del prompt.
_RX_TRAMITE_TCC = re.compile(
    r"\bpor\s+(?:auto|acuerdo)\s+de\s+presidencia\b"
    r"|\b(?:este|el\s+presente)\s+(?:tribunal\s+colegiado|[óo]rgano\s+(?:jurisdiccional\s+)?colegiado)"
    r"\s+(?:\w+\s+){0,3}?(?:registr|admiti|turn|return)\w*"
    r"|\bse\s+(?:turn|return)\w+\s+(?:los\s+autos|el\s+(?:asunto|expediente|toca))\b", re.I)
# LAS PERÍFRASIS QUE SUSTITUYEN UN DATO (SPEC procedencia §0: «se advierte de
# las constancias» en 76 de 81 resultandos de octubre). Un parche no puede
# traer una que el párrafo no tenía; la que ya trae se le señala para quitarla.
_RX_PERIFRASIS = re.compile(
    r"\bse\s+advierte\s+de\s+(?:las|los)\s+(?:constancias|autos)\b"
    r"|\bcuya\s+fecha\s+se\s+advierte\b"
    r"|\bque\s+obran?\s+en\s+(?:autos|el\s+expediente)\b"
    r"|\bla\s+persona\s+a\s+quien\s+resulta\b"
    r"|\bno\s+consta\s+en\s+los\s+datos\s+proporcionados\b", re.I)
_RX_HUECO = re.compile(r"\*{3,}")
_RX_ETIQUETA = re.compile(r"\[\s*[AP]\s*\d{1,4}\b")
_RX_NUMERO = re.compile(r"\d+(?:[.,]\d+)*")


def numeros_de(t: str) -> set:
    """Los números en cifra del texto, sin separadores ni ceros a la izquierda
    («912/2022» → {912, 2022}; «$2,500.00» → {250000}; «3.5» → {35})."""
    fuera = set()
    for x in _RX_NUMERO.findall(t or ""):
        y = re.sub(r"[.,]", "", x).lstrip("0")
        fuera.add(y or "0")
    return fuera


def _dias_mes(t: str) -> set:
    """{(mes, día)} de toda fecha del texto, con año o sin él: «el ocho de
    abril siguiente» no lo lee `fechas_en_autos` (exige los tres elementos) y
    un parche podía cambiarlo por «el nueve de abril siguiente»."""
    try:
        import fechas_en_autos as _fe
    except Exception:
        return set()
    tt = " ".join(str(t or "").split())
    meses = "|".join(_fe._MESES)
    fuera = {(m, d) for (_a, m, d) in _fe.fechas_de(tt)}
    for m in re.finditer(rf"\b({_fe._DIA_RX})\s+de\s+({meses})\b", tt, re.I):
        d, mes = _fe._dia(m.group(1)), _fe._MESES.get(m.group(2).lower())
        if d and mes:
            fuera.add((mes, d))
    for m in re.finditer(rf"\b(\d{{1,2}})\s+de\s+({meses})\b", tt, re.I):
        d, mes = int(m.group(1)), _fe._MESES.get(m.group(2).lower())
        if mes and 1 <= d <= 31:
            fuera.add((mes, d))
    return fuera


# LAS CANTIDADES EN LETRA («seis mil pesos», «quince días»): un número escrito
# con palabras también es un dato que el parche no puede cambiar.
_PAL_NUM = (r"(?:un|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|"
            r"quince|diec\w+|veint\w+|treinta|cuarenta|cincuenta|sesenta|setenta|ochenta|noventa|"
            r"cien|ciento|\w+cientos|\w+cientas|quinientos|quinientas|mil|millones|mill[óo]n)")
_RX_CANTIDAD_LETRA = re.compile(
    rf"\b({_PAL_NUM}(?:\s+(?:y\s+)?{_PAL_NUM})*)\s+(?:de\s+)?(pesos|d[íi]as|meses|a[ñn]os|por\s+ciento|"
    rf"unidades|salarios|semanas)\b", re.I)


def cantidades_en_letra(t: str) -> set:
    return {_plano(m.group(0)) for m in _RX_CANTIDAD_LETRA.finditer(" ".join(str(t or "").split()))}


# LOS NOMBRES PROPIOS: las palabras con mayúscula seguidas, partidas en los
# conectores («Primera Sala Civil» · «Tribunal Superior» · «Justicia») y sin la
# que abre la oración. Cada tramo tiene que estar, palabra por palabra y
# entero, en el párrafo viejo o en las fuentes: «Juzgado Quinto» por «Juzgado
# Cuarto», o una persona que no aparece en ningún papel, se descartan.
_RX_PALABRA = re.compile(r"[A-Za-zÁÉÍÓÚÑÜáéíóúñü][\wÁÉÍÓÚÑÜáéíóúñü.]*")
_GENERICOS = {"ley", "amparo", "constitucion", "constitucion politica", "constitucion federal",
              "estados unidos mexicanos", "tribunal colegiado", "justicia", "justicia federal", "union",
              "federacion", "codigo", "republica", "estado", "estados"}


def nombres_de(t: str) -> list:
    """Los tramos de nombre propio del texto, en plano, sin repetir."""
    t = str(t or "")
    fuera, tramo, fin_ant = [], [], None

    def _cierra():
        if tramo:
            s = _plano(" ".join(tramo))
            if s and s not in fuera:
                fuera.append(s)
        tramo.clear()

    for m in _RX_PALABRA.finditer(t):
        w = m.group(0)
        punto_final = False
        if "." in w[:-1] or (len(w) == 2 and w.endswith(".") and w[0].isupper()):
            pass                                  # abreviatura o inicial («C.T.M.», «S.A», «J.»): se queda
        elif w.endswith("."):
            w, punto_final = w.rstrip("."), True  # «Civil. Pedro…»: el punto cierra el tramo
        antes = t[fin_ant if fin_ant is not None else 0:m.start()]
        if fin_ant is not None and antes.strip():
            _cierra()                             # una coma o un punto parten el tramo
        fin_ant = m.end()
        if not w[:1].isupper():
            _cierra()                             # un conector o una palabra común
            continue
        previo = t[:m.start()].rstrip()
        if not tramo and (not previo or previo[-1] in ".;:!?¿¡«\"“(\n"):
            continue                              # la que abre la oración no es un nombre
        tramo.append(w)
        if punto_final:
            _cierra()
    _cierra()
    return fuera


def _en(s: str, plano: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", plano or ""))


# LO QUE RESOLVIÓ CADA INSTANCIA Y LO QUE TUVO POR ACREDITADO. El conjunto de
# señales del párrafo viejo tiene que ser el del nuevo: ni se quita «absolvió»
# de un «condenó a X y absolvió de Y», ni se voltea «confirmó» en «revocó», ni
# «no acreditó» en «acreditó». Se suman las de `_RX_DESENLACE` (conceder,
# negar, revocar, confirmar, modificar, reponer, sobreseer…).
_RX_SENTIDO_ANT = (
    ("condena", re.compile(r"\bcondena\w*|\bcondenó\b", re.I)),
    ("absuelve", re.compile(r"\babsuel\w+|\babsolv\w+", re.I)),
    ("procedente", re.compile(r"\bprocedentes?\b|\bprocedencia\b", re.I)),
    ("improcedente", re.compile(r"\bimprocedentes?\b|\bimprocedencia\b", re.I)),
    ("fundado", re.compile(r"\bfundad[oa]s?\b", re.I)),
    ("infundado", re.compile(r"\binfundad[oa]s?\b", re.I)),
    ("inoperante", re.compile(r"\binoperantes?\b", re.I)),
    ("acredita", re.compile(r"\bacredit\w*|\bprob[óo]\b|\bprobad[oa]s?\b|\bdemostr\w*", re.I)),
    ("desecha", re.compile(r"\bdesech\w+", re.I)),
    ("nulidad", re.compile(r"\bnulidad\b", re.I)),
    ("validez", re.compile(r"\bvalidez\b", re.I)),
    ("sin_materia", re.compile(r"\bsin\s+materia\b", re.I)),
    ("legitimacion", re.compile(r"\blegitimaci[óo]n\b|\blegitimad[oa]s?\b", re.I)),
    ("prescripcion", re.compile(r"\bprescri\w+|\bcaducidad\b|\bcaduc[óo]\b", re.I)),
    # EN PRETÉRITO, QUE ES COMO NARRAN LOS ANTECEDENTES. Las de `_RX_DESENLACE`
    # son del estudio («procede revocar», «se confirma»): sobre el AR 631/2025
    # no veían «la Sala confirmó la sentencia definitiva» (medido el 3-oct-2026).
    ("confirma", re.compile(r"\bconfirm\w*\s+(?:en\s+sus\s+t[ée]rminos\s+)?(?:la\s+|el\s+|lo\s+)?(?:sentencia|"
                            r"resoluci[óo]n|fallo|interlocutoria|determinaci[óo]n|auto\b|acuerdo|laudo|"
                            r"recurrid|apelad|impugnad|reclamad)", re.I)),
    ("revoca", re.compile(r"\brevoc\w*\s+(?:la\s+|el\s+|lo\s+)?(?:sentencia|resoluci[óo]n|fallo|interlocutoria|"
                          r"determinaci[óo]n|auto\b|acuerdo|laudo|recurrid|apelad|impugnad|reclamad)", re.I)),
    ("modifica", re.compile(r"\bmodific\w*\s+(?:la\s+|el\s+|lo\s+)?(?:sentencia|resoluci[óo]n|fallo|"
                            r"interlocutoria|determinaci[óo]n|auto\b|acuerdo|laudo|recurrid|apelad|impugnad)",
                            re.I)),
    ("admite", re.compile(r"\badmiti\w*|\badmit[ea](?:n|r)?\b", re.I)),
    ("decreta", re.compile(r"\bdecret[óoa]\w*", re.I)),
    ("concede_suspension", re.compile(r"\bconced\w*\s+(?:la\s+)?suspensi[óo]n", re.I)),
    ("niega_suspension", re.compile(r"\bneg[óoa]\w*\s+(?:la\s+)?suspensi[óo]n", re.I)),
)
_RX_NO_PRESENTADA = re.compile(r"\bpor\s+no\s+(?:presentad|interpuest)\w*", re.I)
_RX_NEG_ANT = re.compile(r"\b(?:no|ni|sin|nunca|carec\w*\s+de|falta\s+de)\s+(?:\w+\s+){0,2}$", re.I)


def sentido_antecedente(t: str) -> set:
    """Las señales de lo resuelto o acreditado, «no_» delante si van negadas."""
    tt = " ".join(str(t or "").split())
    fuera = set()
    if _RX_NO_PRESENTADA.search(tt):
        fuera.add("no_presentada")
        tt = _RX_NO_PRESENTADA.sub(" ", tt)
    for clave, rx in tuple(_RX_SENTIDO_ANT) + tuple(_RX_DESENLACE):
        for m in rx.finditer(tt):
            neg = _RX_NEG_ANT.search(tt[max(0, m.start() - 30):m.start()])
            fuera.add(("no_" if neg else "") + clave)
    return fuera


# LAS PARTES POR SU PAPEL. Un parche de redacción no cambia quién hizo qué:
# «la parte demandada interpuso apelación» no pasa a «la parte actora». Sólo
# una corrección de fondo (incongruencia con la resolución) puede nombrar a
# una parte que el párrafo no nombraba.
_RX_ROLES = (
    ("actora", re.compile(r"\bactor(?:a|es|as)?\b", re.I)),
    ("demandada", re.compile(r"\bdemandad[oa]s?\b", re.I)),
    ("quejosa", re.compile(r"\bquejos[oa]s?\b", re.I)),
    ("tercero", re.compile(r"\btercer[oa]s?\s+interesad[oa]s?\b", re.I)),
    ("recurrente", re.compile(r"\brecurrentes?\b", re.I)),
    ("incidentista", re.compile(r"\bincidentistas?\b", re.I)),
    ("apelante", re.compile(r"\bapelantes?\b", re.I)),
    ("ministerio_publico", re.compile(r"\bministerio\s+p[úu]blico\b", re.I)),
    ("autoridad", re.compile(r"\bautoridad(?:es)?\s+(?:responsables?|demandadas?)\b", re.I)),
)


def _roles(t: str) -> set:
    return {k for k, rx in _RX_ROLES if rx.search(t or "")}


_TIPOS_DE_FONDO_ANT = ("incongruencia", "error_juridico")


def bloques_antecedentes(antecedentes) -> list:
    """Los párrafos de los antecedentes, numerados aparte del estudio:
    [{n, indice, texto, fijo}]. `indice` es su posición en la lista (o en los
    renglones, si llegan como cadena)."""
    lista = antecedentes.split("\n") if isinstance(antecedentes, str) else list(antecedentes or [])
    fuera = []
    for k, x in enumerate(lista):
        t = " ".join(str(x or "").split())
        if not t:
            continue
        fijo = "trámite en este tribunal" if _RX_TRAMITE_TCC.search(t) else _fijo(t)
        fuera.append({"n": len(fuera) + 1, "indice": k, "texto": t, "fijo": fijo,
                      "ids": [], "marcas": []})
    return fuera


class AnclasAntecedentes:
    """Lo que un parche de los antecedentes puede decir sin inventar: el acto,
    el escrito, las constancias y lo que aportó el secretario. NO los
    resúmenes del modelo: un dato que sólo consta en prosa de modelo no ancla."""

    def __init__(self, fuentes=None):
        textos = [str(x or "") for x in (fuentes or []) if x]
        todo = "\n".join(textos)
        self.plano = _plano(todo)
        self.fechas = _fechas(todo)
        self.dias_mes = _dias_mes(todo)
        self.numeros = numeros_de(todo)
        self.articulos = articulos_de(todo)
        self.cantidades = cantidades_en_letra(todo)
        self.hay_fuentes = bool(todo.strip())


def _contenido(t: str) -> set:
    return {_raiz(w) for w in re.findall(r"\w{5,}", _plano(t)) if w not in _VACIAS_ATR}


def repite_en(texto: str, otros: list) -> int:
    """El índice, en `otros`, del párrafo que ya dice TODO lo de `texto`
    (el 80% de sus palabras de contenido, todas sus fechas, números,
    artículos, nombres y lo que resolvió), o -1. Es la «repetición pura»."""
    c = _contenido(texto)
    if len(c) < 3:
        return -1
    datos = (_fechas(texto), _dias_mes(texto), numeros_de(texto), articulos_de(texto))
    nombres, sentido = nombres_de(texto), sentido_antecedente(texto)
    for i, o in enumerate(otros or []):
        if not o:
            continue
        if len(c & _contenido(o)) / len(c) < ANT_UMBRAL_REPETICION:
            continue
        if not all(d <= f(o) for d, f in zip(datos, (_fechas, _dias_mes, numeros_de, articulos_de))):
            continue
        po = _plano(o)
        if not all(_en(n, po) for n in nombres) or not sentido <= sentido_antecedente(o):
            continue
        return i
    return -1


def guardas_antecedente(parche: dict, bloque: dict, anclas: AnclasAntecedentes, otros=None) -> str:
    """El motivo para descartar un parche de los antecedentes, o «» si pasa.
    `otros`: el texto vigente de los demás párrafos (para la repetición pura)."""
    if bloque is None:
        return "párrafo inexistente"
    if bloque.get("fijo"):
        return f"párrafo fijo ({bloque['fijo']})"
    accion, tipo = parche.get("accion"), parche.get("tipo")
    viejo = bloque["texto"]
    if accion == "eliminar":
        # SÓLO LA REPETICIÓN PURA: lo que se borra tiene que estar dicho, con
        # todos sus datos, en otro párrafo que se queda.
        if tipo != "repeticion":
            return "eliminar sólo procede por repetición"
        if repite_en(viejo, otros or []) < 0:
            return "eliminar un párrafo que dice algo que ningún otro dice"
        return ""
    nuevo = parche.get("texto") or ""
    if "⟦" in nuevo or "⟧" in nuevo or "[[" in nuevo or _RX_ETIQUETA.search(nuevo):
        return "marca o etiqueta dentro del texto"
    if _RX_ID_INTERNO.search(nuevo) and not _RX_ID_INTERNO.search(viejo):
        return "identificador interno en el texto"
    if _RX_HUECO.search(nuevo) and not _RX_HUECO.search(viejo):
        return "deja un dato en blanco"
    n_v, n_n = _palabras(viejo), _palabras(nuevo)
    if n_n < MIN_PALABRAS:
        return f"párrafo de menos de {MIN_PALABRAS} palabras ({n_n})"
    if accion == "condensar" and n_n >= n_v:
        return "condensar sin acortar"
    # SIN EXCEPCIÓN POR TIPO: un antecedente se corrige con el dato bueno, no
    # con más palabras.
    if n_n > n_v * ANT_CRECIMIENTO_MAX:
        return f"alarga el párrafo más de un 20% ({n_v} → {n_n} palabras)"
    pv = _plano(viejo)
    if _fechas(nuevo) - _fechas(viejo) - anclas.fechas or \
            _dias_mes(nuevo) - _dias_mes(viejo) - anclas.dias_mes:
        return "fecha que no consta en el acto, el escrito ni las constancias"
    art = articulos_de(nuevo) - articulos_de(viejo) - anclas.articulos
    if art:
        return f"artículo que no consta en las fuentes ({', '.join(sorted(art)[:3])})"
    num = numeros_de(nuevo) - numeros_de(viejo) - anclas.numeros
    if num:
        return f"número que no consta en las fuentes ({', '.join(sorted(num)[:3])})"
    cant = cantidades_en_letra(nuevo) - cantidades_en_letra(viejo) - anclas.cantidades
    if cant:
        return f"cantidad que no consta en las fuentes («{sorted(cant)[0]}»)"
    for s in nombres_de(nuevo):
        if s in _GENERICOS or _en(s, pv) or _en(s, anclas.plano):
            continue
        return f"nombre que no consta en las fuentes («{s}»)"
    s_v, s_n = sentido_antecedente(viejo), sentido_antecedente(nuevo)
    if s_v != s_n:
        return f"cambia lo resuelto o lo acreditado ({'/'.join(sorted(s_v)) or 'nada'} → " \
               f"{'/'.join(sorted(s_n)) or 'nada'})"
    if _roles(nuevo) - _roles(viejo) and tipo not in _TIPOS_DE_FONDO_ANT:
        return "nombra una parte que el párrafo no nombraba"
    for m in _RX_CITA_COMILLAS.finditer(nuevo):
        q = _plano(m.group(1))
        if _palabras(q) >= 6 and q not in pv and q not in anclas.plano:
            return "transcripción entrecomillada que no consta"
    if frases_de_herramienta(nuevo):
        return "frase de herramienta"
    if len(_RX_PERIFRASIS.findall(nuevo)) > len(_RX_PERIFRASIS.findall(viejo)):
        return "sustituye un dato por una remisión genérica a las constancias"
    if da_por_cierto(viejo, nuevo):
        return "borra la atribución a la parte: da por cierto lo que sólo se afirma"
    pre_v = _RX_PREFIJO_ORDEN.match(viejo)
    if pre_v:
        pre_n = _RX_PREFIJO_ORDEN.match(nuevo)
        if not pre_n or re.sub(r"\s", "", pre_n.group(1)) != re.sub(r"\s", "", pre_v.group(1)):
            return "pierde el número del antecedente"
    return ""


def aplicar_antecedentes(antecedentes, parches: list, bloques_: list):
    """Los antecedentes con los parches aprobados, en la misma forma en que
    llegaron (lista o cadena). Lo no tocado queda byte por byte."""
    es_cadena = isinstance(antecedentes, str)
    lista = antecedentes.split("\n") if es_cadena else list(antecedentes or [])
    por_n = {b["n"]: b for b in bloques_}
    nuevas, quitar = {}, set()
    for p in parches:
        b = por_n[p["parrafo"]]
        if p["accion"] == "eliminar":
            quitar.add(b["indice"])
            if es_cadena and b["indice"] + 1 < len(lista) and not lista[b["indice"] + 1].strip():
                quitar.add(b["indice"] + 1)
        else:
            nuevas[b["indice"]] = p["texto"]
    fuera = [nuevas.get(k, x) for k, x in enumerate(lista) if k not in quitar]
    return "\n".join(fuera) if es_cadena else fuera


def hallazgos_antecedentes(bloques_: list, fuentes=None) -> list:
    """Lo que el código ya sabe de los antecedentes, situado por párrafo A."""
    fuera = []
    try:
        import fechas_en_autos as _fe
    except Exception:
        _fe = None
    # Las fechas de las fuentes, UNA vez (`sin_respaldo` las recalcula en cada
    # llamada: 0.4 s sobre el 631 por párrafo a párrafo).
    en_autos = _fechas("\n".join(str(x or "") for x in (fuentes or []))) if _fe is not None else set()
    for b in bloques_:
        if b["fijo"]:
            continue
        t, n = b["texto"], b["n"]
        for frase, _ in frases_de_herramienta(t):
            fuera.append(f"[A{n}] frase de herramienta, un tribunal no la escribe: «{frase}»")
        for m in _RX_PERIFRASIS.finditer(t):
            fuera.append(f"[A{n}] remisión genérica en lugar del dato: «{m.group(0)}»")
        if en_autos:
            for f in [ap for k, ap in _fe.fechas_de(t).items() if k not in en_autos][:3]:
                fuera.append(f"[A{n}] fecha que no consta en el acto, el escrito ni las constancias: «{f}»")
        otros = [o["texto"] for o in bloques_ if o["n"] != n]
        j = repite_en(t, otros)
        if j >= 0:
            fuera.append(f"[A{n}] repite lo que ya dice otro párrafo de los antecedentes")
    return fuera[:20]


_RX_FIRMA_ELECTRONICA = re.compile(
    r"EVIDENCIA\s+CRIPTOGR[ÁA]FICA|FIRMA\s+ELECTR[ÓO]NICA\s+AVANZADA|Cadena\s+de\s+firma|"
    r"\bOCSP\b|Datos\s+estampillados|Nombre\s+del\s+respondedor", re.I)
_RX_RESUELVE = re.compile(r"\bR\s?E\s?S\s?U\s?E\s?L\s?V\s?E\b|\bPUNTOS\s+RESOLUTIVOS\b", re.I)


def extracto_del_acto(acto: str, antecedentes=None, cabeza: int = None, cola: int = None) -> str:
    """Lo que del acto ve el revisor: los pasajes donde constan las fechas y
    los números que citan los antecedentes, y el final (los puntos
    resolutivos). Sin antecedentes, o si ninguno de sus datos está en el acto,
    el principio y el final.

    POR QUÉ NO SÓLO EL PRINCIPIO (medido el 3-oct-2026 sobre el AR 631/2025):
    el texto del PDF trae una palabra por renglón en el proemio, y en una
    revisión el relato del juicio de origen está a media sentencia, no en sus
    resultandos: los primeros 7,000 caracteres no llegaban a él. Se compacta
    el espacio y se va a donde están los datos."""
    cabeza = EXTRACTO_CABEZA if cabeza is None else cabeza
    cola = EXTRACTO_COLA if cola is None else cola
    t = " ".join(str(acto or "").split())
    # LA FIRMA ELECTRÓNICA NO ES EL FINAL (mismo día, mismo AR): tras los
    # resolutivos el PDF trae la evidencia criptográfica —cadenas en
    # hexadecimal, OCSP, sellos de tiempo— y la cola caía entera ahí.
    m_firma = _RX_FIRMA_ELECTRONICA.search(t, len(t) // 2)
    if m_firma:
        t = t[:m_firma.start()].rstrip()
    if len(t) <= cabeza + cola + 200:
        return t
    m_res = None
    for m_res in _RX_RESUELVE.finditer(t):
        pass
    final = (t[m_res.start():m_res.start() + cola] if m_res and len(t) - m_res.start() > 200
             else t[-cola:].split(" ", 1)[-1])
    lugares = []
    if antecedentes:
        txt_a = antecedentes if isinstance(antecedentes, str) else "\n".join(str(x) for x in antecedentes)
        try:
            import fechas_en_autos as _fe
            del_acto = _fe.fechas_de(t)
            for clave in _fe.fechas_de(txt_a):
                if clave in del_acto:
                    k = t.find(del_acto[clave])
                    if k >= 0:
                        lugares.append(k)
        except Exception:
            pass
        for na in numeros_con_anio(txt_a):
            k = t.find(na)
            if k >= 0:
                lugares.append(k)
    if not lugares:
        return t[:cabeza].rsplit(" ", 1)[0] + " […] " + final
    # VENTANAS ALREDEDOR DE CADA DATO, fundidas y en el orden del acto, hasta
    # el presupuesto de la cabeza.
    tramos = []
    lugares = sorted(set(lugares))
    ancho = max(500, min(1100, cabeza // len(lugares)))
    for k in lugares:
        a, b = max(0, k - int(ancho * 0.4)), min(len(t) - cola, k + int(ancho * 0.6))
        if b <= a:
            continue
        if tramos and a <= tramos[-1][1]:
            tramos[-1][1] = max(tramos[-1][1], b)
        else:
            tramos.append([a, b])
    fuera, usados = [], 0
    for a, b in tramos:
        if usados >= cabeza:
            break
        b = min(b, a + (cabeza - usados))
        trozo = t[a:b]
        if a > 0:
            trozo = trozo.split(" ", 1)[-1]
        fuera.append(trozo.rsplit(" ", 1)[0] if b < len(t) else trozo)
        usados += b - a
    return " […] ".join(fuera) + " […] " + final


# ═══ EL PARSEO ══════════════════════════════════════════════════════════════
def _objetos_completos(t: str) -> list:
    """Los objetos {…} COMPLETOS de la lista «parches» de una salida cortada.

    MEDIDO (2-oct-2026, AR 631/2025): con razonamiento alto, gemini-3.8-flash
    gastó 38,401 de los 40,000 tokens de salida en pensar y la respuesta se
    cortó a media lista. Lo que llegó entero vale: se rescata objeto por
    objeto, respetando las cadenas, y el que quedó a medias se pierde."""
    k = t.find("[", max(0, t.find('"parches"')))
    if k < 0:
        return []
    fuera, prof, ini, en_cad, esc = [], 0, None, False, False
    for i in range(k + 1, len(t)):
        c = t[i]
        if en_cad:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                en_cad = False
            continue
        if c == '"':
            en_cad = True
        elif c == "{":
            if prof == 0:
                ini = i
            prof += 1
        elif c == "}" and prof > 0:
            prof -= 1
            if prof == 0 and ini is not None:
                try:
                    fuera.append(json.loads(t[ini:i + 1]))
                except Exception:
                    pass
                ini = None
        elif c == "]" and prof == 0:
            break
    return fuera


def _crudos(salida: str):
    """La lista «parches» tal como llegó, o None si la salida no la trae
    legible (vacía, sin JSON, sin la clave, o cortada antes del primer parche
    entero)."""
    t = (salida or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", t)
    datos = None
    try:
        datos = json.loads(t)
    except Exception:
        a, b = t.find("{"), t.rfind("}")
        if a >= 0 and b > a:
            try:
                datos = json.loads(t[a:b + 1])
            except Exception:
                datos = None
    if datos is None and '"parches"' in t:
        rescatados = _objetos_completos(t)
        datos = {"parches": rescatados} if rescatados else None
    if isinstance(datos, dict):
        crudos = datos.get("parches")
    elif isinstance(datos, list):
        crudos = datos
    else:
        return None
    return crudos if isinstance(crudos, list) else None


def salida_legible(salida: str) -> str:
    """«» si la salida trae la lista de parches (aunque venga vacía: «no hay
    nada que corregir» es una respuesta); «sin_respuesta» si llegó vacía;
    «ilegible» si llegó algo que no es esa lista.

    POR QUÉ (revisión adversarial, 3-oct-2026): Gemini devuelve `text=None`
    cuando bloquea el candidato (seguridad, recitación: posible con
    expedientes penales) y la salida puede cortarse antes del primer objeto
    entero. `parsear` da [] en los dos casos, igual que cuando el revisor no
    encontró nada, y la pantalla decía «El revisor leyó el proyecto y no
    encontró nada que corregir»: una garantía falsa. Ahora es «fallo»."""
    if not (salida or "").strip():
        return "sin_respuesta"
    return "" if _crudos(salida) is not None else "ilegible"


def _numero_de_parrafo(x) -> int:
    """El número de UN párrafo («3», 3, «P3», «párrafo 3», 3.0), o 0 si no es
    un solo entero. Antes se juntaban todos los dígitos y «3-4» (una fusión
    que el modelo propone a veces) se leía como el párrafo 34, que suele
    existir en un estudio v4: su contenido se sustituía por otro (revisión
    adversarial, 3-oct-2026)."""
    if isinstance(x, bool):
        return 0
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x) if x.is_integer() else 0
    m = re.fullmatch(r"\s*(?:P|p[áa]rrafo)?\s*\.?\s*(\d{1,4})(?:\.0+)?\s*", str(x or ""), re.I)
    return int(m.group(1)) if m else 0


_RX_ETIQUETA_A = re.compile(r"\s*(?:A|antecedentes?\s*)\s*\.?\s*(\d{1,4})\s*", re.I)


def seccion_y_parrafo(x, seccion=None) -> tuple:
    """(«estudio»|«antecedentes», número) de la etiqueta de un parche: «A3»
    es el tercer párrafo de los antecedentes; «P3», «3» o 3, el del estudio.
    Una clave «seccion» que diga antecedentes también vale (3-oct-2026)."""
    if isinstance(x, str):
        m = _RX_ETIQUETA_A.fullmatch(x)
        if m:
            return "antecedentes", int(m.group(1))
    s = str(seccion or "").strip().lower()
    n = _numero_de_parrafo(x)
    if s.startswith("antecedente") or s == "a":
        return "antecedentes", n
    return "estudio", n


def parsear(salida: str) -> tuple:
    """([parche normalizado], n_mal_formados). Nunca lanza. Cada parche lleva
    su «seccion»: «estudio» o «antecedentes»."""
    crudos = _crudos(salida)
    if crudos is None:
        return [], 0
    fuera, malos = [], 0
    for c in crudos[:MAX_PARCHES * 2]:
        if not isinstance(c, dict):
            malos += 1
            continue
        sec, n = seccion_y_parrafo(c.get("parrafo"), c.get("seccion"))
        accion = str(c.get("accion") or "").strip().lower()
        tipo = str(c.get("tipo") or "").strip().lower().replace(" ", "_")
        tipo = tipo.replace("jurídico", "juridico").replace("repetición", "repeticion") \
                   .replace("extensión", "extension").replace("redacción", "redaccion")
        if tipo not in TIPOS:
            tipo = "redaccion"
        texto = " ".join(str(c.get("texto") or "").split())
        if n <= 0 or accion not in ACCIONES or (accion != "eliminar" and not texto):
            malos += 1
            continue
        fuera.append({"parrafo": n, "accion": accion, "tipo": tipo, "texto": texto,
                      "motivo": " ".join(str(c.get("motivo") or "").split())[:300],
                      "seccion": sec})
    return fuera[:MAX_PARCHES], malos


# ═══ LA APLICACIÓN ══════════════════════════════════════════════════════════
def aplicar(estudio: str, parches: list, bloques_: list) -> str:
    """El estudio con los parches ya aprobados. Cada reemplazo conserva las
    marcas del renglón, delante del texto nuevo; un eliminado se va con el
    blanco que lo seguía. Nada más cambia."""
    lineas = (estudio or "").split("\n")
    por_n = {b["n"]: b for b in bloques_}
    nuevas = {}
    quitar = set()
    for p in parches:
        b = por_n[p["parrafo"]]
        if p["accion"] == "eliminar":
            quitar.add(b["linea"])
            if b["linea"] + 1 < len(lineas) and not lineas[b["linea"] + 1].strip():
                quitar.add(b["linea"] + 1)
        else:
            pref = " ".join(b["marcas"])
            nuevas[b["linea"]] = (pref + " " if pref else "") + p["texto"]
    fuera = []
    for k, ln in enumerate(lineas):
        if k in quitar:
            continue
        fuera.append(nuevas.get(k, ln))
    return "\n".join(fuera)


# ═══ LOS HALLAZGOS DETERMINISTAS: «CORRIGE ESTO» ════════════════════════════
def _donde(bloques_: list, fragmento: str) -> int:
    """El párrafo donde aparece un fragmento de aviso (0 si no se halla)."""
    f = _plano(re.sub(r"[…]|\.\.\.", " ", fragmento or ""))
    if len(f) < 12:
        return 0
    medio = f[max(0, len(f) // 2 - 20):len(f) // 2 + 20]
    for b in bloques_:
        if medio and medio in _plano(b["texto"]):
            return b["n"]
    return 0


def hallazgos(estudio: str, bloques_: list, fuentes=None, n_planteamientos: int = 0,
              objetivo: int = TECHO_PALABRAS) -> list:
    """Lo que los controles sin modelo ya saben del estudio y hoy sólo avisan.
    [str], cada uno con el párrafo cuando se puede situar. Nunca lanza."""
    import marcas as _mc
    limpio = _mc.sin_marcas(estudio or "")
    fuera = []

    def _con(n, txt):
        fuera.append((f"[P{n}] " if n else "") + txt)

    for frase, ctx in frases_de_herramienta(limpio):
        _con(_donde(bloques_, ctx), f"frase de herramienta, un tribunal no la escribe: «{frase}»")
    try:
        import linter_juridico as _lj
        for que, donde in _lj.revisar(limpio):
            _con(_donde(bloques_, donde), f"{que}" + (f" — «{donde[:140]}»" if donde else ""))
    except Exception:
        pass
    try:
        import fechas_en_autos as _fe
        for f in _fe.sin_respaldo(limpio, fuentes or [])[:6]:
            _con(_donde(bloques_, f), f"fecha que no consta en ningún documento del asunto: «{f}»")
    except Exception:
        pass
    try:
        import calidad_estudio as _ce
        for d in _ce.duplicacion_interna(limpio)[:6]:
            _con(_donde(bloques_, d), f"pasaje repetido dentro del estudio: «{d[:140]}»")
        for r in _ce.remisiones_rotas(limpio)[:4]:
            _con(_donde(bloques_, r), f"remisión a un apartado que no existe: «{r}»")
        den = _ce.densidad(limpio)
        if den.get("palabras", 0) > 400 and den.get("pct_propio", 1) < 0.70:
            _con(0, f"el estudio vive de la cita: sólo el {den['pct_propio'] * 100:.0f}% es "
                    f"razonamiento propio")
    except Exception:
        pass
    try:
        import formato_sentencia as _fs
        falt = _fs.sin_contestar(limpio, n_planteamientos or 0)
        if falt:
            _con(0, "conceptos o agravios que el estudio no nombra por su ordinal: "
                    + ", ".join(str(x) for x in falt[:8]))
    except Exception:
        pass
    pal = _palabras(limpio)
    if pal > objetivo:
        _con(0, f"extensión: {pal} palabras; la referencia es {objetivo}. Condensa lo que "
                f"se repite o se demora sin dejar de contestar ningún argumento")
    return fuera[:40]


# ═══ EL PROMPT ══════════════════════════════════════════════════════════════
def _recorta(t: str, n: int) -> str:
    t = (t or "").strip()
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " […]"


def _bloque_criterio(criterios) -> str:
    L = []
    for i, c in enumerate(criterios or [], 1):
        s = str(_get(c, "sentido") or "").strip().replace("_", " ").upper() or "SIN CALIFICAR"
        L.append(f"{i}. [{str(_get(c, 'jerarquia') or 'accesorio').upper()}] "
                 f"{_get(c, 'problema') or ''}\n   SENTIDO: {s}")
        razon = _recorta(" ".join(str(_get(c, "razonamiento") or "").split()), 1500)
        if razon:
            L.append(f"   RAZÓN DEL SECRETARIO: {razon}")
    return "\n".join(L) or "(sin criterio)"


def _bloque_catalogo(material) -> str:
    L = []
    for t in list(_get(material, "tesis", None) or [])[:40]:
        if isinstance(t, dict) and t.get("registro"):
            # EL RUBRO ENTERO Y LA VIGENCIA (3-oct-2026): un rubro nuevo sólo
            # pasa la guarda si es, completo, el de una tesis del material, y
            # el modelo tiene que ver cuál perdió vigencia, como la ve el estudio.
            _sv = ""
            try:
                import fuerza_juridica as _fj
                _sv = " [SIN VIGENCIA: no se cita]" if _fj.sello_perdio_vigencia(t.get("vigencia")) else ""
            except Exception:
                pass
            L.append(f"· registro {t.get('registro')}{_sv}: {_recorta(str(t.get('rubro') or ''), 900)}")
    N = []
    for n in list(_get(material, "normas", None) or [])[:60]:
        if isinstance(n, dict) and n.get("articulo"):
            N.append(f"· artículo {n.get('articulo')} — "
                     f"{_recorta(str(n.get('cuerpo_legal') or n.get('fuente') or ''), 120)}")
    return ("TESIS Y JURISPRUDENCIA DEL MATERIAL:\n" + ("\n".join(L) or "(ninguna)")
            + "\n\nNORMAS DEL MATERIAL:\n" + ("\n".join(N) or "(ninguna)"))


def _bloque_estudio(bloques_: list) -> str:
    L = []
    for b in bloques_:
        etiqueta = f"P{b['n']}"
        if b["fijo"]:
            etiqueta += f" · FIJO ({b['fijo']}): no se toca"
        elif b["ids"]:
            etiqueta += f" · contesta {len(set(b['ids']))} argumento(s): no se elimina"
        L.append(f"[{etiqueta}]\n{b['texto']}")
    return "\n\n".join(L)


def _bloque_antecedentes(bloques_ant: list) -> str:
    """Los párrafos A. El trámite en este tribunal NO se enseña: es el bloque
    procesal, que sale por construcción."""
    L = []
    for b in bloques_ant:
        if b["fijo"] == "trámite en este tribunal":
            continue
        etiqueta = f"A{b['n']}" + (f" · FIJO ({b['fijo']}): no se toca" if b["fijo"] else "")
        L.append(f"[{etiqueta}]\n{_recorta(b['texto'], 600) if b['fijo'] else b['texto']}")
    return "\n\n".join(L)


def _seccion_antecedentes(hallazgos_ant: list, extracto_acto: str) -> str:
    """Lo que se le pide de los antecedentes, en pocas líneas y sin frases
    modelo (un ejemplo literal en un prompt se copia tal cual)."""
    hall = "\n".join(f"- {h}" for h in (hallazgos_ant or [])) or "- (ninguno)"
    return f"""═══ LOS ANTECEDENTES (párrafos A, al final) ═══
Narran el juicio de origen y lo que resolvió cada instancia. Se cotejan con el extracto de la resolución que va abajo. El trámite ante este tribunal no se te enseña y no se toca.
Qué buscar en ellos:
- incongruencia: una fecha, un número de expediente o de toca, un órgano o una parte que no coincide con la resolución.
- hecho_no_acreditado: se da por cierto lo que sólo afirmó una parte en su demanda, su contestación o su recurso; atribúyelo a quien lo afirmó.
- repeticion: el mismo hecho contado dos veces.
- redaccion o extension: oraciones kilométricas, detalle de trámite que no sirve para entender lo que se resuelve.
Reglas propias de los antecedentes:
- Sólo fechas, números, nombres y artículos que ya estén en el párrafo o en la resolución, el escrito o las constancias.
- No cambies lo que cada instancia resolvió ni lo que tuvo por acreditado.
- No alargues un párrafo más de un 20%, por ningún motivo; al menos seis palabras.
- eliminar sólo procede, con tipo repeticion, en un párrafo que no dice nada que no diga ya otro párrafo de los antecedentes.
- Nunca dejes un dato en blanco ni lo sustituyas por una remisión genérica a las constancias o a los autos.
Hallazgos de los controles automáticos en los antecedentes:
{hall}

═══ EXTRACTO DE LA RESOLUCIÓN RECLAMADA O RECURRIDA (su principio y su final) ═══
{extracto_acto or "(no consta)"}

"""


def prompt(bloques_: list, criterios, material, resumen_acto: str, resumen_conceptos: str,
           hallazgos_: list, palabras: int, objetivo: int = TECHO_PALABRAS,
           bloques_ant: list = None, hallazgos_ant: list = None, extracto_acto: str = "") -> str:
    """El encargo del supervisor. Describe la forma de los parches; no trae
    frases modelo (un ejemplo literal en un prompt se copia tal cual). Sin
    `bloques_ant` el texto es, byte por byte, el de antes del 3-oct-2026."""
    hall = "\n".join(f"- {h}" for h in hallazgos_) or "- (ninguno)"
    con_ant = bool(bloques_ant and any(b["fijo"] != "trámite en este tribunal" for b in bloques_ant))
    objeto = "el ESTUDIO DE FONDO y los ANTECEDENTES" if con_ant else "el ESTUDIO DE FONDO"
    seccion_ant = _seccion_antecedentes(hallazgos_ant, extracto_acto) if con_ant else ""
    clave_parrafo = ('el número del párrafo: en el estudio, el entero de su etiqueta P; en los antecedentes, '
                     'la etiqueta entera con su A ("A3").' if con_ant
                     else "el número del párrafo (el entero de su etiqueta P).")
    cola_ant = ("\n═══ LOS ANTECEDENTES, POR PÁRRAFOS ═══\n" + _bloque_antecedentes(bloques_ant) + "\n"
                if con_ant else "")
    return f"""Eres el secretario proyectista que revisa, antes de pasarlo al magistrado, {objeto} de un proyecto de sentencia de un Tribunal Colegiado de Circuito de México. No lo reescribes: propones correcciones puntuales, párrafo por párrafo, y sólo donde hacen falta.

EL CRITERIO DEL SECRETARIO ES INMUTABLE. El sentido de cada problema ya está decidido; ninguna corrección cambia una calificación (fundado, infundado, inoperante, ineficaz, inatendible, fundado pero insuficiente, innecesario, sin materia) ni el desenlace (conceder o negar el amparo; revocar, confirmar o modificar; reponer el procedimiento; «para efectos»; sobreseer).

═══ CRITERIO DEL SECRETARIO ═══
{_bloque_criterio(criterios)}

═══ LO QUE RESOLVIÓ LA RESOLUCIÓN RECLAMADA O RECURRIDA (lo que se tuvo por acreditado) ═══
{_recorta(resumen_acto, 15000) or "(no consta)"}

═══ LO QUE ALEGA LA PARTE (conceptos de violación o agravios) ═══
{_recorta(resumen_conceptos, 12000) or "(no consta)"}

═══ CATÁLOGO CERRADO DE CITAS ═══
Ninguna corrección puede citar un registro, una clave de tesis, un expediente o un artículo que no esté en este catálogo o en el propio párrafo que corriges.
{_bloque_catalogo(material)}

═══ HALLAZGOS DE LOS CONTROLES AUTOMÁTICOS: corrígelos ═══
{hall}

═══ QUÉ BUSCAR ═══
1. error_juridico: una afirmación de derecho equivocada, un precepto aplicado fuera de su supuesto, una regla procesal mal enunciada.
2. incongruencia: un párrafo que contradice a otro, a su propia calificación o al criterio del secretario.
3. hecho_no_acreditado: el estudio da por cierto lo que sólo afirma la parte. Lo que dicen los agravios o los conceptos de violación es justo lo que se verifica; un hecho sólo se afirma como cierto si la resolución reclamada o recurrida lo tuvo por acreditado. Corrige atribuyéndolo a quien lo afirma o apoyándolo en lo que la resolución sí tuvo por acreditado; nunca añadas un hecho nuevo.
4. cita: una tesis o un precepto invocado para algo que no sostiene. No quites ni cambies los registros ni los rubros que el párrafo ya cita: corrige la afirmación que se apoya en ellos. Una cita que añadas sale del catálogo, con su rubro entero, y nunca de una tesis sin vigencia.
5. repeticion: el mismo razonamiento dicho dos veces. Conserva la versión más completa y condensa o elimina la otra.
6. extension: el estudio tiene {palabras} palabras y la referencia es {objetivo}. Si pasa de ella, condensa los párrafos que se demoran, repiten o glosan, sin perder ningún argumento contestado: la brevedad no vale si cuesta exhaustividad.
7. redaccion: frases que un tribunal no escribe (las que hablan del conjunto de documentos como algo que a quien redacta se le entregó o remitió para el estudio, en vez de referirse a las constancias de autos o a la resolución), barroquismos, frases cortadas, oraciones kilométricas.

═══ REGLAS DE CADA PARCHE ═══
- Un parche por párrafo, como mucho. Si un párrafo está bien, no lo toques.
- Los párrafos marcados FIJO no se tocan. Los que «contesta N argumento(s)» no se eliminan: se condensan conservando la respuesta a cada argumento.
- Si el párrafo enumera los argumentos, temas o preceptos que contesta, la versión condensada los conserva todos: se acorta la explicación, nunca la lista de lo contestado.
- Lo que el párrafo atribuye a una parte (lo que sostiene, alega o reproduce en su escrito) sigue atribuido a ella en el texto nuevo.
- Conserva en el texto nuevo, literalmente: la calificación y su dirección, los registros, las claves de tesis, los artículos, las fechas, los rubros entre comillas, el ordinal del concepto o agravio que se contesta y el número o inciso con que empieza una orden de efectos.
- No añadas registros, artículos, fechas, transcripciones ni hechos que no estén en el párrafo o en el catálogo.
- El texto nuevo es el párrafo ENTERO tal como debe quedar, en un solo renglón, con al menos seis palabras, en el registro del estudio (el tribunal habla en tercera persona). No escribas marcas entre ⟦ ⟧ ni identificadores internos.
- No alargues un párrafo más de un 20%, salvo para corregir un error jurídico o una incongruencia.
- eliminar sólo procede en un párrafo que no cite nada, no califique y no conteste argumentos, cuando lo que dice ya está dicho en otro.

{seccion_ant}═══ FORMATO DE LA RESPUESTA ═══
Sólo un objeto JSON con la clave "parches": una lista de objetos con estas claves:
- "parrafo": {clave_parrafo}
- "accion": "reemplazar" (corrige el contenido), "condensar" (mismo contenido, más corto) o "eliminar".
- "tipo": uno de error_juridico, incongruencia, hecho_no_acreditado, cita, repeticion, extension, redaccion.
- "texto": el párrafo nuevo completo; cadena vacía si se elimina.
- "motivo": una frase que diga qué estaba mal.
Si no hay nada que corregir, la lista va vacía.

═══ EL ESTUDIO, POR PÁRRAFOS ═══
{_bloque_estudio(bloques_)}
{cola_ant}"""


# ═══ LAS LLAMADAS ═══════════════════════════════════════════════════════════
_CLIENTE_GEMINI = None


def _cliente_gemini():
    """El cliente propio de AI Studio (GEMINI_API_KEY), como `get_gemini_client`
    de main.py: no se importa main desde aquí."""
    global _CLIENTE_GEMINI
    if _CLIENTE_GEMINI is None:
        clave = os.getenv("GEMINI_API_KEY", "")
        if not clave:
            raise RuntimeError("GEMINI_API_KEY no configurada")
        from google import genai
        _CLIENTE_GEMINI = genai.Client(api_key=clave)
    return _CLIENTE_GEMINI


def _config_gemini():
    from google.genai import types
    raz = (SUPERVISOR_RAZONAMIENTO or "high").strip()
    if raz.isdigit():
        tc = types.ThinkingConfig(thinking_budget=int(raz))
    else:
        tc = types.ThinkingConfig(thinking_level=raz.upper())
    return types.GenerateContentConfig(
        temperature=SUPERVISOR_TEMPERATURA,
        max_output_tokens=SUPERVISOR_MAX_SALIDA,
        response_mime_type="application/json",
        thinking_config=tc)


async def _llamar_gemini(texto_prompt: str, tope_s: float) -> tuple:
    """(salida, uso, modelo). Lanza si falla; `asyncio.TimeoutError` si vence."""
    cli = _cliente_gemini()
    # TURNO Y REINTENTOS (3-oct-2026, `nivel_servicio`): con veinte asuntos a
    # la vez, o la corrida nocturna, Gemini contesta 429/503; se reintenta con
    # espera dentro del mismo tope en vez de entregar el estudio sin revisar.
    import nivel_servicio as _ns
    r = await _ns.llamar_gemini(
        lambda: cli.aio.models.generate_content(model=SUPERVISOR_MODELO, contents=texto_prompt,
                                                config=_config_gemini()),
        tope_s, etiqueta="supervisor")
    um = getattr(r, "usage_metadata", None)
    uso = {}
    for k_s, k_e in (("entrada", "prompt_token_count"), ("salida", "candidates_token_count"),
                     ("razonamiento", "thoughts_token_count"), ("total", "total_token_count")):
        v = getattr(um, k_e, None) if um is not None else None
        if isinstance(v, (int, float)):
            uso[k_s] = int(v)
    return (getattr(r, "text", "") or ""), uso, SUPERVISOR_MODELO


async def _llamar_respaldo(cliente, texto_prompt: str, tope_s: float) -> tuple:
    """(salida, uso, modelo) por el chat_client del resolver. SIN la petición
    de respaldo de `llamada_modelo.crear`: al vencer `wait_for` quedaría viva
    una tarea huérfana (la misma razón que `exhaustivo.reparar`)."""
    import llamada_modelo as _lm
    kw = dict(model=SUPERVISOR_RESPALDO, max_completion_tokens=24000,
              messages=[{"role": "user", "content": texto_prompt}],
              reasoning_effort=SUPERVISOR_RESPALDO_ESFUERZO,
              response_format={"type": "json_object"})
    r = await asyncio.wait_for(_lm._crear_una(cliente, **kw), timeout=tope_s)
    uso = {}
    try:
        import fase6_estudio as _f6
        uso = _f6._uso_de(getattr(r, "usage", None))
    except Exception:
        pass
    return (r.choices[0].message.content or ""), uso, SUPERVISOR_RESPALDO


def coste_usd(uso: dict):
    """El coste si se conoce el precio (variables de entorno); None si no."""
    try:
        pe, ps = float(_PRECIO_ENTRADA), float(_PRECIO_SALIDA)
    except (TypeError, ValueError):
        return None
    sal = int(uso.get("salida") or 0) + int(uso.get("razonamiento") or 0)
    return round((int(uso.get("entrada") or 0) * pe + sal * ps) / 1e6, 5)


# ═══ EL SUPERVISOR ══════════════════════════════════════════════════════════
def _informe_vacio(estado: str = "sin_cambios") -> dict:
    return {"estado": estado, "modelo": "", "segundos": 0.0, "correcciones": [],
            "descartadas": 0}


async def supervisar(estudio: str, criterios=None, material=None, resumen_acto: str = "",
                     resumen_conceptos: str = "", fuentes=None, cliente=None,
                     tope_s: float = None, llamar=None, plan=None) -> tuple:
    """(estudio, informe CONTRATO D). El estudio sale COMO ESTABA si la llamada
    falla o vence, o si ningún parche pasa las guardas.

    `llamar` se inyecta en las pruebas: async (prompt, tope_s) → (salida, uso,
    modelo). En producción, Gemini y, si no abre, el respaldo por `cliente`.
    `plan` es el plan v4 con que se escribió el estudio (o None): sus
    etiquetas dicen qué calificación corresponde a cada argumento marcado."""
    nuevo, _a, inf = await _supervisar(estudio, None, criterios, material, resumen_acto,
                                       resumen_conceptos, fuentes, None, "", cliente, tope_s,
                                       llamar, plan)
    return nuevo, inf


async def supervisar_con_antecedentes(estudio: str, antecedentes, criterios=None, material=None,
                                      resumen_acto: str = "", resumen_conceptos: str = "",
                                      fuentes=None, fuentes_antecedentes=None,
                                      extracto_acto: str = "", cliente=None, tope_s: float = None,
                                      llamar=None, plan=None) -> tuple:
    """(estudio, antecedentes, informe CONTRATO D) en UNA llamada (3-oct-2026).

    `antecedentes`: la lista de párrafos que llevará el proyecto (o una cadena;
    sale en la misma forma). `fuentes_antecedentes`: los textos que anclan sus
    parches —el acto, el escrito, las constancias, lo que aportó el
    secretario—, NUNCA los resúmenes del modelo. `extracto_acto`: lo que del
    acto ve el revisor (`extracto_del_acto`). Si la llamada falla o vence, los
    dos salen como estaban. Las correcciones de los antecedentes van en
    `informe["correcciones"]` con «seccion»: «antecedentes» y su «bloque»
    («A3»); su detalle, en `informe["antecedentes"]`."""
    return await _supervisar(estudio, antecedentes, criterios, material, resumen_acto,
                             resumen_conceptos, fuentes, fuentes_antecedentes, extracto_acto,
                             cliente, tope_s, llamar, plan)


def _revisar_antecedentes(parches: list, bloques_: list, anclas: AnclasAntecedentes) -> tuple:
    """([parches aprobados], [motivos de descarte]) de los antecedentes. Los
    reemplazos van primero; las eliminaciones después, contra el texto ya
    corregido y sin que un párrafo borrado sirva de testigo de otro."""
    por_n = {b["n"]: b for b in bloques_}
    textos = {b["n"]: b["texto"] for b in bloques_}
    motivos, primeros, vistos = [], [], set()
    for p in parches:
        if p["parrafo"] in vistos:
            motivos.append({"parrafo": p["parrafo"], "tipo": p["tipo"], "motivo": "segundo parche al mismo párrafo",
                            "accion": p["accion"], "texto": p.get("texto", "")})
            continue
        vistos.add(p["parrafo"])
        primeros.append(p)
    buenos, eliminados = [], set()
    max_elim = max(1, int(len([b for b in bloques_ if not b["fijo"]]) * MAX_ELIMINAR))
    for p in [x for x in primeros if x["accion"] != "eliminar"] + \
             [x for x in primeros if x["accion"] == "eliminar"]:
        b = por_n.get(p["parrafo"])
        otros = [t for k, t in textos.items() if k != p["parrafo"] and k not in eliminados]
        motivo = guardas_antecedente(p, b, anclas, otros)
        if not motivo and p["accion"] == "eliminar" and len(eliminados) >= max_elim:
            motivo = "demasiadas eliminaciones en una pasada"
        if not motivo and p["accion"] != "eliminar" \
                and " ".join(p["texto"].split()) == " ".join(b["texto"].split()):
            motivo = "sin cambio"
        if motivo:
            motivos.append({"parrafo": p["parrafo"], "tipo": p["tipo"], "motivo": motivo,
                            "accion": p["accion"], "texto": p.get("texto", "")})
            continue
        if p["accion"] == "eliminar":
            eliminados.add(p["parrafo"])
        else:
            textos[p["parrafo"]] = p["texto"]
        buenos.append(p)
    buenos.sort(key=lambda x: x["parrafo"])
    return buenos, motivos


async def _supervisar(estudio, antecedentes, criterios, material, resumen_acto, resumen_conceptos,
                      fuentes, fuentes_antecedentes, extracto_acto, cliente, tope_s, llamar,
                      plan) -> tuple:
    """El supervisor entero: (estudio, antecedentes, informe). Sin
    antecedentes (None) hace exactamente lo de antes del 3-oct-2026."""
    t0 = time.perf_counter()
    tope = float(tope_s or SUPERVISOR_TOPE_S)
    inf = _informe_vacio()
    bl = bloques(estudio)
    ba = bloques_antecedentes(antecedentes) if antecedentes is not None else []
    ba_vivos = [b for b in ba if not b["fijo"]]
    if antecedentes is not None:
        inf["antecedentes"] = {"estado": "sin_cambios", "parrafos": len(ba)}
    if not [b for b in bl if not b["fijo"]] and not ba_vivos:
        inf["estado"] = "sin_cambios"
        return estudio, antecedentes, inf
    try:
        import fase6_estudio as _f6o
        objetivo = max(TECHO_PALABRAS, int(_f6o._objetivo_palabras(material, criterios) or 0)) \
            if material is not None else TECHO_PALABRAS
    except Exception:
        objetivo = TECHO_PALABRAS
    objetivo = min(objetivo, 4000)
    fuentes = [x for x in (fuentes or []) if x] + [resumen_acto or "", resumen_conceptos or ""]
    hall = hallazgos(estudio, bl, fuentes, int(_get(material, "n_planteamientos", 0) or 0), objetivo)
    fuentes_ant = [str(x) for x in (fuentes_antecedentes or []) if x]
    hall_a = hallazgos_antecedentes(ba, fuentes_ant) if ba_vivos else []
    import marcas as _mc
    pal_antes = _palabras(_mc.sin_marcas(estudio))
    texto_prompt = prompt(bl, criterios, material, resumen_acto, resumen_conceptos, hall,
                          pal_antes, objetivo, bloques_ant=ba if ba_vivos else None, hallazgos_ant=hall_a,
                          extracto_acto=extracto_acto)
    salida, uso, modelo = "", {}, ""
    try:
        if llamar is not None:
            salida, uso, modelo = await asyncio.wait_for(llamar(texto_prompt, tope), timeout=tope)
        else:
            try:
                salida, uso, modelo = await _llamar_gemini(texto_prompt, tope)
            except asyncio.TimeoutError:
                raise
            except Exception as ex_g:
                resta = tope - (time.perf_counter() - t0)
                if cliente is None or resta < _MIN_RESPALDO_S:
                    raise
                print(f"   🔎 SUPERVISOR: Gemini no abrió ({type(ex_g).__name__}); "
                      f"respaldo {SUPERVISOR_RESPALDO}")
                inf["error_principal"] = type(ex_g).__name__
                salida, uso, modelo = await _llamar_respaldo(cliente, texto_prompt, resta)
    except asyncio.TimeoutError:
        inf.update(estado="vencido", modelo=modelo or SUPERVISOR_MODELO,
                   segundos=round(time.perf_counter() - t0, 1))
        if "antecedentes" in inf:
            inf["antecedentes"]["estado"] = "vencido"
        return estudio, antecedentes, inf
    except Exception as ex:
        inf.update(estado="fallo", modelo=modelo or SUPERVISOR_MODELO, error=type(ex).__name__,
                   segundos=round(time.perf_counter() - t0, 1))
        if "antecedentes" in inf:
            inf["antecedentes"]["estado"] = "fallo"
        return estudio, antecedentes, inf
    inf["modelo"] = modelo
    inf["uso"] = uso
    _c = coste_usd(uso)
    if _c is not None:
        inf["coste_usd"] = _c
    # UNA SALIDA VACÍA O ILEGIBLE NO ES «SIN CAMBIOS» (3-oct-2026): el revisor
    # no revisó, y la pantalla no puede decir que no encontró nada.
    _ileg = salida_legible(salida)
    if _ileg:
        inf.update(estado="fallo", error=_ileg, segundos=round(time.perf_counter() - t0, 1))
        if "antecedentes" in inf:
            inf["antecedentes"]["estado"] = "fallo"
        return estudio, antecedentes, inf
    parches, malos = parsear(salida)
    # LOS PARCHES DE CADA SECCIÓN, POR SU CUENTA: un «A3» nunca toca el
    # párrafo 3 del estudio, y uno del estudio nunca toca los antecedentes.
    p_est = [p for p in parches if p.get("seccion", "estudio") != "antecedentes"]
    p_ant = [p for p in parches if p.get("seccion") == "antecedentes"]
    anclas = Anclas(material, fuentes, criterios, plan)
    por_n = {b["n"]: b for b in bl}
    buenos, motivos, vistos = [], [], set()
    max_elim = max(1, int(len([b for b in bl if not b["fijo"]]) * MAX_ELIMINAR))
    n_elim = 0
    for p in p_est:
        b = por_n.get(p["parrafo"])
        if p["parrafo"] in vistos:
            motivo = "segundo parche al mismo párrafo"
        else:
            motivo = guardas(p, b, anclas)
        if not motivo and p["accion"] == "eliminar":
            if n_elim >= max_elim:
                motivo = "demasiadas eliminaciones en una pasada"
            else:
                n_elim += 1
        if not motivo and b is not None and p["accion"] != "eliminar" \
                and " ".join(p["texto"].split()) == " ".join(b["texto"].split()):
            motivo = "sin cambio"
        if motivo:
            motivos.append({"parrafo": p["parrafo"], "tipo": p["tipo"], "motivo": motivo})
            continue
        vistos.add(p["parrafo"])
        buenos.append(p)
    inf["propuestas"] = len(parches) + malos
    inf["descartadas"] = len(motivos) + malos
    inf["motivos_descarte"] = motivos[:30]
    nuevo = aplicar(estudio, buenos, bl) if buenos else estudio
    pal_despues = _palabras(_mc.sin_marcas(nuevo))
    if buenos and pal_despues < pal_antes * PISO_PALABRAS:
        # UN SUPERVISOR NO SE COME MEDIO ESTUDIO: se revierte todo.
        inf["descartadas"] += len(buenos)
        inf["motivos_descarte"].append({"parrafo": 0, "tipo": "extension",
                                        "motivo": f"dejaría el estudio en {pal_despues} de "
                                                  f"{pal_antes} palabras: se revierte todo"})
        buenos, nuevo, pal_despues = [], estudio, pal_antes
    if buenos and desenlace_global(_mc.sin_marcas(nuevo)) != desenlace_global(_mc.sin_marcas(estudio)):
        # LA RED DE SEGURIDAD DEL DESENLACE (3-oct-2026): lo que el resolutivo
        # y la rama leen del estudio entero —sentido en plenitud, violación
        # procesal, sólo los efectos, el último «revoca/confirma»— no puede
        # cambiar por la suma de parches que, uno a uno, pasaron. Se revierte
        # todo, como con el piso de palabras.
        inf["descartadas"] += len(buenos)
        inf["motivos_descarte"].append({"parrafo": 0, "tipo": "incongruencia",
                                        "motivo": "cambiaría el desenlace del estudio: se revierte todo"})
        buenos, nuevo, pal_despues = [], estudio, pal_antes
    correcciones = [{"tipo": p["tipo"], "parrafo": p["parrafo"],
                     "antes": por_n[p["parrafo"]]["texto"][:400],
                     "despues": ("" if p["accion"] == "eliminar" else p["texto"][:400]),
                     "accion": p["accion"], "motivo": p["motivo"]} for p in buenos]
    # ── LOS ANTECEDENTES ──
    ante_nuevos = antecedentes
    if antecedentes is None:
        # Sin antecedentes enviados, un «A3» no tiene a qué aplicarse.
        for p in p_ant:
            inf["descartadas"] += 1
            inf["motivos_descarte"].append({"parrafo": f"A{p['parrafo']}", "tipo": p["tipo"],
                                            "motivo": "antecedentes no enviados al revisor"})
    else:
        anclas_a = AnclasAntecedentes(fuentes_ant)
        por_a = {b["n"]: b for b in ba}
        buenos_a, motivos_a = _revisar_antecedentes(p_ant, ba, anclas_a)
        pal_a_antes = sum(_palabras(b["texto"]) for b in ba)
        if buenos_a:
            ante_nuevos = aplicar_antecedentes(antecedentes, buenos_a, ba)
        pal_a_despues = sum(_palabras(b["texto"]) for b in bloques_antecedentes(ante_nuevos))
        n_rev = 0
        if buenos_a and pal_a_despues < pal_a_antes * PISO_PALABRAS:
            # TAMPOCO SE COME MEDIO RELATO: se revierte todo lo de los antecedentes.
            n_rev = len(buenos_a)
            motivos_a.append({"parrafo": 0, "tipo": "extension",
                              "motivo": f"dejaría los antecedentes en {pal_a_despues} de {pal_a_antes} "
                                        f"palabras: se revierte todo", "accion": "", "texto": ""})
            buenos_a, ante_nuevos, pal_a_despues = [], antecedentes, pal_a_antes
        # LO QUE EL REVISOR VIO MAL EN LO RESUELTO Y LA MÁQUINA NO PUEDE
        # CORREGIR SOLA: el sentido de lo que resolvió cada instancia no lo
        # cambia un parche (guarda dura), pero si el revisor lo acusa como
        # incongruencia con la resolución, el secretario tiene que saberlo.
        por_verificar = [{"bloque": f"A{m['parrafo']}", "antes": por_a[m["parrafo"]]["texto"][:240],
                          "propuesta": str(m.get("texto") or "")[:240]}
                         for m in motivos_a
                         if str(m["motivo"]).startswith("cambia lo resuelto") and m["parrafo"] in por_a
                         and m["tipo"] in _TIPOS_DE_FONDO_ANT][:3]
        desc_a = len([m for m in motivos_a if m["parrafo"] != 0]) + n_rev
        inf["antecedentes"].update(
            estado="aplicado" if buenos_a else "sin_cambios", propuestas=len(p_ant),
            descartadas=desc_a,
            motivos_descarte=[{"parrafo": f"A{m['parrafo']}" if m["parrafo"] else 0, "tipo": m["tipo"],
                               "motivo": m["motivo"]} for m in motivos_a][:30],
            palabras={"antes": pal_a_antes, "despues": pal_a_despues}, hallazgos=len(hall_a))
        if por_verificar:
            inf["antecedentes"]["por_verificar"] = por_verificar
        inf["descartadas"] += desc_a
        # EN EL ORDEN DEL DOCUMENTO: los antecedentes van antes que el estudio.
        # «parrafo» 0 (el contrato D lo reserva al estudio); «bloque» dice cuál.
        correcciones = [{"tipo": p["tipo"], "parrafo": 0, "seccion": "antecedentes",
                         "bloque": f"A{p['parrafo']}",
                         "antes": por_a[p["parrafo"]]["texto"][:400],
                         "despues": ("" if p["accion"] == "eliminar" else p["texto"][:400]),
                         "accion": p["accion"], "motivo": p["motivo"]} for p in buenos_a] + correcciones
    inf["correcciones"] = correcciones
    inf["palabras"] = {"antes": pal_antes, "despues": pal_despues}
    inf["hallazgos"] = len(hall)
    inf["estado"] = "aplicado" if correcciones else "sin_cambios"
    inf["segundos"] = round(time.perf_counter() - t0, 1)
    return nuevo, ante_nuevos, inf


_NOMBRE_TIPO = {"error_juridico": ("error jurídico", "errores jurídicos"),
                "incongruencia": ("incongruencia", "incongruencias"),
                "hecho_no_acreditado": ("hecho no acreditado", "hechos no acreditados"),
                "cita": ("cita", "citas"),
                "repeticion": ("repetición", "repeticiones"),
                "extension": ("párrafo condensado", "párrafos condensados"),
                "redaccion": ("de redacción", "de redacción")}


def _cuenta_por_tipo(cor: list) -> str:
    cuenta = {}
    for c in cor:
        cuenta[c["tipo"]] = cuenta.get(c["tipo"], 0) + 1
    return ", ".join(f"{n} {_NOMBRE_TIPO.get(t, (t, t))[0 if n == 1 else 1]}"
                     for t, n in sorted(cuenta.items(), key=lambda x: -x[1]))


def aviso(inf: dict) -> str:
    """El aviso corto que ve el secretario («» si no hay nada que decir). Sin
    antecedentes revisados, el texto es el de siempre."""
    est = (inf or {}).get("estado")
    con_ant = isinstance((inf or {}).get("antecedentes"), dict)
    if est == "aplicado":
        todas = inf.get("correcciones") or []
        cor = [c for c in todas if c.get("seccion") != "antecedentes"]
        cor_a = [c for c in todas if c.get("seccion") == "antecedentes"]
        pal = inf.get("palabras") or {}
        extra = (f" (de {pal['antes']:,} a {pal['despues']:,} palabras)".replace(",", ".")
                 if pal.get("antes") and pal.get("despues") != pal.get("antes") else "")
        trozos = []
        if cor:
            trozos.append(f"{len(cor)} cosa{'s' if len(cor) != 1 else ''} del estudio: "
                          f"{_cuenta_por_tipo(cor)}{extra}")
        if cor_a:
            trozos.append(f"{len(cor_a)} de los antecedentes: {_cuenta_por_tipo(cor_a)}"
                          if cor else
                          f"{len(cor_a)} cosa{'s' if len(cor_a) != 1 else ''} de los antecedentes: "
                          f"{_cuenta_por_tipo(cor_a)}")
        return f"El supervisor corrigió {'; y '.join(trozos)}. Cada cambio está en la ficha del proyecto."
    if est in ("fallo", "vencido"):
        if est == "vencido":
            por = "se pasó del tiempo"
        elif inf.get("error") in ("sin_respuesta", "ilegible"):
            por = "no devolvió una respuesta legible"
        else:
            por = "falló la llamada"
        if con_ant:
            return (f"El supervisor no pudo revisar el estudio ni los antecedentes ({por}): salen como "
                    f"los escribió el redactor.")
        return f"El supervisor no pudo revisar el estudio ({por}): sale como lo escribió el redactor."
    return ""


def aviso_por_verificar(inf: dict) -> str:
    """Lo que el revisor acusó en lo que resolvió una instancia, en los
    antecedentes, y no se aplicó (la guarda no deja que un parche cambie lo
    resuelto): el secretario lo coteja. «» si no hay."""
    pv = (((inf or {}).get("antecedentes") or {}).get("por_verificar")) or []
    if not pv:
        return ""
    cuales = ", ".join(x["bloque"] for x in pv)
    return (f"ANTECEDENTES POR COTEJAR ({cuales}): el revisor advierte que lo que ahí se dice que "
            f"resolvió una instancia no coincide con la resolución. No se corrigió solo, porque lo "
            f"resuelto no lo cambia la máquina: cotéjelo con la resolución antes de firmar.")


def _como_texto(a) -> str:
    """Los antecedentes como texto. `Fases123.antecedentes` es una CADENA, y
    `"\n".join(cadena)` ponía una letra por renglón: las fechas, números y
    transcripciones que sólo constan en los antecedentes dejaban de anclar
    (revisión adversarial, 3-oct-2026; el mismo fallo que `_terminar` ya
    corrigió en `_fuentes_fe`)."""
    if isinstance(a, (list, tuple)):
        return "\n".join(str(x) for x in a if x)
    return str(a or "")


def _plan_de(r):
    """El plan v4 con que se escribió el estudio, o None. Lo mismo que
    `redactor_adelanto._plan_del_estudio`, sin importar aquel módulo."""
    try:
        _pl = getattr(getattr(r, "encargo", None), "plan", None) or {}
        _pl = _pl.get("plan") if isinstance(_pl, dict) else None
        return _pl if isinstance(_pl, dict) else None
    except Exception:
        return None


async def _antecedentes_del_proyecto(r, material, espera_s: float = None) -> tuple:
    """(antecedentes, síntesis): los párrafos de antecedentes que llevará el
    proyecto, y el dict de la síntesis moderna del que salen (o None).

    LOS MISMOS QUE TOMA `_terminar` (redactor_adelanto): los de la síntesis
    moderna si llegó y los trae, y si no, `fases.parrafos_antecedentes()`. La
    síntesis corre a la par del estudio y casi siempre ya terminó; se la espera
    con `shield` para que el tope no la cancele (`_terminar` la vuelve a
    esperar). Si no llegó a tiempo, (None, None): no se revisan."""
    t = getattr(material, "sintesis", None)
    sint = None
    if t is not None:
        try:
            sint = await asyncio.wait_for(asyncio.shield(t), timeout=ESPERA_SINTESIS_S
                                          if espera_s is None else espera_s)
        except asyncio.TimeoutError:
            return None, None
        except Exception:
            sint = None                       # falló: `_terminar` caerá a las fases
        if isinstance(sint, dict) and sint.get("antecedentes"):
            return list(sint["antecedentes"]), sint
    try:
        a = list(r.fases.parrafos_antecedentes())
    except Exception:
        return None, None
    return (a or None), (sint if isinstance(sint, dict) else None)


def entregar_antecedentes(material, antecedentes: list, sintesis: dict = None) -> None:
    """Deja los antecedentes corregidos donde `_terminar` los recoge:
    `material.sintesis`, como una síntesis ya resuelta que conserva lo demás
    que traía (acto, conceptos). Sin síntesis moderna, sólo trae los
    antecedentes y lo demás cae a las fases, como cuando la síntesis falla.
    Se llama desde código asíncrono (el resolver)."""
    fut = asyncio.get_running_loop().create_future()
    fut.set_result(dict(sintesis or {}, antecedentes=list(antecedentes)))
    material.sintesis = fut


async def en_el_resolver(cliente, r, criterios, material, estudio: str, meta: dict,
                         avisos: list, contexto: str = "") -> str:
    """Lo que llaman los dos gemelos del resolver tras `_congruencia_apertura`.
    Devuelve el estudio (corregido o como estaba); anota el informe en
    `meta["supervisor"]` y el aviso en `avisos`. Nunca lanza.

    CON LA BANDERA `procedencia_por_tipo` (3-oct-2026) revisa también los
    antecedentes en la misma llamada, y los corregidos los deja en
    `material.sintesis` (`entregar_antecedentes`), de donde `_terminar` arma
    `relleno.antecedentes`. La firma no cambia: el llamador no se entera."""
    if not SUPERVISOR_ACTIVO:
        if isinstance(meta, dict):
            meta["supervisor"] = _informe_vacio("apagado")
        return estudio
    try:
        fases = getattr(r, "fases", None)
        _fuentes_fases = list(getattr(fases, "fuentes", None) or [])
        fuentes = _fuentes_fases + [
            _como_texto(getattr(fases, "antecedentes", None)),
            str(getattr(fases, "autos", "") or ""), str(contexto or "")]
        ante, sint = (None, None)
        if rige_antecedentes():
            ante, sint = await _antecedentes_del_proyecto(r, material)
        if ante:
            # LO QUE ANCLA LOS ANTECEDENTES: el acto, el escrito, las
            # constancias y lo que aportó el secretario. Ni los resúmenes del
            # modelo ni los propios antecedentes.
            fuentes_ant = [str(x) for x in _fuentes_fases if x] + [
                str(getattr(fases, "autos", "") or ""), str(contexto or "")]
            nuevo, ante_n, inf = await supervisar_con_antecedentes(
                estudio, ante, criterios, material,
                resumen_acto=str(getattr(fases, "resumen_acto", "") or ""),
                resumen_conceptos=str(getattr(fases, "resumen_conceptos", "") or ""),
                fuentes=fuentes, fuentes_antecedentes=fuentes_ant,
                extracto_acto=extracto_del_acto(str((_fuentes_fases or [""])[0] or ""), ante),
                cliente=cliente, plan=_plan_de(r))
            if ante_n != ante:
                entregar_antecedentes(material, ante_n, sint)
        else:
            nuevo, inf = await supervisar(
                estudio, criterios, material,
                resumen_acto=str(getattr(fases, "resumen_acto", "") or ""),
                resumen_conceptos=str(getattr(fases, "resumen_conceptos", "") or ""),
                fuentes=fuentes, cliente=cliente, plan=_plan_de(r))
        if isinstance(meta, dict):
            meta["supervisor"] = inf
        _av = aviso(inf)
        if _av:
            avisos.append(_av)
        _av_pv = aviso_por_verificar(inf)
        if _av_pv:
            avisos.append(_av_pv)
        # HIGIENE DE REGISTROS: cifras, nunca el texto.
        _u = inf.get("uso") or {}
        _ia = inf.get("antecedentes") if isinstance(inf.get("antecedentes"), dict) else None
        print(f"   🔎 SUPERVISOR: {inf.get('estado')} · {inf.get('modelo')} · "
              f"{len(inf.get('correcciones') or [])} corrección(es) · "
              f"{inf.get('descartadas')} descartada(s) · {inf.get('segundos')} s · "
              f"entrada {_u.get('entrada', '?')} · razonamiento {_u.get('razonamiento', '?')} · "
              f"salida {_u.get('salida', '?')}"
              + (f" · antecedentes {_ia.get('estado')}: "
                 f"{sum(1 for c in inf.get('correcciones') or [] if c.get('seccion') == 'antecedentes')} "
                 f"corregido(s), {_ia.get('descartadas', 0)} descartado(s)" if _ia else ""))
        return nuevo
    except Exception as ex:
        print(f"   ⚠️ SUPERVISOR: {type(ex).__name__}")
        if isinstance(meta, dict):
            meta["supervisor"] = dict(_informe_vacio("fallo"), error=type(ex).__name__)
        return estudio


# ═══ VELOCIDAD: EL BARRIDO ADELANTADO (2-oct-2026) ══════════════════════════
# El barrido de preceptos (6-14 s) corría DESPUÉS de componer el .docx, y la
# recomposición se lleva ~30 s casi todos de la síntesis de portada (AR
# 631/2025: «recomposición 29.8s»). Las dos sólo dependen del estudio ya
# definitivo, así que el barrido se adelanta: con el estudio y los resúmenes
# del relleno, en paralelo con la síntesis y el compositor. Al terminar el
# .docx se barre el documento ENTERO como siempre, pero las respuestas ya
# obtenidas se reutilizan y sólo se pregunta lo que el compositor añadió
# (competencia, procedencia…). El resultado es el mismo barrido —mismos pares,
# misma doble confirmación—; lo que cambia es cuándo se pregunta.
class BarridoMemo:
    """`preguntar` y `confirmar` para `barrido_preceptos.barrer` que recuerdan
    lo ya contestado. Lo que falló no se recuerda: se vuelve a preguntar (en
    `confirmar`, sólo se recuerda la confirmación positiva)."""

    def __init__(self):
        self.respuestas = {}       # (ley, num) → dict de la respuesta (sin «n»)
        self.confirmados = {}      # (ley, num) → True sólo si se confirmó inexistente

    async def preguntar(self, lote: list, segundos: float) -> list:
        import barrido_preceptos as _bp
        faltan = [p for p in lote if tuple(p) not in self.respuestas]
        if faltan:
            res = await _bp._preguntar(faltan, segundos)
            for d in res or []:
                try:
                    k = int(d.get("n")) - 1
                except (TypeError, ValueError):
                    continue
                if 0 <= k < len(faltan):
                    self.respuestas[tuple(faltan[k])] = {x: v for x, v in d.items() if x != "n"}
        return [dict(self.respuestas[tuple(p)], n=i) for i, p in enumerate(lote, 1)
                if tuple(p) in self.respuestas]

    async def confirmar(self, pares: list, segundos: float) -> set:
        """SÓLO SE RECUERDA LO CONFIRMADO (3-oct-2026). `_confirmar` devuelve
        el conjunto de los que una segunda pregunta volvió a negar, y en él no
        se distingue «dijo que existe» de «venció el plazo» o «falló la red»
        (cada `_uno` devuelve None). Guardar False para todos los demás hacía
        que el barrido final no volviera a preguntar una cita inventada cuya
        confirmación falló en el adelantado, y lo que habría sido «la cita
        está inventada» bajaba a «no se pudo comprobar». Lo no confirmado se
        vuelve a preguntar: son pocos (sólo los que la primera pasada negó)."""
        import barrido_preceptos as _bp
        faltan = [tuple(p) for p in pares if not self.confirmados.get(tuple(p))]
        if faltan:
            hechos = await _bp._confirmar(faltan, segundos)
            for p in faltan:
                if p in (hechos or set()):
                    self.confirmados[p] = True
        return {tuple(p) for p in pares if self.confirmados.get(tuple(p))}


def texto_para_barrido(relleno, estructura=None) -> str:
    """Lo que el .docx va a llevar y ya se sabe antes de componerlo: el relleno
    (estudio, resúmenes, oportunidad) y la ESTRUCTURA que el adelanto ya
    escribió (competencia, procedencia, resultandos), que es donde el
    compositor cita la Ley de Amparo y la Ley Orgánica. Sin ella, el barrido
    final tendría que preguntar esas citas igual y el adelanto no ahorraría la
    llamada (medido con un doble del buscador: 3.3 s contra 3.3 s)."""
    trozos = []

    def _mete(v):
        if isinstance(v, (list, tuple)):
            for x in v:
                _mete(x.get("texto") if isinstance(x, dict) else x)
        elif v:
            trozos.append(str(v))
    for campo in ("estudio", "antecedentes", "resumen_acto", "resumen_conceptos", "problemas",
                  "oportunidad"):
        _mete(getattr(relleno, campo, None))
    if estructura is not None:
        # LA EXISTENCIA SÓLO LA LLEVA EL DIRECTO (sexta ronda, 3-oct-2026, C1).
        # David: en la revisión «ya viene en la sentencia recurrida; no es
        # usual ni necesario que lo reproduzcamos». En el AR el campo llega
        # vacío y no aporta nada; su «Procedencia.» (81, fr. I, inciso que
        # toque) va en `procedencia`. El considerando de los asuntos
        # relacionados (64 LFPCA, 88 CFPC / 269 CNPCF) lo pone el compositor
        # al armar el .docx: lo pregunta el barrido final, no éste.
        for campo in ("apertura", "visto", "resultandos", "competencia", "existencia", "procedencia"):
            _mete(getattr(estructura, campo, None))
    return "\n".join(trozos)
