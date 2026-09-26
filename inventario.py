# -*- coding: utf-8 -*-
"""EL INVENTARIO DE ARGUMENTOS — el piso determinista del Paso 2a.

POR QUÉ EXISTE. David aprobó el 26-sep-2026 el plan del estudio de fondo
completo (diag/w2_final.md §3-§6). Ese mismo día el banco midió la v2 del
prompt contra la v1 con un localizador ciego (8 casos × 2 corridas, 675 citas
verificadas): la v2 reduce la Solución a la mitad y repite menos, pero contesta
con razón propia sólo el 73 % de los argumentos autónomos (la v1, el 79 %) y
DUPLICA las omisiones graves (20 frente a 10). Al acortar sin saber qué
argumentos hay, funde los que traen un dato propio en una respuesta global.

El remedio aprobado es que el estudio SEPA qué argumentos tiene que contestar:
una lista con identificador, el argumento tal como lo resumió la fase 2 y una
cita literal del escrito que lo ancla. Con la lista, el estudio marca en qué
párrafo contesta cada uno (`marcas.py`) y el código comprueba la exhaustividad
contra la demanda, no contra la prosa del modelo.

QUÉ HACE, SIN MODELO:
  1. Parte el RESUMEN de conceptos de la fase 2 (un párrafo por argumento,
     `fases123_resumenes.instrucciones_resumen_conceptos`) en segmentos. Un
     párrafo del resumen trae a menudo DOS o TRES argumentos —«Refiere que…»,
     «Asimismo, sostiene que…», «…, y solicita que…»—: medido sobre las
     sesiones reales, el piso por párrafo entero perdía argumentos (ver
     `MEDIDA` abajo), así que se parte donde el resumen abre un argumento
     nuevo con un verbo de alegación.
  2. Asigna a cada segmento su concepto por el ORDINAL que el propio resumen
     escribe («En el tercer concepto…»); el contador de planteamientos sólo se
     usa para acotar dónde buscar en el escrito.
  3. Lo ANCLA en el escrito (`fases.fuentes[1]`) con una cita literal de 10 a
     40 palabras, buscada en el tramo de su concepto por las palabras que
     comparte con el segmento; si no hay pasaje que case, la cita va vacía:
     una cita inventada es peor que ninguna.
  4. Saca sus ANCLAS DURAS —artículo con su ley, registro, expediente, cifra,
     fecha— con el mismo extractor del banco (`metricas_estudio.anclas`), sin
     los datos del propio asunto.

MEDIDA (26-sep-2026; sólo lectura de `taller_sesiones`, incluidas las de
terceros, que David autorizó medir sin generar nada):
  · 64 sesiones con escrito (29 de cuentas que no son administracion@ ni
    soporte@, 22 de secretarios ajenos): 1,049 segmentos, mediana 13 por
    sesión (de 3 a 72), unos 2 por párrafo del resumen; el 97 % lleva cita
    literal y el 100 % de las citas se verifica en el escrito; 0.08 s de
    mediana por sesión, 0.65 s la más larga.
  · Contra el inventario ciego del banco (8 casos, 135 argumentos leídos de la
    demanda por el localizador), adjudicado a mano: el piso partido CONTIENE
    123 (91 %) —los 27 decisivos, 60 de 67 autónomos, 21 de 23 reiteraciones,
    15 de 18 residuales— y da segmento propio a 92 (68 %). El piso por párrafo
    entero da 44 segmentos para esos 135: ni con un argumento por segmento
    pasaría del 33 %, y el tope de 80 palabras le cortaba el resto del
    párrafo. LOS 12 QUE FALTAN NO ESTÁN EN EL RESUMEN de la fase 2 (el cuarto
    alegato del 93/2026, los antecedentes que la demanda del 263/2025 usa como
    argumento, invocaciones genéricas de preceptos…): partir no los recupera;
    recuperarlos es cosa del resumen o de la cobertura contra la demanda (V1b).
  · El ancla cae a menos de 1,500 caracteres de la cita del localizador en 69
    de 85 pares (mediana 367); las más lejanas, vistas una a una, son casi
    todas otro pasaje del mismo argumento (la demanda lo repite).
  · Anomalías: en nueve sesiones el contador y el resumen no casan (seis con
    n=1 y de dos a siete conceptos nombrados; tres con n=2 y 3, 5 o 6); en una
    el resumen se salta el segundo. Manda el ordinal del resumen y se anota.
"""
from __future__ import annotations

import bisect
import math
import re
import unicodedata
from collections import Counter, defaultdict

# ═══════════════════════════════════════════════════════════════════════════
# LÍMITES
# ═══════════════════════════════════════════════════════════════════════════
# El contrato del Paso 2 (diag/contrato_paso2.md): el texto del segmento cabe
# en 80 palabras, recortado en frase; la cita, de 10 a 40 palabras literales.
MAX_PALABRAS_TEXTO = 80
CITA_MIN, CITA_MAX = 10, 40
# Un trozo de menos de esto no es un argumento: es la cola de otro («…, y
# solicita el amparo.»). Se devuelve al anterior.
MIN_PALABRAS_SEGMENTO = 8
# La ventana en que se busca el pasaje del escrito, en palabras con contenido.
VENTANA = 32
# Para aceptar un pasaje como ancla: cuánto del peso del segmento aparece en
# él y cuántas palabras distintas comparte. Calibrado con las citas del
# localizador ciego (ver `MEDIDA`): por debajo, el pasaje casaba por palabras
# de oficio («responsable», «sentencia», «prueba»), no por el argumento.
UMBRAL_CITA = 0.22
MIN_COMUNES = 3


# ═══════════════════════════════════════════════════════════════════════════
# TEXTO PLANO QUE CONSERVA LAS POSICIONES
# ═══════════════════════════════════════════════════════════════════════════
def _plano(t: str) -> str:
    """Minúsculas sin acento, CARÁCTER POR CARÁCTER (misma longitud)."""
    fuera = []
    for c in (t or ""):
        d = unicodedata.normalize("NFKD", c)
        base = [x for x in d if not unicodedata.combining(x)]
        fuera.append(base[0].lower() if len(base) == 1 else c.lower())
    return "".join(fuera)


# ═══════════════════════════════════════════════════════════════════════════
# EL ESCRITO, SIN LA MAQUETA DEL PDF
# ═══════════════════════════════════════════════════════════════════════════
# El escrito llega del PDF a renglón fijo: 80 columnas rellenas de espacios y
# LA PALABRA PARTIDA SIN GUION donde acaba el renglón («posesi⏎ones»). Medido
# en las sesiones del banco (642/2024, 103/2025…): el texto guardado en
# `fuentes[1]` trae los renglones cortados así, y el mismo escrito del corpus
# Kingston no. Una cita «literal» sacada del texto crudo llevaría la palabra
# partida; se saca del escrito con los renglones repuestos y los blancos
# colapsados, y así se verifica también.
def _ancho_fijo(lineas: list) -> int:
    """El ancho del renglón si el texto viene a columna fija; 0 si no."""
    largas = [len(x) for x in lineas if len(x) >= 60]
    if len(largas) < 20:
        return 0
    w, n = Counter(largas).most_common(1)[0]
    return w if n >= 0.3 * len(largas) else 0


def normalizar_escrito(texto: str) -> tuple:
    """(texto normalizado, posiciones): cada carácter del normalizado dice de
    qué posición del original viene. Renglones partidos a columna fija se
    unen sin espacio; un guion de fin de renglón entre minúsculas se quita; el
    resto de los blancos se colapsa a uno."""
    texto = texto or ""
    lineas = texto.split("\n")
    w = _ancho_fijo(lineas)
    fuera, pos = [], []
    i = 0
    blanco_pendiente = False
    for k, ln in enumerate(lineas):
        sig = lineas[k + 1] if k + 1 < len(lineas) else ""
        for j, c in enumerate(ln):
            if c.isspace():
                blanco_pendiente = bool(fuera)
                continue
            if blanco_pendiente:
                fuera.append(" ")
                pos.append(i + j)
                blanco_pendiente = False
            fuera.append(c)
            pos.append(i + j)
        # ¿El salto de renglón parte una palabra? A columna fija, un renglón
        # LLENO —justo del ancho y sin blanco al final— seguido de otro que no
        # empieza en blanco es un corte a media palabra, sea cual sea la letra
        # («posesi⏎ones», «CONCEP⏎TO», «4-80-54.⏎113»): si el corte hubiera
        # caído en un espacio, el blanco estaría a un lado o al otro.
        cola = ln.rstrip()
        unir = False
        if cola and sig and not sig[:1].isspace():
            if w and len(ln) == w and cola == ln:
                unir = True
            elif (sig[:1].isalpha() and sig[:1].islower() and cola.endswith("-")
                  and len(cola) >= 2 and cola[-2].isalpha() and cola[-2].islower()):
                if fuera and fuera[-1] == "-":
                    fuera.pop()
                    pos.pop()
                unir = True
        if not unir and k + 1 < len(lineas):
            blanco_pendiente = bool(fuera)
        elif unir:
            blanco_pendiente = False
        i += len(ln) + 1
    return "".join(fuera), pos


def _colapsar(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()


def verificar_cita(cita: str, escrito: str, normalizado: str = None) -> bool:
    """¿La cita está palabra por palabra en el escrito?

    Se compara contra el escrito con los renglones repuestos y los blancos
    colapsados (ver `normalizar_escrito`), y sin distinguir mayúsculas ni
    acentos: el OCR los pierde a su aire y eso no hace menos literal la cita.
    La usa también la pieza del plan (V0 g) para no verificar de dos modos."""
    c = _plano(_colapsar(cita))
    if len(c.split()) < 3:
        return False
    n = normalizado if normalizado is not None else normalizar_escrito(escrito)[0]
    return c in _plano(n)


# ═══════════════════════════════════════════════════════════════════════════
# PALABRAS CON CONTENIDO
# ═══════════════════════════════════════════════════════════════════════════
# Las palabras de oficio no identifican un argumento: todos los conceptos
# hablan de la «responsable», la «sentencia» y el «artículo». Se quitan antes
# de comparar, y lo que queda se pesa por lo raro que es en el escrito.
_VACIAS = set("""
a al algo ante antes aquel aquella aquellos asi aun aunque bajo cada como con
contra cual cuales cuando de del desde donde dos durante el ella ellas ello
ellos en entre era eran es esa esas ese eso esos esta estan estas este esto
estos fue fueron ha han hasta hay la las le les lo los mas me mediante mi mis
misma mismo mismos muy ni no nos o otra otras otro otros para pero por porque
pues que quien se segun ser si sin sino sobre son su sus tal tambien tanto te
toda todas todo todos tras un una uno unos y ya
parte quejosa quejoso recurrente responsable autoridad sala sentencia
concepto conceptos violacion agravio agravios aduce alega sostiene señala
senala refiere manifiesta afirma argumenta considera estima indica expone
precisa agrega anade asimismo ademas finalmente tambien juicio resolucion
reclamada recurrida articulo articulos derecho derechos constitucional
constitucion politica estados unidos mexicanos ley codigo dicha dicho dichos
dichas cual debe debio deben puede pueden hace hacer tiene tienen haber sido
esta estar seria sera caso forma manera virtud efecto efectos respecto
relacion lugar mismo primer primero segundo tercer tercero cuarto quinto
sexto septimo octavo noveno decimo unico formulado planteamiento
""".split())


def _stem(p: str) -> str:
    return p if p.isdigit() else p[:6]


_RX_PALABRA = re.compile(r"[a-z0-9ñ]+")


def _fichas(texto_plano: str) -> list:
    """[(inicio, fin, raíz)] de las palabras con contenido de un texto plano."""
    fuera = []
    for m in _RX_PALABRA.finditer(texto_plano):
        w = m.group(0)
        if w in _VACIAS:
            continue
        if not w.isdigit() and len(w) < 4:
            continue
        if w.isdigit() and len(w) < 2:
            continue
        fuera.append((m.start(), m.end(), _stem(w)))
    return fuera


def _raices(texto: str) -> set:
    return {s for _, _, s in _fichas(_plano(texto))}


# ═══════════════════════════════════════════════════════════════════════════
# LAS ANCLAS DURAS
# ═══════════════════════════════════════════════════════════════════════════
# Lo que un argumento trae de propio y un resumen no puede inventar: el
# artículo con su ley, el registro de una tesis, un expediente, una cifra, una
# fecha (w2_final §3.4-5). Se usa el MISMO extractor que el banco
# (`metricas_estudio.anclas`), para que la cobertura que mide el banco y la
# que comprueba el taller hablen de las mismas anclas.
ENCABEZADO_ESCRITO = 1500


def _extractor():
    import metricas_estudio as _me
    return _me


def anclas_duras(texto: str, excluir=None) -> list:
    """Las anclas duras del texto, normalizadas («art 1635 ccf», «reg
    2009789», «exp 1114/2017», «cifra 85842», «fecha 2025-09-18»), sin
    repetir y en orden de aparición. `excluir`: las del propio asunto."""
    try:
        todas = _extractor().anclas(texto or "")
    except Exception:
        return []
    fuera, vistos = [], set(excluir or ())
    for _, _, k in sorted(todas):
        if k in vistos:
            continue
        vistos.add(k)
        fuera.append(k)
    return fuera


def anclas_del_asunto(escrito: str) -> set:
    """LOS DATOS DEL PROPIO ASUNTO NO SON ANCLAS: el expediente, el toca y la
    fecha de la sentencia reclamada salen en el encabezado del escrito y en
    cualquier respuesta, conteste o no. Es la regla de `metricas_estudio.
    anclas_del_escrito`: lo que aparece en los primeros 1,500 caracteres."""
    try:
        # Sobre el escrito SIN la maqueta del PDF: 1,500 caracteres rellenos a
        # ochenta columnas son la mitad de encabezado que en el texto limpio
        # del corpus, con el que se calibró la regla.
        norm = normalizar_escrito(escrito or "")[0]
        return {k for a, _, k in _extractor().anclas(norm) if a < ENCABEZADO_ESCRITO}
    except Exception:
        return set()


# ═══════════════════════════════════════════════════════════════════════════
# EL RESUMEN, PARTIDO EN ARGUMENTOS
# ═══════════════════════════════════════════════════════════════════════════
_RX_BISAGRA = re.compile(
    r"^\s*(?:en\s+contra\s+de\s+(?:esas|estas|las\s+anteriores|dichas|tales|"
    r"las|aquellas)\s+consideraciones|la\s+parte\s+\w+\s+(?:plantea|formula|"
    r"hace\s+valer|expresa)\s+(?:los\s+siguientes|como)\b)", re.I)
_RX_ANCLA_PAGINA = re.compile(r"\[\[\s*([^\]]{1,40}?)\s*\]\]")

# EL ORDINAL QUE ABRE UN CONCEPTO, tal como lo escribe la fase 2 (medido sobre
# las 64 sesiones con escrito: «En el primer concepto de violación…», «En el
# único agravio formulado…», «Finalmente, en el cuarto concepto…»). Se busca
# sólo al principio del párrafo: «reitera lo dicho en el primer concepto», a
# media frase, no abre nada.
_ORD = {
    "primer": 1, "primero": 1, "primera": 1, "segundo": 2, "segunda": 2,
    "tercer": 3, "tercero": 3, "tercera": 3, "cuarto": 4, "cuarta": 4,
    "quinto": 5, "quinta": 5, "sexto": 6, "sexta": 6, "septimo": 7,
    "septima": 7, "setimo": 7, "octavo": 8, "octava": 8, "noveno": 9,
    "novena": 9, "decimo": 10, "decima": 10, "undecimo": 11, "undecima": 11,
    "duodecimo": 12, "duodecima": 12, "unico": 1, "unica": 1,
    "decimo primero": 11, "decimo segundo": 12, "decimo tercero": 13,
    "decimo cuarto": 14, "decimo quinto": 15, "decimoprimero": 11,
    "decimosegundo": 12, "decimotercero": 13, "decimocuarto": 14,
    "decimoquinto": 15, "ultimo": -1, "ultima": -1,
}
_ALT_ORD = "|".join(sorted((re.escape(k).replace(r"\ ", r"\s+") for k in _ORD),
                           key=len, reverse=True))
_SUST = r"(?:concepto|agravio|motivo)s?"
_RX_ORD_1 = re.compile(r"\b(" + _ALT_ORD + r")\s+" + _SUST + r"\b")
_RX_ORD_2 = re.compile(
    r"\b" + _SUST + r"(?:\s+de\s+(?:violacion|anulacion|impugnacion|"
    r"inconformidad|disenso))?\s+(" + _ALT_ORD + r")\b")
ORDINAL_AL_PRINCIPIO = 160

# LOS VERBOS CON QUE LA FASE 2 ABRE UN ARGUMENTO. El prompt de la fase 2 fija
# el presente de alegación (`VERBOS_PARTE`); medidos en los resúmenes reales
# salen además «afirma», «agrega», «añade», «precisa», «expone», «insiste»…
_VERBOS = (
    r"aduce|alega|argumenta|arguye|sostiene|senala|refiere|manifiesta|afirma|"
    r"asevera|agrega|anade|expone|precisa|insiste|reitera|estima|considera|"
    r"plantea|cuestiona|controvierte|reclama|denuncia|advierte|destaca|"
    r"invoca|solicita|pide|expresa|menciona|indica|explica|acusa|atribuye|"
    r"combate|impugna|objeta|niega|recalca|enfatiza|subraya|puntualiza|"
    r"concluye|propone|critica|reprocha|sustenta|apunta|observa|aclara|"
    r"asegura|distingue|hace\s+valer|se\s+duele|se\s+queja|se\s+inconforma|"
    r"estima|narra|relata|cita|anota|resalta|repara|insta|argumentan")
_SUJETO = (r"(?:(?:la|el)\s+(?:parte\s+)?(?:quejos[oa]|recurrente|inconforme|"
           r"promovente|peticionari[oa]|tercer[oa]\s+interesad[oa]|"
           r"autoridad\s+recurrente|adherente|demandad[oa]|actor[a]?)|"
           r"(?:la|el)\s+recurrente|(?:esta|dicha)\s+parte)")
_CONECTOR = (
    r"asimismo|ademas|tambien|igualmente|de\s+igual\s+(?:forma|manera|modo)|"
    r"por\s+otra\s+parte|por\s+otro\s+lado|en\s+otro\s+aspecto|en\s+diverso\s+"
    r"aspecto|aunado\s+a\s+(?:lo\s+anterior|ello|esto)|adicionalmente|a\s+su\s+"
    r"vez|finalmente|por\s+ultimo|en\s+ese\s+(?:sentido|orden\s+de\s+ideas)|"
    r"en\s+esa\s+linea|por\s+ello|por\s+lo\s+anterior|por\s+tanto|por\s+lo\s+"
    r"tanto|en\s+consecuencia|de\s+ahi\s+que|con\s+base\s+en\s+ello|en\s+tal\s+"
    r"virtud|en\s+el\s+mismo\s+sentido|en\s+adicion|por\s+su\s+parte|al\s+"
    r"respecto|luego|incluso|mas\s+aun|maxime|de\s+esta\s+(?:forma|manera)|"
    r"en\s+relacion\s+con\s+(?:ello|lo\s+anterior|esto)|sobre\s+el\s+particular|"
    r"en\s+suma|en\s+esa\s+misma\s+linea|a\s+mayor\s+abundamiento|"
    r"(?:en\s+cuanto\s+a|respecto\s+(?:de|a)|en\s+relacion\s+con|sobre|por\s+lo\s+"
    r"que\s+hace\s+a|tocante\s+a|acerca\s+de)\s+[^,;.]{2,140}?")
# Abre argumento: [conector(es),] [sujeto] [también] VERBO
_RX_ABRE = re.compile(
    r"^\s*(?:(?:" + _CONECTOR + r")\s*,?\s*){0,2}(?:" + _SUJETO + r"\s*,?\s*)?"
    r"(?:(?:tambien|ademas|igualmente|asimismo|finalmente|incluso)\s*,?\s*)?"
    r"(?:" + _VERBOS + r")\b")
# DENTRO de una frase: «…; además, afirma que…», «…, y solicita que…»,
# «…, además de señalar que…». Sólo con verbo de alegación detrás: una coma y
# una «y» sueltas separan cosas del mismo argumento.
_RX_CORTE_DENTRO = re.compile(
    r"(?:;\s*(?=(?:(?:" + _CONECTOR + r")\s*,?\s*)?(?:" + _SUJETO + r"\s*,?\s*)?"
    r"(?:tambien\s+|ademas\s+|igualmente\s+)?(?:" + _VERBOS + r")\s+(?:que|"
    r"la|el|los|las|de|en|se)\b)"
    r"|,\s+y\s+(?:ademas\s+|tambien\s+)?(?=(?:solicita|pide|sostiene|afirma|"
    r"alega|aduce|senala|refiere|argumenta|reclama|insiste|agrega|anade)\s+que\b)"
    # «…, además de señalar que…», «…, además de examinar que el crédito…»
    r"|,\s+(?=ademas\s+de\s+[a-z]+(?:ar|er|ir)\b)"
    # «…, por lo que solicita que se analicen con perspectiva de género…»: la
    # petición con contenido propio es un argumento (174/2026, C1.f del
    # inventario ciego).
    r"|,\s+por\s+lo\s+que\s+(?=(?:solicita|pide|reclama)\s+que\b)"
    # «omite analizar la temporalidad…, así como pronunciarse sobre el tercer
    # inmueble»: dos omisiones, dos argumentos (174/2026, C1.c y C1.g). Sólo con
    # infinitivo detrás; «así como las declaraciones» es la misma omisión.
    r"|,?\s+(?=asi\s+como\s+(?:a\s+)?[a-z]+(?:ar|er|ir)(?:se|lo|la|los|las|le|les)?\s)"
    r")")
# «Refiere que A y que B»: dos afirmaciones en paralelo. El trozo nuevo se
# escribe con el verbo de la frase («Refiere que B») para que se lea solo.
_RX_Y_QUE = re.compile(r",?\s+y\s+que\s+")
# LO QUE APOYA AL ARGUMENTO ANTERIOR NO ES UNO NUEVO. «Para apoyar esa
# postura, invoca la jurisprudencia…» y «Cita en su apoyo la tesis…» (medido
# en 23/2026, 26/2025, 250/2026) son el sostén del argumento de antes: van con
# él. Separarlos daría un «argumento» que sólo contiene un registro.
_RX_APOYO = re.compile(
    r"^\s*(?:(?:para|a\s+fin\s+de|con\s+el\s+fin\s+de)\s+(?:apoyar|sustentar|"
    r"fundar|robustecer|reforzar|sostener|respaldar|justificar|demostrar|"
    r"acreditar)\b[^,;.]{0,160},?\s*(?:invoca|cita|transcribe|refiere|menciona)|"
    r"(?:invoca|cita|transcribe)\s+(?:en\s+su\s+apoyo|como\s+apoyo|al\s+respecto|"
    r"para\s+(?:ello|tal\s+efecto))|en\s+apoyo\s+de\s+(?:ello|lo\s+anterior|su\s+"
    r"(?:postura|argumento|planteamiento)),?\s*(?:invoca|cita)|"
    r"(?:al\s+efecto|para\s+ello|en\s+ese\s+sentido),\s*(?:invoca|cita|transcribe)\s+"
    r"(?:la|las|el|los)\s+(?:jurisprudencia|tesis|criterio))")

# Abreviaturas con punto que no cierran frase. Las claves de tesis —«1a./J.»,
# «2a. CCXLI/2014», «P./J.»— y las iniciales sueltas —«J. Guadalupe»— tampoco.
_RX_ABREV = re.compile(
    r"\b(?:arts?|fracc?|frs?|n[úu]ms?|no|lic|ing|dra?|mtr[oa]|sra?|srta|mag|arq|"
    r"profr?a?|gral|cd|edo|col|mpio|av|inc|p[áa]gs?|p[áa]rrs?|reg|exp|vol|cfr|"
    r"op|cit|ob|ed|qro|mex|jal|gto|etc|aprox|sig|sigs|pp|p|ss|s\.a|c\.v|"
    r"\d+a|[A-ZÁÉÍÓÚÑ])\.", re.I)
_RX_FIN_FRASE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿«“\"(])")


def _frases(p: str) -> list:
    """Frases de un párrafo del resumen, sin partir en las abreviaturas.

    LA LETRA SUELTA CON PUNTO es inicial («J. Guadalupe») o final de frase
    («…del edificio D. Sostiene que…», 722/2025): se protege salvo que lo que
    sigue abra un argumento. Medido: sin esta salvedad, el segundo argumento
    del primer concepto del 722/2025 se quedaba pegado al primero."""
    q = _RX_ABREV.sub(lambda m: m.group(0).replace(".", "\u2059"), p)
    crudas = []
    for f in _RX_FIN_FRASE.split(q):
        f = f.replace("\u2059", ".").strip()
        if f:
            crudas.append(f)
    fuera = []
    for f in crudas:
        # Una letra suelta protegida que en realidad cerraba la frase.
        partes = re.split(r"(?<=\b[A-ZÁÉÍÓÚÑ]\.)\s+(?=[A-ZÁÉÍÓÚÑ])", f)
        acum = partes[0]
        for x in partes[1:]:
            if _abre_argumento(x):
                fuera.append(acum)
                acum = x
            else:
                acum = acum + " " + x
        fuera.append(acum)
    return fuera


def _abre_argumento(frase: str) -> bool:
    return bool(_RX_ABRE.match(_plano(frase)))


def _ordinal(parrafo: str):
    """El ordinal con que el párrafo abre un concepto, o None."""
    cab = _plano(parrafo[:ORDINAL_AL_PRINCIPIO])
    mejor = None
    for rx in (_RX_ORD_1, _RX_ORD_2):
        for m in rx.finditer(cab):
            if mejor is None or m.start() < mejor[0]:
                clave = re.sub(r"\s+", " ", m.group(1))
                mejor = (m.start(), _ORD.get(clave))
    return mejor[1] if mejor else None


def _partir_dentro(frase: str) -> list:
    """Parte una frase en los puntos donde el resumen abre otro argumento
    dentro de ella (ver `_RX_CORTE_DENTRO` y `_RX_Y_QUE`). El signo que separa
    —«;», «, y»— se queda fuera de los dos trozos; el conector, con el nuevo."""
    pl = _plano(frase)
    cortes = [(m.start(), m.end(), "") for m in _RX_CORTE_DENTRO.finditer(pl)]
    # «…que A y que B» sólo si la frase abre con un verbo de alegación seguido
    # de «que»: entonces las dos son lo que la parte sostiene.
    mv = re.match(r"^\s*(?:(?:" + _CONECTOR + r")\s*,?\s*){0,2}(?:" + _SUJETO +
                  r"\s*,?\s*)?(?:(?:tambien|ademas|igualmente)\s*,?\s*)?((?:" +
                  _VERBOS + r"))\s+que\b", pl)
    if mv:
        verbo = frase[mv.start(1):mv.end(1)]
        for m in _RX_Y_QUE.finditer(pl, mv.end()):
            cortes.append((m.start(), m.end(), verbo.lower() + " que "))
    if not cortes:
        return [frase]
    cortes.sort()
    # UN CORTE QUE DEJA UN TROZO DE MENOS DE `MIN_PALABRAS_SEGMENTO` NO SE
    # HACE: ese trozo es la cola de la misma afirmación («…, y que no sabe leer
    # ni escribir»), y devolverlo después al anterior pegaba dos frases sin su
    # signo (medido en 642/2024).
    # Se decide de izquierda a derecha: lo de la izquierda desde el último
    # corte aceptado, y lo de la derecha hasta el FINAL de la frase (un corte
    # posterior puede no hacerse, y medir contra él descartaba cortes buenos).
    def _pal(a, b):
        return len(frase[a:b].split())
    buenos, ini = [], 0
    for a, b, p in cortes:
        if a < ini:
            continue
        if _pal(ini, a) < MIN_PALABRAS_SEGMENTO or _pal(b, len(frase)) < MIN_PALABRAS_SEGMENTO:
            continue
        buenos.append((a, b, p))
        ini = b
    if not buenos:
        return [frase]
    trozos, ini, pref = [], 0, ""
    for a, b, p in buenos:
        trozos.append(pref + frase[ini:a])
        ini, pref = b, p
    trozos.append(pref + frase[ini:])
    fuera = []
    for t in trozos:
        t = t.strip(" ;,")
        if not t:
            continue
        # El trozo de dentro empieza en minúscula: se sube la primera letra
        # para que se lea como lo que es, un argumento.
        fuera.append(t[:1].upper() + t[1:] if fuera else t)
    return fuera


def _recortar(texto: str, tope: int = MAX_PALABRAS_TEXTO) -> str:
    """≤ `tope` palabras, cortando en frase; si una sola frase se pasa, en la
    última coma o punto y coma antes del tope, con «…»."""
    ps = texto.split()
    if len(ps) <= tope:
        return texto.strip()
    acum = []
    for f in _frases(texto):
        if len(" ".join(acum + [f]).split()) > tope:
            break
        acum.append(f)
    if acum:
        return " ".join(acum).strip()
    corto = " ".join(ps[:tope])
    k = max(corto.rfind(";"), corto.rfind(","))
    if k > len(corto) // 2:
        corto = corto[:k]
    return corto.rstrip(" ,;") + "…"


def _limpiar_marcas_pagina(texto: str) -> tuple:
    """(texto sin [[p.X §Y]], lista de las marcas de página que traía)."""
    pags = [m.group(1).strip() for m in _RX_ANCLA_PAGINA.finditer(texto or "")]
    t = _RX_ANCLA_PAGINA.sub(" ", texto or "")
    return re.sub(r"[ \t]{2,}", " ", t).strip(), pags


def _es_bisagra(p: str) -> bool:
    pl = _plano(p)
    if _RX_BISAGRA.match(pl) and len(p.split()) <= 40:
        return True
    return p.rstrip().endswith(":") and len(p.split()) <= 25


def _es_rotulo(p: str) -> bool:
    """Un renglón que es rótulo y no argumento («RESUMEN DE LOS AGRAVIOS»)."""
    t = p.strip()
    if len(t.split()) > 12:
        return False
    letras = [c for c in t if c.isalpha()]
    return bool(letras) and sum(c.isupper() for c in letras) / len(letras) > 0.8


def _get(obj, k, defecto=None):
    if isinstance(obj, dict):
        return obj.get(k, defecto)
    return getattr(obj, k, defecto)


def _parrafos_resumen(fases) -> list:
    rc = str(_get(fases, "resumen_conceptos", "") or "")
    return [p.strip() for p in rc.split("\n") if p.strip()]


def _letra(i: int) -> str:
    """a…z, y después aa, ab… (un concepto de sesenta páginas trae más de
    veintiséis argumentos: medido en 489/2026, 32 párrafos de resumen)."""
    abc = "abcdefghijklmnopqrstuvwxyz"
    return abc[i] if i < 26 else abc[i // 26 - 1] + abc[i % 26]


def piezas_del_resumen(fases, partir: bool = True) -> tuple:
    """([{concepto, parrafo, texto, paginas}], anomalías) — el resumen en
    argumentos, SIN anclar todavía. `partir=False` da el piso por párrafo
    entero (lo que el contrato describía al principio), para medir."""
    ps = _parrafos_resumen(fases)
    conteo = _get(fases, "conteo", None) or {}
    n = int(conteo.get("n") or 0) if str(conteo.get("estado")) == "contado" else 0
    anomalias = []
    # ── el concepto de cada párrafo ────────────────────────────────────────
    cuerpo = [(i, p) for i, p in enumerate(ps)
              if not (_es_rotulo(p) or (i <= 1 and _es_bisagra(p)))]
    ords = {i: _ordinal(p) for i, p in cuerpo}
    hay_ordinal = any(v is not None for v in ords.values())
    asignado = {}
    if hay_ordinal:
        actual = None
        for i, _ in cuerpo:
            o = ords[i]
            if o == -1:                                  # «el último»
                o = n if n else ((actual or 0) + 1)
            if o is not None:
                actual = o
            elif actual is None:
                # Argumentos antes del primer ordinal: van al primero. Pasa en
                # los resúmenes que abren con una frase general del escrito.
                actual = 1
                anomalias.append("argumento antes del primer ordinal")
            asignado[i] = actual
        vistos = sorted(set(asignado.values()))
        # MANDA EL ORDINAL DEL RESUMEN, y se anota cuando no casa con el
        # contador. Medido en las 64 sesiones con escrito: en seis el
        # contador dio n=1 y el resumen nombra de dos a siete conceptos
        # (308/2026, 526/2024, 529/2024…), y en tres dio 2 donde el resumen
        # nombra tres, cinco y seis (263/2025 entre ellos, que el inventario
        # ciego también parte en seis). El resumen es lo que el estudio lee.
        if n >= 1 and max(vistos) > n:
            anomalias.append(f"el resumen nombra el {max(vistos)}.º y el contador contó {n}")
        huecos = [k for k in range(1, max(vistos)) if k not in vistos]
        if huecos:
            anomalias.append("el resumen no nombra el " +
                             ", ".join(f"{k}.º" for k in huecos[:5]))
    elif n == 1:
        for i, _ in cuerpo:
            asignado[i] = 1
    elif n >= 2 and len(cuerpo) == n:
        for k, (i, _) in enumerate(cuerpo, 1):
            asignado[i] = k
    else:
        # SIN ORDINALES Y SIN UNA CUENTA QUE CASE: el orden de párrafos
        # (contrato del Paso 2). Se anota: el número del concepto puede no
        # ser el del escrito.
        for k, (i, _) in enumerate(cuerpo, 1):
            asignado[i] = k if n != 1 else 1
        if cuerpo:
            anomalias.append("resumen sin ordinales: concepto = orden de párrafos")
    if not cuerpo:
        anomalias.append("resumen sin párrafos de argumento")
    # ── los argumentos dentro de cada párrafo ──────────────────────────────
    piezas = []
    for i, p in cuerpo:
        limpio, pags = _limpiar_marcas_pagina(p)
        if not limpio:
            continue
        conc = asignado[i]
        if not partir:
            if piezas and piezas[-1]["concepto"] == conc and _RX_APOYO.match(_plano(limpio)):
                piezas[-1]["texto"] = (piezas[-1]["texto"] + " " + limpio).strip()
            else:
                piezas.append({"concepto": conc, "parrafo": i, "texto": limpio,
                               "paginas": pags})
            continue
        grupos = []
        for f in _frases(limpio):
            trozos = _partir_dentro(f)
            for k, t in enumerate(trozos):
                nuevo = (k > 0) or not grupos or _abre_argumento(t)
                if grupos and _RX_APOYO.match(_plano(t)):
                    nuevo = False
                if nuevo:
                    grupos.append([t])
                else:
                    grupos[-1].append(t)
        # Los trozos cortos vuelven al anterior: son colas, no argumentos.
        juntos = []
        for g in grupos:
            txt = " ".join(g).strip()
            if juntos and len(txt.split()) < MIN_PALABRAS_SEGMENTO:
                juntos[-1] = (juntos[-1] + " " + txt).strip()
            else:
                juntos.append(txt)
        if len(juntos) >= 2 and len(juntos[0].split()) < MIN_PALABRAS_SEGMENTO:
            juntos[1] = juntos[0] + " " + juntos[1]
            juntos = juntos[1:]
        # UN PÁRRAFO QUE SÓLO APOYA AL ANTERIOR va con él. Medido: 11 de los
        # párrafos de las 64 sesiones con escrito abren «Para apoyar esa
        # postura, invoca la jurisprudencia…» o «En apoyo de lo anterior
        # invoca…» (26/2025, 23/2026, 250/2026…); son el sostén del argumento
        # de antes, y el inventario ciego los cuenta dentro de él.
        if (juntos and piezas and piezas[-1]["concepto"] == conc
                and _RX_APOYO.match(_plano(juntos[0]))):
            piezas[-1]["texto"] = (piezas[-1]["texto"] + " " + juntos[0]).strip()
            juntos = juntos[1:]
        for k, txt in enumerate(juntos):
            piezas.append({"concepto": conc, "parrafo": i, "texto": txt,
                           # La marca de página la lleva el primer argumento
                           # del párrafo, que es donde la fase 2 la pone.
                           "paginas": pags if k == 0 else []})
    return piezas, anomalias


# ═══════════════════════════════════════════════════════════════════════════
# EL ANCLA EN EL ESCRITO
# ═══════════════════════════════════════════════════════════════════════════
class _Escrito:
    """El escrito normalizado, con sus palabras de contenido y su peso."""

    def __init__(self, escrito: str):
        self.crudo = escrito or ""
        self.norm, self.pos = normalizar_escrito(self.crudo)
        self.pl = _plano(self.norm)
        self.fichas = _fichas(self.pl)          # [(ini, fin, raíz)] en norm
        self.inicios = [f[0] for f in self.fichas]
        # EL PESO DE CADA RAÍZ: lo rara que es en el escrito, por bloques de
        # sesenta palabras. «Pericial» en un escrito que la nombra en tres
        # sitios pesa; «prueba», que sale en todos, casi nada.
        bloques = max(1, len(self.fichas) // 60 + 1)
        df = defaultdict(set)
        for k, (_, _, s) in enumerate(self.fichas):
            df[s].add(k // 60)
        self.idf = {s: math.log((bloques + 1) / (len(b) + 0.5)) for s, b in df.items()}
        self.idf_max = math.log(bloques + 2)
        # Dónde sale cada raíz, para no recorrer el escrito entero por cada
        # argumento (un escrito de 600,000 caracteres son ~90,000 palabras).
        self.indice = defaultdict(list)
        for k, (_, _, s) in enumerate(self.fichas):
            self.indice[s].append(k)
        # Las palabras del normalizado (por espacios), para cortar la cita en
        # palabras enteras.
        _ps = [(m.start(), m.end()) for m in re.finditer(r"\S+", self.norm)]
        self.ini_pal = [a for a, _ in _ps]
        self.fin_pal = [b for _, b in _ps]

    def a_norm(self, pos_crudo: int) -> int:
        """La posición en el normalizado que corresponde a una del crudo."""
        return bisect.bisect_left(self.pos, pos_crudo)

    def peso(self, s: str) -> float:
        return self.idf.get(s, self.idf_max)


def _tramo_de(conteo: dict, concepto: int, esc: _Escrito) -> tuple:
    """(ini, fin) en el normalizado donde buscar el argumento de ese concepto:
    su tramo del contador si lo hay, con un margen; si no, el escrito entero."""
    if str((conteo or {}).get("estado")) == "contado":
        tramos = conteo.get("tramos") or []
        valores = conteo.get("valores") or list(range(1, len(tramos) + 1))
        for v, tr in zip(valores, tramos):
            try:
                if int(v) == int(concepto) and len(tr) == 2:
                    a, b = esc.a_norm(int(tr[0])), esc.a_norm(int(tr[1]))
                    if b > a:
                        return max(0, a - 400), min(len(esc.norm), b + 400)
            except (TypeError, ValueError):
                continue
    return 0, len(esc.norm)


def anclar(texto: str, esc: _Escrito, tramo: tuple = None) -> tuple:
    """(cita literal, posición en el normalizado, parecido) del pasaje del
    escrito que mejor casa con el texto; («», -1, parecido) si ninguno pasa
    el umbral."""
    buscadas = _raices(texto)
    if not buscadas or not esc.fichas:
        return "", -1, 0.0
    total = sum(esc.peso(s) for s in buscadas)
    ini, fin = tramo or (0, len(esc.norm))
    k0 = bisect.bisect_left(esc.inicios, ini)
    k1 = bisect.bisect_left(esc.inicios, fin)
    hits = []
    for s in buscadas:
        ks = esc.indice.get(s) or []
        a_, b_ = bisect.bisect_left(ks, k0), bisect.bisect_left(ks, k1)
        hits.extend((k, s) for k in ks[a_:b_])
    hits.sort()
    if not hits:
        return "", -1, 0.0
    mejor = (0.0, 0, None)
    cuenta = Counter()
    peso_vent = 0.0
    j = 0
    # Dos punteros sobre los aciertos: la ventana va de hits[i] a VENTANA
    # palabras de contenido más allá, y pesa cada raíz una vez.
    for i in range(len(hits)):
        while j < len(hits) and hits[j][0] < hits[i][0] + VENTANA:
            s = hits[j][1]
            if cuenta[s] == 0:
                peso_vent += esc.peso(s)
            cuenta[s] += 1
            j += 1
        distintas = sum(1 for v in cuenta.values() if v > 0)
        if distintas >= MIN_COMUNES and peso_vent > mejor[0]:
            mejor = (peso_vent, i, j)
        s = hits[i][1]
        cuenta[s] -= 1
        if cuenta[s] == 0:
            peso_vent -= esc.peso(s)
    parecido = mejor[0] / total if total else 0.0
    if mejor[2] is None or parecido < UMBRAL_CITA:
        return "", -1, round(parecido, 3)
    a_ficha, b_ficha = hits[mejor[1]][0], hits[mejor[2] - 1][0]
    a, b = esc.fichas[a_ficha][0], esc.fichas[b_ficha][1]
    cita = _ajustar_cita(esc, a, b)
    return cita, a, round(parecido, 3)


def _ajustar_cita(esc: "_Escrito", a: int, b: int) -> str:
    """Del primer al último acierto, en palabras enteras, entre 10 y 40."""
    norm, ini_pal, fin_pal = esc.norm, esc.ini_pal, esc.fin_pal
    if not ini_pal:
        return ""
    i = max(0, bisect.bisect_right(ini_pal, a) - 1)
    j = max(i, bisect.bisect_left(fin_pal, b))
    if j - i + 1 > CITA_MAX:
        j = i + CITA_MAX - 1
    while j - i + 1 < CITA_MIN and (j + 1 < len(ini_pal) or i > 0):
        if j + 1 < len(ini_pal):
            j += 1
        else:
            i -= 1
    return norm[ini_pal[i]:fin_pal[min(j, len(fin_pal) - 1)]].strip()


# ═══════════════════════════════════════════════════════════════════════════
# LA FUNCIÓN DEL CONTRATO
# ═══════════════════════════════════════════════════════════════════════════
def segmentos(fases, escrito: str, es_recurso: bool = False,
              partir: bool = True) -> list:
    """El inventario: un segmento por argumento del resumen, anclado en el
    escrito. Nunca lanza: con un resumen vacío devuelve [].

    Cada segmento (contrato del Paso 2):
      id        «C1.a» (C = concepto, A = agravio; la letra, por argumento
                dentro del concepto)
      concepto  el ordinal del concepto (el que escribe el resumen)
      parrafo   el índice del párrafo en `resumen_conceptos` (líneas no vacías)
      texto     el argumento tal como lo resumió la fase 2, ≤ 80 palabras
      cita      10-40 palabras literales del escrito, o «» si no se encontró
      anclas    las anclas duras del segmento, sin los datos del asunto
    y dos campos más que el contrato no pide y la pantalla puede usar:
      pagina    la marca «p.X §Y» del resumen, si el párrafo la traía
      parecido  cuánto del segmento casa con la cita (0-1), para medir
    """
    try:
        piezas, _ = piezas_del_resumen(fases, partir=partir)
    except Exception:
        return []
    if not piezas:
        return []
    pref = "A" if es_recurso else "C"
    conteo = _get(fases, "conteo", None) or {}
    esc = _Escrito(escrito or "")
    propias = anclas_del_asunto(escrito or "")
    letras = Counter()
    fuera = []
    for pz in piezas:
        c = int(pz["concepto"] or 1)
        sid = f"{pref}{c}.{_letra(letras[c])}"
        letras[c] += 1
        texto = _recortar(pz["texto"])
        cita, _, parecido = ("", -1, 0.0)
        if esc.fichas:
            try:
                cita, _, parecido = anclar(pz["texto"], esc, _tramo_de(conteo, c, esc))
            except Exception:
                cita, parecido = "", 0.0
        fuera.append({
            "id": sid, "concepto": c, "parrafo": pz["parrafo"], "texto": texto,
            "cita": cita,
            "anclas": anclas_duras(pz["texto"] + " \n " + cita, excluir=propias),
            "pagina": (pz.get("paginas") or [""])[0],
            "parecido": parecido,
        })
    return fuera


def anomalias(fases) -> list:
    """Lo raro del resumen que el inventario tuvo que suplir (para medir)."""
    try:
        return piezas_del_resumen(fases)[1]
    except Exception as ex:
        return [f"no se pudo leer: {type(ex).__name__}"]


# ═══════════════════════════════════════════════════════════════════════════
# EL BLOQUE DEL PROMPT
# ═══════════════════════════════════════════════════════════════════════════
# Lección de la casa (tres veces medida): lo que va en un prompt como ejemplo
# se copia literal. Aquí van SÓLO datos —el identificador, el concepto, lo que
# el resumen dice que se alega, la cita del escrito y las anclas— y una línea
# que dice qué es cada campo. Ni una frase para imitar.
TEXTO_EN_BLOQUE = 45


def _ordinal_palabra(n: int) -> str:
    nombres = {1: "primer", 2: "segundo", 3: "tercer", 4: "cuarto", 5: "quinto",
               6: "sexto", 7: "séptimo", 8: "octavo", 9: "noveno", 10: "décimo"}
    return nombres.get(n, f"{n}.º")


def bloque_inventario(segs: list, q1: str) -> str:
    """El inventario como lista de datos para el prompt del estudio."""
    if not segs:
        return ""
    q1 = (q1 or "concepto de violación").strip()
    lineas = [
        "",
        "═" * 71,
        f"INVENTARIO DE ARGUMENTOS — {len(segs)} argumentos del escrito",
        "═" * 71,
        f"Un renglón por argumento. Campos: identificador · {q1} al que "
        "pertenece · lo que se alega, según el resumen · cita literal del "
        "escrito donde se plantea (si se localizó) · datos duros que trae "
        "(artículo con su ley, registro, expediente, cifra, fecha).",
        "",
    ]
    for s in segs:
        partes = [str(s.get("id") or ""),
                  f"{_ordinal_palabra(int(s.get('concepto') or 1))} {q1}",
                  _recortar(str(s.get("texto") or ""), TEXTO_EN_BLOQUE)]
        if s.get("cita"):
            partes.append(f"cita: «{s['cita']}»")
        if s.get("anclas"):
            partes.append("datos: " + "; ".join(s["anclas"][:6]))
        lineas.append(" · ".join(partes))
    return "\n".join(lineas) + "\n"
