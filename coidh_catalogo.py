"""
El catálogo de la Corte IDH y el resolvedor de citas de casos — 25-sep-2026.

    from coidh_catalogo import resolver_citas_coidh
    resolver_citas_coidh("Almonacid Arellano y otros Vs. Chile, párr. 124")
    → [{"doc_id": "C-154", "parrafo": 124, "seg": "sentencia", "url_oficial": ".../seriec_154_esp.pdf", ...}]

    python coidh_catalogo.py --construir <carpeta con list_CC.json, list_OC.json, list_SS.json, list_MP.json>
    python coidh_catalogo.py "voto razonado de García Ramírez en Trabajadores Cesados, párr. 12"

POR QUÉ EXISTE
--------------
Iurexia no reconoce casos: `_extract_legal_citations` (main.py) sabe de
artículos y leyes, no de «Caso Almonacid, párr. 124». Este módulo es la pieza
que falta, y es PURO: sin red, sin modelos, sin Qdrant. Lee un JSON que se
arma con el listado oficial del propio sitio de la Corte y dice qué documento,
qué párrafo y qué URL oficial nombra un texto. Qué se hace con eso (buscar el
párrafo en la colección `coidh`, abrir el visor) es de main.py.

EL CATÁLOGO SALE DEL LISTADO, NUNCA DEL NÚMERO
----------------------------------------------
`datos/coidh_catalogo.json` se arma con el servicio que alimenta el buscador
de jurisprudencia de corteidh.or.cr (`get_jurisprudencia_search_tipo.cfm`),
bajado el 25-sep-2026: 597 sentencias contenciosas (CC), 33 entradas de
opiniones consultivas (OC), 905 supervisiones (SS) y 785 medidas provisionales
(MP). Cuadre de las 597: el listado numera la Serie C del 1 al 598 y le falta
la 597 — no está en el sitio; 598 − 1 = 597, sin duplicados.

La URL es la del listado, normalizada a https://www. y minúsculas; jamás se
arma con el número de Serie, porque el número engaña:
  · la 385 (Ruiz Fuentes) enlaza `seriec_384_esp.pdf`, y la 384 (Perrone y
    Preckel) enlaza `seriec_385_esp.pdf`. Leí la portada de `seriec_384_esp.pdf`
    (copia local de la muestra): dice «CASO RUIZ FUENTES Y OTRA VS. GUATEMALA».
    El listado acierta; el nombre del archivo es el que está cruzado.
  · la 441 (Manuela) trae, además del suyo, un enlace a `seriec_461_esp.pdf`
    (la interpretación del mismo caso).
  · 13 sentencias con sufijo `_esp1`, 2 con `_esp2`, 9 con cero a la
    izquierda (`seriec_01_esp.pdf`), 44 con mayúsculas (`Seriec_89_esp.pdf`);
    dos supervisiones cuyo único PDF es una rectificación; cuatro enlaces con
    un espacio al final («castro_se_05.pdf »).
Todo eso queda marcado en `anomalias` de cada documento, salvo el espacio,
que sólo se quita (y la 163 marca como «enlace_extra» su anexo). Las 2,320 entradas
tienen URL; `url_listado` guarda la que decía el sitio. Resoluciones del mismo
asunto se agrupan en `caso_id` por ficha técnica, expediente o nombre: 428
casos (el plan contaba 436 agrupando por el texto crudo del nombre, que parte
en dos a García Prieto «y otro»/«y otros», a Cesados con y sin punto, a Panel
Blanca y Acevedo Buendía por el tipo de comillas, a U'wa por el apóstrofo, y en
cuatro a Hilaire, Constantine y Benjamin, que el sitio acumula en la ficha
269). 883 de 905 supervisiones y 329 de 785 medidas quedan enlazadas a su caso
(las demás medidas son «Asuntos» sin caso contencioso).

LA REGLA QUE MÁS IMPORTA: UN APELLIDO SUELTO NO ES UN CASO
----------------------------------------------------------
El prototipo del plan se disparaba en el 30.8 % de 4,000 preguntas reales
(medido el 25-sep-2026): «de la» — residuo de «Caso de la “Masacre de
Mapiripán”» — casaba 1,149 veces; «el amparo», 92; y «mi cliente José García
Rodríguez fue detenido» salía como García Rodríguez Vs. México (C-482). Aquí:
  · Los alias de una palabra y las parejas de apellidos SÓLO cuentan con una
    pista: «Caso X», «X Vs. Estado» / «X v. State» (con un Estado que tenga
    ese caso), «Serie C No. N» del mismo caso, «Corte IDH» / «Corte
    Interamericana» a ≤150 caracteres, un voto («voto … en X», «X, voto de…»)
    o la cita de párrafo pegada («Tzompaxtle, resolutivo 8»). Con pista de
    «Corte» o de párrafo, además, el nombre va con mayúscula, y no puede ser
    un pedazo de otro nombre propio («Viviana Gallardo», «Ruiz-Mateos»,
    «JUANA ANGEL HERNANDEZ CORZO»: en mayúsculas sostenidas la mayúscula no
    informa).
  · Si el apellido es de los frecuentes (García, López, González…), «caso X»
    no basta: el abogado habla de SU cliente. Hace falta además contexto
    interamericano en el texto, «Vs.», «Serie C» o «párr.» pegado.
  · Los apodos y nombres largos («Campo Algodonero», «Trabajadores Cesados
    del Congreso») valen sin pista sólo escritos como nombre propio (con
    mayúsculas o entre comillas): «los trabajadores cesados del ayuntamiento»
    no es un caso.
  · Frases jurídicas comunes que resultan ser nombres de casos («el amparo»,
    «Tribunal Constitucional», «Corte Suprema de Justicia», «San Juan»…)
    exigen «Vs.», «Serie C», o «Caso»/voto/párrafo con mayúscula.
  · «Serie A No. N» sin «Opinión Consultiva», «OC-» o «Corte IDH» cerca es,
    casi siempre, el Tribunal Europeo (Golder, Series A no. 18), no la OC-18.
  · Un alias que apunta a varios casos NUNCA se elige en silencio: se
    desempata por Estado, año, fecha, autor del voto o nombre propio de la
    resolución, y si no alcanza, el resultado sale sin `doc_id` y con
    `candidatos`, para que el modelo pregunte.
  · Revisión escéptica del 25-sep-2026 (todas se disparaban antes):
    «en el caso de Hidalgo» (una palabra tras «caso de» pide mayúscula y la
    Corte IDH en el texto); «el voto en Hidalgo» (un voto sólo es judicial
    con calificativo o con nombre de juez); «Ley de Hacienda de Hidalgo,
    párrafo 4» (un párrafo pegado a «… de X» no es pista, ni el de «párrafo
    3 del artículo 20»); «Hermanos Gómez S.A.», «Comunidad Campesina de San
    Pedro» (descriptor + una palabra pide pista, no sólo mayúscula); «OC-15»
    sin año ni Corte IDH (orden de compra); «acciones Serie C 500» sin «No.»
    ni Corte IDH; y en el seguimiento, «¿y el artículo 14, párrafo 2?» no
    hereda el caso anterior. Si el número de Serie impreso es de OTRO caso
    que el nombre de la misma cita (errata «Furlan … Serie C No. 212»), sale
    como conflicto con los dos candidatos, no dos resultados «alta».

VARIAS RESOLUCIONES POR CASO
----------------------------
Antes de 2004 fondo, reparaciones y excepciones son Series distintas cuyos
números de párrafo se enciman (Velásquez Rodríguez: C-1, C-4, C-7, más la
interpretación C-9). Si la cita pide un párrafo y el caso tiene más de una
resolución «sustantiva», se desempata por palabra («reparaciones»,
«excepciones», «fondo», «interpretación»), por fecha o por año; si no se
puede, salen candidatos sin elegir. Las interpretaciones, revisiones y
resoluciones de cumplimiento sólo se eligen si la cita las nombra: en la
práctica nadie cita «Caso X, párr. N» queriendo decir la interpretación, y la
cita oficial siempre lo dice («Interpretación de la Sentencia de…»). Cuando
el troceo por párrafo exista, `n_parrafos` por documento afinará esto: el
resolvedor ya descarta la resolución cuyo `n_parrafos` es menor que el pedido.
Un caso nombrado sin párrafo («caso Radilla») va a la resolución principal
(la de fondo) con confianza «media» si hay otras.

MEDIDO (25-sep-2026, sin red ni API; scripts en el scratchpad de la sesión)
---------------------------------------------------------------------------
  · 4,000 últimas preguntas de usuario (Supabase `messages`, sólo lectura):
    recortadas a 4,000 caracteres como en la revisión, se dispara en 1
    (0.03 %: «…Corte Interamericana…: campo algodonero», correcto); enteras
    (escritos pegados de hasta 583 mil caracteres), en 10 (0.25 %), las 10
    con contexto interamericano, 27 citas revisadas a mano y todas bien (2
    como candidatos: Suárez Rosero y «Tribunal Constitucional»). Disparos por
    apellido sin pista: 0. Latencia con el tope de 30 mil caracteres de
    `_extract_legal_citations`: p50 0.1 ms, p99 23 ms, máximo 47 ms. SIN el
    tope, el escrito de 583 mil caracteres tarda ~450 ms: quien lo llame,
    que recorte (con el tope, 5 disparos, 0.13 %).
    Tras la revisión (mismas 4,000): 1 recortadas; enteras 11, el único
    cambio es uno nuevo y correcto («Tzompaxtle Tecpile y otros» junto a la
    Corte IDH: la cola «y otros» en minúsculas ya no tumba la mayúscula).
  · Oro no circular: 460 citas de los cuadernillos con «Serie C No. N,
    párr.» impreso, resueltas SIN el número (nombre + acto + fecha): 458
    coinciden con el número impreso; las otras 2 son erratas del cuadernillo
    (Furlan impreso como 212 y Herrera Ulloa como 109; el listado y la fecha
    dicen 246 y 107). Con el número impreso, esas 2 salen como conflicto
    (candidatos: el caso nombrado y el del número). Con sólo «Caso X Vs.
    Estado, párr. N»: 441 exactas, 17 con candidatos (casos anteriores a
    2004), 0 errores fuera de esas erratas.
  · Los 597 nombres oficiales: 597/597 con la cita completa y sin el número.

Sin red y sin dependencias fuera de la biblioteca estándar.
"""
from __future__ import annotations

import datetime
import html as _html
import json
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

RUTA_CATALOGO = Path(__file__).resolve().parent / "datos" / "coidh_catalogo.json"
VERSION_CATALOGO = 1

# ───────────────────────────────────────────────────────────── normalización

_PLIEGUE: Dict[str, str] = {}


def _plegar(s: str) -> str:
    """Minúsculas y sin acentos, CARÁCTER POR CARÁCTER: la salida mide lo mismo
    que la entrada, así una posición en el texto plegado es la misma en el
    original y la evidencia se recorta del texto que escribió el usuario."""
    out = []
    for ch in s or "":
        r = _PLIEGUE.get(ch)
        if r is None:
            d = unicodedata.normalize("NFD", ch)
            base = "".join(c for c in d if unicodedata.category(c) != "Mn").lower()
            r = base[:1] if base else " "
            _PLIEGUE[ch] = r
        out.append(r)
    return "".join(out)


def _plano(plegado: str) -> str:
    """Todo lo que no es letra o dígito, a espacio (misma longitud)."""
    return re.sub(r"[^a-z0-9]", " ", plegado)


def _norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", _plegar(s)))


def _slug(s: str) -> str:
    return "-".join(re.findall(r"[a-z0-9]+", _plegar(s)))


# ─────────────────────────────────────────────────────── vocabulario curado
# Palabras de los nombres de casos que son español corriente o descriptores
# genéricos: nunca forman un alias de una palabra, y un alias que las lleva es
# «descriptivo» (necesita escribirse como nombre propio). Sacadas a mano de
# las 765 palabras distintas de los 597 nombres (conteo del 25-sep-2026).
GENERICAS = set("""
comunidad comunidades indigena indigenas masacre masacres pueblo pueblos trabajadores cesados cesantes
jubilados familia familiares miembros sus hermanos hermanas ninos ninas nino nina personas asociacion
nacional tribunal constitucional corte suprema justicia congreso contraloria campo aldea abogados activista
administracion administrativo adolescentes afrodescendientes agua aguas aledanos altos amparo azul bello
blanca bueno calle caliente campesina casa centros civil colectivo comerciantes contencioso corporacion
cristo defensor derechos desaparecidos desplazadas detencion diario dirigentes dominicanas empleados empresa
esperanza expulsadas extrabajadores fabrica favela fecundacion federacion flor fuegos genesis grande guerra
habitantes hacienda haitianas hija humanos instituto integrantes internacion judicial lugares maritimos
memoria menor menores militantes militar mujeres municipalidades municipio negra negro nuestra operacion
organismo palmeras panel penal pensionistas piedra plan pollo portuarios primera profesores provisoria
puertos punta radio recluidos reeducacion reten rio servicio sexual sindicato superintendencia television
tentacion tierra tortura tributaria triunfo ultima unico union vecinas verde vereda victimas vitro in dos
cinco diecinueve patriotica sumo lugares palacio cuenca empresa sociedad san santa santo domingo activa cayos
mexico chile peru guatemala colombia argentina brasil uruguay ecuador honduras paraguay venezuela bolivia
salvador panama nicaragua surinam suriname barbados costa rica haiti dominicana trinidad tobago bogota caracas
ordenes
""".split())

# Colas que no cuentan como palabra del nombre: «Atala Riffo y niñas» son dos
# apellidos, no cuatro palabras.
COLA_TOKENS = {"otros", "otras", "otro", "otra", "familiares", "familia", "sus", "miembros", "hija", "hijo",
               "hijas", "hijos", "ninas", "et", "al"}

# Nombres de casos que son también palabras de todos los días: con «caso»
# sólo cuentan escritos con mayúscula («caso Grande», no «un caso grande»).
PALABRAS_COMUNES = set("""
grande rico rios blanco luna vera cruz lagos flores palma rosario mina torres reyes santos franco duque bravo
leon valle campos castillo guerrero delgado navarro ramos rojas salamanca merino roca cordero cuadra montero
duran sales maldonado caballero tamayo moya bernal rangel pereira portugal silva soler rosero lima
""".split())

# Nombres de pila: «caso Juan…» es cómo se narra un asunto propio.
NOMBRES_PILA = set("""
juan jose maria manuel carlos luis jorge pedro ana rosa miguel antonio francisco jesus martin omar beatriz
vicky leonela barbara nicolle yvon manuela lucas alfonso ricardo digna myrna claude heliodoro nadege humberto
""".split())

# Apellidos frecuentes en México y en la región. «El caso García» es, casi
# siempre, el cliente del abogado: sin contexto interamericano no se dispara.
APELLIDOS_FRECUENTES = set("""
garcia lopez gonzalez gonzales rodriguez hernandez martinez perez sanchez ramirez flores gomez diaz torres
reyes cruz morales ortiz gutierrez ramos ruiz alvarez mendoza castillo jimenez vazquez vasquez moreno romero
herrera medina aguilar vargas castro chavez juarez dominguez guzman velazquez velasquez rojas contreras
salazar luna ortega rios soto silva mejia fernandez nunez acosta mendez delgado cabrera cortes
guerrero campos rivera espinoza espinosa sandoval navarro valencia pena leon marin santos franco vega
fuentes blanco soto molina suarez castaneda palacios cardenas carrillo avila ibarra zapata trujillo
""".split())

# Frases que son nombre de caso y a la vez lenguaje jurídico corriente.
FRASES_COMUNES = set((
    "el amparo", "amparo", "tribunal constitucional", "del tribunal constitucional",
    "corte suprema de justicia", "corte suprema", "san juan", "san miguel", "santa rosa", "santo domingo",
    "i v", "asociacion civil", "defensor de derechos humanos", "trabajadores cesados", "fecundacion in vitro",
    "ninos de la calle", "comunidad indigena", "comunidades indigenas", "pueblo indigena", "pueblos indigenas",
    "diecinueve comerciantes", "palacio de justicia", "radio caracas television", "nuestra tierra",
    "cinco pensionistas", "instituto de reeducacion del menor", "personas dominicanas y haitianas expulsadas",
    "adolescentes recluidos", "operacion genesis", "corte primera de lo contencioso administrativo",
    "desaparecidos del palacio de justicia", "mujeres victimas de tortura sexual", "mujeres victimas",
    "comunidad garifuna", "empleados de la fabrica", "integrantes y militantes", "personas dominicanas",
    "trabajadores de la hacienda", "federacion nacional", "asociacion nacional", "miembros del sindicato",
))

# Apodos y variantes que el listado no trae (inglés incluido).
ALIAS_CURADOS = {
    # caso (nombre normalizado + estado) → alias extra
    "gonzalez y otras campo algodonero|mexico": ["cotton field"],
    "ninos de la calle villagran morales y otros|guatemala": ["street children"],
    "penal miguel castro castro|peru": ["castro castro"],
    "masacre de mapiripan|colombia": ["mapiripan massacre"],
    "gudiel alvarez y otros diario militar|guatemala": ["military diary"],
    "artavia murillo y otros fecundacion in vitro|costa rica": ["in vitro fertilization"],
}

ESTADOS_INGLES = {
    "mexico": "México", "chile": "Chile", "peru": "Perú", "guatemala": "Guatemala", "colombia": "Colombia",
    "argentina": "Argentina", "brazil": "Brasil", "brasil": "Brasil", "uruguay": "Uruguay",
    "ecuador": "Ecuador", "honduras": "Honduras", "paraguay": "Paraguay", "venezuela": "Venezuela",
    "bolivia": "Bolivia", "el salvador": "El Salvador", "panama": "Panamá", "nicaragua": "Nicaragua",
    "suriname": "Surinam", "surinam": "Surinam", "barbados": "Barbados", "costa rica": "Costa Rica",
    "haiti": "Haití", "dominican republic": "República Dominicana", "republica dominicana": "República Dominicana",
    "trinidad and tobago": "Trinidad y Tobago", "trinidad y tobago": "Trinidad y Tobago",
}

MESES = dict(enero=1, febrero=2, marzo=3, abril=4, mayo=5, junio=6, julio=7, agosto=8, septiembre=9,
             setiembre=9, octubre=10, noviembre=11, diciembre=12)
MESES_EN = dict(january=1, february=2, march=3, april=4, may=5, june=6, july=7, august=8, september=9,
                october=10, november=11, december=12)

# Variantes de un mismo juez en el campo `votos` del listado → apellido canónico.
AUTORES_VARIANTES = {
    "eduardo vio grossi": "vio grossi",
    "margarette macaulay": "macaulay",
    "f caldas": "caldas",
    "pedro nikken": "nikken",
}


# ════════════════════════════════════════════════ 1 · CONSTRUIR EL CATÁLOGO

def _fecha_iso(texto: str) -> Optional[str]:
    """La última fecha escrita del texto: «26 de septiembre de 2006», «21 de
    noviembre 2017», «26 noviembre de 2002», «30 de Noviembre de 2007»."""
    mejor = None
    for m in re.finditer(r"(\d{1,2})\s*[º°]?\s+(?:de\s+)?([A-Za-zÁÉÍÓÚáéíóú]+)\s+(?:de\s+|del\s+)?(\d{4})",
                         texto or ""):
        mes = MESES.get(_norm(m.group(2)))
        if mes:
            mejor = f"{m.group(3)}-{mes:02d}-{int(m.group(1)):02d}"
    return mejor


def _normalizar_url(u: str) -> str:
    """https://www. y minúsculas (decisión del plan, 3.1): el listado mezcla
    http://, sin www y `Seriec_`. El original se guarda en `url_listado`."""
    u = (u or "").strip()
    m = re.match(r"(?i)^https?://(?:www\.)?corteidh\.or\.cr(/.*)$", u)
    return "https://www.corteidh.or.cr" + m.group(1).lower() if m else u


def _ids_sitio(links: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    ficha = exp = None
    for l in links:
        m = re.search(r"nId_Ficha=(\d+)", l)
        if m and not ficha:
            ficha = m.group(1)
        m = re.search(r"nId_expediente=(\d+)", l)
        if m and not exp:
            exp = m.group(1)
    return ficha, exp


def _tipo_resolucion(acto: str) -> Tuple[str, List[str]]:
    """«Excepciones Preliminares, Fondo, Reparaciones y Costas» → tipo y
    componentes. Las 37 formas distintas del listado caen en 11 tipos."""
    a = _norm(acto)
    if "interpretacion" in a:
        return "interpretacion", []
    if "revision" in a:
        return "revision", []
    if "cumplimiento" in a:
        return "cumplimiento", []
    if "competencia" in a:
        return "competencia", []
    comp = [c for c, w in (("excepciones", "excepci"), ("fondo", "fondo"), ("reparaciones", "reparaciones"))
            if w in a]
    if not a:
        # 13 sentencias del listado no rotulan el acto (Vargas Areco, Trueba
        # Arciniega, Mapiripán…): son la sentencia del caso.
        return "sentencia", ["fondo", "reparaciones"]
    if not comp:
        return "otro", []
    return "_".join(comp), comp


def _partir_nombre(nombre: str) -> Dict[str, Any]:
    """Descompone el nombre oficial: apodos entre comillas, contenido entre
    paréntesis y la base sin artículos al frente ni «y otros» al final."""
    apodos = [a.strip() for a in re.findall(r"[“\"«]([^”\"»]{2,90})[”\"»]", nombre)]
    parent = []
    for p in re.findall(r"\(([^()]*)\)", nombre):
        p2 = re.sub(r"[“\"«”»]", "", p).strip()
        if p2 and p2 not in apodos:
            parent.append(p2)
    base = re.sub(r"\([^()]*\)", " ", nombre)
    base = re.sub(r"[“\"«”»]", " ", base)
    base = re.sub(r"\s+", " ", base).strip(" .,")
    return dict(apodos=apodos, parentesis=parent, base=base)


_COLA = re.compile(r"\s+(?:y|e)\s+(?:otros|otras|otro|otra|familiares|familia|sus\s+miembros|sus\s+familiares"
                   r"|hija|hijo|hijas|hijos|ninas)$")
_ARTICULOS = {"el", "la", "los", "las", "de", "del", "y", "e", "en", "a", "do", "da", "dos", "of", "the"}


def _sin_articulo_inicial(n: str) -> str:
    return re.sub(r"^(?:de\s+la|de\s+los|de\s+las|del|de)\s+", "", n)


def _sin_cola(n: str) -> str:
    prev = None
    while prev != n:
        prev, n = n, _COLA.sub("", n)
    return n


def _slug_autor(nombre: str) -> str:
    n = _norm(re.sub(r"(?i)\bad[\s_-]?hoc\b", " ", nombre))
    n = AUTORES_VARIANTES.get(n, n)
    return "-".join(n.split())


def _parse_votos(votos: Sequence[str], links: Sequence[str]) -> List[Dict[str, Any]]:
    """«Jueces Mudrovitsch y Pérez Manrique» → un voto conjunto con dos
    autores. El slug sale de ESTE campo y no del título que el troceador lea
    en el PDF (revisión B.13): si mañana cambia el troceador, el id del punto
    no cambia."""
    urls_voto = [l for l in links if "/votos/" in l.lower()]
    usados: Set[str] = set()
    out = []
    for i, v in enumerate(votos):
        crudo = re.sub(r"\s+", " ", v).strip()
        resto = re.sub(r"^(?:Jueces|Juezas|Jueza|Juez)\s+", "", crudo)
        partes = [p.strip() for p in re.split(r",\s*|\s+y\s+", resto) if p.strip()]
        autores = []
        for p in partes:
            ad_hoc = bool(re.search(r"(?i)\bad[\s_-]?hoc\b", p))
            nombre = re.sub(r"(?i)\bad[\s_-]?hoc\b", " ", p).strip()
            nombre = re.sub(r"\s+", " ", nombre)
            autores.append(dict(nombre=nombre, slug=_slug_autor(nombre), ad_hoc=ad_hoc))
        url = None
        if autores:
            clave = autores[0]["slug"].split("-")[0]
            cand = [u for u in urls_voto if u not in usados and clave and clave in _norm(u.rsplit("/", 1)[-1])]
            if len(cand) >= 1:
                url = cand[0]
                usados.add(url)
        out.append(dict(n=i + 1, autor=crudo, autores=autores, slug="+".join(a["slug"] for a in autores),
                        conjunto=len(autores) > 1, url=_normalizar_url(url) if url else None))
    return out


_RX_CC = re.compile(r"^Corte IDH\.\s+(?:Caso\s+)?(?P<nombre>.+?)\s+Vs\.\s+(?P<estado>[^.]+?)\.\s*(?P<resto>.*)$")
_RX_RESTO = re.compile(r"^(?P<acto>.*?)\s*(?:Sentencia|Resoluci[oó]n)\b(?P<fecha>.*?)Serie C No\.?\s*(?P<n>\d+)")


def _pdfs(links: Sequence[str], carpeta: str, prefijo: str) -> Dict[str, Any]:
    """Separa, de los enlaces de una entrada, el PDF de la resolución de sus
    resúmenes, rectificaciones, anexos y copias de alto contraste."""
    links = [l.strip() for l in links]          # «castro_se_05.pdf » trae espacio al final
    pdfs = [l for l in links if l.lower().split("?")[0].endswith(".pdf")]
    propios = [l for l in pdfs if re.search(rf"(?i)/docs/{carpeta}/(?:articulos/)?{prefijo}", l)
               and not re.search(r"(?i)altocontraste", l)]
    return dict(
        principal=propios[0] if propios else None,
        otros=[l for l in propios[1:]],
        resumen=next((l for l in pdfs if "resumen" in l.lower()), None),
        rectificaciones=[l for l in pdfs if "rectificacion" in l.lower()],
        anexos=[l for l in pdfs if "/anexo" in l.lower()],
        sueltos=[l for l in pdfs if l not in propios and "resumen" not in l.lower()
                 and "rectificacion" not in l.lower() and "/anexo" not in l.lower()],
    )


def _anomalias_url(url: Optional[str], numero: Optional[int], prefijo: str) -> List[str]:
    if not url:
        return ["sin_pdf"]
    out = []
    fn = url.rsplit("/", 1)[-1]
    # http:// y la falta de «www.» (521 y 81 de las 597) no son anomalías: la
    # normalización las resuelve y `url_listado` guarda lo que decía el sitio.
    if fn != fn.lower():
        out.append("mayusculas")
    m = re.match(rf"(?i){prefijo}_(\d+)_(?:esp?|es)(\d*)\.pdf$", fn)
    if m:
        if m.group(1).startswith("0"):
            out.append("cero_izquierda")
        if m.group(2):
            out.append(f"sufijo_esp{m.group(2)}")
        if numero is not None and int(m.group(1)) != numero:
            out.append(f"archivo_con_otro_numero:{fn}")
    else:
        out.append(f"nombre_de_archivo_raro:{fn}")
    return out


def _palabras_clave(carpeta: Path) -> Dict[str, List[str]]:
    """«Palabras Claves» de las fichas técnicas ya bajadas (ficha_<id>.html).
    El 25-sep-2026 sólo estaba la 335 (Almonacid); no se baja nada aquí."""
    out = {}
    for f in sorted(carpeta.glob("ficha_*.html")):
        h = f.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"Palabras Claves:(?:&nbsp;|\s)*</td>\s*<td[^>]*>(.*?)</td>", h, re.S)
        if not m:
            continue
        t = _html.unescape(re.sub(r"<[^>]+>", " ", m.group(1))).replace("\xa0", " ")
        pals = [re.sub(r"\s+", " ", p).strip() for p in t.split(",") if p.strip()]
        if pals:
            out[f.stem.split("_", 1)[1]] = pals
    return out


def _clase_alias(tokens: Tuple[str, ...], origen: str) -> str:
    """Qué tan peligroso es un alias. Determina qué pista necesita."""
    texto = " ".join(tokens)
    if texto in FRASES_COMUNES or (len(tokens) == 2 and tokens[0] in ("san", "santa", "santo")):
        return "comun"
    sig = [t for t in tokens if t not in _ARTICULOS and t not in COLA_TOKENS]
    if sig and all(len(t) == 1 for t in sig):
        return "iniciales"
    if len(sig) <= 1:
        return "palabra"
    if origen in ("apodo", "curado"):
        return "apodo"
    # Descriptivo si ARRANCA con un descriptor («Comunidad…», «Masacre…»,
    # «Trabajadores…»); «Flor Freire» o «Azul Rojas Marín» son personas.
    if sig[0] in GENERICAS or sig[0].isdigit():
        return "descriptivo"
    if len(sig) == 2:
        return "apellidos"
    return "nombre_largo"


_RIESGO = {"comun": 0, "iniciales": 1, "palabra": 2, "apellidos": 3, "descriptivo": 4, "nombre_largo": 5,
           "apodo": 6}


def _alias_de_casos(casos: Dict[str, Dict[str, Any]]) -> None:
    """Llena `alias` de cada caso: [{"a": texto normalizado, "clase": …, "origen": …}]."""
    df: Dict[str, int] = {}
    for c in casos.values():
        toks = set()
        for n in c["nombres"]:
            toks.update(_norm(re.sub(r"[“\"«”»]", "", n)).split())
        for t in toks:
            df[t] = df.get(t, 0) + 1

    for cid, c in casos.items():
        cand: Dict[str, str] = {}

        def add(texto: str, origen: str):
            t = _norm(texto)
            if not t or t in _ARTICULOS:
                return
            cand.setdefault(t, origen)

        for nombre in c["nombres"]:
            p = _partir_nombre(nombre)
            base = _sin_articulo_inicial(_norm(p["base"]))
            base_corta = _sin_cola(base)
            add(base, "nombre")
            add(base_corta, "nombre")
            for a in p["apodos"]:
                a2 = _sin_articulo_inicial(_norm(a))
                add(a2, "apodo")
                add(_sin_cola(a2), "apodo")
            for par in p["parentesis"]:
                par2 = _sin_cola(_sin_articulo_inicial(_norm(par)))
                add(par2, "apodo" if any(t in GENERICAS for t in par2.split()) else "nombre")
            toks = base_corta.split()
            sig = [t for t in toks if t not in _ARTICULOS]
            if not sig:
                continue
            # Base que empieza con descriptores («Penal Miguel Castro Castro»):
            # también vale sin ellos.
            k = 0
            while k < len(toks) and (toks[k] in GENERICAS or toks[k] in _ARTICULOS):
                k += 1
            if 0 < k < len(toks) and len(toks) - k >= 2:
                add(" ".join(toks[k:]), "nombre")
            # Nombre que arranca con nombres de pila: también sin ellos. El
            # cuadernillo cita «Caso Maldonado Vargas y otros Vs. Chile»; el
            # listado dice «Omar Humberto Maldonado Vargas y otros».
            k = 0
            while k < len(toks) - 1 and toks[k] in NOMBRES_PILA and toks[k + 1] not in _ARTICULOS:
                k += 1
            if k:
                resto = toks[k:]
                add(" ".join(resto), "nombre")
                sig_r = [t for t in resto if t not in _ARTICULOS and t not in COLA_TOKENS]
                if sig_r and sig_r[0] not in GENERICAS and len(sig_r[0]) >= 3:
                    add(sig_r[0], "nombre")
            # Prefijo de dos palabras significativas en nombres de persona:
            # «Cabrera García» de «Cabrera García y Montiel Flores».
            # También «Trabajadores Cesados» de «Trabajadores Cesados del
            # Congreso» (y de Petroperú, y de ENAPU: queda ambiguo, a propósito).
            sig_p = [t for t in sig if len(t) > 1 and t not in COLA_TOKENS]
            if sig_p and sig_p[0] not in NOMBRES_PILA and len(sig_p) >= 3:
                j = toks.index(sig_p[1])
                add(" ".join(toks[: j + 1]), "nombre")
            # Una palabra: el primer apellido, y cualquier palabra propia que
            # sólo tenga este caso («atenco», «mapiripan»).
            primero = sig[0]
            solo_nombre = len([t for t in sig if t not in COLA_TOKENS]) == 1
            if primero not in GENERICAS and len(primero) >= 3 and (primero not in NOMBRES_PILA or solo_nombre):
                add(primero, "nombre")
            for t in set(_norm(re.sub(r"[“\"«”»]", "", nombre)).split()):
                if df.get(t) == 1 and len(t) >= 5 and t not in GENERICAS and t not in NOMBRES_PILA \
                        and not t.isdigit():
                    add(t, "nombre")
        for extra in ALIAS_CURADOS.get(c["_clave_curado"], []):
            add(extra, "curado")
        c["alias"] = sorted(
            ({"a": a, "clase": _clase_alias(tuple(a.split()), o), "origen": o} for a, o in cand.items()),
            key=lambda x: (-len(x["a"]), x["a"]))


def construir(carpeta: str) -> Dict[str, Any]:
    """Arma el catálogo desde los list_*.json del sitio. No toca la red."""
    d = Path(carpeta)
    listas = {s: json.loads((d / f"list_{s}.json").read_text(encoding="utf-8")) for s in ("CC", "OC", "SS", "MP")}
    vacias = [s for s, l in listas.items() if not l]
    if vacias:
        # El servicio responde con código 200 aunque falle (error de ColdFusion):
        # una lista vacía es un listado caído, nunca «la Corte no tiene nada».
        raise ValueError(f"listado vacío en {vacias}: no se construye el catálogo")
    claves = _palabras_clave(d)
    documentos: Dict[str, Dict[str, Any]] = {}
    avisos: List[str] = []

    # ── Sentencias contenciosas
    for e in listas["CC"]:
        cita = re.sub(r"\s+", " ", e["cita"]).strip()
        m = _RX_CC.match(cita)
        if not m:
            avisos.append(f"CC sin parsear: {cita[:120]}")
            continue
        nombre = m.group("nombre").strip()
        if nombre.endswith(").") or (nombre.endswith(".") and not re.search(r"\b[A-Z]\.$", nombre)):
            nombre = nombre[:-1].strip()        # «(Aguado Alfaro y otros). Vs. Perú»
        m2 = _RX_RESTO.match(m.group("resto"))
        if not m2:
            avisos.append(f"CC sin acto/fecha: {cita[:120]}")
            continue
        n = int(m2.group("n"))
        acto = m2.group("acto").strip().rstrip(".").strip()
        tipo, comp = _tipo_resolucion(acto)
        ficha, exp = _ids_sitio(e["links"])
        pdf = _pdfs(e["links"], "casos", "seriec_")
        anom = _anomalias_url(pdf["principal"], n, "seriec")
        for o in pdf["otros"]:
            anom.append(f"enlace_extra:{o.rsplit('/', 1)[-1]}")
        doc_id = f"C-{n}"
        if doc_id in documentos:
            avisos.append(f"Serie C {n} repetida en el listado")
        documentos[doc_id] = dict(
            doc_id=doc_id, serie="C", serie_num=n, clase="CC", nombre=nombre, estado=m.group("estado").strip(),
            acto=acto, tipo=tipo, componentes=comp, fecha=_fecha_iso(m2.group("fecha")), cita=cita,
            url_oficial=_normalizar_url(pdf["principal"]) if pdf["principal"] else None,
            url_listado=pdf["principal"], resumen_url=_normalizar_url(pdf["resumen"]) if pdf["resumen"] else None,
            otros_pdf=[_normalizar_url(x) for x in pdf["otros"] + pdf["rectificaciones"] + pdf["anexos"]],
            catalogo_id=e.get("id"), ficha_id=ficha, expediente_id=exp,
            votos=_parse_votos(e.get("votos") or [], e["links"]), anomalias=anom,
            palabras_clave=claves.get(ficha) if ficha else None,
        )

    # ── Opiniones consultivas (y la decisión Viviana Gallardo, que el sitio
    # publica en la Serie A con el número 101)
    for e in listas["OC"]:
        cita = re.sub(r"\s+", " ", e["cita"]).strip()
        m = re.search(r"Serie A No\.?\s*(\d+)", cita)
        if not m:
            avisos.append(f"OC sin Serie A: {cita[:120]}")
            continue
        n = int(m.group(1))
        moc = re.search(r"Opini[oó]n Consultiva OC-(\d+)/(\d+)", cita)
        pdf = _pdfs(e["links"], "opiniones", "seriea_")
        titulo = re.sub(r"^Corte IDH\.\s+", "", cita)
        titulo = re.split(r"\.\s+Opini[oó]n Consultiva|\.\s+Serie A", titulo)[0].strip()
        anio_oc = None
        if moc:
            aa = int(moc.group(2))
            anio_oc = aa + (1900 if aa >= 79 else 2000)
        ficha, exp = _ids_sitio(e["links"])
        documentos[f"A-{n}"] = dict(
            doc_id=f"A-{n}", serie="A", serie_num=n, clase="OC", nombre=titulo, estado=None,
            acto="Opinión Consultiva" if moc else "Decisión", tipo="opinion_consultiva" if moc else "decision",
            componentes=[], oc=f"OC-{moc.group(1)}/{moc.group(2)}" if moc else None, oc_anio=anio_oc,
            fecha=_fecha_iso(cita), cita=cita,
            url_oficial=_normalizar_url(pdf["principal"]) if pdf["principal"] else None,
            url_listado=pdf["principal"], resumen_url=_normalizar_url(pdf["resumen"]) if pdf["resumen"] else None,
            otros_pdf=[_normalizar_url(x) for x in pdf["otros"]], catalogo_id=e.get("id"), ficha_id=ficha,
            expediente_id=exp, votos=_parse_votos(e.get("votos") or [], e["links"]),
            anomalias=_anomalias_url(pdf["principal"], n, "seriea") + ([] if moc else ["no_es_opinion_consultiva"]),
            palabras_clave=None,
        )

    # ── Casos: se agrupan las resoluciones de un mismo asunto. El sitio las
    # une por ficha técnica y por expediente; el nombre sirve cuando faltan
    # los dos (31 de 597). Así García Prieto «y otro»/«y otros» es un caso, y
    # Hilaire, Constantine y Benjamin (acumulados, ficha 269) también.
    padre: Dict[str, str] = {}

    def raiz(x):
        while padre.setdefault(x, x) != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    def unir(a, b):
        padre[raiz(a)] = raiz(b)

    def clave_nombre(nombre, estado):
        return _norm(_sin_cola(_sin_articulo_inicial(_norm(re.sub(r"\([^()]*\)", " ", nombre))))) + "|" + _norm(estado)

    ccs = [x for x in documentos.values() if x["clase"] == "CC"]
    for x in ccs:
        k = "n:" + clave_nombre(x["nombre"], x["estado"])
        unir("d:" + x["doc_id"], k)
        if x["ficha_id"]:
            unir("d:" + x["doc_id"], "f:" + x["ficha_id"])
        if x["expediente_id"]:
            unir("d:" + x["doc_id"], "x:" + x["expediente_id"])
    grupos: Dict[str, List[Dict[str, Any]]] = {}
    for x in ccs:
        grupos.setdefault(raiz("d:" + x["doc_id"]), []).append(x)

    casos: Dict[str, Dict[str, Any]] = {}
    llave_a_caso: Dict[str, str] = {}
    for g in grupos.values():
        g.sort(key=lambda x: x["serie_num"])
        sust = [x for x in g if x["componentes"]]
        con_fondo = [x for x in sust if "fondo" in x["componentes"]]
        principal = (con_fondo or sust or g)[0]
        base = _sin_cola(_sin_articulo_inicial(_norm(_partir_nombre(principal["nombre"])["base"])))
        cid = f"{'-'.join(base.split()[:6])}-{_slug(principal['estado'])}"
        while cid in casos:
            cid += "-b"
        nombres = sorted({x["nombre"] for x in g})
        casos[cid] = dict(
            caso_id=cid, nombre=principal["nombre"], estado=principal["estado"], nombres=nombres,
            principal=principal["doc_id"], resoluciones=[x["doc_id"] for x in g], supervisiones=[], medidas=[],
            ficha_id=next((x["ficha_id"] for x in g if x["ficha_id"]), None),
            expediente_id=next((x["expediente_id"] for x in g if x["expediente_id"]), None),
            _clave_curado=_norm(principal["nombre"]) + "|" + _norm(principal["estado"]),
        )
        for x in g:
            x["caso_id"] = cid
            x["acto_principal"] = x["doc_id"] == principal["doc_id"]
            llave_a_caso["n:" + clave_nombre(x["nombre"], x["estado"])] = cid
            if x["ficha_id"]:
                llave_a_caso["f:" + x["ficha_id"]] = cid
            if x["expediente_id"]:
                llave_a_caso["x:" + x["expediente_id"]] = cid

    # ── Supervisiones y medidas provisionales: se enlazan al caso por ficha o
    # expediente (864 de 905 supervisiones traen expediente) y, si no, por nombre.
    for serie, carpeta in (("SS", "supervisiones"), ("MP", "medidas")):
        for e in listas[serie]:
            cita = re.sub(r"\s+", " ", e["cita"]).strip()
            ficha, exp = _ids_sitio(e["links"])
            cuerpo = re.sub(r"^Corte IDH\.\s+", "", cita)
            m = re.match(r"^(?:Caso|Asunto)?\s*(?P<nombre>.+?)\s+(?:Vs\.|respecto(?:\s+de)?|con respecto a)\s+"
                         r"(?P<estado>[^.]+?)\.\s*(?P<resto>.*)$", cuerpo)
            nombre = m.group("nombre").strip() if m else cuerpo[:120]
            estado = m.group("estado").strip() if m else None
            resto = m.group("resto") if m else cuerpo
            acto = re.split(r"\.\s+Resoluci[oó]n", resto)[0].strip().rstrip(".")
            fecha = _fecha_iso(resto)
            cid = None
            if ficha and "f:" + ficha in llave_a_caso:
                cid = llave_a_caso["f:" + ficha]
            elif exp and "x:" + exp in llave_a_caso:
                cid = llave_a_caso["x:" + exp]
            elif estado:
                primero = re.split(r",\s*Caso\s+", nombre)[0]
                cid = llave_a_caso.get("n:" + clave_nombre(primero, estado))
            pdf = _pdfs(e["links"], carpeta, "")
            principal = pdf["principal"] or (pdf["sueltos"][0] if pdf["sueltos"] else None)
            es_rectif = False
            if not principal and pdf["rectificaciones"]:
                # 2 supervisiones cuyo único PDF es una rectificación de la
                # sentencia (C-441, C-445): ese archivo ES la resolución.
                principal, es_rectif = pdf["rectificaciones"][0], True
            stem = _slug(cid or nombre)[:60] if (cid or nombre) else "sin-nombre"
            doc_id = f"{serie}-{stem}-{fecha or 'sin-fecha'}"
            k = 2
            while doc_id in documentos:
                doc_id = f"{serie}-{stem}-{fecha or 'sin-fecha'}-{k}"
                k += 1
            anom = [] if principal else ["sin_pdf"]
            if es_rectif:
                anom.append("pdf_es_rectificacion")
            if not fecha:
                anom.append("sin_fecha")
            documentos[doc_id] = dict(
                doc_id=doc_id, serie=serie, serie_num=None, clase=serie, nombre=nombre, estado=estado,
                acto=acto, tipo="supervision" if serie == "SS" else "medidas_provisionales", componentes=[],
                fecha=fecha, cita=cita, caso_id=cid,
                asunto=bool(re.match(r"^Asunto\b", cuerpo)),
                url_oficial=_normalizar_url(principal) if principal else None, url_listado=principal,
                resumen_url=None, otros_pdf=[_normalizar_url(x) for x in (pdf["otros"] + pdf["sueltos"][1:])],
                catalogo_id=e.get("id"), ficha_id=ficha, expediente_id=exp,
                votos=_parse_votos(e.get("votos") or [], e["links"]), anomalias=anom, palabras_clave=None,
            )
            if cid:
                casos[cid]["supervisiones" if serie == "SS" else "medidas"].append(doc_id)

    _alias_de_casos(casos)
    for c in casos.values():
        c.pop("_clave_curado", None)

    # ── Autores de votos: un slug estable por juez, con sus formas escritas.
    autores: Dict[str, Dict[str, Any]] = {}
    for x in documentos.values():
        for v in x["votos"]:
            for a in v["autores"]:
                r = autores.setdefault(a["slug"], dict(slug=a["slug"], nombre=a["nombre"], variantes=[], n_votos=0))
                r["n_votos"] += 1
                if a["nombre"] not in r["variantes"]:
                    r["variantes"].append(a["nombre"])

    cc_nums = sorted(x["serie_num"] for x in documentos.values() if x["clase"] == "CC")
    faltan = sorted(set(range(1, max(cc_nums) + 1)) - set(cc_nums))
    por_clase = {s: sum(1 for x in documentos.values() if x["clase"] == s) for s in ("CC", "OC", "SS", "MP")}
    cifras = dict(
        listado={s: len(listas[s]) for s in listas}, documentos=por_clase,
        serie_c=dict(minimo=cc_nums[0], maximo=cc_nums[-1], distintos=len(set(cc_nums)), faltan=faltan,
                     cuadre=f"{cc_nums[-1]} números − {len(faltan)} que faltan ({faltan}) = {len(set(cc_nums))}"),
        casos=len(casos), casos_con_varias_resoluciones=sum(1 for c in casos.values() if len(c["resoluciones"]) > 1),
        votos_cc=sum(len(x["votos"]) for x in documentos.values() if x["clase"] == "CC"),
        autores=len(autores),
        supervisiones_enlazadas=sum(1 for x in documentos.values() if x["clase"] == "SS" and x.get("caso_id")),
        medidas_enlazadas=sum(1 for x in documentos.values() if x["clase"] == "MP" and x.get("caso_id")),
        con_palabras_clave=sum(1 for x in documentos.values() if x.get("palabras_clave")),
        alias=sum(len(c["alias"]) for c in casos.values()),
    )
    anomalias = [dict(doc_id=x["doc_id"], anomalias=x["anomalias"]) for x in documentos.values() if x["anomalias"]]
    tipos_anom: Dict[str, int] = {}
    for x in anomalias:
        for a in x["anomalias"]:
            k = a.split(":")[0]
            tipos_anom[k] = tipos_anom.get(k, 0) + 1
    cifras["anomalias_por_tipo"] = dict(sorted(tipos_anom.items(), key=lambda kv: -kv[1]))
    cifras["urls_normalizadas"] = sum(1 for x in documentos.values()
                                      if x.get("url_listado") and x["url_listado"] != x.get("url_oficial"))
    return dict(
        version=VERSION_CATALOGO,
        generado=datetime.date.today().isoformat(),
        fuente=("POST https://www.corteidh.or.cr/get_jurisprudencia_search_tipo.cfm "
                "(nId_Tipo_Jurisprudencia=CC|OC|SS|MP, page_rows=3000, lang=es), bajado el 25-sep-2026; "
                "palabras clave de las fichas técnicas ya bajadas"),
        cifras=cifras, avisos=avisos, anomalias=anomalias, autores=autores, casos=casos, documentos=documentos,
    )


# ═══════════════════════════════════════════════════ 2 · CARGAR Y ÍNDICES

@lru_cache(maxsize=1)
def catalogo() -> Dict[str, Any]:
    """El catálogo y sus índices, una vez por proceso (~2,300 documentos)."""
    data = json.loads(RUTA_CATALOGO.read_text(encoding="utf-8"))
    docs = data["documentos"]
    casos = data["casos"]

    alias: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for cid, c in casos.items():
        for a in c["alias"]:
            k = tuple(a["a"].split())
            r = alias.setdefault(k, dict(casos=set(), clase=a["clase"]))
            r["casos"].add(cid)
            if _RIESGO[a["clase"]] < _RIESGO[r["clase"]]:
                r["clase"] = a["clase"]      # dos orígenes: manda el más peligroso
    max_len = max(len(k) for k in alias)

    # Autores: apellido completo, y el primer apellido si sólo lo tiene un juez
    # («Ferrer», «Cançado»; «García» no, que son tres).
    autores: Dict[Tuple[str, ...], str] = {}
    primeros: Dict[str, Set[str]] = {}
    for slug, a in data["autores"].items():
        for v in a["variantes"] + [slug.replace("-", " ")]:
            n = _norm(v)
            if n:
                autores[tuple(n.split())] = slug
                n2 = AUTORES_VARIANTES.get(n)
                if n2:
                    autores[tuple(n2.split())] = slug
        primeros.setdefault(slug.split("-")[0], set()).add(slug)
    for p, slugs in primeros.items():
        if len(slugs) == 1 and len(p) >= 4 and p not in APELLIDOS_FRECUENTES:
            autores.setdefault((p,), next(iter(slugs)))
    # «Ferrer Mac-Gregor» a secas, sin el segundo apellido
    for slug in list(data["autores"]):
        t = slug.split("-")
        if len(t) >= 3:
            autores.setdefault(tuple(t[:-1]), slug)

    estados = {}
    for c in casos.values():
        if c.get("estado"):
            estados[tuple(_norm(c["estado"]).split())] = c["estado"]
    for k, v in ESTADOS_INGLES.items():
        estados[tuple(k.split())] = v

    oc_por_num = {}
    for d in docs.values():
        if d["clase"] == "OC" and d.get("oc"):
            oc_por_num[int(d["oc"].split("-")[1].split("/")[0])] = d["doc_id"]

    return dict(data=data, docs=docs, casos=casos, alias=alias, max_len=max_len, autores=autores,
                autor_max=max(len(k) for k in autores), estados=estados,
                estado_max=max(len(k) for k in estados), oc_por_num=oc_por_num)


def documento(doc_id: str) -> Optional[Dict[str, Any]]:
    return catalogo()["docs"].get(doc_id)


def caso(caso_id: str) -> Optional[Dict[str, Any]]:
    return catalogo()["casos"].get(caso_id)


# ═══════════════════════════════════════════════════════ 3 · EL RESOLVEDOR

_RX_CONTEXTO_IDH = re.compile(
    r"\b(corte\s+idh|coidh|corte\s+interamericana|tribunal\s+interamericano|sistema\s+interamericano"
    r"|inter\s*-?\s*american\s+court|convencion\s+americana|cadh|pacto\s+de\s+san\s+jose"
    r"|convencionalidad|jurisprudencia\s+interamericana)\b")
_RX_CORTE = re.compile(r"\b(corte\s+idh|coidh|corte\s+interamericana|tribunal\s+interamericano"
                       r"|inter\s*-?\s*american\s+court)\b")
_RX_SERIE = re.compile(r"\b(series?)\s+([ac])\s*,?\s*(?:no\s*\.?|n\s*[°º]\.?|num\.?|numero|nro\.?|#)?\s*(\d{1,4})\b")
_RX_OC = re.compile(r"(?<![A-Za-z0-9])(?:OC|oc(?=\s*[-‐–—]\s*\d{1,2}\s*/))\s*[-‐–—]?\s*(\d{1,2})"
                    r"(?:\s*/\s*(\d{2,4}))?(?!\d)")
_RX_OC_TEXTO = re.compile(r"\b(?:opinion(?:es)?\s+consultiva|advisory\s+opinion)\s+(?:oc\s*[-‐–—]?\s*)?"
                          r"(?:no\.?\s*|numero\s*|n[°º]\s*)?(\d{1,2})(?:\s*/\s*(\d{2,4}))?(?!\d)")
_NUMS = r"\d{1,4}(?:\s*(?:-|–|—|a|al|y|e|,|to|and|&)\s*\d{1,4})*"
_RX_MARCA = re.compile(
    r"(?P<kw>p[a]rrs?\s*\.|p[a]rrafos?|p[a]rr\b|¶+|§+|paras?\s*\.|paragraphs?"
    r"|puntos?\s+resolutivos?|resolutivos?|operative\s+paragraphs?"
    r"|considerandos?)\s*(?:no\.?\s*|numeros?\s*)?(?P<nums>" + _NUMS + r")")
_RX_NORMA = re.compile(r"\b(?:articulos?|ley|leyes|codigo|constitucion|constitucional|reglamento|fraccion|inciso"
                       r"|escrito|demanda|contrato|tesis|jurisprudencia|decreto|norma)\b|\barts?\.")
_RX_NO_ES_CASO = re.compile(r"^\s*,?\s*(?:del?\s+(?:la\s+|el\s+)?)?(?:articulo|art\.|arts\.|ley\b|codigo|constitucion"
                            r"|constitucional|reglamento|fraccion|inciso|norma|tesis|jurisprudencia\s+\d)")
# Palabras con mayúscula que pueden rodear el nombre de un caso sin ser parte
# de otro nombre propio.
_NO_NOMBRE = {"caso", "casos", "case", "vs", "v", "versus", "corte", "idh", "coidh", "serie", "series", "sentencia",
              "fondo", "excepciones", "excepcion", "reparaciones", "interpretacion", "voto", "parr", "parrs", "para",
              "la", "el", "los", "las", "de", "del", "y", "e", "en", "the", "et", "al", "of", "en", "cfr", "ver",
              "vease", "asimismo", "tambien", "supra", "infra", "nota", "interamericana", "judgment", "merits",
              "resolutivo", "considerando", "supervision", "mexico", "chile", "peru", "guatemala", "colombia",
              "argentina", "brasil", "uruguay", "ecuador", "honduras", "paraguay", "venezuela", "bolivia", "panama",
              "nicaragua", "salvador", "surinam", "barbados", "costa", "haiti", "republica", "trinidad", "otros",
              "otras", "otro", "otra", "familiares", "en", "sobre", "segun", "desde", "hasta", "como", "que", "por"}
_QUALIF_VOTO = {"razonado", "razonada", "concurrente", "concurrentes", "disidente", "disidentes", "separado",
                "salvado", "conjunto", "parcialmente", "parcial", "individual", "particular", "y", "juridico",
                "concurring", "dissenting", "separate", "reasoned", "partially", "joint", "partly"}
_ANTES_AUTOR = {"de", "del", "la", "los", "las", "of", "juez", "jueza", "jueces", "juezas", "judge", "judges", "ad",
                "hoc", "el", "magistrado", "ministro"}
_SALTOS_VS = {"y", "e", "otros", "otras", "otro", "otra", "et", "al", "and", "others", "sus", "miembros",
              "familiares", "familia", "hijos", "of", "the"}
_PALABRAS_ACTO = [
    ("interpretacion", re.compile(r"\binterpretaci[oó]n\b|\binterpretation\b")),
    ("revision", re.compile(r"\bsolicitud de revisi[oó]n\b|\brevisi[oó]n de la sentencia\b")),
    ("cumplimiento", re.compile(r"\bcumplimiento de sentencia\b")),
    ("competencia", re.compile(r"\bcompetencia\b(?!\s+de\s+(?:los|las)\s+(?:jueces|tribunales))")),
    ("excepciones", re.compile(r"\bexcepci[oó]n(?:es)?\s+preliminar(?:es)?\b|\bpreliminary objections?\b")),
    ("reparaciones", re.compile(r"\breparaciones\b|\breparations\b")),
    ("fondo", re.compile(r"\bfondo\b|\bmerits\b")),
    ("supervision", re.compile(r"\bsupervisi[oó]n\b|\bmonitoring compliance\b")),
    ("medidas", re.compile(r"\bmedidas provisionales\b|\bprovisional measures\b")),
]
_SECUNDARIAS = {"interpretacion", "revision", "cumplimiento", "competencia", "otro"}


def _tokens(plano: str) -> List[Tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group()) for m in re.finditer(r"[a-z0-9]+", plano)]


def _es_mayuscula(original: str, ini: int, fin: int) -> bool:
    """¿Viene escrito como nombre propio? Cada palabra significativa con
    mayúscula inicial. En un tramo escrito TODO EN MAYÚSCULAS (encabezados de
    demandas, rubros) la mayúscula no dice nada: «CASO … ÓRDENES DE PROTECCIÓN»
    no es Órdenes Guerra Vs. Chile."""
    propio = [c for c in original[ini:fin] if c.isalpha()]
    if len(propio) >= 2 and all(c.isupper() for c in propio):
        return False
    ventana = original[max(0, ini - 60):fin + 60]
    letras = [c for c in ventana if c.isalpha()]
    if letras and sum(c.isupper() for c in letras) / len(letras) > 0.6:
        return False
    for m in re.finditer(r"[^\W\d_]+", original[ini:fin]):
        w = m.group()
        if _norm(w) in _ARTICULOS and len(w) <= 3:
            continue
        if _norm(w) in COLA_TOKENS:        # «Hidalgo y otros»: la cola va en minúsculas en el propio listado
            continue
        if not w[0].isupper():
            return False
    return True


def _entre_comillas(original: str, ini: int, fin: int) -> bool:
    antes = original[max(0, ini - 3):ini]
    despues = original[fin:fin + 3]
    return bool(re.search(r"[“\"«']\s*$", antes)) and bool(re.search(r"^\s*[”\"»']", despues))


def _expandir(nums: str) -> List[int]:
    """«216 a 219» → 216…219; «93, 94 y 96» → [93, 94, 96]; «338-341». Los
    años que se cuelan en una lista («párr. 124, 2006») se descartan."""
    out: List[int] = []
    partes = re.findall(r"\d{1,4}|-|–|—|\ba\b|\bal\b|\bto\b", nums)
    i = 0
    while i < len(partes):
        p = partes[i]
        if p.isdigit():
            n = int(p)
            if i + 2 < len(partes) and partes[i + 1] in ("-", "–", "—", "a", "al", "to") and partes[i + 2].isdigit():
                m = int(partes[i + 2])
                if n <= m <= n + 60:
                    out.extend(range(n, m + 1))
                else:
                    out.extend([n, m])
                i += 3
                continue
            if out and 1900 <= n <= 2100:
                i += 1
                continue
            out.append(n)
        i += 1
    return [n for n in out if 0 < n <= 2000]


def _buscar_secuencia(tokens, i, tabla: Dict[Tuple[str, ...], Any], max_len: int):
    """Coincidencia más larga de `tabla` que empieza en el token i."""
    for L in range(min(max_len, len(tokens) - i), 0, -1):
        k = tuple(t[2] for t in tokens[i:i + L])
        if k in tabla:
            return L, tabla[k]
    return 0, None


def _marcas(plegado: str, original: str) -> List[Dict[str, Any]]:
    """Párrafos, resolutivos, considerandos y votos citados en el texto."""
    out = []
    for m in _RX_MARCA.finditer(plegado):
        kw = m.group("kw")
        # «párr» de «párrafo segundo del artículo 1º» no es de un caso
        if _RX_NO_ES_CASO.match(plegado[m.end():m.end() + 40]):
            continue
        if kw.startswith(("punto", "resolutivo", "operative")):
            seg = "resolutivos"
        elif kw.startswith("considerando"):
            seg = "considerandos"
        else:
            seg = None                     # el segmento vigente: sentencia o voto
        nums = _expandir(m.group("nums"))
        if nums:
            out.append(dict(ini=m.start(), fin=m.end(), tipo="num", seg=seg, nums=nums))
    cat = catalogo()
    plano = _plano(plegado)
    toks = _tokens(plano)
    for i, (a, b, t) in enumerate(toks):
        if t not in ("voto", "votos", "opinion", "opinions"):
            continue
        if t.startswith("opinion"):
            if i == 0 or toks[i - 1][2] not in _QUALIF_VOTO:
                continue
        j = i + 1
        while j < len(toks) and toks[j][2] in _QUALIF_VOTO:
            j += 1
        calificado = any(toks[x][2] != "y" for x in range(i + 1, j))
        while j < len(toks) and toks[j][2] in _ANTES_AUTOR:
            j += 1
        autor = None
        fin = toks[j - 1][1] if j - 1 < len(toks) else b
        for salto in range(0, 3):          # hasta dos nombres de pila: «Sergio García Ramírez»
            if j + salto >= len(toks):
                break
            L, slug = _buscar_secuencia(toks, j + salto, cat["autores"], cat["autor_max"])
            if L:
                autor = slug
                fin = toks[j + salto + L - 1][1]
                break
        # «Voto» a secas es derecho electoral: «¿cómo se regula el voto en
        # Hidalgo?» daba Hidalgo y otros Vs. Ecuador (revisión del 25-sep-2026).
        # Sólo es un voto judicial con calificativo (razonado, concurrente,
        # disidente…) o con el nombre de un juez de la Corte.
        if not calificado and not autor:
            continue
        out.append(dict(ini=a, fin=fin, tipo="voto", seg="voto", autor=autor, autor_ini=toks[j][0]
                        if j < len(toks) else fin))
    return sorted(out, key=lambda x: x["ini"])


def _serie_de(serie: str, crudo: str, casos: Set[str]) -> Optional[Dict[str, Any]]:
    """El documento «Serie X No. crudo» si pertenece a uno de `casos`; prueba
    también los prefijos («1541» = 154 + nota 1)."""
    docs = catalogo()["docs"]
    for L in range(len(crudo), 0, -1):
        d = docs.get(f"{serie}-{int(crudo[:L])}")
        if d and d.get("caso_id") in casos:
            return d
    return None


_RX_EUROPA = re.compile(r"\b(tedh|tribunal europeo|corte europea|european court|eur\.? ?ct|echr|cedh"
                        r"|c\. (?:espana|francia|reino unido|italia|alemania|austria|belgica|turquia|suiza|grecia)"
                        r"|v\. (?:the )?(?:united kingdom|france|spain|italy|germany|austria|belgium|turkey))\b")


def _serie_a_valida(plegado: str, ini: int) -> bool:
    """«Serie A No. 18» es la OC-18… o el caso Golder del Tribunal Europeo
    (Series A no. 18). Las sentencias interamericanas citan mucho al TEDH con
    su Serie A: sin «Opinión Consultiva», «OC-» o «Corte IDH» cerca, no cuenta,
    y con marca europea cerca, tampoco."""
    antes = plegado[max(0, ini - 200):ini]
    if _RX_EUROPA.search(antes[-120:]):
        return False
    return bool(re.search(r"opinion(?:es)? consultiva|advisory opinion|\boc\s*-?\s*\d|corte idh|corte interamericana"
                          r"|inter-american court", antes))


_RX_SIGUE_CITA = re.compile(r"(?:serie|series|sentencia|fondo|excepci|reparaci|interpretaci|resoluci|supervisi|merits"
                            r"|judgment|voto|p[aá]rr|para|considerando|resolutivo|punto|opini[oó]n|cfr|nota|supra"
                            r"|vs|no\b)", re.I)


def _corta_la_cita(gap_original: str, gap_plegado: str) -> bool:
    """¿Entre la última marca de una cita y la siguiente empieza otra cosa?
    Punto y aparte seguido de algo que no es la continuación de la cita,
    punto y coma seguido de otra cosa que no sea «párr.»/«resolutivo»/«voto»,
    una cita europea o la palabra «caso» (otro caso, aunque no esté en el
    catálogo: «Véase también Caso Desmond McKenzie»)."""
    if "\n\n" in gap_original or _RX_EUROPA.search(gap_plegado):
        return True
    if re.search(r"\b(?:caso|case|casos|cases)\b", gap_plegado):
        return True
    for m in re.finditer(r"([.;])\s+(\S+)", gap_original):
        sig = m.group(2)
        if _RX_SIGUE_CITA.match(sig) or sig[:1].isdigit():
            continue
        if m.group(1) == ";" or sig[:1].isupper():
            return True
    return False


def _contexto_idh(plegado: str) -> bool:
    return bool(_RX_CONTEXTO_IDH.search(plegado))


def _menciones_de_casos(original: str, plegado: str, plano: str, mascaras: List[Tuple[int, int]],
                        marcas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cat = catalogo()
    toks = _tokens(plano)
    idh = _contexto_idh(plegado)
    hits = []
    i = 0
    while i < len(toks):
        L, r = _buscar_secuencia(toks, i, cat["alias"], cat["max_len"])
        if not L:
            i += 1
            continue
        ini, fin = toks[i][0], toks[i + L - 1][1]
        if any(a <= ini < b or a < fin <= b for a, b in mascaras):
            i += L
            continue
        # «Ruiz-Mateos» no es «Ruiz»: un nombre compuesto con guion que el
        # alias no cubre entero es otro nombre.
        if re.match(r"\s?[-‐]\s?[^\W\d_]", original[fin:fin + 4]) or \
                re.search(r"[^\W\d_]\s?[-‐]\s?$", original[max(0, ini - 4):ini]):
            i += L
            continue
        hits.append(dict(i=i, j=i + L, ini=ini, fin=fin, casos=set(r["casos"]), clase=r["clase"],
                         alias=" ".join(t[2] for t in toks[i:i + L])))
        i += L

    # Alias pegados («González y otras (“Campo Algodonero”)», «Street Children
    # (Villagrán-Morales et al.)») son UNA mención: se intersectan.
    grupos: List[Dict[str, Any]] = []
    for h in hits:
        g = grupos[-1] if grupos else None
        entre = [toks[k][2] for k in range(g["j"], h["i"])] if g else []
        pegados = g and not re.search(r"[.;]\s", original[g["fin"]:h["ini"]])
        if pegados and ((
                h["i"] - g["j"] <= 3 and all(t in _SALTOS_VS for t in entre)
                and (re.search(r"[(“\"«']", original[g["fin"]:h["ini"]]) or (entre and entre[-1] in COLA_TOKENS)))
                or (h["i"] - g["j"] <= 4 and g["casos"] & h["casos"])):
            # Se funde si lo segundo va entre paréntesis o comillas, o tras
            # «y otros», o si los dos trozos son del MISMO caso («Comunidad
            # Mayagna (Sumo) Awas Tingni»). «Fernández Ortega y Rosendo Cantú»
            # son dos casos y quedan separados.
            inter = g["casos"] & h["casos"]
            g["casos"] = inter or (g["casos"] if _RIESGO[g["clase"]] >= _RIESGO[h["clase"]] else h["casos"])
            if _RIESGO[h["clase"]] > _RIESGO[g["clase"]]:
                g["clase"], g["alias"] = h["clase"], h["alias"]
            g["j"], g["fin"] = h["j"], h["fin"]
            g["partes"].append(h)
            continue
        grupos.append(dict(h, partes=[h]))

    menciones = []
    for n, g in enumerate(grupos):
        # ── pistas
        pistas = set()
        antes_toks = [t[2] for t in toks[max(0, g["i"] - 4):g["i"]]]
        prev = list(antes_toks)
        while prev and prev[-1] in ("de", "del", "la", "los", "las", "el", "the", "of"):
            prev.pop()
        # «en caso de (que)…» es una locución condicional, no una cita
        condicional = len(prev) >= 2 and prev[-2] == "en" and len(prev) < len(antes_toks) \
            and antes_toks[len(prev)] in ("de", "del")
        if prev and prev[-1] in ("caso", "casos", "case", "cases") and not condicional:
            pistas.add("caso")
            # «(en) el caso de Hidalgo», «el caso de canales de riego»: con
            # «de» en medio es la locución de todos los días, no la cita
            # («Caso Hidalgo y otros Vs. Ecuador»). Revisión del 25-sep-2026.
            if len(prev) < len(antes_toks) and antes_toks[len(prev)] in ("de", "del"):
                pistas.add("caso_de")
        elif n > 0 and "caso" in grupos[n - 1].get("pistas", set()) and prev and prev[-1] in ("y", "e", "and") \
                and g["i"] - grupos[n - 1]["j"] <= 2:
            pistas.add("caso")         # «casos Fernández Ortega y Rosendo Cantú»
        k = g["j"]
        while k < len(toks) and toks[k][2] in _SALTOS_VS and k - g["j"] < 5:
            k += 1
        estado = None
        if k < len(toks) and toks[k][2] in ("vs", "v", "versus"):
            if toks[k][2] != "v" or plegado[toks[k][1]:toks[k][1] + 1] == ".":
                L, est = _buscar_secuencia(toks, k + 1, cat["estados"], cat["estado_max"])
                if L:
                    pistas.add("vs")
                    estado = est
                    g["fin_vs"] = toks[k + L][1]
        if _RX_CORTE.search(plegado[max(0, g["ini"] - 150):g["fin"] + 150]):
            pistas.add("corte")
        tramo = plegado[g["fin"]:g["fin"] + 250]
        # «Serie C No. N» después del nombre es pista SÓLO si el número (o su
        # prefijo, por la llamada a nota pegada) es de uno de estos casos: un
        # «HERNANDEZ» en mayúsculas seguido de la cita de OTRO caso no cuenta.
        ms = _RX_SERIE.search(tramo)
        if ms and _serie_de(ms.group(2).upper(), ms.group(3), g["casos"]):
            pistas.add("serie")
        # Cita de párrafo pegada: «Tzompaxtle, resolutivo 8», «Almonacid ¶124»,
        # «Gelman, supervisión de 20 de marzo de 2013, considerando 67». Entre
        # el nombre y la marca sólo cabe el resto de la cita (≤100 caracteres,
        # sin fin de pregunta ni salto de línea).
        desde = g.get("fin_vs", g["fin"])
        mm = re.match(r"^([^?!\n;]{0,100}?)(p[a]rrs?\.|p[a]rrafos?|¶|§|paras?\.|resolutivos?|punto\s+resolutivo"
                      r"|considerandos?)\s*\d", plegado[desde:desde + 130])
        # «el Código Civil de Hidalgo, párrafo 3 del artículo 20»: el párrafo
        # es de un artículo, no de un caso (la misma regla que en `_marcas`).
        tras_num = desde + (mm.end() - 1 if mm else 0)
        mnum = re.match(r"\d[\d\s,ay\-–]*", plegado[tras_num:tras_num + 60]) if mm else None
        if mm and mnum and _RX_NO_ES_CASO.match(plegado[tras_num + mnum.end():tras_num + mnum.end() + 40]):
            mm = None
        if mm and not _RX_EUROPA.search(mm.group(1)) and not re.search(r"\b(?:caso|case)\b|;", mm.group(1)):
            abreviada = not mm.group(2).startswith(("parrafo", "resolutivo", "punto"))
            corta = len(mm.group(1)) <= 25
            if corta:
                pistas.add("adyacente_corta")
            if abreviada or corta:
                pistas.add("adyacente")
            if abreviada or mm.group(2).startswith("considerando"):
                pistas.add("adyacente_fuerte")
        antes = plegado[max(0, g["ini"] - 40):g["ini"]]
        if re.search(r"(?:p[a]rrs?\.|p[a]rrafos?|¶|§)\s*\d[\d\s,ay\-–]*\s*(?:de|del|en)\s+(?:la\s+sentencia\s+(?:de|del)\s+)?"
                     r"(?:(?:el\s+)?caso\s+)?$", antes):
            pistas.add("adyacente")
            pistas.add("adyacente_antes")
        # «Ley de Hacienda de Hidalgo, párrafo 4», «el Reglamento de Salamanca,
        # párr. 3»: el alias es complemento de otra cosa («… de Hidalgo»). Ahí
        # el párrafo pegado no es pista (revisión del 25-sep-2026).
        # «la sentencia de Almonacid, párr. 124» sí es cita.
        complemento = bool(antes_toks) and antes_toks[-1] in ("de", "del") and "adyacente_antes" not in pistas \
            and "caso" not in pistas and not (len(antes_toks) >= 2 and antes_toks[-2] in (
                "sentencia", "sentencias", "fallo", "resolucion", "asunto", "caso", "casos"))
        for mv in marcas:
            if mv["tipo"] == "voto" and mv["fin"] <= g["ini"] and g["ini"] - mv["fin"] <= 25 \
                    and re.fullmatch(r"[\s,]*(?:en|in)\s+(?:el\s+|the\s+)?(?:caso\s+|case\s+(?:of\s+)?)?(?:de\s+la\s+|del?\s+)?",
                                     plegado[mv["fin"]:g["ini"]]):
                pistas.add("voto")
            # …o detrás: «Trabajadores Cesados, voto razonado de García Ramírez»
            if mv["tipo"] == "voto" and 0 <= mv["ini"] - g["fin"] <= 30 \
                    and re.fullmatch(r"[\s,.;:()]*(?:el\s+|y\s+|en\s+(?:el|su)\s+)?", plegado[g["fin"]:mv["ini"]]):
                pistas.add("voto")
        if _entre_comillas(original, g["ini"], g["fin"]):
            pistas.add("comillas")
        # ¿El alias es un pedazo de otro nombre propio? «Viviana Gallardo» no
        # es Rondón Gallardo; «JUANA ANGEL HERNANDEZ CORZO» no es Hernández.
        vecino_propio = False
        for w in (re.search(r"([^\W\d_]+)[\s]*$", original[max(0, g["ini"] - 30):g["ini"]]),
                  re.match(r"^[\s]*([^\W\d_]+)", original[g["fin"]:g["fin"] + 30])):
            if w and w.group(1)[:1].isupper() and _norm(w.group(1)) not in _NO_NOMBRE:
                vecino_propio = True
        mayus = _es_mayuscula(original, g["ini"], g["fin"])
        g["pistas"] = pistas

        # ── ¿cuenta? Las reglas de «UN APELLIDO SUELTO NO ES UN CASO», en código
        clase = g["clase"]
        fuertes = pistas & {"vs", "serie"}
        cuenta = False
        via = None
        toks_alias = g["alias"].split()
        frecuente = all(t in APELLIDOS_FRECUENTES or t in _ARTICULOS or t in COLA_TOKENS for t in toks_alias)
        de_pila = clase == "palabra" and any(t in NOMBRES_PILA for t in toks_alias)
        corriente = clase == "palabra" and any(t in PALABRAS_COMUNES for t in toks_alias)
        # El párrafo pegado a «… de Hidalgo» no cuenta como pista (ver arriba).
        if complemento:
            pistas -= {"adyacente", "adyacente_corta", "adyacente_fuerte"}
        sig_alias = [t for t in toks_alias if t not in _ARTICULOS and t not in COLA_TOKENS]
        if clase == "palabra" and "caso_de" in pistas and not fuertes:
            # «en el caso de Hidalgo», «el caso de Ochoa en el juzgado»,
            # «en el caso de canales de riego»: una palabra tras «caso de» sólo
            # cuenta escrita con mayúscula y con la Corte IDH en el texto.
            # «Caso Hidalgo y otros Vs. Ecuador» (sin «de») no pasa por aquí.
            cuenta = mayus and (idh or bool(pistas & {"corte", "voto"}))
        elif clase == "descriptivo" and len(sig_alias) <= 2:
            # Dos palabras que arrancan con un descriptor son o un prefijo
            # («Comunidad Campesina», «Hermanos Gómez», «Defensor de
            # Derechos») o una pareja de apellidos («Flor Freire», «Salvador
            # Chiriboga»): con mayúscula sola no bastan. «La Comunidad
            # Campesina de San Pedro promovió amparo agrario» y «la empresa
            # Hermanos Gómez S.A.» se disparaban (revisión del 25-sep-2026).
            cuenta = bool(pistas & {"comillas", "vs", "serie", "caso", "voto"}) or (
                mayus and bool(pistas & {"corte", "adyacente"}))
        elif clase == "comun":
            cuenta = bool(fuertes) or (mayus and bool(pistas & {"caso", "voto", "adyacente_fuerte"}))
        elif clase == "iniciales":
            cuenta = bool(fuertes) or ("caso" in pistas and original[g["ini"]:g["fin"]].isupper())
        elif clase in ("palabra", "apellidos"):
            if vecino_propio and "voto" not in pistas and "serie" not in pistas and not (estado and "vs" in pistas):
                cuenta = False                 # «caso Maldonado Vargas»: el alias es sólo un pedazo
            elif fuertes:
                cuenta = True
            elif "caso" in pistas:
                if frecuente or de_pila:
                    cuenta = mayus and (idh or bool(pistas & {"corte", "adyacente_fuerte", "voto"}))
                elif corriente:
                    cuenta = mayus
                else:
                    cuenta = True
            elif vecino_propio and not pistas & {"voto"}:
                cuenta = False
            elif frecuente or de_pila:
                # «Ruiz- Mateos c. España, …, párr. 30», «JUANA ANGEL HERNANDEZ
                # CORZO … Corte IDH»: un apellido frecuente suelto sólo cuenta
                # con la cita pegada y abreviada, o dentro de un voto.
                cuenta = mayus and ("voto" in pistas or {"adyacente_corta", "adyacente_fuerte"} <= pistas)
            elif pistas & {"corte", "adyacente", "voto"}:
                cuenta = mayus
        else:                                  # apodo, descriptivo, nombre largo
            # «sentencia de la Corte Interamericana…: campo algodonero» (así,
            # en minúsculas, en una pregunta real del 25-sep-2026) sí cuenta.
            cuenta = mayus or bool(pistas & {"comillas", "vs", "serie", "caso", "corte", "adyacente", "voto"})
        if not cuenta:
            continue
        for p in ("serie", "vs", "caso", "voto", "adyacente", "corte", "comillas"):
            if p in pistas:
                via = p
                break
        casos = set(g["casos"])
        if estado:
            # «X Vs. Chile» con un X que no tiene caso contra Chile no es X:
            # es el pedazo de otro nombre («… Maldonado Vargas Vs. Chile» no
            # es Vargas Areco Vs. Paraguay).
            casos = {c for c in casos if cat["casos"][c]["estado"] == estado}
            if not casos:
                continue
        ini = g["ini"]
        # que la evidencia arranque en «Caso» si lo hay
        mcaso = re.search(r"(?:casos?|case of|case)\s+(?:(?:de\s+)?(?:la|los|las|el|del|the)\s+)?[\"“«']?\s*$",
                          plegado[max(0, ini - 25):ini])
        if mcaso:
            ini = max(0, ini - 25) + mcaso.start()
        menciones.append(dict(ini=ini, fin=g.get("fin_vs", g["fin"]), alias_fin=g["fin"], alias_ini=g["ini"],
                              casos=casos, clase=clase, via=via or "sin_pista", alias=g["alias"],
                              estado=estado, partes=g["partes"]))
    return menciones


def _anio_en(plegado: str) -> Optional[int]:
    m = re.search(r"(?<!\d)(19[7-9]\d|20[0-4]\d)(?!\d)", plegado)
    return int(m.group(1)) if m else None


def _fecha_en(plegado: str) -> Optional[str]:
    m = re.search(r"(\d{1,2})\s*[º°]?\s+(?:de\s+)?([a-z]+)\s+(?:de\s+|del\s+)?(\d{4})", plegado)
    if m and (MESES.get(m.group(2)) or MESES_EN.get(m.group(2))):
        mes = MESES.get(m.group(2)) or MESES_EN.get(m.group(2))
        return f"{m.group(3)}-{mes:02d}-{int(m.group(1)):02d}"
    m = re.search(r"([a-z]+)\s+(\d{1,2}),\s+(\d{4})", plegado)      # «November 24, 2006»
    if m and MESES_EN.get(m.group(1)):
        return f"{m.group(3)}-{MESES_EN[m.group(1)]:02d}-{int(m.group(2)):02d}"
    return None


def _filtrar(docs: List[Dict[str, Any]], pred) -> List[Dict[str, Any]]:
    f = [d for d in docs if pred(d)]
    return f or docs


def _elegir_resolucion(caso_ids: Set[str], tramo: str, parrafos: List[int], seg: str, autor: Optional[str],
                       alias: str) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], str, List[str]]:
    """Del (o los) caso(s) a UN documento, o a candidatos. Devuelve
    (documento, candidatos, confianza, notas)."""
    cat = catalogo()
    notas: List[str] = []
    palabras = {k for k, rx in _PALABRAS_ACTO if rx.search(tramo)}
    anio = _anio_en(tramo)
    fecha = _fecha_en(tramo)

    if seg == "considerandos" or palabras & {"supervision", "medidas"}:
        claves = ("supervisiones",) if "supervision" in palabras or "medidas" not in palabras else ("medidas",)
        if "medidas" in palabras and "supervision" not in palabras:
            claves = ("medidas",)
        docs = [cat["docs"][d] for c in caso_ids for k in claves for d in cat["casos"][c][k]]
        if not docs and seg == "considerandos":
            docs = [cat["docs"][d] for c in caso_ids for d in cat["casos"][c]["resoluciones"]]
    else:
        docs = [cat["docs"][d] for c in caso_ids for d in cat["casos"][c]["resoluciones"]]
    if not docs:
        return None, [], "ambigua", ["el caso no tiene documentos de ese tipo en el catálogo"]

    # Acumulados (Hilaire / Constantine / Benjamin): la resolución cuyo propio
    # nombre lleva el alias.
    propios = [d for d in docs if alias and alias in _norm(d["nombre"])]
    if propios and len(propios) < len(docs):
        docs = propios
    if fecha:
        docs = _filtrar(docs, lambda d: d.get("fecha") == fecha)
    if anio:
        docs = _filtrar(docs, lambda d: (d.get("fecha") or "").startswith(str(anio)))
    if autor:
        con = [d for d in docs if any(autor in v["slug"].split("+") for v in d["votos"])]
        if con:
            docs = con
        else:
            notas.append(f"el listado no registra un voto de {autor} en estas resoluciones")
    if "interpretacion" in palabras:
        docs = _filtrar(docs, lambda d: d["tipo"] == "interpretacion")
    elif docs and docs[0]["clase"] == "CC":
        sust = [d for d in docs if d["tipo"] not in _SECUNDARIAS]
        if sust and len(sust) < len(docs):
            notas.append("se descartan interpretaciones/competencia/cumplimiento: la cita no las nombra")
            docs = sust
        for p in ("excepciones", "fondo", "reparaciones"):
            if p in palabras:
                docs = _filtrar(docs, lambda d: p in d["componentes"])
        # «Fondo» a secas contra «Fondo, Reparaciones y Costas»: si la cita
        # nombra sólo el fondo, gana la que no trae reparaciones.
        if "fondo" in palabras and "reparaciones" not in palabras and len(docs) > 1:
            docs = _filtrar(docs, lambda d: "reparaciones" not in d["componentes"])
    if parrafos:
        docs = _filtrar(docs, lambda d: not d.get("n_parrafos") or max(parrafos) <= d["n_parrafos"])

    casos_restantes = {d.get("caso_id") for d in docs}
    if len(docs) == 1:
        conf = "alta" if len(caso_ids) == 1 else "media"
        return docs[0], [], conf, notas
    if len(casos_restantes) > 1:
        # Varios CASOS: nunca se elige. Un candidato por caso (su principal).
        cands = []
        for c in sorted(casos_restantes, key=lambda x: cat["casos"][x]["nombre"]):
            ds = [d for d in docs if d.get("caso_id") == c]
            pr = next((d for d in ds if d.get("acto_principal")), ds[0])
            cands.append(pr)
        return None, cands, "ambigua", notas + ["el alias apunta a varios casos"]
    # Un caso, varias resoluciones
    if not parrafos and seg == "sentencia":
        pr = next((d for d in docs if d.get("acto_principal")), None)
        if pr is None:
            pr = next((d for d in docs if "fondo" in d["componentes"]), docs[0])
        return pr, [], "media", notas + ["caso sin párrafo: se toma la resolución principal"]
    return None, docs, "ambigua", notas + ["el párrafo puede estar en varias resoluciones del caso"]


def _cita_canonica(d: Dict[str, Any], seg: str, nums: List[int], autor: Optional[str]) -> str:
    base = d["cita"].rstrip(". ")
    if seg == "voto" and autor:
        nombre = catalogo()["data"]["autores"].get(autor, {}).get("nombre", autor)
        base += f". Voto del juez {nombre}"
    if not nums:
        return base + "."
    if seg == "resolutivos":
        et = "punto resolutivo" if len(nums) == 1 else "puntos resolutivos"
    elif seg == "considerandos":
        et = "considerando" if len(nums) == 1 else "considerandos"
    else:
        et = "párr." if len(nums) == 1 else "párrs."
    return f"{base}, {et} {_rango_texto(nums)}."


def _rango_texto(nums: List[int]) -> str:
    nums = sorted(set(nums))
    tramos, a = [], nums[0]
    prev = a
    for n in nums[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        tramos.append(str(a) if a == prev else f"{a} a {prev}")
        if n is not None:
            a = prev = n
    return ", ".join(tramos)


def _llave(d: Dict[str, Any], seg: str, autor: Optional[str], n: int) -> str:
    s = {"sentencia": "s", "resolutivos": "r", "considerandos": "c"}.get(seg)
    if seg == "voto":
        s = f"v:{autor or '?'}"
    return f"{d['doc_id']}|{s}|{n}"


def _resumen_doc(d: Dict[str, Any]) -> Dict[str, Any]:
    return dict(doc_id=d["doc_id"], caso=_nombre_caso(d), caso_id=d.get("caso_id"), serie=d["serie"],
                serie_num=d.get("serie_num"), fecha=d.get("fecha"), tipo_resolucion=d.get("acto"),
                url_oficial=d.get("url_oficial"))


def _nombre_caso(d: Dict[str, Any]) -> str:
    if d["clase"] == "OC":
        return f"{d['oc']} · {d['nombre']}" if d.get("oc") else d["nombre"]
    return f"{d['nombre']} Vs. {d['estado']}" if d.get("estado") else d["nombre"]


def resolver_citas_coidh(texto: str, max_resultados: int = 20, previo: Optional[str] = None) -> List[Dict[str, Any]]:
    """Casos y opiniones consultivas de la Corte IDH que nombra un texto.

    `previo` (opcional) es la llave o el doc_id de la última cita de la
    conversación («C-154|s|124»). Si el texto es una pregunta de seguimiento
    que sólo trae «párr. N» («¿y el párrafo 125?»), el párrafo se hereda de
    ese documento (revisión B.7): la reescritura del hilo está pensada para
    leyes y tesis y puede perder el caso.

    Cada resultado:
      doc_id        «C-154», «A-24», «SS-gelman-uruguay-2013-03-20»; None si ambiguo
      caso, caso_id, serie, serie_num, fecha, tipo_resolucion
      parrafo       el primero pedido (int) o None; `parrafos` la lista completa
      seg           'sentencia' | 'voto' | 'resolutivos' | 'considerandos'
      voto_autor    slug del juez («garcia-ramirez») si la cita es de un voto
      url_oficial   la del listado de corteidh.or.cr, sin #page
      confianza     'alta' | 'media' | 'ambigua'
      candidatos    [{doc_id, caso, …}] cuando no se puede elegir (doc_id=None)
      evidencia     el fragmento del texto que lo disparó
      via           qué pista lo hizo contar: serie, oc, vs, caso, voto, adyacente, corte, comillas, sin_pista
      alias, alias_clase  el nombre que casó y su clase (palabra, apellidos, apodo, descriptivo…)
      llaves        «C-154|s|124» — la llave de la colección `coidh` (plan 3.4)
      cita_canonica la cita oficial del listado con el párrafo
      notas         por qué eligió lo que eligió (o por qué no pudo)
    """
    if not texto or not texto.strip():
        return []
    cat = catalogo()
    original = texto
    plegado = _plegar(texto)
    plano = _plano(plegado)
    marcas = _marcas(plegado, original)

    # Lo que tapa alias: el nombre del juez de un voto («García Ramírez» no es
    # el caso García Rodríguez) y los propios números de serie / OC.
    mascaras = [(m["autor_ini"], m["fin"]) for m in marcas if m["tipo"] == "voto" and m.get("autor")]

    anclas: List[Dict[str, Any]] = []
    # ── llave directa: «Serie C No. 154», «Series C No. 154», «Serie A No. 24»
    for m in _RX_SERIE.finditer(plegado):
        serie = m.group(2).upper()
        n = int(m.group(3))
        if serie == "A" and not _serie_a_valida(plegado, m.start()):
            continue
        # «acciones Serie C 500», «billete de la serie C 154»: sin «No.» y sin
        # nada interamericano cerca, el número sólo sirve para confirmar un
        # nombre de caso pegado, nunca como llave suelta (revisión 25-sep-2026).
        con_no = bool(re.search(r"(?:\bno\s*\.?|\bn\s*[°º]|\bnum|\bnumero|\bnro|#)\s*\d+$", m.group(0)))
        debil = not con_no and not _contexto_idh(plegado[max(0, m.start() - 200):m.end() + 200])
        anclas.append(dict(ini=m.start(), fin=m.end(), directo=f"{serie}-{n}", num_crudo=m.group(3), serie=serie,
                           via="serie", debil=debil))
    # ── opiniones consultivas
    vistos_oc = set()
    for rx in (_RX_OC, _RX_OC_TEXTO):
        fuente = original if rx is _RX_OC else plegado
        for m in rx.finditer(fuente):
            if any(a <= m.start() < b for a, b in vistos_oc):
                continue
            n = int(m.group(1))
            doc = cat["oc_por_num"].get(n)
            if not doc:
                continue
            aa = m.group(2)
            # «la orden de compra OC-15», «según la OC 12 del contrato»: en el
            # derecho mercantil mexicano «OC» es la orden de compra. Sin el año
            # («OC-24/17») sólo vale con la Corte IDH o la opinión consultiva
            # cerca (revisión del 25-sep-2026).
            if rx is _RX_OC and not aa and not (
                    _contexto_idh(plegado[max(0, m.start() - 200):m.end() + 200])
                    or re.search(r"opinion(?:es)? consultiva|advisory opinion", plegado[max(0, m.start() - 200):m.end() + 200])):
                continue
            vistos_oc.add((m.start() - 25, m.end() + 5))
            anclas.append(dict(ini=m.start(), fin=m.end(), directo=doc, oc_anio=aa, via="oc"))
    for a in anclas:
        mascaras.append((a["ini"], a["fin"]))

    menciones = _menciones_de_casos(original, plegado, plano, mascaras, marcas)
    # Una llave directa pegada a una mención (misma cita) se funde con ella.
    usadas = set()
    for mn in menciones:
        for k, a in enumerate(anclas):
            if k in usadas or a.get("via") != "serie":
                continue
            if 0 <= a["ini"] - mn["fin"] <= 250 and not any(
                    o is not mn and mn["fin"] <= o["ini"] < a["ini"] for o in menciones):
                if _serie_de(a["serie"], a["num_crudo"], mn["casos"]):
                    mn["directo"] = a
                    mn["fin_cita"] = a["fin"]
                    usadas.add(k)
                    break
                # Misma cita, número de OTRO caso: «Caso Furlan y familiares Vs.
                # Argentina. … Serie C No. 212, párr. 133» (errata del
                # Cuadernillo 5; Furlan es la 246 y la 212 es Chitay Nech).
                # Antes salían dos resultados y el párr. 133 se le pegaba a
                # Chitay Nech con confianza «alta». Ahora es un conflicto: no
                # se elige (revisión del 25-sep-2026).
                if not _corta_la_cita(original[mn["fin"]:a["ini"]], plegado[mn["fin"]:a["ini"]]):
                    mn["conflicto_serie"] = a
                    mn["fin_cita"] = a["fin"]
                    usadas.add(k)
                    break
    for k, a in enumerate(anclas):
        if k in usadas or a.get("debil"):
            continue
        menciones.append(dict(ini=a["ini"], fin=a["fin"], casos=set(), clase="directo", via=a["via"],
                              alias="", estado=None, directo=a, alias_ini=a["ini"], alias_fin=a["fin"]))
    menciones.sort(key=lambda x: x["ini"])
    # OC dicha dos veces («opinión consultiva OC-24/17»): una sola mención
    dedup = []
    for mn in menciones:
        if dedup and mn.get("directo") and dedup[-1].get("directo") and \
                mn["directo"].get("directo") == dedup[-1]["directo"].get("directo") and mn["ini"] - dedup[-1]["fin"] < 30:
            dedup[-1]["fin"] = max(dedup[-1]["fin"], mn["fin"])
            continue
        dedup.append(mn)
    menciones = dedup

    # ── seguimiento: sin caso nombrado, texto corto y con párrafo → el previo.
    # Si la pregunta habla de una norma o de un escrito («¿y qué dice el
    # artículo 14, párrafo 2?», «dame el párrafo 5 del escrito»), el párrafo
    # es de eso y no del caso anterior (revisión del 25-sep-2026).
    if not menciones and previo and len(texto) <= 400 and not _RX_NORMA.search(plegado):
        d_prev = cat["docs"].get(previo.split("|")[0])
        nums_marcas = [m for m in marcas if m["tipo"] in ("num", "voto")]
        if d_prev and nums_marcas:
            seg_prev = previo.split("|")[1] if previo.count("|") >= 2 else "s"
            menciones.append(dict(ini=nums_marcas[0]["ini"], fin=nums_marcas[0]["ini"], casos={d_prev.get("caso_id")},
                                  clase="heredado", via="heredado", alias="", estado=None,
                                  directo=dict(directo=d_prev["doc_id"], via="heredado"),
                                  alias_ini=0, alias_fin=0, seg_previo=seg_prev))

    # ── marcas → menciones. Una marca es de la mención anterior si cae en su
    # cita: a ≤150 caracteres de lo último que se le asignó y sin que en medio
    # empiece otra cita. En las notas al pie de verdad (medido en un escrito
    # real del 25-sep-2026) «Caso Genie Lacayo, … párr. 77, donde se cita a la
    # Corte Europea …, Ruiz-Mateos c. España, … párr. 30» — el 30 no es de
    # Genie Lacayo.
    for mn in menciones:
        mn["marcas"] = []
    for mk in marcas:
        dueño = None
        if menciones and menciones[0].get("via") == "heredado":
            menciones[0]["marcas"].append(mk)
            continue
        previas = [mn for mn in menciones if mn["ini"] <= mk["ini"]]
        if previas:
            mn = previas[-1]
            ult = max([mn.get("fin_cita", mn["fin"])] + [x["fin"] for x in mn["marcas"]])
            siguiente = [o for o in menciones if o["ini"] > mn["ini"]]
            if mk["tipo"] == "voto" and mn["ini"] <= mk["ini"] <= mn["fin"]:
                dueño = mn
            elif not mn.get("cerrada") and mk["ini"] - ult <= 150 and (not siguiente or mk["ini"] < siguiente[0]["ini"]):
                if _corta_la_cita(original[ult:mk["ini"]], plegado[ult:mk["ini"]]):
                    mn["cerrada"] = True
                else:
                    dueño = mn
        if dueño is None:
            # «párr. 124 de Almonacid», «voto de García Ramírez en Cesados»
            posteriores = [mn for mn in menciones if mn["ini"] >= mk["fin"]]
            if posteriores and posteriores[0]["ini"] - mk["fin"] <= 60 and re.fullmatch(
                    r"[\s,]*(?:(?:de|del|en|in|of)\s+(?:la\s+sentencia\s+(?:de|del)\s+|the\s+|el\s+)?"
                    r"(?:(?:caso|case)\s+(?:of\s+)?)?(?:de\s+la\s+|del?\s+|de\s+los\s+)?)?[\"“«']?\s*",
                    plegado[mk["fin"]:posteriores[0]["ini"]]):
                dueño = posteriores[0]
        if dueño is not None:
            dueño["marcas"].append(mk)

    resultados: List[Dict[str, Any]] = []
    for idx, mn in enumerate(menciones):
        siguiente = menciones[idx + 1]["ini"] if idx + 1 < len(menciones) else len(plegado)
        ult = max([mn.get("fin_cita", mn["fin"])] + [x["fin"] for x in mn["marcas"]])
        fin_tramo = min(siguiente, ult + 200)
        tramo = plegado[mn["ini"]:fin_tramo]
        # grupos (seg, autor) → números, en el orden en que aparecen
        grupos: List[Tuple[str, Optional[str], List[int]]] = []
        seg_actual, autor_actual = "sentencia", None
        sp = mn.get("seg_previo")
        if sp and sp.startswith("v:"):
            seg_actual, autor_actual = "voto", sp[2:]      # «¿y el párrafo 13?» después de citar un voto
        for mk in sorted(mn["marcas"], key=lambda x: x["ini"]):
            if mk["tipo"] == "voto":
                seg_actual, autor_actual = "voto", mk.get("autor")
                grupos.append(("voto", autor_actual, []))
                continue
            seg = mk["seg"] or seg_actual
            autor = autor_actual if seg == "voto" else None
            if grupos and grupos[-1][0] == seg and grupos[-1][1] == autor:
                grupos[-1][2].extend(mk["nums"])
            else:
                grupos.append((seg, autor, list(mk["nums"])))
        # un voto sin números seguido de números del mismo voto ya quedó fundido;
        # quita grupos de voto vacíos si hay otro del mismo autor con números
        grupos = [g for k, g in enumerate(grupos)
                  if g[2] or g[0] != "voto" or not any(h[0] == "voto" and h[1] == g[1] and h[2] for h in grupos)]
        if not grupos:
            grupos = [("sentencia", None, [])]
        evid_fin = max([mn.get("fin_cita", mn["fin"])] + [x["fin"] for x in mn["marcas"]])
        evid_ini = min([mn["ini"]] + [x["ini"] for x in mn["marcas"]])
        evidencia = re.sub(r"\s+", " ", original[evid_ini:evid_fin]).strip()[:200]

        for seg, autor, nums in grupos:
            nums = sorted(dict.fromkeys(nums))[:80]
            notas: List[str] = []
            d = None
            cands: List[Dict[str, Any]] = []
            conf = "alta"
            directo = mn.get("directo")
            if directo:
                did = directo["directo"]
                d = cat["docs"].get(did)
                if directo.get("via") == "heredado":
                    conf = "media"
                    notas.append(f"caso heredado de la cita anterior ({previo})")
                if directo.get("via") == "oc" and d:
                    aa = directo.get("oc_anio")
                    if aa and d.get("oc_anio") and int(aa) % 100 != d["oc_anio"] % 100:
                        conf = "media"
                        notas.append(f"el año escrito (/{aa}) no es el de {d['oc']}")
                if directo.get("via") == "serie":
                    nombre_casos = mn.get("casos") or set()
                    if d is None or (nombre_casos and d.get("caso_id") not in nombre_casos):
                        # «Serie C No. 1541»: la llamada a nota pegada al número
                        # (así vienen los cuadernillos). Se prueba el prefijo que
                        # case con el nombre citado.
                        crudo = directo["num_crudo"]
                        alt = None
                        for L in range(len(crudo) - 1, 0, -1):
                            x = cat["docs"].get(f"{directo['serie']}-{int(crudo[:L])}")
                            if x and nombre_casos and x.get("caso_id") in nombre_casos:
                                alt = x
                                break
                        if alt is not None:
                            notas.append(f"«Serie {directo['serie']} No. {crudo}» leído como {alt['doc_id']}: "
                                         "número con llamada a nota pegada")
                            d = alt
                        elif d is not None and nombre_casos:
                            conf = "media"
                            notas.append("el número de Serie y el nombre citado no coinciden: manda el número")
                    if d is None:
                        if mn.get("casos"):
                            d, cands, conf, n2 = _elegir_resolucion(mn["casos"], tramo, nums, seg, autor, mn["alias"])
                            notas += n2 + [f"Serie {directo['serie']} No. {directo['num_crudo']} no está en el catálogo"]
                        else:
                            continue
            else:
                d, cands, conf, n2 = _elegir_resolucion(mn["casos"], tramo, nums, seg, autor, mn["alias"])
                notas += n2
                if mn["via"] in ("corte", "adyacente", "voto", "comillas", "sin_pista") and conf == "alta":
                    conf = "media" if mn["clase"] in ("palabra", "apellidos") else conf
                cs = mn.get("conflicto_serie")
                if cs:
                    d_serie = cat["docs"].get(f"{cs['serie']}-{int(cs['num_crudo'])}")
                    if d_serie is None:
                        # «Serie C No. 201517» no existe: manda el nombre, con aviso.
                        conf = "media" if conf == "alta" else conf
                        notas.append(f"Serie {cs['serie']} No. {cs['num_crudo']} no está en el catálogo: "
                                     "se toma el caso nombrado")
                    else:
                        notas.append(f"el número impreso (Serie {cs['serie']} No. {cs['num_crudo']} = "
                                     f"{_nombre_caso(d_serie)}) no es del caso nombrado: no se elige")
                        cands = ([d] if d is not None else cands) + [d_serie]
                        d, conf = None, "ambigua"
            if d is None and not cands:
                continue
            if d is not None and seg == "voto" and autor and not any(autor in v["slug"].split("+") for v in d["votos"]):
                notas = [n for n in notas if not n.startswith("el listado no registra un voto")]
                notas.append(f"el listado no registra un voto de {autor} en {d['doc_id']}")
                conf = "media" if conf == "alta" else conf
            r = dict(
                doc_id=d["doc_id"] if d else None,
                caso=_nombre_caso(d) if d else None,
                caso_id=d.get("caso_id") if d else None,
                serie=d["serie"] if d else None,
                serie_num=d.get("serie_num") if d else None,
                fecha=d.get("fecha") if d else None,
                tipo_resolucion=d.get("acto") if d else None,
                parrafo=nums[0] if nums else None,
                parrafos=nums,
                seg=seg,
                voto_autor=autor if seg == "voto" else None,
                url_oficial=d.get("url_oficial") if d else None,
                confianza=conf,
                candidatos=[_resumen_doc(c) for c in cands[:12]],
                evidencia=evidencia,
                via=mn["via"],
                alias=mn.get("alias") or None,
                alias_clase=mn.get("clase") if mn.get("alias") else None,
                llaves=[_llave(d, seg, autor, n) for n in nums] if d else [],
                cita_canonica=_cita_canonica(d, seg, nums, autor) if d else None,
                notas=notas,
            )
            resultados.append(r)

    # Mismo documento dicho dos veces sin párrafo, y luego con párrafo: queda
    # el que tiene párrafo. Duplicados exactos, fuera.
    finales: List[Dict[str, Any]] = []
    for r in resultados:
        clave = (r["doc_id"], tuple(c["doc_id"] for c in r["candidatos"]), r["seg"], r["voto_autor"],
                 tuple(r["parrafos"]))
        if any((f["doc_id"], tuple(c["doc_id"] for c in f["candidatos"]), f["seg"], f["voto_autor"],
                tuple(f["parrafos"])) == clave for f in finales):
            continue
        finales.append(r)
    con_parr = {(f["doc_id"]) for f in finales if f["parrafos"] and f["doc_id"]}
    finales = [f for f in finales if f["parrafos"] or f["seg"] != "sentencia" or f["doc_id"] not in con_parr]
    return finales[:max_resultados]


# ═══════════════════════════════════════════════════════════ línea de órdenes

_SIEMPRE = {"doc_id", "serie", "clase", "nombre", "cita", "tipo", "componentes", "votos"}


def _escribir(cat: Dict[str, Any], ruta: Path) -> None:
    """Un documento y un caso por línea: el archivo se revisa con `git diff`
    (una alta del listado es una línea nueva) y pesa la mitad que con sangría.
    Las claves vacías se omiten, salvo las que el resolvedor lee siempre."""
    def limpio(d):
        return {k: v for k, v in d.items() if k in _SIEMPRE or v not in (None, [], "")}
    partes = ["{"]
    for k in ("version", "generado", "fuente", "cifras", "avisos", "anomalias", "autores"):
        partes.append(f"{json.dumps(k)}: {json.dumps(cat[k], ensure_ascii=False, separators=(',', ':'))},")
    for k in ("casos", "documentos"):
        filas = [f"  {json.dumps(i, ensure_ascii=False)}: "
                 f"{json.dumps(limpio(v) if k == 'documentos' else v, ensure_ascii=False, separators=(',', ':'))}"
                 for i, v in cat[k].items()]
        cierre = "}," if k == "casos" else "}"
        partes.append(f"{json.dumps(k)}: {{\n" + ",\n".join(filas) + f"\n{cierre}")
    partes.append("}")
    ruta.write_text("\n".join(partes) + "\n", encoding="utf-8")


def _main(argv: List[str]) -> int:
    if len(argv) >= 2 and argv[0] == "--construir":
        cat = construir(argv[1])
        # Plan 3.1: un listado caído (ColdFusion responde error con código 200)
        # no puede borrar documentos. Si alguna clase trae MENOS que el
        # catálogo vigente, no se escribe nada salvo con --forzar.
        if RUTA_CATALOGO.exists() and "--forzar" not in argv:
            antes = json.loads(RUTA_CATALOGO.read_text(encoding="utf-8"))["cifras"]["documentos"]
            menos = {k: (antes[k], v) for k, v in cat["cifras"]["documentos"].items() if v < antes.get(k, 0)}
            if menos:
                print(f"NO SE ESCRIBE: el listado nuevo trae menos documentos que el vigente {menos}")
                return 1
        RUTA_CATALOGO.parent.mkdir(parents=True, exist_ok=True)
        _escribir(cat, RUTA_CATALOGO)
        json.loads(RUTA_CATALOGO.read_text(encoding="utf-8"))          # que se relea sin error
        print(json.dumps(cat["cifras"], ensure_ascii=False, indent=1))
        print(f"avisos: {len(cat['avisos'])}  anomalías: {len(cat['anomalias'])}  → {RUTA_CATALOGO}")
        return 0
    for t in argv or [sys.stdin.read()]:
        print(json.dumps(resolver_citas_coidh(t), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
