# -*- coding: utf-8 -*-
"""LA EXHAUSTIVIDAD DEL ESTUDIO — lo que las marcas no comprueban solas.

POR QUÉ EXISTE. El banco del 26-sep-2026 (localizador a ciegas, 7 asuntos, 41
corridas, 707 citas verificadas por código) midió la v3 y la v4 contra la v1:
contestan con razón propia el 90 % y el 79 % de los argumentos decisivos (la
v1, el 98 %) y dejan más omisiones graves por corrida (0.84 y 0.62 frente a
0.50). Las marcas dicen 100 %: el modelo marca el párrafo aunque conteste en
genérico. Leídas una a una, las omisiones graves son de dos clases:

  1. EL ESTUDIO DESPACHA UN ARGUMENTO CON UNA DECLARACIÓN DE «SIN MATERIA» —o
     innecesario, sin objeto, sin beneficio adicional— y los EFECTOS no lo
     recogen, así que la responsable puede volver a resolver igual sin
     desacatar nada (174/2026: custodia, doble jornada, tercer inmueble;
     43/2025: la pericial en informática y la jurisprudencia 2017826).
  2. EL ARGUMENTO CON DATO PROPIO QUEDA ABSORBIDO en una calificación global
     (103/2025, 263/2025, 722/2025).

Este módulo trae los TRES remedios que se construyeron para eso, los tres sin
tocar la v1:

  · `sin_materia_por_su_cuenta` (control del punto 1): párrafos del estudio
    que declaran sin estudio un argumento cuyo criterio NO lo calificó así.
  · `sin_su_dato` (la «marca honesta», punto 2): argumentos con dato propio
    cuya marca cae en un párrafo que sólo los declara sin estudio, sin que su
    dato aparezca ni en los EFECTOS ni en una respuesta de fondo.
  · `reparar` (punto 3): UNA llamada más al modelo del estudio, sólo en la v3
    y la v4, con esos argumentos como DATOS; devuelve el párrafo o la orden
    de efectos que falta, y el código lo inserta sin tocar nada más.

LO QUE SE MIDIÓ ANTES DE ESCRIBIR ESTO (scratchpad/exhaustivo, 26-sep-2026;
402 pares argumento × corrida v3/v4, emparejados a mano con el inventario):

  · LA MARCA NO ESTÁ DONDE ESTÁ LA RESPUESTA. De 317 respuestas propias o
    remisiones que el localizador encontró, sólo 171 (54 %) caen en el
    párrafo marcado; 64 (20 %) en los párrafos siguientes antes de otra
    marca, 51 (16 %) antes de la marca y 31 después de otra. El modelo marca
    el párrafo que abre el apartado —la premisa— y contesta más abajo.
  · POR ESO LA VERSIÓN LITERAL DE LA MARCA HONESTA NO DISTINGUE. «El párrafo
    marcado contiene el dato del argumento (un ancla, o una proporción de sus
    palabras propias)» acusa igual a las respuestas genéricas que a las
    propias: con cualquier umbral, la precisión queda entre el 11 y el 16 %,
    que es la tasa de base (36 genéricas de 359 pares con segmento). Y las
    genéricas SÍ contienen las palabras del argumento: lo nombran para
    declararlo innecesario.
  · LO QUE SÍ DISTINGUE es que el párrafo marcado DECLARE el argumento sin
    estudio: 26 de las 36 genéricas frente a 31 de las 279 propias. Con la
    condición de que el dato no esté en los EFECTOS ni en otro párrafo de
    fondo, queda la regla de `sin_su_dato` (tabla en `UMBRAL_EFECTOS`).
  · 12 de las 27 omisiones graves de la v3/v4 (pares argumento × corrida) NO
    ESTÁN EN EL INVENTARIO: el resumen de la fase 2 no las trae (la
    testimonial aleccionada y la falsedad sobre el hijo en el 103, el oficio
    mal pedido al RAN en el 263). Tampoco 31 de las 67 respuestas genéricas o
    ausentes. Ninguna comprobación sobre las marcas puede verlas; es cosa del
    resumen o de la cobertura contra la demanda (V1b). (Recontado en la
    revisión adversarial: el informe decía 11.)
"""
from __future__ import annotations

import asyncio
import math
import os
import re
from collections import Counter

# ═══════════════════════════════════════════════════════════════════════════
# LA DECLARACIÓN DE «SIN ESTUDIO»
# ═══════════════════════════════════════════════════════════════════════════
# Lo que se busca es la DECISIÓN DE NO ESTUDIAR: «resulta innecesario analizar»,
# «queda sin materia», «carece de objeto examinar», «no produciría un beneficio
# adicional», «a ningún fin práctico conduciría». Sin el verbo de estudio al
# lado, «innecesario» sale en cualquier otra cosa —«litigios innecesarios»,
# «una medida innecesaria», el «beneficio adicional» de un seguro—: la primera
# versión, sin esa exigencia, veía declaraciones en 12 de los 40 engroses
# reales del corpus y ninguna lo era; con ella, en 3, y las tres son de verdad
# (ver `CALIBRACION_ORO`).
_EST = (r"(?:estudi\w*|an[aá]lisis|analiz\w*|exam\w*|pronunci\w*|ocup\w*|resolver|"
        r"determinar|abordar|atender|valorar|decidir|calific\w*)")
_RX_DECL = re.compile(
    r"(?:"
    r"(?:qued\w+|declar\w+|dej\w+|result\w+|est[aá]n?|se\s+encuentra\w*)\s+(?:\w+\s+){0,3}sin\s+materia"
    r"|sin\s+materia\s+(?:el|su|la|los|las)\s+" + _EST +
    r"|(?:resulta\w*|es|son|deviene\w*|torna\w*|vuelve\w*|estima\w*|considera\w*|queda\w*|"
    r"ser[ií]a\w*)\s+(?:\w+\s+){0,2}(?:innecesari[oa]s?|ocios[oa]s?)\b[^.;]{0,90}?" + _EST +
    r"|" + _EST + r"[^.;]{0,90}?(?:resulta\w*|es|son|deviene\w*|torna\w*|ser[ií]a\w*)\s+"
    r"(?:\w+\s+){0,2}(?:innecesari[oa]s?|ocios[oa]s?)\b"
    r"|innecesari[oa]s?\s+(?:el\s+|su\s+|entrar\s+al\s+|ocuparse\s+del?\s+)?" + _EST +
    # LA CALIFICACIÓN MISMA: el apartado que abre con la fórmula y cierra la
    # frase con «Resulta innecesario.» o «Es innecesario en lo relativo a…».
    r"|\b(?:resulta\w*|es|son|deviene\w*)\s+(?:\w+\s+){0,2}innecesari[oa]s?\s*(?:[.;:,]|$|en\s+lo\s+"
    r"relativo|respecto|en\s+cuanto|por\s+lo\s+que|mientras)"
    r"|\b(?:vuelve\w*|torna\w*|hace\w*)\s+innecesari[oa]s?\s+(?:el|los|la|las|su|sus)\b"
    r"|\b(?:no|tampoco)\s+(?:es|resulta|ser[ií]a)\s+necesari[oa]\s+" + _EST +
    r"|carec\w+\s+de\s+(?:objeto|materia|utilidad|sentido\s+pr[aá]ctico)\s+(?:\w+\s+){0,3}" + _EST +
    r"|" + _EST + r"[^.;]{0,120}?carez?c\w+\s+de\s+(?:objeto|materia|utilidad)"
    r"|no\s+(?:puede\s+|podr[ií]a\s+)?(?:produc\w+|report\w+|tendr\w+|traer\w+|generar\w+|aportar\w+|"
    r"conducir\w+\s+a)\s+(?:un|ning[uú]n|mayor)\s+beneficio"
    r"|beneficio\s+(?:adicional|mayor)\s+(?:alguno\s+)?(?:a|para)\s+(?:la\s+parte|el|la|quien)"
    r"|a\s+(?:ning[uú]n\s+fin|nada)\s+pr[aá]ctico\s+conduc\w+"
    r"|quedar[ií]a\s+subordinad\w+"
    r")", re.I)
# LO QUE NO ES UNA DECLARACIÓN DE ESTE TRIBUNAL: el amparo adhesivo, que tiene
# su propia regla (art. 182 LA), y lo que se le atribuye a la responsable o a
# la parte —«la Sala consideró innecesario estudiar…» es un hecho que se
# relata, no una decisión de no estudiar—.
_RX_AJENO = re.compile(
    r"adhesiv|\b(?:consider[oó]|estim[oó]|determin[oó]|sostuvo|concluy[oó]|resolvi[oó]|"
    r"declar[oó]|afirm[oó]|dijo|considerad[oa]s?|establece|dispone|prev[ée]|indica|se[ñn]ala)\s+"
    r"(?:que\s+)?(?:\w+\s+){0,5}(?:innecesari|ocios|sin\s+materia|carec)",
    re.I)
# LA NEGACIÓN: «la nulidad de esos actos no hizo innecesario examinar el
# crédito» dice lo contrario (93/2026, corrida C).
_RX_NEGADO = re.compile(r"\b(?:no|ni|nunca)\s+(?:\w+\s+)?$", re.I)
# EL OBJETO ES «LOS RESTANTES»: el anuncio de método que declara sin estudio
# los demás planteamientos habla de OTROS problemas, no del que nombra
# (103/2025: «su estudio es preferente… y hace innecesario examinar los
# restantes»).
_RX_LOS_DEMAS = re.compile(r"\b(?:restantes|dem[aá]s|accesorios|otros)\b", re.I)
# LA CALIFICACIÓN GENERAL QUE ABRE EL ESTUDIO —«Los conceptos de violación son
# en parte fundados y en parte innecesarios.»— no declara nada de ningún
# argumento en particular: la pone el criterio.
_RX_CALIF_GENERAL = re.compile(
    r"^\s*(?:[A-ZÉÁÍÓÚ]+\.\s+)?(?:Estudio\.\s+)?(?:Los|Las)\s+(?:conceptos|agravios|motivos)"
    r"[^.]{0,60}?\s+son\s+[^.]{0,120}\.", re.I)


def _es_transcripcion(p: str) -> bool:
    """Un rubro o una tesis transcrita: entre comillas y casi todo en mayúsculas.
    Lo que dice una tesis sobre la inoperancia no es una decisión del estudio."""
    t = (p or "").strip()
    if not t or t[0] not in "“\"«":
        return False
    letras = [c for c in t[:90] if c.isalpha()]
    return bool(letras) and sum(c.isupper() for c in letras) / len(letras) > 0.6


def declara_sin_estudio(parrafo: str) -> bool:
    """¿El párrafo decide NO estudiar algo (sin materia, innecesario…)?"""
    t = parrafo or ""
    if _es_transcripcion(t):
        return False
    g = _RX_CALIF_GENERAL.match(t)
    if g:
        t = " " * g.end() + t[g.end():]
    for m in _RX_DECL.finditer(t):
        if _RX_AJENO.search(t[max(0, m.start() - 120): m.end() + 20]):
            continue
        if _RX_NEGADO.search(t[max(0, m.start() - 30): m.start()]):
            continue
        return True
    return False


def _declara_los_demas(parrafo: str) -> bool:
    """¿La declaración recae sobre «los restantes / los demás / los
    accesorios»? Entonces no habla del concepto que el párrafo nombra."""
    t = parrafo or ""
    for m in _RX_DECL.finditer(t):
        if _RX_LOS_DEMAS.search(t[max(0, m.start() - 80): m.end() + 80]):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# EL CRITERIO DE CADA ARGUMENTO
# ═══════════════════════════════════════════════════════════════════════════
# LAS CALIFICACIONES DE FONDO: con ellas el criterio dice que el argumento se
# estudia. Las demás —innecesario, sin materia, inoperante, ineficaz,
# inatendible, fundado pero insuficiente— admiten decir que su estudio no
# conduce a nada (la inoperancia se razona muchas veces así). La escala de
# extensión de la v2 dice lo mismo (`fase6_estudio._prompt_estudio_v2`).
FONDO = {"fundado", "infundado", "esencialmente_fundado", "parcialmente_fundado",
         "sustancialmente_fundado", "fundado_suplido"}


def _norm_sentido(s) -> str:
    return re.sub(r"\s+", "_", str(s or "").strip().lower())


def _get(o, k, defecto=None):
    if isinstance(o, dict):
        return o.get(k, defecto)
    return getattr(o, k, defecto)


def _concepto(seg) -> int:
    try:
        return int(_get(seg, "concepto") or 0)
    except (TypeError, ValueError):
        m = re.match(r"^(?:AD|[CAS])(\d+)", str(_get(seg, "id") or ""))
        return int(m.group(1)) if m else 0


def reparto(criterios: list, problemas: list) -> list:
    """[(criterio, [conceptos que cubre])], con el mismo casamiento que usa el
    prompt para la línea CUBRE (`formato_sentencia.cubre_de`)."""
    try:
        import formato_sentencia as _fs
        return [(c, _fs.cubre_de(c, problemas or [])) for c in (criterios or [])]
    except Exception:
        return [(c, []) for c in (criterios or [])]


def criterios_del_segmento(seg, rep: list) -> list:
    """Los criterios cuyo reparto cubre el concepto del argumento."""
    k = _concepto(seg)
    return [c for c, cub in rep if k and k in cub]


def _hay_permitido(criterios: list) -> bool:
    return any(_norm_sentido(_get(c, "sentido")) not in FONDO and _get(c, "sentido")
               for c in (criterios or []))


# ═══════════════════════════════════════════════════════════════════════════
# LA ESTRUCTURA DEL ESTUDIO: CUERPO, EFECTOS, ADVERTENCIAS
# ═══════════════════════════════════════════════════════════════════════════
_RX_ADVERTENCIAS = re.compile(r"^\s*ADVERTENCIAS?\s*[:.]?\s*$", re.I)


def partes(parrafos: list) -> tuple:
    """(fin del cuerpo, fin de los efectos): índices sobre `parrafos`. Los
    efectos van de uno a otro; lo que sigue —ADVERTENCIAS, si el texto las
    traía— no cuenta como sentencia."""
    n = len(parrafos or [])
    fin = next((i for i, p in enumerate(parrafos or []) if _RX_ADVERTENCIAS.match(p or "")), n)
    try:
        import documento_generado as _dg
        ini = next((i for i in range(fin) if _dg._RX_INICIO_EFECTOS.search(parrafos[i] or "")
                    or _dg._RX_UN_EFECTO.match(parrafos[i] or "")), fin)
    except Exception:
        ini = next((i for i in range(fin) if re.match(r"^\s*EFECTOS\b", parrafos[i] or "")), fin)
    return ini, fin


# ═══════════════════════════════════════════════════════════════════════════
# EL DATO PROPIO DE UN ARGUMENTO
# ═══════════════════════════════════════════════════════════════════════════
# LO QUE UN ARGUMENTO TRAE DE SUYO: sus anclas duras (artículo con su ley,
# registro, expediente, cifra, fecha) que no comparte con más de otros dos del
# inventario, y sus palabras con contenido, pesadas por lo raras que son en el
# inventario —la misma medida que el rescate de `marcas.rastros`—. La
# invocación de los artículos 1, 14, 16 y 17 constitucionales no es un dato:
# la trae casi cualquier concepto de violación, y el localizador la cuenta
# como residual genérico.
GENERICAS = {"art 1 cpeum", "art 14 cpeum", "art 16 cpeum", "art 17 cpeum"}
MAX_COMPARTIDA = 2
# Un argumento sin anclas propias tiene dato propio si le quedan al menos tres
# palabras que no salen en más de dos argumentos del inventario. No cuentan
# las cifras sueltas (las que importan ya son anclas) ni el vocabulario de la
# invocación genérica —«vulneran los artículos 14 y 16 constitucionales», «los
# principios de legalidad y seguridad jurídica»—, que no es un dato.
MIN_RAICES_PROPIAS = 3
RAICES_GENERICAS = {
    "vulner", "transg", "viola", "consti", "garant", "legali", "seguri", "juridi",
    "debido", "proces", "fundam", "motiva", "congru", "exhaus", "princi", "humano",
    "tutela", "judici", "efecti", "acceso", "justic", "pronta", "expedi", "impart",
    "audien", "defens", "certez", "iguald", "conven", "intern", "tratad", "pacto",
    "violan", "violat",
    "nacion", "federa", "politi", "mexica", "estado", "unidos", "fracci", "parraf"}


def _raices(t: str) -> set:
    import inventario as _inv
    return _inv._raices(t or "")


def _anclas(t: str) -> set:
    import inventario as _inv
    return set(_inv.anclas_duras(t or ""))


class Datos:
    """El dato propio de cada argumento del inventario, calculado una vez."""

    def __init__(self, segs: list):
        self.segs = [s for s in (segs or []) if _get(s, "id")]
        self.raices = {str(_get(s, "id")): _raices(str(_get(s, "texto") or "")) for s in self.segs}
        df = Counter(r for rs in self.raices.values() for r in rs)
        n = len(self.segs)
        self.df = df
        self.idf = {r: math.log((n + 1) / (c + 0.5)) for r, c in df.items()}
        cuenta = Counter(a for s in self.segs for a in set(_get(s, "anclas") or []))
        self.anclas = {str(_get(s, "id")): [a for a in (_get(s, "anclas") or [])
                                            if a not in GENERICAS and cuenta[a] <= MAX_COMPARTIDA]
                       for s in self.segs}

    def tiene_dato(self, sid: str) -> bool:
        if self.anclas.get(sid):
            return True
        propias = [r for r in self.raices.get(sid, ()) if self.df[r] <= MAX_COMPARTIDA
                   and not r.isdigit() and r not in RAICES_GENERICAS]
        return len(propias) >= MIN_RAICES_PROPIAS

    def cubre(self, sid: str, texto: str) -> tuple:
        """(¿alguna de sus anclas propias está en el texto?, proporción
        ponderada de sus palabras que están en el texto)."""
        a = bool(set(self.anclas.get(sid) or []) & _anclas(texto)) if self.anclas.get(sid) else False
        rs = self.raices.get(sid) or set()
        tot = sum(self.idf.get(r, 0) for r in rs) or 1.0
        rt = _raices(texto)
        return a, round(sum(self.idf.get(r, 0) for r in rs & rt) / tot, 3)



# ═══════════════════════════════════════════════════════════════════════════
# PUNTO 1 · «SIN MATERIA» POR SU CUENTA
# ═══════════════════════════════════════════════════════════════════════════
_ORD_TXT = {"primer": 1, "primero": 1, "segundo": 2, "tercer": 3, "tercero": 3,
            "cuarto": 4, "quinto": 5, "sexto": 6, "septimo": 7, "séptimo": 7,
            "octavo": 8, "noveno": 9, "decimo": 10, "décimo": 10, "unico": 1, "único": 1}
_RX_ORD = re.compile(r"\b(primer|primero|segundo|tercer|tercero|cuarto|quinto|sexto|s[ée]ptimo|"
                     r"octavo|noveno|d[ée]cimo|[úu]nico)\b", re.I)
_RX_NOMBRA_CONCEPTO = re.compile(r"\b(?:concepto|agravio|motivo)s?\b", re.I)


def _ordinales(p: str) -> set:
    if not _RX_NOMBRA_CONCEPTO.search(p or ""):
        return set()
    return {_ORD_TXT.get(m.group(1).lower().replace("é", "e").replace("ú", "u"), 0)
            for m in _RX_ORD.finditer(p or "")} - {0}


def sin_materia_por_su_cuenta(parrafos: list, mapa: dict, segs: list,
                              criterios: list, problemas: list) -> list:
    """Los párrafos del cuerpo que declaran sin estudio algo que el criterio
    manda estudiar. [{parrafo, ids, conceptos, sentidos, texto}].

    Con marcas (v3/v4), cada identificador marcado en el párrafo se lleva a
    los criterios que cubren su concepto; se acusa si TODOS son de fondo
    (fundado o infundado). Sin marcas —o si el párrafo no lleva ninguna—, por
    los ordinales de concepto que nombra el párrafo o el último apartado
    abierto; y si no nombra ninguno, sólo cuando el criterio entero no tiene
    ninguna calificación que admita no estudiar."""
    ps = list(parrafos or [])
    fin_cuerpo, _ = partes(ps)
    rep = reparto(criterios, problemas)
    por_id = {str(_get(s, "id")): s for s in (segs or [])}
    en_parrafo = {}
    for sid, idxs in (mapa or {}).items():
        for i in idxs or []:
            en_parrafo.setdefault(int(i), []).append(sid)
    fuera, ultimo_ord, ultimos_ids = [], set(), []
    for i in range(fin_cuerpo):
        p = ps[i] or ""
        ords = _ordinales(p[:220])
        if ords:
            ultimo_ord = ords
        propios = [x for x in en_parrafo.get(i, []) if x in por_id]
        if propios:
            ultimos_ids = propios
        elif ords:
            # Un apartado nuevo sin marca: lo que venía marcado ya no manda.
            ultimos_ids = []
        if not declara_sin_estudio(p):
            continue
        # LA MARCA MANDA HASTA LA SIGUIENTE (medido: el modelo marca el primer
        # párrafo del apartado y sigue en los de abajo sin marca). Un párrafo
        # sin marca hereda los identificadores del último que la llevaba.
        ids = propios or ultimos_ids
        sentidos, acusa = [], False
        if _declara_los_demas(p) and not propios:
            # «…hace innecesario examinar los restantes»: habla de otros
            # problemas. Sólo se acusa si el criterio no tiene ninguno que
            # admita no estudiarse.
            sentidos = [_norm_sentido(_get(c, "sentido")) for c in (criterios or [])]
            if bool(criterios) and not _hay_permitido(criterios):
                fuera.append({"parrafo": i, "ids": [], "conceptos": [], "sentidos": sorted(set(sentidos)),
                              "texto": " ".join(p.split()[:24])})
            continue
        if ids:
            malos = []
            for sid in ids:
                cs = criterios_del_segmento(por_id[sid], rep)
                ss = [_norm_sentido(_get(c, "sentido")) for c in cs if _get(c, "sentido")]
                sentidos.extend(ss)
                if ss and all(s in FONDO for s in ss):
                    malos.append(sid)
            ids, acusa = malos, bool(malos)
            conceptos = sorted({_concepto(por_id[x]) for x in malos})
        else:
            conceptos = sorted(_ordinales(p) or ultimo_ord)
            cs = [c for c, cub in rep if set(cub) & set(conceptos)] if conceptos else []
            if cs:
                sentidos = [_norm_sentido(_get(c, "sentido")) for c in cs if _get(c, "sentido")]
                acusa = bool(sentidos) and all(s in FONDO for s in sentidos)
            else:
                sentidos = [_norm_sentido(_get(c, "sentido")) for c in (criterios or [])]
                acusa = bool(criterios) and not _hay_permitido(criterios)
        if acusa:
            fuera.append({"parrafo": i, "ids": ids, "conceptos": conceptos,
                          "sentidos": sorted(set(sentidos)),
                          "texto": " ".join(p.split()[:24])})
    return fuera


# CALIBRADO CONTRA LO REAL (26-sep-2026; scratchpad/exhaustivo/calib_sin_materia):
#   · 40 ENGROSES REALES del corpus Kingston (campo «oro»), con el criterio
#     aproximado por su propia calificación de apertura: acusa a UNO, el ADC
#     810/2025, que es bueno —declara innecesario un agravio de un concepto
#     fundado porque otro da mayor beneficio (art. 189), práctica de siempre—.
#   · LAS 41 CORRIDAS DEL BANCO, con el criterio real de la ficha: salta en 9.
#     Seis son del 174/2026, donde el resumen de la fase 2 dice «único
#     concepto» y la fase 3 repartió la doble jornada, el tercer inmueble y la
#     perspectiva de género en problemas 2 a 4 innecesarios: el control ve todo
#     bajo el problema 1, fundado. Coincide con las omisiones graves del
#     localizador en las tres corridas que las tienen, pero contra el criterio
#     escrito es una atribución equivocada. Las otras tres (103 D y E, 93 E)
#     son párrafos sin marca atribuidos al apartado equivocado.
# ACUSA A UN ENGROSE BUENO: VA EN SOMBRA. Se registra en la ficha y en el
# «listo» (`cobertura.exhaustivo.sin_materia`), no se enseña. La variable
# existe para encenderlo sin desplegar cuando se recalibre.
SIN_MATERIA_VISIBLE = os.getenv("ESTUDIO_SIN_MATERIA_VISIBLE", "0") == "1"
CALIBRACION_ORO = ("40 engroses: acusa 1 (ADC 810/2025, bueno); 41 corridas: salta en 9 "
                   "(6 del 174/2026 por el reparto de la fase 3)")


def aviso_sin_materia(hallazgos: list, q1: str = "concepto de violación") -> str:
    if not hallazgos:
        return ""
    trozos = []
    for h in hallazgos[:5]:
        quien = (", ".join(h["ids"]) if h.get("ids") else
                 (f"{q1} " + ", ".join(str(c) for c in h.get("conceptos") or []) if h.get("conceptos")
                  else "un argumento"))
        trozos.append(f"{quien} («{h['texto']}…»)")
    return ("DECLARADO SIN ESTUDIO SIN QUE EL CRITERIO LO DIGA: el estudio declara sin "
            f"materia o innecesario {'; '.join(trozos)}, y el criterio de ese problema es de "
            "fondo. Revise si esa respuesta alcanza o si el argumento debe contestarse o "
            "nombrarse en los efectos.")


# ═══════════════════════════════════════════════════════════════════════════
# PUNTO 2 · LA MARCA HONESTA: EL ARGUMENTO MARCADO, ¿TIENE SU DATO?
# ═══════════════════════════════════════════════════════════════════════════
# LA REGLA, CALIBRADA CONTRA EL LOCALIZADOR (ver el docstring del módulo y el
# informe): un argumento con dato propio se acusa cuando su marca cae en un
# párrafo que lo declara sin estudio Y su dato no aparece en los EFECTOS
# (ancla propia o `UMBRAL_EFECTOS` de sus palabras) NI en ningún otro párrafo
# de fondo del cuerpo (ancla propia o `UMBRAL_SUSTANTIVO`).
#
# TABLA DE CONFUSIÓN (359 pares argumento del localizador × corrida v3/v4 con
# segmento en el inventario; 27 corridas, 7 asuntos):
#                               acusado   no acusado
#     genérica / ausente            24           12      → exhaustividad 67 %
#     propia / remisión             38          285      → acusa al 12 %
#       (propias 22 de 279 = 8 %; remisiones 16 de 44)
#   precisión 39 % (la tasa de base es 10 %); omisiones graves con segmento:
#   11 de 15. Por segmento × corrida (278): 24 de 36 malos y 25 de 242 buenos,
#   precisión 49 %. Los umbrales se barrieron (efectos 0.3-0.5, fondo
#   0.4-0.7): con 0.3 en los efectos se pierden tres graves; con 0.5 se gana
#   una genérica a cambio de tres buenas; por encima de 0.6 en el fondo ya no
#   cambia nada.
UMBRAL_EFECTOS = 0.4
UMBRAL_SUSTANTIVO = 0.6
#
# LO QUE LA TABLA NO DICE (revisión adversarial, 26-sep-2026, recontado):
#   · El denominador son los argumentos QUE ESTÁN EN EL INVENTARIO. Contando
#     también los que el resumen de la fase 2 no trae, las respuestas genéricas
#     o ausentes son 67 y se acusan 24: el 36 %. De las 27 omisiones graves
#     (pares argumento × corrida) se acusan 11 (41 %); 12 no están en el
#     inventario.
#   · Todo está medido DENTRO de la muestra: la regex y los umbrales se
#     ajustaron sobre estas mismas corridas. Dejando fuera cada asunto para
#     elegir el umbral y midiendo en él, queda 64 % / 12 % (precisión 37 %).
#     Con el emparejamiento automático (similitud + posición) en lugar del
#     hecho a mano, 21 de 67: el mapeo a mano no infla el resultado.
#   · 20 de los 24 aciertos son de dos asuntos (174/2026 y 43/2025): lo que
#     esto ve es el patrón «declarado innecesario y los efectos callan», no el
#     argumento absorbido en una calificación global (263/2025 y 722/2025: 0).


_RX_LISA_Y_LLANA = re.compile(r"\blis[oa]s?\s+y\s+llan[oa]s?\b", re.I)


def es_lisa_y_llana(efectos: str) -> bool:
    """¿Los EFECTOS conceden de manera lisa y llana (o mandan declarar la
    nulidad lisa y llana)? Sin contar la mención negada («sin que pueda
    declarar la nulidad lisa y llana»)."""
    t = efectos or ""
    for m in _RX_LISA_Y_LLANA.finditer(t):
        if not re.search(r"\b(?:no|sin)\b", t[max(0, m.start() - 45):m.start()], re.I):
            return True
    return False


def sin_su_dato(parrafos: list, mapa: dict, segs: list) -> list:
    """[{id, parrafo, motivo, cubre_efectos, cubre_fuera}] de los argumentos
    con dato propio cuya respuesta es sólo una declaración de sin estudio."""
    ps = list(parrafos or [])
    if not ps or not segs:
        return []
    fin_cuerpo, fin_ef = partes(ps)
    efectos = "\n".join(ps[fin_cuerpo:fin_ef])
    if es_lisa_y_llana(efectos):
        # EL MAYOR BENEFICIO (art. 189 LA; revisión adversarial, 26-sep-2026):
        # con una concesión lisa y llana no queda nada que la responsable deba
        # volver a examinar, y declarar innecesario lo demás es lo correcto.
        # Nombrarlo en los EFECTOS contradiría la concesión. Ninguna de las 27
        # corridas v3/v4 del banco tiene efectos así: la calibración no cambia.
        return []
    datos = Datos(segs)
    decl = {i for i in range(fin_cuerpo) if declara_sin_estudio(ps[i])}
    fuera = []
    for s in datos.segs:
        sid = str(_get(s, "id"))
        idx = sorted(int(i) for i in (mapa or {}).get(sid, []) if int(i) < fin_cuerpo)
        if not idx or not datos.tiene_dato(sid):
            continue
        if not any(i in decl for i in idx):
            continue
        a_ef, c_ef = datos.cubre(sid, efectos) if efectos else (False, 0.0)
        if a_ef or c_ef >= UMBRAL_EFECTOS:
            continue
        mejor, en_otro = 0.0, False
        for i in range(fin_cuerpo):
            if i in decl:
                continue
            a, c = datos.cubre(sid, ps[i])
            mejor = max(mejor, c)
            if a or c >= UMBRAL_SUSTANTIVO:
                en_otro = True
                break
        if en_otro:
            continue
        fuera.append({"id": sid, "parrafo": next(i for i in idx if i in decl),
                      "motivo": "declarado sin estudio, sin su dato en los efectos ni en otra respuesta",
                      "cubre_efectos": c_ef, "cubre_fuera": round(mejor, 3)})
    return fuera


_RX_MAYOR_BENEFICIO = re.compile(r"\bmayor\s+beneficio\b|\bart(?:[íi]culo|\.)?\s*189\b", re.I)


def sin_mayor_beneficio(faltan: list, segs: list, criterios: list, problemas: list) -> list:
    """Quita de `faltan` los argumentos cuyo criterio —por el reparto— invoca en
    la RAZÓN DEL SECRETARIO el mayor beneficio del art. 189: ahí la declaración
    de sin estudio es suya y es la correcta, y nombrarlo en los EFECTOS
    contradiría la concesión. Es del criterio, no de lo que el estudio diga
    (43/2025 escribió «no produciría un beneficio adicional» por su cuenta)."""
    rep = reparto(criterios, problemas)
    por_id = {str(_get(s, "id")): s for s in (segs or [])}
    fuera = []
    for f in faltan or []:
        s = por_id.get(f.get("id"))
        cs = criterios_del_segmento(s, rep) if s is not None else []
        if cs and all(_RX_MAYOR_BENEFICIO.search(str(_get(c, "razonamiento") or "")) for c in cs):
            continue
        fuera.append(f)
    return fuera


def revisar_texto(estudio: str, segs: list, criterios: list, problemas: list) -> dict:
    """Los dos controles sobre el estudio CON sus marcas. Nunca lanza. El de
    la marca honesta ya sin lo que el criterio dejó fuera por mayor beneficio."""
    try:
        import marcas as _mc
        limpio, mapa = _mc.separar_marcas(estudio or "")
        ps = _mc.parrafos(limpio)
        return {"sin_dato": sin_mayor_beneficio(sin_su_dato(ps, mapa, segs), segs, criterios, problemas),
                "sin_materia": sin_materia_por_su_cuenta(ps, mapa, segs, criterios, problemas)}
    except Exception as ex:
        return {"sin_dato": [], "sin_materia": [], "error": type(ex).__name__}


# ═══════════════════════════════════════════════════════════════════════════
# PUNTO 3 · LA REPARACIÓN DIRIGIDA (SÓLO v3 Y v4)
# ═══════════════════════════════════════════════════════════════════════════
# UNA llamada más al MISMO modelo del estudio, con el mismo cliente, cuando
# `sin_su_dato` encuentra argumentos cuya respuesta es sólo una declaración de
# sin estudio. Recibe el estudio escrito (con sus marcas), el criterio del
# secretario y, COMO DATOS, la lista de esos argumentos; devuelve, por cada
# uno, UNA pieza: el párrafo que falta (con su marca) o la orden para los
# EFECTOS que lo nombra entre lo que la responsable deberá examinar. El código
# la inserta —el párrafo justo después del que ya lo marcaba; la orden al final
# de los EFECTOS— y no toca nada más. Lo que no pasa las guardas se descarta.
# Si la llamada falla o vence, el estudio sale como estaba y se avisa.
#
# EL RAZONAMIENTO, EL DEL ESTUDIO: «high» (integración, 26-sep-2026). Esta
# llamada escribe párrafos del estudio, y la regla de la casa es que el
# razonamiento del estudio no se baja. `ESFUERZO_REPARAR` puede subirlo o
# bajarlo a «medium» a propósito; cualquier otro valor se lee como «high».
REPARAR_ACTIVO = os.getenv("ESTUDIO_REPARAR", "1") != "0"
_ESFUERZOS_VALIDOS = ("medium", "high")


def esfuerzo_reparar() -> str:
    e = (os.getenv("ESFUERZO_REPARAR", "high") or "").strip().lower()
    return e if e in _ESFUERZOS_VALIDOS else "high"


# MEDIDO (11 llamadas reales, 26-sep-2026): de 4.3 a 8.5 s con razonamiento
# medio, 6-7 mil tokens de entrada. El tope deja margen de sobra para la hora
# pico del proveedor sin colgar la pantalla.
REPARAR_TOPE_S = float(os.getenv("ESTUDIO_REPARAR_TOPE_S", "120"))
REPARAR_MAX_TOKENS = 14000
# A lo sumo tantos argumentos por llamada: una lista más larga es señal de que
# el estudio entero se escribió mal, y eso no se remienda por piezas.
REPARAR_MAX_ARGUMENTOS = 12
TRAMO_ESCRITO = 700          # caracteres del escrito a cada lado de la cita


def _tramo_del_escrito(cita: str, escrito: str) -> str:
    """El pasaje del escrito alrededor de la cita del argumento (normalizado)."""
    if not cita or not escrito:
        return ""
    try:
        import inventario as _inv
        norm = _inv.normalizar_escrito(escrito)[0]
        pl = _inv._plano(norm)
        k = pl.find(_inv._plano(_inv._colapsar(cita)))
        if k < 0:
            return ""
        a, b = max(0, k - TRAMO_ESCRITO), min(len(norm), k + len(cita) + TRAMO_ESCRITO)
        return norm[a:b].strip()
    except Exception:
        return ""


def _bloque_criterio_reparar(criterios: list, rep: list, q1: str) -> str:
    L = []
    for i, (c, cub) in enumerate(rep, 1):
        s = str(_get(c, "sentido") or "").strip() or "sin calificar"
        L.append(f"{i}. [{str(_get(c, 'jerarquia') or 'accesorio').upper()}] {_get(c, 'problema') or ''}")
        L.append(f"   SENTIDO: {s.replace('_', ' ').upper()}")
        if cub:
            L.append(f"   CUBRE, según el reparto: {q1} " + ", ".join(str(x) for x in cub))
        razon = " ".join(str(_get(c, "razonamiento") or "").split())
        if razon:
            L.append(f"   RAZÓN DEL SECRETARIO: {razon[:1500]}")
    return "\n".join(L)


def etiquetas_del_plan(plan) -> dict:
    """{id: etiqueta} de los argumentos del plan (v4), o {}. plan-4: la
    etiqueta es la calificación del ARGUMENTO dentro del sentido de su
    problema, que puede no ser la del problema (p2-congruencia)."""
    if not isinstance(plan, dict):
        return {}
    return {str(s.get("id")): _norm_sentido(s.get("etiqueta")) for s in plan.get("segmentos") or []
            if isinstance(s, dict) and s.get("id") and s.get("etiqueta")}


def prompt_reparacion(estudio: str, criterios: list, material, faltan: list,
                      escrito: str = "", plan: dict = None) -> str:
    """El prompt de la reparación. Sólo descripciones y datos: ni una frase
    que copiar (lección medida tres veces: un ejemplo del prompt se firma
    literal).

    CON PLAN (v4, p2-congruencia): cada argumento lleva, como dato, su
    calificación en el plan, y una línea dice que ésa es la suya. Sin plan
    —la v3— el prompt no cambia ni una coma."""
    _et = etiquetas_del_plan(plan)
    import tipos_asunto as _ta
    tipo = _get(material, "tipo_asunto", "") or "amparo_directo"
    voc = _ta.vocabulario_de(tipo)
    q1 = voc["combate_singular"]
    segs = list(_get(material, "inventario", None) or [])
    por_id = {str(_get(s, "id")): s for s in segs}
    rep = reparto(criterios, _get(material, "problemas", None) or [])
    import marcas as _mc
    limpio, mapa = _mc.separar_marcas(estudio or "")
    ps = _mc.parrafos(limpio)
    datos = Datos(segs)
    filas = []
    for f in faltan:
        s = por_id.get(f["id"])
        if not s:
            continue
        cs = criterios_del_segmento(s, rep)
        num = [str(i) for i, (c, _) in enumerate(rep, 1) if c in cs]
        sent = [str(_get(c, "sentido") or "").replace("_", " ") for c in cs]
        idx = f.get("parrafo")
        hoy = ps[idx] if isinstance(idx, int) and 0 <= idx < len(ps) else ""
        partes_f = [f"{f['id']} · {q1} {_concepto(s)}",
                    (f"lo que se alega, leído del escrito: " if _get(s, "origen") == "escrito"
                     else f"lo que se alega, según el resumen: ")
                    + ' '.join(str(_get(s, 'texto') or '').split())]
        if _get(s, "cita"):
            partes_f.append(f"cita literal del escrito: «{_get(s, 'cita')}»")
        tr = _tramo_del_escrito(str(_get(s, "cita") or ""), escrito)
        if tr:
            partes_f.append(f"pasaje del escrito donde se plantea: «{tr}»")
        if datos.anclas.get(f["id"]):
            partes_f.append("datos duros: " + "; ".join(datos.anclas[f["id"]][:6]))
        partes_f.append("problema del criterio según el reparto: "
                        + (", ".join(f"{n} ({x.upper()})" for n, x in zip(num, sent)) if num
                           else "no se pudo determinar; búscalo en el criterio por lo que se alega"))
        if _et.get(f["id"]):
            partes_f.append(f"calificación de este argumento en el plan: {_et[f['id']].replace('_', ' ').upper()}")
        if hoy:
            partes_f.append(f"lo que dice hoy el estudio en el párrafo que lo marca: «{' '.join(hoy.split())}»")
        filas.append("\n   ".join(partes_f))
    concede = any(_ta.prospera(str(_get(c, "sentido") or "")) for c in (criterios or []))
    _linea_plan = ("\n- El argumento que trae su calificación en el plan se contesta con ella: es la suya\n"
                   "  dentro del sentido de su problema, y no cambia la del problema."
                   if any(_et.get(f["id"]) for f in faltan) else "")
    # SIN TERCERA RESPUESTA PARA EL MAYOR BENEFICIO (revisión adversarial,
    # 26-sep-2026). Se probó una salida «SIN PIEZA» para el argumento que la
    # concesión ya deja sin nada que resolver (art. 189), y en la llamada real
    # sobre 174/2026 A el modelo la usó para dejar fuera C1.i —el crédito que la
    # quejosa sigue pagando— «porque su estudio no produciría un beneficio
    # adicional frente al nuevo análisis»: la misma excusa que es el defecto que
    # se repara, con una concesión para efectos. El caso legítimo del 189 se
    # filtra antes, sin modelo (EFECTOS lisos y llanos en `sin_su_dato`; la razón
    # del secretario que invoca el mayor beneficio en `sin_mayor_beneficio`), y
    # el prompt vuelve a ser el de las once llamadas de la calibración, más la
    # frase de que lo desestimado no va a los EFECTOS (la guarda lo exige igual).
    return f"""Eres el secretario de un Tribunal Colegiado de Circuito. El estudio de fondo que va al
final ya está escrito, y el secretario fijó su criterio. Una revisión automática encontró
argumentos del escrito cuya única respuesta en el estudio es declararlos sin materia,
innecesarios o sin beneficio, sin que su dato aparezca en los EFECTOS ni en otra respuesta de
fondo. Tu tarea es escribir SÓLO lo que falta para cada uno. No reescribes nada del estudio.

QUÉ ESCRIBES POR CADA ARGUMENTO DE LA LISTA — una de dos piezas:
- LA ORDEN PARA LOS EFECTOS, cuando el criterio del problema al que pertenece el argumento lo
  declaró innecesario, o cuando lo que el argumento combate queda comprendido en lo que la
  responsable tendrá que volver a resolver por la concesión: una orden que nombra lo que el
  argumento plantea —el hecho, la prueba, el precepto o el precedente que trae—, no la omisión
  que se le reprocha a la responsable, entre lo que ella deberá examinar al volver a resolver.
  En imperativo y en la misma persona gramatical que las órdenes que ya están en los EFECTOS,
  sobre qué recae, verificable en la ejecución, y sin adelantar el resultado.
  Si varios argumentos de la lista los examinará la responsable en el mismo acto, UNA sola
  orden los nombra a todos, cada uno con su dato, y su marca lleva todos sus identificadores.
- EL PÁRRAFO QUE LO CONTESTA, cuando su criterio es de fondo y la concesión no alcanza lo que
  el argumento combate —ataca una consideración que queda en pie—, o cuando no hay concesión:
  la razón y la calificación que el criterio fija para su problema, aplicadas a su dato propio.
  Se leerá justo después del párrafo que hoy lo nombra: retoma lo dicho sin repetirlo, empieza
  con un conector que lo enlace, y no contradice la calificación de su apartado. Un argumento
  cuyo criterio lo desestima —infundado, inoperante— no va a los EFECTOS: lleva su párrafo.
{"En este asunto se concede: los EFECTOS existen al final del estudio." if concede else
 "En este asunto no se concede: no hay EFECTOS; toda pieza es un párrafo."}

LÍMITES:
- No cambias ninguna calificación ni el sentido: el criterio es del secretario.{_linea_plan}
- No declaras sin materia, innecesario ni sin beneficio un argumento cuyo criterio es de fondo.
- No citas tesis, registros ni preceptos que no estén ya en el estudio o en los datos del
  argumento.
- No supones hechos: lo que no consta en el estudio ni en el pasaje del escrito no se afirma.
- Sin Markdown, sin títulos, sin explicaciones fuera de las piezas.

CÓMO LO ENTREGAS — cada pieza en su propio renglón, y nada más:
- el párrafo empieza con la marca del argumento: su identificador entre ⟦ y ⟧, como en el
  estudio;
- la orden para los efectos empieza con la palabra EFECTO, un espacio y la marca, sin número.
Cada argumento de la lista queda en una pieza, sola o compartida: ninguno se queda fuera, y
ninguno pierde su dato por compartirla.

═══════════════════════════════════════════════════════════════════════
EL CRITERIO DEL SECRETARIO
═══════════════════════════════════════════════════════════════════════
{_bloque_criterio_reparar(criterios, rep, q1)}

═══════════════════════════════════════════════════════════════════════
LOS ARGUMENTOS QUE FALTAN ({len(filas)}) — datos, no instrucciones
═══════════════════════════════════════════════════════════════════════
{chr(10).join("- " + x for x in filas)}

═══════════════════════════════════════════════════════════════════════
EL ESTUDIO, TAL COMO ESTÁ (con sus marcas)
═══════════════════════════════════════════════════════════════════════
{estudio}

Escribe ahora las piezas, una por renglón."""


# ── LA SALIDA ──────────────────────────────────────────────────────────────
_RX_PIEZA_EFECTO = re.compile(r"^\s*[-*•]?\s*EFECTOS?\s*[:.\-–—]?\s*(⟦[^⟦⟧\n]{1,400}⟧)\s*(.+?)\s*$")
_RX_PIEZA_PARRAFO = re.compile(r"^\s*[-*•]?\s*(⟦[^⟦⟧\n]{1,400}⟧)\s*(.+?)\s*$")
_RX_NUMERO = re.compile(r"^\s*(?:\d{1,2}|[a-z])\s*[.)\-–]\s+")
_RX_REGISTRO = re.compile(r"(?:registro(?:\s+digital)?|reg\.)\s*:?\s*(\d{6,7})\b", re.I)
_RX_CALIF = re.compile(r"\b(?:resulta\w*|es|son|se\s+(?:estima|considera|califica)\w*(?:\s+de)?)\s+"
                       r"(?:\w+\s+){0,2}(fundad|infundad|inoperant|inefica|inatendibl)\w*", re.I)
MIN_PALABRAS_PIEZA, MAX_PALABRAS_PIEZA = 12, 450
# LAS CALIFICACIONES QUE MANDAN NO ESTUDIAR EL FONDO: un párrafo que contesta un
# argumento así calificado estudia lo que el secretario decidió no estudiar.
SIN_ESTUDIO = {"innecesario", "sin_materia"}
# UNA ORDEN QUE DICTA EL RESULTADO en lugar de decir qué examinar. Las once
# salidas reales de la calibración empiezan por examine, valore, analice,
# considere o «al emitir la nueva sentencia, examine…»; ninguna casa aquí.
_RX_RESULTADO = re.compile(
    r"\b(?:condene|absuelva|conceda|otorgue|niegue|reconozca|revoque|confirme|decrete|"
    r"declare\s+(?:procedente|improcedente|fundad\w*|infundad\w*|la\s+nulidad|nul\w+|"
    r"prescrit\w*|la\s+prescripci\w+|la\s+caducidad|caduc\w+|probad\w+|acreditad\w+))\b", re.I)


_RX_SIN_PIEZA = re.compile(r"^\s*[-*•]?\s*SIN\s+PIEZA\s*[:.\-–—]?\s*(⟦[^⟦⟧\n]{1,400}⟧)\s*(.*?)\s*$")


def parsear(texto: str, pedidos: list) -> tuple:
    """([(ids, texto)] párrafos, [(ids, texto)] órdenes, [descartes],
    [ids que el modelo dejó SIN PIEZA —el mayor beneficio—])."""
    import marcas as _mc
    pedidos = set(pedidos or [])
    parrafos_, efectos_, fuera, sin_pieza = [], [], [], []
    for ln in (texto or "").split("\n"):
        if not ln.strip():
            continue
        # «SIN PIEZA» (revisión adversarial, 26-sep-2026): el prompt ya no lo
        # ofrece —el modelo lo usó de excusa en 174/2026 A—, pero si lo
        # escribe por su cuenta no se inserta nada y el aviso lo nombra.
        ms = _RX_SIN_PIEZA.match(ln)
        if ms:
            sin_pieza.extend(x for x in _mc.ids_de(ms.group(1)[1:-1])
                             if x in pedidos and x not in sin_pieza)
            continue
        m = _RX_PIEZA_EFECTO.match(ln)
        es_ef = bool(m)
        if not m:
            m = _RX_PIEZA_PARRAFO.match(ln)
        if not m:
            continue
        todos = _mc.ids_de(m.group(1)[1:-1])
        # LOS IDENTIFICADORES QUE NO SE PIDIERON SE QUITAN DE LA MARCA, no
        # tumban la pieza: en la v4 el modelo copia la marca del párrafo vecino
        # con su unidad del plan («⟦U5 C3.d⟧», medido en 722/2025). Sólo se
        # descarta si no queda ninguno de los pedidos.
        ids = [x for x in todos if x in pedidos]
        cuerpo = m.group(2).strip()
        if not ids:
            fuera.append({"ids": todos, "motivo": "identificador que no se pidió"})
            continue
        if "⟦" in cuerpo or "⟧" in cuerpo:
            fuera.append({"ids": ids, "motivo": "marca dentro del texto"})
            continue
        if es_ef:
            cuerpo = _RX_NUMERO.sub("", cuerpo)
        n = len(cuerpo.split())
        if not (MIN_PALABRAS_PIEZA <= n <= MAX_PALABRAS_PIEZA):
            fuera.append({"ids": ids, "motivo": f"extensión fuera de rango ({n} palabras)"})
            continue
        (efectos_ if es_ef else parrafos_).append((ids, cuerpo))
    return parrafos_, efectos_, fuera, sin_pieza


def _direccion(s: str) -> str:
    s = _norm_sentido(s)
    if not s:
        return ""
    import tipos_asunto as _ta
    if _ta.prospera(s):
        return "favor"
    if s in ("infundado", "inoperante", "ineficaz", "inatendible", "fundado_insuficiente"):
        return "contra"
    return ""


def guardas(ids: list, texto: str, es_efecto: bool, estudio: str, segs_por_id: dict,
            rep: list, etiquetas: dict = None) -> str:
    """El motivo para descartar una pieza, o «» si pasa.

    · Un párrafo no puede declarar sin estudio lo que su criterio manda
      estudiar (es el defecto que se está reparando).
    · No puede calificar en la dirección contraria a la del criterio de su
      problema («infundado» donde el criterio dice fundado, o al revés).
    · No puede traer un registro de tesis que no esté en el estudio ni en los
      datos del argumento.
    · (Revisión adversarial, 26-sep-2026.) Una orden de EFECTOS no puede
      recaer sobre un argumento que su criterio desestima —mandaría a la
      responsable reexaminar lo que este tribunal le dio por bueno— ni
      adelantar el resultado (condenar, absolver, conceder, declarar la
      nulidad…): el sentido es del secretario. Y un párrafo no puede contestar
      el fondo de lo que su criterio declaró innecesario."""
    cs = []
    for i in ids:
        s = segs_por_id.get(i)
        if s is not None:
            cs.extend(criterios_del_segmento(s, rep))
    sentidos = [_norm_sentido(_get(c, "sentido")) for c in cs if _get(c, "sentido")]
    # LA CALIFICACIÓN DEL ARGUMENTO EN EL PLAN (v4, plan-4; p2-congruencia):
    # dentro de un problema fundado cabe un argumento infundado —la
    # reconvención del ADC 642/2024—. Si el plan la da para TODOS los de la
    # pieza, la pieza se casa con ella y no con la del problema. Sin plan
    # (v3), la guarda de siempre.
    _et = [etiquetas.get(i) for i in ids] if etiquetas else []
    if _et and all(_et):
        sentidos = [_norm_sentido(x) for x in _et]
    if not es_efecto and declara_sin_estudio(texto) and sentidos and all(x in FONDO for x in sentidos):
        return "declara sin estudio un argumento cuyo criterio es de fondo"
    if not es_efecto and sentidos and all(x in SIN_ESTUDIO for x in sentidos):
        return "contesta un argumento que su criterio declaró innecesario"
    dirs = {_direccion(x) for x in sentidos} - {""}
    if es_efecto and dirs == {"contra"}:
        return "nombra en los efectos un argumento que su criterio desestima"
    if es_efecto:
        _r = _RX_RESULTADO.search(texto)
        if _r:
            return f"la orden adelanta el resultado ({_r.group(0).strip()})"
    if len(dirs) == 1:
        d = dirs.pop()
        for m in _RX_CALIF.finditer(texto):
            k = m.group(1).lower()
            # «fundado pero insuficiente / inoperante» no prospera.
            pero = re.match(r"\w*\s*,?\s*(?:pero|aunque)\s+(?:\w+\s+)?(?:insuficien|inoperan|inefica)",
                            texto[m.end(1):m.end(1) + 60], re.I)
            dir_pieza = "favor" if (k.startswith("fundad") and not pero) else "contra"
            # «no es fundado» va al revés de lo que dice su palabra.
            if re.search(r"\bno\s+$", texto[max(0, m.start() - 4):m.start()], re.I):
                dir_pieza = "contra" if dir_pieza == "favor" else "favor"
            if dir_pieza != d:
                return f"califica en contra del criterio ({m.group(0).strip()})"
    conocidos = set(_RX_REGISTRO.findall(estudio or ""))
    for i in ids:
        for a in (_get(segs_por_id.get(i) or {}, "anclas") or []):
            if a.startswith("reg "):
                conocidos.add(a[4:])
        conocidos |= set(re.findall(r"\b\d{6,7}\b", str(_get(segs_por_id.get(i) or {}, "texto") or "")))
    nuevos = [r for r in _RX_REGISTRO.findall(texto) if r not in conocidos]
    if nuevos:
        return f"cita un registro que no está en el estudio ({', '.join(nuevos[:3])})"
    return ""


# ── LA INSERCIÓN ───────────────────────────────────────────────────────────
def _bloques(estudio: str) -> list:
    """[(primer renglón, último renglón, ids, texto limpio)] del estudio CON
    marcas, un bloque por párrafo como los cuenta `marcas.separar_marcas`: una
    marca sola en su renglón se suma al párrafo siguiente."""
    import marcas as _mc
    lineas = (estudio or "").split("\n")
    fuera, pend, ini_pend = [], [], None
    for k, ln in enumerate(lineas):
        if not ln.strip():
            continue
        ids = [x for _, _, v in _mc.marcas_en(ln) for x in v]
        limpio = _mc.sin_marcas(ln).strip()
        if not limpio:
            if ids:
                pend.extend(ids)
                ini_pend = k if ini_pend is None else ini_pend
            continue
        fuera.append((ini_pend if ini_pend is not None else k, k, pend + ids, limpio))
        pend, ini_pend = [], None
    return fuera


def insertar(estudio: str, parrafos_: list, efectos_: list, segs_por_id: dict) -> tuple:
    """(estudio nuevo, informe). Los párrafos, cada uno justo después del
    último párrafo del cuerpo que ya marcaba alguno de sus argumentos (o, si
    ninguno lo marcaba, después del último de su mismo concepto, o al final
    del cuerpo); las órdenes, al final de los EFECTOS, numeradas en su serie.
    Nada más cambia: todo renglón del estudio sigue ahí, en su orden."""
    lineas = (estudio or "").split("\n")
    bl = _bloques(estudio)
    textos = [b[3] for b in bl]
    fin_cuerpo, fin_ef = partes(textos)
    informe = {"parrafos": [], "efectos": [], "sin_sitio": []}
    tras = {}                       # renglón → [textos a insertar después]
    for ids, t in parrafos_:
        dentro = [b for b in bl[:fin_cuerpo] if set(b[2]) & set(ids)]
        if not dentro:
            conc = {_concepto(segs_por_id.get(i) or {"id": i}) for i in ids}
            dentro = [b for b in bl[:fin_cuerpo]
                      if any(_concepto(segs_por_id.get(x) or {"id": x}) in conc for x in b[2])]
        if dentro:
            ancla = dentro[-1][1]
        elif fin_cuerpo > 0:
            ancla = bl[fin_cuerpo - 1][1]
        else:
            informe["sin_sitio"].append(ids)
            continue
        tras.setdefault(ancla, []).append("⟦" + " ".join(ids) + "⟧ " + t)
        informe["parrafos"].append({"ids": ids, "tras_renglon": ancla})
    if efectos_:
        efs = bl[fin_cuerpo:fin_ef]
        # El rótulo no es una orden: si sólo está el rótulo, no hay dónde.
        ordenes = [b for b in efs if not re.match(r"^\s*EFECTOS\b[^.]{0,60}$", b[3])]
        if not ordenes:
            informe["sin_sitio"].extend(ids for ids, _ in efectos_)
        else:
            ult = ordenes[-1]
            m = re.match(r"^\s*(\d{1,2})\s*[.)]\s+", ult[3])
            ml = re.match(r"^\s*([a-y])\s*\)\s+", ult[3])
            n = int(m.group(1)) if m else None
            letra = ml.group(1) if ml and not m else None
            for ids, t in efectos_:
                if n is not None:
                    n += 1
                    linea = f"{n}. {t}"
                elif letra is not None:
                    letra = chr(ord(letra) + 1)
                    linea = f"{letra}) {t}"
                else:
                    linea = t
                tras.setdefault(ult[1], []).append(linea)
                informe["efectos"].append({"ids": ids})
    nuevas = []
    for k, ln in enumerate(lineas):
        nuevas.append(ln)
        for t in tras.get(k, []):
            nuevas.extend(["", t])
    return "\n".join(nuevas), informe


async def reparar(cliente, estudio: str, criterios: list, material, faltan: list,
                  escrito: str = "", tope_s: float = None, plan: dict = None) -> tuple:
    """(estudio, informe). El estudio sale COMO ESTABA si no hay nada que
    reparar, si la llamada falla o vence, o si ninguna pieza pasa las guardas.
    `informe`: {estado: ok|sin_piezas|sin_pieza|fallo|vencio|nada, pedidos,
    no_pedidos, parrafos, efectos, sin_pieza, sin_respuesta, sin_sitio,
    descartes, registros_nuevos, segundos, error}. «sin_pieza»: el modelo
    dejó todos con su declaración por el mayor beneficio y no se tocó nada."""
    import time as _t
    t0 = _t.perf_counter()
    _todos = list(faltan or [])
    faltan = _todos[:REPARAR_MAX_ARGUMENTOS]
    informe = {"estado": "nada", "pedidos": [f["id"] for f in faltan], "parrafos": [],
               "efectos": [], "descartes": [], "segundos": 0.0, "esfuerzo": esfuerzo_reparar(),
               # Los que pasan del tope no se piden, pero el aviso los nombra.
               "no_pedidos": [f["id"] for f in _todos[REPARAR_MAX_ARGUMENTOS:]],
               "sin_pieza": [], "sin_respuesta": [], "registros_nuevos": []}
    if not faltan or cliente is None:
        return estudio, informe
    segs = list(_get(material, "inventario", None) or [])
    por_id = {str(_get(s, "id")): s for s in segs}
    rep = reparto(criterios, _get(material, "problemas", None) or [])
    try:
        import fase6_estudio as _f6
        import llamada_modelo as _lm
        kw = dict(model=_f6.MODELO_ESTUDIO, max_completion_tokens=REPARAR_MAX_TOKENS,
                  messages=[{"role": "user", "content": prompt_reparacion(
                      estudio, criterios, material, faltan, escrito, plan=plan)}],
                  reasoning_effort=esfuerzo_reparar())
        # SIN LA PETICIÓN DE RESPALDO de `llamada_modelo.crear`: su espera
        # normal para este tope (300 s) pasa del nuestro, y al vencer
        # `wait_for` cancelaría la corrutina pero dejaría viva la tarea
        # original, huérfana. `_crear_una` se cancela entera.
        r = await asyncio.wait_for(_lm._crear_una(cliente, **kw), timeout=tope_s or REPARAR_TOPE_S)
        salida = (r.choices[0].message.content or "").strip()
        try:
            informe["finish_reason"] = str(r.choices[0].finish_reason or "")
            informe["uso"] = _f6._uso_de(getattr(r, "usage", None))
        except Exception:
            pass
    except asyncio.TimeoutError:
        informe.update(estado="vencio", segundos=round(_t.perf_counter() - t0, 1))
        return estudio, informe
    except Exception as ex:
        informe.update(estado="fallo", error=type(ex).__name__,
                       segundos=round(_t.perf_counter() - t0, 1))
        return estudio, informe
    pars, efs, descartes, sin_pieza = parsear(salida, informe["pedidos"])
    buenos_p, buenos_e = [], []
    for lista, destino, es_ef in ((pars, buenos_p, False), (efs, buenos_e, True)):
        for ids, t in lista:
            motivo = guardas(ids, t, es_ef, estudio, por_id, rep, etiquetas_del_plan(plan))
            if motivo:
                descartes.append({"ids": ids, "motivo": motivo})
            else:
                destino.append((ids, t))
    informe["descartes"] = descartes
    informe["segundos"] = round(_t.perf_counter() - t0, 1)
    informe["salida"] = salida[:6000]
    nuevo, ins = insertar(estudio, buenos_p, buenos_e, por_id) if (buenos_p or buenos_e) \
        else (estudio, {"parrafos": [], "efectos": [], "sin_sitio": []})
    puestos = {i for x in ins["parrafos"] + ins["efectos"] for i in x["ids"]}
    # EL MAYOR BENEFICIO: lo que el modelo dejó sin pieza y nada más lo tocó.
    informe["sin_pieza"] = [i for i in sin_pieza if i not in puestos]
    informe["sin_respuesta"] = [i for i in informe["pedidos"]
                                if i not in puestos and i not in informe["sin_pieza"]]
    informe["sin_sitio"] = ins["sin_sitio"]
    if not puestos:
        # NADA SE INSERTÓ —ninguna pieza pasó, o las órdenes no tenían EFECTOS
        # donde ir—: el estudio sale como estaba y el aviso no presume de haber
        # completado nada (antes decía «ESTUDIO COMPLETADO» con la lista vacía).
        informe["estado"] = "sin_piezas" if informe["sin_respuesta"] else "sin_pieza"
        return estudio, informe
    # LOS REGISTROS QUE LA REPARACIÓN TRAE AL ESTUDIO. `revisar` ya corrió
    # sobre el estudio de antes; lo que se cita de nuevo (el registro que la
    # parte invocó) se devuelve para que `_completar_estudio` lo coteje con el
    # material, como `revisar` haría.
    try:
        import fase6_estudio as _f6r
        informe["registros_nuevos"] = sorted(set(_f6r._RX_REGISTRO_CITA.findall(nuevo))
                                             - set(_f6r._RX_REGISTRO_CITA.findall(estudio or "")))
    except Exception:
        pass
    informe.update(estado="ok", parrafos=ins["parrafos"], efectos=ins["efectos"])
    return nuevo, informe


def aviso_reparacion(informe: dict, segs: list) -> str:
    """El aviso VISIBLE de la reparación: qué se añadió (el secretario tiene
    que saber que la máquina escribió otra vez) o que no se pudo."""
    if not informe or informe.get("estado") == "nada":
        return ""
    por_id = {str(_get(s, "id")): s for s in (segs or [])}

    def _quien(ids):
        t = " ".join(str(_get(por_id.get(ids[0]) or {}, "texto") or "").split()[:14])
        return f"{', '.join(ids)}" + (f" («{t}…»)" if t else "")
    # LO QUE QUEDÓ POR HACER se dice siempre (revisión adversarial,
    # 26-sep-2026): el argumento que el modelo no contestó, el que pasó del
    # tope y el que dejó sin pieza por el mayor beneficio no pueden quedar
    # escondidos detrás de un «completado».
    _resto = []
    _sp = [_quien([i]) for i in informe.get("sin_pieza") or []]
    if _sp:
        _resto.append("la segunda revisión juzgó que la concesión ya da lo que pedían, y los "
                      "dejó con su declaración: " + "; ".join(_sp[:6]))
    _sr = [_quien([i]) for i in (informe.get("sin_respuesta") or []) + (informe.get("no_pedidos") or [])]
    if _sr and informe.get("estado") in ("ok", "sin_pieza"):
        _resto.append("quedaron sin completar " + "; ".join(_sr[:6])
                      + ("…" if len(_sr) > 6 else ""))
    if informe.get("estado") == "ok":
        pz = [_quien(x["ids"]) for x in informe.get("parrafos") or []]
        ef = [_quien(x["ids"]) for x in informe.get("efectos") or []]
        trozos = []
        if pz:
            trozos.append("se añadió su respuesta a " + "; ".join(pz[:6]))
        if ef:
            trozos.append("se nombró en los EFECTOS, entre lo que la responsable deberá examinar, "
                          + "; ".join(ef[:6]))
        return ("ESTUDIO COMPLETADO: el estudio despachaba argumentos con una declaración de "
                "sin materia sin que su dato quedara en los efectos ni en otra respuesta; "
                + " y ".join(trozos) + ("; " + "; ".join(_resto) if _resto else "")
                + ". Revise esos pasajes antes de firmar: los escribió una segunda llamada, "
                "con su criterio.")
    if informe.get("estado") == "sin_pieza":
        return ("REVISE LA DECLARACIÓN DE SIN MATERIA: el estudio despacha argumentos sin que su "
                "dato quede en los efectos ni en otra respuesta; " + "; ".join(_resto)
                + ". No se añadió nada: compruebe que la concesión de verdad los deja sin nada "
                "que resolver.")
    faltan = [_quien([i]) for i in (informe.get("pedidos") or []) + (informe.get("no_pedidos") or [])]
    por_que = {"vencio": "la llamada venció", "fallo": "la llamada falló",
               "sin_piezas": "ninguna pieza pasó las comprobaciones"}.get(informe.get("estado"), "no se pudo")
    return ("ESTUDIO SIN COMPLETAR (" + por_que + "): el estudio despacha con una declaración de "
            "sin materia, sin que su dato quede en los efectos ni en otra respuesta, "
            + "; ".join(faltan[:6]) + ("…" if len(faltan) > 6 else "")
            + ". Compruebe si la concesión los alcanza; si no, contéstelos o nómbrelos en los "
            "efectos.")
