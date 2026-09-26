# -*- coding: utf-8 -*-
"""LA CONGRUENCIA INTERNA DEL ESTUDIO — la apertura con su calificación y los
EFECTOS que casan con el cuerpo.

POR QUÉ EXISTE. David, 26-sep-2026: «que no sea repetitivo y el proyecto mismo
sea inteligente (se le denomina congruencia interna) y que conteste todo lo
efectivamente planteado». En el ADC 642/2024 v4 cuatro apartados abrían «Sobre
el primer concepto de violación, en el que la parte quejosa sostiene que…» y
seguían con «Lo anterior, porque…» SIN la calificación en medio: el «Lo
anterior» no tenía antecedente.

LA CAUSA NO ERA (SÓLO) EL MODELO. En el estudio que escribió el modelo la
calificación SÍ estaba —«Es fundado.», «Resulta inoperante.»—, pero en su
propio renglón, y el compositor del .docx tira todo párrafo de menos de seis
palabras (`documento_generado._escribir_estudio`: el filtro que se come los
restos de ficha de una tesis). Medido en el banco p2-estandar (26-sep-2026):
la calificación suelta sale en 7 de 14 corridas v3 y 2 de 14 v4, 24 renglones,
y en NINGUNA de las 14 v1. La forma de la v2 lo pide así —«2) su calificación,
en la frase siguiente»— y el modelo lo leyó como «en el renglón siguiente».

LO QUE HAY AQUÍ, sin modelo:
  · `pegar_calificaciones`: la calificación suelta vuelve al párrafo que abre
    su apartado (y el «Sí.»/«No.» suelto de la moderna, a su respuesta). Es el
    texto del modelo, sin una palabra más: sólo deja de perderse.
  · `huecos`: el apartado que abre sin calificación y el «Lo anterior…» que
    no tiene calificación a la que referirse.
  · `reparar_aperturas`: lo anterior junto; y donde la apertura sigue sin
    calificación, añade la que el CRITERIO DEL SECRETARIO (o el plan) asigna a
    ese apartado, sin tocar nada más. Si no es unívoca, no escribe: avisa.
    Y dos candados (revisión adversarial): nunca la calificación contraria a
    la que el apartado ya razona, y sin la etiqueta del plan no adivina cuál
    de los conceptos de un problema fundado es el que lo funda. Simulado en el
    banco (quitando al modelo sus 24 calificaciones sueltas): añade 17, las 17
    en la dirección que el modelo había escrito; las 2 que no, las avisa.
    El pegado corre también ANTES de la reparación dirigida
    (`redactor_adelanto._congruencia_pegar`): ésta inserta tras el último
    párrafo que marca el concepto, y si es la apertura, la pieza quedaba
    entre la apertura y su calificación.
  · `efectos_sin_cubrir` (en sombra): el argumento fundado cuyo efecto no es
    liso y llano y que los EFECTOS no recogen.

SÓLO LA FAMILIA v2 (v2, v3, v4). La v1 no pasa por aquí: sus 14 corridas del
banco no tienen una sola calificación suelta, y no cambia ni una coma.
"""
from __future__ import annotations

import os
import re
import unicodedata

# ═══════════════════════════════════════════════════════════════════════════
# LA CALIFICACIÓN EN UNA FRASE
# ═══════════════════════════════════════════════════════════════════════════
# Las palabras de la calificación, con las formas de los engroses («lo
# infundado», «no le asiste la razón»). «debidamente fundada y motivada» no
# califica nada.
_CALIF = (r"(?:fundad[oa]s?|infundad[oa]s?|inoperantes?|ineficaz|ineficaces|inatendibles?|"
          r"innecesari[oa]s?|insuficientes?|sin\s+materia|desestim\w*)")
_RX_CALIF = re.compile(r"\b" + _CALIF + r"\b", re.I)
_RX_NO_CALIF = re.compile(
    r"(?:debida|indebida|suficiente|legal)mente\s+fundad\w*|fundad\w*\s+y\s+motivad\w*|"
    r"motivad\w*\s+y\s+fundad\w*|(?:mal|bien)\s+fundad\w*|"
    r"fundad[oa]s?\s+en\s+(?:el|la|los|las|lo\s+dispuesto)\s+(?:art|ley|c[óo]digo|precepto|numeral|"
    r"jurisprudencia|tesis|criterio|dispuesto)\w*|"
    r"temor\s+fundado|fundad[oa]s?\s+(?:sospecha|temor|motivo)", re.I)
_RX_RAZON = re.compile(
    r"\b(?:no\s+)?(?:le\s+|les\s+)?asiste\s+(?:la\s+)?raz[óo]n|\bcarece\w*\s+de\s+raz[óo]n|"
    r"\b(?:no\s+)?tiene\w*\s+raz[óo]n|\b(?:no\s+)?prospera\w*\b", re.I)
# QUIÉN HABLA. «…en el que la parte quejosa sostiene que el agravio era
# infundado» atribuye a la parte; no es la calificación de este tribunal.
_VERBOS_PARTE = (
    r"sostien[ee]n?|sostuv(?:o|ieron)|aduce[n]?|adujo|adujeron|alega[n]?|alegó|alegaron|"
    r"argumenta[n]?|argumentó|afirma[n]?|afirmó|plantea[n]?|planteó|refiere[n]?|refirió|"
    r"se[ñn]ala[n]?|se[ñn]aló|manifiesta[n]?|manifestó|cuestiona[n]?|cuestionó|"
    r"controvierte[n]?|controvirtió|reclama[n]?|reclamó|arguye[n]?|expresa[n]?|combate[n]?|"
    r"impugna[n]?|insiste[n]?|reprocha[n]?|asevera[n]?|esgrime[n]?|solicita[n]?|pretende[n]?|"
    r"indica[n]?|estima[n]?\s+que|considera[n]?\s+que|se\s+duele[n]?|dice[n]?")
_RX_ATRIBUYE = re.compile(r"\b(?:" + _VERBOS_PARTE + r")\b", re.I)
# LA CALIFICACIÓN DE ESTE TRIBUNAL va con un verbo propio delante —«es», «se
# considera», «resulta»— o con el artículo neutro de los engroses —«lo
# infundado»—, o abre la frase.
_RX_CALIF_PROPIA = re.compile(
    r"(?:\b(?:es|son|sea|sean|resulta\w*|se\s+(?:considera|estima|califica|declara|advierte|"
    r"determina|juzga|tiene\s+por)\w*|deviene\w*|devino|queda\w*|torna\w*|lo|estimarse|"
    r"considerarse|calificarse|declararse)\s+(?:\w+\s+){0,3}?" + _CALIF + r"\b)"
    r"|^\s*(?:[A-ZÁÉÍÓÚ]\w*\s+){0,1}" + _CALIF + r"\b", re.I)


def _plano(t: str) -> str:
    t = unicodedata.normalize("NFKD", str(t or "").lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def frases(p: str) -> list:
    """Las frases de un párrafo, sin partir «art.» ni «fr.» ni «1a./J.»."""
    t = re.sub(r"\b(art|arts|fr|fracc|núm|num|no|p|pág|párr|lic|c|ss|reg|cfr)\.", r"\1§", p or "",
               flags=re.I)
    t = re.sub(r"(\d)\.(\d)", r"\1§\2", t)
    t = re.sub(r"\b([0-9]{1,2}a|[A-Z])\.\s*/", r"\1§/", t)
    trozos = re.split(r"(?<=[.;:?!])\s+(?=[«“\"(¿A-ZÁÉÍÓÚÑ0-9])", t)
    return [x.replace("§", ".").strip() for x in trozos if x.strip()]


# LA ORACIÓN PRINCIPAL QUE VUELVE: una coma y, a lo sumo, un sujeto corto
# —«…, su estudio resulta innecesario», «…, el planteamiento es fundado»—.
_RX_VUELVE = re.compile(r",\s*(?:(?!que\b)\w+\s+){0,3}$", re.I)


def califica(frase: str) -> bool:
    """¿La frase trae la calificación de ESTE tribunal?"""
    t = _RX_NO_CALIF.sub(" ", frase or "")
    if _RX_RAZON.search(t):
        return True
    m_at = _RX_ATRIBUYE.search(t)
    for m in _RX_CALIF_PROPIA.finditer(t):
        # Detrás de una atribución —«sostiene que… es infundado»— es la parte
        # la que califica, salvo que medie un corte de la oración o que la
        # oración principal vuelva tras una coma: «…, en el que sostiene que
        # X, se considera infundado.» es la forma de la v1.
        if m_at and m_at.start() < m.start() and not re.search(r"[.;]", t[m_at.end():m.start()]) \
                and not _RX_VUELVE.search(t[:m.start()]):
            continue
        return True
    return False


def califica_parrafo(p: str, n_frases: int = 3) -> bool:
    return any(califica(f) for f in frases(p)[:n_frases])


# ═══════════════════════════════════════════════════════════════════════════
# 1 · LA CALIFICACIÓN SUELTA — el compositor la tiraba
# ═══════════════════════════════════════════════════════════════════════════
# «Es fundado.», «Se considera infundado.», «Resulta inoperante.», «Su
# estudio resulta innecesario.», «Son fundados pero insuficientes.». Corta,
# sola y sin nada más: ni el concepto ni la razón.
_RX_SOLA = re.compile(
    r"^(?:(?:dich[oa]|est[ea]|es[ea]|tal|el|la)\s+(?:argumento|planteamiento|concepto|agravio|motivo|"
    r"disenso)\s+)?"
    r"(?:(?:su\s+estudio\s+)?(?:se\s+(?:considera|estima|califica|declara)n?|es|son|resulta|resultan|"
    r"deviene|devienen|queda|quedan|lo\s+cual\s+es)\s+)?(?:(?:igualmente|tambi[ée]n|asimismo|"
    r"parcialmente|esencialmente|sustancialmente|en\s+parte)\s+)?" + _CALIF +
    r"(?:\s+(?:pero|aunque|y)\s+(?:en\s+parte\s+)?" + _CALIF + r")?"
    r"(?:\s+en\s+parte)?\s*[.;]?$", re.I)
# La respuesta suelta de la moderna: «Sí.», «No.», «Sí lo estaba.», «No lo
# era.». El compositor también la tiraba (menos de seis palabras).
_RX_SI_NO = re.compile(r"^(?:s[íi]|no)(?:\s*,?\s+(?:lo|la|los|las)?\s*\w+)?\s*[.!]?$", re.I)
# LAS QUE EL COMPOSITOR TIRA: menos de seis palabras (`_escribir_estudio`).
# Una calificación de seis o más ya llega al .docx en su párrafo, y `huecos`
# la acepta en el párrafo que sigue a la apertura: no se toca.
MAX_PALABRAS_SOLA = 5
_RX_PREGUNTA = re.compile(r"\?\s*$")
_RX_ROTULO = re.compile(r"^\s*(?:EFECTOS\b|ADVERTENCIAS?\b|[A-ZÁÉÍÓÚÑ\s]{6,}$)")


def _limpio(linea: str) -> str:
    import marcas as _mc
    return _mc.separar_marcas(linea or "")[0].strip()


def _ids(linea: str) -> list:
    import marcas as _mc
    return [x for _, _, v in _mc.marcas_en(linea or "") for x in v]


def es_sola(linea: str) -> bool:
    t = _limpio(linea)
    return bool(t) and len(t.split()) <= MAX_PALABRAS_SOLA and bool(_RX_SOLA.match(t))


def es_si_no(linea: str) -> bool:
    t = _limpio(linea)
    return bool(t) and len(t.split()) <= 4 and bool(_RX_SI_NO.match(t))


def _con_marca(texto: str, ids: list) -> str:
    """El renglón con los identificadores `ids` sumados a su marca del
    comienzo (o con una marca nueva delante)."""
    if not ids:
        return texto
    import marcas as _mc
    ms = _mc.marcas_en(texto)
    if ms and not texto[:ms[0][0]].strip():
        ini, fin, prev = ms[0]
        todos = prev + [x for x in ids if x not in prev]
        return texto[:ini] + "⟦" + " ".join(todos) + "⟧" + texto[fin:]
    return "⟦" + " ".join(ids) + "⟧ " + texto.lstrip()


def _sin_doble_blanco(lineas: list, k: int) -> None:
    """Al quitar el renglón `k`, los dos blancos que lo rodeaban quedan en uno."""
    if 0 < k < len(lineas) and not lineas[k - 1].strip() and not lineas[k].strip():
        del lineas[k]


def _corte_del_cuerpo(lineas: list) -> int:
    """El renglón donde acaba el cuerpo del estudio (empiezan los EFECTOS o
    las ADVERTENCIAS), o len(lineas)."""
    idx = [n for n, ln in enumerate(lineas) if _limpio(ln)]
    fin_c, _ = _partes([_limpio(lineas[n]) for n in idx])
    return idx[fin_c] if fin_c < len(idx) else len(lineas)


def pegar_calificaciones(estudio: str) -> tuple:
    """(estudio, [(renglón, «texto pegado», «atrás»|«adelante»)]).

    La calificación sola se pega al final del párrafo anterior —el que abre
    el apartado—; si el anterior es una pregunta o un rótulo, al comienzo del
    siguiente. El «Sí.»/«No.» solo, al comienzo del siguiente. Sus marcas, si
    las lleva, pasan al párrafo que la recibe. Nada más cambia.

    SÓLO EN EL CUERPO (revisión adversarial, 26-sep-2026): los EFECTOS son
    órdenes numeradas y las ADVERTENCIAS, notas al secretario; un renglón
    corto ahí no es la calificación de ningún apartado, y pegarlo a una orden
    la estropearía."""
    todas = (estudio or "").split("\n")
    corte = _corte_del_cuerpo(todas)
    lineas, cola = todas[:corte], todas[corte:]
    hechas = []
    k = 0
    while k < len(lineas):
        ln = lineas[k]
        sola, sino = es_sola(ln), es_si_no(ln)
        if not (sola or sino):
            k += 1
            continue
        txt, ids = _limpio(ln), _ids(ln)
        prev = next((j for j in range(k - 1, -1, -1) if lineas[j].strip()), None)
        sig = next((j for j in range(k + 1, len(lineas)) if lineas[j].strip()), None)
        prev_t = _limpio(lineas[prev]) if prev is not None else ""
        sig_t = _limpio(lineas[sig]) if sig is not None else ""
        atras = sola and prev is not None and prev_t and not _RX_PREGUNTA.search(prev_t) \
            and not _RX_ROTULO.match(prev_t) and not es_sola(lineas[prev])
        if atras:
            base = lineas[prev].rstrip()
            base = base[:-1] + "." if base[-1:] in ",;:" else (base if base[-1:] in ".!?»”\")…" else base + ".")
            lineas[prev] = _con_marca(base + " " + txt, ids)
            del lineas[k]
            _sin_doble_blanco(lineas, k)
            hechas.append((prev, txt, "atrás"))
            continue
        if sig is not None and sig_t and not _RX_ROTULO.match(sig_t):
            import marcas as _mc
            ms = _mc.marcas_en(lineas[sig])
            if ms and not lineas[sig][:ms[0][0]].strip():
                ini, fin, prev_ids = ms[0]
                resto = lineas[sig][fin:].lstrip()
                todos = prev_ids + [x for x in ids if x not in prev_ids]
                lineas[sig] = "⟦" + " ".join(todos) + "⟧ " + txt + " " + resto
            else:
                lineas[sig] = _con_marca(txt + " " + lineas[sig].lstrip(), ids)
            del lineas[k]
            _sin_doble_blanco(lineas, k)
            hechas.append((k, txt, "adelante"))
            continue
        k += 1
    return "\n".join(lineas + cola), hechas


# ═══════════════════════════════════════════════════════════════════════════
# 2 · LOS APARTADOS Y SUS HUECOS
# ═══════════════════════════════════════════════════════════════════════════
_RX_PALABRA = re.compile(r"\b(?:conceptos?|agravios?|motivos?\s+de\s+(?:disenso|inconformidad)|"
                         r"disensos?|inconformidad(?:es)?)\b", re.I)
_RX_UNICO = re.compile(r"\b[úu]nico\s+(?:concepto|agravio)|\b(?:concepto|agravio)(?:\s+de\s+violaci[óo]n)?"
                       r"\s+[úu]nico", re.I)
# «tercero interesado», «por concepto de» y «en primer lugar» no nombran
# ningún concepto (medido en el ADC 526/2024: `metricas_estudio`).
_RX_FALSO_ORDINAL = re.compile(
    r"\btercer[oa]s?\s+(?:interesad|extra[nñ]|perjudicad|llamad)\w*|"
    r"\b(?:por|en|bajo\s+el)\s+concepto\s+de\b|"
    r"\ben\s+(?:primer|segundo|tercer|cuarto)\s+(?:lugar|t[ée]rmino)\b", re.I)
_RX_METODO = re.compile(
    r"por\s+cuesti[óo]n\s+de\s+m[ée]todo|\bse\s+(?:analizar|estudiar|examinar|abordar)[áa]n?\b|"
    r"\ben\s+el\s+orden\s+(?:en\s+que|propuesto)|art[íi]culo\s+76\s+de\s+la\s+ley|"
    r"prelaci[óo]n\s+l[óo]gica|de\s+manera\s+conjunta|conjuntamente|\bse\s+seguir[áa]\b|"
    r"\bel\s+orden\s+(?:del\s+escrito|en\s+que\s+fueron|de\s+(?:los|su)\s+)", re.I)
_RX_REMITE = re.compile(
    r"\bal\s+(?:examinar|analizar|estudiar|contestar|resolver|dar\s+respuesta)\b|"
    r"\bcomo\s+(?:ya\s+)?se\s+(?:dijo|precis|expus|se[nñ]al|mencion|advirti|indic|estableci|"
    r"determin|vio|razon)\w*|\blo\s+razonado\b|\bse\s+vincula\b", re.I)
# «Lo anterior, porque…», «Lo anterior es así, ya que…», «Ello, toda vez
# que…»: la demostración de lo que se acaba de afirmar.
_RX_LO_ANTERIOR = re.compile(
    r"^\s*(?:Lo\s+anterior|Ello|Esto)\b\s*(?:(?:es|se\s+(?:estima|considera|sostiene|afirma|concluye|"
    r"determina))\s+as[ií]\s*)?,?\s*(?:porque|pues|ya\s+que|toda\s+vez\s+que|dado\s+que|en\s+virtud\s+de\s+que|"
    r"en\s+raz[óo]n\s+de\s+que|en\s+atenci[óo]n\s+a\s+que|debido\s+a\s+que|puesto\s+que|habida\s+cuenta|"
    r"atento\s+a\s+que|en\s+tanto\s+que)\b"
    r"|^\s*(?:Lo\s+anterior|Ello|Esto)\s+(?:es|se\s+(?:estima|considera))\s+as[ií]\b", re.I)
# «Lo anterior, pues aduce que…» NO demuestra nada de este tribunal: es el
# resumen de lo que alega la parte (5 de los 40 engroses lo traen en su
# síntesis de conceptos). Se reconoce por la atribución pegada al conector.
_RX_LO_ANTERIOR_AJENO = re.compile(
    r"^\s*(?:Lo\s+anterior|Ello|Esto)\b[^.;]{0,40}?\b(?:porque|pues|ya\s+que|toda\s+vez\s+que|dado\s+que)"
    r"\s*,?\s*(?:(?:a\s+su\s+(?:juicio|decir|parecer)|en\s+su\s+concepto|seg[úu]n\s+(?:su\s+dicho|"
    r"afirma|sostiene|refiere))|(?:(?!no\b)[\w,]+\s+){0,5}?(?:" + _VERBOS_PARTE + r")(?:(?<=que)|\s+que))\b", re.I)
_RX_PREGUNTA_NUM = re.compile(r"^\s*(?:\d{1,2}\s*[.)]\s*)?¿")


# EL ORDINAL PEGADO A LA PALABRA. `formato_sentencia.nombrados` acepta el
# ordinal a 90 caracteres de «concepto», que sirve para saber si un concepto
# se nombra en algún sitio, pero no para saber dónde ABRE un apartado: «los
# conceptos que… las cláusulas primera y décima tercera» abría uno (722/2025
# v1). Aquí el ordinal va junto al sustantivo: «el primer concepto», «los
# conceptos de violación segundo y tercero», «el tercer y cuarto agravios».
_ORD = {1: r"primer[oa]?|1[°º]", 2: r"segund[oa]s?|2[°º]", 3: r"tercer[oa]?s?|3[°º]",
        4: r"cuart[oa]s?|4[°º]", 5: r"quint[oa]s?|5[°º]", 6: r"sext[oa]s?|6[°º]",
        7: r"s[ée]ptim[oa]s?|7[°º]", 8: r"octav[oa]s?|8[°º]", 9: r"noven[oa]s?|9[°º]",
        10: r"d[ée]cim[oa]s?|10[°º]", 11: r"und[ée]cim[oa]s?", 12: r"duod[ée]cim[oa]s?",
        1.5: r"[úu]nic[oa]s?"}
_UN_ORD = r"(?:" + "|".join(_ORD.values()) + r")"
_SERIE = _UN_ORD + r"(?:\s*(?:,|y|e)\s*" + _UN_ORD + r")*"
_SUST = r"(?:conceptos?(?:\s+de\s+violaci[óo]n)?|agravios?|motivos?\s+de\s+(?:disenso|inconformidad))"
_RX_ORD_ANTES = re.compile(r"\b(" + _SERIE + r")\s+" + _SUST + r"\b", re.I)
_RX_ORD_DESPUES = re.compile(r"\b" + _SUST + r"\s+(" + _SERIE + r")\b", re.I)


def ordinales(frase: str) -> set:
    """Los conceptos que la frase nombra por su ordinal, pegado al sustantivo."""
    t = _RX_FALSO_ORDINAL.sub(" ", frase or "")
    s = set()
    for rx in (_RX_ORD_ANTES, _RX_ORD_DESPUES):
        for m in rx.finditer(t):
            for n, pat in _ORD.items():
                if re.search(r"\b(?:" + pat + r")(?!\w)", m.group(1), re.I):
                    s.add(1 if n == 1.5 else n)
    return s


def _es_rubro(p: str) -> bool:
    t = (p or "").strip()
    letras = [c for c in t[:120] if c.isalpha()]
    return bool(letras) and sum(c.isupper() for c in letras) / len(letras) > 0.6


def apartados(ps: list, fin: int = None) -> list:
    """[{i, fin, tipo: concepto|pregunta, ordinales}] de la parte `ps[:fin]`.

    Un apartado abre donde la PRIMERA frase de un párrafo nombra un concepto
    que ningún apartado anterior había abierto (la estándar y los engroses),
    o donde una pregunta abre (la moderna). No abre: el anuncio de método, la
    remisión, ni volver a nombrar un concepto ya abierto."""
    fin = len(ps) if fin is None else fin
    fuera, vistos = [], set()
    for i in range(fin):
        p = ps[i] or ""
        if _es_rubro(p):
            continue
        if _RX_PREGUNTA_NUM.match(p) and _RX_PREGUNTA.search(p.strip()):
            fuera.append({"i": i, "tipo": "pregunta", "ordinales": set()})
            continue
        fr = frases(p)
        f1 = fr[0] if fr else p
        o = ordinales(f1)
        if not o or _RX_METODO.search(f1) or _RX_REMITE.search(f1) or not (o - vistos):
            continue
        fuera.append({"i": i, "tipo": "concepto", "ordinales": o})
        vistos |= o
    for k, a in enumerate(fuera):
        a["fin"] = fuera[k + 1]["i"] if k + 1 < len(fuera) else fin
    return fuera


def _partes(ps: list) -> tuple:
    try:
        import exhaustivo as _ex
        return _ex.partes(ps)
    except Exception:
        n = len(ps)
        return n, n


def huecos(ps: list) -> list:
    """Los huecos de calificación del cuerpo: [{tipo, parrafo, apartado,
    ordinales, texto}]. `tipo`:
      · «apertura»: el apartado abre sin su calificación —ni en su párrafo ni
        en la primera frase del siguiente—;
      · «lo_anterior»: un «Lo anterior…» sin calificación en su primera frase
        y sin ninguna antes en su apartado;
      · «respuesta»: la moderna, la respuesta que no califica en sus dos
        primeras frases."""
    ps = [p or "" for p in ps]
    fin, _ = _partes(ps)
    fuera = []
    for a in apartados(ps, fin):
        i, f = a["i"], a["fin"]
        if a["tipo"] == "pregunta":
            if i + 1 < f and not califica_parrafo(ps[i + 1], 2):
                fuera.append({"tipo": "respuesta", "parrafo": i + 1, "apartado": i,
                              "ordinales": sorted(ordinales(" ".join(frases(ps[i + 1])[:2]))),
                              "texto": " ".join(ps[i + 1].split()[:24])})
            continue
        abre = califica_parrafo(ps[i]) or (i + 1 < f and califica_parrafo(ps[i + 1], 1))
        if not abre:
            fuera.append({"tipo": "apertura", "parrafo": i, "apartado": i,
                          "ordinales": sorted(a["ordinales"]), "texto": " ".join(ps[i].split()[:24])})
        visto = califica_parrafo(ps[i], 9)
        for j in range(i + 1, f):
            if _RX_LO_ANTERIOR.match(ps[j]) and not _RX_LO_ANTERIOR_AJENO.match(ps[j]) \
                    and not califica_parrafo(ps[j], 1) and not visto:
                fuera.append({"tipo": "lo_anterior", "parrafo": j, "apartado": i,
                              "ordinales": sorted(a["ordinales"]), "texto": " ".join(ps[j].split()[:24])})
            if califica_parrafo(ps[j], 9):
                visto = True
    return fuera


# ═══════════════════════════════════════════════════════════════════════════
# 3 · LA REPARACIÓN: LA CALIFICACIÓN QUE EL CRITERIO ASIGNA
# ═══════════════════════════════════════════════════════════════════════════
def _norm_sentido(s) -> str:
    return re.sub(r"\s+", "_", str(s or "").strip().lower())


def _get(o, k, defecto=None):
    if isinstance(o, dict):
        return o.get(k, defecto)
    return getattr(o, k, defecto)


def frase_calificacion(sentido: str, plural: bool = False) -> str:
    """La frase de la calificación, en la forma del oficio y concordada con
    el concepto (masculino: concepto, agravio, motivo). «» si no hay una."""
    s = _norm_sentido(sentido)
    if s in ("innecesario",):
        return "Su estudio resulta innecesario."
    if s == "sin_materia":
        return "Quedan sin materia." if plural else "Queda sin materia."
    import tipos_asunto as _ta
    if s not in _ta.CALIFICACIONES:
        return ""
    if plural:
        return f"Son {_ta.CALIFICACIONES[s][0]}."
    return "Es " + s.replace("fundado_insuficiente", "fundado pero insuficiente").replace("_", " ") + "."


# LA DIRECCIÓN DE LO QUE EL APARTADO YA RAZONA (revisión adversarial,
# 26-sep-2026). La reparación añade una calificación que el modelo no
# escribió; si el cuerpo del apartado razona la contraria —«es infundado»,
# «no le asiste la razón» donde se iba a añadir «Es fundado.»—, añadirla sería
# el defecto (2) del 642 hecho por la máquina: una apertura que niega su
# propia demostración. Se lee sólo en las frases que califican por cuenta de
# este tribunal (`califica`), sin los rubros de las tesis.
_RX_DIR_FAVOR = re.compile(
    r"\bfundad[oa]s?\b(?!\s*,?\s*(?:pero|aunque)\s+(?:\w+\s+)?(?:insuficien|inoperan|inefica))"
    r"|\b(?:le|les|la)\s+asiste\b|\basiste\s+(?:la\s+)?raz[óo]n|\btiene\w*\s+(?:la\s+)?raz[óo]n", re.I)
# «insuficiente» sólo califica pegado a «fundado pero…»: suelto es de la prueba
# («explicar por qué resulta insuficiente», 174/2026 v3), no del concepto.
_RX_DIR_CONTRA = re.compile(
    r"\b(?:infundad|inoperan|inefica|inatendib|desestim)\w*|\bcarece\w*\s+de\s+raz[óo]n"
    r"|\bfundad\w*\s*,?\s*(?:pero|aunque)\s+(?:\w+\s+)?insuficien", re.I)
_RX_NEGADO = re.compile(r"\bno\s+(?:\w+\s+){0,2}$", re.I)


def direcciones(parrafos_: list) -> set:
    """{«favor», «contra»}: las direcciones de las calificaciones propias que
    ya trae el texto. «no es fundado», «no le asiste la razón» van en contra."""
    fuera = set()
    for p in parrafos_ or []:
        for fr in frases(p):
            if _es_rubro(fr) or not califica(fr):
                continue
            t = _RX_NO_CALIF.sub(" ", fr)
            if _RX_DIR_CONTRA.search(t):
                fuera.add("contra")
            for m in _RX_DIR_FAVOR.finditer(t):
                fuera.add("contra" if _RX_NEGADO.search(t[max(0, m.start() - 24):m.start()]) else "favor")
    return fuera


def _clase(sentido: str) -> str:
    s = _norm_sentido(sentido)
    if not s or s in ("innecesario", "sin_materia", "cae_con_principal", "no_se_estudia",
                      "adhesivo_sin_materia"):
        return ""
    return "favor" if _prospera(s) else "contra"


def calificacion_de_apartado(ordinales_: list, criterios: list, problemas: list,
                             ids: list = None, plan: dict = None, cuerpo: list = None) -> tuple:
    """(sentido, de dónde) de un apartado, o («», motivo) si no es unívoco.

    Del CRITERIO: el sentido de los problemas que cubren los conceptos que el
    apartado nombra (el reparto de la línea CUBRE), cuando es uno solo. Con
    PLAN (v4) y marcas en el apartado, la etiqueta de sus argumentos manda si
    es una sola; si difieren (plan-4: dentro de un problema fundado cabe un
    argumento infundado), el apartado abre con el sentido de su problema
    siempre que alguno de sus argumentos lo lleve —es el que lo funda—.

    DOS CANDADOS (revisión adversarial, 26-sep-2026), con `cuerpo` = los
    párrafos del apartado:
      · SIN LA ETIQUETA DEL PLAN, UN PROBLEMA QUE PROSPERA Y CUBRE OTROS
        CONCEPTOS no dice cuál de ellos lo funda: dentro de él cabe un
        concepto entero infundado. Sólo se añade si el cuerpo ya razona en su
        favor; si no, se avisa.
      · NUNCA LA CONTRARIA DE LO QUE EL APARTADO RAZONA: si el cuerpo sólo
        califica en la otra dirección, no se escribe; se avisa."""
    try:
        import exhaustivo as _ex
        rep = _ex.reparto(criterios, problemas)
    except Exception:
        rep = []
    suyos = [(c, cub) for c, cub in rep if set(cub) & set(ordinales_ or []) and _get(c, "sentido")]
    ss = {_norm_sentido(_get(c, "sentido")) for c, _ in suyos}
    del_criterio = next(iter(ss)) if len(ss) == 1 else ""
    dirs = direcciones(cuerpo) if cuerpo is not None else None

    def _cabe(sentido: str, fuente: str) -> tuple:
        cl = _clase(sentido)
        if cl and dirs and cl not in dirs:
            return "", (f"el apartado ya razona en la dirección contraria a «{sentido}» (la de su "
                        f"{fuente}): la máquina no escribe una calificación que su demostración niega")
        return sentido, fuente

    if plan and ids:
        et = {_norm_sentido(s.get("etiqueta")) for s in plan.get("segmentos") or []
              if s.get("id") in set(ids) and s.get("etiqueta")}
        if len(et) == 1:
            return _cabe(next(iter(et)), "plan")
        if len(et) > 1:
            if del_criterio and del_criterio in et:
                return _cabe(del_criterio, "criterio")
            return "", "los argumentos del apartado llevan calificaciones distintas en el plan"
    if del_criterio:
        otros = set().union(*[set(cub) for _, cub in suyos]) - set(ordinales_ or [])
        if _clase(del_criterio) == "favor" and otros and not (dirs and "favor" in dirs):
            return "", ("el problema que lo cubre prospera y cubre también otros conceptos: sin la "
                        "etiqueta del plan no se sabe si éste es el que lo funda")
        return _cabe(del_criterio, "criterio")
    if not ss:
        return "", "ningún problema del criterio cubre ese concepto"
    return "", "los problemas que cubren ese concepto tienen sentidos distintos"


def _renglon_de(lineas: list, k_parrafo: int) -> int:
    """El renglón de `lineas` que es el párrafo `k_parrafo` del texto limpio
    (renglones no vacíos, sin las marcas solas en su renglón)."""
    k = -1
    for n, ln in enumerate(lineas):
        if not _limpio(ln):
            continue
        k += 1
        if k == k_parrafo:
            return n
    return -1


# LO QUE SE REPARA Y LO QUE SE AVISA — calibrado (26-sep-2026) contra los 40
# engroses reales del corpus Kingston (campo «oro», su Solución) y las 42
# corridas del banco p2-estandar (su texto entregado y su estudio crudo):
#   · «LO ANTERIOR» SIN ANTECEDENTE: 0 de 40 engroses —también sobre el
#     considerando entero, con su síntesis de conceptos: el «Lo anterior, pues
#     aduce que…» que resume a la parte (5 engroses) no cuenta—, 0 de 14
#     corridas v1; en el texto entregado (el .docx), 6 de 13 corridas v3 y 2
#     de 15 v4 (con la de entrega del 642/2024), además del 642/2024 v4 de
#     este encargo, y 0 después de pegar las calificaciones sueltas. Es el
#     defecto: se repara y, si no se puede, se avisa.
#   · APERTURA SIN CALIFICACIÓN, sola: acusa 13 de 40 engroses —el secretario
#     expone a veces el concepto en varios párrafos («Refiere que…», «Sostiene
#     que…») y califica después, sin «Lo anterior» huérfano—. VA EN SOMBRA: se
#     anota, no se repara ni se enseña.
#   · LA RESPUESTA DE LA MODERNA sin calificación: sin corridas modernas ni
#     engroses modernos con que medirla. En sombra.
REPARAR_APERTURAS = os.getenv("ESTUDIO_REPARAR_APERTURAS", "1") != "0"
VISIBLES = ("lo_anterior",)


def reparar_aperturas(estudio: str, criterios: list, problemas: list,
                      plan: dict = None) -> tuple:
    """(estudio, informe). Primero pega las calificaciones sueltas; luego, en
    cada apartado con un «Lo anterior…» que no tiene calificación a la que
    referirse, añade al final de su párrafo de apertura la que el criterio (o
    el plan) le asigna. Nada más cambia. Nunca lanza.

    informe: {pegadas, anadidas: [{apartado, parrafo, ordinales, sentido,
    fuente, frase}], sin_reparar: [{tipo, parrafo, ordinales, motivo,
    texto}], sombra: [huecos que no se reparan ni se enseñan], huecos_antes}."""
    informe = {"pegadas": [], "anadidas": [], "sin_reparar": [], "sombra": [],
               "huecos_antes": 0}
    try:
        import marcas as _mc
        informe["huecos_antes"] = len(huecos(_mc.parrafos(_mc.sin_marcas(estudio or ""))))
        nuevo, pegadas = pegar_calificaciones(estudio or "")
        informe["pegadas"] = [{"renglon": r, "texto": t, "hacia": h} for r, t, h in pegadas]
        lineas = nuevo.split("\n")
        limpio, mapa = _mc.separar_marcas(nuevo)
        ps = _mc.parrafos(limpio)
        por_parrafo = {}
        for sid, idxs in (mapa or {}).items():
            for i in idxs or []:
                por_parrafo.setdefault(int(i), []).append(sid)
        aps = {a["i"]: a for a in apartados(ps, _partes(ps)[0])}
        hs = huecos(ps)
        hechos = set()
        for h in hs:
            if h["tipo"] not in VISIBLES:
                informe["sombra"].append(h)
                continue
            if h["apartado"] in hechos:
                continue
            hechos.add(h["apartado"])
            if not REPARAR_APERTURAS:
                informe["sin_reparar"].append(dict(h, motivo="reparación apagada"))
                continue
            a = aps.get(h["apartado"]) or {}
            ids = []
            for j in range(h["apartado"], a.get("fin", h["apartado"] + 1)):
                ids += [x for x in por_parrafo.get(j, []) if x not in ids]
            sentido, fuente = calificacion_de_apartado(
                h["ordinales"], criterios, problemas, ids, plan,
                cuerpo=ps[h["apartado"]:a.get("fin", h["apartado"] + 1)])
            frase = frase_calificacion(sentido, plural=len(h["ordinales"]) > 1) if sentido else ""
            n = _renglon_de(lineas, h["apartado"])
            if not frase or n < 0:
                informe["sin_reparar"].append(dict(h, motivo=fuente if not sentido else "sin frase"))
                continue
            base = lineas[n].rstrip()
            base = base[:-1] + "." if base[-1:] in ",;:" else (base if base[-1:] in ".!?»”\")…" else base + ".")
            lineas[n] = base + " " + frase
            informe["anadidas"].append({"apartado": h["apartado"], "parrafo": h["parrafo"],
                                        "ordinales": h["ordinales"], "sentido": sentido,
                                        "fuente": fuente, "frase": frase})
        return "\n".join(lineas), informe
    except Exception as ex:
        informe["error"] = type(ex).__name__
        return estudio, informe


def aviso_aperturas(informe: dict, q1: str = "concepto de violación") -> str:
    """El aviso VISIBLE: la calificación que la máquina añadió (el secretario
    tiene que saberlo) y el «Lo anterior» que no pudo dejar con antecedente.
    Lo que sólo se pegó —la calificación que el modelo escribió en su propio
    renglón— no se avisa: es su texto, en su sitio. La sombra, tampoco."""
    if not informe:
        return ""
    import formato_sentencia as _fs

    def _quien(x):
        return _fs.cubre_en_texto(x.get("ordinales") or [], q1) or "un apartado"
    trozos = []
    an = informe.get("anadidas") or []
    if an:
        trozos.append("se añadió al abrir " + "; ".join(
            f"{_quien(x)} su calificación «{x['frase']}», la de su "
            f"{'plan' if x.get('fuente') == 'plan' else 'criterio'}" for x in an[:6]))
    sr = informe.get("sin_reparar") or []
    if sr:
        # LA DEMOSTRACIÓN QUE VA AL REVÉS se dice con su porqué (revisión
        # adversarial, 26-sep-2026): no es un hueco de forma, es una
        # incongruencia entre la calificación asignada y lo que se razona.
        trozos.append("no se pudo añadir en " + "; ".join(
            f"{_quien(x)} («{x['texto'][:90]}…»)"
            + (f", porque {x['motivo']}" if "contraria" in str(x.get("motivo") or "") else "")
            for x in sr[:6]))
    if not trozos:
        return ""
    return ("CALIFICACIÓN AL ABRIR: el estudio demostraba con «Lo anterior…» una calificación "
            "que no había dicho; " + " y ".join(trozos)
            + ". Revise que cada apartado diga su calificación antes de demostrarla.")


# ═══════════════════════════════════════════════════════════════════════════
# 4 · LOS EFECTOS CASAN CON EL CUERPO (en sombra)
# ═══════════════════════════════════════════════════════════════════════════
# Un argumento FUNDADO cuya consecuencia no es liso y llano deja algo que la
# responsable tiene que volver a hacer: o los EFECTOS lo recogen —su dato, un
# ancla propia o `UMBRAL_EFECTOS` de sus palabras, como en `exhaustivo`—, o
# recogen su PROBLEMA —la pregunta del problema en las palabras de un efecto,
# o el ordinal de su concepto—. Si ninguno, la concesión deja ese argumento
# sin consecuencia: incongruencia entre el cuerpo y los efectos.
UMBRAL_PROBLEMA = 0.35
EFECTOS_VISIBLE = os.getenv("ESTUDIO_EFECTOS_VISIBLE", "0") == "1"


def _prospera(s) -> bool:
    import tipos_asunto as _ta
    return bool(_norm_sentido(s)) and _ta.prospera(_norm_sentido(s))


def _cubre_problema(pregunta: str, efectos: str) -> float:
    try:
        import exhaustivo as _ex
        rp, re_ = _ex._raices(pregunta), _ex._raices(efectos)
    except Exception:
        return 0.0
    rp = {r for r in rp if not r.isdigit()}
    return round(len(rp & re_) / len(rp), 3) if rp else 0.0


def efectos_sin_cubrir(ps: list, segs: list, criterios: list, problemas: list,
                       plan: dict = None) -> list:
    """[{id|concepto, problema, cubre_dato, cubre_problema}] de los argumentos
    fundados (por el plan, o por el criterio de su problema) que los EFECTOS
    no recogen. [] si no hay efectos o son lisos y llanos."""
    import exhaustivo as _ex
    ps = list(ps or [])
    fin_c, fin_e = _ex.partes(ps)
    efectos = "\n".join(ps[fin_c:fin_e])
    if not efectos.strip() or _ex.es_lisa_y_llana(efectos):
        return []
    rep = _ex.reparto(criterios, problemas)
    efectos_ord = ordinales(efectos)
    fuera = []
    if segs:
        datos = _ex.Datos(segs)
        et_plan = {s.get("id"): s.get("etiqueta") for s in (plan or {}).get("segmentos") or []}
        for s in datos.segs:
            sid = str(_get(s, "id"))
            cs = _ex.criterios_del_segmento(s, rep)
            if sid in et_plan and et_plan[sid]:
                fundado = _prospera(et_plan[sid])
            else:
                fundado = bool(cs) and all(_prospera(_get(c, "sentido")) for c in cs)
            if not fundado:
                continue
            a, c = datos.cubre(sid, efectos)
            if a or c >= _ex.UMBRAL_EFECTOS:
                continue
            k = _ex._concepto(s)
            cp = max([_cubre_problema(str(_get(x, "problema") or ""), efectos) for x in cs] or [0.0])
            if cp >= UMBRAL_PROBLEMA or (k and k in efectos_ord):
                continue
            fuera.append({"id": sid, "concepto": k, "cubre_dato": c, "cubre_problema": cp})
        return fuera
    # Sin inventario (v2): por concepto, con el reparto.
    for c, cub in rep:
        if not _prospera(_get(c, "sentido")):
            continue
        cp = _cubre_problema(str(_get(c, "problema") or ""), efectos)
        if cp >= UMBRAL_PROBLEMA or (set(cub) & efectos_ord):
            continue
        fuera.append({"id": "", "concepto": sorted(cub), "problema": str(_get(c, "problema") or "")[:120],
                      "cubre_dato": None, "cubre_problema": cp})
    return fuera


# CALIBRADO (26-sep-2026). Las 42 corridas del banco p2-estandar con su
# criterio real: acusa 1 de 14 v1 (174/2026, por concepto), 2 de 14 v3 (las
# dos del 174/2026: 7 argumentos cada una —la doble jornada, el tercer
# inmueble…— que el reparto de la fase 3 cuelga del problema 1, fundado, y
# que los EFECTOS no nombran; son las mismas omisiones graves que el
# localizador vio y `exhaustivo` repara) y 2 de 14 v4 (un argumento cada una:
# 174/2026 C1.f y 43/2025 C2.e, éste con el 38 % de su dato en los efectos,
# al borde del umbral). En los engroses reales sólo 5 de 40 traen efectos no
# lisos reconocibles y no acusa a ninguno: muestra corta para enseñarlo. VA EN
# SOMBRA: `cobertura`/`congruencia` de la ficha, no la pantalla.
CALIBRACION_EFECTOS = ("banco p2-estandar: v1 1/14, v3 2/14 (174/2026, reparto), v4 2/14 "
                       "(1 argumento cada una); engroses: 0/5 con efectos no lisos")


def informe_sombra(informe: dict) -> dict:
    """Lo que va a la ficha y al «listo»: cifras e identificadores, sin el
    texto del estudio (higiene de registros)."""
    if not isinstance(informe, dict):
        return {}
    quita = ("texto",)
    return {
        # Lo pegado antes de la reparación dirigida (`redactor_adelanto.
        # _congruencia_pegar`) y lo pegado aquí, juntos.
        "pegadas": len(informe.get("pegadas") or []) + int(informe.get("pegadas_antes") or 0),
        "anadidas": [{k: v for k, v in x.items() if k not in quita} for x in informe.get("anadidas") or []],
        "sin_reparar": [{k: v for k, v in x.items() if k not in quita} for x in informe.get("sin_reparar") or []],
        "sombra": [{k: v for k, v in x.items() if k not in quita} for x in informe.get("sombra") or []],
        "huecos_antes": informe.get("huecos_antes", 0),
        **({"error": informe["error"]} if informe.get("error") else {}),
    }
