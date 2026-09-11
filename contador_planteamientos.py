# -*- coding: utf-8 -*-
"""Contador determinista de planteamientos (conceptos de violación / agravios).

Sustituye a rubricas_de_conceptos()/conceptos_sin_resumir() de
fases123_pipeline.py. Compatible con Python 3.9+.

Principio: NO se cuenta «ordinal pegado a la palabra CONCEPTO» (63% del corpus
real no rotula así) ni «ordinales a principio de línea» (el escrito transcribe
la sentencia y sus resolutivos también empiezan por PRIMERO). Se cuenta una
CADENA: ordinales 1,2,3…N, en orden creciente, sin saltos, dentro de la sección
de conceptos, descartando enumeraciones comprimidas (lo transcrito) y el
petitorio final.
"""
import re
import unicodedata

# ───────────────────────────── constantes calibradas ────────────────────────
# Distancia por debajo de la cual DOS rúbricas consecutivas no pueden ser dos
# planteamientos distintos: son una enumeración transcrita (resolutivos,
# considerandos, petitorios). Medido sobre los 7 escritos del taller: la
# separación real más corta entre dos conceptos seguidos es 1.910 caracteres
# (406/2025) y 2.385 (393/2025); las enumeraciones transcritas van de 12 a
# 1.237. Se enmascara sólo si hay TRES o más seguidas así de juntas, y el
# tramo real más corto de TRES seguidas del corpus mide 4.399.
RACHA_JUNTAS = 2500
RACHA_MINIMA = 3

# A qué distancia máxima de la cabecera de sección puede arrancar el primer
# planteamiento. Medido: 24, 153, 172, 24 y 9 caracteres en los cinco escritos
# del taller que rotulan con cabecera.
PEGADA_A_CABECERA = 1200

# ───────────────────────────── ordinales ────────────────────────────────────
_UNIDAD = [
    ("PRIMER", 1), ("SEGUND", 2), ("TERCER", 3), ("CUART", 4), ("QUINT", 5),
    ("SEXT", 6), ("SEPTIM", 7), ("OCTAV", 8), ("NOVEN", 9), ("DECIM", 10),
]
# valor -> lista de grafías (sin acento, mayúscula), más larga primero
_ORDINALES = {}


def _mete(valor, *formas):
    _ORDINALES.setdefault(valor, [])
    for f in formas:
        if f not in _ORDINALES[valor]:
            _ORDINALES[valor].append(f)


def _flex(raiz):
    """PRIMER -> PRIMER, PRIMERO, PRIMERA ; SEGUND -> SEGUNDO, SEGUNDA."""
    if raiz in ("PRIMER", "TERCER"):
        return [raiz + "O", raiz + "A", raiz]
    return [raiz + "O", raiz + "A"]


for _raiz, _v in _UNIDAD:
    _mete(_v, *_flex(_raiz))
_mete(11, "UNDECIMO", "UNDECIMA", "DECIMOPRIMERO", "DECIMOPRIMERA",
      "DECIMOPRIMER", "DECIMO PRIMERO", "DECIMO PRIMERA", "DECIMO PRIMER")
_mete(12, "DUODECIMO", "DUODECIMA", "DECIMOSEGUNDO", "DECIMOSEGUNDA",
      "DECIMO SEGUNDO", "DECIMO SEGUNDA")
for _raiz, _v in _UNIDAD[2:]:           # 13..19  (TERCER..NOVEN)
    for _f in _flex(_raiz):
        _mete(10 + _v, "DECIMO" + _f, "DECIMO " + _f)
for _dec, _pref in ((20, "VIGESIM"), (30, "TRIGESIM")):
    _mete(_dec, _pref + "O", _pref + "A")
    for _raiz, _v in _UNIDAD[:9]:       # 21..29 / 31..39
        for _f in _flex(_raiz):
            _mete(_dec + _v, _pref + "O" + _f, _pref + "O " + _f)

_ALT_ORD = sorted(
    ((g, v) for v, gs in _ORDINALES.items() for g in gs),
    key=lambda x: -len(x[0]))
_RX_ORD_TXT = "|".join(re.escape(g).replace(r"\ ", r"\s+") for g, _ in _ALT_ORD)
_VALOR_DE = {g: v for g, v in _ALT_ORD}

_ROMANOS = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
            "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13,
            "XIV": 14, "XV": 15, "XVI": 16, "XVII": 17, "XVIII": 18,
            "XIX": 19, "XX": 20}

# ───────────────────────────── el ordinal que NO rotula nada ────────────────
# «SEGUNDO TRIBUNAL COLEGIADO», «CUARTO CIRCUITO», «Primera Sala», «Juzgado
# Quinto de Primera Instancia», «segundo párrafo», «Décima Época». Son el ruido
# dominante del corpus: una demanda cita decenas de tesis y cada una trae el
# nombre de su tribunal.
_SUSTANTIVO_AJENO = (
    r"TRIBUNAL(?:ES)?|SALA[SD]?|CIRCUITO|JUZGADO|DISTRITO|INSTANCIA|REGION|"
    r"TURNO|SECCION|EPOCA|PARRAFO|PARRAFOS|TERMINO|LUGAR|GRADO|INTERESAD[OA]S?|"
    r"PERJUDICAD[OA]S?|COLEGIAD[OA]S?|UNITARI[OA]S?|SUPLENTE|VISITADURIA|"
    r"FRACCION|APARTADO|TOMO|VOLUMEN|LIBRO|INSTANCIAS|PENAL|CIVIL|LABORAL|"
    r"ADMINISTRATIV[OA]|SUCURSAL|PISO|DIA|SEMANA|MES|ANO|VEZ|PUNTO|"
    r"TRIBUNALES|MINISTR[OA]S?|MAGISTRAD[OA]S?|SECRETARI[OA]S?"
)
_RX_AJENO = re.compile(r"^\s*(?:%s)\b" % _SUSTANTIVO_AJENO)

# ───────────────────────────── cabeceras de sección ─────────────────────────
_RX_CABECERA = re.compile(
    r"(?m)^[^\S\n]{0,10}"
    r"(?:(?:CAPITULO|APARTADO|TITULO)\s+[A-Z0-9]{1,10}[\s.\-–—)]{0,4})?"
    r"(?:[IVXLC]{1,6}|[0-9]{1,2}|[A-Z])?[\s.\-–—)]{0,5}"
    r"(CONCEPTOS?\s+DE\s+VIOLACION(?:ES)?|AGRAVIOS?|"
    r"MOTIVOS?\s+DE\s+INCONFORMIDAD|CONCEPTOS?\s+DE\s+ANULACION|"
    r"CONCEPTOS?\s+DE\s+IMPUGNACION)"
    r"[^\S\n]*[.:\-–—]?[^\S\n]*$")

# ───────────────────────────── el petitorio final ───────────────────────────
_RX_PETICION = re.compile(
    r"(?:ATENTAMENTE\s+)?(?:PIDO|SOLICITO|SUPLICO|PIDE|SOLICITA)\b[^.]{0,80}?"
    r"(?:SE\s+SIRVA|LO\s+SIGUIENTE|SIGUIENTE\s*:)|"
    r"PUNTOS?\s+PETITORIOS?|"
    r"POR\s+LO\s+(?:ANTES\s+|ANTERIORMENTE\s+)?EXPUESTO\s+Y\s+FUNDADO")

# ───────────────────────────── rúbrica explícita ────────────────────────────
_PALABRA = r"(?:CONCEPTOS?\s+DE\s+VIOLACION|AGRAVIOS?|CONCEPTOS?\s+DE\s+IMPUGNACION)"


def _pelar(texto):
    """Mayúsculas sin acento CONSERVANDO LOS DESPLAZAMIENTOS uno a uno.

    unicodedata.normalize() sobre el texto entero cambia la longitud y
    desalinea todas las posiciones; aquí se normaliza carácter a carácter.
    """
    fuera = []
    for c in texto:
        base = "".join(x for x in unicodedata.normalize("NFKD", c)
                       if not unicodedata.combining(x))
        fuera.append((base[:1] or c).upper())
    return "".join(fuera)


# principio de línea, un inciso opcional delante —a) 1.- IX .- i)—, el ordinal,
# y un separador: punto, «.-», dos puntos, raya, paréntesis… o la palabra
# CONCEPTO/AGRAVIO. Sin separador no es rótulo: «segundo término» no rotula.
_RX_CAND = re.compile(
    r"(?m)^[^\S\n]{0,12}"
    r"(?:(?P<inciso>[a-zA-Z0-9]{1,3}|[ivxIVX]{1,4})[\s]{0,2}[.\-–—)][^\S\n]{0,3})?"
    r"(?P<ord>%s|\d{1,2}|[IVXL]{1,6})"
    r"(?P<sep>[^\S\n]{0,3}(?:[.:)\-–—]+|\b(?=%s)))" % (_RX_ORD_TXT, _PALABRA))


class Rubrica(object):
    __slots__ = ("ini", "valor", "texto", "explicita", "alfabeto")

    def __init__(self, ini, valor, texto, explicita, alfabeto="palabra"):
        self.ini, self.valor = ini, valor
        self.texto, self.explicita = texto, explicita
        self.alfabeto = alfabeto

    def __repr__(self):
        return "<%d @%d %r%s>" % (self.valor, self.ini, self.texto,
                                  " EXPL" if self.explicita else "")


def _candidatas(plano):
    fuera = []
    for m in _RX_CAND.finditer(plano):
        cru = " ".join(m.group("ord").split()).upper()
        alfabeto = "palabra"
        if cru.isdigit():
            valor, alfabeto = int(cru), "digito"
            if not (1 <= valor <= 40):
                continue
        elif cru in _ROMANOS:
            valor, alfabeto = _ROMANOS[cru], "romano"
        else:
            valor = _VALOR_DE.get(cru)
            if valor is None:
                continue
        resto = plano[m.end("ord"):m.end("ord") + 40]
        if _RX_AJENO.match(resto):
            continue
        explicita = re.match(r"[\s.:)\-–—]{0,4}%s" % _PALABRA, resto) is not None
        fuera.append(Rubrica(m.start("ord"), valor,
                             " ".join(m.group(0).split()).strip(" .-–—):"),
                             explicita, alfabeto))
    return fuera


def _enmascarar_rachas(cands):
    """Quita las enumeraciones comprimidas: lo transcrito y los petitorios.

    Una racha es una sucesión de rúbricas con ordinal CRECIENTE y separadas por
    menos de RACHA_JUNTAS caracteres. Con RACHA_MINIMA o más miembros no puede
    ser una lista de planteamientos: nadie escribe tres conceptos de violación
    en 2.500 caracteres.
    """
    if not cands:
        return [], []
    exentas = [c for c in cands if c.explicita]
    cands = [c for c in cands if not c.explicita]
    if not cands:
        return exentas, []
    # Las rachas se agrupan DENTRO de cada alfabeto. Un capítulo preliminar
    # numerado en romanos —«I .- NOMBRE Y DOMICILIO DEL QUEJOSO»— no continúa
    # un ordinal en letra, y mezclarlos estrechaba el margen del umbral.
    rachas = []
    for alf in ("palabra", "digito", "romano"):
        grupo = [c for c in cands if c.alfabeto == alf]
        if not grupo:
            continue
        actual = [grupo[0]]
        for c in grupo[1:]:
            ant = actual[-1]
            if c.ini - ant.ini <= RACHA_JUNTAS and c.valor == ant.valor + 1:
                actual.append(c)
            else:
                rachas.append(actual)
                actual = [c]
        rachas.append(actual)
    vivas, muertas = [], []
    for r in rachas:
        (muertas if len(r) >= RACHA_MINIMA else vivas).extend(r)
    vivas.extend(exentas)
    vivas.sort(key=lambda c: c.ini)
    return vivas, muertas


def _corte_petitorio(plano, cands, desde):
    """Dónde acaba el cuerpo del escrito y empieza «pido se sirva…».

    No basta con buscar la frase: los conceptos del 406/2025 terminan cada uno
    con «solicito se me conceda el Amparo». El petitorio se reconoce porque la
    frase va seguida de un REARRANQUE de la numeración (vuelve a PRIMERO o a
    ÚNICO) estando la cadena ya empezada.
    """
    mejor = len(plano)
    for c in cands:
        if c.ini <= desde or c.valor != 1:
            continue
        previo = plano[max(0, c.ini - 400):c.ini]
        if _RX_PETICION.search(previo):
            mejor = min(mejor, c.ini)
    return mejor


def _cadena(cands, desde, hasta):
    """1, 2, 3… sin saltos, en orden, dentro de [desde, hasta)."""
    utiles = [c for c in cands if desde <= c.ini < hasta]
    if not utiles:
        return []
    inicio = next((c for c in utiles if c.valor == 1), None)
    if inicio is None:
        return []
    cadena, pos = [inicio], inicio.ini
    while True:
        quiero = cadena[-1].valor + 1
        sig = next((c for c in utiles if c.ini > pos and c.valor == quiero), None)
        if sig is None:
            break
        cadena.append(sig)
        pos = sig.ini
    return cadena


def planteamientos(texto, etiqueta_recurso=False):
    """Devuelve dict con la cuenta, los rótulos, los cortes y el porqué."""
    texto = texto or ""
    plano = _pelar(texto)
    cands = _candidatas(plano)
    vivas, muertas = _enmascarar_rachas(cands)

    cabeceras = [m.start(1) for m in _RX_CABECERA.finditer(plano)]

    # ── vía A: rúbricas explícitas (ordinal PEGADO a CONCEPTO/AGRAVIO).
    # Se autoidentifican: no necesitan cabecera. Sólo valen si forman cadena
    # desde el primero — así el «TERCER CONCEPTO» suelto del 91/2025, que sale
    # de una sentencia transcrita, no cuela.
    expl = [c for c in vivas if c.explicita]
    via_a = _cadena(expl, 0, _corte_petitorio(plano, cands, 0))
    if len(via_a) >= 2:
        return _empaquetar(texto, via_a, "explicita", cabeceras, muertas, cands)

    # ── vía B: cabecera de sección + ordinales a secas.
    #
    # LA CABECERA SE ELIGE POR CERCANÍA, NO POR LONGITUD DE CADENA. El escrito
    # del 91/2025 lleva la palabra «agravios» al final de una línea corrida
    # —un salto de OCR la deja sola y parece un rótulo— 72.000 caracteres
    # antes del capítulo de agravios de verdad; desde ahí la cadena más larga
    # es el capítulo de PROCEDENCIA (PRIMERA, SEGUNDA… en femenino) y da 5
    # donde hay 3. Medido sobre los cinco escritos que rotulan con cabecera,
    # el primer planteamiento arranca entre 24 y 172 caracteres después de
    # ella; lo que arranca 10.000 caracteres más allá no cuelga de esa
    # cabecera.
    # UN SOLO PLANTEAMIENTO ROTULADO «ÚNICO» ES UNO, NO NINGUNO. Sin esto el
    # escrito que trae un agravio único cae en «no_contado» y pierde la
    # comprobación entera.
    unicos = [c for c in _candidatas_unico(plano) if
              any(0 <= c.ini - cab <= PEGADA_A_CABECERA for cab in cabeceras)]
    mejor, usada, mejor_d = [], None, None
    for cab in cabeceras:
        tope = _corte_petitorio(plano, cands, cab)
        for alfabeto in ("palabra", "digito", "romano"):
            cad = _cadena([c for c in vivas if c.alfabeto == alfabeto], cab, tope)
            if not cad:
                continue
            d = cad[0].ini - cab
            if d > PEGADA_A_CABECERA:
                continue
            if (len(cad), -d) > (len(mejor), -(mejor_d if mejor_d is not None else 10 ** 9)):
                mejor, usada, mejor_d = cad, cab, d
            break
    if mejor:
        return _empaquetar(texto, mejor, "cabecera@%s(+%d)" % (usada, mejor_d),
                           cabeceras, muertas, cands)
    if via_a:
        return _empaquetar(texto, via_a, "explicita-corta", cabeceras,
                           muertas, cands)
    if len(unicos) == 1:
        return _empaquetar(texto, unicos, "unico", cabeceras, muertas, cands)
    return _empaquetar(texto, [], "sin-cadena", cabeceras, muertas, cands)


_RX_UNICO = re.compile(
    r"(?m)^[^\S\n]{0,12}(?P<ord>UNICO|UNICA)"
    r"[^\S\n]{0,3}(?:[.:)\-–—]+|\b(?=%s))" % _PALABRA)


def _candidatas_unico(plano):
    return [Rubrica(m.start("ord"), 1, m.group("ord"), False)
            for m in _RX_UNICO.finditer(plano)]


def _estado(n, via):
    # TRES estados, no dos. «no_contado» no es «no falta nada»: es «no lo sé»,
    # y tiene que verse distinto o la capa vuelve a estar apagada en silencio.
    return "contado" if n >= 1 else "no_contado"


def _empaquetar(texto, cadena, via, cabeceras, muertas, cands):
    tramos = []
    for i, c in enumerate(cadena):
        fin = cadena[i + 1].ini if i + 1 < len(cadena) else len(texto)
        tramos.append((c.ini, fin))
    return {
        "estado": _estado(len(cadena), via),
        "n": len(cadena),
        "rubricas": [c.texto for c in cadena],
        "valores": [c.valor for c in cadena],
        "offsets": [c.ini for c in cadena],
        "tramos": tramos,
        "via": via,
        "cabeceras": cabeceras,
        "descartadas_por_racha": len(muertas),
        "candidatas": len(cands),
    }


# ───────────────────────── ¿está cada uno en el resumen? ────────────────────
# DOS señales, y ninguna de ellas es «parecerse». Medí la huella léxica —las
# palabras que sólo salen en ese planteamiento— contra los siete resúmenes
# reales y NO separa: los conceptos que sí están resumidos puntúan entre 0,000
# y 0,143, y los cuatro que faltan en el 393/2025 puntúan entre 0,000 y 0,071.
# Se solapan enteros. Un resumen parafrasea; no reutiliza vocabulario. Ese
# camino está cerrado y no hay que volver a abrirlo.
#
# Lo que sí separa, medido sobre los siete:
#   (1) el ORDINAL nombrado — seis de siete resúmenes nombran {1..N} completo;
#   (2) el número de APARTADOS que el resumen produjo, que se cuenta solo
#       porque cada uno cierra con una marca de anclaje [[p.X §Y]].
# Y hacen falta las dos, porque se corrigen entre sí: el 888/2026 mete el
# segundo agravio en un párrafo que empieza «Finalmente» y NUNCA escribe
# «segundo» —la señal (1) lo acusaría— pero produce 3 apartados para 2
# agravios, y la señal (2) lo absuelve.

_RX_MARCA = re.compile(r"\[\[\s*(?:P|PAG|PAGINA)?\s*\.?\s*\d", re.I)


def _grafias(valor):
    return [g.replace(" ", r"\s+") for g in _ORDINALES.get(valor, [])]


def _nombrados(resumen):
    """Qué ordinales nombra el resumen. Longitud primero, y un ordinal simple
    no cuenta si lo que hay es un compuesto: «décimo» dentro de «décimo
    cuarto» no acredita el DÉCIMO. Ése es el fallo que hoy da por resumido un
    concepto que no existe."""
    res = _pelar(resumen or "")
    fuera = set()
    for valor in sorted(_ORDINALES, reverse=True):
        for g in _grafias(valor):
            rx = r"\b%s\b" % g
            if valor <= 10:
                rx += r"(?!\s+(?:%s))" % "|".join(
                    g for v in range(1, 10) for g in _grafias(v))
            if re.search(rx, res):
                fuera.add(valor)
                break
    return fuera


def apartados_del_resumen(resumen):
    """Cuántos planteamientos separó el resumen, contados por su estructura.

    La marca de anclaje es el contador más fiable que hay: el prompt obliga a
    cerrar cada apartado con [[p.X §Y]]. Si no hubiera marcas se cae a los
    párrafos con cuerpo menos la bisagra de entrada.
    """
    marcas = len(_RX_MARCA.findall(resumen or ""))
    cuerpos = [p for p in re.split(r"\n\s*\n", (resumen or "").strip())
               if len(p.strip()) > 60]
    parrafos = max(0, len(cuerpos) - 1)   # menos la bisagra de entrada
    return max(marcas, parrafos), ("marcas=%d parrafos=%d" % (marcas, parrafos))


def sin_resumir(texto_fuente, resumen, info=None):
    """Los planteamientos que el escrito trae y el resumen NO trajo.

    Devuelve TRES cosas distintas, no una lista:
      faltan  — hay que volver a pedirlos: el resumen produjo menos apartados
                que planteamientos tiene el escrito. Dispara el reintento.
      sin_rotular — el resumen tiene material de sobra pero no los nombra.
                Es aviso al secretario, NO reintento: gastar tres pasadas de
                modelo sobre algo que ya estaba es el otro modo de fallar.
      estado  — «no_contado» cuando no se pudo contar. No es «no falta nada».
    """
    info = info or planteamientos(texto_fuente)
    if info["estado"] == "no_contado":
        return {"estado": "no_contado", "faltan": [], "sin_rotular": [],
                "n": 0, "apartados": 0,
                "motivo": "no se localizó la sección de planteamientos; "
                          "el escrito no se pudo contar"}

    n = info["n"]
    apart, como = apartados_del_resumen(resumen)
    nombrados = _nombrados(resumen)
    faltan, sin_rotular, detalle = [], [], []
    for i, valor in enumerate(info["valores"]):
        rot = info["rubricas"][i]
        visto = valor in nombrados
        if visto:
            estado = "cubierto"
        elif apart >= n:
            estado = "sin_rotular"
            sin_rotular.append(rot)
        else:
            estado = "falta"
            faltan.append(rot)
        detalle.append({"valor": valor, "rubrica": rot,
                        "ordinal_en_resumen": visto, "estado": estado})
    # Todos rotulados pero menos apartados que planteamientos: el resumen se
    # quedó corto aunque los nombrara. Se avisa por número.
    corto = max(0, n - apart) if not faltan else 0
    return {"estado": "contado", "faltan": faltan, "sin_rotular": sin_rotular,
            "n": n, "apartados": apart, "conteo_por": como,
            "corto_por": corto, "detalle": detalle,
            "motivo": "cadena de %d por %s; el resumen trajo %d apartados"
                      % (n, info["via"], apart)}


def aviso(texto_fuente, resumen, info=None):
    """La frase que ve el secretario. Vacía si no hay nada que decir."""
    s = sin_resumir(texto_fuente, resumen, info)
    if s["estado"] == "no_contado":
        return ("NO SE PUDO CONTAR LOS PLANTEAMIENTOS de este escrito: no se "
                "localizó la sección de conceptos de violación o agravios. "
                "La comprobación de que están todos NO se hizo; revíselo a mano.")
    if s["faltan"]:
        return ("NO SE RESUMIERON TODOS LOS PLANTEAMIENTOS: el escrito tiene "
                "%d y el resumen trajo %d. Falta(n): %s."
                % (s["n"], s["apartados"], ", ".join(s["faltan"])))
    if s["corto_por"]:
        return ("El escrito tiene %d planteamientos y el resumen trajo %d "
                "apartados." % (s["n"], s["apartados"]))
    if s["sin_rotular"]:
        return ("El resumen no nombra por su ordinal %s, aunque trae "
                "apartados suficientes (%d para %d planteamientos): "
                "compruebe que ninguno se quedó fuera."
                % (", ".join(s["sin_rotular"]), s["apartados"], s["n"]))
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# ¿LLEGÓ EL ESCRITO ENTERO?
# ═══════════════════════════════════════════════════════════════════════════
# Ninguna cobertura vale si lo que se indexó ya venía mutilado: el contador
# leería el texto cortado y diría que está completo, que es exactamente lo que
# pasa hoy con el ADC 245/2024 —120.000 caracteres clavados, un tope redondo—.
#
# Sólo dos señales sobreviven la calibración contra los siete escritos reales:
# la longitud exactamente en un tope redondo, y que el acto y el escrito sean
# el MISMO documento. Ambas dan 0 falsos positivos sobre los seis sanos y
# cazan el amputado. La regla que se probó primero —«no termina en signo de
# cierre»— acusaba a cuatro de los seis sanos y se tiró: una comprobación que
# acusa al trabajo correcto es la comprobación que está mal.
_TOPES_REDONDOS = (100_000, 120_000, 150_000, 200_000, 250_000, 300_000,
                   500_000, 600_000)


def sospecha_de_amputacion(texto_acto: str, texto_escrito: str) -> str:
    """El aviso si el escrito llegó cortado, o cadena vacía."""
    import hashlib
    n = len(texto_escrito or "")
    if n in _TOPES_REDONDOS:
        return (f"EL ESCRITO DE LA PARTE MIDE EXACTAMENTE {n:,} CARACTERES, que "
                f"es un tope redondo: casi con seguridad llegó cortado y lo que "
                f"falta no se ha leído. La comprobación de que estén todos los "
                f"planteamientos se hace sobre lo que llegó, así que dirá que "
                f"no falta ninguno aunque falten. Vuelve a subir el escrito.")
    if texto_acto and texto_escrito and (
            hashlib.sha1(texto_acto.encode("utf-8", "ignore")).hexdigest()
            == hashlib.sha1(texto_escrito.encode("utf-8", "ignore")).hexdigest()):
        return ("EL ACTO RECLAMADO Y EL ESCRITO DE LA PARTE SON EL MISMO "
                "DOCUMENTO. Uno de los dos se subió en el sitio del otro: el "
                "resumen de los planteamientos estará resumiendo la sentencia, "
                "y el estudio contestará a lo que no se alegó.")
    return ""
