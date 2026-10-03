# -*- coding: utf-8 -*-
"""LA VERJA PROCESAL (3-oct-2026) — sin modelo.

POR QUÉ EXISTE. Auditoría de 81 proyectos generados (72 AD y 9 AR, del 26 de
septiembre al 3 de octubre de 2026): los diez de octubre traían al menos un
error grave en la parte procesal, y NINGUNO lo dijo al frente. La responsable
salía «TRIBUNAL EL SEIS» en la carátula mientras el resultando decía «la Segunda
Sala Civil…» (9 de 9 AD actuales); la existencia citaba como «expediente» el
número del toca; el CNPCF se citaba como supletorio en Querétaro (18 de 18); la
fecha del auto de Presidencia era «la que se advierte de las constancias» (76 de
81); el MP «omitió formular pedimento» sin que nadie lo hubiera leído (71 de 72);
una sentencia reclamada de enero de 2026 se impugnaba con una demanda de marzo de
2025 y el cómputo la declaraba «oportuna».

Con el modelo nuevo (`ficha_tramite` + `resultandos_por_tipo`) esos datos ya no
los escribe el modelo; pero una pieza que se equivoca por otro camino —una ficha
mal fundida, una plantilla del banco que no se tocó, un tribunal tecleado en
versales— volvería a firmarse en silencio. Esta verja revisa EL TEXTO YA
COMPUESTO (de la carátula al rótulo del Estudio, y los resolutivos), que es lo
único que importa, y lo que acusa sale al frente de los avisos.

LAS REGLAS (SPEC §3.3), cada una con su letra en `revisar_detalle`:
  (a) `*********` fuera de las fechas de la sesión, con el dato que falta;
  (b) fórmulas evasivas, el MP afirmado sin fuente, la oficialía inventada,
      el resultando de derechos sin artículos, «conforme a la ley del acto»;
  (c) fechas del V I S T O y los resultandos que no están en la ficha ni en el
      acto; la fecha del acto en los resolutivos distinta de la del acto;
  (d) la responsable/órgano escrita distinto del dato; sin entidad; el
      tribunal de dos formas, en versales o con artículo duplicado; el
      ponente de la carátula distinto del del turno o del último returno; el
      renglón RECURRENTE que no es quien recurre; nombres en versales y
      etiquetas de rol en la prosa; «X, en representación de X» y «X, por
      conducto de su X»; el órgano que el compositor descartó (ronda 4);
  (e) número del asunto (y la carátula sin él), etiqueta del encabezado
      colada, toca por expediente, existencia del AR por informe justificado;
  (f) concordancias («por el Jueza», «el Coordinadora», «el Legislatura», «el
      Actual Sala», «localizado» con la Sala, «legitimado» con la quejosa, «la
      quejoso», «en contra el», «con sede en la Santiago»); y las partes en la
      prosa (ronda 4): «por Conducto de Su», el colectivo sin artículo, el
      plural del resultando contra el singular de la legitimación;
  (g) supletorio que no corresponde a la sede (CFPC fuera de CDMX, CNPCF en ella);
  (h) LOPJF abrogada, AG 1/2023, «Pleno del Consejo» sin «otrora», el 84 en
      la sesión, el 101 en el turno fuera de la queja;
  (i) AR: el verbo del resultando contra `acto.resolvio`; el sobreseimiento
      que no se resuelve en el mixto;
  (j) fechas imposibles de la ficha y las del propio texto;
  (k) el adhesivo que consta y no se trata (resultando, considerando, resolutivo);
  (l) la clase del acto: «la sentencia reclamada» cuando lo reclamado es una
      resolución, un laudo o un auto (ronda 3);
  (m) la forma de notificación que dice la oportunidad contra la de la ficha
      (ronda 3), y la que afirma sin papel que la diga (ronda 5, F1);
  (n) el par de palabras repetido seguido («solicitud de solicitud de»,
      ronda 5);
  (o) un asunto relacionado que es el propio asunto (sexta ronda);
  (p) el artículo 64 de la LFPCA fuera del par amparo directo ↔ revisión
      fiscal (sexta ronda);
  (q) con la ficha nueva, los asuntos relacionados dichos en un sitio y no en
      el otro: el V I S T O o el rubro sin su considerando, o al revés; números
      que no cuadran; los que nadie marcó en la ficha, o los marcados que se
      perdieron (sexta ronda).

LA REGLA DE LA CASA: «si acusa a las sentencias buenas, la que está mal es la
verificación». Calibrada (3-oct-2026) contra 41 sentencias públicas recientes
del tribunal (11 AD, 10 AR, 10 Q y 10 RF de 2025-2026): CERO avisos en (a),
(b), (d), (e), (g), (i) y (k), y en las demás sólo 10 hallazgos revisados a
mano que son errores del propio engrose (AG 1/2023 abrogado ×2, LOPJF vieja ×4,
fechas del acto que no cuadran ×2, «localizado» con la Sala ×2). Contra los 81
proyectos malos del 26-sep al 3-oct caza las 14 clases de error medidas, todas
al 100%. Por eso casi todas las reglas se anclan a la FÓRMULA del texto
compuesto (el «dictada el …, por …, en el …») y no a cualquier mención: lo que
no se puede leer con certeza, no se acusa; y la responsable se coteja contra
el DATO (ficha o encargo), nunca entre menciones, porque el tribunal abrevia.

RONDA 2 (3-oct-2026), tras pasar la verja por los 13 documentos integrados
(integ1) y los 30 casos del compositor: (k) reconoce el verbo con «i»
(«adhiriéndose», «se adhirió»); (d) la cola «del Tribunal Federal de Justicia
Administrativa» es la misma Sala en los dos sentidos; (i) «sobresee_niega» y
«sobresee_concede» son mixtos, no sobreseimientos; (a) el hueco nombra el
campo nuevo de la ficha (fundamento_surtimiento, fraccion_63) y dice si la
ficha ya lo traía; y (d) coteja el ponente de la carátula con el del turno.
Las mismas 41 sentencias: 0 avisos nuevos.

RONDA 3 (3-oct-2026), tras el banco de oráculo (32 sentencias reales del
tribunal compuestas con el modelo por tipo) y las pruebas de punta a punta (AD
274 y AR 631): (d) coteja la carátula contra el ponente del ÚLTIMO returno (en
los 8 AD, 6 de 8 RF y 2 Q con returno la carátula nombraba al del turno y nada
lo acusaba), el renglón RECURRENTE contra quien recurre, las versales tras «lo
hizo valer / interpuesto por / promovido por», las etiquetas de rol de la
carátula coladas en la prosa y «X, en representación de X»; (e) la carátula sin
el número del asunto; (f) «el Jefa», «el Coordinadora», «el Legislatura», «el
Actual Sala» y «localizado» con prefijo temporal; (g) lee «supletoriamente» y
«artículo 2o. de la Ley de Amparo» y pregunta la sede a `tipos_asunto` (no a
`ficha.sede.cdmx`); (j) la cronología del juicio de amparo indirecto en el AR;
(l) y (m), nuevas. El hueco (a) se agrupa por DATO y dice su clave de la ficha,
para que quien junta los avisos no repita el que ya dio el compositor.

RONDA 4 (3-oct-2026), tras la verificación final contra las 32 sentencias
reales (rev/verif_final.txt) y las decisiones E1-E12 de FIXES_R4: (d) coteja el
órgano contra lo que decidió el compositor (`datos["procesal"]`), no contra lo
crudo de la ficha (Q 261: tres avisos falsos sobre un documento uniforme), y
acusa el órgano DESCARTADO que sigue en la competencia y la existencia (AR 307,
448 y 60 con las fichas viejas); «X, por conducto de su X» y la persona «en su
carácter de unidad administrativa» (RF 7, 2 y 26); el cargo y su órgano son la
misma autoridad en «X, en representación de X» (RF 6-2026, E8); las versales
se buscan DENTRO de la mención y también tras «ampara y protege a» y antes de
«promovió / interpuso» (AD 335, AR 208 y 60); (f) las fórmulas de
representación con mayúscula a media frase (AD 552, Q 342 y 229), el colectivo
sin artículo (AD 335, AR 60) y el plural del resultando contra el singular de
la legitimación o de la carátula (AD 552, E2); (j) no repite el par de fechas
que ya dijo `ficha_tramite.validar` u otra pieza (AR 448, E11), ni la clave
cruda que validar dejó en `fechas_imposibles`; (a) el auto que forma y registra
es `registro.fecha` cuando es otro que el que admite (E7); (k) el adhesivo sin
ningún auto (E5) no exige considerando ni resolutivo; y el prefijo temporal
(«la actual Sala…», «la entonces X, ahora Y») no hace otro órgano (RF 49). Las
mismas 41 sentencias: 0 avisos nuevos.

RONDA 5 (3-oct-2026), tras la segunda verificación (rev/verif_final2.txt) y las
decisiones F1-F6 de FIXES_R5: (a) el hueco tras «juicio de amparo» con el
número detrás es la VÍA, no el número (Q 335 sin fracción del 97: decía «falta
el número (acto.expediente)… LA FICHA SÍ LO TRAE» con el número a la vista);
en la queja su clave es fraccion_97, la del compositor, y se deduplica; (d)
«X, por conducto de su X» compara por el núcleo con
`tipos_asunto.misma_autoridad`, la misma función que el compositor y la
legitimación (F4; RF 7: «Delegación Estatal» y «Representación Estatal» de la
misma unidad); (m) la forma de notificación sin papel (fuente «omision» u
«omision_autoridad» en `ficha["fuentes"]`) que la oportunidad afirma (F1; Q
335, AD 274 y 335); (f) el artículo del texto que contradice el del papel en
un cargo de doble género (F5; AD 128: «la Oficial Mayor» → «el Oficial
Mayor»); y (n), nueva: el par de palabras repetido (RF 4). Las mismas 41
sentencias: 0 avisos nuevos.

SEXTA RONDA (3-oct-2026), las respuestas de David y su visto bueno para todos
los usuarios (contrato C1-C8). Las formas nuevas pasan sin avisos falsos: el
amparo en revisión ya no lleva existencia («ya viene en la sentencia
recurrida») y lleva «Procedencia.» antes de la legitimación; la carátula de la
revisión fiscal dice «RECURRENTE:» sin la Sala y la de la queja no lleva el
órgano; el renglón «RELACIONADO CON …» bajo el encabezado no se lee como el
número del asunto (e); la queja abre con «Demanda de amparo.», cuyas
autoridades no son el juzgado del auto (d) y cuya fecha va antes del auto
recurrido (j); el punto del amparo del AR que nombra acto y autoridad
(«respecto del acto que reclamó de la Sala…, consistente en la sentencia
dictada…») no es el órgano recurrido (d) ni la clase de lo recurrido (l); la
supletoriedad del Código Nacional en lo agrario es la de la Ley Agraria (art.
167), no la de la Ley de Amparo (g, C7). Y tres reglas nuevas para los asuntos
relacionados, que sólo existen si el secretario los marca (C6): (o), (p) y (q).
Además, (a) acusa el marcador de plantilla que nadie sustituyó
(«{actos_del_amparo}»). Las mismas 41 sentencias: 0 avisos nuevos; y en una
calibración ampliada de 557 sentencias más del tribunal (2024-2026: las 340
que mencionan asuntos relacionados y 217 al azar), la verja de antes y la de
ahora dan exactamente los mismos avisos. El banco de oráculo compuesto de punta
a punta con el código de hoy (32 asuntos y 5 con relacionados marcados): 0
avisos nuevos.

INTERFAZ (SPEC §6):
    revisar(texto_procesal, ficha, datos, tipo) -> list[str]   avisos
    arreglar(texto) -> (texto, cambios)                        tipografía segura

`revisar_detalle` devuelve lo mismo con la regla de cada aviso ({"regla", "aviso"})
para las pruebas y la calibración; los de hueco (a) llevan además "dato", "clave"
(la de la ficha, p. ej. «acto.organo»; "" si no se sabe), "claves" y "apartados".
Las dos
aceptan `avisos_previos` (los que ya dieron otras piezas: el hueco cuya clave ya se
avisó no se repite) y `reglas` (las letras que se corren, p. ej. "g" para pasar
la regla del supletorio por los Antecedentes y el Estudio: `revisar_supletorio`).
La verja NO decide la bandera: la llama `documento_generado.componer` cuando
rige `procedencia_por_tipo` (`activa()`).
"""
from __future__ import annotations

import datetime as _dt
import re
import unicodedata

HUECO = "*********"

__all__ = ["revisar", "revisar_detalle", "revisar_supletorio", "arreglar", "secciones", "activa", "HUECO"]


def activa() -> bool:
    """¿Rige la procedencia por tipo en esta petición? Sin la bandera, False."""
    try:
        import contexto_taller as _ct
        return bool(_ct.rige("procedencia_por_tipo"))
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════

def _plano(s: str) -> str:
    """Minúsculas, sin tildes, espacios colapsados: para comparar, nunca para escribir."""
    x = unicodedata.normalize("NFKD", str(s or ""))
    x = "".join(c for c in x if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", x).strip().lower()


def _un_renglon(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _extracto(t: str, i: int, j: int, ancho: int = 70) -> str:
    a, b = max(0, i - ancho), min(len(t), j + ancho)
    return ("…" if a > 0 else "") + _un_renglon(t[a:b]) + ("…" if b < len(t) else "")


def _tipo(tipo: str) -> str:
    try:
        import tipos_asunto as _ta
        n = _ta.normalizar(tipo)
        if n:
            return n
    except Exception:
        pass
    p = _plano(tipo).replace(" ", "_").replace("-", "_")
    for clave, alias in (("amparo_directo", ("ad", "directo", "amparo_directo")),
                         ("amparo_revision", ("ar", "revision", "amparo_revision", "amparo_en_revision")),
                         ("queja", ("q", "qa", "qc", "queja", "recurso_de_queja")),
                         ("revision_fiscal", ("rf", "fiscal", "revision_fiscal"))):
        if p in alias:
            return clave
    return ""


def _es_ficha(ficha) -> bool:
    """Una ficha de trámite de verdad (no un dict vacío ni None)."""
    return isinstance(ficha, dict) and bool(ficha) and any(
        k in ficha for k in ("formato", "tipo", "numero", "admision", "acto",
                             "responsable", "presentacion", "ministerio_publico"))


def _g(d, *ruta, defecto=""):
    """d["a"]["b"]… sin tropezar con None ni tipos raros."""
    x = d
    for k in ruta:
        if not isinstance(x, dict):
            return defecto
        x = x.get(k)
    return defecto if x is None else x


# ═══════════════════════════════════════════════════════════════════════════
# LAS FECHAS: en letra o en cifras, con su posición
# ═══════════════════════════════════════════════════════════════════════════
# Se reusa el lector de `fechas_en_autos` (el de la verja del estudio), que ya
# distingue «acuerdo de primero de julio» de una fecha y lee el año del prefijo
# más largo. Sin él, las reglas de fechas callan: mejor mudas que injustas.

def _fechas(texto: str) -> list:
    """[(date, inicio, fin, literal)] en el orden del texto."""
    try:
        import fechas_en_autos as _fa
    except Exception:
        return []
    # LA LLAMADA DE NOTA PEGADA AL AÑO: «dos mil veinticuatro1» se leía como
    # «dos mil» (el año 2000) y fabricaba una fecha imposible en sentencias
    # buenas (AD 621 y 750/2025). Se blanquea el número con la misma longitud,
    # para que las posiciones no se muevan.
    t = re.sub(r"(?<=[a-záéíóúñ])\d{1,2}(?=\W|$)", lambda m: " " * len(m.group(0)), str(texto or ""))
    out = []
    for m in _fa._RX_LETRAS.finditer(t):
        d, mes = _fa._dia(m.group(1)), _fa._MESES.get(m.group(2).lower())
        toks = m.group(3).split()
        a, usado = None, ""
        for n in range(len(toks), 0, -1):
            if toks[n - 1].lower() == "y":
                continue
            a = _fa._anio(" ".join(toks[:n]))
            if a:
                usado = " ".join(toks[:n])
                break
        if d and mes and a:
            fin = m.start(3) + len(usado) if usado else m.end()
            try:
                out.append((_dt.date(a, mes, d), m.start(), fin, t[m.start():fin]))
            except ValueError:
                out.append((None, m.start(), fin, t[m.start():fin]))
    for m in _fa._RX_CIFRAS.finditer(t):
        if m.group(1):
            d, mes, a = int(m.group(1)), _fa._MESES.get(m.group(2).lower()), int(m.group(3))
        else:
            d, mes, a = int(m.group(4)), int(m.group(5)), int(m.group(6))
        if mes and 1 <= d <= 31 and 1 <= mes <= 12 and 1900 <= a <= 2100:
            try:
                out.append((_dt.date(a, mes, d), m.start(), m.end(), m.group(0)))
            except ValueError:
                out.append((None, m.start(), m.end(), m.group(0)))
    out.sort(key=lambda x: x[1])
    return out


def _iso(x):
    if isinstance(x, _dt.date):
        return x
    m = re.match(r"^\s*(\d{4})-(\d{2})-(\d{2})", str(x or ""))
    if not m:
        return None
    try:
        return _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _fechas_de_valor(x, acumulado: set, profundidad: int = 0):
    """Todas las fechas de un dict/lista/cadena: ISO o escritas."""
    if profundidad > 6 or x is None:
        return
    if isinstance(x, dict):
        for v in x.values():
            _fechas_de_valor(v, acumulado, profundidad + 1)
    elif isinstance(x, (list, tuple, set)):
        for v in x:
            _fechas_de_valor(v, acumulado, profundidad + 1)
    elif isinstance(x, _dt.date):
        acumulado.add(x)
    elif isinstance(x, str):
        d = _iso(x)
        if d:
            acumulado.add(d)
        elif re.search(r"\d|enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
                       r"septiembre|octubre|noviembre|diciembre", x, re.I):
            for f in _fechas(x):
                if f[0]:
                    acumulado.add(f[0])


# ═══════════════════════════════════════════════════════════════════════════
# EL TEXTO EN APARTADOS
# ═══════════════════════════════════════════════════════════════════════════
# El texto llega como lo compone `documento_generado` (un párrafo por renglón,
# «R E S U L T A N D O», «PRIMERO. Rótulo. Texto…»), pero también se lee bien
# un texto de una pieza o el de una versión pública (párrafos numerados «9.
# Competencia.»): los rótulos se buscan tras un salto o tras el final de una
# frase, no sólo al principio de renglón.

_ORD = (r"(?:PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|S[ÉE]PTIMO|OCTAVO|NOVENO|"
        r"D[ÉE]CIMO(?:\s+(?:PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|S[ÉE]PTIMO|OCTAVO|NOVENO))?|"
        r"[ÚU]NICO)")
_ANTES = r"(?:^|(?<=\n)|(?<=[.;:,] )|(?<=[.;:,]  ))"
_RX_VISTO = re.compile(
    r"(?<![A-Za-zÁÉÍÓÚáéíóúñÑ])(?:V\s+I\s+S\s+T\s+O\s*S?|V\s+i\s+s\s+t\s+o\s*s?|VISTOS?)"
    r"(?=\s*[,:]|\s+para\b|\s+los\b)")
_RX_RESULTANDO = re.compile(_ANTES + r"[ \t]*(?:R\s*E\s*S\s*U\s*L\s*T\s*A\s*N\s*D\s*O\s*S?)\s*:?")
_RX_CONSIDERANDO = re.compile(_ANTES + r"[ \t]*(?:C\s*O\s*N\s*S\s*I\s*D\s*E\s*R\s*A\s*N\s*D\s*O\s*S?)\s*:?")
_RX_RESUELVE = re.compile(r"R\s*E\s*S\s*U\s*E\s*L\s*V\s*E\s*:?|[Ss]e\s+resuelve\s*[:;]|PUNTOS\s+RESOLUTIVOS")
_RX_FIN_RESOL = re.compile(
    r"Notif[íi]quese|N\s*o\s*t\s*i\s*f\s*[íi]\s*q\s*u\s*e\s*s\s*e|"
    r"As[íi],?\s+(?:por\s+unanimidad|lo\s+resolvi)|\bS[ÍI]NTESIS\b|=====NOTAS")
_RX_RUBRO = re.compile(
    _ANTES + r"[ \t]*(" + _ORD + r"|\d{1,3})\s*\.[ \t]*(?=[A-ZÁÉÍÓÚÑ¿«“\"])")

# Etiquetas por rótulo. Un rótulo puede llevar varias («Trámite y turno del
# recurso», «Interposición y trámite del recurso de revisión»).
_ETIQUETAS = (
    ("sesion", r"sesi[óo]n"),
    ("returno", r"returno|integraci[óo]n"),
    ("turno", r"\bturno\b"),
    ("derechos", r"derechos"),
    ("tercero", r"tercer[oa]s?\s+interesad"),
    ("presentacion", r"presentaci[óo]n|demanda|interposici[óo]n|^juicio\b"),
    ("tramite", r"tr[áa]mite|substanciaci|sustanciaci|admisi[óo]n|radicaci"),
    ("competencia", r"competencia"),
    ("existencia", r"existencia|certeza"),
    ("legitimacion", r"legitimaci"),
    ("oportunidad", r"oportunidad"),
    ("procedencia", r"procedencia|improcedencia"),
    ("dispensa", r"transcrip|innecesari|reclamad[oa]s?\s+y|recurrid[oa]s?\s+y|agravios|conceptos"),
    ("antecedentes", r"antecedentes"),
    ("adhesivo", r"adhesiv|adhesi[óo]n"),
    # SEXTA RONDA (C6, 3-oct-2026): el considerando de los asuntos relacionados
    # («Conexidad.», «Hecho notorio.» o «Asuntos relacionados.»).
    ("relacionados", r"conexidad|conexi[óo]n|hecho\s+notorio|relacionad|^relaci[óo]n(?:\s+con\b|\s*$)"),
)


def secciones(texto: str) -> list:
    """El documento en apartados: [{zona, rotulo, etiquetas, texto, nombre}].

    zona ∈ caratula | proemio | visto | resultando | considerando | resolutivos.
    """
    t = str(texto or "").replace("\r", "")
    n = len(t)
    m_v = _RX_VISTO.search(t)
    m_r = _RX_RESULTANDO.search(t, m_v.end() if m_v else 0) or _RX_RESULTANDO.search(t)
    p_r = m_r.start() if m_r else None
    m_c = _RX_CONSIDERANDO.search(t, m_r.end() if m_r else 0)
    p_c = m_c.start() if m_c else None
    m_s = _RX_RESUELVE.search(t, m_c.end() if m_c else (m_r.end() if m_r else 0))
    p_s = m_s.start() if m_s else None
    p_v = m_v.start() if (m_v and (p_r is None or m_v.start() < p_r)) else None

    out = []

    def _agrega(zona, rotulo, ini, fin):
        if fin is None or ini is None or fin <= ini:
            return
        cuerpo = t[ini:fin]
        if not cuerpo.strip():
            return
        et = set()
        for clave, rx in _ETIQUETAS:
            if rotulo and re.search(rx, rotulo, re.I):
                et.add(clave)
        if zona == "caratula":
            nombre = "la carátula"
        elif zona == "proemio":
            nombre = "el proemio"
        elif zona == "visto":
            nombre = "el V I S T O"
        elif zona == "resolutivos":
            nombre = "los resolutivos"
        elif rotulo:
            nombre = f"el {zona} «{_un_renglon(rotulo)[:70]}»"
        else:
            nombre = "los " + zona + "s"
        out.append({"zona": zona, "rotulo": _un_renglon(rotulo), "etiquetas": et,
                    "texto": cuerpo, "inicio": ini, "fin": fin, "nombre": nombre})

    # ── carátula y proemio: todo lo que precede al V I S T O ──
    fin_cab = p_v if p_v is not None else (p_r if p_r is not None else 0)
    if fin_cab:
        cab = t[:fin_cab]
        m_p = re.search(r"(?:^|\n|(?<=\.)\s)[^\n]{0,120}?(?:Resoluci[óo]n|Acuerdo)\s+del?\s+[^\n]{0,400}?"
                        r"(?:correspondiente\s+a\s+la\s+sesi[óo]n|sesi[óo]n)", cab)
        if m_p:
            _agrega("caratula", "", 0, m_p.start())
            _agrega("proemio", "", m_p.start(), fin_cab)
        else:
            _agrega("caratula", "", 0, fin_cab)
    if p_v is not None:
        _agrega("visto", "", p_v, p_r if p_r is not None else (p_c if p_c is not None else n))

    def _partir(zona, ini, fin):
        if ini is None or fin is None:
            return
        rubros = []
        for m in _RX_RUBRO.finditer(t, ini, fin):
            resto = t[m.end():min(fin, m.end() + 160)]
            mr = re.match(r"([^\n]{1,140}?)\.(?=\s|$)", resto)
            rot = mr.group(1).strip() if mr else ""
            if m.group(1)[0].isdigit():
                # LOS PÁRRAFOS NUMERADOS SÓLO SON RÓTULO SI EL RÓTULO ES CORTO.
                # «9. Competencia.» lo es; «4. La parte quejosa señaló como…»
                # es un párrafo del relato.
                if not rot or len(rot.split()) > 9:
                    continue
            rubros.append((m.start(), rot[:140]))
        if not rubros or rubros[0][0] > ini + 3:
            rubros.insert(0, (ini, ""))
        for k, (p, rot) in enumerate(rubros):
            _agrega(zona, rot, p, rubros[k + 1][0] if k + 1 < len(rubros) else fin)

    if p_r is not None:
        _partir("resultando", m_r.end(), p_c if p_c is not None else (p_s if p_s is not None else n))
    if p_c is not None:
        _partir("considerando", m_c.end(), p_s if p_s is not None else n)
    if p_s is not None:
        m_f = _RX_FIN_RESOL.search(t, m_s.end())
        _agrega("resolutivos", "", p_s, m_f.start() if m_f else n)
    return out


def _de(secs, zona=None, etiqueta=None, sin=None):
    return [s for s in secs
            if (zona is None or s["zona"] == zona)
            and (etiqueta is None or etiqueta in s["etiquetas"])
            and (sin is None or not (s["etiquetas"] & set(sin)))]


# ═══════════════════════════════════════════════════════════════════════════
# (a) HUECOS FUERA DE LAS DOS FECHAS DE LA SESIÓN
# ═══════════════════════════════════════════════════════════════════════════
_RX_HUECO = re.compile(r"\*{3,}")
# Los huecos legítimos son las fechas que se fijan cuando el asunto se lista:
# «se listó el *********», «sesión [ordinaria] de ********* [siguiente]» —en el
# proemio, en el resultando de la sesión y en el cierre—.
_RX_HUECO_SESION = re.compile(
    r"(?:se\s+list[óo]\s+(?:el\s+)?|listad[oa]\s+(?:el\s+)?|"
    r"sesi[óo]n(?:\s+(?:ordinaria|extraordinaria|p[úu]blica|virtual|remota|privada|"
    r"ordinaria\s+virtual))?\s+(?:de(?:l\s+d[íi]a)?|celebrada\s+el|del)\s+)$", re.I)
# Qué dato falta, por lo que lo precede. Sin esto el aviso dice «hay un hueco»
# y el secretario tiene que adivinar qué se le pide.
# Los rótulos de la carátula van en versales y se leen sin re.I (un «actora:»
# en minúscula dentro de la prosa no es el renglón de la carátula).
#
# CADA DATO CON SU CLAVE DE LA FICHA (ronda 3, 3-oct-2026). El banco de
# oráculo midió el mismo dato avisado hasta cinco veces: Q 335 llevaba 8 avisos
# para 2 datos —el órgano en hueco en el V I S T O, la interposición, la
# competencia y la procedencia, un aviso por apartado, más el del compositor
# «FALTA … (acto.organo)»—; AD 279, cuatro para los tres huecos del adhesivo.
# Ahora el hueco dice qué campo de la ficha lo llena (la misma clave que nombra
# el compositor entre paréntesis), los apartados con el mismo dato van en UN
# aviso, y quien junta los avisos puede callar el que ya dio otra pieza
# (`avisos_previos`, o la clave en `revisar_detalle`).
# La clave es una cadena o una función (apartado, antes, después, antes largo,
# tipo) → cadena; "" = no se sabe qué campo lo llena (entonces se agrupa por
# apartado, como antes).


def _k_inciso(s, a, d, la, tipo):
    # El inciso no es un campo: en el AD lo da la materia (107-V, a)-d)), en el
    # AR la clase de lo recurrido (81-I) y en la queja el inciso del 97.
    return {"queja": "inciso_97", "amparo_directo": "materia", "amparo_revision": "clase_recurrida"}.get(tipo, "")


def _k_fraccion(s, a, d, la, tipo):
    if tipo == "revision_fiscal":
        return "fraccion_63" if "procedencia" in s["etiquetas"] else ""
    return "fraccion_97" if tipo == "queja" else ""


_RX_ROTULO_AUTORIDAD_RESP = re.compile(r"AUTORIDAD(?:ES)?\s+RESPONSABLES?\s*:\s*\n?\s*$")


def _k_organo(s, a, d, la, tipo):
    # EN LOS RECURSOS DE AMPARO «AUTORIDAD RESPONSABLE:» ES DE LA DEMANDA (revisión
    # AR, 3-oct-2026; variante AR 448 sin demanda). Ese rótulo sólo sale en el
    # resultando que copia la demanda de amparo indirecto (AR y, con C3, la queja
    # de la fracción I): su hueco lo llena `demanda.autoridades`. Se atribuía a
    # `acto.organo` —el juzgado que dictó lo recurrido—, la verja decía «LA FICHA
    # SÍ LO TRAE» con el Juez de Distrito y contradecía el aviso correcto del
    # compositor: quien la siguiera ponía al juez de amparo como responsable.
    # `acto.organo` queda para «ÓRGANO RECURRIDO» y las demás fórmulas.
    if tipo in ("amparo_revision", "queja") and _RX_ROTULO_AUTORIDAD_RESP.search(a or ""):
        return "el renglón de las autoridades responsables de la demanda de amparo", "demanda.autoridades"
    return {"amparo_directo": "responsable", "amparo_revision": "acto.organo", "queja": "acto.organo",
            "revision_fiscal": "sala"}.get(tipo, "")


def _k_parte(s, a, d, la, tipo):
    # «RECURRENTE ADHESIVO:» (la carátula de la revisión fiscal, C4, 3-oct-2026)
    # es quien se adhirió, no quien recurre.
    if re.search(r"ADHESIV|ADHERENTE", a):
        return "adhesivo.quien"
    if re.search(r"TERCER", a):
        return "terceros"
    if re.search(r"ACTORA", a):
        return "actora"
    if re.search(r"RECURRENTE", a) or tipo == "amparo_directo":
        return "promovente"
    return "quejoso"


def _k_firma(s, a, d, la, tipo):
    return "secretario" if re.search(r"SECRETARI[OA]\s*:\s*$", a) else "magistrado"


def _k_ponente(s, a, d, la, tipo):
    return "returno.ponente" if "returno" in s["etiquetas"] else "turno.ponente"


def _k_fecha_auto(s, a, d, la, tipo):
    if "adhesivo" in s["etiquetas"] or re.search(r"adhe(?:si|r)|adhir", d, re.I):
        return "adhesivo.admision"
    if "returno" in s["etiquetas"]:
        return "returno.fecha"
    if "turno" in s["etiquetas"]:
        return "turno.fecha"
    if tipo == "queja" and re.search(r"rendido|informe", d, re.I):
        return "informe_101.fecha"
    # «…; y por auto de ********* lo admitió a trámite» (ronda 4, E7): el
    # segundo de los dos autos es la admisión.
    if re.match(r"^\W*(?:este\s+Tribunal\s+Colegiado\s+)?(?:lo|la)\s+admiti", d, re.I):
        return "admision.fecha"
    return ""


def _k_auto_presidencia(s, a, d, la, tipo):
    """EL AUTO QUE FORMA Y REGISTRA NO ES SIEMPRE EL QUE ADMITE (ronda 4,
    3-oct-2026, E7). El compositor escribe dos autos cuando son dos («Por auto
    de Presidencia de X, … lo registró…; y por auto de Y lo admitió») y, en la
    queja de la fracción II, el primero es el que registra y REQUIERE el informe
    con justificación (Q 335 del banco). Ese hueco es `registro.fecha`, la clave
    con que el compositor lo avisa; con `admision.fecha` el aviso salía dos
    veces (los casos de la queja fr. II del compositor, ronda 4)."""
    if re.search(r"\brequiri[óo]\b", d, re.I) or re.search(r";\s*(?:y\s+)?por\s+auto\s+de\b", d, re.I):
        return "registro.fecha"
    return "admision.fecha"


# EL HUECO DE LA VÍA NO ES EL DEL NÚMERO (quinta ronda, 3-oct-2026; Q 335 del
# banco sin fracción del 97, verif_q2 t6). Sin la fracción, el compositor deja
# en hueco la VÍA —«en el juicio de amparo ********* 1201/2026»— y avisa
# «FALTA LA FRACCIÓN DEL ARTÍCULO 97 (fraccion_97)»; la verja leía el hueco tras
# «juicio de amparo» como el número del juicio y decía «falta el número
# (acto.expediente)… LA FICHA DE TRÁMITE SÍ LO TRAE», con el número a la vista
# en la línea siguiente: un aviso falso que además mandaba a buscar un dato que
# no faltaba. Si tras el hueco viene el número (en cifras, otro hueco o el
# marcador TESTADO de las versiones públicas del banco), o la mención es «un
# juicio de amparo *********» (la competencia, que no lleva número), lo que
# falta es la vía: en la queja la decide la fracción del 97 (I indirecto, II
# directo) y es la clave con que avisa el compositor, así que se deduplica.
_RX_NUMERO_TRAS_HUECO = re.compile(r"^\s*(?:\d{1,6}\s*/\s*\d{2,4}\b|\*{3,}|TESTADO\b)")
_DATO_VIA = "la vía del juicio de amparo («indirecto» o «directo»)"


def _k_via(tipo):
    return (_DATO_VIA, "fraccion_97" if tipo == "queja" else "acto.via")


def _k_numero(s, a, d, la, tipo):
    p = _plano(a)
    if re.search(r"\btoca\s*$", p):
        return "acto.toca"
    if re.search(r"\bexpediente\s*$", p):
        return "expediente_tfja" if tipo == "revision_fiscal" else "acto.expediente"
    if re.search(r"\bjuicio\s+de\s+amparo\s*$", p) and (
            _RX_NUMERO_TRAS_HUECO.match(d or "") or re.search(r"\bun\s+juicio\s+de\s+amparo\s*$", p)):
        return _k_via(tipo)
    if re.search(r"\bjuicio(?:\s+de\s+amparo)?(?:\s+indirecto|\s+directo)?\s*$", p):
        return "acto.expediente"
    if re.search(r"\bnumero\s*$", p):
        # «este Tribunal Colegiado lo registró con el número» es el del asunto;
        # «correspondió al Juzgado…, que lo registró con el número» es el del
        # juicio de origen (o el expediente del TFJA).
        pl = _plano(la)
        if re.search(r"tribunal\s+colegiado|presidencia", pl[-140:]):
            return "numero"
        if re.search(r"correspondi|juzgado|sala\b", pl[-200:]):
            return "expediente_tfja" if tipo == "revision_fiscal" else "acto.expediente"
        return "numero"
    return ""


def _k_presentacion(s, a, d, la, tipo):
    if re.search(r"depositad", a, re.I):
        return "deposito_postal"
    if "adhesivo" in s["etiquetas"]:
        return "adhesivo.presentacion"
    # LA QUEJA TAMBIÉN ABRE CON LA DEMANDA (C3, sexta ronda, 3-oct-2026): su
    # fecha es la de la demanda de amparo indirecto, no la del recurso.
    if (tipo in ("amparo_revision", "queja") and "presentacion" in s["etiquetas"]
            and re.search(r"demanda", s["rotulo"], re.I)):
        return "demanda.fecha"
    return "presentacion"


def _k_razon_63(s, a, d, la, tipo):
    # «…artículo 63, fracción *********, …, toda vez que *********»: la razón
    # la escribe sola la fracción (y la cuantía); es el mismo dato.
    return "fraccion_63" if tipo == "revision_fiscal" and "procedencia" in s["etiquetas"] else ""


def _k_notificacion(s, a, d, la, tipo):
    return "adhesivo.notificacion" if "adhesivo" in s["etiquetas"] else "notificacion"


_DATO_DEL_HUECO = tuple((re.compile(p, f), nom, clave) for p, f, nom, clave in (
    (r"inciso\s*$", re.I, "el inciso", _k_inciso),
    (r"materias?\s*$", re.I, "la materia", "materia"),
    (r"fracci[óo]n\s*$", re.I, "la fracción", _k_fraccion),
    (r"(?:AUTORIDAD(?:ES)?\s+RESPONSABLES?|[ÓO]RGANO\s+RECURRIDO|[ÓO]RGANO\s+QUE\s+DICT[ÓO]\s+EL\s+AUTO\s+"
     r"RECURRIDO|SALA\s+RESPONSABLE)\s*:\s*\n?\s*$", 0, "el órgano responsable", _k_organo),
    (r"(?:QUEJOS[OA]|RECURRENTE|TERCER[OA]\s+INTERESAD[OA]|ACTORA)[A-ZÁÉÍÓÚ ]{0,30}:\s*$", 0, "el nombre de la parte",
     _k_parte),
    (r"(?:MAGISTRAD[OA](?:\s+PONENTE)?|SECRETARI[OA])\s*:\s*$", 0, "el nombre de quien firma", _k_firma),
    (r"(?:auto|acuerdo|prove[íi]do)\s+de\s+presidencia\s+de\s*$", re.I, "la fecha del auto de Presidencia",
     _k_auto_presidencia),
    (r"(?:se\s+turn|turnaron|returnaron)[^.]{0,60}?(?:ponencia\s+de|al?\s+(?:la\s+)?magistrad[oa])\s*$", re.I,
     "el ponente", _k_ponente),
    (r"(?:acuerdo|auto)\s+de\s*$", re.I, "la fecha del auto", _k_fecha_auto),
    (r"(?:n[úu]mero|expediente|toca|juicio(?:\s+de\s+amparo)?(?:\s+indirecto|\s+directo)?)\s*$", re.I, "el número",
     _k_numero),
    # «juicio de amparo ********* *********»: el primero es la vía (arriba) y
    # el segundo, el número (quinta ronda); antes se quedaba sin dato ni clave.
    (r"juicio\s+de\s+amparo\s+\*{3,}\s*$", re.I, "el número", "acto.expediente"),
    (r"(?:presentad[oa]|interpuest[oa]|depositad[oa]|recibid[oa]|se\s+present[óo]|se\s+interpuso)\s+el\s*$", re.I,
     "la fecha de presentación", _k_presentacion),
    (r"(?:dictad[oa]|emitid[oa])\s+el\s*$", re.I, "la fecha del acto", "acto.fecha"),
    (r"(?:por|ante)\s+(?:el|la)?\s*$", re.I, "el órgano", _k_organo),
    # «El conocimiento del asunto correspondió a *********, que lo registró…»
    # (AR 307, 448 y 60 del banco, ronda 3): el juzgado, o la Sala en la RF.
    (r"correspondi[óo]\s+(?:al?|a\s+la)\s*$", re.I, "el órgano", _k_organo),
    (r"(?:el|la|los|las)\s+art[íi]culos?\s*$", re.I, "el precepto", ""),
    (r"surti[óo]\s+efectos[^.;]{0,60}?(?:conforme\s+al?|en\s+t[ée]rminos\s+del?)\s*$", re.I,
     "el precepto que rige el surtimiento de la notificación", "fundamento_surtimiento"),
    (r"(?:conforme\s+al?|en\s+t[ée]rminos\s+del?|con\s+fundamento\s+en\s+el)\s*$", re.I, "el precepto", ""),
    (r"\bLa\s+Justicia\s+de\s+la\s+Uni[óo]n\s*$", 0, "el sentido del fallo", ""),
    # «… y el escrito se presentó el *********, por lo que *********» (el
    # considerando del adhesivo sin sus dos fechas): lo que falta es la
    # conclusión del cómputo, que se escribe sola cuando constan las fechas.
    (r"\bpor\s+lo\s+que\s*$", re.I, "el veredicto de oportunidad (se escribe solo cuando constan las dos fechas)", ""),
    # La procedencia de la revisión fiscal: «…artículo 63, fracción *********,
    # …, toda vez que *********» y, en la fracción VI, «…relativa a *********».
    (r"\btoda\s+vez\s+que\s*$", re.I, "la razón por la que procede (el supuesto de la fracción)", _k_razon_63),
    (r"\brelativ[oa]\s+a\s*$", re.I, "de qué trata la resolución impugnada (el supuesto de la fracción)", ""),
    (r"notific\w*[^.;]{0,80}?\s+el\s*$", re.I, "la fecha de la notificación", _k_notificacion),
    (r"\bel\s*$", re.I, "la fecha", ""),
))

# Las claves que NO son un campo que se teclee tal cual: la del inciso (lo
# deciden la materia o la clase de lo recurrido) y quien firma (va en el
# encargo, no en la ficha). De éstas no se dice «la ficha sí lo trae».
_CLAVES_DERIVADAS = {"materia", "clase_recurrida", "magistrado", "secretario"}


def _dato_y_clave(antes, despues, largo, s, tipo):
    for rx, nom, clave in _DATO_DEL_HUECO:
        if rx.search(antes):
            if callable(clave):
                try:
                    clave = clave(s, antes, despues, largo, tipo)
                except Exception:
                    clave = ""
            # La función puede decir también QUÉ dato es (la vía, no el número).
            if isinstance(clave, tuple) and len(clave) == 2:
                nom, clave = clave
            return nom, (clave or "")
    return "el dato que va en ese lugar", ""


def _valor_de_clave(ficha, clave: str) -> str:
    """El valor de «acto.organo» en la ficha, como texto ("" si no está)."""
    f = ficha if isinstance(ficha, dict) else {}
    v = _g(f, *clave.split(".")) if clave else ""
    if isinstance(v, (list, tuple)):
        v = ", ".join(str(x) for x in v if str(x or "").strip())
    v = _un_renglon(v) if isinstance(v, str) else ""
    return "" if (not v or HUECO in v) else v


# LOS CAMPOS NUEVOS DE LA FICHA (ronda 2): el precepto del surtimiento y, en la
# revisión fiscal, la fracción del artículo 63 y la cuantía. Si el hueco es uno
# de ellos —o cualquier otro dato con su clave (ronda 3)—, el aviso dice en qué
# campo se da; y si la ficha YA lo trae, dice que el dato se perdió en el
# camino —eso no lo arregla el secretario, es una pieza que no lo leyó—.
# La cuantía sola no decide la fracción (puede no alcanzar la de la fracción
# I y surtirse otra): si la ficha trae la cuantía y no la fracción, se dice,
# pero no se culpa al camino.
def _nota_campo(clave, ficha, dato="") -> str:
    if not clave:
        return ""
    if dato == "el inciso":
        return f" Lo decide el campo «{clave}» de la ficha de trámite."
    if clave not in _CLAVES_DERIVADAS:
        valor = _valor_de_clave(ficha, clave)
        if valor:
            return (f" LA FICHA DE TRÁMITE SÍ LO TRAE ({clave}: «{valor[:90]}»): el dato no llegó al texto; lo "
                    f"perdió una pieza del camino, no el formulario.")
    if clave in ("magistrado", "secretario"):
        return f" Va en el campo «{clave}» del formulario."
    nota = f" Va en el campo «{clave}» de la ficha de trámite."
    cuantia = _valor_de_clave(ficha, "cuantia") if clave == "fraccion_63" else ""
    if cuantia:
        nota += f" La ficha trae la cuantía («{cuantia[:60]}»): con ella se decide si es la fracción I."
    return nota


# LO QUE YA DIJO OTRA PIEZA. El compositor nombra la clave entre paréntesis
# («FALTA LA FECHA DE PRESENTACIÓN DEL RECURSO (presentacion): …») y la
# oportunidad tiene su aviso propio del precepto del surtimiento; si el mismo
# dato ya está avisado, el hueco de la verja sobra.
_MARCAS_PREVIAS = {
    "fundamento_surtimiento": ("fundamento_surtimiento", "surtimiento va sin precepto",
                               "surtimiento sin precepto", "precepto del surtimiento"),
}


_VEREDICTO = "el veredicto de oportunidad (se escribe solo cuando constan las dos fechas)"
# Avisos de otras piezas que nombran el dato sin su clave entre paréntesis.
_MARCAS_DE_DATO = {_VEREDICTO: ("veredicto de oportunidad",)}


def _ya_avisado(g: dict, previos) -> bool:
    if not previos:
        return False
    clave = g.get("clave") or ""
    marcas = ((f"({clave})", f"«{clave}»", f"({clave}:", f"({clave},") + _MARCAS_PREVIAS.get(clave, ())
              if clave else ()) + _MARCAS_DE_DATO.get(g.get("dato"), ())
    if not marcas:
        return False
    for a in previos:
        pa = str(a or "").lower()
        if any(m.lower() in pa for m in marcas):
            return True
    return False


def _lista_apartados(nombres: list) -> str:
    n = [x.upper() for x in nombres]
    return n[0] if len(n) == 1 else ", ".join(n[:-1]) + " Y " + n[-1]


_RX_FIN_FRASE = re.compile(r"(?<![A-ZÁÉÍÓÚÑ])\.\s+(?=[A-ZÁÉÍÓÚÑ¿«])|\n")
# «{actos_del_amparo}», «{quejoso}», «{responsable_originaria}»: los marcadores
# de las plantillas (`tipos_asunto.RAMAS_REVISION`, el banco), en llaves.
_RX_MARCADOR = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]{2,40}\}")
_MARCADORES_CONOCIDOS = {
    "actos_del_amparo": ("van el acto y la autoridad del amparo (demanda.actos y demanda.autoridades de la ficha "
                         "de trámite) o, sin ellos, la remisión a «los actos precisados en la resolución "
                         "recurrida»"),
    "quejoso": "va el nombre de la parte quejosa",
    "responsable_originaria": "va la autoridad responsable del juicio de amparo",
}


def _regla_huecos(t, secs, ficha, tipo, out, previos=None):
    # 1) Un grupo por DATO (su clave de la ficha), con todos los apartados donde
    #    va en hueco; sin clave, un grupo por apartado, como antes.
    grupos, orden = {}, []
    for s in secs:
        cuerpo = s["texto"]
        for m in _RX_HUECO.finditer(cuerpo):
            antes = cuerpo[max(0, m.start() - 60):m.start()]
            if _RX_HUECO_SESION.search(antes):
                continue
            # Lo que sigue, HASTA EL FIN DE LA FRASE (ronda 4): «… registró el
            # recurso con el número X y requirió…» decide la clave del auto, y
            # «se tuvo a [nombre largo] interponiendo revisión adhesiva» la
            # del adhesivo; 60 caracteres no llegaban. El punto de «S.A. de C.V.»
            # (tras una mayúscula, o seguido de minúscula) no cierra la frase.
            despues = _RX_FIN_FRASE.split(cuerpo[m.end():m.end() + 220], maxsplit=1)[0]
            largo = cuerpo[max(0, m.start() - 220):m.start()]
            dato, clave = _dato_y_clave(antes, despues, largo, s, tipo)
            llave = ("clave", clave) if clave else ("apartado", s["nombre"])
            g = grupos.get(llave)
            if g is None:
                g = grupos[llave] = {"dato": dato, "clave": clave, "apartados": [], "n": 0,
                                     "extracto": _extracto(cuerpo, m.start(), m.end(), 60)}
                orden.append(llave)
            if s["nombre"] not in g["apartados"]:
                g["apartados"].append(s["nombre"])
            g["n"] += 1
    # 2) Lo que otra pieza ya avisó (la misma clave entre paréntesis) se calla.
    #    UN AVISO POR CAMPO, sin juntar campos distintos de un mismo apartado:
    #    quien junta los avisos (documento_generado._hueco_ya_avisado) calla
    #    el de la verja cuando su clave ya está avisada; con dos campos en un
    #    aviso, o callaba también el que nadie avisó, o los repetía los dos
    #    (AD 279 de punta a punta en el banco, ronda 3: el considerando del
    #    adhesivo salía con el aviso de la verja junto a los tres del
    #    compositor).
    for k in orden:
        g = grupos[k]
        if _ya_avisado(g, previos):
            continue
        clave, apartados = g["clave"], g["apartados"]
        extra = g["n"] - len(apartados)
        mas = (f" (y {extra} más en {'el mismo apartado' if len(apartados) == 1 else 'esos apartados'})"
               if extra > 0 else "")
        out.append(("a", f"HUECO EN {_lista_apartados(apartados)}: falta {g['dato']}"
                         f"{' (' + clave + ')' if clave else ''} — «{g['extracto']}»{mas}. Sólo las dos fechas de "
                         f"la sesión se entregan en hueco; este dato hay que darlo en el formulario de trámite o "
                         f"escribirlo antes de firmar.{_nota_campo(clave, ficha, g['dato'])}",
                    {"dato": g["dato"], "clave": clave, "claves": [clave] if clave else [],
                     "apartados": list(apartados)}))
    # 3) EL MARCADOR DE PLANTILLA QUE NADIE SUSTITUYÓ (sexta ronda, 3-oct-2026,
    #    C5). El punto del amparo que confirma lleva ahora «{actos_del_amparo}» y
    #    lo llena `documento_generado` con los actos y las autoridades de la
    #    ficha (o la remisión genérica); si una pieza del camino no lo llena —la
    #    tarjeta de decisión sin su parche, un resolutivo del banco que llega
    #    por otro lado—, el marcador sale tal cual en el proyecto y es peor que
    #    un hueco: ni siquiera parece un dato. Las 41 sentencias del tribunal no
    #    llevan llaves.
    marcas, orden_m = {}, []
    for s in secs:
        for m in _RX_MARCADOR.finditer(s["texto"]):
            g = marcas.get(m.group(0))
            if g is None:
                g = marcas[m.group(0)] = {"apartados": [], "extracto": _extracto(s["texto"], m.start(), m.end(), 60)}
                orden_m.append(m.group(0))
            if s["nombre"] not in g["apartados"]:
                g["apartados"].append(s["nombre"])
    for k in orden_m:
        g = marcas[k]
        que = _MARCADORES_CONOCIDOS.get(k[1:-1], "")
        out.append(("a", f"MARCADOR DE PLANTILLA SIN SUSTITUIR EN {_lista_apartados(g['apartados'])}: «{k}» — "
                         f"«{g['extracto']}». Una pieza del camino no lo llenó y saldría tal cual en el "
                         f"proyecto{': ' + que if que else ''}. Se escribe el dato o se vuelve a generar.",
                    {"dato": "un marcador de plantilla", "clave": "", "claves": [],
                     "apartados": list(g["apartados"])}))


# ═══════════════════════════════════════════════════════════════════════════
# (b) FÓRMULAS EVASIVAS Y AFIRMACIONES POR OMISIÓN
# ═══════════════════════════════════════════════════════════════════════════
# Todas ocupan el sitio de un dato con la promesa de que el dato existe. Las
# del corpus del tribunal NO entran: «quien fue emplazado al presente juicio,
# según las constancias que obran en autos» (tercero interesado, oro AD) remite
# a una constancia para acreditar un hecho que ya se dijo; aquí sólo se acusa
# la remisión que SUSTITUYE a la fecha, al número o al nombre.
_EVASIVAS = (
    (r"que\s+se\s+advierte\s+de\s+(?:las\s+)?constancias", "«que se advierte de las constancias»"),
    (r"(?:fecha|d[íi]a|n[úu]mero|ponencia)\s+(?:que\s+)?(?:se\s+)?advierte\s+de\s+(?:las\s+)?constancias",
     "«se advierte de las constancias»"),
    (r"en\s+los\s+t[ée]rminos\s+que\s+obran\s+en\s+autos", "«en los términos que obran en autos»"),
    (r"(?:la|las|el|los)\s+(?:fechas?|n[úu]meros?|datos?|nombres?|cantidad|monto)\s+que\s+"
     r"(?:obra|obran|consta|constan)\s+en\s+(?:autos|las\s+constancias|el\s+expediente)",
     "«… que obra en autos» en lugar del dato"),
    (r"(?:auto|acuerdo|prove[íi]do)\s+(?:de\s+fecha\s+)?que\s+obra\s+en\s+autos", "«el auto que obra en autos» sin su fecha"),
    (r"la\s+persona\s+a\s+quien(?:es)?\s+resulte?n?\b", "«la persona a quien resulta…»"),
    (r"no\s+(?:consta|constan|se\s+precisa|se\s+indica|se\s+advierte)\s+en\s+los\s+datos\s+proporcionados",
     "«no consta en los datos proporcionados»"),
    (r"(?:en|de|con)\s+los\s+datos\s+proporcionados(?!\s+por)", "«los datos proporcionados»"),
    (r"oficial[íi]a\s+(?:de\s+partes\s+)?correspondiente", "«la oficialía correspondiente»"),
    (r"(?:conforme\s+a|en\s+t[ée]rminos\s+de|seg[úu]n)\s+la\s+ley\s+del\s+acto(?!\s*,?\s*(?:en\s+su\s+|y\s+su\s+)?art[íi]culo)",
     "«conforme a la ley del acto» sin el precepto"),
)
_EVASIVAS = tuple((re.compile(p, re.I), d) for p, d in _EVASIVAS)

_RX_MP_OMITIO = re.compile(
    r"(?:omiti[óo]|no\s+formul[óo]|fue\s+omis[oa]\s+en|se\s+abstuvo\s+de|no\s+present[óo])\s+(?:formular\s+)?pedimento|"
    r"sin\s+que\s+(?:el\s+agente[^.;]{0,90}?)?formulara\s+pedimento", re.I)
_RX_MP_FORMULO = re.compile(r"(?<!no\s)(?<!sin\s)formul[óo]\s+(?:su\s+|el\s+)?pedimento", re.I)
_RX_OFICIALIA_TCC = re.compile(
    r"Oficial[íi]a\s+de\s+Partes(?:\s+Com[úu]n)?\s+de\s+este\s+(?:Tribunal|[óo]rgano)", re.I)


def _regla_evasivas(t, secs, ficha, tipo, out):
    vistos = set()
    for s in secs:
        if "antecedentes" in s["etiquetas"]:
            continue
        tramos = []      # una frase evasiva, un aviso: dos patrones sobre el mismo tramo no se suman
        for rx, desc in _EVASIVAS:
            for m in rx.finditer(s["texto"]):
                if any(m.start() < b and a < m.end() for a, b in tramos):
                    continue
                tramos.append((m.start(), m.end()))
                clave = (s["nombre"], desc)
                if clave in vistos:
                    continue
                vistos.add(clave)
                out.append(("b", f"FÓRMULA EVASIVA EN {s['nombre'].upper()}: {desc} — "
                                 f"«{_extracto(s['texto'], m.start(), m.end(), 60)}». Ocupa el sitio de un "
                                 f"dato: o se escribe el dato (fecha, número, nombre, precepto) o se deja "
                                 f"el hueco con su aviso."))
    # EL RESULTANDO DE DERECHOS SIN UN SOLO ARTÍCULO (57 de 72 AD): «alegó la
    # violación de los artículos constitucionales que precisó en su demanda».
    # El dato es la lista de artículos; sin un número no hay dato.
    for s in _de(secs, zona="resultando", etiqueta="derechos"):
        cuerpo = re.sub(r"^\s*(?:" + _ORD + r"|\d{1,3})\s*\.\s*[^.]{0,120}\.", "", s["texto"])
        if not re.search(r"\d", cuerpo):
            out.append(("b", f"PERÍFRASIS EN {s['nombre'].upper()}: no dice qué artículos de la Constitución "
                             f"señaló la parte quejosa — «{_extracto(cuerpo, 0, min(len(cuerpo), 120), 0)}». Se "
                             f"leen del capítulo de derechos violados de la demanda; si no se tienen, hueco y aviso."))
    # EL MINISTERIO PÚBLICO SÓLO SE NOMBRA SI CONSTA (71 de 72 AD lo afirmaban
    # porque la plantilla lo exigía). Sin ficha no hay con qué cotejar.
    if _es_ficha(ficha):
        mp = str(ficha.get("ministerio_publico") or "").strip().lower()
        for s in _de(secs, zona="resultando"):
            m = _RX_MP_OMITIO.search(s["texto"])
            if m and mp != "sin_pedimento":
                motivo = ("la ficha dice que SÍ hubo pedimento" if mp == "pedimento"
                          else "nadie lo leyó ni lo confirmó en la ficha de trámite")
                out.append(("b", f"AFIRMACIÓN SIN FUENTE EN {s['nombre'].upper()}: el Ministerio Público "
                                 f"«{_un_renglon(m.group(0))}», y {motivo}. Si no consta, el resultando no "
                                 f"dice nada del pedimento."))
                break
            m2 = _RX_MP_FORMULO.search(s["texto"])
            if m2 and mp == "sin_pedimento":
                out.append(("b", f"CONTRADICCIÓN EN {s['nombre'].upper()}: el resultando dice que el "
                                 f"Ministerio Público formuló pedimento y la ficha de trámite dice que no."))
                break
    # LA OFICIALÍA DE PARTES DE ESTE TRIBUNAL, INVENTADA (16 de 72 AD). En el
    # directo la demanda se presenta por conducto de la responsable (art. 176);
    # sólo si el secretario declaró que llegó aquí, se dice.
    if tipo == "amparo_directo":
        via = str(_g(ficha, "via_presentacion") or "").lower() if isinstance(ficha, dict) else ""
        if via != "tribunal":
            for s in _de(secs, zona="resultando", etiqueta="presentacion"):
                m = _RX_OFICIALIA_TCC.search(s["texto"])
                if m:
                    out.append(("b", f"OFICIALÍA INVENTADA EN {s['nombre'].upper()}: «{_un_renglon(m.group(0))}». "
                                     f"La demanda de amparo directo se presenta por conducto de la autoridad "
                                     f"responsable (art. 176 de la Ley de Amparo); si de verdad llegó a este "
                                     f"tribunal, que lo declare el secretario en la ficha de trámite."))
                    break


# ═══════════════════════════════════════════════════════════════════════════
# (c) FECHAS DE LOS RESULTANDOS QUE NO ESTÁN EN LA FICHA NI EN EL ACTO
# ═══════════════════════════════════════════════════════════════════════════
# Las fuentes: la ficha de trámite entera y, de `datos`, el texto del acto y lo
# que tecleó o confirmó el secretario. NO los antecedentes ni el estudio: una
# fecha inventada allí no avala la del resultando.
_CLAVES_FUENTE = ("acto", "presentacion", "notificacion", "procesal", "tramite", "autos",
                  "texto_autos", "auto_admision", "auto_turno", "fechas_autos",
                  "fecha_lista", "fecha_sesion", "ficha_procesal")


def _regla_fechas(t, secs, ficha, datos, out):
    if not _es_ficha(ficha):
        return
    de_ficha: set = set()
    _fechas_de_valor(ficha, de_ficha)
    if not de_ficha:
        return
    conocidas = set(de_ficha)
    for k in _CLAVES_FUENTE:
        _fechas_de_valor((datos or {}).get(k), conocidas)
    for s in secs:
        if s["zona"] not in ("visto", "resultando") or "sesion" in s["etiquetas"]:
            continue
        ajenas = []
        for d, i, j, lit in _fechas(s["texto"]):
            if d is None:
                ajenas.append(f"«{lit}» (fecha imposible)")
            elif d not in conocidas:
                ajenas.append(f"«{lit}»")
        if ajenas:
            vistas = list(dict.fromkeys(ajenas))
            out.append(("c", f"FECHA SIN FUENTE EN {s['nombre'].upper()}: {', '.join(vistas[:4])}"
                             f"{' y otras' if len(vistas) > 4 else ''}. No está en la ficha de trámite ni en "
                             f"el acto: o se corrige la ficha o se quita del resultando."))


# LA FECHA DEL ACTO EN LOS RESOLUTIVOS (RF Yucatán del mapa, 3-oct-2026: «Se
# confirma la sentencia de veintidós de abril…», que era la fecha del auto de
# Presidencia; `fase_origen.fecha_de` la tomó de la prosa). El resolutivo
# repite el acto que ya individualizaron el V I S T O y los resultandos: su
# fecha tiene que estar allí (o en la ficha).
_RX_F_ACTO_RESOL = re.compile(
    r"(?:sentencia|resoluci[óo]n|laudo|auto|interlocutoria)(?:\s+(?:definitiva|interlocutoria|reclamad[oa]|"
    r"recurrid[oa]|impugnad[oa]))?\s*,?\s+(?:dictad[oa]\s+el\s+|de\s+fecha\s+|de\s+)$", re.I)


_RX_DICTO_SENTENCIA_ANTES = re.compile(r"dict[óo]\s+(?:la\s+)?(?:sentencia|resoluci[óo]n|laudo)(?:\s+definitiva)?\s+el\s+$",
                                       re.I)
_RX_DICTO_SENTENCIA_DESPUES = re.compile(r"^[^.;]{0,30}?(?:se\s+)?dict[óo]\s+(?:la\s+)?(?:sentencia|resoluci[óo]n|laudo)",
                                         re.I)


def _regla_fecha_resolutivo(t, secs, ficha, out, tipo=""):
    res = _de(secs, zona="resolutivos")
    if not res:
        return
    # SÓLO LAS FECHAS QUE SON DEL ACTO, no cualquier fecha de los resultandos:
    # en el RF del mapa la fecha equivocada era justamente la del auto de
    # Presidencia, que sí está en los resultandos.
    conocidas: set = set()
    for s in secs:
        if s["zona"] in ("visto", "resultando"):
            for d, i, j, lit in _fechas(s["texto"]):
                antes, despues = s["texto"][max(0, i - 90):i], s["texto"][j:j + 60]
                if d and (_RX_F_ACTO_RESOL.search(antes) or _RX_DICTO_SENTENCIA_ANTES.search(antes)
                          or _RX_DICTO_SENTENCIA_DESPUES.search(despues)):
                    conocidas.add(d)
    f_acto = _iso(_g(ficha, "acto", "fecha"))
    if f_acto:
        conocidas.add(f_acto)
    if not conocidas:
        return
    for s in res:
        # EL PUNTO DEL AMPARO DEL AR (C5, sexta ronda): «…consistente en la
        # resolución de quince de julio…» es el acto reclamado en el amparo
        # indirecto, no lo recurrido; su fecha no se coteja con la del acto.
        cuerpo = _sin_punto_del_amparo(s["texto"]) if tipo == "amparo_revision" else s["texto"]
        for d, i, j, lit in _fechas(cuerpo):
            if d and _RX_F_ACTO_RESOL.search(cuerpo[max(0, i - 90):i]) and d not in conocidas:
                out.append(("c", f"LA FECHA DEL ACTO EN LOS RESOLUTIVOS NO ES LA DEL ACTO: «{lit}» no es la fecha de lo "
                                 f"reclamado o recurrido que dan el V I S T O y los resultandos"
                                 f"{' ni la de la ficha' if f_acto else ''}. El resolutivo individualiza el mismo acto, "
                                 f"con su misma fecha."))
                return


# ═══════════════════════════════════════════════════════════════════════════
# (d) LA RESPONSABLE Y EL TRIBUNAL, ESCRITOS DE UNA SOLA FORMA
# ═══════════════════════════════════════════════════════════════════════════
_FIN_ORGANO = (r"(?=\s*,\s*(?:en\s+(?:el|los|la)\s+(?:toca|expediente|juicio|incidente|autos|recurso|cuaderno|"
               r"procedimiento)|localizad|con\s+(?:sede|residencia)|certeza|que\s+lo\s+registr|quien|"
               r"en\s+t[ée]rminos|en\s+el\s+que|mediante|derivad|relativ|dentro\s+del?)|"
               r"\s+en\s+(?:el|los)\s+(?:toca|expedientes?|juicio|incidente)\b|\s+dentro\s+del?\b|"
               r"\s+en\s+t[ée]rminos\b|\s*;|\.\s|\.$|\s*\n|\s*$)")
# La competencia del AR dice «una sentencia dictada en un juicio de amparo
# indirecto en materia familiar, por la Sala…» (ronda 4: el órgano descartado
# de los AR 307 y 60 seguía ahí y la fórmula no se leía).
_RX_DICTADO_POR = re.compile(
    r"\b(?:dictad[oa]|pronunciad[oa])\s+(?:el\s+(?:d[íi]a\s+)?[^,;:\n]{6,70}?,?\s+|"
    r"en\s+un\s+juicio\s+de\s+amparo\s+indirecto(?:\s+en\s+materia\s+[a-záéíóú]+)?,?\s+)?"
    r"por\s+(?P<o>[^\n;]{4,260}?)" + _FIN_ORGANO)
_RX_TRAMITADO_ANTE = re.compile(r"\btramitad[oa]\s+ante\s+(?P<o>[^\n;]{4,260}?)(?=\s*,\s*localizad)")
_RX_RENDIDO_POR = re.compile(r"\brendido\s+por\s+(?P<o>[^\n;]{4,260}?)" + _FIN_ORGANO)
_RX_REMITIO = re.compile(r"\bremiti[óo]\s+(?P<o>(?:el|la)\s+[^\n;]{4,260}?)(?=\s+en\s+t[ée]rminos|\s*,\s*en\s+t[ée]rminos|\s*;|\.\s)")
_RX_CORRESPONDIO = re.compile(
    r"\bcorrespondi[óo]\s+(?:conocer\s+(?:del\s+asunto|de\s+la\s+demanda|del\s+juicio)\s+)?"
    r"(?:,?\s*por\s+raz[óo]n\s+de\s+turno,?\s+)?(?:a\s+|al\s+)(?P<o>[^\n;]{4,260}?)"
    r"(?=\s*,\s*(?:que|el\s+que|la\s+que|quien|cuy[oa]|mism[oa]\s+que|el\s+cual|la\s+cual)\b|\s+(?:que|quien)\s+(?:lo|la)\s+registr)")
_RX_CARATULA_ORGANO = re.compile(
    r"^[ \t]*(?:AUTORIDAD(?:ES)?\s+RESPONSABLES?|[ÓO]RGANO\s+RECURRIDO|AUTORIDAD\s+RECURRIDA|"
    r"[ÓO]RGANO\s+QUE\s+DICT[ÓO]\s+EL\s+AUTO\s+RECURRIDO|SALA\s+RESPONSABLE)\s*:\s*(?P<o>[^\n]+?)\s*\.?\s*$",
    re.M)
_RX_BLOQUE_RESPONSABLE = re.compile(
    r"AUTORIDAD(?:ES)?\s+RESPONSABLES?\s*:\s*\n?\s*(?P<o>[^\n]+)")
_RX_CABEZA_ORGANO = re.compile(
    r"\b(?:sala|juzgado|tribunal|junta|magistrad[oa]|jue(?:z|za)|pleno|presidente|presidenta|"
    r"director[a]?|direcci[óo]n|secretar[íi]a|secretari[oa]|administraci[óo]n|administrador[a]?|"
    r"subprocurad\w*|procurad\w*|congreso|ayuntamiento|instituto|comisi[óo]n|unidad|delegaci[óo]n|"
    r"delegad[oa]|titular|registrador[a]?|gobernador[a]?|presidencia|coordinaci[óo]n|jef[ea])\b")
_RX_GENERICO = re.compile(
    r"\b(?:responsable|natural|de\s+origen|citad[oa]|aludid[oa]|referid[oa]|mencionad[oa]|dich[oa]|"
    r"precitad[oa]|indicad[oa]|un|una|este|esta|ese|esa)\b")


def _clave_organo(x: str) -> str:
    p = _plano(x)
    p = re.sub(r"\((?:ordenadora|ejecutora)\)", " ", p)
    p = re.sub(r"\bde\s+el\b", "del", p)
    p = re.sub(r"^(?:el|la|los|las|al|del)\s+", "", p)
    # EL PREFIJO TEMPORAL NO HACE OTRO ÓRGANO (ronda 4, 3-oct-2026). RF 49 del
    # banco: la carátula ya sin él («SALA REGIONAL EN QUERÉTARO…») y el dato
    # con él («actual Sala Regional en Querétaro»); la verja decía que la Sala
    # «no se escribe igual». De «la entonces X, ahora Y» (o «(actualmente Y)»)
    # cuenta Y, el nombre de hoy, que es el que trae la ficha.
    m = re.match(r"^entonces\s+.+?[\s,(]+(?:ahora|hoy|actualmente)\s+(?:la\s+|el\s+)?(.+?)\)?$", p)
    if m:
        p = m.group(1)
    p = re.sub(r"^(?:actual|otrora|hoy)\s+", "", p)
    # EL TITULAR Y SU ÓRGANO SON EL MISMO DATO (ronda 3, 3-oct-2026). La queja
    # se escribe ahora con el órgano («el Juzgado Séptimo de Distrito…») aunque
    # la ficha lo haya leído por su titular («Juez Séptimo…», «JUEZA TERCERA…»,
    # «Titular del Juzgado Cuarto…», Q 300 del banco): no es otro órgano. Con 8
    # avisos falsos en los casos del compositor de la queja.
    p = re.sub(r"^titular\s+del\s+juzgado\b", "juzgado", p)
    p = re.sub(r"^jue(?:z|za)\b", "juzgado", p)
    if p.startswith("juzgado "):
        p = re.sub(r"\b(primer|segund|tercer|cuart|quint|sext|septim|octav|noven|decim)a\b", r"\1o", p)
    p = re.sub(r",?\s*(?:con\s+(?:sede|residencia)|residente|ubicad[oa]|localizad[oa])\b.*$", "", p)
    p = re.sub(r"[^\w\s]", " ", p)
    return re.sub(r"\s+", " ", p).strip()


def _parece_organo(x: str) -> bool:
    p = _plano(x)
    if "*" in x or len(p.split()) < 3:
        return False
    if not _RX_CABEZA_ORGANO.search(p):
        return False
    return not _RX_GENERICO.search(p)


def _tramo_en_versales(x: str) -> str:
    """El nombre en versales con que empieza una mención («la MAGISTRADA LAURA…, y con…»), o ""."""
    m = re.match(r"^\s*(?:(?:el|la|los|las)\s+)?((?:[A-ZÁÉÍÓÚÑÜ][A-ZÁÉÍÓÚÑÜ.]+[\s,]+){2,}[A-ZÁÉÍÓÚÑÜ][A-ZÁÉÍÓÚÑÜ.]+)",
                 str(x or ""))
    if not m or len(m.group(1).split()) < 3:
        return ""
    return m.group(1) if _RX_CABEZA_ORGANO.search(_plano(m.group(1))) else ""


def _nombre_en_versales(x: str) -> str:
    """El nombre de una persona (o de una moral) en versales con que empieza la
    mención («GABRIEL REYES ALAMO, contra…»), o "". Dos palabras al menos, de
    tres letras o más; las siglas solas («el IMSS», «INFONAVIT») no cuentan."""
    m = re.match(r"^\s*(?:(?:el|la|los|las)\s+)?((?:[A-ZÁÉÍÓÚÑÜ][A-ZÁÉÍÓÚÑÜ.]*[ \t]+){1,12}[A-ZÁÉÍÓÚÑÜ][A-ZÁÉÍÓÚÑÜ.]*)",
                 str(x or ""))
    if not m:
        return ""
    largas = [w for w in m.group(1).split() if len(w.strip(".")) >= 3 and not re.match(r"^[IVXLC]+$", w)]
    # «PARTE QUEJOSA», «PARTE RECURRENTE Y OTRA»: la figura, no un nombre (así
    # quedan los nombres testados del banco de oráculo).
    if not (_palabras(m.group(1)) and set(_palabras(m.group(1))) - _GENERICAS_PARTE - _PARTICULAS_REPR):
        return ""
    return m.group(1) if len(largas) >= 2 else ""


# LA RACHA EN VERSALES DENTRO DE LA MENCIÓN (ronda 4, 3-oct-2026). El paso a
# prosa sólo convertía el nombre que llegaba ENTERO en mayúsculas: con la forma
# del auto («Sucesión a Bienes de JOSÉ GARCÍA RUIZ», AD 335), «JUAN y PEDRO,
# ambos de apellidos PÉREZ LÓPEZ» (AR 208) o «la sucesión intestamentaria a
# bienes de JUAN PÉREZ» (AR 60), el nombre se quedaba en versales en el V I S
# T O, los resultandos, la legitimación y el resolutivo, y la regla sólo miraba
# el principio de la mención. Ahora se busca la primera racha de dos o más
# palabras en versales (de tres letras o más; sin siglas con punto —«S.A.»,
# «C.T.M.»— ni romanos) en las primeras palabras de la mención.
_RX_TOKEN_VERSAL = re.compile(r"^[A-ZÁÉÍÓÚÑÜ][A-ZÁÉÍÓÚÑÜ.\-]*$")
_GENERICAS_BANCO = {"testado", "representante", "quien"}


def _es_larga(w: str) -> bool:
    limpio = w.strip(".-")
    if "." in limpio or re.match(r"^[IVXLCDM]+$", limpio):
        return False
    return len(re.sub(r"[^A-ZÁÉÍÓÚÑÜ]", "", limpio)) >= 3


def _racha_en_versales(x: str, max_palabras: int = 16, ultima: bool = False) -> str:
    """La racha en versales de un nombre dentro de las primeras (o, con `ultima`,
    las últimas) `max_palabras` palabras de la mención, o ""."""
    toks = str(x or "").split()
    toks = toks[-max_palabras:] if ultima else toks[:max_palabras]
    rachas, actual = [], []
    for w in toks + [""]:
        limpio = w.strip(".,;:()«»\"“”'")
        if limpio and _RX_TOKEN_VERSAL.match(limpio) and any(c.isalpha() for c in limpio):
            actual.append(w.rstrip(",;:"))
            continue
        if actual:
            rachas.append(actual)
        actual = []
    buenas = []
    for r in rachas:
        largas = [w for w in r if _es_larga(w.strip(",;:()«»\"“”'"))]
        propias = set(_palabras(" ".join(largas))) - _GENERICAS_PARTE - _PARTICULAS_REPR - _GENERICAS_BANCO
        if len(largas) >= 2 and propias:
            buenas.append(" ".join(r).strip(" ,;:"))
    if not buenas:
        return ""
    return buenas[-1] if ultima else buenas[0]


# Dónde acaba el nombre tras «promovido por», «ampara y protege a»…: lo que
# viene después («contra la sentencia dictada por la SEGUNDA SALA…») ya no es
# la parte.
_RX_FIN_MENCION = re.compile(r",?\s+(?:contra|en\s+contra|respecto|por\s+los\s+motivos|para\s+los\s+efectos|"
                             r"en\s+t[ée]rminos|quien|mediante|en\s+el\s+(?:juicio|toca|expediente))\b")
# «ampara y protege a X», «no ampara ni protege a X».
_RX_PROTEGE_A = re.compile(r"\bampara\s+(?:y|ni)\s+protege\s+a\s+(?=(?P<o>[^\n;]{4,240}))")
# «X promovió/promovieron juicio…», «X interpuso/interpusieron recurso…»: el
# nombre va ANTES del verbo.
_RX_VERBO_DE_PARTE = re.compile(r"\b(?:promovi(?:ó|o|eron)|interpus(?:o|ieron))\s+(?:el\s+)?(?:juicio|demanda|"
                                r"recurso|amparo|revisi|queja)", re.I)


def _mencion_hasta_el_fin(x: str) -> str:
    m = _RX_FIN_MENCION.search(x)
    return x[:m.start()] if m else x


# «lo hizo valer X», «interpuesto por X», «promovido por X» (ronda 3).
# La mención se toma con una mirada adelante (no se consume): «fue interpuesto
# por parte legítima…, pues lo hizo valer el JEFA…» tiene dos en la misma
# frase, y la primera no debe tragarse la segunda (RF 2 del banco).
_RX_PROMOVIDO_POR = re.compile(
    r"\b(?:lo\s+hizo\s+valer|(?:interpuest|promovid)[oa]s?\s+por)\s+(?=(?P<o>[^\n;]{4,200}))")
# El paréntesis de la carátula, entero y sólo con el rol (no «(aquí quejosa)»
# ni el «(ordenadora)/(ejecutora)» del bloque de la autoridad responsable).
# SÓLO EN VERSALES, o con la mayúscula a media frase que deja el paso de
# versales a prosa («(autoridad Responsable)», AR 208 del banco): el tribunal
# escribe en sus sentencias «Fulano (tercera interesada)» y «(Quejosa)» como
# aclaración del relato (Q 416 y AD 742 de la calibración), y eso no se acusa.
_RX_ETIQUETA_ROL = re.compile(
    r"\((?:autoridad(?:es)?\s+responsables?|autoridad\s+demandada|demandad[oa]s?|actor(?:a|es|as)?|"
    r"quejos[oa]s?|tercer[oa]s?\s+interesad[oa]s?|recurrentes?|parte\s+(?:actora|demandada|quejosa))\)", re.I)


def _etiqueta_de_caratula(m) -> bool:
    dentro = m.group(0)[1:-1]
    palabras = dentro.split()
    return dentro.isupper() or any(w[:1].isupper() for w in palabras[1:])


# El bloque «ACTO(S) RECLAMADO(S):» copia la demanda tal cual (spans del papel):
# lo que diga allí no es prosa del proyecto.
_RX_BLOQUE_RECLAMADO = re.compile(r"ACTOS?\s+RECLAMADOS?\s*:")


def _sin_bloque_reclamado(s) -> str:
    cuerpo = s["texto"]
    if s["zona"] == "resultando" and "presentacion" in s["etiquetas"]:
        return _RX_BLOQUE_RECLAMADO.split(cuerpo)[0]
    return cuerpo


# EL PUNTO DEL AMPARO NOMBRA ACTO Y AUTORIDAD (sexta ronda, 3-oct-2026, C5;
# David: «información valiosa para el lector»). El resolutivo que confirma en el
# amparo en revisión dice ahora «…respecto del acto que reclamó de la Segunda
# Sala Civil…, consistente en la sentencia dictada el …» o «…contra los actos
# que reclamó de X y de Y, precisados en el resultando primero…»: lo que viene
# ahí es el acto y la autoridad DEL AMPARO INDIRECTO —como el bloque «ACTOS
# RECLAMADOS:» del primer resultando—, no el juzgado que dictó lo recurrido ni
# la clase de lo recurrido. Se quita hasta el final del renglón (cada punto es
# su párrafo) antes de buscar el órgano (d) o la clase del acto (l).
_RX_PUNTO_DEL_AMPARO = re.compile(
    r"\b(?:respecto\s+(?:del|de\s+(?:el|los))|contra\s+(?:el|los))\s+actos?\s+(?:que\s+reclam|precisad)[^\n]*",
    re.I)


def _sin_punto_del_amparo(cuerpo: str) -> str:
    return _RX_PUNTO_DEL_AMPARO.sub(" ", str(cuerpo or ""))


# «X, EN REPRESENTACIÓN DE Y, EN REPRESENTACIÓN DE Y» Y «X, EN REPRESENTACIÓN DE
# X» (ronda 3, banco de oráculo de la revisión fiscal): el promovente de la
# carátula ya traía «EN REPRESENTACIÓN DEL SUBDELEGADO…» y el compositor le
# añadía otra vez la cola (RF 2, 7 y 6 de 2025); y cuando recurre la propia
# autoridad demandada salía «el Jefe del Departamento de Pensiones…, en
# representación del Jefe del Departamento de Pensiones…» (RF 21-2025 y
# 6-2026). La unidad jurídica que representa al Instituto del que forma parte
# («el Jefe de la Unidad Jurídica del IMSS, en representación del Instituto
# Mexicano del Seguro Social») no es repetición: la representada es otra.
_RX_EN_REPR = re.compile(
    r"\ben\s+(?:nombre\s+y\s+)?(?:representaci[óo]n|nombre)\s+(?:del?|de\s+la|de\s+los|de\s+las)\s+", re.I)
# Dónde acaba el nombre de la representada: el nombre lleva comas por dentro
# («Subdelegado de Prestaciones Económicas, Delegación Estatal en Baja
# California Sur, …»), así que se corta en el verbo o en el final de la frase.
_RX_FIN_REPRESENTADA = re.compile(
    r"\.\s|;|\n|,\s+(?:interpus|interpone|promovi|present[óo]|hizo\s+valer|en\s+su\s+car[áa]cter|por\s+conducto|"
    r"quien|lo\s+|la\s+cual|el\s+cual|unidad\s+administrativa)|\ben\s+(?:nombre\s+y\s+)?(?:representaci[óo]n|nombre)\s+de",
    re.I)
_PARTICULAS_REPR = {"de", "del", "la", "las", "los", "el", "y", "e", "en", "a", "al", "por", "su"}


def _palabras(x: str) -> list:
    return re.findall(r"[a-z0-9ñ]+", _plano(x))


def _sig(palabras: list, n: int) -> list:
    return [w for w in palabras if w not in _PARTICULAS_REPR][:n]


def _representada(cuerpo: str, fin: int) -> str:
    tramo = cuerpo[fin:fin + 240]
    m = _RX_FIN_REPRESENTADA.search(tramo)
    return tramo[:m.start()] if m else tramo


def _se_parecen(a: list, b: list, umbral: float = 0.7) -> bool:
    """¿El nombre b (palabras significativas) es el que empieza en a? La misma
    cabeza y al menos el 70 % de sus palabras. Para dos «en representación de»
    seguidos en la misma frase, que ya es el defecto (RF 7: la segunda copia
    corrige «Califroniasur»)."""
    if len(b) < 2 or not a or a[0] != b[0]:
        return False
    return sum(1 for w in b if w in set(a)) / len(b) >= umbral


def _subsecuencia(corta: list, larga: list) -> bool:
    it = iter(larga)
    return all(w in it for w in corta)


def _mismo_nombre(x: list, y: list) -> bool:
    """«X, en representación de Y» con X y Y el mismo órgano: la misma cabeza y
    la misma palabra siguiente («Subdelegado de Prestaciones…», «Jefe del
    Departamento…»), y uno es el otro con palabras de más (RF 6-2026: «…de
    Prestaciones Económicas, Delegación…» y «…de Prestaciones, Delegación…»).
    «el Titular de la Unidad Jurídica de la Delegación…, en representación del
    Titular de la Delegación…» NO lo es: la segunda palabra ya es otra (unidad
    / delegación), y es justo la unidad que representa a la autoridad."""
    if len(x) < 2 or len(y) < 2 or x[:2] != y[:2]:
        return False
    return _subsecuencia(y[:8], x) or _subsecuencia(x[:8], y)


# EL CARGO Y SU ÓRGANO SON LA MISMA AUTORIDAD (ronda 4, 3-oct-2026, E8). En la
# RF 6-2026 del banco, «el Subdelegado de Prestaciones Económicas…, en
# representación de la Subdelegación de Prestaciones Económicas…» es X en
# representación de X, y la regla no lo veía porque «subdelegado» no es
# «subdelegación» (el engrose mismo usa las dos: la carátula
# «SUBDELEGACIÓN», el V I S T O «el Subdelegado»). Se comparan por su forma de
# órgano; «Titular de la X» es X, y «Jefe/Jefa de la Unidad…, del
# Departamento…» es la unidad o el departamento.
_EQUIV_CARGO = {}
for _org, _formas in (("subdelegacion", ("subdelegado", "subdelegada")), ("delegacion", ("delegado", "delegada")),
                      ("subdireccion", ("subdirector", "subdirectora")), ("direccion", ("director", "directora")),
                      ("jefatura", ("jefe", "jefa")), ("subjefatura", ("subjefe", "subjefa")),
                      ("coordinacion", ("coordinador", "coordinadora")),
                      ("subadministracion", ("subadministrador", "subadministradora")),
                      ("administracion", ("administrador", "administradora")),
                      ("presidencia", ("presidente", "presidenta")),
                      ("procuraduria", ("procurador", "procuradora")),
                      ("subprocuraduria", ("subprocurador", "subprocuradora")),
                      ("tesoreria", ("tesorero", "tesorera")), ("contraloria", ("contralor", "contralora")),
                      ("gubernatura", ("gobernador", "gobernadora"))):
    for _f in _formas:
        _EQUIV_CARGO[_f] = _org
_ORGANO_TRAS_JEFE = {"unidad", "departamento", "oficina", "area", "division", "seccion", "servicios",
                     "subdireccion", "direccion", "coordinacion", "delegacion", "subdelegacion", "administracion",
                     "subadministracion", "oficialia", "dependencia", "representacion"}


def _canon(pal: list) -> list:
    return [_EQUIV_CARGO.get(w, w) for w in pal]


def _nucleo(sig: list) -> list:
    """«Titular de la X» → X; «Jefe de la Unidad X» → «Unidad X»."""
    if len(sig) > 1 and sig[0] == "titular":
        sig = sig[1:]
    if len(sig) > 1 and sig[0] == "jefatura" and sig[1] in _ORGANO_TRAS_JEFE:
        sig = sig[1:]
    return sig


def _mismo_organo(x: list, y: list) -> bool:
    """Como `_mismo_nombre`, con el cargo y el órgano igualados; además, con las
    tres primeras palabras iguales, el nombre corto entero dentro del largo
    («Subdelegación de Prestaciones Económicas del ISSSTE en Querétaro» frente a
    «Subdelegado de Prestaciones Económicas, Delegación Estatal Querétaro, del
    ISSSTE»: el orden de la sede cambia, la autoridad no)."""
    x, y = _nucleo(_canon(x)), _nucleo(_canon(y))
    if _mismo_nombre(x, y):
        return True
    if len(x) < 3 or len(y) < 3 or x[:3] != y[:3]:
        return False
    corta, larga = (x, y) if len(x) <= len(y) else (y, x)
    return all(w in set(larga) for w in corta[:8])


def _sujeto_es(antes: str, y: str) -> bool:
    """¿El nombre `y` es también el sujeto que va antes («X, en representación
    de Y», «X, por conducto de su Y»)? Se busca su cabeza tras un artículo o un
    verbo, no tras «de» («la Unidad Jurídica del Instituto…, en representación
    del Instituto…» es otra cosa); el sujeto acaba en «por conducto»."""
    corte = max(antes.rfind(". "), antes.rfind(";"), antes.rfind("\n"))
    pal = _palabras(antes[corte + 1:] if corte >= 0 else antes)
    sig_y = _nucleo(_canon(_sig(_palabras(y), 12)))
    if len(sig_y) < 2:
        return False
    for i, w in enumerate(pal):
        cw = _EQUIV_CARGO.get(w, w)
        if not (cw == sig_y[0] or w == "titular" or cw == "jefatura"):
            continue
        prev = pal[i - 1] if i else ""
        prev2 = pal[i - 2] if i > 1 else ""
        if prev in ("de", "del", "y", "e") or (prev in ("la", "las", "los", "el") and prev2 in ("de", "y")):
            continue
        x = _sig(pal[i:], 40)
        if "conducto" in x:
            x = x[:x.index("conducto")]
        if _mismo_organo(x, sig_y):
            return True
        if _misma_autoridad_por_nucleo(pal[i:], y):
            return True
    return False


# UNA SOLA COMPARACIÓN DE AUTORIDADES (quinta ronda, 3-oct-2026, F4; RF 7 del
# banco con el nombre real, vf2_rf/t1). El resultando decía «Inconforme, la Jefa
# de la Unidad Jurídica de la DELEGACIÓN Estatal…, por conducto de su Jefa de la
# Unidad Jurídica de la REPRESENTACIÓN Estatal…, María Luisa Gómez Ríos» y la
# legitimación del mismo documento, «lo hizo valer María Luisa Gómez Ríos, Jefa
# de la Unidad Jurídica de la Delegación Estatal…»: la misma unidad con su
# adscripción escrita de dos formas. `tipos_asunto.misma_autoridad` ya la daba
# por la misma (el núcleo, «jefatura de la unidad juridica», es igual), pero el
# compositor y esta verja comparaban palabra por palabra y no la veían: ni el
# resultando se arreglaba ni salía el aviso. Ahora la verja pregunta lo mismo
# que el compositor y la legitimación, con la misma función.
def _misma_autoridad_por_nucleo(pal_sujeto: list, y: str) -> bool:
    """¿El sujeto (sus palabras, desde la cabeza hasta «por conducto») y `y` son
    la misma autoridad según `tipos_asunto.misma_autoridad`? Sin el módulo,
    False: la regla vieja sigue sola."""
    try:
        import tipos_asunto as _ta
        _misma = getattr(_ta, "misma_autoridad")
    except Exception:
        return False
    pal = list(pal_sujeto)
    if "conducto" in pal:
        pal = pal[:pal.index("conducto")]
        if pal and pal[-1] == "por":
            pal = pal[:-1]
    if len(pal) < 3 or len(_palabras(y)) < 3:
        return False
    try:
        return bool(_misma(" ".join(pal), str(y or "")))
    except Exception:
        return False


# «X, POR CONDUCTO DE SU X» (ronda 4, 3-oct-2026; banco, RF 7 tal cual y RF 2 y
# 26 con la ficha corregida): cuando quien firma es la titular de la misma
# unidad que recurre, el compositor escribía «la Jefa de la Unidad Jurídica de
# la Delegación…, por conducto de su Jefa de la Unidad Jurídica, Fulana», y la
# legitimación «lo hizo valer la Jefa…, por conducto de Fulana, en su carácter
# de unidad administrativa encargada de la defensa jurídica…»: una persona con
# carácter de unidad administrativa. El corpus: «la Jefa de la Unidad Jurídica…
# interpuso» y «lo promovió Fulana, Jefa de la Unidad Jurídica…».
_RX_POR_CONDUCTO_SU = re.compile(r"\bpor\s+conducto\s+de\s+su\s+", re.I)
_RX_FIN_FIGURA = re.compile(r"[,;(\n]|\.\s|\s+en\s+(?:su\s+)?(?:representaci|car[áa]cter)|\s+quien\b|"
                            r"\s+interpus|\s+promovi", re.I)
_RX_PERSONA_COMO_UNIDAD = re.compile(
    r"\bpor\s+conducto\s+de\s+(?!su\b|sus\b|la\b|el\b|los\b|las\b|dicha\b|dicho\b)(?P<n>[^,;\n]{3,90}?),\s+"
    r"en\s+su\s+car[áa]cter\s+de\s+unidad\s+administrativa", re.I)


def _regla_representacion(secs, out):
    dichas = set()
    for s in secs:
        if s["zona"] in ("caratula", "proemio") or "antecedentes" in s["etiquetas"]:
            continue
        cuerpo = s["texto"]
        previo = None
        for m in _RX_EN_REPR.finditer(cuerpo):
            y = _representada(cuerpo, m.end())
            sig_y = _sig(_palabras(y), 8)
            # «…, en representación de Y, en representación de Y»
            if previo is not None and "repetida" not in dichas:
                entre = cuerpo[previo[0].end():m.start()]
                if len(entre) < 300 and not re.search(r"\.\s|;|\n", entre) and \
                        _se_parecen(_sig(_palabras(previo[1]), 12), sig_y):
                    dichas.add("repetida")
                    out.append(("d", f"REPRESENTACIÓN REPETIDA EN {s['nombre'].upper()}: «…en representación de "
                                     f"{_un_renglon(y)[:90]}» dos veces en la misma frase. La representada se dice "
                                     f"una vez."))
            # «X, en representación de X»: el nombre de Y empieza también antes
            # (ronda 4: con el cargo y su órgano igualados, E8).
            if "si_mismo" not in dichas and _sujeto_es(cuerpo[max(0, m.start() - 400):m.start()], y):
                dichas.add("si_mismo")
                out.append(("d", f"QUIEN RECURRE SE REPRESENTA A SÍ MISMO EN {s['nombre'].upper()}: el nombre "
                                 f"que va antes de «en representación de» y el que va después son el mismo "
                                 f"(«{_un_renglon(y)[:90]}»). Si recurre la propia autoridad demandada, se "
                                 f"escribe ella «por conducto de» la unidad jurídica que firmó el oficio (art. "
                                 f"63 LFPCA); si recurre la unidad jurídica, la representada es la autoridad "
                                 f"demandada."))
            previo = (m, y)
        # «X, por conducto de su X» (ronda 4).
        if "conducto" not in dichas:
            for m in _RX_POR_CONDUCTO_SU.finditer(cuerpo):
                tramo = cuerpo[m.end():m.end() + 200]
                mf = _RX_FIN_FIGURA.search(tramo)
                fig = tramo[:mf.start()] if mf else tramo
                if _sujeto_es(cuerpo[max(0, m.start() - 400):m.start()], fig):
                    dichas.add("conducto")
                    out.append(("d", f"QUIEN RECURRE ACTÚA POR CONDUCTO DE SU PROPIO CARGO EN {s['nombre'].upper()}: "
                                     f"«…, por conducto de su {_un_renglon(fig)[:90]}». Si firma la titular de la "
                                     f"misma unidad que recurre, la unidad va sola en el resultando («Inconforme, la "
                                     f"Jefa de la Unidad Jurídica…, interpuso…») y la persona en aposición en la "
                                     f"legitimación («lo hizo valer Fulana, Jefa de la Unidad Jurídica…»)."))
                    break
        if "persona_unidad" not in dichas:
            m = _RX_PERSONA_COMO_UNIDAD.search(cuerpo)
            if m:
                dichas.add("persona_unidad")
                out.append(("d", f"UNA PERSONA «EN SU CARÁCTER DE UNIDAD ADMINISTRATIVA» EN {s['nombre'].upper()}: "
                                 f"«…por conducto de {_un_renglon(m.group('n'))[:60]}, en su carácter de unidad "
                                 f"administrativa…». El carácter es de la unidad que recurre, no de quien firma: "
                                 f"«lo hizo valer {_un_renglon(m.group('n'))[:60]}, [cargo], unidad administrativa "
                                 f"encargada de la defensa jurídica…»."))


def _en_versales(x: str) -> bool:
    letras = [c for c in str(x or "") if c.isalpha()]
    pal = [w for w in re.findall(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]{3,}", str(x or ""))]
    if len(letras) < 15 or len(pal) < 3:
        return False
    return sum(1 for c in letras if c.isupper()) / len(letras) > 0.85


# LA COLA DEL TFJA NO ES OTRO NOMBRE. El V I S T O y el primer resultando de la
# revisión fiscal dicen «la Sala Regional del Centro II del Tribunal Federal de
# Justicia Administrativa» (SPEC §2.1 y §2.2), y la ficha guarda a veces sólo
# «SALA REGIONAL DEL CENTRO II» (así la lee `ficha_tramite` de la carátula del
# TFJA). Es la misma Sala, en los dos sentidos: con la cola en el texto y sin
# ella en el dato (el caso que midió el compositor: 2 avisos falsos por RF con
# una verja anterior), y al revés, con la cola en el dato y el texto sin ella.
# Comprobado el 3-oct-2026 sobre integ1/RF_*.txt y los 5 RF del compositor.
_RX_COLA_TFJA = re.compile(r"^del\s+tribunal\s+federal\s+de\s+justicia\s+(?:fiscal\s+y\s+)?administrativa$")


def _relacion(a: str, b: str) -> str:
    """igual | corta (a es b truncada) | larga (a lleva cola) | distinta."""
    ka, kb = _clave_organo(a), _clave_organo(b)
    if ka == kb:
        return "igual"
    wa, wb = ka.split(), kb.split()
    if len(wa) < len(wb) and wb[:len(wa)] == wa:
        if _RX_COLA_TFJA.match(" ".join(wb[len(wa):])):
            return "igual"
        return "corta"
    if len(wa) > len(wb) and wa[:len(wb)] == wb:
        cola = " ".join(wa[len(wb):])
        # «… y Juzgado Tercero (ejecutora)» es la otra responsable, y «… del
        # Tribunal Federal de Justicia Administrativa» es la fórmula del V I S T O
        # de la revisión fiscal («dictada por {sala} del Tribunal Federal…»).
        if (re.match(r"^(?:y|e)\s", cola) or re.match(r"^(?:ordenadora|ejecutora)\b", cola)
                or _RX_COLA_TFJA.match(cola)):
            return "igual"
        return "larga"
    return "distinta"


def _menciones_organo(secs, tipo):
    """[(sección, texto de la mención, ¿lugar compuesto por código?)]

    Sólo las FÓRMULAS: «dictada el …, por X, en el …», el renglón de la
    carátula, «rendido por X», «remitió X en términos», «correspondió a X, que
    lo registró». Una mención suelta («la Sala responsable», «el juzgado
    natural») no es un nombre y no se compara.
    """
    out = []
    for s in secs:
        z, cuerpo = s["zona"], s["texto"]
        if z == "caratula":
            for m in _RX_CARATULA_ORGANO.finditer(cuerpo):
                out.append((s, m.group("o"), True))
        elif z in ("visto", "resolutivos"):
            # En el AR, el punto del amparo («…respecto del acto que reclamó de
            # la Sala…, consistente en la sentencia dictada … por …») nombra la
            # autoridad del amparo, no el juzgado (C5, sexta ronda).
            m = _RX_DICTADO_POR.search(_sin_punto_del_amparo(cuerpo) if (z == "resolutivos"
                                                                          and tipo == "amparo_revision") else cuerpo)
            if m:
                out.append((s, m.group("o"), z == "resolutivos"))
        elif z == "resultando":
            if "sesion" in s["etiquetas"]:
                continue
            if tipo == "amparo_directo" and "presentacion" in s["etiquetas"]:
                m = _RX_BLOQUE_RESPONSABLE.search(cuerpo)
                if m:
                    out.append((s, re.split(r"\s+\(ordenadora\)|\s+y\s+(?=[A-ZÁÉÍÓÚ][^()]*\(ejecutora\))",
                                            m.group("o"))[0], False))
                else:
                    m = _RX_DICTADO_POR.search(cuerpo)
                    if m:
                        out.append((s, m.group("o"), False))
            elif tipo in ("amparo_revision", "revision_fiscal") and "tramite" in s["etiquetas"]:
                m = _RX_CORRESPONDIO.search(cuerpo)
                if m:
                    out.append((s, m.group("o"), False))
            elif (tipo == "queja" and "presentacion" in s["etiquetas"]
                  and not re.search(r"demanda", s["rotulo"], re.I)):
                # LA DEMANDA NO ES LA INTERPOSICIÓN (C3, sexta ronda, 3-oct-2026):
                # la queja de la fracción I abre ahora con «Demanda de amparo.»,
                # cuyas autoridades y actos («…dictada por la Sala…») son los del
                # amparo indirecto; el órgano que dictó el auto recurrido se lee
                # en la interposición del recurso.
                m = _RX_DICTADO_POR.search(cuerpo)
                if m:
                    out.append((s, m.group("o"), False))
        elif z == "considerando":
            if "competencia" in s["etiquetas"]:
                m = _RX_DICTADO_POR.search(cuerpo) or _RX_TRAMITADO_ANTE.search(cuerpo)
                if m:
                    out.append((s, m.group("o"), True))
            elif "existencia" in s["etiquetas"]:
                # En la revisión, «rendido por» es la responsable del amparo, no
                # el juzgado: del juzgado sólo cuenta «remitió X».
                m = (_RX_REMITIO.search(cuerpo) if tipo == "amparo_revision"
                     else _RX_RENDIDO_POR.search(cuerpo) or _RX_REMITIO.search(cuerpo))
                if m:
                    out.append((s, m.group("o"), True))
    return out


# LO QUE DECIDIÓ EL COMPOSITOR MANDA SOBRE LO CRUDO DE LA FICHA (ronda 4,
# 3-oct-2026, E11). En la Q 261 del banco la ficha traía el juzgado de dos
# formas («Juez Sexto de Distrito de Amparo y Juicios Federales…» en
# acto.organo y «el Juzgado Sexto de Distrito en Materia de Amparo Civil…» en
# responsable); el compositor eligió bien el nombre completo, lo escribió igual
# en todo el documento y avisó «VIENE ESCRITO DE DOS FORMAS». La verja cotejaba
# contra acto.organo, el que se descartó, y acusó tres veces un documento
# uniforme. Ahora el dato es el que el compositor dejó en `datos["procesal"]`
# (`responsable` en el directo, `juzgado` en la revisión y la queja, `sala` en
# la revisión fiscal); la ficha y los datos del encargo, sólo si no hay
# compositor.
_CLAVE_PROCESAL = {"amparo_directo": "responsable", "amparo_revision": "juzgado", "queja": "juzgado",
                   "revision_fiscal": "sala"}
# DÓNDE VA EL ÓRGANO EN CADA TIPO (sexta ronda, 3-oct-2026). El aviso decía
# siempre «carátula, V I S T O, resultandos, competencia, existencia y
# resolutivo»; desde C1, C3 y C4 sólo el directo lleva existencia y el órgano
# en la carátula (el amparo en revisión ya no lleva existencia; la queja y la
# revisión fiscal ya no llevan su renglón en el rubro).
_LUGARES_DEL_DATO = {
    "amparo_directo": "carátula, V I S T O, resultandos, competencia, existencia y resolutivo",
    "amparo_revision": "V I S T O, resultandos, competencia y resolutivo",
    "queja": "V I S T O, interposición del recurso, competencia y resolutivo",
    "revision_fiscal": "V I S T O, resultandos, competencia y resolutivo",
    "": "V I S T O, resultandos, competencia y resolutivo",
}
_LUGARES_DEL_ORGANO = {
    "amparo_revision": "V I S T O, trámite y competencia",
    "queja": "V I S T O, interposición del recurso y competencia",
    "": "V I S T O, trámite y competencia",
}


def _procesal(datos) -> dict:
    d = datos if isinstance(datos, dict) else {}
    return d.get("procesal") if isinstance(d.get("procesal"), dict) else {}


def _crudo_de_la_ficha(ficha, tipo) -> str:
    f = ficha if isinstance(ficha, dict) else {}
    v = {"amparo_directo": f.get("responsable"), "amparo_revision": _g(f, "acto", "organo"),
         "queja": _g(f, "acto", "organo"), "revision_fiscal": f.get("sala")}.get(tipo)
    return _un_renglon(v) if isinstance(v, str) and "*" not in v else ""


def _organo_descartado(ficha, datos, tipo) -> str:
    """El órgano que el compositor DESCARTÓ (AR 307, 448 y 60 del banco: la
    ficha dio por juzgado una autoridad responsable de la demanda), o "".

    Lo dice `procesal.juzgado_descartado` (u `organo_descartado`) si el
    compositor lo expone; si no, se deduce: el compositor declaró la clave
    (`juzgado` vacío o en hueco) y la ficha sí traía un valor."""
    proc = _procesal(datos)
    for k in ("juzgado_descartado", "organo_descartado"):
        v = proc.get(k)
        if isinstance(v, (list, tuple)):
            v = next((x for x in v if isinstance(x, str) and x.strip()), "")
        if isinstance(v, str) and v.strip() and "*" not in v:
            return _un_renglon(v)
    clave = _CLAVE_PROCESAL.get(tipo)
    if not clave or clave not in proc:
        return ""
    v = proc.get(clave)
    if isinstance(v, str) and v.strip() and "*" not in v:
        return ""
    return _crudo_de_la_ficha(ficha, tipo)


def _organo_canonico(ficha, datos, tipo) -> str:
    d = datos or {}
    proc = _procesal(d)
    f = ficha if isinstance(ficha, dict) else {}
    clave = _CLAVE_PROCESAL.get(tipo)
    if clave and clave in proc:
        v = proc.get(clave)
        if isinstance(v, str) and v.strip() and "*" not in v:
            return v.strip()
        # EL COMPOSITOR LO DESCARTÓ: no se coteja contra lo descartado (eso
        # lo dice `_regla_organo_descartado`), ni contra un `organo_recurrido`
        # que trae el mismo nombre por otro camino.
        if _organo_descartado(f, d, tipo):
            return ""
    if tipo == "amparo_directo":
        cands = [f.get("responsable"), proc.get("responsable"), d.get("responsable")]
    elif tipo in ("amparo_revision", "queja"):
        cands = [_g(f, "acto", "organo"), proc.get("juzgado"), d.get("organo_recurrido")]
    elif tipo == "revision_fiscal":
        cands = [f.get("sala"), proc.get("sala"), d.get("organo_recurrido")]
    else:
        cands = []
    for c in cands:
        if isinstance(c, str) and c.strip() and "*" not in c:
            return c.strip()
    return ""


def _regla_organo_descartado(secs, ficha, datos, tipo, out):
    """EL ÓRGANO DESCARTADO QUE SIGUE EN EL TEXTO (ronda 4; rev_3, robustez de
    los AR 307, 448 y 60 con las fichas viejas). El compositor descartó el
    juzgado que traía la ficha —era la Sala Familiar o el Juez Octavo
    Familiar, autoridades responsables del amparo, no quien dictó la
    sentencia recurrida— y dejó el hueco en el V I S T O y el trámite; pero la
    competencia y la existencia lo seguían nombrando, por `organo_recurrido`.
    El documento afirma ahí lo que el compositor ya sabía falso."""
    desc = _organo_descartado(ficha, datos, tipo)
    if not desc or len(_clave_organo(desc).split()) < 2:
        return
    donde, ejemplo = [], ""
    for s, x, _cod in _menciones_organo(secs, tipo):
        x = x.strip(" ,.;")
        if "*" in x or not x:
            continue
        if _relacion(x, desc) in ("igual", "corta", "larga") and s["nombre"] not in donde:
            donde.append(s["nombre"])
            ejemplo = ejemplo or x
    if donde:
        # SIN «Y EXISTENCIA» EN EL AR (C1, sexta ronda, 3-oct-2026): la revisión
        # ya no lleva ese considerando; nombrarlo mandaba a buscar un apartado
        # que el proyecto no tiene.
        out.append(("d", f"EL ÓRGANO QUE SE DESCARTÓ SIGUE EN EL TEXTO ({', '.join(donde)}): «{_un_renglon(ejemplo)}». "
                         f"La ficha lo daba como el que dictó lo recurrido y es una autoridad responsable de la "
                         f"demanda (el acto reclamado, no la sentencia de amparo); por eso va en hueco en el V I S T "
                         f"O. Tiene que ir el mismo juzgado en {_LUGARES_DEL_ORGANO.get(tipo, _LUGARES_DEL_ORGANO[''])} "
                         f"(acto.organo)."))


def _regla_responsable(t, secs, ficha, datos, tipo, out):
    menciones = [(s, x.strip(" ,.;"), cod) for s, x, cod in _menciones_organo(secs, tipo)]
    canon = _organo_canonico(ficha, datos, tipo)
    # SÓLO CONTRA EL DATO. Calibrado sobre 41 sentencias públicas: el tribunal
    # abrevia a mano después de la primera mención («la Sala Regional», «el
    # juzgado segundo administrativo», «…en el Estado»), y compararlas entre sí
    # acusaba 18 veces a sentencias buenas. En el proyecto compuesto, en cambio,
    # TODOS los sitios salen del mismo dato (la ficha, o el campo del encargo
    # en el camino viejo): cualquier diferencia con él es una fuente que se
    # coló —la carátula del extractor contra el resultando del modelo (64 de 72
    # AD)—. Sin ese dato, no se compara.
    cotejar = []
    if canon:
        de_ficha = _es_ficha(ficha) and bool(_organo_canonico(ficha, {}, tipo))
        _clave_p = _CLAVE_PROCESAL.get(tipo, "")
        if _clave_p and canon == str(_procesal(datos).get(_clave_p) or "").strip():
            ref_de = f"lo que decidió el compositor con la ficha de trámite ({_clave_p})"
        else:
            ref_de = "la ficha de trámite" if de_ficha else "los datos del asunto"
        ref = canon
        cotejar = [(s, x, cod) for s, x, cod in menciones if _parece_organo(x)]
    figura = {"amparo_directo": "LA AUTORIDAD RESPONSABLE", "revision_fiscal": "LA SALA",
              }.get(tipo, "EL ÓRGANO QUE DICTÓ LO RECURRIDO")
    dichos = set()
    for s, x, cod in cotejar:
        rel = _relacion(x, ref)
        if rel == "igual":
            continue
        clave = (_clave_organo(x), s["nombre"])
        if clave in dichos:
            continue
        dichos.add(clave)
        if rel == "corta":
            porque = "le falta el final del nombre (la entidad o la sede)"
        elif rel == "larga":
            porque = "le sobra texto al final del nombre"
        else:
            porque = "es otro órgano o el nombre está mal leído"
        out.append(("d", f"{figura} NO SE ESCRIBE IGUAL EN TODO EL DOCUMENTO: en {s['nombre']} dice «{_un_renglon(x)}» y en "
                         f"{ref_de} «{_un_renglon(ref)}»; {porque}. Es el mismo dato en "
                         f"{_LUGARES_DEL_DATO.get(tipo, _LUGARES_DEL_DATO[''])}: tiene que decir lo mismo en todos."))
    # EL DATO MISMO, SIN ENTIDAD: «Sala Familiar del Tribunal Superior de
    # Justicia en el Estado» (640/2024 ×8, 296, 282). Los poderes judiciales
    # locales son 32: sin la entidad, el nombre no identifica a nadie. Sólo se
    # mira el DATO (ficha o encargo); en la prosa el tribunal abrevia a veces
    # después de nombrarla entera, y eso no se acusa.
    if canon and re.search(r"(?:Tribunal\s+Superior\s+de\s+Justicia|Tribunal\s+de\s+Justicia\s+Administrativa|"
                           r"Poder\s+Judicial|Junta\s+Local\s+de\s+Conciliaci[óo]n\s+y\s+Arbitraje)"
                           r"(?:\s+(?:(?:en|de)\s+(?:el|la)|del)\s+(?:Estado|Entidad))?\s*[.,]?\s*$", canon, re.I):
        out.append(("d", f"{figura} SIN ENTIDAD FEDERATIVA: «{_un_renglon(canon)}». Falta «del Estado de …» (o «de la "
                         f"Ciudad de México»); con ese nombre no se sabe de qué tribunal se trata."))
    # LAS VERSALES EN LA PROSA (AR 631: «rendido por la MAGISTRADA LAURA
    # ANGÉLICA…»): se miran todas las fórmulas, del tipo que sea, porque el
    # nombre en versales delata un dato copiado de la carátula sin pasar por
    # `_nombre_de_organo`, esté donde esté.
    # RONDA 3 (3-oct-2026): también tras «lo hizo valer», «interpuesto por» y
    # «promovido por», y ahí también el de una PERSONA. El banco de oráculo:
    # 7 de 8 RF decían en la legitimación «pues lo hizo valer el JEFA DE LA
    # UNIDAD JURÍDICA…» y el AR 208 «interpuesto por el GOBERNADOR DEL ESTADO
    # DE QUERÉTARO (AUTORIDAD RESPONSABLE)»; la prueba de punta a punta del AD
    # 274, «promovido por GABRIEL REYES ALAMO» en el V I S T O y en la
    # legitimación. Un aviso con todos los apartados donde pasa.
    con_versales = []
    for s in secs:
        if s["zona"] in ("caratula", "proemio") or "antecedentes" in s["etiquetas"]:
            continue
        cuerpo = _sin_bloque_reclamado(s)
        hallado = None
        for rx in (_RX_DICTADO_POR, _RX_RENDIDO_POR, _RX_REMITIO, _RX_CORRESPONDIO, _RX_TRAMITADO_ANTE):
            for m in rx.finditer(cuerpo):
                x = _tramo_en_versales(m.group("o").strip(" ,.;"))
                if x:
                    hallado = x
                    break
            if hallado:
                break
        if not hallado:
            for m in list(_RX_PROMOVIDO_POR.finditer(cuerpo)) + list(_RX_PROTEGE_A.finditer(cuerpo)):
                o = m.group("o")
                x = (_tramo_en_versales(o) or _nombre_en_versales(o)
                     or _racha_en_versales(_mencion_hasta_el_fin(o)))
                if x:
                    hallado = x
                    break
        if not hallado:
            # RONDA 4: el nombre ANTES de «promovió / interpuso» (la apertura
            # del resultando de la demanda: «…, la sucesión intestamentaria a
            # bienes de JUAN PÉREZ promovió juicio de amparo indirecto»).
            for m in _RX_VERBO_DE_PARTE.finditer(cuerpo):
                antes = cuerpo[max(0, m.start() - 240):m.start()]
                corte = max(antes.rfind(". "), antes.rfind(";"), antes.rfind("\n"))
                x = _racha_en_versales(antes[corte + 1:] if corte >= 0 else antes, ultima=True)
                if x:
                    hallado = x
                    break
        if hallado:
            con_versales.append((s, hallado))
    if con_versales:
        s0, x0 = con_versales[0]
        donde = ", ".join(s["nombre"] for s, _ in con_versales[:4]) + (" y otros" if len(con_versales) > 4 else "")
        out.append(("d", f"NOMBRE EN VERSALES EN MITAD DE LA PROSA ({donde}): «{_un_renglon(x0)}». "
                         f"Las versales son de la carátula; en el cuerpo, el nombre con su artículo y en "
                         f"minúsculas."))
    # LA ETIQUETA DE ROL DE LA CARÁTULA, COLADA EN LA PROSA («GOBERNADOR DEL
    # ESTADO DE QUERÉTARO (AUTORIDAD RESPONSABLE)», AR 208; «Titular de la
    # Unidad Jurídica… (demandada)», RF 4 del banco de oráculo). En la carátula
    # el paréntesis dice qué parte es; en el cuerpo sobra: el carácter ya lo
    # dice la oración («la autoridad responsable, Gobernador del Estado…»).
    for s in secs:
        if s["zona"] in ("caratula", "proemio") or "antecedentes" in s["etiquetas"]:
            continue
        m = next((x for x in _RX_ETIQUETA_ROL.finditer(_sin_bloque_reclamado(s)) if _etiqueta_de_caratula(x)), None)
        if m:
            out.append(("d", f"ETIQUETA DE ROL DE LA CARÁTULA EN LA PROSA ({s['nombre']}): «{m.group(0)}». En el "
                             f"cuerpo la parte se nombra sin el paréntesis de la carátula; su carácter lo dice la "
                             f"oración."))
            break
    _regla_representacion(secs, out)


_RX_TRIB_PROEMIO = re.compile(r"(?:Resoluci[óo]n|Acuerdo)\s+del?\s+(?P<t>[^\n]{8,260}?Circuito)\b", re.I)
_RX_TRIB_COMPETENCIA = re.compile(r"\b[Ee]ste\s+(?P<t>[^\n;]{4,260}?)\s+es\s+competente")


def _regla_tribunal(t, secs, datos, out):
    formas = []
    for s in _de(secs, zona="proemio"):
        m = _RX_TRIB_PROEMIO.search(s["texto"])
        if m:
            formas.append((s, m.group("t")))
    for s in _de(secs, zona="considerando", etiqueta="competencia"):
        m = _RX_TRIB_COMPETENCIA.search(s["texto"])
        if m:
            x = m.group("t")
            if re.match(r"^(?:el|la)\s", x):
                out.append(("d", f"ARTÍCULO DUPLICADO EN {s['nombre'].upper()}: «Este {_un_renglon(x)[:80]}…». "
                                 f"El nombre del tribunal entra sin su artículo."))
            if _en_versales(x):
                out.append(("d", f"EL TRIBUNAL EN VERSALES EN {s['nombre'].upper()}: «Este {_un_renglon(x)[:90]}…». "
                                 f"En la prosa va con mayúsculas iniciales, como en el proemio."))
            formas.append((s, x))
    # SE COMPARA DENTRO DEL DOCUMENTO, no contra lo tecleado: el proemio pasa
    # el nombre por `_nombre_de_organo` («XXII» → «Vigésimo Segundo») y eso no
    # es escribirlo de dos formas.
    llenas = [(s, x) for s, x in formas if "circuito" in _plano(x)]
    claves = {}
    for s, x in llenas:
        claves.setdefault(_clave_organo(re.sub(r"^\s*(?:el|la)\s+", "", x)), (s, x))
    if len(claves) > 1:
        (s1, x1), (s2, x2) = list(claves.values())[:2]
        out.append(("d", f"EL TRIBUNAL ESCRITO DE DOS FORMAS: en {s1['nombre']} «{_un_renglon(x1)}» y en "
                         f"{s2['nombre']} «{_un_renglon(x2)}». Es la denominación oficial: una sola."))


# EL PONENTE, DE UNA SOLA FORMA (ronda 2, 3-oct-2026). Al pasar la verja por
# los 13 documentos integrados salió lo que ninguna regla miraba: la carátula
# dice «MAGISTRADO PONENTE: MAGISTRADO EJEMPLO» (del encargo) y el resultando
# del turno «a la ponencia de la Magistrada Ejemplo» (de la ficha, leída del
# auto de turno) —5 de 13—. Son dos fuentes del mismo dato que no se cotejan.
# En las 41 sentencias del tribunal la carátula y el turno nombran SIEMPRE a la
# misma persona con el mismo cargo. Con returno o nueva integración el ponente
# cambia de verdad, y la carátula se coteja contra el ÚLTIMO returno (ronda 3);
# sólo se coteja cuando las dos menciones se leen enteras; el género sale del
# CARGO escrito («Magistrada»), nunca del nombre de pila.
_RX_CARATULA_PONENTE = re.compile(
    r"\bMAGISTRAD(?P<g>[OA])\s+PONENTE\s*:\s*(?P<n>(?:[A-ZÁÉÍÓÚÑÜ][A-ZÁÉÍÓÚÑÜ.]*[ \t]*){1,10})")
_RX_TURNO_PONENTE = re.compile(
    r"(?:(?i:ponencia)(?:\s+a\s+cargo)?\s+(?:del?|de\s+la)\s+|"
    r"\b(?:re)?turn(?:aron|ó|o)\s+(?:[^.;,]{0,60}?\s+)?(?:al|a\s+la)\s+(?!(?i:ponencia)\b))"
    r"(?:(?P<cargo>[Mm]agistrad[oa])\s+)?(?P<n>[A-ZÁÉÍÓÚÑ][^,.;:\n(]{1,90})")
_PARTICULAS = {"de", "del", "la", "las", "los", "y"}
_CARGOS_PONENTE = {"magistrado", "magistrada", "secretario", "secretaria", "en", "funciones", "lic", "licenciado",
                   "licenciada", "dr", "dra", "mtro", "mtra", "doctor", "doctora", "maestro", "maestra"}


def _tokens_nombre(x: str) -> list:
    """Las palabras del nombre, sin cargo ni partículas, para comparar."""
    out = []
    for w in re.findall(r"[A-Za-zÁÉÍÓÚÑÜáéíóúñü]+", _plano(x)):
        if w in _PARTICULAS or w in _CARGOS_PONENTE:
            continue
        out.append(w)
    return out


def _nombre_en_prosa(x: str) -> str:
    """«Ismael Camacho Herrera para la formulación…» → «Ismael Camacho Herrera»."""
    keep = []
    for w in str(x or "").split():
        if w[:1].isupper() or (keep and w.lower() in _PARTICULAS):
            keep.append(w)
        else:
            break
    while keep and keep[-1].lower() in _PARTICULAS:
        keep.pop()
    return " ".join(keep)


def _ponente_de_la_ficha(ficha, clave):
    """(nombre, cargo) del ponente del turno o del returno en la ficha: «Magistrada X» → ("X", "Magistrada")."""
    v = _g(ficha, clave, "ponente") if isinstance(ficha, dict) else ""
    v = _un_renglon(v) if isinstance(v, str) else ""
    m = re.match(r"^(?:(?P<c>[Mm]agistrad[oa])\s+)?(?P<n>.+)$", v)
    return (m.group("n"), m.group("c") or "") if (m and v and "*" not in v) else ("", "")


def _regla_ponente(t, secs, ficha, out):
    car = " ".join(s["texto"] for s in _de(secs, zona="caratula"))
    m = _RX_CARATULA_PONENTE.search(car)
    if not m:
        return
    palabras = []
    for w in m.group("n").split():
        if re.match(r"SECRETARI", w):
            break
        palabras.append(w)
    n_c = " ".join(palabras).strip(" .,")
    g_c = m.group("g").lower()
    tk_c = _tokens_nombre(n_c)
    if not tk_c or "*" in n_c or "fulano" in tk_c:
        return
    # CON RETURNO, LA CARÁTULA ES DEL ÚLTIMO RETURNO (ronda 3, 3-oct-2026). Tras
    # las readscripciones de 2025 el returno es lo normal: en el banco de
    # oráculo lo hubo en los 8 AD, en 6 de 8 RF y en las quejas 261 y 335, y en
    # todos la carátula nombraba al ponente del TURNO mientras el resultando
    # «Returno» pasaba los autos a otro (el engrose firma el del returno). La
    # regla vieja callaba en cuanto había returno. Ahora coteja contra el
    # último ponente que nombran los resultandos de returno o nueva
    # integración; si ninguno lo nombra, contra `returno.ponente` de la ficha.
    returnos = _de(secs, zona="resultando", etiqueta="returno")
    if returnos:
        ultimo = None
        for s in returnos:
            for mt in _RX_TURNO_PONENTE.finditer(s["texto"]):
                n_r = _nombre_en_prosa(mt.group("n"))
                tk_r = _tokens_nombre(n_r)
                if tk_r and "*" not in n_r and "fulano" not in tk_r:
                    ultimo = (s["nombre"], n_r, mt.group("cargo") or "", tk_r)
        if ultimo is None:
            n_f, cargo_f = _ponente_de_la_ficha(ficha, "returno")
            tk_f = _tokens_nombre(n_f)
            if not tk_f or "fulano" in tk_f:
                return
            ultimo = ("la ficha de trámite (returno.ponente)", n_f, cargo_f, tk_f)
        donde, n_r, cargo, tk_r = ultimo
        dicho = f"{cargo + ' ' if cargo else ''}{n_r}"
        a, b = set(tk_c), set(tk_r)
        if not (a <= b or b <= a):
            out.append(("d", f"EL PONENTE DE LA CARÁTULA NO ES EL DEL RETURNO: la carátula dice «MAGISTRAD"
                             f"{g_c.upper()} PONENTE: {n_c}» y el último returno pasó los autos a «{dicho}» ({donde}). "
                             f"Tras el returno, la carátula lleva a quien recibió los autos: corrige el ponente "
                             f"de la carátula (o la fecha y el ponente del returno en la ficha de trámite)."))
        elif cargo and cargo[-1].lower() != g_c:
            out.append(("d", f"EL CARGO DEL PONENTE NO CONCUERDA: la carátula dice «MAGISTRAD{g_c.upper()} "
                             f"PONENTE: {n_c}» y {donde} «{dicho}». Es la misma persona con el mismo cargo "
                             f"en los dos sitios."))
        return
    for s in _de(secs, zona="resultando", etiqueta="turno"):
        mt = _RX_TURNO_PONENTE.search(s["texto"])
        if not mt:
            continue
        n_t = _nombre_en_prosa(mt.group("n"))
        tk_t = _tokens_nombre(n_t)
        if not tk_t or "*" in n_t or "fulano" in tk_t:
            return
        cargo = mt.group("cargo") or ""
        dicho = f"{cargo + ' ' if cargo else ''}{n_t}"
        a, b = set(tk_c), set(tk_t)
        if not (a <= b or b <= a):
            out.append(("d", f"EL PONENTE NO ES EL MISMO EN TODO EL DOCUMENTO: la carátula dice «MAGISTRAD"
                             f"{g_c.upper()} PONENTE: {n_c}» y {s['nombre']} «{dicho}». Sin returno de por medio, "
                             f"es la misma persona: o el turno o la carátula está mal; y si hubo returno, falta su "
                             f"resultando (fecha y ponente del returno en la ficha de trámite)."))
        elif cargo and cargo[-1].lower() != g_c:
            out.append(("d", f"EL CARGO DEL PONENTE NO CONCUERDA: la carátula dice «MAGISTRAD{g_c.upper()} "
                             f"PONENTE: {n_c}» y {s['nombre']} «{dicho}». Es la misma persona con el mismo cargo "
                             f"en los dos sitios."))
        return


# EL RENGLÓN DE QUIEN RECURRE (ronda 3, 3-oct-2026; banco de oráculo, Q 300):
# recurrió la tercera interesada y la carátula de la queja decía «RECURRENTE:»
# con el nombre del quejoso, porque la fila se llenaba con la clave «quejoso»;
# en producción pasa igual (datos['quejoso'] es el quejoso del amparo). Sólo
# cuando recurre alguien que NO es el quejoso (tercero, autoridad, Ministerio
# Público) y los dos nombres se pueden leer: sin palabras propias suficientes
# —un nombre testado, «PARTE RECURRENTE»— no se compara.
_RX_CARATULA_RECURRENTE = re.compile(
    r"^[ \t]*(?P<et>[A-ZÁÉÍÓÚÑ ]*\bRECURRENTES?\b[A-ZÁÉÍÓÚÑ ()]*?)\s*:\s*(?P<n>[^\n]+?)\s*\.?\s*$", re.M)
_GENERICAS_PARTE = {"parte", "partes", "quejoso", "quejosa", "quejosos", "recurrente", "recurrentes", "tercero",
                    "tercera", "terceros", "interesado", "interesada", "interesados", "autoridad", "autoridades",
                    "responsable", "responsables", "demandada", "demandado", "actora", "actor", "otros", "otras",
                    "otro", "otra", "ambos", "apellidos", "adherente", "adhesivo"}


def _palabras_propias(x: str) -> set:
    return {w for w in _palabras(x) if len(w) > 2 and w not in _GENERICAS_PARTE and w not in _PARTICULAS_REPR}


def _regla_recurrente_caratula(secs, ficha, out):
    if not _es_ficha(ficha):
        return
    caracter = _plano(_g(ficha, "caracter"))
    prom = _g(ficha, "promovente")
    if caracter not in ("tercero", "autoridad", "ministerio_publico") or not isinstance(prom, str) or "*" in prom:
        return
    tk_p = _palabras_propias(prom)
    if len(tk_p) < 2:
        return
    cab = "\n".join(s["texto"] for s in _de(secs, zona="caratula"))
    malo = None
    for m in _RX_CARATULA_RECURRENTE.finditer(cab):
        tk_n = _palabras_propias(m.group("n"))
        if len(tk_n) < 2:
            continue
        if len(tk_p & tk_n) >= 0.5 * min(len(tk_p), len(tk_n)):
            return
        malo = malo or m
    if malo:
        out.append(("d", f"EL RENGLÓN DE QUIEN RECURRE NO ES QUIEN RECURRE: la carátula dice «{_un_renglon(malo.group('et'))}: "
                         f"{_un_renglon(malo.group('n'))[:90]}» y el recurso lo interpuso {_un_renglon(prom)[:90]} "
                         f"(carácter «{caracter}» en la ficha de trámite). La carátula nombra a quien recurre, con su "
                         f"carácter en el juicio y «Y RECURRENTE»; el quejoso conserva su propio renglón."))


# ═══════════════════════════════════════════════════════════════════════════
# (e) NÚMEROS: el del asunto, el toca y el expediente
# ═══════════════════════════════════════════════════════════════════════════
_RX_NUM = re.compile(r"(?<![\d/])(\d{1,5})\s*/\s*(\d{4})(?![\d/])")
# «ADC 625-2024 ORAL MERCANTIL», «ADA 103/2025», «ADC 174-2026»: las siglas del
# rótulo interno del banco o del formulario, coladas en la prosa (27 de 81). Con
# puntos («A.R.A. 214/2026») es el encabezado de página de una versión pública
# y no se acusa; en la prosa del corpus no aparecen las siglas sin puntos.
_RX_ETIQUETA = re.compile(
    r"\b(?:AD[ACLP]?|AR[ACLP]?|RF|RQ|Q[ACLP]|IR)\s+\d{1,5}[-/]\d{4}\b")


def _num(x) -> str:
    m = _RX_NUM.search(str(x or ""))
    return f"{int(m.group(1))}/{m.group(2)}" if m else ""


def _regla_numeros(t, secs, ficha, datos, tipo, out):
    canon = _num(_g(ficha, "numero")) or _num((datos or {}).get("numero"))
    cab = next(iter(_de(secs, zona="caratula")), None)
    visto = next(iter(_de(secs, zona="visto")), None)
    n_cab = ""
    if cab:
        for renglon in cab["texto"].split("\n"):
            # EL RENGLÓN «RELACIONADO CON …» NO ES EL NÚMERO DEL ASUNTO (C6, sexta
            # ronda, 3-oct-2026): va justo debajo del encabezado y lleva el número
            # de OTRO asunto. Con el encabezado sin número, la verja decía «la
            # carátula dice 33/2024 y el asunto es el 469/2024». Tampoco cuenta
            # dentro del mismo renglón («…469/2024 (RELACIONADO CON …)», AD 469
            # del banco), ni pasada la primera parte («PARTE ACTORA:» de la
            # revisión fiscal, C4).
            if re.match(r"^\s*(?:\(?\s*RELACIONAD|QUEJOS|TERCER|MAGISTRAD|SECRETARI|RECURRENTE|AUTORIDAD|ACTORA|"
                        r"PARTE\s|[ÓO]RGANO|SALA\s|MATERIA)", renglon.strip(), re.I):
                break
            n_cab = _num(re.split(r"\(?\s*RELACIONAD", renglon, maxsplit=1, flags=re.I)[0])
            if n_cab:
                break
    # El V I S T O nombra el asunto y, tras él, «relacionado con el amparo
    # directo civil 452/2025»: ese número es del relacionado (sexta ronda). Y si
    # antes del primer número hay un hueco, el del asunto va en hueco —lo dice
    # (a)—: el primer número que queda es el toca o el del relacionado, no el
    # del asunto («…amparo directo civil *********, relacionado con … 175/2026»).
    n_visto = ""
    if visto:
        _tv = _sin_relacionados(visto["texto"])
        _mv = _RX_NUM.search(_tv)
        if _mv and not _RX_HUECO.search(_tv[:_mv.start()]):
            n_visto = _num(_mv.group(0))
    if n_visto and canon and n_visto != canon:
        out.append(("e", f"EL NÚMERO DEL ASUNTO NO CUADRA: el V I S T O dice {n_visto} y el asunto es el {canon}."))
    elif n_visto and n_cab and n_visto != n_cab:
        out.append(("e", f"EL NÚMERO DEL ASUNTO NO CUADRA: la carátula dice {n_cab} y el V I S T O {n_visto}."))
    if n_cab and canon and n_cab != canon:
        out.append(("e", f"EL NÚMERO DEL ASUNTO NO CUADRA: la carátula dice {n_cab} y el asunto es el {canon}."))
    # LA CARÁTULA SIN EL NÚMERO (ronda 3, banco de oráculo de la RF, 8 de 8):
    # con el encabezado vacío la carátula empezaba en el renglón de quien
    # recurre (entonces «AUTORIDAD RECURRENTE:»; desde C4, 3-oct-2026,
    # «RECURRENTE:») y nada lo decía. El primer renglón identifica el asunto
    # («REVISIÓN FISCAL: 2/2025.»). Sólo con ficha de trámite: es el camino
    # nuevo el que compone la carátula entera.
    if cab and canon and not n_cab and _es_ficha(ficha):
        ejemplo = ""
        try:
            import tipos_asunto as _ta
            ejemplo = _un_renglon(_ta.encabezado_de(tipo, str(_g(ficha, "materia") or ""), canon) or "")
        except Exception:
            ejemplo = ""
        out.append(("e", f"LA CARÁTULA NO TRAE EL NÚMERO DEL ASUNTO: el asunto es el {canon} y la carátula no lo "
                         f"dice{' (su primer renglón: «' + ejemplo + '»)' if ejemplo else ''}. Sin él, el documento "
                         f"no identifica el asunto."))
    for s in secs:
        if s["zona"] in ("visto", "resultando", "resolutivos"):
            m = _RX_ETIQUETA.search(s["texto"])
            if m:
                out.append(("e", f"LA ETIQUETA DEL ENCABEZADO SE COLÓ EN {s['nombre'].upper()}: «{m.group(0)}». "
                                 f"El asunto se cita por su número ({canon or 'núm./año'}), sin las siglas ni la "
                                 f"vía del rótulo interno."))
                break
    # EL TOCA NO ES EL EXPEDIENTE (AD 174/2026: «los autos del expediente
    # 4357/2025», que era el toca de la Sala).
    tocas = set()
    for m in re.finditer(r"\btoca(?:\s+(?:civil|familiar|mercantil|penal|administrativo|de\s+apelaci[óo]n|"
                         r"n[úu]mero))*\s+(\d{1,6}/\d{4})", t, re.I):
        tocas.add(_num(m.group(1)))
    t_ficha = _num(_g(ficha, "acto", "toca"))
    e_ficha = _num(_g(ficha, "acto", "expediente"))
    if t_ficha:
        tocas.add(t_ficha)
    for s in _de(secs, zona="considerando", etiqueta="existencia"):
        for m in re.finditer(r"\bexpediente(?:\s+(?:de\s+origen|natural|n[úu]mero|civil|familiar|mercantil))*\s+"
                             r"(\d{1,6}/\d{4})", s["texto"], re.I):
            n = _num(m.group(1))
            if n in tocas and n != e_ficha:
                out.append(("e", f"LA EXISTENCIA LLAMA EXPEDIENTE AL TOCA: «{_un_renglon(m.group(0))}», y "
                                 f"{n} es el toca de apelación. Los autos remitidos son el toca"
                                 f"{' y el expediente ' + e_ficha if e_ficha else ''}; cada uno con su nombre."))
                break
        # AR: la existencia de la resolución recurrida no se acredita con el
        # informe justificado de la responsable (0 de 57 en el corpus; la
        # responsable no remite los autos del amparo). Desde C1 (3-oct-2026) el
        # AR compuesto ya no lleva existencia y esto sólo mira el que la trae
        # (una plantilla vieja, un engrose); ninguna regla la exige.
        if tipo == "amparo_revision" and re.search(r"informe\s+justificado", s["texto"], re.I):
            out.append(("e", f"EXISTENCIA MAL PLANTEADA EN {s['nombre'].upper()}: en la revisión lo que se "
                             f"acredita es la resolución recurrida, con los autos originales del juicio que "
                             f"remitió el juzgado (art. 89 de la Ley de Amparo), no con el informe justificado "
                             f"de la responsable."))


# ═══════════════════════════════════════════════════════════════════════════
# (f) CONCORDANCIAS Y PULCRITUD QUE NO SE ARREGLAN SOLAS
# ═══════════════════════════════════════════════════════════════════════════
# LOS FEMENINOS QUE `_con_articulo` NO CONOCÍA (ronda 3, 3-oct-2026). El banco
# de oráculo los encontró en los cuatro tipos: «el Jefa de la Unidad Jurídica»
# y «el Subdirectora de Afiliación» (RF 2, 7 y 26), «el Coordinadora de
# Recursos Humanos» (AD 128), «el Legislatura del Estado» (AR 201), y la
# Sala con prefijo temporal, «ante el Actual Sala Regional…, localizado» (RF
# 49). El sustantivo se lee sin distinguir mayúsculas («el JEFA», en versales)
# y salta el prefijo temporal (actual, entonces, hoy, ahora, otrora). Los
# cargos en -dora/-tora/-sora son femeninos (Coordinadora, Directora,
# Procuradora, Asesora); «el acta», «el área» o «el águila» no entran: no son
# de esta lista.
_PREFIJO_TEMPORAL = r"(?i:(?:actual|entonces|hoy|ahora|otrora)\s+)?"
_SUST_FEMENINOS = (r"Jueza|Magistrada|Presidenta|Directora|Subdirectora|Secretaria|Subsecretaria|Sala|Junta|"
                   r"Subprocuradur[íi]a|Procuradur[íi]a|Fiscal[íi]a|Unidad|Delegaci[óo]n|Subdelegaci[óo]n|"
                   r"Administraci[óo]n|Comisi[óo]n|Secretar[íi]a|Subsecretar[íi]a|Coordinaci[óo]n|Direcci[óo]n|"
                   r"Subdirecci[óo]n|Jefa|Subjefa|Delegada|Subdelegada|Encargada|Comisionada|Consejera|Contralora|"
                   r"Tesorera|Legislatura|C[áa]mara|Jefatura|Subjefatura|Oficina|Sindicatura|Gubernatura|"
                   r"Regidur[íi]a|Alcald[íi]a|[A-Za-zÁÉÍÓÚáéíóúÑñ]*(?:dora|tora|sora)")
_CONCORDANCIAS = (
    (r"\b(?:[Ee]l|[Dd]el|[Aa]l|EL|DEL|AL)\s+" + _PREFIJO_TEMPORAL + r"(?i:" + _SUST_FEMENINOS + r")\b",
     "artículo masculino ante un nombre femenino"),
    (r"\b(?:[Ll]a|LA)\s+" + _PREFIJO_TEMPORAL + r"(?:Juzgado|Tribunal|Magistrado|Director|Instituto|Ayuntamiento|"
     r"Congreso)\b",
     "artículo femenino ante un nombre masculino"),
    (r"\bla\s+quejoso\b|\bel\s+quejosa\b|\b(?:del|al)\s+quejosa\b|\bla\s+tercero\s+interesado\b|"
     r"\bel\s+tercera\s+interesada\b|\b(?:del|al)\s+tercera\s+interesada\b",
     "el artículo no concuerda con la parte"),
    (r"\ben\s+contra\s+el\s+(?:acto|auto|laudo)\b|\ben\s+contra\s+el\s+(?:sentencia|resoluci[óo]n)\b",
     "«en contra el» (es «en contra del»)"),
    (r"\b(?:sede|residencia)\s+en\s+(?:la|el)\s+(?!(?i:ciudad|capital|propia|misma|zona|demarcaci[óo]n|"
     r"entidad|regi[óo]n|circunscripci[óo]n|municipio|localidad|delegaci[óo]n|alcald[íi]a)\b)[A-ZÁÉÍÓÚ][a-záéíóú]+",
     "artículo ante el nombre de la ciudad"),
)
_CONCORDANCIAS = tuple((re.compile(p), d) for p, d in _CONCORDANCIAS)
# «LOCALIZADO» CON LA SALA. Se mira el núcleo del complemento —«por la Sala…»,
# «ante la Sala…»— y no la cercanía: con «juzgado» el corpus escribe a veces
# «localizada» concordando con la resolución, y eso no se acusa.
# Con el prefijo temporal también («ante la actual Sala…, localizado»; RF 49 del
# banco de oráculo, ronda 3), y con el artículo equivocado («ante el Actual
# Sala…, localizado»): el sustantivo es la Sala en los dos casos.
_RX_LOCALIZADO = re.compile(r"\b(?:por|ante)\s+(?:la|el)\s+" + _PREFIJO_TEMPORAL +
                            r"(?:Primera\s+|Segunda\s+|Tercera\s+|Cuarta\s+|Quinta\s+|"
                            r"Sexta\s+|S[ée]ptima\s+|Octava\s+|Novena\s+|D[ée]cima\s+)?"
                            r"(?:Sala|Junta|Jueza|Magistrada)\b[^.;]{0,280}?,\s*localizado\b")
_RX_LEGITIMADO = re.compile(r"\blegitimad([oa])s?\b")


_RX_FORMA_FEM = re.compile(r"\bS\.\s*A\b|\bS\.\s*C\b|\bA\.\s*C\b|\bS\.\s*de\s*R\.|\bsociedad\b|"
                           r"\binstituci[óo]n\b|\basociaci[óo]n\b", re.I)


# EL CARGO QUE ENCABEZA EL NOMBRE DE UNA AUTORIDAD manda sobre lo que venga
# detrás: «Titular de la Unidad Jurídica…» no es femenino por «unidad», y
# «Director de Ingresos del Municipio» es masculino por «director», no por
# «municipio». «Titular», «agente» y «fiscal» no dicen el género: se calla.
_CARGO = re.compile(r"^\s*(?:(el|la)\s+)?(titular|agente|fiscal|juez|jueza|jef[ea]|"
                    r"\w*(?:director|administrador|procurador|coordinador|gobernador|registrador|encargad|"
                    r"delegad|secretari|president|magistrad|subdirector|notificador)(?:a|o|e|es|as)?)\b", re.I)


def _genero_parte(nombre: str, datos: dict, clave: str) -> str:
    """«a», «o» o "" (no se sabe o es ambiguo: entonces no se acusa)."""
    g = str((datos or {}).get(f"genero_{clave}") or "").strip().lower()[:1]
    if g in ("a", "f"):
        return "a"
    if g in ("o", "m"):
        return "o"
    m = _CARGO.match(str(nombre or ""))
    if m:
        art, cargo = (m.group(1) or "").lower(), _plano(m.group(2))
        if art:
            return "a" if art == "la" else "o"
        if cargo in ("titular", "agente", "fiscal"):
            return ""
        if cargo in ("jueza", "jefa") or cargo.endswith(("ora", "ada", "aria", "enta")):
            return "a"
        return "o"
    try:
        import tipos_asunto as _ta
        g = _ta.genero_de(nombre) or ""
    except Exception:
        return ""
    # «Banco Nacional de México, S.A.»: el banco es masculino y la sociedad
    # femenina, y el corpus concuerda con cualquiera de los dos. Ambiguo.
    if g == "o" and _RX_FORMA_FEM.search(nombre):
        return ""
    return g


def _regla_concordancias(t, secs, datos, out):
    vistos = set()
    for s in secs:
        if s["zona"] == "caratula" or "antecedentes" in s["etiquetas"]:
            continue
        for rx, desc in _CONCORDANCIAS:
            m = rx.search(s["texto"])
            if m and (desc, s["nombre"]) not in vistos:
                vistos.add((desc, s["nombre"]))
                out.append(("f", f"CONCORDANCIA EN {s['nombre'].upper()}: {desc} — "
                                 f"«{_extracto(s['texto'], m.start(), m.end(), 40)}»."))
        m = _RX_LOCALIZADO.search(s["texto"])
        if m:
            out.append(("f", f"CONCORDANCIA EN {s['nombre'].upper()}: «localizado» con un órgano femenino — "
                             f"«{_extracto(s['texto'], m.end() - 12, m.end(), 50)}». Es «localizada»."))
    # LEGITIMADO / LEGITIMADA CON LA PARTE: sólo cuando el nombre de la parte
    # está en la misma frase y su género se sabe (la carátula concordada, el
    # sustantivo de cabeza de una persona moral o el dato declarado). Por el
    # nombre de pila no se infiere.
    caratula = " ".join(s["texto"] for s in _de(secs, zona="caratula"))
    for clave in ("quejoso", "recurrente"):
        nombre = str((datos or {}).get(clave) or "").strip()
        if len(nombre) < 6:
            continue
        g = _genero_parte(nombre, datos, clave)
        if not g and clave == "quejoso":
            if re.search(r"(?<!PARTE )\bQUEJOSA(?:\s+Y\s+RECURRENTE)?\s*:", caratula):
                g = "a"
            elif re.search(r"(?<!PARTE )\bQUEJOSO(?:\s+Y\s+RECURRENTE)?\s*:", caratula):
                g = "o"
        if not g:
            continue
        pl = _plano(nombre)[:40]
        for s in _de(secs, zona="considerando", etiqueta="legitimacion"):
            cuerpo = s["texto"]
            for m in _RX_LEGITIMADO.finditer(cuerpo):
                # La frase empieza tras un punto seguido de MAYÚSCULA: el de
                # «S.A. de C.V.» no la corta.
                ini = 0
                for mm in re.finditer(r"[.;]\s+(?=[A-ZÁÉÍÓÚÑ¿«])", cuerpo[:m.start()]):
                    ini = mm.end()
                # «…Juan Pérez López; la parte quejosa está legitimada»: el
                # sujeto del adjetivo es LA FIGURA (femenina), no el nombre. Es
                # la fórmula neutra de `legitimacion_de` (3-oct-2026): con ella
                # «legitimada» es correcto sea quien sea la persona.
                if re.search(r"\b(?:la\s+parte|la\s+persona\s+moral|la\s+autoridad)\b[^.;]{0,40}"
                             r"(?:est[áa]|se\s+encuentra)\s+$", cuerpo[ini:m.start()]):
                    continue
                if pl in _plano(cuerpo[ini:m.start()]) and m.group(1) != g:
                    out.append(("f", f"CONCORDANCIA EN {s['nombre'].upper()}: «legitimad{m.group(1)}» con "
                                     f"{nombre[:60]}, que es «{'la' if g == 'a' else 'el'}» "
                                     f"{'quejosa' if g == 'a' and clave == 'quejoso' else 'quejoso' if clave == 'quejoso' else 'recurrente'}."
                                     f" Es «legitimad{g}»."))
                    break


# ── LAS PARTES EN LA PROSA (ronda 4, 3-oct-2026) ──
# LAS FÓRMULAS DE REPRESENTACIÓN CON MAYÚSCULA A MEDIA FRASE. El paso de
# versales a prosa ponía mayúscula a cada palabra: «POR CONDUCTO DE SU CONSEJO
# DE ADMINISTRACIÓN» salía «por Conducto de Su Consejo de Administración» en el
# V I S T O, el resultando, la legitimación y el resolutivo del AD 552, y «por
# Propio Derecho» en su legitimación y su resolutivo; «a Través de Su Albacea»
# en la Q 342 y «y Otra» en la Q 229. El corpus las escribe siempre en
# minúscula (0 de 41 sentencias de la calibración). «de Representación» no
# entra: es el nombre de la Oficina de Representación del ISSSTE (RF 4, 11,
# 19 y 42 de la calibración).
_RX_FORMULA_MAYUS = re.compile(
    r"\b(?:[Pp]or\s+Conducto|[Pp]or\s+(?:[Ss]u\s+)?Propio\s+Derecho|[Pp]or\s+Su\s+(?:[Pp]ropio|[Rr]epresentante|"
    r"[Aa]poderad[oa]|[Aa]lbacea)|a\s+Trav[ée]s\s+de|en\s+Representaci[óo]n\s+de|en\s+Nombre\s+y\s+Representaci|"
    r"(?:de|a\s+trav[ée]s\s+de|por\s+conducto\s+de)\s+Sus?\s+(?=[A-Za-zÁÉÍÓÚáéíóúÑñ])|[Aa]mbos\s+de\s+Apellidos|"
    r"\by\s+Otr[oa]s?\b(?!\s+[A-ZÁÉÍÓÚÑ]))")
# EL COLECTIVO SIN SU ARTÍCULO. El resolutivo del AD 335 decía «no ampara ni
# protege a Sucesión a Bienes de…» mientras el V I S T O, el resultando y la
# legitimación decían «la Sucesión»; el del AR 60, «a sucesión intestamentaria
# a bienes de…». La sucesión, la comunidad, la asociación, el ejido, el
# sindicato y el núcleo de población llevan su artículo.
_RX_COLECTIVO_SIN_ART = re.compile(
    r"\b(?:ampara\s+(?:y|ni)\s+protege\s+a|promovid[oa]s?\s+por|interpuest[oa]s?\s+por|lo\s+hizo\s+valer)\s+"
    r"(?P<n>(?i:sucesi[óo]n|comunidad|asociaci[óo]n|ejido|sindicato|n[úu]cleo\s+de\s+poblaci[óo]n)\b[^,;.\n]{0,50})")
_ART_COLECTIVO = {"sucesion": "la", "comunidad": "la", "asociacion": "la", "ejido": "el", "sindicato": "el",
                  "nucleo": "el"}
# PLURAL EN EL RESULTANDO, SINGULAR EN LA LEGITIMACIÓN O LA CARÁTULA. AD 552:
# el resultando ya decía «promovieron», y la legitimación «…, y Fulano, por
# propio derecho, quien está legitimada» y la carátula «QUEJOSA:». El corpus,
# con varios: «están legitimados» y «QUEJOSOS:» o «RECURRENTES:» (RF 4 y 11, AR
# 266 de la calibración). «la parte quejosa está legitimada» vale para varios
# (la figura es una) y no se acusa; tampoco el considerando que trata a cada
# autoridad por separado («El Gobernador … se encuentra legitimado»): sólo el
# relativo que cuelga de la lista entera.
_RX_VERBO_PLURAL = {
    "amparo_directo": re.compile(r"\bpromovieron\s+(?:el\s+)?(?:juicio|demanda|amparo)", re.I),
    "otro": re.compile(r"\binterpusieron\s+(?:el\s+|este\s+|dicho\s+|los\s+)?(?:recursos?|revisi|queja)", re.I),
}
_RX_LEGIT_SINGULAR = re.compile(r"\b(?:quien|por\s+lo\s+que)\s+(?:est[áa]|se\s+encuentra)\s+legitimad[oa]\b")
_RX_RENGLON_SINGULAR = {
    "amparo_directo": re.compile(r"(?<![A-ZÁÉÍÓÚÑ])(?<!PARTE )QUEJOS[OA](?:\s+Y\s+RECURRENTE)?\s*:"),
    "otro": re.compile(r"(?<![A-ZÁÉÍÓÚÑ])(?<!PARTE )(?:[A-ZÁÉÍÓÚÑ]+\s+)*RECURRENTE\s*:"),
}
_RX_RENGLON_PLURAL = re.compile(r"\b(?:QUEJOS[OA]S|RECURRENTES|PARTE\s+QUEJOSA|PARTE\s+RECURRENTE)\b\s*[A-ZÁÉÍÓÚÑ ]*:")


def _regla_partes_en_prosa(secs, tipo, out):
    # 1) Las fórmulas con mayúscula a media frase: un aviso con los apartados.
    donde, ejemplo = [], ""
    for s in secs:
        if s["zona"] in ("caratula", "proemio") or "antecedentes" in s["etiquetas"]:
            continue
        m = _RX_FORMULA_MAYUS.search(_sin_bloque_reclamado(s))
        if m:
            donde.append(s["nombre"])
            ejemplo = ejemplo or _extracto(_sin_bloque_reclamado(s), m.start(), m.end(), 30)
    if donde:
        out.append(("f", f"FÓRMULA DE REPRESENTACIÓN CON MAYÚSCULAS A MEDIA FRASE ({', '.join(donde[:5])}"
                         f"{' y otros' if len(donde) > 5 else ''}): «{ejemplo}». Son fórmulas, no parte del "
                         f"nombre: «por conducto de su consejo de administración», «por propio derecho», «a través "
                         f"de su albacea», «en representación de», «y otra», en minúsculas."))
    # 2) El colectivo sin artículo: un aviso por apartado.
    for s in secs:
        if s["zona"] in ("caratula", "proemio") or "antecedentes" in s["etiquetas"]:
            continue
        m = _RX_COLECTIVO_SIN_ART.search(_sin_bloque_reclamado(s))
        if m:
            nucleo = _plano(m.group("n")).split()[0]
            art = _ART_COLECTIVO.get(nucleo, "la")
            out.append(("f", f"FALTA EL ARTÍCULO DEL COLECTIVO EN {s['nombre'].upper()}: «…{_un_renglon(m.group(0))}». "
                             f"Va «{art} {_un_renglon(m.group('n')).split()[0]}…», como en el V I S T O y los "
                             f"resultandos."))
    # 3) Plural en el resultando de quien promueve o recurre, singular después.
    clave = "amparo_directo" if tipo == "amparo_directo" else "otro"
    plural = None
    for s in _de(secs, zona="resultando"):
        rot = s["rotulo"]
        if clave == "amparo_directo":
            if not ("presentacion" in s["etiquetas"] and re.search(r"demanda", rot, re.I)):
                continue
        elif not (re.search(r"interposici|recurso", rot, re.I) and not re.search(r"demanda", rot, re.I)):
            continue
        m = _RX_VERBO_PLURAL[clave].search(_sin_bloque_reclamado(s))
        if m:
            plural = (s, m.group(0).split()[0])
            break
    if plural is None:
        return
    singulares = []
    for s in _de(secs, zona="considerando", etiqueta="legitimacion"):
        cuerpo = s["texto"]
        for m in _RX_LEGIT_SINGULAR.finditer(cuerpo):
            ini = 0
            for mm in re.finditer(r"[.;]\s+(?=[A-ZÁÉÍÓÚÑ¿«])", cuerpo[:m.start()]):
                ini = mm.end()
            if re.search(r"\b(?:la\s+parte|la\s+persona\s+moral|la\s+autoridad)\b[^.;]{0,40}$", cuerpo[ini:m.start()]):
                continue
            singulares.append((s["nombre"], m.group(0)))
            break
    cab = "\n".join(s["texto"] for s in _de(secs, zona="caratula"))
    sing_c = list(_RX_RENGLON_SINGULAR[clave].finditer(cab))
    if len(sing_c) == 1 and not _RX_RENGLON_PLURAL.search(cab):
        singulares.append(("la carátula", _un_renglon(sing_c[0].group(0))))
    if singulares:
        lugares = "; ".join(f"{n} «{x}»" for n, x in singulares)
        out.append(("f", f"EL DOCUMENTO DICE QUE SON VARIOS Y LUEGO QUE ES UNO: {plural[0]['nombre']} dice "
                         f"«{plural[1]}» y {lugares}. Si son varios: «están legitimados» (o «la parte "
                         f"{'quejosa' if clave == 'amparo_directo' else 'recurrente'} está legitimada») y en la "
                         f"carátula el rótulo en plural («{'QUEJOSOS' if clave == 'amparo_directo' else 'RECURRENTES'}:»"
                         f" o «PARTE {'QUEJOSA' if clave == 'amparo_directo' else 'RECURRENTE'}:»); si es uno solo, "
                         f"el verbo del resultando va en singular."))


# ── EL ARTÍCULO DEL PAPEL (quinta ronda, 3-oct-2026, F5; AD 128 del banco) ──
# La ficha trae «la Oficial Mayor y Coordinadora de Recursos Humanos del
# Municipio de Cadereyta de Montes, Querétaro» —así lo dice el auto— y el
# resultando del tercero salía «Tiene ese carácter el Oficial Mayor…»: el
# compositor quitaba el artículo y ponía el masculino por omisión. «Oficial»,
# «Titular», «Fiscal», «Agente», «Representante» sirven a los dos géneros; el
# género de quien ocupa el cargo lo dice el artículo del papel, y no se puede
# deducir de otra cosa (nunca del nombre de pila). Se acusa el texto que
# escribe ESE cargo, con las mismas palabras, con el artículo del otro género.
# Sin artículo en el papel no hay contra qué cotejar: el compositor escribe
# «el» con su aviso «EL GÉNERO DEL CARGO NO CONSTA».
_RX_CARGO_DOBLE_GENERO = re.compile(r"^(?:titular|oficial|fiscal|agente|representante|comandante|gerente|"
                                    r"encargad[oa])(?:e?s)?\b")
_CAMPOS_CON_NOMBRE = (("promovente",), ("quejoso",), ("responsable",), ("ejecutora",), ("terceros",), ("actora",),
                      ("autoridad_demandada",), ("representante",), ("figura_representante",), ("sala",),
                      ("acto", "organo"), ("resolucion_impugnada", "autoridad"), ("demanda", "autoridades"))
_ART_GENERO = {"el": "o", "del": "o", "al": "o", "los": "o", "la": "a", "las": "a"}
_VOCAL_FLEX = {"a": "[aá]", "e": "[eé]", "i": "[ií]", "o": "[oó]", "u": "[uúü]", "n": "[nñ]"}


def _rx_flexible(palabra: str) -> str:
    """La palabra sin tildes, como patrón que acepta las dos grafías."""
    return "".join(_VOCAL_FLEX.get(c, re.escape(c)) for c in _plano(palabra))


def _nombres_con_articulo(ficha, datos) -> list:
    """[(campo, género del artículo, valor, patrón del cuerpo, cuerpo llano)] de
    los nombres de la ficha (y del encargo) que empiezan con artículo y un cargo
    de doble género. El patrón son las primeras ocho palabras del cuerpo."""
    vistos, out = set(), []

    def _uno(campo, v):
        if isinstance(v, (list, tuple)):
            for x in v:
                _uno(campo, x)
            return
        if not isinstance(v, str):
            return
        m = re.match(r"^\s*(el|la|los|las)\s+(\S.*)$", v.strip(), re.I)
        if not m or not _RX_CARGO_DOBLE_GENERO.match(_plano(m.group(2))):
            return
        pal = re.findall(r"[^\W\d_]+|\d+", m.group(2))[:8]
        if len(pal) < 2:
            return
        clave = (_ART_GENERO[m.group(1).lower()], " ".join(_plano(p) for p in pal))
        if clave in vistos:
            return
        vistos.add(clave)
        out.append((campo, clave[0], _un_renglon(v), r"\W+".join(_rx_flexible(p) for p in pal), clave[1]))

    for ruta in _CAMPOS_CON_NOMBRE:
        _uno(".".join(ruta), _g(ficha if isinstance(ficha, dict) else {}, *ruta, defecto=None))
    for k in ("quejoso", "recurrente", "responsable", "terceros"):
        _uno(k, (datos or {}).get(k) if isinstance(datos, dict) else None)
    return out


def _regla_articulo_del_papel(secs, ficha, datos, out):
    nombres = _nombres_con_articulo(ficha, datos)
    if not nombres:
        return
    # El mismo cargo con los dos artículos en el papel (dos personas): no se sabe.
    generos = {}
    for _c, g, _v, _rx, llano in nombres:
        generos.setdefault(llano, set()).add(g)
    for campo, g, valor, rx_cuerpo, llano in nombres:
        if len(generos[llano]) > 1:
            continue
        rx = re.compile(r"(?<![\w])(?P<art>el|la|los|las|del|al)\s+(?:" + rx_cuerpo + r")(?![\w])", re.I)
        donde, ejemplo = [], ""
        for s in secs:
            if s["zona"] == "caratula":
                continue
            for m in rx.finditer(s["texto"]):
                if _ART_GENERO[m.group("art").lower()] != g:
                    if s["nombre"] not in donde:
                        donde.append(s["nombre"])
                    ejemplo = ejemplo or _extracto(s["texto"], m.start(), m.end(), 20)
                    break
        if donde:
            cargo = valor.split()[1] if len(valor.split()) > 1 else valor
            out.append(("f", f"EL ARTÍCULO DEL CARGO NO ES EL DEL PAPEL EN {_lista_apartados(donde)}: la ficha de "
                             f"trámite dice «{valor[:90]}» ({campo}) y el texto «{ejemplo}». «{cargo}» sirve a los "
                             f"dos géneros: el que lo dice es el artículo del auto o del escrito, y ése se conserva "
                             f"en todo el documento.",
                        {"clave": campo}))


# ═══════════════════════════════════════════════════════════════════════════
# (g) LA SUPLETORIEDAD QUE CORRESPONDE A LA SEDE
# ═══════════════════════════════════════════════════════════════════════════
# David, 3-oct-2026: el Código Federal de Procedimientos Civiles en toda la
# república; el Código Nacional sólo en la Ciudad de México, donde ya opera.
_RX_CDMX = re.compile(r"ciudad\s+de\s+m[ée]xico|\bcdmx\b|m[ée]xico,?\s+d\.?\s*f\.?|distrito\s+federal", re.I)
# RONDA 3 (rev_2, 3-oct-2026): también «aplicado supletoriamente a la Ley de
# Amparo» y «en relación con el artículo 2o. de la Ley de Amparo» (la remisión
# al supletorio sin la palabra «supletoria»); con esas dos formas un «artículo
# 269 del Código Nacional…» en un proyecto de Querétaro pasaba sin aviso.
_RX_SUPLETORIA_LA = re.compile(
    r"supletori(?:[ao]s?|amente)\b[^.;]{0,140}?Ley\s+de\s+Amparo|"
    r"\bart[íi]culo\s+2o?\.?(?:\s*,\s*(?:primer|segundo|tercer|[úu]ltimo)\s+p[áa]rrafo\s*,)?\s+de\s+la\s+"
    r"Ley\s+de\s+Amparo", re.I)
_RX_CNPCF = re.compile(r"C[óo]digo\s+Nacional\s+de\s+Procedimientos\s+Civiles", re.I)
_RX_CFPC = re.compile(r"C[óo]digo\s+Federal\s+de\s+Procedimientos\s+Civiles", re.I)


def _es_cdmx(ficha, datos, secs):
    """True / False / None (no se sabe: entonces no se acusa).

    LA SEDE LA DECIDE `tipos_asunto` (ronda 3, rev_2): el compositor y la
    existencia citan el código que dice `tipos_asunto.supletorio`, así que la
    verja pregunta a la misma regla —con el mismo orden de fuentes: la sede de
    la ficha y, si falta, la del encargo— y no a `ficha.sede.cdmx`. Con dos
    reglas distintas, un tribunal del Primer Circuito con la ciudad escrita
    «Cd. de México» salía con el CFPC en la existencia y la verja acusaba esa
    misma fórmula. `ficha.sede.cdmx` sólo vale si `tipos_asunto` no está."""
    sede = _g(ficha, "sede") if isinstance(ficha, dict) else None
    sede = sede if isinstance(sede, dict) else {}
    d = datos or {}
    trib = str(sede.get("tribunal") or d.get("tribunal") or "")
    ciudad = str(sede.get("ciudad") or d.get("ciudad") or "")
    _ta = None
    try:
        import tipos_asunto as _ta
    except Exception:
        _ta = None

    def _decide(tr, ci):
        if _ta is None:
            return None
        try:
            if hasattr(_ta, "es_cdmx"):
                r = _ta.es_cdmx(tr, ci)
                return r if isinstance(r, bool) else None
            if hasattr(_ta, "supletorio"):
                r = _ta.supletorio(tr, ci)
                if isinstance(r, dict) and isinstance(r.get("cdmx"), bool) and not r.get("aviso"):
                    return r["cdmx"]
        except Exception:
            return None
        return None

    if trib or ciudad:
        r = _decide(trib, ciudad)
        if r is not None:
            return r
    if _ta is None and isinstance(sede.get("cdmx"), bool):
        return sede["cdmx"]
    # Del propio texto: «Ciudad de México, … Resolución del …» o el circuito.
    pro = " ".join(s["texto"] for s in _de(secs, zona="proemio"))
    if not pro:
        return None
    m = _RX_TRIB_PROEMIO.search(pro)
    if not m:
        return None
    trib_p, ciudad_p = m.group("t"), pro[:m.start()]
    r = _decide(trib_p, ciudad_p.strip(" .,\n"))
    if r is not None:
        return r
    if _RX_CDMX.search(ciudad_p) or _RX_CDMX.search(trib_p):
        return True
    if re.search(r"\bdel\s+primer\s+circuito\b|\bI\s+circuito\b", _plano(trib_p)):
        return True
    return False


# LA SUPLETORIEDAD DE OTRA LEY NO ES LA DE LA LEY DE AMPARO (C7, sexta ronda,
# 3-oct-2026). En lo agrario el surtimiento se funda en el Código Nacional «de
# aplicación supletoria en términos del artículo 167 de la Ley Agraria» (sede en
# la Ciudad de México) o en el artículo 321 del Código Federal «de aplicación
# supletoria en materia agraria» (fuera de ella, mientras no haya declaratoria),
# y la misma frase sigue con «…el plazo de quince días del artículo 17 de la Ley
# de Amparo»: la regla leía «supletoria … Ley de Amparo» y acusaba el Código
# Nacional fuera de la Ciudad de México cuando el secretario lo elige porque ya
# rige para ese juicio (o el Federal en ella). Si entre «supletori…» y «Ley de
# Amparo» se nombra otra ley o la materia agraria, el código es supletorio de
# ésa y la sede no lo decide aquí.
_RX_OTRA_LEY_SUPLIDA = re.compile(
    r"Ley\s+Agraria|materia\s+agraria|C[óo]digo\s+de\s+Comercio|Ley\s+Federal\s+(?:de\s+Procedimiento|del\s+Trabajo)",
    re.I)


def _regla_supletorio(t, secs, ficha, datos, out):
    cdmx = _es_cdmx(ficha, datos, secs)
    if cdmx is None:
        return
    for s in secs:
        for m in _RX_SUPLETORIA_LA.finditer(s["texto"]):
            if _RX_OTRA_LEY_SUPLIDA.search(m.group(0)):
                continue
            antes = s["texto"][max(0, m.start() - 260):m.start()]
            # El código que se cita es el más cercano antes de «supletoria».
            ultimo = None
            for rx, nombre in ((_RX_CNPCF, "cnpcf"), (_RX_CFPC, "cfpc")):
                for mm in rx.finditer(antes):
                    if ultimo is None or mm.start() > ultimo[0]:
                        ultimo = (mm.start(), nombre)
            if not ultimo:
                # «… en términos del artículo 2o. de la Ley de Amparo, se aplica
                # supletoriamente el Código …»: el código va DESPUÉS.
                despues = s["texto"][m.end():m.end() + 220]
                despues = re.split(r"\.\s", despues)[0]
                for rx, nombre in ((_RX_CNPCF, "cnpcf"), (_RX_CFPC, "cfpc")):
                    mm = rx.search(despues)
                    if mm and (ultimo is None or mm.start() < ultimo[0]):
                        ultimo = (mm.start(), nombre)
            if not ultimo:
                continue
            if ultimo[1] == "cnpcf" and not cdmx:
                out.append(("g", f"SUPLETORIEDAD QUE NO CORRESPONDE A LA SEDE EN {s['nombre'].upper()}: se cita el "
                                 f"Código Nacional de Procedimientos Civiles y Familiares como supletorio de la Ley "
                                 f"de Amparo fuera de la Ciudad de México. Aquí rige el Código Federal de "
                                 f"Procedimientos Civiles (arts. 129 y 202)."))
                return
            if ultimo[1] == "cfpc" and cdmx:
                out.append(("g", f"SUPLETORIEDAD QUE NO CORRESPONDE A LA SEDE EN {s['nombre'].upper()}: se cita el "
                                 f"Código Federal de Procedimientos Civiles como supletorio de la Ley de Amparo en la "
                                 f"Ciudad de México, donde ya opera el Código Nacional de Procedimientos Civiles y "
                                 f"Familiares (arts. 312, fracciones II y VIII, y 344)."))
                return


# ═══════════════════════════════════════════════════════════════════════════
# (h) NORMAS ABROGADAS O FUERA DE LUGAR
# ═══════════════════════════════════════════════════════════════════════════
# LOPJF de 2021 y de 1995: 37, 38, 41, 124, 143, 145, 163 (la vigente, DOF
# 20-dic-2024: 35 y 210 para competencia; 229 para inhábiles). AG 1/2023 de la
# SCJN: lo abrogó el AG 11/2025 el 4-sep-2025 (medido: aún lo citan 9
# sentencias del propio tribunal después; no por eso es vigente).
_LOPJF_VIEJOS = {37, 38, 41, 124, 143, 145, 163}
_RX_LOPJF = re.compile(r"(?:de\s+la\s+)?Ley\s+Org[áa]nica\s+del\s+Poder\s+Judicial\s+de\s+la\s+Federaci[óo]n|\bLOPJF\b")
_RX_LOPJF_ANTERIOR = re.compile(r"^\W{0,3}(?:abrogada|anterior|de\s+(?:mil\s+novecientos\s+noventa\s+y\s+cinco|"
                                r"dos\s+mil\s+veintiuno)|\(?(?:1995|2021)\)?|vigente\s+(?:hasta|en\s+la\s+[ée]poca)|"
                                r"publicada\s+en\s+el\s+Diario\s+Oficial\s+de\s+la\s+Federaci[óo]n\s+el\s+"
                                r"(?:siete\s+de\s+junio|veintis[ée]is\s+de\s+mayo))", re.I)
_RX_AG_1_2023 = re.compile(r"Acuerdo\s+General(?:\s+(?:Plenario|n[úu]mero))?\s+1/2023|\bAG\s+1/2023\b", re.I)
_RX_PLENO_CONSEJO = re.compile(r"(?<!otrora )(?<!entonces )(?<!extinto )(?<!extinguido )(?<!desaparecido )"
                               r"Pleno\s+del\s+Consejo\s+de\s+la\s+Judicatura", re.I)


def _articulos_antes(texto: str, fin: int) -> list:
    """Los números de artículo del tramo que precede a «de la Ley Orgánica…»."""
    tramo = texto[max(0, fin - 220):fin]
    # El tramo empieza donde acaba la cita de la ley anterior de la cadena
    # («…de la Ley de Amparo; 35, fracción I, inciso c) y 210 de la Ley Orgánica»).
    corte = -1
    for m in re.finditer(r";|\bLey\b[^,;]{0,90}|Constituci[óo]n[^,;]{0,30}|C[óo]digo[^,;]{0,90}|"
                         r"Reglamento[^,;]{0,90}|Acuerdo[^,;]{0,60}", tramo):
        corte = m.end()
    tramo = tramo[corte:] if corte >= 0 else tramo
    nums = []
    for m in re.finditer(r"(?<![\w/.])(\d{1,3})(?:\s*[o°º]\.?)?(?=\s*(?:,|\s+y\s|\s+de\s+la\s+Ley|\s+fracci|"
                         r"\s+p[áa]rrafo|\s+inciso|\s+bis\b|\s*$|\s+del?\s+la\b))", tramo):
        nums.append(int(m.group(1)))
    return nums


def _regla_normas(t, secs, tipo, out):
    for s in secs:
        cuerpo = s["texto"]
        # LA ANTESALA DE LOS RESOLUTIVOS («Por lo expuesto, fundado y con apoyo
        # en los artículos… 37, fracción V, de la Ley Orgánica…»): en 13 RF de
        # 2025-2026 citaba la ley abrogada (oro RF). Va al final del último
        # considerando o al principio de los resolutivos, según quién componga.
        m_ant = re.search(r"Por\s+lo\s+(?:antes\s+)?expuesto", cuerpo)
        procesal = (s["zona"] == "considerando" and bool(s["etiquetas"] & {"competencia", "oportunidad",
                                                                          "legitimacion", "procedencia"})
                    or s["zona"] == "resolutivos")
        tramos = [cuerpo] if procesal else ([cuerpo[m_ant.start():]] if m_ant else [])
        for cuerpo_n in tramos:
            for m in _RX_LOPJF.finditer(cuerpo_n):
                if _RX_LOPJF_ANTERIOR.search(cuerpo_n[m.end():m.end() + 140]):
                    continue
                viejos = sorted(set(_articulos_antes(cuerpo_n, m.start())) & _LOPJF_VIEJOS)
                if viejos:
                    arts = (f"los artículos {', '.join(map(str, viejos))}" if len(viejos) > 1
                            else f"el artículo {viejos[0]}")
                    out.append(("h", f"LEY ORGÁNICA ABROGADA EN {s['nombre'].upper()}: se cita {arts} de la Ley "
                                     f"Orgánica del Poder Judicial de la Federación, numeración de la ley abrogada. "
                                     f"La vigente (DOF 20-dic-2024): 35 y 210 para la competencia; 229 para los "
                                     f"inhábiles."))
                    break
        m = _RX_AG_1_2023.search(cuerpo)
        if m:
            # EL SENTIDO DEL ABROGAR (rev_2, ronda 3): el aviso decía «…, que
            # abrogó el Acuerdo General 11/2025», al revés. El AG 1/2023 es el
            # abrogado; el 11/2025 (4-sep-2025) es el que lo abrogó.
            out.append(("h", f"ACUERDO ABROGADO EN {s['nombre'].upper()}: se cita el «{_un_renglon(m.group(0))}» de "
                             f"la Suprema Corte, abrogado por el Acuerdo General 11/2025 (4-sep-2025); la delegación "
                             f"se funda en el vigente."))
        # «PLENO DEL CONSEJO» SIN «OTRORA»: en la competencia (el oro AD lo mide
        # como error en 12 competencias de 2025-2026) y en la antesala. En la
        # cola de la sesión anterior a abril de 2026 el corpus escribe «del
        # Pleno del (otrora) Consejo», con el «otrora» a veces omitido; no se
        # acusa allí —el proyecto nuevo cita el Acuerdo 6/2026 del OAJ—.
        m = (_RX_PLENO_CONSEJO.search(cuerpo) if ("competencia" in s["etiquetas"] or s["zona"] == "resolutivos")
             else (_RX_PLENO_CONSEJO.search(cuerpo, m_ant.start()) if m_ant else None))
        if m:
            out.append(("h", f"«PLENO DEL CONSEJO» SIN «OTRORA» EN {s['nombre'].upper()}: "
                             f"«{_extracto(cuerpo, m.start(), m.end(), 40)}». El Consejo de la Judicatura Federal "
                             f"ya no existe: sus acuerdos se citan «del Pleno del otrora Consejo…»."))
        if s["zona"] == "resultando" and "sesion" in s["etiquetas"]:
            m = re.search(r"\bart[íi]culo\s+84\b", cuerpo)
            if m:
                out.append(("h", f"PRECEPTO EQUIVOCADO EN {s['nombre'].upper()}: «{m.group(0)}» — la lista y la "
                                 f"sesión son del artículo 184 de la Ley de Amparo."))
        if (s["zona"] == "resultando" and (s["etiquetas"] & {"turno", "returno"})
                and tipo in ("amparo_directo", "amparo_revision", "revision_fiscal")):
            m = re.search(r"\bart[íi]culo\s+101\b", cuerpo)
            if m:
                out.append(("h", f"PRECEPTO EQUIVOCADO EN {s['nombre'].upper()}: el artículo 101 de la Ley de "
                                 f"Amparo regula el trámite de la QUEJA; el turno es el "
                                 f"{'183' if tipo == 'amparo_directo' else '92'} de la Ley de Amparo."))


# ═══════════════════════════════════════════════════════════════════════════
# (i) EN QUÉ PARÓ EL JUICIO (AR): el verbo del resultando contra la ficha
# ═══════════════════════════════════════════════════════════════════════════
_RX_V_SOBRESEE = re.compile(r"sobresey[óo]|se\s+sobresee|decret[óo]\s+el\s+sobreseimiento|sobresee\s+en\s+el\s+juicio", re.I)
_RX_V_CONCEDE = re.compile(r"(?:conced(?:i[óo]|e)|otorg[óo])\s+(?:a\s+[^.;]{0,60}?\s+)?(?:el\s+)?amparo|"
                           r"(?<!no )ampar[óa]\s+y\s+proteg|conced(?:i[óo]|e)\s+la\s+protecci[óo]n", re.I)
_RX_V_NIEGA = re.compile(r"neg[óo]\s+(?:a\s+[^.;]{0,60}?\s+)?(?:el\s+)?amparo|\bno\s+ampar[óa]|"
                         r"neg[óo]\s+la\s+protecci[óo]n", re.I)


def _partes_resolvio(v: str) -> set:
    """{"sobresee", "concede", "niega"} de una clave compuesta («sobresee_niega»)."""
    hay = set()
    v = re.sub(r"\bno\s+ampar\w*", "niega", str(v or ""))
    for trozo in re.split(r"[_\s,/+-]+|\by\b", v):
        if trozo.startswith("sobre"):
            hay.add("sobresee")
        elif trozo.startswith(("conce", "ampar", "otorg")):
            hay.add("concede")
        elif trozo.startswith(("nieg", "neg")):
            hay.add("niega")
    return hay


def _resolvio(ficha, datos) -> str:
    """concede | niega | sobresee | mixto | sobresee_concede | sobresee_niega |
    concede_niega | "" (no se sabe: no se coteja).

    LAS CLAVES COMPUESTAS SON MIXTOS, NO SOBRESEIMIENTOS. `fase_rama` guarda
    «sobresee_niega» y «sobresee_concede» (_VALIDOS_A_QUO), y `ficha_tramite`
    las convierte en «mixto» + `acto.resolvio_mixto`; pero si la clave cruda
    llega a `acto.resolvio` o a `resolvio_a_quo`, leerla por su prefijo daba
    «sobresee» y la verja acusaba el verbo mixto del compositor, que es el
    correcto (3-oct-2026). Con «mixto» a secas se busca la clave completa en
    `acto.resolvio_mixto` o en `resolvio_a_quo`, para saber qué hizo la otra
    parte del fallo."""
    d = datos or {}
    v = _plano(_g(ficha, "acto", "resolvio") or d.get("resolvio_a_quo") or "")
    if not v:
        return ""
    if v.startswith("mixt"):
        for otro in (_g(ficha, "acto", "resolvio_mixto"), d.get("resolvio_a_quo")):
            partes = _partes_resolvio(_plano(otro))
            if "sobresee" in partes and len(partes) == 2:
                return "sobresee_" + (partes - {"sobresee"}).pop()
        return "mixto"
    partes = _partes_resolvio(v)
    if len(partes) >= 2:
        if partes == {"concede", "niega"}:
            return "concede_niega"
        if len(partes) == 3:
            return "mixto"
        return "sobresee_" + (partes - {"sobresee"}).pop()
    for pref, clave in (("conce", "concede"), ("ampar", "concede"), ("nieg", "niega"),
                        ("neg", "niega"), ("no ampar", "niega"), ("sobre", "sobresee")):
        if v.startswith(pref):
            return clave
    return ""


def _regla_verbo(t, secs, ficha, datos, tipo, out):
    if tipo != "amparo_revision":
        return
    if _g(ficha, "acto", "incidente") is True or str(_g(ficha, "clase_recurrida")).startswith("interlocutoria"):
        return
    esperado = _resolvio(ficha, datos)
    if not esperado:
        return
    hallados = set()
    for s in _de(secs, zona="resultando", sin=("sesion",)):
        cuerpo = s["texto"]
        # Lo que pidió la quejosa no es lo que resolvió el juzgado: el bloque de
        # ACTOS RECLAMADOS se salta.
        cuerpo = re.split(r"ACTOS?\s+RECLAMADOS?\s*:", cuerpo)[0] if "presentacion" in s["etiquetas"] else cuerpo
        if _RX_V_SOBRESEE.search(cuerpo):
            hallados.add("sobresee")
        if _RX_V_CONCEDE.search(cuerpo):
            hallados.add("concede")
        if _RX_V_NIEGA.search(cuerpo):
            hallados.add("niega")
    # EL MIXTO es el sobreseimiento MÁS una decisión de fondo. Con la clave
    # completa («sobresee_niega») se exige además que el resultando diga ese
    # fondo; no se exige que calle el otro, porque un fallo que sobresee,
    # concede y niega a la vez sólo cabe en una de las dos claves del catálogo.
    mixto = esperado == "mixto" or esperado.startswith("sobresee_")
    fondo = esperado.split("_", 1)[1] if esperado.startswith("sobresee_") else ""
    malo = False
    if hallados:
        if esperado == "concede":
            malo = "niega" in hallados or "concede" not in hallados
        elif esperado == "niega":
            malo = "concede" in hallados or "niega" not in hallados
        elif esperado == "sobresee":
            malo = bool(hallados - {"sobresee"})
        elif mixto:
            malo = "sobresee" not in hallados or len(hallados) < 2 or bool(fondo and fondo not in hallados)
        elif esperado == "concede_niega":
            # «en una parte se concedió el amparo y en otra se negó»: el «negó»
            # elíptico no se lee, así que basta con una de las dos y sin
            # sobreseimiento.
            malo = "sobresee" in hallados or not (hallados & {"concede", "niega"})
    if malo:
        dice = " y ".join(sorted({"concede": "concedió", "niega": "negó", "sobresee": "sobreseyó"}[h]
                                 for h in hallados))
        quiere = {"concede": "concedió el amparo", "niega": "negó el amparo", "sobresee": "sobreseyó en el juicio",
                  "mixto": "en una parte sobreseyó y en otra resolvió el fondo",
                  "sobresee_concede": "en una parte sobreseyó y en otra concedió el amparo",
                  "sobresee_niega": "en una parte sobreseyó y en otra negó el amparo",
                  "concede_niega": "en una parte concedió el amparo y en otra lo negó"}[esperado]
        out.append(("i", f"EN QUÉ PARÓ EL JUICIO NO CUADRA: los resultandos dicen que el juzgado {dice}, y la "
                         f"sentencia recurrida {quiere} (leído del papel). El resultando se corrige con lo que "
                         f"dice la sentencia."))
    if mixto:
        res = " ".join(s["texto"] for s in _de(secs, zona="resolutivos"))
        if res and not re.search(r"sobrese", res, re.I):
            out.append(("i", "EL SOBRESEIMIENTO NO SE RESUELVE: el juzgado sobreseyó en una parte y los resolutivos "
                             "no lo mencionan. Si nadie lo recurrió, «Queda firme el sobreseimiento…»; si se "
                             "recurrió, se confirma o se revoca."))


# ═══════════════════════════════════════════════════════════════════════════
# (j) FECHAS IMPOSIBLES: las de la ficha y las del propio texto
# ═══════════════════════════════════════════════════════════════════════════
_RX_F_PRESENTACION = re.compile(
    r"(?:presentad[oa]|recibid[oa]|depositad[oa])\s+(?:en\s+[^,;]{3,120}?\s+)?el\s+$|"
    r"(?:^|[.:]\s+)(?:El|el)\s+$", re.I)
_RX_F_ACTO = re.compile(
    r"(?:(?:contra|en\s+contra\s+de|impugn\w+|reclam[óo]|actos?\s+reclamados?\s*:?)\s+(?:la|el)\s+|"
    r"(?:contra|en\s+contra)\s+del\s+)"
    r"(?:sentencia|resoluci[óo]n|laudo|auto|interlocutoria)"
    r"(?:\s+(?:definitiva|interlocutoria|reclamad[oa]|recurrid[oa]))?\s+(?:dictad[oa]\s+(?:el\s+)?|de\s+(?:fecha\s+)?)$",
    re.I)
_RX_F_NOTIF = re.compile(r"(?:se\s+)?notific[óo]\s+[^.;]{0,90}?\s+el\s+$", re.I)
# La fecha de la demanda en el primer resultando de la queja (sexta ronda):
# «presentado el…», «recibido ante la Oficina…, el…», «depositado en…, el…».
_RX_F_DEMANDA = re.compile(r"(?:presentad[oa]|recibid[oa]|depositad[oa])\s+(?:(?:en|ante)\s+[^;]{3,220}?,?\s+)?el\s+$",
                           re.I)


def _par_presentacion_acto(cuerpo: str):
    """(presentación, acto) leídos en la fórmula de UN resultando; cada uno (date, literal) o None."""
    pres = acto = None
    for d, i, j, lit in _fechas(cuerpo):
        if d is None:
            continue
        antes = cuerpo[max(0, i - 160):i]
        if acto is None and _RX_F_ACTO.search(antes):
            acto = (d, lit)
        elif pres is None and _RX_F_PRESENTACION.search(antes):
            pres = (d, lit)
    return pres, acto


_RX_TRAS_AUDIENCIA = re.compile(r"^\s*,?\s*se\s+celebr[óo]\s+la\s+audiencia\s+(?:constitucional|incidental)"
                                r"(?P<y>\s+y\s+se\s+dict[óo]\s+(?:la\s+)?(?:sentencia|resoluci[óo]n))?", re.I)
_RX_TRAS_SENTENCIA = re.compile(r"^\s*,?\s*(?:se\s+)?dict[óo]\s+(?:la\s+)?sentencia", re.I)


def _lo_que_ya_dice_validar(ficha) -> list:
    """Los avisos de `ficha_tramite.validar` sobre una COPIA de la ficha, para
    comparar (en plano); [] si no está o tropieza."""
    return [_plano(x) for x in _validar_copia(ficha)[0]]


def _validar_copia(ficha) -> tuple:
    """(avisos tal cual, claves de `fechas_imposibles` que deriva ella misma) de
    `ficha_tramite.validar` sobre una COPIA de la ficha con la lista vacía (la
    función escribe en `fechas_imposibles`); ([], set()) si no está o tropieza."""
    if not _es_ficha(ficha):
        return [], set()
    try:
        import copy
        import ficha_tramite as _ft
        f = getattr(_ft, "validar", None)
        if not callable(f):
            return [], set()
        c = copy.deepcopy(ficha)
        c["fechas_imposibles"] = []
        avisos = [str(x) for x in (f(c) or []) if x]
        return avisos, {str(x) for x in (c.get("fechas_imposibles") or [])}
    except Exception:
        return [], set()


# EL MISMO PAR DE FECHAS, DICHO YA POR OTRA PIEZA (ronda 4, 3-oct-2026, E11).
# AR 448 del banco: «FECHA IMPOSIBLE: lo reclamado/recurrido es de quince de
# julio de dos mil veinticinco y el escrito se presentó el trece de mayo…» (la
# verja, leída en el texto) y «FECHA IMPOSIBLE: el recurso se presentó el
# 13/05/2025 y la resolución recurrida es del 15/07/2025» (validar, que el
# compositor suma). Un hecho, dos avisos con formatos distintos. Se reconoce
# por las FECHAS, escritas como sea («13/05/2025», «2025-05-13» o en letra),
# en los avisos de validar sobre una copia y en los `avisos_previos`.
_RX_FECHA_DMY = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_RX_FECHA_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_RX_MES = re.compile(r"enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre",
                     re.I)


def _fechas_del_aviso(a: str) -> set:
    out = set()
    for m in _RX_FECHA_DMY.finditer(a):
        try:
            out.add(_dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1))))
        except ValueError:
            pass
    for m in _RX_FECHA_ISO.finditer(a):
        try:
            out.add(_dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            pass
    if _RX_MES.search(a):
        out |= {f[0] for f in _fechas(a) if f[0]}
    return out


def _ya_dicho_imposible(ya, *fechas) -> bool:
    """¿Algún aviso de FECHA IMPOSIBLE de otra pieza ya nombra todas estas fechas?"""
    fs = {f for f in fechas if isinstance(f, _dt.date)}
    if not fs:
        return False
    for a in ya or ():
        a = str(a or "")
        if "imposible" in _plano(a) and fs <= _fechas_del_aviso(a):
            return True
    return False


def _cronologia_amparo_indirecto(secs, acto, out, ficha=None, ya_raw=None, ya_palabras=None):
    dem = aud = sen = None
    for s in _de(secs, zona="resultando", sin=("sesion",)):
        cuerpo = s["texto"]
        if "presentacion" in s["etiquetas"] and re.search(r"demanda", s["rotulo"], re.I):
            p, _ = _par_presentacion_acto(_RX_BLOQUE_RECLAMADO.split(cuerpo)[0])
            if p and dem is None:
                dem = p
        for d, i, j, lit in _fechas(cuerpo):
            if d is None:
                continue
            despues = cuerpo[j:j + 120]
            ma = _RX_TRAS_AUDIENCIA.match(despues)
            if ma and aud is None:
                aud = (d, lit)
                if ma.group("y") and sen is None:
                    sen = (d, lit)
            elif _RX_TRAS_SENTENCIA.match(despues) and sen is None:
                sen = (d, lit)
    sen = sen or acto
    problemas = []
    # LO QUE YA DICE LA FICHA NO SE REPITE (AR 448 del banco, ronda 3): el
    # compositor suma los avisos de `ficha_tramite.validar`, que ya dicen «la
    # audiencia … es posterior a la sentencia» con las fechas de la ficha; lo
    # del texto sólo se dice si ésa no lo dijo (el texto se apartó de la ficha).
    # RONDA 4: también lo que ya dijo otra pieza con las mismas dos fechas
    # (`avisos_previos`, o las entradas de `fechas_imposibles` de la ficha), y
    # con las fechas del texto aunque el aviso use otras palabras.
    if ya_raw is None:
        ya_raw = _validar_copia(ficha)[0] if (aud or dem) else []
    # Por PALABRAS sólo lo de validar y la ficha (como antes); por FECHAS, además
    # lo de las otras piezas: un aviso cualquiera que diga «audiencia» y
    # «posterior» de otro par no calla éste.
    ya = [_plano(x) for x in (ya_palabras if ya_palabras is not None else ya_raw)]

    def _dicho(par, *palabras):
        return any(all(p in v for p in palabras) for v in ya) or _ya_dicho_imposible(ya_raw, *par)

    if aud and sen and aud[0] > sen[0] and not _dicho((aud[0], sen[0]), "audiencia", "posterior"):
        problemas.append(f"la audiencia ({aud[1]}) es posterior a la sentencia ({sen[1]})")
    if dem and sen and dem[0] > sen[0]:
        if not _dicho((dem[0], sen[0]), "demanda", "posterior"):
            problemas.append(f"la demanda de amparo ({dem[1]}) es posterior a la sentencia que la resolvió "
                             f"({sen[1]})")
    elif dem and aud and dem[0] > aud[0] and not _dicho((dem[0], aud[0]), "demanda", "posterior"):
        problemas.append(f"la demanda de amparo ({dem[1]}) es posterior a la audiencia ({aud[1]})")
    if problemas:
        out.append(("j", "FECHA IMPOSIBLE EN EL TRÁMITE DEL JUICIO DE AMPARO INDIRECTO: " + "; ".join(problemas)
                         + ". El juicio va demanda → audiencia → sentencia; una de esas fechas está mal leída y "
                           "hay que corregirla en la ficha de trámite antes de firmar."))


def _regla_cronologia(t, secs, ficha, datos, tipo, out, previos=None):
    # LO QUE YA DICE `validar` NO SE REPITE (ronda 4, 3-oct-2026, E11). La ficha
    # que llega de `ficha_tramite.armar` trae en `fechas_imposibles` las CLAVES
    # que escribió validar («presentacion < acto.fecha»), y el aviso con las
    # fechas ya lo suma el compositor: la verja decía además «FECHA IMPOSIBLE EN
    # LA FICHA DE TRÁMITE: presentacion < acto.fecha», el mismo hecho con la
    # clave cruda. Sólo se dice lo que validar no deriva por sí misma (lo que
    # escribió otro lector, o la ficha sin validar a mano).
    av_val, claves_val = _validar_copia(ficha)
    ya_palabras = list(av_val)
    if isinstance(ficha, dict):
        for f in (ficha.get("fechas_imposibles") or []):
            ya_palabras.append(f"FECHA IMPOSIBLE: {f}")
            if str(f) in claves_val:
                continue
            out.append(("j", f"FECHA IMPOSIBLE EN LA FICHA DE TRÁMITE: {f}. Mientras no se corrija, la "
                             f"oportunidad no se puede declarar."))
    ya_raw = ya_palabras + [str(a) for a in (previos or []) if a]
    # LA CRONOLOGÍA DEL PROPIO TEXTO: una demanda de marzo de 2025 contra una
    # sentencia de enero de 2026 (AD 174 y 722, 16 proyectos) se declaraba
    # oportuna. Se lee sólo en las fórmulas, dentro de cada resultando:
    # «presentado el X … contra la sentencia dictada el Y».
    pres_asunto = acto = None
    for s in _de(secs, zona="resultando", etiqueta="presentacion"):
        p, a = _par_presentacion_acto(s["texto"])
        if p and a and a[0] > p[0] and not _ya_dicho_imposible(ya_raw, p[0], a[0]):
            out.append(("j", f"FECHA IMPOSIBLE EN {s['nombre'].upper()}: lo reclamado/recurrido es de {a[1]} y "
                             f"el escrito se presentó el {p[1]}, antes de que existiera. Una de las dos está mal; el "
                             f"cómputo de oportunidad no vale hasta corregirla."))
        # La presentación que cuenta para el número: la demanda en el directo,
        # el recurso en los demás.
        del_asunto = (tipo == "amparo_directo" or re.search(r"interposici|recurso", s["rotulo"], re.I))
        if p and del_asunto and pres_asunto is None:
            pres_asunto = p
        if a and acto is None and (tipo == "amparo_directo" or del_asunto):
            acto = a
    if acto is None:
        for s in _de(secs, zona="visto"):
            for d, i, j, lit in _fechas(s["texto"]):
                if d and _RX_F_ACTO.search(s["texto"][max(0, i - 160):i]):
                    acto = (d, lit)
                    break
    if pres_asunto and acto and acto[0] > pres_asunto[0] and not any(x[0] == "j" and "antes de que existiera" in x[1]
                                                                      for x in out) \
            and not _ya_dicho_imposible(ya_raw, pres_asunto[0], acto[0]):
        out.append(("j", f"FECHA IMPOSIBLE: lo reclamado/recurrido es de {acto[1]} y el escrito se presentó el "
                         f"{pres_asunto[1]}, antes de que existiera. Una de las dos está mal; el cómputo de "
                         f"oportunidad no vale hasta corregirla."))
    num = _num(_g(ficha, "numero")) or _num((datos or {}).get("numero"))
    if pres_asunto and num and int(num.split("/")[1]) < pres_asunto[0].year and not any(
            "imposible" in _plano(a) and num in str(a) and pres_asunto[0] in _fechas_del_aviso(str(a))
            for a in ya_raw):
        out.append(("j", f"FECHA IMPOSIBLE: el asunto es el {num} y el escrito se presentó el {pres_asunto[1]}: "
                         f"no se puede registrar en un año anterior a su presentación."))
    # LA CRONOLOGÍA DEL JUICIO DE AMPARO INDIRECTO (AR; ronda 3, rev_3). En el
    # banco de oráculo, 3 de 8 AR salieron con la audiencia posterior a la
    # sentencia (307, 448 y 60) y dos con la demanda posterior a la sentencia
    # que la resolvió, sin un aviso: la regla de arriba sólo lee «presentado el
    # X … contra la sentencia dictada el Y» dentro de un mismo resultando, y el
    # del trámite del juicio dice «el X se celebró la audiencia constitucional y
    # el Y se dictó sentencia». Si la ficha ya trae sus fechas imposibles, ya
    # están dichas arriba y no se repiten.
    # RONDA 4: la ficha de producción trae SIEMPRE las claves de validar (una
    # fecha del turno anterior a la admisión callaba la cronología del juicio
    # entera); ahora se corre siempre y calla lo ya dicho, por sus fechas.
    if tipo == "amparo_revision":
        _cronologia_amparo_indirecto(secs, acto, out, ficha, ya_raw, ya_palabras)
    # LA QUEJA QUE ABRE CON LA DEMANDA (C3, sexta ronda, 3-oct-2026): el auto
    # recurrido —el que desechó la demanda, el que proveyó sobre la suspensión—
    # se dictó en un juicio que empezó con esa demanda; una demanda posterior al
    # auto es una fecha mal leída. Lo que ya dijo `validar` («demanda.fecha >
    # acto.fecha», por sus fechas) no se repite.
    # SÓLO LA PRIMERA FECHA DEL RESULTANDO Y SÓLO CON SU FÓRMULA («Por escrito
    # presentado el…», «recibido ante la Oficina de Correspondencia Común…, el…»):
    # el tribunal cuenta a veces en ese mismo resultando todo el trámite («…El
    # dos de octubre… interpuso recurso de queja», Q 215/2024 de la calibración
    # ampliada) y la regla general de la presentación leía esa fecha como la de
    # la demanda.
    if tipo == "queja" and acto:
        for s in _de(secs, zona="resultando", etiqueta="presentacion"):
            if not re.search(r"demanda", s["rotulo"], re.I):
                continue
            cuerpo_d = _RX_BLOQUE_RECLAMADO.split(s["texto"])[0]
            f1 = next((f for f in _fechas(cuerpo_d) if f[0]), None)
            p = ((f1[0], f1[3]) if f1 and _RX_F_DEMANDA.search(cuerpo_d[max(0, f1[1] - 260):f1[1]]) else None)
            if p and p[0] > acto[0] and not _ya_dicho_imposible(ya_raw, p[0], acto[0]):
                out.append(("j", f"FECHA IMPOSIBLE EN {s['nombre'].upper()}: la demanda de amparo ({p[1]}) es "
                                 f"posterior al auto recurrido ({acto[1]}), que se dictó en ese juicio. Una de las dos "
                                 f"está mal leída: se corrige en la ficha de trámite (demanda.fecha o acto.fecha).",
                            {"clave": "demanda.fecha"}))
            break
    if acto:
        for s in _de(secs, zona="considerando", etiqueta="oportunidad") or _de(secs, zona="considerando",
                                                                                etiqueta="legitimacion"):
            for d, i, j, lit in _fechas(s["texto"]):
                if d and _RX_F_NOTIF.search(s["texto"][max(0, i - 110):i]) and d < acto[0] \
                        and not _ya_dicho_imposible(ya_raw, d, acto[0]):
                    out.append(("j", f"FECHA IMPOSIBLE EN {s['nombre'].upper()}: la notificación es del {lit} y lo "
                                     f"notificado es de {acto[1]}. No se notifica lo que aún no se dicta."))
                    break
            break


# ═══════════════════════════════════════════════════════════════════════════
# (l) LA CLASE DEL ACTO: sentencia, resolución, laudo o auto (ronda 3)
# ═══════════════════════════════════════════════════════════════════════════
# AD 349 y AD 552 del banco de oráculo (rev_1): la ficha dice que lo reclamado
# es una RESOLUCIÓN (la que confirmó un desechamiento: puso fin al juicio) y el
# V I S T O, el resultando y el resolutivo dicen «la resolución dictada…»,
# pero la competencia decía «una sentencia definitiva», la existencia «la
# sentencia reclamada», la legitimación «le causa la sentencia reclamada» y la
# oportunidad «la sentencia reclamada se notificó»: el documento se
# contradice y califica mal el acto. Con un laudo, o con el auto de
# sobreseimiento recurrido en revisión, pasaría igual. Sólo se mira la
# fórmula que nombra AL ACTO («la sentencia reclamada/recurrida/definitiva»,
# «la sentencia dictada el»), no cualquier «sentencia» (la de primera
# instancia, «sentencias definitivas» del 170).
_RX_SENTENCIA_DEL_ACTO = re.compile(
    r"\b(?:la|una|esa|dicha)\s+sentencia\s+(?:definitiva|reclamada|recurrida|impugnada|dictada\s+el)\b", re.I)
_CLASE_CORRECTA = {"resolucion": ("una resolución", "la resolución reclamada"),
                   "laudo": ("un laudo", "el laudo reclamado"),
                   "auto": ("un auto", "el auto recurrido")}


def _clase_del_acto(ficha, tipo) -> str:
    """resolucion | laudo | auto, cuando NO es sentencia y se sabe; "" si no."""
    if not _es_ficha(ficha):
        return ""
    if tipo == "amparo_directo":
        c = _plano(_g(ficha, "acto", "clase"))
        return c if c in ("resolucion", "laudo") else ""
    if tipo == "amparo_revision":
        # En la revisión manda la clase de LO RECURRIDO (`clase_recurrida`):
        # la sentencia del juzgado, la interlocutoria o el auto de
        # sobreseimiento dictado fuera de la audiencia (81-I-d).
        cr = _plano(_g(ficha, "clase_recurrida"))
        if cr:
            return "auto" if cr.startswith("auto") else ""
        c = _plano(_g(ficha, "acto", "clase"))
        return "auto" if c == "auto" and _g(ficha, "acto", "incidente") is not True else ""
    return ""


def _regla_clase(t, secs, ficha, tipo, out):
    clase = _clase_del_acto(ficha, tipo)
    if not clase:
        return
    hallados = []
    for s in secs:
        if s["zona"] in ("caratula", "proemio") or (s["etiquetas"] & {"antecedentes", "sesion", "adhesivo"}):
            continue
        cuerpo = s["texto"]
        if tipo == "amparo_revision" and s["zona"] == "resultando" and "presentacion" in s["etiquetas"]:
            cuerpo = _RX_BLOQUE_RECLAMADO.split(cuerpo)[0]   # los actos del amparo indirecto, tal cual
        if tipo == "amparo_revision" and s["zona"] == "resolutivos":
            # El punto del amparo (C5, sexta ronda): «…contra el acto que
            # reclamó de la Sala…, consistente en la sentencia dictada el …» es
            # el acto del amparo indirecto, no lo recurrido (el auto que sobreseyó).
            cuerpo = _sin_punto_del_amparo(cuerpo)
        m = _RX_SENTENCIA_DEL_ACTO.search(cuerpo)
        if m:
            hallados.append((s, m.group(0)))
    if hallados:
        un, correcta = _CLASE_CORRECTA[clase]
        donde = ", ".join(s["nombre"] for s, _ in hallados[:4]) + (" y otros" if len(hallados) > 4 else "")
        que = "lo recurrido" if tipo == "amparo_revision" else "lo reclamado"
        out.append(("l", f"LA CLASE DEL ACTO NO CUADRA ({donde}): dice «{_un_renglon(hallados[0][1])}» y {que} es "
                         f"{un} (ficha de trámite: {'clase_recurrida' if tipo == 'amparo_revision' else 'acto.clase'}"
                         f" = «{clase}»). Cada mención del acto lo nombra por su clase: «{correcta}».",
                    {"clave": "clase_recurrida" if tipo == "amparo_revision" else "acto.clase"}))


# ═══════════════════════════════════════════════════════════════════════════
# (m) LA FORMA DE NOTIFICACIÓN: la del considerando contra la de la ficha
# ═══════════════════════════════════════════════════════════════════════════
# Quejas 172, 24 y 300 del banco de oráculo (rev_0, ronda 3): la ficha traía
# la notificación electrónica (172 y 24) o por lista (300) y la oportunidad
# decía «de manera personal», con la fracción II del artículo 31 y el día de
# surtimiento de la notificación personal: en la 172 el plazo salió del 7 al 13
# de abril y el del engrose, del 6 al 10. De la forma dependen el día en que
# surte efectos y la fracción; si la ficha la dice, el considerando dice ésa.
# Lista y Boletín no se tienen por distintas: la lista se publica en el Boletín
# (art. 66 LFPCA; igual en los boletines judiciales locales).
_FORMAS_NOTIF = (
    ("personal", re.compile(r"de\s+manera\s+personal|personalmente|en\s+forma\s+personal", re.I)),
    ("lista", re.compile(r"por\s+(?:medio\s+de\s+)?lista|mediante\s+lista|por\s+estrados", re.I)),
    ("oficio", re.compile(r"(?:por|mediante)\s+oficio", re.I)),
    ("electronica", re.compile(r"por\s+v[íi]a\s+electr[óo]nica|electr[óo]nicamente|en\s+forma\s+electr[óo]nica|"
                               r"por\s+medios?\s+electr[óo]nicos?", re.I)),
    ("boletin", re.compile(r"(?:mediante|por|en\s+el)\s+Bolet[íi]n(?:\s+Jurisdiccional|\s+Judicial)?", re.I)),
)
_RX_SE_NOTIFICO = re.compile(r"\bse\s+(?:le\s+)?notific[óo]|\bfue\s+notificad[oa]|\bnotificad[oa]\s+(?:a|al)\b",
                             re.I)
_DICHO_FORMA = {"personal": "de manera personal", "lista": "por lista", "oficio": "por oficio",
                "electronica": "por vía electrónica", "boletin": "mediante Boletín"}


def _forma_de(x) -> str:
    v = _plano(x)
    for pref, clave in (("person", "personal"), ("lista", "lista"), ("estrado", "lista"), ("oficio", "oficio"),
                        ("electr", "electronica"), ("boletin", "boletin")):
        if v.startswith(pref):
            return clave
    return ""


def _forma_dicha(cuerpo: str):
    """La forma que el considerando AFIRMA tras «se notificó…»: (clave, literal)
    o None si no dice ninguna."""
    m = _RX_SE_NOTIFICO.search(cuerpo)
    if not m:
        return None
    tramo = re.split(r"\.\s|;|\bsurti", cuerpo[m.end():m.end() + 260])[0]
    hallado = None
    for clave, rx in _FORMAS_NOTIF:
        mm = rx.search(tramo)
        if mm and (hallado is None or mm.start() < hallado[1]):
            hallado = (clave, mm.start(), mm.group(0))
    return (hallado[0], hallado[2]) if hallado else None


# LA FORMA SIN FUENTE (quinta ronda, 3-oct-2026, F1; Q 335, 229, 261 y 342 y AD
# 274 y 335 del banco). El formulario principal trae `regla_surtimiento =
# Form("personal")` y la ficha copiaba esa omisión como si el secretario la
# hubiera declarado: el considerando afirmaba «de manera personal» —y, con la
# autoridad que recurre, «por oficio» (D2)— sin que ningún papel lo dijera, y
# esta regla callaba porque «personal» era justo lo que no se cotejaba. Ahora la
# ficha marca la fuente: «omision» (el valor por omisión del formulario) u
# «omision_autoridad» (el oficio supuesto de D2). Con esa fuente el
# considerando NO afirma la forma («se notificó… el X y surtió efectos al día
# hábil siguiente», como los engroses); si la afirma, se acusa.
# CADA VALOR TRAE SU PREPOSICIÓN (revisión AD, 3-oct-2026): el aviso decía «la
# ficha la tomó de el valor por omisión»; la contracción la pone el valor.
_FUENTES_SIN_PAPEL = {"omision": "del valor por omisión del formulario («personal»)",
                      "omision_autoridad": "de la suposición de que a la autoridad se le notifica por oficio "
                                           "(art. 31, fr. I)"}


def _fuente_de(ficha, ruta: str) -> str:
    f = ficha.get("fuentes") if isinstance(ficha, dict) else None
    return _plano(f.get(ruta)) if isinstance(f, dict) and isinstance(f.get(ruta), str) else ""


def _regla_forma_notificacion(t, secs, ficha, out):
    if not _es_ficha(ficha):
        return
    fuente = _fuente_de(ficha, "forma_notificacion")
    if fuente in _FUENTES_SIN_PAPEL:
        for s in _de(secs, zona="considerando", etiqueta="oportunidad"):
            if "adhesivo" in s["etiquetas"]:
                continue
            dicha = _forma_dicha(s["texto"])
            if not dicha:
                continue
            out.append(("m", f"LA FORMA DE NOTIFICACIÓN NO CONSTA Y {s['nombre'].upper()} LA AFIRMA: dice que se "
                             f"notificó «{_un_renglon(dicha[1])}», y ningún papel lo dice —la ficha la tomó "
                             f"{_FUENTES_SIN_PAPEL[fuente]} (forma_notificacion)—. Se escribe «se notificó … el "
                             f"[fecha] y surtió efectos …» sin la forma; o compruébala en la constancia de "
                             f"notificación y dila en la ficha, que de ella dependen el día en que surtió efectos y "
                             f"la fracción del artículo 31 (o el precepto de la ley del acto).",
                        {"clave": "forma_notificacion"}))
            return
        return
    f_ficha = _forma_de(ficha.get("forma_notificacion"))
    # «PERSONAL» NO SE COTEJA: es la omisión del formulario principal
    # (`regla_surtimiento`, Form('personal')), y `ficha_tramite.armar` la copia
    # a la ficha como dato del secretario. Contra ella, el «por oficio» de la
    # autoridad que recurre (D2: art. 31, fr. I) o el Boletín del TFJA son el
    # cómputo bien hecho, no una contradicción (integ AD Yucatán, ronda 3). Lo
    # que se acusa es la forma DECLARADA distinta de la omisión —electrónica,
    # lista, oficio, Boletín— que el considerando no respetó.
    if not f_ficha or f_ficha == "personal":
        return
    for s in _de(secs, zona="considerando", etiqueta="oportunidad"):
        if "adhesivo" in s["etiquetas"]:
            continue
        hallado = _forma_dicha(s["texto"])
        if not hallado:
            continue
        if hallado[0] == f_ficha or {hallado[0], f_ficha} == {"lista", "boletin"}:
            return
        out.append(("m", f"LA FORMA DE NOTIFICACIÓN NO CUADRA EN {s['nombre'].upper()}: el considerando dice que se "
                         f"notificó «{_un_renglon(hallado[1])}» y la ficha de trámite dice «{_DICHO_FORMA[f_ficha]}» "
                         f"(forma_notificacion). De la forma dependen el día en que surtió efectos y la fracción del "
                         f"artículo 31 (o el precepto de la ley del acto): o se corrige la ficha, o se vuelve a "
                         f"computar con la forma que dice.",
                    {"clave": "forma_notificacion"}))
        return


# ═══════════════════════════════════════════════════════════════════════════
# (k) EL ADHESIVO QUE CONSTA Y NO SE TRATA
# ═══════════════════════════════════════════════════════════════════════════
# QUÉ CUENTA COMO TRATARLO: cualquier forma de la familia «adhesivo / adhesión /
# adherente / adherirse». EL VERBO CONJUGADO SE ESCRIBE CON «I» («se adhirió»,
# «adhiriéndose», «se adhiriera»), no con «e»: el compositor de la RF escribe
# «se tuvo a X adhiriéndose al recurso» (SPEC §2.2) y la regla vieja, que sólo
# leía «adhesiv|adhesión|adher», acusaba como no tratado un adhesivo que sí lo
# estaba (integ1/RF_xxii, 3-oct-2026).
_RX_TRATO_ADHESIVO = re.compile(r"adhesi(?:v|[óo]n)|adher|adhir", re.I)
# En el AD, la admisión del adhesivo (la mención del art. 181, «promover amparo
# adhesivo», no basta: va en todos los trámites).
_RX_ADMISION_ADHESIVO_AD = re.compile(
    r"adhesiv[oa]\s+(?:promovid|presentad|interpuest)|admiti\w*\s+(?:el\s+)?amparo\s+adhesivo|"
    r"promoviendo\s+amparo\s+adhesivo|tuvo\s+(?:a|por)\s+[^.]{0,80}(?:adhesiv|adhiri)|"
    r"adhiri[ée]ndose|se\s+adhiri[óo]|adhesi[óo]n\s+(?:promovid|presentad|interpuest|formulad)", re.I)


def _adhesivo_sin_auto(ficha, datos) -> bool:
    """E5 (ronda 4, 3-oct-2026; AD 274 del banco): consta quién, pero ni la
    admisión ni la presentación del adhesivo. El compositor ya no escribe su
    considerando ni su resolutivo y pregunta «¿HUBO AMPARO ADHESIVO?» (en el AD
    274 quien «se adhirió» sólo presentó alegatos); la verja no los exige."""
    if _procesal(datos).get("adhesivo_sin_auto"):
        return True
    adh = _g(ficha, "adhesivo") if isinstance(ficha, dict) else None
    if not isinstance(adh, dict):
        return False
    lleno = lambda k: str(adh.get(k) or "").strip() and "*" not in str(adh.get(k))
    return bool(lleno("quien")) and not lleno("admision") and not lleno("presentacion")


def _regla_adhesivo(t, secs, ficha, tipo, out, datos=None):
    adh = _g(ficha, "adhesivo") if isinstance(ficha, dict) else None
    hay = isinstance(adh, dict) and any(str(v or "").strip() for v in adh.values())
    if not hay:
        # Sin ficha: si la carátula o el V I S T O lo nombran como figura del
        # asunto (722/2025: «y su adhesivo»), los resolutivos tienen que decir algo.
        cab = " ".join(s["texto"] for s in secs if s["zona"] in ("caratula", "visto"))
        if not re.search(r"ADHERENTE|\by\s+su\s+adhesiv|\bcon\s+(?:su\s+)?adhesiv|"
                         r"\badhesiv[oa]\s+(?:promovid|interpuest)", cab, re.I):
            return
    zonas = {"resultando": "los resultandos", "considerando": "los considerandos", "resolutivos": "los resolutivos"}
    faltan = [nom for z, nom in zonas.items()
              if any(s["zona"] == z for s in secs)
              and not any(_RX_TRATO_ADHESIVO.search(s["texto"]) for s in secs if s["zona"] == z
                          and "sesion" not in s["etiquetas"])]
    if tipo == "amparo_directo":
        # En el AD el trámite siempre menciona «promover amparo adhesivo» (art.
        # 181): eso no es tratarlo. En los resultandos se busca su admisión.
        if "los resultandos" not in faltan and not any(
                _RX_ADMISION_ADHESIVO_AD.search(s["texto"]) for s in secs if s["zona"] == "resultando"):
            faltan.insert(0, "los resultandos")
    if hay and _adhesivo_sin_auto(ficha, datos):
        # Sin ningún auto, sólo la mención del resultando (con la fecha en
        # hueco): ni considerando ni punto resolutivo.
        faltan = [f for f in faltan if f == "los resultandos"]
    if faltan:
        quien = _g(ficha, "adhesivo", "quien") if hay else ""
        out.append(("k", f"EL {'AMPARO ADHESIVO' if tipo == 'amparo_directo' else 'RECURSO ADHESIVO'}"
                         f"{' de ' + quien if quien else ''} CONSTA Y NO SE TRATA en {', '.join(faltan)}: lleva su "
                         f"resultando, su considerando de legitimación y oportunidad, y su punto resolutivo."))


# ═══════════════════════════════════════════════════════════════════════════
# (n) EL PAR DE PALABRAS REPETIDO (quinta ronda, 3-oct-2026)
# ═══════════════════════════════════════════════════════════════════════════
# RF 4 del banco: «…la nulidad de la resolución negativa ficta recaída a su
# solicitud de solicitud de incorporación al sistema de jubilación…». El
# compositor antepone «recaída a su solicitud de» y la materia ya venía escrita
# «solicitud de…» (así la trae la ficha, y así la teclearía el secretario): la
# palabra salía dos veces y nadie lo decía. Se acusa la COSTURA, no el caso:
# un par de palabras repetido SEGUIDO, en minúsculas («solicitud de solicitud
# de», «de la de la», «del auto del auto»). Los apellidos repetidos («Pérez
# Pérez») van con mayúscula y no entran; «día a día» o «caso por caso»
# tampoco (no son el mismo par seguido).
# UNA SOLA PALABRA REPETIDA NO SE ACUSA: en las 41 sentencias de la calibración
# hay tres erratas así del propio engrose («aunado a que que», Q 356/2026;
# «tribunal colegiado colegiado» y «administradores administradores», RF
# 77/2025), que son de dedo, no de costura; acusarlas sería avisar sobre
# sentencias buenas por algo que el compositor no produce.
_L = r"a-záéíóúñü"
_RX_REPETIDA = re.compile(rf"(?<![\w])(?P<f>[{_L}]+\s+[{_L}]+)\s+(?P=f)(?![\w])")


def _regla_repeticion(secs, out):
    grupos, orden = {}, []
    for s in secs:
        if s["zona"] == "caratula":
            continue
        for m in _RX_REPETIDA.finditer(s["texto"]):
            f = _un_renglon(m.group("f"))
            g = grupos.get(f)
            if g is None:
                g = grupos[f] = {"apartados": [], "extracto": _extracto(s["texto"], m.start(), m.end(), 40)}
                orden.append(f)
            if s["nombre"] not in g["apartados"]:
                g["apartados"].append(s["nombre"])
    for f in orden:
        g = grupos[f]
        out.append(("n", f"PALABRAS REPETIDAS EN {_lista_apartados(g['apartados'])}: «{g['extracto']}». «{f}» va "
                         f"una vez; suele ser la costura de una plantilla con un dato que ya empieza igual (la materia "
                         f"«solicitud de…» tras «recaída a su solicitud de»)."))


# ═══════════════════════════════════════════════════════════════════════════
# (o) (p) (q) LOS ASUNTOS RELACIONADOS (sexta ronda, 3-oct-2026, C6)
# ═══════════════════════════════════════════════════════════════════════════
# David: «siempre y cuando haya asuntos relacionados. No vamos a meter conexidad
# en automático. Hay que habilitar en el taller la opción de con un clic
# precisar si existen asuntos relacionados y con ello se genera el
# considerando». La lista la da el secretario (ficha `relacionados`, fuente
# «secretario», hasta 4) y de ella salen tres piezas: el renglón «RELACIONADO
# CON …» bajo el encabezado, la mención del V I S T O («…, relacionado con el
# amparo en revisión civil 298/2025, interpuesto por…») y el considerando
# («Conexidad.», «Hecho notorio.» o «Asuntos relacionados.»). Las tres tienen
# que decir lo mismo, y ninguna puede existir sin la lista. Lo que se acusa:
#   (o) el propio asunto como relacionado («AD 469/2024 relacionado con el
#       amparo directo 469/2024»): mismo tipo y mismo número. Un AD y una RF con
#       el mismo número son dos expedientes y no se acusan (así los deja también
#       `ficha_tramite.relacionados_de` y `tipos_asunto.considerando_relacionados`);
#   (p) el artículo 64 de la LFPCA fuera de su supuesto: funda la misma sesión
#       SÓLO del amparo directo y la revisión fiscal contra la misma sentencia
#       («Si el particular interpuso amparo directo contra la misma resolución o
#       sentencia impugnada mediante el recurso de revisión, el Tribunal
#       Colegiado de Circuito que conozca del amparo resolverá el citado
#       recurso, lo cual tendrá lugar en la misma sesión en que decida el
#       amparo»; AD 469/2024 del banco);
#   (q) CON LA FICHA NUEVA (la que trae `relacionados`): el V I S T O o el rubro
#       que dicen «relacionado con» sin considerando, o el considerando sin que
#       el V I S T O ni el rubro lo digan; números que no cuadran entre ellos; la
#       conexidad que nadie marcó (la lista vacía) o la lista marcada que el
#       documento perdió.
# CALIBRACIÓN. El tribunal pone el renglón en la carátula y no siempre en el
# V I S T O (AD 469/2024: «(RELACIONADO CON EL RECURSO DE REVISIÓN FISCAL
# 33/2024)» y «CUARTO. Conexidad.», con un V I S T O que no lo dice): por eso el
# anuncio vale en cualquiera de los dos. «Relacionado con el juicio de amparo
# indirecto…» (el de origen) no es un asunto relacionado: sólo cuentan los
# cuatro tipos del índice (directo, revisión, queja, revisión fiscal).
_RX_ASUNTO_REL = re.compile(
    r"(?P<nom>amparo\s+directo|amparo\s+en\s+revisi[óo]n|(?:recurso\s+de\s+)?revisi[óo]n\s+fiscal|"
    r"recurso\s+de\s+queja|recurso\s+de\s+revisi[óo]n|queja)"
    r"(?:\s+(?:en\s+materias?\s+)?(?:civil|penal|administrativ[oa]|laboral|mercantil|familiar|agrari[oa]|"
    r"del?\s+trabajo|com[úu]n))?(?:\s+n[úu]mero)?\s+(?P<num>\d{1,6})\s*/\s*(?P<anio>\d{4})(?![\d/])", re.I)
_RX_RELACIONADO_CON = re.compile(r"\brelacionad[oa]s?\s+con\s+", re.I)
_RX_ENLACE_REL = re.compile(r"\s*,?\s*(?:y\s+)?(?:con\s+)?(?:(?:el|la|los|las)\s+)?(?:juicio\s+de\s+)?", re.I)
# Los rótulos y las fórmulas del tribunal (calibración ampliada, 118 sentencias
# 2024-2026 que mencionan asuntos relacionados): «Conexidad.», «Relación.»
# («Vista la relación que guarda el presente juicio de amparo con el amparo
# directo civil 205/2025…, se resuelven en la propia sesión», AD 194/2025),
# «Conexión en torno al acto reclamado.» («se ordenó relacionar el presente
# asunto con el diverso amparo directo 202/2024», AD 201/2024) y los de C6.
_RX_ROTULO_RELACIONADOS = re.compile(r"^\s*(?:conexidad|conexi[óo]n\b|hecho\s+notorio|asuntos?\s+relacionad|"
                                     r"relaci[óo]n(?:\s+con\b|\s*$))", re.I)
_RX_CONEXION = re.compile(r"\b(?:conexi[óo]n|relaci[óo]n)\s+que\s+guarda|\bse\s+orden[óo]\s+relacionar\b", re.I)
_RX_ESTE_ASUNTO = re.compile(r"\bpresente\s+(?:juicio\s+de\s+)?$", re.I)
_RX_ART_64 = re.compile(r"\b64\b[^.;\d]{0,40}?(?:de\s+la\s+Ley\s+Federal\s+de\s+Procedimiento\s+Contencioso|"
                        r"\bLFPCA\b)", re.I)
_NOMBRE_TIPO = {"amparo_directo": "amparo directo", "amparo_revision": "amparo en revisión",
                "queja": "recurso de queja", "revision_fiscal": "recurso de revisión fiscal"}


def _tipo_rel(nom: str) -> str:
    p = _plano(nom)
    if "fiscal" in p:
        return "revision_fiscal"
    if "directo" in p:
        return "amparo_directo"
    if "queja" in p:
        return "queja"
    return "amparo_revision"


def _mencion_rel(m) -> dict:
    return {"tipo": _tipo_rel(m.group("nom")), "numero": f"{int(m.group('num'))}/{m.group('anio')}",
            "literal": _un_renglon(m.group(0)), "ini": m.start(), "fin": m.end()}


def _lista_tras_relacionado(texto: str, ini: int) -> list:
    """Las menciones seguidas desde `ini` (justo tras «relacionado con»):
    «el amparo directo civil 452/2025, con el X y con el Y». [] si lo que sigue
    no es un asunto del índice («el juicio de amparo indirecto…», «el presente
    asunto»)."""
    fuera, pos = [], ini
    while len(fuera) < 8:
        e = _RX_ENLACE_REL.match(texto, pos)
        m = _RX_ASUNTO_REL.match(texto, e.end() if e else pos)
        if not m:
            break
        fuera.append(_mencion_rel(m))
        pos = m.end()
    return fuera


def _anunciados(texto: str) -> list:
    """Todas las listas «relacionado con …» de un texto, en una."""
    fuera = []
    for m in _RX_RELACIONADO_CON.finditer(texto or ""):
        fuera += _lista_tras_relacionado(texto, m.end())
    return fuera


def _sin_relacionados(texto: str) -> str:
    """El texto sin las listas «relacionado con …» (para leer el número del asunto)."""
    t = str(texto or "")
    tramos = []
    for m in _RX_RELACIONADO_CON.finditer(t):
        lista = _lista_tras_relacionado(t, m.end())
        if lista:
            tramos.append((m.start(), lista[-1]["fin"]))
    for i, j in reversed(tramos):
        t = t[:i] + " " * (j - i) + t[j:]
    return t


def _considerandos_relacionados(secs) -> list:
    """Los considerandos de los asuntos relacionados: por su rótulo («Conexidad.»,
    «Hecho notorio.», «Asuntos relacionados.») o por la fórmula de la conexidad
    («Con vista en la conexión que guarda…»), que sólo se escribe para eso."""
    return [s for s in _de(secs, zona="considerando")
            if _RX_ROTULO_RELACIONADOS.match(s["rotulo"] or "") or _RX_CONEXION.search(s["texto"])]


def _menciones_del_considerando(s) -> list:
    """Los asuntos que nombra el considerando, con el propio («el presente
    juicio de amparo directo…») marcado con "este". Las menciones EN VERSALES
    no cuentan: en el cuerpo son el encabezado de página de la versión pública
    («JUICIO DE AMPARO DIRECTO EN MATERIA CIVIL 248/2024 EN RELACIÓN CON…»,
    AD 248/2024), no prosa; el proyecto compuesto no las escribe."""
    fuera = []
    for m in _RX_ASUNTO_REL.finditer(s["texto"]):
        if m.group("nom").isupper():
            continue
        x = _mencion_rel(m)
        x["este"] = bool(_RX_ESTE_ASUNTO.search(s["texto"][max(0, m.start() - 40):m.start()]))
        fuera.append(x)
    return fuera


def _numero_del_asunto(secs, ficha, datos) -> str:
    n = _num(_g(ficha, "numero")) or _num((datos or {}).get("numero"))
    if n:
        return n
    for z in ("caratula", "visto"):
        s = next(iter(_de(secs, zona=z)), None)
        if s:
            primero = s["texto"].split("\n")[0] if z == "caratula" else s["texto"]
            n = _num(re.split(r"\(?\s*RELACIONAD", _sin_relacionados(primero), maxsplit=1, flags=re.I)[0])
            if n:
                return n
    return ""


def _lista_de_la_ficha(ficha, datos):
    """La lista que marcó el secretario: la del compositor (`procesal`) o la de
    la ficha. None si ninguna de las dos trae la clave (ficha vieja, sin ficha):
    entonces no se coteja contra ella."""
    proc = _procesal(datos)
    for fuente in (proc, ficha if _es_ficha(ficha) else None):
        if isinstance(fuente, dict) and "relacionados" in fuente:
            v = fuente.get("relacionados")
            if isinstance(v, (list, tuple)):
                return [{"tipo": _tipo(str(r.get("tipo") or "")) or str(r.get("tipo") or ""),
                         "numero": _num(r.get("numero"))}
                        for r in v if isinstance(r, dict) and _num(r.get("numero"))]
            if v in (None, "", [], ()):
                return []
    return None


def _prosa_rel(x: dict) -> str:
    return f"{_NOMBRE_TIPO.get(x.get('tipo'), 'asunto')} {x.get('numero')}"


def _lista_en_prosa(xs: list, articulo: bool = True) -> str:
    """«el amparo directo 175/2026 y el recurso de queja 24/2026» (todos los
    nombres son masculinos: «recurso de revisión fiscal», como en `tipos_asunto`)."""
    ps = list(dict.fromkeys(("el " if articulo else "") + _prosa_rel(x) for x in xs))
    return ps[0] if len(ps) == 1 else ", ".join(ps[:-1]) + " y " + ps[-1]


def _regla_relacionado_propio(secs, ficha, datos, tipo, out):
    """(o) EL PROPIO ASUNTO COMO RELACIONADO."""
    propio = _numero_del_asunto(secs, ficha, datos)
    if not propio:
        return
    donde, ejemplo = [], ""

    def _mira(lista, nombre):
        nonlocal ejemplo
        for x in lista:
            if x["numero"] == propio and (not tipo or x["tipo"] == tipo) and not x.get("este"):
                if nombre not in donde:
                    donde.append(nombre)
                ejemplo = ejemplo or x["literal"]
    for s in _de(secs, zona="caratula"):
        _mira(_anunciados(s["texto"]), "la carátula")
    for s in _de(secs, zona="visto"):
        _mira(_anunciados(s["texto"]), "el V I S T O")
    for s in _considerandos_relacionados(secs):
        _mira(_menciones_del_considerando(s), s["nombre"])
    if donde:
        out.append(("o", f"EL ASUNTO APARECE RELACIONADO CONSIGO MISMO ({', '.join(donde)}): «{ejemplo}» es este "
                         f"mismo asunto ({_NOMBRE_TIPO.get(tipo, 'asunto')} {propio}). Un asunto relacionado es OTRO "
                         f"expediente del índice: quítalo de los asuntos relacionados de la ficha de trámite "
                         f"(relacionados) y vuelve a generar.",
                    {"clave": "relacionados"}))


def _oracion(texto: str, i: int, j: int) -> tuple:
    """(inicio, fin) de la oración que contiene [i, j)."""
    ini = 0
    for m in re.finditer(r"[.;]\s+(?=[A-ZÁÉÍÓÚÑ¿«])|\n", texto[:i]):
        ini = m.end()
    m = re.search(r"\.(?:\s+(?=[A-ZÁÉÍÓÚÑ¿«])|\s*$)|\n", texto[j:])
    return ini, (j + m.start() + 1) if m else len(texto)


def _regla_articulo_64(secs, tipo, out):
    """(p) EL 64 DE LA LFPCA SÓLO PARA EL PAR AMPARO DIRECTO ↔ REVISIÓN FISCAL.
    Se mira en los considerandos de los relacionados y en cualquier oración de
    un considerando que hable de la conexión o de la misma sesión."""
    for s in _de(secs, zona="considerando"):
        es_rel = s in _considerandos_relacionados(secs)
        for m in _RX_ART_64.finditer(s["texto"]):
            i, j = _oracion(s["texto"], m.start(), m.end())
            oracion = s["texto"][i:j]
            if not es_rel and not re.search(r"conexi[óo]n|misma\s+sesi[óo]n|relacionad", oracion, re.I):
                continue
            este = tipo
            otros = []
            for x in _RX_ASUNTO_REL.finditer(oracion):
                if x.group("nom").isupper():
                    continue        # el encabezado de página de una versión pública
                mx = _mencion_rel(x)
                if _RX_ESTE_ASUNTO.search(oracion[max(0, x.start() - 40):x.start()]):
                    este = este or mx["tipo"]
                else:
                    otros.append(mx)
            par = {"amparo_directo": "revision_fiscal", "revision_fiscal": "amparo_directo"}
            if este and este not in par:
                malos = otros or [{"tipo": "", "numero": ""}]
            elif este:
                malos = [x for x in otros if x["tipo"] != par[este]]
            else:
                malos = [x for x in otros if x["tipo"] not in par]
            if malos:
                con_nums = [x for x in malos if x["numero"]]
                que = (f"de un {_NOMBRE_TIPO.get(este, 'asunto')}" if este else "de un asunto") + (
                    f" con {_lista_en_prosa(con_nums)}" if con_nums else "")
                out.append(("p", f"EL ARTÍCULO 64 DE LA LFPCA FUERA DE SU SUPUESTO EN {s['nombre'].upper()}: se funda en "
                                 f"él la misma sesión {que}. Ese precepto sólo vale para el amparo directo y la revisión "
                                 f"fiscal que impugnan la misma sentencia («Si el particular interpuso amparo directo "
                                 f"contra la misma resolución o sentencia impugnada mediante el recurso de revisión…»); "
                                 f"para los demás asuntos, la fórmula general: «a fin de evitar el dictado de "
                                 f"resoluciones contradictorias».",
                            {"clave": "relacionados"}))
                return


_RX_ANUNCIO_CARATULA = re.compile(r"\bRELACIONAD[OA]S?\s+CON\b", re.I)
_RX_ANUNCIO_VISTO = re.compile(
    r"\brelacionad[oa]s?\s+con\s+(?:el|la|los|las)\s+(?:juicio\s+de\s+)?(?:amparos?\s+directos?|amparos?\s+en\s+"
    r"revisi[óo]n|recursos?\s+de\s+(?:queja|revisi[óo]n)|revisi[óo]n\s+fiscal|quejas?)\b", re.I)


def _regla_relacionados(secs, ficha, datos, tipo, out):
    """(q) LO QUE DICEN EL RUBRO, EL V I S T O Y EL CONSIDERANDO, ENTRE SÍ Y
    CONTRA LO QUE MARCÓ EL SECRETARIO.

    SÓLO CON LA FICHA NUEVA (la que trae `relacionados`, aunque vacía, o el
    compositor que la expone en `procesal`): ahí las tres piezas salen de UNA
    lista y tienen que decir lo mismo. Sin ella no se coteja: el tribunal pone
    el renglón «RELACIONADO CON …» sin considerando (Q 24/2026 y AR 298/2025,
    que se relacionan entre sí) y el considerando sin la mención en el V I S T O
    (AD 469/2024); en la calibración ampliada (118 sentencias 2024-2026 con
    asuntos relacionados) eso daba 70 avisos sobre sentencias buenas."""
    lista = _lista_de_la_ficha(ficha, datos)
    if lista is None:
        return
    propio = _numero_del_asunto(secs, ficha, datos)

    def _ajenos(xs):
        return [x for x in xs if not x.get("este")
                and not (x["numero"] == propio and (not tipo or x["tipo"] == tipo))]
    cab = "\n".join(s["texto"] for s in _de(secs, zona="caratula"))
    vis = "\n".join(s["texto"] for s in _de(secs, zona="visto"))
    en_caratula = _ajenos(_anunciados(cab))
    en_visto = _ajenos(_anunciados(vis))
    hay_c = bool(en_caratula) or bool(_RX_ANUNCIO_CARATULA.search(cab))
    hay_v = bool(en_visto) or bool(_RX_ANUNCIO_VISTO.search(vis))
    cons = _considerandos_relacionados(secs)
    en_cons = _ajenos([x for s in cons for x in _menciones_del_considerando(s)])
    anuncio = en_caratula or en_visto
    donde = [n for n, hay in (("el V I S T O", hay_v), ("la carátula", hay_c)) if hay]
    donde_anuncio = " y ".join(donde)
    # 1) Las tres piezas, entre sí.
    if donde and not cons:
        dicho = f" {_lista_en_prosa(anuncio).upper()}" if anuncio else " …"
        out.append(("q", f"{donde_anuncio.upper()} {'DICEN' if len(donde) > 1 else 'DICE'} «RELACIONADO CON{dicho}» Y "
                         f"NO HAY CONSIDERANDO DE LOS ASUNTOS RELACIONADOS. Lleva el de conexidad (se resuelven en la "
                         f"misma sesión) o el de hecho notorio (ya se resolvió), justo antes de la dispensa; sale solo "
                         f"de los asuntos relacionados de la ficha de trámite (relacionados).",
                    {"clave": "relacionados"}))
    elif cons and not donde:
        out.append(("q", f"HAY CONSIDERANDO DE ASUNTOS RELACIONADOS ({cons[0]['nombre']}) Y NI EL V I S T O NI LA "
                         f"CARÁTULA LO DICEN: el V I S T O lleva «…, relacionado con …» tras el número del asunto y el "
                         f"rubro, el renglón «RELACIONADO CON …» bajo el encabezado.",
                    {"clave": "relacionados"}))
    elif cons and donde:
        dif = []
        if en_caratula and en_visto and {x["numero"] for x in en_caratula} != {x["numero"] for x in en_visto}:
            dif.append(f"la carátula dice {_lista_en_prosa(en_caratula)} y el V I S T O {_lista_en_prosa(en_visto)}")
        if anuncio and en_cons and {x["numero"] for x in anuncio} != {x["numero"] for x in en_cons}:
            _quien = " y ".join(n for n, l in (("el V I S T O", en_visto), ("la carátula", en_caratula)) if l)
            dif.append(f"{_quien} {'dicen' if ' y ' in _quien else 'dice'} {_lista_en_prosa(anuncio)} y "
                       f"{cons[0]['nombre']} {_lista_en_prosa(en_cons)}")
        if dif:
            out.append(("q", f"LOS ASUNTOS RELACIONADOS NO CUADRAN: {'; '.join(dif)}. Son los mismos en el rubro, el "
                             f"V I S T O y el considerando: los de la ficha de trámite (relacionados).",
                        {"clave": "relacionados"}))
    # 2) Contra lo que marcó el secretario: la conexidad sin lista es la
    #    «conexidad en automático» que David descartó; la lista sin conexidad,
    #    un dato que se perdió en el camino.
    lista = [x for x in lista if not (x["numero"] == propio and (not tipo or x["tipo"] == tipo))]
    en_texto = anuncio or en_cons
    hay_texto = bool(donde or cons)
    if hay_texto and not lista:
        que = (_lista_en_prosa(en_texto) if en_texto
               else (cons[0]["nombre"] if cons else "«relacionado con …» en " + donde_anuncio))
        out.append(("q", f"ASUNTOS RELACIONADOS QUE NADIE MARCÓ: el documento los trae ({que}) y la ficha de trámite "
                         f"no tiene ninguno (relacionados). La conexidad no se pone en automático ni se lee de los "
                         f"papeles: sólo si el secretario marca «Hay asuntos relacionados» en el trámite. Si existen, "
                         f"márcalos; si no, quita el renglón, la mención y el considerando.",
                    {"clave": "relacionados"}))
    elif lista and not hay_texto:
        out.append(("q", f"EL SECRETARIO MARCÓ ASUNTOS RELACIONADOS Y EL DOCUMENTO NO LOS TRAE "
                         f"({_lista_en_prosa(lista)}; relacionados): faltan el renglón «RELACIONADO CON …» del rubro, "
                         f"la mención del V I S T O y el considerando. Lo perdió una pieza del camino, no el "
                         f"formulario: vuelve a generar.",
                    {"clave": "relacionados"}))
    elif lista and en_texto and {x["numero"] for x in lista} != {x["numero"] for x in en_texto}:
        out.append(("q", f"LOS ASUNTOS RELACIONADOS NO SON LOS QUE MARCÓ EL SECRETARIO: la ficha de trámite dice "
                         f"{_lista_en_prosa(lista)} (relacionados) y el documento {_lista_en_prosa(en_texto)}.",
                    {"clave": "relacionados"}))


# ═══════════════════════════════════════════════════════════════════════════
# LA ENTRADA
# ═══════════════════════════════════════════════════════════════════════════
_TODAS = "abcdefghijklmnopq"


def _letras(reglas) -> set:
    """«g», «dg», ["g", "d"] → {"g", "d"}; None o vacío → todas."""
    if not reglas:
        return set(_TODAS)
    if isinstance(reglas, str):
        return {c for c in reglas.lower() if c.isalpha()}
    try:
        return {str(c).strip().lower()[:1] for c in reglas if str(c).strip()}
    except TypeError:
        return set(_TODAS)


def revisar_detalle(texto_procesal: str, ficha: dict = None, datos: dict = None, tipo: str = "", *,
                    avisos_previos=None, reglas=None) -> list:
    """[{"regla": "a".."q", "aviso": str, …}] en el orden de las reglas. Nunca lanza.

    Los de hueco (regla "a") llevan además "dato" (lo que falta, en palabras),
    "clave" (el campo de la ficha que lo llena, la misma que el compositor
    nombra entre paréntesis; "" si no se sabe), "claves" y "apartados". Un
    aviso por campo, con todos los apartados donde ese campo va en hueco.
    `avisos_previos`: los avisos que ya dieron otras piezas (compositor,
    oportunidad); el hueco cuya clave ya está avisada no se repite.
    `reglas`: las letras que se corren («g», «adg»…); por omisión, todas."""
    datos = datos if isinstance(datos, dict) else {}
    if not isinstance(ficha, dict) or not ficha:
        ficha = datos.get("tramite") if isinstance(datos.get("tramite"), dict) else (ficha or {})
    t = str(texto_procesal or "").replace("\r", "")
    tp = _tipo(tipo or (ficha or {}).get("tipo") or datos.get("tipo_asunto") or "")
    out: list = []
    if not t.strip():
        return []
    try:
        secs = secciones(t)
    except Exception:
        return []
    try:
        previos = [str(a) for a in (avisos_previos or []) if a]
    except TypeError:
        previos = []
    quiero = _letras(reglas)
    for letras, regla in (("a", lambda: _regla_huecos(t, secs, ficha, tp, out, previos)),
                          ("b", lambda: _regla_evasivas(t, secs, ficha, tp, out)),
                          ("c", lambda: _regla_fechas(t, secs, ficha, datos, out)),
                          ("c", lambda: _regla_fecha_resolutivo(t, secs, ficha, out, tp)),
                          ("d", lambda: _regla_responsable(t, secs, ficha, datos, tp, out)),
                          ("d", lambda: _regla_organo_descartado(secs, ficha, datos, tp, out)),
                          ("d", lambda: _regla_tribunal(t, secs, datos, out)),
                          ("d", lambda: _regla_ponente(t, secs, ficha, out)),
                          ("d", lambda: _regla_recurrente_caratula(secs, ficha, out)),
                          ("e", lambda: _regla_numeros(t, secs, ficha, datos, tp, out)),
                          ("f", lambda: _regla_concordancias(t, secs, datos, out)),
                          ("f", lambda: _regla_partes_en_prosa(secs, tp, out)),
                          ("f", lambda: _regla_articulo_del_papel(secs, ficha, datos, out)),
                          ("g", lambda: _regla_supletorio(t, secs, ficha, datos, out)),
                          ("h", lambda: _regla_normas(t, secs, tp, out)),
                          ("i", lambda: _regla_verbo(t, secs, ficha, datos, tp, out)),
                          ("j", lambda: _regla_cronologia(t, secs, ficha, datos, tp, out, previos)),
                          ("k", lambda: _regla_adhesivo(t, secs, ficha, tp, out, datos)),
                          ("l", lambda: _regla_clase(t, secs, ficha, tp, out)),
                          ("m", lambda: _regla_forma_notificacion(t, secs, ficha, out)),
                          ("n", lambda: _regla_repeticion(secs, out)),
                          ("o", lambda: _regla_relacionado_propio(secs, ficha, datos, tp, out)),
                          ("p", lambda: _regla_articulo_64(secs, tp, out)),
                          ("q", lambda: _regla_relacionados(secs, ficha, datos, tp, out))):
        if letras not in quiero:
            continue
        # UNA REGLA QUE TROPIEZA NO TUMBA LA VERJA NI EL DOCUMENTO: se calla
        # ella sola. Un aviso de menos es un defecto; un 500 al final de un
        # proyecto ya pagado es perder el trabajo.
        try:
            regla()
        except Exception as ex:  # pragma: no cover - defensivo
            print(f"   ⚠️ VERJA PROCESAL: una regla falló ({type(ex).__name__}: {ex})")
    vistos, final = set(), []
    for x in out:
        r, a = x[0], x[1]
        if r not in quiero or a in vistos:
            continue
        vistos.add(a)
        d = {"regla": r, "aviso": a}
        if len(x) > 2 and isinstance(x[2], dict):
            d.update(x[2])
        final.append(d)
    return final


def revisar(texto_procesal: str, ficha: dict = None, datos: dict = None, tipo: str = "", *,
            avisos_previos=None, reglas=None) -> list:
    """Los avisos de la verja, listos para ir AL PRINCIPIO de la lista de avisos."""
    return [x["aviso"] for x in revisar_detalle(texto_procesal, ficha, datos, tipo,
                                                avisos_previos=avisos_previos, reglas=reglas)]


def revisar_supletorio(texto: str, ficha: dict = None, datos: dict = None, tipo: str = "") -> list:
    """Sólo la regla (g), para el documento ENTERO (SPEC §3.3 g; rev_2, ronda 3).

    El bloque procesal se corta en el rótulo de los Antecedentes o del Estudio,
    y es justo ahí donde el corpus cita el hecho notorio (CFPC 88 / CNPCF 269) y
    la prueba electrónica (210-A / 348): un «artículo 269 del Código Nacional…»
    en el estudio de un proyecto de Querétaro pasaba sin aviso. Quien compone
    le pasa el texto entero (o sólo Antecedentes y Estudio) y antepone lo que
    devuelva; un aviso que ya dio `revisar` sale con el mismo texto, así que
    se descarta con un `if a not in avisos`."""
    return revisar(texto, ficha, datos, tipo, reglas="g")


# ═══════════════════════════════════════════════════════════════════════════
# ARREGLAR: SÓLO TIPOGRAFÍA SEGURA
# ═══════════════════════════════════════════════════════════════════════════
# Lo que se arregla aquí no cambia el sentido de nada y no tiene excepción
# legítima en un proyecto: «de el» en minúscula (el nombre propio va con
# mayúscula y se respeta: «de El Marqués»), «5º» por «5o.», «S.A de C.V.»,
# espacios dobles y «dos ml». Todo lo demás se AVISA, no se toca.
_ARREGLOS = (
    (re.compile(r"\bde el\b"), "del", "«de el» → «del»"),
    (re.compile(r"\ba el\b"), "al", "«a el» → «al»"),
    (re.compile(r"(?<=\d)\s?º\.?"), "o.", "«º» → «o.» (5o., 2o.)"),
    (re.compile(r"(?<=\d)\s?°\.?(?!\s?C\b)"), "o.", "«°» → «o.» (5o., 2o.)"),
    (re.compile(r"(?<=\d)\s?ª\.?"), "a.", "«ª» → «a.»"),
    (re.compile(r"\b(S\.A|S\.A\.P\.I|S\.A\.B|R\.L)(?=\s+[Dd][Ee]\s+C\.\s?V\b)"), r"\1.", "«S.A de C.V.» → «S.A. de C.V.»"),
    (re.compile(r"(?<=\b[Dd][Ee] )(C\.\s?V)(?![.\w])"), r"\1.", "«de C.V» → «de C.V.»"),
    (re.compile(r"\bdos\s+ml\b"), "dos mil", "«dos ml» → «dos mil»"),
    (re.compile(r"(?<=\S)[ \t]{2,}(?=\S)"), " ", "espacios dobles"),
    # EL DOBLE PUNTO (3-oct-2026): una razón social que cierra con punto
    # («…S.A. de C.V.») seguida del punto de la fórmula. Nunca los tres puntos.
    (re.compile(r"(?<!\.)\.[ \t]*\.(?!\.)"), ".", "«..» → «.»"),
)


def arreglar(texto: str) -> tuple:
    """(texto, cambios). Idempotente: arreglar(arreglar(x)[0])[1] == []."""
    t = str(texto or "")
    cambios = []
    for rx, por, desc in _ARREGLOS:
        t, n = rx.subn(por, t)
        if n:
            cambios.append(f"{desc} ({n} {'vez' if n == 1 else 'veces'})")
    return t, cambios
