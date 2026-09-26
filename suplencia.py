"""LA SUPLENCIA DE LA QUEJA, COMO DECISIÓN DEL SECRETARIO.

POR QUÉ EXISTE. Decisión 4 de David sobre la propuesta del estudio de fondo
(26-sep-2026): «Sí» a que la suplencia sea un paso más en la pantalla de
decisión. El motor PROPONE la fracción del artículo 79 y a favor de quién; el
secretario la confirma, la cambia o dice que no hay. Con la suplencia
CONFIRMADA quedan prohibidas las inoperancias de forma a favor de esa parte
—se suple y se estudia—, y la suplencia sólo se expresa en la sentencia cuando
de ella deriva un beneficio.

LO QUE HABÍA, Y POR QUÉ NO BASTABA. La suplencia se decidía por MATERIA dentro
del prompt (`fase6_estudio._SUPLENCIA_ABSOLUTA`), y eso tenía tres huecos,
comprobados el 26-sep-2026:
  · la materia sólo puede ser administrativa, civil, laboral o penal —es lo que
    ofrece el formulario y lo que deduce `fase_precedente.materia_de`—, así que
    las entradas «familiar» y «agraria» no se alcanzaban nunca: la II y la IV
    no se veían en ningún asunto;
  · el texto en el que se buscaban la V administrativa y la VII era
    `_texto_de(material)`, que en la práctica es la palabra de la materia y
    nada más: los dos patrones no disparaban nunca en producción;
  · nadie le preguntaba al secretario, y la suplencia es un umbral que decide
    él (la VII depende de la persona, no de la materia: afirmarla desde un
    patrón sería inventar un hecho del expediente).
Aquí se leen el escrito y los resúmenes de la sesión, se propone con el porqué
a la vista, y lo que llega al estudio es SU decisión.

EL TEXTO DEL ARTÍCULO 79, VERIFICADO Y NO DE MEMORIA. Leído el 26-sep-2026 del
PDF oficial de la Cámara de Diputados (Ley de Amparo, última reforma DOF
16-10-2025; fracciones II, III, IV, V y VI reformadas DOF 13-03-2025) y cotejado
con `normas_ley_de_amparo.json`. Dos precisiones que el texto obliga a hacer:
  · la regla «solo se expresará en las sentencias cuando la suplencia derive
    de un beneficio» está en el PENÚLTIMO párrafo —el que sigue a la fracción
    VII— y vale para las fracciones I, II, III, IV, V y VII, no para la VI;
  · el ÚLTIMO párrafo dice otra cosa: «La suplencia de la queja por violaciones
    procesales o formales sólo podrá operar cuando se advierta que en el acto
    reclamado no existe algún vicio de fondo». Va en el mismo sentido que el
    artículo 189 (Decisión 1 de David: el fondo primero, salvo mayor beneficio).
Y el artículo 171, segundo párrafo, exime de preparar la violación procesal a
casi los mismos sujetos: menores o incapaces, estado civil, orden o estabilidad
de la familia, ejidatarios, comuneros, trabajadores, núcleos de población,
pobreza o marginación y el inculpado en lo penal.

Este módulo no llama al modelo ni a la base: propone (determinista), lee lo que
llega del formulario y arma el bloque del prompt.
"""
from __future__ import annotations

import json
import re
import unicodedata

NINGUNA = "ninguna"

# ═══════════════════════════════════════════════════════════════════════════
# LAS FRACCIONES, CON SU TEXTO VIGENTE
# ═══════════════════════════════════════════════════════════════════════════
# `texto` es una paráfrasis mínima del texto oficial —la pantalla lo enseña y
# el prompt lo recibe—; `test_suplencia.py` comprueba contra la ley que cada
# una conserva las palabras que la definen. `sin_conceptos`: la suplencia opera
# aun sin conceptos de violación o agravios, y sólo se EXPRESA si deriva un
# beneficio (penúltimo párrafo del 79: fracciones I, II, III, IV, V y VII).
# `exime_171`: el sujeto está en el segundo párrafo del artículo 171.
FRACCIONES = (
    {"id": "I", "sin_conceptos": True, "exime_171": False,
     "texto": "En cualquier materia, cuando el acto reclamado se funde en normas "
              "generales declaradas inconstitucionales por la jurisprudencia de la "
              "Suprema Corte de Justicia de la Nación o de los plenos regionales."},
    {"id": "II", "sin_conceptos": True, "exime_171": True,
     "texto": "En favor de las personas menores de edad o incapaces, o en los casos "
              "en que se afecte el orden y desarrollo de la familia."},
    {"id": "III-a", "sin_conceptos": True, "exime_171": True,
     "texto": "En materia penal, en favor de la persona inculpada o sentenciada."},
    {"id": "III-b", "sin_conceptos": True, "exime_171": False,
     "texto": "En materia penal, en favor de la persona ofendida o víctima cuando "
              "tenga el carácter de persona quejosa o adherente."},
    {"id": "IV-a", "sin_conceptos": True, "exime_171": True,
     "texto": "En materia agraria, en los casos de la fracción III del artículo 17: "
              "actos que priven a los núcleos de población ejidal o comunal de la "
              "propiedad, posesión o disfrute de sus derechos agrarios."},
    {"id": "IV-b", "sin_conceptos": True, "exime_171": True,
     "texto": "En materia agraria, en favor de las personas ejidatarias y comuneras "
              "en particular, cuando el acto reclamado afecte sus bienes o derechos "
              "agrarios."},
    {"id": "V", "sin_conceptos": True, "exime_171": True,
     "texto": "En materia laboral, en favor de la persona trabajadora, con "
              "independencia de que la relación esté regulada por el derecho "
              "laboral o por el derecho administrativo."},
    {"id": "VI", "sin_conceptos": False, "exime_171": False,
     "texto": "En otras materias, cuando se advierta una violación evidente de la ley "
              "que haya dejado sin defensa a la persona quejosa o al particular "
              "recurrente; sólo en lo que se refiere a la controversia en el amparo, "
              "sin afectar situaciones procesales resueltas en el procedimiento de "
              "origen."},
    {"id": "VII", "sin_conceptos": True, "exime_171": True,
     "texto": "En cualquier materia, en favor de quienes por sus condiciones de "
              "pobreza o marginación se encuentren en clara desventaja social para "
              "su defensa en el juicio."},
)
_POR_ID = {f["id"]: f for f in FRACCIONES}
_IDS = tuple(f["id"] for f in FRACCIONES) + (NINGUNA,)


def rotulo(fraccion: str) -> str:
    """«IV-b» → «fracción IV, inciso b)»; «ninguna» → «sin suplencia»."""
    f = normalizar_fraccion(fraccion)
    if not f:
        return ""
    if f == NINGUNA:
        return "sin suplencia"
    if "-" in f:
        base, inc = f.split("-", 1)
        return f"fracción {base}, inciso {inc})"
    return f"fracción {f}"


def normalizar_fraccion(x) -> str:
    """Lo que escriba la pantalla o la parte, a la clave del catálogo.

    Acepta «V», «79-V», «fracción IV, inciso b)», «IIIb», «iv-a», «ninguna».
    La III y la IV sin inciso no se inventan: sin inciso no hay clave, salvo
    cuando quien llama sabe cuál es (ver `_pedida`)."""
    s = " ".join(str(x or "").split()).strip()
    if not s:
        return ""
    if s.lower() in (NINGUNA, "sin suplencia", "sin", "no", "none"):
        return NINGUNA
    s = re.sub(r"(?i)^(?:art[íi]culo\s*)?79\s*[-,.]?\s*", "", s)
    s = re.sub(r"(?i)^fracci[óo]n\s*", "", s)
    m = re.match(r"(?i)^(VII|VI|V|IV|III|II|I)(?![IVX])\s*[-,.]?\s*"
                 r"(?:inciso\s*)?([ab])?\)?\s*$", s)
    if not m:
        return ""
    base, inc = m.group(1).upper(), (m.group(2) or "").lower()
    if base in ("III", "IV"):
        return f"{base}-{inc}" if inc else ""
    return base


# ═══════════════════════════════════════════════════════════════════════════
# LA PROPUESTA DEL MOTOR (determinista)
# ═══════════════════════════════════════════════════════════════════════════

def _sin_acentos(x: str) -> str:
    x = unicodedata.normalize("NFD", x or "")
    return "".join(c for c in x if unicodedata.category(c) != "Mn")


def _plano(x: str) -> str:
    return " ".join(str(x or "").split())


# LA II NO LLEGA POR LA MATERIA: el formulario no ofrece «familiar», así que un
# asunto de alimentos o de custodia llega como civil. Se reconoce por lo que
# discute. Se exigen DOS señales distintas —medido contra el banco Kingston: una
# sola palabra suelta («alimentos», «divorcio») aparece en asuntos que no
# afectan a la familia—; con una sola queda como alternativa a revisar.
_RX_FAMILIA = re.compile(
    r"menor(?:es)?\s+de\s+edad|niñ[oa]s?,?\s+niñ[oa]s?\s+y\s+adolescentes|"
    r"inter[ée]s\s+superior\s+de(?:l|\s+la|\s+las|\s+los)?\s+(?:menor|niñ|infancia|adolescen)|"
    r"guarda\s+y\s+custodia|patria\s+potestad|r[ée]gimen\s+de\s+(?:visitas|convivencia)|"
    r"convivencia(?:s)?\s+(?:familiar|paterno|materno|con\s+(?:su|sus|mi|mis)\s+(?:hij|menor))|"
    r"pensi[óo]n\s+alimenticia|acreedor(?:a|es|as)?\s+alimentari|deudor(?:a)?\s+alimentari|"
    r"alimentos\s+(?:a\s+favor|para|en\s+favor)\s+de\s+(?:su|sus|mi|mis|la|las|los|el)\s+"
    r"(?:hij|menor|acreedor|c[óo]nyuge|esposa|esposo)|"
    r"(?:reconocimiento|investigaci[óo]n|desconocimiento)\s+de\s+paternidad|"
    r"(?:juicio|acci[óo]n|demanda)\s+de\s+(?:reconocimiento\s+de\s+)?filiaci[óo]n|"
    r"estado\s+de\s+interdicci[óo]n|violencia\s+familiar|"
    r"(?:juicio|procedimiento|solicitud)\s+de\s+adopci[óo]n|adopci[óo]n\s+de\s+(?:un|una|la|el|los|las)?\s*(?:menor|niñ)|"
    r"p[ée]rdida\s+de\s+la\s+patria", re.I)
# «Adopción» y «filiación» SUELTAS NO CUENTAN: en el ADC 120/2026 —una
# reivindicación— salían de una tesis transcrita que enumera las acciones del
# estado civil («matrimonio, divorcio, filiación, tutela, adopción») y de «la
# adopción de ajustes de procedimiento» de la Convención sobre personas mayores.

# LA IV TAMPOCO: los amparos agrarios los ve un colegiado administrativo y llegan
# como «administrativa» (ADA 263/2025 y 704/2022 del banco Kingston). Se
# reconoce por el órgano y el vocabulario de la Ley Agraria.
_RX_AGRARIO = re.compile(
    r"tribunal\s+(?:unitario\s+)?agrario|tribunal\s+superior\s+agrario|ley\s+agraria|"
    r"ejidatari[oa]s?|comuner[oa]s?\b|bienes\s+comunales|n[úu]cleo\s+(?:de\s+poblaci[óo]n|agrario)|"
    r"comisariado\s+ejidal|parcela\s+ejidal|derechos\s+(?:agrarios|parcelarios|ejidales)|"
    r"registro\s+agrario\s+nacional|procuradur[íi]a\s+agraria|asamblea\s+ejidal|"
    r"sucesi[óo]n\s+(?:de\s+)?derechos\s+(?:agrarios|ejidales)", re.I)
_RX_NUCLEO = re.compile(r"\bejido\b|comunidad\s+(?:agraria|ind[íi]gena)|"
                        r"n[úu]cleo\s+(?:de\s+poblaci[óo]n|agrario)|comisariado|"
                        r"bienes\s+comunales", re.I)

# EN PENAL IMPORTA QUIÉN PROMUEVE: la III tiene dos incisos, uno por lado.
_RX_VICTIMA = re.compile(
    r"(?:en\s+(?:mi|su)\s+(?:car[áa]cter|calidad)\s+de|como)\s+"
    r"(?:v[íi]ctima|ofendid[oa]|asesor[a]?\s+jur[íi]dic[oa]\s+de\s+la\s+v[íi]ctima)|"
    r"(?:quejos[oa]|recurrente|promovente)[^.]{0,60}?\b(?:v[íi]ctima|ofendid[oa])\b", re.I)

# LA VII, DICHA POR LA PROPIA PARTE. Es la única forma honesta de proponerla
# desde el texto: que quien promueve afirme su pobreza, su marginación o su
# desventaja. Cualquier otro indicio —la edad, una discapacidad, que es
# campesino, que no sabe leer— sólo se enseña como alternativa, porque la
# fracción exige la condición de pobreza o marginación y no la presume.
_RX_DESVENTAJA_DICHA = re.compile(
    r"pobreza|marginaci[óo]n|marginad[oa]s?\b|desventaja\s+social|"
    r"(?:condici[óo]n|situaci[óo]n)\s+de\s+vulnerabilidad\s+econ[óo]mica|"
    r"escasos\s+recursos\s+econ[óo]micos|carezco\s+de\s+recursos", re.I)
# LOS INDICIOS QUE SÍ APUNTAN A LA POBREZA O LA MARGINACIÓN. Medido contra los
# 55 escritos del banco Kingston (26-sep-2026): el patrón completo de
# `fase6_estudio._RX_DESVENTAJA` —que incluye «discapacidad», «adulto mayor»,
# «reclusión», «migrante»— marcaba 10 asuntos civiles y administrativos en los
# que nadie alegó pobreza ni marginación (una prescripción, un oral mercantil,
# un contrato). Para enseñar la VII como alternativa se miran sólo los
# indicios de desventaja SOCIAL: pertenencia indígena, analfabetismo, la
# condición campesina que el 642/2024 alegaba con todas sus letras.
_RX_INDICIO_VII = re.compile(
    r"ind[íi]gena|comunidad\s+originaria|analfabet|campesin[oa]s?\b|"
    r"no\s+s[ée]\s+(?:leer|escribir)|no\s+sabe\s+(?:leer|escribir)", re.I)

# QUIÉN NO ES UNA PERSONA TRABAJADORA. La V favorece a la persona, no al ente
# que la emplea: en el ADA 400/2024 del banco quien promovía era un consejo
# estatal que discutía la seguridad social «de sus trabajadores».
_RX_ENTE = re.compile(
    r"\bconsejo\b|organismo|municipio|gobierno|poder\s+(?:ejecutivo|judicial|legislativo)|"
    r"universidad|fideicomiso|sociedad|cooperativa|asociaci[óo]n|comit[ée]|"
    r"\bs\.?\s*p\.?\s*r\.?\b|empresa|comisi[óo]n\s+federal", re.I)

# LA V EN LA VÍA ADMINISTRATIVA, CON SEÑAL FUERTE. El patrón de
# `fase6_estudio._RX_TRABAJADOR_EN_ADMINISTRATIVA` es la puerta; pero «pensión»,
# «seguridad social» o «burocrático» a secas no bastan: en el banco salían del
# «pensión (depósito)» de un arrastre de grúa (ADA 541/2025), de «certeza ni
# seguridad social» en una multa de tránsito (ADA 702/2022) y de «un trámite
# burocrático» (ADA 289/2026). Hace falta una de éstas.
_RX_V_FUERTE = re.compile(
    r"\bISSSTE\b|\bIMSS\b|cuota\s+pensionaria|haber\s+de\s+retiro|\bjubilaci[óo]n|"
    r"\bjubilad[oa]s?\b|\bpensionad[oa]s?\b|cesant[íi]a\s+en\s+edad|"
    r"\bpensi[óo]n\s+(?:por\s+|de\s+)?(?:jubilaci|vejez|cesant|invalidez|viudez|orfandad|retiro|ascendencia)|"
    r"trabajador(?:a|es)?\s+al\s+servicio\s+del\s+estado|"
    r"(?:cese|baja|remoci[óo]n|separaci[óo]n|destituci[óo]n)\s+(?:del?\s+(?:su|mi|el)\s+)?"
    r"(?:cargo|puesto|empleo)\b", re.I)
# «Elemento de seguridad pública» NO es señal: en la multa de tránsito del ADA
# 702/2022 era el agente que levantó la boleta, no quien promovía.

# LO QUE LA PARTE PIDE. El 642/2024 lo pedía con todas las letras —«artículo 79,
# fracción IV, inciso b)»— y ni el engrose ni el proyecto generado lo
# contestaron. Se le enseña al secretario, sea o no lo que el motor propone.
_RX_PIDE = re.compile(
    r"79[^.]{0,60}?fracci[óo]n\s+(VII|VI|V|IV|III|II|I)(?![IVX])[\s.,]*(?:inciso\s+([ab])\))?|"
    r"fracci[óo]n\s+(VII|VI|V|IV|III|II|I)(?![IVX])[\s.,]*(?:inciso\s+([ab])\)[\s,]*)?"
    r"(?:del\s+)?(?:art[íi]culo|numeral|precepto)\s+79", re.I)


def _pedida(texto: str) -> str:
    """La fracción que el escrito pide, si pide una.

    Tiene que hablar de SUPLENCIA —no de lo «supletorio»— y del 79 DE LA LEY DE
    AMPARO. Medido en el banco: en el ADA 61/2026 el «79, 80 y 81 de la Ley de
    Responsabilidades» junto a un «aplicado de manera supletoria» se leía como
    si la parte pidiera la fracción V."""
    t = _plano(texto)
    for m in _RX_PIDE.finditer(t):
        ventana = t[max(0, m.start() - 400):m.end() + 400].lower()
        if not re.search(r"suplencia|supl\w*\s+(?:de\s+)?(?:la\s+)?(?:deficiencia|queja)", ventana):
            continue
        if "amparo" not in t[max(0, m.start() - 200):m.end() + 200].lower():
            continue
        base = (m.group(1) or m.group(3) or "").upper()
        inc = (m.group(2) or m.group(4) or "").lower()
        if base in ("III", "IV") and not inc:
            # Sin inciso, el que cubre a la persona en particular: la III a) al
            # inculpado —el caso de siempre— y la IV b) al ejidatario.
            inc = "a" if base == "III" else "b"
        f = normalizar_fraccion(f"{base}{('-' + inc) if base in ('III', 'IV') else ''}")
        if f:
            return f
    return ""


def _hallazgos(rx, texto: str, tope: int = 3) -> list:
    vistos = []
    for m in rx.finditer(texto or ""):
        w = _plano(m.group(0)).lower()
        if w not in vistos:
            vistos.append(w)
        if len(vistos) >= tope:
            break
    return vistos


def _parte_favorecida(tipo: str, quejoso: str, recurrente: str) -> tuple:
    """(rótulo de la parte, nombre, recurre_la_autoridad)."""
    import tipos_asunto as _ta
    voc = _ta.vocabulario_de(tipo or "amparo_directo")
    if voc.get("promovente") == "quejoso":
        return "la parte quejosa", quejoso, False
    # EN UN RECURSO QUE INTERPONE LA AUTORIDAD, la suplencia no la favorece a
    # ella: favorece al particular que obtuvo o pidió el amparo.
    if recurrente and _parece_autoridad(recurrente):
        return "la parte quejosa", quejoso, True
    return "la parte recurrente", recurrente or quejoso, False


def _parece_autoridad(nombre: str) -> bool:
    try:
        import redactor_adelanto as _ra
        return _ra._parece_autoridad(nombre)
    except Exception:
        return bool(re.search(r"titular|director|secretar[íi]a|tribunal|juzgado|"
                              r"instituto|ayuntamiento|autoridad|procurad",
                              nombre or "", re.I))


def _mismo(a: str, b: str) -> bool:
    a, b = _sin_acentos(a).lower(), _sin_acentos(b).lower()
    a, b = " ".join(a.replace(",", " ").split()), " ".join(b.replace(",", " ").split())
    return bool(a and b) and (a == b or a in b or b in a)


def proponer(materia: str = "", tipo_asunto: str = "", escrito: str = "",
             resumenes: str = "", quejoso: str = "", recurrente: str = "",
             actor_origen: str = "", demandado_origen: str = "") -> dict:
    """La fracción que el motor propone, a favor de quién y por qué.

    `escrito` es la demanda o el recurso; `resumenes`, lo que las fases ya
    escribieron del asunto. Devuelve también las alternativas que conviene que
    el secretario mire —la fracción que la parte pide, los indicios de la VII—
    y el catálogo para que la pantalla no lo tenga que copiar."""
    import fase6_estudio as _f6
    import promovente as _pv
    import tipos_asunto as _ta
    tipo = _ta.normalizar(tipo_asunto or "") or "amparo_directo"
    m = (materia or "").strip().lower()
    parte, nombre, recurre_autoridad = _parte_favorecida(tipo, quejoso or "", recurrente or "")
    a_favor = f"{parte} ({_plano(nombre)[:140]})" if (nombre or "").strip() else parte
    t_parte = _plano(escrito)
    t_asunto = _plano(" ".join(x for x in (escrito, resumenes) if x))
    pedida = _pedida(escrito) or _pedida(resumenes)

    def _salida(fraccion, porque, alternativas=(), quien=a_favor):
        alts, vistas = [], {fraccion}
        for f_alt, p_alt in alternativas:
            if f_alt and f_alt not in vistas:
                vistas.add(f_alt)
                alts.append({"fraccion": f_alt, "rotulo": rotulo(f_alt), "porque": p_alt})
        return {"fraccion": fraccion, "rotulo": rotulo(fraccion),
                "texto": (_POR_ID.get(fraccion) or {}).get("texto", ""),
                "a_favor_de": quien if fraccion != NINGUNA else "",
                # LA PARTE QUE PROMUEVE, SIEMPRE: si el motor no propone nada y
                # el secretario elige una fracción, la pantalla sabe a favor de
                # quién sin volver a preguntar.
                "parte": quien,
                "porque": porque, "alternativas": alts, "pedida": pedida,
                "fracciones": catalogo()}

    # ── DONDE NO HAY JUICIO DE AMPARO NO HAY ARTÍCULO 79 ─────────────────
    if tipo in _f6._SIN_SUPLENCIA:
        return _salida(NINGUNA,
                       "La revisión fiscal no es un juicio de amparo: el artículo 79 "
                       "obliga a «la autoridad que conozca del juicio de amparo», y "
                       "quien recurre es la autoridad.")

    fuertes, alternas = [], []
    # ¿QUIEN PROMUEVE ES UNA PERSONA, O UN ENTE? La V y la VII son de personas.
    ente = (_pv.es_moral(nombre or "") or _parece_autoridad(nombre or "")
            or bool(_RX_ENTE.search(nombre or "")))
    # LA MATERIA QUE YA TRAE SU FRACCIÓN, del mismo catálogo que usa el aviso
    # del estudio: laboral → V, penal → III (y familiar → II, agraria → IV, si
    # algún día se declaran).
    par = _f6._SUPLENCIA_ABSOLUTA.get(m)
    por_materia = par[1].split("fracción")[-1].strip() if par else ""

    # ── V · LA PERSONA TRABAJADORA ───────────────────────────────────────
    if por_materia == "V":
        if (demandado_origen and _mismo(nombre, demandado_origen)) or ente:
            alternas.append(("V", "Materia laboral, pero quien promueve parece la parte "
                                  "empleadora: la fracción V opera sólo en favor de la "
                                  "persona trabajadora."))
        else:
            quien = ("figura como actora en el juicio de origen"
                     if actor_origen and _mismo(nombre, actor_origen)
                     else "es una persona física")
            fuertes.append(("V", f"Materia laboral y quien promueve {quien}: la fracción V "
                                 f"opera en favor de la persona trabajadora aun sin "
                                 f"conceptos de violación. Compruébalo en la carátula."))

    # ── III · PENAL, POR LOS DOS LADOS ───────────────────────────────────
    if por_materia == "III":
        if _RX_VICTIMA.search(t_parte):
            fuertes.append(("III-b", "Materia penal y quien promueve se ostenta como víctima "
                                     "u ofendido: fracción III, inciso b)."))
            alternas.append(("III-a", "Si quien promueve es la persona inculpada o "
                                      "sentenciada, es el inciso a)."))
        else:
            fuertes.append(("III-a", "Materia penal: la fracción III, inciso a), opera en "
                                     "favor de la persona inculpada o sentenciada aun sin "
                                     "conceptos de violación."))
            alternas.append(("III-b", "Si quien promueve es la víctima u ofendido, es el "
                                      "inciso b)."))

    # ── IV · AGRARIA ─────────────────────────────────────────────────────
    # Llega como «administrativa». En civil el vocabulario agrario no basta —la
    # IV rige «en materia agraria»—: ahí sólo se enseña si la parte la pide.
    agr = _hallazgos(_RX_AGRARIO, t_asunto)
    f_agr = "IV-a" if _RX_NUCLEO.search(nombre or "") else "IV-b"
    if por_materia == "IV" or (agr and m in ("administrativa", "") and len(agr) >= 2):
        fuertes.append((f_agr, "Asunto agrario" + (" (" + ", ".join(f"«{h}»" for h in agr) + ")"
                                                   if agr else "")
                               + ": la fracción IV opera aun sin conceptos de violación."))
    elif agr and m in ("administrativa", ""):
        alternas.append((f_agr, "Aparece vocabulario agrario ("
                                + ", ".join(f"«{h}»" for h in agr)
                                + "); la fracción IV rige sólo en materia agraria."))

    # ── V · LA PERSONA TRABAJADORA EN LA VÍA ADMINISTRATIVA ─────────────
    if m == "administrativa" and not ente:
        fuerte = _hallazgos(_RX_V_FUERTE, t_asunto)
        if fuerte:
            fuertes.append(("V", "La relación parece de trabajo regida por el derecho "
                                 "administrativo (" + ", ".join(f"«{h}»" for h in fuerte)
                                 + "): la fracción V opera «con independencia» de que la "
                                   "regule el derecho laboral o el administrativo."))

    # ── II · MENORES, INCAPACES Y FAMILIA ────────────────────────────────
    fam = _hallazgos(_RX_FAMILIA, t_asunto, tope=4)
    if por_materia == "II" or (len(fam) >= 2 and m != "penal"):
        fuertes.append(("II", "El asunto involucra a personas menores de edad o el orden "
                              "de la familia" + (" (" + ", ".join(f"«{h}»" for h in fam[:3]) + ")"
                                                 if fam else "")
                              + ": la fracción II opera aun sin conceptos de violación."))
    elif fam:
        alternas.append(("II", "Hay una señal de menores o de familia ("
                               + ", ".join(f"«{h}»" for h in fam)
                               + "): mira si el acto afecta sus derechos."))

    # ── VII · POBREZA O MARGINACIÓN, SÓLO SI CONSTA ──────────────────────
    dicha = _hallazgos(_RX_DESVENTAJA_DICHA, t_parte)
    otros = [x for x in _hallazgos(_RX_INDICIO_VII, t_parte) if x not in dicha]
    if dicha and not ente:
        (fuertes if not fuertes else alternas).append(
            ("VII", "Quien promueve afirma su condición (" + ", ".join(f"«{h}»" for h in dicha)
                    + "). La fracción VII exige que la pobreza o marginación la deje en "
                      "clara desventaja para defenderse: confírmala sólo si el expediente "
                      "lo acredita."))
    elif otros and not ente:
        alternas.append(("VII", "Hay indicios de desventaja social ("
                                + ", ".join(f"«{h}»" for h in otros)
                                + "), pero la fracción VII exige pobreza o marginación que "
                                  "deje a la persona en clara desventaja: no se presume."))

    # ── LO QUE LA PARTE PIDE, SIEMPRE A LA VISTA ─────────────────────────
    if pedida and pedida != NINGUNA:
        alternas.append((pedida, f"La parte la pide expresamente ({rotulo(pedida)}) y hay "
                                 f"que contestar esa petición."))

    if fuertes:
        f0, p0 = fuertes[0]
        if pedida == f0:
            p0 += " Además, la parte la pide expresamente."
        if recurre_autoridad:
            p0 += " Recurre la autoridad: la suplencia favorece a quien obtuvo o pidió el amparo."
        return _salida(f0, p0, fuertes[1:] + alternas)
    base = {"civil": "Materia civil o mercantil",
            "administrativa": "Materia administrativa",
            "laboral": "Materia laboral",
            "penal": "Materia penal"}.get(m, "Sin materia declarada")
    return _salida(NINGUNA,
                   f"{base} y no se advierte ninguno de los supuestos del artículo 79: "
                   f"se estudia en estricto derecho. La fracción VI (violación evidente "
                   f"que dejó sin defensa) y la I (norma declarada inconstitucional por "
                   f"jurisprudencia) no se pueden leer desde el texto: si las adviertes, "
                   f"elígelas.", alternas)


def proponer_de(r) -> dict:
    """La propuesta para una sesión del taller (un `redactor_adelanto.Resultado`)."""
    e = getattr(r, "encargo", None)
    f = getattr(r, "fases", None)
    p = getattr(r, "partes", None)
    fuentes = list(getattr(f, "fuentes", None) or []) + ["", ""]
    escrito = str(fuentes[1] or "")
    # SIN EL RESUMEN DEL ACTO. Lo que la responsable razonó arrastra citas y
    # protocolos generales («niñas, niños y adolescentes», «menor de edad») que
    # no dicen de quién es el asunto: con la sentencia entera delante, el banco
    # Kingston daba la II en un oral mercantil y en una sociedad anónima. Los
    # antecedentes y el relato ya dicen ante quién se litigó y qué se discute.
    resumenes = " ".join(str(getattr(f, k, "") or "") for k in
                         ("antecedentes", "resumen_conceptos", "relato"))
    # SIN EL ESCRITO LITERAL —sesiones viejas, o un worker que no lo tiene—, lo
    # que la parte dijo de sí misma se lee del resumen de sus conceptos.
    if not escrito.strip():
        escrito = str(getattr(f, "resumen_conceptos", "") or "")
    try:
        import redactor_adelanto as _ra
        materia = _ra.fp_materia(e) if e is not None else ""
    except Exception:
        materia = str(getattr(e, "materia", "") or "")
    return proponer(materia=materia,
                    tipo_asunto=str(getattr(e, "tipo_asunto", "") or ""),
                    escrito=escrito, resumenes=resumenes,
                    quejoso=str(getattr(e, "quejoso", "") or getattr(p, "quejoso", "") or ""),
                    recurrente=str(getattr(e, "recurrente", "") or ""),
                    actor_origen=str(getattr(p, "actor_origen", "") or ""),
                    demandado_origen=str(getattr(p, "demandado_origen", "") or ""))


def catalogo() -> list:
    """Las fracciones para el selector de la pantalla, con su texto vigente."""
    return ([{"id": f["id"], "rotulo": rotulo(f["id"]), "texto": f["texto"]}
             for f in FRACCIONES]
            + [{"id": NINGUNA, "rotulo": rotulo(NINGUNA),
                "texto": "No opera ningún supuesto del artículo 79: estricto derecho."}])


# ═══════════════════════════════════════════════════════════════════════════
# LO QUE LLEGA DEL FORMULARIO
# ═══════════════════════════════════════════════════════════════════════════

def leer(crudo) -> dict:
    """{fraccion, a_favor_de, confirmada} del campo «suplencia», o {} si no hay.

    Un JSON roto o una fracción que no existe no tumban la generación: se
    tratan como si no hubiera llegado nada, que es el comportamiento de antes."""
    d = crudo
    if not isinstance(d, dict):
        try:
            d = json.loads(str(crudo or "").strip() or "{}")
        except Exception:
            return {}
    if not isinstance(d, dict):
        return {}
    fr = normalizar_fraccion(d.get("fraccion"))
    if not fr:
        return {}
    conf = d.get("confirmada")
    conf = conf is True or str(conf).strip().lower() in ("1", "true", "si", "sí")
    quien = _plano(str(d.get("a_favor_de") or ""))[:200]
    return {"fraccion": fr, "a_favor_de": quien if fr != NINGUNA else "",
            "confirmada": conf}


def confirmada(s) -> bool:
    """¿El secretario confirmó un supuesto de suplencia (no «sin suplencia»)?"""
    return (isinstance(s, dict) and bool(s.get("confirmada"))
            and s.get("fraccion") in _POR_ID)


# ═══════════════════════════════════════════════════════════════════════════
# EL BLOQUE DEL PROMPT DEL ESTUDIO
# ═══════════════════════════════════════════════════════════════════════════
# Sólo con la suplencia CONFIRMADA. Sin confirmar —o con «sin suplencia»— no se
# añade nada y el estudio se comporta como antes: lo que decide el umbral es el
# secretario, no un patrón. Van DESCRIPCIONES de lo que hay que hacer, no frases
# para copiar: un ejemplo escrito en el prompt acaba firmado en el documento
# (medido tres veces en este proyecto).

def bloque(s, tipo_asunto: str = "") -> str:
    if not confirmada(s):
        return ""
    import tipos_asunto as _ta
    tipo = _ta.normalizar(tipo_asunto or "") or "amparo_directo"
    voc = _ta.vocabulario_de(tipo)
    q = voc.get("combate") or "conceptos de violación"
    fr = _POR_ID[s["fraccion"]]
    quien = s.get("a_favor_de") or voc.get("parte") or "la parte que promueve"
    L = ["", "═" * 71,
         "SUPLENCIA DE LA QUEJA — LA CONFIRMÓ EL SECRETARIO",
         "═" * 71,
         f"En este asunto opera la suplencia de la deficiencia de los {q}",
         f"del artículo 79, {rotulo(fr['id'])}, de la Ley de Amparo:",
         f"  {fr['texto']}",
         f"A favor de: {quien}.",
         ""]
    forma = ["   un planteamiento suyo por cómo está formulado: por genérico o dogmático,",
             "   por no combatir la razón toral o todas las consideraciones, por reiterar",
             "   lo dicho en la instancia o por atacar sólo una consideración accesoria.",
             "   Donde el planteamiento sea deficiente, reconstruye lo que la parte",
             "   pretende y ESTÚDIALO EN EL FONDO con lo que consta en autos; la suplencia",
             "   no autoriza a suponer hechos que no obran en el expediente."]
    if fr["sin_conceptos"]:
        L += ["1. NINGUNA INOPERANCIA DE FORMA A FAVOR DE ESA PARTE. No declares inoperante"]
        L += forma
        L += [f"   Opera aun ante la ausencia de {q}: si adviertes en el acto un vicio",
              "   que nadie alegó y que beneficiaría a esa parte, no lo calles. Si cabe",
              "   en el sentido dictado, estúdialo; si lo cambiaría, dilo en ADVERTENCIAS",
              "   para que decida el secretario."]
    else:
        # LA VI NO ES ABSOLUTA: opera sobre la violación evidente que dejó sin
        # defensa a la parte, no sobre todo lo que alegó. La prohibición de la
        # inoperancia de forma se ciñe a eso.
        L += ["1. LA FRACCIÓN VI NO ES ABSOLUTA: opera sobre la violación evidente de la",
              "   ley que dejó sin defensa a esa parte, sólo en lo que se refiere a la",
              "   controversia en el amparo y sin afectar situaciones procesales ya",
              "   resueltas en el procedimiento de origen. En lo que toque a esa",
              "   violación, no declares inoperante" + forma[0][2:]]
        L += forma[1:]
        L += ["   En lo demás, los planteamientos se estudian en sus términos."]
    L += ["",
          "2. EL SENTIDO SIGUE SIENDO EL DEL SECRETARIO. Si a un problema de esa parte",
          "   se le dictó una inoperancia, sólo puede descansar en una causa ajena a",
          "   la formulación del argumento —cosa juzgada, argumento novedoso, falsa",
          "   premisa—. Si la única causa posible es la deficiencia del planteamiento,",
          "   estúdialo en el fondo y dilo en ADVERTENCIAS: el sentido dictado choca",
          "   con la suplencia que él mismo confirmó. Y si, suplido, el planteamiento",
          "   prosperaría, dilo también ahí."]
    n = 3
    if fr["exime_171"] and tipo == "amparo_directo":
        L += ["",
              f"{n}. NO SE LE EXIGE HABER PREPARADO LA VIOLACIÓN PROCESAL. El artículo 171,",
              "   segundo párrafo, de la Ley de Amparo exime de ese requisito a este",
              "   sujeto: no declares inoperante una violación procesal suya por no",
              "   haberla impugnado durante el juicio. Sigue haciendo falta que",
              "   trascienda al resultado del fallo."]
        n += 1
    if fr["sin_conceptos"]:
        L += ["",
              f"{n}. CUÁNDO SE ESCRIBE LA PALABRA SUPLENCIA. El artículo 79, en el párrafo",
              "   que sigue a la fracción VII, manda que la suplencia sólo se exprese en",
              "   la sentencia cuando de ella derive un beneficio para esa parte. Si",
              "   suplir no cambia el resultado, no la anuncies ni la invoques: contesta",
              "   lo que la parte pretende y sigue. Si deriva un beneficio, dilo en el",
              "   punto donde se suple y explica qué se suplió. Esta regla gobierna",
              "   sobre cualquier otra instrucción de este encargo que mande anunciar",
              "   la suplencia."]
    else:
        L += ["",
              f"{n}. CUÁNDO SE ESCRIBE. Si adviertes la violación evidente, exprésala y",
              "   razónala en el punto donde suples. Si no la adviertes, no menciones la",
              "   suplencia en la sentencia y dilo en ADVERTENCIAS: el secretario la",
              "   confirmó creyendo verla."]
    n += 1
    L += ["",
          f"{n}. VIOLACIONES PROCESALES O FORMALES. Según el último párrafo del",
          "   artículo 79, la suplencia por violaciones procesales o formales sólo",
          "   opera cuando no se advierte en el acto reclamado algún vicio de fondo."]
    return "\n".join(L)
