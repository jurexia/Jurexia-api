"""LA LEY LOCAL SÓLO ENTRA SI ESTÁ EN LA LITIS.

═══════════════════════════════════════════════════════════════════════════════
POR QUÉ EXISTE
═══════════════════════════════════════════════════════════════════════════════
Revisión fiscal 2/2026, 16-sep-2026. La sentencia recurrida es de la Sala
Regional en Querétaro del Tribunal FEDERAL de Justicia Administrativa y aplicó
la Ley FEDERAL de Procedimiento Contencioso Administrativo. El proyecto dijo:

    «En el ámbito local, el artículo 57 de la Ley de Procedimiento Contencioso
     Administrativo del Estado de Querétaro establece: …», y después el 55 «de
     la propia ley».

Una ley estatal en un asunto federal. David: «es un error que el redactor no
debe permitirse en ningún proyecto».

Cómo entró —y por qué no basta con cerrar esa puerta—: la Sala sólo cita leyes
federales, así que el lector de «la ley con que se dictó el acto» cayó a la más
citada, la LFPCA; el marco la buscó DENTRO de `leyes_queretaro`, donde una ley
federal no puede estar; y el emparejador por palabras aceptó la ley estatal que
comparte «procedimiento contencioso administrativo». Los artículos 57 y 55 de
Querétaro son copia literal del 51 y el 50 de la LFPCA: para una búsqueda por
concepto son aciertos perfectos, y sólo el fuero los distingue. Hay al menos
cuatro puertas por las que una ley entra al proyecto —la búsqueda por concepto,
el completado de preceptos citados, el marco jurídico y lo que el modelo
escribe de memoria—, y cerrar una no cierra las otras.

═══════════════════════════════════════════════════════════════════════════════
LA REGLA, Y POR QUÉ NO ES UN VETO POR FUERO
═══════════════════════════════════════════════════════════════════════════════
«Nunca ley local en un asunto federal» rompería asuntos reales: cuando recurre
una autoridad fiscal ESTATAL que actúa por el convenio de colaboración, su
reglamento interior es ley local y es justo lo que se discute. David lo fijó:
«en amparo directo extraemos la ley sustantiva del acto reclamado; en revisión,
si se entra al fondo, el debate se sujeta a lo resuelto en la sentencia
recurrida y lo alegado en agravios».

De ahí la regla, una sola y en un solo sitio:

    UNA LEY LOCAL ES ADMISIBLE SI Y SÓLO SI ESTÁ NOMBRADA EN LA LITIS
    —el acto o la sentencia recurrida, el escrito de conceptos o agravios, o
    los antecedentes y resúmenes que se hicieron de ellos—.

Las leyes federales y generales siguen las reglas de siempre: aquí no se tocan.

Medido en el 2/2026: «procedimiento contencioso administrativo» aparece 27
veces en el acto y los agravios —es la ley federal— y «contencioso
administrativo DEL ESTADO», cero. La regla distingue sin ambigüedad.

═══════════════════════════════════════════════════════════════════════════════
QUÉ HACE CON UNA CITA INADMISIBLE
═══════════════════════════════════════════════════════════════════════════════
No basta con avisar: el proyecto no puede salir con ella. Pero borrarla a
ciegas se lleva razonamiento correcto —en el 2/2026, detrás del artículo 57
venía «la competencia material de la emisora constituye un presupuesto de
validez del acto», que es cierto y es lo que sostiene el fallo—.

  1. ESPEJO. Si el texto transcrito es, palabra por palabra, el de un precepto
     ADMISIBLE del material —el 57 de Querétaro es el 51 de la LFPCA—, la cita
     se reescribe a ése. Sólo con identidad textual demostrada: nunca por
     parecido de tema.
  2. RETIRO. Si no hay espejo, se retira la frase que invoca la ley, su
     transcripción y la frase siguiente que sólo remite a ella.
  3. CONSTANCIA. Todo lo que se reescribe o se retira queda dicho en un aviso,
     con el texto original: el secretario tiene que poder ver qué se tocó.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

# ═══════════════════════════════════════════════════════════════════════════
# LEER NOMBRES DE LEY
# ═══════════════════════════════════════════════════════════════════════════

_CABEZAS = r"(?:Constituci[óo]n|C[óo]digo|Ley|Reglamento|Estatuto|Decreto|Acuerdo)"

# Un nombre de ordenamiento, desde su cabecera hasta donde deja de nombrar.
_RX_NOMBRE_LEY = re.compile(
    _CABEZAS + r"\s+[^.;:,()«»“”\"\n]{3,120}", re.I)

# Una cita de artículo(s) seguida del ordenamiento que la ancla.
_RX_CITA_ART = re.compile(
    r"\bart[íi]culos?\s+(?P<nums>\d{1,4}(?:[^.;:«“\"\n]{0,90}?\d{1,4})?"
    r"(?:\s*(?:bis|ter|[A-K])\b)?)"
    r"(?P<frac>(?:,?\s*(?:fracci[óo]n(?:es)?|p[áa]rrafos?|incisos?|apartados?)"
    r"[^.;:«“\"\n]{0,60}?)?)"
    r",?\s+(?:de\s+la|de\s+los|de\s+las|del|de|para\s+el)\s+"
    r"(?P<ley>" + _CABEZAS + r"[^.;:,()«»“”\"\n]{3,120})", re.I)

# «de la propia ley», «de dicho ordenamiento»: remite a la última ley nombrada.
_RX_CITA_ANAFORA = re.compile(
    r"\bart[íi]culos?\s+(?P<nums>\d{1,4}(?:[^.;:«“\"\n]{0,40}?\d{1,4})?)"
    r"[^.;:«“\"\n]{0,60}?\s+(?:de\s+la|del|de)\s+"
    r"(?P<ana>(?:propi[ao]|mism[ao]|citad[ao]|referid[ao]|dich[ao]|aludid[ao]|"
    r"invocad[ao]|mencionad[ao])\s+(?:ley|c[óo]digo|reglamento|ordenamiento|"
    r"cuerpo\s+(?:legal|normativo))"
    r"|(?:dich[ao]|es[ae]|la\s+citada|el\s+citado)\s+(?:ley|c[óo]digo|reglamento|"
    r"ordenamiento))", re.I)

# El nombre acaba donde empieza lo que ya no es nombre.
_RX_FIN_DE_NOMBRE = re.compile(
    r"\s+(?:establece|dispone|prev[ée]|se[ñn]ala|regula|permite|fija|faculta|"
    r"ordena|impone|exige|proh[íi]be|autoriza|contempla|define|determina|"
    r"reconoce|sanciona|obliga|otorga|confiere|prescribe|contiene|precisa|"
    r"indica|dice|es|son|fue|era|que|cuyo|cuya|donde|as[íi]|en\s+relaci[óo]n|"
    r"vigente|aplicable|en\s+vigor|y\s+(?:el|la|los|las)|y\s+\d|,\s*"
    r"(?:vigente|aplicable|publicad))\b.*$", re.I)
# EL LÍMITE DE PALABRA NO ES UN DETALLE. Sin él, «es» casaba con el arranque de
# «ESTADO» y «Estados Unidos»: «…Contencioso Administrativo del Estado de
# Querétaro» quedaba en «…Administrativo del», dejaba de parecer ley local y la
# guarda entera daba por buena la cita que existía para atrapar. Así salió la
# primera calibración contra el 2/2026: cero detecciones sobre dos citas malas.

# Entidades con su ley propia que el nombre no siempre marca como estatal.
_LOCALES_SIN_ESTADO = re.compile(r"distrito\s+federal|ciudad\s+de\s+m[ée]xico", re.I)

# Si la mención cuelga de un criterio, es jurisprudencia ajena y está permitida
# —misma regla calibrada de `fase6_estudio._RX_ES_CRITERIO`—.
_RX_ES_CRITERIO = re.compile(
    r"(criterio|tesis|jurisprudencia|precedente|contradicci[óo]n|rubro|"
    r"ejecutoria|registro\s+digital)", re.I)


def _sa(x: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(x or ""))
                   if not unicodedata.combining(c))


# El nombre de una ley local termina en su entidad. Lo que venga detrás
# —«…del Estado de Querétaro basta para imponer», «…en materia civil»— ya no es
# nombre, y una palabra distintiva colada en la cola («civil», «fiscal») haría
# que `misma_ley` negara la identidad y la guarda acusara a una cita buena.
_RX_HASTA_ENTIDAD = re.compile(
    r"^(.*?\b(?:(?:del|para\s+el)\s+Estado\s+(?:Libre\s+y\s+Soberano\s+)?de\s+"
    r"[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+(?:\s+(?:de\s+)?[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+){0,3}"
    r"|Distrito\s+Federal|Ciudad\s+de\s+M[ée]xico))\b")


def _limpia_nombre(nombre: str) -> str:
    n = " ".join(str(nombre or "").split()).strip(" .,;:")
    n = _RX_FIN_DE_NOMBRE.sub("", n).strip(" .,;:")
    m = _RX_HASTA_ENTIDAD.match(n)
    if m:
        n = m.group(1)
    return n


def es_local(nombre: str) -> bool:
    """¿Es ley de una entidad federativa?

    `fase6_rag.fuero_de` acierta en casi todo, pero toma por federal el
    «Código Civil para el DISTRITO FEDERAL» —la palabra «Federal» lo engaña— y
    ése es ley local de la Ciudad de México. Aquí se corrige sin tocar
    `fuero_de`, del que dependen otras rutas.
    """
    if _LOCALES_SIN_ESTADO.search(nombre or ""):
        return True
    try:
        import fase6_rag as _r
        return _r.fuero_de(nombre) == "estatal"
    except Exception:
        return bool(re.search(r"\bdel?\s+estado\b|para\s+el\s+estado", nombre or "", re.I))


def _misma(a: str, b: str) -> bool:
    try:
        import fase6_rag as _r
        return _r.misma_ley(a, b) or _r.misma_ley(b, a)
    except Exception:
        na, nb = _sa(a).lower(), _sa(b).lower()
        return na in nb or nb in na


# ═══════════════════════════════════════════════════════════════════════════
# LA LITIS
# ═══════════════════════════════════════════════════════════════════════════

def textos_de_la_litis(fases) -> list:
    """Lo que el expediente dice: acto o recurrida, escrito, y lo hecho de ellos."""
    fuera = list(getattr(fases, "fuentes", None) or [])
    for campo in ("antecedentes", "resumen_acto", "resumen_conceptos",
                  "resolutivo_recurrida", "resolvio_a_quo"):
        v = getattr(fases, campo, None)
        if isinstance(v, (list, tuple)):
            fuera.extend(str(x) for x in v)
        elif v:
            fuera.append(str(v))
    return [t for t in fuera if str(t or "").strip()]


def leyes_de_la_litis(fases=None, textos=None) -> list:
    """Los ordenamientos nombrados en la litis, tal como se escribieron."""
    textos = list(textos or []) or textos_de_la_litis(fases)
    vistos, fuera = set(), []
    for t in textos:
        plano = " ".join(str(t or "").split())
        for m in _RX_NOMBRE_LEY.finditer(plano):
            n = _limpia_nombre(m.group(0))
            if len(n) < 10:
                continue
            k = _sa(n).lower()
            if k not in vistos:
                vistos.add(k)
                fuera.append(n)
    return fuera


def admisible(nombre: str, litis: list) -> bool:
    """Federal o general: sí (las reglas de siempre). Local: sólo si la litis la nombra."""
    nombre = _limpia_nombre(nombre)
    if not es_local(nombre):
        return True
    return any(_misma(nombre, _limpia_nombre(x)) for x in (litis or []))


# ═══════════════════════════════════════════════════════════════════════════
# LAS CITAS DE UN TEXTO
# ═══════════════════════════════════════════════════════════════════════════

def citas(texto: str) -> list:
    """[{inicio, fin, nums, ley, anafora}] en orden de aparición.

    Las anáforas —«de la propia ley»— se resuelven a la última ley nombrada
    ANTES de ellas: es como las lee un lector, y es la única forma de saber
    que el artículo 55 «de la propia ley» del 2/2026 era de Querétaro.
    """
    t = texto or ""
    hallazgos = []
    for m in _RX_CITA_ART.finditer(t):
        ley = _limpia_nombre(m.group("ley"))
        if len(ley) >= 10:
            hallazgos.append({"inicio": m.start(), "fin": m.start("ley") + len(ley),
                              "nums": re.findall(r"\d{1,4}", m.group("nums")),
                              "ley": ley, "anafora": False})
    for m in _RX_CITA_ANAFORA.finditer(t):
        if any(h["inicio"] <= m.start() < h["fin"] for h in hallazgos):
            continue
        hallazgos.append({"inicio": m.start(), "fin": m.end(),
                          "nums": re.findall(r"\d{1,4}", m.group("nums")),
                          "ley": "", "anafora": True})
    hallazgos.sort(key=lambda h: h["inicio"])
    ultima = ""
    for h in hallazgos:
        if h["anafora"]:
            h["ley"] = ultima
        elif h["ley"]:
            ultima = h["ley"]
    return [h for h in hallazgos if h["ley"]]


def inadmisibles(texto: str, litis: list) -> list:
    """Las citas de ley local que la litis no nombra, fuera de un criterio."""
    fuera = []
    for h in citas(texto):
        if admisible(h["ley"], litis):
            continue
        ventana = (texto[max(0, h["inicio"] - 140):h["inicio"]]
                   + texto[h["fin"]:h["fin"] + 90])
        if _RX_ES_CRITERIO.search(ventana):
            continue          # la ley la nombra un criterio transcrito: permitido
        fuera.append(h)
    return fuera


# ═══════════════════════════════════════════════════════════════════════════
# SANEAR
# ═══════════════════════════════════════════════════════════════════════════

_RX_CITA_ABRE = re.compile(r"[“«\"]")
_RX_CABECERA_EN_CITA = re.compile(
    r"^\s*[“«\"]?\s*art[íi]culo\s+\d{1,4}\s*[.\-–]*\s*", re.I)
# Frases que sólo remiten al precepto transcrito y quedan huérfanas sin él.
_RX_REMITE = re.compile(
    r"^\s*(?:de\s+(?:esta|dicha|esa|la\s+anterior|la\s+citada)\s+disposici[óo]n|"
    r"del\s+(?:precepto|numeral|art[íi]culo)\s+(?:transcrito|citado|anterior|en\s+cita)|"
    r"de\s+(?:dicho|ese|este)\s+(?:precepto|numeral|art[íi]culo)|"
    r"conforme\s+a\s+(?:dicho|ese|este)\s+(?:precepto|numeral)|"
    r"como\s+se\s+advierte\s+del\s+(?:precepto|numeral)|"
    r"de\s+lo\s+transcrito)", re.I)


def _palabras(x: str) -> list:
    return re.findall(r"[a-z0-9]{3,}", _sa(x).lower())


def _cita_que_sigue(texto: str, desde: int):
    """(inicio, fin, contenido) de la transcripción que sigue a la cita, o None."""
    m = _RX_CITA_ABRE.search(texto, desde, min(len(texto), desde + 60))
    if not m:
        return None
    abre = texto[m.start()]
    cierra = {"“": "”", "«": "»", '"': '"'}[abre]
    j = texto.find(cierra, m.start() + 1)
    if j < 0 or j - m.start() > 6000:
        return None
    fin = j + 1
    while fin < len(texto) and texto[fin] in ".;, ":
        fin += 1
    return m.start(), fin, texto[m.start() + 1:j]


def _espejo(transcrito: str, normas: list, litis: list):
    """La norma ADMISIBLE cuyo texto es, palabra por palabra, el transcrito.

    Identidad, no parecido: se exige que el 85 % de las palabras del arranque
    de la transcripción estén, en orden de aparición, en el arranque de la
    norma, y que la norma no sea a su vez inadmisible.
    """
    pt = _palabras(_RX_CABECERA_EN_CITA.sub("", transcrito))[:60]
    if len(pt) < 12:
        return None
    mejor, cuota = None, 0.0
    for n in normas or []:
        ley = str(n.get("cuerpo_legal") or "")
        if not ley or not admisible(ley, litis):
            continue
        cuerpo = re.sub(r"^\s*\[[^\]]{0,400}\]\s*", "", str(n.get("texto") or ""))
        pn = _palabras(_RX_CABECERA_EN_CITA.sub("", cuerpo))[:90]
        if not pn:
            continue
        cn = Counter(pn)
        comunes = sum(min(c, cn[w]) for w, c in Counter(pt).items())
        q = comunes / len(pt)
        if q > cuota:
            mejor, cuota = n, q
    return mejor if cuota >= 0.85 else None


def _articulo_de_norma(n: dict) -> str:
    a = str(n.get("articulo") or "").strip()
    m = re.search(r"\d{1,4}", a)
    return m.group(0) if m else a


def sanear(texto: str, litis: list, normas: list, donde: str = "el estudio"):
    """(texto_saneado, avisos). Reescribe a su espejo o retira; siempre lo dice."""
    if not (texto or "").strip():
        return texto, []
    avisos = []
    # Se trabaja de atrás hacia delante: cada cambio no mueve las posiciones de
    # las citas que quedan por tratar.
    malas = inadmisibles(texto, litis)
    # Una cita reescrita cambia a qué ley remiten las anáforas que la siguen;
    # por eso las decisiones se toman primero, en orden, y se aplican después.
    reescrita_a = {}          # ley original → (ley nueva) cuando hubo espejo
    planes = []
    for h in malas:
        tr = _cita_que_sigue(texto, h["fin"])
        esp = _espejo(tr[2], normas, litis) if tr else None
        if h["anafora"] and h["ley"] in reescrita_a and esp is not None:
            planes.append(("espejo_anafora", h, tr, esp))
        elif esp is not None and not h["anafora"]:
            reescrita_a[h["ley"]] = str(esp.get("cuerpo_legal"))
            planes.append(("espejo", h, tr, esp))
        else:
            planes.append(("retiro", h, tr, None))

    for tipo, h, tr, esp in reversed(planes):
        original = texto[h["inicio"]:(tr[1] if tr else h["fin"])]
        if tipo in ("espejo", "espejo_anafora"):
            nuevo_art = _articulo_de_norma(esp)
            nueva_ley = str(esp.get("cuerpo_legal"))
            if tr:
                cuerpo_cita = _RX_CABECERA_EN_CITA.sub("", tr[2])
                abre, cierra = texto[tr[0]], {"“": "”", "«": "»", '"': '"'}[texto[tr[0]]]
                cola = texto[tr[0] + 1 + len(tr[2]) + 1:tr[1]]
                texto = (texto[:tr[0]] + f"{abre}Artículo {nuevo_art}. {cuerpo_cita}{cierra}"
                         + cola + texto[tr[1]:])
            if tipo == "espejo":
                cita_nueva = f"artículo {nuevo_art} de la {nueva_ley}"
            else:
                cita_nueva = re.sub(r"\d{1,4}", nuevo_art,
                                    texto[h["inicio"]:h["fin"]], count=1)
            texto = texto[:h["inicio"]] + cita_nueva + texto[h["fin"]:]
            # «En el ámbito local, …» deja de ser verdad cuando la ley es federal.
            # Entre la coletilla y la cita suele ir el artículo: «En el ámbito
            # local, EL artículo 57…». Sin contarlo, la coletilla sobrevivía y el
            # proyecto decía «En el ámbito local, el artículo 51 de la Ley
            # FEDERAL…», que es peor que el error original.
            antes = texto[max(0, h["inicio"] - 70):h["inicio"]]
            ml = re.search(r"(?:en\s+el\s+[áa]mbito\s+local|a\s+nivel\s+local|"
                           r"en\s+la\s+legislaci[óo]n\s+local|en\s+sede\s+local)"
                           r",?\s*(?P<det>(?:el|los|la|las)\s+)?$", antes, re.I)
            if ml and not es_local(nueva_ley):
                corte = h["inicio"] - (len(antes) - ml.start())
                resto = (ml.group("det") or "") + texto[h["inicio"]:]
                texto = texto[:corte] + resto[:1].upper() + resto[1:]
            avisos.append(
                f"LEY LOCAL FUERA DE LA LITIS, CORREGIDA ({donde}): se "
                f"citaba «{' '.join(original.split())[:140]}…», que la sentencia "
                f"recurrida y los agravios no invocan. Su texto es el del artículo "
                f"{nuevo_art} de la {nueva_ley}, que sí rige el asunto, y la cita se "
                f"reescribió a ése. Compruébalo.")
        else:
            ini = texto.rfind("\n", 0, h["inicio"])
            ini = 0 if ini < 0 else ini + 1
            # la frase que contiene la cita, no el párrafo entero
            fr = max(texto.rfind(". ", ini, h["inicio"]), ini - 2)
            ini_frase = fr + 2 if fr >= ini else ini
            fin = tr[1] if tr else texto.find(".", h["fin"]) + 1 or h["fin"]
            siguiente = texto[fin:fin + 400]
            ms = re.match(r"\s*([^.]{0,380}\.)", siguiente)
            if ms and _RX_REMITE.match(ms.group(1)):
                fin += ms.end()
            texto = (texto[:ini_frase].rstrip() + ("\n" if texto[ini_frase - 1:ini_frase] == "\n" else " ")
                     + texto[fin:].lstrip())
            avisos.append(
                f"LEY LOCAL FUERA DE LA LITIS, RETIRADA ({donde}): se "
                f"citaba «{' '.join(original.split())[:160]}…». Ni la sentencia "
                f"recurrida ni los agravios invocan esa ley, y no hay en el material "
                f"un precepto aplicable con el mismo texto al que reescribirla. Se "
                f"retiró la cita y su transcripción; revisa que el razonamiento siga "
                f"completo.")
    return re.sub(r"[ \t]{2,}", " ", texto), list(reversed(avisos))


def filtrar_normas(normas: list, litis: list) -> tuple:
    """(normas admisibles, retiradas). Lo que no puede citarse no se le enseña al modelo."""
    buenas, fuera = [], []
    for n in normas or []:
        ley = str((n or {}).get("cuerpo_legal") or "")
        if ley and not admisible(ley, litis):
            fuera.append(f"art. {n.get('articulo')} — {ley}")
        else:
            buenas.append(n)
    return buenas, fuera
