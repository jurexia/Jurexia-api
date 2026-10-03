# -*- coding: utf-8 -*-
"""EL ANÁLISIS NEUTRAL DE LA LITIS, ANTES DE DECIDIR (rediseño del taller,
punto 1 y etapa 2; David, 29-sep-2026).

POR QUÉ. David: «Hoy plan_estudio mezcla dos trabajos: comprender qué debe
resolverse y organizar cómo escribir una decisión elegida. Separaría esos
productos.» Lo más parecido a un análisis previo era el contraste, que no es
neutral (precalifica), lee sólo los resúmenes y reduce la recurrida a UNA razón
toral; las proposiciones P de la recurrida las extraía el PLANIFICADOR, después
de decidir, y salían distintas con cada criterio.

QUÉ PRODUCE (una llamada con razonamiento + una verificación SIN modelo):
  · cuestion_central — la cuestión de derecho concreta de la que depende el
    asunto, sin reducirla a una pregunta general (en el AR 631/2025: la
    transmisión del derecho contractual concreto, no «¿se alteró la cosa
    juzgada?»);
  · razones R — lo que sostiene la recurrida (o el acto): qué afirma, qué
    concluye, y si funciona SOLA (autónoma: basta para sostener el fallo),
    JUNTO con otras (conjunta: premisas necesarias a la vez) o DEPENDIENTE de
    otra; con su cita LITERAL del acto;
  · argumentos — cada segmento del inventario (C1.a, A2.b…): qué pide, por
    qué, y qué razones R combate;
  · hechos H — qué ocurrió, quién lo afirma, dónde consta, con cita, y su
    CONDICIÓN PROBATORIA, que David pidió distinguir porque confundirlas cambia
    el sentido:
        falta_en_insumos        el documento no está en lo que se subió;
        no_acreditado           la parte no lo acreditó en el juicio;
        tenido_por_acreditado   el órgano recurrido lo tuvo por acreditado;
        acreditacion_impugnada  esa determinación probatoria está impugnada;
        no_controvertido        consta y nadie lo discute;
  · faltantes — los insumos que harían falta para decidir.
  · Lo REVISABLE y lo FIRME no se deducen aquí: vienen de la ficha procesal,
    que ya los calcula por código, y se copian.

POR CÓDIGO, NO POR EL MODELO:
  · toda cita se comprueba palabra por palabra contra su fuente
    (`plan_estudio.Texto`); la que no está deja de presentarse como cita
    (`verificada: False`) y un hecho «tenido por acreditado» sin cita
    verificada baja a «sin_verificar»;
  · las referencias (combate, con, segmento) que no existen se quitan;
  · `autonomas_sin_combatir`: las razones autónomas que ningún argumento
    combate. Es el punto 5 de David («si R1 y R2 sostienen autónomamente la
    sentencia, derrotar R1 no elimina R2»), medido sobre el análisis y no
    adivinado al redactar.

LO QUE NO ENTRA. Ni el acervo ni los precedentes de la OAJ: traen calificación y
razón de otros asuntos y empujarían el sentido antes de decidir. El análisis es
de ESTE expediente. Tampoco califica: no dice fundado ni infundado.

NUNCA BLOQUEA. Si falla o tarda, la propuesta se hace sin él (regla de David:
el secretario siempre puede pedir la propuesta al motor) y se dice.

LAS PREMISAS Y LAS PREGUNTAS (2-oct-2026, bandera «preguntas_al_secretario»).
David: «nunca dar por hecho que lo que se dice en los recursos o conceptos de
violación es cierto… sólo si no se dice [en la resolución] se le pregunta al
abogado, pero no se genera ninguna propuesta hasta que no se respondan esas
preguntas, sólo si se consideran indispensables». El análisis entrega además
las PREMISAS de hecho de la parte contrastadas con el acto, verificadas por
fuente; donde el acto calla, una pregunta cerrada. Lo que sí detiene la
propuesta lo decide el CÓDIGO (`preguntas_secretario`), no este módulo: el
análisis sigue sin bloquear nada.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

VERSION = "analisis-3"
# LAS PREMISAS DE LA PARTE, CONTRASTADAS CON EL ACTO (2-oct-2026, David: «nunca
# dar por hecho que lo que se dice en los recursos o conceptos de violación es
# cierto; eso es justo lo que se debe verificar a la luz de lo acreditado en el
# juicio»). Con la bandera «preguntas_al_secretario» el análisis lee lo mismo
# pero entrega más (las premisas, la condición «afirmado_por_la_parte») y deja
# de pedir faltantes abiertos: es otra versión, y la huella lo dice. Con la
# bandera apagada la versión, el prompt y la huella son los de siempre.
# analisis-5 (3-oct-2026, revisión adversarial): el prompt de las premisas deja
# fuera las omisiones de la propia resolución y pide `se_verifica_en`; una
# premisa cuya cita del acto no se halla conserva lo que el modelo declaró. Es
# otro análisis: la huella lo dice y el guardado con la versión anterior se
# recalcula.
VERSION_PREGUNTAS = "analisis-5"


def con_premisas() -> bool:
    """¿Rige «preguntas_al_secretario» en esta petición? (nunca lanza)"""
    try:
        import contexto_taller as _ct
        return bool(_ct.rediseno("preguntas_al_secretario"))
    except Exception:
        return False


def version() -> str:
    return VERSION_PREGUNTAS if con_premisas() else VERSION


# EL ACTO Y EL ESCRITO, ENTEROS (diagnóstico del sesgo a conceder, 29-sep-2026).
# Con 60 mil caracteres cortados por el principio, 13 de los 21 actos del banco
# Kingston llegaban sin su estudio ni sus resolutivos (el más largo mide 193
# mil) y 5 escritos sin sus últimos conceptos (hasta 126 mil). En 10 de las 11
# concesiones erróneas el análisis pedía «el texto íntegro de la sentencia
# reclamada», y sin la respuesta de la Sala el motor la suplía con una
# «omisión». Los topes cubren todo el banco; si un documento los pasa, van el
# principio y el FINAL —donde están el estudio y los resolutivos— con la
# omisión marcada (`recorte`): nunca se corta la cola.
def _entero_env(nombre: str, defecto: int) -> int:
    """Un tope del entorno; vacío o mal escrito («220k») vale el de omisión, en
    vez de tumbar el módulo al importarlo (y con él la justificación y la
    deliberación, que lo importan)."""
    try:
        return max(1, int(str(os.getenv(nombre, "") or defecto).strip()))
    except ValueError:
        return defecto


MAX_ACTO = _entero_env("ANALISIS_MAX_ACTO", 220000)
MAX_ESCRITO = _entero_env("ANALISIS_MAX_ESCRITO", 150000)
MAX_AUTOS = 20000
PRINCIPIO_SI_RECORTA = 0.25
# La salida crece con lo que se lee (más razones, hechos y un argumento por
# segmento: 48 en el 625): el peor caso estimado ronda 19-25 mil tokens entre
# razonamiento y JSON, al filo de los 24 mil de antes (revisión, 29-sep).
MAX_TOKENS = _entero_env("ANALISIS_MAX_TOKENS", 32000)
MODELO = os.getenv("MODELO_ANALISIS", "") or None     # vacío = el de la propuesta
CLAVE_MARCA = "analisis"

CONDICIONES = ("falta_en_insumos", "no_acreditado", "tenido_por_acreditado",
               "acreditacion_impugnada", "no_controvertido")
# «Lo afirma la parte y el acto no lo dice» (2-oct-2026): faltaba en el
# catálogo, y «no_controvertido» —consta y nadie lo discute— se tragaba la
# afirmación de una sola parte. Sólo con la bandera de las preguntas.
CONDICIONES_PREGUNTAS = CONDICIONES + ("afirmado_por_la_parte",)
# Lo que dice el acto de cada premisa de la parte.
EL_ACTO = ("lo_tuvo_por_cierto", "lo_desestimo", "no_se_pronuncia")
MAX_PREMISAS = 15
RELACIONES = ("autonoma", "conjunta", "dependiente")
_RX_JSON = re.compile(r"\{.*\}", re.S)


def _txt(x, n: int = 0) -> str:
    s = " ".join(str(x or "").split())
    return s[:n] if n else s


def recorte(texto, tope: int) -> tuple:
    """(lo que se entrega del documento, caracteres omitidos). Entero si cabe;
    si no, su principio (antecedentes y litis) y su final (estudio y
    resolutivos) con una marca en medio que dice cuánto se omitió, para que el
    modelo no tome el documento por incompleto ni suponga lo que no leyó."""
    t = str(texto or "")
    tope = max(1, int(tope or 1))
    if len(t) <= tope:
        return t, 0
    cab = int(tope * PRINCIPIO_SI_RECORTA)
    cola = tope - cab
    omit = len(t) - cab - cola
    return (t[:cab] + f"\n[… AQUÍ SE OMITEN {omit} CARACTERES DEL MEDIO DEL DOCUMENTO; "
            f"EL PRINCIPIO Y EL FINAL VAN COMPLETOS …]\n" + t[-cola:]), omit


# ¿CIERRA COMO UN DOCUMENTO COMPLETO? Que el sistema no lo recortara no dice
# que lo subido esté entero: el acto del ADA 400/2024 (104 mil) termina en la
# foja 40, a media transcripción de una jurisprudencia, sin resolutivos ni
# firmas. Rotularlo «ÍNTEGRO» prohibía pedir lo que de verdad faltaba y
# borraba el faltante cierto (revisión adversarial, 29-sep). Se mira la cola:
# medido en los 21 actos del banco, los otros 20 traen alguno de estos signos
# en sus últimos 8 mil caracteres.
COLA_CIERRE = 8000
_RX_CIERRE_ACTO = re.compile(r"RESUELVE|RESOLUTIV|Notif[ií]quese|\bfirman?\b|C[oó]nste|as[ií] lo resolvi"
                             r"|lo resolvieron|arch[ií]vese", re.I)
_RX_CIERRE_ESCRITO = re.compile(r"protesto|atentamente|\bpido\b|solicito|por lo expuesto|\bfirma", re.I)


def parece_completo(texto, escrito: bool = False) -> bool:
    """¿Trae su final los signos de cierre de un documento completo?"""
    t = str(texto or "")
    return bool(t.strip()) and bool((_RX_CIERRE_ESCRITO if escrito else _RX_CIERRE_ACTO).search(t[-COLA_CIERRE:]))


def _cabecera(nombre: str, texto, omit: int, escrito: bool = False) -> str:
    n = len(str(texto or ""))
    if not n:
        return f"{nombre} (no se entregó)"
    if omit:
        return f"{nombre} ({n} caracteres; se omiten {omit} del medio, marcados; el principio y el final van completos)"
    if not parece_completo(texto, escrito):
        return (f"{nombre} ({n} caracteres, sin recortar; su final no trae los signos de cierre de un "
                f"documento completo: puede haberse subido incompleto)")
    return f"{nombre} (ÍNTEGRO: {n} caracteres)"


def huella(r) -> str:
    """Lo que el análisis LEE: el acto, el escrito, las constancias, los
    planteamientos y la versión. Si cambia uno, el análisis ya no es de este
    adelanto."""
    f = getattr(r, "fases", None)
    fuentes = list(getattr(f, "fuentes", []) or []) + ["", ""]
    probs = [(_txt((p or {}).get("pregunta")), _txt((p or {}).get("combate")), _txt((p or {}).get("resolvio")))
             for p in (getattr(f, "problemas", None) or []) if isinstance(p, dict)]
    base = json.dumps([version(), recorte(fuentes[0], MAX_ACTO)[0], recorte(fuentes[1], MAX_ESCRITO)[0],
                       recorte(getattr(f, "autos", "") or "", MAX_AUTOS)[0], probs], ensure_ascii=False)
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:20]


# ═══ EL PROMPT ═══════════════════════════════════════════════════════════════

def _prompt_premisas(q: str, recurrida: str) -> str:
    """Lo que el análisis entrega con «preguntas_al_secretario» EN LUGAR de los
    faltantes abiertos: las premisas de hecho de la parte, contrastadas con el
    acto, y sólo donde el acto calla, la pregunta cerrada al secretario.

    SE DESCRIBE LA FORMA DE LA PREGUNTA, NO SE DA UNA: un ejemplo literal en un
    prompt se copia tal cual (medido tres veces en este proyecto)."""
    return f"""premisas: las afirmaciones de HECHO de la parte de las que depende la calificación de un
  {q} (no sus argumentos de derecho), como mucho {MAX_PREMISAS}. Lo que la parte afirma NO es un
  hecho del asunto: es justo lo que hay que contrastar con lo que se tuvo por acreditado en el
  juicio. Cada una:
  id ("F1"…), segmento (el id del inventario), problema (número del planteamiento),
  afirma_la_parte: el hecho que afirma, atribuido a ella,
  cita_escrito: 10 a 40 palabras LITERALES del escrito donde lo afirma,
  el_acto, UNA de estas:
    "lo_tuvo_por_cierto": {recurrida} lo dio por acreditado;
    "lo_desestimo": {recurrida} lo negó, lo tuvo por no probado o valoró la prueba en contra;
    "no_se_pronuncia": ni {recurrida} ni las constancias dicen nada de ese hecho,
  cita_acto: 10 a 40 palabras LITERALES de {recurrida} o de las constancias donde se pronunció
    ("" si no se pronuncia; NUNCA del escrito de la parte),
  carga_de: a quién toca probar ese hecho: "la_parte" (quien lo afirma), "contraparte" o "autoridad",
  se_verifica_en: "autos" si es un hecho del procedimiento que consta en el expediente (qué se alegó
    o se hizo valer, qué se promovió, qué se notificó, qué escrito o prueba se presentó y cuándo): lo
    verifica el secretario en autos y no se decide por la carga de la prueba; "juicio" si es un hecho
    que la parte debía probar en el juicio de origen,
  y SÓLO si el_acto es "no_se_pronuncia":
    pregunta: UNA pregunta cerrada y corta para el secretario, que tiene el expediente delante,
      sobre el hecho mismo y no sobre un documento que haya que traer. Se contesta con sí o no,
      y el «sí» confirma lo que afirma la parte. Si lo que decide es el texto de un documento
      (una cláusula, un acuerdo, una diligencia), pregunta qué dice textualmente ese pasaje y
      nómbralo. Sin preámbulo ni explicación, en menos de 220 caracteres,
    tipo: "si_no", o "texto" cuando pides el texto de un pasaje,
    si_si: la calificación del {q} si la respuesta confirma lo que afirma la parte
      ("fundado", "infundado" o "inoperante"),
    si_no: la calificación si no lo confirma,
    para_que: en una línea, qué decide la respuesta.
  No preguntes lo que {recurrida} ya resolvió, lo que no cambia la calificación ni lo que la parte
  no afirmó. Que {recurrida} omitió pronunciarse, no valoró una prueba o no estudió un argumento NO
  es una premisa de hecho: se comprueba leyendo {recurrida}, que tienes delante; no la listes ni la
  preguntes. Si esa omisión descansa en un hecho del procedimiento (que la parte lo hizo valer, que
  ofreció la prueba), ése sí es premisa, con se_verifica_en "autos". [] si no hay premisas de hecho
  en disputa."""


def prompt(acto: str, escrito: str, autos: str, segmentos: list, problemas: list,
           ficha: str, es_recurso: bool, tipo_asunto: str = "") -> str:
    q = "agravio" if es_recurso else "concepto de violación"
    recurrida = "la sentencia recurrida" if es_recurso else "el acto reclamado"
    acto_r, acto_om = recorte(acto, MAX_ACTO)
    escr_r, escr_om = recorte(escrito, MAX_ESCRITO)
    autos_r, autos_om = recorte(autos, MAX_AUTOS)
    segs = "\n".join(f"  {s.get('id')}: {_txt(s.get('texto'), 600)}" for s in (segmentos or [])
                     if isinstance(s, dict) and s.get("id")) or "  (sin inventario)"
    probs = "\n".join(f"  {i}. {_txt((p or {}).get('pregunta'), 400)}"
                      for i, p in enumerate(problemas or [], 1) if isinstance(p, dict)) or "  (sin planteamientos)"
    # SIN LA BANDERA, EL TEXTO DE SIEMPRE, letra por letra (paridad).
    _cond_extra, _no_califica = "", "- No califiques: nada de fundado, infundado, inoperante ni sentido propuesto."
    _pide = ('faltantes: lista de {"que": …, "por_que_importa": …} con lo que haría falta tener para\n'
             '  decidir y no está en los insumos. [] si no falta nada.')
    if con_premisas():
        _cond_extra = ("\n    \"afirmado_por_la_parte\": sólo lo dice el escrito de la parte; ni el órgano que "
                       "resolvió ni las\n       constancias lo tienen por cierto.")
        _no_califica = ("- No califiques: nada de fundado, infundado, inoperante ni sentido propuesto, salvo en\n"
                        "  si_si y si_no de las premisas, que sólo dicen qué calificación seguiría de cada respuesta.")
        _pide = _prompt_premisas(q, recurrida)
    return f"""Eres secretario de estudio y cuenta de un Tribunal Colegiado de Circuito.
ANTES de decidir nada, analiza la litis de este asunto ({tipo_asunto or 'sin tipo'}). NO
califiques ni propongas sentido: sólo describe con precisión lo que hay que resolver.

LO QUE DEBES ENTREGAR, en JSON, con estas claves:

cuestion_central: la cuestión de derecho CONCRETA de la que depende el asunto, en una
  oración. No la reduzcas a una pregunta general: nombra la figura, el derecho y el acto
  concretos que están en juego.

razones: lista de lo que sostiene {recurrida}. Cada una:
  id ("R1", "R2"…), afirma (qué dice), conclusion (qué concluye con ello),
  relacion: "autonoma" si basta SOLA para sostener lo resuelto; "conjunta" si sólo
  funciona junto con otras (di con cuáles en "con"); "dependiente" si presupone otra
  (di cuál en "con"),
  con: ids de las razones con que se relaciona,
  problemas: números de los planteamientos a que se refiere,
  cita: 10 a 40 palabras COPIADAS LITERALMENTE de {recurrida} donde lo dice.

argumentos: uno por cada segmento del inventario de abajo, con su id tal cual:
  segmento (el id), pide (qué pide), por_que (por qué lo pide),
  combate: ids de las razones R que ataca ([] si no ataca ninguna).

hechos: los hechos que importan para decidir. Cada uno:
  id ("H1"…), que (qué ocurrió), afirma (quién lo afirma: quejoso, recurrente,
  autoridad, a_quo, tercero o constancias), fuente ("acto", "escrito" o "constancias"),
  cita: 10 a 40 palabras LITERALES de esa fuente ("" si el hecho no consta en los insumos),
  condicion, UNA de estas, y distínguelas con cuidado porque confundirlas cambia el sentido:
    "falta_en_insumos": el documento o dato NO está en lo que se entregó (no confundir
       con que la parte no lo haya probado);
    "no_acreditado": la parte no lo acreditó en el juicio;
    "tenido_por_acreditado": el órgano que resolvió lo tuvo por acreditado;
    "acreditacion_impugnada": esa determinación probatoria se combate en el {q};
    "no_controvertido": consta y nadie lo discute.{_cond_extra}

{_pide}

REGLAS:
- Copia las citas palabra por palabra; si no encuentras el pasaje, deja la cita en "".
- No inventes hechos ni razones que no estén en los textos.
- Lo revisable y lo firme ya está en la ficha procesal: no lo repitas ni lo contradigas.
{_no_califica}
- Cada documento dice en su cabecera si va ÍNTEGRO. Lo que va íntegro no es un faltante:
  no pidas su «texto completo», su «parte restante» ni sus resolutivos; búscalos en él. Si la
  cabecera dice que puede estar incompleto o que se omitió una parte, sí puedes pedir lo que falte.

FICHA PROCESAL (datos, calculados por código):
{ficha or '(sin ficha)'}

PLANTEAMIENTOS:
{probs}

INVENTARIO DE ARGUMENTOS DEL ESCRITO (usa estos ids):
{segs}

{_cabecera(recurrida.upper(), acto, acto_om)}:
<<<
{acto_r}
>>>

{_cabecera(f"ESCRITO DE LA PARTE ({q}s)", escrito, escr_om, escrito=True)}:
<<<
{escr_r}
>>>

CONSTANCIAS{f" ({len(str(autos))} caracteres; se omiten {autos_om} del medio, marcados)" if autos_om else ""}:
<<<
{autos_r or '(no se entregaron)'}
>>>

Responde SÓLO con el JSON."""


# ═══ LA VERIFICACIÓN, SIN MODELO ═════════════════════════════════════════════

# ═══ LA CITA CON ERRORES DE LECTURA ══════════════════════════════════════════
#
# Medido en el 103/2025: de 8 citas «sin verificar», las que se revisaron a
# mano estaban en el acto, pero el acto viene de un escaneo con errores de
# lectura («se derivan lidselementos constitutivos desus pretensiones»,
# «la senora pilar felipaz huefta») y el modelo citó el texto limpio. Con la
# comprobación literal, ese hecho bajaba a «sin_verificar» aunque el expediente
# lo dijera.
#
# LA TOLERANCIA ES DE LECTURA, NO DE SENTIDO, y se mide PALABRA POR PALABRA
# contra el tramo alineado (revisión adversarial: la primera versión medía
# letras y aceptaba «probó» donde el acto dice «no probó», «fundado» por
# «infundado» y «2025» por «2024»). Entre la cita y el tramo sólo se admite
# que un grupo corto de palabras se lea distinto con casi las mismas letras
# («los elementos» ↔ «lidselementos», «informado que la» ↔ «informad
# quezla»). Nunca: una palabra de más o de menos, una negación distinta, una
# cifra distinta, un prefijo o una palabra corta cambiada («y» ↔ «o»).
MIN_PALABRAS_OCR = 8
_CORTAS_CON_SENTIDO = frozenset(("y", "e", "o", "u", "si", "con", "mas", "menos", "ante", "sobre", "entre"))
_NEGACIONES = frozenset(("no", "ni", "sin", "nunca", "jamas", "tampoco", "nadie", "ninguno", "ninguna", "nada"))


def _lectura_distinta(qs: list, ss: list) -> bool:
    """¿Es `ss` (el texto) una mala lectura de `qs` (la cita)?"""
    import difflib
    if len(qs) > 3 or len(ss) > 3 or not qs or not ss:
        return False
    if any(w in _NEGACIONES for w in qs + ss):
        return False
    # «escritos o l as pruebas» contra «escritos y l as pruebas»: dentro de un
    # grupo pegado, la palabra corta que cambia el sentido tiene que ser la misma.
    if any(qs.count(w) != ss.count(w) for w in set(qs + ss) if w in _CORTAS_CON_SENTIDO):
        return False
    if any(ch.isdigit() for w in qs + ss for ch in w):
        return False
    a, b = "".join(qs), "".join(ss)
    if abs(len(a) - len(b)) > 2:
        return False
    if len(qs) == 1 and len(ss) == 1 and min(len(a), len(b)) <= 3:
        return False
    # Un prefijo puesto o quitado cambia el sentido («legal»/«ilegal»,
    # «procedente»/«improcedente»); una letra final perdida es lectura.
    if a != b and (a.endswith(b) or b.endswith(a)):
        return False
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() >= 0.75


def _alinea(qw: list, win: list) -> bool:
    import difflib
    # a = la cita, b = el texto: «insert» es texto que la cita no trae (sólo
    # se tolera antes de su arranque y después de su final) y «delete» es
    # cita que el texto no trae (nunca).
    ops = difflib.SequenceMatcher(None, qw, win, autojunk=False).get_opcodes()
    while ops and ops[0][0] == "insert":
        ops.pop(0)
    while ops and ops[-1][0] == "insert":
        ops.pop()
    if not ops:
        return False
    iguales = sum(i2 - i1 for op, i1, i2, _, _ in ops if op == "equal")
    if iguales < 0.7 * len(qw):
        return False
    for op, i1, i2, j1, j2 in ops:
        if op == "equal":
            continue
        if op != "replace" or not _lectura_distinta(qw[i1:i2], win[j1:j2]):
            return False
    return True


def casi_literal(cita: str, texto) -> str:
    """La cita, si está en `texto` (un `plan_estudio.Texto`) salvo errores de
    lectura; si sobra alguna palabra en sus bordes, sin ellas (como
    `_recorte_literal`: sólo QUITA). «» si no está. Anclas de tres palabras
    votan el arranque del tramo."""
    import plan_estudio as _pe
    toks = str(cita or "").split()
    if len(_pe._palabras(cita)) < MIN_PALABRAS_OCR or not texto:
        return ""
    n = len(toks)
    minimo = max(MIN_PALABRAS_OCR, -(-3 * n // 4))
    candidatas = [toks] + [toks[a:n - (f - a)] for f in (1, 2, 3) for a in range(f + 1)]
    for plano in texto.planos:
        tw = plano.split()
        idx = getattr(texto, "_tejas", {}).get(id(plano))
        if idx is None:
            idx = {}
            for i in range(len(tw) - 2):
                idx.setdefault((tw[i], tw[i + 1], tw[i + 2]), []).append(i)
            if not hasattr(texto, "_tejas"):
                texto._tejas = {}
            texto._tejas[id(plano)] = idx
        for c in candidatas:
            if len(c) < minimo:
                continue
            qw = _pe._palabras(" ".join(c))
            votos = {}
            for j in range(len(qw) - 2):
                for p in idx.get((qw[j], qw[j + 1], qw[j + 2]), [])[:50]:
                    votos[p - j] = votos.get(p - j, 0) + 1
            # los arranques cercanos (±3) son el mismo tramo: se suman
            juntos = {}
            for s, v in votos.items():
                juntos[s // 4] = juntos.get(s // 4, 0) + v
            for g, v in sorted(juntos.items(), key=lambda kv: -kv[1])[:3]:
                if v < 2:
                    break
                a = max(0, g * 4 - 6)
                if _alinea(qw, tw[a:a + len(qw) + 12]):
                    return " ".join(c)
    return ""


def _lista(x) -> list:
    """Una lista aunque el modelo mande un valor suelto: «R1», «R1, R2», 2."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, str):
        return [s for s in re.split(r"[,;\s]+", x) if s]
    return [x]


def _relacion(x) -> str:
    """autonoma | conjunta | dependiente. Lo que no se entiende NO se da por
    autónomo: una razón autónoma sin atacar empuja a negar, y esa alarma sólo
    se da cuando el modelo lo dijo."""
    s = _plano_al(x)
    if s.startswith("autonom"):
        return "autonoma"
    if s.startswith("dependient"):
        return "dependiente"
    return "conjunta"


def _plano_al(x) -> str:
    import unicodedata
    s = unicodedata.normalize("NFD", str(x or "").lower())
    return " ".join("".join(ch for ch in s if unicodedata.category(ch) != "Mn").split())


def _fuente(x) -> str:
    s = _plano_al(x)
    if any(k in s for k in ("escrito", "agravio", "demanda", "concepto", "recurso")):
        return "escrito"
    if any(k in s for k in ("constancia", "autos", "prueba", "expediente")):
        return "constancias"
    return "acto"


def textos(acto: str = "", escrito: str = "", autos: str = "") -> dict:
    """Las tres fuentes listas para citar en ellas (`verificar_cita`)."""
    import plan_estudio as _pe
    return {"acto": _pe.Texto(acto or ""), "escrito": _pe.Texto(escrito or ""),
            "constancias": _pe.Texto(autos or "")}


def verificar_cita(c, T: dict, fuente: str = "acto", solo: tuple | None = None) -> tuple:
    """(cita, verificada, fuente donde está, lectura «literal»|«ocr»|«»).

    LA MISMA REGLA PARA TODO EL TALLER (etapa 3): el análisis neutral, la
    justificación de cada solución y su revisión citan con esta función. Se
    busca primero en la fuente declarada y después en las demás —una fuente
    vacía nunca deja pasar la cita de otra con su nombre—; literal, recortada
    por los bordes (`_recorte_literal`, sólo quita) o, al final, salvo errores
    de lectura (`casi_literal`, calibrada: ver arriba).

    `solo` (2-oct-2026): las únicas fuentes donde se busca, SIN caída a las
    demás. La premisa de la parte se cita del escrito y lo que dijo el acto, del
    acto o las constancias: una cita del escrito que se hallaba «en alguna
    fuente» pasaba por lo que el acto tuvo por cierto. La tolerancia de lectura
    (`casi_literal`) sigue valiendo, fuente por fuente."""
    import plan_estudio as _pe
    c = _txt(c, 600)
    fuente = _fuente(fuente)
    if not c:
        return "", False, fuente, ""
    orden = [fuente] + [k for k in ("acto", "escrito", "constancias") if k != fuente]
    if solo:
        _solo = [_fuente(k) for k in solo]
        orden = [k for k in orden if k in _solo]
        if not orden:
            return c, False, fuente, ""
        fuente = orden[0]
    for k in orden:
        t = T.get(k)
        if not t:
            continue
        if t.contiene(c):
            return c, True, k, "literal"
        rec, _ = _pe._recorte_literal(c, [t])
        if rec:
            return rec, True, k, "literal"
    for k in orden:
        if T.get(k):
            rec = casi_literal(c, T[k])
            if rec:
                return rec, True, k, "ocr"
    return c, False, fuente, ""


# EL FALTANTE FALSO: PEDIR EL DOCUMENTO QUE SE ENTREGÓ ENTERO. Medido en el
# banco Kingston (29-sep-2026): con el acto completo, el análisis pedía aún
# «el texto completo de la sentencia reclamada» (103, 526, 722), y en el
# prompt de la propuesta ese faltante se leía como que la Sala no había
# contestado. Se reconoce por su ARRANQUE —el documento mismo como objeto de
# lo que falta—, no por mencionarlo: «la sentencia completa de primera
# instancia, más allá de sus resolutivos reproducidos en la resolución
# reclamada» y «la totalidad del expediente…, pues sólo se entregó el acto
# reclamado» piden otra cosa y se quedan. Sólo se quita si el documento fue
# ÍNTEGRO; recortado (`recorte`), el faltante puede ser cierto.
_PARTE_DE = (r"^\s*(?:(?:los|las)\s+)?(?:(?:puntos\s+)?resolutivos\s+y\s+)?(?:el|la|los|las)?\s*"
             r"(?:texto(?:\s+(?:íntegro|integro|completo))?|parte(?:\s+\w+)?|continuación(?:\s+íntegra)?"
             r"|continuacion(?:\s+integra)?|totalidad|resto|(?:puntos\s+)?resolutivos|considerandos"
             r"|páginas\s+\w+|paginas\s+\w+|hojas\s+\w+)\s+(?:de\s+la|del|de\s+los|de\s+las)\s+")
_COPIA_DE = (r"^\s*(?:(?:una\s+)?copia\s+(?:íntegra|integra|completa)|(?:la|el)\s+(?:versión|version|contenido)"
             r"\s+(?:íntegr[oa]|integr[oa]|complet[oa]))\s+(?:de\s+la|del)\s+")
# LOS NOMBRES DEPENDEN DEL TIPO DE ASUNTO (revisión adversarial, 29-sep): en
# un recurso, «la demanda de amparo» o «el acto reclamado» son documentos que
# NO se subieron —el escrito es el recurso y el acto la sentencia recurrida—,
# y en amparo directo «la sentencia recurrida» es la de primera instancia. Sin
# saber el tipo, no se quita nada.
_NOMBRES = {
    False: (r"(?:sentencia|resolución|resolucion|acto)\s+(?:reclamad|impugnad)[ao]",
            r"(?:demanda\s+de\s+amparo|escrito\s+de\s+demanda(?:\s+de\s+amparo)?"
            r"|conceptos\s+de\s+violaci[oó]n)"),
    True: (r"(?:sentencia|resolución|resolucion)\s+recurrid[ao](?!\s+en\s+apelaci)",
           r"(?:recurso\s+de\s+(?:revisi[oó]n(?:\s+fiscal)?|queja)|escrito\s+de\s+agravios|agravios)"),
}


def _rx_del_documento(nombre: str):
    return re.compile(r"(?:" + _PARTE_DE + "|" + _COPIA_DE + r")" + nombre + r"\b"
                      r"|^\s*(?:(?:la|el)\s+)?" + nombre + r"\s+(?:completa|completo|íntegra|íntegro|integra|integro"
                      r"|entera|entero)\b", re.I)


_RX_DOC = {k: (_rx_del_documento(a), _rx_del_documento(e)) for k, (a, e) in _NOMBRES.items()}

# Si además pide OTRA cosa («la parte restante de la resolución reclamada y las
# constancias completas del expediente agrario», 704/2022; «…así como el
# informe justificado»), se queda entero: quitarlo borraría lo otro, que sí
# falta. Lo que sigue a «y», «así como», «junto con» o «además de» sólo se
# tiene por parte del mismo documento si es una de sus partes.
_RX_CONJ = re.compile(r"(?:,?\s+(?:y|e|así\s+como|asi\s+como|junto\s+con|además\s+de|ademas\s+de)\s+)"
                      r"(?:(?:el|la|los|las|sus?|un|una)\s+)?([a-záéíóúñ]+)", re.I)
_PARTES_DEL_DOCUMENTO = frozenset((
    "resolutivos", "resolutivo", "puntos", "punto", "considerandos", "considerando", "consideraciones",
    "páginas", "paginas", "página", "pagina", "fojas", "foja", "hojas", "hoja", "apartados", "apartado",
    "motivación", "motivacion", "fundamentación", "fundamentacion", "estudio", "parte", "partes", "texto",
    "resto", "análisis", "analisis", "valoración", "valoracion", "razones", "conclusiones", "efectos",
    "firmas", "firma", "antecedentes", "resultandos", "capítulos", "capitulos"))


# Sólo cuenta lo coordinado con el documento mismo, no lo que describe qué
# parte de él falta: «…reclamada, especialmente el apartado séptimo sobre
# legitimación, escisión y costas» (529) sigue pidiendo la sentencia.
_RX_DESCRIPCION = re.compile(r",?\s+(?:especialmente|incluid[oa]s?|en\s+particular|que|pues|porque|ya\s+que"
                             r"|relativ[oa]s?|sobre|donde|en\s+(?:la|el|los|las)\s+que|a\s+partir|posterior(?:es)?"
                             r"|desde|hasta|con\s+(?:sus|los|las)\s+(?:resolutivos|considerandos))\b", re.I)


def _pide_otra_cosa(resto: str) -> bool:
    """¿Lo que sigue al nombre del documento coordina OTRO documento?"""
    m = _RX_DESCRIPCION.search(resto)
    tramo = resto[:m.start()] if m else resto
    return any(c.group(1).lower() not in _PARTES_DEL_DOCUMENTO for c in _RX_CONJ.finditer(tramo))


def depurar_faltantes(faltantes: list, acto: str = "", escrito: str = "", es_recurso=None) -> tuple:
    """(faltantes que quedan, textos de los quitados): fuera los que piden el
    acto o el escrito cuando se entregaron ÍNTEGROS —sin recortar y cerrando
    como un documento completo— y nada más. `es_recurso`: el tipo de asunto
    decide cómo se llaman los dos documentos; None (no se sabe) = no se quita
    nada (ver arriba)."""
    if es_recurso is None:
        return list(faltantes or []), []
    rx_acto, rx_escr = _RX_DOC[bool(es_recurso)]
    acto_entero = len(str(acto or "")) <= MAX_ACTO and parece_completo(acto)
    escr_entero = len(str(escrito or "")) <= MAX_ESCRITO and parece_completo(escrito, escrito=True)
    quedan, fuera = [], []
    for x in faltantes or []:
        q = _txt((x or {}).get("que")) if isinstance(x, dict) else ""
        m = (acto_entero and rx_acto.search(q)) or (escr_entero and rx_escr.search(q)) if q else None
        if m and not _pide_otra_cosa(q[m.end():]):
            fuera.append(q)
        else:
            quedan.append(x)
    return quedan, fuera


def verificar(crudo: dict, acto: str, escrito: str, autos: str, segmentos: list, es_recurso=None) -> dict:
    """El análisis limpio: citas comprobadas, referencias válidas, condiciones
    del catálogo y las razones autónomas sin combatir calculadas por código.
    Nunca lanza por la forma de lo que devolvió el modelo."""
    T = textos(acto, escrito, autos)
    d = crudo if isinstance(crudo, dict) else {}

    ocr = set()

    def _cita(c, fuente: str) -> tuple:
        cita, ok, k, lectura = verificar_cita(c, T, fuente)
        if lectura == "ocr":
            ocr.add(cita)
        return cita, ok, k

    razones, ids_r = [], set()
    # Los ids que el modelo SÍ puso se reservan antes: la razón sin id no
    # puede quedarse con el de una que viene después (y borrarla).
    _explicitos = {_txt(x.get("id")).upper() for x in _lista(d.get("razones"))
                   if isinstance(x, dict) and _txt(x.get("id"))}
    for i, x in enumerate(_lista(d.get("razones")), 1):
        if not isinstance(x, dict):
            continue
        rid = _txt(x.get("id")).upper()
        if rid and rid in ids_r:
            continue
        if not rid:
            k = i
            while f"R{k}" in ids_r or f"R{k}" in _explicitos:
                k += 1
            rid = f"R{k}"
        ids_r.add(rid)
        cita, ok, _ = _cita(x.get("cita"), "acto")
        _probs = []
        for p in _lista(x.get("problemas")):
            try:
                _probs.append(int(str(p).strip()))
            except (TypeError, ValueError):
                pass
        razones.append({"id": rid, "afirma": _txt(x.get("afirma"), 800),
                        "conclusion": _txt(x.get("conclusion"), 600),
                        "relacion": _relacion(x.get("relacion")),
                        "con": [_txt(c).upper() for c in _lista(x.get("con")) if _txt(c)],
                        "problemas": _probs,
                        "cita": cita, "verificada": ok, **({"lectura": "ocr"} if cita in ocr else {})})
    for r_ in razones:
        r_["con"] = [c for c in r_["con"] if c in ids_r and c != r_["id"]]

    # LOS SEGMENTOS, POR SU ID SIN MAYÚSCULAS NI ESPACIOS; el mismo segmento
    # dos veces SUMA sus ataques (antes el segundo se perdía, y con él una
    # razón quedaba «sin combatir»).
    ids_seg = {_txt(s.get("id")).upper(): str(s.get("id")) for s in (segmentos or [])
               if isinstance(s, dict) and s.get("id")}
    argumentos, por_seg = [], {}
    for x in _lista(d.get("argumentos")):
        if not isinstance(x, dict):
            continue
        sid = _txt(x.get("segmento")).upper()
        if ids_seg:
            if sid not in ids_seg:
                continue
            sid = ids_seg[sid]
        combate = [c for c in (_txt(y).upper() for y in _lista(x.get("combate"))) if c in ids_r]
        if sid in por_seg:
            a = por_seg[sid]
            a["combate"] += [c for c in combate if c not in a["combate"]]
            continue
        por_seg[sid] = {"segmento": sid, "pide": _txt(x.get("pide"), 500),
                        "por_que": _txt(x.get("por_que"), 600), "combate": combate}
        argumentos.append(por_seg[sid])

    _prem = con_premisas()
    hechos, ids_h = [], set()
    for i, x in enumerate(_lista(d.get("hechos")), 1):
        if not isinstance(x, dict):
            continue
        fuente = _fuente(x.get("fuente"))
        cond = _txt(x.get("condicion")).lower()
        cond = cond if cond in (CONDICIONES_PREGUNTAS if _prem else CONDICIONES) else "sin_verificar"
        cita, ok, fuente = _cita(x.get("cita"), fuente)
        # LO QUE SÓLO DICE EL ESCRITO NO ES UN HECHO TENIDO POR CIERTO (2-oct-
        # 2026, con la bandera de las preguntas). Si la cita sólo se halla en el
        # escrito de la parte y el modelo dijo «tenido por acreditado» o «no
        # controvertido», la condición se corrige: lo afirma la parte, y la
        # resolución no lo dice. Si también está en el acto o en las
        # constancias, la fuente es ésa.
        if _prem and ok and fuente == "escrito" and cond in ("tenido_por_acreditado", "no_controvertido"):
            _c2, _ok2, _f2, _ = verificar_cita(cita, T, "acto", solo=("acto", "constancias"))
            if _ok2:
                fuente = _f2
            else:
                cond = "afirmado_por_la_parte"
        # LA CONDICIÓN SE SOSTIENE CON SU FUENTE: «falta en insumos» no lleva
        # cita; si trae una que SÍ está en los insumos, la condición se
        # contradice y queda «sin_verificar» (el secretario la corrige).
        # Cualquier otra condición sin cita verificada no se da por buena.
        if cond == "falta_en_insumos":
            if cita and ok:
                cond = "sin_verificar"
            else:
                cita, ok = "", True
        elif not ok:
            cond = "sin_verificar"
        hid = _txt(x.get("id")).upper() or f"H{i}"
        if hid in ids_h:
            hid = f"H{i}"
            while hid in ids_h:
                hid += "b"
        ids_h.add(hid)
        hechos.append({"id": hid, "que": _txt(x.get("que"), 600),
                       "afirma": _txt(x.get("afirma"), 60), "fuente": fuente,
                       "cita": cita, "verificada": ok, "condicion": cond,
                       **({"lectura": "ocr"} if cita and cita in ocr else {})})

    faltantes = [{"que": _txt(x.get("que"), 300), "por_que_importa": _txt(x.get("por_que_importa"), 400)}
                 for x in _lista(d.get("faltantes")) if isinstance(x, dict) and _txt(x.get("que"))]
    # LOS FALTANTES ABIERTOS DEJAN DE PEDIRSE con la bandera de las preguntas:
    # «el texto íntegro de…», «las constancias de…» eran la queja 1 de David.
    # Las preguntas cerradas sobre la premisa los sustituyen.
    if _prem:
        faltantes = []
    faltantes, depurados = depurar_faltantes(faltantes, acto, escrito, es_recurso)

    combatidas = {c for a in argumentos for c in a["combate"]}
    autonomas_sin = [r_["id"] for r_ in razones if r_["relacion"] == "autonoma" and r_["id"] not in combatidas]
    out = {"version": version(), "cuestion_central": _txt(d.get("cuestion_central"), 600),
           "razones": razones, "argumentos": argumentos, "hechos": hechos, "faltantes": faltantes,
           "faltantes_depurados": depurados,
           "autonomas_sin_combatir": autonomas_sin,
           "citas_sin_verificar": sum(1 for x in razones + hechos if x.get("cita") and not x.get("verificada"))}
    if _prem:
        out["premisas"] = verificar_premisas(d.get("premisas"), T, ids_seg, ocr)
    return out


def _el_acto(x) -> str:
    s = _plano_al(x).replace(" ", "_")
    if s.startswith("lo_tuvo") or "cierto" in s or s.startswith("tenid"):
        return "lo_tuvo_por_cierto"
    if s.startswith("lo_desestim") or s.startswith("desestim") or "nego" in s:
        return "lo_desestimo"
    return "no_se_pronuncia"


def _carga(x) -> str:
    s = _plano_al(x)
    if any(k in s for k in ("autoridad", "responsable")):
        return "autoridad"
    if any(k in s for k in ("contrap", "tercer", "contrari", "demandad")):
        return "contraparte"
    return "la_parte"


# DÓNDE SE COMPRUEBA UNA PREMISA (3-oct-2026, revisión adversarial: «"No se
# pronuncia" y la carga de la prueba deciden en contra de los planteamientos de
# omisión»). La carga de la prueba sólo decide los hechos que la parte debía
# probar en el juicio de origen. Dos clases no se deciden por ella:
#   · "resolucion": la omisión que se atribuye a la propia resolución («la Sala
#     omitió pronunciarse», «no valoró la pericial»). Su silencio ES la
#     violación alegada; se comprueba leyéndola, y el motor la tiene delante.
#   · "autos": un hecho del procedimiento (qué se hizo valer, qué se notificó,
#     qué escrito se presentó). Lo verifica el tribunal en el expediente; sin
#     respuesta del secretario queda pendiente, no «no demostrado».
# Se reconoce la omisión por el VERBO DE ESTUDIO con su sujeto implícito («no
# valoró», «omitió pronunciarse», «dejó de estudiar»), no por «omitió» suelto:
# «la autoridad omitió notificarle el crédito» es un hecho del procedimiento.
VERIFICA_EN = ("juicio", "autos", "resolucion")
_ESTUDIO = (r"pronunciarse|pronunciar|estudiar|analizar|valorar|atender|examinar|resolver|considerar"
            r"|tomar\s+en\s+cuenta|ocuparse|dar\s+respuesta|contestar")
_RX_OMISION = re.compile(
    r"\b(?:omiti(?:o|eron)\s+(?:" + _ESTUDIO + r")"
    r"|omiti(?:o|eron)\s+(?:el|su)\s+(?:estudio|analisis|pronunciamiento|valoracion)"
    r"|dej(?:o|aron)\s+de\s+(?:" + _ESTUDIO + r")"
    r"|no\s+(?:se\s+)?(?:pronuncio|valoro|estudio|analizo|atendio|examino|considero|resolvio"
    r"|tomo\s+en\s+cuenta|ocupo|dio\s+respuesta|contesto)\b"
    r"|no\s+fue(?:ron)?\s+(?:valorad|estudiad|analizad|atendid|examinad|considerad)\w*"
    r"|omision\s+de\s+(?:estudio|analisis|pronunciamiento|valoracion|valorar|estudiar|analizar|pronunciarse))")
_RX_PROCESAL = re.compile(
    r"\b(?:hizo\s+valer|hice\s+valer|hicieron\s+valer|planteo|plantee|promovio|promovi|interpuso|interpuse"
    r"|ofrecio|ofreci|exhibio|exhibi|presento|presente|notific\w*|emplaz\w*"
    r"|comparecio|comparecimos|compareci|se\s+admitio|se\s+desecho|alego|alegue|opuso|opuse)\b")


def verifica_en(declarado, afirma: str = "") -> str:
    """«juicio» | «autos» | «resolucion» (ver arriba). Lo que el texto dice
    manda sobre lo que declaró el modelo cuando es una omisión de la
    resolución; si el modelo no dijo nada legible, los verbos procesales
    deciden «autos» y lo demás queda «juicio» (lo de siempre)."""
    a = _plano_al(afirma)
    if a and _RX_OMISION.search(a):
        return "resolucion"
    s = _plano_al(declarado)
    if s.startswith(("autos", "expediente", "procedimiento", "constancias")):
        return "autos"
    if s.startswith(("resolucion", "acto", "sentencia")):
        return "resolucion"
    if s.startswith(("juicio", "origen", "prueba")):
        return "juicio"
    if a and _RX_PROCESAL.search(a):
        return "autos"
    return "juicio"


def decide_la_carga(p: dict) -> bool:
    """¿Decide la carga de la prueba esta premisa si el secretario no contesta?
    Sólo un hecho del juicio de origen sobre el que la resolución calla: lo
    que la resolución declaró (aunque su cita no se hallara) no se decide por
    carga. Una premisa guardada antes de estos campos se lee como siempre
    (juicio)."""
    if not isinstance(p, dict):
        return False
    if p.get("el_acto") != "no_se_pronuncia":
        return False
    return (p.get("se_verifica_en") or "juicio") == "juicio"


def verificar_premisas(lista, T: dict, ids_seg: dict | None = None, ocr: set | None = None) -> list:
    """Las premisas de la parte, comprobadas POR FUENTE (2-oct-2026):
      · cita_escrito se busca SÓLO en el escrito;
      · cita_acto SÓLO en el acto y las constancias, nunca en el escrito; si no
        se halla, la premisa CONSERVA lo que el modelo declaró con
        `cita_acto_verificada` False (3-oct-2026, revisión adversarial: antes
        bajaba a «no se pronuncia», y una paráfrasis o una palabra de más
        convertían «lo tuvo por cierto» en un hecho que decidía la carga de la
        prueba contra la parte, sin preguntarle a nadie). Sin cita verificada no
        se cita ni decide por carga: se comprueba en la resolución;
      · `se_verifica_en` (ver `verifica_en`): una omisión de la propia
        resolución no lleva pregunta —se comprueba leyéndola—;
      · la pregunta al secretario sólo queda si el acto calla y la afirmación
        de la parte se halló en su escrito (una premisa que la parte no
        escribió no se le pregunta a nadie).
    Nunca lanza por la forma de lo que mandó el modelo."""
    out, ids = [], set()
    for i, x in enumerate(_lista(lista), 1):
        if not isinstance(x, dict) or len(out) >= MAX_PREMISAS:
            continue
        afirma = _txt(x.get("afirma_la_parte") or x.get("afirma"), 500)
        if not afirma:
            continue
        fid = _txt(x.get("id")).upper() or f"F{i}"
        while fid in ids:
            fid += "b"
        ids.add(fid)
        seg = _txt(x.get("segmento"))
        if ids_seg:
            seg = ids_seg.get(seg.upper(), "")
        try:
            prob = int(str(x.get("problema") or "").strip())
        except (TypeError, ValueError):
            prob = 0
        ce, ce_ok, _, ce_lec = verificar_cita(x.get("cita_escrito"), T, "escrito", solo=("escrito",))
        ca, ca_ok, ca_f, ca_lec = verificar_cita(x.get("cita_acto"), T, "acto", solo=("acto", "constancias"))
        el_acto = _el_acto(x.get("el_acto"))
        # LA CITA QUE NO SE HALLA NO VUELVE MUDO AL ACTO (3-oct-2026): se
        # conserva lo declarado, sin la cita, marcado sin verificar.
        if el_acto == "no_se_pronuncia" or not ca_ok:
            ca, ca_ok = "", False
        for _c, _l in ((ce, ce_lec), (ca, ca_lec)):
            if _l == "ocr" and ocr is not None and _c:
                ocr.add(_c)
        _ver = verifica_en(x.get("se_verifica_en"), afirma)
        p = {"id": fid, "segmento": seg, "problema": prob, "afirma_la_parte": afirma,
             "cita_escrito": ce if ce_ok else "", "cita_escrito_verificada": bool(ce_ok),
             "el_acto": el_acto, "cita_acto": ca, "fuente_acto": ca_f if ca_ok else "",
             "carga_de": _carga(x.get("carga_de")), "se_verifica_en": _ver}
        if el_acto != "no_se_pronuncia":
            p["cita_acto_verificada"] = bool(ca_ok)
        if el_acto == "no_se_pronuncia" and ce_ok and _ver != "resolucion" and _txt(x.get("pregunta")):
            p.update({"pregunta": _txt(x.get("pregunta"), 400),
                      "tipo": "texto" if _plano_al(x.get("tipo")).startswith(("texto", "literal", "textual"))
                      else "si_no",
                      "si_si": _txt(x.get("si_si"), 120), "si_no": _txt(x.get("si_no"), 120),
                      "para_que": _txt(x.get("para_que"), 300)})
        out.append(p)
    return out


# ═══ LA LLAMADA ══════════════════════════════════════════════════════════════

async def analizar(cliente, r, segmentos: list, ficha: str = "") -> dict | None:
    """El análisis de este adelanto, o None si falla (nunca lanza)."""
    try:
        import fase5_propuesta as _f5
        import llamada_modelo as _lm
        f = r.fases
        fuentes = list(getattr(f, "fuentes", []) or []) + ["", ""]
        acto, escrito = str(fuentes[0] or ""), str(fuentes[1] or "")
        autos = str(getattr(f, "autos", "") or "")
        e = getattr(r, "encargo", None)
        es_rec = bool(e is not None and getattr(e, "es_recurso", False))
        tipo = str(getattr(e, "tipo_asunto", "") or "")
        probs = [p for p in (getattr(f, "problemas", None) or []) if isinstance(p, dict)]
        kw = dict(model=MODELO or _f5.MODELO_PROPUESTA, temperature=0, seed=20260929,
                  max_completion_tokens=MAX_TOKENS,
                  messages=[{"role": "user", "content": prompt(
                      acto, escrito, autos, segmentos, probs, ficha, es_rec, tipo)}])
        if _f5.ESFUERZO_PROPUESTA:
            kw["reasoning_effort"] = _f5.ESFUERZO_PROPUESTA
        resp = await _lm.crear(cliente, **kw)
        crudo = (resp.choices[0].message.content or "").strip()
        if not crudo:
            print("   🔎 ANÁLISIS: volvió vacío; se propone sin él")
            return None
        m = _RX_JSON.search(crudo)
        if not m:
            print("   🔎 ANÁLISIS: sin JSON; se propone sin él")
            return None
        doc = verificar(json.loads(m.group(0)), acto, escrito, autos, segmentos, es_rec)
        print(f"   🔎 ANÁLISIS NEUTRAL: {len(doc['razones'])} razones "
              f"({len(doc['autonomas_sin_combatir'])} autónomas sin combatir) · "
              f"{len(doc['argumentos'])} argumentos · {len(doc['hechos'])} hechos · "
              f"{len(doc['faltantes'])} faltantes ({len(doc.get('faltantes_depurados') or [])} del propio documento, "
              f"quitados) · {doc['citas_sin_verificar']} citas sin verificar"
              + (f" · {len(doc['premisas'])} premisas de la parte "
                 f"({sum(1 for p in doc['premisas'] if p.get('pregunta'))} con pregunta)"
                 if isinstance(doc.get("premisas"), list) else ""))
        return doc
    except Exception as ex:
        print(f"   🔎 ANÁLISIS: falló ({type(ex).__name__}); se propone sin él")
        return None


# ═══ PARA LA PROPUESTA ═══════════════════════════════════════════════════════

_TXT_COND = {"falta_en_insumos": "NO CONSTA en los insumos (no es lo mismo que no probado)",
             "no_acreditado": "la parte NO lo acreditó",
             "tenido_por_acreditado": "el órgano recurrido lo TUVO POR ACREDITADO",
             "acreditacion_impugnada": "su acreditación ESTÁ IMPUGNADA",
             "no_controvertido": "consta y no se controvierte",
             "sin_verificar": "condición SIN VERIFICAR (su cita no se halló)",
             # Sólo nace con la bandera de las preguntas (2-oct-2026).
             "afirmado_por_la_parte": "SÓLO LO AFIRMA LA PARTE: la resolución no lo tiene por cierto"}

_TXT_EL_ACTO = {"lo_tuvo_por_cierto": "la resolución LO TUVO POR CIERTO",
                "lo_desestimo": "la resolución LO DESESTIMÓ",
                "no_se_pronuncia": "la resolución NO SE PRONUNCIA"}
_TXT_RESPUESTA = {"si": "SÍ", "no": "NO", "no_consta": "NO CONSTA (se lee como no acreditado)"}


MAX_RAZONES_BLOQUE = 20
MAX_HECHOS_BLOQUE = 30


def bloque_propuesta(doc: dict | None) -> str:
    """El análisis como DATOS para la propuesta, con su regla. «» si no hay."""
    if not isinstance(doc, dict) or not (doc.get("razones") or doc.get("hechos")):
        return ""
    L = ["", "ANÁLISIS NEUTRAL DE LA LITIS (datos previos a decidir; no califica):"]
    if doc.get("cuestion_central"):
        L.append(f"  CUESTIÓN CENTRAL: {doc['cuestion_central']}")
    if doc.get("razones"):
        L.append("  RAZONES DE LO RESUELTO:")
        for r_ in doc["razones"][:MAX_RAZONES_BLOQUE]:
            rel = {"autonoma": "AUTÓNOMA (basta sola)", "conjunta": "CONJUNTA",
                   "dependiente": "DEPENDIENTE"}.get(r_.get("relacion"), "CONJUNTA")
            con = f" con {', '.join(r_['con'])}" if r_.get("con") else ""
            L.append(f"   {r_['id']} [{rel}{con}] {r_['afirma']} → {r_['conclusion']}"
                     + (f"\n       cita{'' if r_['verificada'] else ' NO verificada'}"
                        f"{' (cotejada salvo errores de lectura del texto: no es copia letra por letra)' if r_.get('lectura') == 'ocr' else ''}"
                        f": «{r_['cita']}»" if r_.get("cita") else ""))
    comb = {}
    for a in doc.get("argumentos") or []:
        for c in a.get("combate") or []:
            comb.setdefault(c, []).append(a["segmento"])
    if doc.get("razones"):
        L.append("  QUÉ ARGUMENTO ATACA CADA RAZÓN: " + "; ".join(
            f"{r_['id']} ← {', '.join(comb.get(r_['id'], [])) or 'NINGUNO'}" for r_ in doc["razones"][:MAX_RAZONES_BLOQUE]))
    if doc.get("autonomas_sin_combatir"):
        L.append(f"  ⚠ RAZONES AUTÓNOMAS QUE NINGÚN ARGUMENTO COMBATE: {', '.join(doc['autonomas_sin_combatir'])}. "
                 f"Si una basta sola para sostener lo resuelto y nadie la ataca, derrotar las demás no "
                 f"cambia el resultado: la solución que prospere tiene que decir por qué no se sostiene.")
    _prem = con_premisas()
    if doc.get("hechos"):
        L.append("  HECHOS, CON SU CONDICIÓN:")
        for h in doc["hechos"][:MAX_HECHOS_BLOQUE]:
            # QUIÉN LO AFIRMA (2-oct-2026, con la bandera de las preguntas): sin
            # este campo la propuesta perdía de quién era la versión del hecho.
            _quien = f" [lo afirma: {h['afirma']}]" if _prem and h.get("afirma") else ""
            L.append(f"   {h['id']} {h['que']}{_quien} — {_TXT_COND.get(h.get('condicion'), h.get('condicion'))}"
                     + (f" ({h.get('fuente')}{', salvo errores de lectura' if h.get('lectura') == 'ocr' else ''}"
                        f": «{str(h['cita'])[:220]}»)" if h.get("cita") else ""))
    if _prem and doc.get("premisas"):
        L.extend(_lineas_premisas(doc["premisas"]))
    if doc.get("faltantes"):
        L.append("  FALTA EN LOS INSUMOS: " + "; ".join(f"{x.get('que')} ({x.get('por_que_importa')})"
                                                     for x in doc["faltantes"][:12]))
    # LO QUE FALTA NO ES «NO ALCANZA» (humo del AR 631/2025, 29-sep): con los
    # faltantes delante, el motor decidió infundado en los dos problemas y aun
    # así puso alcanza=false —«faltan constancias esenciales»—, y la pantalla
    # se quedó sin propuesta que ofrecer. «No alcanza» es para cuando el ACERVO
    # no sostiene un sentido; un expediente incompleto se dice y baja la
    # confianza, pero el tribunal decide con lo que consta, y el secretario
    # siempre puede pedirle al motor una propuesta (regla de David).
    # EL CIERRE DE LA REGLA SIGUE AL CONTRATO DE LA PROPUESTA (3-oct-2026,
    # revisión adversarial): con «propuesta_por_probabilidad» el JSON ya no
    # tiene «alcanza» y la regla 2 prohíbe dejar el sentido vacío; mencionar
    # alcanza=false como salida legítima eran dos órdenes contrarias en el
    # mismo prompt. Sin esa bandera, el texto de siempre.
    _cierre = ("decide con lo que consta (alcanza=false queda sólo para cuando el acervo no da para "
               "sostener ningún sentido).")
    if _por_probabilidad():
        _cierre = ("decide siempre con lo que consta: si el acervo no respalda el sentido, pon "
                   "`sostenida`=false y di qué falta.")
    if _prem:
        # LA REGLA DE LAS PREMISAS (2-oct-2026, David: «nunca dar por hecho que
        # lo que se dice en los recursos o conceptos de violación es cierto»).
        # 3-oct-2026 (revisión adversarial): la carga de la prueba sólo decide
        # los hechos del juicio de origen —no las omisiones de la resolución ni
        # los hechos del procedimiento, ni lo que la resolución declaró aunque
        # su cita no se hallara—, y el hecho se nombra por su contenido y su
        # fuente: «nombra el H# o la F#» metía en la razón claves internas que
        # el secretario no ve y que el estudio copiaba a la prosa que se firma.
        L.append("  REGLA: lo que sólo afirma la parte NO es un hecho del asunto: es lo que hay que "
                 "verificar contra lo que la resolución tuvo por acreditado, las constancias y las "
                 "respuestas del secretario. Un hecho que la resolución tuvo por cierto o desestimó se toma "
                 "como ella lo dijo, salvo que el escrito combata esa valoración y demuestre el error. Si la "
                 "resolución calla sobre un hecho que la parte debía probar en el juicio y el secretario no "
                 "contestó, decide la carga de la prueba: quien afirma y no demuestra no prospera en ese "
                 "punto. Un hecho del procedimiento sin respuesta queda pendiente de verificar en autos; una "
                 "omisión que se atribuye a la resolución se comprueba leyéndola; y lo que la resolución "
                 "declaró sin que su cita se hallara se comprueba en ella: ninguno de esos tres se decide por "
                 "la carga de la prueba. Todo fundado que descanse en un hecho dice cuál es por su contenido "
                 "y dónde consta (la resolución, una constancia o la respuesta del secretario), sin escribir "
                 "los identificadores de esta lista (H1, F2, R3…), que el secretario no ve. Una razón "
                 "autónoma no combatida sostiene lo resuelto. " + _cierre[0].upper() + _cierre[1:])
        return "\n".join(L) + "\n"
    L.append("  REGLA: distingue siempre «no consta en los insumos» de «no acreditado» y de «tenido por "
             "acreditado»; una razón autónoma no combatida sostiene lo resuelto; si falta un insumo "
             "indispensable, dilo en tu razón en vez de suplirlo. Lo que falta en los insumos baja tu "
             "confianza, pero no te impide proponer: " + _cierre)
    return "\n".join(L) + "\n"


def _por_probabilidad() -> bool:
    """¿Rige «propuesta_por_probabilidad» en esta petición? (nunca lanza)"""
    try:
        import contexto_taller as _ct
        return bool(_ct.rediseno("propuesta_por_probabilidad"))
    except Exception:
        return False


MAX_PREMISAS_BLOQUE = 15


def _lineas_premisas(premisas: list) -> list:
    """La sección «PREMISAS DE LA PARTE, CONTRASTADAS CON LA RESOLUCIÓN»: qué
    afirma la parte, qué dijo de ello la resolución (con su cita), y si calló,
    la respuesta del secretario o la carga de la prueba."""
    L = ["  PREMISAS DE LA PARTE, CONTRASTADAS CON LA RESOLUCIÓN (lo que afirma no se da por cierto):"]
    for p in premisas[:MAX_PREMISAS_BLOQUE]:
        if not isinstance(p, dict):
            continue
        cab = f"   {p.get('id')}" + (f" ({p['segmento']})" if p.get("segmento") else "")
        L.append(f"{cab} la parte afirma: {p.get('afirma_la_parte')} — "
                 f"{_TXT_EL_ACTO.get(p.get('el_acto'), _TXT_EL_ACTO['no_se_pronuncia'])}"
                 + (f" ({p.get('fuente_acto') or 'acto'}: «{str(p['cita_acto'])[:220]}»)" if p.get("cita_acto") else ""))
        if p.get("el_acto") != "no_se_pronuncia":
            # LO DECLARADO SIN CITA HALLADA (3-oct-2026): se dice así, y no se
            # decide por la carga de la prueba.
            if p.get("cita_acto_verificada") is False:
                L.append("       su cita de la resolución NO se halló: es lo que leyó el análisis, no un dato "
                         "comprobado; compruébalo en la resolución y no decidas este punto por la carga de la "
                         "prueba")
            continue
        if p.get("se_verifica_en") == "resolucion":
            L.append("       es una omisión que se atribuye a la resolución: se comprueba leyéndola (si no se "
                     "ocupó del punto, la omisión consta); no la decide la carga de la prueba")
            continue
        r_ = p.get("respuesta")
        if r_ not in (None, ""):
            L.append(f"       respuesta del secretario: {_TXT_RESPUESTA.get(r_, '«' + str(r_)[:600] + '»')}")
        elif p.get("se_verifica_en") == "autos":
            L.append("       sin respuesta del secretario: hecho del procedimiento pendiente de verificar en "
                     "autos; razona con lo que consta y dilo (no lo decide la carga de la prueba)")
        else:
            _c = {"la_parte": "a la parte que lo afirma", "contraparte": "a la contraparte",
                  "autoridad": "a la autoridad"}.get(p.get("carga_de"), "a la parte que lo afirma")
            L.append(f"       sin respuesta del secretario: decide la carga de la prueba, que toca {_c}")
    return L


def estado_de_premisa(p: dict) -> str:
    """Lo que la resolución dijo de la premisa, en una etiqueta corta para otros
    prompts (el examinador): «lo_tuvo_por_cierto (cita sin verificar)»,
    «no_se_pronuncia (omisión de la resolución: se comprueba leyéndola)»… Así
    nadie lee como hecho comprobado lo que el análisis no pudo citar (3-oct-2026)."""
    if not isinstance(p, dict):
        return ""
    e = str(p.get("el_acto") or "no_se_pronuncia")
    if e != "no_se_pronuncia":
        return e + (" (cita sin verificar: no decide por carga)" if p.get("cita_acto_verificada") is False else "")
    v = p.get("se_verifica_en")
    if v == "resolucion":
        return e + " (omisión de la resolución: se comprueba leyéndola, no por carga)"
    if v == "autos":
        return e + " (hecho del procedimiento: pendiente de verificar en autos, no por carga)"
    return e


# ═══ LOS IDENTIFICADORES INTERNOS FUERA DE LA PROSA (3-oct-2026) ═════════════
# La regla del análisis pedía «nombra el H# o la F# que lo sostiene» y esos ids
# —que sólo existen en este bloque— llegaban a la razón que ve el secretario y a
# la «RAZÓN DEL SECRETARIO» del estudio. La regla ya no lo pide; esto es la red:
# quita H#, F# y R# sueltos (con sus paréntesis, listas y conectores) de un
# texto. Puro y determinista. Sólo mayúsculas seguidas de 1-2 cifras (y las «b»
# de los duplicados), con límite de palabra: «H2O», «art. 14» o «F. 3» no se
# tocan.
_ID = r"(?<![\w./-])[HFR]\d{1,2}b*(?![\w/-])"
_LISTA_IDS = _ID + r"(?:\s*(?:,|;|/|\by\b|\be\b)\s*" + _ID + r")*"
_RX_IDS_PARENTESIS = re.compile(r"\s*[\(\[]\s*(?:(?:hecho|hechos|premisa|premisas|raz[oó]n|razones)\s+)?"
                                + _LISTA_IDS + r"\s*[\)\]]")
_RX_IDS_CONECTOR = re.compile(r",?\s*\b(?:conforme\s+a(?:l)?|seg[uú]n|v[eé]ase|cfr?\.?|ver)\s+"
                              r"(?:(?:el|la|los|las)\s+)?(?:(?:hecho|hechos|premisa|premisas|raz[oó]n|razones)\s+)?"
                              + _LISTA_IDS, re.I)
_RX_IDS_NOMBRADOS = re.compile(r"\b((?:el|la|los|las|este|esta|ese|esa)\s+"
                               r"(?:hecho|hechos|premisa|premisas|raz[oó]n|razones))\s+" + _LISTA_IDS, re.I)
_RX_IDS_SUELTOS = re.compile(r"\s*" + _LISTA_IDS)


def quitar_ids_internos(texto):
    """El texto sin los identificadores internos del análisis (H1, F2, R3…).
    Para las razones de la propuesta (razon, en_contra, alternativa, checklist)
    antes de servirlas y de armar el criterio. Lo que no es texto vuelve tal cual."""
    if not isinstance(texto, str) or not texto:
        return texto
    if not re.search(_ID, texto):
        return texto
    t = _RX_IDS_PARENTESIS.sub("", texto)
    t = _RX_IDS_CONECTOR.sub("", t)
    t = _RX_IDS_NOMBRADOS.sub(r"\1", t)
    t = _RX_IDS_SUELTOS.sub("", t)
    t = re.sub(r"[ \t]+([,.;:)\]])", r"\1", t)
    t = re.sub(r"\([ \t]*\)", "", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r",\s*([.;:])", r"\1", t)
    # «Según H1 y F2, procede» no deja una coma huérfana al arranque.
    t = re.sub(r"^[\s,;:]+", "", t)
    if t and texto.lstrip()[:1].isupper():
        t = t[0].upper() + t[1:]
    return t.strip() if texto.strip() == texto else t


def quitar_ids_de_propuesta(obj):
    """Una COPIA de la propuesta (o de una parte suya: dict, lista o texto) con
    `quitar_ids_internos` aplicado a todo texto bajo las claves de prosa: razon,
    razonamiento, en_contra, efecto, criterio, nota, que_falta. No toca ids,
    registros, sentidos ni claves estructurales."""
    _PROSA = {"razon", "razonamiento", "en_contra", "efecto", "criterio", "nota", "que_falta",
              "si_prospera", "si_no_prospera", "por_que"}
    if isinstance(obj, list):
        return [quitar_ids_de_propuesta(x) for x in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in _PROSA and isinstance(v, str):
                out[k] = quitar_ids_internos(v)
            elif isinstance(v, (dict, list)):
                out[k] = quitar_ids_de_propuesta(v)
            else:
                out[k] = v
        return out
    return obj
