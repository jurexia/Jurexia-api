# -*- coding: utf-8 -*-
"""EL AUDITOR DE SENTENCIAS, CON LOS PRECEDENTES DEL PROPIO TRIBUNAL (29-sep-2026).

POR QUÉ. David, después de ver la nota que se armó a mano para rebatir un
proyecto sobre el incidente del artículo 211 del Código de Procedimientos
Civiles de Querétaro: «Esto es justo lo que el botón de auditor de sentencias
debería de hacer. La posibilidad de que un magistrado, un juez, suba una
sentencia para que la revise conforme a los precedentes de su tribunal. Y le dé
una nota tan elaborada como esta, advirtiéndole sobre los precedentes y
justificando si se sostiene o no el criterio conforme a esos precedentes. Si el
magistrado pide solo invocar los que le favorecen a su postura, sí podrá
hacerlo. Pero de entrada, el auditor de sentencias debe darnos toda esta
información.» Y el motor: Terra, en esfuerzo bajo (antes gpt-5.2).

LO QUE HACE, antes de que el modelo escriba una palabra:
  1. LEE EL PROYECTO con el mismo esquema con que se leyeron las sentencias
     del tribunal (redactor-sentencias/oaj/leer_sentencias.py): pregunta,
     lo que resolvió la autoridad, lo que se combate, la calificación y la
     razón. Así la consulta y el índice hablan el mismo idioma.
  2. FIJA EL TRIBUNAL del encabezado y comprueba que sus sentencias estén
     leídas (hoy, sólo el Tercer Tribunal Colegiado en Materias Administrativa
     y Civil del Vigésimo Segundo Circuito). Si no lo están, lo DICE: la nota
     no inventa una línea que no se consultó. (Si el encabezado ya lo deja
     claro, ni siquiera se lee el proyecto: `tribunal_sin_lectura`.)
  3. BUSCA, por cada punto, los planteamientos más cercanos de ese tribunal en
     `oaj_precedentes` (de cualquier tipo de asunto: la definitividad se
     discute en quejas y en revisiones), una sentencia por NEUN, sin el propio
     asunto si ya está publicado.
  4. CLASIFICA cada precedente frente a lo que propone el proyecto: coincide,
     contradice o se distingue; lo que no es el mismo punto, fuera. Juzga un
     modelo barato (el lector), con la calificación y la razón a la vista: la
     coincidencia de palabras no decide nada.
  5. ENTREGA un catálogo CERRADO al redactor: sólo esos precedentes se pueden
     citar, por tipo, número y fecha. Con el enfoque «favorables» (lo pide el
     magistrado), los contrarios ni siquiera llegan al prompt.

LO QUE NO HACE. No cita textualmente a los precedentes: en producción no están
sus textos íntegros, sólo lo que leyó el lector (síntesis). El «tema» sí es del
tribunal y se puede citar entre comillas. No da porcentajes: el coseno mide
parecido, no que la regla del precedente gobierne este caso.

NUNCA BLOQUEA. Cualquier falla deja la auditoría como era (sin catálogo) y lo
dice en el bloque. Interruptor sin despliegue: AUDITOR_PRECEDENTES=0 vuelve al
auditor anterior (prompt y modelo de siempre).
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import unicodedata
import uuid

COLECCION = "oaj_precedentes"
VECTOR = "dense"
# El mismo espacio de nombres con que `cargar_qdrant.py` fijó los ids: el
# asunto de un NEUN se trae por su id, sin filtrar ni buscar.
NS = uuid.UUID("6f1c2a52-0a1b-4d7e-9e2f-0a0a0a0a0a0a")

# Los órganos cuyas sentencias están LEÍDAS (planteamientos). Los demás del
# circuito sólo tienen tema y síntesis de sus revisiones fiscales: no alcanza
# para decir qué criterio sostuvieron.
ORGANOS_LEIDOS = ("3TCC",)

MARCA_FAVORABLES = "[ENFOQUE_PRECEDENTES:FAVORABLES]"
MAX_PROYECTO = 120_000
MAX_PUNTOS = 6
PEDIDOS_POR_PUNTO = 60


def _entero_env(nombre: str, defecto: int) -> int:
    try:
        return max(1, int(str(os.getenv(nombre, "") or defecto).strip()))
    except ValueError:
        return defecto


def _real_env(nombre: str, defecto: float) -> float:
    try:
        return float(str(os.getenv(nombre, "") or defecto).strip())
    except ValueError:
        return defecto


# 16 y no 10: en el humo de la queja 176/2019 los diez más parecidos del punto
# del art. 211 eran casi todos de la línea posterior, y quedaban fuera los dos
# que resolvieron igual que el proyecto (58/2020 y 80/2019).
CANDIDATOS_POR_PUNTO = _entero_env("AUDITOR_CANDIDATOS", 16)
# Por grupo (coinciden, contradicen, se distinguen) se enseñan a lo más estos:
# los más antiguos dicen desde cuándo, los más recientes si sigue vigente.
MOSTRADOS_POR_GRUPO = _entero_env("AUDITOR_MOSTRADOS", 8)
# Piso de parecido para que un planteamiento llegue a ser juzgado. Bajo a
# propósito: quien decide si es el mismo punto es el clasificador, no el coseno.
COSENO_MINIMO = _real_env("AUDITOR_COSENO_MINIMO", 0.70)
TOPE_PREPARAR_S = _real_env("AUDITOR_TOPE_S", 150.0)


def activo() -> bool:
    """¿Rige el auditor con precedentes? Se lee en cada consulta."""
    return os.getenv("AUDITOR_PRECEDENTES", "1").strip().lower() in ("1", "true", "si", "sí")


def modelo() -> str:
    """El redactor de la nota. David, 29-sep-2026: «el de Terra en su versión Low»."""
    return os.getenv("AUDITOR_MODELO", "").strip() or "gpt-5.6-terra"


def esfuerzo() -> str:
    return os.getenv("AUDITOR_ESFUERZO", "").strip() or "low"


def modelo_lector() -> str:
    """El que lee el proyecto y clasifica: el mismo de la fase 3 y del índice."""
    m = os.getenv("AUDITOR_LECTOR_MODELO", "").strip()
    if m:
        return m
    try:
        import fases123_pipeline as _f
        return _f.MODELO_FASES
    except Exception:                                   # pragma: no cover
        return "gpt-5.6-luna"


# ═══ LO QUE VIENE EN EL MENSAJE ════════════════════════════════════════════

def enfoque_de(mensaje: str) -> str:
    """«favorables» si el magistrado pidió sólo los precedentes que sostienen
    el proyecto; «completo» en cualquier otro caso (el de entrada)."""
    return "favorables" if MARCA_FAVORABLES in str(mensaje or "") else "completo"


def texto_del_mensaje(mensaje: str) -> str:
    """El proyecto que viene entre los marcadores del modal."""
    t = str(mensaje or "")
    i, j = t.find("<!-- SENTENCIA_INICIO -->"), t.find("<!-- SENTENCIA_FIN -->")
    if i < 0:
        return ""
    fin = j if j > i else len(t)
    return t[i + len("<!-- SENTENCIA_INICIO -->"):fin].strip()


def _plano(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "").lower())
    return "".join(c for c in s if not unicodedata.combining(c))


# El orden importa: «revisión fiscal» antes que «revisión», y la queja antes
# que el amparo (una queja habla del amparo indirecto del que viene).
_TIPOS = (
    ("revision_fiscal", re.compile(r"\brevision\s+fiscal\b")),
    ("queja", re.compile(r"\b(?:recurso\s+de\s+)?queja\b")),
    ("amparo_revision", re.compile(r"\bamparo\s+en\s+revision\b|\brecurso\s+de\s+revision\b")),
    ("amparo_directo", re.compile(r"\bamparo\s+directo\b")),
)


def tipo_del_proyecto(texto: str) -> str:
    """La clave del tipo de asunto, leída del encabezado (primeros 4,000
    caracteres); «» si no se reconoce."""
    cab = _plano(str(texto or "")[:4000])
    for clave, rx in _TIPOS:
        if rx.search(cab):
            return clave
    return ""


_RX_TRIBUNAL = re.compile(
    r"((?:primer|segundo|tercer|cuarto|quinto|sexto|s[eé]ptimo|octavo|noveno|d[eé]cimo)[a-záéíóú\s]{0,40}?"
    r"tribunal\s+colegiado[^\n;]{0,180}?circuito)", re.I)


def tribunal_del_proyecto(texto: str, leido: dict | None = None) -> str:
    """El nombre del tribunal: el que leyó el lector o, si no, el del
    encabezado."""
    t = " ".join(str((leido or {}).get("tribunal") or "").split())
    if t and "colegiado" in t.lower():
        return t
    m = _RX_TRIBUNAL.search(str(texto or "")[:8000])
    return " ".join(m.group(1).split()) if m else t


def tribunal_sin_lectura(texto: str) -> str:
    """El tribunal del encabezado cuando ya se sabe, SIN leer el proyecto, que
    sus sentencias no están leídas; «» si hace falta leerlo para saberlo.

    Casi todos los que auditan son de otros tribunales: leerles el proyecto
    (10-15 s de espera y un centavo) para acabar diciendo que su tribunal no
    está no les da nada. Se lee sólo si el encabezado nombra un órgano leído
    (con renglones partidos, por eso el texto plano) o si no nombra ninguno."""
    import fase_oaj as _fo
    cab = str(texto or "")[:8000]
    plano = " ".join(_plano(cab).split())
    for clave in ORGANOS_LEIDOS:
        nombre = " ".join(_plano(_fo.ORGANOS_OAJ.get(clave, "").split(",")[0]).split())
        if nombre and nombre in plano:
            return ""
    menciones = [" ".join(m.group(1).split()) for m in _RX_TRIBUNAL.finditer(cab)]
    if not menciones or any(_fo.organo_de(t)[0] in ORGANOS_LEIDOS for t in menciones):
        return ""
    return menciones[0]


def expediente_del_proyecto(texto: str, leido: dict | None = None):
    """(número, año) del propio asunto, para que no salga como su precedente."""
    try:
        import fase_oaj as _fo
    except Exception:                                   # pragma: no cover
        return None
    for fuente in (str((leido or {}).get("expediente") or ""), str(texto or "")[:3000]):
        n = _fo.numero_expediente(fuente)
        if n:
            return n
    return None


# ═══ 1 · LA LECTURA DEL PROYECTO ═══════════════════════════════════════════

PROMPT_LECTURA = """Eres secretario de un Tribunal Colegiado de Circuito. Te doy un PROYECTO de sentencia ({tipo}) que se
someterá a sesión. Extrae, SÓLO de lo que está escrito, sus planteamientos con el mismo formato con que se
describen las sentencias ya resueltas del tribunal, para compararlos con ellas.

Devuelve JSON y nada más:
{{
  "tribunal": "el tribunal que resuelve, con su denominación completa tal como aparece en el encabezado",
  "expediente": "el tipo y número del asunto tal como aparece (por ejemplo, «recurso de queja 115/2026»)",
  "autoridad": "la autoridad responsable o recurrente, con su denominación tal como aparece (sin datos de particulares)",
  "acto": "el acto reclamado o la resolución recurrida, en una línea",
  "planteamientos": [
    {{"pregunta": "la cuestión jurídica EN FORMA DE PREGUNTA, concreta al punto (¿...?)",
      "resolvio": "qué había resuelto la autoridad o el órgano recurrido sobre este punto",
      "combate": "qué sostienen los {combate} contra eso",
      "calificacion": "fundado|infundado|inoperante|parcialmente fundado|fundado pero inoperante|no se estudió",
      "razon": "la razón toral que PROPONE el proyecto, en una o dos frases"}}
  ],
  "sentido": "lo que propone resolver el proyecto, breve y literal"
}}
Reglas: como máximo 6 planteamientos, los que el proyecto efectivamente estudia; una pregunta por punto,
sin repetir; nada de doctrina general ni de nombres de particulares; si algo no está en el texto, cadena
vacía. No inventes.

PROYECTO:
{texto}"""

_TIPO_LEGIBLE = {"amparo_directo": "amparo directo", "amparo_revision": "amparo en revisión",
                 "queja": "recurso de queja", "revision_fiscal": "revisión fiscal"}
_CAMPOS = ("pregunta", "resolvio", "combate", "calificacion", "razon")


def _json_de(crudo: str) -> dict:
    t = str(crudo or "").strip()
    try:
        d = json.loads(t)
        return d if isinstance(d, dict) else {}
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            return {}
        try:
            d = json.loads(m.group(0))
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}


def _txt(x, n: int = 0) -> str:
    s = " ".join(str(x or "").split())
    return "" if s.lower() in ("null", "none") else (s[:n] if n else s)


def normalizar_lectura(d: dict) -> dict:
    """La lectura con sus campos en texto y a lo más seis planteamientos que
    tengan pregunta. Nunca lanza."""
    d = d if isinstance(d, dict) else {}
    pl = []
    for p in d.get("planteamientos") or []:
        if not isinstance(p, dict):
            continue
        q = {k: _txt(p.get(k) if k != "resolvio" else (p.get("resolvio") or p.get("resolvió")), 1500)
             for k in _CAMPOS}
        if q["pregunta"]:
            pl.append(q)
        if len(pl) >= MAX_PUNTOS:
            break
    return {"tribunal": _txt(d.get("tribunal"), 300), "expediente": _txt(d.get("expediente"), 200),
            "autoridad": _txt(d.get("autoridad"), 400), "acto": _txt(d.get("acto"), 600),
            "sentido": _txt(d.get("sentido"), 600), "planteamientos": pl}


async def _crear(cliente, **kw):
    import llamada_modelo as _lm
    return await _lm.crear(cliente, **kw)


async def leer_proyecto(cliente, texto: str, tipo: str) -> dict | None:
    """La lectura del proyecto, o None si falla."""
    combate = "conceptos de violación" if tipo == "amparo_directo" else "agravios"
    prompt = PROMPT_LECTURA.format(tipo=_TIPO_LEGIBLE.get(tipo, "asunto de su competencia"),
                                   combate=combate, texto=str(texto or "")[:MAX_PROYECTO])
    r = await _crear(cliente, model=modelo_lector(), temperature=0, seed=20260831,
                     reasoning_effort="none", response_format={"type": "json_object"},
                     max_completion_tokens=6000, messages=[{"role": "user", "content": prompt}])
    d = normalizar_lectura(_json_de(r.choices[0].message.content or ""))
    return d if d["planteamientos"] else None


def texto_consulta(p: dict) -> str:
    """Lo que se embebe: el mismo orden que el índice («pregunta combate
    resolvio»), para que la consulta y los puntos del índice se parezcan por lo
    que son y no por cómo se escribieron."""
    return " ".join(" ".join(str(p.get(k) or "") for k in ("pregunta", "combate", "resolvio")).split())[:12000]


# ═══ 2 · LOS CANDIDATOS DEL TRIBUNAL ═══════════════════════════════════════

async def _esperar(x):
    return await x if inspect.isawaitable(x) else x


def _neun(pl: dict):
    try:
        import fase_oaj as _fo
        return _fo._neun(pl)
    except Exception:                                   # pragma: no cover
        return None


def _mismo_expediente(pl: dict, propio, tipo_propio: str) -> bool:
    if not propio:
        return False
    try:
        import fase_oaj as _fo
        n = _fo.numero_expediente(str((pl or {}).get("alias") or ""))
        tipo_oaj = _fo.tipo_oaj(tipo_propio) if tipo_propio else ""
    except Exception:                                   # pragma: no cover
        return False
    return bool(n and tuple(n) == tuple(propio) and (not tipo_oaj or (pl or {}).get("tipo") == tipo_oaj))


def id_asunto(neun: int) -> str:
    return str(uuid.uuid5(NS, f"asunto:{neun}"))


async def candidatos(qdrant, embed, punto: dict, organo_oaj: str, propio=None, tipo_propio: str = "",
                     tope: int = 0) -> list:
    """Los precedentes más cercanos a un punto del proyecto: una sentencia por
    NEUN (su mejor planteamiento), sin el propio asunto, con los datos del
    asunto (tema, síntesis, sentido, fecha). [] si no hay o algo falla."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    q = texto_consulta(punto)
    if not q or not organo_oaj:
        return []
    vec = await _esperar(embed(q))
    if not vec:
        return []
    filtro = Filter(must=[FieldCondition(key="clase", match=MatchValue(value="planteamiento")),
                          FieldCondition(key="organo", match=MatchValue(value=organo_oaj))])
    r = await _esperar(qdrant.query_points(
        collection_name=COLECCION, query=vec, using=VECTOR, query_filter=filtro,
        limit=PEDIDOS_POR_PUNTO, score_threshold=COSENO_MINIMO, with_payload=True))
    mejor = {}
    for p in list(getattr(r, "points", None) or []):
        pl = getattr(p, "payload", None) or {}
        n = _neun(pl)
        if n is None or _mismo_expediente(pl, propio, tipo_propio):
            continue
        sc = float(getattr(p, "score", 0.0) or 0.0)
        if n not in mejor or sc > mejor[n][0]:
            mejor[n] = (sc, pl)
    elegidos = sorted(mejor.items(), key=lambda kv: -kv[1][0])[:tope or CANDIDATOS_POR_PUNTO]
    if not elegidos:
        return []
    asuntos = {}
    try:
        rs = await _esperar(qdrant.retrieve(collection_name=COLECCION,
                                            ids=[id_asunto(n) for n, _ in elegidos], with_payload=True))
        for a in rs or []:
            pa = getattr(a, "payload", None) or {}
            na = _neun(pa)
            if na is not None:
                asuntos[na] = pa
    except Exception as ex:
        print(f"   ⚖️ AUDITOR: no se trajeron los asuntos ({type(ex).__name__}); se sigue con los planteamientos")
    fuera = []
    for n, (sc, pl) in elegidos:
        a = asuntos.get(n) or {}
        fuera.append({
            "neun": n, "tipo": _txt(pl.get("tipo")), "expediente": _txt(pl.get("alias")),
            "fecha": _txt(pl.get("fecha")), "sentido": _txt(a.get("sentido") or pl.get("sentido"), 300),
            "tema": _txt(a.get("tema") or pl.get("tema"), 700), "sintesis": _txt(a.get("sintesis"), 400),
            "pregunta": _txt(pl.get("pregunta"), 700), "resolvio": _txt(pl.get("resolvio"), 700),
            "combate": _txt(pl.get("combate"), 700), "calificacion": _txt(pl.get("calificacion"), 60),
            "razon": _txt(pl.get("razon"), 900), "coseno": round(sc, 3)})
    return fuera


# ═══ 3 · LA CLASIFICACIÓN FRENTE AL PROYECTO ═══════════════════════════════

PROMPT_CLASIFICAR = """Eres secretario de estudio y cuenta de un Tribunal Colegiado de Circuito. Te doy UN PUNTO que decide un
PROYECTO de sentencia y varios precedentes del MISMO tribunal. Juzga cada precedente frente a ese punto.

EL PUNTO DEL PROYECTO
  Pregunta: {pregunta}
  Lo que resolvió la autoridad: {resolvio}
  Lo que se combate: {combate}
  Lo que propone el proyecto: {calificacion} — {razon}

LOS PRECEDENTES
{precedentes}

Para cada precedente decide, con lo que RESOLVIÓ (su calificación y su razón), no con las palabras que
comparte con el proyecto:
  mismo_punto: "si" si resolvió la misma cuestión jurídica; "parcial" si es la misma cuestión con un supuesto
    distinto que puede importar; "no" si es otra cuestión.
  relacion (si mismo_punto no es "no"): "coincide" si sostuvo el mismo criterio que propone el proyecto;
    "contradice" si sostuvo el criterio contrario; "distingue" si llegó a otra conclusión por circunstancias
    del caso que permiten distinguirlo.
  por_que: una frase concreta que diga qué sostuvo el precedente y por qué coincide, contradice o se distingue.

Devuelve JSON y nada más: {{"precedentes": [{{"id": "P1", "mismo_punto": "...", "relacion": "...", "por_que": "..."}}]}}"""


def _linea_candidato(i: int, c: dict) -> str:
    return (f"P{i}. {c['tipo']} {c['expediente']} ({fecha_legible(c['fecha'])}); sentido: {c['sentido'] or 'no consta'}\n"
            f"    Tema: {c['tema'] or '—'}\n"
            f"    Pregunta: {c['pregunta']}\n"
            f"    Calificación: {c['calificacion'] or '—'}. Razón: {c['razon'] or '—'}")


async def clasificar(cliente, punto: dict, cands: list) -> list:
    """Los candidatos que SÍ tocan el punto, con su relación con el proyecto.
    Si el clasificador falla, [] (mejor callar que presentar como línea lo que
    sólo se parece)."""
    if not cands:
        return []
    prompt = PROMPT_CLASIFICAR.format(
        pregunta=punto.get("pregunta") or "", resolvio=punto.get("resolvio") or "",
        combate=punto.get("combate") or "", calificacion=punto.get("calificacion") or "no consta",
        razon=punto.get("razon") or "", precedentes="\n".join(_linea_candidato(i, c) for i, c in enumerate(cands, 1)))
    r = await _crear(cliente, model=modelo_lector(), temperature=0, seed=20260929,
                     reasoning_effort="low", response_format={"type": "json_object"},
                     max_completion_tokens=5000, messages=[{"role": "user", "content": prompt}])
    d = _json_de(r.choices[0].message.content or "")
    por_id = {}
    for x in d.get("precedentes") or []:
        if isinstance(x, dict):
            por_id[str(x.get("id") or "").strip().upper()] = x
    fuera = []
    for i, c in enumerate(cands, 1):
        x = por_id.get(f"P{i}") or {}
        mp = _plano(x.get("mismo_punto")).strip()
        rel = _plano(x.get("relacion")).strip()
        if mp not in ("si", "parcial") or rel not in ("coincide", "contradice", "distingue"):
            continue
        fuera.append({**c, "mismo_punto": mp, "relacion": rel, "por_que": _txt(x.get("por_que"), 600)})
    return fuera


# ═══ 4 · EL CATÁLOGO PARA EL REDACTOR ══════════════════════════════════════

_MESES = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
          "octubre", "noviembre", "diciembre")


def fecha_legible(f: str) -> str:
    m = re.match(r"^\s*(\d{1,2})-(\d{1,2})-(\d{4})\s*$", str(f or ""))
    if not m:
        return str(f or "fecha no consta")
    d, mes, a = int(m.group(1)), int(m.group(2)), m.group(3)
    return f"{d} de {_MESES[mes - 1]} de {a}" if 1 <= mes <= 12 else str(f)


def _clave_fecha(f: str) -> str:
    m = re.match(r"^\s*(\d{1,2})-(\d{1,2})-(\d{4})\s*$", str(f or ""))
    return f"{m.group(3)}{int(m.group(2)):02d}{int(m.group(1)):02d}" if m else "00000000"


def _renglon(c: dict) -> str:
    return (f"   · {c['tipo']} {c['expediente']}, resuelto el {fecha_legible(c['fecha'])}"
            f"{' (sentido: ' + c['sentido'] + ')' if c.get('sentido') else ''}.\n"
            f"     Punto que resolvió: {c['pregunta']}\n"
            f"     Calificación: {c['calificacion'] or 'no consta'}. Razón (síntesis de la sentencia, no cita "
            f"textual): {c['razon'] or 'no consta'}\n"
            f"     Tema registrado por el tribunal para todo el asunto (textual; puede referirse a otros puntos): "
            f"«{_corto(c['tema'], 300) or 'no consta'}»\n"
            f"     Frente al proyecto: {c['por_que'] or c['relacion']}")


def _corto(t: str, n: int) -> str:
    t = str(t or "")
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " […]"


def _muestra(grupo: list) -> tuple:
    """(los que se enseñan, cuántos más): los más antiguos y los más recientes."""
    if len(grupo) <= MOSTRADOS_POR_GRUPO:
        return grupo, 0
    k = max(1, MOSTRADOS_POR_GRUPO // 3)
    return grupo[:k] + grupo[-(MOSTRADOS_POR_GRUPO - k):], len(grupo) - MOSTRADOS_POR_GRUPO


_TITULOS = {"coincide": "SOSTIENEN EL MISMO CRITERIO QUE EL PROYECTO",
            "contradice": "SOSTIENEN EL CRITERIO CONTRARIO",
            "distingue": "LLEGARON A OTRA CONCLUSIÓN POR CIRCUNSTANCIAS DISTINTAS (distinguibles)"}


def bloque(puntos: list, tribunal: str, enfoque: str, cobertura: str) -> str:
    """El bloque que recibe el redactor. `puntos`: [{punto, precedentes}]."""
    cab = "PRECEDENTES DEL PROPIO TRIBUNAL"
    if cobertura == "sin_tribunal":
        return (f"{cab}: no consta en el proyecto qué tribunal lo resuelve, así que no se consultaron "
                f"sus precedentes. Dilo en la nota, en la sección de precedentes, y no afirmes cuál es la "
                f"línea del tribunal.")
    # Sin consulta, «Sin precedentes del Tribunal» diría que el tribunal no ha
    # resuelto nada; lo que falta es la consulta. En la prueba del 29-sep, con
    # un tribunal ajeno, la nota escribió «No se cuenta con sentencias de este
    # Tribunal sobre los puntos» y tres dictámenes «Sin precedentes».
    sin_consulta = ("En la sección III dictamina con la ley y la jurisprudencia («Se sostiene», «Se sostiene "
                    "con matices» o «No se sostiene»); no uses «Sin precedentes del Tribunal», porque no se "
                    "consultaron.")
    if cobertura == "sin_lectura":
        return (f"{cab}: las sentencias de {tribunal or 'este tribunal'} todavía no están incorporadas a "
                f"Iurexia. La sección II es sólo esta frase, UNA vez y sin subtítulos por punto: «Las sentencias "
                f"de este Tribunal aún no están incorporadas a Iurexia; por ello, esta nota no expone su línea "
                f"sobre los puntos que decide el proyecto.» No afirmes cuál es su línea. {sin_consulta}")
    if cobertura not in ("leida",):
        return (f"{cab}: no se pudieron consultar en esta ocasión. La sección II es sólo esta frase, UNA vez y "
                f"sin subtítulos por punto: «En esta ocasión no se pudieron consultar los precedentes del "
                f"Tribunal; conviene repetir la auditoría.» No afirmes cuál es su línea. {sin_consulta}")
    L = [f"{cab} ({tribunal}).",
         "CATÁLOGO CERRADO: son las sentencias del propio tribunal que resolvieron los mismos puntos, "
         "leídas de sus versiones públicas. Sólo éstas se pueden citar, por su tipo, número y fecha; la "
         "«razón» es una síntesis (no la entrecomilles); el «tema» sí es textual del tribunal."]
    if enfoque == "favorables":
        L.append("ENFOQUE PEDIDO POR EL MAGISTRADO: invocar SÓLO los precedentes que sostienen el criterio "
                 "del proyecto. No cites ni describas precedentes en sentido contrario.")
    advertir = []
    for k, x in enumerate(puntos, 1):
        p = x["punto"]
        L.append("")
        L.append(f"PUNTO {k}. {p.get('pregunta')}")
        L.append(f"   Lo que propone el proyecto: {p.get('calificacion') or 'no consta'} — {p.get('razon') or ''}")
        precs = sorted(x.get("precedentes") or [], key=lambda c: _clave_fecha(c.get("fecha")))
        if enfoque == "favorables":
            contrarios = sum(1 for c in precs if c.get("relacion") == "contradice")
            otros = len(precs)
            precs = [c for c in precs if c.get("relacion") == "coincide"]
            if not precs and otros:
                # NO es «sin precedentes»: el tribunal sí resolvió el punto, en
                # otro sentido. Callarlo del todo dejaría al magistrado ir a la
                # sesión sin saberlo (la nota completa, 29-sep: una línea de 8
                # asuntos en contra). Se dice sin citarlos, y la nota es
                # editable: la advertencia se borra si estorba.
                L.append("   Ningún precedente del tribunal sostiene el criterio del proyecto en este punto. "
                         "Escribe que no hay precedentes del Tribunal que respalden este punto; NUNCA que no "
                         "los hay sobre el punto ni que el Tribunal no se ha pronunciado. En la sección III su "
                         "dictamen es «Sin precedentes del Tribunal que lo respalden», justificado con la ley y "
                         "la jurisprudencia; no hables ahí de apartarse de la línea del Tribunal.")
                if contrarios:
                    advertir.append(k)
                continue
        if not precs:
            L.append("   Sin precedentes del tribunal sobre este punto en las sentencias leídas.")
            continue
        for rel in ("coincide", "contradice", "distingue"):
            grupo = [c for c in precs if c.get("relacion") == rel]
            if grupo:
                vistos, mas = _muestra(grupo)
                L.append(f"   {_TITULOS[rel]} ({len(grupo)}):")
                L.extend(_renglon(c) for c in vistos)
                if mas:
                    L.append(f"   Además, {mas} asunto(s) más resueltos en el mismo sentido entre esas fechas "
                             f"(sin datos individuales: menciónalos sólo como número).")
    if advertir:
        cuales = ", ".join(str(k) for k in advertir)
        L.append("")
        L.append(f"ADVERTENCIA OBLIGATORIA: en {'el punto' if len(advertir) == 1 else 'los puntos'} {cuales} el "
                 f"tribunal ha resuelto en sentido distinto al que propone el proyecto. Cierra la sección V con un "
                 f"párrafo aparte que empiece «Advertencia:» y diga eso en una frase, sin citar ni describir esos "
                 f"asuntos, y que la nota completa los expone.")
    return "\n".join(L)


# ═══ 5 · TODO JUNTO ════════════════════════════════════════════════════════

async def _punto(cliente, qdrant, embed, p, organo_oaj, propio, tipo) -> dict:
    try:
        cands = await candidatos(qdrant, embed, p, organo_oaj, propio, tipo)
        precs = await clasificar(cliente, p, cands) if cands else []
    except Exception as ex:
        print(f"   ⚖️ AUDITOR: un punto sin precedentes por error ({type(ex).__name__}: {str(ex)[:120]})")
        precs = []
    return {"punto": p, "precedentes": precs}


async def preparar(cliente, qdrant, embed, mensaje: str, paso=None) -> dict:
    """Todo lo que el redactor necesita. Nunca lanza.

    {bloque, enfoque, tribunal, organo, cobertura, puntos, coinciden,
     contradicen, distinguen, lectura}. `paso(nombre)` se llama al empezar
    cada etapa (lo pinta la pantalla del chat)."""
    def _p(nombre: str):
        try:
            if paso:
                paso(nombre)
        except Exception:
            pass

    enfoque = enfoque_de(mensaje)
    out = {"bloque": "", "enfoque": enfoque, "tribunal": "", "organo": "", "cobertura": "error",
           "puntos": 0, "coinciden": 0, "contradicen": 0, "distinguen": 0, "lectura": None}
    texto = texto_del_mensaje(mensaje)
    if not texto:
        out["bloque"] = bloque([], "", enfoque, "error")
        return out
    try:
        import fase_oaj as _fo
        tipo = tipo_del_proyecto(texto)
        ajeno = tribunal_sin_lectura(texto)
        if ajeno:
            out.update(tribunal=ajeno, cobertura="sin_lectura", bloque=bloque([], ajeno, enfoque, "sin_lectura"))
            print(f"   ⚖️ AUDITOR: «{ajeno[:60]}» no tiene sentencias leídas; se audita sin leer el proyecto")
            return out
        _p("proyecto")
        leido = await leer_proyecto(cliente, texto, tipo)
        out["lectura"] = leido
        tribunal = tribunal_del_proyecto(texto, leido)
        out["tribunal"] = tribunal
        clave, organo_oaj = _fo.organo_de(tribunal)
        if not tribunal:
            out["cobertura"] = "sin_tribunal"
        elif clave not in ORGANOS_LEIDOS:
            out["cobertura"] = "sin_lectura"
        elif not leido:
            out["cobertura"] = "error"
        else:
            out["organo"] = organo_oaj
            _p(f"tribunal|{clave}")
            propio = expediente_del_proyecto(texto, leido)
            puntos = await asyncio.gather(*[
                _punto(cliente, qdrant, embed, p, organo_oaj, propio, tipo) for p in leido["planteamientos"]])
            out["cobertura"] = "leida"
            out["puntos"] = len(puntos)
            todos = [c for x in puntos for c in x["precedentes"]]
            for rel, k in (("coincide", "coinciden"), ("contradice", "contradicen"), ("distingue", "distinguen")):
                out[k] = len({c["neun"] for c in todos if c.get("relacion") == rel})
            _p(f"contrastar|{out['coinciden']}|{out['contradicen'] if enfoque == 'completo' else 0}")
            out["bloque"] = bloque(puntos, tribunal, enfoque, "leida")
            print(f"   ⚖️ AUDITOR: {tribunal[:60]} · {len(puntos)} punto(s) · coinciden {out['coinciden']} · "
                  f"contradicen {out['contradicen']} · distinguen {out['distinguen']} · enfoque {enfoque}")
            return out
        out["bloque"] = bloque([], tribunal, enfoque, out["cobertura"])
        print(f"   ⚖️ AUDITOR: sin precedentes del tribunal ({out['cobertura']}; «{tribunal[:60]}»)")
        return out
    except Exception as ex:
        print(f"   ⚖️ AUDITOR: la preparación falló ({type(ex).__name__}: {str(ex)[:160]}); se audita sin precedentes")
        out["cobertura"] = "error"
        out["bloque"] = bloque([], out.get("tribunal") or "", enfoque, "error")
        return out


async def preparar_con_tope(cliente, qdrant, embed, mensaje: str, paso=None) -> dict:
    """`preparar` con tope de tiempo: si no llega, se audita sin catálogo."""
    try:
        return await asyncio.wait_for(preparar(cliente, qdrant, embed, mensaje, paso), timeout=TOPE_PREPARAR_S)
    except asyncio.TimeoutError:
        print(f"   ⚖️ AUDITOR: los precedentes no llegaron en {TOPE_PREPARAR_S:.0f} s; se audita sin ellos")
        enf = enfoque_de(mensaje)
        return {"bloque": bloque([], "", enf, "error"), "enfoque": enf, "tribunal": "", "organo": "",
                "cobertura": "tiempo", "puntos": 0, "coinciden": 0, "contradicen": 0, "distinguen": 0,
                "lectura": None}


# ═══ 6 · LO QUE SE LE PIDE AL REDACTOR ═════════════════════════════════════

SYSTEM_PROMPT_AUDITOR = """Eres el auditor de proyectos de sentencia de un Tribunal Colegiado de Circuito: un secretario de estudio y
cuenta con experiencia que revisa, antes de la sesión, si el proyecto se sostiene frente a los precedentes del
propio tribunal y frente al derecho aplicable. Escribes para un magistrado: con precisión, sin adornos y sin
relleno.

RECIBES
· El proyecto, entre <!-- SENTENCIA_INICIO --> y <!-- SENTENCIA_FIN -->.
· «PRECEDENTES DEL PROPIO TRIBUNAL»: un catálogo cerrado de las sentencias del tribunal que resolvieron los
  mismos puntos, ya clasificadas frente al proyecto (sostienen su criterio, lo contradicen o se distinguen).
· «CONTEXTO JURÍDICO RECUPERADO»: legislación y jurisprudencia con su [Doc ID: uuid].

ENTREGAS una NOTA DE AUDITORÍA en este orden (usa exactamente estos encabezados; «#» para el título y «##»
para cada sección):

# NOTA DE AUDITORÍA DEL PROYECTO
## I. Lo que resuelve el proyecto
   El asunto, el acto, cada punto que decide y su calificación, y el sentido que propone. Breve.
## II. Los precedentes del Tribunal sobre los puntos que decide
   Por cada punto (con un subtítulo «### Punto N. …» que diga la cuestión):
   · La línea del Tribunal: qué ha sostenido, en qué asuntos y desde cuándo, en orden cronológico
     (tipo, número y fecha de resolución de cada precedente), y si es reiterada.
   · Los precedentes que sostuvieron el criterio contrario y los que se distinguen, con su fecha, y cuál es el
     criterio más reciente o el que se ha reiterado después.
   · Si el Tribunal no tiene precedentes sobre el punto, dilo en una línea.
## III. ¿Se sostiene el criterio del proyecto?
   Por cada punto, un dictamen explícito —«Se sostiene», «Se sostiene con matices», «Se aparta de la línea
   del Tribunal», «No se sostiene» (cuando la falla es de derecho y no de precedentes), «Sin precedentes
   del Tribunal» (sólo si se consultaron y ninguno resolvió el punto) o, con el enfoque de sólo los
   favorables, «Sin precedentes del Tribunal que lo respalden»— y su justificación: por qué coincide
   o por qué no; si se aparta, qué tendría que justificar el proyecto para hacerlo (las razones del cambio
   de criterio, la diferencia relevante del caso) y qué precedentes tendría que distinguir expresamente.
## IV. Otras observaciones
   Fundamentación y motivación, congruencia (lo que se estudia frente a lo que se resuelve), jurisprudencia
   obligatoria aplicable y, si las hay, fallas de forma que importen. Sólo lo relevante; no inventes defectos.
## V. Conclusión
   El dictamen general en un párrafo y las recomendaciones concretas, en orden de importancia.

REGLAS QUE NO SE ROMPEN
1. Los precedentes del Tribunal SÓLO salen del catálogo. Cítalos por tipo, número y fecha (por ejemplo,
   «recurso de queja 264/2022, resuelto el 9 de febrero de 2023»). Nunca inventes uno, ni un número, ni una
   fecha, ni una votación.
2. La «razón» de cada precedente es una síntesis: exprésala con tus palabras, SIN comillas. Sólo el «tema»
   registrado por el Tribunal es textual y se puede entrecomillar como tal.
3. Las citas textuales (en párrafo aparte que empiece con «>») sólo pueden ser del proyecto o de un
   documento del CONTEXTO JURÍDICO RECUPERADO, copiadas palabra por palabra.
4. La legislación y la jurisprudencia que invoques deben venir del CONTEXTO JURÍDICO RECUPERADO, con su
   [Doc ID: uuid] tal como aparece. Si algo que necesitas no está, dilo sin hablar de ti ni del sistema
   («no se tuvo a la vista el texto vigente del artículo 211 del Código…»); no cites de memoria.
5. Si el catálogo dice que no se consultaron o no hay precedentes del tribunal, dilo con claridad en la
   sección II, con las palabras que indique, y no afirmes cuál es su línea. No confundas «no se
   consultaron» con «no existen».
6. Si el catálogo trae el ENFOQUE «sólo los que sostienen el proyecto», escribe la nota con ese enfoque: la
   sección II expone la línea que sostiene el criterio del proyecto y la III justifica por qué debe
   mantenerse; no cites ni describas precedentes en sentido contrario. Si en un punto ninguno lo respalda,
   di eso —no que el Tribunal no se ha pronunciado— y deja la ADVERTENCIA que el catálogo ordene.
7. Distingue siempre entre que un precedente resolvió el MISMO punto y que sólo es análogo; nunca presentes
   como línea del Tribunal lo que sólo se parece.
8. Nada de porcentajes de similitud, ni de menciones al sistema, a índices, a un «catálogo» o a cómo se
   obtuvieron los precedentes: para el lector son, simplemente, los precedentes del Tribunal. Tampoco
   escribas en primera persona («no tengo», «no cuento con», «no encontré»): la nota es impersonal.

ESTILO: español jurídico mexicano, claro y sobrio; párrafos completos y ordenados; fechas y números exactos;
sin frases de cortesía al inicio ni al final."""
