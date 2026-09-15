"""Argumentos Toulmin para la parte que demanda.

David, 15-sep-2026: «una nueva función en botón (Toulmin) que estructure
argumentos como los que hace el taller de sentencias, es decir, citando la
norma (Constitución, tratados, jurisprudencia internacional y nacional)… para
lograr un documento terminado en Word listo para imprimir».

QUÉ HACE, EN UNA SOLA PETICIÓN Y SIN ESTADO
-------------------------------------------
1. PROBLEMAS. Los hechos y lo que se pide se traducen a 2-4 preguntas
   jurídicas conceptuales —las que, contestadas a favor, dan la pretensión—.
   La búsqueda del acervo sólo funciona con conceptos, no con el relato: está
   medido en `fase6_rag.consulta_conceptual`.
2. MATERIAL. El mismo que usa el taller, a la vez:
   · `fase6_rag.material_del_caso` → tesis y jurisprudencia (v3, vector rubro,
     Corte antes que colegiados) y artículos de ley por fuero;
   · `marco_juridico.construir` → Constitución, tratados y Corte
     Interamericana con caso y párrafo.
3. CATÁLOGO CERRADO. Todo lo encontrado se numera (T1, L1, C1, V1, H1…) y el
   modelo SÓLO puede citar por esos identificadores. No escribe registros,
   rubros ni números de artículo: los pone el servidor al resolver el
   identificador. Así una cita inventada no puede llegar al documento — el
   identificador que no existe se descarta y se avisa.
4. ARGUMENTOS. Cada uno con las seis piezas de Toulmin (afirmación, datos,
   garantía, respaldo, calificador, refutación) para el panel, y además su
   REDACCIÓN en prosa de escrito —sin las etiquetas—, que es lo que va al Word.

LO QUE NO HACE
--------------
· No toca la recuperación del chat (`_texto_concepto`, pasada por concepto,
  HyDE): sólo LEE el acervo con las funciones del taller.
· No guarda nada entre peticiones: Render corre varios workers.
· No aplica ley de otra entidad: la colección estatal es la de la entidad
  elegida o ninguna, y si no hay, se dice en `avisos`.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Awaitable, Callable, Optional

import llamada_modelo as _lm

MODELO_TOULMIN = os.getenv("TOULMIN_MODEL", "gpt-5.6-luna")
ESFUERZO_TOULMIN = os.getenv("TOULMIN_ESFUERZO", "medium")

MAX_HECHOS = 12000
MAX_PRETENSION = 3000
MAX_TESIS = 10
MAX_NORMAS = 10
PLAZO_MATERIAL_SEG = 110

_RX_JSON = re.compile(r"\{[\s\S]*\}")
# Un corchete puede traer un identificador o varios: «[T1]», «[T1, T3]»,
# «[c1; v2]», «[T1 y T3]». Todos se resuelven; lo que quede entre corchetes
# después de sustituir se retira y se avisa (revisión del 15-sep-2026: «[T1, T3]»
# llegaba tal cual al Word).
_RX_GRUPO_ID = re.compile(r"\[\s*((?:[TLCVH]\d{1,2})(?:\s*(?:,|;|y|e|–|-)\s*[TLCVH]\d{1,2})*)\s*\]", re.I)
_RX_UN_ID = re.compile(r"([TLCVH])(\d{1,2})", re.I)
_RX_ID = _RX_UN_ID
_RX_RESTO_CORCHETE = re.compile(r"\[\s*[TLCVH]\s*\d{0,2}[^\]]{0,20}\]", re.I)
_RX_REGISTRO = re.compile(r"\bregistro(?:\s+digital)?\s*(?:n[úu]m(?:ero)?\.?)?\s*[:.]?\s*(\d{6,7})\b", re.I)

Aviso = Optional[Callable[[str, str], Awaitable[None]]]


def _lista(x) -> list:
    """Una lista aunque el modelo mande una cadena con saltos de línea o nada."""
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return [y.strip(" -•·\t") for y in x.splitlines() if y.strip(" -•·\t")]
    return []


# ═══════════════════════════════════════════════════════════════════════════
# 1 · LOS PROBLEMAS JURÍDICOS
# ═══════════════════════════════════════════════════════════════════════════
_PROMPT_PROBLEMAS = """Eres abogado litigante en México. Vas a preparar los fundamentos de un escrito de la PARTE que promueve (no del juez).

TIPO DE ESCRITO: {tipo}
ENTIDAD: {entidad}
MATERIA INDICADA: {materia}

HECHOS:
{hechos}

LO QUE SE PIDE:
{pretension}

Devuelve SOLO un objeto JSON con estas claves:
- "materia": una de civil, familiar, mercantil, laboral, penal, administrativa, amparo, constitucional.
- "problemas": lista de 2 a 4 cadenas; cada una, una pregunta jurídica CONCEPTUAL cuya respuesta favorable sostiene lo que se pide. Cada una nombra la figura jurídica en disputa (la acción, la prestación, el derecho, el presupuesto procesal), sin nombres de personas, sin fechas y sin cantidades, en lenguaje de rubro y no de relato.
- "convencional": true solo si los hechos tocan derechos humanos donde la Constitución remite a tratados (igualdad, niñez, debido proceso, acceso a la justicia, salud, vivienda, trabajo digno, libertad, propiedad frente a la autoridad); false en otro caso.
- "faltantes": como máximo 5 datos de hecho que hacen falta para fundar bien, cada uno en una línea corta (puede estar vacía)."""


async def problemas_del_caso(cliente, hechos: str, pretension: str, tipo: str,
                             materia: str, entidad: str) -> dict:
    kw = dict(model=MODELO_TOULMIN, max_completion_tokens=8000,
              reasoning_effort="low",
              response_format={"type": "json_object"},
              messages=[{"role": "user", "content": _PROMPT_PROBLEMAS.format(
                  tipo=tipo or "demanda", entidad=entidad or "no indicada",
                  materia=materia or "no indicada",
                  hechos=hechos[:MAX_HECHOS], pretension=pretension[:MAX_PRETENSION])}])
    # UNA RESPUESTA CORTADA NO DA ERROR: da un JSON sin cerrar, el regex no lo
    # encuentra y el caso se buscaba con la pretensión cruda. Pasó en la
    # primera prueba de alimentos. Se reintenta una vez antes de rendirse.
    d: dict = {}
    for _intento in range(2):
        r = await _lm.crear(cliente, **kw)
        texto = (r.choices[0].message.content or "").strip()
        m = _RX_JSON.search(texto)
        try:
            d = json.loads(m.group(0)) if m else {}
        except ValueError:
            d = {}
        if d.get("problemas"):
            break
    problemas = []
    for p in _lista(d.get("problemas")):
        if isinstance(p, dict):
            p = next((str(p[k]) for k in ("pregunta", "problema", "texto") if str(p.get(k) or "").strip()), "")
        p = str(p or "").strip()
        if p:
            problemas.append(p)
    problemas = problemas[:4]
    return {
        "materia": str(d.get("materia") or materia or "").strip().lower(),
        "problemas": problemas,
        "convencional": bool(d.get("convencional")),
        "faltantes": [str(x).strip() for x in _lista(d.get("faltantes")) if str(x).strip()][:6],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2 · EL CATÁLOGO CERRADO
# ═══════════════════════════════════════════════════════════════════════════
_CONSTITUCION = "Constitución Política de los Estados Unidos Mexicanos"


def _instancia_con_articulo(inst: str) -> str:
    i = (inst or "").strip()
    if not i:
        return ""
    bajo = i.lower()
    if bajo.startswith(("primera sala", "segunda sala")):
        return f"la {i} de la Suprema Corte de Justicia de la Nación"
    if bajo.startswith("pleno") and "circuito" not in bajo:
        return f"el {i} de la Suprema Corte de Justicia de la Nación"
    if bajo.startswith("pleno"):
        return f"el {i}"
    if bajo.startswith("tribunales colegiados"):
        return f"los {i}"
    if bajo.startswith(("tribunal", "segundo", "primer", "tercer")):
        return f"el {i}"
    return i


def _tipo_tesis(t: dict) -> str:
    tipo = str(t.get("tipo") or "").lower()
    if "jurisprudencia" in tipo:
        return "jurisprudencia"
    if "aislada" in tipo:
        return "tesis aislada"
    return "tesis"


def catalogo(material, marco) -> tuple[str, dict]:
    """(bloque para el prompt, fuentes por identificador).

    La fuente guarda TODO lo que el servidor necesita para escribir la cita;
    el prompt sólo lleva lo que el modelo necesita para razonar con ella.
    """
    fuentes: dict = {}
    p = ["FUENTES DISPONIBLES (sólo existen éstas; cita únicamente por su identificador entre corchetes)"]

    constitucionales = list(getattr(marco, "constitucionales", []) or [])
    if constitucionales:
        p.append("\nCONSTITUCIÓN")
        for i, x in enumerate(constitucionales, 1):
            k = f"C{i}"
            fuentes[k] = {"id": k, "clase": "constitucion", "fuente": _CONSTITUCION,
                          "articulo": str(x.articulo), "texto": x.texto[:2400],
                          "cita": f"artículo {x.articulo} de la {_CONSTITUCION}"}
            p.append(f"[{k}] Artículo {x.articulo} constitucional: {x.texto[:1100]}")

    convencionales = list(getattr(marco, "convencionales", []) or [])
    if convencionales:
        p.append("\nTRATADOS INTERNACIONALES")
        for i, x in enumerate(convencionales, 1):
            k = f"V{i}"
            art = str(x.articulo or "").strip()
            fuentes[k] = {"id": k, "clase": "tratado", "fuente": x.fuente,
                          "articulo": art, "texto": x.texto[:2000],
                          "cita": _cita_tratado(art, x.fuente)}
            p.append(f"[{k}] {x.fuente} — {art}: {x.texto[:800]}")

    coidh = list(getattr(marco, "coidh", []) or [])
    if coidh:
        p.append("\nCORTE INTERAMERICANA DE DERECHOS HUMANOS (se cita por su caso y párrafo)")
        vistos_h = set()
        i = 0
        for x in coidh:
            clave_h = (x.caso.strip().lower(), str(x.parrafo).strip())
            if clave_h in vistos_h:
                continue
            vistos_h.add(clave_h)
            i += 1
            k = f"H{i}"
            cita_h = _cita_coidh(x)
            fuentes[k] = {"id": k, "clase": "coidh", "caso": x.caso, "vs": x.vs,
                          "parrafo": x.parrafo, "serie": x.serie, "tema": x.tema,
                          "texto": x.texto[:2000],
                          "cita": f"Corte Interamericana de Derechos Humanos, {cita_h}"}
            p.append(f"[{k}] {cita_h} — {x.tema}: {x.texto[:800]}")

    locales = list(getattr(marco, "locales", []) or [])
    # SE FILTRA PRIMERO Y SE CORTA DESPUÉS. `material_del_caso` junta las normas
    # problema por problema; cortar la lista cruda a diez dejaba sólo las del
    # primer problema y, tras quitar la Constitución y los repetidos, todavía
    # menos (hallazgo de la revisión, 15-sep-2026).
    normas = list(getattr(material, "normas", []) or [])
    vistas = set()
    lista_leyes = []
    for x in locales:
        lista_leyes.append((x.fuente, str(x.articulo), x.texto))
    for n in normas:
        lista_leyes.append((str(n.get("cuerpo_legal") or ""), str(n.get("articulo") or ""),
                            str(n.get("texto") or "")))
    i = 0
    for ley, art, texto in lista_leyes:
        clave = (ley.strip().lower(), art.strip().lower())
        # Un artículo sin ley identificable no se cita: se escribiría el
        # texto de una ley con el nombre de otra.
        if not ley.strip() or not art.strip() or clave in vistas:
            continue
        # La Constitución ya va en su apartado: el mismo artículo como «ley»
        # sólo duplica la fuente y la cita.
        if "constituci" in ley.lower() and "estados unidos mexicanos" in ley.lower():
            continue
        vistas.add(clave)
        if i >= MAX_NORMAS:
            break
        i += 1
        k = f"L{i}"
        if i == 1:
            p.append("\nLEYES")
        fuentes[k] = {"id": k, "clase": "ley", "fuente": ley.strip(), "articulo": art.strip(),
                      "texto": texto[:2400], "cita": f"artículo {art.strip()} {_de_ley(ley.strip())}"}
        p.append(f"[{k}] {ley.strip()} — Artículo {art.strip()}: {texto[:1000]}")

    tesis = _tesis_repartidas(list(getattr(material, "tesis", []) or []), MAX_TESIS)
    if tesis:
        p.append("\nTESIS Y JURISPRUDENCIA NACIONAL (la Suprema Corte pesa más que un colegiado; la jurisprudencia obliga y la tesis aislada sólo orienta; cada una se usa sólo para lo que dice)")
        for i, t in enumerate(tesis, 1):
            k = f"T{i}"
            reg = str(t.get("registro") or "").strip()
            if not reg:
                continue
            tipo = _tipo_tesis(t)
            inst = str(t.get("instancia") or "").strip()
            rubro = " ".join(str(t.get("rubro") or "").split()).rstrip(" .")
            fuentes[k] = {"id": k, "clase": "tesis", "registro": reg, "rubro": rubro,
                          "instancia": inst, "tipo": tipo, "localizacion": t.get("localizacion") or "",
                          "texto": str(t.get("texto") or "")[:2400],
                          "cita": (f"{tipo} de {_instancia_con_articulo(inst)}, registro digital {reg}, "
                                   f"de rubro: «{rubro}»") if inst else
                                  f"{tipo}, registro digital {reg}, de rubro: «{rubro}»"}
            p.append(f"[{k}] {tipo.upper()} · {inst} · {rubro}\n    {str(t.get('texto') or '')[:1500]}")
    return "\n".join(p), fuentes


def _cita_tratado(ref: str, fuente: str) -> str:
    """«Art. 7 PIDESC» + «Pacto Internacional…» → «artículo 7 del Pacto Internacional…»."""
    m = re.match(r"^\s*art(?:[íi]culo|\.)?\s*(\d+[\w\.]*)", ref or "", re.I)
    if m and fuente:
        return f"artículo {m.group(1).rstrip('.')} {_de_con_articulo(fuente)}"
    if ref and fuente and ref != fuente:
        return f"{ref}, {fuente}"
    return fuente or ref


def _de_con_articulo(nombre: str) -> str:
    bajo = nombre.lower()
    if bajo.startswith(("pacto", "protocolo", "convenio", "estatuto")):
        return f"del {nombre}"
    return f"de la {nombre}"


def _cita_coidh(x) -> str:
    """«Caso X Vs. Y, párr. N (Serie C No. Z)» o «Opinión Consultiva OC-27, párr. N»."""
    caso = (x.caso or "").strip()
    if re.match(r"^OC-\d", caso, re.I):
        c = f"Opinión Consultiva {caso.upper()}"
        if x.parrafo:
            c += f", párr. {x.parrafo}"
        return c
    return x.cita()


def _de_ley(ley: str) -> str:
    """«de la Ley…», «del Código…»: la contracción que el español exige."""
    c = _con_articulo_ley(ley)
    return "del " + c[3:] if c.startswith("el ") else "de " + c


def _tesis_repartidas(tesis: list, tope: int) -> list:
    """Hasta `tope` tesis, sin que un problema se quede sin ninguna.

    `material_del_caso` marca en `para` a qué problemas sirve cada tesis y las
    ordena por jerarquía. Se toman por turnos —la mejor de cada problema, luego
    la segunda…— conservando ese orden dentro de cada uno; las que no traen
    `para` completan al final.
    """
    if len(tesis) <= tope:
        return tesis
    por_problema: dict = {}
    sueltas = []
    for t in tesis:
        para = t.get("para") or []
        if not para:
            sueltas.append(t)
            continue
        for k in para:
            por_problema.setdefault(k, []).append(t)
    elegidas, vistos = [], set()
    while len(elegidas) < tope and any(por_problema.values()):
        for k in sorted(por_problema):
            while por_problema[k] and str(por_problema[k][0].get("registro")) in vistos:
                por_problema[k].pop(0)
            if por_problema[k] and len(elegidas) < tope:
                t = por_problema[k].pop(0)
                vistos.add(str(t.get("registro")))
                elegidas.append(t)
    for t in tesis + sueltas:
        if len(elegidas) >= tope:
            break
        if str(t.get("registro")) not in vistos:
            vistos.add(str(t.get("registro")))
            elegidas.append(t)
    return elegidas


def _con_articulo_ley(ley: str) -> str:
    """«Código Civil del Estado de Querétaro» → «el Código Civil…»; «Ley de…» → «la Ley de…»."""
    bajo = ley.lower()
    if bajo.startswith(("el ", "la ", "los ", "las ")):
        return ley
    femeninas = ("ley", "constitución", "convención", "declaración", "norma", "carta")
    return ("la " if bajo.startswith(femeninas) else "el ") + ley


# ═══════════════════════════════════════════════════════════════════════════
# 3 · LOS ARGUMENTOS
# ═══════════════════════════════════════════════════════════════════════════
_PROMPT_ARGUMENTOS = """Eres abogado litigante mexicano y construyes los fundamentos de un {tipo} para la parte que promueve, en {entidad}.

HECHOS:
{hechos}

LO QUE SE PIDE:
{pretension}

PROBLEMAS JURÍDICOS QUE HAY QUE GANAR:
{problemas}

{catalogo}

CÓMO SE CONSTRUYE CADA ARGUMENTO (modelo de Toulmin, en el orden en que un tribunal lo lee):
1. Afirmación: la conclusión concreta que se quiere que el juzgador acepte.
2. Datos: los hechos del caso que la sostienen, tomados de los HECHOS, sin inventar ninguno. Si falta un hecho decisivo, dilo en "faltantes".
3. Garantía: la regla jurídica que conecta esos hechos con la afirmación, dicha en abstracto y con su fuente.
4. Respaldo: lo que da autoridad a la garantía: Constitución, tratado, Corte Interamericana, jurisprudencia o tesis. Prefiere la Suprema Corte a un colegiado y la jurisprudencia a la tesis aislada; nunca llames jurisprudencia a una tesis aislada.
5. Calificador: con qué fuerza se sostiene y de qué depende (prueba pendiente, excepción, interpretación discutida).
6. Refutación: la mejor objeción que hará la contraparte o el juzgador, y la respuesta que la vence, con su fuente si la hay.

REGLAS DE CITA, SIN EXCEPCIÓN:
- Solo puedes citar las FUENTES DISPONIBLES, escribiendo su identificador entre corchetes, por ejemplo [T2] o [C1]. No escribas números de registro, rubros, números de artículo ni nombres de casos por tu cuenta: el sistema los pone a partir del identificador.
- Si una idea necesita una fuente que no está en la lista, no la cites: escríbela en "faltantes".
- La ley de otra entidad federativa nunca funda. La jurisprudencia de la Corte Interamericana se cita por su caso y párrafo, que ya van en su identificador.
- Si la Constitución o un tratado no vienen al caso, no los fuerces.
- Cada fuente se cita solo para lo que efectivamente dice su rubro o su texto; si una tesis resuelve otra cosa, no la uses.
- El identificador entre corchetes va al final de la frase que apoya, nunca como parte de la oración: no escribas «en [T1]», «según [T1]», «la tesis [T1]» ni «el artículo [L2]».

"redaccion" es el argumento ya escrito para el escrito, en prosa jurídica mexicana formal, en primera persona del singular de la parte que promueve (una sola persona salvo que los HECHOS digan que promueven varias), sin nombrarla, de 180 a 380 palabras, en uno a tres párrafos separados por una línea en blanco. En la redacción NO aparecen las palabras afirmación, datos, garantía, respaldo, calificador ni refutación: se escribe como abogado, con la cita entre corchetes al final de la frase que apoya. No uses frases hechas de relleno ni transcribas la fuente completa.

Devuelve SOLO un objeto JSON:
{{
  "argumentos": [
    {{
      "titulo": "título breve del argumento, con mayúscula solo en la primera palabra y en los nombres propios",
      "afirmacion": "…",
      "datos": ["…"],
      "garantia": {{"texto": "…", "fuentes": ["C1"]}},
      "respaldo": [{{"fuente": "T1", "como_apoya": "…"}}],
      "calificador": "…",
      "refutacion": {{"objecion": "…", "respuesta": "…", "fuentes": ["T3"]}},
      "redaccion": "…"
    }}
  ],
  "faltantes": ["…"]
}}
Entre 2 y 4 argumentos, uno por problema cuando se pueda."""


async def argumentar(cliente, *, hechos: str, pretension: str, tipo: str, entidad: str,
                     problemas: list[str], bloque: str) -> dict:
    kw = dict(model=MODELO_TOULMIN, max_completion_tokens=24000,
              reasoning_effort=ESFUERZO_TOULMIN,
              response_format={"type": "json_object"},
              messages=[{"role": "user", "content": _PROMPT_ARGUMENTOS.format(
                  tipo=tipo or "demanda", entidad=entidad or "la entidad indicada",
                  hechos=hechos[:MAX_HECHOS], pretension=pretension[:MAX_PRETENSION],
                  problemas="\n".join(f"{i}. {x}" for i, x in enumerate(problemas, 1)),
                  catalogo=bloque)}])
    r = await _lm.crear(cliente, **kw)
    texto = (r.choices[0].message.content or "").strip()
    m = _RX_JSON.search(texto)
    if not m:
        raise ValueError("El modelo no devolvió los argumentos en el formato esperado.")
    return json.loads(m.group(0))


# ═══════════════════════════════════════════════════════════════════════════
# 4 · LA VERIFICACIÓN: CADA CITA SE RESUELVE CONTRA EL CATÁLOGO
# ═══════════════════════════════════════════════════════════════════════════
def _ids_validos(lista, fuentes: dict, fuera: set) -> list[str]:
    out = []
    if isinstance(lista, str):
        lista = [lista]
    for x in (lista or []):
        encontrados = [f"{u.group(1).upper()}{int(u.group(2))}" for u in _RX_UN_ID.finditer(str(x or ""))]
        if not encontrados and str(x or "").strip():
            fuera.add(str(x).strip())
        for k in encontrados:
            if k in fuentes:
                if k not in out:
                    out.append(k)
            else:
                fuera.add(k)
    return out


def _limpiar(x) -> str:
    """Los textos del panel sin corchetes crudos: la cita va en las fichas."""
    t = _RX_RESTO_CORCHETE.sub("", _RX_GRUPO_ID.sub("", str(x or "")))
    return re.sub(r"\s{2,}", " ", t).replace(" .", ".").replace(" ,", ",").strip()


def _cita_corta(f: dict) -> str:
    if f.get("clase") == "tesis":
        return f"registro digital {f['registro']}"
    return f["cita"]


def _redactar(texto: str, fuentes: dict, citadas: list[str], fuera: set) -> str:
    """Sustituye [T2] por la cita entre paréntesis; lo que no existe se quita.

    La primera mención de una tesis lleva instancia, registro y rubro; las
    siguientes, sólo el registro: el rubro entero repetido tres veces en una
    página es ruido, no fundamento.
    """
    ya = set()
    # «en [T1]», «según [T1]», «conforme a [T1]»: la cita como complemento de
    # la oración queda coja al volverse paréntesis. Se quita la preposición.
    texto = re.sub(r"\b(?:en|seg[úu]n|conforme a|de acuerdo con|lo dispuesto en|lo resuelto en)\s+(?=\[\s*[TLCVH]\d{1,2})",
                   "", texto or "", flags=re.I)
    def cambio(m):
        partes = []
        for u in _RX_UN_ID.finditer(m.group(1)):
            k = f"{u.group(1).upper()}{int(u.group(2))}"
            f = fuentes.get(k)
            if not f:
                fuera.add(k)
                continue
            if k not in citadas:
                citadas.append(k)
            partes.append(_cita_corta(f) if k in ya else f["cita"])
            ya.add(k)
        return f"({'; '.join(partes)})" if partes else ""
    t = _RX_GRUPO_ID.sub(cambio, texto or "")
    def resto(m):
        fuera.add(m.group(0))
        return ""
    t = _RX_RESTO_CORCHETE.sub(resto, t)
    t = re.sub(r"\s+\(", " (", t)
    t = re.sub(r"\)\s*\(", "; ", t)          # dos citas seguidas, un solo paréntesis
    t = re.sub(r"[ \t]+([.,;:])", r"\1", t)
    # Al descartar una cita inexistente queda «… y .»: se quita la conjunción huérfana.
    t = re.sub(r"\s+(?:y|e|o|u|así como)\s*([.,;:])", r"\1", t)
    return t.strip()


def resolver(salida: dict, fuentes: dict) -> dict:
    fuera: set = set()
    registros_catalogo = {f["registro"] for f in fuentes.values() if f.get("clase") == "tesis"}
    registros_sueltos: set = set()
    argumentos = []
    for a in (salida.get("argumentos") or [])[:4]:
        if not isinstance(a, dict):
            continue
        citadas: list[str] = []
        garantia = a.get("garantia") if isinstance(a.get("garantia"), dict) else {"texto": str(a.get("garantia") or "")}
        refut = a.get("refutacion") if isinstance(a.get("refutacion"), dict) else {"respuesta": str(a.get("refutacion") or "")}
        respaldo = []
        for r_ in (a.get("respaldo") or []):
            if not isinstance(r_, dict):
                continue
            k = _ids_validos([r_.get("fuente")], fuentes, fuera)
            if k:
                respaldo.append({"fuente": k[0], "como_apoya": _limpiar(r_.get("como_apoya"))})
                if k[0] not in citadas:
                    citadas.append(k[0])
        g_fuentes = _ids_validos(garantia.get("fuentes"), fuentes, fuera)
        r_fuentes = _ids_validos(refut.get("fuentes"), fuentes, fuera)
        for k in g_fuentes + r_fuentes:
            if k not in citadas:
                citadas.append(k)
        redaccion = _redactar(str(a.get("redaccion") or ""), fuentes, citadas, fuera)

        # Un número de registro escrito a mano en la prosa, que no está en el
        # catálogo, es exactamente la cita inventada que este módulo evita.
        prosa = " ".join(str(x or "") for x in (
            a.get("redaccion"), a.get("afirmacion"), garantia.get("texto"),
            refut.get("objecion"), refut.get("respuesta"), a.get("calificador")))
        for m in _RX_REGISTRO.finditer(prosa):
            if m.group(1) not in registros_catalogo:
                registros_sueltos.add(m.group(1))
        argumentos.append({
            "titulo": _limpiar(a.get("titulo")),
            "afirmacion": _limpiar(a.get("afirmacion")),
            "datos": [_limpiar(x) for x in _lista(a.get("datos")) if _limpiar(x)],
            "garantia": {"texto": _limpiar(garantia.get("texto")), "fuentes": g_fuentes},
            "respaldo": respaldo,
            "calificador": _limpiar(a.get("calificador")),
            "refutacion": {"objecion": _limpiar(refut.get("objecion")),
                           "respuesta": _limpiar(refut.get("respuesta")),
                           "fuentes": r_fuentes},
            "redaccion": redaccion,
            "citadas": citadas,
        })
    avisos = []
    if fuera:
        avisos.append(f"Se descartaron {len(fuera)} referencias que no estaban en el acervo encontrado; "
                      f"los argumentos solo citan lo verificado.")
    if registros_sueltos:
        avisos.append("La redacción mencionaba registros que no salieron del acervo ("
                      + ", ".join(sorted(registros_sueltos)) + "). Revísalos antes de firmar.")
    return {"argumentos": argumentos,
            "faltantes": [str(x).strip() for x in _lista(salida.get("faltantes")) if str(x).strip()][:8],
            "avisos": avisos}


# ═══════════════════════════════════════════════════════════════════════════
# LA TUBERÍA
# ═══════════════════════════════════════════════════════════════════════════
async def construir(qdrant, embed_juris, embed_leyes, cliente, *, hechos: str,
                    pretension: str, tipo: str = "demanda", materia: str = "",
                    coleccion_estatal: Optional[str] = None, entidad: str = "",
                    aviso: Aviso = None) -> dict:
    import fase6_rag as f6rag
    import marco_juridico as mj

    async def paso(clave: str, detalle: str = ""):
        if aviso:
            try:
                await aviso(clave, detalle)
            except Exception:
                pass

    avisos: list[str] = []
    await paso("problemas", "Planteando los problemas jurídicos del caso")
    prob = await problemas_del_caso(cliente, hechos, pretension, tipo, materia, entidad)
    problemas = prob["problemas"] or [pretension[:300]]
    materia_caso = prob["materia"] or materia
    # La materia del RAG es selectiva sólo para cuatro silos; el resto va sin filtro.
    materia_rag = {"familiar": "civil", "mercantil": "civil", "administrativo": "administrativa"}.get(
        materia_caso, materia_caso)
    if materia_rag not in ("laboral", "civil", "administrativa", "penal"):
        materia_rag = ""
    coleccion = coleccion_estatal
    if materia_caso == "laboral":
        coleccion = None   # la Ley Federal del Trabajo rige; la ley local no funda
    if not coleccion:
        avisos.append("No se consultó legislación estatal"
                      + (f" de {entidad}" if entidad else "")
                      + ": se fundó con Constitución, tratados, leyes federales y jurisprudencia.")

    await paso("acervo", "Buscando en la Constitución, los tratados, las leyes y la jurisprudencia")
    problemas_marco = list(problemas)
    if prob["convencional"]:
        problemas_marco.append("derechos humanos reconocidos en tratados internacionales")
    try:
        material, marco = await asyncio.wait_for(asyncio.gather(
            # Como problema COMPLETO y no como cadena: `material_del_caso` saca
            # de `combate` el hecho con el que busca la ley del acto en la
            # colección estatal; con la pregunta sola buscaba por el andamio.
            f6rag.material_del_caso(qdrant, embed_juris, embed_leyes,
                                    [{"pregunta": p, "combate": (hechos or "")[:800], "resolvio": ""}
                                     for p in problemas],
                                    coleccion, materia_rag, cliente,
                                    contexto=(hechos or "")[:1500]),
            mj.construir(qdrant, embed_leyes, problemas_marco, coleccion,
                         texto_del_acto=""),
        ), timeout=PLAZO_MATERIAL_SEG)
    except asyncio.TimeoutError:
        raise RuntimeError("El acervo tardó demasiado en responder.")
    for a in (getattr(marco, "avisos", []) or []):
        if "no disparó ningún artículo" in a:
            continue
        if a.startswith("No se pudo vectorizar"):
            print(f"[toulmin] marco: {a[:200]}")
            a = "No se pudo consultar el bloque constitucional en este intento."
        avisos.append(a)

    bloque, fuentes = catalogo(material, marco)
    if not fuentes:
        raise RuntimeError("No se encontró material verificado para estos hechos.")
    conteo = {c: sum(1 for f in fuentes.values() if f["clase"] == c)
              for c in ("constitucion", "tratado", "coidh", "ley", "tesis")}
    await paso("material", json.dumps(conteo))

    await paso("argumentos", "Construyendo los argumentos")
    salida = await argumentar(cliente, hechos=hechos, pretension=pretension, tipo=tipo,
                              entidad=entidad, problemas=problemas, bloque=bloque)

    await paso("verificando", "Verificando cada cita contra el acervo")
    res = resolver(salida, fuentes)
    if not res["argumentos"]:
        raise RuntimeError("No se pudieron construir argumentos con el material encontrado.")
    citadas = []
    for a in res["argumentos"]:
        for k in a["citadas"]:
            if k not in citadas:
                citadas.append(k)
    return {
        "problemas": problemas,
        "materia": materia_caso,
        "argumentos": res["argumentos"],
        "fuentes": fuentes,
        "citadas": citadas,
        "faltantes": list(dict.fromkeys(prob["faltantes"] + res["faltantes"]))[:8],
        "avisos": avisos + res["avisos"],
        "conteo": conteo,
    }
