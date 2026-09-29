# -*- coding: utf-8 -*-
"""LA RECUPERACIÓN POR REQUISITOS (rediseño del taller, punto 2 y etapa 2;
David, 29-sep-2026).

POR QUÉ. David: «Tu RAG ya busca por problema, amplía consultas y añade la
figura decisiva. El siguiente avance consiste en que busque para completar una
demostración identificada. (…) ante una sustitución procesal, la búsqueda debe
responder separadamente: qué derecho debe transmitirse, mediante qué
mecanismos, en qué momento, con qué requisitos y con qué consecuencias. Cada
respuesta puede requerir normas y precedentes distintos.»

QUÉ HACE:
  1. `demostracion` — una llamada barata (esfuerzo bajo) arma, para el
     problema PRINCIPAL, la demostración que hay que completar: el derecho o
     la figura, el mecanismo, el momento, los requisitos (hasta cuatro), las
     excepciones y las consecuencias. Cada requisito trae una consulta en
     forma de rubro, un concepto para buscar la norma (aunque ninguna parte la
     cite) y los preceptos que se creen aplicables (ley y artículo).
  2. `recuperar` — por cada requisito, el camino de búsqueda PROBADO de la
     consulta (`fase6_rag.material_para`: traducción conceptual, rerank,
     silo por materia, cesta del acto), y los preceptos nombrados por fuero
     (`completar_preceptos`). Lo hallado se SUMA al material con cupo propio y
     marcado `para_requisito` (nunca sustituye), y el requisito que se queda
     sin ninguna fuente es un HUECO DECLARADO, no un silencio.
  3. `ficha_de_regla` — cada norma que sostiene un requisito con texto,
     fuente, VERSIÓN (la `ultima_reforma` del catálogo de leyes federales; las
     estatales, «versión sin fecha»: el acervo sólo guarda el texto vigente a
     la fecha de ingesta), ámbito (fuero y entidad), y los requisitos,
     excepciones y consecuencia que la demostración le atribuye.

NUNCA BLOQUEA: con tope de tiempo; si falla, se propone con el material de
siempre (regla de David). Los precedentes OAJ no entran: son sentencias de
otros asuntos, no fuente de la regla.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import unicodedata

VERSION = "requisitos-1"
MAX_REQUISITOS = 4
CUPO_TESIS = 3
CUPO_NORMAS = 3
PRECEPTOS_TOPE_S = float(os.getenv("REQUISITOS_PRECEPTOS_TOPE_S", "45"))
MAX_TOKENS = int(os.getenv("REQUISITOS_MAX_TOKENS", "6000"))
_RX_JSON = re.compile(r"\{.*\}", re.S)
_CATALOGO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts",
                         "leyes_federales_catalogo.json")


def _txt(x, n: int = 0) -> str:
    s = " ".join(str(x or "").split())
    return s[:n] if n else s


def _plano(t) -> str:
    t = unicodedata.normalize("NFD", str(t or "").lower())
    return " ".join("".join(ch for ch in t if unicodedata.category(ch) != "Mn").split())


# ═══ LA DEMOSTRACIÓN ═════════════════════════════════════════════════════════

def prompt(principal: dict, analisis: dict | None, decisiva: dict | None, ficha: str,
           tipo_asunto: str = "") -> str:
    a = analisis or {}
    d = decisiva or {}
    razones = "\n".join(f"  {r.get('id')}: {_txt(r.get('afirma'), 300)} → {_txt(r.get('conclusion'), 200)}"
                        for r in (a.get("razones") or [])[:8]) or "  (sin análisis)"
    return f"""Eres secretario de estudio y cuenta de un Tribunal Colegiado ({tipo_asunto or 'sin tipo'}).
Antes de buscar jurisprudencia y leyes, arma la DEMOSTRACIÓN que hay que completar para
resolver el problema principal. No decidas el sentido: di qué hay que probar.

PROBLEMA PRINCIPAL: {_txt(principal.get('pregunta'), 600)}
LO QUE RESOLVIÓ LA RECURRIDA: {_txt(principal.get('resolvio'), 600)}
LO QUE SE COMBATE: {_txt(principal.get('combate'), 600)}
CUESTIÓN CENTRAL (análisis neutral): {_txt(a.get('cuestion_central'), 400) or '(no consta)'}
FIGURA DECISIVA: {_txt(d.get('figura'), 200) or '(no consta)'}
PREGUNTA DECISIVA: {_txt(d.get('pregunta_decisiva'), 400) or '(no consta)'}
RAZONES DE LO RESUELTO:
{razones}
FICHA PROCESAL:
{ficha or '(sin ficha)'}

Responde SÓLO con un JSON con estas claves:
  figura: el derecho o la institución que se discute, con su nombre técnico;
  mecanismo: cómo opera (la vía o el acto jurídico por el que se produce el efecto);
  momento: cuándo debe ocurrir o hasta cuándo puede ocurrir, si importa;
  requisitos: hasta {MAX_REQUISITOS}, los que de verdad deciden. Cada uno:
     id ("Q1"…), enunciado (qué debe cumplirse),
     consulta_rubro (cómo lo diría el rubro de una tesis que lo resuelva),
     concepto_norma (el concepto jurídico con que se buscaría la norma, aunque nadie la cite),
     preceptos: lista de {{"ley": nombre oficial, "articulo": número}} que crees aplicables ([] si no sabes);
  excepciones: lista de enunciados;
  consecuencias: lista de enunciados (qué se sigue si el requisito falta o se cumple).
No inventes preceptos: si no conoces el artículo, deja la lista vacía."""


def _normalizar(d: dict) -> dict | None:
    if not isinstance(d, dict):
        return None
    reqs = []
    for i, q in enumerate(d.get("requisitos") or [], 1):
        if len(reqs) >= MAX_REQUISITOS:
            break
        if not isinstance(q, dict) or not _txt(q.get("enunciado")):
            continue
        prec = []
        for p in (q.get("preceptos") or [])[:4]:
            if isinstance(p, dict) and _txt(p.get("ley")) and re.match(r"^\d{1,4}", _txt(p.get("articulo"))):
                prec.append({"ley": _txt(p.get("ley"), 200), "articulo": re.match(r"^\d{1,4}", _txt(p.get("articulo"))).group(0)})
        reqs.append({"id": _txt(q.get("id")) or f"Q{i}", "enunciado": _txt(q.get("enunciado"), 400),
                     "consulta_rubro": _txt(q.get("consulta_rubro"), 300) or _txt(q.get("enunciado"), 300),
                     "concepto_norma": _txt(q.get("concepto_norma"), 200), "preceptos": prec})
    if not reqs:
        return None
    return {"version": VERSION, "figura": _txt(d.get("figura"), 200), "mecanismo": _txt(d.get("mecanismo"), 400),
            "momento": _txt(d.get("momento"), 300), "requisitos": reqs,
            "excepciones": [_txt(x, 300) for x in (d.get("excepciones") or [])[:6] if _txt(x)],
            "consecuencias": [_txt(x, 300) for x in (d.get("consecuencias") or [])[:6] if _txt(x)]}


async def demostracion(cliente, principal: dict, analisis=None, decisiva=None, ficha: str = "",
                       tipo_asunto: str = "") -> dict | None:
    """La demostración del principal, o None (nunca lanza)."""
    try:
        import fase5_propuesta as _f5
        import llamada_modelo as _lm
        kw = dict(model=os.getenv("MODELO_REQUISITOS", "") or _f5.MODELO_PROPUESTA,
                  temperature=0, seed=20260929, max_completion_tokens=MAX_TOKENS,
                  reasoning_effort=os.getenv("ESFUERZO_REQUISITOS", "low"),
                  messages=[{"role": "user", "content": prompt(principal, analisis, decisiva, ficha, tipo_asunto)}])
        r = await _lm.crear(cliente, **kw)
        crudo = (r.choices[0].message.content or "").strip()
        m = _RX_JSON.search(crudo or "")
        return _normalizar(json.loads(m.group(0))) if m else None
    except Exception as ex:
        print(f"   🧩 REQUISITOS: la demostración falló ({type(ex).__name__}); se sigue sin ella")
        return None


# ═══ LA RECUPERACIÓN, REQUISITO POR REQUISITO ═══════════════════════════════

# Los códigos que, nombrados sin entidad, son del estado: el federal lleva
# «Federal» en el nombre (Código Civil Federal, Código Penal Federal, Código
# Federal de Procedimientos Civiles). «Ley Agraria» o «Ley de Amparo» no
# dicen fuero y son federales: no entran aquí.
_LOCALES_SIN_ENTIDAD = ("codigo civil", "codigo de procedimientos civiles", "codigo penal",
                        "codigo de procedimientos penales", "codigo familiar",
                        "codigo de procedimientos familiares", "codigo urbano")


# Leyes FEDERALES cuyo nombre dice «del Estado» (el Estado mexicano, no una
# entidad): la del ISSSTE y la burocrática.
_FEDERALES_DEL_ESTADO = ("trabajadores del estado", "servicio del estado", "servicios sociales de los trabajadores")


def rige_ley_local(dem: dict | None) -> bool:
    """¿Nombra la demostración alguna ley LOCAL? Por el fuero que dice su
    nombre, o por ser uno de los códigos que sin entidad son del estado."""
    import fase6_rag as _f6r
    for q in (dem or {}).get("requisitos") or []:
        for p in q.get("preceptos") or []:
            ley = str((p or {}).get("ley") or "")
            f = _f6r.fuero_de(ley)
            if any(k in _plano(ley) for k in _FEDERALES_DEL_ESTADO):
                continue
            if f == "estatal" or (not f and _plano(ley).startswith(_LOCALES_SIN_ENTIDAD)):
                return True
    return False


def _por_leyes_nombradas(normas: list, q: dict) -> list:
    """Primero las normas de las leyes que el requisito nombra; después el
    resto, en el orden de la búsqueda."""
    import fase6_rag as _f6r
    leyes = [str(p.get("ley") or "") for p in q.get("preceptos") or [] if isinstance(p, dict)]
    def _nombrada(n):
        return any(_f6r.misma_ley(l, n.get("cuerpo_legal") or "") for l in leyes if l)
    normas = [n for n in normas if isinstance(n, dict)]
    return [n for n in normas if _nombrada(n)] + [n for n in normas if not _nombrada(n)]


async def recuperar(qdrant, embed_juris, embed_leyes, material, dem: dict, *,
                    coleccion_estatal: str = "", materia: str = "", tipo_asunto: str = "",
                    cliente=None, sede_acto: str = "", cuaderno: str = "") -> dict:
    """Busca por cada requisito y SUMA al material, marcado. Devuelve la
    demostración con, por requisito, sus `tesis` y `normas` (registros y
    etiquetas) y la lista de `huecos`. Nunca lanza."""
    import fase6_rag as _f6r
    import contexto_taller as _ct
    tengo_t = {str(t.get("registro") or "") for t in (material.tesis or []) if isinstance(t, dict)}
    tengo_n = {(_plano(n.get("cuerpo_legal")), str(n.get("articulo"))) for n in (material.normas or [])
               if isinstance(n, dict)}
    reqs = [q for q in (dem.get("requisitos") or []) if isinstance(q, dict)]

    # LA LEY DEL ESTADO SÓLO SI RIGE, Y POR REQUISITO. `material_para` abre la
    # «cesta del acto» con la colección de la entidad y la pone PRIMERO;
    # medido en el 103/2025 (sucesión agraria, ley federal): los tres cupos de
    # cada requisito se los llevaron la Ley de Catastro, la de Adolescentes y
    # el procesal civil de Querétaro. Cada requisito nombra las leyes en que
    # descansa: la cesta sólo se abre para el que nombra una local (un precepto
    # local en Q1 no la abre para Q2-Q4).
    def _col(q) -> str | None:
        return (coleccion_estatal or None) if rige_ley_local({"requisitos": [q]}) else None

    async def _uno(q):
        try:
            return await _f6r.material_para(qdrant, embed_juris, embed_leyes, q["consulta_rubro"],
                                            _col(q), materia=materia, cliente=cliente,
                                            hecho=q.get("concepto_norma") or "", sede_acto=sede_acto,
                                            cuaderno=cuaderno)
        except Exception as ex:
            print(f"   🧩 REQUISITOS: {q.get('id')} sin búsqueda ({type(ex).__name__})")
            return None

    if coleccion_estatal and not any(_col(q) for q in reqs):
        print("   🧩 REQUISITOS: ningún requisito nombra ley local; se busca sin la del estado")
    mats = await asyncio.gather(*[_uno(q) for q in reqs])
    nuevas_t, nuevas_n = [], []
    for q, m in zip(reqs, mats):
        q["tesis"], q["normas"] = [], []
        if m is not None:
            for t in _ct.filtrar_tesis(list(m.tesis or []))[:CUPO_TESIS]:
                reg = str(t.get("registro") or "")
                q["tesis"].append(reg)
                if reg and reg not in tengo_t:
                    tengo_t.add(reg)
                    nuevas_t.append(dict(t, para_requisito=q["id"], origen="requisito", cupo_figura=True))
            for n in _por_leyes_nombradas(list(m.normas or []), q)[:CUPO_NORMAS]:
                k = (_plano(n.get("cuerpo_legal")), str(n.get("articulo")))
                q["normas"].append(f"art. {n.get('articulo')} — {n.get('cuerpo_legal')}")
                if k not in tengo_n:
                    tengo_n.add(k)
                    nuevas_n.append(dict(n, para_requisito=q["id"], origen="requisito"))
    # Lo hallado entra ANTES de traer los preceptos nombrados: así el
    # resolvedor ve lo que ya está y no trae dos veces el mismo artículo.
    material.tesis = list(material.tesis or []) + nuevas_t
    material.normas = list(material.normas or []) + nuevas_n
    antes_n = len(material.normas)

    # LOS PRECEPTOS NOMBRADOS, CON LAS MISMAS GUARDAS QUE LOS CITADOS
    # (`preceptos_fuera`): las leyes notorias no se traen, los artículos de
    # la suspensión del amparo no entran cuando el acto es de sede ordinaria
    # (322/2025) y lo que ya está en el material no se repite. El fuero lo
    # decide el nombre: la colección del estado sólo si alguno es local.
    import fase6_estudio as _f6e
    texto = "; ".join(f"artículo {p['articulo']} de la {p['ley']}"
                      for q in reqs for p in q.get("preceptos") or [] if isinstance(p, dict))
    pares = []
    if texto:
        try:
            pares = sorted(_f6e.preceptos_fuera(texto, material)[1])
        except Exception as ex:
            print(f"   🧩 REQUISITOS: preceptos nombrados sin filtrar ({type(ex).__name__})")
    if pares and qdrant is not None:
        col_nombrados = coleccion_estatal if any(_col(q) for q in reqs) else None
        try:
            # Tope propio: el resolvedor puede ir a internet (≈27 s por
            # artículo) y lo hallado arriba no debe perderse por eso.
            await asyncio.wait_for(
                _f6r.completar_preceptos(qdrant, material, pares, col_nombrados,
                                         materia=materia, tipo_asunto=tipo_asunto),
                timeout=PRECEPTOS_TOPE_S)
        except asyncio.TimeoutError:
            print(f"   🧩 REQUISITOS: los preceptos nombrados no llegaron en {PRECEPTOS_TOPE_S:.0f} s")
        except Exception as ex:
            print(f"   🧩 REQUISITOS: preceptos nombrados sin traer ({type(ex).__name__})")
    nombradas = []
    for i, n in enumerate(material.normas or []):
        if not isinstance(n, dict):
            continue
        if i >= antes_n:
            n.setdefault("origen", "requisito")
        for q in reqs:
            if any(isinstance(p, dict) and str(p.get("articulo")) == str(n.get("articulo"))
                   and _f6r.misma_ley(str(p.get("ley") or ""), n.get("cuerpo_legal") or "")
                   for p in q.get("preceptos") or []):
                # El precepto NOMBRADO que ya estaba en el material (de la
                # consulta) también sostiene el requisito: se rotula, y va
                # delante de lo hallado por parecido.
                if i >= antes_n:
                    n.setdefault("para_requisito", q["id"])
                et = f"art. {n.get('articulo')} — {n.get('cuerpo_legal')}"
                if et in q["normas"]:
                    q["normas"].remove(et)
                q["normas"].insert(len(q.setdefault("_nombradas", [])), et)
                q["_nombradas"].append(et)
                nombradas.append((n, q["id"]))
    for q in reqs:
        q.pop("_nombradas", None)
    dem["huecos"] = [q["id"] for q in reqs if not (q.get("tesis") or q.get("normas"))]
    vistas, fichas = set(), []
    for n, qid in nombradas + [(n, n.get("para_requisito")) for n in nuevas_n]:
        k = (_plano(n.get("cuerpo_legal")), str(n.get("articulo")))
        if k not in vistas:
            vistas.add(k)
            fichas.append(ficha_de_regla(dict(n, para_requisito=qid), dem))
    dem["fichas"] = fichas[:12]
    print(f"   🧩 REQUISITOS: {len(reqs)} requisito(s) · {len(nuevas_t)} tesis y "
          f"{len(nuevas_n) + len(material.normas) - antes_n} normas sumadas · {len(dem['huecos'])} hueco(s)")
    return dem


# ═══ LA FICHA DE LA REGLA ════════════════════════════════════════════════════

_CAT: dict | None = None


def _catalogo() -> dict:
    global _CAT
    if _CAT is None:
        try:
            _CAT = {}
            for x in json.load(open(_CATALOGO, encoding="utf-8")):
                for k in (x.get("ley"), x.get("abrev")):
                    if k:
                        _CAT[_plano(k)] = x
        except Exception:
            _CAT = {}
    return _CAT


def version_de(cuerpo_legal: str) -> str:
    """La versión que se puede afirmar: la última reforma del catálogo federal
    (DOF), o «versión sin fecha» (el acervo guarda el texto vigente a la
    fecha de ingesta y las fuentes estatales no traen fecha de reforma)."""
    x = _catalogo().get(_plano(cuerpo_legal))
    if x:
        ur = _txt(x.get("ultima_reforma"))
        return (f"texto vigente con la última reforma {ur}" if ur and ur.lower() != "sin reforma"
                else f"texto original {x.get('dof_original') or ''}".strip())
    return "versión sin fecha"


def _entidad_por_nombre(cuerpo_legal) -> str:
    """Sin `entidad` en la norma (las traídas de internet no la llevan): lo
    que dice su nombre. Nunca «FEDERAL» para un código del estado."""
    import fase6_rag as _f6r
    nombre = str(cuerpo_legal or "")
    if any(k in _plano(nombre) for k in _FEDERALES_DEL_ESTADO):
        return "FEDERAL"
    f = _f6r.fuero_de(nombre)
    if f == "estatal":
        m = re.search(r"(?i)\bdel estado de ([\wáéíóúñ ]+?)(?:$|[,.;(])", nombre)
        return (m.group(1).strip().upper() if m else "ESTATAL")
    return "FEDERAL" if f == "federal" else "SIN FUERO EN EL NOMBRE"


def ficha_de_regla(n: dict, dem: dict | None = None) -> dict:
    """texto + fuente + versión + ámbito + requisitos + excepciones + consecuencia."""
    q = next((x for x in (dem or {}).get("requisitos") or [] if x.get("id") == n.get("para_requisito")), {})
    return {"norma": f"art. {n.get('articulo')} — {n.get('cuerpo_legal')}",
            "texto": _txt(n.get("texto"), 1500),
            "fuente": _txt(n.get("url_pdf")) or ("fuente oficial en línea" if n.get("de_internet") else "acervo"),
            "version": version_de(n.get("cuerpo_legal") or ""),
            "ambito": {"entidad": _txt(n.get("entidad")) or _entidad_por_nombre(n.get("cuerpo_legal")),
                       "ubicacion": _txt(n.get("jerarquia") or n.get("capitulo"))},
            "requisito": q.get("id") or "", "enunciado": q.get("enunciado") or "",
            "excepciones": list((dem or {}).get("excepciones") or [])[:3],
            "consecuencias": list((dem or {}).get("consecuencias") or [])[:3]}


def bloque_propuesta(dem: dict | None, material=None) -> str:
    """La demostración como DATOS para la propuesta. «» sin ella. Con el
    `material`, sólo se nombran las fuentes que de verdad están en él (al
    guardar hay topes, y en el otro worker puede faltar lo que aquí se sumó)."""
    if not isinstance(dem, dict) or not dem.get("requisitos"):
        return ""
    presentes_t = presentes_n = None
    if material is not None:
        presentes_t = {str(t.get("registro") or "") for t in (getattr(material, "tesis", None) or [])
                       if isinstance(t, dict)}
        presentes_n = {f"art. {n.get('articulo')} — {n.get('cuerpo_legal')}"
                       for n in (getattr(material, "normas", None) or []) if isinstance(n, dict)}
    L = ["", "LA DEMOSTRACIÓN QUE HAY QUE COMPLETAR (por requisitos; las fuentes nombradas están en el material):"]
    for k, e in (("FIGURA", "figura"), ("MECANISMO", "mecanismo"), ("MOMENTO", "momento")):
        if dem.get(e):
            L.append(f"  {k}: {_txt(dem[e])}")
    for q in dem["requisitos"]:
        if not isinstance(q, dict):
            continue
        ts = [x for x in (q.get("tesis") or []) if presentes_t is None or str(x) in presentes_t]
        ns = [x for x in (q.get("normas") or []) if presentes_n is None or x in presentes_n]
        fuentes = ", ".join([str(x) for x in ts[:3]] + [str(x) for x in ns[:3]])
        L.append(f"  {q.get('id', '')}. {_txt(q.get('enunciado'))} — "
                 + (f"fuentes: {fuentes}" if fuentes else "SIN FUENTE EN EL ACERVO: dilo, no lo supla"))
    if dem.get("excepciones"):
        L.append("  EXCEPCIONES: " + "; ".join(_txt(x) for x in dem["excepciones"]))
    if dem.get("consecuencias"):
        L.append("  CONSECUENCIAS: " + "; ".join(_txt(x) for x in dem["consecuencias"]))
    for f in (dem.get("fichas") or [])[:6]:
        if not isinstance(f, dict) or (presentes_n is not None and f.get("norma") not in presentes_n):
            continue
        L.append(f"  · {f.get('norma')} ({f.get('version')}; {(f.get('ambito') or {}).get('entidad', '')})")
    L.append("  REGLA: la solución tiene que cubrir cada requisito con su fuente o decir que falta; un "
             "requisito sin fuente no se da por cumplido ni por incumplido de memoria.")
    return "\n".join(L) + "\n"
