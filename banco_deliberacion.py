# -*- coding: utf-8 -*-
"""EL BANCO DE LA DELIBERACIÓN — las compuertas, escritas ANTES de correr.

POR QUÉ EXISTE. `deliberacion.py` (28-sep-2026) está detrás de una bandera
apagada y así se queda hasta que esto diga que supera la línea base. Lo
aprendido en este proyecto: el contraste costó una llamada y no movió nada
(50 % → 48 % en Kingston), y parecía obvio. Una etapa que no pasa sus
compuertas no se enciende (lector2_potencia.md §9).

SE NIEGA A GASTAR SIN PERMISO. Regla de la casa (23-sep-2026: un banco de ~42
USD corrido sin preguntar): sin `--si-gastar` imprime lo que va a correr y lo
que costaría, y sale con código 2 sin tocar la red. `--si-gastar` sólo se pasa
con el sí de David, dado para ESA corrida.

LOS TRES BANCOS
  kingston  24 engroses reales de amparo directo con oro sólido
            (`banco_kingston.banco`). Reutiliza las sesiones guardadas de
            administracion@iurexia.com: sólo delibera, no vuelve a leer ni a
            proponer. El brazo «hoy» son las filas «antes» de
            bancos/kingston/resultados.jsonl (sin coste).
  oaj       planteamientos principales del índice de la OAJ (redactor-sentencias/
            oaj/lectura_json), muestra estratificada por tipo (AR, AD) y por
            oro (prospera / no prospera), con semilla. ENTRADA DERIVADA: la
            pregunta, lo resuelto y lo combatido los escribió un modelo leyendo
            la sentencia; nunca se mezcla en un número publicable. Se excluye
            del espejo la MISMA sentencia (por expediente y por NEUN): si no,
            hay fuga. El principal es el primer planteamiento, la regla del
            taller cuando no hay jerarquía. «No se estudió» no entra: es oro de
            la suerte de los secundarios, no del principal.
  631       la sesión real (soporte@iurexia.com, fila 462), SIN ESCRIBIR. Debe
            recomendar revocar, anunciar el estudio de los conceptos omitidos,
            dar como vía B el razonamiento del juzgado, 0 registros fuera del
            acervo y los secundarios resueltos solos.

LO QUE SE REPORTA, siempre con la línea base impresa al lado: exactitud,
EXACTITUD BALANCEADA y MCC (en la OAJ «no prospera» es el 74-75 %: la
exactitud sola engaña), matriz de confusión, intervalo de Wilson (con n=24 son
±20 puntos: una puerta de cordura, no una prueba), la tasa de acierto de lo
marcado «claro» contra lo «reñido», y la tasa de «otra vía sostenible».

Uso (sin --si-gastar no corre nada; sólo presupuesta):
    .venv/bin/python banco_deliberacion.py --banco kingston
    .venv/bin/python banco_deliberacion.py --banco oaj --n-oaj 100 --con-base
    .venv/bin/python banco_deliberacion.py --banco 631
    .venv/bin/python banco_deliberacion.py --banco todos --si-gastar     # con permiso
    .venv/bin/python banco_deliberacion.py --comparar
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import glob
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

# En el checkout principal, como Kingston (revisión del 29-sep): los resultados
# pagados no pueden morir con el worktree.
AQUI = Path("/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/jurexia-api-git/bancos/deliberacion")
RESULTADOS = AQUI / "resultados.jsonl"
OAJ_DIR = "/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/redactor-sentencias/oaj/lectura_json"
# En el checkout principal (29-sep-2026): desde un worktree la ruta relativa no
# existía y el brazo «hoy» salía vacío sin avisar.
KINGSTON_RESULTADOS = Path("/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/jurexia-api-git/bancos/kingston/resultados.jsonl")
INDICE_OAJ = "/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/redactor-sentencias/oaj/indice_oaj_v2.sqlite"
# Las banderas del banco OAJ: por omisión, las de producción para un secretario
# de fuera (apagadas). `--banderas` las cambia para medir un cambio.
BANDERAS_OAJ = {"fuerza_unificada": False, "fuente_tardia_aviso": False, "normas_al_documento": False,
                "tesis_parte_al_consultar": False, "consulta_provisional": False}


def _fecha_de_neun(neun) -> str:
    """La fecha de la sentencia (dd-mm-aaaa) según el índice OAJ; «» si no."""
    try:
        import sqlite3
        db = sqlite3.connect(f"file:{INDICE_OAJ}?mode=ro&immutable=1", uri=True)
        r = db.execute("SELECT fecha_sentencia FROM asuntos WHERE neun = ? LIMIT 1", (int(neun),)).fetchone()
        return str(r[0]) if r and r[0] else ""
    except Exception:
        return ""
CORREO_KINGSTON = "administracion@iurexia.com"
CORREO_631, NUMERO_631 = "soporte@iurexia.com", "631/2025"
TRIBUNAL_ESPEJO = ("Tercer Tribunal Colegiado en Materias Administrativa y Civil del "
                   "Vigésimo Segundo Circuito")

# ═══ LAS COMPUERTAS — escritas antes de correr, y no se mueven después ══════
COMPUERTAS = (
    ("oaj", "OAJ: la deliberación mejora ≥5 puntos de exactitud balanceada sobre la propuesta "
            "de hoy, con un intervalo que no cruce 0."),
    ("kingston", "Kingston: no empeora la exactitud y baja el cruce niega→concede (hoy 9)."),
    ("citas", "0 citas fuera del acervo y 0 tesis sin vigencia sin su sello."),
    ("claro", "«claro» sólo se enseña si acierta ≥80 % en el banco."),
    ("631", "631: recomienda revocar, anuncia los conceptos omitidos, la vía B es el "
            "razonamiento del juzgado, 0 registros fuera del acervo y secundarios solos."),
)
MEJORA_MINIMA_OAJ = 0.05
NIEGA_CONCEDE_HOY = 9
ACIERTO_MINIMO_CLARO = 0.80

# ═══ LO QUE CUESTA — estimado, confirmar con una o dos llamadas ═════════════
# Tokens por asunto y etapa (entrada, salida), a los precios de gpt-5.6-luna
# (0.20 / 1.20 USD por millón, `inventario_escrito`). La salida incluye el
# razonamiento, que se cobra como salida.
TOKENS_POR_ETAPA = {
    "pregunta decisiva": (20000, 2500),
    "búsqueda (conceptual + rerank)": (6000, 2500),
    "refuerzo por vía (×2)": (24000, 5000),
    "lectura de candidatas": (12000, 3000),
    "abogados (×2, esfuerzo alto)": (80000, 24000),
    "juez (×2, esfuerzo alto)": (50000, 16000),
}
TOKENS_BASE_OAJ = {"propuesta de hoy + contraste": (50000, 22000)}
PRECIO_ENTRADA, PRECIO_SALIDA = 0.20, 1.20


def _usd(tokens: dict) -> float:
    e = sum(x[0] for x in tokens.values())
    s = sum(x[1] for x in tokens.values())
    return (e * PRECIO_ENTRADA + s * PRECIO_SALIDA) / 1e6


def coste_estimado(bancos: list, n_oaj: int, con_base: bool) -> dict:
    """{banco: (asuntos, USD)} y el total. Sin red."""
    por = _usd(TOKENS_POR_ETAPA)
    out = {}
    for b in bancos:
        if b == "kingston":
            out[b] = (24, 24 * por)
        elif b == "oaj":
            out[b] = (n_oaj, n_oaj * (por + (_usd(TOKENS_BASE_OAJ) if con_base else 0.0)))
        elif b == "631":
            out[b] = (1, por)
    out["total"] = (sum(v[0] for v in out.values()), sum(v[1] for v in out.values()))
    return out


# ═══ LAS MÉTRICAS — puras, probadas sin red ══════════════════════════════════

def matriz(pares: list) -> dict:
    """{(oro, predicho): n} con True = prospera. Los None (indeterminados)
    cuentan como error del predicho: no decidir no es acertar."""
    return dict(collections.Counter((o, p) for o, p in pares))


def exactitud(pares: list) -> float:
    return sum(1 for o, p in pares if p is not None and o == p) / max(len(pares), 1)


def exactitud_balanceada(pares: list) -> float:
    def recall(clase):
        xs = [(o, p) for o, p in pares if o is clase]
        return sum(1 for o, p in xs if p is clase) / len(xs) if xs else float("nan")
    r = [x for x in (recall(True), recall(False)) if not math.isnan(x)]
    return sum(r) / len(r) if r else float("nan")


def mcc(pares: list) -> float:
    tp = sum(1 for o, p in pares if o is True and p is True)
    tn = sum(1 for o, p in pares if o is False and p is False)
    fp = sum(1 for o, p in pares if o is False and p is True)
    fn = sum(1 for o, p in pares if o is True and p is not True)
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d else 0.0


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - m), min(1.0, c + m))


def oro_de_calificacion(cal: str):
    """True si prospera, False si no, None si no decide nada del principal
    («no se estudió», vacío, «sin materia»)."""
    c = " ".join(str(cal or "").lower().split())
    if not c or "no se estudi" in c or "sin materia" in c or "innecesari" in c:
        return None
    try:
        import tipos_asunto as _ta
        return bool(_ta.prospera(c.replace(" ", "_")))
    except Exception:                                   # pragma: no cover
        return "fundad" in c and "inoperan" not in c and "insuficien" not in c


def _numero_y_neun(archivo: str) -> tuple:
    """«AR_325-2024_12345678.json» → («AR», «325/2024», «12345678»)."""
    b = os.path.basename(archivo).rsplit(".", 1)[0]
    m = re.match(r"([A-Z]+)_(\d+)-(\d{4})_(\d+)", b)
    if not m:
        return "", "", ""
    return m.group(1), f"{m.group(2)}/{m.group(3)}", m.group(4)


def muestra_oaj(directorio: str = OAJ_DIR, n_por_tipo: int = 200, semilla: int = 20260928,
                tipos: tuple = ("AR", "AD")) -> list:
    """La muestra estratificada: por tipo, mitad prospera y mitad no prospera,
    con semilla. Determinista. Sólo el primer planteamiento de cada sentencia."""
    rng = random.Random(semilla)
    fuera = []
    for tipo in tipos:
        archivos = sorted(glob.glob(os.path.join(directorio, f"{tipo}_*.json")))
        rng.shuffle(archivos)
        cupo = {True: n_por_tipo // 2, False: n_por_tipo - n_por_tipo // 2}
        for f in archivos:
            if not any(cupo.values()):
                break
            try:
                d = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            ps = [p for p in (d.get("planteamientos") or []) if isinstance(p, dict)]
            if not ps or not str(ps[0].get("pregunta") or "").strip():
                continue
            oro = oro_de_calificacion(ps[0].get("calificacion"))
            if oro is None or cupo[oro] <= 0:
                continue
            cupo[oro] -= 1
            t, num, neun = _numero_y_neun(f)
            fuera.append({"archivo": os.path.basename(f), "tipo": t, "numero": num, "neun": neun,
                          "oro": oro, "principal": {k: ps[0].get(k) for k in
                                                    ("pregunta", "resolvio", "combate")},
                          "acto": str(d.get("acto") or ""),
                          "autoridad": str(d.get("autoridad") or ""),
                          # OCULTOS para el modelo: sólo para medir.
                          "calificacion_oro": ps[0].get("calificacion"),
                          "razon_oro": ps[0].get("razon")})
    return fuera


def _tipo_de_fila(f: dict) -> str:
    """«AR» | «AD» | «Q» | «RF» | «» de una fila del espejo, por su tipo o por la
    sigla con que abre el expediente. Por PALABRAS: «ar» dentro de «amparo» no
    es una sigla (así se colaba el amparo directo del mismo número)."""
    import unicodedata
    def _p(x):
        x = unicodedata.normalize("NFKD", str(x or "").lower())
        return " ".join("".join(c for c in x if not unicodedata.combining(c)).replace(".", "").split())
    ta, exp = _p(f.get("tipo_asunto")), _p(f.get("expediente"))
    if "revision fiscal" in ta:
        return "RF"
    if "revision" in ta:
        return "AR"
    if "directo" in ta:
        return "AD"
    if "queja" in ta:
        return "Q"
    for sig in ((ta.split() or [""])[0], (exp.split() or [""])[0]):
        if sig in ("ar", "ad", "rf", "q", "ara", "adc", "ada", "adl", "arc"):
            return {"ara": "AR", "arc": "AR", "adc": "AD", "ada": "AD", "adl": "AD"}.get(sig, sig.upper())
    return ""


def sin_fuga(filas: list, caso: dict) -> list:
    """El espejo sin la MISMA sentencia: por NEUN si la fila lo trae (o su
    enlace), y por tipo y número de expediente. Si no, el modelo ve la
    respuesta. Con el tipo de la fila desconocido, el mismo número basta para
    excluirla: en la duda, fuera."""
    num = str(caso.get("numero") or "").strip()
    neun = str(caso.get("neun") or "").strip()
    tipo = str(caso.get("tipo") or "").strip().upper()
    fuera = []
    for f in filas or []:
        if neun and (str(f.get("neun") or "") == neun or neun in str(f.get("pdf_url") or "")):
            continue
        exp = " ".join(str(f.get("expediente") or "").split())
        mismo_num = bool(num) and re.search(rf"(?<!\d){re.escape(num)}(?!\d)", exp)
        t_f = _tipo_de_fila(f)
        if mismo_num and (not t_f or not tipo or t_f == tipo):
            continue
        fuera.append(f)
    return fuera


def veredicto(delib: dict) -> tuple:
    """(prospera: True/False/None, estado). Lo recomendado si es «claro»; si
    no, la vía en que coincidieron las dos pasadas; si no, None."""
    via = delib.get("recomendada") or delib.get("inclinacion")
    if via not in ("A", "B"):
        return None, delib.get("estado")
    return bool((delib.get("vias") or {}).get(via, {}).get("prospera")), delib.get("estado")


def _textos_de_via(v: dict) -> list:
    cad = v.get("cadena") or {}
    ob = v.get("objecion") or {}
    return ([v.get("razon"), v.get("interpretacion"), cad.get("regla"), cad.get("subsuncion"),
             cad.get("conclusion"), ob.get("de_la_otra_via"), ob.get("respuesta")]
            + [h.get("afirma") for h in cad.get("hechos") or []]
            + [x.get("distincion") for x in v.get("autoridad_contraria") or []])


def citas_fuera(delib: dict) -> int:
    """LA COMPUERTA 3, COMPROBADA DESPUÉS Y POR SU CUENTA: cuántos registros
    llegan a lo que se enseña sin estar en el catálogo —cifras de seis o siete
    dígitos en el texto de las vías o del juez, y apoyos sin `en_acervo`—, y
    cuántas tesis sin vigencia se proponen sin su sello. Debe ser 0 por
    construcción; si no lo es, la verificación E tiene un agujero."""
    validos = {str(e.get("registro")) for e in (delib.get("catalogo") or {}).values()
               if e.get("clase") == "tesis"}
    n = 0
    textos = list(delib.get("por_que") or []) + [delib.get("debilidad")]
    textos += [(delib.get("crux") or {}).get(k) for k in ("que", "si_cambia", "constancia")]
    for v in (delib.get("vias") or {}).values():
        textos += _textos_de_via(v)
        for x in v.get("apoyos") or []:
            if not x.get("en_acervo") or (x.get("registro") and str(x["registro"]) not in validos):
                n += 1
    for t in textos:
        t = str(t or "")
        for m in re.finditer(r"(?<![\d.,/$])\b(\d{6,7})\b(?![\d/])", t):
            # La misma excepción que la verificación: una cantidad con signo
            # de pesos no es un registro.
            if m.group(1) not in validos and "$" not in t[max(0, m.start() - 3):m.start()]:
                n += 1
    return n


def citas_quitadas(delib: dict) -> int:
    """Cuántas referencias quitó la verificación (el modelo las escribió y no
    estaban en el catálogo). Informativo: mide cuánto inventa."""
    return int((delib.get("verificacion") or {}).get("referencias_quitadas") or 0)


def informe(nombre: str, filas: list, base_niega: bool = True) -> dict:
    """Las métricas de un banco, con la línea base impresa al lado."""
    pares = [(f["oro"], f.get("predicho")) for f in filas if f.get("oro") is not None]
    n = len(pares)
    k = sum(1 for o, p in pares if o == p)
    base = [(o, False) for o, _ in pares]
    claros = [(f["oro"], f.get("predicho")) for f in filas if f.get("estado") == "claro"]
    r = {"banco": nombre, "n": n, "exactitud": exactitud(pares),
         "exactitud_balanceada": exactitud_balanceada(pares), "mcc": mcc(pares),
         "wilson": wilson(k, n), "matriz": {f"{o}->{p}": c for (o, p), c in matriz(pares).items()},
         "linea_base_no_prospera": {"exactitud": exactitud(base),
                                    "exactitud_balanceada": exactitud_balanceada(base)},
         "claro": {"n": len(claros), "acierto": exactitud(claros) if claros else None,
                   "wilson": wilson(sum(1 for o, p in claros if o == p), len(claros))},
         "otra_sostenible": sum(1 for f in filas if f.get("otra_sostenible")) / max(len(filas), 1),
         "citas_fuera": sum(f.get("citas_fuera", 0) for f in filas)}
    print(f"\n═══ {nombre} · n={n} ═══")
    print(f"  exactitud {r['exactitud']:.2f}  balanceada {r['exactitud_balanceada']:.2f}  "
          f"MCC {r['mcc']:.2f}  Wilson {r['wilson'][0]:.2f}–{r['wilson'][1]:.2f}")
    print(f"  línea base «no prospera»: exactitud {r['linea_base_no_prospera']['exactitud']:.2f}  "
          f"balanceada {r['linea_base_no_prospera']['exactitud_balanceada']:.2f}")
    print(f"  matriz (oro→predicho): {r['matriz']}")
    if claros:
        print(f"  «claro»: {len(claros)} · acierta {r['claro']['acierto']:.2f} "
              f"(compuerta ≥ {ACIERTO_MINIMO_CLARO:.2f})")
    print(f"  citas fuera del acervo: {r['citas_fuera']} (compuerta: 0)")
    return r


def anotar(fila: dict) -> None:
    AQUI.mkdir(parents=True, exist_ok=True)
    with open(RESULTADOS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(fila, ensure_ascii=False, default=str) + "\n")


# ═══ LAS CORRIDAS — sólo con --si-gastar ═════════════════════════════════════
# Importan `main` SIN el arranque (el lifespan borra las cachés de Gemini de
# producción; importar el módulo no entra en él: verificar_631.py lo hace así).
# Leen la base; no escriben en ella.

def _main():
    import main as _m                                   # noqa: WPS433
    return _m


async def _deliberar_sesion(m, correo: str, numero: str) -> dict:
    ses = m._taller_recuperar_sesion(correo, numero)
    if not ses or ses.get("material") is None:
        raise RuntimeError("sin sesión o sin acervo guardado")
    doc = m._taller_leer_marca(correo, numero, "propuesta") or {}
    resp = doc.get("respuesta") if isinstance(doc.get("respuesta"), dict) else {}
    return await m._taller_deliberar_nucleo(ses["resultado"], ses, resp, "")


async def correr_kingston(solo: int = 0) -> list:
    sys.path.insert(0, str(Path(__file__).parent))
    import banco_kingston as bk
    m = _main()
    filas = []
    for caso in bk.banco()[: (solo or None)]:
        numero = bk.numero_de(caso["asunto"])
        fila = {"banco": "kingston", "asunto": caso["asunto"], "numero": numero,
                "oro": bk.sentido_del_oro(caso["oro"]) == "concede",
                "t0": time.strftime("%Y-%m-%d %H:%M:%S")}
        try:
            d = await _deliberar_sesion(m, bk.CORREO, numero)
            fila["predicho"], fila["estado"] = veredicto(d)
            fila.update({"otra_sostenible": d.get("otra_sostenible"),
                         "citas_fuera": citas_fuera(d), "citas_quitadas": citas_quitadas(d),
                         "uso": d.get("uso"),
                         "recomendada": d.get("recomendada"), "inclinacion": d.get("inclinacion")})
        except Exception as e:
            fila["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        anotar(fila)
        filas.append(fila)
        print(f"  {'✓' if fila.get('predicho') == fila['oro'] else '✗'} {caso['asunto'][:44]:<44} "
              f"{fila.get('estado') or fila.get('error', '')}", flush=True)
    return filas


async def correr_631() -> dict:
    m = _main()
    d = await _deliberar_sesion(m, CORREO_631, NUMERO_631)
    va, vb = (d.get("vias") or {}).get("A") or {}, (d.get("vias") or {}).get("B") or {}
    rec = d.get("recomendada") or d.get("inclinacion")
    checks = {
        "recomienda_revocar": rec == "A" and str(va.get("rama") or "").startswith("revoca"),
        "anuncia_conceptos_omitidos": bool((va.get("conceptos_omitidos") or {}).get("hacen_falta")),
        "via_b_es_el_juzgado": (not vb.get("prospera")) and bool((vb.get("cadena") or {}).get("regla")),
        "cero_registros_fuera": citas_fuera(d) == 0,
        "secundarios_solos": all(s.get("en_A") and s.get("en_B") and not s["en_A"].get("recalificar")
                                 and not s["en_B"].get("recalificar")
                                 for s in d.get("secundarios") or []),
    }
    fila = {"banco": "631", "estado": d.get("estado"), "recomendada": rec, "checks": checks,
            "uso": d.get("uso"), "t0": time.strftime("%Y-%m-%d %H:%M:%S")}
    anotar(fila)
    for k, v in checks.items():
        print(f"  {'✓' if v else '✗'} {k}")
    return fila


async def correr_oaj(n_por_tipo: int, semilla: int, con_base: bool) -> list:
    m = _main()
    import deliberacion as dl
    import fase6_rag as f6r
    import fase_espejo as fe
    import fase5_propuesta as f5
    embed_leyes = (lambda t: m.get_dense_embedding(t, modelo=m.EMBEDDING_MODEL))
    clave, _ = fe.resolver_tribunal(TRIBUNAL_ESPEJO, "22")
    filas = []
    import contexto_taller as _ct
    import fase_oaj as fo
    _, organo = fo.organo_de(TRIBUNAL_ESPEJO, "22")
    for caso in muestra_oaj(OAJ_DIR, n_por_tipo, semilla):
        p = dict(caso["principal"], jerarquia="principal")
        tipo = "amparo_revision" if caso["tipo"] == "AR" else "amparo_directo"
        fila = {"banco": "oaj", "archivo": caso["archivo"], "tipo": caso["tipo"], "oro": caso["oro"]}
        # LA EXCLUSIÓN DE ESTE CASO (29-sep-2026): su NEUN, su número y todo lo
        # fechado desde su sentencia, en TODAS las fuentes (contexto_taller):
        # OAJ, espejo viejo, holdings, co-citación, sembrado y web.
        _exc = {"neuns": [caso["neun"]] if caso.get("neun") else [],
                "expedientes": [caso["numero"]] if caso.get("numero") else [],
                "fecha_corte": _fecha_de_neun(caso.get("neun")) if caso.get("neun") else ""}
        # TODAS LAS BANDERAS FIJAS (revisión del 29-sep): casa=True encendería
        # las «casa» sin decirlo y el banco mediría dos cambios a la vez.
        _ct.poner(True, {"exclusion": _exc, "banderas": dict(BANDERAS_OAJ)})
        fila["exclusion"] = _exc
        try:
            mat = await f6r.material_para(m.qdrant_client, m._embedding_juris, embed_leyes,
                                          p["pregunta"], "leyes_queretaro", cliente=m.chat_client,
                                          hecho=" ".join(str(p.get(k) or "") for k in ("combate", "resolvio")))
            mat.tipo_asunto = tipo
            # LOS PRECEDENTES QUE VE PRODUCCIÓN: los de la OAJ (antes, el espejo
            # viejo, que en el 3TCC ya sólo es el respaldo), con la exclusión.
            espejo = []
            if organo:
                espejo = sin_fuga(await fo.precedentes_oaj(
                    m.qdrant_client, embed_leyes, p, tipo, organo, caso.get("numero") or "") or [], caso)
            if not espejo and clave:
                espejo = sin_fuga(await fe.espejo(m.qdrant_client, embed_leyes, p["pregunta"],
                                                  clave, "22") or [], caso)

            async def _buscar(preg, figura):
                return await f6r.material_para(m.qdrant_client, m._embedding_juris, embed_leyes,
                                               preg, "leyes_queretaro", cliente=m.chat_client,
                                               contexto=figura)
            d = await dl.deliberar(
                m.chat_client, problemas=[p], material=mat,
                resumen_acto=f"{caso['acto']}\n{p.get('resolvio') or ''}",
                resumen_conceptos=str(p.get("combate") or ""),
                # ENTRADA DERIVADA: no hay texto literal; lo único que se puede
                # citar es lo que el índice escribió.
                textos={"acto": f"{caso['acto']}\n{p.get('resolvio') or ''}",
                        "escrito": str(p.get("combate") or "")},
                tipo_asunto=tipo, es_recurso=caso["tipo"] == "AR",
                buscar=_buscar, filas_propias=espejo)
            fila["predicho"], fila["estado"] = veredicto(d)
            fila.update({"otra_sostenible": d.get("otra_sostenible"),
                         "citas_fuera": citas_fuera(d), "citas_quitadas": citas_quitadas(d),
                         "uso": d.get("uso")})
            if con_base:
                _, g, _ = await f5.proponer(m.chat_client, [p], mat, caso["acto"],
                                            str(p.get("combate") or ""), caso["tipo"] == "AR")
                fila["predicho_hoy"] = (bool(__import__("tipos_asunto").prospera(g.sentido))
                                        if getattr(g, "alcanza", False) else None)
        except Exception as e:
            fila["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        anotar(fila)
        filas.append(fila)
    return filas


def kingston_hoy(etapa: str = "base") -> list:
    """El brazo «hoy» de Kingston, sin coste: las filas de la etapa de
    referencia del banco que ya corrió (la última de cada asunto), en la escala
    de este banco. SÓLO las medidas con la exclusión puesta (revisión del
    29-sep): las del 14-sep se midieron con fuga."""
    if not KINGSTON_RESULTADOS.exists():
        return []
    ult = {}
    for x in KINGSTON_RESULTADOS.read_text(encoding="utf-8").splitlines():
        if not x.strip():
            continue
        f = json.loads(x)
        if (f.get("etapa") == etapa and f.get("exclusion") and not f.get("error")
                and f.get("oro") in ("concede", "niega")):
            ult[f["asunto"]] = f
    return [{"asunto": a, "oro": f["oro"] == "concede",
             "predicho": {"concede": True, "niega": False}.get(f.get("propuesto"))}
            for a, f in ult.items()]


def comparar() -> None:
    hoy = kingston_hoy()
    if hoy:
        informe("kingston · la propuesta de hoy (filas «antes»)", hoy)
    if not RESULTADOS.exists():
        print("Sin resultados de la deliberación todavía.")
        return
    filas = [json.loads(x) for x in RESULTADOS.read_text(encoding="utf-8").splitlines() if x.strip()]
    por = collections.defaultdict(list)
    for f in filas:
        if not f.get("error") and f.get("banco") in ("kingston", "oaj"):
            por[f["banco"]].append(f)
    for b, fs in por.items():
        informe(b, fs)
        if b == "oaj" and any("predicho_hoy" in f for f in fs):
            hoy = [dict(f, predicho=f.get("predicho_hoy")) for f in fs if "predicho_hoy" in f]
            informe("oaj · la propuesta de hoy", hoy)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Banco de la deliberación (se niega sin --si-gastar).")
    ap.add_argument("--banco", choices=("kingston", "oaj", "631", "todos"), default="todos")
    ap.add_argument("--n-oaj", type=int, default=400)
    ap.add_argument("--semilla", type=int, default=20260928)
    ap.add_argument("--solo", type=int, default=0)
    ap.add_argument("--con-base", action="store_true",
                    help="en la OAJ, corre también la propuesta de hoy (brazo de comparación)")
    ap.add_argument("--si-gastar", action="store_true")
    ap.add_argument("--comparar", action="store_true")
    ap.add_argument("--banderas", default="", help='JSON; p. ej. {"fuerza_unificada": true}')
    a = ap.parse_args(argv)
    if a.banderas:
        BANDERAS_OAJ.update(json.loads(a.banderas))
    print(f"  banderas efectivas: {BANDERAS_OAJ}")
    if a.comparar:
        comparar()
        return 0
    bancos = ["kingston", "oaj", "631"] if a.banco == "todos" else [a.banco]
    est = coste_estimado(bancos, a.n_oaj, a.con_base)
    print("═══ BANCO DE LA DELIBERACIÓN · coste estimado (confirmar con una o dos llamadas) ═══")
    for k, (n, usd) in est.items():
        print(f"  {k:<10} {n:>4} asunto(s) · ≈ {usd:.2f} USD")
    print("  compuertas, escritas antes de correr:")
    for c, txt in COMPUERTAS:
        print(f"    [{c}] {txt}")
    if not a.si_gastar:
        print("\nNO SE CORRE NADA: pasa --si-gastar sólo con el permiso de David para ESTA "
              "corrida (regla de no gastar API sin preguntar).")
        return 2
    t0 = time.time()
    if "kingston" in bancos:
        asyncio.run(correr_kingston(a.solo))
    if "oaj" in bancos:
        asyncio.run(correr_oaj(a.n_oaj // 2, a.semilla, a.con_base))
    if "631" in bancos:
        asyncio.run(correr_631())
    comparar()
    print(f"\n{time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
