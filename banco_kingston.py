# -*- coding: utf-8 -*-
"""EL BANCO KINGSTON, CONTRA EL TALLER DE VERDAD.

David, 14-sep-2026: «Adelante, empieza por el paso 1 y mídelo sobre los 72».

DE LOS 72 TRIPLETES, 24 TIENEN ORO SÓLIDO —un engrose real con tres o más
calificaciones—. Los otros 48 tienen como oro un ADELANTO: un borrador que se
circula antes de la sesión, con el estudio a medias y huecos «se listó el ____».
Contra ésos toda medida da cero y parece culpa del modelo; ya pasó una vez y
costó una ronda entera. Se mide sobre los 24: 13 niegan, 11 conceden, todos
amparo directo. La línea base tonta —«siempre niega»— es 13/24 = 54%.

SE MIDE EL JUICIO, NO LA REDACCIÓN. El sentido sale de `/taller/proponer`, que es
donde el motor decide; generar el documento entero cuesta cinco veces más y no
cambia el sentido. Cada caso pasa por el taller COMO LO HARÍA UN SECRETARIO:
sus dos documentos, en PDF, por la API de producción. Nada de atajos locales que
midan otra cosa.

DOS ETAPAS SOBRE LAS MISMAS SESIONES. `--etapa antes` corre adelanto, acervo y
propuesta. `--etapa despues` reutiliza la sesión —que vive en Supabase— y sólo
vuelve a proponer, con el código nuevo ya desplegado. Así el antes y el después
comparten material, y la única diferencia es el paso que se está midiendo.

ARREGLADO EL 29-SEP-2026 (rediseño del taller, punto 8; David: «arregla el
banco»). Tres defectos medían en falso:
  1. LA FUGA. 18 de los 24 asuntos están en el índice de la OAJ (el 274/2025 con
     NEUN 38118729, fechado el 28-05-2026) y la única exclusión era el número
     tecleado; el espejo viejo, los holdings, la co-citación y la web no
     excluían nada. Ahora cada asunto viaja con su EXCLUSIÓN —NEUN, número y
     serie, y fecha de corte = la de su sentencia en la OAJ— en el campo
     `evaluacion` del adelanto; la API la guarda en la sesión y todas las
     fuentes la respetan (contexto_taller). Además se cuentan las FUGAS: filas
     del espejo que sean el propio asunto o posteriores al corte.
  2. EL «DESPUÉS» NO RECALCULABA. /taller/proponer servía la propuesta guardada
     y el banco llamaba a eso «después». Ahora las etapas que no son «antes» van
     con `recalcular=1`.
  3. UNA SOLA COSA POR ETAPA. `--banderas '{"fuerza_unificada": false}'` enciende
     o apaga un cambio sólo en esa etapa, sobre las mismas sesiones.
Y los resultados viven en el checkout principal, no en el worktree de turno.

ESTE BANCO ES DE REGRESIÓN Y ESTÁ CONTAMINADO: siete de sus asuntos entraron en
la calibración de la tabla OAJ, y todos se han usado para depurar. Su número no
se publica como exactitud del motor; sirve para ver que un cambio no empeora.

Uso:
    .venv/bin/python banco_kingston.py --etapa base    [--paralelo 3] [--solo N]
    .venv/bin/python banco_kingston.py --etapa fuerza_off --banderas '{"fuerza_unificada": false}'
    .venv/bin/python banco_kingston.py --etapa fuerza_on  --banderas '{"fuerza_unificada": true}'
    .venv/bin/python banco_kingston.py --comparar [--contra fuerza_off]
(«antes» o «base» crean las sesiones con adelanto y acervo; las demás sólo
vuelven a proponer, recalculando.)
"""
import argparse, asyncio, glob, json, os, re, subprocess, sys, time, collections
from pathlib import Path

sys.path.insert(0, "/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/redactor-sentencias/utillaje")
from comparar import sentido as sentido_del_oro, calificaciones  # noqa: E402

BASE = "https://jurexia-api.onrender.com"
CORREO = "administracion@iurexia.com"     # de casa: sin tope, y no ensucia el historial de David
CASOS = "/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/redactor-sentencias/corpus/casos"
# EN EL CHECKOUT PRINCIPAL, no junto al archivo: en un worktree la carpeta no
# existía y banco_deliberacion leía un «hoy» vacío.
AQUI = Path("/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/jurexia-api-git/bancos/kingston")
INDICE_OAJ = "/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/redactor-sentencias/oaj/indice_oaj_v2.sqlite"
ORGANO_OAJ = "Tercer Tribunal Colegiado en Materias Administrativa y Civil del Vig%"
ETAPAS_CON_ADELANTO = ("antes", "base")
# LA LÍNEA BASE ES PRODUCCIÓN DE HOY PARA UN SECRETARIO DE FUERA (revisión del
# 29-sep): con la cuenta de casa, las banderas «casa» se encenderían solas y la
# base mediría dos cambios a la vez. Toda etapa fija TODAS las banderas; las que
# no se pidan, apagadas.
BANDERAS_BASE = {"fuerza_unificada": False, "fuente_tardia_aviso": False, "normas_al_documento": False,
                 "tesis_parte_al_consultar": False, "consulta_provisional": False}
AQUI.mkdir(parents=True, exist_ok=True)
PDFS = AQUI / "pdf"; PDFS.mkdir(exist_ok=True)
RESULTADOS = AQUI / "resultados.jsonl"

# Los que PROSPERAN, en la escala de conceptos del taller. Si el asunto entero
# se propone con uno de éstos, el amparo se concede; con cualquier otro, se niega.
PROSPERAN = {"fundado", "esencialmente_fundado", "sustancialmente_fundado",
             "parcialmente_fundado", "esencialmente fundado",
             "sustancialmente fundado", "parcialmente fundado"}


def banco() -> list:
    """Los 24 con oro sólido, con la misma regla que `correr.casos`."""
    vistos, out = set(), []
    for f in sorted(glob.glob(f"{CASOS}/*.json")):
        if os.path.basename(f).startswith("_"):
            continue
        c = json.load(open(f))
        if not isinstance(c, dict) or not c.get("oro") or c["asunto"] in vistos:
            continue
        if sum(calificaciones(c["oro"]).values()) < 3:
            continue
        vistos.add(c["asunto"]); c["_f"] = f
        out.append(c)
    return out


def numero_de(asunto: str) -> str:
    """«2. ADC 274-2025» → «274/2025». El primer número-año que aparezca."""
    m = re.search(r"(\d{1,4})\s*[-/]\s*(20\d{2})", asunto)
    return f"{m.group(1)}/{m.group(2)}" if m else asunto


def exclusion_de(caso: dict) -> dict:
    """La exclusión del fallo objetivo: su número y los de su serie («ADC
    463-2024, 492-2024»), su NEUN y, si la OAJ lo publicó, la fecha de su
    sentencia como corte (nada fechado ese día o después)."""
    import sqlite3
    nums = sorted({f"{m.group(1)}/{m.group(2)}"
                   for m in re.finditer(r"(\d{1,4})\s*[-/]\s*(20\d{2})", caso["asunto"])})
    neuns, fechas = [], []
    try:
        db = sqlite3.connect(f"file:{INDICE_OAJ}?mode=ro&immutable=1", uri=True)
        for num in nums:
            n, a = num.split("/")
            for neun, alias, fecha in db.execute(
                    "SELECT neun, alias, fecha_sentencia FROM asuntos WHERE tipo='Amparo Directo' "
                    "AND organo LIKE ? AND (alias LIKE ? OR alias LIKE ?)",
                    (ORGANO_OAJ, f"%{n}/{a}%", f"%{n}-{a}%")):
                if re.search(r"(?<!\d)" + n + r"\s*[/\-]\s*" + a, alias or ""):
                    neuns.append(int(neun)); fechas.append(fecha)
    except Exception as e:
        print(f"   ⚠️ índice OAJ no disponible para la exclusión: {e}")
    import datetime as _dt
    def _f(s):
        try:
            d, m, y = str(s).split("-"); return _dt.date(int(y), int(m), int(d))
        except Exception:
            return None
    corte = min((x for x in map(_f, fechas) if x), default=None)
    return {"expedientes": nums, "neuns": sorted(set(neuns)),
            "fecha_corte": corte.isoformat() if corte else "", "serie": caso["asunto"][:60]}


def _comprobar_aplicado(resp: dict, exc: dict, banderas: dict, donde: str, recalculada=None) -> None:
    """LA FILA ES LIMPIA POR LO QUE EL SERVIDOR APLICÓ, NO POR LO QUE SE ENVIÓ
    (revisión del 29-sep): un worker con código viejo ignora sin error los
    campos que no conoce, y sirve la propuesta guardada. La API devuelve a las
    cuentas de casa `evaluacion_aplicada`; si falta o no casa, error."""
    ap = (resp or {}).get("evaluacion_aplicada")
    if not isinstance(ap, dict):
        raise RuntimeError(f"{donde}: el servidor no devolvió evaluacion_aplicada (¿código viejo?)")
    ex = ap.get("exclusion") or {}
    if exc.get("neuns") and sorted(ex.get("neuns") or []) != sorted(exc["neuns"]):
        raise RuntimeError(f"{donde}: la exclusión aplicada no es la pedida ({ex.get('neuns')})")
    if (exc.get("fecha_corte") or "") != (ex.get("fecha_corte") or ""):
        raise RuntimeError(f"{donde}: fecha de corte aplicada {ex.get('fecha_corte')!r}")
    for k, v in (banderas or {}).items():
        if (ap.get("banderas") or {}).get(k) != v:
            raise RuntimeError(f"{donde}: la bandera {k} no rigió como se pidió")
    if recalculada is not None and ap.get("recalculada") is not recalculada:
        raise RuntimeError(f"{donde}: no se recalculó la propuesta")


def fugas_en(espejo, exc: dict) -> int:
    """Filas del espejo que son el propio asunto o posteriores al corte: con la
    exclusión puesta deben ser CERO. Se cuentan, no se esconden."""
    import datetime as _dt
    nums = set(exc.get("expedientes") or []); neuns = set(exc.get("neuns") or [])
    corte = exc.get("fecha_corte") or ""
    n = 0
    for g in (espejo or []):
        for f in ((g or {}).get("filas") or []):
            num = re.search(r"(\d{1,5})\s*[/\-]\s*(\d{4})", str(f.get("expediente") or ""))
            fecha = str(f.get("fecha") or "")
            iso = ""
            m = re.match(r"^(\d{1,2})-(\d{1,2})-(\d{4})", fecha)
            if m:
                iso = f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
            elif re.match(r"^\d{4}-\d{2}-\d{2}", fecha):
                iso = fecha[:10]
            if ((num and f"{int(num.group(1))}/{num.group(2)}" in nums)
                    or (f.get("neun") and int(float(f["neun"])) in neuns)
                    or (corte and iso and iso >= corte)):
                n += 1
    return n


def materia_de(caso: dict) -> str:
    r = " ".join(caso.get("rama") or []).upper()
    return "administrativa" if "ADMINISTRATIV" in r else "civil"


def pdf_de(texto: str, ruta: Path) -> Path:
    """Texto plano → PDF, con cupsfilter (viene con macOS). Cacheado."""
    if ruta.exists() and ruta.stat().st_size > 1000:
        return ruta
    txt = ruta.with_suffix(".txt")
    txt.write_text(texto, encoding="utf-8")
    with open(ruta, "wb") as fh:
        # `-i text/plain` A LA FUERZA. cupsfilter olfatea el contenido y a la
        # demanda del ADC 641-2024 la tomó por una imagen —«cgimagetopdf: problem
        # reading the image file»— por los bytes con que empieza. Con el tipo
        # dicho no adivina.
        subprocess.run(["cupsfilter", "-i", "text/plain", str(txt)], stdout=fh,
                       stderr=subprocess.DEVNULL, check=True)
    return ruta


def veredicto_del_global(g: dict) -> str:
    s = str((g or {}).get("sentido") or "").strip().lower().replace("_", " ")
    if not s:
        return "indeterminado"
    return "concede" if s in {x.replace("_", " ") for x in PROSPERAN} else "niega"


def ya_hechos(etapa: str) -> dict:
    hechos = {}
    if RESULTADOS.exists():
        for ln in RESULTADOS.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                if r.get("etapa") == etapa and not r.get("error") and r.get("exclusion"):
                    # EN LA ETAPA «DESPUÉS» SÓLO CUENTA LO QUE LLEVA CONTRASTE. La
                    # primera corrida arrancó con el despliegue recién vivo y dos
                    # casos los atendió el worker que aún rodaba el código viejo:
                    # salieron «ok» y sin contraste. Darlos por hechos sería medir
                    # el antes y llamarlo después.
                    # «base» también crea sesiones: no se re-corre por traer
                    # el contraste vacío (sobrescribiría la sesión que ya
                    # midieron las otras etapas).
                    if etapa not in ETAPAS_CON_ADELANTO and not r.get("contraste"):
                        continue
                    hechos[r["asunto"]] = r
    return hechos


def anotar(fila: dict) -> None:
    with open(RESULTADOS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(fila, ensure_ascii=False) + "\n")


async def correr_caso(caso: dict, etapa: str, sem: asyncio.Semaphore,
                      banderas: dict | None = None) -> dict:
    import httpx
    asunto = caso["asunto"]; numero = numero_de(asunto)
    oro = sentido_del_oro(caso["oro"])
    exc = exclusion_de(caso)
    banderas = {**BANDERAS_BASE, **(banderas or {})}
    fila = {"etapa": etapa, "asunto": asunto, "numero": numero, "oro": oro,
            "exclusion": exc, "banderas": banderas, "sin_corte": not exc.get("fecha_corte"),
            "t0": time.strftime("%Y-%m-%d %H:%M:%S")}
    async with sem:
        t = time.time()
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=1800)) as cx:
                if etapa in ETAPAS_CON_ADELANTO:
                    slug = re.sub(r"[^A-Za-z0-9]+", "_", asunto)[:40]
                    acto = pdf_de(caso["piezas"]["acto"]["texto"], PDFS / f"{slug}_acto.pdf")
                    dem = pdf_de(caso["piezas"]["demanda"]["texto"], PDFS / f"{slug}_demanda.pdf")
                    datos = {
                        "numero": numero, "tipo_asunto": "amparo_directo",
                        "materia": materia_de(caso), "modo": "generado",
                        "notificacion": "2025-03-03", "presentacion": "2025-03-14",
                        "regla_surtimiento": "personal", "user_email": CORREO,
                        "encabezado": f"BANCO KINGSTON · {asunto}",
                        "tribunal": "Tercer Tribunal Colegiado en Materias Administrativa "
                                    "y Civil del Vigésimo Segundo Circuito",
                        "ciudad": "Querétaro, Querétaro",
                        "evaluacion": json.dumps({"exclusion": exc, "banderas": banderas or {}},
                                                 ensure_ascii=False),
                    }
                    with open(acto, "rb") as fa, open(dem, "rb") as fd:
                        r = await cx.post(f"{BASE}/taller/adelanto", data=datos,
                                          files={"acto": ("acto.pdf", fa, "application/pdf"),
                                                 "conceptos": ("demanda.pdf", fd, "application/pdf")})
                    if r.status_code != 200:
                        raise RuntimeError(f"adelanto {r.status_code}: {r.text[:200]}")
                    fila["t_adelanto"] = round(time.time() - t)
                    r = await cx.post(f"{BASE}/taller/consultar",
                                      data={"numero": numero, "user_email": CORREO,
                                            "coleccion_estatal": "leyes_queretaro"})
                    if r.status_code != 200:
                        raise RuntimeError(f"consultar {r.status_code}: {r.text[:200]}")
                    _cj = r.json() or {}
                    _comprobar_aplicado(_cj, exc, banderas, "consultar")
                    fila["problemas"] = len(_cj.get("problemas") or [])
                    fila["fugas"] = fugas_en(_cj.get("espejo"), exc)
                    fila["t_acervo"] = round(time.time() - t)

                _dp = {"numero": numero, "user_email": CORREO, "recalcular": "1",
                       "banderas": json.dumps(banderas)}
                r = await cx.post(f"{BASE}/taller/proponer", data=_dp)
                if r.status_code != 200:
                    raise RuntimeError(f"proponer {r.status_code}: {r.text[:200]}")
                p = r.json()
                _comprobar_aplicado(p, exc, banderas, "proponer", recalculada=True)
                g = p.get("global") or {}
                fila.update({
                    "sentido_global": g.get("sentido"),
                    "alcanza": g.get("alcanza"),
                    "confianza": g.get("confianza"),
                    "propuesto": veredicto_del_global(g),
                    "por_problema": [x.get("sentido") for x in (p.get("propuestas") or [])],
                    "contraste": p.get("contraste"),            # lo trae el paso 1 cuando existe
                    "t_total": round(time.time() - t),
                })
                fila["acierta"] = (fila["propuesto"] == oro)
        except Exception as e:
            fila["error"] = str(e)[:300]
    anotar(fila)
    marca = "✓" if fila.get("acierta") else ("✗" if not fila.get("error") else "!")
    print(f"  {marca} {asunto[:44]:<44} oro={oro:<8} motor={fila.get('propuesto', '—'):<13} "
          f"{fila.get('t_total', '—')}s {('· ' + fila['error'][:70]) if fila.get('error') else ''}",
          flush=True)
    return fila


async def correr(etapa: str, paralelo: int, solo: int | None, banderas: dict | None = None) -> None:
    casos = banco()
    hechos = ya_hechos(etapa)
    pendientes = [c for c in casos if c["asunto"] not in hechos]
    if solo:
        pendientes = pendientes[:solo]
    print(f"═══ etapa «{etapa}» · {len(casos)} casos · {len(hechos)} hechos · "
          f"{len(pendientes)} por correr · {paralelo} en paralelo ═══", flush=True)
    sem = asyncio.Semaphore(paralelo)
    await asyncio.gather(*(correr_caso(c, etapa, sem, banderas) for c in pendientes))
    comparar()


def comparar(contra: str = "") -> None:
    filas = [json.loads(l) for l in RESULTADOS.read_text(encoding="utf-8").splitlines() if l.strip()]
    por = collections.defaultdict(dict)
    for f in filas:
        if f.get("error"):
            continue
        # LAS FILAS DE ANTES DEL ARREGLO (sin exclusión) no se comparan con las
        # nuevas: se midieron con fuga.
        if not f.get("exclusion"):
            continue
        por[f["etapa"]][f["asunto"]] = f           # la última de cada asunto manda
    print("\n═══ RESULTADO · banco de REGRESIÓN, contaminado (no es la exactitud del motor) ═══")
    print("\n═══ RESULTADO ═══")
    for etapa, d in por.items():
        n = len(d); ok = sum(1 for f in d.values() if f.get("acierta"))
        conf = collections.Counter((f["oro"], f["propuesto"]) for f in d.values())
        fug = sum(int(f.get("fugas") or 0) for f in d.values())
        print(f"  {etapa:<10} sentido acertado {ok}/{n} = {100*ok/max(n,1):.0f}%   "
              f"(línea base «siempre niega»: {sum(1 for f in d.values() if f['oro']=='niega')}/{n})"
              + (f"   ⚠️ FUGAS: {fug}" if fug else ""))
        # LOS SEIS SIN FECHA DE CORTE, APARTE: en ellos lo posterior no se
        # excluye (no están en la OAJ como AD) y «0 fugas» no dice nada.
        con = [f for f in d.values() if not f.get("sin_corte") and f.get("exclusion", {}).get("fecha_corte")]
        sin = [f for f in d.values() if f not in con]
        if sin:
            print(f"             · con corte {sum(1 for f in con if f.get('acierta'))}/{len(con)} · "
                  f"SIN corte {sum(1 for f in sin if f.get('acierta'))}/{len(sin)} (lo posterior no se excluyó)")
        for (o, p), k in sorted(conf.items()):
            print(f"           oro={o:<8} motor={p:<13} {k}")
    ref = contra or ("base" if "base" in por else "antes")
    for et in [e for e in por if e != ref]:
        if ref not in por:
            break
        comunes = set(por[ref]) & set(por[et])
        mejora = sum(1 for a in comunes if por[et][a]["acierta"] and not por[ref][a]["acierta"])
        empeora = sum(1 for a in comunes if por[ref][a]["acierta"] and not por[et][a]["acierta"])
        print(f"\n  «{et}» sobre los {len(comunes)} comunes con «{ref}»: arregla {mejora} y estropea {empeora}")
        for a in sorted(comunes):
            x, y = por[ref][a], por[et][a]
            if x["acierta"] != y["acierta"]:
                print(f"    {'↑' if y['acierta'] else '↓'} {a[:44]:<44} oro={x['oro']:<8} "
                      f"{ref}={x['propuesto']:<8} {et}={y['propuesto']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa")   # base | antes crean sesiones; las demás sólo proponen, recalculando
    ap.add_argument("--paralelo", type=int, default=3)
    ap.add_argument("--solo", type=int)
    ap.add_argument("--banderas", default="")
    ap.add_argument("--contra", default="")
    ap.add_argument("--comparar", action="store_true")
    ap.add_argument("--exclusiones", action="store_true", help="imprime la exclusión de cada asunto y sale")
    a = ap.parse_args()
    if a.exclusiones:
        for c in banco():
            print(c["asunto"][:44], json.dumps(exclusion_de(c), ensure_ascii=False))
    elif a.comparar or not a.etapa:
        comparar(a.contra)
    else:
        asyncio.run(correr(a.etapa, a.paralelo, a.solo, json.loads(a.banderas) if a.banderas else None))
