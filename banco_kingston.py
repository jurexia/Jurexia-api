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

Uso:
    .venv/bin/python banco_kingston.py --etapa antes   [--paralelo 3] [--solo N]
    .venv/bin/python banco_kingston.py --etapa despues [--paralelo 3]
    .venv/bin/python banco_kingston.py --comparar
"""
import argparse, asyncio, glob, json, os, re, subprocess, sys, time, collections
from pathlib import Path

sys.path.insert(0, "/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/redactor-sentencias/utillaje")
from comparar import sentido as sentido_del_oro, calificaciones  # noqa: E402

BASE = "https://jurexia-api.onrender.com"
CORREO = "administracion@iurexia.com"     # de casa: sin tope, y no ensucia el historial de David
CASOS = "/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/redactor-sentencias/corpus/casos"
AQUI = Path(__file__).parent / "bancos" / "kingston"
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
                if r.get("etapa") == etapa and not r.get("error"):
                    # EN LA ETAPA «DESPUÉS» SÓLO CUENTA LO QUE LLEVA CONTRASTE. La
                    # primera corrida arrancó con el despliegue recién vivo y dos
                    # casos los atendió el worker que aún rodaba el código viejo:
                    # salieron «ok» y sin contraste. Darlos por hechos sería medir
                    # el antes y llamarlo después.
                    if etapa != "antes" and not r.get("contraste"):
                        continue
                    hechos[r["asunto"]] = r
    return hechos


def anotar(fila: dict) -> None:
    with open(RESULTADOS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(fila, ensure_ascii=False) + "\n")


async def correr_caso(caso: dict, etapa: str, sem: asyncio.Semaphore) -> dict:
    import httpx
    asunto = caso["asunto"]; numero = numero_de(asunto)
    oro = sentido_del_oro(caso["oro"])
    fila = {"etapa": etapa, "asunto": asunto, "numero": numero, "oro": oro,
            "t0": time.strftime("%Y-%m-%d %H:%M:%S")}
    async with sem:
        t = time.time()
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=1800)) as cx:
                if etapa == "antes":
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
                    fila["problemas"] = len((r.json() or {}).get("problemas") or [])
                    fila["t_acervo"] = round(time.time() - t)

                r = await cx.post(f"{BASE}/taller/proponer",
                                  data={"numero": numero, "user_email": CORREO})
                if r.status_code != 200:
                    raise RuntimeError(f"proponer {r.status_code}: {r.text[:200]}")
                p = r.json()
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


async def correr(etapa: str, paralelo: int, solo: int | None) -> None:
    casos = banco()
    hechos = ya_hechos(etapa)
    pendientes = [c for c in casos if c["asunto"] not in hechos]
    if solo:
        pendientes = pendientes[:solo]
    print(f"═══ etapa «{etapa}» · {len(casos)} casos · {len(hechos)} hechos · "
          f"{len(pendientes)} por correr · {paralelo} en paralelo ═══", flush=True)
    sem = asyncio.Semaphore(paralelo)
    await asyncio.gather(*(correr_caso(c, etapa, sem) for c in pendientes))
    comparar()


def comparar() -> None:
    filas = [json.loads(l) for l in RESULTADOS.read_text(encoding="utf-8").splitlines() if l.strip()]
    por = collections.defaultdict(dict)
    for f in filas:
        if f.get("error"):
            continue
        if f["etapa"] != "antes" and not f.get("contraste"):
            continue                                # worker viejo: no es «después»
        por[f["etapa"]][f["asunto"]] = f           # la última de cada asunto manda
    print("\n═══ RESULTADO ═══")
    for etapa, d in por.items():
        n = len(d); ok = sum(1 for f in d.values() if f.get("acierta"))
        conf = collections.Counter((f["oro"], f["propuesto"]) for f in d.values())
        print(f"  {etapa:<8} sentido acertado {ok}/{n} = {100*ok/max(n,1):.0f}%   "
              f"(línea base «siempre niega»: {sum(1 for f in d.values() if f['oro']=='niega')}/{n})")
        for (o, p), k in sorted(conf.items()):
            print(f"           oro={o:<8} motor={p:<13} {k}")
    for et in [e for e in por if e != "antes"]:
        if "antes" not in por:
            break
        comunes = set(por["antes"]) & set(por[et])
        mejora = sum(1 for a in comunes if por[et][a]["acierta"] and not por["antes"][a]["acierta"])
        empeora = sum(1 for a in comunes if por["antes"][a]["acierta"] and not por[et][a]["acierta"])
        print(f"\n  «{et}» sobre los {len(comunes)} comunes con «antes»: arregla {mejora} y estropea {empeora}")
        for a in sorted(comunes):
            x, y = por["antes"][a], por[et][a]
            if x["acierta"] != y["acierta"]:
                print(f"    {'↑' if y['acierta'] else '↓'} {a[:44]:<44} oro={x['oro']:<8} "
                      f"antes={x['propuesto']:<8} {et}={y['propuesto']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa")   # antes | despues | despues2 …: lo que no es «antes» sólo propone
    ap.add_argument("--paralelo", type=int, default=3)
    ap.add_argument("--solo", type=int)
    ap.add_argument("--comparar", action="store_true")
    a = ap.parse_args()
    if a.comparar or not a.etapa:
        comparar()
    else:
        asyncio.run(correr(a.etapa, a.paralelo, a.solo))
