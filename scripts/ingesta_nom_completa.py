"""INGESTA SERIA DE NORMAS OFICIALES MEXICANAS (11-oct-2026, segunda vuelta).

David: «un fracaso la ingesta de NOM… tránsito es indispensable, ingeniería,
materiales peligrosos, hidrocarburos, salud… vamos por una ingesta seria».
La primera vuelta eran 52 elegidas a mano y una ya sustituida (NOM-034-SCT2-2011,
hoy NOM-034-SCT2/SEDATU-2022). Ésta toma TODAS las NOM definitivas del DOF:

  1. índice día por día del SIDOF (reingesta/nom/indice_nom_dof.jsonl,
     1993 → hoy): toda nota cuyo título menciona una NOM;
  2. definitivas = «Norma Oficial Mexicana NOM-…» y sus «Modificación…»; fuera
     proyectos, respuestas a comentarios, avisos y normas de emergencia;
  3. vigencia: fuera las que un «Aviso/Declaratoria de cancelación» nombra
     después de publicadas; y de cada norma (número + dependencia, p. ej.
     034-SCT2) sólo la versión más reciente —la anterior está sustituida—;
  4. texto de la nota (SIDOF), trozos por numeral (tope por norma para cuidar
     el disco), PDF con Chrome desde el HTML del DOF → gs://iurexia-leyes/nom/;
  5. vectores text-embedding-3-small en `normas_oficiales` y, al final, se
     borran los puntos de las claves sustituidas o canceladas.

Reanudable: caché de notas en reingesta/nom/html, PDF en reingesta/nom/pdf,
lotes ya subidos en reingesta/nom/subidos.txt.
  python3 scripts/ingesta_nom_completa.py --plan      (qué entra; no gasta)
  .venv/bin/python scripts/ingesta_nom_completa.py --notas   (bajar + PDF)
  python3 scripts/ingesta_nom_completa.py --gcs       (subir PDF; python con google-cloud-storage)
  .venv/bin/python scripts/ingesta_nom_completa.py --vectores
"""
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ingesta_nom as base  # noqa: E402  (nota, texto_de, trocear, pdf_de, slug, DIR, COLECCION)
from ingesta_procesal_federal import env  # noqa: E402

DIR = base.DIR
INDICE = DIR / "indice_nom_dof.jsonl"
PLAN = DIR / "plan_nom_completa.json"
SUBIDOS = DIR / "subidos.txt"
TOPE_TROZOS = 220

RX_CLAVE = re.compile(r"NOM[\s‑–—-]*(?:EM[\s‑–—-]*)?(\d{3}(?:/\d)?(?:[\s‑–—-]+\d)?)[\s‑–—-]*([A-Z][A-Z0-9/\s‑–—-]*?)[\s‑–—-]*(\d{4})\b")
RX_DEFINITIVA = re.compile(r"^\s*(norma oficial mexicana|modificaci[oó]n (a|de)( la| los)?( puntos?| numerales?)?.*norma oficial mexicana|acuerdo por el que se modifica.*norma oficial mexicana)", re.I)
RX_FUERA = re.compile(r"proyecto|respuesta|aviso de pr[oó]rroga|aclaraci[oó]n|fe de erratas|de emergencia|NOM-EM|nota aclaratoria|cancelaci[oó]n", re.I)
RX_CANCELA = re.compile(r"(aviso|declaratoria|acuerdo).{0,40}cancela", re.I)

MATERIA = [("STPS", "trabajo"), ("SSA", "salud"), ("SEMARNAT", "ambiental"), ("ECOL", "ambiental"), ("ASEA", "hidrocarburos"),
           ("SECRE", "energia"), ("SEDE", "energia"), ("SEDG", "energia"), ("ENER", "energia"), ("NUCL", "energia"),
           ("CRE", "energia"), ("SENER", "energia"), ("SCT", "transporte"), ("SICT", "transporte"), ("SEDATU", "urbano"),
           ("CONAGUA", "agua"), ("CNA", "agua"), ("SEGOB", "proteccion_civil"), ("SCFI", "consumo"), ("SE", "consumo"),
           ("FITO", "agropecuario"), ("ZOO", "agropecuario"), ("SAG", "agropecuario"), ("PESC", "agropecuario"),
           ("SSP", "seguridad"), ("SEP", "educacion"), ("SHCP", "fiscal"), ("SEMAR", "maritimo"), ("SECTUR", "turismo"),
           ("CONAGUA", "agua"), ("TUR", "turismo")]


def clave_de(titulo: str):
    m = RX_CLAVE.search(titulo.replace("\xa0", " "))
    if not m:
        return None
    num = re.sub(r"[\s‑–—-]+", "-", m.group(1)).strip("-")
    dep = re.sub(r"[\s‑–—]+", "", m.group(2)).strip("-/")
    dep = re.sub(r"-+", "-", dep)
    return f"NOM-{num}-{dep}-{m.group(3)}", num, dep, int(m.group(3))


def grupo_de(num: str, dep: str) -> str:
    """La norma, sin el año: 034 + primera dependencia. ECOL se llamó luego SEMARNAT."""
    dep = re.sub(r"([A-Z])-(\d)", r"\1\2", dep)          # SCT-2 = SCT2 (SCT1/2/3/4 son subsectores distintos)
    primera = re.split(r"[/-]", dep)[0]
    primera = {"ECOL": "SEMARNAT"}.get(primera, primera)
    return f"{num}|{primera}"


def materia_de(dep: str) -> str:
    for k, v in MATERIA:
        if dep.startswith(k) or f"/{k}" in dep:
            return v
    return "otra"


def plan():
    filas = [json.loads(l) for l in INDICE.read_text().splitlines() if l.strip()]
    for f in filas:
        f["dia"] = dt.datetime.strptime(f["fecha"], "%d-%m-%Y").date().isoformat()
    canceladas = {}
    for f in filas:
        if RX_CANCELA.search(f["titulo"]):
            for m in RX_CLAVE.finditer(f["titulo"]):
                c = clave_de(m.group(0))
                if c:
                    canceladas[c[0]] = max(canceladas.get(c[0], ""), f["dia"])
    defin = {}
    for f in sorted(filas, key=lambda x: x["dia"]):
        t = f["titulo"]
        if not RX_DEFINITIVA.search(t) or RX_FUERA.search(t.split(",")[0] + " " + t[:60]):
            continue
        c = clave_de(t)
        if not c:
            continue
        clave, num, dep, anio = c
        d = defin.setdefault(clave, {"clave": clave, "num": num, "dep": dep, "anio": anio, "notas": [], "titulo": ""})
        es_mod = bool(re.match(r"^\s*(modificaci|acuerdo)", t, re.I))
        if not es_mod:
            d["titulo"] = re.sub(r"\s+", " ", t).strip()
            d["publicada"] = f["dia"]
            d["notas"] = [x for x in d["notas"] if x["mod"] and x["dia"] > f["dia"]]  # una republicación reemplaza
        d["notas"].append({"cod": f["codNota"], "dia": f["dia"], "mod": es_mod, "titulo": t[:200]})
    # vigencia
    fuera_cancel, fuera_sustituida = [], []
    por_grupo = {}
    for clave, d in list(defin.items()):
        if not d.get("titulo"):
            del defin[clave]           # sólo modificaciones de una NOM que no está en el índice
            continue
        if clave in canceladas and canceladas[clave] > d.get("publicada", ""):
            fuera_cancel.append(clave)
            del defin[clave]
            continue
        por_grupo.setdefault(grupo_de(d["num"], d["dep"]), []).append(d)
    vigentes = []
    for g, lista in por_grupo.items():
        lista.sort(key=lambda d: (d["anio"], d.get("publicada", "")))
        vigentes.append(lista[-1])
        fuera_sustituida += [d["clave"] for d in lista[:-1]]
    for d in vigentes:
        d["materia"] = materia_de(d["dep"])
    PLAN.write_text(json.dumps({"vigentes": vigentes, "canceladas": fuera_cancel, "sustituidas": fuera_sustituida},
                               ensure_ascii=False, indent=0))
    print(f"definitivas {len(defin) + len(fuera_cancel)} · vigentes {len(vigentes)} · canceladas {len(fuera_cancel)} · "
          f"sustituidas {len(fuera_sustituida)} · notas a bajar {sum(len(d['notas']) for d in vigentes)}")
    from collections import Counter
    print(Counter(d["materia"] for d in vigentes).most_common())
    return vigentes


def bajar_y_pdf():
    vig = json.loads(PLAN.read_text())["vigentes"]
    cods = [(d, n) for d in vig for n in d["notas"]]

    def uno(par):
        d, n = par
        try:
            nota = base.nota(n["cod"])
            if nota.get("cadenaContenido"):
                base.pdf_de(n["cod"], nota)
            return 1
        except Exception:
            return 0
    with cf.ThreadPoolExecutor(3) as ex:
        hechos = sum(ex.map(uno, cods))
    print(f"notas bajadas {hechos}/{len(cods)}", flush=True)


def subir_gcs():
    from google.cloud import storage
    from google.oauth2 import service_account
    info = json.loads(env()["GCP_SA_KEY_JSON"])
    b = storage.Client(credentials=service_account.Credentials.from_service_account_info(info),
                       project=info.get("project_id")).bucket("iurexia-leyes")
    vig = json.loads(PLAN.read_text())["vigentes"]
    n = 0
    for d in vig:
        for nota in d["notas"]:
            p = DIR / "pdf" / f"{nota['cod']}.pdf"
            if p.exists() and p.stat().st_size > 2000:
                bl = b.blob(f"nom/{base.slug(d['clave'])}__{nota['cod']}.pdf")
                if not bl.exists():
                    bl.cache_control = "public, max-age=86400"
                    bl.upload_from_filename(str(p), content_type="application/pdf")
                n += 1
    print("PDF en GCS", n, flush=True)


def filas_de(d):
    fuera, vistos = [], set()
    for orden, n in enumerate(d["notas"]):
        f = DIR / "html" / f"{n['cod']}.json"
        if not f.exists():
            continue
        nota = json.loads(f.read_text())
        if not nota.get("cadenaContenido"):
            continue
        titulo = re.sub(r"\s+", " ", nota.get("titulo") or d["titulo"]).strip()
        corto = re.sub(r"^.*?NOM[^,]*,\s*", "", d["titulo"])[:220] or titulo[:220]
        tiene_pdf = (DIR / "pdf" / f"{n['cod']}.pdf").exists()
        url = f"https://storage.googleapis.com/iurexia-leyes/nom/{base.slug(d['clave'])}__{n['cod']}.pdf" if tiene_pdf else None
        for i, (numeral, txt) in enumerate(base.trocear(base.texto_de(nota["cadenaContenido"]))[:TOPE_TROZOS]):
            h = hashlib.md5(txt.encode()).hexdigest()
            if h in vistos or len(txt) < 40:
                continue
            vistos.add(h)
            cab = f"[{d['clave']} | {corto}" + (f" | numeral {numeral}" if numeral and numeral != "0" else "") + "]"
            pl = {"texto": f"{cab}\n{txt}", "texto_raw": txt, "clave": d["clave"], "titulo": titulo,
                  "materia_nom": d["materia"], "dependencia": nota.get("codOrgaDos") or "", "fecha_dof": nota["fecha"],
                  "es_modificacion": n["mod"], "cod_nota": n["cod"], "numeral": numeral, "chunk_index": i,
                  "ref": d["clave"] + (f", numeral {numeral}" if numeral and numeral != "0" else ""),
                  "ley": d["clave"], "origen": f"{d['clave']} — {corto}", "pdf_url": url,
                  "pdf_oficial": f"https://www.dof.gob.mx/nota_detalle.php?codigo={n['cod']}&fecha={nota['fecha'].replace('-', '/')}",
                  "entidad": "FEDERAL", "jurisdiccion": "federal", "source_type": "nom", "ingesta": "nom-2026-10b"}
            pid = str(uuid.UUID(hashlib.md5(f"nom|{n['cod']}|{i}|{h}".encode()).hexdigest()))
            fuera.append((pid, pl["texto"][:7500], pl))
    return fuera


def vectores():
    import requests
    import time
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchAny, PointStruct
    e = env()
    qc = QdrantClient(url=e["QDRANT_URL"], api_key=e["QDRANT_API_KEY"], timeout=120)
    p = json.loads(PLAN.read_text())
    hechos = set(SUBIDOS.read_text().split()) if SUBIDOS.exists() else set()
    tokens, total = 0, 0
    for d in p["vigentes"]:
        if d["clave"] in hechos:
            continue
        filas = filas_de(d)
        for i in range(0, len(filas), 96):
            lote = filas[i:i + 96]
            for k in range(6):
                r = requests.post("https://api.openai.com/v1/embeddings", timeout=180,
                                  headers={"Authorization": f"Bearer {e['OPENAI_API_KEY']}"},
                                  json={"model": "text-embedding-3-small", "input": [t for _, t, _ in lote]})
                if r.status_code == 200:
                    break
                time.sleep(5 + 5 * k)
            r.raise_for_status()
            tokens += r.json()["usage"]["total_tokens"]
            vec = [x["embedding"] for x in r.json()["data"]]
            qc.upsert(base.COLECCION, wait=True, points=[PointStruct(id=pid, vector={"dense": v}, payload=pl)
                                                         for (pid, _, pl), v in zip(lote, vec)])
        total += len(filas)
        with SUBIDOS.open("a") as f:
            f.write(d["clave"] + "\n")
        if len(hechos) % 50 == 0:
            print(f"  {d['clave']} · trozos acumulados {total} · tokens {tokens}", flush=True)
        hechos.add(d["clave"])
    quitar = sorted(set(p["sustituidas"]) | set(p["canceladas"]))
    for i in range(0, len(quitar), 50):
        qc.delete(base.COLECCION, points_selector=Filter(must=[FieldCondition(key="clave", match=MatchAny(any=quitar[i:i + 50]))]), wait=True)
    print(f"FIN: {qc.count(base.COLECCION, exact=True).count} puntos; nuevos {total}; tokens {tokens} ≈ USD {tokens * 0.02 / 1e6:.3f}; "
          f"quitadas {len(quitar)} claves sustituidas o canceladas", flush=True)


if __name__ == "__main__":
    if "--plan" in sys.argv:
        plan()
    if "--notas" in sys.argv:
        bajar_y_pdf()
    if "--gcs" in sys.argv:
        subir_gcs()
    if "--vectores" in sys.argv:
        vectores()
