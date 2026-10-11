"""Normas Oficiales Mexicanas a la colección `normas_oficiales` (11-oct-2026).

No todas: las que un abogado usa para acreditar un estándar exigible —el
expediente clínico en una negligencia médica, la NOM-046 en violencia sexual,
la NOM-035 y la de teletrabajo en lo laboral, la NOM-151 para la conservación
de mensajes de datos como prueba, las de prácticas comerciales en consumo, las
de tránsito pesado en un accidente, las ambientales en un amparo—.

Fuente: la API pública del SIDOF (Segob), que devuelve la nota del DOF con su
HTML íntegro (`/dof/sidof/notas/nota/<codNota>`). El índice de qué nota es cada
NOM lo armó `reingesta/nom/indice_sidof.py` recorriendo el DOF día por día.

Por cada nota: texto troceado por numeral (≈2,500 car.), PDF hecho con Chrome
desde el HTML del DOF (con su fecha y enlace oficial en la cabecera) en
gs://iurexia-leyes/nom/, y vectores text-embedding-3-small (los del chat).
"""
import hashlib
import html as H
import json
import re
import subprocess
import sys
import time
import urllib.request
import uuid
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingesta_procesal_federal import env  # noqa: E402

DIR = Path("/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/reingesta/nom")
COLECCION = "normas_oficiales"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/129.0 Safari/537.36"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# clave → (materia, [codNota…]) — la primera nota es la NOM; las demás, sus modificaciones vigentes.
SELECCION = {
    "NOM-035-STPS-2018": ("trabajo", [5541828]),
    "NOM-037-STPS-2023": ("trabajo", [5691672]),
    "NOM-036-1-STPS-2018": ("trabajo", [5544579]),
    "NOM-030-STPS-2009": ("trabajo", [5125949]),
    "NOM-019-STPS-2011": ("trabajo", [5185903]),
    "NOM-017-STPS-2008": ("trabajo", [5072773]),
    "NOM-002-STPS-2010": ("trabajo", [5170410]),
    "NOM-009-STPS-2011": ("trabajo", [5188396]),
    "NOM-001-STPS-2008": ("trabajo", [5069976]),
    "NOM-004-SSA3-2012": ("salud", [5272787]),
    "NOM-046-SSA2-2005": ("salud", [5087256, 5430957]),
    "NOM-007-SSA2-2016": ("salud", [5432289]),
    "NOM-024-SSA3-2012": ("salud", [5280847]),
    "NOM-028-SSA2-2009": ("salud", [5106313]),
    "NOM-047-SSA2-2015": ("salud", [5403545]),
    "NOM-005-SSA3-2018": ("salud", [5596456]),
    "NOM-006-SSA3-2011": ("salud", [5240668]),
    "NOM-016-SSA3-2012": ("salud", [5284306]),
    "NOM-025-SSA2-2014": ("salud", [5406383]),
    "NOM-027-SSA3-2013": ("salud", [5312893]),
    "NOM-034-SSA3-2013": ("salud", [5361072]),
    "NOM-035-SSA3-2012": ("salud", [5280848]),
    "NOM-087-SEMARNAT-SSA1-2002": ("salud", [704675]),
    "NOM-253-SSA1-2012": ("salud", [5275587]),
    "NOM-010-SSA2-2010": ("salud", [5166864]),
    "NOM-127-SSA1-2021": ("salud", [5650705]),
    "NOM-151-SCFI-2016": ("consumo", [5478024]),
    "NOM-247-SE-2021": ("consumo", [5646251]),
    "NOM-029-SE-2021": ("consumo", [5646979]),
    "NOM-174-SCFI-2007": ("consumo", [5005193]),
    "NOM-050-SCFI-2004": ("consumo", [708514]),
    "NOM-051-SCFI/SSA1-2010": ("consumo", [5590668, 5137518, 5595259]),
    "NOM-122-SCFI-2010": ("consumo", [5141411]),
    "NOM-160-SCFI-2003": ("consumo", [688971]),
    "NOM-024-SCFI-2013": ("consumo", [5309980]),
    "NOM-028-SCFI-2007": ("consumo", [5005191]),
    "NOM-002-SCFI-2011": ("consumo", [5263188]),
    "NOM-012-SCT-2-2017": ("transporte", [5508944]),
    "NOM-087-SCT-2-2017": ("transporte", [5529381]),
    "NOM-068-SCT-2-2014": ("transporte", [5378850, 5378852]),
    "NOM-034-SCT2-2011": ("transporte", [5220002]),
    "NOM-059-SEMARNAT-2010": ("ambiental", [5173091]),
    "NOM-081-SEMARNAT-1994": ("ambiental", [4866673, 5324105]),
    "NOM-001-SEMARNAT-2021": ("ambiental", [5645374]),
    "NOM-052-SEMARNAT-2005": ("ambiental", [4912592]),
    "NOM-083-SEMARNAT-2003": ("ambiental", [658648]),
    "NOM-161-SEMARNAT-2011": ("ambiental", [5286505]),
    "NOM-138-SEMARNAT/SSA1-2012": ("ambiental", [5313544]),
    "NOM-041-SEMARNAT-2015": ("ambiental", [5396063]),
    "NOM-045-SEMARNAT-2017": ("ambiental", [5515481]),
    "NOM-011-CONAGUA-2015": ("agua", [5387027]),
    "NOM-003-SEGOB-2011": ("proteccion_civil", [5226545, 5400626]),
}
TOPE_TROZOS_POR_NOTA = 400


def nota(cod):
    f = DIR / "html" / f"{cod}.json"
    if f.exists():
        return json.loads(f.read_text())
    f.parent.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        try:
            req = urllib.request.Request(f"https://sidof.segob.gob.mx/dof/sidof/notas/nota/{cod}",
                                         headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read().decode("utf-8", "replace"))["Nota"]
            f.write_text(json.dumps(d, ensure_ascii=False))
            return d
        except Exception:
            time.sleep(3 + 4 * i)
    raise RuntimeError(f"nota {cod}")


class _Texto(HTMLParser):
    BLOQUE = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "table", "title"}

    def __init__(self):
        super().__init__()
        self.partes, self.saltar = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.saltar += 1
        elif tag in self.BLOQUE:
            self.partes.append("\n")
        elif tag == "td":
            self.partes.append(" | ")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.saltar = max(0, self.saltar - 1)
        elif tag in self.BLOQUE:
            self.partes.append("\n")

    def handle_data(self, data):
        if not self.saltar:
            self.partes.append(data)


def texto_de(html):
    p = _Texto()
    p.feed(html)
    t = H.unescape("".join(p.partes)).replace("\xa0", " ")
    lineas = [re.sub(r"[ \t]+", " ", l).strip(" |") for l in t.splitlines()]
    return [l for l in lineas if l]


RE_NUMERAL = re.compile(r"^(\d{1,2}(?:\.\d{1,3}){0,4})\.?\s+\S")


def trocear(lineas, maximo=2500):
    trozos, actual, numeral, n_actual = [], [], "", ""
    largo = 0
    for l in lineas:
        m = RE_NUMERAL.match(l)
        es_titulo = bool(m) and len(l) < 180
        if es_titulo and largo > 1200:
            trozos.append((n_actual, "\n".join(actual)))
            actual, largo = [], 0
        if es_titulo:
            numeral = m.group(1)
            if not actual:
                n_actual = numeral
        if not actual:
            n_actual = numeral
        actual.append(l)
        largo += len(l) + 1
        if largo > maximo * 1.6:
            trozos.append((n_actual, "\n".join(actual)))
            actual, largo = [], 0
    if actual:
        trozos.append((n_actual, "\n".join(actual)))
    return trozos


def pdf_de(cod, d):
    destino = DIR / "pdf" / f"{cod}.pdf"
    if destino.exists() and destino.stat().st_size > 2000:
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    fecha = d["fecha"].replace("-", "/")
    oficial = f"https://www.dof.gob.mx/nota_detalle.php?codigo={cod}&fecha={fecha}"
    banner = (f'<div style="font:12px Arial;border-bottom:1px solid #999;margin-bottom:12px;padding-bottom:6px">'
              f'<b>Diario Oficial de la Federación</b> — publicación del {d["fecha"]} — '
              f'{H.escape(d.get("codOrgaDos") or "")}<br>Fuente oficial: {oficial}</div>')
    cuerpo = d["cadenaContenido"]
    cuerpo = re.sub(r"(<head[^>]*>)", r'\1<base href="https://www.dof.gob.mx/"><meta charset="utf-8">', cuerpo, 1, flags=re.I)
    cuerpo = re.sub(r"(<body[^>]*>)", r"\1" + banner, cuerpo, 1, flags=re.I) if re.search(r"<body", cuerpo, re.I) \
        else banner + cuerpo
    fh = DIR / "pdf" / f"{cod}.html"
    fh.write_text(cuerpo, encoding="utf-8")
    p = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                          f"--print-to-pdf={destino}", fh.as_uri()],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(180):
        if p.poll() is not None:
            break
        time.sleep(1)
    if p.poll() is None:
        p.kill()
    return destino if destino.exists() and destino.stat().st_size > 2000 else None


def slug(clave):
    return re.sub(r"[^A-Za-z0-9-]+", "-", clave)


def main():
    e = env()
    filas, pdfs = [], []
    vistos = set()
    for clave, (materia, cods) in SELECCION.items():
        for orden, cod in enumerate(cods):
            d = nota(cod)
            if not d.get("cadenaContenido"):
                print(f"{clave:28s} {cod} SIN TEXTO en el DOF (sólo imagen): se omite", flush=True)
                continue
            titulo = re.sub(r"\s+", " ", d.get("titulo") or "").strip()
            lineas = texto_de(d["cadenaContenido"])
            trozos = trocear(lineas)[:TOPE_TROZOS_POR_NOTA]
            pdf = pdf_de(cod, d)
            url = None
            if pdf:
                url = f"https://storage.googleapis.com/iurexia-leyes/nom/{slug(clave)}__{cod}.pdf"
                pdfs.append((pdf, f"nom/{slug(clave)}__{cod}.pdf"))
            corto = re.sub(r"^.*?" + re.escape(clave.split("/")[0][:8]), clave.split("/")[0][:8], titulo)[:220]
            for i, (numeral, txt) in enumerate(trozos):
                h = hashlib.md5(txt.encode()).hexdigest()
                if h in vistos or len(txt) < 40:
                    continue
                vistos.add(h)
                cab = f"[{clave} | {corto}" + (f" | numeral {numeral}" if numeral else "") + "]"
                pl = {
                    "texto": f"{cab}\n{txt}", "texto_raw": txt,
                    "clave": clave, "titulo": titulo, "materia_nom": materia,
                    "dependencia": d.get("codOrgaDos") or "", "fecha_dof": d["fecha"],
                    "es_modificacion": orden > 0, "cod_nota": cod, "numeral": numeral, "chunk_index": i,
                    "ref": f"{clave}" + (f", numeral {numeral}" if numeral else ""),
                    "ley": clave, "origen": f"{clave} — {corto}",
                    "pdf_url": url,
                    "pdf_oficial": f"https://www.dof.gob.mx/nota_detalle.php?codigo={cod}&fecha={d['fecha'].replace('-', '/')}",
                    "entidad": "FEDERAL", "jurisdiccion": "federal", "source_type": "nom", "ingesta": "nom-2026-10",
                }
                pid = str(uuid.UUID(hashlib.md5(f"nom|{cod}|{i}|{h}".encode()).hexdigest()))
                filas.append((pid, pl["texto"][:7500], pl))
            print(f"{clave:28s} {cod} trozos {len(trozos):4d} pdf {'sí' if pdf else 'NO'}", flush=True)
    print("TOTAL trozos", len(filas), "PDF", len(pdfs), flush=True)
    if "--seco" in sys.argv:
        return

    try:
        from google.cloud import storage
        from google.oauth2 import service_account
        info = json.loads(e["GCP_SA_KEY_JSON"])
        b = storage.Client(credentials=service_account.Credentials.from_service_account_info(info),
                           project=info.get("project_id")).bucket("iurexia-leyes")
        for ruta, nombre in pdfs:
            bl = b.blob(nombre)
            bl.cache_control = "public, max-age=86400"
            bl.upload_from_filename(str(ruta), content_type="application/pdf")
        print("PDF en GCS:", len(pdfs), flush=True)
    except ImportError:
        print("sin google-cloud-storage: correr con --solo-pdfs en otro python", flush=True)
    if "--solo-pdfs" in sys.argv:
        return

    import requests
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams
    qc = QdrantClient(url=e["QDRANT_URL"], api_key=e["QDRANT_API_KEY"], timeout=120)
    if not qc.collection_exists(COLECCION):
        qc.create_collection(COLECCION, vectors_config={"dense": VectorParams(
            size=1536, distance=Distance.COSINE, on_disk=True)}, on_disk_payload=True)
        for campo in ("clave", "materia_nom", "dependencia"):
            qc.create_payload_index(COLECCION, campo, PayloadSchemaType.KEYWORD)
    tokens = 0
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
        qc.upsert(COLECCION, wait=True, points=[PointStruct(id=pid, vector={"dense": v}, payload=pl)
                                                for (pid, _, pl), v in zip(lote, vec)])
        if (i // 96) % 10 == 0:
            print(f"  {i + len(lote)}/{len(filas)} tokens {tokens}", flush=True)
    print(f"FIN {qc.count(COLECCION, exact=True).count} puntos, tokens {tokens} ≈ USD {tokens * 0.02 / 1e6:.4f}", flush=True)


if __name__ == "__main__":
    main()
