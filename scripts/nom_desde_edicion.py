"""NOM que el DOF sólo publica como imagen: el texto sale del PDF de la edición (11-oct-2026).

Para cada nota vigente sin `cadenaContenido` (la modificación de 2020 de la
NOM-051 —los sellos—, el listado de materiales peligrosos NOM-002-SCT-SEMAR-
ARTF-2023, partes de la NOM-001-SEDE-2012…):
  1. las notas del mismo día (SIDOF /notas/<dd-mm-aaaa>) dan la página donde
     empieza esta y donde empieza la siguiente de la misma edición;
  2. el PDF de la edición (SIDOF /documentos/pdf/<codDiario>) se recorta a esas
     páginas: texto → html/<cod>.json (cadenaContenido sintético) y PDF → pdf/<cod>.pdf;
  3. se borra la clave de subidos.txt para que --gcs y --vectores la rehagan.
"""
import html
import json
import re
import urllib.request
from pathlib import Path

import fitz

DIR = Path("/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/reingesta/nom")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/129.0 Safari/537.36"


def pedir(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=120).read()


def main():
    plan = json.loads((DIR / "plan_nom_completa.json").read_text())
    rehacer = set()
    for d in plan["vigentes"]:
        for n in d["notas"]:
            f = DIR / "html" / f"{n['cod']}.json"
            if not f.exists():
                continue
            nota = json.loads(f.read_text())
            if nota.get("cadenaContenido"):
                continue
            dia = json.loads(pedir(f"https://sidof.segob.gob.mx/dof/sidof/notas/{nota['fecha']}"))
            misma = sorted([x for k in ("NotasMatutinas", "NotasVespertinas", "NotasExtraordinarias")
                            for x in dia.get(k) or [] if str(x.get("codDiario")) == str(nota["codDiario"])],
                           key=lambda x: int(x.get("pagina") or 0))
            p0 = int(nota.get("pagina") or 1)
            siguientes = [int(x["pagina"]) for x in misma if int(x.get("pagina") or 0) > p0]
            ed = DIR / "ediciones" / f"{nota['codDiario']}.pdf"
            ed.parent.mkdir(exist_ok=True)
            if not ed.exists():
                ed.write_bytes(pedir(f"https://sidof.segob.gob.mx/dof/sidof/documentos/pdf/{nota['codDiario']}"))
            doc = fitz.open(str(ed))
            p1 = (siguientes[0] - 1) if siguientes else doc.page_count
            # la página impresa n suele ser el índice n-1; si el título no está ahí, se busca cerca
            clave_corta = re.sub(r"\s", "", d["clave"].split("-")[1])
            ini = p0 - 1
            for delta in (0, 1, -1, 2, -2):
                k = p0 - 1 + delta
                if 0 <= k < doc.page_count and clave_corta in re.sub(r"\s", "", doc[k].get_text()):
                    ini = k
                    break
            fin = max(ini, min(doc.page_count - 1, p1 - 1 + (ini - (p0 - 1))))
            texto = "\n".join(doc[i].get_text() for i in range(ini, fin + 1))
            if len(texto) < 500:
                print("SIN TEXTO", d["clave"], n["cod"])
                continue
            nota["cadenaContenido"] = "<html><body>" + "".join(
                f"<p>{html.escape(l)}</p>" for l in texto.splitlines() if l.strip()) + "</body></html>"
            nota["_origen_texto"] = f"PDF de la edición DOF {nota['fecha']} (codDiario {nota['codDiario']}), págs. {ini + 1}-{fin + 1}"
            f.write_text(json.dumps(nota, ensure_ascii=False))
            corte = fitz.open()
            corte.insert_pdf(doc, from_page=ini, to_page=fin)
            corte.save(str(DIR / "pdf" / f"{n['cod']}.pdf"))
            rehacer.add(d["clave"])
            print(f"{d['clave']:30s} {n['cod']} págs {ini + 1}-{fin + 1} · {len(texto):,} car.", flush=True)
    sub = DIR / "subidos.txt"
    if sub.exists():
        sub.write_text("\n".join(k for k in sub.read_text().split() if k not in rehacer) + "\n")
    print("a rehacer:", sorted(rehacer))


if __name__ == "__main__":
    main()
