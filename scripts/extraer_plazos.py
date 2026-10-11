"""Tabla de PLAZOS VERIFICADOS de la normativa procesal federal (fase 2, 11-oct-2026).

Por cada artículo de `procesal_federal` que menciona un plazo, gpt-6-luna
(esfuerzo bajo) extrae los plazos en forma estructurada: qué acto, cuánto,
hábiles o naturales, desde cuándo corre. NADA entra sin pasar el control:
el `texto_literal` del plazo tiene que estar, letra por letra (sin acentos ni
mayúsculas), en el artículo, y su número tiene que ser el que dice
`cantidad`. Lo que no pasa se descarta y se cuenta.

Salida: datos/plazos_procesales.json  {"<ABREV>|<ref>": [plazo, …]}
Caché reanudable: reingesta/plazos/crudo.jsonl
"""
import json
import re
import sys
import threading
import time
import concurrent.futures as cf
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ)); sys.path.insert(0, str(RAIZ / "scripts"))
import procesal_federal as pf  # noqa: E402
from ingesta_procesal_federal import env  # noqa: E402

CACHE = Path("/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/reingesta/plazos/crudo.jsonl")
SALIDA = RAIZ / "datos" / "plazos_procesales.json"
MODELO = "gpt-6-luna"

NUMEROS = {"un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7,
           "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
           "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiuno": 21,
           "veinticinco": 25, "treinta": 30, "cuarenta": 40, "cuarenta y cinco": 45, "cuarenta y ocho": 48,
           "sesenta": 60, "setenta y dos": 72, "noventa": 90, "ciento veinte": 120, "ciento ochenta": 180,
           "trescientos sesenta y cinco": 365}

INSTRUCCION = """Eres un extractor de PLAZOS PROCESALES del derecho mexicano. Recibes artículos (ley y número) y devuelves SOLO los plazos que el artículo FIJA para un acto procesal (presentar, interponer, contestar, resolver, notificar, celebrar audiencia, cumplir, etc.).
Reglas:
- No inventes: si el artículo no fija la cantidad, no lo incluyas. No incluyas plazos de prescripción sustantiva, penas ni edades.
- "texto_literal" debe copiarse EXACTO del artículo: la cantidad con su unidad, p. ej. "quince días", "setenta y dos horas", "3 días hábiles".
- "tipo_dias": "habiles" o "naturales" sólo si el artículo lo dice o remite expresamente a días hábiles/naturales; si no lo dice, "no_especifica".
- "computo": desde cuándo corre, en pocas palabras y sólo si el artículo lo dice (p. ej. "día siguiente a que surta efectos la notificación"); si no, "".
- "acto": quién hace qué, en una línea (p. ej. "Interponer apelación contra sentencia definitiva").
Responde SOLO con un objeto JSON con este formato: {"plazos":[{"articulo":"<ref tal cual se te dio>","acto":"","cantidad":15,"unidad":"dias|horas|meses|anos","tipo_dias":"habiles|naturales|no_especifica","computo":"","texto_literal":""}]}"""


def num_de(literal: str):
    t = pf.normalizar(literal)
    m = re.search(r"\d+", t)
    if m:
        return int(m.group(0))
    for palabra in sorted(NUMEROS, key=len, reverse=True):
        if re.search(rf"\b{palabra}\b", t):
            return NUMEROS[palabra]
    return None


def articulos():
    from qdrant_client import QdrantClient
    e = env()
    qc = QdrantClient(url=e["QDRANT_URL"], api_key=e["QDRANT_API_KEY"], timeout=120)
    por_art, off = {}, None
    while True:
        pts, off = qc.scroll(pf.COLECCION, limit=500, offset=off,
                             with_payload=["abreviatura", "ref", "texto_raw", "plazos", "chunk_index", "es_transitorio"])
        for p in pts:
            pl = p.payload
            if pl.get("es_transitorio"):
                continue
            k = f"{pl['abreviatura']}|{pl['ref']}"
            d = por_art.setdefault(k, {"tiene": False, "trozos": []})
            d["tiene"] |= bool(pl.get("plazos"))
            d["trozos"].append((pl.get("chunk_index", 0), pl.get("texto_raw", "")))
        if off is None:
            break
    return {k: "\n".join(t for _, t in sorted(v["trozos"])) for k, v in por_art.items() if v["tiene"]}


def llamar(clave_api, lote):
    import requests
    cuerpo = "\n\n".join(f"### {k.split('|')[0]} — {k.split('|')[1]}\n{txt[:6000]}" for k, txt in lote)
    for i in range(5):
        r = requests.post("https://api.openai.com/v1/chat/completions", timeout=180,
                          headers={"Authorization": f"Bearer {clave_api}"},
                          json={"model": MODELO, "reasoning_effort": "low", "max_completion_tokens": 6000,
                                "response_format": {"type": "json_object"},
                                "messages": [{"role": "system", "content": INSTRUCCION},
                                             {"role": "user", "content": cuerpo}]})
        if r.status_code == 200:
            d = r.json()
            return json.loads(d["choices"][0]["message"]["content"] or "{}"), d.get("usage", {})
        time.sleep(4 + 6 * i)
    raise RuntimeError(f"{r.status_code} {r.text[:200]}")


def main():
    e = env()
    arts = articulos()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    hechos = set()
    if CACHE.exists():
        for l in CACHE.read_text().splitlines():
            hechos.update(json.loads(l)["claves"])
    pend = [(k, v) for k, v in sorted(arts.items()) if k not in hechos]
    print("artículos con plazo", len(arts), "pendientes", len(pend), flush=True)
    lotes = [pend[i:i + 5] for i in range(0, len(pend), 5)]
    lock = threading.Lock()
    uso = {"in": 0, "out": 0}

    def uno(lote):
        res, u = llamar(e["OPENAI_API_KEY"], lote)
        with lock:
            uso["in"] += u.get("prompt_tokens", 0); uso["out"] += u.get("completion_tokens", 0)
            with CACHE.open("a") as f:
                f.write(json.dumps({"claves": [k for k, _ in lote], "res": res}, ensure_ascii=False) + "\n")

    with cf.ThreadPoolExecutor(6) as ex:
        for i, _ in enumerate(ex.map(uno, lotes)):
            if i % 40 == 0:
                print(f"  lote {i}/{len(lotes)} tokens in {uso['in']} out {uso['out']}", flush=True)

    # Verificación y tabla final
    tabla, ok, fuera = {}, 0, 0
    for l in CACHE.read_text().splitlines():
        d = json.loads(l)
        claves = d["claves"]
        for p in (d["res"] or {}).get("plazos", []) or []:
            art = str(p.get("articulo", "")).strip()
            k = next((c for c in claves if c.split("|")[1].strip().lower() == art.lower()), None)
            if k is None and len(claves) == 1:
                k = claves[0]
            if k is None:
                k = next((c for c in claves if re.sub(r"\D", "", c.split("|")[1]) == re.sub(r"\D", "", art)), None)
            lit = str(p.get("texto_literal", ""))
            if (k is None or k not in arts or not lit or pf.normalizar(lit) not in pf.normalizar(arts[k])
                    or num_de(lit) != p.get("cantidad") or p.get("unidad") not in ("dias", "horas", "meses", "anos")):
                fuera += 1
                continue
            if p.get("tipo_dias") not in ("habiles", "naturales", "no_especifica"):
                p["tipo_dias"] = "no_especifica"
            if p["tipo_dias"] == "habiles" and "habil" not in pf.normalizar(arts[k]):
                p["tipo_dias"] = "no_especifica"
            if p["tipo_dias"] == "naturales" and "natural" not in pf.normalizar(arts[k]):
                p["tipo_dias"] = "no_especifica"
            ok += 1
            tabla.setdefault(k, []).append({c: p.get(c) for c in ("acto", "cantidad", "unidad", "tipo_dias", "computo", "texto_literal")})
    SALIDA.parent.mkdir(exist_ok=True)
    SALIDA.write_text(json.dumps(tabla, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
    print(f"FIN: {ok} plazos verificados en {len(tabla)} artículos; {fuera} descartados por el control. "
          f"tokens in {uso['in']} out {uso['out']} ≈ USD {uso['in'] * 0.10 / 1e6 + uso['out'] * 0.50 / 1e6:.3f}", flush=True)


if __name__ == "__main__":
    main()
