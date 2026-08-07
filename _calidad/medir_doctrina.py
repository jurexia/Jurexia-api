#!/usr/bin/env python3
"""
Mide si la biblioteca doctrinal responde a lo que un abogado pregunta.

QUÉ MIDE
--------
Preguntas conceptuales REALES —las que un litigante escribe cuando busca
doctrina— contra la colección `doctrina`, y cuenta:

  · si el primer resultado es del autor/obra esperados (acierto@1),
  · si el esperado aparece en el top-3 (acierto@3),
  · si los resultados traen página citable (impresa o de PDF),
  · que NINGÚN resultado sea de subtipo preliminares/bibliografía
    (el fantasma del «Art. 19»: portadas citadas como doctrina).

Recuento determinista, sin juez. Correr antes de abrir la Fase 2 y después
de cada tanda nueva: si el acierto baja, la tanda ensució el retrieval.

USO
---
    python medir_doctrina.py
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ENV = RAIZ / ".env"
COLECCION = "doctrina"
MODELO = "text-embedding-3-small"

# (pregunta, palabras que deben aparecer en autor u obra del top-3)
CASOS = [
    # Fix-Zamudio quedó pendiente de OCR (escaneo de 1993 sin capa de texto):
    # mientras no esté, el Diccionario DPC es la respuesta correcta para el
    # amparo. Cuando se ingiera, volver a exigirlo aquí.
    ("naturaleza jurídica del juicio de amparo mexicano", ["fix-zamudio", "diccionario", "ferrer"]),
    ("concepto de derechos fundamentales y sus garantías", ["carbonell", "diccionario"]),
    ("teorías de la argumentación jurídica de MacCormick y Alexy", ["atienza"]),
    ("qué es el derecho procesal constitucional", ["ferrer", "diccionario", "fix-zamudio"]),
    ("control difuso de constitucionalidad", ["diccionario", "ferrer", "fix-zamudio", "carbonell"]),
    ("control de convencionalidad ex officio", ["diccionario", "ferrer"]),
    ("el amparo contra leyes", ["fix-zamudio", "diccionario"]),
    ("interés legítimo en el proceso constitucional", ["diccionario", "ferrer", "fix-zamudio"]),
    ("la ponderación entre principios constitucionales", ["atienza", "carbonell", "diccionario"]),
    ("suspensión del acto reclamado", ["diccionario", "fix-zamudio"]),
    ("libertad de expresión como derecho fundamental", ["carbonell", "diccionario"]),
    ("el silogismo judicial y la justificación de las decisiones", ["atienza"]),
]


def cargar_env():
    datos = {}
    for l in ENV.read_text(encoding="utf-8", errors="ignore").splitlines():
        l = l.strip()
        if l and not l.startswith("#") and "=" in l:
            k, v = l.split("=", 1)
            datos[k] = v.strip().strip('"').strip("'")
    return datos


def pedir(url, cuerpo, cab):
    r = urllib.request.Request(url, data=json.dumps(cuerpo).encode(), headers=cab)
    for i in range(3):
        try:
            with urllib.request.urlopen(r, timeout=90) as resp:
                return json.load(resp)
        except Exception:
            if i == 2:
                raise
            time.sleep(2)


def main():
    env = cargar_env()
    qu, qk, ok = env["QDRANT_URL"].rstrip("/"), env["QDRANT_API_KEY"], env["OPENAI_API_KEY"]

    print(f"\nBanco doctrinal · {len(CASOS)} preguntas conceptuales")
    print("─" * 78)

    a1 = a3 = con_pagina = contaminados = 0
    latencias = []
    for pregunta, esperados in CASOS:
        v = pedir("https://api.openai.com/v1/embeddings",
                  {"model": MODELO, "input": pregunta},
                  {"Authorization": f"Bearer {ok}", "Content-Type": "application/json"}
                  )["data"][0]["embedding"]
        t0 = time.perf_counter()
        r = pedir(f"{qu}/collections/{COLECCION}/points/query",
                  {"query": v, "using": "dense", "limit": 3,
                   "filter": {"must": [{"key": "subtipo", "match": {"value": "doctrina"}}]},
                   "with_payload": ["autor", "obra", "pagina_impresa", "pagina_pdf", "subtipo"]},
                  {"api-key": qk, "Content-Type": "application/json"})
        latencias.append((time.perf_counter() - t0) * 1000)
        pts = r["result"]["points"]

        def casa(p):
            etiqueta = (str(p["payload"].get("autor", "")) + " " +
                        str(p["payload"].get("obra", ""))).lower()
            return any(e in etiqueta for e in esperados)

        top1 = bool(pts) and casa(pts[0])
        top3 = any(casa(p) for p in pts)
        pagina = bool(pts) and all(
            p["payload"].get("pagina_impresa") or p["payload"].get("pagina_pdf") for p in pts)
        sucio = any(p["payload"].get("subtipo") != "doctrina" for p in pts)

        a1 += top1; a3 += top3; con_pagina += pagina; contaminados += sucio
        marca = "✔" if top3 else "✘"
        quien = (pts[0]["payload"].get("autor", "—")[:26] if pts else "—")
        print(f"  {marca} @1={'sí' if top1 else 'no ':3} {pregunta[:50]:52} → {quien}")

    n = len(CASOS)
    print("─" * 78)
    print(f"acierto@1: {a1}/{n}   acierto@3: {a3}/{n}   con página citable: {con_pagina}/{n}")
    print(f"resultados contaminados (preliminares/bibliografía): {contaminados}")
    print(f"latencia de búsqueda: {sorted(latencias)[len(latencias)//2]:.0f} ms de mediana")


if __name__ == "__main__":
    main()
