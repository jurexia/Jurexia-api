#!/usr/bin/env python3
"""
Pasa a disco los índices `articulo_num` que quedaron en memoria.

POR QUÉ
-------
Once colecciones traían el índice de una ingesta anterior, creado con los
valores por omisión: en RAM y con estructura de rango. El nodo tiene 4 GiB y
el salto por grafo NUNCA consulta rangos —siempre es igualdad exacta
(`articulo_num == 47`)—, así que esa memoria está pagada de más.

Al recrearlo con `on_disk: true`, `lookup: true` y `range: false`, el índice
baja al disco y pierde la mitad que no se usa.

SEGURO PORQUE
-------------
Es un índice, no un dato: el campo `articulo_num` no se toca. Mientras se
recrea, un filtro por ese campo simplemente va más lento (barrido); no
devuelve resultados equivocados. Se hace de una en una, verificando el
recuento antes y después, y midiendo la memoria del nodo en cada paso.

USO
---
    python indices_a_disco.py            # en seco: dice qué haría
    python indices_a_disco.py --escribir
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ENV = RAIZ / ".env"      # el .env vive en la raíz del API; NO va al repo
TECHO_RAM_GIB = 3.20
PAUSA = 1.0


def cargar_env():
    datos = {}
    for linea in ENV.read_text(encoding="utf-8", errors="ignore").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#") and "=" in linea:
            c, v = linea.split("=", 1)
            datos[c] = v.strip().strip('"').strip("'")
    return datos["QDRANT_URL"].rstrip("/"), datos["QDRANT_API_KEY"]


def pedir(url, api, ruta, cuerpo=None, metodo=None):
    d = json.dumps(cuerpo).encode() if cuerpo is not None else None
    r = urllib.request.Request(
        url + ruta, data=d,
        headers={"api-key": api, "Content-Type": "application/json"},
        method=metodo or ("POST" if d else "GET"))
    for i in range(4):
        try:
            with urllib.request.urlopen(r, timeout=120) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if i == 3:
                raise
            time.sleep(2 * (i + 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--escribir", action="store_true")
    args = p.parse_args()
    url, api = cargar_env()

    def mem():
        m = pedir(url, api, "/telemetry?details_level=1")["result"]["memory"]
        return m["resident_bytes"] / 2 ** 30

    cols = sorted(c["name"] for c in
                  pedir(url, api, "/collections")["result"]["collections"]
                  if c["name"].startswith("leyes_"))

    pendientes = []
    for c in cols:
        esq = (pedir(url, api, f"/collections/{c}")["result"]
               .get("payload_schema") or {}).get("articulo_num")
        if esq and not (esq.get("params") or {}).get("on_disk"):
            pendientes.append((c, esq.get("points", 0)))

    modo = "ESCRITURA" if args.escribir else "EN SECO"
    print(f"\nÍndices articulo_num en RAM → disco · {modo}")
    print("─" * 66)
    if not pendientes:
        print("No queda ninguno en RAM.")
        return

    total = sum(n for _, n in pendientes)
    print(f"{len(pendientes)} colecciones · {total:,} puntos indexados en memoria")
    for c, n in pendientes:
        print(f"   {c:28} {n:>8,}")

    if not args.escribir:
        print("\nRelanza con --escribir.")
        return

    antes = mem()
    print(f"\nRAM antes: {antes:.3f} GiB")

    for c, n in pendientes:
        actual = mem()
        if actual >= TECHO_RAM_GIB:
            print(f"  ⛔ ALTO: {actual:.2f} GiB (techo {TECHO_RAM_GIB}). "
                  f"Se detiene antes de {c}.")
            break
        # Borrar y recrear. El campo no se toca en ningún momento.
        pedir(url, api, f"/collections/{c}/index/articulo_num?wait=true",
              None, metodo="DELETE")
        time.sleep(PAUSA)
        pedir(url, api, f"/collections/{c}/index?wait=true",
              {"field_name": "articulo_num",
               "field_schema": {"type": "integer", "on_disk": True,
                                "lookup": True, "range": False}},
              metodo="PUT")
        esq = (pedir(url, api, f"/collections/{c}")["result"]
               .get("payload_schema") or {}).get("articulo_num") or {}
        ok = (esq.get("params") or {}).get("on_disk") and esq.get("points", 0) >= n * 0.99
        print(f"   {c:28} {esq.get('points', 0):>8,} indexados  "
              f"{'✔ en disco' if ok else '⚠️ revisar'}  RAM={mem():.3f} GiB", flush=True)
        time.sleep(PAUSA)

    despues = mem()
    print(f"\nRAM después: {despues:.3f} GiB  (diferencia {despues - antes:+.3f} GiB)")


if __name__ == "__main__":
    main()
