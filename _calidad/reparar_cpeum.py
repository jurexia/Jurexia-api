#!/usr/bin/env python3
"""
Repara el etiquetado de la Constitución en `bloque_constitucional`.

EL DAÑO
-------
De los 355 fragmentos de la CPEUM, 89 estaban etiquetados «Art. 19 CPEUM».
El artículo 19 real son CUATRO de ellos. Los otros 85 —392,195 caracteres,
doce veces el artículo 123, que es el más largo de verdad— son la cola del
documento: la firma del Congreso Constituyente de 1917, la lista de diputados
y los artículos transitorios de TODOS los decretos de reforma.

La ingesta troceó por encabezado de artículo y, al llegar al final del
articulado, siguió metiendo texto en el último cajón que tenía abierto.

POR QUÉ IMPORTA
---------------
Un abogado que pregunte por el artículo 19 —prisión preventiva, auto de
vinculación— podía recibir la lista de diputados de 1917 citada como si fuera
ese artículo. Es una cita falsa con apariencia de norma.

QUÉ HACE ESTE ARREGLO
---------------------
NO borra nada: los transitorios son derecho vigente y ahí vive la información
de entrada en vigor, que no está en ninguna otra parte del corpus. Lo que hace
es dejar de mentir sobre qué son:

  · Los 85 de la cola pierden el «Art. 19» y pasan a citarse por lo que son:
    «CPEUM · Transitorios de reformas» o «CPEUM · Firmas del Constituyente
    (1917)». Se les marca `subtipo` para poder ordenarlos por debajo del
    articulado.
  · Los fragmentos del articulado reciben `articulo_num`, que esta colección
    nunca tuvo, más su índice: así el salto entre artículos también funciona
    en la Constitución.

USO
---
    python reparar_cpeum.py             # en seco
    python reparar_cpeum.py --escribir
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ENV = RAIZ / ".env"
COLECCION = "bloque_constitucional"
LOTE = 20
PAUSA = 0.35
TECHO_RAM_GIB = 3.20

# El artículo 19 de verdad habla de esto y de nada más.
RE_ART19_REAL = re.compile(
    r'(setenta y dos horas|auto de vinculaci[óo]n a proceso|prisi[óo]n preventiva'
    r'|delito que se le imputa|plazo constitucional)', re.I)

# La cola: firmas del Constituyente y transitorios de decretos.
RE_FIRMAS = re.compile(
    r'(Sal[óo]n de Sesiones del Congreso Constituyente|Diputado[s]? por el|R[úu]brica'
    r'|Presidente:\s|Primer Secretario|Segundo Secretario)', re.I)
RE_TRANSITORIO = re.compile(
    r'\b(PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|S[ÉE]PTIMO|OCTAVO|NOVENO'
    r'|D[ÉE]CIMO|UND[ÉE]CIMO|DUOD[ÉE]CIMO|TRANSITORIOS?)\b\s*[\.\-–]', re.I)
RE_DECRETO = re.compile(
    r'(DECRETO por el que|se reforman|se adicionan?|se derogan?'
    r'|Diario Oficial de la Federaci[óo]n|entrar[áa]n? en vigor)', re.I)

RE_REF_ART = re.compile(r'^\s*art\.?\s*(\d{1,3})\s*[o°º]?', re.I)


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


def clasificar(payload: dict) -> tuple[str, int | None]:
    """→ ('articulado' | 'transitorios' | 'firmas', número de artículo)"""
    ref = str(payload.get("ref") or "")
    texto = str(payload.get("texto") or "")
    m = RE_REF_ART.match(ref)
    num = int(m.group(1)) if m else None

    # Idempotencia: lo ya clasificado en una corrida anterior se respeta. Sin
    # esto, un fragmento marcado «firmas» volvía a caer en «transitorios» al
    # relanzar, porque su ref ya no empieza por «Art. 19» y pierde el número.
    ya = payload.get("subtipo")
    if ya in ("firmas", "transitorios"):
        return ya, None
    if ya == "articulado":
        return "articulado", (payload.get("articulo_num") if isinstance(payload.get("articulo_num"), int) else num)

    # La sospecha se limita AL CAJÓN DEL 19, que es el único contaminado: tiene
    # 401,029 caracteres contra los 33,621 del artículo 123, que es el más
    # largo de verdad. El resto del articulado guarda proporciones razonables.
    #
    # Aplicar la regla a toda la Constitución fue un error que se midió: nueve
    # artículos reales se perdían, porque el articulado también dice «se
    # reforma» o «se deroga» al hablar de las facultades del Congreso.
    #
    # Dentro del cajón del 19, en cambio, manda la marca de transitorio. La
    # primera versión preguntaba antes si sonaba al artículo y se le coló un
    # transitorio de la Guardia Nacional: esos también hablan de prisión
    # preventiva.
    if num == 19:
        if RE_FIRMAS.search(texto):
            return "firmas", None
        if RE_TRANSITORIO.search(texto) or RE_DECRETO.search(texto):
            return "transitorios", None
        if not RE_ART19_REAL.search(texto):
            return "transitorios", None

    return ("articulado", num) if num else ("transitorios", None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--escribir", action="store_true")
    args = p.parse_args()
    url, api = cargar_env()

    def ram():
        m = pedir(url, api, "/telemetry?details_level=1")["result"]["memory"]
        return m["resident_bytes"] / 2 ** 30

    # ── Leer todo el articulado constitucional ────────────────────────
    filtro = {"must": [{"key": "tipo", "match": {"value": "constitucion"}}]}
    puntos, off = [], None
    while True:
        cuerpo = {"limit": 1000, "filter": filtro, "with_payload": True, "with_vector": False}
        if off:
            cuerpo["offset"] = off
        r = pedir(url, api, f"/collections/{COLECCION}/points/scroll", cuerpo)["result"]
        puntos += r["points"]
        off = r.get("next_page_offset")
        if not off:
            break

    grupos: dict[str, list] = defaultdict(list)
    numeros: dict[int, list] = defaultdict(list)
    for pt in puntos:
        clase, num = clasificar(pt.get("payload") or {})
        grupos[clase].append(pt["id"])
        if clase == "articulado" and num:
            numeros[num].append(pt["id"])

    modo = "ESCRITURA" if args.escribir else "EN SECO"
    print(f"\nReparación de la CPEUM · {modo}\n{'─' * 68}")
    print(f"fragmentos revisados : {len(puntos):,}")
    for clase in ("articulado", "transitorios", "firmas"):
        print(f"   {clase:14} {len(grupos[clase]):>5}")
    print(f"artículos distintos  : {len(numeros)}")

    if not args.escribir:
        print("\nRelanza con --escribir.")
        return

    if ram() >= TECHO_RAM_GIB:
        sys.exit(f"⛔ Qdrant en {ram():.2f} GiB: no se escribe.")

    def fijar(ids, payload, etiqueta):
        for i in range(0, len(ids), LOTE):
            pedir(url, api, f"/collections/{COLECCION}/points/payload?wait=false",
                  {"payload": payload, "points": ids[i:i + LOTE]})
            time.sleep(PAUSA)
        print(f"   ✔ {len(ids):>4} · {etiqueta}", flush=True)

    # La cola deja de fingir que es un artículo.
    if grupos["transitorios"]:
        fijar(grupos["transitorios"],
              {"ref": "CPEUM · Transitorios de reformas", "subtipo": "transitorios",
               "articulo_num": None},
              "transitorios de decretos de reforma")
    if grupos["firmas"]:
        fijar(grupos["firmas"],
              {"ref": "CPEUM · Firmas del Constituyente (1917)", "subtipo": "firmas",
               "articulo_num": None},
              "firmas del Congreso Constituyente")

    # El articulado recibe su número, que esta colección nunca tuvo.
    total = 0
    for num, ids in sorted(numeros.items()):
        for i in range(0, len(ids), LOTE):
            pedir(url, api, f"/collections/{COLECCION}/points/payload?wait=false",
                  {"payload": {"articulo_num": num, "subtipo": "articulado"},
                   "points": ids[i:i + LOTE]})
            total += len(ids[i:i + LOTE])
            time.sleep(PAUSA)
    print(f"   ✔ {total:>4} · articulado con su articulo_num")

    # Índice en disco: el nodo va con 4 GiB y el salto sólo usa igualdad.
    try:
        pedir(url, api, f"/collections/{COLECCION}/index?wait=true",
              {"field_name": "articulo_num",
               "field_schema": {"type": "integer", "on_disk": True,
                                "lookup": True, "range": False}},
              metodo="PUT")
        print("   ✔ índice articulo_num creado (en disco)")
    except Exception as e:
        print(f"   ⚠️ índice: {e}")

    print(f"\nRAM de Qdrant: {ram():.3f} GiB")


if __name__ == "__main__":
    main()
