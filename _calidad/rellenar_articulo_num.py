#!/usr/bin/env python3
"""
Rellena `articulo_num` (entero) en las colecciones de leyes que no lo tienen.

POR QUÉ
-------
23 de las 34 colecciones de leyes —incluidas `leyes_federales` y `leyes_cdmx`,
las dos más consultadas— no traen `articulo_num`. Sin ese campo indexado no se
puede saltar de un artículo al artículo que cita: la búsqueda cruzada actual
adivina con patrones de cadena sobre `ref` («Artículo 55.», «Artículo 55 ») y
falla en cuanto la colección escribe «Art. 55», que es lo que usan CDMX y
Guanajuato.

El dato NO se inventa: se deriva de `ref`, que ya existe en el 95-100 % de los
puntos («Art. 923», «Artículo 923.», «Artículo 923 BIS»). No se reingesta nada,
no se recalculan vectores: es sólo payload.

FRENOS (ver memoria feedback-qdrant-produccion)
-----------------------------------------------
Escribir en Qdrant es tocar producción. Por eso: lotes pequeños, pausa entre
lotes, `wait=false` y reintento en las lecturas. Y por eso el modo por omisión
es EN SECO: no escribe nada hasta que se pasa --escribir.

USO
---
    python rellenar_articulo_num.py                      # seco, todas
    python rellenar_articulo_num.py --coleccion leyes_oaxaca
    python rellenar_articulo_num.py --coleccion leyes_oaxaca --escribir
    python rellenar_articulo_num.py --escribir           # todas, de verdad
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ENV = RAIZ / ".env"      # el .env vive en la raíz del API; NO va al repo

LOTE_LECTURA = 1000      # puntos por scroll
LOTE_ESCRITURA = 20      # ids por operación — el freno de la memoria
OPS_POR_PETICION = 40    # operaciones agrupadas en una sola llamada
PAUSA = 0.35             # segundos entre peticiones
REINTENTOS = 4
TECHO_RAM_GIB = 3.20     # el nodo tiene 4 GiB: se para con margen de sobra

# «Art. 923» · «Artículo 923.» · «ARTICULO 923 BIS» · «Artículo 5o.»
RE_ART = re.compile(r'^\s*art[íi]?c?u?l?o?\.?\s*[\s\.]*(\d{1,4})\s*[ºo°]?', re.I)


def cargar_env() -> tuple[str, str]:
    if not ENV.exists():
        sys.exit(f"No encuentro {ENV}")
    datos = {}
    for linea in ENV.read_text(encoding="utf-8", errors="ignore").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#") and "=" in linea:
            clave, valor = linea.split("=", 1)
            datos[clave] = valor.strip().strip('"').strip("'")
    url = datos.get("QDRANT_URL", "").rstrip("/")
    api = datos.get("QDRANT_API_KEY", "")
    if not url or not api:
        sys.exit("Faltan QDRANT_URL o QDRANT_API_KEY en el .env")
    return url, api


class Qdrant:
    def __init__(self, url: str, api: str):
        self.url, self.api = url, api

    def _pedir(self, ruta: str, cuerpo=None, metodo=None):
        datos = json.dumps(cuerpo).encode() if cuerpo is not None else None
        pet = urllib.request.Request(
            self.url + ruta, data=datos,
            headers={"api-key": self.api, "Content-Type": "application/json"},
            method=metodo or ("POST" if datos else "GET"))
        ultimo = None
        for intento in range(REINTENTOS):
            try:
                with urllib.request.urlopen(pet, timeout=90) as r:
                    return json.load(r)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                ultimo = e
                time.sleep(1.5 * (intento + 1))   # las lecturas se reintentan
        raise RuntimeError(f"{ruta}: {ultimo}")

    def colecciones(self) -> list[str]:
        r = self._pedir("/collections")
        return sorted(c["name"] for c in r["result"]["collections"])

    def memoria_gib(self) -> float:
        """RSS del proceso de Qdrant, en GiB.

        El panel de Qdrant Cloud muestra ~3.5 de 4 GiB, pero ese número incluye
        la caché de página de los ficheros mapeados —reclamable por el sistema—.
        Lo que de verdad puede tumbar el nodo es la memoria residente del
        proceso, que es lo que se vigila aquí.
        """
        m = self._pedir("/telemetry?details_level=1")["result"]["memory"]
        return m["resident_bytes"] / 2 ** 30

    def info(self, col: str) -> dict:
        return self._pedir(f"/collections/{col}")["result"]

    def recorrer(self, col: str, campos: list[str]):
        desde = None
        while True:
            cuerpo = {"limit": LOTE_LECTURA, "with_payload": campos,
                      "with_vector": False}
            if desde is not None:
                cuerpo["offset"] = desde
            r = self._pedir(f"/collections/{col}/points/scroll", cuerpo)["result"]
            for punto in r["points"]:
                yield punto
            desde = r.get("next_page_offset")
            if desde is None:
                return

    def fijar_payload(self, col: str, ids: list, payload: dict):
        self._pedir(f"/collections/{col}/points/payload?wait=false",
                    {"payload": payload, "points": ids})

    def fijar_payload_lote(self, col: str, operaciones: list[tuple[list, dict]]):
        """Muchas asignaciones en UNA petición.

        Cada artículo lleva su propio número, así que hay una operación por
        valor distinto. Mandarlas una a una costaba un viaje de red completo
        por cada número de artículo —miles por colección— y el trabajo se iba
        a horas. El endpoint /points/batch admite un lote de operaciones y
        deja el mismo número de escrituras con una fracción de los viajes.
        """
        self._pedir(
            f"/collections/{col}/points/batch?wait=false",
            {"operations": [
                {"set_payload": {"payload": payload, "points": ids}}
                for ids, payload in operaciones
            ]})

    def crear_indice(self, col: str, campo: str):
        """Índice EN DISCO y sólo de búsqueda exacta.

        El clúster va con 3.53 de 4 GiB de RAM. El payload de estas 34
        colecciones ya vive en disco (`on_disk_payload=true`), así que rellenar
        el campo no cuesta memoria; el índice sí la costaría si se creara con
        los valores por omisión. Con `on_disk: true` se queda en disco, y con
        `range: false` se ahorra la estructura ordenada que nunca se usa: el
        salto entre artículos es siempre igualdad exacta (`articulo_num == 47`),
        jamás un rango.
        """
        self._pedir(
            f"/collections/{col}/index?wait=true",
            {"field_name": campo,
             "field_schema": {"type": "integer", "on_disk": True,
                              "lookup": True, "range": False}},
            metodo="PUT")


def numero_de_ref(ref: str) -> int | None:
    m = RE_ART.match(ref or "")
    if not m:
        return None
    try:
        n = int(m.group(1))
    except ValueError:
        return None
    # Un artículo con número absurdo es un fallo de troceado, no un artículo.
    return n if 1 <= n <= 9999 else None


def procesar(q: Qdrant, col: str, escribir: bool) -> dict:
    info = q.info(col)
    total = info.get("points_count") or 0
    ya_indexado = "articulo_num" in (info.get("payload_schema") or {})

    faltantes: dict[int, list] = defaultdict(list)
    con_valor = sin_ref = leidos = 0

    for punto in q.recorrer(col, ["ref", "articulo_num"]):
        leidos += 1
        pl = punto.get("payload") or {}
        if isinstance(pl.get("articulo_num"), int):
            con_valor += 1
            continue
        n = numero_de_ref(str(pl.get("ref") or ""))
        if n is None:
            sin_ref += 1        # «Transitorios», «Preámbulo»: no son artículos
            continue
        faltantes[n].append(punto["id"])

    por_escribir = sum(len(v) for v in faltantes.values())
    print(f"  {col:28} {leidos:>7,} leídos · {con_valor:>6,} ya tenían · "
          f"{por_escribir:>6,} a rellenar · {sin_ref:>5,} sin número"
          f"{'  [índice ok]' if ya_indexado else ''}")

    # El índice se crea aunque no quede nada por escribir. Antes se salía aquí
    # y una colección con el campo ya puesto —por una corrida anterior— se
    # quedaba SIN índice para siempre. Le pasó a Aguascalientes y Baja
    # California, y el salto habría hecho barrido completo sobre ellas.
    if escribir and not ya_indexado and (con_valor or por_escribir):
        try:
            q.crear_indice(col, "articulo_num")
            ya_indexado = True
            print(f"      índice articulo_num creado en {col}", flush=True)
        except Exception as e:
            print(f"      ⚠️ no pude crear el índice en {col}: {e}", flush=True)

    if not escribir or not por_escribir:
        return {"col": col, "escritos": 0, "pendientes": por_escribir}

    # Una operación por (número, trozo de ids); se despachan agrupadas.
    operaciones = []
    for numero, ids in sorted(faltantes.items()):
        for i in range(0, len(ids), LOTE_ESCRITURA):
            operaciones.append((ids[i:i + LOTE_ESCRITURA],
                                {"articulo_num": numero}))

    escritos = hito = 0
    for i in range(0, len(operaciones), OPS_POR_PETICION):
        grupo = operaciones[i:i + OPS_POR_PETICION]
        q.fijar_payload_lote(col, grupo)
        escritos += sum(len(ids) for ids, _ in grupo)
        time.sleep(PAUSA)
        if escritos - hito >= 5000:
            hito = escritos
            print(f"      … {escritos:,}/{por_escribir:,}", flush=True)

    print(f"      ✔ {escritos:,} puntos actualizados en {col}")
    return {"col": col, "escritos": escritos, "pendientes": 0}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--coleccion", help="sólo esta colección")
    p.add_argument("--escribir", action="store_true",
                   help="escribe de verdad (por omisión va en seco)")
    args = p.parse_args()

    url, api = cargar_env()
    q = Qdrant(url, api)

    cols = ([args.coleccion] if args.coleccion
            else [c for c in q.colecciones() if c.startswith("leyes_")])

    modo = "ESCRITURA" if args.escribir else "EN SECO (no escribe nada)"
    print(f"\nRelleno de articulo_num · {modo}\n{'─' * 78}")

    if args.escribir:
        print(f"  (vigilando RAM: se detiene por encima de {TECHO_RAM_GIB} GiB)")

    resumen = []
    for col in cols:
        if args.escribir:
            try:
                ram = q.memoria_gib()
            except Exception:
                ram = 0.0     # sin telemetría no se bloquea el trabajo
            if ram >= TECHO_RAM_GIB:
                print(f"\n  ⛔ ALTO: Qdrant en {ram:.2f} GiB residentes "
                      f"(techo {TECHO_RAM_GIB}). Se detiene antes de {col}.")
                print("     El relleno es idempotente: relanzar continúa donde quedó.")
                break
        try:
            resumen.append(procesar(q, col, args.escribir))
        except Exception as e:
            print(f"  {col:28} ERROR: {e}")

    pend = sum(r["pendientes"] for r in resumen)
    hechos = sum(r["escritos"] for r in resumen)
    print(f"{'─' * 78}")
    if args.escribir:
        print(f"Total actualizado: {hechos:,} puntos en {len(resumen)} colecciones")
    else:
        print(f"Se rellenarían {pend:,} puntos. Relanza con --escribir.")


if __name__ == "__main__":
    main()
