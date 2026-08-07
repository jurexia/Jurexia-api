#!/usr/bin/env python3
"""
Mide qué aporta el salto por grafo de citas al contexto que recibe el modelo.

QUÉ MIDE
--------
Para cada consulta real: recupera como lo hace el chat (denso sobre la
colección de la entidad), toma los K primeros, y cuenta:

  · cuántos artículos citan a otro artículo de su misma ley,
  · cuántos de esos citados NO están en el contexto recuperado,
  · cuántos de esos huecos el salto SÍ puede rellenar.

El último número es la ganancia real: artículos que el abogado necesita para
armar el argumento y que hoy el modelo nunca ve.

No es una opinión sobre la calidad de la respuesta: es un recuento
determinista sobre el contexto, reproducible y sin juez.

USO
---
    python medir_salto_grafo.py                 # Querétaro, 12 consultas
    python medir_salto_grafo.py --entidad oaxaca --k 25
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ENV = RAIZ / ".env"      # el .env vive en la raíz del API; NO va al repo
MODELO_EMBED = "text-embedding-3-small"
_CLAVE = {"leyes_federales": "ley"}   # clave del nombre de ley, aprendida

CONSULTAS = [
    "requisitos para la prescripción positiva de un inmueble",
    "plazo para contestar la demanda en juicio ordinario civil",
    "causales de divorcio incausado y su procedimiento",
    "pérdida de la patria potestad por incumplimiento de alimentos",
    "procedimiento para el desahogo de la prueba testimonial",
    "requisitos de la demanda de amparo indirecto",
    "obligaciones del arrendador y del arrendatario",
    "términos para interponer el recurso de apelación",
    "medidas cautelares en materia familiar",
    "responsabilidad civil por hechos ilícitos",
    "procedimiento administrativo de ejecución fiscal",
    "requisitos del testamento público abierto",
]

# ── Mismo criterio que main.py (copiado a propósito: si el de allá cambia,
#    esta medición deja de reflejarlo y el desajuste se ve en la cifra) ──
_RE_ENCABEZADO = re.compile(r'^\s*art[íi]culo\s*\.?\s*\d+[^\s]*\s*[\.\-–]?\s*', re.I)
_RE_CITA = re.compile(
    r'art[íi]culos?\s+((?:\d{1,4}\s*(?:bis|ter|qu[áa]ter)?\s*[,;yeo]*\s*){1,8})', re.I)
_RE_OTRA = re.compile(
    r'\b(?:de|del)\s+(?:la\s+|el\s+)?'
    r'(?:Ley|C[óo]digo|Constituci[óo]n|Reglamento|Decreto|Tratado)\s+'
    r'(?!(?:vigente|anterior|citad[oa]))[A-ZÁÉÍÓÚ]', re.I)
_RE_MISMA = re.compile(
    r'\b(?:de|del)\s+(?:est[ae]|el\s+presente|la\s+presente)\s+'
    r'(?:ley|c[óo]digo|reglamento|ordenamiento)', re.I)


def citados(texto: str, propio):
    if not texto:
        return []
    cuerpo = _RE_ENCABEZADO.sub("", texto.strip(), count=1)
    fuera = []
    for m in _RE_CITA.finditer(cuerpo):
        cola = cuerpo[m.end():m.end() + 90]
        if _RE_OTRA.search(cola) and not _RE_MISMA.search(cola):
            continue
        for n in re.findall(r'\d{1,4}', m.group(1)):
            v = int(n)
            if 1 <= v <= 9999 and v != propio and v not in fuera:
                fuera.append(v)
    return fuera


def cargar_env():
    datos = {}
    for linea in ENV.read_text(encoding="utf-8", errors="ignore").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#") and "=" in linea:
            c, v = linea.split("=", 1)
            datos[c] = v.strip().strip('"').strip("'")
    return datos


def pedir(url, cuerpo, cabeceras, metodo="POST"):
    d = json.dumps(cuerpo).encode() if cuerpo is not None else None
    r = urllib.request.Request(url, data=d, headers=cabeceras, method=metodo)
    for i in range(3):
        try:
            with urllib.request.urlopen(r, timeout=90) as resp:
                return json.load(resp)
        except Exception as e:
            if i == 2:
                raise
            time.sleep(2 * (i + 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entidad", default="queretaro")
    p.add_argument("--k", type=int, default=20, help="fragmentos recuperados")
    args = p.parse_args()

    env = cargar_env()
    qu = env["QDRANT_URL"].rstrip("/")
    qk = env["QDRANT_API_KEY"]
    ok = env["OPENAI_API_KEY"]
    col = f"leyes_{args.entidad}"

    print(f"\nSalto por grafo · {col} · top-{args.k} · {len(CONSULTAS)} consultas")
    print("─" * 82)
    print(f"{'consulta':46} {'citan':>6} {'huecos':>7} {'recupera':>9}")

    t_citan = t_huecos = t_recupera = 0
    latencias = []

    for consulta in CONSULTAS:
        emb = pedir("https://api.openai.com/v1/embeddings",
                    {"model": MODELO_EMBED, "input": consulta},
                    {"Authorization": f"Bearer {ok}",
                     "Content-Type": "application/json"})["data"][0]["embedding"]

        res = pedir(f"{qu}/collections/{col}/points/query",
                    {"query": emb, "using": "dense", "limit": args.k,
                     "with_payload": True},
                    {"api-key": qk, "Content-Type": "application/json"})
        puntos = res["result"]["points"]

        # Lo que YA está en el contexto: (ley, número de artículo)
        presentes = set()
        for pt in puntos:
            pl = pt["payload"]
            a = pl.get("articulo_num")
            if isinstance(a, int):
                presentes.add((str(pl.get("origen") or pl.get("ley") or ""), a))

        citan = 0
        huecos = []          # (ley, num) citados y ausentes
        for pt in puntos:
            pl = pt["payload"]
            propio = pl.get("articulo_num") if isinstance(pl.get("articulo_num"), int) else None
            ley = str(pl.get("origen") or pl.get("ley") or "")
            nums = citados(str(pl.get("texto") or ""), propio)
            if nums:
                citan += 1
            for n in nums:
                if (ley, n) not in presentes and (ley, n) not in huecos:
                    huecos.append((ley, n))

        # ¿El salto puede rellenarlos? Se pregunta EXACTAMENTE como main.py:
        # una consulta por ley con MatchAny, no una por artículo. Medirlo de
        # otra forma daría una latencia que no es la que sufre el abogado.
        por_ley = {}
        for ley, n in huecos[:6]:      # mismo tope que en producción
            por_ley.setdefault(ley, []).append(n)
        # La clave del nombre de ley no es uniforme entre colecciones: las
        # antiguas indexan `origen`, las del script v3 indexan `ley`. Se
        # aprende igual que en main.py.

        t0 = time.perf_counter()
        recupera = 0
        for ley, nums in list(por_ley.items())[:3]:
            for clave in ([_CLAVE.get(col, "origen")] +
                          ["ley" if _CLAVE.get(col, "origen") == "origen" else "origen"]):
                f = {"must": [{"key": clave, "match": {"value": ley}},
                              {"key": "articulo_num", "match": {"any": nums[:4]}}]}
                try:
                    r = pedir(f"{qu}/collections/{col}/points/scroll",
                              {"limit": len(nums[:4]), "filter": f, "with_payload": False},
                              {"api-key": qk, "Content-Type": "application/json"})
                except Exception:
                    continue
                _CLAVE[col] = clave
                recupera += len(r["result"]["points"])
                break
        latencias.append((time.perf_counter() - t0) * 1000)

        t_citan += citan
        t_huecos += len(huecos)
        t_recupera += recupera
        print(f"{consulta[:45]:46} {citan:>6} {len(huecos):>7} {recupera:>9}")

    n = len(CONSULTAS)
    print("─" * 82)
    print(f"{'PROMEDIO por consulta':46} {t_citan/n:>6.1f} {t_huecos/n:>7.1f} {t_recupera/n:>9.1f}")
    print()
    print(f"Artículos citados que HOY no llegan al modelo : {t_huecos} en {n} consultas")
    print(f"Artículos que el salto SÍ recupera            : {t_recupera} "
          f"({t_recupera*100//max(1,min(t_huecos, n*6))}% de los alcanzables)")
    print(f"Coste del salto                               : "
          f"{sum(latencias)/len(latencias):.0f} ms de mediana por consulta")


if __name__ == "__main__":
    main()
