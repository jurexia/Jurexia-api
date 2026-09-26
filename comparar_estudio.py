# -*- coding: utf-8 -*-
"""VARIANTE CONTRA VARIANTE, Y LAS DOS CONTRA EL RUIDO.

POR QUÉ EXISTE. El Paso 1 de la propuesta aprobada por David el 26-sep-2026
limpia el prompt del estudio en escalera y mide cada peldaño (w2_final §6.1).
Una corrida del taller no se repite igual —sin semilla (L8)—, así que una
diferencia entre variantes sólo cuenta si es mayor que la diferencia entre dos
corridas de la MISMA variante. Esto lee lo que dejó `banco_estudio.py`, mide
cada proyecto con `metricas_estudio.medir` y escribe un informe Markdown:

  · por caso, la mediana de las corridas de cada variante;
  · por métrica, una tabla caso por caso: el engrose real de ese asunto, las
    corridas de la base, las medianas, la diferencia y la banda de ruido;
  · un resumen: en cuántos casos mejora o empeora por encima del ruido, con la
    prueba de signo que la propuesta preregistró (§6.6);
  · y lo que BLOQUEA: los casos donde la cobertura empeora —un ancla del
    escrito o un concepto que la base nombraba en ≥ 2/3 de sus corridas y la
    variante en ≤ 1/3 (M1c), o una peor corrida con menos cobertura que la peor
    de la base (M1b)—. Repetir menos a costa de contestar menos no es mejorar.

LA BANDA DE RUIDO de un caso es el recorrido (máx − mín) de las corridas de la
base, y, si se pide con `--ruido v1@r` (el mismo prompt corrido aparte), la
distancia entre las medianas de v1 y v1@r cuando es mayor.

Uso:
    .venv/bin/python comparar_estudio.py --etiqueta estandar --base v1 --variantes v2
    .venv/bin/python comparar_estudio.py --etiqueta estandar --base v1 --ruido v1@r
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import banco_estudio as be
import metricas_estudio as me

# M1c de w2_final §6.3: «A lo contesta en ≥ 2/3 y C en ≤ 1/3».
UMBRAL_BASE = 2 / 3
UMBRAL_NUEVA = 1 / 3


# ═══════════════════════════════════════════════════════════════════════════
# CARGAR Y MEDIR
# ═══════════════════════════════════════════════════════════════════════════
def cargar(raiz: Path, etiqueta: str) -> dict:
    """{caso: {variante: [filas]}} con TODAS las corridas guardadas."""
    fuera = defaultdict(lambda: defaultdict(list))
    base = raiz / etiqueta
    if not base.exists():
        return {}
    for f in sorted(base.glob("*/*.json")):
        if f.name == "manifiesto.json":
            continue
        try:
            fila = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(fila, dict) or "variante" not in fila:
            continue
        fuera[fila["caso"]][fila["variante"]].append(fila)
    for c in fuera.values():
        for v in c.values():
            v.sort(key=lambda x: x.get("k") or 0)
    return {c: dict(v) for c, v in fuera.items()}


def buenas(filas: list) -> list:
    return [f for f in filas if f.get("ok") and not f.get("descartada") and f.get("texto")]


class Medidor:
    """Mide cada corrida una vez, con el escrito de su caso."""

    def __init__(self, raiz: Path = be.AQUI, escritos: dict = None, oros: dict = None):
        self.raiz = raiz
        self._esc = dict(escritos or {})
        self._oro = dict(oros or {})
        self._m = {}

    def escrito(self, numero: str) -> str:
        if numero not in self._esc:
            c = be.POR_NUMERO.get(numero)
            self._esc[numero] = be.escrito_de(c, self.raiz) if c else ""
        return self._esc[numero]

    def oro(self, numero: str):
        """Las medidas del engrose real de ese asunto (None si no hay)."""
        if numero not in self._oro:
            c = be.POR_NUMERO.get(numero)
            t = be.oro_de(c) if c else ""
            self._oro[numero] = me.medir(t, self.escrito(numero) or None) if t else None
        elif isinstance(self._oro[numero], str):
            t = self._oro[numero]
            self._oro[numero] = me.medir(t, self.escrito(numero) or None) if t else None
        return self._oro[numero]

    def medir(self, fila: dict) -> dict:
        clave = (fila["caso"], fila["variante"], fila.get("k"))
        if clave not in self._m:
            self._m[clave] = me.medir(fila["texto"], self.escrito(fila["caso"]) or None)
        return self._m[clave]


# ═══════════════════════════════════════════════════════════════════════════
# AGREGAR Y COMPARAR
# ═══════════════════════════════════════════════════════════════════════════
def valores(ms: list, clave: str) -> list:
    return [float(m[clave]) for m in ms if m.get(clave) is not None]


def rango(xs: list):
    return (max(xs) - min(xs)) if len(xs) >= 2 else None


def fraccion_por_elemento(ms: list, campo: str) -> dict:
    """Qué fracción de las corridas nombra cada ancla (u ordinal)."""
    if not ms:
        return {}
    c = Counter()
    for m in ms:
        for x in set(m["detalle"].get(campo) or []):
            c[x] += 1
    return {x: n / len(ms) for x, n in c.items()}


def regresiones(ms_base: list, ms_nueva: list) -> dict:
    """Lo que la base contestaba y la variante dejó de nombrar."""
    fuera = {"anclas": [], "ordinales": [], "peor_corrida": None}
    if not ms_base or not ms_nueva:
        return fuera
    fb = fraccion_por_elemento(ms_base, "anclas_cubiertas")
    fn = fraccion_por_elemento(ms_nueva, "anclas_cubiertas")
    fuera["anclas"] = sorted(a for a, x in fb.items()
                             if x >= UMBRAL_BASE - 1e-9 and fn.get(a, 0.0) <= UMBRAL_NUEVA + 1e-9)
    ob = fraccion_por_elemento(ms_base, "ordinales_nombrados")
    on = fraccion_por_elemento(ms_nueva, "ordinales_nombrados")
    fuera["ordinales"] = sorted(o for o, x in ob.items()
                                if x >= UMBRAL_BASE - 1e-9 and on.get(o, 0.0) <= UMBRAL_NUEVA + 1e-9)
    cb = valores(ms_base, "cobertura_demanda")
    cn = valores(ms_nueva, "cobertura_demanda")
    if cb and cn and min(cn) < min(cb) - 1e-9:
        fuera["peor_corrida"] = (round(min(cb), 3), round(min(cn), 3))
    return fuera


def signo(ganas: int, pierdes: int) -> float | None:
    """Prueba de signo unilateral: P(X ≥ ganas | n = ganas + pierdes, p = ½)."""
    n = ganas + pierdes
    if n == 0:
        return None
    return sum(math.comb(n, i) for i in range(ganas, n + 1)) / 2 ** n


def veredicto(delta, ruido, mejor) -> str:
    if delta is None or not mejor:
        return "—"
    if ruido is None:
        ruido = 0.0
    if abs(delta) <= ruido + 1e-9:
        return "≈"
    mejora = delta < 0 if mejor == "menos" else delta > 0
    return "mejora" if mejora else "EMPEORA"


def comparar_par(datos: dict, medidor: Medidor, base: str, nueva: str,
                 ruido_con: str = None) -> dict:
    """Todo lo que el informe necesita para «nueva» contra «base»."""
    casos = sorted(datos, key=lambda c: list(be.POR_NUMERO).index(c)
                   if c in be.POR_NUMERO else 999)
    filas = []
    for c in casos:
        vb = buenas(datos[c].get(base, []))
        vn = buenas(datos[c].get(nueva, []))
        vr = buenas(datos[c].get(ruido_con, [])) if ruido_con else []
        if not vb or not vn:
            filas.append({"caso": c, "falta": True, "n_base": len(vb), "n_nueva": len(vn)})
            continue
        mb = [medidor.medir(f) for f in vb]
        mn = [medidor.medir(f) for f in vn]
        mr = [medidor.medir(f) for f in vr]
        por_metrica = {}
        for clave in me.CLAVES:
            xb, xn, xr = valores(mb, clave), valores(mn, clave), valores(mr, clave)
            medb, medn = me.mediana(xb), me.mediana(xn)
            ruido = rango(xb)
            if xr and medb is not None:
                d_r = abs(me.mediana(xr) - medb)
                ruido = d_r if ruido is None else max(ruido, d_r)
            delta = (medn - medb) if (medb is not None and medn is not None) else None
            por_metrica[clave] = {"base": xb, "nueva": xn, "med_base": medb,
                                  "med_nueva": medn, "delta": delta, "ruido": ruido,
                                  "veredicto": veredicto(delta, ruido, me.MEJOR[clave])}
        oro = medidor.oro(c)
        filas.append({"caso": c, "falta": False, "n_base": len(vb), "n_nueva": len(vn),
                      "metricas": por_metrica, "oro": oro,
                      "regresion": regresiones(mb, mn)})
    resumen = {}
    for clave in me.CLAVES:
        vs = [f["metricas"][clave] for f in filas if not f["falta"]]
        mej = sum(1 for v in vs if v["veredicto"] == "mejora")
        emp = sum(1 for v in vs if v["veredicto"] == "EMPEORA")
        igu = sum(1 for v in vs if v["veredicto"] == "≈")
        resumen[clave] = {
            "mejora": mej, "empeora": emp, "ruido": igu, "casos": len(vs),
            "med_base": me.mediana([v["med_base"] for v in vs]),
            "med_nueva": me.mediana([v["med_nueva"] for v in vs]),
            "med_delta": me.mediana([v["delta"] for v in vs]),
            "p_signo": signo(mej, emp) if me.MEJOR[clave] else None}
    bloquea = [f for f in filas if not f["falta"] and (
        f["regresion"]["anclas"] or f["regresion"]["ordinales"]
        or f["regresion"]["peor_corrida"])]
    return {"base": base, "nueva": nueva, "ruido_con": ruido_con, "filas": filas,
            "resumen": resumen, "bloquea": bloquea}


def operacion(datos: dict) -> dict:
    """Corridas buenas, descartadas y con error; commits y tiempos por variante."""
    por = defaultdict(lambda: {"buenas": 0, "descartadas": 0, "errores": 0,
                               "commits": Counter(), "versiones": Counter(),
                               "t_total": [], "palabras": [], "huellas": defaultdict(set),
                               "motivos": Counter()})
    for c, vs in datos.items():
        for v, filas in vs.items():
            p = por[v]
            for f in filas:
                if f.get("descartada"):
                    p["descartadas"] += 1
                    p["motivos"][f["descartada"][:80]] += 1
                elif f.get("ok"):
                    p["buenas"] += 1
                    p["t_total"].append(f.get("t_total"))
                    p["palabras"].append(f.get("palabras"))
                    p["commits"][f.get("commit") or "—"] += 1
                    p["versiones"][f.get("version")] += 1
                else:
                    p["errores"] += 1
                    p["motivos"][(f.get("error") or "")[:80]] += 1
                if f.get("huella_sesion"):
                    p["huellas"][c].add(f["huella_sesion"])
    return dict(por)


# ═══════════════════════════════════════════════════════════════════════════
# EL INFORME
# ═══════════════════════════════════════════════════════════════════════════
def _f(v, nd=2):
    return me._fmt(v, nd)


def _referencia(raiz: Path) -> dict:
    """La distribución del oro que dejó `metricas_estudio --calibrar`."""
    p = raiz / "calibracion.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("referencia") or {}
        except Exception:
            return {}
    return {}


def informe(etiqueta: str, pares: list, oper: dict, ref: dict = None) -> str:
    ref = ref or {}
    L = [f"# Banco del estudio · «{etiqueta}»", "",
         f"Generado {dt.datetime.now().isoformat(timespec='minutes')} por "
         "`comparar_estudio.py`. Cada cifra es la **mediana de las corridas** de una "
         "variante en un caso; «≈» quiere decir dentro de la banda de ruido (recorrido "
         "de las corridas de la base, o base contra su réplica).", ""]
    for par in pares:
        b, n = par["base"], par["nueva"]
        L += [f"## {n} contra {b}" + (f" (ruido también con {par['ruido_con']})"
                                      if par["ruido_con"] else ""), ""]
        faltan = [f for f in par["filas"] if f["falta"]]
        if faltan:
            L.append("Casos sin corridas buenas en alguna de las dos variantes (no se "
                     "comparan): " + ", ".join(
                         f"{f['caso']} ({b}: {f['n_base']}, {n}: {f['n_nueva']})"
                         for f in faltan) + ".")
            L.append("")
        pocas = [f for f in par["filas"] if not f["falta"] and f["n_base"] < 2]
        if pocas:
            L.append("**Sin banda de ruido** en " + ", ".join(f["caso"] for f in pocas)
                     + ": la base tiene una sola corrida buena; cualquier diferencia cuenta.")
            L.append("")
        # ── LO QUE BLOQUEA ─────────────────────────────────────────────────
        L += ["### Cobertura: lo que bloquea", ""]
        if not par["bloquea"]:
            L += ["Ningún caso pierde anclas del escrito ni conceptos nombrados que la base "
                  "nombrara en ≥ 2/3 de sus corridas, y en ninguno la peor corrida de "
                  f"{n} cubre menos que la peor de {b}.", ""]
        else:
            L += [f"**BLOQUEA en {len(par['bloquea'])} caso(s).** Repetir menos no cuenta "
                  "como mejora si se contesta menos (w2_final §6.6).", "",
                  f"| Caso | Anclas que {b} nombraba y {n} ya no | Conceptos | "
                  "Peor corrida (base → nueva) |", "|---|---|---|---|"]
            for f in par["bloquea"]:
                r = f["regresion"]
                L.append(f"| {f['caso']} | {', '.join(r['anclas']) or '—'} | "
                         f"{', '.join(map(str, r['ordinales'])) or '—'} | "
                         f"{'%s → %s' % r['peor_corrida'] if r['peor_corrida'] else '—'} |")
            L.append("")
        # ── RESUMEN POR MÉTRICA ─────────────────────────────────────────────
        L += ["### Resumen por métrica", "",
              f"| Métrica | Mejor | Oro mediana (p90) | {b} | {n} | Δ mediana | "
              "Mejora / empeora / ≈ | p (signo) |", "|---|---|---|---|---|---|---|---|"]
        for clave in me.CLAVES:
            r = par["resumen"][clave]
            o = ref.get(clave) or {}
            oro = f"{_f(o.get('mediana'))} ({_f(o.get('p90'))})" if o else "—"
            mep = (f"{r['mejora']} / {r['empeora']} / {r['ruido']}"
                   if me.MEJOR[clave] else "—")
            L.append(f"| {me.NOMBRE[clave]} | {me.MEJOR[clave] or '—'} | {oro} | "
                     f"{_f(r['med_base'])} | {_f(r['med_nueva'])} | {_f(r['med_delta'])} | "
                     f"{mep} | {_f(r['p_signo'], 3)} |")
        L.append("")
        # ── UNA TABLA POR MÉTRICA ───────────────────────────────────────────
        L += ["### Caso por caso", ""]
        for clave in me.CLAVES:
            L += [f"#### {me.NOMBRE[clave]}", "",
                  f"| Caso | Oro | {b} (corridas) | {b} | {n} (corridas) | {n} | Δ | Ruido | |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for f in par["filas"]:
                if f["falta"]:
                    continue
                v = f["metricas"][clave]
                oro = _f((f["oro"] or {}).get(clave)) if f["oro"] else "—"
                L.append(f"| {f['caso']} | {oro} | {' · '.join(_f(x) for x in v['base'])} | "
                         f"{_f(v['med_base'])} | {' · '.join(_f(x) for x in v['nueva'])} | "
                         f"{_f(v['med_nueva'])} | {_f(v['delta'])} | {_f(v['ruido'])} | "
                         f"{v['veredicto']} |")
            L.append("")
    # ── OPERACIÓN ───────────────────────────────────────────────────────────
    L += ["## Operación", "",
          "| Variante | Buenas | Descartadas | Con error | Commit(s) | Versión(es) | "
          "Tiempo mediano | Palabras (evento) |", "|---|---|---|---|---|---|---|---|"]
    for v, p in sorted(oper.items()):
        commits = ", ".join(f"{k[:8]}×{c}" for k, c in p["commits"].most_common())
        vers = ", ".join(f"v{k}×{c}" for k, c in p["versiones"].most_common())
        L.append(f"| {v} | {p['buenas']} | {p['descartadas']} | {p['errores']} | "
                 f"{commits or '—'} | {vers or '—'} | "
                 f"{_f(me.mediana([x for x in p['t_total'] if x is not None]), 0)} s | "
                 f"{_f(me.mediana([x for x in p['palabras'] if x is not None]), 0)} |")
    avisos = []
    for v, p in sorted(oper.items()):
        if len([k for k in p["commits"] if k != "—"]) > 1:
            avisos.append(f"«{v}» corrió con {len(p['commits'])} commits distintos: hubo un "
                          "despliegue a media tanda; compara sólo corridas del mismo commit.")
        for c, hs in p["huellas"].items():
            if len(hs) > 1:
                avisos.append(f"{c}: la sesión cambió entre corridas de «{v}» "
                              f"({', '.join(sorted(hs))}): el criterio no fue el mismo.")
        for motivo, k in p["motivos"].most_common(5):
            avisos.append(f"«{v}»: {k}× {motivo}")
    if avisos:
        L += [""] + [f"- {a}" for a in avisos]
    L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compara variantes del banco del estudio.")
    ap.add_argument("--etiqueta", default="estandar")
    ap.add_argument("--base", default="v1")
    ap.add_argument("--variantes", default="",
                    help="las que se comparan contra la base (por omisión, todas las demás)")
    ap.add_argument("--ruido", default="",
                    help="réplica de la base (p. ej. v1@r) para la banda v1 contra v1'")
    ap.add_argument("--raiz", default=str(be.AQUI))
    ap.add_argument("--salida", default="")
    a = ap.parse_args(argv)
    raiz = Path(a.raiz)
    datos = cargar(raiz, a.etiqueta)
    if not datos:
        print(f"No hay corridas en {raiz / a.etiqueta}.")
        return 1
    todas = sorted({v for vs in datos.values() for v in vs})
    nuevas = [x.strip() for x in a.variantes.split(",") if x.strip()] or \
        [v for v in todas if v not in (a.base, a.ruido)]
    medidor = Medidor(raiz)
    pares = [comparar_par(datos, medidor, a.base, n, a.ruido or None) for n in nuevas]
    ref = _referencia(raiz)
    if not ref and me.CASOS_KINGSTON.exists():
        ref = me.calibrar(raiz)["referencia"]
    md = informe(a.etiqueta, pares, operacion(datos), ref)
    salida = Path(a.salida) if a.salida else raiz / a.etiqueta / (
        f"informe_{'_'.join(nuevas)}_vs_{a.base}.md")
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n→ {salida}")
    return 2 if any(p["bloquea"] for p in pares) else 0


if __name__ == "__main__":
    sys.exit(main())
