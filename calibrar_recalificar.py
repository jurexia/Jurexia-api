"""CALIBRACIÓN DE LA RECALIFICACIÓN CON LA PREMISA DEL CAMBIO DE SENTIDO.

POR QUÉ EXISTE (26-sep-2026). David: «si cambio sentido hay que tumbar y
regenerar con la premisa del cambio de sentido». En el banco Kingston, con el
principal desestimado frente a la propuesta del motor, cinco accesorios
quedaban «fundado» con la calificación que el motor escribió para la otra vía,
y el engrose real negó los cinco. Aquí se mide, contra los engroses (la tabla
`ORO` de `calibrar_arbol.py` y la calificación que el engrose les dio,
`ORO_CALIF`), qué hace la recalificación con el modelo:

  · quién se recalifica (sin modelo): la regla sólo debe tocar los asuntos en
    que el principal va por la vía CONTRARIA a la del motor —el 722/2025 real
    ya iba en la del motor y no debe recalificar nada—;
  · cómo salen (con el modelo, `--modelo`, una llamada por asunto y como mucho
    dos corridas): cuántos de los que quedaban «fundado» salen ahora como los
    trató el engrose, y si empeora alguno de los que ya estaban bien.

Uso (con el .env cargado; la base sólo se LEE):
    .venv/bin/python calibrar_recalificar.py --volcar /ruta/sesiones.json
    .venv/bin/python calibrar_recalificar.py /ruta/sesiones.json [--modelo] [--corridas 2]
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import dataclasses
import json
import os
import sys
import types
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import calibrar_arbol as ca  # noqa: E402

# LA CALIFICACIÓN QUE LES DIO EL ENGROSE (leída el 26-sep-2026 de los «oro»;
# las notas de `calibrar_arbol.ORO` dicen de dónde). 174/2026 P4 y 43/2025 P2
# se leyeron para esta tabla: «De ahí que resulten infundados los conceptos de
# violación» (174) y «de ahí lo infundado de sus conceptos de violación en
# estudio» (43, la carga de la prueba; lo fundado del 43 es el P3, intereses,
# que no depende del principal y el árbol no toca).
ORO_CALIF = {
    ("274/2025", 2): "infundado",
    ("274/2025", 3): "infundado",
    ("263/2025", 2): "infundado",
    ("103/2025", 2): "inoperante",
    ("103/2025", 3): "infundado",
    ("400/2024", 2): "infundado",
    ("702/2022", 2): "inoperante",
    ("174/2026", 2): "infundado",
    ("174/2026", 4): "infundado",
    ("43/2025", 2): "infundado",
    ("529/2024", 2): "inoperante",
    ("529/2024", 3): "infundado",
    ("590/2024", 2): "infundado",
    ("722/2025", 2): "ineficaz",
    ("722/2025", 3): "ineficaz",
    ("641/2024", 3): "infundado",
}
# La vieja regla es la de la integración del Paso 2, antes de recalificar.
VIEJA = "cdc4215"


def volcar(ruta: str) -> None:
    """Lee (SÓLO LEE) las sesiones de casa de los asuntos de la tabla."""
    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    out = {}
    for exp in sorted({e for e, _ in ca.ORO}):
        r = sb.table("taller_sesiones").select("id, email, expediente, estado, propuestas") \
            .eq("email", ca.CASA).eq("expediente", exp).limit(1).execute()
        if r.data:
            out[exp] = r.data[0]
    Path(ruta).write_text(json.dumps(out, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"{len(out)} sesiones → {ruta}")


def _fila_volcado(fila: dict) -> dict:
    """La fila con la forma de `calibrar_arbol.volcar`, para `_escenario`."""
    e = fila.get("estado") or {}
    return {"id": fila.get("id"), "email": fila.get("email"), "expediente": fila.get("expediente"),
            "propuestas": fila.get("propuestas"),
            "problemas": (e.get("fases") or {}).get("problemas"),
            "tipo": (e.get("encargo") or {}).get("tipo_asunto"),
            "glob": ((e.get("propuesta") or {}).get("respuesta") or {}).get("global")}


def _resultado(fila: dict):
    """(r, material) rehidratados de la sesión, como los tiene el taller."""
    import fases123_pipeline as f123
    import taller_estado as te
    e = fila.get("estado") or {}
    campos = {f.name for f in dataclasses.fields(f123.Fases123)}
    fases = f123.Fases123(**{k: v for k, v in (e.get("fases") or {}).items() if k in campos})
    enc = e.get("encargo") or {}
    r = types.SimpleNamespace(
        fases=fases, encargo=types.SimpleNamespace(
            tipo_asunto=str(enc.get("tipo_asunto") or "amparo_directo"),
            es_recurso=bool(enc.get("es_recurso"))))
    return r, te.material_rehidratado(e.get("material") or {})


def _escenario(fila: dict, r):
    esc = ca._escenario(_fila_volcado(fila), True)
    if esc is None:
        return None
    import taller_estado as te
    return esc + (te.huella_contraste(r),)


def quien(S: dict) -> dict:
    """Sin modelo: por asunto, qué tumba la regla y cómo quedaba antes."""
    import arbol_decision as ad
    import recalificar as rc
    vieja = ca._cargar_vieja(VIEJA)
    fuera = {}
    for exp in sorted({e for e, _ in ca.ORO}):
        fila = S.get(exp)
        if not fila:
            continue
        r, material = _resultado(fila)
        esc = _escenario(fila, r)
        if esc is None:
            continue
        probs, crit, cl, props, smot, tipo, huella = copy.deepcopy(esc)
        cv = copy.deepcopy(crit)
        vieja.aplicar(probs, cv, cl, props, sentido_motor=smot, tipo_asunto=tipo)
        av, det = ad.aplicar(probs, crit, cl, props, sentido_motor=smot, tipo_asunto=tipo,
                             huella_adelanto=huella)
        pend = rc.pendientes(det)
        pral = next(c for c in crit if c["jerarquia"] == "principal")
        pm = next((p for p in props if p["problema"] == pral["problema"]), {})
        fuera[exp] = {"r": r, "material": material, "crit": crit, "detalle": det,
                      "pendientes": pend, "antes": {c["problema"]: c["sentido"] for c in cv},
                      "probs": probs, "motor_principal": pm.get("sentido_propio") or pm.get("sentido"),
                      "clave": rc.clave_de(det)}
    return fuera


async def _correr(caso: dict, cliente, corridas: int) -> list:
    import recalificar as rc
    principal, acc = rc.entradas(caso["r"], caso["crit"], caso["detalle"])
    salidas = []
    for _ in range(corridas):
        salidas.append(await rc.recalificar(caso["r"], caso["material"], principal, acc, "", {},
                                            cliente=cliente, clave_=caso["clave"]))
    return salidas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", nargs="?")
    ap.add_argument("--volcar")
    ap.add_argument("--modelo", action="store_true", help="llama al modelo (una llamada por asunto)")
    ap.add_argument("--corridas", type=int, default=1)
    ap.add_argument("--salida", default="")
    a = ap.parse_args()
    if a.volcar:
        volcar(a.volcar)
        return
    import tipos_asunto as ta
    S = json.loads(Path(a.dump).read_text(encoding="utf-8"))
    casos = quien(S)

    print("═══ A. QUIÉN SE RECALIFICA (sin modelo, principal desestimado) ═══")
    for exp, cs in casos.items():
        print(f"  {exp}: motor propuso el principal {cs['motor_principal']} · "
              + (f"se recalifican {len(cs['pendientes'])}: "
                 + ", ".join(f"P{i}" for i, p in enumerate(cs['probs'], 1)
                             if p['pregunta'] in cs['pendientes'])
                 if cs["pendientes"] else "no se recalifica nada"))

    if not a.modelo:
        return
    from openai import AsyncOpenAI
    cliente = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    corridas = max(1, min(2, a.corridas))

    async def todo():
        res = {}
        for exp, cs in casos.items():
            if cs["pendientes"]:
                res[exp] = await _correr(cs, cliente, corridas)
        return res
    res = asyncio.run(todo())

    print(f"\n═══ B. CONTRA LOS ENGROSES ({corridas} corrida(s) por asunto) ═══")
    tabla, bien_antes, bien_ahora, empeoran, fundados_antes, fundados_bien = [], 0, 0, 0, 0, 0
    for (exp, n), (trato, nota) in ca.ORO.items():
        cs = casos.get(exp)
        if not cs:
            continue
        preg = cs["probs"][n - 1]["pregunta"]
        oro = ORO_CALIF.get((exp, n), "")
        antes = cs["antes"].get(preg, "")
        if preg in cs["pendientes"]:
            ahora = [((s.get("resultados") or {}).get(preg) or {}).get("sentido", "—")
                     for s in res.get(exp, [])]
            caida = [bool(((s.get("resultados") or {}).get(preg) or {}).get("verificado"))
                     for s in res.get(exp, [])]
        else:
            ahora, caida = [antes], [False]

        def _bien(s):
            return bool(s) and s != "—" and ta.prospera(s) == ta.prospera(oro)
        b_antes = _bien(antes)
        b_ahora = all(_bien(s) for s in ahora)
        bien_antes += b_antes
        bien_ahora += b_ahora
        if b_antes and not b_ahora:
            empeoran += 1
        if ta.prospera(antes):
            fundados_antes += 1
            fundados_bien += b_ahora
        exacto = sum(1 for s in ahora if s == oro)
        tabla.append({"asunto": exp, "P": n, "oro": oro, "antes": antes, "ahora": ahora,
                      "cae": caida, "recalificado": preg in cs["pendientes"], "nota": nota})
        print(f"  {exp} P{n} · oro {oro:<11} · antes {antes:<11} · ahora "
              f"{' / '.join(ahora):<24} {'(recalificado)' if preg in cs['pendientes'] else '(no se toca)':<15}"
              f" · dirección {'OK' if b_ahora else 'MAL'} · exacto {exacto}/{len(ahora)}"
              + (" · CAE verificada" if any(caida) else ""))
    total = len(tabla)
    print(f"\n  en la dirección del engrose (prospera / no prospera): antes {bien_antes}/{total} · "
          f"ahora {bien_ahora}/{total} (en todas las corridas)")
    print(f"  los que quedaban «fundado» con la calificación de la otra vía: {fundados_antes}; "
          f"salen ahora como el engrose: {fundados_bien}/{fundados_antes}")
    print(f"  empeoran (bien antes, mal ahora): {empeoran}")
    seg = [s.get("segundos", 0) for ss in res.values() for s in ss]
    est = [s.get("estado") for ss in res.values() for s in ss]
    print(f"  corridas: {len(seg)} · estados {dict((x, est.count(x)) for x in set(est))} · "
          f"segundos: mín {min(seg or [0]):.0f} · máx {max(seg or [0]):.0f}")
    otros = [(exp, t, (s.get("resultados") or {}).get(t, {}).get("sentido"))
             for exp, ss in res.items() for s in ss for t in casos[exp]["pendientes"]
             if not any(casos[exp]["probs"][n - 1]["pregunta"] == t for (e2, n) in ca.ORO if e2 == exp)]
    for exp, t, s in otros:
        print(f"  (fuera de la tabla) {exp} «{t[:70]}» → {s}")
    avisos = [x for ss in res.values() for s in ss for x in (s.get("avisos") or [])]
    for x in avisos:
        print(f"  aviso: {x[:220]}")
    if a.salida:
        Path(a.salida).write_text(json.dumps({"tabla": tabla, "salidas": res}, ensure_ascii=False,
                                             default=str, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
