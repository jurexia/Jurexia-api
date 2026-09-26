"""CALIBRACIÓN DE LA CAÍDA CON EL PRINCIPAL — la regla vieja contra la nueva.

POR QUÉ EXISTE (26-sep-2026, paso 2 del plan que aprobó David). El árbol de
decisión declaraba «inoperante porque descansa en la premisa desestimada» a
todo accesorio con `depende_de`, y el ADC 722/2025 salió con la congruencia de
la condena y las costas sin contestar. La regla nueva (`arbol_decision`,
`presupuesto`) sólo tumba el accesorio cuyo argumento da por cierta la premisa
del principal, y eso tiene que constar. Antes de dar la regla por buena se
mide aquí, sin llamar a ningún modelo:

  A. EN TODAS LAS SESIONES REALES del taller (incluidas las de terceros:
     sólo se leen, no se genera nada), cuántas veces cada regla aplica la
     caída, en dos escenarios: el principal como lo propuso el motor, y el
     principal desestimado (lo que haría un secretario que no le da la razón).
  B. CONTRA LOS ENGROSES REALES (banco Kingston, campo «oro»): en los asuntos
     donde el engrose desestimó el principal, cómo trató cada accesorio que
     la fase 3 colgó de él —caído, contestado con la misma respuesta del
     principal, o estudiado con razón propia—, y qué hace cada regla. La
     lectura de los engroses está abajo (`ORO`), con lo que se leyó.

Uso (con el .env cargado; la base sólo se LEE):
    .venv/bin/python calibrar_arbol.py --volcar /ruta/sesiones.json
    .venv/bin/python calibrar_arbol.py /ruta/sesiones.json [--vieja origin/main]
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
CAE = "Descansa en la premisa que se desestimó"

# ═══ LO QUE HICIERON LOS ENGROSES (leído a mano el 26-sep-2026) ═══════════
# Asuntos del banco Kingston cuya sesión de la cuenta de casa tiene
# accesorios con `depende_de` del principal y cuyo engrose DESESTIMÓ el
# principal. «cae»: el engrose lo declaró sin estudio de fondo por descansar
# en lo desestimado. «misma_respuesta»: lo contestó con la razón del principal,
# pero calificándolo (infundado). «estudiado»: le dio respuesta propia.
# Fuera: 463/2024 y 192/2025 (el «oro» es de otro asunto), 704/2022 y
# 810/2025 (el engrose concede por el principal: es la otra dirección, que no
# cambia), 93/2026 (el oro está incompleto) y 174/2026 P3 (no se identifica
# una respuesta suya en el engrose).
ORO = {
    ("274/2025", 2): ("estudiado", "reclusión y emplazamiento: infundado, y además inoperante por no preparar la violación"),
    ("274/2025", 3): ("estudiado", "porcentaje de la pensión: la Sala sí fundó y motivó"),
    ("263/2025", 2): ("misma_respuesta", "fundamentación del repudio: contestado con la razón del principal, infundado"),
    ("103/2025", 2): ("estudiado", "objeciones/tacha: inoperante con razón propia (el tribunal sí las valoró)"),
    ("103/2025", 3): ("misma_respuesta", "fundamentación y motivación: contestado junto con el principal, infundado"),
    ("400/2024", 2): ("estudiado", "régimen de aportaciones: «igualmente infundado», con razón propia"),
    ("702/2022", 2): ("estudiado", "calificación de la multa: inoperante por inatendible, razón propia"),
    ("174/2026", 2): ("estudiado", "doble jornada: examinada en el fondo, infundado"),
    ("174/2026", 4): ("estudiado", "perspectiva de género: examinada en el fondo"),
    ("43/2025", 2): ("estudiado", "carga de la prueba y presunciones del Código de Comercio: razón propia"),
    ("529/2024", 2): ("cae", "embargo: «inoperante, al sustentarse en disidencias que ya han sido desestimadas»"),
    ("529/2024", 3): ("estudiado", "costas: infundado con razón propia (buena fe no exime), además de remitir al principal"),
    ("590/2024", 2): ("estudiado", "autorización electrónica: examinada a fondo, infundado"),
    ("722/2025", 2): ("estudiado", "prestación B: ineficaz, con las cláusulas del contrato"),
    ("722/2025", 3): ("estudiado", "costas de alzada: ineficaz, con el art. 136"),
    ("641/2024", 3): ("estudiado", "entrega del inmueble: infundado, la Sala valoró las constancias en su conjunto"),
}
CASA = "administracion@iurexia.com"


def volcar(ruta: str) -> None:
    """Lee (SÓLO LEE) las sesiones del taller con lo que el árbol necesita."""
    import os

    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    out, ini = [], 0
    while True:
        r = sb.table("taller_sesiones").select(
            "id, email, expediente, propuestas, "
            "problemas:estado->fases->problemas, "
            "tipo:estado->encargo->tipo_asunto, "
            "proyecto_crit:estado->proyecto->criterios, "
            "glob:estado->propuesta->respuesta->global").order("id") \
            .range(ini, ini + 49).execute()
        out += r.data or []
        if len(r.data or []) < 50:
            break
        ini += 50
    Path(ruta).write_text(json.dumps(out, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"{len(out)} sesiones → {ruta}")


def _cargar_vieja(ref: str):
    """El árbol tal como está en `ref` (por omisión origin/main)."""
    src = subprocess.run(["git", "show", f"{ref}:arbol_decision.py"], cwd=AQUI,
                         capture_output=True, text=True, check=True).stdout
    tmp = Path(tempfile.mkdtemp()) / "arbol_viejo.py"
    tmp.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("arbol_viejo", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _prospera(s: str) -> bool:
    import tipos_asunto as _ta
    return _ta.prospera(s)


def _escenario(ses: dict, desestimar: bool):
    """(problemas, criterios, checklist, propuestas, sentido_motor, tipo) como
    los arma el resolver en modo acervo; con `desestimar`, el principal
    infundado y marcado por el secretario."""
    probs = [p for p in (ses.get("problemas") or []) if isinstance(p, dict)]
    props = [p for p in (ses.get("propuestas") or []) if isinstance(p, dict)]
    if not probs or not props:
        return None
    jer = {p.get("pregunta"): (p.get("jerarquia") or "") for p in probs}
    if not any(v == "principal" for v in jer.values()):
        jer[probs[0].get("pregunta")] = "principal"
    crit = [{"problema": p.get("problema", ""), "sentido": p.get("sentido", ""),
             "razonamiento": p.get("razon", "") or "",
             "jerarquia": jer.get(p.get("problema"), "accesorio") or "accesorio"}
            for p in props]
    pr = next((c for c in crit if c["jerarquia"] == "principal"), None)
    if pr is None or not pr["sentido"]:
        return None
    if desestimar:
        if _prospera(pr["sentido"]):
            pr["sentido"], pr["razonamiento"] = "infundado", ""
        pr["tocado"] = True
    glob = ses.get("glob") or {}
    return (probs, crit, list(glob.get("checklist") or []),
            [{"problema": p.get("problema", ""), "sentido": p.get("sentido", ""),
              "razon": p.get("razon", "") or "", "alcanza": p.get("alcanza", True),
              "sentido_propio": p.get("sentido_propio") or "",
              "razon_propia": p.get("razon_propia") or ""} for p in props],
            str(glob.get("sentido") or ""), str(ses.get("tipo") or ""))


def _correr(mod, esc) -> list:
    probs, crit, cl, props, smot, tipo = copy.deepcopy(esc)
    mod.aplicar(probs, crit, cl, props, sentido_motor=smot, tipo_asunto=tipo)
    return crit


def _cae(c: dict) -> bool:
    return str(c.get("razonamiento") or "").startswith(CAE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", nargs="?")
    ap.add_argument("--volcar")
    ap.add_argument("--vieja", default="origin/main")
    a = ap.parse_args()
    if a.volcar:
        volcar(a.volcar)
        return
    sys.path.insert(0, str(AQUI))
    import arbol_decision as nueva
    vieja = _cargar_vieja(a.vieja)
    S = json.loads(Path(a.dump).read_text(encoding="utf-8"))

    print("═══ A. TODAS LAS SESIONES ═══")
    for desest in (False, True):
        n_ses = n_acc = v_cae = n_cae = cambian_otros = 0
        cuentas_v, cuentas_n = set(), set()
        for ses in S:
            esc = _escenario(ses, desest)
            if esc is None:
                continue
            n_ses += 1
            cv, cn = _correr(vieja, esc), _correr(nueva, esc)
            for i, (x, y) in enumerate(zip(cv, cn)):
                if x["jerarquia"] == "principal":
                    continue
                n_acc += 1
                if _cae(x):
                    v_cae += 1
                    cuentas_v.add(ses["email"])
                if _cae(y):
                    n_cae += 1
                    cuentas_n.add(ses["email"])
                if not _cae(x) and not _cae(y) and x["sentido"] != y["sentido"]:
                    cambian_otros += 1
        print(f"  principal {'DESESTIMADO' if desest else 'como lo propuso el motor'}: "
              f"{n_ses} sesiones, {n_acc} accesorios · caída vieja {v_cae} "
              f"({len(cuentas_v)} cuentas) · caída nueva {n_cae} ({len(cuentas_n)} cuentas) · "
              f"otros cambios de sentido {cambian_otros}")

    print("\n═══ B. CONTRA LOS ENGROSES (principal desestimado, como en el oro) ═══")
    casa = {s["expediente"]: s for s in S if s.get("email") == CASA}
    aciertos = {"vieja": 0, "nueva": 0}
    total = 0
    for (exp, n), (trato, nota) in ORO.items():
        ses = casa.get(exp)
        esc = _escenario(ses, True) if ses else None
        if esc is None:
            print(f"  {exp} P{n}: sin sesión")
            continue
        cv, cn = _correr(vieja, esc), _correr(nueva, esc)
        preg = esc[0][n - 1]["pregunta"]
        x = next(c for c in cv if c["problema"] == preg)
        y = next(c for c in cn if c["problema"] == preg)
        total += 1
        # «cae» sólo acierta con caída; «estudiado» y «misma_respuesta», con
        # una calificación que se razona (el engrose la calificó).
        for nom, c in (("vieja", x), ("nueva", y)):
            if (trato == "cae") == _cae(c):
                aciertos[nom] += 1
        print(f"  {exp} P{n} · oro {trato:<15} · vieja "
              f"{'CAE' if _cae(x) else x['sentido']:<10} · nueva "
              f"{'CAE' if _cae(y) else y['sentido']:<10} · {nota}")
    print(f"\n  coinciden con el engrose: vieja {aciertos['vieja']}/{total} · "
          f"nueva {aciertos['nueva']}/{total}")


if __name__ == "__main__":
    main()
