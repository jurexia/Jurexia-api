# -*- coding: utf-8 -*-
"""EL BANCO DE LA DELIBERACIÓN, sin gastar — 28-sep-2026.

Lo que se comprueba es lo que no depende de la red: que el banco SE NIEGA a
correr sin `--si-gastar` e imprime el coste, que las compuertas están
escritas, que las métricas dan lo que tienen que dar (exactitud balanceada,
MCC, Wilson), que la muestra de la OAJ es estratificada y reproducible, que el
espejo excluye la MISMA sentencia (la fuga) y que la comprobación de citas
fuera del acervo cuenta lo que debe.

    .venv/bin/python test_banco_deliberacion.py
"""
import contextlib
import io
import json
import math
import os
import sys
import tempfile

import banco_deliberacion as bd

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


print("\n1 · se niega a gastar sin permiso")
for banco in ("kingston", "oaj", "631", "todos"):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = bd.main(["--banco", banco])
    out = buf.getvalue()
    ok(rc == 2 and "USD" in out and "--si-gastar" in out and "NO SE CORRE" in out,
       f"«{banco}» sin --si-gastar: imprime el coste y sale con 2")
ok("main" not in sys.modules, "sin --si-gastar no se importa main (ni red, ni arranque)")
est = bd.coste_estimado(["kingston", "oaj", "631"], 400, True)
ok(est["kingston"][0] == 24 and est["oaj"][0] == 400 and est["631"][0] == 1
   and 0 < est["total"][1] < 100, "el presupuesto por banco y el total")
ok(bd.coste_estimado(["oaj"], 100, True)["oaj"][1] > bd.coste_estimado(["oaj"], 100, False)["oaj"][1],
   "el brazo de la propuesta de hoy se presupuesta aparte")
por = bd.coste_estimado(["631"], 0, False)["631"][1]
ok(0.05 < por < 0.25, f"una deliberación ≈ {por:.3f} USD (lector2_potencia §8: 0.07–0.09 incremental)")
ok({c for c, _ in bd.COMPUERTAS} == {"oaj", "kingston", "citas", "claro", "631"},
   "las cinco compuertas, escritas antes de correr")

print("\n2 · las métricas")
P = [(True, True), (True, False), (False, False), (False, False)]
ok(bd.exactitud(P) == 0.75 and bd.exactitud_balanceada(P) == 0.75, "exactitud y balanceada")
ok(abs(bd.mcc(P) - 2 / math.sqrt(12)) < 1e-9, "MCC")
lo, hi = bd.wilson(13, 24)
ok(0.34 < lo < 0.36 and 0.72 < hi < 0.74, "Wilson de 13/24 ≈ 0.35–0.72: la puerta de cordura de Kingston")
base = [(o, False) for o, _ in P]
ok(bd.exactitud_balanceada(base) == 0.5, "«siempre no prospera» da 0.5 balanceada, aunque acierte la mitad")
ok(bd.exactitud([(True, None)]) == 0.0, "no decidir no es acertar")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    r = bd.informe("prueba", [{"oro": True, "predicho": True, "estado": "claro"},
                             {"oro": False, "predicho": True, "estado": "reñido"}])
ok("línea base" in buf.getvalue() and r["claro"]["n"] == 1 and r["claro"]["acierto"] == 1.0,
   "el informe imprime la línea base y separa lo «claro»")

print("\n3 · el oro de la OAJ")
ok(bd.oro_de_calificacion("fundado") is True and bd.oro_de_calificacion("parcialmente fundado") is True,
   "fundado y parcialmente fundado prosperan")
ok(bd.oro_de_calificacion("fundado pero inoperante") is False and bd.oro_de_calificacion("inconducente") is False,
   "«fundado pero inoperante» no prospera")
ok(bd.oro_de_calificacion("no se estudió") is None and bd.oro_de_calificacion("") is None,
   "«no se estudió» no es oro del principal")

print("\n4 · la muestra estratificada y reproducible")
with tempfile.TemporaryDirectory() as d:
    cal = ["fundado", "infundado", "inoperante", "no se estudió", "fundado", "infundado",
           "parcialmente fundado", "infundado", "fundado", "inoperante"]
    for tipo in ("AR", "AD"):
        for i, c in enumerate(cal):
            with open(os.path.join(d, f"{tipo}_{i + 1}-2024_{900000 + i}.json"), "w", encoding="utf-8") as fh:
                json.dump({"acto": "…", "planteamientos": [
                    {"pregunta": f"¿{tipo} {i}?", "resolvio": "…", "combate": "…",
                     "calificacion": c, "razon": "oculta"}]}, fh)
    m1 = bd.muestra_oaj(d, 4, 7)
    m2 = bd.muestra_oaj(d, 4, 7)
    ok([x["archivo"] for x in m1] == [x["archivo"] for x in m2], "con la misma semilla, la misma muestra")
    for tipo in ("AR", "AD"):
        xs = [x for x in m1 if x["tipo"] == tipo]
        ok(len(xs) == 4 and sum(x["oro"] for x in xs) == 2, f"{tipo}: 2 prosperan y 2 no")
    ok(all(x["calificacion_oro"] != "no se estudió" for x in m1), "«no se estudió» fuera de la muestra")
    ok(all(set(x["principal"]) == {"pregunta", "resolvio", "combate"} for x in m1),
       "al modelo sólo le llegan pregunta, lo resuelto y lo combatido (la calificación, oculta)")
    ok(m1[0]["numero"].endswith("/2024") and m1[0]["neun"].startswith("9"), "número y NEUN salen del archivo")

print("\n5 · sin fuga: el espejo sin la MISMA sentencia")
caso = {"tipo": "AR", "numero": "325/2024", "neun": "12345678"}
filas = [{"expediente": "325/2024", "tipo_asunto": "amparo en revisión"},
         {"expediente": "325/2024", "tipo_asunto": "amparo directo"},
         {"expediente": "1325/2024", "tipo_asunto": "amparo en revisión"},
         {"expediente": "99/2023", "tipo_asunto": "queja", "pdf_url": "https://x/12345678.pdf"}]
quedan = bd.sin_fuga(filas, caso)
ok([f["expediente"] for f in quedan] == ["325/2024", "1325/2024"]
   and quedan[0]["tipo_asunto"] == "amparo directo",
   "fuera el mismo AR 325/2024 y la fila con su NEUN; quedan el AD 325/2024 y el AR 1325/2024")

print("\n6 · el veredicto y las citas fuera del acervo")
v = {"A": {"prospera": True, "apoyos": []}, "B": {"prospera": False, "apoyos": []}}
ok(bd.veredicto({"recomendada": "A", "estado": "claro", "vias": v}) == (True, "claro"), "recomendada A: prospera")
ok(bd.veredicto({"recomendada": None, "inclinacion": "B", "estado": "reñido", "vias": v}) == (False, "reñido"),
   "reñido con inclinación: se mide la inclinación")
ok(bd.veredicto({"estado": "no_alcanza", "vias": v})[0] is None, "sin vía: indeterminado")
cat = {"T1": {"clase": "tesis", "registro": "2015688"}}
limpio = {"catalogo": cat, "vias": {"A": {"razon": "(registro 2015688) y $ 250000",
                                          "apoyos": [{"registro": "2015688", "en_acervo": True}]}}}
ok(bd.citas_fuera(limpio) == 0, "lo del catálogo y una cantidad en pesos no cuentan")
sucio = {"catalogo": cat, "por_que": ["según el 2099999"],
         "vias": {"A": {"razon": "", "apoyos": [{"registro": "2088888", "en_acervo": True}]}}}
ok(bd.citas_fuera(sucio) == 2, "un registro en el texto y un apoyo fuera del catálogo: 2")
ok(bd.citas_quitadas({"verificacion": {"referencias_quitadas": 3}}) == 3, "lo quitado se lee del contador")

if FALLOS:
    print(f"\n{len(FALLOS)} FALLA(S)")
    sys.exit(1)
print("\nTODO PASA")
