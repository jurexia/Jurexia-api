# -*- coding: utf-8 -*-
"""La predicción medida (jurimetria_prediccion.py), sin red. 23-nov-2026."""
import sys
import jurimetria_prediccion as jp

fallos = []


def ok(cond, que):
    if not cond:
        fallos.append(que)


assert jp.tablas().ok, "faltan datos/jurimetria_tasas.json.gz"

# Tasas: de lo general a lo particular, y dice de dónde sale.
t = jp.tasa("1:4:LAB", "amparo_directo", "Trabajo")
ok(t and t["nivel"] == "tribunal y materia" and t["n"] > 5000 and 0.3 < t["p"] < 0.65, f"4.º TCC trabajo del 1.º: {t}")
t = jp.tasa("99:1:ADM", "amparo_directo", "administrativa")
ok(t and t["nivel"].startswith("nacional"), f"tribunal sin datos → nacional: {t}")
ok(jp.tasa("1:4:LAB", "tipo_inexistente") is None, "tipo sin datos → None")
ok(jp.predecir(None, "tipo_inexistente")["advertencia"] == "SIN BASE SUFICIENTE", "sin base no se inventa")

# Revisión: lo que resolvió el juez y quién recurre mueven la probabilidad en el sentido medido.
alto = jp.predecir("6:2:ADM", "amparo_revision", "administrativa", juez="concedió el amparo", recurre="la autoridad")
bajo = jp.predecir("6:2:ADM", "amparo_revision", "administrativa", juez="sobreseyó", recurre="el quejoso")
solo = jp.predecir("6:2:ADM", "amparo_revision", "administrativa")
ok(alto["probabilidad"] > 0.6 > solo["probabilidad"] > bajo["probabilidad"],
   f"revisión: alto {alto['probabilidad']} · solo {solo['probabilidad']} · bajo {bajo['probabilidad']}")
ok(len(alto["ajustes"]) == 2 and "79%" in alto["precision_medida"], f"explica los ajustes: {alto}")
ok("79%" in solo.get("advertencia", ""), "sin juez ni recurrente, lo advierte")
d = jp.predecir("1:4:LAB", "amparo_directo", "Trabajo")
ok("leer el asunto" in d.get("advertencia", ""), "amparo directo: advierte que es estadística")

for txt, esperado in [("Concedió el amparo", "AMPARA"), ("negó", "NIEGA"), ("no ampara", "NIEGA"), ("sobreseyó", "SOBRESEE"),
                      ("concedió en parte", "AMPARA_PARCIAL"), ("", None)]:
    ok(jp.juez_canon(txt) == esperado, f"juez_canon({txt!r}) = {jp.juez_canon(txt)}")
for txt, esperado in [("la quejosa", "quejoso"), ("tercero interesado", "tercero"), ("autoridad responsable", "autoridad"),
                      ("Ministerio Público", "autoridad"), (None, None)]:
    ok(jp.recurre_canon(txt) == esperado, f"recurre_canon({txt!r})")

if fallos:
    print("FALLOS:")
    for f in fallos:
        print("  ·", f)
    sys.exit(1)
print("OK · predicción: tasas jerárquicas, revisión con juez y recurrente, advertencias honestas")
