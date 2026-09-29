# -*- coding: utf-8 -*-
"""La revisión por código de la propuesta (rediseño, etapa 3; bandera «revision_semantica»).

    .venv/bin/python test_revision_semantica.py
"""
import ast, sys, types
import fase5_propuesta as f5

FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


P = lambda prob, s, alcanza=True, conf="alta": f5.Propuesta(problema=prob, sentido=s, razon="r", confianza=conf, alcanza=alcanza)  # noqa: E731
G = lambda s: types.SimpleNamespace(sentido=s)                                                                                       # noqa: E731
JER = {"¿Principal?": "principal", "¿Monto?": "accesorio", "¿Costas?": "accesorio"}

print("\n1 · LA AGREGACIÓN (274, 526, 722 del diagnóstico)")
av = f5.revisar_semantica(G("fundado"), [P("¿Principal?", "inoperante"), P("¿Monto?", "fundado", conf="media"),
                                         P("¿Costas?", "infundado")], JER)
ok(len(av) == 1 and "CONCEDE POR UN ACCESORIO" in av[0] and "¿Monto?" in av[0] and "confianza baja" not in av[0],
   "el principal no prospera y concede el monto: se dice cuál accesorio voltea el asunto")
av = f5.revisar_semantica(G("fundado"), [P("¿Principal?", "infundado"), P("¿Costas?", "fundado", conf="baja")], JER)
ok(av and "sin alcanzar o con confianza baja" in av[0], "y si ese accesorio es de confianza baja, lo subraya")
av = f5.revisar_semantica(G("fundado"), [P("¿Principal?", "infundado"), P("¿Costas?", "infundado")], JER)
ok(av and "ningún accesorio prospera" in av[0], "concede sin que nada prospere: se dice")
ok(f5.revisar_semantica(G("fundado"), [P("¿Principal?", "fundado"), P("¿Monto?", "fundado")], JER) == [],
   "el principal prospera: nada que avisar")
ok(f5.revisar_semantica(G("infundado"), [P("¿Principal?", "infundado"), P("¿Monto?", "fundado")], JER) == [],
   "el asunto no concede: esta revisión no tiene nada que decir")

print("\n2 · LAS RAZONES AUTÓNOMAS SIN COMBATIR")
av = f5.revisar_semantica(G("fundado"), [P("¿Principal?", "fundado")], JER, {"autonomas_sin_combatir": ["R2"]})
ok(av and "R2" in av[0] and "autónoma" in av[0], "concede con una autónoma sin combatir: aviso (no filtro)")

print("\n3 · NUNCA LANZA NI DECIDE")
for raro in ((None, None, None), (G(""), [], {}), (G("fundado"), [None, "x"], None), (object(), [P("x", "fundado")], {"x": 1})):
    try:
        f5.revisar_semantica(*raro)
        bien = True
    except Exception:
        bien = False
    ok(bien, f"forma rara: {type(raro[0]).__name__}")
_ps = [P("¿Principal?", "inoperante"), P("¿Monto?", "fundado")]
f5.revisar_semantica(G("fundado"), _ps, JER)
ok([p.sentido for p in _ps] == ["inoperante", "fundado"], "no toca ningún sentido")

print("\n4 · EL CABLEADO")
src = open("main.py", encoding="utf-8").read()
ok('_ctx_rs.rediseno("revision_semantica")' in src and "_f5.revisar_semantica(glob, propuestas, _jer_por_problema, _analisis_p)" in src,
   "la propuesta la corre con su bandera y el análisis neutral, y sólo añade avisos")

print()
if FALLOS:
    print("FALLAS:\n  " + "\n  ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
