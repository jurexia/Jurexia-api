# -*- coding: utf-8 -*-
"""El contexto de la petición: exclusión del fallo objetivo y banderas (rediseño, punto 8).

    .venv/bin/python test_contexto_taller.py
"""
import asyncio, datetime as dt, os, sys
sys.path.insert(0, ".")
import contexto_taller as ct

FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


print("\n1 · FECHAS Y NÚMEROS")
ok(ct.fecha("28-05-2026") == dt.date(2026, 5, 28) and ct.fecha("2026-05-28T10:00") == dt.date(2026, 5, 28)
   and ct.fecha("null") is None, "dd-mm-aaaa e ISO; lo ilegible es None")
ok(ct.numero_anio("3._ARA_17-2025") == (17, "2025"), "el número en cualquier grafía")

print("\n2 · LA EXCLUSIÓN")
e = ct.Exclusion.de_dict({"neuns": ["38118729.0"], "expedientes": ["274/2025", "ADC 492-2024"],
                          "fecha_corte": "2026-05-28", "holding_ids": ["h-1"]})
ok(e.excluye_fila({"neun": 38118729.0}), "el NEUN, aunque venga como float")
ok(e.excluye_fila({"alias": "274-2025"}) and e.excluye_fila({"expediente": "AR 492/2024"}),
   "el número y los de su serie, sin importar el tipo escrito")
ok(e.excluye_fila({"fecha": "28-05-2026"}) and e.excluye_fila({"fecha_sentencia": "2026-07-01"})
   and not e.excluye_fila({"fecha": "27-05-2026"}), "lo fechado desde el corte, fuera; lo anterior, no")
ok(not e.excluye_fila({"fecha": ""}), "sin fecha legible no se excluye por fecha (no se vacía el pozo)")
ok(e.excluye_holding({"holding_id": "h-1"}), "el holding del fallo objetivo")
ok(e.excluye_tesis({"fecha_publicacion": "2026-06-01"}) and not e.excluye_tesis({"fecha_publicacion": "2025-01-01"}),
   "las tesis publicadas desde el corte")
ok(ct.Exclusion.de_dict(e.a_dict()).a_dict() == e.a_dict(), "ida y vuelta a la sesión sin pérdida")

print("\n3 · SÓLO LA CASA, Y SIN FILTRARSE ENTRE PETICIONES")
ct.poner(False, {"exclusion": {"expedientes": ["1/2020"]}, "banderas": {"x": True}})
ok(ct.exclusion() is None and not ct.bandera("x"), "una cuenta de fuera no puede apagar fuentes ni encender banderas")
ct.poner(True, {"exclusion": {"expedientes": ["1/2020"]}})
ok(ct.exclusion() is not None, "la casa sí")


async def _dos():
    async def una(casa, exp):
        ct.poner(casa, {"exclusion": {"expedientes": [exp]}})
        await asyncio.sleep(0.01)
        x = ct.exclusion()
        return sorted(x.expedientes) if x else None
    return await asyncio.gather(una(True, "5/2021"), una(True, "9/2022"), una(False, "7/2023"))

r = asyncio.run(_dos())
ok(r == [[(5, "2021")], [(9, "2022")], None], f"dos peticiones a la vez no se pisan (salió {r})")

print("\n4 · LAS BANDERAS")
os.environ.pop("PRUEBA_BANDERA", None)
ct.poner(True, {})
ok(ct.bandera("b", "PRUEBA_BANDERA", "casa") is True, "por omisión «casa»: encendida para la casa")
ct.poner(False, {})
ok(ct.bandera("b", "PRUEBA_BANDERA", "casa") is False, "y apagada para los demás")
os.environ["PRUEBA_BANDERA"] = "todos"
ok(ct.bandera("b", "PRUEBA_BANDERA", "casa") is True, "con la variable en «todos», para todos")
os.environ["PRUEBA_BANDERA"] = "0"
ct.poner(True, {})
ok(ct.bandera("b", "PRUEBA_BANDERA", "casa") is False, "con «0», para nadie")
ct.poner(True, {"banderas": {"b": True}})
ok(ct.bandera("b", "PRUEBA_BANDERA", "casa") is True, "la bandera pedida en la evaluación manda sobre la variable")
os.environ.pop("PRUEBA_BANDERA", None)

print("\n5 · LAS FUENTES LA RESPETAN")
import fase_oaj as fo
ct.poner(True, {"exclusion": {"neuns": [700], "fecha_corte": "2026-01-01"}})
m = [(0.95, 700, {"alias": "700/2024", "fecha": "10-10-2024"}),
     (0.90, 701, {"alias": "701/2024", "fecha": "02-02-2026"}),
     (0.85, 702, {"alias": "702/2024", "fecha": "05-05-2025"})]
ok([x[1] for x in fo._sin_el_propio(m, "")] == [702],
   "la OAJ quita el NEUN objetivo y lo posterior al corte ANTES de contar rangos")
ct.poner(False, {})
ok([x[1] for x in fo._sin_el_propio(m, "")] == [700, 701, 702], "fuera de una evaluación, nada cambia")


async def _web():
    import fase_internet as fi
    ct.poner(True, {"exclusion": {"expedientes": ["1/2020"]}})
    return await fi.precedentes_verificados(None, "¿Pregunta?")

ok(asyncio.run(_web()).get("buscado") is False, "en una evaluación la web calla")

print("\n6 · EL BANCO KINGSTON")
import banco_kingston as bk
casos = bk.banco()
c274 = next(c for c in casos if "274-2025" in c["asunto"])
x = bk.exclusion_de(c274)
ok(x["neuns"] == [38118729] and x["fecha_corte"] == "2026-05-28" and x["expedientes"] == ["274/2025"],
   "la exclusión del 274/2025 sale del índice OAJ: NEUN y fecha de su sentencia")
c463 = next(c for c in casos if "463-2024" in c["asunto"])
ok(set(bk.exclusion_de(c463)["expedientes"]) == {"463/2024", "492/2024"}, "la serie del 463/2024 incluye el 492/2024")
ok(bk.fugas_en([{"filas": [{"expediente": "274/2025"}, {"expediente": "1/2020", "fecha": "01-06-2026"},
                           {"expediente": "2/2020", "fecha": "01-06-2024"}]}], x) == 2,
   "las fugas se cuentan: el propio asunto y lo posterior al corte")

print()
if FALLOS:
    print("FALLAS:\n  " + "\n  ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
