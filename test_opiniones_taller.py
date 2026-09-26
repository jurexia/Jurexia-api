"""La opinión del secretario sobre cada proyecto — 24-sep-2026.

    .venv/bin/python test_opiniones_taller.py
"""
import opiniones_taller as ot

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


print("\n1 · LO QUE SE GUARDA")
f, e = ot.limpiar("4", "poco", {"citas": "mejorar", "sentido": "bien", "inventado": "bien", "efectos": "regular"},
                  "  Las citas del marco no aplicaban.\n\nY los efectos…  ", "")
ok(not e and f["calificacion"] == 4 and f["correccion"] == "poco", "nota y corrección válidas")
ok(f["aspectos"] == {"citas": "mejorar", "sentido": "bien"}, "sólo aspectos del catálogo con valores válidos")
ok(f["sobre_sentencia"].startswith("Las citas") and "\n\n" in f["sobre_sentencia"], "el texto conserva sus párrafos")
ok(f["sobre_taller"] is None, "lo vacío va como nulo, no como cadena vacía")

print("\n1-bis · LOS DOS ASPECTOS DE LA DECISIÓN 5 (26-sep-2026)")
_claves = [k for k, _ in ot.ASPECTOS]
ok("exhaustividad" in _claves and "sin_repeticion" in _claves,
   "«Contesta cada argumento» y «No repite lo ya razonado» están en el catálogo")
ok(dict(ot.ASPECTOS)["exhaustividad"] == "Contesta cada argumento"
   and dict(ot.ASPECTOS)["sin_repeticion"] == "No repite lo ya razonado",
   "con las palabras de David")
ok(len(set(_claves)) == len(_claves), "ninguna clave repetida (el auditor suma por clave)")
f5, e5 = ot.limpiar(None, "", {"exhaustividad": "mejorar", "sin_repeticion": "bien"}, "", "")
ok(not e5 and f5["aspectos"] == {"exhaustividad": "mejorar", "sin_repeticion": "bien"},
   "se guardan como cualquier otro aspecto")

print("\n2 · LO QUE NO")
ok(ot.limpiar()[1] == ["La opinión llegó vacía: no hay nada que guardar."], "vacía: se rechaza")
ok("La calificación va de 1 a 5." in ot.limpiar(9)[1], "nota fuera de rango")
ok(any("corrección" in x for x in ot.limpiar(None, "tal cual")[1]), "corrección desconocida")
ok(not ot.limpiar(None, "", {}, "", "La pantalla del paso 3 es lenta")[1], "sólo texto del taller: vale")
ok(len(ot.limpiar(None, "", {}, "x" * 9000, "")[0]["sobre_sentencia"]) == ot.TOPE_TEXTO, "el texto se topa")

print("\n3 · LA FOTO DEL PROYECTO")
EST = {"encargo": {"tipo_asunto": "amparo_directo", "materia": "administrativa"},
       "proyecto": {"version": 2, "sentido_global": "", "avisos": ["A", "B"],
                    "criterios": [{"sentido": "inoperante", "jerarquia": "accesorio"},
                                  {"sentido": "fundado", "jerarquia": "principal"}]},
       "proyectos": [{"version": 2, "avisos": ["A", "B"]},
                     {"version": 1, "sentido_global": "infundado", "avisos": ["C"]}]}
f2 = ot.foto_del_proyecto(EST)
ok(f2["version"] == 2 and f2["sentido"] == "fundado" and f2["avisos_n"] == 2,
   "sin versión: la última, con el sentido del PRINCIPAL")
f1 = ot.foto_del_proyecto(EST, 1)
ok(f1["version"] == 1 and f1["sentido"] == "infundado" and f1["avisos"] == ["C"],
   "con versión: ESA versión, con su sentido global y sus avisos")
ok(ot.foto_del_proyecto({})["version"] == 0, "sin estado no revienta")

print("\n4 · LAS PUERTAS ESTÁN CONECTADAS")
src = open("main.py", encoding="utf-8").read()
ok('@app.post("/taller/opinion")' in src and '@app.get("/taller/opinion")' in src, "GET y POST /taller/opinion")
ok('on_conflict="email,expediente,version"' in src, "opinar otra vez sobre la misma versión la corrige")
ok("_ot.foto_del_proyecto(" in src, "y se guarda con la foto del proyecto")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
