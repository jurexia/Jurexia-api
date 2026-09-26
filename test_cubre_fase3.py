"""El «cubre» de la fase 3: se fusiona al descartar un repetido y la plantilla
ya no trae un ejemplo que copiar — 26-sep-2026.

Dos fallas del diagnóstico del estudio de fondo (w2, F6 y L11):
  · `_sin_repetidos` tiraba el problema repetido entero, y con él su «cubre»:
    los planteamientos que sólo él respondía se quedaban sin problema.
  · la plantilla decía `"cubre": [1, 2]` siempre, y en los asuntos de un solo
    concepto el modelo lo copiaba: el estudio leía «conceptos primero y
    segundo» donde había uno.

    .venv/bin/python test_cubre_fase3.py
"""
import inspect

import fases123_pipeline as f123

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


# El caso medido del ADC 642/2024: dos preguntas iguales palabra por palabra.
Q642 = ("¿La Sala responsable vulneró los derechos de legalidad, tutela judicial "
        "efectiva e impartición de justicia al tener por acreditada la identidad del inmueble?")
QID = "¿Se acreditó la identidad del inmueble reclamado con la confesión del demandado?"
QCOS = "¿Procede la condena en costas en ambas instancias?"
QPRES = "¿Se acreditó la prescripción adquisitiva por la cesión verbal alegada?"

print("\n1 · AL DESCARTAR UN REPETIDO, SU «cubre» SE QUEDA")
ps = [{"pregunta": Q642, "cubre": [1, 3], "jerarquia": "principal"},
      {"pregunta": QCOS, "cubre": [4], "jerarquia": "accesorio"},
      {"pregunta": Q642, "cubre": [2], "jerarquia": "accesorio"}]
r = f123._sin_repetidos(ps)
ok(len(r) == 2, f"el repetido se sigue descartando ({len(r)} problemas)")
ok(r[0]["cubre"] == [1, 2, 3], f"y el que se queda hereda su «cubre»: {r[0].get('cubre')}")
ok(r[1]["cubre"] == [4], "el otro no se toca")
ok(ps[0]["cubre"] == [1, 3], "sin mutar lo que devolvió el modelo")
ok(sorted(x for p in r for x in f123.cubre_de(p)) == [1, 2, 3, 4],
   "la unión sigue dando 1..4: nadie queda huérfano")

print("\n2 · LA FUSIÓN VA AL MÁS PARECIDO, Y LA JERARQUÍA DEL PRINCIPAL NO SE PIERDE")
Q_A = "¿La Sala valoró correctamente la prueba confesional del demandado sobre la posesión del predio?"
Q_B = "¿La Sala valoró correctamente la prueba testimonial ofrecida sobre la posesión del predio?"
Q_B2 = "¿La Sala valoró correctamente la prueba testimonial rendida sobre la posesión del predio?"
ps = [{"pregunta": Q_A, "cubre": [1], "jerarquia": "accesorio"},
      {"pregunta": Q_B, "cubre": [2], "jerarquia": "accesorio"},
      {"pregunta": Q_B2, "cubre": [3], "jerarquia": "principal"}]
r = f123._sin_repetidos(ps)
_por = {p["pregunta"]: p for p in r}
ok(len(r) == 2 and Q_B in _por and Q_B2 not in _por, "el tercero es repetido del segundo")
ok(_por[Q_B]["cubre"] == [2, 3] and _por[Q_A]["cubre"] == [1],
   "su «cubre» va al segundo, que es al que se parece más, no al primero")
ok(_por[Q_B]["jerarquia"] == "principal", "y si el descartado era el principal, el que queda lo es")

print("\n3 · `depende_de` SE RENUMERA")
ps = [{"pregunta": QID, "cubre": [1], "jerarquia": "principal", "depende_de": None},
      {"pregunta": QID.replace("reclamado", "demandado"), "cubre": [2], "depende_de": None},
      {"pregunta": QPRES, "cubre": [3], "depende_de": 1},
      {"pregunta": QCOS, "cubre": [4], "depende_de": 3},
      {"pregunta": "¿Debió estudiarse el agravio sobre la valoración de la prueba pericial?",
       "cubre": [5], "depende_de": 2}]
r = f123._sin_repetidos(ps)
ok(len(r) == 4, f"se descarta el segundo ({len(r)} problemas)")
ok(r[0]["cubre"] == [1, 2], "con su «cubre» fusionado")
ok(r[1]["depende_de"] == 1, "el que dependía del 1 sigue dependiendo del 1")
ok(r[2]["depende_de"] == 2, f"el que dependía del 3 ahora depende del 2: {r[2]['depende_de']}")
ok(r[3]["depende_de"] == 1, "el que dependía del repetido depende del que lo absorbió")
sin = [{"pregunta": QID, "depende_de": None}, {"pregunta": QCOS, "depende_de": 1}]
ok(f123._sin_repetidos(sin) == sin, "sin repetidos, nada cambia")
ok(f123._sin_repetidos([]) == [], "sin problemas, lista vacía")

print("\n4 · «cubre» SE LEE COMO VENGA Y SE CIÑE A LO CONTADO")
ok(f123.cubre_de({"cubre": [3, "1", 1]}) == [1, 3], "lista con cadenas y repetidos")
ok(f123.cubre_de({"cubre": "1, 2"}) == [1, 2], "una cadena ya no se lee letra a letra")
ok(f123.cubre_de({"cubre": 4}) == [4] and f123.cubre_de({}) == [] and f123.cubre_de({"cubre": None}) == [],
   "un entero suelto, o nada")
ok(f123._con_cubre_en_rango({"pregunta": "x", "cubre": [1, 2]}, 1)["cubre"] == [1],
   "un solo concepto y el [1, 2] copiado: queda [1]")
ok(f123._con_cubre_en_rango({"pregunta": "x", "cubre": [2, 9]}, 5)["cubre"] == [2],
   "con cinco contados, el 9 no existe")
ok(f123._con_cubre_en_rango({"pregunta": "x", "cubre": [2, 9]}, 0)["cubre"] == [2, 9],
   "sin conteo no hay rango que comprobar")
ok("cubre" not in f123._con_cubre_en_rango({"pregunta": "x"}, 3), "sin «cubre» no se inventa")

print("\n5 · LA PLANTILLA: UN TIPO, NO UN VALOR, Y SÓLO CUANDO SE PIDE")
p0 = f123.prompt_problemas("ACTO", "CONCEPTOS", False, "amparo_directo", 0)
p1 = f123.prompt_problemas("ACTO", "CONCEPTOS", False, "amparo_directo", 1)
p5 = f123.prompt_problemas("ACTO", "CONCEPTOS", False, "amparo_directo", 5)
ok('"cubre"' not in p0 and '"cubre"' not in p1, "sin conteo o con un solo concepto no se pide «cubre»")
ok('"cubre"' in p5 and "de 1 a 5" in p5, "con cinco, se pide, descrito por su tipo y su rango")
ok(all(x not in p for p in (p0, p1, p5) for x in ("[1, 2]", "[3, 7, 9]")),
   "ni el [1, 2] de la plantilla ni el [3, 7, 9] del reparto: ningún valor que copiar")
ok('"clase": "fondo|procesal|procedencia"' in p1 and '"jerarquia": "principal|accesorio"' in p1,
   "el resto de la plantilla sigue igual")
pr = f123.prompt_problemas("ACTO", "CONCEPTOS", True, "queja", 3, [2])
ok("SEGUNDO INTENTO" in pr and "planteamientos 2" in pr, "el segundo intento sigue nombrando los huérfanos")

print("\n6 · EL REPARTO SÓLO SE COMPRUEBA CUANDO SE PIDIÓ")
src = inspect.getsource(f123.correr)
i_rep = src.find("if _n_plant < 2:")
i_cub = src.find("_cub.extend(cubre_de(_p))")
ok(0 < i_rep < i_cub, "con un solo concepto no se busca «cubre»: ni segunda lectura ni aviso de huérfano falso")
ok("_con_cubre_en_rango(_p, _n_plant)" in src, "y el «cubre» se ciñe a lo contado antes de guardarse")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
