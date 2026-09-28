"""La revisión antes de presentar, sin red — 28-sep-2026.

    .venv/bin/python test_revision_escrito.py

Lo que la ley pide a la demanda de amparo (artículos 108 y 175 de la Ley de
Amparo), lo que quedó sin llenar, el cierre, las frases rotas; que un escrito
completo salga limpio, que lo que falta se diga con su fundamento, que no
acuse a la prosa normal y que corra en tiempo lineal.
"""
import time

from revision_escrito import revisar_escrito, tipo_de_escrito, _plano

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


def niveles(r, texto):
    return [h["nivel"] for h in r["hallazgos"] if texto in h["que"]]


INDIRECTO = """QUEJOSO: Juan Pérez García
ASUNTO: Se promueve juicio de amparo indirecto.

C. JUEZ DE DISTRITO EN MATERIA ADMINISTRATIVA EN TURNO
P R E S E N T E

Juan Pérez García, por mi propio derecho, señalando como domicilio para oír notificaciones el
ubicado en Av. Juárez 10, promuevo demanda de amparo indirecto.

TERCERO INTERESADO: no existe.
AUTORIDADES RESPONSABLES: la Alcaldía Cuauhtémoc.
ACTO RECLAMADO: la orden de clausura.

ANTECEDENTES. Bajo protesta de decir verdad manifiesto que el 3 de agosto de 2026 se me notificó la clausura.

PRECEPTOS CONSTITUCIONALES VIOLADOS: artículos 14 y 16 constitucionales.

CONCEPTOS DE VIOLACIÓN. PRIMERO. La clausura carece de fundamentación.

Se acompañan copias de traslado para las partes.

PROTESTO LO NECESARIO
Ciudad de México, a 28 de septiembre de 2026

______________________________
JUAN PÉREZ GARCÍA"""

print("── el amparo indirecto completo ──")
r = revisar_escrito(INDIRECTO)
ok(r["tipo"] == "amparo_indirecto" and r["tipo_nombre"] == "demanda de amparo indirecto", "se reconoce")
ok(r["faltan"] == 0, f"no falta nada ({[h['que'] for h in r['hallazgos'] if h['nivel'] == 'falta']})")
ok(niveles(r, "bajo protesta de decir verdad") == ["bien"], "los antecedentes bajo protesta, bien")
ok(all(h["fundamento"].startswith("artículo 108") for h in r["hallazgos"] if "fracción" in h["fundamento"]),
   "con el artículo 108 como fundamento")

print("── lo que le falta ──")
sin = (INDIRECTO.replace("Bajo protesta de decir verdad manifiesto que", "Manifiesto que")
       .replace("TERCERO INTERESADO: no existe.\n", "")
       .replace("Se acompañan copias de traslado para las partes.", "")
       .replace("a 28 de septiembre de 2026", "a [DATO PENDIENTE: fecha]")
       + "\n[Nombre del abogado]")
r = revisar_escrito(sin)
falta_protesta = [h for h in r["hallazgos"] if "bajo protesta de decir verdad" in h["que"]]
ok(falta_protesta and falta_protesta[0]["nivel"] == "falta"
   and falta_protesta[0]["fundamento"] == "artículo 108, fracción V, de la Ley de Amparo",
   "sin «bajo protesta de decir verdad»: falta, con la fracción V del 108")
ok(niveles(r, "tercero interesado") == ["falta"], "sin tercero interesado: falta")
ok(niveles(r, "copias de traslado") == ["revise"], "sin copias: revíselo (artículo 110)")
ok(any(h["nivel"] == "falta" and "dato pendiente" in h["que"] for h in r["hallazgos"]), "el dato pendiente, falta")
ok(any(h["nivel"] == "falta" and "plantilla" in h["que"] for h in r["hallazgos"]), "la marca de plantilla, falta")
ok(r["hallazgos"][0]["nivel"] == "falta", "lo que falta va primero")

print("── el amparo directo ──")
directo = """C. MAGISTRADOS DEL TRIBUNAL COLEGIADO EN MATERIA CIVIL EN TURNO
Juan Pérez, con domicilio en Av. Juárez 10, promuevo demanda de amparo directo contra la sentencia definitiva.
TERCERO INTERESADO: María López, con domicilio en Calle Uno 2.
AUTORIDAD RESPONSABLE: Sala Civil.
ACTO RECLAMADO: la sentencia definitiva del 1 de agosto de 2026.
PRECEPTOS CONSTITUCIONALES VIOLADOS: artículos 14 y 16.
CONCEPTOS DE VIOLACIÓN. PRIMERO. …
PROTESTO LO NECESARIO. Ciudad de México, a 20 de agosto de 2026. ______________"""
r = revisar_escrito(directo)
ok(r["tipo"] == "amparo_directo", "se reconoce")
ok(niveles(r, "fecha en que se notificó") == ["falta"], "sin la fecha de notificación: falta (artículo 175, fracción V)")
ok(niveles(r, "expediente de la autoridad responsable") == ["revise"], "sin copias: revíselo (artículo 177)")
ok(not niveles(r, "copias de traslado"), "y no las del amparo indirecto (artículo 110)")
r = revisar_escrito(directo.replace("del 1 de agosto de 2026.", "del 1 de agosto de 2026, notificada el 5 de agosto de 2026."))
ok(niveles(r, "fecha en que se notificó") == ["bien"], "con ella, bien")
r = revisar_escrito(directo + "\nSe acompañan las copias de ley.")
ok(not niveles(r, "expediente de la autoridad responsable"), "con copias, no lo pide")

print("── otros escritos ──")
ok(tipo_de_escrito(_plano("PRIMER AGRAVIO. La sentencia recurrida… PROTESTO")) == "recurso", "un recurso")
ok(tipo_de_escrito(_plano("PRESTACIONES: a) el pago… HECHOS… PROTESTO")) == "demanda", "una demanda ordinaria")
ok(tipo_de_escrito(_plano("CONSIDERANDO. PRIMERO… SE RESUELVE: PRIMERO…")) == "resolucion", "una resolución")
r = revisar_escrito("CONSIDERANDO. PRIMERO. Estudio. SE RESUELVE: PRIMERO. Se confirma.")
ok(not any("cierre" in h["que"] for h in r["hallazgos"]), "a una resolución no se le pide «PROTESTO»")
r = revisar_escrito("La parte quejosa sostiene que la autoridad omitió fundar el acto. PROTESTO. Ciudad de México, a 2 de mayo de 2026. ________")
ok(not any("plantilla" in h["que"] for h in r["hallazgos"]), "«la parte quejosa» en prosa no es plantilla")
r = revisar_escrito("Texto con puntuación rota ,, aquí. PROTESTO LO NECESARIO")
ok(any(h["nivel"] == "revise" and h["donde"] for h in r["hallazgos"] if "puntuación" in h["que"].lower() or "signos" in h["que"]),
   "las frases rotas, a revisar y con dónde")

print("── tiempo lineal ──")
for nombre, texto in {
    "blancos": " " * 300_000,
    "corchetes abiertos": "[nombre " * 40_000,
    "equis": "X" * 300_000,
    "notificaciones sin fecha": "notificado el " * 25_000,
    "guiones bajos": "___ de " * 45_000,
}.items():
    t0 = time.perf_counter()
    revisar_escrito(texto)
    ms = (time.perf_counter() - t0) * 1000
    ok(ms < 2000, f"{nombre} ({len(texto):,} caracteres): {ms:.0f} ms")

print(f"\n{'TODO PASA' if not FALLOS else f'{len(FALLOS)} FALLA(S)'}")
raise SystemExit(1 if FALLOS else 0)
