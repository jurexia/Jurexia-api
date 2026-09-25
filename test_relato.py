"""De qué va el asunto: el relato del paso 2 — 17-sep-2026.

David: «me gustaría una tarjeta más grande en la que al secretario se le
explique de qué va el caso (…) con una sola tarjeta en el pipeline». El relato
sale de los tres resúmenes, en paralelo con los problemas, se guarda con la
sesión y llega a la pantalla por /taller/contexto-del-asunto.

    .venv/bin/python test_relato.py
"""
import inspect

import fases123_pipeline as f123

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


print("\n1 · EL PROMPT CUENTA, NO RESUELVE")
p = f123.prompt_relato("ANT", "ACTO", "AGRAVIOS", True, "revision_fiscal",
                       "TITULAR DE LA SUBDELEGACIÓN", "SALA REGIONAL EN QUERÉTARO")
ok("Mira: el asunto tuvo su origen en que" in p and "Aquí empieza el problema, porque" in p
   and "Inconforme con esa determinación" in p, "lleva los conectores de David como hilo")
# «ESA DEMANDA» SE FUE EL 23-SEP-2026. David: en amparo directo es «cómo
# resolvió la responsable ese recurso de apelación» o «el juicio»; sin saber qué
# se resolvió, la pregunta no lo adivina.
ok("¿Qué resolvió la Sala?" in p and "esa demanda" not in p,
   "la pregunta nombra al órgano del tipo de asunto, y sin «esa demanda»")
pr = f123.prompt_relato("ANT", "ACTO", "AGRAVIOS", False, "amparo_directo",
                        lo_resuelto="ese recurso de apelación")
ok("¿Cómo resolvió la Sala responsable ese recurso de apelación?" in pr
   or "ese recurso de apelación?" in pr, "con lo resuelto, lo nombra")
ok("revisión fiscal" in p and "agravios" in p.lower() and "RECURRENTE" in p,
   "y el vocabulario de la revisión fiscal (recurso, agravios, recurrente)")
ok("no enumeres los problemas jurídicos" in p and "no adelantes cómo debería resolverse" in p,
   "prohíbe listar la litis (la pone la pantalla) y adelantar el sentido")
ok("no lo inventes" in p, "y prohíbe inventar")
ok("Quien promueve, según la ficha: TITULAR DE LA SUBDELEGACIÓN." in p
   and "según la ficha: SALA REGIONAL EN QUERÉTARO." in p, "la ficha ancla los nombres")
ok(p.index("ANTECEDENTES:\nANT") < p.index("LO QUE RESOLVIÓ LA SALA:\nACTO") < p.index("AGRAVIOS"),
   "los tres resúmenes van en orden")
ok("Juan José" not in p and "Subdelegación Querétaro del Órgano" not in p,
   "sin ejemplo literal que copiar (el molde va en el hilo, no en un caso)")
pa = f123.prompt_relato("A", "B", "C", False, "amparo_directo")
ok("amparo directo" in pa and "conceptos de violación" in pa and "QUEJOSO" in pa
   and "quién demandó a" in pa, "en amparo directo el hilo empieza por el juicio de origen")
ok("según la ficha" not in pa, "sin ficha no se inventa una")
pq = f123.prompt_relato("A", "B", "C", True, "queja")
ok("¿Qué resolvió el Juzgado de Distrito?" in pq, "en la queja, el Juzgado de Distrito")

print("\n2 · EL CAMPO Y EL PARALELO")
ok(hasattr(f123.Fases123(), "relato") and f123.Fases123().relato == "", "Fases123 lleva relato")
src = inspect.getsource(f123.correr)
ok("quejoso: str = \"\", responsable: str = \"\"" in inspect.getsource(f123.correr).split("\n")[3]
   or "quejoso" in str(inspect.signature(f123.correr)), "correr recibe los nombres de la ficha")
i_lanza = src.find("_tarea_relato = asyncio.ensure_future(_relato())")
i_probs = src.find("prompt_problemas(")
i_recoge = src.find("f.relato = await _tarea_relato")
ok(0 < i_lanza < i_probs < i_recoge, "se lanza ANTES de los problemas y se recoge al final: van en paralelo")
ok("except Exception" in src[i_recoge - 60:i_recoge + 200], "si el relato falla, el adelanto sale igual")

print("\n3 · LAS PUERTAS")
import redactor_adelanto as ra
ok('quejoso=getattr(e, "quejoso", "") or ""' in inspect.getsource(ra.generar),
   "el adelanto pasa quejoso y responsable")
m = open("main.py", encoding="utf-8").read()
ok('"relato": getattr(r.fases, "relato", "") or "",' in m, "se guarda con la sesión (estado.fases.relato)")
i_ctx = m.find('@app.get("/taller/contexto-del-asunto")')
ok(0 < i_ctx < m.find('"relato": getattr(f, "relato", "") or "",', i_ctx) < m.find('@app.post("/taller/consultar")'),
   "y /taller/contexto-del-asunto lo devuelve")
i_res = m.find("def _taller_recuperar_sesion(")
ok('for k, v in (est.get("fases") or {}).items():\n        setattr(f, k, v)' in m[i_res:i_res + 9000],
   "el rescate de la sesión repone todos los campos de fases, relato incluido")
try:
    fr = open("../jurexia-frontend-git/src/components/sentencia/api.ts", encoding="utf-8").read()
    pg = open("../jurexia-frontend-git/src/app/taller/page.tsx", encoding="utf-8").read()
    ok("relato: String(j.relato ?? '')" in fr and "relato: string;" in fr, "el frontend lee el campo")
    ok("De qué va el asunto" in pg and "¿A qué se reduce la litis?" in pg
       and "delAsunto.relato.split(" in pg, "y la tarjeta está en el paso 2 con la litis numerada")
    ok("abierto={paso === 'adelanto' && !delAsunto.relato}" in pg
       and "delAsunto.problemas.length > 0 && !delAsunto.relato && (" in pg,
       "con relato, los pliegues técnicos van cerrados y la lista de problemas no se repite")
except FileNotFoundError:
    print("   (frontend no está al lado: se omite)")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
