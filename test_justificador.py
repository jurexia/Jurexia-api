# -*- coding: utf-8 -*-
"""Un justificador por solución dentro de la deliberación (rediseño, etapa 3;
bandera «soluciones_por_desenlace»).

Reutiliza los datos y el cliente falso de test_deliberacion.py (sólo su parte
de preparación, antes de la primera comprobación).

    .venv/bin/python test_justificador.py
"""
import json, re, sys, types

_src = open("test_deliberacion.py", encoding="utf-8").read()
_prep = _src[:_src.index('print("\\n1 · el contrato')]
_ns: dict = {"__name__": "prep_deliberacion"}
exec(compile(_prep, "test_deliberacion.py", "exec"), _ns)
dl, correr, respuestas, Falso = _ns["dl"], _ns["correr"], _ns["respuestas"], _ns["Falso"]
import contexto_taller as ct
import justificador as jz

FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


ANALISIS = {"razones": [{"id": "R1", "relacion": "autonoma", "afirma": "La sustitución alteró la cosa juzgada."},
                        {"id": "R2", "relacion": "conjunta", "afirma": "No hubo cesión expresa."}],
            "autonomas_sin_combatir": []}
DEMOS = {"requisitos": [{"id": "Q1", "enunciado": "que el adquirente suceda en el derecho litigioso"}]}

print("\n1 · SIN LA BANDERA: LA DELIBERACIÓN DE SIEMPRE")
ct.poner(False, {})
cli = Falso(respuestas())
d = correr(cli, analisis=ANALISIS, demostracion=DEMOS)
ok("soluciones" not in d and set(d["vias"]) == {"A", "B"} and len(cli.de("ABOGADO")) == 2,
   "dos abogados, vías A y B, sin campo «soluciones» (paridad)")
ok(not any("TU SOLUCIÓN" in x[1] for x in cli.de("ABOGADO")),
   "los prompts de los abogados no llevan el anexo de solución")
_clave_sin = dl.clave_de("H", "", ["1"])

print("\n2 · CON LA BANDERA: UNO POR SOLUCIÓN")
ct.poner(True, {}, pruebas=True)
ok(dl.clave_de("H", "", ["1"]) != _clave_sin, "la clave de la marca cambia: no se reutiliza la binaria")
ok(dl.activa_para("prueba@iurexia.com"), "la deliberación corre para la cuenta de prueba aunque el entorno esté apagado")


def abogado_a(p, loc):
    base = json.loads(respuestas()(p.replace("", ""), {}) or "{}") if False else None
    tc = loc["tc"]
    return json.dumps({
        "sentido": "fundado", "razon": f"El adquirente es causahabiente [{tc}].",
        "regla": f"El causahabiente procesal puede sustituirse [{tc}].",
        "hechos": [{"afirma": "Se subroga.", "cita": "el adquirente del inmueble arrendado es causahabiente de la actora "
                                                     "y se subroga en sus derechos litigiosos", "fuente": "escrito"}],
        "subsuncion": "cae", "conclusion": "Fundado.",
        "objecion": {"de_la_otra_via": "cosa juzgada", "respuesta": "no"},
        "autoridad_contraria": [], "propongo_aplicar": [tc], "precedente_propio": [],
        "secundarios": [{"numero": 2, "relacion": "depende", "suerte": {"sentido": "innecesario", "razon": "x"}}],
        "sostenible": True,
        "razones_a_superar": [{"razon": "R1", "caracter": "autonoma", "como_se_supera": "la sustitución no altera la cosa juzgada"}],
        "aplicacion": [{"requisito": "que el adquirente suceda", "hechos": [
            {"afirma": "Se subroga.", "cita": "el adquirente del inmueble arrendado es causahabiente de la actora "
                                              "y se subroga en sus derechos litigiosos", "fuente": "escrito",
             "condicion": "acreditacion_impugnada"}]}],
        "efectos": [{"acto": "la resolución", "parte": "la quejosa", "efecto": "dejarla insubsistente"}],
        "pendientes": [{"que": "la escritura", "bloquea": True}], "resumen": "Prospera."})


cli = Falso(respuestas(abogado_a=abogado_a))
d = correr(cli, analisis=ANALISIS, demostracion=DEMOS)
sols = d.get("soluciones") or []
n_ab = len(cli.de("ABOGADO"))
ok(len(sols) >= 2 and n_ab == len(sols), f"un justificador por solución enumerada ({n_ab} llamadas, {len(sols)} soluciones)")
ok(all("TU SOLUCIÓN" in x[1] for x in cli.de("ABOGADO")), "cada prompt lleva el anexo de SU solución")
ok(all("R1 [AUTONOMA]" in x[1] and "Q1." in x[1] for x in cli.de("ABOGADO")),
   "y ve las razones de la responsable y los requisitos de la regla")
ok(set(d["vias"]) == {"A", "B"} and d["vias"]["A"].get("solucion") and d["vias"]["B"].get("solucion"),
   "la tarjeta sigue recibiendo A y B: la mejor de cada lado, con su solución")
ok(d["vias"]["A"]["prospera"] and not d["vias"]["B"]["prospera"], "A prospera y B no")
_sa = next(s for s in sols if s["id"] == d["vias"]["A"]["solucion"])
ok(_sa["razones_a_superar"] and _sa["aplicacion"] and _sa["efectos"] and "revision" in _sa,
   "la SolucionCandidata de A trae razones a superar, aplicación, efectos y su revisión")
ok(_sa["pendientes"] and _sa["pendientes"][0]["bloquea"] is True and d.get("estado"),
   "un pendiente que «bloquea» sólo informa: la deliberación termina igual")
ok(any(len(x[1]) > 0 for x in cli.de("JUEZ")), "el juez sigue decidiendo entre A y B")

print("\n2b · EL JUEZ BUSCA LA FALLA DECISIVA")
_jz_prompts = [x[1] for x in cli.de("JUEZ")]
ok(_jz_prompts and all("LA FALLA DECISIVA" in p and "LO QUE EL TALLER COMPROBÓ EN LA VÍA 1" in p for p in _jz_prompts),
   "con la bandera, el juez busca la falla decisiva de cada vía y ve la revisión por código")
ok(all("fallas" in pz for pz in d["juez"]["pasadas"] if pz.get("respondio")),
   "las fallas de cada pasada quedan en el documento del juez")
_base = respuestas()


def _con_falla(p, kw):
    r = _base(p, kw)
    if p.startswith("TAREA: JUEZ DE LAS DOS VÍAS") and r:
        x = json.loads(r)
        x["estado"] = "claro"
        x["fallas"] = {"1": {"falla": "no vence la razón autónoma R1", "fatal": True, "eslabon": "razon_autonoma"},
                       "2": {"falla": "ninguna grave", "fatal": False, "eslabon": ""}}
        # la vía 1 es la recomendada por el juez falso cuando es la «fundado»
        return json.dumps(x)
    return r


ct.poner(True, {}, pruebas=True)
cli2 = Falso(_con_falla)
d2 = correr(cli2, analisis=ANALISIS, demostracion=DEMOS)
_rec = d2.get("recomendada")
_fat_rec = any((pz.get("fallas") or {}).get(_rec, {}).get("fatal") for pz in d2["juez"]["pasadas"])
ok(_fat_rec and d2["estado"] != "claro",
   "una falla FATAL en la vía recomendada no deja el estado «claro» (no cambia la vía, sólo su certeza)")
ct.poner(False, {})
cli3 = Falso(respuestas())
d3 = correr(cli3)
ok(all("LA FALLA DECISIVA" not in x[1] for x in cli3.de("JUEZ")) and all("fallas" not in pz for pz in d3["juez"]["pasadas"]),
   "sin la bandera, el juez de siempre (prompt y documento sin fallas)")

print("\n3 · SIN UNA SOLUCIÓN DE CADA LADO, LA DE SIEMPRE")
ct.poner(True, {}, pruebas=True)
cli = Falso(respuestas())
d = correr(cli, tipo_asunto="reclamacion", analisis=ANALISIS)
ok("soluciones" not in d and len(cli.de("ABOGADO")) == 2,
   "tipo no reconocido: se delibera con las dos vías, como siempre")

print("\n4 · LAS PIEZAS PURAS")
ok(jz.anexo_solucion({}) == "" and jz.anexo_solucion(None) == "", "sin solución, sin anexo")
c = [{"id": "S1", "prospera": False, "respondio": True, "revision": {"estado": "con_pendientes"}},
     {"id": "S2", "prospera": True, "respondio": True, "revision": {"estado": "incompleta"}},
     {"id": "S3", "prospera": True, "respondio": True, "revision": {"estado": "completa"}},
     {"id": "S4", "prospera": True, "respondio": False, "revision": {"estado": "completa"}}]
ok(jz.elegir_vias(c) == ("S3", "S1"), "la mejor que prospera por su revisión (no la primera), y la que no")
c[2]["sostenible"] = False
ok(jz.elegir_vias(c)[0] == "S2", "la que su propio justificador declara insostenible va detrás")

ct.poner(False, {})
print()
if FALLOS:
    print("FALLAS:\n  " + "\n  ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
