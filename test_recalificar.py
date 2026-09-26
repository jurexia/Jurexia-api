import re
"""RECALIFICAR CON LA PREMISA DEL CAMBIO DE SENTIDO (26-sep-2026) — sin modelo.

David: «si cambio sentido hay que tumbar y regenerar con la premisa del cambio
de sentido». Con un asunto SINTÉTICO (nada de un expediente real) y dobles del
modelo y de la base:

  1 · quién se tumba, en las dos vías (el principal no prospera / prospera);
  2 · lo tocado, la razón que él tecleó y el global dictado no se tocan;
  3 · la vuelta a la vía del motor: nada que recalificar, vale lo del motor;
  4 · la clave: estable, sin orden, con la razón, con el adelanto;
  5 · aplicar lo recalificado: de la misma clave sí, de otra no; la caída
      verificada con su fórmula; «innecesario» sólo si la premisa lo deja;
  6 · la validación y la cita (misma verificación que `presupuesto`);
  7 · el estado en la fila (funciones puras), junto al del plan;
  8 · el respaldo por fallo: sin calificar, primero en ADVERTENCIAS (v2 a v4),
      la v1 intacta, el guion del plan con su rótulo;
  9 · los gemelos por AST: recalifican en la tarea y ANTES del plan, igual;
 10 · de punta a punta con la base y el modelo de mentira: calcula, guarda,
      reutiliza sin llamar, espera la en curso, y los dos gemelos deciden igual.

    .venv/bin/python test_recalificar.py
"""
import ast
import asyncio
import copy
import json
import os
import sys
import time
import types

os.environ.pop("ESTUDIO_PROMPT", None)
os.environ.pop("PLAN_ESTUDIO", None)

import arbol_decision as ad
import fase6_estudio as f6
import fases123_pipeline as f123
import plan_estudio as pe
import recalificar as rc

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


AQUI = os.path.dirname(os.path.abspath(__file__))

# ═══ EL ASUNTO DE PRUEBA (sintético) ════════════════════════════════════════
P1 = "¿La cesión del contrato liberó a la cedente de responder por las rentas reclamadas?"
P2 = "¿La condena en costas se sustentó en la conducta procesal de la cedente?"
P3 = "¿La Sala debió pronunciarse sobre la excepción de pago parcial opuesta en la contestación?"
P4 = "¿Se violaron las reglas del procedimiento al desechar la prueba pericial contable ofrecida?"
COMB1 = ("la arrendadora reconoció a la cesionaria como arrendataria mediante la cesión del "
         "contrato, por lo que la cedente dejó de responder por las rentas")
COMB2 = ("la cesionaria es la única obligada conforme a la cesión del contrato, por lo que la "
         "condena en costas a la cedente carece de sustento")
COMB3 = "la Sala omitió estudiar la excepción de pago parcial acreditada con recibos"
COMB4 = "se desechó la pericial contable ofrecida en tiempo, lo que trascendió al resultado"


def fase3(*pp):
    base = {P1: {"pregunta": P1, "jerarquia": "principal", "combate": COMB1, "cubre": [1]},
            P2: {"pregunta": P2, "jerarquia": "accesorio", "depende_de": 1, "combate": COMB2, "cubre": [2]},
            P3: {"pregunta": P3, "jerarquia": "accesorio", "depende_de": None, "combate": COMB3, "cubre": [3]},
            P4: {"pregunta": P4, "jerarquia": "accesorio", "depende_de": 1, "combate": COMB4,
                 "clase": "procesal", "cubre": [4]}}
    return [copy.deepcopy(base[p]) for p in pp]


def props(s1="fundado", s2="fundado", s3="infundado", s4="fundado", con=(P1, P2, P3)):
    d = {P1: (s1, "la cesión liberó a la cedente"), P2: (s2, "las costas siguen al principal"),
         P3: (s3, "los recibos no acreditan el pago"), P4: (s4, "la pericial era idónea")}
    return [{"problema": p, "sentido": d[p][0], "razon": d[p][1], "alcanza": True,
             "sentido_propio": "", "razon_propia": ""} for p in con]


def crit(s1="infundado", props_=None, tocado1=True, **extra):
    pr = props_ or props()
    c = []
    for i, p in enumerate(pr):
        c.append({"problema": p["problema"], "sentido": p["sentido"], "razonamiento": p["razon"],
                  "jerarquia": "principal" if i == 0 else "accesorio"})
    c[0]["sentido"], c[0]["razonamiento"] = s1, extra.get("razon1", "")
    if tocado1:
        c[0]["tocado"] = True
    return c


def por(cs):
    return {c["problema"]: c for c in cs}


# ═══════════════════════════════════════════════════════════════════════════
print("\n1 · QUIÉN SE TUMBA, EN LAS DOS VÍAS")
# (a) el motor propuso el principal fundado; el secretario lo desestima.
c = crit("infundado")
av, det = ad.aplicar(fase3(P1, P2, P3), c, [], props(), tipo_asunto="amparo_directo",
                     huella_adelanto="h1")
ok(por(c)[P2]["sentido"] == "" and por(c)[P2]["razonamiento"] == "",
   "no prospera: el accesorio relacionado se TUMBA (sin sentido ni razón)")
ok(det[P2].get("recalificar") and det[P2].get("de") == "por_recalificar"
   and det[P2].get("clave_recalificar") and det[P2].get("principal") == P1,
   "…con `recalificar`, `de: por_recalificar`, la clave y su principal")
ok(por(c)[P3]["sentido"] == "infundado" and not det[P3].get("recalificar"),
   "no prospera: el que no se relaciona con el principal no se toca (contrato)")
ok(rc.pendientes(det) == [P2] and any("SE RECALIFICAN CON TU PREMISA 1" in a and P2[:40] in a for a in av),
   "`pendientes` lo lista y el aviso lo nombra")
ok(not any("otra vía: revisa" in a for a in av), "ya no se avisa «revisa la de la otra vía»: se tumba")
# (b) el motor propuso el principal infundado; el secretario lo declara fundado.
pr_b = props("infundado", "infundado", "infundado")
c = crit("fundado", pr_b)
av, det = ad.aplicar(fase3(P1, P2, P3), c, [], pr_b, tipo_asunto="amparo_directo")
ok(por(c)[P2]["sentido"] == "innecesario" and not det[P2].get("recalificar"),
   "prospera: el que depende queda sin materia por la regla, no se recalifica")
ok(por(c)[P3]["sentido"] == "" and det[P3].get("recalificar"),
   "prospera: el que no depende conserva lo escrito con el principal sin prosperar → se tumba")
# (c) la suerte escrita PARA ESTA VÍA con su razón no se tumba; sin razón, sí.
CL = [{"numero": 1}, {"numero": 2, "relacion": "depende",
                      "si_no_prospera": {"sentido": "infundado", "razon": "la buena fe no exime de costas"}}]
c = crit("infundado")
av, det = ad.aplicar(fase3(P1, P2, P3), c, CL, props(), tipo_asunto="amparo_directo")
ok(por(c)[P2]["sentido"] == "infundado" and not det[P2].get("recalificar"),
   "la suerte que el motor escribió para esta vía, con su razón, se respeta")
CL2 = [{"numero": 1}, {"numero": 2, "relacion": "depende", "si_no_prospera": {"sentido": "infundado", "razon": ""}}]
c = crit("infundado")
av, det = ad.aplicar(fase3(P1, P2, P3), c, CL2, props(), tipo_asunto="amparo_directo")
ok(det[P2].get("recalificar"), "…sin su razón, se recalifica")
# (d) la caída verificada no se recalifica.
CL3 = [{"numero": 1}, {"numero": 2, "relacion": "depende",
                       "si_no_prospera": {"sentido": "inoperante", "razon": "parte de que la cesionaria es la única obligada"},
                       "presupone": {"premisa": "la cesionaria es la única obligada",
                                     "cita": "la cesionaria es la única obligada conforme a la cesión del contrato",
                                     "causa_propia": None}}]
c = crit("infundado")
av, det = ad.aplicar(fase3(P1, P2, P3), c, CL3, props(), tipo_asunto="amparo_directo")
ok(por(c)[P2]["razonamiento"].startswith(ad.CAE_CON_PRINCIPAL) and not det[P2].get("recalificar"),
   "la caída verificada (cita en su planteamiento) cae con el principal: no se recalifica")
# (e) el tema distinto no se recalifica.
c = crit("infundado")
av, det = ad.aplicar(fase3(P1, P2, P3), c, [{"numero": 1}, {"numero": 2, "tema_distinto": True}],
                     props(), tipo_asunto="amparo_directo")
ok(not det[P2].get("recalificar") and por(c)[P2]["sentido"] == "fundado",
   "el tema distinto no cuelga del principal: no se recalifica")
# (f) la procesal que la guarda conserva con la calificación de la otra vía, sí.
pr_f = props(con=(P1, P2, P4))
c = crit("infundado", pr_f)
av, det = ad.aplicar(fase3(P1, P2, P4), c, [], pr_f, tipo_asunto="amparo_directo")
ok(det[P4].get("recalificar") and det[P4].get("procesal") and det[P4].get("guarda") == "procesal",
   "la procesal sin suerte para esta vía se recalifica, marcada procesal (se decide: 74-V y 174)")
ok(not any("VIOLACIÓN PROCESAL y NO se declaró caída" in a and P4[:40] in a for a in av),
   "…y no se le dice «se estudia con su calificación» a la que se tumbó")
# (g) sin la vía del motor no se sabe si cambió: nada se tumba.
c = crit("infundado")
av, det = ad.aplicar(fase3(P1, P2, P3), c, [], [p for p in props() if p["problema"] != P1],
                     tipo_asunto="amparo_directo")
ok(not rc.pendientes(det), "sin la propuesta del principal ni el global, nada se recalifica")
# (h) el principal que el motor propuso en la misma vía (el 722/2025 real).
pr_h = props("infundado", "infundado", "infundado")
c = crit("infundado", pr_h)
av, det = ad.aplicar(fase3(P1, P2, P3), c, [], pr_h, tipo_asunto="amparo_directo")
ok(not rc.pendientes(det) and por(c)[P2]["sentido"] == "infundado",
   "el principal ya iba en la vía del motor (como el 722/2025): nada se recalifica")

print("\n2 · LO QUE ES DEL SECRETARIO NO SE TOCA")
c = crit("infundado")
c[1]["tocado"] = True
av, det = ad.aplicar(fase3(P1, P2, P3), c, [], props(), tipo_asunto="amparo_directo")
ok(por(c)[P2]["sentido"] == "fundado" and not det[P2].get("recalificar") and det[P2]["de"] == "tuya",
   "lo que marcó a mano se queda (su palabra manda)")
c = crit("infundado")
av, det = ad.aplicar(fase3(P1, P2, P3), c, [], props(), tocados={P2}, tipo_asunto="amparo_directo")
ok(por(c)[P2]["sentido"] == "fundado" and not rc.pendientes(det), "también si llega en `tocados`")
c = crit("infundado")
c[1]["sentido"], c[1]["razonamiento"] = "", "la cedente litigó de buena fe"
av, det = ad.aplicar(fase3(P1, P2, P3), c, [], props(), tipo_asunto="amparo_directo")
ok(por(c)[P2]["sentido"] == "" and por(c)[P2]["razonamiento"] == "la cedente litigó de buena fe"
   and det[P2].get("recalificar") and det[P2].get("razon_suya"),
   "la razón que él tecleó sin elegir sentido se queda; sólo el sentido se recalifica")
_k_suya = det[P2]["clave_recalificar"]
c = crit("infundado")
c[1]["sentido"], c[1]["razonamiento"] = "", "la cedente litigó de buena fe"
av, det = ad.aplicar(fase3(P1, P2, P3), c, [], props(), tipo_asunto="amparo_directo",
                     recalificadas={"clave": _k_suya, "resultados": {P2: {
                         "sentido": "infundado", "razon": "la del modelo, que no se usa aquí",
                         "presupone": None, "verificado": False}}})
ok(por(c)[P2]["sentido"] == "infundado" and por(c)[P2]["razonamiento"] == "la cedente litigó de buena fe",
   "de lo recalificado se toma el sentido; la razón sigue siendo la suya")
_c0 = crit("infundado")
_, _d0 = ad.aplicar(fase3(P1, P2, P3), _c0, [], props(), tipo_asunto="amparo_directo")
ok(rc.clave_de(_d0) == _k_suya,
   "su razón NO entra en la clave: la pantalla la devolverá con el sentido y la clave no cambia")
# La vuelta: la pantalla devuelve el sentido recalificado con SU razón; lo
# guardado trae la suya y se vuelve a poner (pantalla y documento, iguales).
c = crit("infundado")
c[1]["sentido"], c[1]["razonamiento"] = "infundado", "la cedente litigó de buena fe"
av, det = ad.aplicar(fase3(P1, P2, P3), c, [], props(), tipo_asunto="amparo_directo",
                     recalificadas={"clave": _k_suya, "resultados": {P2: {
                         "sentido": "infundado", "razon": "la del modelo, que no se usa aquí",
                         "presupone": None, "verificado": False,
                         "razon_suya": "la cedente litigó de buena fe"}}})
ok(por(c)[P2]["sentido"] == "infundado" and por(c)[P2]["razonamiento"] == "la cedente litigó de buena fe"
   and det[P2].get("razon_suya"), "en la vuelta, su razón se repone desde lo guardado")
_pr_s, _ac_s = rc.entradas(types.SimpleNamespace(fases=f123.Fases123(problemas=fase3(P1, P2, P3))),
                           [f6.Criterio(problema=P1, sentido="infundado", jerarquia="principal"),
                            f6.Criterio(problema=P2, sentido="", razonamiento="la cedente litigó de buena fe")],
                           {P2: {"recalificar": True, "principal": P1, "razon_suya": True}})
ok(_ac_s[0]["razon_suya"] == "la cedente litigó de buena fe"
   and "la cedente litigó de buena fe" in rc.prompt(types.SimpleNamespace(
       fases=f123.Fases123(), encargo=types.SimpleNamespace(tipo_asunto="amparo_directo")), None, _pr_s, _ac_s),
   "y el modelo la recibe como dato de ese planteamiento")
c = [{"problema": p["problema"], "sentido": "infundado", "razonamiento": "",
      "jerarquia": "principal" if i == 0 else "accesorio"} for i, p in enumerate(props())]
av, det = ad.aplicar(fase3(P1, P2, P3), c, [], props(), tipo_asunto="amparo_directo",
                     global_dictado=True)
ok(all(x["sentido"] == "infundado" for x in c) and not rc.pendientes(det),
   "el global dictado: su brocha es su palabra y no se recalifica nada")
c = crit("infundado")
ad.aplicar(fase3(P1, P2, P3), c, [], props(), tipo_asunto="amparo_directo")
ok(c[0]["sentido"] == "infundado", "el principal que dictó nunca cambia")

print("\n3 · LA VUELTA A LA VÍA DEL MOTOR")
c = crit("infundado")
ad.aplicar(fase3(P1, P2, P3), c, [], props(), tipo_asunto="amparo_directo")
vuelta = [dict(x) for x in c]
vuelta[0]["sentido"] = "fundado"                # el principal vuelve a lo del motor
av, det = ad.aplicar(fase3(P1, P2, P3), vuelta, [], props(), tipo_asunto="amparo_directo")
ok(not rc.pendientes(det), "con el principal otra vez en la vía del motor, nada que recalificar")
ok(por(vuelta)[P2]["sentido"] != "" and por(vuelta)[P2]["sentido"] in ("fundado", "innecesario"),
   f"el tumbado que vuelve vacío recibe lo del motor y la regla de esa vía: {por(vuelta)[P2]['sentido']}")
r_ = ad.reparto_para_pantalla(fase3(P1, P2, P3), [dict(x, tocado=(i == 0)) for i, x in enumerate(vuelta)],
                              [], props(), tipo_asunto="amparo_directo")
ok(all(not x["recalificar"] for x in r_["criterios"]), "/taller/reparto dice lo mismo")

print("\n4 · LA CLAVE")
k1 = rc.clave(P1, "infundado", "", [P2, P3], "h1", "amparo_directo")
ok(k1 == rc.clave(P1, "Infundado", "  ", [P3, P2], "h1", "amparo_directo"),
   "estable: mayúsculas, espacios y el orden de los pendientes no la cambian")
ok(k1 != rc.clave(P1, "infundado", "otra razón", [P2, P3], "h1", "amparo_directo"),
   "la razón del secretario entra (es la premisa)")
ok(k1 != rc.clave(P1, "infundado", "", [P2, P3], "h2", "amparo_directo"), "el adelanto entra")
ok(k1 != rc.clave(P1, "inoperante", "", [P2, P3], "h1", "amparo_directo"), "el sentido entra")
c1, c2 = crit("infundado"), crit("infundado")
_, d1 = ad.aplicar(fase3(P1, P2, P3), c1, [], props(), tipo_asunto="amparo_directo", huella_adelanto="h1")
c2[1]["sentido"], c2[1]["razonamiento"] = "infundado", "lo que devolvió la recalificación"
_, d2 = ad.aplicar(fase3(P1, P2, P3), c2, [], props(), tipo_asunto="amparo_directo", huella_adelanto="h1")
ok(rc.clave_de(d1) == rc.clave_de(d2) != "",
   "quién entra no depende de lo que trae: la pantalla devuelve lo recalificado y la clave no cambia")

print("\n5 · APLICAR LO RECALIFICADO")
c = crit("infundado")
_, det = ad.aplicar(fase3(P1, P2, P3), c, [], props(), tipo_asunto="amparo_directo", huella_adelanto="h1")
k = rc.clave_de(det)
buena = {"clave": k, "resultados": {P2: {"sentido": "infundado",
                                         "razon": "la condena en costas atendió al resultado del juicio",
                                         "presupone": None, "verificado": False}}}
c = crit("infundado")
av, det = ad.aplicar(fase3(P1, P2, P3), c, [], props(), tipo_asunto="amparo_directo",
                     huella_adelanto="h1", recalificadas=buena)
ok(por(c)[P2]["sentido"] == "infundado" and por(c)[P2]["razonamiento"].startswith("la condena en costas")
   and det[P2]["de"] == "recalificada" and not det[P2].get("recalificar")
   and det[P2]["por_que"].startswith("recalificada por el motor con tu premisa"),
   "la de la misma clave se aplica, con `de: recalificada` y su por qué")
ok(any("RECALIFICADOS CON TU PREMISA 1" in a for a in av) and not rc.pendientes(det),
   "y se dice; ya no queda nada pendiente")
c = crit("infundado")
_, det = ad.aplicar(fase3(P1, P2, P3), c, [], props(), tipo_asunto="amparo_directo",
                    huella_adelanto="OTRO", recalificadas=buena)
ok(por(c)[P2]["sentido"] == "" and det[P2].get("recalificar"), "la de otra clave (otro adelanto) no se aplica")
c = crit("infundado", razon1="la cesión no se notificó a la arrendadora")
_, det = ad.aplicar(fase3(P1, P2, P3), c, [], props(), tipo_asunto="amparo_directo",
                    huella_adelanto="h1", recalificadas=buena)
ok(por(c)[P2]["sentido"] == "", "…ni la hecha con otra razón del secretario")
c = crit("infundado")
_, det = ad.aplicar(fase3(P1, P2, P3), c, [], props(), tipo_asunto="amparo_directo",
                    huella_adelanto="h1", recalificadas={k: {"estado": "listo", **buena}})
ok(por(c)[P2]["sentido"] == "infundado", "la rama entera de la fila ({clave: casilla}) también vale")
caida = {"clave": k, "resultados": {P2: {"sentido": "infundado", "razon": "x y z de fondo",
                                         "presupone": {"cita": "la cesionaria es la única obligada",
                                                       "por_que": "que la cesionaria es la única obligada"},
                                         "verificado": True}}}
c = crit("infundado")
_, det = ad.aplicar(fase3(P1, P2, P3), c, [], props(), tipo_asunto="amparo_directo",
                    huella_adelanto="h1", recalificadas=caida)
ok(por(c)[P2]["sentido"] == "inoperante" and por(c)[P2]["razonamiento"].startswith(ad.CAE_CON_PRINCIPAL)
   and det[P2].get("relacion") == "presupone",
   "la caída verificada se escribe con la fórmula que el estudio reconoce")
# «innecesario» en la vía que prospera se escribe con la fórmula de la sustracción.
pr_b = props("infundado", "infundado", "infundado")
c = crit("fundado", pr_b)
_, det = ad.aplicar(fase3(P1, P2, P3), c, [], pr_b, tipo_asunto="amparo_directo", huella_adelanto="h1")
kb = rc.clave_de(det)
c = crit("fundado", pr_b)
_, det = ad.aplicar(fase3(P1, P2, P3), c, [], pr_b, tipo_asunto="amparo_directo", huella_adelanto="h1",
                    recalificadas={"clave": kb, "resultados": {P3: {"sentido": "innecesario",
                                                                    "razon": "prosperando la liberación no hay pago que revisar",
                                                                    "presupone": None, "verificado": False}}})
ok(por(c)[P3]["sentido"] == "innecesario" and por(c)[P3]["razonamiento"].startswith(ad.SIN_MATERIA),
   "«innecesario» recalificado lleva la fórmula de la sustracción")
# Lo que el secretario marque después la sustituye.
c = crit("infundado")
c[1].update(sentido="fundado", razonamiento="su razón", tocado=True)
_, det = ad.aplicar(fase3(P1, P2, P3), c, [], props(), tipo_asunto="amparo_directo",
                    huella_adelanto="h1", recalificadas=buena)
ok(por(c)[P2]["sentido"] == "fundado" and det[P2]["de"] == "tuya",
   "lo que él marque después manda sobre lo recalificado")

print("\n6 · LA VALIDACIÓN Y LA CITA")
PR = {"problema": P1, "sentido": "infundado", "razon": "", "fase3": fase3(P1)[0]}
AC = [{"problema": P2, "numero": 2, "fase3": fase3(P2)[0], "procesal": False},
      {"problema": P4, "numero": 4, "fase3": fase3(P4)[0], "procesal": True}]
res, faltas, avs = rc.validar({"planteamientos": [
    {"numero": 2, "sentido": "Infundado", "razon": "la condena atendió al resultado del juicio y a la conducta",
     "presupone": None},
    {"numero": 4, "sentido": "fundado", "razon": "la pericial se ofreció en tiempo y era idónea para probar",
     "presupone": None}]}, AC, PR)
ok(not faltas and res[P2]["sentido"] == "infundado" and res[P4]["sentido"] == "fundado",
   "dos bien: pasan, con el sentido normalizado")
res, faltas, avs = rc.validar({"planteamientos": [
    {"numero": 2, "sentido": "innecesario", "razon": "no hay nada que estudiar en absoluto aquí"},
    {"numero": 4, "sentido": "fundado", "razon": ""}]}, AC, PR)
ok(len(faltas) == 2 and not res, "«innecesario» con el principal sin prosperar y la razón vacía: faltas")
res, faltas, avs = rc.validar({"planteamientos": [
    {"numero": 2, "sentido": "infundado", "razon": "la condena atendió al resultado del juicio y a la conducta"}]},
    AC, PR)
ok(any("falta el planteamiento 4" in f for f in faltas), "el que no volvió es una falta")
res, faltas, avs = rc.validar({"planteamientos": [
    {"numero": 2, "sentido": "inoperante", "razon": "parte de una premisa que ya se desestimó en el principal",
     "presupone": {"cita": "la cesionaria es la única obligada conforme a la cesión del contrato",
                   "por_que": "que la cesionaria es la única obligada"}},
    {"numero": 4, "sentido": "fundado", "razon": "la pericial se ofreció en tiempo y era idónea para probar",
     "presupone": {"cita": "se desechó la pericial contable ofrecida en tiempo", "por_que": "x"}}]}, AC, PR)
ok(res[P2]["verificado"] and res[P2]["presupone"]["cita"],
   "la cita literal de lo que se combate, con ancla del principal: la caída consta")
ok(res[P4]["presupone"] is None and not res[P4]["verificado"] and P4 in avs,
   "una procesal nunca cae con el principal: se descarta y se avisa")
res, faltas, avs = rc.validar({"planteamientos": [
    {"numero": 2, "sentido": "infundado", "razon": "la condena atendió al resultado del juicio y a la conducta",
     "presupone": {"cita": "la cesionaria no pagó ninguna renta nunca jamás", "por_que": "x"}},
    {"numero": 4, "sentido": "fundado", "razon": "la pericial se ofreció en tiempo y era idónea para probar"}]},
    AC, PR)
ok(not faltas and not res[P2]["verificado"] and "no consta" in avs.get(P2, "")
   and res[P2]["sentido"] == "infundado",
   "una cita que no consta no es falta: se descarta la caída y queda su calificación de fondo")
_ev_ref = ad.presupuesto({"presupone": {"cita": "la cesionaria no pagó ninguna renta nunca jamás"}},
                         fase3(P2)[0], fase3(P1)[0])
ok(_ev_ref["motivo"] == "cita_no_consta", "(la misma verificación que `arbol_decision.presupuesto`)")
PRp = dict(PR, sentido="fundado")
ok("innecesario" in rc.catalogo(True, False) and "innecesario" not in rc.catalogo(True, True)
   and "innecesario" not in rc.catalogo(False, False) and "sin_materia" not in rc.catalogo(True, False),
   "el catálogo: «innecesario» sólo si el principal prospera y nunca en una procesal")
_p = rc.prompt(types.SimpleNamespace(fases=f123.Fases123(resumen_acto="La Sala condenó."),
                                     encargo=types.SimpleNamespace(tipo_asunto="amparo_directo")),
               None, PR, AC[:1])
ok("cesión del contrato liberó a la cedente" in _p and "combate diciendo: " + COMB2[:40] in _p
   and "las costas siguen al principal" not in _p,
   "el prompt lleva la premisa y lo que se combate, y NO la calificación tumbada ni su razón")
ok("presupone" in _p and "74, fracción V, y 174" in _p and '"planteamientos"' in _p,
   "describe la caída y la guarda procesal y pide JSON")

print("\n7 · EL ESTADO EN LA FILA, JUNTO AL DEL PLAN")
d, dec = rc.fila_pedir(None, "k1", "h", 100.0)
ok(dec == "lanzar" and d["recalificaciones"]["k1"]["estado"] == "en_curso"
   and d["recalificaciones_corridas"] == 1 and d["huella"] == "h" and "planes" in d,
   "sin fila: se reserva la corrida, en un documento que el plan también entiende")
ok(rc.fila_pedir(d, "k1", "h", 110.0) == (None, "en_curso"), "en curso: se espera, no se duplica")
ok(rc.fila_pedir(d, "k1", "h", 100.0 + rc.ABANDONADA_S + 1)[1] == "lanzar",
   "sin latido: abandonada, se relanza")
d2, _ = rc.fila_latido(d, "k1", "h", 150.0)
ok(d2["recalificaciones"]["k1"]["latido"] == 150.0, "el latido")
sal = {"estado": "listo", "resultados": {P2: {"sentido": "infundado", "razon": "r"}}, "avisos": []}
d3, e3 = rc.fila_resultado(d2, "k1", "h", sal, 12.0, 160.0)
ok(e3 == "listo" and rc.fila_pedir(d3, "k1", "h", 170.0) == (None, "listo")
   and rc.guardadas(d3, "h")["k1"]["resultados"][P2]["sentido"] == "infundado",
   "listo: se reutiliza y `guardadas` lo entrega al árbol")
ok(rc.fila_resultado(d3, "k1", "OTRA", sal, 1.0, 1.0) == (None, "otro_adelanto")
   and rc.guardadas(d3, "OTRA") == {}, "lo de otro adelanto ni se escribe ni se lee")
dp, decp = pe.fila_pedir(d3, "kplan", "h", 200.0)
ok(decp == "lanzar" and dp["recalificaciones"]["k1"]["estado"] == "listo",
   "el pedido del plan no se lleva la rama de las recalificaciones")
dn, _ = pe.fila_pedir(d3, "kplan", "h-nuevo", 200.0)
ok("recalificaciones" not in dn, "un adelanto nuevo empieza de cero (la del viejo no sirve)")
d4, e4 = rc.fila_resultado(d3, "k2", "h", {"estado": "error", "resultados": {}, "avisos": ["x"]}, 1, 1)
ok(rc.fila_pedir(d4, "k2", "h", 2.0)[1] == "lanzar", "un error del proveedor se vuelve a lanzar")
d5, _ = rc.fila_resultado(d4, "k3", "h", {"estado": "fallo", "resultados": {}, "avisos": []}, 1, 1)
ok(rc.fila_pedir(d5, "k3", "h", 2.0) == (None, "fallo")
   and rc.estado_de(d5, "h", "k3", 2.0)["estado"] == "fallo", "lo que no pasó la validación no se recalcula")
dd = None
for i in range(rc.TOPE_CORRIDAS):
    dd, _ = rc.fila_pedir(dd, f"t{i}", "h", 1.0)
    dd, _ = rc.fila_resultado(dd, f"t{i}", "h", sal, 1, float(i))
ok(rc.fila_pedir(dd, "otra", "h", 9.0) == (None, "tope")
   and len(dd["recalificaciones"]) <= rc.MAX_CASILLAS, "seis corridas por adelanto, seis casillas")

print("\n8 · EL RESPALDO SI NO LLEGA")


class _Resp:
    def __init__(self, txt):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=txt),
                                              finish_reason="stop")]
        self.usage = None


class Modelo:
    def __init__(self, *resp, espera=0.0, falla=False):
        self.resp, self.kw, self.espera, self.falla = list(resp), [], espera, falla
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._crear))

    async def _crear(self, **kw):
        self.kw.append(kw)
        if self.espera:
            await asyncio.sleep(self.espera)
        if self.falla:
            raise RuntimeError("503")
        p = self.resp.pop(0) if len(self.resp) > 1 else self.resp[0]
        return _Resp(p if isinstance(p, str) else json.dumps(p, ensure_ascii=False))


def fases():
    return f123.Fases123(resumen_acto="La Sala condenó a la cedente al pago de rentas y costas.",
                         resumen_conceptos="Primero. La cesión liberó a la cedente.\nSegundo. Costas.",
                         problemas=fase3(P1, P2, P3), fuentes=["", ""])


def resultado(variante="v2"):
    r = types.SimpleNamespace(fases=fases())
    r.encargo = types.SimpleNamespace(tipo_asunto="amparo_directo", es_recurso=False,
                                      variante_estudio=variante, formato="", suplencia={},
                                      conceptos_violacion="", propuesta_global={}, plan={}, guion="")
    r.avisos = []
    return r


BUENO = {"planteamientos": [{"numero": 2, "sentido": "infundado",
                             "razon": "la condena en costas atendió al resultado del juicio",
                             "presupone": None}]}
_pr, _ac = PR, [AC[0]]
m = Modelo("no es json", "tampoco")
out = asyncio.run(rc.recalificar(resultado(), None, _pr, _ac, cliente=m, clave_="k"))
ok(out["estado"] == "fallo" and not out["resultados"] and len(m.kw) == 2,
   "la validación no pasa dos veces: «fallo», sin resultados, dos llamadas")
ok(m.kw[0]["model"] == f123.MODELO_FASES and m.kw[0]["reasoning_effort"] == "medium"
   and m.kw[0]["response_format"] == {"type": "json_object"},
   "el modelo de las fases, razonamiento MEDIO, JSON estricto")
ok("LO QUE FALLÓ EN TU RESPUESTA ANTERIOR" in m.kw[1]["messages"][0]["content"]
   and "LO QUE FALLÓ" not in m.kw[0]["messages"][0]["content"], "el reintento dice qué falló")
m = Modelo("no es json", BUENO)
out = asyncio.run(rc.recalificar(resultado(), None, _pr, _ac, cliente=m, clave_="k"))
ok(out["estado"] == "listo" and out["resultados"][P2]["sentido"] == "infundado", "el reintento pasa: «listo»")
m = Modelo(BUENO, espera=0.5)
out = asyncio.run(rc.recalificar(resultado(), None, _pr, _ac, cliente=m, clave_="k", tope_s=0.1))
ok(out["estado"] == "error" and any("no llegó" in a for a in out["avisos"]),
   "vence el tope: «error» (la siguiente petición lo reintenta)")
m = Modelo(BUENO, falla=True)
out = asyncio.run(rc.recalificar(resultado(), None, _pr, _ac, cliente=m, clave_="k"))
ok(out["estado"] == "error", "un 503 del proveedor: «error», nunca lanza")
# El estudio: los sin calificar, primero en ADVERTENCIAS, desde la v2; la v1 intacta.
C_sin = [f6.Criterio(problema=P1, sentido="infundado", razonamiento="la cesión no se notificó",
                     jerarquia="principal"),
         f6.Criterio(problema=P2, sentido="", razonamiento="", jerarquia="accesorio")]
C_con = [C_sin[0], f6.Criterio(problema=P2, sentido="infundado", razonamiento="r", jerarquia="accesorio")]
mat2 = f6.Material(tipo_asunto="amparo_directo", materia="civil", variante="v2")
p2_sin = f6.prompt_estudio("acto", "conceptos", C_sin, mat2)
p2_con = f6.prompt_estudio("acto", "conceptos", C_con, mat2)
_blq = p2_sin.split("PLANTEAMIENTOS SIN CALIFICAR")[-1]
ok("PLANTEAMIENTOS SIN CALIFICAR" in p2_sin and P2 in _blq and "PRIMERO" in _blq
   and "la cesión no se notificó" in _blq,
   "v2: el bloque de datos con los sin calificar, la premisa y PRIMERO en ADVERTENCIAS")
ok("SENTIDO: SIN CALIFICAR" in p2_sin and "PLANTEAMIENTOS SIN CALIFICAR" not in p2_con,
   "…y sólo cuando los hay (la v2 de siempre no cambia)")
mat1 = f6.Material(tipo_asunto="amparo_directo", materia="civil", variante="v1")
p1_sin = f6.prompt_estudio("acto", "conceptos", C_sin, mat1)
# LA v1 TAMBIÉN, SÓLO CUANDO LOS HAY (integración, 26-sep-2026): es la de
# producción para quien no es de casa, y con el sentido vacío salía «SENTIDO: »
# y «en parte infundados y en parte ». Sin sin calificar, idéntica.
p1_con = f6.prompt_estudio("acto", "conceptos", C_con, mat1)
ok("PLANTEAMIENTOS SIN CALIFICAR" in p1_sin and "SENTIDO: SIN CALIFICAR" in p1_sin
   and "en parte  " not in p1_sin and not re.search(r"en parte\s*[.,]", p1_sin)
   and "PLANTEAMIENTOS SIN CALIFICAR" not in p1_con,
   "la v1: con un sin calificar, el bloque y una apertura sin huecos; sin ellos, como siempre")
ok("SIN CALIFICAR TRAS TU CAMBIO DE SENTIDO" in rc.aviso_sin_calificar([P2], "infundado")
   and P2[:60] in rc.aviso_sin_calificar([P2], "infundado"), "el aviso al secretario nombra los que quedaron")
# El guion del plan (v4): su rótulo propio, no «no lo califiques».
_probs_pc = pe.problemas_del_criterio(C_sin, types.SimpleNamespace(problemas=fase3(P1, P2)))
ok(_probs_pc[1]["sin_calificar"] and not _probs_pc[0]["sin_calificar"],
   "el plan distingue el tumbado sin calificar del que él dejó sin sentido")
_plan = {"problemas": [{"id": 1, "sentido": "infundado"}, {"id": 2, "sentido": "", "sin_calificar": True},
                       {"id": 3, "sentido": ""}],
         "segmentos": [{"id": "C2.a", "problema_id": 2, "pendiente": "sentido", "etiqueta": ""},
                       {"id": "C3.a", "problema_id": 3, "pendiente": "sentido", "etiqueta": ""}]}
_g = pe.vista(_plan)
ok("SIN CALIFICAR TRAS EL CAMBIO DE SENTIDO: C2.a" in _g and "SIN SENTIDO FIJADO: C3.a" in _g,
   "el guion separa los dos rótulos")
ok("SIN CALIFICAR TRAS EL CAMBIO DE SENTIDO: el secretario" in pe.bloque(_g)
   and "SIN CALIFICAR TRAS EL CAMBIO DE SENTIDO: el secretario" not in pe.bloque("GUION\nAPARTADO 1"),
   "y su descripción entra sólo cuando el guion trae el rótulo (la v4 de siempre no cambia)")

print("\n9 · LOS GEMELOS POR AST")
SRC = open(os.path.join(AQUI, "main.py"), encoding="utf-8").read()
A = ast.parse(SRC)
FN = {n.name: n for n in A.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def llamadas(fn, nombre):
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == nombre]


def ruta(fn):
    for d in fn.decorator_list:
        if isinstance(d, ast.Call) and d.args and isinstance(d.args[0], ast.Constant):
            return getattr(d.func, "attr", ""), d.args[0].value
    return None


for nombre in ("taller_resolver_stream", "taller_resolver"):
    fn = FN[nombre]
    src = ast.get_source_segment(SRC, fn)
    rcl = llamadas(fn, "_taller_recalificar_para")
    ok(len(rcl) == 1 and len(llamadas(fn, "_taller_recalificado_al_resolver")) == 1,
       f"{nombre}: recalifica una vez, con la función común, y aplica igual")
    ok(src.index("_taller_recalificar_para(") < src.index("_taller_plan_para("),
       f"{nombre}: recalifica ANTES del plan (la clave del plan lleva el criterio recalificado)")
    ok({k.arg for k in rcl[0].keywords} >= {"contexto", "suplencia"}
       and ast.get_source_segment(SRC, rcl[0].args[5]) == "_form_crit"
       and "_taller_armar_criterio(r, ses, _glob, **_form_crit)" in src,
       f"{nombre}: con el MISMO formulario con que armó el criterio")
_st = FN["taller_resolver_stream"]
_trab = next(n for n in ast.walk(_st) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_trabajar")
_src_t = ast.get_source_segment(SRC, _trab)
ok(llamadas(_trab, "_taller_recalificar_para")
   and all(_trab.lineno <= c.lineno <= _trab.end_lineno for c in llamadas(_st, "_taller_recalificar_para")),
   "flujo: dentro de `_trabajar` (la tarea), nunca antes de las cabeceras")
ok('{"tipo": "recalificando"}' in _src_t
   and _src_t.index("_taller_recalificar_para(") < _src_t.index('{"tipo": "ordenando"}')
   < _src_t.index("_taller_plan_para(") < _src_t.index("_ra.resolver_en_vivo("),
   "flujo: «recalificando», luego «ordenando» y el plan, y luego el estudio")
ok('"recalificando"' not in ast.get_source_segment(SRC, FN["taller_resolver"]), "el plano, sin evento")
_rp = FN["taller_recalificar"]
ok(ruta(_rp) == ("post", "/taller/recalificar"), "POST /taller/recalificar")
_args = {a.arg for a in _rp.args.args}
_gem = {a.arg for a in FN["taller_resolver_stream"].args.args}
ok({"numero", "user_email", "criterios_json", "global_json", "modo_decision", "sentido_global",
    "global_dictado", "usar_propuesta", "sentido", "problema", "razonamiento", "contexto",
    "suplencia", "formato"} <= _args and _args <= _gem,
   "con el MISMO formulario que los gemelos")
ok(len(llamadas(_rp, "_taller_armar_criterio")) == 1 and len(llamadas(_rp, "_taller_recalificar_para")) == 1
   and not any(getattr(k, "arg", "") == "cobrable" for c in llamadas(_rp, "_taller_puerta") for k in c.keywords),
   "arma el criterio con la función común, recalifica, y no cobra")
_rep = ast.get_source_segment(SRC, FN["taller_reparto"])
ok("recalificadas=_taller_recalificadas_guardadas(" in _rep and "huella_adelanto=" in _rep
   and "chat_client" not in _rep, "/taller/reparto aplica la guardada sin llamar a ningún modelo")
_ped = ast.get_source_segment(SRC, FN["taller_plan_pedir"])
ok(_ped.index("_taller_recalificado_al_pedir(") < _ped.index("_taller_plan_pedido("),
   "/taller/plan/pedir no pide plan con una recalificación pendiente")
_arm_src = ast.get_source_segment(SRC, FN["_taller_armar_criterio"])
ok("recalificadas=recalificadas" in _arm_src and "huella_adelanto=_huella_ac" in _arm_src
   and "global_dictado=bool(_dictado" in _arm_src, "el criterio común le pasa al árbol lo recalificado")

print("\n10 · DE PUNTA A PUNTA (base y modelo de mentira)")
from fastapi import HTTPException  # noqa: E402


class _Res:
    def __init__(self, data):
        self.data = data


class FakeBase:
    def __init__(self):
        self.filas = [{"email": "casa@iurexia.com", "expediente": "1/2026", "plan": None}]
        self.escrituras = 0

    def table(self, nombre):
        return _Q(self)


class _Q:
    def __init__(self, b):
        self.b, self.modo, self.datos, self.f = b, "select", None, []

    def select(self, *_):
        return self

    def update(self, datos):
        self.modo, self.datos = "update", datos
        return self

    def eq(self, col, val):
        self.f.append(("eq", col, val))
        return self

    def is_(self, col, val):
        self.f.append(("is", col, val))
        return self

    def limit(self, _):
        return self

    def _pasa(self, fila):
        for op, col, val in self.f:
            if col == "plan->>rev":
                v = str((fila.get("plan") or {}).get("rev")) if isinstance(fila.get("plan"), dict) else None
            elif col == "plan->rev":
                v = (fila.get("plan") or {}).get("rev") if isinstance(fila.get("plan"), dict) else None
            else:
                v = fila.get(col)
            if op == "eq" and v != val:
                return False
            if op == "is" and v is not None:
                return False
        return True

    def execute(self):
        filas = [f for f in self.b.filas if self._pasa(f)]
        if self.modo == "update":
            for f in filas:
                f.update(copy.deepcopy(self.datos))
                self.b.escrituras += 1
        return _Res([copy.deepcopy(f) for f in filas])


import taller_estado as _te  # noqa: E402

_nombres = ("_taller_tocados", "_taller_glob", "_taller_armar_criterio", "_taller_plan_cas",
            "_taller_plan_leer", "_taller_recalificadas_guardadas", "_taller_recalificar_correr",
            "_taller_recalificar_lanzar", "_taller_recalificar_para", "_taller_recalificado_al_resolver",
            "_taller_criterios_pantalla", "_taller_recalificado_al_pedir")


def entorno(modelo, base):
    ns = {"json": json, "os": os, "time": time, "asyncio": asyncio, "err": lambda e: str(e),
          "HTTPException": HTTPException, "_te": _te, "supabase_admin": base, "chat_client": modelo,
          "_TALLER_EN_MARCHA": set()}
    exec(compile(ast.Module(body=[FN[n] for n in _nombres], type_ignores=[]), "main.py", "exec"), ns)
    return ns


_ses = {"propuestas": [types.SimpleNamespace(**p) for p in props()], "material": None}
FORM = dict(sentido="", problema="", razonamiento="", usar_propuesta=False, modo_decision="",
            sentido_global="", global_dictado="",
            criterios_json=json.dumps([
                {"problema": P1, "sentido": "infundado", "razonamiento": "", "jerarquia": "principal",
                 "tocado": True},
                {"problema": P2, "sentido": "fundado", "razonamiento": "las costas siguen al principal",
                 "jerarquia": "accesorio", "tocado": False},
                {"problema": P3, "sentido": "infundado", "razonamiento": "los recibos no acreditan el pago",
                 "jerarquia": "accesorio", "tocado": False}]))
_b = FakeBase()
_m = Modelo(BUENO)
ns = entorno(_m, _b)
_r = resultado()
arm = ns["_taller_armar_criterio"](_r, _ses, {}, **FORM)
ok([c.sentido for c in arm["crit"]] == ["infundado", "", "infundado"] and rc.pendientes(arm["detalle"]) == [P2],
   "el criterio común tumba el accesorio (y lo deja en el criterio, sin sentido)")
_ev = []
out = asyncio.run(ns["_taller_recalificar_para"]("casa@iurexia.com", "1/2026", _r, _ses, {}, FORM, arm,
                                                 al_esperar=lambda: _ev.append("recalificando")))
ok(out["estado"] == "listo" and len(_m.kw) == 1 and _ev == ["recalificando"]
   and [c.sentido for c in out["arm"]["crit"]] == ["infundado", "infundado", "infundado"]
   and out["arm"]["detalle"][P2]["de"] == "recalificada",
   "se calcula dentro de la petición, avisa «recalificando» y vuelve a armar el criterio con ella")
_fila = _b.filas[0]["plan"]
ok(_fila["recalificaciones"][out["clave"]]["estado"] == "listo" and _fila["recalificaciones_corridas"] == 1
   and _fila["huella"] == _te.huella_contraste(_r), "queda en la columna `plan`, con su corrida y su adelanto")
_ev2 = []
out2 = asyncio.run(ns["_taller_recalificar_para"]("casa@iurexia.com", "1/2026", resultado(), _ses, {}, FORM,
                                                  ns["_taller_armar_criterio"](resultado(), _ses, {}, **FORM),
                                                  al_esperar=lambda: _ev2.append(1)))
ok(out2["estado"] == "listo" and len(_m.kw) == 1 and not _ev2 and out2["clave"] == out["clave"],
   "la misma premisa otra vez (otro worker): se reutiliza sin llamar al modelo ni avisar")
_pant = ns["_taller_criterios_pantalla"](out2["arm"])
ok(_pant[1]["de"] == "recalificada" and _pant[1]["recalificada"] and not _pant[1]["recalificar"]
   and _pant[0]["tocado"] and not _pant[1]["tocado"], "lo que vuelve a la pantalla, como /taller/reparto")
# La pantalla devuelve lo recalificado (tocado: false): el árbol lo reconoce y la clave no cambia.
FORM_v = dict(FORM, criterios_json=json.dumps([
    {"problema": P1, "sentido": "infundado", "razonamiento": "", "jerarquia": "principal", "tocado": True},
    {"problema": P2, "sentido": "infundado", "razonamiento": "la condena en costas atendió al resultado del juicio",
     "jerarquia": "accesorio", "tocado": False},
    {"problema": P3, "sentido": "infundado", "razonamiento": "x", "jerarquia": "accesorio", "tocado": False}]))
_a3 = ns["_taller_armar_criterio"](resultado(), _ses, {}, **FORM_v)
ok(rc.clave_de(_a3["detalle"]) == out["clave"], "la pantalla devuelve lo recalificado y la clave es la misma")
# Una entrada vacía «tocado: false» pasa al árbol; si ya no hay que recalificar, recibe lo del motor.
FORM_vacio = dict(FORM, criterios_json=json.dumps([
    {"problema": P1, "sentido": "fundado", "razonamiento": "", "jerarquia": "principal", "tocado": True},
    {"problema": P2, "sentido": "", "razonamiento": "", "jerarquia": "accesorio", "tocado": False},
    {"problema": P3, "sentido": "infundado", "razonamiento": "x", "jerarquia": "accesorio", "tocado": False}]))
_a4 = ns["_taller_armar_criterio"](resultado(), _ses, {}, **FORM_vacio)
ok(len(_a4["crit"]) == 3 and all(c.sentido for c in _a4["crit"]) and not rc.pendientes(_a4["detalle"])
   and not any("NO SE ESTUDIARON" in a for a in _a4["avisos_r"]),
   "vuelta a la vía del motor: el tumbado que llega vacío recibe lo del motor y se estudia")
FORM_suyo = dict(FORM, criterios_json=json.dumps([
    {"problema": P1, "sentido": "infundado", "razonamiento": "", "jerarquia": "principal", "tocado": True},
    {"problema": P3, "sentido": "", "razonamiento": "", "jerarquia": "accesorio", "tocado": True}]))
_a5 = ns["_taller_armar_criterio"](resultado(), _ses, {}, **FORM_suyo)
ok(len(_a5["crit"]) == 1 and any("NO SE ESTUDIARON 1" in a for a in _a5["avisos_r"]),
   "lo que él dejó sin sentido sigue fuera del estudio, y se dice (como siempre)")
# El gemelo aplica: sustituye el criterio EN SITIO y cambia los avisos del árbol.
_r6 = resultado()
arm6 = ns["_taller_armar_criterio"](_r6, _ses, {}, **FORM)
crit6 = arm6["crit"]
_r6.fases.avisos = list(arm6["avisos_fases"])
ns["_taller_recalificado_al_resolver"](_r6, arm6, out2, crit6)
ok(crit6 is arm6["crit"] and crit6[1].sentido == "infundado"
   and any("RECALIFICADOS CON TU PREMISA" in a for a in _r6.fases.avisos)
   and not any("SE RECALIFICAN CON TU PREMISA" in a for a in _r6.fases.avisos),
   "el gemelo sustituye el criterio en sitio y los avisos del árbol viejo por los nuevos")
# Si no llega: SIN CALIFICAR, y NO SE GENERA (decisión del integrador,
# 26-sep-2026): el gemelo recibe el mensaje que nombra los planteamientos y dice
# qué hacer, y no toca ni el criterio ni los avisos del adelanto.
_b2 = FakeBase()
ns2 = entorno(Modelo("nada", "nada"), _b2)
_r7 = resultado()
arm7 = ns2["_taller_armar_criterio"](_r7, _ses, {}, **FORM)
out7 = asyncio.run(ns2["_taller_recalificar_para"]("casa@iurexia.com", "1/2026", _r7, _ses, {}, FORM, arm7))
_msg7 = ns2["_taller_recalificado_al_resolver"](_r7, arm7, out7, arm7["crit"])
ok(out7["estado"] == "fallo" and arm7["crit"][1].sentido == ""
   and "SIN CALIFICAR TRAS TU CAMBIO DE SENTIDO" in _msg7 and P2[:60] in _msg7
   and "Califícalos tú en la pantalla" in _msg7 and not _r7.avisos,
   "si falla dos veces: queda SIN CALIFICAR, no se genera y el mensaje lo dice")
ok(out7["motivo"] == "fallo" and not out7["reintentable"] and "vuelve a generar" not in _msg7,
   "…y como esa premisa no se reintenta, no le dice que vuelva a generar")
out7b = asyncio.run(ns2["_taller_recalificar_para"]("casa@iurexia.com", "1/2026", resultado(), _ses, {}, FORM,
                                                    ns2["_taller_armar_criterio"](resultado(), _ses, {}, **FORM)))
ok(out7b["estado"] == "fallo" and any("no se vuelve a intentar" in a for a in out7b["avisos"]),
   "…y esa premisa no se vuelve a intentar")
# /taller/plan/pedir: con la guardada, sigue; con una en curso, espera.
_a8, falta = ns["_taller_recalificado_al_pedir"]("casa@iurexia.com", "1/2026", resultado(), _ses, {}, FORM,
                                                 ns["_taller_armar_criterio"](resultado(), _ses, {}, **FORM))
ok(not falta and _a8["crit"][1].sentido == "infundado", "el plan se pide con el criterio ya recalificado")
_b3 = FakeBase()
ns3 = entorno(Modelo(BUENO), _b3)
_a9 = ns3["_taller_armar_criterio"](resultado(), _ses, {}, **FORM)
ns3["_taller_plan_cas"]("casa@iurexia.com", "1/2026", lambda d: rc.fila_pedir(
    d, rc.clave_de(_a9["detalle"]), _te.huella_contraste(resultado()), time.time()))
_, falta = ns3["_taller_recalificado_al_pedir"]("casa@iurexia.com", "1/2026", resultado(), _ses, {}, FORM, _a9)
ok(falta, "con la recalificación en curso, el plan espera")
# La en curso se espera, no se duplica.
_lento = Modelo(BUENO, espera=0.3)
ns4 = entorno(_lento, FakeBase())


async def _dos():
    a = ns4["_taller_armar_criterio"](resultado(), _ses, {}, **FORM)
    t1 = asyncio.ensure_future(ns4["_taller_recalificar_para"]("casa@iurexia.com", "1/2026", resultado(),
                                                               _ses, {}, FORM, a))
    await asyncio.sleep(0.05)
    t2 = await ns4["_taller_recalificar_para"]("casa@iurexia.com", "1/2026", resultado(), _ses, {}, FORM,
                                               ns4["_taller_armar_criterio"](resultado(), _ses, {}, **FORM),
                                               tope_s=5.0)
    return await t1, t2
o1, o2 = asyncio.run(_dos())
ok(o1["estado"] == "listo" and o2["estado"] == "listo" and len(_lento.kw) == 1,
   "dos peticiones con la misma premisa: una llamada; la segunda espera la primera")
# Sin base: se calcula igual para la petición.
ns5 = entorno(Modelo(BUENO), None)
o = asyncio.run(ns5["_taller_recalificar_para"]("casa@iurexia.com", "1/2026", resultado(), _ses, {}, FORM,
                                                ns5["_taller_armar_criterio"](resultado(), _ses, {}, **FORM)))
ok(o["estado"] == "listo", "sin la base, la recalificación sirve igual a esta petición")
# Los dos gemelos deciden igual: la misma función, el mismo criterio.
ok(pe.clave(o["arm"]["crit"], "h", "", {}) == pe.clave(out2["arm"]["crit"], "h", "", {}),
   "el mismo formulario da el mismo criterio recalificado (la misma clave del plan)")

print("\n11 · REVISIÓN ADVERSARIAL (26-sep-2026)")
# (a) LA VUELTA A LA VÍA DEL MOTOR, con un tumbado que NO depende: sin la
# vuelta, se quedaba vacío y salía del estudio («NO SE ESTUDIARON»).
pr_v = props("infundado", "infundado", "infundado")
c = crit("fundado", pr_v)
_, det = ad.aplicar(fase3(P1, P2, P3), c, [], pr_v, tipo_asunto="amparo_directo")
ok(por(c)[P3]["sentido"] == "" and det[P3].get("recalificar"), "(el que no depende se tumba con el principal fundado)")
vuelta = [dict(x) for x in c]
vuelta[0]["sentido"] = "infundado"
_, det = ad.aplicar(fase3(P1, P2, P3), vuelta, [], pr_v, tipo_asunto="amparo_directo")
ok(por(vuelta)[P3]["sentido"] == "infundado" and por(vuelta)[P3]["razonamiento"] == "los recibos no acreditan el pago"
   and not rc.pendientes(det),
   "vuelta a la vía del motor: el tumbado que no depende recibe lo que el motor propuso, con su razón")
_ses_v = {"propuestas": [types.SimpleNamespace(**p) for p in pr_v], "material": None}
_f_v = dict(FORM, criterios_json=json.dumps([
    {"problema": P1, "sentido": "infundado", "razonamiento": "", "jerarquia": "principal", "tocado": True},
    {"problema": P2, "sentido": "", "razonamiento": "", "jerarquia": "accesorio", "tocado": False},
    {"problema": P3, "sentido": "", "razonamiento": "", "jerarquia": "accesorio", "tocado": False}]))
_a_v = entorno(Modelo(BUENO), FakeBase())["_taller_armar_criterio"](resultado(), _ses_v, {}, **_f_v)
ok([c_.sentido for c_ in _a_v["crit"]] == ["infundado", "infundado", "infundado"]
   and not any("NO SE ESTUDIARON" in a for a in _a_v["avisos_r"]),
   "…y por el criterio común: los dos vuelven a lo del motor y se estudian")
# (b) En la vía que prospera no hay caída, aunque lo guardado diga otra cosa.
c = crit("fundado", pr_v)
_, det = ad.aplicar(fase3(P1, P2, P3), c, [], pr_v, tipo_asunto="amparo_directo", huella_adelanto="h1")
_kp = rc.clave_de(det)
c = crit("fundado", pr_v)
_, det = ad.aplicar(fase3(P1, P2, P3), c, [], pr_v, tipo_asunto="amparo_directo", huella_adelanto="h1",
                    recalificadas={"clave": _kp, "resultados": {P3: {
                        "sentido": "infundado", "razon": "los recibos no acreditan el pago parcial alegado",
                        "presupone": {"cita": "la Sala omitió estudiar la excepción", "por_que": "x"},
                        "verificado": True}}})
ok(por(c)[P3]["sentido"] == "infundado" and not por(c)[P3]["razonamiento"].startswith(ad.CAE_CON_PRINCIPAL),
   "con el principal prosperando, lo recalificado nunca se escribe como caída")
# (c) Su razón se queda también en la caída.
c = crit("infundado")
c[1]["sentido"], c[1]["razonamiento"] = "", "la cedente litigó de buena fe"
_, det = ad.aplicar(fase3(P1, P2, P3), c, [], props(), tipo_asunto="amparo_directo",
                    recalificadas={"clave": _k_suya, "resultados": {P2: {
                        "sentido": "inoperante", "razon": "la del modelo",
                        "presupone": {"cita": "la cesionaria es la única obligada", "por_que": "p"},
                        "verificado": True}}})
ok(por(c)[P2]["sentido"] == "inoperante" and por(c)[P2]["razonamiento"] == "la cedente litigó de buena fe",
   "la caída recalificada no le borra la razón que él tecleó")
# (d) SIN EL MÓDULO QUE REGENERA no se tumba: se conserva y se avisa.
_rm = ad._recalificar_mod
ad._recalificar_mod = lambda: None
try:
    c = crit("infundado")
    av, det = ad.aplicar(fase3(P1, P2, P3), c, [], props(), tipo_asunto="amparo_directo")
    ok(por(c)[P2]["sentido"] == "fundado" and not rc.pendientes(det) and any("otra vía" in a for a in av),
       "sin `recalificar` el árbol conserva y avisa, como antes")
finally:
    ad._recalificar_mod = _rm
# (e) LOS PLANTEAMIENTOS DE MÁS DE 400 CARACTERES. El criterio armado los
# recorta a 400 y /taller/reparto no: el árbol tumbaba en la pantalla y en los
# gemelos los estudiaba «por su cuenta» con la calificación de la otra vía.
L2 = P2 + " " + ("Se insiste en que la condena en costas no atendió a la conducta procesal. " * 6)
L3 = P3 + " " + ("Se insiste en que la excepción de pago parcial quedó sin estudio alguno. " * 6)
assert len(L2) > 400 and len(L3) > 400
F3L = [dict(fase3(P1)[0]), dict(fase3(P2)[0], pregunta=L2), dict(fase3(P3)[0], pregunta=L3, depende_de=1)]
PRL = [dict(p, problema={P1: P1, P2: L2, P3: L3}[p["problema"]]) for p in props()]


def critL(corte):
    return [{"problema": x["problema"][:corte], "sentido": x["sentido"], "razonamiento": x["razon"],
             "jerarquia": "principal" if i == 0 else "accesorio", "tocado": i == 0}
            for i, x in enumerate(PRL)]


cL = critL(400)
cL[0]["sentido"] = "infundado"
_, dL = ad.aplicar(copy.deepcopy(F3L), cL, [], copy.deepcopy(PRL), tipo_asunto="amparo_directo",
                   huella_adelanto="h1")
cR = critL(10_000)
cR[0]["sentido"] = "infundado"
rR = ad.reparto_para_pantalla(copy.deepcopy(F3L), cR, [], copy.deepcopy(PRL), tipo_asunto="amparo_directo",
                              huella_adelanto="h1")
ok(len(rc.pendientes(dL)) == 2 and sum(1 for x in rR["criterios"] if x["recalificar"]) == 2,
   "recortado a 400 (gemelos) o entero (pantalla): los mismos dos tumbados")
_cR2 = critL(10_000)
_cR2[0]["sentido"] = "infundado"
_, dR = ad.aplicar(copy.deepcopy(F3L), _cR2, [], copy.deepcopy(PRL), tipo_asunto="amparo_directo",
                   huella_adelanto="h1")
ok(rc.clave_de(dL) == rc.clave_de(dR) != "", "…y la misma clave por las dos puertas")
_rL = types.SimpleNamespace(fases=f123.Fases123(problemas=copy.deepcopy(F3L)))
_prL, _acL = rc.entradas(_rL, [f6.Criterio(problema=x["problema"], sentido=x["sentido"],
                                           razonamiento=x["razonamiento"], jerarquia=x["jerarquia"])
                               for x in cL], dL)
ok(sorted(a["numero"] for a in _acL) == [2, 3] and all(a["fase3"].get("combate") for a in _acL),
   "el modelo los recibe con su número y lo que combaten (no dos «planteamiento 0»)")
# Sin su problema de la fase 3, cada uno con un número propio: la respuesta se
# casa por número y dos «0» se llevaban la misma calificación.
_pr0, _ac0 = rc.entradas(types.SimpleNamespace(fases=f123.Fases123(problemas=[])),
                         [f6.Criterio(problema=x["problema"], sentido=x["sentido"],
                                      razonamiento=x["razonamiento"], jerarquia=x["jerarquia"]) for x in cL], dL)
_n0 = [a["numero"] for a in _ac0]
_res0, _f0, _ = rc.validar({"planteamientos": [
    {"numero": _n0[0], "sentido": "infundado", "razon": "la condena atendió al resultado del juicio"},
    {"numero": _n0[1], "sentido": "inoperante", "razon": "no combate la consideración que sostiene lo resuelto"}]},
    _ac0, _pr0)
ok(len(set(_n0)) == 2 and 0 not in _n0 and not _f0
   and {v["sentido"] for v in _res0.values()} == {"infundado", "inoperante"},
   "sin la fase 3, números distintos y cada respuesta a su planteamiento")
_resL = {"clave": rc.clave_de(dL), "resultados": {
    L2[:400]: {"sentido": "infundado", "razon": "la condena atendió al resultado del juicio", "presupone": None,
               "verificado": False},
    L3[:400]: {"sentido": "infundado", "razon": "los recibos no acreditan el pago parcial", "presupone": None,
               "verificado": False}}}
cR3 = critL(10_000)
cR3[0]["sentido"] = "infundado"
rR3 = ad.reparto_para_pantalla(copy.deepcopy(F3L), cR3, [], copy.deepcopy(PRL), tipo_asunto="amparo_directo",
                               huella_adelanto="h1", recalificadas=_resL)
ok(all(x["recalificada"] and x["sentido"] == "infundado" for x in rR3["criterios"][1:]),
   "lo guardado por los gemelos (recortado) se aplica en la pantalla (entero)")
# (f) ÉL PISA UNO DE LOS TUMBADOS: los demás no se rehacen ni cambian.
F3P = fase3(P1, P2, P4)
PRP = props(con=(P1, P2, P4))
_sesP = {"propuestas": [types.SimpleNamespace(**p) for p in PRP], "material": None}


def resP():
    r = resultado()
    r.fases = f123.Fases123(resumen_acto="La Sala condenó.", problemas=copy.deepcopy(F3P), fuentes=["", ""])
    return r


BUENO2 = {"planteamientos": [
    {"numero": 2, "sentido": "infundado", "razon": "la condena en costas atendió al resultado del juicio",
     "presupone": None},
    {"numero": 3, "sentido": "fundado", "razon": "la pericial contable se ofreció en tiempo y era idónea",
     "presupone": None}]}


def formP(toca2):
    return dict(FORM, criterios_json=json.dumps([
        {"problema": P1, "sentido": "infundado", "razonamiento": "", "jerarquia": "principal", "tocado": True},
        {"problema": P2, "sentido": "fundado", "razonamiento": "suya" if toca2 else "las costas siguen",
         "jerarquia": "accesorio", "tocado": toca2},
        {"problema": P4, "sentido": "fundado", "razonamiento": "la pericial era idónea", "jerarquia": "accesorio",
         "tocado": False}]))


_bP, _mP = FakeBase(), Modelo(BUENO2)
nsP = entorno(_mP, _bP)
aP = nsP["_taller_armar_criterio"](resP(), _sesP, {}, **formP(False))
oP = asyncio.run(nsP["_taller_recalificar_para"]("casa@iurexia.com", "1/2026", resP(), _sesP, {}, formP(False), aP))
ok(oP["estado"] == "listo" and len(_mP.kw) == 1 and sorted(rc.pendientes(aP["detalle"])) == sorted([P2, P4]),
   "(dos tumbados, una llamada)")
aP2 = nsP["_taller_armar_criterio"](resP(), _sesP, {}, **formP(True))
oP2 = asyncio.run(nsP["_taller_recalificar_para"]("casa@iurexia.com", "1/2026", resP(), _sesP, {}, formP(True), aP2))
_s2 = {c_.problema: c_.sentido for c_ in oP2["arm"]["crit"]}
ok(oP2["estado"] == "listo" and len(_mP.kw) == 1 and _s2[P4] == "fundado" and _s2[P2] == "fundado"
   and oP2["clave"] != oP["clave"],
   "él pisa uno: el otro se queda como se recalificó, sin otra llamada (la misma premisa)")
_docP = _bP.filas[0]["plan"]
ok(_docP["recalificaciones_corridas"] == 1, "…y sin gastar otra corrida del adelanto")
_hP = _te.huella_contraste(resP())
_cP = [dict(x) for x in json.loads(formP(True)["criterios_json"])]
rPant = ad.reparto_para_pantalla(copy.deepcopy(F3P), _cP, [], copy.deepcopy(PRP), tipo_asunto="amparo_directo",
                                 recalificadas=rc.guardadas(_docP, _hP), huella_adelanto=_hP)
ok([x["de"] for x in rPant["criterios"]] == ["principal", "tuya", "recalificada"],
   "/taller/reparto dice lo mismo: el suyo es suyo y el otro, recalificado")
_, faltaP = nsP["_taller_recalificado_al_pedir"]("casa@iurexia.com", "1/2026", resP(), _sesP, {}, formP(True), aP2)
ok(not faltaP, "…y el plan se pide sin esperar a nadie")
# (g) LAS CORRIDAS AGOTADAS: lo guardado se aplica y el plan no espera.
_bT = FakeBase()
nsT = entorno(Modelo(BUENO), _bT)
aT = nsT["_taller_armar_criterio"](resultado(), _ses, {}, **FORM)
kT, hT = rc.clave_de(aT["detalle"]), _te.huella_contraste(resultado())
dT, _ = rc.fila_pedir(None, kT, hT, 1.0)
dT, _ = rc.fila_resultado(dT, kT, hT, {"estado": "error", "resultados": {P2: {
    "sentido": "infundado", "razon": "la condena atendió al resultado del juicio", "presupone": None,
    "verificado": False}}, "avisos": []}, 1.0, 2.0)
dT["recalificaciones_corridas"] = rc.TOPE_CORRIDAS
_bT.filas[0]["plan"] = dT
_mT = nsT["chat_client"]
oT = asyncio.run(nsT["_taller_recalificar_para"]("casa@iurexia.com", "1/2026", resultado(), _ses, {}, FORM, aT))
ok(len(_mT.kw) == 0 and oT["arm"]["crit"][1].sentido == "infundado",
   "tope: lo guardado de esta premisa (un «error» con resultado) se aplica igual que en la pantalla")
_, faltaT = nsT["_taller_recalificado_al_pedir"]("casa@iurexia.com", "1/2026", resultado(), _ses, {}, FORM, aT)
ok(not faltaT, "tope: el plan no espera una recalificación que nadie va a calcular")
_bT.filas[0]["plan"] = dict(dT, recalificaciones={})
_, faltaT2 = nsT["_taller_recalificado_al_pedir"]("casa@iurexia.com", "1/2026", resultado(), _ses, {}, FORM, aT)
# Sin nada guardado y sin corridas, el tumbado queda sin calificar: el proyecto
# no se genera sin él (26-sep-2026), así que tampoco se pide un plan sobre él.
ok(faltaT2 == "sin_calificar", "…tampoco sin nada guardado: no espera, y no pide un plan que nadie usaría")
# (h) EL SIN CALIFICAR, EN LA v2, CONSERVA SU CONCEPTO Y SU RAZÓN: sin CUBRE
# el estudio no sabía qué concepto contestaba, y la razón que él tecleó se
# perdía.
_mat_c = f6.Material(tipo_asunto="amparo_directo", materia="civil", variante="v2",
                     problemas=fase3(P1, P2, P3))
_C_c = [C_sin[0], f6.Criterio(problema=P2, sentido="", razonamiento="la cedente litigó de buena fe",
                              jerarquia="accesorio")]
_p_c = f6.prompt_estudio("acto", "conceptos", _C_c, _mat_c)
_tramo = _p_c.split("SENTIDO: SIN CALIFICAR")[1].split("\n\n")[0]
ok("CUBRE:" in _tramo and "RAZÓN DEL SECRETARIO: la cedente litigó de buena fe" in _tramo,
   "v2: el sin calificar lleva su CUBRE y la razón que él escribió")
_p1_c = f6.prompt_estudio("acto", "conceptos", _C_c, f6.Material(tipo_asunto="amparo_directo", materia="civil",
                                                                  variante="v1", problemas=fase3(P1, P2, P3)))
ok("SENTIDO: SIN CALIFICAR" in _p1_c and "RAZÓN DEL SECRETARIO: la cedente litigó de buena fe" in _p1_c,
   "la v1: el sin calificar lo dice y conserva la razón que él escribió")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
