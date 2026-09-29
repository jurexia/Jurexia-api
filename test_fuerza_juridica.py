# -*- coding: utf-8 -*-
"""La fuerza jurídica en un solo sitio (rediseño del taller, punto 3, 29-sep-2026).

    .venv/bin/python test_fuerza_juridica.py

Vigila que quien decide y quien redacta reciban la MISMA fuerza, y que las dos
preguntas no se fundan: ¿vincula al tribunal? / ¿su regla gobierna el caso?
"""
import sys
sys.path.insert(0, ".")
import fuerza_juridica as fj
import tarjeta_decision as td
import contexto_taller as ct
import os
os.environ.pop("FUERZA_UNIFICADA", None)
ct.poner(True, {})          # una cuenta de casa: la bandera, encendida por omisión

FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


TER = "Tercer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito"
TCC_J = {"registro": "2020001", "instancia": "Tribunales Colegiados de Circuito", "tipo": "Jurisprudencia",
         "vincula": True, "rubro": "R", "localizacion": "[J]; 10a. Época; T.C.C.; Gaceta S.J.F.",
         "texto": "… [TESIS: I.4o.A. J/12 (10a.)]"}
SCJN_J = {"registro": "2020002", "instancia": "Segunda Sala", "tipo": "Jurisprudencia", "vincula": True}
PR_J = {"registro": "2020003", "instancia": "Plenos Regionales", "tipo": "Jurisprudencia", "vincula": True}
PROPIA_J = {"registro": "2020004", "instancia": "Tribunales Colegiados de Circuito", "tipo": "Jurisprudencia",
            "vincula": True, "texto": "… [TESIS: XXII.3o.A.C. J/5 K (11a.)]"}

print("\n1 · UNA SOLA REGLA")
ok(td.fuerza_para_colegiado is fj.fuerza_para_colegiado, "la tarjeta y la deliberación leen la de fuerza_juridica")

print("\n2 · ¿VINCULA A ESTE TRIBUNAL?")
t = [dict(TCC_J), dict(SCJN_J), dict(PR_J), dict(PROPIA_J)]
fj.anotar(t, TER)
ok(t[0]["obligatoria"] is False and "217, párr. tercero" in t[0]["fuerza_texto"],
   "la jurisprudencia de otro colegiado ORIENTA (art. 217, párr. tercero), aunque el Semanario diga vincula")
ok(t[0]["vincula_origen"] is True, "y el dato crudo del Semanario se conserva aparte")
ok(t[1]["obligatoria"] is True and t[1]["vincula_al_tribunal"] is True, "la de una Sala obliga")
ok(t[2]["vincula_al_tribunal"] is None and t[2]["obligatoria"] is False and t[2]["por_confirmar"],
   "la de un Pleno Regional sin región queda POR CONFIRMAR, nunca obligatoria")
ok(t[3]["vincula_al_tribunal"] is True and "228" in t[3]["fuerza_texto"],
   "la jurisprudencia PROPIA vincula al propio tribunal (art. 228)")
ok(fj.rotulo(t[0]).startswith("orientadora") and fj.rotulo(t[1]).startswith("OBLIGATORIA")
   and fj.rotulo(t[2]).startswith("fuerza POR CONFIRMAR"), "los rótulos de prompt dicen lo mismo")
ok([fj.orden(x) for x in t] == [2, 0, 1, 0], "el orden: lo que vincula primero, luego lo por confirmar")
ok(fj.epoca_de(TCC_J) == 10, "la época se lee de la localización")

print("\n3 · ¿GOBIERNA EL SUPUESTO? — OTRA PREGUNTA")
ok(all(x["aplicabilidad"] == {"estado": "no_evaluada"} for x in t),
   "anotar la fuerza NO dice que la regla gobierne el caso: aplicabilidad «no_evaluada»")
x = dict(SCJN_J, aplicabilidad={"estado": "distinguible"})
fj.anotar([x], TER)
ok(x["aplicabilidad"] == {"estado": "distinguible"}, "y una aplicabilidad ya medida no se pisa")

print("\n4 · MATERIAL VIEJO Y SIN TRIBUNAL")
vieja = {"registro": "1", "instancia": "Tribunales Colegiados de Circuito", "tipo": "Jurisprudencia",
         "obligatoria": True}          # así se guardaba antes del 29-sep
ok(fj.vincula(vieja) is False and "vincula_al_tribunal" not in vieja,
   "una tesis guardada con obligatoria=vincula ya no se lee obligatoria, y leerla no la modifica")
y = dict(PROPIA_J)
fj.anotar([y])
ok(y["vincula_al_tribunal"] is False, "sin tribunal, la propia no se reconoce (es de otro colegiado)")
fj.anotar([y], TER)
ok(y["vincula_al_tribunal"] is True, "al saberse el tribunal se recalcula")

print("\n5 · LOS QUE LA LEEN")
import fase6_rag as r
d = r._tesis_de(dict(TCC_J))
ok(d["obligatoria"] is False and d["vincula_origen"] is True and d["fuerza"] == "orienta",
   "_tesis_de: la de colegiado sale orientadora y guarda el dato crudo")
import documento_generado as dg
ok("como criterio orientador" in dg.anuncio_de(dict(d, rubro="RUBRO"), "Es aplicable la jurisprudencia"),
   "el documento la anuncia «como criterio orientador»")
import inspect
src = inspect.getsource(r)
ok(src.count('not t.get("vincula_origen", t.get("obligatoria"))') == 2,
   "el ORDEN de la búsqueda no cambia: sigue priorizando la jurisprudencia (vincula_origen)")

print("\n5b · LA CLAVE Y LA VIGENCIA")
d2 = r._tesis_de({"registro": "2030001", "instancia": "Tribunales Colegiados de Circuito", "tipo": "Jurisprudencia",
                  "vincula": True, "clave_tesis": "XXII.3o.A.C. J/5 K (11a.)", "epoca": "Undécima Época",
                  "fecha_publicacion": "2024-05-10"})
ok(d2["clave"] == "XXII.3o.A.C. J/5 K (11a.)" and d2["epoca"] and d2["fecha_publicacion"],
   "_tesis_de ya no tira la clave, la época ni la fecha")
fj.anotar([d2], TER)
ok(d2["vincula_al_tribunal"] is True and "228" in d2["fuerza_texto"],
   "con la clave guardada, la jurisprudencia PROPIA se reconoce (art. 228)")
ok(not fj.sello_perdio_vigencia({"estado": "aclarada"}) and not fj.sello_perdio_vigencia({"estado": "texto_sustituido"})
   and fj.sello_perdio_vigencia({"estado": "abandonada"})
   and not fj.sello_perdio_vigencia({"estado": "abandonada", "parcial": True}),
   "una sola definición de vigencia: aclarada y texto sustituido NO la pierden; la parcial tampoco entera")
import deliberacion as de
ok(de.pierde_vigencia({"estado": "aclarada"}) is False and td.perdio_vigencia({"vigencia": {"estado": "modificada"}}) is True,
   "la deliberación y la tarjeta leen la misma")

print("\n5c · CON LA BANDERA APAGADA (usuarios de fuera, o la etapa «off» del banco)")
ct.poner(False, {})
viejo = r._tesis_de(dict(TCC_J))
ok(viejo["obligatoria"] is True and fj.rotulo(viejo) == "OBLIGATORIA" and fj.orden(viejo) == 0
   and fj.vincula(viejo) is True,
   "apagada, los prompts y el documento leen como antes (obligatoria = es jurisprudencia)")
ok(viejo["vincula_al_tribunal"] is False and viejo["fuerza"] == "orienta",
   "pero la fuerza calculada viaja igual: la tarjeta y la deliberación no cambian")
ct.poner(True, {"banderas": {"fuerza_unificada": False}})
ok(fj.activa() is False, "la casa puede apagarla en una etapa del banco")
ct.poner(True, {})
nuevo = r._tesis_de(dict(TCC_J))
ok(nuevo["obligatoria"] is False and fj.rotulo(nuevo).startswith("orientadora"), "encendida, la regla nueva")
fj.anotar([viejo])
ok(viejo["obligatoria"] is False, "y una tesis anotada con la bandera apagada se RECALCULA al encenderla")

print("\n6 · LAS SENTENCIAS PROPIAS")
s = fj.de_sentencia_propia({"nivel": "posible", "similitud": 66, "cota_inferior": True})
ok(s["vincula"] is False and s["aplicabilidad"]["estado"] == "posible"
   and s["fuerza_texto"] == td.FUERZA_SENTENCIA_PROPIA,
   "una sentencia de la OAJ no vincula; su nivel calibrado es la aplicabilidad medida")
ok(fj.de_sentencia_propia({"expediente": "1/2020"})["aplicabilidad"] == {"estado": "no_evaluada"},
   "el espejo viejo no la mide")

print()
if FALLOS:
    print("FALLAS:\n  " + "\n  ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
