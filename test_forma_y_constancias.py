"""Legitimación vs. personería, efectos de la reposición y constancias
indispensables — ADC 93/2026 v5, 22-sep-2026 (análisis crítico de David).

    .venv/bin/python test_forma_y_constancias.py
"""
import inspect

import constancias as cn
import documento_generado as dg
import fase6_estudio as f6
import promovente as pv
import tipos_asunto as ta

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


print("\n1 · LA PARTE Y QUIEN LA REPRESENTA")
FICHA = "Alondra Zúñiga Gutiérrez, representante legal de GDE Trading Company, sociedad anónima de capital variable"
d = pv.separar(FICHA)
ok(d["parte"].startswith("GDE Trading Company") and d["representante"] == "Alondra Zúñiga Gutiérrez",
   "la ficha del 93/2026: la parte es la moral, la física representa")
ok(d["figura"] == "representante legal" and d["moral"], "con su figura y sabiendo que es moral")
ok(pv.separar("GDE TRADING COMPANY, S.A. DE C.V.")["parte"] == "GDE TRADING COMPANY, S.A. DE C.V.",
   "sin fórmula, el nombre queda tal cual (y no se come el punto de C.V.)")
ok(pv.separar("Pedro Sánchez Ramírez")["representante"] == "", "una persona física sola: sin representante")
d2 = pv.separar("Banco Azteca, S.A., por conducto de su apoderado legal Juan Pérez López")
ok(d2["parte"] == "Banco Azteca, S.A." and d2["representante"] == "Juan Pérez López", "«Y, por conducto de su apoderado X»")
d3 = pv.separar("María López García, en representación de Servicios del Bajío, S. de R.L. de C.V.")
ok(d3["parte"].startswith("Servicios del Bajío"), "«X, en representación de Y»")
leg = ta.legitimacion_de("amparo_directo", d["parte"], d["representante"], figura=d["figura"], moral=d["moral"])
ok("por conducto de su representante legal Alondra Zúñiga Gutiérrez, en términos de los artículos 6º y 11" in leg,
   "la personería se funda en los artículos 6º y 11")
ok("la persona moral quejosa está legitimada para ello conforme al artículo 5º, fracción I" in leg,
   "y la legitimación ad causam se predica de la moral, con el 5º, fracción I")
ok("Alondra Zúñiga Gutiérrez, quien está legitimad" not in leg, "la persona física ya no «resiente el perjuicio»")
ok(ta.legitimacion_de("amparo_directo", "Pedro Sánchez Ramírez").count("legitimado") == 1, "sin representante, el molde de siempre")
ok(pv.por_conducto(d["parte"], d["representante"], d["figura"]).startswith("GDE Trading Company")
   and "por conducto de su representante legal Alondra" in pv.por_conducto(d["parte"], d["representante"], d["figura"]),
   "el resolutivo ampara a la moral, por conducto de su representante")

print("\n2 · LOS EFECTOS DE LA REPOSICIÓN")
EST = ["Último párrafo del estudio.", "EFECTOS DE LA CONCESIÓN",
       "1. Deje insubsistente la sentencia de veintiocho de noviembre.",
       "2. Deje sin efectos la interlocutoria y el acuerdo de preclusión.",
       "3. Admita la ampliación de demanda.", "4. Corra traslado a la autoridad demandada.",
       "5. Cerrada la instrucción, dicte nueva sentencia con plenitud de jurisdicción."]
c, e = dg.partir_efectos(EST)
ok(len(c) == 1 and len(e) == 5 and e[0].startswith("1."), "el rótulo fijo se reconoce y no entra como efecto")
c2, e2 = dg.partir_efectos(["…", "La concesión del amparo exige que la Sala responsable: a) deje insubsistente…", "Hecho lo anterior, deberá…"])
ok(len(e2) == 2, "y la apertura que escribió la v5 («la concesión del amparo exige que») también")
ok(dg.partir_efectos(["La preclusión produce efectos sobre la litis.", "Los efectos de la notificación son otros."])[1] == [],
   "un párrafo que habla de «efectos» no se corta")
crit = [f6.Criterio(problema="p", sentido="fundado", jerarquia="principal")]
ok(f6._efectos_de_reposicion("…deje insubsistente la sentencia y dicte otra en la que atienda los lineamientos.", crit, True).startswith("EFECTOS INCOMPLETOS"),
   "«dicte otra» sobre una violación procesal se acusa")
ok(f6._efectos_de_reposicion("\n".join(EST), crit, True) == "", "los cinco pasos, limpio")
ok(f6._efectos_de_reposicion("x", crit, False) == "" and f6._efectos_de_reposicion("x", [f6.Criterio(problema="p", sentido="infundado")], True) == "",
   "sin violación procesal o sin concesión, nada que decir")
ok("EFECTOS DE LA CONCESIÓN" in inspect.getsource(f6._bloque_criterio) and "REPOSICIÓN paso a paso" in inspect.getsource(f6._bloque_criterio),
   "el prompt pide el rótulo y los pasos")
ok(any("PASO A PASO" in t for t in ta.TECNICA_RESOLUCION["directo_violacion_procesal"]["tecnica"]),
   "y la técnica de la violación procesal los enumera")

print("\n3 · LAS CONSTANCIAS")
PED = [{"que": "Interlocutoria de 18 de septiembre de 2025 que resolvió la reclamación", "para_que": "saber qué plazo aplicó", "indispensable": True, "problema": 1},
       {"que": "acuerdo de 13 de agosto de 2025 (preclusión)", "para_que": "fundamento del cómputo", "indispensable": True, "problema": 1},
       {"que": "la interlocutoria del recurso de reclamación de dieciocho de septiembre", "indispensable": False},
       {"que": "escrito de alegatos de 28 de agosto de 2025", "para_que": "qué argumentos contenía", "indispensable": False, "problema": 2}]
n = cn.normalizar(PED)
ok(len(n) == 3 and n[0]["indispensable"] and n[-1]["que"].startswith("escrito de alegatos"), f"sin repetidas, indispensables primero: {[x['que'][:25] for x in n]}")
ctx = cn.rotular("Interlocutoria de 18 de septiembre de 2025", "La Sala declaró infundado el recurso…")
ok(ctx.startswith("[CONSTANCIA · Interlocutoria de 18 de septiembre de 2025]"), "el aporte viaja rotulado")
ok(cn.aportadas_en(ctx) == ["Interlocutoria de 18 de septiembre de 2025"], "y se lee de vuelta")
f = cn.faltantes(PED, ctx)
ok([x["que"][:20] for x in f] == ["acuerdo de 13 de ago", "escrito de alegatos "], f"faltan las otras dos: {[x['que'][:20] for x in f]}")
b = cn.bloque_para_estudio(PED, ctx)
ok("SÍ están arriba" in b and "NO SE APORTARON" in b and "[INDISPENSABLE]" in b and "NO SUPONGAS SU CONTENIDO" in b, "el bloque del estudio dice cuáles llegaron y cuáles no")
ok("FALTAN CONSTANCIAS INDISPENSABLES" in cn.aviso_faltantes(PED, ctx) and "acuerdo de 13 de agosto" in cn.aviso_faltantes(PED, ctx),
   "el aviso nombra la indispensable que falta")
ok(cn.aviso_faltantes(PED, ctx + "\n" + cn.rotular("acuerdo de 13 de agosto de 2025", "texto")) == "",
   "y se calla cuando las indispensables ya están")
ok(cn.de_fase3([{"apoyo": {"motivo": "constancia", "explicacion": "el escrito de ampliación de 14 de agosto"}}])[0]["que"].startswith("el escrito"),
   "lo que la fase 3 marcaba como apoyo en constancia entra a la lista")

print("\n4 · LAS PUERTAS ESTÁN CONECTADAS")
import fase5_propuesta as f5, redactor_adelanto as ra
src = open("main.py", encoding="utf-8").read()
ok('"constancias": [' in inspect.getsource(f5.prompt_propuesta) and "13. LAS CONSTANCIAS" in inspect.getsource(f5.prompt_propuesta), "la fase 5 las declara")
ok("glob.constancias = _cn.normalizar" in inspect.getsource(f5.proponer), "y se leen al parsear")
ok('"constancias": list(getattr(glob, "constancias"' in src, "/taller/proponer las devuelve")
ok('etiqueta: str = Form("")' in src and "_cn_r.rotular(etiqueta, junto)" in src, "/taller/contexto rotula el aporte")
ok("_bloque_constancias(propuesta_global, contexto, criterios)" in inspect.getsource(f6.prompt_estudio), "el estudio recibe el bloque, y con él los criterios para saber si se resolvió al revés")
ok(inspect.getsource(ra).count("_cn_a.aviso_faltantes(") == 2, "los dos redactores avisan de las que faltan")
ok(inspect.getsource(ra).count("f6._efectos_de_reposicion(estudio, criterios, _vp)") == 2, "y comprueban los efectos")
ok('"representante": _pv.separar(e.quejoso)["representante"]' in inspect.getsource(ra), "el compositor recibe parte y representante")
ok("_pvr.por_conducto(" in inspect.getsource(dg), "y el resolutivo ampara por conducto")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
