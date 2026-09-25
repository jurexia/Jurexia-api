"""Las dos formas de la sentencia — estándar y moderna — 25-sep-2026.

    .venv/bin/python test_formato_sentencia.py
"""
import ast
import asyncio

import fase6_estudio as f6
import formato_sentencia as fs

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


C93 = [f6.Criterio(problema="¿La Sala debió admitir la ampliación de demanda y estudiar "
                            "los argumentos dirigidos contra el crédito fiscal?",
                   sentido="fundado", jerarquia="principal"),
       f6.Criterio(problema="¿La Sala debía pronunciarse sobre los argumentos formulados "
                            "por la actora en sus alegatos?",
                   sentido="innecesario", jerarquia="accesorio")]
P93 = [{"pregunta": C93[0].problema, "cubre": [1]},
       {"pregunta": "¿La Sala debía pronunciarse sobre los argumentos que la actora "
                    "formuló en sus alegatos?", "cubre": [2]}]

print("\n1 · QUÉ FORMA SE PIDIÓ")
ok(fs.normalizar("") == fs.ESTANDAR and fs.normalizar(None) == fs.ESTANDAR, "vacío = estándar")
ok(fs.normalizar("Moderna") == fs.MODERNA and fs.normalizar("versión moderna") == fs.ESTANDAR,
   "sólo «moderna…» es moderna; cualquier otra cosa, estándar")
ok(fs.palabras_moderna(C93) == fs.MODERNA_MIN, "93/2026: un solo problema vivo → el mínimo (1,600)")
ok(fs.palabras_moderna([f6.Criterio("p", "fundado")] * 9) == fs.MODERNA_MAX, "nueve vivos → el tope")

print("\n2 · QUÉ CONCEPTO CALIFICA CADA CRITERIO")
ok(fs.cubre_de(C93[0], P93) == [1], "la pregunta literal casa con su «cubre»")
ok(fs.cubre_de(C93[1], P93) == [2], "retocada con el lápiz, casa igual")
ok(fs.cubre_de(f6.Criterio("¿Prescribió la acción cambiaria directa?", "infundado"), P93) == [],
   "sin pareja segura no se atribuye nada")
ok(fs.cubre_en_texto([1], "concepto de violación") == "el primer concepto de violación", "«el primer», no «el primero»")
ok(fs.cubre_en_texto([1, 3], "agravio") == "los agravios primero y tercero", "plural pospuesto")

print("\n3 · ¿CADA PLANTEAMIENTO RECIBIÓ RESPUESTA?")
est = ("Los conceptos de violación son infundados.\n"
       "Sobre el primer concepto de violación, en el que la parte quejosa sostiene X. "
       "Se considera infundado. Lo anterior, porque…\n"
       "Sobre el segundo concepto de violación, en el que sostiene Y. Es infundado.")
ok(fs.sin_contestar(est, 2) == [], "estándar con los dos nombrados")
ok(fs.sin_contestar(est, 3) == [3], "el tercero no aparece → se dice")
mod = ("1. ¿La Sala debió admitir la ampliación?\n\nSí. El primer concepto de violación "
       "es fundado, porque…\n\n2. ¿Debía atender los alegatos?\n\nDado el sentido del "
       "estudio del primer concepto de violación, queda sin materia el análisis de este planteamiento.")
ok(fs.sin_contestar(mod, 2) == [2], "«este planteamiento» no dice cuál: 93/2026 v5")
ok(fs.sin_contestar(mod.replace("de este planteamiento", "del segundo concepto de violación"), 2) == [],
   "nombrándolo, pasa")
ok(fs.sin_contestar("Los anteriores conceptos de violación son fundados.", 2) == [],
   "sin ningún ordinal no se acusa (ADA 704/2022)")
ok(fs.sin_contestar("Respecto al segundo concepto de violación, se desestima. En este orden, "
                    "resulta inoperante el concepto de violación del quejoso en cuanto al dictamen.", 2) == [],
   "la respuesta en singular sin ordinal cubre al que falta (ADC 640/2024)")
ok(fs.sin_contestar("Se estudian conjuntamente los conceptos de violación primero y segundo…", 4) == [],
   "estudio conjunto declarado")
ok(fs.sin_contestar("Por los fundamentos y motivos del considerando sexto. El primer concepto es fundado.", 2) == [2],
   "«motivos… sexto» no cuenta como un sexto planteamiento")
ok(fs.sin_contestar(est, 1) == [], "con uno solo no hay serie que comprobar")

print("\n4 · LA SÍNTESIS DE LA MODERNA")
ante = ["1. " + "hecho " * 60, "2. " + "trámite " * 60]
acto = ["La Sala " + "resolvió " * 200]
conc = ["En el primer concepto de violación " + "alega " * 150,
        "En el segundo concepto de violación " + "sostiene " * 150]
bueno = {"antecedentes": ["1. hecho relevante", "2. " + "x " * 19 + "x"],
         "resolvio": ["La Sala " + "resolvió " * 39 + "resolvió"],
         "planteamientos": ["En el primer concepto de violación alega " + "a " * 29 + "a",
                            "En el segundo concepto de violación sostiene " + "b " * 29 + "b"]}
f, notas = fs.validar_sintesis(bueno, ante, acto, conc, 2)
ok(f["acto"] == bueno["resolvio"] and f["conceptos"] == bueno["planteamientos"], "lo condensado que pasa, entra")
perdio = dict(bueno, planteamientos=["En el primer concepto de violación alega " + "a " * 40])
f, notas = fs.validar_sintesis(perdio, ante, acto, conc, 2)
ok(f["conceptos"] is conc and any("perdió el 2" in n for n in notas), "si pierde un concepto, va el completo")
largo = dict(bueno, resolvio=acto)
f, _ = fs.validar_sintesis(largo, ante, acto, conc, 2)
ok(f["acto"] is acto, "si no acorta, va el original")
# 93/2026 v10: la sentencia casi entera no la combate nadie; condensarla a una
# décima parte es lo pedido, no un fallo.
decima = dict(bueno, resolvio=["La Sala " + "resolvió " * 45 + "fin"])
f, _ = fs.validar_sintesis(decima, ante, ["La Sala " + "resolvió " * 1500], conc, 2)
ok(f["acto"] == decima["resolvio"], "una décima parte del original vale si dice algo (≥40 palabras)")
f, _ = fs.validar_sintesis(dict(bueno, resolvio=["La Sala resolvió."]), ante, acto, conc, 2)
ok(f["acto"] is acto, "tres palabras no: va el original")
f, notas = fs.validar_sintesis({}, ante, acto, conc, 2)
ok(f == {"antecedentes": ante, "acto": acto, "conceptos": conc}, "si no llega nada, todo completo")
ok(fs.leer_json('texto {"antecedentes": ["1. a"]} cola') == {"antecedentes": ["1. a"]}, "extrae el JSON")
ok(fs.leer_json("nada") == {}, "sin JSON, vacío")

print("\n5 · EL PROMPT DEL ESTUDIO")
m_std = f6.Material(tipo_asunto="amparo_directo", formato="estandar", problemas=P93, n_planteamientos=2)
m_mod = f6.Material(tipo_asunto="amparo_directo", formato="moderna", problemas=P93, n_planteamientos=2)
p_std = f6.prompt_estudio("ACTO", "CONCEPTOS", C93, m_std)
p_mod = f6.prompt_estudio("ACTO", "CONCEPTOS", C93, m_mod)
ok("FORMATO: ESTÁNDAR" in p_std and "FORMATO: VERSIÓN MODERNA" not in p_std, "estándar: su bloque y sólo el suyo")
ok("FORMATO: VERSIÓN MODERNA" in p_mod and "FORMATO: ESTÁNDAR" not in p_mod, "moderna: su bloque y sólo el suyo")
ok("CADA PROBLEMA ABRE CON SU PREGUNTA" not in p_std, "la estándar ya no ordena abrir con la pregunta")
ok("CADA PROBLEMA ABRE CON SU PREGUNTA" in p_mod, "la moderna sí")
ok("Sobre el primer concepto de violación, en el que la parte quejosa sostiene" in p_std,
   "la fórmula de David, con el vocabulario del tipo")
ok("CUBRE: el primer concepto de violación" in p_std and "CUBRE: el segundo concepto de violación" in p_std,
   "cada criterio dice qué concepto califica")
ok("Alrededor de 3733" in p_std and "Alrededor de 1600" in p_mod, "cada forma con su medida")
ok(p_std.rstrip().endswith("Nada más.") and "Y LA FORMA ES LA ESTÁNDAR" in p_std, "el recordatorio va al final")
m_rf = f6.Material(tipo_asunto="revision_fiscal", formato="estandar")
p_rf = f6.prompt_estudio("ACTO", "AGRAVIOS", C93, m_rf, es_recurso=True)
ok("la quejosa" not in p_rf.lower().replace("la parte quejosa", ""),
   "revisión fiscal: ningún ejemplo nombra quejosa a la autoridad")

print("\n6 · LAS REVISIONES DEL ESTUDIO")
corto = "Los conceptos son fundados. " + "palabra " * 1000
av_std = f6.revisar(corto, C93, m_std)
av_mod = f6.revisar(corto, C93, m_mod)
ok(any("Se quedó corto" in a for a in av_std), "estándar: mil palabras son pocas")
ok(not any("Se quedó corto" in a for a in av_mod), "moderna: mil palabras no son «cortas» si pedía 1,600… (≥45 %)")
largo_m = "palabra " * 2600
ok(any("VERSIÓN MODERNA SALIÓ LARGA" in a for a in f6.revisar(largo_m, C93, m_mod)), "moderna que se pasa por la mitad")
con_preg = "Los conceptos son fundados.\n\n1. ¿Debió admitir la ampliación?\n\nSí. " + "p " * 2000
ok(any("LA FORMA ESTÁNDAR SALIÓ CON 1 PREGUNTA" in a for a in f6.revisar(con_preg, C93, m_std)),
   "estándar con preguntas → se dice")
ok(not any("LA FORMA ESTÁNDAR SALIÓ" in a for a in f6.revisar(con_preg, C93, m_mod)), "en la moderna no")

print("\n6b · LA COLETILLA DEL ÓRGANO TRAS EL RUBRO (93/2026 v10)")
import documento_generado as dg
_j = {"tipo": "Jurisprudencia"}
ok(dg._con_sujeto_tras_cita("de la Segunda Sala de la Suprema Corte de Justicia de la Nación.", _j) == "",
   "la Sala sola, detrás del rubro, se va")
ok(dg._con_sujeto_tras_cita("de la Primera Sala de la Suprema Corte de Justicia de la Nación, confirma "
                            "que los requisitos procesales son legítimos cuando permiten decidir.", _j)
   .startswith("La jurisprudencia en cita confirma que"), "con verbo detrás, recupera el sujeto")
ok(dg._con_sujeto_tras_cita("de donde se sigue que la Sala debía admitir la ampliación y correr "
                            "traslado.", _j).startswith("de donde se sigue"), "la prosa que no es órgano se queda")

print("\n7 · LAS PUERTAS ESTÁN CONECTADAS")
src_ra = open("redactor_adelanto.py", encoding="utf-8").read()
arbol = ast.parse(src_ra)
llamadas = {}
for fn in ast.walk(arbol):
    if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name in ("resolver", "resolver_en_vivo"):
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_litis_y_material":
                llamadas[fn.name] = len(n.args)
ok(llamadas == {"resolver": 5, "resolver_en_vivo": 5},
   "los DOS redactores pasan cliente y criterios a `_litis_y_material` (arranca la síntesis)")
ok("_sint.get(\"conceptos\") or r.fases.parrafos_conceptos()" in src_ra, "`_terminar` compone con la síntesis o con el completo")
ok("_fs_x.sin_contestar(" in src_ra, "`_terminar` comprueba la respuesta por concepto")
ok("not _ef_escritos" in src_ra, "el aviso de «los EFECTOS los redactas tú» calla si el estudio ya los escribió")
ok(dg.partir_efectos(["Los conceptos son fundados.", "Es fundado porque…"])[1] == [],
   "y sin efectos escritos el aviso sigue saliendo")
src_m = open("main.py", encoding="utf-8").read()
ok(src_m.count('formato: str = Form(""),') == 2, "los dos gemelos reciben `formato`")
ok(src_m.count("r.encargo.formato = _fs_m.normalizar(formato)") == 2, "y los dos lo ponen en el encargo, siempre")
ok(src_m.count('formato=getattr(r.encargo, "formato", "")') == 2, "y la ficha del proyecto lo guarda en los dos")


# La síntesis de verdad, con un cliente de mentira: recoge, valida y compone.
class _Resp:
    def __init__(self, t):
        self.choices = [type("C", (), {"message": type("M", (), {"content": t})()})()]


class _Cli:
    class chat:
        class completions:
            @staticmethod
            async def create(**kw):
                import json
                return _Resp(json.dumps(bueno))


print("\n8 · LA SÍNTESIS CORRE Y VUELVE")
import fases123_pipeline as f123
import redactor_adelanto as ra
fases = f123.Fases123(antecedentes="\n".join(ante), resumen_acto="\n".join(acto),
                      resumen_conceptos="\n".join(conc), conteo={"estado": "contado", "n": 2})
fases.problemas = P93


class _R:
    pass


r = _R()
r.fases = fases
r.encargo = type("E", (), {"formato": "moderna", "tipo_asunto": "amparo_directo", "es_recurso": False})()
mat = f6.Material()


async def _corre():
    ra._formato_al_material(r, mat, _Cli(), C93)
    return await mat.sintesis


try:
    sal = asyncio.run(_corre())
    ok(mat.formato == "moderna" and mat.n_planteamientos == 2 and len(mat.problemas) == 2,
       "el material sabe forma, reparto y cuántos hay")
    ok(sal["conceptos"] == bueno["planteamientos"] and sal["acto"] == bueno["resolvio"],
       "la síntesis vuelve condensada y validada")
except Exception as ex:
    ok(False, f"la síntesis revienta: {type(ex).__name__}: {ex}")
mat2 = f6.Material()
r.encargo.formato = ""
ra._formato_al_material(r, mat2, _Cli(), C93)
ok(mat2.formato == "estandar" and mat2.sintesis is None, "la estándar no gasta la llamada")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
