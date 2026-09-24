"""El diálogo constitucional en el taller, sin tocar la red — 24-sep-2026.

    .venv/bin/python test_dialogo_constitucional.py
"""
import asyncio
import inspect

import dialogo_constitucional as dc
import fase5_propuesta as f5
import fase6_estudio as f6
import fase6_rag as rag
import marco_juridico as mj

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


def _tesis(reg, rubro="RUBRO", **kw):
    return {"registro": reg, "rubro": rubro, "texto": f"texto de {reg}", "instancia": "Primera Sala", **kw}


print("\n1 · EL CANON, POR REGISTRO")
ok({"160525", "2014332", "2002000", "2006224", "2004748"} <= dc.REGISTROS_METODO, "el método base")
ok({"2007064", "2007621", "2005717", "2005917"} <= dc.REGISTROS_METODO, "y el de la puerta procesal")
ok(all(r.isdigit() for r in dc.REGISTROS_METODO), "sólo registros, nunca claves (160525 no tiene clave en el acervo)")

print("\n2 · LA PUERTA PROCESAL")
ok(dc.hay_puerta([{"pregunta": "¿Procedía el sobreseimiento?", "clase": "procedencia"}]), "por la clase «procedencia»")
ok(dc.hay_puerta([{"pregunta": "¿La prueba debió admitirse?", "clase": "procesal"}]), "por la clase «procesal»")
ok(dc.hay_puerta(["¿Fue correcto desechar la demanda por extemporánea?"]), "sin clase, por la raíz (problemas viejos)")
ok(not dc.hay_puerta([{"pregunta": "¿El bloqueo de cuentas es un acto de molestia?", "clase": "fondo"}]),
   "un problema de fondo no es puerta")

print("\n3 · EL MÉTODO ENTRA AL MATERIAL")
_orig = rag.tesis_por_registro


async def _falso(qdrant, regs):
    return [_tesis(r, tecnica=True) for r in regs]

rag.tesis_por_registro = _falso
try:
    m = f6.Material()
    m.tesis = [_tesis("2016903", "ACTOS BANCARIOS…"), _tesis("2029549", "BLOQUEO FINANCIERO…")]
    n1 = asyncio.run(dc.inyectar(object(), m, [{"pregunta": "¿es acto de molestia?", "clase": "fondo"}]))
    ok(n1 == 5 and [t["registro"] for t in m.tesis[:5]] == [r for r, _ in dc.CANON_BASE],
       "sin puerta: las cinco del método, DELANTE (el tope de 80 al guardar se lleva lo del final)")
    ok(all(t.get("metodo") and t.get("tecnica") and t.get("metodo_para") for t in m.tesis[:5]),
       "marcadas: método, exentas del recorte y con su para qué")
    ok(asyncio.run(dc.inyectar(object(), m, [{"pregunta": "x", "clase": "fondo"}])) == 0, "no se duplican")
    n2 = asyncio.run(dc.inyectar(object(), m, [{"pregunta": "¿procede?", "clase": "procedencia"}]))
    ok(n2 == 4 and dc.con_puerta(m), "con puerta: las cuatro del acceso a la justicia")
    ok(len([t for t in m.tesis if t["registro"] == "2016903"]) == 1, "las del caso siguen, una vez")
finally:
    rag.tesis_por_registro = _orig

print("\n4 · LA PROPUESTA Y LA RAZÓN")
ok(all(not t.get("metodo") for t in f5._tesis_del_material(m)),
   "las del método no roban turnos a las del caso")
pr = f5.prompt_razon("¿Podía extenderse el bloqueo a la persona moral?", "fundado", m,
                     "resolvió", "combate", True, "amparo_revision",
                     directriz="Es acto de molestia.", marco="EL PARÁMETRO DE ESTE ASUNTO — prueba")
ok("EL DIÁLOGO CONSTITUCIONAL" in pr and "EL PARÁMETRO DE ESTE ASUNTO — prueba" in pr,
   "la razón ve el método y el parámetro")
ok("Si NO le favorece" in pr and "no la discute" in pr, "al servicio de la calificación decidida, en las dos direcciones")
ok("de 60 a 160 palabras" in pr, "con espacio para la escalera")
ok("PUERTA PROCESAL" in pr and "8.1 y 25" in pr, "y el 17 con la Convención cuando hay puerta")
sin = f6.Material()
sin.tesis = [_tesis("2016903")]
pr0 = f5.prompt_razon("¿?", "fundado", sin, "", "", True, "amparo_revision")
ok("EL DIÁLOGO CONSTITUCIONAL" not in pr0 and "de 60 a 120 palabras" in pr0, "sin método en el material, todo como antes")
pp = f5.prompt_propuesta([{"pregunta": "¿?"}], m, "resolvió", "combate", True, marco="PARÁMETRO-X")
ok("PARÁMETRO-X" in pp and "sopesa la\nlectura más protectora" in pp, "la propuesta también, en su modo")

print("\n5 · EL ESTUDIO")
bm = f6._bloque_material(m)
ok("CRITERIO DE MÉTODO" in bm, "las del método van marcadas en el material")
ok("{_dc_e.cierre_estudio(material)}" in inspect.getsource(f6.prompt_estudio), "el cierre del diálogo es lo último que lee")
ce = dc.cierre_estudio(m)
ok("NO cuentan entre los tres a seis" in ce and "retórica" in ce, "sin contar entre las del caso y sin forzarlo")
ok(dc.cierre_estudio(sin) == "", "sin método, el estudio no cambia")

print("\n6 · EL MARCO: EL 17 ENTRA POR LA PUERTA, Y DELANTE")
ok("8.1 y 25" in mj.CONSULTA_POR_ARTICULO["17"], "la consulta del 17 nombra los artículos 8.1 y 25")


async def _buscar_falso(qdrant, coleccion, vector, v, limite, filtro=None):
    tipos = []
    try:
        tipos = filtro.must[0].match.any
    except Exception:
        pass
    if "constitucion" in tipos:
        return [{"jerarquia": f"CPEUM > TÍTULO PRIMERO > Art. {a} CPEUM (parte 1)", "tipo": "constitucion",
                 "texto": f"Texto del artículo {a}."} for a in ("1", "4", "14", "16", "17")]
    if "convencion" in tipos:
        return [{"ref": "Artículo 25. Protección Judicial", "origen": "Convención Americana sobre Derechos Humanos",
                 "texto": "Toda persona tiene derecho a un recurso sencillo y rápido…", "tipo": "convencion"}]
    return []


async def _embed(t):
    return [0.0] * 8

_ob = mj._buscar
mj._buscar = _buscar_falso
try:
    probs = ["¿Debió admitirse la demanda de alimentos del menor pese a la extemporaneidad?",
             "¿Se respetó el debido proceso en la audiencia?"]
    sin_p = asyncio.run(mj.construir(None, _embed, probs))
    con_p = asyncio.run(mj.construir(None, _embed, probs, puerta=True))
    a_sin = [x.articulo for x in sin_p.constitucionales]
    a_con = [x.articulo for x in con_p.constitucionales]
    ok("17" not in a_sin, f"sin puerta el cupo de tres se llena con otros ({a_sin})")
    ok("17" in a_con and "1" in a_con, f"con puerta entran el 1º y el 17 ({a_con})")
    ok(any("Protección Judicial" in x.articulo for x in con_p.convencionales), "y la Convención Americana, artículo 25")
    br = mj.bloque_para_razonar(con_p)
    ok("EL PARÁMETRO DE ESTE ASUNTO" in br and "Artículo 17 de la Constitución" in br, "el parámetro para razonar")
    ok("EXTENSIÓN" not in br and "BISAGRA" not in br, "sin las reglas de redacción del marco")
finally:
    mj._buscar = _ob

print("\n" + ("TODAS LAS COMPROBACIONES PASAN" if not FALLOS else f"{len(FALLOS)} FALLAN: {FALLOS}"))
raise SystemExit(1 if FALLOS else 0)
