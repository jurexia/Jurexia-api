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

print("\n4 · LA RAZÓN, EN UN SOLO SENTIDO: SÓLO A FAVOR DE LA PERSONA")
ok(all(not t.get("metodo") for t in f5._tesis_del_material(m)),
   "las del método no roban turnos a las del caso")
base = dict(problema="¿Podía extenderse el bloqueo a la persona moral?", material=m,
            resumen_acto="resolvió", resumen_conceptos="combate", es_recurso=True,
            tipo_asunto="amparo_revision", directriz="Es acto de molestia.",
            marco="EL PARÁMETRO DE ESTE ASUNTO — prueba")
contra = f5.prompt_razon(sentido="fundado", favorece=False, **base)
ok("SIN PRO PERSONA NI INTERPRETACIÓN CONFORME" in contra and "CRITERIOS DEL MÉTODO" not in contra,
   "711 con la UIF recurrente y «fundado»: NO se invocan, y sin criterios del método")
ok("de 60 a 120 palabras" in contra and "EL PARÁMETRO DE ESTE ASUNTO — prueba" in contra,
   "sin escalera no hay espacio extra, y el parámetro sí está")
a_favor = f5.prompt_razon(sentido="infundado", favorece=True, **base)
ok("ESTA CALIFICACIÓN FAVORECE A QUIEN RECLAMA EL DERECHO" in a_favor and "CRITERIOS DEL MÉTODO" in a_favor
   and "de 60 a 160 palabras" in a_favor, "la vía que favorece a la persona: escalera y criterios")
ok("PUERTA PROCESAL" in a_favor and "8.1 y 25" in a_favor, "y el 17 con la Convención cuando hay puerta")
duda = f5.prompt_razon(sentido="fundado", favorece=None, **base)
ok("SÓLO SI LA CALIFICACIÓN DE ARRIBA FAVORECE" in duda, "si no consta quién recurrió, la regla va por delante")
via = {"sentido": "infundado", "posible": True, "norma": "Artículo 115 de la Ley de Instituciones de Crédito",
       "lectura": "la extensión exige petición expresa", "limite": "", "apoyos": ["2029549"]}
ok("LA LECTURA PROTECTORA QUE SEÑALÓ LA PROPUESTA" in f5.prompt_razon(sentido="infundado", favorece=True, via=via, **base),
   "la razón parte de la lectura que señaló la propuesta, si es su vía")
ok("LA LECTURA PROTECTORA QUE SEÑALÓ" not in f5.prompt_razon(sentido="fundado", favorece=False, via=via, **base),
   "y no la trae a la vía contraria")
sin = f6.Material()
sin.tesis = [_tesis("2016903")]
pr0 = f5.prompt_razon("¿?", "fundado", sin, "", "", True, "amparo_revision")
ok("EL DIÁLOGO CONSTITUCIONAL" not in pr0 and "SIN PRO PERSONA" not in pr0 and "de 60 a 120 palabras" in pr0,
   "sin método en el material, todo como antes")

print("\n5 · LA PROPUESTA SEÑALA LA VÍA PROTECTORA")
quien = dc.quien_combate("amparo_revision", "TITULAR DE LA UNIDAD DE INTELIGENCIA FINANCIERA", True)
pp = f5.prompt_propuesta([{"pregunta": "¿?"}], m, "resolvió", "combate", True, marco="PARÁMETRO-X", quien=quien)
ok("PARÁMETRO-X" in pp and "QUIEN COMBATE AQUÍ ES UNA AUTORIDAD" in pp, "la propuesta sabe quién combate")
ok('"via_protectora": {"sentido"' in pp and "LA VÍA PROTECTORA, en `via_protectora`" in pp,
   "y pide la vía protectora, con llaves sencillas en el JSON")
ok("via_protectora" not in f5.prompt_propuesta([{"pregunta": "¿?"}], sin, "r", "c", True),
   "sin método delante no se pide (sería citar de memoria)")
v = f5._via_protectora({"sentido": "Infundado", "posible": True, "norma": "art. 115", "lectura": "x", "apoyos": ["1", "2"]})
ok(v["sentido"] == "infundado" and v["posible"] and v["apoyos"] == ["1", "2"], "se lee y se normaliza")
ok(f5._via_protectora(None) == {} and f5._via_protectora({"posible": True}) == {}, "sin sentido, vacía")

print("\n6 · EL ESTUDIO, EN UN SOLO SENTIDO")
ok("CRITERIO DE MÉTODO" in f6._bloque_material(m), "las del método van marcadas en el material")
crit = [f6.Criterio(problema="¿Podía extenderse el bloqueo?", sentido="fundado", jerarquia="principal")]
m.dialogo_favorece = False
pe_contra = f6.prompt_estudio("acto", "conceptos", crit, m, es_recurso=True)
ok("CRITERIO DE MÉTODO" not in pe_contra and "SIN PRO PERSONA NI INTERPRETACIÓN CONFORME" in pe_contra,
   "contra la persona: ni criterios del método ni peldaño, y se dice por qué")
m.dialogo_favorece = True
pe_favor = f6.prompt_estudio("acto", "conceptos", crit, m, es_recurso=True)
ok("CRITERIO DE MÉTODO" in pe_favor and "La resolución favorece a quien reclama el derecho" in pe_favor,
   "a favor de la persona: el peldaño se construye desde la lectura protectora")
m.dialogo_favorece = None
ok("SÓLO si la resolución favorece" in f6.prompt_estudio("acto", "conceptos", crit, m, es_recurso=True),
   "sin dirección, la regla condicional")
ok("NO cuentan entre los tres a seis" in dc.cierre_estudio(m, True), "sin contar entre las del caso")
ok(dc.cierre_estudio(sin) == "", "sin método, el estudio no cambia")

print("\n6b · A FAVOR DE QUIÉN")
UIF = "TITULAR Y DIRECTOR GENERAL DE LA UNIDAD DE INTELIGENCIA FINANCIERA"
ok(dc.favorece_a_la_persona("fundado", "amparo_revision", UIF, True) is False, "711: fundado con la UIF recurrente → contra la persona")
ok(dc.favorece_a_la_persona("infundado", "amparo_revision", UIF, True) is True, "711: infundado → a favor de la persona (la alternativa)")
ok(dc.favorece_a_la_persona("fundado", "amparo_directo") is True, "amparo directo: fundado favorece al quejoso")
ok(dc.favorece_a_la_persona("inoperante", "amparo_directo") is False, "amparo directo: inoperante no")
ok(dc.favorece_a_la_persona("fundado", "revision_fiscal") is False, "revisión fiscal: fundado favorece a la autoridad")
ok(dc.favorece_a_la_persona("fundado", "amparo_revision", "Interamericana de Aceites, S.A. de C.V.", True) is True,
   "revisión de la persona moral: fundado la favorece")
ok(dc.favorece_a_la_persona("fundado", "amparo_revision", "", True) is None, "sin recurrente, no se adivina")
ok(dc.favorece_a_la_persona("sin_materia", "amparo_directo") is None, "sin materia no tiene dirección")

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

print("\n7 · EL MAPA LEE LO QUE SE RESOLVIÓ, Y LO GENÉRICO VA DETRÁS")
p711 = ["¿El bloqueo de las cuentas de la persona moral podía extenderse por la inclusión de sus "
        "apoderados o autorizada en la Lista de Personas Bloqueadas?"]
r711 = ["El Juzgado de Distrito consideró que el bloqueo vulneró los derechos de legalidad y seguridad "
        "jurídica, porque no existía una determinación fundada y motivada"]
ok(mj._articulos_del_problema(p711) == [], "la pregunta del 711 sola no dispara nada (el marco salía vacío)")
ok(mj._articulos_del_problema(p711 + r711) == ["1", "14", "16"], "con lo que resolvió el juzgado: 1º, 14 y 16")
lab = mj._articulos_del_problema(["¿El despido del trabajador fue justificado?", "violó la legalidad"])
ok(lab.index("123") < lab.index("16"), f"el 123 va antes que la legalidad genérica ({lab})")
_ob = mj._buscar
mj._buscar = _buscar_falso
try:
    solo = asyncio.run(mj.construir(None, _embed, p711))
    con = asyncio.run(mj.construir(None, _embed, p711, temas_extra=r711))
    ok(solo.vacio() and not con.vacio(), "construir: sin «resolvió» vacío, con «resolvió» con parámetro")
finally:
    mj._buscar = _ob

print("\n" + ("TODAS LAS COMPROBACIONES PASAN" if not FALLOS else f"{len(FALLOS)} FALLAN: {FALLOS}"))
raise SystemExit(1 if FALLOS else 0)
