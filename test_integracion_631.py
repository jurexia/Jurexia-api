# -*- coding: utf-8 -*-
"""LA INTEGRACIÓN DE LAS CUATRO PIEZAS DEL 631 (AR 631/2025, 28-sep-2026).

Las fases A (plan-6: la JERARQUÍA del guion), B (art. 93, fr. VI: revocar una
concesión reasume jurisdicción), C1 (la tarjeta del problema principal), C2 (la
deliberación) y D (la técnica de la cita) se escribieron en paralelo. Cada una
pasa sus pruebas; aquí se prueba lo que sólo existe al juntarlas, sin modelo y
sin red:

  1. UNA sola `fuerza_para_colegiado`, en la tarjeta, importada por la
     deliberación (no dos reglas que digan cosas distintas del mismo criterio);
  2. la tarjeta pregunta a `fase_rama.conceptos_omitidos` —sin gancho— con el
     sentido de la VÍA QUE PROSPERA, no el del motor, y su desenlace es el del
     documento (`tipos_asunto.puntos_reasuncion`: el amparo en hueco);
  3. las puertas del servidor (por el fuente, sin importar `main`): la tarjeta
     y la deliberación leen la rama de `info_de_rama` y la deliberación recibe
     el tribunal;
  4. el prompt v4 junta la JERARQUÍA (A) con la técnica de la cita (D) y con la
     reasunción (B) sin órdenes contrarias.

    .venv/bin/python test_integracion_631.py
"""
import ast
import os
import re
import sys

os.environ.pop("ESTUDIO_PROMPT", None)
for _v in ("DELIBERACION_ACTIVA", "DELIBERACION_CUENTAS", "DELIBERACION_REGIONES"):
    os.environ.pop(_v, None)

import deliberacion as dl
import fase6_estudio as f6
import plan_estudio as pe
import tarjeta_decision as td

FALLOS = []
AQUI = os.path.dirname(os.path.abspath(__file__))


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


def fuente(nombre):
    return open(os.path.join(AQUI, nombre), encoding="utf8").read()


def funciones(nombre):
    return {n.name for n in ast.walk(ast.parse(fuente(nombre)))
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


TRIBUNAL = ("Tercer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo "
            "Segundo Circuito")

# ═══════════════════════════════════════════════════════════════════════════
print("\n1 · UNA SOLA FUERZA PARA UN COLEGIADO")
ok(dl.fuerza_para_colegiado is td.fuerza_para_colegiado and dl.clave_de_tesis is td.clave_de_tesis
   and dl.region_de_clave is td.region_de_clave,
   "la deliberación importa la de la tarjeta (y la clave y la región de la clave)")
ok("fuerza_para_colegiado" not in funciones("deliberacion.py")
   and "clave_de_tesis" not in funciones("deliberacion.py"),
   "deliberacion.py ya no define la suya")
_casos = [
    ({"instancia": "Segunda Sala", "tipo": "JURISPRUDENCIA"}, "obliga"),
    ({"instancia": "Tribunales Colegiados de Circuito", "tipo": "JURISPRUDENCIA",
      "obligatoria": True}, "orienta"),
    ({"instancia": "Plenos de Circuito", "tipo": "JURISPRUDENCIA"}, "pleno_circuito"),
    ({"instancia": "Plenos Regionales", "tipo": "JURISPRUDENCIA",
      "localizacion": "Gaceta S.J.F.; Tesis: PR.A.C.CN. J/7 K (12a.)"}, "obliga"),
    ({"instancia": "Tribunales Colegiados de Circuito", "tipo": "TESIS AISLADA",
      "numero_tesis": "XXII.3o.A.C.12 C (11a.)"}, "precedente_propio"),
]
ok(all(dl._fuerza(t, tribunal=TRIBUNAL)["fuerza"] == f
       == td.apoyo_de_tesis(dict(t, registro="1"), TRIBUNAL)["fuerza"] for t, f in _casos),
   "el catálogo de la deliberación y el apoyo de la tarjeta rotulan igual el mismo criterio")
_cat, _ = dl.construir_catalogo(
    [{"registro": "2000001", "rubro": "R.", "instancia": "Plenos Regionales", "tipo": "JURISPRUDENCIA",
      "localizacion": "Tesis: PR.A.C.CN. J/7 K (12a.)"}], tribunal=TRIBUNAL)
ok(next(iter(_cat.values()))["fuerza"] == "obliga" and next(iter(_cat.values()))["escalon"] == 1,
   "con el tribunal del encargo, la escalera sube al escalón 1 el Pleno Regional de su región")

# ═══════════════════════════════════════════════════════════════════════════
print("\n2 · LA TARJETA, LA DELIBERACIÓN Y EL DOCUMENTO DICEN LO MISMO DE LA VÍA QUE REVOCA")
RES = ("La Justicia de la Unión ampara y protege a Unión Ejemplo, A.C., contra el acto que reclamó "
       "a la Sala, por los motivos expuestos en el considerando séptimo de esta sentencia.")
_t_pts, _t_nota = td.desenlace_de("amparo_revision", "concede", "fundado",
                                  quien_recurre="tercero", resolutivo_recurrida=RES)
_d = dl.consecuencia_de("fundado", "amparo_revision", "concede", RES, quien_recurre="tercero")
import tipos_asunto as ta
_doc = [p.replace("{HUECO}", td.HUECO) for p in ta.puntos_reasuncion("", RES)]
ok(_t_pts == _d["desenlace"] == _doc and td.HUECO in _t_pts[1] and "Unión Ejemplo" in _t_pts[1],
   "los mismos puntos en la tarjeta, en la deliberación y en el documento: el amparo en hueco")
ok(_d["conceptos_omitidos"]["reasuncion"] == "concesion" and "fr. VI" in (_t_nota or ""),
   "y las dos anuncian el estudio de los conceptos que el juzgado no estudió")
ok(dl.consecuencia_de("fundado", "amparo_revision", "concede", RES,
                      quien_recurre="quejoso")["conceptos_omitidos"] is None,
   "si recurre la quejosa, ninguna los pide (una sola regla: `tipos_asunto.reasuncion`)")
_src_td = fuente("tarjeta_decision.py")
ok("GANCHOS_OMITIDOS" not in _src_td and "inspect.signature" not in _src_td
   and "_fr.conceptos_omitidos(rama_info, sentido_via, fases)" in _src_td,
   "la tarjeta llama a `fase_rama.conceptos_omitidos` directamente, sin gancho")
ok("para_tarjeta" not in funciones("deliberacion.py"),
   "una sola proyección de la deliberación sobre la tarjeta: `tarjeta_decision.armar`")

# ═══════════════════════════════════════════════════════════════════════════
print("\n3 · LAS PUERTAS DEL SERVIDOR (por el fuente)")
_main = fuente("main.py")
_tarj = _main.split("def taller_tarjeta(numero", 1)[1].split("\n@app.", 1)[0]
ok("info_de_rama(" in _tarj and "fases=r.fases" in _tarj and "resolutivo_recurrida=" in _tarj,
   "GET /taller/tarjeta: la rama de `info_de_rama` y las fases para buscar los conceptos")
ok('"conceptos_violacion": bool(' not in _tarj,
   "ya no manda los conceptos del secretario como booleano")
_nuc = _main.split("async def _taller_deliberar_nucleo", 1)[1].split("\nasync def ", 1)[0]
ok("tribunal=" in _nuc and "quien_recurre=" in _nuc and "info_de_rama(" in _nuc,
   "la deliberación recibe el tribunal (fuerza) y quién recurre (fr. VI)")

# ═══════════════════════════════════════════════════════════════════════════
print("\n4 · EL PROMPT v4: JERARQUÍA (A) + TÉCNICA DE LA CITA (D) + REASUNCIÓN (B)")
GUION = "APARTADO 1 · A1.a A1.b\n  JERARQUÍA DEL PROBLEMA P1 · DECIDE A1.b"
SEGS = [{"id": "A1.a", "concepto": 1, "parrafo": 0, "texto": "invoca la jurisprudencia registro 177529",
         "cita": "", "anclas": []},
        {"id": "A1.b", "concepto": 1, "parrafo": 0, "texto": "la sustitución no alteró la cosa juzgada",
         "cita": "", "anclas": []}]
TECNICA = [{"registro": r, "rubro": f"RUBRO {r}.", "instancia": "Segunda Sala", "tipo": "JURISPRUDENCIA",
            "texto": "texto"} for r in ("171925", "178784", "182039", "174177")]
CRIT = [f6.Criterio("¿La sustitución alteró la cosa juzgada?", "fundado", "r", "principal")]
GLOBAL_AL_REVES = {"sentido": "infundado", "razon": "La razón del motor.",
                   "alternativa": {"sentido": "fundado", "razon": "La vía que se tomó.",
                                   "apoyos": ["170378", "2007402"], "efecto": "El segundo, sin materia."}}


def prompt(var, reasuncion=None, conceptos="", guion=GUION):
    m = f6.Material(tipo_asunto="amparo_revision", materia="civil", formato="estandar",
                    n_planteamientos=1, variante=var, inventario=SEGS, tesis=[dict(t) for t in TECNICA])
    m.reasuncion = reasuncion
    return f6.prompt_estudio("ACTO", "CONC", CRIT, m, es_recurso=True, rama="revoca_fondo_niega",
                             conceptos_violacion=conceptos, propuesta_global=GLOBAL_AL_REVES,
                             guion=guion if var == "v4" else "")


REAS = {"reasuncion": "concesion", "hacen_falta": True, "tenemos": False, "donde": "",
        "sobresee_ademas": True}
p3, p4 = prompt("v3", REAS), prompt("v4", REAS)
_blq = pe.bloque(GUION)
_blq1 = " ".join(_blq.split())
# (a) A × D: la tesis de la parte dentro de un grupo de la JERARQUÍA.
ok("asiste razón a quien la invocó" in _blq1 and _blq1.count("se distingue en una o dos frases") == 1
   and "Si el problema prospera" in _blq1 and "Si el problema no prospera" in _blq1,
   "JERARQUÍA: la tesis de la parte en un grupo que prospera se reconoce; en uno que cae, se distingue")
ok("nunca como apoyo propio del proyecto" in _blq1 and "APLICA Y LE DA LA RAZÓN" in p4,
   "y en las dos direcciones, como la técnica: nunca reciclada como apoyo propio")
_aplicar = p4.split("  4. APLICAR.", 1)[1].split("  5. REMITIR", 1)[0]
ok("JERARQUÍA del guion resuelve por consecuencia" in _aplicar
   and "dentro de un grupo que el guion resuelve" in _aplicar,
   "«4. APLICAR»: la respuesta completa al criterio de la parte, salvo dentro de un grupo de consecuencia")
ok("ninguno de la parte como apoyo." not in p4 and "reciclado como apoyo propio" in p4,
   "el recordatorio final no contradice «APLICA Y LE DA LA RAZÓN»")
# (b) D × la v2 heredada.
ok("LA INSTANCIA VA SIEMPRE" in p3 and "LA INSTANCIA VA SIEMPRE" not in p4
   and "LA INSTANCIA, FUERA DEL ANUNCIO" in p4 and "NO ESCRIBAS TÚ NI EL TIPO NI EL ÓRGANO" in p4,
   "la instancia: la pone el documento en el anuncio y tú fuera de él (ya no dos órdenes contrarias)")
ok("trae la REGLA en una o\n  dos frases" in p3 and "trae la REGLA en una o\n  dos frases" not in p4,
   "la regla del criterio con una sola medida: la de la técnica")
# (c) B × D: los apoyos de la técnica de la fr. VI y los de la vía que se tomó.
ok("son las que hay que citar al justificarla" in p3 and "son las que hay que citar al justificarla" not in p4
   and "la que no sostiene ningún paso tuyo no se cita" in p4 and "171925" in p4,
   "APOYOS DE LA TÉCNICA: siguen nombrados, pero no se manda citarlos todos")
ok("(cítalos desde su texto, abajo)" in p3 and "(cítalos desde su texto, abajo)" not in p4,
   "los apoyos de la vía que se tomó tampoco se mandan citar en bloque")
# (d) B × A: el considerando de los conceptos y el guion.
ok("FALTAN ESOS CONCEPTOS" in p4 and f6._GUION_Y_CONCEPTOS not in p4
   and f6._RECUERDA_CONCEPTOS.strip() not in p4,
   "sin conceptos: el prompt pide no concluir y el guion no anuncia un considerando que no se hará")
p4c = prompt("v4", dict(REAS, tenemos=True, donde="secretario"),
             conceptos="PRIMERO. Concepto de violación de prueba. " * 20)
ok("REVOCAR NO ES NEGAR — REASUNCIÓN" in p4c and f6._GUION_Y_CONCEPTOS in p4c
   and f6._RECUERDA_CONCEPTOS.strip() in p4c
   and p4c.index(f6._RECUERDA_CONCEPTOS.strip()) > p4c.index("Escribe el estudio de fondo."),
   "con conceptos: el guion dice que su considerando va después y no es una desviación")
ok(f6._GUION_Y_CONCEPTOS not in prompt("v4", None, guion="APARTADO 1 · A1.a A1.b").replace(
       "ESTUDIO DE LOS CONCEPTOS", ""),
   "sin reasunción ni sobreseimiento, el guion no habla de ese considerando")
_q = lambda t: {" ".join(x.split()) for x in re.findall(r"«([^«»]{1,300})»", t)}
ok(not (_q(p4c) - _q(p3) - _q(_blq) - _q(prompt("v3", dict(REAS, tenemos=True, donde="secretario"),
                                                conceptos="PRIMERO. Concepto de violación de prueba. " * 20))),
   "los ajustes de la integración no meten ninguna frase entre comillas para copiar")
ok(prompt("v4", REAS, guion="") == p3, "sin guion, la v4 sigue siendo la v3")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
