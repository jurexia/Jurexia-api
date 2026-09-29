# -*- coding: utf-8 -*-
"""La fuente que llega después de redactar (rediseño del taller, punto 7).

    .venv/bin/python test_fuente_tardia.py
"""
import asyncio, inspect, sys, types
sys.path.insert(0, ".")
import fuente_tardia as ft
import contexto_taller as ct
import os
os.environ.pop("FUENTE_TARDIA_AVISO", None)
ct.poner(True, {})

FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


EST = "\n".join([
    "SEXTO. Estudio.",
    "⟦A1.a⟧ Es infundado el agravio, porque conforme a la jurisprudencia registro 2020001 la notificación surtió efectos.",
    "⟦M1⟧",
    "La premisa: el artículo 17 de la Ley del Seguro Social fija el plazo.",
    "II. LO QUE LA RECURRENTE ALEGA APARTE",
    "La recurrente invoca, sin razón, el criterio de registro 2020009.",
    "Por lo expuesto, se confirma.",
])
TES = [{"registro": "2020001", "instancia": "Segunda Sala", "tipo": "Jurisprudencia", "vincula_origen": True},
       {"registro": "2020009", "instancia": "Tribunales Colegiados de Circuito", "tipo": "Tesis Aislada"},
       {"registro": "2020077", "instancia": "Primera Sala", "tipo": "Tesis Aislada"}]
NOR = [{"articulo": "17", "cuerpo_legal": "Ley del Seguro Social"},
       {"articulo": "17", "cuerpo_legal": "Código Fiscal de la Federación"}]

print("\n1 · QUÉ TOCA CADA FUENTE TARDÍA")
cl = ft.clasificar(EST, TES, NOR)
d = {c["fuente"]: c for c in cl}
ok(d["registro 2020001"]["sustantiva"] and d["registro 2020001"]["unidades"] == ["A1.a"]
   and d["registro 2020001"]["vincula"] is True,
   "la jurisprudencia citada en el párrafo que contesta el agravio A1.a toca una unidad que decide")
ok(d["art. 17 — Ley del Seguro Social"]["sustantiva"] and d["art. 17 — Ley del Seguro Social"]["unidades"] == ["M1"],
   "el artículo citado en la premisa M1 (marca sola en su renglón) también")
ok(d["registro 2020009"]["estado"] == "sin_unidad" and not d["registro 2020009"]["sustantiva"],
   "la que sólo aparece en un párrafo sin unidad no se da por sustantiva (se reporta, no se inventa)")
ok(d["registro 2020077"]["estado"] == "no_usada", "la que el estudio no usa, «no_usada»")
ok(d["art. 17 — Código Fiscal de la Federación"]["estado"] == "no_usada",
   "el mismo número de otra ley NO se confunde con la citada")

print("\n1b · LA CITA, PEGADA A SU LEY (revisión del 29-sep)")
N = lambda a, l: {"articulo": a, "cuerpo_legal": l}
ok(not ft._cita_norma("⟦C1.b⟧ El artículo 17 de la Constitución garantiza el acceso; la relación de trabajo es otra cosa.",
                      N("17", "Ley Federal del Trabajo")),
   "«artículo 17 de la Constitución … trabajo» NO es la Ley Federal del Trabajo")
ok(not ft._cita_norma("⟦C2.a⟧ El artículo 14 constitucional no se viola.",
                      N("14", "Ley de Amparo, Reglamentaria de los Artículos 103 y 107 de la Constitución Política de los Estados Unidos Mexicanos")),
   "«14 constitucional» NO es la Ley de Amparo aunque su nombre mencione la Constitución")
ok(ft._cita_norma("⟦C1.a⟧ Se violan los artículos 1o., 14 y 16 de la Constitución Política.",
                  N("14", "Constitución Política de los Estados Unidos Mexicanos")),
   "la enumeración con ordinales («1o., 14 y 16») se reconoce")
ok(ft._cita_norma("⟦C1.a⟧ Conforme al artículo 51 de la LFPCA procede.",
                  N("51", "Ley Federal de Procedimiento Contencioso Administrativo")), "y las siglas")
ok(ft._cita_tesis("la jurisprudencia 1a./J. 67/2014 aplica", {"clave": "1a./J. 67/2014 (10a.)"}),
   "la clave citada sin la época («(10a.)») también")
_d = ft.clasificar("⟦C1.a⟧ Es infundado.\nLo anterior, conforme al registro 2020001.\nSÉPTIMO. Efectos.\nY el registro 2020077.",
                   [dict(TES[0]), dict(TES[2])], [])
ok(_d[0]["unidades"] == ["C1.a"] and _d[1]["estado"] == "sin_unidad",
   "la unidad sigue en el párrafo que continúa la respuesta, y un rótulo la corta")

print("\n2 · EL AVISO Y EL ESTADO DE SALIDA")
av, est = ft.informe_y_aviso(cl)
ok(est == "justificacion_pendiente" and av.startswith("JUSTIFICACIÓN PENDIENTE")
   and "A1.a" in av and "M1" in av and "vincula a este tribunal" in av,
   "sale «justificacion_pendiente» y el aviso nombra las unidades y la fuerza")
ok(ft.informe_y_aviso(ft.clasificar(EST, [TES[2]], []))[1] == "", "sin uso sustantivo, sin estado")

print("\n3 · CON EL ESTUDIO YA LIMPIO (en _terminar), POR EL MAPA")
pars = ["Es infundado el agravio, conforme al artículo 17 de la Ley del Seguro Social.", "Se confirma."]
cl2 = ft.clasificar("\n".join(pars), (), NOR[:1], mapa={"A2": [0]}, parrafos=pars)
ok(cl2[0]["sustantiva"] and cl2[0]["unidades"] == ["A2"], "el mapa {id: [índices]} da las unidades")

print("\n4 · EN EL TALLER")
import redactor_adelanto as ra
src = inspect.getsource(ra)
ok(src.count("avisos = await _fuentes_tardias(r, e, material, estudio, avisos, qdrant, _meta)") == 2,
   "resolver y resolver_en_vivo llaman a la MISMA función (el bloque copiado se juntó)")
_cons = inspect.getsource(ra.consultar)
ok("completar_tesis_citadas" in _cons and "citas_invocadas" in _cons,
   "las tesis que invoca la parte se traen AL CONSULTAR (decisión 3): ya no llegan tarde")
ok("relleno.normas = material.normas" in src,
   "los artículos recuperados en _terminar SÍ llegan al documento (antes se reasignaba otra lista)")


async def _prueba():
    class M:
        tesis = [dict(TES[1])]
        normas = []
        preceptos_de_internet = []
    async def completar_tesis_citadas(qdrant, material, citas, tipo_asunto=""):
        material.tesis.append(dict(TES[0]))
        return ["2020001"]
    import fase6_rag as f6r
    orig = f6r.completar_tesis_citadas
    f6r.completar_tesis_citadas = completar_tesis_citadas
    try:
        r = types.SimpleNamespace(fases=types.SimpleNamespace(fuentes=["", "escrito"]))
        e = types.SimpleNamespace(tipo_asunto="amparo_revision", coleccion_estatal="", materia="")
        meta = {}
        av = await ra._fuentes_tardias(r, e, M(), EST, [], object(), meta)
    finally:
        f6r.completar_tesis_citadas = orig
    return av, meta


av, meta = asyncio.run(_prueba())
ok(meta.get("estado_salida") == "justificacion_pendiente" and any(a.startswith("JUSTIFICACIÓN PENDIENTE") for a in av),
   "una tesis traída después de redactar y usada al contestar un agravio deja el proyecto en «justificacion_pendiente»")
ok(any(c["fuente"] == "registro 2020001" for c in meta.get("fuentes_tardias", [])),
   "y el meta del estudio (ficha y evento «listo») lo registra")
ct.poner(False, {})
av2, meta2 = asyncio.run(_prueba())
ok(not any(a.startswith("JUSTIFICACIÓN PENDIENTE") for a in av2)
   and meta2.get("estado_salida") == "justificacion_pendiente",
   "fuera de casa, en sombra: el aviso no se enseña hasta calibrarlo, pero el meta lo registra")

print()
if FALLOS:
    print("FALLAS:\n  " + "\n  ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
