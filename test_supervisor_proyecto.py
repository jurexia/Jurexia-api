"""El supervisor del proyecto (2-oct-2026, David: «un LLM o agente que, sin
tanto costo, revise el proyecto y ajuste sus errores»). Sin red; el modelo es
un doble. Comprueba:

  1 · los párrafos numerados como los cuenta el mapa de marcas, con los FIJOS;
  2 · las frases de herramienta del AR 631/2025, sin acusar prosa buena;
  3 · el parseo de los parches (JSON limpio, con cercas, basura, mal formados);
  4 · cada guarda, con un caso que pasa y otro que se descarta;
  5 · la aplicación: las marcas siguen ahí, el mapa no pierde un argumento;
  6 · `supervisar` de punta a punta: aplicado, fallo, vencido, sin cambios,
      el piso de palabras, el segundo parche al mismo párrafo, el informe del
      contrato D y el aviso corto;
  7 · el prompt: hallazgos situados por párrafo, sin marcas, sin frases modelo;
  8 · el barrido adelantado: lo contestado no se vuelve a preguntar;
  9 · el camino: los DOS gemelos lo llaman tras `_congruencia_apertura`, el de
      flujo emite «revisando», main.py lo reenvía, el meta lo lleva sólo si
      corrió y la cabecera del camino plano; apagado, `activo()` es False;
 10 · la revisión adversarial del 3-oct-2026: el desenlace no se voltea, el
      calificativo tampoco, «endereza» acotado por el plan, condensar no quita
      una respuesta, registros y rubros intactos, «3-4» no es 34, los rótulos
      del guion son fijos, el memo no recuerda fallos, la salida ilegible es
      «fallo», los antecedentes como texto, X-Supervisor y /taller/proyecto.

    .venv/bin/python test_supervisor_proyecto.py
"""
import ast
import asyncio
import os
import re
import types

for _v in ("SUPERVISOR_PROYECTO", "SUPERVISOR_PROYECTO_ACTIVO", "SUPERVISOR_MODELO",
           "SUPERVISOR_TOPE_S"):
    os.environ.pop(_v, None)

import marcas as mc
import supervisor_proyecto as S

FALLOS = []


def ok(cond, nombre):
    print(("  ✓ " if cond else "  ✗ ") + nombre)
    if not cond:
        FALLOS.append(nombre)


SN = types.SimpleNamespace
CRIT = [SN(problema="¿La adquisición del inmueble sustituye a la actora en la ejecución?",
           sentido="infundado", jerarquia="principal", razonamiento="La propiedad no transmite "
           "los derechos personales del contrato."),
        SN(problema="¿Debió estudiar los restantes conceptos?", sentido="infundado",
           jerarquia="accesorio", razonamiento="")]
MATERIAL = SN(
    tesis=[{"registro": "2026918", "rubro": "COSA JUZGADA Y SUS EFECTOS DIRECTO Y REFLEJO."},
           {"registro": "173604", "rubro": "CAUSAHABIENCIA. PARA EFECTOS PROCESALES…",
            "clave": "1a./J. 82/2013 (10a.)"}],
    normas=[{"articulo": "2294", "cuerpo_legal": "Código Civil del Estado de Querétaro"},
            {"articulo": "49", "cuerpo_legal": "Código de Procedimientos Civiles del Estado de Querétaro"}],
    n_planteamientos=2, variante="v4", inventario=[], problemas=[])
FUENTES = ["La resolución de ocho de abril de dos mil veinticuatro declaró procedente la sustitución.",
           "La recurrente dijo: «la Unión carece de legitimación para ejecutar la sentencia de origen»."]

ESTUDIO = "\n".join([
    "SEXTO. Estudio. Los agravios son infundados.",
    "",
    "⟦C1.a⟧ Sobre el primer agravio, en el que la recurrente sostiene que se alteró la cosa juzgada. Se considera infundado.",
    "",
    "Lo anterior, porque la transmisión del inmueble no acredita la cesión de los derechos personales derivados del contrato, conforme al artículo 2294 del Código Civil del Estado de Querétaro.",
    "",
    "Sirve de apoyo la jurisprudencia de registro digital 2026918, de rubro siguiente:",
    "",
    "«COSA JUZGADA Y SUS EFECTOS DIRECTO Y REFLEJO. DIFERENCIAS Y REQUISITOS PARA SU ACTUALIZACIÓN.»",
    "",
    "Además, la escritura no obra en el material proporcionado, de modo que no puede tenerse por demostrada la adquisición alegada por la recurrente.",
    "",
    "Esto es, la sola adquisición del inmueble no basta para sustituir a la parte actora en la etapa de ejecución.",
    "",
    "⟦C2.a⟧",
    "En relación con el segundo agravio, que reitera lo anterior, se considera infundado por las mismas razones expuestas en este considerando.",
    "",
    "Sirve de apoyo la tesis de rubro «CAUSAHABIENCIA. PARA EFECTOS PROCESALES, SU ACTUALIZACIÓN REQUIERE QUE SE ACREDITE.», que resulta aplicable al caso concreto que se resuelve.",
    "",
    "EFECTOS",
    "1. Dejar insubsistente la resolución reclamada y emitir otra en la que se reitere lo no afectado.",
])

print("1 · los párrafos numerados")
BL = S.bloques(ESTUDIO)
por_n = {b["n"]: b for b in BL}
ok(len(BL) == 11, f"once párrafos con texto (hay {len(BL)}); la marca sola no cuenta")
ok(por_n[1]["fijo"] == "encabezado", "el encabezado «SEXTO. Estudio.» es fijo")
ok(por_n[5]["fijo"] == "rubro", "el rubro transcrito es fijo")
ok(por_n[10]["fijo"] == "rótulo", "el rótulo «EFECTOS» es fijo")
ok(por_n[2]["ids"] == ["C1.a"] and por_n[2]["marcas"] == ["⟦C1.a⟧"]
   and por_n[2]["texto"].startswith("Sobre el primer agravio"),
   "la marca en línea va al párrafo y el texto se enseña sin ella")
ok(por_n[8]["ids"] == ["C2.a"] and por_n[8]["marcas"] == [] and por_n[8]["inicio"] < por_n[8]["linea"],
   "la marca sola en su renglón vale para el párrafo siguiente")
ok(S.bloques("⟦C1.a Sobre el agravio que se dejó a medias, sin cierre de marca alguno.")[0]["fijo"]
   == "marca a medias", "una marca a medias deja el párrafo fijo")

print("2 · las frases de herramienta")
_h = [f for f, _ in S.frases_de_herramienta(
    "no obra entre las constancias remitidas para este estudio; no obra en el material "
    "proporcionado; el material disponible únicamente permite; los insumos del caso")]
ok(len(_h) >= 4, f"las del AR 631/2025 se cazan ({len(_h)})")
ok(not S.frases_de_herramienta("Del material probatorio y de las constancias de autos se "
                               "advierte que la parte quejosa no ofreció la pericial; en el "
                               "material de construcción no hay defecto."),
   "«material probatorio», «constancias de autos» y «material de construcción» no se acusan")

print("3 · el parseo")
ps, malos = S.parsear('{"parches": [{"parrafo": 6, "accion": "condensar", "tipo": "extensión", '
                      '"texto": "uno dos tres cuatro cinco seis", "motivo": "largo"}]}')
ok(len(ps) == 1 and ps[0]["tipo"] == "extension" and malos == 0, "JSON limpio; «extensión» se normaliza")
ps, _ = S.parsear('```json\n{"parches": [{"parrafo": "P3", "accion": "reemplazar", "tipo": "x", '
                  '"texto": "a b c d e f", "motivo": ""}]}\n```')
ok(len(ps) == 1 and ps[0]["parrafo"] == 3 and ps[0]["tipo"] == "redaccion",
   "con cercas de código; «P3» se lee 3; un tipo desconocido es de redacción")
ps, malos = S.parsear('Aquí van: {"parches": [{"parrafo": 0, "accion": "reemplazar", "texto": "x"}, '
                      '{"parrafo": 2, "accion": "reescribir", "texto": "x"}, '
                      '{"parrafo": 2, "accion": "reemplazar", "texto": ""}, '
                      '{"parrafo": 8, "accion": "eliminar", "texto": ""}]} fin')
ok(len(ps) == 1 and ps[0]["accion"] == "eliminar" and malos == 3,
   "con basura alrededor; los mal formados se cuentan")
ok(S.parsear("no es json")[0] == [] and S.parsear('[{"parrafo": 2, "accion": "eliminar"}]')[0][0]["parrafo"] == 2,
   "sin JSON, nada; una lista suelta también vale")
_cortada = ('{\n "parches": [\n {"parrafo": 3, "accion": "condensar", "tipo": "extension", '
            '"texto": "uno dos {tres} cuatro \\"cinco\\" seis", "motivo": "}"},\n'
            ' {"parrafo": 5, "accion": "reemplazar", "tipo": "redaccion", "texto": "Respecto de la escritura pública número 20,1')
ps, _ = S.parsear(_cortada)
ok(len(ps) == 1 and ps[0]["parrafo"] == 3 and ps[0]["texto"] == 'uno dos {tres} cuatro "cinco" seis',
   "salida cortada a media lista (medido en el 631): se rescatan los parches enteros, con llaves y comillas dentro")
ok(S.parsear('{"parches": [{"parrafo": 2, "accion": "reemplazar", "texto": "a\\n\\nb c d e f"}]}')[0][0]["texto"]
   == "a b c d e f", "el texto nuevo va en un solo renglón")

print("4 · las guardas")
AN = S.Anclas(MATERIAL, FUENTES, CRIT)


def g(n, accion, texto="", tipo="redaccion"):
    return S.guardas({"parrafo": n, "accion": accion, "texto": texto, "tipo": tipo}, por_n.get(n), AN)


P2 = por_n[2]["texto"]
P3 = por_n[3]["texto"]
ok(g(99, "reemplazar", "uno dos tres cuatro cinco seis") == "párrafo inexistente", "párrafo inexistente")
ok(g(1, "reemplazar", "SEXTO. Estudio. Los agravios son infundados del todo.").startswith("párrafo fijo"),
   "el encabezado no se toca")
ok(g(5, "reemplazar", "«COSA JUZGADA. OTRA COSA CUALQUIERA QUE NO ES EL RUBRO.»").startswith("párrafo fijo"),
   "el rubro transcrito no se toca")
ok(g(2, "eliminar").startswith("eliminar un párrafo que contesta"), "no se elimina lo que lleva marcas")
ok(g(3, "eliminar") == "eliminar un párrafo con citas", "no se elimina lo que cita")
ok(g(7, "eliminar") == "", "se elimina lo que repite sin citar ni calificar")
_pc = S.bloques("Por tanto, el agravio resulta infundado y debe desestimarse en este punto.")[0]
ok(S.guardas({"parrafo": 1, "accion": "eliminar", "texto": "", "tipo": "repeticion"}, _pc, AN)
   == "eliminar un párrafo que califica", "no se elimina lo que califica")
ok(g(7, "condensar", "Esto es, no basta.").startswith("párrafo de menos de 6"), "menos de seis palabras: fuera")
ok(g(7, "condensar", por_n[7]["texto"] + " y nada más") == "condensar sin acortar", "condensar sin acortar: fuera")
_largo = P3 + " Esa conclusión es la que se sostiene en el presente asunto, sin más."
ok(g(3, "reemplazar", _largo).startswith("alarga el párrafo"), "alargar más de un 20% en redacción: fuera")
ok(g(3, "reemplazar", _largo, "error_juridico") == "", "un error jurídico sí puede alargar")
ok(g(2, "reemplazar", P2.replace("infundado", "fundado")).startswith("cambia la calificación"),
   "voltear la calificación: fuera")
ok(g(2, "condensar", "Sobre el primer agravio, relativo a la cosa juzgada, se considera infundado.") == "",
   "condensar conservando calificación y ordinal: pasa")
_mal = S.bloques("Sobre el primer agravio, que plantea la cosa juzgada, se considera fundado en este punto.")[0]
_rep = {"parrafo": 1, "accion": "reemplazar", "tipo": "incongruencia",
        "texto": "Sobre el primer agravio, que plantea la cosa juzgada, se considera infundado en este punto."}
ok(S.guardas(_rep, _mal, AN) == "", "la incongruencia que endereza al sentido único del criterio: pasa")
ok(S.guardas(dict(_rep, tipo="redaccion"), _mal, AN).startswith("cambia la calificación"),
   "la misma corrección como «redacción»: fuera")
ok(g(3, "reemplazar", P3 + " Ver registro digital 999999.", "error_juridico").startswith("registro fuera"),
   "un registro que no está en el material: fuera")
ok(g(3, "reemplazar", P3.replace("conforme al", "como dice la tesis 173604 y el"), "cita") == "",
   "un registro del material: pasa")
ok(g(3, "reemplazar", P3 + " Así lo dice la 1a./J. 99/2019.", "error_juridico").startswith("número de tesis"),
   "una clave con año que no consta: fuera")
ok(g(3, "reemplazar", P3.replace("2294", "1796"), "cita").startswith("artículo fuera"),
   "un artículo fuera del párrafo y de las normas: fuera")
ok(g(3, "reemplazar", P3.replace("2294 del Código Civil", "49 del Código de Procedimientos Civiles"), "cita") == "",
   "un artículo de las normas del material: pasa")
ok(g(7, "reemplazar", "Esto es, la adquisición del doce de marzo de dos mil veinte no basta para sustituir.")
   == "fecha que no consta", "una fecha que no consta: fuera")
ok(g(7, "reemplazar", "Esto es, la resolución de ocho de abril de dos mil veinticuatro no basta para sustituir.")
   == "", "una fecha de las fuentes: pasa")
ok(g(7, "reemplazar", "⟦C1.b⟧ Esto es, la sola adquisición del inmueble no basta.") == "marca dentro del texto",
   "una marca dentro del texto: fuera")
ok(g(7, "reemplazar", "Esto es, como se dijo en C1.a, la adquisición no basta.")
   == "identificador interno en el texto", "un identificador interno: fuera")
ok(g(9, "condensar", "Sirve de apoyo la tesis citada, aplicable al caso concreto.") ==
   "altera o quita un rubro transcrito", "quitar el rubro de un párrafo que lo lleva: fuera")
ok(g(6, "reemplazar", "Además, la recurrente afirmó que «el juez reconoció expresamente la "
                      "causahabiencia de la incidentista» sin acreditarlo.", "hecho_no_acreditado")
   == "transcripción entrecomillada que no consta", "una transcripción inventada: fuera")
ok(g(6, "reemplazar", "Además, la recurrente afirmó que «la Unión carece de legitimación para ejecutar "
                      "la sentencia de origen», sin acreditarlo.", "hecho_no_acreditado") == "",
   "una transcripción que consta en las fuentes: pasa")
ok(g(6, "reemplazar", "Además, la escritura no obra en el material disponible, por lo que no se acredita.")
   == "frase de herramienta", "escribir una frase de herramienta: fuera")
ok(g(6, "reemplazar", "Además, la escritura no consta en autos, de modo que la adquisición alegada "
                      "por la recurrente es sólo su afirmación.", "hecho_no_acreditado") == "",
   "quitar la frase de herramienta y no dar por cierto el hecho: pasa")
_atr = S.bloques("En el escrito de agravios se reproduce que la primera sentencia rescindió el contrato "
                 "y ordenó entregar el inmueble; por ello la interlocutoria incidió en el fallo.")[0]
_dz = {"parrafo": 1, "accion": "condensar", "tipo": "extension",
       "texto": "La primera sentencia rescindió el contrato; por ello la interlocutoria incidió en el fallo."}
ok(S.guardas(_dz, _atr, AN).startswith("borra la atribución"),
   "condensar borrando «en el escrito de agravios se reproduce»: fuera (prueba real del 631)")
ok(S.guardas(dict(_dz, texto="Según los agravios, la primera sentencia rescindió el contrato; por ello la "
                              "interlocutoria incidió en el fallo."), _atr, AN) == "",
   "condensar conservando la atribución: pasa")
ok(g(2, "condensar", "Sobre el agravio relativo a la cosa juzgada, se considera infundado.")
   == "pierde el ordinal del concepto que contesta", "perder el ordinal: fuera")
ok(g(3, "reemplazar", "En el segundo agravio, la transmisión del inmueble no acredita la cesión de los "
                      "derechos conforme al artículo 2294.").startswith("nombra un concepto"),
   "nombrar un concepto que el párrafo no contestaba: fuera")
ok(g(11, "condensar", "Dejar insubsistente la resolución reclamada y emitir otra.") == "pierde el número de la orden",
   "la orden de efectos sin su número: fuera")
ok(g(11, "condensar", "1. Dejar insubsistente la resolución reclamada y emitir otra.") == "",
   "la orden condensada con su número: pasa")

print("5 · la aplicación")
_parches = [{"parrafo": 2, "accion": "condensar", "tipo": "extension",
             "texto": "Sobre el primer agravio, relativo a la cosa juzgada, se considera infundado."},
            {"parrafo": 7, "accion": "eliminar", "tipo": "repeticion", "texto": ""},
            {"parrafo": 8, "accion": "condensar", "tipo": "extension",
             "texto": "El segundo agravio, que reitera lo anterior, se considera infundado."}]
NUEVO = S.aplicar(ESTUDIO, _parches, BL)
_l, _m = mc.separar_marcas(ESTUDIO)
_l2, _m2 = mc.separar_marcas(NUEVO)
ok(set(_m) == set(_m2) == {"C1.a", "C2.a"}, "el mapa conserva todos los argumentos")
ok("⟦C1.a⟧ Sobre el primer agravio, relativo a la cosa juzgada" in NUEVO, "la marca va delante del texto nuevo")
ok("⟦C2.a⟧\nEl segundo agravio" in NUEVO, "la marca sola sigue delante de su párrafo")
ok("Esto es, la sola adquisición" not in NUEVO and "\n\n\n" not in NUEVO, "el eliminado se va con su blanco")
ok(all(x in NUEVO for x in (por_n[3]["texto"], por_n[5]["texto"], por_n[9]["texto"], por_n[11]["texto"])),
   "lo demás queda intacto")
ok(S.aplicar(ESTUDIO, [], BL) == ESTUDIO, "sin parches, el estudio byte por byte")

print("6 · supervisar de punta a punta")


def _doble(salida, uso=None, espera=0.0, error=None):
    vistos = {}

    async def llamar(texto_prompt, tope):
        vistos["prompt"] = texto_prompt
        vistos["tope"] = tope
        if espera:
            await asyncio.sleep(espera)
        if error:
            raise error
        return salida, uso or {"entrada": 1000, "salida": 200, "razonamiento": 300}, "doble"
    return llamar, vistos


import json as _json
_salida = _json.dumps({"parches": _parches + [
    {"parrafo": 2, "accion": "reemplazar", "tipo": "redaccion",
     "texto": "Sobre el primer agravio se considera infundado, por las razones que siguen."},
    {"parrafo": 3, "accion": "reemplazar", "tipo": "cita", "texto": P3.replace("2294", "1796"),
     "motivo": "x"},
    {"parrafo": 1, "accion": "reemplazar", "tipo": "redaccion", "texto": "SEXTO. Estudio. Otra cosa distinta."}]})
_ll, _vis = _doble(_salida)
nuevo, inf = asyncio.run(S.supervisar(ESTUDIO, CRIT, MATERIAL, "resumen del acto", "resumen de agravios",
                                      FUENTES, llamar=_ll))
ok(inf["estado"] == "aplicado" and len(inf["correcciones"]) == 3 and inf["descartadas"] == 3,
   f"tres aplicadas y tres descartadas ({inf['estado']}, {len(inf['correcciones'])}, {inf['descartadas']})")
ok({m["motivo"] for m in inf["motivos_descarte"]} >= {"segundo parche al mismo párrafo"}
   and any(m["motivo"].startswith("artículo fuera") for m in inf["motivos_descarte"]),
   "los motivos de descarte quedan en el informe")
_c = inf["correcciones"][0]
ok(set(_c) >= {"tipo", "parrafo", "antes", "despues", "motivo"} and len(_c["antes"]) <= 400,
   "cada corrección con las claves del contrato D")
ok(all(k in inf for k in ("estado", "modelo", "segundos", "correcciones", "descartadas"))
   and inf["modelo"] == "doble" and inf["uso"]["razonamiento"] == 300,
   "el informe lleva estado, modelo, segundos, correcciones, descartadas y uso")
ok(inf["palabras"]["despues"] < inf["palabras"]["antes"], "el estudio sale más corto")
ok(nuevo == NUEVO, "el estudio corregido es exactamente el de los tres parches buenos")
_av = S.aviso(inf)
ok(_av.startswith("El supervisor corrigió 3 cosas del estudio:") and "palabras" in _av and len(_av) < 300,
   "el aviso corto lo dice")

_ll, _ = _doble("", error=RuntimeError("caído"))
n2, i2 = asyncio.run(S.supervisar(ESTUDIO, CRIT, MATERIAL, "", "", FUENTES, llamar=_ll))
ok(n2 == ESTUDIO and i2["estado"] == "fallo" and i2["error"] == "RuntimeError", "si falla, el estudio como estaba")
ok("falló" in S.aviso(i2), "y el aviso dice que no se revisó")
_ll, _ = _doble("{}", espera=0.5)
n3, i3 = asyncio.run(S.supervisar(ESTUDIO, CRIT, MATERIAL, "", "", FUENTES, llamar=_ll, tope_s=0.05))
ok(n3 == ESTUDIO and i3["estado"] == "vencido", "si vence (con el tope del doble), como estaba")
_ll, _ = _doble('{"parches": []}')
n4, i4 = asyncio.run(S.supervisar(ESTUDIO, CRIT, MATERIAL, "", "", FUENTES, llamar=_ll))
ok(n4 == ESTUDIO and i4["estado"] == "sin_cambios" and S.aviso(i4) == "", "sin parches: sin cambios y sin aviso")
# UNA SALIDA VACÍA O ILEGIBLE NO ES «SIN CAMBIOS» (revisión del 3-oct-2026).
for _sal, _err in (("esto no es JSON", "ilegible"), ("", "sin_respuesta"), (None, "sin_respuesta"),
                   ('{"resultado": "ok"}', "ilegible"),
                   ('{"parches": [ {"parrafo": 3, "accion": "condensar", "te', "ilegible")):
    _ll, _ = _doble(_sal)
    n4b, i4b = asyncio.run(S.supervisar(ESTUDIO, CRIT, MATERIAL, "", "", FUENTES, llamar=_ll))
    ok(n4b == ESTUDIO and i4b["estado"] == "fallo" and i4b.get("error") == _err
       and "no devolvió una respuesta legible" in S.aviso(i4b),
       f"salida {_sal!r:.30}: «fallo» ({_err}), no «sin cambios», y el aviso lo dice")
# EL PISO: condensar todo a seis palabras dejaría el estudio muy por debajo.
_todo = [{"parrafo": b["n"], "accion": "condensar", "tipo": "extension",
          "texto": "Esto es lo que se resuelve aquí."} for b in BL if not b["fijo"] and not b["ids"]
         and not S.registros_de(b["texto"]) and not S.articulos_de(b["texto"])]
_ll, _ = _doble(_json.dumps({"parches": _todo}))
_piso_ant = S.PISO_PALABRAS
S.PISO_PALABRAS = 0.95
n5, i5 = asyncio.run(S.supervisar(ESTUDIO, CRIT, MATERIAL, "", "", FUENTES, llamar=_ll))
S.PISO_PALABRAS = _piso_ant
ok(n5 == ESTUDIO and i5["estado"] == "sin_cambios"
   and any("se revierte todo" in m["motivo"] for m in i5["motivos_descarte"]),
   "si dejaría el estudio bajo el piso, se revierte todo")
_ll, _vv = _doble('{"parches": []}')
asyncio.run(S.supervisar("SEXTO. Estudio.\n\nEFECTOS", CRIT, MATERIAL, "", "", FUENTES, llamar=_ll))
ok("prompt" not in _vv, "un estudio sin párrafos corregibles no llama al modelo")

S.SUPERVISOR_ACTIVO = False
_m_ap, _av_ap = {}, []
n6 = asyncio.run(S.en_el_resolver(None, SN(fases=None), CRIT, MATERIAL, ESTUDIO, _m_ap, _av_ap))
S.SUPERVISOR_ACTIVO = True
ok(n6 == ESTUDIO and _m_ap["supervisor"]["estado"] == "apagado" and _av_ap == [],
   "con el interruptor de emergencia, «apagado» y sin llamar a nadie")
_m_f, _av_f = {}, []
_orig_sup = S.supervisar


async def _revienta(*a, **k):
    raise ValueError("x")
S.supervisar = _revienta
n7 = asyncio.run(S.en_el_resolver(None, SN(fases=None), CRIT, MATERIAL, ESTUDIO, _m_f, _av_f))
S.supervisar = _orig_sup
ok(n7 == ESTUDIO and _m_f["supervisor"]["estado"] == "fallo", "en_el_resolver nunca lanza")

print("7 · el prompt")
P = _vis["prompt"]
_P_est = P.split("═══ EL ESTUDIO, POR PÁRRAFOS ═══")[1]
ok("⟦" not in _P_est and "C1.a" not in P, "el estudio va sin marcas ni identificadores")
ok("[P1 · FIJO (encabezado): no se toca]" in P and "[P2 · contesta 1 argumento(s): no se elimina]" in P,
   "cada párrafo con su número y su condición")
ok(re.search(r"\[P6\] frase de herramienta.*material proporcionado", P) is not None,
   "el hallazgo de la frase de herramienta, situado en su párrafo")
ok("registro 2026918" in P and "artículo 2294" in P, "el catálogo cerrado del material")
ok("SENTIDO: INFUNDADO" in P and "RAZÓN DEL SECRETARIO" in P, "el criterio inmutable")
ok('"parches"' in P and "reemplazar" in P and "condensar" in P and "eliminar" in P,
   "el formato de los parches, descrito")
ok("no obra en" not in P.split("═══ EL ESTUDIO, POR PÁRRAFOS ═══")[0].split("═══ QUÉ BUSCAR ═══")[1],
   "las instrucciones describen la forma; no traen la frase a copiar")

print("8 · el barrido adelantado")
import barrido_preceptos as bp
_llamadas = []


async def _preg_falso(pares, segundos):
    _llamadas.append(list(pares))
    return [{"n": i, "existe": (num != "999")} for i, (ley, num) in enumerate(pares, 1)]


async def _conf_falso(pares, segundos):
    _llamadas.append(["conf"] + list(pares))
    return set(pares)


_orig_p, _orig_c = bp._preguntar, bp._confirmar
bp._preguntar, bp._confirmar = _preg_falso, _conf_falso
try:
    memo = S.BarridoMemo()
    a = asyncio.run(memo.preguntar([("Ley X", "1"), ("Ley X", "999")], 5))
    b = asyncio.run(memo.preguntar([("Ley X", "999"), ("Ley Y", "7"), ("Ley X", "1")], 5))
    ok(len(_llamadas) == 2 and _llamadas[1] == [("Ley Y", "7")], "sólo se pregunta lo que faltaba")
    ok([d["n"] for d in b] == [1, 2, 3] and b[0]["existe"] is False and b[2]["existe"] is True,
       "las respuestas vuelven con el número de su lote")
    c1 = asyncio.run(memo.confirmar([("Ley X", "999")], 5))
    c2 = asyncio.run(memo.confirmar([("Ley X", "999")], 5))
    ok(c1 == c2 == {("Ley X", "999")} and sum(1 for x in _llamadas if x[0] == "conf") == 1,
       "la confirmación tampoco se repite")
    # LA CONFIRMACIÓN QUE FALLÓ NO SE RECUERDA (3-oct-2026): `_confirmar`
    # devuelve set() si vence o falla la red; el barrido final vuelve a pedirla.
    _veces = []

    async def _conf_falla_luego(pares, segundos):
        _veces.append(list(pares))
        return set() if len(_veces) == 1 else set(pares)
    bp._confirmar = _conf_falla_luego
    memo2 = S.BarridoMemo()
    d1 = asyncio.run(memo2.confirmar([("Ley Z", "884")], 5))
    d2 = asyncio.run(memo2.confirmar([("Ley Z", "884")], 5))
    ok(d1 == set() and d2 == {("Ley Z", "884")} and len(_veces) == 2,
       "una confirmación fallida se vuelve a pedir en el barrido final")
finally:
    bp._preguntar, bp._confirmar = _orig_p, _orig_c
_rel = SN(estudio=["uno", "dos"], antecedentes=["tres"], resumen_acto=[], resumen_conceptos=["cuatro"],
          problemas=None)
_est = SN(apertura="", visto="", resultandos=[{"titulo": "I", "texto": "cinco"}], competencia="seis",
          existencia="", procedencia="siete")
ok(S.texto_para_barrido(_rel) == "uno\ndos\ntres\ncuatro"
   and S.texto_para_barrido(_rel, _est) == "uno\ndos\ntres\ncuatro\ncinco\nseis\nsiete",
   "el texto adelantado sale del relleno y de la estructura que ya escribió el adelanto")

print("9 · el camino")
src = open("redactor_adelanto.py", encoding="utf-8").read()
_arbol = ast.parse(src)
_fn = {n.name: ast.get_source_segment(src, n) for n in _arbol.body
       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
for nombre in ("resolver", "resolver_en_vivo"):
    cuerpo = _fn[nombre]
    i_ap = cuerpo.find("_congruencia_apertura(r, criterios")
    i_sup = cuerpo.find("_sup.en_el_resolver(")
    i_ef = cuerpo.find("_efectos_de_reposicion(")
    i_ter = cuerpo.find("_terminar(")
    ok(0 < i_ap < i_sup < i_ef < i_ter and "if _sup.activo():" in cuerpo,
       f"{nombre}: el supervisor tras la congruencia y antes de efectos y _terminar, con su bandera")
ok('yield {"tipo": "revisando"}' in _fn["resolver_en_vivo"] and "revisando" not in _fn["resolver"],
   "sólo el gemelo de flujo emite «revisando»")
_ter = _fn["_terminar"]
ok("BarridoMemo()" in _ter and "preguntar=_memo_bar.preguntar" in _ter
   and "_r_bar = await _bp.barrer(_plano, material)" in _ter,
   "_terminar adelanta el barrido con la bandera y conserva el camino de siempre")
msrc = open("main.py", encoding="utf-8").read()
ok(re.search(r'elif tipo == "revisando":\s*\n(?:\s*#.*\n)*\s*_cola\.put_nowait\(\{"tipo": "revisando"\}\)', msrc)
   is not None, "main.py reenvía «revisando» por el flujo")
_marbol = ast.parse(msrc)
_mfn = {n.name: ast.get_source_segment(msrc, n) for n in _marbol.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name in ("_taller_meta_listo", "_taller_cabecera_supervisor", "_taller_plan_adelantar")}
_ns = {"_commit_desplegado": lambda: "", "os": os}
exec(_mfn["_taller_meta_listo"], _ns)
exec(_mfn["_taller_cabecera_supervisor"], _ns)
_res_sin = SN(meta_estudio={"variante": "v4"})
_res_con = SN(meta_estudio={"variante": "v4", "supervisor": inf})
ok("supervisor" not in _ns["_taller_meta_listo"](_res_sin) and _ns["_taller_cabecera_supervisor"](_res_sin) == {},
   "sin supervisor, el meta y las cabeceras no cambian")
ok(_ns["_taller_meta_listo"](_res_con)["supervisor"]["estado"] == "aplicado"
   and _ns["_taller_cabecera_supervisor"](_res_con) == {"X-Supervisor": "3 correcciones"},
   "con supervisor, el meta lo lleva y la cabecera dice cuántas correcciones")
ok("**_taller_cabecera_supervisor(r2)" in msrc, "el camino plano pone la cabecera")
_ns2 = {"os": os, "_taller_es_casa": lambda c: c == "casa@x"}
exec(_mfn["_taller_plan_adelantar"], _ns2)
os.environ.pop("PLAN_ESTUDIO", None)
ok(_ns2["_taller_plan_adelantar"]("casa@x") is True and _ns2["_taller_plan_adelantar"]("otro@x") is False,
   "plan adelantado: con la bandera apagada, sólo casa (como hoy)")
os.environ["SUPERVISOR_PROYECTO"] = "todos"
ok(_ns2["_taller_plan_adelantar"]("otro@x") is True and S.activo() is True,
   "con el supervisor para todos, el plan se adelanta para todos")
os.environ["SUPERVISOR_PROYECTO"] = "0"
ok(S.activo() is False and _ns2["_taller_plan_adelantar"]("otro@x") is False, "apagado, nada")
os.environ.pop("SUPERVISOR_PROYECTO", None)
ok(S.activo() is False, "por omisión («casa»), fuera de una sesión de pruebas, no rige")

print("10 · la revisión adversarial del 3-oct-2026")
# (1) EL DESENLACE. Revisión en plenitud: «revocar y negar» → «revocar y conceder».
_cierre = S.bloques("Por lo expuesto, procede revocar la sentencia recurrida y negar el amparo y "
                    "protección de la Justicia Federal a la parte quejosa.")[0]
CRIT_P = [SN(problema="¿Procedía el sobreseimiento?", sentido="fundado", jerarquia="principal", razonamiento=""),
          SN(problema="¿Son fundados los conceptos?", sentido="infundado", jerarquia="accesorio", razonamiento="")]
AN_P = S.Anclas(MATERIAL, FUENTES, CRIT_P)
for _t in ("error_juridico", "incongruencia", "redaccion"):
    ok(S.guardas({"parrafo": 1, "accion": "reemplazar", "tipo": _t,
                  "texto": _cierre["texto"].replace("negar", "conceder")}, _cierre, AN_P)
       .startswith("cambia el desenlace"), f"negar → conceder en plenitud ({_t}): fuera")
_noprocede = S.bloques("En consecuencia, no procede revocar la sentencia recurrida, cuyas consideraciones "
                       "subsisten en sus términos.")[0]
ok(S.guardas({"parrafo": 1, "accion": "reemplazar", "tipo": "error_juridico",
              "texto": "En consecuencia, procede revocar la sentencia recurrida, cuyas consideraciones no "
                       "subsisten en sus términos."}, _noprocede, AN_P).startswith("cambia el desenlace"),
   "«no procede revocar» → «procede revocar»: fuera")
for _de, _a in (("únicamente para efectos de que", "para que"), ("se sobresee en el juicio", "se niega el amparo"),
                ("ordenar la reposición del procedimiento", "ordenar que se dicte otra sentencia")):
    _b = S.bloques(f"Por tanto, lo procedente es {_de} la autoridad responsable deje insubsistente el acto "
                   f"reclamado en este asunto.")[0]
    ok(S.guardas({"parrafo": 1, "accion": "reemplazar", "tipo": "error_juridico",
                  "texto": _b["texto"].replace(_de, _a)}, _b, AN_P).startswith("cambia el desenlace"),
       f"«{_de}» → «{_a}»: fuera")
ok(S.guardas({"parrafo": 1, "accion": "condensar", "tipo": "extension",
              "texto": "Por lo expuesto, procede revocar la sentencia recurrida y negar el amparo a la quejosa."},
             _cierre, AN_P) == "", "condensar el cierre conservando el desenlace: pasa")
ok(S.guardas({"parrafo": 1, "accion": "eliminar", "tipo": "repeticion", "texto": ""}, _cierre, AN_P)
   == "eliminar un párrafo que dicta el desenlace", "eliminar el párrafo del desenlace: fuera")
# La red de seguridad de `supervisar`: aunque una guarda dejara pasar el parche,
# el desenlace del estudio entero no cambia.
_EST_P = ESTUDIO + "\n\n" + _cierre["texto"]
_n_cierre = len(S.bloques(_EST_P))
_ll, _ = _doble(_json.dumps({"parches": [{"parrafo": _n_cierre, "accion": "reemplazar", "tipo": "error_juridico",
                                          "texto": _cierre["texto"].replace("negar", "conceder")}]}))
_g_orig = S.guardas
S.guardas = lambda p, b, a: ""
try:
    n8, i8 = asyncio.run(S.supervisar(_EST_P, CRIT_P, MATERIAL, "", "", FUENTES, llamar=_ll))
finally:
    S.guardas = _g_orig
ok(n8 == _EST_P and i8["estado"] == "sin_cambios"
   and any("desenlace" in m["motivo"] for m in i8["motivos_descarte"]),
   "si la suma de parches cambiara el desenlace del estudio, se revierte todo")
_ll, _ = _doble(_json.dumps({"parches": [{"parrafo": _n_cierre, "accion": "reemplazar", "tipo": "error_juridico",
                                          "texto": _cierre["texto"].replace("negar", "conceder")}]}))
n8b, i8b = asyncio.run(S.supervisar(_EST_P, CRIT_P, MATERIAL, "", "", FUENTES, llamar=_ll))
ok(n8b == _EST_P and i8b["descartadas"] == 1, "supervisar de punta a punta: el volteo del desenlace no entra")

# (2) EL CALIFICATIVO, no sólo la dirección.
for _otro in ("inoperante", "ineficaz", "inatendible", "fundado pero insuficiente"):
    for _t in ("error_juridico", "incongruencia", "redaccion"):
        ok(g(2, "reemplazar", P2.replace("infundado", _otro), _t).startswith("cambia el calificativo"),
           f"infundado → {_otro} ({_t}): fuera")
_dos = S.bloques("Sobre el primer agravio, se considera infundado en una parte y, en otra, es inoperante "
                 "porque no combate las consideraciones.")[0]
ok(S.guardas({"parrafo": 1, "accion": "condensar", "tipo": "extension",
              "texto": "Sobre el primer agravio, se considera infundado en su totalidad."}, _dos, AN)
   .startswith("cambia el calificativo"), "condensar quitando uno de dos calificativos del mismo lado: fuera")
_asiste = S.bloques("Respecto del primer agravio, no le asiste la razón a la recurrente en lo que plantea.")[0]
ok(S.guardas({"parrafo": 1, "accion": "reemplazar", "tipo": "error_juridico",
              "texto": "Respecto del primer agravio, es inoperante lo que plantea la recurrente."}, _asiste, AN)
   .startswith("cambia el calificativo"), "«no le asiste la razón» → «inoperante»: fuera")
ok(S.calificativos("Es fundado pero insuficiente el agravio.") == {"fundado_insuficiente"}
   and S.calificativos("Su estudio resulta innecesario.") == {"innecesario"}
   and S.calificativos("Sirve de apoyo la tesis «AGRAVIOS INOPERANTES. LO SON LOS QUE NO COMBATEN.»") == set(),
   "los calificativos se leen de las frases que califican, sin los rubros")

# (3) «ENDEREZA», ACOTADO. Criterio todo «fundado» y un argumento que el plan
# dejó infundado dentro del problema fundado: el volteo a «fundado» se descarta.
CRIT_F = [SN(problema="¿Prescribió la acción?", sentido="fundado", jerarquia="principal", razonamiento=""),
          SN(problema="¿Los demás conceptos?", sentido="innecesario", jerarquia="accesorio", razonamiento="")]
_est_f = "⟦C2.a⟧ Es infundado, en cambio, el planteamiento relativo a la prescripción del segundo concepto."
_bf = S.bloques(_est_f)[0]
_pf = {"parrafo": 1, "accion": "reemplazar", "tipo": "incongruencia",
       "texto": "Es fundado, en cambio, el planteamiento relativo a la prescripción del segundo concepto."}
ok(S.guardas(_pf, _bf, S.Anclas(MATERIAL, FUENTES, CRIT_F)).startswith("cambia la calificación"),
   "criterio todo fundado, sin plan que lo atribuya: el volteo se descarta")
_plan_inf = {"segmentos": [{"id": "C2.a", "etiqueta": "infundado"}]}
ok(S.guardas(_pf, _bf, S.Anclas(MATERIAL, FUENTES, CRIT_F, _plan_inf)).startswith("cambia la calificación"),
   "el plan dice infundado: el volteo a fundado se descarta")
_plan_fun = {"segmentos": [{"id": "C2.a", "etiqueta": "fundado"}]}
ok(S.guardas(_pf, _bf, S.Anclas(MATERIAL, FUENTES, CRIT_F, _plan_fun)) == "",
   "el plan dice fundado: enderezar a fundado pasa")
ok(S.guardas(dict(_pf, tipo="redaccion"), _bf, S.Anclas(MATERIAL, FUENTES, CRIT_F, _plan_fun))
   .startswith("cambia la calificación"), "y como «redacción», no")
_bi = S.bloques("⟦C1.a⟧ Sobre el primer agravio, que plantea la cosa juzgada, se considera infundado.")[0]
_plan_ino = {"segmentos": [{"id": "C1.a", "etiqueta": "inoperante"}]}
_pi = {"parrafo": 1, "accion": "reemplazar", "tipo": "incongruencia",
       "texto": "Sobre el primer agravio, que plantea la cosa juzgada, se considera inoperante."}
ok(S.guardas(_pi, _bi, S.Anclas(MATERIAL, FUENTES, CRIT, _plan_ino)) == "",
   "el plan dice inoperante y el estudio escribió infundado: la corrección pasa")
ok(S.guardas(_pi, _bi, S.Anclas(MATERIAL, FUENTES, CRIT)).startswith("cambia el calificativo"),
   "la misma corrección sin plan: fuera")

# (4) CONDENSAR NO QUITA LA RESPUESTA A UN ARGUMENTO MARCADO.
INV = [{"id": "C1.a", "texto": "El juez no valoró el dictamen pericial en grafoscopía que demostraba la "
                               "falsedad de la firma del pagaré", "anclas": []},
       {"id": "C1.b", "texto": "La confesional ficta de la demandada debió tenerse por desahogada porque no "
                               "compareció a absolver posiciones", "anclas": []},
       {"id": "C2.a", "texto": "La prescripción de la acción cambiaria directa operó porque transcurrieron "
                               "más de tres años", "anclas": []}]
MAT_INV = SN(**dict(vars(MATERIAL), inventario=INV))
_b2 = S.bloques("⟦C1.a C1.b⟧ Es infundado el primer agravio. Por una parte, el dictamen pericial en grafoscopía "
                "sí fue valorado por el juez, quien explicó por qué no demostraba la falsedad de la firma del "
                "pagaré. Por otra, la confesional ficta de la demandada no podía tenerse por desahogada, pues no "
                "fue citada legalmente a absolver posiciones.")[0]
_AN_INV = S.Anclas(MAT_INV, FUENTES, CRIT)
ok(S.guardas({"parrafo": 1, "accion": "condensar", "tipo": "extension",
              "texto": "Es infundado el primer agravio, porque el dictamen pericial en grafoscopía sí fue "
                       "valorado por el juez, quien explicó por qué no demostraba la falsedad de la firma."},
             _b2, _AN_INV) == "quita la respuesta a C1.b", "condensar dejando sólo la pericial: fuera (C1.b)")
ok(S.guardas({"parrafo": 1, "accion": "condensar", "tipo": "extension",
              "texto": "Es infundado el primer agravio: el juez sí valoró el dictamen pericial en grafoscopía "
                       "sobre la firma del pagaré, y la confesional ficta de la demandada no podía tenerse por "
                       "desahogada sin citarla a absolver posiciones."},
             _b2, _AN_INV) == "", "condensar conservando las dos respuestas: pasa")

# (5) REGISTROS Y RUBROS DEL PÁRRAFO.
_p4 = por_n[4]
ok(not _p4["fijo"] and g(4, "reemplazar", "Sirve de apoyo la jurisprudencia de registro digital 173604, de rubro "
                                          "siguiente:", "cita").startswith("quita o cambia un registro"),
   "cambiar el registro que el párrafo cita: fuera")
ok(g(4, "reemplazar", "Sirve de apoyo la jurisprudencia que se transcribe enseguida, de rubro siguiente:", "cita")
   .startswith("quita o cambia un registro"), "quitar el registro: fuera")
ok(g(3, "reemplazar", P3 + " Así lo sostiene la tesis de rubro «COSA JUZGADA REFLEJA. SUS ALCANCES EN EL "
                           "JUICIO DE ORIGEN […]».", "error_juridico").startswith("rubro entre comillas"),
   "un rubro nuevo que no es el de una tesis del material: fuera")
ok(g(3, "reemplazar", P3 + " Sirve de apoyo la tesis 2026918, de rubro «COSA JUZGADA Y SUS EFECTOS DIRECTO Y "
                           "REFLEJO.».", "error_juridico") == "",
   "un rubro nuevo que es, entero, el de una tesis del material: pasa")
MAT_SV = SN(**dict(vars(MATERIAL), tesis=list(MATERIAL.tesis) + [
    {"registro": "2011111", "rubro": "PRUEBA PERICIAL. SU VALORACIÓN.", "vigencia": {"estado": "abandonada"}}]))
ok(S.guardas({"parrafo": 3, "accion": "reemplazar", "tipo": "cita",
              "texto": P3.replace("conforme al", "como dice la tesis 2011111 y el")}, por_n[3],
             S.Anclas(MAT_SV, FUENTES, CRIT)).startswith("cita una tesis que perdió vigencia"),
   "citar una tesis del material que perdió vigencia: fuera")
ok("[SIN VIGENCIA" in S._bloque_catalogo(MAT_SV) and "PRUEBA PERICIAL. SU VALORACIÓN." in S._bloque_catalogo(MAT_SV),
   "el catálogo enseña el rubro entero y la vigencia")

# (6) EL NÚMERO DE PÁRRAFO ES UN SOLO ENTERO.
for _x, _esp in (("3-4", 0), ("1 y 2", 0), ("P34", 34), (3.0, 3), ("3.0", 3), (3.5, 0), ("párrafo 5", 5),
                 (True, 0), (7, 7), ("P3", 3)):
    ok(S._numero_de_parrafo(_x) == _esp, f"párrafo {_x!r} → {_esp}")
ps, malos = S.parsear('{"parches": [{"parrafo": "3-4", "accion": "condensar", "tipo": "repeticion", '
                      '"texto": "uno dos tres cuatro cinco seis"}]}')
ok(ps == [] and malos == 1, "«3-4» es un parche mal formado, no el párrafo 34")

# (7) LOS RÓTULOS DEL GUION SON FIJOS.
_bg = S.bloques("DESVIACIONES DEL GUION: el apartado del segundo concepto se desarrolla con su diferencia propia.\n\n"
                "APLICA C1.a por la premisa M1 del plan, que se expone primero en este estudio.\n\n"
                "**DESVIACIONES DEL GUION** ninguna que deba informarse al secretario en este punto del estudio.")
ok([b["fijo"] for b in _bg] == ["rótulo del guion"] * 3, "«DESVIACIONES DEL GUION» y «APLICA C1.a» son fijos")
import redactor_adelanto as _ra
ok(_ra._RX_ROTULO_GUION.pattern == S._RX_ROTULO_GUION_COPIA.pattern
   and _ra._RX_DESVIACIONES.pattern == S._RX_DESVIACIONES_COPIA.pattern,
   "la copia de respaldo de los rótulos es igual a la del limpiador")

# (10) LOS ANTECEDENTES SON UNA CADENA: no una letra por renglón.
_vistos_f = {}


async def _sup_espia(estudio, criterios, material, **kw):
    _vistos_f.update(kw)
    return estudio, S._informe_vacio()
S.supervisar = _sup_espia
try:
    asyncio.run(S.en_el_resolver(None, SN(fases=SN(fuentes=["f1"], antecedentes="El ocho de abril se dictó.",
                                                   autos="", resumen_acto="", resumen_conceptos=""),
                                          encargo=SN(plan={"plan": _plan_fun})),
                                 CRIT, MATERIAL, ESTUDIO, {}, []))
finally:
    S.supervisar = _orig_sup
ok("El ocho de abril se dictó." in _vistos_f.get("fuentes", []), "los antecedentes llegan como texto entero")
ok(_vistos_f.get("plan") == _plan_fun, "y el plan v4 del encargo llega a las guardas")
ok(S._como_texto(["a", "b"]) == "a\nb" and S._como_texto(None) == "", "una lista se une; nada, cadena vacía")

# main.py: X-Supervisor expuesta por CORS y /taller/proyecto con el supervisor.
_cors = re.search(r"expose_headers=\[(.*?)\]", msrc, re.S)
ok(_cors is not None and '"X-Supervisor"' in _cors.group(1), "CORS expone X-Supervisor")
_tp = next(n for n in _marbol.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "taller_proyecto")
_ns3 = {"_taller_puerta": lambda e: None, "err": str, "print": lambda *a, **k: None}


class _Q:
    def __init__(self, data):
        self.data = data

    def __getattr__(self, k):
        return lambda *a, **kw: self

    def execute(self):
        return SN(data=self.data)


exec(ast.get_source_segment(msrc, _tp), _ns3)
for _pr, _espera in (({"version": 2, "supervisor": {"estado": "aplicado", "correcciones": []}}, "aplicado"),
                     ({"version": 2}, None)):
    _ns3["supabase_admin"] = SN(table=lambda *_a, _d=[{"estado": {"proyecto": _pr}}]: _Q(_d))
    # (3-oct-2026) taller_proyecto es «def»: corre en el grupo de hilos de FastAPI.
    _rp = _ns3["taller_proyecto"]("1/2026", "x@y")
    _r = (asyncio.run(_rp) if asyncio.iscoroutine(_rp) else _rp)["proyecto"]
    ok(((_r.get("supervisor") or {}).get("estado")) == _espera and ("supervisor" in _r) == (_espera is not None),
       f"/taller/proyecto {'devuelve' if _espera else 'no inventa'} el supervisor")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
