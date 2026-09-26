# -*- coding: utf-8 -*-
"""La lectura del escrito para el inventario de la v3/v4 (`inventario_escrito`).

El inventario de la v3/v4 sale del RESUMEN de la fase 2, y el resumen pierde
argumentos (12 de las 27 omisiones graves medidas el 26-sep-2026 no estaban en
la lista). Una lectura del escrito con el modelo de las fases añade lo que
falta. Esto comprueba, con DOBLES del modelo (ni una llamada de verdad):

  1 · la verificación de las citas: literal contra el escrito, tolerante a los
      renglones partidos del PDF, a mayúsculas y acentos; la parafraseada, la
      que no está y la corta se descartan; la larga se recorta; el «ya está en
      el inventario» del modelo sólo vale si copia del renglón las palabras
      que nombran el dato;
  2 · la fusión: el piso intacto (mismos ids, mismo orden, mismo texto), los
      nuevos detrás de su concepto con los ids que siguen la serie, sin
      duplicar un pasaje del piso ni otro nuevo, con tope repartido entre
      los conceptos;
  3 · la caída al piso: el modelo que falla, que devuelve basura o que no
      llega a tiempo deja el inventario de siempre, y se registra;
  4 · la caché por huella en la fila (`taller_sesiones.plan`, rama
      «inventario»), con compare-and-set: la guardada se usa sin llamar al
      modelo; otra huella, una corrida abandonada o un error se relanzan con
      tope; el plan y la lectura no se pisan; con dos workers, la primera
      lectura buena no se reescribe (ni con un error tardío ni con otra
      lista) y quien relanza usa la de la fila;
  5 · los gemelos (flujo y plano) por AST: los dos vacían la lectura del
      encargo y la fijan ANTES del plan; el plan y el material ven la misma;
  6 · la v1 intacta: ni la ve ni espera por ella, y su prompt no cambia ni una
      coma; sin lectura, el bloque de la v3 sale idéntico al de antes;
  7 · el prompt sin frases modelo ni datos de los asuntos del banco.

    .venv/bin/python test_inventario_escrito.py
"""
import ast
import asyncio
import copy
import json
import os
import re
import time
import types

os.environ.pop("ESTUDIO_PROMPT", None)
os.environ.pop("ESTUDIO_PROMPT_AD", None)
os.environ.pop("INVENTARIO_ESCRITO", None)

import fase6_estudio as f6
import fases123_pipeline as f123
import inventario as inv
import inventario_escrito as ie
import plan_estudio as pe

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


# ═══════════════════════════════════════════════════════════════════════════
# UN ASUNTO DE PRUEBA, inventado: el resumen NO trae la testimonial del
# primer concepto (los testigos aleccionados), el escrito sí.
# ═══════════════════════════════════════════════════════════════════════════
RESUMEN = """En contra de esas consideraciones, la parte quejosa plantea los siguientes conceptos de violación:

En el primer concepto de violación aduce que la Sala responsable transgrede el artículo 14 constitucional, porque tiene por acreditada la identidad del inmueble sin la prueba pericial en topografía que exige la acción reivindicatoria. Refiere que las superficies de cuatro hectáreas y de dos hectáreas y media que se mencionan en autos son contradictorias entre sí.

En el segundo concepto de violación afirma que lo resuelto en el expediente 1114/2017 produce cosa juzgada refleja, porque en aquel juicio se desestimó la misma acción por falta de identidad del predio.

Finalmente, en el tercer concepto de violación aduce que la condena en costas es ilegal porque actuó sin dolo ni mala fe en ninguna de las instancias."""
RENGLON = 80


def _a_columna(texto: str) -> str:
    """Como llega del PDF: a 80 columnas, la palabra partida donde acaba el renglón."""
    plano = " ".join(texto.split())
    fuera = []
    while plano:
        fuera.append(plano[:RENGLON])
        plano = plano[RENGLON:]
    return "\n".join(x.ljust(RENGLON) if len(x) < RENGLON else x for x in fuera)


CABECERA = ("TOCA CIVIL 601/2023. QUEJOSO: JUAN PÉREZ. Sentencia de diecinueve de agosto de "
            "dos mil veinticuatro dictada en el juicio 1414/2020. " + "Datos de trámite. " * 60)
C1 = ("PRIMER CONCEPTO DE VIOLACIÓN. La responsable tuvo por acreditada la identidad sin "
      "que exista en autos la prueba pericial en materia de topografía, que es la prueba "
      "idónea para acreditar la identidad de un bien en la acción reivindicatoria. "
      "Las superficies son contradictorias entre sí: se habla de cuatro hectáreas en la "
      "reconvención y de dos hectáreas y media en la demanda. "
      "Los testigos de la contraria, Pedro Gómez Ruiz y Luis Ávila Soto, contestaron con "
      "palabras idénticas, lo que revela que fueron aleccionados, y ninguno dio la razón de su "
      "dicho ni las circunstancias de tiempo, modo y lugar. ")
C2 = ("SEGUNDO CONCEPTO DE VIOLACIÓN. Lo resuelto en el diverso expediente 1114/2017 "
      "produce cosa juzgada refleja, porque en aquel juicio se desestimó la misma acción "
      "reivindicatoria por falta de identidad del predio. La Sala debió analizar de oficio "
      "la cosa juzgada refleja y declarar improcedente la acción reivindicatoria. ")
C3 = ("TERCER CONCEPTO DE VIOLACIÓN. La condena en costas es ilegal, pues el suscrito "
      "actuó sin dolo ni mala fe en ninguna de las instancias; soy campesino, no sé leer ni "
      "escribir y mi condición es vulnerable. Por lo expuesto, pido se sirva conceder. ")
LIMPIO = CABECERA + C1 + C2 + C3
ESCRITO = _a_columna(LIMPIO)


def _en_crudo(p: int) -> int:
    return p + p // RENGLON


_I = [LIMPIO.find(x) for x in ("PRIMER CONCEPTO", "SEGUNDO CONCEPTO", "TERCER CONCEPTO")]
CONTEO = {"estado": "contado", "n": 3, "valores": [1, 2, 3],
          "tramos": [[_en_crudo(_I[0]), _en_crudo(_I[1])], [_en_crudo(_I[1]), _en_crudo(_I[2])],
                     [_en_crudo(_I[2]), len(ESCRITO)]]}


def fases():
    return f123.Fases123(resumen_conceptos=RESUMEN, conteo=copy.deepcopy(CONTEO),
                         fuentes=["acto", ESCRITO])


PISO = inv.segmentos(fases(), ESCRITO)
IDS_PISO = [s["id"] for s in PISO]
CITA_TESTIGOS = ("Los testigos de la contraria, Pedro Gómez Ruiz y Luis Ávila Soto, contestaron "
                 "con palabras idénticas, lo que revela que fueron aleccionados")
CITA_COSTAS = "La condena en costas es ilegal, pues el suscrito actuó sin dolo ni mala fe"


def _arg(concepto, texto, cita, dato="", en=None):
    return {"concepto": concepto, "argumento": texto, "dato": dato, "cita": cita,
            "en_inventario": en or []}


# Lo que devolvería el modelo: una que el piso ya recoge (y lo dice bien), la
# testimonial que falta, la costas reclamada con palabras que no nombran el
# dato, una parafraseada, una que no está, una corta y una de otro concepto
# que dice estar en un renglón de otro concepto.
CRUDOS = [
    _arg(1, "La responsable tiene por acreditada la identidad sin la pericial en topografía.",
         "tuvo por acreditada la identidad sin que exista en autos la prueba pericial en materia de topografía",
         "prueba pericial en topografía",
         [{"id": "C1.a", "palabras": "sin la prueba pericial en topografía"}]),
    _arg(1, "Los testigos de la contraria contestaron con palabras idénticas y fueron aleccionados, sin razón de su dicho.",
         CITA_TESTIGOS, "Pedro Gómez Ruiz y Luis Ávila Soto; aleccionamiento",
         [{"id": "C1.a", "palabras": "la Sala responsable"}]),
    _arg(1, "Otra razón que el modelo parafraseó en vez de copiar del escrito.",
         "Los testigos de la contraria, Pedro Gómez Ruiz y Luis, respondieron de un modo tan "
         "sospechoso que hace pensar que alguien les dijo qué responder en la audiencia y por eso "
         "no merecen ningún crédito del tribunal de alzada"),
    _arg(2, "Una razón con una cita que no está en el escrito de ninguna manera.",
         "el tribunal colegiado debe revocar la sentencia por violar el debido proceso legal del quejoso"),
    _arg(2, "Una razón con una cita demasiado corta para verificarla.", "cosa juzgada refleja"),
    _arg(3, "La condena en costas es ilegal porque actuó sin dolo ni mala fe.",
         CITA_COSTAS, "sin dolo ni mala fe", [{"id": "C1.b", "palabras": "las superficies"}]),
]

# ═══════════════════════════════════════════════════════════════════════════
print("\n1 · LA VERIFICACIÓN DE LAS CITAS (sin modelo)")
lec = ie.Lectura(ESCRITO)
a, b, n, tot = lec.localizar(CITA_TESTIGOS.upper().replace("Á", "A"))
ok(a >= 0 and n == tot and lec.pasaje(a, b).startswith("Los testigos de la contraria"),
   "la cita se encuentra entera aunque el PDF parta palabras a fin de renglón y cambien mayúsculas "
   "y acentos, y el pasaje sale tal como está en el escrito")
VER, DESC = ie.verificar(CRUDOS, ESCRITO, PISO, CONTEO)
por_texto = {v["texto"][:20]: v for v in VER}
ok(len(VER) == 3, f"de seis argumentos quedan los tres con cita verificable ({len(VER)})")
ok(DESC.get("cita parafraseada") == 1 and DESC.get("cita no verificada") == 2,
   f"la parafraseada y la que no está (o es corta) se descartan y se cuentan ({dict(DESC)})")
_v_a = next(v for v in VER if v["texto"].startswith("La responsable tiene"))
ok(_v_a["en_inventario"] == ["C1.a"],
   "el «ya está» que copia del renglón las palabras del dato (pericial en topografía) vale")
_v_t = next(v for v in VER if v["texto"].startswith("Los testigos"))
ok(_v_t["en_inventario"] == [] and _v_t["en_inventario_modelo"],
   "el «ya está» con palabras que no nombran el dato NO vale: la testimonial queda como nueva "
   "(y lo que dijo el modelo se guarda para medir)")
ok(any(k.startswith("«ya está» en otro concepto") for k in DESC),
   "el «ya está» en un renglón de OTRO concepto se rechaza")
ok(all(ie.CITA_MIN <= len(ie.Lectura.palabras(v["cita"])) <= ie.CITA_MAX for v in VER)
   and all(inv._plano(inv._colapsar(v["cita"])) in inv._plano(inv.normalizar_escrito(ESCRITO)[0]) for v in VER),
   "toda cita que queda es literal del escrito y tiene de 8 a 40 palabras")
_largo = " ".join(C1.split()[:60])
_v_l, _ = ie.verificar([_arg(1, "Una razón con una cita larguísima del primer concepto.", _largo)],
                       ESCRITO, PISO, CONTEO)
ok(len(_v_l) == 1 and len(ie.Lectura.palabras(_v_l[0]["cita"])) == ie.CITA_MAX,
   "la cita de más de 40 palabras se recorta a 40 (sigue siendo literal)")
_v_c, _d_c = ie.verificar([_arg(9, "Una razón que el modelo puso en un concepto que no existe.", CITA_TESTIGOS)],
                          ESCRITO, PISO, CONTEO)
ok(len(_v_c) == 1 and _v_c[0]["concepto"] == 1,
   "el concepto que no existe se corrige por el tramo del contador donde está la cita")
_v_s, _d_s = ie.verificar([_arg(9, "Una razón que el modelo puso en un concepto que no existe.", CITA_TESTIGOS)],
                          ESCRITO, PISO, {})
ok(not _v_s and _d_s.get("concepto fuera del escrito") == 1,
   "sin contador que lo corrija, el argumento de un concepto inexistente se descarta")
ok(ie.leer_json("no es json") == [] and ie.leer_json('{"argumentos": "x"}') == []
   and len(ie.leer_json('texto antes {"argumentos": [{"a": 1}, 3]} y después')) == 1,
   "la respuesta ilegible da lista vacía; el JSON rodeado de texto se rescata")
_leg = ie.texto_legible(ESCRITO)
ok(" ".join(_leg.split()) == inv.normalizar_escrito(ESCRITO)[0],
   "lo que lee el modelo son las mismas palabras que verifica el código (sólo cambian saltos)")
ok(ie._inv_id("(c1.A)") == "C1.a" and ie._inv_id("C 12 ab") == "C12.ab" and ie._inv_id("x") == "",
   "los ids del modelo se normalizan («c1.A» → «C1.a»)")

# ═══════════════════════════════════════════════════════════════════════════
print("\n2 · LA FUSIÓN: EL PISO INTACTO Y LOS NUEVOS CON LA SERIE")
SEGS, INF = ie.fusionar(PISO, VER, ESCRITO)
_piso_en = [s for s in SEGS if s.get("origen") != "escrito"]
ok(_piso_en == PISO, "los segmentos del piso salen idénticos y en su orden (mismos ids: marcas, "
   "plan y mapa no se rompen)")
_nuevos = [s for s in SEGS if s.get("origen") == "escrito"]
ok([s["id"] for s in _nuevos] == ["C1.c"] and INF["nuevos"] == 1 and INF["por_modelo"] == 1,
   f"la testimonial entra como C1.c, la que sigue la serie del concepto ({[s['id'] for s in _nuevos]}, {INF})")
ok([s["id"] for s in SEGS] == ["C1.a", "C1.b", "C1.c", "C2.a", "C3.a"],
   "el nuevo va detrás del último segmento de su concepto")
ok(INF["por_pasaje"] == 1,
   "la de costas (el modelo no probó que el renglón la recoja) no se duplica: su cita pisa la del piso")
_n = _nuevos[0]
ok(_n["cita"] and _n["concepto"] == 1 and "Pedro Gómez Ruiz" in _n["cita"]
   and set(_n) >= {"id", "concepto", "parrafo", "texto", "cita", "anclas", "pagina", "parecido",
                   "concepto_inferido"},
   "el nuevo cumple el contrato del segmento (cita literal, concepto, anclas…)")
_dup = copy.deepcopy(_v_t)
_dup["texto"] = "La misma testimonial, leída dos veces por el modelo."
_S2, _I2 = ie.fusionar(PISO, VER + [_dup], ESCRITO)
ok(len([s for s in _S2 if s.get("origen") == "escrito"]) == 1 and _I2["repetidos"] == 1,
   "dos leídos con el mismo pasaje son uno")
_v4, _ = ie.verificar([_arg(4, "Un concepto que el resumen no trae y el contador tampoco.", CITA_TESTIGOS)],
                      ESCRITO, PISO, {"estado": "contado", "n": 4})
_S4, _ = ie.fusionar(PISO, _v4, ESCRITO)
ok([s["id"] for s in _S4][-1] == "C4.a", "un concepto que el piso no tiene empieza su serie en «a», en su lugar")
_muchos = []
for k, frase in enumerate(re.split(r"(?<=\.) ", C1 + C2 + C3)):
    ws = frase.split()
    if len(ws) >= 10:
        _muchos.append({"concepto": 1, "texto": f"Razón número {k} leída del escrito.", "dato": "",
                        "cita": " ".join(ws[:10]), "en_inventario": []})
_vm, _ = ie.verificar(_muchos, ESCRITO, PISO, CONTEO)
_Sm, _Im = ie.fusionar(PISO, _vm, ESCRITO)
ok(len([s for s in _Sm if s.get("origen") == "escrito"]) <= min(ie.MAX_NUEVOS, len(PISO))
   and (_Im["sobrantes"] > 0 or len(_vm) <= len(PISO)),
   f"EL BLOQUE NO SE DISPARA: como mucho {ie.MAX_NUEVOS} nuevos y nunca más que el piso ({_Im})")
# EL TOPE SE REPARTE ENTRE CONCEPTOS (revisión adversarial): antes se cortaba
# en el orden del escrito y un primer concepto largo dejaba fuera al último.
_vr, _ = ie.verificar([
    _arg(1, "Los testigos fueron aleccionados y contestaron con palabras idénticas.", CITA_TESTIGOS),
    _arg(1, "Ningún testigo dio la razón de su dicho ni las circunstancias.",
         "ninguno dio la razón de su dicho ni las circunstancias de tiempo, modo y lugar"),
    _arg(3, "Es campesino, no sabe leer ni escribir y su condición es vulnerable.",
         "soy campesino, no sé leer ni escribir y mi condición es vulnerable")], ESCRITO, PISO, CONTEO)
_max0 = ie.MAX_NUEVOS
try:
    ie.MAX_NUEVOS = 2
    _Sr, _Ir = ie.fusionar(PISO, _vr, ESCRITO)
finally:
    ie.MAX_NUEVOS = _max0
ok([s["id"] for s in _Sr if s.get("origen") == "escrito"] == ["C1.c", "C3.b"] and _Ir["sobrantes"] == 1
   and [s for s in _Sr if s.get("origen") != "escrito"] == PISO,
   f"con tope 2 y tres leídos (dos del primer concepto, uno del tercero) entra uno de cada concepto ({_Ir})")
ok(inv.segmentos(fases(), ESCRITO, extraidos=None) == PISO
   and inv.segmentos(fases(), ESCRITO, extraidos=[]) == PISO,
   "`inventario.segmentos` sin lectura (None o []) devuelve el piso de siempre")
ok([s["id"] for s in inv.segmentos(fases(), ESCRITO, extraidos=VER)] == [s["id"] for s in SEGS],
   "con lectura, `inventario.segmentos` devuelve la fusión")
_Sx, _Ix = ie.fusionar(PISO, [{"concepto": "x", "palabras": "roto"}], ESCRITO)
ok(_Sx == PISO and _Ix.get("error"), "una lectura con basura no rompe nada: queda el piso")
_bl0 = inv.bloque_inventario(PISO, "concepto de violación")
_bl1 = inv.bloque_inventario(SEGS, "concepto de violación")
ok("leído del escrito" not in _bl0 and _bl1.count("leído del escrito") == 2,
   "el bloque marca los renglones leídos del escrito (cabecera y renglón); sin ellos, no dice nada")
_ps = pe.segmentos_de(fases(), ESCRITO, False, extraidos=VER)
ok([s["id"] for s in _ps] == [s["id"] for s in SEGS] and _ps[2].get("origen") == "escrito"
   and "origen" not in _ps[0],
   "el plan recibe la MISMA lista (la clave lleva los segmentos) y sabe cuál no es del resumen")
ok("leído del escrito: " in pe._bloque_segmentos(_ps, []) and "resumen: " in pe._bloque_segmentos(_ps, []),
   "el prompt del plan dice de dónde sale lo que se alega en cada renglón")

# ═══════════════════════════════════════════════════════════════════════════
print("\n3 · LA LLAMADA ENTERA Y LA CAÍDA AL PISO (modelo de mentira)")
_LLAMADAS = []


async def _llamar_bien(cliente, texto):
    _LLAMADAS.append(texto)
    return json.dumps({"argumentos": CRUDOS}, ensure_ascii=False), {"entrada": 1000, "salida": 500,
                                                                    "razonamiento": 300, "llamadas": 1}


async def _llamar_roto(cliente, texto):
    raise RuntimeError("el proveedor no contesta")


async def _llamar_basura(cliente, texto):
    return "lo siento, no puedo", {"entrada": 1, "salida": 1, "llamadas": 1}


_orig_llamar = ie._llamar
try:
    ie._llamar = _llamar_bien
    SAL = asyncio.run(ie.extraer(None, fases()))
    ie._llamar = _llamar_roto
    SAL_R = asyncio.run(ie.extraer(None, fases()))
    ie._llamar = _llamar_basura
    SAL_B = asyncio.run(ie.extraer(None, fases()))
    SAL_V = asyncio.run(ie.extraer(None, f123.Fases123(resumen_conceptos="", fuentes=["", ""])))
finally:
    ie._llamar = _orig_llamar
ok(SAL["estado"] == "listo" and len(SAL["argumentos"]) == 3 and SAL["coste"] > 0
   and SAL["huella"] == ie.huella(fases(), ESCRITO, False),
   "la lectura buena: verificada, con su uso, su coste y la huella del adelanto")
ok(SAL_R["estado"] == "error" and not SAL_R["argumentos"] and SAL_R["avisos"],
   "el proveedor que falla: «error» sin argumentos (se puede reintentar) y con aviso")
ok(SAL_B["estado"] == "error" and not SAL_B["argumentos"],
   "la respuesta sin JSON: «error», no una lista vacía que pase por «el escrito no trae nada»")
ok(SAL_V["estado"] == "vacio", "sin escrito ni resumen no se llama al modelo: «vacío», queda el piso")
_p = _LLAMADAS[0]
ok("EL ESCRITO:" in _p and "Pedro Gómez Ruiz" in _p and "C1.a · concepto de violación 1 ·" in _p,
   "el modelo lee el escrito entero y el piso con sus ids")
_src_ll = open("inventario_escrito.py", encoding="utf-8").read()
ok("temperature" not in _src_ll.split("async def _llamar")[1].split("def leer_json")[0].split('"""')[2]
   and 'kw["reasoning_effort"] = ESFUERZO' in _src_ll and ie.ESFUERZO in ("medium", "high", "xhigh"),
   f"la llamada: razonamiento {ie.ESFUERZO} (nunca «none») y sin temperatura, que el modelo rechaza")

# ═══════════════════════════════════════════════════════════════════════════
print("\n4 · LA CACHÉ POR HUELLA EN LA FILA (funciones puras)")
HU, H = "ADELANTO1", ie.huella(fases(), ESCRITO, False)
T0 = 1_000_000.0
d1, dec1 = ie.fila_pedir(None, HU, H, T0)
ok(dec1 == "lanzar" and d1["inventario"]["estado"] == "en_curso" and d1["huella"] == HU,
   "sin nada guardado: se reserva la corrida («lanzar»)")
_, dec2 = ie.fila_pedir(d1, HU, H, T0 + 5)
ok(dec2 == "en_curso", "otra petición mientras corre: se espera, no se duplica")
d3, est3 = ie.fila_resultado(d1, HU, H, SAL, T0 + 100)
ok(est3 == "listo" and ie.guardado(d3, HU, H, T0 + 200) == ("listo", SAL["argumentos"]),
   "la lectura se guarda y se lee tal cual")
_, dec4 = ie.fila_pedir(d3, HU, H, T0 + 300)
ok(dec4 == "listo", "con la lectura guardada no se vuelve a llamar al modelo")
ok(ie.guardado(d3, HU, "OTRA", T0)[0] == "ninguno" and ie.fila_pedir(d3, HU, "OTRA", T0)[1] == "lanzar",
   "otra huella de la lectura (otro piso, otra versión) no reutiliza la guardada")
ok(ie.guardado(d3, "OTRO_ADELANTO", H, T0)[0] == "ninguno"
   and ie.fila_resultado(d3, "OTRO_ADELANTO", H, SAL, T0)[1] == "otro_adelanto",
   "lo leído de otro adelanto ni se usa ni se escribe encima")
_ab = ie.fila_pedir(d1, HU, H, T0 + ie.ABANDONADA_S + 1)
ok(ie.guardado(d1, HU, H, T0 + ie.ABANDONADA_S + 1)[0] == "fallo" and _ab[1] == "lanzar"
   and _ab[0]["inventario"]["corridas"] == 2,
   "la corrida sin latido se da por abandonada y se relanza (cuenta)")
dE, _ = ie.fila_resultado(d1, HU, H, SAL_R, T0 + 10)
_dE2, _decE = ie.fila_pedir(dE, HU, H, T0 + 11)
ok(ie.guardado(dE, HU, H, T0 + 11)[0] == "fallo" and _decE == "lanzar",
   "un error del proveedor se reintenta en la petición siguiente")
_dT = copy.deepcopy(d1)
_dT["inventario"].update(estado="error", corridas=ie.TOPE_CORRIDAS)
ok(ie.fila_pedir(_dT, HU, H, T0)[1] == "tope", f"con {ie.TOPE_CORRIDAS} corridas gastadas, tope: queda el piso")
# El plan y la lectura viven en el mismo documento y no se pisan.
_dp, _ = pe.fila_pedir(d3, "CLAVE", HU, T0)
ok(_dp["inventario"] == d3["inventario"], "pedir un plan conserva la lectura guardada")
_dp1, _ = pe.fila_pedir(d1, "CLAVE", HU, T0)
_dp2, _ = pe.fila_resultado(_dp1, "CLAVE", HU, {"segmentos": []}, [], 1.0, T0)
_dl, _ = ie.fila_resultado(_dp2, HU, H, SAL, T0 + 1)
ok(_dl["planes"] == _dp2["planes"] and _dl["inventario"]["estado"] == "listo",
   "guardar la lectura conserva los planes")
_dn, _ = pe.fila_pedir(d3, "CLAVE", "ADELANTO_NUEVO", T0)
ok("inventario" not in _dn, "un adelanto nuevo rehace el documento: la lectura vieja no sobrevive")

# DOS WORKERS: la corrida que otro dio por abandonada y relevó, y que termina
# tarde (revisión adversarial, 26-sep-2026). LO LEÍDO NO SE REESCRIBE: los ids
# que añadió la primera lectura buena pueden estar ya en un plan y un estudio.
_w1, _ = ie.fila_pedir(None, HU, H, T0)                               # corrida 1
_w2, _dw2 = ie.fila_pedir(_w1, HU, H, T0 + ie.ABANDONADA_S + 1)        # la releva la 2
ok(_dw2 == "lanzar" and _w2["inventario"]["corridas"] == 2, "la corrida 1 sin latido la releva la 2")
ok(ie.fila_latido(_w2, HU, H, T0 + 70, corrida=1)[1] == "nada"
   and ie.fila_latido(_w2, HU, H, T0 + 70, corrida=2)[1] == "latido",
   "la relevada no late por la que la relevó (si ésta muere, tiene que poder darse por abandonada)")
ok(ie.fila_resultado(_w2, HU, H, SAL_R, T0 + 80, corrida=1) == (None, "relevada"),
   "un error tardío de la relevada no tumba a la que la relevó mientras ésta siga viva")
_w2m = copy.deepcopy(_w2)
_w2m["inventario"]["latido"] = T0                                      # la 2 murió también
ok(ie.fila_resultado(_w2m, HU, H, SAL_R, T0 + 200, corrida=1)[1] == "error",
   "si la que la relevó también murió, el error se guarda (y la petición siguiente relanza)")
_w3, _ = ie.fila_resultado(_w2, HU, H, SAL, T0 + 90, corrida=2)
_OTRA = dict(SAL, argumentos=[dict(SAL["argumentos"][0], texto="Otra lista, otros ids.")])
ok(ie.fila_resultado(_w3, HU, H, SAL_R, T0 + 95, corrida=1) == (None, "listo")
   and ie.fila_resultado(_w3, HU, H, _OTRA, T0 + 95, corrida=1) == (None, "listo")
   and ie.fila_resultado(_w3, HU, H, _OTRA, T0 + 95) == (None, "listo")
   and ie.guardado(_w3, HU, H, T0 + 96) == ("listo", SAL["argumentos"]),
   "LA PRIMERA LECTURA BUENA NO SE REESCRIBE: ni con un error tardío ni con otra lista "
   "(el mismo id nombraría otro argumento)")
_w1l, _ = ie.fila_resultado(_w2, HU, H, _OTRA, T0 + 85, corrida=1)
ok(_w1l["inventario"]["estado"] == "listo"
   and ie.fila_resultado(_w1l, HU, H, SAL, T0 + 90, corrida=2) == (None, "listo"),
   "si la relevada termina bien primero, la suya es LA lectura y la de la 2 ya no entra")

# ═══════════════════════════════════════════════════════════════════════════
print("\n4b · EL FLUJO DE main.py CON LA BASE Y EL MODELO DE MENTIRA")
SRC_MAIN = open("main.py", encoding="utf-8").read()
ARBOL = ast.parse(SRC_MAIN)
FN = {x.name: x for x in ast.walk(ARBOL) if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))}


class _Res:
    def __init__(self, data):
        self.data = data


class _Base:
    def __init__(self, plan=None):
        self.filas = [{"email": "x@y.mx", "expediente": "1/2026", "plan": plan}]
        self.escrituras = 0

    def table(self, _):
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
            if (op == "eq" and v != val) or (op == "is" and v is not None):
                return False
        return True

    def execute(self):
        filas = [f for f in self.b.filas if self._pasa(f)]
        if self.modo == "update":
            self.b.escrituras += len(filas)
            for f in filas:
                f.update(copy.deepcopy(self.datos))
        return _Res([copy.deepcopy(f) for f in filas])


_NOMBRES = ["_taller_plan_cas", "_taller_plan_leer", "_taller_inv_escrito_adelantar",
            "_taller_inv_escrito_huellas", "_taller_inv_escrito_correr", "_taller_inv_escrito_lanzar",
            "_taller_preinventariar", "_taller_preinventariar_suelta", "_taller_inv_escrito_para",
            "_taller_inventario_al_encargo", "_taller_inv_escrito_corrida"]
ok(all(x in FN for x in _NOMBRES), "main.py trae las piezas del flujo")


def entorno(base, casa=True):
    ns = {"time": time, "asyncio": asyncio, "print": lambda *a, **k: None, "err": str,
          "supabase_admin": base, "chat_client": None, "_TALLER_EN_MARCHA": set(),
          "_te": types.SimpleNamespace(huella_contraste=lambda r: HU),
          "_taller_es_casa": lambda c: casa}
    exec(compile(ast.Module(body=[FN[x] for x in _NOMBRES], type_ignores=[]), "main.py", "exec"), ns)
    return ns


def resultado(variante="v3"):
    return types.SimpleNamespace(
        fases=fases(),
        encargo=types.SimpleNamespace(es_recurso=False, tipo_asunto="amparo_directo",
                                      variante_estudio=variante, inventario_escrito=["DE LA VUELTA ANTERIOR"]))


_LLAMADAS.clear()
try:
    ie._llamar = _llamar_bien
    # (a) La precalculada al terminar el adelanto, y luego el resolver la usa sin llamar.
    B = _Base()
    NS = entorno(B)
    asyncio.run(NS["_taller_preinventariar"]("x@y.mx", "1/2026", resultado()))
    ok(len(_LLAMADAS) == 1 and B.filas[0]["plan"]["inventario"]["estado"] == "listo",
       "al terminar el adelanto se lee el escrito UNA vez y se guarda en la fila")
    r3 = resultado("v3")
    asyncio.run(NS["_taller_inventario_al_encargo"]("x@y.mx", "1/2026", r3))
    ok(len(_LLAMADAS) == 1 and len(r3.encargo.inventario_escrito) == 3,
       "al generar (v3) se usa la guardada, sin volver a llamar al modelo")
    asyncio.run(NS["_taller_preinventariar"]("x@y.mx", "1/2026", resultado()))
    ok(len(_LLAMADAS) == 1, "otro disparo (la consulta) no la repite: ya está")
    # (b) La v1: ni la ve, ni espera, ni llama; y la de la vuelta anterior se vacía.
    r1 = resultado("v1")
    _esperas = []
    asyncio.run(NS["_taller_inventario_al_encargo"]("x@y.mx", "1/2026", r1,
                                                    al_esperar=lambda: _esperas.append(1)))
    ok(r1.encargo.inventario_escrito == [] and not _esperas and len(_LLAMADAS) == 1,
       "v1: la lectura de la vuelta anterior se vacía y no se espera ni se llama a nada")
    # (c) Sin nada guardado, el resolver la calcula dentro de su tarea.
    B2 = _Base()
    NS2 = entorno(B2)
    r4 = resultado("v4")
    _esperas = []
    asyncio.run(NS2["_taller_inventario_al_encargo"]("x@y.mx", "1/2026", r4,
                                                     al_esperar=lambda: _esperas.append(1)))
    ok(len(_LLAMADAS) == 2 and len(r4.encargo.inventario_escrito) == 3 and _esperas
       and B2.filas[0]["plan"]["inventario"]["estado"] == "listo",
       "v4 sin lectura guardada: se calcula dentro de la tarea (la pantalla ve «ordenando») y se guarda")
    # (d) El cliente de fuera de casa no la precalcula.
    B3 = _Base()
    asyncio.run(entorno(B3, casa=False)["_taller_preinventariar"]("x@y.mx", "1/2026", resultado()))
    ok(len(_LLAMADAS) == 2 and B3.escrituras == 0,
       "fuera de casa (y sin v3/v4 global) no se precalcula: nadie paga una lectura que no se usará")
    # (e) El proveedor falla: queda el piso y se registra.
    ie._llamar = _llamar_roto
    B4 = _Base()
    NS4 = entorno(B4)
    _reg = []
    NS4["print"] = lambda *a, **k: _reg.append(" ".join(str(x) for x in a))
    r5 = resultado("v3")
    asyncio.run(NS4["_taller_inventario_al_encargo"]("x@y.mx", "1/2026", r5))
    ok(r5.encargo.inventario_escrito == [] and any("queda el piso" in x for x in _reg),
       "si la lectura falla, el estudio va con el piso y se registra")
    ok(not any("Pedro" in x or "testigos" in x for x in _reg),
       "HIGIENE DE REGISTROS: al registro van números, ni citas ni lo que se alega")

    # (f) En curso en otro worker y no llega a tiempo: el piso, sin lanzar otra.
    async def _colgada(cliente, texto):
        await asyncio.sleep(3600)

    ie._llamar = _colgada
    _dc, _ = ie.fila_pedir(None, HU, H, time.time())
    B5 = _Base(_dc)
    NS5 = entorno(B5)
    t0 = time.time()
    args5, est5 = asyncio.run(NS5["_taller_inv_escrito_para"]("x@y.mx", "1/2026", resultado(), 0.3))
    ok(args5 is None and "no llegó" in est5 and time.time() - t0 < 5,
       f"la que corre en otro worker y no llega en el tope: el piso ({est5})")

    # (g) Se lanza aquí y vence: el piso, y la corrida sigue (no se cancela).
    async def _vence():
        B6 = _Base()
        NS6 = entorno(B6)
        out = await NS6["_taller_inv_escrito_para"]("x@y.mx", "1/2026", resultado(), 0.2)
        vivas = [t for t in NS6["_TALLER_EN_MARCHA"] if not t.done()]
        for t in vivas:
            t.cancel()
        return out, len(vivas)

    (args6, est6), vivas6 = asyncio.run(_vence())
    ok(args6 is None and "no llegó" in est6 and vivas6 == 1,
       "la lanzada aquí que no llega: el piso para este estudio, y la corrida SIGUE para la próxima")

    # (h) DOS WORKERS: ésta relanza una corrida que dio por abandonada, y la
    # relevada termina bien antes. Se usa LA DE LA FILA (la primera buena), no
    # la propia: este estudio y el siguiente ven los mismos ids.
    _OTRA_ARGS = [dict(SAL["argumentos"][0], texto="Lo que leyó la corrida relevada.")]
    B7 = _Base(ie.fila_pedir(None, HU, H, T0)[0])          # reservada hace mucho: abandonada

    async def _llamar_y_la_otra_acaba(cliente, texto):
        fila = B7.filas[0]
        doc = copy.deepcopy(fila["plan"])
        doc["inventario"] = {"huella": H, "version": ie.VERSION, "estado": "listo",
                             "argumentos": _OTRA_ARGS, "corridas": 1, "hecho": time.time()}
        doc["rev"] = int(doc.get("rev") or 0) + 1
        fila["plan"] = doc
        return await _llamar_bien(cliente, texto)

    ie._llamar = _llamar_y_la_otra_acaba
    NS7 = entorno(B7)
    args7, est7 = asyncio.run(NS7["_taller_inv_escrito_para"]("x@y.mx", "1/2026", resultado(), 30))
    ok(est7 == "listo" and args7 == _OTRA_ARGS
       and B7.filas[0]["plan"]["inventario"]["argumentos"] == _OTRA_ARGS,
       "la que relanza usa la lectura de la fila (la primera buena) y no la pisa con la suya")
finally:
    ie._llamar = _orig_llamar
_ap = ast.get_source_segment(SRC_MAIN, FN["_taller_inventario_al_encargo"])
ok("ESPERA_RESOLVER_S" in _ap and ie.ESPERA_RESOLVER_S <= 180,
   f"el resolver espera la lectura como mucho {ie.ESPERA_RESOLVER_S:.0f} s")
ok(ie.ABANDONADA_S < ie.ESPERA_RESOLVER_S and ie.ABANDONADA_S >= 2 * ie.LATIDO_S,
   "una corrida muerta se da por abandonada antes de que venza la espera del resolver")

# ═══════════════════════════════════════════════════════════════════════════
print("\n5 · LOS GEMELOS Y LAS PUERTAS (AST)")
for nombre in ("taller_resolver_stream", "taller_resolver"):
    src = ast.get_source_segment(SRC_MAIN, FN[nombre])
    i_vacia = src.find("r.encargo.inventario_escrito = []")
    i_fija = src.find("await _taller_inventario_al_encargo(")
    i_plan = src.find("await _taller_plan_para(")
    ok(0 < i_vacia < i_fija < i_plan,
       f"{nombre}: vacía la lectura del encargo, la fija dentro de la tarea y ANTES del plan")
_st = ast.get_source_segment(SRC_MAIN, FN["taller_resolver_stream"])
ok('al_esperar=lambda: _cola.put_nowait({"tipo": "ordenando"})' in _st,
   "el gemelo de flujo rotula la espera («ordenando»)")
_pp = ast.get_source_segment(SRC_MAIN, FN["_taller_plan_para"])
ok('extraidos=list(getattr(e, "inventario_escrito", None) or [])' in _pp,
   "el plan del resolver usa la MISMA lectura que el estudio (la del encargo de esta petición)")
_pe_src = ast.get_source_segment(SRC_MAIN, FN["_taller_plan_entradas"])
ok("extraidos=list(extraidos or [])" in _pe_src and '"inventario_escrito"' not in _pe_src,
   "las entradas del plan reciben la lectura explícita; nunca la leen del encargo en memoria")
for nombre in ("taller_plan_pedir", "_taller_plan_desde_propuesta"):
    src = ast.get_source_segment(SRC_MAIN, FN[nombre])
    ok("_taller_inv_escrito_para(" in src and "extraidos=" in src,
       f"{nombre}: pide el plan con la lectura que verá el estudio")
    ok(0 < src.find('startswith("no llegó")') < src.find("await _taller_plan_pedido("),
       f"{nombre}: con la lectura aún en curso no pide un plan con el piso (no casaría con el "
       f"del resolver y gastaría una corrida)")
for nombre in ("taller_adelanto", "taller_consultar"):
    ok("_taller_preinventariar_suelta(" in ast.get_source_segment(SRC_MAIN, FN[nombre]),
       f"{nombre}: dispara la lectura en segundo plano")
SRC_RA = open("redactor_adelanto.py", encoding="utf-8").read()
ok('extraidos=list(getattr(e, "inventario_escrito", None) or []) if e else []' in SRC_RA,
   "el material del estudio funde la lectura que fijó el resolver en el encargo")
import redactor_adelanto as ra  # noqa: E402
ok(any(f.name == "inventario_escrito" for f in __import__("dataclasses").fields(ra.Encargo)),
   "el encargo trae el campo (vacío por omisión)")

# ═══════════════════════════════════════════════════════════════════════════
print("\n6 · LA v1 INTACTA; SIN LECTURA, LA v3 COMO ANTES")
C93 = [f6.Criterio(problema="¿Se acreditó la identidad del inmueble?", sentido="infundado",
                   razonamiento="sí se acreditó con las constancias", jerarquia="principal")]


def _material(variante, extraidos):
    r = types.SimpleNamespace(fases=fases(), avisos=[])
    r.encargo = types.SimpleNamespace(formato="", variante_estudio=variante, tipo_asunto="amparo_directo",
                                      es_recurso=False, inventario_escrito=extraidos)
    m = f6.Material()
    ra._formato_al_material(r, m, None, C93)
    return m


_m1a, _m1b = _material("v1", []), _material("v1", VER)
ok(_m1a.inventario == [] and _m1b.inventario == [], "v1: el material no lleva inventario, haya lectura o no")
ok(f6.prompt_estudio("acto", RESUMEN, C93, _m1b) == f6.prompt_estudio("acto", RESUMEN, C93, _m1a),
   "v1: el prompt es EL MISMO con y sin lectura (ni una coma)")
_m3a, _m3b = _material("v3", []), _material("v3", VER)
ok(_m3a.inventario == PISO and [s["id"] for s in _m3b.inventario] == [s["id"] for s in SEGS],
   "v3: sin lectura, el piso de siempre; con lectura, la fusión")
_p3a = f6.prompt_estudio("acto", RESUMEN, C93, _m3a)
_p3b = f6.prompt_estudio("acto", RESUMEN, C93, _m3b)
ok("leído del escrito" not in _p3a and "C1.c" in _p3b and "leído del escrito" in _p3b,
   "v3: sin lectura, el bloque no cambia; con ella, trae los renglones nuevos marcados")
_txt = "uno dos\n  tres\nCUA-\ntro cinco\n\nseis"
ok(inv.normalizar_escrito(_txt) == inv.normalizar_escrito(_txt, saltos=False)
   and inv.normalizar_escrito(ESCRITO)[0] == " ".join(inv.normalizar_escrito(ESCRITO, saltos=True)[0].split()),
   "`normalizar_escrito` sin la bandera sale como siempre (el piso y V0 no cambian)")
os.environ["INVENTARIO_ESCRITO"] = "off"
ok(ie.interruptor() == "off", "INVENTARIO_ESCRITO=off apaga la lectura (vuelve el piso)")
os.environ.pop("INVENTARIO_ESCRITO")
ok(ie.interruptor() == "casa" and ie.aplica_variante("v3") and ie.aplica_variante("V4")
   and not ie.aplica_variante("v1") and not ie.aplica_variante("v2"),
   "por omisión, sólo casa, y al generar sólo la v3 y la v4")

# ═══════════════════════════════════════════════════════════════════════════
print("\n7 · EL PROMPT: DESCRIPCIONES, NI FRASES MODELO NI DATOS DEL BANCO")
_p_sin = ie.prompt(fases(), ESCRITO, PISO)
_instr = _p_sin[:_p_sin.find("EL INVENTARIO QUE YA EXISTE")]
# Lo que motivó esto (la testimonial aleccionada, el hijo, el oficio al RAN, las
# costas de alzada, el NIP…) no puede estar en las instrucciones: se copiaría.
_prohibidas = [(r"aleccion", re.I), (r"\bhijos?\b", re.I), (r"\bRAN\b", 0),
               (r"Registro Agrario", re.I), (r"164607", 0), (r"costas", re.I), (r"\bNIP\b", 0),
               (r"firma electr", re.I), (r"concubin", re.I), (r"Quer[eé]taro", re.I),
               (r"p\. ej", re.I), (r"\bejemplo", re.I)]
_hay = [w for w, fl in _prohibidas if re.search(w, _instr, fl)]
ok(not _hay, f"las instrucciones no traen ejemplos ni datos de los asuntos del banco ({_hay})")
ok(f"de {ie.CITA_MIN} a {ie.CITA_MAX} palabras SEGUIDAS" in _instr and '"en_inventario"' in _instr
   and "No decides si tienen razón" in _instr,
   "pide la cita literal de 8 a 40 palabras, el renglón que ya la recoge, y NO decidir sentidos")
_p_ag = ie.prompt(fases(), ESCRITO, PISO, es_recurso=True, tipo="amparo_revision")
ok("escrito de agravios" in _p_ag and "agravio" in _p_ag,
   "en un recurso habla de agravios")

print()
if FALLOS:
    print(f"RESULTADO: FALLAN {len(FALLOS)}")
    for f in FALLOS:
        print("   ·", f)
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
