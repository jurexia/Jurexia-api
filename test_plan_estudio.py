"""EL PLAN DEL ESTUDIO (Paso 2b + 3, 26-sep-2026) — sin llamar a ningún modelo.

Comprueba, con un asunto SINTÉTICO (nada de un expediente real: fechas,
números y nombres son de prueba) y dobles del modelo, del inventario y de la
base:

  1 · catálogos, normalización y verificación literal;
  2 · un plan bueno pasa V0 al primer intento y conserva el sentido;
  3 · los planes que V0 debe rechazar: etiqueta distinta al criterio,
      procesal no estudiada, reitera con ancla propia, cita inventada, premisa
      sin rastro, fuente fuera del índice, unidad que mezcla, suplencia,
      cosa juzgada, orden;
  4 · `reparar`: lo que el código corrige solo y lo que dice al hacerlo;
  5 · `preparar`: reintento con la lista, y sin plan si falla dos veces; el
      modelo de las fases con razonamiento MEDIO y JSON estricto;
  6 · el guion por forma (estándar y moderna): cada premisa se expone UNA vez,
      remisiones con destino, pendiente de razón, sin frases modelo;
  7 · Decisión 6: la razón del secretario entra al guion como suya;
  8 · la clave: la razón entra, la forma no, la suplencia sólo confirmada;
  9 · el estado en la fila (funciones puras): tope, en curso, abandono, huella;
 10 · fase6: v1 y v2 IDÉNTICAS a las de antes; la v4 = la v3 + el guion; sin
      guion, la v4 es la v3; los dos redactores pasan el guion;
 11 · los gemelos por AST: el mismo criterio, el mismo plan, el evento
      «ordenando» dentro de la tarea, `razones_segmento` en las tres puertas;
 12 · el criterio común: el grupo llega en los TRES modos;
 13 · el resolver de punta a punta con la base y el modelo de mentira: plan
      reutilizado, plan calculado, plan que vence y queda para la próxima,
      plan rechazado dos veces → v3 con aviso, y fuera de la v4 nada.

    .venv/bin/python test_plan_estudio.py
"""
import ast
import asyncio
import copy
import hashlib
import json
import os
import sys
import types

os.environ.pop("ESTUDIO_PROMPT", None)
os.environ.pop("PLAN_ESTUDIO", None)

import fase6_estudio as f6
import fases123_pipeline as f123
import plan_estudio as pe

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


AQUI = os.path.dirname(os.path.abspath(__file__))

# ═══ EL ASUNTO DE PRUEBA (sintético) ════════════════════════════════════════
ACTO = ("La Sala responsable consideró que la identidad del inmueble quedó acreditada con la "
        "confesión del demandado de poseer dentro de la parcela de la actora, por lo que no era "
        "necesaria la prueba pericial en materia de topografía, conforme al artículo 84 del "
        "Código de Procedimientos Civiles para el Estado. Asimismo, estimó que no existe cosa "
        "juzgada, pues así lo ordenó la ejecutoria del amparo directo 312/2023, que quedó firme. "
        "Por último, condenó en costas en ambas instancias.")
ESCRITO = (
    "CONCEPTOS DE VIOLACIÓN\n"
    "PRIMERO. La responsable violó en mi perjuicio el artículo 16 constitucional porque tuvo por "
    "acreditada la identidad del inmueble sin que se desahogara la prueba pericial en materia de "
    "topografía, que era la idónea para ello. Además, en la demanda se reclamó una superficie de "
    "2.5 hectáreas y la condena se refiere a una fracción de 4-80-54.113 hectáreas, lo que "
    "demuestra que no hay identidad entre lo reclamado y lo condenado.\n\n"
    "SEGUNDO. La Sala omitió examinar de oficio la cosa juzgada refleja que deriva de lo resuelto "
    "en el juicio 1114/2017, en el que se discutió la misma parcela entre las mismas partes.\n\n"
    "TERCERO. Insisto en que sin la pericial en topografía no puede tenerse por identificado el "
    "inmueble. También solicito que se aplique por analogía el criterio del juicio 1114/2017 y la "
    "tesis de registro 196451, que obligan a respetar lo resuelto en aquel asunto.")
_o = [ESCRITO.index(x) for x in ("PRIMERO.", "SEGUNDO.", "TERCERO.")] + [len(ESCRITO)]
CONTEO = {"estado": "contado", "n": 3, "tramos": [[_o[i], _o[i + 1]] for i in range(3)]}
PREG1 = "¿La identidad del inmueble quedó acreditada sin la prueba pericial en topografía?"
PREG2 = "¿La Sala debía examinar la cosa juzgada refleja del juicio 1114/2017?"
RAZON1 = ("La identidad quedó acreditada con la confesión del demandado de poseer dentro de la "
          "parcela de la actora; la diferencia de superficies mide la extensión de la condena y no "
          "la identidad del bien. Apoya el criterio de registro 2001111.")
RAZON2 = ("La sentencia reclamada se dictó en cumplimiento de la ejecutoria del amparo directo "
          "312/2023, que ordenó reexaminar los agravios partiendo de que no se actualizaba la cosa "
          "juzgada, y esa determinación quedó firme.")


def fases(**kw):
    f = f123.Fases123(resumen_acto="La Sala tuvo por acreditada la identidad y negó la cosa juzgada.",
                      resumen_conceptos="Primero: identidad sin pericial.\nSegundo: cosa juzgada "
                                        "refleja.\nTercero: reitera y pide analogía.",
                      problemas=[{"pregunta": PREG1, "cubre": [1, 3], "clase": "fondo",
                                  "jerarquia": "principal"},
                                 {"pregunta": PREG2, "cubre": [2], "clase": "fondo",
                                  "jerarquia": "accesorio", "depende_de": None}],
                      fuentes=[ACTO, ESCRITO], conteo=dict(CONTEO))
    for k, v in kw.items():
        setattr(f, k, v)
    return f


def crit(s1="infundado", s2="inoperante", g1="", g2="", r1=RAZON1, r2=RAZON2):
    return [f6.Criterio(problema=PREG1, sentido=s1, razonamiento=r1, jerarquia="principal", grupo=g1),
            f6.Criterio(problema=PREG2, sentido=s2, razonamiento=r2, jerarquia="accesorio", grupo=g2)]


def material(**kw):
    m = f6.Material(
        tesis=[{"registro": "2001111", "rubro": "IDENTIDAD DEL INMUEBLE. PUEDE ACREDITARSE CON LA "
                "CONFESIÓN DEL DEMANDADO.", "obligatoria": True,
                "texto": "La identidad del bien reclamado puede tenerse por demostrada con la "
                         "confesión del demandado de poseer el inmueble materia del juicio, sin que "
                         "sea indispensable la prueba pericial."},
               {"registro": "2002222", "rubro": "COSA JUZGADA. LO DECIDIDO EN UNA EJECUTORIA DE "
                "AMPARO VINCULA AL TRIBUNAL.", "obligatoria": True,
                "texto": "Lo decidido en una ejecutoria de amparo que quedó firme no puede "
                         "discutirse de nuevo en otro juicio de amparo."}],
        normas=[{"cuerpo_legal": "Código de Procedimientos Civiles para el Estado", "articulo": "84",
                 "texto": "El actor debe probar los hechos constitutivos de su acción."}],
        tipo_asunto="amparo_directo", materia="civil", variante="v4")
    for k, v in kw.items():
        setattr(m, k, v)
    return m


SEGS = [
    {"id": "C1.a", "concepto": 1, "parrafo": 0, "texto": "Sin pericial no hay identidad.",
     "cita": "tuvo por acreditada la identidad del inmueble sin que se desahogara la prueba pericial",
     "anclas": ["art 16 cpeum"]},
    {"id": "C1.b", "concepto": 1, "parrafo": 0, "texto": "Las superficies no coinciden.",
     "cita": "en la demanda se reclamó una superficie de 2.5 hectáreas y la condena se refiere a "
             "una fracción de 4-80-54.113 hectáreas",
     "anclas": ["cifra 2.5 ha", "cifra 4-80-54.113 ha"]},
    {"id": "C2.a", "concepto": 2, "parrafo": 1, "texto": "Cosa juzgada refleja.",
     "cita": "La Sala omitió examinar de oficio la cosa juzgada refleja que deriva de lo resuelto "
             "en el juicio 1114/2017",
     "anclas": ["exp 1114/2017"]},
    {"id": "C3.a", "concepto": 3, "parrafo": 2, "texto": "Reitera la falta de pericial.",
     "cita": "Insisto en que sin la pericial en topografía no puede tenerse por identificado el inmueble",
     "anclas": []},
    {"id": "C3.b", "concepto": 3, "parrafo": 2, "texto": "Pide aplicar por analogía el 1114/2017.",
     "cita": "solicito que se aplique por analogía el criterio del juicio 1114/2017 y la tesis de "
             "registro 196451",
     "anclas": ["exp 1114/2017", "reg 196451"]},
]
SEGS_N = [pe.normalizar_segmento_piso(s) for s in SEGS]

PLAN_BUENO = {
    "proposiciones": [
        {"id": "P1", "dice": "La identidad quedó acreditada con la confesión del demandado",
         "caracter": "toral", "relacion": "suficiente", "fuente": "reclamada",
         "cita": "la identidad del inmueble quedó acreditada con la confesión del demandado de "
                 "poseer dentro de la parcela de la actora", "vinculada_por_ejecutoria": False},
        {"id": "P2", "dice": "No hay cosa juzgada: así lo ordenó la ejecutoria del amparo previo",
         "caracter": "toral", "relacion": "necesaria", "fuente": "reclamada",
         "cita": "no existe cosa juzgada, pues así lo ordenó la ejecutoria del amparo directo 312/2023",
         "vinculada_por_ejecutoria": True}],
    "segmentos": [
        {"id": "C1.a", "problema_id": 1, "vicio": "fondo", "ataca": "P1", "reitera": None,
         "dato": {"texto": "confesión de poseer dentro de la parcela",
                  "cita": "confesión del demandado de poseer dentro de la parcela de la actora",
                  "fuente": "reclamada"},
         "etiqueta": "infundado", "razon": "fondo_desestimado", "trat": "aplica",
         "diferencia": None, "pendiente": None, "sostiene": "sin pericial no hay identidad"},
        {"id": "C1.b", "problema_id": 1, "vicio": "fondo", "ataca": "P1", "reitera": None,
         "dato": {"texto": "las dos superficies",
                  "cita": "una superficie de 2.5 hectáreas y la condena se refiere a una fracción "
                          "de 4-80-54.113 hectáreas", "fuente": "escrito"},
         "etiqueta": "infundado", "razon": "fondo_desestimado", "trat": "aplica",
         "diferencia": None, "pendiente": None},
        {"id": "C2.a", "problema_id": 2, "vicio": "fondo", "ataca": "P2", "reitera": None,
         "dato": None, "etiqueta": "inoperante", "razon": "cosa_juzgada_amparo_previo(P2)",
         "trat": "aplica", "diferencia": None, "pendiente": None},
        {"id": "C3.a", "problema_id": 1, "vicio": "fondo", "ataca": "P1", "reitera": "C1.a",
         "dato": None, "etiqueta": "infundado", "razon": "fondo_desestimado", "trat": "remite",
         "diferencia": None, "pendiente": None},
        {"id": "C3.b", "problema_id": 1, "vicio": "fondo", "ataca": "P1", "reitera": None,
         "dato": None, "etiqueta": "infundado", "razon": "fondo_desestimado", "trat": "desarrolla",
         "diferencia": "precedente", "pendiente": "razon"}],
    "premisas": [
        {"id": "M1", "responde_a": ["P1"], "fuentes": {"tesis": ["T1"], "normas": []},
         "anclas": ["confesión", "dentro de la parcela", "extensión, no identidad"],
         "rastro": "razon",
         "rastro_cita": "confesión del demandado de poseer dentro de la parcela de la actora"},
        {"id": "M2", "responde_a": ["P2"], "fuentes": {"tesis": ["T2"], "normas": []},
         "anclas": ["amparo directo 312/2023", "firme"], "rastro": "razon",
         "rastro_cita": "se dictó en cumplimiento de la ejecutoria del amparo directo 312/2023"}],
    "unidades": [
        {"id": "U1", "problemas": [1], "segmentos": ["C1.a", "C1.b", "C3.a", "C3.b"],
         "premisa": "M1", "objecion": {"de": "la parte quejosa", "anclas": ["pericial en topografía"]}},
        {"id": "U2", "problemas": [2], "segmentos": ["C2.a"], "premisa": "M2", "objecion": None}],
    "propuestas": [], "avisos_al_secretario": [],
    "orden": {"criterio": "promovente", "por_que": "el orden del escrito ya es el lógico"},
}


def norm(d):
    return pe.normalizar(copy.deepcopy(d))


def _v0(plan_crudo, c=None, f=None, m=None, contexto="", suplencia=None):
    return pe.validar(norm(plan_crudo), c or crit(), SEGS_N, f or fases(), m or material(),
                      contexto, suplencia=suplencia or {})


def _rep(plan_crudo, c=None, f=None, m=None, contexto="", suplencia=None, tocados=None):
    return pe.reparar(norm(plan_crudo), c or crit(), SEGS_N, f or fases(), m or material(),
                      contexto, suplencia or {}, tocados=tocados)


def mutar(fn):
    d = copy.deepcopy(PLAN_BUENO)
    fn(d)
    return d


def seg(d, sid):
    return next(s for s in d["segmentos"] if s["id"] == sid)


# ═══════════════════════════════════════════════════════════════════════════
print("\n1 · CATÁLOGOS, NORMALIZACIÓN Y VERIFICACIÓN LITERAL")
ok(len(pe.RAZONES) == 18 and "cosa_juzgada_amparo_previo" in pe.RAZONES,
   "las 17 razones tipificadas de §4.2 (con esencialmente_fundado aparte: 18 claves)")
ok(pe.norm_id("c1.A") == "C1.a" and pe.norm_id("ad2.b") == "AD2.b" and pe.norm_id("C1a") == "C1.a"
   and pe.norm_id("nada") == "", "los identificadores se normalizan y lo que no es id no casa")
ok(pe.norm_sentido("Fundado pero insuficiente") == "fundado_insuficiente"
   and pe.clase_sentido("fundado_insuficiente") == pe.NO_PROSPERA
   and pe.clase_sentido("innecesario") == pe.NO_SE_ESTUDIA
   and pe.clase_sentido("esencialmente fundado") == pe.PROSPERA,
   "el mapa cerrado: prospera / no prospera / no se estudia, con tipos_asunto.prospera")
ok(pe._razon("no_combate(P2)") == ("no_combate", "P2") and pe._razon("inventada") == ("", None),
   "la razón con su proposición; una razón fuera del catálogo queda vacía")
_t = pe.Texto(ESCRITO)
ok(not _t.contiene("tuvo por acreditada la identidad del edificio sin que se desahogara"),
   "una sola palabra cambiada y ya no casa")
ok(_t.contiene("tuvo por acreditada la identidad del inmueble, sin que se desahogara"),
   "la puntuación no cuenta")
ok(_t.contiene("tuvo por acreditada la IDENTIDAD del inmueble sin que se desahogara"),
   "mayúsculas, espacios y acentos no cuentan; las palabras sí")
ok(_t.contiene("La Sala omitió examinar … la cosa juzgada refleja que deriva"),
   "con elisión, cada tramo en orden")
ok(not _t.contiene("el perito oficial dictaminó que la parcela mide nueve hectáreas"),
   "una cita inventada no casa")
ok(not _t.contiene("la prueba pericial", minimo=4), "tres palabras no bastan para verificar")
ok(pe.tipo_de_ancla("art 16 cpeum") == "norma" and pe.tipo_de_ancla("reg 196451") == "precedente"
   and pe.tipo_de_ancla("exp 1114/2017") == "precedente" and pe.tipo_de_ancla("cifra 2.5 ha") == "hecho",
   "la diferencia que trae un ancla propia")
ok(pe.anclas_de_razon(RAZON1)["registros"] == ["2001111"]
   and pe.anclas_de_razon("el artículo 84 y el 16")["articulos"] == ["84"],
   "los registros y artículos que el secretario escribió en su razón")

# ═══════════════════════════════════════════════════════════════════════════
print("\n2 · UN PLAN BUENO PASA V0 Y CONSERVA EL SENTIDO")
_ind = pe.indice_material(material(), fases())
ok([t["registro"] for t in _ind["tesis"]] == ["2001111", "2002222"] and len(_ind["normas"]) == 1,
   "el índice con el recorte del estudio (y la norma local invocada por el acto sobrevive a la litis)")
_m_metodo = material(tesis=material().tesis + [{"registro": "2003333", "metodo": True,
                                                "rubro": "INTERPRETACIÓN CONFORME.", "texto": "x"}])
ok("2003333" not in [t["registro"] for t in pe.indice_material(_m_metodo, fases())["tesis"]],
   "las tesis de método no son fuente de una premisa")
_pb, _avb = _rep(PLAN_BUENO)
_f = pe.validar(_pb, crit(), SEGS_N, fases(), material(), "", suplencia={})
ok(_f == [], f"V0 lo acepta: {_f[:3]}")
ok(_v0(PLAN_BUENO) == [], "y también sin reparar: no había nada que corregir")
ok([s["etiqueta"] for s in _pb["segmentos"]] == ["infundado", "infundado", "inoperante", "infundado", "infundado"],
   "la etiqueta de cada segmento es, por igualdad, la del criterio de su problema")
ok(_pb["propuestas"] == [] and _avb == [], "sin propuestas ni avisos que inventar")
ok(all(s["cita"] for s in _pb["segmentos"]), "cada segmento con su cita literal verificada del piso")
ok(_pb["n_planteamientos"] == 3 and [p["id"] for p in _pb["problemas"]] == [1, 2],
   "el plan lleva los problemas y el número de planteamientos, sellados por el código")

# ═══════════════════════════════════════════════════════════════════════════
print("\n3 · LOS PLANES QUE V0 DEBE RECHAZAR")
_d = mutar(lambda d: seg(d, "C1.a").update(etiqueta="fundado"))
_f = _v0(_d)
ok(any("C1.a" in x and "etiqueta" in x for x in _f), "etiqueta distinta al criterio")
_d = mutar(lambda d: seg(d, "C2.a").update(etiqueta="infundado"))
ok(any("C2.a" in x and "etiqueta" in x for x in _v0(_d)),
   "…también cuando las dos «no prosperan» (inoperante ≠ infundado): igualdad, no dirección")

# procesal no estudiada
_fp = fases(problemas=[{"pregunta": PREG1, "cubre": [1, 3], "clase": "fondo", "jerarquia": "principal"},
                       {"pregunta": PREG2, "cubre": [2], "clase": "procesal", "jerarquia": "accesorio"}])
_cp = crit(s2="fundado", r2="La Sala debió estudiarla: " + RAZON2)
_d = mutar(lambda d: seg(d, "C2.a").update(etiqueta="fundado", razon="cae_con_principal",
                                           trat="no_se_estudia", vicio="procesal"))
_f = _v0(_d, c=_cp, f=_fp)
ok(any("C2.a" in x and "procesal" in x for x in _f), "violación procesal que el criterio estudia y el plan deja fuera")
_cp2 = crit(s1="fundado", s2="innecesario", r2="Queda sin materia.")
_d = mutar(lambda d: (seg(d, "C2.a").update(etiqueta="innecesario", razon="cae_con_principal",
                                            trat="no_se_estudia", vicio="procesal"),
                      [seg(d, x).update(etiqueta="fundado", razon="fundado") for x in ("C1.a", "C1.b", "C3.a", "C3.b")]))
_f = _v0(_d, c=_cp2, f=_fp)
ok(any("innecesario_mayor_beneficio" in x for x in _f),
   "con una concesión de fondo, la procesal sólo se deja por innecesario_mayor_beneficio (art. 189)")
_d2 = copy.deepcopy(_d)
seg(_d2, "C2.a").update(razon="innecesario_mayor_beneficio")
ok(not any("C2.a" in x for x in _v0(_d2, c=_cp2, f=_fp)), "…y con ella, pasa")
_cp3 = crit(s1="infundado", s2="innecesario", r2="Queda sin materia.")
_d3 = mutar(lambda d: seg(d, "C2.a").update(etiqueta="innecesario", razon="cae_con_principal",
                                            trat="no_se_estudia", vicio="procesal"))
_pr3, _ = _rep(_d3, c=_cp3, f=_fp)
ok(not any("C2.a" in x for x in pe.validar(_pr3, _cp3, SEGS_N, _fp, material(), "")),
   "si es el CRITERIO el que la deja sin estudio sin concesión de fondo, el plan no puede cambiarlo…")
ok(any("C2.a" in x and "74" in x for x in _pr3["avisos_al_secretario"]),
   "…pero se le dice al secretario, con los artículos 74, 174 y 189")

# reitera con ancla propia
_d = mutar(lambda d: seg(d, "C3.b").update(reitera="C2.a", trat="remite", pendiente=None))
ok(any("C3.b" in x and "reiterar" in x for x in _v0(_d)), "reitera con ancla propia (reg 196451)")

# cita inventada
_d = mutar(lambda d: seg(d, "C1.b").update(dato={"texto": "el peritaje",
                                                 "cita": "el perito oficial dictaminó que la parcela mide nueve hectáreas",
                                                 "fuente": "constancia"}))
ok(any("C1.b" in x and "dato" in x for x in _v0(_d)), "cita de dato inventada")
_d = mutar(lambda d: d["proposiciones"][0].update(cita="la Sala dijo que el perito era innecesario por economía procesal"))
ok(any("P1" in x and "cita" in x for x in _v0(_d)), "cita de proposición inventada")
_segs_sin = [dict(s, cita="") if s["id"] == "C1.a" else s for s in SEGS_N]
_d = mutar(lambda d: seg(d, "C1.a").update(cita="la responsable inventó una identidad que nunca existió en autos"))
ok(any("C1.a" in x and "cita literal" in x for x in
       pe.validar(norm(_d), crit(), _segs_sin, fases(), material(), "")),
   "segmento sin cita del piso y con una cita del planificador que no está en el escrito")
_d = mutar(lambda d: seg(d, "C1.a").update(cita="violó en mi perjuicio el artículo 16 constitucional porque tuvo"))
ok(not any("C1.a" in x and "cita literal" in x for x in
           pe.validar(norm(_d), crit(), _segs_sin, fases(), material(), "")),
   "…y con una cita literal suya, pasa")

# premisa sin rastro
_d = mutar(lambda d: d["premisas"][0].update(rastro_cita="la confesión ficta basta siempre para identificar cualquier bien"))
ok(any("M1" in x and "rastro" in x for x in _v0(_d)), "premisa sin rastro verificado")
_d = mutar(lambda d: d["premisas"][0].update(rastro="material",
                                              rastro_cita="puede tenerse por demostrada con la confesión del demandado"))
ok(not any("M1" in x and "rastro" in x for x in _v0(_d)), "el rastro en el texto de su fuente también vale")
_d = mutar(lambda d: d["premisas"][0]["fuentes"].update(tesis=["9999999"]))
ok(any("M1" in x and "9999999" in x for x in _v0(_d)), "fuente fuera del índice y de la razón")

# unidad que mezcla
_d = mutar(lambda d: (d["unidades"][0]["segmentos"].append("C2.a"), d["unidades"].pop(1)))
ok(any("U1" in x and "mezcla" in x for x in _v0(_d)), "una unidad que mezcla proposiciones y razones")
_cg = crit(g1="A", g2="A")
ok(not any("mezcla" in x for x in _v0(_d, c=_cg)), "…salvo que el secretario las agrupó: su grupo manda")
_pg, _ = _rep(_d, c=_cg)
ok(any("grupo que marcaste" in x for x in _pg["avisos_al_secretario"]), "…y el panel lo avisa")

# suplencia confirmada
_sup = {"fraccion": "IV-b", "a_favor_de": "la parte quejosa", "confirmada": True}
_d = mutar(lambda d: seg(d, "C2.a").update(razon="generico"))
ok(any("C2.a" in x and "suplencia" in x for x in _v0(_d, suplencia=_sup)),
   "con la suplencia confirmada no hay inoperancia de forma a favor de su parte")
ok(not any("suplencia" in x for x in _v0(_d)), "…sin suplencia, esa razón sí cabe")

# cosa juzgada contra una P no vinculada
_d = mutar(lambda d: d["proposiciones"][1].update(vinculada_por_ejecutoria=False))
ok(any("C2.a" in x and "vinculada" in x for x in _v0(_d)), "cosa juzgada sólo contra lo que la ejecutoria vinculó")

# orden
_d = mutar(lambda d: d["unidades"].reverse())
ok(any("accesorio" in x for x in _v0(_d)), "el accesorio no va antes de su principal")
_fo = fases(problemas=[{"pregunta": PREG1, "cubre": [1, 3], "clase": "fondo", "jerarquia": "accesorio"},
                       {"pregunta": PREG2, "cubre": [2], "clase": "procesal", "jerarquia": "principal"}])
_co = [f6.Criterio(problema=PREG1, sentido="infundado", razonamiento=RAZON1, jerarquia="accesorio"),
       f6.Criterio(problema=PREG2, sentido="inoperante", razonamiento=RAZON2, jerarquia="principal")]
_d = mutar(lambda d: (seg(d, "C2.a").update(vicio="procesal"), d["unidades"].reverse(),
                      d["orden"].update(por_que="el escrito")))
ok(any("189" in x for x in _v0(_d, c=_co, f=_fo)), "procesal antes que fondo sin decir el mayor beneficio")
_d["orden"]["por_que"] = "estudiar primero la procesal da mayor beneficio: repone todo"
ok(not any("189" in x for x in _v0(_d, c=_co, f=_fo)), "…y diciéndolo, pasa (art. 189)")

# el piso entero
_d = mutar(lambda d: d["segmentos"].pop(3))
ok(any("faltan segmentos" in x and "C3.a" in x for x in _v0(_d)), "un segmento del inventario olvidado")
_d = mutar(lambda d: d["segmentos"].append(dict(seg(d, "C1.a"), id="C9.z")))
ok(any("C9.z" in x for x in _v0(_d)), "un segmento inventado")
_d = mutar(lambda d: seg(d, "C2.a").update(problema_id=1))
ok(any("C2.a" in x and "problema_id" in x for x in _v0(_d)), "el problema lo fija el «cubre», no el modelo")
_d = mutar(lambda d: seg(d, "C1.a").update(razon="fundado"))
ok(any("C1.a" in x and "compatible" in x for x in _v0(_d)), "razón incompatible con la etiqueta")
_d = mutar(lambda d: seg(d, "C1.a").update(razon="no_combate(P1)"))
ok(any("C1.a" in x and "infundado" in x for x in _v0(_d)), "«infundado» no admite una razón de inoperancia")

# ═══════════════════════════════════════════════════════════════════════════
print("\n4 · REPARAR: LO QUE EL CÓDIGO CORRIGE SOLO")
_d = mutar(lambda d: seg(d, "C1.a").update(etiqueta="fundado"))
_r, _av = _rep(_d)
ok(seg(_r, "C1.a")["etiqueta"] == "infundado"
   and _r["propuestas"] == [{"seg": "C1.a", "de": "infundado", "a": "fundado",
                             "por_que": "calificación que sugirió el planificador"}],
   "la etiqueta vuelve a la del criterio y la del modelo pasa a PROPUESTA visible")
_r, _ = _rep(_d, tocados=[PREG1])
ok(seg(_r, "C1.a")["etiqueta"] == "infundado" and _r["propuestas"] == [],
   "en un problema que el secretario tocó a mano, ni se propone")
_d = mutar(lambda d: seg(d, "C3.b").update(reitera="C2.a", trat="remite", pendiente=None, diferencia=None))
_r, _av = _rep(_d)
ok(seg(_r, "C3.b")["reitera"] is None and seg(_r, "C3.b")["trat"] == "desarrolla"
   and seg(_r, "C3.b")["diferencia"] == "precedente" and any("196451" in a for a in _av),
   "reitera con ancla propia → desarrolla, con la diferencia del ancla, y lo dice")
_d = mutar(lambda d: seg(d, "C1.b").update(dato={"texto": "x", "cita": "el perito oficial dictaminó que "
                                                                        "la parcela mide nueve hectáreas",
                                                 "fuente": "constancia"}))
_r, _av = _rep(_d)
ok(seg(_r, "C1.b")["dato"] is None and any("C1.b" in a and "dato" in a for a in _av)
   and pe.validar(_r, crit(), SEGS_N, fases(), material(), "") == [],
   "un dato no verificado se borra, con aviso, y el plan sigue valiendo")
_d = mutar(lambda d: d["premisas"][0]["fuentes"].update(tesis=["9999999", "T1"]))
_r, _av = _rep(_d)
ok(_r["premisas"][0]["fuentes"]["tesis"] == ["2001111"] and any("9999999" in a for a in _av),
   "la fuente ajena al índice sale de la premisa; los T del índice se vuelven registros")
_d = mutar(lambda d: d["premisas"][0]["fuentes"].update(tesis=[]))
_r, _ = _rep(_d)
ok("2001111" in _r["premisas"][0]["fuentes"]["tesis"],
   "el registro que el secretario citó en su razón entra a la premisa de su unidad (V0 h)")
_d = mutar(lambda d: seg(d, "C2.a").update(problema_id=1))
_r, _ = _rep(_d)
ok(seg(_r, "C2.a")["problema_id"] == 2, "el problema_id lo repone el código")
_r, _ = _rep(PLAN_BUENO, c=[crit()[0]])
ok(seg(_r, "C2.a")["pendiente"] == "sentido" and seg(_r, "C2.a")["etiqueta"] == "",
   "un problema sin sentido fijado: sin calificación y pendiente «sentido»")
ok(any("P1" in a and "ningún argumento" in a for a in
       _rep(mutar(lambda d: [seg(d, x).update(ataca=None) for x in ("C1.a", "C1.b", "C3.a", "C3.b")]))[0]
       ["avisos_al_secretario"]),
   "una proposición suficiente que nadie ataca: aviso, nunca reclasificación")

# ═══════════════════════════════════════════════════════════════════════════
print("\n5 · PREPARAR: EL MODELO DE LAS FASES, UN REINTENTO, O NADA")


class _Resp:
    def __init__(self, txt, fin="stop"):
        m = types.SimpleNamespace(content=txt)
        self.choices = [types.SimpleNamespace(message=m, finish_reason=fin)]
        self.usage = None


class Modelo:
    """Doble del cliente: devuelve los planes en orden y anota cada llamada."""

    def __init__(self, *planes, espera=0.0):
        self.planes = list(planes)
        self.kw = []
        self.espera = espera
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._crear))

    async def _crear(self, **kw):
        self.kw.append(kw)
        if self.espera:
            await asyncio.sleep(self.espera)
        p = self.planes.pop(0) if len(self.planes) > 1 else self.planes[0]
        return _Resp(p if isinstance(p, str) else json.dumps(p, ensure_ascii=False))


def resultado(f=None, variante="v4", suplencia=None, formato=""):
    r = types.SimpleNamespace()
    r.fases = f or fases()
    r.encargo = types.SimpleNamespace(tipo_asunto="amparo_directo", es_recurso=False,
                                      variante_estudio=variante, formato=formato,
                                      suplencia=suplencia or {}, conceptos_violacion="",
                                      propuesta_global={"checklist": [{"tema": "identidad"}]},
                                      plan={}, guion="")
    r.avisos = []
    return r


_malo = mutar(lambda d: seg(d, "C3.b").update(reitera="C2.a", trat="remite", pendiente=None))
_malo["premisas"][0]["rastro_cita"] = "texto que no está en ninguna parte y no se puede verificar"
_cli = Modelo(_malo, PLAN_BUENO)
_plan, _av, _info = asyncio.run(pe.preparar(_cli, resultado(), crit(), material(), "", {}, segs=SEGS_N))
ok(_plan is not None and _info["intentos"] == 2, "el primero falla V0, el reintento pasa")
ok("LO QUE EL CÓDIGO RECHAZÓ" in _cli.kw[1]["messages"][0]["content"]
   and "M1" in _cli.kw[1]["messages"][0]["content"].split("LO QUE EL CÓDIGO RECHAZÓ")[1][:2000]
   and "LO QUE EL CÓDIGO RECHAZÓ" not in _cli.kw[0]["messages"][0]["content"],
   "el reintento lleva la lista de lo que falló, y el primero no")
ok(_cli.kw[0]["model"] == f123.MODELO_FASES == "gpt-5.6-luna"
   and _cli.kw[0]["reasoning_effort"] == "medium"
   and _cli.kw[0]["response_format"] == {"type": "json_object"},
   "el modelo de las fases, razonamiento MEDIO y JSON estricto")
_cli = Modelo(_malo)
_plan, _av, _info = asyncio.run(pe.preparar(_cli, resultado(), crit(), material(), "", {}, segs=SEGS_N))
ok(_plan is None and _info["intentos"] == 2 and "dos veces" in _av[0],
   "dos rechazos: no hay plan, y se dice por qué")
_cli = Modelo("esto no es JSON", PLAN_BUENO)
_plan, _, _info = asyncio.run(pe.preparar(_cli, resultado(), crit(), material(), "", {}, segs=SEGS_N))
ok(_plan is not None and _info["intentos"] == 2, "una respuesta que no es JSON cuenta como plan vacío y se reintenta")
try:
    asyncio.run(pe.preparar(Modelo(PLAN_BUENO), resultado(fases(fuentes=[ACTO, ""])), crit(),
                            material(), "", {}, segs=SEGS_N))
    ok(False, "sin escrito no hay plan")
except pe.PlanNoDisponible:
    ok(True, "sin escrito no hay plan: no hay cita que verificar")
_p = pe.prompt_plan(tipo_asunto="amparo_directo", probs=pe.problemas_del_criterio(crit(), fases()),
                    segs=SEGS_N, resumen_acto="r", tramos=pe._tramos_del_escrito(fases()),
                    indice=_ind, n=3)
ok(RAZON1 in _p and "sentido fijado por el secretario: infundado" in _p,
   "el planificador ve la razón LITERAL del secretario y su sentido")
ok("── concepto de violación 3 ──" in _p and "196451" in _p, "y el escrito literal, por concepto")
ok("T1 · registro 2001111" in _p and "N1 · Código de Procedimientos Civiles" in _p, "y el índice del material")
ok("Sobre el primer" not in _p and "Lo anterior" not in _p and "Por cuestión de método" not in _p,
   "sin frases modelo: descripciones, tipos y datos")
_largo = fases(fuentes=[ACTO, ESCRITO[:_o[1]] + "X " * 40000 + ESCRITO[_o[1]:]],
               conteo={"estado": "contado", "n": 3,
                       "tramos": [[_o[0], _o[1] + 80000], [_o[1] + 80000, _o[2] + 80000],
                                  [_o[2] + 80000, len(ESCRITO) + 80000]]})
_tr = pe._tramos_del_escrito(_largo, tope_total=9000)
ok(len(_tr[0][1]) < 4000 and "[…]" in _tr[0][1] and _tr[0][1].rstrip().endswith("X"),
   "el tope es por concepto, con cabeza y COLA: el final de un concepto no se pierde")

# ═══════════════════════════════════════════════════════════════════════════
print("\n6 · EL GUION POR FORMA")
_g = pe.vista(_pb, "estandar")
print("      " + _g.replace("\n", "\n      ")[:1800])
ok(_g.count("EXPONE M1") == 1 and _g.count("EXPONE M2") == 1, "cada premisa se expone UNA vez")
ok("APARTADO 1 · abre: concepto de violación 1" in _g and "APARTADO 3 · abre: concepto de violación 3" in _g,
   "estándar: un apartado por concepto, en el orden del escrito")
ok("REMITE C3.a" in _g and "→ apartado 1 (M1)" in _g and "proposición: P1" in _g,
   "la remisión lleva destino y proposición")
ok(_g.count("objeción aquí, una vez") == 1 and _g.index("objeción aquí") < _g.index("APARTADO 2"),
   "la objeción, una vez, en el apartado donde se contesta la unidad")
ok("PENDIENTE DE RAZÓN: C3.b (precedente)" in _g and "ADVERTENCIAS" in _g, "el pendiente de razón, primero en ADVERTENCIAS")
ok("dato (escrito): «una superficie de 2.5 hectáreas" in _g, "el dato verificado, literal, con su fuente")
ok("EXTENSIÓN (techo, no meta)" in _g, "el presupuesto por apartado, como techo")
_pp = copy.deepcopy(_pb)
_pp["orden"]["criterio"] = "prelacion"
_gp = pe.vista(_pp, "estandar")
ok("abre: concepto de violación 1 y 3" in _gp and "REMITE" not in _gp,
   "en prelación, los conceptos de la MISMA unidad se juntan y no se remite hacia adelante")
_gm = pe.vista(_pb, "moderna")
ok(f"APARTADO 1 · problema 1: «{PREG1}»" in _gm and f"APARTADO 2 · problema 2: «{PREG2}»" in _gm
   and _gm.count("EXPONE M1") == 1 and "REMITE" not in _gm,
   "moderna: un apartado por problema con su pregunta; las unidades dentro")
for frase in ("Sobre el primer", "Lo anterior", "Se considera", "Por cuestión de método", "resulta infundado"):
    ok(frase not in _g and frase not in _gm, f"el guion no trae la frase modelo «{frase}»")
_b = pe.bloque(_g)
ok(_b.count("EL GUION DEL ESTUDIO") == 1 and "DESVIACIONES DEL GUION" in _b and _g in _b,
   "el bloque describe sus rótulos y trae el guion entero")
ok(pe.bloque("") == "", "sin guion no hay bloque")

# ═══════════════════════════════════════════════════════════════════════════
print("\n7 · DECISIÓN 6: LA RAZÓN QUE FALTA")
_con = pe.aplicar_razones(_pb, pe.leer_razones_segmento(json.dumps(
    {"c3.B": "Lo resuelto en el 1114/2017 no vincula aquí: es otro juicio y otra acción."})))
ok(seg(_con, "C3.b")["pendiente"] is None and "otra acción" in seg(_con, "C3.b")["razon_secretario"],
   "la razón que escribió entra al segmento como suya")
_gc = pe.vista(_con, "estandar")
ok("razón del secretario para este argumento: «Lo resuelto en el 1114/2017" in _gc
   and "PENDIENTE DE RAZÓN" not in _gc, "y en el guion ya no está pendiente")
ok(seg(pe.aplicar_razones(_pb, {"C3.b": "   "}), "C3.b")["pendiente"] == "razon",
   "una caja vacía no cuenta: sigue pendiente")
ok(pe.leer_razones_segmento("{roto") == {} and pe.leer_razones_segmento("") == {},
   "un JSON roto se trata como vacío")
ok(seg(pe.aplicar_razones(_pb, {"C1.a": "otra cosa"}), "C1.a").get("razon_secretario") is None,
   "sólo se aplica a los pendientes de razón")

# ═══════════════════════════════════════════════════════════════════════════
print("\n8 · LA CLAVE")
_k = pe.clave(crit(), "h", "ctx", {}, "estandar")
ok(_k == pe.clave(crit(), "h", "ctx", {}, "moderna"), "la forma NO entra: el plan es el mismo, cambia la vista")
ok(_k != pe.clave(crit(r1=RAZON1 + " Además."), "h", "ctx", {}, "estandar"), "la razón literal SÍ entra")
ok(_k != pe.clave(crit(g1="A", g2="A"), "h", "ctx", {}, "estandar"), "el grupo entra")
ok(_k != pe.clave(crit(), "h2", "ctx", {}, "estandar") and _k != pe.clave(crit(), "h", "otro", {}, "estandar"),
   "las entradas y el contexto entran")
ok(_k == pe.clave(crit(), "h", "ctx", {"fraccion": "VII", "confirmada": False}, "estandar")
   and _k != pe.clave(crit(), "h", "ctx", {"fraccion": "VII", "confirmada": True}, "estandar"),
   "la suplencia sólo cuenta confirmada")
_r0 = resultado()
_h1 = pe.huella_entradas(_r0, material(), "", SEGS_N)
ok(_h1 == pe.huella_entradas(resultado(), material(), "", SEGS_N)
   and _h1 != pe.huella_entradas(_r0, material(tesis=material().tesis[:1]), "", SEGS_N),
   "la huella cambia si cambia el índice del material que verá el estudio")

# ═══════════════════════════════════════════════════════════════════════════
print("\n9 · EL ESTADO EN LA FILA (funciones puras)")
_t0 = 1000.0
d, dec = pe.fila_pedir(None, "k1", "H", _t0)
ok(dec == "lanzar" and d["corridas"] == 1 and d["planes"]["k1"]["estado"] == "en_curso"
   and d["pedido_clave"] == "k1", "el primer pedido reserva una corrida")
d2, dec = pe.fila_pedir(d, "k1", "H", _t0 + 10)
ok(dec == "en_curso" and d2["corridas"] == 1, "la misma clave en curso: se espera, no se relanza")
d3, dec = pe.fila_pedir(d, "k1", "H", _t0 + 10 + pe.PLAN_ABANDONADO_S + 5)
ok(dec == "lanzar" and d3["corridas"] == 2, "una corrida sin latido se da por abandonada y se relanza")
d4, dec = pe.fila_resultado(d, "k1", "H", {"segmentos": []}, [], 30.0, _t0 + 30)
ok(dec == "listo" and pe.fila_pedir(d4, "k1", "H", _t0 + 40)[1] == "listo", "listo: se sirve")
d5, _ = pe.fila_resultado(d, "k1", "H", None, ["falló"], 30.0, _t0 + 30)
ok(pe.fila_pedir(d5, "k1", "H", _t0 + 40)[1] == "fallo", "una clave que falló no se recalcula")
d6, dec = pe.fila_resultado(d, "k1", "H", None, ["red"], 3.0, _t0 + 30, reintentable=True)
ok(dec == "error" and pe.estado_para_pantalla(d6, "H", _t0 + 31)["estado"] == "fallo"
   and pe.fila_pedir(d6, "k1", "H", _t0 + 40)[1] == "lanzar",
   "un error del proveedor no condena la clave: la pantalla deja de esperar y el próximo pedido reintenta")
dx = None
for i in range(pe.TOPE_CORRIDAS):
    dx, dec = pe.fila_pedir(dx, f"k{i}", "H", _t0)
ok(dec == "lanzar" and pe.fila_pedir(dx, "otra", "H", _t0)[1] == "tope", "cuatro corridas por sesión, contando todas")
ok(pe.fila_pedir(dx, "otra", "H-nueva", _t0)[1] == "lanzar", "rehacer el adelanto empieza de cero")
ok(pe.fila_resultado(d, "k1", "OTRA", {"x": 1}, [], 1.0, _t0)[0] is None,
   "un plan de otro adelanto no se escribe")
ok(pe.estado_para_pantalla(d, "H", _t0 + 5)["estado"] == "en_curso"
   and pe.estado_para_pantalla(d, "H", _t0 + 500)["estado"] == "fallo"
   and pe.estado_para_pantalla(d4, "H", _t0 + 50)["plan"] == {"segmentos": []}
   and pe.estado_para_pantalla(None, "H", _t0)["estado"] == "sin_plan"
   and pe.estado_para_pantalla(d4, "OTRA", _t0)["estado"] == "sin_plan",
   "lo que ve la pantalla: en curso, abandonado = fallo, listo con plan, sin fila o de otro adelanto")
dl, dec = pe.fila_latido(d, "k1", "H", _t0 + 44)
ok(dec == "latido" and dl["planes"]["k1"]["latido"] == _t0 + 44, "el latido")
dp = None
for i in range(9):
    dp, _ = pe.fila_pedir(dp, f"p{i}", "H", _t0 + i, tope=99)
ok(len(dp["planes"]) <= 6 and "p8" in dp["planes"], "como mucho seis casillas; la del pedido se queda")

# ═══════════════════════════════════════════════════════════════════════════
print("\n10 · FASE 6: v1 Y v2 IDÉNTICAS; LA v4 = LA v3 + EL GUION")
# Huellas de los prompts v1 y v2 renderizados con el código de ANTES de esta
# entrega (origin/main de-66e3d), con el criterio real del ADC 93/2026 de
# test_prompt_v2.py. Si cambian, esta entrega movió una coma de producción.
CRIT_93 = [
    {"problema": "¿La Sala debió admitir la ampliación de demanda y estudiar los argumentos "
                 "dirigidos contra el crédito fiscal?", "sentido": "fundado",
     "razonamiento": "La interlocutoria confirmó la preclusión exclusivamente porque reputó "
                     "aplicable el término sumario de cinco días, pese a que el acuerdo de primero "
                     "de julio dejó a salvo la ampliación con referencia al artículo 17 y a la "
                     "jurisprudencia citada. Al no esclarecer ni respetar ese contenido, cerró el "
                     "debate sobre el crédito y afectó la congruencia y exhaustividad.",
     "jerarquia": "principal"},
    {"problema": "¿La Sala debía pronunciarse sobre los argumentos formulados por la actora en "
                 "sus alegatos?", "sentido": "innecesario",
     "razonamiento": "Dado el sentido del estudio del problema principal, queda sin materia el "
                     "análisis de este planteamiento: La nueva sentencia deberá pronunciarse sobre "
                     "los alegatos en la medida en que la ampliación incorpore el crédito fiscal y "
                     "sus cuestiones conexas.", "jerarquia": "accesorio"}]
C93 = [f6.Criterio(problema=d["problema"], sentido=d["sentido"], razonamiento=d["razonamiento"],
                   jerarquia=d["jerarquia"]) for d in CRIT_93]
P93 = [{"pregunta": C93[0].problema, "cubre": [1]}, {"pregunta": C93[1].problema, "cubre": [2]}]
HUELLAS = {
    ("v1", "estandar"): "2b4186f558e719d5c550a3148f39c1aaec26cc8aec84de6ea298fe1bef1f71f2",
    ("v1", "moderna"): "94a6f328d09ba437c3cc353bbc9fba22d2e90ea6682a542c4ffd77eb35a54015",
    # La v2 cambió en p2-exhaustivo (26-sep-2026): el «sin materia» es del
    # criterio y lo que la concesión deja a la responsable va en los EFECTOS
    # (test_exhaustivo.py); y en su revisión adversarial, la razón del
    # secretario y la inoperancia suelta. La v1 no se mueve.
    ("v2", "estandar"): "a3eca1abc777e0f9a832a1329a374b64bda5226d23cc875b3ad253ce15a00736",
    ("v2", "moderna"): "bdfacda5276d843d6fc6108a5194da38cfae4e5dc69c6aeec3f5490b9428369c",
}


def m93(var, forma):
    return f6.Material(tipo_asunto="amparo_directo", materia="administrativa", formato=forma,
                       problemas=P93, n_planteamientos=2, variante=var)


def p93(var, forma, **kw):
    return f6.prompt_estudio("LO QUE RESOLVIÓ LA SALA (resumen de prueba).",
                             "LO QUE SE COMBATE (resumen de prueba).", C93, m93(var, forma),
                             contexto="contexto de prueba",
                             escrito_literal="ESCRITO LITERAL de prueba " * 20, **kw)


for (var, forma), h in HUELLAS.items():
    _a = p93(var, forma)
    ok(hashlib.sha256(_a.encode("utf-8")).hexdigest() == h, f"{var} {forma}: idéntica a la de antes")
    ok(p93(var, forma, guion="GUION QUE NO DEBE ENTRAR") == _a, f"{var} {forma}: el guion no la toca")
ok(f6.VARIANTES == ("v1", "v2", "v3", "v4") and f6.normalizar_variante("C") == "v4"
   and f6.normalizar_variante("clite") == "v3" and f6.normalizar_variante("v9", "v1") == "v1",
   "v3 y v4 son variantes; «C» y «Clite», sus nombres en la propuesta")
ok(all(f6._v2(m93(v, "estandar")) for v in ("v2", "v3", "v4")) and not f6._v2(m93("v1", "estandar")),
   "v3 y v4 usan los controles de la limpieza, no los de la v1")
for forma in ("estandar", "moderna"):
    _v3 = p93("v3", forma)
    _v4 = p93("v4", forma, guion=_g)
    _blq = pe.bloque(_g)
    ok(p93("v4", forma) == _v3 and p93("v4", forma, guion="") == _v3,
       f"{forma}: sin guion, la v4 ES la v3 (el plan falló o venció)")
    ok(_blq in _v4 and _v4.index(_blq) < _v4.rindex("Escribe el estudio de fondo.")
       and _v4.rstrip().endswith("Nada más.") and f6._RECUERDA_GUION.strip() in _v4,
       f"{forma}: con guion, el bloque antes de «Escribe el estudio de fondo.» y el recordatorio al final")
    ok(_v4.replace("\n" + _blq, "", 1).replace("\n" + f6._RECUERDA_GUION.rstrip("\n"), "", 1) == _v3,
       f"{forma}: la v4 = la v3 + el guion, ni un carácter más")


class _CliEstudio:
    def __init__(self):
        self.kw = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._c))

    async def _c(self, **kw):
        self.kw.append(kw)
        if kw.get("stream"):
            async def _gen():
                yield types.SimpleNamespace(usage=None, choices=[types.SimpleNamespace(
                    finish_reason="stop", delta=types.SimpleNamespace(content="Los conceptos son infundados."))])
            return _gen()
        return _Resp("Los conceptos son infundados.")


_ce = _CliEstudio()
asyncio.run(f6.redactar(_ce, "ACTO", "CONC", C93, m93("v4", "estandar"), guion=_g))


async def _vivo():
    async for _ in f6.redactar_en_vivo(_ce, "ACTO", "CONC", C93, m93("v4", "estandar"), guion=_g):
        pass
asyncio.run(_vivo())
ok(len(_ce.kw) == 2 and all(_g in k["messages"][0]["content"] for k in _ce.kw),
   "los DOS redactores pasan el guion al prompt")
ok(all(k.get("reasoning_effort") == f6.ESFUERZO_ESTUDIO == "high" for k in _ce.kw),
   "el razonamiento del estudio sigue en «high»")

SRC_RA = open(os.path.join(AQUI, "redactor_adelanto.py"), encoding="utf-8").read()
_a_ra = ast.parse(SRC_RA)
for nombre, llamada in (("resolver", "redactar"), ("resolver_en_vivo", "redactar_en_vivo")):
    fn = next(n for n in ast.walk(_a_ra) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and n.name == nombre)
    ll = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
          and n.func.attr == llamada]
    ok(len(ll) == 1 and any(k.arg == "guion" for k in ll[0].keywords),
       f"redactor_adelanto.{nombre}: pasa el guion del encargo a f6.{llamada}")

# ═══════════════════════════════════════════════════════════════════════════
print("\n11 · LOS GEMELOS POR AST")
SRC_MAIN = open(os.path.join(AQUI, "main.py"), encoding="utf-8").read()
A_MAIN = ast.parse(SRC_MAIN)
FN = {n.name: n for n in A_MAIN.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def llamadas(fn, nombre):
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == nombre]


def ruta(fn):
    for d in fn.decorator_list:
        if isinstance(d, ast.Call) and d.args and isinstance(d.args[0], ast.Constant):
            return getattr(d.func, "attr", ""), d.args[0].value
    return None


for nombre in ("taller_resolver_stream", "taller_resolver"):
    fn = FN[nombre]
    args = [a.arg for a in fn.args.args]
    ok("razones_segmento" in args, f"{nombre}: recibe `razones_segmento`")
    ok(len(llamadas(fn, "_taller_armar_criterio")) == 1, f"{nombre}: arma el criterio con la función común")
    pl = llamadas(fn, "_taller_plan_para")
    ok(len(pl) == 1 and {k.arg for k in pl[0].keywords} >= {"contexto", "razones_segmento", "tocados"},
       f"{nombre}: pide el plan con contexto, razones y tocados")
    _src_fn = ast.get_source_segment(SRC_MAIN, fn)
    ok("_md.repartir(" not in _src_fn and "_ad.aplicar(" not in _src_fn
       and "_dz.reconciliar(" not in _src_fn, f"{nombre}: no le queda un reparto, un árbol ni un desenlace propios")
_st = FN["taller_resolver_stream"]
_trab = next(n for n in ast.walk(_st) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_trabajar")
ok(llamadas(_trab, "_taller_plan_para") and all(
    _trab.lineno <= c.lineno <= _trab.end_lineno for c in llamadas(_st, "_taller_plan_para")),
   "flujo: el plan se espera DENTRO de `_trabajar` (la tarea), nunca antes de las cabeceras")
_src_trab = ast.get_source_segment(SRC_MAIN, _trab)
ok(_src_trab.index('{"tipo": "ordenando"}') < _src_trab.index("_taller_plan_para(")
   < _src_trab.index("_ra.resolver_en_vivo("), "flujo: «ordenando», el plan, y luego el estudio")
_src_pl = ast.get_source_segment(SRC_MAIN, FN["taller_resolver"])
ok(_src_pl.index("_taller_plan_para(") < _src_pl.index("_ra.resolver("), "plano: el plan antes del estudio")
_ped = FN["taller_plan_pedir"]
ok(ruta(_ped) == ("post", "/taller/plan/pedir") and ruta(FN["taller_plan"]) == ("get", "/taller/plan"),
   "las dos puertas del contrato: GET /taller/plan y POST /taller/plan/pedir")
_args_ped = [a.arg for a in _ped.args.args]
ok(all(x in _args_ped for x in ("numero", "user_email", "criterios_json", "global_json", "contexto",
                                "suplencia", "formato", "razones_segmento")),
   "el pedido recibe los campos del contrato")
ok(all(x in _args_ped for x in ("modo_decision", "sentido_global", "global_dictado", "usar_propuesta",
                                "sentido", "problema", "razonamiento")),
   "…y los mismos campos de decisión que los gemelos (si no, la clave no casaría)")
ok(len(llamadas(_ped, "_taller_armar_criterio")) == 1 and len(llamadas(_ped, "_taller_plan_pedido")) == 1,
   "el pedido arma el MISMO criterio y no espera el plan")
ok(not any(getattr(k, "arg", "") == "cobrable" for c in llamadas(_ped, "_taller_puerta") for k in c.keywords),
   "el pedido no cobra")
_pp_src = ast.get_source_segment(SRC_MAIN, FN["_taller_preproponer"])
ok("_taller_plan_adelantar(email)" in _pp_src and "_taller_plan_desde_propuesta(" in _pp_src,
   "la propuesta calculada sola dispara el plan adelantado (si procede)")
ok('"plan": _taller_plan_ficha(res)' in ast.get_source_segment(SRC_MAIN, FN["_taller_guardar_proyecto"])
   and '"plan": _taller_plan_ficha(res)' in _src_trab,
   "la ficha y el evento «listo» llevan el plan (Mapa del estudio)")
_pd_src = ast.get_source_segment(SRC_MAIN, FN["_taller_plan_para"])
ok("e.plan, e.guion = {}, \"\"" in _pd_src and 'e.variante_estudio = "v3"' in _pd_src,
   "el plan y el guion se ponen SIEMPRE; sin plan, la v3")
ok(os.path.exists(os.path.join(AQUI, "migraciones", "2026-09-26_taller_plan.sql"))
   and "ADD COLUMN IF NOT EXISTS plan jsonb" in open(os.path.join(
       AQUI, "migraciones", "2026-09-26_taller_plan.sql"), encoding="utf-8").read(),
   "la migración de la columna, escrita (la aplica el integrador)")

# ═══════════════════════════════════════════════════════════════════════════
print("\n12 · EL CRITERIO COMÚN: EL GRUPO EN LOS TRES MODOS")
from fastapi import HTTPException  # noqa: E402

_ns = {"json": json, "HTTPException": HTTPException, "err": lambda e: str(e), "os": os,
       "time": __import__("time"), "asyncio": asyncio}
_piezas = [FN[n] for n in ("_taller_tocados", "_taller_armar_criterio", "_taller_glob")]
exec(compile(ast.Module(body=_piezas, type_ignores=[]), "main.py", "exec"), _ns)
_arm = _ns["_taller_armar_criterio"]
_ses = {"propuestas": [types.SimpleNamespace(problema=PREG1, sentido="infundado", razon="r1", alcanza=True,
                                             sentido_propio="", razon_propia=""),
                       types.SimpleNamespace(problema=PREG2, sentido="inoperante", razon="r2", alcanza=True,
                                             sentido_propio="", razon_propia="")]}
_rr = resultado()
_cj = json.dumps([{"problema": PREG1, "grupo": "A", "tocado": False},
                  {"problema": PREG2, "grupo": "A", "tocado": False}])
_g_mod = _arm(_rr, _ses, {}, criterios_json=_cj, modo_decision="global", sentido_global="infundado")
ok([c.grupo for c in _g_mod["crit"]] == ["A", "A"], "modo global: el grupo llega (antes se tiraba)")
ok([c.sentido for c in _g_mod["crit"]] == ["infundado", "inoperante"],
   "…y una entrada «tocado: false» sólo trae su grupo: no pisa el relleno del global")
_a_mod = _arm(_rr, _ses, {}, criterios_json=_cj, modo_decision="acervo")
ok([c.grupo for c in _a_mod["crit"]] == ["A", "A"] and any("SIN SUPERVISIÓN" in a for a in _a_mod["avisos_r"]),
   "modo del motor: el grupo llega, y el aviso de sin supervisión vuelve en su lista")
_t_mod = _arm(_rr, _ses, {"sentido": "fundado", "alcanza": True}, criterios_json=_cj, modo_decision="acervo")
ok(_t_mod["crit"][0].sentido == "fundado" and any("tarjeta final" in a for a in _t_mod["avisos_fases"]),
   "…y los grupos no le quitan al modo del motor su tarjeta final (el desenlace sigue mandando)")
_p_mod = _arm(_rr, _ses, {}, criterios_json=json.dumps(
    [{"problema": PREG1, "sentido": "infundado", "grupo": "B", "razonamiento": "x"},
     {"problema": PREG2, "sentido": "inoperante", "grupo": "B", "razonamiento": "y"}]))
ok([c.grupo for c in _p_mod["crit"]] == ["B", "B"], "por problema: el grupo, como siempre")
ok(_rr.avisos == [] and _rr.fases.avisos == [], "el criterio común no toca el resultado: devuelve los avisos")
_c1 = _arm(_rr, _ses, {}, criterios_json=_cj, modo_decision="global", sentido_global="infundado")["crit"]
ok(pe.clave(_c1, "h", "", {}) == pe.clave(_g_mod["crit"], "h", "", {}),
   "el mismo formulario da el mismo criterio y la misma clave (lo que usan pedir y los dos gemelos)")
try:
    _arm(_rr, _ses, {}, modo_decision="global")
    ok(False, "el modo global sin sentido global")
except HTTPException as ex:
    ok(ex.status_code == 422, "el modo global sin sentido global: 422, como antes")

# ═══════════════════════════════════════════════════════════════════════════
print("\n13 · EL RESOLVER DE PUNTA A PUNTA (base y modelo de mentira)")


class _Res:
    def __init__(self, data):
        self.data = data


class FakeBase:
    """La tabla taller_sesiones en memoria, con los filtros que usa el
    compare-and-set: email, expediente, plan->>rev y «plan is null»."""

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

_inv = types.ModuleType("inventario")
_inv.segmentos = lambda fases_, escrito="", es_recurso=False: copy.deepcopy(SEGS)
_inv_real = sys.modules.get("inventario")
sys.modules["inventario"] = _inv

_nombres = ("_taller_plan_aplica", "_taller_plan_cas", "_taller_plan_leer", "_taller_plan_contraste",
            "_taller_plan_entradas", "_taller_plan_correr", "_taller_plan_lanzar", "_taller_plan_pedido",
            "_taller_plan_para", "_taller_plan_ficha")


def entorno(modelo, base):
    ns = {"json": json, "os": os, "time": __import__("time"), "asyncio": asyncio, "err": lambda e: str(e),
          "HTTPException": HTTPException, "_te": _te, "supabase_admin": base, "chat_client": modelo,
          "_TALLER_EN_MARCHA": set(), "_taller_leer_contraste": lambda e, n: None}
    exec(compile(ast.Module(body=[FN[n] for n in _nombres], type_ignores=[]), "main.py", "exec"), ns)
    return ns


_ses_r = {"material": material()}


async def _resolver(ns, r, tope=5.0, razones=""):
    return await ns["_taller_plan_para"]("casa@iurexia.com", "1/2026", r, _ses_r, crit(),
                                         contexto="", razones_segmento=razones, tope_s=tope)


# (a) fuera de la v4: nada, y lo de la vuelta anterior se vacía
_b = FakeBase()
_mod = Modelo(PLAN_BUENO)
ns = entorno(_mod, _b)
_r = resultado(variante="v2")
_r.encargo.plan, _r.encargo.guion = {"estado": "usado"}, "GUION VIEJO"
out = asyncio.run(_resolver(ns, _r))
ok(out["estado"] == "no_aplica" and _r.encargo.plan == {} and _r.encargo.guion == ""
   and _mod.kw == [] and _b.escrituras == 0,
   "fuera de la v4: sin modelo, sin base, y el plan de la vuelta anterior no se cuela")

# (b) nada en la fila: se calcula aquí, se guarda y se usa
_r = resultado()
out = asyncio.run(_resolver(ns, _r, razones=json.dumps({"C3.b": "No vincula: es otro juicio."})))
ok(out["estado"] == "usado" and len(_mod.kw) == 1 and _r.encargo.variante_estudio == "v4",
   "sin plan en la fila: se calcula dentro de la tarea y se usa")
ok("EXPONE M1" in _r.encargo.guion and "razón del secretario para este argumento: «No vincula" in _r.encargo.guion,
   "el guion va al encargo, con la razón que el secretario escribió (Decisión 6)")
ok(_b.filas[0]["plan"]["planes"][out["clave"]]["estado"] == "listo" and _b.filas[0]["plan"]["corridas"] == 1,
   "y queda en la fila, con su corrida contada")
_fi = ns["_taller_plan_ficha"](types.SimpleNamespace(encargo=_r.encargo))
ok(_fi["estado"] == "usado" and _fi["clave"] == out["clave"] and len(_fi["segmentos"]) == 5
   and all("etiqueta" in s for s in _fi["segmentos"]), "la ficha lleva el mapa: segmento → etiqueta → razón")

# (c) la misma clave otra vez: se reutiliza sin llamar al modelo
_r2 = resultado()
out2 = asyncio.run(_resolver(ns, _r2))
ok(out2["estado"] == "usado" and out2["clave"] == out["clave"] and len(_mod.kw) == 1
   and _b.filas[0]["plan"]["corridas"] == 1, "otra generación con el mismo criterio: plan reutilizado, cero llamadas")

# (d) otro criterio, modelo lento: vence, se escribe sin plan (v3) y el plan queda para la próxima
_lento = Modelo(PLAN_BUENO, espera=0.6)
ns2 = entorno(_lento, _b)
_r3 = resultado()


async def _dos_vueltas():
    o1 = await ns2["_taller_plan_para"]("casa@iurexia.com", "1/2026", _r3, _ses_r,
                                        crit(r1=RAZON1 + " Y más."), tope_s=0.2)
    await asyncio.sleep(1.0)
    _r4 = resultado()
    o2 = await ns2["_taller_plan_para"]("casa@iurexia.com", "1/2026", _r4, _ses_r,
                                        crit(r1=RAZON1 + " Y más."), tope_s=5.0)
    return o1, o2, _r4
o1, o2, _r4 = asyncio.run(_dos_vueltas())
ok(o1["estado"] == "sin_plan" and _r3.encargo.variante_estudio == "v3" and _r3.encargo.guion == ""
   and any("SIN PLAN" in a for a in _r3.avisos), "si vence: sin plan, la v3, y se dice arriba")
ok(o2["estado"] == "usado" and len(_lento.kw) == 1,
   "…y la corrida siguió: la vuelta siguiente la usa sin volver a llamar al modelo")

# (e) dos rechazos de V0: sin plan, v3, y esa clave no se recalcula
_malo2 = Modelo(_malo)
ns3 = entorno(_malo2, _b)
_r5 = resultado()
o = asyncio.run(ns3["_taller_plan_para"]("casa@iurexia.com", "1/2026", _r5, _ses_r,
                                         crit(r2=RAZON2 + " Otra."), tope_s=5.0))
ok(o["estado"] == "sin_plan" and len(_malo2.kw) == 2 and _r5.encargo.variante_estudio == "v3",
   "V0 lo rechaza dos veces: sin plan, v3")
_r6 = resultado()
o = asyncio.run(ns3["_taller_plan_para"]("casa@iurexia.com", "1/2026", _r6, _ses_r,
                                         crit(r2=RAZON2 + " Otra."), tope_s=5.0))
ok(o["estado"] == "sin_plan" and len(_malo2.kw) == 2 and any("no pasó la validación" in a for a in o["avisos"]),
   "…y esa clave no se recalcula")

# (f) el tope de cuatro: ya van tres (b, d, e); la cuarta se lanza, la quinta no
ns4 = entorno(Modelo(PLAN_BUENO), _b)
o4 = asyncio.run(ns4["_taller_plan_para"]("casa@iurexia.com", "1/2026", resultado(), _ses_r,
                                          crit(r1="cuarta razón distinta " + RAZON1), tope_s=5.0))
o = asyncio.run(ns4["_taller_plan_para"]("casa@iurexia.com", "1/2026", resultado(), _ses_r,
                                         crit(r1="quinta razón distinta " + RAZON1), tope_s=5.0))
ok(o4["estado"] == "usado" and _b.filas[0]["plan"]["corridas"] == 4 and o["estado"] == "sin_plan"
   and any("se agotaron" in a for a in o["avisos"]), "la quinta corrida de la sesión no se lanza")

# (g) sin base (o sin la columna): se calcula igual dentro de la petición
ns5 = entorno(Modelo(PLAN_BUENO), None)
_r7 = resultado()
o = asyncio.run(ns5["_taller_plan_para"]("casa@iurexia.com", "1/2026", _r7, _ses_r, crit(), tope_s=5.0))
ok(o["estado"] == "usado" and _r7.encargo.guion, "sin la columna, el plan sirve igual a esta petición")

# (h) sin inventario: sin plan, v3, y se dice
sys.modules["inventario"] = None
_r8 = resultado()
o = asyncio.run(ns5["_taller_plan_para"]("casa@iurexia.com", "1/2026", _r8, _ses_r, crit(), tope_s=5.0))
ok(o["estado"] == "sin_plan" and _r8.encargo.variante_estudio == "v3"
   and any("inventario" in a for a in o["avisos"]), "sin el inventario de argumentos: sin plan, v3, y se dice")
sys.modules["inventario"] = _inv

# (h2) un error del proveedor: sin plan esta vez, y la vuelta siguiente reintenta
class _Caido(Modelo):
    async def _crear(self, **kw):
        self.kw.append(kw)
        raise RuntimeError("503 del proveedor")


_b3 = FakeBase()
_caido = _Caido(PLAN_BUENO)
o = asyncio.run(entorno(_caido, _b3)["_taller_plan_para"]("casa@iurexia.com", "1/2026", resultado(), _ses_r,
                                                          crit(), tope_s=5.0))
_bien = Modelo(PLAN_BUENO)
o2 = asyncio.run(entorno(_bien, _b3)["_taller_plan_para"]("casa@iurexia.com", "1/2026", resultado(), _ses_r,
                                                          crit(), tope_s=5.0))
ok(o["estado"] == "sin_plan" and o2["estado"] == "usado" and len(_bien.kw) == 1
   and _b3.filas[0]["plan"]["corridas"] == 2,
   "un 503 del proveedor no condena la clave: la vuelta siguiente la reintenta (y cuenta corrida)")

# (i) el pedido de la pantalla no espera y no recalcula
_b2 = FakeBase()
_mp = Modelo(PLAN_BUENO, espera=0.3)
ns6 = entorno(_mp, _b2)


async def _pedidos():
    arm = {"crit": crit(), "tocados": set()}
    o1 = await ns6["_taller_plan_pedido"]("casa@iurexia.com", "1/2026", resultado(), _ses_r, arm)
    o2 = await ns6["_taller_plan_pedido"]("casa@iurexia.com", "1/2026", resultado(), _ses_r, arm)
    await asyncio.sleep(0.6)
    o3 = await ns6["_taller_plan_pedido"]("casa@iurexia.com", "1/2026", resultado(), _ses_r, arm)
    return o1, o2, o3
o1, o2, o3 = asyncio.run(_pedidos())
ok(o1["estado"] == "en_curso" and o2["estado"] == "en_curso" and o3["estado"] == "listo"
   and len(_mp.kw) == 1 and o1["clave"] == o3["clave"],
   "pedir: responde al momento, no relanza la misma clave, y luego está listo")
_vista = pe.estado_para_pantalla(_b2.filas[0]["plan"], _te.huella_contraste(resultado()), __import__("time").time())
ok(_vista["estado"] == "listo" and _vista["clave"] == o1["clave"] and _vista["plan"]["segmentos"],
   "GET /taller/plan lo sirve con su clave")

if _inv_real is not None:
    sys.modules["inventario"] = _inv_real
else:
    sys.modules.pop("inventario", None)
try:
    import inventario as _inv_de_verdad  # noqa: E402
    _sg = _inv_de_verdad.segmentos(fases(), ESCRITO, False)
    ok(isinstance(_sg, list) and all({"id", "concepto", "texto", "cita", "anclas"} <= set(x) for x in _sg),
       "el inventario de verdad cumple el contrato que el plan lee")
except ImportError:
    print("      (el inventario de verdad aún no está en esta rama: se usó el doble del contrato)")

# ═══════════════════════════════════════════════════════════════════════════
print("\n14 · REVISIÓN ADVERSARIAL (26-sep-2026)")
# (a) ART. 189: «fondo por encima de los de procedimiento Y FORMA». La forma
# se contaba del lado del fondo.
_d = mutar(lambda d: (seg(d, "C2.a").update(vicio="forma"), d["unidades"].reverse(),
                      d["orden"].update(por_que="el escrito")))
ok(any("189" in x and "en la lista de unidades" in x for x in _v0(_d, c=_co, f=_fo)),
   "art. 189: una violación FORMAL antes que el fondo, sin mayor beneficio dicho, se rechaza")
# (b) Con orden «promovente» la estándar imprime los apartados por número de
# concepto: V0 revisa ESE orden, no sólo la lista de unidades.
_fq = fases(problemas=[{"pregunta": PREG1, "cubre": [2, 3], "clase": "fondo", "jerarquia": "principal"},
                       {"pregunta": PREG2, "cubre": [1], "clase": "procesal", "jerarquia": "principal"}])
_cq = [f6.Criterio(problema=PREG1, sentido="infundado", razonamiento=RAZON1, jerarquia="principal"),
       f6.Criterio(problema=PREG2, sentido="infundado", razonamiento=RAZON2, jerarquia="principal")]


def _procesal_primero(d):
    for x in ("C1.a", "C1.b"):
        seg(d, x).update(problema_id=2, vicio="procesal")
    for x in ("C2.a", "C3.a", "C3.b"):
        seg(d, x).update(problema_id=1)
    d["unidades"] = [{"id": "U1", "problemas": [1], "segmentos": ["C2.a", "C3.a", "C3.b"],
                      "premisa": "M1", "objecion": None},
                     {"id": "U2", "problemas": [2], "segmentos": ["C1.a", "C1.b"],
                      "premisa": "M2", "objecion": None}]
    d["orden"] = {"criterio": "promovente", "por_que": "el del escrito"}


_d = mutar(_procesal_primero)
_f = _v0(_d, c=_cq, f=_fq)
ok(not any("189" in x and "en la lista de unidades" in x for x in _f)
   and any("189" in x and "promovente" in x for x in _f),
   "estándar con orden «promovente»: la procesal del concepto 1 saldría antes que el fondo → se rechaza")
_d["orden"]["criterio"] = "prelacion"
ok(not any("189" in x for x in _v0(_d, c=_cq, f=_fq)),
   "…y con «prelacion» (las unidades, fondo primero) pasa")
_p_orden = pe.prompt_plan(tipo_asunto="amparo_directo", probs=pe.problemas_del_criterio(crit(), fases()),
                          segs=SEGS_N, resumen_acto="r", tramos=pe._tramos_del_escrito(fases()),
                          indice=_ind, n=3)
ok("«promovente» = los apartados siguen el orden del escrito" in _p_orden
   and "forma: vicio formal de la resolución" in _p_orden
   and "procesal o formal da mayor beneficio" in _p_orden,
   "el planificador sabe qué imprime cada orden y qué es cada vicio (descripciones, no frases)")
# (c) Art. 79: la regla de «sólo se expresa si trae beneficio» es del PENÚLTIMO
# párrafo (el último es el de las procesales y formales sin vicio de fondo).
ok("penúltimo párrafo" in pe._DESCRIBE_TRAT["no_se_expresa_art79"]
   and "art. 79, último" not in pe._DESCRIBE_TRAT["no_se_expresa_art79"],
   "art. 79: no se expresa lo suplido sin beneficio (penúltimo párrafo)")
_ds = mutar(lambda d: (d["segmentos"].append(
    {"id": "S1.a", "problema_id": 1, "vicio": "procesal", "ataca": None, "reitera": None, "dato": None,
     "etiqueta": "fundado", "razon": "fundado", "trat": "aplica", "diferencia": None, "pendiente": None}),
    [seg(d, x).update(etiqueta="fundado", razon="fundado") for x in ("C1.a", "C1.b", "C3.a", "C3.b")]))
_rs, _ = _rep(_ds, c=crit(s1="fundado"))
ok(any("S1.a" in a and "79, último párrafo" in a for a in _rs["avisos_al_secretario"]),
   "art. 79, último párrafo: suplir una procesal habiendo vicio de fondo que prospera se avisa")
# (c2) La excepción del art. 189 es una concesión DE FONDO: una de
# procedencia que prospera no deja una procesal sin estudiar en silencio.
_fpr = fases(problemas=[{"pregunta": PREG1, "cubre": [1, 3], "clase": "procedencia", "jerarquia": "principal"},
                        {"pregunta": PREG2, "cubre": [2], "clase": "procesal", "jerarquia": "accesorio"}])
_dpr = mutar(lambda d: (seg(d, "C2.a").update(etiqueta="innecesario", razon="innecesario_mayor_beneficio",
                                              trat="no_se_estudia", vicio="procesal"),
                        [seg(d, x).update(etiqueta="fundado", razon="fundado") for x in ("C1.a", "C1.b", "C3.a", "C3.b")]))
_rpr, _ = _rep(_dpr, c=crit(s1="fundado", s2="innecesario", r2="Queda sin materia."), f=_fpr)
ok(any("C2.a" in a and "74" in a for a in _rpr["avisos_al_secretario"]),
   "una procedencia que prospera no es la concesión de fondo del art. 189: la procesal sin estudio se avisa")
# (d) Un argumento del escrito no desaparece por el tratamiento.
_d = mutar(lambda d: seg(d, "C1.b").update(trat="no_se_expresa_art79"))
ok(any("C1.b" in x and "no_se_expresa_art79" in x for x in _v0(_d)),
   "un concepto del escrito no puede ir a «no se expresa» (eso es sólo lo suplido)")
_d = mutar(lambda d: seg(d, "C1.b").update(trat="no_se_estudia"))
ok(any("C1.b" in x and "no_se_estudia" in x for x in _v0(_d)),
   "ni a «no se estudia» si su criterio lo decide (infundado)")
_ci = crit(s1="fundado", s2="innecesario", r2="Queda sin materia.")
_d = mutar(lambda d: [seg(d, x).update(etiqueta="fundado", razon="fundado") for x in ("C1.a", "C1.b", "C3.a", "C3.b")])
_ri, _avi = _rep(_d, c=_ci)
ok(seg(_ri, "C2.a")["trat"] == "no_se_estudia" and any("C2.a" in a and "no se estudia" in a for a in _avi),
   "si el criterio dice «innecesario», el argumento va a «no se estudia» (organización, no sentido)")
# (e) El adhesivo con un criterio que lo estudia: aviso, no un V0 imposible.
_da = mutar(lambda d: d["segmentos"].append(dict(seg(d, "C2.a"), id="AD1.a")))
ok(not any("adhesivo_sin_materia" in x for x in _v0(_da)),
   "el adhesivo que su criterio estudia no exige adhesivo_sin_materia (V0 no puede cambiar el sentido)")
ok(any("AD1.a" in a and "182" in a for a in _rep(_da)[0]["avisos_al_secretario"]),
   "…pero se avisa (art. 182: sigue la suerte del principal)")
# (f) La remisión a la premisa que expuso OTRA unidad.


def _u3(d):
    d["unidades"][0]["segmentos"] = ["C1.a", "C1.b"]
    d["unidades"].append({"id": "U3", "problemas": [1], "segmentos": ["C3.a", "C3.b"], "premisa": "M1",
                          "objecion": None})


_p3, _ = _rep(mutar(_u3))
_g3 = pe.vista(_p3, "estandar")
_l3 = {ln.strip().split(" · ")[0]: ln for ln in _g3.splitlines()}
ok(_g3.count("EXPONE M1") == 1 and "REMITE C3.a" in _l3 and "→ apartado 1 (M1)" in _l3["REMITE C3.a"],
   "remite a la premisa que expuso OTRA unidad: con su destino, y la premisa una sola vez")
ok("premisa M1 ya expuesta en el apartado 1" in _l3.get("DESARROLLA C3.b", ""),
   "y lo que se desarrolla en otro apartado sabe que la premisa ya está expuesta")
# (g) Las marcas del plan: las unidades que prosperan y ⟦M⟧/⟦U⟧ en el bloque.
_pf, _ = _rep(mutar(lambda d: [seg(d, x).update(etiqueta="fundado", razon="fundado")
                               for x in ("C1.a", "C1.b", "C3.a", "C3.b")]), c=crit(s1="fundado"))
_gf = pe.vista(_pf, "estandar")
ok("UNIDADES QUE PROSPERAN (de ellas salen los efectos): U1 (C1.a, C1.b, C3.a, C3.b)" in _gf
   and "UNIDADES QUE PROSPERAN" not in _g, "el guion dice qué unidades prosperan (y ninguna si nada prospera)")
_bf = pe.bloque(_gf)
ok("MARCAS DEL PLAN" in _bf and "identificador M" in _bf and "identificador U" in _bf,
   "la v4 pide ⟦M⟧ donde se expone la premisa y ⟦U⟧ en cada efecto (w2_final §4.7)")
# (h) La ficha no se lleva textos largos a `estado`.
_fl = pe.para_ficha(pe.aplicar_razones(_pb, {"C3.b": "palabra " * 500}))
ok(len(seg(_fl, "C3.b")["razon_secretario"].split()) <= 60 and all(len(str(s.get("cita", "")).split()) <= 40
                                                                   for s in _fl["segmentos"]),
   "la ficha recorta la cita y la razón del secretario")
# (i) El pedido decide la variante como los gemelos.
_src_ped = ast.get_source_segment(SRC_MAIN, FN["taller_plan_pedir"])
# (integración, 26-sep-2026: con el tipo del encargo, como los gemelos, para
# que `ESTUDIO_PROMPT_AD` encienda la v4 también aquí)
ok("_taller_variante_estudio(user_email, variante_estudio, _tipo_pp)" in _src_ped
   and 'variante_estudio or "v4"' not in _src_ped
   and "encargo" in _src_ped.split("_tipo_pp =", 1)[-1].split("\n", 1)[0],
   "pedir: la variante vacía es la global de SU TIPO, como en el resolver (no «v4» por omisión)")
# (j) Las citas del escrito a renglón fijo: la del inventario (renglones
# repuestos) y la del planificador (texto crudo) valen las dos; una palabra
# cambiada, ninguna. Medido en las 8 sesiones del banco (sólo lectura): antes
# 56 de 101 citas del inventario pasaban V0; ahora 101, y 0 de 101 alteradas.
_ANCHO = 70
_frase = ("el tribunal de alzada omitio valorar las documentales que acreditan la posesiones "
          "pacifica continua publica y de buena fe sobre la parcela en litigio ") * 30
_crudo = "\n".join(_frase[i:i + _ANCHO] for i in range(0, len(_frase), _ANCHO))
_linea_partida = next(ln for ln in _crudo.split("\n") if ln and ln[-1].isalpha() and len(ln) == _ANCHO)
_ult = _linea_partida.split()[-1]
try:
    import inventario as _inv_j  # noqa: F401
    _doble_j = None
except ImportError:
    # Sin el inventario en esta rama: un doble que repone los renglones como
    # él (a columna fija, sin espacio), para probar la fontanería.
    _doble_j = types.ModuleType("inventario")
    _doble_j.normalizar_escrito = lambda t: (" ".join(
        "".join(t.split("\n")).split()), [])
    sys.modules["inventario"] = _doble_j
_tj = pe.Texto(_crudo)
_i_cr = _crudo.index(_linea_partida)
_ventana_cruda = " ".join(_crudo[_i_cr + _ANCHO - 60:_i_cr + _ANCHO + 60].split()[1:-1])
_ventana_rep = " ".join(pe._renglones_repuestos(_crudo).split())
_k = _ventana_rep.find(" ".join(_ventana_cruda.split()[1:4]))
_cita_inv = " ".join(_ventana_rep[_k:].split()[:15])
ok(_tj.contiene(_ventana_cruda, 6), "la cita copiada del texto crudo (con la palabra partida) vale")
ok(_tj.contiene(_cita_inv, 6), "la cita con los renglones repuestos (como la saca el inventario) vale")
_w = _cita_inv.split()
_w[len(_w) // 2] = "zqxjv"
ok(not _tj.contiene(" ".join(_w), 6), "…y con una palabra cambiada, ninguna de las dos lecturas la acepta")
if _doble_j is not None:
    sys.modules.pop("inventario", None)


# ═══════════════════════════════════════════════════════════════════════════
print("\n· REVISIÓN ADVERSARIAL DE LA INTEGRACIÓN (26-sep-2026): LA REGLA PROCESAL SÓLO DONDE RIGE, "
      "Y LA PARÁFRASIS SIN COMILLAS")
_REGLA = "Las VIOLACIONES PROCESALES se deciden todas"
_pp = {t: pe.prompt_plan(tipo_asunto=t, probs=pe.problemas_del_criterio(crit(), fases()), segs=SEGS_N,
                         resumen_acto="r", tramos=pe._tramos_del_escrito(fases()), indice=_ind, n=3)
       for t in ("amparo_directo", "amparo_revision", "queja")}
ok(_REGLA in _pp["amparo_directo"] and _REGLA not in _pp["amparo_revision"] and _REGLA not in _pp["queja"],
   "el prompt del planificador sólo da la regla de los arts. 74-V, 174 y 189 en el amparo directo")
# V0 (j): la procesal que el plan deja sin estudiar se rechaza en el directo (y
# sin tipo); en un amparo en revisión, no (se rige por su técnica, art. 93).
_dj = norm(mutar(lambda d: seg(d, "C2.a").update(etiqueta="fundado", razon="cae_con_principal",
                                                trat="no_se_estudia", vicio="procesal")))
_cpj = crit(s2="fundado", r2="La Sala debió estudiarla: " + RAZON2)


def _v0t(tipo):
    _p = copy.deepcopy(_dj)
    _p["tipo_asunto"] = tipo
    return [x for x in pe.validar(_p, _cpj, SEGS_N, _fp, material(), "", suplencia={})
            if "74, fracción V" in x or "189" in x]


ok(_v0t("amparo_directo") and _v0t("") and not _v0t("amparo_revision") and not _v0t("queja"),
   "V0 (j) con los arts. 74-V y 174 sólo en el amparo directo (y sin tipo, como el árbol)")
_dr = norm(mutar(lambda d: seg(d, "C2.a").update(etiqueta="innecesario", razon="cae_con_principal",
                                                trat="no_se_estudia", vicio="procesal")))


def _avj(tipo):
    _p = copy.deepcopy(_dr)
    _p["tipo_asunto"] = tipo
    _pr, _ = pe.reparar(_p, crit(s1="infundado", s2="innecesario", r2="Queda sin materia."), SEGS_N, _fp,
                        material(), "", {})
    return [a for a in _pr.get("avisos_al_secretario") or [] if "189" in a]


ok(_avj("amparo_directo") and not _avj("amparo_revision"),
   "el aviso de reparar (arts. 74-V, 174 y 189) tampoco sale en un recurso")
_gv = pe.vista({"proposiciones": [{"id": "P1", "dice": "La identidad quedó acreditada con la confesión",
                                   "caracter": "toral", "relacion": "suficiente"}],
                "premisas": [{"id": "M1", "responde_a": ["P1"], "fuentes": {"tesis": ["2001111"], "normas": []},
                              "anclas": ["identidad"]}],
                "unidades": [{"id": "U1", "segmentos": [], "premisa": "M1"}], "segmentos": []})
ok("«La identidad quedó acreditada" not in _gv and "en síntesis: La identidad quedó acreditada" in _gv
   and "no son palabras del acto" in _gv,
   "el guion da la proposición como síntesis del planificador, sin «» (que dicen «literal»)")

# ═══════════════════════════════════════════════════════════════════════════
print("\n· COMPROBACIÓN DE LA REVISIÓN (26-sep-2026): EL ORDINAL QUE NO ES DEL ESCRITO TAMPOCO "
      "LO ESCRIBE EL GUION (v4)")
# El caso del punto 7 en la v4: el contador dio n=2 y el resumen trae TRES
# párrafos sin ordinales; el inventario numera los conceptos por el orden de
# párrafos y el guion abría «concepto de violación 3».
_segs_inf = [{"id": f"C{k}.a", "concepto": k, "problema_id": 1, "etiqueta": "infundado",
              "trat": "desarrolla", "concepto_inferido": True} for k in (1, 2, 3)]


def _guion_inf(inferido: bool) -> str:
    _s = [dict(x, concepto_inferido=inferido) for x in _segs_inf]
    return pe.vista({"tipo_asunto": "amparo_directo", "n_planteamientos": 2, "segmentos": _s,
                     "problemas": [{"id": 1, "pregunta": "¿P?", "sentido": "infundado"}],
                     "unidades": [{"id": f"U{k}", "segmentos": [f"C{k}.a"]} for k in (1, 2, 3)]})


_abre_si = [x for x in _guion_inf(True).splitlines() if x.startswith("APARTADO")]
_abre_no = [x for x in _guion_inf(False).splitlines() if x.startswith("APARTADO")]
ok(_abre_si and not any("concepto de violación" in x for x in _abre_si)
   and all("abre: argumentos" in x for x in _abre_si),
   "con el concepto inferido, el apartado abre por sus argumentos, sin «concepto de violación 3»")
ok(any("abre: concepto de violación 3" in x for x in _abre_no),
   "…y con el número del escrito se sigue escribiendo (control)")
_piso_inf = pe.normalizar_segmento_piso(dict(SEGS[0], concepto_inferido=True))
ok(_piso_inf["concepto_inferido"] is True
   and pe.normalizar_segmento_piso(SEGS[0])["concepto_inferido"] is False,
   "el piso conserva la marca del inventario")
_segs_n_inf = [pe.normalizar_segmento_piso(dict(s, concepto_inferido=True)) for s in SEGS]
_pr_inf, _ = pe.reparar(norm(PLAN_BUENO), crit(), _segs_n_inf, fases(), material(), "", {})
_pr_ord, _ = pe.reparar(norm(PLAN_BUENO), crit(), SEGS_N, fases(), material(), "", {})
ok(all(s.get("concepto_inferido") for s in _pr_inf["segmentos"] if s["id"] in {x["id"] for x in SEGS})
   and not any(s.get("concepto_inferido") for s in _pr_ord["segmentos"]),
   "reparar lleva la marca del piso a los segmentos del plan (lo que lee el guion)")
_r_h = types.SimpleNamespace(fases=fases(), encargo=types.SimpleNamespace(tipo_asunto="amparo_directo"))
ok(pe.huella_entradas(_r_h, material(), "", [_piso_inf])
   == pe.huella_entradas(_r_h, material(), "", [pe.normalizar_segmento_piso(SEGS[0])]),
   "la marca no entra en la clave del plan (un plan guardado sigue sirviendo)")
_bs = pe._bloque_segmentos([_piso_inf], [], 2)
ok("orden del párrafo: el resumen no numera los conceptos" in _bs
   and "orden del párrafo" not in pe._bloque_segmentos([pe.normalizar_segmento_piso(SEGS[0])], [], 2),
   "el planificador lo recibe como dato (sólo en ese caso)")
print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
