# -*- coding: utf-8 -*-
"""La justificación estructurada de una solución (rediseño, etapa 3).

    .venv/bin/python test_justificacion.py
"""
import sys
import analisis_litis as al
import justificacion as ju

FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


ACTO = ("La Sala consideró que la actora acreditó su dedicación preponderante al hogar con la testimonial "
        "de sus hermanas y que el demandado no contestó la demanda ni ofreció pruebas para desvirtuarla. "
        "Fijó la pensión en el veinticinco por ciento de los ingresos del demandado.")
ESCRITO = "El quejoso sostiene que la testimonial es insuficiente porque las testigos son parientes de la actora."
T = al.textos(ACTO, ESCRITO, "")
CAT = {"T1": {"id": "T1", "clase": "tesis", "fuerza": "orienta", "registro": "2011111"},
       "T2": {"id": "T2", "clase": "tesis", "fuerza": "obliga", "registro": "2022222"},
       "N1": {"id": "N1", "clase": "norma", "ley": "Código Civil", "articulo": "267"},
       "O1": {"id": "O1", "clase": "propio", "neun": "30123456", "expediente": "ADC 100/2025"}}
SOL_NIEGA = {"id": "S1", "sentido_rep": "infundado", "tipo_efecto": "niega", "prospera": False}
SOL_CONCEDE = {"id": "S2", "sentido_rep": "fundado", "tipo_efecto": "para_efectos", "prospera": True}
ANALISIS = {"razones": [{"id": "R1", "relacion": "autonoma"}, {"id": "R2", "relacion": "conjunta"}]}

print("\n1 · LA FORMA Y LAS CITAS")
crudo = {
    "conclusion": {"texto": "El concepto es infundado.", "alcance": "todo el concepto"},
    "razones_a_superar": [{"razon": "r1", "caracter": "Autónoma", "como_se_supera": "no la supera: la sostiene"}],
    "regla": {"enunciado": "El parentesco no descalifica por sí solo al testigo.", "fuentes": ["T1", "T9", "2099999"],
              "fuerza": "orienta", "requisitos": ["testigos idóneos"], "excepciones": []},
    "aplicacion": [{"requisito": "testigos idóneos", "hechos": [
        {"afirma": "la actora probó la dedicación al hogar", "fuente": "acto",
         "cita": "acreditó su dedicación preponderante al hogar con la testimonial de sus hermanas",
         "condicion": "tenido_por_acreditado"},
        {"afirma": "algo inventado", "fuente": "acto",
         "cita": "el demandado confesó expresamente todos los hechos de la demanda en la audiencia",
         "condicion": "tenido_por_acreditado"}]}],
    "dificultad": {"objecion": "las testigos son parientes", "respuesta": "el parentesco no basta"},
    "efectos": [{"acto": "x", "parte": "y", "efecto": "z"}],
    "precedente_propio": [{"id": "O1", "postura": "Sigue", "requisito_o_hecho": "igual"},
                          {"id": "T1", "postura": "sigue"}],
    "resumen": " ".join(["palabra"] * 90),
}
s = ju.normalizar(crudo, SOL_NIEGA, cat=CAT, T=T)
H = s["aplicacion"][0]["hechos"]
ok(H[0]["verificada"] and H[0]["condicion"] == "tenido_por_acreditado",
   "la cita que está en el acto se verifica (misma regla que el análisis neutral)")
ok(not H[1]["verificada"] and H[1]["condicion"] == "sin_verificar",
   "la cita inventada no pasa y su condición baja a «sin verificar»")
ok(s["regla"]["fuentes"] == ["T1"] and s["ids_quitados"] == 1,
   "sólo fuentes del catálogo cerrado: T9 se quita; un registro escrito a mano no es fuente")
ok(s["razones_a_superar"][0]["razon"] == "R1" and s["razones_a_superar"][0]["caracter"] == "autonoma",
   "la razón a superar se normaliza (id y carácter)")
ok(s["efectos"] == [], "una solución que no prospera no lleva efectos de concesión")
ok([p["id"] for p in s["precedente_propio"]] == ["O1"] and s["precedente_propio"][0]["postura"] == "sigue"
   and s["precedente_propio"][0]["neun"] == "30123456",
   "precedente propio: sólo ids O del catálogo, con su NEUN puesto por código")
ok(len(s["resumen"].split()) == ju.MAX_RESUMEN, "resumen de 60 palabras como máximo")

print("\n2 · LA REVISIÓN POR CÓDIGO (sólo avisa)")
rv = ju.revisar(s, analisis=ANALISIS, cat=CAT)
ok(rv["estado"] == "con_pendientes" and any("sin verificar" in a for a in rv["avisos"]),
   "un hecho sin verificar deja la solución «con pendientes», no la borra")
s2 = ju.normalizar({"conclusion": "Es fundado.", "regla": {"enunciado": "x", "fuentes": ["T1"], "fuerza": "obligatoria"},
                    "razones_a_superar": [{"razon": "R2", "como_se_supera": "se combate"}]},
                   SOL_CONCEDE, cat=CAT, T=T)
rv2 = ju.revisar(s2, analisis=ANALISIS, cat=CAT)
ok(any("autónoma" in a and "R1" in a for a in rv2["avisos"]),
   "la que prospera sin decir cómo supera la razón autónoma R1: AVISO (el diagnóstico midió que como filtro rompe concesiones buenas)")
ok(any("obligatoria" in a for a in rv2["avisos"]), "se presenta como obligatoria con una fuente que sólo orienta")
ok(any("efectos" in a for a in rv2["avisos"]), "concede sin decir sus efectos (art. 77)")
ok(any("objeción" in a for a in rv2["avisos"]), "no enfrenta la objeción más fuerte")
rv3 = ju.revisar(ju.normalizar({}, SOL_NIEGA, cat=CAT, T=T))
ok(rv3["estado"] == "incompleta" and "conclusión" in rv3["faltan"] and "regla" in rv3["faltan"],
   "vacía: «incompleta», con lo que falta dicho")

print("\n3 · NUNCA LANZA")
for raro in (None, [], "texto", {"aplicacion": "x", "regla": 5, "conclusion": 3, "pendientes": [1, None]}):
    try:
        x = ju.normalizar(raro, SOL_CONCEDE, cat=CAT, T=T)
        ju.revisar(x, analisis={"razones": "x"}, cat=CAT)
        bien = True
    except Exception as e:
        bien = False
    ok(bien, f"forma rara del modelo: {type(raro).__name__}")

print()
if FALLOS:
    print("FALLAS:\n  " + "\n  ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
