"""La Cronología del expediente, sin red — 12-oct-2026.

    .venv/bin/python test_cronologia_expediente.py

Lo que importa: que un hecho sólo entre si su cita y su fecha están en la
página que dice, que la página 37 siga siendo la 37 aunque haya páginas en
blanco, que el cobro sea ⌈páginas/N⌉ con mínimo 1, y que un plazo no se
calcule sobre una fecha que el expediente no dice.
"""
import asyncio
import datetime as dt
import sys

import cobro_consultas as cc
import cronologia_expediente as cx

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


print("── la regla de consultas ──")
ok(cc.consultas_por_paginas(0, 10) == 1, "0 páginas → 1 consulta (mínimo)")
ok(cc.consultas_por_paginas(1, 10) == 1, "1 página → 1")
ok(cc.consultas_por_paginas(10, 10) == 1, "10 páginas → 1")
ok(cc.consultas_por_paginas(11, 10) == 2, "11 páginas → 2")
ok(cc.consultas_por_paginas(500, 10) == 50, "500 páginas → 50")
ok(cc.consultas_por_paginas(500, 20) == 25, "con N=20, 500 páginas → 25")

print("── las páginas conservan su número ──")
texto = "[[PÁGINA 1]]\nuno\n\n[[PÁGINA 3]]\ntres\n\n[[PÁGINA 4]]\ncuatro"
p = cx.partir_por_rotulos(texto, 5)
ok(p == ["uno", "", "tres", "cuatro", ""], "la página en blanco no corre la numeración")
tramos = "[[PÁGINA 1]]\na\n\n[[PÁGINA 2]]\nb\n\n[[PÁGINA 1]]\nc\n\n[[PÁGINA 2]]\nd"
ok(cx.partir_por_rotulos(tramos, 4) == ["a", "b", "c", "d"], "los tramos de Azure que reinician en 1 se suman")
ok(cx.partir_por_rotulos("texto sin rótulos", 3) is None, "sin rótulos → None (no se inventan páginas)")

print("── la cita literal ──")
pag = cx.Pagina(0, 7, "En la Ciudad de México, a quince de marzo de dos mil veinticuatro.\n"
                      "Agréguese a sus autos el escrito de cuenta y se tiene por presen-\n"
                      "tada la contestación de la demanda en tiempo y forma.")
ok(cx.cita_en_pagina("se tiene por presentada la contestación de la demanda en tiempo y forma", pag) == "exacta",
   "la palabra partida con guion al final de la línea coincide")
ok(cx.cita_en_pagina("SE TIENE POR PRESENTADA LA CONTESTACIÓN DE LA DEMANDA", pag) == "exacta",
   "mayúsculas y acentos no importan")
ok(cx.cita_en_pagina("se tiene por presentada la reconvención de la parte actora", pag) is None,
   "una cita que no está no pasa")
ok(cx.cita_en_pagina("la demanda", pag) is None, "una cita de dos palabras no prueba nada")
ok(15 in [d for (_a, _m, d) in pag.fechas] and (2024, 3, 15) in pag.fechas, "la fecha en letra se lee")

print("── verificar un lote ──")
docs = [{"nombre": "exp.pdf", "total": 4, "paginas": [
    "Demanda. Ciudad de México, a 10 de enero de 2024. Comparece Juan Pérez García a demandar "
    "de Inmobiliaria del Valle, S.A. de C.V., el cumplimiento del contrato de arrendamiento.",
    "",
    "Ciudad de México, a quince de marzo de dos mil veinticuatro. Se tiene por presentada la "
    "contestación de la demanda en tiempo y forma. Se concede a la parte actora el término de "
    "tres días para que manifieste lo que a su derecho convenga.",
    "RAZÓN. El dieciocho de marzo de dos mil veinticuatro se notificó personalmente el acuerdo "
    "que antecede a la parte actora."]}]
indice = {(0, i): cx.Pagina(0, i, t) for i, t in enumerate(docs[0]["paginas"], 1)}
ext = {"hechos": [
    {"fecha": "2024-01-10", "hecho": "Juan Pérez presenta la demanda", "pagina": 1,
     "cita": "Comparece Juan Pérez García a demandar de Inmobiliaria del Valle"},
    # El modelo se equivoca de página: la cita está en la 3. Se corrige.
    {"fecha": "2024-03-15", "hecho": "Se tiene por contestada la demanda", "pagina": 1,
     "cita": "Se tiene por presentada la contestación de la demanda en tiempo y forma"},
    # Inventado: la cita no está en ninguna página.
    {"fecha": "2024-02-01", "hecho": "Emplazamiento", "pagina": 1,
     "cita": "se emplazó a la demandada en su domicilio con las formalidades de ley"},
    # Cita verdadera, fecha que no consta.
    {"fecha": "2024-03-20", "hecho": "Contestación", "pagina": 3,
     "cita": "Se tiene por presentada la contestación de la demanda en tiempo y forma"},
    {"fecha": None, "hecho": "sin fecha", "pagina": 3, "cita": "x"}],
    "plazos": [
    {"acto": "Manifestar", "parte": "actora", "cantidad": 3, "unidad": "dias", "tipo_dias": "habiles",
     "texto_literal": "Se concede a la parte actora el término de tres días para que manifieste",
     "pagina": 3, "fecha_acuerdo": "2024-03-15", "fecha_notificacion": "2024-03-18",
     "tipo_notificacion": "personal"},
    {"acto": "Pruebas", "cantidad": 10, "unidad": "dias",
     "texto_literal": "Se concede a la parte actora el término de tres días para que manifieste",
     "pagina": 3}]}
descartes = {"hechos": 0, "cuestiones": 0, "plazos": 0, "motivos": {}}
v = cx.verificar_lote(ext, 0, [1, 3, 4], indice, descartes)
ok(len(v["hechos"]) == 2, f"entran 2 hechos de 5 (entraron {len(v['hechos'])})")
ok(any(h["pagina"] == 3 and h["pagina_corregida"] for h in v["hechos"]), "la página equivocada se corrige a la 3")
ok(descartes["motivos"].get("hechos:cita_no_esta_en_la_pagina") == 1, "el emplazamiento inventado se descarta")
ok(descartes["motivos"].get("hechos:fecha_no_consta") == 1, "la fecha que no consta se descarta")
ok(descartes["motivos"].get("hechos:sin_fecha") == 1, "el hecho sin fecha se descarta")
ok(len(v["plazos"]) == 1 and descartes["motivos"].get("plazos:numero_no_coincide") == 1,
   "el plazo cuyo número no coincide con la cita se descarta")

print("── el plazo se computa sólo con fecha que consta ──")
regla = lambda f, m, e, t: ("CNPCF", "prueba")
pl = cx.computar_plazos(v["plazos"], {"materia": "civil", "entidad": "Ciudad de México"}, indice, docs,
                        regla, dt.date(2024, 3, 19))
ok(pl and pl[0]["fecha_base"] == "2024-03-18", "toma la fecha de notificación que sí consta (pág. 4)")
ok(pl and pl[0]["calculo"] and pl[0]["estado"] == "corre", "con vencimiento y corriendo")
falso = [{**v["plazos"][0], "fecha_notificacion": "2024-03-25", "fecha_acuerdo": "2024-03-26"}]
pl2 = cx.computar_plazos(falso, {}, indice, docs, regla, dt.date(2024, 3, 19))
ok(pl2 and pl2[0]["calculo"] is None and pl2[0]["estado"] == "sin_computo",
   "con fechas que no constan no hay vencimiento")

print("── lotes y repetidos ──")
largo = {"nombre": "a.pdf", "paginas": ["texto de la página número %d " % i * 5 for i in range(1, 31)]}
lotes = cx.armar_lotes([largo, {"nombre": "b.pdf", "paginas": ["", "contenido suficiente aquí"]}], 12, 40000)
ok([len(x[1]) for x in lotes] == [12, 12, 6, 1], f"12+12+6 de a.pdf y 1 de b.pdf ({[len(x[1]) for x in lotes]})")
ok(lotes[1][2] == 12 and lotes[0][2] is None, "cada lote lleva la última página del anterior como contexto")
ok(lotes[3][0] == 1 and lotes[3][1] == [2], "los documentos no se mezclan y la página vacía no viaja")
h = {"fecha": "2024-03-15", "hecho": "Se tiene por contestada la demanda", "documento": 0, "pagina": 3,
     "actor": "Juzgado", "relevancia": "media", "cita": "Se tiene por presentada la contestación"}
dd = cx.deduplicar_hechos([h, {**h, "relevancia": "alta"}, {**h, "fecha": "2024-03-16"}])
ok(len(dd) == 2 and dd[0]["relevancia"] == "alta", "el mismo hecho dos veces queda una, con la relevancia mayor")
a = {"fecha": "2026-03-13", "hecho": "La actora presentó escrito ofreciendo pruebas documentales y pericial",
     "documento": 0, "pagina": 45, "actor": "Constructora Albarrán Hermanos, S.A. de C.V.", "relevancia": "media",
     "cita": "ofrece las siguientes pruebas documentales"}
b = {**a, "hecho": "La demandada presentó escrito ofreciendo pruebas documentales y pericial", "pagina": 46,
     "actor": "Grupo Hotelero Puerta del Sol, S.A. de C.V.", "cita": "ofrece las pruebas de su intención"}
ok(len(cx.deduplicar_hechos([a, b])) == 2, "actora y demandada el mismo día y en páginas contiguas no se funden")


print("── agrupar cuestiones sin modelo ──")


class _CliRoto:
    async def post(self, *a, **k):
        raise RuntimeError("sin red")


cs = [{"cuestion": "Si hubo pago", "sostiene": "actora", "postura": "no pagó", "contraparte": "",
       "documento": 0, "pagina": 3, "cita": "x"}]
cx_contador = cx.Contador()
g = asyncio.run(cx.agrupar_cuestiones(_CliRoto(), cs, cx_contador))
ok(len(g) == 1 and g[0]["citas"][0]["pagina"] == 3, "si el modelo falla, las cuestiones van sin agrupar")

print()
if FALLOS:
    print(f"✗ {len(FALLOS)} fallo(s)")
    sys.exit(1)
print("✓ todo pasa")
