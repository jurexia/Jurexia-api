"""El barrido final contra la cita inventada — 23-sep-2026.

David: «un barrido inteligente a todo el proyecto al terminar para verificar
que no está inventando artículos, con una consulta sumamente rápida por
internet».

CALIBRADO CONTRA LA v6 REAL DEL ADC 93/2026 (43 citas distintas), seis
corridas: 0 acusaciones falsas, 4 de 6 citas inventadas cazadas, 6-14 s. Las
pruebas de aquí no llaman a internet: inyectan la respuesta.

    .venv/bin/python test_barrido_preceptos.py
"""
import asyncio
import inspect
import os

import barrido_preceptos as bp

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


print("\n1 · EL NOMBRE DE LA LEY TERMINA DONDE EMPIEZA EL VERBO")
ok(bp.limpia_ley("Ley Federal de Procedimiento Contencioso Administrativo establecía la")
   == "Ley Federal de Procedimiento Contencioso Administrativo", "«…Administrativo establecía la»")
ok(bp.limpia_ley("Código Fiscal de la Federación regulaban el recurso") == "Código Fiscal de la Federación",
   "«…de la Federación regulaban el recurso»")
ok(bp.limpia_ley("Ley de Amparo deriva que la privación") == "Ley de Amparo", "«Ley de Amparo deriva…»")
ok(bp.limpia_ley("Constitución Política de los Estados Unidos Mexicanos protege las formalidades")
   == "Constitución Política de los Estados Unidos Mexicanos", "«…Mexicanos protege las…»")
for entero in ("Ley de Ciencia y Tecnología",
               "Ley de Adquisiciones, Arrendamientos y Servicios del Sector Público",
               "Ley de Amparo, Reglamentaria de los Artículos 103 y 107 de la Constitución",
               "Ley Orgánica del Tribunal Federal de Justicia Administrativa"):
    ok(bp.limpia_ley(entero) == entero, f"y NO corta un título que lleva «y» dentro: «{entero[:40]}…»")

print("\n2 · LAS CITAS DE UN RUBRO NO SON DEL PROYECTO")
ok(bp._es_de_rubro("CÓDIGO FISCAL DE LA FEDERACIÓN PARA AMPLIARLA"), "un rubro va en mayúsculas")
ok(not bp._es_de_rubro("Código Fiscal de la Federación"), "y una cita del proyecto no")
_con_rubro = ('Sirve de apoyo la jurisprudencia de rubro «DEMANDA DE NULIDAD. ES OBLIGACIÓN DE LA SALA '
              'OTORGAR EL TÉRMINO QUE ESTABLECE EL ARTÍCULO 210 DEL CÓDIGO FISCAL DE LA FEDERACIÓN '
              'PARA AMPLIARLA.» Conforme al artículo 17 de la Ley Federal de Procedimiento '
              'Contencioso Administrativo, la ampliación procede.')
_p = bp.pares_citados(_con_rubro)
ok([n for _, n in _p] == ["17"], f"del rubro no se pregunta nada; de la prosa sí: {_p}")

print("\n3 · LO QUE EL ACERVO YA VERIFICÓ NO SE PREGUNTA")
class _M:
    normas = [{"cuerpo_legal": "Ley Federal de Procedimiento Contencioso Administrativo",
               "articulo": "17", "texto": "ARTÍCULO 17. Se podrá ampliar la demanda…"}]
ok(bp.por_preguntar(_con_rubro, _M()) == [], "el 17 de la LFPCA está en el material: no se pregunta")
ok(len(bp.por_preguntar(_con_rubro, None)) == 1, "sin material, sí se pregunta")

print("\n4 · NUNCA SE ACUSA CON UNA SOLA RESPUESTA")
TXT = "Conforme al artículo 884 de la Ley de Amparo y al artículo 17 de la Ley de Amparo."

async def _primera(pares, seg):
    return [{"n": i, "existe": False if n == "884" else True}
            for i, (_, n) in enumerate(pares, 1)]



async def _si(pares, seg):
    return set(pares)


async def _no(pares, seg):
    return set()

r = asyncio.run(bp.barrer(TXT, None, preguntar=_primera, confirmar=_si))
ok([x["articulo"] for x in r["inexistentes"]] == ["884"], "confirmado por la segunda: se acusa")
r2 = asyncio.run(bp.barrer(TXT, None, preguntar=_primera, confirmar=_no))
ok(r2["inexistentes"] == [] and [x["articulo"] for x in r2["sin_comprobar"]] == ["884"],
   "NO confirmado: baja a «sin comprobar», no se acusa")
ok("DOS VECES" in " ".join(r["avisos"]), "y el aviso dice que se preguntó dos veces")

print("\n5 · NUNCA LANZA, Y CALLA CUANDO NO PUEDE")
async def _revienta(pares, seg):
    raise RuntimeError("sin red")
r3 = asyncio.run(bp.barrer(TXT, None, preguntar=_revienta))
ok(r3["inexistentes"] == [] and len(r3["sin_comprobar"]) == 2,
   "si la consulta falla, no se acusa a nadie y se dice que no se comprobó")
ok(asyncio.run(bp.barrer("", None))["preguntados"] == 0, "sin citas no hay nada que preguntar")

print("\n6 · LA ARITMÉTICA DE LA SEGUNDA PREGUNTA")
ok(bp._fuera_de_rango("884", 271) is True, "884 en una ley de 271: fuera de rango")
ok(bp._fuera_de_rango("271", 271) is False and bp._fuera_de_rango("290", 271) is False,
   "con margen: el último y un poco más, dentro (bis, reformas)")
ok(bp._fuera_de_rango("5", None) is False and bp._fuera_de_rango("5", 10) is False,
   "sin dato o con leyes cortas, no se juzga")

print("\n7 · LA PUERTA ESTÁ CONECTADA")
import redactor_adelanto as ra
ok("_bp.barrer(_plano, material)" in inspect.getsource(ra._terminar),
   "_terminar barre el documento compuesto entero")
ok(bp.BARRIDO_MODELO == "x-ai/grok-4.3",
   f"el motor por omisión es el que ganó la medición ({bp.BARRIDO_MODELO})")
ok("BARRIDO_PRECEPTOS" in open("barrido_preceptos.py", encoding="utf-8").read(),
   "y tiene interruptor para apagarlo desde Render")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
