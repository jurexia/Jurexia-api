# -*- coding: utf-8 -*-
"""Los preceptos que nombra la parte entran al prompt del estudio aunque hayan
llegado al material después de los doce primeros (humo del AR 631/2025,
29-sep-2026; bandera «normas_al_documento»).

    .venv/bin/python test_normas_nombradas.py
"""
import sys
import types
import contexto_taller as ct
import fase6_estudio as f6

FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


def norma(ley, art):
    return {"cuerpo_legal": ley, "articulo": art, "texto": f"Artículo {art}. …"}


RELLENO = [norma("Código Civil Federal", 2400 + i) for i in range(12)]
AL_FINAL = [norma("Código Civil del Estado de Querétaro", 2284), norma("Código Civil del Estado de Querétaro", 2294),
            norma("Código de Procedimientos Civiles del Estado de Querétaro", 49),
            norma("Código Civil del Estado de Querétaro", 49),
            norma("Código de Procedimientos Civiles del Estado de Querétaro", 942)]
INV = [{"texto": "Aduce que el artículo 49 de la Ley Adjetiva Civil para el Estado de Querétaro permite que el "
                 "causahabiente sustituya a quien transmitió el derecho.", "cita": ""},
       {"texto": "Invoca los artículos 2284 y 2294 del Código Civil para el Estado de Querétaro.", "cita": ""}]
m = types.SimpleNamespace(normas=RELLENO + AL_FINAL, inventario=INV)

print("\n1 · SIN LA BANDERA: LAS DOCE DE SIEMPRE")
ct.poner(False, {})
ok(f6._normas_para_el_estudio(m) == RELLENO, "una cuenta de fuera recibe exactamente las doce primeras")

print("\n2 · CON LA BANDERA")
ct.poner(True, {}, pruebas=True)
ns = f6._normas_para_el_estudio(m)
extra = [(n["cuerpo_legal"], n["articulo"]) for n in ns[12:]]
ok(ns[:12] == RELLENO, "las doce de siempre se quedan como estaban (suma, no sustituye)")
ok(("Código de Procedimientos Civiles del Estado de Querétaro", 49) in extra
   and ("Código Civil del Estado de Querétaro", 49) not in extra,
   "«artículo 49 de la Ley Adjetiva Civil» trae el 49 del PROCESAL, no el del civil (el nombre se canoniza)")
ok(("Código Civil del Estado de Querétaro", 2284) in extra and ("Código Civil del Estado de Querétaro", 2294) in extra,
   "la lista «2284 y 2294 del Código Civil…» trae los dos")
ok(("Código de Procedimientos Civiles del Estado de Querétaro", 942) not in extra,
   "lo que nadie nombra se queda fuera")
m2 = types.SimpleNamespace(normas=RELLENO + [norma("Código Civil del Estado de Querétaro", 3000 + i) for i in range(10)],
                           inventario=[{"texto": " ".join(f"artículo {3000 + i} del Código Civil para el Estado de "
                                                           f"Querétaro," for i in range(10))}])
ok(len(f6._normas_para_el_estudio(m2)) == 12 + f6.NORMAS_NOMBRADAS_EXTRA, "con tope de seis de más")
ok(f6._normas_para_el_estudio(types.SimpleNamespace(normas=RELLENO + AL_FINAL, inventario=None)) == RELLENO
   and f6._normas_para_el_estudio(types.SimpleNamespace(normas=None, inventario=INV)) == [],
   "sin inventario o sin normas, nada raro (nunca lanza)")
ct.poner(False, {})

print("\n3 · LA SEGUNDA LEY DE UNA LISTA, CON SU NOMBRE ENTERO")
_t = ("En cuanto a los artículos 14 y 16 de la Constitución Política de los Estados Unidos Mexicanos y 49, 57 y 279 "
      "del Código de Procedimientos Civiles del Estado de Querétaro, su invocación no cambia nada.")
_c = dict((n, c) for n, c in f6.citas_de_articulos(_t))
ok(_c.get("49") == _c.get("279") and "del Estado de Querétaro" in (_c.get("49") or "")
   and _c.get("14", "").startswith("de la Constitución"),
   "«…Mexicanos y 49, 57 y 279 del Código de Procedimientos Civiles del Estado de Querétaro»: la ventana corta ya "
   "no deja la ley en «…Civile» (el verificador acusaba como ausentes tres artículos del material)")
_mat = types.SimpleNamespace(normas=[norma("Código de Procedimientos Civiles del Estado de Querétaro", a)
                                     for a in (49, 57, 279)] + [norma("Constitución Política de los Estados Unidos "
                                                                      "Mexicanos", a) for a in (14, 16)])
ok(not f6.preceptos_fuera(_t, _mat)[0], "y el verificador ya no los da por ausentes")

print()
if FALLOS:
    print("FALLAS:\n  " + "\n  ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
