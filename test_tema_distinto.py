# -*- coding: utf-8 -*-
"""El tema distinto no se declara innecesario.

Medido en el ADR 93/2026: el motor marcó `tema_distinto` en el planteamiento
de los alegatos, escribió su suerte —«fundado; deberán analizarse en la nueva
sentencia»— y la sustracción de materia lo declaró innecesario tres renglones
más abajo. El proyecto habría dicho que no hace falta estudiar lo que su propio
análisis mandaba estudiar.

La prueba se calibra en los dos sentidos: si el escape nuevo también salvara al
accesorio corriente, la sustracción de materia dejaría de existir y eso sería
un fallo peor que el que se arregla.
"""
import modos_decision as md

P1 = "¿La Sala debió admitir la ampliación de demanda presentada por la quejosa?"
P2 = ("¿La Sala debió estudiar los alegatos presentados por la quejosa el "
      "veintiocho de agosto de dos mil veinticinco?")
P3 = "¿Procedía condenar en costas a la autoridad demandada?"

PROBLEMAS = [{"pregunta": P1, "jerarquia": "principal"},
             {"pregunta": P2, "jerarquia": "accesorio"},
             {"pregunta": P3, "jerarquia": "accesorio"}]

PROPUESTAS = [{"problema": P1, "sentido": "fundado", "razon": "…", "alcanza": True},
              {"problema": P2, "sentido": "fundado", "razon": "…", "alcanza": True}]

fallos = []


def dice(cond, que):
    if not cond:
        fallos.append(que)


# ── 1 · la lectura de la lista de comprobación ──────────────────────────
lista = [{"numero": 1, "tema": "Admisión de la ampliación", "papel": "principal",
          "tema_distinto": False},
         {"numero": 2, "tema": "Omisión de estudiar los alegatos",
          "papel": "accesorio", "tema_distinto": True},
         {"numero": 3, "tema": "Costas", "papel": "accesorio",
          "tema_distinto": False}]
d = md.temas_distintos_de(lista, PROBLEMAS)
dice(d == {P2}, f"por número: se esperaba {{P2}} y salió {d}")

# Sin número, por el texto del problema.
lista_sin_num = [{"tema": P2[:60], "papel": "accesorio", "tema_distinto": True}]
d2 = md.temas_distintos_de(lista_sin_num, PROBLEMAS)
dice(d2 == {P2}, f"por texto: se esperaba {{P2}} y salió {d2}")

# Una lista vacía no marca nada.
dice(md.temas_distintos_de([], PROBLEMAS) == set(), "lista vacía marcó algo")

# ── 2 · el reparto respeta el tema distinto ─────────────────────────────
rep, av = md.repartir(PROBLEMAS, md.GLOBAL, "fundado", PROPUESTAS, {},
                      global_dictado=True, temas_distintos={P2})
por = {x["problema"]: x["sentido"] for x in rep}
dice(por.get(P2) != md.INNECESARIO,
     f"el tema distinto se declaró {por.get(P2)}")
dice(por.get(P3) == md.INNECESARIO,
     f"CALIBRACIÓN: el accesorio corriente debía quedar innecesario y quedó "
     f"{por.get(P3)}; si nada queda innecesario, la sustracción murió")
dice(any("TEMA DISTINTO" in a for a in av),
     "no se avisó por qué no se declaró innecesario")

# ── 3 · sin el dato, todo sigue como antes ──────────────────────────────
rep0, _ = md.repartir(PROBLEMAS, md.GLOBAL, "fundado", PROPUESTAS, {},
                      global_dictado=True)
por0 = {x["problema"]: x["sentido"] for x in rep0}
dice(por0.get(P2) == md.INNECESARIO and por0.get(P3) == md.INNECESARIO,
     "sin temas_distintos el comportamiento anterior cambió")

# ── 4 · lo que marcó el secretario sigue mandando ───────────────────────
rep2, _ = md.repartir(
    PROBLEMAS, md.GLOBAL, "fundado", PROPUESTAS,
    {P3: {"sentido": "infundado", "razonamiento": "porque no se acreditó"}},
    global_dictado=False, temas_distintos={P2})
por2 = {x["problema"]: x["sentido"] for x in rep2}
dice(por2.get(P3) == "infundado",
     f"la calificación del secretario se perdió: {por2.get(P3)}")

print("\n".join("  ✗ " + f for f in fallos) if fallos
      else "OK · tema distinto: 4 comprobaciones")
raise SystemExit(1 if fallos else 0)
