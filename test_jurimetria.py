# -*- coding: utf-8 -*-
"""EL ATAJO DE UN SOLO CLIC (botón amarillo «Genera todo el proyecto»).

David: «una vez que se tenga los documentos, el proyecto se genere solo conforme
lo que considere acertado (…) con sentido decidido por el motor».

Dos cosas que se comprueban aquí, y las dos ya habían fallado en vivo:

 1 · que `repartir` en modo ACERVO tome el sentido de las PROPUESTAS. Es la
     rama `else` del módulo, sin nombre propio, y por eso es fácil de romper
     sin darse cuenta.
 2 · que las DOS puertas —el resolvedor de streaming y su gemelo— entiendan la
     misma palabra. Antes no: el gemelo atendía `usar_propuesta`, el de
     streaming lo declaraba y lo tiraba en silencio, y `modo_decision=acervo`
     no existía en ninguno pese a figurar en la firma del formulario. El mismo
     asunto se resolvía o no según qué endpoint tocaras.
"""
import io, re, sys
import modos_decision as _md

fallos = []

# ── 1 · el reparto sale de las propuestas ──────────────────────────────────
problemas = [{"pregunta": "¿La medida cautelar se rige por el CPC?",
              "jerarquia": "principal"},
             {"pregunta": "¿Se omitió el interés superior del menor?",
              "jerarquia": "accesorio"}]
propuestas = [{"problema": "¿La medida cautelar se rige por el CPC?",
               "sentido": "infundado", "razon": "El art. 202 la autoriza.",
               "alcanza": True},
              {"problema": "¿Se omitió el interés superior del menor?",
               "sentido": "fundado", "razon": "No se razonó el interés.",
               "alcanza": True}]
rep, _av = _md.repartir(problemas, _md.ACERVO, "", propuestas, {})
if len(rep) != 2:
    fallos.append(f"ACERVO devolvió {len(rep)} criterios, esperaba 2")
else:
    if rep[0]["sentido"] != "infundado":
        fallos.append(f"el principal salió {rep[0]['sentido']!r}, no del motor")
    if rep[1]["sentido"] != "fundado":
        fallos.append(f"el accesorio salió {rep[1]['sentido']!r}, no del motor")
    if "art. 202" not in rep[0]["razonamiento"]:
        fallos.append("la razón del motor no viajó al criterio")
    if rep[0]["jerarquia"] != "principal":
        fallos.append("se perdió la jerarquía en el reparto")

# NO INVENTA SENTIDO donde el motor no propuso: un problema sin propuesta sale
# vacío y el endpoint lo descarta, en vez de rellenarlo con cualquier cosa.
rep2, _ = _md.repartir(problemas + [{"pregunta": "¿Tercero sin propuesta?"}],
                       _md.ACERVO, "", propuestas, {})
if str(rep2[2].get("sentido") or "").strip():
    fallos.append("ACERVO inventó un sentido para un problema sin propuesta")

# ── 2 · las dos puertas entienden la misma palabra ─────────────────────────
src = io.open("main.py", encoding="utf-8").read()
# Cada resolvedor declara su formulario con `modo_decision`; se cuentan las
# ramas que lo comparan con "acervo".
ramas = len(re.findall(r'modo_decision\s*or\s*""\)\.strip\(\)\.lower\(\)\s*==\s*"acervo"', src))
if ramas < 2:
    fallos.append(f"sólo {ramas} endpoint(s) atienden modo_decision=acervo; "
                  f"hacen falta 2 (streaming y gemelo). Es el fallo de "
                  f"«arreglado en un camino y no en el otro».")

# Y el parámetro muerto: si `usar_propuesta` se declara, alguien tiene que
# leerlo. Se declaraba dos veces y sólo se leía una.
declara = len(re.findall(r'usar_propuesta:\s*bool\s*=\s*Form', src))
lee = len(re.findall(r'elif[^\n]*usar_propuesta', src))
if lee < declara:
    fallos.append(f"`usar_propuesta` se declara {declara} veces y se lee {lee}: "
                  f"hay un endpoint que se lo traga en silencio")

if fallos:
    print("FALLOS:")
    for f in fallos:
        print("  ·", f)
    sys.exit(1)
print("OK · jurimetría: reparto desde las propuestas y las dos puertas la entienden")
