"""Texto de norma, no una respuesta sobre ella — revisión fiscal 2/2026, 17-sep-2026.

La nota al pie del artículo 150 del Reglamento Interior del IMSS transcribió la
negativa del buscador web: «El **artículo 150** que corresponde al… no puedo
citarlo textualmente». La prueba positiva se calibró sobre 15,885 artículos
reales de 16 colecciones de leyes: 4 falsos rechazos (0.025 %), todos formato
raro del propio acervo, y ningún texto de chat dejado pasar.

    .venv/bin/python test_texto_normativo.py
"""
import inspect

import texto_normativo as tn

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


print("\n1 · LO QUE NO ES NORMA")
NOTA_2_2026 = (
    "El **artículo 150** que corresponde al **Reglamento Interior del Instituto Mexicano del "
    "Seguro Social (IMSS)** establece las **atribuciones de las subdelegaciones** dentro de su "
    "circunscripción territorial. La fuente oficial que indicas en los resultados es el documento "
    "alojado en **Orden Jurídico**, pero **no es la publicación oficial primaria del IMSS ni del "
    "DOF**; además, en los resultados proporcionados el texto aparece **parcialmente recortado**, "
    "así que no puedo citarlo *textualmente y completo con todas sus fracciones* con total "
    "seguridad solo a partir de estos fragmentos.")
ok(not tn.es_texto_normativo(NOTA_2_2026)[0], "la nota real del 2/2026")
ok(not tn.es_texto_normativo(NOTA_2_2026.replace("*", ""))[0],
   "…y aunque le quiten los asteriscos (habla un asistente)")
for t, que in (("No localizado. No encontré el texto del artículo 45.", "«No localizado»"),
               ("Aquí tienes el artículo 17 de la Ley del Seguro Social: Al dar los avisos…", "«Aquí tienes»"),
               ("Este artículo establece las atribuciones de las subdelegaciones.", "se describe"),
               ("El artículo 150 que corresponde al Reglamento regula a las subdelegaciones.", "«que corresponde al»")):
    ok(not tn.es_texto_normativo(t)[0], que)

print("\n2 · LO QUE SÍ ES NORMA (formas reales del acervo)")
for t, que in (
        ("Artículo 150. Son atribuciones de las subdelegaciones, dentro de su circunscripción "
         "territorial: I. Registrar a los patrones y demás sujetos obligados.", "artículo con su rótulo"),
        ("[Ley del Seguro Social | CAPITULO I] Artículo 17. Al dar los avisos a que se refiere la "
         "fracción I del artículo 15 de esta Ley, el patrón puede expresar por escrito los motivos.", "con migaja"),
        ("#### Artículo 238. Además de la pena señalada en el artículo anterior, se impondrá de seis "
         "a diez años de prisión.", "encabezado #### de la ingesta de la CDMX"),
        ("La Educación impartida en el estado se basará en los resultados del progreso científico.",
         "«en los resultados» dicho por una ley"),
        ("Las personas morales determinarán el remanente con los datos proporcionados por sus "
         "integrantes.", "«proporcionados por» dicho por una ley"),
        ("- TRANSITORIO --- ÚNICO.- El presente Decreto entrará en vigor al día siguiente.",
         "transitorio con guiones del acervo")):
    ok(tn.es_texto_normativo(t)[0], que)

print("\n3 · LAS DOS PUERTAS")
import busqueda_web as bw
import documento_generado as dg
ok("_tn.es_texto_normativo(texto)" in inspect.getsource(bw.texto_de_articulo),
   "la búsqueda web la exige")
ok("_tn.es_texto_normativo(texto)" in inspect.getsource(dg.cuerpo_para_transcribir),
   "y la verja del documento también, para las tres transcripciones")
dg.avisos_cotejo.clear()
ok(dg.cuerpo_para_transcribir(NOTA_2_2026, 150, "Reglamento Interior del IMSS") == ""
   and dg.avisos_cotejo and "NO SE TRANSCRIBIÓ" in dg.avisos_cotejo[0],
   "la nota del 2/2026 ya no se transcribe, y queda dicho")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
