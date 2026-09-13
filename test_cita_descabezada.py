# -*- coding: utf-8 -*-
"""LA FRASE QUE QUEDA DEBAJO DE UNA CITA.

David, 13-sep-2026, pegando el defecto tal como salió del 650-2025:

    «ALIMENTOS A MENORES DE EDAD. TIENEN UNA TRIPLE DIMENSIÓN, …»
    registro 2023835, reconoce que la obligación alimentaria no se reduce a una
    relación privada entre progenitor y menor, …

El modelo escribe una sola frase y el compositor la parte por el rubro. La cola
arranca con el resto de la ficha —«registro 2023835,»— y detrás viene el verbo
de la oración decapitada.

SE COMPRUEBA EN LAS DOS DIRECCIONES, que es la lección que ya costó una ronda:
que arregle las colas rotas Y que no toque las que estaban bien. El caso
protegido está documentado en el propio código: «registro digital 179849.» es
ficha suelta, no una frase, y ponerle sujeto da «La jurisprudencia en cita
registro digital 179849.», que es peor que el hueco.
"""
import sys
import documento_generado as dg

JUR = {"tipo": "JURISPRUDENCIA", "texto": ""}
AIS = {"tipo": "TESIS AISLADA", "texto": ""}
fallos = []


def caso(cola, espera, tesis=JUR, nota=""):
    sale = dg._con_sujeto_tras_cita(cola, tesis)
    ok = (sale.strip() == espera.strip()) if espera is not None else True
    print(f"  {'OK ' if ok else 'MAL'} {nota}")
    if not ok:
        print(f"       entra: {cola[:88]}")
        print(f"       sale : {sale[:88]}")
        print(f"       debía: {espera[:88]}")
        fallos.append(nota)


print("── LAS QUE ESTABAN ROTAS: se les repone el sujeto ──")
caso("registro 2023835, reconoce que la obligación alimentaria no se reduce a "
     "una relación privada entre progenitor y menor, pues también impone a las "
     "autoridades el deber de adoptar medidas idóneas.",
     "La jurisprudencia en cita reconoce que la obligación alimentaria no se "
     "reduce a una relación privada entre progenitor y menor, pues también "
     "impone a las autoridades el deber de adoptar medidas idóneas.",
     nota="el caso que pegó David (650-2025, ¶235)")

caso("registro 159897, establece que el interés superior debe orientar las "
     "decisiones relacionadas con la infancia y que el desarrollo y el ejercicio "
     "pleno de sus derechos constituyen criterios rectores.",
     "La jurisprudencia en cita establece que el interés superior debe orientar "
     "las decisiones relacionadas con la infancia y que el desarrollo y el "
     "ejercicio pleno de sus derechos constituyen criterios rectores.",
     nota="el segundo del mismo documento (¶255)")

caso("registro digital 2011829, página 1120, confirma que la suspensión puede "
     "tener efectos restitutorios cuando la restitución es provisional y posible.",
     "La jurisprudencia en cita confirma que la suspensión puede tener efectos "
     "restitutorios cuando la restitución es provisional y posible.",
     nota="ficha de dos trozos, se podan los dos")

caso("tesis 1a./J. 44/2021, sostiene que el interés superior obliga a valorar "
     "medidas reforzadas de tutela en los asuntos de alimentos.",
     "La tesis en cita sostiene que el interés superior obliga a valorar "
     "medidas reforzadas de tutela en los asuntos de alimentos.",
     tesis=AIS, nota="tesis aislada: el sustantivo cambia")

print("\n── LAS QUE ESTABAN BIEN: no se tocan ──")
# NO SE VACÍA AQUÍ: se devuelve intacta y es el LLAMADOR quien la tira, con su
# filtro de «más de seis palabras». Así estaba antes y así debe seguir; esta
# función sólo repone sujetos. La primera versión de esta prueba esperaba que
# saliera vacía, y la que estaba mal era la prueba.
caso("registro digital 179849.", "registro digital 179849.",
     nota="ficha suelta sin coma: se devuelve intacta y la tira el llamador")

caso("registro 2023835. La jurisprudencia transcrita es obligatoria para este "
     "Tribunal Colegiado conforme al artículo 217.",
     "registro 2023835. La jurisprudencia transcrita es obligatoria para este "
     "Tribunal Colegiado conforme al artículo 217.",
     nota="la poda NO salta el punto y final: no se come la oración siguiente")

caso("Conforme al criterio citado, la naturaleza alimentaria del derecho obliga "
     "a que las autoridades valoren medidas reforzadas de tutela.",
     "Conforme al criterio citado, la naturaleza alimentaria del derecho obliga "
     "a que las autoridades valoren medidas reforzadas de tutela.",
     nota="empieza en mayúscula: es una oración entera, no se toca")

caso("y confirma que la Sala debió atender los conceptos planteados por la parte "
     "actora en su demanda de nulidad.",
     "y confirma que la Sala debió atender los conceptos planteados por la parte "
     "actora en su demanda de nulidad.",
     nota="arranca por conjunción: anteponer sujeto daría «La tesis en cita y…»")

caso("de la que se desprende que la suspensión procede.",
     "de la que se desprende que la suspensión procede.",
     nota="arranque preposicional: sigue protegido")

caso("confirma que la Sala debió atender los conceptos efectivamente planteados "
     "en la demanda de nulidad de la actora.",
     "La jurisprudencia en cita confirma que la Sala debió atender los conceptos "
     "efectivamente planteados en la demanda de nulidad de la actora.",
     nota="sin ficha delante: el arreglo viejo ya lo hacía y sigue haciéndolo")

print("\n── EL ANUNCIO DICE LA VERDAD SOBRE DÓNDE ESTÁ EL TEXTO ──")
corta = {"tipo": "JURISPRUDENCIA", "instancia": "Primera Sala", "registro": "2023835",
         "obligatoria": True, "rubro": "X", "texto": "Un texto de pocas palabras."}
larga = {**corta, "texto": " ".join(["palabra"] * (dg.MAX_PALABRAS_TESIS_CUERPO + 5))}
vacia = {**corta, "texto": ""}
for t, espera, nota in ((corta, "de rubro y texto siguientes:", "texto corto: va en el cuerpo"),
                        (larga, "de rubro siguiente:", "texto largo: baja a la nota"),
                        (vacia, "de rubro siguiente:", "sin texto: no se promete")):
    sale = dg.anuncio_de(t, "Sirve de apoyo")
    ok = sale.endswith(espera)
    print(f"  {'OK ' if ok else 'MAL'} {nota}")
    if not ok:
        print(f"       sale : …{sale[-60:]}")
        print(f"       debía: …{espera}")
        fallos.append(nota)

print()
if fallos:
    print(f"FALLAN {len(fallos)}: " + " · ".join(fallos))
    sys.exit(1)
print("TODAS LAS COMPROBACIONES PASAN")
