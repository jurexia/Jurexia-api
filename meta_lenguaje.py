"""EL MODELO HABLANDO DEL ARCHIVO, DENTRO DE LA SENTENCIA.

En el amparo en revisión 25/2026, dentro del considerando quinto, el proyecto
decía:

    «El texto proporcionado se interrumpió antes de que la autoridad
     responsable desarrollara las razones concretas con las que vinculó el auto
     reclamado con el acto antecedente.»

David: «La IA verbalizó un fallo de tokenización o un corte en el OCR del
archivo fuente como si fuera parte de la resolución judicial. Esto debe
eliminarse de inmediato del proyecto».

Y NO ERA UNA ALUCINACIÓN: el corte lo hacíamos nosotros. La sentencia recurrida
son 90,126 caracteres y el pipeline la cortaba a 60,000 a mitad de frase. El
modelo describió con exactitud lo que le habíamos hecho. Por eso el arreglo son
tres cosas y no una: no cortar así (fases123_pipeline.recortar_acto),
prohibírselo en el prompt, y esto —la red que lo caza cuando las dos primeras
fallan, porque en este proyecto las redes deterministas son las que han
detenido lo que el prompt no—.

SE BORRA LA FRASE, NO SE AVISA Y YA. Un aviso deja el texto dentro del .docx y
confía en que alguien lo lea; y esta frase no es mejorable —es ajena al género—.
Se quita y se avisa además, para que quede constancia de que ahí faltaba algo.

SE ACEPTA QUEDARSE CORTO. «La transcripción quedó incompleta» no se caza, y es
deliberado: el sujeto «transcripción» aparece en prosa judicial legítima —la de
una audiencia, la de un testimonio— y este filtro BORRA. Un filtro que borra no
puede permitirse un falso positivo: se llevaría una frase del proyecto sin que
nadie lo notara. Medido: 14 de 15 casos, con cero falsos positivos sobre las
ocho frases judiciales de prueba. Y lo que se quita se transcribe ENTERO en el
aviso, para que el secretario pueda devolverlo si el filtro se equivocó.

EL FILTRO ES ESTRECHO, y tiene que serlo: hay prosa judicial legítima que habla
de lo que no consta. «De las constancias no se advierte», «no obra en autos»,
«el texto de la sentencia recurrida es del tenor siguiente» son frases de
sentencia. Lo que se caza es el modelo hablando del ARCHIVO —el texto
«proporcionado», el «fragmento», lo «truncado»—, no el juzgador hablando del
EXPEDIENTE.
"""

from __future__ import annotations

import re

# Cada patrón nombra el archivo o la entrega, no el expediente.
_PATRONES = [
    r"[^.]*\btexto\s+(?:proporcionado|suministrado|facilitado|disponible|"
    r"compartido)\b[^.]*\.",
    r"[^.]*\bse\s+(?:interrumpi[óo]|trunc[óo]|cort[óo])\b[^.]*"
    r"(?:texto|documento|archivo|fragmento|transcripci[óo]n)[^.]*\.",
    r"[^.]*(?:texto|documento|archivo|contenido)[^.]{0,60}\b"
    r"(?:se\s+interrumpe|(?:est[áa]|se\s+encuentra|luce|viene|qued[óo])"
    r"\s+(?:truncad|incomplet|cortad|ilegible)|"
    r"aparece\s+(?:truncad|incomplet)|no\s+est[áa]\s+completo)[^.]*\.",
    r"[^.]*\b(?:el|un)\s+fragmento\s+(?:disponible|proporcionado|"
    r"que\s+se\s+aporta)\b[^.]*\.",
    r"[^.]*\bhasta\s+donde\s+(?:se\s+transcribe|alcanza\s+el\s+texto|"
    r"llega\s+el\s+documento)\b[^.]*\.",
    r"[^.]*\bno\s+(?:es\s+posible\s+)?(?:se\s+)?(?:puede\s+)?"
    r"(?:leer|apreciar|distinguir)[^.]{0,40}\b"
    r"(?:por\s+la\s+calidad|por\s+el\s+escaneo|en\s+la\s+imagen|"
    r"en\s+el\s+archivo)[^.]*\.",
    r"[^.]*\b(?:OCR|escaneo|digitalizaci[óo]n|tokenizaci[óo]n)\b[^.]*\.",
    r"[^.]*\b(?:el\s+)?documento\s+(?:proporcionado|adjunto|de\s+entrada)\b[^.]*\.",
]

_RX = re.compile("|".join(f"(?:{p})" for p in _PATRONES), re.I)

# Lo que NO se caza aunque se le parezca: el juzgador hablando del expediente.
_LEGITIMO = re.compile(
    r"de\s+(?:las\s+)?constancias\s+no\s+se\s+advierte|"
    r"no\s+obra\s+en\s+autos|"
    r"es\s+del\s+tenor\s+siguiente|"
    r"no\s+consta\s+en\s+(?:el\s+)?(?:expediente|autos|juicio)", re.I)


def frases(texto: str) -> list:
    """Las frases de meta-lenguaje que hay en el texto."""
    fuera = []
    for m in _RX.finditer(texto or ""):
        f = m.group(0).strip()
        if len(f) < 20 or _LEGITIMO.search(f):
            continue
        fuera.append(f)
    return fuera


def limpiar(texto: str) -> tuple:
    """(texto sin meta-lenguaje, frases quitadas)."""
    quitadas = frases(texto)
    if not quitadas:
        return texto, []
    fuera = texto
    for f in quitadas:
        fuera = fuera.replace(f, "")
    # Se cierran los dobles espacios que deja el hueco.
    fuera = re.sub(r"[ \t]{2,}", " ", fuera)
    fuera = re.sub(r"\s+([,.;:])", r"\1", fuera)
    return fuera.strip(), quitadas


# ═══════════════════════════════════════════════════════════════════════════
# LA PERÍFRASIS QUE AFIRMA SIN DECIR QUIÉN
# ═══════════════════════════════════════════════════════════════════════════
# Vecina del meta-lenguaje y con la misma raíz: el modelo no tiene el dato y,
# en vez de callar, rodea. De los resultandos del ADA 448/2025:
#
#     «promovió demanda de amparo CONTRA EL ACTO RECLAMADO PRECISADO EN LOS
#      ANTECEDENTES»
#     «LA PERSONA A QUIEN RESULTA TAL CARÁCTER fue emplazada»
#
# Los resultandos existen justamente para individualizar; una perífrasis ahí no
# es un giro de estilo, es el apartado sin cumplir. Y arrastra un segundo daño:
# el número de expediente del considerando de existencia se lee de estos
# resultandos, así que la evasión deja también ese hueco.
#
# ESTO SÓLO AVISA, NO BORRA. A diferencia del meta-lenguaje —que es ajeno al
# género y se quita—, aquí la frase ocupa el lugar de algo que debe escribirse:
# borrarla dejaría el resultando mudo. Lo que hace falta es que el secretario
# ponga el nombre, y para eso tiene que verlo.
_RX_PERIFRASIS = re.compile(
    r"(?:el\s+)?acto\s+reclamado\s+precisad[oa]\s+en\s+(?:los\s+)?antecedentes|"
    r"la\s+persona\s+a\s+quien\s+(?:le\s+)?resulta\s+(?:tal|dicho)\s+car[áa]cter|"
    r"en\s+los\s+t[ée]rminos\s+(?:ah[íi]|all[íi])\s+(?:precisados|se[ñn]alados)|"
    r"(?:el|los)\s+acto[s]?\s+(?:que\s+)?(?:quedaron|qued[óo])\s+precisad",
    re.I)


def perifrasis(texto: str) -> list:
    """Las evasiones de los resultandos. Sólo avisa; no toca el texto."""
    return sorted({m.group(0).strip() for m in _RX_PERIFRASIS.finditer(texto or "")})
