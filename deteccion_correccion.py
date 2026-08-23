#!/usr/bin/env python3
"""¿Está el abogado corrigiendo a la máquina?

MEDIDO ANTES DE ENCHUFARLO (23-ago-2026, 13,723 mensajes reales):
  · primera versión: 5 positivos en 1,000 mensajes, y CUATRO eran falsos —
    «corrige el monto de los intereses» es una instrucción para rehacer un
    escrito, no una queja. Un detector que acierta uno de cada cinco no es
    una alarma: es ruido, y el ruido se acaba silenciando.
  · quitados los imperativos (`corrige`, `revisa`, `te falta`) y añadidos
    cuatro patrones tras LEER el mensaje entero de cada uno.
  · versión final: 13 detecciones en 13,723 mensajes (0.09%), leídas una por
    una. Cero falsos positivos claros.

La primera pasada con patrones flojos daba 99. Buena parte hablaban del caso
del abogado —«está mal el cálculo de mi finiquito»— y no de nosotros. La
diferencia entre 99 y 13 es exactamente la diferencia entre una cifra que
suena bien y una que se sostiene.

POR QUÉ IMPORTA
---------------
99 correcciones de 65 abogados distintos llevaban meses invisibles. Marie
Mejía escribió «estás equivocado, ambas reformas sí existen» el 29 de julio y
canceló el 22 de agosto. Nadie leyó la primera frase.

Una corrección de un litigante es el reporte de calidad más caro de conseguir
y el más barato de ignorar.

EL RIESGO DE ESTO, Y CÓMO SE MIDE
---------------------------------
Un detector que salte con cualquier «está mal» es ruido, y el ruido se apaga.
El abogado escribe «está mal» todo el rato para hablar de SU caso: «está mal
el cálculo de mi finiquito», «considero que el desechamiento es incorrecto»,
«la jefa hizo algo que está mal». Nada de eso nos señala a nosotros.

La diferencia es la SEGUNDA PERSONA dirigida al sistema. «Estás equivocado»
es una corrección; «está mal el laudo» es un caso. Por eso el detector exige
o bien un verbo en segunda persona, o bien una referencia explícita a lo que
la máquina dijo o citas.

Se mide sobre los mensajes reales, y se leen los positivos uno por uno.
"""
import re
import unicodedata


def pelar(s: str) -> str:
    s = unicodedata.normalize('NFD', (s or '').lower())
    return ''.join(c for c in s if unicodedata.category(c) != 'Mn')


# Segunda persona dirigida al sistema. Es la señal más limpia.
TUTEO = re.compile(
    r'\b(?:estas|estabas)\s+(?:muy\s+)?(?:equivocad\w*|mal\b)'
    r'|\bte\s+equivocas\b|\bte\s+equivocaste\b|\btе?\s*lo\s+inventaste\b'
    r'|\bno\s+es\s+correcto\s+lo\s+que\s+(?:dices|citas|pusiste)\b'
    r'|\bnecesitas\s+actualizar\b|\bactualiza\s+tu\s+base\b')
# «CORRIGE», «REVISA» y «TE FALTA» quedaron FUERA a propósito, y esto es lo que
# más se aprende de haberlo medido: son imperativos de trabajo, no quejas.
# Medido sobre 1.000 mensajes reales, la primera versión dio 5 positivos y
# CUATRO eran instrucciones para rehacer un escrito —«corrige el monto de los
# intereses», «corrige el nombre de la exesposa»—. Un detector que acierta una
# de cada cinco no es una alarma: es ruido, y el ruido se acaba silenciando.
#
# El abogado usa «está mal» y «corrige» todo el día para hablar de SU caso. La
# única señal limpia es la que apunta a la máquina en segunda persona, o la que
# cita expresamente lo que la máquina dijo.

# Referencia explícita a lo que el sistema produjo.
SOBRE_LO_DICHO = re.compile(
    r'\b(?:el\s+)?articulo\s+[\w\.]+\s+que\s+(?:citas|mencionas|pusiste)\b'
    r'|\bla\s+(?:tesis|jurisprudencia)\s+que\s+(?:citas|mencionas)\b'
    r'|\bno\s+dice\s+eso\b|\beso\s+no\s+dice\b'
    r'|\besa\s+(?:tesis|ley|reforma)\s+no\s+existe\b'
    r'|\b(?:si|s[ií])\s+existen?\b.{0,30}\bya\s+verifiqu\w+'
    r'|\bya\s+verifiqu\w+.{0,40}\b(?:si|s[ií])\s+existen?\b'
    r'|\blos\s+articulos\s+(?:invocados|citados|que\s+citas)\b[^.\n]{0,40}\bestan\s+(?:mal|equivocados)\b'
    r'|\btu\s+(?:apreciacion|respuesta|analisis|conclusion)\b[^.\n]{0,80}'
    r'(?:es\s+)?(?:incorrecta?|equivocad\w+|errone\w+)'
    r'|\bno\s+es\s+cierto\b[^.\n]{0,30}\b(?:los\s+)?articulos\b'
    r'|\bmi\s+base\s+de\s+datos\b')
# Estos cuatro se añadieron DESPUÉS de comprobar que el detector estricto se
# dejaba fuera correcciones reales: «tu apreciación respecto a que la vía de
# impugnación es el recurso de queja me parece incorrecta» apunta a la máquina
# sin usar segunda persona verbal, y «los artículos invocados DEL CNPCYF están
# equivocados» metía dos palabras entre medias que rompían el patrón.
#
# Cada uno se añadió leyendo el mensaje entero, no por intuición.


# La atribución cruzada entre leyes: «ese artículo es de otra ley». Es el fallo
# concreto que persigue todo este trabajo —citar el 371 de la Ley Federal del
# Trabajo como si fuera del código procesal de Sonora— y el detector estricto
# no lo veía: se descubrió porque una prueba en vivo con esa frase exacta no
# registró nada.
ATRIBUCION_CRUZADA = re.compile(
    r'\b(?:ese|este|el)\s+articulo\s+(?:no\s+)?(?:pertenece|corresponde|es)\b[^.\n]{0,25}'
    r'\b(?:a\s+)?(?:otra\s+ley|otro\s+codigo|otro\s+ordenamiento|de\s+otra\s+ley)\b'
    r'|\bpertenece\s+a\s+(?:otra\s+ley|otro\s+codigo|otro\s+ordenamiento)\b'
    r'|\bes\s+de\s+otra\s+ley\b|\bno\s+es\s+de\s+es[ea]\s+(?:ley|codigo)\b')

# «Eso es incorrecto» y «ese artículo no dice» sólo cuentan AL ABRIR el mensaje,
# y el ancla no es un adorno: sin ella, «el juez dijo que eso es incorrecto en
# su acuerdo» y «mi cliente sostiene que ese artículo no dice nada» saltan, y
# ninguno de los dos habla de nosotros. El corpus no lo habría delatado —los
# patrones sueltos dan cero positivos en los 13.723 mensajes—, así que ese cero
# era suerte, no limpieza. Se descubrió con una prueba trampa escrita a mano.
ABRE_NEGANDO = re.compile(
    r'^\W{0,3}(?:pero\s+|no,?\s+)?eso\s+(?:es|esta)\s+(?:incorrecto|mal|equivocado)\b'
    r'|^\W{0,3}es[ea]\s+articulo\s+no\s+(?:dice|regula|habla|establece)\b')
# MEDIDO: ambos dan CERO positivos nuevos sobre los 13.723 mensajes, y 11 de 11
# en la prueba trampa. Lo honesto es decir las dos cosas: no meten ruido, y
# tampoco han cazado nada todavía. Cubren hacia adelante una queja que sabemos
# que existe porque el sello de correspondencia la vigila desde el otro lado.


def es_correccion(texto: str):
    """Devuelve (bool, señal). Conservador a propósito."""
    t = pelar(texto or '')
    if len(t) > 900:          # un escrito pegado no es una corrección
        return False, ''
    m = TUTEO.search(t)
    if m:
        return True, f'segunda persona: «{m.group(0)}»'
    m = SOBRE_LO_DICHO.search(t)
    if m:
        return True, f'sobre lo citado: «{m.group(0)}»'
    m = ATRIBUCION_CRUZADA.search(t)
    if m:
        return True, f'ley equivocada: «{m.group(0)}»'
    m = ABRE_NEGANDO.search(t)
    if m:
        return True, f'abre negando: «{m.group(0)}»'
    return False, ''
