"""¿Esto es el texto de una norma, o es alguien hablando de ella?

POR QUÉ EXISTE. Revisión fiscal 2/2026, 17-sep-2026. La nota al pie del
artículo 150 del Reglamento Interior del IMSS, en un proyecto de sentencia,
decía:

    «Artículo 150. El **artículo 150** que corresponde al **Reglamento Interior
     del Instituto Mexicano del Seguro Social (IMSS)** establece las
     **atribuciones de las subdelegaciones** (…) así que no puedo citarlo
     *textualmente y completo con todas sus fracciones* con total seguridad
     (…) Lo que sí se alcanza a leer (…) es, en esencia: - …»

Era la NEGATIVA del buscador web, transcrita como si fuera el precepto.

La búsqueda ya rechazaba negativas, pero por LISTA NEGRA de frases y mirando
sólo los primeros 400 caracteres: esta decía «no puedo citarlo» en la posición
464, con asteriscos en medio. Una lista negra siempre deja pasar la siguiente
manera de decir lo mismo.

LA PRUEBA ES POSITIVA: un artículo de ley tiene una forma que una respuesta de
chat no tiene. No lleva Markdown. No habla en primera ni en segunda persona. No
comenta la búsqueda ni la fuente. Y no empieza describiéndose —«El artículo 150
que corresponde al…»—: empieza diciendo lo que dispone.

Se usa en DOS sitios: al traer un precepto de internet, y en la verja del
documento por la que pasa toda transcripción, venga de donde venga.
"""
from __future__ import annotations

import re

# Marcas de formato de chat. Ninguna ley las lleva.
#
# CALIBRADO sobre 4,465 artículos reales: la primera versión rechazaba el 0.99 %
# por dos reglas demasiado anchas. Un guion al principio de renglón lo usa el
# propio acervo en sus transitorios —«- TRANSITORIO --- ÚNICO.-»—, así que una
# viñeta sola no prueba nada: hacen falta dos o más renglones de lista. Y una
# cursiva suelta puede ser un asterisco de llamada; la negrita doble, no.
_RX_MARKDOWN_FUERTE = re.compile(
    r"\*\*[^*\n]{1,200}\*\*"            # **negrita**
    r"|(?m:^\s*#{1,6}\s(?!\s*(?i:art[íi]culo|cap[íi]tulo|t[íi]tulo|secci[óo]n|libro)))"  # encabezados de chat
    r"|`[^`\n]+`"                       # código
    r"|\[[^\]\n]{1,120}\]\(https?://",  # enlaces
    re.S)
_RX_CURSIVA = re.compile(r"(?<![\w*])\*[^*\s][^*\n]{0,200}\*(?![\w*])")
_RX_VINETA = re.compile(r"(?m)^\s*(?:[-•]|\d+\))\s+(?!TRANSITORI|-)\S")

# Quien habla es un asistente: primera o segunda persona, o la búsqueda misma.
# «proporcionados por» salió de aquí: las leyes lo dicen —«los datos
# proporcionados por el contribuyente»—.
_RX_ASISTENTE = re.compile(
    r"\b(?:puedo|pude|podr[íi]a\s+(?:citar|transcribir|ayudar)|no\s+tengo\s+acceso|"
    r"he\s+(?:encontrado|localizado|revisado)|no\s+encontr[ée]\b|no\s+localic[ée]\b|"
    r"te\s+(?:recomiendo|sugiero|comparto)|le\s+(?:recomiendo|sugiero)|indicas|mencionas|"
    r"los\s+resultados\s+(?:proporcionados|disponibles|de\s+(?:la\s+)?b[úu]squeda)|"
    r"fuente\s+oficial\s+que|publicaci[óo]n\s+oficial\s+primaria|"
    r"(?:aparece|est[áa])\s+(?:parcialmente\s+)?recortad[oa]|se\s+alcanza\s+a\s+leer|"
    r"en\s+esencia|no\s+(?:es\s+posible|puedo)\s+(?:citar|transcribir)|"
    r"textualmente\s+y\s+complet|con\s+total\s+seguridad|"
    r"a\s+continuaci[óo]n\s+(?:te|le)\b|aqu[íi]\s+(?:tienes|est[áa]\s+el)|"
    r"espero\s+que|si\s+necesitas)\b", re.I)
_RX_NO_LOCALIZADO = re.compile(r"^\W{0,4}no\s+localizado\b", re.I)

# Una descripción del artículo en vez del artículo: «El artículo 150 que
# corresponde al…», «Este artículo establece…», «Dicho precepto regula…».
_RX_SE_DESCRIBE = re.compile(
    r"^\W{0,4}(?:el|este|dicho|ese|la\s+norma\s+del)\s+"
    r"(?:art[íi]culo|precepto|numeral)(?:\s+\d{1,4}\s*(?:bis|ter|[A-K])?)?\s+"
    r"(?:que\s+(?:corresponde|pertenece|se\s+refiere)|del\s|de\s+la\s|"
    r"establece|dispone|regula|se\s+refiere|trata|se\s+ocupa|prev[ée]|se[ñn]ala)",
    re.I)


def es_texto_normativo(texto: str) -> tuple:
    """(True, "") si puede ser el texto de una norma; (False, motivo) si no."""
    t = " ".join(str(texto or "").split())
    if not t:
        return False, "vacío"
    # La migaja del acervo y la cabecera propia van fuera antes de mirar.
    t = re.sub(r"^\s*\[[^\]]{0,400}\]\s*", "", t)
    t = re.sub(r"^[\s«»\"'“”]*art[íi]culo\s+\d{1,4}\s*(?:bis|ter|[A-K])?\s*[.\-–]*\s*",
               "", t, flags=re.I)
    crudo = str(texto or "")
    # SIN LA REGLA DE VIÑETAS, y sin exigir nada del encabezado «####
    # ARTÍCULO»: los dos los trae el propio acervo de la Ciudad de México desde
    # su ingesta, y rechazaban el 0.33 % de artículos reales. La negrita y las
    # frases de asistente bastan para reconocer una respuesta de chat.
    if _RX_MARKDOWN_FUERTE.search(crudo) or len(_RX_CURSIVA.findall(crudo)) >= 2:
        return False, "trae formato de chat (Markdown)"
    if _RX_NO_LOCALIZADO.search(t):
        return False, "el buscador dice que no lo localizó"
    m = _RX_ASISTENTE.search(t)
    if m:
        return False, f"habla un asistente («{m.group(0)}»)"
    if _RX_SE_DESCRIBE.search(t):
        return False, "describe el artículo en vez de transcribirlo"
    return True, ""
