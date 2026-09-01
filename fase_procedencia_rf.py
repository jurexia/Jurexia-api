"""LA PROCEDENCIA DE LA REVISIÓN FISCAL, MOTIVADA DE OFICIO.

David: «En la revisión fiscal es estrictamente obligatorio que el Tribunal
Colegiado motive de oficio por qué se surte la procedencia —por ejemplo,
verificando si el crédito fiscal de $847,738.77 supera el umbral de las 3,500
UMAs conforme al art. 63, fr. I, LFPCA—. La autoridad dedicó sus tres agravios
a justificar la procedencia, por lo que el Colegiado no puede limitarse a una
fórmula vacía de dos líneas».

Lo que salía era esa fórmula vacía, y encima llamaba JUICIO al recurso:

    «El juicio es procedente y no se advierte causa de improcedencia que impida
     el análisis de la controversia.»

═══════════════════════════════════════════════════════════════════════════
LO QUE ESTE MÓDULO AFIRMA Y LO QUE NO
═══════════════════════════════════════════════════════════════════════════
La aritmética es determinista y comprobable: cuantía contra 3,500 veces el
valor diario de la UMA del año en que se emitió la resolución recurrida. Eso se
calcula y se escribe.

Lo que NO se hace es decidir por el secretario cuando falta una pieza. Si no se
puede leer la cuantía, o no se conoce la UMA de ese año, NO se escribe un
razonamiento a medias: se deja la fórmula del corpus y se avisa. Un considerando
de procedencia mal motivado es peor que uno escueto, porque el escueto se ve.

LOS VALORES DE LA UMA son los publicados por el INEGI, vigentes del 1 de febrero
de cada año. Se anotan aquí porque son cifra pública y estable, pero cada una
lleva su año: si el asunto es de un año que no está en la tabla, no se inventa
—se avisa y se deja el hueco—. El propio proyecto ya usó 113.14 para 2025 al
sintetizar los agravios de la RF 44/2025, lo que da un punto de contraste.
"""

from __future__ import annotations

import re

# Valor DIARIO de la UMA, por año (INEGI, vigente desde el 1 de febrero).
UMA_DIARIA = {
    2016: 73.04, 2017: 75.49, 2018: 80.60, 2019: 84.49, 2020: 86.88,
    2021: 89.62, 2022: 96.22, 2023: 103.74, 2024: 108.57, 2025: 113.14,
}

VECES = 3500          # artículo 63, fracción I, de la LFPCA

# «un crédito fiscal por $847,738.77», «crédito fiscal de $ 1,234,567.00».
_RX_CUANTIA = re.compile(
    r"cr[ée]dito\s+fiscal\s+(?:determinado\s+)?(?:por|de|en)\s+(?:la\s+"
    r"cantidad\s+de\s+)?\$?\s*([\d][\d,\. ]{3,20})", re.I)


def cuantia_de(texto: str) -> float:
    """El monto del crédito fiscal, o 0.0.

    Se busca sólo donde el texto lo DECLARA crédito fiscal. Un «primer número
    con forma de dinero» dentro de cien mil caracteres de expediente casa
    siempre —con una foja, con un número de oficio, con un año—.
    """
    for m in _RX_CUANTIA.finditer(texto or ""):
        crudo = m.group(1).strip().rstrip(".,")
        # 847,738.77 → 847738.77 ; 1.234.567,00 no se usa en México.
        n = crudo.replace(" ", "").replace(",", "")
        try:
            v = float(n)
        except ValueError:
            continue
        if v > 0:
            return v
    return 0.0


def umbral(anio: int) -> float:
    """3,500 veces la UMA diaria de ese año, o 0.0 si no consta."""
    u = UMA_DIARIA.get(int(anio or 0))
    return round(VECES * u, 2) if u else 0.0


def _pesos(x: float) -> str:
    return f"${x:,.2f}"


def parrafo(texto_fuente: str, anio_resolucion: int) -> tuple:
    """(párrafo de procedencia, avisos). Cadena vacía si no se puede motivar."""
    avisos = []
    c = cuantia_de(texto_fuente)
    u = UMA_DIARIA.get(int(anio_resolucion or 0))
    if not c:
        avisos.append(
            "NO SE PUDO LEER LA CUANTÍA DEL CRÉDITO FISCAL en los documentos, "
            "así que la procedencia sale con la fórmula del corpus y sin "
            "motivar el supuesto del artículo 63. En revisión fiscal esa "
            "motivación es oficiosa: escríbela antes de firmar.")
        return "", avisos
    if not u:
        avisos.append(
            f"NO CONSTA EL VALOR DE LA UMA PARA {anio_resolucion}: no se pudo "
            f"comparar la cuantía de {_pesos(c)} contra las 3,500 UMAs del "
            f"artículo 63, fracción I. Compruébalo y escríbelo.")
        return "", avisos

    t = umbral(anio_resolucion)
    if c > t:
        p = (f"El recurso es procedente en términos del artículo 63, fracción "
             f"I, de la Ley Federal de Procedimiento Contencioso "
             f"Administrativo, toda vez que el asunto versa sobre una "
             f"resolución en la que se determinó un crédito fiscal por "
             f"{_pesos(c)}, cantidad que excede de tres mil quinientas veces "
             f"el valor diario de la Unidad de Medida y Actualización vigente "
             f"al momento de la emisión de la resolución recurrida "
             f"—{_pesos(u)} en {anio_resolucion}, esto es, {_pesos(t)}—, sin "
             f"que sea necesario que el asunto revista, además, importancia y "
             f"trascendencia.")
        return p, avisos

    p = (f"El crédito fiscal determinado asciende a {_pesos(c)}, cantidad que "
         f"NO excede de tres mil quinientas veces el valor diario de la Unidad "
         f"de Medida y Actualización vigente al momento de la emisión de la "
         f"resolución recurrida —{_pesos(u)} en {anio_resolucion}, esto es, "
         f"{_pesos(t)}—, por lo que la procedencia no puede sustentarse en la "
         f"fracción I del artículo 63 de la Ley Federal de Procedimiento "
         f"Contencioso Administrativo.")
    avisos.append(
        f"LA CUANTÍA NO ALCANZA EL UMBRAL: {_pesos(c)} contra {_pesos(t)} "
        f"(3,500 UMAs de {anio_resolucion}). El recurso sólo procede si se "
        f"surte OTRA fracción del artículo 63 —o el supuesto de importancia y "
        f"trascendencia—: decídela y escríbela, porque de esto depende que el "
        f"recurso se estudie o se deseche.")
    return p, avisos
