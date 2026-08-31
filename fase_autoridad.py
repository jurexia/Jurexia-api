"""LA AUTORIDAD RESPONSABLE SE LEE, NO SE PREGUNTA.

David, 30-ago-2026: «no deberías de preguntarme eso pues viene en la propia
sentencia reclamada».

Tiene razón y mi arreglo anterior era la solución equivocada al problema
correcto. Sin la autoridad el proyecto salía con cuatro huecos en la parte que
se ejecuta —la competencia, la existencia del acto, los efectos y el
resolutivo—, y yo lo resolví exigiéndosela al secretario. Pero el dato está en
el documento que él ya subió y que ya pasó por OCR: pedírselo es hacerle
teclear lo que el sistema tiene delante.

Cada campo que se le pide y podría deducirse es un minuto suyo y una ocasión de
equivocarse. El adelanto vale por lo que le ahorra.

SE LEE DEL ENCABEZADO, que es donde toda resolución se identifica: «Junta
Especial Número Cincuenta de la Federal de Conciliación y Arbitraje», «Primera
Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro», «Sala
Regional del Tribunal Federal de Justicia Administrativa».

Y NO SE INVENTA: si no se reconoce ninguna, se devuelve vacío y el formulario
la pide. Un nombre de autoridad equivocado en el resolutivo es peor que un
hueco, porque el hueco se ve.
"""

from __future__ import annotations

import re

# Las autoridades que dictan un acto reclamado, por cómo se nombran a sí mismas
# en su propio encabezado. El orden importa: lo más específico primero.
_PATRONES = (
    r"Junta\s+Especial\s+N[úu]mero\s+[\wáéíóúñ]+(?:\s+bis)?"
    r"(?:\s+de\s+la\s+(?:Local|Federal)\s+de\s+Conciliaci[óo]n\s+y\s+Arbitraje)?"
    r"(?:\s+en\s+el\s+Estado\s+de\s+[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+)?",
    r"Junta\s+(?:Local|Federal)\s+de\s+Conciliaci[óo]n\s+y\s+Arbitraje"
    r"(?:\s+d[el]{1,2}\s+[A-ZÁÉÍÓÚÑ][\wáéíóúñ\s]{2,40})?",
    r"Tribunal\s+(?:Laboral|Colegiado|Unitario)[\w\sáéíóúñ,]{5,80}",
    # LO ESPECÍFICO ANTES QUE LO GENÉRICO: «Sala Superior» casaba primero y se
    # quedaba con media identidad del Tribunal de Justicia Administrativa.
    r"Tribunal\s+de\s+Justicia\s+Administrativa[\w\sáéíóúñ,]{0,80}",
    r"(?:Primera|Segunda|Tercera|Cuarta|Quinta|Sexta|S[ée]ptima|Octava|Novena|"
    r"D[ée]cima)\s+Sala\s+(?:Civil|Familiar|Penal|Administrativa|Especializada)?"
    r"[\w\sáéíóúñ]{0,70}",
    r"Sala\s+(?:Familiar|Civil|Penal|Regional|Superior|Especializada)"
    r"[\w\sáéíóúñ]{0,70}",
    r"Juez\s+[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+\s+de\s+Primera\s+Instancia[\w\sáéíóúñ]{0,60}",
    r"Tribunal\s+Unitario\s+Agrario[\w\s,\d]{0,40}",
)

# Un encabezado no se extiende más allá de esto: si el patrón se come media
# página es que casó con prosa, no con el nombre de un órgano.
MAX_NOMBRE = 130


def _limpiar(x: str) -> str:
    x = " ".join((x or "").split())
    x = re.sub(r"[,;.]\s*$", "", x)
    # La coletilla procesal no es parte del nombre.
    # EL NOMBRE ACABA DONDE EMPIEZA EL ASUNTO. «Primera Sala Civil Juicio
    # sumario civil» salía entero porque el patrón sigue tragando palabras: el
    # encabezado pone el órgano y a continuación el expediente, y hay que
    # cortar ahí.
    x = re.split(r"\s+(?:con\s+residencia|con\s+sede|en\s+el\s+juicio|"
                 r"al\s+resolver|dict[óo]|pronunci[óo]|juicio|toca|expediente|"
                 r"sentencia|laudo|resoluci[óo]n|amparo)\b", x, flags=re.I)[0]
    x = x.strip(" ,;.")
    # Los encabezados van en versales; el cuerpo de la sentencia no.
    if x.isupper() and len(x) > 12:
        menores = {"de", "del", "la", "las", "los", "el", "y", "en", "al"}
        x = " ".join(w.lower() if w.lower() in menores else w.capitalize()
                     for w in x.split())
    return x.strip()[:MAX_NOMBRE]


def de_texto(acto: str) -> str:
    """La autoridad que dictó el acto, leída de su propio encabezado.

    Se mira sobre todo el PRINCIPIO del documento, que es donde toda resolución
    se identifica; si ahí no aparece, se busca en el resto pero exigiendo que
    salga al menos dos veces —una mención aislada suele ser una cita, no el
    emisor—.
    """
    t = " ".join((acto or "").split())
    if not t:
        return ""
    cabeza = t[:3000]
    for p in _PATRONES:
        m = re.search(p, cabeza, re.I)
        if m:
            return _limpiar(m.group(0))
    # Fuera de la cabecera hace falta insistencia para creérselo.
    mejor, veces = "", 0
    for p in _PATRONES:
        halladas = [_limpiar(x.group(0)) for x in re.finditer(p, t, re.I)]
        if not halladas:
            continue
        from collections import Counter
        nombre, n = Counter(halladas).most_common(1)[0]
        if n > veces and n >= 2:
            mejor, veces = nombre, n
    return mejor
