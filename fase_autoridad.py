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
    # UN TRIBUNAL COLEGIADO NUNCA ES LA RESPONSABLE de un asunto que resuelve
    # otro colegiado: ni en amparo directo, ni en queja, ni en revisión. Estaba
    # en el patrón y el resultado fue que en cuatro de cinco asuntos reales la
    # «autoridad responsable» salió de una CITA DE TESIS del acto —«Tribunal
    # Colegiado en Materia Administrativa del Primer Circuito, publicada en la
    # página 1620»— y en uno de ellos era ESTE MISMO tribunal.
    r"Tribunal\s+(?:Laboral|Unitario|Agrario|Unitario\s+Agrario)[\w\sáéíóúñ,]{5,80}",
    r"(?:Juez|Jueza)\s+(?:Primero|Segundo|Tercero|Cuarto|Quinto|Sexto|S[ée]ptimo"
    r"|Octavo|Noveno|D[ée]cimo)?\s*de\s+Distrito[\w\sáéíóúñ,]{0,80}",
    r"Sala\s+Regional[\w\sáéíóúñ]{0,60}",
    r"Juzgado\s+(?:Primero|Segundo|Tercero|Cuarto|Quinto|Sexto|S[ée]ptimo"
    r"|Octavo|Noveno|D[ée]cimo)?\s*de\s+Distrito[\w\sáéíóúñ,]{0,80}",
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
    # «AMPARO» A SECAS CORTABA EL NOMBRE POR LA MITAD. Media judicatura se
    # llama «Juez Primero de Distrito en Materia de Amparo Civil…» y el corte
    # dejaba «Juez Primero de Distrito en Materia». Sólo se corta cuando abre
    # una frase nueva —«en el amparo indirecto 795/2023»—, no cuando forma
    # parte de la denominación del órgano.
    x = re.split(r"\s+(?:con\s+residencia|con\s+sede|en\s+el\s+juicio|"
                 r"al\s+resolver|dict[óo]|pronunci[óo]|juicio|toca|expediente|"
                 r"sentencia|laudo|resoluci[óo]n|"
                 r"en\s+el\s+amparo|del\s+amparo)\b", x, flags=re.I)[0]
    x = x.strip(" ,;.")
    # Y LA COLA DE UNA CITA DE TESIS NO ES PARTE DEL NOMBRE. «…del Primer
    # Circuito, publicada en la página 1620, Tomo…» es una localización del
    # Semanario, no una autoridad.
    x = re.split(r"\s*,?\s*(?:publicad[ao]|visible|consultable|registro\s+digital"
                 r"|Semanario|p[áa]gina|Tomo|[ÉE]poca|con\s+fecha)\b",
                 x, flags=re.I)[0].strip(" ,;.")
    # Y EL NOMBRE ACABA EN NOMBRE PROPIO. Al pasar a quedarme con la
    # coincidencia MÁS LARGA —para no perder «de la Federal de Conciliación y
    # Arbitraje»— el patrón empezó a arrastrar el verbo que sigue: «Primera Sala
    # Civil del Tribunal Superior de Justicia del Estado de Querétaro resolvió».
    # Elegir la más larga premia justo lo que el patrón traga de más, así que
    # hay que devolverlo. Se cortan las palabras finales en minúscula: el nombre
    # de una autoridad termina en mayúscula —«Querétaro», «Arbitraje»— y las
    # minúsculas de dentro («de la», «del Estado») nunca van al final.
    partes = x.split()
    while partes and partes[-1][:1].islower():
        partes.pop()
    x = " ".join(partes)
    # Los encabezados van en versales; el cuerpo de la sentencia no.
    if x.isupper() and len(x) > 12:
        menores = {"de", "del", "la", "las", "los", "el", "y", "en", "al"}
        # LOS ROMANOS NO SE CAPITALIZAN: «II».capitalize() da «Ii», y el
        # resolutivo salía diciendo «la Sala Regional del Centro Ii». Aquí
        # también, no sólo en el compositor: la autoridad se normaliza dos
        # veces y basta con que una de las dos la rompa.
        _rom = re.compile(r"^[IVXLCDM]{2,7}$")
        x = " ".join(w if _rom.match(w.strip(".,;:")) else
                     (w.lower() if w.lower() in menores else w.capitalize())
                     for w in x.split())
    return x.strip()[:MAX_NOMBRE]


# QUIÉN NO PUEDE SER LA RESPONSABLE. La Suprema Corte y sus Salas aparecen en
# las cabeceras porque los autos y las sentencias citan sus tesis, no porque
# hayan dictado el acto: ningún colegiado revisa a la Corte. Ya pasó —el auto
# de una queja hizo que la «responsable» fuera la Segunda Sala— y es de la misma
# familia que el Tribunal Colegiado que ya se excluyó.
_RX_NUNCA = re.compile(
    r"Suprema\s+Corte|Tribunal\s+Colegiado|Tribunal\s+Pleno|"
    r"(?:Primera|Segunda)\s+Sala\s+de\s+la\s+Suprema|Pleno\s+Regional", re.I)


def _nunca_responsable(nombre: str) -> bool:
    return bool(_RX_NUNCA.search(nombre or ""))


# ═══════════════════════════════════════════════════════════════════════════
# QUÉ ÓRGANO PUEDE SER, SEGÚN EL TIPO DE ASUNTO
# ═══════════════════════════════════════════════════════════════════════════
# De la auditoría de David, sobre la queja: «en competencia dice que el auto lo
# dictó el Juez Quinto Civil en un amparo indirecto, pero fue la Secretaria en
# funciones de Jueza de Distrito». Y tenía razón: en el auto recurrido —que es
# un auto de un JUZGADO DE DISTRITO— se nombra al juez del juicio natural del
# que vino el amparo, y el lector se quedaba con ése.
#
# El fallo no está en los patrones: cada uno es correcto en su sitio. Está en
# que la función no sabía de qué tipo de asunto se trataba, así que aplicaba
# los once a los cuatro tipos por igual. En una queja o en una revisión de
# amparo, lo recurrido lo dictó un órgano FEDERAL de control; un Juez de
# Primera Instancia o una Sala local no pueden serlo, por más veces que el
# documento los nombre. Y al revés: en un amparo directo la responsable es un
# tribunal local o una junta, nunca un Juzgado de Distrito.
#
# Se filtra DESPUÉS de extraer, no antes: los patrones no se tocan —cada uno
# costó su corrección— y así el filtro se puede leer y discutir por separado.
_SOLO_FEDERAL = re.compile(r"de\s+Distrito|Tribunal\s+Colegiado", re.I)
_LOCAL = re.compile(
    r"Primera\s+Instancia|Sala\s+(?:Civil|Familiar|Penal)|"
    r"Tribunal\s+Superior\s+de\s+Justicia", re.I)

_ADMITE = {
    # En un recurso contra lo resuelto por un juez de amparo, el órgano es
    # federal de control. Lo local que aparezca viene del juicio de origen.
    "queja": lambda n: bool(_SOLO_FEDERAL.search(n)),
    "amparo_revision": lambda n: bool(_SOLO_FEDERAL.search(n)),
    # En la revisión fiscal es una Sala del Tribunal Federal de Justicia
    # Administrativa. Nunca un juzgado ni una sala local.
    "revision_fiscal": lambda n: bool(
        re.search(r"Sala\s+Regional|Justicia\s+Administrativa|Sala\s+Superior",
                  n, re.I)),
    # En el amparo directo, cualquiera MENOS un órgano federal de control: el
    # Juzgado de Distrito que aparezca viene de un amparo indirecto citado.
    "amparo_directo": lambda n: not _SOLO_FEDERAL.search(n),
}


def _admisible(nombre: str, tipo: str) -> bool:
    f = _ADMITE.get((tipo or "").strip().lower())
    return True if f is None else bool(f(nombre))


def de_texto(acto: str, tipo: str = "") -> str:
    """La autoridad que dictó el acto, leída del propio documento.

    DECÍA «leída de su propio encabezado» Y ERA MEDIA VERDAD. En un laudo
    escaneado el encabezado no es texto: es el SELLO de la Junta, y el OCR lo
    despedaza. Del laudo del 382/2024 salió esto:

        JUNTA FEDERAL DE / UNCILIACIÓN Y ARBITRA / JUNTA ESPECIAL / No. 50 /
        UERÉTARO QR / ANTIR

    Mi extractor se quedaba con la PRIMERA coincidencia y devolvía «Junta
    Especial Numero 50», medio nombre, mientras cinco mil caracteres más abajo
    el cuerpo de la resolución la nombraba entera y bien escrita: «Junta
    Especial Número 50 de la Federal de Conciliación y Arbitraje». El principio
    del documento es el peor sitio donde mirar cuando viene de un escáner.

    Ahora se recorre entero y gana la coincidencia MÁS COMPLETA, no la primera.
    Un nombre más largo del mismo patrón es siempre mejor: contiene al corto y
    añade lo que le falta. Y se sigue exigiendo insistencia fuera de la cabecera
    —dos menciones— para no confundir una cita con el emisor.
    """
    t = " ".join((acto or "").split())
    if not t:
        return ""

    # ═══════════════════════════════════════════════════════════════════
    # LA CABECERA MANDA; EL CUERPO SÓLO SI LA CABECERA CALLA
    # ═══════════════════════════════════════════════════════════════════
    # Tercer intento, y los dos anteriores fallaron por lo contrario:
    #
    #   · «la PRIMERA coincidencia» se quedaba con el sello escaneado del
    #     laudo, que el OCR despedaza en «JUNTA FEDERAL DE / UNCILIACIÓN Y
    #     ARBITRA / UERÉTARO QR»;
    #   · «la coincidencia MÁS LARGA en todo el documento» eligió, en un amparo
    #     directo CIVIL sobre reivindicación, el «Tribunal Unitario Agrario,
    #     Distrito 42, visibles a fojas 14» que la sentencia menciona en su
    #     segunda mitad al listar las pruebas del juicio agrario previo. La
    #     responsable era la Sala Civil, y estaba en el proemio.
    #
    # La regla buena las concilia: la responsable de un acto se identifica en su
    # ENCABEZADO, así que ahí se busca la más completa; y sólo si el encabezado
    # no da ninguna —porque venía en un sello que el OCR rompió— se mira el
    # cuerpo entero y gana la que MÁS SE REPITE, que es la que el documento
    # nombra una y otra vez porque es la suya.
    cabeza = t[:6000]
    for p in _PATRONES:
        cands = [_limpiar(m.group(0)) for m in re.finditer(p, cabeza, re.I)]
        cands = [c for c in cands if len(c) > 12 and not _nunca_responsable(c)
                 and _admisible(c, tipo)]
        if cands:
            return max(cands, key=len)
    # SI LA CABECERA NO DA NINGUNA, se sigue al cuerpo. No es un caso raro: el
    # AUTO que se recurre en una queja no nombra a quien lo dictó —dice «el
    # secretario da cuenta a la jueza»— y su identidad aparece más abajo, en el
    # cuerpo o en la firma. Ahí gana la que más se repite, que es la suya.

    # Fuera de la cabecera hace falta insistencia para creérselo.
    mejor, veces = "", 0
    for p in _PATRONES:
        halladas = [_limpiar(x.group(0)) for x in re.finditer(p, t, re.I)]
        halladas = [h for h in halladas if not _nunca_responsable(h)
                    and _admisible(h, tipo)]
        if not halladas:
            continue
        from collections import Counter
        # NO BASTA LA MÁS FRECUENTE: con empate a uno, `most_common` devuelve la
        # primera que se vio, y en el auto del 143/2026 eso era «Juzgado de
        # Distrito» a secas —que no identifica a nadie— mientras el nombre
        # completo, «Juzgado Tercero de Distrito en Materia de Amparo Civil,
        # Administrativo y de Trabajo y de Juicios Federales», quedaba a su
        # lado sin ser mirado. Se ordena por concreción, luego por frecuencia y
        # luego por longitud.
        cuenta = Counter(halladas)
        nombre = max(cuenta, key=lambda x: (
            bool(re.search(r"(?:Juez|Jueza|Juzgado)\s+\w+\s+de\s+Distrito"
                           r"|Junta\s+Especial|Sala\s+Regional"
                           r"|Tribunal\s+Unitario", x, re.I)),
            cuenta[x], len(x)))
        n = cuenta[nombre]
        # DOS MENCIONES ERAN DEMASIADO EXIGENTES PARA UN AUTO. La regla nació
        # para no confundir una cita con el emisor, y sirve cuando el documento
        # es largo; pero el auto que se recurre en una queja nombra a quien lo
        # dictó UNA sola vez —«Jueza Cuarto de Distrito en Materia de Amparo
        # Civil»— y con el umbral en dos, la responsable salía vacía en las dos
        # quejas de la prueba.
        #
        # Basta una cuando el patrón nombra un órgano jurisdiccional CONCRETO
        # —un juzgado o un juez de distrito con su número—: eso no es una cita
        # de tesis, es quien resolvió. Para los patrones genéricos se siguen
        # pidiendo dos.
        _concreto = re.search(r"(?:Juez|Jueza|Juzgado)\s+\w+\s+de\s+Distrito"
                              r"|Junta\s+Especial|Sala\s+Regional"
                              r"|Tribunal\s+Unitario", nombre, re.I)
        if n > veces and (n >= 2 or _concreto):
            mejor, veces = nombre, n
    return mejor
