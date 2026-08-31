"""QUE NADA DEL PROYECTO SEA DE OTRO ASUNTO.

David, 31-ago-2026: «asegúrate de que los formatos de salida no estén
contaminados con datos que no correspondan al asunto que proyecta el
secretario».

Es una petición que se puede comprobar sin modelo, y por eso se comprueba: un
proyecto se construye con los documentos que el secretario subió y con los
datos que tecleó. Todo NOMBRE PROPIO, todo NÚMERO DE EXPEDIENTE y toda CANTIDAD
que aparezca en la sentencia y no esté en ninguno de esos sitios viene de otra
parte, y esa otra parte casi siempre es otro asunto: una plantilla ajena, un
molde de forma del acervo, una cita mal recortada.

YA HA PASADO, y por eso existe esto:
  · el resolutivo nombró como autoridad responsable a un tribunal que salía de
    una CITA DE TESIS del acto reclamado, y en un caso era el propio tribunal
    que resuelve;
  · las plantillas precargadas metían «Querétaro, Querétaro» y el nombre de un
    magistrado que no era el del secretario;
  · un molde de forma del acervo podía arrastrar los hechos del expediente
    ajeno del que se copió.

LO QUE NO SE PERSIGUE: los nombres de los órganos jurisdiccionales genéricos,
los de las leyes y los de las partes que el propio encargo declara. Y las
CITAS DE TESIS traen nombres de tribunales y de salas que son legítimos: se
excluyen mirando si están dentro de una transcripción.
"""

from __future__ import annotations

import re
import unicodedata

# Un nombre propio: dos o más palabras capitalizadas seguidas. Se piden dos para
# no perseguir cada inicio de frase.
_RX_NOMBRE = re.compile(
    r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,}(?:\s+(?:de|del|la|las|los|y)\s+)?"
    r"(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,}){1,4})\b")
_RX_EXPEDIENTE = re.compile(r"\b(\d{1,5}\s*/\s*\d{2,4})\b")
_RX_CANTIDAD = re.compile(r"\$\s?([\d,]{3,}\.?\d{0,2})")

# Lo que es del oficio y no de un asunto.
_GENERICO = {
    "tribunal colegiado", "poder judicial", "suprema corte", "justicia de la",
    "ley de amparo", "constitucion politica", "estados unidos", "codigo civil",
    "codigo federal", "ley federal", "consejo de la", "semanario judicial",
    "sistema integral", "diario oficial", "corte interamericana",
    "derechos humanos", "juez de distrito", "sala regional", "junta especial",
    "libro de control", "seguimiento de expedientes", "primera sala",
    "segunda sala", "pleno de circuito", "plenos de circuito", "gaceta del",
    "tesis aislada", "registro digital",
    # LOS QUE SALIERON COMO FALSOS POSITIVOS EN LA PRUEBA DE LOS CINCO. Ninguno
    # es un dato de asunto: son el vocabulario del oficio.
    "constitucion federal", "constitucion general", "ministerio publico",
    "ley organica", "ley reglamentaria", "ley agraria", "codigo nacional",
    "estado mexicano", "servicio publico", "sector publico", "libro segundo",
    "constituyente permanente", "judicatura federal", "acuerdo general",
    "tribunales colegiados", "plenos regionales", "pleno regional",
    "procedimiento contencioso", "procedimientos penales", "partes comun",
    "poder ejecutivo", "poder legislativo", "camara de diputados",
    "secretaria de hacienda", "gaceta oficial", "periodico oficial",
}

# Los números que parecen expediente y no lo son: acuerdos generales, claves de
# tesis y de contradicción. «3/2013» es el Acuerdo General del Consejo, no el
# expediente de nadie, y salía acusado en los cinco proyectos de la prueba.
_RX_NO_ES_EXPEDIENTE = re.compile(
    r"(?:acuerdo\s+general|jurisprudencia|tesis|contradicci[óo]n|diverso|"
    r"[12]a\.?\s*/\s*J\.?|P\.?\s*/\s*J\.?|clave)[^.;]{0,40}$", re.I)


def _norm(x: str) -> str:
    y = unicodedata.normalize("NFKD", (x or "").lower())
    return "".join(c for c in y if not unicodedata.combining(c))


def _generico(nombre: str) -> bool:
    n = _norm(nombre)
    return any(g in n for g in _GENERICO)


def _fuera_de_cita(texto: str) -> str:
    """El texto sin lo transcrito.

    Dentro de una transcripción —de tesis o de precepto— aparecen nombres de
    salas, de tribunales y de asuntos que son legítimos: son parte de lo citado,
    no del expediente. Perseguirlos daría cientos de falsos positivos.
    """
    sin = re.sub(r"[«\"“][^»\"”]{60,}[»\"”]", " ", texto or "")
    # Y los bloques de identificación de tesis, que van en versales.
    sin = re.sub(r"^[A-ZÁÉÍÓÚÑ ,.;:()\d/-]{40,}$", " ", sin, flags=re.M)
    return sin


def revisar(sentencia: str, fuentes: list, encargo: dict) -> list:
    """Los datos del proyecto que no están en ninguna fuente ni en el encargo.

    `fuentes` son los textos de los documentos que el secretario subió —acto,
    recurso, constancias—. `encargo` es lo que tecleó.
    """
    # LA GUARDA MIDE LOS DOCUMENTOS, NO EL ENCARGO. La primera versión sumaba
    # los dos y el encargo solo —nombres, tribunal, expediente— ya pasaba de
    # 500 caracteres, así que la guarda nunca protegía: con las fuentes vacías
    # el detector acusaba de «datos ajenos» a TODO el documento.
    #
    # Y las fuentes tienen que venir del MISMO texto que vio el modelo. Estos
    # expedientes son escaneos: leídos con un extractor sin OCR devuelven 31, 86
    # o 836 caracteres, y contra eso cualquier medida da contaminación. Pasó:
    # acusé al proyecto de meter «Knight First Amendment Institute v. Trump» en
    # una queja mexicana, y ese caso estaba LITERALMENTE en el escrito de
    # agravios de la parte —11,515 caracteres de OCR lo confirmaron—. El
    # proyecto lo relataba bien; el que leía mal era yo.
    docs = _norm(" ".join(f or "" for f in (fuentes or [])))
    if len(docs) < 3000:
        return ["No se pudo comprobar la contaminación: los documentos del "
                "asunto no traen texto suficiente (¿escaneo sin OCR?). Un "
                "detector sin fuentes acusa de todo."]
    # EL NÚMERO DEL PROPIO ASUNTO SE ESCRIBE DE DOS MANERAS. El encargo lo trae
    # como «143-2026» —así viaja en la URL y en el formulario— y la sentencia lo
    # escribe «143/2026», que es como se cita un expediente. El detector los
    # veía distintos y acusaba al proyecto de inventar su propio número.
    enc = " ".join(str(v) for v in (encargo or {}).values())
    base = (docs + " " + _norm(enc) + " " + _norm(enc.replace("-", "/")))
    limpio = _fuera_de_cita(sentencia or "")
    avisos = []

    ajenos = []
    for m in _RX_NOMBRE.finditer(limpio):
        n = m.group(1).strip()
        if _generico(n) or len(n) < 8:
            continue
        if _norm(n) not in base:
            ajenos.append(n)
    # Se cuentan y se reportan los que más se repiten: un nombre ajeno que
    # aparece una vez puede ser una construcción del modelo; uno que aparece
    # cinco veces es un dato de otro asunto instalado en el proyecto.
    if ajenos:
        from collections import Counter
        top = [f"«{n}» ×{v}" for n, v in Counter(ajenos).most_common(6)]
        avisos.append(
            f"NOMBRES QUE NO ESTÁN EN NINGÚN DOCUMENTO DEL ASUNTO: {', '.join(top)}. "
            f"Compruébalos: un nombre que no viene del expediente ni del encargo "
            f"viene de otra parte.")

    exp_base = {e.replace(" ", "") for e in _RX_EXPEDIENTE.findall(base)}
    exp_aj = set()
    for m in _RX_EXPEDIENTE.finditer(limpio):
        # Lo que lo precede dice si es un expediente o un acuerdo general.
        antes = limpio[max(0, m.start() - 60):m.start()]
        if _RX_NO_ES_EXPEDIENTE.search(antes):
            continue
        e = m.group(1).replace(" ", "")
        if e not in exp_base:
            exp_aj.add(e)
    if exp_aj:
        avisos.append(
            f"EXPEDIENTES QUE NO CONSTAN EN LAS FUENTES: {sorted(exp_aj)[:8]}. "
            f"Un número de expediente inventado se firma sin que nadie lo note.")

    cant_base = {c.replace(",", "") for c in _RX_CANTIDAD.findall(base)}
    cant_aj = {c.replace(",", "") for c in _RX_CANTIDAD.findall(limpio)} - cant_base
    if cant_aj:
        avisos.append(
            f"CANTIDADES QUE NO CONSTAN EN LAS FUENTES: {sorted(cant_aj)[:8]}. "
            f"En una condena, una cifra que no viene de autos es un error grave.")
    return avisos
