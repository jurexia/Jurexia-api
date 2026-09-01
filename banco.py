"""EL BANCO DE PLANTILLAS — las frases del oficio, no prosa de máquina.

David, 30-ago-2026: «El generador crea los resultandos y considerandos con un
formato muy genérico. Me parece que puedes crear una base de datos para las
plantillas que sirva para los secretarios de toda la república y que su
adelanto sea verdaderamente útil y no tengan que ajustarlo a sus formatos».

Tiene razón en el diagnóstico: el modelo escribía una competencia correcta y
anodina donde el oficio tiene UNA frase, con su cadena de fundamentos en un
orden que no es casual. Un secretario que recibe eso lo reescribe entero, y
entonces el adelanto no le ahorró nada.

MEDIDO SOBRE 363 DOCUMENTOS del corpus: 62 para la competencia, 86 para la
existencia y la legitimación, 44 para los resultandos, 87 para la dispensa y el
resolutivo, 30 quejas, 26 revisiones y 28 revisiones fiscales. De cada apartado
se sacó la redacción LITERAL más frecuente, con los datos sustituidos por
marcadores.

LAS DOS REGLAS QUE LO HACEN SEGURO:

1. UN MARCADOR QUE NO SÉ RELLENAR SE QUEDA EN HUECO VISIBLE. Nunca se inventa
   y nunca se borra dejando la frase coja: `*********` se ve, y lo que se ve se
   rellena. Es la misma disciplina que el resto del redactor.

2. LA PLANTILLA NO SUSTITUYE AL CRITERIO. Sólo se usa donde la redacción es
   FORMAL —competencia, existencia, legitimación, dispensa, resolutivo—. Los
   antecedentes y el estudio los sigue escribiendo el modelo con el expediente
   delante, porque ahí no hay fórmula que valga.

EL BANCO ES DEL TRIBUNAL QUE LO GENERÓ. Las frases traen su nombre y sus
acuerdos; los marcadores {tribunal} y {responsable} los cambian, pero un
secretario de otro circuito debe revisar la cadena de fundamentos: el punto
tercero, fracción XXII, del Acuerdo General 3/2013 es el que reparte la
jurisdicción de ÉSTE. Se avisa.
"""

from __future__ import annotations

import json
import os
import re

_RUTA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "banco_plantillas.json")
HUECO = "*********"

# Qué tipo del pipeline corresponde a qué llave del banco.
_LLAVE = {
    "amparo_directo": "amparo-directo",
    "amparo_revision": "revision",
    "queja": "queja",
    "revision_fiscal": "revision",     # sin banco propio: se avisa
}

_BANCO = None


def cargar() -> dict:
    global _BANCO
    if _BANCO is None:
        try:
            with open(_RUTA, encoding="utf8") as fh:
                _BANCO = json.load(fh)
        except Exception as e:
            print(f"   ⚠️ banco de plantillas no disponible: {e}")
            _BANCO = {}
    return _BANCO


def del_tipo(tipo: str) -> dict:
    return cargar().get(_LLAVE.get((tipo or "").strip().lower(), ""), {}) or {}


def apartado(tipo: str, ident: str) -> dict:
    """El apartado del banco cuyo id contiene `ident`."""
    t = del_tipo(tipo)
    for grupo in ("considerandos", "resultandos"):
        for a in (t.get(grupo) or []):
            if ident in str(a.get("id", "")) or ident in str(a.get("rotulo", "")).lower():
                return a
    return {}


# EL NOMBRE DEL TRIBUNAL VIENE ESCRITO A MANO EN LAS PLANTILLAS, no como
# marcador: el extractor copió la frase literal y ahí estaba. Para un
# secretario de Yucatán eso significaría firmar el tribunal de Querétaro, que
# es exactamente lo que este trabajo existe para evitar. Se convierte en
# marcador antes de rellenar.
# El «Este» se queda fuera del marcador: forma parte de la frase, no del
# nombre. Comérselo dejaba «Primer Tribunal Colegiado… es competente» sin
# artículo, que no es español.
_RX_TRIBUNAL_CORPUS = re.compile(
    r"Tercer\s+Tribunal\s+Colegiado\s+en\s+Materias\s+"
    r"Administrativa\s+y\s+Civil\s+del\s+Vig[ée]simo\s+Segundo\s+Circuito",
    re.I)
_RX_ESTADO_CORPUS = re.compile(r"del\s+Estado\s+de\s+Quer[ée]taro", re.I)


def _a_marcadores(texto: str) -> str:
    """Convierte en marcador lo que el extractor copió literal del corpus."""
    t = _RX_TRIBUNAL_CORPUS.sub("{tribunal}", texto or "")
    return t


_RX_MARCA = re.compile(r"\{([a-z_áéíóúñ0-9]+)\}", re.I)


def rellenar(texto: str, datos: dict) -> tuple:
    """(texto relleno, marcadores que quedaron en hueco)."""
    if not texto:
        return "", []
    faltan = []

    def _uno(m):
        clave = m.group(1)
        v = datos.get(clave)
        if v is None or not str(v).strip():
            faltan.append(clave)
            return HUECO
        return str(v)

    return _RX_MARCA.sub(_uno, texto), faltan


def _sin_rotulo(texto: str) -> str:
    """Quita el «PRIMERO. Competencia.» que la plantilla trae dentro.

    El compositor pone el ordinal —lo calcula— y el rótulo. Si la plantilla lo
    trae también, sale dos veces.
    """
    t = (texto or "").strip()
    t = re.sub(r"^\s*(?:\{ordinal\}|PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|"
               r"SEXTO|S[ÉE]PTIMO|OCTAVO)\s*\.\s*", "", t)
    # y el rótulo propio del apartado, que va en negrita aparte
    t = re.sub(r"^[A-ZÁÉÍÓÚÑ][^.]{3,60}\.\s+(?=[A-ZÉ])", "", t, count=1)
    return t.strip()


def texto_de(tipo: str, ident: str, datos: dict) -> tuple:
    """(párrafo listo, marcadores en hueco). Vacío si no hay plantilla."""
    a = apartado(tipo, ident)
    if not a or a.get("generado"):
        return "", []
    pl = a.get("plantilla") or ""
    if not pl.strip():
        return "", []
    relleno, faltan = rellenar(_a_marcadores(_sin_rotulo(pl)), datos)
    return relleno, faltan


def nota_de(tipo: str, ident: str) -> str:
    """La nota al pie fija de ese apartado, si la lleva."""
    a = apartado(tipo, ident)
    n = a.get("nota_al_pie") or ""
    return n if isinstance(n, str) and len(n) > 20 else ""


# La revisión fiscal toma prestado el banco de la revisión de amparo para las
# fórmulas de trámite, pero NO sus rótulos: el adelanto real de la RF 44/2025
# rotula «SEGUNDO. Legitimación y oportunidad» y el prestado imponía
# «Legitimación y oportunidad para interponer el recurso de revisión», que es
# del amparo en revisión. Los rótulos de estos tipos salen del catálogo, que
# los tiene medidos sobre sus propios adelantos.
PRESTADO = {"revision_fiscal"}


def rotulo_de(tipo: str, ident: str, por_defecto: str = "") -> str:
    if (tipo or "").strip().lower() in PRESTADO:
        return por_defecto
    a = apartado(tipo, ident)
    r = str(a.get("rotulo") or "").strip()
    r = re.sub(r"^\s*(?:\{ordinal\}|PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|"
               r"SEXTO|S[ÉE]PTIMO|OCTAVO)\s*\.\s*", "", r)
    return r or por_defecto


def aviso_de_procedencia(tipo: str, tribunal: str) -> str:
    """Lo que hay que decirle a un secretario de OTRO circuito.

    El banco salió de un tribunal concreto. Los nombres se sustituyen, pero la
    cadena de fundamentos incluye el acuerdo que reparte SU jurisdicción, y eso
    no lo arregla un marcador.
    """
    t = del_tipo(tipo)
    if not t:
        return ""
    if "Vigésimo Segundo" in (tribunal or ""):
        return ""
    return ("Las fórmulas de competencia y trámite salen del corpus del Tercer "
            "Tribunal Colegiado del Vigésimo Segundo Circuito. Los nombres ya "
            "se sustituyeron, pero la cadena de fundamentos cita el Acuerdo "
            "General que reparte SU jurisdicción: revísala contra la de tu "
            "circuito antes de firmar.")
