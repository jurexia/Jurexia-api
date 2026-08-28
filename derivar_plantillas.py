"""Deriva las plantillas del adelanto a partir de los propios documentos.

LA PLANTILLA NO LA ESCRIBIMOS NOSOTROS. Es la regla de oro del corpus, y aquí
se aplica literalmente: en vez de teclear el considerando de competencia a
mano, se toman las N versiones que el secretario ya firmó, se alinean, y lo que
sobrevive idéntico en la mayoría es el machote; los huecos son los datos.

Medido el 28-ago-2026 sobre 387 competencias del corpus KINGSTON: dentro de un
mismo tipo de asunto son 90% idénticas en amparo directo, 93% en queja y 74% en
amparo en revisión. Es decir, son plantillas, y merecen tratarse como tales.

    python derivar_plantillas.py "amparo directo" "PRIMERO. Competencia"
"""

from __future__ import annotations

import collections
import difflib
import glob
import re
import sys
from typing import Iterable, Optional

try:
    import docx
except ImportError:                                    # pragma: no cover
    docx = None

RAIZ = "/Volumes/KINGSTON"

# ── Qué asunto es, por el nombre del archivo ──────────────────────────────
_PREFIJOS = [
    (("ADA", "ADC", "AD "), "amparo directo"),
    (("ARA", "ARC", "AR "), "amparo en revisión"),
    (("QC", "QA", "RQ", "Q "), "queja"),
    (("RF",), "revisión fiscal"),
    (("CC",), "conflicto competencial"),
    (("RC", "REC"), "reclamación"),
    (("IN", "INC"), "inconformidad"),
]


def tipo_de_asunto(nombre: str) -> Optional[str]:
    n = nombre.upper()
    for prefijos, etiqueta in _PREFIJOS:
        if n.startswith(prefijos):
            return etiqueta
    return None


# ── Las secciones que se pueden derivar ───────────────────────────────────
SECCIONES = {
    "PRIMERO. Presentación": r"^PRIMERO\.\s*Presentaci[óo]n",
    "SEGUNDO. Derechos humanos": r"^SEGUNDO\.\s*Derechos humanos",
    "TERCERO. Tercero interesado": r"^TERCERO\.\s*Parte tercera",
    "CUARTO. Trámite": r"^CUARTO\.\s*Tr[áa]mite",
    "QUINTO. Turno": r"^QUINTO\.\s*Turno",
    "SEXTO. Sesión remota": r"^SEXTO\.\s*Verificaci[óo]n",
    "PRIMERO. Competencia": r"^PRIMERO\.\s*Competencia",
    "SEGUNDO. Existencia": r"^SEGUNDO\.\s*Existencia",
    "TERCERO. Legitimación": r"^TERCERO\.\s*Legitimaci[óo]n",
}


def recolectar(seccion: str, tipo: str, tope: int = 500) -> list[str]:
    """Todas las versiones de una sección para un tipo de asunto."""
    rx = re.compile(SECCIONES[seccion])
    fuera = []
    for ruta in sorted(glob.glob(f"{RAIZ}/**/*.docx", recursive=True)):
        if "~$" in ruta or len(fuera) >= tope:
            continue
        if tipo_de_asunto(ruta.split("/")[-1]) != tipo:
            continue
        try:
            ps = [x.text.strip() for x in docx.Document(ruta).paragraphs]
        except Exception:
            continue
        for p in ps:
            if rx.match(p) and len(p.split()) > 15:
                fuera.append(p)
                break
    return fuera


# ── El derivador ──────────────────────────────────────────────────────────

def derivar(versiones: list[str], umbral: float = 0.8, minimo: int = 14) -> list[str]:
    """Los tramos del machote: lo que sobrevive idéntico en la mayoría.

    SE CUENTA POR POSICIÓN, NO POR CADENA. El primer intento contaba los
    bloques comunes como texto y daba 0%: cada pareja alinea y corta los
    bloques en sitios ligeramente distintos, así que la misma frase aparece
    como veinte cadenas casi iguales y ninguna llega al umbral.

    Lo que sí funciona: sobre una versión de referencia se marca, carácter a
    carácter, en cuántas de las demás versiones coincide. Las posiciones que
    sobreviven en `umbral` de los casos, agrupadas en tramos contiguos de al
    menos `minimo` caracteres, son el machote. Los huecos entre ellos son los
    datos del asunto.
    """
    if len(versiones) < 3:
        return []
    # La referencia es la versión MÁS REPRESENTATIVA —la que más se parece a
    # las demás—, no la más larga: con la más larga se toma un caso atípico.
    muestra = versiones[:40]
    patron = max(muestra, key=lambda a: sum(
        difflib.SequenceMatcher(None, a, b, autojunk=False).quick_ratio()
        for b in muestra if b is not a))

    votos = [0] * len(patron)
    comparadas = 0
    for otra in versiones:
        if otra is patron:
            continue
        comparadas += 1
        sm = difflib.SequenceMatcher(None, patron, otra, autojunk=False)
        for i, _, n in sm.get_matching_blocks():
            for k in range(i, i + n):
                votos[k] += 1
    if not comparadas:
        return []

    minimo_votos = umbral * comparadas
    tramos, actual = [], []
    for k, v in enumerate(votos):
        if v >= minimo_votos:
            actual.append(patron[k])
        else:
            if len(actual) >= minimo:
                tramos.append("".join(actual))
            actual = []
    if len(actual) >= minimo:
        tramos.append("".join(actual))
    return [t for t in (x.strip() for x in tramos) if len(t) >= minimo]


def como_plantilla(tramos: Iterable[str]) -> str:
    """Los tramos, con «{…}» donde va el dato del caso."""
    return " {…} ".join(t.strip() for t in tramos)


def informe(tipo: str, seccion: str, tope: int = 500) -> None:
    v = recolectar(seccion, tipo, tope)
    print(f"\n══ {seccion} · {tipo} ══")
    print(f"   versiones halladas: {len(v)}")
    if len(v) < 3:
        print("   insuficientes para derivar.")
        return
    tramos = derivar(v)
    fijo = sum(len(t) for t in tramos)
    largo = sum(len(x) for x in v) / len(v)
    print(f"   tramos invariantes: {len(tramos)} · {fijo} caracteres fijos "
          f"de {largo:.0f} de media ({fijo / largo:.0%} de machote)\n")
    print("   " + como_plantilla(tramos).replace(" {…} ", "\n   {…}\n   "))


if __name__ == "__main__":
    if docx is None:
        sys.exit("hace falta python-docx")
    tipo = sys.argv[1] if len(sys.argv) > 1 else "amparo directo"
    seccion = sys.argv[2] if len(sys.argv) > 2 else "PRIMERO. Competencia"
    informe(tipo, seccion)
