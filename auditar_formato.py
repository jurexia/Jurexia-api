"""Audita un proyecto generado contra el formato REAL de los engroses.

No compara contra lo que yo creo que debe ser: compara contra lo que midieron
20,000 párrafos de las carpetas del tribunal. Cada regla lleva al lado el
porcentaje del corpus que la cumple, para que se vea que no es una opinión.
"""

from __future__ import annotations

import re
import sys
from collections import Counter

from docx import Document

# ── La especificación, medida ────────────────────────────────────────────
#
# NO es «el valor más frecuente»: es el CONJUNTO de valores que cubre ~80% del
# corpus. La diferencia importa. En un .docx `None` significa «hereda del
# estilo», y para el interlineado de una transcripción `None` (49%) es tan común
# como 1.0 (48%): exigir 1.0 marcaba como defectuoso el proyecto del propio
# secretario. Un auditor que reprueba al maestro está midiendo mal.
#
# tipo → (campo, valores aceptables, % del corpus que cae dentro)
ESPERADO = {
    "cuerpo": [
        ("first_line_indent", {709, 708, None}, 84),
        ("alignment", {"JUSTIFY", None}, 99),
        ("line_spacing", {1.5, None}, 92),
        ("size", {14.0, None}, 90),
        ("italic", {False}, 97),
    ],
    "transcripcion": [
        # Aquí None NO vale: el 82% lleva alguna sangría explícita.
        ("left_indent", {709, 708, 851}, 82),
        ("first_line_indent", {None, 0, 11}, 89),
        ("alignment", {"JUSTIFY", None}, 99),
        ("line_spacing", {1.0, None}, 97),
        ("size", {13.0, None}, 86),
        ("italic", {True}, 100),
    ],
    "encabezado": [
        ("alignment", {"JUSTIFY", None}, 96),
        ("line_spacing", {1.5, None}, 93),
        ("size", {14.0, None}, 88),
        ("italic", {False}, 100),
    ],
}

_RX_RUBRO = re.compile(r"[“\"][A-ZÁÉÍÓÚÑ][^”\"]{18,}")
_ORD = re.compile(r"^(PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|S[ÉE]PTIMO|OCTAVO|ÚNICO)\.")
_ROT = re.compile(r"^(Consideraciones|Conceptos de|Agravios|Problema|Solución|Estudio|Antecedentes|TEMA)")


def clase(p, t: str):
    if not t:
        return None
    rs = [r for r in p.runs if (r.text or "").strip()]
    if not rs:
        return None
    # La cursiva manda sobre el ordinal: los RESOLUTIVOS de la sentencia
    # recurrida se transcriben y empiezan por «SEGUNDO.», pero son cita ajena,
    # no encabezado propio. Comprobar el ordinal primero los clasificaba mal y
    # el auditor reprobaba engroses correctos.
    if all(r.italic for r in rs) and len(t.split()) > 40:
        return "transcripcion"
    if _ORD.match(t):
        return "encabezado"
    if all(r.bold for r in rs) and len(t.split()) < 14 and _ROT.match(t):
        return "rotulo"
    if _RX_RUBRO.search(t) and len(t.split()) < 90:
        return "cita_rubro"
    if len(t.split()) > 25:
        return "cuerpo"
    return None


def _leer(p, campo: str):
    pf = p.paragraph_format
    rs = [r for r in p.runs if (r.text or "").strip()]
    if campo in ("left_indent", "first_line_indent"):
        v = getattr(pf, campo)
        return None if v is None else round(v.twips)
    if campo == "alignment":
        return str(pf.alignment).split()[0] if pf.alignment is not None else None
    if campo == "line_spacing":
        return round(pf.line_spacing, 2) if pf.line_spacing else None
    if campo == "size":
        return rs[0].font.size.pt if rs and rs[0].font.size else None
    if campo == "italic":
        return bool(rs and all(r.italic for r in rs))
    return None


def auditar(ruta: str, desde: str = "SEXTO") -> list[str]:
    """Los defectos de formato del estudio. Sólo mira lo que generamos."""
    doc = Document(ruta)
    ps = list(doc.paragraphs)
    inicio = next((i for i, p in enumerate(ps)
                   if p.text.strip().startswith(desde)), 0)

    fallos: Counter = Counter()
    total: Counter = Counter()
    ejemplo: dict = {}
    for p in ps[inicio:]:
        t = p.text.strip()
        c = clase(p, t)
        if c not in ESPERADO:
            continue
        total[c] += 1
        for campo, aceptables, pct in ESPERADO[c]:
            real = _leer(p, campo)
            if real not in aceptables:
                esp = sorted((x for x in aceptables if x is not None),
                             key=str) or ["heredado"]
                fallos[(c, campo, esp[0], real, pct)] += 1
                ejemplo.setdefault((c, campo), t[:70])

    avisos = []
    for (c, campo, esp, real, pct), n in fallos.most_common():
        if n < max(2, 0.25 * total[c]):
            continue
        avisos.append(
            f"{c}/{campo}: {n} de {total[c]} párrafos tienen {real!r}; el "
            f"{pct}% del corpus usa {esp!r} o lo hereda. "
            f"Ej.: «{ejemplo.get((c, campo),'')}…»")
    return avisos


if __name__ == "__main__":
    for a in auditar(sys.argv[1]) or ["sin defectos de formato ✔"]:
        print(" ·", a)
