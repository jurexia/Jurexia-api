"""
El sello de vigencia de las tesis: cuál perdió vigencia, por cuál y desde cuándo — 25-sep-2026.

    import vigencia_tesis as vt
    v = vt.de("2009817")
    vt.etiqueta(v)
    → «ABANDONADA por la P./J. 2/2022 (11a.), registro 2024159, desde el 11 de febrero de 2022»

POR QUÉ EXISTE
--------------
David preguntó si los Tribunales Colegiados pueden ejercer control difuso sobre
normas del juicio de origen y el chat le dio como vigentes la P. X/2015 (10a.)
(2009817) y la P. IX/2015 (2009816), ABANDONADAS por la P./J. 2/2022 (11a.)
(2024159) desde febrero de 2022. La nota estaba en el acervo —en
`payload["precedentes"]` de `jurisprudencia_nacional_v3`— y ninguna ruta la
llevaba al modelo: le llegaban la cabecera y el rubro (TESIS_SOLO_RUBRO), sin
un solo atributo de vigencia, y el sello final dio por buenas las dos porque su
registro existe. La sustituta, además, queda en el puesto 38 del vector de la
abandonada: por similitud nunca llega.

Este módulo no busca nada: lee dos archivos pequeños y responde por registro.
  · datos/vigencia_tesis.json — las 558 tesis cuya pérdida de vigencia CONSTA
    en el Semanario (nota de la propia tesis, de la que la reemplaza o de una
    tercera). Lo genera scripts/vigencia_tesis_generar.py; precisión medida
    contra el SJF: 40/40 en muestra estratificada, más 11/11.
  · datos/vigencia_curada.json — lo que las reglas no pueden ver porque el
    Semanario no lo anota (160584, P. LXVI/2011, «orientadores», frente a la
    P./J. 21/2014). Con fuente «curaduria» y otra redacción: «superada en los
    hechos; el Semanario no lo anota». La curada NUNCA pisa a la expresa: si
    mañana el Semanario anota la tesis, manda su nota.

LO QUE NO ES
------------
No es «ver jurisprudencia»: las 6,282 tesis que contendieron en una
contradicción (o cuya contradicción se declaró sin materia, improcedente o
inexistente) NO perdieron vigencia y aquí no están. Mezclarlas le pondría
«abandonada» a criterios vigentes.

UN FALLO AQUÍ NO PUEDE COSTAR LA CONSULTA
-----------------------------------------
Todo es perezoso y tolerante, como `linea_coidh.lineas()`: si falta un archivo
o está roto se avisa UNA vez por consola, `de()` devuelve None y la consulta
sigue exactamente como antes del sello. Ninguna función de aquí lanza.
"""
from __future__ import annotations

import html
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

RUTA_INDICE = Path(__file__).resolve().parent / "datos" / "vigencia_tesis.json"
RUTA_CURADA = Path(__file__).resolve().parent / "datos" / "vigencia_curada.json"

# Lo que dice la etiqueta de cada estado del índice (vigencia_tesis_generar.py).
_ESTADO = {
    "abandonada": "ABANDONADA",
    "interrumpida": "INTERRUMPIDA",
    "sustituida": "SUSTITUIDA",
    "superada": "SUPERADA",
    "modificada": "MODIFICADA",
    "sin_efectos": "SIN EFECTOS",
    "aclarada": "ACLARADA",
    "texto_sustituido": "TEXTO SUSTITUIDO",
}
# La aclaración y la republicación corregida no cambian el criterio: cambian
# el texto que se cita. Se avisan con otra frase, no con «perdió vigencia».
_CORRECCION = ("aclarada", "texto_sustituido")

# De quién es la nota que se muestra: la del Semanario en la propia tesis no
# vale lo mismo que la de la tesis nueva, ni que nuestra curaduría.
_ORIGEN_NOTA = {
    "nota_propia": "Nota del Semanario en esta tesis",
    "tesis_nueva": "Nota de la tesis que la reemplaza",
    "cita_tercera": "Nota del Semanario en otra tesis",
    "curaduria": "Curaduría Iurexia",
}

_MESES = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
          "septiembre", "octubre", "noviembre", "diciembre")


# ═══════════════════════════════════════════════════════════════ carga

def _leer(ruta: Path, que: str) -> Dict[str, Dict[str, Any]]:
    """{registro: entrada} de un archivo del índice, o {} si falta o está roto."""
    try:
        crudo = json.loads(ruta.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"   📛 VIGENCIA: no está {ruta.name} ({que}); se sigue sin ese sello")
        return {}
    except Exception as e:
        print(f"   📛 VIGENCIA: no pude leer {ruta.name} ({que}: {type(e).__name__}); se sigue sin ese sello")
        return {}
    tesis = crudo.get("tesis") if isinstance(crudo, dict) and isinstance(crudo.get("tesis"), dict) else crudo
    if not isinstance(tesis, dict):
        print(f"   📛 VIGENCIA: {ruta.name} no tiene la forma esperada; se sigue sin ese sello")
        return {}
    salida: Dict[str, Dict[str, Any]] = {}
    for reg, v in tesis.items():
        r = str(reg).strip()
        if r.isdigit() and isinstance(v, dict) and v.get("estado"):
            salida[r] = dict(v, registro=r)
    return salida


@lru_cache(maxsize=1)
def indice() -> Dict[str, Dict[str, Any]]:
    """Las dos capas, una vez por proceso. La EXPRESA va encima de la curada:
    lo que dice el Semanario manda sobre lo que inferimos nosotros."""
    try:
        todo = dict(_leer(RUTA_CURADA, "capa curada"))
        todo.update(_leer(RUTA_INDICE, "índice expreso"))
        return todo
    except Exception as e:                      # pragma: no cover — defensa última
        print(f"   📛 VIGENCIA: el índice falló ({type(e).__name__}); se sigue sin sello")
        return {}


def de(registro: Any) -> Optional[Dict[str, Any]]:
    """La pérdida de vigencia de la tesis con ese registro digital, o None si
    no consta (o si no hay índice). Devuelve una COPIA: quien la tenga no
    puede tocar la del índice."""
    try:
        r = str(registro or "").strip()
        if not r.isdigit():
            return None
        v = indice().get(r)
        return dict(v) if v else None
    except Exception:
        return None


def cadena(registro: Any, saltos: int = 3) -> List[str]:
    """Los reemplazos en orden, siguiendo la cadena si el reemplazo también
    perdió vigencia (A → B → C), hasta `saltos` y sin ciclos."""
    salida: List[str] = []
    try:
        actual, vistos = str(registro or "").strip(), {str(registro or "").strip()}
        for _ in range(max(0, saltos)):
            v = de(actual)
            sig = str((v or {}).get("por_registro") or "").strip()
            if not sig or sig in vistos:
                break
            salida.append(sig)
            vistos.add(sig)
            actual = sig
    except Exception:
        pass
    return salida


# ═══════════════════════════════════════════════════════════════ cómo se dice

def _fecha(desde: Optional[str]) -> str:
    """«2022-02-11» → «el 11 de febrero de 2022»; «2013-07» → «julio de 2013»."""
    s = str(desde or "")
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m and 1 <= int(m.group(2)) <= 12:
        return f"el {int(m.group(3))} de {_MESES[int(m.group(2)) - 1]} de {m.group(1)}"
    m = re.fullmatch(r"(\d{4})-(\d{2})", s)
    if m and 1 <= int(m.group(2)) <= 12:
        return f"{_MESES[int(m.group(2)) - 1]} de {m.group(1)}"
    return ""


def _corta(texto: Optional[str], tope: int) -> str:
    t = re.sub(r"\s+", " ", str(texto or "")).strip()
    if len(t) <= tope:
        return t
    c = t[:tope - 1]
    esp = c.rfind(" ")
    if esp > tope * 0.6:
        c = c[:esp]
    return c.rstrip(" ,;:") + "…"


def curada(v: Optional[Dict[str, Any]]) -> bool:
    return bool(v) and v.get("fuente") == "curaduria"


def _con_articulo(asunto: str) -> str:
    """«amparo en revisión 151/2021» → «el amparo en revisión 151/2021»: el
    índice guarda el asunto tal como lo nombra la nota, sin artículo."""
    a = str(asunto or "").strip()
    if not a or re.match(r"(?i)(el|la|los|las)\s", a):
        return a
    fem = re.match(r"(?i)(contradicci|solicitud|controversia|acci[óo]n|queja|revisi[óo]n)", a)
    return ("la " if fem else "el ") + a


def etiqueta(v: Optional[Dict[str, Any]]) -> str:
    """La frase que lee el abogado y el modelo, p. ej.:
    «ABANDONADA por la P./J. 2/2022 (11a.), registro 2024159, desde el 11 de febrero de 2022»
    «ABANDONADA EN PARTE por la 1a./J. 67/2014 (10a.), registro …»
    «SUPERADA EN LOS HECHOS por la P./J. 21/2014 (10a.), registro 2006225, desde el 28 de abril
     de 2014 (el Semanario no lo anota)»"""
    try:
        if not v:
            return ""
        est = _ESTADO.get(str(v.get("estado") or ""), str(v.get("estado") or "SIN VIGENCIA").upper())
        if curada(v):
            est += " EN PARTE, EN LOS HECHOS" if v.get("parcial") else " EN LOS HECHOS"
        elif v.get("parcial"):
            est += " EN PARTE"
        if v.get("alcance"):
            # Con rayas y no con paréntesis: el alcance suele nombrar incisos
            # —«inciso d)»— y el paréntesis se cerraba antes de tiempo.
            est += f" —{v['alcance']}—"
        clave, reg = v.get("por_clave"), v.get("por_registro")
        if clave:
            por = f" por la {clave}" + (f", registro {reg}" if reg else "")
        elif reg:
            por = f" por la tesis de registro {reg}"
        elif v.get("por_resolucion"):
            por = f" al resolverse {_con_articulo(v['por_resolucion'])}"
        else:
            por = ""
        f = _fecha(v.get("desde"))
        return (est + por + (f", desde {f}" if f else "")
                + (" (el Semanario no lo anota)" if curada(v) else ""))
    except Exception:
        return ""


def atributos_xml(v: Optional[Dict[str, Any]]) -> str:
    """Los atributos de <documento>, con espacio delante, o "" si la tesis no
    perdió vigencia. Van en el tag porque TESIS_SOLO_RUBRO recorta el cuerpo
    y un aviso metido en el texto se lo come el recorte."""
    try:
        if not v:
            return ""
        e = lambda x: html.escape(str(x), quote=True)  # noqa: E731
        partes = [f' vigencia="{e(v.get("estado"))}"']
        if v.get("parcial"):
            partes.append(' vigencia_parcial="si"')
        if v.get("por_registro"):
            partes.append(f' reemplazada_por="{e(v["por_registro"])}"')
        if v.get("por_clave"):
            partes.append(f' reemplazo_clave="{e(v["por_clave"])}"')
        if v.get("desde"):
            partes.append(f' vigencia_desde="{e(v["desde"])}"')
        if curada(v):
            partes.append(' vigencia_fuente="curaduria"')
        return "".join(partes)
    except Exception:
        return ""


def linea_visible(v: Optional[Dict[str, Any]], tope_nota: int = 220) -> str:
    """La línea que va DENTRO del contenido, antes del rubro: el atributo lo
    lee un modelo atento; la línea la lee cualquiera, y la nota literal es la
    prueba de que no nos lo inventamos."""
    try:
        if not v:
            return ""
        if v.get("estado") in _CORRECCION:
            cabeza = "⚠️ TEXTO CORREGIDO: "
            cola = " Cita la versión corregida."
        elif v.get("parcial"):
            cabeza = "⚠️ PERDIÓ VIGENCIA EN PARTE: "
            cola = " En esa parte no la presentes como vigente."
        else:
            cabeza = "⚠️ PERDIÓ VIGENCIA: "
            cola = " No la presentes como vigente."
        nota = re.sub(r"^\s*Nota\s*:\s*", "", str(v.get("nota") or ""))
        origen = _ORIGEN_NOTA.get(str(v.get("fuente") or ""), "Nota")
        prueba = f" {origen}: «{_corta(nota, tope_nota)}»" if nota.strip() else ""
        return cabeza + etiqueta(v) + "." + prueba + cola
    except Exception:
        return ""


def marcador(v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Lo que viaja a la app en FUENTES_PREVIAS, CITATION_META y /cita, para
    que el visor pueda pintar la franja después (el frontend aún no la lee)."""
    try:
        if not v:
            return None
        return {
            "estado": v.get("estado"),
            "etiqueta": etiqueta(v),
            "parcial": bool(v.get("parcial")),
            "por_registro": v.get("por_registro"),
            "por_clave": v.get("por_clave"),
            "desde": v.get("desde"),
            "fuente": v.get("fuente"),
        }
    except Exception:
        return None
