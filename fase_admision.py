"""LA FICHA SE LEE DEL AUTO DE ADMISIÓN, NO SE TECLEA.

David, 11-sep-2026: «basta con subir el auto de admisión y de allí derivar qué
expediente, qué tribunal resolverá, la autoridad responsable, el o los terceros
interesados; lo mismo con la revisión y los demás recursos. Esto elimina la
necesidad de meter todos los datos. Pero sólo déjalo como posibilidad
optativa».

Tiene razón y el documento se lo pone fácil: el auto de admisión es el único
papel del expediente que dice, con todas sus letras y en su primera página,
quién es quién y cómo se llama el asunto. Medido sobre los dos autos reales que
guarda el taller:

  A.D.C. 393/2025   «…Secretario de Acuerdos del Tercer Tribunal Colegiado en
                    Materias Administrativa y Civil del Vigésimo Segundo
                    Circuito…», «…fórmese el expediente número 393/2025…»,
                    «…demanda de amparo presentada por Nancy López
                    Hernández…», «…tercera interesada … al Instituto del Fondo
                    Nacional de la Vivienda para los Trabajadores…»
  R.F. 91/2025      «…fórmese el expediente número 91/2025…», «…oficio de
                    expresión de agravios presentado por Karen Yadira Meza
                    Ruiz, Administradora Desconcentrada Jurídica de
                    Querétaro…», «…Expediente 695/25-09-01-7-OT…»

LO QUE SE LEE SIN MODELO Y LO QUE NO. El número del expediente, el tribunal y
la ciudad están en fórmulas fijas —«fórmese el expediente número», «Secretario
de Acuerdos del»— y se leen con un barrido: no hay razón para pagarle a un
modelo por eso ni para arriesgarse a que lo reformule. Los NOMBRES de las
partes no tienen fórmula fija y ahí sí entra el modelo.

Y NADA DE LO QUE DEVUELVE SE ACEPTA SIN COMPROBAR QUE ESTÁ EN EL PAPEL. Cada
valor tiene que aparecer LITERAL en el texto del auto —normalizando espacios y
tildes— o se tira. Es una ficha que el secretario va a firmar: un nombre
inventado aquí viaja a la carátula y al resolutivo.

LA TRAMPA DE ESTE DOCUMENTO, y hay que decirla porque no es obvia: el auto
nombra DOS autoridades responsables. En el 393/2025 dice «se tiene como
autoridad responsable al Juzgado Quinto de Primera Instancia Civil», que es la
EJECUTORA, mientras que la sentencia reclamada la dictó la Primera Sala Civil,
que es la ORDENADORA. La que va en la carátula y en el resolutivo es la
ordenadora. Se piden las dos, separadas, y la ejecutora sale sólo como dato.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata

MODELO_ADMISION = os.getenv("MODELO_ADMISION", "gpt-5.6-luna")


def _plano(x: str) -> str:
    x = unicodedata.normalize("NFD", (x or "").lower())
    x = "".join(c for c in x if unicodedata.category(c) != "Mn")
    return " ".join(x.split())


# ── lo que se lee sin modelo ───────────────────────────────────────────────
_RX_NUMERO = re.compile(
    r"f[óo]rmese\s+el\s+expediente\s+n[úu]mero\s+([0-9]{1,5}\s*/\s*[0-9]{2,4})", re.I)
# El rótulo de la cabecera —«A.D.C.393/2025», «R.F. 91/2025»— es el respaldo.
_RX_ROTULO = re.compile(
    r"\b(A\.?\s?D\.?\s?C?\.?|R\.?\s?F\.?|A\.?\s?R\.?|R\.?\s?Q\.?|Q\.?\s?A\.?)\s*"
    r"([0-9]{1,5}\s*/\s*[0-9]{2,4})\b")
_RX_TRIBUNAL = re.compile(
    r"Secretari[oa]\s+de\s+Acuerdos\s+d[e]l\s+((?:Primer|Segundo|Tercer|Cuarto|"
    r"Quinto|Sexto|S[ée]ptimo|Octavo|Noveno|D[ée]cimo)[\w\sáéíóúñ]{10,120}?"
    r"Circuito)", re.I)
# DOS PALABRAS ANTES DE LA COMA SÓLO SI LAS UNE UN «de». Sin eso, «Conste
# Querétaro, Querétaro» —la palabra final de la razón de cuenta pegada a la
# ciudad— entraba entera y la carátula habría dicho «Conste Querétaro».
# «Santiago de Querétaro, Querétaro» sí es el nombre de una ciudad.
_RX_CIUDAD = re.compile(
    r"([A-ZÁÉÍÓÚÑ][\wáéíóúñ]+(?:\s+de\s+[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+)?,\s*"
    r"[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+),\s+(?:a\s+)?(?:los\s+)?(?:uno|dos|tres|cuatro|"
    r"cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|"
    r"diecis[éeí]is|diecisiete|dieciocho|diecinueve|veinti\w+|treinta)\s+de\s+"
    r"(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|"
    r"noviembre|diciembre)", re.I)

# El rótulo dice de qué recurso se trata, y es la forma más corta y fiable.
_POR_ROTULO = {"adc": "amparo_directo", "ad": "amparo_directo",
               "rf": "revision_fiscal", "ar": "amparo_revision",
               "rq": "queja", "qa": "queja"}


def _norm_num(x: str) -> str:
    return re.sub(r"\s+", "", x or "")


def deterministas(texto: str) -> dict:
    """Lo que está en fórmula fija. Sin modelo y sin coste."""
    t = texto or ""
    plano = " ".join(t.split())
    fuera: dict = {}

    m = _RX_NUMERO.search(plano)
    if m:
        fuera["numero"] = _norm_num(m.group(1))
    r = _RX_ROTULO.search(plano[:400])
    if r:
        _rot = re.sub(r"[.\s]", "", r.group(1)).lower()
        fuera["tipo_asunto"] = _POR_ROTULO.get(_rot, "")
        # EL RÓTULO CONFIRMA EL NÚMERO, no lo sustituye: «fórmese el expediente
        # número» es la orden; el rótulo es cómo se archiva. Si discrepan, no
        # se elige: se dice. Un expediente equivocado en la carátula manda el
        # proyecto al asunto de otro.
        if fuera.get("numero") and _norm_num(r.group(2)) != fuera["numero"]:
            fuera["aviso_numero"] = (
                f"EL AUTO SE CONTRADICE SOBRE EL NÚMERO: en la cabecera dice "
                f"«{_norm_num(r.group(2))}» y en el acuerdo «{fuera['numero']}». "
                f"Comprueba cuál es antes de seguir.")
        elif not fuera.get("numero"):
            fuera["numero"] = _norm_num(r.group(2))

    m = _RX_TRIBUNAL.search(plano)
    if m:
        fuera["tribunal"] = " ".join(m.group(1).split())
    m = _RX_CIUDAD.search(plano)
    if m:
        fuera["ciudad"] = " ".join(m.group(1).split())
    return fuera


# ── y lo que sí necesita leerse ───────────────────────────────────────────
def prompt(texto: str, tipo: str = "") -> str:
    import tipos_asunto as _ta
    _t = tipo or "amparo_directo"
    try:
        _q = _ta.caratula_de(_t)
        _figuras = ", ".join(f"«{e.lower()}»" for e, _c, _o in _q)
    except Exception:
        _figuras = "«quejoso», «autoridad responsable», «tercero interesado»"
    return f"""Eres el secretario de un Tribunal Colegiado fichando un asunto que
acaba de llegar. Tienes delante el AUTO DE ADMISIÓN y sólo eso.

Extrae los nombres de las partes. Reglas que no se negocian:

- COPIA LITERAL. Cada nombre tiene que aparecer en el auto tal como lo
  escribes, palabra por palabra. No lo abrevies, no lo completes, no lo
  corrijas. Si el auto escribe «INFONAVIT», escribes «INFONAVIT».
- SI NO CONSTA, CADENA VACÍA. Es una ficha que se va a firmar: un nombre
  inventado aquí acaba en la carátula y en el punto resolutivo.
- DOS AUTORIDADES, NO UNA. El auto suele nombrar a la ORDENADORA —la que
  dictó el acto que se reclama, normalmente una Sala o un Tribunal— y a la
  EJECUTORA —la que lo ejecuta, normalmente un Juzgado—. La que va en la
  carátula es la ORDENADORA. Ponlas en campos distintos y no las mezcles.
- VARIOS TERCEROS. Si hay más de un tercero interesado, sepáralos con « y ».

Las figuras de este tipo de asunto son: {_figuras}.

EL AUTO:
──────────────────────────────────────────
{texto[:24000]}
──────────────────────────────────────────

Devuelve JSON y nada más:
{{"quejoso": "quien promueve el amparo o el recurso, literal",
 "responsable_ordenadora": "la que dictó el acto reclamado, literal",
 "responsable_ejecutora": "la que lo ejecuta, literal, o vacío",
 "tercero_interesado": "literal, o vacío",
 "expediente_origen": "el número del juicio o toca de origen, o vacío",
 "magistrado": "el ponente, SÓLO si el auto lo dice, o vacío"}}"""


_CAMPOS = ("quejoso", "responsable_ordenadora", "responsable_ejecutora",
           "tercero_interesado", "expediente_origen", "magistrado")


def _esta_en_el_papel(valor: str, plano_texto: str) -> bool:
    """¿Aparece literal en el auto? Se toleran el «y» de una lista y poco más."""
    v = _plano(valor)
    if len(v) < 3:
        return False
    if v in plano_texto:
        return True
    # Una lista de terceros separada por « y »: cada mitad tiene que constar.
    partes = [p.strip() for p in re.split(r"\s+y\s+", v) if len(p.strip()) > 3]
    return bool(partes) and all(p in plano_texto for p in partes)


async def leer(cliente, texto: str, tipo: str = "") -> dict:
    """La ficha que se propone al secretario, con lo que no se pudo leer."""
    base = deterministas(texto)
    ficha = {k: "" for k in _CAMPOS}
    avisos = []
    if base.get("aviso_numero"):
        avisos.append(base.pop("aviso_numero"))
    if cliente and (texto or "").strip():
        try:
            import llamada_modelo as _lm
            r = await _lm.crear(
                cliente, model=MODELO_ADMISION,
                messages=[{"role": "user",
                           "content": prompt(texto, base.get("tipo_asunto") or tipo)}],
                max_completion_tokens=1200,
                response_format={"type": "json_object"})
            crudo = (r.choices[0].message.content or "").strip()
            m = re.search(r"\{.*\}", crudo, re.S)
            d = json.loads(m.group(0) if m else crudo)
            plano = _plano(texto)
            for k in _CAMPOS:
                v = " ".join(str(d.get(k) or "").split())
                if not v:
                    continue
                if _esta_en_el_papel(v, plano):
                    ficha[k] = v
                else:
                    # NO SE ACEPTA LO QUE NO ESTÁ. Y se dice cuál se tiró: un
                    # campo que desaparece en silencio parece un dato que el
                    # auto no traía, y no es lo mismo.
                    avisos.append(
                        f"SE DESCARTÓ «{v[:70]}» como {k.replace('_', ' ')}: no "
                        f"aparece con esas palabras en el auto. Escríbelo tú si "
                        f"es el correcto.")
        except Exception as ex:
            avisos.append(f"No se pudieron leer las partes del auto: "
                          f"{type(ex).__name__}. La ficha sale sólo con lo que "
                          f"se lee sin modelo; complétala a mano.")
    ficha.update({k: v for k, v in base.items() if v})
    ficha["avisos"] = avisos
    # La ordenadora es la que va a la carátula; el campo del formulario se
    # llama `responsable` y es el que el resto del sistema consume.
    ficha["responsable"] = ficha.get("responsable_ordenadora", "")
    return ficha
