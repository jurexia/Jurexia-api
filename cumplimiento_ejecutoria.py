# -*- coding: utf-8 -*-
"""LA SENTENCIA RECLAMADA DICTADA EN CUMPLIMIENTO DE UNA EJECUTORIA (30-sep-2026).

David, sobre el AD 323/2025 (soporte@): «la sentencia había sido dictada en
cumplimiento. […] Sólo si se dejó plenitud de jurisdicción o libertad de
jurisdicción a la responsable, el ulterior amparo puede ser materia de
análisis […]. De lo contrario, el juicio de amparo es improcedente porque se
trata de una sentencia dictada en cumplimiento a una sentencia de amparo. No
podemos entrar en un bucle de amparos. Puede suceder que la resolución sea
compleja: por un lado haya libertad de jurisdicción y por otro no. En el caso de
que no lo haya, los conceptos son inoperantes, y sólo se analiza lo relativo a
lo resuelto en plenitud de jurisdicción. Esto búscalo en tesis de la Suprema
Corte, ya está definido.»

LAS REGLAS, verificadas en `jurisprudencia_nacional_v3` el 30-sep-2026 (el
Semanario no dejó consultarse: pide captcha):

  · MIXTA —la ejecutoria vinculó una parte y dejó libre otra—: NO se sobresee.
    Lo que combate lo vinculado es inoperante; lo libre se estudia y se concede
    o se niega. 2a./J. 113/2012 (2001857, contradicción), XI.1o.A.T.15 K
    (2008199), II.1o.T. J/7 (2015559), I.11o.C. J/10 K (2025645).
  · TODO VINCULADO —la ejecutoria fijó el sentido y la responsable sólo acató—:
    improcedencia (art. 61, fr. IX, de la Ley de Amparo) y se SOBRESEE (art.
    63, fr. V). 2a./J. 113/2012 (parte final), 1a./J. 57/2018 (2018315),
    2a./J. 140/2007 (171753), 2a. CVI/2017 (2014516).
  · EXCESO O DEFECTO en el cumplimiento: inoperante sin sobreseer; es materia de
    la vista del artículo 196, del recurso de inconformidad (art. 201) o de la
    denuncia de repetición. P./J. 98/97 (197240), I.5o.C. J/3 (2007297).
  · CÓMO SE DISTINGUE: por los efectos Y por las consideraciones y lineamientos
    de la ejecutoria, no sólo por sus efectos. 1a./J. 75/2014 (2007970),
    1a. CX/2015 (2008717), 2a./J. 9/2016 (2010987).
  · LA NORMA CUYA APLICACIÓN ORDENÓ LA EJECUTORIA, tachada de inconstitucional
    por quien fue tercero interesado en el primer amparo: CRITERIO DIVIDIDO. La
    Primera Sala (1a. IX/2022, 2024239) admite estudiarla por primera vez contra
    la sentencia de cumplimiento, respetando la legalidad fijada en la
    ejecutoria; la Segunda (2a. CXLVII/2017, 2015151) exigía combatirla en
    revisión contra la ejecutoria. P./J. 2/2013 (2002704) sólo impone la
    inoperancia si fue EL MISMO quejoso quien pudo impugnarla antes. Eso no lo
    decide el código: se le presenta al secretario.

LO QUE HACE ESTE MÓDULO:
  1. `clasificar`: un modelo (el lector) decide, planteamiento por
     planteamiento, si ataca lo VINCULADO o lo LIBRE —o es mixto, exceso o
     defecto, constitucionalidad o no consta—, con los efectos de la ejecutoria
     delante. Sin los efectos no se clasifica: se piden como constancia.
  2. `aplicar`: lo vinculado y el exceso o defecto quedan INOPERANTES con su
     fórmula —si el secretario no los tocó—, igual que `arbol_decision` hace
     con los accesorios que caen con el principal.
  3. `bloque_contexto`: lo que la propuesta, el plan y el estudio leen
     (`main._con_autos`), con la ejecutoria, sus efectos y la clasificación.
  4. `sobreseer_propuesto` / `sobreseer_confirmado`: el sobreseimiento se
     PROPONE cuando la ejecutoria no dejó libertad alguna, y sólo se escribe
     cuando el secretario lo confirma.

Rige detrás de la bandera `cumplimiento_ejecutoria` («casa» por omisión).
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata

VINCULACIONES = ("vinculado", "libre", "mixto", "exceso_defecto", "constitucionalidad", "no_consta")

# Los apoyos, por registro: se traen del acervo como las tesis de la técnica
# (`tipos_asunto.tecnica_de` → `fase6_rag.tesis_por_registro`), y el prompt
# sólo nombra las que llegaron.
APOYOS_MIXTA = ["2001857", "2008199", "2015559", "2025645"]
APOYOS_IMPROCEDENCIA = ["2018315", "171753", "2014516"]
APOYOS_EXCESO = ["197240", "2007297"]
APOYOS_LINEAMIENTOS = ["2007970", "2008717", "2010987"]
APOYOS_CONSTITUCIONALIDAD = ["2024239", "2015151", "2002704"]


def rige() -> bool:
    try:
        import contexto_taller as _ct
        return bool(_ct.rediseno("cumplimiento_ejecutoria"))
    except Exception:
        return False


def _plano(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "").lower())
    return " ".join("".join(c for c in s if not unicodedata.combining(c)).split())


def _pregunta(p) -> str:
    if isinstance(p, dict):
        return " ".join(str(p.get("pregunta") or "").split())
    return " ".join(str(getattr(p, "problema", "") or getattr(p, "pregunta", "") or p or "").split())


def huella(efectos: str, problemas: list) -> str:
    """La clasificación vale mientras no cambien los efectos ni las preguntas."""
    base = json.dumps([_plano(efectos)[:6000], [_plano(_pregunta(p)) for p in (problemas or [])]],
                      ensure_ascii=False)
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


# ═══ 1 · LA CLASIFICACIÓN ═══════════════════════════════════════════════════

PROMPT_CLASIFICAR = """Eres secretario de estudio y cuenta de un Tribunal Colegiado de Circuito. La sentencia reclamada en
este amparo directo se dictó EN CUMPLIMIENTO de la ejecutoria del {ejecutoria}. Tienes que decidir,
planteamiento por planteamiento, si combate lo que esa ejecutoria dejó VINCULADO a la responsable o lo que ésta
resolvió con LIBERTAD DE JURISDICCIÓN.

LO QUE MANDÓ LA EJECUTORIA (sus efectos y lo que conste de sus consideraciones):
{efectos}

LO QUE RESOLVIÓ LA RESPONSABLE EN LA SENTENCIA RECLAMADA:
{acto}

LOS PLANTEAMIENTOS DEL ASUNTO (cada uno con lo que resolvió la responsable y lo que se combate):
{problemas}

CÓMO SE DECIDE (Suprema Corte: 2a./J. 113/2012, 1a./J. 75/2014, P./J. 98/97):
· vinculado: combate lo que la responsable decidió PORQUE la ejecutoria se lo ordenó —su sentido, sus lineamientos,
  sus consideraciones— o lo que la ejecutoria dejó definido o intocado. Es cosa juzgada.
· libre: combate lo que la responsable resolvió con criterio propio, porque la ejecutoria se lo dejó a su arbitrio
  («con plenitud de jurisdicción», «con libertad de jurisdicción», «se pronuncie sobre…») o no lo tocó y la
  responsable lo resolvió por primera vez o de nuevo.
· mixto: una parte combate lo vinculado y otra lo libre. Di cuál es cuál.
· exceso_defecto: alega que la responsable hizo más o menos de lo que la ejecutoria mandaba.
· constitucionalidad: plantea que es inconstitucional la norma cuya aplicación ordenó la ejecutoria. No decidas su
  suerte: sólo identifícalo.
· no_consta: con lo que tienes no puede saberse.
Se distingue por lo que la ejecutoria ORDENÓ, no por el tema: si mandó aplicar un artículo de cierta manera, discutir
esa aplicación es vinculado; si además mandó resolver «con plenitud» las demás pretensiones, lo que la responsable dijo
de ellas (intereses, montos, plazos, costas, cláusulas…) es libre, aunque verse sobre el mismo contrato.

Decide también, del texto de los efectos, «hay_libertad»: ¿la ejecutoria dejó a la responsable ALGO que resolver con
criterio propio? Si fijó todo el sentido y la responsable sólo tenía que acatar, false.

Devuelve JSON y nada más:
{{"problemas": [{{"n": 1, "vinculacion": "vinculado|libre|mixto|exceso_defecto|constitucionalidad|no_consta",
  "efecto": "el inciso o la parte de la ejecutoria que lo decide, en pocas palabras",
  "por_que": "una frase concreta", "parte_vinculada": "sólo si es mixto", "parte_libre": "sólo si es mixto"}}],
 "hay_libertad": true, "resumen": "dos frases: qué vinculó la ejecutoria y qué dejó libre"}}"""


def _linea_problema(i: int, p) -> str:
    if isinstance(p, dict):
        return (f"{i}. {_pregunta(p)}\n   Resolvió la responsable: {' '.join(str(p.get('resolvio') or '—').split())[:900]}\n"
                f"   Se combate: {' '.join(str(p.get('combate') or '—').split())[:900]}")
    return f"{i}. {_pregunta(p)}"


def _json_de(crudo: str) -> dict:
    t = str(crudo or "").strip()
    candidatos = [t]
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        candidatos.append(m.group(0))
    for s in candidatos:
        try:
            d = json.loads(s)
            return d if isinstance(d, dict) else {}
        except Exception:
            continue
    return {}


def normalizar(d: dict, problemas: list) -> dict:
    """La clasificación con una entrada por problema (en su orden), sin valores
    fuera del catálogo. Nunca lanza."""
    d = d if isinstance(d, dict) else {}
    por_n = {}
    for x in d.get("problemas") or []:
        if isinstance(x, dict):
            try:
                por_n[int(x.get("n"))] = x
            except (TypeError, ValueError):
                continue
    fuera = []
    for i, p in enumerate(problemas or [], 1):
        x = por_n.get(i) or {}
        v = _plano(x.get("vinculacion")).replace(" ", "_")
        v = v if v in VINCULACIONES else "no_consta"
        fuera.append({"pregunta": _pregunta(p), "vinculacion": v,
                      "efecto": " ".join(str(x.get("efecto") or "").split())[:300],
                      "por_que": " ".join(str(x.get("por_que") or "").split())[:600],
                      "parte_vinculada": " ".join(str(x.get("parte_vinculada") or "").split())[:500] if v == "mixto" else "",
                      "parte_libre": " ".join(str(x.get("parte_libre") or "").split())[:500] if v == "mixto" else ""})
    hay = d.get("hay_libertad")
    hay_libertad = bool(hay) if isinstance(hay, bool) else True    # ante la duda, NO se propone sobreseer
    vs = [f["vinculacion"] for f in fuera]
    todo = bool(vs) and all(v in ("vinculado", "exceso_defecto") for v in vs)
    return {"problemas": fuera, "hay_libertad": hay_libertad, "todo_vinculado": todo,
            "resumen": " ".join(str(d.get("resumen") or "").split())[:700]}


async def clasificar(cliente, cumplimiento: dict, problemas: list, resumen_acto: str = "", modelo: str = "") -> dict | None:
    """La clasificación, o None si no hay efectos que leer o el modelo falla.
    Sin los efectos de la ejecutoria no se adivina: se pide la constancia."""
    efectos = str((cumplimiento or {}).get("efectos") or "").strip()
    if not efectos or not problemas:
        return None
    import llamada_modelo as _lm
    if not modelo:
        try:
            import fases123_pipeline as _f
            modelo = _f.MODELO_FASES
        except Exception:
            modelo = "gpt-5.6-luna"
    prompt = PROMPT_CLASIFICAR.format(
        ejecutoria=(cumplimiento or {}).get("ejecutoria") or "amparo anterior",
        efectos=efectos[:9000], acto=" ".join(str(resumen_acto or "").split())[:12000] or "(no consta)",
        problemas="\n".join(_linea_problema(i, p) for i, p in enumerate(problemas, 1)))
    try:
        r = await _lm.crear(cliente, model=modelo, reasoning_effort="medium",
                            response_format={"type": "json_object"}, max_completion_tokens=6000,
                            messages=[{"role": "user", "content": prompt}])
        d = _json_de(r.choices[0].message.content or "")
    except Exception as ex:
        print(f"   ⚠️ CUMPLIMIENTO: no se pudo clasificar ({type(ex).__name__}: {str(ex)[:120]})")
        return None
    out = normalizar(d, problemas)
    out["huella"] = huella(efectos, problemas)
    return out


# ═══ 2 · LA CALIFICACIÓN QUE SE SIGUE ═══════════════════════════════════════

def razon_vinculado(ejecutoria: str, efecto: str, por_que: str) -> str:
    """La razón de un planteamiento inoperante por combatir lo vinculado."""
    ej = ejecutoria or "el amparo anterior"
    tramo = f" ({efecto})" if efecto else ""
    motivo = f" {por_que.rstrip('.')}." if por_que else ""
    return (f"Combate lo que la responsable resolvió vinculada por la ejecutoria dictada en el {ej}{tramo}, "
            f"no con libertad de jurisdicción; en ese aspecto rige la cosa juzgada y no puede examinarse en "
            f"este juicio (2a./J. 113/2012 y 1a./J. 57/2018).{motivo}")


def razon_exceso(ejecutoria: str, por_que: str) -> str:
    ej = ejecutoria or "el amparo anterior"
    motivo = f" {por_que.rstrip('.')}." if por_que else ""
    return (f"Alega que la responsable se excedió o se quedó corta al cumplir la ejecutoria dictada en el {ej}; "
            f"eso no es materia de este amparo, sino de la vista del artículo 196, del recurso de inconformidad "
            f"(artículo 201) o de la denuncia de repetición del acto reclamado (P./J. 98/97).{motivo}")


def _clave(t: str) -> str:
    return _plano(t)[:220]


def aplicar(criterios: list, clasificacion: dict, ejecutoria: str = "", tocados: set | None = None) -> tuple:
    """Deja INOPERANTES, en sitio, los criterios de lo vinculado y del exceso o
    defecto que el secretario no tocó. `criterios`: dicts u objetos con
    problema/sentido/razonamiento (o razon). Devuelve (avisos, detalle)."""
    avisos, detalle = [], {}
    if not isinstance(clasificacion, dict):
        return avisos, detalle
    por_pregunta = {_clave(x.get("pregunta")): x for x in clasificacion.get("problemas") or []}
    tocados = {(_clave(t)) for t in (tocados or set())}
    for c in criterios or []:
        es_dict = isinstance(c, dict)
        prob = (c.get("problema") if es_dict else getattr(c, "problema", "")) or ""
        x = por_pregunta.get(_clave(prob))
        if not x:
            continue
        v = x.get("vinculacion")
        detalle[prob] = {"de": "cumplimiento", "vinculacion": v, "efecto": x.get("efecto", ""),
                         "por_que": x.get("por_que", "")}
        tocado = (_clave(prob) in tocados) or bool(c.get("tocado") if es_dict else getattr(c, "tocado", False))
        if v == "constitucionalidad":
            avisos.append(
                f"«{prob[:90]}» plantea la inconstitucionalidad de la norma cuya aplicación ordenó la ejecutoria "
                f"del {ejecutoria or 'amparo anterior'}. CRITERIO DIVIDIDO: la Primera Sala (1a. IX/2022) permite "
                f"estudiarla por primera vez contra la sentencia de cumplimiento, respetando la legalidad fijada; la "
                f"Segunda (2a. CXLVII/2017) exigía combatirla en revisión contra la ejecutoria. Decide tú si se estudia.")
            continue
        if v not in ("vinculado", "exceso_defecto") or tocado:
            continue
        razon = (razon_vinculado(ejecutoria, x.get("efecto", ""), x.get("por_que", "")) if v == "vinculado"
                 else razon_exceso(ejecutoria, x.get("por_que", "")))
        antes = (c.get("sentido") if es_dict else getattr(c, "sentido", "")) or ""
        # SU APOYO ES LA JURISPRUDENCIA QUE LA RAZÓN CITA (verificada en el
        # Semanario): sin él, la propuesta salía «SIN APOYO del acervo» aunque
        # la razón invocara la 2a./J. 113/2012 y la 1a./J. 57/2018.
        apoyo = (APOYOS_MIXTA[:1] + APOYOS_IMPROCEDENCIA[:1]) if v == "vinculado" else APOYOS_EXCESO[:1]
        if es_dict:
            c["sentido"] = "inoperante"
            c["razonamiento"] = razon
            if "razon" in c:
                c["razon"] = razon
            if "apoyos" in c and not c.get("apoyos"):
                c["apoyos"] = list(apoyo)
        else:
            setattr(c, "sentido", "inoperante")
            if hasattr(c, "razonamiento"):
                setattr(c, "razonamiento", razon)
            if hasattr(c, "razon"):
                setattr(c, "razon", razon)
            if hasattr(c, "apoyos") and not getattr(c, "apoyos", None):
                setattr(c, "apoyos", list(apoyo))
        if antes and antes != "inoperante":
            avisos.append(f"«{prob[:90]}»: {antes} → inoperante, porque combate lo que la ejecutoria del "
                          f"{ejecutoria or 'amparo anterior'} dejó vinculado. Si no es así, márcalo tú.")
    return avisos, detalle


def sobreseer_propuesto(origen: dict | None) -> bool:
    """¿La ejecutoria no dejó libertad alguna? (con clasificación hecha)."""
    c = (origen or {}).get("clasificacion") or {}
    cu = (origen or {}).get("cumplimiento") or {}
    return bool(cu.get("consta") and c and c.get("hay_libertad") is False
                and not any(p.get("vinculacion") in ("libre", "mixto", "constitucionalidad")
                            for p in c.get("problemas") or []))


def sobreseer_confirmado(origen: dict | None = None) -> bool:
    """El sobreseimiento sólo se ESCRIBE si el secretario lo confirmó."""
    if origen is None:
        try:
            import contexto_taller as _ct
            origen = _ct.origen()
        except Exception:
            origen = None
    return bool(rige() and isinstance(origen, dict) and origen.get("sobreseer_confirmado")
                and (origen.get("cumplimiento") or {}).get("consta"))


# ═══ 3 · LO QUE LEEN LA PROPUESTA, EL PLAN Y EL ESTUDIO ═════════════════════

_ETIQUETA = {"vinculado": "VINCULADO por la ejecutoria → inoperante",
             "libre": "LIBERTAD de jurisdicción → se estudia",
             "mixto": "MIXTO → inoperante en lo vinculado, se estudia lo libre",
             "exceso_defecto": "EXCESO O DEFECTO en el cumplimiento → inoperante (no es materia de este amparo)",
             "constitucionalidad": "INCONSTITUCIONALIDAD de la norma que la ejecutoria mandó aplicar → la decide el secretario",
             "no_consta": "no consta"}


def bloque_contexto(origen: dict | None = None) -> str:
    """El bloque para los prompts (vacío si no rige o no consta)."""
    if origen is None:
        try:
            import contexto_taller as _ct
            origen = _ct.origen()
        except Exception:
            origen = None
    if not rige() or not isinstance(origen, dict):
        return ""
    cu = origen.get("cumplimiento") or {}
    if not cu.get("consta"):
        return ""
    ej = cu.get("ejecutoria") or "un amparo anterior"
    L = [f"LA SENTENCIA RECLAMADA SE DICTÓ EN CUMPLIMIENTO DE LA EJECUTORIA DEL {ej.upper()}.",
         "Sólo es materia de este amparo lo que la responsable resolvió con libertad de jurisdicción. Lo que la "
         "ejecutoria dejó vinculado es cosa juzgada: los planteamientos que lo combaten son inoperantes (2a./J. "
         "113/2012); el exceso o defecto en el cumplimiento se ventila en la vista del artículo 196, el recurso de "
         "inconformidad o la denuncia de repetición, no aquí (P./J. 98/97). Se distingue por lo que la ejecutoria "
         "ordenó —sus efectos, sus consideraciones y lineamientos (1a./J. 75/2014)—, no por el tema."]
    efectos = str(cu.get("efectos") or "").strip()
    if efectos:
        L.append("LO QUE MANDÓ ESA EJECUTORIA (tal como consta en autos):\n" + efectos[:4000])
    else:
        L.append("SUS EFECTOS NO CONSTAN EN EL MATERIAL: no supongas qué quedó vinculado; si un planteamiento parece "
                 "combatir lo que esa ejecutoria ordenó, dilo en ADVERTENCIAS y no lo califiques por eso.")
    cl = origen.get("clasificacion") or {}
    if cl.get("problemas"):
        L.append("CÓMO QUEDÓ CADA PLANTEAMIENTO FRENTE A ESA EJECUTORIA:")
        for i, p in enumerate(cl["problemas"], 1):
            v = p.get("vinculacion", "no_consta")
            linea = f"  {i}. {p.get('pregunta', '')[:220]} — {_ETIQUETA.get(v, v)}"
            if p.get("efecto"):
                linea += f" [{p['efecto']}]"
            if p.get("por_que"):
                linea += f": {p['por_que']}"
            if v == "mixto":
                linea += (f" (vinculado: {p.get('parte_vinculada') or 'no consta'}; "
                          f"libre: {p.get('parte_libre') or 'no consta'})")
            L.append(linea)
        if cl.get("resumen"):
            L.append("En síntesis: " + cl["resumen"])
    L.append("AL REDACTAR: antes de los conceptos, un párrafo que precise qué ordenó la ejecutoria, qué quedó vinculado "
             "y qué resolvió la responsable con libertad de jurisdicción; después, lo vinculado se declara inoperante "
             "con esa razón, sin estudiarlo de nuevo, y lo libre se estudia a fondo.")
    if origen.get("sobreseer_confirmado"):
        L.append("EL SECRETARIO CONFIRMÓ QUE LA EJECUTORIA NO DEJÓ LIBERTAD ALGUNA: se sobresee (arts. 61, fr. IX, y "
                 "63, fr. V, de la Ley de Amparo); no hay estudio de fondo.")
    return "\n".join(L)


# ═══ 4 · EL CONSIDERANDO Y EL RESOLUTIVO DEL SOBRESEIMIENTO ═════════════════

IMPROCEDENTE = {
    "rotulo": "Improcedencia del juicio",
    "fundamento": "artículos 61, fracción IX, y 63, fracción V, de la Ley de Amparo",
    "considerando": (
        "La sentencia reclamada se dictó en cumplimiento de la ejecutoria pronunciada en el {ejecutoria}, y en ella "
        "la autoridad responsable no tuvo libertad de jurisdicción alguna: se limitó a acatar lo que esa ejecutoria "
        "le ordenó. Lo así resuelto constituye cosa juzgada y un nuevo juicio de amparo no puede revisarlo, de modo "
        "que se actualiza la causa de improcedencia prevista en el artículo 61, fracción IX, de la Ley de Amparo, "
        "que lo excluye contra las resoluciones dictadas en los juicios de amparo o en ejecución de éstas. En "
        "consecuencia, procede sobreseer en el juicio con fundamento en el artículo 63, fracción V, del mismo "
        "ordenamiento. Si la parte quejosa estima que la responsable se excedió o incurrió en defecto al cumplir, "
        "su defensa es la prevista en los artículos 196 y 201 de la Ley de Amparo, o la denuncia de repetición del "
        "acto reclamado, no un nuevo amparo."),
    "resolutivo": "ÚNICO. Se sobresee en el presente juicio de amparo promovido por {quejoso}.",
}


def considerando_improcedencia(origen: dict | None = None) -> str:
    if origen is None:
        try:
            import contexto_taller as _ct
            origen = _ct.origen()
        except Exception:
            origen = None
    ej = ((origen or {}).get("cumplimiento") or {}).get("ejecutoria") or "juicio de amparo anterior"
    return IMPROCEDENTE["considerando"].replace("{ejecutoria}", ej)
