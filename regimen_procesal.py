"""QUÉ CÓDIGO PROCESAL RIGE — entidad, materia y autoridad (11-oct-2026).

David: «me imagino que así el motor sería mucho más preciso y no mezclaría
procesos judiciales estatales con procesos judiciales federales».

El selector de fuentes no basta: el abogado tendría que saber de antemano qué
código rige. La mezcla sólo existe en tres materias, y aquí se decide cada una:

  · CIVIL Y FAMILIAR — el CNPCF entra por Declaratoria del Congreso local,
    escalonada por vía (la CDMX lo aplica a hipotecario y arrendamiento orales
    y a lo familiar sin divorcio desde dic-2024, todo lo demás el 1-abr-2027);
    sin declaratoria rige el código de procedimientos local; en lo federal,
    el CFPC. A más tardar el 1-abr-2027 rige en todo el país.
  · ADMINISTRATIVO — lo decide la AUTORIDAD que emitió el acto: federal (SAT,
    IMSS…) → LFPCA ante el TFJA; estatal o municipal → la ley de justicia
    administrativa de la entidad.
  · LABORAL BUROCRÁTICO — la LFTSE sólo para trabajadores de la Federación;
    los del estado o el municipio tienen su ley local.

Penal (CNPP), mercantil (Código de Comercio), amparo y agrario son únicos en
todo el país: se dice y no se excluye nada.

`resolver` devuelve qué excluir de la fuente procesal federal, qué código
local inyectar (clave de `_CODIGOS_DE_LA_MATERIA` en main.py), el texto corto
para la pantalla y la nota para el modelo. Los datos de vigencia viven en
datos/regimen_procesal.json y dicen de dónde salió cada uno.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
from typing import Optional

import procesal_federal as pf

_RUTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos", "regimen_procesal.json")
try:
    with open(_RUTA, encoding="utf-8") as _f:
        DATOS = json.load(_f)
except Exception:  # sin el fichero, el resolvedor no excluye nada
    DATOS = {"cnpcf_limite": "2027-04-01", "cnpcf_entidades": {}, "cnpcf_orden_federal": {}}

_NOMBRES = {
    "CIUDAD_DE_MEXICO": "Ciudad de México", "ESTADO_DE_MEXICO": "Estado de México", "MEXICO": "Estado de México",
    "QUERETARO": "Querétaro", "MICHOACAN": "Michoacán", "NUEVO_LEON": "Nuevo León", "YUCATAN": "Yucatán",
    "SAN_LUIS_POTOSI": "San Luis Potosí",
}
_ALIAS = {"CDMX": "CIUDAD_DE_MEXICO", "DISTRITO_FEDERAL": "CIUDAD_DE_MEXICO", "EDOMEX": "ESTADO_DE_MEXICO",
          "MEXICO": "ESTADO_DE_MEXICO"}

_RX_FAMILIAR = re.compile(r"\b(familiar|divorci\w*|alimentos|pension alimenticia|guarda|custodia|convivencia|"
                          r"patria potestad|adopci\w*|filiaci\w*|paternidad|sucesi\w*|testament\w*|intestad\w*|"
                          r"herencia|matrimonio|concubin\w*|violencia familiar|menor(es)?)\b")
_RX_CIVIL_FEDERAL = re.compile(r"\b(juzgado de distrito|juez de distrito|juicio (civil )?federal|orden federal|"
                               r"la federacion (es|sea) parte|cfpc|codigo federal de procedimientos civiles)\b")
_RX_PIDE_CNPCF = re.compile(r"\b(cnpcf|codigo nacional de procedimientos civiles)\b")
_RX_ADMIN_FEDERAL = re.compile(
    r"\b(sat|servicio de administracion tributaria|imss|infonavit|issste|conagua|cofepris|profeco|profepa|cfe|"
    r"comision federal|aduana\w*|tfja|tribunal federal de justicia administrativa|autoridad federal|condusef|"
    r"semarnat|sict|sct|secretaria de (hacienda|economia|salud|energia|educacion publica|infraestructura|"
    r"medio ambiente|bienestar|gobernacion|seguridad|marina|defensa|relaciones exteriores|trabajo)|"
    r"credito fiscal federal|impuesto sobre la renta|isr|iva|cuotas obrero)\b")
_RX_ADMIN_LOCAL = re.compile(
    r"\b(municip\w*|ayuntamiento|alcaldia|predial|transito|tesoreria|secretaria de finanzas del estado|"
    r"gobierno del estado|estatal|licencia de funcionamiento|tribunal de justicia administrativa del estado|"
    r"tenencia|control vehicular|catastr\w*|uso de suelo|desarrollo urbano|fotomulta|multa de transito|"
    r"policia municipal|contraloria (del estado|municipal))\b")
_RX_BUROCRATA_LOCAL = re.compile(
    r"\b(trabajador\w* (al servicio )?del (estado|municipio|ayuntamiento|gobierno del estado)|burocrat\w* estatal|"
    r"ayuntamiento|municipio|gobierno del estado|servidor(es)? public\w* (estatal|municipal)\w*|"
    r"tribunal (estatal )?de conciliacion y arbitraje del estado)\b")
_RX_BUROCRATA_FEDERAL = re.compile(
    r"\b(lftse|tribunal federal de conciliacion|dependencia federal|gobierno federal|trabajador\w* de la federacion|"
    r"issste|apartado b)\b")


def clave_estado(estado: Optional[str]) -> Optional[str]:
    if not estado:
        return None
    k = re.sub(r"[^A-Z_]", "", pf.normalizar(estado).upper().replace(" ", "_"))
    k = _ALIAS.get(k, k)
    return None if k in ("", "ALL", "TODOS", "TODOS_LOS_ESTADOS", "NACIONAL", "FEDERAL") else k


def nombre_estado(clave: str) -> str:
    return _NOMBRES.get(clave) or clave.replace("_", " ").title().replace(" De ", " de ")


def _hoy(hoy):
    return hoy or _dt.date.today()


def _fecha(s: str) -> _dt.date:
    return _dt.date.fromisoformat(s)


def _civil_familiar(est: Optional[str], q: str, hoy: _dt.date) -> dict:
    limite = _fecha(DATOS.get("cnpcf_limite", "2027-04-01"))
    lim_txt = "1 de abril de 2027"
    if _RX_CIVIL_FEDERAL.search(q):
        if hoy >= limite:
            return {"materia": "civil_familiar", "rige": "CNPCF", "excluir": [], "inyectar": None,
                    "chip": "Civil federal: CNPCF (rige en todo el país desde el 1-abr-2027)",
                    "nota": "Juicio civil del orden federal: rige el CNPCF; los iniciados antes siguen con el CFPC."}
        return {"materia": "civil_familiar", "rige": "CFPC", "excluir": ["CNPCF"], "inyectar": None,
                "chip": f"Civil federal: Código Federal de Procedimientos Civiles (el CNPCF rige en lo federal a más tardar el {lim_txt})",
                "nota": (DATOS.get("cnpcf_orden_federal") or {}).get("texto", "")}
    if hoy >= limite:
        return {"materia": "civil_familiar", "rige": "CNPCF", "excluir": ["CFPC"], "inyectar": None,
                "chip": "Civil y familiar: CNPCF (vigente en todo el país desde el 1-abr-2027)",
                "nota": "El CNPCF rige en todo el país desde el 1 de abril de 2027; los juicios iniciados antes "
                        "se siguen tramitando conforme al código procesal anterior."}
    if not est:
        return {"materia": "civil_familiar", "rige": "MIXTO", "excluir": ["CFPC"], "inyectar": None,
                "chip": f"Civil y familiar: depende de la entidad (CNPCF por declaratoria; en todas a más tardar el {lim_txt})",
                "nota": "No se sabe la entidad: el CNPCF rige donde el Congreso local ya emitió su declaratoria y en "
                        f"todas a más tardar el {lim_txt}; antes, el código de procedimientos local. Pregunta la "
                        "entidad si el plazo o la vía dependen de ello."}
    nombre = nombre_estado(est)
    reg = (DATOS.get("cnpcf_entidades") or {}).get(est)
    familiar = bool(_RX_FAMILIAR.search(q))
    if reg and reg.get("estado") == "verificada" and reg.get("fases"):
        aplica = None
        for fase in reg["fases"]:
            if _fecha(fase["desde"]) > hoy:
                continue
            if fase["materia"] not in ("todo", "familiar" if familiar else "civil"):
                continue
            if fase.get("rx") and not re.search(fase["rx"], q):
                continue
            if fase.get("rx_excluye") and re.search(fase["rx_excluye"], q):
                continue
            aplica = fase
        if aplica:
            return {"materia": "civil_familiar", "rige": "CNPCF", "excluir": ["CFPC"], "inyectar": None,
                    "chip": f"{'Familiar' if familiar else 'Civil'} en {nombre}: CNPCF (declaratoria; desde {aplica['desde']})",
                    "nota": f"En {nombre} el CNPCF rige para: {aplica['supuestos']} ({reg['fuente']}). Funda en el CNPCF; "
                            "el código de procedimientos local sólo para asuntos iniciados antes."}
        siguiente = min((f["desde"] for f in reg["fases"] if _fecha(f["desde"]) > hoy), default=None)
        return {"materia": "civil_familiar", "rige": "LOCAL", "excluir": ["CNPCF", "CFPC"], "inyectar": "procesal_civil_local",
                "chip": f"{'Familiar' if familiar else 'Civil'} en {nombre}: código de procedimientos local"
                        + (f" (el CNPCF entra para este supuesto el {siguiente})" if siguiente else ""),
                "nota": f"En {nombre} el CNPCF sólo rige para: "
                        + "; ".join(f["supuestos"] for f in reg["fases"] if _fecha(f["desde"]) <= hoy)
                        + f" ({reg['fuente']}). Este asunto no está en esos supuestos: rige el código de procedimientos "
                        "local de la entidad. No apliques plazos ni vías del CNPCF."}
    if reg and reg.get("estado") == "reportada" and _fecha(reg.get("desde", "2999-01-01")) <= hoy:
        return {"materia": "civil_familiar", "rige": "MIXTO", "excluir": ["CFPC"], "inyectar": "procesal_civil_local",
                "chip": f"Civil y familiar en {nombre}: CNPCF en aplicación gradual — confirma si tu asunto ya entra",
                "nota": f"{nombre}: {reg['fuente']}. La aplicación es gradual y no tenemos el detalle por vía: presenta "
                        "el CNPCF y el código local, di cuál rige según la fecha y la vía, y pide confirmar la fase."}
    if _RX_PIDE_CNPCF.search(q):
        return {"materia": "civil_familiar", "rige": "LOCAL", "excluir": ["CFPC"], "inyectar": "procesal_civil_local",
                "chip": f"Civil y familiar en {nombre}: rige el código local; el CNPCF aún no tiene declaratoria publicada",
                "nota": f"{nombre} no tiene declaratoria del CNPCF publicada en el DOF: hoy rige su código de procedimientos "
                        f"local y el CNPCF entrará a más tardar el {lim_txt}. Si el abogado pregunta por el CNPCF, contéstale "
                        "y advierte que todavía no rige ahí."}
    return {"materia": "civil_familiar", "rige": "LOCAL", "excluir": ["CNPCF", "CFPC"], "inyectar": "procesal_civil_local",
            "chip": f"Civil y familiar en {nombre}: código de procedimientos local (CNPCF a más tardar el {lim_txt})",
            "nota": f"{nombre} no tiene declaratoria del CNPCF publicada en el DOF (revisado al 11-oct-2026): rige su "
                    f"código de procedimientos local. El CNPCF entrará a más tardar el {lim_txt}. No apliques plazos ni "
                    "vías del CNPCF a este asunto."}


def _administrativo(est: Optional[str], q: str) -> dict:
    fed, loc = bool(_RX_ADMIN_FEDERAL.search(q)), bool(_RX_ADMIN_LOCAL.search(q))
    if fed and not loc:
        return {"materia": "administrativo", "rige": "FEDERAL", "excluir": [], "inyectar": None,
                "chip": "Administrativo federal: juicio de nulidad ante el TFJA (LFPCA); recursos de la LFPA y del CFF",
                "nota": "El acto es de autoridad federal: rige la LFPCA (juicio de nulidad ante el TFJA) y, en sede "
                        "administrativa, la LFPA o el CFF. No mezcles la ley de justicia administrativa local."}
    nombre = nombre_estado(est) if est else "la entidad"
    if loc and not fed:
        return {"materia": "administrativo", "rige": "LOCAL", "excluir": ["LFPCA", "LOTFJA", "LFPA", "CFF"],
                "inyectar": "admin_local" if est else None,
                "chip": f"Administrativo local: tribunal de justicia administrativa de {nombre} y su ley procesal",
                "nota": f"El acto es de autoridad estatal o municipal: rige la ley de justicia (o procedimiento) "
                        f"administrativa de {nombre}, no la LFPCA ni el TFJA."}
    return {"materia": "administrativo", "rige": "MIXTO", "excluir": [], "inyectar": "admin_local" if est else None,
            "chip": "Administrativo: depende de la autoridad (federal → TFJA y LFPCA; estatal o municipal → tribunal local)",
            "nota": "No se sabe si el acto es de autoridad federal o local: explica las dos vías (TFJA con la LFPCA, o el "
                    "tribunal de justicia administrativa de la entidad con su ley) y pregunta qué autoridad lo emitió."}


def _laboral(est: Optional[str], q: str) -> dict:
    if _RX_BUROCRATA_LOCAL.search(q) and not _RX_BUROCRATA_FEDERAL.search(q):
        nombre = nombre_estado(est) if est else "la entidad"
        return {"materia": "laboral", "rige": "LOCAL_BUROCRATICO", "excluir": ["LFTSE"],
                "inyectar": "burocratico_local" if est else None,
                "chip": f"Laboral burocrático local: ley de los trabajadores al servicio de {nombre}",
                "nota": f"Trabajador del estado o del municipio: rige la ley burocrática de {nombre} ante su tribunal "
                        "de arbitraje; la LFTSE es sólo para la Federación y la LFT, en todo caso, supletoria."}
    if _RX_BUROCRATA_FEDERAL.search(q):
        return {"materia": "laboral", "rige": "LFTSE", "excluir": [], "inyectar": None,
                "chip": "Laboral burocrático federal: LFTSE ante el Tribunal Federal de Conciliación y Arbitraje",
                "nota": "Trabajador de la Federación: rige la LFTSE (apartado B); la LFT es supletoria."}
    return {"materia": "laboral", "rige": "LFT", "excluir": ["LFTSE"], "inyectar": None,
            "chip": "Laboral: Ley Federal del Trabajo (procedimiento igual ante tribunales federales y locales)",
            "nota": "Rige el procedimiento de la LFT; la competencia federal o local la fija su art. 527."}


_UNICOS = {
    "penal": ("CNPP", "Penal: Código Nacional de Procedimientos Penales (rige en todo el país)"),
    "mercantil": ("CCom", "Mercantil: Código de Comercio (federal, también ante jueces locales)"),
    "amparo": ("LAmp", "Amparo: Ley de Amparo"),
    "agrario": ("LAgra", "Agrario: Ley Agraria y Ley Orgánica de los Tribunales Agrarios"),
}


def resolver(estado: Optional[str], det: dict, hoy: Optional[_dt.date] = None) -> list:
    """[régimen por materia] — vacío si la consulta no es procesal."""
    if not det or not det.get("procesal"):
        return []
    q, est, h = det.get("q", ""), clave_estado(estado), _hoy(hoy)
    fuera = []
    for m in det.get("materias") or []:
        if m == "civil_familiar":
            fuera.append(_civil_familiar(est, q, h))
        elif m == "administrativo":
            fuera.append(_administrativo(est, q))
        elif m == "laboral":
            fuera.append(_laboral(est, q))
        elif m in _UNICOS:
            ab, chip = _UNICOS[m]
            fuera.append({"materia": m, "rige": ab, "excluir": [], "inyectar": None, "chip": chip, "nota": ""})
    return fuera


def excluidas(regimenes: list) -> set:
    """Lo que se excluye: sólo si NINGÚN régimen de la consulta lo necesita."""
    quitar = set()
    for r in regimenes:
        quitar |= set(r.get("excluir") or [])
    for r in regimenes:
        quitar.discard(r.get("rige"))
    return quitar


def nota_para_modelo(regimenes: list) -> str:
    partes = [f"{r['chip']}. {r['nota']}".strip() for r in regimenes if r.get("chip")]
    if not partes:
        return ""
    return ("RÉGIMEN PROCESAL APLICABLE (decidido por entidad, materia y autoridad; no lo cambies): "
            + " | ".join(partes)
            + " Funda vía, plazo y recurso SÓLO en el código que rige; si mencionas el otro, di por qué no aplica.")
