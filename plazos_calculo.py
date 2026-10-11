"""CALCULADORA DE PLAZOS PROCESALES — determinista y con su fundamento (11-oct-2026).

El modelo no cuenta días: los cuenta esto. Cada regla sale del texto vigente
de su código (leído en `procesal_federal`, oct-2026) y se cita al lado:

  · cuándo SURTE EFECTOS la notificación (según el tipo de notificación);
  · desde cuándo EMPIEZA a correr el plazo;
  · qué días NO se cuentan (fines de semana, festivos del propio código) y si
    el plazo es de días hábiles o naturales.

Lo que la ley deja al calendario de cada tribunal (vacaciones, suspensiones de
labores) no se adivina: el abogado lo agrega en `inhabiles_extra` o
`periodos_inhabiles`, y la respuesta lo advierte siempre.
"""
from __future__ import annotations

import datetime as dt
from typing import Iterable, Optional

D = dt.date


def _n_lunes(anio: int, mes: int, n: int) -> D:
    d = D(anio, mes, 1)
    d += dt.timedelta(days=(0 - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


def _fijos(anio, lista):
    return {D(anio, m, d): motivo for (m, d, motivo) in lista}


def _lft74(anio):
    f = {D(anio, 1, 1): "1 de enero", _n_lunes(anio, 2, 1): "primer lunes de febrero (5 de febrero)",
         _n_lunes(anio, 3, 3): "tercer lunes de marzo (21 de marzo)", D(anio, 5, 1): "1 de mayo",
         D(anio, 9, 16): "16 de septiembre", _n_lunes(anio, 11, 3): "tercer lunes de noviembre (20 de noviembre)",
         D(anio, 12, 25): "25 de diciembre"}
    if anio % 6 == 0:
        f[D(anio, 10, 1)] = "1 de octubre (transmisión del Ejecutivo)"
    return f


_AMP19 = [(1, 1, "1 de enero"), (2, 5, "5 de febrero"), (3, 21, "21 de marzo"), (5, 1, "1 de mayo"),
          (5, 5, "5 de mayo"), (9, 14, "14 de septiembre"), (9, 16, "16 de septiembre"), (10, 12, "12 de octubre"),
          (11, 20, "20 de noviembre"), (12, 25, "25 de diciembre")]
_LOPJF229 = [(1, 1, "1 de enero"), (2, 5, "5 de febrero"), (3, 21, "21 de marzo"), (5, 1, "1 de mayo"),
             (9, 14, "14 de septiembre"), (9, 16, "16 de septiembre"), (11, 20, "20 de noviembre")]
_LFPA28 = [(1, 1, "1 de enero"), (2, 5, "5 de febrero"), (3, 21, "21 de marzo"), (5, 1, "1 de mayo"),
           (5, 5, "5 de mayo"), (9, 1, "1 de septiembre"), (9, 16, "16 de septiembre"), (11, 20, "20 de noviembre"),
           (12, 25, "25 de diciembre")]


def _cff12(anio):
    return {D(anio, 1, 1): "1 de enero", _n_lunes(anio, 2, 1): "primer lunes de febrero",
            _n_lunes(anio, 3, 3): "tercer lunes de marzo", D(anio, 5, 1): "1 de mayo", D(anio, 5, 5): "5 de mayo",
            D(anio, 9, 16): "16 de septiembre", _n_lunes(anio, 11, 3): "tercer lunes de noviembre",
            D(anio, 12, 25): "25 de diciembre"}


# surte: días (hábiles) después de la notificación en que surte efectos, por tipo.
# inicio: "siguiente" = el plazo corre desde el día siguiente al que surte;
#         "mismo" = desde el día en que surte (CNPP art. 94).
REGLAS = {
    "LAmp": {"nombre": "Ley de Amparo", "sabado": True, "festivos": lambda a: _fijos(a, _AMP19),
             "surte": {"personal": 1, "lista": 1, "electronica": 1}, "inicio": "siguiente", "tipo": "habiles",
             "fund": ["art. 19 (días hábiles)", "art. 22 (cómputo: hábiles, desde el día siguiente al que surte, incluye el vencimiento)",
                      "art. 31, fr. II (surte efectos el día siguiente al de la notificación o de la lista)"],
             "avisos": ["Para la DEMANDA, la notificación del acto surte efectos conforme a la ley del acto (art. 18): verifica esa regla.",
                        "En materia penal los plazos del amparo se computan de momento a momento (art. 22)."]},
    "CNPCF": {"nombre": "Código Nacional de Procedimientos Civiles y Familiares", "sabado": True, "festivos": _lft74,
              "surte": {"personal": 0, "lista": 1, "electronica": 0}, "inicio": "siguiente", "tipo": "habiles",
              "fund": ["art. 149 (días hábiles: todos menos sábados, domingos y festivos legales)",
                       "art. 227 (emplazamiento o notificación personal: corre el día siguiente; por el medio oficial: el día siguiente a que surte; por medio judicial electrónico: surte el día de la confirmación)",
                       "art. 211 (la publicación surte efectos al día siguiente)"],
              "avisos": ["En alimentos, derechos de niñas, niños y adolescentes, controversias familiares y violencia familiar, todos los días y horas son hábiles (art. 150): marca «familiar urgente»."]},
    "CNPP": {"nombre": "Código Nacional de Procedimientos Penales", "sabado": True, "festivos": _lft74,
             "surte": {"personal": 1, "lista": 1, "electronica": 1}, "inicio": "mismo", "tipo": "habiles",
             "fund": ["art. 82 (las personales surten efectos al día siguiente; las de lista o estrado, al día siguiente de su publicación)",
                      "art. 94 (no se cuentan sábados, domingos ni inhábiles; los plazos en días corren a partir del día en que surte efectos; los de horas, de momento a momento)"],
             "avisos": ["Los plazos en horas (detención, vinculación) corren de momento a momento y no excluyen inhábiles."]},
    "LFT": {"nombre": "Ley Federal del Trabajo", "sabado": True, "festivos": _lft74,
            "surte": {"personal": 0, "lista": 1, "electronica": 2}, "inicio": "siguiente", "tipo": "habiles",
            "fund": ["art. 747 (personales: surten en el acto; boletín o estrados: al día siguiente; buzón electrónico: en dos días)",
                     "art. 733 (corren desde el día siguiente al que surte; se cuenta el del vencimiento)",
                     "art. 734 (no se cuentan los días en que el Tribunal no actúe según su calendario)", "art. 74 (descanso obligatorio)"],
            "avisos": ["Agrega los días que tu tribunal laboral suspendió labores (art. 734)."]},
    "LFPCA": {"nombre": "Ley Federal de Procedimiento Contencioso Administrativo", "sabado": True, "festivos": _lft74,
              "surte": {"personal": 1, "lista": 1, "electronica": 1}, "inicio": "siguiente", "tipo": "habiles",
              "fund": ["art. 70 (surten efectos el día hábil siguiente)", "art. 74 (corren desde el día siguiente al que surte; sólo días hábiles, con las Salas abiertas)"],
              "avisos": ["Agrega los periodos vacacionales y suspensiones que publique el TFJA."]},
    "CFF": {"nombre": "Código Fiscal de la Federación", "sabado": True, "festivos": _cff12,
            "surte": {"personal": 1, "lista": 1, "electronica": 1}, "inicio": "siguiente", "tipo": "habiles",
            "fund": ["art. 135 (surten efectos al día hábil siguiente)", "art. 12 (días que no se cuentan; vacaciones generales de las autoridades fiscales)"],
            "avisos": ["Agrega las vacaciones generales del SAT o de la autoridad fiscal (art. 12)."]},
    "LFPA": {"nombre": "Ley Federal de Procedimiento Administrativo", "sabado": True, "festivos": lambda a: _fijos(a, _LFPA28),
             "surte": {"personal": 0, "lista": 0, "electronica": 0}, "inicio": "siguiente", "tipo": "habiles",
             "fund": ["art. 38 (las personales surten el día en que se hacen; el plazo corre desde el día siguiente)", "art. 28 (días inhábiles)",
                      "art. 29 (plazos por periodos, meses o años)"],
             "avisos": ["Agrega las vacaciones de la dependencia (art. 28)."]},
    "CCom": {"nombre": "Código de Comercio", "sabado": True, "festivos": _lft74,
             "surte": {"personal": 1, "lista": 1, "electronica": 1}, "inicio": "siguiente", "tipo": "habiles",
             "fund": ["art. 1075 (surten efectos al día siguiente; corren desde el día siguiente al que surten; se cuenta el del vencimiento)",
                      "art. 1064 (hábiles todos menos domingos y los días en que no laboren los tribunales)"],
             "avisos": ["Los sábados se excluyen porque los tribunales no laboran; si el tuyo sí, quita esa exclusión. Agrega el calendario del tribunal."]},
    "CFPC": {"nombre": "Código Federal de Procedimientos Civiles", "sabado": True,
             "festivos": lambda a: {**_fijos(a, _LOPJF229), D(a, 12, 25): "25 de diciembre"},
             "surte": {"personal": 1, "lista": 1, "electronica": 1}, "inicio": "siguiente", "tipo": "habiles",
             "fund": ["art. 321 (surte efectos el día siguiente)", "art. 284 (corren desde el día siguiente al que surte; se cuenta el del vencimiento)",
                      "art. 281 y LOPJF art. 229 (días inhábiles en el Poder Judicial de la Federación)"],
             "avisos": ["Agrega los periodos vacacionales del Poder Judicial de la Federación."]},
    "LOCAL": {"nombre": "Código procesal local", "sabado": True, "festivos": _lft74,
              "surte": {"personal": 1, "lista": 1, "electronica": 1}, "inicio": "siguiente", "tipo": "habiles",
              "fund": [], "avisos": ["Regla genérica: verifica en tu código local cuándo surte efectos la notificación y el calendario del tribunal."]},
}
REGLAS["LAgra"] = {**REGLAS["CFPC"], "nombre": "Ley Agraria (supletorio: CFPC)",
                   "avisos": ["La Ley Agraria remite al Código Federal de Procedimientos Civiles como supletorio; agrega el calendario del tribunal agrario."]}
REGLAS["LOTA"] = REGLAS["LAgra"]
REGLAS["LFTSE"] = {**REGLAS["LFT"], "nombre": "LFTSE (supletoria la LFT)"}
REGLAS["CFF_RECURSO"] = REGLAS["CFF"]


def _inhabiles(regla: dict, desde: D, hasta: D, extra: set, periodos: list) -> dict:
    fuera = {}
    for anio in range(desde.year, hasta.year + 1):
        fuera.update(regla["festivos"](anio))
    for d in extra:
        fuera.setdefault(d, "inhábil agregado por el abogado")
    for a, b in periodos:
        d = a
        while d <= b:
            fuera.setdefault(d, "periodo inhábil agregado (vacaciones o suspensión)")
            d += dt.timedelta(days=1)
    return fuera


def _es_habil(d: D, regla: dict, inh: dict) -> Optional[str]:
    """None si es hábil; el motivo si no lo es."""
    if d.weekday() == 6:
        return "domingo"
    if d.weekday() == 5 and regla.get("sabado", True):
        return "sábado"
    return inh.get(d)


def _sumar_meses(d: D, meses: int) -> D:
    m = d.month - 1 + meses
    anio, mes = d.year + m // 12, m % 12 + 1
    import calendar
    return D(anio, mes, min(d.day, calendar.monthrange(anio, mes)[1]))


def calcular(fecha_notificacion: str, cantidad: int, unidad: str = "dias", regla: str = "LOCAL",
             tipo_notificacion: str = "personal", tipo_dias: Optional[str] = None,
             inhabiles_extra: Iterable[str] = (), periodos_inhabiles: Iterable[Iterable[str]] = (),
             familiar_urgente: bool = False, sabados_habiles: bool = False) -> dict:
    R = dict(REGLAS.get(regla) or REGLAS["LOCAL"])
    if sabados_habiles:
        R["sabado"] = False
    noti = D.fromisoformat(fecha_notificacion)
    cantidad = int(cantidad)
    if cantidad <= 0 or cantidad > 400:
        raise ValueError("cantidad fuera de rango")
    extra = {D.fromisoformat(x) for x in inhabiles_extra if x}
    periodos = [(D.fromisoformat(a), D.fromisoformat(b)) for a, b in (tuple(p) for p in periodos_inhabiles) if a and b]
    tope = noti + dt.timedelta(days=cantidad * 3 + 120 if unidad == "dias" else cantidad * 400 + 60)
    inh = _inhabiles(R, noti - dt.timedelta(days=5), tope, extra, periodos)
    avisos = list(R.get("avisos") or [])
    tipo = tipo_dias or R["tipo"]
    if regla == "CNPCF" and familiar_urgente:
        tipo = "naturales"
        avisos.insert(0, "Familiar urgente: todos los días son hábiles (CNPCF art. 150); se cuentan días naturales.")
    if unidad == "horas":
        return {"regla": R["nombre"], "vencimiento": None, "fundamento": R["fund"],
                "explicacion": f"Plazo de {cantidad} horas: corre de momento a momento desde la hora en que se practicó "
                               "o surtió efectos la notificación; no se excluyen inhábiles.",
                "avisos": avisos, "pasos": []}

    pasos, inhabiles_saltados = [], []
    # 1) cuándo surte efectos
    desplaz = R["surte"].get(tipo_notificacion, 1)
    surte = noti
    k = 0
    while k < desplaz:
        surte += dt.timedelta(days=1)
        if tipo == "naturales" or not _es_habil(surte, R, inh):
            k += 1
    pasos.append({"fecha": surte.isoformat(), "que": "surte efectos la notificación" + (" (el mismo día)" if desplaz == 0 else "")})
    # 2) desde cuándo corre
    inicio = surte if R["inicio"] == "mismo" else surte + dt.timedelta(days=1)
    if unidad in ("meses", "mes", "anos", "años"):
        meses = cantidad * (12 if unidad.startswith("a") else 1)
        venc = _sumar_meses(surte if R["inicio"] == "mismo" else surte, meses)
        while _es_habil(venc, R, inh):
            inhabiles_saltados.append({"fecha": venc.isoformat(), "motivo": _es_habil(venc, R, inh)})
            venc += dt.timedelta(days=1)
        expl = f"Plazo de {cantidad} {unidad}: concluye el mismo número de día del mes que corresponda; si es inhábil, el siguiente hábil."
        return {"regla": R["nombre"], "notificacion": noti.isoformat(), "surte_efectos": surte.isoformat(),
                "inicio": inicio.isoformat(), "vencimiento": venc.isoformat(), "tipo_dias": "calendario",
                "inhabiles_saltados": inhabiles_saltados, "fundamento": R["fund"], "explicacion": expl,
                "avisos": avisos, "pasos": pasos}
    # 3) contar
    d, contados = inicio, 0
    dias = []
    while True:
        motivo = None if tipo == "naturales" else _es_habil(d, R, inh)
        if motivo:
            inhabiles_saltados.append({"fecha": d.isoformat(), "motivo": motivo})
        else:
            contados += 1
            dias.append(d.isoformat())
            if contados == cantidad:
                break
        d += dt.timedelta(days=1)
    venc = d
    if tipo == "naturales" and regla != "CNPCF":
        while _es_habil(venc, R, inh):  # si el último natural cae en inhábil, se corre al hábil siguiente
            inhabiles_saltados.append({"fecha": venc.isoformat(), "motivo": _es_habil(venc, R, inh) + " (vencimiento recorrido)"})
            venc += dt.timedelta(days=1)
    return {"regla": R["nombre"], "notificacion": noti.isoformat(), "surte_efectos": surte.isoformat(),
            "inicio": dias[0] if dias else inicio.isoformat(), "vencimiento": venc.isoformat(), "tipo_dias": tipo,
            "dias_contados": dias, "inhabiles_saltados": inhabiles_saltados, "fundamento": R["fund"],
            "explicacion": (f"{cantidad} días {'hábiles' if tipo == 'habiles' else 'naturales'}: del "
                            f"{(dias[0] if dias else inicio.isoformat())} al {venc.isoformat()}, incluido el día del vencimiento."),
            "avisos": avisos + ["No incluye vacaciones ni suspensiones de labores del órgano que no hayas agregado."],
            "pasos": pasos}
