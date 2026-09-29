# -*- coding: utf-8 -*-
"""LAS SOLUCIONES POSIBLES DE UN ASUNTO, POR CÓDIGO (rediseño del taller,
etapa 3; decisión 6 de David, 29-sep-2026: «soluciones por desenlace, tope 4,
sin tasa base como desempate»).

POR QUÉ. Hoy la deliberación es un binario A/B (`deliberacion._VIA_SENTIDOS`):
prospera o no prospera. Pero en un amparo directo «conceder» no es una sola
solución —no es lo mismo reponer el procedimiento que conceder para efectos o
conceder liso y llano—, y en la revisión las ramas del art. 93 son muchas.
Cada solución distinta necesita su propia justificación (etapa 3, punto 3) y
su propia falla decisiva (punto 5). Esta enumeración NO decide nada: dice qué
soluciones caben, con la consecuencia y los puntos resolutivos que el código
ya sabe calcular, y por qué cabe cada una.

QUÉ NO HACE:
  · no usa la tasa base ni precedentes para elegir o para cortar;
  · no enumera lo que es hecho o cómputo —sin materia, desecha,
    extemporaneidad—: eso lo declara el secretario o lo fija el cómputo;
  · con un tipo de asunto no reconocido (p. ej. la reclamación) devuelve []
    y un aviso, en vez de heredar el binario del amparo directo.

Las soluciones van en ORDEN DEL CÓDIGO (la que no prospera primero) y con ids
S1..Sn; qué papel tienen en pantalla (la propuesta y la contraria) lo decide
quien las proyecta, no este orden. Puro: no llama al modelo.
"""
from __future__ import annotations

VERSION = "soluciones-1"
TOPE = 4

# Las marcas de mayor beneficio que SÍ piden una concesión lisa y llana. La
# lista de `modos_decision._MAYOR_BENEFICIO` trae además «sobreseimiento» e
# «improcedencia del juicio», que en un amparo directo civil suelen ser la
# acción improcedente de origen (fondo), no un beneficio mayor: aquí no cuentan.
_LISA_Y_LLANA = ("lisa y llana", "liso y llano", "nulidad lisa", "prescripción", "prescripcion",
                 "caducidad", "cosa juzgada", "mayor beneficio")

_NOTA_EFECTO = {
    "niega": "No se concede el amparo: los conceptos no prosperan.",
    "para_efectos": "Se concede para que la responsable deje insubsistente el acto y dicte otro "
                    "siguiendo lo que resuelva la ejecutoria (art. 77 de la Ley de Amparo).",
    "reposicion": "Se concede para que se reponga el procedimiento desde la violación procesal "
                  "(arts. 170, fracción I, y 174 de la Ley de Amparo); el fondo queda sin estudiar "
                  "salvo que su estudio dé mayor beneficio (art. 189).",
    "liso_y_llano": "Se concede de manera lisa y llana: el vicio impide a la responsable reiterar el "
                    "acto (mayor beneficio, art. 189 de la Ley de Amparo).",
}


def _texto(p) -> str:
    if isinstance(p, dict):
        return " ".join(str(p.get(k) or "") for k in ("pregunta", "combate", "resolvio"))
    return str(p or "")


def _clases(problemas: list) -> list:
    import violacion_procesal as _vp
    out = []
    for p in problemas or []:
        try:
            out.append(_vp.clase_de(p))
        except Exception:
            out.append("fondo")
    return out


def _pide_liso(p) -> bool:
    t = _texto(p).lower()
    return any(x in t for x in _LISA_Y_LLANA)


def _sol(sentido: str, tipo_efecto: str, origen: str, sostienen: list, cons: dict) -> dict:
    import tipos_asunto as _ta
    return {"sentido_rep": sentido, "prospera": bool(_ta.prospera(sentido)),
            "tipo_efecto": tipo_efecto, "origen": origen, "problemas": sostienen,
            "rama": cons.get("rama", ""), "desenlace": list(cons.get("desenlace") or []),
            "desenlace_nota": cons.get("desenlace_nota"),
            "conceptos_omitidos": cons.get("conceptos_omitidos"),
            "nota_efecto": _NOTA_EFECTO.get(tipo_efecto, "")}


def posibles(tipo_asunto: str, problemas: list, *, resolvio_a_quo: str = "", quien_recurre: str = "",
             sobresee_ademas: bool = False, resolutivo_recurrida: str = "", quejoso: str = "",
             responsable: str = "", tenemos_conceptos=None, tope: int = TOPE) -> dict:
    """{"version", "tipo", "soluciones": [...], "avisos": [...]}.

    Cada solución: {id, sentido_rep, prospera, tipo_efecto, origen, problemas
    (índices que la sostienen), rama, desenlace, desenlace_nota,
    conceptos_omitidos, nota_efecto}. Nunca lanza."""
    avisos: list = []
    try:
        import tipos_asunto as _ta
        import deliberacion as _dl
        t = _ta.normalizar(tipo_asunto or "")
        if not t:
            return {"version": VERSION, "tipo": "", "soluciones": [],
                    "avisos": [f"Tipo de asunto no reconocido ({tipo_asunto or 'sin tipo'}): no se "
                               f"enumeran soluciones; la deliberación sigue con sus dos vías."]}
        probs = [p for p in (problemas or []) if p]
        clases = _clases(probs)
        idx = lambda c: [i for i, x in enumerate(clases) if x == c]                     # noqa: E731

        def cons(sentido: str, **kw) -> dict:
            return _dl.consecuencia_de(sentido, t, resolvio_a_quo, resolutivo_recurrida, quejoso,
                                       responsable, tenemos_conceptos, quien_recurre=quien_recurre,
                                       sobresee_ademas=sobresee_ademas, **kw)

        sols: list = []
        if t == "amparo_directo":
            sols.append(_sol("infundado", "niega", "siempre cabe que ningún concepto prospere",
                             list(range(len(probs))), cons("infundado")))
            if idx("procesal"):
                sols.append(_sol("fundado", "reposicion",
                                 "hay un problema de clase procesal (violación al procedimiento)",
                                 idx("procesal"), cons("fundado")))
            fondo = idx("fondo") + idx("procedencia")
            if fondo or not probs:
                sols.append(_sol("fundado", "para_efectos", "hay problemas de fondo que pueden prosperar",
                                 fondo, cons("fundado")))
            liso = [i for i in fondo if _pide_liso(probs[i])]
            if liso:
                sols.append(_sol("fundado", "liso_y_llano",
                                 "un problema de fondo pide mayor beneficio (prescripción, caducidad, "
                                 "cosa juzgada, nulidad lisa y llana)", liso, cons("fundado")))
        elif t == "amparo_revision":
            import fase_rama as _fr
            a = str(resolvio_a_quo or "").strip().lower()
            if not a:
                avisos.append("No se sabe qué resolvió el juzgado: la rama de cada solución queda "
                              "por determinar.")
            sols.append(_sol("infundado", "confirma", "siempre cabe que los agravios no prosperen",
                             list(range(len(probs))), cons("infundado")))
            sols.append(_sol("fundado", "revoca", "los agravios pueden prosperar",
                             list(range(len(probs))), cons("fundado")))
            if idx("procedencia") and a in ("concede", "sobresee_concede") \
                    and str(quien_recurre or "").lower() != "quejoso":
                sols.append(_sol("fundado", "revoca_sobresee",
                                 "un agravio de procedencia puede prosperar (fr. II del art. 93)",
                                 idx("procedencia"), cons("fundado", procedencia=True)))
            combate = " ".join(str((p or {}).get("combate") or "") for p in probs if isinstance(p, dict))
            extra = []
            if _fr.hay_violacion_procesal(combate):
                extra.append(("repone", {"violacion_procesal": True},
                              "un agravio afirma una violación en el procedimiento del juicio de amparo"))
            if a == "concede" and _fr.solo_los_efectos(combate):
                extra.append(("modifica_efectos", {"solo_efectos": True},
                              "los agravios sólo atacan los efectos de la concesión"))
            for tipo_ef, kw, por_que in extra:
                rama = _ta.rama_revision(a, "fundado", quien_recurre=quien_recurre, **kw)
                puntos = list((_ta.RAMAS_REVISION.get(rama) or {}).get("puntos") or [])
                sols.append(_sol("fundado", tipo_ef, por_que, list(range(len(probs))),
                                 {"rama": rama, "desenlace": puntos, "desenlace_nota": None,
                                  "conceptos_omitidos": None}))
        elif t == "queja":
            sols.append(_sol("infundado", "no_prospera", "siempre cabe que los agravios no prosperen",
                             list(range(len(probs))), cons("infundado")))
            sols.append(_sol("fundado", "prospera", "los agravios pueden prosperar",
                             list(range(len(probs))), cons("fundado")))
        elif t == "revision_fiscal":
            sols.append(_sol("infundado", "confirma", "siempre cabe que los agravios no prosperen",
                             list(range(len(probs))), cons("infundado")))
            sols.append(_sol("fundado", "revoca", "los agravios pueden prosperar",
                             list(range(len(probs))), cons("fundado")))
        # SIN DUPLICADOS (misma rama y mismo tipo de efecto) Y CON TOPE. El
        # corte nunca deja fuera la única del otro lado.
        vistas, unicas = set(), []
        for s in sols:
            k = (s["rama"], s["tipo_efecto"])
            if k not in vistas:
                vistas.add(k)
                unicas.append(s)
        if len(unicas) > tope:
            fuera = [s["tipo_efecto"] for s in unicas[tope:]]
            avisos.append(f"Hay más soluciones posibles que el tope de {tope}: quedan fuera {', '.join(fuera)}.")
            unicas = unicas[:tope]
        for i, s in enumerate(unicas, 1):
            s["id"] = f"S{i}"
        return {"version": VERSION, "tipo": t, "soluciones": unicas, "avisos": avisos}
    except Exception as ex:
        return {"version": VERSION, "tipo": "", "soluciones": [],
                "avisos": [f"No se pudieron enumerar las soluciones ({type(ex).__name__}); la "
                           f"deliberación sigue con sus dos vías."]}


def lado_opuesto(soluciones: list, referencia: dict | None) -> dict | None:
    """La MEJOR del lado contrario a `referencia` para la segunda casilla de la
    tarjeta (que sólo tiene dos): la primera del otro lado en orden del código.
    Si S2 fuera del mismo lado que S1, desaparecería de pantalla en silencio."""
    if not isinstance(referencia, dict):
        return None
    return next((s for s in soluciones or [] if isinstance(s, dict)
                 and s.get("prospera") != referencia.get("prospera")), None)
