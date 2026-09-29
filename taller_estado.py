"""Lo que una fase del taller deja para la siguiente, y cómo vuelve entero.

POR QUÉ EXISTE. Revisión fiscal 2/2026, 17-sep-2026, medido en los registros de
producción minuto a minuto. El proyecto tardaba 16 minutos y una parte era
trabajo hecho DOS y TRES veces:

  - La consulta del acervo —nueve preguntas al modelo y ocho reordenaciones,
    unos 30 s— corría en /taller/consultar, OTRA VEZ en /taller/proponer y OTRA
    en /taller/resolver. El acervo se guardaba en la fila (`estado.material`)
    pero al rehidratar la sesión nadie lo reponía, así que el worker que no lo
    había consultado veía «material no está» y volvía a consultar. Con dos
    workers y sin afinidad, eso es casi siempre. Y cada consulta elige tesis
    distintas —la reordenación es del modelo—, así que la propuesta se hacía
    sobre un acervo y el estudio sobre otro.

  - El artículo 150 del Reglamento Interior del IMSS se traía de internet en
    la propuesta (30 s), se guardaba con el material… y el resolver, que había
    perdido el material, lo volvía a traer (otros 30 s).

  - El contraste de la propuesta —una llamada con razonamiento alto, 55 a
    125 s— sólo necesita lo que el adelanto ya produjo: los planteamientos y
    los dos resúmenes. Esperaba en serie dentro de /taller/proponer, cuando
    podía haberse calculado mientras el secretario consultaba el acervo.

Aquí viven las piezas puras de esas dos correcciones: el acervo que va a la
fila COMPLETO y vuelve como el mismo `Material` —con sondeo, principios, sede
del acto, cuaderno y entidad, que antes se perdían—, y la huella y la espera
del contraste adelantado. Nada de esto toca la base: `main.py` pone las
lecturas y escrituras alrededor.

Regla de la casa ([[estado-entre-workers]]): lo que hace falta en una petición
posterior va a la fila y se repone al rescatar la sesión. La memoria del
proceso es caché, no verdad.
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import time

# Tope de elementos por lista al guardar. No recorta nada real —el material
# lleva 80 tesis como mucho y un sondeo trae 40 razonados—; es un freno por si
# un día una lista crece sin medida y la fila deja de caber.
_TOPE_LISTA = 200


def _lista(xs, n: int = _TOPE_LISTA) -> list:
    return [x for x in (xs or [])][:n]


def sondeo_ligero(s) -> dict | None:
    """El `fase_precedente.Sondeo` como dict de JSON puro. None si no hay."""
    if s is None:
        return None
    try:
        d = dataclasses.asdict(s) if dataclasses.is_dataclass(s) else dict(vars(s))
    except Exception:
        return None
    # Ida y vuelta por JSON: lo que no sea JSON se convierte a texto en vez de
    # tumbar el guardado del material entero.
    try:
        return json.loads(json.dumps(d, ensure_ascii=False, default=str))
    except Exception:
        return None


def sondeo_rehidratado(d):
    """El dict de vuelta a `Sondeo`, con sus métodos. None si no se puede."""
    if not isinstance(d, dict) or not d:
        return None
    try:
        import fase_precedente as _fp
        campos = {f.name for f in dataclasses.fields(_fp.Sondeo)}
        return _fp.Sondeo(**{k: v for k, v in d.items() if k in campos})
    except Exception:
        return None


def material_ligero(m) -> dict:
    """El acervo tal como va a la fila. `completo=True` marca que lleva TODO lo
    que los prompts leen; sin esa marca el rescate de la sesión no lo repone y
    se vuelve a consultar, que es lo que pasaba con las filas antiguas."""
    def _dicts(xs, n):
        return [x for x in (xs or []) if isinstance(x, dict)][:n]
    return {
        # LAS DE LA FIGURA, FUERA DEL TOPE DE 80 (fase E): van al final
        # (`fase6_rag.sumar_figura`) y no deben ni recortar lo de siempre ni
        # perderse ellas en un material grande.
        "tesis": (_dicts([t for t in (getattr(m, "tesis", None) or [])
                          if not (isinstance(t, dict) and t.get("cupo_figura"))], 80)
                  + _dicts([t for t in (getattr(m, "tesis", None) or [])
                            if isinstance(t, dict) and t.get("cupo_figura") and not t.get("para_requisito")], 12)
                  # LAS DE UN REQUISITO (rediseño, etapa 2), con cupo propio: van
                  # al final y el de la figura las tiraba al guardar.
                  + _dicts([t for t in (getattr(m, "tesis", None) or [])
                            if isinstance(t, dict) and t.get("para_requisito")], 12)),
        "normas": (_dicts([n for n in (getattr(m, "normas", None) or [])
                           if not (isinstance(n, dict) and n.get("para_requisito"))], 80)
                   + _dicts([n for n in (getattr(m, "normas", None) or [])
                             if isinstance(n, dict) and n.get("para_requisito")], 16)),
        "convencional": _dicts(getattr(m, "convencional", []), 24),
        "materia": str(getattr(m, "materia", "") or ""),
        "tipo_asunto": str(getattr(m, "tipo_asunto", "") or ""),
        # EL ESPEJO VIAJA POR LA FILA, NO POR LA MEMORIA. Son seis filas de siete
        # campos por planteamiento: kilobyte y medio.
        "espejo": _dicts(getattr(m, "espejo", []), 8),
        # LO QUE FALTABA Y OBLIGABA A CONSULTAR OTRA VEZ: sin el sondeo, la
        # propuesta perdía la corriente del acervo y el estudio sus moldes y
        # objeciones; sin los principios, el estudio no sabía con qué nociones
        # razona el circuito; sin sede y cuaderno, la verja de la suspensión
        # se quedaba ciega.
        "principios": _lista(getattr(m, "principios", []), 24),
        "sede_del_acto": str(getattr(m, "sede_del_acto", "") or ""),
        "cuaderno": str(getattr(m, "cuaderno", "") or ""),
        "entidad": str(getattr(m, "entidad", "") or ""),
        "tribunal": str(getattr(m, "tribunal", "") or ""),
        # ¿LA CONSULTA QUEDÓ PROVISIONAL? (rediseño, etapa 2: si vence la espera
        # de la pregunta decisiva, falta la figura que decide).
        "consulta_estado": dict(getattr(m, "consulta_estado", None) or {}),
        "requisitos": dict(getattr(m, "requisitos", None) or {}),
        "preceptos_de_internet": _lista(getattr(m, "preceptos_de_internet", []), 24),
        "sondeo": sondeo_ligero(getattr(m, "sondeo", None)),
        # LA PREGUNTA DECISIVA VIAJA CON EL MATERIAL (SPEC E3, AR 631/2025):
        # la propuesta, el estudio y la tarjeta la leen de aquí, en el worker
        # que sea. Un dict pequeño; si no es un dict, no va.
        "decisiva": (getattr(m, "decisiva", None)
                     if isinstance(getattr(m, "decisiva", None), dict) else None),
        "completo": True,
    }


def material_rehidratado(d: dict):
    """Un `Material` con lo guardado. None si no hay nada aprovechable."""
    if not isinstance(d, dict) or not (d.get("tesis") or d.get("normas")):
        return None
    import fase6_estudio as _f6m
    m = _f6m.Material()
    m.tesis = list(d.get("tesis") or [])
    m.normas = list(d.get("normas") or [])
    m.convencional = list(d.get("convencional") or [])
    m.materia = str(d.get("materia") or "")
    m.tipo_asunto = str(d.get("tipo_asunto") or "amparo_directo")
    m.espejo = [x for x in (d.get("espejo") or []) if isinstance(x, dict)]
    m.principios = list(d.get("principios") or [])
    m.sede_del_acto = str(d.get("sede_del_acto") or "")
    m.cuaderno = str(d.get("cuaderno") or "")
    m.entidad = str(d.get("entidad") or "")
    m.preceptos_de_internet = list(d.get("preceptos_de_internet") or [])
    m.sondeo = sondeo_rehidratado(d.get("sondeo"))
    m.decisiva = d.get("decisiva") if isinstance(d.get("decisiva"), dict) else None
    # El tribunal viaja con el material; la fuerza NO se recalcula aquí (el
    # material vuelve tal cual se guardó): la recalculan quienes la leen
    # —propuesta, estudio, documento— con `fuerza_juridica.anotar`, que
    # corrige un material guardado antes del 29-sep (`obligatoria = vincula`).
    m.tribunal = str(d.get("tribunal") or "")
    m.consulta_estado = dict(d.get("consulta_estado") or {})
    m.requisitos = dict(d.get("requisitos") or {})
    return m


def esta_completo(d) -> bool:
    return isinstance(d, dict) and bool(d.get("completo")) and bool(
        d.get("tesis") or d.get("normas"))


# ═══ EL CONTRASTE ADELANTADO ═════════════════════════════════════════════════

def problemas_de(r) -> list:
    """Los planteamientos tal como los recibe la propuesta. UNA sola forma de
    construirlos, porque la huella se calcula sobre ellos."""
    f = getattr(r, "fases", None)
    problemas = [p if isinstance(p, dict) else {"pregunta": str(p)}
                 for p in (getattr(f, "problemas", None) or [])]
    if not problemas and getattr(f, "problema_global", ""):
        problemas = [{"pregunta": f.problema_global}]
    return problemas


def entradas_contraste(r) -> tuple:
    """(problemas, resumen_acto, resumen_conceptos, es_recurso): lo ÚNICO que
    el contraste lee. Ni el acervo ni el contexto del secretario entran, y por
    eso puede calcularse antes de consultar."""
    f = getattr(r, "fases", None)
    e = getattr(r, "encargo", None)
    return (problemas_de(r),
            "\n".join((f.parrafos_acto() if f else []) or []),
            "\n".join((f.parrafos_conceptos() if f else []) or []),
            bool(e is not None and getattr(e, "es_recurso", False)))


def huella_contraste(r) -> str:
    """Identifica el adelanto del que sale un contraste. Si el secretario
    rehace el adelanto o cambia un planteamiento, la huella cambia y el
    contraste guardado no se usa."""
    base = json.dumps(list(entradas_contraste(r)), ensure_ascii=False,
                      sort_keys=True, default=str)
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


# LAS TAREAS SUELTAS LATEN. Mientras corren, rescriben su marca cada 45 s con
# `latido`; sin latido en este tiempo, el worker que las corría ya no está —un
# despliegue lo mata con SIGTERM y nadie escribe «fallo»— y se hace el trabajo
# aquí. Medido en producción (18-sep-2026): la corrida arrancó en la instancia
# vieja, murió con ella, y sin latido la pantalla esperó diez minutos.
LATIDO_ABANDONADO_S = 150.0
CONTRASTE_ABANDONADO_S = LATIDO_ABANDONADO_S
CONSULTA_ABANDONADA_S = LATIDO_ABANDONADO_S


def abandonada(doc: dict, ahora=time.time,
               abandonado: float = LATIDO_ABANDONADO_S) -> bool:
    """¿Una marca «en_curso» sin señales de vida?"""
    if not isinstance(doc, dict) or doc.get("estado") != "en_curso":
        return False
    try:
        ultimo = float(doc.get("latido") or doc.get("desde") or 0)
    except (TypeError, ValueError):
        ultimo = 0.0
    return ahora() - ultimo > abandonado


async def esperar_marca(huella: str, leer, que: str = "el contraste adelantado",
                        tope: float = 150.0, abandonado: float = CONTRASTE_ABANDONADO_S,
                        ahora=time.time, dormir=asyncio.sleep,
                        cada: float = 3.0) -> dict | None:
    """La marca que dejó una tarea del adelanto —{huella, estado, desde, …}—
    cuando está «listo» y es de ESTE adelanto; None si hay que hacer el
    trabajo aquí.

    `leer()` devuelve el dict guardado o None. Se espera sólo mientras la fila
    dice «en_curso» y el que lo calcula sigue vivo; nunca más de `tope`
    segundos, que es menos de lo que costaría calcularlo de nuevo.
    """
    t0 = ahora()
    avisado = False
    while True:
        doc = leer()
        if not isinstance(doc, dict) or doc.get("huella") != huella:
            return None
        estado = doc.get("estado")
        if estado == "listo":
            return doc
        if estado != "en_curso":
            return None
        if abandonada(doc, ahora, abandonado):
            print(f"   ⏳ {que} no dio señales en {abandonado:.0f} s: se hace aquí")
            return None
        if ahora() - t0 >= tope:
            print(f"   ⏳ {que} sigue en curso tras {tope:.0f} s de espera: se hace aquí")
            return None
        if not avisado:
            print(f"   ⏳ {que} sigue en curso: se espera")
            avisado = True
        await dormir(cada)


async def esperar_contraste(huella: str, leer, tope: float = 150.0,
                            ahora=time.time, dormir=asyncio.sleep,
                            cada: float = 3.0) -> list | None:
    """Los planteamientos contrastados que dejó el adelanto, o None si hay que
    calcularlos aquí."""
    doc = await esperar_marca(huella, leer, "el contraste adelantado", tope,
                              CONTRASTE_ABANDONADO_S, ahora, dormir, cada)
    if doc is None:
        return None
    items = doc.get("items")
    return list(items) if isinstance(items, list) else None


async def esperar_consulta(huella: str, leer, tope: float = 90.0,
                           ahora=time.time, dormir=asyncio.sleep,
                           cada: float = 2.0) -> bool:
    """¿La consulta automática del acervo de ESTE adelanto está lista? Espera
    mientras corre; False si hay que consultar aquí."""
    doc = await esperar_marca(huella, leer, "la consulta automática del acervo",
                              tope, CONSULTA_ABANDONADA_S, ahora, dormir, cada)
    return doc is not None


# ═══ LOS ESTADOS DE LA SESIÓN (rediseño, etapa 4, pieza 6) ═══════════════════
#
# POR QUÉ. Cada marca de la fila guarda sólo la huella del ADELANTO, y hay
# marcas que leen más de lo que esa huella cubre (leído en main.py el
# 29-sep-2026):
#   · el contraste se calcula CON la ficha procesal (`_taller_precontrastar`
#     le pasa `_taller_ficha_bloque(r)`) y se guarda con la huella del
#     adelanto, que no la lleva;
#   · el análisis lee la ficha y los segmentos, y `analisis_litis.huella` no;
#   · el plan recibe ficha y contraste, y `plan_estudio.huella_entradas` no;
#   · los requisitos guardan «a0/a1» —si HABÍA análisis—, no CUÁL;
#   · el proyecto no guarda huella ninguna.
# El caso que lo vuelve grave es el del AR 631/2025: el secretario corrige en
# el formulario quién recurre, los planteamientos no cambian, la huella del
# adelanto tampoco, y el contraste, la propuesta y el proyecto hechos con la
# ficha equivocada siguen pasando por buenos. Y la pantalla no tiene UNA
# insignia que diga en qué punto está el asunto: junta a mano el estado crudo
# de tres marcas (`_taller_avance`).
#
# Aquí viven, PURAS, las dos piezas que lo arreglan sin tocar ninguna marca de
# hoy: `manifiesto()` arma lo que una marca «consume» —la huella de cada
# insumo que leyó— y `estado_sesion()` lo compara con lo de ahora. No leen la
# base, ni banderas, ni el reloj si no se les da: main.py pone las lecturas y
# la bandera «estados_sesion» alrededor (aún sin cablear).
#
# LAS MARCAS QUE HAY HOY EN LA FILA `taller_sesiones` (leídas de main.py como
# texto) Y LA HUELLA QUE LLEVA CADA UNA:
#   estado.huella            la identidad del adelanto (`huella_contraste`). No
#                            es una marca: es contra lo que se guardan todas.
#   estado.consulta          {huella, estado, desde, latido, segundos}. Huella:
#                            la del adelanto. Va en la MISMA escritura que el
#                            material (`_taller_guardar_material`).
#   estado.material          el acervo (`material_ligero`). Sin huella propia.
#     .requisitos            la demostración; huella «adelanto|a0/1|d0/1».
#     .decisiva              la pregunta decisiva que vio la consulta.
#     .consulta_estado       completa | provisional (y qué falta).
#   estado.decisiva          {huella: «adelanto:ficha», estado, doc}: la ÚNICA
#                            que ya lleva la ficha (`pregunta_decisiva.huella`).
#   estado.contraste         {huella, estado, items, segundos}. Huella: adelanto.
#   estado.analisis          {huella, huella_analisis, estado, doc}. Huella:
#                            adelanto; huella_analisis: acto, escrito, autos,
#                            problemas y versión (ni ficha ni segmentos).
#   estado.propuesta         {huella, estado, respuesta, [ficha, origen]}. Huella:
#                            adelanto. La columna `propuestas` no lleva ninguna.
#   estado.global_propuesta  {huella, global, desde}: la última global que vio
#                            la pantalla. Huella: adelanto. Sin campo estado.
#   estado.deliberacion      {huella, clave, estado, deliberacion}; clave =
#                            adelanto + contexto + registros del material.
#   estado.proyecto          la ficha del último proyecto (y `proyectos`, la
#                            pila de doce). SIN huella.
#   plan (otra columna)      {rev, version, huella: adelanto, corridas,
#                            pedido_clave, planes: {clave: casilla}, inventario:
#                            la lectura del escrito, recalificaciones: {clave:
#                            casilla}}. La clave del plan lleva criterio,
#                            entradas, contexto, suplencia y decisiva.
#   estado.avisos, .evaluacion, .encargo, .fases, .partes…: datos, no marcas.
#   La marca «tarjeta» ya no existe: se recalcula en cada GET (28-sep).
#
# LO QUE CADA UNA CONSUMIRÁ está en `MARCAS` (abajo). Al cablearlo, cada
# escritora pega `consume = manifiesto(entradas, marca)` DENTRO de su marca —el
# parche atómico la sustituye entera— y nada más cambia: la huella de hoy sigue
# guardando las escrituras como siempre. Una marca en curso lleva su consume
# desde que arranca (el `extra=` de `_taller_con_latido`), para que la insignia
# diga «en_analisis» desde el primer latido y no «nada».

MANIFIESTO_VERSION = "estados-1"

# QUÉ ES CADA INSUMO Y CÓMO SE TOMA SU HUELLA. «identidad»: el valor ya ES una
# identidad —una huella, una clave, una versión— y se copia tal cual.
# «contenido»: se guarda el sha1 de lo que se leyó, sin tiempos ni costes (lo
# mismo leído otra vez con otro `segundos` no es otra cosa).
INSUMOS = {
    "adelanto": ("identidad", "estado.huella = huella_contraste(r): planteamientos, "
                              "los dos resúmenes y es_recurso"),
    "ficha": ("contenido", "el texto de ficha_procesal.bloque(de_resultado(r))"),
    "inventario": ("contenido", "los segmentos que vio quien lee —piso más lectura del "
                                "escrito—: (id, concepto, anclas), como huella_entradas['segs']"),
    "analisis": ("contenido", "el documento del análisis neutral; «» si se trabajó sin él"),
    "decisiva": ("contenido", "la cuestión decisiva, su figura y los hechos que deciden; "
                              "«» sin ella"),
    "material.registros": ("contenido", "los registros de las tesis del material, ordenados"),
    "material.normas": ("contenido", "cuerpo_legal|artículo de las normas del material, ordenados"),
    "material.espejo": ("contenido", "las filas del espejo (NEUN, nivel, similitud) CON la "
                                     "versión de la tabla de calibración de la OAJ"),
    "requisitos": ("contenido", "la demostración por requisitos (material.requisitos)"),
    "contraste": ("contenido", "los planteamientos contrastados; «» si no estaba listo"),
    "propuesta": ("contenido", "la respuesta de la propuesta"),
    "global": ("contenido", "la propuesta global que leen el árbol y el plan"),
    "recalificacion": ("identidad", "la clave de la recalificación que se usó; «» sin ella"),
    "criterio": ("contenido", "problema, sentido, razón, jerarquía y grupo de cada criterio"),
    "contexto": ("contenido", "el contexto que escribió el secretario"),
    "solucion": ("identidad", "solucion_id@versión de la solución elegida (etapa 3; reservado)"),
    "regla": ("identidad", "arbol_decision.REGLA_VERSION"),
    "version_plan": ("identidad", "la versión efectiva del plan (plan-6, plan-7…)"),
    "plan": ("identidad", "la clave del plan con que se escribió; «» sin plan"),
    "formato": ("identidad", "estándar | moderna"),
    "variante": ("identidad", "la variante del prompt del estudio (v1…v4)"),
    "commit": ("identidad", "el commit desplegado: se registra y NUNCA invalida"),
}

# EL COMMIT POR SÍ SOLO NO INVALIDA NADA. Cada despliegue lo cambia; si
# invalidara, todos los proyectos saldrían «desactualizados» al día siguiente
# de cualquier arreglo que no tocó su camino. Se guarda para el auditor.
NO_INVALIDA = frozenset({"commit"})

# LO QUE LEE EL PLAN, que el proyecto hereda. Todo sale de lo que
# `_taller_plan_entradas` y `_taller_plan_correr` le pasan al planificador.
_CONSUME_PLAN = ("adelanto", "ficha", "inventario", "analisis", "decisiva",
                 "material.registros", "material.normas", "requisitos", "contraste",
                 "global", "recalificacion", "criterio", "contexto", "solucion",
                 "regla", "version_plan")

# EL GRAFO, EXPLÍCITO. `consume`: los insumos que la marca LEE (leído en cada
# escritora de main.py). `depende_de`: las marcas cuya caída la arrastra.
# Lo que NO está, y por qué:
#   · el análisis NO consume el inventario: corre en el adelanto con el piso de
#     segmentos, y la lectura del escrito añade los suyos DESPUÉS, siempre. Si
#     lo consumiera, todo análisis saldría desactualizado en cuanto la lectura
#     termina —una alarma que acusa a todos no mide nada—. Lo que el análisis
#     no vio queda «sin_clasificar» en el plan (etapa 4) y se avisa como
#     pendiente del proyecto, uno por uno.
#   · los requisitos NO consumen el material: `requisitos.demostracion` lee el
#     principal, el análisis, la decisiva y la ficha, y la recuperación SUMA al
#     material. Si lo consumieran, su propia escritura los invalidaría.
#   · la recalificación NO consume el criterio: su salida ES el criterio que
#     leen el plan y el proyecto. Su identidad es su clave (principal, sentido,
#     razón y pendientes), y un criterio distinto usa otra casilla.
#   · ni el plan ni el estudio leen la deliberación (plan_estudio,
#     arbol_decision, recalificar, fase6_estudio y redactor_adelanto no la
#     nombran): sólo la tarjeta. Lo que pasa de ella al plan pasa por el
#     secretario, en el criterio, que es insumo propio. Por eso recalibrar la
#     tabla de la OAJ deja desactualizada la deliberación y no el proyecto.
MARCAS = {
    "inventario": {"vive": "plan.inventario", "consume": ("adelanto",), "depende_de": ()},
    "decisiva": {"vive": "estado.decisiva", "consume": ("adelanto", "ficha"), "depende_de": ()},
    "contraste": {"vive": "estado.contraste", "consume": ("adelanto", "ficha"), "depende_de": ()},
    "analisis": {"vive": "estado.analisis", "consume": ("adelanto", "ficha"), "depende_de": ()},
    "consulta": {"vive": "estado.consulta + estado.material",
                 "consume": ("adelanto", "contexto", "decisiva"), "depende_de": ("decisiva",)},
    "requisitos": {"vive": "estado.material.requisitos",
                   "consume": ("adelanto", "ficha", "analisis", "decisiva"),
                   "depende_de": ("consulta", "analisis", "decisiva")},
    "propuesta": {"vive": "estado.propuesta",
                  "consume": ("adelanto", "ficha", "contexto", "decisiva", "analisis", "contraste",
                              "requisitos", "material.registros", "material.normas"),
                  "depende_de": ("consulta", "contraste", "analisis", "requisitos")},
    "global_propuesta": {"vive": "estado.global_propuesta",
                         "consume": ("adelanto", "propuesta"), "depende_de": ("propuesta",)},
    "deliberacion": {"vive": "estado.deliberacion",
                     "consume": ("adelanto", "contexto", "decisiva", "analisis", "propuesta",
                                 "material.registros", "material.normas", "material.espejo"),
                     "depende_de": ("propuesta",)},
    "recalificacion": {"vive": "plan.recalificaciones[clave]",
                       "consume": ("adelanto", "contexto", "global", "regla"),
                       "depende_de": ("global_propuesta",)},
    "plan": {"vive": "plan.planes[clave]", "consume": _CONSUME_PLAN,
             "depende_de": ("inventario", "consulta", "requisitos", "contraste", "analisis",
                            "global_propuesta", "recalificacion")},
    # El proyecto depende de lo mismo que el plan, y del plan: un proyecto
    # escrito SIN plan (la v3, o la v1/v2) también cae si cae lo de arriba.
    "proyecto": {"vive": "estado.proyecto (y proyectos[0])",
                 "consume": _CONSUME_PLAN + ("plan", "formato", "variante", "commit"),
                 "depende_de": ("plan", "inventario", "consulta", "requisitos", "contraste",
                                "analisis", "global_propuesta", "recalificacion")},
}

# ¿«LLEGÓ» INVALIDA? Una marca que consumió «» (el análisis no estaba) y ahora
# lo hay. Decisión pendiente de David; por omisión SÍ: la propuesta que se hizo
# sin el contraste es otra propuesta que la que se haría hoy. Con False, la
# llegada se reporta (invalida: False) y la marca sigue vigente.
LLEGADA_DESACTUALIZA = True

# CUÁNDO UNA CORRIDA SIN LATIDO SE DA POR MUERTA. Cada rama tiene el suyo en su
# módulo y aquí no se importan (plan_estudio y compañía leen el entorno al
# importarse; esto es puro): test_estados_sesion.py comprueba que casan.
_ABANDONO_S = {"plan": 60.0,             # plan_estudio.PLAN_ABANDONADO_S
               "inventario": 60.0,       # inventario_escrito.ABANDONADA_S
               "recalificacion": 45.0}   # recalificar.ABANDONADA_S

ESTADOS_SESION = ("en_analisis", "faltan_insumos", "justificacion_pendiente",
                  "borrador_sin_plan", "proyecto_verificado")

# Lo que cambia en cada lectura sin cambiar lo leído: si entrara en la huella,
# la misma propuesta leída dos veces «cambiaría». Los de tiempo sólo se quitan
# si son un número: «hecho» o «desde» con texto pueden ser un hecho del asunto.
_VOLATILES = frozenset({"uso", "consume", "evaluacion_aplicada"})
_VOLATILES_NUM = frozenset({"segundos", "coste", "coste_usd", "desde", "latido", "hecho"})


def _ws(x) -> str:
    return " ".join(str(x if x is not None else "").split())


def _seq(x) -> list:
    """Una lista para recorrer: lo que no lo sea —un `true`, un número, un
    texto donde iba una lista— no se recorre. Una fila escrita a mano o por un
    código viejo no tumba la insignia."""
    return list(x) if isinstance(x, (list, tuple)) else []


def _vacio(x) -> bool:
    if x is None:
        return True
    if isinstance(x, str):
        return not x.strip()
    if isinstance(x, (list, tuple, dict, set, frozenset)):
        return not x
    return False


def _plano(x):
    """Lo leído como JSON estable: dataclasses y objetos a dict, sin volátiles.
    Un objeto sin dict iría por `repr` —con su dirección de memoria— y la huella
    cambiaría en cada proceso."""
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        x = dataclasses.asdict(x)
    elif not isinstance(x, (dict, list, tuple, set, frozenset, str, int, float, bool,
                            type(None))) and hasattr(x, "__dict__"):
        x = dict(vars(x))
    if isinstance(x, dict):
        return {str(k): _plano(v) for k, v in x.items()
                if str(k) not in _VOLATILES
                and not (str(k) in _VOLATILES_NUM and isinstance(v, (int, float))
                         and not isinstance(v, bool))}
    if isinstance(x, (list, tuple)):
        return [_plano(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted((_plano(v) for v in x), key=lambda v: json.dumps(v, sort_keys=True, default=str))
    return x


def _sha(x) -> str:
    base = json.dumps(x, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]


def _h_contenido(x) -> str:
    """«» si no hay nada (se SABE que falta); si no, el sha1 de lo leído."""
    return "" if _vacio(x) else _sha(_plano(x))


def _h_identidad(x) -> str:
    return _ws(x)[:120]


def _h_texto(x) -> str:
    t = _ws(x)
    return hashlib.sha1(t.encode("utf-8")).hexdigest()[:12] if t else ""


def _h_inventario(segs) -> str:
    """Los (id, concepto, anclas) EN ORDEN, como `huella_entradas['segs']`: el
    orden del inventario es el de la exposición."""
    if not isinstance(segs, (list, tuple)):
        return _h_contenido(segs)
    if not segs:
        return ""
    return _sha([[str(s.get("id") or ""), str(s.get("concepto") or ""),
                  sorted(str(a) for a in _seq(s.get("anclas")))]
                 if isinstance(s, dict) else _ws(s) for s in segs])


def _h_decisiva(doc) -> str:
    """La cuestión, la figura y los hechos que deciden: con ellos se buscan las
    tesis de la figura y se arma la clave del plan. Una no formulada es «»."""
    if not (isinstance(doc, dict) and doc.get("formulada") and _ws(doc.get("pregunta_decisiva"))):
        return ""
    return _sha([_ws(doc.get("pregunta_decisiva")), _ws(doc.get("figura")),
                 [_ws(h) for h in _seq(doc.get("hechos_que_deciden")) if _ws(h)]])


def _h_criterio(crit) -> str:
    """Los mismos campos que `plan_estudio.clave` (sin importarlo): problema,
    sentido, razón, jerarquía y grupo. Vale igual para `Criterio` y para el
    dict que manda la pantalla."""
    if isinstance(crit, str):
        try:
            crit = json.loads(crit) if crit.strip() else []
        except ValueError:
            return _h_texto(crit)
    if not isinstance(crit, (list, tuple)) or not crit:
        return ""

    def _g(c, k):
        return c.get(k) if isinstance(c, dict) else getattr(c, k, None)
    return _sha([{"problema": _ws(_g(c, "problema")),
                  "sentido": _ws(_g(c, "sentido")).lower(),
                  "razon": _ws(_g(c, "razonamiento") or _g(c, "razon")),
                  "jerarquia": _ws(_g(c, "jerarquia")).lower(),
                  "grupo": _ws(_g(c, "grupo"))} for c in crit])


_HUELLA = {
    "adelanto": _h_identidad, "ficha": _h_texto, "inventario": _h_inventario,
    "analisis": _h_contenido, "decisiva": _h_decisiva, "requisitos": _h_contenido,
    "contraste": _h_contenido, "propuesta": _h_contenido, "global": _h_contenido,
    "recalificacion": _h_identidad, "criterio": _h_criterio, "contexto": _h_texto,
    "solucion": _h_identidad, "regla": _h_identidad, "version_plan": _h_identidad,
    "plan": _h_identidad, "formato": _h_identidad, "variante": _h_identidad,
    "commit": _h_identidad,
}


def version_tabla_oaj(cal) -> str:
    """La identidad de la tabla de calibración de la OAJ
    (`fase_oaj.cargar_calibracion()`). El JSON no trae número de versión y se
    relee al cambiar en disco: su contenido ES su versión. «» sin tabla."""
    return _h_contenido(cal) if isinstance(cal, dict) else ""


def huella_material(m, version_oaj=None) -> dict:
    """{material.registros, material.normas[, material.espejo]} del acervo.

    Se mira el material TAL COMO VA A LA FILA (`material_ligero`, con sus
    topes): si se mirara el objeto en memoria, un espejo de nueve grupos daría
    otra huella que el de ocho que se guardó, y todo lo leído de la fila saldría
    «cambiado». Las tesis de la figura y las de los requisitos cuentan: el que
    leyó después de sumarlas las vio.

    El espejo sólo con `version_oaj` (aunque sea «»): sus porcentajes salen de
    esa tabla, y recalibrarla cambia lo que la pantalla enseñaría con las mismas
    filas. Sin la versión no se sabe, y lo que no se sabe no se compara."""
    if m is None:
        d = {}
    elif isinstance(m, dict):
        d = m
    else:
        d = material_ligero(m)
    tesis = [t for t in _seq(d.get("tesis")) if isinstance(t, dict)]
    normas = [n for n in _seq(d.get("normas")) if isinstance(n, dict)]
    espejo = [e for e in _seq(d.get("espejo")) if isinstance(e, dict)]
    regs = sorted({_ws(t.get("registro") or t.get("clave")) for t in tesis} - {""})
    arts = sorted({f"{_ws(n.get('cuerpo_legal') or n.get('fuente'))}|{_ws(n.get('articulo'))}"
                   for n in normas} - {"|"})
    out = {"material.registros": _sha(regs) if regs else "",
           "material.normas": _sha(arts) if arts else ""}
    if version_oaj is not None:
        grupos = sorted([_ws(e.get("problema")),
                         sorted([_ws(f.get("neun") or f.get("expediente")), _ws(f.get("nivel")),
                                 _ws(f.get("similitud")), bool(f.get("cota_inferior"))]
                                for f in _seq(e.get("filas")) if isinstance(f, dict))]
                        for e in espejo)
        vacio = not (tesis or normas or espejo)
        out["material.espejo"] = "" if vacio else _sha({"filas": grupos, "tabla": _ws(version_oaj)})
    return out


def manifiesto(entradas: dict, marca: str = "") -> dict:
    """El «consume»: {v, insumo: huella} de lo que se leyó.

    `entradas` trae lo CRUDO que main tiene a mano —el texto de la ficha, el
    documento del análisis, el `Material` o su dict, los segmentos, el
    criterio…— y aquí se toma la huella de cada uno según `INSUMOS`. Para el
    espejo, `version_oaj` (ver `version_tabla_oaj`).

    Una clave AUSENTE es «no se sabe» y no se compara nunca; un valor vacío
    (None, «», []) es «se sabe que faltaba» y queda «». Por eso una escritora
    que no leyó el análisis (la bandera estaba apagada) NO pone la clave: si
    pusiera None, la llegada del análisis la daría por desactualizada.

    Con `marca`, sólo los insumos que esa marca consume (`MARCAS`)."""
    ent = entradas if isinstance(entradas, dict) else {}
    out = {"v": MANIFIESTO_VERSION}
    mat = {}
    if "material" in ent:
        mat = huella_material(ent.get("material"), ent["version_oaj"] if "version_oaj" in ent else None)
    for k in INSUMOS:
        if k in mat:
            out[k] = mat[k]
        elif k in ent and k in _HUELLA:
            out[k] = _HUELLA[k](ent[k])
    return consume_de(marca, out) if marca else out


def consume_de(marca: str, huellas: dict) -> dict:
    """De unas huellas ya calculadas (las de `manifiesto` sin marca), sólo las
    que `marca` consume. Una marca desconocida da {}: su escritora guarda sin
    consume y la marca queda «sin_registro», nunca se tumba una escritura."""
    spec = MARCAS.get(marca)
    if spec is None:
        return {}
    h = huellas if isinstance(huellas, dict) else {}
    out = {"v": str(h.get("v") or MANIFIESTO_VERSION)}
    for k in spec["consume"]:
        if k in h:
            out[k] = h[k]
    return out


# ─── el grafo ───────────────────────────────────────────────────────────────

def orden() -> list:
    """Las marcas en orden topológico (cada una después de las que la
    arrastran). Un ciclo es un error de este archivo: lo caza la prueba."""
    hechas, fuera, quedan = set(), [], list(MARCAS)
    while quedan:
        listas = [m for m in quedan if all(d in hechas for d in MARCAS[m]["depende_de"])]
        if not listas:
            raise ValueError(f"ciclo en MARCAS: {quedan}")
        for m in listas:
            fuera.append(m)
            hechas.add(m)
            quedan.remove(m)
    return fuera


def dependientes(marca: str) -> list:
    """Las marcas que caen si cae `marca` (cierre transitivo), en orden."""
    caen, cambio = {marca}, True
    while cambio:
        cambio = False
        for m, spec in MARCAS.items():
            if m not in caen and any(d in caen for d in spec["depende_de"]):
                caen.add(m)
                cambio = True
    caen.discard(marca)
    return [m for m in orden() if m in caen]


def arrastran_a(marca: str) -> list:
    """Las marcas que pueden arrastrar a `marca` (las de las que depende, en
    cierre transitivo), en orden: dónde buscar la causa de su caída."""
    arriba, cambio = set(MARCAS.get(marca, {}).get("depende_de", ())), True
    while cambio:
        cambio = False
        for m in list(arriba):
            for d in MARCAS[m]["depende_de"]:
                if d not in arriba:
                    arriba.add(d)
                    cambio = True
    return [m for m in orden() if m in arriba]


def invalida_por(insumo: str) -> list:
    """Las marcas que deja desactualizadas un cambio de `insumo`: las que lo
    consumen y todo lo que cuelga de ellas. El commit, ninguna."""
    if insumo in NO_INVALIDA:
        return []
    caen = set()
    for m, spec in MARCAS.items():
        if insumo in spec["consume"]:
            caen.add(m)
            caen.update(dependientes(m))
    return [m for m in orden() if m in caen]


def grafo() -> dict:
    """La tabla entera, para el auditor y la pantalla: qué consume cada marca,
    de qué depende y qué arrastra al caer."""
    return {"version": MANIFIESTO_VERSION,
            "insumos": {k: {"tipo": t, "que": q} for k, (t, q) in INSUMOS.items()},
            "marcas": {m: {"vive": s["vive"], "consume": list(s["consume"]),
                           "depende_de": list(s["depende_de"]), "arrastra": dependientes(m)}
                       for m, s in MARCAS.items()},
            "no_invalida": sorted(NO_INVALIDA), "orden": orden()}


# ─── la fila ────────────────────────────────────────────────────────────────

def _partes(fila_estado, plan=None) -> tuple:
    """(estado, plan) de lo que llegue: el `estado` suelto o la fila entera
    {estado, plan, …}. El `estado` nunca tiene una clave «estado» en su primer
    nivel; la fila sí."""
    f = fila_estado if isinstance(fila_estado, dict) else {}
    if isinstance(f.get("estado"), dict) and "huella" not in f:
        est = f["estado"]
        if plan is None:
            plan = f.get("plan")
    else:
        est = f
    return est, (plan if isinstance(plan, dict) else {})


def _consume(doc) -> dict:
    c = doc.get("consume") if isinstance(doc, dict) else None
    return c if isinstance(c, dict) and c else {}


def _del_adelanto(doc, hu: str) -> bool:
    """¿La marca es de este adelanto? La de la decisiva lleva «adelanto:ficha»
    y la de los requisitos «adelanto|a1|d0»: manda lo de antes del separador."""
    if not (hu and isinstance(doc, dict)):
        return False
    return str(doc.get("huella") or "").split(":")[0].split("|")[0] == hu


def _global_de(est: dict, hu: str):
    """La global de ESTE adelanto, como la elige `main._taller_global_de_fila`:
    la marca propia primero, si no la de la propuesta calculada sola."""
    gp = est.get("global_propuesta")
    if _del_adelanto(gp, hu) and isinstance(gp.get("global"), dict):
        return gp["global"]
    pr = est.get("propuesta")
    resp = pr.get("respuesta") if isinstance(pr, dict) else None
    if _del_adelanto(pr, hu) and pr.get("estado") == "listo" \
            and isinstance(resp, dict) and isinstance(resp.get("global"), dict):
        return resp["global"]
    return None


def actuales_de_fila(fila_estado, plan=None, version_oaj=None) -> dict:
    """Lo que se puede saber de lo ACTUAL con la fila sola, con las mismas
    reglas que `manifiesto`. Es lo que usa /taller/en-curso, que lista veinte
    asuntos sin rehidratar ninguno.

    No sale de la fila —y por eso no se compara si main no lo pone en
    `actuales`—: la ficha (hace falta el resultado), el inventario (el piso de
    segmentos se calcula), el criterio y el contexto (vienen de la pantalla),
    la regla, el commit, la clave del plan que se pediría y la recalificación.
    Una marca en curso tampoco dice nada todavía: lo actual es lo que dejará.
    """
    est, _pl = _partes(fila_estado, plan)
    hu = _ws(est.get("huella"))
    cur = {"v": MANIFIESTO_VERSION}
    if not hu:
        return cur
    cur["adelanto"] = _h_identidad(hu)
    dec = est.get("decisiva")
    if _del_adelanto(dec, hu):
        if dec.get("estado") == "listo":
            cur["decisiva"] = _h_decisiva(dec.get("doc"))
        elif dec.get("estado") in ("fallo", "error"):
            cur["decisiva"] = ""
    for k, campo in (("contraste", "items"), ("analisis", "doc"), ("propuesta", "respuesta")):
        d = est.get(k)
        if _del_adelanto(d, hu):
            if d.get("estado") == "listo":
                cur[k] = _h_contenido(d.get(campo))
            elif d.get("estado") in ("fallo", "error"):
                cur[k] = ""
    mat, con = est.get("material"), est.get("consulta")
    if isinstance(mat, dict) and (mat.get("tesis") or mat.get("normas") or mat.get("espejo")):
        cur.update(huella_material(mat, version_oaj))
        req = mat.get("requisitos")
        if _del_adelanto(req, hu):
            if req.get("requisitos"):
                cur["requisitos"] = _h_contenido(req)
            elif req.get("estado") in ("tiempo", "fallo"):
                cur["requisitos"] = ""
    elif _del_adelanto(con, hu) and con.get("estado") in ("fallo", "error"):
        cur.update(huella_material(None, version_oaj))
    g = _global_de(est, hu)
    if g is not None:
        cur["global"] = _h_contenido(g)
    return cur


def _reloj(ahora):
    return ahora if callable(ahora) else (lambda: float(ahora))


def _crudo(marca: str, doc: dict, ahora) -> str:
    """listo | en_curso | fallo de una marca presente. Sin `ahora` no se
    envejece nada: una marca en curso sigue en curso aunque su latido sea de
    ayer (lo decide quien tiene reloj, no esta función)."""
    e = _ws(doc.get("estado")).lower()
    if e == "en_curso":
        if ahora is not None and abandonada(doc, _reloj(ahora),
                                            _ABANDONO_S.get(marca, LATIDO_ABANDONADO_S)):
            return "fallo"
        return "en_curso"
    if e in ("fallo", "error", "tiempo"):
        return "fallo"
    # «listo», «vacio» (la lectura que no halló nada) y las marcas sin campo
    # estado —global_propuesta, proyecto, la demostración—, que existen hechas.
    return "listo"


def _doc_marca(marca: str, est: dict, pl: dict, claves: dict):
    if marca == "inventario":
        d = pl.get("inventario")
    elif marca in ("plan", "recalificacion"):
        rama = pl.get("planes" if marca == "plan" else "recalificaciones")
        k = claves.get(marca) or ""
        d = rama.get(k) if (k and isinstance(rama, dict)) else None
    elif marca == "requisitos":
        m = est.get("material")
        d = m.get("requisitos") if isinstance(m, dict) else None
    else:
        d = est.get(marca)
    return d if isinstance(d, dict) and d else None


def _comparar(marca: str, cons: dict, cur: dict, llegada: bool) -> list:
    fuera = []
    for k in MARCAS[marca]["consume"]:
        if k in NO_INVALIDA or k not in cons or k not in cur:
            continue
        antes, ahora = cons[k], cur[k]
        if antes == ahora:
            continue
        tipo = "llegada" if antes in ("", None) else ("retirada" if ahora in ("", None) else "cambio")
        fuera.append({"marca": marca, "insumo": k, "tipo": tipo, "antes": antes, "ahora": ahora,
                      "invalida": tipo != "llegada" or llegada})
    return fuera


def _pendientes(ficha: dict) -> list:
    """Lo que impide decir «verificado» aunque nada esté desactualizado. Todo
    se lee de la ficha del proyecto: nada nuevo se escribe para esto."""
    fuera = []
    for f in _seq(ficha.get("fuentes_tardias")):
        if isinstance(f, dict) and f.get("sustantiva"):
            fuera.append({"tipo": "fuente_tardia", "fuente": _ws(f.get("fuente")),
                          "unidades": [str(u) for u in _seq(f.get("unidades"))][:8]})
    fp = ficha.get("plan") if isinstance(ficha.get("plan"), dict) else {}
    for src in (ficha, fp):
        # LO QUE EL CÓDIGO LE CAMBIÓ AL PLANIFICADOR (`plan_estudio._cambios`,
        # que `para_ficha` copia a la ficha con su bandera): {segmento | unidad
        # | plan: id, campo, antes, despues, regla, cuenta}. Sólo lo que
        # «cuenta» —calificación, razón, lo que deja algo sin estudio—: la
        # organización pondría pendiente casi todo proyecto. Sin «cuenta» se
        # tiene por que cuenta (marcar de más sólo avisa); «aceptado» es la
        # marca que pondrá la pantalla cuando el secretario lo vea.
        for c in _seq(src.get("cambios_sin_justificar")):
            if isinstance(c, dict) and c.get("cuenta", True) and not c.get("aceptado"):
                fuera.append({"tipo": "cambio_sin_justificar",
                              **{k: c.get(k) for k in ("segmento", "unidad", "plan", "seg", "campo",
                                                       "antes", "despues", "regla") if k in c}})
        # LA EXCLUSIÓN SIN PRUEBA (`exclusiones.SIN_PRUEBA`); el nombre del
        # mapa, «justificacion_pendiente», se acepta igual.
        for x in _seq(src.get("exclusiones")):
            if isinstance(x, dict) and x.get("estado") in ("sin_prueba", "justificacion_pendiente"):
                fuera.append({"tipo": "exclusion", "id": x.get("id"), "clase": x.get("tipo"),
                              "objeto": x.get("segmento_o_problema") or x.get("objeto")})
    # LA RELACIÓN QUE EL PLAN NO ENTENDIÓ: con la etapa 4 ya no se hace pasar
    # por «necesaria» (plan_estudio.normalizar); queda «sin_clasificar» en la
    # proposición, y eso no es un proyecto verificado.
    for p in _seq(fp.get("proposiciones")):
        if isinstance(p, dict) and _ws(p.get("relacion")).lower() == "sin_clasificar":
            fuera.append({"tipo": "sin_clasificar", "proposicion": p.get("id")})
    for s in _seq(fp.get("segmentos")):
        if not isinstance(s, dict):
            continue
        et = _ws(s.get("etiqueta")).lower()
        if et == "sin_clasificar":
            fuera.append({"tipo": "sin_clasificar", "seg": s.get("id")})
        # (Revisión adversarial del 29-sep: aquí había un «fundado
        # insuficiente sin base» que miraba `con_p` en el segmento. `con_p` es
        # del catálogo RAZONES, no del segmento, y la ficha del plan
        # (`para_ficha`) no copia ni `ataca` ni `razon_p`: no podía saltar
        # nunca. Cuando el plan lleve la base que sobrevive —etapa 4, pieza 5—
        # se comprobará con la base de verdad.)
    if ficha.get("estado_salida") == "justificacion_pendiente" \
            and not any(p["tipo"] == "fuente_tardia" for p in fuera):
        fuera.append({"tipo": "estado_salida", "valor": "justificacion_pendiente"})
    unicos, vistos = [], set()
    for p in fuera:
        k = json.dumps(p, sort_keys=True, default=str)
        if k not in vistos:
            vistos.add(k)
            unicos.append(p)
    return unicos


def _faltan(est: dict) -> list:
    """Los insumos que el propio taller sabe que faltan para decidir, de ESTE
    adelanto: la consulta provisional (la decisiva no llegó), los faltantes que
    señaló el análisis y las constancias indispensables de la global."""
    hu = _ws(est.get("huella"))
    fuera = []
    mat, con = est.get("material"), est.get("consulta")
    ce = mat.get("consulta_estado") if isinstance(mat, dict) else None
    if isinstance(ce, dict) and ce.get("estado") == "provisional" \
            and (con is None or _del_adelanto(con, hu)):
        for q in (_seq(ce.get("faltan")) or ["?"]):
            fuera.append({"origen": "consulta", "que": _ws(q)})
    an = est.get("analisis")
    if _del_adelanto(an, hu) and an.get("estado") == "listo" and isinstance(an.get("doc"), dict):
        for x in _seq(an["doc"].get("faltantes")):
            if isinstance(x, dict) and _ws(x.get("que")):
                fuera.append({"origen": "analisis", "que": _ws(x.get("que"))[:300]})
    g = _global_de(est, hu)
    for c in _seq((g or {}).get("constancias")):
        if isinstance(c, dict) and c.get("indispensable") and _ws(c.get("que")):
            fuera.append({"origen": "constancias", "que": _ws(c.get("que"))[:300]})
    return fuera


def _vacia() -> dict:
    return {"version": MANIFIESTO_VERSION, "estado": "", "desactualizado": False, "motivos": [],
            "por_marca": {}, "pendientes": [], "faltan": [], "en_curso": []}


def estado_sesion(fila_estado: dict, actuales: dict = None, ahora=None, *, plan=None,
                  version_oaj=None, llegada_desactualiza=None) -> dict:
    """En qué punto está el asunto, para UNA insignia.

    `fila_estado`: el `estado` de la fila, o la fila entera {estado, plan}.
    `plan`: la columna `plan`, si no viene en la fila. `actuales`: las huellas
    de ahora que main calcula con la sesión viva (`manifiesto(...)`); lo que
    trae MANDA sobre lo que se deduce de la fila (`actuales_de_fila`), y lo que
    falta no se compara. `ahora`: número o función; sin él, nada envejece.

    Devuelve {version, estado, desactualizado, motivos, por_marca, pendientes,
    faltan, en_curso}:
      · estado: en_analisis | faltan_insumos | justificacion_pendiente |
        borrador_sin_plan | proyecto_verificado, o «» si no hay nada que decir
        —ninguna marca lleva «consume» (lo de hoy, y con la bandera apagada)
        o el proyecto es anterior a esto—: no se pinta una insignia inventada.
      · desactualizado: la marca «proyecto» quedó desactualizada. Es un
        indicador aparte, no un estado: un proyecto verificado puede estarlo.
      · motivos: por qué, desde la causa (el insumo que cambió) hasta el
        arrastre. Tipos: cambio | llegada | retirada | arrastre.
      · por_marca: {marca: {estado, motivos}}, con estado ausente | en_curso |
        fallo | sin_registro | desactualizada | vigente.

    Pura: no escribe, no lanza por una fila rara, no mira banderas. No cambia
    ningún sentido ni bloquea nada: sólo dice."""
    est, pl = _partes(fila_estado, plan)
    lleg = LLEGADA_DESACTUALIZA if llegada_desactualiza is None else bool(llegada_desactualiza)
    cur = actuales_de_fila(est, pl, version_oaj)
    for k, v in (actuales or {}).items() if isinstance(actuales, dict) else ():
        cur[str(k)] = v
    proyecto = est.get("proyecto") if isinstance(est.get("proyecto"), dict) and est.get("proyecto") else None
    c_proy = _consume(proyecto)

    # QUÉ CASILLA DEL PLAN Y QUÉ RECALIFICACIÓN SON «LAS» DE ESTA SESIÓN: las
    # que usó el proyecto; sin proyecto, las de ahora. Un proyecto escrito sin
    # plan («») no se cuelga de una casilla ajena.
    claves = {}
    if "plan" in c_proy:
        claves["plan"] = _ws(c_proy["plan"])
    elif "plan" in cur:
        claves["plan"] = _ws(cur["plan"])
    else:
        claves["plan"] = _ws(pl.get("pedido_clave"))
    c_plan = _consume(_doc_marca("plan", est, pl, claves))
    for fuente in (c_proy, c_plan, cur):
        if "recalificacion" in fuente:
            claves["recalificacion"] = _ws(fuente["recalificacion"])
            break

    docs = {m: _doc_marca(m, est, pl, claves) for m in MARCAS}
    if not any(_consume(d) for d in docs.values()):
        return _vacia()

    por_marca = {}
    for m in orden():
        doc = docs[m]
        if doc is None:
            por_marca[m] = {"estado": "ausente", "motivos": []}
            continue
        crudo = _crudo(m, doc, ahora)
        if crudo != "listo":
            por_marca[m] = {"estado": crudo, "motivos": []}
            continue
        cons = _consume(doc)
        if not cons:
            por_marca[m] = {"estado": "sin_registro", "motivos": []}
            continue
        if cons.get("v") != MANIFIESTO_VERSION:
            # Otra forma de tomar las huellas: comparar sería comparar peras
            # con manzanas y lo daría todo por cambiado.
            por_marca[m] = {"estado": "sin_registro", "motivos": [
                {"marca": m, "tipo": "otra_version", "antes": cons.get("v"),
                 "ahora": MANIFIESTO_VERSION, "invalida": False}]}
            continue
        motivos = _comparar(m, cons, cur, lleg)
        motivos += [{"marca": m, "tipo": "arrastre", "de": d, "invalida": True}
                    for d in MARCAS[m]["depende_de"]
                    if por_marca.get(d, {}).get("estado") == "desactualizada"]
        por_marca[m] = {"estado": "desactualizada" if any(x["invalida"] for x in motivos)
                        else "vigente", "motivos": motivos}

    en_curso = [m for m in orden() if por_marca[m]["estado"] == "en_curso"]
    faltan = _faltan(est)
    fuera = {"version": MANIFIESTO_VERSION, "estado": "", "desactualizado": False, "motivos": [],
             "por_marca": por_marca, "pendientes": [], "faltan": faltan, "en_curso": en_curso}

    if proyecto is None:
        fuera["estado"] = "en_analisis" if (en_curso or not faltan) else "faltan_insumos"
        return fuera

    if por_marca["proyecto"]["estado"] == "sin_registro":
        # Un proyecto anterior a los estados: no se sabe con qué se hizo, así
        # que ni verificado ni desactualizado.
        fuera["motivos"] = [{"marca": "proyecto", "tipo": "anterior_a_estados", "invalida": False}]
        return fuera

    arriba = set(arrastran_a("proyecto")) | {"proyecto"}
    fuera["desactualizado"] = por_marca["proyecto"]["estado"] == "desactualizada"
    fuera["motivos"] = [x for m in orden() if m in arriba for x in por_marca[m]["motivos"]]
    fuera["pendientes"] = _pendientes(proyecto)
    fp = proyecto.get("plan") if isinstance(proyecto.get("plan"), dict) else {}
    # BORRADOR SIN PLAN SE DERIVA, NO SE GUARDA: la ficha ya dice si el plan se
    # usó. Una ficha sin plan ({}) es de la v1/v2, que no planean: no cuenta.
    if fp and fp.get("estado") != "usado":
        fuera["estado"] = "borrador_sin_plan"
    elif fuera["pendientes"]:
        fuera["estado"] = "justificacion_pendiente"
    elif faltan:
        fuera["estado"] = "faltan_insumos"
    else:
        fuera["estado"] = "proyecto_verificado"
    return fuera
