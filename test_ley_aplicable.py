# -*- coding: utf-8 -*-
"""La ley aplicable la decide la materia, no el lugar (28-sep-2026).

Parte 1, sin red: la materia de las SEIS consultas reales de la prueba de
anuncios (grabar58.py, escritas como las escribe quien no es abogado), los
controles negativos (burócrata, policía, impuesto sobre la renta, pensión del
IMSS, arrendamiento financiero, amparo), las reglas del prompt y que nada de
esto sea un localismo de Querétaro.

Parte 2, el código de main.py: que las dos frases que ENSEÑABAN el error ya no
están y que el prompt de redacción de los tres escalones lleva las reglas.

Parte 3, Qdrant REAL en sólo lectura (sólo si hay credenciales en el .env): que
la búsqueda de la vía trae, en Querétaro, el artículo que manda sumarios los
juicios de alimentos, y qué trae en otras entidades.

    python test_ley_aplicable.py
"""
import asyncio
import contextlib
import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ley_aplicable as la  # noqa: E402

FALLOS = []


def ok(cond, que):
    print(("   PASA   " if cond else "   FALLA  ") + que)
    if not cond:
        FALLOS.append(que)


# Las seis de grabar58.py, literales.
PRUEBA = {
    "pension": ("Vivo en Querétaro. El papá de mis dos hijos dejó de darme la pensión hace cinco meses "
                "y ya no me contesta. ¿Qué puedo hacer? Redáctame el escrito que tengo que presentar.", "familiar"),
    "despido": ("Me corrieron de mi trabajo en Querétaro después de cuatro años y no me quieren pagar "
                "nada. Ganaba 12 mil pesos al mes. ¿Qué me corresponde? Redáctame lo que tengo que presentar.",
                "laboral"),
    "renta": ("Voy a rentar mi casa en Querétaro. Redáctame un contrato de arrendamiento que me proteja "
              "si el inquilino deja de pagar o se va sin entregar la casa.", "civil"),
    "cobranza": ("Tengo un negocio pequeño en Querétaro. Un cliente me firmó un pagaré por 45 mil pesos, "
                 "se venció hace dos meses y no me paga. ¿Qué hago? Redáctame lo que tengo que presentar.",
                 "mercantil"),
    "rescision": ("Tengo una empresa en Querétaro. Un trabajador faltó cuatro días seguidos este mes sin "
                  "avisar ni justificar. Quiero terminar la relación laboral conforme a la ley. Redáctame "
                  "el aviso de rescisión.", "laboral"),
    "fraude": ("Me estafaron en Querétaro: le pagué 18 mil pesos a una persona por un auto que vendía en "
               "internet, nunca me lo entregó y ya no contesta. Redáctame la denuncia.", None),
}

print("\n1 · LA MATERIA DE LAS SEIS CONSULTAS REALES")
for clave, (texto, esperada) in PRUEBA.items():
    got = la.materia(texto)
    ok(got == esperada, f"{clave}: {got!r} (esperada {esperada!r}), sólo por el texto")
# El Estratega no cambia lo que el texto dice sin duda.
ok(la.materia(PRUEBA["cobranza"][0], estratega="civil") == "mercantil",
   "un pagaré es mercantil aunque el Estratega diga civil")
ok(la.materia(PRUEBA["renta"][0], estratega="mercantil") == "civil",
   "rentar una casa es civil aunque el Estratega diga mercantil")
ok(la.materia(PRUEBA["fraude"][0], estratega="penal") is None,
   "la estafa: el Estratega dice penal y el módulo calla (no es de sus cuatro)")

print("\n2 · CONTROLES NEGATIVOS: DONDE CALLAR ES LO CORRECTO")
casos_none = {
    "policía dado de baja": "Soy policía municipal y me dieron de baja sin procedimiento. ¿Qué hago?",
    "burócrata estatal": "Soy trabajadora del gobierno del estado y me despidieron. Redáctame la demanda.",
    "maestro del magisterio": "Soy maestro, del magisterio estatal, y me despidieron sin causa.",
    "amparo laboral": "Quiero promover amparo directo contra el laudo del tribunal laboral por mi despido.",
    "pagaré y renta a la vez": "El inquilino de mi casa me firmó pagarés por las rentas y no paga.",
}
for que, texto in casos_none.items():
    ok(la.materia(texto) is None, f"{que} → None ({la.materia(texto)!r})")
ok(la.materia("¿Procede el amparo indirecto contra la clausura de un establecimiento mercantil "
              "ordenada por una autoridad administrativa?", estratega="administrativo") is None,
   "amparo contra la clausura de un establecimiento mercantil → None (comparar.py)")
ok("mercantil" not in la.materias_en_texto("Me clausuraron mi establecimiento mercantil por no tener licencia"),
   "«establecimiento mercantil» no es materia mercantil")
ok(la.materia("Promoví amparo directo contra la sentencia del juicio ejecutivo mercantil") is None,
   "con amparo el módulo calla aunque el acto sea mercantil")
ok(la.materia("¿Cuánto tarda un juicio ejecutivo mercantil?") == "mercantil"
   and la.materia("Quiero demandar en la vía oral mercantil") == "mercantil",
   "el juicio y la vía mercantil sí son señal")
ok(la.materia("El inquilino de mi casa me firmó pagarés por las rentas y no paga.",
              estratega="mercantil") == "mercantil",
   "pagaré y renta a la vez: desempata el Estratega si nombra una de las dos")
ok("civil" not in la.materias_en_texto("¿Cómo presento mi declaración del impuesto sobre la renta?"),
   "«impuesto sobre la renta» no es arrendamiento")
ok("familiar" not in la.materias_en_texto("El IMSS me negó la pensión por invalidez."),
   "la pensión del IMSS no es alimentos")
ok(la.materias_en_texto("Firmé un arrendamiento financiero de un camión con una arrendadora financiera")
   == {"mercantil"}, "el arrendamiento financiero es mercantil y sólo mercantil")
ok(la.materia("cualquier cosa", elegida="laboral") == "laboral", "la materia elegida (Genio) manda")
ok(la.materia(PRUEBA["cobranza"][0], elegida="amparo") is None,
   "una materia elegida que no es de las cuatro deja al módulo callado")

print("\n3 · LA INSTRUCCIÓN DINÁMICA")
m_ins = la.instruccion("mercantil", "Queretaro")
l_ins = la.instruccion("laboral", "Queretaro")
ok(la.instruccion("civil", "Queretaro") is None and la.instruccion("familiar", "Queretaro") is None
   and la.instruccion(None, "Queretaro") is None, "sólo hay instrucción dinámica en materia federal")
ok(all(s in m_ins for s in ("Código de Comercio", "Ley General de Títulos y Operaciones de Crédito",
                            "Código Civil Federal (artículo 2)", "artículo 1054", "NO fundan nada")),
   "mercantil: Código de Comercio, LGTOC, CCF por el art. 2, supletoriedad del 1054, el código del estado no funda")
ok(all(s in l_ins for s in ("Ley Federal del Trabajo", "684-A", "685 Ter", "684-C", "872, apartado B",
                            "artículo 518", "(artículo 48)", "artículo 50", "(artículo 49)", "(artículo 52)",
                            "artículo 527")),
   "laboral: conciliación previa, sus excepciones, la solicitud, la constancia, 518, 48 vs 50/49/52, 527")
ok("no se suma" in l_ins and "veinte días" in l_ins, "laboral: los veinte días del 50 no se suman al 48")
ok("sin explicarlo ni mencionarlo" in m_ins and "sin explicarlo ni mencionarlo" in l_ins,
   "las dos piden no explicar el criterio al usuario")
y_ins = la.instruccion("mercantil", "Yucatan")
ok("Yucatan" in y_ins and "Quer" not in y_ins, "con otro estado, habla de ese estado y de ningún otro")
s_ins = la.instruccion("laboral", None)
ok(s_ins and "del estado" in s_ins and "None" not in s_ins, "sin entidad no se rompe ni imprime None")

ok("patrón es particular" in l_ins and "escrito de parte" in l_ins,
   "laboral: el patrón es particular salvo que se diga otra cosa, y se entrega escrito de parte")
ok("Cuando lo que se pide es reclamar" in l_ins and "aviso de rescisión que da el patrón" in l_ins,
   "laboral: la solicitud de conciliación sólo cuando se reclama; el aviso de rescisión del patrón "
   "se redacta tal cual (anuncio 5)")

print("\n3b · LA LEY AJENA FUERA DEL CONTEXTO")
aj = la.es_ley_ajena
q_renta = PRUEBA["renta"][0]
ok(aj("civil", "leyes_federales", "Código Civil Federal", q_renta)
   and aj("familiar", "leyes_federales", "Código Federal de Procedimientos Civiles", PRUEBA["pension"][0]),
   "civil/familiar del estado: fuera el CCF y el CFPC")
ok(not aj("familiar", "leyes_federales", "Código Nacional de Procedimientos Civiles y Familiares",
          PRUEBA["pension"][0]), "el Código Nacional se queda (rige donde se declaró su vigencia)")
ok(not aj("civil", "leyes_queretaro", "Código Civil del Estado de Querétaro", q_renta)
   and not aj("civil", "leyes_federales", "Ley General de los Derechos de Niñas, Niños y Adolescentes", q_renta),
   "civil: la ley del estado y las leyes generales se quedan")
ok(not aj("civil", "leyes_federales", "Código Civil Federal",
          "¿El Código Civil Federal regula distinto el arrendamiento que el de Querétaro?"),
   "si el usuario habla de lo federal, el CCF se queda")
ok(aj("mercantil", "leyes_queretaro", "Código de Procedimientos Civiles del Estado de Querétaro", "")
   and not aj("mercantil", "leyes_federales", "Código Civil Federal", "")
   and not aj("mercantil", "leyes_federales", "Código de Comercio", ""),
   "mercantil: fuera la ley del estado; el CCF (art. 2 CCom) y el Código de Comercio se quedan")
ok(aj("laboral", "leyes_queretaro", "Ley de los Trabajadores del Estado de Querétaro", "")
   and aj("laboral", "leyes_federales", "Ley Federal de los Trabajadores al Servicio del Estado", "")
   and aj("laboral", "leyes_federales", "Ley Reglamentaria de la Fracción XIII Bis del Apartado B, del Artículo 123", "")
   and not aj("laboral", "leyes_federales", "Ley Federal del Trabajo", "")
   and not aj("laboral", "leyes_federales", "Ley del Seguro Social", ""),
   "laboral: fuera la ley del estado y la del apartado B; la LFT y la LSS se quedan")
ok(not any(aj(m, s, "Código Civil Federal", "") for m in la.MATERIAS
           for s in ("jurisprudencia_nacional_v3", "bloque_constitucional", "sentencias_holdings")),
   "la jurisprudencia, la Constitución y las sentencias no se tocan nunca")
ok(not aj(None, "leyes_queretaro", "Código Civil del Estado de Querétaro", ""), "sin materia no se quita nada")

print("\n3c · EL CÓDIGO NACIONAL, ENTIDAD POR ENTIDAD")
qp = PRUEBA["pension"][0]
cn = "Código Nacional de Procedimientos Civiles y Familiares"
ok(la.vigencia_cnpcf("CDMX")["alcance"] == "plena" and la.vigencia_cnpcf("ciudad de méxico")["alcance"] == "plena"
   and la.vigencia_cnpcf("QUERETARO")["alcance"] == "parcial" and la.vigencia_cnpcf("Querétaro")
   and la.vigencia_cnpcf("JALISCO") is None and la.vigencia_cnpcf(None) is None,
   "la tabla: CDMX plena, Querétaro parcial, el resto sin anotar (regla general)")
ok(all(v.get("fuente") and v.get("nombre") for v in la.VIGENCIA_CNPCF.values()), "cada entrada lleva su fuente")
ok(la.es_ley_ajena("familiar", "leyes_federales", cn, qp, estado="QUERETARO"),
   "Querétaro (parcial): el Código Nacional sale del contexto de la pensión")
ok(not la.es_ley_ajena("familiar", "leyes_federales", cn, qp + " Vivo en San Juan del Río.", estado="QUERETARO")
   and not la.es_ley_ajena("familiar", "leyes_federales", cn, "¿Qué dice el Código Nacional sobre alimentos?",
                           estado="QUERETARO")
   and not la.es_ley_ajena("familiar", "leyes_federales", cn, "Quiero el juicio oral de alimentos",
                           estado="QUERETARO"),
   "…salvo que se hable de San Juan del Río, del Código Nacional o de la oralidad")
ok(not la.es_ley_ajena("familiar", "leyes_federales", cn, qp, estado="CDMX")
   and not la.es_ley_ajena("familiar", "leyes_federales", cn, qp, estado="JALISCO"),
   "CDMX (plena) y entidades sin anotar: el Código Nacional se queda")
ip_q = la.instruccion_procesal("familiar", "QUERETARO")
ip_c = la.instruccion_procesal("civil", "CDMX")
ok(ip_q and "San Juan del Río" in ip_q and "no cites el Código Nacional" in ip_q
   and ip_c and "15 de noviembre de 2025" in ip_c and "la Ciudad de México" in ip_c and "No mezcles" in ip_c,
   "la instrucción dice qué código rige en cada una")
ok(la.instruccion_procesal("familiar", "JALISCO") is None and la.instruccion_procesal("mercantil", "CDMX") is None,
   "sin anotar o en materia federal no hay instrucción procesal")
ok("usa el del estado" not in la.REGLAS_REDACCION and "el que rija para ese asunto en ese lugar" in la.REGLAS_REDACCION,
   "la regla general ya no elige el código del estado por omisión (en la CDMX sería el viejo)")

print("\n4 · LAS REGLAS DEL PROMPT DE REDACCIÓN")
R = la.REGLAS_REDACCION
ok("LA DECIDE LA MATERIA, NO EL LUGAR" in R, "el apartado existe")
ok("Quer" not in R and "Querétaro" not in la.instruccion("laboral", "Jalisco"),
   "ningún localismo: Querétaro no aparece en las reglas generales")
ok(not re.search(r"\bej(emplo)?[:.]|\bp\. ?ej\b|por ejemplo", R, re.I),
   "sin ejemplos que se puedan copiar literales (feedback_ejemplo_en_prompt)")
ok("no son supletorios de los códigos de un estado" in R, "el CCF y el CFPC no suplen a un código estatal")
ok("nunca es supletorio de él" in R and "en un solo código" in R and "no mezcles artículos de los dos" in R,
   "el Código Nacional no es supletorio y no se mezclan dos códigos procesales")
ok("La vía ordinaria es residual" in R, "la vía ordinaria es residual")
ok("el patrón es particular y rige la Ley Federal del Trabajo" in R
   and "las leyes burocráticas sólo rigen a quien trabaja para el Estado" in R,
   "por omisión el patrón es particular; la ley burocrática sólo si trabaja para el Estado")

print("\n5 · EL ARTÍCULO DE LA VÍA")
ok(la.conceptos_de_via(PRUEBA["pension"][0], "familiar") == ["alimentos"], "pensión → alimentos")
ok(la.conceptos_de_via(PRUEBA["renta"][0], "civil") == ["arrendamiento"], "renta → arrendamiento")
ok(la.conceptos_de_via(PRUEBA["cobranza"][0], "mercantil") == [], "en materia federal no se busca vía local")
ok(la.consulta_de_via([]) is None, "sin conceptos no hay consulta")
cv = la.consulta_de_via(["alimentos"])
ok("sumari" in cv and "especial" in cv and "controversia del orden familiar" in cv and "alimentos" in cv,
   "la consulta usa el vocabulario de los códigos")
ok(la.consulta_de_via(["divorcio", "guarda y custodia"]).count(" y ") >= 1, "dos conceptos se enlazan con «y»")

print("\n6 · EL CÓDIGO DE main.py")
fuente = Path(__file__).resolve().parent.joinpath("main.py").read_text(encoding="utf-8")
ok('f"3. Las leyes federales (Código Civil Federal, etc.) son SUPLETORIAS' not in fuente,
   "ya no se inyecta «Las leyes federales (Código Civil Federal, etc.) son SUPLETORIAS»")
ok("me apoyo en el Código Civil Federal\ncomo supletorio" not in fuente,
   "el prompt de consulta ya no enseña el CCF «como supletorio» de un código estatal")
ok("entrelazando legislación federal, estatal, jurisprudencia" not in fuente,
   "el prompt de redacción ya no pide mezclar federal y estatal en 5-8 fuentes")
ok("_REDACCION_NUCLEO + ley_aplicable.REGLAS_REDACCION + _REDACCION_REGISTROS" in fuente,
   "las reglas van entre el núcleo y los registros")
ok('_plan_estratega["materia_ley"] = _materia_ley' in fuente and 'ley_aplicable.instruccion(' in fuente,
   "la materia llega del recuperador al prompt por el buzón del Estratega")
ok('VIA_PROCESAL_ACTIVA' in fuente, "la búsqueda de la vía se apaga sin desplegar")
ok('LEY_AJENA_FUERA' in fuente and 'ley_aplicable.es_ley_ajena(' in fuente,
   "el filtro de ley ajena existe y se apaga sin desplegar")
ok('_plan_estratega["estado_ley"] = effective_estado' in fuente
   and '_ent_ley = _plan_estratega.get("estado_ley") or _estado_for_llm' in fuente,
   "el estado de la pregunta llega a la instrucción aunque el selector esté vacío")
_i = fuente.index("_instr_ley = None")
ok(_i < fuente.index("if _estado_for_llm:\n                    estado_humano"),
   "la instrucción por materia se calcula FUERA del bloque que sólo corre con el selector")

with contextlib.redirect_stdout(io.StringIO()):
    import main
ok(la.REGLAS_REDACCION in main.SYSTEM_PROMPT_CHAT_DRAFTING, "el prompt de los tres escalones lleva las reglas")
ok(main.SYSTEM_PROMPT_CHAT_DRAFTING.index("LEY APLICABLE") < main.SYSTEM_PROMPT_CHAT_DRAFTING.index("ELIGE EL REGISTRO"),
   "y van antes del despachador de registros")
ok("como supletorio" not in main.INVENTORY_CONTEXT, "INVENTORY_CONTEXT ya no dice «como supletorio»")

# La trampa medida el 28-sep: los artículos de leyes_federales llegan al filtro
# SIN `origen` (se deduce del texto al armar el contexto). El filtro tiene que
# rellenarlo antes de mirar, o no ve el Código Civil Federal.
_sr = main.SearchResult
_docs = [
    _sr(id="a", score=.5, silo="leyes_federales", origen=None,
        texto="[MATERIA: civil] [Código Civil Federal | TITULO C | CAPITULO III]\nArtículo 2442.- ..."),
    _sr(id="b", score=.5, silo="leyes_federales", origen=None,
        texto="[MATERIA: laboral] [Ley Federal de los Trabajadores al Servicio del Estado, Reglamentaria "
              "del Apartado B) del Artículo 123 Constitucional | TITULO]\nArtículo 46.- ..."),
    _sr(id="c", score=.5, silo="leyes_queretaro", origen="Código Civil del Estado de Querétaro",
        texto="Artículo 2320. ..."),
]
ok(not la.es_ley_ajena("civil", "leyes_federales", _docs[0].origen, PRUEBA["renta"][0]),
   "sin rellenar, el filtro NO ve el CCF (la trampa existe)")
main.enrich_missing_metadata(_docs)
ok(la.es_ley_ajena("civil", "leyes_federales", _docs[0].origen, PRUEBA["renta"][0])
   and la.es_ley_ajena("laboral", "leyes_federales", _docs[1].origen, PRUEBA["despido"][0])
   and not la.es_ley_ajena("civil", "leyes_queretaro", _docs[2].origen, PRUEBA["renta"][0]),
   f"con enrich_missing_metadata sí: {_docs[0].origen!r} y {(_docs[1].origen or '')[:52]!r} fuera")
_pos_enrich = fuente.index("enrich_missing_metadata(search_results)")
_pos_filtro = fuente.index("ley_aplicable.es_ley_ajena(")
ok(_pos_enrich < _pos_filtro, "main.py rellena el origen ANTES de filtrar")

print("\n7 · QDRANT REAL, SÓLO LECTURA: EL ARTÍCULO DE LA VÍA")
_BUCLE = asyncio.new_event_loop()
try:
    e = {}
    for l in Path("/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/jurexia-api-git/.env").read_text().splitlines():
        if "=" in l and not l.startswith("#"):
            k, v = l.split("=", 1)
            e[k.strip()] = v.strip().strip('"')
    main.qdrant_client = __import__("qdrant_client").AsyncQdrantClient(
        url=e["QDRANT_URL"], api_key=e["QDRANT_API_KEY"], timeout=60)
    # El cliente de embeddings nace en el arranque del servidor (lifespan);
    # importado a secas vale None y cada embedding falla con AttributeError.
    main.openai_client = __import__("openai").AsyncOpenAI(api_key=e["OPENAI_API_KEY"])
    vivo = True
except Exception as ex:
    vivo = False
    print(f"   (sin Qdrant: {type(ex).__name__}; se salta la parte real)")

if vivo:
    def via(conceptos, estado):
        with contextlib.redirect_stdout(io.StringIO()):
            return _BUCLE.run_until_complete(main._articulos_de_la_via(la.consulta_de_via(conceptos), estado))

    r = via(["alimentos"], "QUERETARO")
    textos = [(x.ref, x.origen, " ".join((x.texto or "").split())[:140]) for x in r]
    ok(any("sumariamente" in (x.texto or "") and "alimentos" in (x.texto or "") for x in r),
       f"Querétaro, alimentos: trae el artículo que los manda sumarios → {[t[0] for t in textos]}")
    ok(all("Procedimientos Civiles" in (x.origen or "") for x in r), "y sólo del código procesal")
    r = via(["arrendamiento"], "QUERETARO")
    ok(any("arrendamiento" in (x.texto or "").lower() for x in r),
       f"Querétaro, arrendamiento → {[x.ref for x in r]}")
    # Calibración, no umbral: qué trae en entidades con códigos distintos.
    for est in ("CDMX", "JALISCO", "NUEVO_LEON", "YUCATAN", "SONORA"):
        r = via(["alimentos"], est)
        print(f"   ·  {est}: {[(x.ref, (x.origen or '')[:48]) for x in r]}")
        for x in r[:1]:
            print(f"        «{' '.join((x.texto or '').split())[:170]}»")
    ok(via(["alimentos"], None) == [], "sin entidad no se busca nada")
    r = via(["alimentos"], "CDMX")
    ok(r and all("Nacional" in (x.origen or "") for x in r),
       f"CDMX (plena): la vía sale del Código Nacional → {[(x.ref, (x.origen or '')[:40]) for x in r]}")
    for x in r[:2]:
        print(f"        «{' '.join((x.texto or '').split())[:170]}»")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}:")
    for f_ in FALLOS:
        print("  ·", f_)
    sys.exit(1)
print("TODO PASA")
