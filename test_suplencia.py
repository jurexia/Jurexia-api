"""La suplencia de la queja como paso de la pantalla de decisión — 26-sep-2026.

Decisión 4 de David: el motor propone la fracción del artículo 79 y a favor de
quién; el secretario confirma. Con la suplencia confirmada se prohíben las
inoperancias de forma a favor de esa parte, y la suplencia sólo se expresa en
la sentencia cuando deriva un beneficio.

    .venv/bin/python test_suplencia.py

El texto del 79 se cotejó a mano con el PDF oficial (DOF 16-10-2025); aquí se
coteja el catálogo contra `normas_ley_de_amparo.json`, que dice lo mismo.
La sección 8 corre sobre el banco Kingston si el corpus está en el disco.
"""
import glob
import json
import os
import re
import unicodedata

import fase6_estudio as f6
import suplencia as sp

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


print("\n1 · EL CATÁLOGO DICE LO QUE DICE LA LEY")
_art = json.load(open("normas_ley_de_amparo.json", encoding="utf-8"))["articulos"]
A79 = " ".join(str(_art["79"]).split())
A171 = " ".join(str(_art["171"]).split())
_CLAVES_LEY = {
    "I": ("normas generales", "declaradas inconstitucionales", "plenos regionales"),
    "II": ("menores de edad o incapaces", "orden y desarrollo de la familia"),
    "III-a": ("persona inculpada o sentenciada",),
    "III-b": ("ofendida o víctima", "quejosa o adherente"),
    "IV-a": ("fracción III del artículo 17",),
    "IV-b": ("ejidatarias y comuneras en particular", "bienes o derechos agrarios"),
    "V": ("persona trabajadora", "derecho administrativo"),
    "VI": ("violación evidente", "sin defensa", "controversia en el amparo"),
    "VII": ("pobreza o marginación", "clara desventaja social para su defensa en el juicio"),
}
for f in sp.FRACCIONES:
    frases = _CLAVES_LEY[f["id"]]
    ok(all(x in A79 for x in frases) and all(x in f["texto"] for x in frases),
       f"fracción {f['id']}: sus palabras están en la ley y en el catálogo")
ok("fracciones I, II, III, IV, V y VII" in A79
   and [f["id"] for f in sp.FRACCIONES if not f["sin_conceptos"]] == ["VI"],
   "sólo la VI exige conceptos, y sólo en las demás se expresa «cuando derive de un beneficio»")
ok("solo se expresará en las sentencias cuando la suplencia derive de un beneficio" in A79,
   "la regla de expresión está en la ley tal como la cita el bloque")
ok("violaciones procesales o formales sólo podrá operar" in A79,
   "y el último párrafo, el de las violaciones procesales o formales")
for f in sp.FRACCIONES:
    if f["exime_171"]:
        ok(any(x in A171 for x in {"II": ("menores de edad",), "III-a": ("persona inculpada",),
                                    "IV-a": ("núcleos de población",), "IV-b": ("ejidatarios",),
                                    "V": ("trabajadores",), "VII": ("pobreza o marginación",)}[f["id"]]),
           f"fracción {f['id']}: su sujeto está en el segundo párrafo del 171")

print("\n2 · LO QUE LLEGA DEL FORMULARIO")
ok(sp.normalizar_fraccion("V") == "V" and sp.normalizar_fraccion("79-VII") == "VII",
   "claves simples y con el artículo delante")
ok(sp.normalizar_fraccion("fracción IV, inciso b)") == "IV-b" and sp.normalizar_fraccion("IIIb") == "III-b",
   "fracción con inciso, escrita de dos maneras")
ok(sp.normalizar_fraccion("III") == "" and sp.normalizar_fraccion("VIII") == "",
   "la III sin inciso y una fracción que no existe no se inventan")
ok(sp.normalizar_fraccion("sin suplencia") == sp.NINGUNA, "«sin suplencia»")
L = sp.leer('{"fraccion":"V","a_favor_de":"la parte quejosa (Ana)","confirmada":true}')
ok(L == {"fraccion": "V", "a_favor_de": "la parte quejosa (Ana)", "confirmada": True},
   "JSON confirmado")
ok(sp.leer('{"fraccion":"V","confirmada":false}')["confirmada"] is False, "sin confirmar")
ok(sp.leer("") == {} and sp.leer("{roto") == {} and sp.leer('{"fraccion":"XX"}') == {},
   "vacío, roto o inventado = como si no llegara nada")
ok(sp.leer({"fraccion": "ninguna", "a_favor_de": "x", "confirmada": "1"})
   == {"fraccion": "ninguna", "a_favor_de": "", "confirmada": True},
   "«sin suplencia» confirmada no lleva a favor de nadie")
ok(sp.confirmada(L) and not sp.confirmada(sp.leer('{"fraccion":"ninguna","confirmada":true}'))
   and not sp.confirmada(sp.leer('{"fraccion":"V"}')),
   "sólo cuenta como confirmada una fracción real que él confirmó")

print("\n3 · LA PROPUESTA DEL MOTOR")
p = sp.proponer("laboral", "amparo_directo", "Demanda del trabajador despedido.", "",
                quejoso="Ana Pérez López", actor_origen="Ana Pérez López",
                demandado_origen="Servicios del Bajío, S.A. de C.V.")
ok(p["fraccion"] == "V" and "Ana Pérez López" in p["a_favor_de"] and "actora" in p["porque"],
   "laboral, promueve la actora del juicio de origen → V, a su favor")
p = sp.proponer("laboral", "amparo_directo", "", "", quejoso="Servicios del Bajío, S.A. de C.V.")
ok(p["fraccion"] == sp.NINGUNA and any(a["fraccion"] == "V" for a in p["alternativas"]),
   "laboral, promueve la empresa → sin suplencia (la V a la vista, explicada)")
p = sp.proponer("laboral", "amparo_directo", "", "", quejoso="Juan Ruiz",
                demandado_origen="Juan Ruiz")
ok(p["fraccion"] == sp.NINGUNA, "laboral, promueve el demandado del juicio de origen → sin suplencia")
p = sp.proponer("administrativa", "amparo_directo",
                "Se me negó la pensión por jubilación que otorga el ISSSTE.", "",
                quejoso="María Gómez")
ok(p["fraccion"] == "V", "administrativa, pensión del ISSSTE de una persona → V")
p = sp.proponer("administrativa", "amparo_directo",
                "Cuotas del IMSS determinadas a mi representada.", "",
                quejoso="Constructora Norte, S.A. de C.V.")
ok(p["fraccion"] == sp.NINGUNA, "administrativa, la empresa discute cuotas del IMSS → sin suplencia")
p = sp.proponer("administrativa", "amparo_directo",
                "SUSPENSIÓN DEL ACTO RECLAMADO. Solicito la suspensión.", "", quejoso="Luis Mora")
ok(p["fraccion"] == sp.NINGUNA, "«suspensión» no es una pensión (el defecto del \\b)")
p = sp.proponer("penal", "amparo_directo", "El sentenciado reclama la condena.", "",
                quejoso="Pedro Sánchez")
ok(p["fraccion"] == "III-a" and any(a["fraccion"] == "III-b" for a in p["alternativas"]),
   "penal → III, inciso a), con el b) a la vista")
p = sp.proponer("penal", "amparo_directo",
                "Promuevo en mi carácter de víctima contra el sobreseimiento.", "",
                quejoso="Rosa Díaz")
ok(p["fraccion"] == "III-b", "penal y promueve la víctima → III, inciso b)")
p = sp.proponer("civil", "amparo_directo",
                "Se discute la guarda y custodia y la pensión alimenticia de mis hijos.", "",
                quejoso="Laura Soto")
ok(p["fraccion"] == "II", "civil con custodia y alimentos → II (el formulario no ofrece «familiar»)")
p = sp.proponer("civil", "amparo_directo",
                "Tesis: las acciones del estado civil (matrimonio, divorcio, filiación, "
                "tutela, adopción) y los menores de edad.", "", quejoso="Esteban Guzmán")
ok(p["fraccion"] == sp.NINGUNA and any(a["fraccion"] == "II" for a in p["alternativas"]),
   "una lista doctrinal con «filiación» y «adopción» no es un asunto familiar (ADC 120/2026)")
p = sp.proponer("administrativa", "amparo_directo",
                "Sentencia del Tribunal Unitario Agrario sobre mis derechos como ejidataria "
                "de la parcela ejidal.", "", quejoso="Rosa Rangel")
ok(p["fraccion"] == "IV-b", "agrario que llega como administrativa, promueve una ejidataria → IV, b)")
p = sp.proponer("administrativa", "amparo_directo",
                "El Tribunal Unitario Agrario privó al ejido de sus bienes; Ley Agraria.", "",
                quejoso="Ejido San Isidro")
ok(p["fraccion"] == "IV-a", "promueve el ejido → IV, a)")

print("\n3-bis · LA FRACCIÓN VII")
p = sp.proponer("civil", "amparo_directo",
                "Vivo en condiciones de pobreza extrema y marginación, sin abogado.", "",
                quejoso="Tomás Luna")
ok(p["fraccion"] == "VII" and "acredita" in p["porque"],
   "civil y la parte afirma pobreza y marginación → VII, sólo si el expediente lo acredita")
p = sp.proponer("civil", "amparo_directo",
                "El suscrito no sabe leer ni escribir y es campesino.", "", quejoso="Tomás Luna")
ok(p["fraccion"] == sp.NINGUNA and any(a["fraccion"] == "VII" for a in p["alternativas"]),
   "indicios sin pobreza dicha → la VII a la vista, no propuesta")
p = sp.proponer("civil", "amparo_directo",
                "La actora es adulta mayor con discapacidad; reclusión del deudor.", "",
                quejoso="Tomás Luna")
ok(p["fraccion"] == sp.NINGUNA and not any(a["fraccion"] == "VII" for a in p["alternativas"]),
   "edad, discapacidad o reclusión no son pobreza ni marginación (medido en el banco)")
p = sp.proponer("laboral", "amparo_directo", "Soy trabajadora en pobreza.", "",
                quejoso="Eva Cruz", actor_origen="Eva Cruz")
ok(p["fraccion"] == "V" and any(a["fraccion"] == "VII" for a in p["alternativas"]),
   "con la V ya propuesta, la VII queda como alternativa")

print("\n3-ter · LO QUE LA PARTE PIDE Y DÓNDE NO HAY SUPLENCIA")
p = sp.proponer("civil", "amparo_directo",
                "SUPLENCIA DE LA QUEJA. Con fundamento en el artículo 79, fracción IV. inciso "
                "b) de la Ley de Amparo, solicito se suplan los planteamientos; soy campesino.",
                "", quejoso="Francisco Gutiérrez")
ok(p["pedida"] == "IV-b" and {a["fraccion"] for a in p["alternativas"]} >= {"IV-b", "VII"},
   "el 642/2024: pide la IV b) y alega ser campesino → las dos a la vista")
p = sp.proponer("administrativa", "amparo_directo",
                "artículos 78 y 79 de la Ley de Responsabilidades, fracción V; aplicado de "
                "manera supletoria el Código", "", quejoso="Gilberto Olvera")
ok(p["pedida"] == "", "un «79» de otra ley junto a «supletoria» no es una petición (ADA 61/2026)")
p = sp.proponer("administrativa", "revision_fiscal", "pensión del ISSSTE", "",
                quejoso="Titular de la Unidad Jurídica del ISSSTE")
ok(p["fraccion"] == sp.NINGUNA and "juicio de amparo" in p["porque"],
   "revisión fiscal → sin suplencia: no es juicio de amparo")
p = sp.proponer("laboral", "amparo_revision", "", "", quejoso="Ana Pérez",
                recurrente="Director General del Instituto de Salud")
ok(p["fraccion"] == "V" and p["a_favor_de"].startswith("la parte quejosa")
   and "Recurre la autoridad" in p["porque"],
   "revisión que interpone la autoridad → favorece a la quejosa, no a quien recurre")
ok(len(p["fracciones"]) == len(sp.FRACCIONES) + 1 and p["fracciones"][-1]["id"] == sp.NINGUNA,
   "la propuesta trae el catálogo entero para el selector")

print("\n4 · EL AVISO DEL ESTUDIO MIRA LA VII (defecto L4)")
INO = [f6.Criterio("¿Procede?", "inoperante")]
FUN = [f6.Criterio("¿Procede?", "fundado")]
av = f6._aviso_de_suplencia(INO, "civil", "la quejosa vive en pobreza", "amparo_directo")
ok(bool(av) and any("fracción VII" in x for x in av),
   "civil + inoperante + indicios de pobreza → recuerda la VII (antes: nada)")
ok(f6._aviso_de_suplencia(INO, "civil", "contrato de compraventa", "amparo_directo") == [],
   "civil sin indicios → nada, como antes")
ok(f6._aviso_de_suplencia(FUN, "civil", "pobreza", "amparo_directo") == [],
   "sin inoperancia no hay nada que salvar")
ok(any("79, fracción V" in x for x in f6._aviso_de_suplencia(INO, "laboral", "", "amparo_directo")),
   "laboral sigue igual")
ok(f6._aviso_de_suplencia(INO, "civil", "pobreza", "revision_fiscal") == [],
   "revisión fiscal: nunca")
ok(not f6._RX_TRABAJADOR_EN_ADMINISTRATIVA.search("SUSPENSIÓN DEL ACTO RECLAMADO")
   and f6._RX_TRABAJADOR_EN_ADMINISTRATIVA.search("la pensión del ISSSTE"),
   "«pensión» ya no se encuentra dentro de «suspensión»")

print("\n5 · EL BLOQUE DEL PROMPT")
S_V = {"fraccion": "V", "a_favor_de": "la parte quejosa (Ana)", "confirmada": True}
b = sp.bloque(S_V, "amparo_directo")
ok("fracción V" in b and "la parte quejosa (Ana)" in b, "nombra la fracción y a favor de quién")
ok("NINGUNA INOPERANCIA DE FORMA" in b and "no combatir la razón toral" in b
   and "ESTÚDIALO EN EL FONDO" in b, "prohíbe la inoperancia de forma: se suple y se estudia")
ok("sólo se exprese en" in b and "derive un beneficio" in b, "sólo se expresa si deriva un beneficio")
ok("171" in b and "último párrafo" in b, "exime de preparar la violación (171) y recoge el último párrafo")
ok("EL SENTIDO SIGUE SIENDO EL DEL SECRETARIO" in b and "ADVERTENCIAS" in b,
   "no toca el sentido: la discrepancia va a ADVERTENCIAS")
ok("«" not in b, "sin frases entre comillas que copiar (un ejemplo en el prompt se firma literal)")
b3 = sp.bloque({"fraccion": "III-b", "a_favor_de": "x", "confirmada": True}, "amparo_directo")
ok("171" not in b3, "la víctima (III b) no está en el 171: no se le promete la exención")
b6 = sp.bloque({"fraccion": "VI", "a_favor_de": "x", "confirmada": True}, "amparo_directo")
ok("NO ES ABSOLUTA" in b6 and "violación evidente" in b6 and "derive un beneficio" not in b6,
   "la VI: sólo ante violación evidente, sin la regla de expresión de las otras")
bq = sp.bloque(dict(S_V), "queja")
ok("agravios" in bq and "171" not in bq, "en un recurso: «agravios» y sin el 171 del amparo directo")
ok(sp.bloque({"fraccion": "V", "confirmada": False}) == ""
   and sp.bloque({"fraccion": "ninguna", "confirmada": True}) == "" and sp.bloque({}) == "",
   "sin confirmar, «sin suplencia» o nada → ningún bloque")

print("\n6 · EL CAMINO: FORMULARIO → ENCARGO → MATERIAL → PROMPT")
import fases123_pipeline as f123
import redactor_adelanto as ra


class _R:
    pass


r = _R()
r.fases = f123.Fases123()
r.encargo = ra.Encargo(numero="1/2026", encabezado="", quejoso="Ana", magistrado="", secretario="",
                       notificacion=None, presentacion=None, regla_surtimiento="", plazo=15,
                       plantilla="")
ok(r.encargo.suplencia == {}, "el encargo nace sin suplencia")
mat = f6.Material(tipo_asunto="amparo_directo")
r.encargo.suplencia = sp.leer(json.dumps(S_V))
ra._formato_al_material(r, mat)
ok(mat.suplencia == S_V, "la suplencia del encargo llega al material")
C = [f6.Criterio("¿La Sala valoró la prueba?", "infundado")]
p_con = f6.prompt_estudio("ACTO", "CONCEPTOS", C, mat)
ok("SUPLENCIA DE LA QUEJA — LA CONFIRMÓ EL SECRETARIO" in p_con, "confirmada: el bloque entra al prompt")
r.encargo.suplencia = sp.leer("")
ra._formato_al_material(r, mat)
ok(mat.suplencia == {}, "la siguiente petición sin suplencia la borra del material (vive en la sesión)")
p_sin = f6.prompt_estudio("ACTO", "CONCEPTOS", C, mat)
r.encargo.suplencia = sp.leer('{"fraccion":"V","a_favor_de":"x","confirmada":false}')
ra._formato_al_material(r, mat)
p_prop = f6.prompt_estudio("ACTO", "CONCEPTOS", C, mat)
ok("LA CONFIRMÓ EL SECRETARIO" not in p_sin and p_sin == p_prop,
   "sin confirmar, el prompt es idéntico al de antes: comportamiento actual")
ok(p_con.replace(sp.bloque(S_V, "amparo_directo"), "") == p_sin,
   "la única diferencia es el bloque: no toca el resto del prompt")

print("\n7 · LOS GEMELOS Y LA PANTALLA ESTÁN CONECTADOS")
src = open("main.py", encoding="utf-8").read()
ok(src.count('suplencia: str = Form(""),') == 2, "el campo en los DOS endpoints de resolver")
ok(src.count("r.encargo.suplencia = _sp_m.leer(suplencia)") == 2,
   "y en los dos se asigna SIEMPRE, no sólo si llega")
_ini = src.index('@app.post("/taller/resolver/stream")')
_ini2 = src.index('@app.post("/taller/resolver")')
ok(src.index("r.encargo.suplencia = _sp_m.leer(suplencia)", _ini) < _ini2
   and src.index("r.encargo.suplencia = _sp_m.leer(suplencia)", _ini2) > _ini2,
   "una asignación en cada gemelo")
ok('"suplencia": _taller_suplencia_propuesta(r),' in src,
   "la propuesta sale en /taller/contexto-del-asunto, que la pantalla ya pide")
ok('"suplencia": dict(getattr(getattr(res, "encargo", None), "suplencia", None) or {}),' in src,
   "y la ficha del proyecto guarda con qué suplencia salió")
src_ra = open("redactor_adelanto.py", encoding="utf-8").read()
ok(src_ra.count("_litis_y_material(r, material,") >= 2,
   "los dos redactores pasan por `_litis_y_material` → `_formato_al_material`")

print("\n8 · CALIBRACIÓN CONTRA EL BANCO KINGSTON (engroses reales)")
BANCO = "/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/redactor-sentencias/corpus/casos/"
if not os.path.isdir(BANCO):
    print("   (el corpus no está en este disco: se salta)")
else:
    vistos, props = set(), {}
    for fp in sorted(glob.glob(BANCO + "*.json")):
        if os.path.basename(fp).startswith("_"):
            continue
        d = json.load(open(fp, encoding="utf-8"))
        asunto = unicodedata.normalize("NFC", d["asunto"])
        if asunto in vistos:
            continue
        vistos.add(asunto)
        pz = d.get("piezas") or {}
        dem = (pz.get("demanda") or {}).get("texto", "") or ""
        eng = (pz.get("engrose") or {}).get("texto", "") or d.get("oro", "")
        q = re.search(r"QUEJOS[OA]S?:\s*(.+)", eng)
        props[asunto] = sp.proponer(
            "administrativa" if asunto.startswith("ADA") else "civil", "amparo_directo",
            dem, "", quejoso=q.group(1).strip().rstrip(".") if q else "")["fraccion"]
    AGRARIOS = {"ADA 103-2025", "ADA 263-2025 AGRARIO", "ADA 448-2025",
                "ADA 704-2022 AGRARIO PRESCRIPCIÓN", "ADA 767-2025"}
    FAMILIA = {"ADC 282-2025", "ADC 284-2026 (llego por incompetencia de juzgado)",
               "ADC 296-2025 SUMARIO CON MENORES (NIÑOS)", "ADC 560-2025",
               "ADC 640-2024 RECONOCIMIENTO DE PATERNIDAD"}
    AGRARIOS = {unicodedata.normalize("NFC", a) for a in AGRARIOS}
    FAMILIA = {unicodedata.normalize("NFC", a) for a in FAMILIA}
    ok(all(props.get(a) == "IV-b" for a in AGRARIOS),
       f"los {len(AGRARIOS)} amparos del Tribunal Unitario Agrario → IV, b)")
    ok(all(props.get(a) == "II" for a in FAMILIA),
       f"los {len(FAMILIA)} de menores o familia (custodia, alimentos, paternidad, un menor quejoso) → II")
    resto = {a: f for a, f in props.items() if a not in AGRARIOS | FAMILIA}
    malos = {a: f for a, f in resto.items() if f != sp.NINGUNA}
    ok(not malos, f"los otros {len(resto)} (civiles, mercantiles, fiscales, empresas) → sin "
                  f"suplencia; propuestas de más: {malos or 'ninguna'}")
    print(f"   {len(props)} asuntos: {sum(1 for f in props.values() if f != sp.NINGUNA)} con "
          f"propuesta, {sum(1 for f in props.values() if f == sp.NINGUNA)} sin suplencia")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
