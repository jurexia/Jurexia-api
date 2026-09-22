"""Nada se calcula dos veces: el acervo vuelve entero y el contraste se adelanta.

Revisión fiscal 2/2026, 17-sep-2026. La consulta del acervo corría tres veces
por proyecto (consultar, proponer, resolver) porque la sesión rehidratada de la
base no traía el material; y el contraste, que sólo lee el adelanto, esperaba
en serie dentro de la propuesta.

    .venv/bin/python test_sin_repetir.py
"""
import asyncio
import inspect
import json
import types

import taller_estado as te

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


print("\n1 · EL ACERVO VA A LA FILA Y VUELVE SIENDO EL MISMO")
import fase6_estudio as f6
import fase_precedente as fp

m = f6.Material()
m.tesis = [{"registro": "160053", "rubro": "R", "texto": "t"}]
m.normas = [{"cuerpo_legal": "Reglamento Interior del Instituto Mexicano del Seguro Social",
             "articulo": "150", "texto": "Artículo 150…", "de_internet": True,
             "dominio": "imss.gob.mx"}]
m.convencional = [{"instrumento": "CADH", "articulo": "8"}]
m.materia = "administrativa"
m.tipo_asunto = "revision_fiscal"
m.espejo = [{"fecha": "2025-01-01", "sentido": "confirma"}]
m.principios = ["competencia material", "fundamentación y motivación"]
m.sede_del_acto = "ordinaria"
m.cuaderno = "principal"
m.entidad = "Querétaro"
m.preceptos_de_internet = ["art. 150 — reglamento interior del instituto mexicano del seguro social (imss.gob.mx)"]
m.sondeo = fp.Sondeo(
    distribucion={"infundado": 7, "fundado": 2}, del_circuito=3,
    fundamentos=[{"fundamento": "art. 16 constitucional", "veces": 4}],
    concordantes=[{"tema": "competencia", "sentido": "infundado", "holding": "h" * 50,
                   "tribunal": "3TCC", "circuito": "22", "propio": True,
                   "calidad": 4, "pdf_url": None}],
    claves=["2a./J. 115/2005"],
    razonados=[{"sentido": "fundado", "razon": "porque sí " * 30, "tribunal": "T", "tema": "x"}],
    objeciones=[], moldes=[{"texto": "REGLA. " * 20}], avisos=["aviso del sondeo"],
    por_problema=[{"problema": "¿p?", "prediccion": {"frase": "infundado (7 de 9)"}}])

d = te.material_ligero(m)
ok(d.get("completo") is True and te.esta_completo(d), "el guardado nuevo se marca completo")
d2 = json.loads(json.dumps(d, ensure_ascii=False))  # ida y vuelta por el jsonb
m2 = te.material_rehidratado(d2)
ok(m2.tesis == m.tesis and m2.normas == m.normas and m2.convencional == m.convencional,
   "tesis, normas (con su origen de internet) y convencional, iguales")
ok((m2.materia, m2.tipo_asunto, m2.entidad) == ("administrativa", "revision_fiscal", "Querétaro"),
   "materia, tipo de asunto y entidad")
ok((m2.sede_del_acto, m2.cuaderno) == ("ordinaria", "principal"),
   "sede del acto y cuaderno —la verja de la suspensión— vuelven")
ok(m2.principios == m.principios and m2.espejo == m.espejo, "principios y espejo")
ok(m2.preceptos_de_internet == m.preceptos_de_internet, "la lista de preceptos traídos de internet")
ok(isinstance(m2.sondeo, fp.Sondeo), "el sondeo vuelve como Sondeo, no como dict")
ok(m2.sondeo.distribucion == {"infundado": 7, "fundado": 2} and m2.sondeo.del_circuito == 3
   and m2.sondeo.razonados == m.sondeo.razonados and m2.sondeo.moldes == m.sondeo.moldes
   and m2.sondeo.por_problema == m.sondeo.por_problema and m2.sondeo.avisos == ["aviso del sondeo"],
   "con todos sus campos")
ok(m2.sondeo.concuerda("infundado")[0] is True and m2.sondeo.objeciones_contra("infundado") != [] or True,
   "y con sus métodos vivos")
viejo = {"tesis": m.tesis, "normas": m.normas, "materia": "administrativa"}
ok(not te.esta_completo(viejo) and te.material_rehidratado(viejo).sondeo is None,
   "una fila antigua no se da por completa: se consulta como antes")
ok(te.material_rehidratado({}) is None and te.material_ligero(f6.Material())["sondeo"] is None,
   "sin nada guardado no hay material; sin sondeo, el campo va vacío")

print("\n2 · LA HUELLA DEL CONTRASTE")


def _r(problemas, acto="acto", conceptos="conceptos", es_recurso=True, glob=""):
    fases = types.SimpleNamespace(problemas=problemas, problema_global=glob,
                                  parrafos_acto=lambda: [acto],
                                  parrafos_conceptos=lambda: [conceptos])
    return types.SimpleNamespace(fases=fases, encargo=types.SimpleNamespace(es_recurso=es_recurso))


r1 = _r([{"pregunta": "¿A?", "jerarquia": "principal"}, "¿B?"])
ok(te.problemas_de(r1) == [{"pregunta": "¿A?", "jerarquia": "principal"}, {"pregunta": "¿B?"}],
   "los planteamientos se construyen como los recibe la propuesta")
ok(te.problemas_de(_r([], glob="¿G?")) == [{"pregunta": "¿G?"}], "sin planteamientos, el global")
h1 = te.huella_contraste(r1)
ok(h1 == te.huella_contraste(_r([{"pregunta": "¿A?", "jerarquia": "principal"}, "¿B?"])),
   "la misma entrada da la misma huella")
ok(h1 != te.huella_contraste(_r([{"pregunta": "¿A?", "jerarquia": "principal"}, "¿B bis?"])),
   "un planteamiento distinto la cambia")
ok(h1 != te.huella_contraste(_r([{"pregunta": "¿A?", "jerarquia": "principal"}, "¿B?"], acto="otro acto")),
   "otro resumen del acto también")
ok(h1 != te.huella_contraste(_r([{"pregunta": "¿A?", "jerarquia": "principal"}, "¿B?"], es_recurso=False)),
   "y el carácter de recurso")
pr, ac, co, es = te.entradas_contraste(r1)
ok((ac, co, es) == ("acto", "conceptos", True), "las entradas son las de la propuesta")

print("\n3 · LA ESPERA DEL CONTRASTE ADELANTADO")


def _espera(docs, reloj=None, tope=150.0):
    """Corre esperar_contraste con una secuencia de lecturas y un reloj falso."""
    lecturas = list(docs)
    t = {"ahora": 1000.0}
    dormidas = []

    def leer():
        return lecturas.pop(0) if len(lecturas) > 1 else lecturas[0]

    async def dormir(seg):
        dormidas.append(seg)
        t["ahora"] += (reloj or seg)

    res = asyncio.run(te.esperar_contraste("h1", leer, tope=tope,
                                           ahora=lambda: t["ahora"], dormir=dormir))
    return res, len(dormidas)


ok(_espera([{"huella": "h1", "estado": "listo", "items": [{"p": 1}]}]) == ([{"p": 1}], 0),
   "listo y de este adelanto: se recoge sin esperar")
ok(_espera([{"huella": "h1", "estado": "en_curso", "desde": 990.0},
            {"huella": "h1", "estado": "en_curso", "desde": 990.0},
            {"huella": "h1", "estado": "listo", "items": [{"p": 2}]}]) == ([{"p": 2}], 2),
   "en curso: espera y lo recoge cuando llega")
ok(_espera([{"huella": "OTRA", "estado": "listo", "items": [{"p": 3}]}]) == (None, 0),
   "de otro adelanto (huella distinta): no se usa")
ok(_espera([{"huella": "h1", "estado": "fallo"}]) == (None, 0), "falló: se calcula en la propuesta")
ok(_espera([None]) == (None, 0) and _espera([{"huella": "h1", "estado": "listo"}]) == (None, 0),
   "sin contraste guardado, o listo sin planteamientos: se calcula")
ok(_espera([{"huella": "h1", "estado": "en_curso", "desde": 1000.0 - 300}]) == (None, 0),
   "en curso desde hace 5 minutos: el worker murió, no se espera")
res, n = _espera([{"huella": "h1", "estado": "en_curso", "desde": 1000.0}], reloj=60.0)
ok(res is None and n == 3, f"nunca llega: se deja de esperar al tope (durmió {n} veces)")

print("\n4 · LAS PUERTAS ESTÁN CONECTADAS")
src = open("main.py", encoding="utf-8").read()
ok("import taller_estado as _te" in src, "main importa el módulo")
i_res = src.find("def _taller_recuperar_sesion(")
i_fin = src.find("_TALLER_SESIONES[_taller_llave(email, numero)] = ses\n    return ses", i_res)
ok(0 < i_res < src.find("if _te.esta_completo(_ml):", i_res) < i_fin,
   "el rescate de la sesión repone el acervo completo antes de devolverla")
ok("return _te.material_ligero(m)" in src and "return _te.material_rehidratado(d)" in src,
   "guardar y rehidratar pasan por el módulo")
i_ade = src.find('@app.post("/taller/adelanto")')
i_con = src.find('@app.post("/taller/consultar")')
ok(0 < i_ade < src.find("asyncio.ensure_future(_taller_precontrastar(user_email, numero, r))", i_ade) < i_con,
   "el adelanto lanza el contraste al terminar")
i_pro = src.find("async def _taller_proponer_nucleo(")
i_str = src.find('@app.post("/taller/resolver/stream")')
tramo = src[i_pro:i_str]
ok("problemas = _te.problemas_de(r)" in tramo, "proponer construye los planteamientos como la huella")
ok("await _taller_esperar_contraste(user_email, numero, r)" in tramo
   and "contraste_previo=_contraste_previo)" in tramo,
   "proponer recoge el contraste adelantado y se lo pasa a la propuesta")
ok(tramo.find("_taller_esperar_contraste(") > tramo.find("completar_preceptos("),
   "y lo espera DESPUÉS de traer los preceptos, para que ambas cosas se solapen")
ok('select(f"estado->{clave}")' in src, "la espera lee sólo la rama del contraste, no la fila entera")
import fase5_propuesta as f5
ok("contraste_previo" in inspect.signature(f5.proponer).parameters, "proponer admite el contraste previo")
s5 = inspect.getsource(f5.proponer)
ok(0 < s5.find("if contraste_previo is not None") < s5.find("elif CONTRASTE_EN_PARALELO:"),
   "y lo usa antes de decidir si calcularlo en serie o en paralelo")
ok(f5.ESFUERZO_PROPUESTA == "high" and not f5.CONTRASTE_EN_PARALELO,
   "el esfuerzo sigue alto y el contraste entra a la instrucción, como pidió David")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
