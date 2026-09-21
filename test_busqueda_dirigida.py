"""El acervo como herramienta: contrato, topes y lo que no se negocia.

Sin red: el modelo y Qdrant son de mentira. Lo que se mide contra el acervo de
verdad es otra cosa y vive en `redactor-sentencias/rag/medir_dirigida.py`.

    .venv/bin/python test_busqueda_dirigida.py
"""
import asyncio
import json
import types

import busqueda_dirigida as bd
import fase6_estudio as f6

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


def _msg(texto="", llamadas=None):
    tc = [types.SimpleNamespace(
        id=f"c{i}", function=types.SimpleNamespace(name=n, arguments=json.dumps(a)))
        for i, (n, a) in enumerate(llamadas or [])]
    return types.SimpleNamespace(choices=[types.SimpleNamespace(
        message=types.SimpleNamespace(content=texto, tool_calls=tc or None))],
        usage=None)


class ModeloFalso:
    """Devuelve la secuencia que le den y apunta cómo lo llamaron."""

    def __init__(self, guion):
        self.guion, self.llamadas = list(guion), []

    async def crear(self, cliente, **kw):
        self.llamadas.append(kw)
        return self.guion.pop(0) if self.guion else _msg("listo")


class QdrantFalso:
    def __init__(self, puntos=None):
        self.puntos = puntos if puntos is not None else [
            {"registro": "160053", "rubro": "COMPETENCIA. FUNDAMENTACIÓN", "vincula": True,
             "instancia": "Segunda Sala", "texto": "…"},
            {"registro": "2016056", "rubro": "NOTIFICACIÓN POR ESTRADOS", "vincula": False,
             "instancia": "TCC", "texto": "…"}]
        self.consultas = []


async def _embed(t):
    return [0.0] * 8


def _parchar(monkey_lm, monkey_rag):
    import sys
    sys.modules["llamada_modelo"] = monkey_lm
    sys.modules["fase6_rag"] = monkey_rag


print("\n1 · EL CONTRATO QUE VE EL MODELO")
hs = bd.herramientas()
nombres = [h["function"]["name"] for h in hs]
ok(nombres == ["buscar_tesis", "leer_articulo", "comprobar_registro"],
   f"tres herramientas y ninguna más: {nombres}")
desc = json.dumps(hs, ensure_ascii=False)
ok("RUBRO" in desc and "no\\n" not in desc,
   "a buscar_tesis se le pide la consulta redactada COMO UN RUBRO")
ok("nunca lo cites de memoria" in desc and "si no existe, no lo cites" in desc,
   "leer_articulo y comprobar_registro prohíben citar de memoria")
ok(all("required" in h["function"]["parameters"] for h in hs), "todas declaran lo obligatorio")

print("\n2 · LA RESTRICCIÓN DEL PROVEEDOR (por qué el módulo existe así)")
import inspect
src = inspect.getsource(bd.con_acervo)
ok('kw["reasoning_effort"] = "none"' in src,
   "con herramientas se pide «none» EXPRESAMENTE (omitirlo da el mismo 400)")
ok('if esfuerzo:' in src and 'kw["reasoning_effort"] = esfuerzo' in src,
   "y en la vuelta final, sin herramientas, se devuelve el esfuerzo de quien llamó")
ok(src.index('kw.pop("tools", None)') < src.index('kw["reasoning_effort"] = esfuerzo'),
   "primero se quitan las herramientas y después se restaura el razonamiento")
ok("/v1/responses" in bd.__doc__, "la cabecera deja dicha la ruta para razonar Y buscar a la vez")

print("\n3 · EL BUCLE: busca, lee lo que vuelve, y para")
lm = types.SimpleNamespace(crear=None)
rag = types.SimpleNamespace(
    COLECCION_JURIS="jurisprudencia_nacional_v3",
    _buscar=None, _tesis_de=None, resolver_articulo=None, tesis_por_registro=None)


def _montar(guion, puntos=None):
    q = QdrantFalso(puntos)
    m = ModeloFalso(guion)
    lm.crear = m.crear

    async def _buscar(qd, col, vec, v, limite, filtro=None):
        q.consultas.append((col, vec, limite))
        return list(q.puntos)

    rag._buscar = _buscar
    rag._tesis_de = lambda p: {"registro": str(p.get("registro") or ""),
                               "rubro": p.get("rubro") or "", "instancia": p.get("instancia") or "",
                               "texto": p.get("texto") or "", "obligatoria": bool(p.get("vincula")),
                               "tipo": "", "materia": "", "localizacion": ""}

    async def _art(qd, col, ley, num):
        return {"cuerpo_legal": ley, "articulo": str(num), "texto": "Artículo …"} if "Federal" in ley else {}

    rag.resolver_articulo = _art

    async def _reg(qd, regs):
        return [{"registro": regs[0], "rubro": "R", "instancia": "TCC", "obligatoria": False}] \
            if regs and regs[0] == "160053" else []

    rag.tesis_por_registro = _reg
    _parchar(lm, rag)
    return q, m


q, m = _montar([_msg("", [("buscar_tesis", {"consulta": "COMPETENCIA. FUNDAMENTACIÓN"})]),
                _msg("", [("buscar_tesis", {"consulta": "OTRA COSA"})]),
                _msg('{"registros": ["160053"]}')])
r, hall = asyncio.run(bd.con_acervo(object(), q, _embed, dict(messages=[{"role": "user", "content": "x"}])))
ok(hall.vueltas == 2 and len(hall.consultas) == 2, f"dos búsquedas y para cuando el modelo deja de pedir: {hall.vueltas}")
ok(q.consultas and q.consultas[0][0] == "jurisprudencia_nacional_v3" and q.consultas[0][1] == "rubro",
   "busca en la colección de jurisprudencia contra el vector rubro, como producción")
ok(len(m.llamadas) == 3 and "tools" in m.llamadas[0] and "tools" in m.llamadas[1],
   "las vueltas con petición llevan herramientas")
ok(hall.registros() == ["160053", "2016056"], "los hallazgos guardan lo que trajo, sin repetir")

print("\n4 · EL TOPE ES DE VERDAD")
q, m = _montar([_msg("", [("buscar_tesis", {"consulta": f"C{i}"})]) for i in range(9)])
r, hall = asyncio.run(bd.con_acervo(object(), q, _embed, dict(messages=[{"role": "user", "content": "x"}]), tope=2))
ok(hall.vueltas == 2, f"un modelo que pide sin parar se corta en el tope: {hall.vueltas}")
ok("tools" not in m.llamadas[-1], "y la última vuelta va SIN herramientas, para que cierre")

print("\n5 · LO QUE DEVUELVE CADA HERRAMIENTA")
q, _ = _montar([])
h = bd.Hallazgos()
res = asyncio.run(bd._ejecutar(q, _embed, "buscar_tesis", {"consulta": "X"}, h))
ok(json.loads(res)[0]["registro"] == "160053" and "texto" not in json.loads(res)[0],
   "buscar_tesis devuelve registro y rubro; el texto entero no va al prompt")
res = asyncio.run(bd._ejecutar(q, _embed, "leer_articulo", {"ley": "Ley Federal X", "articulo": "51"}, h))
ok("Artículo" in res and h.normas, "leer_articulo trae el texto y lo apunta")
res = asyncio.run(bd._ejecutar(q, _embed, "leer_articulo", {"ley": "Ley de Nadie", "articulo": "9"}, h))
ok("NO lo transcribas de memoria" in res, "y si no está, se lo dice con todas las letras")
res = asyncio.run(bd._ejecutar(q, _embed, "comprobar_registro", {"registro": "999999"}, h))
ok("NO EXISTE" in res and "No la cites" in res, "un registro inventado se desmiente")
res = asyncio.run(bd._ejecutar(q, _embed, "inventada", {}, h))
ok("No existe la herramienta" in res, "una herramienta que no existe no rompe nada")


class QRoto(QdrantFalso):
    pass


async def _explota(*a, **k):
    raise RuntimeError("Qdrant caído")


rag._buscar = _explota
res = asyncio.run(bd._ejecutar(QRoto(), _embed, "buscar_tesis", {"consulta": "X"}, h))
ok("La búsqueda falló" in res, "si el acervo se cae, el modelo sigue con lo que tenga")

print("\n6 · LO QUE ENCUENTRA ENTRA AL MATERIAL (o el verificador lo acusaría)")
mat = f6.Material()
mat.tesis = [{"registro": "160053", "rubro": "YA ESTABA"}]
mat.normas = [{"cuerpo_legal": "Ley X", "articulo": "1"}]
h2 = bd.Hallazgos()
h2.tesis = [{"registro": "160053", "rubro": "repetida"}, {"registro": "2016056", "rubro": "NUEVA"}]
h2.normas = [{"cuerpo_legal": "Ley X", "articulo": "1"}, {"cuerpo_legal": "Ley Y", "articulo": "2"}]
t, n = h2.al_material(mat)
ok((t, n) == (1, 1), f"sólo lo nuevo se suma: {t} tesis y {n} normas")
ok(len(mat.tesis) == 2 and mat.tesis[1]["rubro"] == "NUEVA", "y queda donde el redactor lo ve")
ok(h2.al_material(None) == (0, 0), "sin material, no revienta")
# EL DEFECTO DEL 2/2026: buscar_tesis lo encuentra y comprobar_registro lo
# confirma, así que el mismo registro venía dos veces en la misma lista.
mat3 = f6.Material()
h3 = bd.Hallazgos()
h3.tesis = [{"registro": "174219", "rubro": "A"}, {"registro": "174219", "rubro": "A"}]
h3.normas = [{"cuerpo_legal": "Ley Z", "articulo": "3"}, {"cuerpo_legal": "Ley Z", "articulo": "3"}]
ok(h3.al_material(mat3) == (1, 1) and len(mat3.tesis) == 1 and len(mat3.normas) == 1,
   "una tesis repetida DENTRO de los hallazgos entra una sola vez")

print("\n7 · EL REFUERZO PIDE LO QUE FALTA, NO LO QUE YA HAY")
src_r = inspect.getsource(bd.reforzar)
ok("LO QUE YA TIENES para ese punto" in bd._PROMPT_REFUERZO,
   "al modelo se le enseñan los rubros que el material ya trae")
ok("Si lo que ya tienes basta, no busques nada" in bd._PROMPT_REFUERZO,
   "y se le permite no buscar: el refuerzo no obliga a gastar")
ok("TU TAREA NO ES RESOLVERLO" in bd._PROMPT_REFUERZO,
   "no decide el punto: sólo trae autoridad (el sentido es del secretario)")
ok("asyncio.gather" in src_r and "tope_problemas" in src_r,
   "los planteamientos van a la vez y con tope")
ok("h.al_material(material)" in src_r, "y lo hallado se suma al material")
mat2 = f6.Material()
ok(asyncio.run(bd.reforzar(object(), None, _embed, [], mat2)) == {},
   "sin planteamientos no hace nada")

print("\n7b · AL MATERIAL ENTRA LO QUE EL MODELO SE QUEDÓ, NO TODO LO QUE VIO")
ok(bd._faltaba("bla bla\nFALTABA: 2000091, 175087") == {"2000091", "175087"},
   "se leen los registros de la línea FALTABA")
ok(bd._faltaba("FALTABA: nada") == set() and bd._faltaba("sin línea") == set(),
   "«nada» o sin línea: no entra nada (en la duda, no se mete)")
ok('os.getenv("REFUERZO_TODO", "0") != "1"' in src_r
   and "hall.tesis = [t for t in hall.tesis" in src_r,
   "por omisión se filtra a lo que se quedó; REFUERZO_TODO=1 mete la unión entera")

print("\n8 · LA PUERTA EN EL PIPELINE")
m = open("main.py", encoding="utf-8").read()
i_pre = m.find("async def _taller_preconsultar(")
i_fin = m.find("async def _taller_preproponer(")
tramo = m[i_pre:i_fin]
ok("_bd.reforzar(" in tramo, "la consulta automática refuerza el acervo")
ok(tramo.index("_bd.reforzar(") < tramo.index("_taller_guardar_material("),
   "ANTES de guardar el material: si no, lo hallado no llega a la fila")
ok('os.getenv("REFUERZO_ACERVO", "1") != "0"' in tramo, "con interruptor para apagarlo")
ok("_taller_con_latido(email, numero, \"consulta\", huella, _desde,\n                    _bd.reforzar(" in tramo,
   "y late mientras busca, para que nadie lo dé por muerto")
ok("except Exception as _exr:" in tramo, "si falla, la consulta sigue valiendo")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
