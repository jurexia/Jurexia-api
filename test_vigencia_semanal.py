"""La regeneración semanal del índice de vigencia — 26-sep-2026.

    .venv/bin/python test_vigencia_semanal.py

Sin red y sin Qdrant (git, sólo con repos de juguete en una carpeta temporal): el SJF es un doble que contesta como el de
verdad —una ficha JSON si el registro existe y, si no, 200 con la página de
Incapsula (medido: el SJF NO da 404)—. Se comprueba: el avance registro a
registro con huecos y la racha sin fichas (y el salto de numeración, y el
bloqueo), que un error no se guarda como tesis, la extracción de las
afectadas de una tesis nueva, que la versión del SJF manda sobre la de Qdrant,
que una tesis nueva fuera del acervo sirve de reemplazo y nunca de afectada,
la cordura que detiene, la idempotencia (sin novedades → el mismo JSON), que
las pruebas llevan el .env y que en seco no se toca el índice de ningún
checkout, y el LaunchAgent: su lanzador (que funciona aunque el worktree
dedicado no exista y avisa si falla), el worktree con su marca y su .env, y el
candado —con un marcador de arranque, sin carreras; dentro de la corrida
semanal (VIGENCIA_SEMANAL_DENTRO=1) esa prueba se salta—.
"""
import datetime as dt
import importlib.util
import json
import os
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


def _modulo(nombre, ruta):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sem = _modulo("vigencia_semanal", RAIZ / "scripts" / "vigencia_semanal.py")
gen, dl = sem.gen, sem.dl
INCAPSULA_HTML = (b'<html style="height:100%"><head><META NAME="ROBOTS" CONTENT="NOINDEX, NOFOLLOW">'
                  b'<script type="text/javascript" src="/_Incapsula_Resource?SWJIYLWA=719d34d31c8e3a6e6fffd425f7e0">'
                  b'</script></head><body></body></html>')


# ═══════════════════════════════════════════════════════════════ el SJF de mentira
class _Resp:
    def __init__(self, cuerpo):
        self.cuerpo = cuerpo

    def read(self):
        return self.cuerpo

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class SJFFalso:
    """`fichas`: {registro: dict} de la Gaceta (`isSemanal=false`); `semanales`:
    las que sólo tiene el Semanario semanal (`isSemanal=true`), como lo publicado
    después del 10-jul-2026. Lo que no está contesta la página de Incapsula (o
    404 con `con_404`). `bloqueado`: todo es Incapsula. `rotos`: {registro:
    cuerpo} que contesta tal cual (errores que no deben guardarse)."""

    def __init__(self, fichas, bloqueado=False, con_404=False, rotos=None, caidos=(), semanales=None):
        self.fichas, self.bloqueado, self.con_404 = dict(fichas), bloqueado, con_404
        self.rotos, self.caidos = dict(rotos or {}), set(caidos)
        self.semanales = dict(semanales or {})
        self.pedidos = []                 # a la Gaceta
        self.pedidos_semanal = []         # al Semanario semanal

    def __call__(self, req, timeout=30):
        assert req.get_method() == "GET" and "sjf2.scjn.gob.mx" in req.full_url
        r = req.full_url.split("/tesis/")[1].split("?")[0]
        if "isSemanal=true" in req.full_url:
            self.pedidos_semanal.append(r)
            if self.bloqueado or r not in self.semanales:
                return _Resp(INCAPSULA_HTML)
            # como el de verdad: la ficha del Semanario semanal trae «semanal»: 1
            return _Resp(json.dumps(dict(self.semanales[r], semanal=1), ensure_ascii=False).encode())
        assert "isSemanal=false" in req.full_url
        self.pedidos.append(r)
        if r in self.caidos:
            raise OSError("timed out")
        if r in self.rotos:
            return _Resp(self.rotos[r])
        if self.bloqueado or r not in self.fichas:
            if self.con_404 and not self.bloqueado:
                raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
            return _Resp(INCAPSULA_HTML)
        return _Resp(json.dumps(self.fichas[r], ensure_ascii=False).encode())


def descargador(cache, sjf):
    reloj, pausas = [0.0], []

    def dormir(s):
        pausas.append(s)
        reloj[0] += s
    D = dl.Descargador(Path(cache), abrir=sjf, dormir=dormir, reloj=lambda: reloj[0],
                       validar=True, reintentos=2, aligerar=True, semanal_si_falta=True)
    D.pausas = pausas
    return D


def ficha(reg, clave, rubro, prec="Amparo en revisión 1/2026. 1 de junio de 2026.", tj=0,
          fecha="2026-07-10 10:17:00.0"):
    return {"ius": int(reg), "claveTesis": clave, "rubro": f"<p>{rubro}</p>",
            "precedentes": "".join(f"<p>{p}</p>" for p in prec.split("\n\n")), "ta_tj": tj,
            "tipoTesis": "Tesis Jurisprudenciales" if tj else "Tesis Aisladas", "epoca": "Duodécima Época",
            "fechaPublicacion": fecha, "instancia": "Pleno", "materias": "Común",
            "lstTemasDetalle": [{"descripcion": "PESADO"}], "idProg": "PESADO"}


def v3(reg, clave, rubro, prec="Amparo en revisión 1/2015. 1 de junio de 2015.", tipo="TESIS AISLADA",
       fecha="2015-06-01"):
    return {"registro": str(reg), "clave_tesis": clave, "rubro": rubro, "precedentes": prec, "tipo": tipo,
            "epoca": "Décima Época", "fecha_publicacion": fecha, "instancia": "Pleno", "materia": "Común"}


def guardar_ficha(cache, f):
    Path(cache).mkdir(parents=True, exist_ok=True)
    Path(cache, f"{f['ius']}.json").write_text(json.dumps(f, ensure_ascii=False), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════ 1 · tesis nuevas
print("\n1 · TESIS NUEVAS: REGISTRO A REGISTRO, CON HUECOS Y RACHAS SIN FICHA")
with tempfile.TemporaryDirectory() as d:
    fichas = {str(r): ficha(r, f"I.1o.A.{r} A (12a.)", f"RUBRO {r}.") for r in (1000, 1001, 1002, 1004, 1007)}
    sjf = SJFFalso(fichas)
    D = descargador(Path(d, "c"), sjf)
    b = sem.buscar_nuevas(D, 1001, 1000, ["1000"], fallos=5, saltos=(20,))
    ok(b["halladas"] == [1001, 1002, 1004, 1007] and b["huecos"] == [1003, 1005, 1006] and b["ultimo"] == 1007,
       f"halla 1001, 1002, 1004 y 1007; los huecos 1003, 1005 y 1006 ({b['halladas']}, {b['huecos']})")
    ok(b["hasta"] == 1012 and sjf.pedidos[:12] == [str(r) for r in range(1001, 1013)],
       "para tras 5 registros seguidos sin ficha (1008-1012)")
    ok(sjf.pedidos[12:] == ["1000", "1027"] and b["saltos"] == [],
       f"al cerrar la racha pregunta al testigo (¿bloqueo?) y da un salto largo (+20) ({sjf.pedidos[12:]})")
    guardadas = sorted(p.name for p in Path(d, "c").iterdir())
    ok(guardadas == ["1000.json", "1001.json", "1002.json", "1004.json", "1007.json"],
       "sólo las fichas se guardan: ni la página de Incapsula ni los huecos")
    ok(D.pausas and min(D.pausas) >= 1.0, f"al menos 1 s entre peticiones (mínimo {min(D.pausas):.2f} s)")
    ok("lstTemasDetalle" not in json.loads(Path(d, "c", "1001.json").read_text())
       and "precedentes" in json.loads(Path(d, "c", "1001.json").read_text()),
       "se guarda aligerada: sin lstTemasDetalle ni idProg, con precedentes")

with tempfile.TemporaryDirectory() as d:
    fichas = {str(r): ficha(r, f"I.1o.A.{r} A (12a.)", f"RUBRO {r}.") for r in (1000, 1001, 1021)}
    sjf = SJFFalso(fichas, con_404=True)
    b = sem.buscar_nuevas(descargador(Path(d, "c"), sjf), 1001, 1000, ["1000"], fallos=5, saltos=(20, 50))
    ok(b["saltos"] == [1021],
       f"si un salto (+20 desde la última, 1001) da ficha, la numeración brincó ({b['saltos']})")
    ok(b["halladas"] == [1001, 1021] and b["ultimo"] == 1021 and b["hasta"] == 1026,
       f"… se recorre el tramo y se sigue registro a registro después ({b['halladas']}, hasta {b['hasta']})")
    ok(set(range(1002, 1021)) == set(b["huecos"]), "el tramo sin fichas queda como huecos")
    ok(sorted(p.name for p in Path(d, "c").iterdir()) == ["1000.json", "1001.json", "1021.json"],
       "con 404 en vez de Incapsula, igual: nada de errores en la caché")

with tempfile.TemporaryDirectory() as d:
    # Lo que la Gaceta aún no compila está en el Semanario semanal (medido: la
    # 2032444, del 7-ago-2026, sólo sale con isSemanal=true).
    gaceta = {str(r): ficha(r, f"I.1o.A.{r} A (12a.)", f"RUBRO {r}.") for r in (1000, 1001)}
    semanal = {str(r): ficha(r, f"I.1o.A.{r} A (12a.)", f"RUBRO {r}.") for r in (1002, 1003, 1005)}
    sjf = SJFFalso(gaceta, semanales=semanal)
    D = descargador(Path(d, "c"), sjf)
    b = sem.buscar_nuevas(D, 1001, 1000, ["1000"], fallos=5, saltos=(20,))
    ok(b["halladas"] == [1001, 1002, 1003, 1005] and b["huecos"] == [1004] and D.del_semanal == 3
       and sorted(p.name for p in Path(d, "c").iterdir()) == ["1000.json", "1001.json", "1002.json", "1003.json",
                                                             "1005.json"],
       f"lo que la Gaceta no tiene se pide al Semanario semanal: halla 1002, 1003 y 1005 ({b['halladas']})")
    ok("1001" not in sjf.pedidos_semanal and "1002" in sjf.pedidos and "1002" in sjf.pedidos_semanal,
       "primero la Gaceta (como se armó la caché); al Semanario sólo si la Gaceta no la tiene")
    ok("1003" not in sjf.pedidos and "1005" not in sjf.pedidos,
       "y, pasada la Gaceta, se le pregunta primero al Semanario: una petición por tesis nueva, no dos")

with tempfile.TemporaryDirectory() as d:
    sjf = SJFFalso({"1000": ficha(1000, "X", "R.")}, bloqueado=True)
    try:
        sem.buscar_nuevas(descargador(Path(d, "c"), sjf), 1001, 1000, ["1000"], fallos=5, saltos=(20,))
        bloqueo = False
    except sem.Bloqueado:
        bloqueo = True
    ok(bloqueo and not Path(d, "c").exists(),
       "si el testigo tampoco llega, NO es el final de la numeración: Bloqueado, y nada guardado")

with tempfile.TemporaryDirectory() as d:
    rotos = {"1001": json.dumps({"status": 500, "error": "Internal"}).encode(),
             "1002": json.dumps(ficha(9999, "OTRA", "OTRA.")).encode(), "1003": b"null"}
    sjf = SJFFalso({"1000": ficha(1000, "X", "R.")}, rotos=rotos, caidos={"1004"})
    D = descargador(Path(d, "c"), sjf)
    res = {r: D.tesis(r) for r in ("1001", "1002", "1003", "1004")}
    ok(all("_error" in v and v["_tipo"] == "error" for v in res.values())
       and not any(Path(d, "c", f"{r}.json").exists() for r in res),
       "un JSON de error, la ficha de OTRO registro, `null` y un timeout son «error» y no se guardan")
    ok(sjf.pedidos.count("1004") == 3 and sjf.pedidos.count("1001") == 3,
       "los errores se reintentan (2 veces, con espera); la página de Incapsula no")
    guardar_ficha(Path(d, "c"), ficha(1000, "X", "R.", prec="La versión vieja."))
    sjf.caidos.add("1000")
    r = D.tesis("1000", refrescar=True)
    ok("_error" in r and "La versión vieja." in Path(d, "c", "1000.json").read_text(),
       "un refresco fallido deja la copia anterior como estaba")

# ═══════════════════════════════════════════════════════════════ 1b · huecos y carriles
hoy = dt.date(2026, 9, 27)
with tempfile.TemporaryDirectory() as d:
    sjf = SJFFalso({"1003": ficha(1003, "X", "R.")})
    hh, siguen = sem.repasar_huecos(descargador(Path(d, "c"), sjf),
                                    {"1003": "2026-09-20", "1005": "2026-09-20", "1006": "2026-07-01",
                                     "1008": "2026-09-27"}, hoy)
    ok(hh == [1003] and siguen == {"1005": "2026-09-20", "1008": "2026-09-27"} and sorted(sjf.pedidos) == ["1003", "1005"],
       "los huecos se vuelven a pedir: el que ya tiene ficha sale, el de hoy espera y el de hace 12 semanas se olvida")
with tempfile.TemporaryDirectory() as d:
    sjf = SJFFalso({}, bloqueado=True)
    D = descargador(Path(d, "c"), sjf)
    estado0 = sem.estado_vacio()
    try:
        sem.actualizar_cache(D, [v3(2032214, "X (12a.)", "LA ÚLTIMA.")], estado0, Path(d, "c"), {}, {}, hoy,
                             esperas_red=(60,))
        parado = False
    except sem.Bloqueado:
        parado = True
    ok(parado and 60 in D.pausas and not Path(d, "c").exists() and estado0 == sem.estado_vacio(),
       "si el SJF no contesta al empezar (red que no ha vuelto, Incapsula): espera, reintenta y para sin tocar nada")
carril = {"vuelta": 1, "hechos": ["a", "b"]}
t1 = sem.elegir(["a", "b", "c", "d"], carril, 1)
t2 = sem.elegir(["a", "b", "c", "d"], {"vuelta": 1, "hechos": ["a", "b", "c"]}, 3)
c3 = {"vuelta": 1, "hechos": ["a", "b", "c"]}
sem.elegir(["a", "b", "c", "d"], c3, 3)
ok(t1 == ["c"] and t2 == ["d", "a", "b"] and c3["vuelta"] == 2 and c3["hechos"] == [],
   "el repaso sigue donde se quedó y, al acabar la vuelta, empieza otra")
ok(sem.orden_general([v3(160000, "a", "r"), v3(2020000, "b", "r"), v3(2010000, "c", "r", tipo="JURISPRUDENCIA"),
                      v3(170000, "d", "r", tipo="JURISPRUDENCIA")]) == ["2010000", "170000", "2020000", "160000"],
   "carril general: jurisprudencias primero y, en cada grupo, de la más nueva a la más vieja")

# ═══════════════════════════════════════════════════════════════ 2 · afectadas de una tesis nueva
print("\n2 · LAS AFECTADAS DE UNA TESIS NUEVA")
ACERVO = [
    v3(2009817, "P. X/2015 (10a.)", "CONTROL CONSTITUCIONAL Y CONVENCIONAL. SU EJERCICIO EN AMPARO DIRECTO."),
    v3(2009816, "P. IX/2015 (10a.)", "CONTROL CONSTITUCIONAL Y CONVENCIONAL. LÍMITES EN AMPARO DIRECTO."),
    v3(2019000, "2a./J. 5/2019 (10a.)", "UNA JURISPRUDENCIA DE LA SEGUNDA SALA QUE SE INTERRUMPE.",
       tipo="JURISPRUDENCIA", fecha="2019-02-01"),
    v3(2020500, "I.3o.C.99 C (10a.)", "UNA DE COLEGIADO QUE NADIE TOCA."),
]
N1 = ficha(2032300, "P./J. 7/2026 (12a.)",
           "CONTROL DE REGULARIDAD CONSTITUCIONAL. SU ALCANCE [ABANDONO DE LAS TESIS AISLADAS P. IX/2015 (10a.) "
           "Y P. X/2015 (10a.)].", tj=1)
N2 = ficha(2032301, "2a./J. 30/2026 (12a.)", "UNA JURISPRUDENCIA NUEVA DE LA SEGUNDA SALA.",
           prec="Contradicción de criterios 10/2026.\n\nEsta tesis interrumpe el criterio sostenido en la diversa "
                "2a./J. 5/2019 (10a.), de rubro: \"UNA JURISPRUDENCIA DE LA SEGUNDA SALA QUE SE INTERRUMPE.\"", tj=1)
with tempfile.TemporaryDirectory() as d:
    cache = Path(d, "c")
    for f in (N1, N2):
        guardar_ficha(cache, f)
    gen.SJF_CACHE = str(cache)
    nuevas = gen.cargar_nuevas_sjf(str(cache), {t["registro"] for t in ACERVO})
    ok([t["registro"] for t in nuevas] == ["2032300", "2032301"] and nuevas[0]["tipo"] == "JURISPRUDENCIA"
       and nuevas[0]["rubro"].startswith("CONTROL DE REGULARIDAD") and "<p>" not in nuevas[0]["rubro"],
       "las fichas de la caché que no están en el acervo entran como tesis nuevas, con la forma de la v3")
    regs, detalle = sem.afectadas_de(ACERVO, nuevas, ["2032300", "2032301"])
    ok(sorted(regs) == ["2009816", "2009817", "2019000"],
       f"«[ABANDONO DE LAS TESIS … P. IX/2015 Y P. X/2015]» y «interrumpe … 2a./J. 5/2019»: las tres ({sorted(regs)})")
    ok("2032300" not in regs and "2020500" not in regs, "ni la nueva ni una tesis ajena")
    regs2, _ = sem.afectadas_de(ACERVO, nuevas, ["2032301"])
    ok(regs2 == ["2019000"], "sólo las de las nuevas que se piden (las ya revisadas no se repiten)")
    gen.SJF_CACHE = None

# ═══════════════════════════════════════════════════════════════ 3 · el SJF manda
print("\n3 · LA VERSIÓN DEL SJF MANDA SOBRE LA DE QDRANT")
T_SJF = [
    v3(2019978, "2a. XL/2019 (10a.)", "UNA TESIS QUE EL SJF ABANDONÓ DESPUÉS DE LA INGESTA.",
       prec="Amparo en revisión 1/2019. 1 de marzo de 2019."),
    v3(2031000, "2a./J. 7/2026 (12a.)", "LA JURISPRUDENCIA QUE LA ABANDONA.", prec="", tipo="JURISPRUDENCIA",
       fecha="2026-08-14"),
    v3(2019979, "2a. XLI/2019 (10a.)", "UNA TESIS CON ESPACIOS RAROS EN EL SJF.",
       prec="Amparo en revisión 2/2019. 1 de marzo de 2019. Ponente: Fulano.\n\nNota: Esta tesis se publicó."),
]
with tempfile.TemporaryDirectory() as d:
    cache = Path(d, "c")
    gen.SJF_CACHE = None
    sin = gen.compacto(gen.generar(T_SJF, [])[0])
    guardar_ficha(cache, ficha(2019978, "2a. XL/2019 (10a.)", "X.", prec="Amparo en revisión 1/2019. 1 de marzo de "
                               "2019.\n\nNota: La presente tesis fue abandonada por la tesis 2a./J. 7/2026 (12a.), "
                               "publicada el viernes 14 de agosto de 2026."))
    # la misma nota de Qdrant, con lo que el SJF trae de más: NBSP, dobles espacios, tabulador y saltos finales
    f79 = ficha(2019979, "2a. XLI/2019 (10a.)", "X.")
    f79["precedentes"] = ("<p>Amparo en revisión 2/2019.  1 de marzo\xa0de 2019. Ponente:\tFulano.</p>"
                          "<p>Nota: Esta tesis se publicó.</p><br>\n\n")
    guardar_ficha(cache, f79)
    gen.SJF_CACHE = str(cache)
    con = gen.compacto(gen.generar(T_SJF, [])[0])
    ok("2019978" not in sin and con.get("2019978", {}).get("por_registro") == "2031000"
       and con["2019978"]["fuente"] == "nota_propia",
       "la nota añadida en el SJF después de la ingesta cuenta aunque Qdrant no esté cortado")
    ok(gen.precedentes_completos(T_SJF[2]) == (T_SJF[2]["precedentes"], "sjf"),
       "el HTML del SJF sale idéntico al texto de Qdrant (NBSP, espacios dobles, tabulador, saltos al final)")
    Path(cache, "2019979.json").write_text(json.dumps({"ius": 2019979, "precedentes": ""}))
    ok(gen.precedentes_completos(T_SJF[2]) == (T_SJF[2]["precedentes"], "qdrant"),
       "una ficha sin `precedentes` no pisa el de Qdrant")
    gen.SJF_CACHE = None

# El SJF manda… salvo que Qdrant sea más nuevo (revisión del 26-sep-2026: el
# repaso vuelve a una tesis del carril general una vez al año; una ficha rancia
# no puede tapar lo que trajo una reingesta posterior).
with tempfile.TemporaryDirectory() as d:
    cache = Path(d, "c")
    gen.SJF_CACHE = str(cache)
    base = "Amparo en revisión 3/2019. 1 de marzo de 2019. Ponente: Fulano."
    nota = "Nota: Esta tesis fue abandonada por la tesis 2a./J. 9/2026 (12a.)."
    t_q = v3(2019990, "2a. L/2019 (10a.)", "X.", prec=f"{base}\n\n{nota}")
    t_q["ingesta"] = "v3_semanario_2026-08"
    f = ficha(2019990, "2a. L/2019 (10a.)", "X.", prec=base)
    f["_bajada"] = "2026-09-27T03:40:00-06:00"
    guardar_ficha(cache, f)
    ok(gen.precedentes_completos(t_q) == (t_q["precedentes"], "qdrant_mas_nuevo"),
       "si Qdrant contiene lo del SJF y algo más (una nota que trajo la ingesta), manda Qdrant")
    # otra redacción: Qdrant NO contiene lo del SJF, así que sólo decide la fecha
    t_d = dict(t_q, precedentes="Amparo en revisión 3/2019. 1 de marzo de 2019. Ponente: Mengano.")
    f["_bajada"] = "2026-07-02T03:40:00-06:00"
    guardar_ficha(cache, f)
    ok(gen.precedentes_completos(t_d) == (t_d["precedentes"], "qdrant_mas_nuevo"),
       "si la ficha se bajó ANTES de la ingesta (2-jul < ago-2026), manda Qdrant")
    f["_bajada"] = "2026-09-27T03:40:00-06:00"
    guardar_ficha(cache, f)
    ok(gen.precedentes_completos(t_d) == (base, "sjf"), "si se bajó después, manda el SJF")
    largo = "Amparo directo 1/2019. " + "Precedente largo. " * 200
    t_c = v3(2019991, "2a. LI/2019 (10a.)", "X.", prec=largo[:gen.TOPE_QDRANT])
    t_c["ingesta"] = "v3_semanario_2026-08"
    fc = ficha(2019991, "2a. LI/2019 (10a.)", "X.", prec=largo.strip() + "\n\n" + nota)
    fc["_bajada"] = "2026-07-02T03:40:00-06:00"
    guardar_ficha(cache, fc)
    ok(gen.precedentes_completos(t_c)[1] == "sjf" and gen.precedentes_completos(t_c)[0].endswith(nota),
       "aunque sea más vieja, si Qdrant está cortado y la ficha lo continúa tal cual, manda la ficha (trae el final)")
    f.pop("_bajada")
    guardar_ficha(cache, f)
    os.utime(Path(cache, "2019990.json"), (dt.datetime(2026, 7, 2).timestamp(),) * 2)
    ok(gen.precedentes_completos(t_d)[1] == "qdrant_mas_nuevo",
       "una ficha de antes, sin `_bajada`, se fecha por el archivo")
    t_sin = dict(t_d)
    t_sin.pop("ingesta")
    ok(gen.precedentes_completos(t_sin) == (base, "sjf"), "sin `ingesta` en el volcado, manda el SJF, como antes")
    ok(gen.texto_sjf("<p> Amparo en revisión 5/2020.</p>") == "Amparo en revisión 5/2020.",
       "texto_sjf recorta por los dos lados (la 2029910 traía un espacio al principio)")
    gen.SJF_CACHE = None

with tempfile.TemporaryDirectory() as d:
    # Un espacio al principio no es «nota distinta» (salía en el mensaje del commit).
    q = "Amparo en revisión 4/2020. 1 de marzo de 2020."
    sjf = SJFFalso({"2029910": ficha(2029910, "X", "R.", prec=" " + q)})
    rr = sem.refrescar(descargador(Path(d, "c"), sjf), ["2029910"], lambda r: (q, False), ["2029910"])
    ok(rr["ok"] == ["2029910"] and rr["cambiadas"] == [],
       f"el repaso no cuenta como distinta la que sólo trae un espacio al principio ({rr['cambiadas']})")
    ok("_bajada" in json.loads(Path(d, "c", "2029910.json").read_text()),
       "cada ficha bajada guarda cuándo se bajó (`_bajada`)")

# ═══════════════════════════════════════════════════════════════ 4 · la nueva fuera del acervo como reemplazo
print("\n4 · UNA TESIS NUEVA FUERA DEL ACERVO COMO REEMPLAZO")
T_ACERVO = ACERVO + [
    v3(190001, "2a. L/98", "UNA VIEJA QUE CITA SIN ÉPOCA.",
       prec="Amparo en revisión 9/98.\n\nNota: Esta tesis fue interrumpida por la tesis 2a./J. 30/2026."),
    v3(190002, "2a. LI/98", "UNA VIEJA QUE CITA CON ÉPOCA.",
       prec="Amparo en revisión 10/98.\n\nNota: Esta tesis fue interrumpida por la tesis 2a./J. 30/2026 (12a.)."),
]
N3 = ficha(2032302, "P./J. 8/2026 (12a.)", "UNA NUEVA QUE DICE DE SÍ MISMA QUE LA ABANDONARON.",
           prec="Nota: La presente tesis fue abandonada por la tesis P./J. 1/2027 (12a.).")
with tempfile.TemporaryDirectory() as d:
    cache = Path(d, "c")
    for f in (N1, N2, N3):
        guardar_ficha(cache, f)
    gen.SJF_CACHE = str(cache)
    nuevas = gen.cargar_nuevas_sjf(str(cache), {t["registro"] for t in T_ACERVO})
    idx, _, E = gen.generar(T_ACERVO, nuevas)
    comp = gen.compacto(idx)
    e817 = idx.get("2009817", {})
    ok(e817.get("por_registro") == "2032300" and e817.get("por_en_acervo") is False
       and e817.get("fuente") == "tesis_nueva" and comp["2009817"]["por_clave"] == "P./J. 7/2026 (12a.)",
       "la P. X/2015 sale abandonada por la 2032300, que aún no está en el acervo (por_en_acervo = false)")
    ok(idx.get("2019000", {}).get("por_registro") == "2032301", "y la 2a./J. 5/2019, interrumpida por la 2032301")
    ok(not any(r in idx for r in ("2032300", "2032301", "2032302")),
       "una tesis nueva nunca es la afectada (el chat aún no la puede traer)")
    ok(idx.get("190002", {}).get("por_registro") == "2032301",
       "una nota vieja que cita la clave CON su época se resuelve a la nueva")
    ok("190001" in idx and idx["190001"]["por_registro"] is None,
       "pero sin época, no: sería de antes de la Décima y la nueva le robaría la cita a otra")
    A = gen.Acervo(T_ACERVO, nuevas)
    ok(A.resolver_clave("P. X/2015 (10a.)") == ("2009817", "alta") and "2032300" not in A.en_acervo,
       "lo que ya resolvía el acervo se resuelve igual")
    gen.SJF_CACHE = None

# ═══════════════════════════════════════════════════════════════ 5 · la cordura
print("\n5 · LA CORDURA QUE DETIENE")
viejo = {str(r): {"estado": "abandonada"} for r in range(1, 559)}
d_11 = sem.comparar(viejo, {r: v for r, v in viejo.items() if int(r) > 11})
d_12 = sem.comparar(viejo, {r: v for r, v in viejo.items() if int(r) > 12})
nuevo_51 = dict(viejo)
for r in range(600, 651):
    nuevo_51[str(r)] = {"estado": "superada"}
d_51 = sem.comparar(viejo, nuevo_51)
ok(sem.cordura(558, d_11) == [] and sem.cordura(558, d_12) and "desaparecen 12" in sem.cordura(558, d_12)[0],
   "558 entradas: se pueden ir 11 (2 %), 12 no")
ok(sem.cordura(558, d_51) and "cambian 51" in sem.cordura(558, d_51)[0] and len(d_51["altas"]) == 51,
   "51 altas de golpe: se detiene")
ok(sem.comparar(viejo, dict(viejo))["total"] == 0, "sin novedades, cero cambios")
d_c = sem.comparar({"1": {"estado": "abandonada", "desde": "2022"}}, {"1": {"estado": "abandonada", "desde": "2023"}})
ok(d_c["cambios"] == ["1"] and d_c["campos"]["1"] == ["desde"], "el diff dice qué campo cambió")


def _volcado(ruta, tesis):
    relleno = [v3(3000000 + i, f"III.{i}o.C.1 C (10a.)", f"RELLENO {i} SIN NOTA NINGUNA.") for i in range(1000)]
    ruta.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in tesis + relleno), encoding="utf-8")


def _semanal(d, indice, *extra):
    return subprocess.run([sys.executable, str(RAIZ / "scripts" / "vigencia_semanal.py"), "--seco",
                           "--sin-actualizar", "--reusar-volcado", "--dir", str(d), "--indice", str(indice),
                           "--env", str(Path(d, "no_hay.env")), "--sin-log", "--pruebas", *extra],
                          capture_output=True, text=True, timeout=300)


with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    _volcado(d / "tesis_v3.jsonl", T_ACERVO)
    for f in (N1, N2):
        guardar_ficha(d / "sjf_cache", f)
    real = RAIZ / "datos" / "vigencia_tesis.json"
    indice = d / "indice.json"
    shutil.copy(real, indice)
    antes = indice.read_bytes()
    r = _semanal(d, indice)
    ok(r.returncode == 3 and "CORDURA" in r.stdout and indice.read_bytes() == antes,
       f"de 558 a 4 entradas: la corrida se detiene (código {r.returncode}) y el índice no se toca")
    diff = json.loads((d / "ultimo_diff.json").read_text())
    ok(len(diff["diff"]["bajas"]) > 500 and ("2009817" in diff["altas"] or "2009817" in diff["cambios"]),
       "y deja el diff por registro en <dir>/ultimo_diff.json para que lo mire una persona")

# ═══════════════════════════════════════════════════════════════ 6 · idempotencia
print("\n6 · SIN NOVEDADES, EL MISMO JSON")
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    _volcado(d / "tesis_v3.jsonl", T_ACERVO)
    for f in (N1, N2):
        guardar_ficha(d / "sjf_cache", f)
    indice = d / "indice.json"
    g = subprocess.run([sys.executable, str(RAIZ / "scripts" / "vigencia_tesis_generar.py"), "--sjf-cache",
                        str(d / "sjf_cache"), "--tesis-cache", str(d / "tesis_v3.jsonl"), "--salida", str(indice)],
                       capture_output=True, text=True, timeout=300)
    antes = indice.read_bytes() if indice.exists() else b""
    r = _semanal(d, indice)
    ok(g.returncode == 0 and r.returncode == 0 and "sin cambios en el índice" in r.stdout and indice.read_bytes() == antes,
       f"la corrida sin novedades termina en 0, «sin cambios», y no reescribe el índice (ni la fecha) "
       f"(rc={r.returncode})")
    r2 = _semanal(d, indice)
    ok(r2.returncode == 0 and indice.read_bytes() == antes, "y una segunda vez, igual")

# La actualización de la caché sin novedades tampoco cambia el índice: el SJF
# devuelve lo mismo que ya había, no hay registros nuevos y el repaso avanza.
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    cache = d / "sjf_cache"
    tesis = [v3(2032214, "PR.A.C.CS. J/5 C (12a.)", "LA ÚLTIMA DEL ACERVO.", tipo="JURISPRUDENCIA",
                fecha="2026-05-29")] + T_SJF
    fichas = {"2032214": ficha(2032214, "PR.A.C.CS. J/5 C (12a.)", "LA ÚLTIMA DEL ACERVO.", tj=1),
              "2024159": ficha(2024159, "P./J. 2/2022 (11a.)", "TESTIGO.", tj=1),
              "2019978": ficha(2019978, "2a. XL/2019 (10a.)", "X.", prec="Amparo en revisión 1/2019. 1 de marzo de "
                               "2019.\n\nNota: La presente tesis fue abandonada por la tesis 2a./J. 7/2026 (12a.)."),
              "2031000": ficha(2031000, "2a./J. 7/2026 (12a.)", "LA JURISPRUDENCIA QUE LA ABANDONA.", prec="", tj=1),
              "2019979": ficha(2019979, "2a. XLI/2019 (10a.)", "X.", prec=T_SJF[2]["precedentes"])}
    guardar_ficha(cache, fichas["2019978"])
    gen.SJF_CACHE = str(cache)
    base = gen.compacto(gen.generar(tesis, gen.cargar_nuevas_sjf(str(cache), {t["registro"] for t in tesis}))[0])
    estado = sem.estado_vacio()
    sjf = SJFFalso(fichas)
    D = descargador(cache, sjf)
    res = sem.actualizar_cache(D, tesis, estado, cache, {"tesis": base}, {}, hoy, fallos=3, saltos=(10,),
                               repaso=3, repaso_calientes=1)
    despues = gen.compacto(gen.generar(tesis, gen.cargar_nuevas_sjf(str(cache), {t["registro"] for t in tesis}))[0])
    ok(despues == base and res["nuevas"]["bajadas"] == 0 and estado["ultimo_secuencial"] == 2032214,
       f"sin tesis nuevas y con el SJF igual: el mismo índice ({len(base)} entradas) y la frontera en 2032214")
    ok(res["repaso"]["ok"] == 3 and len(estado["repaso"]["calientes"]["hechos"]) == 1
       and len(estado["repaso"]["general"]["hechos"]) == 2 and res["repaso"]["cambiadas"] == [],
       f"el repaso avanza (1 caliente + 2 generales) y no ve notas distintas ({res['repaso']})")
    res2 = sem.actualizar_cache(D, tesis, estado, cache, {"tesis": base}, {}, hoy, fallos=3, saltos=(10,),
                                repaso=3, repaso_calientes=1)
    todos = set(estado["repaso"]["general"]["hechos"]) | set(estado["repaso"]["calientes"]["hechos"])
    ok(len(todos) == 4 and res2["nuevas"]["desde"] == 2032215, "la segunda corrida sigue donde se quedó")
    # Y cuando el SJF sí publica: la nueva abandona la P. X/2015 y su nota llega a la vieja al rebajarla.
    tesis2 = tesis + ACERVO
    sjf.fichas["2032215"] = ficha(2032215, "P./J. 7/2026 (12a.)",
                              "CONTROL DE REGULARIDAD [ABANDONO DE LA TESIS AISLADA P. X/2015 (10a.)].", tj=1)
    sjf.fichas["2009817"] = ficha(2009817, "P. X/2015 (10a.)", "X.", prec="Amparo directo en revisión 1046/2012.\n\n"
                              "Nota: La presente tesis fue abandonada por la tesis P./J. 7/2026 (12a.).")
    res3 = sem.actualizar_cache(D, tesis2, estado, cache, {"tesis": base}, {}, hoy, fallos=3, saltos=(10,),
                                repaso=0, repaso_calientes=0)
    ok(res3["nuevas"]["bajadas"] == 1 and res3["afectadas"]["afectadas"] == 1
       and res3["afectadas"]["cambiadas"] == ["2009817"] and Path(cache, "2009817.json").exists(),
       f"una tesis nueva: se baja, se lee su «[ABANDONO …]» y se REBAJA la afectada, que trae nota nueva "
       f"({res3['afectadas']['cambiadas']})")
    idx3 = gen.generar(tesis2, gen.cargar_nuevas_sjf(str(cache), {t["registro"] for t in tesis2}))[0]
    ok(idx3["2009817"]["fuente"] == "nota_propia" and idx3["2009817"]["por_registro"] == "2032215"
       and idx3["2009817"]["por_en_acervo"] is False,
       "y en el índice manda la nota propia de la afectada, con la nueva como reemplazo")
    gen.SJF_CACHE = None

# ═══════════════════════════════════════════════════════════════ 6b · el .env de las pruebas y el seco
print("\n6b · LAS PRUEBAS LLEVAN EL .env; EN SECO NO SE TOCA EL ÍNDICE")
with tempfile.TemporaryDirectory() as d:
    # El worktree dedicado es HERMANO del repo: main.py no halla ningún .env al
    # importarse y test_vigencia_tesis.py tronaba («Missing credentials»).
    d = Path(d)
    (d / "prueba_env.py").write_text(
        "import os, sys\n"
        "sys.exit(0 if os.environ.get('CLAVE_DE_PRUEBA') == 'secreto de juguete'"
        " and os.environ.get('VIGENCIA_SEMANAL_DENTRO') == '1' else 1)\n")
    (d / "juguete.env").write_text("# comentario\nexport CLAVE_DE_PRUEBA=\"secreto de juguete\"\nOTRA=1\n")
    ev = sem.vars_de_env(d / "juguete.env")
    ok(ev.get("CLAVE_DE_PRUEBA") == "secreto de juguete" and ev.get("OTRA") == "1" and sem.vars_de_env(d / "no") == {},
       "vars_de_env lee el .env (comillas, export, comentarios); sin archivo, nada")
    try:
        sem.correr_pruebas(d, ["prueba_env.py"], ev)
        con_env = True
    except sem.FalloPruebas:
        con_env = False
    _guardada = os.environ.pop("CLAVE_DE_PRUEBA", None)
    try:
        sem.correr_pruebas(d, ["prueba_env.py"], {})
        sin_env = True
    except sem.FalloPruebas:
        sin_env = False
    if _guardada is not None:
        os.environ["CLAVE_DE_PRUEBA"] = _guardada
    ok(con_env and not sin_env, "las pruebas corren con las variables del --env (sin ellas, fallan)")

with tempfile.TemporaryDirectory() as d:
    # Corrido a mano desde un checkout que no es el dedicado, --seco sin
    # --indice dejaba modificado su datos/vigencia_tesis.json versionado.
    d = Path(d)
    raiz = d / "checkout"
    (raiz / "scripts").mkdir(parents=True)
    (raiz / "datos").mkdir()
    for s in ("vigencia_semanal.py", "vigencia_tesis_generar.py", "sjf_cache_descargar.py"):
        shutil.copy(RAIZ / "scripts" / s, raiz / "scripts" / s)
    shutil.copy(RAIZ / "vigencia_tesis.py", raiz / "vigencia_tesis.py")
    trabajo = d / "vig"
    trabajo.mkdir()
    _volcado(trabajo / "tesis_v3.jsonl", T_ACERVO)
    guardar_ficha(trabajo / "sjf_cache", N1)
    g = subprocess.run([sys.executable, str(raiz / "scripts" / "vigencia_tesis_generar.py"), "--sjf-cache",
                        str(trabajo / "sjf_cache"), "--tesis-cache", str(trabajo / "tesis_v3.jsonl"), "--salida",
                        str(raiz / "datos" / "vigencia_tesis.json")], capture_output=True, text=True, timeout=300)
    guardar_ficha(trabajo / "sjf_cache", N2)          # la semana siguiente: la 2a./J. 30/2026 interrumpe otra
    antes = (raiz / "datos" / "vigencia_tesis.json").read_bytes()
    r = subprocess.run([sys.executable, str(raiz / "scripts" / "vigencia_semanal.py"), "--seco", "--sin-actualizar",
                        "--reusar-volcado", "--dir", str(trabajo), "--env", str(d / "no_hay.env"), "--sin-log",
                        "--worktree", str(d / "wt-no-existe"), "--pruebas"],
                       capture_output=True, text=True, timeout=300)
    prop = trabajo / "propuesta" / "vigencia_tesis.json"
    ok(g.returncode == 0 and r.returncode == 0 and (raiz / "datos" / "vigencia_tesis.json").read_bytes() == antes
       and prop.exists() and "2019000" in json.loads(prop.read_text())["tesis"]
       and "la propuesta queda en" in r.stdout,
       f"--seco sin --indice: la propuesta (con la interrumpida nueva) queda en <dir>/propuesta y el índice del "
       f"checkout, intacto (rc={r.returncode}) {r.stdout[-300:] if r.returncode else ''}")

# ═══════════════════════════════════════════════════════════════ 7 · el LaunchAgent y el arranque
print("\n7 · EL LAUNCHAGENT Y EL ARRANQUE")
PL = RAIZ / "scripts" / "launchd" / "com.iurexia.vigencia-semanal.plist"
p = plistlib.loads(PL.read_bytes())
ok(p["Label"] == "com.iurexia.vigencia-semanal" and p["StartCalendarInterval"] == {"Weekday": 0, "Hour": 3,
                                                                                   "Minute": 30},
   "domingo 03:30 con StartCalendarInterval (corre al despertar si la Mac dormía)")
ok(".venv/bin" in p["EnvironmentVariables"]["PATH"] and "/usr/bin" in p["EnvironmentVariables"]["PATH"]
   and not p.get("RunAtLoad") and "StandardOutPath" not in p and "StandardErrorPath" not in p,
   "un PATH con git y el .venv, no corre al cargarse y el registro lo abre el lanzador (sin StandardOutPath: "
   "launchd no arranca si falta la carpeta)")
LANZADOR = p["ProgramArguments"][2] if p["ProgramArguments"][:2] == ["/bin/zsh", "-c"] else ""
ok(LANZADOR and subprocess.run(["zsh", "-n", "-c", LANZADOR]).returncode == 0
   and "show origin/main:scripts/vigencia_semanal.sh" in LANZADOR
   and "wt-vigencia-semanal/scripts" not in LANZADOR and "osascript" in LANZADOR,
   "el plist lleva el lanzador dentro: saca el .sh de origin/main (no de un worktree que quizá no existe) y avisa")
_com = PL.read_text(encoding="utf-8").split("<!--", 1)[1].split("-->", 1)[0]
ok("launchctl bootstrap gui/$(id -u)" in _com and "launchctl bootout gui/$(id -u)/com.iurexia.vigencia-semanal" in _com
   and "preparar" in _com and "--" not in _com,
   "el comentario trae la instalación y la desinstalación, un comando cada una (y sin «--», que XML no admite)")
SH = (RAIZ / "scripts" / "vigencia_semanal.sh").read_text(encoding="utf-8")
ok(subprocess.run(["zsh", "-n", str(RAIZ / "scripts" / "vigencia_semanal.sh")]).returncode == 0,
   "vigencia_semanal.sh: sintaxis de zsh")
_codigo_sh = [l for l in SH.splitlines() if not l.lstrip().startswith("#")]
ok("worktree add -q --detach" in SH and "lockf -t 0" in SH and "reset -q --hard origin/main" in SH
   and not any(re.search(r"\bgit\b.*\bpush\b", l) for l in _codigo_sh)
   and not any("--force" in l and "worktree remove" not in l for l in _codigo_sh),
   "el arranque: candado, worktree desprendido en origin/main; el push lo hace el Python, nunca --force")
PYS = (RAIZ / "scripts" / "vigencia_semanal.py").read_text(encoding="utf-8")
_llamadas_push = re.findall(r'git\(raiz, "push"[^)]*\)', PYS)
ok(_llamadas_push and all(c == 'git(raiz, "push", "-q", "origin", "HEAD:main")' for c in _llamadas_push)
   and not re.search(r'"(--force|--force-with-lease|-f|\+HEAD:main)"', PYS),
   f"el push es por fast-forward a main ({len(_llamadas_push)} llamadas) y no hay --force en ninguna parte")

# El arranque, de verdad, contra un repo de juguete (nada del repo real): crea
# el worktree dedicado con su marca y su .env, limpia lo suelto, respeta el
# candado, devuelve el código del Python y se niega a tocar el checkout
# principal, un clon ajeno o un worktree enlazado que no lleva la marca. Y el
# lanzador del plist, que funciona aunque el worktree no exista todavía.
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    G = ["git", "-c", "user.name=prueba", "-c", "user.email=prueba@example.invalid", "-c", "init.defaultBranch=main"]

    def _g(*a):
        return subprocess.run(G + list(a), capture_output=True, text=True, timeout=60)
    _g("init", "-q", "--bare", str(d / "origin.git"))
    _g("clone", "-q", str(d / "origin.git"), str(d / "semilla"))
    (d / "semilla" / "scripts").mkdir(parents=True)
    (d / "semilla" / "scripts" / "vigencia_semanal.py").write_text("# de juguete\n")
    (d / "semilla" / "x.txt").write_text("limpio\n")
    (d / "semilla" / ".gitignore").write_text(".env\n")          # como el repo de verdad
    _g("-C", str(d / "semilla"), "add", "-A")
    _g("-C", str(d / "semilla"), "commit", "-q", "-m", "inicial")
    _g("-C", str(d / "semilla"), "push", "-q", "origin", "HEAD:main")
    _g("clone", "-q", str(d / "origin.git"), str(d / "repo"))
    (d / "juguete.env").write_text("CLAVE_DE_JUGUETE=1\n")
    py = d / "py.sh"
    # PY_MARCA: avisa que arrancó (con el candado ya tomado); PY_SOLTAR: no
    # termina hasta que exista ese archivo. Así la prueba del candado no
    # depende de cuánto tarde nadie en arrancar.
    py.write_text('#!/bin/zsh\necho "PY $*"\necho "SIN_AVISO=${VIGENCIA_SIN_AVISO:-}"\n'
                  '[[ -n "${PY_MARCA:-}" ]] && : >| "$PY_MARCA"\n'
                  'if [[ -n "${PY_SOLTAR:-}" ]]; then\n'
                  '  for i in {1..1200}; do [[ -e "$PY_SOLTAR" ]] && break; sleep 0.05; done\nfi\n'
                  'exit "${PY_RC:-0}"\n')
    py.chmod(0o755)
    base_env = {**os.environ, "VIGENCIA_REPO": str(d / "repo"), "VIGENCIA_WT": str(d / "wt"),
                "VIGENCIA_DIR": str(d / "dir"), "VIGENCIA_LOG": str(d / "log.txt"), "VIGENCIA_PY": str(py),
                "VIGENCIA_ENV": str(d / "juguete.env"), "VIGENCIA_SIN_AVISO": "1", "VIGENCIA_ESPERA_RED": "0"}
    base_env.pop("VIGENCIA_CON_CANDADO", None)
    SHP = str(RAIZ / "scripts" / "vigencia_semanal.sh")

    def _sh(*args, **env):
        return subprocess.run(["zsh", SHP, *args], env={**base_env, **env}, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=120)

    def _log():
        return (d / "log.txt").read_text() if (d / "log.txt").exists() else ""

    def _ultima():
        return _log().split("══ vigencia semanal · arranque")[-1]

    def _lanzar(*args, **env):
        # VIGENCIA_SIN_AVISO siempre puesto: una prueba nunca manda una notificación de verdad.
        return subprocess.run(["/bin/zsh", "-c", LANZADOR, "lanzador", *args], env={**base_env, **env},
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)

    rl = _lanzar(VIGENCIA_WT=str(d / "wt_l"))
    ok(rl.returncode == 1 and "✗ lanzador: no pude sacar scripts/vigencia_semanal.sh de origin/main" in _log()
       and "(aviso)" in _log() and not (d / "wt_l").exists(),
       f"el lanzador, si origin/main aún no trae el .sh: sale 1, lo escribe y AVISA (antes: 127 en silencio) "
       f"(rc={rl.returncode})")
    r1 = _sh("--seco")
    wt_ok = (d / "wt" / "scripts" / "vigencia_semanal.py").exists()
    rama = subprocess.run(["git", "-C", str(d / "wt"), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True,
                          text=True).stdout.strip() if wt_ok else ""
    gd_wt = subprocess.run(["git", "-C", str(d / "wt"), "rev-parse", "--absolute-git-dir"], capture_output=True,
                           text=True).stdout.strip() if wt_ok else ""
    ok(r1.returncode == 0 and wt_ok and rama == "HEAD" and "--sin-log --seco" in _log()
       and f"--worktree {d / 'wt'}" in _log(),
       "la primera vez crea el worktree dedicado, desprendido en origin/main, y le pasa --seco al Python")
    ok(wt_ok and Path(gd_wt, sem.MARCA).is_file(),
       "… y le deja su marca en la carpeta de administración (.git/worktrees/<nombre>/)")
    ok(wt_ok and (d / "wt" / ".env").is_symlink() and os.readlink(d / "wt" / ".env") == str(d / "juguete.env"),
       "… y le enlaza el .env del repo (un enlace, no una copia de los secretos)")
    if wt_ok:
        (d / "wt" / "x.txt").write_text("sucio\n")
        (d / "wt" / "suelto.txt").write_text("suelto\n")
    r2 = _sh()
    ok(r2.returncode == 0 and wt_ok and (d / "wt" / "x.txt").read_text() == "limpio\n"
       and not (d / "wt" / "suelto.txt").exists() and (d / "wt" / ".env").is_symlink(),
       "cada corrida deja el worktree EXACTAMENTE en origin/main (y el .env sigue enlazado)")
    if os.environ.get("VIGENCIA_SEMANAL_DENTRO"):
        # Dentro de la corrida semanal esta prueba es compuerta del commit: no
        # se juega la semana en ella (el candado se probó al instalar).
        print("   SALTA  el candado con dos corridas a la vez (dentro de la corrida semanal)")
    else:
        marca, soltar = d / "py.arranco", d / "py.soltar"
        primera = subprocess.Popen(["zsh", SHP], env={**base_env, "PY_MARCA": str(marca), "PY_SOLTAR": str(soltar)},
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import time as _t
        for _ in range(1200):                      # hasta 60 s a que la primera tenga el candado
            if marca.exists() or primera.poll() is not None:
                break
            _t.sleep(0.05)
        tenia = marca.exists()
        r3 = _sh()
        soltar.touch()
        primera.wait(timeout=60)
        ok(tenia and r3.returncode == 75 and primera.returncode == 0 and "ya hay otra corrida" in _log(),
           f"con una corrida en marcha, la segunda sale con 75 sin tocar nada (rc={r3.returncode}, "
           f"la primera arrancó: {tenia})")
    ok(_sh(PY_RC="3").returncode == 3, "el código del Python (3: la cordura detuvo) es el del arranque")
    n_py = _log().count("PY ")
    rp = _sh("preparar")
    ok(rp.returncode == 0 and "✓ listo: worktree dedicado" in _ultima() and _log().count("PY ") == n_py,
       "«preparar» (la instalación) deja listo el worktree y NO corre el Python")
    ok(_sh(VIGENCIA_ENV=str(d / "no.env")).returncode == 1 and "no está el .env" in _ultima(),
       "sin .env no arranca (sin él no hay Qdrant ni pasan las pruebas)")
    ok(_sh(VIGENCIA_WT=str(d / "repo")).returncode != 0, "se niega a usar el checkout principal como worktree")
    _g("clone", "-q", str(d / "origin.git"), str(d / "ajeno"))
    (d / "ajeno" / "x.txt").write_text("de otra sesión\n")
    ok(_sh(VIGENCIA_WT=str(d / "ajeno")).returncode == 5 and (d / "ajeno" / "x.txt").read_text() == "de otra sesión\n",
       "y a tocar un checkout que no es un worktree enlazado de este repo")
    _g("-C", str(d / "repo"), "worktree", "add", "-q", "--detach", str(d / "otra-sesion"), "origin/main")
    (d / "otra-sesion" / "x.txt").write_text("trabajo de otra sesión\n")
    ro = _sh("--seco", VIGENCIA_WT=str(d / "otra-sesion"))
    ok(ro.returncode == 5 and (d / "otra-sesion" / "x.txt").read_text() == "trabajo de otra sesión\n"
       and "no lleva la marca del dedicado" in _ultima(),
       "ni un worktree enlazado del mismo repo que no lleva la marca (el de otra sesión): no lo resetea")

    # El lanzador del plist, con el .sh ya en origin/main y SIN el worktree:
    shutil.copy(SHP, d / "semilla" / "scripts" / "vigencia_semanal.sh")
    _g("-C", str(d / "semilla"), "add", "-A")
    _g("-C", str(d / "semilla"), "commit", "-q", "-m", "el arranque")
    _g("-C", str(d / "semilla"), "push", "-q", "origin", "HEAD:main")
    rl2 = _lanzar("--seco", VIGENCIA_WT=str(d / "wt_l"), VIGENCIA_SIN_AVISO="del lanzador")
    gd_l = subprocess.run(["git", "-C", str(d / "wt_l"), "rev-parse", "--absolute-git-dir"], capture_output=True,
                          text=True).stdout.strip() if (d / "wt_l").exists() else ""
    ok(rl2.returncode == 0 and (d / "wt_l" / "scripts" / "vigencia_semanal.sh").exists()
       and gd_l and Path(gd_l, sem.MARCA).is_file() and f"--worktree {d / 'wt_l'} " in _ultima()
       and "--seco" in _ultima() and "SIN_AVISO=1" in _ultima(),
       "el lanzador saca el .sh de origin/main y lo corre aunque el worktree no exista: lo crea con su marca "
       "(y el .sh corre sin avisar: avisa el lanzador)")
    rl3 = _lanzar(VIGENCIA_WT=str(d / "wt_l"), PY_RC="6")
    ok(rl3.returncode == 6 and "✗ lanzador: salió con código 6" in _log(),
       f"si el .sh sale ≠ 0, el lanzador devuelve ese código, lo escribe y avisa (rc={rl3.returncode})")


print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}:")
    for f in FALLOS:
        print("  ·", f)
    sys.exit(1)
print("TODO PASA")
