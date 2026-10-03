# -*- coding: utf-8 -*-
"""EL CABLEADO DE LA PROCEDENCIA POR TIPO (3-oct-2026), sin red.

    .venv/bin/python test_cableado_procedencia.py

Con la bandera `procedencia_por_tipo`, el adelanto NO llama a
`redactar_estructura` (la llamada que escribía el V I S T O y los resultandos a
ciegas): lee el acto para la ficha de trámite a la vez que las fases 1-3, arma
la ficha con el auto y lo confirmado por el secretario, y la `Estructura` sale
del compositor por tipo. La ficha viaja en el encargo (y en la sesión), y el
proyecto se recompone de ella sin volver al modelo, también en el worker que no
leyó el acto. Sin la bandera, el camino de hoy, idéntico.

Las piezas nuevas que escriben otros (`ficha_tramite`, `resultandos_por_tipo`)
se sustituyen por DOBLES con las firmas del §6 de la especificación; el modelo
(fases 1-3, partes, estructura, síntesis) también. Se prueba el montaje real
del redactor: `generar` y `_componer_generado` enteros, no piezas sueltas.

SEGUNDA RONDA (3-oct-2026), secciones 10 a 13: /taller/estado dice si rige la
bandera para la cuenta; la AUTORIDAD que recurre en la revisión o en la queja
se computa con su regla (oficio, art. 31, fr. I) cuando llegó la de los
particulares, y la regla queda en el encargo para el otro worker;
/taller/reglas-surtimiento acepta `papel`; el .docx se descarga como
«NNN-AAAA PROCEDENCIA.docx»; y los tres campos nuevos de la ficha
(fraccion_63, cuantia, fundamento_surtimiento) viajan sin filtrarse. Las
funciones de main.py se ejecutan sacadas del archivo (sin importar main ni
arrancar el API).

TERCERA RONDA (3-oct-2026), secciones 14 a 20 (dueño F de FIXES_R3.md):
quién recurre se LEE DEL ESCRITO cuando el formulario no lo dice (AR 631/2025
de punta a punta) y su carácter queda fijo; la autoridad que recurre se
computa con «oficio» TAMBIÉN SIN BANDERA (D2) y, con ella, la forma de
notificación que consta (electrónica, lista, oficio) manda sobre la regla de
omisión; la revisión fiscal por correo se mide con el DEPÓSITO (D3), en el
adelanto y en el worker que rehidrata la sesión; al retomar, lo leído vuelve
como leído y no como del secretario; el auto se lee acotado y en un hilo;
/taller/desde-expediente recibe `tramite_json`; y el docstring de
/taller/reglas-surtimiento dice lo que hace (D4).

CUARTA RONDA (3-oct-2026), secciones 23 a 25 (dueño F de FIXES_R4.md): el
correo con UNA regla (E6, `ficha_tramite.via_postal`: con depósito anterior a
la presentación cuenta el depósito aunque la vía diga «responsable», en el
adelanto y en el otro worker; la sección 16 cambia con ella); el auto de
Presidencia que forma y registra (`fecha_registro`, E7) viaja por
`tramite_json`, con la sesión y al retomar, y el camino SISE avisa si dos
autos dan fechas de admisión distintas; y el cargo del ponente leído del auto
llega a la carátula dentro del valor del ponente (E3), también desde
/taller/desde-admision.

SEXTA RONDA (3-oct-2026, visto bueno de David para todos los usuarios): la
sección 1 (y la 10) esperan la bandera «todos» por omisión (C8: «0» la apaga,
«casa» la limita a las cuentas de casa); los dobles llevan las 22 claves; y la
sección 26 prueba que «relacionados» —los asuntos que el secretario marca con
un clic (C6)— viaja como «fecha_registro»: por tramite_json, en la ficha, al
compositor y a documento_generado, con la sesión al otro worker y, al
retomar, de vuelta a la tarjeta como del secretario.
"""
import ast
import asyncio
import copy
import datetime
import json
import os
import sys
import tempfile
import types

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
os.environ.pop("PROCEDENCIA_POR_TIPO", None)

import contexto_taller as ct
import documento_generado as dg
import ensamblar_adelanto as ens
import fase_partes as fpartes
import fases123_pipeline as f123
import redactor_adelanto as ra

FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


# ═══ LOS DOBLES ═════════════════════════════════════════════════════════════
LL = {"redactar_estructura": 0, "leer_acto": 0, "leer_auto": 0, "armar": [],
      "componer_tipo": [], "dg_componer": [], "orden": [], "demanda": None, "form": None,
      # Tercera ronda: la lectura del escrito (quién recurre) y el texto del auto.
      "leer_escrito": 0, "escrito_args": None, "auto_texto": None}
# Los tres campos de la ficha que estrena la segunda ronda (3-oct-2026): la
# fracción del 63 LFPCA y la cuantía (revisión fiscal) y el precepto que rige el
# surtimiento. El cableado sólo tiene que dejarlos pasar, sin filtrarlos.
NUEVAS = ("fraccion_63", "cuantia", "fundamento_surtimiento")

ft = types.ModuleType("ficha_tramite")
ft.FORMATO = 1


async def _leer_acto(cliente, texto_acto, tipo, numero="", *, demanda=""):
    LL["leer_acto"] += 1
    LL["demanda"] = demanda
    return {"acto": {"clase": "sentencia", "fecha": "2026-01-15",
                     "organo": "Primera Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro",
                     "toca": "374/2025", "expediente": "905/2024"},
            "fuentes": {"acto.fecha": "acto"}, "avisos": ["AVISO DEL ACTO"]}


def _leer_auto(texto, tipo=""):
    LL["leer_auto"] += 1
    LL["auto_texto"] = texto
    if "TURNO" in texto:
        return {"turno": {"fecha": "2026-02-12", "ponente": "Magistrada Ana Pérez"},
                "admision": {"fecha": "2026-02-12"},
                "fuentes": {"turno.fecha": "auto", "admision.fecha": "auto"},
                "avisos": ["AVISO DEL TURNO"]}
    return {"admision": {"fecha": "2026-02-10"}, "turno": {"fecha": "2026-02-11", "ponente": "X"},
            "ministerio_publico": "", "fuentes": {"admision.fecha": "auto", "turno.fecha": "auto"},
            "avisos": ["AVISO DEL AUTO"]}


def _de_formulario(form):
    LL["form"] = dict(form)
    d = {"fuentes": {}}
    for k in NUEVAS:
        if form.get(k):
            d[k] = form[k]
            d["fuentes"][k] = "secretario"
    if form.get("forma_notificacion"):
        d["forma_notificacion"] = form["forma_notificacion"]
        d["fuentes"]["forma_notificacion"] = "secretario"
    if form.get("fecha_admision"):
        d["admision"] = {"fecha": form["fecha_admision"]}
        d["fuentes"]["admision.fecha"] = "secretario"
    # Cuarta ronda (E7): el auto de Presidencia que forma y registra.
    if form.get("fecha_registro"):
        d["registro"] = {"fecha": form["fecha_registro"]}
        d["fuentes"]["registro.fecha"] = "secretario"
    if form.get("ministerio_publico"):
        d["ministerio_publico"] = form["ministerio_publico"]
        d["fuentes"]["ministerio_publico"] = "secretario"
    for k in ("deposito_postal", "presentacion", "sentido"):
        if form.get(k):
            d[k] = form[k]
            d["fuentes"][k] = "secretario"
    # Sexta ronda (C6): los asuntos relacionados, texto «tipo|numero|estado;…».
    if form.get("relacionados"):
        d["relacionados"] = [dict(zip(("tipo", "numero", "estado"), (x.split("|") + ["misma_sesion"])[:3]))
                             for x in form["relacionados"].split(";") if x]
        d["fuentes"]["relacionados"] = "secretario"
    return d


def _armar(encargo, *, auto=None, acto=None, ficha_procesal=None, partes=None, declarado=None,
           escrito=None):
    LL["armar"].append({"auto": auto, "acto": acto, "ficha_procesal": ficha_procesal,
                        "declarado": declarado, "partes": partes, "escrito": escrito,
                        # La regla con que el encargo llega a la ficha: la de la
                        # autoridad, si recurre una (se rehace ANTES de armar).
                        "regla": getattr(encargo, "regla_surtimiento", None)})
    LL["orden"].append("armar")
    if getattr(ft, "ROMPER_ARMAR", False):
        raise RuntimeError("armar roto a propósito")
    adm = ((declarado or {}).get("admision") or (auto or {}).get("admision") or {})
    # Cuarta ronda (E7): el registro, con la misma precedencia que la admisión.
    reg = ((declarado or {}).get("registro") or (auto or {}).get("registro") or {})
    return {"formato": 1, "tipo": ra._tipo_de_encargo(encargo), "numero": encargo.numero,
            "admision": adm, "turno": (auto or {}).get("turno") or {},
            **({"registro": reg} if reg else {}),
            "acto": (acto or {}).get("acto") or {},
            # UNA FECHA DE VERDAD, para comprobar que la sesión la guarda en ISO.
            "registrada": datetime.date(2026, 2, 10),
            "fuentes": {"admision.fecha": "secretario" if (declarado or {}).get("admision") else "auto"},
            "avisos": ["AVISO DE LA FICHA"], "fechas_imposibles": [],
            # Lo declarado de los campos nuevos pasa a la ficha (lo decide el
            # armar de verdad; el doble sólo lo copia).
            **{k: (declarado or {})[k] for k in NUEVAS if (declarado or {}).get(k)},
            # Tercera ronda: el depósito postal y la vía (RF por correo).
            **{k: (declarado or {})[k] for k in ("deposito_postal", "via_presentacion")
               if (declarado or {}).get(k)},
            # Sexta ronda (C6): los relacionados sólo salen de lo declarado.
            "relacionados": list((declarado or {}).get("relacionados") or [])}


# LA LECTURA DEL ESCRITO (tercera ronda): la firma que se pidió a la pieza de la
# ficha, `async leer_escrito(cliente, texto_escrito, tipo)`. Lo que devuelve lo
# fija `ft.ESCRITO` (un dict, None o una excepción que lanza).
async def _leer_escrito(cliente, texto_escrito, tipo):
    LL["leer_escrito"] += 1
    LL["escrito_args"] = (texto_escrito, tipo)
    if isinstance(ft.ESCRITO, BaseException):
        raise ft.ESCRITO
    return copy.deepcopy(ft.ESCRITO)


def _a_formulario(ficha):
    return {"fecha_admision": ((ficha or {}).get("admision") or {}).get("fecha", "")}


def _sede_de(tribunal, ciudad):
    return {"tribunal": tribunal, "ciudad": ciudad, "circuito": "XXII", "cdmx": False}


ft.leer_acto, ft.leer_auto, ft.de_formulario = _leer_acto, _leer_auto, _de_formulario
ft.armar, ft.a_formulario, ft.sede_de = _armar, _a_formulario, _sede_de
ft.leer_escrito, ft.ESCRITO = _leer_escrito, None
# Las 22 claves del contrato con la pantalla (las de ficha_tramite.CLAVES_FORMULARIO):
# la cuarta ronda (E7) añade «fecha_registro», el auto que forma y registra; la
# sexta (C6), «relacionados», los asuntos que el secretario marca con un clic.
ft.CLAVES_FORMULARIO = (
    "fecha_admision", "fecha_turno", "ponente_turno", "fecha_returno",
    "ponente_returno", "ministerio_publico", "adhesivo_quien",
    "adhesivo_presentacion", "adhesivo_admision", "adhesivo_notificacion",
    "fecha_acto", "organo_acto", "toca", "expediente_origen",
    "fecha_informe_101", "deposito_postal", "forma_notificacion",
    "fraccion_63", "cuantia", "fundamento_surtimiento", "fecha_registro",
    "relacionados")
ft.ROMPER_ARMAR = False
sys.modules["ficha_tramite"] = ft

rpt = types.ModuleType("resultandos_por_tipo")
VISTO_COMPUESTO = "para resolver el juicio de amparo directo civil 512/2026, promovido por Ana; y,"
TITULOS_COMPUESTOS = ["Presentación de la demanda de amparo.", "Trámite del juicio de amparo.", "Turno."]


def _componer_tipo(tipo, ficha, datos):
    LL["componer_tipo"].append({"tipo": tipo, "ficha": ficha, "datos": dict(datos)})
    if rpt.FALLAR:
        raise RuntimeError("compositor roto a propósito")
    if rpt.VACIO:
        # Lo que hace el compositor de verdad cuando falla por dentro: NO lanza,
        # devuelve todo vacío con el aviso para que el cableado caiga al viejo.
        return {"visto": "", "resultandos": [], "datos_extra": {},
                "avisos": ["EL COMPOSITOR POR TIPO FALLÓ (KeyError: 'x'); se compone por el camino anterior."]}
    return {"visto": VISTO_COMPUESTO,
            "resultandos": [{"titulo": t, "texto": f"Texto de {t}"} for t in TITULOS_COMPUESTOS],
            "avisos": ["AVISO DEL COMPOSITOR"],
            "datos_extra": {"expediente": "905/2024", "toca": "374/2025",
                            "fecha_acto_iso": "2026-01-15"}}


rpt.componer, rpt.FALLAR, rpt.VACIO = _componer_tipo, False, False
sys.modules["resultandos_por_tipo"] = rpt


async def _redactar_estructura(cliente, datos):
    LL["redactar_estructura"] += 1
    return dg.Estructura(visto="VISTO DEL MODELO",
                         resultandos=[{"titulo": "Del modelo.", "texto": "prosa del modelo"}],
                         competencia="competencia del modelo", avisos=["AVISO DEL MODELO"])


def _dg_componer(datos, estructura, computo, fecha_en_letra, ruta_salida, **kw):
    LL["dg_componer"].append({"datos": dict(datos), "estructura": estructura, "computo": computo})
    import docx
    docx.Document().save(ruta_salida)
    return ruta_salida


async def _correr(cliente, texto_acto, texto_conceptos, es_recurso=False, tipo_asunto="",
                  quejoso="", responsable=""):
    f = f123.Fases123()
    f.antecedentes = "Primero. Antecedente del asunto."
    f.resumen_acto = "La Sala confirmó la sentencia."
    f.resumen_conceptos = "La quejosa alega falta de fundamentación."
    f.problemas = [{"pregunta": "¿Está fundada la sentencia?", "jerarquia": "principal"}]
    return f


async def _fichar(cliente, texto_acto, texto_conceptos, tipo_asunto="amparo_directo"):
    return fpartes.Partes(quejoso="Ana López Ruiz", tercero_interesado="Banco del Centro, S.A.",
                          tipo_asunto=tipo_asunto)


async def _sintetizar(cliente, **kw):
    return {}


_desenlace_real = ra._leer_desenlace_del_a_quo


def _desenlace(f, texto_acto):
    LL["orden"].append("desenlace")
    return _desenlace_real(f, texto_acto)


dg.redactar_estructura = _redactar_estructura
dg.componer = _dg_componer
f123.correr = _correr
fpartes.fichar = _fichar
import fase_sintesis
fase_sintesis.sintetizar = _sintetizar
ra._leer_desenlace_del_a_quo = _desenlace

ACTO_AD = ("PRIMERA SALA CIVIL DEL TRIBUNAL SUPERIOR DE JUSTICIA DEL ESTADO DE QUERÉTARO. "
           "Toca civil 374/2025. Querétaro, Querétaro, a quince de enero de dos mil veintiséis. "
           "VISTOS para resolver el toca 374/2025, relativo al recurso de apelación interpuesto "
           "por Ana López Ruiz contra la sentencia dictada en el expediente 905/2024. "
           "RESUELVE: PRIMERO. Se confirma la sentencia apelada. ") * 3
CONCEPTOS = ("CONCEPTOS DE VIOLACIÓN. Primero. La sentencia carece de fundamentación y motivación, "
             "en contravención de los artículos 14 y 16 constitucionales. ") * 5
ACTO_AR = ("JUZGADO PRIMERO DE DISTRITO EN EL ESTADO DE QUERÉTARO. Juicio de amparo indirecto "
           "1234/2025. Querétaro, Querétaro, a diez de diciembre de dos mil veinticinco. VISTOS "
           "para resolver el juicio de amparo 1234/2025 promovido por Ana López Ruiz contra actos "
           "del Director de Ingresos del Municipio de Querétaro. RESUELVE: PRIMERO. La Justicia "
           "de la Unión ampara y protege a Ana López Ruiz. ") * 3


def encargo(tipo="amparo_directo", **kw):
    base = dict(numero="512/2026", encabezado="AMPARO DIRECTO CIVIL: 512/2026",
                quejoso="Ana López Ruiz", magistrado="Ana Pérez", secretario="Luis Gómez",
                notificacion=datetime.date(2026, 1, 20), presentacion=datetime.date(2026, 2, 3),
                responsable="Primera Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro",
                tribunal="Tercer Tribunal Colegiado en Materias Administrativa y Civil del "
                         "Vigésimo Segundo Circuito",
                ciudad="Querétaro, Querétaro", modo="generado", tipo_asunto=tipo, materia="civil")
    base.update(kw)
    return ra.Encargo(**base)


def generar(e, acto=ACTO_AD):
    d = tempfile.mkdtemp(prefix="cableado_")
    ct.poner(False)
    return asyncio.run(ra.generar(object(), e, acto, CONCEPTOS, os.path.join(d, "adelanto.docx")))


def reiniciar():
    for k in LL:
        LL[k] = 0 if isinstance(LL[k], int) else ([] if isinstance(LL[k], list) else None)
    rpt.FALLAR = rpt.VACIO = False
    ft.ROMPER_ARMAR = False
    ft.ESCRITO = None
    ft.leer_escrito = _leer_escrito


# ═══════════════════════════════════════════════════════════════════════════
# SEXTA RONDA (3-oct-2026, C8): «TODOS» POR OMISIÓN. David, al dar el visto
# bueno: «empuja para todos los usuarios, no nada más para los de casa». La
# variable PROCEDENCIA_POR_TIPO sigue siendo el freno: «0» la apaga para todos
# y «casa» la limita a las cuentas de casa (de pruebas), sin desplegar nada.
print("\n1 · LA BANDERA: «procedencia_por_tipo», «todos» por omisión, y `rige`")
ok(ct.BANDERAS_REDISENO.get("procedencia_por_tipo") == "PROCEDENCIA_POR_TIPO",
   "la bandera está en BANDERAS_REDISENO con su variable de entorno")
ok(ct.OMISION_REDISENO.get("procedencia_por_tipo") == "todos",
   "su omisión es «todos» (C8: empuja para todos los usuarios)")
ok(callable(getattr(ct, "rige", None)), "contexto_taller.rige existe (la puerta del §6)")
ct.poner(False, pruebas=True)
ok(ct.rige("procedencia_por_tipo") and ra.rige_procedencia_por_tipo(),
   "sin entorno, rige para las cuentas de prueba")
ct.poner(False, pruebas=False)
ok(ra.rige_procedencia_por_tipo(), "sin entorno, rige TAMBIÉN para un secretario de fuera")
os.environ["PROCEDENCIA_POR_TIPO"] = "casa"
ct.poner(False, pruebas=False)
ok(not ra.rige_procedencia_por_tipo(), "PROCEDENCIA_POR_TIPO=casa la limita: un secretario de fuera no")
ct.poner(False, pruebas=True)
ok(ra.rige_procedencia_por_tipo(), "PROCEDENCIA_POR_TIPO=casa: la cuenta de casa (de pruebas) sí")
os.environ["PROCEDENCIA_POR_TIPO"] = "0"
ct.poner(False, pruebas=False)
ok(not ra.rige_procedencia_por_tipo(), "PROCEDENCIA_POR_TIPO=0 la apaga para un secretario de fuera")
ct.poner(False, pruebas=False)
os.environ["PROCEDENCIA_POR_TIPO"] = "todos"
ok(ra.rige_procedencia_por_tipo(), "PROCEDENCIA_POR_TIPO=todos la enciende para todos")
ct.poner(True, {"banderas": {"procedencia_por_tipo": False}})
ok(not ra.rige_procedencia_por_tipo(), "una evaluación de casa la apaga aunque el entorno diga «todos»")
os.environ["PROCEDENCIA_POR_TIPO"] = "0"
ct.poner(False, pruebas=True)
ok(not ra.rige_procedencia_por_tipo(), "PROCEDENCIA_POR_TIPO=0 la apaga también en casa")
os.environ.pop("PROCEDENCIA_POR_TIPO")
ct._CTX.set(None)
ok(ct.rige_para("procedencia_por_tipo", False, True) and ct.rige_para("procedencia_por_tipo", False, False),
   "rige_para: la misma regla para una cuenta dada (sin entorno, de pruebas y de fuera: las dos)")
os.environ["PROCEDENCIA_POR_TIPO"] = "casa"
ok(ct.rige_para("procedencia_por_tipo", False, True) and not ct.rige_para("procedencia_por_tipo", False, False),
   "rige_para con PROCEDENCIA_POR_TIPO=casa: de pruebas sí, de fuera no")
os.environ.pop("PROCEDENCIA_POR_TIPO")
ok(not ct.rige_para("procedencia_por_tipo", True, True, {"banderas": {"procedencia_por_tipo": "no"}}),
   "rige_para respeta la evaluación de casa")
ok(ct._CTX.get() is None, "rige_para NO deja puesto el contexto de la petición (lo pone el adelanto)")
ok(ra.Encargo(numero="1/2026", encabezado="", quejoso="", magistrado="", secretario="",
              notificacion=datetime.date(2026, 1, 1),
              presentacion=datetime.date(2026, 1, 2)).tramite is None,
   "Encargo.tramite nace en None (camino de siempre)")

# ═══════════════════════════════════════════════════════════════════════════
print("\n2 · LAS ENTRADAS: el formulario, el auto y los autos de SISE")
ok(ra.tramite_de_formulario("") == ({}, ""), "sin tramite_json, nada y sin error")
ok(ra.tramite_de_formulario("{no es json")[1] != "", "un JSON roto es un error (422 antes del OCR)")
ok(ra.tramite_de_formulario("[1, 2]")[1] != "", "una lista no es el objeto del formulario")
_decl, _mal = ra.tramite_de_formulario(json.dumps({"fecha_admision": "2026-02-09",
                                                  "ministerio_publico": "sin_pedimento"}))
ok(_mal == "" and _decl.get("admision") == {"fecha": "2026-02-09"}
   and _decl["fuentes"]["admision.fecha"] == "secretario",
   "el formulario pasa por ficha_tramite.de_formulario (fuente «secretario»)")
_vacio = ra.leer_auto_tramite("", "amparo_directo")
ok(set(_vacio) == {"avisos"} and "NO SALIÓ TEXTO" in _vacio["avisos"][0],
   "un auto sin texto no se lee: sólo su aviso")
reiniciar()
_auto = ra.leer_auto_tramite("AUTO DE PRESIDENCIA " * 20, "amparo_directo")
ok(LL["leer_auto"] == 1 and _auto["admision"]["fecha"] == "2026-02-10",
   "un auto con texto pasa por ficha_tramite.leer_auto")
_fun = ra.fundir_autos([_leer_auto("AUTO DE ADMISIÓN", "")], [_leer_auto("AUTO DE TURNO", "")])
ok(_fun["admision"]["fecha"] == "2026-02-10", "la admisión, del auto de admisión")
ok(_fun["turno"] == {"fecha": "2026-02-12", "ponente": "Magistrada Ana Pérez"},
   "el turno, del auto de turno (no el que menciona el de admisión)")
ok(_fun["fuentes"] == {"admision.fecha": "auto", "turno.fecha": "auto"}
   and _fun["avisos"] == ["AVISO DEL AUTO", "AVISO DEL TURNO"],
   "las fuentes van con su campo; los avisos de los dos, sin repetir")
ok("ministerio_publico" not in _fun, "un campo vacío en los dos autos no se inventa")
ok(ra.fundir_autos([], []) == {}, "sin autos, nada")

# ═══════════════════════════════════════════════════════════════════════════
print("\n3 · CON LA BANDERA: EL ADELANTO NO LLAMA A redactar_estructura")
os.environ["PROCEDENCIA_POR_TIPO"] = "todos"
reiniciar()
e = encargo()
e.tramite_auto = _leer_auto("AUTO DE ADMISIÓN", "amparo_directo")
e.tramite_declarado = _decl
r = generar(e)
ok(LL["redactar_estructura"] == 0, "redactar_estructura NO se llamó")
ok(LL["leer_acto"] == 1, "el acto se leyó UNA vez para la ficha (ficha_tramite.leer_acto)")
ok(LL["demanda"] == CONCEPTOS, "en amparo directo, el escrito va como demanda (de ahí, los derechos)")
ok(len(LL["armar"]) == 1, "la ficha se armó una vez")
_a = LL["armar"][0] if LL["armar"] else {}
ok(_a.get("auto") is e.tramite_auto and _a.get("declarado") is e.tramite_declarado,
   "armar recibió lo leído del auto y lo confirmado por el secretario")
ok((_a.get("acto") or {}).get("acto", {}).get("toca") == "374/2025",
   "armar recibió lo leído del acto")
ok(isinstance(_a.get("ficha_procesal"), dict) and _a.get("partes") is not None,
   "armar recibió la ficha procesal y las partes")
ok(isinstance(r.encargo.tramite, dict) and r.encargo.tramite.get("numero") == "512/2026",
   "la ficha quedó en el encargo")
ok(r.encargo.tramite.get("admision") == {"fecha": "2026-02-09"},
   "lo del secretario mandó sobre lo leído del auto (lo decide armar; aquí, el doble)")
ok(r.encargo.tramite.get("registrada") == "2026-02-10",
   "la ficha que viaja es JSON: las fechas, en ISO")
ok(all(a in r.encargo.tramite.get("avisos", []) for a in
       ("AVISO DE LA FICHA", "AVISO DEL AUTO", "AVISO DEL ACTO")),
   "los avisos del auto y del acto viajan en la ficha aunque armar no los copie")
_c = LL["dg_componer"][-1] if LL["dg_componer"] else {}
_est = _c.get("estructura")
ok(_est is not None and _est.visto == VISTO_COMPUESTO, "el V I S T O salió del compositor")
ok(_est is not None and [x["titulo"] for x in _est.resultandos] == TITULOS_COMPUESTOS,
   "los resultandos salieron del compositor, en su orden")
ok(_est is not None and _est.apertura == "" and _est.competencia == ""
   and _est.existencia == "" and _est.procedencia == "",
   "apertura, competencia, existencia y procedencia vacías: las pone el código")
ok(_c.get("datos", {}).get("tramite") is r.encargo.tramite,
   "documento_generado.componer recibe datos['tramite'] (la ficha)")
ok(_c.get("datos", {}).get("procesal") == {"expediente": "905/2024", "toca": "374/2025",
                                           "fecha_acto_iso": "2026-01-15"},
   "y datos['procesal'] = datos_extra del compositor")
_vistos = [x["datos"] for x in LL["componer_tipo"]]
ok(_vistos and all("procesal" not in d and "tramite" not in d for d in _vistos),
   "el compositor nunca recibe de vuelta lo que él produjo")
ok(_vistos and _vistos[-1].get("numero") == "512/2026" and _vistos[-1].get("tribunal"),
   "el compositor recibe los datos de siempre (número, tribunal…)")
ok(r.estructura is _est, "la estructura del resultado es la compuesta")
ok("AVISO DEL COMPOSITOR" in r.avisos and "AVISO DE LA FICHA" in r.avisos,
   "los avisos del compositor y de la ficha llegan al secretario")
ok("AVISO DEL MODELO" not in r.avisos, "nada del camino viejo")
ok(json.loads(json.dumps(ra.tramite_serializable(r.encargo.tramite))) == r.encargo.tramite,
   "la ficha sobrevive a la ida y vuelta de la sesión (jsonb)")

# ═══════════════════════════════════════════════════════════════════════════
print("\n4 · EL PROYECTO, EN EL OTRO WORKER: REUTILIZA LA FICHA, NO EL MODELO")
_n_leer, _n_armar = LL["leer_acto"], len(LL["armar"])
# El encargo como lo rehidrata `_taller_recuperar_sesion_crudo`: sólo con lo
# guardado (sin las entradas del auto ni del formulario) y sin estructura.
e2 = encargo(tramite=json.loads(json.dumps(ra.tramite_serializable(r.encargo.tramite))))
relleno = ens.Relleno(encabezado=e2.encabezado, numero_asunto=e2.numero, quejoso=e2.quejoso,
                      magistrado=e2.magistrado, secretario=e2.secretario,
                      antecedentes=["Antecedente."], estudio=["Estudio de fondo."])
_ruta2 = os.path.join(tempfile.mkdtemp(prefix="cableado_"), "proyecto.docx")
_, _av2, _est2 = asyncio.run(ra._componer_generado(
    object(), e2, relleno, r.computo, _ruta2, estructura_previa=None, acto=ACTO_AD,
    partes=r.partes, fases=r.fases))
ok(LL["redactar_estructura"] == 0, "redactar_estructura sigue sin llamarse")
ok(LL["leer_acto"] == _n_leer and len(LL["armar"]) == _n_armar,
   "ni se vuelve a leer el acto ni a armar la ficha: se reutiliza la de la sesión")
ok(_est2.visto == VISTO_COMPUESTO and LL["dg_componer"][-1]["datos"].get("procesal"),
   "se recompone con el compositor y los considerandos reciben datos['procesal']")
ok("AVISO DE LA FICHA" in _av2, "los avisos de la ficha también salen en el proyecto")

print("\n5 · SESIÓN SIN FICHA O COMPOSITOR ROTO: EL CAMINO DE SIEMPRE, CON SU AVISO")
_n = LL["redactar_estructura"]
e3 = encargo()          # una sesión de antes de la bandera: sin ficha
_, _av3, _est3 = asyncio.run(ra._componer_generado(
    object(), e3, relleno, r.computo, _ruta2, estructura_previa=None, acto=ACTO_AD,
    partes=r.partes, fases=r.fases))
ok(LL["redactar_estructura"] == _n + 1 and _est3.visto == "VISTO DEL MODELO",
   "sin ficha, aunque rija la bandera: la llamada de siempre")
ok("tramite" not in LL["dg_componer"][-1]["datos"] and "procesal" not in LL["dg_componer"][-1]["datos"],
   "y documento_generado no recibe ni tramite ni procesal")
rpt.FALLAR = True
e4 = encargo(tramite=dict(r.encargo.tramite))
_, _av4, _est4 = asyncio.run(ra._componer_generado(
    object(), e4, relleno, r.computo, _ruta2, estructura_previa=None, acto=ACTO_AD,
    partes=r.partes, fases=r.fases))
rpt.FALLAR = False
ok(LL["redactar_estructura"] == _n + 2 and _est4.visto == "VISTO DEL MODELO",
   "si el compositor falla, los resultandos los escribe el modelo (el documento no se cae)")
ok(_av4 and _av4[0].startswith("LOS RESULTANDOS NO SE PUDIERON COMPONER CON LA FICHA DE TRÁMITE"),
   "y el primer aviso lo dice")
ok("procesal" not in LL["dg_componer"][-1]["datos"] and LL["dg_componer"][-1]["datos"].get("tramite"),
   "sin procesal (no hubo compositor), con la ficha para la verja")
rpt.VACIO = True
_, _av5, _est5 = asyncio.run(ra._componer_generado(
    object(), encargo(tramite=dict(r.encargo.tramite)), relleno, r.computo, _ruta2,
    estructura_previa=None, acto=ACTO_AD, partes=r.partes, fases=r.fases))
rpt.VACIO = False
ok(LL["redactar_estructura"] == _n + 3 and _est5.visto == "VISTO DEL MODELO",
   "si el compositor devuelve todo vacío (su manera de fallar sin lanzar), el camino viejo")
ok(_av5 and _av5[0].startswith("LOS RESULTANDOS NO SE PUDIERON COMPONER") and "KeyError" in _av5[0],
   "y el aviso lleva lo que dijo el compositor")

print("\n6 · ARMAR ROTO: LA FICHA MÍNIMA, TODO EN HUECO, Y EL AVISO")
reiniciar()
ft.ROMPER_ARMAR = True
e5 = encargo()
r5 = generar(e5)
ft.ROMPER_ARMAR = False
ok(LL["redactar_estructura"] == 0, "no se vuelve a la llamada a ciegas")
ok(isinstance(r5.encargo.tramite, dict) and r5.encargo.tramite.get("tipo") == "amparo_directo"
   and r5.encargo.tramite.get("numero") == "512/2026" and "acto" not in r5.encargo.tramite,
   "la ficha mínima: tipo, número y sede, nada inventado")
ok(any(a.startswith("LA FICHA DE TRÁMITE NO SE PUDO ARMAR") for a in r5.avisos),
   "el aviso de la ficha que no se armó llega al secretario")
ok(LL["dg_componer"] and LL["dg_componer"][-1]["estructura"].visto == VISTO_COMPUESTO,
   "y el compositor compone igual (con sus huecos)")

print("\n7 · RECURSO: EL DESENLACE DEL A QUO SE LEE ANTES DE ARMAR LA FICHA")
reiniciar()
r6 = generar(encargo("amparo_revision", encabezado="AMPARO EN REVISIÓN: 512/2026",
                     responsable="Director de Ingresos del Municipio de Querétaro"), acto=ACTO_AR)
ok(LL["orden"] == ["desenlace", "armar"], f"con la bandera: desenlace y luego ficha ({LL['orden']})")
ok(LL["demanda"] == "", "en un recurso, los agravios NO se pasan como demanda")
ok(LL["redactar_estructura"] == 0 and LL["componer_tipo"]
   and LL["componer_tipo"][-1]["tipo"] == "amparo_revision",
   "el compositor recibe el tipo normalizado del recurso")

# ═══════════════════════════════════════════════════════════════════════════
print("\n8 · SIN LA BANDERA: EL CAMINO DE HOY")
os.environ["PROCEDENCIA_POR_TIPO"] = "0"
reiniciar()
e7 = encargo()
e7.tramite_auto = _leer_auto("AUTO DE ADMISIÓN", "amparo_directo")
e7.tramite_declarado = _decl
r7 = generar(e7)
ok(LL["redactar_estructura"] == 1, "redactar_estructura se llama una vez, como hoy")
ok(LL["leer_acto"] == 0 and LL["armar"] == [] and LL["componer_tipo"] == [],
   "ni se lee el acto para la ficha, ni se arma, ni se compone por tipo")
ok(r7.encargo.tramite is None, "el encargo no lleva ficha")
_c7 = LL["dg_componer"][-1]
ok(_c7["estructura"].visto == "VISTO DEL MODELO" and r7.estructura is _c7["estructura"],
   "la estructura es la del modelo")
ok("tramite" not in _c7["datos"] and "procesal" not in _c7["datos"],
   "documento_generado no recibe ni tramite ni procesal")
ok("AVISO DEL MODELO" in r7.avisos and "AVISO DEL COMPOSITOR" not in r7.avisos,
   "los avisos son los de siempre")
reiniciar()
r8 = generar(encargo("amparo_revision", encabezado="AMPARO EN REVISIÓN: 512/2026",
                     responsable="Director de Ingresos del Municipio de Querétaro"), acto=ACTO_AR)
ok(LL["orden"] == ["desenlace"], "en el recurso, el desenlace se lee una vez y donde siempre")
ok(isinstance(ra._datos_estructura(e2, "", acto=ACTO_AD).get("ficha_procesal"), dict)
   and "procesal" not in ra._datos_estructura(e2, "", acto=ACTO_AD),
   "_datos_estructura sin bandera no añade procesal aunque el encargo traiga ficha")
os.environ["PROCEDENCIA_POR_TIPO"] = "todos"
_d9 = ra._datos_estructura(e2, "", acto=ACTO_AD)
ok(_d9.get("tramite") is e2.tramite and _d9.get("procesal", {}).get("toca") == "374/2025",
   "_datos_estructura con bandera y ficha añade tramite y procesal")

# ═══════════════════════════════════════════════════════════════════════════
print("\n9 · main.py: LOS CAMPOS NUEVOS Y LA SESIÓN (lectura estática; no se arranca el API)")
with open(os.path.join(AQUI, "main.py"), encoding="utf-8") as _fh:
    _src = _fh.read()
_arbol = ast.parse(_src)
_fun = {n.name: n for n in ast.walk(_arbol) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _args(nombre):
    n = _fun.get(nombre)
    return [a.arg for a in n.args.args] if n else []


def _fuente(nombre):
    n = _fun.get(nombre)
    return ast.get_source_segment(_src, n) if n else ""


ok({"tramite_json", "admision"} <= set(_args("taller_adelanto")),
   "/taller/adelanto acepta tramite_json y admision")
_ad = _fun.get("taller_adelanto")
_def = {a.arg: d for a, d in zip(_ad.args.args[-len(_ad.args.defaults):], _ad.args.defaults)} if _ad else {}
ok(ast.unparse(_def.get("tramite_json")) == "Form('')" and ast.unparse(_def.get("admision")) == "File(None)",
   "los dos son opcionales: Form('') y File(None)")
ok("tramite_de_formulario" in _fuente("taller_adelanto") and "HTTPException(422" in _fuente("taller_adelanto"),
   "el trámite mal escrito es un 422")
ok('"tramite"' in _fuente("_taller_guardar_sesion") and "tramite_serializable" in _fuente("_taller_guardar_sesion"),
   "la sesión guarda la ficha serializada")
ok("tramite=(e.get(\"tramite\")" in _fuente("_taller_recuperar_sesion_crudo"),
   "y la sesión la devuelve al encargo")
ok('ficha["tramite"]' in _fuente("taller_desde_admision") and 'ficha.pop("tramite"' in _fuente("taller_desde_admision"),
   "/taller/desde-admision devuelve ficha.tramite con la bandera y la quita sin ella")
ok("auto_admision" in _fuente("taller_desde_expediente") and "auto_turno" in _fuente("taller_desde_expediente")
   and "admision=_autos_tr" in _fuente("taller_desde_expediente"),
   "/taller/desde-expediente lee los autos de admisión y de turno y los pasa al adelanto")


# ═══ LAS FUNCIONES DE main.py, EJECUTADAS SIN IMPORTAR main ═════════════════
# Importar main levanta clientes y lee configuración: aquí se compilan sólo los
# nodos que se prueban, en un espacio de nombres con dobles de lo que llaman.
def _de_main(*nombres, **dobles):
    nodos = []
    for nombre in nombres:
        n = copy.deepcopy(_fun[nombre])
        n.decorator_list = []          # @app.get(...) no existe aquí
        nodos.append(n)
    ns = dict(dobles)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodos, type_ignores=[])),
                 "main.py", "exec"), ns)
    return ns


def _casa(u):
    if u == "rompe@x":
        raise RuntimeError("la lista de casa no contesta")
    return u.endswith("@casa")


M = _de_main("_taller_rige_procedencia", "_taller_estado_procedencia",
             "_taller_nombre_procedencia", "taller_reglas_surtimiento",
             _taller_es_casa=_casa,
             _taller_cuenta_de_pruebas=lambda u: u == "pruebas@x")

# ═══════════════════════════════════════════════════════════════════════════
print("\n10 · GET /taller/estado DICE SI RIGE LA BANDERA PARA ESA CUENTA")
_estado = _fuente("taller_estado")
ok('"procedencia_por_tipo": _taller_estado_procedencia(user_email)' in _estado,
   "/taller/estado devuelve «procedencia_por_tipo» con la puerta de la cuenta")
os.environ.pop("PROCEDENCIA_POR_TIPO", None)
ct._CTX.set(None)
ok(M["_taller_estado_procedencia"]("pruebas@x") is True,
   "sin entorno: True para una cuenta de pruebas")
ok(M["_taller_estado_procedencia"]("fuera@x") is True,
   "sin entorno: True también para un secretario de fuera (omisión «todos», C8)")
os.environ["PROCEDENCIA_POR_TIPO"] = "casa"
ok(M["_taller_estado_procedencia"]("fuera@x") is False and M["_taller_estado_procedencia"]("pruebas@x") is True,
   "PROCEDENCIA_POR_TIPO=casa: False para uno de fuera, True para la cuenta de pruebas")
os.environ["PROCEDENCIA_POR_TIPO"] = "todos"
ok(M["_taller_estado_procedencia"]("fuera@x") is True, "PROCEDENCIA_POR_TIPO=todos: True para todos")
os.environ["PROCEDENCIA_POR_TIPO"] = "0"
ok(M["_taller_estado_procedencia"]("pruebas@x") is False, "PROCEDENCIA_POR_TIPO=0: False también en pruebas")
os.environ["PROCEDENCIA_POR_TIPO"] = "todos"
ok(M["_taller_estado_procedencia"]("rompe@x") is False,
   "si decidir la cuenta lanza, False y el estado no se cae")
ok(ct._CTX.get() is None, "preguntar el estado no deja puesto el contexto de la petición")
_real = M["_taller_rige_procedencia"]


def _lanza(*a, **k):
    raise RuntimeError("roto a propósito")


M["_taller_rige_procedencia"] = _lanza
ok(M["_taller_estado_procedencia"]("pruebas@x") is False,
   "si la puerta misma lanza, False (nunca lanza)")
M["_taller_rige_procedencia"] = lambda *a, **k: "sí"
ok(M["_taller_estado_procedencia"]("pruebas@x") is True,
   "lo que devuelva la puerta sale como bool (JSON true/false)")
M["_taller_rige_procedencia"] = _real

# ═══════════════════════════════════════════════════════════════════════════
print("\n11 · /taller/reglas-surtimiento ACEPTA «papel» Y EL .docx SE LLAMA «PROCEDENCIA»")
ok("papel" in _args("taller_reglas_surtimiento")
   and ast.unparse(_fun["taller_reglas_surtimiento"].args.defaults[-1]) == "''",
   "el parámetro «papel» es opcional (vacío por omisión)")
_rs = M["taller_reglas_surtimiento"]
_ar_aut = asyncio.run(_rs(tipo_asunto="amparo_revision", responsable="", papel="autoridad"))
_ar_sin = asyncio.run(_rs(tipo_asunto="amparo_revision", responsable=""))
ok(_ar_aut["por_omision"] == "oficio" and _ar_aut["reglas"][0]["clave"] == "oficio",
   "AR con papel «autoridad»: la regla de omisión es «oficio» (art. 31, fr. I) y va primero")
ok(_ar_sin["por_omision"] == "personal", "AR sin papel: la de siempre («personal»)")
ok(asyncio.run(_rs(tipo_asunto="amparo_revision", responsable="", papel=" Autoridad "))["por_omision"]
   == "oficio", "el papel se normaliza (espacios, mayúsculas)")
ok(asyncio.run(_rs(tipo_asunto="queja", responsable="", papel="autoridad"))["por_omision"] == "oficio",
   "en la queja, lo mismo")
ok(asyncio.run(_rs(tipo_asunto="amparo_revision", responsable="", papel="quejoso")) == _ar_sin,
   "si recurre la quejosa, la lista de siempre")
_tfja = "Sala Regional del Centro II del Tribunal Federal de Justicia Administrativa"
ok(asyncio.run(_rs(tipo_asunto="amparo_directo", responsable=_tfja, papel="autoridad"))
   == asyncio.run(_rs(tipo_asunto="amparo_directo", responsable=_tfja)),
   "fuera de la revisión y la queja, el papel no cambia nada")
ok(M["_taller_nombre_procedencia"]("512/2026") == "512-2026 PROCEDENCIA.docx",
   "el nombre del archivo: «512-2026 PROCEDENCIA.docx»")
_fr = [n for n in ast.walk(_fun["taller_adelanto"]) if isinstance(n, ast.Call)
       and ast.unparse(n.func) == "FileResponse"]
_kw_fr = {k.arg: ast.unparse(k.value) for k in (_fr[0].keywords if _fr else [])}
ok(_kw_fr.get("filename") == "_taller_nombre_procedencia(numero)",
   "/taller/adelanto descarga con ese nombre (Content-Disposition)")
ok("ADELANTO" not in _kw_fr.get("filename", "ADELANTO"),
   "y no con «ADELANTO.docx»")
ok("@app.post(\"/taller/adelanto\")" in _src, "la ruta no cambia: sigue siendo /taller/adelanto")

# ═══════════════════════════════════════════════════════════════════════════
print("\n12 · RECURRE UNA AUTORIDAD: EL CÓMPUTO CON SU REGLA (oficio, art. 31, fr. I)")
import fase0_oportunidad as f0
import tipos_asunto as ta
os.environ["PROCEDENCIA_POR_TIPO"] = "todos"
DIRECTOR = "Director de Ingresos del Municipio de Querétaro"
_AR = dict(encabezado="AMPARO EN REVISIÓN: 512/2026", responsable=DIRECTOR)


def _ar(tipo="amparo_revision", **kw):
    base = dict(_AR)
    base.update(kw)
    return encargo(tipo, **base)


def _como_la_sesion(enc):
    """El cómputo como lo rehace el worker que resuelve
    (`main._taller_recuperar_sesion_crudo`): con lo que guarda el encargo."""
    return f0.computar(enc.notificacion, enc.presentacion, enc.regla_surtimiento, enc.plazo,
                       enc.responsable, getattr(enc, "dias_inhabiles_extra", None),
                       getattr(enc, "tipo_asunto", "") or "amparo_directo",
                       getattr(enc, "inhabiles_responsable", "") or None,
                       surtio_manual=f0.surtio_manual_de(enc)[0],
                       plazo_anios=ta.anios_de(enc.tipo_asunto, enc.excepcion_plazo or ""))


def _hay(avisos, inicio):
    return [a for a in avisos if str(a).startswith(inicio)]


# El caso de siempre: el secretario tecleó a la AUTORIDAD como «quien promueve»
# y la regla de omisión. Los papeles se separan con el resolutivo del juzgado
# DESPUÉS del primer cómputo: ahí se sabe quién recurre, y ahí se rehace.
reiniciar()
r12 = generar(_ar(quejoso=DIRECTOR), acto=ACTO_AR)
ok(r12.encargo.recurrente == DIRECTOR and r12.encargo.quejoso == "Ana López Ruiz",
   "los papeles se separaron (recurre la autoridad, la quejosa es otra)")
ok(r12.encargo.regla_surtimiento == "oficio", "la regla de omisión («personal») pasó a «oficio»")
ok(r12.computo.regla.clave == "oficio" and r12.computo.surtio == datetime.date(2026, 1, 20),
   "el cómputo surte el mismo día de la notificación (fr. I), no al día hábil siguiente")
_pers = f0.computar(datetime.date(2026, 1, 20), datetime.date(2026, 2, 3), "personal", 10, DIRECTOR,
                    None, "amparo_revision", None)
ok(r12.computo.inicio < _pers.inicio, f"el plazo arranca antes ({r12.computo.inicio} y no {_pers.inicio})")
ok(len(_hay(r12.avisos, "RECURRE UNA AUTORIDAD: EL CÓMPUTO SE HIZO CON LA NOTIFICACIÓN POR OFICIO")) == 1,
   "un aviso lo dice, una vez")
ok("de manera personal" in (_hay(r12.avisos, "RECURRE UNA AUTORIDAD")[:1] or [""])[0],
   "y nombra la regla que venía declarada")
_c12 = LL["dg_componer"][-1] if LL["dg_componer"] else {}
ok(_c12.get("computo") is r12.computo and _c12.get("computo").regla.clave == "oficio",
   "el documento se compone con el cómputo rehecho")
ok(_c12.get("datos", {}).get("papel_recurrente") == "autoridad",
   "y los considerandos ven el mismo papel («autoridad»): cómputo y fundamento no discrepan")
ok(LL["armar"] and LL["armar"][-1]["regla"] == "oficio",
   "la ficha de trámite se arma DESPUÉS del cambio: su forma de notificación dice lo mismo")
_s12 = _como_la_sesion(r12.encargo)
ok((_s12.surtio, _s12.inicio, _s12.vencimiento) == (r12.computo.surtio, r12.computo.inicio,
                                                    r12.computo.vencimiento),
   "el otro worker, con la regla guardada en el encargo, computa lo mismo")

reiniciar()
r12b = generar(_ar(regla_surtimiento="lista", recurrente=DIRECTOR), acto=ACTO_AR)
ok(r12b.encargo.regla_surtimiento == "oficio" and "por lista" in
   (_hay(r12b.avisos, "RECURRE UNA AUTORIDAD")[:1] or [""])[0],
   "«lista» también es de los particulares: pasa a «oficio» y el aviso la nombra")

reiniciar()
r12c = generar(_ar("queja", encabezado="QUEJA: 512/2026", recurrente=DIRECTOR), acto=ACTO_AR)
ok(r12c.encargo.regla_surtimiento == "oficio" and r12c.computo.regla.clave == "oficio",
   "en la queja de la autoridad, lo mismo")

reiniciar()
_e_el = _ar(recurrente=DIRECTOR)
_e_el.tramite_declarado = {"forma_notificacion": "electronica",
                           "fuentes": {"forma_notificacion": "secretario"}}
r12d = generar(_e_el, acto=ACTO_AR)
ok(r12d.encargo.regla_surtimiento == "electronica"
   and _hay(r12d.avisos, "RECURRE UNA AUTORIDAD Y LA NOTIFICACIÓN FUE ELECTRÓNICA"),
   "si declaró en «Trámite» que fue electrónica: la regla de la fr. III, con su aviso")

for _regla, _extra in (("electronica", {}), ("oficio", {}),
                       ("otra", {"surte_efectos": "2026-01-22"})):
    reiniciar()
    _rr = generar(_ar(regla_surtimiento=_regla, recurrente=DIRECTOR, **_extra), acto=ACTO_AR)
    ok(_rr.encargo.regla_surtimiento == _regla and not _hay(_rr.avisos, "RECURRE UNA AUTORIDAD"),
       f"lo que eligió el secretario («{_regla}») se respeta, sin aviso")

reiniciar()
r12e = generar(_ar(quejoso="Ana López Ruiz"), acto=ACTO_AR)
ok(r12e.encargo.regla_surtimiento == "personal" and not _hay(r12e.avisos, "RECURRE UNA AUTORIDAD"),
   "si recurre la quejosa, la regla de los particulares se queda")

reiniciar()
r12f = generar(_ar("revision_fiscal", encabezado="REVISIÓN FISCAL: 512/2026", recurrente=DIRECTOR),
               acto=ACTO_AR)
ok(r12f.encargo.regla_surtimiento == "personal" and not _hay(r12f.avisos, "RECURRE UNA AUTORIDAD"),
   "en la revisión fiscal no se toca (la notificación la rige la LFPCA)")

reiniciar()
r12g = generar(encargo(recurrente=DIRECTOR))
ok(r12g.encargo.regla_surtimiento == "personal" and not _hay(r12g.avisos, "RECURRE UNA AUTORIDAD"),
   "en el amparo directo no se toca")

# EL DÍA QUE DECIDE: presentado el 5 de febrero, con la personal vence el 6
# (en tiempo) y con el oficio el 4 (fuera).
reiniciar()
r12h = generar(_ar(recurrente=DIRECTOR, presentacion=datetime.date(2026, 2, 5)), acto=ACTO_AR)
ok(r12h.computo.oportuna is False, "con la regla de la autoridad, el recurso del 5 de febrero sale fuera")
ok(r12h.avisos.count(ra.AVISO_EXTEMPORANEA) == 1, "el aviso de extemporánea sale una vez")
ok(len(_hay(r12h.avisos, "CON LA REGLA DE LA AUTORIDAD EL RECURSO SALE FUERA DE PLAZO")) == 1,
   "y otro dice que con la de los particulares salía en tiempo")

# LO QUE DECÍA EL GUARDIÁN LABORAL DEJA DE SER VERDAD Y SE QUITA.
reiniciar()
r12i = generar(_ar(recurrente=DIRECTOR, materia="laboral", regla_surtimiento="lista"), acto=ACTO_AR)
ok(r12i.encargo.regla_surtimiento == "oficio"
   and not _hay(r12i.avisos, "El cómputo se hizo con notificación PERSONAL"),
   "el aviso «se hizo con notificación PERSONAL» del guardián laboral ya no sale")

# SIN LA BANDERA, TAMBIÉN (D2 del integrador, tercera ronda, 3-oct-2026). En
# la segunda ronda esto iba detrás de la bandera y, sin ella, el párrafo salía
# «conforme al *********» con la orden de regenerar (rev_4).
os.environ["PROCEDENCIA_POR_TIPO"] = "0"
reiniciar()
r12j = generar(_ar(quejoso=DIRECTOR), acto=ACTO_AR)
ok(r12j.encargo.regla_surtimiento == "oficio" and r12j.computo.regla.clave == "oficio"
   and r12j.computo.surtio == datetime.date(2026, 1, 20)
   and len(_hay(r12j.avisos, "RECURRE UNA AUTORIDAD: EL CÓMPUTO SE HIZO CON LA NOTIFICACIÓN POR OFICIO")) == 1,
   "sin la bandera también: la autoridad que recurre se computa con «oficio» (fr. I), con su aviso")
ok(LL["armar"] == [] and r12j.encargo.tramite is None,
   "sin la bandera, nada más del camino nuevo (ni ficha ni armar)")
_c12j = LL["dg_componer"][-1] if LL["dg_componer"] else {}
ok(_c12j.get("computo") is r12j.computo and _c12j.get("datos", {}).get("papel_recurrente") == "autoridad",
   "el documento del camino viejo se compone con el cómputo rehecho y el mismo papel")
reiniciar()
r12k = generar(_ar(recurrente=DIRECTOR, materia="laboral", regla_surtimiento="lista"), acto=ACTO_AR)
ok(r12k.encargo.regla_surtimiento == "oficio"
   and not _hay(r12k.avisos, "El cómputo se hizo con notificación PERSONAL"),
   "sin la bandera, el aviso del guardián laboral («se hizo con notificación PERSONAL») también se quita")
reiniciar()
r12l = generar(_ar(quejoso="Ana López Ruiz"), acto=ACTO_AR)
ok(r12l.encargo.regla_surtimiento == "personal" and not _hay(r12l.avisos, "RECURRE UNA AUTORIDAD"),
   "sin la bandera, si recurre la quejosa, la regla de los particulares se queda")
os.environ["PROCEDENCIA_POR_TIPO"] = "todos"

# LA DECISIÓN, SUELTA.
ok(ra.regla_de_la_autoridad(_ar(), "autoridad")[0] == "oficio", "regla_de_la_autoridad: AR + autoridad → oficio")
ok(ra.regla_de_la_autoridad(_ar(), "tercero") == ("", ""), "la tercera interesada es particular: nada")
ok(ra.regla_de_la_autoridad(_ar(), "") == ("", ""), "papel desconocido: nada (no se supone)")
ok(ra.regla_de_la_autoridad(_ar(), "autoridad", "electrónica")[0] == "electronica",
   "forma declarada «electrónica» (con tilde) → electronica")
ok(ra.regla_de_la_autoridad(_ar(regla_surtimiento="tja_qro_boletin"), "autoridad") == ("", ""),
   "una regla que no es de los particulares no se toca")


def _romper(*a, **k):
    raise RuntimeError("cómputo roto a propósito")


_av_x = ["previo"]
_c_x = r12j.computo
ok(ra._recomputar_para_la_autoridad(_ar(recurrente=DIRECTOR), _c_x, r12j.fases, r12j.partes,
                                    ACTO_AR, _av_x, _romper) is _c_x and _av_x == ["previo"],
   "si rehacer el cómputo falla, el de antes y los avisos intactos (nunca lanza)")

# ═══════════════════════════════════════════════════════════════════════════
print("\n13 · LOS TRES CAMPOS NUEVOS DE LA FICHA VIAJAN SIN FILTRARSE")
_form13 = {"fecha_admision": "2026-02-09", "fraccion_63": "III", "cuantia": "$1,234,567.89",
           "fundamento_surtimiento": "el artículo 126 del Código de Procedimientos Civiles "
                                     "del Estado de Querétaro"}
reiniciar()
_d13, _m13 = ra.tramite_de_formulario(json.dumps(_form13, ensure_ascii=False))
ok(_m13 == "" and LL["form"] == _form13,
   "tramite_de_formulario entrega a de_formulario el JSON entero, sin quitar claves")
ok(all(_d13.get(k) == _form13[k] for k in NUEVAS), "lo interpretado vuelve tal cual")
e13 = encargo("revision_fiscal", encabezado="REVISIÓN FISCAL: 512/2026")
e13.tramite_declarado = _d13
r13 = generar(e13)
ok(LL["armar"] and all(LL["armar"][-1]["declarado"].get(k) == _form13[k] for k in NUEVAS),
   "armar recibe los tres en lo declarado")
ok(all(r13.encargo.tramite.get(k) == _form13[k] for k in NUEVAS),
   "la ficha del encargo los lleva (serializada para la sesión)")
ok(LL["componer_tipo"] and all(LL["componer_tipo"][-1]["ficha"].get(k) == _form13[k] for k in NUEVAS),
   "el compositor los recibe en la ficha")
ok(LL["dg_componer"] and all((LL["dg_componer"][-1]["datos"].get("tramite") or {}).get(k) == _form13[k]
                             for k in NUEVAS),
   "y documento_generado, en datos['tramite']")
_ida = json.loads(json.dumps(ra.tramite_serializable(r13.encargo.tramite)))
ok(all(_ida.get(k) == _form13[k] for k in NUEVAS), "sobreviven a la ida y vuelta de la sesión (jsonb)")

# Y CON EL ficha_tramite DE VERDAD, si ya los trae (lo escribe otra pieza en
# esta misma ronda): formulario → declarado → armar → formulario, sin pérdida.
import importlib.util
_spec = importlib.util.spec_from_file_location("ficha_tramite_real", os.path.join(AQUI, "ficha_tramite.py"))
_ftr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ftr)
if all(k in getattr(_ftr, "CLAVES_FORMULARIO", ()) for k in NUEVAS):
    _doble = sys.modules["ficha_tramite"]
    sys.modules["ficha_tramite"] = _ftr
    try:
        _d13r, _m13r = ra.tramite_de_formulario(json.dumps(_form13, ensure_ascii=False))
        _ida13 = _ftr.a_formulario(_d13r)
        ok(_m13r == "" and all(_ida13.get(k) for k in NUEVAS),
           "ficha_tramite real: de_formulario los acepta y a_formulario los devuelve")
        _fr13 = ra.armar_tramite(encargo("revision_fiscal", encabezado="REVISIÓN FISCAL: 512/2026",
                                         tramite_declarado=_d13r))
        ok(all(_ftr.a_formulario(_fr13).get(k) == _ida13.get(k) for k in NUEVAS),
           "ficha_tramite real: lo declarado llega igual a la ficha armada y serializada")
    finally:
        sys.modules["ficha_tramite"] = _doble
else:
    print(f"   (aviso, no cuenta) ficha_tramite real todavía con {len(getattr(_ftr, 'CLAVES_FORMULARIO', ()))} "
          f"claves: el viaje de las tres nuevas sólo se comprobó con el doble")

# ═══════════════════════════════════════════════════════════════════════════
# TERCERA RONDA (3-oct-2026) — dueño F de FIXES_R3.md
# ═══════════════════════════════════════════════════════════════════════════
import time as _time_t

os.environ["PROCEDENCIA_POR_TIPO"] = "todos"


def _primero(avisos, inicio):
    return (_hay(avisos, inicio)[:1] or [""])[0]


def _datos_dg():
    return (LL["dg_componer"][-1] if LL["dg_componer"] else {}).get("datos", {})


# ═══════════════════════════════════════════════════════════════════════════
print("\n14 · QUIÉN RECURRE, LEÍDO DEL ESCRITO, SI EL FORMULARIO NO LO DICE (AR 631/2025)")
# Punta a punta con los papeles reales del 631: ni «quien promueve» ni
# «recurrente» en el formulario, y el proyecto salió con «QUEJOSA Y
# RECURRENTE: Unión de Trabajadores…» cuando recurría la adquirente del
# inmueble, «hoy TERCERO INTERESADO».
IMPULSORA = "Impulsora de Desarrollos Inmobiliarios VV, S.A. de C.V."
reiniciar()
ft.ESCRITO = {"promovente": IMPULSORA, "caracter": "TERCERO INTERESADO",
              "representante": "Juan Pérez López", "figura_representante": "apoderado legal",
              "fuentes": {"promovente": "escrito", "caracter": "escrito"},
              "avisos": ["AVISO DEL ESCRITO"]}
r14 = generar(_ar(quejoso="", recurrente=""), acto=ACTO_AR)
ok(LL["leer_escrito"] == 1 and LL["escrito_args"] == (CONCEPTOS, "amparo_revision"),
   "sin quién recurre en el formulario: se lee del escrito (los agravios), una vez, con el tipo")
ok(r14.encargo.recurrente == IMPULSORA and r14.encargo.papel_recurrente == "tercero",
   "el encargo fija a la recurrente y su carácter («tercero»)")
ok(r14.encargo.quejoso == "Ana López Ruiz",
   "la quejosa es la que nombra el punto resolutivo del juzgado, no la que recurre")
_a14 = _primero(r14.avisos, "SE LEYÓ DEL ESCRITO QUIÉN RECURRE")
ok(len(_hay(r14.avisos, "SE LEYÓ DEL ESCRITO QUIÉN RECURRE")) == 1
   and f"«{IMPULSORA}»" in _a14 and "en su carácter de parte tercera interesada" in _a14
   and "por conducto de su apoderado legal Juan Pérez López" in _a14,
   "un aviso lo dice, con el nombre, el carácter y el representante")
ok("punto resolutivo" in _a14, "y dice de dónde salió la quejosa")
ok(not _hay(r14.avisos, "SE SEPARARON LOS PAPELES"),
   "la separación por el resolutivo no se repite (los papeles los fijó el escrito)")
ok(_datos_dg().get("papel_recurrente") == "tercero" and _datos_dg().get("recurrente") == IMPULSORA,
   "los considerandos reciben a la recurrente y su papel")
ok(LL["armar"] and (LL["armar"][-1].get("escrito") or {}).get("promovente") == IMPULSORA,
   "armar recibe lo leído del escrito (fuente «escrito»)")
ok("AVISO DEL ESCRITO" in (r14.encargo.tramite or {}).get("avisos", []),
   "los avisos de la lectura del escrito viajan en la ficha")
ok(r14.encargo.regla_surtimiento == "personal" and not _hay(r14.avisos, "RECURRE UNA AUTORIDAD"),
   "la tercera interesada es particular: la regla de omisión se queda")

reiniciar()
SINDICO = "Síndico Municipal de Corregidora, Querétaro"
ft.ESCRITO = {"promovente": SINDICO, "caracter": "autoridad responsable"}
r14b = generar(_ar(quejoso="", recurrente=""), acto=ACTO_AR)
ok(not ra._parece_autoridad(SINDICO), "(el nombre no trae ninguna palabra de cargo del vocabulario)")
ok(r14b.encargo.recurrente == SINDICO and r14b.encargo.papel_recurrente == "autoridad"
   and _datos_dg().get("papel_recurrente") == "autoridad",
   "el carácter del escrito manda: «autoridad», aunque el nombre no lo delate")
ok(r14b.encargo.regla_surtimiento == "oficio" and r14b.computo.surtio == datetime.date(2026, 1, 20),
   "y la autoridad se computa con «oficio»: surte el día de la notificación")

reiniciar()
ft.ESCRITO = {"promovente": "Ana López Ruiz", "caracter": "parte quejosa"}
r14c = generar(_ar(quejoso="", recurrente=""), acto=ACTO_AR)
ok(r14c.encargo.quejoso == "Ana López Ruiz" and r14c.encargo.recurrente == ""
   and r14c.encargo.papel_recurrente == "quejoso" and _datos_dg().get("papel_recurrente") == "quejoso",
   "recurre la quejosa: quejoso = el del escrito, sin recurrente aparte, papel «quejoso»")
ok("en su carácter de parte quejosa" in _primero(r14c.avisos, "SE LEYÓ DEL ESCRITO QUIÉN RECURRE"),
   "con su aviso")

for _kw14, _que in ((dict(quejoso=DIRECTOR), "«quien promueve»"),
                    (dict(quejoso="", recurrente=DIRECTOR), "«recurrente»")):
    reiniciar()
    ft.ESCRITO = {"promovente": IMPULSORA, "caracter": "tercero interesado"}
    _r = generar(_ar(**_kw14), acto=ACTO_AR)
    ok(LL["leer_escrito"] == 0 and _r.encargo.recurrente == DIRECTOR
       and not _hay(_r.avisos, "SE LEYÓ DEL ESCRITO"),
       f"si el secretario tecleó {_que}, el escrito no se lee y manda lo tecleado")

reiniciar()
ft.ESCRITO = RuntimeError("lector roto a propósito")
r14e = generar(_ar(quejoso="", recurrente=""), acto=ACTO_AR)
ok(LL["leer_escrito"] == 1 and _hay(r14e.avisos, "NO TECLEASTE QUIÉN RECURRE Y DEL ESCRITO NO SE PUDO LEER (RuntimeError)"),
   "si la lectura del escrito revienta: aviso, y el adelanto sigue")
ok(r14e.encargo.quejoso == "Ana López Ruiz" and r14e.encargo.papel_recurrente == "",
   "…con la quejosa leída de los documentos, como antes, y sin papel fijado")

reiniciar()
ft.ESCRITO = {}
r14f = generar(_ar(quejoso="", recurrente=""), acto=ACTO_AR)
ok(_hay(r14f.avisos, "NO TECLEASTE QUIÉN RECURRE Y EL ESCRITO NO LO DICE CON CLARIDAD")
   and r14f.encargo.recurrente == "",
   "si el escrito no dice quién recurre: aviso, y nada se supone")

reiniciar()
del ft.leer_escrito
r14g = generar(_ar(quejoso="", recurrente=""), acto=ACTO_AR)
ft.leer_escrito = _leer_escrito
ok(LL["leer_escrito"] == 0 and r14g.encargo.quejoso == "Ana López Ruiz"
   and not _hay(r14g.avisos, "SE LEYÓ DEL ESCRITO") and not _hay(r14g.avisos, "NO TECLEASTE QUIÉN RECURRE"),
   "si ficha_tramite aún no trae leer_escrito: el camino de antes, sin aviso nuevo ni error")

reiniciar()
ft.ESCRITO = {"promovente": "Ana López Ruiz", "caracter": "tercero interesado"}
r14h = generar(_ar(quejoso="", recurrente=""), acto=ACTO_AR)
ok(r14h.encargo.recurrente == "" and r14h.encargo.papel_recurrente == ""
   and _hay(r14h.avisos, "EL ESCRITO DEL RECURSO NOMBRA A «Ana López Ruiz»"),
   "el escrito llama tercera a la parte a la que el juzgado amparó: no se fija nada y se avisa")

reiniciar()
_MP = "Agente del Ministerio Público de la Federación adscrito al Juzgado Primero de Distrito"
ft.ESCRITO = {"promovente": _MP, "caracter": "Ministerio Público de la Federación"}
r14i = generar(_ar(quejoso="", recurrente=""), acto=ACTO_AR)
ok(r14i.encargo.recurrente == _MP and r14i.encargo.papel_recurrente == ""
   and "artículo 5o., fracción IV" in _primero(r14i.avisos, "SE LEYÓ DEL ESCRITO QUIÉN RECURRE"),
   "el Ministerio Público: recurrente fijado, su papel no (lo funda la ficha) y se avisa")

reiniciar()
ft.ESCRITO = {"promovente": IMPULSORA, "caracter": "tercero"}
generar(encargo("revision_fiscal", encabezado="REVISIÓN FISCAL: 512/2026", quejoso=""))
ok(LL["leer_escrito"] == 0, "en la revisión fiscal no se lee (recurre siempre la autoridad)")
reiniciar()
generar(encargo(quejoso=""))
ok(LL["leer_escrito"] == 0, "en el amparo directo no se lee (no es un recurso)")

os.environ["PROCEDENCIA_POR_TIPO"] = "0"
reiniciar()
ft.ESCRITO = {"promovente": IMPULSORA, "caracter": "tercero"}
r14j = generar(_ar(quejoso="", recurrente=""), acto=ACTO_AR)
ok(LL["leer_escrito"] == 0 and r14j.encargo.recurrente == "" and r14j.encargo.papel_recurrente == "",
   "sin la bandera, el escrito no se lee: el camino de hoy")
os.environ["PROCEDENCIA_POR_TIPO"] = "todos"

reiniciar()


async def _lento(cliente, texto_escrito, tipo):
    await asyncio.sleep(2)
    return {"promovente": IMPULSORA, "caracter": "tercero"}


ft.leer_escrito = _lento
_tope = ra.TOPE_LEER_ESCRITO
ra.TOPE_LEER_ESCRITO = 0.05
_t0 = _time_t.perf_counter()
r14k = generar(_ar(quejoso="", recurrente=""), acto=ACTO_AR)
_dt14 = _time_t.perf_counter() - _t0
ra.TOPE_LEER_ESCRITO = _tope
ft.leer_escrito = _leer_escrito
ok(_hay(r14k.avisos, "NO TECLEASTE QUIÉN RECURRE Y LA LECTURA DEL ESCRITO NO CONTESTÓ A TIEMPO")
   and _dt14 < 1.5 and r14k.encargo.recurrente == "",
   f"la lectura del escrito tiene tope: si no contesta, se sigue sin ella ({_dt14:.2f} s)")

ok(ra.caracter_del_escrito("hoy TERCERO INTERESADO") == "tercero"
   and ra.caracter_del_escrito("tercera interesada") == "tercero"
   and ra.caracter_del_escrito("Autoridad Responsable") == "autoridad"
   and ra.caracter_del_escrito("parte quejosa") == "quejoso"
   and ra.caracter_del_escrito("Ministerio Público") == "ministerio_publico"
   and ra.caracter_del_escrito("apoderado") == "" and ra.caracter_del_escrito(None) == "",
   "caracter_del_escrito: lo que no reconoce no lo supone")
_e14 = _ar(quejoso=DIRECTOR)
_e14.es_recurso = True
_e14.papel_recurrente = "tercero"
ok(ra.papel_del_recurrente(_e14) == "tercero",
   "papel_del_recurrente respeta el papel fijado aunque el nombre parezca de autoridad")
_e14.papel_recurrente = "otra cosa"
ok(ra.papel_del_recurrente(_e14, None, "") in ("autoridad", "tercero", "quejoso", ""),
   "un papel fijado que no es de la lista no se usa")

# ═══════════════════════════════════════════════════════════════════════════
print("\n15 · LA FORMA DE NOTIFICACIÓN QUE CONSTA MANDA SOBRE LA DE OMISIÓN (D2; Q 24 y Q 172/2026)")
_Q = dict(encabezado="QUEJA: 512/2026", quejoso="Ana López Ruiz")


def _decl_forma(forma):
    return {"forma_notificacion": forma, "fuentes": {"forma_notificacion": "secretario"}}


reiniciar()
_e15 = _ar("queja", **_Q)
_e15.tramite_declarado = _decl_forma("electronica")
r15 = generar(_e15, acto=ACTO_AR)
ok(r15.encargo.regla_surtimiento == "electronica" and r15.computo.regla.clave == "electronica"
   and r15.computo.surtio == datetime.date(2026, 1, 20),
   "queja de la quejosa, regla de omisión y notificación electrónica declarada: fr. III, surte el mismo día")
ok(len(_hay(r15.avisos, "LA NOTIFICACIÓN FUE ELECTRÓNICA (así lo declaraste")) == 1,
   "un aviso lo dice, con la fuente")
ok(LL["armar"] and LL["armar"][-1]["regla"] == "electronica",
   "la ficha se arma después: su forma dice lo mismo que el cómputo")
_c15 = LL["dg_componer"][-1] if LL["dg_componer"] else {}
ok(_c15.get("computo") is r15.computo, "el documento se compone con el cómputo rehecho")

reiniciar()
_e15b = _ar(**dict(_Q, encabezado="AMPARO EN REVISIÓN: 512/2026"))
_e15b.tramite_declarado = _decl_forma("lista")
r15b = generar(_e15b, acto=ACTO_AR)
_pers15 = f0.computar(datetime.date(2026, 1, 20), datetime.date(2026, 2, 3), "personal", 10, DIRECTOR,
                      None, "amparo_revision", None)
ok(r15b.encargo.regla_surtimiento == "lista" and r15b.computo.surtio == _pers15.surtio
   and _hay(r15b.avisos, "LA NOTIFICACIÓN FUE POR LISTA"),
   "por lista: la regla «lista» (el considerando ya no dice «de manera personal»), mismas fechas")

reiniciar()
ft.ESCRITO = {"promovente": "Ana López Ruiz", "caracter": "quejosa", "forma_notificacion": "Electrónica"}
r15c = generar(_ar(quejoso="", recurrente=""), acto=ACTO_AR)
ok(r15c.encargo.regla_surtimiento == "electronica"
   and "así se lee en el escrito del recurso" in _primero(r15c.avisos, "LA NOTIFICACIÓN FUE ELECTRÓNICA"),
   "la forma leída del escrito también cuenta, y el aviso dice de dónde salió")

reiniciar()
_e15d = _ar(**dict(_Q, encabezado="AMPARO EN REVISIÓN: 512/2026"))
_e15d.tramite_auto = {"forma_notificacion": "lista", "admision": {"fecha": "2026-02-10"},
                      "fuentes": {"admision.fecha": "auto"}}
r15d = generar(_e15d, acto=ACTO_AR)
ok(r15d.encargo.regla_surtimiento == "lista"
   and "así se lee en el auto de admisión" in _primero(r15d.avisos, "LA NOTIFICACIÓN FUE POR LISTA"),
   "la del auto también, con su fuente")

reiniciar()
ft.ESCRITO = {"promovente": "Ana López Ruiz", "caracter": "quejosa", "forma_notificacion": "electronica"}
_e15e = _ar(quejoso="", recurrente="")
_e15e.tramite_declarado = _decl_forma("lista")
r15e = generar(_e15e, acto=ACTO_AR)
ok(r15e.encargo.regla_surtimiento == "lista", "lo declarado manda sobre lo leído del escrito")

reiniciar()
_e15f = _ar(**dict(_Q, encabezado="AMPARO EN REVISIÓN: 512/2026"))
_e15f.tramite_declarado = _decl_forma("oficio")
r15f = generar(_e15f, acto=ACTO_AR)
ok(r15f.encargo.regla_surtimiento == "personal"
   and _hay(r15f.avisos, "LA NOTIFICACIÓN CONSTA COMO HECHA POR OFICIO"),
   "«por oficio» a la quejosa no puede ser: no se cambia la regla y se avisa")

reiniciar()
_e15g = _ar(**dict(_Q, encabezado="AMPARO EN REVISIÓN: 512/2026", regla_surtimiento="lista"))
_e15g.tramite_declarado = _decl_forma("electronica")
r15g = generar(_e15g, acto=ACTO_AR)
ok(r15g.encargo.regla_surtimiento == "lista" and not _hay(r15g.avisos, "LA NOTIFICACIÓN FUE"),
   "si el secretario eligió una regla (no la de omisión), se respeta")

for _tipo15, _kw15 in (("amparo_directo", {}),
                       ("revision_fiscal", dict(encabezado="REVISIÓN FISCAL: 512/2026"))):
    reiniciar()
    _e15h = encargo(_tipo15, **_kw15)
    _e15h.tramite_declarado = _decl_forma("electronica")
    _r = generar(_e15h)
    ok(_r.encargo.regla_surtimiento == "personal" and not _hay(_r.avisos, "LA NOTIFICACIÓN FUE"),
       f"en {_tipo15} no se toca: la notificación del acto la rige su propia ley")

reiniciar()
_e15i = _ar(recurrente=DIRECTOR)
ft.ESCRITO = None
_e15i.tramite_auto = {"forma_notificacion": "electrónica"}
r15i = generar(_e15i, acto=ACTO_AR)
ok(r15i.encargo.regla_surtimiento == "electronica"
   and "así se lee en el auto de admisión" in _primero(r15i.avisos, "RECURRE UNA AUTORIDAD Y LA NOTIFICACIÓN FUE ELECTRÓNICA"),
   "la autoridad notificada electrónicamente (leído del auto): fr. III, y el aviso no dice «lo declaraste»")

reiniciar()
_e15j = _ar(**dict(_Q, encabezado="AMPARO EN REVISIÓN: 512/2026",
                  presentacion=datetime.date(2026, 2, 5)))
_e15j.tramite_declarado = _decl_forma("electronica")
r15j = generar(_e15j, acto=ACTO_AR)
ok(r15j.computo.oportuna is False and r15j.avisos.count(ra.AVISO_EXTEMPORANEA) == 1
   and len(_hay(r15j.avisos, "CON LA FORMA DE NOTIFICACIÓN QUE CONSTA EL RECURSO SALE FUERA DE PLAZO")) == 1,
   "el día que decide: con la electrónica sale fuera, con la de omisión salía en tiempo, y se dice")

os.environ["PROCEDENCIA_POR_TIPO"] = "0"
reiniciar()
_e15k = _ar(**dict(_Q, encabezado="AMPARO EN REVISIÓN: 512/2026"))
_e15k.tramite_declarado = _decl_forma("electronica")
r15k = generar(_e15k, acto=ACTO_AR)
ok(r15k.encargo.regla_surtimiento == "personal" and not _hay(r15k.avisos, "LA NOTIFICACIÓN FUE"),
   "sin la bandera, la forma no cambia la regla de la quejosa (sólo la autoridad es [siempre])")
os.environ["PROCEDENCIA_POR_TIPO"] = "todos"

_ef = _ar()
_ef.tramite_declarado = {"forma_notificacion": "Por lista"}
ok(ra.forma_de_notificacion(_ef, {"forma_notificacion": "electrónica"}, {"forma_notificacion": "oficio"})
   == ("lista", "secretario"), "forma_de_notificacion: lo declarado primero")
_ef.tramite_declarado = {}
_ef.tramite_auto = {"forma_notificacion": "oficio"}
ok(ra.forma_de_notificacion(_ef, {"forma_notificacion": "electrónica"}) == ("electronica", "escrito")
   and ra.forma_de_notificacion(_ef, {}) == ("oficio", "auto")
   and ra.forma_de_notificacion(_ar(), None, {"forma_notificacion": "lista"}) == ("lista", "acto")
   and ra.forma_de_notificacion(_ar(), {"forma_notificacion": "por correo"}) == ("", ""),
   "…luego el escrito, el auto y el acto; lo que no se reconoce no cuenta")

# ═══════════════════════════════════════════════════════════════════════════
print("\n16 · LA REVISIÓN FISCAL POR CORREO SE MIDE CON EL DEPÓSITO (D3; RF 2/2025)")
TFJA = "Sala Regional del Centro II del Tribunal Federal de Justicia Administrativa"
UNIDAD = "Unidad de Asuntos Jurídicos de la Administración Desconcentrada Jurídica de Querétaro"
_plazo_rf = ta.plazo_de("revision_fiscal", "")["dias"]
_V = f0.computar(datetime.date(2026, 1, 20), None, "personal", _plazo_rf, TFJA, None,
                 "revision_fiscal", None).vencimiento
_RECEP = _V + datetime.timedelta(days=4)
_RF = dict(encabezado="REVISIÓN FISCAL: 512/2026", responsable=TFJA, recurrente=UNIDAD, presentacion=_RECEP)


def _rf(deposito=None, **kw):
    base = dict(_RF)
    base.update(kw)
    _e = encargo("revision_fiscal", **base)
    if deposito is not None:
        _e.tramite_declarado = {"deposito_postal": deposito.isoformat(),
                                "fuentes": {"deposito_postal": "secretario"}}
    return _e


_con_recepcion = f0.computar(datetime.date(2026, 1, 20), _RECEP, "personal", _plazo_rf, TFJA, None,
                             "revision_fiscal", None)
ok(_con_recepcion.oportuna is False, f"(con la recepción del {_RECEP} el recurso sale fuera: vence el {_V})")


def _sin_tilde(x):
    import unicodedata as _u
    return "".join(ch for ch in _u.normalize("NFD", str(x)) if _u.category(ch) != "Mn").upper()


def _av_dep(avisos):
    """Los avisos que dicen que la oportunidad se midió con el depósito: el de
    `computar` (si lo recibe) o el del cableado (si no), nunca los dos."""
    return [a for a in avisos if "SE MIDIO CON LA FECHA" in _sin_tilde(a) and "DEPOSIT" in _sin_tilde(a)]


reiniciar()
r16 = generar(_rf(deposito=_V))
ok(r16.computo.oportuna is True, f"depositado el {_V}, el último día: en tiempo")
ok(ra.AVISO_EXTEMPORANEA not in r16.avisos, "sin el aviso de extemporánea")
_a16 = (_av_dep(r16.avisos)[:1] or [""])[0]
ok(len(_av_dep(r16.avisos)) == 1, "UN aviso dice que se midió con el depósito (no dos)")
ok("XV.4o.1 A (11a.)" in _a16 and "COMPRUEBALO" in _sin_tilde(_a16)
   and "RECEPCION" in _sin_tilde(_a16) and "EXTEMPORANEO" in _sin_tilde(_a16),
   "y dice el criterio, que con la recepción salía fuera, y pide comprobarlo")
ok(getattr(r16.computo, "deposito", None) == _V,
   "el cómputo lleva el depósito con que se midió (para el párrafo y la sesión)")
ok(getattr(r16.computo, "recepcion", None) in (None, _RECEP),
   "y, si computar la guarda, la recepción en la Sala")
ok(LL["dg_componer"] and LL["dg_componer"][-1]["computo"] is r16.computo,
   "el documento se compone con el cómputo del depósito")
_par16 = f0.parrafo_oportunidad(r16.computo, tipo="revision_fiscal")
ok(f0.fecha_en_letra(_V) in _par16 and f0.fecha_en_letra(_RECEP) not in _par16.split("entonces")[-1],
   "el párrafo de oportunidad remata con la fecha del depósito, no con la de recepción")

# EL OTRO WORKER, CON LAS FUNCIONES DE main.py DE VERDAD: se guarda la sesión
# en una base de juguete (jsonb = ida y vuelta JSON) y se rehidrata.


class _BaseDeJuguete:
    def __init__(self):
        self.fila = None

    def table(self, nombre):
        return _Consulta(self)


class _Consulta:
    def __init__(self, base):
        self.base = base

    def upsert(self, fila, on_conflict=None):
        self.base.fila = json.loads(json.dumps(fila, ensure_ascii=False, default=str))
        return self

    def select(self, *a, **k):
        return self

    def eq(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def execute(self):
        if self.base.fila is None:
            return types.SimpleNamespace(data=[])
        return types.SimpleNamespace(data=[dict(self.base.fila, actualizado_en="sello-1", propuestas=[])])


_BD = _BaseDeJuguete()
S = _de_main("_taller_guardar_sesion", "_taller_recuperar_sesion_crudo",
             supabase_admin=_BD, _TALLER_SESIONES={}, _taller_llave=lambda c, n: (c, n),
             time=_time_t, os=os, _types=types,
             _te=types.SimpleNamespace(huella_contraste=lambda r: "huella", esta_completo=lambda m: False),
             _taller_global_de_fila=lambda est, r: None)


def _otro_worker(r, numero="512/2026"):
    S["_taller_guardar_sesion"]("pruebas@x", numero, r, tempfile.mkdtemp(prefix="cableado_"))
    S["_TALLER_SESIONES"].clear()
    return S["_taller_recuperar_sesion_crudo"]("pruebas@x", numero)["resultado"]


_s16 = _otro_worker(r16)
ok(_s16.computo.oportuna is True and _s16.computo.vencimiento == r16.computo.vencimiento
   and getattr(_s16.computo, "deposito", None) == _V,
   "el worker que rehidrata la sesión (main real) también cuenta con el depósito: en tiempo")

reiniciar()
r16b = generar(_rf())
ok(r16b.computo.oportuna is False and not _av_dep(r16b.avisos),
   "sin depósito, la recepción: fuera de plazo, como antes")

reiniciar()
_e16c = _rf(deposito=_V)
_e16c.tramite_declarado["via_presentacion"] = "responsable"
r16c = generar(_e16c)
# CUARTA RONDA (E6): ya no. Un oficio que viaja por correo también lo recibe
# la responsable; con depósito anterior a la presentación, cuenta el depósito
# (la regla única, la misma con que el compositor narra el depósito). Antes:
# fuera de plazo, y el resultando decía que se depositó en plazo (RF 2/2025).
ok(r16c.computo.oportuna is True and len(_av_dep(r16c.avisos)) == 1,
   "E6: la ficha dice vía «responsable» pero consta el depósito: cuenta el depósito (en tiempo)")

reiniciar()
r16d = generar(_rf(deposito=_RECEP + datetime.timedelta(days=1)))
ok(r16d.computo.oportuna is False and _hay(r16d.avisos, "EL DEPÓSITO POSTAL (")
   and "ES POSTERIOR A LA PRESENTACIÓN" in _primero(r16d.avisos, "EL DEPÓSITO POSTAL ("),
   "un depósito posterior a la recepción no puede ser: se mide con la recepción y se avisa")

reiniciar()
r16e = generar(_rf(deposito=_RECEP))
ok(r16e.computo.oportuna is False and not _av_dep(r16e.avisos),
   "depósito igual a la recepción: nada cambia")

os.environ["PROCEDENCIA_POR_TIPO"] = "0"
reiniciar()
r16f = generar(_rf(deposito=_V))
ok(r16f.computo.oportuna is False and r16f.encargo.tramite is None,
   "sin la bandera no hay ficha ni depósito: el cómputo de hoy")
os.environ["PROCEDENCIA_POR_TIPO"] = "todos"

reiniciar()
_e16g = _ar(recurrente=DIRECTOR)
_e16g.tramite_declarado = {"deposito_postal": "2026-01-25", "fuentes": {"deposito_postal": "secretario"}}
r16g = generar(_e16g, acto=ACTO_AR)
ok(not _av_dep(r16g.avisos) and ra.deposito_que_cuenta(r16g.encargo) is None,
   "fuera de la revisión fiscal el depósito no cuenta")

_e16 = _rf()
ok(ra.args_de_presentacion(_e16) == (_RECEP, {}), "args_de_presentacion sin depósito: la recepción")
import inspect as _insp16
_comp_real = ra.f0.computar
if "deposito" in _insp16.signature(_comp_real).parameters:
    ok(ra.args_de_presentacion(_e16, _V) == (_RECEP, {"deposito": _V}),
       "computar recibe `deposito`: la recepción se queda como presentación y el depósito va aparte")
else:
    ok(ra.args_de_presentacion(_e16, _V) == (_V, {}),
       "computar todavía no recibe `deposito`: el depósito ocupa la presentación")


def _computar_sin_deposito(notificacion, presentacion=None, regla="personal", plazo=15,
                           responsable=None, inhabiles_extra=None, tipo_asunto="amparo_directo",
                           inhabiles_responsable=None, surtio_manual=None, plazo_anios=0):
    return _comp_real(notificacion, presentacion, regla, plazo, responsable, inhabiles_extra,
                      tipo_asunto, inhabiles_responsable, surtio_manual=surtio_manual,
                      plazo_anios=plazo_anios)


ra.f0.computar = _computar_sin_deposito
try:
    ok(ra.args_de_presentacion(_e16, _V) == (_V, {}),
       "con un computar SIN el parámetro: el depósito ocupa la presentación (la fecha que cuenta)")
    reiniciar()
    r16h = generar(_rf(deposito=_V))
    _a16h = _primero(r16h.avisos, "REVISIÓN FISCAL POR CORREO: LA OPORTUNIDAD SE MIDIÓ")
    ok(r16h.computo.oportuna is True and len(_av_dep(r16h.avisos)) == 1
       and "XV.4o.1 A (11a.)" in _a16h and "CON LA FECHA DE RECEPCIÓN EL RECURSO SALÍA EXTEMPORÁNEO" in _a16h,
       "…y el adelanto mide en tiempo, con el aviso del cableado (el de computar no existe ahí)")
    ok(getattr(r16h.computo, "deposito", None) == _V and getattr(r16h.computo, "recepcion", None) == _RECEP,
       "…y cuelga del cómputo el depósito y la recepción")
finally:
    ra.f0.computar = _comp_real

# ═══════════════════════════════════════════════════════════════════════════
print("\n17 · AL RETOMAR, LO LEÍDO VUELVE COMO LEÍDO, NO COMO DEL SECRETARIO (rev_6)")
_doble17 = sys.modules["ficha_tramite"]
sys.modules["ficha_tramite"] = _ftr
try:
    _e17 = encargo("amparo_revision", encabezado="AMPARO EN REVISIÓN: 512/2026", responsable=DIRECTOR)
    _acto_viejo = {"acto": {"fecha": "2026-01-15", "organo": "Juzgado Primero de Distrito en el Estado de Querétaro",
                            "expediente": "905/2025"},
                   "fuentes": {"acto.fecha": "acto", "acto.organo": "acto", "acto.expediente": "acto"}}
    _auto17 = {"turno": {"fecha": "2026-02-12", "ponente": "Ana Pérez"},
               "fuentes": {"turno.fecha": "auto", "turno.ponente": "auto"}}
    _f17 = _ftr.armar(_e17, declarado=_ftr.de_formulario({"fecha_admision": "2026-02-09"}),
                      acto=_acto_viejo, auto=_auto17)
    _p17 = ra.tramite_para_pantalla(_f17)
    ok(_p17.get("tramite", {}).get("fecha_admision") == "2026-02-09"
       and _p17["tramite"].get("fecha_acto") == "" and _p17["tramite"].get("organo_acto") == ""
       and _p17["tramite"].get("fecha_turno") == "",
       "«tramite» lleva sólo lo que confirmó el secretario")
    ok(_p17.get("leido", {}).get("fecha_acto") == "2026-01-15"
       and _p17["leido"].get("organo_acto", "").startswith("Juzgado Primero")
       and _p17["leido"].get("expediente_origen") == "905/2025"
       and _p17["leido"].get("fecha_turno") == "2026-02-12" and "fecha_admision" not in _p17["leido"],
       "«leido» lleva lo del acto y del auto, sin pisar lo del secretario")
    ok(_p17.get("fuentes", {}).get("fecha_admision") == "secretario"
       and _p17["fuentes"].get("fecha_acto") == "acto" and _p17["fuentes"].get("fecha_turno") == "auto",
       "«fuentes» dice de dónde salió cada una")
    # Lo que la pantalla reenvía al regenerar es SÓLO lo del secretario: el
    # acto corregido manda. Con el reenvío de antes (la ficha entera), lo viejo
    # le ganaba como «secretario».
    _acto_nuevo = {"acto": {"fecha": "2026-01-20", "organo": "Juzgado Segundo de Distrito en el Estado de Querétaro",
                            "expediente": "77/2025"},
                   "fuentes": {"acto.fecha": "acto", "acto.organo": "acto", "acto.expediente": "acto"}}
    _reenvio, _ = ra.tramite_de_formulario(json.dumps({k: v for k, v in _p17["tramite"].items() if v}))
    _f17b = _ftr.armar(_e17, declarado=_reenvio, acto=_acto_nuevo, auto=_auto17)
    ok(_f17b["acto"]["fecha"] == "2026-01-20" and _f17b["acto"]["expediente"] == "77/2025"
       and _f17b["fuentes"].get("acto.fecha") == "acto",
       "al regenerar con el acto corregido, lo nuevo del papel manda (no lo viejo como «secretario»)")
    _reenvio_viejo, _ = ra.tramite_de_formulario(json.dumps(
        {k: v for k, v in _ftr.a_formulario(_f17).items() if v}))
    _f17c = _ftr.armar(_e17, declarado=_reenvio_viejo, acto=_acto_nuevo, auto=_auto17)
    ok(_f17c["acto"]["fecha"] == "2026-01-15",
       "(control: reenviando la ficha entera, como antes, lo viejo ganaba)")
    ok(ra.tramite_para_pantalla(None) == {} and ra.tramite_para_pantalla("x") == {},
       "sin ficha, nada")

    M17 = _de_main("_taller_tramite_para_pantalla")
    _ses17 = types.SimpleNamespace(tramite=_f17)
    _pp = M17["_taller_tramite_para_pantalla"](_ses17)
    ok(isinstance(_pp, dict) and set(_pp) == {"tramite", "tramite_leido", "tramite_fuentes"}
       and _pp["tramite"] == _p17["tramite"] and _pp["tramite_leido"] == _p17["leido"]
       and _pp["tramite_fuentes"] == _p17["fuentes"],
       "main: /taller/contexto-del-asunto recibe tramite (secretario), tramite_leido y tramite_fuentes")
    ok(M17["_taller_tramite_para_pantalla"](types.SimpleNamespace(tramite=None)) is None,
       "main: sin ficha, None (la clave ni se escribe)")
    os.environ["PROCEDENCIA_POR_TIPO"] = "0"
    ok(M17["_taller_tramite_para_pantalla"](_ses17) is None, "main: sin la bandera, None")
    os.environ["PROCEDENCIA_POR_TIPO"] = "todos"
finally:
    sys.modules["ficha_tramite"] = _doble17
ok("**(_taller_tramite_para_pantalla(e) or {})" in _fuente("taller_contexto_del_asunto")
   or "**(_taller_tramite_para_pantalla(e) or {})" in _src,
   "contexto-del-asunto mete las tres claves en «encargo»")

# ═══════════════════════════════════════════════════════════════════════════
print("\n18 · EL AUTO SE LEE ACOTADO Y EN UN HILO (rev_6: 15 s con 166 páginas en el bucle)")
reiniciar()
_largo = "AUTO DE PRESIDENCIA. Querétaro, a diez de febrero de dos mil veintiséis. " * 1500
_d18 = ra.leer_auto_tramite(_largo, "amparo_directo")
ok(LL["leer_auto"] == 1 and len(LL["auto_texto"] or "") <= ra.TOPE_AUTO,
   f"un «auto» de {len(_largo)} caracteres se lee hasta {ra.TOPE_AUTO}")
ok(_hay(_d18.get("avisos", []), "EL AUTO DE ADMISIÓN QUE SUBISTE TRAE")
   and "AVISO DEL AUTO" in _d18.get("avisos", []),
   "con aviso de que se recortó, junto a los de la lectura")
reiniciar()
_corto = "AUTO DE PRESIDENCIA " * 20
_d18b = ra.leer_auto_tramite(_corto, "amparo_directo")
ok(LL["auto_texto"] == _corto.strip() and not _hay(_d18b.get("avisos", []), "EL AUTO DE ADMISIÓN QUE SUBISTE TRAE"),
   "un auto normal se lee entero y sin aviso")


def _llamadas_al_auto(nombre):
    """(directas, en hilo): las llamadas a leer_auto_tramite dentro de `nombre`."""
    n = _fun.get(nombre)
    directas = en_hilo = 0
    for x in ast.walk(n) if n else []:
        if not isinstance(x, ast.Call):
            continue
        f_ = ast.unparse(x.func)
        if f_.endswith("leer_auto_tramite"):
            directas += 1
        if f_ == "asyncio.to_thread" and x.args and ast.unparse(x.args[0]).endswith("leer_auto_tramite"):
            en_hilo += 1
    return directas, en_hilo


for _ep in ("taller_desde_admision", "taller_adelanto", "taller_desde_expediente"):
    _dir, _hilo = _llamadas_al_auto(_ep)
    ok(_dir == 0 and _hilo >= 1, f"{_ep}: leer_auto_tramite sólo en asyncio.to_thread ({_hilo} en hilo, {_dir} directas)")

# ═══════════════════════════════════════════════════════════════════════════
print("\n19 · /taller/desde-expediente RECIBE Y PASA «tramite_json» (rev_6, camino SISE)")
_dx = _fun.get("taller_desde_expediente")
_def_dx = {a.arg: d for a, d in zip(_dx.args.args[-len(_dx.args.defaults):], _dx.args.defaults)} if _dx else {}
ok("tramite_json" in _args("taller_desde_expediente") and ast.unparse(_def_dx.get("tramite_json")) == "Form('')",
   "el parámetro existe y es opcional: Form('')")
_llamada_ad = [x for x in ast.walk(_dx) if isinstance(x, ast.Call) and ast.unparse(x.func) == "_con_omisiones"]
_kw_ad = {k.arg: ast.unparse(k.value) for k in (_llamada_ad[0].keywords if _llamada_ad else [])}
ok(_kw_ad.get("tramite_json", "").startswith("tramite_json"),
   f"y se pasa a taller_adelanto ({_kw_ad.get('tramite_json')!r})")

# ═══════════════════════════════════════════════════════════════════════════
print("\n20 · D4: EL DOCSTRING DE /taller/reglas-surtimiento DICE LO QUE HACE")
_doc = ast.get_docstring(_fun["taller_reglas_surtimiento"]) or ""
ok("Sin él, la lista de\n    siempre" not in _doc and "la lista de siempre" not in _doc.replace("\n", " "),
   "ya no dice «Sin él, la lista de siempre» (falso desde la segunda ronda)")
_doc1 = " ".join(_doc.split())
ok("SEA QUIEN SEA LA RESPONSABLE" in _doc1 and "ARTÍCULO 31 DE LA LEY DE AMPARO" in _doc1,
   "dice que en la revisión y la queja rigen las del artículo 31 sea quien sea la responsable")
_ar_tfja = asyncio.run(M["taller_reglas_surtimiento"](tipo_asunto="amparo_revision", responsable=TFJA))
ok(_ar_tfja["por_omision"] == "personal"
   and not any(x["clave"] in ("lfpca_boletin", "tja_qro_boletin") for x in _ar_tfja["reglas"]),
   "y es verdad: un AR con responsable del TFJA ofrece las del 31, sin el boletín del TFJA")

# ═══════════════════════════════════════════════════════════════════════════
print("\n21 · POR HTTP, «tramite_json» SÓLO CON LAS CLAVES DEL FORMULARIO (rev_6)")
reiniciar()
_d21, _m21 = ra.tramite_de_formulario(json.dumps({"fecha_admision": "2026-02-09",
                                                  "presentacion": "2026-02-05", "sentido": "lo que sea"}))
ok(_m21 == "" and LL["form"] == {"fecha_admision": "2026-02-09"},
   "presentacion y sentido no llegan a de_formulario")
ok("presentacion" not in _d21 and _hay(_d21.get("avisos", []), "«TRÁMITE EN ESTE TRIBUNAL» TRAÍA CLAVES QUE NO SON")
   and "presentacion, sentido" in _primero(_d21.get("avisos", []), "«TRÁMITE EN ESTE TRIBUNAL» TRAÍA"),
   "y un aviso nombra las que se tiraron")
_d21b, _ = ra.tramite_de_formulario(json.dumps({"fecha_admision": "2026-02-09"}))
ok(not _d21b.get("avisos"), "con sólo claves del contrato, sin aviso")

# ═══════════════════════════════════════════════════════════════════════════
print("\n22 · LA SESIÓN GUARDA EL PAPEL FIJADO Y EL OTRO WORKER LO VE (main real)")
_s14 = _otro_worker(r14, "631/2025")
ok(_s14.encargo.papel_recurrente == "tercero" and _s14.encargo.recurrente == IMPULSORA,
   "el encargo rehidratado trae a la recurrente y su papel")
ok(ra.papel_del_recurrente(_s14.encargo, _s14.partes, "") == "tercero",
   "y el papel que ve el otro worker es el mismo")
_s14b = _otro_worker(r14b, "632/2025")
ok(_s14b.encargo.papel_recurrente == "autoridad" and _s14b.encargo.regla_surtimiento == "oficio"
   and _s14b.computo.surtio == r14b.computo.surtio,
   "la autoridad leída del escrito: el otro worker la ve y computa igual (oficio)")
ok('"papel_recurrente"' in _fuente("_taller_guardar_sesion")
   and "papel_recurrente=str(e.get(\"papel_recurrente\"" in _fuente("_taller_recuperar_sesion_crudo"),
   "(lectura estática) se escribe al guardar y se lee al recuperar")
# AL RETOMAR Y REGENERAR (la pantalla): contexto-del-asunto devuelve quién
# recurre y su carácter, y /taller/adelanto acepta el carácter de vuelta.
_ctx_src = _src[_src.index('"encargo": {\n            "numero": getattr(e, "numero", "") or "",'):][:3000]
ok('"recurrente": getattr(e, "recurrente", "") or ""' in _ctx_src
   and '"papel_recurrente": getattr(e, "papel_recurrente", "") or ""' in _ctx_src,
   "contexto-del-asunto devuelve en «encargo» al recurrente y su carácter")
_ad22 = _fun["taller_adelanto"]
_def22 = {a.arg: d for a, d in zip(_ad22.args.args[-len(_ad22.args.defaults):], _ad22.args.defaults)}
ok(ast.unparse(_def22.get("papel_recurrente")) == "Form('')",
   "/taller/adelanto acepta «papel_recurrente» (opcional)")
_enc22 = [x for x in ast.walk(_ad22) if isinstance(x, ast.Call) and ast.unparse(x.func) == "_ra.Encargo"]
_kw22 = {k.arg: ast.unparse(k.value) for k in (_enc22[0].keywords if _enc22 else [])}
ok("_ra.PAPELES_FIJOS" in _kw22.get("papel_recurrente", ""),
   "y lo pasa al encargo sólo si es uno de los papeles que se pueden fijar")
_expr22 = next((k.value for k in (_enc22[0].keywords if _enc22 else []) if k.arg == "papel_recurrente"), None)


def _p22(valor):
    """La expresión de main.py, evaluada tal cual con `papel_recurrente`."""
    return eval(compile(ast.Expression(copy.deepcopy(_expr22)), "main.py", "eval"),
                {"_ra": ra, "papel_recurrente": valor})


ok(_expr22 is not None and _p22(" Tercero ") == "tercero" and _p22("autoridad") == "autoridad"
   and _p22("ministerio_publico") == "" and _p22("") == "" and _p22(None) == "",
   "la expresión de main: «Tercero» vale; el Ministerio Público, lo vacío y None, no")


# ═══════════════════════════════════════════════════════════════════════════
# CUARTA RONDA (3-oct-2026) — dueño F de FIXES_R4.md
# ═══════════════════════════════════════════════════════════════════════════
print("\n23 · E6: UNA SOLA REGLA DEL CORREO (ficha_tramite.via_postal), TAMBIÉN EN EL OTRO WORKER")
# RF 2/2025 con la ficha de fichas2: vía «responsable» + depósito del 24 de
# octubre. El resultando narraba el depósito y el considerando desechaba.
_a23 = [a for a in r16c.avisos if a.startswith("LA VÍA DE PRESENTACIÓN DE LA FICHA DICE «responsable»")]
ok(len(_a23) == 1 and f0.fecha_en_letra(_V) in _a23[0] and ra.AVISO_EXTEMPORANEA not in r16c.avisos,
   "un aviso (uno) dice que la vía decía «responsable» y se contó el depósito; sin el de extemporánea")
_s23 = _otro_worker(r16c, "513/2026")
ok(_s23.computo.oportuna is True and getattr(_s23.computo, "deposito", None) == _V,
   "el worker que rehidrata la sesión (main real) cuenta igual: con el depósito, en tiempo")
_llam23 = []


def _vp_no(f):
    _llam23.append(dict(f))
    return False


def _vp_roto(f):
    raise RuntimeError("roto a propósito")


ft.via_postal = _vp_no
try:
    _e23 = _rf()
    _e23.tramite = {"deposito_postal": _V.isoformat(), "via_presentacion": "postal"}
    ok(ra.deposito_que_cuenta(_e23) is None and _llam23
       and _llam23[-1].get("presentacion") == _RECEP.isoformat(),
       "deposito_que_cuenta pregunta a ficha_tramite.via_postal (con la presentación tecleada) y la obedece")
    ft.via_postal = lambda f: True
    _e23.tramite = {"deposito_postal": _V.isoformat(), "via_presentacion": "responsable"}
    ok(ra.deposito_que_cuenta(_e23) == _V,
       "…y si la pieza dice que viajó por correo, cuenta aunque la vía diga otra cosa")
    ft.via_postal = _vp_roto
    ok(ra.deposito_que_cuenta(_e23) == _V
       and ra.via_postal_de({"deposito_postal": "2026-03-01"}, datetime.date(2026, 2, 1)) is False,
       "si via_postal lanza, la misma regla aquí (depósito no posterior a la presentación)")
finally:
    del ft.via_postal
ok(ra.via_postal_de({}, _RECEP) is False and ra.via_postal_de(None) is False
   and ra.via_postal_de({"deposito_postal": _V.isoformat(), "via_presentacion": "responsable"}, _RECEP) is True
   and ra.via_postal_de({"deposito_postal": _RECEP.isoformat()}, _RECEP) is True,
   "sin la pieza: sin depósito, no; depósito anterior o igual a la presentación, sí, diga lo que diga la vía")
# UN AVISO POR DATO (E11): si la validación de la ficha ya dijo lo de la vía.
_e23b = _rf()
_e23b.tramite = {"deposito_postal": _V.isoformat(), "via_presentacion": "responsable",
                 "avisos": ["HAY FECHA DE DEPÓSITO POSTAL Y LA VÍA DICE «responsable»: compruébalo."]}
ok(ra._aviso_via_con_deposito(_e23b, _V, []) == "", "si la validación de la ficha ya lo dijo, no se repite")
_e23b.tramite = {"deposito_postal": _V.isoformat(), "via_presentacion": "postal"}
ok(ra._aviso_via_con_deposito(_e23b, _V, []) == "", "con la vía «postal», nada que avisar")
if callable(getattr(_ftr, "via_postal", None)):
    ok(_ftr.via_postal({"deposito_postal": _V.isoformat(), "via_presentacion": "responsable",
                        "presentacion": _RECEP.isoformat()}) is True
       and _ftr.via_postal({"deposito_postal": (_RECEP + datetime.timedelta(days=1)).isoformat(),
                            "presentacion": _RECEP.isoformat()}) is False,
       "ficha_tramite.via_postal real: depósito anterior ⇒ correo aunque diga «responsable»; posterior, no")
    ft.via_postal = _ftr.via_postal
    try:
        reiniciar()
        _e23c = _rf(deposito=_V)
        _e23c.tramite_declarado["via_presentacion"] = "responsable"
        r23c = generar(_e23c)
        ok(r23c.computo.oportuna is True and len(_av_dep(r23c.avisos)) == 1,
           "y con ella el adelanto cuenta el depósito (RF 2/2025)")
    finally:
        del ft.via_postal
else:
    print("   (aviso, no cuenta) ficha_tramite real todavía sin via_postal: la regla se probó con la de aquí")

# ═══════════════════════════════════════════════════════════════════════════
print("\n24 · E7: EL AUTO QUE FORMA Y REGISTRA («fecha_registro») VIAJA POR tramite_json Y AL RETOMAR")
reiniciar()
_form24 = {"fecha_admision": "2026-05-15", "fecha_registro": "2026-02-10"}
_d24, _m24 = ra.tramite_de_formulario(json.dumps(_form24))
ok(_m24 == "" and LL["form"] == _form24 and not _hay(_d24.get("avisos") or [], "«TRÁMITE EN ESTE TRIBUNAL» TRAÍA"),
   "con las 22 claves, «fecha_registro» llega a de_formulario y no se tira")
_e24 = _rf()
_e24.tramite_declarado = _d24
r24 = generar(_e24)
ok(LL["armar"] and (LL["armar"][-1]["declarado"] or {}).get("registro") == {"fecha": "2026-02-10"}
   and (r24.encargo.tramite or {}).get("registro") == {"fecha": "2026-02-10"},
   "armar lo recibe en lo declarado y la ficha del encargo lo lleva")
ok(LL["componer_tipo"] and LL["componer_tipo"][-1]["ficha"].get("registro") == {"fecha": "2026-02-10"},
   "el compositor lo recibe en la ficha")
_s24 = _otro_worker(r24, "514/2026")
ok((_s24.encargo.tramite or {}).get("registro") == {"fecha": "2026-02-10"}
   and _s24.encargo.tramite.get("admision") == {"fecha": "2026-05-15"},
   "y vuelve con la sesión al otro worker (main real), distinto de la admisión")
# SISE: los autos de Presidencia, cada uno leído por separado.
_f24 = ra.fundir_autos(
    [{"admision": {"fecha": "2026-02-10"}, "fuentes": {"admision.fecha": "auto"}},
     {"admision": {"fecha": "2026-05-15"}, "fuentes": {"admision.fecha": "auto"}}],
    [{"turno": {"fecha": "2026-05-16", "ponente": "Jenica Campos Juárez", "titulo": "Magistrada"},
      "fuentes": {"turno.fecha": "auto", "turno.ponente": "auto", "turno.titulo": "auto"}}])
_a24 = _primero(_f24.get("avisos") or [], "EN EL EXPEDIENTE HAY 2 AUTOS DE PRESIDENCIA")
ok(_f24["admision"] == {"fecha": "2026-02-10"} and "diez de febrero" in _a24 and "quince de mayo" in _a24
   and "forma y registra" in _a24,
   "SISE (RF 7/2025): dos autos con fecha de admisión distinta y ninguno leído como registro: se dice, "
   "sin adivinar cuál es cuál")
_f24b = ra.fundir_autos(
    [{"registro": {"fecha": "2026-02-10"}, "fuentes": {"registro.fecha": "auto"}},
     {"admision": {"fecha": "2026-05-15"}, "fuentes": {"admision.fecha": "auto"}}], [])
ok(_f24b.get("registro") == {"fecha": "2026-02-10"} and _f24b.get("admision") == {"fecha": "2026-05-15"}
   and (_f24b.get("fuentes") or {}).get("registro.fecha") == "auto"
   and not _hay(_f24b.get("avisos") or [], "EN EL EXPEDIENTE HAY"),
   "si la lectura los distinguió, el registro y la admisión viajan cada uno de su auto, sin aviso")
ok(_f24["turno"].get("titulo") == "Magistrada" and _f24["fuentes"].get("turno.titulo") == "auto",
   "(E3) el cargo del ponente viaja con su turno y su fuente")
if "fecha_registro" in getattr(_ftr, "CLAVES_FORMULARIO", ()):
    _doble24 = sys.modules["ficha_tramite"]
    sys.modules["ficha_tramite"] = _ftr
    try:
        _d24r, _m24r = ra.tramite_de_formulario(json.dumps(_form24))
        ok(_m24r == "" and ((_d24r.get("registro") or {}).get("fecha") == "2026-02-10")
           and (_d24r.get("fuentes") or {}).get("registro.fecha") == "secretario",
           "ficha_tramite real: «fecha_registro» entra como registro.fecha del secretario")
        _e24r = encargo("revision_fiscal", encabezado="REVISIÓN FISCAL: 7/2025", tramite_declarado=_d24r)
        _p24 = ra.tramite_para_pantalla(ra.armar_tramite(_e24r))
        ok(_p24.get("tramite", {}).get("fecha_registro") == "2026-02-10"
           and _p24.get("fuentes", {}).get("fecha_registro") == "secretario",
           "al retomar, vuelve en «tramite» como del secretario")
        _f24r = _ftr.armar(_e24r, declarado=_ftr.de_formulario({"fecha_admision": "2026-05-15"}),
                           auto={"registro": {"fecha": "2026-02-10"}, "fuentes": {"registro.fecha": "auto"}})
        _p24b = ra.tramite_para_pantalla(_f24r)
        ok(_p24b.get("leido", {}).get("fecha_registro") == "2026-02-10"
           and _p24b.get("tramite", {}).get("fecha_registro") == ""
           and _p24b.get("fuentes", {}).get("fecha_registro") == "auto",
           "y el leído del auto vuelve como leído, no como del secretario")
    finally:
        sys.modules["ficha_tramite"] = _doble24
else:
    print(f"   (aviso, no cuenta) ficha_tramite real todavía con {len(getattr(_ftr, 'CLAVES_FORMULARIO', ()))} "
          f"claves y sin «fecha_registro»: el viaje se comprobó con el doble de 22")

# ═══════════════════════════════════════════════════════════════════════════
print("\n25 · E3: EL CARGO DEL PONENTE LLEGA A LA CARÁTULA (dentro del valor del ponente)")
# AD 128, RF 4, Q 342 del banco: el auto dice «a la ponencia a cargo de la
# magistrada Jenica Campos Juárez» y la carátula salía «MAGISTRADO PONENTE».
_fic25 = {"returno": {"fecha": "2026-05-20", "ponente": "Jenica Campos Juárez", "titulo": "Magistrada"},
          "turno": {"fecha": "2026-05-16", "ponente": "Luis Armando Pérez Topete"},
          "fuentes": {"returno.fecha": "auto", "returno.ponente": "auto", "returno.titulo": "auto",
                      "turno.fecha": "auto", "turno.ponente": "auto"}}
_pl25 = {"ponente_returno": "Jenica Campos Juárez", "ponente_turno": "Luis Armando Pérez Topete"}
_c25 = ra.ponentes_con_cargo(_pl25, _fic25)
ok(_c25["ponente_returno"] == "Magistrada Jenica Campos Juárez"
   and _c25["ponente_turno"] == "Luis Armando Pérez Topete"
   and _pl25["ponente_returno"] == "Jenica Campos Juárez",
   "el cargo leído del auto va delante del nombre; sin cargo leído, el nombre tal cual (nunca por el "
   "nombre de pila); la entrada no se toca")
ok(ra.ponentes_con_cargo(_c25, _fic25) == _c25, "no se duplica («Magistrada Magistrada …»)")
ok(ra.ponentes_con_cargo(_pl25, dict(_fic25, fuentes={"returno.ponente": "secretario",
                                                     "returno.titulo": "auto"}))["ponente_returno"]
   == "Jenica Campos Juárez",
   "el cargo del auto no se pega al nombre que tecleó el secretario")
ok(ra.ponentes_con_cargo({"ponente_returno": "Ana Pérez"}, _fic25)["ponente_returno"] == "Ana Pérez",
   "ni a otra persona")
ok(ra.ponentes_con_cargo(_pl25, {"returno": {"ponente": "Jenica Campos Juárez",
                                             "titulo": "Licenciada"}})["ponente_returno"]
   == "Jenica Campos Juárez",
   "un tratamiento no es un cargo: no va al rótulo")
ok(ra.ponentes_con_cargo(_pl25, {"returno": [{"ponente": "Otro Nombre", "titulo": "Magistrado"},
                                             {"ponente": "Jenica Campos Juárez", "titulo": "Magistrada"}]}
                         )["ponente_returno"] == "Magistrada Jenica Campos Juárez",
   "returno en lista: el último")
ok(ra.ponentes_con_cargo(None, _fic25) == {} and ra.ponentes_con_cargo(_pl25, None) == _pl25,
   "nunca lanza")
ok(ra.falta_cargo_del_ponente(_pl25) and not ra.falta_cargo_del_ponente({"ponente_returno": "Magistrada X"})
   and not ra.falta_cargo_del_ponente({}) and not ra.falta_cargo_del_ponente(None),
   "falta_cargo_del_ponente: sólo con un ponente sin cargo")
_doble25 = sys.modules["ficha_tramite"]
sys.modules["ficha_tramite"] = _ftr
try:
    _p25 = ra.tramite_para_pantalla(dict(_fic25, tipo="amparo_directo"))
    ok(_p25.get("leido", {}).get("ponente_returno") == "Magistrada Jenica Campos Juárez"
       and _p25.get("fuentes", {}).get("ponente_returno") == "auto"
       and _p25.get("leido", {}).get("ponente_turno") == "Luis Armando Pérez Topete",
       "al retomar, el ponente leído vuelve con su cargo (y su fuente)")
    _d25, _ = ra.tramite_de_formulario(json.dumps({"fecha_returno": "2026-05-20",
                                                   "ponente_returno": _p25["leido"]["ponente_returno"]}))
    _e25 = encargo("amparo_directo", magistrado="", tramite_declarado=_d25)
    _f25 = _ftr.armar(_e25, declarado=_d25)
    _rot25, _nom25, _av25 = dg._ponente_de_caratula({"magistrado": ""}, _f25, {})
    ok(_rot25 == "MAGISTRADA PONENTE" and "Magistrada" not in _nom25 and "Jenica Campos Juárez" in _nom25,
       f"confirmado por el secretario y de vuelta: la carátula dice «{_rot25}: {_nom25}»")
    ok(ra.tramite_para_pantalla(_f25).get("tramite", {}).get("ponente_returno")
       == "Magistrada Jenica Campos Juárez",
       "y al retomar otra vez, el secretario ve lo que confirmó, con el cargo")
    _d25c, _ = ra.tramite_de_formulario(json.dumps({"fecha_returno": "2026-05-20",
                                                    "ponente_returno": "Jenica Campos Juárez"}))
    _rot25c = dg._ponente_de_caratula({"magistrado": ""}, _ftr.armar(_e25, declarado=_d25c), {})[0]
    ok(_rot25c != "MAGISTRADA PONENTE",
       f"(control: sin el cargo en el valor la carátula no puede saberlo y dice «{_rot25c}»)")
finally:
    sys.modules["ficha_tramite"] = _doble25

# /taller/desde-admision (main real, con dobles de lo que llama)
_LEE25 = []
_FA25 = {"ficha": {}}
_fa25 = types.ModuleType("fase_admision")
_fauto25 = types.ModuleType("fase_autos")
_ft25 = types.ModuleType("ficha_tramite")


async def _fa25_leer(cliente, texto):
    return copy.deepcopy(_FA25["ficha"])


def _ft25_leer_auto(texto, tipo=""):
    _LEE25.append(tipo)
    return {"returno": {"fecha": "2026-05-20", "ponente": "Jenica Campos Juárez", "titulo": "Magistrada"},
            "fuentes": {"returno.fecha": "auto", "returno.ponente": "auto", "returno.titulo": "auto"}}


def _ft25_a_formulario(f):
    r = (f or {}).get("returno") or {}
    return {"fecha_returno": r.get("fecha", ""), "ponente_returno": r.get("ponente", "")}


_fa25.leer = _fa25_leer
_fauto25.leer = lambda t: {}
_ft25.leer_auto, _ft25.a_formulario = _ft25_leer_auto, _ft25_a_formulario


async def _texto25(subida):
    return "AUTO DE PRESIDENCIA. Se returnan los autos a la ponencia a cargo de la magistrada. " * 20


M25 = _de_main("taller_desde_admision", _taller_puerta=lambda u: None,
               _extract_text_from_upload=_texto25, HTTPException=RuntimeError, chat_client=None,
               asyncio=asyncio, err=str, _taller_rige_procedencia=lambda *a, **k: True,
               File=lambda *a, **k: None, Form=lambda *a, **k: None, UploadFile=object)
_mods25 = {k: sys.modules.get(k) for k in ("fase_admision", "fase_autos", "ficha_tramite")}
sys.modules.update({"fase_admision": _fa25, "fase_autos": _fauto25, "ficha_tramite": _ft25})
try:
    _base25 = {"tipo_asunto": "amparo_directo", "numero": "128/2025", "materia": "civil",
               "responsable": "Primera Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro"}
    _FA25["ficha"] = dict(_base25, tramite={"fecha_returno": "2026-05-20",
                                            "ponente_returno": "Jenica Campos Juárez"})
    _LEE25.clear()
    _r25a = asyncio.run(M25["taller_desde_admision"](user_email="pruebas@x", admision=object()))
    ok(_r25a["ficha"]["tramite"].get("ponente_returno") == "Magistrada Jenica Campos Juárez" and len(_LEE25) == 1,
       "desde-admision: la propuesta de fase_admision sin cargo se completa con el del auto (una relectura)")
    _FA25["ficha"] = dict(_base25)
    _LEE25.clear()
    _r25b = asyncio.run(M25["taller_desde_admision"](user_email="pruebas@x", admision=object()))
    ok(_r25b["ficha"]["tramite"].get("ponente_returno") == "Magistrada Jenica Campos Juárez" and len(_LEE25) == 1,
       "desde-admision: si la lee main, con su cargo y sin leer el auto dos veces")
    _FA25["ficha"] = dict(_base25, tramite={"ponente_returno": "Magistrada Jenica Campos Juárez"})
    _LEE25.clear()
    _r25c = asyncio.run(M25["taller_desde_admision"](user_email="pruebas@x", admision=object()))
    ok(_r25c["ficha"]["tramite"].get("ponente_returno") == "Magistrada Jenica Campos Juárez" and not _LEE25,
       "desde-admision: si ya trae el cargo, ni se relee")
finally:
    for _k25, _v25 in _mods25.items():
        if _v25 is None:
            sys.modules.pop(_k25, None)
        else:
            sys.modules[_k25] = _v25

# ═══════════════════════════════════════════════════════════════════════════
print("\n26 · C6: LOS ASUNTOS RELACIONADOS VIAJAN POR tramite_json, CON LA SESIÓN Y AL RETOMAR")
# SEXTA RONDA (3-oct-2026). David: «hay que habilitar en el taller la opción de
# con un clic precisar si existen asuntos relacionados y con ello se genera el
# considerando». «relacionados» viaja como «fecha_registro»: por tramite_json,
# en la ficha del encargo, al compositor y a documento_generado, con la sesión
# al otro worker y, al retomar, de vuelta a la tarjeta como del secretario.
os.environ["PROCEDENCIA_POR_TIPO"] = "todos"
reiniciar()
_REL26 = "amparo_directo|452/2025|misma_sesion;revision_fiscal|33/2024|resuelto"
_ESP26 = [{"tipo": "amparo_directo", "numero": "452/2025", "estado": "misma_sesion"},
          {"tipo": "revision_fiscal", "numero": "33/2024", "estado": "resuelto"}]
_form26 = {"fecha_admision": "2026-02-09", "relacionados": _REL26}
_d26, _m26 = ra.tramite_de_formulario(json.dumps(_form26))
ok(_m26 == "" and LL["form"] == _form26 and not _hay(_d26.get("avisos") or [], "«TRÁMITE EN ESTE TRIBUNAL» TRAÍA"),
   "con las 22 claves, «relacionados» llega a de_formulario y no se tira")
_e26 = encargo("amparo_directo")
_e26.tramite_declarado = _d26
r26 = generar(_e26)
ok(LL["armar"] and (LL["armar"][-1]["declarado"] or {}).get("relacionados") == _ESP26
   and (r26.encargo.tramite or {}).get("relacionados") == _ESP26,
   "armar lo recibe en lo declarado y la ficha del encargo lo lleva")
ok(LL["componer_tipo"] and LL["componer_tipo"][-1]["ficha"].get("relacionados") == _ESP26,
   "el compositor lo recibe en la ficha (para el V I S T O)")
ok(LL["dg_componer"] and (LL["dg_componer"][-1]["datos"].get("tramite") or {}).get("relacionados") == _ESP26,
   "y documento_generado, en datos['tramite'] (para el rubro y el considerando)")
_s26 = _otro_worker(r26, "516/2026")
ok((_s26.encargo.tramite or {}).get("relacionados") == _ESP26,
   "y vuelve con la sesión al otro worker (main real, jsonb de ida y vuelta)")
reiniciar()
_d26b, _ = ra.tramite_de_formulario(json.dumps({"fecha_admision": "2026-02-09"}))
_e26b = encargo("amparo_directo")
_e26b.tramite_declarado = _d26b
r26b = generar(_e26b)
ok(not (r26b.encargo.tramite or {}).get("relacionados"),
   "sin marcar nada (interruptor apagado: la clave no viaja), la ficha no trae relacionados")

if "relacionados" in getattr(_ftr, "CLAVES_FORMULARIO", ()):
    _doble26 = sys.modules["ficha_tramite"]
    sys.modules["ficha_tramite"] = _ftr
    try:
        # Por HTTP de_formulario no sabe el número del asunto: el propio
        # (512/2026) pasa y lo quita armar, con aviso.
        _d26r, _m26r = ra.tramite_de_formulario(json.dumps(
            dict(_form26, relacionados=_REL26 + ";amparo_directo|512/2026")))
        ok(_m26r == "" and len(_d26r.get("relacionados") or []) == 3
           and (_d26r.get("fuentes") or {}).get("relacionados") == "secretario",
           "ficha_tramite real: «relacionados» entra como lista del secretario")
        _f26r = ra.armar_tramite(encargo("amparo_directo", tramite_declarado=_d26r))
        ok(_f26r.get("relacionados") == _ESP26 and _hay(_f26r.get("avisos") or [], "EL ASUNTO RELACIONADO")
           and (_f26r.get("fuentes") or {}).get("relacionados") == "secretario",
           "armar_tramite: el propio asunto fuera, con aviso; los otros dos, del secretario")
        ok((ra.tramite_serializable(_f26r) or {}).get("relacionados") == _ESP26,
           "la ficha serializada para la sesión los conserva")
        _p26 = ra.tramite_para_pantalla(_f26r)
        ok(_p26.get("tramite", {}).get("relacionados") == _REL26
           and _p26.get("fuentes", {}).get("relacionados") == "secretario"
           and "relacionados" not in _p26.get("leido", {}),
           f"al retomar, vuelve en «tramite» como del secretario: «{_p26.get('tramite', {}).get('relacionados')}»")
        _reenvio26, _ = ra.tramite_de_formulario(json.dumps({k: v for k, v in _p26["tramite"].items() if v}))
        ok(_reenvio26.get("relacionados") == _ESP26,
           "y al regenerar desde la tarjeta reconstruida, los mismos relacionados")
        M26 = _de_main("_taller_tramite_para_pantalla")
        _pp26 = M26["_taller_tramite_para_pantalla"](types.SimpleNamespace(tramite=_f26r))
        ok(isinstance(_pp26, dict) and _pp26["tramite"].get("relacionados") == _REL26
           and _pp26["tramite_fuentes"].get("relacionados") == "secretario",
           "main: /taller/contexto-del-asunto lleva «relacionados» en «tramite» (el interruptor se enciende)")
        _f26s = ra.armar_tramite(encargo("amparo_directo", tramite_declarado=_d26b))
        ok(ra.tramite_para_pantalla(_f26s).get("tramite", {}).get("relacionados") == "",
           "sin relacionados, «» (el interruptor queda apagado)")
        ok(_ftr.a_formulario(_ftr.leer_auto("Querétaro, Querétaro, a nueve de febrero de dos mil veintiséis. "
                                            "Fórmese el expediente número 512/2026, relacionado con el amparo "
                                            "directo 452/2025. Se admite la demanda."))["relacionados"] == "",
           "desde-admision: lo que proponga el auto nunca trae relacionados (ni aunque el auto los mencione)")
    finally:
        sys.modules["ficha_tramite"] = _doble26
else:
    print(f"   (aviso, no cuenta) ficha_tramite real todavía con {len(getattr(_ftr, 'CLAVES_FORMULARIO', ()))} "
          f"claves y sin «relacionados»: el viaje se comprobó con el doble de 22")


os.environ.pop("PROCEDENCIA_POR_TIPO", None)
print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
