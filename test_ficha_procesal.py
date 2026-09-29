# -*- coding: utf-8 -*-
"""LA FICHA PROCESAL DEL ASUNTO (SPEC_E2, 28-sep-2026), SIN LLAMAR A NINGÚN MODELO.

AR 631/2025: el proyecto intermedio salió con «QUEJOSA Y RECURRENTE» para la
tercera interesada, «ÓRGANO RECURRIDO: MAGISTRADA…» y una síntesis en la que
la quejosa «adquirió el inmueble». Cada pieza adivinaba los papeles. Aquí se
comprueba:

1. La ficha del 631 sintético: recurre la tercera interesada contra una
   concesión, con un sobreseimiento de otro acto que nadie impugnó.
2. Una revisión de la quejosa (el juzgado negó; y otra en que sobreseyó).
3. Un amparo directo.
4. Que la ficha llega como bloque de DATOS a la propuesta, el contraste, la
   deliberación, el estudio, la síntesis y la estructura, y a la tarjeta; y
   que sin ficha cada prompt queda idéntico al de antes.
5. Que las piezas leen la ficha en vez de volver a deducir los papeles.

Todo es SINTÉTICO: nombres, números y fechas de prueba. Los modelos son falsos
y se cuentan las llamadas.

    .venv/bin/python test_ficha_procesal.py
"""
import asyncio
import datetime as _dt
import json
import os
import re
import sys
import types

os.environ.pop("ESTUDIO_PROMPT", None)

import deliberacion as dl
import documento_generado as dg
import fase5_propuesta as f5
import fase6_estudio as f6
import fase_partes as fpa
import fase_sintesis as fs
import fases123_pipeline as f123
import ficha_procesal as fp
import redactor_adelanto as ra
import tarjeta_decision as td

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


AQUI = os.path.dirname(os.path.abspath(__file__))

# ═══ EL 631 SINTÉTICO ════════════════════════════════════════════════════════
QUEJOSA = "Unión Ejemplo, A.C."
RECURRENTE = "Inmobiliaria Ejemplo, S.A. de C.V."
RESOL = ("La Justicia de la Unión ampara y protege a Unión Ejemplo, A.C., en contra del acto "
         "atribuido a la Sala Civil Uno del Tribunal Superior de Justicia, por los motivos "
         "expresados en el considerando séptimo de esta sentencia y para los efectos precisados en "
         "el último considerando.")
RECURRIDA = (
    "JUZGADO SÉPTIMO DE DISTRITO EN EL ESTADO DE EJEMPLO\n"
    "AMPARO INDIRECTO 950/2024\n"
    "VISTOS para resolver los autos del juicio de amparo 950/2024. "
    "CONSIDERANDO QUINTO. Se sobresee respecto del acto atribuido al Juzgado Quinto Civil. "
    "CONSIDERANDO SÉPTIMO. Es fundado el segundo concepto de violación y suficiente para conceder; "
    "resulta innecesario el estudio de los restantes conceptos de violación.\n"
    "Por lo expuesto, se R E S U E L V E: PRIMERO. Se sobresee en el juicio respecto del acto del "
    "Juzgado Quinto Civil. SEGUNDO. " + RESOL + " Notifíquese.")
PROBLEMAS = [{"pregunta": "¿La sustitución alteró la cosa juzgada?", "cubre": [1],
              "clase": "fondo", "jerarquia": "principal",
              "resolvio": "La sustitución alteró la cosa juzgada.",
              "combate": "La sustitución sólo cambió quién ejecuta."}]


def fases_631(**kw):
    f = f123.Fases123(resumen_acto="El Juzgado de Distrito sobreseyó respecto del Juzgado Quinto y "
                                   "concedió el amparo contra la resolución de la Sala.",
                      problemas=[dict(p) for p in PROBLEMAS],
                      fuentes=[RECURRIDA, "AGRAVIOS. PRIMERO. La sustitución no alteró la cosa juzgada."])
    f.resolutivo_recurrida = RESOL
    f.resolvio_a_quo = "niega"             # lo que guardaba la sesión del 631 (el recuento)
    for k, v in kw.items():
        setattr(f, k, v)
    return f


def encargo_631(**kw):
    e = ra.Encargo(numero="631/2025", encabezado="AMPARO EN REVISIÓN 631/2025",
                   # Lo que guardó el formulario en el 631: la RECURRENTE como «quejoso».
                   quejoso=RECURRENTE, magistrado="M", secretario="S",
                   notificacion=_dt.date(2025, 4, 25), presentacion=_dt.date(2025, 5, 15),
                   tipo_asunto="amparo_revision", responsable="Magistrada de la Sala Civil Uno")
    e.es_recurso = True
    for k, v in kw.items():
        setattr(e, k, v)
    return e


def resultado(e=None, f=None, partes=None):
    return types.SimpleNamespace(encargo=e or encargo_631(), fases=f or fases_631(),
                                 partes=partes, avisos=[])


# ═══════════════════════════════════════════════════════════════════════════
print("\n1 · EL 631: RECURRE LA TERCERA INTERESADA CONTRA UNA CONCESIÓN")
fi = fp.de_resultado(resultado())
ok(fi.get("formato") == 1 and fi["tipo_asunto"] == "amparo_revision", "ficha de una revisión, formato 1")
ok(fi["quejosa"]["nombre"] == QUEJOSA and fi["quejosa"]["fuente"] == "resolutivo del juzgado",
   "la quejosa sale del resolutivo del juzgado, no de lo tecleado (que era la recurrente)")
rc = fi["recurrente"]
ok(rc["nombre"] == RECURRENTE and rc["papel"] == "tercero" and rc["caracter"] == "parte tercera interesada",
   "recurre la tercera interesada, con su carácter")
ok(fi["organo_recurrido"]["nombre"].startswith("Juzgado Séptimo de Distrito"),
   "el órgano recurrido es el Juzgado de Distrito, no la Magistrada del acto reclamado")
ok(not any("magistrad" in r["autoridad"].lower() for r in fi["responsables"])
   and any("Sala Civil Uno" in r["autoridad"] for r in fi["responsables"])
   and any("Juzgado Quinto Civil" in r["autoridad"] for r in fi["responsables"]),
   "responsables: la Sala (sin repetirla por su Magistrada) y el Juzgado Quinto, cada uno con su acto")
_pts = fi["resolvio"]["puntos"]
ok([(p["ordinal"], p["que"]) for p in _pts] == [("PRIMERO", "sobresee"), ("SEGUNDO", "concede")],
   "lo que resolvió el juzgado, punto por punto: sobresee (PRIMERO) y concede (SEGUNDO)")
ok(fi["resolvio"]["que"] == "concede" and fi["resolvio"]["fuente"] == "punto resolutivo del juzgado",
   "qué hizo el juzgado: la misma fuente única de `que_hizo_el_juzgado` (su resolutivo manda)")
ok(any(t["nombre"] == RECURRENTE for t in fi["terceros"]), "la recurrente figura como tercera interesada")
ok(len(fi["materia_revision"]) == 1 and fi["materia_revision"][0].startswith("concesión"),
   "materia de la revisión: la concesión, que es lo que perjudica a la tercera")
ok(len(fi["firme"]) == 1 and fi["firme"][0].startswith("sobreseimiento"),
   "el sobreseimiento del otro acto queda firme: nadie lo impugnó")
a93 = fi["art_93"]
ok("fracción VI" in a93["rige"], "rige el art. 93, fracción VI")
ok(a93["si_prospera"]["rama"] == "revoca_fondo_niega" and a93["si_prospera"]["reasuncion"] == "concesion"
   and "conceptos de violación" in a93["si_prospera"]["resultado"]
   and a93["si_prospera"].get("alcance", "").startswith("sólo en la materia de la revisión"),
   "si prospera: revoca la concesión, reasume jurisdicción y sólo en la materia de la revisión")
ok(a93["si_no_prospera"]["rama"] == "confirma_concede", "si no prospera: confirma, subsiste la concesión")
ok(fi["adhesivo"]["consta"] is False, "la revisión adhesiva no consta")
# Lo que falta, dicho (revisión adversarial de la fase E): el acto del
# Juzgado Quinto (el punto sólo lo nombra) y si hay más terceros que la
# recurrente. Nada más.
ok(len(fi["avisos"]) == 2
   and any("acto reclamado a Juzgado Quinto Civil" in a for a in fi["avisos"])
   and any("más terceros interesados" in a for a in fi["avisos"]),
   f"en el 631 sintético sólo faltan el acto del Juzgado Quinto y los demás terceros ({fi['avisos']})")
_sala = next(r for r in fi["responsables"] if "Sala Civil Uno" in r["autoridad"])
ok(_sala["acto"] == "la resolución de la Sala" and "resumen de la recurrida" in _sala["fuente"],
   "el acto de la Sala no es «el acto atribuido a la Sala»: sale de lo concedido, con su fuente")
ok(a93["si_no_prospera"]["fundamento"] == "artículo 93, fracción VI, de la Ley de Amparo",
   "si no prospera, la confirmación se funda en la fr. VI (recurre la tercera), no en «V y VI»")
ok("fracciones II y III" in (a93.get("previo") or ""),
   "recurre la tercera: la II y la III, como orden de estudio antes del fondo")

b = fp.bloque(fi)
ok(b.startswith("LA FICHA PROCESAL DEL ASUNTO") and "[fuente: resolutivo del juzgado]" in b,
   "el bloque son datos con su fuente")
for rot in ("Parte quejosa (promovió el amparo): Unión Ejemplo, A.C.",
            "Recurrente: Inmobiliaria Ejemplo, S.A. de C.V. (parte tercera interesada)",
            "Órgano que dictó la resolución recurrida: Juzgado Séptimo de Distrito",
            "Materia de la revisión", "No impugnado por quien recurre",
            "Artículo 93 de la Ley de Amparo que rige: artículo 93, fracción VI",
            "Si el recurso prospera:", "Si no prospera:"):
    ok(rot in b, f"el bloque trae «{rot[:60]}»")
# NI FRASES MODELO: nada de puntos resolutivos escritos ni fórmulas del proyecto.
ok(not re.search(r"PRIMERO\.\s+Se|ampara y protege a|Se propone|este Tribunal Colegiado", b),
   "el bloque no trae frases del proyecto que el modelo pueda copiar")
ok(all(re.match(r"^(LA FICHA|  [^ ].*: )", l) for l in b.strip().splitlines()),
   "cada renglón es «rótulo: valor»")

ln = fp.linea(fi)
ok("recurre: Inmobiliaria Ejemplo" in ln and "tercera interesada" in ln and "art. 93, fr. VI" in ln
   and "firme: sobreseimiento" in ln and "\n" not in ln, f"la línea de la tarjeta: {ln[:140]}…")
pt = fp.para_tarjeta(fi)
ok(set(pt) >= {"tipo", "quejosa", "responsables", "terceros", "recurrida", "recurrente", "materia", "avisos"}
   and pt["recurrente"] == {"quien": RECURRENTE, "caracter": "tercera interesada"}
   and pt["recurrida"]["organo"].startswith("Juzgado Séptimo")
   and [x["sentido"] for x in pt["recurrida"]["resolvio"]] == ["sobresee", "concede"]
   and "firme" not in pt["materia"] and pt["firme"].startswith("sobreseimiento")
   and (pt.get("art_93") or {}).get("fraccion") == "VI",
   "para la tarjeta, con la forma FICHA del contrato (lo firme, aparte; la fracción que rige)")
ok(json.loads(json.dumps(fi, ensure_ascii=False)) == fi, "la ficha es JSON puro")

# Con la ficha de partes leída de la sentencia (el tercero con nombre propio).
_p = fpa.Partes(quejoso=QUEJOSA, tercero_interesado=RECURRENTE,
                autoridad_responsable="Sala Civil Uno del Tribunal Superior de Justicia",
                tipo_asunto="amparo_revision")
fi_p = fp.de_resultado(resultado(partes=_p))
ok(fi_p["terceros"] == [{"nombre": RECURRENTE, "fuente": "ficha de partes"}]
   and fi_p["organo_recurrido"]["nombre"].startswith("Juzgado Séptimo"),
   "con ficha de partes: el tercero de la ficha, y el órgano sigue siendo el juzgado (no la Sala)")

# Sin resolutivo legible: se dice.
_f_sin = fases_631(resolutivo_recurrida="", resolvio_a_quo="", fuentes=["", ""])
fi_sin = fp.de_resultado(resultado(f=_f_sin))
ok(any("No consta qué resolvió el juzgado" in a for a in fi_sin["avisos"])
   and any("órgano" in a for a in fi_sin["avisos"])
   and "No consta:" in fp.bloque(fi_sin),
   "lo que no consta se dice en los avisos y en el bloque")

# ═══════════════════════════════════════════════════════════════════════════
print("\n2 · REVISIÓN DE LA QUEJOSA")
RES_NIEGA = ("La Justicia de la Unión no ampara ni protege a Pedro Ejemplo Pérez, contra el acto "
             "reclamado al Director de Catastro Municipal, por las razones expuestas.")
REC_NIEGA = ("JUZGADO SEGUNDO DE DISTRITO EN EL ESTADO DE EJEMPLO\nAMPARO INDIRECTO 12/2025\n"
             "Por lo expuesto, se R E S U E L V E: ÚNICO. " + RES_NIEGA + " Notifíquese.")
e_q = encargo_631(numero="100/2025", quejoso="Pedro Ejemplo Pérez", responsable="Director de Catastro Municipal")
f_q = fases_631(fuentes=[REC_NIEGA, "AGRAVIOS"], resolutivo_recurrida=RES_NIEGA, resolvio_a_quo="")
fq = fp.de_resultado(resultado(e_q, f_q))
ok(fq["recurrente"]["papel"] == "quejoso" and fq["recurrente"]["nombre"] == "Pedro Ejemplo Pérez"
   and fq["recurrente"]["aparte"] == "", "recurre la propia quejosa (quejosa y recurrente coinciden)")
ok(fq["resolvio"]["que"] == "niega" and fq["materia_revision"] and fq["materia_revision"][0].startswith("negativa")
   and fq["firme"] == [], "materia: la negativa; nada firme")
ok(fq["art_93"]["rige"] == "artículo 93, fracción V, de la Ley de Amparo"
   and fq["art_93"]["si_prospera"]["rama"] == "revoca_fondo_concede"
   and fq["art_93"]["si_prospera"]["reasuncion"] == ""
   and fq["art_93"]["si_no_prospera"]["rama"] == "confirma_niega",
   "fracción V: si prospera, revoca la negativa y concede; si no, confirma")
ok(fq["responsables"] and fq["responsables"][0]["autoridad"] == "Director de Catastro Municipal",
   "la responsable del acto, leída del resolutivo")
ok("recurre la quejosa" in fp.linea(fq), "la línea dice que recurre la quejosa")

RES_SOB = ("Se sobresee en el juicio de amparo promovido por Pedro Ejemplo Pérez, contra el acto "
           "reclamado al Director de Catastro Municipal.")
f_s = fases_631(fuentes=[REC_NIEGA.replace(RES_NIEGA, RES_SOB), "AGRAVIOS"],
                resolutivo_recurrida=RES_SOB, resolvio_a_quo="")
fsb = fp.de_resultado(resultado(e_q, f_s))
ok(fsb["resolvio"]["que"] == "sobresee" and "fracciones I y V" in fsb["art_93"]["rige"]
   and fsb["art_93"]["si_prospera"]["reasuncion"] == "sobreseimiento"
   and fsb["materia_revision"][0].startswith("sobreseimiento"),
   "la quejosa contra un sobreseimiento: fracciones I y V, se levanta y se estudian los conceptos")

# La quejosa que recurre la concesión combate sus términos o efectos: no queda firme.
e_qc = encargo_631(quejoso=QUEJOSA)
fqc = fp.de_resultado(resultado(e_qc, fases_631(fuentes=[RECURRIDA.replace(
    "PRIMERO. Se sobresee en el juicio respecto del acto del Juzgado Quinto Civil. SEGUNDO. ", "ÚNICO. "),
    "AGRAVIOS"])))
ok(fqc["recurrente"]["papel"] == "quejoso" and fqc["firme"] == []
   and fqc["materia_revision"] and fqc["materia_revision"][0].startswith("concesión")
   and "términos o efectos" in fqc["materia_revision"][0]
   and fqc["art_93"]["rige"] == "artículo 93, fracción V, de la Ley de Amparo",
   "la quejosa contra su propia concesión: la materia son sus términos o efectos (fr. V)")

# La autoridad que recurre una concesión (el 711/2025: la UIF).
e_a = encargo_631(numero="711/2025", quejoso="Unidad de Inteligencia Financiera",
                  responsable="Unidad de Inteligencia Financiera")
fa = fp.de_resultado(resultado(e_a, fases_631()))
ok(fa["recurrente"]["papel"] == "autoridad" and "fracción VI" in fa["art_93"]["rige"]
   and fa["quejosa"]["nombre"] == QUEJOSA,
   "recurre la autoridad contra una concesión: fracción VI, y la quejosa sigue siendo quien pidió el amparo")

# ═══════════════════════════════════════════════════════════════════════════
print("\n3 · UN AMPARO DIRECTO")
e_d = ra.Encargo(numero="93/2026", encabezado="AMPARO DIRECTO ADMINISTRATIVO: 93/2026",
                 quejoso="Comercial Ejemplo, S.A. de C.V.", magistrado="M", secretario="S",
                 notificacion=_dt.date(2026, 1, 5), presentacion=_dt.date(2026, 1, 20),
                 tipo_asunto="amparo_directo",
                 responsable="Sala Regional de Ejemplo del Tribunal Federal de Justicia Administrativa")
f_d = f123.Fases123(fuentes=["SENTENCIA DE LA SALA", "DEMANDA"], expediente_origen="1234/25-01-01-1",
                    fecha_origen="diez de noviembre de dos mil veinticinco")
p_d = fpa.Partes(quejoso="Comercial Ejemplo, S.A. de C.V.", tercero_interesado="Administración Local de Ejemplo",
                 tipo_asunto="amparo_directo")
fd = fp.de_resultado(types.SimpleNamespace(encargo=e_d, fases=f_d, partes=p_d))
ok(fd["tipo_asunto"] == "amparo_directo" and not fd["es_recurso"] and fd["recurrente"] == {}
   and fd["organo_recurrido"] == {} and fd["art_93"] is None, "sin recurrente, sin órgano recurrido, sin art. 93")
ok(fd["responsables"] and fd["responsables"][0]["autoridad"].startswith("Sala Regional de Ejemplo")
   and "1234/25-01-01-1" in fd["responsables"][0]["acto"]
   and "diez de noviembre" in fd["responsables"][0]["acto"],
   "la Sala responsable y la resolución reclamada con su fecha y expediente")
ok(fd["terceros"] == [{"nombre": "Administración Local de Ejemplo", "fuente": "ficha de partes"}],
   "el tercero interesado, de la ficha de partes")
ok(fd["desenlace"]["si_prospera"]["resultado"] == "concede"
   and fd["desenlace"]["si_no_prospera"]["resultado"] == "niega", "desenlace por código: concede / niega")
bd = fp.bloque(fd)
ok("Tipo de asunto: amparo directo" in bd and "Recurrente" not in bd and "Artículo 93" not in bd
   and "Amparo adhesivo: no consta" in bd and "Si algún concepto prospera: concede" in bd,
   "el bloque del amparo directo no habla de recursos")
ok("responsable: Sala Regional" in fp.linea(fd) and "tercero: Administración" in fp.linea(fd),
   "la línea del amparo directo: responsable y tercero")
ok(fp.para_tarjeta(fd)["recurrida"] is None and fp.para_tarjeta(fd)["recurrente"] is None,
   "para la tarjeta: en el amparo directo no hay recurrida ni recurrente")
_fd_ad = fp.de_resultado(types.SimpleNamespace(
    encargo=e_d, fases=f123.Fases123(fuentes=["S", "La tercera promovió amparo adhesivo."]), partes=p_d))
ok(_fd_ad["adhesivo"] == {"consta": True, "donde": "escrito"}, "el amparo adhesivo, si consta, con su lugar")
ok(fp.bloque({}) == "" and fp.linea({}) == "" and fp.para_tarjeta({}) is None
   and fp.de_resultado(types.SimpleNamespace(encargo=None)) == {},
   "sin encargo, ficha vacía: nada que pintar ni que decir")

# ═══════════════════════════════════════════════════════════════════════════
print("\n4 · LA FICHA ENTRA COMO DATOS EN CADA PIEZA")
BLQ = fp.bloque(fi)

# — el contraste y la propuesta —
mat = f6.Material()
mat.tipo_asunto = "amparo_revision"
pc0 = f5.prompt_contraste(PROBLEMAS, "acto", "agravios", True)
pc1 = f5.prompt_contraste(PROBLEMAS, "acto", "agravios", True, ficha=BLQ)
ok(pc0 == f5.prompt_contraste(PROBLEMAS, "acto", "agravios", True, ficha="")
   and BLQ.strip() in pc1 and pc1.replace(BLQ.strip() + "\n\n", "") == pc0,
   "contraste: con ficha, el bloque y nada más; sin ficha, idéntico")
pp0 = f5.prompt_propuesta(PROBLEMAS, mat, "acto", "agravios", True)
pp1 = f5.prompt_propuesta(PROBLEMAS, mat, "acto", "agravios", True, ficha=BLQ)
ok(BLQ.strip() in pp1 and pp1.replace(BLQ.strip() + "\n\n", "") == pp0
   and pp1.index("LA FICHA PROCESAL") < pp1.index("LOS PROBLEMAS JURÍDICOS"),
   "propuesta: la ficha antes de los problemas; sin ficha, idéntica")


class _R:
    def __init__(self, txt):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=txt),
                                              finish_reason="stop")]
        self.usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                                           completion_tokens_details=None)


class Falso:
    def __init__(self, contestar):
        self.contestar, self.llamadas = contestar, []
        self.chat = self
        self.completions = self

    async def create(self, **kw):
        p = kw["messages"][0]["content"]
        self.llamadas.append(p)
        return _R(self.contestar(p))


def _contesta(p):
    if "EL CONTRASTE" in p.splitlines()[0] or "haces\nEL CONTRASTE" in p[:300]:
        return json.dumps({"contraste": [{"numero": 1, "razon_toral": "r", "la_combate": True,
                                          "sobrevive": False, "veredicto_previo": "a_examinar",
                                          "por_que": "x"}]})
    return json.dumps({"propuestas": [], "global": {}})


cli = Falso(_contesta)
asyncio.run(f5.proponer(cli, PROBLEMAS, mat, "acto", "agravios", True, ficha=BLQ))
ok(len(cli.llamadas) == 2 and all(BLQ.strip() in p for p in cli.llamadas),
   f"proponer: 2 llamadas falsas (contraste + propuesta) y las dos ven la ficha ({len(cli.llamadas)})")

# — la deliberación —
pd0 = dl.prompt_pregunta_decisiva(PROBLEMAS[0], None, "a", "c", "texto", "amparo_revision", True)
pd1 = dl.prompt_pregunta_decisiva(PROBLEMAS[0], None, "a", "c", "texto", "amparo_revision", True, ficha=BLQ)
ok(BLQ.strip() in pd1 and pd1.replace(BLQ.strip() + "\n\n", "") == pd0,
   "pregunta decisiva: con la ficha; sin ella, idéntica")
_kw_ab = dict(pral=PROBLEMAS[0], pi=0, problemas=PROBLEMAS, decisiva={"pregunta_decisiva": "¿?"},
              contraste=None, resumen_acto="a", resumen_conceptos="c", textos={"acto": "t"},
              cat=dl.construir_catalogo([], [])[0], tipo_asunto="amparo_revision", es_recurso=True)
try:
    pa0 = dl.prompt_abogado("A", **_kw_ab)
    pa1 = dl.prompt_abogado("A", ficha=BLQ, **_kw_ab)
    ok(BLQ.strip() in pa1 and pa1.replace(BLQ.strip() + "\n\n", "") == pa0,
       "abogado: con la ficha; sin ella, idéntico")
except Exception as ex:
    ok(False, f"abogado: el prompt se arma ({type(ex).__name__}: {ex})")
try:
    _v = {"A": {"sentido": "fundado", "respondio": True}, "B": {"sentido": "infundado", "respondio": True}}
    pj0 = dl.prompt_juez(("A", "B"), _v, decisiva={}, cat=dl.construir_catalogo([], [])[0],
                         constancias_faltantes=[])
    pj1 = dl.prompt_juez(("A", "B"), _v, decisiva={}, cat=dl.construir_catalogo([], [])[0],
                         constancias_faltantes=[], ficha=BLQ)
    ok(BLQ.strip() in pj1 and pj1.replace(BLQ.strip() + "\n\n", "") == pj0,
       "juez: con la ficha; sin ella, idéntico")
except Exception as ex:
    ok(False, f"juez: el prompt se arma ({type(ex).__name__}: {ex})")

cli_d = Falso(lambda p: "{}")
_doc = asyncio.run(dl.deliberar(cli_d, problemas=PROBLEMAS, resumen_acto="a", resumen_conceptos="c",
                                textos={"acto": RECURRIDA}, tipo_asunto="amparo_revision",
                                es_recurso=True, ficha=BLQ))
_tareas = [p.splitlines()[0] for p in cli_d.llamadas]
ok(cli_d.llamadas and all(BLQ.strip() in p for p in cli_d.llamadas if p.startswith("TAREA: LA PREGUNTA")
                          or p.startswith("TAREA: ABOGADO")),
   f"deliberar: la pregunta decisiva y los abogados ven la ficha ({len(cli_d.llamadas)} llamadas falsas)")

# — el estudio —
_crit = [f6.Criterio(problema=PROBLEMAS[0]["pregunta"], sentido="fundado", razonamiento="r",
                     jerarquia="principal")]
for var in ("v1", "v2", "v4"):
    m0 = f6.Material()
    m0.tipo_asunto, m0.variante = "amparo_revision", var
    e0 = f6.prompt_estudio("acto", "agravios", _crit, m0, es_recurso=True)
    m1 = f6.Material()
    m1.tipo_asunto, m1.variante, m1.ficha_procesal = "amparo_revision", var, BLQ
    e1 = f6.prompt_estudio("acto", "agravios", _crit, m1, es_recurso=True)
    ok(BLQ.strip() in e1 and e1.replace("\n" + BLQ.strip() + "\n", "", 1) == e0,
       f"estudio {var}: la ficha en el encabezado de datos; sin ella, idéntico")

# `_formato_al_material` la pone en cada petición.
_m = f6.Material()
_m.ficha_procesal = "LA DE LA VUELTA ANTERIOR"
ra._formato_al_material(resultado(), _m, None, _crit)
ok(_m.ficha_procesal == BLQ, "el material recibe la ficha de ESTA sesión en cada petición")

# — la síntesis —
ps0 = fs.prompt(tipo_asunto="amparo_revision", expediente="631/2025", quejoso=QUEJOSA, sentido="fundado",
                estudio="E", recurrente=RECURRENTE, papel_recurrente="tercero", organo="Juzgado X")
ps1 = fs.prompt(tipo_asunto="amparo_revision", expediente="631/2025", quejoso=QUEJOSA, sentido="fundado",
                estudio="E", recurrente=RECURRENTE, papel_recurrente="tercero", organo="Juzgado X", ficha=BLQ)
ok("Parte recurrente:" in ps0 and "Parte recurrente:" not in ps1 and BLQ.strip() in ps1,
   "síntesis: la ficha sustituye a los renglones sueltos de recurrente y órgano")
ok(ps0 == fs.prompt(tipo_asunto="amparo_revision", expediente="631/2025", quejoso=QUEJOSA,
                    sentido="fundado", estudio="E", recurrente=RECURRENTE,
                    papel_recurrente="tercero", organo="Juzgado X", ficha=""),
   "síntesis sin ficha: idéntica")

# — el compositor —
_d = ra._datos_estructura(encargo_631(), "", acto=RECURRIDA, partes=None, fases=fases_631())
ok(_d["quejoso"] == QUEJOSA and _d["recurrente"] == RECURRENTE and _d["papel_recurrente"] == "tercero"
   and _d["organo_recurrido"].startswith("Juzgado Séptimo de Distrito") and _d["tercero"] == RECURRENTE,
   "compositor: quejosa, recurrente, carácter, órgano y tercero salen de la ficha")
ok(_d["ficha_procesal"].get("tipo_asunto") == "amparo_revision" and _d["ficha_bloque"] == BLQ,
   "compositor: la ficha y su bloque viajan en los datos")
_pe = dg.prompt_estructura(_d)
ok(BLQ.strip() in _pe, "estructura (carátula, competencia, legitimación): el bloque de la ficha como datos")
_d0 = dict(_d, ficha_bloque="")
ok(BLQ.strip() not in dg.prompt_estructura(_d0), "estructura sin ficha: sin bloque")

# — la tarjeta —
ok(td.vacia("sin_propuesta").get("ficha", "falta") is None, "tarjeta vacía: ficha None")
_resp = {"global": {"sentido": "fundado", "razon": "r", "alternativa": {"sentido": "infundado", "razon": "a"}},
         "propuestas": [{"problema": PROBLEMAS[0]["pregunta"], "sentido": "fundado", "razon": "r"}],
         "contraste": []}
try:
    _tj = td.armar(_resp, None, PROBLEMAS, [], None, None,
                   {"tipo_asunto": "amparo_revision", "que_hizo": "concede", "quien_recurre": "tercero",
                    "ficha": pt}, None, fases=fases_631())
    ok(_tj.get("ficha") == json.loads(json.dumps(pt, ensure_ascii=False)),
       "tarjeta: la ficha llega con la forma del contrato")
    _tj2 = td.armar(_resp, None, PROBLEMAS, [], None, None, {"tipo_asunto": "amparo_revision"}, None)
    ok(_tj2.get("ficha", "falta") is None, "tarjeta sin ficha: None")
except Exception as ex:
    ok(False, f"tarjeta: se arma con la ficha ({type(ex).__name__}: {ex})")

# ═══════════════════════════════════════════════════════════════════════════
print("\n5 · LAS PIEZAS LEEN LA FICHA, NO VUELVEN A DEDUCIR")
_src = open(os.path.join(AQUI, "main.py"), encoding="utf-8").read()


def _cuerpo(nombre):
    m = re.search(r"\n(?:async )?def " + nombre + r"\(.*?(?=\n(?:async )?def |\n@app)", _src, re.S)
    return m.group(0) if m else ""


ok("_taller_ficha(r)" in _cuerpo("_taller_recurrente") and "_taller_ficha(r)" in _cuerpo("_taller_papel_recurrente"),
   "main: quién recurre y su carácter se leen de la ficha")
ok("ficha=_taller_ficha_bloque(r)" in _cuerpo("_taller_precontrastar"), "main: el contraste adelantado recibe la ficha")
_i_prop = _src.find("await _f5.proponer(")
ok(_i_prop > 0 and "ficha=_taller_ficha_bloque(r))" in _src[_i_prop:_i_prop + 900],
   "main: la propuesta recibe la ficha")
ok("ficha=_taller_ficha_bloque(" in _cuerpo("_taller_deliberar_nucleo"), "main: la deliberación recibe la ficha")
ok('rama_info["ficha"] = _fp_tj.para_tarjeta(' in _src, "main: la tarjeta recibe la ficha")
_ra_src = open(os.path.join(AQUI, "redactor_adelanto.py"), encoding="utf-8").read()
ok('ficha=str(datos.get("ficha_bloque") or "")' in _ra_src, "la síntesis recibe el bloque de la ficha")

print("\n6 · REVISIÓN ADVERSARIAL DE LA FASE E: LO QUE LA FICHA DABA POR FIJADO SIN SERLO")
# (a) responsables distintas que se fundían.
for _a, _b, _esp in (
        ("Director de Ingresos adscrito a la Secretaría de Finanzas del Estado de Ejemplo",
         "Secretaría de Finanzas del Estado de Ejemplo", False),
        ("Juez Quinto de lo Civil de Ejemplo", "Actuario adscrito al Juzgado Quinto de lo Civil de Ejemplo", False),
        ("Juzgado Quinto de lo Civil", "Secretario de Acuerdos del Juzgado Quinto de lo Civil", False),
        ("Director de Desarrollo Urbano del Municipio de Ejemplo", "Municipio de Ejemplo", False),
        ("Primera Sala Civil del Tribunal Superior de Justicia", "Tribunal Superior de Justicia", False),
        ("MAGISTRADA NOMBRE DE PRUEBA, INTEGRANTE DE LA PRIMERA SALA CIVIL DEL TRIBUNAL SUPERIOR DE "
         "JUSTICIA DEL ESTADO DE EJEMPLO",
         "la Primera Sala Civil del Tribunal Superior de Justicia en el Estado de Ejemplo", True),
        ("Magistrada de la Sala Civil Uno", "Sala Civil Uno del Tribunal Superior de Justicia", True)):
    ok(fp._misma_autoridad(_a, _b) is _esp,
       f"¿misma autoridad? «{_a[:45]}» / «{_b[:45]}» → {_esp}")
# (b) ordenadora y ejecutora: dos responsables, cada una con su acto.
_RES_OE = ("Por lo expuesto, se R E S U E L V E: PRIMERO. Se sobresee en el juicio respecto de la "
           "ejecución atribuida al Director de Ingresos adscrito a la Secretaría de Finanzas del Estado "
           "de Ejemplo. SEGUNDO. La Justicia de la Unión ampara y protege a Comercial Ejemplo, S.A. de "
           "C.V., en contra de la resolución atribuida a la Secretaría de Finanzas del Estado de Ejemplo, "
           "para los efectos precisados en el último considerando. Notifíquese.")
_f_oe = fases_631(fuentes=["JUZGADO PRIMERO DE DISTRITO EN EL ESTADO DE EJEMPLO\n" + _RES_OE, "AGRAVIOS."],
                  resolutivo_recurrida=_RES_OE)
_e_oe = encargo_631(quejoso="Comercial Ejemplo, S.A. de C.V.", responsable="Secretaría de Finanzas del Estado de Ejemplo")
_e_oe.recurrente = "Secretaría de Finanzas del Estado de Ejemplo"
_fi_oe = fp.de_resultado(resultado(_e_oe, _f_oe))
ok(len(_fi_oe["responsables"]) == 2
   and any("Director de Ingresos" in r["autoridad"] for r in _fi_oe["responsables"]),
   f"ordenadora y ejecutora: dos responsables ({[r['autoridad'][:30] for r in _fi_oe['responsables']]})")
# (c) el acto de la concesión, de sus efectos, cuando el punto sólo nombra a la autoridad.
_RES_EF = ("Por lo expuesto, se R E S U E L V E: ÚNICO. La Justicia de la Unión ampara y protege a Unión "
           "Ejemplo, A.C., en contra del acto atribuido a la Sala Civil Uno del Tribunal Superior de "
           "Justicia, para los efectos precisados en el último considerando. Notifíquese.")
_REC_EF = ("JUZGADO SÉPTIMO DE DISTRITO EN EL ESTADO DE EJEMPLO\nCONSIDERANDO OCTAVO. Efectos. Se concede "
           "para que la Sala responsable deje insubsistente la resolución de dos de julio de dos mil "
           "veinticuatro, dictada en el toca civil 000/2024, y emita otra. " + _RES_EF)
_fi_ef = fp.de_resultado(resultado(f=fases_631(fuentes=[_REC_EF, "AGRAVIOS."], resolutivo_recurrida=_RES_EF,
                                               resumen_acto="")))
_r_ef = (_fi_ef["responsables"] or [{}])[0]
ok(_r_ef.get("acto", "").startswith("la resolución de dos de julio") and "efectos" in _r_ef.get("fuente", ""),
   f"el acto de la Sala sale de los efectos de la concesión ({_r_ef.get('acto', '')[:50]})")
_RES_NO = _RES_EF
_fi_no = fp.de_resultado(resultado(f=fases_631(
    fuentes=["JUZGADO SÉPTIMO DE DISTRITO EN EL ESTADO DE EJEMPLO\n" + _RES_NO, "AGRAVIOS."],
    resolutivo_recurrida=_RES_NO, resumen_acto="")))
ok(any("No consta cuál es el acto reclamado a Sala Civil Uno" in a for a in _fi_no["avisos"])
   and not any("atribuido" in (r.get("acto") or "") for r in _fi_no["responsables"]),
   "sin efectos ni resumen, no se da «el acto atribuido a…» como acto: se avisa que no consta")
# (d) el sobreseimiento de una tesis transcrita no es el del juzgado.
_REC_TS = ("JUZGADO SÉPTIMO DE DISTRITO\nRESULTANDO PRIMERO. En otro juicio se sobreseyó respecto de la "
           "orden de visita, dictada por la Dirección Ejemplo. CONSIDERANDO CUARTO. Sirve de apoyo la "
           "tesis: «SE SOBRESEE RESPECTO DEL ACTO DE EJECUCIÓN CUANDO…». CONSIDERANDO QUINTO. Procede "
           "sobreseer respecto de la resolución interlocutoria de tres de mayo, dictada por el Juzgado "
           "Quinto Civil. R E S U E L V E: ÚNICO. Se concede. Notifíquese.")
ok(fp._sobreseido(_REC_TS)[0].startswith("la resolución interlocutoria de tres de mayo"),
   f"lo sobreseído sale de los considerandos, no de los antecedentes ni de una tesis ({fp._sobreseido(_REC_TS)[0][:40]})")
# (d2) el órgano de la recurrida es de Distrito: la responsable leída en la
# ficha de partes no se cuela como órgano recurrido.
for _leida in ("Juzgado Quinto de Primera Instancia Civil", "Tribunal Superior de Justicia del Estado de Ejemplo"):
    _o = ra._organo_recurrido(encargo_631(), types.SimpleNamespace(autoridad_responsable=_leida), RECURRIDA)
    ok(_o.startswith("Juzgado Séptimo de Distrito"), f"órgano recurrido con «{_leida[:30]}» leída: {_o[:40]}")
# (e) la quejosa que recurre: fracción V, sin II ni III.
_fi_q = fp.armar(types.SimpleNamespace(tipo_asunto="amparo_revision", es_recurso=True, quejoso="Unión Ejemplo, A.C.",
                                       recurrente="", responsable="Sala Civil Uno"),
                 fases_631(resolutivo_recurrida="Por lo expuesto, se R E S U E L V E: ÚNICO. La Justicia de la "
                           "Unión no ampara ni protege a Unión Ejemplo, A.C. Notifíquese.",
                           resolvio_a_quo="niega",
                           fuentes=["R E S U E L V E: ÚNICO. La Justicia de la Unión no ampara ni protege a "
                                    "Unión Ejemplo, A.C. Notifíquese.", "AGRAVIOS."]))
ok((_fi_q.get("art_93") or {}).get("si_no_prospera", {}).get("fundamento") ==
   "artículo 93, fracción V, de la Ley de Amparo" and not (_fi_q.get("art_93") or {}).get("previo"),
   "recurre la quejosa: la confirmación se funda en la fr. V y no hay II ni III")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
