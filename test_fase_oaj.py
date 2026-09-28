# -*- coding: utf-8 -*-
"""La fuente OAJ del espejo habla sólo con tabla y al 85%, y si no, calla.

Se corre sola y SIN RED: un Qdrant falso que respeta los filtros y el umbral,
y un embebedor falso que devuelve el propio texto para poder ver qué se
consultó.

    .venv/bin/python test_fase_oaj.py

LO QUE VIGILA, Y POR QUÉ CADA COSA
==================================
 · SIN TABLA NO HAY PORCENTAJE. Si el JSON de calibración no existe, no trae
   el tipo, está mal formado o trae `umbral_85: null`, la función devuelve []
   —y ni siquiera paga el embebido—. Inventar el porcentaje es de la especie
   de la tesis inventada.
 · LA TABLA REAL SE LEE. Viene con cuatro columnas (la cuarta, los pares que
   sostienen el tramo); la primera versión exigía tres y callaba siempre sin
   decirlo. Un tramo de un solo par no pone porcentaje, y nunca se enseña
   «100%».
 · SÓLO DONDE SE MIDIÓ. La tabla es del 3TCC; en otro tribunal, calla.
 · LA CONSULTA ES EL PLANTEAMIENTO ENTERO, como la arma la producción: por
   `consultar()`, que entrega cadenas, y no sólo por `_espejo_propio` con
   dicts a mano, que es como la primera prueba pasó en falso.
 · BAJO EL 85%, NADA. Y el porcentaje se redondea hacia abajo.
 · UN NEUN, UNA FILA. Tres planteamientos de la misma sentencia son un
   precedente, no tres. Y un NEUN guardado como 12345.0 sigue siendo el 12345.
 · EL RESPALDO POR TEMA sólo cuando ningún planteamiento pasa, y la fila dice
   que vino por tema.
 · EL FILTRO: otro tribunal u otro tipo no entran aunque se parezcan más.
 · EL MAPA DEL TRIBUNAL al nombre exacto con que la OAJ lo indexa, y sólo si
   CONSTA el circuito 22: un tribunal de otra plaza no es «su propio tribunal».
 · EL FORMATO de la fila, campo por campo, y que viaje entero por la fila de
   la sesión y por la respuesta de /taller.
 · EN EL TALLER: la OAJ primero, sin piso de filas, sin resumen y sin repetir
   entre problemas; y si calla, el espejo viejo exactamente como antes.
"""
import asyncio
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import types

sys.path.insert(0, ".")

import fase_oaj as fo

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


# ═══════════════════════════════════════════════════════════════════════════
# LOS FALSOS
# ═══════════════════════════════════════════════════════════════════════════
class _Punto:
    def __init__(self, score, payload):
        self.score = score
        self.payload = payload


class _Resp:
    def __init__(self, puntos):
        self.points = puntos


class QdrantFalso:
    """Respeta el filtro `must` de igualdad y el `score_threshold`, como Qdrant.

    `puntos` es [(score, payload)] o una función texto_consultado → esa lista,
    para que cada planteamiento pueda recuperar cosas distintas.
    """

    def __init__(self, puntos, falla=False):
        self.puntos = puntos
        self.falla = falla
        self.llamadas = []

    async def query_points(self, collection_name, query, using, query_filter,
                           limit, score_threshold, with_payload):
        self.llamadas.append({"coleccion": collection_name, "using": using,
                              "limit": limit, "umbral": score_threshold,
                              "filtro": {c.key: c.match.value
                                         for c in query_filter.must},
                              "consulta": query})
        if self.falla:
            raise RuntimeError("Qdrant caído (simulado)")
        cands = self.puntos(query) if callable(self.puntos) else self.puntos
        filtro = self.llamadas[-1]["filtro"]
        fuera = [_Punto(s, pl) for s, pl in cands
                 if all(pl.get(k) == v for k, v in filtro.items())
                 and (score_threshold is None or s >= score_threshold)]
        fuera.sort(key=lambda p: -p.score)
        return _Resp(fuera[:limit])

    def oaj(self):
        return [ll for ll in self.llamadas if ll["coleccion"] == fo.COLECCION]


CONSULTAS = []


async def embed(t):
    # Devuelve el texto: el Qdrant falso no mira el vector, y así se puede
    # comprobar QUÉ se embebió.
    CONSULTAS.append(t)
    return t


async def embed_roto(t):
    raise RuntimeError("embebedor caído (simulado)")


ORG3 = fo.ORGANOS_OAJ["3TCC"]
ORG1 = fo.ORGANOS_OAJ["1TCC"]


def pl(neun, score_tag="", clase="planteamiento", tipo="Amparo Directo",
       organo=ORG3, **extra):
    alias = f"{neun % 1000}/2025" if isinstance(neun, int) else "1/2025"
    d = {"clase": clase, "neun": neun, "alias": alias,
         "tipo": tipo, "organo": organo, "circuito": "VIGÉSIMO SEGUNDO CIRCUITO",
         "fecha": "14-03-2025", "tema": "Firma electrónica de la notificación",
         "sintesis": "…", "sentido": "concede"}
    if clase == "planteamiento":
        d.update({"pregunta": f"¿Pregunta del precedente {neun}{score_tag}?",
                  "combate": "c", "resolvio": "r", "calificacion": "fundado",
                  "razon": "la constancia carece de firma", "autoridad": "Sala",
                  "acto": "sentencia"})
    d.update(extra)
    return d


# La forma que escribe el calibrador: [cos_min, cos_max, prob, n].
CAL = {
    "planteamiento": {
        # 0.70 es donde la tabla alcanza 0.85 con pares suficientes: el corte.
        "Amparo Directo": [[0.0, 0.60, 0.10, 40], [0.60, 0.70, 0.50, 20],
                           [0.70, 0.75, 0.869, 12], [0.75, 1.0, 0.95, 9]],
    },
    "asunto": {
        "Amparo Directo": [[0.0, 0.65, 0.20, 50], [0.65, 1.0, 0.90, 10]],
    },
}

# ── TRAMOS DE LA TABLA REAL (calibracion 28-sep-2026, 3er TCC XXII) ─────────
# Copiados tal cual del final de cada tabla de `asunto`, con su `umbral_85`.
# Son los que la revisión reprodujo: con la versión anterior, «100% de
# similitud» en amparo directo y en queja, donde el calibrador dejó `null`.
REAL = {
    "asunto": {
        "Amparo Directo": [[0.545, 0.6505, 0.083, 60], [0.6511, 0.6781, 0.15, 20],
                           [0.6874, 0.7075, 0.286, 7], [0.7096, 0.7348, 0.667, 6],
                           [0.7466, 0.7466, 1.0, 1], [0.7488, 0.7488, 1.0, 1],
                           [0.7665, 0.7665, 1.0, 1]],
        "Queja": [[0.5056, 0.5665, 0.273, 11], [0.5671, 0.5833, 0.286, 7],
                  [0.5886, 0.7697, 0.306, 49], [0.7707, 0.7835, 0.5, 2],
                  [0.787, 0.787, 1.0, 1], [0.7949, 0.7949, 1.0, 1],
                  [0.8339, 0.8339, 1.0, 1]],
        "Revisión Fiscal": [[0.5842, 0.6459, 0.24, 25], [0.6459, 0.6546, 0.25, 4],
                            [0.6586, 0.6711, 0.333, 6], [0.6724, 0.7464, 0.35, 20],
                            [0.7612, 0.7866, 0.9, 10]],
    },
    "planteamiento": {},
    "umbral_85": {"asunto": {"Amparo Directo": None, "Amparo en revisión": None,
                             "Queja": None, "Revisión Fiscal": 0.7612},
                  "planteamiento": {}},
}

_TMP = tempfile.mkdtemp(prefix="oaj_cal_")
_N = [0]


def usar_calibracion(d):
    """Una ruta nueva por escenario: la caché va por (ruta, fecha)."""
    _N[0] += 1
    ruta = os.path.join(_TMP, f"cal_{_N[0]}.json")
    if d is not None:
        with open(ruta, "w", encoding="utf-8") as fh:
            fh.write(d if isinstance(d, str) else json.dumps(d))
    fo.RUTA_CALIBRACION = ruta


def correr(coro):
    return asyncio.run(coro)


def callado(fn, *a):
    """(resultado, lo que imprimió)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a)
    return r, buf.getvalue()


PROB = {"pregunta": "¿La constancia de notificación electrónica debía llevar "
                    "firma electrónica avanzada?",
        "combate": "La Sala omitió valorar la falta de firma.",
        "resolvio": "La Sala tuvo por legal la notificación.",
        "impedimento": None}


# ═══════════════════════════════════════════════════════════════════════════
print("\n1 · EL MAPA DEL TRIBUNAL AL NOMBRE DE LA OAJ, SÓLO SI CONSTA EL 22")
_R = ", con residencia en Querétaro, Querétaro"
CASOS = [
    ("Primer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito",
     "22", "", "1TCC", "Primer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito" + _R),
    ("Segundo Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito",
     "22", "", "2TCC", "Segundo Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito" + _R),
    # El «Segundo» del CIRCUITO no es el ordinal del tribunal.
    ("Tercer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito",
     "22", "", "3TCC", "Tercer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito" + _R),
    ("Tribunal Colegiado en Materias Administrativa y de Trabajo del Vigésimo Segundo Circuito",
     "22", "", "TCC_ADM", "Tribunal Colegiado en Materias Administrativa y de Trabajo del Vigésimo Segundo Circuito" + _R),
    ("Tribunal Colegiado en Materias Penal y Administrativa del Vigésimo Segundo Circuito",
     "22", "", "TCC_PENAL", "Tribunal Colegiado en Materias Penal y Administrativa del Vigésimo Segundo Circuito" + _R),
    # El nombre basta aunque el circuito no se haya leído: en romanos, o con
    # la cláusula del 22 escrita y el parámetro vacío.
    ("Tercer Tribunal Colegiado en Materias Administrativa y Civil del XXII Circuito",
     "", "", "3TCC", "Tercer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito" + _R),
    # Sin cláusula de circuito, la residencia en Querétaro lo acredita.
    ("Tercer Tribunal Colegiado en Materias Administrativa y Civil",
     "", "Querétaro, Querétaro", "3TCC", "Tercer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito" + _R),
    # Fuera del 22 no hay mapa: se calla en vez de adivinar.
    ("Segundo Tribunal Colegiado en Materia Civil del Primer Circuito", "1", "", None, ""),
    ("", "", "", None, ""),
    # Los que la revisión reprodujo: circuito vacío resuelto a Querétaro.
    ("Primer Tribunal Colegiado en Materia Penal del Decimoquinto Circuito", "", "", None, ""),
    ("Primer Tribunal Colegiado del Decimoquinto Circuito", "", "", None, ""),
    ("Tribunal Colegiado en Materia de Trabajo del 7o. Circuito", "", "", None, ""),
    ("Tercer Tribunal Colegiado en Materia Civil", "", "", None, ""),
    ("Tercer Tribunal Colegiado en Materia Administrativa", "", "", None, ""),
    ("Primer Tribunal Colegiado de Circuito del Centro Auxiliar de la Tercera "
     "Región, con residencia en Guadalajara, Jalisco", "", "", None, ""),
    # Un auxiliar NUNCA, aunque la ficha diga Querétaro.
    ("Primer Tribunal Colegiado Auxiliar", "", "Querétaro, Querétaro", None, ""),
    # Con una cláusula de otro circuito, la ciudad de la ficha no lo arregla.
    ("Primer Tribunal Colegiado del Decimoquinto Circuito", "", "Querétaro, Querétaro", None, ""),
]
for nombre, circ, ciudad, clave, organo in CASOS:
    got = fo.organo_de(nombre, circ, ciudad)
    ok(got == (clave, organo),
       f"«{(nombre or '(vacío)')[:44]}…»{' +' + ciudad[:9] if ciudad else ''} → {got[0]!r}"
       + (" con residencia" if got[1].endswith(_R) else ""))

print("\n2 · LOS TIPOS DEL TALLER A LA GRAFÍA DE LA OAJ")
for t, esperado in (("amparo_directo", "Amparo Directo"),
                    ("amparo_revision", "Amparo en revisión"),
                    ("queja", "Queja"), ("revision_fiscal", "Revisión Fiscal"),
                    ("Amparo en revisión", "Amparo en revisión"),
                    ("rf", "Revisión Fiscal"), ("reclamacion", ""), ("", "")):
    ok(fo.tipo_oaj(t) == esperado, f"{t!r} → {fo.tipo_oaj(t)!r}")

print("\n3 · LA CONSULTA ES «pregunta combate resolvio», O NO HAY CONSULTA")
ok(fo.texto_consulta(PROB) == " ".join([PROB["pregunta"], PROB["combate"],
                                         PROB["resolvio"]]),
   "del problema entero, en ese orden, sin el impedimento")
ok(fo.texto_consulta("¿Sólo la pregunta?") == "",
   "la pregunta sola NO consulta: la tabla se midió con las tres piezas")
ok(fo.texto_consulta({"pregunta": "¿P?", "combate": "c"}) == "",
   "un planteamiento sin «resolvio» no consulta")
ok(fo.texto_consulta({"pregunta": "¿P?", "resolvio": "r", "combate": "  "}) == "",
   "ni uno con el «combate» en blanco")
ok(fo.texto_consulta({"pregunta": "", "combate": ""}) == "",
   "un problema vacío no consulta")

print("\n4 · LA TABLA: ESCALONADA, CON PARES DETRÁS, SIN EXTRAPOLAR")
t = fo.tabla_de(CAL, "planteamiento", "Amparo Directo")
ok(t is not None and all(len(x) == 4 for x in t),
   "la tabla de cuatro columnas —la que escribe el calibrador— se lee")
ok(fo.probabilidad(t, 0.72) == 0.869 and fo.probabilidad(t, 0.99) == 0.95,
   "el coseno cae en su tramo")
ok(fo.probabilidad(fo.tabla_de({"a": {"X": [[0.7, 1.0, 0.9, 5], [0.5, 0.6, 0.5, 5]]}},
                                "a", "X"), 0.65) == 0.5,
   "un coseno en el HUECO entre tramos toma el de abajo (tabla desordenada a propósito)")
ok(fo.probabilidad(fo.tabla_de({"a": {"X": [[0.5, 0.6, 0.9, 5]]}}, "a", "X"), 0.4) is None,
   "por debajo del primer tramo no hay dato y no se extrapola")
_uno = fo.tabla_de({"a": {"X": [[0.5, 0.7, 0.3, 40], [0.75, 0.75, 1.0, 1],
                                [0.80, 0.80, 1.0, 2]]}}, "a", "X")
ok(fo.probabilidad(_uno, 0.76) == 0.3 and fo.probabilidad(_uno, 0.95) == 0.3,
   "un tramo de uno o dos pares NO pone porcentaje: se lee el sostenido de abajo")
ok(fo._corte({}, "a", "X", _uno) is None,
   "y tampoco es corte: el «1.0» de un solo par no es un 85% fiable")
_tres = fo.tabla_de({"a": {"X": [[0.5, 0.7, 0.3], [0.7, 1.0, 0.95]]}}, "a", "X")
ok(_tres is not None and fo.probabilidad(_tres, 0.9) is None
   and fo._corte({}, "a", "X", _tres) is None,
   "una tabla de tres columnas se lee, pero sin pares no se sabe qué la sostiene: calla")
for malo, que in (([[0.1, 0.2]], "tramo de dos números"),
                  ([[0.1, 0.2, 1.3, 5]], "probabilidad > 1"),
                  ([[0.3, 0.2, 0.9, 5]], "tramo al revés"),
                  ([[0.1, 0.2, "x", 5]], "texto en vez de número"),
                  ([[0.1, 0.2, 0.9, 2.5]], "pares que no son entero"),
                  ([[0.1, 0.2, 0.9, -1]], "pares negativos"),
                  ([[0.1, 0.2, 0.9, 5, 7]], "cinco columnas"),
                  ([], "tabla vacía")):
    ok(fo.tabla_de({"planteamiento": {"Queja": malo}}, "planteamiento", "Queja") is None,
       f"tabla mal formada ({que}) → se descarta entera")
fo._AVISADOS.clear()
_r, _log = callado(fo.tabla_de, {"asunto": {"Queja": [[0.1, 0.2]]}}, "asunto", "Queja")
ok(_r is None and "descartada" in _log and "asunto/Queja" in _log,
   "y el descarte por forma se dice en el log: el silencio no pasa por normal")
ok(fo._corte(CAL, "planteamiento", "Amparo Directo", t) == 0.70,
   "el corte es el inicio del primer tramo sostenido que llega a 0.85")
_endurecida = dict(CAL, umbral_85={"planteamiento": {"Amparo Directo": 0.76}})
ok(fo._corte(_endurecida, "planteamiento", "Amparo Directo", t) == 0.76,
   "`umbral_85` más alto ENDURECE el corte")
_blanda = dict(CAL, umbral_85={"planteamiento": {"Amparo Directo": 0.50}})
ok(fo._corte(_blanda, "planteamiento", "Amparo Directo", t) == 0.70,
   "`umbral_85` más bajo NO lo ablanda: manda la tabla")
_nula = dict(CAL, umbral_85={"planteamiento": {"Amparo Directo": None}})
ok(fo._corte(_nula, "planteamiento", "Amparo Directo", t) is None,
   "`umbral_85: null` es «no hay corte fiable»: se calla, no se deduce otro")
_otra = dict(CAL, umbral_85={"planteamiento": {"Queja": None}})
ok(fo._corte(_otra, "planteamiento", "Amparo Directo", t) == 0.70,
   "el `null` de OTRO tipo no apaga éste")

print("\n4b · CON LA TABLA REAL")
for tipo in ("Amparo Directo", "Queja"):
    tr = fo.tabla_de(REAL, "asunto", tipo)
    ok(tr is not None and fo._corte(REAL, "asunto", tipo, tr) is None,
       f"{tipo}: se lee, y con `umbral_85: null` no hay corte")
    ok(all((fo.probabilidad(tr, c) or 0) < 0.85 for c in (0.75, 0.80, 0.95)),
       f"{tipo}: ningún coseno alto sale con 85% por un tramo de un par")
trf = fo.tabla_de(REAL, "asunto", "Revisión Fiscal")
ok(fo._corte(REAL, "asunto", "Revisión Fiscal", trf) == 0.7612,
   "Revisión Fiscal: el corte es el del calibrador, 0.7612 (tramo de 10 pares)")
ok(fo.probabilidad(trf, 0.77) == 0.9 and fo.probabilidad(trf, 0.95) == 0.9,
   "y por encima, 90%: el último tramo sostenido, recortado como la isotónica")
usar_calibracion(REAL)
_rf = [(0.95, pl(801, clase="asunto", tipo="Revisión Fiscal")),
       (0.77, pl(802, clase="asunto", tipo="Revisión Fiscal")),
       (0.75, pl(803, clase="asunto", tipo="Revisión Fiscal"))]
PRF = dict(PROB)
filas = correr(fo.precedentes_oaj(QdrantFalso(_rf), embed, PRF, "revision_fiscal", ORG3))
ok([(f["neun"], f["similitud"], f["fuente"]) for f in filas]
   == [(801, 90, "tema"), (802, 90, "tema")],
   f"RF en el 3TCC habla por tema al 90%, y 0.75 queda fuera (salió "
   f"{[(f['neun'], f['similitud']) for f in filas]})")
CONSULTAS.clear()
q = QdrantFalso([(0.99, pl(804, clase="asunto"))])
ok(correr(fo.precedentes_oaj(q, embed, PROB, "amparo_directo", ORG3)) == []
   and not CONSULTAS and not q.llamadas,
   "AD con la tabla real calla, y ni embebe ni consulta")

print("\n5 · SIN TABLA NO HAY PORCENTAJE")
PUNTOS = [(0.80, pl(101)), (0.72, pl(101, "-bis")), (0.71, pl(102)),
          (0.65, pl(103)),
          (0.90, pl(201, organo=ORG1)),            # otro tribunal
          (0.90, pl(301, tipo="Queja")),           # otro tipo
          (0.70, pl(101, clase="asunto"))]

usar_calibracion(None)   # la ruta no existe
CONSULTAS.clear()
q = QdrantFalso(PUNTOS)
ok(correr(fo.precedentes_oaj(q, embed, PROB, "amparo_directo", ORG3)) == [],
   "JSON ausente → []")
ok(not CONSULTAS and not q.llamadas,
   "y sin tabla ni se embebe ni se consulta: no se paga lo que no se puede decir")

usar_calibracion({"planteamiento": {"Queja": CAL["planteamiento"]["Amparo Directo"]}})
ok(correr(fo.precedentes_oaj(QdrantFalso(PUNTOS), embed, PROB,
                             "amparo_directo", ORG3)) == [],
   "el tipo no está en la tabla → []")

usar_calibracion("{esto no es json")
ok(correr(fo.precedentes_oaj(QdrantFalso(PUNTOS), embed, PROB,
                             "amparo_directo", ORG3)) == [],
   "JSON ilegible → []")

usar_calibracion({"planteamiento": {"Amparo Directo": [[0.0, 1.0, 0.80, 50]]}})
ok(correr(fo.precedentes_oaj(QdrantFalso(PUNTOS), embed, PROB,
                             "amparo_directo", ORG3)) == [],
   "una tabla que nunca llega a 0.85 → []")

usar_calibracion(CAL)
for prob_malo, que in (("¿La constancia de notificación electrónica debía llevar "
                        "firma electrónica avanzada?", "la pregunta sola, como cadena"),
                       ({"pregunta": PROB["pregunta"], "combate": PROB["combate"]},
                        "un dict sin «resolvio»")):
    CONSULTAS.clear()
    q = QdrantFalso(PUNTOS)
    ok(correr(fo.precedentes_oaj(q, embed, prob_malo, "amparo_directo", ORG3)) == []
       and not CONSULTAS and not q.llamadas,
       f"{que} → [] sin embeber: la tabla no vale para esa forma de consulta")

print("\n6 · CON TABLA: FILTRO, UMBRAL, UN NEUN POR FILA Y FORMATO")
usar_calibracion(CAL)
CONSULTAS.clear()
q = QdrantFalso(PUNTOS)
filas = correr(fo.precedentes_oaj(q, embed, PROB, "amparo_directo", ORG3))
ok([f["neun"] for f in filas] == [101, 102],
   f"salen 101 y 102, en orden de coseno (salió {[f['neun'] for f in filas]})")
ok(CONSULTAS == [fo.texto_consulta(PROB)],
   "se embebió «pregunta combate resolvio», una sola vez")
ll = q.llamadas[0] if q.llamadas else {}
ok(ll.get("coleccion") == "oaj_precedentes" and ll.get("using") == "dense",
   "contra oaj_precedentes, vector «dense»")
ok(ll.get("filtro") == {"clase": "planteamiento", "tipo": "Amparo Directo",
                        "organo": ORG3},
   "filtro por clase, tipo y órgano: otro tribunal y otro tipo no entran")
ok(ll.get("umbral") == 0.70, "a Qdrant se le pide desde el corte de la tabla")
ok(len(q.llamadas) == 1, "si los planteamientos hablan, no se consulta el respaldo")
f0 = filas[0] if filas else {}
ok(f0.get("score") == 0.8 and "-bis" not in f0.get("pregunta", ""),
   "del NEUN 101 se queda el MEJOR planteamiento (0.80), no el de 0.72")
CAMPOS = {"tipo_asunto", "expediente", "fecha", "sentido", "tema", "score",
          "pdf_url", "similitud", "fuente", "pregunta", "razon",
          "calificacion", "autoridad", "neun", "enlace_oaj"}
ok(all(set(f) == CAMPOS for f in filas), "cada fila trae exactamente los campos pactados")
ok(f0.get("tipo_asunto") == "Amparo Directo" and f0.get("expediente") == "101/2025"
   and f0.get("fecha") == "14-03-2025" and f0.get("sentido") == "concede"
   and f0.get("pdf_url") == "" and f0.get("fuente") == "planteamiento"
   and f0.get("calificacion") == "fundado"
   and f0.get("razon") == "la constancia carece de firma"
   and f0.get("autoridad") == "Sala"
   and f0.get("enlace_oaj") == "https://ejusticia.cjf.gob.mx/BuscadorSISE/",
   "los valores salen del payload con su nombre de la tarjeta")
ok(isinstance(f0.get("similitud"), int) and isinstance(f0.get("neun"), int),
   "similitud y NEUN son enteros")
ok(f0.get("similitud") == 95 and (filas[1]["similitud"] if len(filas) > 1 else 0) == 86,
   "0.95 → 95% y 0.869 → 86%: el porcentaje se redondea hacia abajo")

# Seis como máximo.
muchos = [(0.80 - i * 0.001, pl(500 + i)) for i in range(10)]
filas = correr(fo.precedentes_oaj(QdrantFalso(muchos), embed, PROB,
                                  "amparo_directo", ORG3))
ok(len(filas) == 6, f"como máximo seis filas (salieron {len(filas)})")

# Nunca «100%».
usar_calibracion({"planteamiento": {"Amparo Directo": [[0.0, 0.7, 0.2, 30],
                                                        [0.7, 1.0, 1.0, 8]]}})
filas = correr(fo.precedentes_oaj(QdrantFalso([(0.9, pl(111))]), embed, PROB,
                                  "amparo_directo", ORG3))
ok([f["similitud"] for f in filas] == [99],
   "un tramo sostenido en 1.0 se enseña como 99%: no se afirma certeza")

# El NEUN: sin él no se enseña; como float entero, sí.
usar_calibracion(CAL)
fo._AVISADOS.clear()
filas, _log = callado(lambda: correr(fo.precedentes_oaj(
    QdrantFalso([(0.9, pl(0)), (0.9, pl("x")), (0.9, pl(True)), (0.9, pl(12.5))]),
    embed, PROB, "amparo_directo", ORG3)))
ok(filas == [], "una fila sin NEUN válido (0, «x», True, 12.5) no se enseña")
ok("sin NEUN válido" in _log,
   "y si NINGÚN punto trae NEUN válido, el log lo dice: no es «no hay precedentes»")
filas = correr(fo.precedentes_oaj(
    QdrantFalso([(0.9, pl(12345.0)), (0.85, pl("23456.0")), (0.8, pl(" 34567 "))]),
    embed, PROB, "amparo_directo", ORG3))
ok([f["neun"] for f in filas] == [12345, 23456, 34567]
   and all(isinstance(f["neun"], int) for f in filas),
   "12345.0, «23456.0» y « 34567 » son NEUN enteros: pandas no apaga la fuente")

# `umbral_85` más estricto que la tabla: 102 (0.71) queda fuera.
usar_calibracion(_endurecida)
filas = correr(fo.precedentes_oaj(QdrantFalso(PUNTOS), embed, PROB,
                                  "amparo_directo", ORG3))
ok([f["neun"] for f in filas] == [101], "con `umbral_85` en 0.76 sólo pasa 101")

print("\n6b · SÓLO EN EL TRIBUNAL DONDE SE MIDIÓ LA TABLA")
P1TCC = [(0.80, pl(701, organo=ORG1))]
usar_calibracion(CAL)
CONSULTAS.clear()
q = QdrantFalso(P1TCC)
ok(correr(fo.precedentes_oaj(q, embed, PROB, "amparo_directo", ORG1)) == []
   and not CONSULTAS and not q.llamadas,
   "sin lista `organos` vale sólo el 3TCC: en el Primer Tribunal calla, sin embeber")
usar_calibracion(dict(CAL, organos=["1TCC"]))
ok([f["neun"] for f in correr(fo.precedentes_oaj(QdrantFalso(P1TCC), embed, PROB,
                                                 "amparo_directo", ORG1))] == [701],
   "con `organos: [\"1TCC\"]` habla en el Primer Tribunal")
usar_calibracion(dict(CAL, organos=[ORG1]))
ok(correr(fo.precedentes_oaj(QdrantFalso(PUNTOS), embed, PROB,
                             "amparo_directo", ORG3)) == [],
   "y una lista que no incluye al 3TCC lo apaga a él")

print("\n7 · BAJO EL UMBRAL, NADA")
usar_calibracion(CAL)
bajos = [(0.69, pl(101)), (0.60, pl(102)), (0.64, pl(101, clase="asunto"))]
q = QdrantFalso(bajos)
ok(correr(fo.precedentes_oaj(q, embed, PROB, "amparo_directo", ORG3)) == [],
   "planteamientos bajo 0.70 y asunto bajo 0.65 → []")
ok(len(q.llamadas) == 2, "y sí se probó el respaldo antes de callar")

print("\n8 · EL RESPALDO POR TEMA")
respaldo = [(0.65, pl(101)), (0.70, pl(401, clase="asunto")),
            (0.66, pl(402, clase="asunto")), (0.60, pl(403, clase="asunto"))]
q = QdrantFalso(respaldo)
filas = correr(fo.precedentes_oaj(q, embed, PROB, "amparo_directo", ORG3))
ok([f["neun"] for f in filas] == [401, 402],
   f"ningún planteamiento pasa → salen los asuntos por tema (salió {[f['neun'] for f in filas]})")
ok(all(f["fuente"] == "tema" and f["pregunta"] == "" and f["razon"] == ""
       and f["calificacion"] == "" for f in filas),
   "la fila dice que vino por tema y no finge pregunta, razón ni calificación")
ok(filas and filas[0]["similitud"] == 90, "con la tabla de `asunto`, no con la de planteamiento")
ok(len(q.llamadas) == 2 and q.llamadas[1]["filtro"]["clase"] == "asunto"
   and q.llamadas[1]["umbral"] == 0.65,
   "el respaldo filtra clase=asunto desde su propio corte")

usar_calibracion({"asunto": CAL["asunto"]})
filas = correr(fo.precedentes_oaj(QdrantFalso(respaldo), embed, PROB,
                                  "amparo_directo", ORG3))
ok([f["neun"] for f in filas] == [401, 402],
   "sin tabla de planteamiento pero con la de asunto, habla el respaldo")

print("\n9 · LOS ERRORES NO TUMBAN NADA")
usar_calibracion(CAL)
ok(correr(fo.precedentes_oaj(QdrantFalso(PUNTOS, falla=True), embed, PROB,
                             "amparo_directo", ORG3)) == [], "Qdrant caído → []")
ok(correr(fo.precedentes_oaj(QdrantFalso(PUNTOS), embed_roto, PROB,
                             "amparo_directo", ORG3)) == [], "embebedor caído → []")
ok(correr(fo.precedentes_oaj(QdrantFalso(PUNTOS), embed, PROB,
                             "reclamacion", ORG3)) == [], "tipo que no es de los cuatro → []")
ok(correr(fo.precedentes_oaj(QdrantFalso(PUNTOS), embed, PROB,
                             "amparo_directo", "")) == [], "sin órgano → []")
ok(correr(fo.precedentes_oaj(None, embed, PROB, "amparo_directo", ORG3)) == [],
   "sin cliente → []")

print("\n10 · LA COBERTURA NO PROMETE «TODAS»")
ok("todas" not in fo.NOTA_COBERTURA.lower()
   and "denominación anterior" in fo.NOTA_COBERTURA
   and "ya leídas" in fo.NOTA_COBERTURA
   and "no quiere decir" in fo.NOTA_COBERTURA,
   "dice que va con el nombre actual, que los planteamientos son de las "
   "sentencias leídas y que la ausencia no prueba nada")

print("\n11 · EL VOCABULARIO PROHIBIDO")
PROHIBIDAS = re.compile(
    r"\bracha[s]?\b|\bseguidas\b|\bjurisprudencia\b|\breiteraci[oó]n\b|"
    r"\binterrump", re.I)
fuente = open("fase_oaj.py", encoding="utf-8").read()
cuerpo = fuente.split('"""', 2)[-1]
cuerpo = re.sub(r'"""[\s\S]*?"""', "", cuerpo)
cuerpo = re.sub(r"^\s*#.*$", "", cuerpo, flags=re.M)
malas = sorted(set(m.group(0).lower() for m in PROHIBIDAS.finditer(cuerpo)))
ok(not malas, f"el código del módulo no usa palabras prohibidas (salió: {malas})")

# ═══════════════════════════════════════════════════════════════════════════
print("\n12 · EN EL TALLER: LA OAJ PRIMERO, Y SI CALLA, EL ESPEJO VIEJO")
import fase_espejo as fe
import redactor_adelanto as ra

P1 = dict(PROB, impedimento={"motivo": "inoperancia",
                             "explicacion": "el concepto reitera el agravio"})
P2 = {"pregunta": "¿La revisión de gabinete concluyó en doce meses?",
      "combate": "El plazo del 46-A se excedió.", "resolvio": "Concluyó a tiempo."}
P3 = {"pregunta": "¿Procedía la condena en costas?", "combate": "x", "resolvio": "y"}
# Un planteamiento viejo, sin «resolvio»: no consulta la OAJ.
P4 = {"pregunta": "¿La constancia de notificación era válida?", "combate": "z"}
GLOBAL = "¿La notificación electrónica de la resolución determinante fue legal?"


def encargo(tribunal="Tercer Tribunal Colegiado en Materias Administrativa y "
                     "Civil del Vigésimo Segundo Circuito", ciudad=""):
    return types.SimpleNamespace(
        tribunal=tribunal, ciudad=ciudad, tipo_asunto="amparo_directo",
        coleccion_estatal="", es_recurso=False, materia="administrativa",
        encabezado="AMPARO DIRECTO ADMINISTRATIVO: 1/2026", responsable="")


def resultado(problemas, **kw):
    return types.SimpleNamespace(
        encargo=encargo(**kw), avisos=[],
        fases=types.SimpleNamespace(problema_global=GLOBAL, problemas=problemas,
                                    antecedentes="", resumen_acto=""))


def consultas_como_produccion(problemas):
    """La lista que `consultar()` arma y le pasa a `_espejo_propio`: CADENAS."""
    fuera = [GLOBAL]
    for p in problemas:
        fuera.append(p.get("pregunta", ""))
        x = p.get("impedimento")
        if isinstance(x, dict) and x.get("explicacion"):
            fuera.append(f"¿{str(x.get('motivo')).capitalize()}: {x['explicacion']}?")
    return fuera


def por_problema(consulta):
    # P1 recupera 101 y 102; P2 recupera 101 (repetido) y 601; P3 nada. Y
    # cualquier consulta que NO sea un planteamiento entero —la global, una
    # sintética, una pregunta sola— recuperaría 101 y 901 al 95%: si llegara a
    # consultarse, se vería en la tarjeta.
    if consulta.startswith(PROB["pregunta"] + " "):
        return [(0.80, pl(101)), (0.71, pl(102))]
    if consulta.startswith(P2["pregunta"] + " "):
        return [(0.78, pl(101)), (0.74, pl(601))]
    if consulta.startswith(P3["pregunta"] + " "):
        return [(0.50, pl(999))]
    return [(0.95, pl(101)), (0.95, pl(901))]


r_ = resultado([P1, P2, P3, P4])
usar_calibracion(CAL)
CONSULTAS.clear()
esp = correr(ra._espejo_propio(QdrantFalso(por_problema), embed, r_,
                               consultas_como_produccion([P1, P2, P3, P4])))
ok(sorted(CONSULTAS) == sorted(fo.texto_consulta(p) for p in (P1, P2, P3)),
   "con las CADENAS que manda consultar(), la OAJ embebe los tres planteamientos "
   "enteros y nada más: ni el global, ni la sintética, ni el que no trae «resolvio»")
ok(len(esp) == 2, f"dos grupos: P3 no tiene nada al 85% (salieron {len(esp)})")
g1 = esp[0] if esp else {}
g2 = esp[1] if len(esp) > 1 else {}
ok(set(g1) == {"problema", "tribunal", "filas", "resumen", "cobertura"},
   "el grupo tiene la forma de siempre")
ok(g1.get("problema") == P1["pregunta"] and g2.get("problema") == P2["pregunta"],
   "cada grupo se rotula con la pregunta de SU planteamiento")
ok(g1.get("cobertura") == fo.NOTA_COBERTURA, "con la nota de cobertura de la OAJ")
ok(g1.get("tribunal") == fe.TRIBUNALES_22["3TCC"],
   "el tribunal se lee con su nombre de siempre, sin la residencia del filtro")
ok([f["neun"] for f in g1.get("filas", [])] == [101, 102],
   "el primer planteamiento se queda con 101 y 102: ninguna sintética se los quitó")
ok([f["neun"] for f in g2.get("filas", [])] == [601],
   "el 101 no se repite en el segundo problema")
ok(len(g2.get("filas", [])) == 1 and len(g2.get("filas", [])) < fe.PISO_FILAS,
   "UNA fila basta: PISO_FILAS no aplica a la fuente OAJ")
ok(g1.get("resumen") == "" and g2.get("resumen") == "",
   "sin renglón de resumen, ni con dos filas ni con una («De estos 1 asuntos…»)")
ok(not any(f["neun"] == 901 for g in esp for f in g["filas"]),
   "nada de lo que sólo habría salido con la pregunta sola")

# La OAJ calla (sin tabla) → el espejo viejo, tal como hoy, con su piso.
_original = fe.espejo
LLAMADAS_VIEJO = []


async def espejo_viejo_falso(qdrant, embed, problema, clave, circ="22"):
    LLAMADAS_VIEJO.append((problema, clave, circ))
    n = 3 if problema == P1["pregunta"] else 2   # el resto queda bajo el piso
    # Un expediente por consulta: el global y P1 empiezan igual («¿La…»), y
    # con el prefijo de la pregunta la deduplicación los habría confundido.
    k = len(LLAMADAS_VIEJO)
    return [{"tipo_asunto": "Amparo Directo", "expediente": f"{k}{i}/2024",
             "fecha": "2024-01-0%d" % (i + 1), "sentido": "niega",
             "tema": "notificacion_electronica", "score": 0.72, "pdf_url": ""}
            for i in range(n)]


fe.espejo = espejo_viejo_falso
try:
    usar_calibracion(None)
    _consultas = consultas_como_produccion([P1, P2])
    esp = correr(ra._espejo_propio(QdrantFalso(por_problema), embed,
                                   resultado([P1, P2]), _consultas))
finally:
    fe.espejo = _original
ok([x[0] for x in LLAMADAS_VIEJO] == _consultas
   and all(x[1:] == ("3TCC", "22") for x in LLAMADAS_VIEJO),
   "sin la OAJ, el espejo viejo recibe las mismas cadenas que antes, con clave y circuito")
ok(len(esp) == 1 and esp[0]["problema"] == P1["pregunta"]
   and esp[0]["cobertura"] == fe.NOTA_COBERTURA,
   "y su tarjeta sale igual que antes: su cobertura, y los de dos filas bajo el piso")

# Fuera del circuito 22, o sin que conste, la OAJ no se consulta.
usar_calibracion(CAL)
for trib in ("Segundo Tribunal Colegiado en Materia Civil del Primer Circuito",
             "Tercer Tribunal Colegiado en Materia Administrativa",
             "Primer Tribunal Colegiado de Circuito del Centro Auxiliar de la "
             "Tercera Región, con residencia en Guadalajara, Jalisco",
             "Primer Tribunal Colegiado en Materia Penal del Decimoquinto Circuito"):
    q = QdrantFalso(por_problema)
    esp = correr(ra._espejo_propio(q, embed, resultado([P1], tribunal=trib),
                                   consultas_como_produccion([P1])))
    ok(not q.oaj() and not any("similitud" in f for g in esp for f in g["filas"]),
       f"«{trib[:50]}…» → la OAJ ni se consulta")

print("\n12b · POR consultar(), COMO EN PRODUCCIÓN")
import fase6_estudio as f6

_md, _sp, _tr = ra.f6rag.material_del_caso, ra._sondear_precedente, ra.f6rag.tesis_por_registro
RECIBIDO = {}


async def material_falso(qdrant, embed_juris, embed_leyes, problemas, *a, **k):
    RECIBIDO["problemas"] = list(problemas)
    return f6.Material()


async def sondeo_falso(*a, **k):
    return None


async def tesis_falsas(*a, **k):
    return []


ra.f6rag.material_del_caso = material_falso
ra._sondear_precedente = sondeo_falso
ra.f6rag.tesis_por_registro = tesis_falsas
try:
    usar_calibracion(CAL)
    CONSULTAS.clear()
    rp = resultado([P1, P2, P3, P4])
    mat = correr(ra.consultar(QdrantFalso(por_problema), None, embed, rp))
finally:
    ra.f6rag.material_del_caso = _md
    ra._sondear_precedente = _sp
    ra.f6rag.tesis_por_registro = _tr
ok(RECIBIDO.get("problemas") == consultas_como_produccion([P1, P2, P3, P4]),
   "el material sigue recibiendo sus cadenas de siempre (global, preguntas, sintéticas)")
ok(sorted(CONSULTAS) == sorted(fo.texto_consulta(p) for p in (P1, P2, P3)),
   "y la OAJ embebió «pregunta combate resolvio» de cada planteamiento completo, "
   "nada más")
ok([g["problema"] for g in mat.espejo] == [P1["pregunta"], P2["pregunta"]]
   and [[f["neun"] for f in g["filas"]] for g in mat.espejo] == [[101, 102], [601]],
   f"la tarjeta sale de los planteamientos reales (salió "
   f"{[[f['neun'] for f in g['filas']] for g in mat.espejo]})")

print("\n13 · LAS FILAS VIAJAN ENTERAS: SESIÓN Y RESPUESTA")
import taller_estado as te

usar_calibracion(CAL)
grupos = correr(ra._espejo_propio(QdrantFalso(por_problema), embed,
                                  resultado([P1, P2]),
                                  consultas_como_produccion([P1, P2])))
m = f6.Material()
m.tesis = [{"registro": "1", "rubro": "R", "texto": "t"}]
m.espejo = grupos
d = json.loads(json.dumps(te.material_ligero(m), ensure_ascii=False))
m2 = te.material_rehidratado(d)
ok(grupos and m2 is not None and m2.espejo == grupos,
   "por la fila de la sesión y de vuelta, sin perder similitud, pregunta, razón ni NEUN")
ok(grupos and set(grupos[0]["filas"][0]) == CAMPOS, "y con todos los campos")

# La respuesta de /taller filtra el espejo por grupo, no por campo. main.py no
# se puede importar aquí (arranca el servicio entero), así que se lee: si un día
# alguien proyecta campos en esa comprensión, esto lo dice.
src = open("main.py", encoding="utf-8").read()
ok(re.search(r'"espejo":\s*\[x for x in \(getattr\(material, "espejo", \[\]\) or \[\]\)'
             r'\s*if isinstance\(x, dict\)\]', src) is not None,
   "main.py manda cada grupo del espejo entero, sin proyectar campos")

print("\n14 · LA TABLA VIAJA CON EL REPO")
# El .gitignore se come los *.json. Si la tabla no se versiona, en Render la
# fuente OAJ calla siempre y nadie lo nota.
try:
    rc = subprocess.run(["git", "check-ignore", "-q", "oaj_calibracion.json"],
                        capture_output=True).returncode
    ok(rc == 1, "oaj_calibracion.json NO está ignorado por git")
except FileNotFoundError:
    print("   (sin git: no se comprobó)")

print()
if FALLOS:
    print("FALLAS:")
    for f in FALLOS:
        print("  ✗", f)
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
