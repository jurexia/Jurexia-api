# -*- coding: utf-8 -*-
"""La fuente OAJ del espejo habla sólo con tabla y al 85%, y si no, calla.

Se corre sola y SIN RED: un Qdrant falso que respeta los filtros y el umbral,
y un embebedor falso que devuelve el propio texto para poder ver qué se
consultó.

    .venv/bin/python test_fase_oaj.py

LO QUE VIGILA, Y POR QUÉ CADA COSA
==================================
 · SIN TABLA NO HAY PORCENTAJE. Si el JSON de calibración no existe, no trae
   el tipo o está mal formado, la función devuelve [] —y ni siquiera paga el
   embebido—. Inventar el porcentaje es de la especie de la tesis inventada.
 · BAJO EL 85%, NADA. Y el porcentaje se redondea hacia abajo.
 · UN NEUN, UNA FILA. Tres planteamientos de la misma sentencia son un
   precedente, no tres.
 · EL RESPALDO POR TEMA sólo cuando ningún planteamiento pasa, y la fila dice
   que vino por tema.
 · EL FILTRO: otro tribunal u otro tipo no entran aunque se parezcan más.
 · EL MAPA DEL TRIBUNAL al nombre exacto con que la OAJ lo indexa.
 · EL FORMATO de la fila, campo por campo, y que viaje entero por la fila de
   la sesión y por la respuesta de /taller.
 · EN EL TALLER: la OAJ primero, sin piso de filas y sin repetir entre
   problemas; y si calla, el espejo viejo exactamente como antes.
"""
import asyncio
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


CAL = {
    "planteamiento": {
        # 0.70 es donde la tabla alcanza 0.85: ése es el corte.
        "Amparo Directo": [[0.0, 0.60, 0.10], [0.60, 0.70, 0.50],
                           [0.70, 0.75, 0.869], [0.75, 1.0, 0.95]],
    },
    "asunto": {
        "Amparo Directo": [[0.0, 0.65, 0.20], [0.65, 1.0, 0.90]],
    },
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


PROB = {"pregunta": "¿La constancia de notificación electrónica debía llevar "
                    "firma electrónica avanzada?",
        "combate": "La Sala omitió valorar la falta de firma.",
        "resolvio": "La Sala tuvo por legal la notificación.",
        "impedimento": None}


# ═══════════════════════════════════════════════════════════════════════════
print("\n1 · EL MAPA DEL TRIBUNAL AL NOMBRE DE LA OAJ")
_R = ", con residencia en Querétaro, Querétaro"
CASOS = [
    ("Primer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito",
     "22", "1TCC", "Primer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito" + _R),
    ("Segundo Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito",
     "22", "2TCC", "Segundo Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito" + _R),
    # El «Segundo» del CIRCUITO no es el ordinal del tribunal.
    ("Tercer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito",
     "22", "3TCC", "Tercer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito" + _R),
    ("Tribunal Colegiado en Materias Administrativa y de Trabajo del Vigésimo Segundo Circuito",
     "22", "TCC_ADM", "Tribunal Colegiado en Materias Administrativa y de Trabajo del Vigésimo Segundo Circuito" + _R),
    ("Tribunal Colegiado en Materias Penal y Administrativa del Vigésimo Segundo Circuito",
     "22", "TCC_PENAL", "Tribunal Colegiado en Materias Penal y Administrativa del Vigésimo Segundo Circuito" + _R),
    # Fuera del 22 no hay mapa: se calla en vez de adivinar.
    ("Segundo Tribunal Colegiado en Materia Civil del Primer Circuito", "1", None, ""),
    ("", "", None, ""),
]
for nombre, circ, clave, organo in CASOS:
    got = fo.organo_de(nombre, circ)
    ok(got == (clave, organo),
       f"«{(nombre or '(vacío)')[:44]}…» → {got[0]!r}"
       + (" con residencia" if got[1].endswith(_R) else ""))

print("\n2 · LOS TIPOS DEL TALLER A LA GRAFÍA DE LA OAJ")
for t, esperado in (("amparo_directo", "Amparo Directo"),
                    ("amparo_revision", "Amparo en revisión"),
                    ("queja", "Queja"), ("revision_fiscal", "Revisión Fiscal"),
                    ("Amparo en revisión", "Amparo en revisión"),
                    ("rf", "Revisión Fiscal"), ("reclamacion", ""), ("", "")):
    ok(fo.tipo_oaj(t) == esperado, f"{t!r} → {fo.tipo_oaj(t)!r}")

print("\n3 · LA CONSULTA ES «pregunta combate resolvio»")
ok(fo.texto_consulta(PROB) == " ".join([PROB["pregunta"], PROB["combate"],
                                         PROB["resolvio"]]),
   "del problema entero, en ese orden, sin el impedimento")
ok(fo.texto_consulta("¿Sólo la pregunta?") == "¿Sólo la pregunta?",
   "una sesión vieja con la pregunta sola busca con ella")
ok(fo.texto_consulta({"pregunta": "", "combate": ""}) == "",
   "un problema vacío no consulta")

print("\n4 · LA TABLA: ESCALONADA, SIN EXTRAPOLAR, Y DESCARTADA SI ESTÁ MAL")
t = fo.tabla_de(CAL, "planteamiento", "Amparo Directo")
ok(fo.probabilidad(t, 0.72) == 0.869 and fo.probabilidad(t, 0.99) == 0.95,
   "el coseno cae en su tramo")
ok(fo.probabilidad(fo.tabla_de({"a": {"X": [[0.7, 1.0, 0.9], [0.5, 0.6, 0.5]]}},
                                "a", "X"), 0.65) == 0.5,
   "un coseno en el HUECO entre tramos toma el de abajo (tabla desordenada a propósito)")
ok(fo.probabilidad(fo.tabla_de({"a": {"X": [[0.5, 0.6, 0.9]]}}, "a", "X"), 0.4) is None,
   "por debajo del primer tramo no hay dato y no se extrapola")
for malo, que in (([[0.1, 0.2]], "tramo de dos números"),
                  ([[0.1, 0.2, 1.3]], "probabilidad > 1"),
                  ([[0.3, 0.2, 0.9]], "tramo al revés"),
                  ([[0.1, 0.2, "x"]], "texto en vez de número"),
                  ([], "tabla vacía")):
    ok(fo.tabla_de({"planteamiento": {"Queja": malo}}, "planteamiento", "Queja") is None,
       f"tabla mal formada ({que}) → se descarta entera")
ok(fo._corte(CAL, "planteamiento", "Amparo Directo", t) == 0.70,
   "el corte es el inicio del primer tramo que llega a 0.85")
_endurecida = dict(CAL, umbral_85={"planteamiento": {"Amparo Directo": 0.76}})
ok(fo._corte(_endurecida, "planteamiento", "Amparo Directo", t) == 0.76,
   "`umbral_85` más alto ENDURECE el corte")
_blanda = dict(CAL, umbral_85={"planteamiento": {"Amparo Directo": 0.50}})
ok(fo._corte(_blanda, "planteamiento", "Amparo Directo", t) == 0.70,
   "`umbral_85` más bajo NO lo ablanda: manda la tabla")

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

usar_calibracion({"planteamiento": {"Amparo Directo": [[0.0, 1.0, 0.80]]}})
ok(correr(fo.precedentes_oaj(QdrantFalso(PUNTOS), embed, PROB,
                             "amparo_directo", ORG3)) == [],
   "una tabla que nunca llega a 0.85 → []")

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

# Sin NEUN no se enseña.
filas = correr(fo.precedentes_oaj(QdrantFalso([(0.9, pl(0)), (0.9, pl("x"))]),
                                  embed, PROB, "amparo_directo", ORG3))
ok(filas == [], "una fila sin NEUN válido no se enseña: no se podría buscar en la OAJ")

# `umbral_85` más estricto que la tabla: 102 (0.71) queda fuera.
usar_calibracion(_endurecida)
filas = correr(fo.precedentes_oaj(QdrantFalso(PUNTOS), embed, PROB,
                                  "amparo_directo", ORG3))
ok([f["neun"] for f in filas] == [101], "con `umbral_85` en 0.76 sólo pasa 101")

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

print("\n10 · EL RENGLÓN DE RESUMEN")
base = {"tipo_asunto": "Amparo Directo", "expediente": "1/2025", "fecha": "01-01-2025",
        "sentido": "concede", "tema": "Firma electrónica", "score": 0.8}
ok(fo.resumen([base, dict(base, sentido="")]) == "",
   "si a una fila le falta el sentido, no se resume: la cuenta se leería como de todas")
ok(fo.resumen([base, dict(base, tema="Inoperancia de los conceptos de violación")]) == "",
   "un tema en PROSA que ya contiene el resultado también calla")
r = fo.resumen([base, dict(base, expediente="2/2025")])
ok(r.startswith("De estos 2 asuntos propios") and "Compárelo usted" in r,
   "con sentidos completos y tema limpio, se describe y se devuelve al secretario")

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

r_ = types.SimpleNamespace(encargo=types.SimpleNamespace(
    tribunal="Tercer Tribunal Colegiado en Materias Administrativa y Civil "
             "del Vigésimo Segundo Circuito",
    tipo_asunto="amparo_directo"))
P1 = dict(PROB)
P2 = {"pregunta": "¿La revisión de gabinete concluyó en doce meses?",
      "combate": "El plazo del 46-A se excedió.", "resolvio": "Concluyó a tiempo."}
P3 = {"pregunta": "¿Procedía la condena en costas?", "combate": "x", "resolvio": "y"}


def por_problema(consulta):
    # P1 recupera 101 y 102; P2 recupera 101 (repetido) y 601; P3 nada.
    if consulta.startswith("¿La constancia"):
        return [(0.80, pl(101)), (0.71, pl(102))]
    if consulta.startswith("¿La revisión"):
        return [(0.78, pl(101)), (0.74, pl(601))]
    return [(0.50, pl(999))]


usar_calibracion(CAL)
CONSULTAS.clear()
esp = correr(ra._espejo_propio(QdrantFalso(por_problema), embed, r_, [P1, P2, P3]))
ok(len(esp) == 2, f"dos grupos: P3 no tiene nada al 85% (salieron {len(esp)})")
g1 = esp[0] if esp else {}
g2 = esp[1] if len(esp) > 1 else {}
ok(set(g1) == {"problema", "tribunal", "filas", "resumen", "cobertura"},
   "el grupo tiene la forma de siempre")
ok(g1.get("problema") == P1["pregunta"], "el grupo se rotula con la pregunta")
ok(g1.get("cobertura") == "Índice de la OAJ: todas las sentencias públicas "
                          "del tribunal de los cuatro tipos.",
   "con la nota de cobertura de la OAJ")
ok(g1.get("tribunal") == fe.TRIBUNALES_22["3TCC"],
   "el tribunal se lee con su nombre de siempre, sin la residencia del filtro")
ok([f["neun"] for f in g2.get("filas", [])] == [601],
   "el 101 no se repite en el segundo problema")
ok(len(g2.get("filas", [])) == 1 and len(g2.get("filas", [])) < fe.PISO_FILAS,
   "UNA fila basta: PISO_FILAS no aplica a la fuente OAJ")
ok(any(c.startswith(P2["pregunta"] + " " + P2["combate"]) for c in CONSULTAS),
   "se consultó con el problema ENTERO, no con la pregunta sola")

# La OAJ calla (sin tabla) → el espejo viejo, tal como hoy, con su piso.
_original = fe.espejo
LLAMADAS_VIEJO = []


async def espejo_viejo_falso(qdrant, embed, problema, clave, circ="22"):
    LLAMADAS_VIEJO.append((problema, clave, circ))
    n = 3 if problema == P1["pregunta"] else 2   # P2 queda bajo el piso
    return [{"tipo_asunto": "Amparo Directo", "expediente": f"{problema[:3]}{i}/2024",
             "fecha": "2024-01-0%d" % (i + 1), "sentido": "niega",
             "tema": "notificacion_electronica", "score": 0.72, "pdf_url": ""}
            for i in range(n)]


fe.espejo = espejo_viejo_falso
try:
    usar_calibracion(None)
    esp = correr(ra._espejo_propio(QdrantFalso(por_problema), embed, r_, [P1, P2]))
finally:
    fe.espejo = _original
ok(len(LLAMADAS_VIEJO) == 2 and LLAMADAS_VIEJO[0] == (P1["pregunta"], "3TCC", "22"),
   "sin la OAJ se llama al espejo viejo con la pregunta, la clave y el circuito")
ok(len(esp) == 1 and esp[0]["cobertura"] == fe.NOTA_COBERTURA,
   "y su tarjeta sale igual que antes: su cobertura, y P2 (dos filas) bajo el piso")

# Fuera del circuito 22, ni una ni otra.
r_fuera = types.SimpleNamespace(encargo=types.SimpleNamespace(
    tribunal="Segundo Tribunal Colegiado en Materia Civil del Primer Circuito",
    tipo_asunto="amparo_directo"))
usar_calibracion(CAL)
ok(correr(ra._espejo_propio(QdrantFalso(por_problema), embed, r_fuera, [P1])) == [],
   "fuera del circuito 22 no hay mapa y el espejo calla")

print("\n13 · LAS FILAS VIAJAN ENTERAS: SESIÓN Y RESPUESTA")
import fase6_estudio as f6
import taller_estado as te

usar_calibracion(CAL)
grupos = correr(ra._espejo_propio(QdrantFalso(por_problema), embed, r_, [P1, P2]))
m = f6.Material()
m.tesis = [{"registro": "1", "rubro": "R", "texto": "t"}]
m.espejo = grupos
d = json.loads(json.dumps(te.material_ligero(m), ensure_ascii=False))
m2 = te.material_rehidratado(d)
ok(m2 is not None and m2.espejo == grupos,
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
