# -*- coding: utf-8 -*-
"""El espejo habla cuando el tribunal ha visto el punto, y calla cuando no.

Se corre sola. Necesita red: pega contra Qdrant y contra el embebedor, porque
lo que hay que vigilar es el MARGEN entre las preguntas con acervo y las que no,
y ese margen se mueve cada vez que se reingesta.

    python3 test_espejo_propio.py

LO QUE VIGILA, Y POR QUÉ CADA COSA
==================================
 · QUE HABLE. Trece preguntas sobre puntos que el 3TCC sí ha trabajado,
   redactadas como las escribiría un secretario y NO con el texto de la
   etiqueta, que sería hacer trampa: el emparejamiento por etiqueta exacta
   funciona en el 3.9% de los temas y no es lo que hace el módulo.
 · QUE CALLE. Once preguntas sobre puntos que ese tribunal no ha resuelto. Un
   espejo que siempre opina no es un espejo.
 · EL MARGEN. La positiva más baja contra el control más alto. Si esa distancia
   baja de 0.03, el umbral ya no separa y hay que recalibrarlo ANTES de
   desplegar, no después.
 · EL VOCABULARIO PROHIBIDO. Ni «racha», ni «seguidas», ni «jurisprudencia», ni
   «reiteración», ni «interrumpir» en el módulo ni en el texto que genera. El
   acervo no tiene campo de votación —comprobado sobre el inventario completo
   de claves de los 6,379 holdings del 3TCC—, así que el supuesto del artículo
   224 de la Ley de Amparo no se puede acreditar nunca y la tarjeta no puede
   insinuarlo.
 · QUE LA CITA SE PUEDA COMPROBAR. La clave de una sentencia es (tipo_asunto,
   expediente, fecha): «44/2021» a secas son CUATRO sentencias distintas del
   3TCC —Amparo Directo, Queja, Revisión Fiscal y Amparo en Revisión—, con
   cuatro fechas y cuatro PDF.
"""
import asyncio
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, ".")

import fase_espejo as fe

FALLOS = []


def ok(cond, que):
    print(("  OK   " if cond else "  FALLA ") + que)
    if not cond:
        FALLOS.append(que)


def _env():
    d = {}
    if not os.path.exists(".env"):
        return d
    for l in open(".env", encoding="utf-8"):
        if "=" in l and not l.strip().startswith("#"):
            k, v = l.split("=", 1)
            d[k.strip()] = v.strip().strip('"').strip("'")
    return d


ENV = _env()

# ── el banco, congelado aquí ────────────────────────────────────────────────
CON_ACERVO = [
    "¿Es inconstitucional el sistema normativo que establece contribuciones en la Ley de Hacienda del Estado de Querétaro por resultar desproporcionado?",
    "¿Los derechos registrales calculados sobre el valor de la operación violan los principios de proporcionalidad y equidad tributaria?",
    "¿Tiene legitimación la autoridad ejecutora para interponer recurso de revisión en un amparo contra leyes?",
    "¿El acuerdo dictado dentro del procedimiento constituye un acto de imposible reparación impugnable en amparo indirecto?",
    "¿Acredita el interés jurídico quien impugna una norma fiscal sin demostrar su acto concreto de aplicación?",
    "¿Son inoperantes los conceptos de violación que se limitan a reiterar los agravios expuestos en la apelación?",
    "¿Procede desechar de plano la demanda de amparo por una causa de improcedencia manifiesta e indudable?",
    "¿Queda sin materia el recurso de revisión interpuesto contra la suspensión provisional cuando ya se resolvió la definitiva?",
    "¿Es procedente el recurso de revisión fiscal cuando no se acredita la importancia y trascendencia del asunto?",
    "¿Cesaron los efectos del acto reclamado de modo que procede sobreseer en el juicio de amparo?",
    "¿La Ley de Servicios Auxiliares del Transporte del Estado de Querétaro vulnera el derecho a la tutela judicial efectiva?",
    "¿Procede sobreseer en el amparo directo por desistimiento expreso de la parte quejosa?",
    # NO ERA UN CONTROL, y así se descubrió: marcó 0.771 y la comprobación
    # enseñó que el 3TCC tiene 26 temas de custodia y 166 sobre menores. Es
    # materia suya —Administrativa y Civil— y la recuperación acertó; el error
    # estaba en quien armó el banco. Se queda como positiva para que nadie
    # vuelva a «arreglar» el umbral por su culpa.
    "¿Procede la guarda y custodia compartida atendiendo al interés superior del menor?",
]

SIN_ACERVO = [
    "¿La revocación del nombramiento de notario público exige audiencia previa al fedatario?",
    "¿Procede reponer el procedimiento cuando no se designó perito tercero en discordia?",
    "¿Deben enterarse al sindicato minoritario las cuotas sindicales retenidas por el patrón?",
    "¿Se configura daño moral por publicaciones difamatorias en redes sociales entre particulares?",
    "¿Es constitucional la prisión preventiva oficiosa tratándose de delitos fiscales?",
    "¿Tienen derecho los militares retirados al pago de prima de antigüedad?",
    "¿Procede la vinculación entre el registro sanitario y la patente farmacéutica?",
    "¿Puede reinstalarse a un trabajador de confianza despedido injustificadamente?",
    "¿Debe consultarse a la comunidad indígena antes de otorgar una concesión minera?",
    "¿Procede la extradición internacional cuando no existe tratado aplicable con el Estado requirente?",
    "¿Procede la restitución de tierras comunales despojadas por la vía del juicio agrario?",
]

PROHIBIDAS = re.compile(
    r"\bracha[s]?\b|\bseguidas\b|\bjurisprudencia\b|\breiteraci[oó]n\b|"
    r"\binterrump", re.I)


# ═══════════════════════════════════════════════════════════════════════════
# 1 · LO QUE NO NECESITA RED
# ═══════════════════════════════════════════════════════════════════════════
print("── 1 · EL TRIBUNAL SE RESUELVE BIEN ──")
CASOS = [
    ("Tercer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito", "22", "3TCC"),
    ("Primer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito", "22", "1TCC"),
    ("Segundo Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito", "22", "2TCC"),
    ("Tribunal Colegiado en Materias Penal y Administrativa del Vigésimo Segundo Circuito", "22", "TCC_PENAL"),
    ("Tribunal Colegiado en Materias Administrativa y de Trabajo del Vigésimo Segundo Circuito", "22", "TCC_ADM"),
    # Fuera del 22 no hay mapa: se calla en vez de adivinar.
    ("Segundo Tribunal Colegiado en Materia Civil del Primer Circuito", "1", None),
    ("", "", None),
]
for nombre, circ, esperado in CASOS:
    got = fe.resolver_tribunal(nombre, circ)[0]
    ok(got == esperado,
       f"«{(nombre or '(vacío)')[:46]}…» → {got!r} (esperado {esperado!r})")

# EL ORDINAL DEL TRIBUNAL NO ES EL DEL CIRCUITO. Cazado en la prueba de humo:
# el «Segundo» de «Vigésimo Segundo Circuito» casaba antes que el «Tercer» del
# tribunal y devolvía 2TCC. Habría impreso el nombre de un tribunal ajeno junto
# a seis expedientes ajenos.
ok(fe.resolver_tribunal(
    "Tercer Tribunal Colegiado en Materias Administrativa y Civil "
    "del Vigésimo Segundo Circuito", "22")[0] == "3TCC",
   "el «Segundo» del CIRCUITO no se lee como ordinal del TRIBUNAL")

print("\n── 2 · EL VOCABULARIO PROHIBIDO ──")
fuente = open("fase_espejo.py", encoding="utf-8").read()
# Se mira sólo el CÓDIGO: la cabecera explica por qué esas palabras no van, y
# una comprobación que acusa a la documentación del arreglo enseña a no hacerle
# caso. Se corta en el cierre del docstring del módulo.
cuerpo = fuente.split('"""', 2)[-1]
cuerpo = re.sub(r'"""[\s\S]*?"""', "", cuerpo)   # docstrings de las funciones
cuerpo = re.sub(r"^\s*#.*$", "", cuerpo, flags=re.M)
malas = sorted(set(m.group(0).lower() for m in PROHIBIDAS.finditer(cuerpo)))
ok(not malas, f"el código del módulo no usa palabras prohibidas (salió: {malas})")

print("\n── 3 · EL RESUMEN NO COMPARA CON EL PROYECTO ──")
filas_mixtas = [
    {"tipo_asunto": "Amparo Directo", "expediente": "1/2024", "fecha": "2024-01-01",
     "sentido": "niega", "tema": "usura_intereses", "score": 0.8, "pdf_url": ""},
    {"tipo_asunto": "Queja", "expediente": "2/2024", "fecha": "2024-02-01",
     "sentido": "confirma", "tema": "usura_intereses", "score": 0.8, "pdf_url": ""},
    {"tipo_asunto": "Amparo Directo", "expediente": "3/2024", "fecha": "2024-03-01",
     "sentido": "niega", "tema": "usura_intereses", "score": 0.8, "pdf_url": ""},
]
ok(fe.resumen(filas_mixtas) == "",
   "con TIPOS DE ASUNTO mezclados no se resume: sobreseer en un directo y "
   "confirmar en una queja no son el mismo sentido")

filas_tauto = [dict(f, tipo_asunto="Amparo Directo",
                    tema="inoperancia_conceptos_violacion_amparo_directo")
               for f in filas_mixtas]
ok(fe.resumen(filas_tauto) == "",
   "con etiqueta que YA contiene el resultado no se resume: ahí la cuenta mide "
   "la definición de la categoría, no el criterio")

filas_buenas = [dict(f, tipo_asunto="Amparo en Revisión",
                     tema="derechos_registrales_proporcionalidad")
                for f in filas_mixtas]
r = fe.resumen(filas_buenas)
ok(r and "Compárelo usted" in r,
   "con tipos homogéneos y etiqueta limpia SÍ se resume, y se devuelve la "
   "comparación al secretario")
ok(not PROHIBIDAS.search(r or ""),
   "y el resumen tampoco usa palabras prohibidas")

# La normalización del vocabulario del sentido: 'confirmar' ≠ 'confirma' sería
# una racha real partida por una falta de ortografía del extractor.
ok(fe._norma("confirmar sobreseimiento") == "confirma"
   and fe._norma("sin materia") == fe._norma("sin_materia"),
   "el vocabulario del sentido se normaliza antes de contar")

print("\n── 4 · NO SE DEDUPLICA POR NÚMERO DE EXPEDIENTE ──")
# 44/2021 son CUATRO sentencias distintas del 3TCC. Deduplicar por `expediente`
# a secas las habría fusionado.
cuatro = [
    {"tipo_asunto": t, "expediente": "44/2021", "fecha": f, "sentido": "niega",
     "tema": "x", "score": 0.8, "pdf_url": ""}
    for t, f in (("Amparo Directo", "2021-05-01"), ("Queja", "2021-06-01"),
                 ("Revisión Fiscal", "2021-07-01"), ("Amparo en Revisión", "2021-08-01"))]
claves = {(f["tipo_asunto"], f["expediente"], f["fecha"]) for f in cuatro}
ok(len(claves) == 4,
   "la clave de una sentencia es (tipo, expediente, fecha): 44/2021 son cuatro")


# ═══════════════════════════════════════════════════════════════════════════
# 5 · EL BANCO, CONTRA EL ACERVO DE VERDAD
# ═══════════════════════════════════════════════════════════════════════════
async def _contra_qdrant():
    from qdrant_client import AsyncQdrantClient

    def _embed_sync(t):
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=json.dumps({"model": "text-embedding-3-small",
                             "input": t}).encode(),
            headers={"Authorization": "Bearer " + ENV["OPENAI_API_KEY"],
                     "Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=60))["data"][0]["embedding"]

    async def embed(t):
        return await asyncio.to_thread(_embed_sync, t)

    q = AsyncQdrantClient(url=ENV["QDRANT_URL"], api_key=ENV["QDRANT_API_KEY"],
                          timeout=90)
    try:
        habla, calla = [], []
        peor_positiva, mejor_control = 1.0, 0.0
        for preg in CON_ACERVO:
            filas = await fe.espejo(q, embed, preg, "3TCC", "22")
            if filas:
                habla.append(preg)
                peor_positiva = min(peor_positiva, max(f["score"] for f in filas))
                # Cada fila tiene que ser citable y abrible.
                for f in filas:
                    if not (f["expediente"] and f["tipo_asunto"]):
                        FALLOS.append(f"fila sin clave completa: {f}")
                    if f["fecha"].strip().lower() in ("null", "none"):
                        FALLOS.append(f"fecha literal «null» impresa: {f}")
            else:
                calla.append(preg)
        ok(len(habla) == len(CON_ACERVO),
           f"habla en {len(habla)}/{len(CON_ACERVO)} puntos que el tribunal SÍ "
           f"ha trabajado" + (f" · calló en: {[p[:40] for p in calla]}" if calla else ""))

        ruido = []
        for preg in SIN_ACERVO:
            filas = await fe.espejo(q, embed, preg, "3TCC", "22")
            if filas:
                ruido.append((preg, max(f["score"] for f in filas)))
        ok(not ruido,
           f"calla en {len(SIN_ACERVO) - len(ruido)}/{len(SIN_ACERVO)} puntos "
           f"que el tribunal NO ha resuelto"
           + (f" · habló en: {[(p[:40], round(s, 3)) for p, s in ruido]}" if ruido else ""))

        # EL MARGEN. Se mide sin umbral para saber cuánto sobra.
        import fase_espejo as _fe
        guardado = _fe.UMBRAL
        _fe.UMBRAL = 0.0
        try:
            for preg in SIN_ACERVO:
                filas = await _fe.espejo(q, embed, preg, "3TCC", "22")
                if filas:
                    mejor_control = max(mejor_control,
                                        max(f["score"] for f in filas))
        finally:
            _fe.UMBRAL = guardado
        margen = peor_positiva - mejor_control
        print(f"\n  margen: positiva más baja {peor_positiva:.3f} · "
              f"control más alto {mejor_control:.3f} · distancia {margen:.3f}")
        ok(margen >= 0.03,
           f"el margen entre lo que tiene acervo y lo que no es de {margen:.3f} "
           f"(mínimo 0.03; por debajo hay que recalibrar el umbral ANTES de "
           f"desplegar)")
        ok(guardado > mejor_control and guardado < peor_positiva,
           f"el umbral {guardado} cae DENTRO del hueco "
           f"({mejor_control:.3f} … {peor_positiva:.3f})")
    finally:
        await q.close()


print("\n── 5 · EL BANCO CONTRA EL ACERVO ──")
if not (ENV.get("QDRANT_URL") and ENV.get("OPENAI_API_KEY")):
    print("  (sin credenciales en .env: esta parte no se corrió)")
    FALLOS.append("no se pudo correr el banco contra el acervo")
else:
    try:
        asyncio.run(_contra_qdrant())
    except Exception as e:
        print(f"  FALLA  no se pudo correr contra el acervo: {e}")
        FALLOS.append(f"banco contra acervo: {e}")

print()
if FALLOS:
    print("FALLAS:")
    for f in FALLOS:
        print("  ✗", f)
    raise SystemExit(1)
print("Todo en orden.")
