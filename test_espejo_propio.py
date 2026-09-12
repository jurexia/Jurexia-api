# -*- coding: utf-8 -*-
"""El espejo habla cuando el tribunal ha visto el punto, y calla cuando no.

Se corre sola. Necesita red: pega contra Qdrant y contra el embebedor, porque
lo que hay que vigilar es el MARGEN entre las preguntas con acervo y las que no,
y ese margen se mueve cada vez que se reingesta.

    python3 test_espejo_propio.py

LO QUE VIGILA, Y POR QUÉ CADA COSA
==================================
 · QUE HABLE. Los planteamientos REALES que el pipeline produce, congelados en
   abajo, desde seis asuntos del propio secretario. Un
   banco escrito a mano daba 13 de 13 y era mentira: contra los de verdad, el
   espejo hablaba en cero. Sólo mide quien mide la entrada real.
 · QUE CALLE. Siete puntos que ese tribunal no ha resuelto. Un espejo que
   siempre opina no es un espejo.
 · QUE EL UMBRAL QUEDE POR ENCIMA DE TODO CONTROL. No hay hueco limpio —el peor
   planteamiento real puntúa por debajo del mejor control—, así que se vigila el
   lado por el que se falla barato: callar.
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

# ── EL BANCO, Y POR QUÉ ES ÉSTE Y NO OTRO ───────────────────────────────────
# La primera versión de esta prueba traía 24 preguntas escritas a mano y daba
# 13/13 y 0/11. Era mentira: estaban escritas a partir de los temas mayores del
# acervo —o sea, preguntando lo que el corpus sabía contestar— y en lenguaje de
# doctrina. Contra los planteamientos REALES que la fase 3 produce, el espejo
# hablaba en CERO de veintitrés.
#
# Así que el banco positivo son los planteamientos de verdad, tal como salen del
# pipeline, congelados abajo desde seis asuntos reales
# del propio secretario. Es el único banco que mide lo que va a pasar.
#
# Van escritos aquí y no en un fichero suelto porque el .gitignore del repo se
# come los .json y un banco que no viaja con la prueba no es un banco. Son los
# planteamientos, no los expedientes: cuestiones jurídicas, sin partes ni datos
# de nadie.
CON_ACERVO = [
    # 410/2026
    "¿La quejosa podía reclamar el emplazamiento y las actuaciones del juicio de nulidad como tercera extraña a juicio?",
    "¿La inexistencia de sentencia definitiva justificaba sobreseer respecto de los actos de ejecución reclamados?",
    "¿Procedía suplir la deficiencia de la queja y valorar la sentencia ofrecida como prueba?",
    # 393/2025
    "¿La Sala examinó las excepciones conforme a los términos en que fueron planteadas?",
    "¿La Sala empleó parámetros idóneos para examinar si las tasas del crédito habitacional eran compatibles con su finalidad social?",
    "¿La Sala examinó la procedencia jurídica de los intereses moratorios pactados por el INFONAVIT?",
    "¿La Sala examinó la compatibilidad de las tasas con el derecho al mínimo vital y la situación económica de la quejosa?",
    "¿La Sala examinó la forma de imputar los pagos frente al derecho a adquirir el dominio de la vivienda?",
    "¿La Sala podía considerar hechos relativos a programas de apoyo y reestructura durante la pandemia?",
    "¿La Sala podía confirmar una condena por cantidad distinta de la líquida reclamada?",
    "¿La Sala motivó suficientemente la valoración de las tasas y de la cantidad reclamada?",
    "¿La Sala fue exhaustiva al pronunciarse sobre las objeciones formuladas contra los intereses y la imputación de pagos?",
    # 410/2025
    "¿La quejosa estaba vinculada al juicio de nulidad por haber sido representada en su promoción?",
    "¿Existían la sentencia administrativa y los actos atribuidos a su ejecución?",
    "¿Debía suplirse la deficiencia de la queja y valorarse la sentencia ofrecida como prueba?",
    # 91/2025
    "¿La constancia de notificación electrónica debía contener la firma electrónica avanzada del funcionario competente?",
    "¿La revisión de gabinete concluyó dentro del plazo de doce meses previsto en el artículo 46-A del Código Fiscal de la Federación?",
    # 93/2026
    "¿La parte quejosa presentó oportunamente la ampliación de demanda conforme al plazo aplicable?",
    "¿La Sala responsable debía pronunciarse sobre los alegatos presentados por la parte quejosa?",
    # 412/2026
    "¿La controversia derivada del convenio correspondía a la jurisdicción laboral?",
    "¿La responsable valoró la contraprestación pactada en la cláusula séptima del convenio?",
    "¿La obligación de no competencia contravino la libertad de trabajo y la prohibición de renunciar a derechos laborales?",
    "¿La sentencia es congruente al declarar infundada la incompetencia y aplicar el artículo 33 de la Ley Federal del Trabajo?",
]

# Puntos que este tribunal no ha resuelto, de materias que no son las suyas.
SIN_ACERVO = [
    "¿La revocación del nombramiento de notario público exige audiencia previa al fedatario?",
    "¿Procede la vinculación entre el registro sanitario y la patente farmacéutica?",
    "¿Procede la extradición internacional cuando no existe tratado aplicable con el Estado requirente?",
    "¿Debe consultarse a la comunidad indígena antes de otorgar una concesión minera?",
    "¿Tienen derecho los militares retirados al pago de prima de antigüedad?",
    "¿Es constitucional la prisión preventiva oficiosa tratándose de delitos fiscales?",
    "¿Se configura daño moral por publicaciones difamatorias en redes sociales entre particulares?",
]

# LA COBERTURA REAL, Y ES BAJA. Medida sobre los 23 planteamientos: el espejo
# habla en DOS. Ojo con el número, porque yo mismo me equivoqué al leerlo: el
# 6 de 23 que salió al calibrar contaba los planteamientos cuya MEJOR
# coincidencia pasaba el umbral, y el módulo exige TRES sentencias distintas por
# encima de él. Al deduplicar los trozos por sentencia, cuatro de esos seis se
# quedan en una o dos. La cobertura verdadera es del 9%.
#
# No se exige que hable en todos —las distribuciones se solapan: el peor
# planteamiento real puntúa por debajo del mejor control, y no hay umbral que lo
# consiga sin mentir en alguno—. Se exige que hable en ALGUNO y que no hable en
# NINGUNO de los controles. Si la cobertura cae a cero, la función dejó de
# existir sin avisar, que es como se apagan estas cosas.
MINIMO_COBERTURA = 2

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
        ok(len(CON_ACERVO) > 0, "el banco de planteamientos reales se cargó")
        ok(len(habla) >= MINIMO_COBERTURA,
           f"habla en {len(habla)}/{len(CON_ACERVO)} planteamientos REALES "
           f"(mínimo {MINIMO_COBERTURA}; medido: 2, o sea el 9%). Por debajo, "
           f"la función se apagó sin avisar")

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
        print(f"\n  control más alto sin umbral: {mejor_control:.3f} · "
              f"umbral {guardado}")
        # NO SE EXIGE UN HUECO LIMPIO, porque medido no lo hay: el peor
        # planteamiento real puntúa por debajo del mejor control. Lo que se
        # exige es que el umbral quede POR ENCIMA de todo control, que es el
        # lado por el que se puede fallar barato.
        ok(guardado > mejor_control,
           f"el umbral {guardado} queda por encima del control más alto "
           f"({mejor_control:.3f}); si no, el espejo opinaría sobre puntos que "
           f"el tribunal no ha visto")
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
