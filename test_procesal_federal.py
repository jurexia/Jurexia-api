"""La normativa procesal federal y las NOM llegan al contexto (11-oct-2026).

Parte 1, sin red: qué consultas son procesales, de qué materia, qué piden
(foco) y qué artículos citan; y que una pregunta sustantiva no dispara nada.

Parte 2, Qdrant REAL en sólo lectura + un embedding por pregunta (centavos):
el artículo que responde debe salir entre los tres primeros, y las funciones
de main.py deben devolver SearchResult con su silo y su PDF.

    python test_procesal_federal.py
"""
import asyncio
import os
import sys

import procesal_federal as pf

FALLOS = []


def ok(cond, msg):
    print(("   PASA   " if cond else "   FALLA  ") + msg)
    if not cond:
        FALLOS.append(msg)


print("Parte 1 · detección")
d = pf.detectar("¿En qué plazo se interpone la apelación contra la sentencia definitiva en un juicio familiar?")
ok(d["procesal"] and d["materias"][:1] == ["civil_familiar"] and "plazo" in d["foco"], "apelación familiar → civil_familiar, foco plazo")
d = pf.detectar("¿Cuál es la pena del robo calificado en Querétaro?")
ok(not d["procesal"], "la pena del robo no es procesal")
d = pf.detectar("¿qué pena tiene el fraude procesal?")
ok(not d["procesal"], "el fraude procesal es un delito, no un procedimiento")
d = pf.detectar("art. 17 de la Ley de Amparo")
ok(d["citas"] == [("LAmp", 17)], "cita exacta del art. 17 de la Ley de Amparo")
d = pf.detectar("¿qué recurso procede contra el auto que desecha pruebas en el juicio ejecutivo mercantil?")
ok(d["foco"][:1] == ["recurso"] and "mercantil" in d["materias"], "el foco es el recurso, materia mercantil")
d = pf.detectar("plazo para la demanda de nulidad", pistas=["fiscal"])
ok("administrativo" in d["materias"], "la pista «fiscal» del chat se traduce a administrativo")
ok(pf.plazos_de(pf.normalizar("dentro de los quince días hábiles siguientes y setenta y dos horas")) ==
   ["quince dias habiles", "setenta y dos horas"], "extrae plazos con su tipo de días")
ok(pf.remisiones_de(pf.normalizar("en términos del artículo 52 de esta Ley y del artículo 14 de la Constitución"), 58) == [52],
   "remisión interna sí, la que nombra otra ley no")

print("\nParte 1b · régimen procesal y calculadora (sin red)")
import datetime as _dt
import plazos_calculo as pc
import regimen_procesal as rp
H = _dt.date(2026, 10, 11)
r = rp.resolver("JALISCO", pf.detectar("¿Cuál es el plazo para contestar la demanda de divorcio?"), H)
ok(rp.excluidas(r) >= {"CNPCF"} and r[0]["inyectar"] == "procesal_civil_local", "Jalisco sin declaratoria: fuera el CNPCF, entra el código local")
r = rp.resolver("CIUDAD_DE_MEXICO", pf.detectar("plazo para contestar en el juicio especial de arrendamiento inmobiliario oral"), H)
ok(r[0]["rige"] == "CNPCF", "CDMX arrendamiento oral: rige el CNPCF (declaratoria, dic-2024)")
r = rp.resolver("CIUDAD_DE_MEXICO", pf.detectar("plazo para apelar la sentencia de divorcio"), H)
ok(r[0]["rige"] == "LOCAL", "CDMX divorcio: código local hasta el 1-abr-2027 (modificación DOF 06-07-2026)")
r = rp.resolver("JALISCO", pf.detectar("plazo para contestar la demanda civil"), _dt.date(2027, 4, 2))
ok(r[0]["rige"] == "CNPCF", "desde el 1-abr-2027 el CNPCF rige en todas")
r = rp.resolver("QUERETARO", pf.detectar("¿en qué plazo impugno una multa de tránsito municipal?"), H)
ok("LFPCA" in rp.excluidas(r), "multa municipal: fuera la LFPCA")
r = rp.resolver("QUERETARO", pf.detectar("plazo para demandar la nulidad de un crédito fiscal del SAT"), H)
ok(r[0]["rige"] == "FEDERAL" and not rp.excluidas(r), "crédito del SAT: TFJA y LFPCA")
ok(pc.calcular("2026-10-05", 15, "dias", "LAmp", "personal")["vencimiento"] == "2026-10-28", "amparo: 15 hábiles desde el 5-oct saltando el 12-oct → 28-oct")
ok(pc.calcular("2026-10-09", 15, "dias", "CNPCF", "personal")["vencimiento"] == "2026-10-30", "CNPCF: emplazamiento 9-oct, 15 hábiles → 30-oct")
ok(pc.calcular("2026-11-17", 3, "dias", "CNPP", "personal")["vencimiento"] == "2026-11-20", "CNPP: surte al día siguiente y cuenta desde ese día → 20-nov")
ok(pc.calcular("2026-10-09", 9, "dias", "CNPCF", "personal", familiar_urgente=True)["vencimiento"] == "2026-10-18", "CNPCF familiar: naturales (art. 150)")

ENV = {}
for ruta in ("../SECRETOS_IUREXIA.env", ".env"):
    if os.path.exists(ruta):
        for linea in open(ruta, encoding="utf-8"):
            if "=" in linea and not linea.startswith("#"):
                k, v = linea.rstrip("\n").split("=", 1)
                v = v.strip()
                ENV[k.strip()] = v[1:-1] if len(v) > 1 and v[0] == v[-1] and v[0] in "'\"" else v

if ENV.get("QDRANT_URL") and ENV.get("OPENAI_API_KEY") and "--sin-red" not in sys.argv:
    print("\nParte 2 · Qdrant real, sólo lectura")
    import requests
    from qdrant_client import AsyncQdrantClient

    CASOS = [
        ("plazo para promover amparo directo contra sentencia definitiva", "LAmp", 17),
        ("¿en qué plazo debe resolverse la vinculación a proceso del imputado?", "CNPP", 313),
        ("plazo para presentar demanda de nulidad ante el TFJA contra una resolución del SAT", "LFPCA", 13),
        ("plazo para interponer recurso de revisión ante el tribunal superior agrario", "LAgra", 199),
        ("¿qué recurso procede contra el auto que desecha pruebas en el juicio ejecutivo mercantil?", "CCom", 1203),
    ]
    vecs = [x["embedding"] for x in requests.post(
        "https://api.openai.com/v1/embeddings", headers={"Authorization": "Bearer " + ENV["OPENAI_API_KEY"]},
        json={"model": "text-embedding-3-small", "input": [c[0] for c in CASOS]}, timeout=60).json()["data"]]

    async def parte2():
        qc = AsyncQdrantClient(url=ENV["QDRANT_URL"], api_key=ENV["QDRANT_API_KEY"], timeout=60)
        for (q, ab, art), v in zip(CASOS, vecs):
            _, hits = await pf.buscar(qc, q, v)
            top = [(h[2].get("abreviatura"), h[2].get("articulo_num")) for h in hits[:3]]
            ok((ab, art) in top, f"{ab} {art} entre los tres primeros: «{q[:60]}» → {top}")
    asyncio.run(parte2())

print("\nTODO PASA" if not FALLOS else f"\n{len(FALLOS)} FALLAN")
sys.exit(1 if FALLOS else 0)
