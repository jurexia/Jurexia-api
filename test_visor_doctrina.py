"""La doctrina en el visor, como las demás fuentes — 25-sep-2026.

    .venv/bin/python test_visor_doctrina.py

David, con captura: «En todas estas nuevas el visor no está disponible,
tenemos que implementarlo como todas nuestras fuentes». La doctrina viajaba con
`pdf_url` = url_oficial + «#page=N»: el proxy /api/ley/pdf rechazaba esa
dirección y el visor decía «No se pudo abrir el PDF aquí». El contrato nuevo es
el de la Corte IDH: `pdf_url` = `url_oficial` SIN «#page», `pagina` (la del PDF
del capítulo, `pagina_pdf`), `pagina_impresa`, `ancla` (~15 palabras literales
del cuerpo, sin el folio ni el título corrido), `obra`, `autor`, `anio`.

Sin red y sin gastar API: Qdrant es falso. Los textos de prueba tienen la
FORMA exacta de los trozos reales —leídos de la colección `doctrina` el
25-sep-2026: folio y título corrido arriba en el Diccionario, título corrido
ENCIMA del folio en la Panorámica, sílabas separadas en Carbonell, páginas
largas partidas a media línea— y se recortan a unas pocas palabras.
"""
import asyncio
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.getcwd())
import main  # noqa: E402
import doctrina as dm  # noqa: E402

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


def correr(coro):
    return asyncio.run(coro)


# ═══════════════════════════════════════════════════════════════ fixtures

UNAM = "https://archivos.juridicas.unam.mx/www/bjv/libros/8/3632/11.pdf"
ID_DICC = "0005a968-a5c5-52fc-a6d9-daaf7ec9ef67"
ID_PANO = "001229b2-4031-5baf-8658-e5b5f53352e4"
ID_CARB = "0008e8b0-38a6-5902-bfb7-66b0eb7d41d2"
ID_LEY = "11111111-2222-3333-4444-555555555555"

PL_DICC = {   # Diccionario, t. I: folio y título corrido ARRIBA, debajo del folio
    "autor": "Ferrer Mac-Gregor, Martínez Ramírez y Figueroa Mejía (coords.)",
    "obra": "Diccionario de derecho procesal constitucional y convencional, t. I",
    "anio": 2014, "editorial": "UNAM, Instituto de Investigaciones Jurídicas",
    "subtipo": "doctrina", "capitulo_pdf": "011", "pagina_pdf": 313, "pagina_impresa": 280,
    # La URL con «#page» pegado, a propósito: nada de eso puede llegar al visor.
    "url_oficial": UNAM + "#page=313",
    "texto": ("280\nCreación de derechos por el juez constitucional\n"
              "Esta última modalidad, íntimamente relacionada con el proceso de ju-\n"
              "dicialización de las demandas por derechos humanos, de su justiciabilidad \n"
              "(Abramovich y Pautassi, 2009)"),
}
PL_PANO = {   # Panorámica, página impar: capítulo ENCIMA del folio
    "autor": "Eduardo Ferrer Mac-Gregor",
    "obra": "Panorámica del derecho procesal constitucional y convencional",
    "anio": 2017, "editorial": "UNAM-Marcial Pons", "subtipo": "doctrina",
    "capitulo_pdf": "011", "pagina_pdf": 13, "pagina_impresa": 151,
    "url_oficial": "https://archivos.juridicas.unam.mx/www/bjv/libros/7/3384/11.pdf",
    "texto": ("VII.    Mauro Cappelletti y el Derecho Procesal Constitucional ComparadO\n151\n"
              "aquella suerte de irónico olvido que es el destino de las leyes que no se aplican; \n"
              "un método para dar al individuo"),
}
PL_CARB = {   # Carbonell: sin folio arriba, sílabas separadas, sin página impresa
    "autor": "Miguel Carbonell", "obra": "Los derechos fundamentales en México", "anio": 2004,
    "subtipo": "doctrina", "capitulo_pdf": "006", "pagina_pdf": 106, "pagina_impresa": None,
    "url_oficial": "https://archivos.juridicas.unam.mx/www/bjv/libros/3/1408/6.pdf",
    "texto": "la co mu ni -\nca ción de ma sas; de for ma que po dría mos sa ber, en prin ci pio, que la in -\nfor ma ción",
}
PL_LEY = {"texto": "Artículo 1o. En los Estados Unidos Mexicanos…", "ref": "Art. 1",
          "origen": "Constitución Política de los Estados Unidos Mexicanos"}


class ColeccionAusente(Exception):
    """Lo que hace Qdrant con una colección que no existe (404)."""


class QdrantFalso:
    """Sólo lectura, como el de verdad para esta prueba: retrieve y query_points."""

    def __init__(self, colecciones):
        self.col = colecciones
        self.consultas = []

    async def retrieve(self, collection_name, ids, with_payload=True, **kw):
        if collection_name not in self.col:
            raise ColeccionAusente(collection_name)
        quiero = {str(i).lower() for i in ids}
        return [p for p in self.col[collection_name] if str(p.id).lower() in quiero]

    async def query_points(self, collection_name, query=None, using=None, limit=3, query_filter=None,
                           with_payload=True, **kw):
        self.consultas.append((collection_name, using, limit))
        if collection_name not in self.col:
            raise ColeccionAusente(collection_name)
        return SimpleNamespace(points=[SimpleNamespace(id=p.id, score=0.71, payload=p.payload)
                                      for p in self.col[collection_name][:limit]])


QD = QdrantFalso({
    "doctrina": [SimpleNamespace(id=ID_DICC, payload=PL_DICC), SimpleNamespace(id=ID_PANO, payload=PL_PANO),
                 SimpleNamespace(id=ID_CARB, payload=PL_CARB)],
    "bloque_constitucional": [SimpleNamespace(id=ID_LEY, payload=PL_LEY)],
})
main.qdrant_client = QD


# ═══════════════════════════════════════════════════════════════ 1 · el ancla
print("\n1 · EL ANCLA: LAS PRIMERAS PALABRAS LITERALES DEL CUERPO, SIN FOLIO NI TÍTULO CORRIDO")
a = dm.ancla(PL_DICC["texto"])
ok(a.startswith("Esta última modalidad, íntimamente") and "280" not in a and "Creación" not in a,
   f"Diccionario: fuera el folio «280» y el título corrido ({a!r})")
ok("ju- dicialización" in a, "literal: el corte de renglón queda como en el PDF («ju- dicialización»), no «arreglado»")
ok(len(a.split()) == dm.PALABRAS_ANCLA == 15, f"quince palabras ({len(a.split())})")
a = dm.ancla(PL_PANO["texto"])
ok(a.startswith("aquella suerte de irónico olvido"), f"Panorámica impar: fuera el capítulo y el folio ({a!r})")
a = dm.ancla("eduardo ferrer mac-gregor\x08\npanorámica del derecho procesal...\n200\n"
             "artículo 102 de la Constitución Federal de 1857, que reconoció definitivamente ")
ok(a.startswith("artículo 102 de la Constitución") and "\x08" not in a and " " not in a,
   f"Panorámica par: fuera el autor (con su letra de control), el título recortado y el folio ({a!r})")
a = dm.ancla("xxiv.    La democracia y el juez constitucional\n641\n"
             "A)    \x07El fallo parcialmente condenatorio\nSin entrar en detalles, la Suprema Corte")
ok(a.startswith("A) El fallo parcialmente condenatorio"),
   f"con el título corrido ENCIMA del folio, el epígrafe de debajo es texto y se queda ({a!r})")
a = dm.ancla("aberse sometido a sí mismos al peso de la\nLAS RAZONES DEL DERECHO\n153\n"
             "10 En Habermas, el concepto de acción comunicativa")
ok(a.startswith("aberse sometido a sí mismos"),
   f"media página (Atienza): el renglón de encima del folio es cuerpo, no se corta hacia las notas ({a!r})")
a = dm.ancla(PL_CARB["texto"])
ok(a.startswith("la co mu ni ca ción de ma sas;"),
   f"Carbonell: sin folio no se corta nada; el guion suelto no cuenta como palabra ({a!r})")
a = dm.ancla("Contenido\n" + ". " * 40 + "\nDerechos políticos 566")
ok(a == "Contenido Derechos políticos 566", f"los puntos guía de un índice no se comen el ancla ({a!r})")
a = dm.ancla("577\nComo puede observarse, existe un contraste absoluto entre la doctrina de \nLocke y la de Rousseau")
ok(a.startswith("Como puede observarse"), "un renglón de cuerpo debajo del folio (con espacio al final) no se toma por título")
a = dm.ancla("490\nd) El Poder Ejecutivo y el Poder Legislativo del Estado.\nIII. Conocer y resolver")
ok(a.startswith("d) El Poder Ejecutivo"), "ni uno que termina en punto")
ok(dm.ancla("") is None and dm.ancla("280\n") is None, "sin texto, sin ancla (None, no una cadena vacía)")
ok(dm.ancla("12\nUn trozo de un solo renglón") == "Un trozo de un solo renglón",
   "un trozo de una sola línea tras el folio no se vacía")


# ═══════════════════════════════════════════════════════════════ 2 · contrato
print("\n2 · EL CONTRATO: URL SIN #page, PÁGINA DEL PDF Y ANCLA APARTE")
f = dm.fragmento(ID_DICC, PL_DICC, 0.71)
c = dm.contrato(f, 0.70)
ok(c["pdf_url"] == UNAM and c["url_oficial"] == UNAM and "#" not in c["pdf_url"],
   "pdf_url = url_oficial, sin «#page» aunque el payload lo trajera")
ok(c["pagina"] == 313 and isinstance(c["pagina"], int) and c["pagina_impresa"] == 280,
   "pagina = pagina_pdf (313, entero), pagina_impresa = 280")
ok(c["ancla"] == dm.ancla(PL_DICC["texto"]) and c["obra"] == PL_DICC["obra"] and c["autor"] == PL_DICC["autor"]
   and c["anio"] == 2014, "ancla, obra, autor y año")
ok(c["origen"] == f"{PL_DICC['autor']}, «{PL_DICC['obra']}», 2014" and c["ref"] == "p. 280"
   and c["silo"] == "doctrina" and c["jurisdiccion"] == "Doctrina" and c["score"] == 0.70,
   "el rótulo de siempre: «Autor, «Obra», año» y «p. 280» (la impresa)")
sr = main.SearchResult(**c)
ok(sr.pagina == 313 and sr.ancla and sr.obra and sr.autor and sr.anio == 2014 and sr.pagina_impresa == 280,
   "SearchResult acepta los campos nuevos (opcionales)")
cc = dm.contrato(dm.fragmento(ID_CARB, PL_CARB), 0.70)
ok(cc["pagina_impresa"] is None and cc["pagina"] == 106 and cc["ref"] == "",
   "sin página impresa: el visor abre la 106 del PDF, pero no se rotula «p. 106» (el folio impreso es otro)")
_bloque = dm.bloque_para_prompt([dm.fragmento(ID_CARB, PL_CARB)])
ok("p. 106" not in _bloque and "no la pongas ni la inventes" in _bloque,
   "y al modelo no se le da la página del PDF como si fuera la del libro")
ok(main.SearchResult(id="x", score=1, texto="t", silo="leyes_federales").obra is None,
   "en las demás fuentes los campos nuevos quedan en None")


# ═══════════════════════════════════════════════════════════════ 3 · buscar
print("\n3 · doctrina.buscar TRAE EL ANCLA Y LA PÁGINA DEL PDF")
frags = correr(dm.buscar(QD, [0.0] * 8, "¿qué es la creación judicial de derechos?"))
ok(len(frags) == 3 and all(fr_.get("ancla") and fr_.get("pagina_pdf") for fr_ in frags)
   and all("#" not in fr_["url_oficial"] for fr_ in frags),
   f"los fragmentos traen ancla, pagina_pdf y la URL sin #page ({len(frags)})")
ok(frags[0]["pagina"] == 280 and frags[2]["pagina"] is None and frags[2]["pagina_pdf"] == 106,
   "`pagina` del fragmento es la impresa que se ROTULA; sin ella, nada (la del PDF va aparte)")
ok(QD.consultas and QD.consultas[-1][0] == "doctrina", "consultó la colección `doctrina` (sólo lectura)")


# ═══════════════════════════════════════════════════════════════ 4 · marcadores
print("\n4 · FUENTES_PREVIAS Y CITATION_META LLEVAN EL CONTRATO")
s_dicc = main.SearchResult(**dm.contrato(dm.fragmento(ID_DICC, PL_DICC), 0.70))
s_ley = main.SearchResult(id=ID_LEY, score=0.9, texto=PL_LEY["texto"], ref="Art. 1",
                          origen=PL_LEY["origen"], silo="bloque_constitucional")
marc = main._marcador_fuentes_previas([s_dicc, s_ley])
dat = json.loads(marc[marc.index(":") + 1:marc.rindex("-->")])
e, el = dat[ID_DICC], dat[ID_LEY]
ok(e["silo"] == "doctrina" and e["pdf_url"] == UNAM and e["url_oficial"] == UNAM and e["pagina"] == 313
   and e["pagina_impresa"] == 280 and e["ancla"] and e["obra"] and e["autor"] and e["anio"] == 2014,
   "FUENTES_PREVIAS: pdf_url sin #page, pagina, pagina_impresa, ancla, obra, autor, año")
ok(not any(k in el for k in ("pagina", "pagina_impresa", "ancla", "obra", "autor", "anio", "url_oficial")),
   "y ni una clave nueva en la fuente de una ley")
ok(main._campos_doctrina(s_ley) == {} and main._campos_coidh(s_dicc) == {},
   "cada contrato sólo para su silo")
# Revisión (25-sep-2026): una fuente de doctrina armada a la antigua —«#page»
# pegado y sin `pagina`— no pierde la página al quitarle el «#page».
s_vieja = main.SearchResult(id="v", score=0.7, texto="t", silo="doctrina", pdf_url=UNAM + "#page=313")
cv = main._campos_doctrina(s_vieja)
ok(cv["pdf_url"] == UNAM and cv["url_oficial"] == UNAM and cv["pagina"] == 313,
   "una fuente vieja con «#page=313»: la URL sin #page y la página pasa a `pagina`")
texto = f"La creación judicial de derechos (Ansolabehere, p. 280) [Doc ID: {ID_DICC}]."
salida = main._marcadores_del_sello(texto, main.build_doc_id_map([s_dicc, s_ley]), [s_dicc, s_ley])
meta = next(x for x in salida if "CITATION_META" in x)
src = json.loads(meta[meta.index(":") + 1:meta.rindex("-->")])["sources"][ID_DICC]
ok(src["pdf_url"] == UNAM and "#" not in src["pdf_url"] and src["pagina"] == 313 and src["ancla"]
   and src["url_oficial"] == UNAM and src["obra"] == PL_DICC["obra"],
   "CITATION_META: la fuente doctrinal citada, con el contrato")


# ═══════════════════════════════════════════════════════════════ 5 · /cita y turnos siguientes
print("\n5 · /cita Y LAS FUENTES YA VERIFICADAS (LOS TURNOS SIGUIENTES)")
cita = correr(main.resolver_cita(ID_DICC))
ok(cita["silo"] == "doctrina" and cita["pdf_url"] == UNAM and cita["url_oficial"] == UNAM
   and cita["pagina"] == 313 and cita["pagina_impresa"] == 280 and cita["ancla"] == dm.ancla(PL_DICC["texto"])
   and cita["obra"] == PL_DICC["obra"] and cita["autor"] == PL_DICC["autor"] and cita["anio"] == 2014,
   "/cita/{doc_id}: el contrato completo, sin #page")
ok(cita["origen"] == f"{PL_DICC['autor']} — {PL_DICC['obra']}" and cita["ref"] == f"{PL_DICC['obra']}, p. 280"
   and cita["tipo_criterio"] == "doctrina" and cita["instancia"] == PL_DICC["autor"],
   "y el rótulo de /cita de siempre («autor — obra», «obra, p. 280»)")
ver = correr(main._fuentes_ya_verificadas([ID_DICC, ID_PANO, ID_LEY]))
d_ver = {v.id: v for v in ver}
vd = d_ver.get(ID_DICC)
ok(vd is not None and vd.silo == "doctrina" and vd.score == 2.0 and vd.pdf_url == UNAM and vd.pagina == 313
   and vd.ancla and vd.origen.startswith(PL_DICC["autor"]) and vd.ref == "p. 280",
   "turno siguiente: la doctrina vuelve con origen, referencia y contrato (antes volvía sin rótulo ni página)")
vp = d_ver.get(ID_PANO)
ok(vp is not None and vp.pagina == 13 and vp.ancla.startswith("aquella suerte"), "también la Panorámica")
ok(d_ver.get(ID_LEY) is not None and d_ver[ID_LEY].silo == "bloque_constitucional" and d_ver[ID_LEY].pagina is None,
   "y la ley sigue su ruta de siempre")
m2 = main._marcador_fuentes_previas(ver)
d2 = json.loads(m2[m2.index(":") + 1:m2.rindex("-->")])
ok(d2[ID_DICC]["pagina"] == 313 and d2[ID_DICC]["ancla"] and "#" not in d2[ID_DICC]["pdf_url"],
   "y el FUENTES_PREVIAS del turno siguiente vuelve a llevar página y ancla")


# ═══════════════════════════════════════════════════════════════ 6 · el chat
print("\n6 · EL CHAT ARMA LA DOCTRINA CON EL CONTRATO")
fuente = open("main.py", encoding="utf-8").read()
gs = fuente[fuente.index("async def generate_stream("):]
i_armado = gs.index("SearchResult(**_doctrina_mod.contrato(_f, 0.70))")
i_import = gs.rfind("import doctrina as _doctrina_mod", 0, i_armado)
ok(i_import != -1 and "for _f in (_frags or []):" in gs[i_import:i_armado],
   "generate_stream importa `doctrina` ANTES de usarlo (la tarjeta lo importa más abajo y lo vuelve local)")
bloque = gs[gs.index("if _doctrina_task is not None:"):i_armado + 200]
ok("#page=" not in bloque, "ni un «#page=» en el armado de la doctrina")
ok("#page=" not in dm.contrato(dm.fragmento("x", dict(PL_DICC, url_oficial=UNAM + "#page=9")), 0.7)["pdf_url"],
   "ni aunque la URL de la ingesta lo trajera")


print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}:")
    for f_ in FALLOS:
        print("  ·", f_)
    sys.exit(1)
print("TODO PASA")
