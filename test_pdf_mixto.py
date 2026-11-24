"""El PDF mixto —hojas con texto y hojas escaneadas— sin red ni Azure (23-nov-2026).

    .venv/bin/python test_pdf_mixto.py

El 23-nov un Platinum subió «DOCUMENTOS_UNIDOS_COMPLETO.pdf»: ~35 hojas de
audiencias con texto y 94 hojas escaneadas. El juez del texto nativo vio
95,669 caracteres en el documento entero, dijo «tiene texto» y el OCR no
corrió: dos tercios del archivo nunca se leyeron, y el análisis contestó que
el documento «no contiene la resolución ministerial de NEAP», que sí estaba.

Aquí se arma ese archivo en pequeño y se comprueba que cada hoja escaneada se
lee —sólo esas, con el parámetro `pages` de Azure— y vuelve a su lugar; que
lo que ya funcionaba (todo texto, todo escaneado) no cambia; y que si el OCR
falla el abogado recibe por lo menos lo de antes. Azure y el almacén son
falsos y cuentan las llamadas, que es lo que cuesta.
"""
import asyncio
import random
import re
import sys

import fitz

import cache_ocr as co

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


class CuboFalso:
    def __init__(self):
        self.objetos = {}
        self.subidas = 0

    def download(self, ruta):
        if ruta not in self.objetos:
            raise Exception("{'statusCode': 400, 'error': 'not_found', 'message': 'Object not found'}")
        return self.objetos[ruta]

    def upload(self, ruta, datos, opciones):
        self.subidas += 1
        self.objetos[ruta] = datos

    def list(self, prefijo, opciones=None):
        return []

    def remove(self, rutas):
        pass


class ClienteFalso:
    def __init__(self):
        self.cubo = CuboFalso()
        self.storage = self

    def from_(self, nombre):
        return self.cubo


async def _esperar_subidas():
    while co._TAREAS:
        await asyncio.gather(*list(co._TAREAS), return_exceptions=True)


PARRAFO = ("En la audiencia inicial el agente del Ministerio Público formuló imputación "
           "por el delito de fraude previsto en el artículo 230 del Código Penal, y la "
           "defensa ofreció los estados de cuenta y los pagarés suscritos por el deudor. ")
ESCANEADA = "RESOLUCION DE NO EJERCICIO DE LA ACCION PENAL hoja {n} " * 8
# Un escrito de verdad no repite el mismo párrafo: con uno solo, el juez lo
# toma —con razón— por un sello. Cada hoja de texto lleva palabras propias.
_azar = random.Random(11)
_VOCAB = ["".join(_azar.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(_azar.randint(4, 9)))
          for _ in range(3000)]


def _pagina_texto(d, n):
    p = d.new_page()
    relleno = " ".join(_azar.choice(_VOCAB) for _ in range(90))
    p.insert_textbox(fitz.Rect(40, 40, 560, 800), f"Hoja {n}. " + PARRAFO + relleno, fontsize=8)


def _pagina_escaneada(d, n, sello=""):
    """Una hoja que es FOTO: se dibuja texto en otra página, se vuelve imagen y
    se pega sin capa de texto. `sello`, si va, es el texto de la FIREL encima."""
    tmp = fitz.open()
    t = tmp.new_page()
    t.insert_textbox(fitz.Rect(40, 40, 560, 800), ESCANEADA.format(n=n), fontsize=12)
    pix = t.get_pixmap(dpi=40)
    tmp.close()
    p = d.new_page()
    p.insert_image(p.rect, pixmap=pix)
    if sello:
        p.insert_text((40, 820), sello, fontsize=6)


def _pdf(plan: str) -> bytes:
    """plan: T = hoja con texto, E = escaneada, S = escaneada con sello, B = en blanco."""
    d = fitz.open()
    for n, c in enumerate(plan, 1):
        if c == "T":
            _pagina_texto(d, n)
        elif c == "E":
            _pagina_escaneada(d, n)
        elif c == "S":
            _pagina_escaneada(d, n, sello="FIRMANTE DEL JUZGADO 706a6620636a6632 15/05/26")
        else:
            d.new_page()
    datos = d.tobytes()
    d.close()
    return datos


print("\n1 · LAS PIEZAS")
import main

ok(main._rango_azure([1, 2, 3, 7, 9, 10]) == "1-3,7,9-10", "el rango de Azure se comprime")
ok(main._rango_azure([5]) == "5" and main._rango_azure([]) == "", "una sola hoja, y ninguna")

pdf = _pdf("TTEESTBT")          # 8 hojas: 2 texto, 2 escaneadas, 1 con sello, 1 texto, 1 blanco, 1 texto
textos, con_imagen = main._leer_pdf_paginas(pdf)
faltan = main._hojas_sin_texto(textos, con_imagen)
ok(len(textos) == 8, "PyMuPDF ve las 8 hojas")
ok(faltan == [3, 4, 5], f"faltan las dos escaneadas y la del sello, no la blanca (faltan={faltan})")
_todo = "\n\n".join(textos).strip()
ok(main._texto_nativo_sirve(_todo, 8),
   "el juez del documento entero dice «tiene texto»: así se colaba el mixto")

print("\n2 · LA RIQUEZA POR TRAMOS (el documento largo ya no pasa por sello)")
random.seed(7)
# Como el español: los nexos («que», «los», «por»…) son lo más frecuente y
# detrás va una cola larga de palabras propias (ley de Zipf).
_nexos = ["que", "los", "del", "las", "por", "con", "una", "para", "como", "sus", "esta", "sin"]
vocab = _nexos + ["".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(random.randint(4, 10)))
                  for _ in range(12000)]
pesos = [1 / (i + 1) for i in range(len(vocab))]
largo = random.choices(vocab, weights=pesos, k=90000)       # 90 mil palabras, como un código
_entera = len({w for w in largo}) / len(largo)
ok(_entera < 0.12, f"medido entero, un texto largo y bueno baja de 0.12 ({_entera:.3f})")
ok(main._riqueza_de_palabras(largo) > 0.2, f"por tramos no ({main._riqueza_de_palabras(largo):.3f})")
sello = ("FIRMANTE DEL JUZGADO FIRMA ELECTRONICA CERTIFICADO " * 400).split()
ok(main._riqueza_de_palabras(sello) < 0.05, "un sello repetido sigue en el suelo")
ok(main._texto_nativo_sirve(" ".join(largo), 240), "el código de 240 páginas con texto ya no va al OCR")
_roto = random.choices([w for w in vocab if w not in _nexos], weights=pesos[len(_nexos):], k=90000)
ok(not main._texto_nativo_sirve(" ".join(_roto), 240),
   "una capa rota igual de larga (sin nexos del español) sigue yendo al OCR")
_sello_firel = "NOMBRE DE LA FIRMANTE DEL JUZGADO\n706a6620636a663200000000000000000001d0ad\n15/05/26 18:00:00\n" * 22
ok(not main._texto_nativo_sirve(_sello_firel, 22), "el sello de la FIREL ×22 sigue sin pasar (563-2025)")


async def cableado():
    print("\n3 · CABLEADO EN _extract_text_from_upload (Azure falso)")
    main.supabase_admin = ClienteFalso()
    main.AZURE_DOCINT_KEY = "llave-falsa"
    main.AZURE_DOCINT_ENDPOINT = "https://falso.invalid"
    llamadas = []

    def _leido(n):
        return f"RESOLUCION DE NO EJERCICIO DE LA ACCION PENAL, hoja {n}, leída por OCR. " * 6

    async def azure_falso(contenido, paginas_esperadas=0, rango="", por_pagina=False):
        llamadas.append({"rango": rango, "por_pagina": por_pagina, "esperadas": paginas_esperadas})
        if por_pagina:
            hojas = set()
            for tramo in rango.split(","):
                a, _, b = tramo.partition("-")
                hojas.update(range(int(a), int(b or a) + 1))
            return {n: _leido(n) for n in sorted(hojas)}
        return "\n\n".join(f"[[PÁGINA {n}]]\n{_leido(n)}" for n in range(1, (paginas_esperadas or 1) + 1))

    main._ocr_azure = azure_falso

    t = await main._extract_text_from_upload(main._SubidaDeBytes(pdf, "DOCUMENTOS_UNIDOS.pdf"))
    await _esperar_subidas()
    ok(len(llamadas) == 1 and llamadas[0]["rango"] == "3-5" and llamadas[0]["por_pagina"],
       f"Azure se paga UNA vez y sólo por las hojas 3-5 ({llamadas})")
    pags = re.findall(r"\[\[PÁGINA (\d+)\]\]", t)
    ok(pags == ["1", "2", "3", "4", "5", "6", "8"], f"cada hoja en su lugar y la blanca fuera ({pags})")
    ok("hoja 3, leída por OCR" in t and "hoja 5, leída por OCR" in t, "las escaneadas traen su texto")
    ok(all(f"Hoja {n}." in t for n in (1, 2, 6, 8)) and t.count("Ministerio Público") == 4,
       "las de texto conservan el suyo")
    ok(t.index("hoja 3, leída") < t.index("Hoja 6."), "en el orden del archivo")
    ok("706a6620636a6632" not in t, "en la hoja del sello manda lo leído, no el sello")
    _partes = await main._paginas_de_bytes(pdf, "DOCUMENTOS_UNIDOS.pdf")
    ok(len(_partes) == 7, f"la depuración lo parte por hojas ({len(_partes)})")

    llamadas.clear()
    t2 = await main._extract_text_from_upload(main._SubidaDeBytes(pdf, "otro-nombre.pdf"))
    ok(t2 == t and not llamadas, "el mismo archivo otra vez: de la caché, sin pagar Azure")

    print("\n4 · LO QUE YA FUNCIONABA NO CAMBIA")
    llamadas.clear()
    solo_texto = _pdf("TTTTB")
    t3 = await main._extract_text_from_upload(main._SubidaDeBytes(solo_texto, "escrito.pdf"))
    ok(not llamadas and "[[PÁGINA" not in t3 and "Hoja 4." in t3,
       "todo texto (con una hoja en blanco): capa nativa, sin OCR ni rótulos")
    llamadas.clear()
    solo_escaneo = _pdf("EEEE")
    t4 = await main._extract_text_from_upload(main._SubidaDeBytes(solo_escaneo, "escaneo.pdf"))
    ok(len(llamadas) == 1 and not llamadas[0]["por_pagina"] and llamadas[0]["esperadas"] == 4,
       "todo escaneado: el OCR del documento entero, como siempre")
    ok("hoja 4, leída por OCR" in t4, "y trae sus cuatro hojas")

    print("\n5 · SI EL OCR NO SE PUEDE, POR LO MENOS LO DE ANTES")
    llamadas.clear()
    otro = _pdf("TTETT")

    async def azure_caido(contenido, paginas_esperadas=0, rango="", por_pagina=False):
        llamadas.append(rango)
        raise RuntimeError("Azure no responde")

    main._ocr_azure = azure_caido
    t5 = await main._extract_text_from_upload(main._SubidaDeBytes(otro, "falla.pdf"))
    await _esperar_subidas()
    ok(llamadas == ["3"] and "Hoja 5." in t5, "Azure caído: la capa de texto, sin tumbar la lectura")
    ok(co.ruta(co.huella(otro)) not in main.supabase_admin.cubo.objetos,
       "y lo leído a medias no entra en la caché")

    main._ocr_azure = azure_falso
    llamadas.clear()
    t6 = await main._extract_text_from_upload(main._SubidaDeBytes(_pdf("TTEET"), "estudio.pdf"),
                                              permitir_ocr=False)
    ok(not llamadas and "Hoja 5." in t6, "Estudio con el freno: sin OCR pagado, con la capa de texto")


asyncio.run(cableado())

print()
print("TODO PASA" if not FALLOS else f"{len(FALLOS)} FALLAS")
sys.exit(1 if FALLOS else 0)
