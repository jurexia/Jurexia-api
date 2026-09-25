"""La caché del OCR, sin red ni Azure — 24-sep-2026.

    .venv/bin/python test_cache_ocr.py

Lo que importa no es que se guarde un fichero: es que el MISMO PDF no vuelva
a pagarse, que un documento leído a medias no se congele, y que si el almacén
se cae la lectura siga como antes. Azure y el cubo son falsos: cuentan las
llamadas, que es lo que cuesta dinero.
"""
import asyncio
import sys
import time

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
        self.caido = False
        self.subidas = 0

    def download(self, ruta):
        if self.caido:
            raise RuntimeError("el almacén no responde")
        if ruta not in self.objetos:
            raise Exception("{'statusCode': 400, 'error': 'not_found', 'message': 'Object not found'}")
        return self.objetos[ruta]["datos"]

    def upload(self, ruta, datos, opciones):
        if self.caido:
            raise RuntimeError("el almacén no responde")
        self.subidas += 1
        self.objetos[ruta] = {"datos": datos, "tipo": opciones.get("content-type"),
                              "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())}

    def list(self, prefijo, opciones):
        filas = [{"name": r[len(prefijo) + 1:], "updated_at": o["updated_at"]}
                 for r, o in self.objetos.items() if r.startswith(prefijo + "/")]
        return sorted(filas, key=lambda f: f["updated_at"])

    def remove(self, rutas):
        for r in rutas:
            self.objetos.pop(r, None)


class ClienteFalso:
    def __init__(self):
        self.cubo = CuboFalso()
        self.storage = self

    def from_(self, nombre):
        assert nombre == "expedientes", nombre
        return self.cubo


async def _esperar_subidas():
    while co._TAREAS:
        await asyncio.gather(*list(co._TAREAS), return_exceptions=True)


def _pdf_escaneado(paginas: int) -> bytes:
    """Páginas sin capa de texto: lo que llega de un escáner."""
    d = fitz.open()
    for _ in range(paginas):
        d.new_page()
    datos = d.tobytes()
    d.close()
    return datos


print("\n1 · LA HUELLA Y LA FICHA")
ok(co.huella(b"abc") == co.huella(b"abc"), "el mismo PDF da la misma huella")
ok(co.huella(b"abc") != co.huella(b"abd"), "un byte distinto, otra huella")
ahora = 1_790_000_000.0
texto = "[[PÁGINA 1]]\n" + "El quejoso promovió juicio de amparo directo. " * 200
f = co.desempaquetar(co.empaquetar(texto, 3, ahora=ahora), ahora=ahora + 60)
ok(f is not None and f["texto"] == texto and f["paginas"] == 3, "ida y vuelta sin perder un carácter")
ok(len(co.empaquetar(texto, 3)) < len(texto.encode()) / 5, "comprimido ocupa menos de la quinta parte")
ok(co.desempaquetar(co.empaquetar(texto, 3, ahora=ahora), ahora=ahora + 13 * 86400, dias=14) is not None,
   "a los 13 días todavía vale")
ok(co.desempaquetar(co.empaquetar(texto, 3, ahora=ahora), ahora=ahora + 15 * 86400, dias=14) is None,
   "a los 15 días está caducada")
ok(co.desempaquetar(co.empaquetar(texto, 3, ahora=ahora + 7200), ahora=ahora) is None,
   "una fecha del futuro no pasa por fresca")
ok(co.desempaquetar(b"basura") is None, "datos rotos → None, sin lanzar")
ok(co.desempaquetar(co.empaquetar("   ", 1)) is None, "texto vacío no cuenta como lectura")
_motor = co.MOTOR
co.MOTOR = "azure-prebuilt-read@2099-01-01"
ok(co.desempaquetar(co.empaquetar(texto, 3, ahora=ahora), ahora=ahora) is not None, "mismo motor, vale")
_otro = co.empaquetar(texto, 3, ahora=ahora)
co.MOTOR = _motor
ok(co.desempaquetar(_otro, ahora=ahora) is None, "lo leído con otro motor deja de valer solo")


async def parte_almacen():
    print("\n2 · EL ALMACÉN")
    c = ClienteFalso()
    h = co.huella(b"pdf-uno")
    ok(await co.leer(c, h) is None, "lo que nunca se guardó → None, sin lanzar")
    co.guardar_en_segundo_plano(c, h, texto, 3)
    await _esperar_subidas()
    ok(c.cubo.subidas == 1 and co.ruta(h) in c.cubo.objetos, "se guarda en ocr-cache/v1/<huella>.json.gz")
    ok(c.cubo.objetos[co.ruta(h)]["tipo"] == "application/gzip", "con su tipo")
    ok(await co.leer(c, h) == texto, "y se lee de vuelta idéntica")
    ok(await co.leer(None, h) is None, "sin cliente de Supabase no pasa nada")
    c.cubo.caido = True
    ok(await co.leer(c, h) is None, "almacén caído al leer → None, sin lanzar")
    co.guardar_en_segundo_plano(c, co.huella(b"pdf-dos"), texto, 3)
    await _esperar_subidas()
    ok(True, "almacén caído al guardar → sin lanzar")

    print("\n3 · LA PURGA")
    c = ClienteFalso()
    for nombre, edad in (("a.json.gz", 20), ("b.json.gz", 15), ("c.json.gz", 2)):
        c.cubo.objetos[f"{co.PREFIJO}/{nombre}"] = {
            "datos": b"", "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                        time.gmtime(time.time() - edad * 86400))}
    c.cubo.objetos[f"{co.PREFIJO}/.emptyFolderPlaceholder"] = {"datos": b"", "updated_at": "2020-01-01T00:00:00Z"}
    n = co.purgar(c, dias=14)
    quedan = sorted(r.rsplit("/", 1)[1] for r in c.cubo.objetos)
    ok(n == 2 and quedan == [".emptyFolderPlaceholder", "c.json.gz"],
       f"borra las dos caducadas y respeta la nueva y lo ajeno (quedan {quedan})")


asyncio.run(parte_almacen())


async def parte_main():
    print("\n4 · CABLEADO EN _extract_text_from_upload (Azure falso)")
    import main
    cliente = ClienteFalso()
    main.supabase_admin = cliente
    main.AZURE_DOCINT_KEY = "llave-falsa"
    main.AZURE_DOCINT_ENDPOINT = "https://falso.invalid"
    llamadas = []
    leido = "[[PÁGINA 1]]\n" + "Texto que Azure leyó del escaneo. " * 40

    async def azure_falso(contenido, paginas_esperadas=0, rango=""):
        llamadas.append((paginas_esperadas, rango))
        return leido

    main._ocr_azure = azure_falso
    pdf = _pdf_escaneado(3)
    t1 = await main._extract_text_from_upload(main._SubidaDeBytes(pdf, "escaneo.pdf"))
    await _esperar_subidas()
    ok(t1 == leido and len(llamadas) == 1, "primera vez: Azure lee y se paga una vez")
    t2 = await main._extract_text_from_upload(main._SubidaDeBytes(pdf, "otro-nombre.pdf"))
    ok(t2 == leido and len(llamadas) == 1, "segunda vez, aunque cambie el nombre: de caché, Azure no se toca")
    await main._extract_text_from_upload(main._SubidaDeBytes(_pdf_escaneado(4), "distinto.pdf"))
    await _esperar_subidas()
    ok(len(llamadas) == 2, "otro PDF sí se lee")

    print("\n5 · LOS TRAMOS: SÓLO LO LEÍDO ENTERO SE GUARDA")
    llamadas.clear()
    tramo_vacio = {"si": False}

    async def azure_que_topa(contenido, paginas_esperadas=0, rango=""):
        llamadas.append((paginas_esperadas, rango))
        if not rango:
            raise main._OCRIncompleto("Azure leyó 2 de 5", leidas=2, texto=leido)
        return "" if tramo_vacio["si"] else "[[PÁGINA 1]]\nlo que faltaba del expediente. " * 5

    main._ocr_azure = azure_que_topa
    grande = _pdf_escaneado(5)
    t = await main._extract_text_from_upload(main._SubidaDeBytes(grande, "grande.pdf"))
    await _esperar_subidas()
    ok("lo que faltaba" in t and len(llamadas) == 2, "completo por tramos: entero y en dos llamadas")
    llamadas.clear()
    await main._extract_text_from_upload(main._SubidaDeBytes(grande, "grande.pdf"))
    ok(len(llamadas) == 0, "y la segunda vez sale de caché")

    tramo_vacio["si"] = True
    llamadas.clear()
    grande2 = _pdf_escaneado(6)
    t = await main._extract_text_from_upload(main._SubidaDeBytes(grande2, "grande2.pdf"))
    await _esperar_subidas()
    ok(t.startswith("[[PÁGINA 1]]") and co.ruta(co.huella(grande2)) not in cliente.cubo.objetos,
       "un tramo que vuelve vacío: se entrega lo leído pero NO se guarda")
    llamadas.clear()
    await main._extract_text_from_upload(main._SubidaDeBytes(grande2, "grande2.pdf"))
    ok(len(llamadas) >= 1, "y la próxima vez Azure lo vuelve a intentar")

    print("\n6 · SI AZURE FALLA Y RESPONDE GEMINI, NO SE GUARDA")
    llamadas.clear()

    async def azure_caido(contenido, paginas_esperadas=0, rango=""):
        llamadas.append(1)
        raise RuntimeError("Azure no terminó")

    def gemini_caido():
        raise RuntimeError("sin Gemini en la prueba")

    main._ocr_azure = azure_caido
    main.get_gemini_client = gemini_caido
    pdf7 = _pdf_escaneado(7)
    await main._extract_text_from_upload(main._SubidaDeBytes(pdf7, "falla.pdf"))
    await _esperar_subidas()
    ok(co.ruta(co.huella(pdf7)) not in cliente.cubo.objetos, "nada guardado cuando Azure no leyó")

    print("\n7 · CON EL ALMACÉN CAÍDO, LA LECTURA SIGUE")
    cliente.cubo.caido = True
    main._ocr_azure = azure_falso
    llamadas.clear()
    t = await main._extract_text_from_upload(main._SubidaDeBytes(_pdf_escaneado(8), "sin-almacen.pdf"))
    await _esperar_subidas()
    ok(t == leido and len(llamadas) == 1, "Azure lee igual que antes de la caché")


asyncio.run(parte_main())

print("\n" + ("TODO PASA" if not FALLOS else f"{len(FALLOS)} FALLAN: {FALLOS}"))
raise SystemExit(1 if FALLOS else 0)
