"""Prueba de punta a punta de la Bóveda contra Supabase, Qdrant, Azure y OpenAI reales.

Cuenta de prueba aprobada por David: jdm.juridicoo@gmail.com (Platinum) — se cobra y se devuelve todo.
Puerta: test-trigger@jurexia.com (gratuita), sólo lectura.
Documentos: contratos FICTICIOS de scratchpad/muestras.

Lo único simulado: el token. No se puede iniciar sesión como el usuario, así que
`boveda._usuario` acepta «Bearer prueba:<uid>»; el PLAN se lee de verdad de la base.
No entra al lifespan de main.py (borraría cachés de Gemini de producción).

Uso (con .env cargado; cuesta ~0.05 USD de Azure + OpenAI y deja el saldo de la cuenta en cero):
  python datos_prueba_boveda/generar_muestras.py /tmp/muestras 60
  python -u prueba_boveda_punta_a_punta.py /tmp/muestras /tmp/salida_boveda

12-oct-2026: 45/45 (ver BOVEDA.md, «Probado de punta a punta»).
"""
import asyncio
import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
os.environ["BOVEDA_TRABAJADOR"] = "0"

from fastapi import FastAPI, HTTPException  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

import boveda  # noqa: E402
import boveda_indice as bi  # noqa: E402
import cobro_consultas as cobro  # noqa: E402

UID = "2cb17590-c36e-4d1b-83b1-be873ba6d5ed"        # jdm.juridicoo@gmail.com (Platinum)
UID_GRATIS = "d04b53a8-6f4b-4681-8ba8-786c15704b59"  # test-trigger@jurexia.com (gratuita)
TOK = {"Authorization": f"Bearer prueba:{UID}"}
TOK_GRATIS = {"Authorization": f"Bearer prueba:{UID_GRATIS}"}

ESCANEADOS = ["020_suministro_escaneado.pdf", "060_suministro_escaneado.pdf"]
CON_TEXTO = ["001_arrendamiento.pdf", "002_prestacion_de_servicios.pdf", "003_compraventa.pdf",
             "005_suministro.pdf", "006_arrendamiento.pdf", "007_prestacion_de_servicios.pdf",
             "009_confidencialidad.pdf", "011_arrendamiento.pdf"]
WORD = ["004_confidencialidad.docx", "008_compraventa.docx", "012_prestacion_de_servicios.docx"]

R = {"pasos": [], "fallas": []}


def ok(cond, que, **extra):
    R["pasos"].append({"ok": bool(cond), "que": que, **extra})
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}" + (f"  {extra}" if extra else ""))
    if not cond:
        R["fallas"].append(que)


async def usuario_falso(authorization):
    if not authorization or not authorization.startswith("Bearer prueba:"):
        raise HTTPException(status_code=401, detail="Autenticación requerida")
    uid = authorization.split("prueba:", 1)[1]
    plan, es_admin, leido = await cobro.plan_del_usuario(boveda._cliente(), uid, boveda._lista_admins())
    return boveda.Usuario(id=uid, plan=plan, es_admin=es_admin, plan_leido=leido)


boveda._usuario = usuario_falso

# La puerta de la sesión (29-oct-2026, puerta_sesion.py) comprueba el token
# del /chat contra Supabase porque el cuerpo dice `user_id`: aquí acepta el
# mismo testigo de prueba que la Bóveda.
import puerta_sesion as _ps  # noqa: E402
_sesion_de_verdad = _ps.sesion_del_token


async def _sesion_de_prueba(authorization, ruta=""):
    tok = _ps._extraer(authorization)
    if tok.startswith("prueba:"):
        return _ps.Sesion(id=tok.split(":", 1)[1].strip().lower(), correo="")
    return await _sesion_de_verdad(authorization, ruta)


_ps.sesion_del_token = _sesion_de_prueba


def sb():
    return boveda._cliente()


def perfil(uid=UID):
    r = sb().table("user_profiles").select("queries_used, consultas_recargadas").eq("id", uid).execute()
    return r.data[0]


async def esperar(cli, carpeta, condicion, tope_s=600, cada=2.0):
    t0 = time.time()
    while time.time() - t0 < tope_s:
        d = (await cli.get(f"/boveda/carpeta/{carpeta}", headers=TOK)).json()
        if condicion(d):
            return d, time.time() - t0
        await asyncio.sleep(cada)
    raise TimeoutError("la cola no terminó")


async def main():
    muestras, salida = Path(sys.argv[1]), Path(sys.argv[2])
    salida.mkdir(parents=True, exist_ok=True)
    verdad = json.loads((muestras / "verdad.json").read_text())
    inicial = perfil()
    R["saldo_inicial"] = inicial
    print("saldo inicial", inicial)

    app = FastAPI()
    app.include_router(boveda.router)
    cli = AsyncClient(transport=ASGITransport(app=app), base_url="http://prueba", timeout=120)

    # Carpeta de prueba de la cuenta de prueba.
    carpeta = sb().table("expedientes").insert({
        "user_id": UID, "tipo": "asunto", "titulo": "PRUEBA Bóveda (Claude) — se borra al terminar",
    }).execute().data[0]["id"]
    R["carpeta"] = carpeta
    cobros_ids = []
    bucle = None
    try:
        # ── Puerta ──────────────────────────────────────────────────────────
        print("\n── Puerta Platinum ──")
        r = await cli.get("/boveda/reglas", headers=TOK_GRATIS)
        ok(r.status_code == 200 and r.json()["platinum"] is False, "gratuita: /reglas dice platinum=false")
        r = await cli.get(f"/boveda/carpeta/{carpeta}", headers=TOK_GRATIS)
        ok(r.status_code == 403 and r.json()["detail"]["codigo"] == "requiere_platinum",
           "gratuita: 403 requiere_platinum", status=r.status_code)
        r = await cli.post(f"/boveda/carpeta/{carpeta}/subidas", headers=TOK_GRATIS,
                           json={"archivos": [{"nombre": "x.pdf", "tamano": 10}]})
        ok(r.status_code == 403, "gratuita: no puede reservar subidas")
        r = await cli.get(f"/boveda/carpeta/{carpeta}")
        ok(r.status_code == 401, "sin token: 401")
        r = await cli.get("/boveda/reglas", headers=TOK)
        ok(r.status_code == 200 and r.json()["platinum"] is True, "Platinum: /reglas dice platinum=true",
           disponibles=r.json().get("disponibles"))

        # ── Subir ───────────────────────────────────────────────────────────
        print("\n── Subir ──")
        archivos = [(n, (muestras / n).read_bytes()) for n in ESCANEADOS + CON_TEXTO + WORD]
        archivos.append(("roto.pdf", b"%PDF-1.4 esto no es un pdf de verdad"))
        archivos.append(("copia_del_001.pdf", (muestras / "001_arrendamiento.pdf").read_bytes()))
        pedido = [{"nombre": n, "tamano": len(b), "ref": str(i)} for i, (n, b) in enumerate(archivos)]
        pedido.append({"nombre": "notas.txt", "tamano": 10, "ref": "txt"})
        t0 = time.time()
        r = (await cli.post(f"/boveda/carpeta/{carpeta}/subidas", headers=TOK, json={"archivos": pedido})).json()
        ok(len(r["aceptados"]) == len(archivos) and [x["nombre"] for x in r["rechazados"]] == ["notas.txt"],
           "reserva: acepta PDF/Word y rechaza .txt", rechazados=r["rechazados"])
        listos, fallidos = [], []
        for a in r["aceptados"]:
            nombre, datos = archivos[int(a["ref"])]
            mime = boveda.bt.MIME_DOCX if nombre.endswith(".docx") else "application/pdf"
            try:
                await asyncio.to_thread(lambda: sb().storage.from_("boveda").upload_to_signed_url(
                    a["path"], a["token"], datos, {"content-type": mime}))
                listos.append(a["id"])
            except Exception as e:
                fallidos.append((nombre, str(e)[:120]))
        ok(not fallidos, "subida directa con URL firmada", fallidos=fallidos)
        r = (await cli.post(f"/boveda/carpeta/{carpeta}/subidas/listas", headers=TOK,
                            json={"ids": listos})).json()
        ok(r["recibidos"] == len(listos), "aviso de subidas listas", recibidos=r["recibidos"])
        R["seg_subida"] = round(time.time() - t0, 1)
        # Un cliente NO puede escribir directo en el cubo (sin políticas).
        try:
            from supabase import create_client
            anon = create_client(os.environ["SUPABASE_URL"], os.environ["NEXT_PUBLIC_SUPABASE_ANON_KEY"])
            anon.storage.from_("boveda").upload(f"{UID}/{carpeta}/intruso.pdf", b"%PDF", {"content-type": "application/pdf"})
            ok(False, "la clave pública NO puede subir al cubo")
        except Exception:
            ok(True, "la clave pública NO puede subir al cubo")

        # ── Contar páginas (trabajador real) ────────────────────────────────
        print("\n── Revisión (contar páginas) ──")
        bucle = asyncio.ensure_future(boveda._bucle())
        d, seg = await esperar(cli, carpeta, lambda d: d["resumen"]["por_revisar"] == 0)
        R["seg_revision"] = round(seg, 1)
        por_nombre = {x["nombre"]: x for x in d["documentos"]}
        ok(por_nombre["roto.pdf"]["estado"] == "error", "un PDF dañado se rechaza sin cobrar",
           error=por_nombre["roto.pdf"]["error"])
        ok(por_nombre["copia_del_001.pdf"]["estado"] == "error" or por_nombre["001_arrendamiento.pdf"]["estado"] == "error",
           "el mismo archivo dos veces en la carpeta se rechaza",
           error=(por_nombre["copia_del_001.pdf"]["error"] or por_nombre["001_arrendamiento.pdf"]["error"]))
        cot = d["cotizacion"]
        ocr_esperadas = sum(por_nombre[n]["paginas"] for n in ESCANEADOS)
        ok(cot["paginas_ocr"] == ocr_esperadas, "las escaneadas se cuentan como OCR",
           paginas_ocr=cot["paginas_ocr"], paginas=cot["paginas"])
        esperado = boveda.consultas_ingesta(cot["paginas_ocr"], cot["paginas_texto"])
        ok(cot["consultas"] == esperado and cot["documentos"] == len(ESCANEADOS + CON_TEXTO + WORD),
           "cotización = ⌈escaneadas/10 + texto/100⌉", consultas=cot["consultas"], disponibles=cot["disponibles"])
        R["cotizacion"] = cot

        # ── Confirmar (cobro) y forzar un fallo para ver la devolución ──────
        print("\n── Confirmar y cobrar ──")
        bucle.cancel()
        antes = perfil()
        r = await cli.post(f"/boveda/carpeta/{carpeta}/confirmar", headers=TOK,
                           json={"ids": cot["ids"], "consultas_esperadas": cot["consultas"] + 1})
        ok(r.status_code == 409, "si la cotización no coincide, 409 y no cobra")
        ok(perfil() == antes, "…y el saldo no se movió")
        r = (await cli.post(f"/boveda/carpeta/{carpeta}/confirmar", headers=TOK,
                            json={"ids": cot["ids"], "consultas_esperadas": cot["consultas"]})).json()
        tanda = r["tanda_id"]
        despues = perfil()
        ok(despues["queries_used"] - antes["queries_used"] == cot["consultas"],
           "cobro del MISMO contador del chat (queries_used)", cobradas=r["consultas_cobradas"])
        t = sb().table("boveda_tandas").select("*").eq("id", tanda).execute().data[0]
        cobros_ids.append(t["cobro_id"])
        c = sb().table("consultas_cobros").select("*").eq("id", t["cobro_id"]).execute().data[0]
        ok(c["unidades"] == cot["consultas"] and c["motivo"] == "boveda_ingesta", "fila en consultas_cobros",
           referencia=c["referencia"])
        # El escaneado de 5 páginas «cambia» DESPUÉS de cobrar: debe fallar
        # («el archivo cambió») y devolverse lo que no se usó. Se simula en la
        # huella: sobrescribir el objeto no sirve, la CDN de Storage sigue
        # sirviendo el original (y un usuario no puede sobrescribir: el cubo
        # no tiene políticas y la URL firmada no admite upsert).
        doc_cambiado = por_nombre["020_suministro_escaneado.pdf"]
        sb().table("boveda_documentos").update({"huella": "0" * 64}).eq("id", doc_cambiado["id"]).execute()

        # ── Ingesta (trabajador real) ───────────────────────────────────────
        print("\n── Ingesta ──")
        bucle = asyncio.ensure_future(boveda._bucle())
        d, seg = await esperar(cli, carpeta, lambda d: d["resumen"]["en_proceso"] == 0, tope_s=900)
        R["seg_ingesta"] = round(seg, 1)
        por_nombre = {x["nombre"]: x for x in d["documentos"]}
        ok(por_nombre["020_suministro_escaneado.pdf"]["estado"] == "error",
           "el archivo cambiado tras cotizar se rechaza", error=por_nombre["020_suministro_escaneado.pdf"]["error"])
        listos_n = d["resumen"]["listos"]
        ok(listos_n == len(ESCANEADOS + CON_TEXTO + WORD) - 1, "los demás quedan listos", listos=listos_n)
        await asyncio.sleep(1)
        t = sb().table("boveda_tandas").select("*").eq("id", tanda).execute().data[0]
        ok_docs = [x for x in d["documentos"] if x["estado"] == "listo"]
        ocr_ok = sum(x["paginas_ocr"] or 0 for x in ok_docs)
        justas = boveda.consultas_ingesta(ocr_ok, sum(x["paginas"] for x in ok_docs) - ocr_ok)
        ok(t["estado"] == "terminada" and t["devueltas"] == cot["consultas"] - justas,
           "al cerrar la tanda se devuelve lo que no se usó", cobradas=t["consultas"], devueltas=t["devueltas"])
        ok(perfil()["queries_used"] - antes["queries_used"] == justas, "el contador refleja la devolución")
        n_pts = await bi.contar(UID, carpeta)
        frag = sum(x["fragmentos"] or 0 for x in ok_docs)
        ok(n_pts == frag and n_pts > 0, "puntos en Qdrant = fragmentos indexados", puntos=n_pts)
        R["documentos"] = d["documentos"]

        # ── Buscar ──────────────────────────────────────────────────────────
        print("\n── Buscar ──")
        t0 = time.time()
        r = (await cli.post(f"/boveda/carpeta/{carpeta}/buscar", headers=TOK,
                            json={"pregunta": "¿a qué tribunales se someten las partes?", "limite": 5})).json()
        ok(r["resultados"] and "jurisdic" in r["resultados"][0]["texto"].lower() or "tribunal" in r["resultados"][0]["texto"].lower(),
           "la búsqueda trae la cláusula de jurisdicción", ms=round(1000 * (time.time() - t0)),
           primero=f"{r['resultados'][0]['doc_nombre']} p. {r['resultados'][0]['pagina']}")
        r = await cli.post(f"/boveda/carpeta/{carpeta}/buscar", headers=TOK_GRATIS, json={"pregunta": "pena"})
        ok(r.status_code == 403, "otra cuenta no puede buscar en esta carpeta")
        hs = await bi.buscar("pena", user_id=UID_GRATIS, carpeta_id=carpeta, limite=20)
        ok(not hs, "Qdrant: otro usuario con la MISMA carpeta no ve nada")

        # ── Documento: URL firmada y páginas ────────────────────────────────
        un_pdf = next(x for x in ok_docs if x["nombre"] == "003_compraventa.pdf")
        r = (await cli.get(f"/boveda/documento/{un_pdf['id']}", headers=TOK)).json()
        ok(r["url"] and "token=" in r["url"], "URL firmada para el visor")
        r = await cli.get(f"/boveda/documento/{un_pdf['id']}", headers=TOK_GRATIS)
        ok(r.status_code in (403, 404), "otra cuenta no obtiene la URL")
        r = (await cli.get(f"/boveda/documento/{un_pdf['id']}/paginas", headers=TOK)).json()
        ok(len(r["paginas"]) == un_pdf["paginas"], "texto por página guardado")

        # ── Tabla de revisión ───────────────────────────────────────────────
        print("\n── Tabla de revisión ──")
        cols = (await cli.get("/boveda/columnas-sugeridas")).json()["columnas"]
        cols.append({"id": "", "titulo": "Renovación", "pregunta": "¿El contrato se renueva automáticamente?",
                     "tipo": "si_no"})
        tabla = (await cli.post(f"/boveda/carpeta/{carpeta}/tablas", headers=TOK,
                                json={"titulo": "Revisión de contratos (prueba)", "columnas": cols})).json()
        cot_t = (await cli.post(f"/boveda/tablas/{tabla['id']}/cotizar", headers=TOK, json={})).json()
        ok(cot_t["celdas"] == listos_n * len(cols) and cot_t["consultas"] == boveda.consultas_celdas(cot_t["celdas"]),
           "cotización de la tabla = ⌈celdas/10⌉", celdas=cot_t["celdas"], consultas=cot_t["consultas"])
        antes_t = perfil()
        r = (await cli.post(f"/boveda/tablas/{tabla['id']}/ejecutar", headers=TOK,
                            json={"consultas_esperadas": cot_t["consultas"]})).json()
        ok(perfil()["queries_used"] - antes_t["queries_used"] == cot_t["consultas"], "cobro de la tabla")
        tt = sb().table("boveda_tandas").select("cobro_id").eq("id", r["tanda_id"]).execute().data[0]
        cobros_ids.append(tt["cobro_id"])
        t0 = time.time()
        while True:
            tb = (await cli.get(f"/boveda/tablas/{tabla['id']}", headers=TOK)).json()
            if tb["pendientes"] == 0:
                break
            if time.time() - t0 > 600:
                raise TimeoutError("la tabla no terminó")
            await asyncio.sleep(2)
        R["seg_tabla"] = round(time.time() - t0, 1)
        celdas = tb["celdas"]
        listas = [c for c in celdas if c["estado"] == "lista"]
        enc = [c for c in listas if (c["respuesta"] or {}).get("encontrado")]
        ver = [c for c in enc if c["respuesta"].get("verificada")]
        ok(len(listas) == cot_t["celdas"], "todas las celdas respondidas", listas=len(listas))
        ok(len(ver) >= 0.95 * len(enc), "≥95% de citas verificadas", verificadas=f"{len(ver)}/{len(enc)}")
        # Exactitud contra la verdad (las 6 sugeridas).
        nombres = {f["id"]: f["nombre"] for f in tb["filas"]}
        acierto = {"pena": [0, 0], "monto": [0, 0], "vigencia": [0, 0], "jurisdiccion": [0, 0]}
        for c in listas:
            v = verdad.get(nombres[c["documento_id"]])
            if not v or c["columna_id"] not in acierto:
                continue
            val = (c["respuesta"].get("valor") or "").lower()
            if c["columna_id"] == "pena":
                bien = (str(v["pena_convencional_pct"]) in val) if v["pena_convencional_pct"] else not c["respuesta"].get("encontrado") or val.startswith("no")
            elif c["columna_id"] == "monto":
                bien = f"{v['monto']:,}" in c["respuesta"]["valor"] or str(v["monto"]) in c["respuesta"]["valor"].replace(",", "")
            elif c["columna_id"] == "vigencia":
                bien = str(v["vigencia_meses"]) in val
            else:
                import unicodedata
                p = lambda t: "".join(ch for ch in unicodedata.normalize("NFD", t) if unicodedata.category(ch) != "Mn").lower()
                bien = p(v["ciudad"].split(",")[0]).split()[-1] in p(val)
            acierto[c["columna_id"]][0] += bool(bien)
            acierto[c["columna_id"]][1] += 1
        ok(all(a == b for a, b in acierto.values()), "exactitud contra la verdad",
           **{k: f"{a}/{b}" for k, (a, b) in acierto.items()})
        costo = sb().table("boveda_celdas").select("costo_usd, tokens_entrada, tokens_salida").eq("tabla_id", tabla["id"]).execute().data
        R["tabla_usd"] = round(sum(float(x["costo_usd"] or 0) for x in costo), 5)
        R["tabla_celdas"] = len(costo)
        # Exportar.
        x = await cli.get(f"/boveda/tablas/{tabla['id']}/exportar?formato=xlsx", headers=TOK)
        (salida / "tabla.xlsx").write_bytes(x.content)
        from openpyxl import load_workbook
        wb = load_workbook(io.BytesIO(x.content))
        ok(x.status_code == 200 and wb["Revisión"].max_row == listos_n + 1 and wb["Citas"].max_row > listos_n,
           "Excel exportado (Revisión + Citas)")
        w = await cli.get(f"/boveda/tablas/{tabla['id']}/exportar?formato=docx", headers=TOK)
        (salida / "tabla.docx").write_bytes(w.content)
        import docx
        dd = docx.Document(io.BytesIO(w.content))
        ok(w.status_code == 200 and len(dd.tables[0].rows) == listos_n + 1, "Word exportado")
        R["muestra_celdas"] = [
            {"doc": nombres[c["documento_id"]], "col": c["columna_id"], **c["respuesta"]} for c in listas[:8]]

        # ── Chat con la Bóveda como fuente (main.py real, sin lifespan) ─────
        print("\n── Chat ──")
        R["chat"] = await probar_chat(carpeta)

        # ── Borrar: todo desaparece ─────────────────────────────────────────
        print("\n── Borrar ──")
        rutas = [x["storage_path"] for x in sb().table("boveda_documentos").select("storage_path")
                 .eq("expediente_id", carpeta).execute().data]
        r = (await cli.delete(f"/boveda/carpeta/{carpeta}", headers=TOK)).json()
        ok(r["borrados"] == len(rutas), "vaciar la Bóveda de la carpeta", borrados=r["borrados"])
        await asyncio.sleep(2)
        ok(await bi.contar(UID, carpeta) == 0, "Qdrant: 0 puntos de la carpeta")
        quedan = sb().storage.from_("boveda").list(f"{UID}/{carpeta}")
        ok(not quedan, "Storage: 0 archivos (y 0 .paginas.json.gz)", quedan=len(quedan or []))
        pend = sb().table("boveda_purgas").select("id").eq("expediente_id", carpeta).is_("hecho_en", "null").execute().data
        ok(not pend, "purgas hechas")
        ok(not sb().table("boveda_tablas").select("id").eq("expediente_id", carpeta).execute().data, "tablas borradas")
    finally:
        if bucle:
            bucle.cancel()
        # ── Neto cero ───────────────────────────────────────────────────────
        print("\n── Devolver todo ──")
        for cid in cobros_ids:
            if cid:
                await cobro.devolver_cobro(sb(), cid)
        if R.get("chat", {}).get("cobrada"):
            sb().rpc("devolver_consulta", {"p_user_id": UID}).execute()
        try:
            sb().table("boveda_documentos").delete().eq("expediente_id", carpeta).execute()
            await boveda._purgar_pendientes()
            sb().table("expedientes").delete().eq("id", carpeta).execute()
        except Exception as e:
            print("limpieza:", e)
        final = perfil()
        R["saldo_final"] = final
        ok(final == inicial, "saldo de la cuenta de prueba de vuelta al inicial", inicial=inicial, final=final)
        (salida / "e2e.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str))
        print(f"\n{len(R['pasos']) - len(R['fallas'])}/{len(R['pasos'])} pasan")
        await cli.aclose()


async def probar_chat(carpeta: str) -> dict:
    import main
    from openai import AsyncOpenAI
    from qdrant_client import AsyncQdrantClient
    from fastembed import SparseTextEmbedding
    # Los clientes a mano, como en la receta de «probar sin el arranque».
    main.qdrant_client = AsyncQdrantClient(url=main.QDRANT_URL, api_key=main.QDRANT_API_KEY, timeout=30)
    main.openai_client = AsyncOpenAI(api_key=main.OPENAI_API_KEY)
    main.chat_client = AsyncOpenAI(api_key=main.OPENAI_API_KEY)
    main.deepseek_client = AsyncOpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
    main.sparse_encoder = await asyncio.to_thread(lambda: SparseTextEmbedding(model_name="Qdrant/bm25"))
    import httpx
    main._http_pool = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
    boveda.configurar(main.supabase_admin, main.ADMIN_EMAILS)
    bi.configurar(main.openai_client)
    antes = perfil()
    cli = AsyncClient(transport=ASGITransport(app=main.app), base_url="http://prueba", timeout=240)
    pregunta = ("Según los contratos de mi carpeta, ¿cuál es la pena convencional pactada en el contrato "
                "de compraventa 003 y a qué tribunales se sometieron las partes?")
    cuerpo = {"messages": [{"role": "user", "content": pregunta}], "user_id": UID,
              "estado": "QUERETARO", "boveda_carpeta_id": carpeta}
    t0 = time.time()
    texto = ""
    async with cli.stream("POST", "/chat", json=cuerpo, headers={**TOK, "X-Razonamiento-Vivo": "1"}) as r:
        estado = r.status_code
        async for trozo in r.aiter_text():
            texto += trozo
    seg = round(time.time() - t0, 1)
    cobrada = perfil()["queries_used"] - antes["queries_used"] == 1
    import re
    meta = re.search(r"<!-- CITATION_META:(\{.*?\}) -->", texto, re.S)
    fuentes = json.loads(meta.group(1)).get("sources", {}) if meta else {}
    de_boveda = {k: v for k, v in fuentes.items() if v.get("silo") == "boveda"}
    ok(estado == 200, "chat responde", seg=seg)
    ok("<!--PASO:boveda|" in texto, "el chat recuperó fragmentos de la Bóveda")
    ok(bool(de_boveda), "la respuesta cita documentos de la Bóveda", citas_boveda=len(de_boveda))
    ok(all(v.get("boveda_doc_id") and v.get("pagina") for v in de_boveda.values()),
       "cada cita de la Bóveda trae documento y página para el visor")
    # Sin sesión válida la Bóveda no entra (el resto del chat sigue).
    limpio = re.sub(r"<!--.*?-->", "", texto, flags=re.S)
    return {"segundos": seg, "cobrada": cobrada, "citas_boveda": list(de_boveda.values())[:3],
            "respuesta": limpio.strip()[:1500]}


asyncio.run(main())
