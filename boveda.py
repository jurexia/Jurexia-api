"""LA BÓVEDA DE IUREXIA.   (oct-2026)

Por carpeta, de 50 a 500 documentos (PDF y Word) indexados por usuario y
carpeta, para preguntarles juntos: en el chat (fuente adicional, cita a
documento y página) y en la tabla de revisión (filas = documentos, columnas =
preguntas). Diseño y costos medidos: BOVEDA.md. Sólo Platinum.

EL RECORRIDO DE UN DOCUMENTO
  1. POST /boveda/carpeta/{id}/subidas       la API revisa plan y cuotas y da
     una URL firmada por archivo; el navegador sube DIRECTO a Storage.
  2. POST …/subidas/listas                   el navegador avisa → «subido».
  3. El trabajador lo descarga, saca su huella y cuenta páginas (cuáles son
     escaneadas) SIN pagar nada → «revisado».
  4. POST …/cotizar   «47 documentos · 312 páginas (40 escaneadas) → 8 consultas».
  5. POST …/confirmar cobra (todo o nada) y los pone «en_cola».
  6. El trabajador hace OCR sólo de las páginas escaneadas, parte en
     fragmentos por página, indexa en Qdrant → «listo». Si alguno falla, al
     cerrar la tanda se devuelve lo que no se usó.

LA COLA vive en Supabase y la trabajan los tres procesos de uvicorn a la vez:
cada uno toma trabajo con FOR UPDATE SKIP LOCKED (boveda_tomar_documento,
boveda_tomar_fila). Nada vive sólo en memoria: un despliegue a media ingesta
deja trabajos «huérfanos» que se retoman a los 15 minutos.

EL DUEÑO SALE DEL TOKEN, NUNCA DEL CUERPO. Y toda lectura de Qdrant filtra por
ese dueño (boveda_indice exige user_id).
"""
from __future__ import annotations

import asyncio
import gzip
import hashlib
import io
import json
import math
import os
import socket
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

import boveda_indice as bi
import boveda_tabla as btab
import boveda_texto as bt
import cobro_consultas as cobro

router = APIRouter()

CUBO = "boveda"


def _ent(nombre: str, defecto: int) -> int:
    try:
        return int(os.getenv(nombre, str(defecto)))
    except ValueError:
        return defecto


# ── La regla de cobro (BOVEDA.md) ──────────────────────────────────────────
PAG_OCR_POR_CONSULTA = _ent("BOVEDA_PAGINAS_OCR_POR_CONSULTA", 10)
PAG_TEXTO_POR_CONSULTA = _ent("BOVEDA_PAGINAS_TEXTO_POR_CONSULTA", 100)
CELDAS_POR_CONSULTA = _ent("BOVEDA_CELDAS_POR_CONSULTA", 10)

# ── Cuotas por usuario ─────────────────────────────────────────────────────
MAX_DOCS_CARPETA = _ent("BOVEDA_MAX_DOCS_CARPETA", 500)
MAX_DOCS_USUARIO = _ent("BOVEDA_MAX_DOCS_USUARIO", 2000)
MAX_PAGINAS_USUARIO = _ent("BOVEDA_MAX_PAGINAS_USUARIO", 10000)
MAX_PAGINAS_DOC = _ent("BOVEDA_MAX_PAGINAS_DOC", 1500)
MAX_MB_ARCHIVO = _ent("BOVEDA_MAX_MB_ARCHIVO", 50)
MAX_GB_USUARIO = _ent("BOVEDA_MAX_GB_USUARIO", 5)
MAX_COLUMNAS = 20
SUBIDAS_POR_LLAMADA = 100

# ── El trabajador ──────────────────────────────────────────────────────────
DOCS_A_LA_VEZ = _ent("BOVEDA_DOCS_A_LA_VEZ", 1)      # por proceso (3 procesos en Render)
FILAS_A_LA_VEZ = _ent("BOVEDA_FILAS_A_LA_VEZ", 3)
OCR_A_LA_VEZ = _ent("BOVEDA_OCR_A_LA_VEZ", 2)
ESPERA_MAXIMA_S = 30
TRABAJADOR = f"{socket.gethostname()}:{os.getpid()}"

ACTIVOS = ("en_cola", "procesando", "listo")
PENDIENTES_DE_REVISION = ("subiendo", "subido", "revisando")


# ═══ Utilidades ════════════════════════════════════════════════════════════
def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _db(fn):
    """Una llamada síncrona a Supabase, fuera del bucle de eventos."""
    return await asyncio.to_thread(fn, cobro.cliente())


def consultas_ingesta(paginas_ocr: int, paginas_texto: int) -> int:
    u = paginas_ocr / max(1, PAG_OCR_POR_CONSULTA) + paginas_texto / max(1, PAG_TEXTO_POR_CONSULTA)
    return 0 if u <= 0 else max(1, math.ceil(u - 1e-9))


def consultas_celdas(celdas: int) -> int:
    return cobro.consultas_por(celdas, CELDAS_POR_CONSULTA)


def _ruta_paginas(storage_path: str) -> str:
    return storage_path + ".paginas.json.gz"


def _ext(nombre: str) -> str:
    return "pdf" if bt.tipo_de(nombre) == "pdf" else "docx"


class Rechazo(Exception):
    """Un problema del ARCHIVO (no nuestro): no se reintenta y se le dice al abogado."""


async def _carpeta_propia(user_id: str, carpeta_id: str) -> dict:
    try:
        uuid.UUID(carpeta_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Carpeta no encontrada")
    r = await _db(lambda sb: sb.table("expedientes").select("id, user_id, titulo, cliente_nombre")
                  .eq("id", carpeta_id).eq("user_id", user_id).limit(1).execute())
    if not r.data:
        raise HTTPException(status_code=404, detail="Carpeta no encontrada")
    return r.data[0]


async def _documento_propio(user_id: str, doc_id: str) -> dict:
    try:
        uuid.UUID(doc_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    r = await _db(lambda sb: sb.table("boveda_documentos").select("*")
                  .eq("id", doc_id).eq("user_id", user_id).limit(1).execute())
    if not r.data:
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    return r.data[0]


async def _tabla_propia(user_id: str, tabla_id: str) -> dict:
    try:
        uuid.UUID(tabla_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Tabla no encontrada")
    r = await _db(lambda sb: sb.table("boveda_tablas").select("*")
                  .eq("id", tabla_id).eq("user_id", user_id).limit(1).execute())
    if not r.data:
        raise HTTPException(status_code=404, detail="Tabla no encontrada")
    return r.data[0]


async def _leer_paginas(doc: dict) -> list[str]:
    datos = await _db(lambda sb: sb.storage.from_(CUBO).download(_ruta_paginas(doc["storage_path"])))
    ficha = json.loads(gzip.decompress(datos).decode("utf-8"))
    return list(ficha.get("paginas") or [])


async def _guardar_paginas(doc: dict, paginas: list[str], motor: str) -> None:
    ficha = {"v": 1, "doc_id": doc["id"], "motor": motor, "paginas": paginas}
    datos = gzip.compress(json.dumps(ficha, ensure_ascii=False).encode("utf-8"), compresslevel=6)
    await _db(lambda sb: sb.storage.from_(CUBO).upload(
        _ruta_paginas(doc["storage_path"]), datos,
        {"content-type": "application/gzip", "upsert": "true"}))


_despertar: Optional[asyncio.Event] = None


def _avisar_trabajador() -> None:
    """Hay trabajo nuevo: que el trabajador de ESTE proceso no espere su siesta."""
    if _despertar is not None:
        _despertar.set()


# ═══ Puerta y reglas ═══════════════════════════════════════════════════════
@router.get("/boveda/reglas")
async def boveda_reglas(authorization: str = Header(None)):
    """Lo que el front necesita para pintar la Bóveda o el candado. No da 403:
    quien no es Platinum recibe `platinum: false` y ve la invitación."""
    u = await cobro.usuario_del_token(authorization)
    disponibles = None
    if u.es_platinum:
        try:
            disponibles = (await cobro.consultas_disponibles(u.id)).get("disponibles")
        except Exception:
            disponibles = None
    return {
        "platinum": u.es_platinum, "plan": u.plan, "disponibles": disponibles,
        "reglas": {"paginas_ocr_por_consulta": PAG_OCR_POR_CONSULTA,
                   "paginas_texto_por_consulta": PAG_TEXTO_POR_CONSULTA,
                   "celdas_por_consulta": CELDAS_POR_CONSULTA},
        "cuotas": {"docs_carpeta": MAX_DOCS_CARPETA, "docs_usuario": MAX_DOCS_USUARIO,
                   "paginas_usuario": MAX_PAGINAS_USUARIO, "paginas_documento": MAX_PAGINAS_DOC,
                   "mb_archivo": MAX_MB_ARCHIVO, "gb_usuario": MAX_GB_USUARIO},
    }


async def _uso_usuario(user_id: str) -> dict:
    r = await _db(lambda sb: sb.table("boveda_documentos").select("estado, paginas, tamano")
                  .eq("user_id", user_id).limit(MAX_DOCS_USUARIO + 500).execute())
    docs = r.data or []
    return {
        "documentos": len(docs),
        "bytes": sum(int(d.get("tamano") or 0) for d in docs),
        "paginas": sum(int(d.get("paginas") or 0) for d in docs if d.get("estado") in ACTIVOS),
    }


# ═══ La carpeta ════════════════════════════════════════════════════════════
def _doc_publico(d: dict) -> dict:
    return {k: d.get(k) for k in ("id", "nombre", "estado", "paginas", "paginas_ocr", "tamano",
                                  "fragmentos", "error", "creado_en", "mime_type", "tanda_id")} | {
        "aproximadas": bt.tipo_de(d.get("nombre") or "", d.get("mime_type")) == "docx"}


def _cotizacion(docs: list[dict], disponibles: Optional[int], uso: dict) -> dict:
    ocr = sum(int(d.get("paginas_ocr") or 0) for d in docs)
    total = sum(int(d.get("paginas") or 0) for d in docs)
    consultas = consultas_ingesta(ocr, total - ocr)
    excede = uso["paginas"] + total > MAX_PAGINAS_USUARIO
    return {
        "ids": [d["id"] for d in docs], "documentos": len(docs), "paginas": total,
        "paginas_ocr": ocr, "paginas_texto": total - ocr, "consultas": consultas,
        "disponibles": disponibles,
        "alcanza": disponibles is None or disponibles >= consultas,
        "excede_cuota": excede,
        "mensaje_cuota": (f"Tu Bóveda llegaría a {uso['paginas'] + total:,} páginas; el máximo es "
                          f"{MAX_PAGINAS_USUARIO:,}. Borra documentos que ya no uses." if excede else None),
    }


@router.get("/boveda/carpeta/{carpeta_id}")
async def boveda_carpeta(carpeta_id: str, authorization: str = Header(None)):
    u = await cobro.exigir_platinum(authorization)
    await _carpeta_propia(u.id, carpeta_id)
    r = await _db(lambda sb: sb.table("boveda_documentos").select("*")
                  .eq("expediente_id", carpeta_id).eq("user_id", u.id)
                  .order("creado_en").limit(MAX_DOCS_CARPETA + 100).execute())
    docs = r.data or []
    t = await _db(lambda sb: sb.table("boveda_tablas").select("id, titulo, columnas, actualizado_en")
                  .eq("expediente_id", carpeta_id).eq("user_id", u.id).order("creado_en").execute())
    estados: dict[str, int] = {}
    for d in docs:
        estados[d["estado"]] = estados.get(d["estado"], 0) + 1
    por_cobrar = [d for d in docs if d["estado"] == "revisado"]
    cotizacion = None
    if por_cobrar:
        disp = (await cobro.consultas_disponibles(u.id)).get("disponibles") if not u.es_admin else None
        cotizacion = _cotizacion(por_cobrar, disp, await _uso_usuario(u.id))
    return {
        "carpeta_id": carpeta_id,
        "documentos": [_doc_publico(d) for d in docs],
        "resumen": {
            "total": len(docs),
            "listos": estados.get("listo", 0),
            "por_revisar": sum(estados.get(e, 0) for e in PENDIENTES_DE_REVISION),
            "por_cobrar": len(por_cobrar),
            "en_proceso": estados.get("en_cola", 0) + estados.get("procesando", 0),
            "errores": estados.get("error", 0),
            "paginas_listas": sum(int(d.get("paginas") or 0) for d in docs if d["estado"] == "listo"),
        },
        "cotizacion": cotizacion,
        "tablas": [{"id": x["id"], "titulo": x["titulo"], "columnas": len(x.get("columnas") or []),
                    "actualizado_en": x["actualizado_en"]} for x in (t.data or [])],
    }


class ArchivoNuevo(BaseModel):
    nombre: str = Field(..., min_length=1, max_length=300)
    tamano: int = Field(..., ge=1)
    mime: Optional[str] = None
    ref: Optional[str] = Field(None, max_length=40)   # lo devuelve tal cual: el front casa archivo y URL


class SubidasRequest(BaseModel):
    archivos: list[ArchivoNuevo] = Field(..., min_length=1, max_length=SUBIDAS_POR_LLAMADA)


@router.post("/boveda/carpeta/{carpeta_id}/subidas")
async def boveda_subidas(carpeta_id: str, req: SubidasRequest, authorization: str = Header(None)):
    """Reserva los documentos y da una URL firmada por archivo. El navegador
    sube con supabase.storage.from('boveda').uploadToSignedUrl(path, token, file)."""
    u = await cobro.exigir_platinum(authorization)
    await _carpeta_propia(u.id, carpeta_id)
    en_carpeta = await _db(lambda sb: sb.table("boveda_documentos").select("id", count="exact")
                           .eq("expediente_id", carpeta_id).eq("user_id", u.id).limit(1).execute())
    n_carpeta = int(en_carpeta.count or 0)
    uso = await _uso_usuario(u.id)
    aceptados, rechazados, filas = [], [], []
    refs: dict[str, Optional[str]] = {}
    tope_bytes = MAX_GB_USUARIO * 1024 ** 3
    bytes_nuevos = 0
    for a in req.archivos:
        nombre = a.nombre.strip().replace("/", "_").replace("\\", "_")
        motivo = None
        if not bt.tipo_de(nombre, a.mime):
            motivo = "Sólo PDF y Word (.docx)."
        elif a.tamano > MAX_MB_ARCHIVO * 1024 * 1024:
            motivo = f"Pesa más de {MAX_MB_ARCHIVO} MB."
        elif n_carpeta + len(filas) >= MAX_DOCS_CARPETA:
            motivo = f"La Bóveda de una carpeta admite {MAX_DOCS_CARPETA} documentos."
        elif uso["documentos"] + len(filas) >= MAX_DOCS_USUARIO:
            motivo = f"Tu Bóveda admite {MAX_DOCS_USUARIO:,} documentos en total."
        elif uso["bytes"] + bytes_nuevos + a.tamano > tope_bytes:
            motivo = f"Tu Bóveda admite {MAX_GB_USUARIO} GB de archivos."
        if motivo:
            rechazados.append({"nombre": nombre, "motivo": motivo, "ref": a.ref})
            continue
        doc_id = str(uuid.uuid4())
        ext = _ext(nombre)
        bytes_nuevos += a.tamano
        refs[doc_id] = a.ref
        filas.append({
            "id": doc_id, "user_id": u.id, "expediente_id": carpeta_id, "nombre": nombre,
            "mime_type": bt.MIME_PDF if ext == "pdf" else bt.MIME_DOCX, "tamano": a.tamano,
            "storage_path": f"{u.id}/{carpeta_id}/{doc_id}.{ext}", "estado": "subiendo",
        })
    if filas:
        await _db(lambda sb: sb.table("boveda_documentos").insert(filas).execute())
        sem = asyncio.Semaphore(8)

        async def firmar(f):
            async with sem:
                s = await _db(lambda sb: sb.storage.from_(CUBO).create_signed_upload_url(f["storage_path"]))
            return {"id": f["id"], "nombre": f["nombre"], "path": f["storage_path"], "ref": refs.get(f["id"]),
                    "token": s["token"], "url": s.get("signed_url") or s.get("signedUrl")}

        aceptados = await asyncio.gather(*(firmar(f) for f in filas))
    return {"aceptados": aceptados, "rechazados": rechazados}


class ListasRequest(BaseModel):
    ids: list[str] = Field(default_factory=list, max_length=500)
    fallidos: list[str] = Field(default_factory=list, max_length=500)


@router.post("/boveda/carpeta/{carpeta_id}/subidas/listas")
async def boveda_subidas_listas(carpeta_id: str, req: ListasRequest, authorization: str = Header(None)):
    u = await cobro.exigir_platinum(authorization)
    await _carpeta_propia(u.id, carpeta_id)
    n = 0
    if req.ids:
        r = await _db(lambda sb: sb.table("boveda_documentos")
                      .update({"estado": "subido", "actualizado_en": _ahora()})
                      .in_("id", req.ids).eq("user_id", u.id).eq("expediente_id", carpeta_id)
                      .eq("estado", "subiendo").execute())
        n = len(r.data or [])
    if req.fallidos:
        await _db(lambda sb: sb.table("boveda_documentos").delete()
                  .in_("id", req.fallidos).eq("user_id", u.id).eq("estado", "subiendo").execute())
    _avisar_trabajador()
    return {"recibidos": n}


@router.post("/boveda/carpeta/{carpeta_id}/cotizar")
async def boveda_cotizar(carpeta_id: str, authorization: str = Header(None)):
    u = await cobro.exigir_platinum(authorization)
    await _carpeta_propia(u.id, carpeta_id)
    r = await _db(lambda sb: sb.table("boveda_documentos").select("*")
                  .eq("expediente_id", carpeta_id).eq("user_id", u.id)
                  .in_("estado", ["revisado", *PENDIENTES_DE_REVISION]).execute())
    docs = r.data or []
    listos = [d for d in docs if d["estado"] == "revisado"]
    disp = None if u.es_admin else (await cobro.consultas_disponibles(u.id)).get("disponibles")
    c = _cotizacion(listos, disp, await _uso_usuario(u.id))
    c["pendientes_de_revision"] = len(docs) - len(listos)
    c["sin_costo"] = u.es_admin
    return c


class ConfirmarRequest(BaseModel):
    ids: list[str] = Field(..., min_length=1, max_length=MAX_DOCS_CARPETA)
    consultas_esperadas: int = Field(..., ge=0)


@router.post("/boveda/carpeta/{carpeta_id}/confirmar")
async def boveda_confirmar(carpeta_id: str, req: ConfirmarRequest, authorization: str = Header(None)):
    """Cobra la ingesta y la pone en cola. Si la cotización cambió desde que
    el abogado la vio, no cobra: devuelve la nueva (409)."""
    u = await cobro.exigir_platinum(authorization)
    await _carpeta_propia(u.id, carpeta_id)
    r = await _db(lambda sb: sb.table("boveda_documentos").select("*")
                  .in_("id", req.ids).eq("user_id", u.id).eq("expediente_id", carpeta_id)
                  .eq("estado", "revisado").execute())
    docs = r.data or []
    if not docs:
        raise HTTPException(status_code=409, detail={"codigo": "nada_que_cobrar",
                                                      "mensaje": "Esos documentos ya no esperan confirmación."})
    uso = await _uso_usuario(u.id)
    c = _cotizacion(docs, None, uso)
    if c["excede_cuota"]:
        raise HTTPException(status_code=422, detail={"codigo": "cuota_paginas", "mensaje": c["mensaje_cuota"]})
    if c["consultas"] != req.consultas_esperadas or len(docs) != len(req.ids):
        raise HTTPException(status_code=409, detail={"codigo": "cotizacion_cambio",
                                                      "mensaje": "La cotización cambió; revísala de nuevo.",
                                                      "cotizacion": c})
    consultas = 0 if u.es_admin else c["consultas"]
    cobro_res = {"movimiento_id": None, "disponibles": None}
    if consultas:
        cobro_res = await cobro.cobrar_consultas(
            u.id, consultas, f"boveda:ingesta:{carpeta_id}:{len(docs)}docs:{c['paginas']}pag")
        if not cobro_res.get("allowed"):
            raise cobro.error_de_saldo(cobro_res)
    tanda_id = str(uuid.uuid4())
    try:
        await _db(lambda sb: sb.table("boveda_tandas").insert({
            "id": tanda_id, "user_id": u.id, "expediente_id": carpeta_id, "tipo": "ingesta",
            "unidades": c["paginas"], "consultas": consultas,
            "movimiento_id": cobro_res.get("movimiento_id")}).execute())
        upd = await _db(lambda sb: sb.table("boveda_documentos")
                        .update({"estado": "en_cola", "tanda_id": tanda_id, "intentos": 0,
                                 "error": None, "actualizado_en": _ahora()})
                        .in_("id", [d["id"] for d in docs]).eq("estado", "revisado").execute())
    except Exception:
        await cobro.devolver_consultas(cobro_res.get("movimiento_id"))
        raise
    puestos = upd.data or []
    if len(puestos) < len(docs):
        # Dos clics a la vez: lo que tomó el otro no se cobra aquí.
        justas = consultas_ingesta(sum(int(d.get("paginas_ocr") or 0) for d in puestos),
                                   sum(int(d.get("paginas") or 0) - int(d.get("paginas_ocr") or 0) for d in puestos))
        sobran = consultas - (0 if u.es_admin else justas)
        if sobran > 0:
            await cobro.devolver_consultas(cobro_res.get("movimiento_id"), sobran)
            consultas -= sobran
            await _db(lambda sb: sb.table("boveda_tandas").update({"consultas": consultas})
                      .eq("id", tanda_id).execute())
    _avisar_trabajador()
    return {"tanda_id": tanda_id, "documentos": len(puestos), "consultas_cobradas": consultas,
            "disponibles": cobro_res.get("disponibles")}


# ═══ Documentos: ver y borrar ══════════════════════════════════════════════
@router.get("/boveda/documento/{doc_id}")
async def boveda_documento(doc_id: str, authorization: str = Header(None)):
    """Metadatos y una URL firmada (1 h) para abrir el archivo en el visor."""
    u = await cobro.exigir_platinum(authorization)
    d = await _documento_propio(u.id, doc_id)
    s = await _db(lambda sb: sb.storage.from_(CUBO).create_signed_url(d["storage_path"], 3600))
    return _doc_publico(d) | {"url": s.get("signedURL") or s.get("signedUrl") or s.get("signed_url")}


@router.get("/boveda/documento/{doc_id}/paginas")
async def boveda_documento_paginas(doc_id: str, authorization: str = Header(None)):
    """El texto de cada página (para Word, que no tiene páginas de verdad, y
    para resaltar el pasaje citado)."""
    u = await cobro.exigir_platinum(authorization)
    d = await _documento_propio(u.id, doc_id)
    if d["estado"] != "listo":
        raise HTTPException(status_code=409, detail="El documento todavía no está listo.")
    return {"id": d["id"], "nombre": d["nombre"], "paginas": await _leer_paginas(d)}


async def _purgar_ya() -> None:
    try:
        await _purgar_pendientes()
    except Exception as e:  # el barrendero del trabajador lo reintenta
        print(f"   🗄️ Bóveda: purga inmediata falló ({type(e).__name__}); queda en cola")


@router.delete("/boveda/documento/{doc_id}")
async def boveda_borrar_documento(doc_id: str, authorization: str = Header(None)):
    u = await cobro.usuario_del_token(authorization)   # borrar lo suyo no exige Platinum
    await _documento_propio(u.id, doc_id)
    await _db(lambda sb: sb.table("boveda_documentos").delete().eq("id", doc_id).eq("user_id", u.id).execute())
    await _purgar_ya()
    return {"borrado": True}


@router.delete("/boveda/carpeta/{carpeta_id}")
async def boveda_borrar_carpeta(carpeta_id: str, authorization: str = Header(None)):
    """Vacía la Bóveda de la carpeta (la carpeta y sus demás gavetas se quedan)."""
    u = await cobro.usuario_del_token(authorization)
    await _carpeta_propia(u.id, carpeta_id)
    await _db(lambda sb: sb.table("boveda_tablas").delete().eq("expediente_id", carpeta_id).eq("user_id", u.id).execute())
    r = await _db(lambda sb: sb.table("boveda_documentos").delete().eq("expediente_id", carpeta_id)
                  .eq("user_id", u.id).execute())
    await _purgar_ya()
    try:
        await bi.borrar(user_id=u.id, carpeta_id=carpeta_id)
    except Exception:
        pass
    return {"borrados": len(r.data or [])}


@router.delete("/boveda/todo")
async def boveda_borrar_todo(authorization: str = Header(None)):
    """Toda la Bóveda del usuario: documentos, tablas, vectores y archivos."""
    u = await cobro.usuario_del_token(authorization)
    await borrar_boveda_de_usuario(u.id)
    return {"borrado": True}


async def borrar_boveda_de_usuario(user_id: str) -> None:
    """También la llama /cuenta/eliminar."""
    await _db(lambda sb: sb.table("boveda_tablas").delete().eq("user_id", user_id).execute())
    await _db(lambda sb: sb.table("boveda_documentos").delete().eq("user_id", user_id).execute())
    await _purgar_ya()
    try:
        await bi.borrar(user_id=user_id)
    except Exception as e:
        print(f"   🗄️ Bóveda: vectores de {user_id[:8]} quedan para el barrendero ({type(e).__name__})")


# ═══ Buscar (sin modelo: no cobra) ═════════════════════════════════════════
class BuscarRequest(BaseModel):
    pregunta: str = Field(..., min_length=2, max_length=2000)
    limite: int = Field(12, ge=1, le=40)


@router.post("/boveda/carpeta/{carpeta_id}/buscar")
async def boveda_buscar(carpeta_id: str, req: BuscarRequest, authorization: str = Header(None)):
    u = await cobro.exigir_platinum(authorization)
    await _carpeta_propia(u.id, carpeta_id)
    hs = await bi.buscar(req.pregunta, user_id=u.id, carpeta_id=carpeta_id, limite=req.limite)
    return {"resultados": [h.__dict__ for h in hs]}


async def fuentes_para_chat(authorization: Optional[str], carpeta_id: str, pregunta: str,
                            limite: int = 12) -> tuple[list[bi.Hallazgo], dict]:
    """Para /chat: los fragmentos de la Bóveda de ESA carpeta del dueño del
    token. Lanza HTTPException si no hay token, no es Platinum o la carpeta no
    es suya: el chat lo atrapa y sigue sin Bóveda."""
    u = await cobro.exigir_platinum(authorization)
    carpeta = await _carpeta_propia(u.id, carpeta_id)
    return await bi.buscar(pregunta, user_id=u.id, carpeta_id=carpeta_id, limite=limite), carpeta


# ═══ Tablas de revisión ════════════════════════════════════════════════════
class Columna(BaseModel):
    id: Optional[str] = Field(None, max_length=40)
    titulo: str = Field(..., min_length=1, max_length=80)
    pregunta: str = Field(..., min_length=3, max_length=600)
    tipo: str = Field("texto")


def _columnas_limpias(cols: list[Columna]) -> list[dict]:
    if len(cols) > MAX_COLUMNAS:
        raise HTTPException(status_code=422, detail=f"Máximo {MAX_COLUMNAS} columnas.")
    out, vistos = [], set()
    for c in cols:
        cid = (c.id or "").strip() or uuid.uuid4().hex[:8]
        if cid in vistos:
            cid = uuid.uuid4().hex[:8]
        vistos.add(cid)
        out.append({"id": cid, "titulo": c.titulo.strip(), "pregunta": c.pregunta.strip(),
                    "tipo": c.tipo if c.tipo in btab.TIPOS else "texto"})
    return out


@router.get("/boveda/columnas-sugeridas")
async def boveda_columnas_sugeridas():
    return {"columnas": btab.COLUMNAS_SUGERIDAS, "tipos": btab.TIPOS}


class TablaRequest(BaseModel):
    titulo: str = Field("Tabla de revisión", min_length=1, max_length=120)
    columnas: list[Columna] = Field(default_factory=list)


@router.post("/boveda/carpeta/{carpeta_id}/tablas")
async def boveda_crear_tabla(carpeta_id: str, req: TablaRequest, authorization: str = Header(None)):
    u = await cobro.exigir_platinum(authorization)
    await _carpeta_propia(u.id, carpeta_id)
    cols = _columnas_limpias(req.columnas)
    r = await _db(lambda sb: sb.table("boveda_tablas").insert({
        "user_id": u.id, "expediente_id": carpeta_id, "titulo": req.titulo.strip(), "columnas": cols}).execute())
    return r.data[0]


class TablaCambio(BaseModel):
    titulo: Optional[str] = Field(None, min_length=1, max_length=120)
    columnas: Optional[list[Columna]] = None


@router.patch("/boveda/tablas/{tabla_id}")
async def boveda_editar_tabla(tabla_id: str, req: TablaCambio, authorization: str = Header(None)):
    """Cambiar título o columnas. Una columna borrada se lleva sus celdas; una
    columna cuya PREGUNTA cambió deja sus celdas como «desactualizadas»."""
    u = await cobro.exigir_platinum(authorization)
    t = await _tabla_propia(u.id, tabla_id)
    cambio: dict[str, Any] = {"actualizado_en": _ahora()}
    if req.titulo is not None:
        cambio["titulo"] = req.titulo.strip()
    if req.columnas is not None:
        nuevas = _columnas_limpias(req.columnas)
        viejas = {c["id"]: c for c in (t.get("columnas") or [])}
        quedan = {c["id"] for c in nuevas}
        quitar = [cid for cid in viejas if cid not in quedan]
        if quitar:
            await _db(lambda sb: sb.table("boveda_celdas").delete().eq("tabla_id", tabla_id)
                      .in_("columna_id", quitar).execute())
        cambiadas = [c["id"] for c in nuevas if c["id"] in viejas and viejas[c["id"]]["pregunta"] != c["pregunta"]]
        if cambiadas:
            # La respuesta vieja contesta otra pregunta: se borra para no confundir.
            await _db(lambda sb: sb.table("boveda_celdas").delete().eq("tabla_id", tabla_id)
                      .in_("columna_id", cambiadas).neq("estado", "procesando").execute())
        cambio["columnas"] = nuevas
    r = await _db(lambda sb: sb.table("boveda_tablas").update(cambio).eq("id", tabla_id).execute())
    return r.data[0] if r.data else t


@router.delete("/boveda/tablas/{tabla_id}")
async def boveda_borrar_tabla(tabla_id: str, authorization: str = Header(None)):
    u = await cobro.usuario_del_token(authorization)
    await _tabla_propia(u.id, tabla_id)
    await _db(lambda sb: sb.table("boveda_tablas").delete().eq("id", tabla_id).eq("user_id", u.id).execute())
    return {"borrada": True}


async def _celdas_de(tabla_id: str) -> list[dict]:
    out, desde = [], 0
    while True:   # una tabla de 500 × 20 son 10,000 celdas: se pagina
        r = await _db(lambda sb: sb.table("boveda_celdas")
                      .select("documento_id, columna_id, estado, respuesta, actualizado_en")
                      .eq("tabla_id", tabla_id).order("id").range(desde, desde + 999).execute())
        out.extend(r.data or [])
        if len(r.data or []) < 1000:
            return out
        desde += 1000


async def _documentos_listos(user_id: str, carpeta_id: str) -> list[dict]:
    r = await _db(lambda sb: sb.table("boveda_documentos")
                  .select("id, nombre, paginas, mime_type, storage_path, estado")
                  .eq("expediente_id", carpeta_id).eq("user_id", user_id).eq("estado", "listo")
                  .order("nombre").limit(MAX_DOCS_CARPETA + 100).execute())
    return r.data or []


@router.get("/boveda/tablas/{tabla_id}")
async def boveda_tabla(tabla_id: str, authorization: str = Header(None)):
    u = await cobro.exigir_platinum(authorization)
    t = await _tabla_propia(u.id, tabla_id)
    docs = await _documentos_listos(u.id, t["expediente_id"])
    celdas = await _celdas_de(tabla_id)
    return {
        "tabla": {k: t[k] for k in ("id", "titulo", "columnas", "expediente_id", "actualizado_en")},
        "filas": [{"id": d["id"], "nombre": d["nombre"], "paginas": d["paginas"],
                   "aproximadas": bt.tipo_de(d["nombre"], d.get("mime_type")) == "docx"} for d in docs],
        "celdas": celdas,
        "pendientes": sum(1 for c in celdas if c["estado"] in ("en_cola", "procesando")),
    }


class CorridaRequest(BaseModel):
    doc_ids: Optional[list[str]] = Field(None, max_length=MAX_DOCS_CARPETA)
    columna_ids: Optional[list[str]] = Field(None, max_length=MAX_COLUMNAS)
    rehacer: bool = False
    consultas_esperadas: Optional[int] = None


async def _celdas_a_correr(u, t: dict, req: CorridaRequest) -> tuple[list[tuple[str, str]], list[dict]]:
    docs = await _documentos_listos(u.id, t["expediente_id"])
    if req.doc_ids:
        quiero = set(req.doc_ids)
        docs = [d for d in docs if d["id"] in quiero]
    cols = [c for c in (t.get("columnas") or []) if not req.columna_ids or c["id"] in req.columna_ids]
    existentes = {(c["documento_id"], c["columna_id"]): c["estado"] for c in await _celdas_de(t["id"])}
    pares = []
    for d in docs:
        for c in cols:
            estado = existentes.get((d["id"], c["id"]))
            if estado in ("en_cola", "procesando"):
                continue
            if estado == "lista" and not req.rehacer:
                continue
            pares.append((d["id"], c["id"]))
    return pares, docs


@router.post("/boveda/tablas/{tabla_id}/cotizar")
async def boveda_tabla_cotizar(tabla_id: str, req: CorridaRequest, authorization: str = Header(None)):
    u = await cobro.exigir_platinum(authorization)
    t = await _tabla_propia(u.id, tabla_id)
    pares, _ = await _celdas_a_correr(u, t, req)
    consultas = consultas_celdas(len(pares))
    disp = None if u.es_admin else (await cobro.consultas_disponibles(u.id)).get("disponibles")
    return {"celdas": len(pares), "consultas": consultas, "disponibles": disp,
            "alcanza": disp is None or disp >= consultas, "sin_costo": u.es_admin}


@router.post("/boveda/tablas/{tabla_id}/ejecutar")
async def boveda_tabla_ejecutar(tabla_id: str, req: CorridaRequest, authorization: str = Header(None)):
    u = await cobro.exigir_platinum(authorization)
    t = await _tabla_propia(u.id, tabla_id)
    pares, _ = await _celdas_a_correr(u, t, req)
    if not pares:
        return {"celdas": 0, "consultas_cobradas": 0}
    consultas = consultas_celdas(len(pares))
    if req.consultas_esperadas is not None and req.consultas_esperadas != consultas:
        raise HTTPException(status_code=409, detail={"codigo": "cotizacion_cambio",
                                                      "mensaje": "La cotización cambió; revísala de nuevo.",
                                                      "cotizacion": {"celdas": len(pares), "consultas": consultas}})
    cobradas = 0 if u.es_admin else consultas
    cobro_res = {"movimiento_id": None, "disponibles": None}
    if cobradas:
        cobro_res = await cobro.cobrar_consultas(u.id, cobradas, f"boveda:tabla:{tabla_id}:{len(pares)}celdas")
        if not cobro_res.get("allowed"):
            raise cobro.error_de_saldo(cobro_res)
    tanda_id = str(uuid.uuid4())
    try:
        await _db(lambda sb: sb.table("boveda_tandas").insert({
            "id": tanda_id, "user_id": u.id, "expediente_id": t["expediente_id"], "tipo": "tabla",
            "tabla_id": tabla_id, "unidades": len(pares), "consultas": cobradas,
            "movimiento_id": cobro_res.get("movimiento_id")}).execute())
        filas = [{"tabla_id": tabla_id, "documento_id": d, "columna_id": c, "user_id": u.id,
                  "estado": "en_cola", "respuesta": None, "tanda_id": tanda_id, "intentos": 0,
                  "actualizado_en": _ahora()} for d, c in pares]
        for i in range(0, len(filas), 500):
            lote = filas[i:i + 500]
            await _db(lambda sb: sb.table("boveda_celdas")
                      .upsert(lote, on_conflict="tabla_id,documento_id,columna_id").execute())
    except Exception:
        await cobro.devolver_consultas(cobro_res.get("movimiento_id"))
        raise
    _avisar_trabajador()
    return {"tanda_id": tanda_id, "celdas": len(pares), "consultas_cobradas": cobradas,
            "disponibles": cobro_res.get("disponibles")}


# ── Exportar ───────────────────────────────────────────────────────────────
def _texto_celda(resp: Optional[dict], estado: str) -> str:
    if estado in ("en_cola", "procesando"):
        return "(pendiente)"
    if not resp:
        return ""
    if resp.get("error"):
        return "(no se pudo responder)"
    return str(resp.get("valor") or "")


def _cita_celda(resp: Optional[dict]) -> str:
    if not resp or not resp.get("cita"):
        return ""
    p = f", p. {resp['pagina']}" if resp.get("pagina") else ""
    v = "" if resp.get("verificada") else " [cita no verificada]"
    return f"«{resp['cita']}»{p}{v}"


def _xlsx(titulo: str, columnas: list[dict], filas: list[dict], celdas: dict) -> bytes:
    from openpyxl import Workbook
    from openpyxl.comments import Comment
    from openpyxl.styles import Alignment, Font, PatternFill
    wb = Workbook()
    ws = wb.active
    ws.title = "Revisión"
    ws.append(["Documento"] + [c["titulo"] for c in columnas])
    for celda in ws[1]:
        celda.font = Font(bold=True, color="FFFFFF")
        celda.fill = PatternFill("solid", fgColor="1F1F1F")
        celda.alignment = Alignment(wrap_text=True, vertical="top")
    for f in filas:
        fila = [f["nombre"]]
        for c in columnas:
            x = celdas.get((f["id"], c["id"]), {})
            fila.append(_texto_celda(x.get("respuesta"), x.get("estado", "")))
        ws.append(fila)
        r = ws.max_row
        for j, c in enumerate(columnas, start=2):
            x = celdas.get((f["id"], c["id"]), {})
            cita = _cita_celda(x.get("respuesta"))
            if cita:
                ws.cell(row=r, column=j).comment = Comment(cita[:2000], "Iurexia")
    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = 42 if col[0].column > 1 else 36
        for celda in col:
            celda.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "B2"
    citas = wb.create_sheet("Citas")
    citas.append(["Documento", "Columna", "Respuesta", "Cita literal", "Página", "Verificada"])
    for celda in citas[1]:
        celda.font = Font(bold=True)
    for f in filas:
        for c in columnas:
            x = celdas.get((f["id"], c["id"]), {})
            resp = x.get("respuesta") or {}
            citas.append([f["nombre"], c["titulo"], _texto_celda(resp, x.get("estado", "")),
                          resp.get("cita") or "", resp.get("pagina"),
                          "Sí" if resp.get("verificada") else ("No" if resp.get("cita") else "")])
    for letra, ancho in zip("ABCDEF", (36, 22, 40, 70, 8, 11)):
        citas.column_dimensions[letra].width = ancho
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _docx(titulo: str, carpeta: str, columnas: list[dict], filas: list[dict], celdas: dict) -> bytes:
    import docx
    from docx.enum.section import WD_ORIENT
    from docx.shared import Pt, RGBColor
    d = docx.Document()
    s = d.sections[0]
    s.orientation = WD_ORIENT.LANDSCAPE
    s.page_width, s.page_height = s.page_height, s.page_width
    d.add_heading(titulo, level=1)
    if carpeta:
        d.add_paragraph(f"Carpeta: {carpeta}")
    tabla = d.add_table(rows=1, cols=len(columnas) + 1)
    tabla.style = "Table Grid"
    enc = tabla.rows[0].cells
    enc[0].text = "Documento"
    for j, c in enumerate(columnas, start=1):
        enc[j].text = c["titulo"]
    for celda in enc:
        for p in celda.paragraphs:
            for run in p.runs:
                run.bold = True
    for f in filas:
        cs = tabla.add_row().cells
        cs[0].text = f["nombre"]
        for j, c in enumerate(columnas, start=1):
            x = celdas.get((f["id"], c["id"]), {})
            resp = x.get("respuesta") or {}
            p = cs[j].paragraphs[0]
            p.add_run(_texto_celda(resp, x.get("estado", "")))
            if resp.get("cita"):
                q = cs[j].add_paragraph()
                r = q.add_run(f"«{resp['cita']}»" + (f" (p. {resp['pagina']})" if resp.get("pagina") else ""))
                r.italic = True
                r.font.size = Pt(8)
                if not resp.get("verificada"):
                    a = q.add_run("  cita no verificada")
                    a.font.size = Pt(8)
                    a.font.color.rgb = RGBColor(0xB0, 0x40, 0x20)
    for fila in tabla.rows:
        for celda in fila.cells:
            for p in celda.paragraphs:
                for run in p.runs:
                    if run.font.size is None:
                        run.font.size = Pt(9)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


@router.get("/boveda/tablas/{tabla_id}/exportar")
async def boveda_tabla_exportar(tabla_id: str, formato: str = Query("xlsx", pattern="^(xlsx|docx)$"),
                                authorization: str = Header(None)):
    u = await cobro.exigir_platinum(authorization)
    t = await _tabla_propia(u.id, tabla_id)
    carpeta = await _carpeta_propia(u.id, t["expediente_id"])
    docs = await _documentos_listos(u.id, t["expediente_id"])
    celdas = {(c["documento_id"], c["columna_id"]): c for c in await _celdas_de(tabla_id)}
    cols = t.get("columnas") or []
    nombre_carpeta = carpeta.get("titulo") or carpeta.get("cliente_nombre") or ""
    base = "".join(ch if ch.isalnum() or ch in " -_" else "_" for ch in t["titulo"])[:60].strip() or "tabla"
    if formato == "xlsx":
        datos = await asyncio.to_thread(_xlsx, t["titulo"], cols, docs, celdas)
        tipo = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        datos = await asyncio.to_thread(_docx, t["titulo"], nombre_carpeta, cols, docs, celdas)
        tipo = bt.MIME_DOCX
    return Response(content=datos, media_type=tipo,
                    headers={"Content-Disposition": f'attachment; filename="{base}.{formato}"'})


# ═══ EL TRABAJADOR ═════════════════════════════════════════════════════════
_TAREAS: set = set()
_sem_ocr: Optional[asyncio.Semaphore] = None


def _sembrar(coro) -> asyncio.Task:
    t = asyncio.ensure_future(coro)
    _TAREAS.add(t)
    t.add_done_callback(_TAREAS.discard)
    return t


async def _descargar(doc: dict) -> bytes:
    try:
        return await _db(lambda sb: sb.storage.from_(CUBO).download(doc["storage_path"]))
    except Exception as e:
        if "not found" in str(e).lower() or "404" in str(e):
            raise Rechazo("El archivo no llegó completo; vuelve a subirlo.")
        raise


async def _revisar(doc: dict) -> None:
    """Cuenta páginas y cuáles irán a OCR. No paga nada."""
    datos = await _descargar(doc)
    huella = hashlib.sha256(datos).hexdigest()
    dup = await _db(lambda sb: sb.table("boveda_documentos").select("id, nombre")
                    .eq("expediente_id", doc["expediente_id"]).eq("huella", huella)
                    .neq("id", doc["id"]).limit(1).execute())
    if dup.data:
        raise Rechazo(f"Ya está en la Bóveda de esta carpeta como «{dup.data[0]['nombre']}».")
    try:
        ins = await asyncio.to_thread(bt.inspeccionar, datos, doc["nombre"], doc.get("mime_type"))
    except Exception:
        raise Rechazo("No se pudo abrir el archivo: puede estar dañado o protegido con contraseña.")
    if ins.paginas > MAX_PAGINAS_DOC:
        raise Rechazo(f"Tiene {ins.paginas:,} páginas; el máximo por documento es {MAX_PAGINAS_DOC:,}.")
    if ins.aproximadas and not any(t.strip() for t in ins.textos):
        raise Rechazo("El documento de Word no tiene texto.")
    await _db(lambda sb: sb.table("boveda_documentos").update({
        "huella": huella, "tamano": len(datos), "paginas": ins.paginas, "paginas_ocr": ins.n_ocr,
        "estado": "revisado", "error": None, "tomado_en": None, "actualizado_en": _ahora(),
    }).eq("id", doc["id"]).execute())


async def _ingerir(doc: dict) -> None:
    global _sem_ocr
    t0 = time.perf_counter()
    datos = await _descargar(doc)
    if doc.get("huella") and hashlib.sha256(datos).hexdigest() != doc["huella"]:
        raise Rechazo("El archivo cambió después de cotizarlo; súbelo de nuevo.")
    ins = await asyncio.to_thread(bt.inspeccionar, datos, doc["nombre"], doc.get("mime_type"))
    paginas = list(ins.textos)
    motor = "texto"
    if ins.paginas_ocr:
        # El mismo archivo ya leído en OTRA carpeta del mismo usuario: se reutiliza.
        previo = await _db(lambda sb: sb.table("boveda_documentos").select("id, storage_path")
                           .eq("user_id", doc["user_id"]).eq("huella", doc.get("huella") or "-")
                           .eq("estado", "listo").neq("id", doc["id"]).limit(1).execute())
        leidas = None
        if previo.data:
            try:
                leidas = await _leer_paginas(previo.data[0])
                motor = "reutilizado"
            except Exception:
                leidas = None
        if leidas and len(leidas) == len(paginas):
            paginas = leidas
        else:
            if _sem_ocr is None:
                _sem_ocr = asyncio.Semaphore(OCR_A_LA_VEZ)
            async with _sem_ocr:
                ocr = await bt.ocr_azure(datos, ins.paginas_ocr)
            for n, t in ocr.items():
                if 1 <= n <= len(paginas):
                    paginas[n - 1] = t
            motor = "azure"
    if not any(p.strip() for p in paginas):
        raise Rechazo("No se encontró texto legible en el documento.")
    await _guardar_paginas(doc, paginas, motor)
    frags = bt.fragmentar(paginas)
    uso = bi.Uso()
    n = await bi.indexar_documento(user_id=doc["user_id"], carpeta_id=doc["expediente_id"], doc_id=doc["id"],
                                   nombre=doc["nombre"], fragmentos=frags, uso=uso)
    await _db(lambda sb: sb.table("boveda_documentos").update({
        "estado": "listo", "fragmentos": n, "caracteres": sum(len(p) for p in paginas),
        "error": None, "tomado_en": None, "actualizado_en": _ahora(),
    }).eq("id", doc["id"]).execute())
    print(f"   🗄️ Bóveda: {doc['id'][:8]} listo · {ins.paginas} pág ({ins.n_ocr} OCR, {motor}) · "
          f"{n} fragmentos · {uso.tokens} tok · {time.perf_counter() - t0:.1f}s · usuario {doc['user_id'][:8]}")


async def _procesar_documento(doc: dict) -> None:
    revisando = doc["estado"] == "revisando"
    try:
        if revisando:
            await _revisar(doc)
        else:
            await _ingerir(doc)
    except Rechazo as r:
        await _db(lambda sb: sb.table("boveda_documentos").update({
            "estado": "error", "error": str(r), "tomado_en": None, "actualizado_en": _ahora(),
        }).eq("id", doc["id"]).execute())
    except Exception as e:  # nuestro: se reintenta hasta 3 veces
        print(f"   🗄️ Bóveda: {doc['id'][:8]} falló ({type(e).__name__}: {str(e)[:160]}) intento {doc.get('intentos')}")
        agotado = int(doc.get("intentos") or 0) >= 3
        await _db(lambda sb: sb.table("boveda_documentos").update({
            "estado": "error" if agotado else ("subido" if revisando else "en_cola"),
            "error": "No se pudo procesar este documento." if agotado else None,
            "tomado_en": None, "actualizado_en": _ahora(),
        }).eq("id", doc["id"]).execute())
    if not revisando and doc.get("tanda_id"):
        await _cerrar_tanda(doc["tanda_id"])


async def _procesar_fila(tabla_id: str, doc_id: str, columna_ids: list[str]) -> None:
    tanda_ids: set = set()
    try:
        celdas = (await _db(lambda sb: sb.table("boveda_celdas").select("columna_id, tanda_id")
                            .eq("tabla_id", tabla_id).eq("documento_id", doc_id)
                            .in_("columna_id", columna_ids).execute())).data or []
        tanda_ids = {c["tanda_id"] for c in celdas if c.get("tanda_id")}
    except Exception:
        pass
    try:
        t = (await _db(lambda sb: sb.table("boveda_tablas").select("*").eq("id", tabla_id).limit(1).execute())).data
        d = (await _db(lambda sb: sb.table("boveda_documentos").select("*").eq("id", doc_id).limit(1).execute())).data
        if not t or not d or d[0]["estado"] != "listo":
            raise Rechazo("El documento ya no está en la Bóveda.")
        t, d = t[0], d[0]
        cols = [c for c in (t.get("columnas") or []) if c["id"] in columna_ids]
        if not cols:
            return
        paginas = await _leer_paginas(d)

        async def buscar(pregunta: str):
            return await bi.buscar(pregunta, user_id=d["user_id"], carpeta_id=d["expediente_id"],
                                   doc_ids=[d["id"]], limite=10)

        fila = await btab.responder_fila(bi._cliente_openai(), nombre=d["nombre"], paginas=paginas, columnas=cols,
                                         aproximadas=bt.tipo_de(d["nombre"], d.get("mime_type")) == "docx",
                                         buscar=buscar)
        n = max(1, len(fila.celdas))
        for c in fila.celdas:
            payload = {
                "estado": "error" if c.error else "lista", "respuesta": c.a_json(),
                "tokens_entrada": fila.tokens_entrada // n, "tokens_salida": fila.tokens_salida // n,
                "costo_usd": round(fila.usd / n, 6), "tomado_en": None, "actualizado_en": _ahora(),
            }
            await _db(lambda sb, p=payload, cid=c.columna_id: sb.table("boveda_celdas").update(p)
                      .eq("tabla_id", tabla_id).eq("documento_id", doc_id).eq("columna_id", cid).execute())
        print(f"   🗄️ Bóveda tabla {tabla_id[:8]} · {d['nombre'][:30]} · {len(cols)} col · "
              f"{fila.tokens_entrada}→{fila.tokens_salida} tok · {fila.usd:.5f} USD · {fila.segundos}s")
    except Exception as e:
        motivo = str(e) if isinstance(e, Rechazo) else "No se pudo responder."
        if not isinstance(e, Rechazo):
            print(f"   🗄️ Bóveda tabla {tabla_id[:8]} fila {doc_id[:8]}: {type(e).__name__}: {str(e)[:160]}")
        await _db(lambda sb: sb.table("boveda_celdas").update({
            "estado": "error", "respuesta": {"error": motivo}, "tomado_en": None, "actualizado_en": _ahora(),
        }).eq("tabla_id", tabla_id).eq("documento_id", doc_id).in_("columna_id", columna_ids)
            .eq("estado", "procesando").execute())
    for tid in tanda_ids:
        await _cerrar_tanda(tid)


async def _cerrar_tanda(tanda_id: str) -> None:
    """Cuando ya no queda nada pendiente de una tanda, se devuelve lo cobrado
    de más (documentos o celdas que fallaron). Lo marca terminada PRIMERO y
    con condición, para que dos procesos no devuelvan dos veces."""
    try:
        t = (await _db(lambda sb: sb.table("boveda_tandas").select("*").eq("id", tanda_id).limit(1).execute())).data
        if not t or t[0]["estado"] == "terminada":
            return
        t = t[0]
        if t["tipo"] == "ingesta":
            r = await _db(lambda sb: sb.table("boveda_documentos").select("estado, paginas, paginas_ocr")
                          .eq("tanda_id", tanda_id).execute())
            filas = r.data or []
            if any(f["estado"] in ("en_cola", "procesando") for f in filas):
                return
            ok = [f for f in filas if f["estado"] == "listo"]
            ocr = sum(int(f.get("paginas_ocr") or 0) for f in ok)
            justas = consultas_ingesta(ocr, sum(int(f.get("paginas") or 0) for f in ok) - ocr)
        else:
            r = await _db(lambda sb: sb.table("boveda_celdas").select("estado").eq("tanda_id", tanda_id).execute())
            filas = r.data or []
            if any(f["estado"] in ("en_cola", "procesando") for f in filas):
                return
            justas = consultas_celdas(sum(1 for f in filas if f["estado"] == "lista"))
        devolver = max(0, int(t["consultas"]) - int(t["devueltas"] or 0) - (justas if t["consultas"] else 0))
        marcada = await _db(lambda sb: sb.table("boveda_tandas").update({
            "estado": "terminada", "terminado_en": _ahora(), "devueltas": int(t["devueltas"] or 0) + devolver,
        }).eq("id", tanda_id).eq("estado", "en_curso").execute())
        if not marcada.data:
            return   # otro proceso la cerró
        if devolver and t.get("movimiento_id"):
            await cobro.devolver_consultas(t["movimiento_id"], devolver)
            print(f"   🗄️ Bóveda: tanda {tanda_id[:8]} devuelve {devolver} consulta(s) de {t['consultas']}")
    except Exception as e:
        print(f"   ⚠️ Bóveda: no se pudo cerrar la tanda {tanda_id[:8]} ({type(e).__name__}: {str(e)[:160]})")


async def _purgar_pendientes(lote: int = 50) -> int:
    """Borra de Qdrant y de Storage lo que el disparador dejó en boveda_purgas."""
    r = await _db(lambda sb: sb.table("boveda_purgas").select("*").is_("hecho_en", "null")
                  .order("id").limit(lote).execute())
    filas = r.data or []
    if not filas:
        return 0
    rutas = []
    for f in filas:
        if f.get("documento_id"):
            await bi.borrar(user_id=f["user_id"], doc_id=f["documento_id"])
        if f.get("storage_path"):
            rutas += [f["storage_path"], _ruta_paginas(f["storage_path"])]
    for i in range(0, len(rutas), 100):
        await _db(lambda sb, rr=rutas[i:i + 100]: sb.storage.from_(CUBO).remove(rr))
    await _db(lambda sb: sb.table("boveda_purgas").update({"hecho_en": _ahora()})
              .in_("id", [f["id"] for f in filas]).execute())
    return len(filas)


async def _mantenimiento() -> None:
    try:
        while await _purgar_pendientes():
            pass
    except Exception as e:
        print(f"   🗄️ Bóveda: purga pendiente ({type(e).__name__}: {str(e)[:120]})")
    try:
        # Reservas que nunca se subieron (el navegador se cerró a media subida).
        await _db(lambda sb: sb.table("boveda_documentos").delete().eq("estado", "subiendo")
                  .lt("creado_en", datetime.fromtimestamp(time.time() - 6 * 3600, timezone.utc).isoformat())
                  .execute())
    except Exception:
        pass


async def _bucle() -> None:
    """Un despachador por proceso: toma trabajo mientras tenga lugares libres;
    sin trabajo, duerme cada vez más (hasta 30 s) salvo que lo despierten."""
    global _despertar
    _despertar = asyncio.Event()
    docs_activos: set = set()
    filas_activas: set = set()
    espera, ultimo_mant = 1.0, 0.0
    sin_tablas = False
    print(f"   🗄️ Bóveda: trabajador {TRABAJADOR} en marcha")
    while True:
        hubo = False
        try:
            while len(docs_activos) < DOCS_A_LA_VEZ:
                r = await _db(lambda sb: sb.rpc("boveda_tomar_documento", {"p_trabajador": TRABAJADOR}).execute())
                if not r.data:
                    break
                tarea = _sembrar(_procesar_documento(r.data[0]))
                docs_activos.add(tarea)
                tarea.add_done_callback(lambda t: (docs_activos.discard(t), _avisar_trabajador()))
                hubo = True
            while len(filas_activas) < FILAS_A_LA_VEZ:
                r = await _db(lambda sb: sb.rpc("boveda_tomar_fila", {"p_trabajador": TRABAJADOR}).execute())
                if not r.data or not r.data[0].get("tabla_id") or not r.data[0].get("columnas"):
                    break
                f = r.data[0]
                tarea = _sembrar(_procesar_fila(f["tabla_id"], f["documento_id"], f["columnas"] or []))
                filas_activas.add(tarea)
                tarea.add_done_callback(lambda t: (filas_activas.discard(t), _avisar_trabajador()))
                hubo = True
            if time.time() - ultimo_mant > 60:
                ultimo_mant = time.time()
                await _mantenimiento()
            sin_tablas = False
        except Exception as e:
            texto = str(e)
            if "PGRST202" in texto or "does not exist" in texto or "boveda_" in texto and "404" in texto:
                if not sin_tablas:
                    print("   🗄️ Bóveda: faltan las tablas o funciones (migration_boveda.sql); reviso cada 10 min")
                sin_tablas = True
                await asyncio.sleep(600)
                continue
            print(f"   🗄️ Bóveda: el despachador tropezó ({type(e).__name__}: {texto[:160]})")
        espera = 1.0 if hubo else min(espera * 2, ESPERA_MAXIMA_S)
        _despertar.clear()
        try:
            await asyncio.wait_for(_despertar.wait(), timeout=espera)
            espera = 1.0
        except asyncio.TimeoutError:
            pass


def arrancar() -> None:
    """main.py lo llama al arrancar cada proceso. BOVEDA_TRABAJADOR=0 lo apaga."""
    if os.getenv("BOVEDA_TRABAJADOR", "1") != "1":
        print("   🗄️ Bóveda: trabajador apagado (BOVEDA_TRABAJADOR=0)")
        return
    _sembrar(_bucle())
