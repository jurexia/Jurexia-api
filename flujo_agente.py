"""El agente de los flujos de trabajo (25-sep-2026).

David, al probar el primer flujo: «eso no es workflow, es una respuesta a un
prompt». Tenía razón: el flujo mandaba una sola consulta con los pasos
escritos, y el modelo contestaba con una lista de faltantes.

Ahora el flujo es un agente que construye el escrito POR PARTES. En cada parte:

  1. ESTE MÓDULO lee el encargo, la carpeta, lo ya confirmado y lo ya
     redactado, y devuelve qué sabe de cada dato que la parte necesita: una
     propuesta con alternativas para que el abogado sólo palomee, o un hueco
     honesto cuando el dato es de hecho y no consta.
  2. El abogado confirma en pantalla.
  3. La parte se redacta con /chat en modo redacción y con TODO el acervo
     (eso vive en el frontend; aquí no se busca nada).

Por qué aparte de main.py: es un ayudante barato y sin estado —un modelo, una
respuesta JSON, sin Qdrant—, y no debe tocar el camino crítico del chat. Si
algo falla aquí, el flujo sigue con los campos vacíos y el abogado los llena.

Coste medido con gpt-6-luna (0.10/0.50 USD por millón): una llamada con 40
mil caracteres de carpeta ronda 0.002 USD. Por eso no descuenta consultas; lo
que cuesta es redactar, y eso se cobra en /chat como siempre.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict, deque
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()

FLUJO_MODELO = os.getenv("FLUJO_MODELO", "gpt-6-luna")
FLUJO_ESFUERZO = os.getenv("FLUJO_ESFUERZO", "low")
FLUJO_PLAZO_SEG = float(os.getenv("FLUJO_PLAZO_SEG", "60"))
# Freno por usuario: un flujo completo usa de 4 a 8 llamadas; 40 en diez
# minutos es holgado para trabajar y corta cualquier bucle.
FLUJO_TOPE = int(os.getenv("FLUJO_TOPE", "40"))
FLUJO_VENTANA_SEG = 600

_cliente = None
_supabase = None
_llamadas: dict[str, deque] = defaultdict(deque)


def _openai():
    global _cliente
    if _cliente is None:
        from openai import AsyncOpenAI
        clave = os.getenv("OPENAI_API_KEY", "")
        if not clave:
            raise HTTPException(status_code=503, detail="El agente de flujos no está configurado.")
        _cliente = AsyncOpenAI(api_key=clave)
    return _cliente


def _admin():
    global _supabase
    if _supabase is None:
        url, clave = os.getenv("SUPABASE_URL", ""), os.getenv("SUPABASE_SERVICE_KEY", "")
        if not (url and clave):
            raise HTTPException(status_code=503, detail="Supabase no configurado")
        from supabase import create_client
        _supabase = create_client(url, clave)
    return _supabase


# ── Lo que llega ─────────────────────────────────────────────────────────────

TipoCampo = Literal["texto", "parrafo", "fecha", "opcion", "varias", "documento"]


class CampoFlujo(BaseModel):
    id: str = Field(..., min_length=1, max_length=40)
    etiqueta: str = Field(..., min_length=1, max_length=200)
    tipo: TipoCampo
    ayuda: Optional[str] = Field(None, max_length=600)
    opciones: Optional[list[str]] = Field(None, max_length=12)
    obligatorio: bool = True


class DeducirRequest(BaseModel):
    flujo: str = Field(..., max_length=60)
    entrega: str = Field(..., max_length=200)
    parte: str = Field(..., max_length=200)
    objetivo_parte: Optional[str] = Field(None, max_length=2000)
    campos: list[CampoFlujo] = Field(..., min_length=1, max_length=20)
    conocido: dict[str, Any] = Field(default_factory=dict)
    encargo: str = Field("", max_length=8000)
    carpeta: Optional[str] = Field(None, max_length=60000)
    documento: Optional[str] = Field(None, max_length=12000)
    estado: Optional[str] = Field(None, max_length=60)


# ── Lo que se le pide al modelo ──────────────────────────────────────────────

SISTEMA = """Eres el agente de flujos de trabajo de Iurexia, asistente jurídico para el derecho mexicano. El abogado construye un escrito paso a paso contigo. En este turno preparas UNA parte del escrito: decides qué datos ya se conocen, cuáles propones y cuáles hay que pedirle.

Devuelve SÓLO un objeto JSON con esta forma exacta:
{"pensamiento": ["..."], "campos": [{"id": "...", "propuesta": ..., "opciones": ["..."], "confianza": "...", "origen": "...", "nota": "..."}], "pedir_documento": null, "consulta": "..."}

pensamiento: de 3 a 6 frases breves, en primera persona, que narren lo que revisaste y concluiste para ESTA parte: de dónde sacaste cada dato, qué falta y por qué importa. Nada de fórmulas vacías ni de repetir el encargo.

campos: uno por cada campo pedido, en el mismo orden, con:
- id: el mismo id del campo.
- propuesta: el valor que propones. Un texto para texto, parrafo, fecha y opcion; una lista de textos para varias; null si no hay base para proponerlo. Para documento, un resumen de lo que ya consta de ese documento o null.
- opciones: para opcion y varias, de 2 a 6 alternativas razonables EN ESTE CASO, empezando por la propuesta (en varias incluye las propuestas y otras plausibles que el abogado podría marcar). Si el campo trae opciones fijas, úsalas tal cual. Para texto y parrafo, lista vacía salvo que haya candidatos reales en los documentos.
- confianza: "alta" si consta en el encargo, la carpeta o lo confirmado; "media" si es la inferencia jurídica más probable; "baja" si es sólo una sugerencia; "ninguna" si la propuesta es null.
- origen: "encargo", "carpeta", "confirmado", "criterio" o "ninguno".
- nota: una frase breve para el abogado sobre por qué lo propones o qué debe verificar ("" si nada).

Reglas duras:
1. Los DATOS DE HECHO nunca se inventan: nombres de personas, domicilios, fechas, números de expediente, montos, nombres y cargos de servidores públicos, número del juzgado. Si no constan, propuesta null, confianza "ninguna", y en la nota di qué documento los trae.
2. Los campos de CRITERIO (vía y órgano competente por materia y territorio, autoridades por su función —ordenadora o ejecutora—, derechos y preceptos violados, líneas de argumento, excepciones, pruebas idóneas, efectos de la suspensión) sí los propones con tu mejor juicio jurídico conforme al derecho mexicano vigente, ajustados a los hechos del caso.
2 bis. Todo campo opcion o varias lleva SIEMPRE una propuesta: la alternativa más probable en este caso (confianza "media" si no consta expresamente), para que el abogado sólo confirme. Si el campo trae opciones fijas, la propuesta es una de ellas; por ejemplo, una persona física que no dice lo contrario promueve «Por su propio derecho».
3. Fechas en formato AAAA-MM-DD.
4. Lo confirmado por el abogado es definitivo: no lo contradigas y úsalo para proponer lo demás.
5. pedir_documento: un objeto {"nombre": "...", "motivo": "..."} sólo si esta parte no puede redactarse bien sin un documento que no está en la carpeta (el acto reclamado, la demanda que se contesta, la sentencia que se impugna, el contrato que se revisa). Si no hace falta, null.
6. consulta: una frase de búsqueda jurídica precisa (materia, figura, preceptos y hechos clave) para hallar en el acervo las normas y la jurisprudencia que esta parte necesita.
7. Escribe en español de México, con lenguaje jurídico preciso y sin adornos."""


def _recortar(texto: Optional[str], tope: int) -> str:
    texto = (texto or "").strip()
    return texto if len(texto) <= tope else texto[:tope] + "\n[…]"


def _mensaje_usuario(req: DeducirRequest) -> str:
    campos = [
        {k: v for k, v in c.model_dump().items() if v not in (None, [], "")}
        for c in req.campos
    ]
    partes = [
        f"# ESCRITO QUE SE CONSTRUYE\n{req.entrega}",
        f"# PARTE DE ESTE TURNO\n{req.parte}" + (f"\n{req.objetivo_parte}" if req.objetivo_parte else ""),
        "# CAMPOS QUE NECESITA ESTA PARTE\n" + json.dumps(campos, ensure_ascii=False, indent=1),
        "# LO QUE EL ABOGADO YA CONFIRMÓ\n" + (json.dumps(req.conocido, ensure_ascii=False, indent=1) if req.conocido else "(nada todavía)"),
        f"# ENCARGO DEL ABOGADO\n{_recortar(req.encargo, 8000) or '(sin encargo)'}",
    ]
    if req.estado:
        partes.append(f"# ENTIDAD DEL ABOGADO\n{req.estado}")
    if req.carpeta:
        partes.append(f"# CARPETA DEL ASUNTO\n{_recortar(req.carpeta, 60000)}")
    if req.documento:
        partes.append(f"# LO QUE YA ESTÁ REDACTADO DEL ESCRITO\n{_recortar(req.documento, 12000)}")
    return "\n\n".join(partes)


# ── Lo que sale: se normaliza SIEMPRE ────────────────────────────────────────
# El frontend pinta formularios con esto. Un campo que falte o llegue con otro
# tipo no puede tumbar la pantalla: se completa con un hueco honesto.

_CONFIANZAS = {"alta", "media", "baja", "ninguna"}
_ORIGENES = {"encargo", "carpeta", "confirmado", "criterio", "ninguno"}


def _texto(v: Any, tope: int = 4000) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        v = "; ".join(str(x) for x in v if x not in (None, ""))
    s = str(v).strip()
    return s[:tope] if s else None


def _lista(v: Any, tope: int = 8, largo: int = 400) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, (list, tuple)):
        return []
    vistos, salida = set(), []
    for x in v:
        s = _texto(x, largo)
        if s and s.lower() not in vistos:
            vistos.add(s.lower())
            salida.append(s)
        if len(salida) >= tope:
            break
    return salida


def normalizar(crudo: Any, campos: list[CampoFlujo]) -> dict:
    datos = crudo if isinstance(crudo, dict) else {}
    por_id = {}
    for c in datos.get("campos") or []:
        if isinstance(c, dict) and isinstance(c.get("id"), str):
            por_id.setdefault(c["id"], c)

    salida = []
    for campo in campos:
        c = por_id.get(campo.id, {})
        if campo.tipo == "varias":
            propuesta: Any = _lista(c.get("propuesta"), 10) or None
        else:
            propuesta = _texto(c.get("propuesta"))
        opciones = _lista(c.get("opciones"), 8)
        if campo.opciones:
            # Las opciones fijas del flujo mandan; las del modelo se añaden
            # detrás sólo si no repiten.
            fijas = _lista(campo.opciones, 12)
            opciones = fijas + [o for o in opciones if o.lower() not in {f.lower() for f in fijas}]
            opciones = opciones[:10]
        if campo.tipo in ("opcion", "varias") and propuesta:
            for p in (propuesta if isinstance(propuesta, list) else [propuesta]):
                if p.lower() not in {o.lower() for o in opciones}:
                    opciones.insert(0, p)
        confianza = c.get("confianza") if c.get("confianza") in _CONFIANZAS else ("media" if propuesta else "ninguna")
        if not propuesta:
            confianza = "ninguna"
        origen = c.get("origen") if c.get("origen") in _ORIGENES else ("criterio" if propuesta else "ninguno")
        salida.append({
            "id": campo.id,
            "propuesta": propuesta,
            "opciones": opciones,
            "confianza": confianza,
            "origen": origen if propuesta else "ninguno",
            "nota": _texto(c.get("nota"), 400) or "",
        })

    pedir = datos.get("pedir_documento")
    if isinstance(pedir, dict) and _texto(pedir.get("nombre")):
        pedir = {"nombre": _texto(pedir.get("nombre"), 200), "motivo": _texto(pedir.get("motivo"), 400) or ""}
    else:
        pedir = None

    return {
        "pensamiento": _lista(datos.get("pensamiento"), 6, 400),
        "campos": salida,
        "pedir_documento": pedir,
        "consulta": _texto(datos.get("consulta"), 500) or "",
    }


# ── Quién pregunta ───────────────────────────────────────────────────────────

async def _usuario(request: Request) -> str:
    token = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Falta el token de sesión")
    try:
        quien = await asyncio.to_thread(_admin().auth.get_user, token)
        return quien.user.id
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Sesión inválida o expirada")


def _frenar(uid: str) -> None:
    ahora = time.monotonic()
    cola = _llamadas[uid]
    while cola and ahora - cola[0] > FLUJO_VENTANA_SEG:
        cola.popleft()
    if len(cola) >= FLUJO_TOPE:
        raise HTTPException(status_code=429, detail="Demasiadas peticiones al agente. Espera unos minutos.")
    cola.append(ahora)


@router.post("/flujo/deducir")
async def flujo_deducir(req: DeducirRequest, request: Request):
    uid = await _usuario(request)
    _frenar(uid)

    t0 = time.perf_counter()
    kwargs: dict = {
        "model": FLUJO_MODELO,
        "messages": [
            {"role": "system", "content": SISTEMA},
            {"role": "user", "content": _mensaje_usuario(req)},
        ],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 6000,
    }
    if FLUJO_ESFUERZO:
        kwargs["reasoning_effort"] = FLUJO_ESFUERZO

    crudo: Any = None
    motivo = ""
    try:
        r = await asyncio.wait_for(_openai().chat.completions.create(**kwargs), timeout=FLUJO_PLAZO_SEG)
        texto = (r.choices[0].message.content or "").strip()
        crudo = json.loads(texto) if texto else None
        uso = getattr(r, "usage", None)
        print(f"   🧭 FLUJO {req.flujo} · «{req.parte[:40]}» · {time.perf_counter() - t0:.1f}s · "
              f"{getattr(uso, 'prompt_tokens', '?')}→{getattr(uso, 'completion_tokens', '?')} tok · usuario {uid[:8]}")
    except asyncio.TimeoutError:
        motivo = "El agente tardó demasiado; llena los datos y continúa."
        print(f"   🧭 FLUJO {req.flujo}: se agotó el plazo ({FLUJO_PLAZO_SEG:.0f}s)")
    except json.JSONDecodeError:
        motivo = "El agente no devolvió datos legibles; llena los datos y continúa."
        print(f"   🧭 FLUJO {req.flujo}: respuesta no es JSON")
    except HTTPException:
        raise
    except Exception as e:  # el flujo sigue aunque el agente falle
        motivo = "El agente no respondió; llena los datos y continúa."
        print(f"   🧭 FLUJO {req.flujo}: {type(e).__name__}: {str(e)[:200]}")

    salida = normalizar(crudo, req.campos)
    if motivo:
        salida["aviso"] = motivo
    return salida
