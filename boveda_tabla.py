"""LA BÓVEDA · LA TABLA DE REVISIÓN.   (oct-2026)

Filas = documentos, columnas = preguntas del abogado («¿quiénes son las
partes?», «¿vigencia?», «¿pena convencional?», «¿jurisdicción?»). Cada celda
es un JSON con la respuesta, la CITA LITERAL que la sostiene y la página.

UNA LLAMADA POR FILA, NO POR CELDA. El documento se lee una vez y responde
todas las columnas: con diez columnas, cobrar diez veces la lectura del mismo
contrato sería pagar diez veces lo mismo. Un contrato de 10 páginas son ~7 mil
tokens de entrada; a 0.10 USD el millón, una décima de centavo por fila.

LA CITA SE VERIFICA, NO SE CREE. El modelo copia el fragmento y dice la
página; aquí se busca ese fragmento en el texto de verdad (boveda_texto.
verificar_cita). Si está, se marca verificada y se corrige la página si el
modelo la dijo mal. Si no está, la celda lo dice: el abogado ve «cita no
verificada» y no una cita inventada con cara de buena.

DOCUMENTOS LARGOS. Hasta CONTEXTO_COMPLETO caracteres se manda el documento
entero, con la marca de cada página. Más allá, para cada pregunta se buscan
en Qdrant las páginas de ESE documento que más se le parecen y se mandan sólo
ésas (enteras, para que la cita se pueda comprobar).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import boveda_texto as bt

MODELO = os.getenv("BOVEDA_MODELO_TABLA", "gpt-6-luna")
ESFUERZO = os.getenv("BOVEDA_ESFUERZO_TABLA", "low")
PRECIO_ENTRADA = float(os.getenv("BOVEDA_PRECIO_ENTRADA", "0.10")) / 1_000_000
PRECIO_SALIDA = float(os.getenv("BOVEDA_PRECIO_SALIDA", "0.50")) / 1_000_000
CONTEXTO_COMPLETO = int(os.getenv("BOVEDA_CONTEXTO_COMPLETO", "160000"))   # caracteres
PAGINAS_POR_PREGUNTA = 4
PLAZO_S = float(os.getenv("BOVEDA_PLAZO_FILA_S", "180"))

TIPOS = {
    "texto": "respuesta breve (una o dos frases)",
    "fecha": "la fecha en formato DD/MM/AAAA si es exacta; si es un plazo, dilo («12 meses desde el 05/03/2025»)",
    "monto": "la cantidad con su moneda («$75,000.00 MXN más IVA»)",
    "si_no": "«Sí» o «No», seguido de una precisión breve si hace falta",
    "lista": "elementos separados por «; »",
}

COLUMNAS_SUGERIDAS = [
    {"id": "partes", "titulo": "Partes", "tipo": "lista",
     "pregunta": "¿Quiénes son las partes que celebran el documento y con qué carácter (arrendador, cliente, etc.)?"},
    {"id": "fecha", "titulo": "Fecha de firma", "tipo": "fecha",
     "pregunta": "¿En qué fecha se firmó o celebró el documento?"},
    {"id": "vigencia", "titulo": "Vigencia", "tipo": "texto",
     "pregunta": "¿Cuál es la vigencia o plazo del documento y se renueva automáticamente?"},
    {"id": "monto", "titulo": "Contraprestación", "tipo": "monto",
     "pregunta": "¿Cuál es el monto o contraprestación pactada?"},
    {"id": "pena", "titulo": "Pena convencional", "tipo": "texto",
     "pregunta": "¿Se pactó una pena convencional? ¿De cuánto y por qué incumplimiento?"},
    {"id": "jurisdiccion", "titulo": "Jurisdicción", "tipo": "texto",
     "pregunta": "¿A qué jurisdicción o tribunales se someten las partes?"},
]

SISTEMA = """Eres un abogado mexicano que revisa documentos para una tabla de revisión.
Responde CADA pregunta usando SÓLO el documento que se te da. No uses conocimiento externo ni supongas.

Para cada pregunta devuelve:
- "columna_id": el id de la pregunta, tal cual.
- "encontrado": true si el documento contesta la pregunta; false si no la contesta.
- "valor": la respuesta en el formato pedido para esa columna. Si no se encontró: "No se menciona".
- "cita": un fragmento COPIADO LITERALMENTE del documento (mismas palabras, sin parafrasear, sin agregar ni quitar nada, sin puntos suspensivos) que sostiene la respuesta; entre 40 y 300 caracteres. Vacío si no se encontró.
- "pagina": el número de la página donde está la cita, según las marcas «=== PÁGINA N ===». null si no se encontró.

Devuelve ÚNICAMENTE un objeto JSON: {"respuestas": [ ... ]} con una entrada por pregunta, en el mismo orden."""


@dataclass
class Celda:
    columna_id: str
    valor: str = ""
    encontrado: bool = False
    cita: str = ""
    pagina: Optional[int] = None
    verificada: bool = False
    similitud: float = 0.0
    error: str = ""

    def a_json(self) -> dict:
        return {"valor": self.valor, "encontrado": self.encontrado, "cita": self.cita,
                "pagina": self.pagina, "verificada": self.verificada,
                "similitud": self.similitud, **({"error": self.error} if self.error else {})}


@dataclass
class Fila:
    celdas: list[Celda] = field(default_factory=list)
    tokens_entrada: int = 0
    tokens_salida: int = 0
    segundos: float = 0.0
    caracteres_contexto: int = 0
    paginas_enviadas: int = 0

    @property
    def usd(self) -> float:
        return self.tokens_entrada * PRECIO_ENTRADA + self.tokens_salida * PRECIO_SALIDA


def _contexto(nombre: str, paginas: list[str], elegidas: Optional[list[int]] = None,
              aproximadas: bool = False) -> str:
    nums = elegidas or list(range(1, len(paginas) + 1))
    partes = [f"DOCUMENTO: {nombre}"
              + (" (Word: las páginas son aproximadas, de ~3,000 caracteres)" if aproximadas else "")]
    for n in sorted(set(nums)):
        if 1 <= n <= len(paginas):
            partes.append(f"=== PÁGINA {n} ===\n{paginas[n - 1]}")
    return "\n\n".join(partes)


def _preguntas(columnas: list[dict]) -> str:
    lineas = []
    for c in columnas:
        formato = TIPOS.get(c.get("tipo") or "texto", TIPOS["texto"])
        lineas.append(f'- columna_id "{c["id"]}": {c["pregunta"]}  [formato: {formato}]')
    return "PREGUNTAS:\n" + "\n".join(lineas)


async def paginas_relevantes(buscar, columnas: list[dict], n_paginas: int) -> list[int]:
    """Para documentos largos: por cada pregunta, las páginas que más se le
    parecen dentro del MISMO documento (buscar = función que ya lleva el filtro)."""
    elegidas: list[int] = [1]                     # la primera casi siempre trae partes y fecha
    resultados = await asyncio.gather(*(buscar(c["pregunta"]) for c in columnas), return_exceptions=True)
    for r in resultados:
        if isinstance(r, Exception):
            continue
        vistas = []
        for h in r:
            if h.pagina not in vistas:
                vistas.append(h.pagina)
            if len(vistas) >= PAGINAS_POR_PREGUNTA:
                break
        elegidas.extend(vistas)
    if n_paginas > 1:
        elegidas.append(n_paginas)                # la última: firmas, lugar y fecha
    return sorted(set(p for p in elegidas if 1 <= p <= n_paginas))


def _normalizar(crudo, columnas: list[dict]) -> list[Celda]:
    lista = crudo.get("respuestas") if isinstance(crudo, dict) else crudo
    por_id = {}
    if isinstance(lista, list):
        for r in lista:
            if isinstance(r, dict) and r.get("columna_id"):
                por_id[str(r["columna_id"])] = r
    celdas = []
    for c in columnas:
        r = por_id.get(c["id"])
        if not r:
            celdas.append(Celda(columna_id=c["id"], error="sin respuesta del modelo"))
            continue
        pag = r.get("pagina")
        try:
            pag = int(pag) if pag not in (None, "", "null") else None
        except (TypeError, ValueError):
            m = re.search(r"\d+", str(pag))
            pag = int(m.group()) if m else None
        celdas.append(Celda(
            columna_id=c["id"],
            valor=str(r.get("valor") or "").strip()[:2000],
            encontrado=bool(r.get("encontrado")),
            cita=str(r.get("cita") or "").strip().strip("«»\"“”")[:600],
            pagina=pag,
        ))
    return celdas


def verificar(celdas: list[Celda], paginas: list[str]) -> None:
    for c in celdas:
        if not c.encontrado or not c.cita:
            continue
        v = bt.verificar_cita(c.cita, paginas, c.pagina)
        c.verificada, c.similitud = v.verificada, v.similitud
        if v.verificada and v.pagina:
            c.pagina = v.pagina


async def responder_fila(cliente, *, nombre: str, paginas: list[str], columnas: list[dict],
                         aproximadas: bool = False, buscar=None) -> Fila:
    """Todas las columnas de UN documento en una llamada. `buscar(pregunta)`
    sólo hace falta si el documento rebasa CONTEXTO_COMPLETO."""
    t0 = time.perf_counter()
    total = sum(len(p) for p in paginas)
    elegidas = None
    if total > CONTEXTO_COMPLETO and buscar is not None:
        elegidas = await paginas_relevantes(buscar, columnas, len(paginas))
    contexto = _contexto(nombre, paginas, elegidas, aproximadas)
    if len(contexto) > CONTEXTO_COMPLETO * 1.5:          # aun elegidas, demasiado: se recorta
        contexto = contexto[: int(CONTEXTO_COMPLETO * 1.5)]
    kwargs = {
        "model": MODELO,
        "messages": [{"role": "system", "content": SISTEMA},
                     {"role": "user", "content": contexto + "\n\n" + _preguntas(columnas)}],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 4000 + 400 * len(columnas),
    }
    if ESFUERZO:
        kwargs["reasoning_effort"] = ESFUERZO
    fila = Fila(caracteres_contexto=len(contexto),
                paginas_enviadas=len(elegidas) if elegidas else len(paginas))
    ultimo_error = ""
    for intento in range(2):
        try:
            r = await asyncio.wait_for(cliente.chat.completions.create(**kwargs), timeout=PLAZO_S)
            uso = getattr(r, "usage", None)
            fila.tokens_entrada += int(getattr(uso, "prompt_tokens", 0) or 0)
            fila.tokens_salida += int(getattr(uso, "completion_tokens", 0) or 0)
            texto = (r.choices[0].message.content or "").strip()
            fila.celdas = _normalizar(json.loads(texto), columnas)
            break
        except json.JSONDecodeError:
            ultimo_error = "respuesta ilegible"
        except asyncio.TimeoutError:
            ultimo_error = "el modelo tardó demasiado"
        except Exception as e:  # noqa: BLE001
            ultimo_error = f"{type(e).__name__}"
            await asyncio.sleep(2)
    if not fila.celdas:
        fila.celdas = [Celda(columna_id=c["id"], error=ultimo_error or "sin respuesta") for c in columnas]
    verificar(fila.celdas, paginas)
    fila.segundos = round(time.perf_counter() - t0, 2)
    return fila
