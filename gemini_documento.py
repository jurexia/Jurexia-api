# -*- coding: utf-8 -*-
"""Gemini 3.1 Pro directo (API de Gemini, clave nueva) para el análisis de documentos de Platinum.

POR QUÉ EXISTE (27-sep-2026)
----------------------------
El 26-sep el análisis de documentos dejó OpenRouter (de66e3d): Platinum pasó de
`google/gemini-3.1-pro-preview` a gpt-6-luna con razonamiento `medium`. Un
abogado Platinum de los que más lo usan (≈50 análisis por semana, casi todos
escritos y denuncias sobre su propio expediente penal de CDMX) escribió al día
siguiente que «desde hoy contesta como ChatGPT… sólo hace resúmenes y no acata
lo que le delimito». Los registros de Render le dan la razón:

    48 análisis con Gemini 3.1 Pro  → 11,254 caracteres de mediana, 3 citas verificadas
    13 análisis con gpt-6-luna      → 36,117 caracteres de mediana, 1 cita verificada

(misma ruta, mismo prompt salvo una línea, respuestas completas: el largo lo
decide el modelo). En toda la plataforma las citas verificadas por análisis
bajaron de 4.2 a 1.7 de media. David decidió devolver Gemini 3.1 Pro a
Platinum, pero directo con la clave nueva de la API de Gemini, sin OpenRouter.

CÓMO ENCAJA
-----------
El análisis de documentos consume trozos con la forma de OpenAI
(`choices[0].delta.content`, `finish_reason` y el `native_finish_reason` de
OpenRouter): de eso dependen el repliegue a luna si el motor no abre, la
continuación cuando Gemini corta por RECITATION y el sello de citas. En vez de
reescribir ese bucle, `ClienteGeminiDocumento` imita a `AsyncOpenAI` en lo
único que se usa —`await cliente.chat.completions.create(model, messages,
stream=True, max_tokens, temperature)`— y por dentro habla el SDK nativo
(`google-genai`), que sí dice por qué paró (RECITATION, SAFETY…) sin traducir.

LO QUE NO ES OBVIO
------------------
· La clave empieza por «AQ.»: parece de Vertex en modo exprés, pero es de AI
  Studio. Con `vertexai=True` responde 403 (la API de Vertex no está habilitada
  en su proyecto); con la API de desarrolladores de Gemini funciona (probado el
  27-sep: «LISTO» en 3.4 s).
· Los errores de apertura (clave, cuota, modelo) salen al PRIMER trozo del
  stream, no al pedirlo. Aquí se lee el primer trozo dentro de `create()` para
  que el error suba donde el análisis lo sabe tratar: al abrir, con repliegue.
· Filtros de seguridad en OFF: son expedientes penales (violencia, delitos
  sexuales, amenazas). Un bloqueo por «contenido peligroso» a mitad de una
  denuncia de fraude procesal sería el fallo más caro posible para el abogado.
· Temperatura 0.3 y 32,768 tokens de salida: los MISMOS parámetros con los que
  OpenRouter llamaba a este modelo hasta el 25-sep, que es la calidad que el
  abogado echa de menos. El razonamiento queda en el de omisión del modelo
  (dinámico), igual que entonces. Todo se mueve con variables sin desplegar.
"""
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

# La clave nueva vive aparte para no mezclar su cuota con la de los Genios y el
# resto de rutas de Gemini, que siguen con GEMINI_API_KEY.
CLAVE_ENV = "GEMINI_API_KEY_DOCUMENTO"
TEMPERATURA = float(os.getenv("DOCUMENT_GEMINI_TEMPERATURA", "0.3"))
MAX_SALIDA = int(os.getenv("DOCUMENT_GEMINI_MAX_SALIDA", "32768"))

_cliente_genai = None


def clave() -> str:
    return (os.getenv(CLAVE_ENV) or "").strip()


def disponible() -> bool:
    """¿Hay clave para Gemini directo? Sin ella, Platinum sigue con luna."""
    return bool(clave())


def es_modelo_gemini_directo(modelo: Optional[str]) -> bool:
    """Un id «gemini-…» sin «/» es de la API de Gemini directa; con «/»
    (google/gemini-…) sigue siendo de OpenRouter, como antes del 26-sep."""
    return bool(modelo) and "/" not in modelo and str(modelo).startswith("gemini-")


def _genai():
    global _cliente_genai
    if _cliente_genai is None:
        from google import genai
        if not disponible():
            raise RuntimeError(f"{CLAVE_ENV} no configurada")
        _cliente_genai = genai.Client(api_key=clave())
    return _cliente_genai


# ── De la forma de OpenAI a la de Gemini ─────────────────────────────────────

def _a_gemini(messages: List[Dict[str, Any]]):
    """(instrucción de sistema, contents). Los `system` se juntan en la
    instrucción; `assistant` es `model` para Gemini (lo usa la continuación
    tras un corte por recitación)."""
    from google.genai import types
    sistema: List[str] = []
    contents = []
    for m in messages:
        rol = m.get("role")
        texto = m.get("content") or ""
        if not isinstance(texto, str):
            texto = str(texto)
        if rol == "system":
            sistema.append(texto)
            continue
        contents.append(types.Content(role="model" if rol == "assistant" else "user",
                                      parts=[types.Part(text=texto)]))
    return ("\n\n".join(s for s in sistema if s) or None), contents


def _config(instruccion: Optional[str], max_tokens: Optional[int], temperature: Optional[float]):
    from google.genai import types
    apagado = types.HarmBlockThreshold.OFF
    return types.GenerateContentConfig(
        system_instruction=instruccion,
        max_output_tokens=int(max_tokens or MAX_SALIDA),
        temperature=TEMPERATURA if temperature is None else float(temperature),
        safety_settings=[types.SafetySetting(category=c, threshold=apagado) for c in (
            types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        )],
    )


# ── De la forma de Gemini a la de OpenAI ─────────────────────────────────────

# Lo que el análisis entiende: «stop» terminó bien, «length» se acabó el tope,
# «content_filter» lo cortó un filtro (y dispara la continuación con luna) y
# «error» cualquier otra cosa. El nombre nativo viaja en model_extra, donde el
# bucle ya lo buscaba para OpenRouter.
_FIN = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "IMAGE_SAFETY": "content_filter",
}


def _nombre(fin: Any) -> Optional[str]:
    if fin is None:
        return None
    n = getattr(fin, "name", None) or str(fin)
    n = n.split(".")[-1]
    return None if n in ("FINISH_REASON_UNSPECIFIED", "None", "") else n


def _texto_de(chunk: Any) -> str:
    """Sólo el texto de la respuesta: nunca las partes de razonamiento
    (`thought`), que no se piden pero se filtran por si el modelo las manda."""
    try:
        cands = getattr(chunk, "candidates", None) or []
        if not cands:
            return ""
        partes = getattr(getattr(cands[0], "content", None), "parts", None) or []
        return "".join(p.text for p in partes if getattr(p, "text", None) and not getattr(p, "thought", False))
    except Exception:
        return ""


def _trozo(texto: Optional[str] = None, fin: Optional[str] = None, nativo: Optional[str] = None):
    choice = SimpleNamespace(
        delta=SimpleNamespace(content=texto or None),
        finish_reason=fin,
        native_finish_reason=nativo,
        model_extra={"native_finish_reason": nativo} if nativo else {},
    )
    return SimpleNamespace(choices=[choice])


class _Stream:
    """El stream de Gemini con la forma del de OpenAI. El primer trozo ya se
    leyó al abrir (ver módulo)."""

    def __init__(self, primero, resto):
        self._primero = primero
        self._resto = resto

    def __aiter__(self):
        return self._generar()

    async def _generar(self):
        fin_visto = False
        for chunk in self._iterar_primero():
            async for t in self._convertir(chunk):
                fin_visto = fin_visto or bool(t.choices[0].finish_reason)
                yield t
        async for chunk in self._resto:
            async for t in self._convertir(chunk):
                fin_visto = fin_visto or bool(t.choices[0].finish_reason)
                yield t
        if not fin_visto:
            # Un stream que se acaba sin decir por qué: se da por bueno, como
            # hace OpenAI al cerrar sin finish_reason explícito.
            yield _trozo(fin="stop", nativo="STOP")

    def _iterar_primero(self):
        if self._primero is not None:
            yield self._primero
            self._primero = None

    @staticmethod
    async def _convertir(chunk):
        texto = _texto_de(chunk)
        cands = getattr(chunk, "candidates", None) or []
        nativo = _nombre(getattr(cands[0], "finish_reason", None)) if cands else None
        if texto:
            yield _trozo(texto=texto)
        if nativo:
            yield _trozo(fin=_FIN.get(nativo, "error"), nativo=nativo)


class _Completions:
    async def create(self, model: str, messages: List[Dict[str, Any]], stream: bool = True,
                     max_tokens: Optional[int] = None, temperature: Optional[float] = None,
                     **_ignorados):
        instruccion, contents = _a_gemini(messages)
        it = await _genai().aio.models.generate_content_stream(
            model=model, contents=contents, config=_config(instruccion, max_tokens, temperature))
        it = it.__aiter__()
        # El error de apertura sale aquí, al primer trozo (ver módulo).
        try:
            primero = await it.__anext__()
        except StopAsyncIteration:
            primero = None
        bloqueo = getattr(getattr(primero, "prompt_feedback", None), "block_reason", None) if primero else None
        if bloqueo and not (getattr(primero, "candidates", None) or []):
            raise RuntimeError(f"Gemini bloqueó el documento al abrir: {_nombre(bloqueo) or bloqueo}")
        return _Stream(primero, it)


class ClienteGeminiDocumento:
    """Lo mínimo de `AsyncOpenAI` que usa el análisis de documentos."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=_Completions())


cliente = ClienteGeminiDocumento()
