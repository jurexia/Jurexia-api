# -*- coding: utf-8 -*-
"""Gemini 3.1 Pro directo para el análisis de documentos de Platinum (27-sep-2026).

Sin red ni claves: un Gemini falso entrega trozos con la forma del SDK
(`candidates[0].content.parts`, `finish_reason`) y se comprueba que el
adaptador los devuelve con la forma de OpenAI que lee el análisis de
documentos: texto en `delta.content`, el motivo en `finish_reason` y el nombre
nativo en `model_extra["native_finish_reason"]` (RECITATION dispara la
continuación con luna). Y que un error de apertura sube al abrir, que es donde
main.py sabe replegarse.

    python test_gemini_documento.py
"""
import asyncio
import contextlib
import importlib
import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
FALLOS = []


def ok(cond, que):
    print(("   PASA   " if cond else "   FALLA  ") + que)
    if not cond:
        FALLOS.append(que)


def correr(co):
    return asyncio.new_event_loop().run_until_complete(co)


os.environ.pop("GEMINI_API_KEY_DOCUMENTO", None)
import gemini_documento as gd  # noqa: E402

print("\n1 · QUÉ VA POR GEMINI DIRECTO")
ok(gd.es_modelo_gemini_directo("gemini-3.1-pro-preview"), "«gemini-3.1-pro-preview» es Gemini directo")
ok(not gd.es_modelo_gemini_directo("google/gemini-3.1-pro-preview"), "con «/» sigue siendo OpenRouter")
ok(not gd.es_modelo_gemini_directo("gpt-6-luna") and not gd.es_modelo_gemini_directo(None), "luna y None no")
ok(not gd.disponible(), "sin GEMINI_API_KEY_DOCUMENTO no está disponible")
os.environ["GEMINI_API_KEY_DOCUMENTO"] = "AQ.falsa-para-pruebas"
ok(gd.disponible(), "con la variable, sí")
os.environ.pop("GEMINI_API_KEY_DOCUMENTO")

print("\n2 · DE MENSAJES DE OPENAI A GEMINI")
instr, contents = gd._a_gemini([
    {"role": "system", "content": "Eres Iurexia."},
    {"role": "system", "content": "CONTEXTO JURÍDICO RECUPERADO: …"},
    {"role": "user", "content": "DOCUMENTO ADJUNTO…"},
    {"role": "assistant", "content": "Parte ya escrita"},
    {"role": "user", "content": "Continúa"},
])
ok(instr == "Eres Iurexia.\n\nCONTEXTO JURÍDICO RECUPERADO: …", "los system se juntan en la instrucción")
ok([c.role for c in contents] == ["user", "model", "user"], "assistant pasa a «model» (continuación tras recitación)")
ok(contents[1].parts[0].text == "Parte ya escrita", "el texto viaja intacto")
cfg = gd._config("x", None, None)
ok(cfg.max_output_tokens == 32768 and abs(cfg.temperature - 0.3) < 1e-9,
   "por omisión 32,768 tokens y temperatura 0.3: lo que usaba OpenRouter hasta el 25-sep")
ok(len(cfg.safety_settings) == 4 and all(s.threshold.name == "OFF" for s in cfg.safety_settings),
   "filtros de seguridad en OFF (expedientes penales)")


# ── Un Gemini falso ──────────────────────────────────────────────────────────
def _parte(texto, thought=False):
    return SimpleNamespace(text=texto, thought=thought)


def _chunk(partes=None, fin=None, bloqueo=None):
    cand = SimpleNamespace(content=SimpleNamespace(parts=partes or []),
                           finish_reason=(SimpleNamespace(name=fin) if fin else None))
    return SimpleNamespace(candidates=[cand] if (partes or fin) else [],
                           prompt_feedback=SimpleNamespace(block_reason=bloqueo) if bloqueo else None)


class _GenaiFalso:
    def __init__(self, chunks=None, error_al_primero=None):
        self._chunks = chunks or []
        self._error = error_al_primero
        self.pedido = None
        outer = self

        class _Models:
            async def generate_content_stream(self, model, contents, config):
                outer.pedido = (model, contents, config)

                async def gen():
                    if outer._error:
                        raise outer._error
                    for c in outer._chunks:
                        yield c
                return gen()
        self.aio = SimpleNamespace(models=_Models())


def _consumir(chunks=None, error=None):
    falso = _GenaiFalso(chunks, error)
    gd._genai = lambda: falso

    async def _todo():
        stream = await gd.cliente.chat.completions.create(
            model="gemini-3.1-pro-preview",
            messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
            stream=True, max_tokens=1000, temperature=0.3)
        salida = []
        async for t in stream:
            c = t.choices[0]
            salida.append((c.delta.content, c.finish_reason, (c.model_extra or {}).get("native_finish_reason")))
        return salida
    return correr(_todo()), falso


print("\n3 · DE TROZOS DE GEMINI A TROZOS DE OPENAI")
sal, falso = _consumir([
    _chunk([_parte("pensando…", thought=True)]),
    _chunk([_parte("Primer "), _parte("párrafo.")]),
    _chunk([_parte(" Segundo.")], fin="STOP"),
])
textos = [t for t, _, _ in sal if t]
ok(textos == ["Primer párrafo.", " Segundo."], f"sólo el texto, nunca el razonamiento ({textos})")
ok(sal[-1][1:] == ("stop", "STOP"), f"STOP → finish_reason «stop» ({sal[-1]})")
ok(falso.pedido[0] == "gemini-3.1-pro-preview" and falso.pedido[2].max_output_tokens == 1000,
   "el modelo y el tope pedidos llegan a Gemini")

sal, _ = _consumir([_chunk([_parte("Art. 1…")]), _chunk(fin="RECITATION")])
ok(sal[-1][1:] == ("content_filter", "RECITATION"),
   "RECITATION → «content_filter» con nativo RECITATION: dispara la continuación con luna")
sal, _ = _consumir([_chunk([_parte("mucho texto")], fin="MAX_TOKENS")])
ok(sal[-1][1] == "length", "MAX_TOKENS → «length»")
sal, _ = _consumir([_chunk([_parte("x")], fin="SAFETY")])
ok(sal[-1][1:] == ("content_filter", "SAFETY"), "SAFETY → «content_filter»")
sal, _ = _consumir([_chunk([_parte("x")], fin="OTHER")])
ok(sal[-1][1:] == ("error", "OTHER"), "OTHER → «error»")
sal, _ = _consumir([_chunk([_parte("sin motivo al final")])])
ok(sal[-1][1:] == ("stop", "STOP"), "un stream que acaba sin motivo se da por terminado bien")

print("\n4 · LOS ERRORES SUBEN AL ABRIR (donde main.py se repliega a luna)")
try:
    _consumir(error=RuntimeError("403 PERMISSION_DENIED"))
    ok(False, "un error del primer trozo sube desde create()")
except RuntimeError as e:
    ok("403" in str(e), "un error del primer trozo sube desde create()")
try:
    _consumir([_chunk(bloqueo=SimpleNamespace(name="PROHIBITED_CONTENT"))])
    ok(False, "un documento bloqueado al abrir levanta error")
except RuntimeError as e:
    ok("PROHIBITED_CONTENT" in str(e), "un documento bloqueado al abrir levanta error (y hay repliegue)")

print("\n5 · MAIN.PY: RUTA, OMISIÓN Y REPLIEGUE")
with contextlib.redirect_stdout(io.StringIO()):
    import main  # noqa: E402
importlib.reload(gd)  # el cliente real otra vez (las pruebas de arriba lo sustituyeron)
main._gemini_doc = gd
cli, params = main._via_documento("gemini-3.1-pro-preview", "medium")
ok(cli is gd.cliente and params == {"max_tokens": 32768, "temperature": 0.3},
   "gemini-… va por el cliente de Gemini directo con max_tokens y temperatura, sin reasoning_effort")
cli, params = main._via_documento("google/gemini-3.1-pro-preview")
ok(cli is main.deepseek_client, "google/gemini-… sigue yendo por OpenRouter (reversa de antes)")
cli, params = main._via_documento("gpt-6-luna", "medium")
ok(cli is main.chat_client and params.get("reasoning_effort") == "medium", "luna sigue por OpenAI con su esfuerzo")
ok(main.DOCUMENT_MODEL_PLATINUM == main.DOCUMENT_MODEL,
   "sin GEMINI_API_KEY_DOCUMENTO, Platinum sigue con luna (el código se puede desplegar antes que la clave)")
fuente = Path("main.py").read_text(encoding="utf-8")
ok('"gemini-3.1-pro-preview" if _gemini_doc.disponible() else DOCUMENT_MODEL' in fuente,
   "con la clave, Platinum pasa por omisión a gemini-3.1-pro-preview")
i = fuente.index("response = await _abrir(model_to_use, esfuerzo_doc)")
tramo = fuente[i:i + 900]
ok("_repliegue = (DOCUMENT_MODEL, esfuerzo_doc or DOCUMENT_ESFUERZO)" in tramo,
   "si Gemini no abre, Platinum cae a luna con SU esfuerzo (medium), no al low de todos")
ok('"Gemini directo (clave nueva)"' in fuente, "el registro dice por dónde fue («Gemini directo (clave nueva)»)")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}:")
    for f in FALLOS:
        print("  ·", f)
    sys.exit(1)
print("TODO PASA")
