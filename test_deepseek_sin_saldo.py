# -*- coding: utf-8 -*-
"""DeepSeek sin saldo → luna, y el Flash Lite que existe (22-nov-2026). Sin red.

El 22-nov la cuenta de DeepSeek quedó en −0.01 USD y cada llamada contestaba 402
«Insufficient Balance»; y `gemini-3.1-flash-lite-preview` ya no existía en la API
v1 desde el 18-nov (HyDE y el hilo fallaban en silencio). Aquí se comprueba:
  1 · con la cuenta no disponible, `deepseek_disponible()` dice que no;
  2 · se mira el saldo como mucho cada diez minutos, no en cada consulta;
  3 · un 402 de saldo lo anota al instante; otro error no;
  4 · DEEPSEEK_OFICIAL=0 lo apaga a mano;
  5 · al recargar, vuelve sola a DeepSeek;
  6 · el chat y Sálvame cambian de motor (y luna va sin razonar);
  7 · el modelo «-preview» retirado se corrige solo aunque Render lo fije.
    python test_deepseek_sin_saldo.py
"""
import asyncio
import io
import os
import sys

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


for _k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY",
           "MISTRAL_API_KEY", "ELEVENLABS_API_KEY", "AZURE_DOCINT_KEY", "AZURE_SPEECH_KEY"):
    os.environ[_k] = "falsa-para-pruebas"
os.environ["QDRANT_URL"] = "http://127.0.0.1:9"
os.environ["QDRANT_API_KEY"] = ""
os.environ["SUPABASE_URL"] = ""
os.environ["SUPABASE_SERVICE_KEY"] = ""
os.environ["GEMINI_LITE_MODEL"] = "gemini-3.1-flash-lite-preview"   # como lo tiene Render
os.environ.pop("DEEPSEEK_OFICIAL", None)

print("\n── Cargando main.py (sin arranque) ──")
import main  # noqa: E402

print("\n7 · el Flash Lite retirado")
ok(main.GEMINI_LITE_MODEL == "gemini-3.1-flash-lite", f"GEMINI_LITE_MODEL = {main.GEMINI_LITE_MODEL}")


class _Resp:
    def __init__(self, disponible):
        self.status_code = 200
        self._d = disponible

    def json(self):
        return {"is_available": self._d, "balance_infos": []}


class _Cliente:
    llamadas = 0
    disponible = False

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, headers=None):
        _Cliente.llamadas += 1
        assert url.endswith("/user/balance")
        return _Resp(_Cliente.disponible)


main.httpx.AsyncClient = _Cliente


async def pruebas():
    print("\n1 · cuenta sin saldo")
    main._DS_SALDO.update(disponible=True, visto=0.0)
    ok(await main.deepseek_disponible() is False, "no disponible")

    print("\n2 · el saldo se mira cada diez minutos como mucho")
    antes = _Cliente.llamadas
    for _ in range(20):
        await main.deepseek_disponible()
    ok(_Cliente.llamadas == antes, f"20 consultas, {_Cliente.llamadas - antes} lecturas del saldo")

    print("\n5 · al recargar vuelve sola")
    _Cliente.disponible = True
    main._DS_SALDO["visto"] -= 601
    ok(await main.deepseek_disponible() is True, "pasados diez minutos, con saldo: DeepSeek")

    print("\n3 · el 402 de saldo lo anota al instante; otro error no")
    ok(main.deepseek_sin_saldo(RuntimeError("Error code: 500 - boom")) is False, "un 500 no cuenta")
    ok(await main.deepseek_disponible() is True, "sigue disponible tras el 500")
    ok(main.deepseek_sin_saldo(RuntimeError("Error code: 402 - Insufficient Balance (request_id: x)")) is True,
       "el 402 de saldo sí")
    ok(await main.deepseek_disponible() is False, "y la siguiente consulta ya va por luna")

    print("\n4 · DEEPSEEK_OFICIAL=0 lo apaga a mano")
    main._DS_SALDO.update(disponible=True, visto=10 ** 12)
    os.environ["DEEPSEEK_OFICIAL"] = "0"
    ok(await main.deepseek_disponible() is False, "apagado")
    os.environ.pop("DEEPSEEK_OFICIAL")


asyncio.run(pruebas())

print("\n6 · el chat y Sálvame cambian de motor")
src = io.open("main.py", encoding="utf-8").read()
i = src.find("# DeepSeek sin saldo → luna (ver `deepseek_disponible`)")
j = src.find("_client_name = 'gemini' if use_gemini", i)
tramo = src[i:j]
ok(i > 0 and 0 < j - i < 900, "el cambio está justo antes de nombrar el cliente")
ok("active_client = chat_client" in tramo and "active_model = DEEPSEEK_RESPALDO_MODELO" in tramo
   and "use_thinking = False" in tramo, "pasa a OpenAI con luna y sin razonar")
s = src.find("async def stream_response():", src.find("SALVAME — Generando amparo de salud"))
salvame = src[s:s + 3000]
ok("await deepseek_disponible()" in salvame and "max_completion_tokens" in salvame
   and "**_salvame_motor" in salvame, "Sálvame elige motor con el saldo")
f = src.find("❌ Fallo del stream:")
ok("deepseek_sin_saldo(e)" in src[f:f + 300], "el fallo del stream anota el 402")
ok(main.DEEPSEEK_RESPALDO_MODELO == "gpt-6-luna", f"respaldo = {main.DEEPSEEK_RESPALDO_MODELO}")

if FALLOS:
    print("\nFALLOS:")
    for x in FALLOS:
        print("  ·", x)
    sys.exit(1)
print("\nOK · DeepSeek sin saldo va por luna y vuelve solo; el Flash Lite es el que existe")
