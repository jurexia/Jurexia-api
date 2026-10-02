# -*- coding: utf-8 -*-
"""El embedding de una consulta larga y densa no tumba el chat — 2-oct-2026.

    .venv/bin/python test_embedding_tope.py

Un usuario Platinum pegó en el chat sus recibos de nómina: RFC, CURP, claves e
importes tokenizan a 1.93 caracteres por token, así que 17,090 caracteres eran
8,859 tokens contra el tope de 8,192 del modelo de embeddings. El recorte por
caracteres (20,000) no lo alcanzaba, la API devolvía 400 y tenacity repetía la
MISMA entrada tres veces antes de soltar «Fallo del stream». Sin red: el cliente
de OpenAI es falso y cuenta las llamadas.
"""
import asyncio
import contextlib
import io
import os
import sys

import httpx
from openai import BadRequestError, APIConnectionError
from tenacity import RetryError, wait_none

sys.path.insert(0, os.getcwd())
with contextlib.redirect_stdout(io.StringIO()):
    import main

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


def _400(mensaje):
    req = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    return BadRequestError(mensaje, response=httpx.Response(400, request=req), body=None)


class _Embeddings:
    def __init__(self, tope_chars=None, error=None):
        self.tope, self.error, self.entradas = tope_chars, error, []

    async def create(self, model, input):
        self.entradas.append(len(input))
        if self.error is not None:
            raise self.error
        if self.tope is not None and len(input) > self.tope:
            raise _400("Error code: 400 - {'error': {'message': \"Invalid 'input': maximum context length is "
                       "8192 tokens.\", 'type': 'invalid_request_error'}}")
        return type("R", (), {"data": [type("D", (), {"embedding": [0.0] * 8})()]})()


def _cliente(**kw):
    emb = _Embeddings(**kw)
    main.openai_client = type("C", (), {"embeddings": emb})()
    return emb


def correr(co):
    with contextlib.redirect_stdout(io.StringIO()):
        return asyncio.new_event_loop().run_until_complete(co)


print("\n1 · EL TOPE EN CARACTERES ALCANZA A LO DENSO")
ok(main.MAX_CHARS_EMBEDDING <= 14_000, f"tope de {main.MAX_CHARS_EMBEDDING:,} caracteres (a 1.93 car./token, menos de 8,192 tokens)")
ok(len(main._acotar_para_embedding("x" * 27_443)) == main.MAX_CHARS_EMBEDDING, "el mensaje de 27,443 caracteres se acota")
ok(main._acotar_para_embedding("corto") == "corto", "lo corto no se toca")

print("\n2 · SI AUN ASÍ SE PASA, SE PARTE A LA MITAD")
emb = _cliente(tope_chars=5_000)
v = correr(main.get_dense_embedding("a" * 13_000))
ok(v == [0.0] * 8, "devuelve el vector")
ok(emb.entradas == [13_000, 6_500, 3_250], f"13,000 → 6,500 → 3,250 caracteres ({emb.entradas})")

print("\n3 · OTRO 400 NO SE REPITE")
emb = _cliente(error=_400("Invalid 'input': input must be a string"))
try:
    correr(main.get_dense_embedding("hola"))
    ok(False, "un 400 que no es de largo sube")
except BadRequestError:
    ok(True, "un 400 que no es de largo sube")
ok(emb.entradas == [4], f"una sola llamada, sin los tres reintentos ({emb.entradas})")

print("\n4 · LO PASAJERO SÍ SE REINTENTA")
emb = _cliente(error=APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/embeddings")))
try:
    correr(main.get_dense_embedding.retry_with(wait=wait_none())("hola"))
    ok(False, "un error de conexión se reintenta tres veces")
except RetryError:
    ok(emb.entradas == [4, 4, 4], f"un error de conexión se reintenta tres veces ({emb.entradas})")

print("\n" + ("TODO PASA" if not FALLOS else f"{len(FALLOS)} FALLAN: {FALLOS}"))
raise SystemExit(1 if FALLOS else 0)
