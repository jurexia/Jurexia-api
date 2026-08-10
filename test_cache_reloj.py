#!/usr/bin/env python3
"""La invariante del caché de genios: aquí caduca ANTES que en Google.

Si se rompe, el servidor manda `cached_content=cachedContents/…` de un caché
que Google ya borró y la respuesta es un 403 —o un 400 INVALID_ARGUMENT según
la ruta— que llega tal cual al abogado. Peor: el nombre muerto se queda
anotado y la conversación no vuelve a funcionar. Pasó el 10-ago-2026 con una
suscriptora Pro que escribió once veces a soporte el mismo día.

No toca red ni Gemini: aísla las funciones de reloj y mueve `time.time()`.

    python3 test_cache_reloj.py
"""
import datetime as dt
import re
import sys
import time
from pathlib import Path

FUENTE = Path(__file__).with_name("cache_manager.py").read_text(encoding="utf-8")
TTL_MIN = 3
TTL = TTL_MIN * 60


class _Estado:
    def __init__(self):
        self.cache_name = None
        self.cache_created_at = 0.0


_ESTADOS = {}


def _get_state(genio_id):
    return _ESTADOS.setdefault(genio_id, _Estado())


class _Log:
    def __getattr__(self, _):
        return lambda *a, **k: None


def _cargar():
    """Extrae del módulo real sólo lo que depende del reloj."""
    ns = {
        "time": time, "_get_state": _get_state, "logger": _Log(),
        "CACHE_TTL_MINUTES": TTL_MIN, "Optional": __import__("typing").Optional,
    }
    for nombre in ("_is_cache_valid", "_reloj_desde_caducidad", "invalidar", "get_cache_name"):
        m = re.search(rf"^def {nombre}\(.*?(?=\n\n\ndef |\n\n\nasync def |\n\ndef |\n\nasync def )",
                      FUENTE, re.S | re.M)
        if not m:
            print(f"  no encontré {nombre}() en cache_manager.py")
            sys.exit(1)
        exec(m.group(0), ns)
    return ns


NS = _cargar()
fallos = []


def revisar(titulo, ok, detalle=""):
    print(f"  {'ok  ' if ok else 'MAL '} {titulo}{'  ·  ' + detalle if detalle else ''}")
    if not ok:
        fallos.append(titulo)


# ── 1 · Leer no rejuvenece el caché ──────────────────────────────────────
print("\nUn chat que escribe cada 2 minutos, con TTL de 3:")
st = _get_state("amparo")
st.cache_name = "cachedContents/x"
reloj_real = time.time
t0 = reloj_real()
st.cache_created_at = t0
muertos = 0
for minuto in (0, 2, 4, 6, 8, 10):
    time.time = (lambda m: (lambda: t0 + m * 60))(minuto)
    entregado = NS["get_cache_name"]("amparo")
    vivo_en_google = minuto * 60 < TTL
    if entregado and not vivo_en_google:
        muertos += 1
    en_google = "vivo" if vivo_en_google else "borrado"
    aqui = "entrega nombre" if entregado else "devuelve None"
    print(f"      min {minuto:2d}  Google: {en_google:>7s}   servidor: {aqui}")
time.time = reloj_real
revisar("nunca entrega un caché que Google ya borró", muertos == 0,
        f"{muertos} punteros muertos")

# ── 2 · Adoptar un caché ajeno respeta su vida restante ──────────────────
print("\nCaché adoptado de otra instancia:")
f = NS["_reloj_desde_caducidad"]


class _Remoto:
    def __init__(self, seg):
        self.expire_time = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=seg)


class _SinFecha:
    pass


class _FechaRara:
    expire_time = "no es una fecha"


for restante, esperado in ((170, True), (60, True), (10, True), (-30, False)):
    edad = reloj_real() - f(_Remoto(restante))
    revisar(f"le quedan {restante:>4}s en Google", (edad < TTL * 0.98) == esperado,
            f"edad local {edad:.0f}s")
for obj, etiqueta in ((_SinFecha(), "sin expire_time"), (_FechaRara(), "expire_time ilegible")):
    edad = reloj_real() - f(obj)
    revisar(f"{etiqueta} se da por caducado", edad >= TTL * 0.98, f"edad local {edad:.0f}s")

# ── 3 · invalidar() ──────────────────────────────────────────────────────
print("\nInvalidación tras un rechazo de Gemini:")
st.cache_name = "cachedContents/vigente"
st.cache_created_at = reloj_real()
NS["invalidar"]("amparo", "cachedContents/otro")
revisar("no borra un caché distinto al que falló", st.cache_name == "cachedContents/vigente")
NS["invalidar"]("amparo", "cachedContents/vigente")
revisar("borra el que falló", st.cache_name is None)

print()
if fallos:
    print(f"{len(fallos)} FALLOS: " + "; ".join(fallos))
    sys.exit(1)
print("La invariante se cumple.")
