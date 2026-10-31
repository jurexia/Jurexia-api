"""El trabajo no se va con la pantalla (respuesta_guardada.py, 30-oct-2026).

Prueba de punta a punta con uvicorn de verdad: un cliente HTTP que lee dos
trozos y cuelga, como el teléfono que se va a WhatsApp. Sin producción: la
base es una tabla en memoria y no se importa main.py.

    python test_sigue_sin_pantalla.py

Necesita fastapi, uvicorn y httpx.
"""
import asyncio
import json
import socket
import sys
import threading
import time
import uuid

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

import respuesta_guardada as rg

USUARIO = str(uuid.uuid4())


# ── Una tabla `respuestas_chat` en memoria, con la forma del cliente de Supabase ──

class _Resultado:
    def __init__(self, data):
        self.data = data


class _Consulta:
    def __init__(self, filas: dict):
        self.filas = filas
        self.op = None
        self.carga = None
        self.filtros: list = []

    def upsert(self, fila, on_conflict=None):
        self.op, self.carga = "upsert", dict(fila)
        return self

    def update(self, cambios):
        self.op, self.carga = "update", dict(cambios)
        return self

    def select(self, columnas):
        self.op, self.carga = "select", columnas
        return self

    def eq(self, k, v):
        self.filtros.append((k, v))
        return self

    def limit(self, n):
        return self

    def _coinciden(self):
        return [f for f in self.filas.values() if all(f.get(k) == v for k, v in self.filtros)]

    def execute(self):
        if self.op == "upsert":
            self.filas[self.carga["peticion_id"]] = self.carga
            return _Resultado([self.carga])
        if self.op == "update":
            hechas = self._coinciden()
            for f in hechas:
                f.update(self.carga)
            return _Resultado(hechas)
        return _Resultado([dict(f) for f in self._coinciden()])


class _Auth:
    def get_user(self, token):
        class U:
            pass
        r = U()
        r.user = U()
        r.user.id = USUARIO if token == "valido" else None
        return r


class BaseFalsa:
    def __init__(self):
        self.filas: dict = {}
        self.auth = _Auth()

    def table(self, nombre):
        assert nombre == "respuestas_chat"
        return _Consulta(self.filas)


# ── Un flujo lento que dice si lo cancelaron ──────────────────────────────────

RASTRO: dict = {}


async def flujo_lento(clave: str, n: int, pausa: float):
    RASTRO[clave] = {"trozos": 0, "cancelado": False, "termino": False}
    try:
        for i in range(n):
            await asyncio.sleep(pausa)
            RASTRO[clave]["trozos"] += 1
            yield f"data: {json.dumps({'token': f'[{i}]'})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
        RASTRO[clave]["termino"] = True
    except BaseException:
        # Aquí /analyze-document devuelve la consulta si no hubo una letra.
        RASTRO[clave]["cancelado"] = True
        raise


app = FastAPI()


@app.post("/analizar")
async def analizar(request: Request):
    d = await request.json()
    return StreamingResponse(
        rg.con_respaldo(
            flujo_lento(d["clave"], d["n"], d["pausa"]),
            authorization=request.headers.get("authorization"),
            peticion_id=d.get("peticion_id"),
            conversation_id=None, expediente_id=None,
            pregunta="📄 **Documento adjunto:** prueba.pdf\n\nAnaliza", estado="JALISCO",
            formato="documento",
            tope_sin_pantalla_s=d.get("tope", rg.TOPE_SIN_PANTALLA_S),
            mirar_cada_s=0.3,
        ),
        media_type="text/event-stream",
    )


def puerto_libre() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


FALLAS = 0


def ok(cond, msg):
    global FALLAS
    print(("  ✔ " if cond else "  ✘ ") + msg)
    if not cond:
        FALLAS += 1


def esperar(cond, segundos):
    fin = time.time() + segundos
    while time.time() < fin:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


def pedir(base, clave, n, pausa, *, leer=None, token="valido", peticion=None, tope=None):
    """Pide un análisis; con `leer`, cuelga tras ese número de trozos."""
    peticion = peticion or str(uuid.uuid4())
    cuerpo = {"clave": clave, "n": n, "pausa": pausa, "peticion_id": peticion}
    if tope is not None:
        cuerpo["tope"] = tope
    leidos = []
    with httpx.Client(timeout=30) as c:
        with c.stream("POST", f"{base}/analizar", json=cuerpo,
                      headers={"authorization": f"Bearer {token}"} if token else {}) as r:
            for linea in r.iter_lines():
                if linea.startswith("data: "):
                    leidos.append(linea)
                    if leer is not None and len(leidos) >= leer:
                        break   # colgar a media respuesta
    return peticion, leidos


def main():
    base_falsa = BaseFalsa()
    rg.configurar(base_falsa)
    puerto = puerto_libre()
    servidor = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=puerto, log_level="warning"))
    hilo = threading.Thread(target=servidor.run, daemon=True)
    hilo.start()
    assert esperar(lambda: servidor.started, 10), "uvicorn no arrancó"
    base = f"http://127.0.0.1:{puerto}"
    filas = base_falsa.filas

    print("1) La pantalla se queda hasta el final")
    pid, leidos = pedir(base, "a", 5, 0.05)
    ok(len(leidos) == 6, f"la pantalla recibe los 5 trozos y el final ({len(leidos)})")
    ok(esperar(lambda: filas.get(pid, {}).get("estado") == "terminada", 3), "copia terminada")
    ok(json.loads(json.dumps(filas[pid]["respuesta"])) == "[0][1][2][3][4]", "con todo el texto")

    print("2) La pantalla se va a los dos trozos (el teléfono que se va a WhatsApp)")
    pid, leidos = pedir(base, "b", 20, 0.1, leer=2)
    ok(len(leidos) == 2, "la pantalla sólo vio dos trozos")
    ok(esperar(lambda: RASTRO["b"]["termino"], 8), "el servidor siguió escribiendo hasta el final")
    ok(not RASTRO["b"]["cancelado"], "y nadie canceló el flujo (no hay devolución de consulta)")
    ok(esperar(lambda: filas.get(pid, {}).get("estado") == "terminada", 3), "la copia queda terminada")
    ok(filas[pid]["respuesta"] == "".join(f"[{i}]" for i in range(20)), "con los 20 trozos")

    print("3) La pantalla se va y luego pulsa Detener")
    pid, _ = pedir(base, "c", 200, 0.1, leer=2)
    time.sleep(0.5)
    ok(filas[pid]["estado"] == "en_curso", "sigue en curso sin pantalla")
    # Lo que hace la pantalla al detener: su copia «en_curso» → «interrumpida».
    filas[pid]["estado"] = "interrumpida"
    ok(esperar(lambda: RASTRO["c"]["cancelado"], 3), "el servidor corta el flujo (y ahí devolvería la consulta)")
    ok(not RASTRO["c"]["termino"], "sin llegar al final")
    ok(esperar(lambda: filas[pid].get("terminada_at") is not None, 3), "la copia se cierra")
    ok(filas[pid]["estado"] == "interrumpida" and 0 < len(filas[pid]["respuesta"] or "") < 600,
       f"interrumpida, con lo escrito hasta ahí ({len(filas[pid]['respuesta'] or '')} car.)")

    print("4) Tope sin pantalla")
    pid, _ = pedir(base, "d", 200, 0.1, leer=1, tope=1.0)
    ok(esperar(lambda: RASTRO["d"]["cancelado"], 4), "pasado el tope se corta")
    ok(esperar(lambda: filas.get(pid, {}).get("estado") == "interrumpida", 3), "y la copia queda interrumpida")

    print("5) Sin copia (sin token): como antes, cerrar la conexión corta el flujo")
    pid, _ = pedir(base, "e", 200, 0.1, leer=2, token=None)
    ok(esperar(lambda: RASTRO["e"]["cancelado"], 4), "el flujo se cancela al irse la pantalla")
    ok(pid not in filas, "y no hay copia")

    print("6) El apagado espera lo que se escribe sin pantalla")
    pid, _ = pedir(base, "f", 8, 0.1, leer=1)
    fut = asyncio.run_coroutine_threadsafe(rg.esperar_productores(5), _bucle(servidor))
    fut.result(timeout=10)
    ok(RASTRO["f"]["termino"], "terminó antes de apagar")
    ok(esperar(lambda: filas.get(pid, {}).get("estado") == "terminada", 3), "con su copia terminada")

    servidor.should_exit = True
    hilo.join(timeout=10)
    print(f"\n{FALLAS} FALLAS" if FALLAS else "\nTodo en verde")
    sys.exit(1 if FALLAS else 0)


def _bucle(servidor):
    """El bucle de eventos en que corre uvicorn (el del hilo)."""
    for t in rg._productores:
        return t.get_loop()
    # Sin productores vivos: cualquier tarea de fondo sirve para hallarlo.
    for t in rg._tareas:
        return t.get_loop()
    raise RuntimeError("no encontré el bucle de uvicorn")


if __name__ == "__main__":
    main()
