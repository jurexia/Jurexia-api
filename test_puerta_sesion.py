#!/usr/bin/env python3
"""La puerta de la sesión — 29-oct-2026.

El incidente: sin token, la API de producción contestaba con los datos de una
cuenta a quien supiera su correo (/taller/estado, /taller/en-curso,
/taller/sise-pendiente, el .docx de /taller/descargar…) y dejaba usar su cuota.
Esta prueba fija la puerta que lo cierra (puerta_sesion.py):

  · sin token → 401, en TODO /taller/ (salvo el catálogo) y en toda ruta que
    diga `user_email` o `user_id`, venga donde venga;
  · con el token de otra cuenta → 403, aunque se cuele también el correo
    propio;
  · con el suyo → pasa, y el endpoint lee la misma cuenta que el token;
  · lo público sigue abierto, y la administración por su vía.

La app REAL de main.py, sin arranque (el lifespan borra las cachés de Gemini de
producción: nada de `with TestClient(main.app)`), por httpx con el transporte
ASGI. Sin red y sin gasto: Supabase y el motor son de mentira, las llaves son
falsas a la fuerza y Qdrant apunta a un puerto cerrado.

    .venv/bin/python test_puerta_sesion.py
"""
import asyncio
import base64
import contextlib
import io
import json
import os
import sys
import time
import types
import uuid

import dotenv
import httpx

dotenv.load_dotenv = lambda *a, **k: False
for _k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY",
           "MISTRAL_API_KEY", "ELEVENLABS_API_KEY", "AZURE_DOCINT_KEY", "AZURE_SPEECH_KEY"):
    os.environ[_k] = "falsa-para-pruebas"
os.environ.update(QDRANT_URL="http://127.0.0.1:9", QDRANT_API_KEY="",
                  SUPABASE_URL="", SUPABASE_SERVICE_KEY="")
os.environ.pop("SESION_EXIGIR", None)

sys.path.insert(0, ".")
with contextlib.redirect_stdout(io.StringIO()):
    import main  # noqa: E402
assert main.supabase_admin is None, "main.py cargó un Supabase de verdad: la prueba no sigue"
import puerta_sesion as ps  # noqa: E402


async def _ocr_prohibido(*_a, **_k):
    raise AssertionError("OCR en una prueba sin red")
main._extract_text_from_upload = _ocr_prohibido
main.get_gemini_client = lambda *a, **k: (_ for _ in ()).throw(AssertionError("Gemini en una prueba sin red"))
import rate_limiter  # noqa: E402
from fastapi import Form, Request  # noqa: E402

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


# ── Tokens con forma de JWT (la puerta descarta lo que no la tiene) ─────────

def _b64(d: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")


def jwt(sub: str, exp: float = None) -> str:
    exp = time.time() + 3600 if exp is None else exp
    return (f"{_b64({'alg': 'HS256', 'typ': 'JWT'})}."
            f"{_b64({'sub': sub, 'exp': exp, 'role': 'authenticated'})}.firma-{uuid.uuid4().hex[:8]}")


A, B, ADMIN = "secretaria@tribunal.mx", "victima@tribunal.mx", "jdm.juridico@gmail.com"
A_ID, B_ID, ADMIN_ID = (str(uuid.uuid4()) for _ in range(3))
T_A, T_B, T_ADMIN = jwt(A_ID), jwt(B_ID), jwt(ADMIN_ID)
CAIDO = jwt("caido")
USUARIOS = {T_A: (A_ID, A), T_B: (B_ID, B), T_ADMIN: (ADMIN_ID, ADMIN)}
LLAMADAS = []


class ErrorApi(Exception):
    """Como AuthApiError de supabase_auth: trae el `status` de la respuesta."""
    def __init__(self, status):
        super().__init__(f"status {status}")
        self.status = status


class _Auth:
    def get_user(self, tok):
        LLAMADAS.append(tok)
        if tok == CAIDO:
            raise ConnectionError("Supabase no contesta")
        if tok not in USUARIOS:
            raise ErrorApi(401)
        uid, correo = USUARIOS[tok]
        return types.SimpleNamespace(user=types.SimpleNamespace(id=uid, email=correo))


# ── Un Supabase en memoria: filas por tabla, filtros por igualdad ───────────

class _Consulta:
    def __init__(self, sb, tabla):
        self.sb, self.tabla, self.filtros, self.op, self.datos = sb, tabla, [], "select", None

    def eq(self, col, val):
        self.filtros.append((col, val))
        return self

    def update(self, datos):
        self.op, self.datos = "update", datos
        return self

    def insert(self, datos):
        self.op, self.datos = "insert", datos
        return self

    def __getattr__(self, _nombre):       # select, limit, order, lt, not_, like…
        return self

    def __call__(self, *a, **k):
        return self

    def execute(self):
        self.sb.ops.append((self.tabla, self.op, dict(self.filtros), self.datos))
        filas = [f for f in self.sb.filas.get(self.tabla, [])
                 if all(f.get(c) == v for c, v in self.filtros)]
        return types.SimpleNamespace(data=filas if self.op == "select" else [])


class _Cubo:
    def __init__(self, sb):
        self.sb = sb

    def download(self, ruta):
        if ruta in self.sb.archivos:
            return self.sb.archivos[ruta]
        raise RuntimeError("no está")

    def __getattr__(self, _nombre):
        return lambda *a, **k: []


class SbFalso:
    def __init__(self):
        self.auth = _Auth()
        self.ops, self.rpcs, self.archivos = [], [], {}
        self.filas = {"user_profiles": [
            {"id": A_ID, "email": A, "subscription_type": "ultra_secretarios", "estado": None},
            {"id": B_ID, "email": B, "subscription_type": "ultra_secretarios", "estado": None},
        ]}
        self.storage = types.SimpleNamespace(from_=lambda _cubo: _Cubo(self))

    def table(self, nombre):
        return _Consulta(self, nombre)

    def rpc(self, nombre, params):
        self.rpcs.append((nombre, dict(params)))
        datos = {"consume_query": {"allowed": True, "used": 1, "limit": 100},
                 "get_quota_status": {"used": 3, "limit": 100, "subscription_type": "pro_monthly"},
                 }.get(nombre, False)
        return types.SimpleNamespace(execute=lambda: types.SimpleNamespace(data=datos))

    def cobros(self):
        return [p.get("p_user_id") for n, p in self.rpcs if n == "consume_query"]


SB = SbFalso()
main.supabase_admin = SB


# Rutas de eco, montadas DESPUÉS de crear la app: si también las cubre la
# puerta, la puerta es de la app y no de una lista de endpoints.
@main.app.get("/taller/__eco")
async def _eco_taller(request: Request, user_email: str = ""):
    s = ps.sesion_de(request)
    return {"sesion": s.correo if s else None, "user_email": user_email}


@main.app.post("/__eco_libre")
async def _eco_libre(request: Request, user_email: str = Form(""), user_id: str = Form("")):
    s = ps.sesion_de(request)
    return {"sesion": s.correo if s else None, "user_email": user_email, "user_id": user_id}


def con(tok, **mas):
    return {"Authorization": f"Bearer {tok}", **mas}


async def _pedir(metodo, ruta, **kw):
    rate_limiter._rate_limiter._requests.clear()
    transporte = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transporte, base_url="http://prueba", timeout=60) as c:
        with contextlib.redirect_stdout(io.StringIO()) as registro:
            r = await c.request(metodo, ruta, **kw)
    r.registro = registro.getvalue()
    return r


def pedir(metodo, ruta, **kw):
    return asyncio.run(_pedir(metodo, ruta, **kw))


def detalle(r):
    try:
        d = r.json().get("detail")
    except Exception:
        return r.text[:80]
    return d if isinstance(d, str) else json.dumps(d)[:80]


# ── Qué rutas dicen de quién son: se leen de la propia app ──────────────────

def _rutas():
    """(plantilla, método, dónde viaja la identidad, nombre) de cada ruta de la
    app, incluidas las de los routers de la Bóveda, los flujos y el respaldo."""
    import fastapi.routing as fr
    planas = []

    def recorrer(rutas):
        for r in rutas:
            if isinstance(r, fr.APIRoute):
                planas.append(r)
            elif hasattr(r, "effective_candidates"):          # _IncludedRouter
                for c in r.effective_candidates():
                    original = getattr(c, "original_route", None) or c
                    if isinstance(original, fr.APIRoute):
                        planas.append(original)
                    elif hasattr(c, "effective_candidates"):
                        recorrer([c])
            elif hasattr(r, "routes"):
                recorrer(r.routes)
    recorrer(main.app.routes)
    fuera = []
    for r in planas:
        d = r.dependant
        donde = {}
        for p in d.query_params:
            if p.name in ("user_email", "user_id"):
                donde[p.name] = "query"
        for p in d.path_params:
            if p.name in ("user_email", "user_id"):
                donde[p.name] = "path"
        for p in d.body_params:
            if p.name in ("user_email", "user_id"):
                donde[p.name] = "form"
            anotacion = getattr(p.field_info, "annotation", None) or getattr(p, "type_", None)
            campos = getattr(anotacion, "model_fields", {}) or {}
            for n in ("user_email", "user_id"):
                if n in campos:
                    donde[n] = "json"
        for m in sorted(r.methods - {"HEAD"}):
            fuera.append((r.path, m, donde))
    return fuera


# Las que leen la identidad a mano (request.form(), request.json() o un
# `payload: dict`): la firma no la dice, se declara aquí.
A_MANO = {
    ("/taller/origen", "POST"): {"user_email": "json"},
    ("/redactor-sentencia/v2/generate", "POST"): {"user_email": "form"},
    ("/redaccion-sentencias-estatal", "POST"): {"user_email": "form"},
    ("/redactor/tcc-v4/regenerate-summary", "POST"): {"user_email": "json"},
    ("/redactor/tcc-v4/finalize", "POST"): {"user_email": "json"},
}

RUTAS = _rutas()
for (ruta, metodo), donde in A_MANO.items():
    for i, (r, m, d) in enumerate(RUTAS):
        if r == ruta and m == metodo:
            RUTAS[i] = (r, m, {**d, **donde})


def _ruta_concreta(plantilla, valores):
    import re
    return re.sub(r"\{([^}:]+)(:[^}]*)?\}", lambda m: valores.get(m.group(1), "x-1"), plantilla)


def _peticion(ruta, metodo, donde, correo, uid):
    """La petición que dice ser de (correo, uid), cada dato en su sitio."""
    valores = {"user_email": correo, "user_id": uid}
    kw = {"params": {}, "headers": {}}
    form, cuerpo = {}, {}
    for nombre, sitio in donde.items():
        if sitio == "query":
            kw["params"][nombre] = valores[nombre]
        elif sitio == "form":
            form[nombre] = valores[nombre]
        elif sitio == "json":
            cuerpo[nombre] = valores[nombre]
    if form:
        kw["files"] = {"_": (None, "")}
        kw["data"] = form
    elif cuerpo:
        kw["json"] = cuerpo
    return metodo, _ruta_concreta(ruta, valores), kw


print("\n1 · LAS RUTAS QUE DICEN DE QUIÉN SON, SIN TOKEN → 401")
dicen = [(r, m, d) for r, m, d in RUTAS if d and r not in ps.NO_SE_MIRA
         and r not in ps.TOLERAN_ANONIMO and "__eco" not in r]
ok(len(dicen) >= 50, f"la app tiene {len(dicen)} rutas que dicen user_email o user_id (≥50)")
escapan = []
for ruta, metodo, donde in dicen:
    m, u, kw = _peticion(ruta, metodo, donde, B, B_ID)
    r = pedir(m, u, **kw)
    if r.status_code != 401 or r.headers.get("x-sesion") != "falta":
        escapan.append(f"{metodo} {ruta} → {r.status_code}")
ok(not escapan, "todas contestan 401 de la puerta" + (": " + "; ".join(escapan) if escapan else ""))
ok(not LLAMADAS, "y sin token no se le pregunta nada a Supabase")
nombres = {r for r, _m, _d in dicen}
for esperada in ("/taller/estado", "/taller/en-curso", "/taller/descargar", "/taller/sise-pendiente",
                 "/taller/olvidar", "/taller/adelanto", "/quota/status/{user_id}", "/chat",
                 "/conversacion/continuar", "/chat-sentencia", "/api/jurimetria",
                 "/redactor/tcc-beta/generate", "/precedentes/gratis", "/taller/origen"):
    ok(esperada in nombres, f"{esperada} está entre ellas")

print("\n2 · TODO /taller/ EXIGE SESIÓN, DIGA O NO DE QUIÉN ES (salvo el catálogo)")
taller = sorted({(r, m) for r, m, _d in RUTAS if r.startswith("/taller/") and not r.startswith("/taller/__")})
ok(len(taller) >= 33, f"{len(taller)} rutas /taller/ (≥33)")
escapan = []
for ruta, metodo in taller:
    if ruta in ps.NO_SE_MIRA:
        continue
    r = pedir(metodo, ruta)
    if r.status_code != 401:
        escapan.append(f"{metodo} {ruta} → {r.status_code}")
ok(not escapan, "sin token y sin correo, todas 401" + (": " + "; ".join(escapan) if escapan else ""))
r = pedir("POST", "/taller/desde-sise", files={"promocion": ("p.pdf", b"%PDF-1.4", "application/pdf")},
          data={"numero": "1/2026", "user_email": B})
ok(r.status_code == 401 and "complemento" in detalle(r),
   f"el complemento viejo (sin token) recibe su propio 401: «{detalle(r)[:60]}…»")

print("\n3 · LO PÚBLICO SIGUE ABIERTO")
for metodo, ruta, kw in (("GET", "/health", {}), ("GET", "/api/wake", {}), ("GET", "/taller/tipos", {}),
                         ("GET", "/taller/reglas-surtimiento", {}), ("GET", "/taller/extension", {}),
                         ("GET", "/plazos/tabla", {}), ("POST", "/gratis/sesion", {}),
                         ("GET", "/boveda/columnas-sugeridas", {}), ("GET", "/salud/recuperacion", {})):
    r = pedir(metodo, ruta, **kw)
    ok(r.status_code not in (401, 403) and not r.headers.get("x-sesion"),
       f"{metodo} {ruta} sin token → {r.status_code}")
r = pedir("POST", "/chat/traba", content=json.dumps({"etapa": "fetch", "ms": 9000, "user_id": B_ID}),
          headers={"Content-Type": "text/plain"})
ok(r.status_code == 200 and r.json().get("ok"), "/chat/traba (sendBeacon, con user_id y sin token) → 200")
r = pedir("POST", "/precedentes/gratis", json={"consulta": "suspensión contra embargo", "testigo": "a.b"})
ok(r.status_code not in (401, 403), f"/precedentes/gratis anónimo (testigo, sin correo) no pide sesión → {r.status_code}")
ok(not LLAMADAS, "ninguna ruta pública le preguntó a Supabase por un token")

print("\n4 · EL TOKEN DE A PIDIENDO LO DE B → 403")
casos = [
    ("GET", "/taller/estado", {"params": {"user_email": B}}),
    ("GET", "/taller/en-curso", {"params": {"user_email": B}}),
    ("GET", "/taller/sise-pendiente", {"params": {"user_email": B}}),
    ("GET", "/taller/descargar", {"params": {"numero": "466/2025", "user_email": B}}),
    ("GET", "/taller/proyecto", {"params": {"numero": "466/2025", "user_email": B}}),
    ("GET", "/taller/documentos", {"params": {"numero": "466/2025", "user_email": B}}),
    ("POST", "/taller/olvidar", {"data": {"numero": "466/2025", "user_email": B}}),
    ("POST", "/taller/origen", {"json": {"numero": "466/2025", "user_email": B}}),
    ("GET", f"/quota/status/{B_ID}", {}),
    ("POST", "/chat", {"json": {"messages": [{"role": "user", "content": "hola"}], "user_id": B_ID}}),
    ("POST", "/conversacion/continuar", {"json": {"user_id": B_ID, "messages": [{"role": "user", "content": "x"}]}}),
    ("POST", "/api/jurimetria", {"data": {"user_email": B, "descripcion": "despido"}}),
    ("POST", "/redactor/tcc-v4/finalize", {"json": {"user_email": B, "job_id": "x"}}),
    ("POST", "/redactor-sentencia/v2/generate", {"data": {"user_email": B}}),
    ("POST", "/precedentes/gratis", {"json": {"consulta": "suspensión contra embargo", "user_email": B}}),
    ("POST", "/analyze-document", {"files": {"file": ("a.pdf", b"%PDF-1.4", "application/pdf")},
                                   "data": {"user_id": B_ID}}),
]
for metodo, ruta, kw in casos:
    r = pedir(metodo, ruta, headers=con(T_A), **kw)
    ok(r.status_code == 403 and r.headers.get("x-sesion") == "ajena",
       f"{metodo} {ruta} a nombre de B con el token de A → {r.status_code}")
ok(not [x for x in SB.rpcs if x[0] == "consume_query"], "y a B no se le cobró nada")
r = pedir("POST", "/__eco_libre", params={"user_email": A}, data={"user_email": B}, headers=con(T_A))
ok(r.status_code == 403, f"el correo propio en la URL y el ajeno en el formulario → {r.status_code}")
r = pedir("GET", f"/taller/__eco?user_email={A}&user_email={B}", headers=con(T_A))
ok(r.status_code == 403, f"dos user_email en la URL, uno ajeno → {r.status_code}")
r = pedir("POST", "/__eco_libre", data={"user_email": A, "user_id": B_ID}, headers=con(T_A))
ok(r.status_code == 403, f"el correo propio con el id ajeno → {r.status_code}")
r = pedir("GET", "/taller/__eco", params={"user_email": f"  {A.upper()} "}, headers=con(T_A))
ok(r.status_code == 200, "mayúsculas y espacios no son otra cuenta")
r = pedir("POST", "/admin/users/" + B_ID + "/block", headers=con(T_A))
ok(r.status_code == 403 and r.headers.get("x-sesion") is None,
   f"administración con un token que no es de administración → 403 de _verify_admin ({r.status_code})")

print("\n5 · EL TOKEN DE A CON LO SUYO → PASA, Y EL ENDPOINT LEE LA MISMA CUENTA")
r = pedir("GET", "/taller/__eco", params={"user_email": A}, headers=con(T_A))
ok(r.status_code == 200 and r.json() == {"sesion": A, "user_email": A},
   f"GET de eco con su token → {r.status_code} {r.text[:80]}")
r = pedir("POST", "/__eco_libre", data={"user_email": A, "user_id": A_ID}, headers=con(T_A))
ok(r.status_code == 200 and r.json()["sesion"] == A, f"POST de formulario con su correo y su id → {r.status_code}")
r = pedir("GET", "/taller/estado", params={"user_email": A}, headers=con(T_A))
ok(r.status_code == 200 and r.json().get("tiene_acceso") is True,
   f"/taller/estado con su token → {r.status_code} (tiene_acceso={r.json().get('tiene_acceso') if r.status_code == 200 else '—'})")
r = pedir("GET", "/taller/en-curso", params={"user_email": A}, headers=con(T_A))
ok(r.status_code == 200 and "asuntos" in r.json(), f"/taller/en-curso con su token → {r.status_code}")
SB.archivos[main._ruta_taller(A, "466/2025")] = b"PK\x03\x04proyecto de A"
SB.archivos[main._ruta_taller(B, "466/2025")] = b"PK\x03\x04proyecto de B"
r = pedir("GET", "/taller/descargar", params={"numero": "466/2025", "user_email": A}, headers=con(T_A))
ok(r.status_code == 200 and r.content == b"PK\x03\x04proyecto de A",
   f"/taller/descargar con su token → {r.status_code}, y es SU proyecto")
r = pedir("GET", "/taller/descargar", params={"numero": "466/2025", "user_email": B}, headers=con(T_A))
ok(r.status_code == 403 and b"proyecto de B" not in r.content, "el de B, con el token de A, no sale")
r = pedir("GET", "/taller/descargar", params={"numero": "466/2025", "user_email": B})
ok(r.status_code == 401 and b"proyecto de B" not in r.content, "ni sin token (la prueba del incidente)")
r = pedir("GET", f"/quota/status/{A_ID}", headers=con(T_A))
ok(r.status_code == 200 and r.json().get("limit") == 100, f"/quota/status de su id con su token → {r.status_code}")
r = pedir("POST", "/admin/users/" + B_ID + "/block", headers=con(T_ADMIN))
ok(r.status_code not in (401, 403), f"administración bloquea a B con SU token de administración → {r.status_code}")

print("\n6 · /analyze-document: LA APP MÓVIL SIN TOKEN SE ATIENDE COMO ANÓNIMA")
import fitz  # noqa: E402

ANALISIS = ["## Los agravios\n\n", "Dos agravios."]


def _trozo(texto=None, fin=None):
    delta = types.SimpleNamespace(content=texto)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta, finish_reason=fin, model_extra={})])


class _Motor:
    def __init__(self):
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        async def gen():
            for t in ANALISIS:
                yield _trozo(t)
            yield _trozo(None, "stop")
        return gen()


def _pdf() -> bytes:
    """Dos páginas con texto seleccionable que pasa `_texto_nativo_sirve` (el
    de test_documento_respaldo.py): sin OCR, que aquí está prohibido."""
    d = fitz.open()
    parrafo = ("PRIMER AGRAVIO. La sentencia recurrida valoró indebidamente la prueba testimonial "
               "ofrecida por la parte actora, pues omitió considerar las contradicciones en que "
               "incurrieron los declarantes respecto de la fecha, el lugar y las circunstancias del "
               "acto reclamado. SEGUNDO AGRAVIO. El juzgador de origen carecía de competencia por razón "
               "de territorio, ya que el domicilio de la demandada se ubica en Apizaco, Tlaxcala, y no "
               "en la capital del estado como erróneamente se asentó en el auto admisorio. TERCERO. "
               "Se violaron las formalidades esenciales del procedimiento al negarse el desahogo de la "
               "pericial en materia de grafoscopía ofrecida oportunamente.")
    for i in range(2):
        p = d.new_page()
        p.insert_textbox(fitz.Rect(50, 50, 550, 800), f"Hoja {i + 1}. " + parrafo, fontsize=10)
    return d.tobytes()


main._via_documento = lambda modelo, esfuerzo=None: (_Motor(), {})
main.DOCUMENTO_LATIDO_S = 0.05


def analizar(cabeceras, user_id):
    SB.rpcs.clear()
    return pedir("POST", "/analyze-document", headers=cabeceras,
                 files={"file": ("agravios.pdf", _pdf(), "application/pdf")},
                 data={"prompt": "¿Qué agravios hace valer?", "usar_acervo": "0", "user_id": user_id})


r = analizar({}, B_ID)
ok(r.status_code == 200 and "Dos agravios" in r.text, f"sin token, con el id de B: el análisis sale ({r.status_code})")
ok(B_ID not in SB.cobros(), f"…pero a B no se le cobra (cobros: {SB.cobros()})")
ok("se atiende como anónimo" in r.registro, "y queda el renglón que cuenta cuántas veces pasa")
r = analizar(con(T_A), A_ID)
ok(r.status_code == 200 and "Dos agravios" in r.text and SB.cobros() == [A_ID],
   f"con su token y su id: el análisis sale y se le cobra a A, y sólo a A ({SB.cobros()})")
r = analizar(con(T_A), B_ID)
ok(r.status_code == 403 and not SB.cobros(), f"con el token de A y el id de B → {r.status_code}, sin cobro")

print("\n7 · TOKENS QUE NO VALEN, Y UN SUPABASE QUE NO CONTESTA")
LLAMADAS.clear()
r = pedir("GET", "/taller/__eco", params={"user_email": A}, headers=con(jwt("nadie")))
ok(r.status_code == 401 and r.headers.get("x-sesion") == "caducada", f"token que Supabase rechaza → {r.status_code}")
r = pedir("GET", "/taller/__eco", params={"user_email": A}, headers=con("undefined"))
ok(r.status_code == 401, f"«Bearer undefined» → {r.status_code}")
r = pedir("GET", "/taller/__eco", params={"user_email": A}, headers=con(jwt(A_ID, exp=time.time() - 120)))
ok(r.status_code == 401, f"un JWT vencido → {r.status_code}")
ok(len(LLAMADAS) == 1, f"ni el que no es JWT ni el vencido gastan viaje a Supabase ({len(LLAMADAS)} viaje)")
r = pedir("GET", "/taller/__eco", params={"user_email": A}, headers=con(CAIDO))
ok(r.status_code == 503, f"Supabase caído → {r.status_code} (no se le dice que su sesión caducó)")
r = pedir("GET", "/taller/__eco", params={"user_email": A}, headers={"Authorization": f"bearer   {T_A}"})
ok(r.status_code == 200, "«bearer» en minúsculas y con espacios también vale")
viejo = main.supabase_admin
main.supabase_admin = None
ps._CACHE.clear()
r = pedir("GET", "/taller/__eco", params={"user_email": A}, headers=con(T_A))
ok(r.status_code == 503, f"sin cliente de Supabase → {r.status_code}")
main.supabase_admin = viejo

print("\n8 · LA CACHÉ: UNA LLAMADA A SUPABASE POR TOKEN, Y NUNCA MÁS QUE EL TOKEN")
ps._CACHE.clear()
LLAMADAS.clear()
for _ in range(5):
    pedir("GET", "/taller/__eco", params={"user_email": A}, headers=con(T_A))
ok(LLAMADAS.count(T_A) == 1, f"cinco sondeos, {LLAMADAS.count(T_A)} viaje(s) a Supabase")
MALO = jwt("nadie")
for _ in range(3):
    pedir("GET", "/taller/__eco", headers=con(MALO))
ok(LLAMADAS.count(MALO) == 1, "un token rechazado tampoco repite viaje durante 30 s")
CORTO = jwt(A_ID, exp=time.time() + 1)
USUARIOS[CORTO] = (A_ID, A)
pedir("GET", "/taller/__eco", headers=con(CORTO))
time.sleep(1.2)
USUARIOS.pop(CORTO)
r = pedir("GET", "/taller/__eco", headers=con(CORTO))
ok(r.status_code == 401, "la caché no sobrevive al `exp` del token")

print("\n9 · LA PALANCA DE EMERGENCIA: SESION_EXIGIR=0")
ps.EXIGIR = False
r = pedir("GET", "/taller/__eco", params={"user_email": B})
ok(r.status_code == 200 and r.json()["sesion"] is None and "SESION_EXIGIR=0" in r.registro,
   f"sin token pasa, con aviso en el registro → {r.status_code}")
r = pedir("GET", "/taller/__eco", params={"user_email": B}, headers=con(T_A))
ok(r.status_code == 403, "pero un token de otra cuenta sigue siendo 403")
r = pedir("GET", "/taller/__eco", headers=con(jwt("nadie")))
ok(r.status_code == 401, "y un token que no vale sigue siendo 401")
r = pedir("POST", "/connect/lawyers/index", json={"cedula_number": "123", "bio": "x"})
ok(r.status_code == 401, "el directorio de abogados no se abre con la palanca")
ps.EXIGIR = True

print("\n10 · EL DIRECTORIO DE ABOGADOS: SÓLO EL PERFIL PROPIO, SÓLO SUS CAMPOS")
SB.filas["lawyer_profiles"] = [{"id": A_ID, "cedula_number": "111"}, {"id": B_ID, "cedula_number": "222"}]
SB.ops.clear()
r = pedir("POST", "/connect/lawyers/index", json={"cedula_number": "222", "bio": "hackeado",
                                                  "verification_status": "verified"})
ok(r.status_code == 401, f"sin token → {r.status_code}")
r = pedir("POST", "/connect/lawyers/index", headers=con(T_A),
          json={"cedula_number": "222", "bio": "hackeado", "verification_status": "verified"})
ok(r.status_code == 200 and r.json().get("indexed") is False and
   not [o for o in SB.ops if o[0] == "lawyer_profiles" and o[1] == "update"],
   f"con el token de A y la cédula de B → no se toca nada ({r.json()})")
r = pedir("POST", "/connect/lawyers/index", headers=con(T_A),
          json={"cedula_number": "111", "bio": "mi bio", "verification_status": "verified", "is_pro_active": True})
cambios = [o for o in SB.ops if o[0] == "lawyer_profiles" and o[1] == "update"]
ok(r.status_code == 200 and len(cambios) == 1 and cambios[0][2] == {"id": A_ID, "cedula_number": "111"}
   and cambios[0][3] == {"bio": "mi bio"},
   f"con su cédula → sólo su fila y sólo la bio ({cambios[0][2:] if cambios else '—'})")

print("\n11 · EL NAVEGADOR PUEDE LEER EL 401 Y SU CABECERA")
r = pedir("GET", "/taller/estado", params={"user_email": B}, headers={"Origin": "https://www.iurexia.com"})
ok(r.status_code == 401 and r.headers.get("access-control-allow-origin") == "*",
   "el 401 lleva Access-Control-Allow-Origin (si no, el navegador lo vería como «Failed to fetch»)")
ok("x-sesion" in (r.headers.get("access-control-expose-headers") or "").lower(), "y expone X-Sesion")
r = pedir("OPTIONS", "/taller/estado", headers={"Origin": "https://www.iurexia.com",
                                                "Access-Control-Request-Method": "GET",
                                                "Access-Control-Request-Headers": "authorization"})
ok(r.status_code == 200 and "authorization" in (r.headers.get("access-control-allow-headers") or "").lower(),
   f"el preflight con Authorization pasa ({r.status_code})")

print("\n12 · EL REGISTRO NO LLEVA EL CORREO ENTERO")
r = pedir("GET", "/taller/estado", params={"user_email": B}, headers=con(T_A))
ok(B not in r.registro and "vi…@tribunal.mx" in r.registro, "el 403 se apunta con el correo enmascarado")
ok(ps.enmascarar("") == "(sin correo)" and ps.enmascarar("jdm.juridico@gmail.com") == "jd…@gmail.com",
   "enmascarar")

print("\n13 · LA PUERTA LEE EL CUERPO IGUAL QUE QUIEN LO USA (sin rendijas)")
cuerpo_chat = json.dumps({"messages": [{"role": "user", "content": "hola"}], "user_id": B_ID})
for tipo in ("application/json; charset=utf-8", "application/merge-patch+json; charset=utf-8",
             "Application/JSON"):
    r = pedir("POST", "/chat", content=cuerpo_chat, headers={"Content-Type": tipo})
    ok(r.status_code == 401, f"/chat con «{tipo}» y el id de B, sin token → {r.status_code}")
r = pedir("POST", "/taller/origen", content=json.dumps({"numero": "1/2026", "user_email": B}),
          headers=con(T_A, **{"Content-Type": "text/plain; x=application/json"}))
ok(r.status_code == 403, f"/taller/origen lee JSON si el tipo CONTIENE application/json: la puerta también → {r.status_code}")
r = pedir("POST", "/chat", headers=con(T_A),
          json={"messages": [{"role": "user", "content": "hola"}], "user_id": [A_ID, B_ID]})
ok(r.status_code == 403, f"user_id como lista con uno ajeno → {r.status_code}")
r = pedir("POST", "/__eco_libre", headers=con(T_A), files={"user_email": ("x.txt", b"victima", "text/plain")})
ok(r.status_code == 403, f"user_email como archivo en el formulario → {r.status_code} (no se cuela sin mirar)")
r = pedir("POST", "/__eco_libre", headers=con(T_A), data={"user_email": ["", A]})
ok(r.status_code == 200, f"un user_email vacío junto al propio no estorba → {r.status_code}")
r = pedir("POST", "/__eco_libre", data={"user_id": "undefined", "user_email": "null"})
ok(r.status_code == 200, f"«undefined»/«null» (el navegador sin nadie) no son una cuenta → {r.status_code}")

print(f"\n{'TODO PASA' if not FALLOS else f'{len(FALLOS)} FALLOS'}")
raise SystemExit(1 if FALLOS else 0)
