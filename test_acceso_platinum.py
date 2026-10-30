"""LAS HERRAMIENTAS DE PLATINUM, EN EL SERVIDOR — 12-oct-2026.

    .venv/bin/python test_acceso_platinum.py           # sin red
    .venv/bin/python test_acceso_platinum.py --real    # y además con la base real (sólo lee)

Lo que importa:
  · Pro no pasa; Platinum, Ultra y administración sí. La cortesía —los
    testers del dominio @iurexia.com y Jenica— pasa sólo con el correo
    verificado, y sin volverse administración: la Bóveda le cobra;
  · si la base no responde se cierra (503), no se abre;
  · en /chat (Sentencia y Precedentes) y en /toulmin/stream la puerta va
    ANTES del cobro: al rechazado no se le llama `consume_query`;
  · Jurimetría manda el token cuando llega.

Sin red: un Supabase falso y llaves falsas, y el `app` de main.py sin su
arranque (el lifespan borraría las cachés de Gemini de producción y
arrancaría el trabajador de la Bóveda). Con --real, `user_profiles` se lee de
la base de producción con las credenciales de ../jurexia-api-git/.env; nada
se escribe: cualquier otra tabla sale vacía y los `rpc` (el cobro) se
interceptan siempre.
"""
import asyncio
import os
import sys
import uuid

FALLOS = []
REAL = "--real" in sys.argv


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


# ── Un Supabase falso ───────────────────────────────────────────────────
class _Res:
    def __init__(self, data):
        self.data = data


class _Consulta:
    """select/eq/limit/execute sobre una lista de filas; cualquier otro
    método encadenado se acepta y no hace nada."""

    def __init__(self, sb, tabla):
        self.sb, self.tabla, self.filtros = sb, tabla, []

    def eq(self, campo, valor):
        self.filtros.append((campo, valor))
        return self

    def execute(self):
        if self.sb.caida:
            raise ConnectionError("base caída")
        filas = self.sb.tablas.get(self.tabla, [])
        return _Res([f for f in filas if all(f.get(c) == v for c, v in self.filtros)][:1])

    def __getattr__(self, _nombre):
        return lambda *a, **k: self


class _Usuario:
    def __init__(self, uid):
        self.id = uid
        self.email = next((p["email"] for p in PERFILES if p["id"] == uid), None)


def tok(uid):
    """Un token con forma de JWT (29-oct-2026): la puerta de la sesión
    (puerta_sesion.py) descarta sin preguntar lo que no la tiene, y desde
    entonces el `user_id` o el `user_email` de la petición se demuestran con
    el token de esa cuenta."""
    import base64
    import json
    import time
    carga = base64.urlsafe_b64encode(json.dumps({"sub": uid, "exp": time.time() + 3600}).encode())
    return f"e30.{carga.decode().rstrip('=')}.firma"


class _Auth:
    def get_user(self, token):
        if token.startswith("tok-"):
            uid = token[4:]
        elif token.count(".") == 2:
            import base64
            import json
            carga = token.split(".")[1]
            uid = json.loads(base64.urlsafe_b64decode(carga + "=" * (-len(carga) % 4)))["sub"]
        else:
            raise ValueError("token inválido")

        class R:
            user = _Usuario(uid)
        return R()


class SbFalso:
    def __init__(self, perfiles):
        self.tablas = {"user_profiles": perfiles}
        self.caida = False
        self.rpcs = []
        self.auth = _Auth()

    def table(self, nombre):
        return _Consulta(self, nombre)

    def rpc(self, nombre, params):
        self.rpcs.append((nombre, dict(params)))
        datos = {"allowed": True, "used": 1, "limit": 100} if nombre == "consume_query" else False

        class _E:
            def execute(_self):
                return _Res(datos)
        return _E()

    def cobros(self):
        return [p.get("p_user_id") for n, p in self.rpcs if n == "consume_query"]


V = "2026-07-27T00:00:00+00:00"
PERFILES = [
    {"id": "u-pro", "email": "abogada@ejemplo.mx", "subscription_type": "pro_monthly", "email_verificado_at": V},
    {"id": "u-plat", "email": "socio@ejemplo.mx", "subscription_type": "platinum_monthly", "email_verificado_at": V},
    {"id": "u-anual", "email": "anual@ejemplo.mx", "subscription_type": "platinum_annual", "email_verificado_at": V},
    {"id": "u-ultra", "email": "secretario@ejemplo.mx", "subscription_type": "ultra_secretarios", "email_verificado_at": V},
    {"id": "u-admin", "email": "jdm.juridico@gmail.com", "subscription_type": "gratuito", "email_verificado_at": None},
    {"id": "u-tester", "email": "Prueba@IUREXIA.com", "subscription_type": "gratuito", "email_verificado_at": V},
    {"id": "u-tester-sin", "email": "inventado@iurexia.com", "subscription_type": "gratuito", "email_verificado_at": None},
    {"id": "u-jenica", "email": "jenycampos@hotmail.com", "subscription_type": "pro_monthly", "email_verificado_at": V},
    {"id": "u-imitador", "email": "alguien@iurexia.com.mx", "subscription_type": "pro_monthly", "email_verificado_at": V},
]
ADMINS = {"administracion@iurexia.com", "jdm.juridico@gmail.com"}

import acceso_platinum as ap  # noqa: E402
import cobro_consultas as cc  # noqa: E402
from fastapi import HTTPException  # noqa: E402

print("\n── La regla ──")
ok(not ap.tiene_platinum("pro_monthly", "a@b.mx", ADMINS, True), "Pro no")
ok(not ap.tiene_platinum("pro_annual", "a@b.mx", ADMINS, True), "Pro anual no")
ok(not ap.tiene_platinum("gratuito", None), "gratuito sin correo no")
ok(not ap.tiene_platinum(None, None), "sin plan ni correo no")
for p in ("platinum_monthly", "platinum_annual", "ultra_secretarios"):
    ok(ap.tiene_platinum(p, "a@b.mx", ADMINS), f"{p} sí")
ok(ap.tiene_platinum("gratuito", " JDM.Juridico@gmail.com ", ADMINS), "administración sí (sin importar mayúsculas ni espacios)")
ok(ap.tiene_platinum("gratuito", "prueba@iurexia.com", ADMINS, True), "tester del dominio, verificado: sí")
ok(not ap.tiene_platinum("gratuito", "prueba@iurexia.com", ADMINS, False), "tester del dominio SIN verificar: no")
ok(ap.tiene_platinum("pro_monthly", "jenycampos@hotmail.com", ADMINS, True), "Jenica verificada: sí")
ok(not ap.tiene_platinum("pro_monthly", "jenycampos@hotmail.com", ADMINS, False), "Jenica sin verificar: no")
ok(not ap.tiene_platinum("pro_monthly", "alguien@iurexia.com.mx", ADMINS, True), "@iurexia.com.mx no es el dominio")
ok(not ap.tiene_platinum("pro_monthly", "iurexia.com@gmail.com", ADMINS, True), "«iurexia.com» antes de la arroba no cuenta")
ok(not ap.tiene_platinum("pro_monthly", "x@sub.iurexia.com", ADMINS, True), "un subdominio no cuenta")
ok("jenycampos@hotmail.com" not in ADMINS and not ap.es_cortesia("administracion@iurexia.com", False),
   "la cortesía no toca la lista de administración")


async def _acceso():
    sb = SbFalso(PERFILES)
    a = {u["id"]: await ap.acceso_del_usuario(sb, u["id"], ADMINS) for u in PERFILES}
    ok(not a["u-pro"].platinum and a["u-pro"].leido, "perfil Pro → no, leído")
    ok(a["u-plat"].platinum and not a["u-plat"].es_admin, "Platinum → sí, sin ser administración")
    ok(a["u-ultra"].platinum, "Ultra → sí")
    ok(a["u-admin"].platinum and a["u-admin"].es_admin, "administración → sí, como administración")
    ok(a["u-tester"].platinum and a["u-tester"].cortesia and not a["u-tester"].es_admin,
       "tester verificado → sí, por cortesía, NO administración")
    ok(not a["u-tester-sin"].platinum, "«inventado@iurexia.com» sin verificar → no")
    ok(a["u-jenica"].platinum and a["u-jenica"].cortesia and not a["u-jenica"].es_admin,
       "Jenica → sí, por cortesía, NO administración")
    ok(not a["u-imitador"].platinum, "@iurexia.com.mx → no")
    sin = await ap.acceso_del_usuario(sb, "u-no-existe", ADMINS)
    ok(not sin.platinum and sin.leido, "perfil inexistente → no, leído (no es un tropiezo)")
    nadie = await ap.acceso_del_usuario(sb, None, ADMINS)
    ok(not nadie.platinum and nadie.leido and not sb.caida, "sin usuario → no, sin leer la base")
    j = await ap.acceso_por_correo(sb, " JenyCampos@Hotmail.com ", ADMINS)
    ok(j.platinum and j.cortesia, "por correo (Jurimetría sin token): Jenica sí")

    print("\n── exigir: 403, 503 ──")
    try:
        await ap.exigir(sb, "u-pro", ADMINS, "La búsqueda de Precedentes")
        ok(False, "Pro → 403")
    except HTTPException as e:
        ok(e.status_code == 403 and e.detail["codigo"] == "requiere_platinum"
           and e.detail["plan"] == "pro_monthly"
           and e.detail["mensaje"] == "La búsqueda de Precedentes es exclusiva del plan Platinum.",
           f"Pro → 403 requiere_platinum con su plan y el aviso ({e.status_code})")
    try:
        await ap.exigir(sb, None, ADMINS)
        ok(False, "sin usuario → 403")
    except HTTPException as e:
        ok(e.status_code == 403, f"sin usuario → 403, no 503: un 503 lo reintentaría la web ({e.status_code})")
    sb.caida = True
    try:
        await ap.exigir(sb, "u-plat", ADMINS)
        ok(False, "base caída → 503")
    except HTTPException as e:
        ok(e.status_code == 503, f"base caída, aun siendo Platinum → 503: se cierra ({e.status_code})")
    sb.caida = False
    ok((await ap.exigir(sb, "u-jenica", ADMINS)).cortesia, "Jenica → pasa")

    print("\n── cobro_consultas (la puerta que comparten Cronología y Bóveda) ──")
    r = await cc.exigir_platinum(sb, "u-tester", ADMINS, "La Cronología")
    ok(r == {"plan": "gratuito", "es_admin": False, "cortesia": True}, f"tester → pasa sin ser administración ({r})")
    try:
        await cc.exigir_platinum(sb, "u-pro", ADMINS, "La Cronología")
        ok(False, "Pro → 403 en la Cronología")
    except HTTPException as e:
        ok(e.status_code == 403 and e.detail["codigo"] == "requiere_platinum", "Pro → 403 en la Cronología")
    ok(await cc.plan_del_usuario(sb, "u-jenica", ADMINS) == ("pro_monthly", False, True),
       "plan_del_usuario conserva su forma: (plan, es_admin, leído)")
    ok(await cc.plan_del_usuario(sb, "", ADMINS) == (None, False, False), "plan_del_usuario sin usuario: como antes")

    print("\n── Bóveda: pasa como Platinum, paga como Platinum ──")
    import boveda
    boveda.configurar(sb, ADMINS)
    uj = await boveda._usuario("Bearer tok-u-jenica")
    ok(uj.es_platinum and uj.cortesia and not uj.es_admin, "Jenica: es_platinum sí, es_admin no")
    ok(await boveda._disponibles(uj) is not None, "a Jenica la Bóveda le cuenta el saldo (no es «sin tope»)")
    up = await boveda._usuario("Bearer tok-u-pro")
    try:
        await boveda._platinum("Bearer tok-u-pro")
        ok(False, "Pro → 403 en la Bóveda")
    except HTTPException as e:
        ok(not up.es_platinum and e.status_code == 403, "Pro → 403 en la Bóveda")


asyncio.run(_acceso())


# ── Los endpoints, con el app de main.py ────────────────────────────────
# Llaves falsas A LA FUERZA: load_dotenv no pisa lo que ya está, así que
# aunque hubiera un .env junto al repo nada saldría hacia los servicios.
for _k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY",
           "MISTRAL_API_KEY", "ELEVENLABS_API_KEY", "AZURE_DOCINT_KEY", "AZURE_SPEECH_KEY"):
    os.environ[_k] = "falsa-para-pruebas"
os.environ["QDRANT_URL"] = "http://127.0.0.1:9"
os.environ["QDRANT_API_KEY"] = ""
os.environ["SUPABASE_URL"] = ""
os.environ["SUPABASE_SERVICE_KEY"] = ""

print("\n── Cargando main.py (sin arranque) ──")
import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

print("\n── Qué marcador es de Platinum ──")
h = main.herramienta_platinum_del_chat
ok(h("[MODO_PRECEDENTES] [CORTE:TCC] ¿qué dicen los colegiados?") == "La búsqueda de Precedentes", "[MODO_PRECEDENTES]")
ok(h("[AUDITAR_SENTENCIA]\nArchivo: x.pdf") == "La auditoría de sentencias", "[AUDITAR_SENTENCIA]")
ok(h("…<!-- SENTENCIA_INICIO -->texto<!-- SENTENCIA_FIN -->") == "La auditoría de sentencias", "SENTENCIA_INICIO")
ok(h("¿Qué es el amparo directo?") is None, "una consulta normal no")
ok(h("[MODO_REDACCION_PRO] redacta una demanda") is None, "la redacción no (tiene su propio escalón)")
ok(h("") is None and h(None) is None, "vacío no")

cliente = TestClient(main.app, raise_server_exceptions=False)
import rate_limiter  # noqa: E402


def _sin_limite():
    """Todas las peticiones salen de la misma IP: el limitador (10/min en
    /chat) respondería 429 a la undécima. No es lo que se prueba."""
    rate_limiter._rate_limiter._requests.clear()


def chat(sb, user_id, texto, historia=()):
    _sin_limite()
    sb.rpcs.clear()
    main.supabase_admin = sb
    mensajes = [{"role": r, "content": c} for r, c in historia] + [{"role": "user", "content": texto}]
    cuerpo = {"messages": mensajes}
    cabeceras = {}
    if user_id:
        cuerpo["user_id"] = user_id
        cabeceras["Authorization"] = f"Bearer {tok(user_id)}"
    with cliente.stream("POST", "/chat", json=cuerpo, headers=cabeceras) as r:
        if r.status_code != 200:
            r.read()
            return r.status_code, r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
        # Pasó la puerta: basta el primer trozo; lo que sigue es la búsqueda.
        for _ in r.iter_raw():
            break
        return 200, None


PREC = "[MODO_PRECEDENTES] [CORTE:TCC] ¿Procede la suspensión contra el embargo de cuentas?"
SENT = "[AUDITAR_SENTENCIA]\nArchivo: proyecto.pdf\n<!-- SENTENCIA_INICIO -->CONSIDERANDO…<!-- SENTENCIA_FIN -->"


def chat_rechaza(sb, uid, texto, que, codigo=403):
    s, cuerpo = chat(sb, uid, texto)
    detalle = cuerpo.get("detail") if isinstance(cuerpo, dict) else None
    bien = s == codigo and (codigo != 403 or (isinstance(detalle, dict) and detalle.get("codigo") == "requiere_platinum"))
    ok(bien, f"{que} → {codigo} ({s}: {str(detalle)[:90]})")
    ok(not sb.rpcs, f"{que} → sin consume_query ni is_user_blocked (rpcs: {[n for n, _ in sb.rpcs]})")


def chat_pasa(sb, uid, texto, que, historia=()):
    s, cuerpo = chat(sb, uid, texto, historia)
    ok(s == 200 and uid in sb.cobros(), f"{que} → pasa y se cobra su consulta ({s}, cobros: {sb.cobros()})")


print("\n── /chat: Precedentes ──")
sb = SbFalso(PERFILES)
chat_rechaza(sb, "u-pro", PREC, "Pro")
chat_rechaza(sb, None, PREC, "sin user_id")
chat_rechaza(sb, "u-tester-sin", PREC, "@iurexia.com sin verificar")
chat_pasa(sb, "u-plat", PREC, "Platinum")
chat_pasa(sb, "u-ultra", PREC, "Ultra")
chat_pasa(sb, "u-admin", PREC, "administración")
chat_pasa(sb, "u-tester", PREC, "tester")
chat_pasa(sb, "u-jenica", PREC, "Jenica")
sb.caida = True
chat_rechaza(sb, "u-plat", PREC, "Platinum con la base caída", codigo=503)
sb.caida = False

print("\n── /chat: Sentencia ──")
chat_rechaza(sb, "u-pro", SENT, "Pro")
chat_pasa(sb, "u-anual", SENT, "Platinum anual")

print("\n── /chat: lo demás no cambia ──")
chat_pasa(sb, "u-pro", "¿Qué es el amparo directo?", "Pro con una consulta normal")
chat_pasa(sb, "u-pro", "¿Y en materia laboral?", "Pro con el marcador en un mensaje ANTERIOR: cuenta el último",
          historia=[("user", PREC), ("assistant", "…")])

print("\n── /chat: el escalón Platinum de la redacción ──")
import contextlib  # noqa: E402
import io  # noqa: E402

RED = "[MODO_REDACCION_PLATINUM] Redacta una demanda de rescisión de un contrato de arrendamiento."


def registro(uid):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        chat(sb, uid, RED)
    return buf.getvalue()


ok("REDACCIÓN PLATINUM sin plan Platinum" in registro("u-pro"), "Pro → cae a Redacción Pro")
for uid, que in (("u-plat", "Platinum"), ("u-jenica", "Jenica"), ("u-tester", "tester con plan gratuito")):
    r = registro(uid)
    ok("REDACCIÓN PLATINUM sin plan" not in r and "PLATINUM DEGRADADO" not in r, f"{que} → se queda en Platinum")


print("\n── /toulmin/stream ──")
TOUL = {"hechos": "El arrendatario dejó de pagar seis meses de renta pese a los requerimientos.",
        "pretension": "La rescisión del contrato y el pago de rentas.", "clase": "demanda"}


def toulmin(sb, uid):
    _sin_limite()
    sb.rpcs.clear()
    main.supabase_admin = sb
    viejos = (main.qdrant_client, main.chat_client)
    main.qdrant_client, main.chat_client = object(), object()   # que no se detenga en el 503 de servicio
    try:
        with cliente.stream("POST", "/toulmin/stream", json=TOUL,
                            headers={"Authorization": f"Bearer tok-{uid}"}) as r:
            if r.status_code != 200:
                r.read()
                return r.status_code, r.json()
            return 200, None
    finally:
        main.qdrant_client, main.chat_client = viejos


s, cuerpo = toulmin(sb, "u-pro")
ok(s == 403 and cuerpo["detail"]["codigo"] == "requiere_platinum", f"Pro → 403 ({s})")
ok(not sb.cobros(), "Pro → sin consume_query")
s, _ = toulmin(sb, "u-plat")
ok("u-plat" in sb.cobros(), f"Platinum → pasa la puerta y se cobra ({s})")
s, _ = toulmin(sb, "u-jenica")
ok("u-jenica" in sb.cobros(), f"Jenica → pasa ({s})")


print("\n── /expediente/cronologia/cotizar (usa cobro_consultas.exigir_platinum) ──")


def cotizar(sb, uid):
    _sin_limite()
    main.supabase_admin = sb
    r = cliente.post("/expediente/cronologia/cotizar", json={"paginas": 20},
                     headers={"Authorization": f"Bearer tok-{uid}"})
    return r.status_code, r.json()


s, c = cotizar(sb, "u-pro")
ok(s == 403 and c["detail"]["codigo"] == "requiere_platinum", f"Pro → 403 ({s})")
s, c = cotizar(sb, "u-jenica")
ok(s == 200 and c.get("consultas") == 2, f"Jenica → cotiza 20 páginas en 2 consultas ({s}: {c.get('consultas')})")
s, c = cotizar(sb, "u-tester-sin")
ok(s == 403, f"@iurexia.com sin verificar → 403 ({s})")


print("\n── /api/jurimetria ──")


def jurimetria(sb, correo, token=None):
    _sin_limite()
    main.supabase_admin = sb
    viejo = main.qdrant_client
    main.qdrant_client = None
    try:
        r = cliente.post("/api/jurimetria", data={"user_email": correo, "descripcion": "despido injustificado"},
                         headers={"Authorization": f"Bearer {token}"} if token else {})
        return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text)
    finally:
        main.qdrant_client = viejo


s, c = jurimetria(sb, "abogada@ejemplo.mx", token=tok("u-pro"))
ok(s == 403 and c["detail"] == "La Jurimetría es exclusiva del plan Platinum.", f"Pro con su token → 403 en texto ({s}: {c})")
# Desde el 29-oct-2026 el correo solo ya no basta (puerta_sesion.py): sin
# token, 401; con el token de otra cuenta, 403 antes de mirar ningún plan.
s, c = jurimetria(sb, "socio@ejemplo.mx")
ok(s == 401, f"el correo de un Platinum SIN token → 401, ya no el repliegue al correo ({s})")
s, c = jurimetria(sb, "socio@ejemplo.mx", token=tok("u-pro"))
ok(s == 403 and c["detail"] != "La Jurimetría es exclusiva del plan Platinum.",
   f"correo de un Platinum con el token de un Pro → 403 de la puerta ({s})")
s, c = jurimetria(sb, "socio@ejemplo.mx", token=tok("u-plat"))
ok(s not in (401, 403), f"un Platinum con su token → pasa la puerta ({s})")
s, c = jurimetria(sb, "socio@ejemplo.mx", token="basura")
ok(s == 401, f"token inválido → 401, no el repliegue al correo ({s})")
s, c = jurimetria(sb, "jenycampos@hotmail.com", token=tok("u-jenica"))
ok(s not in (401, 403), f"Jenica con su token → pasa la puerta ({s})")


# ── Con la base real: sólo lee user_profiles ────────────────────────────
if REAL:
    from dotenv import dotenv_values

    _env = {}
    for _ruta in (".env", "../jurexia-api-git/.env"):
        if os.path.exists(_ruta):
            _env = dotenv_values(_ruta)
            break
    if not (_env.get("SUPABASE_URL") and _env.get("SUPABASE_SERVICE_KEY")):
        sys.exit("--real necesita SUPABASE_URL y SUPABASE_SERVICE_KEY en .env o ../jurexia-api-git/.env")

    class SoloLectura(SbFalso):
        """user_profiles se lee de verdad y SÓLO se lee (select); las demás
        tablas salen vacías y los rpc se interceptan."""

        def __init__(self, real):
            super().__init__([])
            self.real = real

        def table(self, nombre):
            if nombre != "user_profiles":
                return _Consulta(self, "_vacia")
            real = self.real

            class _SoloSelect:
                def select(self, *a, **k):
                    return real.table("user_profiles").select(*a, **k)
            return _SoloSelect()

    real = SoloLectura(main._SupabasePorHilo(_env["SUPABASE_URL"], _env["SUPABASE_SERVICE_KEY"]))

    def _id(correo):
        r = real.table("user_profiles").select("id").eq("email", correo).limit(1).execute()
        return r.data[0]["id"] if r.data else None

    print("\n── Base real: quién pasa (sólo lectura) ──")

    async def _reales():
        casos = [("administracion@iurexia.com", True, True, True),   # administración y, además, del dominio
                 ("jenycampos@hotmail.com", True, False, True),
                 ("yair@iurexia.com", True, False, True)]
        for correo, pasa, admin, cortesia in casos:
            a = await ap.acceso_del_usuario(real, _id(correo), main.ADMIN_EMAILS)
            ok(a.leido and a.platinum == pasa and a.es_admin == admin and a.cortesia == cortesia,
               f"{correo}: plan {a.plan}, platinum={a.platinum}, admin={a.es_admin}, cortesía={a.cortesia}")
    asyncio.run(_reales())

    _pro = real.table("user_profiles").select("id, email").eq("subscription_type", "pro_monthly") \
        .not_.like("email", "%@iurexia.com").neq("email", "jenycampos@hotmail.com").limit(1).execute().data
    print("\n── Base real: /chat con Precedentes ──")
    if _pro:
        chat_rechaza(real, _pro[0]["id"], PREC, f"un Pro real ({_pro[0]['id'][:8]}…)")
    else:
        ok(False, "no encontré un perfil pro_monthly fuera de la cortesía")
    chat_rechaza(real, str(uuid.uuid4()), PREC, "un usuario que no existe")
    chat_pasa(real, _id("administracion@iurexia.com"), PREC, "administracion@ (platinum_annual)")
    chat_pasa(real, _id("jenycampos@hotmail.com"), PREC, "Jenica (pro_monthly, cortesía)")


print(f"\n{'TODO PASA' if not FALLOS else f'{len(FALLOS)} FALLAN'}")
sys.exit(1 if FALLOS else 0)
