"""EL PLAN ESTUDIO Y SU FRENO DE 560 — 2-nov-2026.

    .venv/bin/python test_freno_estudio.py

Sin red, sin gastar API y SIN EL ARRANQUE de main.py: el lifespan borraría
las cachés de Gemini de producción. `load_dotenv` se anula y las llaves se
fuerzan falsas antes de `import main`, y se comprueba que `supabase_admin`
quedó vacío: nada de lo que corre aquí puede cobrar ni escribir de verdad.

Se comprueba:
  1. el freno puro (freno_estudio.py): 559 no frena, 560 sí; el cobro de
     `consume_query` (del plan o de recarga); el contador ilegible no frena
     y queda en el registro; los avisos;
  2. las listas: Estudio está en cada colección de planes que nombra a
     Platinum (en todos los módulos, no sólo en main.py), con sus 600 hojas,
     100 flujos, rango 8, y el limitador no cambia para nadie más;
  3. el cobro de la Bóveda y la Cronología: con Estudio, sólo lo que cabe en
     las 560, sin llamar a `consumir_consultas` si no cabe;
  4. el OCR: con el freno, la caché sí y Azure no; y /extract-text, que
     es por donde Sentencia manda sus PDF, frena con la sesión de Estudio;
  5. la tienda (A8): no pisa un plan de Stripe mayor;
  6. en /chat y en /analyze-document el freno va antes de elegir motor.
"""
import asyncio
import ast
import contextlib
import io
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


def registro(fn, *a, **k):
    """Lo que `fn` imprime, y lo que devuelve (o la excepción que lanza)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            res = fn(*a, **k)
        except Exception as e:      # noqa: BLE001 — se devuelve para mirarla
            res = e
    return buf.getvalue(), res


import freno_estudio as fe  # noqa: E402
from fastapi import HTTPException  # noqa: E402

E = "platinum_estudio"

print("\n── 1. El freno ──")
ok(fe.es_estudio(E) and fe.es_estudio(" Platinum_Estudio ") and not fe.es_estudio("platinum_monthly")
   and not fe.es_estudio(None), "es_estudio: sólo platinum_estudio (sin importar mayúsculas)")
ok(not fe.frenado(E, 559), "559 usadas: la consulta 560 todavía es de élite")
ok(fe.frenado(E, 560) and fe.frenado(E, 3000), "560 usadas: desde la 561, con el freno")
ok(not fe.frenado("platinum_monthly", 9999) and not fe.frenado("pro_monthly", 9999)
   and not fe.frenado(None, 9999), "ningún otro plan frena, use lo que use")
txt, res = registro(fe.frenado, E, None, "prueba")
ok(res is False and "FRENO DE ESTUDIO SIN CONTADOR (prueba)" in txt,
   "contador ilegible (None) → como Platinum, y queda la línea en el registro")
txt, res = registro(fe.frenado, E, "abc")
ok(res is False and "SIN CONTADOR" in txt, "contador ilegible (texto) → como Platinum, con registro")
ok(fe.frenado(E, "600"), "el contador como texto numérico se lee")
txt, _ = registro(fe.frenado, "platinum_monthly", None)
ok(txt == "", "a otro plan sin contador no se le grita nada")

# Lo que contesta consume_query (Supabase, 2-nov-2026): del plan, `used` ya
# trae sumada la consulta; de recarga, no.
c = lambda used, sub=E, recarga=False, allowed=True: {  # noqa: E731
    "allowed": allowed, "de_recarga": recarga, "used": used, "limit": 3100, "subscription_type": sub}
ok(fe.frenado_tras_cobro(c(560)) is False, "cobro con used=560 → ésta fue la 560: élite")
ok(fe.frenado_tras_cobro(c(561)) is True, "cobro con used=561 → ésta es la 561: con freno")
ok(fe.frenado_tras_cobro(c(3100, recarga=True)) is True, "cobro de recarga (tope lleno) → con freno")
ok(fe.frenado_tras_cobro(c(900, sub="platinum_monthly")) is False, "Platinum nunca")
ok(fe.frenado_tras_cobro({}) is None and fe.frenado_tras_cobro(None) is None
   and fe.frenado_tras_cobro(c(900, allowed=False)) is None,
   "sin cobro que leer → None: decide la lectura del perfil, o Platinum")
ok(fe.usadas_antes_del_cobro(c(1)) == 0 and fe.usadas_antes_del_cobro(c(3100, recarga=True)) == 3100,
   "usadas antes del cobro: del plan resta 1; de recarga, no")

ok(fe.elite_restantes(E, 500) == 60 and fe.elite_restantes(E, 700) == 0
   and fe.elite_restantes(E, None) is None and fe.elite_restantes("platinum_monthly", 10) is None,
   "elite_restantes: lo que queda de las 560; None sin freno o sin contador")

APROBADO = ("Sus 560 consultas con la IA Jurídica de Élite de este mes ya se usaron; Bóveda, "
            "Cronología y los documentos escaneados vuelven con su siguiente periodo. Las "
            "consultas y la redacción siguen sin límite.")
ok(fe.MENSAJE_AGOTADAS == APROBADO, "el aviso es, letra por letra, el aprobado el 2-nov")
try:
    fe.exigir_elite(E, 560)
    ok(False, "agotadas → 402")
except HTTPException as e:
    ok(e.status_code == 402 and e.detail["codigo"] == "freno_estudio" and e.detail["mensaje"] == APROBADO
       and e.detail["disponibles"] == 0, "agotadas → 402 freno_estudio con el aviso aprobado")
try:
    fe.exigir_elite(E, 550, 30)
    ok(False, "quedan 10, pide 30 → 402")
except HTTPException as e:
    m = e.detail["mensaje"]
    ok(e.status_code == 402 and "necesita 30 consultas" in m and "le quedan 10" in m
       and m.endswith("siguen sin límite."), f"quedan 10 y pide 30 → 402 con su número ({m[:70]}…)")
ok(fe.exigir_elite(E, 530, 30) is None, "quedan 30 y pide 30 → pasa")
ok(fe.exigir_elite("platinum_monthly", 9999, 500) is None, "Platinum: aquí no se mira (lo cuida el cobro)")
txt, res = registro(fe.exigir_elite, E, None, 30, "prueba")
ok(res is None and "SIN CONTADOR" in txt, "sin contador → pasa como Platinum, con registro")
ok("le queda 1." in fe.mensaje_insuficientes(1, 5), "singular: «le queda 1»")
s = fe.SinElite()
ok(isinstance(s, HTTPException) and s.status_code == 402 and s.detail == APROBADO,
   "SinElite: 402 con el aviso en texto (lo que pintan /analyze-document y Jurimetría)")
ok(fe.resumen(E, 600) == {"consultas_elite": 560, "restantes": 0, "mensaje": APROBADO}
   and fe.resumen(E, 100)["mensaje"] is None and fe.resumen("pro_monthly", 1) is None,
   "resumen para la pantalla: lo que queda y, agotadas, el aviso")


print("\n── 2. Las listas ──")
import esfuerzo_redaccion as er  # noqa: E402
import acceso_platinum as ap  # noqa: E402
import lectura_por_plan as lp  # noqa: E402
import flujo_agente as fa  # noqa: E402
import rate_limiter as rl  # noqa: E402
import rango_planes as rp  # noqa: E402

ok(E in er.PLANES_PRO and E in er.PLANES_PLATINUM, "esfuerzo_redaccion: Estudio en PLANES_PRO y PLANES_PLATINUM")
ok(er.esfuerzo_permitido("platinum", E) == "platinum", "el escalón Platinum de la redacción le corresponde")
ok(ap.tiene_platinum(E, "x@y.mx") and ap.Acceso(E, False, False, True, usadas=560).frenado
   and not ap.Acceso(E, False, False, True, usadas=559).frenado
   and not ap.Acceso(E, False, False, True).frenado,
   "acceso_platinum: pasa la puerta; `.frenado` desde 560; sin contador, no")
ok(lp.tope_del_plan(E) == 600 and lp.nombre_del_plan(E) == "Estudio"
   and lp.nombre_del_plan("platinum_monthly") == "Platinum",
   "lectura_por_plan: 600 hojas, y se llama «Estudio» (Platinum sigue siendo Platinum)")
ok(fa.limite_flujos(E) == 100 == fe.FLUJOS_AL_MES, "flujo_agente: 100 flujos al mes, el número de la tarjeta")
ok([fa.limite_flujos(p) for p in ("platinum_monthly", "platinum_annual", "ultra_secretarios",
                                  "pro_monthly", "pro_annual", "basico_monthly", "gratuito", None)]
   == [60, 60, 60, 30, 30, 0, 0, 0], "flujo_agente: los demás planes, como antes (60/30/0)")
ANTES = {"gratuito": 10, "pro_monthly": 30, "pro_annual": 30, "platinum_monthly": 60,
         "platinum_annual": 60, "ultra_secretarios": 40}
ok({k: v for k, v in rl.RATE_LIMITS.items() if k != E} == ANTES and rl.DEFAULT_RATE_LIMIT == 10,
   "rate_limiter: nada cambia para los demás planes")
ok(rp.rango(E) == 8 > rp.rango("ultra_secretarios") == 7 and rp.rango("cualquiera") == 0,
   "rango_planes: Estudio 8, sobre Ultra 7; un plan desconocido cuenta como gratuito")

# CADA COLECCIÓN QUE NOMBRA A PLATINUM NOMBRA A ESTUDIO. Se lee el código, no
# se importa: un `{…}`, `(…)` o `[…]` literal con «platinum_monthly» —o un
# dict con esa llave— que no traiga «platinum_estudio» es una lista olvidada.
# Fuera quedan, a propósito, los mapas de producto de las tiendas (allí el
# plan es el VALOR: Estudio no se vende en App Store ni en Play, A10).
RAIZ = pathlib.Path(__file__).parent
olvidos = []
for f in sorted(RAIZ.glob("*.py")):
    if f.name.startswith("test_") or f.name == pathlib.Path(__file__).name:
        continue
    try:
        arbol = ast.parse(f.read_text(encoding="utf8"))
    except SyntaxError:
        continue
    for n in ast.walk(arbol):
        if isinstance(n, ast.Dict):
            elems = n.keys
        elif isinstance(n, (ast.Set, ast.Tuple, ast.List)):
            elems = n.elts
        else:
            continue
        cadenas = {e.value for e in elems if isinstance(e, ast.Constant) and isinstance(e.value, str)}
        if "platinum_monthly" in cadenas and E not in cadenas:
            olvidos.append(f"{f.name}:{n.lineno}")
ok(not olvidos, f"ninguna colección de planes olvida a Estudio ({', '.join(olvidos) or 'ninguna'})")

fuente_main = (RAIZ / "main.py").read_text(encoding="utf8")
arbol_main = ast.parse(fuente_main)


def _literal(nombre):
    for n in arbol_main.body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == nombre for t in n.targets):
            return n.value
    return None


salvame = ast.literal_eval(_literal("SALVAME_LIMITS"))
ok(salvame.get(E) == salvame["platinum_monthly"], "main.SALVAME_LIMITS: Estudio como Platinum (10 al día, 30 al mes)")
ok(E in ast.literal_eval(_literal("PLANES_CON_JURIMETRIA")), "main.PLANES_CON_JURIMETRIA (y por unión Jurisconsulto)")
ok("'pro_monthly', 'pro_annual', 'platinum_monthly'" not in fuente_main,
   "main.py: las tres listas de «Pro o más» son PLANES_PRO, no copias")


print("\n── 3. El cobro de la Bóveda y la Cronología ──")
import cobro_consultas as cc  # noqa: E402
import boveda  # noqa: E402


class _Res:
    def __init__(self, data):
        self.data = data


class SbCobro:
    """`consumir_consultas` y la lectura de user_profiles, contadas."""

    def __init__(self, usadas=None, caida=False, plan=E):
        self.usadas, self.caida, self.plan = usadas, caida, plan
        self.rpcs, self.lecturas = [], 0

    def table(self, _nombre):
        sb = self

        class Q:
            def __getattr__(self, _n):
                return lambda *a, **k: self

            def execute(self):
                sb.lecturas += 1
                if sb.caida:
                    raise ConnectionError("base caída")
                return _Res([{"queries_used": sb.usadas, "queries_limit": 3100,
                              "consultas_recargadas": 0, "subscription_type": sb.plan}])
        return Q()

    def rpc(self, nombre, params):
        self.rpcs.append((nombre, dict(params)))

        class X:
            def execute(_s):
                return _Res({"allowed": True, "cobro_id": "c-1", "unidades": params.get("p_unidades"),
                             "de_plan": params.get("p_unidades"), "de_recarga": 0, "restantes": 2000})
        return X()


async def _cobros():
    sb = SbCobro()
    try:
        await cc.cobrar_consultas(sb, "u", 1, "boveda_ingesta", plan=E, usadas=560)
        ok(False, "agotadas → 402")
    except HTTPException as e:
        ok(e.status_code == 402 and e.detail["codigo"] == "freno_estudio" and not sb.rpcs,
           "560 usadas → 402 freno_estudio SIN llamar a consumir_consultas")
    try:
        await cc.cobrar_consultas(sb, "u", 30, "cronologia", plan=E, usadas=550)
        ok(False, "no cabe → 402")
    except HTTPException as e:
        ok(e.status_code == 402 and not sb.rpcs, "quedan 10 y pide 30 → 402, sin cobrar")
    d = await cc.cobrar_consultas(sb, "u", 30, "cronologia", plan=E, usadas=500)
    ok(len(sb.rpcs) == 1 and sb.lecturas == 0 and d["restantes"] == 30 and d["restantes_del_tope"] == 2000,
       f"cabe → se cobra sin otra lectura; «restantes» es lo que queda de las 560 ({d.get('restantes')})")
    sb = SbCobro(usadas=600)
    try:
        await cc.cobrar_consultas(sb, "u", 1, "boveda_tabla", plan=E)
        ok(False, "sin `usadas`, lee y frena")
    except HTTPException as e:
        ok(e.status_code == 402 and sb.lecturas == 1 and not sb.rpcs,
           "sin `usadas` de quien llama: lee el contador una vez y frena")
    sb = SbCobro(caida=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        await cc.cobrar_consultas(sb, "u", 5, "boveda_tabla", plan=E)
    ok(len(sb.rpcs) == 1 and "SIN CONTADOR" in buf.getvalue(),
       "contador ilegible → se cobra como a Platinum y queda «SIN CONTADOR» en el registro")
    sb = SbCobro(plan="platinum_monthly")
    d = await cc.cobrar_consultas(sb, "u", 5, "cronologia", plan="platinum_monthly", usadas=9999)
    ok(len(sb.rpcs) == 1 and sb.lecturas == 0 and d["restantes"] == 2000 and "restantes_del_tope" not in d,
       "Platinum: el cobro de siempre, sin lecturas ni cambios en la respuesta")

    s = await cc.saldo_consultas(SbCobro(usadas=120), "u")
    ok(s["disponibles"] == 440 and s["freno"]["restantes"] == 440 and not s["ilimitado"],
       "saldo de Estudio: contra las 560, no contra las 3,100")
    s = await cc.saldo_consultas(SbCobro(usadas=120, plan="platinum_monthly"), "u")
    ok(s["disponibles"] == 2980 and "freno" not in s, "saldo de Platinum: el de siempre")

    falso = SbCobro(usadas=10)
    viejo = boveda._cliente
    boveda._cliente = lambda: falso
    try:
        u = boveda.Usuario(id="u", plan=E, es_admin=False, plan_leido=True, usadas=500)
        ok(await boveda._disponibles(u) == 60 and falso.lecturas == 0,
           "Bóveda: lo que queda de las 560, de la lectura de la puerta (sin otra)")
        u = boveda.Usuario(id="u", plan=E, es_admin=False, plan_leido=True, usadas=None)
        ok(await boveda._disponibles(u) == 550 and falso.lecturas == 1,
           "Bóveda sin contador en la puerta: relee el saldo")
    finally:
        boveda._cliente = viejo


asyncio.run(_cobros())


print("\n── 4-6. Con main.py, sin arranque ──")
import dotenv  # noqa: E402

dotenv.load_dotenv = lambda *a, **k: False
for _k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY",
           "MISTRAL_API_KEY", "ELEVENLABS_API_KEY", "AZURE_DOCINT_KEY", "AZURE_SPEECH_KEY"):
    os.environ[_k] = "falsa-para-pruebas"
os.environ["AZURE_DOCINT_ENDPOINT"] = "http://127.0.0.1:9"
os.environ["QDRANT_URL"] = "http://127.0.0.1:9"
os.environ["QDRANT_API_KEY"] = ""
os.environ["SUPABASE_URL"] = ""
os.environ["SUPABASE_SERVICE_KEY"] = ""
import main  # noqa: E402

if main.supabase_admin is not None:
    sys.exit("supabase_admin NO quedó vacío: se aborta antes de tocar nada de verdad")


def _pdf_escaneado() -> bytes:
    """Un PDF de 2 hojas que son sólo imagen: lo que manda al OCR."""
    import fitz
    d = fitz.open()
    for _ in range(2):
        p = d.new_page()
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 60, 60), False)
        pix.clear_with(200)
        p.insert_image(fitz.Rect(50, 50, 300, 300), pixmap=pix)
    b = d.tobytes()
    d.close()
    return b


async def _ocr():
    pdf = _pdf_escaneado()
    azure = []

    async def _azure(contenido, paginas_esperadas=0, **k):
        azure.append(paginas_esperadas)
        return "TEXTO RECONOCIDO " * 40

    async def _sin_cache(_c, _h):
        return None

    async def _en_cache(_c, _h):
        return "TEXTO QUE YA SE HABÍA LEÍDO"

    viejos = (main._ocr_azure, main.cache_ocr.leer, main.cache_ocr.guardar_en_segundo_plano)
    main._ocr_azure = _azure
    main.cache_ocr.guardar_en_segundo_plano = lambda *a, **k: None
    try:
        main.cache_ocr.leer = _sin_cache
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                await main._extract_text_from_upload(main._SubidaDeBytes(pdf, "escaneo.pdf"), permitir_ocr=False)
            ok(False, "con el freno y sin caché → SinElite")
        except fe.SinElite as e:
            ok(e.status_code == 402 and e.detail == APROBADO and not azure,
               "con el freno y sin caché → 402 con el aviso, y Azure NO se llama")
        main.cache_ocr.leer = _en_cache
        with contextlib.redirect_stdout(io.StringIO()):
            t = await main._extract_text_from_upload(main._SubidaDeBytes(pdf, "escaneo.pdf"), permitir_ocr=False)
        ok(t == "TEXTO QUE YA SE HABÍA LEÍDO" and not azure, "con el freno, lo que ya está en caché sí se lee (gratis)")
        main.cache_ocr.leer = _sin_cache
        with contextlib.redirect_stdout(io.StringIO()):
            t = await main._extract_text_from_upload(main._SubidaDeBytes(pdf, "escaneo.pdf"))
        ok(t.startswith("TEXTO RECONOCIDO") and azure == [2], "sin el freno (Platinum, o Estudio antes de 560): Azure como siempre")
    finally:
        main._ocr_azure, main.cache_ocr.leer, main.cache_ocr.guardar_en_segundo_plano = viejos


asyncio.run(_ocr())


print("\n── 4b. /extract-text (el PDF de Sentencia) con el freno ──")


def _pdf_con_texto() -> bytes:
    """Una hoja con texto de verdad: no necesita OCR. Variado, porque
    `_texto_nativo_sirve` mide la riqueza (un sello repetido no pasa)."""
    import fitz
    d = fitz.open()
    p = d.new_page()
    parrafo = ("PRIMERO. Se confirma la sentencia recurrida. El tribunal colegiado estudia los "
               "agravios del quejoso sobre la valoración de pruebas, la competencia del juzgado, "
               "la oportunidad de la demanda, el interés jurídico, la suplencia de la queja y "
               "la motivación del acto reclamado. ")
    extra = " ".join(f"concepto{chr(97 + i % 26)}{chr(97 + i // 26)} razonado" for i in range(130))
    p.insert_textbox(fitz.Rect(30, 30, 580, 820), parrafo + extra, fontsize=7)
    b = d.tobytes()
    d.close()
    return b


async def _extract_text():
    from starlette.datastructures import UploadFile as _UF
    pdf, pdf_txt = _pdf_escaneado(), _pdf_con_texto()
    azure, sesiones, lecturas = [], [], []

    async def _azure(contenido, paginas_esperadas=0, **k):
        azure.append(paginas_esperadas)
        return "TEXTO RECONOCIDO " * 40

    async def _sin_cache(_c, _h):
        return None

    perfiles = {"tok-estudio-600": (E, 600), "tok-estudio-559": (E, 559),
                "tok-platinum": ("platinum_monthly", 9999)}

    async def _sesion(authorization, ruta=""):
        tok = main._puerta._extraer(authorization)
        sesiones.append(ruta)
        if tok == "tok-caido":
            raise HTTPException(503, "Supabase caído")
        if tok not in perfiles:
            raise HTTPException(401, "token que no vale")
        return main._puerta.Sesion(id=tok, correo=f"{tok}@x.mx")

    async def _acceso(uid):
        lecturas.append(uid)
        plan, usadas = perfiles[uid]
        return ap.Acceso(plan, False, False, True, usadas=usadas)

    async def pedir(contenido, nombre, token=None):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                r = await main.extract_text_from_document(
                    file=_UF(file=io.BytesIO(contenido), filename=nombre),
                    authorization=(f"Bearer {token}" if token else None))
            except Exception as e:      # noqa: BLE001 — se devuelve para mirarla
                r = e
        return r, buf.getvalue()

    viejos = (main._ocr_azure, main.cache_ocr.leer, main.cache_ocr.guardar_en_segundo_plano,
              main._puerta.sesion_del_token, main._acceso_platinum)
    main._ocr_azure, main.cache_ocr.leer = _azure, _sin_cache
    main.cache_ocr.guardar_en_segundo_plano = lambda *a, **k: None
    main._puerta.sesion_del_token, main._acceso_platinum = _sesion, _acceso
    try:
        r, _ = await pedir(pdf, "sentencia.pdf", "tok-estudio-600")
        ok(isinstance(r, fe.SinElite) and r.status_code == 402 and r.detail == APROBADO and not azure,
           "Estudio con 560 usadas manda un escaneo → 402 con el aviso aprobado, y Azure NO se llama")
        r, _ = await pedir(pdf_txt, "sentencia.pdf", "tok-estudio-600")
        ok(isinstance(r, dict) and "Se confirma la sentencia" in r.get("text", "") and not azure,
           "Estudio frenado con un PDF que trae su texto → se lee (no cuesta OCR)")
        r, _ = await pedir(pdf, "sentencia.pdf", "tok-estudio-559")
        ok(isinstance(r, dict) and r["text"].startswith("TEXTO RECONOCIDO") and azure == [2],
           "Estudio con 559 usadas → todavía con OCR, como Platinum")
        azure.clear()
        r, _ = await pedir(pdf, "sentencia.pdf", "tok-platinum")
        ok(isinstance(r, dict) and azure == [2], "Platinum (9,999 usadas) → OCR como siempre")
        azure.clear(), sesiones.clear(), lecturas.clear()
        r, _ = await pedir(pdf, "sentencia.pdf")
        ok(isinstance(r, dict) and azure == [2] and not sesiones and not lecturas,
           "sin token → como antes de Estudio: OCR y ni una lectura más")
        azure.clear()
        r, _ = await pedir(pdf, "sentencia.pdf", "tok-inventado")
        ok(isinstance(r, HTTPException) and r.status_code == 401 and not azure,
           "un token que no vale → 401, como en el resto de la API (la web renueva y repite)")
        r, txt = await pedir(pdf, "sentencia.pdf", "tok-caido")
        ok(isinstance(r, dict) and azure == [2] and "FRENO DE ESTUDIO SIN SESIÓN (/extract-text)" in txt,
           "Supabase no contesta → como Platinum, y queda la línea en el registro")
        sesiones.clear(), lecturas.clear()
        import lectura_docx

        async def _docx(_c, _sep):
            return "Texto de un Word que se lee en el servidor. " * 5
        viejo_docx = lectura_docx.leer_docx
        lectura_docx.leer_docx = _docx
        try:
            r, _ = await pedir(b"PK-falso", "demanda.docx", "tok-estudio-600")
        finally:
            lectura_docx.leer_docx = viejo_docx
        ok(isinstance(r, dict) and not sesiones and not lecturas,
           "un .docx no pasa por el OCR: ni se comprueba el token ni se lee el perfil")
    finally:
        (main._ocr_azure, main.cache_ocr.leer, main.cache_ocr.guardar_en_segundo_plano,
         main._puerta.sesion_del_token, main._acceso_platinum) = viejos


asyncio.run(_extract_text())


class SbTienda:
    def __init__(self, plan, stripe_id):
        self.fila = {"subscription_type": plan, "stripe_subscription_id": stripe_id}
        self.escrito = None

    def table(self, _n):
        sb = self

        class Q:
            def __init__(self):
                self.upd = None

            def select(self, *a, **k):
                return self

            def update(self, datos):
                self.upd = datos
                return self

            def eq(self, *a, **k):
                return self

            def limit(self, *a, **k):
                return self

            def execute(self):
                if self.upd is not None:
                    sb.escrito = self.upd["subscription_type"]
                    return _Res([])
                return _Res([dict(sb.fila)])
        return Q()


def tienda(actual, stripe_id, nuevo):
    sb = SbTienda(actual, stripe_id)
    viejo = main.supabase_admin
    main.supabase_admin = sb
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            queda = main._escribir_plan("u", nuevo, {"prueba": True})
    finally:
        main.supabase_admin = viejo
    return queda, sb.escrito


print("\n── 5. La tienda no pisa un plan de Stripe mayor (A8) ──")
ok(tienda(E, "sub_1", "pro_monthly") == (E, None), "Estudio en Stripe + «Pro» en la app → se queda Estudio, nada se escribe")
ok(tienda(E, "sub_1", "gratuito") == (E, None), "aviso EXPIRED de la tienda con Stripe vivo → no escribe gratuito")
ok(tienda("pro_monthly", "sub_1", "platinum_monthly") == ("platinum_monthly", "platinum_monthly"),
   "Stripe Pro + Platinum en la app (mayor) → se escribe")
ok(tienda("platinum_monthly", None, "gratuito") == ("gratuito", "gratuito"),
   "sin Stripe la tienda manda como siempre (dio el plan y lo quita)")
ok(tienda("gratuito", "", "pro_monthly") == ("pro_monthly", "pro_monthly"), "id de Stripe vacío = sin Stripe")
ok(rp.tienda_puede_escribir("pro_monthly", "sub_1", "pro_monthly") is False,
   "mismo plan con Stripe: no hay nada que escribir")


print("\n── 6. El freno va antes de elegir motor ──")


def _cuerpo(nombre):
    for n in ast.walk(arbol_main):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == nombre:
            return ast.get_source_segment(fuente_main, n)
    return ""


chat_src = _cuerpo("chat_endpoint")
i_freno = chat_src.find("_estudio_frenado = _freno.frenado_tras_cobro(_cobro_chat)")
i_apaga = chat_src.find("is_chat_drafting_platinum = False", i_freno)
usos = [chat_src.find(x, i_freno) for x in ("system_prompt + ACABADO_PLATINUM",
                                              "active_model = REDACTOR_PLATINUM_MODEL",
                                              'yield "<!--MODE:PLATINUM-->"')]
ok(i_freno > 0 and 0 < i_apaga < min(usos) and all(u > 0 for u in usos),
   "/chat: el freno apaga el escalón Platinum ANTES del prompt, el motor y la insignia")
ok("nonlocal _resolved_genio_ids, is_chat_drafting_platinum" in chat_src,
   "/chat: el flujo puede apagar la bandera de la ruta (nonlocal)")
i_sent = chat_src.find("elif is_sentencia:")
ok(0 < chat_src.find("if _estudio_frenado:", i_sent) < chat_src.find("_ap_m.modelo()", i_sent),
   "/chat: la auditoría de sentencias con el freno usa el motor de Pro, antes que el de élite")
ok(chat_src.find("_cobro_chat.update(q_data)") > 0, "/chat: el cobro de consume_query se guarda para el freno")
ok("_acc_freno_chat.plan, _acc_freno_chat.usadas" in chat_src,
   "/chat: sin cobro, el respaldo es la lectura del perfil de la puerta")
doc_src = _cuerpo("analyze_document")
ok("select('subscription_type, email, estado, queries_used')" in doc_src
   and doc_src.find("_freno.frenado(sub_type") < doc_src.find("rpc('consume_query'"),
   "/analyze-document: el contador viene en la MISMA lectura del perfil, antes del cobro")
ok("_doc_elite = is_platinum_or_admin and not _freno_doc" in doc_src
   and "DOCUMENT_MODEL_PLATINUM if _doc_elite else DOCUMENT_MODEL" in doc_src,
   "/analyze-document: con el freno, el motor de documentos de Pro")
ok(doc_src.count("permitir_ocr=not _freno_doc") == 2 and doc_src.count("except _freno.SinElite:") == 2,
   "/analyze-document: el PDF escaneado y el Word de imágenes, sin OCR pagado con el freno")
jur_src = _cuerpo("jurimetria_endpoint")
ok(jur_src.count("permitir_ocr=_ocr_jur") == 2, "Jurimetría: los adjuntos escaneados, sin OCR pagado con el freno")
crono_c, crono_n = _cuerpo("cronologia_cotizar"), _cuerpo("cronologia_crear")
ok("_freno.exigir_elite(" in crono_c and "plan=_acc_crono.get(\"plan\"), usadas=_acc_crono.get(\"usadas\")" in crono_n,
   "Cronología: el freno al cotizar y al cobrar")


print()
if FALLOS:
    print(f"{len(FALLOS)} FALLAN:")
    for f in FALLOS:
        print("   ·", f)
    sys.exit(1)
print("TODO PASA")
