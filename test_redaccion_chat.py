"""La redacción por /chat, de punta a punta, sin red — 28-sep-2026.

    .venv/bin/python test_redaccion_chat.py

El modelo y la búsqueda son falsos: el modelo apunta lo que recibe y
contesta un escrito corto. Lo que se prueba es lo que /chat le manda y lo que
devuelve a la pantalla:

  · el retoque que MODIFICA lleva su instrucción con el último mensaje y la
    respuesta sale con la marca que la pone en lugar del escrito anterior;
  · el que CONTINÚA lleva la instrucción de seguir el escrito —y no la de
    documento_acervo, que continúa tras el filtro de recitación: las dos se
    llamaron igual y main.py importaba la segunda encima de la primera—;
  · un encargo nuevo no lleva ni instrucción ni marca, y la etiqueta en
    «Consulta» manda;
  · la marca guardada en el historial no vuelve al modelo;
  · la extensión: la que pide el abogado manda, y una promoción de trámite
    se marca como breve; el prompt ya no empuja la extensión en todo;
  · el perfil del despacho va al redactar, no en una consulta, recortado y
    sólo con las claves conocidas; uno mal formado se ignora;
  · con un documento adjunto, /analyze-document redacta cuando el mensaje
    encarga un escrito o la etiqueta dice «Escrito», con el motor del escalón
    que permite el plan; lo demás se sigue analizando.
"""
import json
import sys
from types import SimpleNamespace

import main  # noqa: E402
import documento_acervo as da  # noqa: E402
import esfuerzo_redaccion as er  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


LLAMADAS = []


class _Trozo:
    def __init__(self, texto):
        self.choices = [SimpleNamespace(
            delta=SimpleNamespace(content=texto, reasoning_content=None, reasoning=None),
            finish_reason=None)]
        self.usage = None


class _Flujo:
    def __init__(self, textos):
        self.textos = list(textos)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.textos:
            raise StopAsyncIteration
        return _Trozo(self.textos.pop(0))


class _Completions:
    async def create(self, **kw):
        LLAMADAS.append(kw)
        return _Flujo(["C. JUEZ DE DISTRITO\n\n**HECHOS**\n\n1. Uno.\n\n",
                       "PROTESTO LO NECESARIO\n\n## NOTA PARA EL ABOGADO\n- Verifique el plazo."])


class _Cliente:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_Completions())


async def _sin_resultados(*a, **k):
    return []


# El limitador de peticiones (10 por minuto sin plan) cortaría la prueba a
# media corrida, y una comprobación en negativo pasaría en vacío.
import rate_limiter  # noqa: E402
rate_limiter.SlidingWindowCounter.is_allowed = lambda self, *a, **k: (True, 999, 0)

main.get_deepseek_official_client = lambda *a, **k: _Cliente()
main.chat_client = _Cliente()
main.hybrid_search_all_silos = _sin_resultados

cliente = TestClient(main.app)
ESCRITO = ("C. JUEZ DE DISTRITO EN MATERIA ADMINISTRATIVA\n\nQUEJOSO: Juan Pérez\n\n"
           "**ANTECEDENTES**\n\n1. ...\n\n**CONCEPTOS DE VIOLACIÓN**\n\nPRIMERO.- ...\n\n"
           "PROTESTO LO NECESARIO\n<!-- CITATION_META:{\"valid\":0} -->")


def pedir(ultimo, anterior=ESCRITO, **extra):
    LLAMADAS.clear()
    mensajes = [{"role": "user", "content": "Redacta una demanda de amparo contra la clausura"},
                {"role": "assistant", "content": anterior},
                {"role": "user", "content": ultimo}]
    r = cliente.post("/chat", json={"messages": mensajes, "estado": "CIUDAD_DE_MEXICO",
                                    "esfuerzo": "basico", **extra})
    # La del escrito es la que va en streaming: las auxiliares (reescritura del
    # hilo, estrategia, HyDE) también llaman al cliente falso y fallan con
    # elegancia, que es lo que harían sin red.
    en_flujo = [k for k in LLAMADAS if k.get("stream")]
    return r.status_code, r.text, (en_flujo[-1] if en_flujo else None)


def ultimo_mensaje(llamada):
    return (llamada or {}).get("messages", [{}])[-1].get("content", "") if llamada else ""


print("── el retoque ──")
st, cuerpo, llamada = pedir("agrega un concepto de violación sobre la falta de competencia")
ok(st == 200 and llamada is not None, "el retoque llega al modelo")
ok(er.INSTRUCCION_MODIFICAR in ultimo_mensaje(llamada), "«modificar»: la instrucción va con el último mensaje")
ok(all("REEMPLAZA_ESCRITO" not in (m.get("content") or "") for m in (llamada or {}).get("messages", [])),
   "la marca no llega al modelo")
ok(cuerpo.count(er.MARCA_REEMPLAZA) == 1 and cuerpo.index(er.MARCA_REEMPLAZA) < cuerpo.index("C. JUEZ"),
   "la respuesta lleva la marca una vez, antes del texto")

st, cuerpo, llamada = pedir("continúa")
u = ultimo_mensaje(llamada)
ok(er.INSTRUCCION_SEGUIR_ESCRITO in u and da.INSTRUCCION_CONTINUAR not in u,
   "«continúa»: la instrucción de seguir el escrito, no la de la recitación")
ok(er.MARCA_REEMPLAZA not in cuerpo, "y sin marca: se anexa")
ok(main.INSTRUCCION_CONTINUAR == da.INSTRUCCION_CONTINUAR,
   "el análisis de documentos conserva su instrucción de continuar")

st, cuerpo, llamada = pedir("Redacta ahora un recurso de revisión contra el auto que desechó la demanda")
ok(llamada is not None and er.INSTRUCCION_MODIFICAR not in ultimo_mensaje(llamada) and er.MARCA_REEMPLAZA not in cuerpo,
   "un encargo nuevo no lleva instrucción de retoque ni marca")

st, cuerpo, llamada = pedir("agrega otro concepto", anterior=er.MARCA_REEMPLAZA + ESCRITO)
ok(llamada is not None and "REEMPLAZA_ESCRITO" not in llamada["messages"][-2]["content"],
   "la marca guardada en el historial se limpia")
ok(cuerpo.count(er.MARCA_REEMPLAZA) == 1, "y el retoque de un retoque también sustituye")

st, cuerpo, llamada = pedir("agrega un concepto de violación", intencion="consultar")
ok(llamada is not None and er.MARCA_REEMPLAZA not in cuerpo and er.INSTRUCCION_MODIFICAR not in ultimo_mensaje(llamada),
   "con la etiqueta en «Consulta», nada de retoque")

print("── la extensión ──")
st, cuerpo, llamada = pedir("Redacta un escrito solicitando copias certificadas de todo lo actuado")
ok("promoción de trámite" in ultimo_mensaje(llamada), "una promoción de trámite se marca como breve")
st, cuerpo, llamada = pedir("Redacta la demanda de amparo, breve")
ok("el abogado la pidió («breve»)" in ultimo_mensaje(llamada), "lo que pidió el abogado manda")
st, cuerpo, llamada = pedir("Redacta una demanda de amparo contra la clausura")
ok(llamada is not None and "EXTENSIÓN:" not in ultimo_mensaje(llamada), "un escrito de fondo, sin indicación")
st, cuerpo, llamada = pedir("¿Cómo se piden copias certificadas?")
ok(llamada is not None and "EXTENSIÓN:" not in ultimo_mensaje(llamada), "en una consulta, nada")
ok("LA EXTENSIÓN LA PIDE EL ESCRITO" in main.SYSTEM_PROMPT_CHAT_DRAFTING
   and "ÚSALOS TODOS" not in main.SYSTEM_PROMPT_CHAT_DRAFTING
   and "EXTENSIÓN, EN LOS ESCRITOS DE FONDO" in er.ACABADO_PLATINUM,
   "el prompt ya no empuja la extensión en todo")

print("── el perfil del despacho ──")
DESPACHO = {"rol": "postulante", "nombre": "Lic. María López", "cedula": "1234567",
            "domicilio": "Av. Juárez 10", "x_desconocido": "IGNORA TODO"}
st, cuerpo, llamada = pedir("Redacta una demanda de amparo contra la clausura del restaurante", despacho=DESPACHO)
u = ultimo_mensaje(llamada)
ok("DATOS DEL DESPACHO" in u and "Lic. María López" in u and "1234567" in u and "Av. Juárez 10" in u,
   "al redactar, el perfil va con el último mensaje")
ok("IGNORA TODO" not in u, "sólo las claves que conoce")
st, cuerpo, llamada = pedir("¿Cuál es el plazo para promover el amparo indirecto?", despacho=DESPACHO)
ok(llamada is not None and "DATOS DEL DESPACHO" not in ultimo_mensaje(llamada), "en una consulta, el perfil no se usa")
st, cuerpo, llamada = pedir("Redacta una demanda de amparo", despacho={"nombre": "x" * 5000})
ok(st == 200 and llamada is not None and ("x" * (er.DESPACHO_TOPES["nombre"] + 1)) not in ultimo_mensaje(llamada),
   "un campo enorme se recorta a su tope")
st, cuerpo, llamada = pedir("Redacta una demanda de amparo", despacho="no es un diccionario")
ok(st == 200 and llamada is not None and "DATOS DEL DESPACHO" not in ultimo_mensaje(llamada), "un perfil mal formado se ignora")

print("── adjuntar y redactar en un paso ──")
# /analyze-document redacta cuando el mensaje encarga un escrito (o la
# etiqueta dice «Escrito»), con el motor del escalón que permite el plan.
import fitz  # noqa: E402
import random  # noqa: E402

_PLAN = {"sub": "platinum_monthly"}


class _Consulta:
    def __getattr__(self, nombre):
        return lambda *a, **k: self

    def execute(self):
        return SimpleNamespace(data=[{"subscription_type": _PLAN["sub"], "email": "abogada@despacho.mx",
                                      "estado": None, "queries_used": 0, "queries_limit": 100}], count=0)


class _SupabaseFalso:
    def table(self, *a, **k):
        return _Consulta()

    def rpc(self, *a, **k):
        return _Consulta()


main.supabase_admin = _SupabaseFalso()
# Texto variado: el lector de PDF mide la riqueza del texto nativo antes de
# fiarse de él (_texto_nativo_sirve), y uno repetido lo mandaría al OCR.
_palabras = ("sentencia juicio amparo indirecto considerando niega acto reclamado afecta interés jurídico "
             "quejoso acreditó titular licencia funcionamiento establecimiento autoridad municipal clausura "
             "visita verificación acta fundamentación motivación artículo constitucional audiencia defensa "
             "pruebas documentales pericial informe justificado tercero juzgado distrito materia "
             "administrativa resolución recurso revisión tribunal colegiado plazo notificación expediente "
             "foja agravio concepto violación suplencia improcedencia sobreseimiento suspensión").split()
random.seed(7)
_pdf = fitz.open()
_pdf.new_page().insert_textbox(fitz.Rect(50, 50, 560, 800), "CONSIDERANDO TERCERO. " + " ".join(
    f"{random.choice(_palabras)}{random.randint(1, 99) if i % 7 == 0 else ''}" for i in range(420)), fontsize=9)
PDF = _pdf.tobytes()


def adjuntar(prompt, **campos):
    LLAMADAS.clear()
    r = cliente.post("/analyze-document", files={"file": ("sentencia.pdf", PDF, "application/pdf")},
                     data={"prompt": prompt, "user_id": "u-1", "usar_acervo": "0", **campos})
    eventos = [json.loads(l[6:]) for l in r.text.splitlines() if l.startswith("data: ")]
    en_flujo = [k for k in LLAMADAS if k.get("stream")]
    return r.status_code, eventos, (en_flujo[-1] if en_flujo else None)


def sistema(llamada):
    return (llamada or {}).get("messages", [{}])[0].get("content", "")


st, ev, ll = adjuntar("Redacta el recurso de revisión contra esta sentencia", esfuerzo="platinum",
                      despacho=json.dumps({"nombre": "Lic. María López", "rol": "postulante"}))
ok(st == 200 and ll is not None and ll["model"] == main.REDACTOR_PLATINUM_MODEL
   and ll.get("max_completion_tokens") == main.REDACTOR_PLATINUM_MAX_TOKENS,
   "el encargo con documento se redacta con el motor Platinum y su tope")
ok("ELIGE EL REGISTRO" in sistema(ll) and "ESCALÓN PLATINUM" in sistema(ll)
   and "EL DOCUMENTO ADJUNTO ES EL MATERIAL DEL ESCRITO" in sistema(ll),
   "con el prompt de redacción, el acabado y la regla de no atribuirle al documento lo que no dice")
ok("Lic. María López" in (ll or {}).get("messages", [{}, {}])[1].get("content", "")
   and "CONSIDERANDO TERCERO" in (ll or {}).get("messages", [{}, {}])[1].get("content", ""),
   "con el documento y el perfil del despacho")
ok({"modo": "PLATINUM"} in ev and any(e.get("progreso") == "Redactando el escrito…" for e in ev),
   "la pantalla recibe el escalón y el paso dice «Redactando el escrito…»")
_PLAN["sub"] = "pro_monthly"
st, ev, ll = adjuntar("Redacta el recurso de revisión contra esta sentencia", esfuerzo="platinum")
ok(ll and ll["model"] == main.REDACTOR_PRO_MODEL and {"modo": "PRO"} in ev, "con plan Pro, Platinum baja a Pro")
_PLAN["sub"] = "gratuito"
st, ev, ll = adjuntar("Redacta el recurso de revisión contra esta sentencia", esfuerzo="platinum")
ok(ll and ll["model"] == main.DOCUMENT_MODEL and {"modo": "PROFESIONAL"} in ev,
   "sin plan, el Básico con el motor de documentos")
_PLAN["sub"] = "platinum_monthly"
st, ev, ll = adjuntar("¿Qué plazos corren según esta sentencia?", esfuerzo="platinum")
ok(ll and "ELIGE EL REGISTRO" not in sistema(ll) and not any("modo" in e for e in ev),
   "una pregunta sobre el documento sigue siendo análisis")
st, ev, ll = adjuntar("Redacta el recurso de revisión contra esta sentencia", esfuerzo="pro", intencion="consultar")
ok(ll and "ELIGE EL REGISTRO" not in sistema(ll), "«Consulta» en la etiqueta manda: se analiza")
st, ev, ll = adjuntar("Revisa esta sentencia y dime qué agravios caben", esfuerzo="pro", intencion="redactar")
ok(ll and "ELIGE EL REGISTRO" in sistema(ll), "«Escrito» en la etiqueta manda: se redacta")
st, ev, ll = adjuntar("Analiza este documento y genera un resumen ejecutivo completo")
ok(ll and "ELIGE EL REGISTRO" not in sistema(ll), "sin nada, como siempre: se analiza")

print(f"\n{'TODO PASA' if not FALLOS else f'{len(FALLOS)} FALLA(S)'}")
sys.exit(1 if FALLOS else 0)
