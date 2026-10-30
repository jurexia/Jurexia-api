#!/usr/bin/env python3
"""/analyze-document deja su copia en `respuestas_chat` (28-oct-2026).

La ruta REAL de main.py con un motor de mentira y un Supabase en memoria: sin
red y sin gasto. Sin arranque: el lifespan borra las cachés de Gemini de
producción, así que nada de `with TestClient(main.app)`; httpx habla con la
app por ASGI, que tampoco lo ejecuta.

    .venv/bin/python test_documento_respaldo.py
"""
import asyncio
import json
import os
import sys
import types
import uuid

import dotenv
import fitz
import httpx

# Ni una credencial de producción. main.py busca el .env SUBIENDO directorios,
# y desde un worktree encuentra el del checkout principal: no se lee ninguno, y
# las llaves van falsas a la fuerza, como en test_acceso_platinum.py.
dotenv.load_dotenv = lambda *a, **k: False
for _k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY",
           "MISTRAL_API_KEY", "ELEVENLABS_API_KEY", "AZURE_DOCINT_KEY", "AZURE_SPEECH_KEY"):
    os.environ[_k] = "falsa-para-pruebas"
os.environ.update(QDRANT_URL="http://127.0.0.1:9", QDRANT_API_KEY="",
                  SUPABASE_URL="", SUPABASE_SERVICE_KEY="")

sys.path.insert(0, ".")
import main  # noqa: E402

assert main.supabase_admin is None, "main.py cargó un Supabase de verdad: la prueba no sigue"
import respuesta_guardada as rg  # noqa: E402

USR = str(uuid.uuid4())
CONV = str(uuid.uuid4())
FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


# ── Supabase en memoria: sólo lo que toca el respaldo ─────────────────────

class _Consulta:
    def __init__(self, sb, tabla):
        self.sb, self.tabla, self.op, self.datos, self.filtros = sb, tabla, None, None, []

    def upsert(self, datos, on_conflict=None):
        self.op, self.datos = "upsert", datos
        return self

    def update(self, datos):
        self.op, self.datos = "update", datos
        return self

    def eq(self, col, val):
        self.filtros.append((col, val))
        return self

    def execute(self):
        self.sb.llamadas.append((self.tabla, self.op, dict(self.datos), dict(self.filtros)))
        return types.SimpleNamespace(data=[])


class _Sb:
    def __init__(self):
        self.llamadas = []
        self.auth = types.SimpleNamespace(get_user=self._get_user)

    def _get_user(self, token):
        if token != "buen-token":
            raise RuntimeError("token inválido")
        return types.SimpleNamespace(user=types.SimpleNamespace(id=USR))

    def table(self, nombre):
        return _Consulta(self, nombre)


# ── Un motor que escribe despacio ─────────────────────────────────────────

ANALISIS = ["## Los agravios\n\n", "La recurrente hace valer dos agravios. ",
            "El primero combate la valoración de la prueba.\n\n", "El segundo, la competencia."]


def _trozo(texto=None, fin=None):
    delta = types.SimpleNamespace(content=texto)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta, finish_reason=fin, model_extra={})])


class _Motor:
    def __init__(self):
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        async def gen():
            for t in ANALISIS:
                await asyncio.sleep(0.01)
                yield _trozo(t)
            yield _trozo(None, "stop")
        return gen()


main._via_documento = lambda modelo, esfuerzo=None: (_Motor(), {})


def _pdf() -> bytes:
    """Dos páginas con texto seleccionable que pasa `_texto_nativo_sirve`."""
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


async def _analizar(sb, campos, cabeceras):
    rg.configurar(sb)
    transporte = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transporte, base_url="http://prueba") as c:
        r = await c.post("/analyze-document", headers=cabeceras,
                         files={"file": ("agravios.pdf", _pdf(), "application/pdf")},
                         data={"prompt": "¿Qué agravios hace valer la recurrente?", "usar_acervo": "0",
                               **campos})
    for _ in range(200):          # las escrituras del respaldo van en segundo plano
        if not rg._tareas:
            break
        await asyncio.sleep(0.01)
    eventos = [json.loads(l[6:]) for l in r.text.split("\n") if l.startswith("data: ")]
    return r, eventos


VISIBLE = "📄 **Documento adjunto:** agravios.pdf\n\n¿Qué agravios hace valer la recurrente?"

print("\n1 · CON peticion_id Y TOKEN: la copia lleva el documento leído y el análisis")
sb = _Sb()
PET = str(uuid.uuid4())
r, ev = asyncio.run(_analizar(sb, {"peticion_id": PET, "conversation_id": CONV, "mensaje_visible": VISIBLE},
                              {"Authorization": "Bearer buen-token"}))
doc = next((e["documento"] for e in ev if "documento" in e), None)
ok(r.status_code == 200 and any(e.get("done") for e in ev), "el flujo llega entero y termina con done")
ok(doc is not None and "SEGUNDO AGRAVIO" in doc["texto"], "el evento `documento` trae el texto leído")
filas = [x for x in sb.llamadas if x[0] == "respuestas_chat"]
ok([x[1] for x in filas] == ["upsert", "update"], f"una fila al empezar y su cierre ({[x[1] for x in filas]})")
abre, cierra = filas[0][2], filas[1][2]
ok(abre["peticion_id"] == PET and abre["user_id"] == USR and abre["conversation_id"] == CONV,
   "la fila es del usuario DEL TOKEN y de su conversación")
ok(abre["pregunta"] == VISIBLE and abre["estado"] == "en_curso", "nace con lo que ve el abogado, en curso")
ok(cierra["estado"] == "terminada" and cierra["respuesta"] == "".join(ANALISIS),
   "se cierra terminada con el análisis tal cual lo pinta la pantalla")
ok(cierra.get("pregunta") == rg.mensaje_con_documento(VISIBLE, "agravios.pdf", doc["texto"], doc["recortado"]),
   "la pregunta guardada lleva el documento oculto, igual que `mensajeConDocumento`")

print("\n2 · SIN peticion_id (bundles viejos, la carpeta de expedientes): ni una escritura")
sb2 = _Sb()
r2, ev2 = asyncio.run(_analizar(sb2, {}, {"Authorization": "Bearer buen-token"}))
ok(any(e.get("done") for e in ev2), "el análisis sale igual")
ok(sb2.llamadas == [], "no se toca respuestas_chat")

print("\n3 · SIN token: tampoco se respalda (el dueño sale del token, nunca del cuerpo)")
sb3 = _Sb()
r3, ev3 = asyncio.run(_analizar(sb3, {"peticion_id": str(uuid.uuid4()), "user_id": ""}, {}))
ok(any(e.get("done") for e in ev3) and sb3.llamadas == [], "sale el análisis y no hay copia")

print("\n1b · el formulario del navegador trae CRLF: la copia guarda LF, como la pantalla")
sb5 = _Sb()
asyncio.run(_analizar(sb5, {"peticion_id": str(uuid.uuid4()), "mensaje_visible": VISIBLE.replace("\n", "\r\n")},
                      {"Authorization": "Bearer buen-token"}))
ok(sb5.llamadas and sb5.llamadas[0][2]["pregunta"] == VISIBLE
   and sb5.llamadas[1][2]["pregunta"].startswith(VISIBLE + "\n\n<!-- DOCUMENTO_INICIO -->"),
   "«…agravios.pdf\\r\\n\\r\\n¿Qué…» se guarda «…agravios.pdf\\n\\n¿Qué…»")

print("\n4 · SIN mensaje_visible: la pregunta se arma como la arma ChatInput")
sb4 = _Sb()
asyncio.run(_analizar(sb4, {"peticion_id": str(uuid.uuid4())}, {"Authorization": "Bearer buen-token"}))
ok(sb4.llamadas and sb4.llamadas[0][2]["pregunta"] == VISIBLE, "«📄 **Documento adjunto:** nombre» + la instrucción")

print()
if FALLOS:
    print("FALLOS:\n  · " + "\n  · ".join(FALLOS))
    sys.exit(1)
print("El análisis de un documento deja su copia en el servidor, con el documento leído.")
