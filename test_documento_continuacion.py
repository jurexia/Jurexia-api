# -*- coding: utf-8 -*-
"""CUANDO GEMINI PRO SE CALLA POR RECITACIÓN, LA RESPUESTA CONTINÚA.

Medido el 14-sep-2026: con acervo, el motor Platinum corta la transcripción de
un artículo a media frase (`finish_reason=error`, `native=RECITATION`). No se
puede obligar al motor real a recitar, así que aquí se le sustituye por uno
falso que se calla a propósito, y se comprueba EL CABLEADO: que la respuesta
siga con el motor de documentos desde el último carácter, que la continuación
reciba lo ya escrito y la instrucción de continuar, que no se repita nada, y
que si la continuación también se corta, se diga con la nota y no en silencio.

Sin red: la búsqueda en el acervo también se sustituye. Sin `user_id`, así que
no hay perfil que leer ni consulta que cobrar.
"""
import io, json, re, sys, os
from types import SimpleNamespace
sys.path.insert(0, os.getcwd())

fallos = []
def ok(cond, nota):
    print(f"  {'OK ' if cond else 'MAL'} {nota}")
    if not cond: fallos.append(nota)

from docx import Document
d = Document(); d.add_paragraph("Contrato de arrendamiento en Querétaro. SEGUNDA. Plazo de un año.")
buf = io.BytesIO(); d.save(buf)

from fastapi.testclient import TestClient
import main
from documento_acervo import INSTRUCCION_CONTINUAR, NOTA_TRANSCRIPCION_CORTADA

def trozo(texto=None, fin=None, nativo=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content=texto), finish_reason=fin,
        model_extra={"native_finish_reason": nativo} if nativo else {})])

llamadas = []
def motor_falso(guion):
    """`guion`: lista de listas de trozos, una por llamada al motor."""
    async def _crear(client, etiqueta, **kw):
        llamadas.append({"etiqueta": etiqueta, "model": kw.get("model"), "messages": kw.get("messages")})
        trozos = guion[len(llamadas) - 1]
        async def _gen():
            for t in trozos:
                yield t
        return _gen()
    return _crear

async def sin_acervo(**kw):
    return []

def correr(guion):
    llamadas.clear()
    main._crear_con_amortiguador = motor_falso(guion)
    main.hybrid_search_all_silos = sin_acervo
    tokens = []
    with TestClient(main.app) as c:
        with c.stream("POST", "/analyze-document",
                      files={"file": ("c.docx", buf.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
                      data={"prompt": "Analiza la cláusula segunda", "estado": "QUERETARO"}) as r:
            for line in r.iter_lines():
                if isinstance(line, bytes): line = line.decode("utf-8", "replace")
                if line.startswith("data: "):
                    dd = json.loads(line[6:])
                    if "token" in dd: tokens.append(dd["token"])
    return "".join(tokens)

# ── 1. Pro se calla por RECITATION; Flash termina ───────────────────────────
texto = correr([
    [trozo("El artículo 2378 dice: «Vencido un contrato"), trozo(None, "error", "RECITATION")],
    [trozo(" de arrendamiento, tendrá derecho el inquilino…»"), trozo(" Por tanto, opera la prórroga."), trozo(None, "stop", "STOP")],
])
ok(texto == "El artículo 2378 dice: «Vencido un contrato de arrendamiento, tendrá derecho el inquilino…» Por tanto, opera la prórroga.",
   f"la respuesta se lee de corrido, sin repetir ni saludar: {texto!r}")
ok(len(llamadas) == 2, f"dos llamadas al motor ({len(llamadas)})")
ok(llamadas[1]["etiqueta"] == "analyze-document-continuacion", "la segunda es la continuación")
ok(llamadas[1]["model"] == main.DOCUMENT_MODEL, f"la continuación usa el motor de documentos ({llamadas[1]['model']})")
msgs = llamadas[1]["messages"]
ok(msgs[0]["role"] == "system" and msgs[1]["role"] == "user", "conserva sistema y consulta originales")
ok(msgs[2] == {"role": "assistant", "content": "El artículo 2378 dice: «Vencido un contrato"}, "le entrega lo ya escrito como turno del asistente")
ok(msgs[3] == {"role": "user", "content": INSTRUCCION_CONTINUAR}, "y la instrucción de continuar")
ok(NOTA_TRANSCRIPCION_CORTADA not in texto, "sin nota: la continuación terminó bien")

# ── 2. La continuación TAMBIÉN se corta: se dice ────────────────────────────
texto2 = correr([
    [trozo("Transcribo: «Artículo 2382."), trozo(None, "error", "RECITATION")],
    [trozo(" Si después de terminado"), trozo(None, "error", "RECITATION")],
])
ok(texto2.startswith("Transcribo: «Artículo 2382. Si después de terminado"), "la continuación añadió lo que pudo")
ok(texto2.endswith(NOTA_TRANSCRIPCION_CORTADA), "y al cortarse otra vez, lo dice con la nota")
ok(len(llamadas) == 2, "no hay tercera llamada: se continúa una vez, no en bucle")

# ── 3. Un STOP normal no dispara nada ───────────────────────────────────────
texto3 = correr([[trozo("Análisis completo."), trozo(None, "stop", "STOP")]])
ok(texto3 == "Análisis completo." and len(llamadas) == 1, "con STOP normal hay una sola llamada y ningún añadido")

print()
if fallos:
    print(f"{len(fallos)} FALLO(S):"); [print("  -", f) for f in fallos]; sys.exit(1)
print("todo en orden")
