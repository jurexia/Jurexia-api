#!/usr/bin/env python3
"""Pruebas sin red de respuesta_guardada.py. No importa main: nada de
arranque, nada que toque las cachés de producción.

    python3 test_respuesta_guardada.py
"""
import asyncio
import sys
import types
import uuid

import respuesta_guardada as rg

PET = str(uuid.uuid4())
USR = str(uuid.uuid4())
CONV = str(uuid.uuid4())


# ── texto_para_historial ───────────────────────────────────────────────────

def test_quita_control_y_conserva_meta():
    crudo = ("<!--PING--><!--PASO:entender--><!--PASO:buscar|federal--><!--SOURCES:12-->"
             "<!--MODE:PLATINUM-->Respuesta **fundada**.<!--REGISTROS_FUERA:123,456-->"
             "\n\n<!-- CITATION_META:{\"valid\": 1} -->\n\n<!-- PRECEDENTES_META:[] -->")
    t = rg.texto_para_historial(crudo)
    assert t.startswith("Respuesta **fundada**.")
    assert "PASO" not in t and "PING" not in t and "SOURCES" not in t and "MODE:" not in t
    assert "REGISTROS_FUERA" not in t
    assert "<!-- CITATION_META:{\"valid\": 1} -->" in t
    assert "PRECEDENTES_META" in t


def test_fuentes_previas_fuera():
    crudo = "## Resumen\n\n\n<!-- FUENTES_PREVIAS:{\"a\": {\"origen\": \"x\"}} -->\n\nTexto.\n\n<!-- CITATION_META:{} -->"
    t = rg.texto_para_historial(crudo)
    assert "FUENTES_PREVIAS" not in t
    assert t.startswith("## Resumen") and "Texto." in t and "CITATION_META" in t


def test_razonamiento_en_vivo_como_bloque():
    crudo = "<!--thinking-->Pienso <!--thinking-->más<!--/thinking-->La respuesta."
    assert rg.texto_para_historial(crudo) == "<!--THINKING_START-->Pienso más<!--THINKING_END-->La respuesta."


def test_razonamiento_sin_respuesta_lleva_aviso():
    t = rg.texto_para_historial("<!--thinking-->sólo razono<!--/thinking-->")
    assert t.startswith("<!--THINKING_START-->sólo razono<!--THINKING_END-->")
    assert "agotó los tokens" in t


def test_sin_marcadores_queda_igual():
    assert rg.texto_para_historial("Hola.\n\nAdiós.") == "Hola.\n\nAdiós."


# ── con_respaldo con un Supabase de mentira ───────────────────────────────

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
        if self.sb.falla:
            raise RuntimeError("supabase caído")
        self.sb.llamadas.append((self.tabla, self.op, self.datos, tuple(self.filtros)))
        return types.SimpleNamespace(data=[])


class _SbFalso:
    def __init__(self, uid=USR, falla=False):
        self.llamadas, self.falla, self._uid = [], falla, uid
        self.auth = types.SimpleNamespace(get_user=self._get_user)

    def _get_user(self, token):
        if token != "buen-token" or not self._uid:
            raise RuntimeError("token inválido")
        return types.SimpleNamespace(user=types.SimpleNamespace(id=self._uid))

    def table(self, nombre):
        return _Consulta(self, nombre)


async def _flujo(trozos, corta_en=None):
    for i, t in enumerate(trozos):
        if corta_en is not None and i == corta_en:
            raise asyncio.CancelledError()
        yield t


async def _correr(sb, trozos, peticion=PET, auth="Bearer buen-token", corta_en=None):
    rg.configurar(sb)
    salida = []
    gen = rg.con_respaldo(_flujo(trozos, corta_en), authorization=auth, peticion_id=peticion,
                          conversation_id=CONV, expediente_id=None, pregunta="¿Procede?", estado="QUERETARO")
    try:
        async for t in gen:
            salida.append(t)
    except asyncio.CancelledError:
        pass
    # que terminen las escrituras en segundo plano
    for _ in range(50):
        if not rg._tareas:
            break
        await asyncio.sleep(0.01)
    return salida


def test_pasa_el_stream_tal_cual_y_guarda():
    sb = _SbFalso()
    trozos = ["<!--PASO:entender-->", "Primera parte. ", "Segunda.", "\n\n<!-- CITATION_META:{} -->"]
    salida = asyncio.run(_correr(sb, trozos))
    assert salida == trozos
    (t1, op1, d1, _), (t2, op2, d2, f2) = sb.llamadas
    assert (t1, op1) == ("respuestas_chat", "upsert")
    assert d1["user_id"] == USR and d1["peticion_id"] == PET and d1["conversation_id"] == CONV
    assert d1["estado"] == "en_curso" and d1["pregunta"] == "¿Procede?" and d1["guardada_at"] is None
    assert (t2, op2) == ("respuestas_chat", "update")
    assert d2["estado"] == "terminada"
    assert d2["respuesta"] == "Primera parte. Segunda.\n\n<!-- CITATION_META:{} -->"
    assert "pregunta" not in d2   # el chat no reescribe la pregunta al cerrar
    assert ("peticion_id", PET) in f2 and ("user_id", USR) in f2


def test_flujo_cancelado_queda_interrumpida():
    """Si el flujo de dentro se cancela (Detener, tope, apagado), la copia
    queda interrumpida con lo que hubiera."""
    sb = _SbFalso()
    asyncio.run(_correr(sb, ["Va ", "a medias", "nunca llega"], corta_en=2))
    assert sb.llamadas[-1][1] == "update"
    assert sb.llamadas[-1][2]["estado"] == "interrumpida"
    assert sb.llamadas[-1][2]["respuesta"] == "Va a medias"


def test_sin_peticion_o_sin_token_no_toca_la_base():
    for peticion, auth in ((None, "Bearer buen-token"), ("no-es-uuid", "Bearer buen-token"), (PET, None)):
        sb = _SbFalso()
        salida = asyncio.run(_correr(sb, ["a", "b"], peticion=peticion, auth=auth))
        assert salida == ["a", "b"]
        assert sb.llamadas == []


def test_token_ajeno_no_respalda():
    sb = _SbFalso()
    salida = asyncio.run(_correr(sb, ["a"], auth="Bearer otro"))
    assert salida == ["a"] and sb.llamadas == []


def test_supabase_caido_no_rompe_la_consulta():
    sb = _SbFalso(falla=True)
    salida = asyncio.run(_correr(sb, ["la ", "respuesta"]))
    assert salida == ["la ", "respuesta"]


def test_la_pantalla_se_va_y_el_de_dentro_sigue():
    """EL TRABAJO NO SE VA CON LA PANTALLA (30-oct-2026). Starlette cierra el
    generador de fuera (aclose) cuando el navegador se va: el de dentro ya no
    se corta ahí, sigue hasta el final y la copia queda terminada y entera."""
    eventos = []

    async def dentro():
        try:
            yield "uno"
            await asyncio.sleep(0.05)
            yield "dos"
            await asyncio.sleep(0.05)
            yield "tres"
            eventos.append("terminó")
        finally:
            eventos.append("cerrado")

    sb = _SbFalso()

    async def correr():
        rg.configurar(sb)
        gen = rg.con_respaldo(dentro(), authorization="Bearer buen-token", peticion_id=PET,
                              conversation_id=None, expediente_id=None, pregunta="", estado=None)
        assert await gen.__anext__() == "uno"
        await gen.aclose()                     # la pantalla se fue
        assert eventos == []                   # y el de dentro sigue vivo
        for _ in range(100):
            if not rg._tareas and not rg._productores:
                break
            await asyncio.sleep(0.01)

    asyncio.run(correr())
    assert eventos == ["terminó", "cerrado"]
    cierre = sb.llamadas[-1][2]
    assert cierre["estado"] == "terminada" and cierre["respuesta"] == "unodostres"


class _SbConEstado(_SbFalso):
    """Como _SbFalso, y además contesta el `select estado` con lo que diga
    `self.estado`: así la pantalla «pulsa Detener»."""
    def __init__(self):
        super().__init__()
        self.estado = "en_curso"

    def table(self, nombre):
        c = _Consulta(self, nombre)
        sb = self

        def select(_cols):
            c.op = "select"
            return c

        def limit(_n):
            return c

        ejecutar = c.execute

        def execute():
            if c.op == "select":
                return types.SimpleNamespace(data=[{"estado": sb.estado}])
            return ejecutar()

        c.select, c.limit, c.execute = select, limit, execute
        return c


async def _sin_fin(eventos):
    try:
        i = 0
        while True:
            yield f"[{i}]"
            i += 1
            await asyncio.sleep(0.02)
    except BaseException:
        eventos.append("cancelado")
        raise


def _correr_sin_pantalla(sb, eventos, *, tope, al_irse=None):
    async def correr():
        rg.configurar(sb)
        gen = rg.con_respaldo(_sin_fin(eventos), authorization="Bearer buen-token", peticion_id=PET,
                              conversation_id=None, expediente_id=None, pregunta="", estado=None,
                              tope_sin_pantalla_s=tope, mirar_cada_s=0.05)
        await gen.__anext__()
        await gen.aclose()
        if al_irse:
            al_irse()
        for _ in range(300):
            if not rg._tareas and not rg._productores:
                break
            await asyncio.sleep(0.01)
    asyncio.run(correr())


def test_detener_corta_el_de_dentro_aunque_no_haya_pantalla():
    """Sin pantalla, la copia puesta en «interrumpida» (el botón Detener) corta
    el flujo: el de dentro se cancela —y ahí /analyze-document devuelve la
    consulta si no escribió nada— y la copia queda interrumpida."""
    sb, eventos = _SbConEstado(), []
    _correr_sin_pantalla(sb, eventos, tope=60, al_irse=lambda: setattr(sb, "estado", "interrumpida"))
    assert eventos == ["cancelado"]
    cierre = sb.llamadas[-1][2]
    assert cierre["estado"] == "interrumpida" and cierre["respuesta"].startswith("[0]")


def test_el_tope_sin_pantalla_corta():
    sb, eventos = _SbConEstado(), []
    _correr_sin_pantalla(sb, eventos, tope=0.2)
    assert eventos == ["cancelado"]
    assert sb.llamadas[-1][2]["estado"] == "interrumpida"


# ── /analyze-document: la copia se arma como la armaría la pantalla ───────

VISIBLE = "📄 **Documento adjunto:** agravios.pdf\n\n¿Qué agravios hace valer la recurrente?"


def _sse(d):
    import json as _j
    return "data: " + _j.dumps(d, ensure_ascii=False) + "\n\n"


async def _correr_doc(sb, trozos, corta_en=None):
    rg.configurar(sb)
    salida = []
    gen = rg.con_respaldo(_flujo(trozos, corta_en), authorization="Bearer buen-token", peticion_id=PET,
                          conversation_id=CONV, expediente_id=None, pregunta=VISIBLE, estado="TLAXCALA",
                          formato="documento")
    try:
        async for t in gen:
            salida.append(t)
    except asyncio.CancelledError:
        pass
    for _ in range(50):
        if not rg._tareas:
            break
        await asyncio.sleep(0.01)
    return salida


def test_mensaje_con_documento_es_espejo_del_front():
    # El mismo texto que arma `mensajeConDocumento` (src/lib/memoria-conversacion.ts).
    assert rg.mensaje_con_documento("V", "a.pdf", "TEXTO") == (
        "V\n\n<!-- DOCUMENTO_INICIO -->\n"
        "CONTENIDO DEL DOCUMENTO ADJUNTO «a.pdf» (texto leído por Iurexia):\n\n"
        "TEXTO\n<!-- DOCUMENTO_FIN -->")
    assert "(texto leído por Iurexia; es la primera parte, el documento es más largo):" in \
        rg.mensaje_con_documento("V", "a.pdf", "T", recortado=True)
    assert rg.encabezado_de_adjunto("a.pdf", "¿Procede?") == "📄 **Documento adjunto:** a.pdf\n\n¿Procede?"


def test_documento_guarda_la_pregunta_con_el_documento_y_los_tokens():
    sb = _SbFalso()
    trozos = [_sse({"progreso": "Leyendo el documento…"}), ": latido\n\n",
              _sse({"documento": {"nombre": "agravios.pdf", "texto": "PRIMER AGRAVIO. La sentencia…",
                                  "recortado": False}}),
              _sse({"limite_plan": {"leidas": 20, "total": 31, "plan": "Pro"}}),
              _sse({"token": "## Agravios\n\n"}), _sse({"token": "La recurrente sostiene…"}),
              _sse({"token": "\n<!-- CITATION_META:{} -->"}),
              _sse({"done": True, "filename": "agravios.pdf", "chars_analyzed": 30})]
    salida = asyncio.run(_correr_doc(sb, trozos))
    assert salida == trozos                                   # el flujo pasa intacto
    (_, op1, d1, _), (_, op2, d2, _) = sb.llamadas
    assert op1 == "upsert" and d1["pregunta"] == VISIBLE and d1["estado"] == "en_curso"
    assert op2 == "update" and d2["estado"] == "terminada"
    assert d2["respuesta"] == "## Agravios\n\nLa recurrente sostiene…\n<!-- CITATION_META:{} -->"
    assert d2["pregunta"] == rg.mensaje_con_documento(VISIBLE, "agravios.pdf", "PRIMER AGRAVIO. La sentencia…")


def test_documento_error_sustituye_a_lo_escrito():
    sb = _SbFalso()
    asyncio.run(_correr_doc(sb, [_sse({"token": "Empiezo"}),
                                 _sse({"error": "No pudimos completar el análisis de este documento."})]))
    d = sb.llamadas[-1][2]
    assert d["estado"] == "terminada"
    assert d["respuesta"] == "❌ Error al analizar documento: No pudimos completar el análisis de este documento."
    assert "pregunta" not in d                                # nunca se leyó el documento


def test_documento_cortado_queda_interrumpida_con_el_documento():
    sb = _SbFalso()
    trozos = [_sse({"documento": {"nombre": "a.pdf", "texto": "HECHOS", "recortado": True}}),
              _sse({"token": "Va "}), _sse({"token": "a medias"}), _sse({"token": "nunca llega"})]
    asyncio.run(_correr_doc(sb, trozos, corta_en=3))
    d = sb.llamadas[-1][2]
    assert d["estado"] == "interrumpida" and d["respuesta"] == "Va a medias"
    assert d["pregunta"] == rg.mensaje_con_documento(VISIBLE, "a.pdf", "HECHOS", recortado=True)


def test_documento_enorme_cabe_en_la_columna_con_su_cierre():
    p, _ = rg.armar_documento(_sse({"documento": {"nombre": "x.pdf", "texto": "palabra " * 40_000}}), VISIBLE)
    assert len(p) <= rg.TOPE_PREGUNTA
    assert p.endswith("\n<!-- DOCUMENTO_FIN -->") and "es la primera parte" in p


def test_traba():
    for cuerpo, ok in ((b'{"etapa":"sesion","ms":20000}', True),
                       (b'{"etapa":"rm -rf","ms":1}', False),
                       (b'no es json', False)):
        class _Req:
            async def body(self, c=cuerpo):
                return c
        assert asyncio.run(rg.chat_traba(_Req()))["ok"] is ok


if __name__ == "__main__":
    fallos = []
    for nombre, f in list(globals().items()):
        if nombre.startswith("test_") and callable(f):
            try:
                f()
                print(f"  ok   {nombre}")
            except Exception as e:
                fallos.append(nombre)
                print(f"  MAL  {nombre}: {type(e).__name__}: {e}")
    print()
    if fallos:
        print("FALLOS: " + ", ".join(fallos)); sys.exit(1)
    print("La respuesta se respalda sin tocar el stream ni romper la consulta.")
