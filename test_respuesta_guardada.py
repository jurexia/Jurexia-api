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
    assert ("peticion_id", PET) in f2 and ("user_id", USR) in f2


def test_corte_del_navegador_queda_interrumpida():
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


def test_aclose_cierra_el_de_dentro():
    """Starlette puede cerrar el generador de fuera con aclose(): el de dentro
    tiene que cerrarse en ese momento (su finally corre), como antes."""
    cerrado = []

    async def dentro():
        try:
            yield "uno"
            yield "dos"
        finally:
            cerrado.append(True)

    async def correr():
        rg.configurar(_SbFalso())
        gen = rg.con_respaldo(dentro(), authorization="Bearer buen-token", peticion_id=PET,
                              conversation_id=None, expediente_id=None, pregunta="", estado=None)
        assert await gen.__anext__() == "uno"
        await gen.aclose()
        for _ in range(50):
            if not rg._tareas:
                break
            await asyncio.sleep(0.01)

    asyncio.run(correr())
    assert cerrado == [True]


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
