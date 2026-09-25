"""Pruebas del agente de flujos: la salida se normaliza pase lo que pase.

El frontend pinta formularios con esta respuesta; un campo ausente o con otro
tipo no puede tumbar la pantalla del abogado.

    .venv/bin/python test_flujo_agente.py
"""
import asyncio

from fastapi import HTTPException

import flujo_agente as fa
from flujo_agente import CampoFlujo, DeducirRequest, normalizar

CAMPOS = [
    CampoFlujo(id="quejoso", etiqueta="Quejoso", tipo="texto"),
    CampoFlujo(id="autoridades", etiqueta="Autoridades responsables", tipo="varias"),
    CampoFlujo(id="interes", etiqueta="Interés", tipo="opcion", opciones=["Jurídico", "Legítimo"]),
    CampoFlujo(id="notificacion", etiqueta="Fecha de notificación", tipo="fecha"),
]


def test_sin_respuesta_deja_huecos_honestos():
    r = normalizar(None, CAMPOS)
    assert [c["id"] for c in r["campos"]] == ["quejoso", "autoridades", "interes", "notificacion"]
    assert all(c["propuesta"] is None and c["confianza"] == "ninguna" for c in r["campos"])
    # Las opciones fijas del flujo se ofrecen aunque el modelo no conteste.
    assert r["campos"][2]["opciones"] == ["Jurídico", "Legítimo"]
    assert r["pedir_documento"] is None and r["pensamiento"] == [] and r["consulta"] == ""


def test_tipos_equivocados_se_corrigen():
    crudo = {
        "pensamiento": "una sola frase, no lista",
        "campos": [
            {"id": "quejoso", "propuesta": ["María", "López"], "confianza": "altísima"},
            {"id": "autoridades", "propuesta": "Subdelegado del IMSS", "opciones": None},
            {"id": "interes", "propuesta": "Jurídico", "opciones": ["jurídico", "Otro"]},
            {"id": "desconocido", "propuesta": "x"},
        ],
        "pedir_documento": {"nombre": "", "motivo": "x"},
    }
    r = normalizar(crudo, CAMPOS)
    q, a, i, n = r["campos"]
    assert q["propuesta"] == "María; López" and q["confianza"] == "media"
    assert a["propuesta"] == ["Subdelegado del IMSS"] and a["opciones"] == ["Subdelegado del IMSS"]
    # Fijas primero; «jurídico» del modelo no se repite con otra capitalización.
    assert i["opciones"] == ["Jurídico", "Legítimo", "Otro"]
    assert n["propuesta"] is None and n["origen"] == "ninguno"
    assert r["pensamiento"] == ["una sola frase, no lista"]
    assert r["pedir_documento"] is None


def test_propuesta_fuera_de_opciones_se_agrega():
    r = normalizar({"campos": [{"id": "interes", "propuesta": "Legítimo e individual"}]}, CAMPOS)
    assert r["campos"][2]["opciones"][0] == "Legítimo e individual"


def test_mensaje_lleva_lo_confirmado_y_recorta():
    req = DeducirRequest(
        flujo="amparo-indirecto", entrega="Demanda de amparo indirecto", parte="Datos",
        campos=CAMPOS, conocido={"quejoso": "María López"}, encargo="x" * 20,
        carpeta="y" * 100, documento=None, estado="QUERETARO",
    )
    m = fa._mensaje_usuario(req)
    assert "María López" in m and "QUERETARO" in m and "CARPETA DEL ASUNTO" in m
    assert "LO QUE YA ESTÁ REDACTADO" not in m


def test_freno_por_usuario():
    tope = fa.FLUJO_TOPE
    fa.FLUJO_TOPE = 2
    try:
        fa._llamadas.clear()
        fa._frenar("u1")
        fa._frenar("u1")
        try:
            fa._frenar("u1")
            raise AssertionError("la tercera llamada debió frenarse")
        except HTTPException as e:
            assert e.status_code == 429
        fa._frenar("u2")  # otro usuario no se ve afectado
    finally:
        fa.FLUJO_TOPE = tope


def test_endpoint_sigue_si_el_modelo_falla():
    class Roto:
        class chat:
            class completions:
                @staticmethod
                async def create(**kw):
                    raise RuntimeError("caído")

    async def _uid(_):
        return "usuario-prueba"

    antes = (fa._openai, fa._usuario)
    fa._openai, fa._usuario = (lambda: Roto), _uid
    try:
        fa._llamadas.clear()
        req = DeducirRequest(flujo="f", entrega="E", parte="P", campos=CAMPOS)
        r = asyncio.run(fa.flujo_deducir(req, None))
        assert r["aviso"] and len(r["campos"]) == 4
    finally:
        fa._openai, fa._usuario = antes


if __name__ == "__main__":
    fallos = []
    for nombre, prueba in list(globals().items()):
        if nombre.startswith("test_") and callable(prueba):
            try:
                prueba()
                print(f"  ✓ {nombre}")
            except Exception as e:  # noqa: BLE001
                fallos.append(nombre)
                print(f"  ✗ {nombre}: {type(e).__name__}: {e}")
    print()
    if fallos:
        print(f"FALLAN {len(fallos)}: " + " · ".join(fallos))
        raise SystemExit(1)
    print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
