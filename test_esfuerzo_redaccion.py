"""Pruebas del esfuerzo de redacción — sin red. `python test_esfuerzo_redaccion.py`."""
from esfuerzo_redaccion import (
    detectar_redaccion, es_ajuste_de_escrito, esfuerzo_permitido,
    normalizar_esfuerzo, parece_escrito, pide_escrito,
)

ENCARGOS = [
    "Redacta una demanda de amparo indirecto contra la orden de clausura del municipio",
    "redáctame los agravios contra la sentencia de primera instancia",
    "Redacta:\nuna contestación de demanda en materia mercantil",
    "Por favor, redacta un escrito de ofrecimiento de pruebas",
    "Hola, necesito que me redactes una demanda de divorcio incausado en Querétaro",
    "¿Me puedes redactar un recurso de revisión?",
    "¿Puedes elaborar una demanda laboral por despido injustificado?",
    "Hazme una demanda de pensión alimenticia con estos hechos: ...",
    "Elabora un contrato de arrendamiento de casa habitación",
    "Prepara los conceptos de violación para un amparo directo",
    "Necesito una demanda de nulidad contra un crédito fiscal del SAT",
    "Quiero un escrito de desistimiento",
    "Formula agravios contra el auto que desechó la prueba pericial",
    "Proyecta el estudio de fondo del amparo en revisión 123/2026",
    "Con base en lo anterior, redacta la demanda completa",
    "Genera argumentos para sostener la prescripción de la acción",
    "Escribe la demanda de amparo con los datos que te di",
    "Redacción de una denuncia penal por fraude",
    "Ayúdame a redactar una queja contra el acuerdo que negó la suspensión",
    "Vamos a redactar el recurso de apelación",
    "Ayúdame a preparar la contestación de la demanda",
    "Ayúdame con la demanda de amparo",
    "Necesito que me hagas un escrito de alegatos",
    "Necesito ayuda para redactar una demanda de alimentos",
    "Ahora sí elabora la queja pero señala al tribunal colegiado al inicio",
    "Ahora lee las últimas páginas de las quejas, unifica tu razonamiento y elabora la queja",
    "nesecito que me elabores una apelacion analizando los siguientes hechos",
    "Ahora, en relación con el contrato que te anexé, necesito que me hagas un contrato de comodato",
    "hasme un escrito porque no me han pagado desde marzo",
    "Dame el escrito sin explicaciones y con leyes aplicables de Veracruz",
    "¿Puedes realizar una contrarréplica de un escrito de réplica?",
    "Formato contestación de amparo indirecto con causal de improcedencia",
    "¿Cómo sería el escrito para presentarlo al juez?",
    "Hacer un contrato de obra a precio unitario de un edificio de 8 niveles",
    "¿Podrías reescribir el amparo?",
    "Elabora un interrogatorio en materia laboral para los testigos de la demandada",
]

CONSULTAS = [
    "¿Cómo se redacta una demanda de amparo?",
    "¿Cuál es el plazo para promover amparo directo?",
    "Cómo impugnar una multa de tránsito en Querétaro",
    "¿Qué alego si me negaron la suspensión?",
    "Necesito saber los requisitos de la demanda de amparo",
    "Escribe el artículo 17 de la Ley de Amparo",
    "Dame los requisitos de la demanda de divorcio",
    "Jurisprudencia sobre la suplencia de la queja en materia penal",
    "¿En qué plazo se redacta el engrose?",
    "El juez redactó mal la sentencia, ¿qué hago?",
    "Explícame la diferencia entre amparo directo e indirecto",
    "genera un resumen de la reforma judicial",
    "Genera un resumen de la sentencia que te pasé",
    "me puedes dar la lista de datos que te faltan para redactar el contrato",
    "Si se va a vender una moto a plazos, ¿qué contrato debo redactar?",
    "Soy víctima de lesiones dolosas. Estoy redactando un escrito y tengo dudas del plazo",
    "Deja de mover el cuadro donde el suscrito te redacta los mensajes",
    "Dame los requisitos de la demanda de divorcio en Jalisco",
    "Dame la jurisprudencia sobre el amparo contra leyes",
    "¿Puedes hacer un análisis de la sentencia?",
    "El juez elabora el acta de la audiencia sin firmas, ¿es válida?",
    "Tengo una demanda, es chiquita, ¿te la puedo compartir?",
    "Ayúdame a entender el juicio de amparo",
    "Vamos a analizar el recurso de revisión",
    "",
]

ESCRITO = """**C. JUEZ DE DISTRITO EN MATERIA ADMINISTRATIVA**

JUAN PÉREZ, por mi propio derecho…

**HECHOS**

1. El 3 de marzo…

**CONCEPTOS DE VIOLACIÓN**

PRIMERO. …

**PUNTOS PETITORIOS**

PRIMERO. Tenerme por presentado…"""

CONSULTA_RESPUESTA = """El plazo para promover el amparo directo es de quince días, conforme al
artículo 17 de la Ley de Amparo. La jurisprudencia 1a./J. 25/2017 precisa que…"""


def main() -> None:
    fallos = []
    for m in ENCARGOS:
        if not pide_escrito(m):
            fallos.append(f"debió ser ENCARGO: {m!r}")
    for m in CONSULTAS:
        if pide_escrito(m):
            fallos.append(f"debió ser CONSULTA: {m!r}")

    assert parece_escrito(ESCRITO), "el escrito no se reconoce"
    assert parece_escrito("SEXTO. Estudio. Son fundados los conceptos…")
    assert not parece_escrito(CONSULTA_RESPUESTA), "una consulta pasa por escrito"

    for m in ("Agrega un concepto de violación sobre la falta de fundamentación",
              "hazlo más extenso", "Corrige el proemio: el quejoso es persona moral",
              "Ahora, amplía los hechos", "Quita el tercer punto petitorio",
              "continúa"):
        assert es_ajuste_de_escrito(m), m
    for m in ("¿Por qué citaste el artículo 17?", "gracias", "Perfecto, gracias",
              "¿agrega algo la reforma?", "Explícame el segundo concepto"):
        assert not es_ajuste_de_escrito(m), m

    hist = [
        {"role": "system", "content": "CARPETA DE TRABAJO …"},
        {"role": "user", "content": "Redacta una demanda de amparo contra la clausura"},
        {"role": "assistant", "content": ESCRITO},
        {"role": "user", "content": "Agrega un concepto de violación sobre audiencia previa"},
    ]
    assert detectar_redaccion(hist) == "ajuste"
    hist[-1]["content"] = "¿Qué tesis respaldan el primer concepto?"
    assert detectar_redaccion(hist) == ""
    hist[2]["content"] = CONSULTA_RESPUESTA
    hist[-1]["content"] = "amplía la respuesta"
    assert detectar_redaccion(hist) == "", "tras una consulta, «amplía» es consulta"
    assert detectar_redaccion(hist[:2]) == "pide"
    assert detectar_redaccion([]) == ""
    oferta = CONSULTA_RESPUESTA + "\n\n¿Desea que redacte la demanda de amparo con estos datos?"
    assert detectar_redaccion([
        {"role": "user", "content": "¿Cuál es el plazo del amparo directo?"},
        {"role": "assistant", "content": oferta},
        {"role": "user", "content": "Sí, por favor"}]) == "acepta"
    assert detectar_redaccion([
        {"role": "user", "content": "¿Cuál es el plazo del amparo directo?"},
        {"role": "assistant", "content": CONSULTA_RESPUESTA},
        {"role": "user", "content": "Sí, por favor"}]) == "", "un «sí» sin oferta es consulta"
    for m in ("¿Me puedes volver a dar el contrato con los campos llenos?",):
        assert not es_ajuste_de_escrito(m), "una pregunta no es ajuste"
    assert es_ajuste_de_escrito("Me puedes volver a dar el contrato con los campos llenos")

    # El plan manda sobre el desplegable.
    assert esfuerzo_permitido("platinum", "platinum_monthly") == "platinum"
    assert esfuerzo_permitido("platinum", "pro_monthly") == "pro"
    assert esfuerzo_permitido("pro", "pro_annual") == "pro"
    assert esfuerzo_permitido("basico", "platinum_annual") == "basico"
    assert esfuerzo_permitido("platinum", "basico_monthly") == "basico"
    assert esfuerzo_permitido("pro", "gratuito") == "basico"
    assert esfuerzo_permitido("platinum", None) == "basico"
    assert esfuerzo_permitido("platinum", None, es_admin=True) == "platinum"
    assert esfuerzo_permitido(None, "platinum_monthly") == "basico"
    assert esfuerzo_permitido("platino", "ultra_secretarios") == "platinum"
    assert normalizar_esfuerzo("Profesional") == "basico"
    assert normalizar_esfuerzo("cualquiera") is None

    if fallos:
        print("\n".join(fallos))
        raise SystemExit(f"{len(fallos)} frase(s) mal clasificadas")
    print(f"OK · {len(ENCARGOS)} encargos, {len(CONSULTAS)} consultas, ajustes y planes")


if __name__ == "__main__":
    main()
