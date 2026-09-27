"""Pruebas del esfuerzo de redacción — sin red. `python test_esfuerzo_redaccion.py`."""
import json

from esfuerzo_redaccion import (
    OFERTA_TRAS_REVISION, _sin_marcadores, detectar_redaccion,
    es_ajuste_de_escrito, esfuerzo_permitido, normalizar_esfuerzo,
    parece_escrito, pide_escrito,
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
    # 27-sep-2026: frases de abogado que se atendían como consulta.
    "Quiero que me ayudes con la demanda de divorcio",
    "Necesito que me ayudes con los agravios contra la sentencia",
    "Quisiera que me ayudes con el amparo contra la clausura",
    "Quiero que me ayudes a hacer la contestación de la demanda",
    "Necesito que me ayudes con la redacción de la demanda laboral",
    "Contesta la demanda que te pasé",
    "Sí, contesta la demanda negando los hechos 3 y 4",
    "Contéstame la vista que me dieron sobre las pruebas",
    "Te pasé la demanda. Contesta la demanda oponiendo la prescripción",
    "Hazlo en formato de demanda",
    "Ahora hazlo en formato de demanda de amparo",
    "Pásalo a escrito para presentarlo al juez",
    "Conviértelo en un amparo indirecto",
    "Dámelo en formato de demanda",
    "Hazlo un escrito de agravios",
    "¿Puedes pasarlo a formato de demanda?",
    "¿Podrías convertirlo en una demanda de amparo?",
    "¿Podrías contestar la demanda con los datos que te di?",
    "Redacta un esqueleto argumentativo para cambiar el sentido del proyecto",
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
    # Las vecinas de las frases nuevas, que tienen que seguir siendo consulta.
    "Quiero que me ayudes a entender la demanda",
    "Necesito que me ayudes con una duda sobre la demanda",
    "Necesito que me ayudes con el plazo del amparo",
    "Quiero que me ayudes con la demanda, ¿qué requisitos lleva?",
    "Quiero que me ayudes con la jurisprudencia sobre alimentos",
    "¿Me ayudas con la demanda?",
    "Contesta mi pregunta sobre el amparo",
    "Contesta brevemente",
    "Si contesta la demanda fuera de plazo, ¿se tiene por no contestada?",
    "si contesta la demanda tarde que pasa",
    "Hola, si contesta la demanda extemporáneamente qué efectos tiene",
    "Contesta: ¿cuándo prescribe la acción cambiaria?",
    "¿El demandado contesta la demanda en nueve días?",
    "Mi contraparte ya contestó la demanda, ¿qué sigue?",
    "Ponlo en una tabla",
    "Hazlo más corto",
    "Hazlo un poco más largo",
    "¿Puedes pasarme la jurisprudencia de la reforma?",
    "¿Puedes ponerlo en una tabla?",
    "¿Puedes poner ejemplos de agravios?",
    "¿Puedes pasar ejemplos de demandas?",
    "¿Puedes convertir la sentencia en un resumen?",
    "¿Puedes transformar la sentencia en un resumen?",
    "Contesta la pregunta del amparo",
    "¿Puedes contestar la pregunta del amparo?",
    # El paso «Revisar» de Toulmin pega el borrador entero detrás de la orden.
    "Revisa este borrador de contestación de demanda y dime qué fundamentos le faltan.\n\n"
    "BORRADOR:\n" + "Antecedentes del juicio. " * 40
    + "El demandado fue emplazado el tres de marzo y contesta la demanda oponiendo excepciones.",
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

    # EL HISTORIAL REAL (27-sep-2026): la respuesta vuelve con el mapa de
    # fuentes AL FINAL —decenas de miles de caracteres— y a veces con el
    # razonamiento al principio. La oferta y el escrito tienen que verse igual.
    meta = "\n\n<!-- CITATION_META:" + json.dumps({"sources": {
        f"{i:08d}-0000-4000-8000-000000000000": {
            "origen": "Ley de Amparo", "ref": f"Art. {i}", "texto": "Texto del artículo. " * 40}
        for i in range(12)}}, ensure_ascii=False) + " -->"
    fuera = "\n<!--REGISTROS_FUERA:2021234,2019876-->"
    pensar = "<!--THINKING_START-->Reviso el plazo y ofrezco el escrito.<!--THINKING_END-->"
    for previa in (oferta + meta, oferta + meta + fuera, pensar + oferta + meta,
                   oferta + meta[:4000]):   # el mapa cortado a la mitad, sin cierre
        assert detectar_redaccion([
            {"role": "user", "content": "¿Cuál es el plazo del amparo directo?"},
            {"role": "assistant", "content": previa},
            {"role": "user", "content": "Sí, por favor"}]) == "acepta", previa[-60:]
    assert detectar_redaccion([
        {"role": "user", "content": "Redacta una demanda de amparo contra la clausura"},
        {"role": "assistant", "content": pensar + ESCRITO + meta + fuera},
        {"role": "user", "content": "Agrega un concepto de violación sobre audiencia previa"},
    ]) == "ajuste"
    assert parece_escrito(pensar + "SEXTO. Estudio. Son fundados los conceptos…" + meta)
    assert not parece_escrito(pensar + CONSULTA_RESPUESTA + meta)
    assert _sin_marcadores("a<!-- x -->b<!--THINKING_START-->c<!--THINKING_END-->d") == "abd"
    assert _sin_marcadores("sin marcas") == "sin marcas"
    assert _sin_marcadores("texto<!-- sin cierre") == "texto"
    assert _sin_marcadores("") == ""

    # La despedida de la revisión de sentencia: la de hoy y la que quedó en las
    # conversaciones guardadas antes del 27-sep-2026.
    dictamen = "## VII. CONCLUSIONES\n\n1) **Calificación Global**: REQUIERE CAMBIO DE SENTIDO.\n\n"
    vieja = ("¿Quieres que redacte un esqueleto argumentativo para fortalecer o cambiar el "
             "proyecto? Si es así, selecciona el Genio de la Materia que corresponda a este "
             "caso, activa el modo 'Redacción Especializada', envíame un mensaje con un "
             "simple 'ok' y yo me encargaré del resto.")
    for cierre in (OFERTA_TRAS_REVISION, vieja):
        for si in ("sí", "Sí", "ok", "Ok, adelante"):
            assert detectar_redaccion([
                {"role": "user", "content": "AUDITAR_SENTENCIA …"},
                {"role": "assistant", "content": f'{dictamen}**"{cierre}"**{meta}'},
                {"role": "user", "content": si}]) == "acepta", (cierre[:30], si)
        assert detectar_redaccion([
            {"role": "user", "content": "AUDITAR_SENTENCIA …"},
            {"role": "assistant", "content": f'{dictamen}**"{cierre}"**{meta}'},
            {"role": "user", "content": "¿Por qué dices que el sentido es injusto?"}]) == ""

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
