"""La violación procesal se examina sobre la resolución que la decidió — ADC
93/2026, 22-sep-2026.

CALIBRADO contra los dos documentos reales del asunto —la sentencia
definitiva y la demanda de amparo—, que MENCIONAN la reclamación y la
preclusión sin ser la resolución del incidente. La primera versión, por
señales sueltas, las clasificaba como tal: eso habría mandado al modelo
confrontar la demanda de la quejosa como razón toral.

    .venv/bin/python test_violacion_procesal.py
"""
import inspect
import json
import os

import violacion_procesal as vp

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


RESUMEN_DAVID = (
    "La reclamación se declaró infundada porque el juicio se tramitó en la vía "
    "sumaria y el plazo para ampliar la demanda es el de cinco días del artículo "
    "58-6 de la LFPCA, que corrió del 5 al 12 de agosto de 2025; la ampliación se "
    "presentó el 14, por lo que fue extemporánea y se confirmó el auto de 13 de "
    "agosto que tuvo por precluido el derecho.")
INTERLOCUTORIA = (
    "SALA REGIONAL EN QUERÉTARO. Expediente 765/25-09-01-8-ST. Recurso de "
    "reclamación. Visto para resolver el recurso de reclamación interpuesto por la "
    "actora en contra del auto de trece de agosto de dos mil veinticinco. "
    "CONSIDERANDO. SEGUNDO. Los agravios son infundados. La recurrente sostiene que "
    "el acuerdo de primero de julio le concedió el plazo del artículo 17; sin "
    "embargo, el juicio se tramita en la vía sumaria y conforme al artículo 58-6 el "
    "plazo para ampliar la demanda es de cinco días, sin que sea obligación del "
    "Magistrado instructor otorgarlo expresamente, siempre que se respete. El plazo "
    "transcurrió del cinco al doce de agosto y el escrito se presentó el catorce, "
    "por lo que fue extemporáneo. Por lo expuesto se resuelve: ÚNICO. Es infundado "
    "el recurso de reclamación y se confirma el auto recurrido.")
ACUERDO = (
    "Querétaro, a trece de agosto de dos mil veinticinco. Visto el escrito de la "
    "actora por el que pretende ampliar la demanda, y toda vez que el plazo de cinco "
    "días previsto en el artículo 58-6 de la Ley Federal de Procedimiento Contencioso "
    "Administrativo transcurrió del cinco al doce de agosto, se tiene por precluido su "
    "derecho para ampliar la demanda y se desecha la ampliación por extemporánea.")
CONTRATO = ("CLÁUSULA 64. El patrón se obliga a cubrir a los trabajadores sindicalizados "
            "una prima de antigüedad equivalente a doce días de salario por cada año de "
            "servicios, conforme al contrato colectivo de trabajo vigente.")
PERITAJE = ("DICTAMEN PERICIAL EN MATERIA DE GRAFOSCOPÍA. Con base en el cotejo de las firmas "
            "cuestionadas con las indubitables, el suscrito perito concluye que las firmas "
            "que calzan el mandamiento de ejecución no provienen del puño y letra del funcionario.")
LEY = ("Artículo 17. Se podrá ampliar la demanda, dentro de los diez días siguientes a aquél "
       "en que surta efectos la notificación del acuerdo que admita su contestación, en los "
       "siguientes casos: I. Cuando se impugne una negativa ficta.")
DEMANDA = ("H. TRIBUNAL COLEGIADO. PRESENTE. La suscrita, en mi carácter de representante "
           "legal, vengo a interponer demanda de amparo directo. Conceptos de violación. "
           "PRIMERO. La Sala responsable consideró que no ejercí mi derecho a ampliar la "
           "demanda, lo que es ilegal porque el recurso de reclamación se declaró infundado "
           "sin atender que el acuerdo no me otorgó plazo expresamente. Solicito se conceda el amparo.")

print("\n1 · LO QUE ES LA RESOLUCIÓN DEL INCIDENTE")
ok(vp.clasificar(RESUMEN_DAVID)[0] == vp.RESOLUCION_PROCESAL, "los motivos de la reclamación, en palabras del secretario")
ok(vp.clasificar(INTERLOCUTORIA)[0] == vp.RESOLUCION_PROCESAL, "la interlocutoria entera")
ok(vp.clasificar(ACUERDO)[0] == vp.RESOLUCION_PROCESAL, "el acuerdo de preclusión mismo")

print("\n2 · LO QUE SÓLO LA MENCIONA, O NO ES ELLA")
ok(vp.clasificar(CONTRATO)[0] == vp.CONSTANCIA, "una cláusula del contrato colectivo: constancia")
ok(vp.clasificar(PERITAJE)[0] == vp.CONSTANCIA, "un peritaje: constancia")
ok(vp.clasificar(LEY)[0] == vp.OTRO, "un artículo de ley: otro")
ok(vp.clasificar(DEMANDA)[0] != vp.RESOLUCION_PROCESAL, "una demanda que RELATA la reclamación: no es la resolución")
ok(vp.clasificar("")[0] == vp.OTRO, "vacío: otro")

_S = ("/private/tmp/claude-501/-Users-josedavidalcantarmendoza-Documents-Viaje-a-Europa/"
      "5f71a5c8-bc09-427e-90e2-108495d0f272/scratchpad/93/")
if os.path.exists(_S + "fuente0.txt") and os.path.exists(_S + "fuente1.txt"):
    print("\n2b · LOS DOS DOCUMENTOS REALES DEL 93/2026")
    ok(vp.clasificar(open(_S + "fuente0.txt").read())[0] != vp.RESOLUCION_PROCESAL,
       "la sentencia definitiva (75k) no es la resolución del incidente")
    ok(vp.clasificar(open(_S + "fuente1.txt").read())[0] != vp.RESOLUCION_PROCESAL,
       "la demanda de amparo (62k) tampoco")

print("\n3 · CUÁL PROBLEMA ES LA VIOLACIÓN PROCESAL")
P1 = {"pregunta": "¿La Sala debió admitir la ampliación de demanda y estudiar los argumentos dirigidos contra el crédito fiscal?",
      "combate": "sostiene que ejerció oportunamente el derecho de ampliar la demanda y que el acuerdo que tuvo por precluido ese derecho fue ilegal"}
P2 = {"pregunta": "¿La Sala debía pronunciarse sobre los argumentos formulados por la actora en sus alegatos?",
      "combate": "omitió estudiar seis argumentos de los alegatos"}
ok(vp.es_problema_procesal(P1), "el del 93/2026 (no dice «violación procesal» por ningún lado)")
ok(not vp.es_problema_procesal(P2), "el de los alegatos no lo es")
ok(vp.es_problema_procesal({"pregunta": "¿…?", "clase": "procesal"}), "la fase 3 lo declara con `clase`")
ok(not vp.es_problema_procesal({"pregunta": "¿debió admitir la ampliación de demanda?", "clase": "fondo"}),
   "y si la fase 3 dice «fondo», manda la fase 3")
ok(vp.hay([P2], [], RESUMEN_DAVID), "con la interlocutoria aportada, hay violación procesal aunque el problema no lo diga")
ok(not vp.hay([P2], [], CONTRATO), "con un contrato aportado y un problema de fondo, no la hay")

print("\n4 · EL BLOQUE DEL PROMPT")
b = vp.bloque(RESUMEN_DAVID, para="estudio")
ok("RESOLUCIÓN QUE DECIDIÓ LA VIOLACIÓN PROCESAL" in b and "171 y 172" in b, "rótulo y fundamento")
# LA TÉCNICA APROBADA EL 26-SEP-2026 (decisión 2). La vieja —«una por una»,
# «PROHIBIDO despacharla en un párrafo»— fabricaba cuatro párrafos con la misma
# respuesta; la nueva identifica y confronta cada razón, junta las que caen por
# lo mismo, deja caer la dependiente salvo que un efecto la presuponga y parte
# la mixta.
ok("IDENTIFICA las razones" in b and "CONFRONTA cada razón" in b and "Ninguna se queda" in b,
   "la técnica: identificar y confrontar cada razón, sin dejar ninguna sin respuesta")
ok("SE CONTESTAN JUNTAS" in b and "nombrándolas a todas" in b,
   "las que caen por la misma respuesta se contestan juntas, nombrándolas")
ok("DEPENDE DE OTRA CAE CON ELLA" in b and "SALVO QUE UN EFECTO DE LA CONCESIÓN LA PRESUPONGA" in b
   and "ADVERTENCIAS" in b, "la dependiente cae con la otra salvo que un efecto la presuponga")
ok("RAZÓN MIXTA SE PARTE" in b, "la razón mixta se parte")
ok("una por una" not in b and "PROHIBIDO despacharla" not in b,
   "y ya no manda un párrafo por razón")
ok("esa consideración no" not in b, "sin la frase entre comillas que se copiaba (lección del ejemplo)")
ok("nunca como\n     «documento aportado»" in b, "y no se cita como «documento aportado»")
bp = vp.bloque(RESUMEN_DAVID, para="propuesta")
ok("razon_toral" in bp, "en la propuesta, la razón toral es la de esta resolución")
bc = vp.bloque(CONTRATO)
ok("CONSTANCIA DE AUTOS" in bc and "171" not in bc, "una constancia sigue siendo una constancia")

print("\n5 · LAS PUERTAS ESTÁN CONECTADAS")
import fase5_propuesta as f5, fase6_estudio as f6, redactor_adelanto as ra, tipos_asunto as ta
ok("violacion_procesal" in inspect.getsource(f5._bloque_contexto), "la propuesta clasifica lo aportado")
ok("violacion_procesal" in inspect.getsource(f6._bloque_aportado), "el estudio también")
ok(inspect.getsource(ra).count("_vpm.hay(") == 2, "los dos redactores del estudio detectan la violación por lo que se combate")
ok(any(x["fuente"].startswith("artículos 171, 172") for x in ta.tecnica_de("amparo_directo", "", True)),
   "la técnica de los artículos 171 y 172 entra al prompt")
src = open("main.py", encoding="utf-8").read()
ok('"clase": _clase_ctx' in src, "/taller/contexto dice qué es lo que llegó")

print("\n6 · LA MISMA TÉCNICA EN EL CATÁLOGO (tipos_asunto), Y EL ORDEN DEL 189")
_tvp = " ".join(ta.TECNICA_RESOLUCION["directo_violacion_procesal"]["tecnica"])
ok("se identifican las razones" in _tvp and "SE CONTESTAN JUNTAS" in _tvp
   and "LA RAZÓN QUE DEPENDE DE OTRA CAE CON ELLA" in _tvp and "LA RAZÓN MIXTA SE PARTE" in _tvp,
   "las cuatro piezas de la regla también en tipos_asunto")
ok("una por una" not in _tvp and "PROHIBIDO despacharla" not in _tvp,
   "y tampoco ahí manda un párrafo por razón")
ok("se deciden todas" in _tvp and "74, fracción V, y 174" in _tvp,
   "si prospera, las demás procesales no quedan sin materia: se deciden todas")
# Revisión del 26-sep-2026: las dos técnicas llegan JUNTAS al estudio y ésta
# mandaba todo el fondo a «sin materia» sin la salvedad del 189 que la del
# orden sí dice.
_si_pros = next(x for x in ta.TECNICA_RESOLUCION["directo_violacion_procesal"]["tecnica"]
                if x.startswith("SI PROSPERA"))
ok("beneficio mayor que la reposición" in _si_pros and "189" in _si_pros,
   "y el fondo que da más que reponer no queda sin materia: se estudia antes (art. 189)")
_tor = " ".join(ta.TECNICA_RESOLUCION["directo_orden_de_estudio"]["tecnica"])
ok("SON DE ESTUDIO PREFERENTE" not in _tor,
   "el orden ya no dice que las procesales son de estudio preferente")
ok("EL FONDO VA PRIMERO" in _tor and "MAYOR BENEFICIO" in _tor,
   "art. 189 vigente: el fondo primero, salvo mayor beneficio")
ok("189" in ta.TECNICA_RESOLUCION["directo_orden_de_estudio"]["fuente"]
   and "174" in ta.TECNICA_RESOLUCION["directo_orden_de_estudio"]["fuente"],
   "con su fundamento: 74-V, 174 y 189")
_la = json.load(open("normas_ley_de_amparo.json", encoding="utf-8"))["articulos"]["189"]
ok("se privilegiará el estudio de los conceptos de violación de fondo por encima de los de "
   "procedimiento y forma" in _la,
   "y la regla sale del texto vigente del 189, que está en el repositorio")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
