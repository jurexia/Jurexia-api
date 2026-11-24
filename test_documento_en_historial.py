"""El documento en la conversación, por plan — 23-nov-2026.

    .venv/bin/python test_documento_en_historial.py

Sin red ni API. El caso: un Platinum subió un expediente que, leído entero,
da ~300 mil caracteres. El texto viajaba a sus preguntas de seguimiento con
un tope de 120 mil para todos los planes: el seguimiento veía el 40%. Y si la
conversación crecía, el documento de en medio bajaba a 3 mil caracteres.

Se comprueba:
  1. el tope por plan (Platinum 300 mil, Pro 200 mil, el resto 120 mil) y
     que nunca pasa de la mitad de la memoria del plan;
  2. el documento que ABRE una conversación larga conserva su tope al
     recortarse el historial (antes, el ancla bajaba a 120 mil);
  3. el documento que va EN MEDIO toma lo que quede, no 3 mil;
  4. el recorte nunca se pasa del presupuesto, y sin documento todo sigue
     igual que antes.
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import main  # noqa: E402

fallos = []


def ok(cond, nombre):
    print(("  ✓ " if cond else "  ✗ ") + nombre)
    if not cond:
        fallos.append(nombre)


def doc(chars: int) -> main.Message:
    cuerpo = ("Hoja del expediente con la resolución del no ejercicio de la acción penal. " * (chars // 75 + 1))[:chars]
    return main.Message(role="user", content=(
        "📄 **Documento adjunto:** EXPEDIENTE.pdf\n\n¿Procede el recurso?\n\n<!-- DOCUMENTO_INICIO -->\n"
        "CONTENIDO DEL DOCUMENTO ADJUNTO «EXPEDIENTE.pdf» (texto leído por Iurexia):\n\n"
        + cuerpo + "\n<!-- DOCUMENTO_FIN -->"))


def turnos(n: int, largo: int) -> list:
    out = []
    for i in range(n):
        out.append(main.Message(role="user", content=f"Pregunta {i}: " + "y además " * (largo // 20)))
        out.append(main.Message(role="assistant", content=f"Respuesta {i}: " + "conforme al artículo " * (largo // 40)))
    return out


print("1. El tope por plan")
T = main.tope_documento_en_historial
ok(T("platinum") == 300_000, f"Platinum 300 mil ({T('platinum'):,})")
ok(T("pro") == 200_000, f"Pro 200 mil ({T('pro'):,})")
ok(T("basico") == 120_000 and T("gratuito") == 120_000, "Básico y Gratuito, los 120 mil de siempre")
ok(all(T(n) <= max(120_000, main.tope_de_historial(n) // 2) for n in ("gratuito", "basico", "pro", "platinum")),
   "nunca más de la mitad de la memoria del plan (salvo el tope de siempre)")
ok(T("desconocido") == 120_000, "un plan que no se conoce, el tope de siempre")

print("\n2. El documento que abre una conversación larga")
tope = main.tope_de_historial("platinum")
conv = [doc(300_000), main.Message(role="assistant", content="Análisis. " * 2000)] + turnos(14, 24_000)
antes_total = sum(len(m.content) for m in conv)
h, antes, despues = main._recortar_historial(conv, presupuesto=tope, ancla_max=T("platinum"))
ok(antes_total > tope, f"la conversación no cabe ({antes_total:,} > {tope:,})")
ok(despues <= tope, f"el recorte no se pasa ({despues:,} ≤ {tope:,})")
ok(len(h[0].content) > 250_000, f"el documento conserva lo suyo ({len(h[0].content):,} car.)")
h_viejo, _, _ = main._recortar_historial(conv, presupuesto=tope)
ok(len(h_viejo[0].content) <= main.HISTORIAL_ANCLA_MAX, "sin el tope del plan, el ancla baja a 120 mil como antes")
ok(h[-1].content == conv[-1].content, "la última pregunta entra entera")

print("\n3. El documento a mitad de la conversación")
conv2 = turnos(3, 20_000) + [doc(300_000), main.Message(role="assistant", content="Análisis. " * 2000)] + turnos(12, 24_000)
h2, a2, d2 = main._recortar_historial(conv2, presupuesto=tope, ancla_max=T("platinum"))
i_doc = next(i for i, m in enumerate(conv2) if "DOCUMENTO_INICIO" in m.content)
ok(d2 <= tope, f"no se pasa ({d2:,} ≤ {tope:,})")
ok(len(h2[i_doc].content) > 30_000,
   f"toma lo que queda, no la cabeza de 3 mil ({len(h2[i_doc].content):,} car.)")
ok("DOCUMENTO_INICIO" in h2[i_doc].content and "Hoja del expediente" in h2[i_doc].content,
   "y empieza por el documento, no sólo por la pregunta")

print("\n4. Sin documento, como antes")
conv3 = turnos(30, 24_000)
h3, a3, d3 = main._recortar_historial(conv3, presupuesto=tope, ancla_max=T("platinum"))
h3v, a3v, d3v = main._recortar_historial(conv3, presupuesto=tope)
ok(d3 <= tope, "no se pasa")
ok([m.content for m in h3[1:]] == [m.content for m in h3v[1:]],
   "los turnos sin documento salen idénticos con y sin el tope nuevo")
chica = turnos(3, 2_000)
h4, a4, d4 = main._recortar_historial(chica, presupuesto=tope, ancla_max=T("platinum"))
ok(h4 is chica or [m.content for m in h4] == [m.content for m in chica], "lo que cabe no se toca")

print()
print("TODO PASA" if not fallos else f"{len(fallos)} FALLAS")
sys.exit(1 if fallos else 0)
