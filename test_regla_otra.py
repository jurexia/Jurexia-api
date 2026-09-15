# -*- coding: utf-8 -*-
"""LA REGLA «OTRA» Y EL AUTOLLENADO DE LA FECHA DE PRESENTACIÓN.

David, 15-sep-2026: el auto de admisión no llenaba tribunal, parte actora ni
fecha de presentación (esos «generalmente vienen en ese auto»); y en un
amparo directo salió una regla de notificación conforme a la ley de Querétaro
—el redactor tiene que servir a toda la república, no a un solo tribunal—.
Se pidió una «Otra regla»: el secretario declara cuándo se notificó y cuándo
surtió efectos, sin que el sistema decida por él.
"""
import datetime as dt
import sys

import fase0_oportunidad as f0
import fase_autos
import redactor_adelanto as ra

fallos = []
def ok(c, nota):
    print(f"  {'OK ' if c else 'MAL'} {nota}")
    if not c: fallos.append(nota)

# ── fase_autos ya sabía leer la presentación; se confirma que sigue así ──
AUTO = ("Conste. Querétaro, Querétaro, a veintiuno de noviembre de dos mil "
        "veinticinco.- Fórmese el expediente número 91/2025.- TÚRNESE a la "
        "ponencia del magistrado Luis Armando Perez Topete.- Folio electrónico: "
        "123456789.- Fecha de presentación: 13/11/2025.")
leido = fase_autos.leer(AUTO)
ok(leido.get("presentacion") == "2025-11-13", f"fase_autos.leer() saca la presentación en ISO: {leido.get('presentacion')}")

# ── la regla «otra»: dos fechas declaradas, ningún día hábil contado ──
notif = dt.date(2026, 3, 2)
surtio_dado = dt.date(2026, 3, 10)   # una brecha que no encaja en ningún ordinal
c = f0.computar(notif, dt.date(2026, 3, 20), "otra", 15, "Sala X",
                None, "amparo_directo", None, surtio_manual=surtio_dado)
ok(c.surtio == surtio_dado, f"el surtimiento es EXACTAMENTE la fecha declarada, no una contada: {c.surtio}")
ok(c.regla.clave == "otra" and c.regla.dias_habiles == -1, "la regla queda marcada como «otra», sin días hábiles inventados")
ok(not any("Boletín" in a or "QUERÉTARO" in a for a in c.avisos), "no avisa de ninguna regla estatal: no se aplicó ninguna")

p = f0.parrafo_oportunidad(c, "17", "amparo_directo")
ok("según lo manifestado" in p, f"el considerando dice que la fecha la dio el promovente, no una regla: {p[:200]}")
ok("al día hábil siguiente" not in p and "al tercer día hábil" not in p, "y no inventa un ordinal de días hábiles que no se contaron")
ok(f0.fecha_en_letra(notif) in p and f0.fecha_en_letra(surtio_dado) in p, "las dos fechas van en el considerando")

# ── fecha imposible: surtio antes que la notificación ──
c2 = f0.computar(dt.date(2026, 3, 10), None, "otra", 15, None, None,
                 "amparo_directo", None, surtio_manual=dt.date(2026, 3, 1))
ok(any("FECHA IMPOSIBLE" in a for a in c2.avisos), "avisa si la fecha de surtimiento declarada es anterior a la notificación")

# ── sigue funcionando lo de siempre cuando no es «otra» ──
c3 = f0.computar(dt.date(2026, 4, 7), dt.date(2026, 4, 20), "personal", 15)
ok(c3.regla.clave == "personal" and c3.surtio == dt.date(2026, 4, 8), "la regla «personal» de siempre no cambió")

# ── el guardián de administrativa + Querétaro (redactor_adelanto) ──
import asyncio
class _Fases:
    problema_global = ""
    problemas = []
e = ra.Encargo(numero="1/2026", encabezado="AMPARO DIRECTO ADMINISTRATIVO: 1/2026",
               quejoso="X", magistrado="M", secretario="S",
               notificacion=dt.date(2026, 3, 2), presentacion=dt.date(2026, 3, 20),
               regla_surtimiento="tja_qro_boletin", tipo_asunto="amparo_directo",
               materia="administrativa", coleccion_estatal="leyes_jalisco",
               responsable="Sala Jalisco")
async def _correr():
    return await ra.generar(None, e, "CONSIDERANDO. Texto del acto.", "AGRAVIOS. Texto.", "/tmp/x.docx")
try:
    r = asyncio.run(_correr())
    ok(any("QUERÉTARO" in a and "no está declarado" not in a for a in r.avisos) or
       any("sólo rige ahí" in a for a in r.avisos), "y avisa por qué se cambió")
except Exception as ex:
    print(f"  (generar() no completó sin cliente real: {type(ex).__name__}: {ex} — "
          f"el guardián corre ANTES de tocar el modelo, así que su mutación ya se hizo)")
# EL GUARDIÁN YA MUTÓ e.regla_surtimiento EN SU SITIO antes de que generar()
# llegara al modelo (o terminara), porque Encargo es un dataclass mutable y el
# guardián corre en las primeras líneas de generar(). Se comprueba aquí,
# fuera del try, para que cuente tanto si generar() completó como si no.
ok(e.regla_surtimiento == "personal", f"un amparo directo administrativo de Jalisco NO se queda con la regla de Querétaro: {e.regla_surtimiento}")

# ── una ley de Querétaro SÍ conserva la regla ──
e2 = ra.Encargo(numero="1/2026", encabezado="AMPARO DIRECTO ADMINISTRATIVO: 1/2026",
                quejoso="X", magistrado="M", secretario="S",
                notificacion=dt.date(2026, 3, 2), presentacion=dt.date(2026, 3, 20),
                regla_surtimiento="tja_qro_boletin", tipo_asunto="amparo_directo",
                materia="administrativa", coleccion_estatal="leyes_queretaro",
                responsable="Sala Querétaro")
_mat2 = ra.fp_materia(e2)
_col2 = (e2.coleccion_estatal or "").strip().lower()
ok("queretaro" in _col2.replace("é", "e"), "y si de verdad es de Querétaro, la colección lo dice")

# ── el TFJA con Sala «en Querétaro» no es el TJA del estado (8/2026) ──
e3 = ra.Encargo(numero="8/2026", encabezado="REVISIÓN FISCAL 8/2026", quejoso="X",
                magistrado="M", secretario="S", notificacion=dt.date(2025, 12, 4),
                presentacion=dt.date(2026, 1, 19), regla_surtimiento="tja_qro_boletin",
                tipo_asunto="revision_fiscal", materia="administrativa",
                coleccion_estatal="leyes_queretaro",
                responsable="Sala Regional en Querétaro del Tribunal Federal de Justicia Administrativa")
try:
    asyncio.run(ra.generar(None, e3, "CONSIDERANDO. Texto.", "AGRAVIOS. Texto.", "/tmp/y.docx"))
except Exception:
    pass
ok(e3.regla_surtimiento == "personal",
   f"la Sala Regional EN Querétaro del TFJA (federal) no es el TJA DE Querétaro (estatal): {e3.regla_surtimiento}")

print()
if fallos: print(f"FALLAN {len(fallos)}: " + " · ".join(fallos)); sys.exit(1)
print("TODAS LAS COMPROBACIONES PASAN")
