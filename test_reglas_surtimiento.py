"""Cuándo surte efectos: la ley que rige el acto, no Querétaro — ADC 93/2026, 22-sep-2026.

David metió a mano que la notificación surtió efectos el 10 de diciembre y el
cómputo salió extemporáneo: la reconstrucción de la sesión desde la base
rehacía el cómputo SIN esa fecha (personal → surtió el 8 → venció el 15 de
enero). Con la fecha, vence el 19 y la demanda del 19 está en tiempo. Y la
regla correcta para el TFJA ni siquiera exige declararla: artículo 65 LFPCA,
boletín al tercer día hábil.

    .venv/bin/python test_reglas_surtimiento.py
"""
import datetime as dt
import inspect

import fase0_oportunidad as f0
import redactor_adelanto as ra

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


N, P, S = dt.date(2025, 12, 5), dt.date(2026, 1, 19), dt.date(2025, 12, 10)
TFJA = "Sala Regional en Querétaro del Tribunal Federal de Justicia Administrativa"

print("\n1 · EL 93/2026 CON LA REGLA DE LA LFPCA")
c = f0.computar(N, P, "lfpca_boletin", 15, TFJA, None, "amparo_directo", None)
ok(c.surtio == S, f"publicado en boletín el 5 (viernes) → surte el 10: {c.surtio}")
ok(c.vencimiento == P and c.oportuna is True, f"vence el 19 de enero y la demanda del 19 está EN TIEMPO: {c.vencimiento} · {c.oportuna}")
par = f0.parrafo_oportunidad(c, "artículo 17 de la Ley de Amparo", "amparo_directo", desglosar=True)
ok("tercer día hábil siguiente" in par and "artículo 65 de la Ley Federal de Procedimiento Contencioso Administrativo" in par,
   "el considerando dice «al tercer día hábil siguiente, conforme al artículo 65…»")
ok("Boletín Jurisdiccional" in par, "y nombra el Boletín Jurisdiccional")

print("\n2 · LA FECHA DECLARADA A MANO SOBREVIVE A LA RECONSTRUCCIÓN")
e = ra.Encargo(numero="93/2026", encabezado="X", quejoso="X", magistrado="X", secretario="X",
               notificacion=N, presentacion=P, regla_surtimiento="otra", surte_efectos="2025-12-10",
               tipo_asunto="amparo_directo", materia="administrativa", responsable=TFJA)
fecha, aviso = f0.surtio_manual_de(e)
ok(fecha == S and not aviso, "surtio_manual_de lee la fecha del encargo")
c2 = f0.computar(N, P, "otra", 15, TFJA, None, "amparo_directo", None, surtio_manual=fecha)
ok(c2.oportuna is True and c2.vencimiento == P, "y con ella el cómputo da en tiempo")
e.surte_efectos = ""
ok(f0.surtio_manual_de(e)[0] is None and "no diste la fecha" in f0.surtio_manual_de(e)[1], "sin fecha: aviso, no invento")
e.regla_surtimiento = "personal"; e.surte_efectos = "2025-12-10"
ok(f0.surtio_manual_de(e) == (None, ""), "sólo cuenta con «otra regla»")
src = open("main.py", encoding="utf-8").read()
i = src.index("def _taller_recuperar_sesion")
ok("surtio_manual=_f0.surtio_manual_de(encargo)[0]" in src[i:i + 12000],
   "la reconstrucción desde la base pasa la fecha declarada (antes no)")
ok("f0.surtio_manual_de(e)" in inspect.getsource(ra.generar), "y el adelanto usa la MISMA puerta")

print("\n3 · QUÉ SE OFRECE, SEGÚN LA LEY DEL ACTO")
r = f0.reglas_para("amparo_directo", TFJA)
ok(r["fuero"] == "tfja" and r["por_omision"] == "lfpca_boletin", "TFJA → boletín de la LFPCA por omisión")
ok([x["clave"] for x in r["reglas"]] == ["lfpca_boletin", "lfpca", "otra"], "y sólo boletín / personal / otra")
r = f0.reglas_para("revision_fiscal", "")
ok(r["fuero"] == "tfja", "la revisión fiscal es del TFJA aunque no se diga la responsable")
r = f0.reglas_para("amparo_directo", "Tribunal de Justicia Administrativa del Estado de Querétaro")
ok(r["por_omision"] == "tja_qro_boletin", "TJA de Querétaro → su boletín")
r = f0.reglas_para("amparo_directo", "Sala Unitaria del Tribunal de Justicia Administrativa del Estado de Jalisco")
ok(r["por_omision"] == "otra" and "tja_qro_boletin" not in [x["clave"] for x in r["reglas"]],
   "otro TJA estatal → «otra regla» (su ley lo dice; el de Querétaro NO se ofrece)")
r = f0.reglas_para("amparo_directo", "Junta Especial Número Uno de la Local de Conciliación y Arbitraje")
ok("personal" in [x["clave"] for x in r["reglas"]] and r["por_omision"] == "personal", "una Junta → personal por omisión")
ok('@app.get("/taller/reglas-surtimiento")' in src and '"reglas_surtimiento": _reglas' in src,
   "la pantalla tiene su puerta, y el auto de admisión ya trae las reglas")
ok('ficha["encabezado"] = _encab' in src, "y el encabezado se compone desde el auto de admisión")

print("\n4 · LA CUARENTENA DEL BOLETÍN DE QUERÉTARO SOBRE EL TFJA")
ok('e.regla_surtimiento = "lfpca_boletin"' in inspect.getsource(ra.generar),
   "el boletín de Querétaro sobre el TFJA se corrige a lfpca_boletin, no a «personal»")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
