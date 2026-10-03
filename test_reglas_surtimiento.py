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

print("\n5 · «POR LISTA» ANTE EL TFJA NO ES UNA REGLA (3-oct-2026, rev_2)")
# Con «lista» y una Sala del TFJA se contaba un día y el considerando citaba los
# artículos 65 y 70 de la LFPCA, cuyo 65 hace surtir el Boletín al TERCER día.
# El aviso va con la bandera `procedencia_por_tipo` (el camino viejo no gana avisos).
import contexto_taller as ct
# LA BANDERA NACE «todos» (C8, 3-oct-2026): para ver el camino viejo hay que
# apagarla expresamente; antes bastaba no ponerla.
ct.poner(True, {"banderas": {"procedencia_por_tipo": False}}, pruebas=True)
c = f0.computar(dt.date(2025, 3, 3), dt.date(2025, 3, 20), "lista", 15, TFJA, None, "amparo_directo", None)
ok(not any("POR LISTA" in a for a in c.avisos), "sin la bandera, sin aviso nuevo")
ct.poner(True, {"banderas": {"procedencia_por_tipo": True}}, pruebas=True)
c = f0.computar(dt.date(2025, 3, 3), dt.date(2025, 3, 20), "lista", 15, TFJA, None, "amparo_directo", None)
ok(any(a.startswith("LA SALA DEL TFJA NO NOTIFICA «POR LISTA»") for a in c.avisos),
   "con la bandera, el cómputo avisa y pide la regla del Boletín Jurisdiccional")
ok(f0.surtimiento_nacional("amparo_directo", "administrativa", TFJA, "lista") == ("", ""),
   "y el respaldo nacional no le atribuye el día siguiente de los artículos 65 y 70")
c = f0.computar(N, P, "lfpca_boletin", 15, TFJA, None, "amparo_directo", None)
ok(not any("POR LISTA" in a for a in c.avisos), "con la regla del Boletín, ningún aviso")
ct.poner(True, {"banderas": {"procedencia_por_tipo": False}}, pruebas=True)

print("\n6 · EL DEPÓSITO POSTAL ENTRA AL CÓMPUTO (D3, 3-oct-2026)")
# El cableado (redactor_adelanto) lo pasa si `computar` lo recibe: la firma es el
# contrato.
ok("deposito" in inspect.signature(f0.computar).parameters, "computar(..., deposito=fecha)")
c = f0.computar(dt.date(2024, 10, 1), dt.date(2024, 11, 6), "lfpca_boletin", 15, TFJA, None,
                "revision_fiscal", None, deposito=dt.date(2024, 10, 24))
ok(c.oportuna is True and c.recepcion == dt.date(2024, 11, 6) and c.deposito == dt.date(2024, 10, 24),
   "RF 2/2025: en tiempo con el depósito; la recepción queda registrada aparte")

print("\n7 · LO AGRARIO: EL CÓDIGO NACIONAL EN LA CIUDAD DE MÉXICO, EL 321 FUERA (C7, 3-oct-2026)")
# David: «Si va con el Código Nacional, y no el Federal, entonces hay que adecuar
# al Código Nacional». Ley Agraria 167 (DOF 14-11-2025) → CNPCF; su aplicación
# espera la declaratoria (transitorio Segundo). La sede decide, con la misma
# regla del supletorio de la Ley de Amparo (`tipos_asunto.es_cdmx`).
ct.poner(True, {"banderas": {"procedencia_por_tipo": True}}, pruebas=True)
TUA = "Tribunal Unitario Agrario del Distrito 8"
T_CDMX, C_CDMX = "Primer Tribunal Colegiado en Materia Administrativa del Primer Circuito", "Ciudad de México"
T_QRO, C_QRO = ("Tercer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito",
                "Querétaro, Querétaro")
NA, PA = dt.date(2026, 3, 2), dt.date(2026, 3, 20)       # lunes 2 de marzo; presentada el 20
_CN = "del Código Nacional de Procedimientos Civiles y Familiares, de aplicación supletoria en términos del " \
      "artículo 167 de la Ley Agraria"
R = f0.REGLAS_SURTE
ok(R["cnpcf_personal"].dias_habiles == 0 and R["cnpcf_personal"].fundamento == "artículo 227, fracción I, " + _CN
   and R["cnpcf_lista"].dias_habiles == 1 and R["cnpcf_lista"].fundamento == "artículo 211 " + _CN
   and R["cnpcf_electronica"].dias_habiles == 0
   and R["cnpcf_electronica"].fundamento == "artículo 227, fracción III, " + _CN
   and R["cnpcf_lista"].descripcion == "por lista" and R["cnpcf_electronica"].descripcion == "por vía electrónica",
   "las tres reglas del Código Nacional: personal el mismo día (227-I), lista al siguiente (211), "
   "electrónica el mismo día (227-III), cada una con su fundamento")
ok(f0.fuero_de("amparo_directo", TUA) == "agrario" and f0.fuero_de("", "Tribunal Superior Agrario") == "agrario"
   and f0.fuero_de("", "el Tribunal Agrario") == "agrario"
   and f0.fuero_de("amparo_directo", TFJA) == "tfja"
   and f0.fuero_de("", "Junta Especial Número Uno de la Local de Conciliación y Arbitraje") != "agrario",
   "fuero_de reconoce el Tribunal Unitario, el Superior y el «Tribunal Agrario»; el TFJA sigue siendo tfja")

# ── EN LA CIUDAD DE MÉXICO ──
e_ag = ra.Encargo(numero="12/2026", encabezado="AMPARO DIRECTO ADMINISTRATIVO: 12/2026", quejoso="Ejido X",
                  magistrado="X", secretario="X", notificacion=NA, presentacion=PA, regla_surtimiento="personal",
                  tipo_asunto="amparo_directo", materia="administrativa", responsable=TUA,
                  tribunal=T_CDMX, ciudad=C_CDMX)
_rg, _av = ra.regla_agraria(e_ag)
# INTEGRACIÓN (3-oct-2026): la salida del juicio iniciado antes ya no es «Otra
# regla» sino la personal del Código Federal (`cfpc_personal`), que el guardián
# no toca. Antes se comprobaba ««Otra regla»» en este aviso.
# REVISIÓN DE NORMAS Y FRONT (3-oct-2026): el aviso del guardián dice sólo qué
# cambió y por qué, con la sede como REGLA DE LA CASA («donde la regla del taller
# aplica el Código Nacional», ya no «donde ya opera»); el transitorio y la salida
# del juicio iniciado antes los dice `f0.aviso_cnpcf_agrario`, que `generar`
# añade siempre que se cuente con el Código Nacional en la Ciudad de México
# (también cuando la regla llega sola de la pantalla). Antes se comprobaban aquí
# «transitorio Tercero» y «Personal — Código Federal (art. 321)».
ok(_rg == "cnpcf_personal" and _av.startswith("LO AGRARIO EN LA CIUDAD DE MÉXICO SE CUENTA CON EL CÓDIGO NACIONAL")
   and "artículo 227, fracción I" in _av and "un día hábil antes" in _av
   and "donde la regla del taller aplica el Código Nacional" in _av and "ya opera" not in _av
   and "«Otra regla»" not in _av,
   "el guardián: la «personal» que llega por omisión pasa a la del Código Nacional, con aviso; la sede, como "
   "regla de la casa")
e_ag.regla_surtimiento = "lista"
_rg, _av = ra.regla_agraria(e_ag)
ok(_rg == "cnpcf_lista" and "Las fechas no cambian" in _av and "artículo 211" in _av
   and "donde la regla del taller aplica el Código Nacional" in _av,
   "y la «lista» pasa a la del 211 (mismas fechas, otro precepto)")
e_ag.regla_surtimiento, e_ag.materia, e_ag.responsable = "personal", "agraria", ""
ok(ra.regla_agraria(e_ag)[0] == "cnpcf_personal", "también si sólo la materia dice «agraria»")
e_ag.responsable = TUA
_src_gen = inspect.getsource(ra.generar)
ok("regla_agraria(e, _mat)" in _src_gen
   and _src_gen.index("regla_agraria(e, _mat)") < _src_gen.index("c = _computar_con(e.regla_surtimiento)"),
   "el guardián corre en el adelanto ANTES del cómputo, junto a los de laboral y Querétaro")
# EN EL ADELANTO DE VERDAD (como test_regla_otra): sin cliente, `generar` cae al
# tocar el modelo, pero el guardián ya mutó la regla en las primeras líneas.
import asyncio
e_gen = ra.Encargo(numero="13/2026", encabezado="AMPARO DIRECTO ADMINISTRATIVO: 13/2026", quejoso="Ejido Y",
                   magistrado="X", secretario="X", notificacion=NA, presentacion=PA, regla_surtimiento="personal",
                   tipo_asunto="amparo_directo", materia="administrativa", responsable=TUA,
                   tribunal=T_CDMX, ciudad=C_CDMX)
try:
    asyncio.run(ra.generar(None, e_gen, "CONSIDERANDO. Texto del acto.", "CONCEPTOS. Texto.", "/dev/null"))
except Exception:
    pass
ok(e_gen.regla_surtimiento == "cnpcf_personal",
   f"generar() deja en el encargo la regla del Código Nacional (la sesión la guarda): {e_gen.regla_surtimiento}")
ca = f0.computar(NA, PA, "cnpcf_personal", 15, TUA, None, "amparo_directo", None)
ok(ca.surtio == NA and ca.inicio == dt.date(2026, 3, 3) and ca.vencimiento == dt.date(2026, 3, 24),
   f"CDMX: la personal surte el mismo día y el plazo arranca al día siguiente: "
   f"{ca.surtio} · {ca.inicio} · {ca.vencimiento}")
pa = f0.parrafo_oportunidad(ca, "", "amparo_directo", sin_precepto_en_hueco=True)
# EL 227-I NO DICE CUÁNDO SURTE (revisión de normas, 3-oct-2026): dice que los
# términos corren desde el día siguiente al de la notificación personal; el
# considerando lo dice así y de ahí saca el día. Antes se comprobaba «surtió
# efectos el mismo día, conforme al artículo 227, fracción I…».
ok("de manera personal y surtió efectos el mismo día, pues conforme al artículo 227, fracción I, " + _CN
   + ", los términos empiezan a correr el día siguiente al de la notificación personal, por lo que el plazo" in pa
   and "mismo día, conforme al artículo 227" not in pa
   and ", es decir, el dos de marzo" not in pa
   and "del tres al veinticuatro de marzo de dos mil veintiséis" in pa,
   "el considerando: «surtió efectos el mismo día, pues conforme al artículo 227, fracción I…, los términos "
   "empiezan a correr el día siguiente…», sin «, es decir, el …»")
ok(f0.aviso_fundamento(ca, "amparo_directo", en_hueco=True) == ""
   and "(artículo 227, fracción I, del Código Nacional" in f0.aviso_forma_no_consta(ca, "amparo_directo")
   and "y surtió efectos el mismo día, pues conforme al artículo 227" in f0.parrafo_oportunidad(
       ca, "", "amparo_directo", sin_precepto_en_hueco=True, forma_consta=False),
   "con precepto no hay aviso de fundamento; sin forma que conste, el aviso F1 dice con qué regla se contó")
# EL TRANSITORIO TAMBIÉN EN LA CIUDAD DE MÉXICO (revisión de normas y front, 3-oct-
# 2026, hallazgo alto): la pantalla propone la del Código Nacional y el guardián
# no la veía; ahora («», aviso). Antes se comprobaba ("", "") con cnpcf_personal.
_pcn, _acn = f0.surtimiento_nacional("amparo_directo", "administrativa", TUA, "cnpcf_personal",
                                     tribunal=T_CDMX, ciudad=C_CDMX)
ok(f0.surtimiento_nacional("amparo_directo", "administrativa", TUA, "personal", tribunal=T_CDMX,
                           ciudad=C_CDMX) == ("", "")
   and _pcn == "" and _acn == f0.aviso_cnpcf_agrario("cnpcf_personal")
   and _acn.startswith("EL SURTIMIENTO SE CONTÓ CON EL CÓDIGO NACIONAL") and "transitorio Tercero" not in _acn[:60]
   and "el Tercero deja los juicios en trámite" in _acn and "declaratoria del Congreso de la Unión" in _acn
   and "regla del taller" in _acn and "«Personal — Código Federal (art. 321)»" in _acn,
   "surtimiento_nacional en la Ciudad de México con la del Código Nacional: sin precepto (ya trae el suyo) y "
   "con el aviso del transitorio, la declaratoria federal y la salida al 321")
_pcl, _acl = f0.surtimiento_nacional("amparo_directo", "agraria", TUA, "cnpcf_lista", tribunal=T_CDMX, ciudad=C_CDMX)
ok(_pcl == "" and _acl.startswith("LA NOTIFICACIÓN POR LISTA SE FUNDÓ EN EL ARTÍCULO 211")
   and "las fechas no cambian" in _acl
   and f0.surtimiento_nacional("amparo_directo", "agraria", TUA, "cnpcf_personal", tribunal=T_QRO,
                               ciudad=C_QRO) == ("", "")
   and f0.surtimiento_nacional("amparo_directo", "agraria", TUA, "cnpcf_electronica", tribunal=T_CDMX,
                               ciudad=C_CDMX) == ("", ""),
   "la lista del Código Nacional lo dice sin cambiar fechas; fuera de la Ciudad de México (elegida a propósito) "
   "y la electrónica, nada")
# EXTEMPORÁNEA CON EL CÓDIGO NACIONAL, EN TIEMPO CON EL 321 (hallazgo alto, punto 2).
_pr25 = dt.date(2026, 3, 25)
_cn25 = f0.computar(NA, _pr25, "cnpcf_personal", 15, TUA, None, "amparo_directo", None)
_cf25 = f0.computar(NA, _pr25, "cfpc_personal", 15, TUA, None, "amparo_directo", None)
_ax = f0.contraste_cnpcf_agrario(_cn25, _cf25)
ok(_cn25.oportuna is False and _cf25.oportuna is True
   and _ax.startswith("CON EL CÓDIGO NACIONAL LA DEMANDA SALE EXTEMPORÁNEA Y CON EL ARTÍCULO 321")
   and "veinticuatro de marzo de dos mil veintiséis" in _ax and "veinticinco de marzo de dos mil veintiséis" in _ax
   and f0.contraste_cnpcf_agrario(ca, ccf if False else f0.computar(NA, PA, "cfpc_personal", 15, TUA, None,
                                                                    "amparo_directo", None)) == "",
   "presentada el 25 de marzo: con el 227-I vence el 24 (extemporánea) y con el 321 el 25 (en tiempo); el aviso "
   "dice las dos fechas, y en tiempo con las dos no dice nada")
ro = f0.reglas_para("amparo_directo", TUA, tribunal=T_CDMX, ciudad=C_CDMX)
# INTEGRACIÓN (3-oct-2026): la personal del Código Federal es `cfpc_personal`
# (antes, la «personal» genérica, que el guardián cambiaba aunque se eligiera).
ok(ro["fuero"] == "agrario" and ro["por_omision"] == "cnpcf_personal"
   and [x["clave"] for x in ro["reglas"]] == ["cnpcf_personal", "cnpcf_lista", "cnpcf_electronica",
                                              "cfpc_personal", "lista", "otra"]
   and ro["reglas"][0]["etiqueta"].startswith("Personal — Código Nacional (art. 227, fr. I)")
   and ro["reglas"][3]["etiqueta"] == "Personal — Código Federal (art. 321): surte al día siguiente",
   "reglas_para en la Ciudad de México: las tres del Código Nacional primero y por omisión la personal suya; "
   "la del Código Federal, con su propia clave")
# ── LA PERSONAL DEL CÓDIGO FEDERAL ELEGIDA A PROPÓSITO (integración, 3-oct-2026) ──
_cf = R["cfpc_personal"]
ok(_cf.dias_habiles == 1 and _cf.descripcion == "de manera personal"
   and _cf.fundamento == ("artículo 321 del Código Federal de Procedimientos Civiles, de aplicación supletoria "
                          "en materia agraria"),
   "cfpc_personal: surte al día siguiente, «de manera personal», con el 321 del CFPC supletorio en lo agrario")
e_ag.regla_surtimiento = "cfpc_personal"
ok(ra.regla_agraria(e_ag) == ("", ""),
   "el guardián NO cambia la personal del Código Federal elegida a propósito en la Ciudad de México")
e_ag.regla_surtimiento = "personal"
ccf = f0.computar(NA, PA, "cfpc_personal", 15, TUA, None, "amparo_directo", None)
pcf = f0.parrafo_oportunidad(ccf, "", "amparo_directo", sin_precepto_en_hueco=True)
ok(ccf.surtio == dt.date(2026, 3, 3) and ccf.inicio == dt.date(2026, 3, 4)
   and "surtió efectos al día hábil siguiente, conforme al artículo 321 del Código Federal de Procedimientos "
       "Civiles, de aplicación supletoria en materia agraria" in pcf
   and f0.aviso_fundamento(ccf, "amparo_directo", en_hueco=True) == ""
   and f0.surtimiento_nacional("amparo_directo", "administrativa", TUA, "cfpc_personal",
                               tribunal=T_CDMX, ciudad=C_CDMX) == ("", ""),
   "en la Ciudad de México, la del Código Federal surte al día siguiente, cita el 321 y no lleva aviso")
c_dom = f0.computar(dt.date(2026, 3, 1), PA, "cnpcf_electronica", 15, TUA, None, "amparo_directo", None)
ok(not any(a.startswith("LA NOTIFICACIÓN CAE EN DÍA INHÁBIL") for a in c_dom.avisos)
   and any(a.startswith("LA NOTIFICACIÓN CAE EN DÍA INHÁBIL") for a in f0.computar(
       dt.date(2026, 3, 1), PA, "cnpcf_personal", 15, TUA, None, "amparo_directo", None).avisos),
   "la electrónica del Código Nacional en domingo no es error de captura; la personal sí se avisa")

# ── FUERA DE LA CIUDAD DE MÉXICO ──
e_ag.tribunal, e_ag.ciudad, e_ag.materia = T_QRO, C_QRO, "administrativa"
ok(ra.regla_agraria(e_ag) == ("", ""), "Querétaro: el guardián no toca la «personal»")
cq = f0.computar(NA, PA, "personal", 15, TUA, None, "amparo_directo", None)
ok(cq.surtio == dt.date(2026, 3, 3) and cq.inicio == dt.date(2026, 3, 4),
   f"Querétaro: la personal surte al día siguiente: {cq.surtio} · {cq.inicio}")
p321, a321 = f0.surtimiento_nacional("amparo_directo", "administrativa", TUA, "personal",
                                     tribunal=T_QRO, ciudad=C_QRO)
ok(p321 == "el artículo 321 del Código Federal de Procedimientos Civiles, de aplicación supletoria en materia agraria"
   and a321.startswith("EL SURTIMIENTO SE FUNDÓ EN EL ARTÍCULO 321 DEL CÓDIGO FEDERAL")
   and "el tribunal no reside en la Ciudad de México" in a321 and "transitorio Segundo" in a321
   and "1 de abril de 2027" in a321 and "«Personal — Código Nacional»" in a321,
   "surtimiento_nacional fuera de la Ciudad de México: el 321 del CFPC, con el aviso del transitorio")
pq = f0.parrafo_oportunidad(cq, "", "amparo_directo", sin_precepto_en_hueco=True, fundamento_surtimiento=p321)
ok("surtió efectos al día hábil siguiente, conforme al artículo 321 del Código Federal de Procedimientos Civiles, "
   "de aplicación supletoria en materia agraria, es decir, el tres de marzo de dos mil veintiséis" in pq
   and "Código Nacional" not in pq,
   "y el considerando lo cita en el sitio del precepto")
_ps, _as = f0.surtimiento_nacional("amparo_directo", "agraria", "", "lista")
ok(_ps == p321 and "no se pudo saber si el tribunal reside en la Ciudad de México" in _as,
   "sin sede conocida, también el 321 (con la «lista»), y el aviso dice que no se supo la sede")
# LA SALIDA DE LA LISTA ES LA LISTA DEL CÓDIGO NACIONAL (revisión AD, 3-oct-2026):
# el aviso mandaba a «Personal — Código Nacional» (surte el mismo día) y quien lo
# siguiera arrancaba el plazo un día antes de una notificación por lista.
_pl, _al = f0.surtimiento_nacional("amparo_directo", "agraria", TUA, "lista", tribunal=T_QRO, ciudad=C_QRO)
ok(_pl == p321 and "elige «Por lista — Código Nacional» (art. 211: surte al día siguiente; las fechas no cambian"
   in _al and "«Personal — Código Nacional»" not in _al
   and "«Personal — Código Nacional»" in a321 and "«Por lista" not in a321,
   "fuera de la Ciudad de México, con la «lista» el aviso manda a la lista del 211; con la personal, a la del 227-I")
rq = f0.reglas_para("amparo_directo", TUA, tribunal=T_QRO, ciudad=C_QRO)
# INTEGRACIÓN (3-oct-2026): fuera de la Ciudad de México la que se propone es
# la personal del Código Federal con su clave (antes, «personal»).
ok(rq["por_omision"] == "cfpc_personal"
   and [x["clave"] for x in rq["reglas"]] == ["cfpc_personal", "lista", "cnpcf_personal", "cnpcf_lista",
                                              "cnpcf_electronica", "otra"]
   and f0.reglas_para("amparo_directo", TUA)["por_omision"] == "cfpc_personal",
   "reglas_para fuera de la Ciudad de México (o sin sede): la del Código Federal por omisión, las del Código "
   "Nacional ofrecidas")
# Y FUERA DE ELLA, CON `cfpc_personal`, EL MISMO RESULTADO QUE CON LA GENÉRICA:
# el 321 en el párrafo y el aviso del transitorio (no se sabe si se eligió).
_pcf, _acf = f0.surtimiento_nacional("amparo_directo", "administrativa", TUA, "cfpc_personal",
                                     tribunal=T_QRO, ciudad=C_QRO)
pcq = f0.parrafo_oportunidad(f0.computar(NA, PA, "cfpc_personal", 15, TUA, None, "amparo_directo", None),
                             "", "amparo_directo", sin_precepto_en_hueco=True, fundamento_surtimiento=_pcf)
ok(_pcf == p321 and _acf == a321 and _pcf.split(" ", 1)[-1] in pcq
   and pcq.count("artículo 321") == 1,
   "fuera de la Ciudad de México cfpc_personal da el mismo párrafo y el mismo aviso que la personal genérica")

# ── LO ELEGIDO A PROPÓSITO SE RESPETA; LO QUE NO ES AGRARIO NO SE TOCA ──
e_ag.tribunal, e_ag.ciudad = T_CDMX, C_CDMX
_respeta = []
for _r in ("otra", "oficio", "electronica", "lfpca", "lfpca_boletin", "cnpcf_electronica", "cnpcf_lista"):
    e_ag.regla_surtimiento = _r
    _respeta.append(ra.regla_agraria(e_ag) == ("", ""))
ok(all(_respeta), "el guardián no pisa lo elegido a propósito (otra, oficio, electrónica, lfpca, las del CNPCF)")
e_ag.regla_surtimiento = "personal"
e_ag.tipo_asunto = "amparo_revision"
ok(ra.regla_agraria(e_ag) == ("", ""), "ni la revisión: lo notificado es del juez de amparo (art. 31 LA)")
e_ag.tipo_asunto, e_ag.materia, e_ag.responsable = "amparo_directo", "civil", "Juzgado Primero Civil de Primera Instancia"
ok(ra.regla_agraria(e_ag) == ("", ""), "ni un civil en la Ciudad de México")

# ── LA FORMA QUE CONSTA EN LA CIUDAD DE MÉXICO (revisión AD, 3-oct-2026) ──
# La personal del Código Nacional llega sola y surte el mismo día; si la ficha
# dice «por lista», la del 211 surte al día siguiente: el plazo cambia un día.
e_pf = ra.Encargo(numero="14/2026", encabezado="AMPARO DIRECTO AGRARIO: 14/2026", quejoso="Ejido Z",
                  magistrado="X", secretario="X", notificacion=NA, presentacion=PA,
                  regla_surtimiento="cnpcf_personal", tipo_asunto="amparo_directo", materia="agraria",
                  responsable=TUA, tribunal=T_CDMX, ciudad=C_CDMX)
_rf, _af = ra.regla_agraria_por_forma(e_pf, "por lista", "secretario")
ok(_rf == "cnpcf_lista" and _af.startswith("LA NOTIFICACIÓN FUE POR LISTA (así lo declaraste")
   and "ARTÍCULO 211" in _af and "un día hábil después" in _af,
   f"cnpcf_personal con la forma «por lista» pasa a la del 211, con aviso: {_af[:90]}")
ok(ra.regla_agraria_por_forma(e_pf, "electrónica", "acto")[0] == "cnpcf_electronica"
   and ra.regla_agraria_por_forma(e_pf, "personal", "secretario") == ("", "")
   and ra.regla_agraria_por_forma(e_pf, "", "") == ("", ""),
   "la electrónica pasa a la del 227-III; la personal o sin forma, nada")
e_pf.regla_surtimiento = "cfpc_personal"
ok(ra.regla_agraria_por_forma(e_pf, "lista", "secretario") == ("", ""),
   "la del Código Federal elegida a propósito no se toca")
e_pf.regla_surtimiento, e_pf.tipo_asunto = "cnpcf_personal", "amparo_revision"
ok(ra.regla_agraria_por_forma(e_pf, "lista", "secretario") == ("", ""), "ni fuera del amparo directo")
e_pf.tipo_asunto = "amparo_directo"
# DE PUNTA A PUNTA EN `generar` (la pieza que corre tras leer los papeles): con
# la forma declarada «lista», el cómputo se rehace con la del 211 y el aviso del
# guardián (que hablaba de la personal) se va; con la del Código Nacional,
# siempre el aviso del transitorio; extemporánea con el 227-I y en tiempo con el
# 321, el contraste.
_src_gen2 = inspect.getsource(ra.generar)
ok("_agrario_con_el_codigo_nacional(e, c, avisos, _computar_con, _av_agr," in _src_gen2
   and _src_gen2.index("_agrario_con_el_codigo_nacional(") > _src_gen2.index("_acto_l, _av_acto = _acto_tramite"),
   "generar llama a la pieza agraria en el amparo directo, después de leer los papeles")
_cc = lambda regla: f0.computar(NA, _pr25, regla, 15, TUA, None, "amparo_directo", None)
e_pf.regla_surtimiento = "cnpcf_personal"
_avs = ["LO AGRARIO EN LA CIUDAD DE MÉXICO SE CUENTA CON EL CÓDIGO NACIONAL: (del guardián)"]
_c0 = _cc("cnpcf_personal")
_c1 = ra._agrario_con_el_codigo_nacional(e_pf, _c0, _avs, _cc, _avs[0], "lista", "secretario")
ok(e_pf.regla_surtimiento == "cnpcf_lista" and _c1.oportuna is True and _c1.vencimiento == dt.date(2026, 3, 25)
   and not any(a.startswith("LO AGRARIO EN LA CIUDAD") for a in _avs)
   and any(a.startswith("LA NOTIFICACIÓN FUE POR LISTA") for a in _avs)
   and f0.aviso_cnpcf_agrario("cnpcf_lista") in _avs,
   "por lista: vence el 25 y la demanda del 25 está en tiempo; sin el aviso de la personal y con el del 211")
e_pf.regla_surtimiento = "cnpcf_personal"
_avs = []
_c2 = ra._agrario_con_el_codigo_nacional(e_pf, _cc("cnpcf_personal"), _avs, _cc, "", "", "")
ok(e_pf.regla_surtimiento == "cnpcf_personal" and _c2.oportuna is False
   and f0.aviso_cnpcf_agrario("cnpcf_personal") in _avs
   and any(a.startswith("CON EL CÓDIGO NACIONAL LA DEMANDA SALE EXTEMPORÁNEA") for a in _avs),
   "la del Código Nacional llegada de la pantalla (sin guardián): el transitorio y, extemporánea, el contraste")
e_pf.tribunal, e_pf.ciudad = T_QRO, C_QRO
_avs = []
ra._agrario_con_el_codigo_nacional(e_pf, _cc("cnpcf_personal"), _avs, _cc, "", "", "")
ok(not any("transitorio" in a or "Tercero" in a for a in _avs if not a.startswith("CON EL CÓDIGO NACIONAL")),
   "fuera de la Ciudad de México la del Código Nacional se eligió a propósito: sin el aviso del transitorio")

# ── MERCANTIL Y TFJA, INTACTOS ──
ok(f0.surtimiento_nacional("amparo_directo", "mercantil", "Juez Primero Civil", "personal",
                           tribunal=T_CDMX, ciudad=C_CDMX)[0] == "el artículo 1075 del Código de Comercio"
   and f0.surtimiento_nacional("amparo_directo", "mercantil", "Juez Primero Civil", "personal")[0]
   == "el artículo 1075 del Código de Comercio"
   and "artículos 65 y 70 de la Ley Federal de Procedimiento Contencioso" in f0.surtimiento_nacional(
       "amparo_directo", "administrativa", TFJA, "personal", tribunal=T_QRO, ciudad=C_QRO)[0]
   and f0.surtimiento_nacional("amparo_directo", "administrativa", TFJA, "lista") == ("", "")
   and f0.surtimiento_nacional("amparo_directo", "civil", "Juzgado Primero Civil", "personal",
                               tribunal=T_CDMX, ciudad=C_CDMX) == ("", ""),
   "mercantil (1075) y TFJA (65 y 70) como antes, con o sin sede; un civil sigue sin omisión nacional")
ok(f0.reglas_para("amparo_directo", TFJA, tribunal=T_CDMX, ciudad=C_CDMX)["por_omision"] == "lfpca_boletin"
   and [x["clave"] for x in f0.reglas_para("queja", TUA, tribunal=T_CDMX, ciudad=C_CDMX)["reglas"]][:2]
   == ["personal", "lista"],
   "reglas_para del TFJA no cambia con la sede; la queja sigue con las del 31 aunque la responsable sea agraria")
ct.poner(True, {"banderas": {"procedencia_por_tipo": False}}, pruebas=True)

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
