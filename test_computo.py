"""EL BANCO DEL CÓMPUTO DE PLAZOS.

Nació el 12-sep-2026, cuando David vio que un amparo directo salía extemporáneo
y el proyecto no le daba el fondo pese a que él había calificado los conceptos
de violación, y que la regla de notificación era la de un estado concreto en un
taller que sirve a toda la república.

Lo que vigila, y por qué cada cosa:

 · LAS 103 SESIONES REALES NO SE MUEVEN. Es la comprobación que impide que un
   arreglo del cómputo cambie en silencio plazos que ya estaban bien. Se
   alimenta de `taller_sesiones`; sin base, ese bloque se salta.
 · EL ENGROSE DE ORO ADA 240/2026 se reproduce al día. Es el documento firmado
   contra el que la casa sabe que la aritmética no se ha roto.
 · LOS INHÁBILES DE LA RESPONSABLE se descuentan SÓLO donde el escrito se
   presenta ante ella —amparo directo y revisión fiscal— y no en amparo en
   revisión ni en queja, donde se presenta ante órgano federal.
   P./J. 4/2022 (11a.), registro digital 2024494.
 · LAS DOS VÍAS de la decisión del secretario sobre un cómputo extemporáneo.
 · Y que nada sobreviva en memoria de proceso: el API corre con -w 2.

Se corre solo:

    python3 test_computo.py
"""
import sys, json, datetime as dt, copy
sys.path.insert(1, "/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/jurexia-api-git")

REPO = "/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/jurexia-api-git"
sys.path.insert(2, REPO)
import importlib.util
def _repo_mod(nombre):
    spec = importlib.util.spec_from_file_location("repo_" + nombre, f"{REPO}/{nombre}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules['repo_' + nombre] = m
    spec.loader.exec_module(m)
    return m
viejo = _repo_mod("fase0_oportunidad")

import fase0_oportunidad as f0
import documento_generado as dg
import fase6_estudio as f6
from docx import Document

D = lambda x: dt.date.fromisoformat(x) if x else None
FALLOS = []
def check(etiqueta, cond, detalle=""):
    print(f"   {'PASA ' if cond else 'FALLA'}  {etiqueta}" + (f"   {detalle}" if detalle else ""))
    if not cond: FALLOS.append(etiqueta)

def titulo(n, t):
    print("\n" + "═"*92); print(f"{n}. {t}"); print("═"*92)

# ══════════════════════════════════════════════════════════════════════════
titulo(1, "(d) LO QUE NO DEBE CAMBIAR — 103 sesiones reales + el engrose de oro")
# ══════════════════════════════════════════════════════════════════════════
S = json.load(open("/tmp/audit_f0/sesiones.json"))
ig = di = 0
for s in S:
    if not (s["notif"] and s["pres"]): continue
    ex = [D(x) for x in (s["extra"] or [])]
    a = viejo.computar(D(s["notif"]), D(s["pres"]), s["regla"], s["plazo"] or 15, s["resp"], ex)
    b = f0.computar(D(s["notif"]), D(s["pres"]), s["regla"], s["plazo"] or 15, s["resp"], ex,
                    tipo_asunto=s["tipo"] or "amparo_directo")
    mismo = (a.surtio == b.surtio and a.inicio == b.inicio and a.vencimiento == b.vencimiento
             and a.oportuna == b.oportuna and a.dias == b.dias
             and a.inhabiles_en_medio == b.inhabiles_en_medio)
    if mismo: ig += 1
    else:
        di += 1
        print(f"      ✗ {s['exp']:12} viejo {a.vencimiento} {a.oportuna} | nuevo {b.vencimiento} {b.oportuna}")
check("las 103 sesiones dan EXACTAMENTE lo de hoy sin declarar nada", di == 0, f"idénticas {ig} · distintas {di}")

c = f0.computar(dt.date(2026,2,23), None, "tja_qro_boletin", 15, tipo_asunto="amparo_directo")
check("engrose de oro ADA 240/2026 (surte 26-feb, plazo 27-feb→20-mar, 16-mar nombrado)",
      c.surtio == dt.date(2026,2,26) and c.inicio == dt.date(2026,2,27)
      and c.vencimiento == dt.date(2026,3,20) and dt.date(2026,3,16) in c.inhabiles_en_medio,
      f"{c.surtio} {c.inicio} {c.vencimiento}")

# 245/2024 — el caso de control con veinte días de holgura
c = f0.computar(dt.date(2024,3,4), dt.date(2024,3,5), "personal", 15,
                "Junta Especial Número 50", tipo_asunto="amparo_directo")
check("245/2024 (control, holgura de 20 días) sigue OPORTUNA", c.oportuna is True,
      f"vence {c.vencimiento}, día {c.dia_de_presentacion} de 15")
# ...y no se mueve ni declarando un periodo vacacional ajeno a la ventana
c2 = f0.computar(dt.date(2024,3,4), dt.date(2024,3,5), "personal", 15, "Junta Especial Número 50",
                 tipo_asunto="amparo_directo", inhabiles_responsable="2024-07-16..2024-07-31")
check("245/2024 no se mueve con una declaración fuera de la ventana",
      c2.vencimiento == c.vencimiento and c2.oportuna is True)

# 844/2026 y 406/2025 — oportunas hoy, oportunas después
for exp, notif, pres, regla, resp in (("844/2026","2026-09-02","2026-09-09","personal","SALA"),
                                      ("406/2025","2025-04-29","2025-05-19","lista","Juez de Primera Instancia Civil")):
    a = viejo.computar(D(notif), D(pres), regla, 15, resp)
    b = f0.computar(D(notif), D(pres), regla, 15, resp, tipo_asunto="amparo_directo")
    check(f"{exp} idéntica (vence {a.vencimiento}, oportuna {a.oportuna})",
          a.vencimiento == b.vencimiento and a.oportuna == b.oportuna)

# ══════════════════════════════════════════════════════════════════════════
titulo(2, "(a) AMPARO DIRECTO CON PERIODO VACACIONAL DE LA RESPONSABLE — 93/2026 real")
# ══════════════════════════════════════════════════════════════════════════
# Sesión real de taller_sesiones, 11-sep-2026: amparo directo 93/2026, Sala
# Regional en Querétaro del TFJA, notificación 5-dic-2025, presentada 19-ene-2026.
NOT93, PRE93, RESP93 = dt.date(2025,12,5), dt.date(2026,1,19), "Sala Regional en Querétaro del TFJA"
sin = f0.computar(NOT93, PRE93, "lfpca", 15, RESP93, tipo_asunto="amparo_directo")
check("93/2026 SIN declarar: extemporánea por cuatro días (como hoy)",
      sin.vencimiento == dt.date(2026,1,15) and sin.oportuna is False, f"vence {sin.vencimiento}")
check("y el aviso dice CUÁNTOS días de la responsable lo salvarían",
      any("bastan 2 día" in a for a in sin.avisos),
      [a[:90] for a in sin.avisos if "bastan" in a][:1])

con = f0.computar(NOT93, PRE93, "lfpca", 15, RESP93, tipo_asunto="amparo_directo",
                  inhabiles_responsable="2025-12-16..2026-01-05")
check("93/2026 declarando el cierre de la Sala: OPORTUNA",
      con.vencimiento == dt.date(2026,1,19) and con.oportuna is True, f"vence {con.vencimiento}")
check("el descuento se aplicó y queda medido (resp_aplicados)", con.resp_aplicados is True)
check("el aviso funda el descuento en el artículo 176, no en el 19",
      any("artículo 176" in a and "responsable no laboró" in a for a in con.avisos))
check("y avisa de los 13 días declarados que YA eran inhábiles (no infla el trabajo)",
      any("13 YA eran inhábiles" in a for a in con.avisos))

p = f0.parrafo_oportunidad(con, "artículo 17 de la Ley de Amparo", "amparo_directo")
check("el considerando separa los dos fundamentos y cita el registro 2024494",
      "artículo 19 de la Ley de Amparo" in p and "artículo 176" in p and "2024494" in p)
print("\n   CONSIDERANDO:\n   " + p[:900].replace("\n", "\n   "))

# ══════════════════════════════════════════════════════════════════════════
titulo(3, "(a bis) EL DESCUENTO SÓLO DONDE EL ESCRITO SE PRESENTA ANTE LA RESPONSABLE")
# ══════════════════════════════════════════════════════════════════════════
ESPERA = {"amparo_directo": True, "revision_fiscal": True,
          "amparo_revision": False, "queja": False}
for tipo, debe in ESPERA.items():
    a = f0.computar(dt.date(2026,6,22), None, "personal", 15, "X", tipo_asunto=tipo)
    b = f0.computar(dt.date(2026,6,22), None, "personal", 15, "X", tipo_asunto=tipo,
                    inhabiles_responsable="2026-06-29..2026-07-10")
    movio = b.vencimiento != a.vencimiento
    check(f"{tipo:16} {'SE MUEVE' if debe else 'NO se mueve'}", movio is debe,
          f"{a.vencimiento} → {b.vencimiento}")
    if not debe:
        check(f"   …y {tipo} explica POR QUÉ con su artículo",
              any("NO se descontaron" in x for x in b.avisos),
              [x[:120] for x in b.avisos if "NO se descontaron" in x][:1])

# ══════════════════════════════════════════════════════════════════════════
titulo(4, "(c) UN CASO DE CADA TIPO CON SU PLAZO")
# ══════════════════════════════════════════════════════════════════════════
import tipos_asunto as ta
CUATRO = [
    ("amparo_directo",  15, "artículo 17 de la Ley de Amparo"),
    ("amparo_revision", 10, "artículo 86 de la Ley de Amparo"),
    ("queja",            5, "artículo 98 de la Ley de Amparo"),
    ("revision_fiscal", 15, "artículo 63 de la Ley Federal de Procedimiento Contencioso Administrativo"),
]
for tipo, dias, fund in CUATRO:
    pz = ta.plazo_de(tipo, "")
    check(f"{tipo:16} plazo = {dias} días", pz.get("dias") == dias, f"catálogo dice {pz.get('dias')}")
    print(f"          fundamento del catálogo: {pz.get('fundamento')}")
    c = f0.computar(dt.date(2026,3,2), None, "personal", dias, "X", tipo_asunto=tipo)
    check(f"   …y el cómputo cuenta {dias} días hábiles", len(c.dias) == dias,
          f"del {c.inicio} al {c.vencimiento}")
# queja EN CUALQUIER TIEMPO — artículo 98, fracción II
c = f0.computar(dt.date(2025,1,10), dt.date(2026,9,1), "personal", 0, "X", tipo_asunto="queja")
check("queja por omitir tramitar la demanda: EN CUALQUIER TIEMPO, nunca extemporánea",
      c.en_cualquier_tiempo and c.oportuna is not False and not c.cierra_por_extemporaneidad)
av = f0.aplicar_decision(c, "oportuna", "x"*60)
check("   …y ahí la decisión del secretario sobra y se le dice", "NO HACÍA FALTA" in av[0])

# ══════════════════════════════════════════════════════════════════════════
titulo(5, "(b) EXTEMPORÁNEO EN EL QUE EL SECRETARIO DECIDE ENTRAR AL FONDO")
# ══════════════════════════════════════════════════════════════════════════
ESTUDIO = ("Es FUNDADO el primer concepto de violacion. La responsable omitio "
           "valorar la documental de foja 120, que acredita el pago. "
           "En esas condiciones, procede conceder el amparo.\n\n"
           "El segundo concepto de violacion es INOPERANTE.")
DATOS = {"tipo_asunto": "amparo_directo", "numero": "96/2025",
         "encabezado": "AMPARO DIRECTO 96/2025", "quejoso": "Juan Perez",
         "responsable": "la Junta Especial Numero Dos", "magistrado": "M",
         "secretario": "S", "tribunal": "Tercer Tribunal Colegiado",
         "ciudad": "Queretaro", "acto": "el laudo de 10 de febrero de 2025",
         "antecedentes": "La Junta dicto laudo absolutorio."}
def componer(c, ruta):
    est = dg.Estructura(apertura="V.", visto="para resolver.",
                        resultandos=[{"titulo": "Laudo reclamado", "texto": "La Junta dicto laudo absolutorio."}],
                        competencia="", existencia="", procedencia="")
    dg.componer(dict(DATOS), est, c, f0.fecha_en_letra, ruta,
                antecedentes=["La Junta dicto laudo absolutorio."],
                resumen_acto=["Laudo absolutorio."],
                resumen_conceptos=["Omision de valorar una documental."],
                estudio=f6.parrafos(ESTUDIO), calificaciones=["fundado", "inoperante"],
                tipo_asunto="amparo_directo")
    ps = [p.text.strip() for p in Document(ruta).paragraphs]
    return "\n".join(ps), ps, est

NOTIF, PRES = dt.date(2025,3,13), dt.date(2025,4,30)
MOT = ("la Junta Especial Número Dos suspendió labores del catorce al veinticinco de abril de "
       "dos mil veinticinco, días que el calendario del Poder Judicial de la Federación tiene "
       "por hábiles, y la demanda se presentó por su conducto (artículo 176 de la Ley de Amparo)")

cA = f0.computar(NOTIF, PRES, "personal", 15, tipo_asunto="amparo_directo")
tA, pA, eA = componer(cA, "/tmp/plan_unico/A_sin_decision.docx")
check("A · SIN DECISIÓN: sobresee y NO entra al fondo (el camino de hoy)",
      "foja 120" not in tA and "Se sobresee" in tA, f"{len(pA)} párrafos")

cB = f0.computar(NOTIF, PRES, "personal", 15, tipo_asunto="amparo_directo")
avB = f0.aplicar_decision(cB, "oportuna", MOT)
tB, pB, eB = componer(cB, "/tmp/plan_unico/B_oportuna.docx")
check("B · «FUE OPORTUNA»: el estudio de fondo ESTÁ en la ejecutoria",
      "foja 120" in tB and "INOPERANTE" in tB, f"{len(pB)} párrafos")
check("B · el resolutivo ampara, no sobresee",
      "ampara y protege" in tB and "Se sobresee" not in tB)
check("B · la razón del secretario va LITERAL en el considerando", "suspendió labores del catorce" in tB)
check("B · el considerando cita el artículo 62 (improcedencia oficiosa)", "artículo 62" in tB)
check("B · la aritmética NO se tocó: c.oportuna sigue siendo False", cB.oportuna is False)

cC = f0.computar(NOTIF, PRES, "personal", 15, tipo_asunto="amparo_directo")
f0.aplicar_decision(cC, "reserva", "acepto el cómputo; pido el estudio para llevarlo a sesión del Pleno")
tC, pC, eC = componer(cC, "/tmp/plan_unico/C_reserva.docx")
i_anexo = next((i for i,t in enumerate(pC) if "A N E X O" in t or "Anexo de trabajo" in t), None)
i_res = next((i for i,t in enumerate(pC) if "R E S U E L V E" in t), None)
check("C · «EN RESERVA»: la ejecutoria sigue sobreseyendo", "Se sobresee" in tC)
check("C · el estudio está, pero DETRÁS de los resolutivos",
      "foja 120" in tC and i_anexo is not None and i_res is not None and i_anexo > i_res,
      f"resolutivos en {i_res}, anexo en {i_anexo}")
check("C · el anexo dice que NO forma parte de la ejecutoria", "NO FORMA PARTE DE LA EJECUTORIA" in tC)
check("C · la ejecutoria (hasta el resolutivo) es la misma que sin decidir",
      pC[:i_res] == pA[:next(i for i,t in enumerate(pA) if "R E S U E L V E" in t)])

# EL AVISO SE VE
for a in (eB.avisos or [])[:1]: print("\n   AVISO B:", a[:260])
for a in (eC.avisos or [])[:1]: print("   AVISO C:", a[:260])

# ══════════════════════════════════════════════════════════════════════════
titulo(6, "LA DECISIÓN NO PUEDE SER UN CLIC — y no puede quedarse pegada")
# ══════════════════════════════════════════════════════════════════════════
for m in ("", "sí", "ok", "ya lo vi", "procede", "xxxxxxxxxxxx"):
    c = f0.computar(NOTIF, PRES, "personal", 15, tipo_asunto="amparo_directo")
    f0.aplicar_decision(c, "oportuna", m)
    check(f"motivo «{m or '(vacío)'}» NO aplica la decisión", c.decision == "")
c = f0.computar(NOTIF, PRES, "personal", 15, tipo_asunto="amparo_directo")
f0.aplicar_decision(c, "oportuna", MOT); antes = c.decision
f0.aplicar_decision(c, "", "")
check("resolver otra vez SIN la casilla borra la decisión (no es pegajosa)",
      antes == "oportuna" and c.decision == "" and c.cierra_por_extemporaneidad is True)
c = f0.computar(dt.date(2026,9,2), dt.date(2026,9,9), "personal", 15, tipo_asunto="amparo_directo")
av = f0.aplicar_decision(c, "oportuna", MOT)
check("sobre un cómputo YA oportuno la decisión no se estampa y se dice por qué",
      c.decision == "" and "YA DA" in av[0])

# ══════════════════════════════════════════════════════════════════════════
titulo(7, "LAS GUARDAS DEL LECTOR DE TRAMOS — el error se nombra, no se traga")
# ══════════════════════════════════════════════════════════════════════════
CASOS = [("2026-07-10..2026-06-29", "AL REVÉS"), ("2026-02-30", "No entendí"),
         ("2026-01-01..2026-12-31", "abarca"), ("2010-05-05", "fuera de la Ley de Amparo")]
for txt, marca in CASOS:
    r = f0.leer_inhabiles_responsable(txt)
    check(f"«{txt}» → error nombrado ({marca})", bool(r.errores) and marca in r.errores[0],
          r.errores[0][:90] if r.errores else "SIN ERROR")

# ══════════════════════════════════════════════════════════════════════════
titulo(8, "gunicorn -w 2: nada sobrevive en memoria de proceso ni contamina el global")
# ══════════════════════════════════════════════════════════════════════════
antes_cal = (set(f0.CALENDARIO_AMPARO.sueltos), dict(f0.REGLAS_SURTE))
for s in S[:40]:
    if not s["notif"]: continue
    f0.computar(D(s["notif"]), D(s["pres"]), s["regla"], s["plazo"] or 15, s["resp"],
                [D(x) for x in (s["extra"] or [])], tipo_asunto=s["tipo"] or "amparo_directo",
                inhabiles_responsable="2026-06-29..2026-07-10")
check("CALENDARIO_AMPARO y REGLAS_SURTE intactos tras 40 cómputos con declaración",
      antes_cal == (set(f0.CALENDARIO_AMPARO.sueltos), dict(f0.REGLAS_SURTE)))
# la rehidratación con los ocho argumentos
kw = dict(tipo_asunto="amparo_directo", inhabiles_responsable="2025-12-16..2026-01-05")
w_adelanto = f0.computar(NOT93, PRE93, "lfpca", 15, RESP93, None, **kw)
w_resolver_hoy = f0.computar(NOT93, PRE93, "lfpca", 15, RESP93)          # como llama main.py hoy
w_resolver_ok = f0.computar(NOT93, PRE93, "lfpca", 15, RESP93, None, **kw)
check("el worker que compone el .docx debe recibir los MISMOS argumentos",
      w_adelanto.oportuna is True and w_resolver_hoy.oportuna is False
      and w_resolver_ok.oportuna is True,
      f"adelanto {w_adelanto.vencimiento} · resolver-hoy {w_resolver_hoy.vencimiento} · resolver-arreglado {w_resolver_ok.vencimiento}")

# ══════════════════════════════════════════════════════════════════════════
titulo(9, "EL LOCALISMO QUE SIGUE VIVO — 351/2026 (medición del encargo 3)")
# ══════════════════════════════════════════════════════════════════════════
# Sesión real: amparo directo 351/2026, responsable Sala Regional del TFJA,
# regla guardada `tja_qro_boletin` (Boletín del TJA del ESTADO de Querétaro).
qro = f0.computar(dt.date(2026,4,10), dt.date(2026,5,11), "tja_qro_boletin", 15,
                  "Sala Regional del Centro II del TFJA", tipo_asunto="amparo_directo")
lfp = f0.computar(dt.date(2026,4,10), dt.date(2026,5,11), "lfpca", 15,
                  "Sala Regional del Centro II del TFJA", tipo_asunto="amparo_directo")
print(f"   con la regla de QUERÉTARO (3 días, fundamento vacío): vence {qro.vencimiento} · oportuna {qro.oportuna}")
print(f"   con la ley que le toca  (art. 70 LFPCA, 1 día)     : vence {lfp.vencimiento} · oportuna {lfp.oportuna}")
check("EL LOCALISMO PRODUCE UN FALSO OPORTUNO y esta entrega NO lo arregla todavía",
      qro.oportuna is True and lfp.oportuna is False,
      "→ va en la entrega 2: cuarentena de la regla sin fundamento")
check("la regla sin fundamento sigue en el catálogo (hay que sacarla)",
      f0.REGLAS_SURTE["tja_qro_boletin"].fundamento == "")

print("\n" + "═"*92)
print(f"RESULTADO: {'TODAS LAS COMPROBACIONES PASAN' if not FALLOS else 'FALLAN ' + str(len(FALLOS))}")
for f in FALLOS: print("   ✗", f)
print("═"*92)
