# -*- coding: utf-8 -*-
"""Antecedentes sin enumerar y en prosa (2-oct-2026).

David: «debemos quitar la enumeración de antecedentes». Detrás de la bandera
`antecedentes_en_prosa`. Esto comprueba:

  1 · la limpieza del número es ESTRECHA: quita «1. » y «2) » al abrir el
      párrafo y nada más («15 de marzo…», «1.5», los años, se quedan);
  2 · la fórmula de entrada: la que va sola se borra, la fundida con el primer
      hecho se queda y el documento no añade la suya;
  3 · con la bandera, el prompt pide prosa encadenada, sin números, sin fórmula
      de entrada, de 3 a 6 párrafos y 350-450 palabras, y CONSERVA el cierre
      «en qué paró» y la variante de cumplimiento;
  4 · la síntesis moderna y la regla de remisión del estudio moderno, sin
      números;
  5 · `parrafos_antecedentes` y el documento generado: sin números, una sola
      entrada, el último hecho conservado y los efectos numerados intactos;
  6 · LA REGLA DE ORO: con la bandera apagada, los textos son los de la base
      (b89e049) letra por letra y el documento numera como antes;
  7 · el aviso de fechas recibe los antecedentes como texto, no letra por letra.

    .venv/bin/python test_antecedentes_prosa.py
"""
import datetime as _dt
import os
import re
import subprocess
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.pop("ANTECEDENTES_EN_PROSA", None)

import contexto_taller as ct
import origen_acto as oa
import documento_generado as dg
import fases123_pipeline as fp
import fases123_resumenes as fr
import formato_sentencia as fs

FALLOS = []
AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = "b89e049"


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


def encendida():
    ct.poner(True, {"banderas": {"antecedentes_en_prosa": True}}, pruebas=True)
    ct.poner_origen(None)


def apagada():
    ct.poner(True, {"banderas": {"antecedentes_en_prosa": False}}, pruebas=True)
    ct.poner_origen(None)


def de_la_base(modulo: str):
    """El módulo tal como estaba en la base, para comparar letra por letra."""
    try:
        src = subprocess.run(["git", "show", f"{BASE}:{modulo}.py"], cwd=AQUI,
                             capture_output=True, text=True, check=True).stdout
    except Exception:
        return None
    m = types.ModuleType(f"_base_{modulo}")
    m.__file__ = os.path.join(AQUI, f"{modulo}.py")
    exec(compile(src, m.__file__, "exec"), m.__dict__)
    return m


# ═══════════════════════════════════════════════════════════════════════════
print("\n1 · LA LIMPIEZA DEL NÚMERO ES ESTRECHA")
ok(fr.sin_numeracion(["1. Por auto de…", "2) En proveído de…", "10. Seguido el juicio…"])
   == ["Por auto de…", "En proveído de…", "Seguido el juicio…"], "quita «1. », «2) » y «10. »")
ok(fr.sin_numeracion(["15 de marzo de dos mil veinticuatro, el actor demandó."])
   == ["15 de marzo de dos mil veinticuatro, el actor demandó."], "no toca «15 de marzo…»")
ok(fr.sin_numeracion(["1.5 millones de pesos se reclamaron."]) == ["1.5 millones de pesos se reclamaron."],
   "no toca «1.5 millones»")
ok(fr.sin_numeracion(["2024. Año en que se presentó."]) == ["2024. Año en que se presentó."],
   "no toca un año (cuatro cifras)")
ok(fr.sin_numeracion(["El actor citó el artículo 1. del código."])
   == ["El actor citó el artículo 1. del código."], "sólo al abrir el párrafo")
ok(fr.sin_numeracion(["", "  ", "3. Algo"]) == ["Algo"], "los vacíos se van")

print("\n2 · LA FÓRMULA DE ENTRADA, UNA VEZ")
_sola = ("Para contextualizar el estudio de los motivos de disenso, es necesario relatar los "
         "siguientes antecedentes:")
_fundida = ("Para contextualizar el estudio de los motivos de disenso, conviene precisar que por "
            "escrito presentado el quince de marzo de dos mil veinticuatro, Fulano demandó de "
            "Mengano el pago de pesos.")
ok(fr.es_solo_entrada(_sola) and fr.es_solo_entrada("Previo al análisis de los conceptos de "
                                                     "violación, es menester relatar los antecedentes."),
   "la entrada sola se reconoce")
ok(not fr.es_solo_entrada(_fundida), "la entrada fundida con un hecho (con fecha) no es «sola»")
ok(not fr.es_solo_entrada("Por auto de cinco de mayo se admitió la demanda."), "un hecho no es entrada")
_ps, _trae = fr.antecedentes_en_prosa([_sola, "1. Por auto de…", "2. Seguido el juicio…"])
ok(_ps == ["Por auto de…", "Seguido el juicio…"] and not _trae,
   "la entrada sola se borra y los números se quitan")
_ps, _trae = fr.antecedentes_en_prosa(["1. " + _fundida, "2. Seguido el juicio…"])
ok(_ps[0] == _fundida and _trae, "la fundida se queda y avisa que ya trae entrada")

# ═══════════════════════════════════════════════════════════════════════════
print("\n3 · EL PROMPT, CON LA BANDERA")
encendida()
_on = fr.instrucciones_antecedentes("amparo_directo")
ok(_on.startswith("ANTECEDENTES\n") and "QUINTO" not in _on,
   "el rótulo sin ordinal (lo calcula el documento)")
ok("Un hecho procesal por" not in _on and "nada de encadenar" not in _on
   and "PÁRRAFOS CORTOS" not in _on, "ya no pide un hecho por párrafo")
ok("EN PROSA ENCADENADA Y SIN ENUMERAR" in _on and "Ni números" in _on and "viñetas" in _on
   and "incisos" in _on, "pide prosa encadenada sin números, viñetas ni incisos")
ok("entre 3 y 6 párrafos" in _on and "350 a 450 palabras" in _on and "645" not in _on
   and " 17 " not in _on, "más corto: 3 a 6 párrafos y 350-450 palabras (no 17 y 645)")
ok("SIN FÓRMULA DE ENTRADA" in _on and "ARRANCA con una de estas fórmulas" not in _on,
   "la fórmula de entrada se la queda el documento")
ok("SÓLO LOS HECHOS QUE HACEN FALTA" in _on, "sólo los hechos que hacen falta para la solución")
ok("Y EL ÚLTIMO PÁRRAFO DICE EN QUÉ PARÓ" in _on and _on.endswith(
    "donde debía ir el verbo.\n- NO opines, NO califiques y NO adelantes el estudio."),
   "el cierre «en qué paró» se conserva entero")
ok("«Inconforme con esa resolución…»" in _on and "ASÍ SE ENLAZAN" in _on,
   "los conectores del corpus siguen ahí")
ok(not re.search(r"«\d{1,2}\.", _on), "ningún ejemplo numerado en el prompt")
_pa = fp.prompt_antecedentes("DOC", "amparo_directo")
ok(_pa.endswith("en prosa, un párrafo por línea y sin numerarlos. Sólo el apartado."),
   "la última orden del prompt repite que no se numera")
# En cumplimiento, el hilo del amparo anterior se sigue pidiendo, sin «un hecho por párrafo».
ct.poner(True, {"banderas": {"antecedentes_en_prosa": True}}, pruebas=True)
ct.poner_origen(oa.origen("Primera Sala Civil del Tribunal Superior de Justicia", "", "",
                          "VISTOS. La presente se dicta en cumplimiento de la ejecutoria dictada en el "
                          "juicio de amparo directo civil 590/2023, que concedió el amparo."))
_cu = fr.instrucciones_antecedentes("amparo_directo")
ok("EN CUMPLIMIENTO de la ejecutoria del\n  amparo directo civil 590/2023. Cuéntalo en su orden y "
   "encadenado:" in _cu and "la sentencia nueva" in _cu and "un hecho por párrafo" not in _cu,
   "cumplimiento: el amparo anterior y la sentencia nueva, encadenados")
encendida()

print("\n4 · LA MODERNA, CON LA BANDERA")
_si = fs.prompt_sintesis("A", "B", "C", [{"pregunta": "p"}], [], "la Sala", "conceptos de violación", 2)
ok('"antecedentes": ["párrafo", "párrafo"]' in _si and '"1. …"' not in _si,
   "el ejemplo del JSON ya no lleva «1. …»")
ok("numerados, con sus fechas" not in _si and "sin numerar" in _si and "en qué paró" in _si,
   "la regla 4 pide prosa sin numerar y conserva el último hecho")
for _v in ("v1", "v2"):
    _fm = fs.forma_del_estudio(fs.MODERNA, "conceptos de violación", "concepto de violación",
                               "quejosa", "fundado", 1600, _v)
    ok("NO REMITAS A LOS ANTECEDENTES POR UN NÚMERO: no llevan ninguno" in _fm
       and "su numeración cambia" not in _fm and "antecedente\n  7" not in _fm,
       f"forma moderna {_v}: la razón de no remitir por número, reescrita y sin el ejemplo")

print("\n5 · LA FUENTE COMÚN Y EL DOCUMENTO, CON LA BANDERA")
_f = fp.Fases123(antecedentes="ANTECEDENTES\n1. Por auto de cinco de mayo se admitió.\n"
                              "2. Seguido el juicio, el Juez absolvió.")
ok(_f.parrafos_antecedentes() == ["Por auto de cinco de mayo se admitió.",
                                  "Seguido el juicio, el Juez absolvió."],
   "parrafos_antecedentes: sin el rótulo a secas y sin números")
_f2 = fp.Fases123(antecedentes="QUINTO. ANTECEDENTES\n1. Por auto de cinco de mayo se admitió.")
ok(_f2.parrafos_antecedentes() == ["Por auto de cinco de mayo se admitió."],
   "parrafos_antecedentes: también con el rótulo de siempre")

try:
    import fase0_oportunidad as _f0
    from docx import Document
    _HAY_DOCX = True
except Exception as _ex:  # pragma: no cover
    _HAY_DOCX = False
    ok(False, f"no se pudo importar lo del documento: {_ex}")

ANT = [_sola,
       "1. Por escrito presentado el quince de marzo de dos mil veinticuatro, Fulano demandó de "
       "Mengano el pago de pesos; por auto de veinte de marzo se admitió la demanda.",
       "2. Seguido el juicio por sus etapas, el quince de mayo de dos mil veinticinco el Juez "
       "dictó sentencia en la que absolvió al demandado."]
EFECTOS = ["Es fundado el concepto de violación.", "EFECTOS DE LA CONCESIÓN",
           "1. Deje insubsistente la sentencia reclamada.",
           "2. Dicte otra en la que valore la pericial."]


def componer(antecedentes, salida):
    _est = dg.Estructura(apertura="V.", visto="para resolver.",
                         resultandos=[{"titulo": "Presentación de la demanda",
                                       "texto": "Se presentó la demanda."}],
                         competencia="", existencia="", procedencia="")
    _c = _f0.computar(_dt.date(2026, 3, 2), _dt.date(2026, 3, 9), plazo=15)
    dg.componer({"tipo_asunto": "amparo_directo", "numero": "93/2026",
                 "encabezado": "AMPARO DIRECTO CIVIL 93/2026", "quejoso": "Juan Pérez",
                 "responsable": "Sala Civil", "magistrado": "M", "secretario": "S",
                 "tribunal": "Tercer Tribunal Colegiado", "ciudad": "Querétaro"},
                _est, _c, _f0.fecha_en_letra, salida, antecedentes=antecedentes,
                estudio=list(EFECTOS), calificaciones=["fundado"], tipo_asunto="amparo_directo")
    ps = [q.text for q in Document(salida).paragraphs if q.text.strip()]
    k = [i for i, t in enumerate(ps) if re.match(r"^[A-ZÉ]+\.\s*Antecedentes\.", t)]
    if not k:
        return ps, None, []
    fin = next((j for j in range(k[0] + 1, len(ps))
                if re.match(r"^(?:PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|SÉPTIMO|OCTAVO)\.", ps[j])),
               len(ps))
    return ps, ps[k[0]], ps[k[0] + 1:fin]


if _HAY_DOCX:
    try:
        encendida()
        _ps, _cab, _cuerpo = componer(list(ANT), "/tmp/_antecedentes_prosa_on.docx")
        ok(_cab is not None, "el considerando «Antecedentes.» se conserva")
        _todo = " ".join([_cab or ""] + _cuerpo)
        ok(len(re.findall(r"Previo al análisis|Para contextualizar", _todo)) == 1,
           "una sola fórmula de entrada (la del documento)")
        ok(bool(_cuerpo) and not any(re.match(r"^\d{1,2}[.)]\s", t) for t in _cuerpo),
           "ningún antecedente numerado")
        ok(bool(_cuerpo) and "absolvió al demandado" in _cuerpo[-1], "el último hecho, «en qué paró», sigue al final")
        ok(any(t.startswith("1. Dejar insubsistente") or t.startswith("1. Deje insubsistente") for t in _ps)
           and any(t.startswith("2. Dict") for t in _ps), "los efectos siguen numerados")
        # Fundida: se queda la del modelo y el documento no añade la suya.
        _ps2, _cab2, _cuerpo2 = componer(["1. " + _fundida, "2. Seguido el juicio, el Juez absolvió."],
                                         "/tmp/_antecedentes_prosa_fund.docx")
        _todo2 = " ".join([_cab2 or ""] + _cuerpo2)
        ok(len(re.findall(r"Previo al análisis|Para contextualizar", _todo2)) == 1
           and _cuerpo2 and _cuerpo2[0].startswith("Para contextualizar"),
           "entrada fundida: una sola, la del modelo, sin número")

        print("\n6 · LA REGLA DE ORO: APAGADA, COMO EN LA BASE")
        apagada()
        _ps0, _cab0, _cuerpo0 = componer(list(ANT), "/tmp/_antecedentes_prosa_off.docx")
        ok(_cab0 is not None and "Previo al análisis de los planteamientos" in _cab0,
           "documento: la entrada del compositor, como antes")
        # Así sale en la base, y es justo lo que David vio: la entrada del
        # modelo numerada «1.», detrás de la del compositor, y otro «1.» en el
        # primer hecho porque el modelo ya lo traía numerado. Apagada, igual.
        ok([t.split(" ", 1)[0] for t in _cuerpo0] == ["1.", "1.", "2."]
           and _cuerpo0[0].startswith("1. Para contextualizar"),
           "documento: numerados como en la base (con su «1.» doble)")
    except Exception as ex:  # pragma: no cover
        ok(False, f"el documento no se pudo componer: {type(ex).__name__}: {ex}")
else:
    print("\n6 · LA REGLA DE ORO: APAGADA, COMO EN LA BASE")

apagada()
_f3 = fp.Fases123(antecedentes="QUINTO. ANTECEDENTES\n1. Por auto de cinco de mayo se admitió.\n"
                               "ANTECEDENTES")
ok(_f3.parrafos_antecedentes() == ["1. Por auto de cinco de mayo se admitió.", "ANTECEDENTES"],
   "parrafos_antecedentes: como antes (el número se queda)")
ok(fp.prompt_antecedentes("DOC", "amparo_directo").endswith(
    "Escribe el apartado de antecedentes, un párrafo por línea. Sólo el apartado."),
   "prompt_antecedentes: la última orden de siempre")
_b_fr = de_la_base("fases123_resumenes")
_b_fs = de_la_base("formato_sentencia")
if _b_fr is None or _b_fs is None:
    ok(False, f"no se pudo leer la base {BASE} con git")
else:
    _JUZ = ("Juzgado Segundo de Primera Instancia Especializado en Oralidad Mercantil del "
            "Distrito Judicial de Querétaro")
    _CUMPL = ("VISTOS. La presente se dicta en cumplimiento de la ejecutoria dictada en el juicio "
              "de amparo directo civil 590/2023, que concedió el amparo.")
    for _nombre, _origen in (("sin origen", None),
                             ("única instancia y cumplimiento",
                              oa.origen(_JUZ, "…el juicio oral mercantil…", "", _CUMPL))):
        for _casa in (True, False):
            if _casa:
                apagada()
            else:
                ct.poner(False)
            ct.poner_origen(_origen)
            _dist = []
            for t in ("amparo_directo", "amparo_revision", "queja", "revision_fiscal", ""):
                if fr.instrucciones_antecedentes(t) != _b_fr.instrucciones_antecedentes(t):
                    _dist.append(f"instrucciones {t or 'sin tipo'}")
            if (fs.prompt_sintesis("A", "B", "C", [{"pregunta": "p"}], [], "la Sala",
                                   "conceptos de violación", 2)
                    != _b_fs.prompt_sintesis("A", "B", "C", [{"pregunta": "p"}], [], "la Sala",
                                             "conceptos de violación", 2)):
                _dist.append("síntesis")
            for _fm in (fs.MODERNA, fs.ESTANDAR):
                for _v in ("v1", "v2"):
                    _a = fs.forma_del_estudio(_fm, "agravios", "agravio", "recurrente", "infundados", 900, _v)
                    _b = _b_fs.forma_del_estudio(_fm, "agravios", "agravio", "recurrente", "infundados", 900, _v)
                    if _a != _b:
                        _dist.append(f"forma {_fm} {_v}")
            _quien = "cuenta de casa con la bandera apagada" if _casa else "cuenta de fuera"
            ok(not _dist, f"{_nombre}, {_quien}: idéntico a la base"
                          f"{(' — difieren: ' + ', '.join(_dist)) if _dist else ''}")

print("\n7 · EL AVISO DE FECHAS LEE LOS ANTECEDENTES COMO TEXTO")
_src = open(os.path.join(AQUI, "redactor_adelanto.py"), encoding="utf-8").read()
ok('"\\n".join(getattr(r.fases, "antecedentes", None) or [])' not in _src,
   "ya no se une letra por letra la cadena de los antecedentes")
_une = (lambda _a: "\n".join(map(str, _a)) if isinstance(_a, (list, tuple)) else str(_a or ""))
ok(_une("cinco de mayo") == "cinco de mayo" and _une(["a", "b"]) == "a\nb" and _une(None) == "",
   "la cadena pasa entera; una lista, unida por renglones")

ct.poner(False)
ct.poner_origen(None)
print()
if FALLOS:
    print("FALLAS:\n  " + "\n  ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
