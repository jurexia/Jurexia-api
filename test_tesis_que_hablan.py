# -*- coding: utf-8 -*-
"""CITAR LA TESIS, HACERLA HABLAR Y APLICARLA (AR 631/2025, 28-sep-2026).

David: «Después de citar tesis hay que hacerlas hablar. Y después aplicarlas al
caso concreto. Es lo que se estila.» La técnica se midió a mano en diez
sentencias de la Corte y cinco engroses del colegiado
(redactor-sentencias/scjn_estilo/): ninguna de sus 101 citas queda sin regla ni
aplicación; en la Solución del 631 quedaban seis de nueve.

Lo que se prueba, sin modelo y sin red:
  1. el control 1-duodecies contra los textos anotados: cero avisos en la Corte
     y en los engroses, entre seis y ocho en el 631 (si el recurso está en disco);
  2. el mismo control con textos en la forma que escribe el modelo, en las dos
     direcciones: acusa la fila de rubros, la cita contradictoria y la tesis de la
     parte reciclada, y NO acusa la técnica bien hecha;
  3. el compositor ya no convierte en «Sirve de apoyo» la cita que el modelo
     distinguió (era la causa de la cita contradictoria del 631);
  4. `_sin_eco` no borra la explicación propia de una tesis cuyo texto bajó al pie;
  5. la v4 lleva la técnica por sustitución y la v2/v3 no se mueven;
  6. el molde de forma ya no se corta antes de la aplicación.

    .venv/bin/python test_tesis_que_hablan.py
"""
import glob
import json
import os
import re
import sys

os.environ.pop("ESTUDIO_PROMPT", None)

import documento_generado as dg
import fase6_estudio as f6
import fase_precedente as fp
import plan_estudio as pe

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


def defectos(malos, reg):
    return next((m["defectos"] for m in malos if m["registro"] == reg), [])


# ═══════════════════════════════════════════════════════════════════════════
print("\n1 · EL CONTROL CONTRA LOS TEXTOS ANOTADOS (calibración)")
BASE = "/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/redactor-sentencias"
LIMPIO = os.path.join(BASE, "scjn_estilo", "anotacion", "limpio")
CASOS = os.path.join(BASE, "corpus", "casos")
R631 = os.path.join(BASE, "regresion-631-2025", "trabajo", "631-2025-verificacion.txt")
SCJN10 = ["19-2024", "28-2022", "560-2022", "1774-2021", "202-2024", "348-2013", "15-2016",
          "2234-2017", "1031-2015", "13-2017"]
TCC5 = ["ADC_642_2024_ORD_CIVIL_SOBRE_REIVINDICACIO_N_8eff8c.json", "ADC_174_2026_aabe46.json",
        "ADA_702_2022_INFRACCIO_N_4e4f48.json", "ADA_767_2025_a82a87.json", "ADC_43_2025_120576.json"]


def scjn_texto(ruta):
    t = re.sub(r"⟨p\d+⟩", " ", open(ruta, encoding="utf-8").read())
    return "\n".join(" ".join(p.split()) for p in re.split(r"\n(?=¶\d+\.)", t) if p.strip())


def tcc_partes(oro):
    ls = [x.strip() for x in oro.split("\n") if x.strip()]
    i_sol = next(i for i, x in enumerate(ls) if x.lower().startswith("solución"))
    i_con = next((i for i, x in enumerate(ls) if x.lower().startswith(("conceptos de violación", "agravios"))), None)
    fin = next((i for i, x in enumerate(ls) if i > i_sol and re.match(r"^(S[ÉE]PTIMO|OCTAVO|NOVENO)\.\s", x)), len(ls))
    res = "\n".join(ls[i_con + 1:i_sol]) if i_con is not None and i_con < i_sol else ""
    return "\n".join(ls[i_sol + 1:fin]), res


def aviso_de(estudio, resumen=""):
    regs, rubs = f6.registros_de_la_parte(resumen)
    return f6.tesis_que_no_hablan(estudio, None, regs, rubs)


if os.path.isdir(LIMPIO) and os.path.isdir(CASOS) and os.path.isfile(R631):
    arch = [a for a in sorted(glob.glob(os.path.join(LIMPIO, "*.txt"))) if not a.endswith(".notas.txt")]
    n_scjn = 0
    for clave in SCJN10:
        ruta = next(a for a in arch if f"__{clave}_" in os.path.basename(a))
        n_scjn += len(aviso_de(scjn_texto(ruta)))
    ok(n_scjn == 0, f"las diez sentencias de la Corte anotadas: cero avisos ({n_scjn})")
    n_tcc = 0
    for f in TCC5:
        est, res = tcc_partes(json.load(open(os.path.join(CASOS, f)))["oro"])
        n_tcc += len(aviso_de(est, res))
    ok(n_tcc == 0, f"los cinco engroses del colegiado anotados: cero avisos ({n_tcc})")
    ls = [x.strip() for x in open(R631, encoding="utf-8").read().split("\n")]
    sol = "\n".join(x for x in ls[ls.index("Solución") + 1:
                                 next(i for i, x in enumerate(ls) if x.startswith("Por lo expuesto"))] if x)
    agr = "\n".join(x for x in ls[ls.index("Agravios") + 1:ls.index("Solución")] if x)
    m631 = aviso_de(sol, agr)
    ok(6 <= len(m631) <= 8, f"la Solución del AR 631/2025 (verificación): entre seis y ocho ({len(m631)})")
    ok(all("apilada" in defectos(m631, r) for r in ("170307", "162826", "164590")),
       "631: la pila de cuatro rubros sin una palabra entre ellos")
    ok("contradictoria" in defectos(m631, "2015679"), "631: la 2015679, «sirve de apoyo» y «no resulta aplicable»")
    ok(all("de la parte" in defectos(m631, r) for r in ("177529", "2023791")),
       "631: las dos de la recurrente del final, recicladas como apoyo")
    ok(not defectos(m631, "168958"), "631: la 168958 sí habla (sus límites se dijeron dos párrafos antes)")
    n_resto = 0
    for ruta in arch:
        if not any(f"__{c}_" in os.path.basename(ruta) for c in SCJN10):
            n_resto += len(aviso_de(scjn_texto(ruta)))
    ok(n_resto == 0, f"fuera de la muestra: los otros {len(arch) - 10} extractos de la Corte, cero avisos ({n_resto})")
else:
    print("   (sin el recurso en disco: se omite la calibración; el resto de la prueba sí corre)")

# ═══════════════════════════════════════════════════════════════════════════
print("\n2 · EL CONTROL EN LA FORMA DEL MODELO (las dos direcciones)")
R = {
    "2026918": "COSA JUZGADA Y SUS EFECTOS DIRECTO Y REFLEJO. DIFERENCIAS Y REQUISITOS PARA SU ACTUALIZACIÓN",
    "176546": "FUNDAMENTACIÓN Y MOTIVACIÓN DE LAS RESOLUCIONES JURISDICCIONALES, DEBEN ANALIZARSE A LA LUZ DE "
              "LOS ARTÍCULOS 14 Y 16 DE LA CONSTITUCIÓN POLÍTICA DE LOS ESTADOS UNIDOS MEXICANOS, RESPECTIVAMENTE",
    "170307": "FUNDAMENTACIÓN Y MOTIVACIÓN. LA DIFERENCIA ENTRE LA FALTA Y LA INDEBIDA SATISFACCIÓN DE AMBOS "
              "REQUISITOS CONSTITUCIONALES TRASCIENDE AL ORDEN EN QUE DEBEN ESTUDIARSE LOS CONCEPTOS DE VIOLACIÓN "
              "Y A LOS EFECTOS DEL FALLO PROTECTOR",
    "162826": "FUNDAMENTACIÓN Y MOTIVACIÓN. ARGUMENTOS QUE DEBEN EXAMINARSE PARA DETERMINAR LO FUNDADO O "
              "INFUNDADO DE UNA INCONFORMIDAD CUANDO SE ALEGA LA AUSENCIA DE AQUÉLLA O SE TACHA DE INDEBIDA",
    "164590": "FUNDAMENTACIÓN Y MOTIVACIÓN. PARA CUMPLIR CON ESTAS GARANTÍAS, EL JUEZ DEBE RESOLVER CON BASE EN "
              "EL SUSTENTO LEGAL CORRECTO, AUN CUANDO EXISTA ERROR U OMISIÓN EN LA CITA DEL PRECEPTO",
    "2015679": "DERECHO HUMANO A LA IGUALDAD JURÍDICA. RECONOCIMIENTO DE SU DIMENSIÓN SUSTANTIVA O DE HECHO EN "
               "EL ORDENAMIENTO JURÍDICO MEXICANO",
    "177529": "PROCEDIMIENTO SEGUIDO EN UNA VÍA INCORRECTA. POR SÍ MISMO CAUSA AGRAVIO AL DEMANDADO Y, POR "
              "ENDE, CONTRAVIENE SU GARANTÍA DE SEGURIDAD JURÍDICA",
}
TESIS = [{"registro": k, "rubro": v, "tipo": "JURISPRUDENCIA", "instancia": "Primera Sala",
          "obligatoria": True, "texto": " ".join(["regla"] * 150)} for k, v in R.items()]
PARTE = {"170307", "162826", "164590", "2015679", "177529"}
RESUMEN = ("En apoyo invoca la jurisprudencia 1.3o.C. J/47, registro digital 170307; la IV.2o.C. J/12, "
           "registro digital 162826; la VI.2o.C. J/318, registro digital 164590; la 1a./J. 125/2017, "
           "registro digital 2015679, y la 1a./J. 74/2005, registro digital 177529.")


def cita(reg, verbo="Sirve de apoyo la jurisprudencia"):
    return f"{verbo} de registro digital {reg}, de rubro «{R[reg]}»."


MAL = "\n".join([
    "Es fundada la inconformidad relacionada con la falta de aplicación de los artículos 49 y 57 del "
    "Código de Procedimientos Civiles del Estado de Querétaro al dato concreto de la escritura.",
    cita("176546"), cita("170307"), cita("162826"), cita("164590"),
    "No se pierde de vista que la parte recurrente sostiene que la interlocutoria sólo determinó la "
    "legitimación para ejecutar, sin modificar las prestaciones de la condena.",
    cita("2015679"),
    "La jurisprudencia en cita no resulta aplicable, porque exige una comparación jurídicamente "
    "relevante y la parte recurrente no la propone.",
    "El argumento relativo al consentimiento es inoperante porque no combate la consideración toral.",
    cita("177529"),
    "Por ende, el vicio advertido alcanza la consideración que sostuvo la concesión y procede revocar.",
])
m_mal = f6.tesis_que_no_hablan(MAL, TESIS, PARTE, [])
ok(all("apilada" in defectos(m_mal, r) for r in ("176546", "170307", "162826", "164590")),
   "la fila de cuatro rubros sin prosa entre ellos ni detrás: «apilada»")
ok(all("de la parte" in defectos(m_mal, r) for r in ("170307", "162826", "164590", "177529")),
   "las de la parte con «Sirve de apoyo» y nada que las conteste: «de la parte»")
ok(defectos(m_mal, "2015679") == ["contradictoria"],
   "«Sirve de apoyo» y, en el renglón siguiente, «no resulta aplicable»: «contradictoria»")

BIEN = "\n".join([
    "Sobre el primer agravio, en el que la parte recurrente sostiene que la sustitución no alteró la cosa "
    "juzgada, se considera fundado.",
    "La cosa juzgada tiene un efecto directo, que impide volver a juzgar lo mismo entre las mismas partes, "
    "y un efecto reflejo, que obliga a tomar lo resuelto como premisa de un juicio distinto; para que se "
    "actualice cualquiera de los dos se requiere identidad de las partes, de la cosa y de la causa, y sus "
    "requisitos no se satisfacen cuando sólo cambia quién está legitimado para ejecutar. Sirve de apoyo "
    "la jurisprudencia de registro digital 2026918, de rubro «" + R["2026918"] + "».",
    "Trasladada esa regla al caso, la resolución de ocho de abril de dos mil veinticuatro no alteró la "
    "cosa ni la causa de la condena: la rescisión, la entrega y las rentas siguen siendo las mismas, y sólo "
    "cambió la persona que las cobra tras la escritura 20,128, de modo que ni el efecto directo ni el "
    "reflejo de la cosa juzgada se tocan.",
    "La recurrente invoca la jurisprudencia de registro digital 177529, de rubro «" + R["177529"] + "», "
    "que resolvía un juicio seguido en una vía distinta de la que la ley prevé para la acción ejercida; "
    "aquí nadie discute la vía del juicio de origen, sino quién puede ejecutar su sentencia, de modo que "
    "ese supuesto no corresponde al de este asunto.",
    "Respecto de la jurisprudencia de registro digital 2015679, de rubro «" + R["2015679"] + "», no "
    "resulta aplicable, porque exige comparar dos situaciones jurídicas equivalentes con trato distinto, y "
    "la recurrente no identifica término de comparación alguno en las constancias.",
])
m_bien = f6.tesis_que_no_hablan(BIEN, TESIS, PARTE, [])
ok(m_bien == [], "la técnica bien hecha —regla, aplicación, distinción de las de la parte—: cero avisos"
   + (f" ({m_bien})" if m_bien else ""))

CORTE = "\n".join([
    "En opinión de la quejosa, en este caso se actualiza la responsabilidad solidaria del hospital, en "
    "atención a las tesis que invoca y que se intitulan: «" + R["170307"] + "»; «" + R["162826"] + "» y «"
    + R["164590"] + "».",
    "Como se advierte en el contenido de dichos criterios, en todos ellos se parte de la base de que la "
    "fundamentación exige el precepto correcto; debe analizarse entonces si en este asunto eso ocurrió.",
])
ok(f6.tesis_que_no_hablan(CORTE, TESIS, PARTE, []) == [],
   "la pila de la Corte con su premisa común enunciada enseguida (JRC AD 13/2017, ¶72-73): cero avisos")
NARRA = ("La recurrente cita las tesis de rubro «" + R["170307"] + "», «" + R["162826"] + "», «"
         + R["164590"] + "».\nEl Juzgado de Distrito concedió el amparo.")
ok(not any("apilada" in d["defectos"] for d in f6.tesis_que_no_hablan(NARRA, TESIS, set(), [])),
   "la pila que se NARRA (lo que citó otro) no es una pila del estudio")
ok(not f6.tesis_que_no_hablan("Sirve de apoyo el criterio de registro 2026918:\nLa cosa juzgada, en su "
                              "efecto directo y reflejo, exige identidad de partes, cosa y causa; aquí "
                              "sólo cambió quien ejecuta.", TESIS, set(), []),
   "sin rubro en el texto, el del material por su registro (y la regla dicha después): sin aviso")

regs, rubs = f6.registros_de_la_parte(RESUMEN, [{"texto": "cita la tesis registro 2023791", "cita": "",
                                                  "anclas": ["registro 187528"]}])
ok({"170307", "162826", "164590", "2015679", "177529", "2023791", "187528"} <= regs,
   "los registros de la parte salen del resumen de agravios y del inventario")

MAT = f6.Material(tipo_asunto="amparo_revision", materia="civil", formato="estandar",
                  n_planteamientos=1, variante="v4", tesis=TESIS)
C1 = [f6.Criterio("¿La sustitución alteró la cosa juzgada?", "fundado", "r", "principal")]
av_mal = f6.revisar(MAL, C1, MAT, resumen_conceptos=RESUMEN)
av_bien = f6.revisar(BIEN, C1, MAT, resumen_conceptos=RESUMEN)
_cnh = [a for a in av_mal if a.startswith("CRITERIOS QUE NO HABLAN")]
ok(len(_cnh) == 1 and "EN FILA SIN REGLA" in _cnh[0] and "ANUNCIADO COMO APOYO Y DECLARADO INAPLICABLE" in _cnh[0]
   and "DE LA PARTE, COMO APOYO" in _cnh[0] and "2015679" in _cnh[0],
   "revisar(): un aviso con los tres defectos y sus registros")
ok(not any(a.startswith("CRITERIOS QUE NO HABLAN") for a in av_bien), "revisar(): nada con la técnica bien hecha")
ok(not any("de la parte" in a.lower() and "COMO APOYO" in a
           for a in f6.revisar(MAL.replace("\n" + cita("2015679"), ""), C1, MAT)),
   "sin resumen ni inventario, el control no dice que un criterio es de la parte")

# ═══════════════════════════════════════════════════════════════════════════
print("\n3 · EL COMPOSITOR YA NO CONVIERTE EN APOYO LO QUE SE DISTINGUIÓ")
from docx import Document


def componer(parrafos, tesis=TESIS):
    d = Document()
    notas = []
    dg._escribir_estudio(d, list(parrafos), tesis, notas)
    return [p.text for p in d.paragraphs if p.text.strip()], notas


_d, _n = componer(["Respecto de la jurisprudencia de registro 2015679, de rubro «" + R["2015679"] + "», no "
                   "resulta aplicable, porque exige una comparación jurídicamente relevante y la parte "
                   "recurrente no la propone."])
ok(_d[0].startswith("Respecto de la jurisprudencia de la Primera Sala") and "Sirve de apoyo" not in " ".join(_d),
   "«Respecto de…» se queda: ya no sale «Sirve de apoyo… 2015679» (631, líneas 112-114)")
ok(f6.tesis_que_no_hablan("\n".join(_d), TESIS, PARTE, []) == [],
   "y el documento compuesto ya no trae una cita contradictoria")
_d, _ = componer(["No resulta aplicable la jurisprudencia de registro 2015679, de rubro «" + R["2015679"]
                  + "», porque exige una comparación entre dos sujetos y la recurrente no la propone."])
ok(_d[0].startswith("No resulta aplicable la jurisprudencia de la Primera Sala")
   and _d[-1].startswith("Ello, porque exige"),
   "«No resulta aplicable…» se conserva y la media frase de detrás arranca con sujeto")
_d, _ = componer(["La recurrente invoca la jurisprudencia de registro 177529, de rubro «" + R["177529"]
                  + "», pero ese criterio resolvía la elección de la vía y aquí nadie la discute."])
ok(_d[0].startswith("La recurrente invoca la jurisprudencia") and _d[-1].startswith("Sin embargo, ese criterio"),
   "«La recurrente invoca…» se conserva; la adversativa de detrás, con su arranque")
_d, _ = componer(["La recurrente invoca la jurisprudencia de registro 177529, de rubro «" + R["177529"]
                  + "», que resolvía un juicio seguido en una vía distinta de la que la ley prevé."])
ok(_d[-1].startswith("Ese criterio resolvía un juicio"), "«…, que resolvía…»: el relativo es el criterio")
_d, _ = componer(["No resultan aplicables la jurisprudencia de registro 177529, de rubro «" + R["177529"]
                  + "», y la jurisprudencia de registro 2015679, de rubro «" + R["2015679"]
                  + "», porque ninguna resuelve la legitimación para ejecutar una sentencia firme."])
ok(_d[0].startswith("No resulta aplicable la jurisprudencia") and
   any(x.startswith("Tampoco resulta aplicable la jurisprudencia") for x in _d)
   and not any("sirve de apoyo" in x.lower() for x in _d),
   "la segunda de una negación encadenada sale «Tampoco…», no «También sirve de apoyo»")
_d, _ = componer(["Sirve de apoyo la jurisprudencia de registro 177529, de rubro «" + R["177529"] + "»."])
ok(_d[0].startswith("Sirve de apoyo la jurisprudencia de la Primera Sala"), "la fórmula de apoyo, como siempre")
ok(dg._verbo_de_enlace("No obstante, sirve de apoyo la jurisprudencia") == "Sirve de apoyo"
   and dg._verbo_de_enlace("La misma conclusión se obtiene de la jurisprudencia de la Primera Sala, de")
   == "Sirve de apoyo" and dg._verbo_de_enlace("La Primera Sala, en la jurisprudencia") == "Sirve de apoyo",
   "«no obstante» no niega, y un arranque que nombra órgano o no es fórmula sigue siendo «Sirve de apoyo»")
_ais = dict(TESIS[0], tipo="TESIS AISLADA", obligatoria=False)
ok(dg.anuncio_de(_ais, "No resulta aplicable la tesis").startswith("No resulta aplicable la tesis aislada")
   and "orientador" not in dg.anuncio_de(_ais, "No resulta aplicable la tesis")
   and "como criterio orientador" in dg.anuncio_de(_ais, "Sirve de apoyo la tesis"),
   "lo que se distingue no lleva «como criterio orientador»; lo que apoya, sí")
_d, _n = componer([cita("2026918")])
ok("de registro digital 2026918" in _d[0] and R["2026918"] in _d[1] and len(_d) == 2
   and any("Texto:" in x for x in _n),
   "en el cuerpo quedan el anuncio con su registro y el rubro; el texto largo, sólo en la nota")

# ═══════════════════════════════════════════════════════════════════════════
print("\n4 · _sin_eco NO BORRA LA EXPLICACIÓN DE UNA TESIS QUE BAJÓ AL PIE")
TEXTO_T = ("La cosa juzgada tiene un efecto directo que impide volver a juzgar lo mismo entre las "
           "mismas partes y un efecto reflejo que obliga a tomar lo resuelto como premisa de otro juicio. "
           + " ".join(["Relleno de la tesis para que su texto pase de ochenta palabras."] * 12))
PARAF = ("La cosa juzgada tiene un efecto directo que impide juzgar lo mismo entre las mismas partes y "
         "un efecto reflejo que la vuelve premisa de otro juicio distinto.")
COPIA = ("La cosa juzgada tiene un efecto directo que impide volver a juzgar lo mismo entre las mismas "
         "partes y un efecto reflejo que obliga a tomar lo resuelto como premisa de otro juicio.")
ok(dg._sin_eco(PARAF, TEXTO_T, al_pie=False) == "", "con el texto en el cuerpo, la paráfrasis pegada es eco (como antes)")
ok(dg._sin_eco(PARAF, TEXTO_T, al_pie=True) == PARAF, "con el texto al pie, la regla dicha con palabras propias se queda")
ok(dg._sin_eco(COPIA, TEXTO_T, al_pie=True) == "", "con el texto al pie, la copia casi literal sí se va")
_tl = [dict(TESIS[0], texto=TEXTO_T)]
_d, _ = componer([cita("2026918"), PARAF + " Aplicado al caso, la escritura 20,128 sólo cambió quién ejecuta."], _tl)
ok(any(PARAF[:60] in x for x in _d), "compuesto: la explicación que sigue a la cita sobrevive al compositor")
ok(dg._texto_al_pie(_tl[0]) and not dg._texto_al_pie(dict(TESIS[0], texto="breve")),
   "`_texto_al_pie` decide igual que `escribir_cita`")
# EL AVISO 1-nonies NO PROMETE UN BORRADO QUE YA NO OCURRE. Con el texto al pie
# la paráfrasis se conserva; con el texto corto, en el cuerpo, se sigue borrando.
_corta = ("La cosa juzgada tiene un efecto directo que impide volver a juzgar lo mismo entre las mismas "
          "partes y un efecto reflejo que obliga a tomar lo resuelto como premisa de otro juicio.")
_rep = cita("2026918") + "\n" + COPIA + " " + COPIA
for _txt_t, _espera in ((TEXTO_T, "se vuelven a contar"), (_corta, "BORRÓ al componer")):
    _m9 = f6.Material(tipo_asunto="amparo_revision", materia="civil", formato="estandar", n_planteamientos=1,
                      variante="v4", tesis=[dict(TESIS[0], texto=_txt_t)])
    _av9 = [a for a in f6.revisar(_rep, C1, _m9) if "2026918" in a and ("BORRÓ" in a or "vuelven a contar" in a)]
    ok(len(_av9) == 1 and _espera in _av9[0],
       f"1-nonies: {'texto al pie → se conserva y se dice' if _txt_t is TEXTO_T else 'texto en el cuerpo → se borró'}")

# ═══════════════════════════════════════════════════════════════════════════
print("\n5 · LA v4 LLEVA LA TÉCNICA; LA v2 Y LA v3 NO SE MUEVEN")
GUION = "APARTADO 1 · A1.a A1.b\n  JERARQUÍA DEL PROBLEMA P1 · DECIDE A1.b"
GUION_SJ = "APARTADO 1 · A1.a A1.b"
SEGS = [{"id": "A1.a", "concepto": 1, "parrafo": 0, "texto": "invoca la jurisprudencia registro 177529",
         "cita": "", "anclas": []},
        {"id": "A1.b", "concepto": 1, "parrafo": 0, "texto": "la sustitución no alteró la cosa juzgada",
         "cita": "", "anclas": []}]
for tipo, rec in (("amparo_directo", False), ("amparo_revision", True)):
    for forma in ("estandar", "moderna"):
        def _m(var, inv=True):
            return f6.Material(tipo_asunto=tipo, materia="civil", formato=forma, n_planteamientos=1,
                               variante=var, inventario=SEGS if inv else [])
        p3 = f6.prompt_estudio("ACTO", "CONC", C1, _m("v3"), es_recurso=rec)
        p4 = f6.prompt_estudio("ACTO", "CONC", C1, _m("v4"), es_recurso=rec, guion=GUION)
        p4sj = f6.prompt_estudio("ACTO", "CONC", C1, _m("v4"), es_recurso=rec, guion=GUION_SJ)
        _hit = [v for v, _ in f6._REEMPLAZOS_TECNICA if v in p3]
        ok(len(_hit) == len(f6._REEMPLAZOS_TECNICA),
           f"{tipo}/{forma}: los {len(f6._REEMPLAZOS_TECNICA)} ajustes encuentran su texto en la v3 con inventario ({len(_hit)})")
        # Los que INSERTAN dejan su texto viejo dentro del nuevo; los que
        # sustituyen, no lo dejan en ninguna parte.
        ok(all(p4.count(n) == 1 for _, n in f6._REEMPLAZOS_TECNICA[2:]) and f6._TECNICA_MEDIDA_NUEVA in p4
           and f6._TECNICA_PARTE_NUEVA + f6._TECNICA_PARTE_GRUPO in p4
           and not any(v in p4 for v, n in f6._REEMPLAZOS_TECNICA if v not in n),
           f"{tipo}/{forma}: con guion y JERARQUÍA, la técnica entera y el grupo de consecuencia")
        ok(f6._TECNICA_PARTE_GRUPO not in p4sj and f6._TECNICA_PARTE_NUEVA in p4sj,
           f"{tipo}/{forma}: sin JERARQUÍA, la técnica sin lo del grupo")
        ok(f6.prompt_estudio("ACTO", "CONC", C1, _m("v4"), es_recurso=rec) == p3,
           f"{tipo}/{forma}: sin guion, la v4 es la v3 (sin la técnica: la v3 no se toca)")
        ok("Antes este ejemplo" not in p4 and "bórralo" not in p4 and "transcribe el texto íntegro de la tesis "
           "debajo" not in " ".join(p4.split()),
           f"{tipo}/{forma}: fuera el comentario filtrado y la afirmación falsa del documento")
        ok(p4.rstrip().endswith("Nada más.") and p4.index(f6._RECUERDA_TECNICA.strip()) > p4.index("Escribe el estudio de fondo."),
           f"{tipo}/{forma}: el recordatorio de la técnica, entre lo último que se lee")
        _q = lambda t: {" ".join(x.split()) for x in re.findall(r"«([^«»]{1,300})»", t)}
        ok(not (_q(p4) - _q(p3) - _q(pe.bloque(GUION))),
           f"{tipo}/{forma}: la técnica no mete ninguna frase entre comillas para copiar")
_p2 = f6.prompt_estudio("ACTO", "CONC", C1, f6.Material(tipo_asunto="amparo_directo", materia="civil",
                                                       formato="estandar", n_planteamientos=1, variante="v2"))
ok("Antes este ejemplo" in _p2 and "DESPUÉS DE LA CITA, NO LA REPITAS" in _p2,
   "la v2 se queda como está (sus instantáneas no se mueven: test_inventario.py)")
_p1 = f6.prompt_estudio("ACTO", "CONC", C1, f6.Material(tipo_asunto="amparo_directo", materia="civil",
                                                       formato="estandar", n_planteamientos=1, variante="v1"))
ok("Antes este ejemplo" in _p1, "la v1, congelada, conserva su copia del comentario (quitarla movería su instantánea)")
ok(pe.bloque(GUION).count("se distingue en una o dos frases") == 1,
   "la JERARQUÍA: la tesis de la parte dentro de un grupo, una o dos frases y nunca como apoyo")

# ═══════════════════════════════════════════════════════════════════════════
print("\n6 · EL MOLDE DE FORMA LLEGA HASTA LA APLICACIÓN")
_rell = "Consideración sobre la competencia del tribunal y la oportunidad del recurso. " * 30
_txt = (_rell + "\nEl artículo 49 del Código dispone lo que dispone.\nDe dicho numeral se advierte que el "
        "causahabiente puede sustituir a quien transmitió el derecho; el límite es que no altere lo "
        "resuelto.\nEn el caso concreto, la adquirente exhibió la escritura y pidió continuar la "
        "ejecución. Esa constancia acredita la transmisión.\nOtro párrafo: el quejoso dijo otra cosa.")
_tr = fp._tramo_de_regla(_txt)
ok(fp.ROTULO_APLICACION in _tr and "exhibió la escritura" in _tr and "el quejoso dijo otra cosa" not in _tr
   and _tr.index("De dicho numeral") < _tr.index(fp.ROTULO_APLICACION),
   "la regla, el rótulo y el primer párrafo de aplicación; el resto del caso ajeno no")
ok(len(fp._tramo_de_regla(_rell * 3 + "\nDe dicho numeral se advierte que x. " * 40 + "\nEn el caso concreto, "
                         + "y. " * 600)) <= fp.TOPE_TRAMO, "con tope")
_larga = "x" * 3000 + "\n" + fp.ROTULO_APLICACION + "\nAPLICACIÓN ENTERA"
ok(fp._molde_para_el_bloque(_larga).endswith("APLICACIÓN ENTERA")
   and len(fp._molde_para_el_bloque(_larga)) <= fp.TOPE_MOLDE_EN_BLOQUE,
   "al bloque: si no cabe, se recorta por delante y la aplicación se queda")
_s = fp.Sondeo()
_s.moldes = [{"tramo": _tr, "tribunal": "colegiado", "calidad": 5}]
ok("rótulo de aplicación" in fp.bloque(_s) and fp.ROTULO_APLICACION in fp.bloque(_s),
   "el bloque describe el rótulo y lo muestra")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
