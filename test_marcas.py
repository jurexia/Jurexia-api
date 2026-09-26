"""Las marcas del estudio (Paso 2a del plan del estudio de fondo) — 26-sep-2026.

Con la v3 el estudio escribe «⟦C1.a C1.b⟧» al principio del párrafo que
contesta esos argumentos. Esto comprueba lo que la propuesta (w2_final §4.7)
exige de las marcas y lo que David no toleraría que fallara:

  1 · qué es una marca y qué no («[[p.7 §3]]», la nota al pie de los
      resúmenes, nunca lo es);
  2 · el filtro del flujo: la marca partida entre dos trozos, en CUALQUIER
      punto, no llega a la pantalla; un «⟦» sin cierre se devuelve tal cual;
      y NUNCA SE TRAGA TEXTO (propiedad sobre textos y cortes al azar);
  3 · el texto final: sin marcas sale idéntico byte por byte; con marcas, el
      mapa dice el párrafo de cada una;
  4 · lo que ve la pantalla escribiéndose es lo mismo que el texto final;
  5 · el control V1: faltan, rescatados, sin rastro, cobertura y el aviso.

    .venv/bin/python test_marcas.py
"""
import random
import re

import marcas as mc

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


def filtrar(trozos):
    f = mc.FiltroMarcas()
    return "".join(f.alimentar(t) for t in trozos) + f.cerrar()


# ═══════════════════════════════════════════════════════════════════════════
print("\n1 · QUÉ ES UNA MARCA")
ok(mc.ids_de("C1.a C1.b") == ["C1.a", "C1.b"], "dos identificadores separados por espacio")
ok(mc.ids_de("C1.a, C2.b y C3.c") == ["C1.a", "C2.b", "C3.c"], "con comas y una «y»")
ok(mc.ids_de("c1.A") == ["C1.a"], "se normaliza: prefijo en mayúscula, letra en minúscula")
ok(mc.ids_de("C1.a–C1.c") == ["C1.a", "C1.b", "C1.c"], "un rango se despliega")
ok(mc.ids_de("M1") == ["M1"] and mc.ids_de("U2") == ["U2"] and mc.ids_de("A3.b") == ["A3.b"]
   and mc.ids_de("AD1.a") == ["AD1.a"], "premisas, unidades, agravios y adhesivos")
ok(mc.ids_de("p.7 §3") == [], "«[[p.7 §3]]» —la nota al pie de los resúmenes— no es marca")
ok(mc.ids_de("C1.a y lo demás") == [], "si un solo elemento no es identificador, no es marca")
ok(mc.ids_de("") == [] and mc.ids_de("   ") == [], "vacía no es marca")

# ═══════════════════════════════════════════════════════════════════════════
print("\n2 · EL FILTRO DEL FLUJO")
TXT = ("Los conceptos de violación son infundados.\n\n"
       "⟦C1.a C1.b⟧ Sobre el primer concepto de violación, en el que la parte quejosa "
       "sostiene que la pericial era la prueba idónea. Se considera infundado.\n\n"
       "Lo anterior, porque la identidad se acreditó con la confesión.\n\n"
       "⟦C2.a⟧ Sobre el segundo concepto de violación, es inoperante.")
LIMPIO = ("Los conceptos de violación son infundados.\n\n"
          "Sobre el primer concepto de violación, en el que la parte quejosa "
          "sostiene que la pericial era la prueba idónea. Se considera infundado.\n\n"
          "Lo anterior, porque la identidad se acreditó con la confesión.\n\n"
          "Sobre el segundo concepto de violación, es inoperante.")
ok(filtrar([TXT]) == LIMPIO, "de una vez: las marcas se quitan con su blanco")
malos = []
for i in range(len(TXT) + 1):
    if filtrar([TXT[:i], TXT[i:]]) != LIMPIO:
        malos.append(i)
ok(not malos, f"partido en DOS trozos en cada una de las {len(TXT) + 1} posiciones"
   + (f": falla en {malos[:5]}" if malos else ""))
malos = []
for i in range(0, len(TXT), 3):
    for j in range(i, len(TXT), 7):
        if filtrar([TXT[:i], TXT[i:j], TXT[j:]]) != LIMPIO:
            malos.append((i, j))
ok(not malos, "partido en TRES trozos, en cualquier par de posiciones"
   + (f": falla en {malos[:3]}" if malos else ""))
ok(filtrar(list(TXT)) == LIMPIO, "carácter por carácter")
ok(filtrar(["⟦C1.a⟧", " Sobre el primero."]) == "Sobre el primero.",
   "el blanco que sigue a la marca se come aunque llegue en el trozo siguiente")
ok(filtrar(["texto ⟦C1.a⟧\nOtro párrafo."]) == "texto \nOtro párrafo.",
   "una marca al final del renglón no se lleva el salto")
sin_cierre = "El monto era ⟦ según la sentencia " + "x" * (mc.LIMITE + 50) + " y sigue."
_f_sc = mc.FiltroMarcas()
ok(_f_sc.alimentar(sin_cierre) == sin_cierre,
   f"«⟦» sin cierre en {mc.LIMITE} caracteres se devuelve tal cual, SIN esperar al final del flujo")
ok(filtrar([sin_cierre[:40], sin_cierre[40:120], sin_cierre[120:]]) == sin_cierre,
   "…también si llega a trozos")
# LA MARCA LARGA (revisión adversarial, 26-sep-2026): la regla de la v3 pide
# nombrar juntos los argumentos que se contestan a la vez, y en dos de las 64
# sesiones reales la marca con todo el concepto más largo pasaba de 200
# caracteres (407 con 72 argumentos). Con el tope viejo llegaba entera a la
# pantalla y al .docx.
_ids72 = [f"C1.{mc_l}" for mc_l in ([chr(97 + i) for i in range(26)]
                                    + ["a" + chr(97 + i) for i in range(26)]
                                    + ["b" + chr(97 + i) for i in range(20)])]
_larga = "Abre.\n⟦" + " ".join(_ids72) + "⟧ Sobre todos, infundados.\nSigue."
ok(len(_larga) > 400, f"la marca de prueba mide lo que la real ({len(_larga)} caracteres)")
_l_larga, _m_larga = mc.separar_marcas(_larga)
ok(_l_larga == "Abre.\nSobre todos, infundados.\nSigue." and len(_m_larga) == 72
   and _m_larga["C1.bt"] == [1], "el texto final la quita y el mapa trae los 72")
ok(filtrar([_larga[i:i + 9] for i in range(0, len(_larga), 9)]) == "Abre.\nSobre todos, infundados.\nSigue.",
   "y el flujo también, aunque llegue en trozos de nueve")
# EL MISMO TOPE EN LOS DOS: una marca justo en el borde se quita o se deja
# igual en la pantalla y en el texto final.
for _extra in (-1, 0, 1):
    _cont = "C1.a" + " " * (mc.LIMITE + _extra - 4)      # una marca válida de ese largo
    _t = "Uno ⟦" + _cont + "⟧ dos."
    _quita = _extra <= 0
    _fin, _mp = mc.separar_marcas(_t)
    ok(len(_cont) == mc.LIMITE + _extra
       and ("⟦" not in filtrar([_t])) == _quita
       and "⟦" not in _fin and "⟧" not in _fin and _fin == "Uno dos." and _mp == {"C1.a": [0]},
       f"en el borde del tope ({len(_cont)} caracteres dentro): el flujo "
       f"{'la quita' if _quita else 'la deja pasar tal cual (nunca se come texto)'}; "
       "el texto final la quita siempre")
ok(mc.ids_de("C1.a/C1.b") == ["C1.a", "C1.b"] and mc.ids_de("C1.a | C2.b") == ["C1.a", "C2.b"]
   and mc.ids_de("C1.a – C1.c") == ["C1.a", "C1.b", "C1.c"],
   "barras y rango con blancos también son marca (si no, llegaban al .docx)")
_nota_larga = "Nota [[" + "p.7 " * 80 + "]] fin."
_f_nl = mc.FiltroMarcas()
ok(_f_nl.alimentar(_nota_larga[:-10]).startswith("Nota [[p.7"),
   f"un «[[» sin cerrar se suelta a los {mc.LIMITE_CORCHETES}: la nota al pie no espera como «⟦»")
ok(filtrar(["Queda un ⟦ abierto"]) == "Queda un ⟦ abierto",
   "«⟦» abierto al final del flujo: `cerrar` lo devuelve")
raro = "Un corchete ⟦que no es marca⟧ y sigue."
ok(filtrar([raro]) == raro, "«⟦…⟧» con algo que no son identificadores sale tal cual")
nota = "Se resumió así. [[p.7 §3]] Y sigue."
ok(filtrar([nota]) == nota and filtrar(["Se resumió así. [", "[p.7 §3]] Y sigue."]) == nota,
   "la nota al pie «[[p.7 §3]]» pasa intacta, entera o partida")
ok(filtrar(["[[C1.a]] Sobre el primero."]) == "Sobre el primero."
   and filtrar(["[", "[C1.a", "]] Sobre el primero."]) == "Sobre el primero.",
   "la marca escrita con corchetes dobles también se quita, aunque llegue partida")
ok(filtrar(["termina en [", "x"]) == "termina en [x", "un «[» suelto al final de un trozo no se pierde")
ok(filtrar(["termina en ["]) == "termina en [", "…ni al final del flujo")
# LA PROPIEDAD: SIN MARCAS VÁLIDAS, LO QUE SALE ES LO QUE ENTRÓ. Textos al azar
# con los signos que pueden confundir al filtro, cortados al azar.
random.seed(26)
ALFABETO = list("abc CDM1.⟦⟧[]\n, ") + ["C1", "p.7 §3", "⟦C1", "[[", "]]", "⟧ ", "x" * 40]
traga = 0
for _ in range(3000):
    t = "".join(random.choice(ALFABETO) for _ in range(random.randint(0, 60)))
    t = t.replace("⟦C1.a⟧", "⟦C1 .a⟧")        # sin marcas válidas por construcción
    if mc.marcas_en(t):
        continue
    cortes = sorted(random.sample(range(len(t) + 1), k=min(len(t) + 1, random.randint(0, 4))))
    trozos, a = [], 0
    for c in cortes:
        trozos.append(t[a:c])
        a = c
    trozos.append(t[a:])
    if filtrar(trozos) != t:
        traga += 1
ok(traga == 0, "3,000 textos al azar sin marcas válidas, cortados al azar: el filtro "
   f"devuelve exactamente lo que entró ({traga} distintos)")
# Y CON MARCAS: lo que sale es el texto sin ellas, nunca menos.
random.seed(27)
perdidas = 0
for _ in range(1500):
    partes = []
    for _k in range(random.randint(1, 6)):
        partes.append(random.choice(["⟦C1.a⟧ ", "⟦C2.b C3.c⟧ ", "", ""]))
        partes.append(random.choice(["Texto del párrafo", "con ⟦ un signo raro", "[[p.2 §1]]",
                                     "otra frase.", "\n"]))
    t = "".join(partes)
    esperado = t
    for m in ("⟦C1.a⟧ ", "⟦C2.b C3.c⟧ "):
        esperado = esperado.replace(m, "")
    cortes = sorted(random.sample(range(len(t) + 1), k=min(len(t) + 1, 3)))
    trozos, a = [], 0
    for c in cortes:
        trozos.append(t[a:c])
        a = c
    trozos.append(t[a:])
    if filtrar(trozos) != esperado:
        perdidas += 1
ok(perdidas == 0, f"1,500 textos con marcas cortados al azar: sale el texto sin ellas ({perdidas} mal)")

# ═══════════════════════════════════════════════════════════════════════════
print("\n3 · EL TEXTO FINAL Y EL MAPA")
sin = "SEXTO. Estudio.\nLos conceptos son infundados.\n\nSe resumió. [[p.7 §3]]\n  \nFin."
ok(mc.separar_marcas(sin) == (sin, {}), "sin marcas el texto sale IDÉNTICO, byte por byte")
ok(mc.separar_marcas("") == ("", {}) and mc.separar_marcas(None) == ("", {}), "vacío o None")
limpio, mapa = mc.separar_marcas(TXT)
ok(limpio == LIMPIO, "con marcas, el texto limpio es el mismo que el del flujo")
ok(mapa == {"C1.a": [1], "C1.b": [1], "C2.a": [3]},
   f"el mapa da el párrafo (renglones no vacíos) de cada argumento: {mapa}")
t2 = ("Abre.\n⟦C1.a⟧\nEl párrafo que contesta.\nOtro ⟦C1.b⟧ a media frase ⟦M1⟧ y más.\n"
      "⟦U1⟧ Efecto primero.\n⟦C9.z⟧")
l2, m2 = mc.separar_marcas(t2)
ok(l2 == "Abre.\nEl párrafo que contesta.\nOtro a media frase y más.\nEfecto primero.",
   "marca sola en su renglón: el renglón desaparece; a media frase: sin blancos dobles")
ok(m2 == {"C1.a": [1], "C1.b": [2], "M1": [2], "U1": [3], "C9.z": [3]},
   f"la marca sola va al párrafo siguiente; la del final, al último; M y U también: {m2}")
l3, m3 = mc.separar_marcas("⟦C1.a⟧ Uno.\n⟦C1.a C1.b⟧ Dos.")
ok(m3 == {"C1.a": [0, 1], "C1.b": [1]}, "un argumento en dos párrafos lleva los dos")
ok(mc.separar_marcas("[[C1.a]] Uno. [[p.7 §3]]") == ("Uno. [[p.7 §3]]", {"C1.a": [0]}),
   "corchetes dobles: se quita la marca y se queda la nota al pie")
ok(mc.sin_marcas(TXT) == LIMPIO, "`sin_marcas` es el texto limpio")
# LA MARCA A MEDIAS (revisión adversarial, 26-sep-2026): sin su cierre o sin
# su apertura no la reconocía ninguna forma y llegaba al .docx.
_l, _m = mc.separar_marcas("Abre.\n⟦C1.a C1.b Sobre el primero, infundado.\nC2.a⟧ Sobre el segundo.")
ok(_l == "Abre.\nSobre el primero, infundado.\nSobre el segundo." and _m == {"C1.a": [1], "C1.b": [1], "C2.a": [2]},
   "«⟦» sin cierre o «⟧» sin apertura: se quitan con sus identificadores y la prosa se queda")
ok(mc.separar_marcas("Un ⟦ signo raro en la prosa.") == ("Un signo raro en la prosa.", {})
   and mc.separar_marcas("Un corchete ⟦que no es marca⟧ y sigue.")[0] == "Un corchete que no es marca y sigue."
   and mc.separar_marcas("Sólo cierra ⟧ aquí.") == ("Sólo cierra aquí.", {}),
   "un «⟦» o «⟧» sin identificadores: en el texto final se va EL SIGNO y la prosa se queda entera")
ok(mc.separar_marcas("Sin signos.\n\nNi marcas.") == ("Sin signos.\n\nNi marcas.", {}),
   "y sin «⟦», «⟧» ni «[[» el texto sigue saliendo idéntico")
ok(mc.separar_marcas("a [[⟦C1.a⟧]] b") == ("a [[ ]] b", {"C1.a": [0]})
   and filtrar(["a [[⟦C1.a⟧]] b"]) == "a [[]] b",
   "una marca dentro de corchetes que no son marca: la quitan los dos (el texto y el flujo)")

# ═══════════════════════════════════════════════════════════════════════════
print("\n4 · LO QUE SE VE ESCRIBIÉNDOSE ES LO QUE SE FIRMA")
t4 = ("Los conceptos son en parte fundados.\n\n⟦C1.a C1.b⟧\nSobre el primero, fundado.\n\n"
      "⟦C2.a⟧ Sobre el segundo, inoperante.\n\nEFECTOS\n⟦U1⟧ La responsable deja insubsistente.")


def _norm(x):
    return "\n".join(ln.strip() for ln in x.split("\n") if ln.strip())


def _norm_signos(x):
    # Un «⟦» suelto en la prosa: el flujo lo deja pasar (nunca se come nada) y
    # el texto final quita EL SIGNO (nunca llega al .docx). La prosa, igual.
    return _norm("\n".join(re.sub(r"[ \t]+", " ", ln.replace("⟦", " ").replace("⟧", " "))
                           for ln in x.split("\n")))


_flujo = filtrar([t4[i:i + 5] for i in range(0, len(t4), 5)])
ok(_norm(_flujo) == _norm(mc.sin_marcas(t4)),
   "el texto del flujo y el final coinciden (salvo renglones en blanco)")
ok("⟦" not in _flujo and "⟧" not in _flujo and "⟦" not in mc.sin_marcas(t4),
   "ni una marca en ninguno de los dos")
# La regla de operación de la propuesta (§6.3): texto del flujo = texto final.
random.seed(28)
distintos = 0
for _ in range(1000):
    parrs = []
    for _k in range(random.randint(1, 7)):
        marca = random.choice(["", "", "⟦C1.a⟧ ", "⟦C1.b C2.a⟧ ", "⟦M1⟧ ", "⟦U1⟧ ", "⟦C3.c⟧\n"])
        cuerpo = random.choice(["Sobre el primero, infundado.", "Lo anterior, porque sí [[p.3 §1]].",
                                "Con un ⟦ suelto en medio.", "Efecto: dicte otra."])
        parrs.append(marca + cuerpo)
    t = "\n\n".join(parrs)
    corte = sorted(random.sample(range(len(t) + 1), k=min(len(t) + 1, 4)))
    trozos, a = [], 0
    for c in corte:
        trozos.append(t[a:c])
        a = c
    trozos.append(t[a:])
    _fl, _fi = filtrar(trozos), mc.sin_marcas(t)
    if _norm_signos(_fl) != _norm_signos(_fi) or "⟦" in _fi or "⟧" in _fi \
            or any(m_ in _fl for m_ in ("⟦C", "⟦M", "⟦U")):
        distintos += 1
ok(distintos == 0, f"1,000 estudios al azar con marcas: flujo = texto final, salvo el signo suelto que "
   f"el final quita; ninguna marca en ninguno y ningún signo en el final ({distintos} distintos)")

# ═══════════════════════════════════════════════════════════════════════════
print("\n5 · EL CONTROL V1")
SEGS = [
    {"id": "C1.a", "texto": "Aduce que la prueba pericial en topografía era la idónea para acreditar "
                            "la identidad del inmueble reivindicado.", "anclas": []},
    {"id": "C1.b", "texto": "Refiere que las superficies de cuatro hectáreas y de dos hectáreas y "
                            "media son contradictorias.", "anclas": ["cifra 4"]},
    {"id": "C2.a", "texto": "Sostiene que lo resuelto en el expediente 1114/2017 produce cosa "
                            "juzgada refleja.", "anclas": ["exp 1114/2017"]},
    {"id": "C3.a", "texto": "Afirma que la condena en costas desconoce su condición de campesino "
                            "que no sabe leer ni escribir.", "anclas": []},
]
estudio = ("Los conceptos son infundados.\n"
           "Sobre el primero, la identidad del inmueble se acreditó sin pericial topográfica.\n"
           "La resolución del juicio 1114/2017 no produce cosa juzgada refleja porque no decidió el fondo.")
cob = mc.verificar(SEGS, {"C1.a": [1]}, estudio)
ok(cob["faltan"] == ["C1.b", "C2.a", "C3.a"] and cob["marcados"] == 1 and cob["total"] == 4,
   "faltan los que no tienen marca")
ok("C2.a" in cob["rescatados"], "se rescata por su ancla propia (el expediente 1114/2017 está en el texto)")
ok(cob["sin_rastro"] == ["C1.b", "C3.a"] or set(cob["sin_rastro"]) == {"C1.b", "C3.a"},
   f"sin marca y sin rastro: {cob['sin_rastro']}")
ok(cob["cobertura"] == 0.25 and cob["cobertura_con_rescate"] == 0.5, "cobertura y cobertura con rescate")
cob2 = mc.verificar(SEGS, {"C1.a": [1], "C1.b": [1], "C2.a": [2], "C3.a": [2], "M1": [1], "C7.q": [2]}, estudio)
ok(cob2["faltan"] == [] and cob2["cobertura"] == 1.0 and cob2["desconocidos"] == ["C7.q"],
   "todos marcados: cobertura 1; un id que no está en el inventario se anota (M y U no)")
ok(mc.verificar([], {}, "x")["cobertura"] == 1.0, "sin inventario no hay nada que acusar")
por_texto = mc.verificar(SEGS[3:], {}, "Sobre las costas: la condición de campesino que no sabe leer "
                                       "ni escribir no releva de la condena.")
ok(por_texto["rescatados"] == ["C3.a"], "se rescata por texto (sus palabras propias en un párrafo)")
ok(mc.aviso(SEGS, cob2) == "", "sin argumentos sin rastro, no hay aviso")
av = mc.aviso(SEGS, cob)
ok(av.startswith("ARGUMENTOS SIN RESPUESTA IDENTIFICABLE") and "C1.b" in av and "C3.a" in av
   and "C2.a" not in av and "superficies" in av,
   "el aviso nombra sólo los sin rastro, con lo que se alega")
muchos = [{"id": f"C1.{chr(97 + i)}", "texto": f"argumento {i}", "anclas": []} for i in range(9)]
av9 = mc.aviso(muchos, mc.verificar(muchos, {}, "nada que ver"))
ok("y 3 más" in av9, "con más de seis, dice cuántos más")
ok(mc.UMBRAL_RASTRO == 0.15, "el umbral del rescate es el calibrado (0.15: 1.2 % de falsa alarma)")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
