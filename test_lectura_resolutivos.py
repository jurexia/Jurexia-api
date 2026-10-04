"""LO QUE RESOLVIÓ EL JUZGADO, LEÍDO DE SUS PUNTOS (4-oct-2026, AR 380/2025).

El taller propuso «se confirma la sentencia recurrida y se sobresee» sobre una
sentencia cuyo único punto de fondo NEGABA el amparo. La cadena:

  1. `_RX_RESUELVE` llevaba `re.I` y casaba el verbo de la prosa —«las
     sentencias que resuelven»— y la fórmula de cierre que va DESPUÉS de los
     puntos —«NOTIFÍQUESE. A S Í lo resuelve y firma…»—; como se toma el último,
     la «sección resolutiva» era la firma electrónica;
  2. sin puntos que leer, el recuento de palabras de todo el documento contó la
     única «causa de sobreseimiento», que el juez DESCARTABA;
  3. con «sobresee» y la vía que confirma, la rama fue «confirma_sobresee».

Además el resolutivo tenía DOS puntos —el de fondo y uno de oficina (versión
pública)— y por eso no se reproducía. El texto de abajo replica esa estructura
con datos inventados (sin datos del expediente real).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OPENAI_API_KEY", "sk-falsa")

import fase_rama as fr  # noqa: E402

FALLOS = []


def ok(cond, msg):
    print(("   PASA   " if cond else "   FALLA  ") + msg)
    if not cond:
        FALLOS.append(msg)


SELLO = "MARÍA FERNANDA RUIZ SOTO 706a6620636a6632000000000000000000008bb4 15/05/26 18:00:00 18"
SENTENCIA_380 = f"""
JUICIO DE AMPARO INDIRECTO 123/2024
RESULTANDO: PRIMERO. Por escrito presentado el uno de marzo, EMPRESA EJEMPLO, S.A. DE C.V., promovió
juicio de amparo contra el auto que suspendió el juicio contencioso, hasta en tanto queden firmes las
sentencias que resuelven en definitiva los diversos juicios 1/2020 y 2/2020.
CONSIDERANDO: TERCERO. Las partes no hicieron valer causas de improcedencia y este órgano jurisdiccional
de oficio no advierte que se actualice motivo de improcedencia o causa de sobreseimiento respecto del
acto reclamado, de modo que procede el estudio de fondo. Se resuelven cuestiones de devolución de saldo.
SÉPTIMO. Los conceptos de violación son infundados. OCTAVO. Publíquese la versión pública.
Por lo expuesto y fundado, y con apoyo en los artículos 65, 73 a 77 y demás relativos de la Ley de Amparo.
SE RESUELVE: PRIMERO. La Justicia de la Unión no ampara ni protege a EMPRESA EJEMPLO, S.A. DE C.V., a
través de su {SELLO} apoderado legal Juan Ejemplo Pérez, respecto del acto reclamado identificado en el
considerado segundo de este fallo, atribuido a la Sala Regional del Centro, por los motivos establecidos
en el considerando séptimo de esta sentencia.
SEGUNDO. En acatamiento a lo resuelto en el considerando octavo de este fallo, captúrese el día de su
publicación la presente sentencia en versión pública, con la certificación secretarial respectiva; y
agréguese al expediente el acuse de recibo electrónico que justifique su registro.
NOTIFÍQUESE.
A S Í lo resuelve y firma Pedro Ejemplo Juez, Juez Sexto de Distrito, ante el secretario que autoriza.
EVIDENCIA CRIPTOGRÁFICA - TRANSACCIÓN Archivo Firmado: 1234.p7m FIRMANTE Nombre: PEDRO EJEMPLO
"""

print("\n1 · EL AR 380/2025: LA SECCIÓN, LO QUE DICE Y SU PUNTO DE FONDO")
sec = fr.seccion_resolutiva(SENTENCIA_380)
ok(sec.startswith("PRIMERO. La Justicia de la Unión no ampara ni protege")
   and "captúrese" in sec and "lo resuelve y firma" not in sec and "FIRMANTE" not in sec,
   f"la sección va del rótulo «SE RESUELVE:» a «NOTIFÍQUESE», no de la firma: {sec[:90]}…")
ok("706a66" not in sec and "18:00:00" not in sec and "a través de su apoderado legal" in sec,
   "el sello de la firma que el PDF metió a media frase se quita")
ok(fr.que_dice_el_resolutivo(sec) == "niega" and fr.resolvio_segun_resolutivos(SENTENCIA_380) == "niega",
   "lo que resolvió: «niega» (antes: la firma, y nada)")
rr = fr.resolutivo_recurrida(SENTENCIA_380)
ok(rr.startswith("La Justicia de la Unión no ampara ni protege a EMPRESA EJEMPLO")
   and rr.endswith("por los motivos establecidos en el considerando séptimo de la sentencia recurrida.")
   and "captúrese" not in rr and "considerando segundo de la sentencia recurrida" in rr,
   f"el punto de fondo se reproduce; el de oficina (versión pública) no cuenta; la errata «considerado» no pasa: {rr[:120]}…")
ok(fr.resolvio_a_quo(SENTENCIA_380) == "niega",
   "el respaldo sobre el texto entero también dice «niega»: la cola manda sobre el recuento")
ok(fr._resolvio_de(" ".join(SENTENCIA_380.split())) != "sobresee",
   "el recuento ya no cuenta «causa de sobreseimiento» que el juez descarta")

print("\n2 · LA SESIÓN GUARDADA CON LA LECTURA VIEJA SE CORRIGE SOLA")


class Fases:
    resolutivo_recurrida = ""
    resolvio_a_quo = "sobresee"
    antecedentes = ""
    fuentes = [SENTENCIA_380, "escrito de agravios"]


ok(fr.que_hizo_el_juzgado(Fases(), declarado="sobreseyó el juicio de amparo") == "niega",
   "con el papel en la sesión manda lo que dicen sus puntos, no lo guardado ni lo declarado")


class FasesSinPuntos(Fases):
    fuentes = ["Una sentencia sin puntos legibles que narra que el juez negó el amparo y sobreseyó.", ""]


ok(fr.que_hizo_el_juzgado(FasesSinPuntos(), declarado="negó el amparo solicitado") == "niega",
   "si el papel está y sus puntos no se dejan leer, lo declarado por el motor va antes que el recuento guardado")


class FasesSinPapel(Fases):
    fuentes = []


ok(fr.que_hizo_el_juzgado(FasesSinPapel()) == "sobresee",
   "sin el papel (sesiones viejas) queda lo guardado, como antes")

print("\n3 · LO QUE YA FUNCIONABA SIGUE IGUAL")
T631 = ("CONSIDERANDO … R E S U E L V E ÚNICO. La Justicia de la Unión ampara y protege a la Unión de "
        "Trabajadores Ejemplo, C.T.M. en contra el acto atribuido a la Primera Sala Civil, por los motivos "
        "expresados en el considerando séptimo de este fallo y para los efectos precisados en el último "
        "considerando. Notifíquese personalmente. Así lo resolvió la Jueza Séptimo de Distrito.")
ok(fr.resolvio_segun_resolutivos(T631) == "concede" and fr.resolutivo_recurrida(T631).startswith(
    "La Justicia de la Unión ampara y protege"), "AR 631: un solo punto que concede, reproducible")
TMIX = ("R E S U E L V E: PRIMERO. Se sobresee en el juicio respecto del acto atribuido al Juez Segundo. "
        "SEGUNDO. La Justicia de la Unión no ampara ni protege a Ana Ejemplo contra el acto del Juez "
        "Primero, por las razones del considerando quinto. NOTIFÍQUESE.")
ok(fr.resolvio_segun_resolutivos(TMIX) == "sobresee_niega" and fr.resolutivo_recurrida(TMIX) == "",
   "dos puntos de fondo: mixta y sin reproducir")
TSE = ("Por lo expuesto y fundado, se resuelve: ÚNICO. Se sobresee en el presente juicio de amparo promovido "
       "por Luis Ejemplo, por las razones del considerando tercero de esta resolución. Notifíquese.")
ok(fr.resolvio_segun_resolutivos(TSE) == "sobresee", "«se resuelve:» en minúscula, con sus dos puntos")
TOCR = ("…los conceptos de violación son fundados. R E S U E\nL V E PRIMERO La Justicia de la Unión ampara y "
        "protege a Rosa Ejemplo contra el acto reclamado para los efectos del considerando sexto. "
        "Notifíquese.")
ok(fr.resolvio_segun_resolutivos(TOCR) == "concede", "un rótulo que el OCR parte: la cola lo dice")
TCITA = ("En la tesis de rubro «EL JUEZ RESUELVE SOBRE LA SUSPENSIÓN», se dijo… "
         "R E S U E L V E: PRIMERO. La Justicia de la Unión no ampara ni protege a Eva Ejemplo. NOTIFÍQUESE.")
ok(fr.resolvio_segun_resolutivos(TCITA) == "niega",
   "un «RESUELVE» en versales dentro de un rubro no abre la sección: la abre el que trae «PRIMERO.»")

print("\n4 · CONFIRMAR ES CONFIRMAR LO QUE RESOLVIÓ EL JUZGADO, EN SUS TÉRMINOS")
import tipos_asunto as ta  # noqa: E402
import tarjeta_decision as td  # noqa: E402
_pc = ta.puntos_confirma("confirma_niega", rr)
ok(_pc == ["PRIMERO. Se confirma la sentencia recurrida.", "SEGUNDO. " + rr],
   "confirma_niega con el punto del juzgado que niega: se reproduce en sus términos")
ok(ta.puntos_confirma("confirma_sobresee", rr) == list(ta.RAMAS_REVISION["confirma_sobresee"]["puntos"]),
   "si el punto del juzgado NO dice lo que la rama (niega ≠ sobresee), la fórmula de la rama")
ok(ta.puntos_confirma("confirma_niega", "")[1] == ta.RAMAS_REVISION["confirma_niega"]["puntos"][1],
   "sin punto reproducible, la fórmula de siempre")
ok(ta.puntos_confirma("revoca_fondo_concede", rr) == [], "una rama que no confirma: nada")
_pts, _ = td.desenlace_de("amparo_revision", fr.que_hizo_el_juzgado(Fases(), ""), "infundado",
                          quien_recurre="quejoso", resolutivo_recurrida=fr.resolutivo_de_fases(Fases()))
ok(_pts and _pts[0] == "PRIMERO. Se confirma la sentencia recurrida."
   and _pts[1].startswith("SEGUNDO. La Justicia de la Unión no ampara ni protege a EMPRESA EJEMPLO")
   and not any("sobrese" in x.lower() for x in _pts),
   "la vía que confirma del AR 380/2025: confirma la NEGATIVA, en los términos del juzgado, sin sobreseer")
_pts_f, _ = td.desenlace_de("amparo_revision", fr.que_hizo_el_juzgado(Fases(), ""), "fundado",
                            quien_recurre="quejoso", resolutivo_recurrida=fr.resolutivo_de_fases(Fases()))
ok(_pts_f and _pts_f[0] == "PRIMERO. Se revoca la sentencia recurrida." and "ampara y protege" in _pts_f[1]
   and not any("sobrese" in x.lower() for x in _pts_f),
   "y la vía que revoca: revoca y ampara, sin «levantar un sobreseimiento» que no existió")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
