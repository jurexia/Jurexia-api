"""FASE 0 del redactor de sentencias — ficha del asunto y cómputo de la oportunidad.

AQUÍ NO ENTRA NINGÚN MODELO DE LENGUAJE, Y ES DELIBERADO.

Un plazo mal contado invalida la sentencia. Contar días hábiles es aritmética
sobre un calendario: tiene una respuesta correcta y se puede demostrar. Meter
un modelo aquí sólo añadiría una forma nueva de equivocarse, sin ganar nada.

═══════════════════════════════════════════════════════════════════════════
SON DOS CALENDARIOS, NO UNO
═══════════════════════════════════════════════════════════════════════════

  1. CUÁNDO SURTE EFECTOS la notificación del acto reclamado se rige por la
     ley que gobierna ESE acto —contencioso administrativo local, laboral,
     civil del estado, fiscal federal— y se cuenta sobre el calendario de
     inhábiles de LA AUTORIDAD RESPONSABLE, que tiene sus propias vacaciones
     y suspensiones.

  2. EL PLAZO PARA PROMOVER EL AMPARO corre sobre el calendario del artículo
     19 de la Ley de Amparo, que es otro.

Usar uno solo para las dos cosas da fechas equivocadas en cuanto la
responsable tiene un periodo vacacional que el PJF no tiene, o al revés.

═══════════════════════════════════════════════════════════════════════════

El algoritmo NO se dedujo de la ley: se leyó del engrose real ADA 240/2026 y
se comprobó que lo reproduce al día:

    «la sentencia reclamada se notificó a la parte quejosa el veintitrés de
     febrero de dos mil veintiséis mediante Boletín Jurisdiccional y surtió
     efectos al tercer día hábil siguiente, es decir, el veintiséis de febrero
     […] por lo que el plazo […] fue del veintisiete de febrero al veinte de
     marzo […] sin contar sábados y domingos por ser inhábiles en términos del
     artículo 19 de la Ley de Amparo, así como el dieciséis de marzo»

Y el «dieciséis de marzo» no era un capricho: es el TERCER LUNES DE MARZO, a
donde la Ley Federal del Trabajo traslada el 21 de marzo. Se descubrió
calculándolo, no suponiéndolo.

═══════════════════════════════════════════════════════════════════════════
LO QUE LA AUDITORÍA DICE HOY — 28-ago-2026, y hay que leerlo antes de usar
═══════════════════════════════════════════════════════════════════════════

Contrastado contra 49 cómputos LEGIBLES de adelantos y engroses firmados
(`auditar_fase0.py`):

    sin los calendarios oficiales ....  4  ( 8%)
    con dos reglas deducidas .........  9  (18%)
    CON LOS CALENDARIOS DEL OAJ ...... 37  (76%)   ← estado actual

La pieza que faltaba no era lógica: eran DATOS. Los periodos vacacionales y
las semanas santas no se derivan de ninguna regla —la de 2025 fue del 16 al 18
de abril y la de 2026 es del 1 al 3— y hay que transcribirlos del sitio del
OAJ.

NO ES APTO PARA FIRMAR TODAVÍA. Sirve para proponer y para revisar; no para
sustituir el cómputo del secretario.

De los 12 fallos que quedan, DOS son del propio auditor (desfase de +365 días
= error al heredar el año en la extracción). De los otros diez, la mitad se va
por ±1 día: es la REGLA DE SURTIMIENTO, que cambia según la ley que rige el
acto y que el auditor adivina por palabra clave («boletín», «por lista»).

Conclusión de ingeniería: la vía de notificación y el plazo NO deben
inferirse. Tienen que ser campos que el secretario confirme, porque un plazo
mal contado invalida la sentencia y una heurística de palabra clave no es
base para eso.

Dos correcciones ya entraron con evidencia y subieron la concordancia del 8%
al 18%:
  · la notificación PERSONAL surte al día hábil siguiente, no el mismo día
    (ADA 448-2025 y 449-2025);
  · la Semana Santa completa es inhábil, no sólo jueves y viernes
    (el vencimiento firmado del 7 de mayo de 2025 sólo sale con la semana
     del 14 al 18 de abril entera).

Faltan los periodos vacacionales del PJF de los demás años. Cada uno que se
añada debe comprobarse contra un engrose firmado, como estos dos.
"""

from __future__ import annotations

import re

from datetime import date as _date

import datetime as _dt
import re as _re
from dataclasses import dataclass, field
from typing import Iterable, Optional

Fecha = _dt.date


# ═══════════════════════════════════════════════════════════════════════════
# Calendario — se instancia una vez por jurisdicción
# ═══════════════════════════════════════════════════════════════════════════

def _lunes_ordinal(anio: int, mes: int, n: int) -> Fecha:
    d = _dt.date(anio, mes, 1)
    d += _dt.timedelta(days=(0 - d.weekday()) % 7)
    return d + _dt.timedelta(weeks=n - 1)


@dataclass
class Calendario:
    """Un calendario de días hábiles.

    `nombre` se usa en el considerando cuando hay que decir por qué un día no
    contó. `periodos` son rangos cerrados —las vacaciones del órgano—;
    `sueltos`, días aislados por acuerdo o fuerza mayor.
    """
    nombre: str
    fundamento: str = ""
    fijos: set[tuple[int, int]] = field(default_factory=set)   # (día, mes)
    trasladados: dict[int, int] = field(default_factory=dict)  # mes: n-ésimo lunes
    sueltos: set[Fecha] = field(default_factory=set)
    periodos: list[tuple[Fecha, Fecha]] = field(default_factory=list)

    def inhabiles_del_anio(self, anio: int) -> set[Fecha]:
        dias = {_dt.date(anio, m, d) for d, m in self.fijos}
        for mes, ordinal in self.trasladados.items():
            dias.add(_lunes_ordinal(anio, mes, ordinal))
        dias |= {f for f in self.sueltos if f.year == anio}
        for ini, fin in self.periodos:
            cur = ini
            while cur <= fin:
                if cur.year == anio:
                    dias.add(cur)
                cur += _dt.timedelta(days=1)
        return dias

    def es_habil(self, f: Fecha) -> bool:
        return f.weekday() < 5 and f not in self.inhabiles_del_anio(f.year)

    def siguiente_habil(self, f: Fecha) -> Fecha:
        while not self.es_habil(f):
            f += _dt.timedelta(days=1)
        return f

    def sumar(self, desde: Fecha, n: int) -> list[Fecha]:
        """Los `n` días hábiles desde `desde`, incluyéndolo si lo es."""
        dias: list[Fecha] = []
        cur = desde
        while len(dias) < n:
            if self.es_habil(cur):
                dias.append(cur)
            cur += _dt.timedelta(days=1)
        return dias


# ── El calendario federal del amparo (art. 19 LA) ─────────────────────────
#
# OJO A QUIEN TOQUE ESTO: la lista es de consulta obligada contra el texto
# vigente y contra el acuerdo del CJF del año. No se cambia de memoria. Los
# periodos vacacionales del PJF se cargan en `sueltos`/`periodos` por año.
_d = _dt.date

# ── LOS DÍAS INHÁBILES OFICIALES ──────────────────────────────────────────
#
# Copiados del sitio del Órgano de Administración Judicial, que es la fuente
# que David señaló:
#     https://www.oaj.gob.mx/transparencia/paginas/diasinhabiles.htm
#
# NO se derivan de reglas: se transcriben. Los periodos vacacionales (16-31 de
# julio y 16-31 de diciembre) y la Semana Santa cambian cada año y no hay
# fórmula que los prediga — la de 2025 fue del 16 al 18 de abril y la de 2026
# es del 1 al 3.
#
# Al transcribir hay una trampa: en el sitio las llamadas a nota van PEGADAS al
# número del día, así que «Lunes 23,4» es el lunes 2 con las notas 3 y 4, y
# «Lunes 156» es el lunes 15 con la nota 6. Se desambigua exigiendo que el
# resto tenga forma de lista de notas y comprobando el día de la semana.
#
# PARA AÑADIR UN AÑO: se copia del sitio y se vuelve a correr `auditar_fase0.py`.
INHABILES_OAJ = {
    2024: {_d(2024,1,1), _d(2024,2,5), _d(2024,3,18), _d(2024,3,21), _d(2024,3,27), _d(2024,3,28), _d(2024,3,29), _d(2024,5,1), _d(2024,9,16), _d(2024,10,1), _d(2024,11,1), _d(2024,11,18), _d(2024,11,20)},
    2025: {_d(2025,1,1), _d(2025,2,3), _d(2025,2,5), _d(2025,3,17), _d(2025,3,21), _d(2025,4,16), _d(2025,4,17), _d(2025,4,18), _d(2025,5,1), _d(2025,5,2), _d(2025,5,5), _d(2025,9,15), _d(2025,9,16), _d(2025,11,17), _d(2025,11,20)},
    2026: {_d(2026,1,1), _d(2026,2,2), _d(2026,2,5), _d(2026,3,16), _d(2026,4,1), _d(2026,4,2), _d(2026,4,3), _d(2026,5,1), _d(2026,5,4), _d(2026,5,5), _d(2026,9,14), _d(2026,9,15), _d(2026,9,16), _d(2026,10,12), _d(2026,11,2), _d(2026,11,16), _d(2026,11,20), _d(2026,12,25)},
}

# EL «31» PERDIÓ EL 1 EN 2024 Y EN 2025, y nadie se enteró durante meses.
# Estaba escrito `(_d(2024,7,16), _d(2024,7,3))`: el periodo empezaba el 16 y
# terminaba el 3, trece días ANTES. El bucle `while cur <= fin` no produce ni
# una fecha, así que las dos segundas quincenas de vacaciones del Poder Judicial
# —julio y diciembre— de 2024 y 2025 se contaron ENTERAS como hábiles. Son seis
# semanas de días inhábiles perdidos, y todo cómputo que cruce esas ventanas
# salía corto. No es un localismo: es nacional y nos afectaba a todos.
#
# Lo delata su propio vecino: 2026 sí dice 31.
PERIODOS_OAJ = {
    2024: [(_d(2024,7,16), _d(2024,7,31)), (_d(2024,12,16), _d(2024,12,31))],
    2025: [(_d(2025,7,16), _d(2025,7,31)), (_d(2025,12,16), _d(2025,12,31))],
    2026: [(_d(2026,7,16), _d(2026,7,31)), (_d(2026,12,16), _d(2026,12,31))],
}


def _revisar_periodos() -> None:
    """Un rango invertido no vuelve a pasar en silencio.

    Cuesta microsegundos al importar y habría cazado esto el primer día.
    """
    for anio, ps in PERIODOS_OAJ.items():
        for ini, fin in ps:
            if fin < ini:
                raise ValueError(
                    f"Periodo vacacional {anio} invertido: {ini} → {fin}. "
                    f"Un rango al revés no produce ningún día inhábil y el "
                    f"cómputo sale corto sin avisar.")


_revisar_periodos()


# ═══════════════════════════════════════════════════════════════════════════
# CADA INHÁBIL CON SU FUNDAMENTO (3-oct-2026)
# ═══════════════════════════════════════════════════════════════════════════
# El considerando decía de TODO inhábil «por ser inhábiles en términos del
# artículo 19 de la Ley de Amparo», también del dos de mayo de 2025, del
# diecisiete de marzo o de las vacaciones. El 19 no nombra ninguno de ésos: el
# dos de mayo lo declaró una circular, el tercer lunes de marzo es el traslado
# de la Ley Federal del Trabajo y las vacaciones las fija el Órgano conforme a
# la Ley Orgánica. Un fundamento que no dice lo que se le atribuye es un
# fundamento falso en un considerando firmado.
#
# LOS DATOS SON LAS NOTAS DE LA PROPIA PÁGINA DEL OAJ (la misma fuente de
# INHABILES_OAJ), descargada y leída el 3-oct-2026: cada día lleva sus
# llamadas —1 Ley de Amparo, 2 LOPJF, 3/4 LFT o el Acuerdo General del otrora
# CJF que reglamenta su organización (art. 6, fr. III: «Los lunes en que por
# disposición del artículo 74 de la Ley Federal del Trabajo deje de
# laborarse»), 5/6 la circular del año—. Lo que aquí no está (un año sin
# cargar, un inhábil declarado a mano) se queda en el artículo 19, cuyo texto
# sí cubre «aquellos en que se suspendan las labores en el órgano
# jurisdiccional»: no se inventa un instrumento.
#
# EL ARTÍCULO 229 DE LA LOPJF NO SE CITA PARA ESOS DÍAS: su lista (sábados,
# domingos, 1o. de enero, 5 de febrero, 21 de marzo, 1o. de mayo, 14 y 16 de
# septiembre, 20 de noviembre) no trae ni los lunes de la LFT ni los de las
# circulares; atribuírselos sería el mismo error con otro número. Los suyos ya
# los nombra el 19 de la Ley de Amparo, que es la ley del juicio.
#
# LAS VACACIONES: artículo 226 de la LOPJF vigente (DOF 20-12-2024: «las y los
# Magistrados de Circuito y las y los Jueces de Distrito disfrutarán
# anualmente de dos periodos vacacionales… en los periodos que fije el Órgano
# de Administración Judicial»), verificado en el texto local de la ley. Sólo
# desde 2025: las de 2024 las regía la ley abrogada y se quedan en el 19. El
# acuerdo del Órgano que fija las fechas no lo nombra su propia página (cita
# la ley), y no se inventa su número.
_ART19_FIJOS = {(1, 1), (5, 2), (21, 3), (1, 5), (5, 5), (14, 9), (16, 9),
                (12, 10), (20, 11), (25, 12)}
_AG_CJF = ("del Acuerdo General del Pleno del otrora Consejo de la Judicatura "
           "Federal que reglamenta la organización y funcionamiento del propio Consejo")
FUENTES_INHABIL = {
    "art19": {"texto": "artículo 19 de la Ley de Amparo", "corto": "art. 19 LA"},
    "lft_ii": {"texto": ("artículo 74, fracción II, de la Ley Federal del Trabajo, en "
                         "relación con el artículo 6, fracción III, " + _AG_CJF),
               "corto": "art. 74, fr. II, LFT"},
    "lft_iii": {"texto": ("artículo 74, fracción III, de la Ley Federal del Trabajo, en "
                          "relación con el artículo 6, fracción III, " + _AG_CJF),
                "corto": "art. 74, fr. III, LFT"},
    "lft_vi": {"texto": ("artículo 74, fracción VI, de la Ley Federal del Trabajo, en "
                         "relación con el artículo 6, fracción III, " + _AG_CJF),
               "corto": "art. 74, fr. VI, LFT"},
    "c7_2024": {"texto": "Circular 7/2024 del Pleno del otrora Consejo de la Judicatura Federal",
                "corto": "Circular 7/2024 CJF"},
    "c1_2025": {"texto": ("Circular 1/2025 de la Secretaría Ejecutiva del Pleno del "
                          "otrora Consejo de la Judicatura Federal"),
                "corto": "Circular 1/2025 SE CJF"},
    "c3_2025": {"texto": ("Circular 3/2025 de la Secretaría Ejecutiva del Pleno del "
                          "Órgano de Administración Judicial"),
                "corto": "Circular 3/2025 SE OAJ"},
    "c3_2026": {"texto": ("Circular 3/2026 de la Secretaría Ejecutiva del Pleno del "
                          "Órgano de Administración Judicial"),
                "corto": "Circular 3/2026 SE OAJ"},
    "vac": {"texto": "artículo 226 de la Ley Orgánica del Poder Judicial de la Federación",
            "corto": "vacaciones, art. 226 LOPJF"},
}
# Los días que declaró una CIRCULAR (notas 5 y 6 de la página del OAJ).
CIRCULAR_OAJ = {
    _d(2024, 3, 27): "c7_2024", _d(2024, 3, 28): "c7_2024", _d(2024, 3, 29): "c7_2024",
    _d(2024, 10, 1): "c7_2024", _d(2024, 11, 1): "c7_2024",
    _d(2025, 4, 16): "c1_2025", _d(2025, 4, 17): "c1_2025", _d(2025, 4, 18): "c1_2025",
    _d(2025, 5, 2): "c1_2025", _d(2025, 9, 15): "c3_2025",
    _d(2026, 4, 1): "c3_2026", _d(2026, 4, 2): "c3_2026", _d(2026, 4, 3): "c3_2026",
    _d(2026, 5, 4): "c3_2026", _d(2026, 9, 15): "c3_2026", _d(2026, 11, 2): "c3_2026",
}
_LFT_TRASLADO = {(2, 1): "lft_ii", (3, 3): "lft_iii", (11, 3): "lft_vi"}


def clave_del_inhabil(f: Fecha) -> str:
    """De dónde sale que ese día no corra: 'art19' | 'vac' | 'lft_*' | 'c*_AAAA'.

    EL ORDEN IMPORTA. Las vacaciones primero, para que el veinticinco de
    diciembre no parta en dos el tramo del dieciséis al treinta y uno; luego los
    días que nombra el 19; luego la circular; luego el lunes de la LFT. Lo que
    no cae en nada, al 19 (suspensión de labores), como hasta hoy."""
    if f.year >= 2025 and any(i <= f <= j for i, j in PERIODOS_OAJ.get(f.year, [])):
        return "vac"
    if (f.day, f.month) in _ART19_FIJOS:
        return "art19"
    if f in CIRCULAR_OAJ:
        return CIRCULAR_OAJ[f]
    if f.weekday() == 0:
        for (mes, n), clave in _LFT_TRASLADO.items():
            if f.month == mes and f == _lunes_ordinal(f.year, mes, n):
                return clave
    return "art19"


# ═══════════════════════════════════════════════════════════════════════════
# EL CALENDARIO DEL TRIBUNAL FEDERAL DE JUSTICIA ADMINISTRATIVA (27-sep-2026)
# ═══════════════════════════════════════════════════════════════════════════
# En la revisión fiscal el escrito se presenta ante la Sala del TFJA y los días
# hábiles son «aquellos en que se encuentren abiertas al público las oficinas
# de las Salas del Tribunal» (art. 74, fr. II, LFPCA): se cuentan los inhábiles
# DEL TFJA, «no así los días inhábiles que marca la Ley de Amparo y los periodos
# de asueto del Poder Judicial de la Federación» (tesis 2007213; en el mismo
# sentido la 239295 de la Segunda Sala). El cómputo usaba el calendario del
# PJF: sumaba sus vacaciones y omitía las del TFJA, que no coinciden (el TFJA
# para del 15 de julio y del 15 de diciembre; el PJF, del 16).
#
# LOS DATOS SON LOS ACUERDOS DEL PLENO GENERAL DE LA SALA SUPERIOR que fijan
# el «calendario oficial de suspensión de labores», verificados en el DOF:
#   2024 · SS/1/2024 (DOF 11-01-2024) y SS/4/2023 para el 1 de enero
#   2025 · SS/1/2025 (DOF 14-01-2025), con SS/22/2025 (DOF 03-12-2025)
#   2026 · SS/2/2026 (DOF 12-01-2026)
#   2027 · sin publicar al 27-sep-2026; sólo el 1 de enero (SS/2/2026)
# Cada año hay que volver a cargarlo: el calendario se publica a mediados de
# enero. Sin el año cargado, el cómputo avisa.
def _iso(x):
    return _dt.date.fromisoformat(x)


INHABILES_TFJA = {
    2024: {_iso(x) for x in (
        "2024-01-01 2024-02-05 2024-03-18 2024-03-27 2024-03-28 2024-03-29 "
        "2024-05-01 2024-07-15 2024-07-16 2024-07-17 2024-07-18 2024-07-19 "
        "2024-07-22 2024-07-23 2024-07-24 2024-07-25 2024-07-26 2024-07-29 "
        "2024-07-30 2024-07-31 2024-08-26 2024-09-16 2024-10-01 2024-11-01 "
        "2024-11-18 2024-12-16 2024-12-17 2024-12-18 2024-12-19 2024-12-20 "
        "2024-12-23 2024-12-24 2024-12-25 2024-12-26 2024-12-27 2024-12-30 "
        "2024-12-31").split()},
    2025: {_iso(x) for x in (
        "2025-01-01 2025-02-03 2025-03-17 2025-04-16 2025-04-17 2025-04-18 "
        "2025-05-01 2025-05-02 2025-05-05 2025-07-14 2025-07-15 2025-07-16 "
        "2025-07-17 2025-07-18 2025-07-21 2025-07-22 2025-07-23 2025-07-24 "
        "2025-07-25 2025-07-28 2025-07-29 2025-07-30 2025-07-31 2025-08-01 "
        "2025-08-25 2025-09-15 2025-09-16 2025-11-17 2025-12-15 2025-12-16 "
        "2025-12-17 2025-12-18 2025-12-19 2025-12-22 2025-12-23 2025-12-24 "
        "2025-12-25 2025-12-26 2025-12-29 2025-12-30 2025-12-31").split()},
    2026: {_iso(x) for x in (
        "2026-01-01 2026-01-02 2026-02-02 2026-03-16 2026-04-01 2026-04-02 "
        "2026-04-03 2026-05-01 2026-05-04 2026-05-05 2026-07-13 2026-07-14 "
        "2026-07-15 2026-07-16 2026-07-17 2026-07-20 2026-07-21 2026-07-22 "
        "2026-07-23 2026-07-24 2026-07-27 2026-07-28 2026-07-29 2026-07-30 "
        "2026-07-31 2026-08-28 2026-09-14 2026-09-15 2026-09-16 2026-10-12 "
        "2026-11-02 2026-11-16 2026-12-14 2026-12-15 2026-12-16 2026-12-17 "
        "2026-12-18 2026-12-21 2026-12-22 2026-12-23 2026-12-24 2026-12-25 "
        "2026-12-28 2026-12-29 2026-12-30 2026-12-31").split()},
    2027: {_iso("2027-01-01")},
}
# El año cargado COMPLETO (con su acuerdo). 2027 sólo trae el 1 de enero.
#
# CADA DÍA CON EL ACUERDO QUE LO FIJÓ CUANDO CORRÍA EL PLAZO (3-oct-2026). En la
# RF 6/2026 el considerando citó «los Acuerdos SS/1/2025 y SS/2/2026» para un
# plazo del 11 de diciembre de 2025 al 9 de enero de 2026, y el engrose funda el
# dos de enero de 2026 en el SS/22/2025 (DOF 03-12-2025, nota 7 de ese engrose):
# el SS/2/2026 se publicó el 12 de enero, DESPUÉS de esos días. Por eso:
#   · el calendario de 2025 es el SS/1/2025 «con SS/22/2025» (el comentario de
#     arriba), y el SS/22/2025 se cita sólo para los días que caen después de
#     su publicación —antes no existía y no pudo fijar nada—;
#   · el 1 de enero de cada año lo fija el calendario del año ANTERIOR (2024:
#     SS/4/2023; 2027: SS/2/2026, también arriba);
#   · el 2 de enero de 2026, el SS/22/2025 (la nota del engrose).
_ACUERDOS_TFJA_ANIO = {
    2023: (("SS/4/2023", None),),
    2024: (("SS/1/2024", _dt.date(2024, 1, 11)),),
    2025: (("SS/1/2025", _dt.date(2025, 1, 14)), ("SS/22/2025", _dt.date(2025, 12, 3))),
    2026: (("SS/2/2026", _dt.date(2026, 1, 12)),),
}
_ACUERDO_TFJA_DIA = {_dt.date(2026, 1, 2): ("SS/22/2025",)}


def _lista_de_acuerdos(acs) -> str:
    acs = list(acs)
    if not acs:
        return ""
    return (f"el Acuerdo {acs[0]}" if len(acs) == 1
            else "los Acuerdos " + ", ".join(acs[:-1]) + " y " + acs[-1])


# Los años que no son sólo el 1 de enero del siguiente (2023 está por ése).
ACUERDO_TFJA = {a: _lista_de_acuerdos(c for c, _f in xs)
                for a, xs in _ACUERDOS_TFJA_ANIO.items() if a >= 2024}


def _acuerdos_del_anio(anio: int, hasta=None) -> tuple:
    """Los acuerdos del calendario de `anio` publicados a la fecha `hasta`
    (el primero, el del calendario, siempre)."""
    xs = _ACUERDOS_TFJA_ANIO.get(anio, ())
    return tuple(c for i, (c, f) in enumerate(xs)
                 if i == 0 or f is None or hasta is None or f <= hasta)


def acuerdos_tfja_del_dia(d) -> tuple:
    """El acuerdo (o acuerdos) del Pleno General de la Sala Superior que hizo
    inhábil ese día, como regía cuando corrió."""
    if d in _ACUERDO_TFJA_DIA:
        return _ACUERDO_TFJA_DIA[d]
    if (d.month, d.day) == (1, 1) and (d.year - 1) in _ACUERDOS_TFJA_ANIO:
        return _acuerdos_del_anio(d.year - 1, d)
    return _acuerdos_del_anio(d.year, d)
# Suspensiones LOCALES: sólo para la Sala que las dictó. Se aplican cuando el
# nombre de la responsable la identifica.
INHABILES_TFJA_LOCAL = {
    # SRQ/01/2026 (DOF 05-03-2026): «no correrán términos y plazos procesales»
    # el 23 de febrero de 2026 en la Sala Regional en Querétaro.
    "quer[ée]taro": {_iso("2026-02-23")},
}

CALENDARIO_TFJA = Calendario(
    nombre="Tribunal Federal de Justicia Administrativa",
    fundamento=("artículo 74, fracción II, de la Ley Federal de Procedimiento "
                "Contencioso Administrativo"),
    sueltos=set().union(*INHABILES_TFJA.values()),
)


def calendario_tfja(responsable: str = "") -> "Calendario":
    """El del TFJA con las suspensiones locales de la Sala que se nombra."""
    import copy as _copy
    c = _copy.deepcopy(CALENDARIO_TFJA)
    for patron, dias in INHABILES_TFJA_LOCAL.items():
        if re.search(patron, responsable or "", re.I):
            c.sueltos = set(c.sueltos) | set(dias)
    return c


def fundamento_tfja(anios, dias=None, hasta=None) -> str:
    """«artículo 74, fracción II, de la LFPCA y el Acuerdo SS/2/2026…».

    Con `dias` (los inhábiles entre semana del TFJA que el cómputo saltó) se
    citan los acuerdos que fijaron ESOS días (`acuerdos_tfja_del_dia`); sin
    ellos, los de los años del plazo vigentes a la fecha `hasta` (el
    vencimiento)."""
    acs: list = []
    _dias = sorted(d for d in (dias or []) if d is not None)
    if _dias:
        for d in _dias:
            for a in acuerdos_tfja_del_dia(d):
                if a not in acs:
                    acs.append(a)
    else:
        for anio in sorted(set(anios or [])):
            if anio not in ACUERDO_TFJA:
                continue
            _h = hasta if (hasta is not None and hasta.year == anio) else None
            for a in _acuerdos_del_anio(anio, _h):
                if a not in acs:
                    acs.append(a)
    base = CALENDARIO_TFJA.fundamento
    if not acs:
        return base
    ac = (f"del Acuerdo {acs[0]}" if len(acs) == 1
          else "de los Acuerdos " + ", ".join(acs[:-1]) + " y " + acs[-1])
    return (f"{base}, y {ac} del Pleno General de la Sala Superior del Tribunal "
            f"Federal de Justicia Administrativa, que "
            f"{'fija' if len(acs) == 1 else 'fijan'} su calendario de "
            f"suspensión de labores")


# El día en que el surtimiento del Boletín del TFJA pasa del tercero al segundo
# día hábil: 240 días naturales desde el 09-06-2026 (LFPCA, transitorio Tercero).
LFPCA_65_REFORMADO = _dt.date(2027, 2, 4)


CALENDARIO_AMPARO = Calendario(
    nombre="Poder Judicial de la Federación",
    fundamento="artículo 19 de la Ley de Amparo",
    # El art. 19 dice «catorce Y dieciséis de septiembre» — el 14 faltaba.
    # Y también «cinco de febrero, veintiuno de marzo… veinte de noviembre»
    # (verificación de normas, 27-sep-2026): hasta 2026 los cubría la lista de
    # la OAJ, pero desde 2027, sin esa lista cargada, el viernes 5-feb-2027
    # contaría como hábil contra la letra del artículo. Los lunes trasladados de
    # la Ley Federal del Trabajo se conservan: la OAJ declara inhábiles ambos.
    fijos={(1, 1), (5, 2), (21, 3), (1, 5), (5, 5), (14, 9), (16, 9), (12, 10),
           (20, 11), (25, 12)},
    trasladados={2: 1, 3: 3, 11: 3},
    sueltos=set().union(*INHABILES_OAJ.values()),
    periodos=[p for v in PERIODOS_OAJ.values() for p in v],
)


# ── Calendarios de autoridades responsables ───────────────────────────────
#
# Cada responsable tiene el suyo. Aquí sólo van los verificados; el resto se
# añade conforme se confirmen sus acuerdos de suspensión de labores. Mientras
# una responsable no esté declarada, `computar` avisa y usa el federal, que es
# la aproximación menos mala, PERO deja constancia de que fue una aproximación.
CALENDARIOS_RESPONSABLE: dict[str, Calendario] = {
    # Poder Judicial del Estado de Querétaro — calendario 2026, copiado de
    #     https://www.poderjudicialqro.gob.mx/nv/calendario.php
    # (acuerdo del Consejo de la Judicatura, art. 140 fr. XII de su Ley
    # Orgánica). El propio acuerdo dice: «Los plazos procesales no se
    # computarán en los días inhábiles».
    #
    # MIRA LO DISTINTO QUE ES DEL FEDERAL, que es justo la razón de que haya
    # dos calendarios: sus vacaciones de julio van del 20 al 31 (el PJF, del
    # 16 al 31), las de diciembre del 17 al 31 (el PJF, del 16), y además
    # tiene DOS periodos extraordinarios que el federal no contempla.
    #
    # OJO: esto es el PODER JUDICIAL del estado —materia civil, familiar,
    # penal local—. El Tribunal de Justicia Administrativa de Querétaro, que
    # fue la responsable del ADA 240/2026, es otro órgano y tiene su propio
    # calendario, que aún no está aquí.
    "pj_queretaro": Calendario(
        nombre="Poder Judicial del Estado de Querétaro",
        fundamento="acuerdo del Consejo de la Judicatura del Estado",
        sueltos={
            _d(2026, 1, 1), _d(2026, 2, 2), _d(2026, 3, 16), _d(2026, 4, 2),
            _d(2026, 4, 3), _d(2026, 5, 1), _d(2026, 9, 15), _d(2026, 9, 16),
            _d(2026, 11, 2), _d(2026, 11, 16), _d(2026, 12, 25),
        },
        periodos=[
            (_d(2026, 6, 29), _d(2026, 7, 10)),   # primer extraordinario
            (_d(2026, 7, 20), _d(2026, 7, 31)),   # primer ordinario
            (_d(2026, 12, 17), _d(2026, 12, 31)), # segundo ordinario
            (_d(2027, 1, 11), _d(2027, 1, 22)),   # segundo extraordinario
        ],
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Cuándo surte efectos — depende de la ley que rige el ACTO, no del amparo
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ReglaSurte:
    """Cómo surte efectos una notificación en una materia y por una vía.

    `dias_habiles` = cuántos hábiles después de la notificación surte. Cero
    significa el mismo día.
    """
    clave: str
    descripcion: str          # va literal al considerando
    dias_habiles: int
    fundamento: str = ""


# Las reglas VERIFICADAS contra engroses reales. No se inventan: cada una entra
# aquí cuando se ha leído en un documento firmado o David la confirma.
#
#   · `tja_qro_boletin` sale del ADA 240/2026, comprobado al día.
#
# Lo que falta —laboral, civil local, fiscal federal, penal— se añade igual: se
# lee de un engrose, se comprueba que el cómputo lo reproduce, y entra.
REGLAS_SURTE: dict[str, ReglaSurte] = {
    # ═══════════════════════════════════════════════════════════════════
    # LA CLAVE DICE DE QUIÉN ES LA REGLA, Y ESO IMPORTA
    # ═══════════════════════════════════════════════════════════════════
    # `tja_qro_boletin` es del Tribunal de Justicia Administrativa DE
    # QUERÉTARO, y era el valor POR OMISIÓN de todo el pipeline. Un secretario
    # de Yucatán, de Jalisco o de Nuevo León generaba su proyecto y el cómputo
    # se hacía con la regla de otro estado, en silencio. Un plazo mal contado
    # invalida la sentencia: es el peor sitio donde heredar un valor ajeno.
    #
    # Ahora la omisión es `personal`, que es la regla general del artículo 31,
    # fracción I, de la Ley de Amparo y vale en toda la república; las reglas
    # locales se piden por su clave y se avisa de a quién pertenecen.
    "tja_qro_boletin": ReglaSurte(
        clave="tja_qro_boletin",
        descripcion="mediante Boletín Jurisdiccional",
        dias_habiles=3,
        fundamento="",
    ),
    "personal": ReglaSurte(
        clave="personal",
        descripcion="de manera personal",
        # Al DÍA HÁBIL SIGUIENTE, no el mismo día. Derivado de los engroses
        # ADA 448-2025 y 449-2025: notificación personal el 7 de abril, plazo
        # iniciado el 9. Con surtimiento el mismo día habría iniciado el 8.
        dias_habiles=1,
    ),
    "lista": ReglaSurte(
        clave="lista",
        descripcion="por lista",
        dias_habiles=1,
    ),
    # ═══════════════════════════════════════════════════════════════════
    # CONTENCIOSO ADMINISTRATIVO FEDERAL: EL BOLETÍN SURTE AL TERCER DÍA
    # ═══════════════════════════════════════════════════════════════════
    # David, 22-sep-2026, sobre el ADC 93/2026: «en materia federal no hay
    # duda. La Ley Federal de Procedimiento Contencioso Administrativo
    # establece que la notificación por boletín surte efectos a los tres días
    # siguientes. Esa es la opción que debe desplegarse».
    #
    # Consultado en el acervo (leyes_federales), artículo 65, último párrafo:
    #     «La notificación surtirá sus efectos al tercer día hábil siguiente
    #      a aquél en que se haya realizado la publicación en el Boletín
    #      Jurisdiccional o al día hábil siguiente a aquél en que las partes
    #      sean notificadas personalmente en las instalaciones designadas
    #      por el Tribunal».
    # Comprobado con el 93/2026: publicación en boletín el 5 de diciembre de
    # 2025 (viernes) → 8, 9, 10 → surtió el 10, que es la fecha que David
    # había tenido que declarar a mano con «otra regla».
    "lfpca_boletin": ReglaSurte(
        clave="lfpca_boletin",
        descripcion="mediante Boletín Jurisdiccional",
        dias_habiles=3,
        fundamento="artículo 65 de la Ley Federal de Procedimiento Contencioso Administrativo",
    ),
    # Y LA PERSONAL ante el propio Tribunal Federal: al día hábil siguiente
    # (artículo 65, mismo párrafo; el 70 lo repite en general). La clave
    # `lfpca` se conserva por los encargos guardados que la traen.
    "lfpca": ReglaSurte(
        clave="lfpca",
        descripcion="de manera personal",
        dias_habiles=1,
        fundamento="artículos 65 y 70 de la Ley Federal de Procedimiento Contencioso Administrativo",
    ),
    # ═══════════════════════════════════════════════════════════════════
    # LAS DOS REGLAS DEL 31 QUE FALTABAN (3-oct-2026)
    # ═══════════════════════════════════════════════════════════════════
    # En el AR del XXII Circuito recurrió el Director de Ingresos y el
    # considerando escribió «surtió efectos al día hábil siguiente, conforme
    # al artículo 31, fracción II»: la regla de los PARTICULARES aplicada a una
    # autoridad. A las autoridades se les notifica por oficio y la
    # notificación surte efectos «desde el momento en que hayan quedado
    # legalmente hechas» (art. 31, fr. I, LA); la electrónica, cuando se
    # genera la constancia de la consulta (fr. III). Un día de diferencia en
    # el arranque del plazo, en el supuesto frecuente de la autoridad que
    # recurre la concesión. El precepto lo escribe `fundamento_de_surtimiento`
    # porque sólo es de la Ley de Amparo en los recursos (en el amparo directo
    # la notificación del acto la rige la ley del acto).
    "oficio": ReglaSurte(
        clave="oficio",
        descripcion="por oficio",
        dias_habiles=0,
    ),
    "electronica": ReglaSurte(
        clave="electronica",
        descripcion="por vía electrónica",
        dias_habiles=0,
    ),
    # ═══════════════════════════════════════════════════════════════════
    # LO AGRARIO CON EL CÓDIGO NACIONAL (3-oct-2026)
    # ═══════════════════════════════════════════════════════════════════
    # David: «Si va con el Código Nacional, y no el Federal, entonces hay que
    # adecuar al Código Nacional». La Ley Agraria, artículo 167 (reformado DOF
    # 14-11-2025): «El Código Nacional de Procedimientos Civiles y Familiares
    # es de aplicación supletoria…». El transitorio Segundo de esa reforma ata
    # su aplicación a la declaratoria del Código Nacional (en las entidades, la
    # de su Congreso; en el orden federal, la del Congreso de la Unión;
    # automática el 1-abr-2027) y el Tercero deja los juicios en trámite con la
    # legislación con que empezaron. Dónde opera ya lo decide la sede
    # (`tipos_asunto.es_cdmx`, David: el CFPC en toda la república salvo la
    # Ciudad de México); el guardián de materia de `redactor_adelanto` cambia a
    # éstas la «personal» y la «lista» que llegan por omisión. Cotejado con el
    # texto del CNPCF descargado del DOF:
    #   · art. 227, fr. I — «Los términos empezarán a correr: I. El día
    #     siguiente en que se hubiere hecho el emplazamiento o notificación
    #     personal»: el plazo corre desde el día siguiente al de la notificación,
    #     esto es, surte EL MISMO DÍA (cero hábiles);
    #   · art. 211 — lo publicado en el medio de comunicación procesal oficial
    #     se da «por hecha el día de su publicación y surtiendo sus efectos al
    #     día siguiente» (un hábil);
    #   · art. 227, fr. III — la notificación por medio de comunicación judicial
    #     «surtirá efectos el mismo día a aquel en que por sistema se confirme
    #     que recibió el archivo electrónico» (cero hábiles).
    # El 1-abr-2027 el Código Nacional rige en toda la república (salvo los
    # juicios iniciados antes): entonces estas reglas dejan de ser sólo de la
    # Ciudad de México y hay que revisar el guardián y `surtimiento_nacional`.
    "cnpcf_personal": ReglaSurte(
        clave="cnpcf_personal",
        descripcion="de manera personal",
        dias_habiles=0,
        fundamento="artículo 227, fracción I, del Código Nacional de Procedimientos Civiles y "
                   "Familiares, de aplicación supletoria en términos del artículo 167 de la "
                   "Ley Agraria",
    ),
    "cnpcf_lista": ReglaSurte(
        clave="cnpcf_lista",
        descripcion="por lista",
        dias_habiles=1,
        fundamento="artículo 211 del Código Nacional de Procedimientos Civiles y Familiares, "
                   "de aplicación supletoria en términos del artículo 167 de la Ley Agraria",
    ),
    "cnpcf_electronica": ReglaSurte(
        clave="cnpcf_electronica",
        descripcion="por vía electrónica",
        dias_habiles=0,
        fundamento="artículo 227, fracción III, del Código Nacional de Procedimientos Civiles "
                   "y Familiares, de aplicación supletoria en términos del artículo 167 de la "
                   "Ley Agraria",
    ),
    # LA PERSONAL DEL CÓDIGO FEDERAL, ELEGIDA A PROPÓSITO (integración, 3-oct-
    # 2026). En la Ciudad de México el guardián de materia cambia la «personal»
    # que llega por omisión por la del Código Nacional; el secretario cuyo
    # juicio agrario empezó antes de que el Código Nacional rigiera para él
    # (transitorio Tercero de la reforma al 167) necesita poder decir «el 321,
    # de verdad», y con la «personal» genérica no había forma: su elección y la
    # omisión del formulario eran la misma clave. Ésta es la suya; el guardián
    # no la toca. Surte al día siguiente: «Toda notificación surtirá sus
    # efectos el día siguiente al en que se practique» (art. 321 CFPC). Fuera
    # de la Ciudad de México es también la que el desplegable propone.
    "cfpc_personal": ReglaSurte(
        clave="cfpc_personal",
        descripcion="de manera personal",
        dias_habiles=1,
        fundamento="artículo 321 del Código Federal de Procedimientos Civiles, de aplicación "
                   "supletoria en materia agraria",
    ),
}

# La regla del Código Nacional que sustituye a la «de siempre» cuando ésta
# llega por omisión en lo agrario de la Ciudad de México (el guardián de
# `redactor_adelanto.regla_agraria`).
CNPCF_POR_OMISION = {"personal": "cnpcf_personal", "lista": "cnpcf_lista"}


# ═══════════════════════════════════════════════════════════════════════════
# QUÉ REGLAS SE OFRECEN — según la ley que rige el acto, no según Querétaro
# ═══════════════════════════════════════════════════════════════════════════
# El desplegable de la pantalla ofrecía siempre las mismas cinco, con la del
# Boletín de Querétaro entre ellas para todos. David: «este redactor no es
# exclusivamente para Querétaro. Si se deja abierta la posibilidad de un
# plazo diverso (…) es porque la ley que rige el acto así lo prevea (según la
# entidad federativa) pero en materia federal no hay duda».
#
# Aquí vive la única lista: la pantalla la pide (/taller/reglas-surtimiento)
# con el tipo de asunto y la responsable, y pinta lo que vuelve.

_RX_TFJA = re.compile(
    r"tribunal\s+federal\s+de\s+justicia\s+(?:fiscal\s+y\s+)?administrativa|TFJA|"
    r"sala\s+(?:regional|superior|especializada)", re.I)
_RX_TJA_QRO = re.compile(
    r"tribunal\s+de\s+justicia\s+administrativa\s+del\s+estado\s+de\s+quer[ée]taro|"
    r"TJA[^.]{0,40}quer[ée]taro", re.I)
_RX_TJA_ESTATAL = re.compile(
    r"tribunal\s+de\s+justicia\s+administrativa\s+(?:del\s+estado|de\s+la\s+ciudad|de\s+\w+)|"
    r"tribunal\s+(?:estatal\s+)?de\s+lo\s+contencioso\s+administrativo", re.I)


_RX_TFJA_NOMBRE = re.compile(
    r"tribunal\s+federal\s+de\s+justicia\s+(?:fiscal\s+y\s+)?administrativa|\bTFJA\b|\bTFJFA\b", re.I)
_RX_ESTATAL = re.compile(r"del\s+estado\b|estatal|de\s+la\s+ciudad\s+de\s+m[ée]xico|"
                         r"del\s+distrito\s+federal|\bTJA\b", re.I)
# LOS TRIBUNALES AGRARIOS (3-oct-2026): «Tribunal Unitario Agrario del Distrito
# 8», «Tribunal Superior Agrario», «Tribunal Agrario». Su notificación la rige
# la Ley Agraria y, en lo que no dice, su supletorio (art. 167).
_RX_AGRARIO_ORGANO = re.compile(
    r"\btribunal(?:es)?\s+(?:(?:unitario|superior)s?\s+)?agrari[oa]s?\b|"
    r"\bmagistrad[oa]\s+(?:del\s+tribunal\s+)?(?:unitario\s+)?agrari[oa]\b", re.I)
_RX_JUICIO_AGRARIO = re.compile(
    r"\b(?:juicio|procedimiento|controversia|expediente)\s+agrari[oa]\b", re.I)


def es_agrario(materia: str = "", responsable: str = "", expediente: str = "",
               contexto: str = "") -> str:
    """De dónde se sabe que el juicio de origen es agrario: «materia»,
    «órgano», «expediente», «contexto», o «» si no se sabe (como
    `es_mercantil`). Lo agrario suele registrarse como «AMPARO DIRECTO
    ADMINISTRATIVO» (art. 35, fr. I, inciso b), LOPJF: la administrativa,
    «incluida la agraria»), así que la materia sola no basta."""
    if (materia or "").strip().lower() in ("agraria", "agrario"):
        return "materia"
    if _RX_AGRARIO_ORGANO.search(" ".join(str(responsable or "").split())):
        return "órgano"
    if _RX_JUICIO_AGRARIO.search(str(expediente or "")):
        return "expediente"
    if _RX_JUICIO_AGRARIO.search(str(contexto or "")):
        return "contexto"
    return ""


def sede_cdmx(tribunal: str = "", ciudad: str = ""):
    """True | False | None: la MISMA regla de sede que el supletorio de la Ley de
    Amparo (`tipos_asunto.es_cdmx`: el Primer Circuito, o la ciudad). None si
    no se puede decidir o si la pieza falla: entonces no se afirma la Ciudad de
    México."""
    if not (str(tribunal or "").strip() or str(ciudad or "").strip()):
        return None
    try:
        import tipos_asunto as _ta_sede
        _r = _ta_sede.es_cdmx(str(tribunal or ""), str(ciudad or ""))
        return _r if isinstance(_r, bool) else None
    except Exception:
        return None


def fuero_de(tipo_asunto: str = "", responsable: str = "") -> str:
    """'tfja' | 'tja_qro' | 'tja_estatal' | 'agrario' | 'federal' | 'local' | ''."""
    r = " ".join((responsable or "").split())
    # LO AGRARIO, ANTES QUE NADA (3-oct-2026): un Tribunal Unitario Agrario es
    # federal, pero su notificación no es la de cualquier órgano federal.
    if _RX_AGRARIO_ORGANO.search(r):
        return "agrario"
    if _RX_TJA_QRO.search(r):
        return "tja_qro"
    if _RX_TJA_ESTATAL.search(r):
        return "tja_estatal"
    # EL TFJA, por su nombre o por sus siglas; y una «Sala Regional» o «Sala
    # Superior» que no diga «del Estado» es la del federal —los tribunales
    # estatales se nombran por su entidad—.
    if _RX_TFJA_NOMBRE.search(r) or (_RX_TFJA.search(r) and not _RX_ESTATAL.search(r)):
        return "tfja"
    if (tipo_asunto or "").strip().lower() == "revision_fiscal":
        return "tfja"
    try:
        from fase_normas import autoridad_es_federal
        if autoridad_es_federal(r):
            return "federal"
    except Exception:
        pass
    if re.search(r"del\s+estado|estatal|de\s+la\s+ciudad\s+de\s+m[ée]xico", r, re.I):
        return "local"
    return ""


def reglas_para(tipo_asunto: str = "", responsable: str = "", papel: str = "",
                tribunal: str = "", ciudad: str = "") -> dict:
    """{fuero, por_omision, reglas: [{clave, etiqueta, dias_habiles, fundamento}]}

    Lo que el desplegable enseña para ESTE asunto. «Otra regla» va siempre:
    es la salida cuando la ley del acto prevé algo que el catálogo no trae.

    `tribunal` y `ciudad` (los del colegiado, 3-oct-2026) sólo cuentan en lo
    agrario: deciden si la regla de omisión es la del Código Nacional (sede en
    la Ciudad de México) o la de siempre (el 321 del Código Federal).
    """
    f = fuero_de(tipo_asunto, responsable)

    def _r(clave, etiqueta):
        x = REGLAS_SURTE[clave]
        return {"clave": clave, "etiqueta": etiqueta,
                "dias_habiles": x.dias_habiles, "fundamento": x.fundamento}

    otra = {"clave": "otra", "etiqueta": "Otra regla — yo declaro cuándo surtió efectos",
            "dias_habiles": -1, "fundamento": ""}
    # EN LOS RECURSOS DE AMPARO, LAS REGLAS DEL 31 COMPLETAS (3-oct-2026): lo
    # notificado es una resolución del juicio de amparo, sea quien sea la
    # responsable del acto, así que va antes que el fuero. Si recurre una
    # autoridad, la suya —oficio, desde que queda hecha— es la de omisión.
    _t = (tipo_asunto or "").strip().lower()
    if _t in ("amparo_revision", "queja"):
        _del_31 = [_r("oficio", "Por oficio a autoridad — surte desde que queda hecha (art. 31, fr. I, LA)"),
                   _r("electronica", "Electrónica — surte al generarse la constancia de consulta (art. 31, fr. III, LA)")]
        _base = ([_r("personal", "Personal — surte al día hábil siguiente (art. 31, fr. II, LA)"),
                  _r("lista", "Por lista — surte al día hábil siguiente (art. 31, fr. II, LA)")])
        _aut = (papel or "").strip().lower() == "autoridad"
        return {"fuero": f, "por_omision": "oficio" if _aut else "personal",
                "reglas": (_del_31 + _base if _aut else _base + _del_31) + [otra]}
    if f == "agrario":
        # LO AGRARIO: LAS TRES DEL CÓDIGO NACIONAL Y LAS DOS DE SIEMPRE
        # (3-oct-2026). David: «Si va con el Código Nacional, y no el Federal,
        # entonces hay que adecuar al Código Nacional». Por omisión, la de la
        # sede: en la Ciudad de México, la personal del Código Nacional; fuera
        # de ella (o sin saberlo), la personal del Código Federal (art. 321). Las
        # cinco se ofrecen siempre: el transitorio Tercero de la reforma al 167
        # deja los juicios iniciados antes con su código, y el 1-abr-2027 el
        # Código Nacional rige en toda la república.
        # LA PERSONAL DEL CÓDIGO FEDERAL ES `cfpc_personal`, NO LA GENÉRICA
        # (integración, 3-oct-2026): elegida a propósito en la Ciudad de México
        # ya no se confunde con la «personal» del formulario, que el guardián de
        # materia cambia por la del Código Nacional. Fuera de ella el resultado
        # es el mismo que con la genérica (el 321 y el aviso del transitorio,
        # `surtimiento_nacional`).
        _cnpcf = [
            _r("cnpcf_personal", "Personal — Código Nacional (art. 227, fr. I): el plazo corre al día siguiente"),
            _r("cnpcf_lista", "Por lista — Código Nacional (art. 211): surte al día siguiente"),
            _r("cnpcf_electronica", "Electrónica — Código Nacional (art. 227, fr. III): surte el día en que "
                                    "el sistema confirma la recepción")]
        _cfpc = [
            _r("cfpc_personal", "Personal — Código Federal (art. 321): surte al día siguiente"),
            _r("lista", "Por lista — Código Federal (art. 321): surte al día siguiente")]
        if sede_cdmx(tribunal, ciudad) is True:
            return {"fuero": f, "por_omision": "cnpcf_personal", "reglas": _cnpcf + _cfpc + [otra]}
        return {"fuero": f, "por_omision": "cfpc_personal", "reglas": _cfpc + _cnpcf + [otra]}
    if f == "tfja":
        return {"fuero": f, "por_omision": "lfpca_boletin", "reglas": [
            _r("lfpca_boletin", "Boletín Jurisdiccional del TFJA — surte al tercer día hábil (art. 65 LFPCA)"),
            _r("lfpca", "Personal ante el TFJA — surte al día hábil siguiente (arts. 65 y 70 LFPCA)"),
            otra]}
    if f == "tja_qro":
        return {"fuero": f, "por_omision": "tja_qro_boletin", "reglas": [
            _r("tja_qro_boletin", "Boletín del TJA de Querétaro — surte al tercer día hábil"),
            _r("personal", "Personal — surte al día hábil siguiente"),
            otra]}
    if f == "tja_estatal":
        # LA LEY DE ESA ENTIDAD DICE CÓMO SURTE; el catálogo aún no la trae.
        return {"fuero": f, "por_omision": "otra", "reglas": [
            otra,
            # NO ES EL 31 DE LA LEY DE AMPARO: la notificación del acto la rige
            # la ley de esa entidad (art. 18 LA, «conforme a la ley del acto»).
            # Decía «art. 31, fr. I», que además es la de las autoridades.
            _r("personal", "Personal — surte al día hábil siguiente (según la ley del acto)")]}
    base = [_r("personal", "Personal — surte al día hábil siguiente"),
            _r("lista", "Por lista — surte al día hábil siguiente")]
    if f in ("federal", ""):
        # Sin saber la responsable, se ofrece todo lo federal y la salida.
        return {"fuero": f, "por_omision": "personal", "reglas": base + [
            _r("lfpca_boletin", "Boletín Jurisdiccional del TFJA — surte al tercer día hábil (art. 65 LFPCA)"),
            otra]}
    return {"fuero": f, "por_omision": "personal", "reglas": base + [
        _r("tja_qro_boletin", "Boletín del TJA de Querétaro — surte al tercer día hábil (sólo asuntos de Querétaro)"),
        otra]}


def surtio_manual_de(encargo) -> tuple:
    """(fecha | None, aviso) — la fecha en que el secretario declaró que
    surtió efectos, cuando eligió «otra regla».

    UNA SOLA PUERTA. La leía el adelanto y NO la leía la reconstrucción de la
    sesión desde la base, así que el worker que resolvía rehacía el cómputo
    sin ella: el 93/2026 salía «personal · surtió el 8 · vence el 15 de enero
    · EXTEMPORÁNEA» con la fecha del 10 declarada a mano, que da el 19 y en
    tiempo. Cuarta vez que el reparto entre workers muerde en el cómputo.
    """
    if str(getattr(encargo, "regla_surtimiento", "") or "").strip().lower() != "otra":
        return None, ""
    se = str(getattr(encargo, "surte_efectos", "") or "").strip()
    if not se:
        return None, ("Elegiste «otra regla» de notificación pero no diste la "
                      "fecha en que surtió efectos; el cómputo siguió con la "
                      "notificación personal. Declárala para que el considerando "
                      "cuente el plazo con la fecha que tú diste, no con una regla "
                      "que no aplicaste.")
    try:
        import datetime as _dt
        return _dt.date.fromisoformat(se[:10]), ""
    except ValueError:
        return None, (f"La fecha «{se}» en que dijiste que surtió efectos la "
                      f"notificación no es válida (usa AAAA-MM-DD); el cómputo "
                      f"siguió con la notificación personal.")


# ═══════════════════════════════════════════════════════════════════════════
# El cómputo
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Computo:
    notificacion: Fecha
    regla: ReglaSurte
    surtio: Fecha
    inicio: Fecha
    vencimiento: Fecha
    plazo: int
    dias: list[Fecha]
    cal_responsable: Calendario
    cal_amparo: Calendario
    presentacion: Optional[Fecha] = None
    inhabiles_en_medio: list[Fecha] = field(default_factory=list)
    # LOS INHÁBILES ENTRE LA NOTIFICACIÓN Y EL INICIO DEL PLAZO (3-oct-2026).
    # Mueven el surtimiento o el arranque y el considerando tiene que nombrarlos:
    # en el AD 335/2025 el viernes 21 de marzo empujó el plazo al lunes 24, y en
    # la Q 24/2026 las vacaciones del 19 al 31 de diciembre y el 1 de enero lo
    # llevaron al 2 de enero; el párrafo saltaba del 18 de diciembre al 2 de
    # enero sin decir por qué. Van APARTE de `inhabiles_en_medio` (que es el
    # tramo del plazo y lo leen la tabla y las pruebas) y nunca con los días de
    # la responsable, que sólo cuentan dentro del plazo.
    inhabiles_previos: list[Fecha] = field(default_factory=list)
    # LA REVISIÓN FISCAL POR CORREO (D3, 3-oct-2026): `presentacion` es la fecha
    # con que se midió la oportunidad —la del DEPÓSITO en el Servicio Postal
    # Mexicano— y `recepcion`, la del sello de la Sala. Vacíos fuera de ese caso.
    deposito: Optional[Fecha] = None
    recepcion: Optional[Fecha] = None
    # Avisos que el secretario TIENE que leer antes de firmar.
    avisos: list[str] = field(default_factory=list)
    # NO HAY PLAZO QUE CONTAR. Los recursos que proceden en cualquier tiempo
    # —peligro de privación de la vida o ataques a la libertad (artículo 17,
    # fracción IV) y omisión de tramitar la demanda de amparo (artículo 98,
    # fracción II)— no tienen vencimiento, y por tanto no pueden ser
    # extemporáneos. Es un CAMPO, no una propiedad, porque quien lo sabe es el
    # catálogo de tipos y tiene que poder decírselo al cómputo.
    sin_plazo: bool = False
    # PLAZOS DE AÑOS (art. 17, fracciones II y III, LA): ocho y siete años no
    # se cuentan en días hábiles. Cuando va, `plazo` es 0 y `dias` va vacío.
    plazo_anios: int = 0

    # LOS DÍAS DE LA RESPONSABLE, y si de verdad se aplicaron. SEPARADOS de
    # `inhabiles_en_medio` porque el considerando los funda distinto: aquéllos
    # son el artículo 19 de la Ley de Amparo y éstos el acuerdo de la
    # responsable más el artículo 176 (P./J. 4/2022, registro 2024494).
    resp_dias: list = field(default_factory=list)
    resp_en_medio: list = field(default_factory=list)
    resp_tramos_en_medio: list = field(default_factory=list)
    # LA MEDICIÓN DE QUE LA CAPA ESTÁ ENCENDIDA. Viajan a estado.computo.
    resp_declarados: bool = False
    resp_aplicados: bool = False
    receptor: object = None
    responsable_nombre: str = ""

    # ═══════════════════════════════════════════════════════════════════════
    # LO QUE EL SECRETARIO RESOLVIÓ AL LEER EL CÓMPUTO
    # ═══════════════════════════════════════════════════════════════════════
    # David, 12-sep-2026: «si el cómputo es extemporáneo sólo avisar, pero
    # nunca impedir el estudio de fondo si el secretario decide generar
    # proyecto de fondo. Recuerda que la tarjeta final gobierna el proyecto».
    #
    # ESTO NO APAGA EL CÓMPUTO. La aritmética de arriba no se toca nunca:
    # `oportuna` sigue diciendo lo que dicen las fechas y el desglose sigue
    # saliendo entero en el papel. Lo que se guarda aquí es OTRA COSA —qué
    # resolvió quien lo leyó— y son dos vías distintas, porque son dos
    # afirmaciones jurídicas distintas y no se pueden confundir en una casilla:
    #
    #   «oportuna» → el cómputo automático parte de un dato incompleto y él
    #                afirma que el escrito se presentó EN TIEMPO. La ejecutoria
    #                declara la oportunidad CON SU RAZÓN escrita y entra al
    #                fondo. Es el caso del amparo directo 93/2026: la Sala
    #                Regional del TFJA cerró dos días que el calendario federal
    #                tiene por hábiles, y con ellos el asunto está en plazo.
    #   «reserva»  → el cómputo es correcto y el escrito SÍ es extemporáneo,
    #                pero quiere el estudio escrito para llevarlo a sesión. La
    #                ejecutoria no cambia —sobresee o desecha— y el estudio va
    #                DETRÁS de los resolutivos, rotulado como lo que es.
    #
    # Por qué no una sola casilla «genera el fondo igualmente»: porque las dos
    # vías producen documentos incompatibles y quien firma tiene que haber
    # elegido cuál. Una casilla obliga al compositor a adivinar, y adivinar
    # aquí es firmar «se sobresee» debajo de un estudio que concede.
    #
    # El motivo no es un campo administrativo: se imprime LITERAL en el
    # considerando (vía «oportuna») o en la cabecera del anexo (vía
    # «reserva»). Quien lo teclea lo firma, y eso es lo que impide que sea un
    # clic por inercia — un clic no se puede reproducir en un considerando.
    decision: str = ""              # "" | "oportuna" | "reserva"
    motivo: str = ""
    # ¿LA RESERVA LA PIDIÓ ÉL O SE APLICÓ SOLA? Cuando el secretario califica
    # el fondo y el cómputo da extemporánea, la reserva se estampa sin que él
    # elija nada: el proyecto sale completo, que es lo que pidió David tres
    # veces. Pero entonces NO hubo petición ni razón declarada, y el anexo no
    # puede decir que las hubo — lo firma una persona.
    decision_automatica: bool = False

    @property
    def en_cualquier_tiempo(self) -> bool:
        """Los tipos sin plazo —vida, libertad— no pueden ser extemporáneos.

        ESTA PROPIEDAD YA EXISTÍA Y SU DOCSTRING MENTÍA EN PASADO. Decía que
        `computar` con plazo 0 «reventaba» con un IndexError, y el 1-sep-2026 se
        comprobó que seguía reventando: `sumar(inicio, 0)` devuelve `[]` y la
        línea siguiente hace `dias[-1]`. La propiedad describía una defensa que
        nadie había construido. Ahora sí la hay, en `computar`, y esto sólo la
        lee.
        """
        if getattr(self, "plazo_anios", 0):
            return False
        return bool(self.sin_plazo) or not self.plazo or self.plazo <= 0

    @property
    def anticipada(self) -> bool:
        """Se presentó ANTES de que el plazo arrancara."""
        return (self.presentacion is not None
                and self.presentacion < self.inicio)

    @property
    def oportuna(self) -> Optional[bool]:
        """NADIE LLEGA TARDE POR LLEGAR TEMPRANO.

        Esta comparación decía `self.inicio <= self.presentacion`, y con ello
        declaraba EXTEMPORÁNEO lo presentado antes de que el plazo arrancara.
        Es falso, y es de los errores caros: quien se ostenta sabedor del acto
        y promueve sin esperar a que se lo notifiquen está en tiempo —el plazo
        marca hasta cuándo, no a partir de cuándo puede acudirse—. Un proyecto
        que sobresee por extemporáneo un amparo promovido con anticipación no
        se cae en sesión: se cae en revisión, y con costas de credibilidad.

        Salió comparando un adelanto real: la demanda se presentó el 23 de
        abril y la tabla la declaró fuera de plazo contra un vencimiento de
        julio. Lo tarde y lo temprano no son la misma cosa.

        Lo anticipado se marca aparte —`anticipada`— porque la prosa sí debe
        decirlo: no es lo mismo llegar el último día que llegar antes de que
        el reloj empiece, y quien firma querrá saber por qué no hay cómputo
        que cuadre.

        SIN PLAZO NO HAY EXTEMPORANEIDAD. `redactor_adelanto` intentaba decirlo
        con `c.oportuna = True` sobre esta propiedad de sólo lectura, y eso
        reventaba con AttributeError toda queja por omisión de tramitar la
        demanda y todo amparo contra actos que amenazan la vida o la libertad.
        Se responde aquí, que es donde se sabe.
        """
        if self.en_cualquier_tiempo:
            return True
        if self.presentacion is None:
            return None
        return self.presentacion <= self.vencimiento

    @property
    def dia_de_presentacion(self) -> Optional[int]:
        if self.presentacion is None or self.presentacion not in self.dias:
            return None
        return self.dias.index(self.presentacion) + 1

    # ═══════════════════════════════════════════════════════════════════════
    # LAS TRES PREGUNTAS QUE EL DOCUMENTO LE HACE AL CÓMPUTO
    # ═══════════════════════════════════════════════════════════════════════
    # Viven aquí y no en el compositor por una razón medida: allí la condición
    # `oportuna is False and not en_cualquier_tiempo` estaba escrita a mano y
    # gobernaba TRES decisiones estructurales del .docx —el apartado de fondo,
    # el de efectos y el punto resolutivo—. Con la decisión del secretario de
    # por medio pasarían a ser tres copias de una condición de cuatro
    # términos, y basta que UNA se quede sin actualizar para que salga un
    # proyecto con estudio de fondo y resolutivo de sobreseimiento: la misma
    # incongruencia que el módulo existe para impedir, entrando al revés.

    @property
    def cierra_por_extemporaneidad(self) -> bool:
        """¿La EJECUTORIA se resuelve por improcedencia, sin entrar al fondo?"""
        if self.en_cualquier_tiempo:
            return False
        if self.oportuna is not False:
            return False
        return self.decision != "oportuna"

    @property
    def rectificada(self) -> bool:
        """El cómputo da extemporánea y el secretario declaró que no lo es."""
        return (self.decision == "oportuna"
                and self.oportuna is False
                and not self.en_cualquier_tiempo)

    @property
    def fondo_en_reserva(self) -> bool:
        """Se sobresee, y el estudio va detrás como anexo de trabajo."""
        return self.cierra_por_extemporaneidad and self.decision == "reserva"

    @property
    def escribe_fondo(self) -> bool:
        """¿Se escribe el estudio, dentro o fuera de la ejecutoria?"""
        return (not self.cierra_por_extemporaneidad) or self.fondo_en_reserva


# ═══════════════════════════════════════════════════════════════════════════
# ANTE QUIÉN SE PRESENTA EL ESCRITO — de eso, y sólo de eso, depende que el
# calendario de la autoridad responsable entre al cómputo
# ═══════════════════════════════════════════════════════════════════════════
#
# La regla NO es «la responsable es parte». Es que ELLA RECIBE EL ESCRITO.
# Lo dice con todas las letras la jurisprudencia 2a./J. 36/2018 (10a.),
# registro digital 2016696: «por disposición del artículo 176 de la Ley de
# Amparo, es ante la autoridad responsable del acto reclamado y no ante el
# Tribunal Colegiado de Circuito, que inicia el trámite del juicio de amparo
# directo […] y por ello para el cómputo del plazo relativo deben excluirse
# los días inhábiles de la responsable».
#
# Por eso el descuento NO puede aplicarse a los cuatro tipos: en el amparo en
# revisión el recurso se interpone «por conducto del órgano jurisdiccional que
# haya dictado la resolución recurrida» (artículo 86) y en la queja «ante el
# órgano jurisdiccional que conozca del juicio de amparo» (artículo 99). Ahí
# los días de la responsable son jurídicamente irrelevantes, y descontarlos
# alargaría un plazo que la ley no alarga.

@dataclass(frozen=True)
class Receptor:
    """Quién recibe el escrito, y con qué fundamento."""
    clave: str
    quien: str            # va literal al considerando
    fundamento: str       # el artículo que lo dice
    aporta_calendario: bool
    # La cita que justifica el descuento. Vacía cuando no hay descuento.
    apoyo: str = ""


# Rubro literal, verificado el 12-sep-2026 contra la ficha del Semanario
# (https://sjf2.scjn.gob.mx/services/sjftesismicroservice/api/public/tesis/2024494:
# `ius` 2024494, `claveTesis` «P./J. 4/2022 (11a.)», instancia Pleno,
# tipoTesis «Tesis Jurisprudenciales»). Se guarda una sola vez para que no haya
# dos versiones del rubro circulando por el repositorio.
TESIS_DESCUENTO_RESPONSABLE = (
    "la jurisprudencia P./J. 4/2022 (11a.) del Pleno de la Suprema Corte de "
    "Justicia de la Nación, de registro digital 2024494, de rubro: «DEMANDA DE "
    "AMPARO DIRECTO. PARA EL CÓMPUTO DEL PLAZO DE SU PRESENTACIÓN DEBEN "
    "EXCLUIRSE LOS DÍAS INHÁBILES QUE ESTABLECE EL ARTÍCULO 19 DE LA LEY DE "
    "AMPARO Y AQUELLOS EN QUE LA AUTORIDAD RESPONSABLE SUSPENDA ACTIVIDADES, "
    "AUN CUANDO ESTÉN CONTEMPLADOS COMO HÁBILES POR LA REFERIDA LEGISLACIÓN, "
    "SIN QUE ELLO IMPLIQUE DESCONTAR LOS DÍAS EN QUE EL TRIBUNAL COLEGIADO DE "
    "CIRCUITO SUSPENDA LABORES POR SITUACIONES EXTRAORDINARIAS»")

# Revisión fiscal: aquí el fundamento NO es la Ley de Amparo. El artículo 74,
# fracción II, de la LFPCA define los hábiles como «aquellos en que se
# encuentren abiertas al público las oficinas de las Salas del Tribunal». El
# criterio que lo dice expresamente —registro digital 2007213— es AISLADA, y
# así se cita: el peso lo lleva la ley.
TESIS_DESCUENTO_TFJA = (
    "el artículo 74, fracción II, de la Ley Federal de Procedimiento "
    "Contencioso Administrativo, conforme al cual son hábiles los días en que "
    "se encuentren abiertas al público las oficinas de las Salas del Tribunal; "
    "criterio que recoge la tesis aislada XI.2o.A.T.3 A (10a.), de registro "
    "digital 2007213")

RECEPTOR_DEL_ESCRITO: dict[str, Receptor] = {
    "amparo_directo": Receptor(
        clave="amparo_directo",
        quien="la autoridad responsable",
        fundamento="artículo 176 de la Ley de Amparo",
        aporta_calendario=True,
        apoyo=TESIS_DESCUENTO_RESPONSABLE),
    "revision_fiscal": Receptor(
        clave="revision_fiscal",
        quien="la Sala responsable del Tribunal Federal de Justicia Administrativa",
        fundamento=("artículo 63 de la Ley Federal de Procedimiento "
                    "Contencioso Administrativo, en relación con el 74, "
                    "fracción II, del mismo ordenamiento"),
        aporta_calendario=True,
        apoyo=TESIS_DESCUENTO_TFJA),
    "amparo_revision": Receptor(
        clave="amparo_revision",
        quien="el órgano jurisdiccional que dictó la resolución recurrida",
        fundamento="artículo 86 de la Ley de Amparo",
        aporta_calendario=False),
    "queja": Receptor(
        clave="queja",
        quien="el órgano jurisdiccional que conoce del juicio de amparo",
        fundamento="artículo 99 de la Ley de Amparo",
        aporta_calendario=False),
}


def receptor_de(tipo: str) -> Receptor:
    """Quién recibe el escrito de ese tipo de asunto.

    Un tipo que no esté en el catálogo NO hereda el descuento: cae en el
    receptor federal, que es el supuesto conservador. Alargar un plazo por un
    tipo desconocido sería inventar oportunidad.
    """
    try:
        import tipos_asunto as _ta
        t = _ta.normalizar(tipo) or ""
    except Exception:
        t = (tipo or "").strip().lower()
    return RECEPTOR_DEL_ESCRITO.get(t, RECEPTOR_DEL_ESCRITO["amparo_revision"])


# ═══════════════════════════════════════════════════════════════════════════
# Los días en que la responsable no laboró — los declara quien los sabe
# ═══════════════════════════════════════════════════════════════════════════
#
# NO HAY CATÁLOGO NACIONAL Y NO SE PUEDE FABRICAR. Son miles de órganos —salas
# civiles, juntas de conciliación, tribunales administrativos de treinta y dos
# estados, salas del TFJA— y cada uno publica su acuerdo donde quiere. El
# `CALENDARIOS_RESPONSABLE` de este módulo tenía UNA entrada, de un estado, y
# se buscaba por clave mientras el pipeline le pasaba el nombre libre de la
# autoridad: sobre las 103 sesiones reales del taller, CERO podían activarla.
# Una capa escrita, documentada y apagada.
#
# Así que el dato se PIDE. Y se pide en la forma más barata que lo cubre todo:
# tramos. Un periodo vacacional es un tramo; un día de suspensión es un tramo
# de un día. Una sola línea:
#
#     2026-06-29..2026-07-10, 2026-01-02, 2026-01-05
#
# El tramo se guarda COMO TRAMO, no como diez días sueltos, porque el
# considerando lo escribe como tramo: «del veintinueve de junio al diez de
# julio», que es como está en los engroses, y no como una lista de diez fechas.

@dataclass
class InhabilesResponsable:
    """Lo que el secretario declaró sobre el calendario de la responsable."""
    dias: set = field(default_factory=set)
    tramos: list = field(default_factory=list)   # [(ini, fin)]; un día es (d, d)
    errores: list = field(default_factory=list)
    declarado: bool = False


def _fecha_iso(x) -> Optional[Fecha]:
    if isinstance(x, _dt.date):
        return x
    try:
        return _dt.date.fromisoformat(str(x).strip())
    except Exception:
        return None


def leer_inhabiles_responsable(texto) -> InhabilesResponsable:
    """Lee «2026-06-29..2026-07-10, 2026-01-02» y devuelve días y tramos.

    Acepta también una lista de cadenas o de fechas, para que la sesión de
    Supabase pueda rehidratarse sin volver a formatear nada.

    LOS ERRORES NO SE TRAGAN. Un rango invertido no produce ningún día y el
    plazo sale corto sin avisar: es exactamente lo que pasó en este módulo con
    `(_d(2024,7,16), _d(2024,7,3))`, que se comió dos quincenas de vacaciones
    del Poder Judicial durante meses. Aquí un rango al revés es un aviso que el
    secretario lee, no un silencio.
    """
    r = InhabilesResponsable()
    if not texto:
        return r
    if isinstance(texto, (list, tuple, set)):
        piezas = [str(x).strip() if not isinstance(x, _dt.date) else x.isoformat()
                  for x in texto]
    else:
        piezas = [p.strip() for p in str(texto).replace(";", ",").split(",")]
    for p in piezas:
        if not p:
            continue
        crudo = p
        # «a», «al» y el guion largo se escriben solos al copiar de un acuerdo.
        for sep in ("..", " al ", " a ", "—", "–"):
            if sep in p:
                p = p.replace(sep, "..", 1)
                break
        ini_s, _, fin_s = p.partition("..")
        ini, fin = _fecha_iso(ini_s), (_fecha_iso(fin_s) if fin_s.strip() else _fecha_iso(ini_s))
        if ini is None or fin is None:
            r.errores.append(
                f"No entendí «{crudo}» como fecha o tramo de la responsable. Se "
                f"escribe 2026-06-29 para un día y 2026-06-29..2026-07-10 para un "
                f"periodo. ESE DÍA NO SE DESCONTÓ.")
            continue
        if fin < ini:
            r.errores.append(
                f"El tramo «{crudo}» está AL REVÉS: termina ({fin.isoformat()}) "
                f"antes de empezar ({ini.isoformat()}). Un rango invertido no "
                f"produce ni un día inhábil y el plazo saldría corto. NO SE "
                f"DESCONTÓ NADA de ese tramo.")
            continue
        if (fin - ini).days > 120:
            r.errores.append(
                f"El tramo «{crudo}» abarca {(fin - ini).days + 1} días. Ningún "
                f"periodo vacacional dura tanto: comprueba el año. NO SE "
                f"DESCONTÓ.")
            continue
        if not (2013 <= ini.year <= 2100 and 2013 <= fin.year <= 2100):
            r.errores.append(
                f"El tramo «{crudo}» cae fuera de la Ley de Amparo vigente "
                f"(2 de abril de 2013). NO SE DESCONTÓ.")
            continue
        r.tramos.append((ini, fin))
        cur = ini
        while cur <= fin:
            r.dias.add(cur)
            cur += _dt.timedelta(days=1)
    r.declarado = bool(r.dias) or bool(r.errores)
    return r


# ═══════════════════════════════════════════════════════════════════════════
# POR QUÉ SE CUENTA ASÍ — los fundamentos del cómputo
# ═══════════════════════════════════════════════════════════════════════════
# David, 25-sep-2026: «falta el fundamento de por qué se computa en esos
# términos (porque surte efectos a partir del día siguiente cuando es
# notificación personal, porque al tercer día si es boletín)». Cotejado con el
# texto oficial de la Ley de Amparo (DOF 16-10-2025):
#   · art. 18 — los plazos del 17 «se computarán a partir del día siguiente a
#     aquél en que surta efectos, conforme a la ley del acto, la notificación»;
#   · art. 22 — los demás plazos «comenzarán a correr a partir del día
#     siguiente al en que surta sus efectos la notificación»;
#   · art. 31, fr. II — las notificaciones a los particulares surten «desde el
#     día siguiente al de la notificación personal o al de la fijación y
#     publicación de la lista». (La fr. I es la de las autoridades.)
_INICIO_POR_TIPO = {
    "amparo_directo": "artículo 18 de la Ley de Amparo",
    "amparo_revision": "artículo 22 de la Ley de Amparo",
    "queja": "artículo 22 de la Ley de Amparo",
    # La revisión fiscal no: su artículo 63 ya dice «dentro de los quince días
    # siguientes a aquél en que surta sus efectos la notificación».
}


def fundamento_de_inicio(tipo: str) -> str:
    """El precepto que hace correr el plazo desde el día siguiente."""
    import tipos_asunto as _ta_i
    return _INICIO_POR_TIPO.get(_ta_i.normalizar(tipo) or "amparo_directo", "")


def fundamento_de_surtimiento(regla, tipo: str, papel: str = "") -> str:
    """El precepto que dice cuándo surtió efectos la notificación. Vacío si
    no se sabe: en el amparo directo lo fija la ley del acto, y la de cada
    entidad no está en el catálogo — no se inventa.

    EN LOS RECURSOS, LA FRACCIÓN DEL 31 QUE CORRESPONDE (3-oct-2026): I para el
    oficio a una autoridad, II para la personal y la lista a los particulares,
    III para la electrónica. Antes era siempre la II. Y SI RECURRE UNA
    AUTORIDAD con la regla de los particulares, NO se cita la II —sería fundar
    en la regla ajena un día de surtimiento que no le corresponde—: vacío, y
    `aviso_fundamento` lo dice."""
    f = str(getattr(regla, "fundamento", "") or "").strip()
    if f:
        return f
    import tipos_asunto as _ta_s
    if _ta_s.normalizar(tipo) in ("amparo_revision", "queja"):
        clave = getattr(regla, "clave", "")
        if clave == "oficio":
            return "artículo 31, fracción I, de la Ley de Amparo"
        if clave == "electronica":
            return "artículo 31, fracción III, de la Ley de Amparo"
        if clave in ("personal", "lista"):
            # En los recursos lo notificado es una resolución del juicio de
            # amparo: la regla es la de la propia Ley de Amparo.
            if (papel or "").strip().lower() == "autoridad":
                return ""
            return "artículo 31, fracción II, de la Ley de Amparo"
    return ""


class TramosInhabiles(list):
    """Los tramos de siempre —una lista de (inicio, fin), en orden— que además
    saben de qué fundamento es cada uno (`claves`, paralela a la lista).

    ES UNA LISTA A PROPÓSITO (3-oct-2026): quien la recorre como `for a, b in
    tramos` —la leyenda de la tabla, el considerando del adhesivo que compone
    `resultandos_por_tipo`— sigue funcionando igual; `tramos_en_letra` la
    reconoce y escribe cada tramo con su fundamento."""
    claves: list


def _es_calendario_pjf(cal) -> bool:
    """El calendario del amparo (art. 19 LA), no el del TFJA: sólo en éste los
    inhábiles tienen fundamentos distintos que separar."""
    return cal is not None and str(getattr(cal, "fundamento", "") or "") == CALENDARIO_AMPARO.fundamento


def tramos_inhabiles(fechas, cal) -> list:
    """Los inhábiles como tramos: dos inhábiles se unen si entre ellos sólo
    hay días que tampoco corren. Del 16 de diciembre al 1 de enero son trece
    fechas sueltas y un solo tramo.

    CON EL CALENDARIO DEL AMPARO, UN TRAMO POR FUNDAMENTO (3-oct-2026): dos
    inhábiles se unen sólo si son del MISMO fundamento y entre ellos no hay más
    que fines de semana o días de ese mismo fundamento. Antes «del uno al cinco
    de mayo de 2025» metía en el artículo 19 el dos de mayo, que lo declaró la
    Circular 1/2025. Devuelve `TramosInhabiles` (una lista, con sus claves)."""
    fs = sorted(set(f for f in (fechas or []) if f))
    if not _es_calendario_pjf(cal):
        tramos = []
        for f in fs:
            if tramos:
                ini, fin = tramos[-1]
                cur, puente = fin + _dt.timedelta(days=1), True
                while cur < f:
                    if cal is not None and cal.es_habil(cur):
                        puente = False
                        break
                    cur += _dt.timedelta(days=1)
                if puente:
                    tramos[-1] = (ini, f)
                    continue
            tramos.append((f, f))
        return tramos
    por_clave: dict = {}
    for f in fs:
        k = clave_del_inhabil(f)
        lst = por_clave.setdefault(k, [])
        if lst:
            ini, fin = lst[-1]
            cur, puente = fin + _dt.timedelta(days=1), True
            while cur < f:
                if cur.weekday() < 5 and (cal.es_habil(cur) or clave_del_inhabil(cur) != k):
                    puente = False
                    break
                cur += _dt.timedelta(days=1)
            if puente:
                lst[-1] = (ini, f)
                continue
        lst.append((f, f))
    pares = sorted(((t, k) for k, ts in por_clave.items() for t in ts), key=lambda x: x[0][0])
    out = TramosInhabiles(t for t, _ in pares)
    out.claves = [k for _, k in pares]
    return out


def grupos_inhabiles(tramos, art19_al_final: bool = False) -> list:
    """[(clave, [tramos])], cada grupo con su fundamento. Sin claves, un solo
    grupo 'art19'.

    EN ORDEN CRONOLÓGICO (D6, 3-oct-2026): los grupos van por la fecha de su
    primer tramo. El considerando decía «ni el uno de enero de dos mil
    veintiséis…, ni del dieciséis al treinta y uno de diciembre de dos mil
    veinticinco», el año nuevo antes que diciembre, porque el grupo del artículo
    19 iba siempre primero (rev_4). `art19_al_final` conserva el orden de antes
    para quien escribe «ni {esto}, por ser inhábiles en términos del 19» detrás
    (`tramos_en_letra`)."""
    claves = list(getattr(tramos, "claves", None) or ["art19"] * len(tramos or []))
    grupos: dict = {}
    for t, k in sorted(zip(list(tramos or []), claves), key=lambda x: x[0][0]):
        grupos.setdefault(k, []).append(t)
    orden = list(grupos)
    if art19_al_final:
        orden = [k for k in grupos if k != "art19"] + (["art19"] if "art19" in grupos else [])
    return [(k, grupos[k]) for k in orden]


def _conforme_a_fuente(clave: str) -> str:
    """«conforme al artículo 226 de la…», «conforme a la Circular 1/2025…»."""
    t = FUENTES_INHABIL.get(clave, FUENTES_INHABIL["art19"])["texto"]
    if t.startswith("artículos"):
        return "conforme a los " + t
    if t.startswith("artículo"):
        return "conforme al " + t
    return "conforme a la " + t


def _lista_compacta(tramos) -> str:
    """Como `tramos_en_letra`, pero los días sueltos del mismo mes van juntos:
    «el uno y el cinco de mayo de dos mil veinticinco», no el año dos veces."""
    bloques: list = []      # cada bloque: ("dias", [fechas]) o ("tramo", (ini, fin))
    for ini, fin in tramos or []:
        if ini == fin and bloques and bloques[-1][0] == "dias" and \
                (bloques[-1][1][-1].month, bloques[-1][1][-1].year) == (ini.month, ini.year):
            bloques[-1][1].append(ini)
        elif ini == fin:
            bloques.append(("dias", [ini]))
        else:
            bloques.append(("tramo", (ini, fin)))
    partes = []
    for tipo_b, v in bloques:
        if tipo_b == "tramo":
            partes.append(tramos_en_letra([v]))
        elif len(v) == 1:
            partes.append(f"el {fecha_en_letra(v[0])}")
        else:
            cola = f"{_UNIDADES[v[-1].day]} de {_MESES[v[-1].month]} de {_anio_en_letra(v[-1].year)}"
            partes.append("el " + ", el ".join(_UNIDADES[x.day] for x in v[:-1]) + f" y el {cola}")
    if not partes:
        return ""
    return partes[0] if len(partes) == 1 else ", ".join(partes[:-1]) + f" y {partes[-1]}"


def _razon_del_grupo(clave: str, tramos) -> str:
    """Lo que se dice de un grupo que no es del artículo 19."""
    if clave == "vac":
        pl = len(tramos) > 1
        return (f"por corresponder {'a los periodos vacacionales' if pl else 'al periodo vacacional'} "
                f"del Poder Judicial de la Federación, {_conforme_a_fuente('vac')}")
    n = sum((b - a).days + 1 for a, b in tramos)
    return f"por ser {'inhábiles' if n > 1 else 'inhábil'} {_conforme_a_fuente(clave)}"


def clausula_inhabiles(c, cal=None, con_previos=None) -> str:
    """«sin contar sábados y domingos, ni …, por ser inhábiles en términos del
    artículo 19 de la Ley de Amparo, ni el dos de mayo…, por ser inhábil conforme
    a la Circular 1/2025…» — la cláusula entera, cada tramo con su fundamento.

    ES LA PUERTA PARA QUIEN ESCRIBA UN CÓMPUTO (el considerando de oportunidad,
    el del adhesivo): recibe el cómputo y devuelve desde «sin contar» hasta el
    último fundamento, sin punto ni coma final.

    LOS INHÁBILES PREVIOS AL PLAZO TAMBIÉN (3-oct-2026): los que caen entre la
    notificación y el inicio (`inhabiles_previos`) movieron el surtimiento o el
    arranque, y sin nombrarlos el párrafo salta fechas sin razón —Q 24/2026:
    «surtió efectos… el dos de enero» de una notificación del dieciocho de
    diciembre; el engrose los funda—. Y LOS GRUPOS EN ORDEN CRONOLÓGICO (D6): si
    el primero no es el del artículo 19, los sábados y domingos llevan su
    fundamento solos y el grupo del 19 lo vuelve a decir en su sitio.

    `con_previos`: si se nombran los previos. Sin decirlo (None), con la bandera
    `procedencia_por_tipo` (el considerando del adhesivo sólo existe en el
    camino nuevo); `parrafo_oportunidad` lo pide expreso, porque el camino nuevo
    es bandera Y ficha y eso lo sabe quien compone, no este módulo."""
    cal = cal if cal is not None else getattr(c, "cal_amparo", None)
    # Los previos, CON LA BANDERA (no son [siempre]: el camino viejo sigue
    # nombrando sólo los del plazo). El orden cronológico (D6) sí es para todos.
    if con_previos is None:
        con_previos = _rige_procedencia()
    _prev = list(getattr(c, "inhabiles_previos", []) or []) if con_previos else []
    _todos = sorted(set(_prev + list(getattr(c, "inhabiles_en_medio", []) or [])))
    tr = tramos_inhabiles(_todos, cal)
    fund = str(getattr(cal, "fundamento", "") or CALENDARIO_AMPARO.fundamento)
    if not tr:
        return f"sin contar sábados y domingos por ser inhábiles en términos del {fund}"
    if not isinstance(tr, TramosInhabiles):
        return (f"sin contar sábados y domingos, ni {tramos_en_letra(tr)}, por ser "
                f"inhábiles en términos del {fund}")
    grupos = grupos_inhabiles(tr)
    # EL GRUPO DEL ARTÍCULO 19 VA CON LOS SÁBADOS Y DOMINGOS, UNA SOLA VEZ
    # (3-oct-2026, integración): el orden cronológico estricto partía la cita
    # —«…sábados y domingos por ser inhábiles en términos del artículo 19…, ni
    # el diecisiete de noviembre (LFT)…, ni el veinte de noviembre, por ser
    # inhábil en términos del artículo 19…»— y el mismo fundamento salía dos
    # veces en una frase. Los demás grupos siguen en orden cronológico.
    grupos = ([g for g in grupos if g[0] == "art19"] + [g for g in grupos if g[0] != "art19"])
    s = "sin contar sábados y domingos"
    if grupos and grupos[0][0] == "art19":
        s += f", ni {_lista_compacta(grupos[0][1])}, por ser inhábiles en términos del {fund}"
        resto = grupos[1:]
    else:
        s += f" por ser inhábiles en términos del {fund}"
        resto = grupos
    for k, ts in resto:
        if k == "art19":
            _n = sum((b - a).days + 1 for a, b in ts)
            s += (f", ni {_lista_compacta(ts)}, por ser {'inhábiles' if _n > 1 else 'inhábil'} "
                  f"en términos del {fund}")
        else:
            s += f", ni {_lista_compacta(ts)}, {_razon_del_grupo(k, ts)}"
    return s


def a_quien_se_notifico(tipo: str, papel: str = "") -> str:
    """«a la parte quejosa», «a la parte recurrente», «a la autoridad recurrente».

    SIN GÉNERO (3-oct-2026). El párrafo decía «se notificó al quejoso» de Ana
    López Ruiz: el vocabulario del tipo trae el sustantivo en masculino y el
    género de una persona física no se sabe —ni se infiere de su nombre de
    pila—. «La parte» concuerda siempre. La revisión fiscal la interpone
    siempre una autoridad (art. 63 LFPCA)."""
    import tipos_asunto as _ta_q
    t = _ta_q.normalizar(tipo) or "amparo_directo"
    if t == "revision_fiscal":
        return "a la autoridad recurrente"
    if t == "amparo_directo":
        return "a la parte quejosa"
    if (papel or "").strip().lower() == "autoridad":
        return "a la autoridad recurrente"
    return "a la parte recurrente"


def conforme_al_precepto(texto: str) -> str:
    """«el artículo 126 del Código…» → «conforme al artículo 126 del Código…».

    Lo que el secretario escribe en «fundamento_surtimiento» viene en cualquier
    forma: con artículo o sin él, con «conforme a» delante o con punto final.
    Se deja en la que pide la frase del considerando; vacío si no trae nada."""
    t = " ".join(str(texto or "").split()).strip(" .;,")
    if not t or t.strip("*") == "":
        return ""
    t = re.sub(r"^(?:conforme|de\s+conformidad\s+con|en\s+t[ée]rminos\s+de)\s+", "", t, flags=re.I)
    t = re.sub(r"^(?:al|a\s+los|a\s+la|a\s+las|a|del|de\s+los|de\s+la|de\s+las)\s+",
               lambda m: {"al": "el ", "a los": "los ", "a la": "la ", "a las": "las ",
                          "del": "el ", "de los": "los ", "de la": "la ", "de las": "las "}.get(
                   " ".join(m.group(0).lower().split()), ""), t, flags=re.I)
    low = t.lower()
    if low.startswith(("artículos ", "articulos ")):
        return "conforme a los " + t
    if low.startswith(("artículo ", "articulo ", "art. ")):
        return "conforme al " + t
    if low.startswith("el "):
        return "conforme al " + t[3:]
    return "conforme a " + t


# ═══════════════════════════════════════════════════════════════════════════
# EL SURTIMIENTO QUE SÍ ES NACIONAL (3-oct-2026, bandera)
# ═══════════════════════════════════════════════════════════════════════════
# En el amparo directo el surtimiento de la notificación del acto lo rige la
# ley del acto, y la de cada entidad no está en el catálogo: hueco. Pero hay
# materias y órganos cuya ley es FEDERAL y lo dice para toda la república. Se
# usan como omisión SÓLO los que el texto local de la ley confirma (leídos el
# 3-oct-2026 en leyes/LEYES_FEDERALES):
#   · MERCANTIL — Código de Comercio, art. 1075, segundo párrafo: «Las
#     notificaciones personales surten efectos al día siguiente del que se
#     hayan practicado, y las demás surten al día siguiente, de aquel en que
#     se hubieren hecho por boletín…». Ley federal, en tribunales locales y
#     federales. OJO: en el juicio oral mercantil la sentencia dictada en
#     audiencia se tiene por notificada en el acto (art. 1390 Bis 22).
#   · TFJA — LFPCA, art. 70: «Las notificaciones surtirán sus efectos, el día
#     hábil siguiente a aquél en que fueren hechas» (y el 65, que es el de la
#     regla `lfpca`).
#   · AGRARIO (3-oct-2026; David: «Si va con el Código Nacional, y no el
#     Federal, entonces hay que adecuar al Código Nacional»). Antes no entraba:
#     el artículo 167 de la Ley Agraria (reformado DOF 14-11-2025) remite al
#     Código Nacional de Procedimientos Civiles y Familiares, y el 321 del CFPC
#     ya no es el supletorio que nombra. Pero el transitorio Segundo de esa
#     reforma ata su aplicación a la declaratoria del Código Nacional (la del
#     Congreso de cada entidad; en el orden federal, la del Congreso de la
#     Unión; automática el 1-abr-2027) y el Tercero deja los juicios en trámite
#     con la legislación con que empezaron. Así que se decide por la SEDE, con
#     la misma regla que el supletorio de la Ley de Amparo (`sede_cdmx` →
#     `tipos_asunto.es_cdmx`):
#       - EN LA CIUDAD DE MÉXICO, las reglas `cnpcf_personal` (art. 227, fr. I:
#         surte el mismo día), `cnpcf_lista` (art. 211) y `cnpcf_electronica`
#         (art. 227, fr. III), que ya traen su precepto: aquí no se añade nada.
#         El guardián de materia de `redactor_adelanto` cambia a ellas la
#         «personal» y la «lista» que llegan por omisión, con aviso.
#       - FUERA DE ELLA (o sin saber la sede), la «personal» y la «lista» de
#         siempre (surten al día siguiente) con el artículo 321 del CFPC
#         («Toda notificación surtirá sus efectos el día siguiente al en que se
#         practique»), y un aviso que explica el transitorio. Lo mismo con
#         `cfpc_personal`, la personal del Código Federal que el desplegable
#         propone fuera de la Ciudad de México (integración, 3-oct-2026); en
#         ella sólo llega elegida a propósito y no lleva aviso.
#     El 1-abr-2027 el Código Nacional rige en toda la república (salvo los
#     juicios iniciados antes): esta rama habrá que revisarla entonces.
# NO ENTRA el laboral: el 747 de la LFT hace surtir la personal el mismo día, y
# la regla «personal» cuenta un día: no casan.
#
# LO MERCANTIL NO SIEMPRE SE LLAMA MERCANTIL (3-oct-2026, rev_1). Los colegiados
# registran lo mercantil como «AMPARO DIRECTO CIVIL», así que la materia de la
# ficha dice «civil» y el respaldo no entraba: en el AD 456/2025 la responsable
# era el «Juzgado Primero de Primera Instancia Especializado en Oralidad
# Mercantil» y el considerando salió con el surtimiento en hueco, donde el
# engrose cita el 1075. Se reconoce también por el ÓRGANO (si es mercantil y no
# «Civil y Mercantil», que conoce de las dos) y por el EXPEDIENTE o el contexto
# («juicio ejecutivo mercantil», «vía oral mercantil»).
#
# Y EL TFJA SÓLO CON LA NOTIFICACIÓN PERSONAL (rev_2). Con «lista» se escribía
# «surtió efectos al día hábil siguiente, conforme a los artículos 65 y 70 de la
# LFPCA», y el 65 dice que lo que no es personal va por Boletín y surte al
# TERCER día hábil: el precepto contradecía el día. Ante el TFJA «por lista» no
# es una regla: el aviso de `computar` pide la del Boletín (`lfpca_boletin`).
_RX_ORGANO_MERCANTIL = re.compile(r"\bmercantil(?:es)?\b", re.I)
_RX_ORGANO_MIXTO = re.compile(
    r"\bcivil(?:es)?\b[^.]{0,40}?\bmercantil|\bmercantil(?:es)?\b[^.]{0,40}?\bcivil", re.I)
_RX_JUICIO_MERCANTIL = re.compile(
    r"\b(?:juicio|v[íi]a|procedimiento|controversia)\s+(?:(?:oral|ejecutiv[oa]|ordinari[oa]|"
    r"especial|sumari[oa])\s+)?mercantil\b|\bejecutivo\s+mercantil\b|\boralidad\s+mercantil\b", re.I)


def es_mercantil(materia: str = "", responsable: str = "", expediente: str = "",
                 contexto: str = "") -> str:
    """De dónde se sabe que el juicio de origen es mercantil: «materia»,
    «órgano», «expediente», «contexto», o «» si no se sabe."""
    if (materia or "").strip().lower() == "mercantil":
        return "materia"
    r = " ".join(str(responsable or "").split())
    if r and _RX_ORGANO_MERCANTIL.search(r) and not _RX_ORGANO_MIXTO.search(r):
        return "órgano"
    if _RX_JUICIO_MERCANTIL.search(str(expediente or "")) or (
            _RX_ORGANO_MERCANTIL.search(str(expediente or ""))
            and not _RX_ORGANO_MIXTO.search(str(expediente or ""))):
        return "expediente"
    if _RX_JUICIO_MERCANTIL.search(str(contexto or "")):
        return "contexto"
    return ""


PRECEPTO_321_AGRARIO = ("el artículo 321 del Código Federal de Procedimientos Civiles, "
                        "de aplicación supletoria en materia agraria")


def aviso_321_agrario(cdmx=None, regla: str = "") -> str:
    """El aviso del surtimiento agrario fundado en el 321 del CFPC: por qué ése y
    no el Código Nacional, y qué hacer si el Código Nacional ya rige para ese
    juicio. `cdmx` False (fuera de la Ciudad de México) o None (sede
    desconocida) cambia sólo el porqué.

    LA SALIDA ES LA GEMELA DE LA FORMA (revisión AD, 3-oct-2026). Con la «lista»
    el aviso mandaba a «Personal — Código Nacional» (227-I, surte el mismo día):
    quien lo siguiera arrancaba el plazo un día hábil antes y el considerando
    decía «de manera personal» de una notificación por lista; una demanda del
    último día salía extemporánea. La gemela de la lista es la del 211, que
    surte al día siguiente: las fechas no cambian, cambia el precepto."""
    _sede = ("el tribunal no reside en la Ciudad de México" if cdmx is False else
             "no se pudo saber si el tribunal reside en la Ciudad de México (no hay ciudad ni "
             "circuito legibles)")
    _salida = ("elige «Por lista — Código Nacional» (art. 211: surte al día siguiente; las fechas "
               "no cambian, cambia el precepto) y vuelve a generar."
               if str(regla or "").strip().lower() in ("lista", "cnpcf_lista") else
               "elige «Personal — Código Nacional» (art. 227, fr. I: surte el mismo día y el plazo "
               "arranca un día hábil antes) y vuelve a generar.")
    return ("EL SURTIMIENTO SE FUNDÓ EN EL ARTÍCULO 321 DEL CÓDIGO FEDERAL DE PROCEDIMIENTOS "
            "CIVILES (toda notificación surte al día siguiente), porque el juicio de origen es "
            f"agrario y {_sede}. COMPRUÉBALO: el artículo 167 de la Ley Agraria (reformado DOF "
            "14-11-2025) ya remite al Código Nacional de Procedimientos Civiles y Familiares, "
            "pero su transitorio Segundo ata esa aplicación a la declaratoria del Código Nacional "
            "(la del Congreso de cada entidad; en el orden federal, la del Congreso de la Unión) y "
            "el Tercero deja los juicios en trámite con la legislación con que empezaron; el 1 de "
            "abril de 2027 el Código Nacional rige en toda la república, salvo los juicios "
            "iniciados antes. Si el Código Nacional ya rige para ese juicio, " + _salida)


def aviso_cnpcf_agrario(regla: str = "cnpcf_personal") -> str:
    """El aviso del surtimiento agrario fundado en el CÓDIGO NACIONAL en la
    Ciudad de México: la regla del taller lo propone ahí, pero la ley no lo
    impone a todos los juicios (revisión de normas y front, 3-oct-2026).

    POR QUÉ SIEMPRE, Y NO SÓLO CUANDO EL GUARDIÁN CAMBIA LA REGLA. La pantalla
    pide `/taller/reglas-surtimiento`, recibe `cnpcf_personal` como omisión y
    la pone sola en el formulario; al servidor ya no llega la «personal»
    genérica, así que el guardián de materia (`redactor_adelanto.regla_agraria`)
    no actuaba y su aviso del transitorio nunca salía. Es la simetría de
    `cfpc_personal` fuera de la Ciudad de México: también ahí es la propuesta y
    no se sabe si alguien la eligió. El transitorio Tercero de la reforma al
    167 deja los juicios en trámite con su legislación, casi todos los agrarios
    que hoy llegan en amparo directo empezaron antes, y para ellos el 321 del
    CFPC surte al día siguiente: con el Código Nacional el plazo vence un día
    hábil antes y una demanda oportuna sale extemporánea.

    Y LA SEDE ES REGLA DE LA CASA, NO HECHO DE LA LEY: para el orden federal el
    transitorio Segundo ata el 167 reformado a la declaratoria del Congreso de
    la Unión, no a la de la Ciudad de México."""
    _comprueba = (
        "COMPRUÉBALO con la fecha en que se inició el juicio agrario y con la declaratoria del "
        "orden federal: el artículo 167 de la Ley Agraria (reformado DOF 14-11-2025) remite al "
        "Código Nacional, pero su transitorio Segundo ata esa aplicación, en el orden federal, a "
        "la declaratoria del Congreso de la Unión, y el Tercero deja los juicios en trámite con "
        "la legislación con que empezaron.")
    if str(regla or "").strip().lower() == "cnpcf_lista":
        return ("LA NOTIFICACIÓN POR LISTA SE FUNDÓ EN EL ARTÍCULO 211 DEL CÓDIGO NACIONAL DE "
                "PROCEDIMIENTOS CIVILES Y FAMILIARES (surte al día siguiente de su publicación), "
                "porque el juicio de origen es agrario y el tribunal reside en la Ciudad de México, "
                "donde la regla del taller aplica ese código. " + _comprueba + " Si el juicio se "
                "inició antes de que el Código Nacional rigiera para él, el precepto es el artículo "
                "321 del Código Federal de Procedimientos Civiles, que también surte al día "
                "siguiente: las fechas no cambian; corrige el precepto en el considerando.")
    return ("EL SURTIMIENTO SE CONTÓ CON EL CÓDIGO NACIONAL DE PROCEDIMIENTOS CIVILES Y "
            "FAMILIARES (artículo 227, fracción I: los términos corren desde el día siguiente al "
            "de la notificación personal, así que surte el mismo día), porque el juicio de origen "
            "es agrario y el tribunal reside en la Ciudad de México, donde la regla del taller "
            "aplica ese código. " + _comprueba + " Si el juicio se inició antes de que el Código "
            "Nacional rigiera para él, el surtimiento es el del artículo 321 del Código Federal de "
            "Procedimientos Civiles (al día siguiente) y el plazo vence un día hábil después: "
            "elige «Personal — Código Federal (art. 321)» y vuelve a generar.")


def contraste_cnpcf_agrario(c_nacional, c_federal) -> str:
    """El aviso en mayúsculas cuando el cómputo con el Código Nacional sale
    EXTEMPORÁNEO y con el 321 del CFPC saldría EN TIEMPO; «» si no (revisión de
    normas y front, 3-oct-2026). El día en que surte decide la oportunidad, y
    el transitorio Tercero puede hacer aplicable la regla que da en tiempo."""
    if (getattr(c_nacional, "oportuna", None) is not False
            or getattr(c_federal, "oportuna", None) is not True):
        return ""
    return ("CON EL CÓDIGO NACIONAL LA DEMANDA SALE EXTEMPORÁNEA Y CON EL ARTÍCULO 321 DEL CÓDIGO "
            "FEDERAL SALDRÍA EN TIEMPO: con la notificación personal del artículo 227, fracción I, "
            f"el plazo venció el {fecha_en_letra(c_nacional.vencimiento)}; con la del 321 (surte al "
            f"día siguiente) vencería el {fecha_en_letra(c_federal.vencimiento)}. Antes de desechar, "
            "comprueba cuándo se inició el juicio agrario: si empezó antes de que el Código Nacional "
            "rigiera para él (transitorio Tercero de la reforma al artículo 167 de la Ley Agraria), "
            "elige «Personal — Código Federal (art. 321)» y vuelve a generar.")


def surtimiento_nacional(tipo: str, materia: str = "", responsable: str = "",
                         regla: str = "", expediente: str = "",
                         contexto: str = "", tribunal: str = "",
                         ciudad: str = "") -> tuple:
    """(precepto, aviso) que rige el surtimiento cuando la ley del acto es
    federal y lo dice; ("", "") si no se puede afirmar. Sólo para la
    notificación personal o por lista que surte al día siguiente y cuya regla
    no trae ya su precepto. `expediente` (el de la ficha en prosa: «juicio
    oral mercantil 625/2024») y `contexto` (el encabezado, los antecedentes)
    sirven para reconocer lo mercantil que se registró como civil.

    `tribunal` y `ciudad` (los del colegiado, por palabra clave; 3-oct-2026)
    deciden la rama AGRARIA: en la Ciudad de México, «» —la regla del Código
    Nacional ya trae su precepto, y la «personal» de siempre no se funda en el
    321—; fuera de ella o sin saber la sede, el 321 del CFPC con el aviso del
    transitorio (`aviso_321_agrario`). Con la personal o la lista del Código
    Nacional en la Ciudad de México, («», aviso del transitorio Tercero,
    `aviso_cnpcf_agrario`): el precepto no se toca, el aviso sí sale."""
    import tipos_asunto as _ta_n
    if _ta_n.normalizar(tipo) not in ("amparo_directo", "revision_fiscal"):
        return "", ""
    # LA PERSONAL DEL CÓDIGO FEDERAL (`cfpc_personal`, integración, 3-oct-2026)
    # trae su precepto, pero fuera de la Ciudad de México es también la que el
    # desplegable propone, así que no se sabe si alguien la eligió: conserva el
    # aviso del transitorio, como la «personal» genérica. En la Ciudad de México
    # sólo llega a propósito (la propuesta ahí es la del Código Nacional) y no
    # se avisa de nada. El precepto que se devuelve es el mismo que el de la
    # regla, de modo que el párrafo no cambia; sólo engancha el aviso.
    if str(regla or "") == "cfpc_personal":
        _cdmx_c = sede_cdmx(tribunal, ciudad)
        return (("", "") if _cdmx_c is True
                else (PRECEPTO_321_AGRARIO, aviso_321_agrario(_cdmx_c, "cfpc_personal")))
    # Y LA DEL CÓDIGO NACIONAL EN LA CIUDAD DE MÉXICO, POR LA MISMA RAZÓN
    # (revisión de normas y front, 3-oct-2026): allí es la que el desplegable
    # propone (`reglas_para`) y la que pone el guardián, así que tampoco se sabe
    # si alguien la eligió. Sin precepto («», la regla ya trae el suyo) y con el
    # aviso del transitorio Tercero (`aviso_cnpcf_agrario`). Fuera de ella sólo
    # llega elegida a propósito y no se avisa. La electrónica nunca es omisión.
    if str(regla or "") in ("cnpcf_personal", "cnpcf_lista"):
        if sede_cdmx(tribunal, ciudad) is True:
            return "", aviso_cnpcf_agrario(str(regla))
        return "", ""
    r = REGLAS_SURTE.get(str(regla or ""))
    if r is None or r.dias_habiles != 1 or r.fundamento:
        return "", ""
    # LO AGRARIO, POR LA SEDE (3-oct-2026). Va primero: un juicio agrario no es
    # mercantil, y un Tribunal Unitario Agrario no es una Sala del TFJA aunque
    # el asunto se registre como administrativo. Se reconoce por la materia, el
    # órgano o el expediente de origen («juicio agrario 451/2023»), NO por el
    # contexto: unos antecedentes civiles pueden mencionar un juicio agrario
    # anterior, y citar el 321 «en materia agraria» ahí sería falso.
    if es_agrario(materia, responsable, expediente):
        _cdmx = sede_cdmx(tribunal, ciudad)
        if _cdmx is True:
            return "", ""
        return PRECEPTO_321_AGRARIO, aviso_321_agrario(_cdmx, str(regla or ""))
    _merc = es_mercantil(materia, responsable, expediente, contexto)
    if _merc:
        _de_donde = {"materia": "porque el juicio de origen es mercantil",
                     "órgano": ("porque la responsable es un órgano mercantil, aunque el "
                                "asunto se registró como de otra materia"),
                     "expediente": ("porque el expediente de origen es de un juicio mercantil, "
                                    "aunque el asunto se registró como de otra materia"),
                     "contexto": ("porque el juicio de origen se describe como mercantil, "
                                  "aunque el asunto se registró como de otra materia")}[_merc]
        return ("el artículo 1075 del Código de Comercio",
                "EL SURTIMIENTO SE FUNDÓ EN EL ARTÍCULO 1075 DEL CÓDIGO DE COMERCIO "
                "(las notificaciones personales y las demás surten al día siguiente), "
                f"{_de_donde}, y esa ley es federal. "
                "COMPRUÉBALO: si la sentencia se dictó en la audiencia de un juicio "
                "oral mercantil, se tuvo por notificada en el acto (art. 1390 Bis 22) "
                "y la regla es otra; si tienes el precepto, escríbelo en la ficha "
                "(«fundamento del surtimiento»).")
    if fuero_de(tipo, responsable) == "tfja":
        if str(regla or "") != "personal":
            return "", ""
        return ("los " + REGLAS_SURTE["lfpca"].fundamento,
                "EL SURTIMIENTO SE FUNDÓ EN LOS ARTÍCULOS 65 Y 70 DE LA LFPCA (la "
                "notificación personal surte al día hábil siguiente), porque la "
                "responsable es una Sala del Tribunal Federal de Justicia "
                "Administrativa. COMPRUÉBALO: si se notificó por Boletín "
                "Jurisdiccional, la regla es la del tercer día hábil (art. 65).")
    return "", ""


def aviso_fundamento(c, tipo: str, papel: str = "", en_hueco: bool = False,
                     fundamento_surtimiento: str = "", forma_consta: bool = True,
                     fuente_forma: str = "") -> str:
    """Lo que el considerando no puede fundar solo. `en_hueco`: el párrafo
    dejó el precepto en hueco en vez de «conforme a la ley del acto».
    Con `fundamento_surtimiento` (el que dio el secretario, o la omisión
    nacional de `surtimiento_nacional`) el precepto ya está escrito.
    `forma_consta`/`fuente_forma` (F1, quinta ronda): si la forma no consta, el
    aviso no la repite («la notificación surtió efectos…»), como el párrafo."""
    if str(fuente_forma or "").strip().lower() in FUENTES_FORMA_OMISION:
        forma_consta = False
    if getattr(c.regla, "clave", "") == "otra":
        return ""
    if fundamento_de_surtimiento(c.regla, tipo, papel):
        return ""
    import tipos_asunto as _ta_a
    if conforme_al_precepto(fundamento_surtimiento) and not (
            _ta_a.normalizar(tipo) in ("amparo_revision", "queja")
            and (papel or "").strip().lower() == "autoridad"):
        return ""
    if (_ta_a.normalizar(tipo) in ("amparo_revision", "queja")
            and (papel or "").strip().lower() == "autoridad"
            and getattr(c.regla, "clave", "") in ("personal", "lista")):
        return ("RECURRE UNA AUTORIDAD Y EL CÓMPUTO USÓ LA REGLA DE LOS "
                "PARTICULARES (notificación " + c.regla.descripcion + ", surte al día "
                "hábil siguiente, artículo 31, fracción II). A la autoridad se le "
                "notifica por oficio y surte efectos desde que la notificación "
                "queda hecha (artículo 31, fracción I, de la Ley de Amparo), o, si "
                "fue electrónica, al generarse la constancia de consulta "
                "(fracción III): el plazo empieza un día antes. El fundamento del "
                "surtimiento va en HUECO: elige «oficio» o «electrónica» como "
                "regla de notificación y vuelve a generar.")
    if en_hueco:
        # D1 (3-oct-2026): YA NO HAY HUECO. El considerando dice «surtió efectos
        # al día hábil siguiente, es decir, el…» sin cláusula de precepto, como
        # los engroses de la ponencia de David (AD 274/2025 y 335/2025). El
        # aviso RECOMIENDA el artículo; no es una tarea pendiente del texto.
        _la_notif = (f"la notificación {c.regla.descripcion}" if forma_consta
                     else "la notificación")
        return ("SE RECOMIENDA CITAR EL PRECEPTO DEL SURTIMIENTO: el considerando dice "
                f"que {_la_notif} surtió efectos "
                f"{_ORDINAL_SURTE.get(c.regla.dias_habiles, 'al día hábil siguiente')} "
                "sin citar el artículo que lo dispone, como lo hacen los engroses que "
                "no lo citan, porque el de la ley que rige el acto no está en el "
                "catálogo. Si lo tienes, escríbelo en la ficha («fundamento del "
                "surtimiento») y el considerando lo citará.")
    return ("EL SURTIMIENTO VA SIN PRECEPTO: el considerando dice que la "
            f"notificación {c.regla.descripcion} surtió efectos "
            f"{_ORDINAL_SURTE.get(c.regla.dias_habiles, 'al día hábil siguiente')} "
            "«conforme a la ley del acto», porque el catálogo no trae el artículo "
            "de esa ley. Escríbelo: es lo que sostiene el día en que arrancó el plazo.")


def tramos_en_letra(tramos) -> str:
    """[(29-jun, 10-jul), (2-ene, 2-ene)] → «del veintinueve de junio al diez de
    julio y el dos de enero». Un tramo de un día se dice como día.

    CON `TramosInhabiles` (los de `tramos_inhabiles` sobre el calendario del
    amparo), CADA GRUPO CON SU FUNDAMENTO ENTRE PARÉNTESIS y los del artículo
    19 al final, sin él (3-oct-2026). Así quien escribe «sin contar sábados y
    domingos, ni {esto}, por ser inhábiles en términos del artículo 19 de la Ley
    de Amparo» —el considerando del adhesivo— ya no atribuye al 19 el dos de
    mayo de una circular. Para escribir la cláusula entera, `clausula_inhabiles`."""
    if isinstance(tramos, TramosInhabiles) and any(
            k != "art19" for k in (getattr(tramos, "claves", None) or [])):
        partes_g = []
        for k, ts in grupos_inhabiles(tramos, art19_al_final=True):
            if k == "art19":
                partes_g.append(_lista_compacta(ts))
            else:
                r = _razon_del_grupo(k, ts)
                r = r.replace("por ser ", "", 1).replace("por corresponder al ", "", 1) \
                     .replace("por corresponder a los ", "", 1)
                partes_g.append(f"{_lista_compacta(ts)} ({r})")
        return ", ni ".join(partes_g)
    if isinstance(tramos, TramosInhabiles):
        return _lista_compacta(tramos)
    partes = []
    for ini, fin in tramos:
        if ini == fin:
            partes.append(f"el {fecha_en_letra(ini)}")
        elif (ini.month, ini.year) == (fin.month, fin.year):
            partes.append(f"del {_UNIDADES[ini.day]} al {fecha_en_letra(fin)}")
        else:
            partes.append(f"del {fecha_en_letra(ini)} al {fecha_en_letra(fin)}")
    if not partes:
        return ""
    if len(partes) == 1:
        return partes[0]
    return ", ".join(partes[:-1]) + f" y {partes[-1]}"


def suspensiones_que_lo_salvarian(c, tope: int = 20) -> Optional[int]:
    """Cuántos días de suspensión de la responsable DENTRO del plazo harían
    oportuno lo que hoy sale extemporáneo. None si no aplica o si ni con `tope`.

    ES LA CALIBRACIÓN DEL AVISO. El aviso viejo —«No hay calendario declarado
    para X»— salía en el 100% de las sesiones y por eso no lo leía nadie. Éste
    sólo habla cuando el dato que falta puede cambiar el veredicto, y dice
    cuántos días harían falta. En el amparo directo 93/2026 la respuesta es
    DOS, y el asunto está declarado extemporáneo por cuatro.

    Cada día hábil que se declara inhábil dentro del plazo empuja el
    vencimiento al siguiente hábil: se busca hacia adelante, no se estima.
    """
    if c.presentacion is None or c.oportuna is not False:
        return None
    v = c.vencimiento
    for k in range(1, tope + 1):
        v = c.cal_amparo.siguiente_habil(v + _dt.timedelta(days=1))
        if c.presentacion <= v:
            return k
    return None


def _el(x: str) -> str:
    """«la demanda de amparo», «el recurso de queja». Gemelo de `_del`: aquél
    contrae la preposición y éste sólo pone el artículo. Sin él salía «al
    presentarse DE LA demanda de amparo por conducto de dicha autoridad», que
    no es español y habría ido a un considerando firmado."""
    x = (x or "").strip()
    return f"el {x}" if x.split()[:1] and x.split()[0] in (
        "recurso", "amparo", "juicio") else f"la {x}"


def lista_en_letra_con_anio(fechas) -> str:
    """Como `lista_en_letra`, pero si las fechas cruzan de año lleva el año.

    El párrafo cerraba siempre con «del referido año», y en un plazo que va del
    nueve de diciembre al diecinueve de enero eso es FALSO para la mitad de los
    días. Sale en cuanto el plazo cruza diciembre, que es precisamente cuando
    más días inhábiles hay que nombrar.
    """
    fs = sorted(fechas)
    if not fs:
        return ""
    if fs[0].year == fs[-1].year:
        return lista_en_letra(fs) + " del referido año"
    partes = [f"el {_UNIDADES[f.day]} de {_MESES[f.month]} de {_anio_en_letra(f.year)}"
              for f in fs]
    return ", ".join(partes[:-1]) + f" y {partes[-1]}"


def computar(
    notificacion: Fecha,
    presentacion: Optional[Fecha] = None,
    regla: str = "personal",
    plazo: int = 15,
    responsable: Optional[str] = None,
    inhabiles_extra: Optional[list] = None,
    tipo_asunto: str = "amparo_directo",
    inhabiles_responsable=None,
    surtio_manual: Optional[Fecha] = None,
    plazo_anios: int = 0,
    deposito: Optional[Fecha] = None,
) -> Computo:
    """El cómputo completo, con los dos calendarios.

    `deposito` (D3, 3-oct-2026) es la fecha en que el oficio de una REVISIÓN
    FISCAL se depositó en el Servicio Postal Mexicano: si consta, la oportunidad
    se mide con ella y `presentacion` queda como la recepción en la Sala
    (`Computo.recepcion`). En otro tipo de asunto, o si es posterior a la
    recepción, no se usa y se avisa.

    `plazo_anios` (8 o 7) es el plazo en AÑOS del artículo 17, fracciones II y
    III, de la Ley de Amparo: se cuenta de fecha a fecha desde que surtió
    efectos la notificación, no en días hábiles; si el último día es inhábil,
    el plazo se extiende al siguiente hábil.

    `plazo` en días hábiles: 15 para amparo directo (art. 17 LA), 10 para la
    revisión (art. 86), 5 para la queja urgente.
    `responsable` es el nombre —o la clave— de la autoridad responsable.

    `inhabiles_extra` son los días que el SECRETARIO declara inhábiles para el
    ÓRGANO FEDERAL y que el calendario del OAJ no trae: un inhábil por circuito
    de los que declaran las circulares del Órgano de Administración Judicial,
    una contingencia. OJO: en amparo directo, los días en que el propio
    Tribunal Colegiado suspendió labores NO se descuentan —así lo dice la
    P./J. 4/2022—, y este campo no es el sitio para ponerlos.

    `tipo_asunto` decide ANTE QUIÉN se presenta el escrito, y con ello si los
    días de la responsable entran o no (arts. 176 y 86/99 LA, 63 LFPCA).

    `inhabiles_responsable` son los días y periodos en que la AUTORIDAD
    RESPONSABLE no laboró, declarados por quien los sabe:
    «2026-06-29..2026-07-10, 2026-01-02». En amparo directo se descuentan DEL
    PLAZO —no sólo del surtimiento— porque la demanda se presenta por su
    conducto; en amparo en revisión y en queja no se descuentan, y se dice.

    `surtio_manual` es LA REGLA «OTRA»: cuando el secretario declara él mismo
    la fecha en que la notificación surtió efectos —porque su asunto no
    encaja en ninguna de las reglas del catálogo y el redactor no le va a
    aplicar la de un tribunal ajeno— esa fecha se usa TAL CUAL, sin contar
    ningún día hábil y sin afirmar ningún fundamento que no conste. Sólo
    tiene efecto cuando `regla == "otra"`.
    """
    avisos: list[str] = []

    # ── EL DEPÓSITO POSTAL DE LA REVISIÓN FISCAL (D3) ──────────────────────
    # En la RF 2/2025 el depósito (24-oct-2024) caía en el plazo y la recepción
    # (6-nov) no: el proyecto desechaba por extemporáneo un recurso que el
    # tribunal confirmó. El tribunal mide con el depósito (RF 28/2025: «si el
    # referido medio de impugnación se depositó en la oficina de Correos de
    # México… es oportuno», con la tesis XV.4o.1 A (11a.)). Se mide con él y se
    # avisa: la ley dice «ante la responsable» (art. 63 LFPCA).
    _deposito = _fecha_iso(deposito) if deposito not in (None, "") else None
    _recepcion = None
    if _deposito is not None:
        if receptor_de(tipo_asunto).clave != "revision_fiscal":
            avisos.append(
                f"EL DEPÓSITO POSTAL ({_deposito.isoformat()}) NO SE USÓ: la fecha de "
                f"depósito sólo mide la oportunidad en la revisión fiscal interpuesta por "
                f"correo. El cómputo se hizo con la presentación.")
            _deposito = None
        elif presentacion is not None and _deposito > presentacion:
            avisos.append(
                f"FECHA IMPOSIBLE: el depósito postal ({_deposito.isoformat()}) es "
                f"POSTERIOR a la recepción en la Sala ({presentacion.isoformat()}). No se "
                f"usó; el cómputo se hizo con la recepción. Corrige la fecha que esté mal.")
            _deposito = None
        else:
            _recepcion, presentacion = presentacion, _deposito

    _anios = int(plazo_anios or 0)
    _sin_plazo = (plazo is None or int(plazo) <= 0) and not _anios
    _plazo = 0 if (_sin_plazo or _anios) else int(plazo)
    if _sin_plazo:
        avisos.append(
            "NO SE COMPUTÓ PLAZO: este asunto no lo tiene —procede en cualquier "
            "tiempo—, así que no hay vencimiento ni puede declararse "
            "extemporáneo. Las fechas de notificación y de surtimiento sí se "
            "calcularon, porque los antecedentes las nombran.")

    if presentacion is not None and presentacion < notificacion:
        avisos.append(
            f"FECHA IMPOSIBLE: la presentación ({presentacion.isoformat()}) es "
            f"ANTERIOR a la notificación ({notificacion.isoformat()}). O una de "
            f"las dos está mal capturada, o se promovió antes de la "
            f"notificación formal por conocimiento previo del acto —y eso hay "
            f"que decirlo y razonarlo—. NO se ha calculado el plazo sobre este "
            f"supuesto.")

    _hoy = _date.today()
    for _que, _f in (("notificación", notificacion), ("presentación", presentacion)):
        if _f is None:
            continue
        if _f > _hoy:
            avisos.append(
                f"FECHA EN EL FUTURO: la {_que} ({_f.isoformat()}) es posterior "
                f"a hoy ({_hoy.isoformat()}). Revisa la captura.")
        elif _f.year < 2013:
            avisos.append(
                f"FECHA FUERA DE ÉPOCA: la {_que} ({_f.isoformat()}) es anterior "
                f"a la Ley de Amparo vigente (2 de abril de 2013). Si el dato es "
                f"correcto, el asunto se rige por la ley abrogada y este cómputo "
                f"no le sirve.")

    _extra = {d for d in (inhabiles_extra or []) if d}

    # ── LOS DÍAS DE LA AUTORIDAD RESPONSABLE ───────────────────────────────
    rec = receptor_de(tipo_asunto)
    ir = leer_inhabiles_responsable(inhabiles_responsable)
    avisos.extend(ir.errores)
    _resp = set(ir.dias) if rec.aporta_calendario else set()

    # UN SOLO deepcopy, y sólo si hace falta. `CALENDARIO_AMPARO` es objeto de
    # módulo: mutarlo dejaría esos días inhábiles para todos los asuntos que
    # atendiera este proceso después, y con `gunicorn -w 2` eso es un worker
    # envenenado y otro sano resolviendo el mismo expediente distinto.
    cal_amparo = CALENDARIO_AMPARO
    if _extra or _resp:
        import copy as _copy
        cal_amparo = _copy.deepcopy(CALENDARIO_AMPARO)
        # SON DOS LISTAS QUE SE SUMAN, NO UNA QUE SUSTITUYE A LA OTRA. La
        # P./J. 4/2022 lo dice literal: se excluyen «los días inhábiles
        # establecidos por el artículo 19 de la Ley de Amparo, aun cuando la
        # autoridad responsable no haya suspendido labores, así como aquellos
        # en que dicha autoridad suspenda actividades, no obstante que estén
        # contemplados como hábiles por la referida legislación». La unión de
        # los dos conjuntos es exactamente eso.
        cal_amparo.sueltos = set(cal_amparo.sueltos) | _extra | _resp

    _manual = regla == "otra" and surtio_manual is not None
    if _manual:
        # NO HAY REGLA QUE APLICAR: el secretario dio las dos fechas y el
        # cómputo no cuenta ningún día hábil ni afirma ningún fundamento. Los
        # `dias_habiles` quedan en -1 como centinela: `parrafo_oportunidad()`
        # lo usa para no escribir «al Nth día hábil», que sería inventado.
        r = ReglaSurte(clave="otra", descripcion="conforme a lo manifestado",
                       dias_habiles=-1, fundamento="")
    else:
        r = REGLAS_SURTE.get(regla)
        if r and str(getattr(r, "clave", "")).endswith("_qro_boletin"):
            avisos.append(
                "El cómputo usa la regla del Boletín Jurisdiccional del Tribunal de "
                "Justicia Administrativa de QUERÉTARO (surte al tercer día hábil). "
                "Si tu asunto es de otra entidad, comprueba cómo surte efectos la "
                "notificación en la ley que rige el acto: un plazo mal contado "
                "invalida la sentencia.")
        # EL BOLETÍN DEL TFJA Y LA REFORMA DE 2026 (verificación de normas,
        # 27-sep-2026). El art. 65, último párrafo, LFPCA dice desde el DOF
        # 09-06-2026 que la notificación por Boletín surte «al segundo día
        # hábil»; el transitorio Tercero difiere los plazos de ese artículo a
        # los 240 días naturales, que caen el 04-02-2027. Hasta entonces se
        # cuenta el tercero, con el texto anterior; desde entonces, el segundo.
        if (r is not None and str(getattr(r, "clave", "")) == "lfpca_boletin"
                and notificacion >= _dt.date(2026, 6, 10)):
            if notificacion >= LFPCA_65_REFORMADO:
                r = ReglaSurte(clave="lfpca_boletin", descripcion=r.descripcion,
                               dias_habiles=2,
                               fundamento="artículo 65 de la Ley Federal de "
                                          "Procedimiento Contencioso Administrativo")
            else:
                r = ReglaSurte(clave="lfpca_boletin", descripcion=r.descripcion,
                               dias_habiles=3,
                               fundamento="artículo 65 de la Ley Federal de "
                                          "Procedimiento Contencioso Administrativo, "
                                          "en su texto anterior a la "
                                          "reforma publicada en el Diario Oficial de la "
                                          "Federación el nueve de junio de dos mil "
                                          "veintiséis, aplicable conforme al artículo "
                                          "Tercero transitorio de ésta")
            avisos.append(
                "BOLETÍN DEL TFJA: el artículo 65 reformado (DOF 09-06-2026) dice "
                "que surte al SEGUNDO día hábil, pero su transitorio Tercero "
                "difiere los plazos de ese artículo 240 días (hasta el "
                "04-02-2027). Se contó "
                + ("el segundo." if notificacion >= LFPCA_65_REFORMADO else
                   "el tercero, con el texto anterior.")
                + " Si tu tribunal lo lee de otro modo, cambia la regla.")
        if r is None:
            r = REGLAS_SURTE["personal"]
            avisos.append(
                f"La regla de surtimiento «{regla}» no está declarada. Se contó "
                "como notificación personal. COMPRUEBA la ley que rige el acto "
                "antes de firmar."
            )
        # ANTE EL TFJA «POR LISTA» NO ES UNA REGLA (3-oct-2026, rev_2). La LFPCA
        # notifica por Boletín Jurisdiccional lo que no es personal (arts. 66 y
        # 67), y eso surte al TERCER día hábil (art. 65): contar uno es arrancar
        # el plazo dos hábiles antes, y una demanda oportuna sale extemporánea.
        # Con la bandera: el camino viejo no gana avisos.
        if (_rige_procedencia() and str(getattr(r, "clave", "")) == "lista"
                and receptor_de(tipo_asunto).clave in ("amparo_directo", "revision_fiscal")
                and fuero_de(tipo_asunto, responsable or "") == "tfja"):
            avisos.append(
                "LA SALA DEL TFJA NO NOTIFICA «POR LISTA»: lo que no es personal se "
                "notifica por Boletín Jurisdiccional (artículos 66 y 67 de la LFPCA) y "
                "surte efectos al tercer día hábil (artículo 65). El cómputo se hizo con "
                "la regla que elegiste (un día); si la notificación fue por Boletín, "
                "elige «Boletín Jurisdiccional del TFJA» y vuelve a generar.")

    cal_resp = CALENDARIOS_RESPONSABLE.get(responsable or "", CALENDARIO_AMPARO)
    if _resp:
        # EL SURTIMIENTO TAMBIÉN. Si la responsable no laboró, su notificación
        # no pudo surtir efectos corriendo sus días. Hasta hoy los días
        # declarados sólo entraban al calendario del PLAZO y el surtimiento no
        # se movía: con la regla del boletín eso son hasta tres días de
        # diferencia, que arrastran el inicio y el vencimiento.
        import copy as _copy
        cal_resp = _copy.deepcopy(cal_resp)
        cal_resp.nombre = (responsable or "").strip() or "la autoridad responsable"
        cal_resp.fundamento = rec.fundamento
        cal_resp.sueltos = set(cal_resp.sueltos) | _resp

    # LA REVISIÓN FISCAL CORRE CON EL CALENDARIO DEL TFJA (art. 74, fr. II,
    # LFPCA; tesis 2007213 y 239295): el surtimiento y el plazo, sin las
    # vacaciones del Poder Judicial de la Federación y con las del Tribunal.
    _es_rf = rec.clave == "revision_fiscal"
    _base_rf = None
    if _es_rf:
        import copy as _copy
        _base_rf = calendario_tfja(responsable or "")
        cal_resp = _copy.deepcopy(_base_rf)
        cal_resp.nombre = (responsable or "").strip() or _base_rf.nombre
        cal_resp.sueltos = set(cal_resp.sueltos) | _resp
        cal_amparo = _copy.deepcopy(_base_rf)
        cal_amparo.sueltos = set(cal_amparo.sueltos) | _extra | _resp

    # 1) Surtimiento — calendario de la RESPONSABLE, salvo que el secretario
    # ya haya dicho la fecha (regla «otra»).
    if _manual:
        surtio = surtio_manual
        if surtio < notificacion:
            avisos.append(
                f"FECHA IMPOSIBLE: dijiste que la notificación surtió efectos "
                f"el {surtio.isoformat()}, ANTES de que se notificara "
                f"({notificacion.isoformat()}). Revisa las dos fechas antes "
                f"de firmar.")
    else:
        surtio = notificacion
        contados = 0
        while contados < r.dias_habiles:
            surtio += _dt.timedelta(days=1)
            if cal_resp.es_habil(surtio):
                contados += 1

    # 2) El plazo — calendario del AMPARO, con los inhábiles declarados dentro
    inicio = cal_amparo.siguiente_habil(surtio + _dt.timedelta(days=1))
    dias = cal_amparo.sumar(inicio, _plazo) if _plazo else []
    vence = dias[-1] if dias else inicio
    if _anios:
        # AÑOS CALENDARIO, EN DÍAS NATURALES (1a./J. 41/2023 (11a.), registro
        # 2026377: «debe computarse en años calendario; esto es incluyendo
        # todos los días naturales»), sin descontar los inhábiles del art. 19.
        # Corre desde el día siguiente al surtimiento (art. 18) y concluye en
        # la misma fecha de ese primer día, N años después —así lo cuentan los
        # colegiados: «del tres de abril de dos mil trece al tres de abril de
        # dos mil veintiuno»—.
        inicio = surtio + _dt.timedelta(days=1)
        try:
            vence = inicio.replace(year=inicio.year + _anios)
        except ValueError:                      # 29 de febrero → 1 de marzo
            vence = _dt.date(inicio.year + _anios, 3, 1)
        # SI EL ÚLTIMO DÍA ES INHÁBIL, ni la ley ni la jurisprudencia dicen qué
        # pasa. No se declara extemporánea en automático la demanda del día
        # hábil siguiente: se corre al siguiente hábil y se avisa.
        if not cal_amparo.es_habil(vence):
            _nominal = vence
            vence = cal_amparo.siguiente_habil(vence)
            avisos.append(
                f"EL ÚLTIMO DÍA DEL PLAZO DE {_anios} AÑOS CAYÓ EN INHÁBIL "
                f"({_nominal.isoformat()}) y se corrió al siguiente hábil "
                f"({vence.isoformat()}). La Ley de Amparo no lo regula y no hay "
                f"jurisprudencia: si la demanda se presentó ese día, decídelo tú.")
        # EL DÍA ANIVERSARIO ES FRONTERA: contar 365/366 días incluyendo el
        # primero termina la víspera. Si la demanda cae justo ese día, lo
        # decide quien firma.
        if presentacion is not None and presentacion == vence:
            avisos.append(
                f"LA DEMANDA SE PRESENTÓ EL DÍA ANIVERSARIO ({vence.isoformat()}): "
                f"contado de fecha a fecha está en tiempo, pero un cómputo de "
                f"365 días que incluya el primero termina la víspera. La "
                f"jurisprudencia 1a./J. 41/2023 no fija el día exacto: decídelo tú.")
        dias = []

    # Los inhábiles entre semana dentro del plazo, SEPARADOS POR FUNDAMENTO.
    # Un día que ya era inhábil por el artículo 19 se atribuye al artículo 19
    # aunque la responsable también hubiera cerrado: la tesis descuenta esos
    # días «aun cuando la autoridad responsable no haya suspendido labores», y
    # nombrarlo dos veces en el considerando sería un error de bulto.
    _art19 = _base_rf if _es_rf else CALENDARIO_AMPARO
    if _es_rf:
        # EL FUNDAMENTO NOMBRA EL ACUERDO DEL AÑO; si falta un año, se avisa.
        _anios_v = {surtio.year, inicio.year, vence.year}
        cal_amparo.fundamento = fundamento_tfja(_anios_v)
        # EL JUICIO EN LÍNEA 2.0 ESTUVO SUSPENDIDO del 18 de mayo al 25 de mayo
        # de 2026 a las 08:29 (G/JGA/58/2026 y 59/2026, DOF 26 y 27-05-2026),
        # sólo para los expedientes que se tramitan en línea: no se descuenta
        # solo, porque en papel esos días corrieron. Se avisa si caen dentro.
        _jel = (_dt.date(2026, 5, 18), _dt.date(2026, 5, 25))
        if inicio <= _jel[1] and vence >= _jel[0]:
            avisos.append(
                "SI EL JUICIO SE TRAMITÓ EN LÍNEA (Sistema de Justicia en Línea "
                "2.0), sus plazos estuvieron suspendidos del 18 de mayo de 2026 "
                "a las 8:29 horas del 25 de mayo (Acuerdos G/JGA/58/2026 y "
                "G/JGA/59/2026): declara esos días. En papel corrieron.")
        _faltan = sorted(a for a in _anios_v if a not in ACUERDO_TFJA)
        if _faltan:
            avisos.append(
                f"NO ESTÁ CARGADO EL CALENDARIO DEL TFJA DE {', '.join(map(str, _faltan))} "
                f"(su Pleno General lo publica en el DOF a mediados de enero): el "
                f"plazo de la revisión fiscal se contó sólo sin sábados ni domingos "
                f"en ese año. Comprueba sus días inhábiles antes de firmar.")
    enmedio, resp_enmedio, cur = [], [], inicio
    while cur <= vence and not _anios:
        if cur.weekday() < 5 and not cal_amparo.es_habil(cur):
            if not _art19.es_habil(cur) or cur in _extra:
                enmedio.append(cur)
            else:
                resp_enmedio.append(cur)
        cur += _dt.timedelta(days=1)

    # LOS INHÁBILES ENTRE LA NOTIFICACIÓN Y EL INICIO (3-oct-2026, rev_0 y rev_1).
    # El bucle de arriba arranca en `inicio`, así que el considerando sólo
    # fundaba los del plazo: AD 335/2025 (notificado el 19 de marzo, surtió el
    # 20, el viernes 21 inhábil por el art. 19 y el plazo al lunes 24) callaba
    # el 21 que el engrose nombra; Q 24/2026 saltaba las vacaciones de
    # diciembre. Sólo los que de verdad se saltaron: antes del surtimiento, en
    # el calendario con que se contó el surtimiento; después, en el del plazo.
    # Y sólo los oficiales (o los que declaró el secretario para el órgano
    # federal): los de la responsable cuentan dentro del plazo y nada más. Con
    # la regla «otra» el surtimiento lo declaró el secretario y lo anterior a él
    # no se explica aquí.
    previos = []
    if not _anios:
        cur = notificacion + _dt.timedelta(days=1)
        while cur < inicio:
            if cur.weekday() < 5:
                if cur <= surtio:
                    _salto = (not _manual) and not cal_resp.es_habil(cur)
                    _oficial = not _art19.es_habil(cur)
                else:
                    _salto = not cal_amparo.es_habil(cur)
                    _oficial = not _art19.es_habil(cur) or cur in _extra
                if _salto and _oficial:
                    previos.append(cur)
            cur += _dt.timedelta(days=1)

    if _es_rf:
        # LOS ACUERDOS DEL TFJA QUE FIJARON LOS DÍAS QUE SE SALTARON, como
        # regían cuando corrió el plazo (RF 6/2026: el 2 de enero de 2026, el
        # SS/22/2025; no el SS/2/2026, publicado el 12 de enero).
        _dias_tfja = [d for d in previos + enmedio
                      if d in INHABILES_TFJA.get(d.year, set())]
        cal_amparo.fundamento = fundamento_tfja(_anios_v, _dias_tfja, vence)

    # ── LA NOTIFICACIÓN O LA PRESENTACIÓN EN DÍA INHÁBIL (3-oct-2026) ───────
    # AD 552/2024: notificación personal el domingo 30 de junio de 2024; RF
    # 26/2025: Boletín Jurisdiccional el domingo 8 de diciembre. El cómputo las
    # aceptaba sin aviso, y es la fecha de la que depende todo. Se avisa —no se
    # corrige: una notificación electrónica o una promoción ante la guardia en
    # día inhábil existen—. El calendario es el de quien notifica o recibe: el
    # TFJA en la revisión fiscal, el del Poder Judicial de la Federación en los
    # recursos y, en el amparo directo (la responsable, cuyo calendario no se
    # tiene), los fines de semana, los días que se declararon suyos y los
    # feriados que observa toda la república. Con la bandera, como el anterior.
    _feriados = {(1, 1), (1, 5), (16, 9), (25, 12)}

    def _inhabil_de_quien(f: Fecha, cal) -> bool:
        if cal is not None:
            return not cal.es_habil(f)
        return f.weekday() >= 5 or f in ir.dias or (f.month, f.day) in _feriados

    _cal_notif = (cal_resp if _es_rf else
                  cal_amparo if rec.clave in ("amparo_revision", "queja") else None)
    _rige_pt = _rige_procedencia()
    # La electrónica del Código Nacional en lo agrario (art. 227, fr. III) es
    # tan electrónica como la del 31: puede caer en inhábil sin ser error.
    if (_rige_pt and str(getattr(r, "clave", "")) not in ("electronica", "cnpcf_electronica")
            and _inhabil_de_quien(notificacion, _cal_notif)):
        avisos.append(
            f"LA NOTIFICACIÓN CAE EN DÍA INHÁBIL ({_DIAS_SEMANA[notificacion.weekday()]} "
            f"{fecha_en_letra(notificacion)}): compruébala contra la constancia. Una "
            f"notificación {r.descripcion} en día inhábil suele ser un error de captura, "
            f"y de esa fecha depende todo el cómputo.")
    if (_rige_pt and presentacion is not None and _deposito is None
            and _inhabil_de_quien(presentacion, cal_amparo if rec.clave != "amparo_directo" else None)):
        avisos.append(
            f"LA PRESENTACIÓN CAE EN DÍA INHÁBIL ({_DIAS_SEMANA[presentacion.weekday()]} "
            f"{fecha_en_letra(presentacion)}): compruébala contra el sello o el acuse. Si "
            f"se presentó por vía electrónica o ante la guardia, el cómputo vale; si no, la "
            f"fecha está mal capturada.")

    # Los tramos, recortados a la ventana del plazo y a los días que de verdad
    # descontaron: el considerando escribe «del veintinueve de junio al diez de
    # julio», no diez fechas seguidas.
    _dentro = set(resp_enmedio)
    resp_tramos = []
    for _i, _f in ir.tramos:
        _ha = [d for d in _fechas_entre(max(_i, inicio), min(_f, vence)) if d in _dentro]
        if _ha:
            resp_tramos.append((_ha[0], _ha[-1]))

    if _extra:
        dentro = sorted(d for d in _extra if inicio <= d <= vence)
        if dentro:
            avisos.append(
                f"Se contaron como inhábiles los {len(dentro)} día(s) que "
                f"declaraste dentro del plazo: {lista_en_letra(dentro)}.")
        antes = sorted(d for d in _extra if surtio < d < inicio)
        if antes:
            avisos.append(
                f"Los {len(antes)} día(s) que declaraste antes del arranque "
                f"({lista_en_letra(antes)}) corrieron el inicio del plazo al "
                f"{fecha_en_letra(inicio)}.")
        sobran = sorted(d for d in _extra if d > vence or d <= surtio)
        if sobran:
            avisos.append(
                f"Declaraste {len(sobran)} día(s) inhábil(es) fuera de la "
                f"ventana del cómputo ({lista_en_letra(sobran)}): no lo "
                f"cambian.")
        # LA RAMA QUE FALTABA, Y ES LA QUE MINTIÓ EN 93/2026: declarar días que
        # YA eran inhábiles no es trabajo útil, y decir «se contaron» hace pasar
        # por útil el trabajo que no sirvió, justo en el asunto que acabó
        # extemporáneo. Los diez días que el secretario declaró a mano ahí eran
        # los diez de la segunda quincena de diciembre que el calendario ya
        # traía.
        _ya = sorted(d for d in dentro if not _art19.es_habil(d))
        if _ya:
            avisos.append(
                f"OJO: {len(_ya)} de esos días YA eran inhábiles en el "
                f"calendario federal ({lista_en_letra(_ya)}), así que "
                f"declararlos no movió el vencimiento ni un día.")
        # ¿DE QUIÉN SON ESOS DÍAS? En amparo directo la pregunta no es de
        # higiene: la P./J. 4/2022 descuenta los días de la RESPONSABLE y
        # excluye expresamente los de suspensión extraordinaria del TRIBUNAL
        # COLEGIADO. El rótulo de este campo decía «aquí sólo los de tu
        # tribunal», que es la categoría contraria. Se PREGUNTA, no se acusa:
        # un inhábil por circuito de los que declaran las circulares del Órgano
        # de Administración Judicial es un uso legítimo de este campo.
        if dentro and rec.clave == "amparo_directo":
            avisos.append(
                f"COMPRUEBA DE QUIÉN SON esos {len(dentro)} día(s): se "
                f"descontaron del calendario FEDERAL. Si son días en que "
                f"suspendió labores TU tribunal, en amparo directo no se "
                f"descuentan (P./J. 4/2022, registro digital 2024494); si son "
                f"de la responsable, van en su propio campo y el considerando "
                f"los funda con el {rec.fundamento}.")

    if ir.dias and not rec.aporta_calendario:
        # DECLARÓ Y NO SE APLICA. Se le dice, con su artículo: no es un olvido
        # del sistema, es que la ley no lo permite en este tipo de asunto.
        avisos.append(
            f"Declaraste {len(ir.dias)} día(s) de suspensión de la autoridad "
            f"responsable y NO se descontaron. En este asunto el escrito se "
            f"presenta ante {rec.quien} ({rec.fundamento}), no ante la "
            f"responsable, de modo que sus días no alargan el plazo. Si lo que "
            f"cerró fue el órgano que RECIBE el escrito, decláralo en «días "
            f"inhábiles adicionales».")
    elif _resp:
        if resp_enmedio:
            avisos.append(
                f"Se descontaron {len(resp_enmedio)} día(s) hábil(es) en que la "
                f"responsable no laboró ({tramos_en_letra(resp_tramos)}), "
                f"conforme al {rec.fundamento}.")
        _ya_r = sorted(d for d in _resp
                       if inicio <= d <= vence and not _art19.es_habil(d)
                       and d.weekday() < 5)
        if _ya_r:
            avisos.append(
                f"De los días de la responsable que declaraste, {len(_ya_r)} YA "
                f"eran inhábiles en el calendario federal "
                f"({lista_en_letra(_ya_r)}): no cambiaron el cómputo.")
        if not resp_enmedio:
            avisos.append(
                "Ninguno de los días de la responsable que declaraste cayó en "
                "día hábil dentro del plazo: el cómputo no cambió.")

    c = Computo(
        notificacion=notificacion, regla=r, surtio=surtio, inicio=inicio,
        vencimiento=vence, plazo=_plazo, dias=dias, presentacion=presentacion,
        inhabiles_en_medio=enmedio, cal_responsable=cal_resp,
        cal_amparo=cal_amparo, avisos=avisos, sin_plazo=_sin_plazo,
        resp_dias=sorted(ir.dias), resp_en_medio=resp_enmedio,
        resp_tramos_en_medio=resp_tramos, resp_declarados=bool(ir.dias),
        resp_aplicados=bool(resp_enmedio), receptor=rec,
        responsable_nombre=(responsable or "").strip(),
        plazo_anios=_anios,
        inhabiles_previos=previos, deposito=_deposito, recepcion=_recepcion,
    )
    if _deposito is not None:
        _con_recepcion = ""
        if (_recepcion is not None and _recepcion > vence
                and not c.en_cualquier_tiempo):
            _con_recepcion = (f" Con la fecha de recepción ({_recepcion.isoformat()}) el "
                              f"recurso habría sido EXTEMPORÁNEO (venció el "
                              f"{vence.isoformat()}): de esa fecha depende el sentido.")
        avisos.append(
            f"LA OPORTUNIDAD SE MIDIÓ CON LA FECHA DEL DEPÓSITO EN EL SERVICIO POSTAL "
            f"MEXICANO ({_deposito.isoformat()})"
            + (f", no con la de recepción en la Sala ({_recepcion.isoformat()})"
               if _recepcion is not None else "")
            + ", como lo hace el tribunal con la tesis aislada XV.4o.1 A (11a.), de "
              "registro digital 2025728. COMPRUÉBALO: la fecha debe constar en la guía o el "
              "sobre con el sello original, y el artículo 63 de la Ley Federal de "
              "Procedimiento Contencioso Administrativo dice «mediante escrito que se "
              "presente ante la responsable»." + _con_recepcion)

    # ── EL AVISO SE CALIBRA SOLO ──────────────────────────────────────────
    # El viejo —«No hay calendario declarado para X»— salía en las 103 sesiones
    # del acervo, y un aviso que sale siempre no lo lee nadie: se confunde con
    # el fondo. Éste sólo habla donde el dato que falta puede cambiar algo, y
    # dice cuánto haría falta.
    #
    # Y PUEDE CALLARSE CON SEGURIDAD cuando el escrito ya está en tiempo,
    # porque declarar días de la responsable sólo puede ALARGAR el plazo: se
    # añaden inhábiles al calendario, de modo que el vencimiento se mueve hacia
    # adelante o se queda donde está, nunca hacia atrás. Lo oportuno no puede
    # volverse extemporáneo por este dato. Está comprobado sobre las 103
    # sesiones reales, no supuesto.
    if rec.aporta_calendario and not ir.dias:
        if c.oportuna is False:
            k = suspensiones_que_lo_salvarian(c)
            if k:
                avisos.append(
                    f"EXTEMPORÁNEA, PERO NO DECLARASTE NINGÚN DÍA DE LA "
                    f"RESPONSABLE: bastan {k} día(s) hábil(es) de suspensión "
                    f"suya dentro del plazo para que el escrito esté en tiempo "
                    f"({rec.fundamento}). Compruébalo en su acuerdo de "
                    f"suspensión de labores antes de proponer la improcedencia.")
            else:
                avisos.append(
                    "EXTEMPORÁNEA. No declaraste días de la responsable, pero "
                    "ni un periodo vacacional completo suyo dentro del plazo "
                    "cambiaría el resultado.")
        elif presentacion is None or rec.clave == "revision_fiscal":
            # Sin fecha de presentación no hay veredicto que proteger, y en
            # revisión fiscal el considerando SIEMPRE desglosa la ventana del
            # plazo: si falta el calendario del Tribunal, esa ventana sale
            # corta en el papel aunque el sentido no cambie.
            if rec.clave == "revision_fiscal":
                avisos.append(
                    "EL PLAZO SE CONTÓ CON EL CALENDARIO OFICIAL DEL TFJA "
                    "(acuerdos de su Pleno General), no con el del Poder Judicial "
                    "de la Federación. Si la Sala tuvo una suspensión local que no "
                    "esté en ese calendario, decláralo.")
            else:
                avisos.append(
                    f"NO DECLARASTE DÍAS DE LA RESPONSABLE. El plazo se contó sólo "
                    f"con el calendario federal, y el escrito se presenta ante "
                    f"{rec.quien} ({rec.fundamento}): sus vacaciones y suspensiones "
                    f"tampoco se computan. Si las tuvo dentro del plazo, decláralas.")
    elif (not rec.aporta_calendario and c.oportuna is False
            and not any(inicio <= d <= vence for d in _extra)):
        # EL MISMO AVISO, DEL OTRO LADO. En amparo en revisión y en queja lo que
        # importa es el calendario del órgano que RECIBE el escrito, y sus
        # periodos vacacionales sí se descuentan: jurisprudencia PR.A.C.CN. J/7 K
        # (12a.), registro digital 2032618, para los diez días del artículo 86.
        # Las quincenas del Poder Judicial ya están en el calendario; lo que no
        # está es una suspensión extraordinaria de ESE juzgado.
        k = suspensiones_que_lo_salvarian(c)
        if k:
            avisos.append(
                f"EXTEMPORÁNEA. Bastan {k} día(s) hábil(es) en que {rec.quien} "
                f"—ante quien se presenta el escrito, {rec.fundamento}— hubiera "
                f"suspendido labores para que estuviera en tiempo. Si los hubo, "
                f"declárelos en «días inhábiles adicionales».")
    return c


_DIAS_SEMANA = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")


def _rige_procedencia() -> bool:
    """¿Rige `procedencia_por_tipo`? (FIXES_R3: lo que no es [siempre] va tras la
    bandera, y el camino viejo queda idéntico). False si falta el contexto."""
    try:
        import contexto_taller as _ct_f0
        _f = getattr(_ct_f0, "rige", None)
        return bool(_f("procedencia_por_tipo")) if callable(_f) else False
    except Exception:
        return False


def _fechas_entre(a: Fecha, b: Fecha):
    cur = a
    while cur <= b:
        yield cur
        cur += _dt.timedelta(days=1)


# ═══════════════════════════════════════════════════════════════════════════
# Las fechas en letra — obligatorio en documento judicial
# ═══════════════════════════════════════════════════════════════════════════

_UNIDADES = ["", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete",
             "ocho", "nueve", "diez", "once", "doce", "trece", "catorce",
             "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve",
             "veinte", "veintiuno", "veintidós", "veintitrés", "veinticuatro",
             "veinticinco", "veintiséis", "veintisiete", "veintiocho",
             "veintinueve", "treinta", "treinta y uno"]

_MESES = ["", "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
          "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

_DECENAS = {30: "treinta", 40: "cuarenta", 50: "cincuenta", 60: "sesenta",
            70: "setenta", 80: "ochenta", 90: "noventa"}


def _anio_en_letra(a: int) -> str:
    if not 2000 <= a <= 2099:
        return str(a)
    r = a - 2000
    if r == 0:
        return "dos mil"
    if r < 30:
        return f"dos mil {_UNIDADES[r]}"
    d, u = (r // 10) * 10, r % 10
    return f"dos mil {_DECENAS[d]}" + (f" y {_UNIDADES[u]}" if u else "")


def fecha_en_letra(f: Fecha) -> str:
    """23/02/2026 → «veintitrés de febrero de dos mil veintiséis»."""
    return f"{_UNIDADES[f.day]} de {_MESES[f.month]} de {_anio_en_letra(f.year)}"


def lista_en_letra(fechas: Iterable[Fecha]) -> str:
    fs = list(fechas)
    if not fs:
        return ""
    partes = [f"el {_UNIDADES[f.day]} de {_MESES[f.month]}" for f in fs]
    if len(partes) == 1:
        return partes[0]
    return ", ".join(partes[:-1]) + f" y {partes[-1]}"


# ═══════════════════════════════════════════════════════════════════════════
# LA DECISIÓN DEL SECRETARIO, VALIDADA
# ═══════════════════════════════════════════════════════════════════════════
VIAS_DECISION = ("", "oportuna", "reserva")

# Longitud mínima del motivo. No es una cifra estética: el motivo se imprime
# literal en un considerando firmado, y por debajo de esto no cabe una razón
# —cabe una etiqueta—. Medido contra el motivo real que salvaría el 93/2026:
# «la Sala Regional del TFJA suspendió labores el 2 y el 5 de enero de 2026,
# días que el calendario federal tiene por hábiles» son 118 caracteres.
MOTIVO_MINIMO = 40

# Lo que se teclea cuando lo que se quiere es pasar de pantalla. La lista no
# es de palabras prohibidas por gusto: es el acuse de recibo convertido en
# motivo, que es exactamente el clic por inercia con otro disfraz.
_MOTIVO_VACIO = _re.compile(
    r"^(s[ií]|ok|okey|okay|vale|adelante|dale|listo|ya|as[ií] es|correcto|"
    r"proced[ea]|est[aá] bien|porque s[ií]|sin motivo|no aplica|n/?a|test|"
    r"prueba|x+|\W+)[\s.!]*$", _re.I)


def aplicar_decision(c: Computo, via: str = "", motivo: str = "") -> list:
    """Estampa en el cómputo lo que el secretario resolvió — o explica por qué no.

    ═══════════════════════════════════════════════════════════════════════
    DEVUELVE AVISOS Y NO LANZA NUNCA
    ═══════════════════════════════════════════════════════════════════════
    Un 422 en este punto tira la resolución entera, con el estudio de fondo ya
    escrito y ya pagado. La casa ya tiene medido lo que cuesta un error de
    captura convertido en excepción: el «Failed to fetch» del plazo cero.
    Aquí el camino malo es no aplicar la decisión y decirlo a gritos, que deja
    el proyecto exactamente como estaba —improcedencia— y no pierde el trabajo.

    ═══════════════════════════════════════════════════════════════════════
    LA COMPROBACIÓN NO PUEDE ACUSAR AL TRABAJO CORRECTO
    ═══════════════════════════════════════════════════════════════════════
    Por eso la decisión SÓLO se estampa donde hay algo que decidir. Si el
    cómputo ya da oportuna —porque él corrigió las fechas, o declaró los días
    inhábiles que faltaban— la decisión sobra y NO se aplica: se avisa y se
    sigue. Esto no es cortesía, es la defensa contra el peor de los casos que
    este diseño abre: una decisión estampada en la sesión de Supabase el lunes
    sobre unas fechas que el martes ya son otras. El API corre con gunicorn
    -w 2 y la sesión se rehidrata de la base; un indicador rancio que
    sobreviviera a la corrección de las fechas aplicaría una rectificación a
    un cómputo que ya no la necesita, y el considerando saldría explicando por
    qué es oportuno algo que nadie discutía.
    """
    avisos: list = []
    via = (via or "").strip().lower()
    motivo = (motivo or "").strip()

    if via not in VIAS_DECISION:
        return [f"DECISIÓN DE OPORTUNIDAD NO RECONOCIDA: «{via[:40]}». El "
                f"proyecto sigue el cómputo. Las vías son «oportuna» —afirmas "
                f"que se presentó en tiempo— y «reserva» —aceptas la "
                f"extemporaneidad y pides el estudio como anexo de trabajo—."]
    if not via:
        # Y BORRA LA ANTERIOR. Sin esto la decisión es PEGAJOSA: `r` puede venir
        # de `_TALLER_SESIONES` —el mismo worker que atendió la resolución
        # anterior— con `c.decision` ya estampada, y volver a resolver SIN la
        # casilla dejaría el fondo puesto. El formulario es la única fuente de
        # verdad; lo que no venga en él, no existe.
        c.decision, c.motivo = "", ""
        return avisos

    if c.en_cualquier_tiempo:
        return [f"NO HACÍA FALTA DECIDIR NADA: este asunto procede EN "
                f"CUALQUIER TIEMPO, así que no tiene plazo ni puede ser "
                f"extemporáneo. La decisión «{via}» no se aplicó y el proyecto "
                f"entra al fondo como siempre."]
    if c.oportuna is not False:
        return [f"NO SE APLICÓ TU DECISIÓN «{via}» PORQUE EL CÓMPUTO YA DA "
                f"OPORTUNA (vence el {fecha_en_letra(c.vencimiento)}; se "
                f"presentó el {fecha_en_letra(c.presentacion) if c.presentacion else '—'}). "
                f"Si venías de un cómputo extemporáneo y corregiste las fechas "
                f"o declaraste días inhábiles, esto es lo esperado: ya no hay "
                f"nada que rectificar y el proyecto entra al fondo por derecho "
                f"propio. COMPRUEBA que el considerando no invoque una razón "
                f"que ya no hace falta."]

    if len(motivo) < MOTIVO_MINIMO or _MOTIVO_VACIO.match(motivo):
        return [f"NO SE APLICÓ TU DECISIÓN «{via}» PORQUE FALTA EL MOTIVO. "
                f"Ese texto no es un trámite: se imprime LITERAL en "
                + ("el considerando de oportunidad de la ejecutoria que vas a "
                   "firmar" if via == "oportuna" else
                   "la cabecera del anexo de trabajo")
                + f", y por eso tiene que poder leerse solo. Escribe qué hace "
                  f"que el cómputo automático no valga —por ejemplo: «la "
                  f"autoridad responsable suspendió labores del 2 al 5 de "
                  f"enero de 2026, días que el calendario federal tiene por "
                  f"hábiles»—. Mínimo {MOTIVO_MINIMO} caracteres; escribiste "
                  f"{len(motivo)}. El proyecto sale, entretanto, resolviendo "
                  f"la improcedencia."]

    c.decision = via
    c.motivo = motivo
    if via == "oportuna":
        avisos.append(
            "DECLARASTE QUE LA PRESENTACIÓN FUE OPORTUNA pese al cómputo. El "
            "proyecto entra al fondo y el considerando de oportunidad lleva el "
            "desglose completo del cómputo Y tu razón, literal. Lo que firmas "
            "es esa razón: si no se sostiene, el sobreseimiento vuelve en "
            "revisión.")
    else:
        avisos.append(
            "PEDISTE EL ESTUDIO EN RESERVA. La ejecutoria NO cambia: sigue "
            "resolviendo la improcedencia por extemporaneidad, con su "
            "resolutivo. El estudio de fondo va detrás de los puntos "
            "resolutivos, en un anexo rotulado que NO forma parte de la "
            "ejecutoria. Si el Pleno no comparte la extemporaneidad, ese anexo "
            "es el engrose; si la comparte, bórralo antes de listar.")
    return avisos


def conforme_a(fundamento: str) -> str:
    """«conforme AL artículo 86» y «conforme A LOS artículos 61 y 63».

    El catálogo trae unas veces singular y otras plural, y la preposición
    cambia con el número. Estaba escrito «conforme al {fundamento}» en el aviso
    del compositor desde que se escribió —«conforme al artículos 61, fracción
    XIV»— y era inocuo mientras sólo saliera en un aviso. Ahora este texto sale
    EN EL PAPEL, y una concordancia rota en un considerando es de las cosas que
    se ven antes que ninguna otra.
    """
    f = (fundamento or "").lstrip()
    return ("a los " if f.startswith("artículos") else "al ") + f


def _la(escrito: str) -> str:
    """«la demanda de amparo», «el recurso de queja» — para el sujeto.

    `_del` sirve para el complemento («la presentación DE LA demanda») y aquí
    hace falta el sujeto, que lleva otro artículo. Usar `_del` daba «El cómputo
    dice que de la demanda de amparo es extemporánea».
    """
    x = (escrito or "").strip()
    return (f"el {x}" if x.split()[:1] and x.split()[0] in
            ("recurso", "amparo", "juicio") else f"la {x}")


def conflicto_con_la_tarjeta(c: Computo, sentidos, tipo: str = "amparo_directo") -> str:
    """La tarjeta final dice una cosa y el cómputo otra: se nombra el choque.

    ═══════════════════════════════════════════════════════════════════════
    POR QUÉ ESTO Y NO UNA CASILLA EN EL FORMULARIO
    ═══════════════════════════════════════════════════════════════════════
    David: «hoy me generó extemporaneidad en un directo y no me dio el fondo a
    pesar de que en el taller CALIFIQUÉ LOS CONCEPTOS DE VIOLACIÓN… recuerda
    que la tarjeta final gobierna el proyecto».

    Calificar los conceptos ES una decisión sobre la oportunidad, aunque no se
    haya tomado como tal: nadie califica de fundado el concepto de violación de
    una demanda que tiene por extemporánea. Cuando la tarjeta lleva una
    calificación que prospera y el cómputo cierra por extemporaneidad, el
    proyecto tiene DOS desenlaces incompatibles dentro y hoy gana el cómputo en
    silencio. Eso es lo que hay que romper.

    Y ES EL ANTÍDOTO CONTRA EL CLIC POR INERCIA. Una casilla «genera el fondo
    igualmente» viviría permanentemente en la pantalla, se marcaría una vez y
    se quedaría marcada. Esto no existe hasta que hay un choque real: aparece
    con las dos mitades escritas y pidiendo que se resuelva una. Medido sobre
    las 103 sesiones reales del acervo: 3 son extemporáneas y sólo UNA lleva la
    tarjeta empezada —el 93/2026, que es exactamente el asunto que David
    reclama—. No es una pantalla que se vea todos los días; es una que se ve
    cuando hay algo que decidir.
    """
    import tipos_asunto as _ta
    if not c.cierra_por_extemporaneidad:
        return ""
    ss = [str(s or "").strip() for s in (sentidos or []) if str(s or "").strip()]
    prosperan = [s for s in ss if _ta.prospera(s)]
    if not prosperan:
        return ""
    v = _ta.vocabulario_de(tipo)
    _ex = _ta.extemporaneo_de(tipo)
    return (
        f"DOS DESENLACES INCOMPATIBLES EN EL MISMO PROYECTO, Y NO LOS PUEDO "
        f"RESOLVER YO. La tarjeta final califica de «{prosperan[0]}» "
        f"{len(prosperan)} de {len(ss)} {v['combate']}, y eso saca adelante el "
        f"asunto. El cómputo dice que {_la(v['escrito'])} es EXTEMPORÁNEA "
        f"—venció el {fecha_en_letra(c.vencimiento)} y se presentó el "
        f"{fecha_en_letra(c.presentacion) if c.presentacion else '—'}—, y eso "
        f"lo cierra sin fondo. Hoy gana el cómputo y tu calificación se "
        f"descarta en silencio: es lo que hay que romper. DECIDE TÚ, en la "
        f"pantalla de resolución:\n"
        f"  · «la presentación FUE oportuna» + tu razón → el proyecto entra al "
        f"fondo y esa razón se imprime en el considerando de oportunidad.\n"
        f"  · «acepto la extemporaneidad, quiero el estudio en reserva» + tu "
        f"razón → la ejecutoria resuelve la improcedencia conforme "
        f"{conforme_a(_ex['fundamento'])} y el estudio va detrás, en un anexo "
        f"de trabajo.\n"
        f"  · no decides nada → sale lo de hoy: improcedencia, sin fondo.")


def _rectificacion(c: Computo, tipo: str) -> str:
    """El remate del considerando cuando el secretario desestima el cómputo.

    NO INVENTA DERECHO NI LO SUPONE. La razón es suya y se reproduce literal;
    lo único que pone este módulo es el marco de cada vía, que sale del
    catálogo `tipos_asunto.EXTEMPORANEO[...]['rectificado']` — para el amparo
    directo, que no se actualiza la causa de improcedencia del artículo 61,
    fracción XIV, de la Ley de Amparo, cuyo análisis es oficioso conforme al
    artículo 62 del mismo ordenamiento (texto vigente, verificado).
    """
    import tipos_asunto as _ta
    v = _ta.vocabulario_de(tipo)
    rec = _ta.extemporaneo_de(tipo).get("rectificado") or ""
    m = (c.motivo or "").strip().rstrip(".")
    return (f" Ahora bien, ese cómputo debe completarse en los términos "
            f"siguientes: {m}. Por tanto, la presentación {_del(v['escrito'])} "
            f"fue OPORTUNA" + (f" y {rec}" if rec else "."))


# ═══════════════════════════════════════════════════════════════════════════
# El párrafo del considerando
# ═══════════════════════════════════════════════════════════════════════════

_ORDINAL_SURTE = {0: "el mismo día", 1: "al día hábil siguiente",
                  2: "al segundo día hábil siguiente",
                  3: "al tercer día hábil siguiente"}



def _es_decir_surtio(c) -> str:
    """«, es decir, el {día en que surtió}», o nada si surtió EL MISMO DÍA de la
    notificación (E12, cuarta ronda, 3-oct-2026; Q 172 y 24/2026): «surtió
    efectos el mismo día, conforme al artículo 31, fracción III…, es decir, el
    treinta y uno de marzo» repetía la fecha que la frase acababa de dar. Los
    engroses: «surtiendo efectos el mismo día, conforme a lo previsto en el
    artículo 31, fracción III»."""
    if getattr(c, "surtio", None) is not None and c.surtio == getattr(c, "notificacion", None):
        return ""
    return f", es decir, el {fecha_en_letra(c.surtio)}"


# La razón de medir con el depósito postal (D3). Tesis verificada en el engrose
# de la RF 28/2025 (nota 9: registro digital 2025728, Tribunales Colegiados,
# Undécima Época, aislada, Gaceta, libro 21, enero de 2023, tomo VI, p. 6647).
TESIS_DEPOSITO_POSTAL = (
    "la tesis aislada XV.4o.1 A (11a.), de registro digital 2025728, de rubro: «RECURSO "
    "DE REVISIÓN FISCAL. PARA DETERMINAR LA OPORTUNIDAD EN SU PRESENTACIÓN DEBE "
    "CONSIDERARSE LA FECHA DE DEPÓSITO EN LA OFICINA DE CORREOS DE MÉXICO, CONTENIDA EN "
    "LA GUÍA DE MENSAJERÍA CON SELLO ORIGINAL»")
COLA_DEPOSITO_POSTAL = (
    "pues tratándose del recurso interpuesto por correo, su oportunidad se mide con la "
    "fecha del depósito, criterio que recoge " + TESIS_DEPOSITO_POSTAL)


# F1 (quinta ronda, 3-oct-2026): LA FORMA DE NOTIFICACIÓN SIN FUENTE. El
# formulario trae «personal» por omisión (`regla_surtimiento: Form("personal")`)
# y la ficha la convertía en `forma_notificacion` como si la hubiera declarado
# el secretario; el considerando afirmaba «de manera personal» en Q 335, 229,
# 261 y 342/2025 y en AD 274 y 335/2025 sin que ningún papel lo dijera, y con la
# autoridad recurrente, «por oficio» (D2). La ficha marca esas fuentes; con
# ellas el párrafo no afirma la forma y el aviso dice con qué regla se contó.
FUENTES_FORMA_OMISION = ("omision", "omision_autoridad")


def aviso_forma_no_consta(c, tipo: str = "", papel: str = "") -> str:
    """El aviso de F1: «LA FORMA DE NOTIFICACIÓN NO CONSTA (forma_notificacion):
    … el cómputo la contó personal (artículo 31, fracción II…)…». Dice la regla
    con que se contó —la personal de los particulares, el oficio de la
    autoridad (art. 31, fr. I) o, en el amparo directo, la de la ley del acto— y
    pide comprobarla en la constancia. «» con la regla «otra», que declaró el
    secretario. `c` es el cómputo o la regla."""
    regla = getattr(c, "regla", c)
    clave = str(getattr(regla, "clave", "") or "")
    if not clave or clave == "otra":
        return ""
    _f = fundamento_de_surtimiento(regla, tipo, papel)
    como = str(getattr(regla, "descripcion", "") or clave)
    como = "notificación " + (como[len("de manera "):] if como.startswith("de manera ") else como)
    if _f:
        _por = f" ({_f})"
    else:
        import tipos_asunto as _ta_f
        _por = (" (la regla de la ley que rige el acto)"
                if _ta_f.normalizar(tipo) == "amparo_directo" else "")
    return ("LA FORMA DE NOTIFICACIÓN NO CONSTA (forma_notificacion): ningún papel dice cómo "
            f"se notificó y se contó como {como}{_por}. El considerando no la afirma: dice la "
            "fecha y cuándo surtió efectos. Compruébala en la constancia de notificación y, si "
            "fue otra, elige esa regla y vuelve a generar.")


def _del(x: str) -> str:
    """«de la demanda de amparo», «del recurso de queja». La contracción
    depende del sustantivo, y «de el recurso» no es español."""
    x = (x or "").strip()
    return f"del {x}" if x.split()[:1] and x.split()[0] in (
        "recurso", "amparo", "juicio") else f"de la {x}"


def parrafo_oportunidad(c: Computo, fundamento: str = "17",
                        tipo: str = "amparo_directo", desglosar=None,
                        papel: str = "", sin_precepto_en_hueco: bool = False,
                        fundamento_surtimiento: str = "", con_previos: bool = False,
                        forma_consta: bool = True, fuente_forma: str = "") -> str:
    """El párrafo tal como lo escribe el secretario. [… docstring existente …]

    DOS AÑADIDOS, y ninguno toca la aritmética:
      1. Si se descontaron días de la responsable, SE DESGLOSA SIEMPRE. Un
         plazo que se alargó por un dato que no está en ninguna ley publicada
         no puede afirmarse sin enseñarlo.
      2. Esos días llevan SU PROPIO FUNDAMENTO —artículo 176 de la Ley de
         Amparo, o 63 y 74 de la LFPCA—, no el artículo 19. Decir de un día en
         que cerró un tribunal local que fue inhábil «en términos del artículo
         19 de la Ley de Amparo» es un fundamento falso en un considerando
         firmado.

    TRES MÁS (3-oct-2026):
      3. A QUIÉN SE NOTIFICÓ, SIN GÉNERO: «a la parte quejosa», «a la parte
         recurrente», «a la autoridad recurrente» (`a_quien_se_notifico`). Decía
         «se notificó al quejoso» de una quejosa: el género de una persona
         física no se sabe y no se infiere de su nombre.
      4. CADA INHÁBIL CON SU FUNDAMENTO (`clausula_inhabiles`): el artículo 19
         para los suyos; la circular, el lunes de la LFT o las vacaciones
         (art. 226 LOPJF) para los demás.
      5. `fundamento_surtimiento`: el precepto que rige el surtimiento cuando
         la regla no lo trae (amparo directo: la ley del acto). Lo da el
         secretario en la ficha —«el artículo 126 del Código de Procedimientos
         Civiles del Estado de Querétaro»— o una omisión nacional verificada
         (`surtimiento_nacional`); ocupa el sitio del hueco.

    TERCERA RONDA (3-oct-2026):
      6. `con_previos=True` (el camino nuevo: bandera Y ficha, lo decide quien
         compone): la cláusula nombra también los inhábiles entre la
         notificación y el inicio del plazo (`Computo.inhabiles_previos`).
      7. D1: con `sin_precepto_en_hueco` y sin precepto, la cláusula «conforme
         a…» se OMITE (ya no va en hueco).
      8. D3: si el cómputo se midió con el depósito postal (`Computo.deposito`),
         el remate dice «si el oficio se depositó en el Servicio Postal
         Mexicano el…» y la tesis que lo sostiene.

    QUINTA RONDA (3-oct-2026):
      9. F1: LA FORMA DE NOTIFICACIÓN QUE NO CONSTA NO SE AFIRMA. Con
         `forma_consta=False` —o `fuente_forma` «omision» u «omision_autoridad»,
         la que marca la ficha cuando la forma sólo salió del valor por omisión
         del formulario— el párrafo dice «se notificó a la parte X el {fecha} y
         surtió efectos al día hábil siguiente…», sin «de manera personal» ni
         «por oficio», como los engroses del banco (AD 274 y 335/2025; Q 335,
         229, 261 y 342/2025 decían «de manera personal» sin que nada lo
         dijera). El cómputo y el precepto del surtimiento no cambian; el aviso
         lo da `aviso_forma_no_consta`.
     10. «conforme a los artículos 65 y 70…» con `conforme_a`: la regla «lfpca»
         trae el fundamento en plural y salía «conforme al artículos».
    """
    import tipos_asunto as _ta
    if str(fuente_forma or "").strip().lower() in FUENTES_FORMA_OMISION:
        forma_consta = False
    v = _ta.vocabulario_de(tipo)
    _a_quien = a_quien_se_notifico(tipo, papel)
    _por_quien = "por " + _a_quien[2:] if _a_quien.startswith("a ") else _a_quien
    # EL FUNDAMENTO ENTERO, no «el 17». Tres de los cuatro que llaman aquí
    # pasaban el valor por omisión y el párrafo decía «en términos del 17»
    # —así quedó guardado en el estado del 93/2026—. El catálogo sabe cuál es
    # el precepto de cada vía; se le pregunta cuando no se dijo.
    if not fundamento or fundamento.strip() == "17":
        try:
            fundamento = (_ta.plazo_de(tipo, "").get("fundamento")
                          or "artículo 17 de la Ley de Amparo")
        except Exception:
            fundamento = "artículo 17 de la Ley de Amparo"
    # SIEMPRE DESGLOSADO (25-sep-2026). La versión corta —«resultó oportuna, a
    # la luz del artículo 17…»— afirmaba el resultado sin el cómputo ni su
    # fundamento, y David la reclamó con el 93/2026 delante. Queda disponible
    # con `desglosar=False` para quien la pida expresamente.
    if desglosar is None:
        desglosar = True
    if not desglosar:
        if c.anticipada:
            cierre = (f", pues se presentó el {fecha_en_letra(c.presentacion)}, "
                      f"esto es, con anterioridad al inicio del plazo, lo que "
                      f"no le resta oportunidad")
        else:
            cierre = ("" if c.presentacion is None
                      else f", pues se presentó el {fecha_en_letra(c.presentacion)}")
        return (f"Igualmente, la presentación {_del(v['escrito'])} resultó "
                f"oportuna, a la luz del {fundamento}{cierre}.")
    # EL PLAZO DE AÑOS (art. 17, fr. II y III): de fecha a fecha, sin
    # desglose de hábiles, que en ocho años no dice nada.
    if getattr(c, "plazo_anios", 0):
        _n = {7: "siete", 8: "ocho"}.get(c.plazo_anios, str(c.plazo_anios))
        # LA FRACCIÓN DEL 17 QUE DA LOS AÑOS, no el plazo general de quince días.
        fundamento = {8: "artículo 17, fracción II, de la Ley de Amparo",
                      7: "artículo 17, fracción III, de la Ley de Amparo"}.get(
                          c.plazo_anios, fundamento)
        _surte_a = (_ORDINAL_SURTE.get(c.regla.dias_habiles, "al día hábil siguiente")
                    if getattr(c.regla, "clave", "") != "otra" else "")
        p = (f"Por cuanto hace a la oportunidad en la presentación "
             f"{_del(v['escrito'])}, el plazo es de {_n} años, en términos del "
             f"{fundamento}. {v['recurrido'][:1].upper() + v['recurrido'][1:]} "
             f"se notificó {_a_quien} el "
             f"{fecha_en_letra(c.notificacion)}"
             + (f" y surtió efectos {_surte_a}{_es_decir_surtio(c)}" if _surte_a else
                f"; esa notificación surtió efectos el {fecha_en_letra(c.surtio)}")
             + f", por lo que el plazo, computado en años calendario, de fecha a "
               f"fecha y con todos los días naturales —jurisprudencia 1a./J. "
               f"41/2023 (11a.), de registro digital 2026377—, transcurrió del "
               f"{fecha_en_letra(c.inicio)} al {fecha_en_letra(c.vencimiento)}")
        if c.presentacion is None:
            return p + "."
        if c.oportuna:
            return (p + f"; entonces, si se presentó el "
                    f"{fecha_en_letra(c.presentacion)}, es claro que fue "
                    f"hecho valer oportunamente.")
        return (p + f"; entonces, si se presentó el "
                f"{fecha_en_letra(c.presentacion)}, resulta evidente su "
                f"extemporaneidad.")
    # DESDE CUÁNDO CORRE, CON SU PRECEPTO, y los inhábiles como tramos con su
    # fundamento: «ni del dieciséis de diciembre… al uno de enero…» en vez de
    # trece fechas sueltas colgadas de un «así como» sin razón.
    _f_ini = fundamento_de_inicio(tipo)
    _desde_el_siguiente = (f", que corre a partir del día siguiente al en que "
                           f"surtió efectos la notificación, en términos del "
                           f"{_f_ini}," if _f_ini else "")
    # «del cuatro al veinticinco de marzo de dos mil veintiséis», no el año
    # dos veces cuando el plazo no cambia de mes.
    _rango = tramos_en_letra([(c.inicio, c.vencimiento)])
    # «sin contar sábados y domingos, ni …, por ser inhábiles en términos del
    # artículo 19…, ni el dos de mayo…, por ser inhábil conforme a la Circular
    # 1/2025…»: cada tramo con su fundamento.
    _sin_contar = clausula_inhabiles(c, con_previos=con_previos)
    if getattr(c.regla, "clave", "") == "otra":
        # LA REGLA «OTRA»: NO SE AFIRMA UN FUNDAMENTO QUE NO CONSTA. El
        # secretario declaró las dos fechas; el considerando las dice, sin
        # inventarle un «al Nth día hábil» a una regla que no aplicó.
        p = [
            f"Por cuanto hace a la oportunidad en la presentación "
            f"{_del(v['escrito'])}, en términos del {fundamento}, "
            f"{v['recurrido']} se notificó {_a_quien} el "
            f"{fecha_en_letra(c.notificacion)}; esa notificación surtió "
            f"efectos el {fecha_en_letra(c.surtio)}, según lo manifestado "
            f"{_por_quien}, por lo que el plazo para la promoción "
            f"{_del(v['escrito'])}{_desde_el_siguiente} fue {_rango}, {_sin_contar}",
        ]
    else:
        surte = _ORDINAL_SURTE.get(c.regla.dias_habiles, "al día hábil siguiente")
        # EL FUNDAMENTO DE LA REGLA SE ESCRIBE cuando la regla lo trae: «al
        # tercer día hábil siguiente, conforme al artículo 65 de la Ley
        # Federal de Procedimiento Contencioso Administrativo». Un plazo que
        # arranca tres días después de la publicación no se afirma sin decir
        # de dónde sale.
        _f_surte = fundamento_de_surtimiento(c.regla, tipo, papel)
        # «CONFORME A LA LEY DEL ACTO» ES DEL AMPARO DIRECTO. En un recurso lo
        # notificado es una resolución del juicio de amparo y su regla es el
        # 31 de la Ley de Amparo: si no se puede afirmar cuál fracción (la
        # autoridad notificada como particular), hueco y aviso (3-oct-2026).
        _recurso_sin_fund = (not _f_surte and _ta.normalizar(tipo) in ("amparo_revision", "queja")
                             and (papel or "").strip().lower() == "autoridad")
        # Y CON LA PROCEDENCIA POR TIPO (3-oct-2026), TAMPOCO EN EL AMPARO
        # DIRECTO: «conforme a la ley del acto» ocupa el sitio del artículo que
        # no se tiene, y la verja procesal lo acusa como fórmula evasiva. Quien
        # llama lo pide con `sin_precepto_en_hueco`; el aviso nombra el dato.
        # EL PRECEPTO QUE DIO EL SECRETARIO (o la omisión nacional verificada)
        # ocupa el sitio del hueco; nunca el de una regla que ya trae el suyo.
        # D1 (3-oct-2026): SIN PRECEPTO, EN EL CAMINO NUEVO, SE OMITE LA
        # CLÁUSULA —«y surtió efectos al día hábil siguiente, es decir, el…»—,
        # como los engroses de la ponencia de David (AD 274/2025: «surtió efectos
        # al día siguiente, por lo que el plazo…»); ni hueco que el secretario
        # tenga que tocar en casi todos los AD (7 de 8 del banco) ni la fórmula
        # evasiva. El aviso lo recomienda (`aviso_fundamento`). El parámetro
        # conserva su nombre porque así lo pasa `documento_generado`. El recurso
        # de la autoridad notificada como particular sigue en hueco: ahí el
        # precepto que correspondería es otro y afirmar el día sería falso.
        _f_decl = conforme_al_precepto(fundamento_surtimiento)
        # «CONFORME A LOS ARTÍCULOS 65 Y 70» (quinta ronda, 3-oct-2026; la regla
        # «Personal ante el TFJA» en revisión fiscal y en el amparo directo
        # contra el TFJA): el fundamento de la regla viene a veces en plural y
        # la preposición la decide `conforme_a`.
        _fund_regla = (f", conforme {conforme_a(_f_surte)}" if _f_surte
                       else f", {_f_decl}" if (_f_decl and not _recurso_sin_fund)
                       else ", conforme al *********" if _recurso_sin_fund
                       else "" if sin_precepto_en_hueco
                       else ", conforme a la ley del acto")
        # EL 227, FRACCIÓN I, NO DICE CUÁNDO SURTE (revisión de normas, 3-oct-
        # 2026). Dice «Los términos empezarán a correr: I. El día siguiente en
        # que se hubiere hecho el emplazamiento o notificación personal»; en
        # todo el CNPCF ninguna regla dice cuándo surte la personal, y el cero
        # es una inferencia. «Surtió efectos el mismo día, conforme al 227-I» le
        # atribuía a la ley lo que no dice; ahora el considerando dice lo que la
        # ley dice y de ahí saca el día. El precepto de la regla no cambia (lo
        # usan la tabla, los avisos y el desplegable).
        if getattr(c.regla, "clave", "") == "cnpcf_personal" and _f_surte:
            _fund_regla = (f", pues conforme {conforme_a(_f_surte)}, los términos empiezan a "
                           f"correr el día siguiente al de la notificación personal")
        # F1: sin forma que conste, la frase no la afirma.
        _forma = f" {c.regla.descripcion}" if forma_consta else ""
        p = [
            f"Por cuanto hace a la oportunidad en la presentación "
            f"{_del(v['escrito'])}, en términos del {fundamento}, "
            f"{v['recurrido']} se notificó {_a_quien} el "
            f"{fecha_en_letra(c.notificacion)}{_forma} y surtió "
            f"efectos {surte}{_fund_regla}{_es_decir_surtio(c)}, por lo que "
            f"el plazo para la promoción {_del(v['escrito'])}"
            f"{_desde_el_siguiente} fue {_rango}, {_sin_contar}",
        ]

    # ── LOS DÍAS DE LA RESPONSABLE, CON SU PROPIO FUNDAMENTO ──────────────
    _rec = getattr(c, "receptor", None)
    if getattr(c, "resp_en_medio", None) and _rec is not None:
        # SE DICE «LA AUTORIDAD RESPONSABLE», NO SU NOMBRE TECLEADO. El campo
        # `responsable` es texto libre y en el acervo real trae sellos sin
        # identidad —«SALA», «JUZGADO»— y nombres sin artículo, que en esta
        # frase salen como «los días en que Sala Regional suspendió». La
        # carátula y el resolutivo ya la nombran; aquí basta identificarla por
        # su papel, que además es lo que hace la jurisprudencia.
        p.append(
            f"; tampoco se computaron los días en que la autoridad responsable "
            f"suspendió sus labores, esto es, "
            f"{tramos_en_letra(c.resp_tramos_en_medio)}, pues al presentarse "
            f"{_el(v['escrito'])} por su conducto, en términos del "
            f"{_rec.fundamento}, esos días no pueden correr en su perjuicio, "
            f"conforme a {_rec.apoyo}")

    if c.presentacion is not None:
        # UN PÁRRAFO RECTIFICADO NO PUEDE REMATAR EN «EXTEMPORANEIDAD». Éste
        # es exactamente el defecto que ya se corrigió una vez aquí —la frase
        # que afirmaba y negaba— y vuelve por la puerta de la rectificación: si
        # el desglose cierra «resulta evidente su EXTEMPORANEIDAD» y dos
        # líneas más abajo se declara oportuna, el considerando se contradice
        # solo y quien lo lea en sesión lo tumba con razón.
        if c.rectificada:
            veredicto = ("el cómputo que antecede —hecho sobre el calendario "
                         "de días hábiles del artículo 19 de la Ley de Amparo "
                         "y la regla de surtimiento declarada— arrojaría su "
                         "extemporaneidad")
        else:
            veredicto = ("fue hecho valer con anterioridad al inicio del plazo, "
                         "lo que no le resta oportunidad" if c.anticipada
                         else "es claro que fue hecho valer oportunamente" if c.oportuna
                         else "resulta evidente su EXTEMPORANEIDAD")
        _ultimo = (", último día del plazo," if (c.presentacion == c.vencimiento
                                                  and not c.en_cualquier_tiempo) else "")
        _dep = getattr(c, "deposito", None)
        if _dep is not None and _dep == c.presentacion:
            # D3: LA FECHA QUE CUENTA ES LA DEL DEPÓSITO, y se dice cuál es y
            # por qué (RF 28/2025: «si el referido medio de impugnación se
            # depositó en la oficina de Correos de México… es oportuno», con la
            # tesis que se transcribe). La recepción la narra el resultando.
            p.append(f"; entonces, si el oficio se depositó en el Servicio Postal Mexicano "
                     f"el {fecha_en_letra(_dep)}{_ultimo or ','} {veredicto}"
                     + ("" if c.rectificada else f", {COLA_DEPOSITO_POSTAL}") + ".")
        else:
            p.append(f"; entonces, si se presentó el {fecha_en_letra(c.presentacion)}"
                     f"{_ultimo or ','} {veredicto}.")
    else:
        p.append(".")
    if c.rectificada:
        p.append(_rectificacion(c, tipo))
    return "".join(p)


# ═══════════════════════════════════════════════════════════════════════════
# Los calendarios de la síntesis
# ═══════════════════════════════════════════════════════════════════════════

def calendario_mes(anio: int, mes: int, c: Computo) -> list[list[str]]:
    """Rejilla domingo→sábado con el conteo `día/n` en los días del plazo,
    igual que las dos tablas del adelanto."""
    filas: list[list[str]] = [["Domingo", "Lunes", "Martes", "Miércoles",
                               "Jueves", "Viernes", "Sábado"]]
    primero = _dt.date(anio, mes, 1)
    hueco = (primero.weekday() + 1) % 7          # la rejilla arranca en domingo
    siguiente = _dt.date(anio + (mes == 12), (mes % 12) + 1, 1)
    dias_mes = (siguiente - primero).days

    fila = [""] * hueco
    for d in range(1, dias_mes + 1):
        f = _dt.date(anio, mes, d)
        fila.append(f"{d}/{c.dias.index(f) + 1}" if f in c.dias else str(d))
        if len(fila) == 7:
            filas.append(fila)
            fila = []
    if fila:
        filas.append(fila + [""] * (7 - len(fila)))
    return filas


def calendarios_del_plazo(c: Computo) -> list[tuple[str, list[list[str]]]]:
    """Un calendario por mes que toque el plazo, rotulado como en el adelanto
    («FEBRERO 2026», «MARZO 2026»)."""
    meses: list[tuple[int, int]] = []
    cur = c.inicio.replace(day=1)
    fin = c.vencimiento.replace(day=1)
    while cur <= fin:
        meses.append((cur.year, cur.month))
        cur = _dt.date(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1)
    return [(f"{_MESES[m].upper()} {a}", calendario_mes(a, m, c)) for a, m in meses]
