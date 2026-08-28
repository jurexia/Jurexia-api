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
(`scratchpad/auditar_fase0.py`):

    reproduce exacto ....  9   (18%)
    discrepa ............ 40

NO ES APTO PARA FIRMAR TODAVÍA. Sirve para proponer y para revisar; no para
sustituir el cómputo del secretario.

El desfase NO está donde parecía. De los 40 fallos, **32 se van por ±1 día**
—20 con el vencimiento un día tarde y 12 un día pronto—, y sólo 8 tienen
desfases grandes. Un ±1 sistemático no es calendario incompleto: es la REGLA
DE SURTIMIENTO, que cambia según la ley que rige el acto y que aquí se está
adivinando por palabra clave en el texto («boletín», «por lista»).

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

import datetime as _dt
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
CALENDARIO_AMPARO = Calendario(
    nombre="Poder Judicial de la Federación",
    fundamento="artículo 19 de la Ley de Amparo",
    fijos={(1, 1), (1, 5), (5, 5), (16, 9), (12, 10), (25, 12)},
    trasladados={2: 1, 3: 3, 11: 3},
    # SUSPENSIONES DEL ÓRGANO. No salen del art. 19: salen de los acuerdos del
    # CJF y son las que más descuadran el cómputo. La Semana Santa de 2025 se
    # dedujo de los engroses ADA 448-2025 y 449-2025, donde el vencimiento
    # firmado (7 de mayo) sólo se reproduce si la semana del 14 al 18 de abril
    # está entera como inhábil.
    #
    # FALTAN LOS DEMÁS AÑOS Y LOS DOS PERIODOS VACACIONALES ANUALES. Cada uno
    # que se añada hay que comprobarlo contra un engrose firmado, como éste.
    periodos=[(_dt.date(2025, 4, 14), _dt.date(2025, 4, 18))],
)


# ── Calendarios de autoridades responsables ───────────────────────────────
#
# Cada responsable tiene el suyo. Aquí sólo van los verificados; el resto se
# añade conforme se confirmen sus acuerdos de suspensión de labores. Mientras
# una responsable no esté declarada, `computar` avisa y usa el federal, que es
# la aproximación menos mala, PERO deja constancia de que fue una aproximación.
CALENDARIOS_RESPONSABLE: dict[str, Calendario] = {}


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
}


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
    # Avisos que el secretario TIENE que leer antes de firmar.
    avisos: list[str] = field(default_factory=list)

    @property
    def oportuna(self) -> Optional[bool]:
        if self.presentacion is None:
            return None
        return self.inicio <= self.presentacion <= self.vencimiento

    @property
    def dia_de_presentacion(self) -> Optional[int]:
        if self.presentacion is None or self.presentacion not in self.dias:
            return None
        return self.dias.index(self.presentacion) + 1


def computar(
    notificacion: Fecha,
    presentacion: Optional[Fecha] = None,
    regla: str = "tja_qro_boletin",
    plazo: int = 15,
    responsable: Optional[str] = None,
) -> Computo:
    """El cómputo completo, con los dos calendarios.

    `plazo` en días hábiles: 15 para amparo directo (art. 17 LA), 10 para la
    revisión (art. 86), 5 para la queja urgente.
    `responsable` es la clave en CALENDARIOS_RESPONSABLE; si no está declarada
    se usa el federal y queda constancia en `avisos`.
    """
    avisos: list[str] = []

    r = REGLAS_SURTE.get(regla)
    if r is None:
        r = REGLAS_SURTE["personal"]
        avisos.append(
            f"La regla de surtimiento «{regla}» no está declarada. Se contó "
            "como notificación personal. COMPRUEBA la ley que rige el acto "
            "antes de firmar."
        )

    cal_resp = CALENDARIOS_RESPONSABLE.get(responsable or "", CALENDARIO_AMPARO)
    if responsable and responsable not in CALENDARIOS_RESPONSABLE:
        avisos.append(
            f"No hay calendario declarado para «{responsable}». El surtimiento "
            "se calculó con el calendario federal. Si esa autoridad tuvo "
            "vacaciones o suspensión en el periodo, la fecha puede variar."
        )

    # 1) Surtimiento — calendario de la RESPONSABLE
    surtio = notificacion
    contados = 0
    while contados < r.dias_habiles:
        surtio += _dt.timedelta(days=1)
        if cal_resp.es_habil(surtio):
            contados += 1

    # 2) El plazo — calendario del AMPARO
    inicio = CALENDARIO_AMPARO.siguiente_habil(surtio + _dt.timedelta(days=1))
    dias = CALENDARIO_AMPARO.sumar(inicio, plazo)
    vence = dias[-1]

    # Los inhábiles entre semana que caen dentro del plazo: son los que el
    # considerando nombra uno a uno («así como el dieciséis de marzo»).
    enmedio, cur = [], inicio
    while cur <= vence:
        if cur.weekday() < 5 and not CALENDARIO_AMPARO.es_habil(cur):
            enmedio.append(cur)
        cur += _dt.timedelta(days=1)

    return Computo(
        notificacion=notificacion, regla=r, surtio=surtio, inicio=inicio,
        vencimiento=vence, plazo=plazo, dias=dias, presentacion=presentacion,
        inhabiles_en_medio=enmedio, cal_responsable=cal_resp,
        cal_amparo=CALENDARIO_AMPARO, avisos=avisos,
    )


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
# El párrafo del considerando
# ═══════════════════════════════════════════════════════════════════════════

_ORDINAL_SURTE = {0: "el mismo día", 1: "al día hábil siguiente",
                  2: "al segundo día hábil siguiente",
                  3: "al tercer día hábil siguiente"}


def parrafo_oportunidad(c: Computo, fundamento: str = "17") -> str:
    """El párrafo tal como lo escribe el secretario.

    Calcado del engrose ADA 240/2026 y parametrizado. No se «redacta»: se
    rellena, porque su forma es fija y su contenido es aritmético.
    """
    surte = _ORDINAL_SURTE.get(c.regla.dias_habiles, "al día hábil siguiente")
    p = [
        "Igualmente, la presentación de la demanda resultó oportuna, a la luz "
        f"del precepto {fundamento} del mencionado ordenamiento, toda vez que "
        "la sentencia reclamada se notificó a la parte quejosa el "
        f"{fecha_en_letra(c.notificacion)} {c.regla.descripcion} y surtió "
        f"efectos {surte}, es decir, el {fecha_en_letra(c.surtio)}, por lo que "
        "el plazo para la promoción del juicio constitucional fue del "
        f"{fecha_en_letra(c.inicio)} al {fecha_en_letra(c.vencimiento)}, sin "
        f"contar sábados y domingos por ser inhábiles en términos del "
        f"{c.cal_amparo.fundamento}",
    ]
    if c.inhabiles_en_medio:
        p.append(f", así como {lista_en_letra(c.inhabiles_en_medio)} del referido año")
    if c.presentacion is not None:
        veredicto = ("es claro que fue hecho valer oportunamente" if c.oportuna
                     else "resulta evidente su extemporaneidad")
        p.append(f", entonces si se presentó el {fecha_en_letra(c.presentacion)}, "
                 f"{veredicto}.")
    else:
        p.append(".")
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
