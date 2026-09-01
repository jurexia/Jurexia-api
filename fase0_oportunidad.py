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


CALENDARIO_AMPARO = Calendario(
    nombre="Poder Judicial de la Federación",
    fundamento="artículo 19 de la Ley de Amparo",
    # El art. 19 dice «catorce Y dieciséis de septiembre» — el 14 faltaba.
    fijos={(1, 1), (1, 5), (5, 5), (14, 9), (16, 9), (12, 10), (25, 12)},
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
    # Contencioso administrativo federal. Leída de la nota al pie de
    # «Notificación en Revisión Fiscal.docx» del propio corpus:
    #     «ARTÍCULO 70. Las notificaciones surtirán sus efectos, el día hábil
    #      siguiente a aquél en que fueren hechas.»
    # Comprobada con su ejemplo trabajado: notificación del 2 de septiembre de
    # 2024, surtió el 3, plazo de 15 días vencido el 25. Reproduce al día.
    "lfpca": ReglaSurte(
        clave="lfpca",
        descripcion="conforme a la Ley Federal de Procedimiento Contencioso Administrativo",
        dias_habiles=1,
        fundamento="artículo 70 de la Ley Federal de Procedimiento Contencioso Administrativo",
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
        """
        if self.presentacion is None:
            return None
        return self.presentacion <= self.vencimiento

    @property
    def dia_de_presentacion(self) -> Optional[int]:
        if self.presentacion is None or self.presentacion not in self.dias:
            return None
        return self.dias.index(self.presentacion) + 1


def computar(
    notificacion: Fecha,
    presentacion: Optional[Fecha] = None,
    regla: str = "personal",
    plazo: int = 15,
    responsable: Optional[str] = None,
    inhabiles_extra: Optional[list] = None,
) -> Computo:
    """El cómputo completo, con los dos calendarios.

    `plazo` en días hábiles: 15 para amparo directo (art. 17 LA), 10 para la
    revisión (art. 86), 5 para la queja urgente.
    `responsable` es la clave en CALENDARIOS_RESPONSABLE; si no está declarada
    se usa el federal y queda constancia en `avisos`.

    `inhabiles_extra` son los días que el SECRETARIO declara inhábiles y que el
    calendario federal no trae: un acuerdo de suspensión de labores de su
    tribunal, una contingencia, un día no laborable local. David: «podemos
    darle al secretario la opción de ingresar días inhábiles adicionales para
    considerar en el cómputo».

    Es la pieza que faltaba: el sistema sabe los inhábiles del artículo 19 de la
    Ley de Amparo, los sábados y domingos y los periodos vacacionales del Poder
    Judicial de la Federación, pero NO puede saber que el tribunal de Mérida
    suspendió labores un martes por un huracán. Eso lo sabe quien estuvo ahí, y
    si no se le pregunta el plazo sale corto.
    """
    avisos: list[str] = []

    # LOS INHÁBILES DECLARADOS ENTRAN AL CALENDARIO FEDERAL, que es el que rige
    # el plazo del recurso. Se copia el calendario para no contaminar el global:
    # es un objeto de módulo y mutarlo dejaría esos días inhábiles para todos
    # los asuntos que atendiera este proceso después.
    _extra = {d for d in (inhabiles_extra or []) if d}
    cal_amparo = CALENDARIO_AMPARO
    if _extra:
        import copy as _copy
        cal_amparo = _copy.deepcopy(CALENDARIO_AMPARO)
        cal_amparo.sueltos = set(cal_amparo.sueltos) | _extra

    r = REGLAS_SURTE.get(regla)
    if r and str(getattr(r, "clave", "")).endswith("_qro_boletin"):
        avisos.append(
            "El cómputo usa la regla del Boletín Jurisdiccional del Tribunal de "
            "Justicia Administrativa de QUERÉTARO (surte al tercer día hábil). "
            "Si tu asunto es de otra entidad, comprueba cómo surte efectos la "
            "notificación en la ley que rige el acto: un plazo mal contado "
            "invalida la sentencia.")
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

    # 2) El plazo — calendario del AMPARO, con los inhábiles que el secretario
    #    haya declarado ya dentro.
    inicio = cal_amparo.siguiente_habil(surtio + _dt.timedelta(days=1))
    dias = cal_amparo.sumar(inicio, plazo)
    vence = dias[-1]

    # Los inhábiles entre semana que caen dentro del plazo: son los que el
    # considerando nombra uno a uno («así como el dieciséis de marzo»).
    enmedio, cur = [], inicio
    while cur <= vence:
        if cur.weekday() < 5 and not cal_amparo.es_habil(cur):
            enmedio.append(cur)
        cur += _dt.timedelta(days=1)

    # QUEDA CONSTANCIA DE LO QUE EL SECRETARIO DECLARÓ y de si sirvió de algo:
    # un día declarado fuera del plazo no cambia el cómputo, y decírselo evita
    # que crea que lo tuvo en cuenta.
    if _extra:
        dentro = sorted(d for d in _extra if inicio <= d <= vence)
        if dentro:
            avisos.append(
                f"Se contaron como inhábiles los {len(dentro)} día(s) que "
                f"declaraste dentro del plazo: {lista_en_letra(dentro)}.")
        # UN DÍA ANTERIOR AL INICIO NO ES UN DÍA IGNORADO. Si el secretario
        # declara inhábil el día en que el plazo iba a arrancar, el arranque se
        # corre —y con él el vencimiento—. Decirle «fuera del plazo, no cambia
        # el cómputo» sería falso: cambió el cómputo entero.
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

    return Computo(
        notificacion=notificacion, regla=r, surtio=surtio, inicio=inicio,
        vencimiento=vence, plazo=plazo, dias=dias, presentacion=presentacion,
        inhabiles_en_medio=enmedio, cal_responsable=cal_resp,
        cal_amparo=cal_amparo, avisos=avisos,
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


def _del(x: str) -> str:
    """«de la demanda de amparo», «del recurso de queja». La contracción
    depende del sustantivo, y «de el recurso» no es español."""
    x = (x or "").strip()
    return f"del {x}" if x.split()[:1] and x.split()[0] in (
        "recurso", "amparo", "juicio") else f"de la {x}"


def parrafo_oportunidad(c: Computo, fundamento: str = "17",
                        tipo: str = "amparo_directo", desglosar=None) -> str:
    """El párrafo tal como lo escribe el secretario.

    ═══════════════════════════════════════════════════════════════════════
    Y COMO LO ESCRIBE ÉL ES CORTO, salvo cuando hay algo que demostrar.
    ═══════════════════════════════════════════════════════════════════════
    Esto desglosaba SIEMPRE el cómputo día por día. Contado sobre los adelantos
    reales del corpus, el desglose aparece en UNO de cada cuarenta y cinco:

        amparo directo      desglose  1/45 · «oportuna a la luz del art.» 22/45
        amparo en revisión  desglose  2/45 · «oportuna a la luz del art.» 29/45
        queja               desglose  0/21 · «oportuna a la luz del art.» 16/21

    Lo que el secretario escribe es una declaración con su precepto: «la
    presentación de la demanda resultó oportuna, a la luz del artículo 17 de la
    Ley de Amparo». El cómputo lo ha hecho —de eso depende que el asunto se
    resuelva o se sobresea— pero no lo pone en el papel cuando sale en tiempo,
    porque no hay nada que demostrar.

    Cuando sale EXTEMPORÁNEA sí se desglosa, y ahí la aritmética no es adorno:
    es la prueba de la improcedencia y quien firma tiene que poder recorrerla.

    Y el vocabulario sale del tipo: en una queja no se dice «la sentencia
    reclamada se notificó a la parte quejosa» ni «el juicio constitucional».
    """
    import tipos_asunto as _ta
    v = _ta.vocabulario_de(tipo)
    if desglosar is None:
        # Se desglosa si no está en tiempo, o si el tipo lo acostumbra.
        desglosar = (c.oportuna is False) or _ta.normalizar(tipo) == "revision_fiscal"
    if not desglosar:
        # LO ANTICIPADO SE DICE. Quien firma verá una fecha de presentación
        # anterior al arranque del plazo y, si el papel no lo explica, pensará
        # que hay un error de captura. Se explica en la misma frase.
        if c.anticipada:
            cierre = (f", pues se presentó el {fecha_en_letra(c.presentacion)}, "
                      f"esto es, con anterioridad al inicio del plazo, lo que "
                      f"no le resta oportunidad")
        else:
            cierre = ("" if c.presentacion is None
                      else f", pues se presentó el {fecha_en_letra(c.presentacion)}")
        return (f"Igualmente, la presentación {_del(v['escrito'])} resultó "
                f"oportuna, a la luz del {fundamento}{cierre}.")
    surte = _ORDINAL_SURTE.get(c.regla.dias_habiles, "al día hábil siguiente")
    # EL DESGLOSE NO ADELANTA EL VEREDICTO. Abría con «resultó oportuna» y
    # terminaba, cuando el cómputo no daba, diciendo «resulta evidente su
    # extemporaneidad»: la misma frase afirmaba y negaba. Se abre neutro y el
    # veredicto llega al final, que es donde lo pone el corpus.
    p = [
        f"Por cuanto hace a la oportunidad en la presentación "
        f"{_del(v['escrito'])}, en términos del {fundamento}, "
        f"{v['recurrido']} se notificó al {v['promovente']} el "
        f"{fecha_en_letra(c.notificacion)} {c.regla.descripcion} y surtió "
        f"efectos {surte}, es decir, el {fecha_en_letra(c.surtio)}, por lo que "
        f"el plazo para la promoción {_del(v['escrito'])} fue del "
        f"{fecha_en_letra(c.inicio)} al {fecha_en_letra(c.vencimiento)}, sin "
        f"contar sábados y domingos por ser inhábiles en términos del "
        f"{c.cal_amparo.fundamento}",
    ]
    if c.inhabiles_en_medio:
        p.append(f", así como {lista_en_letra(c.inhabiles_en_medio)} del referido año")
    if c.presentacion is not None:
        veredicto = ("fue hecho valer con anterioridad al inicio del plazo, "
                     "lo que no le resta oportunidad" if c.anticipada
                     else "es claro que fue hecho valer oportunamente" if c.oportuna
                     else "resulta evidente su EXTEMPORANEIDAD")
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
