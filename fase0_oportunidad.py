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
    # NO HAY PLAZO QUE CONTAR. Los recursos que proceden en cualquier tiempo
    # —peligro de privación de la vida o ataques a la libertad (artículo 17,
    # fracción IV) y omisión de tramitar la demanda de amparo (artículo 98,
    # fracción II)— no tienen vencimiento, y por tanto no pueden ser
    # extemporáneos. Es un CAMPO, no una propiedad, porque quien lo sabe es el
    # catálogo de tipos y tiene que poder decírselo al cómputo.
    sin_plazo: bool = False

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


def tramos_en_letra(tramos) -> str:
    """[(29-jun, 10-jul), (2-ene, 2-ene)] → «del veintinueve de junio al diez de
    julio y el dos de enero». Un tramo de un día se dice como día."""
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
) -> Computo:
    """El cómputo completo, con los dos calendarios.

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
    """
    avisos: list[str] = []

    _sin_plazo = plazo is None or int(plazo) <= 0
    _plazo = 0 if _sin_plazo else int(plazo)
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

    # 1) Surtimiento — calendario de la RESPONSABLE
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

    # Los inhábiles entre semana dentro del plazo, SEPARADOS POR FUNDAMENTO.
    # Un día que ya era inhábil por el artículo 19 se atribuye al artículo 19
    # aunque la responsable también hubiera cerrado: la tesis descuenta esos
    # días «aun cuando la autoridad responsable no haya suspendido labores», y
    # nombrarlo dos veces en el considerando sería un error de bulto.
    _art19 = CALENDARIO_AMPARO
    enmedio, resp_enmedio, cur = [], [], inicio
    while cur <= vence:
        if cur.weekday() < 5 and not cal_amparo.es_habil(cur):
            if not _art19.es_habil(cur) or cur in _extra:
                enmedio.append(cur)
            else:
                resp_enmedio.append(cur)
        cur += _dt.timedelta(days=1)

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
    )

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


def _del(x: str) -> str:
    """«de la demanda de amparo», «del recurso de queja». La contracción
    depende del sustantivo, y «de el recurso» no es español."""
    x = (x or "").strip()
    return f"del {x}" if x.split()[:1] and x.split()[0] in (
        "recurso", "amparo", "juicio") else f"de la {x}"


def parrafo_oportunidad(c: Computo, fundamento: str = "17",
                        tipo: str = "amparo_directo", desglosar=None) -> str:
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
    """
    import tipos_asunto as _ta
    v = _ta.vocabulario_de(tipo)
    if desglosar is None:
        desglosar = ((c.oportuna is False)
                     or _ta.normalizar(tipo) == "revision_fiscal"
                     or bool(getattr(c, "resp_aplicados", False)))
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
    surte = _ORDINAL_SURTE.get(c.regla.dias_habiles, "al día hábil siguiente")
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
        p.append(f", así como {lista_en_letra_con_anio(c.inhabiles_en_medio)}")

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
        p.append(f", entonces si se presentó el {fecha_en_letra(c.presentacion)}, "
                 f"{veredicto}.")
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
