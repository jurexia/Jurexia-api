"""El circuito completo del adelanto: de dos PDF a un .docx en la mano.

Encadena lo que ya existe por separado:

    Fase 0   fase0_oportunidad.py   ficha y cómputo — SIN modelo, es aritmética
    Fase 1-3 fases123_pipeline.py   los dos resúmenes y los problemas jurídicos
    Fase 7   ensamblar_adelanto.py  relleno de la plantilla REAL del secretario

Y se detiene donde tiene que detenerse: el sentido del fallo y el criterio no
los pone la máquina. El .docx sale con esos huecos marcados, igual que el
adelanto de papel, y `huecos_pendientes()` los enumera para que nadie firme un
documento con un `*****` dentro.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import re
from dataclasses import dataclass, field
from typing import Optional

import ensamblar_adelanto as ens
import fase0_oportunidad as f0
import promovente as _pv
import fase_partes as fpartes


def _parece_autoridad(nombre: str) -> bool:
    """¿Este nombre es un órgano del Estado y no un particular? Se mira el
    vocabulario del cargo, no la forma jurídica: una sociedad anónima nunca
    se llama «Titular de…» ni «Director General de…»."""
    n = (nombre or "").lower()
    # UNA SOCIEDAD NO ES AUTORIDAD aunque se llame «Administradora…» (3-oct-2026).
    if re.search(r"\bs\.?\s*a\.?\b|\bs\.?\s*de\s*r\.?\s*l\.?|\ba\.?\s*c\.?\b|\bs\.?\s*c\.?\b|\bsapi\b|\bsas\b", n):
        return False
    return any(k in n for k in (
        # LAS UNIDADES DEL SAT (3-oct-2026): «Administración Desconcentrada
        # Jurídica…» recurre como autoridad y no se reconocía.
        "administración desconcentrada", "administracion desconcentrada",
        "administración local", "administración general", "administración central",
        "subadministración", "subadministracion",
        "titular", "director", "directora", "unidad de", "secretaría", "secretaria de",
        "juzgado", "tribunal", "sala ", "magistrad", "juez", "ayuntamiento", "instituto",
        "comisión", "comision", "fiscal", "procurad", "presidente municipal", "gobernador",
        "congreso", "servicio de administración", "delegación", "subsecretar", "jefe de",
        "coordinador", "administrador", "autoridad", "consejo de la judicatura"))


def _mismo_nombre(a: str, b: str) -> bool:
    import unicodedata as _u
    def _n(x):
        x = _u.normalize("NFD", (x or "").lower())
        x = "".join(c for c in x if _u.category(c) != "Mn")
        return " ".join(x.replace(",", " ").split())
    _a, _b = _n(a), _n(b)
    return bool(_a and _b) and (_a == _b or _a in _b or _b in _a)


def _misma_parte(a: str, b: str) -> bool:
    """La misma parte escrita de dos maneras (AR 631/2025, 28-sep-2026): ver
    `promovente.misma_parte`; y, como antes, igualdad o una dentro de otra."""
    return _mismo_nombre(a, b) or _pv.misma_parte(a, b)


def _una_edicion(x: str, y: str) -> bool:
    """¿A lo sumo una letra cambiada, puesta o quitada? («gomes»/«gomez»)."""
    if abs(len(x) - len(y)) > 1:
        return False
    if len(x) == len(y):
        return sum(1 for p, q in zip(x, y) if p != q) <= 1
    corta, larga = (x, y) if len(x) < len(y) else (y, x)
    for i in range(len(larga)):
        if larga[:i] + larga[i + 1:] == corta:
            return True
    return False


def _palabra_casa(p: str, w: str) -> bool:
    """«ma» y «maria» (abreviatura: las letras de la corta, en orden, al
    principio de la larga), «gomes» y «gomez» (una edición en palabras de
    cuatro letras o más), o iguales."""
    if p == w:
        return True
    corta, larga = (p, w) if len(p) <= len(w) else (w, p)
    if len(corta) <= 3 and len(larga) > len(corta) and larga[0] == corta[0]:
        it = iter(larga)
        return all(ch in it for ch in corta)
    return len(corta) >= 4 and _una_edicion(p, w)


def _parte_parecida(a: str, b: str) -> bool:
    """La misma parte escrita con una abreviatura o una errata (revisión del
    28-sep-2026, regresión de SPEC B): «María de la Luz Hernández Ruiz» y «Ma.
    de la Luz Hernández Ruiz», «Gómez» y «Gomes». `_misma_parte` exige las
    palabras idénticas y, cuando la quejosa recurrente tecleaba su nombre con
    otra grafía que el resolutivo, se la tomaba por OTRA parte —«tercero»— y se
    invertían la legitimación, el pro persona y la fracción del 93. Aquí cada
    palabra que nombra en el nombre más corto tiene que casar con una del otro
    (`_palabra_casa`), con dos como mínimo: «Inmobiliaria Hernández Ruiz» no
    es «María Hernández Ruiz»."""
    if _misma_parte(a, b):
        return True
    pa, pb = _pv.palabras_que_nombran(a), _pv.palabras_que_nombran(b)
    if len(pa) < 2 or len(pb) < 2:
        return False
    corta, larga = (pa, pb) if len(pa) <= len(pb) else (pb, pa)
    libres = list(larga)
    for p in corta:
        j = next((k for k, w in enumerate(libres) if _palabra_casa(p, w)), None)
        if j is None:
            return False
        libres.pop(j)
    return True
import fase6_estudio as f6
import fase6_rag as f6rag
import marco_juridico as mjur
import fases123_pipeline as f123


@dataclass
class Encargo:
    """Lo que el secretario aporta. Todo esto lo sabe de memoria o lo lee de un
    sello; nada de esto justifica pagar el OCR de un expediente."""
    numero: str                       # «512/2026»
    encabezado: str                   # «AMPARO DIRECTO ADMINISTRATIVO: 512/2026»
    quejoso: str
    magistrado: str
    secretario: str
    notificacion: _dt.date
    presentacion: _dt.date
    # LA OMISIÓN ES LA REGLA GENERAL DE LA LEY DE AMPARO, no la de un
    # tribunal concreto: esto lo usan secretarios de toda la república.
    regla_surtimiento: str = "personal"
    # CUANDO regla_surtimiento ES «otra»: la fecha, ISO, en que el propio
    # secretario declara que la notificación surtió efectos — sin que el
    # sistema le aplique la regla de un tribunal que no es el suyo. Vacía en
    # cualquier otro caso.
    surte_efectos: str = ""
    # EL PLAZO LO PONE LA LEY SEGÚN EL TIPO. Cero significa «el que
    # corresponda»; se resuelve al computar, con `tipos_asunto`.
    plazo: int = 0
    # La excepción de plazo que el secretario haya declarado: en la queja,
    # «suspension» (dos días) u «omision_tramite» (en cualquier tiempo).
    excepcion_plazo: str = ""
    # Los días que el secretario declara inhábiles y el calendario federal no
    # trae: un acuerdo de suspensión de labores de su tribunal, una
    # contingencia, un no laborable local.
    dias_inhabiles_extra: list = field(default_factory=list)
    # ═══════════════════════════════════════════════════════════════════════
    # LOS DÍAS EN QUE LA RESPONSABLE NO LABORÓ
    # ═══════════════════════════════════════════════════════════════════════
    # En amparo directo la demanda se presenta ANTE LA RESPONSABLE (artículo
    # 176 de la Ley de Amparo), y por eso del plazo se excluyen DOS listas que
    # se suman: los inhábiles del artículo 19 y los días en que esa autoridad
    # suspendió actividades —P./J. 4/2022 (11a.), registro digital 2024494—.
    # Lo mismo vale en la revisión fiscal, donde el escrito se presenta ante la
    # Sala. En el amparo en revisión y en la queja NO, porque ahí se presenta
    # ante órgano federal.
    #
    # Este dato no lo puede saber el sistema: lo declara quien lo sabe. Viaja
    # como texto de tramos —«2025-12-16..2026-01-05, 2026-02-12»— porque un
    # periodo vacacional es un tramo y una suspensión suelta es un tramo de un
    # día, y así una sola línea cubre los dos casos.
    inhabiles_responsable: str = ""
    # LAS DOS FECHAS DE LA SESIÓN, que el secretario sí sabe y el sistema no
    # puede deducir de ningún documento: el proyecto se escribe ANTES de que la
    # sesión ocurra. Salían como dos comodines de asteriscos en los cinco
    # proyectos —el hallazgo más repetido de la última medición—, y era un
    # hueco honesto pero evitable: basta preguntarlo. Vacías = siguen en hueco,
    # que es mejor que inventarlas. ISO.
    fecha_lista: str = ""
    fecha_sesion: str = ""
    responsable: Optional[str] = None
    es_recurso: bool = False
    # ═══ QUIÉN RECURRE NO ES QUIÉN PIDIÓ EL AMPARO (23-sep-2026) ═══════════
    # Amparo en revisión 711/2025: el amparo lo promovió Interamericana de
    # Aceites y Lubricantes y lo GANÓ; quien recurrió fue la Unidad de
    # Inteligencia Financiera. El formulario de admisión guardó a la UIF en
    # `quejoso` —porque es «quien promueve el recurso»— y el resolutivo salió
    # «La Justicia de la Unión no ampara ni protege a [la UIF]». David: «es
    # absurdo e incongruente. Se niega el amparo a la persona moral».
    #
    # El amparo se concede o se niega a quien lo PIDIÓ; el recurso se declara
    # fundado o infundado a quien lo INTERPUSO. Son dos papeles, y coinciden
    # sólo cuando recurre el propio quejoso. Vacío = coinciden.
    recurrente: str = ""
    # ═══ EL CARÁCTER CON QUE RECURRE, CUANDO CONSTA EN EL ESCRITO (3-oct-2026) ═
    # AR 631/2025 de punta a punta: el formulario no dijo quién recurría y el
    # proyecto salió con «QUEJOSA Y RECURRENTE: Unión de Trabajadores…» y «en su
    # carácter de parte quejosa», cuando el escrito de revisión lo firmaba la
    # adquirente del inmueble, «hoy TERCERO INTERESADO». El escrito lo dice; se
    # lee (`ficha_tramite.leer_escrito`) y el carácter queda FIJO aquí:
    # «quejoso» | «autoridad» | «tercero». `papel_del_recurrente` lo respeta
    # en vez de deducirlo del vocabulario del nombre —un «Síndico Municipal» no
    # casa con ninguna palabra de cargo y salía «tercero»—. Vacío = no consta y
    # se deduce como siempre. Viaja en la sesión: el otro worker ve el mismo.
    papel_recurrente: str = ""
    # LA HERRAMIENTA NO ES DE UN TRIBUNAL, ES DE TODOS. Estos tres campos son
    # lo que impide que un secretario de otro circuito firme «Resolución del
    # Tercer Tribunal Colegiado… del Vigésimo Segundo Circuito» sin verlo: en
    # modo `generado` el documento se escribe entero con ESTOS datos y no hay
    # plantilla ajena de la que heredar identidad.
    # El tipo decide el ESQUELETO del documento: los recursos no llevan
    # «Existencia del acto reclamado», la queja hace el cómputo en prosa y cada
    # uno rotula la dispensa a su manera. Medido por tipo, no supuesto.
    tipo_asunto: str = "amparo_directo"
    tribunal: str = ""                # «Primer Tribunal Colegiado… del Décimo…»
    ciudad: str = ""                  # «Mérida, Yucatán»
    modo: str = "generado"            # plantilla | generado
    plantilla: str = ""               # el .docx del propio tribunal
    coleccion_estatal: str = ""       # «leyes_queretaro», para el RAG del fondo
    # LA MATERIA, DECLARADA. Decide el silo del RAG y el filtro del sondeo de
    # precedentes. Vacía = se deduce del tribunal y del encabezado, como antes.
    materia: str = ""
    # LOS CONCEPTOS DE VIOLACIÓN, cuando el recurso va a levantar un
    # sobreseimiento. Entonces el colegiado asume jurisdicción y tiene que
    # estudiarlos por primera vez —artículo 93, fracción I— y NO CONSTAN en el
    # expediente del recurso: sólo aparecen si la sentencia recurrida los
    # relató. Los aporta el secretario.
    conceptos_violacion: str = ""
    # LA FORMA DE LA SENTENCIA: «estandar» (concepto por concepto, extensión de
    # siempre) o «moderna» (pregunta y respuesta, condensada). La fija cada
    # generación desde la pantalla; vacío es la estándar. Ver
    # `formato_sentencia.py`.
    formato: str = ""
    # LA VARIANTE DEL PROMPT DEL ESTUDIO —«v1» o «v2»—, fijada en CADA
    # petición por los dos endpoints del taller: la global de ESTUDIO_PROMPT,
    # o la que pida una cuenta de casa. Vacío = la global. Ver
    # `fase6_estudio.VARIANTES` (26-sep-2026).
    variante_estudio: str = ""
    # LA SUPLENCIA DE LA QUEJA, COMO LA DECIDIÓ EL SECRETARIO en la pantalla de
    # decisión (David, 26-sep-2026): {fraccion, a_favor_de, confirmada}. La fija
    # cada generación, como la forma: no se guarda con la sesión, viaja en el
    # formulario. Vacío = no decidió nada y el estudio se comporta como antes.
    # Ver `suplencia.py`.
    suplencia: dict = field(default_factory=dict)
    # EL PLAN DEL ESTUDIO Y SU GUION (v4, Paso 2, 26-sep-2026). Los fija en
    # CADA petición `main._taller_plan_para` —vacíos si la variante no es la
    # v4 o si el plan no llegó—, por la misma razón que la forma: el encargo
    # vive en la memoria del worker y un plan de la vuelta anterior no puede
    # colarse en ésta. El guion llega al prompt como argumento de
    # `prompt_estudio`; el plan, a la ficha (pestaña «Mapa del estudio»).
    plan: dict = field(default_factory=dict)
    guion: str = ""
    # LA LECTURA DEL ESCRITO PARA EL INVENTARIO (v3/v4, 26-sep-2026): los
    # argumentos que un modelo leyó en el escrito y el código verificó
    # (`inventario_escrito`). La fija en CADA petición
    # `main._taller_inventario_al_encargo` —vacía fuera de la v3/v4 o si no
    # llegó—, antes del plan; `_formato_al_material` la funde con el piso.
    inventario_escrito: list = field(default_factory=list)
    # ═══ LA FICHA DE TRÁMITE (3-oct-2026, bandera `procedencia_por_tipo`) ══
    # Los HECHOS procesales del asunto —auto de Presidencia, turno, returno,
    # Ministerio Público, adhesivo, el acto con su fecha, órgano, toca y
    # expediente— cada uno con su FUENTE (`ficha_tramite.armar`). De ella
    # componen el V I S T O y los resultandos `resultandos_por_tipo.componer`
    # y los considerandos procesales `documento_generado.componer`, sin leer
    # con regex la prosa de un modelo. VIAJA EN LA SESIÓN (`estado.encargo.
    # tramite`): con -w 2 el worker que resuelve no es el que leyó el acto, y
    # sin ella el proyecto volvería a la llamada a ciegas. None = no se armó
    # (bandera apagada o sesión anterior): el camino de siempre.
    tramite: Optional[dict] = None
    # LO QUE ENTRA A LA FICHA ANTES DEL ADELANTO, puesto por `main` en CADA
    # petición: lo leído del auto de admisión/turno (`ficha_tramite.leer_auto`,
    # fuente «auto») y lo que el secretario confirmó en el formulario
    # (`ficha_tramite.de_formulario(tramite_json)`, fuente «secretario», que
    # manda). No se guardan con la sesión: ya quedaron fundidos en `tramite`.
    tramite_auto: dict = field(default_factory=dict)
    tramite_declarado: dict = field(default_factory=dict)


@dataclass
class Resultado:
    ruta: str
    computo: f0.Computo
    fases: f123.Fases123
    huecos: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)
    # Se guardan para poder REENSAMBLAR cuando llegue el criterio, sin releer
    # los PDF ni volver a pagar los resúmenes.
    encargo: Optional["Encargo"] = None
    estudio: str = ""
    advertencias: str = ""
    partes: Optional["fpartes.Partes"] = None
    # Las partes estructurales ya escritas, para no volver a pedirlas.
    estructura: object = None
    # CON QUÉ SE ESCRIBIÓ EL ESTUDIO: la variante del prompt, el
    # `finish_reason` y el uso de tokens (26-sep-2026). Lo llenan los dos
    # redactores y lo lee la ficha del proyecto: sin esto no hay manera de
    # saber, mirando un proyecto viejo, con qué prompt salió ni si se cortó.
    meta_estudio: dict = field(default_factory=dict)


    @property
    def listo_para_el_secretario(self) -> bool:
        """El documento está completo HASTA donde puede estarlo sin criterio."""
        return not self.avisos


# ═══ EL RELOJ ═════════════════════════════════════════════════════════════
# David, 30-ago-2026: «¿cómo aceleramos la generación de la sentencia sin
# perder calidad? mide cuánto tarda». Lo primero es medir por fase: sin eso se
# optimiza lo que se supone lento, que casi nunca es lo que lo es.
import time as _time
from contextlib import contextmanager

TIEMPOS: dict = {}


@contextmanager
def cronometrar(nombre: str):
    t0 = _time.perf_counter()
    try:
        yield
    finally:
        dt = _time.perf_counter() - t0
        TIEMPOS[nombre] = round(dt, 1)
        print(f"   ⏱️  {nombre}: {dt:.1f}s")


def reloj_resumen(total: float = 0.0) -> str:
    if not TIEMPOS:
        return ""
    partes = " · ".join(f"{k} {v}s" for k, v in TIEMPOS.items())
    return f"{partes}" + (f" · TOTAL {total:.1f}s" if total else "")


def _registros_ya_estan(aviso: str, material) -> bool:
    """¿Los registros que denuncia ese aviso están YA en el material?

    El aviso lo escribe `fase6_estudio.revisar`, que corre antes de completar
    las tesis citadas. Si después se trajeron del acervo, el aviso miente.
    """
    _rs = re.findall(r"\d{6,7}", aviso or "")
    if not _rs:
        return False
    _hay = {str(t.get("registro", "")) for t in (getattr(material, "tesis", None) or [])}
    return all(_r in _hay for _r in _rs)


# El aviso del cómputo que da fuera de plazo. Una sola redacción: lo escribe
# el primer cómputo y, si recurre una autoridad, el que se rehace con su regla.
AVISO_EXTEMPORANEA = ("EL CÓMPUTO DA EXTEMPORÁNEA. Compruébalo antes de seguir: "
                      "si es correcto, el asunto no se resuelve en el fondo.")


async def generar(cliente, e: Encargo, texto_acto: str, texto_conceptos: str,
                  ruta_salida: str, texto_autos: str = "") -> Resultado:
    """El circuito entero. `cliente` es el AsyncOpenAI de main.py.

    `texto_autos` son las constancias del expediente, si el secretario las
    subió. NO son fuente de derecho y no se confunden con el acervo: son los
    documentos del caso —contrato colectivo, reglamento, peritajes— que no
    están en ninguna base pública y sin los cuales hay asuntos que no se pueden
    resolver. En el ADL 382/2024 el motor dijo «falta el texto de las cláusulas
    40 y 83» y ese texto estaba en un PDF que el secretario había subido y que
    el pipeline no leía.
    """
    avisos: list[str] = []

    # ── Fase 0 — aritmética, sin modelo ──────────────────────────────────
    # LA REGLA POR DEFECTO NO PUEDE SER LA DE OTRA MATERIA. `tja_qro_boletin`
    # —Boletín Jurisdiccional, surte al tercer día hábil— es del Tribunal de
    # Justicia Administrativa de Querétaro, y por ser el valor por omisión se
    # aplicó a un amparo LABORAL contra un laudo de la Junta Federal. Los laudos
    # se notifican PERSONALMENTE (artículo 742 de la Ley Federal del Trabajo) y
    # el propio proyecto se contradecía: los antecedentes decían que el actuario
    # notificó en persona y el cómputo hablaba de boletín.
    #
    # Un plazo mal contado invalida la sentencia, así que aquí no se hereda un
    # valor por omisión de otra materia: si la materia es laboral y nadie
    # declaró otra cosa, se cuenta personal y SE AVISA.
    # ── El plazo y el tipo, del catálogo ─────────────────────────────────
    import tipos_asunto as _ta
    _tipo = _ta.normalizar(getattr(e, "tipo_asunto", "")) or "amparo_directo"
    e.tipo_asunto = _tipo
    # `es_recurso` DEJA DE SER UN CAMPO APARTE. Era independiente del tipo y
    # podía contradecirlo —un amparo directo marcado como recurso escribía
    # «agravios» donde van conceptos de violación—. Lo dice el tipo y punto.
    e.es_recurso = _tipo != "amparo_directo"
    _pl = _ta.plazo_de(_tipo, getattr(e, "excepcion_plazo", ""))
    if _pl.get("aviso"):
        avisos.append(_pl["aviso"])
    # LOS PLAZOS DE AÑOS NO SE CUENTAN EN DÍAS HÁBILES (verificación de normas,
    # 27-sep-2026): el artículo 17 da «hasta ocho años» (fracción II) y «siete
    # años» (fracción III), y sumarlos como 2,920 y 2,555 hábiles daba unos once
    # y diez años. El cómputo los cuenta de fecha a fecha (`plazo_anios`).
    _anios_plazo = _ta.anios_de(_tipo, getattr(e, "excepcion_plazo", ""))
    if _pl["en_cualquier_tiempo"]:
        # NO ES UN PLAZO LARGO: ES QUE NO HAY PLAZO. Contar días aquí y
        # declarar extemporaneidad sería inventar una causa de improcedencia.
        e.plazo = 0
        avisos.append(
            f"Este recurso procede EN CUALQUIER TIEMPO ({_pl['fundamento']}), "
            f"así que no se computa plazo ni puede declararse extemporáneo.")
    elif not e.plazo:
        e.plazo = _pl["dias"]

    _mat = fp_materia(e)
    if _mat == "laboral" and e.regla_surtimiento in ("tja_qro_boletin", "lista"):
        e.regla_surtimiento = "personal"
        avisos.append(
            "El cómputo se hizo con notificación PERSONAL, que es como se "
            "notifican los laudos (artículo 742 de la Ley Federal del "
            "Trabajo). Venía declarada la regla del Boletín Jurisdiccional del "
            "Tribunal de Justicia Administrativa de Querétaro, que es de otra "
            "materia. Confírmalo contra la constancia de notificación.")
    # LA REGLA DE QUERÉTARO SÓLO VALE PARA ASUNTOS DE QUERÉTARO. Es la misma
    # trampa que la de arriba, por otra puerta: `tja_qro_boletin` es del
    # Tribunal de Justicia Administrativa DE QUERÉTARO, y nada impedía que se
    # aplicara a un amparo directo administrativo de cualquier otro estado —el
    # redactor sirve a toda la república, no a un solo tribunal. Se exige que
    # la colección estatal declarada sea la de Querétaro; si no, se cuenta
    # personal y se avisa, igual que con laboral.
    elif _mat == "administrativa" and e.regla_surtimiento == "tja_qro_boletin":
        _col = (getattr(e, "coleccion_estatal", "") or "").strip().lower()
        _resp_notif = getattr(e, "responsable", "") or ""
        # «DE QUERÉTARO» NO ES «CON SEDE EN QUERÉTARO». El Tribunal de Justicia
        # Administrativa DEL ESTADO de Querétaro (TJA, la regla de este boletín)
        # y el Tribunal FEDERAL de Justicia Administrativa (TFJA) son dos
        # instituciones distintas, y el TFJA tiene una Sala Regional QUE SE
        # LLAMA «en Querétaro» sin ser de Querétaro: es federal, resuelve
        # materia fiscal de toda su zona, y su notificación no se rige por el
        # Boletín Jurisdiccional del tribunal estatal. En la revisión fiscal
        # 8/2026 (responsable «Sala Regional en Querétaro del Tribunal FEDERAL
        # de Justicia Administrativa»), `coleccion_estatal` decía
        # «leyes_queretaro» —correcto para el RAG del fondo, que sí puede
        # necesitar ley local— y el guardián de abajo lo daba por bueno para
        # la regla de notificación, que es una pregunta distinta.
        _es_federal = "federal" in _resp_notif.lower()
        if _es_federal and f0.fuero_de(getattr(e, "tipo_asunto", ""), _resp_notif) == "tfja":
            # ES EL TFJA: su boletín también surte al tercer día hábil, pero
            # por el artículo 65 de la LFPCA. Se cambia a ESA regla, no a la
            # personal: contar personal aquí adelanta el surtimiento dos
            # días y puede volver extemporáneo un escrito en tiempo.
            e.regla_surtimiento = "lfpca_boletin"
            avisos.append(
                "El cómputo se hizo con el BOLETÍN JURISDICCIONAL DEL TFJA "
                "(artículo 65 de la Ley Federal de Procedimiento Contencioso "
                "Administrativo: surte al tercer día hábil). Venía declarada la "
                "regla del boletín del Tribunal de Justicia Administrativa DEL "
                f"ESTADO DE QUERÉTARO, y la responsable, «{_resp_notif}», es el "
                "tribunal FEDERAL. Si la notificación fue personal, elige esa "
                "regla.")
        elif "queretaro" not in _col.replace("é", "e") or _es_federal:
            e.regla_surtimiento = "personal"
            avisos.append(
                "El cómputo se hizo con notificación PERSONAL (artículo 31, "
                "fracción I, de la Ley de Amparo). Venía declarada la regla "
                "del Boletín Jurisdiccional del Tribunal de Justicia "
                "Administrativa DEL ESTADO DE QUERÉTARO, que sólo rige ahí"
                + (f" —y la responsable, «{_resp_notif}», es un órgano FEDERAL, "
                   "no el tribunal estatal de Querétaro" if _es_federal else
                   ", y este asunto no está declarado como de Querétaro")
                + ". Si de verdad se rige por ese boletín, elige la entidad "
                "correcta; si no, comprueba en la ley que rige el acto cómo "
                "surte efectos la notificación —o usa «Otra regla» y declara "
                "tú las dos fechas.")
    # LO AGRARIO EN LA CIUDAD DE MÉXICO SE CUENTA CON EL CÓDIGO NACIONAL
    # (3-oct-2026). David: «Si va con el Código Nacional, y no el Federal,
    # entonces hay que adecuar al Código Nacional». El artículo 167 de la Ley
    # Agraria (DOF 14-11-2025) remite al Código Nacional, que ya opera en la
    # sede de la Ciudad de México: su notificación personal surte EL MISMO DÍA
    # (art. 227, fr. I), no al siguiente, y el plazo arranca un día hábil antes.
    # La «personal» o la «lista» que llegan por omisión se cambian a las suyas
    # y se avisa; lo elegido a propósito (otra, oficio, electrónica, lfpca…) se
    # respeta. Fuera de la Ciudad de México no se toca: el considerando la funda
    # en el 321 del CFPC (`f0.surtimiento_nacional`), con el aviso del
    # transitorio.
    _reg_agr, _av_agr = regla_agraria(e, _mat)
    if _reg_agr:
        e.regla_surtimiento = _reg_agr
        avisos.append(_av_agr)
    # EL CERO VIAJA HASTA EL CÓMPUTO. `e.plazo or 15` lo convertía en quince
    # días: el «en cualquier tiempo» que se acababa de declarar se perdía en el
    # camino, y por eso hacía falta corregirlo después escribiendo sobre
    # `c.oportuna` —una @property de sólo lectura—, que reventaba con
    # AttributeError. Se pasa el plazo que es, y el cómputo sabe qué hacer con
    # el cero: `sin_plazo`, sin vencimiento y sin extemporaneidad posible.
    _plazo_computo = 0 if _pl["en_cualquier_tiempo"] else (e.plazo or 15)
    # LA REGLA «OTRA»: el secretario ya declaró las dos fechas —cuándo se
    # notificó (e.notificacion, de siempre) y cuándo surtió efectos— y el
    # cómputo no tiene que adivinar ni aplicar ninguna regla ajena.
    # UNA SOLA PUERTA para leerla: la misma que usa la reconstrucción de la
    # sesión desde la base. Hasta hoy ésta la leía y aquélla no.
    _surtio_manual, _av_surtio = f0.surtio_manual_de(e)
    if _av_surtio:
        avisos.append(_av_surtio)

    # UNA SOLA LLAMADA, CON LA REGLA COMO ÚNICA VARIABLE: si más abajo se sabe
    # que recurre una autoridad (`regla_de_la_autoridad`), se recomputa con
    # los mismos datos y sólo otra regla de surtimiento (3-oct-2026).
    # `deposito` (3-oct-2026, D3): la revisión fiscal por correo se mide con la
    # fecha en que el oficio se depositó en el Servicio Postal Mexicano
    # (`deposito_que_cuenta`); la misma puerta que usa la sesión
    # (`args_de_presentacion`), para que los dos workers cuenten igual.
    def _computar_con(regla, deposito=None):
        _pres_c, _kw_dep = args_de_presentacion(e, deposito)
        _c = f0.computar(e.notificacion, _pres_c, regla,
                         _plazo_computo, e.responsable,
                         getattr(e, "dias_inhabiles_extra", None),
                         # EL TIPO DECIDE SI EL DESCUENTO DE LA RESPONSABLE APLICA:
                         # sólo donde el escrito se presenta ante ella.
                         getattr(e, "tipo_asunto", "") or "amparo_directo",
                         getattr(e, "inhabiles_responsable", "") or None,
                         surtio_manual=_surtio_manual,
                         plazo_anios=_anios_plazo, **_kw_dep)
        return marcar_deposito(_c, deposito, e.presentacion)

    c = _computar_con(e.regla_surtimiento)
    avisos.extend(c.avisos)
    if c.oportuna is False:
        avisos.append(AVISO_EXTEMPORANEA)

    # ── La autoridad, leída del acto ─────────────────────────────────────
    # No se le pregunta al secretario lo que está en el documento que ya subió.
    # Cada campo que se le pide y podría deducirse es un minuto suyo y una
    # ocasión de equivocarse: el adelanto vale por lo que le ahorra.
    import fase_autoridad as _fa
    # ── EL OTRO CAMINO: LO QUE ÉL TECLEA, QUE NO VALIDABA NADIE ───────────
    # Medido sobre las 89 autoridades del acervo del taller: 37 las tecleó el
    # secretario sin pasar por el extractor, y entre ellas hay siete sellos que
    # no nombran a nadie —«JUZGADO» tres veces, «juez», «JUNTA»— y «la Segunda
    # Sala de la Suprema Corte», que es exactamente lo que el veto existe para
    # impedir. Ese texto sale tal cual en la carátula y en el resolutivo.
    #
    # SE AVISA, NUNCA SE BORRA. Manda lo que él escriba: «Legislatura del
    # Estado de Querétaro» y «Agencia de Movilidad» son responsables legítimas
    # en amparo indirecto aunque este módulo no sepa leerlas, y tacharlas sería
    # la comprobación acusando al trabajo correcto.
    _tecleada = (e.responsable or "").strip()
    if _tecleada and _fa.sello_sin_identidad(_tecleada):
        avisos.append(
            f"«{_tecleada}» no nombra a un órgano concreto: le falta el número, "
            f"la materia o el lugar. Ese texto sale tal cual en la carátula y "
            f"en el resolutivo. Complétalo en el encargo.")
    if _tecleada and _fa._nunca_responsable(_tecleada):
        avisos.append(
            f"«{_tecleada}» no puede ser la autoridad responsable: ningún "
            f"tribunal colegiado revisa a la Suprema Corte ni a otro colegiado. "
            f"Compruébalo en el proemio del acto reclamado.")
    if not _tecleada:
        # EL TIPO FILTRA QUÉ ÓRGANO PUEDE SER. Sin él, en la queja se quedaba
        # con el juez del juicio natural que el auto recurrido nombra por
        # dentro, en vez de con el Juzgado de Distrito que lo dictó.
        leida = _fa.de_texto(texto_acto, e.tipo_asunto)
        if leida:
            e.responsable = leida
            # CON SU RESERVA. El registro era mudo sobre su propia fiabilidad,
            # y este dato acaba en doce sitios del documento, cuatro de ellos
            # puntos resolutivos.
            print(f"   ⚖️ autoridad responsable leída del acto: «{leida[:70]}»"
                  f" — compruébala en la carátula")
        else:
            avisos.append(
                "No se pudo leer la autoridad responsable del acto reclamado, y "
                "sin ella la competencia, los efectos y el resolutivo salen con "
                "hueco. Escríbela en el encargo.")

    # ── Las constancias, acotadas ────────────────────────────────────────
    # Un expediente son cien mil caracteres y el prompt no los aguanta. Cuando
    # no cabe entero se conserva lo NORMATIVO —las cláusulas, los preceptos del
    # reglamento—, que es lo que un acervo público no puede darnos; el relato
    # de los hechos ya viene por el acto reclamado y por la demanda.
    autos = _acotar_autos(texto_autos)

    # ── Fases 1-3 — lectura ──────────────────────────────────────────────
    # La ficha de partes se hace AQUÍ, con los documentos delante, y viaja al
    # estudio. Sin ella el redactor resuelve los sujetos por proximidad, y en un
    # juicio con reconvención y tercero interesado la proximidad miente.
    # LA ESTRUCTURA ARRANCABA A CIEGAS, Y ESO SÍ SE PERDÍA. El comentario que
    # había aquí decía que corriéndola en paralelo con las fases «no se pierde
    # nada, porque la competencia y la existencia sólo necesitan los datos del
    # encargo». Es cierto de la competencia; no de los RESULTANDOS, que también
    # los escribe esta llamada y que tienen que individualizar el acto y nombrar
    # al tercero interesado. Sin el acto ni la ficha de partes delante, el
    # modelo no podía más que la perífrasis, y salía:
    #
    #     «promovió demanda de amparo CONTRA EL ACTO RECLAMADO PRECISADO EN LOS
    #      ANTECEDENTES»
    #     «LA PERSONA A QUIEN RESULTA TAL CARÁCTER fue emplazada»
    #
    # Y con ella se llevaba el {expediente} del considerando SEGUNDO, que se lee
    # de los resultandos ya escritos: la evasión y el asterisco eran el mismo
    # defecto. Lo señaló David como dos hallazgos separados; es uno.
    #
    # SE CONSERVA EL PARALELO donde de verdad lo hay: la ficha de partes y la
    # estructura van encadenadas —la segunda necesita la primera— pero las dos
    # juntas siguen corriendo a la vez que las fases de lectura, así que la
    # espera sigue siendo la del más lento y no la suma.
    #
    # ═══ CON LA PROCEDENCIA POR TIPO NO SE ESCRIBE NADA A CIEGAS (3-oct-2026) ═
    # Aun con el acto y las partes delante, `redactar_estructura` escribe el
    # V I S T O y TODOS los resultandos sin el auto de admisión, sin el de
    # turno y sin la demanda; y de su prosa se leen luego con regex el
    # expediente de la existencia, la fecha del resolutivo de la RF y el
    # inciso del 97. Con la bandera NO SE LLAMA: a la vez que las fases 1-3, el modelo
    # sólo LEE el acto (`ficha_tramite.leer_acto`, JSON anclado al papel), y
    # cuando ya están la ficha procesal y las partes se arma la ficha de
    # trámite; la `Estructura` la compone `resultandos_por_tipo.componer`, sin
    # modelo, en `_componer_generado`.
    _por_tipo = (e.modo or "").lower() == "generado" and rige_procedencia_por_tipo()
    # ═══ QUIÉN RECURRE, DEL ESCRITO, SI EL FORMULARIO NO LO DICE (3-oct-2026) ═
    # AR 631/2025 de punta a punta: ni «quien promueve» ni «recurrente» venían
    # en el formulario, y la ficha de partes puso de recurrente a la quejosa
    # amparada. El escrito de revisión lo decía en su primera línea («…, hoy
    # TERCERO INTERESADO»). Con la bandera, en los recursos y SÓLO si el
    # secretario no lo tecleó, se lee del escrito (`ficha_tramite.leer_escrito`,
    # anclado al papel) a la vez que el acto. Lo tecleado manda siempre.
    _quien_del_escrito = bool(
        _por_tipo and e.es_recurso
        and _tipo_de_encargo(e) != "revision_fiscal"
        and not str(getattr(e, "quejoso", "") or "").strip()
        and not str(getattr(e, "recurrente", "") or "").strip())

    async def _partes_y_estructura():
        if _por_tipo:
            _p, _acto_l, _esc_l = await asyncio.gather(
                fpartes.fichar(cliente, texto_acto, texto_conceptos, e.tipo_asunto),
                _leer_acto_tramite(cliente, texto_acto, e, texto_conceptos),
                (_leer_escrito_tramite(cliente, texto_conceptos, e) if _quien_del_escrito
                 else asyncio.sleep(0, result=(None, []))))
            return _p, None, _acto_l, _esc_l
        _p = await fpartes.fichar(cliente, texto_acto, texto_conceptos,
                                  e.tipo_asunto)
        _est = None
        if (e.modo or "").lower() == "generado":
            import documento_generado as _dg
            _est = await _dg.redactar_estructura(
                cliente, _datos_estructura(e, acto=texto_acto, partes=_p))
        return _p, _est, None, None

    # ── EL ORIGEN DEL ACTO, ANTES DE LEER (30-sep-2026) ─────────────────────
    # En amparo directo las fases 1-3 nombran al órgano con
    # `tipos_asunto.sujetos_de`, que desde hoy lo lee del contexto: sin esto,
    # un juicio oral mercantil volvía a salir como «la Sala». Se lee del nombre
    # de la responsable y del acto; al terminar se afina con los antecedentes.
    _oa = _ct_o = _o_pre = None
    try:
        import contexto_taller as _ct_o
        # SIEMPRE SE PONE, aunque sea None: el contexto conserva el origen
        # entre `poner()`s, y un banco que corre varios asuntos en un proceso
        # no debe heredar el del anterior.
        _ct_o.poner_origen(None)
        import tipos_asunto as _ta_o
        if _ta_o.normalizar(e.tipo_asunto or "amparo_directo") == "amparo_directo":
            import origen_acto as _oa
            _o_pre = _oa.origen(getattr(e, "responsable", "") or "", "", "", texto_acto or "",
                                numero=getattr(e, "numero", "") or "")
            _ct_o.poner_origen(_o_pre)
    except Exception as _eo:
        print(f"   ⚠️ origen del acto sin leer: {type(_eo).__name__}")

    with cronometrar("fases1-3+partes+estructura"):
        f, (partes, estructura_previa, _acto_tramite, _escrito_tramite) = await asyncio.gather(
            f123.correr(cliente, texto_acto, texto_conceptos, e.es_recurso,
                        e.tipo_asunto,
                        quejoso=getattr(e, "quejoso", "") or "",
                        responsable=getattr(e, "responsable", "") or ""),
            _partes_y_estructura())
    avisos.extend(f.avisos)
    avisos.extend(partes.avisos)
    if _o_pre is not None:
        try:
            f.origen = _oa.origen(getattr(e, "responsable", "") or "", f.antecedentes or "",
                                  f.resumen_acto or "", texto_acto or "",
                                  numero=getattr(e, "numero", "") or "")
            _ct_o.poner_origen(f.origen)
            _dice = _oa.aviso(f.origen)
            if _dice:
                print(f"   🏛️ {_dice}")
            _c = f.origen.get("cumplimiento") or {}
            if _c.get("consta") and not _c.get("efectos") and _ct_o.rediseno("cumplimiento_ejecutoria"):
                avisos.append(
                    f"La sentencia reclamada se dictó en cumplimiento de la ejecutoria del "
                    f"{_c.get('ejecutoria') or 'amparo anterior'}, y el acto no transcribe sus efectos. "
                    f"Sin ellos no se puede separar lo que quedó vinculado de lo que se resolvió con "
                    f"libertad de jurisdicción: súbela como constancia o pega sus efectos.")
        except Exception as _eo2:
            print(f"   ⚠️ origen del acto sin afinar: {type(_eo2).__name__}")

    # ── EL NOMBRE DE QUIEN PROMUEVE, LEÍDO EN VEZ DE TECLEADO ──────────────
    # David: «muchos de los datos pueden obtenerse de los documentos
    # escaneados». Éste es uno: `fase_partes.fichar` lo saca del acto y del
    # escrito —medido sobre los cinco expedientes reales: 3 exactos, 2
    # parciales, CERO invenciones—.
    #
    # LLEGA COMO PROPUESTA, NO COMO HECHO. Va con su aviso para que quien firma
    # lo confirme, porque las dos discrepancias medidas no eran de lectura sino
    # de criterio: en la cuota pensionaria el representante frente al
    # representado, y en el ARA el alias «y/o» perdido. Eso lo resuelve un
    # vistazo, no un algoritmo. Y si el secretario lo escribió, manda él.
    # ═══ EN UN RECURSO, EL QUEJOSO SE LEE DE LA SENTENCIA RECURRIDA ═══════
    # No del formulario, que pide «quien promueve» y en un recurso eso es el
    # RECURRENTE. `fase_partes` lee de la propia sentencia quién promovió el
    # amparo; cuando lo que tecleó el secretario es OTRA persona y esa persona
    # es una autoridad —que es el caso de siempre: el amparo lo gana el
    # particular y recurre la responsable—, se separan los dos papeles: la
    # autoridad pasa a `recurrente` y el quejoso leído ocupa su sitio. Así el
    # resolutivo niega o concede el amparo a quien lo pidió.
    #
    # Y NO SÓLO CUANDO RECURRE UNA AUTORIDAD (28-sep-2026, AR 631/2025): la
    # tercera interesada que recurrió una concesión quedó de «quejoso», y como
    # no es autoridad, nada la separaba. Decide `_quejoso_del_amparo`, que lee
    # además el punto resolutivo del juzgado —a quién amparó— y la ficha: si
    # dice que la quejosa es otra, lo tecleado es la recurrente. Se hace AQUÍ,
    # en la raíz, para que el encargo que se guarda ya lleve los dos papeles.
    #
    # SI EL ESCRITO YA DIJO QUIÉN RECURRE (3-oct-2026, `fijar_quien_recurre`),
    # los papeles quedan fijados por él y esta separación no se repite: con el
    # nombre de la quejosa recurrente puesto como «quien promueve», compararlo
    # con el resolutivo podía volver a partirla en dos.
    _esc_l, _av_esc = _escrito_tramite or (None, [])
    _fijado_por_escrito = False
    if _quien_del_escrito:
        _fijado_por_escrito = fijar_quien_recurre(e, _esc_l, partes, texto_acto, avisos)
        for _a in _av_esc:
            if _a not in avisos:
                avisos.append(_a)
    if getattr(e, "es_recurso", False) and not _fijado_por_escrito:
        _q_tecleado = str(getattr(e, "quejoso", "") or "").strip()
        try:
            _res_ad = _resolutivo_del_a_quo(None, texto_acto or "")
        except Exception:
            _res_ad = ""
        _q_leido = _quejoso_del_amparo(e, partes, _res_ad) if _q_tecleado else ""
        if (_q_leido and _q_tecleado and _q_leido != _q_tecleado
                and not str(getattr(e, "recurrente", "") or "").strip()):
            e.recurrente = _q_tecleado
            e.quejoso = _q_leido
            _caracter = ("la autoridad" if _parece_autoridad(_q_tecleado)
                         else "la parte tercera interesada")
            avisos.insert(0,
                f"SE SEPARARON LOS PAPELES: quien pidió el amparo es «{_q_leido}» "
                f"(leído de la sentencia recurrida) y quien recurre es {_caracter} "
                f"«{_q_tecleado[:90]}». El amparo se concede o se niega a la primera; "
                f"el recurso se califica a la segunda. Compruébalo en la carátula.")
        # EL MISMO NOMBRE CON OTRA GRAFÍA NO SEPARA LOS PAPELES (revisión del
        # 28-sep-2026): se toma como la quejosa que recurre, y se dice.
        try:
            import fase_rama as _fr_g
            _q_res = _fr_g.quejoso_del_resolutivo(_res_ad) if _res_ad else ""
        except Exception:
            _q_res = ""
        if (_q_res and _q_tecleado and not _misma_parte(_q_res, _q_tecleado)
                and _parte_parecida(_q_res, _q_tecleado)):
            avisos.insert(0,
                f"EL NOMBRE TECLEADO «{_q_tecleado[:90]}» Y EL DEL RESOLUTIVO DEL JUZGADO "
                f"«{_q_res[:90]}» SE ESCRIBEN DISTINTO, pero se tomaron como la misma "
                f"parte: recurre la quejosa. Si quien recurre es otra parte, escríbela "
                f"como recurrente en el encargo.")
    if not str(getattr(e, "quejoso", "") or "").strip():
        _leido = str(getattr(partes, "quejoso", "") or "").strip()
        # LA QUE RECURRE NO ES LA QUEJOSA (3-oct-2026): si el escrito fijó a
        # otra parte como recurrente, la ficha de partes —que también lee el
        # escrito— puede dar su nombre como «quejoso». Mejor el hueco a la vista.
        if _leido and _fijado_por_escrito and _parte_parecida(
                _leido, str(getattr(e, "recurrente", "") or "")):
            _leido = ""
        if _leido:
            e.quejoso = _leido
            avisos.insert(0,
                f"EL NOMBRE DE QUIEN PROMUEVE SE LEYÓ DE LOS DOCUMENTOS: "
                f"«{_leido}». No lo tecleaste, así que compruébalo en la "
                f"carátula: si la parte tiene alias, representante o son "
                f"varios, corrígelo.")
        else:
            avisos.insert(0,
                "NO SE PUDO LEER EL NOMBRE DE QUIEN PROMUEVE de los "
                "documentos, y no lo tecleaste: la carátula sale con el hueco "
                "a la vista. Escríbelo.")

    # ── LA FICHA DE TRÁMITE, CON LOS PAPELES YA SEPARADOS (3-oct-2026) ────
    # AQUÍ Y NO ANTES: quejosa y recurrente se acaban de fijar arriba, y la
    # ficha procesal que entra a la de trámite los lee del encargo. El
    # desenlace del a quo (lectura determinista del PDF) se adelanta a este
    # punto para que la ficha y el resultando del AR digan el mismo verbo que
    # el resolutivo; sin la bandera se sigue leyendo donde siempre.
    # SUS AVISOS VIAJAN CON ELLA (`tramite["avisos"]`) y salen en la
    # `Estructura` que se compone de ella, en el adelanto y en el proyecto.
    _desenlace_leido = False
    if _por_tipo:
        if e.es_recurso:
            _leer_desenlace_del_a_quo(f, texto_acto)
            _desenlace_leido = True

    # ── LA AUTORIDAD QUE RECURRE: SURTE AL QUEDAR HECHA (3-oct-2026) ──────
    # El cómputo de arriba corrió antes de saber QUIÉN recurre: eso se sabe
    # ahora, con los papeles separados y el resolutivo del juzgado leído. Si
    # recurre una autoridad en la revisión o en la queja con la regla de los
    # particulares (la de omisión, «personal», o «lista»), se rehace UNA vez
    # con la suya —«oficio», art. 31, fr. I— y la regla queda en el encargo:
    # la sesión la guarda y el worker que resuelva recomputa con ella, no con
    # la vieja. ANTES DE ARMAR LA FICHA, para que su forma de notificación
    # diga lo mismo que el cómputo.
    # SIN BANDERA TAMBIÉN (D2 del integrador, 3-oct-2026): sin ella el párrafo
    # dejaba «conforme al *********» y la orden de regenerar en el caso
    # frecuente de la autoridad que recurre la concesión, y la tabla del
    # cómputo seguía citando la fracción II. Es una corrección, no un diseño.
    # CON LA BANDERA, ADEMÁS, LA FORMA LEÍDA O DECLARADA (rev_0, Q 24/2026 y
    # Q 172/2026): si el formulario, el escrito, el auto o el acto dicen que la
    # notificación fue electrónica, por lista o por oficio y la regla llegó
    # como la de omisión («personal»), se cuenta con esa forma y se avisa. En
    # la Q 24 la de omisión daba en tiempo un escrito extemporáneo.
    _acto_l, _av_acto = _acto_tramite or (None, [])
    if e.es_recurso:
        _con_forma = rige_procedencia_por_tipo()
        _forma, _fuente_forma = (forma_de_notificacion(e, _esc_l, _acto_l)
                                 if _con_forma else ("", ""))
        c = _recomputar_para_la_autoridad(e, c, f, partes, texto_acto, avisos, _computar_con,
                                          forma=_forma, fuente_forma=_fuente_forma,
                                          por_forma=_con_forma)
    else:
        c = _agrario_con_el_codigo_nacional(e, c, avisos, _computar_con, _av_agr,
                                            *forma_de_notificacion(e, _esc_l, _acto_l))

    if _por_tipo:
        e.tramite = armar_tramite(e, _acto_l, f, partes, texto_acto, avisos_previos=_av_acto,
                                  escrito=_esc_l)
        # ── LA REVISIÓN FISCAL POR CORREO: CUENTA EL DEPÓSITO (D3, 3-oct-2026) ─
        # RF 2/2025 del banco: el depósito (24-oct-2024) caía dentro del plazo
        # (9 a 29 de octubre) y la recepción en la Sala (6-nov) fuera; el
        # proyecto desechaba por extemporáneo un recurso que el engrose
        # confirmó. La fecha sale de la ficha (por eso aquí, ya armada).
        c = _recomputar_con_deposito(e, c, avisos, _computar_con)

    # ── Fase 7 — el documento ────────────────────────────────────────────
    relleno = ens.Relleno(
        encabezado=e.encabezado, numero_asunto=e.numero, quejoso=e.quejoso,
        magistrado=e.magistrado, secretario=e.secretario,
        oportunidad=f0.parrafo_oportunidad(c),
        antecedentes=f.parrafos_antecedentes(),
        resumen_acto=f.parrafos_acto(),
        resumen_conceptos=f.parrafos_conceptos(),
        problemas=f.parrafos_problemas(),
        presentacion=f0.fecha_en_letra(e.presentacion)
                     if getattr(e, 'presentacion', None) else '',
        responsable=getattr(e, 'responsable', '') or '',
        es_recurso=e.es_recurso,
    )
    estructura = None
    if (e.modo or "").lower() == "generado":
        with cronometrar("estructura+docx"):
            # YA NO ES UNA TAREA. La estructura se resuelve arriba, encadenada
            # tras la ficha de partes; aquí llega hecha.
            estructura = estructura_previa
            ruta, av_gen, estructura = await _componer_generado(
                cliente, e, relleno, c, ruta_salida,
                estructura_previa=estructura, acto=texto_acto, partes=partes,
                fases=f,
                # EL ADELANTO NO LLEVA PREGUNTAS y no es un olvido: los
                # problemas se fijan en `/taller/proponer`, que corre después.
                # Aquí todavía no existen.
                criterios=None)
        avisos.extend(av_gen)
    else:
        with cronometrar("ensamblado"):
            ruta = ens.ensamblar(e.plantilla, relleno, ruta_salida)
            # El documento se lee antes de entregarlo: lo que quedó de la
            # plantilla no se ve leyendo por encima, se ve contándolo.
            avisos.extend(ens.residuo_de_plantilla(ruta, e.numero, e.plantilla))

    # Las constancias cuelgan de las fases porque son lo único que se serializa
    # entero al guardar la sesión: colgarlas del Resultado las perdería en
    # cuanto la petición siguiente cayera en el otro worker de gunicorn.
    f.autos = autos
    # LOS TEXTOS DE ORIGEN, para poder comprobar después que nada del proyecto
    # viene de fuera del asunto. Se guarda un extracto: comprobar la
    # contaminación no justifica duplicar el expediente entero en la sesión.
    # EL TOPE DE 120.000 CORTABA EL ESCRITO ANTES DE QUE NADIE LO LEYERA.
    # Medido en el ADC 245/2024: el escrito son 187.743 caracteres y aquí se
    # guardaban 120.000, así que 67.743 —y con ellos los conceptos del final—
    # no llegaban ni al detector de contaminación ni, ahora, al estudio. Se
    # sube a cubrir un escrito grande de verdad; el modelo aguanta de sobra
    # (medido: 153.256 caracteres = 38.314 tokens, sin despeinarse).
    f.fuentes = [(texto_acto or "")[:600000], (texto_conceptos or "")[:600000]]
    # SE LEE AQUÍ, QUE ES DONDE ESTÁ EL PAPEL. Después ya no: `fuentes` no
    # viaja en el estado de la sesión y el worker que resuelva puede no ser
    # éste. Las dos lecturas son deterministas —un barrido sobre el texto, sin
    # modelo— y las dos las necesita el resolutivo.
    if e.es_recurso and not _desenlace_leido:
        _leer_desenlace_del_a_quo(f, texto_acto)
    if autos:
        print(f"   📁 constancias del expediente: {len(autos)} caracteres")

    # EL ADELANTO TAMBIÉN SE COMPRUEBA. La revisión de contaminación sólo
    # corría en `_terminar()`, es decir, al final del camino largo: quien pide
    # únicamente el adelanto —que es la mayoría— se llevaba los resultandos con
    # sus nombres, expedientes y cantidades sin que nadie los contrastara con
    # los documentos que subió. Y el adelanto es precisamente donde van los
    # datos duros del asunto: quién promovió, contra qué, cuándo y por cuánto.
    _r0 = Resultado(ruta=ruta, computo=c, fases=f, encargo=e, partes=partes)
    for _a in _revisar_contaminacion(_r0, e):
        if _a not in avisos:
            avisos.append(_a)

    return Resultado(ruta=ruta, computo=c, fases=f, encargo=e, partes=partes,
                     huecos=ens.huecos_pendientes(ruta), avisos=avisos,
                     estructura=estructura)


def _leer_desenlace_del_a_quo(f, texto_acto: str) -> None:
    """Lo que resolvió el juzgado y el origen, leídos del PDF de la recurrida,
    colgados de las fases. Sin modelo. Es el bloque que vivía dentro de
    `generar`, sacado tal cual a una función (3-oct-2026) para poder leerlo
    ANTES de armar la ficha de trámite cuando rige `procedencia_por_tipo`; sin
    la bandera se llama en el mismo sitio de siempre."""
    try:
        import fase_rama as _fr_a
        f.resolutivo_recurrida = _fr_a.resolutivo_recurrida(texto_acto or "")
        # LOS PUNTOS RESOLUTIVOS PRIMERO (28-sep-2026, AR 631/2025): el
        # recuento de verbos sobre el PDF entero dio «niega» porque la
        # sentencia narra un amparo ANTERIOR que «negó el amparo»; su
        # resolutivo decía «ampara y protege». El recuento, sólo si los
        # puntos resolutivos no se dejan leer.
        f.resolvio_a_quo = (_fr_a.resolvio_segun_resolutivos(texto_acto or "")
                            or _fr_a.resolvio_a_quo(texto_acto or "",
                                                    resolutivo=f.resolutivo_recurrida))
        import fase_origen as _fo_a
        _dd = _fo_a.datos_del_documento(texto_acto or "")
        f.expediente_origen = _dd.get("expediente", "")
        f.fecha_origen = _dd.get("fecha", "")
        print(f"   ⚖️ el juzgado {f.resolvio_a_quo or '(no consta)'}"
              f" · resolutivo reproducible: "
              f"{'sí' if f.resolutivo_recurrida else 'no'}")
        print(f"   📑 origen leído del PDF: expediente "
              f"{f.expediente_origen or '(no consta)'} · fecha "
              f"{f.fecha_origen or '(no consta)'}")
    except Exception as _ex:
        print(f"   ⚠️ no se pudo leer el desenlace del a quo: {type(_ex).__name__}")


# ═══════════════════════════════════════════════════════════════════════════
# La segunda mitad: el criterio del secretario entra AQUÍ y sólo aquí
# ═══════════════════════════════════════════════════════════════════════════
#
# El proceso se parte en dos a propósito, porque así es como David lo describió:
# la máquina lee y ordena, él decide, la máquina redacta la demostración. Entre
# `generar()` y `resolver()` hay una persona, y ese es el punto del diseño.
#
#   generar()   →  adelanto con los problemas jurídicos planteados
#   consultar() →  lo que el acervo tiene sobre esos problemas, para que decida
#   resolver()  →  la sentencia, con su criterio dentro


async def consultar(qdrant, embed_juris, embed_leyes,
                    r: Resultado, cliente=None,
                    contexto: str = "") -> f6.Material:
    """Lo que el acervo dice sobre los problemas del caso.

    Se le enseña ANTES de pedirle el criterio: decidir el sentido sin ver la
    jurisprudencia obligatoria del tema es exactamente el error que este
    utillaje existe para evitar.
    """
    # Los problemas de la Fase 3 son DICCIONARIOS —pregunta, resolvió, combate,
    # impedimento—, no cadenas. Con datos de prueba sintéticos nunca se notó;
    # con el primer caso real, `'dict' object has no attribute 'strip'`.
    problemas = ([r.fases.problema_global] if r.fases.problema_global else [])
    for p in (r.fases.problemas or []):
        pregunta = p.get("pregunta", "") if isinstance(p, dict) else str(p)
        if pregunta:
            problemas.append(pregunta)
        # EL ACERVO SE CONSULTA POR LAS DOS. El impedimento técnico es una
        # cuestión jurídica que hay que fundar —la inoperancia se razona con
        # tesis, no se declara— pero consultarlo SÓLO a él inclinaba la
        # balanza antes de que nadie decidiera: el RAG volvía cargado de
        # material para inoperar y vacío de material para entrar al fondo. Un
        # buscador que sólo busca razones para no contestar acaba
        # encontrándolas.
        if isinstance(p, dict):
            for _clave, _pre in (("impedimento", "inoperancia"),
                                 ("apoyo", "sustento")):
                _x = p.get(_clave)
                if isinstance(_x, dict) and _x.get("explicacion"):
                    # EL VICIO, NO EL GÉNERO (2-oct-2026, David: «la cita de
                    # inoperancia siempre es la misma»): con la bandera
                    # `inoperancia_por_vicio`, el impedimento pregunta por SU
                    # vicio; sin ella, «¿Inoperancia: …?» como antes.
                    import vicio_inoperancia as _vi_c
                    problemas.append(_vi_c.pregunta_sintetica(_clave, _pre, _x))
    coleccion = (r.encargo.coleccion_estatal if r.encargo else "") or None
    # LA LEY DEL ESTADO NO PINTA NADA EN UN LABORAL FEDERAL. En el ADL 382/2024
    # —IMSS contra un enfermero, ante la Junta Federal— el marco jurídico salió
    # citando el artículo 142 de la LEY ORGÁNICA MUNICIPAL DE QUERÉTARO, que
    # regula el recurso de inconformidad contra multas y licencias de comercio,
    # y el Código de Procedimientos Civiles del Estado, que se traía sólo para
    # decir que no rige. Una ejecutoria no cita leyes impertinentes para
    # explicar que no aplican.
    #
    # Se busca en el acervo estatal sólo cuando la controversia puede regirse
    # por ley local. En laboral la rige la Ley Federal del Trabajo, salvo el
    # burocrático estatal, que se reconoce porque el patrón es el propio Estado
    # o un municipio.
    if fp_materia(r.encargo) == "laboral" and not _burocratico_estatal(r):
        coleccion = None
    # LA REVISIÓN FISCAL ES FEDERAL. La sentencia que se revisa es de una Sala
    # del Tribunal Federal de Justicia Administrativa y el fondo se rige por el
    # Código Fiscal de la Federación y la LFPCA. En la 61/2025 la colección
    # estatal metió al material la Ley de Procedimientos Administrativos y el
    # Código Fiscal DE QUERÉTARO, y de ahí salió traído el «artículo 134» del
    # Código Civil del estado en un asunto de notificaciones fiscales.
    try:
        import tipos_asunto as _ta_f
        if _ta_f.normalizar(getattr(r.encargo, "tipo_asunto", "")) == "revision_fiscal":
            coleccion = None
    except Exception:
        pass

    # EL SONDEO DE PRECEDENTE VA EN PARALELO al material. Cuesta menos de dos
    # segundos —se mide— y responde una pregunta que hasta ahora nadie hacía:
    # cómo resolvieron otros colegiados este mismo problema. Se le enseña al
    # redactor junto con el material, pero SEPARADO de él, porque un precedente
    # de otro tribunal no funda: orienta.
    # ═══ LOS DOS DATOS QUE EL SISTEMA DERIVA DEL EXPEDIENTE ══════════════════
    #
    # David: «no quiero que le impongas al modelo que invoque esa ley, sino que
    # modifiques la ARQUITECTURA para que lo entienda». Esto es esa
    # arquitectura: quién dictó el acto reclamado y de qué cuaderno viene la
    # recurrida se LEEN, y de ahí sale si la Ley de Amparo gobierna el acto o no.
    #
    # Se lee de los tres sitios y en este orden: el texto crudo cuando está
    # —sólo se guarda en 14 de 108 sesiones—, y si no, los antecedentes y el
    # resumen del acto, que están en las 108.
    _sede, _cuaderno = "", ""
    if getattr(r.encargo, "es_recurso", False):
        try:
            import fase_rama as _fr_s
            _f = r.fases
            _base = "\n".join(x for x in (
                (getattr(_f, "fuentes", None) or [""])[0] if getattr(_f, "fuentes", None) else "",
                getattr(_f, "antecedentes", "") or "",
                getattr(_f, "resumen_acto", "") or "") if x)
            _sede, _quien = _fr_s.sede_del_acto(_base)
            _cuaderno, _porque = _fr_s.cuaderno_recurrido(_base)
            print(f"   ⚖️ sede del acto: {_sede or '(no consta)'}"
                  f" · cuaderno: {_cuaderno or '(no consta)'}"
                  f" · {(_quien or '')[:60]}")
            # SE DICE EN VOZ ALTA. Que el sistema lo dedujo bien o mal lo tiene
            # que poder ver el secretario, no descubrirlo en el documento.
            if _sede == "ordinaria" and _cuaderno == "principal":
                r.avisos.append(
                    f"EL ACTO RECLAMADO NO SE RIGE POR LA LEY DE AMPARO: lo "
                    f"dictó {(_quien or 'una autoridad ordinaria')[:80]} y el "
                    f"recurso va contra la sentencia del cuaderno principal, no "
                    f"contra el incidente de suspensión. El fondo se juzga con "
                    f"la ley que aplicó esa autoridad, y los criterios sobre la "
                    f"suspensión del amparo se dejaron fuera de la búsqueda. "
                    f"Si esto no es así en este asunto, dilo: cambia el material.")
        except Exception as _ex:
            print(f"   ⚠️ no se pudo derivar la sede del acto: {_ex}")

    material, sondeo, espejo = await asyncio.gather(
        f6rag.material_del_caso(qdrant, embed_juris, embed_leyes,
                                problemas, coleccion, fp_materia(r.encargo),
                                # EL TRADUCTOR DE LA CONSULTA. Sin cliente la
                                # búsqueda sigue funcionando con la pregunta
                                # cruda; con él, alcanza el rubro.
                                cliente,
                                # Y LO QUE EL SECRETARIO YA SABE DEL ASUNTO,
                                # como ancla propia. Ver `material_para`.
                                contexto,
                                # Y los dos datos derivados, que deciden si la
                                # suspensión del amparo viene a cuento.
                                _sede, _cuaderno),
        _sondear_precedente(qdrant, embed_leyes, r, problemas),
        # EL ESPEJO, EN EL MISMO TIRO. Cuesta 0.2 s por planteamiento y corre a
        # la vez que el material: no se nota en la espera.
        _espejo_propio(qdrant, embed_leyes, r, problemas))
    material.sondeo = sondeo
    material.espejo = espejo or []
    # Y LOS DOS DATOS DERIVADOS VIAJAN CON EL MATERIAL, que es lo que llega a
    # todos los prompts y a todos los verificadores.
    material.sede_del_acto = _sede
    material.cuaderno = _cuaderno
    material.materia = fp_materia(r.encargo)
    # Y EL TIPO, para que la prosa del estudio nombre a las partes con las
    # figuras de ESTE recurso y no con las del amparo directo.
    material.tipo_asunto = r.encargo.tipo_asunto
    # LA FUERZA RESPECTO DE ESTE TRIBUNAL (rediseño, punto 3): su circuito, su
    # región y su designación deciden si una tesis lo vincula.
    material.tribunal = str(getattr(r.encargo, "tribunal", "") or "")
    # LAS TESIS QUE INVOCA LA PARTE, ANTES DE PROPONER Y DE PLANEAR (rediseño,
    # punto 7, decisión 3 de David del 29-sep-2026). Antes se traían DESPUÉS de
    # redactar (`_fuentes_tardias`): el estudio contestaba el argumento sin el
    # texto de la tesis que la parte invocó, y lo tardío sólo producía ficha.
    # Traídas aquí son fuente de premisa: las ven la propuesta, el plan y el
    # estudio. Sólo entra lo que EXISTE en el acervo, por registro o clave.
    try:
        import fases123_pipeline as _f123q
        import contexto_taller as _ct_q
        _esc_q = (list(getattr(r.fases, "fuentes", []) or []) + ["", ""])[1]
        _citas_q = sorted(_f123q.citas_invocadas(_esc_q)) if _ct_q.rediseno("tesis_parte_al_consultar") else []
        if _citas_q and qdrant is not None:
            _nq = await f6rag.completar_tesis_citadas(
                qdrant, material, _citas_q,
                tipo_asunto=getattr(r.encargo, "tipo_asunto", "") or "")
            if _nq:
                print(f"   ⚖️ las {len(_nq)} tesis que invoca la parte, traídas ANTES de proponer: "
                      f"{', '.join(_nq[:6])}")
    except Exception as _eq:
        print(f"   ⚠️ tesis de la parte sin traer antes de proponer: {type(_eq).__name__}")
    # EN UNA EVALUACIÓN, NADA PUBLICADO DESDE LA SENTENCIA QUE SE MIDE
    # (contexto_taller; `fecha_publicacion` la guarda _tesis_de desde el 29-sep).
    import contexto_taller as _ct
    _exc = _ct.exclusion()
    if _exc is not None:
        _antes = len(material.tesis or [])
        material.tesis = [t for t in (material.tesis or []) if not _exc.excluye_tesis(t)]
        if len(material.tesis) != _antes:
            print(f"   🧪 evaluación: {_antes - len(material.tesis)} tesis posteriores al corte fuera")
    import fuerza_juridica as _fj
    _fj.anotar(material.tesis, material.tribunal)
    material.entidad = _entidad_de(coleccion)
    if sondeo is not None:
        for a in (sondeo.avisos or []):
            if a not in r.avisos:
                r.avisos.append(a)

    # ═══════════════════════════════════════════════════════════════════════
    # LAS TESIS DE LA TÉCNICA, QUE NADIE VA A PEDIR
    # ═══════════════════════════════════════════════════════════════════════
    # La búsqueda va detrás de los PROBLEMAS DEL CASO. Pero si procede el
    # reenvío, o si el colegiado puede sustituir a la Sala, no es un problema
    # del expediente: es la regla con la que se escribe el resolutivo, y nadie
    # la formula como pregunta.
    #
    # Medido en la revisión fiscal 91/2025: el estudio argumentó el reenvío con
    # todas las letras y no citó NINGUNA autoridad, porque la búsqueda había
    # ido detrás de la notificación electrónica. Las cuatro tesis que lo
    # sostienen estaban en la colección; nadie las pidió.
    #
    # Se piden por su REGISTRO, que es lo contrario de adivinar: o existen con
    # ese número, o no viene nada.
    try:
        import tipos_asunto as _ta_t
        _regs = []
        for _regla in _ta_t.tecnica_de(getattr(r.encargo, "tipo_asunto", "")):
            _regs += list(_regla.get("apoyos") or [])
        if _regs:
            _ya = {str(t.get("registro") or "") for t in (material.tesis or [])}
            _nuevas = [t for t in await f6rag.tesis_por_registro(qdrant, _regs)
                       if t["registro"] not in _ya]
            if _nuevas:
                material.tesis = list(material.tesis or []) + _nuevas
                print(f"   ⚖️ tesis de la técnica añadidas: "
                      f"{', '.join(t['registro'] for t in _nuevas)}")
    except Exception as _et:
        print(f"   ⚠️ no se pudieron añadir las tesis de la técnica: {_et}")
    return material


_ENTIDADES = {
    "aguascalientes": "Aguascalientes", "bajacalifornia": "Baja California",
    "bajacaliforniasur": "Baja California Sur", "campeche": "Campeche",
    "chiapas": "Chiapas", "chihuahua": "Chihuahua", "cdmx": "Ciudad de México",
    "coahuila": "Coahuila", "colima": "Colima", "durango": "Durango",
    "guanajuato": "Guanajuato", "guerrero": "Guerrero", "hidalgo": "Hidalgo",
    "jalisco": "Jalisco", "edomex": "Estado de México", "mexico": "Estado de México",
    "michoacan": "Michoacán", "morelos": "Morelos", "nayarit": "Nayarit",
    "nuevoleon": "Nuevo León", "oaxaca": "Oaxaca", "puebla": "Puebla",
    "queretaro": "Querétaro", "quintanaroo": "Quintana Roo",
    "sanluispotosi": "San Luis Potosí", "sinaloa": "Sinaloa", "sonora": "Sonora",
    "tabasco": "Tabasco", "tamaulipas": "Tamaulipas", "tlaxcala": "Tlaxcala",
    "veracruz": "Veracruz", "yucatan": "Yucatán", "zacatecas": "Zacatecas",
}


def _entidad_de(coleccion: str) -> str:
    """«leyes_queretaro» → «Querétaro». Sin adivinar: si no la conozco, vacío."""
    c = (coleccion or "").strip().lower().replace("leyes_", "").replace("_", "")
    return _ENTIDADES.get(c, "")


# Cuánto del expediente cabe. Generoso: es material que no está en ningún otro
# sitio y su ausencia ya costó un asunto.
MAX_AUTOS = 24000
_RX_CLAUSULA = re.compile(
    r"(cl[áa]usula|art[íi]culo|reglamento|contrato\s+colectivo|"
    r"fracci[óo]n|condiciones\s+generales\s+de\s+trabajo)", re.I)


def _acotar_autos(texto: str) -> str:
    """El expediente, y si no cabe, su parte normativa."""
    t = " ".join((texto or "").split())
    if len(t) <= MAX_AUTOS:
        return t
    # Se trocea en párrafos y se conservan los que traen norma, en su orden.
    trozos = re.split(r"(?<=[.;])\s+(?=[A-ZÁÉÍÓÚ])", t)
    con_norma = [x for x in trozos if _RX_CLAUSULA.search(x)]
    fuera, total = [], 0
    for x in (con_norma or trozos):
        if total + len(x) > MAX_AUTOS:
            break
        fuera.append(x)
        total += len(x)
    return " ".join(fuera)


_RX_BUROCRATICO = re.compile(
    r"trabajadores?\s+al\s+servicio\s+del\s+estado|burocr[áa]tic|"
    r"tribunal\s+de\s+conciliaci[óo]n\s+y\s+arbitraje\s+del\s+estado|"
    r"ley\s+de\s+los\s+trabajadores\s+del\s+estado", re.I)


def _burocratico_estatal(r) -> bool:
    """¿Es un laboral que SÍ se rige por ley local? Sólo el burocrático."""
    f = getattr(r, "fases", None)
    texto = " ".join(str(getattr(f, k, "") or "") for k in
                     ("antecedentes", "resumen_acto", "problema_global"))
    e = getattr(r, "encargo", None)
    texto += " " + str(getattr(e, "encabezado", "") or "")
    texto += " " + str(getattr(e, "responsable", "") or "")
    return bool(_RX_BUROCRATICO.search(texto))


def fp_materia(e) -> str:
    """La materia del asunto. La DECLARADA manda sobre la deducida.

    David: «en función de la materia el RAG es selectivo (como en laboral) y
    eso lo determina un campo seleccionado por el secretario». Hasta ahora la
    materia se deducía del nombre del tribunal y del encabezado, y eso acierta
    casi siempre —un colegiado de trabajo no ve otra cosa— pero falla justo
    donde más cuesta: en un tribunal MIXTO «en materias administrativa y
    civil», donde el nombre no decide, y en un asunto laboral que llega a un
    colegiado administrativo, que es cuando el silo laboral hace falta.

    La deducción se conserva como respaldo: quien no declare materia sigue
    teniendo el comportamiento de siempre.
    """
    declarada = str(getattr(e, "materia", "") or "").strip().lower()
    if declarada:
        return declarada
    try:
        import fase_precedente as fp
    except Exception:
        return ""
    return fp.materia_de(getattr(e, "encabezado", ""),
                         getattr(e, "tribunal", ""),
                         getattr(e, "tipo_asunto", ""))


async def _espejo_propio(qdrant, embed, r: Resultado, problemas: list):
    """Las sentencias del PROPIO tribunal sobre estos puntos. Ver fase_espejo.

    Hermana de `_sondear_precedente` y con el MISMO embebedor de 1536: aquí
    conviven dos modelos y `jurisprudencia_nacional_v3` es de 3072. Pasarle el
    de jurisprudencia devuelve «expected dim: 3072, got 1536», y como esto
    captura sus propios errores para no tumbar la sentencia, en producción se
    vería simplemente como «el tribunal no ha visto esto».

    NO SE CUELGA DE `fp.sondear`. Aquél es nacional a propósito; éste es del
    tribunal propio y necesita su propio filtro y su propio umbral.

    DOS FUENTES, Y SE PRUEBA PRIMERO LA NUEVA. El índice de la OAJ
    (`fase_oaj.py`) busca planteamiento contra planteamiento y da un porcentaje
    calibrado; si no dice nada —sin tabla de calibración, sin colección, nada
    al 85% ni «posible» al 50%—, se cae al acervo viejo exactamente como antes.
    Con sólo posibles la OAJ SÍ habla: son su nivel de abajo, con su
    probabilidad real, y la tarjeta no calla por falta del de arriba.

    `problemas` es la lista de CONSULTAS que arma `consultar()` —cadenas: el
    problema global, la pregunta de cada planteamiento y las sintéticas de
    inoperancia y sustento— y sólo la usa el espejo viejo. La fuente OAJ no la
    lee: toma los planteamientos enteros de `r.fases.problemas` (ver
    `_espejo_oaj`).
    """
    try:
        import fase_espejo as fe
        import fase_precedente as fp
    except Exception:
        return []
    if not (qdrant and problemas):
        return []
    e = r.encargo
    _circ = fp.circuito_de(getattr(e, "tribunal", ""))
    clave, largo = fe.resolver_tribunal(getattr(e, "tribunal", ""), _circ)
    if not clave:
        # Fuera del circuito 22 no hay mapa de tribunales y el espejo calla.
        return []
    _txt = [p if isinstance(p, str) else str((p or {}).get("pregunta") or p)
            for p in problemas]

    # UNA SOLA FUENTE POR TARJETA, NO UNA POR PLANTEAMIENTO. La pantalla
    # enseña UNA nota de cobertura, la del primer grupo, para toda la tarjeta:
    # mezclar grupos de la OAJ con grupos del acervo viejo pondría la cobertura
    # de uno debajo de los precedentes del otro. Si la OAJ habla en alguno, la
    # tarjeta es suya entera.
    oaj = await _espejo_oaj(qdrant, embed, r, _circ, largo)
    if oaj:
        return oaj
    try:
        tiros = await asyncio.gather(*[
            fe.espejo(qdrant, embed, t, clave, _circ or "22") for t in _txt],
            return_exceptions=True)
    except Exception as exc:
        print(f"   ⚠️ espejo del tribunal omitido: {exc}")
        return []
    fuera, vistas = [], set()
    for t, filas in zip(_txt, tiros):
        if isinstance(filas, BaseException) or not filas:
            continue
        limpias = []
        for f in filas:
            # SIN REPETIR ENTRE PROBLEMAS. Dos planteamientos del mismo asunto
            # recuperan sentencias solapadas; la misma sentencia dos veces en
            # pantalla parece dos precedentes y es uno.
            k = (f["tipo_asunto"], f["expediente"], f["fecha"])
            if k in vistas:
                continue
            vistas.add(k)
            limpias.append(f)
        if len(limpias) < fe.PISO_FILAS:
            continue
        fuera.append({"problema": t, "tribunal": largo, "filas": limpias,
                      "resumen": fe.resumen(limpias),
                      "cobertura": fe.NOTA_COBERTURA})
    if fuera:
        print(f"   🪞 espejo del propio tribunal ({clave}): "
              f"{sum(len(x['filas']) for x in fuera)} sentencias propias en "
              f"{len(fuera)} de {len(_txt)} planteamientos")
    return fuera


async def _espejo_oaj(qdrant, embed, r: Resultado, circ: str,
                      largo: str) -> list:
    """El espejo desde el índice de la OAJ. [] si no habla en ningún punto.

    Mismo formato que el espejo viejo —{problema, tribunal, filas, resumen,
    cobertura}— para que la tarjeta, la fila de la sesión y el rescate no
    distingan de dónde vino; lo que cambia viaja DENTRO de cada fila
    (similitud, cota_inferior, nivel, pregunta, calificación, razón, NEUN). En
    cada grupo van primero las filas «mismo_problema» y detrás las «posible».

    Por eso mismo el front de `main`, que no lee `nivel`, pintaría un posible
    del 50% como una sentencia propia más: el front de la rama sale ANTES que
    esto (ver «El orden de despliegue» en `fase_oaj`; `OAJ_POSIBLES=0` es la
    reversa).
    """
    try:
        import fase_oaj as fo
    except Exception:
        return []
    e = r.encargo
    # QUE CONSTE EL CIRCUITO 22, NO QUE NO CONSTE OTRO. `resolver_tribunal`
    # toma el circuito vacío por el 22, y `circuito_de` devuelve vacío con
    # «… del Decimoquinto Circuito» o con un tribunal auxiliar: el espejo viejo
    # ya enseñaba así sentencias de Querétaro a secretarios de otra plaza, y
    # esta fuente les pondría encima un porcentaje. Aquí se calla.
    clave, organo = fo.organo_de(getattr(e, "tribunal", ""), circ,
                                 getattr(e, "ciudad", ""))
    if not organo:
        return []
    # LOS PLANTEAMIENTOS ENTEROS, DE LA FASE 3, Y NO LA LISTA DE CONSULTAS.
    # `consultar()` reduce cada planteamiento a su pregunta y le suma el
    # problema global y las preguntas sintéticas de inoperancia y sustento:
    # sirve para el material, pero la tabla de la OAJ se midió con «pregunta
    # combate resolvio». Con la pregunta sola el porcentaje de la tarjeta no
    # habría salido de ninguna medición, y las sintéticas, que van antes en la
    # deduplicación, les quitaban precedentes a los planteamientos reales. Un
    # planteamiento al que le falte una de las tres piezas no consulta.
    planteamientos = [
        p for p in (getattr(getattr(r, "fases", None), "problemas", None) or [])
        if isinstance(p, dict) and fo.texto_consulta(p)]
    if not planteamientos:
        return []
    tipo = getattr(e, "tipo_asunto", "") or ""
    # EL PROPIO ASUNTO NO ES SU PRECEDENTE (ver `fo.numero_expediente`). Se
    # quita DENTRO de la búsqueda, antes de contar rangos —el primero se lee
    # con su propia tabla— y otra vez aquí, por si una fila llegara por otro
    # camino.
    _numero = getattr(e, "numero", "") or ""
    try:
        tiros = await asyncio.gather(*[
            fo.precedentes_oaj(qdrant, embed, p, tipo, organo, _numero)
            for p in planteamientos], return_exceptions=True)
    except Exception as exc:
        print(f"   ⚠️ precedentes OAJ omitidos: {exc}")
        return []
    _propio = fo.numero_expediente(_numero)
    hablan = []
    for p, filas in zip(planteamientos, tiros):
        if isinstance(filas, BaseException) or not filas:
            continue
        if _propio:
            filas = [f for f in filas
                     if fo.numero_expediente(f.get("expediente")) != _propio]
        if filas:
            hablan.append((p, filas))
    limpias_de = [[] for _ in hablan]
    vistas = set()
    # SIN REPETIR ENTRE PROBLEMAS, como en el espejo viejo, y EN DOS PASADAS:
    # primero el nivel «mismo problema» de todos los planteamientos y después
    # los posibles. En una sola pasada, una sentencia que es «posible» para el
    # primer planteamiento y «mismo problema» para el tercero saldría abajo,
    # como posible, y arriba ya no: el orden de los planteamientos le habría
    # quitado el nivel que la tabla le da.
    #
    # Y EL PRINCIPAL PRIMERO en cada pasada. La tarjeta «El problema principal
    # y su solución» (tarjeta_decision._tu_tribunal) y la deliberación leen
    # SÓLO el grupo del principal: si un accesorio anterior se quedara con una
    # sentencia que también es del principal, el principal la perdería justo
    # donde decide. En la tarjeta vieja no cambia nada: la sentencia sale una
    # vez, sólo que bajo el principal.
    orden = sorted(range(len(hablan)), key=lambda i: str(
        (hablan[i][0] or {}).get("jerarquia") or "").strip().lower() != "principal")
    for pasada_posibles in (False, True):
        for i in orden:
            _p, filas = hablan[i]
            for f in filas:
                # Una fila sin `nivel` es del nivel de arriba: así la
                # escribía esta fuente antes de los dos niveles.
                es_posible = f.get("nivel") == fo.NIVEL_POSIBLE
                if es_posible != pasada_posibles:
                    continue
                # Aquí la clave es el NEUN, que identifica la sentencia en la
                # OAJ sin la ambigüedad de «44/2021», que son cuatro asuntos
                # distintos.
                k = f.get("neun") or (f.get("tipo_asunto"), f.get("expediente"),
                                      f.get("fecha"))
                if k in vistas:
                    continue
                vistas.add(k)
                limpias_de[i].append(f)

    fuera = []
    for (p, _filas), limpias in zip(hablan, limpias_de):
        # SIN PISO DE FILAS. `fe.PISO_FILAS` existe porque el coseno crudo del
        # acervo viejo no distinguía el punto del vecino; una coincidencia
        # calibrada vale sola. Y un planteamiento con sólo posibles también
        # se enseña: el nivel de abajo no necesita al de arriba para hablar.
        if not limpias:
            continue
        # El nombre PARA LEER es el de siempre; el de la OAJ, con su
        # residencia, es el valor exacto del filtro y no hace falta en pantalla.
        #
        # SIN RENGLÓN DE RESUMEN. El `sentido` de la fila es el del resolutivo
        # de la SENTENCIA y la coincidencia es con UNO de sus planteamientos
        # (o con su tema): contarlos cruza las dos escalas que `fase_espejo`
        # documenta como la causa del error del 45%. Y sin piso de filas, el
        # renglón salía como «De estos 1 asuntos propios, todos…». Cada fila ya
        # trae su calificación literal; compara el secretario.
        fuera.append({"problema": str(p.get("pregunta") or "").strip(),
                      "tribunal": largo or organo, "filas": limpias,
                      "resumen": "", "cobertura": fo.NOTA_COBERTURA})
    if fuera:
        _todas = [f for x in fuera for f in x["filas"]]
        _pos = sum(1 for f in _todas if f.get("nivel") == fo.NIVEL_POSIBLE)
        print(f"   🪞 espejo del propio tribunal (OAJ, {clave}): "
              f"{len(_todas)} precedentes ({len(_todas) - _pos} mismo problema, "
              f"{_pos} posibles) en {len(fuera)} de {len(planteamientos)} "
              f"planteamientos")
    return fuera


async def _sondear_precedente(qdrant, embed, r: Resultado, problemas: list):
    """El acervo de colegiados, sondeado por el problema, no por el escrito.

    El fraseo de la demanda envenena la búsqueda —es el fallo ya medido de
    HyDE—, así que se sondea con el problema jurídico que la fase 3 normalizó
    por concepto. Si algo falla, se devuelve None y el redactor sigue: el
    sondeo mejora la sentencia, no la condiciona.

    EL EMBEBEDOR ES EL DE 1536, NO EL DE JURISPRUDENCIA. Aquí hay dos modelos
    distintos conviviendo: `jurisprudencia_nacional_v3` se indexó con
    text-embedding-3-large (3072 dimensiones) y `sentencias_holdings` con
    text-embedding-3-small (1536). Le pasé el de jurisprudencia por costumbre y
    Qdrant devolvió «expected dim: 3072, got 1536»; como el sondeo captura sus
    propios errores para no tumbar la sentencia, en producción se habría visto
    sencillamente como «no hay precedentes». Lo cazó la prueba local, que es
    justamente para lo que sirve correrla antes de desplegar.
    """
    try:
        import fase_precedente as fp
    except Exception:
        return None
    if not (qdrant and problemas):
        return None
    e = r.encargo
    # NO SE PIDEN: se leen. La materia está en el encabezado y en el nombre del
    # tribunal; el circuito, en ese mismo nombre.
    materia = fp_materia(e)
    if not materia:
        return None
    # UNO POR PROBLEMA, EN PARALELO. Sondeaba `problemas[0]` y ya: el acervo
    # decía cómo se resuelve la cuestión principal y callaba sobre las demás,
    # que es donde el secretario más agradece la señal —un accesorio que el
    # 90% de los tribunales declara inoperante se resuelve en dos renglones—.
    # Son N búsquedas contra la misma colección; corriendo a la vez, la espera
    # es la de la más lenta.
    _circ = fp.circuito_de(getattr(e, "tribunal", ""))
    _txt = [p if isinstance(p, str) else str((p or {}).get("pregunta") or p)
            for p in problemas]
    try:
        _todos = await asyncio.gather(*[
            fp.sondear(qdrant, embed, t, materia, circuito=_circ)
            for t in _txt], return_exceptions=True)
    except Exception as exc:
        print(f"   ⚠️ sondeo de precedente omitido: {exc}")
        return None
    _sondeos = [x if not isinstance(x, BaseException) else None for x in _todos]
    if not any(_sondeos):
        return None
    s = next(x for x in _sondeos if x)
    # Los demás viajan colgados del primero: `Material.sondeo` es lo que lee el
    # estudio y no se le cambia la forma, pero la predicción de cada problema
    # tiene que llegar a la pantalla.
    s.por_problema = [
        {"problema": t, "prediccion": fp.prediccion(x) if x else {}}
        for t, x in zip(_txt, _sondeos)]
    # SE DEJA CONSTANCIA AUNQUE VAYA BIEN. Hasta ahora este paso sólo hablaba
    # cuando fallaba, y tras desplegarlo no había manera de saber desde los
    # registros si había corrido: el silencio significaba «no falló», no
    # «funcionó». Es el mismo defecto que el aviso que nadie veía. Una línea.
    _pred = [d["prediccion"].get("frase", "—") for d in s.por_problema]
    print(f"   ⚖️ jurimetría por problema: " + " · ".join(_pred[:4]))

    print(f"   ⚖️ precedente[{materia}]: "
          f"{sum(s.distribucion.values())} sentencias del tema · "
          f"{len(s.moldes)} moldes · {len(s.razonados)} con razón escrita"
          + (f" · avisos: {len(s.avisos)}" if s.avisos else ""))
    return s



def _reasuncion_del_asunto(r, criterios) -> dict | None:
    """¿Este proyecto reasume jurisdicción y estudia conceptos que nadie
    estudió? (art. 93, frs. I, V y VI). None si no se pudo calcular —entonces el
    estudio decide por la rama—; si se calculó, un dict con «reasuncion»
    (vacía = no aplica), los conceptos que ya había y de dónde, quién recurre y
    si la recurrida también sobreseyó. Sin modelo; nunca lanza.

    AR 631/2025 (28-sep-2026): el juzgado concedió, recurrió la tercera
    interesada y el proyecto negó sin estudiar los conceptos que el juzgado
    declaró innecesarios. Antes de pedirlos se buscan en el material: la
    demanda entre las constancias, o la recurrida si los transcribe."""
    try:
        import fase_rama as _fr_a
        import tipos_asunto as _ta_a
        e = getattr(r, "encargo", None)
        if e is None or _ta_a.normalizar(getattr(e, "tipo_asunto", "")) != "amparo_revision":
            return {"reasuncion": ""}
        # LA PROCEDENCIA, DE LO QUE PROSPERA DE VERDAD (revisión del 28-sep-2026):
        # que el principal sea de procedencia no basta; tiene que prosperar.
        _proc = _procedencia_prospera(r, criterios)
        info = dict(info_de_rama(r), clase_principal="", procedencia=_proc)
        _sent = "fundado" if any(_ta_a.prospera(str(getattr(c, "sentido", "")))
                                 for c in (criterios or [])) else "infundado"
        co = _fr_a.conceptos_omitidos(info, _sent, getattr(r, "fases", None))
        if not co:
            return {"reasuncion": "", "quien_recurre": info.get("quien_recurre", ""),
                    "procedencia": _proc}
        _txt, _donde = _fr_a.conceptos_disponibles(info.get("conceptos_violacion", ""),
                                                   getattr(r, "fases", None))
        # ¿QUEDÓ FIRME EL SOBRESEIMIENTO? (AR 631/2025, al generar en pantalla,
        # 28-sep-2026): el registro decía «la recurrida también sobreseyó» y
        # los resolutivos salieron sin «Queda firme…», porque el documento sólo
        # lo ponía si el estudio lo escribía. Se decide aquí, por código y con
        # la misma regla que la ficha (`tipos_asunto.sobreseimiento_firme`).
        _sob = bool(info.get("sobresee_ademas"))
        try:
            import ficha_procesal as _fp_a
            _adh = bool(_fp_a.donde_adhesivo(getattr(r, "fases", None), "amparo_revision"))
        except Exception:
            _adh = False
        _firme = _ta_a.sobreseimiento_firme(_sob, info.get("quien_recurre", ""), _adh)
        print(f"   ⚖️ REASUNCIÓN ({co['reasuncion']}, {co['fundamento']}): conceptos "
              + (f"de {_donde} ({len(_txt)} caracteres)" if _donde else "NO CONSTAN")
              + (" · la recurrida también sobreseyó" if _sob else "")
              + (" (queda firme: nadie lo impugnó)" if _firme else
                 " (consta revisión adhesiva: firmeza por comprobar)" if _sob and _adh else ""))
        return {**co, "conceptos": _txt, "quien_recurre": info.get("quien_recurre", ""),
                "sobresee_ademas": _sob, "adhesiva": _adh,
                "sobreseimiento_firme": _firme, "procedencia": _proc}
    except Exception as _er:
        print(f"   ⚠️ REASUNCIÓN: no se pudo calcular: {type(_er).__name__}")
        return None


async def _tesis_de_la_tecnica(qdrant, material, rama: str) -> None:
    """Las tesis de la técnica de ESTA rama que el adelanto no pudo traer (lo
    trae sin rama). Las de la reasunción (art. 93, fr. VI: 171925, 178784,
    182039 y 174177, comprobadas en el acervo el 28-sep-2026) sólo aplican
    cuando se revoca una concesión, y eso se sabe al resolver. Por registro: o
    existen con ese número o no viene nada, y el prompt sólo nombra las que
    llegaron (`fase6_estudio._bloque_tecnica`). Nunca lanza."""
    # SIN RAMA TAMBIÉN (30-sep-2026): en amparo directo la rama va vacía, y la
    # técnica de la sentencia en cumplimiento (`tipos_asunto.TECNICA_CUMPLIMIENTO`,
    # `solo_si_estan`) tiene que traer sus apoyos al resolver en sesiones cuyo
    # adelanto no los trajo. Sólo se piden las reglas marcadas: con la rama
    # vacía, en revisión no lo está ninguna.
    if qdrant is None:
        return
    try:
        import tipos_asunto as _ta_t
        import fase6_rag as _f6r_t
        _regs = [x for _regla in _ta_t.tecnica_de(getattr(material, "tipo_asunto", "") or "", rama)
                 if _regla.get("solo_si_estan") for x in (_regla.get("apoyos") or [])]
        _ya = {str(t.get("registro") or "") for t in (material.tesis or [])}
        _faltan = [x for x in _regs if x not in _ya]
        if not _faltan:
            return
        _nuevas = [t for t in await _f6r_t.tesis_por_registro(qdrant, _faltan)
                   if t["registro"] not in _ya]
        if _nuevas:
            material.tesis = list(material.tesis or []) + _nuevas
            print(f"   ⚖️ tesis de la técnica de la rama añadidas: "
                  f"{', '.join(t['registro'] for t in _nuevas)}")
    except Exception as _et:
        print(f"   ⚠️ no se pudieron añadir las tesis de la rama: {type(_et).__name__}")


# ═══ LAS TESIS DEL VICIO, ANTES DEL PLAN (2-oct-2026) ═══════════════════════
# David: «en la cita de jurisprudencias sobre inoperancia siempre es la misma».
# Hasta hoy la única tesis de inoperancia «del vicio» la traía /taller/razonar,
# pegándola al material EN LA MEMORIA del worker que atendía (gunicorn -w 2):
# llegaba al estudio sólo si el mismo worker resolvía, y el plan no la tenía en
# su índice, así que todos los apartados inoperantes acababan en la tesis
# genérica que trajo la co-citación. Ahora el resolver, ANTES del plan, trae
# 1-2 tesis por cada vicio presente en el criterio y las pone en una COPIA del
# material —nada en memoria— con cupo propio y marcadas `tecnica` y `vicio`:
# entran al índice del plan (`plan_estudio.indice_material`) y al bloque del
# estudio. Es determinista (similitud, sin rerank) para que la clave del plan
# salga igual en los dos workers.
CUPO_TESIS_POR_VICIO = 2
TOPE_VICIOS_POR_ASUNTO = 4
ESPERA_TESIS_DEL_VICIO_S = 12.0
# Sólo lo que se cita COMO registro —«registro 2012073», «(2012073)», que es
# como la razón lo escribe—: una cifra suelta de seis dígitos puede ser un
# importe o un expediente, y traerla por registro metería una tesis ajena.
_RX_REGISTRO = re.compile(r"registro(?:\s+digital)?(?:\s+n[úu]mero)?\s*:?\s*(\d{6,7})(?!\d)"
                          r"|\((\d{6,7})\)", re.I)


# LA RAZÓN CORTA ES LA DIRECTRIZ DEL SECRETARIO; LA LARGA, EL PÁRRAFO DEL MOTOR
# (revisión adversarial, 3-oct-2026). Lo que llega como razón casi nunca son
# las dos líneas que él escribe: es el párrafo de /taller/razonar (60-160
# palabras), la razón global del modo acervo (hasta 900 caracteres) o la del
# motor puesta sobre el principal. En ese texto largo cualquier pista casa con
# algo que no es el vicio, y su resultado pisaba el vicio que la fase 3 sí
# declaró. Ahora la fase 3 manda sobre el texto largo; la directriz corta del
# secretario, cuando nombra un vicio, sigue mandando sobre todo (él decide).
RAZON_CORTA_CHARS = 280


def _problema_de_la_fase3(r, problema: str) -> tuple:
    """(combate, resolvio, vicio declarado) del problema de la fase 3 que casa
    con `problema`; vacíos si no está. El vicio declarado es la clave del
    impedimento si es del catálogo; si no, el que nombra su explicación."""
    import vicio_inoperancia as _vi
    combate, resolvio, declarado = "", "", ""
    try:
        import arbol_decision as _ad
        k = _ad.clave_problema(problema or "")
        for p in (getattr(getattr(r, "fases", None), "problemas", None) or []):
            if isinstance(p, dict) and _ad.clave_problema(str(p.get("pregunta") or "")) == k:
                combate = str(p.get("combate") or "")
                resolvio = str(p.get("resolvio") or "")
                imp = p.get("impedimento")
                if isinstance(imp, dict):
                    declarado = str(imp.get("vicio") or "").strip().lower()
                    if declarado not in _vi.VICIOS:
                        declarado = _vi.vicio_de_texto(str(imp.get("explicacion") or ""))
                break
    except Exception:
        pass
    return combate, resolvio, declarado


def _tipo_de(r, tipo_asunto=None) -> str:
    if tipo_asunto is not None:
        return str(tipo_asunto or "")
    return str(getattr(getattr(r, "encargo", None), "tipo_asunto", "") or "")


def vicio_y_argumento(r, problema: str, sentido: str, razon: str = "", *,
                      tipo_asunto=None) -> tuple:
    """(vicio, argumento, resolvió) de un planteamiento calificado.

    El vicio, en este orden: el que nombra la directriz CORTA del secretario
    (él decide), el que la fase 3 declaró en el impedimento del problema, el
    que nombra la razón larga (el párrafo del motor) y el de la calificativa
    (la inoperancia a secas, `no_combate`). Sólo cuenta un vicio que exista en
    la vía (`vicio_inoperancia.cabe_en_la_via`, la misma partición que pide la
    fase 3). «» si la calificativa no es de técnica. El argumento es la
    pregunta y lo que la combate; lo que se resolvió, lo que dijo el órgano:
    con eso se ordena por pertinencia. El MISMO cálculo en /taller/razonar y
    antes del plan."""
    import vicio_inoperancia as _vi
    combate, resolvio, declarado = _problema_de_la_fase3(r, problema)
    razon = razon or ""
    del_texto = _vi.vicio_de_texto(razon)
    if len(" ".join(razon.split())) <= RAZON_CORTA_CHARS:
        candidatos = (del_texto, declarado)
    else:
        candidatos = (declarado, del_texto)
    tipo = _tipo_de(r, tipo_asunto)
    elegido = next((v for v in candidatos if v and _vi.cabe_en_la_via(v, tipo)), "")
    vicio = _vi.vicio_de(sentido, "", elegido)
    argumento = " ".join(f"{problema or ''} {combate}".split())
    return vicio, argumento, resolvio


def vicio_declarado(r, problema: str, *, tipo_asunto=None) -> str:
    """El vicio que la fase 3 declaró para el problema, si cabe en la vía; «»
    si no declaró ninguno. Para las inoperancias por SEGMENTO de la v4
    (3-oct-2026): un problema infundado puede tener argumentos que el plan
    califica inoperantes por ese vicio."""
    import vicio_inoperancia as _vi
    _, _, declarado = _problema_de_la_fase3(r, problema)
    if declarado and _vi.cabe_en_la_via(declarado, _tipo_de(r, tipo_asunto)):
        return declarado
    return ""


async def material_con_tesis_del_vicio(qdrant, embed_juris, r, material, criterios, *,
                                       tope_s: float = ESPERA_TESIS_DEL_VICIO_S,
                                       suplencia=None, por_segmento=None):
    """UNA COPIA del material con las tesis del vicio de cada criterio de
    técnica (inoperante, inatendible, ineficaz, innecesario, fundado pero
    insuficiente), o el MISMO material si no hay nada que añadir, la bandera
    `inoperancia_por_vicio` está apagada o la búsqueda no llega a tiempo.

    · Un vicio, una búsqueda (`fase6_rag.tesis_de_la_calificativa`), todas en
      paralelo y con tope; como mucho TOPE_VICIOS_POR_ASUNTO vicios y
      CUPO_TESIS_POR_VICIO tesis por vicio.
    · Con suplencia confirmada no se traen tesis de los vicios de FORMA (no
      combatir, accesoria, genérico, reiterar): esa inoperancia está prohibida.
    · Si una ya estaba en el material, se marca en su sitio (en la copia) en
      vez de duplicarla; si otro vicio ya la tomó, se pasa a la siguiente.
    · Los registros que la razón del criterio cita (la de /taller/razonar los
      tomó de estas mismas búsquedas) se traen por registro si faltan, salvo
      que hayan perdido vigencia: lo que la razón invoca tiene que estar en el
      material que verá el estudio. Pasan los mismos filtros que las de la
      búsqueda: la vía del rubro y lo que la evaluación excluye.
    · `suplencia`: la del formulario cuando quien llama la tiene (el pedido del
      plan, el precálculo con {}); None = la del encargo, que el resolver fija
      desde SU formulario. Así la clave del plan sale igual en los dos workers.
    · `por_segmento` (la v4; None = según la variante del encargo): también los
      problemas cuyo criterio no es de técnica (infundado, fundado) pero cuyo
      impedimento de la fase 3 declara un vicio, porque el plan v4 califica
      inoperantes ARGUMENTOS dentro de ellos (3-oct-2026).
    Nunca lanza."""
    import vicio_inoperancia as _vi
    if material is None or not criterios or not _vi.activa():
        return material
    try:
        return await asyncio.wait_for(
            _material_con_tesis_del_vicio(qdrant, embed_juris, r, material, criterios,
                                          suplencia=suplencia, por_segmento=por_segmento),
            timeout=tope_s)
    except asyncio.TimeoutError:
        print(f"   ⚠️ tesis del vicio: no llegaron en {tope_s:.0f} s; el estudio sigue sin ellas")
    except Exception as _e:
        print(f"   ⚠️ tesis del vicio: {type(_e).__name__}; el estudio sigue sin ellas")
    return material


async def _material_con_tesis_del_vicio(qdrant, embed_juris, r, material, criterios, *,
                                        suplencia=None, por_segmento=None):
    import copy as _copy
    import vicio_inoperancia as _vi
    e = getattr(r, "encargo", None)
    tipo = str(getattr(e, "tipo_asunto", "") or getattr(material, "tipo_asunto", "") or "")
    try:
        import suplencia as _sp
        _sup_d = (getattr(e, "suplencia", None) if suplencia is None else suplencia) or {}
        _supl = bool(_sp.confirmada(_sup_d))
    except Exception:
        _supl = False
    if por_segmento is None:
        try:
            import fase6_estudio as _f6s
            por_segmento = _f6s.normalizar_variante(
                getattr(e, "variante_estudio", "") or "", "") == "v4"
        except Exception:
            por_segmento = False
    # El principal primero: si hay que recortar vicios, se recorta lo accesorio.
    _orden = sorted([c for c in criterios if c is not None],
                    key=lambda c: str(getattr(c, "jerarquia", "") or "") != "principal")
    por_vicio: dict = {}
    citados: list = []
    for c in _orden:
        vicio, arg, res_ = vicio_y_argumento(r, str(getattr(c, "problema", "") or ""),
                                             str(getattr(c, "sentido", "") or ""),
                                             str(getattr(c, "razonamiento", "") or ""),
                                             tipo_asunto=tipo)
        if not vicio or (_supl and vicio in _vi.VICIOS_DE_FORMA):
            continue
        g = por_vicio.setdefault(vicio, {"args": [], "res": [],
                                         "calif": str(getattr(c, "sentido", "") or "")})
        g["args"].append(arg)
        g["res"].append(res_)
        for _m in _RX_REGISTRO.findall(str(getattr(c, "razonamiento", "") or "")):
            reg = _m[0] or _m[1]
            if reg not in [x for x, _ in citados]:
                citados.append((reg, vicio))
    # LAS INOPERANCIAS POR SEGMENTO DE LA v4 (revisión adversarial,
    # 3-oct-2026): el plan califica inoperantes ARGUMENTOS dentro de un problema
    # infundado o fundado (deriva_de_desestimado, novedoso, falsa_premisa…), y
    # sin esto no tenían ninguna tesis de su vicio: la regla v4 manda razonar
    # sin cita si el material no la trae. Se busca el vicio que la fase 3
    # DECLARÓ para el problema, con calificativa «inoperante», DESPUÉS de los
    # criterios de técnica (si hay que recortar, se recorta esto) y sin sumar
    # su argumento a un vicio que ya buscó un criterio de técnica: esa búsqueda
    # queda como estaba. Determinista: entra al índice del plan y a su clave.
    # Los vicios que sólo elige el planificador sin impedimento de la fase 3
    # no se cubren aquí (harían falta después del plan, fuera de la clave).
    if por_segmento:
        for c in _orden:
            _sent = str(getattr(c, "sentido", "") or "")
            if _vi.vicio_de(_sent):
                continue
            _dec = vicio_declarado(r, str(getattr(c, "problema", "") or ""), tipo_asunto=tipo)
            if not _dec or _dec in por_vicio or (_supl and _dec in _vi.VICIOS_DE_FORMA):
                continue
            _v_seg = _vi.vicio_de("inoperante", "", _dec)
            _, arg, res_ = vicio_y_argumento(r, str(getattr(c, "problema", "") or ""),
                                             "inoperante", "", tipo_asunto=tipo)
            por_vicio[_v_seg] = {"args": [arg], "res": [res_], "calif": "inoperante"}
    if not por_vicio:
        return material
    vicios = list(por_vicio)[:TOPE_VICIOS_POR_ASUNTO]
    tesis_m = list(getattr(material, "tesis", None) or [])
    ya = {str(t.get("registro") or ""): i for i, t in enumerate(tesis_m) if isinstance(t, dict)}

    async def _de(v):
        g = por_vicio[v]
        return await f6rag.tesis_de_la_calificativa(
            qdrant, embed_juris, g["calif"], "", CUPO_TESIS_POR_VICIO + 4, vicio=v,
            argumento=" ".join(g["args"])[:600], resolvio=" ".join(g["res"])[:600],
            tipo_asunto=tipo)

    listas = await asyncio.gather(*[_de(v) for v in vicios], return_exceptions=True)
    nuevas, marcadas, tomadas, resumen = [], {}, set(), []
    for v, lista in zip(vicios, listas):
        if isinstance(lista, Exception) or not lista:
            continue
        n = 0
        for t in lista:
            reg = str(t.get("registro") or "")
            if not reg or reg in tomadas:
                continue
            tomadas.add(reg)
            if reg in ya:
                marcadas[ya[reg]] = (v, t.get("de_la_calificativa"))
            else:
                nuevas.append(t)
            n += 1
            resumen.append(f"{v}→{reg}")
            if n >= CUPO_TESIS_POR_VICIO:
                break
    # Lo que cita la razón y falta: por registro (o existe o no viene nada).
    _faltan = [(reg, v) for reg, v in citados if reg not in ya and reg not in tomadas]
    if _faltan:
        try:
            _traidas = {t["registro"]: t for t in await f6rag.tesis_por_registro(
                qdrant, [reg for reg, _ in _faltan[:2 * TOPE_VICIOS_POR_ASUNTO]])}
        except Exception:
            _traidas = {}
        # LOS MISMOS FILTROS QUE LAS DE LA BÚSQUEDA (revisión adversarial,
        # 3-oct-2026): «UN SOLO FILTRO para todas las entradas de tesis». Una
        # tesis de otro recurso, o en una corrida del banco una publicada
        # después del corte, no entra al material marcada como del vicio.
        try:
            import contexto_taller as _ct_tv
            _pasan = {str(t.get("registro") or "")
                      for t in _ct_tv.filtrar_tesis(list(_traidas.values()))}
        except Exception:
            _pasan = set(_traidas)
        for reg, v in _faltan:
            t = _traidas.get(reg)
            if (t is None or f6rag._perdio_vigencia(t) or reg not in _pasan
                    or not f6rag.apta_para_el_recurso(t.get("rubro", ""), tipo)):
                continue
            t["vicio"] = v
            t["de_la_calificativa"] = por_vicio.get(v, {}).get("calif") or v
            nuevas.append(t)
            tomadas.add(reg)
            resumen.append(f"{v}→{reg} (citada en la razón)")
    if not nuevas and not marcadas:
        return material
    copia = _copy.copy(material)
    lista_tesis = []
    for i, t in enumerate(tesis_m):
        if i in marcadas:
            t = dict(t)
            t["tecnica"] = True
            t["vicio"], _cal = marcadas[i]
            t["de_la_calificativa"] = _cal
        lista_tesis.append(t)
    copia.tesis = lista_tesis + nuevas
    # HIGIENE DE REGISTROS: sólo vicios y registros, nada del asunto.
    print(f"   ⚖️ tesis del vicio antes del plan: {', '.join(resumen)}")
    return copia


def _formato_al_material(r, material, cliente=None, criterios=None) -> None:
    """La forma de la sentencia y el reparto de la fase 3, al material.

    Aquí y no en cada redactor: `_litis_y_material` es el único paso por el
    que pasan los dos antes de escribir el estudio. Si es la versión moderna,
    arranca también la síntesis de los resúmenes, EN PARALELO al estudio —no
    alarga la espera— y deja la tarea colgada del material para `_terminar`.
    Sin cliente no hay síntesis y van los resúmenes completos: nunca falta
    nada por no haber condensado."""
    try:
        e = getattr(r, "encargo", None)
        # LA SUPLENCIA, POR EL MISMO CAMINO Y SIEMPRE —y lo primero, para que
        # ningún fallo de lo que sigue la deje a medias—: el material vive en la
        # sesión de una generación a la siguiente, y la suplencia confirmada de
        # la vuelta anterior no puede colarse en ésta si el secretario la quitó.
        material.suplencia = dict(getattr(e, "suplencia", None) or {}) if e else {}
        # LA REASUNCIÓN DE JURISDICCIÓN, POR EL MISMO CAMINO Y SIEMPRE (art. 93,
        # frs. I, V y VI; AR 631/2025, 28-sep-2026). Se vacía primero —None = no
        # calculada, y el estudio decide por la rama— y se calcula en cada
        # petición: el material vive en la memoria del worker.
        material.reasuncion = None
        material.reasuncion = _reasuncion_del_asunto(r, criterios)
        # LA FICHA PROCESAL, POR EL MISMO CAMINO Y SIEMPRE (SPEC_E2): se vacía
        # y se vuelve a armar en cada petición —es pura y sin modelo— para que
        # el estudio la lea en su encabezado de datos.
        material.ficha_procesal = ""
        try:
            import ficha_procesal as _fp_m
            material.ficha_procesal = _fp_m.bloque(_fp_m.de_resultado(r))
        except Exception as _efp:
            print(f"   ⚠️ FICHA PROCESAL: no se pudo poner en el material: {type(_efp).__name__}")
        # Y EL INVENTARIO SE VACÍA AQUÍ, antes de nada que pueda fallar: el de
        # la vuelta anterior no puede sobrevivir a una excepción de más abajo.
        material.inventario = []
        import formato_sentencia as _fs
        material.formato = _fs.normalizar(getattr(e, "formato", "") if e else "")
        # LA VARIANTE DEL PROMPT, EN CADA PETICIÓN, como la forma: el material
        # vive en la memoria del worker y una v2 de la vuelta anterior no puede
        # colarse en la v1 de ésta. Sin variante en el encargo, la global.
        import fase6_estudio as _f6v
        material.variante = _f6v.normalizar_variante(
            getattr(e, "variante_estudio", "") if e else "",
            _f6v.variante_global(getattr(e, "tipo_asunto", "") if e else ""))
        material.problemas = [p for p in (getattr(r.fases, "problemas", None) or [])
                              if isinstance(p, dict)]
        # EL INVENTARIO DE ARGUMENTOS, SÓLO PARA LA v3 Y LA v4 (Paso 2a,
        # 26-sep-2026). Se calcula en CADA petición —y se vacía en las demás
        # variantes— por lo mismo que la variante: el material vive en la
        # memoria del worker. Es determinista y sin modelo (medido: 0.08 s de
        # mediana, 0.65 s el escrito más largo de las 64 sesiones con escrito).
        # Lo que falle aquí deja la v3 escribiendo como la v2, nunca sin estudio.
        material.inventario = []
        if _f6v.con_inventario(material):
            try:
                import inventario as _inv_m
                _esc_m = (list(getattr(r.fases, "fuentes", []) or []) + ["", ""])[1]
                # Con la LECTURA DEL ESCRITO que fijó el resolver en el
                # encargo (vacía si no llegó: entonces el piso, como antes).
                # La misma lista que vio el plan: su clave lleva los segmentos.
                material.inventario = _inv_m.segmentos(
                    r.fases, _esc_m, bool(getattr(e, "es_recurso", False)) if e else False,
                    extraidos=list(getattr(e, "inventario_escrito", None) or []) if e else [])
                _conc_m = sorted({x["concepto"] for x in material.inventario})
                print(f"   🧭 INVENTARIO: {len(material.inventario)} argumentos en "
                      f"{len(_conc_m)} concepto(s) · "
                      f"{sum(1 for x in material.inventario if x.get('cita'))} anclados en el escrito · "
                      f"{sum(1 for x in material.inventario if x.get('origen') == 'escrito')} "
                      f"añadidos por la lectura del escrito")
            except Exception as _ei:
                material.inventario = []
                print(f"   ⚠️ INVENTARIO: no se pudo armar: {type(_ei).__name__}")
        _c = getattr(r.fases, "conteo", None) or {}
        material.n_planteamientos = int(_c.get("n") or 0) \
            if str(_c.get("estado")) == "contado" else 0
        material.sintesis = None
        if material.formato == _fs.MODERNA and cliente is not None:
            material.sintesis = asyncio.ensure_future(
                _sintetizar_moderna(cliente, r, criterios or [], material))
        print(f"   📐 FORMATO: {_fs.rotulo(material.formato)} · "
              f"{material.n_planteamientos} planteamientos contados · "
              f"prompt {material.variante}")
    except Exception as _ef:
        print(f"   ⚠️ FORMATO: no se pudo fijar: {type(_ef).__name__}: {_ef}")


async def _sintetizar_moderna(cliente, r, criterios, material) -> dict:
    """Antecedentes, lo resuelto y los planteamientos, condensados.

    Una llamada al motor de las fases —barato y rápido— con la orden de no
    inventar y de conservar todos los planteamientos con su ordinal; lo que
    vuelve se comprueba apartado por apartado y lo que no pasa se queda como
    estaba. Devuelve {antecedentes, acto, conceptos} listos para el relleno."""
    import formato_sentencia as _fs
    import tipos_asunto as _ta_s
    f = r.fases
    ante, acto, conc = (f.parrafos_antecedentes(), f.parrafos_acto(),
                        f.parrafos_conceptos())
    base = {"antecedentes": ante, "acto": acto, "conceptos": conc}
    try:
        e = r.encargo
        _t = getattr(e, "tipo_asunto", "") or "amparo_directo"
        organo = _ta_s.sujetos_de(_t)["organo"][0]
        q = "agravios" if e.es_recurso else "conceptos de violación"
        t0 = _time.perf_counter()
        crudo = await f123._pedir(cliente, _fs.prompt_sintesis(
            "\n".join(ante), "\n".join(acto), "\n".join(conc),
            material.problemas, criterios, organo, q,
            material.n_planteamientos), 6000, json_estricto=True)
        fuera, notas = _fs.validar_sintesis(
            _fs.leer_json(crudo), ante, acto, conc, material.n_planteamientos)
        # LO QUE SE CONDENSA NO PUEDE TRAER ARCHIVO NI META-LENGUAJE: la misma
        # limpieza que pasan los resúmenes de la fase 1-2.
        try:
            import meta_lenguaje as _ml_s
            for k in ("antecedentes", "acto", "conceptos"):
                if fuera[k] is not base[k]:
                    fuera[k] = [x for x in (_ml_s.limpiar(y)[0] for y in fuera[k]) if x.strip()]
        except Exception:
            pass
        def _p(x):
            return sum(len(str(y).split()) for y in x)
        print(f"   📐 SÍNTESIS MODERNA en {_time.perf_counter() - t0:.1f}s: "
              f"antecedentes {_p(ante)}→{_p(fuera['antecedentes'])} · "
              f"acto {_p(acto)}→{_p(fuera['acto'])} · "
              f"planteamientos {_p(conc)}→{_p(fuera['conceptos'])}"
              + (f" · {'; '.join(notas)}" if notas else ""))
        return fuera
    except Exception as _es:
        print(f"   ⚠️ SÍNTESIS MODERNA: {type(_es).__name__}: {_es} — van los resúmenes completos")
        return base


def _litis_y_material(r, material, avisos: list, cliente=None,
                      criterios=None) -> list:
    """La litis del asunto, y el material ya sin ley local ajena a ella.

    Lo que no puede citarse no se le enseña al modelo: la guarda del final
    corrige lo que se escape, pero la mejor cita mala es la que nunca se
    escribe. Devuelve la litis para que `_terminar` la reutilice.

    Y LA FORMA DE LA SENTENCIA, porque éste es el único paso común a los dos
    redactores antes del estudio. Ver `_formato_al_material`.
    """
    _formato_al_material(r, material, cliente, criterios)
    try:
        import litis_normativa as _ln
        litis = _ln.leyes_de_la_litis(getattr(r, "fases", None))
        buenas, fuera = _ln.filtrar_normas(getattr(material, "normas", None) or [], litis)
        if fuera:
            material.normas = buenas
            print(f"   ⚖️ LITIS: fuera del material {len(fuera)} precepto(s) de ley "
                  f"local que ni el acto ni el escrito invocan: {fuera[:4]}")
        return litis
    except Exception as _el:
        print(f"   ⚠️ LITIS: no se pudo acotar el material: {type(_el).__name__}")
        return []

def _rama_de(r, criterios, estudio: str = "") -> str:
    """La rama de la revisión con UNA fuente de verdad (28-sep-2026): lo que
    hizo el juzgado con el orden de `fase_rama.que_hizo_el_juzgado` —su punto
    resolutivo primero (577c700)— y el sentido de los criterios; con el
    `estudio` ya escrito, además, lo que concluyó al reasumir jurisdicción
    (`fase_rama.sentido_en_plenitud`), que es lo que decide el segundo punto al
    levantar un sobreseimiento o al revocar una concesión (art. 93, frs. V y
    VI). «» fuera de la revisión o si falla. Sin modelo."""
    try:
        import tipos_asunto as _ta_r, fase_rama as _fr_r
        e = getattr(r, "encargo", None)
        if e is None or _ta_r.normalizar(getattr(e, "tipo_asunto", "")) != "amparo_revision":
            return ""
        _que = _fr_r.que_hizo_el_juzgado(r.fases, getattr(e, "resolvio_declarado", "") or "")
        _sent = "fundado" if any(_ta_r.prospera(str(getattr(c, "sentido", "")))
                                 for c in (criterios or [])) else "infundado"
        # QUIÉN RECURRE Y SI LO QUE PROSPERA ES LA PROCEDENCIA (revisión del
        # 28-sep-2026): la quejosa que gana su recurso contra una concesión no
        # pierde el amparo (fr. V); la improcedencia que prospera sobresee (fr.
        # II). La misma lectura que la tarjeta y la pantalla.
        _quien = papel_del_recurrente(e, getattr(r, "partes", None),
                                      _resolutivo_del_a_quo(getattr(r, "fases", None)))
        return _ta_r.rama_revision(
            _que, _sent,
            sentido_amparo=_fr_r.sentido_en_plenitud(str(estudio or "")) if estudio else "",
            quien_recurre=_quien, procedencia=_procedencia_prospera(r, criterios))
    except Exception as _e:
        print(f"   ⚠️ TALLER: no se pudo fijar la rama: {type(_e).__name__}")
        return ""


def _rama_tecnica(r, criterios, contexto: str = "") -> tuple:
    """(rama, violación procesal) ANTES de redactar, para los dos gemelos.

    LA RAMA TÉCNICA, ANTES DE REDACTAR. Se calculaba al COMPONER, con el
    estudio ya escrito, así que el modelo nunca supo en qué escenario estaba.
    Y LA PRIMERA VEZ QUE LO PUSE SE ME FUE DE ÁMBITO: quedó en la función de
    al lado y `resolver_en_vivo` —que es por donde pasa TODA la pantalla— lo
    usaba sin tenerlo; cada generación moría con «name '_rama' is not
    defined». Por eso vive aquí, una vez, y los dos la llaman (28-sep-2026)."""
    _rama, _vp = _rama_de(r, criterios), False
    try:
        # LA VIOLACIÓN PROCESAL SE RECONOCE POR LO QUE SE COMBATE, no por que
        # la pregunta diga «violación procesal»: la del 93/2026 decía «¿debió
        # admitir la ampliación de demanda…?» y esta marca no la veía, así que
        # el estudio no recibió la técnica de los artículos 171 y 172.
        import violacion_procesal as _vpm
        _vp = _vpm.hay(list(getattr(r.fases, "problemas", None) or []),
                       criterios, contexto)
    except Exception as _e:
        print(f"   ⚠️ TALLER: no se pudo fijar la rama técnica: {type(_e).__name__}")
    return _rama, _vp


def _aviso_tardio_visible() -> bool:
    """El aviso de «justificación pendiente» se ENSEÑA sólo con la bandera
    «fuente_tardia_aviso» (FUENTE_TARDIA_AVISO = todos | casa | 0; por omisión
    casa): David, decisión 2 del 29-sep, nada se activa para todos antes de
    calibrarlo contra engroses buenos. El meta lo registra siempre."""
    import contexto_taller as _ct
    return _ct.bandera("fuente_tardia_aviso", "FUENTE_TARDIA_AVISO", "casa")


async def _fuentes_tardias(r, e, material, estudio: str, avisos: list, qdrant,
                          meta: dict = None) -> list:
    """Lo que entra DESPUÉS de redactar el estudio, y qué toca.

    Era un bloque copiado igual en `resolver` y en `resolver_en_vivo` (medido
    el 29-sep-2026): trae del acervo las tesis citadas por registro y los
    preceptos citados sin tenerlos, y ajusta los avisos. Ahora, además
    (rediseño, punto 7), CLASIFICA lo traído con `fuente_tardia`: si una fuente
    tardía se usa en una unidad que contesta un argumento, expone una premisa o
    fija un efecto, la unidad se escribió sin su texto a la vista y el proyecto
    sale como «justificacion_pendiente», con las unidades nombradas, en vez de
    pasar por verificado. Devuelve los avisos (la lista se reasigna dentro).
    """
    _t0 = {str(t.get("registro") or "") for t in (material.tesis or []) if isinstance(t, dict)}
    _n0 = {(str(n.get("articulo")), str(n.get("cuerpo_legal") or n.get("fuente") or ""))
           for n in (material.normas or []) if isinstance(n, dict)}
    # LOS PRECEPTOS QUE EL ESTUDIO CITÓ SIN TENERLOS. Si están en el acervo se
    # traen y se transcriben; el aviso se queda sólo para los que no existen.
    try:
        import fase6_rag as _f6r
        # LAS TESIS QUE EL ESTUDIO NOMBRA POR REGISTRO Y LAS QUE LA PARTE
        # INVOCÓ, si no están en el material, se traen del acervo por su
        # registro o su clave: así el compositor las anuncia con su rubro y
        # baja su ficha al pie en vez de dejarlas en prosa (61/2025: cuatro
        # criterios de la Segunda Sala «de rubro «…», registro N» sin ficha).
        if qdrant is not None:
            try:
                import fases123_pipeline as _f123c
                _esc = (list(getattr(r.fases, "fuentes", []) or []) + ["", ""])[1]
                _citas = _f123c.citas_invocadas(_esc) | _f123c.citas_invocadas(str(estudio or ""))
                _nuevas = await _f6r.completar_tesis_citadas(
                    qdrant, material, sorted(_citas),
                    tipo_asunto=getattr(e, "tipo_asunto", "") or "")
                if _nuevas:
                    avisos.append(f"{len(_nuevas)} tesis citadas se trajeron del acervo "
                                  f"para verificarlas y anunciarlas con su ficha: "
                                  f"{', '.join(_nuevas[:6])}{'…' if len(_nuevas) > 6 else ''}.")
                    # Y SE RETIRA EL AVISO QUE ACABA DE QUEDAR FALSO. `revisar`
                    # corrió ANTES que esto y denunció esos registros como «no
                    # están en el material: no se citan hasta comprobarlos»; y
                    # aquí se acaban de traer, verificar y fichar. El proyecto
                    # salía con los dos avisos, uno contra el otro, y el
                    # secretario no tiene con qué saber cuál vale. Medido en la
                    # revisión fiscal 2/2026 con el registro 167062, que el
                    # documento final SÍ cita, con su ficha completa al pie.
                    avisos[:] = [_a for _a in avisos
                                 if not (_a.startswith("REGISTROS QUE NO ESTÁN EN EL MATERIAL")
                                         and _registros_ya_estan(_a, material))]
            except Exception as _ex2:
                print(f"   ⚠️ no se pudieron completar las tesis citadas: {_ex2}")
        _pares = sorted(f6.preceptos_fuera(estudio, material)[1])
        if _pares and qdrant is not None:
            _traidos = await _f6r.completar_preceptos(
                qdrant, material, _pares, getattr(e, "coleccion_estatal", "") or None,
                materia=str(getattr(e, "materia", "") or ""),
                tipo_asunto=getattr(e, "tipo_asunto", "") or "")
            if _traidos:
                # SE LE PREGUNTA OTRA VEZ AL MATERIAL, NO SE COMPARAN CADENAS.
                # Esto decía `f"art. {x[1]} — {x[0]}" not in _traidos`, y
                # comparaba el nombre CITADO por el estudio —«ley del seguro
                # social», en minúsculas— contra el nombre OFICIAL del acervo
                # —«Ley del Seguro Social»—, que no coinciden nunca. Medido en
                # la revisión fiscal 2/2026: los artículos 17 y 251 SÍ se
                # trajeron, y el aviso los siguió acusando como ausentes junto
                # al que de verdad faltaba. Un aviso que acusa a los inocentes
                # enseña a no leer los avisos. El material ya completado es la
                # única fuente de verdad sobre qué sigue faltando.
                _quedan = sorted(f6.preceptos_fuera(estudio, material)[1])
                # Y LO QUE VINO DE INTERNET SE DICE APARTE. Al recalcular el
                # aviso desaparecía el de «preceptos que no están en el
                # material» —correcto, porque ya están— pero nadie decía que
                # algunos se habían transcrito de un sitio web. El hueco
                # tapado en silencio es peor que el hueco declarado.
                _web_p = list(getattr(material, "preceptos_de_internet", []) or [])
                if _web_p:
                    avisos.append(
                        f"{len(_web_p)} PRECEPTO(S) NO ESTÁN EN EL ACERVO y se "
                        f"transcribieron de su fuente oficial en línea: "
                        f"{', '.join(_web_p[:4])}"
                        f"{'…' if len(_web_p) > 4 else ''}. La nota al pie dice "
                        f"de dónde salió cada uno. COTÉJALOS antes de firmar: "
                        f"no pasaron por la verificación del acervo.")
                avisos = [a for a in avisos
                          if not str(a).startswith("PRECEPTOS CITADOS QUE NO ESTÁN")]
                if _quedan:
                    avisos.append("PRECEPTOS CITADOS QUE NO ESTÁN EN EL MATERIAL: "
                                  + str(sorted(f"art. {a} — {c}" for c, a in _quedan))
                                  + ". Compruébalos antes de firmar.")
    except Exception as _ex:
        print(f"   ⚠️ no se pudieron completar los preceptos citados: {_ex}")
    # LO TRAÍDO, CON LA FUERZA PARA ESTE TRIBUNAL Y SIN LO EXCLUIDO (revisión
    # del 29-sep): completar_tesis_citadas las anota sin tribunal.
    try:
        import fuerza_juridica as _fj_t
        import contexto_taller as _ct_t
        material.tesis = _ct_t.filtrar_tesis(material.tesis)
        _fj_t.anotar(material.tesis, getattr(material, "tribunal", "") or getattr(e, "tribunal", "") or "")
    except Exception as _eat:
        print(f"   ⚠️ tesis tardías sin anotar: {type(_eat).__name__}")
    # ═══ ¿TOCA UNA PREMISA? (rediseño, punto 7) ═══
    try:
        import fuente_tardia as _ft
        _tn = [t for t in (material.tesis or []) if isinstance(t, dict)
               and str(t.get("registro") or "") not in _t0]
        # QUIÉN LA CITÓ, DE VERDAD: `completar_tesis_citadas` marca todas como
        # «de la parte»; desde la decisión 3 las de la parte llegan al
        # consultar, así que las tardías son casi siempre del estudio.
        try:
            import fases123_pipeline as _f123p
            _esc_p = (list(getattr(r.fases, "fuentes", []) or []) + ["", ""])[1]
            _de_parte = {re.sub(r"\s+", "", str(x)).upper() for x in _f123p.citas_invocadas(_esc_p)}
            for _t in _tn:
                _t["citada_por_la_parte"] = (
                    str(_t.get("registro") or "") in _de_parte
                    or re.sub(r"\s+", "", str(_t.get("clave") or "")).upper() in _de_parte)
        except Exception:
            pass
        _nn = [n for n in (material.normas or []) if isinstance(n, dict)
               and (str(n.get("articulo")), str(n.get("cuerpo_legal") or n.get("fuente") or "")) not in _n0]
        if _tn or _nn:
            _cl = _ft.clasificar(estudio, _tn, _nn)
            _av_ft, _estado_ft = _ft.informe_y_aviso(_cl)
            if isinstance(meta, dict):
                meta["fuentes_tardias"] = list(meta.get("fuentes_tardias") or []) + _cl
                if _estado_ft:
                    meta["estado_salida"] = _estado_ft
            if _av_ft and _aviso_tardio_visible():
                avisos.insert(0, _av_ft)
            print(f"   🕰️ fuentes tardías: {len(_cl)} "
                  f"({sum(1 for c in _cl if c['sustantiva'])} en unidades que deciden)")
    except Exception as _eft:
        print(f"   ⚠️ fuentes tardías sin clasificar: {type(_eft).__name__}")
    return avisos


async def resolver(cliente, r: Resultado, criterios: list[f6.Criterio],
                   material: f6.Material, ruta_salida: str,
                   marco: str = "", qdrant=None, contexto: str = "") -> Resultado:
    """La sentencia: el mismo documento, ahora con el estudio de fondo dentro.

    Se REENSAMBLA desde la plantilla en vez de editar el adelanto, porque el
    ensamblador trabaja sobre los formatos de la plantilla original y aplicarlo
    dos veces sobre su propia salida duplica bloques.
    """
    if r.encargo is None:
        raise ValueError("El resultado no trae el encargo: no se puede reensamblar.")
    e = r.encargo
    # LA RAMA TÉCNICA, ANTES DE REDACTAR. Se calculaba al COMPONER, con el
    # estudio ya escrito, así que el modelo nunca supo en qué escenario estaba.
    #
    # Y LA PRIMERA VEZ QUE LO PUSE SE ME FUE DE ÁMBITO: quedó en la función de
    # al lado y `resolver_en_vivo` —que es por donde pasa TODA la pantalla— lo
    # usaba sin tenerlo. Cada generación moría con «name '_rama' is not
    # defined»; el servidor devolvía 200 con su evento de error y la pantalla
    # se quedaba muda. Lo vieron los registros de Render, no el guardián.
    _rama, _vp = _rama_tecnica(r, criterios, contexto)


    # EL MARCO SE ESCRIBE A LA VEZ QUE EL ESTUDIO. Son dos llamadas
    # independientes —la del marco sólo mira el material constitucional, la del
    # estudio mira el caso— y ponerlas en paralelo hace que el marco no cueste
    # un segundo de espera. Que APAREZCA ya no depende de que el modelo del
    # estudio se acuerde de escribirlo: lo coloca el compositor.
    # SIN APARTADO DE MARCO JURÍDICO, EN NINGUNA DE LAS DOS FORMAS. David,
    # 25-sep-2026: «el marco jurídico se me hace innecesario ya que en cada
    # caso se cita ley, jurisprudencia para resolver. Hay que prescindir del
    # marco jurídico». Medido en el ADC 93/2026: 791-873 palabras de repaso
    # constitucional y convencional que no cambiaban ninguna respuesta, y que
    # transcribían ley estatal de Querétaro en un juicio federal. El material
    # constitucional sigue llegando al estudio: se cita donde decide.
    # `redactar_marco` y la poda de `marco_escrito` en `_terminar` se quedan
    # vivas pero sin llamada; `tarea_marco` va siempre en None.
    tarea_marco = None


    _litis_y_material(r, material, [], cliente, criterios)
    # LAS TESIS DE LA TÉCNICA DE ESTA RAMA (art. 93, fr. VI), que el adelanto
    # no podía traer porque aún no sabía la rama (28-sep-2026).
    await _tesis_de_la_tecnica(qdrant, material, _rama)
    # LO QUE SE ANOTA DEL ESTUDIO, igual que el gemelo en vivo: la variante,
    # el `finish_reason` y los tokens. Va a la ficha por `_terminar`.
    _meta = {}
    with cronometrar("estudio de fondo"):
        estudio, advertencias, avisos = await f6.redactar(
            cliente, r.fases.resumen_acto, r.fases.resumen_conceptos,
            criterios, material, e.es_recurso, r.partes, marco, contexto,
            # LO QUE YA SE DECIDIÓ AL PROPONER: de qué problema cuelga el
            # resultado, qué les pasa a los demás, y la objeción más seria.
            # Se calculaba, se enseñaba en pantalla y no llegaba hasta aquí:
            # `Global.bloque()` no lo llamaba nadie.
            propuesta_global=getattr(e, "propuesta_global", None),
            rama=_rama, violacion_procesal=_vp,
            conceptos_violacion=getattr(e, "conceptos_violacion", "") or "",
            # EL ESCRITO DE LA PARTE, LITERAL. `fuentes[1]` es el texto del
            # recurso o de la demanda tal como se leyó del PDF. Hasta ahora
            # moría en el adelanto: la fase que CONTESTA los conceptos recibía
            # el resumen —unas 472 palabras— y CERO caracteres del escrito.
            escrito_literal=(list(getattr(r.fases, "fuentes", []) or []) + ["", ""])[1],
            meta=_meta,
            # EL GUION DEL PLAN (v4): lo puso `main._taller_plan_para` en el
            # encargo de ESTA petición; vacío fuera de la v4.
            guion=str(getattr(e, "guion", "") or ""))
    # LA CALIFICACIÓN SUELTA, A SU APERTURA ANTES DE LA REPARACIÓN DIRIGIDA
    # (revisión adversarial de p2-congruencia): la reparación inserta cada
    # pieza tras el último párrafo que marca su argumento, y si ése era la
    # apertura, la pieza quedaba entre la apertura y su «Es fundado.», que
    # después se pegaba a la pieza. Sin modelo; sólo la familia v2.
    estudio = _congruencia_pegar(material, estudio, _meta)
    # LA REPARACIÓN DIRIGIDA (v3/v4, p2-exhaustivo): lo mismo que el gemelo en
    # vivo, en el mismo sitio —antes de los efectos, las constancias y los
    # preceptos, que leen el estudio ya completado—.
    _faltan = _por_completar(material, criterios, estudio, _plan_del_estudio(r),
                             _concede_del_estudio(r, criterios))
    if _faltan:
        with cronometrar("completar el estudio"):
            estudio = await _completar_estudio(cliente, r, criterios, material, estudio,
                                               _faltan, _meta, avisos)
    # LA CALIFICACIÓN AL ABRIR (familia v2, p2-congruencia): sin modelo, igual
    # que en el gemelo en vivo.
    estudio = _congruencia_apertura(r, criterios, material, estudio, _meta, avisos)
    # EL SUPERVISOR DEL PROYECTO (2-oct-2026, David: «un LLM o agente que, sin
    # tanto costo, revise el proyecto y ajuste sus errores»). Aquí, con el
    # estudio completo y sus marcas, y ANTES de los efectos, las fuentes
    # tardías y `_terminar`: todo lo que viene después re-valida sin modelo el
    # texto corregido. Bandera `supervisor_proyecto`; apagada, no se llama.
    import supervisor_proyecto as _sup
    if _sup.activo():
        with cronometrar("supervisor"):
            estudio = await _sup.en_el_resolver(cliente, r, criterios, material, estudio,
                                                _meta, avisos, contexto)
    # LOS EFECTOS DE UNA VIOLACIÓN PROCESAL SE ORDENAN PASO A PASO (v5 del
    # 93/2026: «dicte otra» sobre una reposición). Se comprueba aquí porque
    # aquí se sabe si la hay.
    # CON LA RAMA, que ya lee lo que el estudio concluyó (28-sep-2026): en un
    # «revoca y niega» no hay efectos que ordenar (AR 631/2025).
    _av_ef = f6._efectos_de_reposicion(estudio, criterios, _vp,
                                       rama=_rama_de(r, criterios, estudio))
    if _av_ef:
        avisos.insert(0, _av_ef)
    # LAS CONSTANCIAS INDISPENSABLES QUE NO SE APORTARON, dichas arriba del
    # todo: el proyecto se escribió sin verlas.
    try:
        import constancias as _cn_a
        _pg_cn = getattr(e, "propuesta_global", None) or {}
        _pral_cn = next((c for c in (criterios or [])
                         if str(getattr(c, "jerarquia", "")).lower() == "principal"),
                        (criterios or [None])[0])
        _al_reves_cn = bool(_pral_cn is not None and _pg_cn.get("sentido")
                            and not f6._misma_direccion(
                                str(_pg_cn.get("sentido") or ""),
                                str(getattr(_pral_cn, "sentido", "") or "")))
        _av_cn = _cn_a.aviso_faltantes(
            _pg_cn.get("constancias") or [], contexto, al_reves=_al_reves_cn)
        if _av_cn:
            avisos.insert(0, _av_cn)
    except Exception as _exc_cn:
        print(f"   ⚠️ constancias: {type(_exc_cn).__name__}")

    avisos = await _fuentes_tardias(r, e, material, estudio, avisos, qdrant, _meta)
    return await _terminar(cliente, r, e, criterios, material, estudio,
                           advertencias, avisos, tarea_marco, ruta_salida, qdrant, marco,
                           contexto, meta_estudio=_meta)


async def resolver_en_vivo(cliente, r: Resultado, criterios: list[f6.Criterio],
                           material: f6.Material, ruta_salida: str,
                           marco: str = "", qdrant=None, contexto: str = ""):
    """La sentencia, viéndose escribir. Rinde trozos y, al final, el Resultado."""
    e = r.encargo
    # LA RAMA TÉCNICA, ANTES DE REDACTAR. Se calculaba al COMPONER, con el
    # estudio ya escrito, así que el modelo nunca supo en qué escenario estaba.
    #
    # Y LA PRIMERA VEZ QUE LO PUSE SE ME FUE DE ÁMBITO: quedó en la función de
    # al lado y `resolver_en_vivo` —que es por donde pasa TODA la pantalla— lo
    # usaba sin tenerlo. Cada generación moría con «name '_rama' is not
    # defined»; el servidor devolvía 200 con su evento de error y la pantalla
    # se quedaba muda. Lo vieron los registros de Render, no el guardián.
    _rama, _vp = _rama_tecnica(r, criterios, contexto)

    avisos: list[str] = []
    # SIN APARTADO DE MARCO JURÍDICO, EN NINGUNA DE LAS DOS FORMAS. David,
    # 25-sep-2026: «el marco jurídico se me hace innecesario ya que en cada
    # caso se cita ley, jurisprudencia para resolver. Hay que prescindir del
    # marco jurídico». Medido en el ADC 93/2026: 791-873 palabras de repaso
    # constitucional y convencional que no cambiaban ninguna respuesta, y que
    # transcribían ley estatal de Querétaro en un juicio federal. El material
    # constitucional sigue llegando al estudio: se cita donde decide.
    # `redactar_marco` y la poda de `marco_escrito` en `_terminar` se quedan
    # vivas pero sin llamada; `tarea_marco` va siempre en None.
    tarea_marco = None

    estudio = advertencias = ""
    _meta = {}
    _litis_y_material(r, material, avisos, cliente, criterios)
    # LAS TESIS DE LA TÉCNICA DE ESTA RAMA, igual que en `resolver`.
    await _tesis_de_la_tecnica(qdrant, material, _rama)
    t0 = _time.perf_counter()
    # LAS MARCAS NO LLEGAN A LA PANTALLA (Paso 2a): el filtro retiene desde «⟦»
    # hasta «⟧» y quita la marca, aunque llegue partida entre dos trozos. Pasa
    # con cualquier variante: sin marcas, el texto sale tal cual y sólo se
    # retrasa lo que tarde en llegar un cierre que nunca llega (200 caracteres
    # como mucho). El texto final lo limpia `_terminar`.
    import marcas as _mc_v
    _filtro = _mc_v.FiltroMarcas()
    async for paso in f6.redactar_en_vivo(
            cliente, r.fases.resumen_acto, r.fases.resumen_conceptos,
            criterios, material, e.es_recurso, r.partes, marco, contexto,
            propuesta_global=getattr(e, "propuesta_global", None),
            rama=_rama, violacion_procesal=_vp,
            conceptos_violacion=getattr(e, "conceptos_violacion", "") or "",
            # EL ESCRITO DE LA PARTE, LITERAL. `fuentes[1]` es el texto del
            # recurso o de la demanda tal como se leyó del PDF. Hasta ahora
            # moría en el adelanto: la fase que CONTESTA los conceptos recibía
            # el resumen —unas 472 palabras— y CERO caracteres del escrito.
            escrito_literal=(list(getattr(r.fases, "fuentes", []) or []) + ["", ""])[1],
            # EL GUION DEL PLAN (v4), igual que el gemelo `resolver`.
            guion=str(getattr(e, "guion", "") or "")):
        if paso.get("tipo") == "texto":
            _dato = _filtro.alimentar(paso.get("dato") or "")
            if _dato:
                yield {"tipo": "texto", "dato": _dato}
        else:
            _resto = _filtro.cerrar()
            if _resto:
                yield {"tipo": "texto", "dato": _resto}
            estudio = paso.get("estudio", "")
            advertencias = paso.get("advertencias", "")
            avisos.extend(paso.get("avisos", []))
            _meta = dict(paso.get("meta") or {})
    _resto = _filtro.cerrar()
    if _resto:
        yield {"tipo": "texto", "dato": _resto}
    TIEMPOS["estudio de fondo"] = round(_time.perf_counter() - t0, 1)
    # LA CALIFICACIÓN SUELTA, A SU APERTURA, antes de la reparación dirigida
    # (igual que en `resolver`).
    estudio = _congruencia_pegar(material, estudio, _meta)
    # LA REPARACIÓN DIRIGIDA (v3/v4, p2-exhaustivo): igual que en `resolver`.
    # La pantalla ve «completando» mientras corre la llamada; sólo si la hay.
    _faltan = _por_completar(material, criterios, estudio, _plan_del_estudio(r),
                             _concede_del_estudio(r, criterios))
    if _faltan:
        yield {"tipo": "completando"}
        with cronometrar("completar el estudio"):
            estudio = await _completar_estudio(cliente, r, criterios, material, estudio,
                                               _faltan, _meta, avisos)
    # LA CALIFICACIÓN AL ABRIR (familia v2, p2-congruencia): sin modelo.
    estudio = _congruencia_apertura(r, criterios, material, estudio, _meta, avisos)
    # EL SUPERVISOR DEL PROYECTO, igual que en `resolver` (2-oct-2026). La
    # pantalla ve «revisando» mientras corre la llamada; sólo si rige.
    import supervisor_proyecto as _sup
    if _sup.activo():
        yield {"tipo": "revisando"}
        with cronometrar("supervisor"):
            estudio = await _sup.en_el_resolver(cliente, r, criterios, material, estudio,
                                                _meta, avisos, contexto)
    # CON LA RAMA, que ya lee lo que el estudio concluyó (28-sep-2026): en un
    # «revoca y niega» no hay efectos que ordenar (AR 631/2025).
    _av_ef = f6._efectos_de_reposicion(estudio, criterios, _vp,
                                       rama=_rama_de(r, criterios, estudio))
    if _av_ef:
        avisos.insert(0, _av_ef)
    # LAS CONSTANCIAS INDISPENSABLES QUE NO SE APORTARON, dichas arriba del
    # todo: el proyecto se escribió sin verlas.
    try:
        import constancias as _cn_a
        _pg_cn = getattr(e, "propuesta_global", None) or {}
        _pral_cn = next((c for c in (criterios or [])
                         if str(getattr(c, "jerarquia", "")).lower() == "principal"),
                        (criterios or [None])[0])
        _al_reves_cn = bool(_pral_cn is not None and _pg_cn.get("sentido")
                            and not f6._misma_direccion(
                                str(_pg_cn.get("sentido") or ""),
                                str(getattr(_pral_cn, "sentido", "") or "")))
        _av_cn = _cn_a.aviso_faltantes(
            _pg_cn.get("constancias") or [], contexto, al_reves=_al_reves_cn)
        if _av_cn:
            avisos.insert(0, _av_cn)
    except Exception as _exc_cn:
        print(f"   ⚠️ constancias: {type(_exc_cn).__name__}")

    yield {"tipo": "componiendo"}
    avisos = await _fuentes_tardias(r, e, material, estudio, avisos, qdrant, _meta)
    res = await _terminar(cliente, r, e, criterios, material, estudio,
                          advertencias, avisos, tarea_marco, ruta_salida, qdrant, marco,
                          contexto, meta_estudio=_meta)
    yield {"tipo": "listo", "resultado": res}


def _reindexar(mapa: dict, lineas: list, pars: list) -> dict:
    """El mapa, con el índice del párrafo tal como lo compone el documento.

    `separar_marcas` cuenta los renglones no vacíos del estudio; el documento
    lo compone con `f6.parrafos`, que quita el encabezado «SEXTO. Estudio.» y
    los rótulos sueltos. Se casan en orden: cada renglón con el párrafo que
    termina igual; uno que el documento quitó apunta al siguiente."""
    a_doc, j = {}, 0
    for i, ln in enumerate(lineas):
        t = ln.strip()
        k = j
        while k < len(pars) and not (pars[k] and t.endswith(pars[k])):
            k += 1
        if k < len(pars):
            a_doc[i] = k
            j = k + 1
        else:
            a_doc[i] = min(j, max(len(pars) - 1, 0))
    return {ident: sorted({a_doc.get(i, i) for i in idxs}) for ident, idxs in (mapa or {}).items()}


# ═══ LA LIMPIEZA DEFENSIVA DE LOS RÓTULOS DEL GUION (26-sep-2026) ═══════════
# El guion del plan (v4) es interno: sus rótulos no se escriben en la sentencia
# (plan_estudio.bloque lo dice). Si el modelo copia alguno como renglón suelto,
# llegaría al .docx. Se quitan SÓLO los renglones que EMPIEZAN por un rótulo del
# guion inequívoco —en mayúsculas y seguido del identificador del plan—; una
# sentencia no escribe «APLICA C1.a» ni «APARTADO 2 ·». Calibrado contra los
# engroses reales del corpus (campo «oro»): 0 renglones quitados. El renglón
# «DESVIACIONES DEL GUION» es la explicación que el guion manda poner en
# ADVERTENCIAS: si quedó en el cuerpo, se lleva allí, no se tira.
_ID_PLAN = r"(?:AD|[CAS])\d+\.[a-z]{1,2}"
_RX_ROTULO_GUION = re.compile(
    r"^[ \t>*#\-]*(?:"
    r"GUION DEL ESTUDIO\b"
    r"|APARTADO \d+ ·"
    r"|(?:APLICA|REMITE|DESARROLLA|RESIDUAL|NO SE ESTUDIA) " + _ID_PLAN + r"\b"
    r"|EXPONE M\d+\b"
    # Los de la jerarquía (plan-6, 28-sep-2026): tampoco los escribe una
    # sentencia, en mayúsculas y seguidos de su dato del plan.
    r"|JERARQUÍA DEL PROBLEMA \d+"
    r"|(?:INNECESARIOS POR SUFICIENCIA|CAEN POR DERIVAR) " + _ID_PLAN + r"\b"
    r")")
_RX_DESVIACIONES = re.compile(r"^[ \t>*#\-]*DESVIACIONES DEL GUION\b")


def limpiar_rotulos_del_guion(estudio: str, advertencias: str = "") -> tuple:
    """(estudio, advertencias, renglones quitados, renglones llevados a
    ADVERTENCIAS). Ver `_RX_ROTULO_GUION`. No toca nada más: un renglón que no
    empieza por un rótulo del guion se queda aunque lo nombre dentro."""
    fuera, llevados, quedan = 0, [], []
    for ln in str(estudio or "").split("\n"):
        if _RX_DESVIACIONES.match(ln):
            llevados.append(ln.strip(" \t>*#-"))
            continue
        if _RX_ROTULO_GUION.match(ln):
            fuera += 1
            continue
        quedan.append(ln)
    if llevados:
        advertencias = "\n\n".join(x for x in [str(advertencias or "").strip()] + llevados if x)
    return "\n".join(quedan), advertencias, fuera, len(llevados)


def _marcas_y_cobertura(r, e, material, estudio: str, advertencias: str,
                        criterios: list = None) -> dict:
    """Separa las marcas y aplica el control V1. Nunca lanza.

    Devuelve {estudio, advertencias (sin marcas), aviso (el visible, o «»),
    meta: {mapa, cobertura}}. El aviso VISIBLE sólo sale si falta la marca de
    un argumento Y el rescate por anclas y texto tampoco lo encuentra (ver
    `marcas.verificar`); lo demás va en sombra, sólo en `meta`."""
    fuera = {"estudio": estudio or "", "advertencias": advertencias or "",
             "aviso": "", "meta": {}}
    try:
        import marcas as _mc_t
        limpio, mapa = _mc_t.separar_marcas(estudio or "")
        adv_limpias, mapa_adv = _mc_t.separar_marcas(advertencias or "")
        fuera["estudio"], fuera["advertencias"] = limpio, adv_limpias
        segs = list(getattr(material, "inventario", None) or [])
        if not (segs or mapa or mapa_adv):
            return fuera
        pars = f6.parrafos(limpio)
        mapa_doc = _reindexar(mapa, _mc_t.parrafos(limpio), pars)
        # EL ARRANQUE DE CADA PÁRRAFO, EN LISTA Y CON EL MISMO ÍNDICE QUE EL
        # MAPA (revisión adversarial, 26-sep-2026): la pestaña «Mapa del
        # estudio» de la pieza de pantallas lee `parrafos` así, en el primer
        # nivel del «listo» y de la ficha; sin él enseña «párrafo 14» sin su
        # texto. Catorce palabras por párrafo: unos 100 bytes cada uno.
        meta = {"mapa": mapa_doc,
                "parrafos": [" ".join(p.split()[:14]) for p in pars][:400]}
        if segs:
            cob = _mc_t.verificar(segs, mapa, limpio)
            # LO QUE LA PANTALLA NECESITA PARA EL «MAPA DEL ESTUDIO»: cada
            # argumento con lo que se alega y su cita, y el arranque de cada
            # párrafo marcado. Recortado: la ficha se apila doce veces.
            # Unas 300 letras por argumento: la mediana son 13 argumentos (4 KB
            # por ficha) y el asunto más largo medido, 72 (22 KB).
            cob["segmentos"] = [
                {"id": x.get("id"), "concepto": x.get("concepto"),
                 "texto": " ".join(str(x.get("texto") or "").split()[:25]),
                 "cita": " ".join(str(x.get("cita") or "").split()[:30]),
                 "pagina": x.get("pagina") or ""}
                for x in segs][:120]
            _usados = sorted({k for v in mapa_doc.values() for k in v})
            cob["parrafos"] = {str(k): " ".join(pars[k].split()[:14])
                               for k in _usados if 0 <= k < len(pars)}
            cob["en_advertencias"] = sorted(mapa_adv)
            cob["visible"] = bool(cob.get("sin_rastro")) and _f6_con_inventario(material)
            # LOS DOS CONTROLES DE p2-exhaustivo SOBRE EL TEXTO FINAL, EN SOMBRA:
            # lo que queda sin su dato después de la reparación, y el «sin
            # materia» que el criterio no dijo (calibrado: acusa a un engrose
            # bueno, así que no se enseña; `exhaustivo.SIN_MATERIA_VISIBLE`).
            try:
                import exhaustivo as _ex_t
                _ps_t = _mc_t.parrafos(limpio)
                _pr_t = list(getattr(material, "problemas", None) or [])
                _pl_t = (getattr(e, "plan", None) or {})
                _pl_t = _pl_t.get("plan") if isinstance(_pl_t, dict) else None
                _sm = _ex_t.sin_materia_por_su_cuenta(_ps_t, mapa, segs, criterios or [], _pr_t,
                                                      plan=_pl_t if isinstance(_pl_t, dict) else None)
                cob["exhaustivo"] = {
                    "sin_dato": _ex_t.sin_su_dato(_ps_t, mapa, segs),
                    "sin_materia": [{k: v for k, v in h.items() if k != "texto"} for h in _sm],
                }
                if _sm and _ex_t.SIN_MATERIA_VISIBLE and _f6_con_inventario(material):
                    import tipos_asunto as _ta_x
                    fuera["aviso_sin_materia"] = _ex_t.aviso_sin_materia(
                        _sm, _ta_x.vocabulario_de(getattr(e, "tipo_asunto", "") or "amparo_directo")["combate_singular"])
            except Exception as _ex_x:
                print(f"   ⚠️ EXHAUSTIVO: {type(_ex_x).__name__}")
            meta["cobertura"] = cob
            # HIGIENE DE REGISTROS: identificadores y cifras, nunca el texto.
            print(f"   🧭 MARCAS: {cob['marcados']}/{cob['total']} marcados · "
                  f"rescatados {len(cob['rescatados'])} · sin rastro "
                  f"{cob['sin_rastro'][:12]}" + (" (aviso visible)" if cob["visible"] else " (sombra)")
                  + (f" · desconocidos {cob['desconocidos'][:6]}" if cob["desconocidos"] else ""))
            if cob["visible"]:
                import tipos_asunto as _ta_m
                _q1 = _ta_m.vocabulario_de(getattr(e, "tipo_asunto", "") or "amparo_directo")["combate_singular"]
                fuera["aviso"] = _mc_t.aviso(segs, cob, _q1)
        fuera["meta"] = meta
    except Exception as _ex:
        print(f"   ⚠️ MARCAS: no se pudieron separar: {type(_ex).__name__}")
        # SI LO QUE REVENTÓ FUE LA SEPARACIÓN MISMA, el estudio seguiría con
        # sus marcas camino del .docx (revisión adversarial, 26-sep-2026). Se
        # quitan a lo bruto: sin mapa, pero ninguna marca llega a la sentencia.
        for _k in ("estudio", "advertencias"):
            if "⟦" in (fuera[_k] or ""):
                fuera[_k] = re.sub(r"[ \t]*⟦[^⟦⟧\n]{0,1200}⟧[ \t]?", " ", fuera[_k])
                fuera[_k] = "\n".join(_l.strip() for _l in fuera[_k].split("\n"))
    return fuera


def _f6_con_inventario(material) -> bool:
    try:
        return bool(f6.con_inventario(material))
    except Exception:
        return False


# ═══ LA REPARACIÓN DIRIGIDA (p2-exhaustivo, 26-sep-2026) ════════════════════
# Los DOS gemelos la llaman en el mismo sitio —en cuanto tienen el estudio,
# antes de los efectos, las constancias y los preceptos—, con las mismas dos
# funciones. Sólo v3/v4 (las que traen inventario y marcas); en la v1 y la v2
# `_por_completar` devuelve [] sin mirar nada.
def _avisos_de_rama_al_dia(todos: list, de_ahora: list) -> list:
    """Si la composición de ahora dijo su rama, fuera los avisos de rama que no
    son suyos —los del adelanto, compuesto sin criterio— (AR 631/2025,
    28-sep-2026: «rama confirma_sobresee … infundado» y «NO SE PUDO LEER DEL
    PDF» junto al correcto «rama revoca_fondo_niega»). Nunca lanza."""
    try:
        import documento_generado as _dg_r
        if not any(_dg_r.es_aviso_de_rama(a) for a in (de_ahora or [])):
            return list(todos)
        _ahora = {str(a) for a in de_ahora}
        return [a for a in todos if not _dg_r.es_aviso_de_rama(a) or str(a) in _ahora]
    except Exception as _eav:
        print(f"   ⚠️ avisos de rama: {type(_eav).__name__}")
        return list(todos)


def _plan_del_estudio(r):
    """El plan (v4) con que se escribió este estudio, o None (v3)."""
    _pl = (getattr(getattr(r, "encargo", None), "plan", None) or {})
    _pl = _pl.get("plan") if isinstance(_pl, dict) else None
    return _pl if isinstance(_pl, dict) else None


def _concede_del_estudio(r, criterios):
    """¿El asunto concede? (`plan_estudio.concede_de`), o None si no se sabe.
    Para que la reparación dirigida decida la concesión «para efectos» con el
    sentido y no con lo que el estudio haya escrito (revisión adversarial de
    plan-6, 28-sep-2026)."""
    try:
        import plan_estudio as _pe_c
        return _pe_c.concede_de(r, criterios)
    except Exception:
        return None


def _por_completar(material, criterios, estudio: str, plan: dict = None, concede=None) -> list:
    """Los argumentos cuya respuesta es sólo una declaración de sin estudio,
    sin su dato en los efectos ni en otra respuesta (`exhaustivo.sin_su_dato`).
    Con el PLAN (v4, plan-6), sin lo que éste resuelve por consecuencia de su
    principal —innecesarios por suficiencia, los que caen por derivar—, salvo
    en una concesión para efectos (`exhaustivo.sin_consecuencia_del_plan`): en
    el AR 631/2025, un «revoca y niega», cada uno disparaba una llamada más al
    modelo que los volvía a contestar. Sin modelo. Nunca lanza."""
    try:
        if not _f6_con_inventario(material):
            return []
        import exhaustivo as _ex
        if not _ex.REPARAR_ACTIVO:
            return []
        segs = list(getattr(material, "inventario", None) or [])
        if not segs:
            return []
        return _ex.revisar_texto(estudio or "", segs, criterios,
                                 list(getattr(material, "problemas", None) or []), plan=plan,
                                 concede=concede)["sin_dato"]
    except Exception as _ex_p:
        print(f"   ⚠️ COMPLETAR: no se pudo revisar: {type(_ex_p).__name__}")
        return []


async def _completar_estudio(cliente, r, criterios, material, estudio: str,
                             faltan: list, meta: dict, avisos: list) -> str:
    """Una llamada más al modelo del estudio para lo que falta. Devuelve el
    estudio (con las piezas insertadas, o como estaba si falló); anota el
    informe en `meta["completado"]` y el aviso visible en `avisos`."""
    try:
        import exhaustivo as _ex
        _esc = (list(getattr(getattr(r, "fases", None), "fuentes", []) or []) + ["", ""])[1]
        # EL PLAN (v4): la calificación de cada argumento dentro de su problema
        # (plan-4, p2-congruencia). Sin plan —v3— la reparación de siempre.
        _pl = (getattr(getattr(r, "encargo", None), "plan", None) or {})
        _pl = _pl.get("plan") if isinstance(_pl, dict) else None
        nuevo, informe = await _ex.reparar(cliente, estudio, criterios, material, faltan,
                                           escrito=_esc, plan=_pl if isinstance(_pl, dict) else None)
        _av = _ex.aviso_reparacion(informe, list(getattr(material, "inventario", None) or []))
        if _av:
            avisos.insert(0, _av)
        # LO QUE LA REPARACIÓN CITA DE NUEVO PASA POR LA MISMA REVISIÓN QUE EL
        # ESTUDIO (revisión adversarial, 26-sep-2026). `f6.revisar` corrió antes
        # de la reparación: un registro que sólo trae la pieza nueva —el que
        # invocó la parte— salía sin el aviso de «no está en el material». Se
        # escribe con la misma forma, así que si luego se trae del acervo
        # `_registros_ya_estan` lo retira como a los demás.
        _validos = {str(t.get("registro", "")) for t in (getattr(material, "tesis", None) or [])
                    if isinstance(t, dict)}
        _reg_fuera = [x for x in informe.get("registros_nuevos") or [] if x not in _validos]
        if _reg_fuera:
            avisos.append(f"REGISTROS QUE NO ESTÁN EN EL MATERIAL: {sorted(_reg_fuera)}. "
                          "No se citan hasta comprobarlos en el Semanario.")
        if isinstance(meta, dict):
            meta["completado"] = {k: v for k, v in informe.items() if k != "salida"}
        # HIGIENE DE REGISTROS: identificadores y cifras, nunca el texto.
        print(f"   🧩 COMPLETAR: {informe.get('estado')} · pedidos {informe.get('pedidos')} · "
              f"párrafos {len(informe.get('parrafos') or [])} · efectos "
              f"{len(informe.get('efectos') or [])} · descartes {len(informe.get('descartes') or [])} · "
              f"{informe.get('segundos')} s")
        return nuevo
    except Exception as _ex_c:
        print(f"   ⚠️ COMPLETAR: {type(_ex_c).__name__}")
        return estudio


# ═══ LA CALIFICACIÓN AL ABRIR (p2-congruencia, 26-sep-2026) ════════════════
# ADC 642/2024 v4: cuatro apartados abrían con el concepto y seguían con «Lo
# anterior, porque…» sin calificación en medio. El modelo SÍ la había escrito,
# en su propio renglón —«Es fundado.»—, y el compositor tira los párrafos de
# menos de seis palabras. Aquí se pega a su apertura y, si aun así el «Lo
# anterior» no tiene a qué referirse, se añade la calificación que el criterio
# (o el plan) da a ese apartado. Sin modelo. SÓLO LA FAMILIA v2: las 14
# corridas v1 del banco no tienen una calificación suelta.
def _congruencia_pegar(material, estudio: str, meta: dict) -> str:
    """La calificación que el modelo escribió sola en su renglón, pegada a su
    apertura (`congruencia.pegar_calificaciones`), ANTES de la reparación
    dirigida. Cuenta lo pegado en `meta` para el informe de
    `_congruencia_apertura`. Sólo la familia v2. Nunca lanza."""
    try:
        if not f6._v2(material):
            return estudio
        import congruencia as _cg
        nuevo, hechas = _cg.pegar_calificaciones(estudio or "")
        if isinstance(meta, dict):
            meta["_cg_pegadas"] = len(hechas)
        return nuevo
    except Exception as _ex_cp:
        print(f"   ⚠️ CONGRUENCIA (pegar): {type(_ex_cp).__name__}")
        return estudio


def _congruencia_apertura(r, criterios, material, estudio: str, meta: dict,
                          avisos: list) -> str:
    """El estudio con la calificación en su apertura. Anota el informe en
    `meta["congruencia"]` y el aviso visible en `avisos`. Nunca lanza."""
    try:
        if not f6._v2(material):
            return estudio
        import congruencia as _cg
        _pl = (getattr(getattr(r, "encargo", None), "plan", None) or {})
        _pl = _pl.get("plan") if isinstance(_pl, dict) else None
        nuevo, inf = _cg.reparar_aperturas(estudio or "", criterios,
                                           list(getattr(material, "problemas", None) or []),
                                           plan=_pl if isinstance(_pl, dict) else None)
        import tipos_asunto as _ta_cg
        _q1 = _ta_cg.vocabulario_de(getattr(material, "tipo_asunto", "") or "amparo_directo")["combate_singular"]
        _av = _cg.aviso_aperturas(inf, _q1)
        if _av:
            avisos.insert(0, _av)
        # LO PEGADO ANTES DE LA REPARACIÓN DIRIGIDA (`_congruencia_pegar`)
        # cuenta con lo de ahora: el informe dice lo que se pegó en total.
        inf["pegadas_antes"] = int((meta.pop("_cg_pegadas", 0) if isinstance(meta, dict) else 0) or 0)
        if isinstance(meta, dict):
            meta["congruencia"] = _cg.informe_sombra(inf)
        # HIGIENE DE REGISTROS: cifras, nunca el texto.
        print(f"   🧷 CONGRUENCIA: {len(inf.get('pegadas') or []) + inf['pegadas_antes']} "
              f"calificación(es) pegada(s) · "
              f"{len(inf.get('anadidas') or [])} añadida(s) · {len(inf.get('sin_reparar') or [])} "
              f"sin reparar · {len(inf.get('sombra') or [])} en sombra")
        return nuevo
    except Exception as _ex_cg:
        if isinstance(meta, dict):
            meta.pop("_cg_pegadas", None)
        print(f"   ⚠️ CONGRUENCIA: {type(_ex_cg).__name__}")
        return estudio


def _congruencia_efectos(e, material, estudio_limpio: str, criterios, meta: dict) -> None:
    """LOS EFECTOS CASAN CON EL CUERPO, EN SOMBRA (`congruencia.efectos_sin_cubrir`):
    sobre el texto que se compone, después de la reparación dirigida. Sólo la
    familia v2. Nunca lanza."""
    try:
        if not f6._v2(material):
            return
        import congruencia as _cg
        _pl = (getattr(e, "plan", None) or {})
        _pl = _pl.get("plan") if isinstance(_pl, dict) else None
        _segs = list(getattr(material, "inventario", None) or []) if f6.con_inventario(material) else []
        h = _cg.efectos_sin_cubrir(f6.parrafos(estudio_limpio or ""), _segs, criterios or [],
                                   list(getattr(material, "problemas", None) or []),
                                   plan=_pl if isinstance(_pl, dict) else None)
        if isinstance(meta, dict):
            meta.setdefault("congruencia", {})["efectos_sin_cubrir"] = h
        if h:
            print(f"   🧷 CONGRUENCIA: {len(h)} argumento(s) fundado(s) sin huella en los EFECTOS (sombra)")
    except Exception as _ex_ce:
        print(f"   ⚠️ CONGRUENCIA (efectos): {type(_ex_ce).__name__}")


def _revisar_contaminacion(r, e) -> list:
    """Que nada del proyecto sea de otro asunto.


    David: «asegúrate de que los formatos de salida no estén contaminados con
    datos que no correspondan al asunto que proyecta el secretario». Se
    comprueba sin modelo: todo nombre propio, número de expediente y cantidad
    del proyecto debe estar en los documentos que él subió o en lo que tecleó.
    Lo que no esté viene de otra parte.
    """
    try:
        import contaminacion as _c
    except Exception:
        return []
    fuentes = list(getattr(getattr(r, "fases", None), "fuentes", []) or [])
    autos = str(getattr(getattr(r, "fases", None), "autos", "") or "")
    if autos:
        fuentes.append(autos)
    enc = {k: getattr(e, k, "") for k in
           ("numero", "encabezado", "quejoso", "responsable", "magistrado",
            "secretario", "tribunal", "ciudad")}
    try:
        with open(r.ruta, "rb"):
            pass
        import docx as _dx
        d = _dx.Document(r.ruta)
        texto = "\n".join(p.text for p in d.paragraphs)
        for tb in d.tables:
            for fila in tb.rows:
                texto += "\n" + " | ".join(c.text for c in fila.cells)
    except Exception:
        return []
    return _c.revisar(texto, fuentes, enc)


async def _terminar(cliente, r, e, criterios, material, estudio,
                    advertencias, avisos, tarea_marco, ruta_salida, qdrant=None,
                    marco: str = "", contexto: str = "",
                    meta_estudio: dict = None):
    """De la salida del modelo al documento entregado.

    Vive fuera de `resolver()` porque la versión en vivo hace exactamente lo
    mismo cuando el flujo termina, y tener dos copias de esto es tener dos
    sitios donde se rompe la congruencia.
    """
    # ═══ LAS MARCAS SE SEPARAN ANTES DE COMPONER (Paso 2a, 26-sep-2026) ════
    # Lo primero, porque todo lo de abajo —la litis, el relleno, el .docx—
    # tiene que ver el texto que se firma, y las marcas son registro interno.
    # Aquí convergen los dos gemelos, así que deciden igual. El mapa y la
    # cobertura van a la ficha y al evento «listo» por `meta_estudio`.
    meta_estudio = dict(meta_estudio or {})
    # LOS RÓTULOS DEL GUION QUE SE COLARON, fuera (antes de las marcas: el mapa
    # de marcas a párrafos se calcula sobre el texto que se compone).
    estudio, advertencias, _n_rot, _n_desv = limpiar_rotulos_del_guion(estudio, advertencias)
    if _n_rot or _n_desv:
        print(f"   🧹 GUION: {_n_rot} rótulo(s) del plan quitados del estudio"
              + (f", {_n_desv} desviación(es) llevadas a ADVERTENCIAS" if _n_desv else ""))
        avisos.insert(0, "LIMPIEZA: el estudio copió "
                      + (f"{_n_rot} renglón(es) con rótulos internos del guion (APARTADO, APLICA, "
                         f"EXPONE…), que se quitaron" if _n_rot else "")
                      + ("; " if _n_rot and _n_desv else "")
                      + ("la explicación «DESVIACIONES DEL GUION» en el cuerpo, que se llevó a "
                         "ADVERTENCIAS" if _n_desv else "")
                      + ". Revisa que no falte nada en el apartado donde estaban.")
    _v1 = _marcas_y_cobertura(r, e, material, estudio, advertencias, criterios)
    estudio, advertencias = _v1["estudio"], _v1["advertencias"]
    if _v1["aviso"]:
        avisos.insert(0, _v1["aviso"])
    if _v1.get("aviso_sin_materia"):
        avisos.insert(0, _v1["aviso_sin_materia"])
    meta_estudio.update(_v1["meta"])
    # LO QUE HIZO LA REPARACIÓN DIRIGIDA viaja con la cobertura, que es lo que
    # el «listo» y la ficha ya llevan (`main._taller_meta_listo`).
    _compl = meta_estudio.pop("completado", None)
    if _compl and isinstance(meta_estudio.get("cobertura"), dict):
        meta_estudio["cobertura"].setdefault("exhaustivo", {})["completado"] = _compl
    # LOS EFECTOS CASAN CON EL CUERPO (p2-congruencia), en sombra.
    _congruencia_efectos(e, material, estudio, criterios, meta_estudio)
    # ═══ LA LEY LOCAL SÓLO ENTRA SI ESTÁ EN LA LITIS ═══════════════════════
    # Aquí convergen los dos redactores del estudio, y el marco llega unas
    # líneas más abajo: es el único sitio por el que pasa TODO lo que se va a
    # componer. Se sanea antes de armar el relleno, porque el relleno copia el
    # estudio y las normas en el momento de construirse.
    import litis_normativa as _ln
    try:
        _litis = _ln.leyes_de_la_litis(getattr(r, "fases", None))
        _buenas, _fuera = _ln.filtrar_normas(material.normas, _litis)
        material.normas = _buenas
        estudio, _av_l = _ln.sanear(estudio, _litis, _buenas, "el estudio de fondo")
        estudio, _n_loc = _ln.quitar_local_ante_federal(estudio)
        if _n_loc:
            print(f"   ⚖️ LITIS: {_n_loc} «local» quitado(s) ante ley federal en el estudio")
        # LOS AVISOS DE LA LITIS VAN PRIMERO: describen algo que se tocó en el
        # texto que se va a firmar, no algo mejorable.
        for _a in reversed(_av_l):
            avisos.insert(0, _a)
    except Exception as _el:
        _litis, _buenas = [], material.normas
        print(f"   ⚠️ LITIS: no se pudo sanear el estudio: {type(_el).__name__}: {_el}")
    # LA VERSIÓN MODERNA LLEVA LOS RESÚMENES CONDENSADOS. La síntesis corrió
    # a la vez que el estudio; aquí se recoge. Si no llegó, o no pasó sus
    # comprobaciones, van los completos: la forma corta nunca paga con un
    # planteamiento perdido.
    _sint = {}
    _t_sint = getattr(material, "sintesis", None)
    if _t_sint is not None:
        try:
            _sint = await asyncio.wait_for(_t_sint, timeout=120) or {}
        except Exception as _ets:
            print(f"   ⚠️ SÍNTESIS MODERNA no llegó: {type(_ets).__name__}")
            _sint = {}
    relleno = ens.Relleno(
        encabezado=e.encabezado, numero_asunto=e.numero, quejoso=e.quejoso,
        magistrado=e.magistrado, secretario=e.secretario,
        oportunidad=f0.parrafo_oportunidad(r.computo),
        antecedentes=_sint.get("antecedentes") or r.fases.parrafos_antecedentes(),
        resumen_acto=_sint.get("acto") or r.fases.parrafos_acto(),
        resumen_conceptos=_sint.get("conceptos") or r.fases.parrafos_conceptos(),
        problemas=r.fases.parrafos_problemas(),
        estudio=f6.parrafos(estudio),
        tesis=material.tesis,
        normas=material.normas,
        calificaciones=[c.sentido for c in criterios],
        presentacion=f0.fecha_en_letra(e.presentacion)
                     if getattr(e, 'presentacion', None) else '',
        responsable=getattr(e, 'responsable', '') or '',
        es_recurso=e.es_recurso,
    )
    # ═══ LOS ARTÍCULOS QUE DE VERDAD CITÓ ══════════════════════════════
    # Antes se buscaban por parecido ANTES de escribir, cuatro por problema, y
    # si el estudio acababa citando otros se quedaban sin texto y sin nota al
    # pie. Ahora se lee qué citó y se piden ESOS por número: `articulo_num`
    # está indexado en las 34 colecciones de leyes. No es adivinar lo que hará
    # falta, es traer lo que hizo falta.
    try:
        if qdrant is None:
            raise RuntimeError("sin cliente de Qdrant")
        import fase_normas as _fn
        with cronometrar("artículos citados"):
            # EL FUERO DE LA AUTORIDAD decide si el acervo del estado siquiera
            # se consulta. Se lee de lo que ya se sabe del asunto —la autoridad
            # responsable y el acto—, no del estudio, que aún puede no
            # nombrarla.
            # EL CAMPO SE LLAMA `responsable`, NO `autoridad_responsable`.
            # El getattr con valor por omisión apuntaba a un atributo que el
            # Encargo no tiene (se llama así en `fase_partes.Partes`, no aquí),
            # así que devolvía cadena vacía SIEMPRE y este detector llevaba
            # desde que se escribió sin ver nunca el nombre de la autoridad.
            # Es la avería silenciosa de manual: nada falla, nada avisa, y la
            # capa se comporta igual que si no existiera.
            #
            # MEDIDO ANTES DE ENCENDERLO, sobre las 89 autoridades del acervo:
            # sólo 4 pasan a fuero federal, y las cuatro son Salas Regionales
            # del Tribunal Federal de Justicia Administrativa. Ninguna local se
            # pierde por el camino: las Salas Civiles, las Juntas, los Jueces
            # administrativos de Querétaro y el Unitario Agrario siguen
            # consultando el acervo estatal.
            _quien = " ".join(str(x) for x in (
                getattr(e, "responsable", "") or "",
                getattr(e, "autoridad_responsable", ""),
                getattr(e, "acto_reclamado", ""),
                getattr(material, "acto_reclamado", ""),
                " ".join(getattr(material, "autoridades", None) or []),
            ) if x)
            _fed = _fn.autoridad_es_federal(_quien)
            if _fed:
                # QUE SE VEA CUÁL LO DECIDIÓ. Una capa que apaga el acervo
                # estatal no puede hacerlo sin dejar nombre: si algún día
                # apagara el de un asunto local, el registro lo dice.
                print(f"   ⚖️ autoridad del fuero FEDERAL "
                      f"«{(getattr(e, 'responsable', '') or '')[:60]}»: no se "
                      f"consulta el acervo estatal salvo que la cita nombre "
                      f"una ley local")
            _extra = await _fn.recuperar(
                qdrant, estudio,
                (e.coleccion_estatal or "") if hasattr(e, "coleccion_estatal") else "",
                fuero_federal=_fed)
        if _extra:
            _ya = {(str(n_.get("articulo")), str(n_.get("cuerpo_legal") or
                                                 n_.get("fuente") or ""))
                   for n_ in (material.normas or [])}
            nuevos = [x for x in _extra
                      if (x["articulo"], x["cuerpo_legal"]) not in _ya]
            # EL MISMO CANDADO DE LA LITIS (revisión del 29-sep): antes de
            # sumarlos al documento, fuera lo que la litis declara inadmisible
            # —p. ej. un Código Civil local citado de memoria en un asunto que
            # sólo litiga el procesal—, con su aviso.
            try:
                if _litis:
                    nuevos, _fuera_n = _ln.filtrar_normas(nuevos, _litis)
                    if _fuera_n:
                        avisos.append("Artículos citados que la litis no admite y NO se "
                                      "transcriben: " + "; ".join(
                                          f"art. {x.get('articulo')} — {x.get('cuerpo_legal')}"
                                          for x in _fuera_n[:6]) + ".")
            except Exception as _elf:
                print(f"   ⚠️ litis sobre los artículos recuperados: {type(_elf).__name__}")
            material.normas = list(material.normas or []) + nuevos
            # AL DOCUMENTO TAMBIÉN (29-sep-2026). El relleno se armó arriba con
            # la lista anterior y aquí se reasignaba otra: los artículos
            # recuperados nunca llegaban a `_componer_generado`.
            import contexto_taller as _ct_nd
            if _ct_nd.rediseno("normas_al_documento"):
                relleno.normas = material.normas
            print(f"   ⚖️ artículos citados recuperados: {len(nuevos)} nuevos "
                  f"de {len(_extra)} hallados")
            # Y SON FUENTE TARDÍA: el estudio los citó sin su texto (punto 7).
            try:
                import fuente_tardia as _ft
                # LAS NOTORIAS NO SON FUENTE TARDÍA (revisión del 29-sep): la
                # Ley de Amparo, la Constitución y la LOPJF se citan sin tener
                # su texto a propósito (`preceptos_fuera` las exime); marcarlas
                # dejaba casi todo proyecto en «justificación pendiente».
                _no_notorias = [x for x in nuevos if not any(
                    n in str(x.get("cuerpo_legal") or "").lower() for n in f6._NOTORIAS)]
                _cl = _ft.clasificar(estudio, (), _no_notorias,
                                     mapa=(meta_estudio or {}).get("mapa"),
                                     parrafos=f6.parrafos(estudio))
                _av_ft, _estado_ft = _ft.informe_y_aviso(_cl)
                if isinstance(meta_estudio, dict):
                    meta_estudio["fuentes_tardias"] = list(meta_estudio.get("fuentes_tardias") or []) + _cl
                    if _estado_ft:
                        meta_estudio["estado_salida"] = _estado_ft
                if _av_ft and _aviso_tardio_visible():
                    # UN SOLO AVISO: si ya había uno de las tesis tardías, se
                    # rehace con TODAS las fuentes tardías del proyecto.
                    _todas = [c for c in (meta_estudio or {}).get("fuentes_tardias") or []]
                    _av_uno, _ = _ft.informe_y_aviso(_todas or _cl)
                    avisos[:] = [a for a in avisos if not str(a).startswith("JUSTIFICACIÓN PENDIENTE")]
                    avisos.insert(0, _av_uno or _av_ft)
            except Exception as _eft:
                print(f"   ⚠️ artículos tardíos sin clasificar: {type(_eft).__name__}")
    except Exception as _ea:
        avisos.append(f"No se pudieron recuperar los artículos citados: {_ea}")

    marco_escrito = ""
    if tarea_marco is not None:
        with cronometrar("marco escrito"):
            try:
                marco_escrito = await tarea_marco
                # El marco es el tercer redactor, y fue EL QUE escribió la ley
                # de Querétaro en la revisión fiscal 2/2026.
                try:
                    marco_escrito, _av_m = _ln.sanear(
                        marco_escrito, _litis, _buenas, "el marco jurídico")
                    marco_escrito, _n_locm = _ln.quitar_local_ante_federal(marco_escrito)
                    if _n_locm:
                        print(f"   ⚖️ LITIS: {_n_locm} «local» quitado(s) ante ley federal en el marco")
                    for _a in reversed(_av_m):
                        avisos.insert(0, _a)
                except Exception as _elm:
                    print(f"   ⚠️ LITIS: no se pudo sanear el marco: {type(_elm).__name__}")
                import documento_generado as _dg4
                avisos.extend(_dg4.revisar_marco(marco_escrito, marco or ""))
            except Exception as _em:
                avisos.append(f"No se pudo escribir el marco jurídico: {_em}")
    # EL BARRIDO ADELANTADO (2-oct-2026, velocidad del supervisor): se pregunta
    # ya por los artículos del estudio y de los resúmenes, mientras la
    # síntesis de portada y el compositor trabajan. Al terminar el .docx se
    # barre el documento entero como siempre, reutilizando lo ya contestado.
    # Mismo barrido, otro momento; sólo con la bandera del supervisor y si el
    # barrido está encendido (el interruptor de Render sigue mandando).
    _memo_bar, _t_bar_pre = None, None
    try:
        import supervisor_proyecto as _sup_b
        import barrido_preceptos as _bp_pre
        if _sup_b.activo() and _bp_pre.BARRIDO_ACTIVO:
            _memo_bar = _sup_b.BarridoMemo()
            _t_bar_pre = asyncio.ensure_future(_bp_pre.barrer(
                _sup_b.texto_para_barrido(relleno, getattr(r, "estructura", None)), material,
                preguntar=_memo_bar.preguntar, confirmar=_memo_bar.confirmar))
    except Exception as _ebp:
        print(f"   ⚠️ BARRIDO adelantado: {type(_ebp).__name__}")
        _memo_bar, _t_bar_pre = None, None
    if (e.modo or "").lower() == "generado":
        with cronometrar("recomposición"):
            ruta, av_gen, _est = await _componer_generado(
                cliente, e, relleno, r.computo, ruta_salida,
                estructura_previa=getattr(r, "estructura", None),
                # Las constancias ya leídas viajan con las fases; si la
                # estructura hubiera que rehacerla, que no sea a ciegas.
                acto=(getattr(getattr(r, "fases", None), "fuentes", []) or [""])[0],
                fases=getattr(r, "fases", None),
                # AQUÍ SÍ, y es el único sitio donde existen: `_terminar` los
                # recibe del secretario y son los que fijan el sentido.
                criterios=criterios,
                partes=getattr(r, "partes", None),
                marco_escrito=marco_escrito,
                # LA REASUNCIÓN (art. 93, fr. VI): si los conceptos no constaron,
                # el resolutivo del amparo va con hueco (28-sep-2026).
                reasuncion=getattr(material, "reasuncion", None))
        avisos.extend(av_gen)
    else:
        with cronometrar("ensamblado"):
            ruta = ens.ensamblar(e.plantilla, relleno, ruta_salida)
    # CON LA RAMA (28-sep-2026): en un «revoca y niega» no hay efectos que
    # redactar, por fundados que sean los agravios (AR 631/2025).
    _rama_t = _rama_de(r, criterios, estudio)
    _, aviso_efectos = ens.formula_resolutivo(
        relleno.calificaciones, rama=_rama_t,
        sentido_amparo=__import__("fase_rama").sentido_en_plenitud(str(estudio or "")))
    # SALVO QUE LA EJECUTORIA NO CONCEDA NADA. Cuando el cómputo cierra por
    # extemporaneidad, el único resolutivo desecha el recurso y el estudio se
    # va al anexo: pedirle al secretario que redacte «los EFECTOS de la
    # concesión» es mandarlo a corregir algo que su proyecto no tiene.
    # NI CUANDO EL ESTUDIO YA LOS ESCRIBIÓ. Este aviso es de antes de que el
    # estudio redactara los efectos con su rótulo (22-sep-2026): con
    # calificación mixta daba por hecho que nadie los escribía, y en el ADC
    # 93/2026 salió de v6 a v10 con el SÉPTIMO ya puesto en cinco pasos.
    # Un aviso que acusa al proyecto bueno enseña a no leer los avisos.
    _ef_escritos = []
    try:
        import documento_generado as _dg_ef
        _ef_escritos = _dg_ef.partir_efectos(list(relleno.estudio or []))[1]
    except Exception:
        pass
    if (aviso_efectos and not _ef_escritos
            and not getattr(r.computo, "cierra_por_extemporaneidad", False)):
        avisos.append(aviso_efectos)
    # Deduplicado: el aviso del nombre se dispara una vez por párrafo donde
    # aparece, y el secretario no necesita leer tres veces lo mismo.
    for a in ens.avisos_ensamblado:
        if a not in avisos:
            avisos.append(a)
    for a in ens.residuo_de_plantilla(ruta, e.numero, e.plantilla):
        if a not in avisos:
            avisos.append(a)
    # LA CONGRUENCIA VA LA PRIMERA. Es el único aviso de esta lista que no
    # describe algo mejorable sino algo que no se puede firmar, y el secretario
    # tiene que verlo antes que los otros trece.
    incongruente = ens.revisar_congruencia(ruta, relleno.calificaciones,
                                           e.tipo_asunto)
    for a in reversed(incongruente):
        if a not in avisos:
            avisos.insert(0, a)

    # ═══ LA ÚLTIMA PUERTA: EL DOCUMENTO QUE SE ENTREGA ════════════════════
    # Las tres capas de arriba —material acotado, estudio saneado, marco
    # saneado— cubren las puertas conocidas. Ésta mira el .docx ya compuesto,
    # con sus notas al pie, por si una ley local entra por una que no se ha
    # encontrado. Si aparece algo aquí, el proyecto NO sale en silencio: el
    # aviso va el primero y dice que no se puede firmar así.
    try:
        import zipfile as _zf
        with _zf.ZipFile(ruta) as _z:
            _xml = _z.read("word/document.xml").decode("utf-8", "replace")
            _z_doc_xml = _xml
            if "word/footnotes.xml" in _z.namelist():
                _xml += _z.read("word/footnotes.xml").decode("utf-8", "replace")
        _plano = re.sub(r"<[^>]+>", "", re.sub(r"</w:p>", "\n", _xml))
        # UNA FECHA QUE NO CONSTA EN AUTOS NO SE AFIRMA. Toda fecha completa del
        # estudio tiene que estar en alguna fuente del asunto —los documentos
        # leídos, los resúmenes, lo aportado—. Calibrado sobre las tres
        # versiones del 93/2026 (0 acusaciones falsas: la que parecía inventada,
        # «cuatro de agosto», estaba en la demanda).
        try:
            import fechas_en_autos as _fe
            _fuentes_fe = list(getattr(r.fases, "fuentes", None) or []) + [
                str(getattr(r.fases, "resumen_acto", "") or ""),
                str(getattr(r.fases, "resumen_conceptos", "") or ""),
                # LOS ANTECEDENTES SON UNA CADENA (2-oct-2026): «"\n".join»
                # sobre ella metía un salto entre cada LETRA, y el aviso de
                # fechas buscaba en un texto ilegible. Mismo fallo que se curó
                # en `fase_rama`; se acepta la lista por si algún día lo es.
                (lambda _a: "\n".join(map(str, _a)) if isinstance(_a, (list, tuple))
                 else str(_a or ""))(getattr(r.fases, "antecedentes", "")),
                str(getattr(r.fases, "autos", "") or ""),
                str(contexto or "")]
            _av_fe = _fe.aviso(str(estudio or ""), _fuentes_fe)
            if _av_fe:
                print(f"   📅 {_av_fe[:200]}")
                avisos.append(_av_fe)
        except Exception as _exfe:
            print(f"   ⚠️ no se pudieron comprobar las fechas: {_exfe}")
        # EL BARRIDO FINAL CONTRA LA CITA INVENTADA. Sobre el documento
        # ENTERO y ya compuesto, que es donde están todas las citas: las del
        # estudio, las del marco y las de los resultandos. Pregunta una sola
        # cosa —¿existe este artículo?— a la fuente oficial en línea, en
        # lotes y en paralelo. Medido el 23-sep-2026 sobre la v6 del ADC
        # 93/2026: 6-14 s, cero acusaciones falsas en seis corridas.
        try:
            import barrido_preceptos as _bp
            if _memo_bar is not None:
                # Lo adelantado, recogido (con tope: si aún pregunta, se espera
                # lo que se habría tardado igual) y reutilizado.
                with cronometrar("barrido de preceptos"):
                    try:
                        await asyncio.wait_for(_t_bar_pre, timeout=_bp.BARRIDO_SEGUNDOS + 10)
                    except Exception:
                        pass
                    _r_bar = await _bp.barrer(_plano, material, preguntar=_memo_bar.preguntar,
                                              confirmar=_memo_bar.confirmar)
            else:
                _r_bar = await _bp.barrer(_plano, material)
            for _a in (_r_bar.get("avisos") or []):
                avisos.insert(0, _a)
        except Exception as _exbar:
            print(f"   ⚠️ BARRIDO: no se pudo comprobar las citas: {type(_exbar).__name__}")
        # EL DESENLACE, DICHO IGUAL EN TODO EL PROYECTO. `revisar_congruencia`
        # sólo conocía las fórmulas del amparo; en un recurso, «se confirma» en
        # el estudio contra «Se revoca» en el resolutivo pasaba limpio. Así salió
        # la revisión fiscal 2/2026: confirmaba en el estudio y en la síntesis,
        # revocaba en el cierre y en los resolutivos. Los resolutivos mandan
        # —son los que resuelven—, y cualquier parte que diga lo contrario deja
        # el proyecto marcado como no firmable, el primero de la lista.
        try:
            import desenlace as _dz_f
            for _a in reversed(_dz_f.contradicciones(
                    re.sub(r"<[^>]+>", "", re.sub(r"</w:p>", "\n",
                           _z_doc_xml)), e.tipo_asunto)):
                print(f"   🚨 DESENLACE: {_a[:220]}")
                avisos.insert(0, _a)
        except Exception as _edf:
            print(f"   ⚠️ DESENLACE: no se pudo revisar el documento final: {type(_edf).__name__}")
        _restan = _ln.inadmisibles(_plano, _litis)
        if _restan:
            _lista = "; ".join(f"art. {', '.join(h['nums'])} de la {h['ley']}"
                               for h in _restan[:4])
            print(f"   🚨 LITIS: el documento final aún cita ley local ajena: {_lista}")
            avisos.insert(0,
                "NO FIRMABLE TAL COMO ESTÁ — LEY LOCAL AJENA A LA LITIS: el "
                f"proyecto cita {_lista}, y ni la sentencia recurrida o el acto "
                "reclamado ni el escrito de agravios o conceptos invocan esa ley. "
                "Retírala o sustitúyela por el precepto que sí rige el asunto "
                "antes de listarlo.")
    except Exception as _elf:
        print(f"   ⚠️ LITIS: no se pudo revisar el documento final: {type(_elf).__name__}")

    # LA CALIDAD DEL FONDO, MEDIDA SOBRE EL DOCUMENTO ENTREGADO. No es una
    # opinión: son las cinco medidas que salieron de contar los defectos de los
    # engroses reales —densidad, exhaustividad, congruencia interna, promesa
    # cumplida y duplicación—. El secretario ve el número, no un adjetivo.
    try:
        import calidad_estudio as _ce
        from docx import Document as _Doc
        _txt = "\n".join(p.text for p in _Doc(ruta).paragraphs)
        # LA v2 CALIFICA UNA VEZ POR APARTADO, con la fórmula de David («Se
        # considera infundado.»): se cuenta también esa forma, o el recuento
        # la acusaría de dejar planteamientos sin respuesta. Ver
        # `calidad_estudio._RX_CALIFICA_AMPLIA`.
        import fase6_estudio as _f6m
        _v2_m = _f6m._v2(material)
        _m = _ce.medir(_txt, amplia=_v2_m)
        _d = _m["densidad"]
        if _d["palabras"] > 400 and _d["pct_propio"] < 0.70:
            avisos.append(
                f"EL ESTUDIO VIVE DE LA CITA: sólo el {_d['pct_propio']*100:.0f}% "
                f"es razonamiento propio ({_d['transcritas']} palabras "
                f"transcritas de {_d['palabras']}). La referencia de los "
                f"engroses es el 45%, así que esto no es un desastre —pero el "
                f"objetivo es el 70%.")
        for _r in _m["remisiones_rotas"]:
            avisos.append(f"REMISIÓN A UN APARTADO QUE NO EXISTE: «{_r}». Es el "
                          f"defecto que se caza leyendo el resolutivo en voz alta.")
        if _m["procedencia_contradice"]:
            avisos.append("LA PROCEDENCIA CONTRADICE EL FALLO: dice improcedente "
                          "en un asunto que se resuelve por el fondo.")
        if _m["promesa"]["rota"]:
            avisos.append("SE PROMETIÓ NO TRANSCRIBIR Y SE TRANSCRIBIÓ: el "
                          "documento dice que es innecesario reproducir el acto "
                          "y luego lo reproduce.")
        _dup = _ce.duplicacion_interna(_ce.estudio_de(_txt))
        if _dup:
            avisos.append(
                f"{len(_dup)} PASAJE(S) REPETIDO(S) dentro del estudio: «"
                f"{_dup[0][:90]}…». Un pasaje repetido no refuerza; delata que "
                f"se escribió por trozos.")
        # EL MARCO NORMATIVO DE OTRA VÍA. Acotado al considerando de
        # competencia y al de procedencia: fuera de ahí una cita de otra ley
        # puede ser legítima —un criterio análogo, una remisión— y prohibirla
        # empobrecería el estudio.
        import tipos_asunto as _ta_v
        for _pat, _porque in _ta_v.preceptos_ajenos(e.tipo_asunto, _txt):
            avisos.insert(0, f"PRECEPTO DE OTRA VÍA EN EL MARCO: {_porque}.")

        # LO QUE NO SE LEYÓ DEL EXPEDIENTE. Va arriba del todo: si de mil
        # páginas sólo entraron treinta y tres, eso condiciona TODO lo que
        # venga debajo, y hasta hoy no se decía en ninguna parte.
        try:
            import fases123_pipeline as _f123
            for _que, _entro, _habia in _f123.descartado():
                avisos.insert(0,
                    f"NO SE LEYÓ {_que.upper()} ENTERO: entraron {_entro:,} de "
                    f"{_habia:,} caracteres ({_entro * 100 // max(_habia, 1)}%, "
                    f"unas {_entro // 3000} páginas de {_habia // 3000}). Lo que "
                    f"no entró NO se ha valorado. Si el asunto se juega en esas "
                    f"páginas, sube sólo la parte que importa."
                    .replace(",", "."))
        except Exception:
            pass

        # ── LA PREGUNTA EXPRESA Y EL CIERRE QUE DICE ──────────────────────
        # Los dos vienen de Lara Chagoyán, y los dos son medibles: la pregunta
        # ya se calculaba y se perdía por el camino, y el cierre que remite es
        # lo único de su § 3.4 que los engroses de David NO hacen mal —los
        # cinco dicen en vez de remitir— y que el pipeline sí hacía.
        try:
            _pe = _ce.preguntas_expresas(_txt, [c for c in (criterios or [])])
            if _pe.get("kilometricas"):
                avisos.insert(0,
                    f"{len(_pe['kilometricas'])} pregunta(s) del apartado de "
                    f"materia pasan de {_ce.MAX_PREGUNTA} caracteres. Una "
                    f"pregunta que no cabe en una línea no fija la cuestión: "
                    f"la reformula entera. Si dentro lleva una «o», son dos "
                    f"problemas: «{_pe['kilometricas'][0][:100]}…»")
            import formato_sentencia as _fs_pe
            if _pe.get("sin_pregunta") and _fs_pe.normalizar(
                    getattr(material, "formato", "")) == _fs_pe.MODERNA:
                avisos.insert(0,
                    f"{len(_pe['sin_pregunta'])} de {_pe['problemas']} problemas "
                    f"NO se plantean como pregunta en el estudio. Fijar la "
                    f"cuestión con la pregunta expresa es lo que separa un "
                    f"apartado que se entiende de uno que hay que releer: "
                    f"«{_pe['sin_pregunta'][0]}…»")
            for _q, _d in _ce.cierre_ciego(_txt):
                # EN LA v2 NO HAY PÁRRAFO DE CIERRE QUE TENGA QUE DECIR «QUÉ, POR
                # QUÉ Y PARA QUÉ» (decisión 3 de David): lo que se acusa es la
                # remisión en blanco al final del estudio, y se dice así.
                if _v2_m:
                    _q = ("el estudio termina remitiendo a lo expuesto en vez de "
                          "decirlo: una remisión lleva su contenido —qué apartado "
                          "y qué proposición—")
                avisos.insert(0, f"{_q}: «…{_d[:120]}…»")
        except Exception as _ep:
            print(f"   ⚠️ no se pudo medir la delimitación: {_ep}")

        # EL RESULTANDO QUE NO INFORMA. Va el primero de la lista porque de ese
        # dato dependen el inciso del 97, la rama del 93 y que quien lea el
        # proyecto sepa de qué va el asunto.
        for _q, _d in _ta_v.resultando_evasivo(_txt, e.tipo_asunto):
            avisos.insert(0, f"{_q}" + (f": «…{_d[:130]}…»" if _d else "."))

        # EL LINTER. Frases cortadas, comillas huérfanas y el marcador genérico
        # donde debería ir el nombre.
        try:
            import linter_juridico as _lj
            for _que, _donde in _lj.revisar(_txt):
                avisos.append(f"SINTAXIS — {_que}"
                              + (f": «…{_donde[:110]}…»" if _donde else ""))
        except Exception as _el:
            print(f"   ⚠️ linter no disponible: {_el}")

        for _e in _ce.estadistica_en_el_texto(_txt):
            avisos.append(f"LA CIFRA DEL ACERVO SE COLÓ EN LA SENTENCIA: «{_e[:110]}». "
                          f"El criterio no se vota: quítala.")
        # CADA CONCEPTO, NOMBRADO. Las dos formas lo exigen —«Sobre el primer
        # concepto…» en la estándar, «No. El primer concepto es infundado…» en
        # la moderna— y es lo que deja comprobar que la versión corta no dejó
        # ninguno fuera. Se mide sobre el estudio que escribió el modelo, no
        # sobre el documento: los resúmenes de arriba nombran todos siempre.
        # Calibrado: 0 de 14 engroses reales del banco Kingston acusados.
        try:
            import formato_sentencia as _fs_x
            _falt_x = _fs_x.sin_contestar(
                estudio, getattr(material, "n_planteamientos", 0))
            if _falt_x:
                avisos.insert(0, _fs_x.aviso_sin_contestar(_falt_x, e.es_recurso))
        except Exception as _efx:
            print(f"   ⚠️ no se pudo comprobar la respuesta por concepto: {_efx}")
        _ex = _m["exhaustividad"]
        if _ex.get("contesta_todo") is False:
            avisos.append(
                f"QUEDAN PLANTEAMIENTOS SIN RESPUESTA: se anuncian "
                f"{_ex['planteamientos_anunciados']} y se emiten "
                f"{_ex['calificaciones_emitidas']} calificaciones, sin decir que "
                f"se estudian conjuntamente. Es omisión de estudio.")
    except Exception as _ec:
        print(f"   ⚠️ no se pudo medir la calidad del estudio: {_ec}")

    # NADA DEL PROYECTO PUEDE SER DE OTRO ASUNTO. Se comprueba sobre el .docx ya
    # escrito, que es lo que el secretario va a leer, y contra los documentos
    # que él subió.
    # SE COMPRUEBA DOS VECES —el adelanto y la sentencia—, así que hay que
    # unificar lo que digan las dos. Sin esto el proyecto salía con dos avisos
    # que NO PUEDEN SER CIERTOS A LA VEZ: uno afirmaba que la cantidad
    # «no consta en las fuentes» y el otro que las fuentes no se pueden leer.
    # Quien lee eso no sabe a cuál hacer caso, y con razón.
    r_final = Resultado(ruta=ruta, computo=r.computo, fases=r.fases, encargo=e)
    for a in _revisar_contaminacion(r_final, e):
        if a not in avisos:
            avisos.append(a)

    _todos = list(r.avisos) + avisos
    _vistos, _limpios = set(), []
    for a in _todos:
        _k = str(a)[:70]                    # el mismo aviso con un nombre más
        if _k in _vistos:                   # no es un aviso nuevo
            continue
        _vistos.add(_k)
        _limpios.append(a)
    # LA AUSENCIA DE PRUEBA NO ES PRUEBA, y tampoco al revés: si una pasada SÍ
    # pudo comprobar y encontró algo, el «no se pudo comprobar» de la otra
    # sobra y sólo resta credibilidad a lo que sí se halló.
    _hallo = any(str(a).startswith(("NOMBRES QUE NO", "EXPEDIENTES QUE NO",
                                    "CANTIDADES QUE NO")) for a in _limpios)
    if _hallo:
        _limpios = [a for a in _limpios
                    if not str(a).startswith("No se pudo comprobar la contaminación")]

    # LA RAMA DE LA REVISIÓN, POR EL MISMO MOTIVO. El adelanto se compone antes
    # de que exista el estudio, así que no puede saber qué resolvió el a quo y
    # anota la rama «sin_determinar». Cuando la sentencia sí lo determina, el
    # proyecto salía con LOS DOS avisos: «el a quo no consta qué resolvió» y,
    # dos líneas más abajo, «el a quo sobresee». El segundo es el bueno —tiene
    # el estudio delante— y el primero sólo siembra duda sobre él.
    if any("RESOLUTIVO DE REVISIÓN, rama" in str(a)
           and "sin_determinar" not in str(a) for a in _limpios):
        _limpios = [a for a in _limpios
                    if "RESOLUTIVO DE REVISIÓN, rama «sin_determinar»" not in str(a)
                    and not str(a).startswith("NO CONSTA QUÉ RESOLVIÓ EL JUZGADO")]
    # Y NO SÓLO «sin_determinar» (28-sep-2026, AR 631/2025): el adelanto se
    # compuso sin criterio y anotó su propia rama —«rama confirma_sobresee: el a
    # quo sobresee; el recurso resultó infundado», «EL SENTIDO DE LA SENTENCIA
    # RECURRIDA NO SE PUDO LEER DEL PDF»—, que viajó en la sesión hasta el
    # proyecto, junto al aviso correcto «rama revoca_fondo_niega». Si esta
    # composición dijo su rama, todos los avisos de rama del adelanto sobran:
    # quedan los de esta composición, que leyó el resolutivo del juzgado con el
    # estudio delante.
    _limpios = _avisos_de_rama_al_dia(_limpios, avisos)

    # LA VARIANTE VA SIEMPRE, aunque el modelo no haya dicho nada más: es lo
    # que el arnés de medición usa para descartar corridas que no casan.
    _meta_f = dict(meta_estudio or {})
    _meta_f.setdefault("variante", getattr(material, "variante", "v1") or "v1")
    return Resultado(ruta=ruta, computo=r.computo, fases=r.fases, encargo=e,
                     partes=r.partes, estudio=estudio, advertencias=advertencias,
                     huecos=ens.huecos_pendientes(ruta),
                     avisos=_limpios, meta_estudio=_meta_f)




# ── Los ANTECEDENTES ya se generan ────────────────────────────────────────
#
# Medidos sobre 199 apartados reales antes de escribir una línea de prompt:
# 645 palabras en 17 párrafos de 37 —crónica de trámite, frases cortas—, con
# verbos de procedimiento (dictó 186, admitió 112, interpuso 82) y arranques
# fijos («Por auto de», «En proveído de», «Seguido el juicio»).
#
# NO son el resumen del acto, aunque salgan de la misma sentencia: el resumen
# cuenta lo que la responsable RESOLVIÓ y por qué; los antecedentes, lo que
# PASÓ en el juicio de origen. Uno es razonamiento, el otro es crónica. Por eso
# se piden con otro prompt y sobre el documento ENTERO —el recorte del resumen
# se queda con el estudio de fondo, donde el trámite ya no está—.


def _resolutivo_del_a_quo(fases=None, acto: str = "") -> str:
    """Los puntos resolutivos del juzgado: el que se leyó del PDF en el adelanto
    y, si no, la sección resolutiva de la sentencia recurrida («» si no hay)."""
    import fase_rama as _fr_q
    _r = str(getattr(fases, "resolutivo_recurrida", "") or "")
    if _r.strip():
        return _r
    _t = acto or (list(getattr(fases, "fuentes", None) or []) + [""])[0]
    return _fr_q.seccion_resolutiva(str(_t or ""))


def _quejoso_del_amparo(e: Encargo, partes=None, resolutivo: str = "") -> str:
    """A quién se concede o niega el amparo.

    En un recurso el formulario guarda a «quien promueve» y eso es el
    RECURRENTE; quien pidió el amparo está en la ficha de partes, leída de la
    sentencia recurrida. Si lo tecleado es una autoridad y la ficha dice otro
    nombre, manda la ficha. Se resuelve AQUÍ, al armar los datos, y no sólo al
    fichar: la regeneración desde una sesión guardada no vuelve a fichar y el
    711/2025 tenía a la UIF de quejosa en el encargo ya persistido.

    EL RESOLUTIVO DEL JUZGADO MANDA (28-sep-2026, AR 631/2025): la tercera
    interesada que recurrió quedó guardada como «quejoso» (venía así de la
    admisión) y no es una autoridad, así que la regla de la UIF no la veía; el
    proyecto salió con «QUEJOSA Y RECURRENTE: IMPULSORA…», legitimación por el
    art. 6 y una síntesis en la que «la parte quejosa adquirió el inmueble».
    El punto resolutivo del juzgado dice a quién amparó —o no amparó, o en qué
    juicio sobreseyó—: si lo tecleado es otra parte, ése es el quejoso. Es la
    misma fuente de verdad que ya decide qué hizo el juzgado (577c700). Sin
    resolutivo legible, la ficha de partes cuando lo tecleado es una autoridad o
    la tercera interesada según la propia ficha."""
    _tecleado = str(getattr(e, "quejoso", "") or "").strip()
    _leido = str(getattr(partes, "quejoso", "") or "").strip()
    if not getattr(e, "es_recurso", False):
        return _tecleado
    try:
        import fase_rama as _fr_q
        _del_res = _fr_q.quejoso_del_resolutivo(resolutivo) if resolutivo else ""
    except Exception:
        _del_res = ""
    # SÓLO SI ES OTRA PARTE DE VERDAD (revisión del 28-sep-2026): el nombre
    # tecleado de la quejosa recurrente escrito con otra grafía que el
    # resolutivo («Ma.» por «María», una errata) no la convierte en otra parte
    # (`_parte_parecida`). Si se parece, sigue siendo ella y quien recurre.
    if _del_res and not _parte_parecida(_del_res, _tecleado):
        return _del_res
    _tercero = str(getattr(partes, "tercero_interesado", "") or "").strip()
    if (_leido and _tecleado and not _parte_parecida(_leido, _tecleado)
            and (_parece_autoridad(_tecleado) or _misma_parte(_tecleado, _tercero))):
        return _leido
    return _tecleado


def _organo_recurrido(e: Encargo, partes=None, acto: str = "") -> str:
    """El órgano cuya sentencia se revisa (la carátula y la competencia).

    EN EL AMPARO EN REVISIÓN ES UN JUZGADO DE DISTRITO (28-sep-2026, AR
    631/2025): la ficha de partes puede traer a la Sala responsable y el
    formulario también —el 631 salió con «ÓRGANO RECURRIDO: MAGISTRADA…» y la
    competencia «dictada … por la MAGISTRADA»—. Se toma de la ficha si es un
    juzgado o un tribunal (nunca una Sala ni una Magistrada, que en la revisión
    son la responsable del acto) y, si no, de la propia sentencia recurrida
    (`fase_rama.juzgado_de_la_recurrida`). En los demás recursos, como antes."""
    if not getattr(e, "es_recurso", False):
        return ""
    _leido = str(getattr(partes, "autoridad_responsable", "") or "").strip()
    _tecleado = str(getattr(e, "responsable", "") or "").strip()
    import tipos_asunto as _ta_o
    if _ta_o.normalizar(getattr(e, "tipo_asunto", "")) == "amparo_revision":
        # «DE DISTRITO» O UN TRIBUNAL DE AMPARO (revisión adversarial de la
        # fase E, 28-sep-2026): bastaba «juzgad|juez|tribunal», y en el AR
        # 631/2025 la ficha de partes podía leer al «Juzgado Quinto de Primera
        # Instancia Civil» (responsable) o al «Tribunal Superior de Justicia
        # del Estado» y darlo como órgano de la recurrida, «fijado por código».
        if (_leido and re.search(r"\bde\s+distrito\b|tribunal\s+(?:colegiado|unitario)",
                                 _leido, re.I)
                and not re.search(r"\bsala\b|magistrad", _leido, re.I)
                and not _mismo_nombre(_leido, _tecleado)):
            return _leido
        try:
            import fase_rama as _fr_o
            return _fr_o.juzgado_de_la_recurrida(acto or "")
        except Exception:
            return ""
    # SIN EL RENGLÓN DEL ÓRGANO EN LA CARÁTULA NO HAY DUPLICADO QUE EVITAR
    # (integración, 3-oct-2026, consecuencia de C3 y C4): el leído se callaba
    # cuando coincidía con lo tecleado porque ése ya salía en el rubro; hoy la
    # queja y la revisión fiscal no lo llevan y lo tecleado en `responsable` es
    # la ordenadora del auto de admisión (`tipos_asunto.responsable_es_el_organo`).
    _sin_renglon = not any(cl == "responsable" for _e, cl, _o in _ta_o.caratula_de(
        getattr(e, "tipo_asunto", "")))
    if _leido and re.search(r"juzgad|tribunal|sala\b|juez", _leido, re.I) \
            and (_sin_renglon or not _mismo_nombre(_leido, _tecleado)):
        return _leido
    return ""


def _recurrente_de(e: Encargo, partes=None, resolutivo: str = "") -> str:
    _propio = str(getattr(e, "recurrente", "") or "").strip()
    if _propio:
        return _propio
    _tecleado = str(getattr(e, "quejoso", "") or "").strip()
    if _quejoso_del_amparo(e, partes, resolutivo) != _tecleado:
        return _tecleado
    return ""


def _consta_que_recurre_la_quejosa(e: Encargo, partes=None, resolutivo: str = "") -> bool:
    """¿Hay PRUEBA de que quien recurre es la propia quejosa? (revisión del
    28-sep-2026, AR 631/2025). Lo tecleado es quien promueve el recurso; es la
    quejosa si el resolutivo del juzgado la nombra —ampara, no ampara o
    sobresee en el juicio que ella promovió— o si la ficha de partes la da como
    quejosa (con la tolerancia de `_parte_parecida`). Y también si el juzgado
    no concedió nada —negó o sobreseyó—: sólo la quejosa resiente ese fallo."""
    _tecleado = str(getattr(e, "quejoso", "") or "").strip()
    if not _tecleado:
        return False
    try:
        import fase_rama as _fr_p
        _del_res = _fr_p.quejoso_del_resolutivo(resolutivo) if resolutivo else ""
        _que = _fr_p.que_dice_el_resolutivo(resolutivo) if resolutivo else ""
    except Exception:
        _del_res, _que = "", ""
    if _del_res and _parte_parecida(_del_res, _tecleado):
        return True
    _leido = str(getattr(partes, "quejoso", "") or "").strip()
    if _leido and _parte_parecida(_leido, _tecleado):
        return True
    return _que in ("niega", "sobresee", "sobresee_niega")


# Los caracteres que `Encargo.papel_recurrente` puede fijar. El Ministerio
# Público no: su legitimación (art. 5o., fr. IV) la escribe la ficha de trámite
# con su propio carácter, y `ficha_procesal` no sabe calificarle el recurso.
PAPELES_FIJOS = ("quejoso", "autoridad", "tercero")


def papel_del_recurrente(e: Encargo, partes=None, resolutivo: str = "") -> str:
    """«quejoso» | «autoridad» | «tercero» | «» — en qué carácter recurre quien
    recurre (28-sep-2026). Vacío fuera de los recursos. Un recurrente que no es
    la quejosa ni una autoridad es la parte tercera interesada: en el 631 lo
    era la adquirente del inmueble, y con eso cambia la legitimación (art. 5o.,
    fr. III, no el 6o.), la dirección del diálogo (si prospera, pierde la
    quejosa) y la fracción del 93 que rige (la VI).

    «» TAMBIÉN CUANDO NO CONSTA (revisión del 28-sep-2026): sin recurrente
    propio ni otra parte identificada, esto contestaba «quejoso» y apagaba la
    fracción VI sin avisar —el fallo del 631 otra vez, con la tercera guardada
    como «quejoso» y un resolutivo que remite a «la parte quejosa precisada en
    el resultando primero»—. «quejoso» sólo con prueba positiva
    (`_consta_que_recurre_la_quejosa`); si no, «», que para
    `tipos_asunto.reasuncion` es «no consta y se aplica»."""
    if not getattr(e, "es_recurso", False):
        return ""
    import tipos_asunto as _ta_p
    if _ta_p.normalizar(getattr(e, "tipo_asunto", "")) == "revision_fiscal":
        return "autoridad"
    # EL CARÁCTER LEÍDO DEL ESCRITO MANDA (3-oct-2026, AR 631/2025): el propio
    # recurso dice con qué carácter se interpone; el vocabulario del nombre es
    # una conjetura.
    _fijo = str(getattr(e, "papel_recurrente", "") or "").strip().lower()
    if _fijo in PAPELES_FIJOS:
        return _fijo
    _rec = _recurrente_de(e, partes, resolutivo)
    if not _rec:
        return "quejoso" if _consta_que_recurre_la_quejosa(e, partes, resolutivo) else ""
    # EL RECURRENTE PROPIO QUE ES LA QUEJOSA (el formulario lo trae escrito):
    # no es otra parte.
    if str(getattr(e, "recurrente", "") or "").strip() and _parte_parecida(
            _rec, _quejoso_del_amparo(e, partes, resolutivo)):
        return "quejoso"
    return "autoridad" if _parece_autoridad(_rec) else "tercero"


def info_de_rama(r, declarado: str = "") -> dict:
    """Lo que la tarjeta y la pantalla necesitan para saber si hay que estudiar
    conceptos no estudiados (`fase_rama.conceptos_omitidos`), leído de una
    sesión: qué hizo el juzgado (su resolutivo manda), quién recurre, lo que el
    secretario aportó y si la recurrida también sobreseyó. Sin modelo."""
    import fase_rama as _fr_i
    import tipos_asunto as _ta_i
    e = getattr(r, "encargo", None)
    f = getattr(r, "fases", None)
    _decl = declarado or str(getattr(e, "resolvio_declarado", "") or "")
    _res = _resolutivo_del_a_quo(f)
    return {"tipo_asunto": _ta_i.normalizar(getattr(e, "tipo_asunto", "") or ""),
            "que_hizo": _fr_i.que_hizo_el_juzgado(f, _decl),
            "quien_recurre": papel_del_recurrente(e, getattr(r, "partes", None), _res) if e else "",
            "conceptos_violacion": str(getattr(e, "conceptos_violacion", "") or ""),
            "sobresee_ademas": _fr_i.sobreseyo_ademas(f, _decl),
            # LA CLASE DEL PRINCIPAL (revisión del 28-sep-2026): si es de
            # procedencia y prospera, se revoca y se sobresee (art. 93, fr. II),
            # sin reasumir ni pedir los conceptos.
            "clase_principal": clase_del_principal(getattr(f, "problemas", None))}


def clase_del_principal(problemas) -> str:
    """«fondo» | «procesal» | «procedencia» | «» del problema principal de la
    fase 3 (el que marca `jerarquia`; si ninguno, el primero, como el árbol)."""
    ps = [p for p in (problemas or []) if isinstance(p, dict)]
    if not ps:
        return ""
    pral = next((p for p in ps if str(p.get("jerarquia") or "").strip().lower() == "principal"),
                ps[0])
    return str(pral.get("clase") or "").strip().lower()


def _procedencia_prospera(r, criterios) -> bool:
    """¿El agravio que prospera es el principal y es de procedencia? Sólo
    entonces la revocación sobresee (art. 93, fr. II): si el principal de
    procedencia cae y prospera otro de fondo, rige lo de siempre."""
    try:
        import tipos_asunto as _ta_pp
        if clase_del_principal(getattr(getattr(r, "fases", None), "problemas", None)) \
                != "procedencia":
            return False
        crit = list(criterios or [])
        pral = next((c for c in crit
                     if str(getattr(c, "jerarquia", "") or "").strip().lower() == "principal"),
                    crit[0] if crit else None)
        return bool(pral is not None and _ta_pp.prospera(str(getattr(pral, "sentido", "") or "")))
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# LA PROCEDENCIA POR TIPO: EL CABLEADO (3-oct-2026)
# ═══════════════════════════════════════════════════════════════════════════
# David: «que los resultandos funcionen en toda la república sin que el
# secretario deba modificar más… un modelo por tipo que sea fiable y no deje
# margen de error en el proyecto». Las piezas las escriben `ficha_tramite`
# (los hechos con su fuente), `resultandos_por_tipo` (la prosa, sin modelo) y
# `verja_procesal` (la revisión del bloque); aquí sólo se conectan. Se
# importan DENTRO de cada función: si una falta o revienta, el adelanto no se
# cae —cae al camino de siempre o al hueco con su aviso, nunca a la memoria—.
#
# LA REGLA DEL CAMINO: rige la bandera Y el encargo trae ficha. Una sesión
# generada antes de encender la bandera no tiene ficha y sigue por donde vino
# (la llamada de siempre); una que la tiene se recompone de ella en cada
# documento —es pura y cuesta milisegundos—, también en el worker que no leyó
# el acto. Un dato que falta sale en HUECO con el aviso que lo nombra: el
# compositor nunca lo suple con una perífrasis.

def rige_procedencia_por_tipo() -> bool:
    """¿Rige `procedencia_por_tipo` en esta petición? False si el contexto del
    taller o la bandera no existen: un worker con código a medias cae al camino
    de siempre en vez de tumbar el adelanto."""
    try:
        import contexto_taller as _ct_pt
        _rige = getattr(_ct_pt, "rige", None) or getattr(_ct_pt, "rediseno")
        return bool(_rige("procedencia_por_tipo"))
    except Exception:
        return False


def _tipo_de_encargo(e) -> str:
    """La clave del tipo como la entiende la ficha («amparo_revision», no
    «revision»)."""
    _t = str(getattr(e, "tipo_asunto", "") or "")
    try:
        import tipos_asunto as _ta_t
        return _ta_t.normalizar(_t) or "amparo_directo"
    except Exception:
        return _t or "amparo_directo"


# ═══════════════════════════════════════════════════════════════════════════
# EL CÓMPUTO DEL RECURSO DE UNA AUTORIDAD (3-oct-2026, `procedencia_por_tipo`)
# ═══════════════════════════════════════════════════════════════════════════
# En el AR del XXII Circuito recurrió el Director de Ingresos y el considerando
# contó «surtió efectos al día hábil siguiente, conforme al artículo 31,
# fracción II»: la regla de los PARTICULARES aplicada a una autoridad. A la
# autoridad se le notifica por oficio y la notificación surte «desde el momento
# en que hayan quedado legalmente hechas» (art. 31, fr. I, LA); la electrónica,
# al generarse la constancia de la consulta (fr. III). El desplegable llega
# casi siempre con la de omisión, «personal», porque quien lo llena aún no ha
# dicho —ni tiene por qué saber— quién recurre. Es UN DÍA HÁBIL en el arranque
# del plazo, en el supuesto frecuente de la autoridad que recurre la concesión.
# Los considerandos ya citan la fracción que corresponde a la regla; lo que
# faltaba era que el cómputo usara la regla correcta.
_REGLAS_DE_PARTICULARES = ("personal", "lista")
# Los guardianes del cómputo (materia laboral, boletín de Querétaro) avisan con
# esta frase que contaron como personal; si luego se cuenta con la regla de la
# autoridad, ese aviso ya no es verdad y se quita.
_AVISO_CONTO_PERSONAL = "El cómputo se hizo con notificación PERSONAL"


# De dónde salió la forma de notificación, en la voz del aviso.
_DE_DONDE_LA_FORMA = {
    "secretario": "así lo declaraste en «Trámite en este tribunal»",
    "escrito": "así se lee en el escrito del recurso",
    "auto": "así se lee en el auto de admisión",
    "acto": "así se lee en la resolución recurrida",
}


def _plano_min(x) -> str:
    """Minúsculas, sin tildes y con los espacios colapsados."""
    import unicodedata as _u
    t = _u.normalize("NFD", str(x or "").lower())
    return " ".join("".join(ch for ch in t if _u.category(ch) != "Mn").split())


def _forma_normalizada(x) -> str:
    """«Electrónica», «por lista», «OFICIO»… → electronica | lista | oficio |
    personal | boletin | «» (lo que no se reconoce no se supone)."""
    t = _plano_min(x)
    if not t:
        return ""
    for clave, raiz in (("electronica", "electronic"), ("lista", "lista"),
                        ("oficio", "oficio"), ("boletin", "boletin"),
                        ("personal", "personal")):
        if raiz in t:
            return clave
    return ""


def forma_de_notificacion(e, escrito=None, acto=None) -> tuple:
    """(forma, fuente) con que se notificó lo recurrido, de la fuente que
    manda: lo que el secretario declaró en «Trámite en este tribunal»
    («secretario») > el escrito del recurso («escrito») > el auto de admisión
    («auto») > la resolución recurrida («acto»). («», «») si ninguna lo dice.
    Sin modelo: sólo mira lo ya leído."""
    for fuente, d in (("secretario", getattr(e, "tramite_declarado", None)),
                      ("escrito", escrito),
                      ("auto", getattr(e, "tramite_auto", None)),
                      ("acto", acto)):
        if isinstance(d, dict):
            forma = _forma_normalizada(d.get("forma_notificacion"))
            if forma:
                return forma, fuente
    return "", ""


def regla_de_la_autoridad(e, papel: str, forma_declarada: str = "",
                          fuente_forma: str = "secretario") -> tuple:
    """(regla, aviso) con que se computa el recurso de una AUTORIDAD en el
    amparo en revisión o en la queja; ("", "") si no hay nada que cambiar.

    SÓLO SI LA REGLA QUE LLEGÓ ES LA DE LOS PARTICULARES: «personal» (la de
    omisión) o «lista». Lo que el secretario eligió a propósito —«oficio»,
    «electronica»— y «otra», con la fecha en que él declara que surtió, se
    respeta. Si la notificación fue electrónica (`forma_declarada`, de la
    fuente `fuente_forma`: lo declarado o, con la bandera, lo leído), la regla
    es la de la fracción III; si no, el oficio de la fracción I. Las dos surten
    el mismo día (cero hábiles): lo que cambia entre ellas es el precepto que
    se cita, no la fecha."""
    import tipos_asunto as _ta_r
    if _ta_r.normalizar(str(getattr(e, "tipo_asunto", "") or "")) not in ("amparo_revision", "queja"):
        return "", ""
    if str(papel or "").strip().lower() != "autoridad":
        return "", ""
    llego = str(getattr(e, "regla_surtimiento", "") or "").strip().lower()
    if llego not in _REGLAS_DE_PARTICULARES:
        return "", ""
    _desc = getattr(f0.REGLAS_SURTE.get(llego), "descripcion", "") or llego
    _vieja = (f"notificación {_desc}, que surte al día hábil siguiente (fracción II), "
              f"la de los particulares")
    if _forma_normalizada(forma_declarada) == "electronica":
        _de = _DE_DONDE_LA_FORMA.get(fuente_forma) or _DE_DONDE_LA_FORMA["secretario"]
        return "electronica", (
            f"RECURRE UNA AUTORIDAD Y LA NOTIFICACIÓN FUE ELECTRÓNICA ({_de}): EL CÓMPUTO SE "
            "HIZO CON LA FRACCIÓN III del artículo 31 de la Ley de Amparo —surte efectos al "
            "generarse la constancia de la consulta—, y no con la regla que venía declarada "
            f"({_vieja}): el plazo arranca un día hábil antes. Si la constancia dice otra cosa, "
            "elige «otra regla» con la fecha en que surtió y vuelve a generar.")
    return "oficio", (
        "RECURRE UNA AUTORIDAD: EL CÓMPUTO SE HIZO CON LA NOTIFICACIÓN POR OFICIO, que surte "
        "efectos desde que queda hecha (artículo 31, fracción I, de la Ley de Amparo), y no "
        f"con la regla que venía declarada ({_vieja}): el plazo arranca un día hábil antes. "
        "Si a esa autoridad se le notificó por vía electrónica, elige «electrónica» "
        "(fracción III); si la constancia dice otra cosa, «otra regla» con la fecha en que "
        "surtió. Y vuelve a generar.")


def regla_por_forma(e, papel: str, forma: str, fuente: str = "") -> tuple:
    """(regla, aviso) cuando la forma de notificación CONSTA (declarada o
    leída) y la regla llegó como la de omisión, «personal», en la revisión o
    en la queja (3-oct-2026, D2; rev_0, Q 24/2026, Q 172/2026 y Q 300/2025).
    («», aviso) si la forma no puede ser la de quien recurre; («», «») si no
    hay nada que cambiar. La autoridad la decide antes `regla_de_la_autoridad`.

    SÓLO EN LOS RECURSOS DE AMPARO: ahí lo notificado es una resolución del
    juez de amparo y rige el artículo 31 de la Ley de Amparo. En el amparo
    directo y en la revisión fiscal la notificación del acto la rige su propia
    ley, y estas reglas no son las suyas."""
    import tipos_asunto as _ta_f
    if _ta_f.normalizar(str(getattr(e, "tipo_asunto", "") or "")) not in ("amparo_revision", "queja"):
        return "", ""
    if str(getattr(e, "regla_surtimiento", "") or "").strip().lower() != "personal":
        return "", ""
    _forma = _forma_normalizada(forma)
    _de = _DE_DONDE_LA_FORMA.get(fuente) or "así consta"
    _omision = "la regla de omisión (notificación personal, que surte al día hábil siguiente)"
    if _forma == "electronica":
        return "electronica", (
            f"LA NOTIFICACIÓN FUE ELECTRÓNICA ({_de}): EL CÓMPUTO SE HIZO CON LA FRACCIÓN III "
            "del artículo 31 de la Ley de Amparo —surte efectos al generarse la constancia de "
            f"la consulta—, y no con {_omision}: el plazo arranca un día hábil antes. "
            "Compruébalo en la constancia de notificación; si dice otra cosa, elige la regla "
            "que corresponda y vuelve a generar.")
    if _forma == "lista":
        return "lista", (
            f"LA NOTIFICACIÓN FUE POR LISTA ({_de}): el cómputo se hizo con esa forma y así la "
            "nombra el considerando. Surte al día hábil siguiente, como la personal (artículo "
            "31, fracción II, de la Ley de Amparo), así que las fechas no cambian. Compruébalo "
            "en la constancia de notificación.")
    if _forma == "oficio":
        _p = str(papel or "").strip().lower()
        if _p in ("quejoso", "tercero"):
            _quien = "la parte quejosa" if _p == "quejoso" else "la parte tercera interesada"
            return "", (
                f"LA NOTIFICACIÓN CONSTA COMO HECHA POR OFICIO ({_de}), pero quien recurre es "
                f"{_quien}, y por oficio se notifica a las autoridades: el cómputo siguió con "
                "la notificación personal. Coteja la constancia de notificación y, si hace "
                "falta, elige la regla que corresponda y vuelve a generar.")
        return "oficio", (
            f"LA NOTIFICACIÓN FUE POR OFICIO ({_de}): EL CÓMPUTO SE HIZO CON LA FRACCIÓN I del "
            "artículo 31 de la Ley de Amparo —surte efectos desde que queda hecha—, y no con "
            f"{_omision}: el plazo arranca un día hábil antes. Compruébalo en la constancia de "
            "notificación.")
    return "", ""


def regla_agraria(e, materia: str = "") -> tuple:
    """(regla, aviso) con que se computa el amparo directo AGRARIO cuando el
    colegiado reside en la Ciudad de México (3-oct-2026, guardián de materia);
    ("", "") si no hay nada que cambiar.

    SÓLO EN EL AMPARO DIRECTO: ahí la notificación del acto la rige su ley (la
    Agraria y, en lo que no dice, su supletorio, art. 167). En la revisión y en
    la queja lo notificado es una resolución del juez de amparo (art. 31 LA).
    SÓLO LO QUE LLEGA POR OMISIÓN: «personal» → `cnpcf_personal` (art. 227, fr.
    I, del Código Nacional: surte el mismo día) y «lista» → `cnpcf_lista` (art.
    211: surte al día siguiente, como antes; cambia el precepto). Lo agrario se
    reconoce por la materia o por la responsable («Tribunal Unitario Agrario…»);
    la sede, con `f0.sede_cdmx(tribunal, ciudad)` —la misma regla del
    supletorio de la Ley de Amparo—. Fuera de la Ciudad de México o sin saber
    la sede, nada: el considerando funda la de siempre en el 321 del CFPC.

    LA DEL CÓDIGO FEDERAL ELEGIDA A PROPÓSITO NO SE TOCA (integración,
    3-oct-2026): `cfpc_personal` es la clave de quien dice «el 321, de verdad»
    (el juicio empezó antes de que el Código Nacional rigiera para él); sólo la
    «personal» y la «lista» GENÉRICAS, las que manda el formulario sin que nadie
    elija, pasan por aquí (`f0.CNPCF_POR_OMISION`)."""
    import tipos_asunto as _ta_ag
    if _ta_ag.normalizar(str(getattr(e, "tipo_asunto", "") or "")) != "amparo_directo":
        return "", ""
    llego = str(getattr(e, "regla_surtimiento", "") or "").strip().lower()
    nueva = f0.CNPCF_POR_OMISION.get(llego, "")
    if not nueva:
        return "", ""
    _resp = str(getattr(e, "responsable", "") or "")
    _mat = str(materia or getattr(e, "materia", "") or "")
    if not f0.es_agrario(_mat, _resp):
        return "", ""
    if f0.sede_cdmx(getattr(e, "tribunal", "") or "", getattr(e, "ciudad", "") or "") is not True:
        return "", ""
    _r = f0.REGLAS_SURTE[nueva]
    _quien = (f"la responsable, «{' '.join(_resp.split())[:90]}», es un tribunal agrario"
              if f0.es_agrario("", _resp) else "el asunto está declarado como agrario")
    # EL TRANSITORIO YA NO VA AQUÍ (revisión de normas y front, 3-oct-2026): lo
    # dice `f0.aviso_cnpcf_agrario`, que `generar` añade SIEMPRE que se cuente con
    # el Código Nacional en la Ciudad de México —la regla la ponga este guardián
    # o la pantalla, que desde el desplegable la propone sola y por eso nunca
    # pasaba por aquí—, y que el considerando repite con el mismo texto (la
    # unión de avisos del proyecto lo deja una vez). Este aviso dice sólo qué
    # cambió y por qué. Y LA SEDE ES REGLA DE LA CASA, NO HECHO DE LA LEY: decía
    # «donde ya opera el Código Nacional», y para el orden federal el
    # transitorio Segundo ata el 167 reformado a la declaratoria del Congreso de
    # la Unión, no a la de la Ciudad de México.
    _por_que = (f"Se cambió porque {_quien} y este tribunal reside en la Ciudad de México, donde la "
                "regla del taller aplica el Código Nacional.")
    if nueva == "cnpcf_personal":
        return nueva, (
            "LO AGRARIO EN LA CIUDAD DE MÉXICO SE CUENTA CON EL CÓDIGO NACIONAL: el cómputo se "
            "hizo con la notificación personal del " + _r.fundamento + " —los términos corren "
            "desde el día siguiente al de la notificación: surte el mismo día—, y no con la regla "
            "que venía (notificación personal, surte al día hábil siguiente): el plazo arranca un "
            "día hábil antes. " + _por_que)
    return nueva, (
        "LO AGRARIO EN LA CIUDAD DE MÉXICO SE FUNDA EN EL CÓDIGO NACIONAL: la notificación por "
        "lista surte al día siguiente de su publicación, conforme al " + _r.fundamento + ". Las "
        "fechas no cambian; cambia el precepto que cita el considerando. " + _por_que)


def regla_agraria_por_forma(e, forma: str, fuente: str = "") -> tuple:
    """(regla, aviso) cuando el amparo directo agrario se cuenta con la personal
    del Código Nacional y la forma de notificación que CONSTA es otra; («», «»)
    si no hay nada que cambiar (revisión AD, 3-oct-2026).

    EN LA CIUDAD DE MÉXICO LA FORMA CAMBIA LA FECHA. `cnpcf_personal` llega sola
    (la propone la pantalla o la pone el guardián) y surte el mismo día; si la
    ficha dice que la notificación fue por lista —declarada en la tarjeta o
    leída—, nada la cambiaba a la del 211, que surte al día siguiente: el plazo
    arrancaba un día hábil antes y el considerando decía «de manera personal»
    contra la ficha. Antes de hoy la personal y la lista contaban igual en el
    amparo directo y la discrepancia era sólo de texto. La electrónica pasa a la
    del 227, fracción III (mismo día: cambia el precepto). Sólo se toca la
    personal del Código Nacional: lo demás se eligió a propósito."""
    import tipos_asunto as _ta_pf
    if _ta_pf.normalizar(str(getattr(e, "tipo_asunto", "") or "")) != "amparo_directo":
        return "", ""
    if str(getattr(e, "regla_surtimiento", "") or "").strip().lower() != "cnpcf_personal":
        return "", ""
    _forma = _forma_normalizada(forma)
    _de = _DE_DONDE_LA_FORMA.get(fuente) or "así consta"
    if _forma == "lista":
        return "cnpcf_lista", (
            f"LA NOTIFICACIÓN FUE POR LISTA ({_de}): EL CÓMPUTO SE HIZO CON EL ARTÍCULO 211 DEL "
            "CÓDIGO NACIONAL DE PROCEDIMIENTOS CIVILES Y FAMILIARES —la publicación surte al día "
            "siguiente—, y no con la notificación personal del artículo 227, fracción I (los "
            "términos corren desde el día siguiente al de la notificación): el plazo arranca un día "
            "hábil después. Compruébalo en la constancia de notificación.")
    if _forma == "electronica":
        return "cnpcf_electronica", (
            f"LA NOTIFICACIÓN FUE ELECTRÓNICA ({_de}): el cómputo se hizo con el artículo 227, "
            "fracción III, del Código Nacional de Procedimientos Civiles y Familiares —surte el mismo "
            "día en que el sistema confirma la recepción— y así la nombra el considerando. Si la "
            "recepción se confirmó otro día, elige «Otra regla» con esa fecha y vuelve a generar.")
    return "", ""


def _agrario_con_el_codigo_nacional(e, c, avisos: list, computar_con, aviso_guardian: str = "",
                                    forma: str = "", fuente_forma: str = ""):
    """El amparo directo agrario contado con el Código Nacional, después de
    leer los papeles (revisión AD y de normas y front, 3-oct-2026). Devuelve el
    cómputo (rehecho si la forma lo pide); nunca lanza.

      1. LA FORMA QUE CONSTA (`regla_agraria_por_forma`): con la personal del
         Código Nacional y la notificación por lista, la del 211 (el plazo
         arranca un día hábil después); electrónica, la del 227-III. El aviso
         del guardián, que hablaba de la personal, se quita.
      2. EL TRANSITORIO, SIEMPRE (`f0.surtimiento_nacional` → «», aviso): la
         regla del Código Nacional llega sola desde la pantalla y el guardián
         no la veía; el mismo texto lo repite el considerando.
      3. EXTEMPORÁNEA CON EL CÓDIGO NACIONAL Y EN TIEMPO CON EL 321
         (`f0.contraste_cnpcf_agrario`): el aviso con las dos fechas."""
    try:
        regla, aviso = regla_agraria_por_forma(e, forma, fuente_forma)
        if regla:
            if aviso_guardian:
                avisos[:] = [a for a in avisos if a != aviso_guardian]
            _antes = e.regla_surtimiento
            nuevo = computar_con(regla)
            _cambiar_computo(
                e, c, nuevo, avisos, aviso, regla=regla,
                fuera_de_plazo=(
                    "CON LA FORMA DE NOTIFICACIÓN QUE CONSTA LA DEMANDA SALE FUERA DE PLAZO, y con la "
                    "personal del Código Nacional salía en tiempo: el día en que surte la notificación "
                    "decide la oportunidad. Coteja la constancia de notificación antes de firmar."))
            print(f"   ⏱️ AGRARIO, FORMA DE NOTIFICACIÓN: cómputo rehecho con «{regla}» (venía «{_antes}»)")
            c = nuevo
        _regla = str(getattr(e, "regla_surtimiento", "") or "")
        if _regla.startswith("cnpcf_"):
            _pn, _av_cn = f0.surtimiento_nacional(
                getattr(e, "tipo_asunto", "") or "amparo_directo",
                str(getattr(e, "materia", "") or ""), str(getattr(e, "responsable", "") or ""),
                _regla, tribunal=getattr(e, "tribunal", "") or "", ciudad=getattr(e, "ciudad", "") or "")
            if _av_cn and not _pn and _av_cn not in avisos:
                avisos.append(_av_cn)
            if _regla == "cnpcf_personal" and getattr(c, "oportuna", None) is False:
                _av_x = f0.contraste_cnpcf_agrario(c, computar_con("cfpc_personal"))
                if _av_x and _av_x not in avisos:
                    avisos.append(_av_x)
    except Exception as _ex:
        print(f"   ⚠️ agrario con el Código Nacional sin revisar: {type(_ex).__name__}: {str(_ex)[:120]}")
    return c


def _cambiar_computo(e, c, nuevo, avisos: list, aviso: str, fuera_de_plazo: str = "",
                     regla: str = "") -> None:
    """Pone en `avisos` (en su sitio) el paso del cómputo `c` al `nuevo`: se
    quitan los del cómputo viejo que el nuevo ya no dice —y el de
    extemporánea si dejó de serlo—, se añaden los nuevos y `aviso`, que
    explica el cambio. `fuera_de_plazo`, si el nuevo sale fuera y el viejo no.
    Con `regla`, la deja en el encargo (la sesión la guarda y el worker que
    resuelva recomputa con ella) y quita el «se hizo con notificación
    PERSONAL» de los guardianes, que deja de ser verdad."""
    if regla:
        e.regla_surtimiento = regla
    _nuevos = list(getattr(nuevo, "avisos", None) or [])
    _caducos = [a for a in (getattr(c, "avisos", None) or []) if a not in _nuevos]
    if nuevo.oportuna is not False:
        _caducos.append(AVISO_EXTEMPORANEA)
    avisos[:] = [a for a in avisos
                 if a not in _caducos
                 and not (regla and str(a).startswith(_AVISO_CONTO_PERSONAL))]
    for a in _nuevos:
        if a not in avisos:
            avisos.append(a)
    if nuevo.oportuna is False and AVISO_EXTEMPORANEA not in avisos:
        avisos.append(AVISO_EXTEMPORANEA)
    if aviso:
        avisos.append(aviso)
    if fuera_de_plazo and nuevo.oportuna is False and c.oportuna is not False:
        avisos.append(fuera_de_plazo)


def _recomputar_para_la_autoridad(e, c, fases, partes, texto_acto: str, avisos: list,
                                  computar_con, *, forma: str = "", fuente_forma: str = "",
                                  por_forma: bool = False):
    """El cómputo `c` rehecho con la regla que corresponde a quien recurre y
    a la forma de notificación que consta; `c` tal cual si no hay nada que
    cambiar. `computar_con(regla)` es el cómputo de `generar` con todos sus
    datos y sólo la regla como variable.

    1. LA AUTORIDAD QUE RECURRE (`regla_de_la_autoridad`) [siempre desde el
       3-oct-2026, D2]: oficio (fr. I) o, si la notificación fue
       electrónica, la fr. III.
    2. CON LA BANDERA (`por_forma`), LA FORMA QUE CONSTA (`regla_por_forma`):
       `forma` y `fuente_forma` vienen de `forma_de_notificacion`. Sin la
       bandera la forma sólo puede venir de lo declarado, que sin ella no
       llega (main no lo lee).

    EL PAPEL, DE LA MISMA FUENTE QUE LOS CONSIDERANDOS: `papel_del_recurrente`
    con el resolutivo del juzgado, que es lo que usa `ficha_procesal.armar` y,
    de ella, la legitimación y la oportunidad del documento. Así el cómputo y
    el párrafo que lo funda nunca discrepan sobre quién recurre.

    Nunca lanza: si algo falla, el cómputo de antes y los avisos intactos."""
    try:
        _papel = papel_del_recurrente(e, partes, _resolutivo_del_a_quo(fases, texto_acto))
        if por_forma:
            _forma, _fuente = forma, (fuente_forma or "secretario")
        else:
            _forma = str((getattr(e, "tramite_declarado", None) or {}).get("forma_notificacion") or "")
            _fuente = "secretario"
        regla, aviso = regla_de_la_autoridad(e, _papel, _forma, _fuente)
        _de_la_autoridad = bool(regla)
        if not regla and por_forma and _forma:
            regla, aviso = regla_por_forma(e, _papel, _forma, _fuente)
            if not regla:
                if aviso and aviso not in avisos:
                    avisos.append(aviso)
                return c
        if not regla:
            return c
        nuevo = computar_con(regla)
    except Exception as _ex:
        print(f"   ⚠️ cómputo del recurso sin rehacer con su regla: "
              f"{type(_ex).__name__}: {str(_ex)[:120]}")
        return c
    _antes = e.regla_surtimiento
    _cambiar_computo(
        e, c, nuevo, avisos, aviso, regla=regla,
        fuera_de_plazo=(
            "CON LA REGLA DE LA AUTORIDAD EL RECURSO SALE FUERA DE PLAZO, y con la de los "
            "particulares que venía declarada salía en tiempo: el día en que surte la "
            "notificación decide la oportunidad. Coteja la constancia de notificación antes "
            "de firmar." if _de_la_autoridad else
            "CON LA FORMA DE NOTIFICACIÓN QUE CONSTA EL RECURSO SALE FUERA DE PLAZO, y con la "
            "regla de omisión (notificación personal) salía en tiempo: el día en que surte la "
            "notificación decide la oportunidad. Coteja la constancia de notificación antes "
            "de firmar."))
    print(f"   ⏱️ {'RECURRE UNA AUTORIDAD' if _de_la_autoridad else 'FORMA DE NOTIFICACIÓN'}: "
          f"cómputo rehecho con «{regla}» (venía «{_antes}») · "
          f"{'en tiempo' if nuevo.oportuna else 'EXTEMPORÁNEO' if nuevo.oportuna is False else 'sin plazo'}")
    return nuevo


# ═══════════════════════════════════════════════════════════════════════════
# LA REVISIÓN FISCAL POR CORREO: LA FECHA DEL DEPÓSITO (3-oct-2026, D3)
# ═══════════════════════════════════════════════════════════════════════════
# El integrador decidió (D3) medir la oportunidad con la fecha en que el oficio
# se depositó en el Servicio Postal Mexicano cuando el recurso viajó por
# correo y consta el depósito: es la práctica medida en los engroses de
# revisión fiscal del banco (criterio XV.4o.1 A (11a.)). En RF 2/2025 el
# depósito caía en plazo y la recepción no, y el proyecto desechaba por
# extemporáneo un recurso que el tribunal confirmó. Va con la bandera: el
# depósito sólo existe en la ficha de trámite.

# UNA SOLA REGLA PARA EL CORREO (cuarta ronda, 3-oct-2026, E6). En RF 2/2025
# la ficha traía la vía «responsable» Y el depósito del 24 de octubre: el
# compositor narraba el depósito («mediante oficio depositado en el Servicio
# Postal Mexicano…») porque le bastaba la fecha, y aquí se descartaba porque
# la vía no decía «postal». El documento se contradecía solo: el resultando
# decía que se depositó en plazo y el considerando desechaba por
# extemporáneo. Un oficio que viaja por correo TAMBIÉN lo recibe la
# responsable, así que esa ficha es verosímil. La regla es una
# (`ficha_tramite.via_postal`): depósito presente y anterior o igual a la
# presentación ⇒ por correo, diga lo que diga la vía; la usan el compositor,
# esta puerta y el worker que rehidrata la sesión (main.py).
def via_postal_de(ficha, presentacion=None) -> bool:
    """¿El recurso viajó por correo? `ficha_tramite.via_postal(ficha)` con la
    presentación que se tecleó (la del cómputo) en lugar de la de la ficha,
    si se da. Mientras esa pieza no exista —o si lanza—, la misma regla aquí:
    hay depósito y no es posterior a la presentación. Nunca lanza."""
    if not isinstance(ficha, dict):
        return False
    f = dict(ficha)
    if isinstance(presentacion, _dt.date):
        f["presentacion"] = presentacion.isoformat()
    try:
        import ficha_tramite as _ft
        _vp = getattr(_ft, "via_postal", None)
        if callable(_vp):
            return bool(_vp(f))
    except Exception as _ex:
        print(f"   ⚠️ via_postal de la ficha falló; se aplica la misma regla aquí: "
              f"{type(_ex).__name__}: {str(_ex)[:120]}")
    dep = _fecha_iso(f.get("deposito_postal"))
    if dep is None:
        return False
    pres = _fecha_iso(f.get("presentacion"))
    return pres is None or dep <= pres


def deposito_que_cuenta(e):
    """La fecha del depósito postal con que se mide la oportunidad de una
    revisión fiscal por correo, o None. Sale de la ficha de trámite del
    encargo (`deposito_postal`) cuando el recurso viajó por correo según la
    regla única (`via_postal_de`, E6: aunque la vía diga «responsable»); la
    misma puerta para el adelanto y para la sesión que rehidrata el otro
    worker. None también si el depósito no es ANTERIOR a la presentación que
    se tecleó: igual no cambia nada, y posterior es imposible (lo acusa la
    ficha)."""
    if _tipo_de_encargo(e) != "revision_fiscal":
        return None
    t = getattr(e, "tramite", None)
    if not isinstance(t, dict):
        return None
    dep = _fecha_iso(t.get("deposito_postal"))
    if dep is None:
        return None
    pres = getattr(e, "presentacion", None)
    if not via_postal_de(t, pres):
        return None
    if pres is not None and dep >= pres:
        return None
    return dep


def _aviso_via_con_deposito(e, dep, avisos) -> str:
    """El aviso de que la vía de la ficha dice otra cosa y aun así se contó
    el depósito (E6), o «» si la vía dice «postal» o nada, o si ya lo dijo
    otra pieza (la validación de la ficha): un aviso por dato (E11)."""
    t = getattr(e, "tramite", None)
    if not isinstance(t, dict) or dep is None:
        return ""
    via = " ".join(str(t.get("via_presentacion") or "").split())
    if not via or _plano_min(via) == "postal":
        return ""
    _vp = _plano_min(via)
    for a in list(avisos or []) + list(t.get("avisos") or []):
        _a = _plano_min(a)
        if "deposit" in _a and re.search(r"\bvia\b", _a) and _vp in _a:
            return ""
    return (f"LA VÍA DE PRESENTACIÓN DE LA FICHA DICE «{via}», PERO CONSTA UN DEPÓSITO POSTAL "
            f"({f0.fecha_en_letra(dep)}) ANTERIOR A LA RECEPCIÓN: se tomó como enviado por correo "
            "—un oficio que llega por correo también lo recibe la responsable— y la oportunidad "
            "se midió con el depósito, igual que lo narra el resultando. Si el oficio no viajó "
            "por correo, borra la fecha del depósito en «Trámite en este tribunal» y vuelve a "
            "generar.")


def args_de_presentacion(e, deposito=None) -> tuple:
    """(fecha que va como `presentacion` a `fase0_oportunidad.computar`,
    kwargs de más). Con `deposito`: si `computar` sabe recibirlo (parámetro
    `deposito` o `deposito_postal`), la presentación sigue siendo la
    recepción y el depósito va aparte, para que el párrafo diga las dos; si
    no, el depósito ocupa el lugar de la presentación, que es la fecha con
    que se mide la oportunidad. Sin `deposito`, la presentación tecleada."""
    pres = getattr(e, "presentacion", None)
    if deposito is None:
        return pres, {}
    try:
        import inspect as _insp
        _ps = _insp.signature(f0.computar).parameters
    except Exception:
        _ps = {}
    for nombre in ("deposito", "deposito_postal"):
        if nombre in _ps:
            return pres, {nombre: deposito}
    return deposito, {}


def marcar_deposito(c, deposito, recepcion=None):
    """El cómputo con `deposito` (y la `recepcion` en la Sala) colgados si
    `computar` no los dejó: el párrafo y la sesión pueden saber así que la
    oportunidad se midió con el depósito. Nunca lanza."""
    if c is None or deposito is None:
        return c
    for k, v in (("deposito", deposito), ("recepcion", recepcion)):
        if v is not None and getattr(c, k, None) in (None, ""):
            try:
                setattr(c, k, v)
            except Exception:
                pass
    return c


def _recomputar_con_deposito(e, c, avisos: list, computar_con):
    """El cómputo `c` medido con el depósito postal (`deposito_que_cuenta`),
    con su aviso; `c` tal cual si no consta o algo falla (nunca lanza)."""
    try:
        if _tipo_de_encargo(e) != "revision_fiscal" or not isinstance(getattr(e, "tramite", None), dict):
            return c
        _dep_crudo = _fecha_iso((e.tramite or {}).get("deposito_postal"))
        _pres = getattr(e, "presentacion", None)
        if _dep_crudo is not None and _pres is not None and _dep_crudo > _pres:
            _a = (f"EL DEPÓSITO POSTAL ({f0.fecha_en_letra(_dep_crudo)}) ES POSTERIOR A LA "
                  f"PRESENTACIÓN QUE TECLEASTE ({f0.fecha_en_letra(_pres)}): no puede ser. La "
                  "oportunidad se midió con la presentación; corrige la fecha que esté mal y "
                  "vuelve a generar.")
            if _a not in avisos:
                avisos.append(_a)
            return c
        dep = deposito_que_cuenta(e)
        if dep is None:
            return c
        nuevo = computar_con(e.regla_surtimiento, deposito=dep)
    except Exception as _ex:
        print(f"   ⚠️ cómputo con el depósito postal sin rehacer: {type(_ex).__name__}: {str(_ex)[:120]}")
        return c
    # UN SOLO AVISO. Si `computar` recibió el depósito, su propio aviso ya
    # dice con qué fecha midió y qué pasaba con la recepción (y viaja en
    # `nuevo.avisos`); éste sólo sale cuando el depósito tuvo que ocupar el
    # lugar de la presentación (`args_de_presentacion`).
    _ya_lo_dice = any("DEPOSIT" in _plano_min(a).upper() and "SE MIDIO" in _plano_min(a).upper()
                      for a in (getattr(nuevo, "avisos", None) or []))
    aviso = ""
    if not _ya_lo_dice:
        aviso = (f"REVISIÓN FISCAL POR CORREO: LA OPORTUNIDAD SE MIDIÓ CON LA FECHA EN QUE EL OFICIO "
                 f"SE DEPOSITÓ EN EL SERVICIO POSTAL MEXICANO ({f0.fecha_en_letra(dep)}), y no con la "
                 f"de su recepción ({f0.fecha_en_letra(_pres)}), conforme al criterio XV.4o.1 A "
                 "(11a.). Compruébalo contra el sobre o la guía postal; si el oficio no viajó por "
                 "correo, borra la fecha del depósito en «Trámite en este tribunal» y vuelve a "
                 "generar.")
        if nuevo.oportuna is not False and c.oportuna is False:
            aviso += (" CON LA FECHA DE RECEPCIÓN EL RECURSO SALÍA EXTEMPORÁNEO: el depósito "
                      "decide la oportunidad.")
    _cambiar_computo(e, c, nuevo, avisos, aviso)
    # LA VÍA DECÍA OTRA COSA (E6): se contó el depósito por la regla única, y
    # se dice por qué, una vez (si la validación de la ficha no lo dijo ya).
    _a_via = _aviso_via_con_deposito(e, dep, avisos)
    if _a_via:
        avisos.append(_a_via)
    print(f"   ⏱️ RF POR CORREO: cómputo con el depósito del {dep.isoformat()} · "
          f"{'en tiempo' if nuevo.oportuna else 'EXTEMPORÁNEO' if nuevo.oportuna is False else 'sin plazo'}")
    return nuevo


def _con_valor(v) -> bool:
    """¿Trae dato? Un dict cuenta si alguno de sus valores lo trae."""
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, dict):
        return any(_con_valor(x) for x in v.values())
    if isinstance(v, (list, tuple, set)):
        return any(_con_valor(x) for x in v)
    return True


def _unir_avisos(*listas) -> list:
    """Las listas de avisos en orden, sin repetir ninguno."""
    fuera = []
    for lista in listas:
        for a in (lista or []):
            if a and a not in fuera:
                fuera.append(a)
    return fuera


def _a_json(x):
    """Lo que `json` no sabe escribir, en texto: fechas en ISO, conjuntos en
    lista. La ficha va a la sesión (jsonb) y a la pantalla."""
    if isinstance(x, (_dt.date, _dt.datetime)):
        return x.isoformat()
    if isinstance(x, (set, frozenset, tuple)):
        return list(x)
    return str(x)


def tramite_serializable(t) -> Optional[dict]:
    """La ficha de trámite tal como se guarda en `estado.encargo.tramite`:
    sólo tipos de JSON. None si no es una ficha o no se puede escribir (la
    sesión se guarda igual, sin ella, y el proyecto vuelve al camino de
    siempre: mejor eso que un adelanto que no se guarda)."""
    if not isinstance(t, dict):
        return None
    try:
        import json as _js
        return _js.loads(_js.dumps(t, ensure_ascii=False, default=_a_json))
    except Exception:
        return None


def _tomar_ruta(d, ruta: str):
    """El valor de «acto.fecha» en un dict anidado, o None."""
    x = d
    for k in str(ruta or "").split("."):
        if not isinstance(x, dict) or k not in x:
            return None
        x = x[k]
    return x


def _poner_ruta(d: dict, ruta: str, valor) -> None:
    """Pone `valor` en «acto.fecha» de un dict anidado, creando lo que falte."""
    partes_r = str(ruta or "").split(".")
    x = d
    for k in partes_r[:-1]:
        if not isinstance(x.get(k), dict):
            x[k] = {}
        x = x[k]
    x[partes_r[-1]] = valor


# ═══════════════════════════════════════════════════════════════════════════
# EL CARGO DEL PONENTE VIAJA CON SU NOMBRE (cuarta ronda, 3-oct-2026, E3)
# ═══════════════════════════════════════════════════════════════════════════
# AD 128, AD 279, AD 552, RF 4, RF 49 y Q 342 del banco: la carátula decía
# «MAGISTRADO PONENTE: JENICA CAMPOS JUÁREZ» y el auto de returno decía «a la
# ponencia a cargo de la magistrada Jenica Campos Juárez». `leer_auto` sí lee
# el cargo (`turno.titulo` / `returno.titulo`), pero el formulario «Trámite en
# este tribunal» sólo tiene el nombre: lo que /taller/desde-admision propone y
# lo que la pantalla pinta al retomar es `a_formulario`, sin cargo; el
# secretario lo confirma, vuelve como `tramite_json` con fuente «secretario»
# y el cargo leído se pierde por el camino. Va DENTRO del valor
# («Magistrada Jenica Campos Juárez»), que es lo que la carátula
# (`documento_generado._ponente_sin_cargo`) y la prosa del turno ya saben
# leer. El cargo sólo sale de lo escrito en el auto; nunca del nombre de pila.
_PONENTES_FORMULARIO = (("ponente_turno", "turno"), ("ponente_returno", "returno"))
_RX_YA_CON_CARGO = re.compile(
    r"^\s*(?:(?:el|la)\s+)?(?:magistrad[oa]|secretari[oa]|juez|jueza)\b", re.I)
_RX_TITULO_PONENTE = re.compile(r"^(?:magistrad[oa]|secretari[oa])\b", re.I)


def _grupo_del_ponente(ficha: dict, grupo: str) -> dict:
    """El dict del turno o del returno; del returno, el ÚLTIMO si viene como
    lista (como lo lee la carátula)."""
    g = ficha.get(grupo) if isinstance(ficha, dict) else None
    if isinstance(g, list):
        g = next((x for x in reversed(g) if isinstance(x, dict) and x.get("ponente")), None)
    return g if isinstance(g, dict) else {}


def falta_cargo_del_ponente(planas) -> bool:
    """¿Alguno de los ponentes del formulario trae nombre sin cargo?"""
    if not isinstance(planas, dict):
        return False
    for clave, _g in _PONENTES_FORMULARIO:
        v = " ".join(str(planas.get(clave) or "").split())
        if v and not _RX_YA_CON_CARGO.match(v):
            return True
    return False


def ponentes_con_cargo(planas, ficha, fuente=None) -> dict:
    """Las claves planas del formulario con el cargo del auto delante del
    nombre del ponente: «Jenica Campos Juárez» + `returno.titulo`
    «Magistrada» → «Magistrada Jenica Campos Juárez». Nunca lanza; devuelve
    una copia.

    No se toca el valor que ya trae un cargo, ni se pega un cargo que no sea
    de magistrado o de secretario en funciones. EL CARGO Y EL NOMBRE, DE LA
    MISMA FUENTE: si el secretario tecleó a otra persona y el «Magistrada»
    era del auto, no se le pega. Con `fuente` (la sub-ficha de una sola
    fuente en `tramite_para_pantalla`), sólo el cargo de esa fuente —o sin
    fuente, si el nombre es de ella—."""
    out = dict(planas) if isinstance(planas, dict) else {}
    if not isinstance(ficha, dict):
        return out
    try:
        fuentes = ficha.get("fuentes") if isinstance(ficha.get("fuentes"), dict) else {}
        for clave, grupo in _PONENTES_FORMULARIO:
            v = " ".join(str(out.get(clave) or "").split())
            if not v or _RX_YA_CON_CARGO.match(v):
                continue
            _g = _grupo_del_ponente(ficha, grupo)
            tit = " ".join(str(_g.get("titulo") or "").split()).strip(" .,;")
            if not tit or len(tit) > 90 or not _RX_TITULO_PONENTE.match(tit):
                continue
            # EL CARGO ES DE ESA PERSONA: si la ficha nombra a otra (otra
            # lectura, otro auto), no se pega.
            _pf, _pv = _plano_min(_g.get("ponente")), _plano_min(v)
            if _pf and not (_pf.endswith(_pv) or _pv.endswith(_pf)):
                continue
            f_t = str(fuentes.get(f"{grupo}.titulo") or "")
            f_p = str(fuentes.get(f"{grupo}.ponente") or "")
            if f_t and f_p and f_t != f_p:
                continue
            if fuente is not None and (f_t or f_p) != fuente:
                continue
            out[clave] = f"{tit[:1].upper()}{tit[1:]} {v}"
    except Exception as _ex:
        print(f"   ⚠️ cargo del ponente sin poner: {type(_ex).__name__}: {str(_ex)[:120]}")
    return out


def tramite_para_pantalla(ficha) -> dict:
    """Lo que la pantalla pinta al RETOMAR un asunto, separado por su fuente
    (rev_6, 3-oct-2026), con las claves planas del formulario
    (`ficha_tramite.a_formulario`):

      {"tramite": las claves con SÓLO lo que el secretario confirmó (lo demás «»),
       "leido":   {clave: valor} de lo que vino del auto, del acto, del escrito
                  o de la ficha procesal —sin pisar lo del secretario—,
       "fuentes": {clave: fuente} de cada clave con valor}

    {} si no hay ficha o no se puede convertir. Nunca lanza.

    POR QUÉ. Devolver `a_formulario(ficha)` entero ponía lo leído en la
    tarjeta como si lo hubiera tecleado él; al volver a generar viajaba en
    `tramite_json`, `de_formulario` le ponía fuente «secretario» y le ganaba a
    la relectura. Medido con las funciones reales en un AR: se retomó, se
    subió el acto correcto (20-ene-2026, Juzgado Segundo, 77/2025) y la ficha
    conservó 15-ene, Juzgado Primero y 905/2025 como «del secretario», el
    juzgado sin aviso. Lo leído vuelve como leído: la pantalla lo marca y no
    lo reenvía mientras él no lo toque."""
    if not isinstance(ficha, dict):
        return {}
    try:
        import ficha_tramite as _ft
        completo = _ft.a_formulario(ficha)
        if not isinstance(completo, dict) or not completo:
            return {}
        fuentes = ficha.get("fuentes") if isinstance(ficha.get("fuentes"), dict) else {}

        def _solo(de_esta):
            sub = {"tipo": ficha.get("tipo")} if ficha.get("tipo") else {}
            for ruta, fu in fuentes.items():
                if fu != de_esta:
                    continue
                v = _tomar_ruta(ficha, ruta)
                if v is not None:
                    _poner_ruta(sub, ruta, v)
            _f = _ft.a_formulario(sub) if len(sub) > ("tipo" in sub) else {}
            # EL CARGO DEL PONENTE, DENTRO DE SU VALOR (E3), sólo si es de
            # esta misma fuente (`ponentes_con_cargo`).
            return ponentes_con_cargo(_f, ficha, fuente=de_esta) if isinstance(_f, dict) else {}

        # EL CARGO DEL PONENTE (E3, cuarta ronda): el formulario no tiene campo
        # para él y al retomar se perdía; va dentro del valor del ponente.
        completo = ponentes_con_cargo(completo, ficha)
        del_secretario = _solo("secretario")
        tramite = {k: (str(del_secretario.get(k) or "")) for k in completo}
        leido = {k: str(v) for k, v in completo.items()
                 if str(v or "").strip() and not str(del_secretario.get(k) or "").strip()}
        fuentes_k = {k: "secretario" for k, v in tramite.items() if v.strip()}
        _otras = []
        for fu in fuentes.values():
            if fu != "secretario" and fu not in _otras:
                _otras.append(fu)
        _por_fuente = {fu: _solo(fu) for fu in _otras}
        for k, v in leido.items():
            fuentes_k[k] = next((fu for fu in _otras if str(_por_fuente[fu].get(k) or "") == v), "")
        return {"tramite": tramite, "leido": leido, "fuentes": fuentes_k}
    except Exception as _ex:
        print(f"   ⚠️ trámite sin separar para la pantalla: {type(_ex).__name__}: {str(_ex)[:120]}")
        return {}


def tramite_de_formulario(texto: str) -> tuple:
    """Lo que el secretario confirmó en «Trámite en este tribunal»
    (`tramite_json`, claves planas del formulario), ya en forma de ficha con
    fuente «secretario» (`ficha_tramite.de_formulario`).

    Devuelve (declarado, error). `error` no vacío = el JSON está mal escrito y
    hay que contestar 422 ANTES de gastar el OCR. Si lo que falla es la pieza
    que lo interpreta —un fallo nuestro, no del secretario— no se le bloquea:
    sale un aviso que lo dice y la ficha se arma con lo leído."""
    t = str(texto or "").strip()
    if not t:
        return {}, ""
    try:
        import json as _js
        form = _js.loads(t)
    except Exception:
        return {}, "«tramite_json» no es JSON válido"
    if not isinstance(form, dict):
        return {}, "«tramite_json» debe ser un objeto con las claves del formulario"
    try:
        import ficha_tramite as _ft
        # POR HTTP, SÓLO LAS CLAVES DEL FORMULARIO (rev_6, 3-oct-2026). Con
        # `{"presentacion": "2026-02-05"}` metido en el JSON, el resultando
        # decía «presentado el cinco de febrero» y la oportunidad contaba con
        # el tres que se tecleó arriba; con `sentido`, un texto libre se
        # saltaba el catálogo del §6. Las claves de más quedan para usos
        # internos explícitos (`de_formulario` directo), no para la pantalla.
        _claves = tuple(getattr(_ft, "CLAVES_FORMULARIO", ()) or ())
        _fuera = sorted(str(k) for k in form if _claves and k not in _claves)
        if _fuera:
            form = {k: v for k, v in form.items() if k in _claves}
        d = _ft.de_formulario(form)
        d = d if isinstance(d, dict) else {}
        if _fuera:
            d["avisos"] = _unir_avisos(d.get("avisos"), [
                "«TRÁMITE EN ESTE TRIBUNAL» TRAÍA CLAVES QUE NO SON DEL FORMULARIO ("
                + ", ".join(_fuera[:8]) + ("…" if len(_fuera) > 8 else "")
                + "): no se usaron. Las fechas de notificación y presentación van en la ficha "
                  "del asunto."])
        return d, ""
    except Exception as _ex:
        print(f"   ⚠️ FICHA DE TRÁMITE: el formulario no se pudo leer "
              f"({type(_ex).__name__}: {str(_ex)[:120]})")
        return {"avisos": [
            "LO QUE CONFIRMASTE EN «TRÁMITE EN ESTE TRIBUNAL» NO SE PUDO LEER "
            f"({type(_ex).__name__}). Los resultandos salen con lo leído del auto y del "
            "acto: compruébalos contra lo que tecleaste."]}, ""


# UN AUTO DE PRESIDENCIA CABE AQUÍ (rev_6, 3-oct-2026): la lectura crecía al
# cuadrado con el largo —15 s con un documento de 166 páginas subido como
# «auto de admisión»— y corría en el bucle de eventos, parando el worker para
# todos. Se lee el principio, con aviso si se recortó; y main la llama en un
# hilo (`asyncio.to_thread`).
TOPE_AUTO = 30000


def leer_auto_tramite(texto: str, tipo: str = "") -> dict:
    """Lo que dice el auto de admisión (o de turno) para la ficha de trámite,
    SIN modelo (`ficha_tramite.leer_auto`). Nunca lanza: si el auto no trae
    texto o la lectura falla, un dict con su aviso y nada más. Lee a lo sumo
    los primeros `TOPE_AUTO` caracteres (con aviso si había más). Pura y sin
    estado: se puede llamar desde un hilo."""
    t = str(texto or "").strip()
    if len(t) < 120:
        return {"avisos": [
            "DEL AUTO DE ADMISIÓN QUE SUBISTE NO SALIÓ TEXTO LEGIBLE"
            + (f" (sólo {len(t)} caracteres)" if t else "")
            + ": la fecha del auto de Presidencia, el turno y el ponente salen de lo que "
            "confirmaste en el formulario o en hueco."]}
    _recorte = []
    if len(t) > TOPE_AUTO:
        _recorte = [
            f"EL AUTO DE ADMISIÓN QUE SUBISTE TRAE {len(t):,} CARACTERES".replace(",", " ")
            + f": se leyeron sólo los primeros {TOPE_AUTO:,}".replace(",", " ")
            + ", que es donde va un auto de Presidencia o de turno. Si subiste el expediente "
            "entero, sube sólo el auto y vuelve a leerlo; si no, comprueba en el formulario la "
            "fecha del auto, el turno y el ponente."]
        t = t[:TOPE_AUTO]
    try:
        import ficha_tramite as _ft
        d = _ft.leer_auto(t, tipo or "")
        d = dict(d) if isinstance(d, dict) else {}
        if _recorte:
            d["avisos"] = _unir_avisos(d.get("avisos"), _recorte)
        return d
    except Exception as _ex:
        print(f"   ⚠️ FICHA DE TRÁMITE: el auto no se pudo leer ({type(_ex).__name__}: {str(_ex)[:120]})")
        return {"avisos": [
            f"EL AUTO DE ADMISIÓN NO SE PUDO LEER ({type(_ex).__name__}): la fecha del auto "
            "de Presidencia, el turno y el ponente salen de lo que confirmaste en el "
            "formulario o en hueco."]}


# Lo que se toma del auto de TURNO antes que del de admisión: el turno y el
# returno. Todo lo demás —el auto de Presidencia, el número, el Ministerio
# Público, el adhesivo— manda el de admisión.
_DEL_AUTO_DE_TURNO = ("turno", "returno")
_LISTAS_DE_FICHA = ("avisos", "fuentes", "fechas_imposibles")


def _aviso_dos_autos_de_presidencia(adm, fundido) -> list:
    """[aviso] si dos autos de Presidencia del expediente dan fechas de
    admisión DISTINTAS y ninguno se leyó como el que registra (E7, cuarta
    ronda); [] si no.

    RF 7/2025: el 10 de febrero se radicó el recurso (y se declinó la
    competencia) y el 15 de mayo se admitió; la ficha sólo tenía el campo de
    la admisión y el resultando decía que el auto del 15 de mayo «registró el
    recurso… y lo admitió». Ya hay campo para el registro (`registro.fecha`,
    «fecha_registro»); si la lectura no los distinguió, aquí se toma el
    primero como la admisión (como siempre) y se dice, sin adivinar cuál es
    cuál."""
    if _con_valor((fundido or {}).get("registro")):
        return []
    fechas = []
    for d in adm or []:
        f = _fecha_iso(_tomar_ruta(d, "admision.fecha"))
        if f is not None and f not in fechas:
            fechas.append(f)
    if len(fechas) < 2:
        return []
    _en_letra = " y ".join(f0.fecha_en_letra(f) for f in fechas[:3])
    return [f"EN EL EXPEDIENTE HAY {len(fechas)} AUTOS DE PRESIDENCIA CON FECHA DE ADMISIÓN DISTINTA "
            f"({_en_letra}): se tomó el primero como el que admite. Si uno forma y registra el "
            "recurso (o lo radica, o pide el informe) y otro lo admite, pon la fecha del primero "
            "en «Auto de Presidencia que forma y registra» y la del que admite en la del auto de "
            "admisión («Trámite en este tribunal»), y vuelve a generar."]


def fundir_autos(de_admision=(), de_turno=()) -> dict:
    """Las lecturas (`leer_auto_tramite`) de los autos de admisión y de turno
    de un expediente de SISE, en una sola: cada campo, de la primera lectura
    que lo trae —la del auto de turno primero sólo para `turno` y `returno`—,
    con su fuente; los avisos y las fechas imposibles de todas, sin repetir.

    POR QUÉ POR SEPARADO Y NO PEGADOS. En el camino SISE la depuración ya
    separó los dos autos (`depurar_expediente`), y leídos juntos la fecha del
    primero se confundía con la del segundo: el turno saldría fechado el día
    de la admisión."""
    adm = [d for d in (de_admision or []) if isinstance(d, dict) and d]
    tur = [d for d in (de_turno or []) if isinstance(d, dict) and d]
    claves = []
    for d in adm + tur:
        for k in d:
            if k not in claves and k not in _LISTAS_DE_FICHA:
                claves.append(k)
    fuera, fuentes = {}, {}
    for k in claves:
        orden = (tur + adm) if k in _DEL_AUTO_DE_TURNO else (adm + tur)
        for d in orden:
            if _con_valor(d.get(k)):
                fuera[k] = d[k]
                for fk, fv in (d.get("fuentes") or {}).items():
                    if str(fk).split(".")[0] == k and fk not in fuentes:
                        fuentes[fk] = fv
                break
    if fuentes:
        fuera["fuentes"] = fuentes
    _av = _unir_avisos(*[d.get("avisos") for d in adm + tur], _aviso_dos_autos_de_presidencia(adm, fuera))
    if _av:
        fuera["avisos"] = _av
    _fi = _unir_avisos(*[d.get("fechas_imposibles") for d in adm + tur])
    if _fi:
        fuera["fechas_imposibles"] = _fi
    return fuera


async def _leer_acto_tramite(cliente, texto_acto: str, e, texto_conceptos: str = "") -> tuple:
    """(lo leído del acto para la ficha, avisos). Una llamada JSON anclada al
    papel (`ficha_tramite.leer_acto`), lanzada a la vez que las fases 1-3. Si
    falla, (None, [aviso]): la ficha se arma sin ella y lo que el acto debía
    dar —fecha, órgano, toca, expediente— sale en hueco, nunca de memoria.

    LA DEMANDA, SÓLO EN AMPARO DIRECTO. Ahí el escrito que sube el secretario
    ES la demanda, y de ella salen los derechos que se dicen violados —el
    resultando que el modelo inventaba u omitía porque nunca la vio— y la
    ordenadora tal como la señala el quejoso. En un recurso el escrito son
    los agravios: pasarlo como demanda pondría al lector a buscar la demanda
    de amparo en el papel equivocado."""
    _tipo = _tipo_de_encargo(e)
    try:
        import ficha_tramite as _ft
        _kw = {"demanda": texto_conceptos or ""} if (_tipo == "amparo_directo" and texto_conceptos) else {}
        d = await _ft.leer_acto(cliente, texto_acto or "", _tipo,
                                str(getattr(e, "numero", "") or ""), **_kw)
        return (d if isinstance(d, dict) else None), []
    except Exception as _ex:
        print(f"   ⚠️ FICHA DE TRÁMITE: el acto no se pudo leer ({type(_ex).__name__}: {str(_ex)[:120]})")
        return None, [
            f"LA RESOLUCIÓN RECLAMADA O RECURRIDA NO SE PUDO LEER PARA LOS RESULTANDOS "
            f"({type(_ex).__name__}): su fecha, el órgano que la dictó, el toca y el expediente "
            "salen en hueco. Complétalos en «Trámite en este tribunal» y vuelve a generar."]


# ═══════════════════════════════════════════════════════════════════════════
# QUIÉN RECURRE, LEÍDO DEL ESCRITO (3-oct-2026, E3 de la prueba de punta a punta)
# ═══════════════════════════════════════════════════════════════════════════
# AR 631/2025 con los papeles reales: el formulario no traía ni «quien
# promueve» ni «recurrente», y el proyecto salió con «QUEJOSA Y RECURRENTE:
# Unión de Trabajadores…», «en su carácter de parte quejosa» y la legitimación
# del artículo 6o. El escrito de revisión lo firmaba la adquirente del
# inmueble como «TERCERO INTERESADO». El escrito es el propio recurso: dice
# quién lo interpone y con qué carácter. `ficha_tramite.leer_escrito` lo lee
# (sin modelo primero, anclado al papel); aquí se fijan con ello el
# recurrente y su papel, con aviso, y SÓLO si el secretario no lo tecleó.

# Lo que la lectura del escrito puede tardar antes de seguir sin ella: es una
# ayuda, y un proveedor colgado no puede retener el adelanto entero (rev_6).
TOPE_LEER_ESCRITO = 75  # > ficha_tramite.TOPE_SEGUNDOS_MODELO (60): que no tire lo determinista ya leído

_CARACTER_DEL_ESCRITO = (
    ("ministerio_publico", r"ministerio\s+publico|fiscalia\s+general\s+de\s+la\s+republica"),
    ("autoridad", r"\bautoridad"),
    ("tercero", r"\btercer[oa]\b"),
    ("quejoso", r"\bquejos[oa]s?\b"),
)
_CARACTER_EN_PROSA = {
    "quejoso": "en su carácter de parte quejosa",
    "tercero": "en su carácter de parte tercera interesada",
    "autoridad": "en su carácter de autoridad",
    "ministerio_publico": "en su carácter de Ministerio Público",
}


def caracter_del_escrito(x) -> str:
    """«TERCERO INTERESADO», «autoridad responsable», «parte quejosa»… →
    «tercero» | «autoridad» | «quejoso» | «ministerio_publico» | «» (lo que no
    se reconoce no se supone)."""
    t = _plano_min(x).replace("_", " ")
    if not t:
        return ""
    for clave, rx in _CARACTER_DEL_ESCRITO:
        if re.search(rx, t):
            return clave
    return ""


async def _leer_escrito_tramite(cliente, texto_escrito: str, e) -> tuple:
    """(lo leído del escrito del recurso —quién recurre, con qué carácter, su
    representante y su figura—, avisos). `ficha_tramite.leer_escrito(cliente,
    texto, tipo)`, con tope de `TOPE_LEER_ESCRITO` segundos. Nunca lanza: si
    la pieza aún no existe o falla, (None, [aviso]) y el adelanto sigue como
    antes, leyendo a la quejosa de la sentencia recurrida."""
    _tipo = _tipo_de_encargo(e)
    try:
        import inspect as _insp
        import ficha_tramite as _ft
        _leer = getattr(_ft, "leer_escrito", None)
        if _leer is None:
            print("   ⚠️ FICHA DE TRÁMITE: ficha_tramite.leer_escrito no existe; "
                  "quién recurre sale de la sentencia recurrida")
            return None, []
        d = _leer(cliente, texto_escrito or "", _tipo)
        if _insp.isawaitable(d):
            d = await asyncio.wait_for(d, timeout=TOPE_LEER_ESCRITO)
        # LA LECTURA QUE CORRIÓ Y NO DIJO NADA es un dict vacío, no la pieza
        # que falta: `fijar_quien_recurre` avisa que el escrito no lo dice.
        return (d if isinstance(d, dict) else {}), []
    except asyncio.TimeoutError:
        print(f"   ⚠️ FICHA DE TRÁMITE: la lectura del escrito no contestó en {TOPE_LEER_ESCRITO} s")
        return None, [
            "NO TECLEASTE QUIÉN RECURRE Y LA LECTURA DEL ESCRITO NO CONTESTÓ A TIEMPO: la "
            "carátula y la legitimación salen con lo que dice la sentencia recurrida. Escribe "
            "quién recurre en la ficha del asunto y vuelve a generar."]
    except Exception as _ex:
        print(f"   ⚠️ FICHA DE TRÁMITE: el escrito no se pudo leer ({type(_ex).__name__}: {str(_ex)[:120]})")
        return None, [
            f"NO TECLEASTE QUIÉN RECURRE Y DEL ESCRITO NO SE PUDO LEER ({type(_ex).__name__}): "
            "la carátula y la legitimación salen con lo que dice la sentencia recurrida. "
            "Escribe quién recurre en la ficha del asunto y vuelve a generar."]


def fijar_quien_recurre(e, escrito, partes=None, texto_acto: str = "", avisos=None) -> bool:
    """Fija en el encargo quién recurre y con qué carácter, con lo leído del
    escrito del recurso (`_leer_escrito_tramite`); True si lo fijó. Pone
    PRIMERO el aviso «SE LEYÓ DEL ESCRITO QUIÉN RECURRE…». Nunca lanza.

    - Recurre la quejosa: `quejoso` = el nombre del escrito, `recurrente` vacío,
      `papel_recurrente` «quejoso».
    - Recurre otra parte (tercero, autoridad, Ministerio Público):
      `recurrente` = el nombre del escrito y `papel_recurrente` su carácter (el
      Ministerio Público no: ver `PAPELES_FIJOS`); la quejosa, la que nombra el
      punto resolutivo del juzgado o, si no, la de la ficha de partes, nunca
      la misma persona que recurre.
    - El escrito llama tercero o autoridad a la misma parte a la que el juzgado
      resolvió en su punto resolutivo: no se fija nada y se avisa (una de las
      dos lecturas está mal, y no se elige a ciegas)."""
    avisos = avisos if avisos is not None else []
    try:
        if not isinstance(escrito, dict):
            return False
        quien =" ".join(str(escrito.get("promovente") or escrito.get("recurrente")
                             or escrito.get("quien") or "").split()).strip(" ,;")
        papel = caracter_del_escrito(escrito.get("caracter") or escrito.get("papel"))
        if not quien or not papel:
            avisos.insert(0,
                "NO TECLEASTE QUIÉN RECURRE Y EL ESCRITO NO LO DICE CON CLARIDAD: la carátula y "
                "la legitimación salen con lo que dice la sentencia recurrida. Escribe quién "
                "recurre en la ficha del asunto y vuelve a generar.")
            return False
        try:
            import fase_rama as _fr_e
            _res = _resolutivo_del_a_quo(None, texto_acto or "")
            _q_res = str(_fr_e.quejoso_del_resolutivo(_res) or "").strip() if _res else ""
        except Exception:
            _q_res = ""
        if papel != "quejoso" and _q_res and _parte_parecida(_q_res, quien):
            avisos.insert(0,
                f"EL ESCRITO DEL RECURSO NOMBRA A «{quien[:90]}» "
                f"{_CARACTER_EN_PROSA[papel].upper()}, PERO ES LA PARTE A LA QUE SE REFIERE EL "
                f"PUNTO RESOLUTIVO DEL JUZGADO («{_q_res[:90]}»): no se fijó quién recurre. "
                "Escríbelo en la ficha del asunto y vuelve a generar.")
            return False
        _q_de_donde = ""
        if papel == "quejoso":
            e.quejoso = quien
            e.recurrente = ""
            e.papel_recurrente = "quejoso"
        else:
            e.recurrente = quien
            e.papel_recurrente = papel if papel in PAPELES_FIJOS else ""
            _q_partes = str(getattr(partes, "quejoso", "") or "").strip()
            if _q_res and not _parte_parecida(_q_res, quien):
                e.quejoso, _q_de_donde = _q_res, "del punto resolutivo de la sentencia recurrida"
            elif _q_partes and not _parte_parecida(_q_partes, quien):
                e.quejoso, _q_de_donde = _q_partes, "de los documentos"
        _rep = " ".join(str(escrito.get("representante") or "").split())
        _fig = " ".join(str(escrito.get("figura_representante") or escrito.get("figura") or "").split())
        _por = (f", por conducto de su {_fig} {_rep}" if (_rep and _fig)
                else f", por conducto de {_rep}" if _rep else "")
        _aviso = (f"SE LEYÓ DEL ESCRITO QUIÉN RECURRE: «{quien[:120]}», {_CARACTER_EN_PROSA[papel]}"
                  f"{_por}. No lo tecleaste en la ficha del asunto: compruébalo en la carátula y en "
                  "la legitimación; si recurre otra parte, escríbela como recurrente y vuelve a "
                  "generar.")
        if _q_de_donde:
            _aviso += f" La parte quejosa, «{e.quejoso[:90]}», se leyó {_q_de_donde}."
        if papel == "ministerio_publico":
            _aviso += (" La legitimación del Ministerio Público (artículo 5o., fracción IV, de la "
                       "Ley de Amparo) sale de la ficha de trámite: revísala.")
        avisos.insert(0, _aviso)
        print(f"   🧾 QUIÉN RECURRE, DEL ESCRITO: «{quien[:70]}» ({papel})")
        return True
    except Exception as _ex:
        print(f"   ⚠️ quién recurre sin fijar desde el escrito: {type(_ex).__name__}: {str(_ex)[:120]}")
        return False


def _ficha_minima(e) -> dict:
    """La ficha cuando `ficha_tramite.armar` no se pudo ejecutar: el tipo, el
    número y la sede, nada más. El compositor la llena de HUECOS con su aviso,
    que es lo honesto; inventar el resto es justo lo que esta pieza quita."""
    _sede = {"tribunal": str(getattr(e, "tribunal", "") or ""),
             "ciudad": str(getattr(e, "ciudad", "") or ""), "circuito": "", "cdmx": False}
    _formato = 1
    try:
        import ficha_tramite as _ft
        _formato = getattr(_ft, "FORMATO", 1)
        _s = _ft.sede_de(_sede["tribunal"], _sede["ciudad"])
        if isinstance(_s, dict):
            _sede = _s
    except Exception:
        pass
    # LOS RELACIONADOS QUE MARCÓ EL SECRETARIO NO SE PIERDEN (C6, 3-oct-2026):
    # son un dato suyo, no una lectura que pudo fallar. Sin esto, si `armar`
    # falla, el rubro, el V I S T O y el considerando de conexidad se caían.
    _rel = (getattr(e, "tramite_declarado", None) or {}).get("relacionados")
    _rel = [r for r in _rel if isinstance(r, dict)] if isinstance(_rel, list) else []
    return {"formato": _formato, "tipo": _tipo_de_encargo(e),
            "numero": str(getattr(e, "numero", "") or ""),
            "materia": str(getattr(e, "materia", "") or ""),
            "sede": _sede, "relacionados": _rel, "avisos": [],
            "fuentes": {"relacionados": "secretario"} if _rel else {},
            "fechas_imposibles": []}


def armar_tramite(e, acto=None, fases=None, partes=None, texto_acto: str = "",
                  avisos_previos=(), escrito=None) -> dict:
    """La ficha de trámite del asunto (§1 de la especificación), con sus
    avisos dentro. Funde por precedencia —secretario > auto > acto > ficha
    procesal— en `ficha_tramite.armar`; aquí sólo se le entregan las cuatro
    fuentes y se garantiza que lo que avisaron la lectura del auto, la del
    formulario y la del acto llegue al secretario aunque `armar` no lo copie.
    Nunca lanza: si `armar` falla, la ficha mínima (todo en hueco) y el aviso
    que lo dice.

    `escrito` (3-oct-2026, E3): lo leído del escrito del recurso
    (`_leer_escrito_tramite`). Va a `armar(…, escrito=)` —fuente «escrito»,
    detrás del auto y delante de la ficha procesal— si `armar` lo recibe; si
    no, sólo sus avisos (el recurrente y su papel ya quedaron en el encargo)."""
    _fp = None
    try:
        import ficha_procesal as _fp_t
        _fp = _fp_t.armar(e, fases, partes, acto=texto_acto or "")
    except Exception as _ex:
        print(f"   ⚠️ FICHA DE TRÁMITE: sin ficha procesal ({type(_ex).__name__})")
        _fp = None

    def _entrada(d):
        # Un dict que sólo trae avisos (la lectura que falló) no es una fuente.
        if not isinstance(d, dict) or not any(k != "avisos" for k in d):
            return None
        return d

    _auto = getattr(e, "tramite_auto", None) or {}
    _decl = getattr(e, "tramite_declarado", None) or {}
    _fallo = []
    try:
        import ficha_tramite as _ft
        _kw_esc = {}
        if _entrada(escrito) is not None:
            try:
                import inspect as _insp_a
                if "escrito" in _insp_a.signature(_ft.armar).parameters:
                    _kw_esc = {"escrito": escrito}
            except (TypeError, ValueError):
                _kw_esc = {}
        ficha = _ft.armar(e, auto=_entrada(_auto), acto=_entrada(acto),
                          ficha_procesal=_fp or None, partes=partes,
                          declarado=_entrada(_decl), **_kw_esc)
        if not isinstance(ficha, dict):
            raise TypeError(f"armar devolvió {type(ficha).__name__}")
    except Exception as _ex:
        print(f"   ⚠️ FICHA DE TRÁMITE: no se pudo armar ({type(_ex).__name__}: {str(_ex)[:160]})")
        ficha = _ficha_minima(e)
        _fallo = [f"LA FICHA DE TRÁMITE NO SE PUDO ARMAR ({type(_ex).__name__}): el V I S T O y "
                  "los resultandos salen con HUECOS en cada dato del trámite. Complétalos en "
                  "«Trámite en este tribunal» y vuelve a generar, o escríbelos en el documento."]
    ficha["avisos"] = _unir_avisos(_fallo, ficha.get("avisos"),
                                   (_auto or {}).get("avisos"), (_decl or {}).get("avisos"),
                                   (acto or {}).get("avisos") if isinstance(acto, dict) else None,
                                   (escrito or {}).get("avisos") if isinstance(escrito, dict) else None,
                                   avisos_previos)
    _serial = tramite_serializable(ficha)
    if _serial is None:
        print("   ⚠️ FICHA DE TRÁMITE: no es serializable; viaja tal cual en memoria")
        return ficha
    print(f"   🧾 FICHA DE TRÁMITE: {_tipo_de_encargo(e)} · "
          f"{len(_serial.get('fuentes') or {})} campos con fuente · "
          f"{len(_serial.get('avisos') or [])} avisos")
    return _serial


def _compuesto_por_tipo(e, datos: dict) -> tuple:
    """(lo que devuelve `resultandos_por_tipo.componer`, «») cuando rige la
    bandera y el encargo trae ficha; (None, «») si no toca; (None, por qué)
    si el compositor falló."""
    ficha = getattr(e, "tramite", None)
    if not isinstance(ficha, dict) or not rige_procedencia_por_tipo():
        return None, ""
    try:
        import resultandos_por_tipo as _rpt
        # LOS DATOS DE SIEMPRE, SIN LO QUE ÉL MISMO PRODUCE: `procesal` de una
        # composición anterior no puede volver a entrar como si fuera dato.
        _d = {k: v for k, v in (datos or {}).items() if k not in ("tramite", "procesal")}
        res = _rpt.componer(_tipo_de_encargo(e), ficha, _d)
        if not isinstance(res, dict) or not isinstance(res.get("resultandos"), list):
            raise TypeError("componer no devolvió {visto, resultandos, avisos, datos_extra}")
        # EL COMPOSITOR NO LANZA: cuando falla por dentro, o no conoce el
        # tipo, devuelve el V I S T O y los resultandos VACÍOS con el aviso que
        # lo explica, para que aquí se caiga al camino viejo. Un documento sin
        # resultandos no es una composición.
        if not str(res.get("visto") or "").strip() and not res.get("resultandos"):
            _por_que = " ".join(str(a) for a in (res.get("avisos") or []))[:300]
            print(f"   ⚠️ PROCEDENCIA POR TIPO: el compositor no compuso nada ({_por_que[:160]})")
            return None, _por_que or "no compuso ni el V I S T O ni los resultandos"
        return res, ""
    except Exception as _ex:
        print(f"   ⚠️ PROCEDENCIA POR TIPO: el compositor falló ({type(_ex).__name__}: {str(_ex)[:160]})")
        return None, f"{type(_ex).__name__}: {str(_ex)[:160]}"


def _estructura_de_lo_compuesto(res: dict, ficha: dict):
    """La `Estructura` de siempre con el V I S T O y los resultandos del
    compositor. La apertura va vacía (la compone `documento_generado`); la
    competencia, la existencia y la procedencia también (las ponen el banco y
    el código con los datos de la ficha, `datos["procesal"]`)."""
    import documento_generado as _dg_e
    return _dg_e.Estructura(
        apertura="", visto=str(res.get("visto") or ""),
        resultandos=[{"titulo": str(x.get("titulo") or ""), "texto": str(x.get("texto") or "")}
                     for x in (res.get("resultandos") or []) if isinstance(x, dict)],
        competencia="", existencia="", procedencia="",
        # LO QUE EL COMPOSITOR CALLÓ POR REPETIDO NO VUELVE POR LA FICHA (3-oct-2026,
        # integración, E11): `avisos_repetidos` son los de `validar` que el
        # compositor ya dijo con su propia clave.
        avisos=_unir_avisos(res.get("avisos"),
                            [a for a in ((ficha or {}).get("avisos") or [])
                             if a not in (res.get("avisos_repetidos") or [])]))


def _datos_estructura(e: Encargo, antecedentes: str = "", acto: str = "",
                     partes=None, fases=None) -> dict:
    """Lo que la estructura necesita.

    Decía «TODO sale del encargo» y por eso se lanzaba a ciegas. Del encargo
    salen la competencia y la existencia; los RESULTANDOS necesitan el acto
    —para individualizar la sentencia recurrida con su fecha y su expediente—
    y la ficha de partes —para nombrar al tercero interesado en vez de escribir
    «la persona a quien resulta tal carácter»—.

    `fases` (28-sep-2026): su punto resolutivo del juzgado decide quién es el
    quejoso cuando el formulario guardó a la recurrente (AR 631/2025).
    """
    import fase0_oportunidad as _f0
    # LOS PAPELES SALEN DE LA FICHA PROCESAL (SPEC_E2, 28-sep-2026): quejosa,
    # recurrente y su carácter, tercero y órgano recurrido se fijan UNA vez
    # —`ficha_procesal.armar`, que reutiliza `_quejoso_del_amparo`,
    # `_recurrente_de`, `papel_del_recurrente` y `_organo_recurrido`— y de ahí
    # los leen la carátula, la competencia, la legitimación, los resolutivos y
    # la síntesis. En el 631 cada una los adivinaba por su lado. Si la ficha
    # no se puede armar, las mismas deducciones sueltas de antes.
    _ficha = {}
    try:
        import ficha_procesal as _fp_d
        _ficha = _fp_d.armar(e, fases, partes, acto=acto)
    except Exception as _ex_f:
        print(f"   ⚠️ FICHA PROCESAL: no se pudo armar ({type(_ex_f).__name__}); "
              f"los papeles salen de las deducciones sueltas")
        _ficha = {}
    if _ficha:
        _q_ficha = (_ficha.get("quejosa") or {}).get("nombre") or ""
        _rc_ficha = _ficha.get("recurrente") or {}
        _terceros = "; ".join(t.get("nombre", "") for t in (_ficha.get("terceros") or [])
                              if t.get("nombre"))
        _recurrente_d = _rc_ficha.get("aparte") or ""
        _papel_d = _rc_ficha.get("papel") or ""
        _organo_d = (_ficha.get("organo_recurrido") or {}).get("nombre") or ""
        import ficha_procesal as _fp_b
        _ficha_bloque = _fp_b.bloque(_ficha)
    else:
        _terceros = ""
        if partes is not None:
            _terceros = str(getattr(partes, "tercero_interesado", "") or "")
        _res_aq = ""
        if getattr(e, "es_recurso", False):
            try:
                _res_aq = _resolutivo_del_a_quo(fases, acto)
            except Exception:
                _res_aq = ""
            # LA RECURRENTE QUE NO ES LA QUEJOSA NI UNA AUTORIDAD ES LA TERCERA
            # INTERESADA: si la ficha no la trae, se nombra así en los resultandos.
            if not _terceros.strip() and papel_del_recurrente(e, partes, _res_aq) == "tercero":
                _terceros = _recurrente_de(e, partes, _res_aq)
        _q_ficha = _quejoso_del_amparo(e, partes, _res_aq)
        _recurrente_d = _recurrente_de(e, partes, _res_aq)
        _papel_d = papel_del_recurrente(e, partes, _res_aq)
        _organo_d = _organo_recurrido(e, partes, acto)
        _ficha_bloque = ""
    datos = {
        # LA CABEZA DEL ACTO, que es donde se identifica: fecha, órgano,
        # expediente y toca. No el documento entero —eso ya lo leen las fases—
        # sino lo justo para individualizarlo sin adivinar.
        "acto": (acto or "")[:200000],
        "tercero": _terceros,
        # EL NÚMERO DEL PROPIO ASUNTO. Sin él, `fase_origen.numero_de` no puede
        # descartarlo y devuelve el toca de ESTE recurso como si fuera el
        # expediente de origen: el dict tenía `encabezado`, que es otra cosa.
        "numero": e.numero,
        "tribunal": e.tribunal or "",
        "ciudad": e.ciudad or "",
        "encabezado": e.encabezado,
        # LA PARTE Y QUIEN LA REPRESENTA, SEPARADAS EN LA FUENTE. «Alondra
        # Zúñiga Gutiérrez, representante legal de GDE Trading Company, S.A.
        # de C.V.» llegaba entero como `quejoso` y así salía en la carátula,
        # en la legitimación y en el resolutivo (v5 del ADC 93/2026: «ampara
        # y protege a Alondra…»). La parte es la representada; la persona
        # física sólo tiene la personería (arts. 6 y 11 de la Ley de Amparo).
        "quejoso": _pv.separar(_q_ficha)["parte"] or _q_ficha,
        # Quien recurrió, cuando no es el quejoso. Vacío = es el mismo.
        "recurrente": _recurrente_d,
        # Y EN QUÉ CARÁCTER (28-sep-2026): la legitimación de la tercera
        # interesada no es la del quejoso (AR 631/2025).
        "papel_recurrente": _papel_d,
        # EL ÓRGANO RECURRIDO ES EL JUZGADO, no la responsable del amparo. El
        # formulario guarda en `responsable` a la autoridad del acto reclamado
        # —en el 711/2025, la UIF— y la carátula de la revisión rotulaba ese
        # dato como «ÓRGANO RECURRIDO». Lo recurrido es la sentencia del juez
        # de distrito; la ficha de partes lo lee de ella. Sólo para la carátula:
        # `responsable` sigue siendo la del acto en los otros doce sitios.
        "organo_recurrido": _organo_d,
        # LA FICHA ENTERA Y SU BLOQUE DE DATOS: el bloque va al prompt de la
        # estructura (carátula, competencia, legitimación) y al de la síntesis.
        "ficha_procesal": _ficha,
        "ficha_bloque": _ficha_bloque,
        "representante": _pv.separar(e.quejoso)["representante"],
        "figura_representante": _pv.separar(e.quejoso)["figura"],
        "quejoso_moral": _pv.separar(e.quejoso)["moral"],
        "responsable": e.responsable or "",
        "magistrado": e.magistrado,
        "secretario": e.secretario,
        "presentacion": _f0.fecha_en_letra(e.presentacion),
        # EL TIPO DE ASUNTO, que faltaba y por eso el prompt de estructura
        # escribía siempre «una sentencia de amparo directo» y pedía identificar
        # el acto por «sala, toca y expediente» aunque fuera una queja.
        "tipo_asunto": getattr(e, "tipo_asunto", "amparo_directo"),
        "es_recurso": e.es_recurso,
        # Las dos fechas de la sesión, en letra como todo lo demás del cuerpo.
        # Vacías si el secretario no las declaró: entonces siguen en hueco.
        # SE COMPRUEBA QUE LA FECHA SE PUDO LEER, no que el campo venga lleno.
        # Escrito como estaba, un «15/03/2026» —que es como se teclea una fecha
        # en España y en México— pasaba el `if`, `_fecha_iso` devolvía None y
        # `fecha_en_letra(None)` tumbaba el adelanto con un 500 DESPUÉS de
        # treinta y cinco segundos de trabajo ya pagado. Lo introduje yo hoy.
        "fecha_lista": _f0.fecha_en_letra(_fecha_iso(getattr(e, "fecha_lista", "")))
                       if _fecha_iso(getattr(e, "fecha_lista", "")) else "",
        "fecha_sesion": _f0.fecha_en_letra(_fecha_iso(getattr(e, "fecha_sesion", "")))
                        if _fecha_iso(getattr(e, "fecha_sesion", "")) else "",
        "antecedentes": antecedentes,
        # QUÉ RESOLVIÓ EL ÓRGANO RECURRIDO, tal como lo resumió el motor al
        # preparar la propuesta. Decide el verbo del resolutivo cuando los
        # antecedentes no llegan a decirlo —que es lo que pasó en la revisión
        # 410/2026: narraban el juicio de nulidad y nunca decían en qué paró el
        # amparo, así que salió «Se ********* la sentencia recurrida» tres
        # párrafos después de que el estudio dijera «se confirma»—.
        "resolvio_declarado": getattr(e, "resolvio_declarado", "") or "",
    }
    # LA FICHA DE TRÁMITE Y LO QUE DE ELLA SALE PARA LOS CONSIDERANDOS
    # (3-oct-2026, bandera `procedencia_por_tipo`). `procesal` son los
    # `datos_extra` del compositor —expediente, toca, fecha y órgano del acto,
    # inciso del 97, juicio de amparo, sala…— y en `documento_generado.componer`
    # MANDAN sobre lo que hoy se lee con regex de la prosa de los resultandos.
    # Sin bandera o sin ficha, las dos claves no existen y todo sigue igual.
    if isinstance(getattr(e, "tramite", None), dict) and rige_procedencia_por_tipo():
        datos["tramite"] = e.tramite
        # EL GÉNERO DE LA PERSONA FÍSICA, SÓLO SI EL PAPEL LO DICE (3-oct-2026):
        # «la actora Ana…», «el quejoso Juan…». Lo usan el rótulo de la carátula
        # y la verja; del nombre de pila no se infiere nada.
        try:
            import tipos_asunto as _ta_g
            _papel_g = " ".join(str(x or "") for x in (acto, antecedentes))
            for _k, _dk in (("quejoso", "genero_quejoso"), ("recurrente", "genero_recurrente")):
                _g = _ta_g.genero_en_el_papel(str(datos.get(_k) or ""), _papel_g)
                if _g and not datos.get(_dk):
                    datos[_dk] = _g
        except Exception:
            pass
        _res_pt, _ = _compuesto_por_tipo(e, datos)
        if _res_pt is not None:
            datos["procesal"] = dict(_res_pt.get("datos_extra") or {})
    return datos


def _fecha_iso(x):
    """La fecha ISO del formulario, o None si no se puede leer.

    Se devuelve None —y el hueco se queda— en vez de una fecha aproximada: una
    fecha de sesión equivocada en el resultando es peor que un asterisco, que
    al menos se ve.
    """
    import datetime as _dt
    import re as _re
    t = str(x or "").strip()
    if not t:
        return None
    try:
        return _dt.date.fromisoformat(t[:10])
    except Exception:
        pass
    # «15/03/2026» y «15-03-2026», que es como se teclea una fecha aquí.
    m = _re.match(r"^(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})$", t)
    if m:
        try:
            return _dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None
    return None


async def _componer_generado(cliente, e: Encargo, relleno, computo,
                             ruta_salida: str, estructura_previa=None,
                             marco_escrito: str = "", acto: str = "",
                             partes=None, criterios=None, fases=None,
                             reasuncion=None):
    """El documento escrito entero. Devuelve (ruta, avisos, estructura)."""
    import documento_generado as dg
    import fase0_oportunidad as _f0

    # LA SEGUNDA LLAMADA TAMBIÉN NECESITA EL ACTO Y LAS PARTES. Se arregló la
    # de arriba y ésta se quedó igual: cuando `estructura_previa` es None
    # —porque se recompone el documento sin haber pasado por el adelanto— la
    # estructura volvía a escribirse a ciegas, y con ella la perífrasis.
    datos = _datos_estructura(e, "\n".join(relleno.antecedentes or []),
                              acto=acto, partes=partes, fases=fases)
    # LO LEÍDO DEL PAPEL VIAJA HASTA EL RESOLUTIVO. Se calculó en el adelanto
    # sobre el PDF de la recurrida y va en el estado de la sesión, así que
    # existe también cuando resuelve el otro worker.
    # DEL OBJETO DE FASES, NO DEL RELLENO. `relleno` es el ensamblado que
    # alimenta la plantilla —encabezado, resúmenes, estudio— y no lleva estos
    # dos campos: leerlos de ahí devolvía cadena vacía en silencio, y el
    # resolutivo seguía saliendo de la prosa del modelo. Se vio en la
    # comprobación de la 650/2025: el modelo dijo «negó el amparo» —falso, lo
    # concedió— y su versión ganó igual, que es exactamente el fallo que esto
    # venía a cerrar. Un getattr con valor por omisión no avisa de nada.
    datos["resolvio_a_quo"] = getattr(fases, "resolvio_a_quo", "") or ""
    datos["resolutivo_recurrida"] = getattr(fases, "resolutivo_recurrida", "") or ""
    # LA MISMA LECTURA QUE LA TARJETA, Y CON EL LECTOR DE HOY (4-oct-2026, AR
    # 380/2025). La sesión guardó «sobresee» —el recuento sobre la firma
    # electrónica— y ningún punto reproducible; la propuesta y el proyecto
    # confirmaban un sobreseimiento que el juzgado nunca decretó. Si la sesión
    # trae el papel, se vuelve a leer: lo que hizo el juzgado con
    # `que_hizo_el_juzgado` (el orden de fuentes de la tarjeta) y su punto de
    # fondo con `resolutivo_recurrida`.
    try:
        import fase_rama as _fr_d
        if _tipo_de_encargo(e) == "amparo_revision":
            _que_d = _fr_d.que_hizo_el_juzgado(fases, str(getattr(e, "resolvio_declarado", "") or ""))
            if _que_d:
                datos["resolvio_a_quo"] = _que_d
            datos["resolutivo_recurrida"] = _fr_d.resolutivo_de_fases(fases)
    except Exception as _ex_d:
        print(f"   ⚠️ no se pudo releer el desenlace del a quo: {type(_ex_d).__name__}")
    datos["expediente_origen"] = getattr(fases, "expediente_origen", "") or ""
    datos["fecha_origen"] = getattr(fases, "fecha_origen", "") or ""
    # SI SE REASUME JURISDICCIÓN (art. 93, frs. I, V y VI; 28-sep-2026): quién
    # recurre, si los conceptos constaron y si la recurrida también sobreseyó.
    # Sin ella, el compositor decide por la rama y por el estudio.
    if isinstance(reasuncion, dict):
        datos["reasuncion"] = {k: v for k, v in reasuncion.items() if k != "conceptos"}
    # ═══ CON LA FICHA DE TRÁMITE, LA ESTRUCTURA SE COMPONE (3-oct-2026) ════
    # Bandera `procedencia_por_tipo` y encargo con ficha: el V I S T O y los
    # resultandos los escribe `resultandos_por_tipo.componer`, sin modelo, con
    # los datos ya completos de ESTA composición (el desenlace del a quo y la
    # reasunción se acaban de poner). Se recompone siempre, también en el
    # resolver y en el worker que no leyó el acto: la ficha viaja en la sesión
    # y componer cuesta milisegundos; reutilizar una estructura en memoria
    # sería volver a depender de qué worker atiende. Si el compositor falla,
    # el camino de siempre con un aviso que lo dice: un documento con los
    # resultandos del modelo y la advertencia delante es mejor que ninguno.
    est = None
    _res_pt, _fallo_pt = _compuesto_por_tipo(e, datos)
    if _res_pt is not None:
        datos["tramite"] = e.tramite
        datos["procesal"] = dict(_res_pt.get("datos_extra") or {})
        est = _estructura_de_lo_compuesto(_res_pt, e.tramite)
        print(f"   🧾 PROCEDENCIA POR TIPO: V I S T O y {len(est.resultandos)} resultandos "
              f"compuestos sin modelo · {len(est.avisos)} avisos")
    elif _fallo_pt:
        datos.pop("procesal", None)
    # LA ESTRUCTURA SE ESCRIBE UNA VEZ. El resolver recompone el documento
    # entero, y volver a pedirla al modelo son treinta segundos por nada: no
    # depende del estudio ni del criterio, sólo del asunto.
    if est is None:
        est = estructura_previa or await dg.redactar_estructura(cliente, datos)
    if _fallo_pt:
        est.avisos = _unir_avisos(
            [f"LOS RESULTANDOS NO SE PUDIERON COMPONER CON LA FICHA DE TRÁMITE ({_fallo_pt}): "
             "los escribió el modelo, como antes de este cambio. Coteja cada fecha, número y "
             "nombre de los resultandos con el auto de admisión y con el acto antes de firmar."],
            (getattr(e, "tramite", None) or {}).get("avisos"), est.avisos)

    # LA SÍNTESIS DE LA PORTADA. Se pide con el estudio YA REDACTADO, no con
    # los datos del asunto: una síntesis escrita antes del estudio resumiría lo
    # que se pensaba resolver, no lo que se resolvió, y es justo el desajuste
    # que hace inservible un resumen. Si falla, el documento sale sin ella.
    _sint = {}
    try:
        import fase_sintesis as _fs
        # LOS PAPELES YA RECONCILIADOS, no lo tecleado (28-sep-2026): en el AR
        # 631/2025 se le dijo «parte promovente: Impulsora» —la recurrente— y la
        # síntesis contó que «la parte quejosa adquirió el inmueble» y que «la
        # autoridad responsable concedió el amparo».
        _sint = await _fs.sintetizar(
            cliente,
            tipo_asunto=(getattr(e, "tipo_asunto", "") or ""),
            expediente=(getattr(e, "numero", "") or ""),
            quejoso=str(datos.get("quejoso") or getattr(e, "quejoso", "") or ""),
            sentido=str(getattr(relleno, "calificaciones", "") or ""),
            estudio="\n\n".join(relleno.estudio or []),
            recurrente=str(datos.get("recurrente") or ""),
            papel_recurrente=str(datos.get("papel_recurrente") or ""),
            organo=str(datos.get("organo_recurrido") or ""),
            # LA FICHA PROCESAL COMO DATOS (SPEC_E2): quién es quién y qué se
            # revisa, con su fuente, en lugar de dos renglones sueltos.
            ficha=str(datos.get("ficha_bloque") or ""))
    except Exception:
        _sint = {}

    ruta = dg.componer(
        datos, est, computo, _f0.fecha_en_letra, ruta_salida,
        antecedentes=relleno.antecedentes,
        resumen_acto=relleno.resumen_acto,
        resumen_conceptos=relleno.resumen_conceptos,
        problemas=relleno.problemas,
        estudio=relleno.estudio,
        calificaciones=relleno.calificaciones,
        tesis=relleno.tesis,
        marco_escrito=marco_escrito,
        # El tipo decide el esqueleto: los recursos no llevan «Existencia del
        # acto reclamado» y la queja hace el cómputo en prosa.
        tipo_asunto=(getattr(e, "tipo_asunto", "") or
                     ("amparo_revision" if e.es_recurso else "amparo_directo")),
        normas=getattr(relleno, "normas", None),
        # LAS PREGUNTAS, para que el compositor las escriba. El pipeline ya
        # las tenía y no llegaban al documento; se pasan aquí porque componer
        # es lo único que garantiza que salgan.
        criterios=criterios,
        sintesis=_sint)
    avisos = list(est.avisos)
    if not e.tribunal:
        avisos.append(
            "No se indicó el TRIBUNAL que resuelve: la competencia y la "
            "fórmula de apertura salen incompletas. Es el dato que hace que "
            "esto sirva fuera de un solo circuito.")
    return ruta, avisos, est


def resumen_legible(r: Resultado) -> str:
    """Lo que se le enseña al secretario cuando termina el proceso."""
    lineas = [
        f"Adelanto generado: {r.ruta.split('/')[-1]}",
        f"Oportunidad: {'en tiempo' if r.computo.oportuna else 'EXTEMPORÁNEA'} "
        f"· plazo del {r.computo.inicio} al {r.computo.vencimiento} "
        f"({r.computo.plazo} días hábiles)",
        f"Antecedentes: {len(r.fases.antecedentes.split())} palabras "
        f"en {len(r.fases.parrafos_antecedentes())} párrafos",
        f"Resumen del acto: {len(r.fases.resumen_acto.split())} palabras",
        f"Resumen de conceptos: {len(r.fases.resumen_conceptos.split())} palabras",
        f"Problemas jurídicos: {len(r.fases.problemas)}"
        + (" (+ el global)" if r.fases.problema_global else ""),
    ]
    if r.avisos:
        lineas.append("\nAVISOS:")
        lineas += [f"  · {a}" for a in r.avisos]
    if r.huecos:
        lineas.append("\nPENDIENTE DE TU CRITERIO:")
        lineas += [f"  · {h[:100]}" for h in r.huecos]
    return "\n".join(lineas)
