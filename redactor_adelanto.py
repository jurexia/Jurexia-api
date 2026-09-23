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
    return any(k in n for k in (
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
    c = f0.computar(e.notificacion, e.presentacion, e.regla_surtimiento,
                    _plazo_computo, e.responsable,
                    getattr(e, "dias_inhabiles_extra", None),
                    # EL TIPO DECIDE SI EL DESCUENTO DE LA RESPONSABLE APLICA:
                    # sólo donde el escrito se presenta ante ella.
                    getattr(e, "tipo_asunto", "") or "amparo_directo",
                    getattr(e, "inhabiles_responsable", "") or None,
                    surtio_manual=_surtio_manual)
    avisos.extend(c.avisos)
    if c.oportuna is False:
        avisos.append("EL CÓMPUTO DA EXTEMPORÁNEA. Compruébalo antes de seguir: "
                      "si es correcto, el asunto no se resuelve en el fondo.")

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
    async def _partes_y_estructura():
        _p = await fpartes.fichar(cliente, texto_acto, texto_conceptos,
                                  e.tipo_asunto)
        _est = None
        if (e.modo or "").lower() == "generado":
            import documento_generado as _dg
            _est = await _dg.redactar_estructura(
                cliente, _datos_estructura(e, acto=texto_acto, partes=_p))
        return _p, _est

    with cronometrar("fases1-3+partes+estructura"):
        f, (partes, estructura_previa) = await asyncio.gather(
            f123.correr(cliente, texto_acto, texto_conceptos, e.es_recurso,
                        e.tipo_asunto,
                        quejoso=getattr(e, "quejoso", "") or "",
                        responsable=getattr(e, "responsable", "") or ""),
            _partes_y_estructura())
    avisos.extend(f.avisos)
    avisos.extend(partes.avisos)

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
    if getattr(e, "es_recurso", False):
        _q_leido = str(getattr(partes, "quejoso", "") or "").strip()
        _q_tecleado = str(getattr(e, "quejoso", "") or "").strip()
        if (_q_leido and _q_tecleado and not _mismo_nombre(_q_leido, _q_tecleado)
                and _parece_autoridad(_q_tecleado)
                and not str(getattr(e, "recurrente", "") or "").strip()):
            e.recurrente = _q_tecleado
            e.quejoso = _q_leido
            avisos.insert(0,
                f"SE SEPARARON LOS PAPELES: quien pidió el amparo es «{_q_leido}» "
                f"(leído de la sentencia recurrida) y quien recurre es la autoridad "
                f"«{_q_tecleado[:90]}». El amparo se concede o se niega a la primera; "
                f"el recurso se califica a la segunda. Compruébalo en la carátula.")
    if not str(getattr(e, "quejoso", "") or "").strip():
        _leido = str(getattr(partes, "quejoso", "") or "").strip()
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
    if e.es_recurso:
        try:
            import fase_rama as _fr_a
            f.resolvio_a_quo = _fr_a.resolvio_a_quo(texto_acto or "")
            f.resolutivo_recurrida = _fr_a.resolutivo_recurrida(texto_acto or "")
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
                    problemas.append(
                        f"¿{str(_x.get('motivo') or _pre).capitalize()}: "
                        f"{_x['explicacion']}?")
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



def _litis_y_material(r, material, avisos: list) -> list:
    """La litis del asunto, y el material ya sin ley local ajena a ella.

    Lo que no puede citarse no se le enseña al modelo: la guarda del final
    corrige lo que se escape, pero la mejor cita mala es la que nunca se
    escribe. Devuelve la litis para que `_terminar` la reutilice.
    """
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
    _rama, _vp = "", False
    try:
        import tipos_asunto as _ta_r, fase_rama as _fr_r
        if _ta_r.normalizar(e.tipo_asunto) == "amparo_revision":
            _que = _fr_r.resolvio_a_quo(
                "", "\n".join(r.fases.antecedentes or []),
                declarado=getattr(e, "resolvio_declarado", "") or "")
            _sent = "fundado" if any(
                _ta_r.prospera(str(getattr(c, "sentido", "")))
                for c in (criterios or [])) else "infundado"
            _rama = _ta_r.rama_revision(_que, _sent)
        # LA VIOLACIÓN PROCESAL SE RECONOCE POR LO QUE SE COMBATE, no por que
        # la pregunta diga «violación procesal»: la del 93/2026 decía «¿debió
        # admitir la ampliación de demanda…?» y esta marca no la veía, así que
        # el estudio no recibió la técnica de los artículos 171 y 172.
        import violacion_procesal as _vpm
        _vp = _vpm.hay(list(getattr(r.fases, "problemas", None) or []),
                       criterios, contexto)
    except Exception as _e:
        print(f"   ⚠️ TALLER: no se pudo fijar la rama técnica: {type(_e).__name__}")


    # EL MARCO SE ESCRIBE A LA VEZ QUE EL ESTUDIO. Son dos llamadas
    # independientes —la del marco sólo mira el material constitucional, la del
    # estudio mira el caso— y ponerlas en paralelo hace que el marco no cueste
    # un segundo de espera. Que APAREZCA ya no depende de que el modelo del
    # estudio se acuerde de escribirlo: lo coloca el compositor.
    tarea_marco = None
    if (e.modo or "").lower() == "generado" and (marco or "").strip():
        import documento_generado as _dg2
        tarea_marco = asyncio.create_task(_dg2.redactar_marco(
            cliente, marco,
            [p for p in (r.fases.problemas or [])], e.es_recurso, e.tipo_asunto))


    _litis_y_material(r, material, [])
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
            escrito_literal=(list(getattr(r.fases, "fuentes", []) or []) + ["", ""])[1])
    # LOS EFECTOS DE UNA VIOLACIÓN PROCESAL SE ORDENAN PASO A PASO (v5 del
    # 93/2026: «dicte otra» sobre una reposición). Se comprueba aquí porque
    # aquí se sabe si la hay.
    _av_ef = f6._efectos_de_reposicion(estudio, criterios, _vp)
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
    return await _terminar(cliente, r, e, criterios, material, estudio,
                           advertencias, avisos, tarea_marco, ruta_salida, qdrant, marco,
                           contexto)


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
    _rama, _vp = "", False
    try:
        import tipos_asunto as _ta_r, fase_rama as _fr_r
        if _ta_r.normalizar(e.tipo_asunto) == "amparo_revision":
            _que = _fr_r.resolvio_a_quo(
                "", "\n".join(r.fases.antecedentes or []),
                declarado=getattr(e, "resolvio_declarado", "") or "")
            _sent = "fundado" if any(
                _ta_r.prospera(str(getattr(c, "sentido", "")))
                for c in (criterios or [])) else "infundado"
            _rama = _ta_r.rama_revision(_que, _sent)
        # LA VIOLACIÓN PROCESAL SE RECONOCE POR LO QUE SE COMBATE, no por que
        # la pregunta diga «violación procesal»: la del 93/2026 decía «¿debió
        # admitir la ampliación de demanda…?» y esta marca no la veía, así que
        # el estudio no recibió la técnica de los artículos 171 y 172.
        import violacion_procesal as _vpm
        _vp = _vpm.hay(list(getattr(r.fases, "problemas", None) or []),
                       criterios, contexto)
    except Exception as _e:
        print(f"   ⚠️ TALLER: no se pudo fijar la rama técnica: {type(_e).__name__}")

    avisos: list[str] = []
    tarea_marco = None
    if (e.modo or "").lower() == "generado" and (marco or "").strip():
        import documento_generado as _dg3
        tarea_marco = asyncio.create_task(_dg3.redactar_marco(
            cliente, marco, [p for p in (r.fases.problemas or [])],
            e.es_recurso, e.tipo_asunto))

    estudio = advertencias = ""
    _litis_y_material(r, material, avisos)
    t0 = _time.perf_counter()
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
            escrito_literal=(list(getattr(r.fases, "fuentes", []) or []) + ["", ""])[1]):
        if paso.get("tipo") == "texto":
            yield paso
        else:
            estudio = paso.get("estudio", "")
            advertencias = paso.get("advertencias", "")
            avisos.extend(paso.get("avisos", []))
    TIEMPOS["estudio de fondo"] = round(_time.perf_counter() - t0, 1)
    _av_ef = f6._efectos_de_reposicion(estudio, criterios, _vp)
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
    res = await _terminar(cliente, r, e, criterios, material, estudio,
                          advertencias, avisos, tarea_marco, ruta_salida, qdrant, marco,
                          contexto)
    yield {"tipo": "listo", "resultado": res}


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
                    marco: str = "", contexto: str = ""):
    """De la salida del modelo al documento entregado.

    Vive fuera de `resolver()` porque la versión en vivo hace exactamente lo
    mismo cuando el flujo termina, y tener dos copias de esto es tener dos
    sitios donde se rompe la congruencia.
    """
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
    relleno = ens.Relleno(
        encabezado=e.encabezado, numero_asunto=e.numero, quejoso=e.quejoso,
        magistrado=e.magistrado, secretario=e.secretario,
        oportunidad=f0.parrafo_oportunidad(r.computo),
        antecedentes=r.fases.parrafos_antecedentes(),
        resumen_acto=r.fases.parrafos_acto(),
        resumen_conceptos=r.fases.parrafos_conceptos(),
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
            material.normas = list(material.normas or []) + nuevos
            print(f"   ⚖️ artículos citados recuperados: {len(nuevos)} nuevos "
                  f"de {len(_extra)} hallados")
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
                marco_escrito=marco_escrito)
        avisos.extend(av_gen)
    else:
        with cronometrar("ensamblado"):
            ruta = ens.ensamblar(e.plantilla, relleno, ruta_salida)
    _, aviso_efectos = ens.formula_resolutivo(relleno.calificaciones)
    # SALVO QUE LA EJECUTORIA NO CONCEDA NADA. Cuando el cómputo cierra por
    # extemporaneidad, el único resolutivo desecha el recurso y el estudio se
    # va al anexo: pedirle al secretario que redacte «los EFECTOS de la
    # concesión» es mandarlo a corregir algo que su proyecto no tiene.
    if aviso_efectos and not getattr(r.computo, "cierra_por_extemporaneidad", False):
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
                "\n".join(getattr(r.fases, "antecedentes", None) or []),
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
        _m = _ce.medir(_txt)
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
            if _pe.get("sin_pregunta"):
                avisos.insert(0,
                    f"{len(_pe['sin_pregunta'])} de {_pe['problemas']} problemas "
                    f"NO se plantean como pregunta en el estudio. Fijar la "
                    f"cuestión con la pregunta expresa es lo que separa un "
                    f"apartado que se entiende de uno que hay que releer: "
                    f"«{_pe['sin_pregunta'][0]}…»")
            for _q, _d in _ce.cierre_ciego(_txt):
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

    return Resultado(ruta=ruta, computo=r.computo, fases=r.fases, encargo=e,
                     partes=r.partes, estudio=estudio, advertencias=advertencias,
                     huecos=ens.huecos_pendientes(ruta),
                     avisos=_limpios)




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


def _quejoso_del_amparo(e: Encargo, partes=None) -> str:
    """A quién se concede o niega el amparo.

    En un recurso el formulario guarda a «quien promueve» y eso es el
    RECURRENTE; quien pidió el amparo está en la ficha de partes, leída de la
    sentencia recurrida. Si lo tecleado es una autoridad y la ficha dice otro
    nombre, manda la ficha. Se resuelve AQUÍ, al armar los datos, y no sólo al
    fichar: la regeneración desde una sesión guardada no vuelve a fichar y el
    711/2025 tenía a la UIF de quejosa en el encargo ya persistido."""
    _tecleado = str(getattr(e, "quejoso", "") or "").strip()
    _leido = str(getattr(partes, "quejoso", "") or "").strip()
    if (getattr(e, "es_recurso", False) and _leido and _tecleado
            and not _mismo_nombre(_leido, _tecleado) and _parece_autoridad(_tecleado)):
        return _leido
    return _tecleado


def _recurrente_de(e: Encargo, partes=None) -> str:
    _propio = str(getattr(e, "recurrente", "") or "").strip()
    if _propio:
        return _propio
    _tecleado = str(getattr(e, "quejoso", "") or "").strip()
    if _quejoso_del_amparo(e, partes) != _tecleado:
        return _tecleado
    return ""


def _datos_estructura(e: Encargo, antecedentes: str = "", acto: str = "",
                     partes=None) -> dict:
    """Lo que la estructura necesita.

    Decía «TODO sale del encargo» y por eso se lanzaba a ciegas. Del encargo
    salen la competencia y la existencia; los RESULTANDOS necesitan el acto
    —para individualizar la sentencia recurrida con su fecha y su expediente—
    y la ficha de partes —para nombrar al tercero interesado en vez de escribir
    «la persona a quien resulta tal carácter»—.
    """
    import fase0_oportunidad as _f0
    _terceros = ""
    if partes is not None:
        _terceros = str(getattr(partes, "tercero_interesado", "") or "")
    return {
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
        "quejoso": _pv.separar(_quejoso_del_amparo(e, partes))["parte"] or _quejoso_del_amparo(e, partes),
        # Quien recurrió, cuando no es el quejoso. Vacío = es el mismo.
        "recurrente": _recurrente_de(e, partes),
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
                             partes=None, criterios=None, fases=None):
    """El documento escrito entero. Devuelve (ruta, avisos, estructura)."""
    import documento_generado as dg
    import fase0_oportunidad as _f0

    # LA SEGUNDA LLAMADA TAMBIÉN NECESITA EL ACTO Y LAS PARTES. Se arregló la
    # de arriba y ésta se quedó igual: cuando `estructura_previa` es None
    # —porque se recompone el documento sin haber pasado por el adelanto— la
    # estructura volvía a escribirse a ciegas, y con ella la perífrasis.
    datos = _datos_estructura(e, "\n".join(relleno.antecedentes or []),
                              acto=acto, partes=partes)
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
    datos["expediente_origen"] = getattr(fases, "expediente_origen", "") or ""
    datos["fecha_origen"] = getattr(fases, "fecha_origen", "") or ""
    # LA ESTRUCTURA SE ESCRIBE UNA VEZ. El resolver recompone el documento
    # entero, y volver a pedirla al modelo son treinta segundos por nada: no
    # depende del estudio ni del criterio, sólo del asunto.
    est = estructura_previa or await dg.redactar_estructura(cliente, datos)

    # LA SÍNTESIS DE LA PORTADA. Se pide con el estudio YA REDACTADO, no con
    # los datos del asunto: una síntesis escrita antes del estudio resumiría lo
    # que se pensaba resolver, no lo que se resolvió, y es justo el desajuste
    # que hace inservible un resumen. Si falla, el documento sale sin ella.
    _sint = {}
    try:
        import fase_sintesis as _fs
        _sint = await _fs.sintetizar(
            cliente,
            tipo_asunto=(getattr(e, "tipo_asunto", "") or ""),
            expediente=(getattr(e, "numero", "") or ""),
            quejoso=(getattr(e, "quejoso", "") or ""),
            sentido=str(getattr(relleno, "calificaciones", "") or ""),
            estudio="\n\n".join(relleno.estudio or []))
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
