"""FASE 6 — el estudio de fondo, con el criterio del secretario.

Aquí es donde el proyecto entero cobra sentido, y donde se sostiene la regla
que David fijó desde el principio:

    su CRITERIO manda el sentido
    el CORPUS manda la forma
    la LEY manda el fundamento

La máquina no decide el fallo. Construye la mejor demostración posible del
fallo que el secretario ya decidió, y si encuentra un obstáculo serio —una
jurisprudencia obligatoria en contra, una causal de improcedencia— lo SEÑALA
en un apartado de advertencias en lugar de cambiar el sentido por su cuenta.

EL REGISTRO ESTÁ MEDIDO, no inventado. Sobre 40 estudios firmados del corpus:

    largo ............ mediana 3,733 palabras · p90 6,618 (117 estudios)
    párrafo .......... 49 palabras (p90 101)
    frase ............ 35 palabras (p90 69)
    conectores ....... «Lo anterior» 26/40, «En ese sentido» 23, «Por tanto» 22,
                       «En consecuencia» 22, «No obstante» 18, «En efecto» 16
    calificación ..... fundado 103 · infundado 83 · inoperante 67 · ineficaz 15
    la autoridad ..... «la responsable» 27/40, «la autoridad responsable» 25/40
    el órgano ........ «este Tribunal Colegiado» 12/40, y voz impersonal
                       («se estima», «se considera») 31

Y dos recursos que el corpus usa y funcionan, aunque sean minoritarios: la
calificación anunciada en las primeras líneas (40%) y la cuestión jurídica
planteada como pregunta explícita y respondida acto seguido (18%).
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

import vigencia_tesis as _vig

# Las entidades donde el Código Nacional YA rige. Vacío por omisión: la
# vigencia es un dato del mundo, no del acervo, y ponerla a ojo sería inventar.
# Se declara con CNPCF_VIGENTE=jalisco,colima… cuando se sepa con certeza.
CNPCF_VIGENTE = {x.strip().lower() for x in
                 os.getenv("CNPCF_VIGENTE", "").split(",") if x.strip()}

MODELO_ESTUDIO = os.getenv("MODELO_ESTUDIO", "gpt-5.6-luna")

# El estudio SÍ razona: es el único paso del pipeline donde hay que construir
# una demostración, no extraer lo que ya está escrito. Los resúmenes van sin
# razonamiento porque son lectura; esto es argumentación.
ESFUERZO_ESTUDIO = os.getenv("ESFUERZO_ESTUDIO", "high")

# Medido sobre 117 estudios de fondo reales de las carpetas del tribunal —no
# sobre los 40 del primer muestreo—. La DISPERSIÓN es lo que importa: el p90
# está en 6,618 palabras, así que el tope de aviso no puede ser un múltiplo de
# la mediana. Con el umbral anterior (1.6 x mediana) se marcaba como excesivo
# el 25% de los engroses del propio secretario, incluido el de este caso.
# El texto de una tesis de la Undécima Época con Hechos/Criterio/Justificación
# ronda los 3,000 caracteres. Cortarlo es peor que no citarla.
TESIS_CARACTERES = 4000
# La norma es la premisa mayor: se entrega entera o no se entrega.
NORMA_CARACTERES = 4000

# CUÁNTAS TESIS ENTRAN AL PROMPT. El RAG devuelve todas las que encuentra —44 en
# el QA 143-2026— y meterlas enteras da un prompt de 108,000 caracteres del que
# el 93% es material. Medido: con ese prompt el estudio citó UNA sola tesis, y
# la instrucción de fundar quedaba al 2% del texto, enterrada bajo 30,000 tokens
# de jurisprudencia. Diez bien elegidas se leen; cuarenta y cuatro se hojean.
MAX_TESIS_PROMPT = 10
# Las de la figura que no traía ningún problema (fase E): cupo propio, fuera de
# las diez (`_bloque_material`).
MAX_TESIS_FIGURA_PROMPT = 3
MAX_NORMAS_PROMPT = 12

PALABRAS_ESTUDIO = 3733
PALABRAS_ESTUDIO_P90 = 6618

# ═══ LA MEDIDA DE LA SOLUCIÓN, NO DEL CONSIDERANDO (26-sep-2026) ════════════
# Los 3,733 de arriba son la mediana del considerando ENTERO —con el resumen
# de la sentencia reclamada y el de los conceptos dentro—, y el prompt los
# pedía para la Solución sola: contaba dos veces los resúmenes y empujaba a
# rellenar (diagnóstico del estudio, F1). Medido hoy sobre los 24 engroses de
# oro sólido del banco Kingston —campo «oro», desde el rótulo «Solución» hasta
# los resolutivos o el considerando de efectos; el ADC 590/2024 no lleva
# rótulo y su considerando entero es la Solución—:
#
#     mediana 3,183 · p90 5,361 · mínimo 1,303 (ADA 263/2025) · 23 engroses
#
# Sin el ADC 810/2025: 21,787 palabras, cuatro veces el siguiente, con las
# fichas de las tesis desglosadas renglón por renglón; con él el p90 salta a
# 8,417 porque la cola la forman tres engroses (43/2025, 590/2024 y 810/2025)
# que transcriben constancias y ejecutorias enteras.
#
# EN LA v2 ES UN TECHO, NO UNA META (propuesta aprobada por David, fila 12):
# un asunto con una sola cuestión viva se resuelve en mucho menos, y nada se
# escribe para acercarse a una cifra. La v1 sigue pidiendo sus 3,733.
SOLUCION_MEDIANA = 3183
SOLUCION_P90 = 5361
# Por debajo de esto no hay un solo engrose del banco (el más corto, 1,303):
# el aviso de «se quedó corto» de la v2 no acusa a ninguno de los 24. El de la
# v1 —45 % de 3,733, es decir 1,680— acusaba a tres (263, 274 y 282/2025).
SOLUCION_PISO = 1000
# El aviso de exceso de la v2 salta a 1.25 veces el techo. Calibrado: acusa a
# los mismos tres que el de la v1 (43, 590 y 810), no a uno más; con el techo
# pelado acusaría también al ADC 192/2025 (5,425), que es un engrose bueno.
SOLUCION_EXCESO = round(1.25 * SOLUCION_P90)


# ═══ LAS DOS VARIANTES DEL PROMPT DEL ESTUDIO (26-sep-2026) ══════════════════
# David aprobó la limpieza del prompt («Paso 1» de la propuesta del estudio):
# quitar las órdenes que se contradicen y las capas que fabrican repetición.
# Se hace como VARIANTE y no encima de lo que hay, porque se mide antes de
# encender: «v1» es el prompt de producción, congelado por una prueba de
# instantánea (test_prompt_v2.py) con UNA sola corrección que vale para todos
# por ser de ley —el orden del artículo 189—; «v2» es la limpieza.
#
# CÓMO SE ELIGE. `ESTUDIO_PROMPT` fija la de todos (por omisión «v1»); una
# cuenta de casa puede pedir otra en el formulario (`variante_estudio`), y la
# petición manda. Viaja como la forma de la sentencia: encargo → material →
# prompt (ver `redactor_adelanto._formato_al_material`). Render no reinicia al
# guardar una variable: cambiar la global exige un despliegue.
#
# EL PASO 2 AÑADE DOS (26-sep-2026, contrato del Paso 2). El banco midió la v2
# contra la v1 ese día: la v2 reduce la Solución a la mitad pero contesta con
# razón propia sólo el 73 % de los argumentos autónomos (la v1, el 79 %) y
# duplica las omisiones graves; al acortar sin saber qué argumentos hay, funde
# los que traen dato propio en una respuesta global. De ahí:
#   · «v3» = la v2 + el INVENTARIO de argumentos (`inventario.py`) como lista
#     de datos + la regla de las MARCAS (`marcas.py`). Sin llamada nueva.
#   · «v4» = la v3 + el plan del estudio (guion), que trae la pieza del plan.
#     Sin el plan, la v4 escribe como la v3: aquí no hay guion que añadir.
# Las dos son de la familia de la v2: todo lo que la v2 cambió respecto de la
# v1 lo siguen teniendo (ver `_v2`).
VARIANTES = ("v1", "v2", "v3", "v4")
# «A», «B», «C-lite» y «C» son los nombres de la propuesta (w2_final §6.1).
_ALIAS_VARIANTE = {"a": "v1", "b": "v2", "1": "v1", "2": "v2", "3": "v3",
                   "4": "v4", "clite": "v3", "c-lite": "v3", "c_lite": "v3",
                   "c": "v4"}
# Las que escriben marcas y reciben el inventario.
CON_INVENTARIO = ("v3", "v4")


def normalizar_variante(x, por_omision: str = "") -> str:
    """«v1»…«v4» o `por_omision` si no se reconoce. Nunca inventa una."""
    t = str(x or "").strip().lower()
    t = _ALIAS_VARIANTE.get(t, t)
    return t if t in VARIANTES else por_omision


def variante_global(tipo_asunto: str = "") -> str:
    """La de todos: `ESTUDIO_PROMPT`, y «v1» si falta o no se reconoce.

    ENCENDIDO POR TIPO (contrato del Paso 2): en amparo directo manda
    `ESTUDIO_PROMPT_AD` si está puesta y se reconoce; es el primer tipo que se
    encenderá (w2_final §6.1, paso 4: recursos en sombra). Sin tipo, o sin esa
    variable, la global. Ninguna de las dos cambia de valor en esta entrega.

    Se lee en cada petición y no al importar, para que una prueba pueda
    cambiarla; en Render da igual, porque la variable sólo cambia con un
    despliegue."""
    # ENCENDIDA PARA TODOS (David, 26-sep-2026: «enciende ambas para todos»,
    # tras la tercera medición ciega: en la moderna la v4 contesta más y con
    # cinco veces menos omisiones graves; en la estándar, más argumentos
    # autónomos, graves iguales y mucho menos repetición). `ESTUDIO_PROMPT=v1`
    # la apaga sin tocar código (Render: con un despliegue).
    glob = normalizar_variante(os.getenv("ESTUDIO_PROMPT", "v4"), "v4")
    if tipo_asunto:
        try:
            if _ta_p.normalizar(tipo_asunto) == "amparo_directo":
                return normalizar_variante(os.getenv("ESTUDIO_PROMPT_AD", ""), glob)
        except Exception:
            pass
    return glob


def _v2(material) -> bool:
    """¿Es de la familia de la v2? La v3 y la v4 SON la v2 más lo suyo: cada
    sitio que pregunta esto —el prompt, el cierre, la calidad que cuenta una
    calificación por apartado— tiene que tratarlas como a la v2."""
    return normalizar_variante(getattr(material, "variante", "v1"), "v1") in ("v2",) + CON_INVENTARIO


def con_inventario(material) -> bool:
    """¿Esta variante recibe el inventario y escribe marcas? (v3 y v4)"""
    return normalizar_variante(getattr(material, "variante", "v1"), "v1") in CON_INVENTARIO


def _objetivo_palabras(material, criterios) -> int:
    """La medida del corpus para la estándar; la de `formato_sentencia` para la
    moderna. El aviso de «se quedó corto» mide contra ESTE número: medido contra
    el del corpus, toda sentencia moderna saldría acusada de corta.

    EN LA FAMILIA v2, LA ESTÁNDAR MIDE CONTRA SU REFERENCIA (p2-congruencia,
    26-sep-2026): 2,000 a 2,500 palabras según los problemas vivos, el rango
    que David aceptó para que cada argumento reciba su respuesta
    (`formato_sentencia.referencia_estandar`). La moderna, igual en todas: su
    medida por problemas vivos."""
    import formato_sentencia as _fs_o
    if _fs_o.normalizar(getattr(material, "formato", "")) == _fs_o.MODERNA:
        return _fs_o.palabras_moderna(criterios)
    if _v2(material):
        return _fs_o.referencia_estandar(criterios)
    return PALABRAS_ESTUDIO

CONECTORES = ("Lo anterior", "En ese sentido", "Por tanto", "En consecuencia",
              "No obstante", "En efecto", "Ahora bien")

# El sentido se dicta en singular («ineficaz») pero se escribe concordando con
# «los conceptos»: «son ineficaces». Sin esto sale «son ineficaz» en la primera
# línea de la sentencia, que es donde más se nota.
import tipos_asunto as _ta_p
_PLURAL = {k: v[0] for k, v in _ta_p.CALIFICACIONES.items()}
_PLURAL_VIEJO = {"fundado": "fundados", "infundado": "infundados",
           "inoperante": "inoperantes", "ineficaz": "ineficaces",
           "fundados": "fundados", "infundados": "infundados",
           "inoperantes": "inoperantes", "ineficaces": "ineficaces"}


def _calificacion(criterios: list["Criterio"]) -> str:
    """La frase de apertura, concordada y en el orden en que se estudian.

    Con sentidos distintos el corpus no dice «fundados e inoperantes» a secas,
    sino que anuncia el resultado mixto: «en parte fundados y en parte
    inoperantes».
    """
    vistos: list[str] = []
    for c in criterios:
        pl = _PLURAL.get(c.sentido.strip().lower(), c.sentido.strip().lower())
        if pl not in vistos:
            vistos.append(pl)
    if not vistos:
        return "el que corresponda"
    if len(vistos) == 1:
        return vistos[0]
    return " y ".join("en parte " + v for v in vistos)


@dataclass
class Criterio:
    """Lo que el secretario decidió para un problema jurídico."""
    problema: str
    sentido: str                      # fundado | infundado | inoperante |
                                      # ineficaz | innecesario
    razonamiento: str = ""            # el porqué, que es lo que de verdad alinea
    # PRINCIPAL o ACCESORIO. El principal es aquel del que dependen los demás:
    # si prospera, el estudio de los otros queda sin materia. Sin esta marca no
    # se puede aplicar la sustracción de materia ni ordenar el estudio por
    # prelación lógica, que es como se ordena un engrose.
    jerarquia: str = "accesorio"
    # LA LETRA DEL GRUPO, cuando el secretario decidió que dos o más
    # planteamientos se resuelven con UNA SOLA línea argumentativa. Vacío es lo
    # normal: cada problema con su apartado.
    grupo: str = ""
    # La distribución del acervo sobre ESTE problema: {"sentido", "porcentaje",
    # "n", "confianza", "frase"}. No funda nada —un colegiado no obliga a
    # otro— pero dice si el sentido va con la corriente o contra ella, y eso
    # cambia lo que hay que escribir.
    prediccion: dict = field(default_factory=dict)


@dataclass
class Material:
    """Lo que el RAG encontró para un problema. Sólo entra lo VERIFICADO.

    LO VERIFICADO Y LO TRANSCRITO NO SON LO MISMO. Desde el
    16-sep-2026 puede entrar en `normas` un precepto que el acervo no
    tenía, transcrito de su fuente oficial en línea: viaja con
    `de_internet=True`, su `dominio` y su `url`, y la lista de los que
    entraron así se cuelga en `preceptos_de_internet`. Quien escriba
    la próxima función que lea `normas` tiene que saberlo: la premisa
    de que todo aquí está cotejado ya no vale para esos.
    """
    tesis: list[dict] = field(default_factory=list)      # registro, rubro, texto
    normas: list[dict] = field(default_factory=list)     # cuerpo_legal, articulo, texto
    convencional: list[dict] = field(default_factory=list)
    # Los estudios de fondo van APARTE y son molde de FORMA, nunca fundamento.
    moldes: list[dict] = field(default_factory=list)
    # El sondeo del acervo de precedentes: cómo resolvieron otros este mismo
    # problema. No funda —un colegiado no obliga a otro— pero dice si uno se
    # está apartando de la corriente y de dónde sacar la objeción que hay que
    # responder. Es `fase_precedente.Sondeo`; se guarda suelto para no cruzar
    # los imports.
    # LOS PRINCIPIOS CON LOS QUE EL CIRCUITO RAZONA ESTA CUESTIÓN. Salen de los
    # holdings —el 99.3% los trae— y llegan sin coste: viajan en la misma
    # consulta de la co-citación. Para la suspensión, por ejemplo: apariencia
    # del buen derecho (16 sentencias), interés social (11), peligro en la
    # demora, conservación de la materia del amparo. Decirle al modelo con qué
    # nociones razona el tribunal es decirle por dónde va el razonamiento, no
    # sólo qué citar.
    principios: list = field(default_factory=list)
    sondeo: object = None
    # ═══ EL ESPEJO DEL PROPIO TRIBUNAL ═══════════════════════════════════
    # Seis sentencias del tribunal que redacta, sobre este mismo punto, con su
    # fecha, su sentido literal y su enlace. Ver `fase_espejo.py`.
    #
    # NO ENTRA AL PROMPT, Y ESO ES DELIBERADO. `_bloque_precedente` no lo
    # incluye: un número de expediente dentro del prompt acaba copiado literal
    # en la prosa firmada —ya ha pasado tres veces con los ejemplos, está
    # anotado más abajo— y aquí el material son expedientes PROPIOS con enlace,
    # así que copiarlos sería citar en una sentencia asuntos que nadie eligió
    # citar. El espejo es para la PANTALLA del secretario, que abre el PDF y
    # compara; el documento no lo menciona.
    espejo: list = field(default_factory=list)
    # LOS DOS DATOS DERIVADOS. Quién dictó el acto reclamado del amparo
    # indirecto —«amparo» u «ordinaria»— y de qué cuaderno viene la recurrida
    # —«principal» o «incidental»—. Juntos deciden si la Ley de Amparo gobierna
    # el acto, y de ahí cuelgan el filtro de jurisprudencia y el verificador de
    # preceptos. Ver `fase_rama.sede_del_acto` y `fase_rama.cuaderno_recurrido`.
    sede_del_acto: str = ""
    cuaderno: str = ""
    # LA PREGUNTA DECISIVA DEL PRINCIPAL (SPEC E3, AR 631/2025): el documento
    # de `pregunta_decisiva.formular`, o None. Viaja con el material por la
    # misma razón que la materia: llega a la propuesta, al estudio y a la
    # tarjeta sin un parámetro nuevo en cada sitio.
    decisiva: object = None
    # LA MATERIA VIAJA CON EL MATERIAL, no como parámetro. Hay cuatro sitios que
    # arman el prompt y cada parámetro nuevo es un sitio donde olvidarlo; el
    # Material ya llega a todos. Y aquí importa de veras: entregar la
    # arquitectura equivocada no es un defecto de forma, es mandar escribir al
    # revés —en laboral se pide ciclos cortos y en administrativa lo contrario—.
    materia: str = ""
    # Para nombrar a las partes con las figuras que existen en este tipo.
    tipo_asunto: str = "amparo_directo"
    # LA FICHA PROCESAL DEL ASUNTO, ya escrita como bloque de datos
    # (`ficha_procesal.bloque`; SPEC_E2, 28-sep-2026). La fija en CADA
    # petición `redactor_adelanto._formato_al_material`, como la reasunción:
    # el material vive en la memoria del worker. Vacía = el prompt queda
    # idéntico al de antes (la v1 está congelada por instantánea).
    ficha_procesal: str = ""
    # LA ENTIDAD, por el mismo motivo que la materia: viaja con el material
    # porque el material ya llega a todos los prompts. Sale de la colección
    # estatal que eligió el secretario («leyes_queretaro» → «Querétaro»).
    entidad: str = ""
    # EL TRIBUNAL QUE RESUELVE (rediseño, punto 3, 29-sep-2026): la fuerza de
    # cada tesis se calcula respecto de él (`fuerza_juridica.anotar`).
    tribunal: str = ""
    # ¿COMPLETA O PROVISIONAL? {"estado": "completa" | "provisional", "faltan":
    # [...]} (rediseño, etapa 2). Provisional = venció la espera de la pregunta
    # decisiva y falta la búsqueda de su figura: la propuesta SALE igual (el
    # secretario siempre puede pedirla al motor), pero no como recomendación.
    consulta_estado: dict = field(default_factory=dict)
    # LA DEMOSTRACIÓN POR REQUISITOS (rediseño, etapa 2; `requisitos.py`).
    requisitos: dict = field(default_factory=dict)
    # LA FORMA DE LA SENTENCIA —«estandar» o «moderna»—, por la misma razón que
    # la materia: dos redactores arman el prompt y el material llega a los dos.
    # Ver `formato_sentencia.py`.
    formato: str = "estandar"
    # EL REPARTO DE LA FASE 3: cada problema con su «cubre», la lista de
    # planteamientos que contesta, y cuántos trae el escrito. Con él la forma
    # estándar sabe qué concepto califica cada criterio.
    problemas: list = field(default_factory=list)
    n_planteamientos: int = 0
    # La tarea de la síntesis de la versión moderna, que corre a la vez que el
    # estudio y se recoge al componer. None en la estándar.
    sintesis: object = None
    # LA VARIANTE DEL PROMPT DEL ESTUDIO —«v1» o «v2»—, por la misma razón que
    # la forma: los dos redactores arman el prompt y el material llega a los
    # dos. La fija `redactor_adelanto._formato_al_material` en CADA petición,
    # porque el material vive en la memoria del worker de una generación a la
    # siguiente. Ver `VARIANTES` arriba.
    variante: str = "v1"
    # EL INVENTARIO DE ARGUMENTOS (Paso 2a, 26-sep-2026): los segmentos de
    # `inventario.segmentos`, sólo con la v3 y la v4. Lo fija
    # `redactor_adelanto._formato_al_material` en CADA petición —vacío en las
    # demás variantes—, por lo mismo que la variante: el material vive en la
    # memoria del worker y el inventario de una vuelta no puede colarse en la
    # siguiente. Lo leen el prompt (el bloque) y `_terminar` (el control V1).
    inventario: list = field(default_factory=list)
    # LA SUPLENCIA QUE DECIDIÓ EL SECRETARIO (Decisión 4 de David, 26-sep-2026):
    # {fraccion, a_favor_de, confirmada}, o vacío. Viaja con el material por lo
    # mismo que la forma: dos redactores arman el prompt y a los dos les llega el
    # material. La fija `redactor_adelanto._formato_al_material` en CADA
    # petición. Ver `suplencia.py`.
    suplencia: dict = field(default_factory=dict)


# LA ÚNICA EXCEPCIÓN A «INNEGOCIABLE», y hubo que escribirla porque el pipeline
# se contradecía a sí mismo. En el 382/2024 —un trabajador despedido— el bloque
# del criterio decía «NO cambies el sentido» y la regla de suplencia decía que
# la inoperancia no cabe. El modelo obedeció a la que iba rotulada INNEGOCIABLE,
# escribió «inoperante» y le añadió un descargo sobre la suplencia. Hizo lo
# único que podía hacer con dos órdenes contrarias.
#
# No se le quita la autoridad al secretario: el sentido sigue siendo suyo. Lo
# que se le quita al modelo es la posibilidad de escribir la inoperancia SIN
# haber intentado antes el fondo, que es lo que el artículo 79 manda. Si tras
# suplir el argumento en su mejor versión la inoperancia se sostiene, se escribe
# y se explica. Si no se sostiene, se dice en las advertencias, que es el cauce
# que este pipeline ya tenía para discrepar.
# EL TEXTO DEL 79 LO DICE, Y NO DICE LO QUE YO SUPONÍA. Traído del archivo
# oficial, no de memoria: «En los casos de las fracciones I, II, III, IV, V y
# VII de este artículo la suplencia se dará AUN ANTE LA AUSENCIA de conceptos
# de violación o agravios». La VI no está en esa lista: ésa exige una violación
# evidente que haya dejado sin defensa, y por eso no entra aquí.
_SUPLENCIA_ABSOLUTA = {
    "laboral": ("la persona trabajadora", "79, fracción V"),
    "penal": ("la persona inculpada o sentenciada", "79, fracción III"),
    "familiar": ("la persona menor de edad o incapaz", "79, fracción II"),
    "agraria": ("el núcleo de población o el ejidatario", "79, fracción IV"),
}

# LA FRACCIÓN V NO SE ACABA EN LA MATERIA LABORAL, y ése era el hueco que se
# comía el asunto de la cuota pensionaria. Dice, literal: «en favor de la
# persona trabajadora, CON INDEPENDENCIA DE QUE LA RELACIÓN entre la persona
# empleadora y empleada esté regulada por el derecho laboral O POR EL DERECHO
# ADMINISTRATIVO». Una pensión del ISSSTE se litiga en materia administrativa y
# llega aquí por revisión fiscal, pero quien promueve sigue siendo una persona
# trabajadora: la suplencia opera igual. Clasificar el asunto como
# «administrativa» lo dejaba fuera, que es tanto como quitarle la suplencia por
# haber elegido bien la vía.
# EL \b DEL PRINCIPIO (26-sep-2026): sin él, «pensión» se encontraba dentro de
# «SUSPENSIÓN», que está en toda demanda de amparo. Medido en el banco Kingston:
# dos amparos agrarios (ADA 263/2025 y 448/2025) salían como pensionarios.
_RX_TRABAJADOR_EN_ADMINISTRATIVA = re.compile(
    r"\b(?:pensi[óo]n|pensionari|jubilaci[óo]n|jubilad|cesant[íi]a|"
    r"cuota\s+pensionaria|haber\s+de\s+retiro|ISSSTE|IMSS|"
    r"burocr[áa]tic|trabajador(?:a|es)?\s+al\s+servicio\s+del\s+estado|"
    r"seguridad\s+social)", re.I)

# LA FRACCIÓN VII NO DEPENDE DE LA MATERIA SINO DE LA PERSONA, así que aquí no
# se puede decidir: se le RECUERDA LA REGLA a quien redacta y se le pide que
# mire las constancias. Afirmar desde un patrón que alguien está en pobreza o
# marginación sería inventarse un hecho del expediente.
_RX_DESVENTAJA = re.compile(
    r"pobreza|marginaci[óo]n|ind[íi]gena|comunidad\s+originaria|discapacidad|"
    r"adulto\s+mayor|persona\s+mayor|analfabet|migrante|reclusi[óo]n", re.I)


# DÓNDE NO HAY SUPLENCIA, Y ESTO LO ROMPÍ YO. El artículo 79 empieza diciendo
# «La autoridad que conozca DEL JUICIO DE AMPARO deberá suplir la deficiencia»,
# y una revisión fiscal no es un juicio de amparo: es el recurso del artículo
# 63 de la LFPCA, y quien recurre es LA AUTORIDAD. No hay parte débil a la que
# suplir, y suplir a favor del ISSSTE contra su propio pensionado sería el
# mundo al revés.
#
# El aviso saltó en el proyecto de la cuota pensionaria porque hoy amplié la
# fracción V a los trabajadores que litigan en la vía administrativa. La
# ampliación es correcta —el texto oficial dice «con independencia de que la
# relación esté regulada por el derecho laboral o por el derecho
# administrativo»— pero se me olvidó mirar la VÍA: eso vale en el AMPARO que
# venga después contra la sentencia del Tribunal, no en este recurso.
_SIN_SUPLENCIA = {"revision_fiscal"}


def _aviso_de_suplencia(criterios: list, materia: str,
                       material: str = "", tipo_asunto: str = "") -> list:
    import tipos_asunto as _ta_s
    if _ta_s.normalizar(tipo_asunto) in _SIN_SUPLENCIA:
        return []
    m = (materia or "").strip().lower()
    par = _SUPLENCIA_ABSOLUTA.get(m)
    # El trabajador que litiga en la vía administrativa.
    if not par and _RX_TRABAJADOR_EN_ADMINISTRATIVA.search(material or ""):
        par = ("la persona trabajadora o pensionada", "79, fracción V")
    if not any("inoperan" in str(getattr(c, "sentido", "")).lower()
               for c in (criterios or [])):
        return []
    # LA VII SE MIRA ANTES DE RENDIRSE (defecto L4, 26-sep-2026). Aquí había un
    # `if not par: return []` ANTES de mirar la fracción VII, así que en civil,
    # mercantil o administrativa —las materias sin suplencia absoluta— la
    # desventaja social no se recordaba nunca, que es justo donde la VII es la
    # única puerta: ella «no depende de la materia sino de la persona».
    if not par:
        if not _RX_DESVENTAJA.search(material or ""):
            return []
        return ["",
                "── UNA SALVEDAD, Y SÓLO UNA ──",
                "Se te dicta INOPERANTE y la materia de este asunto no trae",
                "suplencia absoluta. Pero la fracción VII del artículo 79 de la Ley",
                "de Amparo no depende de la materia sino de la persona: opera «en",
                "favor de quienes por sus condiciones de pobreza o marginación se",
                "encuentren en clara desventaja social para su defensa en el",
                "juicio», aun sin conceptos de violación. En el material hay",
                "indicios de esa condición. NO LA AFIRMES si el expediente no la",
                "acredita —eso sería inventar un hecho—; si consta, estudia el",
                "planteamiento en el fondo antes de escribir la inoperancia, y si al",
                "suplirlo prospera, dilo en ADVERTENCIAS para que el secretario lo",
                "valore.",
                ]
    quien, precepto = par
    extra = []
    if _RX_DESVENTAJA.search(material or ""):
        extra = ["",
                 "Y MIRA ADEMÁS LA FRACCIÓN VII, que no depende de la materia",
                 "sino de la persona: opera «en favor de quienes por sus",
                 "condiciones de pobreza o marginación se encuentren en clara",
                 "desventaja social para su defensa en el juicio». En el material",
                 "hay indicios de esa condición. NO LA AFIRMES si el expediente no",
                 "la acredita —eso sería inventar un hecho—, pero si consta, dilo y",
                 "suple con ella.",
                 ]
    return ["",
            "── UNA SALVEDAD, Y SÓLO UNA ──",
            f"Se te dicta INOPERANTE en un asunto de materia {m}. Si quien",
            f"promueve es {quien}, la suplencia del artículo {precepto} de la Ley",
            "de Amparo es ABSOLUTA y opera aun sin conceptos de violación. No",
            "puedes escribir esa inoperancia sin haber hecho antes esto:",
            "",
            "  1. RECONSTRUYE el planteamiento en su mejor versión posible, la",
            "     que la parte habría escrito con el mejor abogado, y DÉJALO",
            "     ESCRITO: «Suplida la deficiencia, el concepto plantea que…».",
            "  2. ESTÚDIALO EN EL FONDO así reconstruido.",
            "  3. Y sólo si ni siquiera así toca ninguna razón del acto, escribe",
            "     la inoperancia y di exactamente qué versión examinaste.",
            "",
            "Si al suplirlo resulta FUNDADO, no escribas la inoperancia: dilo en",
            "ADVERTENCIAS con todas las letras para que el secretario lo valore.",
            "Ésta es la única orden que está por encima del sentido dictado, y no",
            "es criterio: es un mandato del artículo 79 que ningún acuerdo de",
            "ponencia puede dispensar.",
            ] + extra


def _bloque_suplencia(material) -> str:
    """La suplencia CONFIRMADA por el secretario, como bloque propio del prompt.

    Sin confirmar —o con «sin suplencia»— no añade nada: el estudio se comporta
    como antes. Ver `suplencia.bloque`."""
    try:
        import suplencia as _sp
        return _sp.bloque(getattr(material, "suplencia", None) or {},
                          getattr(material, "tipo_asunto", "") or "")
    except Exception as _es:
        print(f"   ⚠️ SUPLENCIA: no se pudo armar el bloque: {type(_es).__name__}")
        return ""


# A FAVOR o EN CONTRA de quien promueve. Es lo único que hay que comparar: el
# acervo habla del fallo y el criterio, del planteamiento.
_A_FAVOR = {"concede", "revoca", "modifica", "fundado", "fundado_suplido",
            "ampara"}
_EN_CONTRA = {"niega", "confirma", "sobresee", "desecha", "infundado",
              "inoperante", "ineficaz", "inatendible"}


def _misma_direccion(a: str, b: str) -> bool:
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    if not a or not b:
        return True
    for grupo in (_A_FAVOR, _EN_CONTRA):
        if a in grupo and b in grupo:
            return True
    return not ((a in _A_FAVOR and b in _EN_CONTRA)
                or (a in _EN_CONTRA and b in _A_FAVOR))


def _bloque_criterio(criterios: list[Criterio], materia: str = "",
                     material_texto: str = "", tipo_asunto: str = "",
                     formato: str = "", problemas: list = None,
                     variante: str = "v1", decisiva: dict = None) -> str:
    if not criterios:
        return ""
    import formato_sentencia as _fs_c
    _moderna = _fs_c.normalizar(formato) == _fs_c.MODERNA
    # LA v2 CAMBIA CUATRO COSAS DE ESTE BLOQUE Y NINGUNA DEL SENTIDO (26-sep-
    # 2026): cómo se abre el grupo, la medida de lo que no se estudia, el
    # cierre —que en la v2 se decide al final del prompt, en un solo sitio— y
    # los efectos, que se describen en vez de enseñarse con un ejemplo que se
    # copiaba. La suplencia de abajo es la misma en las dos.
    _v2c = normalizar_variante(variante, "v1") == "v2"
    lineas = ["", "═" * 71,
              "EL CRITERIO DEL SECRETARIO — DIRECTIVA INNEGOCIABLE",
              "═" * 71,
              "Tu papel NO es decidir el fallo: es CONSTRUIR la mejor demostración",
              "jurídica posible del sentido que él ya fijó. Elige los argumentos, las",
              "tesis y el orden de estudio que lo sostengan con el mayor rigor.", "",
              # ── LA PREGUNTA EXPRESA ────────────────────────────────────────
              # Roberto Lara Chagoyán, «Sobre la estructura de las sentencias en
              # México», § 3.2, principio de delimitación: «la desgracia de
              # muchas malas sentencias comienza con el descuido del deber de
              # fijar cuidadosamente la cuestión… Una forma de mejorar los
              # planteamientos es utilizar la PREGUNTA EXPRESA».
              #
              # El pipeline ya calculaba estas preguntas —bien formuladas— y
              # las tiraba: medido sobre el proyecto de la cuota pensionaria,
              # tres preguntas calculadas y CERO en el documento. Lo que se
              # arregla no es formularlas, es no perderlas.
              # DESDE EL 25-SEP-2026 ESTO ES SÓLO DE LA VERSIÓN MODERNA. En la
              # estándar los problemas guían y no se escriben: el estudio va
              # concepto por concepto. Ver `formato_sentencia.py`.
              ] + _fs_c.forma_del_criterio(
                  formato, _ta_p.vocabulario_de(tipo_asunto or "amparo_directo")["combate_singular"],
                  _ta_p.vocabulario_de(tipo_asunto or "amparo_directo")["parte"],
                  variante=variante)
    # EL ORDEN DE ESTUDIO ES EL DE PRELACIÓN LÓGICA, no el de llegada: primero
    # el principal, del que dependen los demás. Un engrose que estudia un
    # accesorio antes que el problema del que depende obliga a rehacerlo.
    _ord = sorted(enumerate(criterios),
                  key=lambda x: (0 if (x[1].jerarquia or "").lower() == "principal"
                                 else 1, x[0]))
    # LA CUESTIÓN DECISIVA DEL PRINCIPAL (SPEC E3, AR 631/2025): la recurrida
    # planteó «¿alteró la cosa juzgada?» y lo que decide es si el adquirente
    # puede sustituirse en la ejecución; el estudio se escribía sobre la
    # primera. Va como DATO bajo el problema cuya pregunta es la que se
    # formuló, y sólo bajo ése: si el secretario cambió de principal o
    # reescribió su pregunta, ya no es la cuestión de ese problema.
    _dec_lineas, _dec_en = [], None
    try:
        import pregunta_decisiva as _pd_c
        if _pd_c.util(decisiva):
            _rec = _pd_c._plano(decisiva.get("pregunta_recurrida"))
            _dec_en = next((id(c) for _, c in _ord if _rec and _pd_c._plano(c.problema) == _rec),
                           None)
            _dec_lineas = _pd_c.lineas_criterio(decisiva) if _dec_en is not None else []
    except Exception:
        _dec_lineas, _dec_en = [], None
    for i, (_, c) in enumerate(_ord, 1):
        _g = str(getattr(c, "grupo", "") or "").strip()
        lineas.append(f"{i}. [{(c.jerarquia or 'accesorio').upper()}]"
                      f"{f' [GRUPO {_g}]' if _g else ''} {c.problema}")
        if _dec_lineas and id(c) == _dec_en:
            lineas.extend(_dec_lineas)
        # SIN CALIFICAR TRAS EL CAMBIO DE SENTIDO (26-sep-2026): lo dice el
        # bloque de datos que sigue (`_bloque_sin_calificar`). Sólo desde la
        # v2; la v1 está congelada. Conserva el puente con su concepto (CUBRE),
        # su grupo y la razón que él haya escrito —revisión adversarial: sin
        # CUBRE el estudio no sabía qué concepto contestaba, y su razón se
        # perdía—; lo demás (innecesario, caída, corriente del acervo) supone
        # un sentido que no tiene.
        # LA v1 TAMBIÉN (integración, 26-sep-2026): es la de producción para
        # las cuentas que no son de casa, y con el sentido vacío imprimía
        # «SENTIDO: » a secas. Un sentido vacío sólo llega aquí cuando el árbol
        # lo tumbó para recalificar y la recalificación no llegó (main.py filtra
        # los demás), así que la v1 de siempre no cambia ni una coma.
        _sin_calif = not str(c.sentido or "").strip()
        lineas.append("   SENTIDO: SIN CALIFICAR (ver «PLANTEAMIENTOS SIN CALIFICAR»)" if _sin_calif
                      else f"   SENTIDO: {c.sentido.upper()}")
        # QUÉ PLANTEAMIENTOS CALIFICA. Es el puente entre el problema, que
        # decide, y el concepto, que es lo que se escribe —en la estándar como
        # apertura del apartado; en la moderna, nombrado en la respuesta—.
        _cub = _fs_c.cubre_de(c, problemas or [])
        if _cub:
            lineas.append("   CUBRE: " + _fs_c.cubre_en_texto(
                _cub, _ta_p.vocabulario_de(tipo_asunto or "amparo_directo")["combate_singular"]))
        # EL GRUPO LO DECIDE EL SECRETARIO. La arquitectura ya prohíbe resolver
        # dos planteamientos con una calificación conjunta «salvo que declares
        # que se estudian juntos y por qué»; faltaba quién lo declarara.
        if _g:
            # EL GRUPO NO BORRA LA RESPUESTA DE CADA UNO. «No los contestes
            # por separado» chocaba con la regla nueva —cada argumento, una
            # respuesta identificable— y un argumento con dato propio metido en
            # la respuesta común es el que luego falta.
            # LA v1 RECIBE EL MISMO TEXTO (revisión adversarial de la
            # integración, 26-sep-2026): «Estudiar juntos» ya se ve en todas
            # las cuentas y el grupo viaja en los tres modos, así que la orden
            # de la v1 —«no los contestes por separado»— llegaba a producción
            # contra lo que promete la pantalla (inexhaustividad, art. 74, fr.
            # II). Sólo cambia cuando hay grupo: la v1 sin grupo, ni una coma.
            lineas.append(f"   SE ESTUDIA JUNTO CON LOS DEMÁS DEL GRUPO {_g}: "
                          f"un solo apartado que ABRE diciendo QUÉ LOS UNE —la "
                          f"consideración que atacan y la razón que los "
                          f"decide— con una calificación conjunta. La premisa "
                          f"común se expone una vez; dentro, cada argumento "
                          f"recibe una respuesta identificable, y el que trae "
                          f"un dato propio, la suya.")
        if _sin_calif:
            if c.razonamiento:
                lineas.append(f"   RAZÓN DEL SECRETARIO: {c.razonamiento}")
            lineas.append("")
            continue
        # LA PROCESAL QUE NO SE ESTUDIA POR MAYOR BENEFICIO NO «QUEDÓ SIN
        # MATERIA» (26-sep-2026, decisión 1 de David). Los arts. 74-V y 174
        # mandan decidir todas las violaciones procesales; la única salida es
        # que una concesión de fondo dé mayor beneficio (art. 189), y eso es
        # lo que la sentencia tiene que decir —con esa razón—, no la fórmula de
        # la sustracción de materia. El árbol deja el 189 en el razonamiento.
        if ((c.sentido or "").lower() == "innecesario"
                and "189" in str(getattr(c, "razonamiento", "") or "")):
            lineas.append("   NO SE ESTUDIA POR MAYOR BENEFICIO: la concesión de "
                          "fondo da a quien promueve más de lo que le daría la "
                          "reposición. Se dice en una o dos frases, con esa razón "
                          "y el artículo 189 de la Ley de Amparo; no se escribe "
                          "que quedó sin materia ni se contesta su fondo.")
        elif (c.sentido or "").lower() == "innecesario" and _v2c:
            # LOS EFECTOS RESPALDAN LA DECLARACIÓN (26-sep-2026, banco del
            # localizador: 174/2026 y 43/2025 declaraban innecesarios la doble
            # jornada o la pericial en informática y los efectos no los
            # nombraban; la responsable podía resolver igual sin desacatar).
            lineas.append("   NO SE ESTUDIA: quedó sin materia por el sentido "
                          "del principal. Una o dos frases que lo declaran "
                          "innecesario y dicen por qué; su fondo no se "
                          "contesta. Si lo que lo deja sin materia es que la "
                          "responsable tendrá que volver a resolver lo que "
                          "combate, sus argumentos con dato propio se nombran "
                          "en los EFECTOS entre lo que deberá examinar.")
        elif (c.sentido or "").lower() == "innecesario":
            lineas.append("   NO SE ESTUDIA: quedó sin materia por el sentido "
                          "del principal. Se dice en una frase y se pasa; ni "
                          "lo califiques ni lo contestes.")
        # CAE CON EL PRINCIPAL. El árbol de decisión lo marcó inoperante
        # porque descansa en la premisa que el principal desestimó: se dice
        # eso, con la premisa, y no se entra al fondo. ADC 93/2026: el estudio
        # escribía «inoperante» y debajo ordenaba lo que el motor había
        # propuesto para el sentido contrario.
        elif str(getattr(c, "razonamiento", "") or "").startswith(
                "Descansa en la premisa que se desestimó"):
            lineas.append("   CAE CON EL PRINCIPAL: se declara en UN párrafo que "
                          "ABRE CON LA RESPUESTA («No. …», «Es inoperante, "
                          "porque…»), dice qué premisa se desestimó y por qué su "
                          "estudio no produciría ningún fin práctico. NO se "
                          "estudia de fondo, NO se citan tesis de apoyo —para "
                          "decir que una premisa cayó no hace falta ninguna—, NO "
                          "se ordena nada a la responsable sobre él, y NO se le "
                          "aplica ninguna suerte que el motor hubiera escrito "
                          "para el sentido contrario.")
        # LA CORRIENTE DEL ACERVO. Si el sentido va CONTRA lo que hicieron los
        # demás tribunales sobre el mismo tema, el estudio tiene que hacerse
        # cargo de la objeción: apartarse del criterio mayoritario se puede
        # —para eso existe la contradicción de tesis— pero no en silencio.
        if c.prediccion and c.prediccion.get("frase"):
            _p = c.prediccion
            # «CONCEDE» Y «FUNDADO» SON LA MISMA DIRECCIÓN. El acervo guarda el
            # sentido del FALLO —concede, niega, confirma, revoca— y aquí se
            # califica el PLANTEAMIENTO —fundado, infundado—. Comparándolos
            # como cadenas, la alarma de «va contra la corriente» saltaba
            # siempre, y una alarma que salta siempre deja de leerse.
            _va = _misma_direccion(_p.get("sentido", ""), c.sentido)
            lineas.append(f"   EL ACERVO: {_p['frase']}")
            lineas.append("   (dato interno para calibrar: NO lo escribas en "
                          "la sentencia)")
            if not _va and _p.get("confianza") in ("alta", "media"):
                lineas.append(
                    "   ⚠ EL SENTIDO VA CONTRA LA CORRIENTE del acervo. No lo "
                    "cambies: hazte cargo. Enuncia la razón contraria —la de "
                    "quienes resolvieron al revés— y explica por qué aquí no "
                    "aplica. Un estudio que se aparta sin decirlo se cae.")
        if c.razonamiento:
            lineas.append(f"   RAZÓN DEL SECRETARIO: {c.razonamiento}")
        elif (c.sentido or "").lower() == "innecesario":
            pass          # su razón es el sentido del principal, ya dicha
        else:
            # Sin razón escrita, el modelo tiene que suponerla, y ahí es donde
            # el proyecto deja de parecerse a lo que el secretario pensaba.
            lineas.append("   (sin razón escrita: constrúyela con el material y "
                          "señálalo en las advertencias)")
        lineas.append("")
    lineas += [
        "",
        "LA CIFRA DEL ACERVO NO SE ESCRIBE. Es un dato para que TÚ calibres el",
        "peso de la objeción, no un argumento: una sentencia que dice «el 82% de",
        "los tribunales concede» es impublicable, porque el criterio no se vota.",
        "Lo que sí se escribe —cuando el sentido va contra la corriente— es la",
        "RAZÓN de quienes resolvieron al revés y por qué aquí no aplica. Nunca el",
        "porcentaje, nunca el número de sentencias, nunca «la mayoría de los",
        "tribunales».",
        "",
        "SI ENCUENTRAS UN OBSTÁCULO SERIO para ese sentido —una jurisprudencia",
        "obligatoria en contra, una causal de improcedencia—, NO cambies el",
        "sentido: dilo en el apartado ADVERTENCIAS para que él lo valore.",
    ]
    # ── EL CIERRE QUE NO REMITE, SINO QUE DICE ─────────────────────────────
    # Lara Chagoyán, § 3.4, coherencia interna: «los reenvíos que se hacen de
    # una a otra parte de la sentencia deben quedar claros, especialmente a la
    # hora de los puntos resolutivos. Es más conveniente especificar el QUÉ,
    # POR QUÉ y PARA QUÉ se decidió en un determinado sentido que extraviar al
    # lector para que encuentre por él mismo el significado último del
    # resolutivo. A veces no se entiende que después de haber repetido cierta
    # información varias veces en la sentencia, se termine con un mensaje
    # lacónico, por no decir críptico a la hora de los puntos resolutivos».
    #
    # Nuestro resolutivo dice «por los motivos y fundamentos expuestos en el
    # último considerando», que es exactamente ese mensaje críptico.
    #
    # EN LA v2 ESTO NO VA AQUÍ (David, 26-sep-2026, decisión 3, opción b: «sin
    # cierre por defecto; un cierre breve sólo cuando hay tres o más apartados
    # con resultados distintos»). Este bloque mandaba TRES frases y el final
    # del prompt mandaba recapitular «con sustancia»: dos órdenes para el
    # mismo párrafo, y ninguna decía cuándo no hacía falta. La regla única
    # va al final del prompt, calculada por `_cierre_permitido`.
    if not _v2c:
        lineas += ["",
                   "── CÓMO TERMINA EL ESTUDIO ──",
                   "El último párrafo, antes del resolutivo, dice en TRES frases:",
                   "  · QUÉ se decide —confirmar, revocar, conceder, negar—;",
                   "  · POR QUÉ, en una frase que resuma la razón que lo sostiene, no",
                   "    una remisión a otro apartado;",
                   "  · PARA QUÉ, es decir, qué tiene que hacer ahora la autoridad, si",
                   "    es que tiene que hacer algo.",
                   "Está PROHIBIDO cerrar con «por las razones expuestas», «en las",
                   "relatadas consideraciones» o cualquier fórmula que mande al lector",
                   "a buscar por su cuenta lo que ya se dijo.", ""]
    # ── LOS EFECTOS, CON RÓTULO FIJO Y COMO ÓRDENES ──────────────────────
    # ADC 93/2026 v5: el estudio escribió los efectos completos —«La concesión
    # del amparo exige que la Sala: a) deje insubsistente…; b) deje sin
    # efectos la preclusión…; c) admita la ampliación…»— y el compositor no
    # los reconoció (esperaba «debe producir los efectos»), así que los dejó
    # dentro del estudio y en el considerando de EFECTOS puso la fórmula
    # genérica «dicte otra en la que atienda los lineamientos». David: «al
    # tratarse de una concesión por violación al procedimiento, la Sala no
    # puede dictar otra sentencia de inmediato». Un rótulo fijo es lo que
    # permite recogerlos sin adivinar.
    import tipos_asunto as _ta_ef
    _concede_ef = any(_ta_ef.prospera(str(getattr(c, "sentido", "") or ""))
                      for c in criterios)
    # LOS EFECTOS SON DEL AMPARO (27-sep-2026). El bloque se pedía en los cuatro
    # tipos en cuanto un planteamiento prosperaba, y en una queja o una
    # revisión fiscal fundada no hay concesión que tenga efectos: el modelo los
    # escribía y el documento los tiraba en silencio. En la revisión de amparo
    # sólo los hay cuando es ESTE tribunal el que concede o modifica los efectos
    # de la concesión; si confirma, niega, sobresee o repone, no.
    _tipo_ef = _ta_ef.normalizar(tipo_asunto) or "amparo_directo"
    _cuando_ef = []
    if _tipo_ef in ("queja", "revision_fiscal"):
        _concede_ef = False
    elif _tipo_ef == "amparo_revision":
        _cuando_ef = ["SÓLO SI, al resolver esta revisión, ESTE tribunal concede el",
                      "amparo —revoca la negativa o el sobreseimiento y concede— o",
                      "modifica los efectos de la concesión. Si confirma, niega,",
                      "sobresee o repone el procedimiento, no escribas efectos."]
    # LOS EFECTOS: DE TRES A CINCO, BREVES Y EN INFINITIVO (David, 27-sep-2026:
    # «los efectos no están bien configurados, son muy extensos… inicia con
    # "Efectos. 1. Deje insubsistente…" cuando debería decir "Efectos. Con
    # fundamento en el artículo … de la Ley de Amparo, la autoridad responsable
    # deberá: …", con efectos más reducidos pero que comprendan el objetivo de
    # la concesión. Regularmente los efectos son 3 a máximo 5»). La apertura la
    # escribe el documento (`tipos_asunto.APERTURA_EFECTOS`); aquí sólo las
    # órdenes, y por eso en infinitivo: cuelgan de «deberá:». Medido en los 642,
    # 103 y 93/2026 del 26-sep: cinco o seis órdenes de 40 a 75 palabras que
    # volvían a argumentar el estudio.
    #
    # EN LA v2, LOS EFECTOS SE DESCRIBEN (fila 14b de la propuesta). La lista
    # «1. Deje insubsistente…; 2. Deje sin efectos…; 3. Admita…» es un ejemplo
    # con la forma exacta de una sentencia, y un ejemplo así se firma literal
    # —ya pasó con cuatro moldes de este mismo prompt—. Lo que sigue dice qué
    # tiene que tener cada orden, no cómo se escribe. El rótulo se queda: es el
    # que permite al compositor recogerlos sin adivinar.
    _letra_ef = {2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis"}
    _max_ef = _letra_ef.get(_ta_ef.EFECTOS_MAX, str(_ta_ef.EFECTOS_MAX))
    _n_ef = (f"{_letra_ef.get(_ta_ef.EFECTOS_MIN, str(_ta_ef.EFECTOS_MIN))} A "
             f"{_max_ef}").upper()
    if _concede_ef and _v2c:
        lineas += ["",
                   "── LOS EFECTOS, AL FINAL Y CON ESTE RÓTULO ──"] + _cuando_ef + [
                   "Después del último párrafo del estudio escribe, en su propia línea",
                   "y sin nada más, el rótulo:",
                   "",
                   "    EFECTOS DE LA CONCESIÓN",
                   "",
                   "y debajo las órdenes a la responsable como LISTA NUMERADA, una por",
                   "párrafo, SIN frase de introducción: el documento abre el considerando",
                   "con el fundamento —el artículo 77 de la Ley de Amparo— y con quien debe",
                   "cumplir —la autoridad responsable—, seguido de la palabra deberá y dos",
                   "puntos.",
                   f"DE {_n_ef} ÓRDENES según la complejidad del asunto; nunca más de",
                   f"{_max_ef}. Juntas comprenden el objetivo de la concesión: qué deja",
                   "insubsistente la responsable, qué emite o repone en su lugar, qué tiene",
                   "que decidir de nuevo y con qué alcance, y que lo demás lo resuelve con",
                   "plenitud de jurisdicción. Lo que recae sobre el mismo acto va en UNA orden.",
                   "Cada orden es UNA oración breve —de ordinario no más de cuarenta",
                   "palabras— que EMPIEZA CON EL VERBO EN INFINITIVO, porque cuelga de esa",
                   "palabra; dice sobre qué acto o actuación recae y se puede verificar",
                   "en la ejecución sin interpretarla. Dice QUÉ hacer, no por qué: no",
                   "repite las razones del estudio ni enumera pruebas una por una.",
                   "Ninguna remite a «los lineamientos de esta ejecutoria» en lugar de decir",
                   "qué hay que hacer. Sin prosa entre ellas. Los efectos se escriben SÓLO",
                   "aquí: el estudio no los adelanta.",
                   "Cuando la concesión deja en manos de la responsable argumentos que el",
                   "estudio no contestó, una de esas órdenes dice cuáles son —una sola,",
                   "aunque sean varios—, cada uno con su dato en pocas palabras: es lo que",
                   "la obliga a examinarlos al volver a resolver.",
                   "SI LA CONCESIÓN ES POR UNA VIOLACIÓN PROCESAL, la responsable NO",
                   "puede dictar otra sentencia de inmediato: las órdenes disponen la",
                   "REPOSICIÓN en el orden en que ha de cumplirse —qué se deja",
                   "insubsistente; qué actuación viciada y qué resolución que la confirmó",
                   "se dejan sin efectos; qué se admite, se practica o se ordena en su",
                   "lugar y cómo sigue el trámite; y sólo al final, cerrada de nuevo la",
                   "instrucción, el dictado de la sentencia de fondo con plenitud de",
                   f"jurisdicción—, agrupando los pasos para no pasar de {_max_ef}.", ""]
    elif _concede_ef:
        # LA v1 ESTÁ CONGELADA (test_prompt_v2.py): sus efectos siguen como
        # estaban y el documento los pasa a infinitivo al componerlos
        # (`documento_generado.componer_efectos`). Sólo se le dice CUÁNDO en la
        # revisión, que antes los pedía también cuando se negaba.
        # SIN ALZADA (30-sep-2026, AD 323/2025): el segundo paso de la
        # reposición dejaba sin efectos «la resolución del recurso ordinario»,
        # que en única instancia no existe. Sólo si la instancia consta como
        # única (`tipos_asunto.unica_instancia`); si no, la v1 congelada.
        _rep_v1 = ["efectos la resolución del recurso ordinario y la actuación viciada;"]
        if _ta_ef.unica_instancia(tipo_asunto):
            _rep_v1 = ["efectos la actuación viciada y, si se combatió durante el juicio, la",
                       "resolución que la confirmó;"]
        lineas += ["",
                   "── LOS EFECTOS, AL FINAL Y CON ESTE RÓTULO ──"] + _cuando_ef + [
                   "Después del último párrafo del estudio escribe, en su propia línea",
                   "y sin nada más, el rótulo:",
                   "",
                   "    EFECTOS DE LA CONCESIÓN",
                   "",
                   "y debajo las órdenes a la responsable como LISTA NUMERADA, una por",
                   "párrafo, en imperativo y cada una verificable en la ejecución:",
                   "«1. Deje insubsistente…»; «2. Deje sin efectos…»; «3. Admita…»;",
                   "«4. Corra traslado…»; «5. Cerrada la instrucción, dicte…». Sin",
                   "prosa entre ellas y sin la fórmula «dicte otra en la que atienda los",
                   "lineamientos de esta ejecutoria», que no se puede ejecutar sin",
                   "interpretarla.",
                   "SI LA CONCESIÓN ES POR UNA VIOLACIÓN PROCESAL, la responsable NO",
                   "puede dictar otra sentencia de inmediato: los efectos ordenan la",
                   "REPOSICIÓN paso a paso —dejar insubsistente la sentencia; dejar sin"] + _rep_v1 + [
                   "admitir la ampliación / la prueba / emplazar, según sea el caso;",
                   "correr traslado a la contraparte; desahogar lo que proceda y abrir",
                   "alegatos; y, cerrada de nuevo la instrucción, dictar la sentencia de",
                   "fondo con plenitud de jurisdicción—.", ""]

    lineas += _aviso_de_suplencia(criterios, materia, material_texto, tipo_asunto)
    return "\n".join(lineas)


def _texto_de(m) -> str:
    """Todo lo que se sabe del asunto, en un solo hilo, para poder mirarlo."""
    partes = []
    for campo in ("resumen", "hechos", "antecedentes", "agravios", "conceptos",
                  "acto_reclamado", "tema", "materia", "resumen_sentencia"):
        v = getattr(m, campo, None)
        if isinstance(v, str):
            partes.append(v)
        elif isinstance(v, (list, tuple)):
            partes += [str(x) for x in v]
    return " ".join(partes)[:400000]


# ═══ LOS PRECEPTOS QUE NOMBRA LA PARTE, AUNQUE LLEGARAN AL FINAL ════════════
# Humo del AR 631/2025 (29-sep-2026): el artículo 49 del código procesal de
# Querétaro y los 2284 y 2294 del civil —la ley que gobierna el asunto, y los
# que invoca la recurrente— estaban en el material, pero en los lugares 43 a 47;
# el corte de doce los dejaba fuera y el estudio escribió que «no se encuentran
# entre las normas aportadas», mientras las doce primeras traían artículos del
# Código Civil Federal que no venían al caso. SUMA, NO SUSTITUYE (la misma regla
# que `fase5_propuesta._bloque_normas`): las doce de siempre se quedan y, detrás,
# hasta seis de las que el inventario de argumentos NOMBRA por su artículo y su
# ley, sin importar cómo llegaron al material. Bandera «normas_al_documento».
NORMAS_NOMBRADAS_EXTRA = 6
_GENERICAS_LEY = {"codigo", "ley", "articulo", "para", "estado", "estados", "unidos", "mexicanos", "general"}


def _voces_ley(x) -> set:
    import documento_generado as _dg_v
    try:
        t = _dg_v.canonizar_ley(str(x or ""))
    except Exception:                                   # pragma: no cover
        t = str(x or "")
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    out = set()
    for w in re.findall(r"[a-z]+", t):
        if w in _VACIAS or w in _GENERICAS_LEY or len(w) < 3:
            continue
        out.add(re.sub(r"(es|s)$", "", w) if len(w) > 5 else w)
    return out


def _normas_para_el_estudio(m) -> list:
    """Las normas que entran al prompt del estudio (ver arriba). Nunca lanza:
    ante cualquier sorpresa, las doce de siempre."""
    normas = list(getattr(m, "normas", None) or [])
    base = normas[:MAX_NORMAS_PROMPT]
    try:
        import contexto_taller as _ct_n
        if not _ct_n.rediseno("normas_al_documento") or len(normas) <= MAX_NORMAS_PROMPT:
            return base
        texto = "\n".join(f"{s.get('texto') or ''} {s.get('cita') or ''}"
                           for s in (getattr(m, "inventario", None) or []) if isinstance(s, dict))
        if not texto.strip():
            return base
        extra, vistos = [], set()
        for art, cola in citas_de_articulos(texto):
            vc = _voces_ley(cola)
            if not vc:
                continue
            mejor, puntos = None, 0
            for i, n in enumerate(normas[MAX_NORMAS_PROMPT:], MAX_NORMAS_PROMPT):
                if not isinstance(n, dict) or str(n.get("articulo") or "").strip() != str(art).strip():
                    continue
                pt = len(vc & _voces_ley(n.get("cuerpo_legal") or ""))
                if pt > puntos:
                    mejor, puntos = i, pt
            if mejor is not None and mejor not in vistos and puntos >= min(2, len(vc)):
                vistos.add(mejor)
                extra.append(normas[mejor])
        return base + extra[:NORMAS_NOMBRADAS_EXTRA]
    except Exception:
        return base


def _bloque_material(m: Material) -> str:
    p = ["", "═" * 71, "MATERIAL PARA FUNDAR", "═" * 71]
    # Las obligatorias primero: ya vienen ordenadas, así que el recorte se lleva
    # las orientadoras del final, que es lo que sobra.
    # ═══════════════════════════════════════════════════════════════════════
    # LAS TESIS DE LA TÉCNICA NO COMPITEN POR EL CUPO
    # ═══════════════════════════════════════════════════════════════════════
    # El tope son diez y el acervo devuelve cuarenta, así que `[:10]` es un
    # recorte de verdad. Las tesis de la técnica se añadían al FINAL de la
    # lista y el recorte se las llevaba SIEMPRE: en la revisión fiscal 91/2025
    # se trajeron las cuatro del reenvío —consta en los registros— y ninguna
    # llegó al prompt. Dos intentos de arreglarlo tocaron el texto de la
    # instrucción, que no era el problema: la instrucción prometía unas tesis
    # que no estaban.
    #
    # No se sube el tope: las del fondo siguen siendo diez, y las de la técnica
    # van aparte porque responden a otra pregunta.
    _tec = [t for t in (m.tesis or []) if t.get("tecnica")]
    # LAS DE LA FIGURA TAMPOCO (revisión adversarial de la fase E, 28-sep-2026):
    # las que sólo trajo la búsqueda sobre la cuestión decisiva
    # (`fase6_rag.sumar_figura`, marca `cupo_figura`) entraban DELANTE y se
    # quedaban hasta con ocho de las diez plazas; las obligatorias de los
    # demás problemas salían del prompt. Van aparte, con su tope pequeño. Las
    # que ya traía la búsqueda de un problema siguen en su sitio.
    _fig = [t for t in (m.tesis or [])
            if t.get("cupo_figura") and not t.get("tecnica")][:MAX_TESIS_FIGURA_PROMPT]
    tesis = [t for t in (m.tesis or [])
             if not t.get("tecnica") and not t.get("cupo_figura")][:MAX_TESIS_PROMPT] \
        + _fig + _tec
    normas = _normas_para_el_estudio(m)
    if tesis:
        import fuerza_juridica as _fj
        _fj.anotar(tesis, str(getattr(m, "tribunal", "") or ""))
        p.append("\nTESIS Y JURISPRUDENCIA (existen: salen del acervo, no de tu memoria).")
        p.append("  La OBLIGATORIA vincula a este Tribunal y se invoca como razón que")
        p.append("  decide; la ORIENTADORA sólo ilustra y se cita como apoyo. Tratarlas")
        if _fj.activa():
            p.append("  igual es un error de fondo, no de estilo. La fuerza viene calculada")
            p.append("  para ESTE tribunal (art. 217 LA): la jurisprudencia de otro")
            p.append("  colegiado le ORIENTA, no lo obliga. Y que un criterio obligue no")
            p.append("  dice que su regla gobierne este caso: aplícalo sólo si los hechos")
            p.append("  satisfacen sus condiciones; si no, distínguelo.")
        else:
            p.append("  igual es un error de fondo, no de estilo.")
        p.append("  PREFIERE LA SUPREMA CORTE. Entre dos criterios que sirven igual, se")
        p.append("  cita el del Pleno o de una Sala antes que el de un Tribunal")
        p.append("  Colegiado: pesa más y evita el reproche de haberse quedado corto.")
        p.append("  Vienen ordenados: los primeros son los que más aplican.")
        p.append("  SI UN CRITERIO LLEVA «LO CITAN N SENTENCIAS DE ESTE")
        p.append("  CIRCUITO», ése es el que el tribunal aplica de manera")
        p.append("  constante para esta cuestión: preferirlo no es una")
        p.append("  sugerencia de estilo, es escribir la sentencia que este")
        p.append("  circuito reconoce como suya. Y puedes decirlo —«criterio")
        p.append("  reiterado de este circuito»— cuando de verdad lo sea.")
        for t in tesis:
            # DOS DATOS DISTINTOS, Y YO LOS TENÍA FUNDIDOS EN UNO. Que un
            # criterio VINCULE y que SEA jurisprudencia no es lo mismo: hay
            # tesis aisladas de la Corte que orientan y jurisprudencia de
            # colegiado que obliga en su circuito. La etiqueta decía sólo lo
            # primero, así que el modelo no tenía cómo saber que estaba citando
            # una tesis aislada —y la llamó jurisprudencia—.
            fuerza = _fj.rotulo(t)
            tipo = str(t.get("tipo") or "").strip() or "tipo no declarado"
            # CUÁNTAS VECES LO CITA EL CIRCUITO PARA ESTA CUESTIÓN. Es un
            # dato que la búsqueda por parecido no puede dar: sale de contar
            # qué invocaron las sentencias que ya resolvieron lo mismo. Y es
            # argumento, no adorno —«el criterio que este circuito aplica de
            # manera constante» pesa distinto que «una tesis que encontré»—.
            _vec = int(t.get("veces") or 0)
            _uso = (f" — LO CITAN {_vec} SENTENCIAS DE ESTE CIRCUITO al "
                    f"resolver esta misma cuestión") if _vec >= 2 else (
                   " — usado por el circuito en esta cuestión" if _vec else "")
            # DE LA LÍNEA DE LA CORTE, buscada en internet y confirmada en el
            # acervo por registro y por rubro (ver `fase_internet`). Se dice:
            # es la evolución del criterio, y un estudio que resuelve con la
            # tesis de 2018 ignorando la de 2024 sale desactualizado aunque
            # cite bien.
            _linea = (" — DE LA LÍNEA DE LA SUPREMA CORTE SOBRE ESTE PROBLEMA: "
                      "úsala, es la evolución del criterio") if t.get("de_internet") else ""
            # DEL MÉTODO (24-sep-2026): dice CÓMO se interpreta, no QUÉ se
            # resuelve. Sin la marca el redactor la contaba entre las del caso.
            if t.get("metodo"):
                _linea = (" — CRITERIO DE MÉTODO: dice CÓMO interpretar; se cita "
                          "sólo en el peldaño del diálogo constitucional")
            # EL SELLO DE VIGENCIA (25-sep-2026). Una jurisprudencia
            # abandonada ya no obliga, y la co-citación del circuito premia
            # justo a los criterios viejos y muy citados: sin la marca, «LO
            # CITAN 9 SENTENCIAS» empujaba a fundar en una tesis sin vigencia.
            _v = t.get("vigencia")
            if _v:
                # LA MISMA DEFINICIÓN QUE LA TARJETA Y LA DELIBERACIÓN
                # (fuerza_juridica, rediseño punto 3): «aclarada» o de «texto
                # sustituido» no pierden vigencia; se dice qué cambió.
                if (_fj.sello_perdio_vigencia(_v) if _fj.activa()
                        else not _v.get("parcial")):
                    fuerza = "SIN VIGENCIA"
                    _linea += (f" — ⚠️ {_vig.etiqueta(_v)}: no la invoques como vigente"
                               + (f"; funda en la {_v['por_clave']}" if _v.get("por_clave") else ""))
                elif _fj.activa():
                    _linea += (f" — ⚠️ {_vig.etiqueta(_v)}"
                               + ("" if not _v.get("parcial") else
                                  ": perdió vigencia sólo EN PARTE; invócala sólo en lo que sigue vigente")
                               + (f"; ver la {_v['por_clave']}" if _v.get("por_clave") else ""))
                else:
                    # Sin la bandera, exactamente como antes.
                    _linea += (f" — ⚠️ {_vig.etiqueta(_v)}: no la invoques como vigente"
                               + (f"; funda en la {_v['por_clave']}" if _v.get("por_clave") else ""))
            p.append(f"\n  · [{fuerza}] [{tipo}] Registro "
                     f"{t.get('registro','')} — {t.get('instancia','')}{_uso}{_linea}")
            p.append(f"    {t.get('rubro','')}")
            if t.get("localizacion"):
                p.append(f"    {t['localizacion']}")
            # ENTERA. Con 900 caracteres el criterio quedaba cortado y el
            # modelo razonaba desde el rubro: así le hizo decir a la tesis
            # 182597 LO CONTRARIO de lo que sostiene, y era el único punto
            # donde la quejosa tenía apoyo. El rubro es un título, no la regla.
            p.append(f"    {(t.get('texto') or '')[:TESIS_CARACTERES]}")
    # LOS PRINCIPIOS DEL CIRCUITO, antes de los preceptos: son el marco con el
    # que se razona, y van delante de lo que se cita.
    if getattr(m, "principios", None):
        p.append("\n\nCON QUÉ RAZONA ESTE CIRCUITO LA CUESTIÓN.")
        p.append("  Son las nociones que las sentencias del acervo emplean al")
        p.append("  resolver este mismo punto. No las cites como si fueran")
        p.append("  fuente: son el ARMAZÓN del razonamiento, y un estudio que")
        p.append("  las ignora suena ajeno a como resuelve este tribunal.")
        for _pr in m.principios[:6]:
            p.append(f"    · {_pr}")

    if normas:
        p.append("\n\nPRECEPTOS:")
        for n in normas:
            p.append(f"\n  · {n.get('cuerpo_legal','')} — Art. {n.get('articulo','')}")
            # ENTERO. Con 700 caracteres el artículo 47 de la Ley Federal del
            # Trabajo llegaba cortado en la fracción I y la fracción X —«sin
            # causa justificada», que era la bisagra del asunto— empieza en el
            # 2,348. Es el mismo fallo que ya tuvieron las tesis, con la misma
            # consecuencia: razonar desde el encabezado.
            p.append(f"    {(n.get('texto') or '')[:NORMA_CARACTERES]}")
    if m.convencional:
        p.append("\n\nCONVENCIONAL:")
        for c in m.convencional:
            p.append(f"\n  · {c.get('rubro','')}\n    {(c.get('texto') or '')[:600]}")
    if m.moldes:
        p.append("\n\nESTUDIOS DE FONDO — SÓLO COMO MODELO DE REDACCIÓN:")
        p.append("  Imita su prosa y su orden. PROHIBIDO citarlos como fundamento:")
        p.append("  no son jurisprudencia. Para fundar, usa las tesis y los preceptos.")
        for e in m.moldes:
            p.append(f"\n  · {e.get('tribunal','')} · {e.get('expediente','')}")
            p.append(f"    {(e.get('holding') or '')[:800]}")
    return "\n".join(p)


def _recorte_limpio(x: str, tope: int) -> str:
    """El mismo corte por frontera que usan las fases de lectura."""
    try:
        from fases123_pipeline import _cortar_bien
        return _cortar_bien(x or "", tope)
    except Exception:
        return (x or "")[:tope]


def _bloque_constancias(propuesta_global, contexto: str,
                        criterios: list | None = None) -> str:
    """Las constancias que la propuesta pidió ver: cuáles llegaron y cuáles
    no, con la orden de no suponer las que faltan.

    ═══ CUANDO EL SECRETARIO RESOLVIÓ AL REVÉS, ESAS CONSTANCIAS NO MANDAN ═══
    (23-sep-2026, revisión 711/2025). El motor propuso INFUNDADO y, para su
    vía, listó tres constancias «indispensables». David resolvió FUNDADO con
    su propio criterio: bastaba lo dicho por el juez de distrito sobre el
    alcance del oficio de bloqueo. Este bloque siguió entrando tal cual, con
    la orden «lo que dependa de ellas se dice como NO ACREDITADO», y el
    estudio obedeció a las dos órdenes a la vez: escribió que sin esas
    constancias no podía validarse el bloqueo… y al pie declaró fundado el
    recurso y negó el amparo. David: «una terrible falta de congruencia
    interna».

    Las constancias que el motor echa en falta son las que NECESITARÍA SU
    razonamiento. Cuando el secretario elige otro camino, pueden no hacer
    falta en absoluto. Aquí se distingue: si resolvió al revés, se le dice al
    estudio qué pidió el motor y por qué ya no gobierna, y se le prohíbe
    convertir esa ausencia en la conclusión."""
    if not isinstance(propuesta_global, dict):
        return ""
    pedidas = propuesta_global.get("constancias") or []
    if not pedidas:
        return ""
    _pral = next((c for c in (criterios or [])
                  if str(getattr(c, "jerarquia", "")).lower() == "principal"),
                 (criterios or [None])[0])
    _al_reves = bool(_pral is not None and propuesta_global.get("sentido")
                     and not _misma_direccion(str(propuesta_global.get("sentido") or ""),
                                              str(getattr(_pral, "sentido", "") or "")))
    try:
        import constancias as _cn
        if not _al_reves:
            return _cn.bloque_para_estudio(pedidas, contexto or "")
        falt = _cn.faltantes(pedidas, contexto or "")
        if not falt:
            return _cn.bloque_para_estudio(pedidas, contexto or "")
        lineas = ["", "═" * 71,
                  "CONSTANCIAS QUE EL MOTOR PIDIÓ PARA UNA VÍA QUE NO SE TOMÓ",
                  "═" * 71,
                  "El motor habría resuelto "
                  + str(propuesta_global.get("sentido") or "").replace("_", " ")
                  + " y, para ESE razonamiento, quería ver:"]
        for c in falt:
            lineas.append(f"  · {c['que']}")
        lineas.append(
            "El secretario resolvió por otra vía, con su propio criterio —lo tienes "
            "arriba como RAZÓN DEL SECRETARIO—. Esas constancias servían al camino "
            "del motor, no al suyo. POR TANTO:\n"
            "- NO escribas que el bloqueo, el acto o el planteamiento «no puede "
            "validarse» ni que algo «no está acreditado» POR FALTAR ESTAS "
            "constancias. Eso sería resolver por la vía del motor y firmar por la "
            "del secretario: una sentencia que se contradice a sí misma.\n"
            "- Construye el estudio sobre la base del secretario y sobre lo que SÍ "
            "consta: lo que dijo el a quo, lo que dice el oficio, lo que reconoce la "
            "recurrente. Un hecho que obra en autos es prueba bastante para su vía.\n"
            "- Si, aun en la vía del secretario, un dato concreto te resultara de "
            "verdad necesario y no constara, NO lo conviertas en la conclusión: "
            "dilo en el apartado ADVERTENCIAS, con su nombre, para que quien firma "
            "lo busque en el expediente antes de listar.")
        return "\n".join(lineas) + "\n"
    except Exception:
        return ""


def _bloque_aportado(contexto: str) -> str:
    """Lo que el secretario subió porque el acervo no lo tenía, rotulado por
    lo que ES.

    ADC 93/2026 (22-sep-2026): la interlocutoria de la reclamación —la
    resolución que decidió la violación procesal— entraba aquí como
    «documento aportado», y el estudio la despachó en un párrafo tras razonar
    sobre la regla abstracta. Ahora `violacion_procesal.clasificar` dice qué
    llegó, y si es la resolución del incidente el bloque trae la técnica: sus
    razones son la razón toral: cada una se identifica y se confronta, y las
    que caen por la misma respuesta se contestan juntas, nombrándolas
    (técnica aprobada por David el 26-sep-2026, `tipos_asunto`).
    """
    c = (contexto or "").strip()
    if not c:
        return ""
    import violacion_procesal as _vp
    # POR PÁRRAFO, NO POR CARACTER. Estas constancias las subió el secretario
    # y alimentan prosa que se firma; cortarlas a mitad de frase es la misma
    # puerta por la que salió «el texto proporcionado se interrumpió» dentro
    # del considerando quinto.
    return _vp.bloque(c, para="estudio", tope=20000, recortar=_recorte_limpio)



# ═══════════════════════════════════════════════════════════════════════════
# CÓMO ESCRIBE UN TRIBUNAL QUE ESCRIBE BIEN — medido, no opinado
# ═══════════════════════════════════════════════════════════════════════════
# Cuatro agentes leyeron 966 sentencias de calidad alta y 980 de calidad media
# del acervo —laboral, administrativa, civil y penal, siete circuitos— y
# recuperaron el estudio de fondo completo de las mejores. El acervo se había
# puntuado a sí mismo; aquí sólo se midió la diferencia.
#
# LO QUE NO ERA: no es citar más, ni escribir más largo, ni invocar más derechos
# humanos. Hay estudios de calidad media de 225,000 caracteres y sentencias del
# 0.06% superior de cinco páginas. La extensión ACOMPAÑA a la calidad; no la
# produce. Y la doctrina es CONTRASEÑAL: 7% arriba contra 13% abajo.
#
# LO QUE ERA: poner por escrito las operaciones que normalmente se quedan en la
# cabeza del redactor. Derivar la regla en abstracto antes de aplicarla aparece
# en el 33% de las mejores civiles y en el 3.3% de las medias —la mayor
# diferencia relativa de todo el análisis—.

_ARQUITECTURA_COMUN = """
═══════════════════════════════════════════════════════════════════════
EL CUERPO NO TRANSCRIBE: EL TEXTO ÍNTEGRO VA A LA NOTA AL PIE
═══════════════════════════════════════════════════════════════════════
Esto no es una preferencia de estilo: es cómo queda maquetado el documento, y
si escribes contra ello el proyecto sale roto.

El documento baja SOLO a la nota al pie el texto íntegro de cada precepto que
identificas y de cada tesis larga. Tú no lo copias. Y como no hay transcripción
en el cuerpo, TODA FRASE QUE LA ANUNCIE O REMITA A ELLA QUEDA APUNTANDO A NADA:

  PROHIBIDO                        LO QUE SE ESCRIBE EN SU LUGAR
  «…establece lo siguiente.»       «El artículo 38 del Código Fiscal de la
  «…dispone lo siguiente.»          Federación exige que el acto notificado
  «…señala lo siguiente:»           conste por escrito y lleve la firma del
                                    funcionario competente.»
  «Del precepto transcrito…»       «Del artículo 38…», «De esa disposición…»
  «El criterio transcrito…»        «El criterio citado…», «Esa tesis…»
  «la transcripción que antecede»  «lo dispuesto en ese precepto»

Medido en la revisión fiscal 91/2025: el proyecto decía «El artículo 38 del
Código Fiscal de la Federación establece lo siguiente.» y debajo, en vez del
texto, empezaba otro párrafo. Y decía «Del precepto transcrito deriva que…»
sobre un precepto que sólo estaba al pie. Ese proyecto no se puede firmar.

LA REGLA, EN UNA LÍNEA: nombra el precepto o la tesis y DI LO QUE DICE, dentro
de tu propia frase. Quien firme comprobará el texto en la nota.

═══════════════════════════════════════════════════════════════════════
EN REVISIÓN SE REVOCA; SÓLO EN AMPARO SE DEJA INSUBSISTENTE
═══════════════════════════════════════════════════════════════════════
David: «en revisión la sentencia no se deja insubsistente, se revoca. Sólo en
amparo (cuando se concede) se ordena que se deje insubsistente el acto
reclamado».

No son dos maneras de decir lo mismo. Son dos figuras, y cada una vive en su
sitio:

  · REVISIÓN (o cualquier recurso). El tribunal es ÓRGANO REVISOR de esa misma
    sentencia y la REVOCA: con eso deja de existir. No hay a quién ordenarle
    que la deje insubsistente, porque ya no está. Si después hay reenvío, lo
    que se ordena es DICTAR OTRA —«dicte otra sentencia en la que se ocupe de
    los conceptos cuyo estudio omitió»—, no dejar insubsistente nada.

  · AMPARO (directo, o indirecto en revisión, cuando se CONCEDE). El tribunal
    NO revoca el acto reclamado: no es su superior jerárquico. Concede la
    protección y ORDENA a la autoridad responsable que lo deje insubsistente y
    dicte otro. Ahí sí, y por eso la fórmula existe.

NO ESCRIBAS «deje insubsistente la sentencia recurrida» EN UN RECURSO. Es la
fórmula del amparo colocada donde no cabe, y describe una potestad que este
tribunal no está ejerciendo.



═══════════════════════════════════════════════════════════════════════
CÓMO SE ESCRIBE ESTE ESTUDIO
═══════════════════════════════════════════════════════════════════════
Esto está medido sobre 1,946 sentencias del propio acervo, comparando las que
el corpus puntuó alto contra las que puntuó en la media. No son preferencias de
estilo: son las operaciones que separan a unas de otras.

1. TRANSCRIBE LA FUENTE, NO LA RESUMAS. Antes de aplicar un precepto,
   transcríbelo entre comillas: «El artículo N de [ley] establece: "…"».
   Recorta con […] si hace falta, pero nunca parafrasees la norma en el lugar
   donde debería ir su texto. Arriba se cumple en 15 de 15; abajo, en el 58%.

2. DERIVA LA REGLA EN ABSTRACTO. Tras cada transcripción, una frase puente que
   extraiga la regla: «De dicho numeral se advierte que…», «Del precepto
   transcrito deriva la regla de que…». Esa frase vale para CUALQUIER caso
   igual: ahí todavía no nombras a quien promueve, ni al órgano, ni el
   expediente. 33% arriba contra 3.3% abajo.

3. DI PARA QUÉ EXISTE LA NORMA. Un párrafo de finalidad: qué problema resuelve
   el precepto y a qué derecho sirve. 53-62% arriba contra 26-43% abajo.

4. ENUNCIA EL LÍMITE DE LA REGLA. Toda regla se escribe con su frontera: «No
   basta [X]», «El [órgano] no debe [Y]», «Tal es la regla general, que
   encuentra excepción cuando…». SI NO PUEDES FORMULAR EL LÍMITE, LA REGLA ESTÁ
   MAL FORMULADA: no la des por terminada. 53% arriba contra 31% abajo.

5. RAZÓN PROPIA PRIMERO, CITA DESPUÉS. Cada tramo abre con dos o tres párrafos
   de razonamiento del tribunal SIN citar nada, y sólo entonces entra la tesis
   que lo respalda: «resulta aplicable la jurisprudencia…». Nunca abras un tramo
   con la cita: ése es el patrón de las medias, donde la tesis sustituye al
   razonamiento en vez de apoyarlo.

6. VE A LA EJECUTORIA, NO SÓLO A LA TESIS. Si el criterio viene de una
   contradicción o de un asunto identificable, nómbralo y resume en tres a seis
   líneas los hechos que la Corte tuvo enfrente. Y si afirmas que OBLIGA,
   escribe por qué: expediente, órgano, fecha de sesión, votación y el precepto
   que le da fuerza (artículo 217 o 223 de la Ley de Amparo). Sin esos datos no
   escribas que obliga: cítalo como criterio orientador.

7. NOMBRA LA OPERACIÓN. Prohibido el salto tesis → conclusión. Después de cada
   criterio di qué haces con él: aplicación directa, analogía, identidad de
   razón, orientador, o distinguible. Si es analogía, di en qué se parecen los
   hechos. Si lo descartas, di por qué no aplica.

8. UN AGRAVIO, UN TRAMO. Por cada concepto: (a) transcribe entre comillas lo
   que alegó la parte o la consideración que vas a calificar; (b) la
   calificación en oración propia, corta y aislada, en punto y aparte —«Esa
   determinación resulta ilegal.»—; (c) la razón; (d) la tesis de apoyo.
   Prohibido resolver dos con una calificación conjunta, salvo que declares que
   se estudian juntos y por qué.

9. RESPONDE LA MEJOR OBJECIÓN DEL QUE PIERDE. Por cada cuestión, un párrafo
   dedicado al argumento contrario más fuerte, respondido con razón propia. 93%
   arriba contra 58% abajo. No vale el espantapájaros: identifica el argumento
   real —lo tienes en el sondeo del acervo— y desmóntalo.

10. TITULA POR FUNCIÓN. Cada apartado anuncia qué se decide ahí: «Decisión del
    asunto», «Parámetro de control constitucional», «Aplicación al caso»,
    «Efectos de la concesión», «Costas». Nunca «Estudio» a secas.

11. NO TE VAYAS POR LA PUERTA PROCESAL. Antes de declarar inoperante, intenta
    el fondo bajo suplencia o causa de pedir y DEJA CONSTANCIA ESCRITA de ese
    intento. Sólo el 7% de las mejores sale por un filtro procesal, contra el
    14% de las medias.

12. NO ALARGUES. Si un párrafo no transcribe una fuente, no deriva una regla,
    no aplica una regla a un hecho del expediente o no responde una objeción,
    SOBRA. La extensión acompaña a la calidad; no la produce.
"""

# ═══ LO COMÚN, EN LA v2 (26-sep-2026) ═════════════════════════════════════
# Cinco reglas de arriba se contradecían con otras del mismo prompt, y el
# modelo obedecía a la que leyera al final (diagnóstico del estudio, C7 y C4):
#   · la 1 mandaba TRANSCRIBIR el precepto entre comillas y el bloque de
#     encima prohíbe transcribirlo en el cuerpo; la 2 enseñaba «Del precepto
#     transcrito…», que ese mismo bloque prohíbe (fila 6 de la propuesta);
#   · la 8 mandaba transcribir lo que alegó la parte —el resumen ya está
#     arriba— y prohibía calificar dos juntos (fila 5);
#   · la 9 mandaba un párrafo de objeción «por cada cuestión», y la objeción
#     se ordenaba además desde otros tres sitios: salía contestada tres y
#     cuatro veces (fila 8);
#   · la 10 mandaba titular cada apartado y la FORMA prohíbe los rótulos
#     (fila 7).
# Aquí van quitadas o reescritas; lo demás, igual. La 11 remite a la regla
# de suplencia del cuerpo, que en la v2 es una sola.
_ARQUITECTURA_COMUN_V2 = """
═══════════════════════════════════════════════════════════════════════
EL CUERPO NO TRANSCRIBE: EL TEXTO ÍNTEGRO VA A LA NOTA AL PIE
═══════════════════════════════════════════════════════════════════════
Esto no es una preferencia de estilo: es cómo queda maquetado el documento, y
si escribes contra ello el proyecto sale roto.

El documento baja SOLO a la nota al pie el texto íntegro de cada precepto que
identificas y de cada tesis larga. Tú no lo copias. Y como no hay transcripción
en el cuerpo, TODA FRASE QUE LA ANUNCIE O REMITA A ELLA QUEDA APUNTANDO A NADA:

  PROHIBIDO                        LO QUE SE ESCRIBE EN SU LUGAR
  «…establece lo siguiente.»       «El artículo 38 del Código Fiscal de la
  «…dispone lo siguiente.»          Federación exige que el acto notificado
  «…señala lo siguiente:»           conste por escrito y lleve la firma del
                                    funcionario competente.»
  «Del precepto transcrito…»       «Del artículo 38…», «De esa disposición…»
  «El criterio transcrito…»        «El criterio citado…», «Esa tesis…»
  «la transcripción que antecede»  «lo dispuesto en ese precepto»

Medido en la revisión fiscal 91/2025: el proyecto decía «El artículo 38 del
Código Fiscal de la Federación establece lo siguiente.» y debajo, en vez del
texto, empezaba otro párrafo. Y decía «Del precepto transcrito deriva que…»
sobre un precepto que sólo estaba al pie. Ese proyecto no se puede firmar.

LA REGLA, EN UNA LÍNEA: nombra el precepto o la tesis y DI LO QUE DICE, dentro
de tu propia frase. Quien firme comprobará el texto en la nota.

═══════════════════════════════════════════════════════════════════════
EN REVISIÓN SE REVOCA; SÓLO EN AMPARO SE DEJA INSUBSISTENTE
═══════════════════════════════════════════════════════════════════════
David: «en revisión la sentencia no se deja insubsistente, se revoca. Sólo en
amparo (cuando se concede) se ordena que se deje insubsistente el acto
reclamado».

No son dos maneras de decir lo mismo. Son dos figuras, y cada una vive en su
sitio:

  · REVISIÓN (o cualquier recurso). El tribunal es ÓRGANO REVISOR de esa misma
    sentencia y la REVOCA: con eso deja de existir. No hay a quién ordenarle
    que la deje insubsistente, porque ya no está. Si después hay reenvío, lo
    que se ordena es DICTAR OTRA —«dicte otra sentencia en la que se ocupe de
    los conceptos cuyo estudio omitió»—, no dejar insubsistente nada.

  · AMPARO (directo, o indirecto en revisión, cuando se CONCEDE). El tribunal
    NO revoca el acto reclamado: no es su superior jerárquico. Concede la
    protección y ORDENA a la autoridad responsable que lo deje insubsistente y
    dicte otro. Ahí sí, y por eso la fórmula existe.

NO ESCRIBAS «deje insubsistente la sentencia recurrida» EN UN RECURSO. Es la
fórmula del amparo colocada donde no cabe, y describe una potestad que este
tribunal no está ejerciendo.



═══════════════════════════════════════════════════════════════════════
CÓMO SE ESCRIBE ESTE ESTUDIO
═══════════════════════════════════════════════════════════════════════
Esto está medido sobre 1,946 sentencias del propio acervo, comparando las que
el corpus puntuó alto contra las que puntuó en la media. No son preferencias de
estilo: son las operaciones que separan a unas de otras.

1. DERIVA LA REGLA EN ABSTRACTO. Donde expones una premisa, nombra el precepto
   y di lo que establece dentro de tu frase; enseguida, una frase puente que
   extraiga la regla de modo que valga para CUALQUIER caso igual: ahí todavía
   no nombras a quien promueve, ni al órgano, ni el expediente. La frase puente
   retoma el precepto por su número o como «dicho numeral», nunca como
   «transcrito»: en el cuerpo no hay transcripción. 33% arriba contra 3.3%
   abajo.

2. DI PARA QUÉ EXISTE LA NORMA, donde la premisa se expone: qué problema
   resuelve el precepto y a qué derecho sirve. 53-62% arriba contra 26-43%
   abajo.

3. ENUNCIA EL LÍMITE DE LA REGLA. Toda regla se escribe con su frontera: lo que
   no basta, lo que el órgano no debe hacer, o la excepción que la regla general
   admite. SI NO PUEDES FORMULAR EL LÍMITE, LA REGLA ESTÁ MAL FORMULADA: no la
   des por terminada. 53% arriba contra 31% abajo.

4. RAZÓN PROPIA PRIMERO, CITA DESPUÉS. Donde expones una premisa, el
   razonamiento del tribunal va delante y la tesis que lo respalda detrás.
   Nunca abras un tramo con la cita: ése es el patrón de las medias, donde la
   tesis sustituye al razonamiento en vez de apoyarlo.

5. VE A LA EJECUTORIA, NO SÓLO A LA TESIS. Si el criterio viene de una
   contradicción o de un asunto identificable, nómbralo y resume en tres a seis
   líneas los hechos que la Corte tuvo enfrente. Y si afirmas que OBLIGA,
   escribe por qué: expediente, órgano, fecha de sesión, votación y el precepto
   que le da fuerza (artículo 217 o 223 de la Ley de Amparo). Sin esos datos no
   escribas que obliga: cítalo como criterio orientador.

6. NOMBRA LA OPERACIÓN. Prohibido el salto tesis → conclusión. Después de cada
   criterio di qué haces con él: aplicación directa, analogía, identidad de
   razón, orientador, o distinguible. Si es analogía, di en qué se parecen los
   hechos. Si lo descartas, di por qué no aplica.

7. CADA ARGUMENTO, UNA RESPUESTA IDENTIFICABLE. Quien lea el estudio tiene que
   poder señalar dónde se contestó cada argumento de la parte, y con qué
   calificación si no es la de su apartado. Estudiar varios juntos es correcto
   cuando atacan la misma consideración y caen por la misma razón: se dice que
   se estudian juntos, se nombran y se dice qué los une. Lo que no se hace es
   volver a contar lo que alegó: el resumen de arriba ya lo contó.

8. NO TE VAYAS POR LA PUERTA PROCESAL. Antes de declarar inoperante un
   argumento, busca su causa de pedir e intenta el fondo; la inoperancia se
   declara cuando ni así toca la razón que decide. Sólo el 7% de las mejores
   sale por un filtro procesal, contra el 14% de las medias. (Si opera la
   suplencia de la queja, rige la regla de suplencia de este prompt.)

9. NO ALARGUES. Si un párrafo no enuncia una regla con su fuente, no la aplica
   a un hecho del expediente, no contesta un argumento ni una objeción, SOBRA.
   La extensión acompaña a la calidad; no la produce.
"""

# ── Y AQUÍ EL HALLAZGO QUE DESMONTA LO QUE YO HABÍA CONSTRUIDO ───────────────
# Yo había hecho que el marco jurídico se escribiera ENTERO al principio y el
# caso viniera después. En administrativa eso es exactamente lo que hacen las
# buenas. En laboral y en civil es exactamente lo que hacen las MEDIAS.
#
# Medido: en laboral, la sentencia de calidad 5 cierra el circuito regla→caso
# entre tres y cinco veces y el primer anclaje al expediente cae al 20% del
# texto; la de calidad 3 hace UN ciclo largo, con el primer anclaje al 60%, y el
# 37% no vuelve nunca al caso. En administrativa está al revés: las medias
# vuelven al caso sin parar porque nunca se alejan lo bastante para construir
# una regla (0.29 anclajes por 10,000 caracteres contra 0.18 arriba).
#
# No hay una arquitectura buena: hay una por materia, y son opuestas.

_ARQUITECTURA = {
    "laboral": """
═══════════════════════════════════════════════════════════════════════
ARQUITECTURA — MATERIA LABORAL
═══════════════════════════════════════════════════════════════════════
ESCRIBE EN CICLOS CORTOS, NO EN DOS MITADES. La sentencia media expone derecho
durante media página o dos tercios y aplica al final, una sola vez; el 37% no
escribe nunca «en el caso concreto». Tú haces TRES A CINCO ciclos de regla →
caso, y el primero cae antes del 25% del estudio. Ningún bloque de derecho se
cierra sin un párrafo inmediato que empiece por «En el caso concreto…» y aplique
esa regla a un hecho nombrado, con su fecha y su foja.

ORDEN. Primer párrafo: declara qué examinas, en qué orden y bajo qué principio
—mayor beneficio, causa de pedir, suplencia cuando quien promueve es el trabajador—,
con la tesis que autoriza ese orden. 40% arriba contra 12% abajo.

CITAS. Pide primero criterios de la SEGUNDA SALA y de materia laboral: arriba
son el 64% y el 56%. Abajo la materia más citada es Común (40%), es decir,
técnica de amparo genérica: esos criterios sostienen el ORDEN del estudio, nunca
el fondo. Cinco registros distintos como mínimo (media medida: 6.33 arriba,
3.46 abajo). Jurisprudencia sobre tesis aislada, y Undécima Época sobre las
anteriores —27% arriba contra 11%—; si usas una anterior a la reforma, escribe
la razón.

CONSTITUCIÓN. Amarra la regla a un precepto CON apartado y fracción y úsalo como
premisa, no como adorno de apertura. Los dos anclajes medidos son el artículo 17
—justicia pronta, fondo sobre formalismo: 73% arriba contra 38%— y el 123 con su
apartado y fracción.

CIERRE. Dos partes obligatorias: (1) los efectos como LISTA NUMERADA de órdenes
en imperativo a la responsable, cada una verificable —«1. Deje insubsistente el
laudo; 2. Dicte otro en el que…»—: 53% arriba contra 32%; (2) un párrafo que
diga qué conceptos quedan sin estudiar y por qué. Si hay amparo adhesivo,
pronúnciate.

CONCENTRA. Agrupa conceptos conexos en una sola cuestión y desarróllala a fondo:
arriba se resuelven 1.28 agravios por sentencia y abajo 1.75. Menos temas, más
desarrollo en cada uno.
""",
    "administrativa": """
═══════════════════════════════════════════════════════════════════════
ARQUITECTURA — MATERIA ADMINISTRATIVA
═══════════════════════════════════════════════════════════════════════
AQUÍ ES AL REVÉS QUE EN LABORAL, y está medido: la regla se construye en un
BLOQUE CONTINUO Y ABSTRACTO, y el caso entra después, una sola vez, cuando la
regla ya está completa. Las medias vuelven al caso sin parar porque nunca se
alejan lo bastante para construir una regla. NO escribas «en el caso concreto»
ni «en la especie» hasta que el bloque de regla esté cerrado, y escribe ese
bloque sin nombrar a quien promueve, al órgano ni al expediente.

DECLARA EL RÉGIMEN ANTES DE EMPEZAR. Di cuál de los dos casos es: (a) hay tesis
exactamente aplicable —aplícala y detente—; o (b) hay que fijar el alcance de
una regla —constrúyela—. Si es (b), el bloque abstracto es obligatorio, con su
párrafo de finalidad y, cuando la regla tenga condiciones, la enumeración
explícita de requisitos: 80% arriba contra 50%. Elaborar la regla en vez de
limitarse a aplicarla es la ÚNICA diferencia que se sostiene en todos los cortes
de esta materia.

APERTURA. Título descriptivo de lo que se decide. Fija el orden de estudio
citando y TRANSCRIBIENDO el precepto que lo manda (artículo 93 de la Ley de
Amparo). Si quien recurre es la autoridad, declara que no opera la suplencia.

CAPA INTERAMERICANA — es la única capa de fuentes que discrimina en esta materia
y sobrevive el control por circuito y por longitud: 38% arriba contra 6% abajo.
Cuando el problema toque un derecho humano, inserta un peldaño con TRES piezas,
las tres o ninguna: (i) instrumento con artículo; (ii) fuente interamericana con
localizador —caso «X vs. México» con número de párrafo, u Opinión Consultiva con
su fecha—; (iii) el ancla de obligatoriedad (P./J. 21/2014). La palabra
«convencionalidad» sin instrumento y sin párrafo está PROHIBIDA. Ese peldaño va
DESPUÉS de la regla constitucional y ANTES del caso, nunca como coda final.

NORMA REFORMADA. Si el precepto cambió y el cambio importa, pon un cuadro
comparativo de dos columnas con el texto anterior y el vigente, y construye la
premisa sobre el texto, no sobre su resumen.

EN ESTA MATERIA NO HAGAS —todo esto está INVERTIDO, es decir, lo hacen MÁS las
medias que las buenas—: preferir a la Corte por ser la Corte (la distribución
por instancia es idéntica en los tres niveles); invocar la Constitución para
subir de nivel (35.4% arriba contra 43.5% abajo); aplicar por analogía (31%
contra 55%: es el atajo de quien no construyó la regla); citar exposición de
motivos (6% contra 25%); acumular «en efecto», «ello es así».
""",
    "civil": """
═══════════════════════════════════════════════════════════════════════
ARQUITECTURA — MATERIA CIVIL
═══════════════════════════════════════════════════════════════════════
LA CADENA DE CUATRO ESLABONES, entera y en este orden, por cada cuestión de
fondo: (1) NOMBRAS el precepto entero —número y ley— y dices lo que establece,
con tus palabras y dentro de tu frase; su texto íntegro lo baja el documento a
la nota al pie y tú no lo copias; (2) derivas la regla con una frase puente
—«De dicho numeral es posible advertir que, por regla general,…»—; (3) entra la
autoridad: «es aplicable la jurisprudencia X, sustentada por la Primera Sala…»,
con su rubro; el texto de la tesis lo pone el documento, no tú; (4) nombras la
operación que enlaza ese criterio con estos hechos. La cadena completa se cumple en el 53% de
las de calidad máxima y en el 19% de las medias.

Y ESCRIBE EN CICLOS: no toda la regla al principio y todo el caso al final. Eso
último es el patrón de la calidad media.

SI SÓLO TIENES EL REGISTRO Y NO EL TEXTO DE LA TESIS, NO LA CITES: OMÍTELA. Tres
tesis transcritas como mínimo cuando el asunto tenga dos o más cuestiones de
fondo (mediana medida: 4 rubros arriba, 1 abajo).

EJECUTORIA. Cuando la tesis venga de una contradicción o de un amparo
identificable, nombra el asunto, cuenta en tres a seis líneas los hechos que la
Corte tuvo enfrente y escribe el paralelismo: «Las características del presente
caso se asemejan a lo ocurrido en el diverso decidido por el Alto Tribunal y,
cambiando lo necesario, conducen a resultado similar». 15 de 15 arriba.

SUPLENCIA. Antes del fondo comprueba si el caso cae en un supuesto de suplencia
—menores, materia familiar, orden público, violación manifiesta—. Si cae,
anúnciala en la misma frase del veredicto y fúndala con precepto y tesis: 53%
arriba contra 25%. Si no cae, no la menciones.

PRINCIPIOS. Nombra expresamente los que gobiernan —congruencia, exhaustividad,
seguridad jurídica, interés superior—: la meta medida es cuatro o más.
""",
    "penal": """
═══════════════════════════════════════════════════════════════════════
ARQUITECTURA — MATERIA PENAL
═══════════════════════════════════════════════════════════════════════
AQUÍ LA FORMA DE TRAER EL CRITERIO CAMBIA, y es contraintuitivo: las de calidad
máxima NO transcriben un solo rubro en mayúsculas (0.0 por estudio contra 1.2 en
las medias). Transcriben el RAZONAMIENTO NUMERADO de la ejecutoria —mediana de
24.5 párrafos contra 3.5—. Lo universal es traer el CONTENIDO del criterio, no
su clave; en penal el contenido es la ejecutoria, no la tesis.

ABRE CON LA HIPÓTESIS NORMATIVA IMPERSONAL: «Cuando…», «Tratándose de…», «Para
que…». 40% arriba contra 21%.

LA FRONTERA NEGATIVA ES EL RASGO FUERTE de esta materia: «no basta», «no debe»,
«no procede» —53% arriba contra 31%—, con la excepción explícita nombrada y, si
la hay, la vía alternativa.

CITAS: EXCLUYE tesis de Tribunales Colegiados. La suplencia de la queja en favor
del reo (artículo 79, fracción III) es absoluta: opera aun sin conceptos.
""",
}


def _bloque_circuito(tipo_asunto: str, criterios: list) -> str:
    """Cómo resuelve este circuito, contado sobre su propio acervo.

    No es una instrucción de estilo: es el recuento de 12,272 expedientes del
    Vigésimo Segundo con 65,282 agravios calificados. Se le enseña al modelo
    para dos cosas: que sepa qué desenlace corresponde a las calificaciones que
    ha fijado el secretario, y que note cuándo la combinación es rara.
    """
    try:
        import tabla_circuito as _tc
        import tipos_asunto as _ta_c
        t = _tc.TABLA_CIRCUITO.get(_ta_c.normalizar(tipo_asunto))
    except Exception:
        return ""
    if not t or not t.get("sentidos"):
        return ""
    lineas = [f"\n\nCÓMO RESUELVE ESTE CIRCUITO, MEDIDO SOBRE {t['expedientes']:,} "
              f"EXPEDIENTES SUYOS", "",
              "No es un consejo de estilo: es el recuento de su propio acervo. Cada",
              "línea dice con qué frecuencia sale ese desenlace y cómo se calificaron",
              "los agravios en él."]
    for sent, dat in t["sentidos"].items():
        cal = " · ".join(f"{k} {v}%" for k, v in dat["calificaciones"].items())
        lineas.append(f"  · {sent.upper():22s} {dat['frecuencia']:4.1f}% de los asuntos "
                      f"→ {cal}")
    # LO RARO SE SEÑALA, no se prohíbe. Confirmar con un agravio fundado ocurre
    # en el 1% del circuito: es posible —se confirma por razones distintas de
    # las del juzgado— pero merece que el estudio lo explique.
    _fund = [c for c in (criterios or [])
             if str(getattr(c, "sentido", "")).lower().startswith(("fundad", "esencial"))]
    if _fund:
        lineas.append("")
        lineas.append("HAS FIJADO AL MENOS UN PLANTEAMIENTO FUNDADO. Mira arriba qué "
                      "desenlace corresponde: si el que vas a escribir es de los que "
                      "casi nunca llevan un fundado, DILO Y EXPLÍCALO en el estudio. "
                      "No lo escondas: un revisor que conoce el circuito lo va a notar.")
    return "\n".join(lineas) + "\n"


def _bloque_conceptos(rama: str, conceptos: str, variante: str = "v1",
                      reasuncion: dict = None) -> str:
    """Los conceptos de violación, cuando hay que estudiarlos por primera vez.

    DOS SUPUESTOS (28-sep-2026):
    · el recurso LEVANTA UN SOBRESEIMIENTO: el colegiado asume jurisdicción
      —artículo 93, fracciones I y V (decía «fracción I»: la I manda examinar
      los agravios contra el sobreseimiento; el estudio de fondo que sigue lo
      manda la V)— y resuelve lo que el Juzgado de Distrito no resolvió;
    · el recurso REVOCA UNA CONCESIÓN (fracción VI): recurre la autoridad o la
      tercera interesada, sus agravios prosperan, y el tribunal estudia los
      conceptos que el juzgado no estudió antes de conceder o negar. AR
      631/2025: el proyecto negó sin estudiarlos. Ver `_bloque_reasuncion`.

    `reasuncion` es lo que `redactor_adelanto` dejó en el material
    (`material.reasuncion`): None si no se calculó —entonces manda la rama—, o
    un dict con su «reasuncion» (vacía = no aplica, p. ej. recurre la quejosa),
    los conceptos que ya estaban en el material y de dónde salieron.

    David: «en el proyecto lo que se estila es abrir un nuevo considerando de
    estudio de los conceptos de violación, y aquí puede ocurrir, tal y como
    ocurre en amparo directo, que los conceptos resulten fundados, infundados o
    inoperantes».
    """
    _re = reasuncion if isinstance(reasuncion, dict) else None
    if rama == "revoca_fondo_niega" and (_re is None or _re.get("reasuncion") == "concesion"):
        return _bloque_reasuncion(conceptos or str((_re or {}).get("conceptos") or ""), _re or {})
    if not rama.startswith("revoca_sobreseimiento"):
        return ""
    _origen_c = "tal como los aportó el secretario"
    if not (conceptos or "").strip() and _re and str(_re.get("conceptos") or "").strip():
        # LOS QUE YA ESTABAN EN EL MATERIAL (la demanda entre las constancias,
        # o la recurrida que los transcribe), con su procedencia (28-sep-2026).
        conceptos = str(_re.get("conceptos"))
        _origen_c = _ORIGEN_CONCEPTOS.get(str(_re.get("donde") or ""), _origen_c)
    if not (conceptos or "").strip():
        return ("\n\nFALTAN LOS CONCEPTOS DE VIOLACIÓN. Este recurso levanta el "
                "sobreseimiento, así que hay que estudiarlos, y no constan. NO "
                "LOS INVENTES ni los deduzcas de los agravios: son escritos "
                "distintos. Escribe el apartado del estudio de los agravios, "
                "cierra diciendo que procede levantar el sobreseimiento, y "
                "añade en ADVERTENCIAS que el estudio de los conceptos de "
                "violación queda pendiente porque no obran en el expediente "
                "del recurso.\n")
    # EN LA v2 NO HAY «CUATRO PASOS» A LOS QUE REMITIR (26-sep-2026): la
    # limpieza los cambió por funciones por apartado, y una remisión a una
    # técnica que el prompt ya no enseña apunta a nada.
    _tecnica_c = ("con la MISMA forma de construir cada apartado que usaste con "
                  "los agravios" if normalizar_variante(variante, "v1") == "v2"
                  else "con la MISMA técnica de los cuatro pasos\nque usaste con los agravios")
    return f"""

ESTUDIO DE LOS CONCEPTOS DE VIOLACIÓN — UN CONSIDERANDO NUEVO

Este recurso levanta el sobreseimiento, y con eso el tribunal ASUME
JURISDICCIÓN: no devuelve el asunto al Juzgado de Distrito, lo resuelve él.

Después del apartado en que declares fundado el agravio y levantes el
sobreseimiento, ABRE UN APARTADO NUEVO —con su propio rótulo, «Estudio de los
conceptos de violación»— y estúdialos {_tecnica_c}.

TRES COSAS QUE NO SE CONFUNDEN:
- Los conceptos de violación son de la DEMANDA DE AMPARO y van contra el ACTO
  RECLAMADO. Los agravios son del RECURSO y van contra la sentencia del
  juzgado. No mezcles unos con otros ni los llames igual.
- Este estudio es de PRIMERA VEZ: nadie los ha examinado antes. No digas «el
  juzgado consideró» sobre ellos, porque el juzgado sobreseyó sin entrar.
- Su desenlace es propio: los conceptos pueden ser FUNDADOS, INFUNDADOS o
  INOPERANTES, igual que en un amparo directo, y de ahí sale si se ampara o no
  se ampara. Que el agravio fuera fundado sólo probó que no debió sobreseerse.

LOS CONCEPTOS DE VIOLACIÓN, {_origen_c}:
──────────────────────────────────────────
{conceptos.strip()[:400000]}
──────────────────────────────────────────
"""


# DE DÓNDE SALIERON LOS CONCEPTOS, dicho al estudio (28-sep-2026): los que no
# aportó el secretario se tomaron del material, y si están incompletos el
# estudio tiene que decirlo, no suplirlos.
_ORIGEN_CONCEPTOS = {
    "secretario": "tal como los aportó el secretario",
    "constancias": ("tal como constan en la demanda de amparo que obra entre las "
                    "constancias (si se ven incompletos, dilo en ADVERTENCIAS)"),
    "recurrida": ("tal como los transcribe la sentencia recurrida (si se ven "
                  "incompletos, dilo en ADVERTENCIAS)"),
}


def _bloque_ficha(material) -> str:
    """LA FICHA PROCESAL, en el encabezado de los datos del estudio (SPEC_E2,
    28-sep-2026). Junto a la ficha de partes: quién promovió, quién recurre y
    con qué carácter, qué resolvió el juzgado por acto, qué es materia de la
    revisión y qué quedó firme, y la fracción del art. 93 que rige. En el AR
    631/2025 el estudio no sabía que la recurrente era la tercera interesada
    ni que el sobreseimiento del otro acto no se había impugnado. «» si no hay
    ficha (entonces el prompt no cambia)."""
    b = str(getattr(material, "ficha_procesal", "") or "").strip()
    return ("\n" + b + "\n") if b else ""


# LA TESIS DE LA REASUNCIÓN, CUANDO FALTAN LOS CONCEPTOS, TAMBIÉN HABLA (AR
# 631/2025, al generar en pantalla, 28-sep-2026). El estudio cerró el párrafo
# del art. 93 con «Sirve de apoyo el criterio de registro 171925:» y el párrafo
# siguiente ya era el segundo agravio: ni lo que el criterio exige ni por qué
# eso impide decidir aquí (SPEC_D: cita → regla con palabras propias →
# aplicación al caso). Descripción de lo que se hace, sin frase que copiar.
_CITA_REASUNCION_SIN_CONCEPTOS = (
    "SI CITAS EL CRITERIO QUE MANDA REASUMIR JURISDICCIÓN (uno de los apoyos de esta técnica, "
    "sólo si llegó entre las tesis del material), HAZLO HABLAR Y APLÍCALO, como cualquier otra "
    "cita: después de citarlo, di con tus palabras lo que exige —que el órgano revisor, al "
    "revocar la concesión, analice los conceptos cuyo estudio omitió el juzgado antes de "
    "conceder o negar, sin importar quién recurra— y aplícalo a este expediente: esa exigencia "
    "es justo la que impide decidir aquí el amparo, porque los conceptos no obran en el "
    "expediente del recurso; por eso el punto del amparo queda pendiente. Si no vas a decir las "
    "dos cosas, no lo cites: no cierres un párrafo con la cita ni pases de ella al siguiente "
    "agravio.\n")


def _bloque_reasuncion(conceptos: str, reas: dict) -> str:
    """ARTÍCULO 93, FRACCIÓN VI: revocada la concesión, el tribunal reasume
    jurisdicción y estudia los conceptos de violación que el juzgado no
    estudió (AR 631/2025, 28-sep-2026: el proyecto pasó de «procede revocarla»
    a «no ampara ni protege» sin estudiar ninguno).

    Descripción de lo que hay que hacer, NUNCA frases para copiar (lección del
    proyecto: los ejemplos del prompt se firman literales). La dependencia es la
    misma del plan-6 entre argumentos: lo que descansa en la tesis ya
    desestimada cae por las mismas razones o por derivar; lo de contenido
    propio se contesta. El resolutivo sale de la conclusión de este considerando
    (`fase_rama.sentido_en_plenitud`), por eso se pide que la diga."""
    firme = ("\n- La sentencia recurrida también sobreseyó respecto de algún acto, y quien "
             "recurre no es la parte quejosa, la única a quien ese sobreseimiento perjudica: "
             "no es materia de la revisión. Dilo en el estudio y declara que queda firme; por "
             "eso la revocación se acota a la materia de la revisión."
             if reas.get("sobreseimiento_firme") else
             "\n- La sentencia recurrida también sobreseyó respecto de algún acto. Si "
             "ningún agravio combate ese sobreseimiento, dilo y declara que queda firme; "
             "por eso la revocación se acota a la materia de la revisión."
             if reas.get("sobresee_ademas") else
             "\n- Si la sentencia recurrida resolvió algo que ningún agravio combate —un "
             "sobreseimiento respecto de otra autoridad, por ejemplo—, dilo y declara que "
             "queda firme; la revocación se acota entonces a la materia de la revisión.")
    if not (conceptos or "").strip() and reas.get("hacen_falta") == "por_confirmar":
        # LA RECURRIDA NO DICE QUE QUEDARAN CONCEPTOS SIN ESTUDIAR (revisión
        # del 28-sep-2026): la fr. VI manda estudiar los «no estudiados»; si el
        # juzgado los examinó todos, no hay nada que reasumir. Descripción de lo
        # que hay que hacer, sin frases para copiar.
        return ("\n\nREVOCAR NO ES NEGAR (artículo 93, fracción VI, de la Ley de Amparo). "
                "El Juzgado de Distrito concedió y el recurso lo interpone quien no pidió el "
                "amparo: si los agravios prosperan, el tribunal reasume jurisdicción sobre los "
                "conceptos de violación que el juzgado no estudió." + firme + "\n"
                "LA SENTENCIA RECURRIDA NO DICE QUE EL JUZGADO DEJARA CONCEPTOS SIN ESTUDIAR, y "
                "los conceptos no están en el expediente del recurso. Compruébalo en la "
                "recurrida que tienes: si los examinó todos y la quejosa no combatió los que "
                "desestimó (revisión adhesiva, artículo 82), no queda nada que reasumir; dilo, "
                "con la parte de la recurrida que lo muestra, y concluye si se concede o se "
                "niega. Si alguno quedó sin estudiar, NO LO INVENTES ni lo deduzcas de los "
                "agravios: di que el tribunal reasume jurisdicción, NO concluyas si se concede "
                "o se niega y añade en ADVERTENCIAS que el estudio de esos conceptos queda "
                "pendiente porque no obran en el expediente del recurso.\n"
                + _CITA_REASUNCION_SIN_CONCEPTOS)
    if not (conceptos or "").strip():
        return ("\n\nREVOCAR NO ES NEGAR (artículo 93, fracción VI, de la Ley de Amparo). "
                "El Juzgado de Distrito concedió y el recurso lo interpone quien no pidió el "
                "amparo: si los agravios prosperan, el tribunal reasume jurisdicción y tiene "
                "que estudiar los conceptos de violación cuyo estudio omitió el juzgado antes "
                "de conceder o negar." + firme + "\n"
                "FALTAN ESOS CONCEPTOS DE VIOLACIÓN: no constan en el expediente del recurso. "
                "NO LOS INVENTES ni los deduzcas de los agravios: son escritos distintos. "
                "Escribe el estudio de los agravios; si prosperan, di que se revoca y que el "
                "tribunal reasume jurisdicción, pero NO concluyas si se concede o se niega el "
                "amparo: esa conclusión depende de unos conceptos que no tienes. Añade en "
                "ADVERTENCIAS que el estudio de los conceptos de violación no estudiados queda "
                "pendiente porque no obran en el expediente del recurso.\n"
                + _CITA_REASUNCION_SIN_CONCEPTOS)
    origen = _ORIGEN_CONCEPTOS.get(str(reas.get("donde") or "secretario"),
                                   _ORIGEN_CONCEPTOS["secretario"])
    return f"""

REVOCAR NO ES NEGAR — REASUNCIÓN DE JURISDICCIÓN (artículo 93, fracción VI,
de la Ley de Amparo)

El Juzgado de Distrito concedió el amparo y el recurso lo interpone quien no lo
pidió. Si los agravios prosperan, eso prueba que la concesión no se sostiene
por la razón que dio el juzgado; no dice si procede por otra. El tribunal no
devuelve el asunto: reasume jurisdicción y estudia los conceptos de violación
cuyo estudio omitió el juzgado —los que declaró innecesarios al conceder con
uno—, y de ese estudio sale si concede o niega.

CÓMO SE ORDENA:{firme}
- Después del estudio de los agravios, ABRE UN CONSIDERANDO PROPIO para los
  conceptos de violación no estudiados, con su rótulo. Son de la DEMANDA DE
  AMPARO y van contra el ACTO RECLAMADO; no los confundas con los agravios ni
  digas que el juzgado los consideró: no entró a ellos.
- Resuélvelos con la misma dependencia que los agravios: los que descansan en
  la tesis que el estudio de los agravios ya desestimó caen por las mismas
  razones, o son inoperantes por derivar de ella, en grupo y diciendo de qué
  tesis derivan; los de contenido propio —otra prueba, otro vicio, otra
  consecuencia— se contestan en lo suyo, con el material.
- Si todos caen, se niega el amparo. Si alguno prospera, se concede por una
  razón distinta de la del juzgado, y entonces sí fijas los efectos de esa
  concesión.
- Cierra ese considerando diciendo, con tus palabras, si procede conceder o
  negar el amparo: de esa conclusión sale el punto resolutivo.

LOS CONCEPTOS DE VIOLACIÓN NO ESTUDIADOS, {origen}:
──────────────────────────────────────────
{conceptos.strip()[:400000]}
──────────────────────────────────────────
"""


# Los apoyos de la fr. VI que tratan de CALIFICAR los conceptos no estudiados
# (178784: inoperantes los que descansan en lo ya desestimado; 182039: lo mismo
# de los agravios). Sin los conceptos no hay qué calificar.
_APOYOS_VI_DE_LOS_CONCEPTOS = frozenset(("178784", "182039"))
_TECNICA_VI_SIN_CONCEPTOS = (
    "SIN LOS CONCEPTOS NO SE CALIFICAN. Los conceptos de violación que el juzgado "
    "no estudió no obran en el expediente del recurso: si quedaron conceptos sin "
    "estudiar, el estudio dice que el suyo queda pendiente por esa razón, y ninguno "
    "se declara fundado, infundado ni inoperante.")


def _bloque_tecnica(tipo_asunto: str, rama: str = "",
                    violacion_procesal: bool = False, material=None) -> str:
    """Cómo se resuelve ESTE escenario, según la Ley de Amparo.

    Hasta ahora el prompt no decía ni una vez «levantar el sobreseimiento», ni
    «plenitud de jurisdicción», ni «mayor beneficio». El catálogo sabía qué
    resolutivos poner —`RAMAS_REVISION` tiene once ramas medidas— pero nadie le
    decía al modelo qué hay que ESTUDIAR para llegar a cada una.

    Se le dan SÓLO las reglas de su escenario. Darle el cuadro entero es darle
    instrucciones para casos que no son el suyo, y este proyecto ya sabe qué
    pasa con las instrucciones que no vienen a cuento.
    """
    import tipos_asunto as _ta_t
    reglas = _ta_t.tecnica_de(tipo_asunto, rama, violacion_procesal)
    if not reglas:
        return ""
    # SIN LOS CONCEPTOS NO HAY CONSIDERANDO QUE CALIFICARLOS (revisión del
    # 28-sep-2026, AR 631/2025). La técnica de la fr. VI manda un considerando
    # propio con la dependencia —los conceptos que descansan en lo desestimado
    # caen o son inoperantes— y anuncia como apoyos 178784 y 182039, que tratan
    # justamente de eso; el bloque de los conceptos, en cambio, dice que faltan y
    # que no se concluya. Con el plan agotado o fallido la v4 cae a la v3, y ahí
    # el estudio recibía las dos órdenes. Se resuelve en la fuente: sin
    # conceptos, ese renglón dice que el estudio queda pendiente y esos dos
    # apoyos no se anuncian. Con conceptos, nada cambia.
    _reas = getattr(material, "reasuncion", None) if material is not None else None
    _sin_conceptos = bool(isinstance(_reas, dict) and _reas.get("reasuncion") == "concesion"
                          and not _reas.get("tenemos")
                          and not str(_reas.get("conceptos") or "").strip())
    _regla_vi = _ta_t.TECNICA_RESOLUCION.get("revision_reasume_concesion")
    partes = ["\n\nTÉCNICA DE RESOLUCIÓN DE ESTE ASUNTO — no es método, es la ley"]
    for r in reglas:
        partes.append(f"\n· {r['cuando']}\n  Fundamento: {r['fuente']}")
        _es_vi_sin = _sin_conceptos and r is _regla_vi
        for t in r["tecnica"]:
            if _es_vi_sin and t.startswith("UN CONSIDERANDO PROPIO"):
                t = _TECNICA_VI_SIN_CONCEPTOS
            partes.append(f"    – {t}")
        # DECIRLE QUE LOS APOYOS ESTÁN AHÍ. Se traen al acervo por su registro
        # —y llegan: «tesis de la técnica añadidas: 193181, 2000895, 196875,
        # 188742» en los registros de la revisión fiscal 91/2025— pero el
        # modelo no los usó, porque nada le decía que respondieran a ESTA
        # cuestión: los estaba leyendo como material del fondo, y del fondo no
        # tratan.
        #
        # No se le dan los registros a secas —eso invita a citar de memoria—:
        # se le dice que ya los tiene entre las tesis del material, con su
        # texto, y que cite desde ahí.
        _apoyos = list(r.get("apoyos") or [])
        # SÓLO LOS QUE LLEGARON (28-sep-2026). Los de la reasunción (art. 93,
        # fr. VI) se traen al resolver, porque la rama no se sabe en el
        # adelanto: si el acervo no los devolvió, anunciarlos sería invitar a
        # citarlos de memoria. Las reglas de siempre no cambian.
        if r.get("solo_si_estan"):
            _hay = {str(t.get("registro") or "") for t in (getattr(material, "tesis", None) or [])}
            _apoyos = [x for x in _apoyos if str(x) in _hay]
        if _es_vi_sin:
            _apoyos = [x for x in _apoyos if str(x) not in _APOYOS_VI_DE_LOS_CONCEPTOS]
        if _apoyos:
            partes.append(
                "    – APOYOS PARA ESTA TÉCNICA: entre las tesis del acervo que "
                "tienes abajo están las de registro "
                + ", ".join(str(x) for x in _apoyos)
                + ". Tratan exactamente de esta cuestión —no del fondo del "
                  "asunto— y son las que hay que citar al justificarla. "
                  "Cítalas como las demás, desde el texto que se te dio.")
    return "\n".join(partes) + "\n"


def _bloque_global(g, criterios: list = None, de_otros=None) -> str:
    """LO QUE YA SE DECIDIÓ ANTES DE ESCRIBIR, y que el estudio no veía.

    Al preparar la propuesta, el motor calcula tres cosas que el secretario lee
    en pantalla y sobre las que decide: de qué problema cuelga el resultado,
    qué les pasa a los demás, y POR DÓNDE SE CAE la solución —el mejor
    argumento de quien resolvería al revés—.

    Nada de eso llegaba aquí. `Global.bloque()` no lo llamaba nadie: se
    escribió para la pantalla y se quedó en la pantalla. El estudio redactaba
    el razonamiento sin saber cuál era la objeción que tenía que vencer ni qué
    temas quedaban sin materia, y por eso los declaraba inoperantes con una
    etiqueta en vez de con una razón.

    La objeción es lo que más se nota: un estudio que sabe por dónde le van a
    atacar contesta a eso; uno que no lo sabe se limita a afirmar.
    """
    if not isinstance(g, dict) or not g:
        return ""
    partes = []
    # ¿EL SECRETARIO SIGUIÓ AL MOTOR O RESOLVIÓ AL REVÉS? Si el principal va
    # en la dirección contraria a la propuesta del motor, el `efecto` del
    # motor —«se pronuncie sobre los alegatos relevantes»— describe la vía
    # que NO se tomó, y con «escríbelo así de explícito» delante acababa en el
    # proyecto. ADC 93/2026: el accesorio salió «inoperante» con los efectos
    # del «fundado» del motor debajo. Lo que les pasa a los demás lo dice
    # ahora el árbol de decisión, en la razón de cada criterio.
    _pral = next((c for c in (criterios or [])
                  if str(getattr(c, "jerarquia", "")).lower() == "principal"),
                 (criterios or [None])[0])
    _al_reves = bool(_pral is not None and g.get("sentido")
                     and not _misma_direccion(str(g.get("sentido") or ""),
                                              str(getattr(_pral, "sentido", "") or "")))
    if g.get("problema_que_decide") and not _al_reves:
        partes.append(f"DE ESTE PROBLEMA CUELGA EL RESULTADO:\n{g['problema_que_decide']}")
    if g.get("efecto") and not _al_reves:
        partes.append(f"QUÉ LES PASA A LOS DEMÁS:\n{g['efecto']}\n"
                      f"Escríbelo así de explícito en el estudio: un tema que "
                      f"queda sin materia se DICE que queda sin materia, y se "
                      f"dice por qué; no se despacha con la palabra "
                      f"«inoperante» y punto.")
    if _al_reves and g.get("razon"):
        # LA RAZÓN DEL MOTOR ES AHORA LA OBJECIÓN. Quien resolvería al revés
        # es el propio motor, y su razón es lo que el estudio tiene que vencer.
        partes.append(
            f"LA OBJECIÓN MÁS SERIA A ESTA SOLUCIÓN —el motor habría resuelto "
            f"{str(g.get('sentido') or '').replace('_', ' ')}, por esto—:\n{g['razon']}\n"
            # Sin frases hechas (revisión adversarial de la fase E): se
            # describe qué hace el párrafo, no cómo empieza.
            f"CONTÉSTALA EN EL ESTUDIO: un párrafo que reconoce la objeción como "
            f"advertida y ponderada, y a renglón seguido la razón del caso que la "
            f"vence. Lo que el motor escribió como "
            f"«efecto» o como suerte de los demás temas en SU vía NO se usa: "
            f"es de la vía que no se tomó.")
    _alt = g.get("alternativa") if isinstance(g.get("alternativa"), dict) else {}
    if _al_reves and str(_alt.get("razon") or "").strip():
        # LA VÍA QUE SE TOMÓ, ya escrita por el motor como alternativa: su
        # razón toral, sus apoyos y lo que les pasa a los accesorios EN ESTA
        # vía. Es material de trabajo, no mandato; la calificación ya está
        # fijada arriba.
        # LOS QUE INVOCÓ OTRO NO SON APOYO DE ESTA VÍA (revisión del 28-sep-2026,
        # AR 631/2025): en la revisión, la tesis que citó la quejosa en su
        # demanda —o el juzgado en la recurrida— se contesta, no se recicla. En
        # el 631 el 188480 de la quejosa salía aquí como apoyo de la vía que le
        # quitaba el amparo. Se nombran aparte, para contestarlos.
        _otros = {str(x) for x in (de_otros or ())}
        _todos = [str(a) for a in (_alt.get("apoyos") or [])[:6]]
        _ajenos = [a for a in _todos
                   if any(x in _otros for x in _RX_REGISTRO.findall(a))]
        _ap = ", ".join(a for a in _todos if a not in _ajenos)
        partes.append(
            f"CÓMO SE SOSTIENE LA VÍA QUE SE TOMÓ (material del motor):\n"
            f"{_alt['razon']}"
            + (f"\nApoyos del acervo para esta vía: {_ap} (cítalos desde su texto, abajo)." if _ap else "")
            + (f"\nCriterios que invocó otra parte o el órgano recurrido, y que esta vía "
               f"tiene que contestar —no son apoyo suyo—: {', '.join(_ajenos)}." if _ajenos else "")
            + (f"\nEn esta vía, los accesorios: {_alt['efecto']}" if str(_alt.get("efecto") or "").strip() else ""))
    if g.get("en_contra") and not _al_reves:
        partes.append(
            f"LA OBJECIÓN MÁS SERIA A ESTA SOLUCIÓN —el mejor argumento de "
            f"quien resolvería al revés—:\n{g['en_contra']}\n"
            f"CONTÉSTALA EN EL ESTUDIO. No la menciones para descartarla de "
            f"una línea: es lo que el magistrado va a preguntar en la sesión y "
            f"lo que un amparo posterior va a explotar. Un estudio que no se "
            f"hace cargo de la mejor objeción está incompleto aunque acierte "
            f"el sentido.\n"
            f"CON EL LENGUAJE DEL OFICIO: en un proyecto NO se escribe «la "
            f"objeción más seria/fuerte/importante» ni la palabra «objeción». "
            f"Se escribe «No se pierde de vista que…», «No pasa inadvertido "
            f"que…», «En diverso aspecto, una de las disidencias más "
            f"relevantes es…», «No se soslaya la inconformidad de la "
            f"recurrente en el sentido de que…». Y la contestas a renglón "
            f"seguido, con la razón del caso.")
    if not partes:
        return ""
    # ── ESTO NO ES UNA ORDEN, Y SE DICE ──────────────────────────────────
    #
    # El rótulo era «LO QUE YA SE DECIDIÓ, Y QUE TIENES QUE HONRAR» y este
    # bloque iba ANTES del criterio del secretario. Su contenido es la
    # propuesta DEL MOTOR, así que cuando el secretario decidía lo contrario el
    # prompt daba dos órdenes y la primera —la enfática— ganaba.
    #
    # Medido en el ADC 536/2025: el motor propuso el primer concepto FUNDADO,
    # el secretario lo marcó INFUNDADO, el reparto lo aplicó bien… y el estudio
    # escribió «el primer concepto de violación es fundado». La instrucción se
    # perdía aquí, en el último metro.
    #
    # Ahora el criterio va primero y esto va después, rotulado como lo que es:
    # material de trabajo. Y si el secretario resolvió algún punto al revés, se
    # dice expresamente, para que la objeción no se lea como un mandato.
    _suyos = ", ".join(
        f"«{str(getattr(c, 'problema', ''))[:60]}» → {getattr(c, 'sentido', '')}"
        for c in (criterios or []) if getattr(c, "sentido", ""))
    _nota = ""
    if _suyos:
        _nota = ("\nOJO: EL SENTIDO YA ESTÁ FIJADO ARRIBA por el criterio del "
                 "secretario, y es el que manda:\n  " + _suyos
                 + "\nSi algo de lo de abajo apunta a otro sentido, es la "
                   "propuesta del motor, que el secretario NO siguió. Úsalo "
                   "sólo como material —la objeción, los apoyos, el efecto en "
                   "los demás temas— nunca como la calificación.\n")
    return ("\n\nLO QUE EL MOTOR HABÍA PROPUESTO (material, no mandato)\n"
            + _nota + "\n" + "\n\n".join(partes) + "\n")


def _bloque_escrito_literal(escrito: str, resumen: str) -> str:
    """EL ESCRITO DE LA PARTE, EN SUS PROPIAS PALABRAS.

    Hasta ahora la fase que CONTESTA los conceptos no veía los conceptos: se le
    entregaba el resumen —unos 472 palabras— y con eso tenía que responder a
    ocho planteamientos. Del escrito llegaba, al momento de escribir la
    respuesta, CERO caracteres.

    Es la raíz de las tres quejas a la vez: no se puede contestar un concepto
    que no está —de ahí la falta de exhaustividad—; sin las palabras de la parte
    no hay nada concreto que citar y el estudio cae en el molde —de ahí la
    repetición—; y sin el texto delante el modelo rellena con lo que le suena
    —de ahí los nombres y las cifras inventados—.

    El resumen se queda: ordena y da el hilo. Pero debajo va el escrito, para
    que la respuesta se pegue a lo que la parte dijo de verdad.
    """
    t = (escrito or "").strip()
    if len(t) < 400:
        return ""
    return f"""

═══════════════════════════════════════════════════════════════════════
EL ESCRITO DE LA PARTE, LITERAL
═══════════════════════════════════════════════════════════════════════
Lo de arriba es el resumen; esto es lo que la parte escribió. CONTESTA CONTRA
ESTO, no contra el resumen: cita sus palabras cuando importen, y no des por
planteado nada que no esté aquí.

Y CUÉNTALOS. Si aquí hay ocho conceptos, el estudio contesta ocho. Dejar uno
sin respuesta no es una omisión menor: hace la sentencia incongruente e
inexhaustiva, y eso se combate en amparo.
──────────────────────────────────────────
{t}
──────────────────────────────────────────
"""


def _bloque_arquitectura(materia: str, variante: str = "v1") -> str:
    """Lo común más UNA arquitectura de materia. Nunca dos: son opuestas."""
    m = (materia or "").strip().lower()
    _v2a = normalizar_variante(variante, "v1") == "v2"
    comun = _ARQUITECTURA_COMUN_V2 if _v2a else _ARQUITECTURA_COMUN
    propia = _ARQUITECTURA.get(m, "")
    if propia and _v2a:
        propia = _arquitectura_materia_v2(propia)
    if not propia:
        # Sin materia identificada se entrega sólo lo común. Entregar la
        # arquitectura equivocada es peor que no entregar ninguna: en laboral
        # manda escribir en ciclos cortos y en administrativa manda justo lo
        # contrario.
        return comun
    return comun + propia


# ═══ LA ARQUITECTURA DE MATERIA, EN LA v2 (26-sep-2026) ════════════════════
# Se deja entera salvo lo que contradice las reglas nuevas, que son de la
# propuesta aprobada por David: la cuota de citas (laboral, cinco registros
# como mínimo; civil, tres tesis transcritas) choca con «cada premisa con su
# apoyo, máximo dos»; el título descriptivo y la transcripción del precepto de
# la apertura administrativa chocan con «sin rótulos» y «el cuerpo no
# transcribe»; el cierre obligatorio de la laboral, con «sin cierre por
# defecto»; y la suplencia civil anunciada siempre, con el artículo 79,
# penúltimo párrafo: «solo se expresará en las sentencias cuando la suplencia
# derive de un beneficio».
#
# PENÚLTIMO, NO ÚLTIMO (revisión adversarial, 26-sep-2026, leído en el texto
# vigente, DOF 16-10-2025): esa frase cierra el párrafo que empieza «En los
# casos de las fracciones I, II, III, IV, V y VII…»; el último párrafo del 79
# es otro —la suplencia por violaciones procesales o formales sólo opera si no
# hay vicio de fondo—. La propuesta decía «último párrafo», y un prompt que
# cita mal el párrafo enseña al modelo a citarlo mal en la sentencia.
#
# POR SUSTITUCIÓN Y NO COPIA, para que la v2 no se quede atrás cuando alguien
# afine la v1. Si un día una de estas frases cambia en la v1, la sustitución
# no casa y la v2 conservaría la orden vieja: test_prompt_v2.py comprueba que
# ninguna de las órdenes retiradas sobrevive, y ése es el aviso.
_ARQUITECTURA_V2_CAMBIOS = (
    ("el fondo. Cinco registros distintos como mínimo (media medida: 6.33 arriba,\n"
     "3.46 abajo). Jurisprudencia",
     "el fondo. Jurisprudencia"),
    ("CIERRE. Dos partes obligatorias: (1) los efectos como LISTA NUMERADA de órdenes\n"
     "en imperativo a la responsable, cada una verificable —«1. Deje insubsistente el\n"
     "laudo; 2. Dicte otro en el que…»—: 53% arriba contra 32%; (2) un párrafo que\n"
     "diga qué conceptos quedan sin estudiar y por qué. Si hay amparo adhesivo,\n"
     "pronúnciate.",
     "LO QUE QUEDA SIN ESTUDIAR se dice, con su razón, en el apartado donde toca;\n"
     "y si se concede, los efectos van bajo su rótulo como lista numerada de\n"
     "órdenes verificables (53% arriba contra 32%). Si hay amparo adhesivo,\n"
     "pronúnciate."),
    ("APERTURA. Título descriptivo de lo que se decide. Fija el orden de estudio\n"
     "citando y TRANSCRIBIENDO el precepto que lo manda (artículo 93 de la Ley de\n"
     "Amparo).",
     "APERTURA. Fija el orden de estudio con el precepto que lo manda, nombrado y\n"
     "dicho dentro de tu frase: en el amparo directo, el artículo 189 de la Ley de\n"
     "Amparo; en la revisión, el 93."),
    ("SI SÓLO TIENES EL REGISTRO Y NO EL TEXTO DE LA TESIS, NO LA CITES: OMÍTELA. Tres\n"
     "tesis transcritas como mínimo cuando el asunto tenga dos o más cuestiones de\n"
     "fondo (mediana medida: 4 rubros arriba, 1 abajo).",
     "SI SÓLO TIENES EL REGISTRO Y NO EL TEXTO DE LA TESIS, NO LA CITES: OMÍTELA."),
    ("—menores, materia familiar, orden público, violación manifiesta—. Si cae,\n"
     "anúnciala en la misma frase del veredicto y fúndala con precepto y tesis: 53%\n"
     "arriba contra 25%. Si no cae, no la menciones.",
     "—menores, materia familiar, orden público, violación manifiesta—. Si cae y\n"
     "de ella deriva un beneficio, anúnciala en la misma frase del veredicto y\n"
     "fúndala con precepto y tesis: 53% arriba contra 25%. Si no cae, o no deriva\n"
     "beneficio, no la menciones (artículo 79, penúltimo párrafo, de la Ley de\n"
     "Amparo)."),
)


def _arquitectura_materia_v2(texto: str) -> str:
    for viejo, nuevo in _ARQUITECTURA_V2_CAMBIOS:
        texto = texto.replace(viejo, nuevo)
    return texto


def _sentido_del_fallo(criterios: list) -> str:
    """De la calificación de los conceptos al sentido de la sentencia.

    El acervo clasifica sentencias —concede, niega, confirma— y el secretario
    califica conceptos —fundado, infundado, inoperante—. Son dos escalas y hay
    que traducir: basta un concepto fundado para que el amparo se conceda,
    aunque los demás caigan.
    """
    if not criterios:
        return ""
    for c in criterios:
        s = str(getattr(c, "sentido", "") or "").strip().lower()
        if _ta_p.prospera(s):
            return "concede"
    return "niega"


def _bloque_ley_de_la_via(m: Material) -> str:
    """Qué ley gobierna esta vía, dicho ANTES de que el modelo razone.

    Detectar la ley equivocada en el documento terminado es la red; esto es la
    barandilla. En el proyecto de la revisión fiscal salieron CINCO citas de la
    Ley de Amparo —el 74, el 76 y el 79 aplicados como derecho vigente al fondo
    de un recurso que no se rige por esa ley— y el secretario las vio de un
    vistazo: su propio engrose la cita UNA vez, el artículo 92, para el turno.
    """
    import tipos_asunto as _ta_v
    # CON LOS DOS DATOS DERIVADOS: la regla cambia de forma según lo que el
    # sistema sepa del acto. Ver `tipos_asunto.ley_de_la_via`.
    t = _ta_v.ley_de_la_via(getattr(m, "tipo_asunto", ""),
                            getattr(m, "sede_del_acto", ""),
                            getattr(m, "cuaderno", ""))
    if not t:
        return ""
    return "\n── LA LEY QUE GOBIERNA ESTA VÍA ──\n" + t + "\n"


def _bloque_precedente(m: Material, criterios: list = None) -> str:
    """El sondeo del acervo de colegiados, si lo hubo.

    Va SEPARADO del material que funda, y así rotulado: un colegiado no obliga a
    otro. Sirve para saber si uno se aparta de la corriente —y decirlo— y para
    tomar del acervo la objeción que hay que responder, en vez de inventarse una
    fácil de tumbar.
    """
    s = getattr(m, "sondeo", None)
    if s is None:
        return ""
    try:
        import fase_precedente as fp
        return fp.bloque(s, _sentido_del_fallo(criterios or []))
    except Exception:
        return ""


def prompt_estudio(resumen_acto: str, resumen_conceptos: str,
                   criterios: list[Criterio], material: Material,
                   es_recurso: bool = False, partes=None, marco=None,
                   contexto: str = "", materia: str = "",
                   propuesta_global=None, rama: str = "",
                   violacion_procesal: bool = False,
                   conceptos_violacion: str = "",
                   # EL ESCRITO DE LA PARTE, LITERAL. Ver `_bloque_escrito_literal`.
                   escrito_literal: str = "",
                   # EL GUION DEL PLAN DEL ESTUDIO (v4; `plan_estudio.vista`).
                   # Viaja como ARGUMENTO y no colgado del material, que vive en
                   # la memoria del worker de una generación a la siguiente: un
                   # guion de la vuelta anterior no puede colarse en ésta.
                   # Vacío = sin plan. Sólo lo lee la v4.
                   guion: str = "") -> str:
    # LA v4 ES LA v3 CON EL GUION (contrato del Paso 2). Se captura la llamada
    # entera ANTES de definir nada —`locals()` en la primera línea sólo tiene
    # los parámetros— para que la v3 se arme con exactamente lo mismo, también
    # con los parámetros que añada la pieza del inventario.
    if normalizar_variante(getattr(material, "variante", "v1"), "v1") == "v4":
        return _prompt_estudio_v4(dict(locals()))
    # LA VARIANTE LA TRAE EL MATERIAL, como la forma (ver `VARIANTES`). La v1
    # es lo que sigue, congelado por test_prompt_v2.py; la v2 vive aparte
    # para que tocarla no pueda mover ni una coma de la v1. La v3 y la v4 son
    # la v2 con el inventario y la regla de las marcas (ver `_partes_v3`); con
    # esos dos textos vacíos, la v2 sale idéntica (también por instantánea).
    if _v2(material):
        _inv_b, _inv_r = _partes_v3(material, es_recurso) if con_inventario(material) else ("", "")
        return _prompt_estudio_v2(
            resumen_acto, resumen_conceptos, criterios, material,
            es_recurso=es_recurso, partes=partes, marco=marco,
            contexto=contexto, materia=materia,
            propuesta_global=propuesta_global, rama=rama,
            violacion_procesal=violacion_procesal,
            conceptos_violacion=conceptos_violacion,
            escrito_literal=escrito_literal,
            bloque_inventario=_inv_b, recordatorio_marcas=_inv_r)
    q = "agravios" if es_recurso else "conceptos de violación"
    # CÓMO SE LA NOMBRA. Estaba escrito «la parte quejosa» dentro de un EJEMPLO
    # de este prompt, y el modelo lo copiaba: en la revisión fiscal el proyecto
    # decía «En el primer agravio la quejosa sostiene…» refiriéndose al SAT,
    # que nunca fue quejoso. Es la CUARTA vez en este proyecto que un ejemplo
    # del prompt se firma literal.
    # EL TIPO VIAJA CON EL MATERIAL, no como un parámetro más: hay DOS sitios
    # que llaman a esta función —el que transmite en vivo y el que no— y un
    # parámetro nuevo se olvida en uno de los dos. Ha pasado.
    import tipos_asunto as _ta_e
    import dialogo_constitucional as _dc_e
    # EN UN SOLO SENTIDO (24-sep-2026). Si la resolución no favorece a quien
    # reclama el derecho, los criterios del método no se enseñan —no hay
    # peldaño donde citarlos— y el cierre dice que no se invocan. La dirección
    # la calcula quien llama y viaja con el material, como el tipo de asunto.
    _fav_dc = getattr(material, "dialogo_favorece", None)
    _mat_vista = material
    if _fav_dc is False and any(t.get("metodo") for t in (getattr(material, "tesis", None) or [])):
        import copy as _copy_dc
        _mat_vista = _copy_dc.copy(material)
        _mat_vista.tesis = [t for t in (material.tesis or []) if not t.get("metodo")]
    _voc = _ta_e.vocabulario_de(
        getattr(material, "tipo_asunto", "") or "amparo_directo")
    parte = _voc["parte"]
    promovente = _voc["promovente"]
    # El singular sale del catálogo, no de quitarle la última letra al plural.
    q1 = _voc["combate_singular"]
    # LA PRIMERA FRASE DECLARABA EL ASUNTO COMO AMPARO en los cuatro tipos, y
    # con esa premisa todo el léxico del amparo —quejoso, responsable, demanda—
    # queda autorizado por implicación aunque los ejemplos se corrijan.
    _clase = ("una sentencia de amparo directo" if _voc["nombre"] == "amparo directo"
              else f"la resolución de un {_voc['nombre']}")
    _sjs = _ta_e.sujetos_de(
        getattr(material, "tipo_asunto", "") or "amparo_directo")
    _org = " o ".join(f"«{x}»" for x in _sjs["organo"][:2])
    # EL RÓTULO EN MAYÚSCULAS es la forma más imitable que hay: le enseña al
    # modelo cómo llamar al órgano antes de que escriba una palabra. Decía «LO
    # QUE RESOLVIÓ LA RESPONSABLE» en los cuatro tipos, y en una queja lo que
    # se recurre lo resolvió el Juzgado de Distrito, que no es responsable de
    # nada: es el órgano de control cuya decisión se revisa.
    _org_rotulo = _sjs["organo"][0].upper()
    # EL EJEMPLO DE LA REFUTACIÓN decía «la Sala afirmó X», y un ejemplo se
    # copia con su sujeto: en un juicio de única instancia (30-sep-2026, AD
    # 323/2025) el proyecto llamaba Sala al juez. Ahí se nombra al órgano por lo
    # que es; en los demás casos, la v1 congelada.
    _ej_afirmo = "la Sala"
    if _ta_e.unica_instancia(getattr(material, "tipo_asunto", "") or "amparo_directo"):
        _ej_afirmo = _sjs["organo"][0]
    # Los SIN CALIFICAR no entran en la frase de apertura («en parte fundados
    # y en parte » salía en la v1 con un sentido vacío); sin ellos, igual.
    calif = _calificacion([c for c in criterios if str(getattr(c, "sentido", "") or "").strip()]
                          or criterios)
    # ── LA FORMA: ESTÁNDAR O MODERNA ──────────────────────────────────────
    # David, 25-sep-2026. Ver `formato_sentencia.py`. Las órdenes que antes
    # imponían la pregunta en todos los casos viven ahí, en un solo bloque, y
    # se repite una línea al final porque lo último es lo que más se obedece.
    import formato_sentencia as _fs_e
    _formato = _fs_e.normalizar(getattr(material, "formato", ""))
    _objetivo = _objetivo_palabras(material, criterios)
    _forma = _fs_e.forma_del_estudio(_formato, q, q1, parte, calif, _objetivo)
    _recuerda_forma = (
        f"Y LA FORMA ES LA MODERNA: cada problema con su pregunta sola en su "
        f"párrafo, la respuesta enseguida nombrando el {q1} que contesta, y "
        f"alrededor de {_objetivo} palabras en total sin dejar ningún {q1} sin "
        f"respuesta.\n"
        if _formato == _fs_e.MODERNA else
        f"Y LA FORMA ES LA ESTÁNDAR: sin preguntas ni rótulos numerados; cada "
        f"{q1} abre con «Sobre el primer {q1}, en el que {parte} sostiene…», "
        f"sigue su calificación y la demostración arranca con «Lo anterior…». "
        f"Ningún {q1} sin su apartado.\n")
    # EL MARCO SE REPITE AL FINAL. Medido en el proyecto 360/2025: se le
    # entregaron 6,338 caracteres de marco —artículo 4º constitucional y
    # Convención sobre los Derechos del Niño— y el estudio salió con CERO
    # menciones a ambos. El material iba en el 78% del prompt, sin imperativo.
    # Es el mismo fallo que tuvieron las citas, y se arregla igual: repitiendo
    # la orden al final, que es lo último que el modelo lee antes de escribir.
    cierre_marco = ""
    # EL MARCO NO SE ESCRIBE COMO CAPA, EN NINGUNA FORMA (David, 25-sep-2026:
    # «hay que prescindir del marco jurídico»). Antes esta orden pedía un
    # apartado de marco antes del caso; ahora el material constitucional se usa
    # sólo donde decide, dentro de la respuesta a cada planteamiento.
    if isinstance(marco, str) and marco.strip():
        cierre_marco = """
Y EL MARCO JURÍDICO QUE SE TE DIO, SÓLO DONDE DECIDE. La sentencia no lleva
apartado de marco: si un precepto constitucional o convencional es la premisa
de un planteamiento, enúncialo AHÍ, en una frase, y sigue. Nada de repaso
general de derechos humanos, de la Convención Americana o de la Corte
Interamericana que no cambie la respuesta.
"""
    # ═══ EL ORDEN DEL ARTÍCULO 189, LA ÚNICA CORRECCIÓN DE LA v1 (26-sep-2026) ═
    # La fórmula de método decía «privilegiando el estudio de las violaciones
    # procesales que inciden en el sentido del fallo», y se copiaba literal en
    # tres de cada cuatro estudios. El artículo 189 vigente (DOF 13-03-2025)
    # dice lo contrario: «se privilegiará el estudio de los conceptos de
    # violación de fondo por encima de los de procedimiento y forma, a menos que
    # invertir el orden redunde en un mayor beneficio para la persona quejosa».
    # David: «sí, alinear al art. 189… también alinea a cómo debe resolverse
    # (mayor beneficio art 189)». Vale para todos por ser de ley, y por eso es
    # lo ÚNICO que cambia en la v1: la frase del molde y, fuera de las
    # comillas, que si se invierte el orden se diga en qué consiste el beneficio.
    #
    # SÓLO EN EL AMPARO DIRECTO (revisión adversarial, 26-sep-2026). El molde
    # vive en el prompt de los cuatro tipos, y el artículo 189 está en el
    # capítulo del amparo directo: habla de «conceptos de violación» y de
    # «la persona quejosa». Puesto en una revisión fiscal, el molde que se
    # copia literal decía «los agravios… como lo ordena el artículo 189» y
    # «mayor beneficio para la autoridad recurrente»: un precepto mal citado en
    # el primer párrafo del estudio. En los recursos el orden lo fija su
    # técnica (el 93 en la revisión), así que ahí la v1 sigue exactamente como
    # estaba en producción.
    _ad_189 = (not es_recurso) and _voc["nombre"] == "amparo directo"
    _metodo_orden = (
        f"estudio de los de fondo sobre los de procedimiento y forma, como lo\n"
        f"       ordena el artículo 189 de esa ley.» (Si inviertes ese orden porque\n"
        f"       estudiar primero una violación procesal redunda en un mayor beneficio\n"
        f"       para {parte}, dilo y di en qué consiste ese beneficio.)"
        if _ad_189 else
        "estudio de las violaciones procesales que inciden en el sentido del fallo.»")
    # LA REGLA DE FUNDAR, CON LA FUERZA COMÚN (revisión del 29-sep): con la
    # fuerza unificada, «obligatoria» es «vincula a ESTE tribunal», y la
    # jurisprudencia de otro colegiado se cita como orientadora, no se calla.
    import fuerza_juridica as _fj_rf
    _regla_funda = ("FUNDA CON LAS TESIS DEL MATERIAL: la OBLIGATORIA para este tribunal, como"
                    " razón que decide; si no la hay, la ORIENTADORA, como apoyo y diciendo que orienta."
                    if _fj_rf.activa() else "FUNDA CON LAS TESIS OBLIGATORIAS DEL MATERIAL.")
    return f"""Eres el secretario de un Tribunal Colegiado de Circuito redactando el
estudio de fondo de {_clase}. Escribes mejor que la media del
oficio: con más orden, más precisión y menos relleno, pero en su mismo registro.

FORMA — medida sobre 40 engroses firmados, no inventada:
- ABRE con el encabezado ordinal y la CALIFICACIÓN: «SEXTO. Estudio. Los {q}
  son {calif}.» Anunciar el resultado y luego demostrarlo es el orden que mejor
  se lee, y el que sigue el 40% de los engroses reales.
- FRASE de unas 35 palabras, SUBORDINADA; PÁRRAFO de unas 49, es decir UNA O
  DOS FRASES POR PÁRRAFO. Es la medida real del corpus y no es un capricho: la
  prosa judicial encadena la premisa y su consecuencia dentro de la misma
  oración —«toda vez que», «en tanto que», «sin que obste»— en vez de apilar
  cinco frases cortas bajo un mismo párrafo, que es como escribe un informe.
- CONECTORES, por orden de uso real: {', '.join(f'«{c}»' for c in CONECTORES)}.
  No repitas el mismo dos veces seguidas.
- EL ÓRGANO RECURRIDO es {_org}; este tribunal se
  nombra «este Tribunal Colegiado» y usa voz impersonal («se estima», «se
  considera»). Nunca primera persona del singular.
- LA EXTENSIÓN SE REPARTE, NO SE ESTIRA. Alrededor de {_objetivo}
  palabras EN TOTAL, y ese total se gasta donde se decide el asunto:

    · EL TEMA PRINCIPAL se estudia a fondo: la premisa normativa, la
      jurisprudencia que la sostiene, la aplicación a estos hechos, los
      argumentos reforzadores y la objeción previsible con su refutación. Aquí
      la extensión está justificada y aquí es donde debe estar.

    · UN TEMA INOPERANTE se resuelve en DOS O TRES PÁRRAFOS: qué se alegó, por
      qué no combate la razón toral, y ya. La inoperancia se razona, no se
      desarrolla: alargarla no la hace más firme, la hace más discutible.

    · UN TEMA QUE CAE PORQUE CAYÓ EL PRINCIPAL —queda sin materia, o subsiste
      la razón que ya se dio— se despacha también en DOS O TRES PÁRRAFOS,
      diciendo POR QUÉ sigue esa suerte. No se vuelve a razonar el fondo de
      algo que ya no puede cambiar el resultado.

  ESTO NO ES RECORTAR NI DEJAR TEMAS SIN CONTESTAR. Todos se contestan —la
  exhaustividad se revisa de oficio y un tema olvidado es un amparo de
  vuelta—; lo que cambia es cuánto se les dedica. Un proyecto que trata igual
  lo que decide y lo accesorio es más largo, no más completo, y obliga a quien
  lo lee a buscar dónde está la razón.

  Y NO RELLENES. Si un apartado queda corto porque el tema es corto, está
  bien. Repetir la misma razón con otras palabras no añade nada y es lo que un
  revisor marca primero.
- Sin Markdown y sin viñetas.
{_forma}
NO REPITAS LO QUE YA ESTÁ ESCRITO — esto es lo primero:
- Los dos resúmenes que vienen abajo —lo que resolvió la responsable y lo que se
  combate— YA OCUPAN SU PROPIO APARTADO en la sentencia, antes del tuyo. Se te
  dan para que sepas de qué va el asunto, NO para que los reproduzcas.
- TU TEXTO EMPIEZA CON LA CALIFICACIÓN GENERAL —una frase: «los agravios son
  fundados», «resultan en parte infundados y en parte inoperantes»— y ACTO
  SEGUIDO abre el primer apartado en la forma que fija el bloque FORMATO de
  arriba. Nada de recuento.
  Puedes referirte a lo que la responsable sostuvo cuando lo estés refutando
  —«{_ej_afirmo} afirmó X; ese razonamiento es incorrecto porque…»—, pero no vuelvas
  a contar la resolución ni a enumerar los agravios: el lector acaba de leerlos
  dos párrafos más arriba y se encuentra lo mismo por tercera vez.
- Y NO ESCRIBAS RÓTULOS. Nada de «Agravios:», «Conceptos de violación:» ni
  «Solución:»: el documento ya los trae de la plantilla y salen duplicados.

AQUÍ SÍ SE AGRUPA, Y SE ANUNCIA — la regla que él sigue sin excepción:
- La síntesis de arriba respetó el orden y el número que propuso quien promueve.
  ES AQUÍ donde se reordena o se juntan varios, y NUNCA en silencio: se dice
  antes de empezar y con fundamento en el ARTÍCULO 76 DE LA LEY DE AMPARO.
      «Por cuestión de método, los {q} se analizarán agrupados por bloques
       temáticos, conforme al artículo 76 de la Ley de Amparo, privilegiando el
       {_metodo_orden}
      «se procede al análisis conjunto de los {q} identificados como TERCERO y
       QUINTO, dada su estrecha vinculación con el fondo del asunto.»
- EL CRITERIO PARA AGRUPAR NO ES EL ARTÍCULO CONSTITUCIONAL INVOCADO —casi todos
  repiten el 14, el 16 y el 17— sino EL NUDO DE LA SENTENCIA QUE SE ATACA: el
  presupuesto procesal, el elemento de la acción o la prueba concreta en disputa.
- Y SI NO REAGRUPAS, ANÚNCIALO SIN MENTIR. La fórmula que se usaba aquí
  —«atendiendo a su prelación lógica… en el orden propuesto»— se contradice
  sola: la prelación lógica es el orden que fija el TRIBUNAL según lo que
  decide primero, y «el orden propuesto» es el que trae el recurrente. Se
  copiaba literal y el proyecto prometía una cosa y hacía la otra.
  Escribe lo que de verdad ocurra, con tus palabras:
  · si sigues el orden del recurrente porque ya es el lógico, dilo así y di
    por qué lo es;
  · si lo sigues por comodidad, no invoques prelación lógica: basta con decir
    que se analizan en el orden en que se plantean.
  NO copies ninguna de estas dos frases: escribe la tuya.

ARQUITECTURA — CUATRO PASOS POR CADA {q1}, SIEMPRE LOS CUATRO Y EN ESTE ORDEN.

Es la técnica silogística, y no es un adorno de método: es lo que permite
comprobar, leyendo, que no quedó nada sin contestar y que lo contestado se
sostiene. Un apartado por planteamiento, en el orden en que se plantearon,
abierto por su ordinal en letra.

LOS CUATRO PASOS SIEMPRE, PERO NO SIEMPRE DEL MISMO TAMAÑO. Esto es lo que
decide si el proyecto se lee o se sufre, y lo que separa a un secretario
experto de uno que rellena:

  · En el {q1} PRINCIPAL —el que decide el asunto— los cuatro pasos van
    completos: premisa con su jurisprudencia, aplicación detallada a estos
    hechos, y la objeción previsible refutada. Aquí la extensión se gana.

  · En un {q1} que resulta INOPERANTE: **ENTRE TRES Y SIETE PÁRRAFOS, nunca
    más**. Y se resuelve como lo resuelve un secretario: se enuncia el
{__import__('vicio_inoperancia').regla_inoperante_v1(q1)}

  · En un {q1} que SIGUE LA SUERTE del principal —queda sin materia, o
    subsiste la razón ya dada—: **ENTRE TRES Y SIETE PÁRRAFOS**, diciendo POR
    QUÉ sigue esa suerte. No se vuelve a razonar el fondo de algo que ya no
    puede cambiar el resultado.

  CUENTA LOS PÁRRAFOS AL ESCRIBIRLOS. Si un apartado accesorio pasa de siete,
  sobra: quita desarrollo, no quites la tesis ni la conclusión.

Ninguno se queda sin contestar: la exhaustividad se revisa de oficio. Lo que
cambia es cuánto se le dedica a cada uno. Si tu apartado más largo no es el del
tema principal, has repartido mal el proyecto y hay que rehacer el reparto.

  PASO 1 — EL PLANTEAMIENTO, EN SU VERSIÓN MÁS FUERTE.
  Se enuncia con la voz de quien lo formula, no con la del tribunal, y en su
  mejor versión: «En el primer {q1} {parte} sostiene que…». Prohibido
  caricaturizarlo para tumbarlo después: si el argumento tiene un punto, se
  dice, y luego se explica por qué no basta. Un planteamiento debilitado al
  enunciarlo produce una respuesta que no responde.

  PERO NO ESCRIBAS QUE LO ESTÁS HACIENDO. «{parte} sostiene, EN SU VERSIÓN MÁS
  FAVORABLE, que…» delata que sigues una instrucción en vez de redactar una
  sentencia: en un engrose eso no se dice, se hace. Lo mismo con «la mejor
  objeción a esta conclusión es…», que anuncia un ejercicio de método donde
  debe ir un argumento de parte: se escribe «{parte} alega que…» o «no obsta
  que se afirme que…». David tachó las dos a mano.

  PASO 2 — LA PREMISA NORMATIVA, ABSTRACTA Y CON SU FUENTE.
  Qué dice la norma o el criterio que gobierna el punto, enunciado de modo que
  valga para cualquier caso igual: «Del artículo … deriva la regla de que…»,
  «el artículo … impone que…», «con arreglo a la jurisprudencia de la Segunda
  Sala derivada de la contradicción de tesis…». Nunca «del precepto
  transcrito»: no hay transcripción, el texto está en la nota al pie. Aquí NO se nombra todavía al promovente, ni al
  órgano recurrido, ni el expediente: si la frase no vale para otro asunto
  idéntico, no es una premisa, es una conclusión adelantada.

  PASO 3 — LA APLICACIÓN AL CASO, CONFRONTANDO.
  Aquí entran los hechos de ESTE expediente contra la regla del paso 2, y aquí
  —y sólo aquí— se abre con «en el caso», «en la especie». La confrontación se
  escribe: qué dice la constancia, qué exige la norma, y por qué encaja o no.
  Un salto del paso 2 al paso 4 sin este eslabón es la afirmación sin prueba
  que se cae en revisión.

  CÓMO SE ENCADENAN LOS CUATRO PASOS. No son cuatro párrafos puestos uno
  detrás de otro: son un razonamiento que avanza, y cada párrafo lo dice al
  avanzar. Salvo el que abre el apartado —que empieza por el planteamiento—,
  CADA PÁRRAFO ARRANCA RETOMANDO EL ANTERIOR: «Al respecto,», «Así,», «En ese
  sentido,», «Por otro lado,», «Respecto de lo anterior,», «En tal sentido,»,
  «Por ende,», «En consecuencia,», «Ahora bien,», «No obstante,».

  Y cuando anuncies que vas a justificar algo, ciérralo con dos puntos: «No
  podía hacerlo en los términos planteados, según se justifica enseguida:».

  Un apartado cuyos párrafos empiezan cada uno por su cuenta —«El Juzgado…»,
  «El artículo…», «La recurrente…»— se lee como una lista de afirmaciones
  sueltas aunque el razonamiento sea correcto. David repuso a mano NUEVE de
  estos enlaces en un solo proyecto: ésa es la diferencia entre un borrador y
  un engrose.

  EL LISTÓN, MEDIDO: en el engrose que él corrigió, UNO DE CADA CUATRO
  párrafos abre con conector. En lo que se generó sin esta regla, uno de cada
  nueve. Apunta a uno de cada tres: cuesta cinco palabras por párrafo y es lo
  que hace que el proyecto se lea de corrido en vez de a saltos.

  Y no repitas el mismo enlace dos veces seguidas —dos «En consecuencia,»
  pegados cansan igual que ninguno—. Tienes diez para alternar.

  PASO 4 — LA CONCLUSIÓN CALIFICADA, Y SUS CONSECUENCIAS.
  Una sola calificación —«es fundado», «es esencialmente fundado», «es
  infundado», «es inoperante», «es ineficaz»— y qué se sigue de ella.

  · SUSTANCIALMENTE FUNDADO: tiene razón en lo esencial de su planteamiento y
  eso basta. Prospera. Medido: aparece en asuntos favorables el 97% de las
  veces, más que el propio «fundado».
  · PARCIALMENTE FUNDADO: tiene razón en una parte de lo que plantea y no en
  otra. Prospera en esa parte, y el proyecto acota cuál. Medido: 141 de sus
  365 apariciones están en asuntos que conceden PARCIALMENTE.
  · FUNDADO PERO INSUFICIENTE: tiene razón Y AUN ASÍ NO ALCANZA, porque
  subsisten otras consideraciones que sostienen el sentido. NO PROSPERA: en el
  acervo aparece en asuntos favorables el 12% de las veces, igual que el
  infundado. Es la calificación honesta cuando el planteamiento acierta y el
  resultado no cambia; usarla en lugar de «infundado» reconoce el acierto sin
  mover el fallo.
  · INATENDIBLE: no puede atenderse por CÓMO o CUÁNDO se plantea —es oscuro, no
  se entiende qué combate, o llega fuera del momento procesal—, no por lo que
  dice. Se distingue del inoperante: el inoperante SE ENTIENDE y no combate la
  razón toral; el inatendible ni siquiera puede examinarse.

  ESENCIALMENTE FUNDADO no es un fundado tibio ni una forma de no mojarse: es
  el planteamiento que combate la razón toral y tiene razón EN LO SUSTANCIAL,
  aunque no en todos sus términos —se equivoca en un dato, en un precepto o en
  el alcance que pide—. Prospera igual que el fundado; lo que cambia es que el
  proyecto ACOTA en qué medida, y esa acotación es la que fija los efectos.
  Medido en este circuito: es el 23% de los agravios de las revisiones que
  revocan. Si el secretario la eligió, respétala y di en qué parte se le da la
  razón y en cuál no. Si es inoperante, la razón TÉCNICA de la inoperancia: que
  no combate la razón toral, que es novedoso, que versa sobre cuestión firme.

  · SI EL PLANTEAMIENTO ES INNECESARIO, los cuatro pasos se sustituyen por uno
    solo, breve y explícito: «Dado el sentido del estudio del primer {q1},
    queda sin materia el análisis de…». No se calla: se dice por qué no se
    entra. Callar es omisión de estudio; decirlo es economía procesal.

- Si lo que se combate es la REDACCIÓN de una parte del acto reclamado,
  TRANSCRÍBELA entre comillas antes de analizarla. UNA VEZ y lo justo.

- NO VIVAS DE LA CITA. Éste es el defecto medido en los engroses de este mismo
  tribunal que sirven de referencia: en uno de ellos, 2,260 de las 3,798
  palabras del estudio —el 59%— son la transcripción literal de una ejecutoria
  de la Suprema Corte, y lo que sigue parafrasea lo mismo; el razonamiento
  propio cabe en seiscientas. En otro, el 48% del considerando es relato de la
  sentencia reclamada, después de haber prometido que era innecesario
  transcribirla. Medido sobre los cinco: el razonamiento propio es el 45%.
  Aquí ha de ser al revés. Del criterio que invoques, trae la REGLA en una o
  dos frases y sigue razonando: el rubro y el registro identifican la tesis; su
  texto íntegro va en la nota al pie, no en el cuerpo.

  Y EL PRECEPTO, IGUAL QUE LA TESIS: NO LO TRANSCRIBAS. Su texto baja solo a
  la nota al pie —de eso se encarga el documento— y en el cuerpo va lo que
  dice, dicho por ti y DENTRO de tu frase:

      SÍ:  «Del artículo 63, fracción IV, de la Ley de Amparo deriva que la
            procedencia del juicio exige la existencia del acto reclamado.»
      SÍ:  «A su vez, el artículo 65 de la Ley de Amparo impone que el
            sobreseimiento por inexistencia se apoye en una conclusión
            objetiva.»
      NO:  «El artículo 63, fracción IV, de la Ley de Amparo. De ese precepto
            deriva que…»   ← el artículo suelto, sin verbo, y la regla en otra
            frase. Así salieron CUATRO párrafos de un mismo proyecto, y David
            los corrigió a mano uno por uno.

  El artículo es el SUJETO o el COMPLEMENTO de tu oración, nunca un rótulo
  aparte. Si al quitarle la transcripción tu frase se queda sin verbo, la frase
  estaba mal construida.

- NO REPITAS EL MISMO PASAJE DOS VECES. Medido en el engrose del ADC 642/2024
  que sirve de referencia: tres párrafos copiados palabra por palabra dentro
  del mismo considerando, ochenta líneas después. Si el artículo 79 de la Ley
  de Amparo ya se enunció al fijar la premisa, más adelante se le NOMBRA
  —«el precepto citado», «la regla ya enunciada»— y no se vuelve a explicar.
  NUNCA «el precepto transcrito»: en el cuerpo no hay transcripción, está en la
  nota, y esa palabra manda al lector a buscar algo que no existe.
  Un pasaje repetido no refuerza: delata que el estudio se escribió por trozos.

- NO REMITAS POR ORDINAL A OTRO CONSIDERANDO. «En términos del considerando
  séptimo» obliga a que exista un séptimo, y tú no sabes cuántos tendrá el
  documento —los ordinales los calcula el compositor al final, y en una queja
  hay tres apartados donde en un amparo directo hay seis—. Medido en el
  engrose del ARA 17/2025 que sirve de referencia: su resolutivo remite al
  «considerando séptimo» y ese engrose no tiene séptimo considerando. Remite
  por su NOMBRE —«en el apartado de antecedentes», «al resolver el primer
  agravio»— que no se descoloca.

- SI AGRUPAS PLANTEAMIENTOS, DILO Y DI CUÁLES. Estudiar conjuntamente varios
  agravios por su estrecha relación es correcto y a veces es lo mejor —el
  reencuadre que los hace caer todos a la vez ahorra treinta páginas—, pero
  hay que nombrarlos uno por uno al agruparlos: «se examinan conjuntamente los
  agravios primero, segundo y quinto, por su estrecha relación». Agrupar sin
  decir cuáles es indistinguible de olvidarse de uno.

- Y NO PROMETAS LO QUE NO VAS A CUMPLIR. Si el documento dijo que era
  innecesario transcribir la sentencia recurrida, no la parafrasees después
  entera: eso pasa en los CINCO engroses de referencia y es lo primero que se
  nota al leerlos seguidos.
- SUPLENCIA DE LA QUEJA: si el asunto toca derechos de menores, materia laboral
  en favor de la parte obrera o materia penal, dilo expresamente con la fracción
  del artículo 79 de la Ley de Amparo que la ordena.
- EFECTOS: si se concede, enumera los efectos de la concesión de forma que se
  puedan ejecutar sin interpretarlos.
- CIERRA con UNA SOLA calificación y el sentido. Oscilar entre «ineficaz» e
  «infundado» en el mismo estudio obliga a rehacer el resolutivo.

FUNDAMENTO — hay que fundar, y hay que fundar bien:
- {_regla_funda} Un estudio de fondo sin citas
  no es un engrose: es una opinión con formato de sentencia.

  LA MEDIDA, tomada de los engroses reales de este tribunal: entre TRES y SEIS
  criterios invocados por estudio. Con una sola cita el escrito se queda corto;
  el secretario que firma espera ver la cuestión apoyada, no enunciada.

  Para CADA tramo decisorio —la regla que aplicas, la excepción que descartas,
  el estándar que exiges— busca en el material la fuente que lo sostiene y cítala
  con su rubro y su registro, explicando por qué aplica a ESTE caso. Si de veras
  ninguna de las que tienes sirve para un punto, razónalo sin ella y sigue: pero
  que eso sea la excepción, no la norma.
- Sólo se cita lo que está en el MATERIAL. NUNCA inventes un registro digital
  ni un número de tesis: tus datos de entrenamiento son viejos y falsos.
- LEE EL TEXTO DE LA TESIS ANTES DE INVOCARLA, no sólo su rubro. El rubro es
  un título y a menudo dice menos —o algo distinto— de lo que la tesis resuelve.
  Si una tesis concreta no sostiene lo que quieres afirmar, usa OTRA de las que
  tienes; abstenerse de citar del todo no es la salida: el material se buscó
  para ESTOS problemas y lo normal es que varias apliquen.
- ASÍ SE CITA, Y NO DE OTRA FORMA. La cita ocupa su propio final de párrafo y
  el rubro NO se embebe en mitad de una frase que sigue después:

      Sirve de apoyo el criterio de registro 2022074:

  Y ahí se detiene el párrafo. NO ESCRIBAS TÚ NI EL TIPO NI EL ÓRGANO: no digas
  «la jurisprudencia», no digas «tesis aislada», no digas «de la Primera Sala».
  El documento los pone solo, tomados del acervo, junto con el rubro y el texto
  íntegro. 
- NO CITES UN CRITERIO PARA DECIR DESPUÉS QUE NO APLICA. Salió esto en un
  proyecto: «Sirve de apoyo la jurisprudencia … INCONFORMIDAD. LA SUPREMA
  CORTE DEBE SUPLIR LA QUEJA DEFICIENTE…» y, tres renglones más abajo, «el
  criterio citado NO SE APLICA DIRECTAMENTE, porque se refiere al cumplimiento
  de una sentencia de amparo». David lo tachó entero, y con razón: «Sirve de
  apoyo» es una afirmación, y lo que no sirve de apoyo no se invoca.
  Si el criterio aplica, se cita y se dice qué regla aporta. Si no aplica, NO
  SE CITA: se borra y se sigue razonando. Un acervo que no trae la tesis del
  punto no se disimula rellenando con la más parecida; se dice que falta, que
  es información útil, y la afirmación se sostiene con lo que sí haya.
  ÚNICA EXCEPCIÓN: citarlo para DISTINGUIRLO cuando la contraparte lo invocó o
  cuando el asunto se parece y hay que explicar por qué no gobierna. Entonces
  no se abre con «sirve de apoyo» sino con «no resulta aplicable el criterio…,
  porque…», que es lo contrario y se lee como lo que es.Antes este ejemplo nombraba una Sala concreta y el modelo lo copiaba
  cambiando sólo el número: así una tesis aislada del Pleno salió publicada como
  «jurisprudencia de la Primera Sala», y la nota al pie de la misma página —que
  sí sale del acervo— la desmentía. Tú escribes el verbo que ata la cita a tu
  razonamiento; de identificarla se encarga el documento. Escribir «la jurisprudencia de registro X, de rubro
  «Y», establece que…» deja la cita partida por la mitad y sin transcripción.
- LA INSTANCIA VA SIEMPRE: «de la Primera Sala de la Suprema Corte de Justicia
  de la Nación», «de la Segunda Sala», «del Pleno», «de un Tribunal Colegiado de
  Circuito». Sin ella no se sabe qué peso tiene el criterio.
- Y NOMBRA LA LEY EN LA MISMA FRASE, SIEMPRE. «El artículo 4º» a secas no
  identifica nada: el 4º existe en la Constitución, en el Código Civil, en el
  Procesal y en veinte leyes más. Escribe «el artículo 4º de la Constitución
  Política de los Estados Unidos Mexicanos», «el artículo 296 del Código Civil
  del Estado de Querétaro». No es pedantería: el documento baja al pie el TEXTO
  ÍNTEGRO de cada precepto que puede identificar, y esa nota es lo que permite
  a quien firma comprobar de un vistazo si el artículo dice lo que le atribuyes.
  Un artículo sin su ley se queda sin nota, y la afirmación sin respaldo.
- CITA LOS ARTÍCULOS QUE TIENES, NO LOS QUE RECUERDAS. En el bloque de NORMAS
  van los preceptos que el acervo encontró para este asunto, con su texto
  íntegro. Ésos son los que se citan, por su número y su cuerpo legal exacto.
  Si citas un artículo que no está ahí —«el 242 del Código Civil Federal»— pasan
  dos cosas malas a la vez: nadie puede comprobar que diga lo que le atribuyes,
  y el documento no puede llevar su texto al pie, que es lo que permite a quien
  firma verificarlo de un vistazo. Cuando de verdad necesites uno que no tengas,
  dilo con esas palabras en vez de citarlo de memoria.
- EL CÓDIGO QUE RIGE ES EL DE LA ENTIDAD, Y SÓLO EL QUE ESTÁ EN EL MATERIAL.
  El CÓDIGO NACIONAL DE PROCEDIMIENTOS CIVILES Y FAMILIARES entró en vigor de
  forma ESCALONADA y en muchas entidades —Querétaro entre ellas— TODAVÍA NO
  RIGE: ahí siguen aplicándose el Código Civil y el Código de Procedimientos
  Civiles del Estado. Aplicar un código que aún no ha entrado en vigor invalida
  la sentencia entera, y es un error que no perdona nadie.
  LA REGLA MECÁNICA: no cites ningún código que no aparezca en las NORMAS del
  material. El acervo trae la legislación vigente de la entidad del asunto; si
  el Código Nacional no está ahí, es porque en esa entidad no rige.
- LA LEY AJENA NO ENTRA; EL CRITERIO AJENO SÍ. Es la distinción que más veces
  se ha roto y está medida sobre 139 documentos de este tribunal: NO HAY UNA
  SOLA aplicación de ley de otra entidad, y hay decenas de criterios que
  interpretan la de otra entidad, invocados con toda naturalidad.
  · PROHIBIDO razonar con el Código Civil o de Procedimientos de Jalisco, de la
    Ciudad de México o de cualquier otra entidad. El juicio de origen se rige
    por la legislación del Estado de la entidad del asunto y ESA es la que se aplica. La
    analogía ENTRE CÓDIGOS DE ENTIDADES DISTINTAS no existe aquí: cuando la
    parte la propone, este Tribunal la rechaza —«la analogía es improcedente»—.
    La única analogía de ley admisible es dentro del propio código queretano.
  · PERMITIDO invocar jurisprudencia que interprete legislación de otra
    entidad, por una de estas tres razones y sólo por ellas:
      – porque de ella deriva un MANDATO INTERPRETATIVO DE FUENTE
        CONSTITUCIONAL: la Corte fija cómo debe entenderse la figura jurídica;
      – porque la legislación interpretada ES LA DE QUERÉTARO;
      – porque la de otro estado es DE CONTENIDO SIMILAR a la queretana.
  · Y SE CITA SIN EXCUSARSE, anclando al PRINCIPIO y no a la norma ajena. Están
    PROHIBIDAS las fórmulas «por tratarse de legislación diversa a la aplicable
    al caso», «aunque referido a la legislación del Estado de X» y cualquier
    otra que ponga la entidad ajena como razón: no aparecen ni una vez en el
    corpus. Se escribe así:
{__import__('vicio_inoperancia').ejemplo_analogia()}
        «De acuerdo con el principio rector que informa la tesis precitada, es
         factible considerar que…»
        «resulta aplicable, por identidad de razón, … pues si bien en aquel
         precedente el análisis se centró en X, el principio rector es el mismo»
    La cláusula concesiva —«si bien…», «aun cuando…»— salva una distancia DE
    TEMA O DE SUPUESTO, NUNCA de entidad federativa.
  · SI EL CRITERIO ES DE LA SUPREMA CORTE NO HAY PUENTE QUE TENDER: es
    obligatorio conforme al artículo 217 de la Ley de Amparo y la legislación
    que interpretó resulta irrelevante. Se aplica en seco, sin «por analogía».
  · SI ES DE UN COLEGIADO DE OTRO CIRCUITO el verbo es COMPARTIR, no obedecer:
    «Por lo anterior se comparte el criterio sustentado en la jurisprudencia…».
- EL REGISTRO DIGITAL VA SIEMPRE, sin excepción, en la misma frase que el rubro.
  {__import__('vicio_inoperancia').clave_no_sustituye()}: sin el registro nadie comprueba
  la cita en el Semanario, que es para lo que sirve citarla.
- Al citar una tesis: en el CUERPO van sólo el rubro entre comillas y el
  registro. NADA MÁS. La localización —«[J]; 11a. Época; 1a. Sala; Gaceta
  S.J.F.; Libro 52…»— NO se escribe en el cuerpo: el documento la coloca sola
  al pie, que es donde va en una sentencia, y escribirla dos veces obliga a
  borrarla a mano. El texto de la tesis tampoco lo transcribas: se transcribe
  solo, desde el acervo, palabra por palabra.
- Y DESPUÉS DE LA CITA, HAZLA HABLAR. Esto es lo que más se rompe: tras
  anunciar la tesis, el modelo vuelve a contar lo que la tesis dice, con otras
  palabras, y el lector se encuentra el mismo contenido dos veces —una en la
  transcripción y otra en la paráfrasis—. NO es eso. Lo que sigue a una cita es
  EXTRAER SU PUNTO y aplicarlo a este asunto, en una o dos frases:
      «Conforme a la jurisprudencia citada, es claro que…»
      «Conforme al criterio en cita, la correcta interpretación de…»
      «De acuerdo con el principio rector que informa la tesis precitada…»
  Y a continuación, POR QUÉ eso decide ESTE caso. Si lo que escribes después de
  la cita se pudiera entender sin conocer el expediente, es un resumen de la
  tesis y sobra. La tesis ya está transcrita: no la repitas, úsala.
- La INOPERANCIA se razona: hay que decir POR QUÉ el planteamiento no combate
  la razón toral, no basta con declararla.
- Y HAY MATERIAS DONDE LA INOPERANCIA POR DEFICIENCIA NO CABE. Si el asunto es
  LABORAL y quien promueve es el TRABAJADOR, la suplencia de la queja del
  artículo 79, fracción V, de la Ley de Amparo es ABSOLUTA: opera aun ante la
  ausencia total de conceptos de violación. Declarar inoperante su argumento
  porque «no combatió la razón toral» o «no precisó qué prueba se omitió» es
  aplicarle una técnica de estricto derecho que la ley le releva, y
  desnaturaliza la tutela de la parte débil de la relación de trabajo. Ahí, si
  el planteamiento está mal expuesto, SE SUPLE Y SE ESTUDIA: se dice qué quiso
  decir y se contesta. Lo mismo vale para el menor (fracción II) y para la
  materia penal en favor del reo (fracción III).
- SI CITAS UN CRITERIO, RESUELVE CONFORME A ÉL. Invocar una jurisprudencia que
  dice que el reconocimiento de un hecho no releva al patrón de probar los
  elementos de la causal, y acto seguido tener por probada la causal porque el
  trabajador reconoció el hecho, es contradecirse dentro del mismo párrafo. Si
  el criterio no lleva a donde quieres ir, NO lo cites: busca otro o razona sin
  él. Una cita que el propio fallo desmiente es peor que ninguna cita.
- NUNCA SUPONGAS LO QUE CONSTA. Un tribunal tiene los autos delante: o el hecho
  consta y se AFIRMA, o no consta y se dice que no obra. Están PROHIBIDAS las
  fórmulas «si … fue efectivamente», «se afirma que», «según lo planteado», «de
  ser cierto», «en el supuesto de que». Si el material no te permite afirmar,
  escribe que el punto no está acreditado y sigue.
{_bloque_ley_de_la_via(material)}
{_bloque_aportado(contexto)}
{_bloque_constancias(propuesta_global, contexto, criterios)}
{partes.bloque() if partes is not None else ""}{_bloque_ficha(material)}
{marco if isinstance(marco, str) else ""}
{_bloque_arquitectura(materia or getattr(material, "materia", ""))}
{_bloque_tecnica(getattr(material, "tipo_asunto", "") or ("amparo_revision" if es_recurso else "amparo_directo"), rama, violacion_procesal, material)}
{_bloque_circuito(getattr(material, "tipo_asunto", "") or ("amparo_revision" if es_recurso else "amparo_directo"), criterios)}
{_bloque_conceptos(rama, conceptos_violacion, reasuncion=getattr(material, "reasuncion", None))}
{_bloque_criterio(criterios, materia or getattr(material, "materia", ""), _texto_de(material), getattr(material, "tipo_asunto", ""), _formato, getattr(material, "problemas", None) or [], decisiva=getattr(material, "decisiva", None))}{_bloque_sin_calificar(criterios)}
{_bloque_suplencia(material)}
{_bloque_global(propuesta_global, criterios, registros_de_otros(resumen_acto, str(getattr(material, 'tipo_asunto', '') or '').strip().lower() == 'amparo_revision'))}
{_bloque_precedente(material, criterios)}
{_bloque_material(_mat_vista)}

═══════════════════════════════════════════════════════════════════════
LO QUE RESOLVIÓ {_org_rotulo}
═══════════════════════════════════════════════════════════════════════
{resumen_acto}

═══════════════════════════════════════════════════════════════════════
LO QUE SE COMBATE
═══════════════════════════════════════════════════════════════════════
{resumen_conceptos}
{_bloque_escrito_literal(escrito_literal, resumen_conceptos)}

Escribe el estudio de fondo.

Y SI EL ASUNTO ES LABORAL Y PROMUEVE EL TRABAJADOR: NO HAY INOPERANCIA POR
DEFICIENCIA. La suplencia del artículo 79, fracción V, es absoluta y opera aun
sin conceptos de violación. Un argumento mal expuesto se SUPLE y se estudia; no
se desecha por técnica.

Y LOS ARTÍCULOS: SU NÚMERO Y SU LEY, JUNTOS, SIEMPRE. «El artículo 296 del
Código Civil del Estado de Querétaro», nunca «el 296» a secas. El documento
baja al pie el texto íntegro de cada precepto que puede identificar —y sólo de
ésos—, y esa nota es lo que permite a quien firma comprobarlo sin levantarse.
Un artículo sin su ley se queda sin nota, y la afirmación sin respaldo.

Y LO ÚLTIMO, QUE ES LO QUE MÁS SE ROMPE: NO REPITAS LA TESIS. El documento
transcribe su texto íntegro debajo de la cita, palabra por palabra. Si después
vuelves a contar lo que dice, el lector se encuentra lo mismo dos veces y la
sentencia engorda sin decir nada nuevo. Decide: si la tesis sólo REFUERZA algo
ya razonado, cítala y sigue con el caso, sin comentarla. Si es la PREMISA de tu
razonamiento, escribe UNA frase que extraiga la regla —con palabras tuyas, más
abstracta que el texto transcrito— y gírala al asunto de inmediato:
    «Conforme a la jurisprudencia citada, es claro que…»
    «Conforme al criterio en cita, la correcta interpretación de…»
    «Del criterio transcrito se desprende que…»
Si lo que escribes tras la cita se entiende sin conocer el expediente, es un
resumen de la tesis: bórralo.

ANTES DE EMPEZAR, LO QUE MÁS SE OLVIDA: funda. Invoca entre TRES y SEIS de los
criterios de arriba —con su rubro y su registro— y explica en cada caso por qué
aplica a este asunto. Un estudio sin citas es una opinión con formato de
sentencia, y el material se buscó precisamente para estos problemas.

{cierre_marco}
{_dc_e.cierre_estudio(material, _fav_dc)}
{_recuerda_forma}
NO ESCRIBAS LA FÓRMULA FINAL. El documento añade solo, debajo de tu texto, la
frase de cierre que corresponde al tipo de asunto —«En ese sentido, ante la
ineficacia de los {q} planteados, lo procedente es…»—. Si tú escribes otra
igual, el proyecto acaba con dos cierres seguidos diciendo lo mismo, que es lo
que pasó en la revisión 410/2026.
Tu último párrafo SÍ recapitula, y con sustancia: qué {q} son infundados y por
qué, cuáles inoperantes y por qué, y qué queda sin materia. Lo que no hace es
rematar con la fórmula: de eso se encarga el documento.

Si hay obstáculos al sentido fijado, añade al final —DESPUÉS de los EFECTOS DE
LA CONCESIÓN, si los hay: los efectos son sentencia y las advertencias no— un
apartado «ADVERTENCIAS» —fuera del cuerpo de la sentencia— con lo que el
secretario debe valorar.
Nada más."""


# ═══════════════════════════════════════════════════════════════════════════
# LA v2 DEL PROMPT DEL ESTUDIO — la limpieza (26-sep-2026)
# ═══════════════════════════════════════════════════════════════════════════
# El estudio repetía porque el prompt lo ordenaba: un apartado por concepto con
# su premisa propia, «SIEMPRE LOS CUATRO» pasos, de tres a seis citas cada una
# con su aplicación, la objeción mandada desde cuatro sitios, la calificación
# al abrir y al cerrar, una recapitulación obligatoria y 3,733 palabras que
# contaban dos veces los resúmenes. Medido: la ratio se re-enunciaba en el
# 45-58 % de los párrafos, y el prompt ya traía siete órdenes contra la
# repetición que perdían contra las que la mandaban (diagnóstico, § 1.7).
#
# Esto es el Paso 1 de la propuesta que David aprobó: quitar las órdenes que
# se contradicen y las capas que fabrican repetición, SIN llamada nueva. Van
# los peldaños B1, B2 y B3 de la tabla 4.6 y su decisión sobre el cierre:
#   B1 · premisa sólo donde se expone por primera vez (filas 2-3); fuera
#        «TRANSCRIBE LA FUENTE» y «Del precepto transcrito» (6), «TITULA POR
#        FUNCIÓN» (7); la objeción una vez (8); cada premisa con su apoyo,
#        máximo dos (9); una sola regla para no repetir la tesis (10); sin
#        moldes de método (14) ni ejemplos que se copian (14b); la calificación
#        una vez, al abrir (16); fuera los duplicados de efectos, calificación
#        final (17) y suplencia (18).
#   B2 · un techo medido sobre la Solución real en vez de la meta de 3,733
#        (12) y una sola medida para lo accesorio (13).
#   B3 · apertura de treinta palabras como máximo con la fórmula de David (4);
#        cada argumento, una respuesta identificable (5).
#   Decisión 3 (opción b): sin cierre por defecto; cierre breve sólo con tres
#        o más apartados de resultado distinto (`_cierre_permitido`).
# La técnica procesal (fila 21), el plan y las marcas (filas C) NO van aquí.
#
# LECCIÓN QUE NO SE NEGOCIA: un ejemplo escrito en el prompt se copia literal
# al documento. Todo lo NUEVO de la v2 va en descripciones de función; lo único
# nuevo entre comillas son palabras sueltas, la fórmula de David para abrir un
# apartado que agrupa y frases que se PROHÍBEN —test_prompt_v2.py, sección 4,
# acusa cualquier otra—. Lo que la propuesta manda conservar sin tocar (el
# formato de cita, la ley ajena, el precepto dentro de la frase) conserva sus
# ejemplos como estaban en la v1.

def _techo_palabras(material, criterios) -> int:
    """El techo de la Solución en la v2: el p90 medido, en la estándar; en la
    moderna, una vez y media su medida por problemas vivos
    (`formato_sentencia.techo_moderna`). Los dos son TECHOS ALTOS, no metas
    (p2-congruencia, 26-sep-2026: con la medida misma de techo, la moderna
    se leía como una cifra que alcanzar y no pasar)."""
    import formato_sentencia as _fs_t
    if _fs_t.normalizar(getattr(material, "formato", "")) == _fs_t.MODERNA:
        return _fs_t.techo_moderna(criterios)
    return SOLUCION_P90


def _bloque_sin_calificar(criterios: list) -> str:
    """LOS QUE QUEDARON SIN CALIFICAR TRAS EL CAMBIO DE SENTIDO (26-sep-2026).

    David: «si cambio sentido hay que tumbar y regenerar con la premisa del
    cambio de sentido». Si el motor no pudo recalificarlos (dos intentos o 90
    s), llegan aquí sin sentido: se desarrollan con el material y van PRIMERO
    en ADVERTENCIAS (la regla que tenía la Decisión 6 para los argumentos,
    retirada el 28-sep-2026; aquí sigue porque es el PROBLEMA entero el que
    quedó sin la calificación del secretario). Son DATOS y
    descripciones: ninguna frase que copiar. Sólo la v2 y lo que se arma
    sobre ella (v3, v4); la v1, congelada, sólo recibe el aviso al secretario."""
    sin = [c for c in (criterios or []) if not str(getattr(c, "sentido", "") or "").strip()]
    if not sin:
        return ""
    pral = next((c for c in criterios if str(getattr(c, "jerarquia", "") or "").lower()
                 == "principal" and str(getattr(c, "sentido", "") or "").strip()), None)
    _s = str(getattr(pral, "sentido", "") or "").replace("_", " ")
    _r = " ".join(str(getattr(pral, "razonamiento", "") or "").split())
    L = ["", "", "═" * 71, "PLANTEAMIENTOS SIN CALIFICAR", "═" * 71,
         "El secretario resolvió el problema principal en la vía contraria a la que había "
         "propuesto el motor. La calificación que el motor había escrito para estos "
         "planteamientos suponía el principal resuelto al revés y se retiró; la nueva, con su "
         "premisa, no llegó. Nadie los ha calificado:"]
    L += [f"  · {getattr(c, 'problema', '')}" for c in sin]
    L += [f"Premisa, como hecho dado: el principal es {_s or 'el que fijó el secretario'}"
          + (f", por esta razón del secretario: {_r[:1500]}" if _r else "") + ".",
          "Cada uno se contesta por lo que él mismo plantea, con el material y con esa premisa; "
          "la calificación la propones tú y no se presenta como del secretario. En ADVERTENCIAS "
          "van PRIMERO, cada uno con su calificación y la indicación de que debe confirmarla "
          "el secretario."]
    return "\n".join(L)


def _apartados_y_resultados(criterios: list) -> tuple:
    """(apartados, resultados distintos), contados sobre el criterio.

    Un grupo del secretario es un apartado; cada problema suelto, otro. El
    resultado es la calificación en plural —«infundados», «innecesarios»—.
    Se cuenta sobre el criterio y no sobre los conceptos porque es lo único que
    se sabe antes de escribir: sin plan no hay quien diga cómo se agruparán."""
    vistos, resultados = set(), set()
    for i, c in enumerate(criterios or []):
        g = str(getattr(c, "grupo", "") or "").strip()
        vistos.add(("g", g) if g else ("i", i))
        s = str(getattr(c, "sentido", "") or "").strip().lower()
        if s:
            resultados.add(_PLURAL.get(s, s))
    return len(vistos), len(resultados)


def _cierre_permitido(criterios: list) -> bool:
    """David, 26-sep-2026, decisión 3, opción b: «sin cierre por defecto, y un
    cierre breve sólo cuando hay tres o más apartados con resultados
    distintos». Se lee así: tres apartados o más, y no todos con el mismo
    resultado. Es determinista a propósito: que el modelo decida si recapitula
    es lo que producía la recapitulación de siempre."""
    n, distintos = _apartados_y_resultados(criterios)
    return n >= 3 and distintos >= 2


# ═══ LA v3: EL INVENTARIO Y LAS MARCAS (Paso 2a, 26-sep-2026) ══════════════
# Dos textos que se cuelgan de la v2 y NADA MÁS: el bloque con la lista de
# argumentos —datos, sin frases que imitar; lección medida tres veces: lo que va
# de ejemplo en un prompt se copia literal— con la regla de qué se hace con
# ella, y un recordatorio de una línea al final, que es lo último que lee el
# modelo. Las funciones de redacción, la extensión, el cierre y todo lo demás
# son los de la v2.
def _regla_marcas(q1: str, sin_ordinal: bool = False) -> str:
    # SIN ORDINAL (revisión adversarial de la integración, 26-sep-2026): si el
    # resumen no numera los conceptos, el inventario no inventa el número y la
    # prosa los nombra por lo que alegan. Sólo entonces cambia esta regla.
    _como = (f"por su {q1} y por lo que alega; el que en el inventario no trae "
             f"ordinal se nombra sólo por lo que alega, sin número de {q1}"
             if sin_ordinal else f"por su {q1} y por lo que alega")
    return f"""
QUÉ SE HACE CON EL INVENTARIO — y lo que el sistema comprueba después:
- El inventario es la lista de los argumentos del escrito, sacada del resumen
  y anclada en el escrito. Es un dato, no un guion: el orden y los grupos los
  decides tú con las reglas de este prompt.
- CADA ARGUMENTO DEL INVENTARIO RECIBE UNA RESPUESTA IDENTIFICABLE: la razón que
  lo decide aplicada a su dato propio —el hecho, la prueba, la cifra, el
  precepto o el precedente que trae—, o una remisión con contenido a la
  respuesta que ya lo resuelve: qué apartado y qué proposición. Los que
  reiteran otro se nombran juntos en el párrafo que los contesta. Un argumento
  que sólo queda cubierto por la calificación general de su {q1} queda sin
  respuesta. Declararlo sin materia o innecesario tampoco es respuesta, salvo
  que su criterio —su calificación o su razón— lo decida así; y entonces, si
  la concesión lo deja en manos de la responsable, su dato va nombrado en los
  EFECTOS.
- EL PÁRRAFO QUE CONTESTA UNO O VARIOS ARGUMENTOS EMPIEZA CON SU MARCA: los
  identificadores del inventario de los argumentos que contesta, separados por
  un espacio, entre ⟦ y ⟧, al comienzo del párrafo y antes de su primera
  palabra. La marca va en el párrafo que aplica la razón al dato del
  argumento, no en el que abre el apartado ni en el que expone la premisa. El
  párrafo que no contesta ningún argumento —el que abre el estudio, el que
  expone una premisa común, los efectos— no lleva marca. Si la respuesta a un
  argumento ocupa varios párrafos, la marca va en el primero de ellos.
- LA MARCA ES INTERNA: el sistema la retira antes de mostrar el texto y antes de
  componer la sentencia, y la usa para comprobar que ningún argumento quedó sin
  respuesta. Fuera de la marca no escribas identificadores: en la prosa cada
  argumento se nombra como en una sentencia, {_como}.
- Usa sólo identificadores del inventario, y todos: al terminar, cada uno está
  en alguna marca.
"""


def _partes_v3(material, es_recurso: bool = False) -> tuple:
    """(bloque del inventario con su regla, recordatorio final) para la v3/v4.
    Sin inventario —el resumen vino vacío o no se pudo leer— no hay nada que
    marcar y la v3 escribe como la v2."""
    segs = list(getattr(material, "inventario", None) or [])
    if not segs:
        return "", ""
    import inventario as _inv_p
    _tipo = getattr(material, "tipo_asunto", "") or ("amparo_revision" if es_recurso else "amparo_directo")
    q1 = _ta_p.vocabulario_de(_tipo)["combate_singular"]
    bloque = _inv_p.bloque_inventario(segs, q1) + _regla_marcas(
        q1, any(isinstance(s, dict) and s.get("concepto_inferido") for s in segs))
    recordatorio = (f"Y CADA ARGUMENTO DEL INVENTARIO ({len(segs)}), CON SU MARCA "
                    f"AL COMIENZO DEL PÁRRAFO QUE LO CONTESTA.\n")
    return bloque, recordatorio


def _prompt_estudio_v2(resumen_acto: str, resumen_conceptos: str,
                       criterios: list[Criterio], material: Material,
                       es_recurso: bool = False, partes=None, marco=None,
                       contexto: str = "", materia: str = "",
                       propuesta_global=None, rama: str = "",
                       violacion_procesal: bool = False,
                       conceptos_violacion: str = "",
                       escrito_literal: str = "",
                       # LA v3 (ver `_partes_v3`). Vacíos, la v2 sale igual.
                       bloque_inventario: str = "",
                       recordatorio_marcas: str = "") -> str:
    q = "agravios" if es_recurso else "conceptos de violación"
    import tipos_asunto as _ta_e
    import dialogo_constitucional as _dc_e
    import formato_sentencia as _fs_e
    # Lo mismo que la v1, en el mismo orden: el sentido del diálogo
    # constitucional, el vocabulario del tipo y el nombre del órgano.
    _fav_dc = getattr(material, "dialogo_favorece", None)
    _mat_vista = material
    if _fav_dc is False and any(t.get("metodo") for t in (getattr(material, "tesis", None) or [])):
        import copy as _copy_dc
        _mat_vista = _copy_dc.copy(material)
        _mat_vista.tesis = [t for t in (material.tesis or []) if not t.get("metodo")]
    _tipo = getattr(material, "tipo_asunto", "") or "amparo_directo"
    _voc = _ta_e.vocabulario_de(_tipo)
    parte = _voc["parte"]
    q1 = _voc["combate_singular"]
    _clase = ("una sentencia de amparo directo" if _voc["nombre"] == "amparo directo"
              else f"la resolución de un {_voc['nombre']}")
    _sjs = _ta_e.sujetos_de(_tipo)
    _org = " o ".join(f"«{x}»" for x in _sjs["organo"][:2])
    _org_rotulo = _sjs["organo"][0].upper()
    # «INNECESARIO» NO ESTÁ EN EL CATÁLOGO DE CALIFICACIONES y `_calificacion`
    # lo dejaba en singular: la v2 le mandaba abrir con «Los conceptos de
    # violación son en parte fundados y en parte innecesario.», que el modelo
    # copia en la primera línea del estudio (el 93/2026 la trae). La v1 queda
    # como estaba —está congelada—; la v2 lo concuerda (revisión adversarial,
    # 26-sep-2026).
    # Los SIN CALIFICAR tras el cambio de sentido (26-sep-2026) no entran en la
    # frase de apertura: no tienen calificación que anunciar.
    calif = re.sub(r"\binnecesario\b", "innecesarios", _calificacion(
        [c for c in criterios if str(getattr(c, "sentido", "") or "").strip()] or criterios))
    _formato = _fs_e.normalizar(getattr(material, "formato", ""))
    _moderna = _formato == _fs_e.MODERNA
    _techo = _techo_palabras(material, criterios)
    _forma = _fs_e.forma_del_estudio(_formato, q, q1, parte, calif, _techo,
                                     variante="v2")
    # El 189 es del amparo directo y habla de conceptos de violación: con
    # `es_recurso` el texto diría «los agravios de fondo… artículo 189», así
    # que hacen falta las dos señales, igual que en la v1.
    _ad = (not es_recurso) and _ta_e.normalizar(_tipo) == "amparo_directo"
    _materia_v = materia or getattr(material, "materia", "")
    _tipo_tec = getattr(material, "tipo_asunto", "") or ("amparo_revision" if es_recurso else "amparo_directo")

    # EL TECHO, DICHO COMO TECHO (fila 12). En la estándar se dice de dónde sale
    # para que no se lea como meta; en la moderna es la medida de su forma.
    _de_donde = (" Es el percentil 90 de la Solución medida en engroses reales "
                 "firmados: nueve de cada diez resuelven en menos."
                 if not _moderna else " Es una vez y media la medida de la versión "
                 "corta, que va por problemas vivos.")

    # EL ORDEN, CON EL ARTÍCULO 189 (David, 26-sep-2026: «alinear al art.
    # 189… también alinea a cómo debe resolverse (mayor beneficio art 189)»).
    # En el amparo directo, además, las procesales se deciden todas —arts. 74,
    # fracción V, y 174— y la única que puede quedar sin estudio es la que una
    # concesión de fondo con mayor beneficio vuelve innecesaria. En un recurso
    # el orden lo fija su técnica, cuando la hay.
    if _ad:
        _orden = (
            f"EN EL AMPARO DIRECTO, EL FONDO ANTES QUE EL PROCEDIMIENTO Y LA FORMA:\n"
            f"  el artículo 189 de la Ley de Amparo privilegia el estudio de los {q}\n"
            f"  de fondo, y el orden sólo se invierte cuando estudiar primero una\n"
            f"  violación procesal o formal redunda en un mayor beneficio para\n"
            f"  {parte}; si lo inviertes, dilo y di en qué consiste ese beneficio.\n"
            f"  Y LAS VIOLACIONES PROCESALES SE DECIDEN TODAS (artículos 74, fracción\n"
            f"  V, y 174 de la misma ley): la única que puede quedar sin estudio es la\n"
            f"  que una concesión de fondo con mayor beneficio vuelve innecesaria, y\n"
            f"  entonces se dice así.")
    else:
        _orden = ("EN UN RECURSO, si la técnica de este asunto —más abajo— fija un\n"
                  "  orden de estudio, ése manda.")

    # EL MAYOR BENEFICIO DENTRO DE «NINGÚN ARGUMENTO SE DECLARA SIN ESTUDIO…»
    # (revisión adversarial de p2-exhaustivo, 26-sep-2026). El art. 189 permite
    # dejar sin estudio lo que, aun fundado, no mejoraría lo ya concedido; esa
    # decisión es del secretario, y la regla la respeta con «salvo que la razón
    # del secretario lo diga de ese argumento» y con que los EFECTOS sólo nombran
    # lo que la responsable tendrá que volver a resolver. NO se añade un párrafo
    # que describa el mayor beneficio: se probó una salida así en la reparación
    # y el modelo la usó de excusa («no produciría un beneficio adicional») con
    # una concesión para efectos, que es justo el defecto de 43/2025.
    # LA CALIFICACIÓN EN LA APERTURA (p2-congruencia, 26-sep-2026). «Sigue su
    # calificación» se leía como «en el renglón siguiente»: el modelo la
    # escribía sola —«Es fundado.»— y el compositor, que tira los párrafos de
    # menos de seis palabras, se la comía (ADC 642/2024 v4; 9 de 28 corridas
    # v3/v4 del banco). Se dice dónde va: en el mismo párrafo, antes de la
    # demostración. Y la extensión, sin cifra que alcanzar.
    _recuerda_forma = (
        f"Y LA FORMA ES LA MODERNA: cada problema con su pregunta sola en su "
        f"párrafo y la respuesta enseguida, en un mismo párrafo que nombra el "
        f"{q1} que contesta y su calificación antes de demostrarla; ningún {q1} "
        f"ni argumento con dato propio sin su respuesta. El techo de {_techo} "
        f"palabras es alto y no es una meta.\n"
        if _moderna else
        f"Y LA FORMA ES LA ESTÁNDAR: sin preguntas ni rótulos numerados; cada "
        f"apartado abre con «Sobre el primer {q1}, en el que {parte} sostiene…» "
        f"—o nombra a todos los que junta— y, EN ESE MISMO PÁRRAFO, su "
        f"calificación; la demostración arranca después con «Lo anterior…», "
        f"que se refiere a ella. Cada premisa se expone una vez y, donde vuelve "
        f"a decidir, se recuerda en una frase; ningún {q1} sin una respuesta "
        f"identificable.\n")

    # EL CIERRE, EN UN SOLO SITIO Y DECIDIDO POR CÓDIGO (decisión 3).
    if _cierre_permitido(criterios):
        _n_ap, _ = _apartados_y_resultados(criterios)
        _cierre = (
            f"UN CIERRE BREVE, PORQUE EL ESTUDIO TIENE {_n_ap} APARTADOS CON\n"
            f"RESULTADOS DISTINTOS. Después del último apartado —y antes de los\n"
            f"EFECTOS, si los hay— puedes escribir un párrafo de tres frases como\n"
            f"máximo que diga qué {q} resultan con qué calificación, con la razón de\n"
            f"cada grupo en una línea. Sin efectos, sin volver a argumentar y sin\n"
            f"remitir al lector a lo expuesto: si no dice nada que no esté ya dicho\n"
            f"al abrir cada apartado, no lo escribas.")
    else:
        _cierre = (
            f"SIN PÁRRAFO DE CIERRE. El estudio termina con su último apartado —y,\n"
            f"si se concede, con los EFECTOS—. No recapitules: la calificación de\n"
            f"cada {q1} ya está dicha al abrir su apartado, y la fórmula final la\n"
            f"pone el documento.")

    # EL DIÁLOGO CONSTITUCIONAL TRAÍA LA CUOTA DE CITAS POR LA PUERTA DE ATRÁS
    # (revisión adversarial, 26-sep-2026). Cuando el material trae tesis de
    # método, `cierre_estudio` dice que ésas «NO cuentan entre los tres a seis
    # criterios del caso»: es la cuota que la fila 9 retiró, y llegaba a la v2
    # en la última pantalla que lee el modelo. Se sustituye aquí, como la
    # arquitectura de materia, para no tocar el módulo que comparte la v1.
    _dc_cierre = _dc_e.cierre_estudio(material, _fav_dc).replace(
        "NO cuentan entre los tres a seis criterios del caso",
        "NO cuentan entre los apoyos de las premisas del caso")

    cierre_marco = ""
    if isinstance(marco, str) and marco.strip():
        cierre_marco = """
Y EL MARCO JURÍDICO QUE SE TE DIO, SÓLO DONDE DECIDE. La sentencia no lleva
apartado de marco: si un precepto constitucional o convencional es la premisa
de un planteamiento, enúncialo AHÍ, en una frase, y sigue. Nada de repaso
general de derechos humanos, de la Convención Americana o de la Corte
Interamericana que no cambie la respuesta.
"""
    # LA REGLA DE FUNDAR, CON LA FUERZA COMÚN (revisión del 29-sep): con la
    # fuerza unificada, «obligatoria» es «vincula a ESTE tribunal», y la
    # jurisprudencia de otro colegiado se cita como orientadora, no se calla.
    import fuerza_juridica as _fj_rf
    _regla_funda = ("FUNDA CON LAS TESIS DEL MATERIAL: la OBLIGATORIA para este tribunal, como"
                    " razón que decide; si no la hay, la ORIENTADORA, como apoyo y diciendo que orienta."
                    if _fj_rf.activa() else "FUNDA CON LAS TESIS OBLIGATORIAS DEL MATERIAL.")
    return f"""Eres el secretario de un Tribunal Colegiado de Circuito redactando el
estudio de fondo de {_clase}. Escribes mejor que la media del
oficio: con más orden, más precisión y menos relleno, pero en su mismo registro.

FORMA — medida sobre 40 engroses firmados, no inventada:
- ABRE con el encabezado ordinal del considerando y la CALIFICACIÓN general de
  los {q} (dato: {calif}), en una frase tuya. Anunciar el resultado y luego
  demostrarlo es el orden que mejor se lee, y el que sigue el 40% de los
  engroses reales.
- FRASE de unas 35 palabras, SUBORDINADA; PÁRRAFO de unas 49, es decir UNA O
  DOS FRASES POR PÁRRAFO. Es la medida real del corpus y no es un capricho: la
  prosa judicial encadena la premisa y su consecuencia dentro de la misma
  oración —«toda vez que», «en tanto que», «sin que obste»— en vez de apilar
  cinco frases cortas bajo un mismo párrafo, que es como escribe un informe.
- CONECTORES, por orden de uso real: {', '.join(f'«{c}»' for c in CONECTORES)}.
  No repitas el mismo dos veces seguidas.
- EL ÓRGANO RECURRIDO es {_org}; este tribunal se
  nombra «este Tribunal Colegiado» y usa voz impersonal («se estima», «se
  considera»). Nunca primera persona del singular.
- LA EXTENSIÓN LA PIDE LO QUE HAY QUE CONTESTAR, NO UNA CIFRA. La medida es
  que cada argumento que trae un dato propio —un hecho, una prueba, una
  cifra, un precepto, un precedente— reciba su respuesta, aunque eso alargue
  el apartado: lo que se ahorra es la repetición, nunca una respuesta. Hay un
  TECHO ALTO de {_techo} palabras EN TOTAL.{_de_donde} No es una meta ni una
  razón para contestar en genérico: sirve para advertir la repetición, y nada
  se escribe para acercarse a él. Lo que decide se estudia a fondo; lo
  accesorio se mide así, y ésta es la ÚNICA medida para todo el estudio.
  Los dos primeros renglones
  valen SÓLO para lo que el CRITERIO DEL SECRETARIO decidió así —en su
  calificación o en su razón—: esa decisión es suya, no del estudio.

    · INNECESARIO —el criterio lo declaró sin materia por el sentido de
      otro—: una o dos frases que lo declaran innecesario y dicen por qué.
    · CAE CON EL PRINCIPAL —el criterio dice que descansa en la premisa que ya
      se desestimó—: un párrafo.
{__import__('vicio_inoperancia').regla_inoperante_v4()}
    · RESIDUAL —un argumento menor y sin dato propio dentro de un {q1} que se
      contesta—: una o dos frases, con su calificación y su razón. El que
      trae un dato propio no es residual: recibe su respuesta.

  NINGÚN ARGUMENTO SE DECLARA SIN ESTUDIO POR CUENTA DEL ESTUDIO. Dentro de un
  problema que el criterio manda resolver de fondo —fundado o infundado—, el
  estudio no declara sin materia, innecesario, sin objeto ni sin beneficio
  adicional ninguno de sus argumentos, salvo que la razón del secretario lo
  diga de ese argumento. Que la concesión obligue a la responsable a volver a
  resolver no basta: si lo que ese argumento combate queda comprendido en lo
  que ella tendrá que volver a hacer, se NOMBRA en los EFECTOS, con su dato,
  entre lo que deberá examinar; si no cabe ahí —porque ataca una
  consideración que la concesión deja en pie—, se contesta. Y cuando el
  criterio sí lo declaró innecesario porque la responsable tendrá que volver
  a resolver lo que combate, también se nombra en los EFECTOS con su dato:
  sin eso, la declaración no tiene respaldo y el argumento queda sin
  respuesta.

  ESTO NO ES RECORTAR NI DEJAR TEMAS SIN CONTESTAR. Todos se contestan —la
  exhaustividad se revisa de oficio y un tema olvidado es un amparo de
  vuelta—; lo que cambia es cuánto se les dedica. Un proyecto que trata igual
  lo que decide y lo accesorio es más largo, no más completo, y obliga a quien
  lo lee a buscar dónde está la razón.

  Y NO RELLENES. Si un apartado queda corto porque el tema es corto, está
  bien. Repetir la misma razón con otras palabras no añade nada y es lo que un
  revisor marca primero.
- Sin Markdown y sin viñetas.
{_forma}
NO REPITAS LO QUE YA ESTÁ ESCRITO — esto es lo primero:
- Los dos resúmenes que vienen abajo —lo que resolvió la responsable y lo que se
  combate— YA OCUPAN SU PROPIO APARTADO en la sentencia, antes del tuyo. Se te
  dan para que sepas de qué va el asunto, NO para que los reproduzcas.
- TU TEXTO EMPIEZA CON LA CALIFICACIÓN GENERAL —una frase— y ACTO SEGUIDO abre
  el primer apartado en la forma que fija el bloque FORMATO de arriba. Nada de
  recuento.
  Puedes referirte a lo que la responsable sostuvo cuando lo estés refutando,
  pero no vuelvas a contar la resolución ni a enumerar los {q}: el lector
  acaba de leerlos dos párrafos más arriba y se encontraría lo mismo por
  tercera vez.
- Y NO ESCRIBAS RÓTULOS. Nada de «Agravios:», «Conceptos de violación:» ni
  «Solución:»: el documento ya los trae de la plantilla y salen duplicados.

EL ORDEN Y LOS GRUPOS — se deciden aquí y se anuncian:
- La síntesis de arriba respetó el orden y el número que propuso quien promueve.
  ES AQUÍ donde se reordena o se juntan varios, y NUNCA en silencio: se dice
  antes de empezar y con fundamento en el ARTÍCULO 76 DE LA LEY DE AMPARO.
- EL CRITERIO PARA JUNTAR ES LA CONSIDERACIÓN QUE SE ATACA Y LA RAZÓN QUE LA
  DECIDE, NO EL TEMA. Dos {q} que hablan de lo mismo pero atacan
  consideraciones distintas, o que caen por razones distintas, no se juntan.
  Tampoco los junta el artículo constitucional que invocan —casi todos repiten
  el 14, el 16 y el 17—.
- EL ORDEN ES EL DE PRELACIÓN LÓGICA: primero lo que decide la procedencia, y
  cada accesorio después de su principal.
  {_orden}
- Si sigues el orden del escrito porque ya es el lógico, dilo y di por qué; si
  lo sigues sin más, no invoques la prelación lógica.
- El anuncio va en una o dos frases, con tus palabras: qué orden sigues y, si
  juntas, cuáles juntas y qué los une. Un anuncio que no dice qué une a los que
  junta no anuncia nada.

CÓMO SE CONSTRUYE CADA APARTADO — por funciones, en este orden, y sólo las que
hagan falta. No son pasos que se repitan en cada {q1}: son lo que un apartado
puede necesitar.

  1. ABRIR. Identifica y califica en la primera o segunda frase, en la forma
     que fija el bloque FORMATO. Si el apartado junta varios {q}, dice qué los
     une: la consideración que atacan y la razón que los decide.

  2. EXPONER LA PREMISA, SÓLO DONDE SE EXPONE POR PRIMERA VEZ: la regla con su
     fuente dentro de la frase, su límite y el apoyo que la sostiene, con su
     punto extraído. Cada premisa se expone UNA vez en todo el estudio. Cuando
     otro apartado depende de la misma regla no la vuelvas a construir:
     recuerda en una frase la proposición concreta que decide ESTE argumento y
     aplícala.

  3. PASAR AL CASO. Una frase con la constancia que activa la regla. Aquí —y
     sólo aquí— se entra en los hechos de este expediente: qué dice la
     constancia, qué exige la regla y por qué encaja o no.

  4. APLICAR. Los argumentos que no traen un dato propio se contestan en un
     párrafo que los nombra a todos. El que trae un dato propio —un hecho, una
     prueba, una cifra, un precepto, un precedente— lleva su párrafo o su
     frase. Sólo la proposición se comparte; el hecho es de cada argumento.

  5. REMITIR CON CONTENIDO. Si un argumento ya quedó contestado en otro
     apartado, se remite nombrando ese apartado, con la proposición que decide
     lo que este argumento tiene de distinto y el puente con él. Una remisión
     que sólo manda a lo ya dicho deja el argumento sin respuesta.

  6. DESARROLLAR LO NUEVO. Si un argumento trae algo que ninguna premisa ya
     expuesta contesta, se dice qué trae de distinto y se construye SÓLO eso.

  7. LA OBJECIÓN, UNA VEZ. La objeción seria —la que te dan abajo como
     material del motor, o la que planteó la contraparte— se contesta una sola
     vez en todo el estudio, en el apartado donde pesa y después de la razón
     decisoria. Ahí es también donde se reconoce lo que el argumento de la
     parte tiene de fuerte: si tiene un punto, se dice, y se explica por qué no
     basta. Caricaturizarlo para tumbarlo produce una respuesta que no
     responde.

  8. RESIDUALES. Una o dos frases cada uno, con su calificación y su razón.

  9. CERRAR EL APARTADO. Con la consecuencia y, si el siguiente depende de
     éste, con el puente. La calificación no se repite: ya se dijo al abrir.

  CÓMO SE DISTINGUE LO QUE SOBRA DE LO QUE NO:
  · REPETICIÓN INNECESARIA: si se quita el pasaje, no se pierde ninguna
    proposición. Se quita.
  · RECAPITULACIÓN ÚTIL: más corta que lo que resume y seguida de una
    inferencia; sólo cabe como recordatorio de una proposición justo antes de
    aplicarla a un dato nuevo.
  · APLICACIÓN DISTINTA DE LA MISMA REGLA: premisa común con otro hecho. Lleva
    su paso de subsunción, no una premisa nueva.
  · DESARROLLO INDISPENSABLE: ninguna proposición ya expuesta contesta lo
    distintivo del argumento. Se desarrolla.

  NO ESCRIBAS QUE LO ESTÁS HACIENDO. Anunciar que el planteamiento se toma «en
  su versión más favorable», o rotular «la mejor objeción a esta conclusión»,
  delata que sigues una instrucción en vez de redactar una sentencia: en un
  engrose eso no se dice, se hace. Lo que alega la parte se atribuye a la
  parte. David tachó las dos a mano.

  CÓMO SE ENCADENAN LOS PÁRRAFOS. No son párrafos puestos uno detrás de otro:
  son un razonamiento que avanza, y cada párrafo lo dice al avanzar. Salvo el
  que abre el apartado, CADA PÁRRAFO ARRANCA RETOMANDO EL ANTERIOR: «Al
  respecto,», «Así,», «En ese sentido,», «Por otro lado,», «Respecto de lo
  anterior,», «En tal sentido,», «Por ende,», «En consecuencia,», «Ahora
  bien,», «No obstante,». Y cuando anuncies que vas a justificar algo,
  ciérralo con dos puntos.

  Un apartado cuyos párrafos empiezan cada uno por su cuenta —«El Juzgado…»,
  «El artículo…», «La recurrente…»— se lee como una lista de afirmaciones
  sueltas aunque el razonamiento sea correcto. David repuso a mano NUEVE de
  estos enlaces en un solo proyecto: ésa es la diferencia entre un borrador y
  un engrose.

  EL LISTÓN, MEDIDO: en el engrose que él corrigió, UNO DE CADA CUATRO
  párrafos abre con conector. En lo que se generó sin esta regla, uno de cada
  nueve. Apunta a uno de cada tres: cuesta cinco palabras por párrafo y es lo
  que hace que el proyecto se lea de corrido en vez de a saltos.

  Y no repitas el mismo enlace dos veces seguidas —dos «En consecuencia,»
  pegados cansan igual que ninguno—. Tienes diez para alternar.

  QUÉ DICE CADA CALIFICACIÓN:
  · SUSTANCIALMENTE FUNDADO: tiene razón en lo esencial de su planteamiento y
  eso basta. Prospera. Medido: aparece en asuntos favorables el 97% de las
  veces, más que el propio «fundado».
  · PARCIALMENTE FUNDADO: tiene razón en una parte de lo que plantea y no en
  otra. Prospera en esa parte, y el proyecto acota cuál. Medido: 141 de sus
  365 apariciones están en asuntos que conceden PARCIALMENTE.
  · FUNDADO PERO INSUFICIENTE: tiene razón Y AUN ASÍ NO ALCANZA, porque
  subsisten otras consideraciones que sostienen el sentido. NO PROSPERA: en el
  acervo aparece en asuntos favorables el 12% de las veces, igual que el
  infundado. Es la calificación honesta cuando el planteamiento acierta y el
  resultado no cambia; usarla en lugar de «infundado» reconoce el acierto sin
  mover el fallo.
  · INATENDIBLE: no puede atenderse por CÓMO o CUÁNDO se plantea —es oscuro, no
  se entiende qué combate, o llega fuera del momento procesal—, no por lo que
  dice. Se distingue del inoperante: el inoperante SE ENTIENDE y no combate la
  razón toral; el inatendible ni siquiera puede examinarse.

  ESENCIALMENTE FUNDADO no es un fundado tibio ni una forma de no mojarse: es
  el planteamiento que combate la razón toral y tiene razón EN LO SUSTANCIAL,
  aunque no en todos sus términos —se equivoca en un dato, en un precepto o en
  el alcance que pide—. Prospera igual que el fundado; lo que cambia es que el
  proyecto ACOTA en qué medida, y esa acotación es la que fija los efectos.
  Medido en este circuito: es el 23% de los agravios de las revisiones que
  revocan. Si el secretario la eligió, respétala y di en qué parte se le da la
  razón y en cuál no. Si es inoperante, la razón TÉCNICA de la inoperancia: que
  no combate la razón toral, que es novedoso, que versa sobre cuestión firme.

- Si lo que se combate es la REDACCIÓN de una parte del acto reclamado,
  TRANSCRÍBELA entre comillas antes de analizarla. UNA VEZ y lo justo.

- NO VIVAS DE LA CITA. Éste es el defecto medido en los engroses de este mismo
  tribunal que sirven de referencia: en uno de ellos, 2,260 de las 3,798
  palabras del estudio —el 59%— son la transcripción literal de una ejecutoria
  de la Suprema Corte, y lo que sigue parafrasea lo mismo; el razonamiento
  propio cabe en seiscientas. En otro, el 48% del considerando es relato de la
  sentencia reclamada, después de haber prometido que era innecesario
  transcribirla. Medido sobre los cinco: el razonamiento propio es el 45%.
  Aquí ha de ser al revés. Del criterio que invoques, trae la REGLA en una o
  dos frases y sigue razonando: el rubro y el registro identifican la tesis; su
  texto íntegro va en la nota al pie, no en el cuerpo.

  Y EL PRECEPTO, IGUAL QUE LA TESIS: NO LO TRANSCRIBAS. Su texto baja solo a
  la nota al pie —de eso se encarga el documento— y en el cuerpo va lo que
  dice, dicho por ti y DENTRO de tu frase:

      SÍ:  «Del artículo 63, fracción IV, de la Ley de Amparo deriva que la
            procedencia del juicio exige la existencia del acto reclamado.»
      SÍ:  «A su vez, el artículo 65 de la Ley de Amparo impone que el
            sobreseimiento por inexistencia se apoye en una conclusión
            objetiva.»
      NO:  «El artículo 63, fracción IV, de la Ley de Amparo. De ese precepto
            deriva que…»   ← el artículo suelto, sin verbo, y la regla en otra
            frase. Así salieron CUATRO párrafos de un mismo proyecto, y David
            los corrigió a mano uno por uno.

  El artículo es el SUJETO o el COMPLEMENTO de tu oración, nunca un rótulo
  aparte. Si al quitarle la transcripción tu frase se queda sin verbo, la frase
  estaba mal construida.

- NO REPITAS EL MISMO PASAJE DOS VECES. Medido en el engrose del ADC 642/2024
  que sirve de referencia: tres párrafos copiados palabra por palabra dentro
  del mismo considerando, ochenta líneas después. Si el artículo 79 de la Ley
  de Amparo ya se enunció al fijar la premisa, más adelante se le NOMBRA
  —«el precepto citado», «la regla ya enunciada»— y no se vuelve a explicar.
  NUNCA «el precepto transcrito»: en el cuerpo no hay transcripción, está en la
  nota, y esa palabra manda al lector a buscar algo que no existe.
  Un pasaje repetido no refuerza: delata que el estudio se escribió por trozos.

- NO REMITAS POR ORDINAL A OTRO CONSIDERANDO. «En términos del considerando
  séptimo» obliga a que exista un séptimo, y tú no sabes cuántos tendrá el
  documento —los ordinales los calcula el compositor al final, y en una queja
  hay tres apartados donde en un amparo directo hay seis—. Medido en el
  engrose del ARA 17/2025 que sirve de referencia: su resolutivo remite al
  «considerando séptimo» y ese engrose no tiene séptimo considerando. Remite
  por su NOMBRE —«en el apartado de antecedentes», «al resolver el primer
  agravio»— que no se descoloca.

- SI JUNTAS PLANTEAMIENTOS, DI CUÁLES Y QUÉ LOS UNE. Estudiar varios juntos es
  correcto cuando atacan la misma consideración y caen por la misma razón —el
  reencuadre que los hace caer todos a la vez ahorra treinta páginas—, pero
  hay que nombrarlos uno por uno por su ordinal y decir cuál es esa
  consideración y cuál esa razón. Juntar sin decir cuáles es indistinguible de
  olvidarse de uno.

- Y NO PROMETAS LO QUE NO VAS A CUMPLIR. Si el documento dijo que era
  innecesario transcribir la sentencia recurrida, no la parafrasees después
  entera: eso pasa en los CINCO engroses de referencia y es lo primero que se
  nota al leerlos seguidos.

- LA SUPLENCIA DE LA QUEJA, EN UN SOLO SITIO. Donde la ley la manda —la persona
  menor de edad o incapaz (artículo 79, fracción II, de la Ley de Amparo), la
  materia penal en favor de la persona inculpada o sentenciada (fracción III),
  la persona trabajadora (fracción V), entre otras—, un argumento mal expuesto
  se SUPLE y se estudia: no se declara inoperante por deficiencia en su
  impugnación —porque «no combatió la razón toral» o «no precisó qué prueba se
  omitió»—, que es aplicarle la técnica de estricto derecho que la ley le
  releva. Y la suplencia SÓLO SE EXPRESA en la sentencia cuando de ella deriva
  un beneficio (artículo 79, penúltimo párrafo): si al suplir nada cambia, no se
  menciona.

FUNDAMENTO — hay que fundar, y hay que fundar bien:
- {_regla_funda} Una premisa que decide sin
  apoyo no es un engrose: es una opinión con formato de sentencia.

  LA MEDIDA: cada premisa que decide, con su apoyo —el que de verdad la
  sostiene, y como máximo dos—. Ninguna tesis se cita dos veces en el estudio:
  si la misma vuelve a servir, se la nombra como ya citada y se aplica. La que
  sólo refuerza lo ya fundado NO se cita. Si de veras ninguna de las que tienes
  sostiene una premisa, razónala sin ella y sigue: pero que eso sea la
  excepción, no la norma.
- Sólo se cita lo que está en el MATERIAL. NUNCA inventes un registro digital
  ni un número de tesis: tus datos de entrenamiento son viejos y falsos.
- LEE EL TEXTO DE LA TESIS ANTES DE INVOCARLA, no sólo su rubro. El rubro es
  un título y a menudo dice menos —o algo distinto— de lo que la tesis resuelve.
  Si una tesis concreta no sostiene lo que quieres afirmar, usa OTRA de las que
  tienes; dejar sin apoyo la premisa que decide no es la salida.
- ASÍ SE CITA, Y NO DE OTRA FORMA. La cita ocupa su propio final de párrafo y
  el rubro NO se embebe en mitad de una frase que sigue después:

      Sirve de apoyo el criterio de registro 2022074:

  Y ahí se detiene el párrafo. NO ESCRIBAS TÚ NI EL TIPO NI EL ÓRGANO: no digas
  «la jurisprudencia», no digas «tesis aislada», no digas «de la Primera Sala».
  El documento los pone solo, tomados del acervo, junto con el rubro y el texto
  íntegro.
- NO CITES UN CRITERIO PARA DECIR DESPUÉS QUE NO APLICA. Salió esto en un
  proyecto: «Sirve de apoyo la jurisprudencia … INCONFORMIDAD. LA SUPREMA
  CORTE DEBE SUPLIR LA QUEJA DEFICIENTE…» y, tres renglones más abajo, «el
  criterio citado NO SE APLICA DIRECTAMENTE, porque se refiere al cumplimiento
  de una sentencia de amparo». David lo tachó entero, y con razón: «Sirve de
  apoyo» es una afirmación, y lo que no sirve de apoyo no se invoca.
  Si el criterio aplica, se cita y se dice qué regla aporta. Si no aplica, NO
  SE CITA: se borra y se sigue razonando. Un acervo que no trae la tesis del
  punto no se disimula rellenando con la más parecida; se dice que falta, que
  es información útil, y la afirmación se sostiene con lo que sí haya.
  ÚNICA EXCEPCIÓN: citarlo para DISTINGUIRLO cuando la contraparte lo invocó o
  cuando el asunto se parece y hay que explicar por qué no gobierna. Entonces
  no se abre con «sirve de apoyo» sino con «no resulta aplicable el criterio…,
  porque…», que es lo contrario y se lee como lo que es.Antes este ejemplo nombraba una Sala concreta y el modelo lo copiaba
  cambiando sólo el número: así una tesis aislada del Pleno salió publicada como
  «jurisprudencia de la Primera Sala», y la nota al pie de la misma página —que
  sí sale del acervo— la desmentía. Tú escribes el verbo que ata la cita a tu
  razonamiento; de identificarla se encarga el documento. Escribir «la jurisprudencia de registro X, de rubro
  «Y», establece que…» deja la cita partida por la mitad y sin transcripción.
- LA INSTANCIA VA SIEMPRE: «de la Primera Sala de la Suprema Corte de Justicia
  de la Nación», «de la Segunda Sala», «del Pleno», «de un Tribunal Colegiado de
  Circuito». Sin ella no se sabe qué peso tiene el criterio.
- Y NOMBRA LA LEY EN LA MISMA FRASE, SIEMPRE. «El artículo 4º» a secas no
  identifica nada: el 4º existe en la Constitución, en el Código Civil, en el
  Procesal y en veinte leyes más. Escribe «el artículo 4º de la Constitución
  Política de los Estados Unidos Mexicanos», «el artículo 296 del Código Civil
  del Estado de Querétaro». No es pedantería: el documento baja al pie el TEXTO
  ÍNTEGRO de cada precepto que puede identificar, y esa nota es lo que permite
  a quien firma comprobar de un vistazo si el artículo dice lo que le atribuyes.
  Un artículo sin su ley se queda sin nota, y la afirmación sin respaldo.
- CITA LOS ARTÍCULOS QUE TIENES, NO LOS QUE RECUERDAS. En el bloque de NORMAS
  van los preceptos que el acervo encontró para este asunto, con su texto
  íntegro. Ésos son los que se citan, por su número y su cuerpo legal exacto.
  Si citas un artículo que no está ahí —«el 242 del Código Civil Federal»— pasan
  dos cosas malas a la vez: nadie puede comprobar que diga lo que le atribuyes,
  y el documento no puede llevar su texto al pie, que es lo que permite a quien
  firma verificarlo de un vistazo. Cuando de verdad necesites uno que no tengas,
  dilo con esas palabras en vez de citarlo de memoria.
- EL CÓDIGO QUE RIGE ES EL DE LA ENTIDAD, Y SÓLO EL QUE ESTÁ EN EL MATERIAL.
  El CÓDIGO NACIONAL DE PROCEDIMIENTOS CIVILES Y FAMILIARES entró en vigor de
  forma ESCALONADA y en muchas entidades —Querétaro entre ellas— TODAVÍA NO
  RIGE: ahí siguen aplicándose el Código Civil y el Código de Procedimientos
  Civiles del Estado. Aplicar un código que aún no ha entrado en vigor invalida
  la sentencia entera, y es un error que no perdona nadie.
  LA REGLA MECÁNICA: no cites ningún código que no aparezca en las NORMAS del
  material. El acervo trae la legislación vigente de la entidad del asunto; si
  el Código Nacional no está ahí, es porque en esa entidad no rige.
- LA LEY AJENA NO ENTRA; EL CRITERIO AJENO SÍ. Es la distinción que más veces
  se ha roto y está medida sobre 139 documentos de este tribunal: NO HAY UNA
  SOLA aplicación de ley de otra entidad, y hay decenas de criterios que
  interpretan la de otra entidad, invocados con toda naturalidad.
  · PROHIBIDO razonar con el Código Civil o de Procedimientos de Jalisco, de la
    Ciudad de México o de cualquier otra entidad. El juicio de origen se rige
    por la legislación del Estado de la entidad del asunto y ESA es la que se aplica. La
    analogía ENTRE CÓDIGOS DE ENTIDADES DISTINTAS no existe aquí: cuando la
    parte la propone, este Tribunal la rechaza —«la analogía es improcedente»—.
    La única analogía de ley admisible es dentro del propio código queretano.
  · PERMITIDO invocar jurisprudencia que interprete legislación de otra
    entidad, por una de estas tres razones y sólo por ellas:
      – porque de ella deriva un MANDATO INTERPRETATIVO DE FUENTE
        CONSTITUCIONAL: la Corte fija cómo debe entenderse la figura jurídica;
      – porque la legislación interpretada ES LA DE QUERÉTARO;
      – porque la de otro estado es DE CONTENIDO SIMILAR a la queretana.
  · Y SE CITA SIN EXCUSARSE, anclando al PRINCIPIO y no a la norma ajena. Están
    PROHIBIDAS las fórmulas «por tratarse de legislación diversa a la aplicable
    al caso», «aunque referido a la legislación del Estado de X» y cualquier
    otra que ponga la entidad ajena como razón: no aparecen ni una vez en el
    corpus. Se escribe así:
{__import__('vicio_inoperancia').ejemplo_analogia()}
        «De acuerdo con el principio rector que informa la tesis precitada, es
         factible considerar que…»
        «resulta aplicable, por identidad de razón, … pues si bien en aquel
         precedente el análisis se centró en X, el principio rector es el mismo»
    La cláusula concesiva —«si bien…», «aun cuando…»— salva una distancia DE
    TEMA O DE SUPUESTO, NUNCA de entidad federativa.
  · SI EL CRITERIO ES DE LA SUPREMA CORTE NO HAY PUENTE QUE TENDER: es
    obligatorio conforme al artículo 217 de la Ley de Amparo y la legislación
    que interpretó resulta irrelevante. Se aplica en seco, sin «por analogía».
  · SI ES DE UN COLEGIADO DE OTRO CIRCUITO el verbo es COMPARTIR, no obedecer:
    se dice que este tribunal comparte ese criterio, no que lo acata.
- EL REGISTRO DIGITAL VA SIEMPRE, sin excepción, en la misma frase que el rubro.
  {__import__('vicio_inoperancia').clave_no_sustituye()}: sin el registro nadie comprueba
  la cita en el Semanario, que es para lo que sirve citarla.
- Al citar una tesis: en el CUERPO van sólo el rubro entre comillas y el
  registro. NADA MÁS. La localización —«[J]; 11a. Época; 1a. Sala; Gaceta
  S.J.F.; Libro 52…»— NO se escribe en el cuerpo: el documento la coloca sola
  al pie, que es donde va en una sentencia, y escribirla dos veces obliga a
  borrarla a mano. El texto de la tesis tampoco lo transcribas: se transcribe
  solo, desde el acervo, palabra por palabra.
- DESPUÉS DE LA CITA, NO LA REPITAS: ÚSALA. Es lo que más se rompe. El
  documento transcribe el texto íntegro de la tesis debajo de la cita, palabra
  por palabra; si después vuelves a contar lo que dice, el lector se encuentra
  lo mismo dos veces y la sentencia engorda sin decir nada nuevo. Si la tesis
  es la PREMISA de tu razonamiento, lo que sigue a la cita es UNA frase que
  extrae su punto con palabras tuyas —más abstracta que el texto transcrito— y
  lo gira de inmediato a este asunto: por qué eso decide ESTE caso. Si lo que
  escribes tras la cita se entiende sin conocer el expediente, es un resumen de
  la tesis: bórralo.
- La INOPERANCIA se razona: hay que decir POR QUÉ el planteamiento no combate
  la razón toral, no basta con declararla.
- SI CITAS UN CRITERIO, RESUELVE CONFORME A ÉL. Invocar una jurisprudencia que
  dice que el reconocimiento de un hecho no releva al patrón de probar los
  elementos de la causal, y acto seguido tener por probada la causal porque el
  trabajador reconoció el hecho, es contradecirse dentro del mismo párrafo. Si
  el criterio no lleva a donde quieres ir, NO lo cites: busca otro o razona sin
  él. Una cita que el propio fallo desmiente es peor que ninguna cita.
- NUNCA SUPONGAS LO QUE CONSTA. Un tribunal tiene los autos delante: o el hecho
  consta y se AFIRMA, o no consta y se dice que no obra. Están PROHIBIDAS las
  fórmulas «si … fue efectivamente», «se afirma que», «según lo planteado», «de
  ser cierto», «en el supuesto de que». Si el material no te permite afirmar,
  escribe que el punto no está acreditado y sigue.
{_bloque_ley_de_la_via(material)}
{_bloque_aportado(contexto)}
{_bloque_constancias(propuesta_global, contexto, criterios)}
{partes.bloque() if partes is not None else ""}{_bloque_ficha(material)}
{marco if isinstance(marco, str) else ""}
{_bloque_arquitectura(_materia_v, "v2")}
{_bloque_tecnica(_tipo_tec, rama, violacion_procesal, material)}
{_bloque_circuito(_tipo_tec, criterios)}
{_bloque_conceptos(rama, conceptos_violacion, "v2", reasuncion=getattr(material, "reasuncion", None))}
{_bloque_criterio(criterios, _materia_v, _texto_de(material), getattr(material, "tipo_asunto", ""), _formato, getattr(material, "problemas", None) or [], variante="v2", decisiva=getattr(material, "decisiva", None))}{_bloque_sin_calificar(criterios)}
{_bloque_suplencia(material)}
{_bloque_global(propuesta_global, criterios, registros_de_otros(resumen_acto, str(getattr(material, 'tipo_asunto', '') or '').strip().lower() == 'amparo_revision'))}
{_bloque_precedente(material, criterios)}
{_bloque_material(_mat_vista)}

═══════════════════════════════════════════════════════════════════════
LO QUE RESOLVIÓ {_org_rotulo}
═══════════════════════════════════════════════════════════════════════
{resumen_acto}

═══════════════════════════════════════════════════════════════════════
LO QUE SE COMBATE
═══════════════════════════════════════════════════════════════════════
{resumen_conceptos}
{_bloque_escrito_literal(escrito_literal, resumen_conceptos)}{bloque_inventario}

Escribe el estudio de fondo.
{cierre_marco}
{_dc_cierre}
{_recuerda_forma}{recordatorio_marcas}
NO ESCRIBAS LA FÓRMULA FINAL. El documento añade solo, debajo de tu texto, la
frase de cierre que corresponde al tipo de asunto. Si tú escribes otra igual,
el proyecto acaba con dos cierres seguidos diciendo lo mismo, que es lo que
pasó en la revisión 410/2026.
{_cierre}

Si hay obstáculos al sentido fijado, añade al final —DESPUÉS de los EFECTOS DE
LA CONCESIÓN, si los hay: los efectos son sentencia y las advertencias no— un
apartado «ADVERTENCIAS» —fuera del cuerpo de la sentencia— con lo que el
secretario debe valorar.
Nada más."""


# ═══ LA v4: LA v3 CON EL GUION DEL PLAN (Paso 2, 26-sep-2026) ════════════════
# La v3 —v2 + inventario de argumentos + marcas— la construye otra pieza. Aquí
# no se copia ni se reescribe: se pide el prompt de la v3 con la MISMA llamada
# (el material con la variante «v3») y se le inserta el guion. Así la v4 no
# puede divergir de la v3 más que en el guion, que es lo que se mide (C frente
# a C-lite dice si el valor viene del planificador o de las marcas, §6.6).
#
# SIN GUION —el plan falló dos veces, venció o no hay inventario— la v4 ES la
# v3, sin un carácter de diferencia: es la salida prevista y el resolver la
# anota como tal (main.py `_taller_plan_para`).
#
# DÓNDE VA: justo antes de «Escribe el estudio de fondo.», después de todo el
# material y de los resúmenes, porque lo último es lo que más se obedece; y una
# línea al final que lo recuerda. El bloque sólo trae descripciones de sus
# rótulos: ni una frase para copiar (lección del ejemplo que se firma literal).
_MARCA_ESCRIBE = "\nEscribe el estudio de fondo."
# (Hasta el 28-sep-2026 terminaba «…y lo PENDIENTE DE RAZÓN primero en
# ADVERTENCIAS»: la Decisión 6 se retiró —plan-6— y nada va ahí por eso.)
_RECUERDA_GUION = (
    "EL GUION DE ARRIBA MANDA LA ORGANIZACIÓN: un apartado por cada APARTADO, "
    "cada premisa sólo donde dice EXPONE y cada argumento con la respuesta que "
    "le asigna.\n")

# LA JERARQUÍA DEL GUION (plan-6, 28-sep-2026, AR 631/2025). El texto que la
# v3 hereda de la v2 sólo deja declarar algo innecesario, o contestarlo por
# consecuencia, si lo dice el CRITERIO del secretario (0ad0379: el «sin
# materia» es del criterio, no del estudio). La jerarquía que calcula el plan
# deriva del sentido que él fijó para el principal, así que vale como su
# criterio; pero con ese texto intacto el modelo contestaba igual cada
# argumento, o lo ponía en DESVIACIONES DEL GUION. SÓLO cuando el guion trae
# JERARQUÍA se ajusta, por reemplazo sobre la v3 ya armada: el texto de
# `_prompt_estudio_v2` —y sus instantáneas— no se toca, ni la v1 (congelada).
_REEMPLAZOS_JERARQUIA = (
    ("  que su criterio —su calificación o su razón— lo decida así; y entonces, si",
     "  que su criterio —su calificación o su razón— lo decida así, o que la\n"
     "  JERARQUÍA del guion lo resuelva por consecuencia de su principal; y entonces, si"),
    ("  calificación o en su razón—: esa decisión es suya, no del estudio.",
     "  calificación o en su razón—, o para lo que la JERARQUÍA del guion resuelve\n"
     "  por consecuencia del principal: esa decisión es suya, no del estudio."),
    ("  adicional ninguno de sus argumentos, salvo que la razón del secretario lo\n"
     "  diga de ese argumento.",
     "  adicional ninguno de sus argumentos, salvo que la razón del secretario lo\n"
     "  diga de ese argumento o que la JERARQUÍA del guion lo resuelva por\n"
     "  consecuencia de su principal."),
    ("     que sólo manda a lo ya dicho deja el argumento sin respuesta.",
     "     que sólo manda a lo ya dicho deja el argumento sin respuesta. Lo que la\n"
     "     JERARQUÍA del guion resuelve con su principal no necesita otra\n"
     "     proposición: el puente es la dependencia."),
    ("  revocan. Si el secretario la eligió, respétala y di en qué parte se le da la\n"
     "  razón y en cuál no.",
     "  revocan. Si el secretario la eligió, respétala y di en qué parte se le da la\n"
     "  razón y en cuál no; dentro de un grupo que la JERARQUÍA del guion resuelve\n"
     "  con su principal, se acota una vez para el grupo, no por argumento."),
    # LAS CUATRO QUE SEGUÍAN PIDIENDO UNA RESPUESTA POR ARGUMENTO (revisión
    # adversarial, 28-sep-2026): con ellas intactas el modelo tenía cómo
    # justificar un párrafo y un «Es fundado…» para cada argumento que la
    # jerarquía manda contestar con su principal, y volvía a mezclar. La
    # cuarta, la EXTENSIÓN del bloque del guion, se ajusta en `plan_estudio`.
    ("demostrarla, y no cambia la del problema.",
     "demostrarla, y no cambia la del problema. Salvo lo que la JERARQUÍA del guion\n"
     "resuelve por consecuencia de su principal: eso lleva la calificación y la\n"
     "respuesta de su grupo, no una propia."),
    ("     frase. Sólo la proposición se comparte; el hecho es de cada argumento.",
     "     frase. Sólo la proposición se comparte; el hecho es de cada argumento. Lo\n"
     "     que la JERARQUÍA del guion resuelve por consecuencia de su principal se\n"
     "     contesta dentro de la respuesta de su grupo, con su dato y su marca."),
    ("  cifra, un precepto, un precedente— reciba su respuesta, aunque eso alargue",
     "  cifra, un precepto, un precedente— reciba su respuesta —la de su grupo si la\n"
     "  JERARQUÍA del guion lo resuelve por consecuencia de su principal—, aunque eso alargue"),
)


def _con_jerarquia(base: str) -> str:
    """La v3 con los ajustes de la JERARQUÍA. Cada uno, una vez; el que no
    encuentra su texto no hace nada (la v3 sigue valiendo tal cual)."""
    for viejo, nuevo in _REEMPLAZOS_JERARQUIA:
        base = base.replace(viejo, nuevo, 1)
    return base


# ═══ LA CITA, SU REGLA Y SU APLICACIÓN (AR 631/2025, 28-sep-2026) ═══════════
# David: «un diálogo jurídico que revele una argumentación de alta calidad.
# Después de citar tesis hay que hacerlas hablar. Y después aplicarlas al caso
# concreto. Es lo que se estila.» La técnica se midió a mano en diez
# sentencias de la Corte y cinco engroses del colegiado (82 + 19 usos de
# criterio: ninguno sin su regla dicha ni su aplicación; de catorce criterios
# invocados por otro, ninguno reciclado como apoyo) y está en
# redactor-sentencias/scjn_estilo/estilo_scjn_tesis.md. En la Solución del 631
# quedaban seis de nueve sin regla ni aplicación, y la causa estaba en el
# propio texto que la v4 hereda de la v2:
#   · «DESPUÉS DE LA CITA, NO LA REPITAS» decía que el documento transcribe la
#     tesis debajo de la cita —la baja a la nota— y mandaba borrar lo que se
#     escribiera tras ella «si se entiende sin conocer el expediente», que es
#     justo extraer la regla en abstracto: prohibía hacerla hablar;
#   · «la que sólo refuerza lo ya fundado NO se cita» chocaba con la regla 4 de
#     la arquitectura y con la Corte, donde el cierre es el 28 % de sus usos;
#   · la tesis que invoca la parte sólo tenía una fórmula de apertura, y un
#     comentario de desarrollo se había colado dentro del prompt.
# Se sustituye aquí, sobre la v3 ya armada y SÓLO con guion, como la
# JERARQUÍA: el texto de `_prompt_estudio_v2` —y sus instantáneas— no se toca,
# ni la v1, congelada (su copia del comentario filtrado, en la «ÚNICA
# EXCEPCIÓN» de la v1, se queda: quitarla movería su instantánea).
# DESCRIPCIONES, NUNCA FRASES PARA COPIAR (lección medida tres veces en este
# proyecto): ninguna frase nueva entre comillas.
_TECNICA_MEDIDA_VIEJA = (
    "  LA MEDIDA: cada premisa que decide, con su apoyo —el que de verdad la\n"
    "  sostiene, y como máximo dos—. Ninguna tesis se cita dos veces en el estudio:\n"
    "  si la misma vuelve a servir, se la nombra como ya citada y se aplica. La que\n"
    "  sólo refuerza lo ya fundado NO se cita. Si de veras ninguna de las que tienes\n"
    "  sostiene una premisa, razónala sin ella y sigue: pero que eso sea la\n"
    "  excepción, no la norma.")
_TECNICA_MEDIDA_NUEVA = (
    "  LA MEDIDA: cada premisa que decide, con su apoyo —el que de verdad la\n"
    "  sostiene, y como máximo dos—. Ninguna tesis se cita dos veces en el estudio:\n"
    "  si la misma vuelve a servir, se la nombra como ya citada y se aplica. Una\n"
    "  tesis que cierra un razonamiento tuyo es legítima —así usa la Corte más de\n"
    "  la cuarta parte de sus citas— sólo si el párrafo anterior ya dijo, con tus\n"
    "  palabras, la misma proposición que ella sostiene y ya la aplicó a un dato\n"
    "  del caso; si no, no cierra nada: o la haces hablar y la aplicas, o no la\n"
    "  citas. La Corte y los engroses del colegiado que sirven de referencia\n"
    "  citan poco: un criterio por cada mil seiscientas palabras de estudio, de\n"
    "  media. En uno de dos mil a dos mil quinientas palabras lo usual es de tres\n"
    "  a cinco; más suele ser una fila de rubros sin regla. No es una cuota que\n"
    "  alcanzar. Si de veras ninguna de las que tienes sostiene una premisa,\n"
    "  razónala sin ella y sigue: pero que eso sea la excepción, no la norma.")
_TECNICA_PARTE_VIEJA = (
    "  ÚNICA EXCEPCIÓN: citarlo para DISTINGUIRLO cuando la contraparte lo invocó o\n"
    "  cuando el asunto se parece y hay que explicar por qué no gobierna. Entonces\n"
    "  no se abre con «sirve de apoyo» sino con «no resulta aplicable el criterio…,\n"
    "  porque…», que es lo contrario y se lee como lo que es.Antes este ejemplo nombraba una Sala concreta y el modelo lo copiaba\n"
    "  cambiando sólo el número: así una tesis aislada del Pleno salió publicada como\n"
    "  «jurisprudencia de la Primera Sala», y la nota al pie de la misma página —que\n"
    "  sí sale del acervo— la desmentía. Tú escribes el verbo que ata la cita a tu\n"
    "  razonamiento; de identificarla se encarga el documento. Escribir «la jurisprudencia de registro X, de rubro\n"
    "  «Y», establece que…» deja la cita partida por la mitad y sin transcripción.")
_TECNICA_PARTE_NUEVA = (
    "  SALVO EL CRITERIO QUE INVOCÓ OTRO —la parte, la responsable o el órgano\n"
    "  recurrido—, o el que sin que nadie lo invocara parece gobernar el asunto:\n"
    "  ése se contesta, y nunca se recicla como apoyo del proyecto. En la\n"
    "  Corte, de catorce criterios invocados por otro, ninguno volvió como apoyo:\n"
    "  once se distinguieron, dos se confirmaron con su regla y uno se acotó. Se\n"
    "  identifica, se dice en una frase para qué lo invocó y se elige UNA de tres\n"
    "  respuestas:\n"
    "  · APLICA Y LE DA LA RAZÓN: pasa a ser premisa, con su regla dicha con tus\n"
    "    palabras y aplicada a la constancia como cualquier otra, y se dice\n"
    "    expresamente que en ese punto asiste razón a quien lo invocó.\n"
    "  · APLICA, PERO NO ALCANZA LO QUE PRETENDE: se concede su regla y se traza\n"
    "    su frontera con el dato del caso que queda fuera de ella.\n"
    "  · NO APLICA Y SE DISTINGUE: primero el supuesto de hecho o normativo que\n"
    "    resolvía —qué tuvo enfrente quien lo emitió y cuál es su punto\n"
    "    decisivo—; después su razón, si la diferencia está ahí; luego el dato\n"
    "    concreto del expediente que saca a este asunto de ese supuesto; y la\n"
    "    conclusión, sin descalificar el criterio en sí. Al menos dos oraciones\n"
    "    con contenido: decir que se refiere a otra cosa sin decir a cuál no\n"
    "    distingue nada.\n"
    "  Cuando se distingue o se acota, el arranque del párrafo lo dice —se lo\n"
    "  atribuye a quien lo invocó, o niega que sea aplicable—, nunca un verbo de\n"
    "  apoyo. Varios criterios de la parte con una misma premisa se contestan\n"
    "  juntos: la premisa común se enuncia una vez y se prueba contra los hechos.\n"
    "  Si la parte invoca criterios genéricos —exhaustividad, fundamentación,\n"
    "  igualdad—, se formula la regla que su argumento necesitaría y se dice que\n"
    "  ninguno la sostiene y qué supuestos sí cubren.\n"
    "  Tú escribes el arranque que ata la cita a tu razonamiento, y en la misma\n"
    "  oración su registro digital y su rubro entre comillas, copiados del\n"
    "  material: con esas dos llaves el documento la reconoce, pone el tipo y el\n"
    "  órgano y baja el texto. La regla que atribuyes a un criterio nunca va dentro de la\n"
    "  oración que anuncia su cita: esa oración termina en el rubro, ahí se\n"
    "  corta el párrafo y el texto baja a la nota; la regla va antes del anuncio\n"
    "  o en el párrafo que sigue.")
# Sólo con JERARQUÍA en el guion (encaje con el plan-6): dentro de un grupo que
# se resuelve por consecuencia, la técnica completa no; en el principal, sí.
_TECNICA_PARTE_GRUPO = (
    "\n  Dentro de un grupo que la JERARQUÍA del guion resuelve por consecuencia de\n"
    "  su principal, manda lo que ella describe para las tesis de la parte; la\n"
    "  respuesta completa, con su técnica, es la del principal.")
_TECNICA_DESPUES_VIEJA = (
    "- DESPUÉS DE LA CITA, NO LA REPITAS: ÚSALA. Es lo que más se rompe. El\n"
    "  documento transcribe el texto íntegro de la tesis debajo de la cita, palabra\n"
    "  por palabra; si después vuelves a contar lo que dice, el lector se encuentra\n"
    "  lo mismo dos veces y la sentencia engorda sin decir nada nuevo. Si la tesis\n"
    "  es la PREMISA de tu razonamiento, lo que sigue a la cita es UNA frase que\n"
    "  extrae su punto con palabras tuyas —más abstracta que el texto transcrito— y\n"
    "  lo gira de inmediato a este asunto: por qué eso decide ESTE caso. Si lo que\n"
    "  escribes tras la cita se entiende sin conocer el expediente, es un resumen de\n"
    "  la tesis: bórralo.")
_TECNICA_DESPUES_NUEVA = (
    "- LA CITA, SU REGLA Y SU APLICACIÓN. Es lo que más se rompe, y lo que separa\n"
    "  un engrose de una lista de rubros. Medido en diez sentencias de la Corte y\n"
    "  cinco engroses del colegiado: de ciento una citas, ninguna quedó sin su\n"
    "  regla dicha ni su aplicación; en un proyecto generado aquí quedaron seis\n"
    "  de nueve. El texto de una tesis larga NO queda debajo de la cita: baja a\n"
    "  la nota al pie, y en el cuerpo quedan el anuncio con su registro y el\n"
    "  rubro. Quien lee el proyecto no sabe qué sostiene la tesis si tú no lo\n"
    "  dices. Antes de invocar un criterio decide para qué lo traes, porque eso\n"
    "  fija lo que va alrededor:\n"
    "  · PREMISA —su regla decide el punto—. En el mismo tramo, antes del anuncio\n"
    "    o en el párrafo que sigue al rubro, dices con tus palabras la regla que\n"
    "    sostiene: más general y más corta que su texto, sin nombrar todavía a\n"
    "    las partes, en una a tres oraciones (entre cuarenta y ciento veinte\n"
    "    palabras, más o menos). Si contiene requisitos o supuestos, cuáles son y\n"
    "    cuál pone en juego este asunto; su razón, cuando la aplicación depende\n"
    "    de ella; su alcance o su límite; y, si interpreta otra ley u otra época,\n"
    "    por qué vale aquí. Luego pasas al expediente: cada elemento de la regla\n"
    "    contra la constancia que lo activa —fecha, foja, documento, cifra, quién\n"
    "    hizo qué—, con los términos de la regla y no los del texto de la tesis,\n"
    "    en una extensión igual o mayor que la de la regla, hasta la consecuencia\n"
    "    para el argumento. Si la tesis es corta, el documento deja su texto\n"
    "    debajo del rubro; aun entonces lo que sigue es su regla dicha por ti y\n"
    "    su aplicación, no otra tesis.\n"
    "  · CIERRE de un razonamiento ya hecho: sólo con la condición de LA MEDIDA\n"
    "    —la misma proposición, dicha y aplicada en el párrafo anterior—. Detrás\n"
    "    no va nada.\n"
    "  · FÓRMULA PROCESAL —la adhesión sin materia, la inoperancia por novedad,\n"
    "    el mayor beneficio—: la aplicación va en el mismo párrafo y la cita lo\n"
    "    cierra.\n"
    "  · CRITERIO QUE INVOCÓ OTRO: una de las tres respuestas descritas arriba.\n"
    "  Si un criterio no cabe en ninguna de las cuatro, no se invoca. Hacerla\n"
    "  hablar no es repetirla: una paráfrasis pegada a su texto es eco, y la\n"
    "  prueba es que tu regla sea más abstracta y más corta que la tesis. Y nunca\n"
    "  pongas dos anuncios seguidos sin prosa tuya entre ellos, salvo que los dos\n"
    "  sostengan la misma proposición y la digas una vez; tres o más seguidos son\n"
    "  una fila de rubros sin regla.")
# EL EJEMPLO SIN RUBRO (AR 631/2025, 28-sep-2026). La v2 enseña el anuncio con
# el registro solo y, veinte renglones abajo, exige el registro «en la misma
# frase que el rubro» y el rubro entre comillas en el cuerpo. Mientras «LA
# INSTANCIA VA SIEMPRE» empujaba a escribir el anuncio entero, el modelo ponía
# el rubro; al retirarla, copió el ejemplo y el compositor no reconoció ni una
# cita. Se dice lo que va detrás del registro, sin frase entre comillas.
_TECNICA_EJEMPLO_VIEJO = (
    "  Y ahí se detiene el párrafo. NO ESCRIBAS TÚ NI EL TIPO NI EL ÓRGANO: no digas\n")
_TECNICA_EJEMPLO_NUEVO = (
    "  Detrás del registro, en la misma oración, el rubro entre comillas tal como\n"
    "  viene en el material; y ahí se detiene el párrafo.\n"
    "  NO ESCRIBAS TÚ NI EL TIPO NI EL ÓRGANO: no digas\n")
_REEMPLAZOS_TECNICA = (
    (_TECNICA_MEDIDA_VIEJA, _TECNICA_MEDIDA_NUEVA),
    (_TECNICA_EJEMPLO_VIEJO, _TECNICA_EJEMPLO_NUEVO),
    (_TECNICA_PARTE_VIEJA, _TECNICA_PARTE_NUEVA),
    (_TECNICA_DESPUES_VIEJA, _TECNICA_DESPUES_NUEVA),
    # «2. EXPONER LA PREMISA»: si la premisa sale de un criterio, su regla,
    # sus elementos, su razón y su alcance.
    ("     punto extraído. Cada premisa se expone UNA vez en todo el estudio.",
     "     punto extraído —si sale de un criterio, su regla dicha con tus palabras,\n"
     "     los elementos que este asunto pone en juego, su razón y su alcance: lo\n"
     "     describe LA CITA, SU REGLA Y SU APLICACIÓN—. Cada premisa se expone UNA\n"
     "     vez en todo el estudio."),
    # «4. APLICAR»: el dato propio que es un precedente de la parte.
    # (Integración del 28-sep-2026: justo encima, el ajuste de la JERARQUÍA
    # dice que lo del grupo de consecuencia se contesta dentro de la respuesta
    # del grupo; sin la última cláusula, esta frase pedía además la respuesta
    # completa de la técnica para el mismo criterio. Sin JERARQUÍA en el guion
    # no hay grupos y la cláusula no aplica.)
    ("\n\n  5. REMITIR CON CONTENIDO.",
     "\n     Si ese dato propio es un criterio que la parte invocó, su respuesta es\n"
     "     una de las tres que se describen para el criterio que invocó otro, no\n"
     "     volver a citarlo como apoyo; dentro de un grupo que el guion resuelve\n"
     "     por consecuencia de su principal, la que el guion describe.\n\n"
     "  5. REMITIR CON CONTENIDO."),
    # La regla 4 de la arquitectura: las dos posiciones de la Corte.
    ("   Nunca abras un tramo con la cita: ése es el patrón de las medias, donde la\n"
     "   tesis sustituye al razonamiento en vez de apoyarlo.",
     "   Nunca abras un tramo con la cita: ése es el patrón de las medias, donde la\n"
     "   tesis sustituye al razonamiento en vez de apoyarlo. Si el anuncio va\n"
     "   primero, el párrafo que sigue al rubro retoma el criterio y dice su regla\n"
     "   antes de aplicarla: las dos posiciones las usa la Corte, y ninguna admite\n"
     "   la cita sin su regla dicha."),
    # «NO VIVAS DE LA CITA» (integración del 28-sep-2026) pedía la regla del
    # criterio «en una o dos frases»; la técnica pide de una a tres oraciones
    # con sus elementos, su razón y su alcance cuando la aplicación depende de
    # ellos. Una sola medida: la de la técnica.
    ("  Aquí ha de ser al revés. Del criterio que invoques, trae la REGLA en una o\n"
     "  dos frases y sigue razonando:",
     "  Aquí ha de ser al revés. Del criterio que invoques, trae la REGLA dicha\n"
     "  por ti, con la medida de LA CITA, SU REGLA Y SU APLICACIÓN, y sigue\n"
     "  razonando:"),
    # «LA INSTANCIA VA SIEMPRE» (integración del 28-sep-2026) chocaba con «NO
    # ESCRIBAS TÚ NI EL TIPO NI EL ÓRGANO» del anuncio y con la técnica, que
    # dice que de identificar la cita se encarga el documento. Las dos valen,
    # cada una en su sitio: el anuncio lo completa el documento; fuera de él
    # —la regla dicha por ti, la respuesta al criterio que invocó otro— el
    # órgano lo dices tú. Sin ejemplos entre comillas: se copiaban.
    ("- LA INSTANCIA VA SIEMPRE: «de la Primera Sala de la Suprema Corte de Justicia\n"
     "  de la Nación», «de la Segunda Sala», «del Pleno», «de un Tribunal Colegiado de\n"
     "  Circuito». Sin ella no se sabe qué peso tiene el criterio.",
     "- LA INSTANCIA, FUERA DEL ANUNCIO: en la oración que anuncia la cita la pone\n"
     "  el documento; cuando en tu prosa hables de un criterio —al decir su regla,\n"
     "  al contestar el que invocó otro—, di de qué órgano es, con su nombre\n"
     "  completo. Sin ella no se sabe qué peso tiene el criterio."),
    # La regla de las marcas (sólo con inventario): el precedente que trae.
    ("  reiteran otro se nombran juntos en el párrafo que los contesta.",
     "  reiteran otro se nombran juntos en el párrafo que los contesta. Si el dato\n"
     "  propio es un precedente que la parte invocó, su respuesta es una de las\n"
     "  tres que se describen para el criterio que invocó otro, nunca volver a\n"
     "  citarlo como apoyo."),
)
# LOS QUE SÓLO ESTÁN EN ALGUNOS ASUNTOS (integración del 28-sep-2026, AR
# 631/2025). Dos textos de la v2 empujaban a citar en bloque, y la técnica dice
# que un criterio que no cabe en ninguna de sus cuatro funciones no se invoca:
#   · «APOYOS PARA ESTA TÉCNICA … son las que hay que citar»: en el 631 (rama
#     revoca_fondo_niega, sin conceptos de violación) mandaba citar cuatro
#     tesis, dos de ellas sobre conceptos que descansan en otros desestimados
#     —un estudio que ese proyecto no puede hacer sin los conceptos—;
#   · «Apoyos del acervo para esta vía: … (cítalos …)» cuando el secretario
#     resolvió al revés que el motor.
# Siguen diciendo DÓNDE están los apoyos; ya no mandan citarlos todos.
_REEMPLAZOS_TECNICA_CONDICIONALES = (
    ("y son las que hay que citar al justificarla. Cítalas como las demás, desde "
     "el texto que se te dio.",
     "y sostienen los pasos de esta técnica: cada una se cita, desde el texto que "
     "se te dio, sólo para un paso que de verdad des en este estudio, con su regla "
     "dicha por ti y aplicada como cualquier otra; la que no sostiene ningún paso "
     "tuyo no se cita."),
    (" (cítalos desde su texto, abajo).",
     " (el que sostenga una premisa tuya se cita desde su texto, abajo, con la "
     "técnica de la cita; los demás no)."),
)

# Lo último que se lee es lo que más se obedece (va con el del guion).
# (Integración del 28-sep-2026: decía «ninguno de la parte como apoyo», y en
# la dirección en que la parte gana —el 631 con el sentido que dictó David— la
# técnica y la JERARQUÍA piden reconocerle la razón a quien invocó el criterio
# que decide. Lo prohibido es reciclarlo como hallazgo propio.)
_RECUERDA_TECNICA = (
    "Y CADA CRITERIO QUE CITES, CON SU REGLA DICHA POR TI Y APLICADA A UNA "
    "CONSTANCIA DEL EXPEDIENTE: ninguno en fila sin regla, ninguno de la parte "
    "reciclado como apoyo propio: a quien lo invocó se le contesta.\n")


def _con_tecnica(base: str, jerarquia: bool = False) -> str:
    """La v3 con la técnica de la cita. Cada ajuste, una vez; el que no
    encuentra su texto no hace nada (la regla de las marcas sólo está con
    inventario). Con JERARQUÍA, además, lo del criterio de la parte dentro de
    un grupo de consecuencia."""
    for viejo, nuevo in _REEMPLAZOS_TECNICA:
        if jerarquia and viejo == _TECNICA_PARTE_VIEJA:
            nuevo = nuevo + _TECNICA_PARTE_GRUPO
        base = base.replace(viejo, nuevo, 1)
    for viejo, nuevo in _REEMPLAZOS_TECNICA_CONDICIONALES:
        base = base.replace(viejo, nuevo)
    return base


# EL CONSIDERANDO DE LOS CONCEPTOS QUE NADIE ESTUDIÓ NO ESTÁ EN EL GUION
# (integración del 28-sep-2026, AR 631/2025). El plan organiza el estudio de
# los AGRAVIOS; cuando el recurso levanta un sobreseimiento (art. 93, frs. I y
# V) o revoca una concesión y reasume jurisdicción (fr. VI), el prompt pide
# además un considerando propio para los conceptos de violación. Con el guion
# diciendo «un apartado por cada APARTADO» y «si el guion te parece equivocado,
# síguelo igual», el modelo tenía cómo omitir ese considerando o mandarlo a
# DESVIACIONES DEL GUION. Sólo cuando el prompt trae los conceptos que hay que
# estudiar —si faltan, se pide lo contrario: no concluir y avisar— se dice
# dónde va ese considerando y que no es una desviación.
_MARCAS_CONSIDERANDO_CONCEPTOS = (
    "LOS CONCEPTOS DE VIOLACIÓN NO ESTUDIADOS,",               # fr. VI, con conceptos
    "ESTUDIO DE LOS CONCEPTOS DE VIOLACIÓN — UN CONSIDERANDO NUEVO",  # frs. I y V
)
_GUION_UN_APARTADO = "- Un apartado por cada APARTADO del guion, en su orden."
_GUION_Y_CONCEPTOS = (
    "- Un apartado por cada APARTADO del guion, en su orden. El guion ordena el\n"
    "  estudio de los agravios: el considerando de los conceptos de violación que\n"
    "  el tribunal estudia por primera vez, descrito más arriba, va después de su\n"
    "  último apartado, no está en el guion y no es una desviación de él; y la\n"
    "  conclusión con que se cierra —si procede conceder o negar— es parte de ese\n"
    "  considerando, no un párrafo de cierre.")
_RECUERDA_CONCEPTOS = (
    "DESPUÉS DEL ÚLTIMO APARTADO DEL GUION, EL CONSIDERANDO DE LOS CONCEPTOS DE "
    "VIOLACIÓN QUE NADIE ESTUDIÓ, con su conclusión.\n")


def _prompt_estudio_v4(args: dict) -> str:
    import copy as _copy_v4
    import plan_estudio as _pe
    guion = str(args.pop("guion", "") or "")
    _m3 = _copy_v4.copy(args["material"])
    _m3.variante = "v3"
    args["material"] = _m3
    base = prompt_estudio(**args)
    blq = _pe.bloque(guion)
    if not blq:
        return base
    _jer = _pe._RX_JERARQUIA in guion
    if _jer:
        base = _con_jerarquia(base)
    # LA TÉCNICA DE LA CITA, con guion (ver `_REEMPLAZOS_TECNICA`).
    base = _con_tecnica(base, _jer)
    _recuerda = _RECUERDA_GUION + _RECUERDA_TECNICA
    if any(m in base for m in _MARCAS_CONSIDERANDO_CONCEPTOS):
        blq = blq.replace(_GUION_UN_APARTADO, _GUION_Y_CONCEPTOS, 1)
        _recuerda = _RECUERDA_GUION + _RECUERDA_CONCEPTOS + _RECUERDA_TECNICA
    i = base.rfind(_MARCA_ESCRIBE)
    if i < 0:
        return base.rstrip() + "\n" + blq + "\n" + _recuerda
    con = base[:i] + "\n" + blq + base[i:]
    j = con.rfind("\nNada más.")
    if j >= 0 and not con[j + len("\nNada más."):].strip():
        return con[:j] + "\n" + _recuerda.rstrip("\n") + con[j:]
    return con.rstrip() + "\n" + _recuerda


# ═══════════════════════════════════════════════════════════════════════════
# Verificación antes de entregar
# ═══════════════════════════════════════════════════════════════════════════

# Un registro digital NO es cualquier cifra de seis o siete dígitos. En los
# expedientes aparecen números de recibo —«2024/2837851»—, de operación y de
# expediente que casan igual, y contarlos como tesis inventadas produce alarmas
# falsas justo donde la alarma tiene que valer: probado sobre el ARA 103-2025,
# los dos «registros inventados» eran los recibos de pago del Registro Público.
#
# Se exige que la cifra vaya ANUNCIADA como registro, que es como se cita.
_RX_REGISTRO = re.compile(r"\b(\d{6,7})\b")
_RX_REGISTRO_CITA = re.compile(
    r"(?:registro(?:\s+digital)?|reg\.)\s*[:\s]\s*(\d{6,7})\b", re.I)
# «ineficaz» lleva z y «ineficaces» c: la raíz «ineficac» sola NUNCA casaba
# la forma singular, y el verificador acusaba de no calificar a un estudio
# que calificaba en su primera línea.
_RX_CALIF = re.compile(r"\b(fundad|infundad|inoperant|inefica[cz])\w*", re.I)

# El nombre de la ley se lee en lo que SIGUE al número, no con un patrón que
# intente prever cómo se escribe. El primer intento exigía «de la|el|los|las» y
# no casaba «del Código Civil», que es la forma más común: cero hallazgos y un
# verificador que parecía funcionar porque nunca decía nada.
_RX_ARTICULO = re.compile(r"art[íi]culos?\s+(\d{1,4})\s*(?:bis|ter)?\.?\s*([^.;:]{0,80})", re.I)

# LAS CITAS DE ARTÍCULOS, UNA POR NÚMERO. «artículos 134 y 137 del Código
# Fiscal de la Federación» son dos citas; «artículo 6º de la LFPCA» lleva el
# ordinal pegado; y «artículo 68 del Código Fiscal de la Federación y 42 de la
# Ley Federal de Procedimiento Contencioso Administrativo» son dos leyes. El
# regex de arriba veía UNA cita en cada caso (61/2025: el 134, el 6 y el 42
# se quedaban sin traer ni transcribir).
# LA COLA EN UN LOOKAHEAD Y CORTADA EN EL ARTÍCULO SIGUIENTE (AR 631/2025).
# Consumida, se tragaba la lista que venía detrás: en el agravio «los artículos
# 14 y 16 de la Constitución…, así como los artículos 49, 57 y 279 del Código
# Procesal Civil… y 2284 y 2294 del Código Civil local» sólo se leían el 14 y
# el 16, y los cinco preceptos que deciden el asunto no se traían al material
# antes de proponer ni de redactar.
_RX_ARTICULOS_LISTA = re.compile(
    r"art[íi]culos?\s+(?P<lista>\d{1,4}(?:\s*(?:º|°|o\.|bis|ter))?"
    r"(?:\s*(?:,|y|e)\s*\d{1,4}(?:\s*(?:º|°|o\.|bis|ter))?)*)\.?\s*(?=(?P<cola>[^.;:]{0,110})(?P<mas>[^.;:]{0,110}))",
    re.I)
# «y 2284 y 2294 del Código Civil local»: también una LISTA tras la «y».
_RX_Y_OTRO_DE_LEY = re.compile(
    r"\b(?:y|e)\s+(?:el\s+|los\s+)?(\d{1,4}(?:\s*(?:º|°|o\.))?"
    r"(?:\s*(?:,|y|e)\s*\d{1,4}(?:\s*(?:º|°|o\.))?)*)\s+((?:de|del)\s+(?:la|el|los|las)?\s*"
    r"(?:constituci[óo]n|c[óo]digo|ley|reglamento|convenci[óo]n|pacto|tratado)[^,;:()]{4,90})", re.I)
_RX_OTRO_ARTICULO = re.compile(r"\bart[íi]culos?\s+\d", re.I)


def citas_de_articulos(texto: str) -> list:
    """[(número, cola)] de cada artículo citado, con la cola —lo que sigue— que
    nombra la ley. Reparte las listas y las continuaciones «y N de la Ley».
    Las perífrasis —«legislación procesal civil»— se leen como el código
    (`documento_generado.canonizar_ley`)."""
    import documento_generado as _dg_cl
    fuera = []
    for m in _RX_ARTICULOS_LISTA.finditer(_dg_cl.canonizar_ley(texto or "")):
        nums = re.findall(r"\d{1,4}", m.group("lista"))
        cola = " ".join(m.group("cola").split())
        cola = _RX_OTRO_ARTICULO.split(cola, 1)[0]
        # la primera ley acaba donde empieza «y 42 de la Ley…»
        cont = _RX_Y_OTRO_DE_LEY.search(cola)
        cola_1 = cola[:cont.start()] if cont else cola
        for nu in nums:
            fuera.append((nu, cola_1))
        if cont:
            # EL NOMBRE DE LA SEGUNDA LEY, ENTERO (humo del AR 631/2025,
            # 29-sep-2026): «artículos 14 y 16 de la Constitución Política de
            # los Estados Unidos Mexicanos y 49, 57 y 279 del Código de
            # Procedimientos Civiles del Estado de Querétaro» — la ventana de
            # 110 lo dejaba en «…Procedimientos Civile» y el verificador
            # acusaba como ausentes tres artículos que estaban en el material.
            # Qué es continuación se decide como siempre, en la ventana corta;
            # sólo su ley se completa con lo que sigue.
            ley_2 = cont.group(2)
            larga = " ".join((m.group("cola") + (m.group("mas") or "")).split())
            larga = _RX_OTRO_ARTICULO.split(larga, 1)[0]
            c2 = _RX_Y_OTRO_DE_LEY.match(larga, cont.start())
            if c2 and c2.group(1) == cont.group(1) and c2.group(2).startswith(ley_2.rstrip()):
                ley_2 = c2.group(2)
            for nu in re.findall(r"\d{1,4}", cont.group(1)):
                fuera.append((nu, ley_2))
    return fuera

_VACIAS = {"de", "del", "la", "el", "los", "las", "y", "en", "que", "propio",
           "citado", "mencionado", "invocado", "referido", "aludido", "a", "su"}

# «si la copropiedad fue efectivamente reconocida», «se afirma que», «de ser
# cierto»: un tribunal con los autos delante no supone lo que consta.
# «COMO SI … HUBIERA» NO SUPONE NADA: es una comparación retórica («como si
# en ella se hubiera producido la preclusión»). ADC 93/2026 v4 acusó dos así.
_RX_CONDICIONAL = re.compile(
    r"\b(?<!como\s)(si\s+(?:\w+\s+){0,3}(?:fue|fuera|hubiera|resultara|efectivamente)"
    r"|se\s+afirma\s+que|seg[úu]n\s+lo\s+planteado|de\s+ser\s+cierto"
    r"|en\s+el\s+supuesto\s+de\s+que|de\s+haberse\s+acreditado)\b", re.I)

# LAS LEYES QUE NO HACE FALTA TENER EN EL MATERIAL PARA CITARLAS. Son las que
# gobiernan el juicio pase lo que pase, y exigirlas en las NORMAS acusaría a todo
# proyecto correcto.
_NOTORIAS = ("ley de amparo", "constitución", "constitucion",
             "ley orgánica del poder judicial", "ley organica del poder judicial")

# …PERO LA LEY DE AMPARO DEJA DE SER NOTORIA CUANDO NO RIGE EL ACTO.
#
# Medido en el amparo en revisión 322/2025, tras arreglar la recuperación: el
# material ya no traía NI UNA norma ni UNA tesis de la suspensión del amparo, y
# el estudio siguió citando los artículos 128 y 147 —de memoria del modelo—
# para juzgar una medida provisional que dictó un juez de primera instancia. El
# verificador de preceptos inventados no protestaba porque exime «ley de amparo»
# por su nombre.
#
# Esa exención es correcta casi siempre: la Ley de Amparo rige la procedencia,
# la oportunidad, la legitimación, la suplencia y el resolutivo de CUALQUIER
# asunto de colegiado. Deja de serlo justo cuando el sistema ha DERIVADO que el
# acto reclamado lo dictó una autoridad ordinaria y que el recurso no va contra
# el incidente de suspensión: ahí, un artículo de la Ley de Amparo que no esté
# en el material es exactamente lo que hay que denunciar.
#
# Y sólo se levanta para el TRAMO DE LA SUSPENSIÓN, los artículos 125 a 169. Los
# demás —61, 63, 74, 76, 79, 81, 86, 93— siguen exentos, porque siguen rigiendo.
_SUSPENSION_LA = range(125, 170)


# Las comillas de un rubro llegan de tres formas —tipográficas, latinas y
# rectas— según lo que escriba el modelo. Cubrir sólo una deja el verificador
# mudo: probado con « » y no saltaba ni una alarma.
_RX_RUBRO_CITADO = re.compile(
    r"[“«\"]\s*[A-ZÁÉÍÓÚÑ][^”»\"]{25,}[”»\"]")
# LA CLAVE DE UNA TESIS —«2a./J. 58/2010», «I.3o.C. 12/2024»—. Sirve para
# saber si un estudio se apoyó en alguna, aunque no dé el registro: medido en
# los engroses Kingston, casi ninguno escribe «registro» en el cuerpo y todos
# los que citan dan rubro o clave (v2 de `revisar`, 26-sep-2026).
_RX_CLAVE_TESIS = re.compile(
    r"\b(?:\d{1,2}a\.|P\.|PC\.|[IVXL]+\.\d{1,2}o\.[A-Z]?\.?)\s*/?\s*J?\.?\s*\d{1,4}/\d{4}")


def _frases(t: str) -> set:
    return {re.sub(r"\W+", " ", f).strip().lower()
            for f in re.split(r"(?<=[.])\s+", t or "")
            if len(f.split()) >= 8}


def _solapamiento(a: str, b: str) -> float:
    """Qué proporción de las frases de `a` reaparece casi igual en `b`."""
    fa, fb = _frases(a), _frases(b)
    if not fa:
        return 0.0
    # Se compara por trozos: una frase reescrita comparte casi todas sus palabras.
    voc_b = [set(f.split()) for f in fb]
    repetidas = 0
    for f in fa:
        p = set(f.split())
        if any(len(p & v) / max(1, len(p)) > 0.75 for v in voc_b):
            repetidas += 1
    return repetidas / len(fa)


# ═══ LA LEY AJENA ═══════════════════════════════════════════════════════════
# Barrido de 139 documentos de este tribunal: CERO aplicaciones de ley de otra
# entidad. Es la regla más firme del corpus y la que el redactor rompió en el
# proyecto 360/2025 —«por analogía y por tratarse de legislación diversa»—.
#
# LA TRAMPA, y por eso el primer arreglo produjo alarmas falsas: el texto de una
# tesis transcrita NOMBRA la ley que interpretó —«los artículos 940 y 941 del
# Código de Procedimientos Civiles para el Distrito Federal»— y eso es CITA, no
# aplicación. Verificadas las 38 apariciones de códigos ajenos en el corpus: las
# 38 van dentro de una tesis o de un precedente transcrito.
#
# El deslinde es mecánico: lo entrecomillado es cita ajena; lo demás es la prosa
# del secretario, y ahí la ley de fuera no puede estar.
# LAS 32, Y LA AJENA SE CALCULA. Esta lista tenía 31 entidades: todas menos
# Querétaro, escrito así porque el verificador nació para un tribunal de
# Querétaro. El resultado es que un secretario de Yucatán que aplica —bien— el
# Código Civil de Yucatán recibía la acusación de estar invocando ley ajena,
# mientras que aplicar el de Querétaro pasaba sin que nadie dijera nada. Y ni
# siquiera servía a este tribunal: el Vigésimo Segundo Circuito cubre Querétaro
# E HIDALGO, e Hidalgo estaba en la lista de ajenas.
#
# Ahora se declaran las 32 y la ajena es «todas menos la del asunto», que sale
# del material.
_ENTIDADES_TODAS = (
    "QUERETARO",
    "AGUASCALIENTES", "BAJA CALIFORNIA", "CAMPECHE", "COAHUILA", "COLIMA",
    "CHIAPAS", "CHIHUAHUA", "DISTRITO FEDERAL", "CIUDAD DE MEXICO", "DURANGO",
    "GUANAJUATO", "GUERRERO", "HIDALGO", "JALISCO", "MEXICO", "MICHOACAN",
    "MORELOS", "NAYARIT", "NUEVO LEON", "OAXACA", "PUEBLA", "QUINTANA ROO",
    "SAN LUIS POTOSI", "SINALOA", "SONORA", "TABASCO", "TAMAULIPAS", "TLAXCALA",
    "VERACRUZ", "YUCATAN", "ZACATECAS", "BAJA CALIFORNIA SUR",
)
_RX_NORMA_CERCA = re.compile(r"(c[óo]digo|ley|legislaci[óo]n|art[íi]culos?|"
                             r"reglamento)", re.I)
# Si la mención cuelga de un CRITERIO, no es aplicación de ley ajena sino cita
# de jurisprudencia ajena —que sí está permitida—. Distinguirlo importa: en el
# 360/2025, «el criterio referido a la legislación del Estado de Puebla» es una
# excusa territorial mal escrita, no la aplicación del código poblano, y
# acusarla de lo segundo manda al secretario a buscar un error que no está ahí.
_RX_ES_CRITERIO = re.compile(r"(criterio|tesis|jurisprudencia|precedente|"
                             r"contradicci[óo]n|rubro)", re.I)

# Las fórmulas que ponen la entidad ajena como razón. Ninguna aparece en el
# corpus; la primera es la que el redactor escribió y hay que matar.
_RX_EXCUSA_ENTIDAD = re.compile(
    r"(por\s+tratarse\s+de\s+legislaci[óo]n\s+diversa"
    r"|legislaci[óo]n\s+diversa\s+a\s+la\s+aplicable"
    r"|aunque\s+(?:referid[oa]|se\s+refiera)\s+a\s+la\s+legislaci[óo]n\s+d"
    r"|aunque\s+referid[oa]s?\s+a\s+otras?\s+legislaci"
    r"|si\s+bien\s+(?:se\s+trata|corresponde|es)\s+de\s+(?:una\s+)?"
    r"legislaci[óo]n\s+(?:de\s+otr|diversa|ajena)"
    r"|por\s+analog[íi]a\s+y\s+por\s+tratarse)", re.I)


def _prosa_propia(estudio: str, material=None) -> str:
    """Lo que escribió el secretario, sin lo que transcribe de otros.

    EL DESLINDE NO PUEDE SER POR COMILLAS. Comprobado sobre el proyecto
    360/2025: el ensamblador pega el texto de la tesis como párrafo suelto, SIN
    comillas, y ahí dentro van «los artículos 940 y 941 del Código de
    Procedimientos Civiles para el Distrito Federal». Filtrando por comillas
    salían cuatro infracciones donde no había ninguna.

    El deslinde exacto es contra el ACERVO: lo que coincide con el texto de una
    tesis que el acervo entregó es transcripción; lo demás es prosa propia.
    """
    t = re.sub(r"[“«\"][^”»\"]{20,}[”»\"]", " ", estudio)
    t = re.sub(r"\([^)]{0,120}LEGISLACI[ÓO]N[^)]{0,80}\)", " ", t, flags=re.I)
    if material is None:
        return t
    fuente = " ".join(_norm_frase(x.get("texto", "") + " " + x.get("rubro", ""))
                      for x in getattr(material, "tesis", []) or [])
    if not fuente.strip():
        return t
    quedan = []
    for frase in re.split(r"(?<=[.;:])\s+", t):
        n = _norm_frase(frase)
        if len(n) > 40 and n in fuente:
            continue          # está en el acervo palabra por palabra: es cita
        quedan.append(frase)
    return " ".join(quedan)


def _norm_frase(x: str) -> str:
    x = re.sub(r"[^a-z0-9 ]+", " ", _sin_acentos_est(x).lower())
    return re.sub(r"\s+", " ", x).strip()


def _sin_acentos_est(x: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", (x or "").upper())
                   if not unicodedata.combining(c))


def _ajenas_para(material) -> tuple:
    """Las entidades que NO son la del asunto."""
    import unicodedata
    ent = str(getattr(material, "entidad", "") or "").strip()
    if not ent:
        # SIN ENTIDAD DECLARADA NO SE ACUSA A NADIE. No saber de qué estado es
        # el asunto no autoriza a suponer que es de Querétaro.
        return ()
    x = unicodedata.normalize("NFKD", ent.upper())
    x = "".join(c for c in x if not unicodedata.combining(c))
    return tuple(e for e in _ENTIDADES_TODAS if e != x)


def _leyes_ajenas_aplicadas(estudio: str, material=None) -> list[str]:
    """Entidades cuya LEY se invoca en PROSA PROPIA. Las transcripciones no cuentan."""
    limpio = _sin_acentos_est(_prosa_propia(estudio, material))
    halladas: list[str] = []
    for ent in _ajenas_para(material):
        for m in re.finditer(r"\b" + re.escape(ent) + r"\b", limpio):
            # «Estado de México» exige el rótulo; «México» a secas es el país.
            if ent == "MEXICO" and not re.search(
                    r"ESTADO\s+DE\s*$", limpio[max(0, m.start() - 12):m.start()]):
                continue
            ventana = limpio[max(0, m.start() - 140):m.start()]
            # LA VENTANA MIRA A LOS DOS LADOS. En el 360/2025 la palabra que
            # delata la cita va DETRÁS —«…del Estado de Puebla, resulta
            # ilustrativo el criterio…»— y mirando sólo hacia atrás el aviso
            # salía como aplicación de ley poblana, que no es lo que ocurrió.
            if _RX_ES_CRITERIO.search(ventana + limpio[m.end():m.end() + 90]):
                continue      # es jurisprudencia ajena: permitida
            if _RX_NORMA_CERCA.search(ventana[-110:]):
                halladas.append(ent.title())
                break
    return halladas



# ═══════════════════════════════════════════════════════════════════════════
# LO QUE LA MEDICIÓN AÑADIÓ A LA REVISIÓN
# ═══════════════════════════════════════════════════════════════════════════

# LA QUE NO SOBREVIVIÓ A SU PROPIA CALIBRACIÓN. Escribí una verificación de
# «citas huérfanas»: después de cada cita debía aparecer, en los 1,200
# caracteres siguientes, la fórmula que dijera qué se hace con ella. La probé
# contra el acervo antes de enviarla y saltó en el 95% de las citas de las
# sentencias de calidad 5 y en el 100% de las de calidad 3. No distinguía nada.
#
# La causa era estructural, no de vocabulario: la amplié y siguió saltando en el
# 85%. En una sentencia real la fórmula va DELANTE de la cita —«resulta
# aplicable la jurisprudencia 2a./J. 52/98, registro 195741, que dice: …»— y yo
# la buscaba detrás. Mirando hacia atrás habría pasado todo, porque esa fórmula
# es justamente la estándar. La comprobación no medía nada y se quitó.
#
# Lo que sí reprodujo la calibración, y con holgura: en laboral las de calidad 5
# citan 6 tesis de mediana y las de calidad 3 citan 2 —84 citas en 12 estudios
# contra 20 en 13—. En civil la mediana es 2 en los dos niveles. Por eso el
# mínimo de cinco registros va SÓLO en el bloque laboral del prompt.

_RX_ANCLAJE = re.compile(
    r"(en\s+el\s+caso\s+concreto|en\s+la\s+especie|en\s+el\s+caso\s+a\s+estudio"
    r"|en\s+el\s+asunto\s+que\s+nos\s+ocupa|en\s+el\s+caso\s+que\s+se\s+analiza"
    r"|en\s+el\s+caso\s+sujeto\s+a\s+estudio|en\s+el\s+presente\s+(?:caso|asunto))",
    re.I)

_RX_COIDH = re.compile(r"Corte\s+Interamericana|interamerican|convencionalidad", re.I)
_RX_CASO_CoIDH = re.compile(
    r"\bcaso\s+[A-ZÁÉÍÓÚ][\w.\-]*.{0,60}?\bvs?\.?\s+[A-ZÁÉÍÓÚ]", re.I)
_RX_PARRAFO_CoIDH = re.compile(r"p[áa]rr(?:afo)?s?\.?\s*\d+|§\s*\d+", re.I)

_RX_ORDEN_NUMERADA = re.compile(
    r"^\s*(?:\d+[.)]|[a-z][.)])\s*(?:deje|dej[eé]|declare|reponga|emita|dicte"
    r"|resuelva|ordene|deber[áa]|proceda|realice|valore"
    # EN INFINITIVO desde el 27-sep-2026: las órdenes cuelgan de «deberá:».
    r"|dejar|declarar|reponer|emitir|dictar|resolver|ordenar|proceder|realizar"
    r"|valorar|admitir|analizar|examinar|reiterar|pronunciarse|hecho\s+lo\s+anterior)",
    re.I | re.M)


# LA SEGUNDA QUE NO SOBREVIVIÓ. El hallazgo más vistoso del análisis era que las
# buenas cierran el circuito regla→caso de tres a cinco veces, con el primer
# anclaje al 20% del texto, contra un ciclo largo y el 60% en las medias. Quise
# convertirlo en comprobación y lo intenté tres veces:
#
#   · contando anclajes y exigiendo tres → saltaba en 6 de 6 sentencias de
#     calidad 5, porque la mediana real que medí es 1.5, no 4;
#   · exigiendo al menos uno en estudios largos → 3 de 8 civiles de calidad
#     máxima acusadas, y en civil la señal está invertida: el primer anclaje
#     llega al 48% en las buenas y al 31% en las medias;
#   · acotada sólo a laboral → 2 de 6 buenas contra 1 de 8 medias. Al revés otra
#     vez, y por una razón simple: las buenas son más largas (33 mil caracteres
#     de mediana contra 20 mil), así que pasan el umbral de longitud más a
#     menudo y se exponen más al aviso.
#
# El rasgo puede ser cierto y aun así no ser comprobable con una expresión
# regular: «volver al caso» se escribe de cien maneras y sólo cuento cinco. La
# instrucción SIGUE EN EL PROMPT —ahí es un consejo y no cuesta nada— pero no se
# convierte en acusación automática. Una comprobación que señala a una de cada
# tres sentencias bien escritas no protege al secretario: le enseña a no leer
# los avisos, y el día que salte uno de verdad tampoco lo leerá.

_RX_INSTRUMENTO = re.compile(
    r"(convenci[óo]n\s+americana|pacto\s+de\s+san\s+jos[ée]|pacto\s+internacional"
    r"|convenci[óo]n\s+(?:sobre|de|interamericana)|protocolo\s+de\s+san\s+salvador"
    r"|declaraci[óo]n\s+(?:americana|universal))", re.I)


# LA TERCERA QUE NO SOBREVIVIÓ A SU CALIBRACIÓN. Un auditor propuso avisar
# cuando la suplencia sólo sirve para negar —«no permite», «aun bajo», «no
# significa»—, que es exactamente lo que hace el ADL 382/2024. La escribí y la
# probé contra ese mismo documento: cuenta SIETE usos y sólo TRES niegan; los
# otros anuncian la suplencia, la usan para justificar la inoperancia o la usan
# bien. Para que saltara tenía que bajar el umbral hasta ajustarlo a este caso,
# y ajustar un detector a un documento es la definición de no medir nada.
#
# El defecto es real —la suplencia no produce ni un examen de oficio— pero es
# una propiedad semántica y no la sé detectar con una expresión regular. Lo que
# sí se detecta, y ya está arriba, es que el estudio no deje ESCRITA la versión
# suplida que examinó. Ésa saltó a la primera en este documento y basta.

def _tipo_mal_atribuido(estudio: str, material) -> str:
    """Llamar «jurisprudencia» a una tesis aislada, en la prosa.

    El anuncio de la cita ya lo compone el documento con los campos del acervo,
    así que ahí el error es imposible. Pero el modelo sigue escribiendo prosa
    alrededor —«conforme a la jurisprudencia citada…»— y ahí sí puede
    equivocarse. Es barato comprobarlo: se mira si en el entorno de cada
    registro de una tesis AISLADA aparece la palabra jurisprudencia.

    No se comprueba al revés. Llamar «criterio» o «tesis» a una jurisprudencia
    es impreciso pero no falso; llamar jurisprudencia a lo que no lo es le
    atribuye una fuerza vinculante que no tiene, y eso sí cambia el fallo.
    """
    aisladas = {str(x.get("registro")): x for x in (getattr(material, "tesis", None) or [])
                if "AISLAD" in str(x.get("tipo") or "").upper()}
    if not aisladas or not estudio:
        return ""
    # LA VENTANA ES LA FRASE, no un número de caracteres. Con 260 a cada lado,
    # «Conforme al criterio de registro 191358… Y la jurisprudencia 2001812
    # obliga» daba positivo: la palabra pertenecía a la OTRA cita. La
    # atribución vive en la misma oración que el registro, y ahí se busca.
    todos = {str(x.get("registro")) for x in (getattr(material, "tesis", None) or [])
             if x.get("registro")}
    malas = []
    for reg in aisladas:
        for m in re.finditer(re.escape(reg), estudio):
            ini = max((estudio.rfind(c, 0, m.start()) for c in ".;\n"), default=-1) + 1
            fin = min((x for x in (estudio.find(c, m.end()) for c in ".;\n")
                       if x != -1), default=len(estudio))
            frase = estudio[ini:fin]
            # Y si en esa misma oración hay otro registro, no se puede saber a
            # cuál se refiere la palabra: no se acusa.
            if len([r for r in todos if r in frase]) > 1:
                continue
            if re.search(r"jurisprudencia", frase, re.I):
                malas.append(reg)
                break
    if malas:
        return (f"Se llama JURISPRUDENCIA a {'la tesis aislada' if len(malas)==1 else 'las tesis aisladas'} "
                f"de registro {', '.join(sorted(malas))}. Una tesis aislada "
                f"orienta, no vincula: atribuirle fuerza obligatoria cambia el "
                f"peso del argumento que sostiene.")
    return ""


def _convencional_completo(estudio: str) -> str:
    """Que lo interamericano se pueda comprobar. No que se cite más.

    ESTA TAMBIÉN SE RECORTÓ CON LA CALIBRACIÓN, aunque menos. Empecé exigiendo
    las tres piezas —instrumento, caso con nombre y número de párrafo— y saltaba
    en 4 de cada 10 sentencias administrativas de calidad 5. Estaba señalando lo
    normal: citar el artículo 25 de la Convención Americana sin nombrar ningún
    caso de la Corte Interamericana es correcto y frecuente.

    Quedan los dos supuestos donde de verdad se esconde el error, y ninguno es
    cuestión de estilo:

    1. SE NOMBRA UN CASO Y NO SE DICE DÓNDE. «Caso Fulano vs. México» sin
       párrafo es una atribución que nadie puede comprobar, y es exactamente la
       forma que toma una cita inventada.
    2. SE INVOCA LA CONVENCIONALIDAD SIN NADA DETRÁS: ni instrumento, ni
       artículo, ni caso. El análisis del acervo la encontró como contraseñal
       —aparenta altura y no sostiene nada—.
    """
    texto = estudio or ""
    if not _RX_COIDH.search(texto):
        return ""
    caso = _RX_CASO_CoIDH.search(texto)
    if caso and not _RX_PARRAFO_CoIDH.search(texto):
        return ("Se nombra un caso de la Corte Interamericana "
                f"(«{caso.group(0)[:60]}…») sin número de párrafo. Sin "
                "localizador la atribución no se puede comprobar, y ésa es la "
                "forma que toma una cita inventada: o se completa o se quita.")
    if not caso and not _RX_INSTRUMENTO.search(texto):
        return ("Se invoca el control de convencionalidad o a la Corte "
                "Interamericana sin nombrar instrumento ni caso. Una invocación "
                "que no se apoya en nada aparenta altura y no sostiene el fallo.")
    return ""


_RX_REPOSICION = re.compile(
    r"insubsistente|sin\s+efectos|reponga|reponer|reposici[óo]n|admita|admitir|emplace|"
    r"emplazar|corra\s+traslado|traslado|desahog", re.I)
_RX_NUEVA_SENTENCIA = re.compile(
    r"nueva\s+(?:sentencia|resoluci[óo]n)|dicte\s+otra|dictar\s+otra|otra\s+(?:sentencia|resoluci[óo]n)|"
    r"plenitud\s+de\s+jurisdicci[óo]n", re.I)


def _concede_el_proyecto(estudio: str, criterios: list, rama: str = "") -> bool:
    """¿Este proyecto concede —y le toca fijar efectos—?

    CON LA RAMA MANDA LA RAMA (28-sep-2026, AR 631/2025). Se deducía de que
    algún planteamiento prosperara, y en una revisión fundada que revoca una
    concesión y NIEGA eso es falso: el 631 salió con «Se concede y los efectos
    van en prosa» y «EFECTOS INCOMPLETOS PARA UNA VIOLACIÓN PROCESAL» en un
    proyecto que no concede nada. La rama se corrige con lo que el estudio
    concluyó al reasumir jurisdicción (`tipos_asunto.ejecutoria_concede`): si
    los conceptos no estudiados prosperan, se concede por razón distinta y
    entonces sí hay efectos. Sin rama —amparo directo—, como antes."""
    import tipos_asunto as _ta_ce
    if rama:
        import fase_rama as _fr_ce
        _c = _ta_ce.ejecutoria_concede(rama, _fr_ce.sentido_en_plenitud(estudio or ""))
        if _c is not None:
            return _c
    return any(_ta_ce.prospera(str(getattr(c, "sentido", "") or c)) for c in (criterios or []))


def _efectos_de_reposicion(estudio: str, criterios: list, violacion_procesal: bool,
                           rama: str = "") -> str:
    """Si se concede por violación procesal, los efectos tienen que ordenar la
    reposición paso a paso, no «dicte otra». ADC 93/2026 v5. Con la rama de la
    revisión, sólo si ESTE proyecto concede (`_concede_el_proyecto`)."""
    if not violacion_procesal:
        return ""
    if not _concede_el_proyecto(estudio, criterios, rama):
        return ""
    t = str(estudio or "")
    i = t.upper().rfind("EFECTOS DE LA CONCESIÓN")
    bloque = t[i:] if i >= 0 else t[-2500:]
    pasos = len(set(m.group(0).lower() for m in _RX_REPOSICION.finditer(bloque)))
    if pasos >= 3 and _RX_NUEVA_SENTENCIA.search(bloque):
        return ""
    return ("EFECTOS INCOMPLETOS PARA UNA VIOLACIÓN PROCESAL: se concede por una "
            "irregularidad del procedimiento y la responsable no puede dictar otra "
            "sentencia de inmediato. Los efectos tienen que ordenar la reposición "
            "paso a paso —dejar insubsistente la sentencia, dejar sin efectos la "
            "actuación viciada y la resolución del recurso que la confirmó, admitir "
            "lo que se desechó y seguir el procedimiento, y sólo entonces dictar la "
            "nueva sentencia—, en no más de cinco órdenes. Un «dictar otra» aquí produce "
            "requerimientos de cumplimiento defectuoso (artículos 192 a 196 de la "
            "Ley de Amparo).")


def _cierre_operativo(estudio: str, criterios: list, rama: str = "") -> str:
    """Si se concede, los efectos van como órdenes numeradas y verificables.

    EL SUBSTRING QUE DABA POR CONCEDIDO TODO LO NEGADO: «fundado» está dentro
    de «in-fundado», así que `"fundado" in sentido` era cierto para CADA
    planteamiento declarado infundado. Medido el 16-sep-2026 en la revisión
    fiscal 2/2026: el proyecto desecha el recurso por extemporáneo y el aviso
    le decía al secretario que revisara los EFECTOS DE LA CONCESIÓN. Un aviso
    falso no es ruido inofensivo: se leen los veinticuatro, y los falsos
    entierran a los verdaderos.

    `tipos_asunto.prospera` es el único sitio donde se decide esto, y lleva
    dentro la excepción que cuesta medir —«fundado_insuficiente» tiene
    «fundad» y NO prospera—.

    Y CON LA RAMA DE LA REVISIÓN, la rama (28-sep-2026): que un agravio
    prospere no es que el proyecto conceda (`_concede_el_proyecto`).
    """
    concede = _concede_el_proyecto(estudio, criterios, rama)
    if not concede or "efecto" not in (estudio or "").lower():
        return ""
    if not _RX_ORDEN_NUMERADA.search(estudio or ""):
        return ("Se concede y los efectos van en prosa. Van como lista "
                "numerada de tres a cinco órdenes en infinitivo, que cuelgan de "
                "«Con fundamento en el artículo 77 de la Ley de Amparo, la "
                "autoridad responsable deberá:» —«1. Dejar insubsistente el "
                "laudo; 2. Dictar otro en el que…»—, cada una verificable. Así "
                "se cumplen y así se comprueba su cumplimiento.")
    return ""


# Palabras con las que una sentencia se refiere a una ley YA nombrada, en vez
# de nombrarla. Nunca forman parte del nombre de un ordenamiento.
_RX_ANAFORICA = re.compile(
    r"\b(relativ[ao]s?|citad[ao]s?|invocad[ao]s?|mencionad[ao]s?|aludid[ao]s?|"
    r"referid[ao]s?|indicad[ao]s?|se[ñn]alad[ao]s?|aplicable|aplicables|"
    r"en\s+cita|en\s+consulta|de\s+la\s+materia|del\s+ramo|en\s+comento|"
    r"de\s+m[ée]rito|supracitad[ao]s?|antes\s+citad[ao]s?)\b", re.I)


def preceptos_fuera(estudio: str, material: Material) -> tuple:
    """(etiquetas, pares) de los artículos que el estudio cita y el material no
    trae. Las etiquetas van al aviso; los pares —(cuerpo legal, artículo)— se
    los lleva el resolver para traerlos del acervo si existen (322/2025: el
    artículo 210 del código procesal de Querétaro, citado bien y no traído)."""
    en_material = {(str(n.get("cuerpo_legal", "")).lower(), str(n.get("articulo", "")))
                   for n in material.normas}
    leyes_material = {c for c, _ in en_material}
    def _voces(x: str) -> set[str]:
        return {w for w in re.findall(r"[\wáéíóúñ]+", x.lower())
                if w not in _VACIAS and len(w) > 2}

    # ¿EL SISTEMA DERIVÓ QUE LA LEY DE AMPARO NO RIGE ESTE ACTO? Ver arriba.
    _sin_susp = (str(getattr(material, "sede_del_acto", "")) == "ordinaria"
                 and str(getattr(material, "cuaderno", "")) == "principal")
    fuera: set[str] = set()
    pares: set = set()
    for art, cola in citas_de_articulos(estudio):
        cola_n = " ".join(cola.split()).lower()
        if any(n in cola_n for n in _NOTORIAS):
            # La excepción de la excepción: el capítulo de la suspensión del
            # amparo, cuando el acto no se rige por esa ley.
            if _sin_susp and "amparo" in cola_n:
                try:
                    if int(art) in _SUSPENSION_LA and (
                            "ley de amparo", str(art)) not in en_material:
                        fuera.add(f"artículo {art} de la Ley de Amparo")
                except (TypeError, ValueError):
                    pass
            continue
        vc = _voces(cola_n)
        # La ley se reconoce por sus voces propias, pero hay que quedarse con la
        # QUE MÁS CASA, no con la primera. «Código Civil del Estado de Querétaro»
        # y «Código Civil Federal» comparten «código» y «civil»: con el primer
        # acierto ganaba el federal y el verificador denunciaba como inventado un
        # artículo correctamente citado del código local. Un aviso falso enseña a
        # ignorar los avisos, que es peor que no tenerlos.
        mejor, puntos = None, 0
        # POR IDENTIDAD, NO POR PALABRAS COMPARTIDAS. «Ley Federal de
        # Procedimiento Contencioso Administrativo» compartía cuatro palabras
        # con la LFPA del material y se daba por ella (61/2025). Si el estudio
        # nombra la ley, se casa con `misma_ley`; el recuento de voces queda
        # sólo para cuando no la nombra entera.
        _mc = re.search(r"(?:de|del)\s+(?:la|el|los|las)?\s*"
                        r"((?:constituci[óo]n|c[óo]digo|ley|reglamento|convenci[óo]n|pacto|tratado)"
                        r"[^,;:()]{4,90})", " ".join(cola.split()), re.I)
        _citado = ""
        if _mc:
            _pal = " ".join(_mc.group(1).split()).strip(" .").split()
            _corte = len(_pal)
            for _i, _w in enumerate(_pal[1:], 1):
                if _w[:1].islower() and _w.lower() not in (
                        "de", "del", "la", "el", "los", "las", "y", "e", "para", "sobre",
                        "en", "al", "a", "por", "con", "su", "sus", "o", "u"):
                    _corte = _i
                    break
            _citado = " ".join(_pal[:_corte]).lower()
            # UN NOMBRE ANAFÓRICO NO NOMBRA NINGUNA LEY. «la ley federal
            # RELATIVA», «el código CITADO», «la ley INVOCADA» apuntan a una
            # ley nombrada antes; tomarlos por nombre propio produce un
            # precepto fantasma que el secretario no puede comprobar porque no
            # existe. Salió en la revisión fiscal 2/2026 —«art. 50 — ley
            # federal relativa»— leyendo el rubro de una tesis, que va en
            # mayúsculas y por eso el corte por minúscula no lo vio.
            if _RX_ANAFORICA.search(_citado):
                _citado = ""
        try:
            import fase6_rag as _f6r_id
            _misma = _f6r_id.misma_ley
        except Exception:
            _misma = None
        for cuerpo in leyes_material:
            vl = _voces(cuerpo)
            if not vl:
                continue
            if _citado and _misma is not None:
                if _misma(_citado, cuerpo) and len(vc & vl) > puntos:
                    mejor, puntos = cuerpo, len(vc & vl)
                continue
            n_comun = len(vc & vl)
            if n_comun >= max(2, len(vl) // 2) and n_comun > puntos:
                mejor, puntos = cuerpo, n_comun
        if mejor and (mejor, art) not in en_material:
            fuera.add(f"art. {art} — {mejor}")
            pares.add((mejor, str(art)))
        elif not mejor:
            # LA LEY NO ESTÁ EN EL MATERIAL. Antes esto no se veía: la
            # detección sólo miraba las leyes ya traídas, así que «artículo
            # 137 del Código Fiscal de la Federación» en un material sin CFF
            # no era «fuera» —era nada—. Se lee el nombre de la cola tal como
            # el estudio lo escribió y se manda a traer.
            _mn = re.match(r"(?:,\s*)?(?:(?:fracci[óo]n|p[áa]rrafo|inciso)[^,]{0,30},?\s*)?"
                           r"(?:de|del)\s+(?:la|el|los|las)?\s*"
                           r"((?:constituci[óo]n|c[óo]digo|ley|reglamento|convenci[óo]n|pacto|tratado)"
                           r"[^,;:()]{4,90})", " ".join(cola.split()), re.I)
            if _mn:
                # EL NOMBRE SE LEE CON SUS MAYÚSCULAS: «Código Fiscal de la
                # Federación regulan…» acaba en «Federación», porque «regulan»
                # va en minúscula y no es conector. La lista de verbos de abajo
                # queda como red para cuando el nombre viene en minúsculas.
                # UNA PREGUNTA NO ES UNA CITA. Desde que el material se
                # completa ANTES de proponer, este lector recibe también las
                # preguntas de los problemas —«¿El artículo 150 … del
                # Reglamento Interior del IMSS faculta a…?»— y el nombre salía
                # con la cola de la pregunta pegada: «…seguro social? la». Con
                # ese nombre se buscaba en el acervo y se preguntaba a la web.
                _nombre = " ".join(_mn.group(1).split())
                _nombre = re.split(r"[?¿!¡;:»\"]", _nombre)[0].strip(" .,")
                _pal = _nombre.split()
                _corte = len(_pal)
                for _i, _w in enumerate(_pal[1:], 1):
                    if _w[:1].islower() and _w.lower() not in (
                            "de", "del", "la", "el", "los", "las", "y", "e", "para", "sobre",
                            "en", "al", "a", "por", "con", "su", "sus", "o", "u"):
                        _corte = _i
                        break
                _nombre = " ".join(_pal[:_corte]).lower()
                # EL NOMBRE ACABA DONDE EMPIEZA EL VERBO: «…Contencioso
                # Administrativo fija los requisitos» no es el nombre de la ley.
                _nombre = re.sub(r"\s+(?:establece|dispone|prev[eé]|se[ñn]ala|regula|permite|fija|faculta|"
                                 r"ordena|impone|exige|proh[íi]be|autoriza|contempla|define|determina|"
                                 r"consagra|reconoce|garantiza|sanciona|obliga|otorga|confiere|prescribe|"
                                 r"contiene|precisa|indica|refiere|dice|es|son|fue|era|y|que|en|al|con|sin|"
                                 r"cuyo|cuya|donde|as[íi]|tambi[ée]n)\b.*$", "", _nombre)
                # NI AQUÍ UN NOMBRE ANAFÓRICO. Este camino lee el nombre tal
                # como está escrito, y en el rubro de una tesis —que va en
                # mayúsculas— «LA LEY FEDERAL RELATIVA» pasaba entero: se
                # mandaba a buscar al acervo y a la web una ley que no existe,
                # y el aviso le pedía al secretario comprobar un precepto
                # fantasma.
                if len(_nombre) >= 8 and not _RX_ANAFORICA.search(_nombre):
                    fuera.add(f"art. {art} — {_nombre}")
                    pares.add((_nombre, str(art)))
    return fuera, pares


# ═══ 1-duodecies · LA TESIS QUE NO HABLA (AR 631/2025, 28-sep-2026) ═════════
# David: «Después de citar tesis hay que hacerlas hablar. Y después aplicarlas
# al caso concreto. Es lo que se estila.» Medido a mano en 82 usos de criterio
# de diez sentencias de la Suprema Corte y 19 de cinco engroses de este
# tribunal (redactor-sentencias/scjn_estilo/anotacion/anotacion.csv): NINGUNO
# queda sin su regla dicha ni su aplicación. En la Solución del AR 631/2025
# (verificación del 28-sep) quedaban seis de nueve: una pila de cuatro rubros
# sin una palabra entre ellos, dos más de la recurrente al final sin nada
# detrás, y una tesis de la recurrente anunciada con «Sirve de apoyo» y
# declarada inaplicable en el renglón siguiente.
#
# Es un control de TEXTO, determinista y sin modelo, y por eso es tosco: no
# sabe si una frase «dice la regla» de una tesis; mide si la prosa que la
# rodea comparte su vocabulario (el del rubro) o remite a ella. Por eso se
# calibró sobre los mismos textos anotados —la Corte y los engroses deben dar
# CERO; el 631, entre seis y ocho— y los umbrales de abajo son los que salieron
# de esa calibración, no una intuición. La reproduce test_tesis_que_hablan.py
# (§1) cuando el recurso está en disco.
#
# Lee las dos formas del texto: la del modelo —el anuncio y el rubro en el
# mismo párrafo— y la compuesta —el anuncio acaba en dos puntos y el rubro va
# en el párrafo siguiente, con el texto al pie—, que es la forma del 631 y de
# los engroses con que se calibró.

# Las pistas de que un texto en mayúsculas es el rubro de un criterio y no el
# nombre de una parte ni la transcripción de una constancia.
_RX_PISTA_CRITERIO = re.compile(
    r"\b(?:rubros?|jurisprudencias?|tesis|registro|criterios?|precedentes?|"
    r"ejecutoria|intitulan|t[íi]tulo|siguientes?\s*:|transcribe|en\s+cita)\b", re.I)
# «registro 171925» CON UN SOLO ESPACIO (AR 631/2025, 28-sep-2026, al generar
# en pantalla): la forma anterior exigía dos separadores tras «registro» —el
# espacio y otro, o «digital»— y no veía «Sirve de apoyo el criterio de
# registro 171925:», que es justo la forma que el prompt enseña al modelo. El
# control corría ciego sobre toda cita por registro sin rubro en el texto. Se
# añade SÓLO esa forma: «Registro: 2009468» (la ficha de una nota al pie de la
# Corte) sigue sin contar, como en la calibración.
_RX_REG_EN_CITA = re.compile(
    r"registro(?:\s+(?:digital\s*)?(?:n[úu]mero\s*)?[:\s]\s*|\s+)(\d{6,7})\b", re.I)
_RX_RUBRO_COMILLAS = re.compile(r"[“«\"]\s*([A-ZÁÉÍÓÚÑÜ][^”»\"«“\n]{20,1500}?)\s*[”»\"]")
_RX_RUBRO_SIN_COMILLAS = re.compile(
    r"(?<![\wÁÉÍÓÚÑÜáéíóúñü])((?:[A-ZÁÉÍÓÚÑÜ][A-ZÁÉÍÓÚÑÜ0-9.,;()\-–/°º]*[ \t]+){7,}"
    r"[A-ZÁÉÍÓÚÑÜ0-9][A-ZÁÉÍÓÚÑÜ0-9.,;()\-–/°º]*)")

# El verbo que ata la cita al razonamiento como APOYO. Negado —«no resulta
# aplicable», «sin que sea aplicable»— es lo contrario, y se mira aparte.
_RX_VERBO_APOYO = re.compile(
    r"\b(?:sirve[n]?\s+de\s+(?:apoyo|sustento)|(?:es|son|resulta[n]?)\s+aplicables?|"
    r"cobra[n]?\s+aplicaci[óo]n|tiene[n]?\s+aplicaci[óo]n|robustece[n]?|corrobora[n]?|"
    r"(?:encuentra|halla)\s+(?:apoyo|sustento)|tiene\s+sustento|apoya[n]?\s+(?:lo\s+anterior|"
    r"esta|esa|dicha|tal)|sustenta[n]?\s+(?:lo\s+anterior|esta|esa|dicha|tal)|"
    r"en\s+apoyo\s+de\s+(?:lo\s+anterior|esta|esa|tal)|ilustra[n]?)\b", re.I)
_RX_NIEGA_ANTES = re.compile(r"(?:\bno|\bni|\bsin\s+que|\btampoco)\s+(?:\w+\s+){0,2}$", re.I)
_RX_NO_APLICA = re.compile(
    r"\b(?:no\s+(?:le\s+)?(?:resulta[n]?|es|son|sea[n]?|cobra[n]?|tiene[n]?)\s+"
    r"(?:exactamente\s+|directamente\s+)?(?:aplicables?|aplicaci[óo]n)|inaplicables?|"
    r"no\s+(?:es|resulta)\s+(?:el\s+caso\s+de\s+)?aplicarl[ao]s?|no\s+se\s+aplica[n]?\b|"
    r"no\s+aplica[n]?\b|no\s+(?:sirve[n]?|puede[n]?\s+servir)\s+de\s+apoyo)", re.I)
# Lo que dice que el criterio de la parte se distinguió, se acotó o que a la
# parte le asiste razón: cualquiera de las tres respuestas de la técnica.
_RX_CONTESTA_CRITERIO = re.compile(
    r"\b(?:no\s+(?:resulta[n]?|es|son|sea[n]?)\s+aplicables?|inaplicables?|"
    r"se\s+distingue|supuestos?\s+(?:distint|divers)|no\s+corresponde[n]?|"
    r"a\s+diferencia\s+de|hechos\s+que\s+(?:dieron|le\s+dieron)\s+origen|"
    r"hechos\s+que\s+dieron\s+lugar|no\s+sostiene[n]?|ninguno\s+de\s+ellos|ninguna\s+de\s+ellas|"
    r"al\s+margen\s+de\s+(?:la\s+)?validez|(?:asiste|tiene)\s+(?:la\s+)?raz[óo]n|"
    r"no\s+alcanza|no\s+llega\s+a|debe\s+entenderse\s+en\s+el\s+contexto|"
    r"premisa\s+(?:f[áa]ctica|distinta|diversa)|se\s+refiere\s+(?:exclusivamente\s+)?a\s+"
    r"(?:un|una|el|la|los|las)\s+supuesto)", re.I)
# La remisión expresa al criterio en lo que sigue: «el criterio en cita»,
# «conforme a esa jurisprudencia», «trasladados esos lineamientos». «Lo
# anterior» NO cuenta: en el 631 abre la aplicación de la premisa propia, no
# la de la tesis, y el ¶93 de la verificación es justo ese caso.
_RX_REMITE_CRITERIO = re.compile(
    r"\b(?:en\s+cita|citad[oa]s?|invocad[oa]s?|transcrit[oa]s?|precitad[oa]s?|"
    r"(?:dich[oa]s?|es[ea]s?|est[ea]s?|aquel(?:la|los|las)?|tal(?:es)?)\s+"
    r"(?:criterios?|tesis|jurisprudencias?|precedentes?|ejecutorias?|lineamientos)|"
    r"(?:del?|al|el|la|los|las)\s+(?:criterios?|tesis|jurisprudencias?|precedentes?|ejecutorias?)\s+"
    r"(?:en\s+cita|citad|invocad|referid|aludid|mencionad|que\s+antecede|anterior|transcrit)|"
    r"conforme\s+a\s+(?:dich|es|est|la\s+juris|la\s+tesis|el\s+criterio)|"
    r"de\s+acuerdo\s+con\s+(?:dich|es|est|la\s+juris|la\s+tesis|el\s+criterio)|"
    r"trasladad[oa]s?|en\s+ell[oa]s?\s+se\s+(?:destac|sostuv|estableci|determin|precis|dijo)|"
    r"en\s+todos\s+ellos|deja\s+en\s+claro|doctrina\s+jurisprudencial|"
    r"(?:el|la)\s+(?:Pleno|Primera\s+Sala|Segunda\s+Sala|Suprema\s+Corte)\s+"
    r"(?:ha\s+)?(?:sostuv|consider|determin|estableci|resolvi|precis|sostien|consider))", re.I)

# LA CITA QUE SE NARRA NO ES DEL ESTUDIO: «Sirvieron de apoyo a la
# determinación del Colegiado las jurisprudencias…», «Cita las tesis de
# rubro…». Es el relato de lo que hizo otro —el inferior, la parte— y su pila
# no es una pila del estudio. Salió en dos extractos de la Corte fuera de la
# muestra anotada (ADR 1461/2014, ¶45; ADR 499/2015, resumen de agravios).
_RX_NARRA_CITA = re.compile(
    r"\b(?:cit(?:a|an|ó|aron)|invoc(?:a|an|ó|aron)|se\s+apoy(?:a|ó|aron)|"
    r"apoy(?:ó|aron)\s+su|sustent(?:ó|aron)\s+su|fund(?:ó|aron)\s+su|"
    r"transcribi(?:ó|eron)|sirvieron\s+de\s+apoyo\s+a\s+la\s+determinaci[óo]n|"
    r"en\s+apoyo\s+de\s+su\s+(?:planteamiento|argumento|postura|pretensi[óo]n))\b", re.I)

# Palabras que no distinguen un criterio de otro: están en la prosa de
# cualquier estudio y, contadas, harían «hablar» a cualquier rubro.
_VACIAS_CRITERIO = frozenset("""
para como entre sobre deben debe puede pueden cuando este esta estos estas dicho
dicha dichos dichas cual cuales mismo misma todo toda todos todas otro otra otros
otras tanto sino solo aunque donde desde hasta ante bajo tras segun respecto
conforme relativo relativa relativos relativas tambien porque pues aquel aquella
amparo juicio sentencia leyes articulo articulos constitucion constitucional
politica estados unidos mexicanos federal general caso casos parte partes
autoridad autoridades derecho derechos tribunal tribunales colegiado colegiados
circuito suprema corte justicia nacion sala pleno tesis jurisprudencia criterio
criterios registro digital rubro texto materia cuyo cuya cuyos cuyas sera seran
fue fueron sido tiene tienen haber hacer debe deberan asimismo ello ellos ellas
dicho mismo cada contra existe existen resulta resultan virtud efecto
""".split())


def _pal_criterio(t: str) -> set:
    """Las raíces (seis letras) de las palabras con contenido."""
    import unicodedata as _ud
    t = _ud.normalize("NFKD", (t or "").lower())
    t = "".join(c for c in t if not _ud.combining(c))
    return {w[:6] for w in re.findall(r"[a-z]{4,}", t) if w not in _VACIAS_CRITERIO}


def _parece_rubro(r: str, suelto: bool = False) -> bool:
    letras = [c for c in r if c.isalpha()]
    if len(letras) < 20:
        return False
    if sum(1 for c in letras if c.isupper()) / len(letras) < 0.85:
        return False
    pal = [w for w in r.split() if any(c.isalpha() for c in w)]
    if len(pal) < (8 if suelto else 5):
        return False
    # «R E S U E L V E» y los ordinales no son rubros.
    if sum(1 for w in pal if len(w.strip(".,;:()")) >= 4) < 4:
        return False
    return True


def citas_de_criterios(texto: str, tesis=None) -> list:
    """Cada criterio citado en el texto, en orden, con su rubro y su registro.

    Un rubro es un texto en mayúsculas —entre comillas, o suelto si es largo—
    con una pista de cita cerca (jurisprudencia, tesis, rubro, registro…) o
    pegado a otro rubro: así no cuentan el nombre de una parte ni una
    constancia transcrita en mayúsculas. El registro que va a menos de 250
    caracteres del rubro, sin otro rubro en medio, es el suyo; el que queda
    solo se busca en `tesis` (el material) para tener su rubro.

    Devuelve dicts {ini, fin, rubro, registro}: posiciones sobre `texto`."""
    t = texto or ""
    por_reg = {str(x.get("registro") or ""): x for x in (tesis or []) if isinstance(x, dict)}
    rubros = []
    for m in _RX_RUBRO_COMILLAS.finditer(t):
        if _parece_rubro(m.group(1)):
            rubros.append([m.start(), m.end(), m.group(1), True])
    sueltos = []
    for m in _RX_RUBRO_SIN_COMILLAS.finditer(t):
        if any(a <= m.start(1) < b or a < m.end(1) <= b for a, b, _, _ in rubros):
            continue
        sueltos.append([m.start(1), m.end(1), m.group(1), False])
    # UN RUBRO SUELTO SE PARTE EN TROZOS en cuanto trae una minúscula —«2o.»,
    # «(10a.)», «Bis»—: en la calibración, un solo rubro de la Corte salía como
    # tres citas seguidas y hacía una pila que no existe. Los trozos separados
    # por una o dos palabras cortas y sin comillas en medio son el mismo rubro.
    unidos = []
    for x in sueltos:
        if unidos:
            hueco = t[unidos[-1][1]:x[0]]
            if len(hueco) <= 20 and len(hueco.split()) <= 2 and not re.search(r"[“”«»\"\n]", hueco):
                unidos[-1][1] = x[1]
                unidos[-1][2] = t[unidos[-1][0]:x[1]]
                continue
        unidos.append(x)
    rubros += [x for x in unidos if _parece_rubro(x[2], suelto=True)]
    rubros.sort()
    buenos = []
    for a, b, r, _q in rubros:
        antes = t[max(0, a - 250):a]
        despues = t[b:b + 250]
        pegado = bool(buenos) and a - buenos[-1][1] <= 60
        if _RX_PISTA_CRITERIO.search(antes) or _RX_REG_EN_CITA.search(despues[:120]) or pegado:
            buenos.append([a, b, r])
    citas = [{"ini": a, "fin": b, "rubro": r, "registro": ""} for a, b, r in buenos]
    for m in _RX_REG_EN_CITA.finditer(t):
        reg = m.group(1)
        # El rubro más cercano sin registro, a menos de 250 caracteres y sin
        # otro rubro entre los dos.
        mejor, dist = None, 251
        for c in citas:
            if c["registro"]:
                continue
            d = (c["ini"] - m.end()) if c["ini"] >= m.end() else (m.start() - c["fin"])
            if 0 <= d < dist:
                entre = [x for x in citas if x is not c and
                         min(m.end(), c["fin"]) <= x["ini"] < max(m.start(), c["ini"])]
                if not entre:
                    mejor, dist = c, d
        if mejor is not None:
            mejor["registro"] = reg
            mejor["ini"], mejor["fin"] = min(mejor["ini"], m.start()), max(mejor["fin"], m.end())
            continue
        if any(c["ini"] <= m.start() < c["fin"] for c in citas):
            continue
        x = por_reg.get(reg) or {}
        # «solo_registro»: la forma del modelo, «…el criterio de registro N:»,
        # sin el rubro en el texto; el compositor lo pone debajo.
        citas.append({"ini": m.start(), "fin": m.end(),
                      "rubro": str(x.get("rubro") or ""), "registro": reg,
                      "solo_registro": True})
    citas.sort(key=lambda c: c["ini"])
    return citas


# Lo que en el hueco entre dos citas no es prosa propia: el anuncio y la ficha.
_RX_FORMULA_CITA = re.compile(
    r"\b(?:sirve[n]?\s+de\s+(?:apoyo|sustento)|(?:es|son|resulta[n]?)\s+aplicables?|"
    r"as[íi]\s+como|tambi[ée]n|asimismo|igualmente|y|e|la|el|las|los|de|del|al|a|en|"
    r"que|se|su|sus|con|por|jurisprudencias?|tesis|aisladas?|criterios?|rubros?|"
    r"textos?|siguientes?|registros?|digital|n[úu]mero|primera|segunda|sala|pleno|"
    r"suprema|corte|justicia|naci[óo]n|tribunal(?:es)?|colegiados?|circuito|alto|"
    r"ese|este|orientador|obligatori[oa]|como|carácter|[ée]poca|ídem|idem|ib[íi]d|"
    r"transcribe|continuaci[óo]n|intitulan?|t[íi]tulo|n[úu]meros?)\b", re.I)


def _palabras_propias(t: str) -> int:
    t = re.sub(r"\(?\b\d+[a-z]?\.?\s*/?\s*J?\.?\s*\d+/\d{2,4}\s*(?:\(\d+a\.\))?\)?", " ", t or "")
    t = _RX_FORMULA_CITA.sub(" ", t)
    return len([w for w in re.findall(r"[A-Za-zÁÉÍÓÚÑÜáéíóúñü]{3,}", t)])


# LO QUE ABRE OTRO PUNTO DEL ESTUDIO: el siguiente agravio, concepto o
# planteamiento, o un rótulo. Si el párrafo que sigue a una cita lo abre, entre
# la cita y él no se dijo nada de ella.
_RX_ABRE_OTRO_PUNTO = re.compile(
    r"^\s*(?:(?:Por\s+lo\s+que\s+(?:hace|toca|respecta)|En\s+(?:cuanto|lo\s+que\s+"
    r"(?:hace|toca|respecta)|relaci[óo]n)|Respecto|Sobre|Por\s+otra\s+parte|En\s+otro\s+orden"
    r"|Finalmente|Ahora\s+bien)[^.\n]{0,120}?\b(?:agravios?|conceptos?\s+de\s+violaci[óo]n|"
    r"planteamientos?|argumentos?|motivos?\s+de\s+(?:inconformidad|disenso))\b"
    r"|(?:(?:primer|segund|tercer|cuart|quint|sext|s[ée]ptim|octav|noven|d[ée]cim)\w*|"
    r"[úu]ltimo)\s+(?:agravio|concepto)\b)", re.I)


def _cierra_sin_hablar(t: str, c: dict) -> bool:
    """«Sin explicar»: la cita en la forma del modelo (sólo el registro; el
    compositor pone debajo el rubro y el texto) CIERRA su párrafo y el párrafo
    siguiente ya abre otro agravio, otro concepto u otro apartado. Entre la
    tesis y el punto siguiente no queda ni su regla ni su aplicación.

    AR 631/2025, al generar en pantalla (28-sep-2026): «Sirve de apoyo el
    criterio de registro 171925:» al final del párrafo del art. 93 y, debajo,
    «Por lo que hace al segundo agravio…». Por vocabulario «hablaba» —el
    párrafo de antes parafrasea el art. 93 con las palabras del rubro y el
    segundo agravio comparte «recurrente», «conceptos»…—, así que ni la
    huérfana en sombra la veía. Por forma es inequívoca: la Corte, tras
    transcribir, sigue con la regla y la aplicación (SPEC_D), no con el punto
    siguiente. Los textos de la calibración traen el rubro en el cuerpo, así
    que esta forma no aparece en ellos: los ceros se mantienen."""
    if not c.get("solo_registro"):
        return False
    fin_par = t.find("\n", c["fin"])
    resto = t[c["fin"]:fin_par if fin_par >= 0 else len(t)]
    if re.search(r"[A-Za-zÁÉÍÓÚÑÜáéíóúñü]{3,}", resto):
        return False                                    # el párrafo sigue: no cierra
    if fin_par < 0:
        return False
    sig = t[fin_par + 1:]
    sig = sig[:sig.find("\n")] if "\n" in sig else sig
    if not sig.strip():
        return False
    rotulo = (len(sig.split()) <= 10 and not re.search(r"[.:;]\s*$", sig.strip()))
    return bool(_RX_ABRE_OTRO_PUNTO.search(sig)) or rotulo


# LOS UMBRALES, DE LA CALIBRACIÓN (28-sep-2026): con ellos, 0 avisos en las
# diez sentencias anotadas de la Corte, 0 en los cinco engroses, 0 en los otros
# 109 extractos de la Corte y 18 engroses Kingston, y 7 en la Solución del 631.
UMBRAL_HABLA_ANTES = 0.40     # del rubro, dicho en los dos párrafos de antes
UMBRAL_HABLA_DESPUES = 0.34   # del rubro, retomado en lo que sigue
MIN_PROSA_ENTRE_CITAS = 8     # menos que esto entre dos citas es una pila
VENTANA_DESPUES = 1600        # caracteres de lo que sigue a la cita
# LO QUE LA CALIBRACIÓN DEJÓ EN SOMBRA: la «huérfana» suelta. Medida por el
# vocabulario, la cita que no habla del 631 y el corolario de la Corte —que
# cierra un razonamiento dicho con otras palabras que las del rubro— dan las
# mismas cifras: con los umbrales de arriba acusaba 21 citas de las diez
# sentencias de la Corte y 2 de los engroses, que no tienen ninguna. Se mide
# y se registra; al secretario sólo le llega lo que la calibración separa sin
# error: la pila sin prosa, la contradicción y la tesis de la parte reciclada.
DEFECTOS_AL_SECRETARIO = ("apilada", "contradictoria", "de la parte", "sin explicar")


def tesis_sin_hablar(estudio: str, tesis=None, de_la_parte=None,
                     rubros_de_la_parte=None, traza: list = None) -> list:
    """1-duodecies: los criterios citados que no hablan.

    Por cada cita (ver `citas_de_criterios`) y cada pila —citas seguidas con
    menos de `MIN_PROSA_ENTRE_CITAS` palabras propias entre ellas— se mide si
    «habla»: si los dos párrafos de antes comparten al menos el
    `UMBRAL_HABLA_ANTES` de las palabras de su rubro, o lo que sigue (hasta la
    próxima cita) el `UMBRAL_HABLA_DESPUES`, o lo que sigue remite a ella
    («el criterio en cita», «trasladados esos lineamientos»…). Defectos:
      · «apilada»: tres o más seguidas, sin remisión detrás y con la mayoría
        sin hablar (la Corte apila hasta seis, pero enuncia enseguida la
        premisa que comparten: JRC AD 13/2017, ¶72-73);
      · «contradictoria»: anunciada con un verbo de apoyo y declarada
        inaplicable en lo que sigue (la 2015679 del 631);
      · «de la parte»: la invocó la parte (`de_la_parte`: registros; o
        `rubros_de_la_parte`) y se usa con verbo de apoyo, sin ninguna de las
        tres respuestas de la técnica (aplicarla dándole la razón, acotarla,
        distinguirla) y sin que lo que sigue la retome;
      · «sin explicar»: en la forma del modelo, anunciada como apoyo al cerrar
        su párrafo, y el siguiente ya abre otro punto (`_cierra_sin_hablar`;
        el 171925 del 631 generado en pantalla);
      · «huérfana»: no habla. SÓLO EN SOMBRA (ver `DEFECTOS_AL_SECRETARIO`).
    Sin rubro no se puede medir si habla: esa cita sólo entra en las pilas y
    en la contradicción.

    Devuelve [{registro, rubro, defectos: [..]}] SÓLO de los que tienen algún
    defecto, uno por criterio (el mismo registro dos veces es un criterio).
    `traza`, si se pasa, recibe las medidas de cada cita (para calibrar)."""
    t = estudio or ""
    citas = citas_de_criterios(t, tesis)
    if not citas:
        return []
    parte = {str(x) for x in (de_la_parte or []) if str(x).strip()}
    import unicodedata as _ud

    def _nr(x):
        x = _ud.normalize("NFKD", (x or "").upper())
        x = "".join(c for c in x if not _ud.combining(c))
        return re.sub(r"[^A-Z0-9]+", " ", x).strip()[:60]
    rub_parte = {_nr(r) for r in (rubros_de_la_parte or []) if len(_nr(r)) >= 25}

    # Las pilas.
    pilas, actual = [], [0]
    for k in range(1, len(citas)):
        hueco = t[citas[k - 1]["fin"]:citas[k]["ini"]]
        if _palabras_propias(hueco) < MIN_PROSA_ENTRE_CITAS:
            actual.append(k)
        else:
            pilas.append(actual)
            actual = [k]
    pilas.append(actual)

    def _inicio_parrafo(pos, atras):
        """El comienzo del párrafo `atras` párrafos antes del que contiene `pos`."""
        p = pos
        for _ in range(atras + 1):
            j = t.rfind("\n", 0, max(0, p - 1))
            if j < 0:
                return 0
            p = j
        return p + 1

    malos = {}

    def _marca(c, defecto):
        clave = c["registro"] or _nr(c["rubro"]) or f"@{c['ini']}"
        d = malos.setdefault(clave, {"registro": c["registro"], "rubro": c["rubro"],
                                     "defectos": []})
        if defecto not in d["defectos"]:
            d["defectos"].append(defecto)

    for pila in pilas:
        pri, ult = citas[pila[0]], citas[pila[-1]]
        tope_atras = citas[pila[0] - 1]["fin"] if pila[0] > 0 else 0
        antes = t[max(tope_atras, _inicio_parrafo(pri["ini"], 2)):pri["ini"]]
        sig = citas[pila[-1] + 1]["ini"] if pila[-1] + 1 < len(citas) else len(t)
        despues = t[ult["fin"]:min(sig, ult["fin"] + VENTANA_DESPUES)]
        w_antes, w_desp = _pal_criterio(antes), _pal_criterio(despues)
        remite = bool(_RX_REMITE_CRITERIO.search(despues[:500]))
        mudas = 0
        verbo_pila = False
        narrada = False
        for k in pila:
            c = citas[k]
            # EL ANUNCIO ES EL DE SU PÁRRAFO —o el del anterior, si la cita
            # abre párrafo, que es la forma compuesta: «…de rubro siguiente:»
            # y debajo el rubro—. Con una ventana fija de caracteres se colaba
            # la frase de la cita anterior: en el 631, el «no resulta
            # aplicable» de la 2015679 hacía pasar por distinguida a la 177529.
            tope = citas[k - 1]["fin"] if k > 0 else 0
            ini_par = t.rfind("\n", 0, c["ini"]) + 1
            if len(t[ini_par:c["ini"]].split()) < 3 and ini_par > 0:
                ini_par = t.rfind("\n", 0, ini_par - 1) + 1
            anuncio = t[max(tope, ini_par, c["ini"] - 400):c["ini"]]
            # El verbo del anuncio; en una pila, «y el de rubro…» hereda el
            # del primero.
            m_v = None
            for m_ in _RX_VERBO_APOYO.finditer(anuncio):
                if not _RX_NIEGA_ANTES.search(anuncio[:m_.start()][-40:]):
                    m_v = m_
            apoyo = bool(m_v) or (k != pila[0] and verbo_pila)
            if k == pila[0]:
                verbo_pila = bool(m_v)
                narrada = bool(_RX_NARRA_CITA.search(anuncio))
            sig_k = citas[k + 1]["ini"] if k + 1 < len(citas) else len(t)
            cola = t[c["fin"]:min(sig_k, c["fin"] + 350)]
            defectos = []
            w_r = _pal_criterio(c["rubro"])
            ov_a = ov_d = None
            habla_despues = remite
            if len(w_r) >= 3:
                ov_a = len(w_r & w_antes) / len(w_r)
                ov_d = len(w_r & w_desp) / len(w_r)
                habla_despues = remite or ov_d >= UMBRAL_HABLA_DESPUES
                if ov_a < UMBRAL_HABLA_ANTES and not habla_despues:
                    defectos.append("huérfana")
                    mudas += 1
            if apoyo and _RX_NO_APLICA.search(cola):
                defectos.append("contradictoria")
            # «sin explicar»: anunciada como apoyo al cerrar el párrafo, y lo
            # que sigue ya es otro punto (ver `_cierra_sin_hablar`).
            if apoyo and k == pila[-1] and _cierra_sin_hablar(t, c):
                defectos.append("sin explicar")
            es_parte = bool((c["registro"] and c["registro"] in parte) or
                            (c["rubro"] and _nr(c["rubro"]) in rub_parte))
            if es_parte and apoyo and "contradictoria" not in defectos \
                    and not habla_despues \
                    and not _RX_CONTESTA_CRITERIO.search(anuncio + " " + cola):
                defectos.append("de la parte")
            if isinstance(traza, list):
                traza.append({"registro": c["registro"], "rubro": c["rubro"][:70],
                              "pila": len(pila), "antes": ov_a, "despues": ov_d,
                              "remite": remite, "apoyo": bool(apoyo), "parte": es_parte,
                              "defectos": list(defectos), "ini": c["ini"]})
            for x in defectos:
                _marca(c, x)
        if len(pila) >= 3 and not remite and not narrada and mudas * 2 > len(pila):
            for k in pila:
                _marca(citas[k], "apilada")
                if isinstance(traza, list):
                    for x in traza:
                        if x["ini"] == citas[k]["ini"] and "apilada" not in x["defectos"]:
                            x["defectos"].append("apilada")
    return list(malos.values())


def tesis_que_no_hablan(estudio: str, tesis=None, de_la_parte=None,
                        rubros_de_la_parte=None) -> list:
    """Lo que `tesis_sin_hablar` dice al secretario: sólo los defectos que la
    calibración separa sin error (`DEFECTOS_AL_SECRETARIO`)."""
    fuera = []
    for d in tesis_sin_hablar(estudio, tesis, de_la_parte, rubros_de_la_parte):
        dd = [x for x in d["defectos"] if x in DEFECTOS_AL_SECRETARIO]
        if dd:
            fuera.append({**d, "defectos": dd})
    return fuera


def _texto_plano(x) -> str:
    """Un resumen puede llegar como texto o como lista de párrafos."""
    if isinstance(x, (list, tuple)):
        return "\n".join(str(p) for p in x)
    return str(x or "")


def registros_de_otros(resumen_acto, es_revision: bool) -> set:
    """Los registros que, en una revisión, relata la sentencia recurrida: los
    citó la quejosa en su demanda o los invocó el juzgado. En el estudio son
    criterios que invocó otro (revisión del 28-sep-2026, AR 631/2025)."""
    if not es_revision:
        return set()
    return set(registros_de_la_parte("", None, _texto_plano(resumen_acto))[0])


def registros_de_la_parte(resumen_conceptos: str = "", inventario=None,
                          de_otros: str = "") -> tuple:
    """(registros, rubros) que invocó la parte, leídos del resumen de sus
    conceptos o agravios y del inventario. Sin ellos, el control no puede
    decir que un criterio es de la parte y no lo dice.

    `de_otros` (revisión del 28-sep-2026, AR 631/2025): en la revisión, lo que
    relata la sentencia recurrida —las tesis que citó la quejosa en su demanda y
    las que invocó el propio juzgado—. También son criterios que invocó otro: en
    el 631, el 188480 lo había citado la quejosa y el prompt lo ofrecía como
    apoyo de la vía que le quitaba el amparo, sin que 1-duodecies lo viera."""
    textos = [resumen_conceptos or "", de_otros or ""]
    for s in (inventario or []):
        if isinstance(s, dict):
            textos += [str(s.get("texto") or ""), str(s.get("cita") or ""),
                       " ".join(str(a) for a in (s.get("anclas") or []))]
    todo = "\n".join(textos)
    regs = set(_RX_REG_EN_CITA.findall(todo))
    regs |= set(re.findall(r"\bregistros?\s+(?:digitales?\s+)?(?:n[úu]meros?\s+)?(\d{6,7})\b", todo, re.I))
    rubros = [m.group(1) for m in _RX_RUBRO_COMILLAS.finditer(todo) if _parece_rubro(m.group(1))]
    return regs, rubros


def revisar(estudio: str, criterios: list[Criterio], material: Material,
            resumen_acto: str = "", marco: str = "", rama: str = "",
            resumen_conceptos: str = "") -> list[str]:
    """Lo comprobable sin modelo. Ninguna de estas es opinión.

    `rama`: la rama técnica del asunto (SPEC_B), para que el cierre operativo
    no acuse un «se concede» en un revoca-y-niega (AR 631/2025).
    `resumen_conceptos`: lo que se combate, para saber qué criterios invocó la
    parte (1-duodecies, SPEC_D). Sin él se leen del inventario, si lo hay."""
    avisos: list[str] = []
    # CUATRO COMPROBACIONES CAMBIAN CON LA v2 (26-sep-2026): las que
    # acusarían a la salida buena de la limpieza —la de pocas citas, la de
    # «se quedó corto», la del cierre que oscila y la de exceso—. Cada una
    # dice abajo con qué se calibró. La v1 se revisa como siempre.
    _v2r = _v2(material)

    # Lo que añadió la medición sobre 1,946 sentencias del acervo.
    for comprobacion in (
            _tipo_mal_atribuido(estudio, material),
            _convencional_completo(estudio),
            _cierre_operativo(estudio, criterios, rama)):
        if comprobacion:
            avisos.append(comprobacion)

    # 1. Registros inventados — el fallo que descalifica.
    validos = {str(t.get("registro", "")) for t in material.tesis}
    # Sólo las cifras anunciadas como registro: lo demás son recibos y expedientes.
    citados = set(_RX_REGISTRO_CITA.findall(estudio))
    inventados = {r for r in citados if r not in validos}
    if inventados:
        avisos.append(f"REGISTROS QUE NO ESTÁN EN EL MATERIAL: {sorted(inventados)}. "
                      "No se citan hasta comprobarlos en el Semanario.")

    # 1-bis. Un estudio SIN NINGUNA cita, teniendo obligatorias pertinentes en el
    #        material, es una opinión con formato de sentencia. Salió midiendo el
    #        ADC 125-2026: el acervo ofreció 33 tesis —incluidas dos sobre el
    #        derecho de habitación de menores, que era el tema exacto— y el
    #        estudio no invocó ni una. La causa fue el propio prompt, que tras
    #        los arreglos avisaba tres veces contra citar mal y ninguna a favor
    #        de citar bien.
    # CON LA FUERZA UNIFICADA, «obligatoria» ya no es «es jurisprudencia»: el
    # aviso sigue contando la jurisprudencia citable (revisión del 29-sep), si
    # no se quedaría sin objeto en los asuntos sin jurisprudencia de la Corte.
    obligatorias = [t for t in material.tesis if t.get("obligatoria") or t.get("vincula_origen")]
    # EN LA v2, «PREMISA SIN NINGÚN APOYO», Y EN SOMBRA. La v2 manda un apoyo
    # por premisa, máximo dos, y ninguna tesis dos veces: un estudio bueno de
    # un solo problema vivo cita una. «Menos de dos… entre tres y seis» lo
    # acusaría siempre. Se baja a CERO apoyos —ni registro, ni rubro, ni
    # clave— y aun así acusa a engroses buenos. Medido de nuevo en la revisión
    # adversarial (26-sep-2026) sobre la Solución de los 24 Kingston: por
    # registro o clave, SEIS no citan nada (los ADC 296/2025, 481, 526, 529,
    # 625 y 641/2024); con el rubro entre comillas —que es lo que mira esta
    # condición— queda UNO, el ADC 641/2024, que no cita ninguna tesis y es
    # oro sólido. Y `_RX_RUBRO_CITADO` caza cualquier texto en mayúsculas
    # entre comillas, sea tesis o no: no está calibrado como «apoyo». Por eso
    # en la v2 sólo se registra, hasta que se calibre con lo que el material
    # traía en cada caso.
    if _v2r:
        if obligatorias and not citados and not _RX_RUBRO_CITADO.search(estudio) \
                and not _RX_CLAVE_TESIS.search(estudio):
            print(f"   🔎 SOMBRA v2 · premisa sin apoyo: el estudio no cita ninguna "
                  f"tesis teniendo {len(obligatorias)} obligatorias en el material")
    elif obligatorias and len(citados) < 2:
        cuantas = "NI UNA TESIS" if not citados else "una sola tesis"
        avisos.append(f"El estudio cita {cuantas} teniendo "
                      f"{len(obligatorias)} obligatorias en el material "
                      f"(p. ej. {obligatorias[0].get('registro','')}). Los "
                      f"engroses de este tribunal invocan entre tres y seis: "
                      f"revisa si la cuestión quedó apoyada o sólo enunciada.")

    # 1-ter. Toda cita necesita su registro. Una tesis identificada sólo por su
    #        clave —«2a./J. 58/2010»— no se puede comprobar en el Semanario, que
    #        es justo para lo que se cita.
    sin_registro = 0
    for m_ in _RX_RUBRO_CITADO.finditer(estudio):
        ventana = estudio[max(0, m_.start() - 320):m_.start()]
        if not _RX_REGISTRO.search(ventana):
            sin_registro += 1
    if sin_registro:
        avisos.append(f"{sin_registro} tesis se citan SIN REGISTRO DIGITAL. La "
                      f"clave no basta: sin el registro no se comprueban en el "
                      f"Semanario.")

    # 1-quater. El estudio no repite los resúmenes que ya están arriba.
    if resumen_acto:
        eco = _solapamiento(resumen_acto, estudio)
        if eco > 0.35:
            avisos.append(f"El estudio REPITE el resumen del acto: {100*eco:.0f}% "
                          f"de sus frases ya estaban en el apartado anterior. El "
                          f"lector se lo encuentra dos veces.")

    # 1-quinquies. Si el acervo trajo criterios de la Suprema Corte y el estudio
    #              sólo invocó Colegiados, se avisa. David: «que se cite
    #              jurisprudencia de la SCJN preferentemente».
    def _scjn(t):
        i = (t.get("instancia") or "").upper()
        return any(x in i for x in ("PRIMERA SALA", "SEGUNDA SALA", "PLENO",
                                    "SUPREMA CORTE"))
    hay_scjn = [t for t in material.tesis if _scjn(t)]
    citó_scjn = [t for t in material.tesis
                 if _scjn(t) and str(t.get("registro", "")) in citados]
    if hay_scjn and citados and not citó_scjn:
        avisos.append(f"El estudio sólo cita criterios de Tribunales Colegiados "
                      f"teniendo {len(hay_scjn)} de la Suprema Corte en el "
                      f"material (p. ej. {hay_scjn[0].get('registro','')}). "
                      f"Un criterio de la Corte pesa más.")

    # 1-sexies. LA LEY DE OTRA ENTIDAD, aplicada en prosa propia.
    ajenas = _leyes_ajenas_aplicadas(estudio, material)
    if ajenas:
        avisos.append(
            f"SE INVOCA LEGISLACIÓN DE OTRA ENTIDAD ({', '.join(ajenas)}) fuera "
            f"de una cita. El juicio de origen se rige por las leyes de "
            f"{getattr(material, 'entidad', '') or 'la entidad del asunto'}; la "
            f"analogía entre códigos de entidades distintas no procede. La "
            f"jurisprudencia ajena sí se puede invocar; la ley no.")

    # 1-septies. La excusa territorial. No existe en el corpus y delata que el
    #            redactor tendió un puente donde no hacía falta ninguno.
    excusas = {m.group(0).strip() for m in _RX_EXCUSA_ENTIDAD.finditer(estudio)}
    if excusas:
        avisos.append(
            f"Se justifica una cita por la ENTIDAD de la que procede "
            f"({'; '.join(sorted(excusas))}). El criterio ajeno se invoca "
            f"anclado al principio rector, sin excusarse: la concesiva salva "
            f"una distancia de tema, nunca de entidad federativa.")

    # 1-octies. RETIRADO el 25-sep-2026. Acusaba «el marco se recibió y no se
    #           escribió» cuando el estudio no mencionaba el artículo
    #           constitucional del material; desde que no hay apartado de marco
    #           y el precepto se cita sólo donde decide, no mencionarlo es lo
    #           correcto en la mayoría de los asuntos.

    # 1-nonies. LA TESIS REPETIDA. Tras la cita, el modelo vuelve a contar lo
    #           que la tesis dice en vez de extraer su punto y aplicarlo. Se
    #           mide por solapamiento entre el texto de la tesis del acervo y
    #           lo que el estudio escribe justo después de invocarla.
    repetidas = []
    for t_ in material.tesis:
        reg = str(t_.get("registro") or "")
        cuerpo = (t_.get("texto") or "").strip()
        # 20 palabras, no 30: el umbral de 30 dejaba fuera tesis reales —la
        # 2018735 tiene 29— y el aviso no saltaba nunca donde más importa.
        if not (reg and len(cuerpo.split()) >= 20):
            continue
        m_ = re.search(re.escape(reg), estudio)
        if not m_:
            continue
        despues = estudio[m_.end():m_.end() + 900]
        if despues and _solapamiento(cuerpo, despues) > 0.30:
            repetidas.append(reg)
    if repetidas:
        # EL AVISO MIDE LO QUE ESCRIBIÓ EL MODELO, NO LO QUE SE ENTREGA: el
        # compositor borra el eco después. Decirle al secretario que el
        # proyecto repite una tesis cuando ya no la repite lo manda a buscar
        # algo que no está, y un aviso que no se comprueba deja de leerse.
        # PERO DESDE EL 28-SEP-2026 SÓLO LO BORRA SI EL TEXTO QUEDÓ EN EL
        # CUERPO (AR 631/2025): con el texto al pie, la frase que dice qué
        # sostiene la tesis es la que la hace hablar y se queda; sólo la copia
        # casi literal se va (`documento_generado.UMBRAL_ECO_AL_PIE`). Para
        # esas, el aviso no promete un borrado que no ocurre.
        import documento_generado as _dg_eco
        _por_reg = {str(t_.get("registro") or ""): t_ for t_ in material.tesis}
        _al_pie = [r for r in repetidas if _dg_eco._texto_al_pie(_por_reg.get(r) or {})]
        _en_cuerpo = [r for r in repetidas if r not in _al_pie]
        if _en_cuerpo:
            avisos.append(
                f"{'La tesis' if len(_en_cuerpo) == 1 else 'Las tesis'} "
                f"{', '.join(_en_cuerpo)} venían repetidas tras su cita y el eco se "
                f"BORRÓ al componer. El documento sale limpio; queda dicho por si "
                f"al leerlo echas en falta el enlace con el caso, que es lo que "
                f"debía ir ahí: «Conforme a la jurisprudencia citada, es claro que…»")
        if _al_pie:
            avisos.append(
                f"{'La tesis' if len(_al_pie) == 1 else 'Las tesis'} "
                f"{', '.join(_al_pie)} se vuelven a contar tras su cita con palabras "
                f"muy pegadas a su texto, que va en la nota al pie. Se conserva lo "
                f"que no es copia literal, porque es lo único que en el cuerpo dice "
                f"qué sostiene la tesis; revisa que lo diga más corto y más abstracto "
                f"que la nota, y que después lo aplique al caso.")

    # 1-decies. UN CÓDIGO QUE NO ESTÁ EN EL ACERVO NO RIGE AQUÍ. El Código
    #           Nacional de Procedimientos Civiles y Familiares entró en vigor
    #           escalonadamente y en Querétaro todavía no: siguen rigiendo el
    #           Código Civil y el de Procedimientos Civiles del Estado. Se citó
    #           igual, y aplicar una ley no vigente invalida la sentencia.
    # LA REGLA DEL ACERVO NO BASTA PARA LA VIGENCIA. Comprobado: el Código
    # Nacional de Procedimientos Civiles y Familiares ESTÁ en el acervo de
    # Querétaro —se ingirió con el resto—, así que «no cites lo que no esté en
    # el acervo» nunca lo habría detenido. Su entrada en vigor es escalonada y
    # el acervo no sabe de fechas: la vigencia se declara, no se deduce.
    #
    # Por omisión NO se da por vigente en ninguna entidad, que es el lado
    # seguro: avisar de más cuesta una comprobación; avisar de menos, una
    # sentencia que aplica una ley que aún no rige.
    _fuentes = " ".join(str(n_.get("cuerpo_legal") or n_.get("fuente") or "")
                        for n_ in material.normas).lower()
    for _cod, _ley in (("nacional de procedimientos civiles",
                        "Código Nacional de Procedimientos Civiles y Familiares"),
                       ("nacional de procedimientos penales",
                        "Código Nacional de Procedimientos Penales")):
        if _cod in estudio.lower() and _cod.split()[1] not in CNPCF_VIGENTE:
            avisos.append(
                f"SE CITA EL {_ley.upper()} y NO está en el acervo de esta "
                f"entidad. Su entrada en vigor es escalonada: comprueba que ya "
                f"rija en el Estado, porque de lo contrario la ley aplicable es "
                f"el código local y aplicar una no vigente invalida la sentencia.")

    # 1-undecies. INOPERANCIA EN LABORAL DEL TRABAJADOR. La suplencia del
    #             artículo 79, fracción V, es ABSOLUTA: opera aun sin conceptos
    #             de violación. Declarar inoperante el argumento del obrero por
    #             deficiencia en la impugnación le aplica una técnica de
    #             estricto derecho que la ley le releva. Lo detectó el dictamen
    #             de un colega sobre el ADL 382/2024 y era el defecto de fondo
    #             más grave del proyecto.
    _es_laboral = bool(re.search(r"\blaboral\b|junta\s+(?:especial|local|federal)|"
                                 r"ley\s+federal\s+del\s+trabajo|trabajador",
                                 estudio, re.I))
    # LA VÍA MANDA SOBRE LA MATERIA. Este aviso invoca la suplencia del
    # artículo 79, fracción V, de la Ley de Amparo, que rige el AMPARO. En una
    # revisión fiscal no hay quejoso ni suplencia que aplicar, y el detector
    # —que busca «trabajador» en el estudio— se dispara con cualquier asunto
    # del IMSS: la revisión fiscal 2/2026 iba sobre la baja de un asegurado
    # del régimen obligatorio y salió rotulada como asunto LABORAL. El mismo
    # deslinde ya existe para el aviso hermano, en `_SIN_SUPLENCIA`.
    import tipos_asunto as _ta_lab
    _via_sin_suplencia = _ta_lab.normalizar(
        getattr(material, "tipo_asunto", "") or "") in _SIN_SUPLENCIA
    if _es_laboral and not _via_sin_suplencia and re.search(r"\binoperant", estudio, re.I):
        # MENCIONAR LA SUPLENCIA NO ES APLICARLA, y yo estaba dando por buena
        # la mención. El aviso se apagaba en cuanto el estudio escribía la
        # palabra; en el 382/2024 la escribió CINCO veces y no suplió ni una.
        # Lo que prueba que se aplicó es la reconstrucción: «suplida la
        # deficiencia, el concepto plantea que…». Eso sí se puede buscar.
        _reconstruye = re.search(
            r"suplid[ao]\s+la\s+deficiencia|supliendo\s+la\s+deficiencia|"
            r"en\s+su\s+mejor\s+versi[óo]n|reconstruid[ao]\s+el\s+(?:concepto|argumento)|"
            r"el\s+concepto,?\s+suplid[ao]", estudio, re.I)
        avisos.append(
            "Se declara INOPERANTE un planteamiento en un asunto LABORAL. Si "
            "quien promueve es el trabajador, la suplencia del artículo 79, "
            "fracción V, de la Ley de Amparo es absoluta y opera aun sin "
            "conceptos de violación: el argumento mal expuesto se suple y se "
            "estudia, no se desecha por técnica."
            + ("" if _reconstruye else
               " Y el estudio NO deja escrita la versión suplida que examinó: "
               "mencionar la suplencia no es haberla aplicado."))

    # 1-duodecies. LA TESIS QUE NO HABLA (AR 631/2025, 28-sep-2026). Ver
    #    `tesis_sin_hablar`: la técnica de la Corte es citar, decir la regla
    #    con palabras propias y aplicarla a una constancia; y el criterio que
    #    invocó la parte se contesta, no se recicla. Calibrado sobre diez
    #    sentencias de la Corte y cinco engroses del colegiado (cero avisos) y
    #    la Solución del 631 (siete de nueve): al secretario llega lo que la
    #    calibración separa sin error —la fila de rubros sin regla, la cita
    #    contradictoria, la tesis de la parte como apoyo y, desde el 631
    #    generado en pantalla, la que cierra su párrafo y deja paso al agravio
    #    siguiente sin decir qué exige («sin explicar»)—; la cita suelta que
    #    no habla sólo se registra, en sombra (`DEFECTOS_AL_SECRETARIO`).
    try:
        # EN LA REVISIÓN, TAMBIÉN LO QUE RELATA LA RECURRIDA (revisión del
        # 28-sep-2026): las tesis de la quejosa y del juzgado son de otro.
        _de_otros = (_texto_plano(resumen_acto) if str(getattr(material, "tipo_asunto", "") or "")
                     .strip().lower() == "amparo_revision" else "")
        _regs_p, _rubs_p = registros_de_la_parte(
            resumen_conceptos, getattr(material, "inventario", None) or [], _de_otros)
        _sin_hablar = tesis_sin_hablar(estudio, material.tesis, _regs_p, _rubs_p)
    except Exception as _e_th:
        _sin_hablar = []
        print(f"   ⚠️ 1-duodecies no corrió: {type(_e_th).__name__}")
    if _sin_hablar:
        _n_huer = sum(1 for d in _sin_hablar if "huérfana" in d["defectos"])
        if _n_huer:
            print(f"   🔎 SOMBRA 1-duodecies · {_n_huer} cita(s) sin regla ni "
                  f"aplicación medibles (sólo se registra)")

        def _quien(d):
            return d["registro"] or f"«{(d['rubro'] or '')[:60]}…»"
        _por = {k: [_quien(d) for d in _sin_hablar if k in d["defectos"]]
                for k in DEFECTOS_AL_SECRETARIO}
        _partes_av = []
        if _por["apilada"]:
            _partes_av.append(
                f"EN FILA SIN REGLA: {', '.join(_por['apilada'])} van seguidos, sin "
                f"prosa propia entre ellos, y nada de lo que sigue dice qué sostienen "
                f"ni los aplica")
        if _por["contradictoria"]:
            _partes_av.append(
                f"ANUNCIADO COMO APOYO Y DECLARADO INAPLICABLE: {', '.join(_por['contradictoria'])}")
        if _por["sin explicar"]:
            _partes_av.append(
                f"CITADO AL CERRAR, SIN DECIR QUÉ SOSTIENE NI APLICARLO: "
                f"{', '.join(_por['sin explicar'])} se anuncia(n) como apoyo al final de un "
                f"razonamiento y el párrafo siguiente ya pasa a otro punto; tras la tesis va "
                f"lo que exige, con palabras propias, y por qué eso decide —o impide "
                f"decidir— aquí, o no se cita")
        if _por["de la parte"]:
            _partes_av.append(
                f"DE LA PARTE, COMO APOYO: {', '.join(_por['de la parte'])} los invocó "
                f"la parte y el estudio los usa con un verbo de apoyo, sin "
                f"distinguirlos, acotarlos ni decir que le asiste razón")
        if _partes_av:
            avisos.append(
                "CRITERIOS QUE NO HABLAN. " + "; ".join(_partes_av) + ". Cada "
                "criterio que se cita lleva su regla dicha con palabras propias y "
                "su aplicación a una constancia del expediente; el que invocó la "
                "parte se distingue —su supuesto, su razón y el dato del caso que "
                "lo saca de él—, se acota, o se aplica diciendo que en ese punto "
                "le asiste razón.")

    # 2. El sentido dictado tiene que aparecer… SALVO la inoperancia que la
    #    suplencia prohíbe. Esta regla y la anterior se contradecían: una
    #    reprochaba escribir «inoperante» y la otra reprochaba NO escribirlo.
    #    Entre las dos dejaban al modelo sin salida buena, y eligió obedecer a
    #    la que iba rotulada innegociable.
    _mat = str(getattr(material, "materia", "") or "").strip().lower()
    # LA SUPLENCIA QUE EL SECRETARIO CONFIRMÓ TAMBIÉN CUENTA (revisión,
    # 26-sep-2026). El bloque de suplencia le prohíbe al estudio la inoperancia
    # de forma a favor de esa parte; si aquí sólo valiera la materia, una II
    # confirmada en un asunto civil de alimentos, o una IV en un agrario que
    # llega como administrativo, recibía el reproche seco de no haber escrito la
    # inoperancia que el propio encargo le prohibió.
    try:
        import suplencia as _sp_rv
        _supl_conf = _sp_rv.confirmada(getattr(material, "suplencia", None) or {})
    except Exception:
        _supl_conf = False
    for c in criterios:
        raiz = c.sentido[:7].lower()
        if not raiz or raiz in estudio.lower():
            continue
        # SE MIRA EL SENTIDO ENTERO, NO LA RAÍZ: «inoperante»[:7] es «inopera»,
        # que nunca empieza por «inoperan», así que esta salvedad no se había
        # dado jamás y el aviso salía siempre seco (revisión, 26-sep-2026).
        # Y NUNCA EN LA VÍA SIN SUPLENCIA (la revisión fiscal): ahí la
        # salvedad invocaría un artículo 79 que no rige.
        if (str(c.sentido or "").lower().startswith("inoperan")
                and not _via_sin_suplencia
                and (_mat in _SUPLENCIA_ABSOLUTA or _supl_conf)):
            _con = ("con la suplencia de la queja que se confirmó en la pantalla "
                    "de decisión" if _supl_conf else
                    f"en materia {_mat}, con la suplencia del artículo 79")
            avisos.append(
                f"El criterio pedía «{c.sentido}» y el estudio no lo escribió. "
                f"{_con[0].upper() + _con[1:]}, eso "
                f"puede ser lo CORRECTO: revisa si el estudio suplió el "
                f"planteamiento y lo resolvió en el fondo. Si es así, la "
                f"calificación cambió y hay que confirmarla.")
            continue
        avisos.append(f"El criterio pedía «{c.sentido}» y esa calificación "
                      f"no aparece en el estudio.")

    # 3. Largo.
    n = len(estudio.split())
    _obj_r = _objetivo_palabras(material, criterios)
    import formato_sentencia as _fs_r
    _moderna_r = _fs_r.normalizar(getattr(material, "formato", "")) == _fs_r.MODERNA
    # EN LA v2 DE LA ESTÁNDAR NO HAY META CONTRA LA QUE QUEDARSE CORTO: hay un
    # techo. Lo corto se mide contra la Solución real: ningún engrose del
    # banco Kingston baja de 1,303 palabras, así que el piso de 1,000 no acusa
    # a ninguno de los 24. El 45 % de 3,733 que usa la v1 acusaba a tres.
    # LA REFERENCIA DE LA v2 (p2-congruencia, 26-sep-2026): 2,000 a 2,500
    # palabras según los problemas vivos (`_objetivo_palabras`), con el 45 %
    # de la v1 y nunca por debajo del piso. Calibrado: el umbral más alto que
    # sale (1,125, con cuatro problemas vivos o más) queda por debajo del
    # engrose más corto del banco Kingston (1,303): sigue sin acusar a
    # ninguno de los 24.
    if _v2r and not _moderna_r:
        _corto = max(SOLUCION_PISO, round(0.45 * _obj_r))
        if n < _corto:
            avisos.append(f"El estudio tiene {n} palabras; ningún engrose real "
                          f"de referencia resuelve en menos de {_corto}, y la "
                          f"referencia de este formato es de unas {_obj_r} cuando "
                          f"cada argumento recibe su respuesta. Revisa si quedó "
                          f"algún planteamiento sin razonar o contestado en genérico.")
    elif n < 0.45 * _obj_r:
        avisos.append(f"El estudio tiene {n} palabras; la mediana de los "
                      f"engroses es {PALABRAS_ESTUDIO}. Se quedó corto."
                      if _obj_r == PALABRAS_ESTUDIO else
                      f"El estudio tiene {n} palabras y la versión moderna pedía "
                      f"unas {_obj_r}. Se quedó corto.")

    # 4. Higiene.
    if "**" in estudio or "##" in estudio:
        avisos.append("Se coló Markdown.")
    # 4-bis. Rubros citados que no casan con ninguna tesis del material. Un
    #        registro correcto con el rubro cambiado es más difícil de ver que
    #        un registro inventado, y engaña igual.
    import unicodedata as _ud

    def _n(x):
        x = _ud.normalize("NFKD", (x or "").upper())
        return re.sub(r"[^A-Z0-9]+", " ", x).strip()

    rubros_material = [_n(t.get("rubro", "")) for t in material.tesis]
    # Lo entrecomillado que empieza por «Artículo N» es un precepto, no un
    # rubro: se transcribe así por diseño y no debe sonar como cita inventada.
    _rx_precepto = re.compile(r"^\s*ART[ÍI]CULO\s+\d", re.I)
    # UN PRECEPTO TRANSCRITO NO ES UN RUBRO. «"Artículo 568. La sentencia que
    # decrete los alimentos…"» es la ley entrecomillada, que es exactamente lo
    # que el corpus manda hacer con el precepto local decisivo, y saltaba como
    # rubro inventado. El aviso importa demasiado para dejar que se ahogue en
    # falsos positivos.
    # LAS COMILLAS ANGULARES CONTABAN COMO NADA. Esta comprobación sólo miraba
    # «"» y «“»; el documento escribe los rubros con « », así que la alarma de
    # rubro inventado no sonaba nunca donde el proyecto de verdad la escribe.
    for m_ in re.finditer(r"[“«\"]([A-ZÁÉÍÓÚÑ][^”»\"]{25,}?)[”»\"]", estudio):
        if _rx_precepto.match(m_.group(1)):
            continue                      # es la ley transcrita, no un rubro
        # UN RUBRO VA EN MAYÚSCULAS. Lo que va entre comillas en minúsculas es
        # una transcripción —del acuerdo, de la sentencia, de un escrito— y
        # no tiene que casar con ningún rubro. ADC 93/2026 v4: se acusó como
        # «rubro que no casa» la transcripción literal del acuerdo de primero
        # de julio («Ahora bien, en atención a la jurisprudencia número…»).
        _letras = [ch for ch in m_.group(1) if ch.isalpha()]
        if _letras and sum(ch.isupper() for ch in _letras) < 0.8 * len(_letras):
            continue
        cit = _n(m_.group(1))
        if len(cit) < 30:
            continue
        if not any(r.startswith(cit[:60]) or cit.startswith(r[:60])
                   for r in rubros_material if r):
            avisos.append(f"RUBRO CITADO QUE NO CASA CON EL ACERVO: "
                          f"«{m_.group(1)[:70]}…». Compruébalo.")
            break

    # 4-bis-2. CADA APARTADO ABRE CON SU RESPUESTA. ADC 93/2026 v3: el
    #          apartado 2 pasaba de la pregunta a «Sirve de apoyo la
    #          jurisprudencia…» sin contestar; la calificación llegaba cuatro
    #          párrafos después. La regla del prompt lo pide; esto lo comprueba.
    _lineas = [x.strip() for x in estudio.split("\n") if x.strip()]
    for _i, _l in enumerate(_lineas[:-1]):
        if not re.match(r"^\d{1,2}\.\s*¿", _l):
            continue
        _sig = _lineas[_i + 1]
        if re.match(r"^(?:sirve[n]?\s+de\s+apoyo|resulta[n]?\s+aplicable|es\s+aplicable|"
                    r"al\s+respecto,?\s+(?:la|el)\s+(?:jurisprudencia|tesis|criterio)|"
                    r"[«“\"])", _sig, re.I):
            avisos.append(
                f"EL APARTADO «{_l[:60]}…» NO ABRE CON LA RESPUESTA: pasa de la "
                f"pregunta a una tesis («{_sig[:50]}…»). El párrafo que sigue a "
                f"la pregunta contesta —«No. …», «Sí, porque…»— y la tesis viene "
                f"después.")
            break

    # 4-ter. El condicional sobre autos. Determinista y barato, y cierra el
    #        modo de falla que el panel puso en segundo lugar por impacto.
    condicionales = _RX_CONDICIONAL.findall(estudio)
    if condicionales:
        avisos.append(f"{len(condicionales)} frases SUPONEN hechos en vez de "
                      f"afirmarlos contra autos: {sorted(set(condicionales))[:5]}. "
                      f"Ciérralas o suprímelas.")

    # 4-quater. Una sola calificación al cierre.
    cierre = " ".join(estudio.split()[-160:]).lower()
    # «FUNDADO» NO SIEMPRE ES UNA CALIFICACIÓN. «Por lo expuesto y fundado»,
    # «fundado y motivado» (artículo 16) y «de estimarse fundados» (lo que
    # PIDE la recurrente) aparecían en el cierre y la alarma sonaba en un
    # proyecto que calificaba una sola cosa —revisión 322/2025—. Se calibra:
    # si acusa a un cierre correcto, la que está mal es la comprobación.
    _rx_no_calif = re.compile(r"(?:expuesto\s+y|de\s+estimarse|estimarse|resultar)\s*$")
    def _es_calif(m_):
        antes = cierre[max(0, m_.start() - 24):m_.start()]
        despues = cierre[m_.end():m_.end() + 14]
        if _rx_no_calif.search(antes):
            return False
        if re.match(r"\s+y\s+motivad", despues):
            return False
        return True
    # Y LO QUE SE VIGILA ES SI PROSPERA O NO. «Inoperantes» e «ineficacia de
    # los agravios» son la misma suerte para el resolutivo; lo que lo obliga
    # a rehacer es que el cierre diga a la vez que prospera y que no.
    califs = {("prospera" if m.group(1).lower().startswith("fundad") else "no_prospera")
              for m in _RX_CALIF.finditer(cierre) if _es_calif(m)}
    # EN LA v2 NO HAY CIERRE QUE PUEDA OSCILAR. Sin cierre por defecto
    # (decisión 3 de David), las últimas 160 palabras son el último apartado,
    # que puede citar lo que la responsable estimó fundado o infundado sin que
    # eso sea una calificación del tribunal. Y cuando la v2 sí permite un
    # cierre breve es porque hay resultados distintos, justo el caso en que
    # esta comprobación ya no mira.
    if (not _v2r and len(califs) > 1
            and len({c.sentido[:7].lower() for c in criterios}) == 1):
        avisos.append(f"El cierre oscila entre calificaciones {sorted(califs)}; "
                      f"el criterio pedía una sola. Obliga a rehacer el resolutivo.")

    # 5. Preceptos citados que no salieron del acervo. Un artículo inventado de
    #    un código sustantivo es tan grave como un registro inventado, y hasta
    #    ahora sólo se vigilaban los registros.
    fuera, _pares = preceptos_fuera(estudio, material)
    if fuera:
        avisos.append(f"PRECEPTOS CITADOS QUE NO ESTÁN EN EL MATERIAL: "
                      f"{sorted(fuera)}. Compruébalos antes de firmar.")

    # 6. Largo por exceso. El corpus tiene una medida y pasarse al doble no es
    #    rigor: es repetir el argumento con otras palabras.
    # EN LA v2 SE MIDE CONTRA LA SOLUCIÓN, que es lo que el estudio escribe:
    # los 6,618 son el p90 del considerando con sus resúmenes. Salta a 1.25
    # veces el techo del prompt; calibrado, acusa a los mismos tres engroses
    # Kingston que el aviso de la v1 (ver `SOLUCION_EXCESO`).
    if _v2r and not _moderna_r:
        if n > SOLUCION_EXCESO:
            avisos.append(f"El estudio tiene {n} palabras y el techo de la "
                          f"Solución es {SOLUCION_P90}: nueve de cada diez "
                          f"engroses reales resuelven en menos. Revisa si hay "
                          f"repetición.")
    elif n > PALABRAS_ESTUDIO_P90:
        avisos.append(f"El estudio tiene {n} palabras; sólo el 10% de los "
                      f"engroses reales pasa de {PALABRAS_ESTUDIO_P90}. "
                      f"Revisa si hay repetición.")
    # LA MODERNA TIENE SU PROPIA MEDIDA, y pasarse de ella por la mitad es
    # haber escrito la estándar con preguntas: lo que el secretario eligió
    # precisamente para no recibir.
    if _moderna_r and n > _fs_r.MODERNA_TECHO_FACTOR * _obj_r:
        avisos.append(f"LA VERSIÓN MODERNA SALIÓ LARGA: {n} palabras donde se "
                      f"pedían unas {_obj_r}. Busca lo que no decide —recuentos, "
                      f"paráfrasis de tesis, objeciones que nadie planteó— y quítalo.")
    # Y LA ESTÁNDAR NO LLEVA PREGUNTAS. Si el modelo las escribió igual, el
    # secretario recibe la forma que no pidió; se le dice dónde.
    if not _moderna_r:
        _pregs = re.findall(r"(?m)^\s*\d{1,2}\.\s*¿[^\n]{0,90}", estudio or "")
        if _pregs:
            avisos.append(f"LA FORMA ESTÁNDAR SALIÓ CON {len(_pregs)} PREGUNTA(S) "
                          f"COMO RÓTULO —«{_pregs[0].strip()[:80]}…»—. En esta forma "
                          f"cada concepto abre con «Sobre el … concepto, en el que…»; "
                          f"cámbialas o genera la versión moderna.")

    # 7. La medida de la prosa. El modelo tiende a apilar frases cortas dentro
    #    de párrafos largos —informe—; el corpus hace lo contrario: párrafo
    #    compacto con la frase subordinada larga. Se avisa, no se corrige solo.
    ps = [x for x in estudio.split("\n") if len(x.split()) > 4]
    if ps:
        med_p = sorted(len(x.split()) for x in ps)[len(ps) // 2]
        if med_p > 75:
            avisos.append(f"Párrafos de {med_p} palabras de mediana frente a las "
                          f"49 del corpus: se lee como informe, no como engrose.")

    if not _RX_CALIF.search(" ".join(estudio.split()[:80])):
        avisos.append("No califica en las primeras líneas: se pierde el orden "
                      "de anunciar y demostrar.")
    return avisos


def parrafos(estudio: str) -> list[str]:
    """El estudio, listo para el ensamblador.

    Se le quita el encabezado «SEXTO. Estudio.» que el modelo escribe, porque el
    ensamblador ya pone el suyo desde la plantilla, y dos encabezados seguidos
    delatan el documento. La CALIFICACIÓN que va pegada a él —«Los conceptos son
    ineficaces»— SE CONSERVA: es la frase que abre el estudio.
    """
    t = re.sub(r"^\s*(?:SEXTO|S[ÉE]PTIMO|QUINTO|CUARTO|OCTAVO)\.\s*"
               r"Estudio(?:\s+de\s+(?:fondo|los?\s+\w+))?\.?\s*",
               "", estudio.strip(), flags=re.I)
    fuera = []
    for linea in t.split("\n"):
        linea = linea.strip()
        if not linea:
            continue
        # El ensamblador ya pone los rótulos desde la plantilla; cuando el
        # modelo escribe los suyos —«Agravios:», «Solución:»— salen dos veces
        # seguidos y el documento se lee como un borrador sin repasar.
        if re.fullmatch(r"(?:Agravios|Conceptos de violaci[óo]n|Soluci[óo]n|"
                        r"Consideraciones relevantes[^:]*|Problemas? jur[íi]dicos?"
                        r"[^:]*)\s*[:.]?", linea, re.I):
            continue
        fuera.append(linea)
    return fuera


# Una marca de la v3/v4 (`marcas.py`) al principio del renglón de un rótulo.
_RX_MARCA_DELANTE = r"⟦[^⟦⟧\n]{1,1200}⟧[ \t]*"


def separar_advertencias(estudio: str) -> tuple[str, str]:
    """Aparta el apartado de ADVERTENCIAS: no es parte de la sentencia.

    Se le enseña al secretario en pantalla, pero NO entra en el .docx: una
    sentencia no lleva notas del redactor a su lector.
    """
    # UNA MARCA DELANTE DEL RÓTULO NO LO ESCONDE (revisión adversarial,
    # 26-sep-2026). Este corte corre sobre el texto CON las marcas de la v3/v4
    # —`_terminar` las separa después— y «⟦C3.e⟧ ADVERTENCIAS:» no casaba: las
    # advertencias se quedaban dentro de la sentencia. Sin «⟦» —v1, v2— el
    # patrón es el de siempre.
    m = re.search(r"\n\s*(?:" + _RX_MARCA_DELANTE + r")?ADVERTENCIAS?\s*[:\n]", estudio, re.I)
    if not m:
        return estudio.strip(), ""
    cuerpo, adv = estudio[:m.start()].strip(), estudio[m.end():].strip()
    # LOS EFECTOS NO SON UNA ADVERTENCIA. ADC 93/2026 v12: el modelo escribió
    # ADVERTENCIAS y DEBAJO «EFECTOS DE LA CONCESIÓN» con sus cinco órdenes; el
    # corte se los llevó a la nota del redactor y el proyecto salió SIN
    # considerando de efectos, con el resolutivo concediendo «para los efectos»
    # que no estaban. El prompt pedía los dos «al final» sin fijar el orden.
    me = _RX_EFECTOS_TRAS_ADV.search("\n" + adv)
    if me:
        _a = "\n" + adv
        cuerpo = cuerpo + "\n\n" + _a[me.start():].strip()
        adv = _a[:me.start()].strip()
    return cuerpo, adv


_RX_EFECTOS_TRAS_ADV = re.compile(
    r"\n\s*(?:" + _RX_MARCA_DELANTE + r")?(?:\*\*|__)?\s*EFECTOS(?:\s+DE\s+LA\s+(?:CONCESI[ÓO]N|PROTECCI[ÓO]N"
    r"\s+CONSTITUCIONAL))?\s*(?:\*\*|__)?\s*[.:]?\s*\n", re.I)


# ═══ LO QUE SE ANOTA DE CADA ESTUDIO (26-sep-2026) ════════════════════════
# La propuesta del estudio empieza por medir, y la ficha del proyecto no
# guardaba con qué se hizo: ni la variante del prompt, ni si el modelo acabó o
# se cortó en el tope de tokens, ni cuánto gastó. Sólo números: ni el prompt
# ni la respuesta pasan por aquí (higiene de registros).
def _meta_vacia(material) -> dict:
    return {"variante": normalizar_variante(getattr(material, "variante", "v1"), "v1"),
            "finish_reason": "", "uso": {}}


def _uso_de(u) -> dict:
    """El uso de tokens de la respuesta, venga como objeto o como dict."""
    if not u:
        return {}

    def _g(x, k):
        return x.get(k) if isinstance(x, dict) else getattr(x, k, None)
    fuera = {}
    for k_sal, k_ent in (("entrada", "prompt_tokens"), ("salida", "completion_tokens"),
                         ("total", "total_tokens")):
        v = _g(u, k_ent)
        if isinstance(v, (int, float)):
            fuera[k_sal] = int(v)
    _det_s = _g(u, "completion_tokens_details")
    _r = _g(_det_s, "reasoning_tokens") if _det_s else None
    if isinstance(_r, (int, float)):
        fuera["razonamiento"] = int(_r)
    _det_e = _g(u, "prompt_tokens_details")
    _c = _g(_det_e, "cached_tokens") if _det_e else None
    if isinstance(_c, (int, float)):
        fuera["en_cache"] = int(_c)
    return fuera


async def redactar_en_vivo(cliente, resumen_acto: str, resumen_conceptos: str,
                           criterios: list[Criterio], material: Material,
                           es_recurso: bool = False, partes=None, marco=None,
                           contexto: str = "", propuesta_global=None,
                           rama: str = "", violacion_procesal: bool = False,
                           conceptos_violacion: str = "",
                          escrito_literal: str = "",
                           guion: str = ""):
    """El estudio, trozo a trozo, según lo escribe el modelo.

    David: «que el usuario vea el texto escribiéndose sería de ayuda». No
    acorta el reloj —el estudio son los mismos setenta segundos— pero cambia
    por completo la espera: setenta segundos de pantalla quieta se sienten como
    una avería, y viéndose escribir se sienten como trabajo.

    Va rindiendo cada trozo y, al final, el texto entero. Quien lo consume
    distingue por el tipo: «texto» mientras escribe, «fin» cuando termina.
    """
    kw = dict(model=MODELO_ESTUDIO, max_completion_tokens=16000, stream=True,
              messages=[{"role": "user", "content": prompt_estudio(
                  resumen_acto, resumen_conceptos, criterios, material,
                  es_recurso, partes, marco, contexto,
                  propuesta_global=propuesta_global, rama=rama,
                  violacion_procesal=violacion_procesal,
                  conceptos_violacion=conceptos_violacion,
        escrito_literal=escrito_literal,
                  # EL GUION DEL PLAN (v4), en los DOS redactores.
                  guion=guion)}])
    if ESFUERZO_ESTUDIO:
        kw["reasoning_effort"] = ESFUERZO_ESTUDIO
    # EL USO SE PIDE AL FLUJO (26-sep-2026): sin `include_usage` el flujo no
    # dice cuántos tokens gastó, y la ficha del proyecto no puede decir si un
    # estudio se cortó por el tope (`finish_reason = length`) ni cuánto costó.
    # Si el proveedor no admitiera la opción se llama sin ella: medir es una
    # mejora, no un requisito.
    kw["stream_options"] = {"include_usage": True}
    entero = []
    meta = _meta_vacia(material)
    try:
        flujo = await cliente.chat.completions.create(**kw)
    except Exception as _exs:
        if "stream_options" not in str(_exs):
            raise
        kw.pop("stream_options", None)
        flujo = await cliente.chat.completions.create(**kw)
    async for trozo in flujo:
        if getattr(trozo, "usage", None):
            meta["uso"] = _uso_de(trozo.usage)
        if not trozo.choices:
            continue
        if getattr(trozo.choices[0], "finish_reason", None):
            meta["finish_reason"] = str(trozo.choices[0].finish_reason)
        pieza = trozo.choices[0].delta.content or ""
        if pieza:
            entero.append(pieza)
            yield {"tipo": "texto", "dato": pieza}
    crudo = "".join(entero).strip()
    estudio, advertencias = separar_advertencias(crudo)
    # LAS MARCAS NO SON TEXTO DEL ESTUDIO (v3/v4): los controles leen el texto
    # sin ellas. El estudio sale CON ellas: `_terminar` las separa y guarda el
    # mapa. Sin marcas —v1, v2— `sin_marcas` devuelve el texto idéntico.
    import marcas as _mc_r
    yield {"tipo": "fin", "estudio": estudio, "advertencias": advertencias,
           "avisos": revisar(_mc_r.sin_marcas(estudio), criterios, material, resumen_acto,
                             marco if isinstance(marco, str) else "", rama=rama,
                             resumen_conceptos=resumen_conceptos),
           "meta": meta}


async def redactar(cliente, resumen_acto: str, resumen_conceptos: str,
                   criterios: list[Criterio], material: Material,
                   es_recurso: bool = False, partes=None, marco=None,
                   contexto: str = "", propuesta_global=None,
                   rama: str = "", violacion_procesal: bool = False,
                   conceptos_violacion: str = "",
                   escrito_literal: str = "",
                   meta: dict = None,
                   guion: str = "") -> tuple[str, str, list[str]]:
    """Devuelve (estudio, advertencias, avisos).

    `meta`, si se pasa, sale con la variante, el `finish_reason` y el uso de
    tokens: lo mismo que el gemelo en vivo rinde en su evento «fin»."""
    kw = dict(model=MODELO_ESTUDIO, max_completion_tokens=16000,
              messages=[{"role": "user", "content": prompt_estudio(
                  resumen_acto, resumen_conceptos, criterios, material,
                  es_recurso, partes, marco, contexto,
                  propuesta_global=propuesta_global, rama=rama,
                  violacion_procesal=violacion_procesal,
                  conceptos_violacion=conceptos_violacion,
        escrito_literal=escrito_literal,
                  # EL GUION DEL PLAN (v4), en los DOS redactores.
                  guion=guion)}])
    if ESFUERZO_ESTUDIO:
        kw["reasoning_effort"] = ESFUERZO_ESTUDIO
    import llamada_modelo as _lm
    r = await _lm.crear(cliente, **kw)
    if isinstance(meta, dict):
        meta.update(_meta_vacia(material))
        try:
            meta["finish_reason"] = str(r.choices[0].finish_reason or "")
            meta["uso"] = _uso_de(getattr(r, "usage", None))
        except Exception:
            pass
    crudo = (r.choices[0].message.content or "").strip()
    estudio, advertencias = separar_advertencias(crudo)
    # Lo mismo que el gemelo en vivo: los controles, sin las marcas.
    import marcas as _mc_r
    _limpio = _mc_r.sin_marcas(estudio)
    avisos = revisar(_limpio, criterios, material, resumen_acto,
                     marco if isinstance(marco, str) else "", rama=rama,
                     resumen_conceptos=resumen_conceptos)
    if partes is not None:
        import fase_partes
        avisos.extend(fase_partes.revisar_partes(_limpio, partes))
    return estudio, advertencias, avisos
