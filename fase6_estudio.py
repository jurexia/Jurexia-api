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
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

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

# CUÁNTAS TESIS ENTRAN AL PROMPT. El RAG devuelve todas las que encuentra —44 en
# el QA 143-2026— y meterlas enteras da un prompt de 108,000 caracteres del que
# el 93% es material. Medido: con ese prompt el estudio citó UNA sola tesis, y
# la instrucción de fundar quedaba al 2% del texto, enterrada bajo 30,000 tokens
# de jurisprudencia. Diez bien elegidas se leen; cuarenta y cuatro se hojean.
MAX_TESIS_PROMPT = 10
MAX_NORMAS_PROMPT = 12

PALABRAS_ESTUDIO = 3733
PALABRAS_ESTUDIO_P90 = 6618

CONECTORES = ("Lo anterior", "En ese sentido", "Por tanto", "En consecuencia",
              "No obstante", "En efecto", "Ahora bien")

# El sentido se dicta en singular («ineficaz») pero se escribe concordando con
# «los conceptos»: «son ineficaces». Sin esto sale «son ineficaz» en la primera
# línea de la sentencia, que es donde más se nota.
_PLURAL = {"fundado": "fundados", "infundado": "infundados",
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
    sentido: str                      # fundado | infundado | inoperante | ineficaz
    razonamiento: str = ""            # el porqué, que es lo que de verdad alinea


@dataclass
class Material:
    """Lo que el RAG encontró para un problema. Sólo entra lo VERIFICADO."""
    tesis: list[dict] = field(default_factory=list)      # registro, rubro, texto
    normas: list[dict] = field(default_factory=list)     # cuerpo_legal, articulo, texto
    convencional: list[dict] = field(default_factory=list)
    # Los estudios de fondo van APARTE y son molde de FORMA, nunca fundamento.
    moldes: list[dict] = field(default_factory=list)


def _bloque_criterio(criterios: list[Criterio]) -> str:
    if not criterios:
        return ""
    lineas = ["", "═" * 71,
              "EL CRITERIO DEL SECRETARIO — DIRECTIVA INNEGOCIABLE",
              "═" * 71,
              "Tu papel NO es decidir el fallo: es CONSTRUIR la mejor demostración",
              "jurídica posible del sentido que él ya fijó. Elige los argumentos, las",
              "tesis y el orden de estudio que lo sostengan con el mayor rigor.", ""]
    for i, c in enumerate(criterios, 1):
        lineas.append(f"{i}. {c.problema}")
        lineas.append(f"   SENTIDO: {c.sentido.upper()}")
        if c.razonamiento:
            lineas.append(f"   RAZÓN DEL SECRETARIO: {c.razonamiento}")
        else:
            # Sin razón escrita, el modelo tiene que suponerla, y ahí es donde
            # el proyecto deja de parecerse a lo que el secretario pensaba.
            lineas.append("   (sin razón escrita: constrúyela con el material y "
                          "señálalo en las advertencias)")
        lineas.append("")
    lineas += [
        "SI ENCUENTRAS UN OBSTÁCULO SERIO para ese sentido —una jurisprudencia",
        "obligatoria en contra, una causal de improcedencia—, NO cambies el",
        "sentido: dilo en el apartado ADVERTENCIAS para que él lo valore.",
    ]
    return "\n".join(lineas)


def _bloque_material(m: Material) -> str:
    p = ["", "═" * 71, "MATERIAL PARA FUNDAR", "═" * 71]
    # Las obligatorias primero: ya vienen ordenadas, así que el recorte se lleva
    # las orientadoras del final, que es lo que sobra.
    tesis = (m.tesis or [])[:MAX_TESIS_PROMPT]
    normas = (m.normas or [])[:MAX_NORMAS_PROMPT]
    if tesis:
        p.append("\nTESIS Y JURISPRUDENCIA (existen: salen del acervo, no de tu memoria).")
        p.append("  La OBLIGATORIA vincula a este Tribunal y se invoca como razón que")
        p.append("  decide; la ORIENTADORA sólo ilustra y se cita como apoyo. Tratarlas")
        p.append("  igual es un error de fondo, no de estilo.")
        p.append("  PREFIERE LA SUPREMA CORTE. Entre dos criterios que sirven igual, se")
        p.append("  cita el del Pleno o de una Sala antes que el de un Tribunal")
        p.append("  Colegiado: pesa más y evita el reproche de haberse quedado corto.")
        p.append("  Vienen ordenados: los primeros son los que más aplican.")
        for t in tesis:
            fuerza = "JURISPRUDENCIA OBLIGATORIA" if t.get("obligatoria") else "tesis orientadora"
            p.append(f"\n  · [{fuerza}] Registro {t.get('registro','')} — {t.get('instancia','')}")
            p.append(f"    {t.get('rubro','')}")
            if t.get("localizacion"):
                p.append(f"    {t['localizacion']}")
            # ENTERA. Con 900 caracteres el criterio quedaba cortado y el
            # modelo razonaba desde el rubro: así le hizo decir a la tesis
            # 182597 LO CONTRARIO de lo que sostiene, y era el único punto
            # donde la quejosa tenía apoyo. El rubro es un título, no la regla.
            p.append(f"    {(t.get('texto') or '')[:TESIS_CARACTERES]}")
    if normas:
        p.append("\n\nPRECEPTOS:")
        for n in normas:
            p.append(f"\n  · {n.get('cuerpo_legal','')} — Art. {n.get('articulo','')}")
            p.append(f"    {(n.get('texto') or '')[:700]}")
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


def prompt_estudio(resumen_acto: str, resumen_conceptos: str,
                   criterios: list[Criterio], material: Material,
                   es_recurso: bool = False, partes=None, marco=None) -> str:
    q = "agravios" if es_recurso else "conceptos de violación"
    calif = _calificacion(criterios)
    # EL MARCO SE REPITE AL FINAL. Medido en el proyecto 360/2025: se le
    # entregaron 6,338 caracteres de marco —artículo 4º constitucional y
    # Convención sobre los Derechos del Niño— y el estudio salió con CERO
    # menciones a ambos. El material iba en el 78% del prompt, sin imperativo.
    # Es el mismo fallo que tuvieron las citas, y se arregla igual: repitiendo
    # la orden al final, que es lo último que el modelo lee antes de escribir.
    cierre_marco = ""
    if isinstance(marco, str) and marco.strip():
        cierre_marco = f"""
Y USA EL MARCO JURÍDICO QUE SE TE DIO. No es material de consulta: es una capa
del estudio y va ESCRITA, después de anunciar el sentido y antes de entrar al
caso concreto. Arranca por la figura jurídica discutida; PARAFRASEA el precepto
constitucional —«el artículo 4º de la Constitución reconoce…»—, TRANSCRIBE entre
comillas el precepto local decisivo, y trae la fuente convencional o a la Corte
Interamericana SÓLO si el problema las exige. Cierra con la bisagra que devuelve
al expediente —«Con ese marco jurídico, es posible dar solución a los
planteamientos de la parte quejosa.»— y entra al caso. Un marco que se recibe y
no se escribe deja el estudio resolviendo sin premisa mayor.
"""
    return f"""Eres el secretario de un Tribunal Colegiado de Circuito redactando el
estudio de fondo de una sentencia de amparo. Escribes mejor que la media del
oficio: con más orden, más precisión y menos relleno, pero en su mismo registro.

FORMA — medida sobre 40 engroses firmados, no inventada:
- ABRE con el encabezado ordinal y la CALIFICACIÓN: «SEXTO. Estudio. Los {q}
  son {calif}.» Anunciar el resultado y luego demostrarlo es el orden que mejor
  se lee, y el que sigue el 40% de los engroses reales.
- PLANTEA LA CUESTIÓN COMO PREGUNTA y respóndela acto seguido. Lo hace el 18%
  y ordena el estudio entero.
- FRASE de unas 35 palabras, SUBORDINADA; PÁRRAFO de unas 49, es decir UNA O
  DOS FRASES POR PÁRRAFO. Es la medida real del corpus y no es un capricho: la
  prosa judicial encadena la premisa y su consecuencia dentro de la misma
  oración —«toda vez que», «en tanto que», «sin que obste»— en vez de apilar
  cinco frases cortas bajo un mismo párrafo, que es como escribe un informe.
- CONECTORES, por orden de uso real: {', '.join(f'«{c}»' for c in CONECTORES)}.
  No repitas el mismo dos veces seguidas.
- LA AUTORIDAD es «la responsable» o «la autoridad responsable»; el órgano se
  nombra «este Tribunal Colegiado» y usa voz impersonal («se estima», «se
  considera»). Nunca primera persona del singular.
- EXTENSIÓN: alrededor de {PALABRAS_ESTUDIO} palabras. No cortes por brevedad:
  si crees que terminas, desarrolla los efectos de la concesión, los argumentos
  reforzadores y las objeciones previsibles con su refutación.
- Sin Markdown, sin viñetas, sin esquemas.

NO REPITAS LO QUE YA ESTÁ ESCRITO — esto es lo primero:
- Los dos resúmenes que vienen abajo —lo que resolvió la responsable y lo que se
  combate— YA OCUPAN SU PROPIO APARTADO en la sentencia, antes del tuyo. Se te
  dan para que sepas de qué va el asunto, NO para que los reproduzcas.
- TU TEXTO EMPIEZA EN LA SOLUCIÓN. Abre con la calificación y entra a demostrar.
  Puedes referirte a lo que la responsable sostuvo cuando lo estés refutando
  —«la Sala afirmó X; ese razonamiento es incorrecto porque…»—, pero no vuelvas
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
       estudio de las violaciones procesales que inciden en el sentido del fallo.»
      «se procede al análisis conjunto de los {q} identificados como TERCERO y
       QUINTO, dada su estrecha vinculación con el fondo del asunto.»
- EL CRITERIO PARA AGRUPAR NO ES EL ARTÍCULO CONSTITUCIONAL INVOCADO —casi todos
  repiten el 14, el 16 y el 17— sino EL NUDO DE LA SENTENCIA QUE SE ATACA: el
  presupuesto procesal, el elemento de la acción o la prueba concreta en disputa.
- Y SI NO REAGRUPAS, DILO IGUAL: «por razón de método y atendiendo a su
  prelación lógica, los {q} se analizarán en el orden propuesto».

ARQUITECTURA — para que se vea de un vistazo que no quedó nada sin contestar:
- UN APARTADO POR CADA {q[:-1]}, en el orden en que se plantearon, cada uno
  abierto por su ordinal en letra («En el primer {q[:-1]} la parte quejosa
  sostiene que…»). El discurso corrido impide comprobar la exhaustividad y deja
  al tribunal expuesto al reproche de omisión de estudio.
- Si lo que se combate es la REDACCIÓN de una parte del acto reclamado,
  TRANSCRÍBELA entre comillas antes de analizarla.
- SUPLENCIA DE LA QUEJA: si el asunto toca derechos de menores, materia laboral
  en favor de la parte obrera o materia penal, dilo expresamente con la fracción
  del artículo 79 de la Ley de Amparo que la ordena.
- EFECTOS: si se concede, enumera los efectos de la concesión de forma que se
  puedan ejecutar sin interpretarlos.
- CIERRA con UNA SOLA calificación y el sentido. Oscilar entre «ineficaz» e
  «infundado» en el mismo estudio obliga a rehacer el resolutivo.

FUNDAMENTO — hay que fundar, y hay que fundar bien:
- FUNDA CON LAS TESIS OBLIGATORIAS DEL MATERIAL. Un estudio de fondo sin citas
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

      Sirve de apoyo la jurisprudencia de la Primera Sala de la Suprema Corte
      de Justicia de la Nación, de registro 2022074, de rubro y texto
      siguientes:

  Y ahí se detiene el párrafo. El documento coloca solo, debajo, el rubro y el
  texto íntegro de la tesis. Escribir «la jurisprudencia de registro X, de rubro
  «Y», establece que…» deja la cita partida por la mitad y sin transcripción.
- LA INSTANCIA VA SIEMPRE: «de la Primera Sala de la Suprema Corte de Justicia
  de la Nación», «de la Segunda Sala», «del Pleno», «de un Tribunal Colegiado de
  Circuito». Sin ella no se sabe qué peso tiene el criterio.
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
    por la legislación del Estado de Querétaro y ESA es la que se aplica. La
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
        «Sustenta esa consideración, por analogía, la jurisprudencia 2a./J.
         58/2010 de la Segunda Sala de la Suprema Corte de Justicia de la
         Nación, de registro …, de rubro y texto siguientes:»
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
  La clave —«2a./J. 58/2010»— no lo sustituye: sin el registro nadie comprueba
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
- NUNCA SUPONGAS LO QUE CONSTA. Un tribunal tiene los autos delante: o el hecho
  consta y se AFIRMA, o no consta y se dice que no obra. Están PROHIBIDAS las
  fórmulas «si … fue efectivamente», «se afirma que», «según lo planteado», «de
  ser cierto», «en el supuesto de que». Si el material no te permite afirmar,
  escribe que el punto no está acreditado y sigue.
{partes.bloque() if partes is not None else ""}
{marco if isinstance(marco, str) else ""}
{_bloque_criterio(criterios)}
{_bloque_material(material)}

═══════════════════════════════════════════════════════════════════════
LO QUE RESOLVIÓ LA RESPONSABLE
═══════════════════════════════════════════════════════════════════════
{resumen_acto}

═══════════════════════════════════════════════════════════════════════
LO QUE SE COMBATE
═══════════════════════════════════════════════════════════════════════
{resumen_conceptos}

Escribe el estudio de fondo.

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
Si hay obstáculos al sentido fijado, añade al final un apartado «ADVERTENCIAS»
—fuera del cuerpo de la sentencia— con lo que el secretario debe valorar.
Nada más."""


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

_VACIAS = {"de", "del", "la", "el", "los", "las", "y", "en", "que", "propio",
           "citado", "mencionado", "invocado", "referido", "aludido", "a", "su"}

# «si la copropiedad fue efectivamente reconocida», «se afirma que», «de ser
# cierto»: un tribunal con los autos delante no supone lo que consta.
_RX_CONDICIONAL = re.compile(
    r"\b(si\s+(?:\w+\s+){0,3}(?:fue|fuera|hubiera|resultara|efectivamente)"
    r"|se\s+afirma\s+que|seg[úu]n\s+lo\s+planteado|de\s+ser\s+cierto"
    r"|en\s+el\s+supuesto\s+de\s+que|de\s+haberse\s+acreditado)\b", re.I)

_NOTORIAS = ("ley de amparo", "constitución", "constitucion",
             "ley orgánica del poder judicial", "ley organica del poder judicial")


# Las comillas de un rubro llegan de tres formas —tipográficas, latinas y
# rectas— según lo que escriba el modelo. Cubrir sólo una deja el verificador
# mudo: probado con « » y no saltaba ni una alarma.
_RX_RUBRO_CITADO = re.compile(
    r"[“«\"]\s*[A-ZÁÉÍÓÚÑ][^”»\"]{25,}[”»\"]")


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
_ENTIDADES_AJENAS = (
    "AGUASCALIENTES", "BAJA CALIFORNIA", "CAMPECHE", "COAHUILA", "COLIMA",
    "CHIAPAS", "CHIHUAHUA", "DISTRITO FEDERAL", "CIUDAD DE MEXICO", "DURANGO",
    "GUANAJUATO", "GUERRERO", "HIDALGO", "JALISCO", "MEXICO", "MICHOACAN",
    "MORELOS", "NAYARIT", "NUEVO LEON", "OAXACA", "PUEBLA", "QUINTANA ROO",
    "SAN LUIS POTOSI", "SINALOA", "SONORA", "TABASCO", "TAMAULIPAS", "TLAXCALA",
    "VERACRUZ", "YUCATAN", "ZACATECAS",
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


def _leyes_ajenas_aplicadas(estudio: str, material=None) -> list[str]:
    """Entidades cuya LEY se invoca en PROSA PROPIA. Las transcripciones no cuentan."""
    limpio = _sin_acentos_est(_prosa_propia(estudio, material))
    halladas: list[str] = []
    for ent in _ENTIDADES_AJENAS:
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


def revisar(estudio: str, criterios: list[Criterio], material: Material,
            resumen_acto: str = "", marco: str = "") -> list[str]:
    """Lo comprobable sin modelo. Ninguna de estas es opinión."""
    avisos: list[str] = []

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
    obligatorias = [t for t in material.tesis if t.get("obligatoria")]
    if obligatorias and len(citados) < 2:
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
            f"Querétaro; la analogía entre códigos de entidades distintas no "
            f"procede. La jurisprudencia ajena sí se puede invocar; la ley no.")

    # 1-septies. La excusa territorial. No existe en el corpus y delata que el
    #            redactor tendió un puente donde no hacía falta ninguno.
    excusas = {m.group(0).strip() for m in _RX_EXCUSA_ENTIDAD.finditer(estudio)}
    if excusas:
        avisos.append(
            f"Se justifica una cita por la ENTIDAD de la que procede "
            f"({'; '.join(sorted(excusas))}). El criterio ajeno se invoca "
            f"anclado al principio rector, sin excusarse: la concesiva salva "
            f"una distancia de tema, nunca de entidad federativa.")

    # 1-octies. El marco que se construyó y no se escribió. Medido en el
    #           360/2025: 6,338 caracteres entregados, cero usados.
    if marco:
        arts = set(re.findall(r"Artículo (\d{1,3}) de la Constitución", marco))
        usados = {a for a in arts
                  if re.search(r"art[íi]culo\s+" + a + r"[ºo°]?\s+"
                               r"(?:de\s+la\s+)?constituci", estudio, re.I)}
        if arts and not usados:
            avisos.append(
                f"El marco jurídico trajo el artículo {', '.join(sorted(arts))} "
                f"constitucional y el estudio NO lo menciona. El marco se "
                f"recibió y no se escribió: el estudio resuelve sin premisa mayor.")

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
        avisos.append(
            f"Tras citar {'la tesis' if len(repetidas) == 1 else 'las tesis'} "
            f"{', '.join(repetidas)} el estudio REPITE su contenido en vez de "
            f"extraer su punto. La tesis ya se transcribe: lo que sigue a la "
            f"cita es «Conforme a la jurisprudencia citada, es claro que…» y el "
            f"porqué aplica a ESTE asunto.")

    # 1-decies. UN CÓDIGO QUE NO ESTÁ EN EL ACERVO NO RIGE AQUÍ. El Código
    #           Nacional de Procedimientos Civiles y Familiares entró en vigor
    #           escalonadamente y en Querétaro todavía no: siguen rigiendo el
    #           Código Civil y el de Procedimientos Civiles del Estado. Se citó
    #           igual, y aplicar una ley no vigente invalida la sentencia.
    _fuentes = " ".join(str(n_.get("fuente", "")) for n_ in material.normas).lower()
    for _cod, _ley in (("nacional de procedimientos civiles",
                        "Código Nacional de Procedimientos Civiles y Familiares"),
                       ("nacional de procedimientos penales",
                        "Código Nacional de Procedimientos Penales")):
        if _cod in estudio.lower() and _cod not in _fuentes:
            avisos.append(
                f"SE CITA EL {_ley.upper()} y NO está en el acervo de esta "
                f"entidad. Su entrada en vigor es escalonada: comprueba que ya "
                f"rija en el Estado, porque de lo contrario la ley aplicable es "
                f"el código local y aplicar una no vigente invalida la sentencia.")

    # 2. El sentido dictado tiene que aparecer.
    for c in criterios:
        raiz = c.sentido[:7].lower()
        if raiz and raiz not in estudio.lower():
            avisos.append(f"El criterio pedía «{c.sentido}» y esa calificación "
                          f"no aparece en el estudio.")

    # 3. Largo.
    n = len(estudio.split())
    if n < 0.45 * PALABRAS_ESTUDIO:
        avisos.append(f"El estudio tiene {n} palabras; la mediana de los "
                      f"engroses es {PALABRAS_ESTUDIO}. Se quedó corto.")

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
        cit = _n(m_.group(1))
        if len(cit) < 30:
            continue
        if not any(r.startswith(cit[:60]) or cit.startswith(r[:60])
                   for r in rubros_material if r):
            avisos.append(f"RUBRO CITADO QUE NO CASA CON EL ACERVO: "
                          f"«{m_.group(1)[:70]}…». Compruébalo.")
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
    califs = {m.group(1)[:7] for m in _RX_CALIF.finditer(cierre)}
    if len(califs) > 1 and len({c.sentido[:7].lower() for c in criterios}) == 1:
        avisos.append(f"El cierre oscila entre calificaciones {sorted(califs)}; "
                      f"el criterio pedía una sola. Obliga a rehacer el resolutivo.")

    # 5. Preceptos citados que no salieron del acervo. Un artículo inventado de
    #    un código sustantivo es tan grave como un registro inventado, y hasta
    #    ahora sólo se vigilaban los registros.
    en_material = {(str(n.get("cuerpo_legal", "")).lower(), str(n.get("articulo", "")))
                   for n in material.normas}
    leyes_material = {c for c, _ in en_material}
    def _voces(x: str) -> set[str]:
        return {w for w in re.findall(r"[\wáéíóúñ]+", x.lower())
                if w not in _VACIAS and len(w) > 2}

    fuera: set[str] = set()
    for art, cola in _RX_ARTICULO.findall(estudio):
        cola_n = " ".join(cola.split()).lower()
        if any(n in cola_n for n in _NOTORIAS):
            continue
        vc = _voces(cola_n)
        # La ley se reconoce por sus voces propias, pero hay que quedarse con la
        # QUE MÁS CASA, no con la primera. «Código Civil del Estado de Querétaro»
        # y «Código Civil Federal» comparten «código» y «civil»: con el primer
        # acierto ganaba el federal y el verificador denunciaba como inventado un
        # artículo correctamente citado del código local. Un aviso falso enseña a
        # ignorar los avisos, que es peor que no tenerlos.
        mejor, puntos = None, 0
        for cuerpo in leyes_material:
            vl = _voces(cuerpo)
            if not vl:
                continue
            n_comun = len(vc & vl)
            if n_comun >= max(2, len(vl) // 2) and n_comun > puntos:
                mejor, puntos = cuerpo, n_comun
        if mejor and (mejor, art) not in en_material:
            fuera.add(f"art. {art} — {mejor}")
    if fuera:
        avisos.append(f"PRECEPTOS CITADOS QUE NO ESTÁN EN EL MATERIAL: "
                      f"{sorted(fuera)}. Compruébalos antes de firmar.")

    # 6. Largo por exceso. El corpus tiene una medida y pasarse al doble no es
    #    rigor: es repetir el argumento con otras palabras.
    if n > PALABRAS_ESTUDIO_P90:
        avisos.append(f"El estudio tiene {n} palabras; sólo el 10% de los "
                      f"engroses reales pasa de {PALABRAS_ESTUDIO_P90}. "
                      f"Revisa si hay repetición.")

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


def separar_advertencias(estudio: str) -> tuple[str, str]:
    """Aparta el apartado de ADVERTENCIAS: no es parte de la sentencia.

    Se le enseña al secretario en pantalla, pero NO entra en el .docx: una
    sentencia no lleva notas del redactor a su lector.
    """
    m = re.search(r"\n\s*ADVERTENCIAS?\s*[:\n]", estudio, re.I)
    if not m:
        return estudio.strip(), ""
    return estudio[:m.start()].strip(), estudio[m.end():].strip()


async def redactar_en_vivo(cliente, resumen_acto: str, resumen_conceptos: str,
                           criterios: list[Criterio], material: Material,
                           es_recurso: bool = False, partes=None, marco=None):
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
                  es_recurso, partes, marco)}])
    if ESFUERZO_ESTUDIO:
        kw["reasoning_effort"] = ESFUERZO_ESTUDIO
    entero = []
    flujo = await cliente.chat.completions.create(**kw)
    async for trozo in flujo:
        if not trozo.choices:
            continue
        pieza = trozo.choices[0].delta.content or ""
        if pieza:
            entero.append(pieza)
            yield {"tipo": "texto", "dato": pieza}
    crudo = "".join(entero).strip()
    estudio, advertencias = separar_advertencias(crudo)
    yield {"tipo": "fin", "estudio": estudio, "advertencias": advertencias,
           "avisos": revisar(estudio, criterios, material, resumen_acto,
                             marco if isinstance(marco, str) else "")}


async def redactar(cliente, resumen_acto: str, resumen_conceptos: str,
                   criterios: list[Criterio], material: Material,
                   es_recurso: bool = False, partes=None, marco=None
                   ) -> tuple[str, str, list[str]]:
    """Devuelve (estudio, advertencias, avisos)."""
    kw = dict(model=MODELO_ESTUDIO, max_completion_tokens=16000,
              messages=[{"role": "user", "content": prompt_estudio(
                  resumen_acto, resumen_conceptos, criterios, material,
                  es_recurso, partes, marco)}])
    if ESFUERZO_ESTUDIO:
        kw["reasoning_effort"] = ESFUERZO_ESTUDIO
    r = await cliente.chat.completions.create(**kw)
    crudo = (r.choices[0].message.content or "").strip()
    estudio, advertencias = separar_advertencias(crudo)
    avisos = revisar(estudio, criterios, material, resumen_acto,
                     marco if isinstance(marco, str) else "")
    if partes is not None:
        import fase_partes
        avisos.extend(fase_partes.revisar_partes(estudio, partes))
    return estudio, advertencias, avisos
