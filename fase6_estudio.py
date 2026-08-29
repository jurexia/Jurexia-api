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
    if m.tesis:
        p.append("\nTESIS Y JURISPRUDENCIA (existen: salen del acervo, no de tu memoria).")
        p.append("  La OBLIGATORIA vincula a este Tribunal y se invoca como razón que")
        p.append("  decide; la ORIENTADORA sólo ilustra y se cita como apoyo. Tratarlas")
        p.append("  igual es un error de fondo, no de estilo.")
        for t in m.tesis:
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
    if m.normas:
        p.append("\n\nPRECEPTOS:")
        for n in m.normas:
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
                   es_recurso: bool = False, partes=None) -> str:
    q = "agravios" if es_recurso else "conceptos de violación"
    calif = _calificacion(criterios)
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
- Al citar una tesis: en el CUERPO van sólo el rubro entre comillas y el
  registro. NADA MÁS. La localización —«[J]; 11a. Época; 1a. Sala; Gaceta
  S.J.F.; Libro 52…»— NO se escribe en el cuerpo: el documento la coloca sola
  al pie, que es donde va en una sentencia, y escribirla dos veces obliga a
  borrarla a mano. El texto de la tesis tampoco lo transcribas: se transcribe
  solo, desde el acervo, palabra por palabra.
- La INOPERANCIA se razona: hay que decir POR QUÉ el planteamiento no combate
  la razón toral, no basta con declararla.
- NUNCA SUPONGAS LO QUE CONSTA. Un tribunal tiene los autos delante: o el hecho
  consta y se AFIRMA, o no consta y se dice que no obra. Están PROHIBIDAS las
  fórmulas «si … fue efectivamente», «se afirma que», «según lo planteado», «de
  ser cierto», «en el supuesto de que». Si el material no te permite afirmar,
  escribe que el punto no está acreditado y sigue.
{partes.bloque() if partes is not None else ""}
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

Escribe el estudio de fondo. Si hay obstáculos al sentido fijado, añade al
final un apartado «ADVERTENCIAS» —fuera del cuerpo de la sentencia— con lo que
el secretario debe valorar. Nada más."""


# ═══════════════════════════════════════════════════════════════════════════
# Verificación antes de entregar
# ═══════════════════════════════════════════════════════════════════════════

_RX_REGISTRO = re.compile(r"\b(\d{6,7})\b")
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


def revisar(estudio: str, criterios: list[Criterio], material: Material) -> list[str]:
    """Lo comprobable sin modelo. Ninguna de estas es opinión."""
    avisos: list[str] = []

    # 1. Registros inventados — el fallo que descalifica.
    validos = {str(t.get("registro", "")) for t in material.tesis}
    citados = set(_RX_REGISTRO.findall(estudio))
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
    for m_ in re.finditer(r"[“\"]([A-ZÁÉÍÓÚÑ][^”\"]{25,}?)[”\"]", estudio):
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


async def redactar(cliente, resumen_acto: str, resumen_conceptos: str,
                   criterios: list[Criterio], material: Material,
                   es_recurso: bool = False, partes=None
                   ) -> tuple[str, str, list[str]]:
    """Devuelve (estudio, advertencias, avisos)."""
    kw = dict(model=MODELO_ESTUDIO, max_completion_tokens=16000,
              messages=[{"role": "user", "content": prompt_estudio(
                  resumen_acto, resumen_conceptos, criterios, material,
                  es_recurso, partes)}])
    if ESFUERZO_ESTUDIO:
        kw["reasoning_effort"] = ESFUERZO_ESTUDIO
    r = await cliente.chat.completions.create(**kw)
    crudo = (r.choices[0].message.content or "").strip()
    estudio, advertencias = separar_advertencias(crudo)
    avisos = revisar(estudio, criterios, material)
    if partes is not None:
        import fase_partes
        avisos.extend(fase_partes.revisar_partes(estudio, partes))
    return estudio, advertencias, avisos
