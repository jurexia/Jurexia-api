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
    sentido: str                      # fundado | infundado | inoperante |
                                      # ineficaz | innecesario
    razonamiento: str = ""            # el porqué, que es lo que de verdad alinea
    # PRINCIPAL o ACCESORIO. El principal es aquel del que dependen los demás:
    # si prospera, el estudio de los otros queda sin materia. Sin esta marca no
    # se puede aplicar la sustracción de materia ni ordenar el estudio por
    # prelación lógica, que es como se ordena un engrose.
    jerarquia: str = "accesorio"
    # La distribución del acervo sobre ESTE problema: {"sentido", "porcentaje",
    # "n", "confianza", "frase"}. No funda nada —un colegiado no obliga a
    # otro— pero dice si el sentido va con la corriente o contra ella, y eso
    # cambia lo que hay que escribir.
    prediccion: dict = field(default_factory=dict)


@dataclass
class Material:
    """Lo que el RAG encontró para un problema. Sólo entra lo VERIFICADO."""
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
    sondeo: object = None
    # LA MATERIA VIAJA CON EL MATERIAL, no como parámetro. Hay cuatro sitios que
    # arman el prompt y cada parámetro nuevo es un sitio donde olvidarlo; el
    # Material ya llega a todos. Y aquí importa de veras: entregar la
    # arquitectura equivocada no es un defecto de forma, es mandar escribir al
    # revés —en laboral se pide ciclos cortos y en administrativa lo contrario—.
    materia: str = ""
    # Para nombrar a las partes con las figuras que existen en este tipo.
    tipo_asunto: str = "amparo_directo"
    # LA ENTIDAD, por el mismo motivo que la materia: viaja con el material
    # porque el material ya llega a todos los prompts. Sale de la colección
    # estatal que eligió el secretario («leyes_queretaro» → «Querétaro»).
    entidad: str = ""


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
_SUPLENCIA_ABSOLUTA = {
    "laboral": ("el trabajador", "79, fracción V"),
    "penal": ("el reo", "79, fracción III"),
}


def _aviso_de_suplencia(criterios: list, materia: str) -> list:
    m = (materia or "").strip().lower()
    if m not in _SUPLENCIA_ABSOLUTA:
        return []
    if not any("inoperan" in str(getattr(c, "sentido", "")).lower()
               for c in (criterios or [])):
        return []
    quien, precepto = _SUPLENCIA_ABSOLUTA[m]
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
            ]


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


def _bloque_criterio(criterios: list[Criterio], materia: str = "") -> str:
    if not criterios:
        return ""
    lineas = ["", "═" * 71,
              "EL CRITERIO DEL SECRETARIO — DIRECTIVA INNEGOCIABLE",
              "═" * 71,
              "Tu papel NO es decidir el fallo: es CONSTRUIR la mejor demostración",
              "jurídica posible del sentido que él ya fijó. Elige los argumentos, las",
              "tesis y el orden de estudio que lo sostengan con el mayor rigor.", ""]
    # EL ORDEN DE ESTUDIO ES EL DE PRELACIÓN LÓGICA, no el de llegada: primero
    # el principal, del que dependen los demás. Un engrose que estudia un
    # accesorio antes que el problema del que depende obliga a rehacerlo.
    _ord = sorted(enumerate(criterios),
                  key=lambda x: (0 if (x[1].jerarquia or "").lower() == "principal"
                                 else 1, x[0]))
    for i, (_, c) in enumerate(_ord, 1):
        lineas.append(f"{i}. [{(c.jerarquia or 'accesorio').upper()}] {c.problema}")
        lineas.append(f"   SENTIDO: {c.sentido.upper()}")
        if (c.sentido or "").lower() == "innecesario":
            lineas.append("   NO SE ESTUDIA: quedó sin materia por el sentido "
                          "del principal. Se dice en una frase y se pasa; ni "
                          "lo califiques ni lo contestes.")
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
        "SI ENCUENTRAS UN OBSTÁCULO SERIO para ese sentido —una jurisprudencia",
        "obligatoria en contra, una causal de improcedencia—, NO cambies el",
        "sentido: dilo en el apartado ADVERTENCIAS para que él lo valore.",
    ]
    lineas += _aviso_de_suplencia(criterios, materia)
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
            # DOS DATOS DISTINTOS, Y YO LOS TENÍA FUNDIDOS EN UNO. Que un
            # criterio VINCULE y que SEA jurisprudencia no es lo mismo: hay
            # tesis aisladas de la Corte que orientan y jurisprudencia de
            # colegiado que obliga en su circuito. La etiqueta decía sólo lo
            # primero, así que el modelo no tenía cómo saber que estaba citando
            # una tesis aislada —y la llamó jurisprudencia—.
            fuerza = "OBLIGATORIA" if t.get("obligatoria") else "orientadora"
            tipo = str(t.get("tipo") or "").strip() or "tipo no declarado"
            p.append(f"\n  · [{fuerza}] [{tipo}] Registro "
                     f"{t.get('registro','')} — {t.get('instancia','')}")
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


def _bloque_aportado(contexto: str) -> str:
    """El documento que el secretario subió porque el acervo no lo tenía."""
    c = (contexto or "").strip()
    if not c:
        return ""
    return ("\n═══════════════════════════════════════════════════════════════\n"
            "DOCUMENTO APORTADO POR EL SECRETARIO — no estaba en el acervo\n"
            "═══════════════════════════════════════════════════════════════\n"
            "Lo trae quien tiene el expediente delante. Cítalo por lo que dice,\n"
            "identificándolo como el documento aportado; NO le inventes un\n"
            "registro ni lo trates como jurisprudencia.\n\n"
            # POR PÁRRAFO, NO POR CARACTER. Estas constancias las subió el
            # secretario y alimentan prosa que se firma; cortarlas a mitad de
            # frase es la misma puerta por la que salió «el texto proporcionado
            # se interrumpió» dentro del considerando quinto.
            + _recorte_limpio(c, 20000) + "\n")



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
fondo: (1) transcribes el texto literal del precepto entre comillas; (2) derivas
la regla con una frase puente —«De dicho numeral es posible advertir que, por
regla general,…»—; (3) entra la autoridad TRANSCRITA, no citada: «es aplicable
la jurisprudencia X, sustentada por la Primera Sala…, de rubro y texto
siguientes:» y sigue el rubro y el texto íntegro; (4) nombras la operación que
enlaza ese criterio con estos hechos. La cadena completa se cumple en el 53% de
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


def _bloque_arquitectura(materia: str) -> str:
    """Lo común más UNA arquitectura de materia. Nunca dos: son opuestas."""
    m = (materia or "").strip().lower()
    propia = _ARQUITECTURA.get(m, "")
    if not propia:
        # Sin materia identificada se entrega sólo lo común. Entregar la
        # arquitectura equivocada es peor que no entregar ninguna: en laboral
        # manda escribir en ciclos cortos y en administrativa manda justo lo
        # contrario.
        return _ARQUITECTURA_COMUN
    return _ARQUITECTURA_COMUN + propia


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
        if s.startswith("fundad"):
            return "concede"
    return "niega"


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
                   contexto: str = "", materia: str = "") -> str:
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
planteamientos de {parte}.»— y entra al caso. Un marco que se recibe y
no se escribe deja el estudio resolviendo sin premisa mayor.
"""
    return f"""Eres el secretario de un Tribunal Colegiado de Circuito redactando el
estudio de fondo de {_clase}. Escribes mejor que la media del
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
- EL ÓRGANO RECURRIDO es {_org}; este tribunal se
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

ARQUITECTURA — CUATRO PASOS POR CADA {q1}, SIEMPRE LOS CUATRO Y EN ESTE ORDEN.

Es la técnica silogística, y no es un adorno de método: es lo que permite
comprobar, leyendo, que no quedó nada sin contestar y que lo contestado se
sostiene. Un apartado por planteamiento, en el orden en que se plantearon,
abierto por su ordinal en letra.

  PASO 1 — EL PLANTEAMIENTO, EN SU VERSIÓN MÁS FUERTE.
  Se enuncia con la voz de quien lo formula, no con la del tribunal, y en su
  mejor versión: «En el primer {q1} {parte} sostiene que…». Prohibido
  caricaturizarlo para tumbarlo después: si el argumento tiene un punto, se
  dice, y luego se explica por qué no basta. Un planteamiento debilitado al
  enunciarlo produce una respuesta que no responde.

  PASO 2 — LA PREMISA NORMATIVA, ABSTRACTA Y CON SU FUENTE.
  Qué dice la norma o el criterio que gobierna el punto, enunciado de modo que
  valga para cualquier caso igual: «Del precepto transcrito deriva la regla de
  que…», «con arreglo a la jurisprudencia de la Segunda Sala derivada de la
  contradicción de tesis…». Aquí NO se nombra todavía al promovente, ni al
  órgano recurrido, ni el expediente: si la frase no vale para otro asunto
  idéntico, no es una premisa, es una conclusión adelantada.

  PASO 3 — LA APLICACIÓN AL CASO, CONFRONTANDO.
  Aquí entran los hechos de ESTE expediente contra la regla del paso 2, y aquí
  —y sólo aquí— se abre con «en el caso», «en la especie». La confrontación se
  escribe: qué dice la constancia, qué exige la norma, y por qué encaja o no.
  Un salto del paso 2 al paso 4 sin este eslabón es la afirmación sin prueba
  que se cae en revisión.

  PASO 4 — LA CONCLUSIÓN CALIFICADA, Y SUS CONSECUENCIAS.
  Una sola calificación —«es fundado», «es infundado», «es inoperante»— y qué
  se sigue de ella. Si es inoperante, la razón TÉCNICA de la inoperancia: que
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
  Aquí ha de ser al revés. Transcribe cuando la letra decide —el precepto
  discutido, el párrafo cuya redacción se combate— y nunca para llenar. Del
  criterio que invoques, trae la REGLA en una o dos frases y sigue razonando:
  el rubro y el registro identifican la tesis; su texto íntegro va en la nota
  al pie, no en el cuerpo.

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

      Sirve de apoyo el criterio de registro 2022074:

  Y ahí se detiene el párrafo. NO ESCRIBAS TÚ NI EL TIPO NI EL ÓRGANO: no digas
  «la jurisprudencia», no digas «tesis aislada», no digas «de la Primera Sala».
  El documento los pone solo, tomados del acervo, junto con el rubro y el texto
  íntegro. Antes este ejemplo nombraba una Sala concreta y el modelo lo copiaba
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
{_bloque_aportado(contexto)}
{partes.bloque() if partes is not None else ""}
{marco if isinstance(marco, str) else ""}
{_bloque_arquitectura(materia or getattr(material, "materia", ""))}
{_bloque_precedente(material, criterios)}
{_bloque_criterio(criterios, materia or getattr(material, "materia", ""))}
{_bloque_material(material)}

═══════════════════════════════════════════════════════════════════════
LO QUE RESOLVIÓ {_org_rotulo}
═══════════════════════════════════════════════════════════════════════
{resumen_acto}

═══════════════════════════════════════════════════════════════════════
LO QUE SE COMBATE
═══════════════════════════════════════════════════════════════════════
{resumen_conceptos}

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
    r"|resuelva|ordene|deber[áa]|proceda|realice|valore)", re.I | re.M)


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


def _cierre_operativo(estudio: str, criterios: list) -> str:
    """Si se concede, los efectos van como órdenes numeradas y verificables."""
    concede = any("fundado" in str(getattr(c, "sentido", "") or c).lower()
                  for c in (criterios or []))
    if not concede or "efecto" not in (estudio or "").lower():
        return ""
    if not _RX_ORDEN_NUMERADA.search(estudio or ""):
        return ("Se concede y los efectos van en prosa. Las sentencias mejor "
                "calificadas los escriben como lista numerada de órdenes en "
                "imperativo a la responsable —«1. Deje insubsistente el laudo; "
                "2. Dicte otro en el que…»—, cada una verificable. Así se "
                "cumplen y así se comprueba su cumplimiento.")
    return ""


def revisar(estudio: str, criterios: list[Criterio], material: Material,
            resumen_acto: str = "", marco: str = "") -> list[str]:
    """Lo comprobable sin modelo. Ninguna de estas es opinión."""
    avisos: list[str] = []

    # Lo que añadió la medición sobre 1,946 sentencias del acervo.
    for comprobacion in (
            _tipo_mal_atribuido(estudio, material),
            _convencional_completo(estudio),
            _cierre_operativo(estudio, criterios)):
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
        # EL AVISO MIDE LO QUE ESCRIBIÓ EL MODELO, NO LO QUE SE ENTREGA: el
        # compositor borra el eco después. Decirle al secretario que el
        # proyecto repite una tesis cuando ya no la repite lo manda a buscar
        # algo que no está, y un aviso que no se comprueba deja de leerse.
        avisos.append(
            f"{'La tesis' if len(repetidas) == 1 else 'Las tesis'} "
            f"{', '.join(repetidas)} venían repetidas tras su cita y el eco se "
            f"BORRÓ al componer. El documento sale limpio; queda dicho por si "
            f"al leerlo echas en falta el enlace con el caso, que es lo que "
            f"debía ir ahí: «Conforme a la jurisprudencia citada, es claro que…»")

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
    if _es_laboral and re.search(r"\binoperant", estudio, re.I):
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

    # 2. El sentido dictado tiene que aparecer… SALVO la inoperancia que la
    #    suplencia prohíbe. Esta regla y la anterior se contradecían: una
    #    reprochaba escribir «inoperante» y la otra reprochaba NO escribirlo.
    #    Entre las dos dejaban al modelo sin salida buena, y eligió obedecer a
    #    la que iba rotulada innegociable.
    _mat = str(getattr(material, "materia", "") or "").strip().lower()
    for c in criterios:
        raiz = c.sentido[:7].lower()
        if not raiz or raiz in estudio.lower():
            continue
        if raiz.startswith("inoperan") and _mat in _SUPLENCIA_ABSOLUTA:
            avisos.append(
                f"El criterio pedía «{c.sentido}» y el estudio no lo escribió. "
                f"En materia {_mat}, con la suplencia del artículo 79, eso "
                f"puede ser lo CORRECTO: revisa si el estudio suplió el "
                f"planteamiento y lo resolvió en el fondo. Si es así, la "
                f"calificación cambió y hay que confirmarla.")
            continue
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
                           es_recurso: bool = False, partes=None, marco=None,
                           contexto: str = ""):
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
                  es_recurso, partes, marco, contexto)}])
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
                   es_recurso: bool = False, partes=None, marco=None,
                   contexto: str = "") -> tuple[str, str, list[str]]:
    """Devuelve (estudio, advertencias, avisos)."""
    kw = dict(model=MODELO_ESTUDIO, max_completion_tokens=16000,
              messages=[{"role": "user", "content": prompt_estudio(
                  resumen_acto, resumen_conceptos, criterios, material,
                  es_recurso, partes, marco, contexto)}])
    if ESFUERZO_ESTUDIO:
        kw["reasoning_effort"] = ESFUERZO_ESTUDIO
    import llamada_modelo as _lm
    r = await _lm.crear(cliente, **kw)
    crudo = (r.choices[0].message.content or "").strip()
    estudio, advertencias = separar_advertencias(crudo)
    avisos = revisar(estudio, criterios, material, resumen_acto,
                     marco if isinstance(marco, str) else "")
    if partes is not None:
        import fase_partes
        avisos.extend(fase_partes.revisar_partes(estudio, partes))
    return estudio, advertencias, avisos
