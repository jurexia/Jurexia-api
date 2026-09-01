"""FASES 1-3 del redactor: los dos resúmenes y los problemas jurídicos.

David describió el orden y aquí está COMPROBADO sobre 40 estudios de fondo
firmados del corpus KINGSTON: el estudio abre con el resumen de lo que resolvió
la responsable y sigue con el de lo que se reclama. **El 81% lo hace en ese
orden** (17 de 21 donde ambos bloques son detectables), y los dos viven en el
primer quinto del estudio — el acto al 9%, los conceptos al 16%.

LA REGLA DE ESTILO QUE NADIE ESCRIBE PERO TODOS SIGUEN
══════════════════════════════════════════════════════

El contraste de TIEMPO VERBAL es lo que hace que la prosa suene a tribunal:

    lo que hizo la responsable  →  PASADO
        consideró (20) · concluyó (12) · determinó (8) · resolvió (6)
        precisó (3) · señaló (3) · sostuvo (3)

    lo que reclama la parte     →  PRESENTE
        argumenta (30) · alega (22) · aduce (22) · sostiene (18)
        señala (7) · refiere (4) · manifiesta (4)

Uno ya ocurrió y consta; el otro se está diciendo ahora ante el tribunal.
Invertirlo delata al escrito.

Y los sujetos también son fijos: «la Sala», «el tribunal», «la Sala
responsable», «la autoridad responsable» para el primero; «la quejosa», «el
quejoso», «la parte quejosa» para el segundo.

LAS MEDIDAS
═══════════
    resumen del acto reclamado ....... mediana 438 palabras
    resumen de conceptos o agravios .. mediana 472 palabras
    estudio completo ................. mediana 3,454 palabras

Es decir, **el 26% del estudio son los dos resúmenes**. No son un preámbulo:
son un cuarto del documento, y son la parte que David dijo que NO necesita su
intervención.

QUÉ SE ESCANEA Y QUÉ NO — decisión de coste de David (28-ago-2026)
══════════════════════════════════════════════════════════════════
La fecha de presentación NO se saca leyendo la demanda escaneada: el secretario
la lee del sello en un segundo, y pagar OCR de un expediente entero para
obtener un dato es tirar el dinero. En la ficha, esos campos los teclea él.

El OCR se reserva para donde de verdad rinde: el **auto de trámite**, y sobre
todo el **acto reclamado** y los **conceptos**, que es lo que alimenta estos
dos resúmenes y el estudio.
"""

from __future__ import annotations

# ── Vocabulario medido, para que el prompt no lo invente ──────────────────

# ESTAS LISTAS SE QUEDAN COMO RESPALDO, pero el que manda es el catálogo.
# Estaban medidas sobre engroses de AMPARO DIRECTO y se entregaban a los cuatro
# tipos; `SUJETOS_PARTE[:3]` son las tres variantes de «quejoso» y ninguna de
# «recurrente», así que el prompt ORDENABA llamar quejosa a la autoridad
# hacendaria. Ahora `tipos_asunto.sujetos_de(tipo)` decide, y esto queda para
# cuando el tipo no consta.
import tipos_asunto as _ta_r


def _sujetos(tipo: str) -> dict:
    return _ta_r.sujetos_de(tipo or "amparo_directo")


SUJETOS_RESPONSABLE = (
    "la Sala", "la Sala responsable", "la autoridad responsable",
    "la responsable", "el tribunal de origen", "el juez de origen",
)

VERBOS_RESPONSABLE = (          # SIEMPRE en pretérito
    "consideró", "concluyó", "determinó", "resolvió", "precisó",
    "señaló", "sostuvo", "estimó",
)

SUJETOS_PARTE = (
    "el quejoso", "la quejosa", "la parte quejosa", "el recurrente",
    "el peticionario de garantías", "el impetrante",
)

VERBOS_PARTE = (                # SIEMPRE en presente
    "argumenta", "alega", "aduce", "sostiene", "señala", "refiere",
    "manifiesta", "se duele",
)

PALABRAS_RESUMEN_ACTO = 438
PALABRAS_RESUMEN_CONCEPTOS = 472


def instrucciones_resumen_acto(tipo_asunto: str = "") -> str:
    # NI SIQUIERA RECIBÍA EL BOOLEANO. Es el prompt que fija cómo se llama al
    # órgano —«la responsable»— y no sabía nada del asunto, así que en una
    # queja ordenaba llamar «responsable» al Juzgado de Distrito, que es el
    # órgano de control cuya decisión se recurre, no una parte.
    _sj = _sujetos(tipo_asunto)["organo"]
    return f"""RESUMEN DEL ACTO RECLAMADO O SENTENCIA RECURRIDA

Abre el estudio con esto. Cuenta qué resolvió la autoridad y con qué razones,
de modo que quien lea entienda la resolución impugnada sin tenerla enfrente.

- TIEMPO VERBAL: PRETÉRITO, sin excepción. {', '.join(VERBOS_RESPONSABLE[:6])}.
  Lo que la responsable hizo ya ocurrió y consta en autos.
- SUJETO: {', '.join(_sj[:4])}. Nunca su nombre propio.
- NO LA CALIFIQUES TODAVÍA. Aquí sólo se reconstruye su razonamiento con
  fidelidad; el juicio viene después, en el estudio.
- CADA AFIRMACIÓN ANCLADA a su origen, para que el secretario coteje sin
  releer. Se marca con [[p.7 §3]] al final de la frase —página y párrafo— y NO
  entre paréntesis: el ensamblador convierte esas marcas en NOTAS AL PIE con la
  forma «Cfr. página 7, párrafo 3», que es como se cita en una sentencia. Un
  «(p. 7)» en mitad del texto ensucia la prosa y hay que borrarlo a mano.
- EXTENSIÓN: alrededor de {PALABRAS_RESUMEN_ACTO} palabras, que es la mediana
  medida en los engroses reales."""


# ── Cómo se estructuran los conceptos, medido sobre 72 apartados reales ──────
#
# LA REGLA DE ORO, y es contraintuitiva: NO SE AGRUPA EN LA SÍNTESIS, SE AGRUPA
# EN LA SOLUCIÓN. La síntesis respeta el orden y el número que propuso el
# quejoso; el reagrupamiento se anuncia después, al abrir el estudio, y siempre
# con fundamento en el artículo 76 de la Ley de Amparo. Sólo 6 de 72 apartados
# agrupan ya dentro de la síntesis.
#
# Y NO ES UN PÁRRAFO POR CONCEPTO: es un APARTADO por concepto, con MEDIANA DE
# TRES párrafos cada uno. El apartado entero ronda los 10 párrafos.
#
# El ordinal explícito es OPCIONAL —32% lo usa, 42% corre por conectores— pero
# la separación NO lo es.
PARRAFOS_POR_CONCEPTO = 3
CONCEPTOS_TIPICOS = "de 1 a 7; lo normal, entre 2 y 4"

BISAGRA_CONCEPTOS = (
    "En contra de esas consideraciones, la parte quejosa plantea los siguientes "
    "conceptos de violación:")
BISAGRA_AGRAVIOS = (
    "En contra de las anteriores consideraciones, la parte recurrente formula "
    "los agravios siguientes:")

# Los conectores con que enlaza cuando no numera, por frecuencia real.
CONECTORES_CONCEPTOS = ("Finalmente", "Asimismo", "Además", "También",
                        "Por otro lado", "Aunado a lo anterior", "Adicionalmente",
                        "En diverso aspecto")


def instrucciones_resumen_conceptos(es_recurso: bool = False,
                                    tipo_asunto: str = "") -> str:
    # EL EJE ES EL TIPO, NO UN BOOLEANO. Un booleano abre dos caminos donde
    # hacen falta cuatro: los tres recursos entraban por la misma rama y esa
    # rama sólo cambiaba «conceptos de violación» por «agravios», nunca quién
    # promueve. `es_recurso` se conserva por si alguien llama sin el tipo.
    _t = tipo_asunto or ("amparo_revision" if es_recurso else "amparo_directo")
    _voc = _ta_r.vocabulario_de(_t)
    _sj = _sujetos(_t)
    q = _voc["combate"]
    sing = _voc["combate_singular"]
    parte = _voc["parte"]
    bisagra = _sj["bisagra"]
    return f"""RESUMEN DE LOS {q.upper()}

Va inmediatamente después del resumen del acto reclamado.

ESTRUCTURA — medida sobre 72 apartados reales de este tribunal:
- ABRE CON LA BISAGRA, tal cual: «{bisagra}»
- UN APARTADO POR {sing.upper()}, en el orden y con el número que planteó quien
  promueve. NO los fundas, NO los reordenas y NO los agrupas aquí: el
  reagrupamiento por temas se anuncia DESPUÉS, al abrir el estudio, y con el
  artículo 76 de la Ley de Amparo. Una demanda puede traer siete {q} repetitivos
  y aun así la síntesis los respeta uno por uno.
- CADA APARTADO, unos {PARRAFOS_POR_CONCEPTO} párrafos. No una línea: un
  {sing} resumido en media frase no se puede contestar después.
- ENLÁZALOS de una de estas dos formas, sin mezclarlas:
    · con el ORDINAL: «En el primer {sing} {parte} aduce que…», «En el
      segundo {sing} afirma que…», «En el tercero sostiene que…»
    · o con CONECTORES: {', '.join(f'«{c}»' for c in CONECTORES_CONCEPTOS[:6])}.
- EL ÚLTIMO SE ABRE CON «Finalmente». Aparece así en 46 de 72 apartados.
- SI HAY UNO SOLO, se dice: «En el único {sing} formulado, {parte} se duele
  de…»

Y lo de siempre:
- TIEMPO VERBAL: PRESENTE, sin excepción. {', '.join(VERBOS_PARTE[:6])}.
- SUJETO: {', '.join(_sj['parte'][:3])}.
- NO LOS CALIFIQUES. Aquí sólo se expone lo que se alega; el juicio viene en el
  estudio.
- EXTENSIÓN: alrededor de {PALABRAS_RESUMEN_CONCEPTOS} palabras en total."""


def instrucciones_problemas(global_primero: bool = True) -> str:
    """Los problemas jurídicos, del contraste entre los dos resúmenes.

    David planteó que un PROBLEMA GLOBAL puede ser más práctico que varios
    sueltos, y tiene razón en los asuntos de una sola cuestión toral: el
    estudio se ordena mejor alrededor de una pregunta que de cinco.
    """
    base = """PROBLEMAS JURÍDICOS

Salen del CONTRASTE entre los dos resúmenes: lo que la responsable resolvió
frente a lo que se combate. No de la demanda sola ni del acto solo.

- Cada uno se redacta COMO PREGUNTA, que es como se resuelve.
- SEÑALA LAS DOS COSAS, no sólo una. Este apartado pedía únicamente el
  IMPEDIMENTO que llevaría a inoperancia —que el planteamiento no combata la
  razón toral, que sea novedoso, que verse sobre cuestión firme— y no pedía
  nada en el sentido contrario. Un cuestionario que sólo pregunta por lo que
  descalifica produce un expediente lleno de razones para no entrar, y eso no
  es neutralidad: es una tesis con formato de pregunta.
    · «impedimento»: el obstáculo técnico, si de verdad lo adviertes.
    · «apoyo»: lo que el planteamiento tiene A SU FAVOR, si lo tiene —que
      combate la razón toral de frente, que hay jurisprudencia obligatoria en
      ese sentido, que la constancia que invoca consta—.
  Si sólo ves uno de los dos, pon el otro en null. Lo que no vale es mirar
  sólo hacia un lado.
- No propongas el sentido. El sentido lo pone el secretario.

- JERARQUÍA. Marca UNO como «principal»: aquel del que dependen los demás,
  el que si prospera vuelve innecesario estudiar el resto. Los demás son
  «accesorio». Si de verdad son independientes entre sí —cada uno se sostiene
  y se resuelve solo— marca «principal» sólo el primero y di en «depende_de»
  null en todos: la independencia se declara, no se supone.

- LA PREGUNTA NO PUEDE SER TENDENCIOSA. Una pregunta que ya lleva dentro la
  respuesta no es un problema jurídico, es una conclusión disfrazada. «¿Fue
  ilegal que la responsable omitiera valorar la prueba?» presupone la omisión
  y presupone la ilegalidad; lo que se pregunta es «¿la responsable valoró la
  prueba pericial y, de no haberlo hecho, esa omisión trasciende al fallo?».
  Escribe la pregunta de modo que las dos respuestas quepan en ella."""
    if global_primero:
        base += """
- FORMULA PRIMERO UN PROBLEMA GLOBAL: la cuestión toral de la que dependen
  las demás. Si el asunto se resuelve con ella, el estudio se ordena alrededor
  de esa sola pregunta y se gana claridad. Sólo desglosa en problemas
  particulares cuando de verdad sean independientes entre sí."""
    return base


# ═══════════════════════════════════════════════════════════════════════════
# QUINTO. Antecedentes — medido sobre 199 apartados reales del corpus
# ═══════════════════════════════════════════════════════════════════════════
#
# NO ES EL RESUMEN DEL ACTO, y confundirlos es el error natural: los dos salen
# de la misma sentencia. Pero el resumen cuenta lo que la responsable RESOLVIÓ
# y por qué; los antecedentes cuentan lo que PASÓ en el juicio de origen. Uno
# es razonamiento, el otro es crónica.
#
# Y se nota en la medida: 645 palabras en 17 párrafos de 37 —frases cortas de
# trámite— frente a las 438 del resumen en prosa larga.
#
# Los verbos lo confirman: dictó (186), admitió (112), interpuso (82),
# resolvió (80), confirmó (64), turnó (49). Todos de PROCEDIMIENTO, todos en
# pretérito. Y los párrafos arrancan «Por auto de», «En proveído de», «En auto
# de», «Seguido el juicio», «Inconforme con esa resolución».

PALABRAS_ANTECEDENTES = 645
PARRAFOS_ANTECEDENTES = 17

# Las cuatro entradas que usa el corpus, por frecuencia.
ENTRADAS_ANTECEDENTES = (
    "Para contextualizar el estudio de los motivos de disenso",           # 51
    "Previo al análisis de los conceptos de violación que se propone",    # 39
    "Previo al análisis de los conceptos de violación, es menester",      # 28
    "A efecto de dar claridad a la presente resolución",                  # 26
)

VERBOS_ANTECEDENTES = ("dictó", "admitió", "interpuso", "resolvió",
                       "confirmó", "turnó", "presentó", "revocó")

ARRANQUES_ANTECEDENTES = ("Por auto de", "En proveído de", "En auto de",
                          "Seguido el juicio", "Inconforme con esa resolución",
                          "Radicada la demanda")


def instrucciones_antecedentes() -> str:
    return f"""QUINTO. ANTECEDENTES

Lo que PASÓ en el juicio de origen, en orden cronológico. NO es el resumen de
lo que la responsable resolvió —eso va aparte y después—: aquí sólo se cuenta
el trámite, para que quien lea entienda de dónde viene el asunto.

- ARRANCA con una de estas fórmulas: «{ENTRADAS_ANTECEDENTES[0]}…» o
  «{ENTRADAS_ANTECEDENTES[3]}…».
- PÁRRAFOS CORTOS: mediana de 37 palabras, unos {PARRAFOS_ANTECEDENTES} en
  total, alrededor de {PALABRAS_ANTECEDENTES} palabras. Un hecho procesal por
  párrafo, nada de encadenar.
- PRETÉRITO y verbos de TRÁMITE: {', '.join(VERBOS_ANTECEDENTES[:6])}.
- ASÍ EMPIEZAN los párrafos en los engroses reales:
  {'; '.join(f'«{a}…»' for a in ARRANQUES_ANTECEDENTES[:5])}.
- CADA FECHA EN LETRA, como en todo documento judicial.
- Los puntos resolutivos de las sentencias de origen se TRANSCRIBEN entre
  comillas cuando importan al asunto.
- NO opines, NO califiques y NO adelantes el estudio."""
