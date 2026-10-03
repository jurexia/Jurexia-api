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


def sujetos_responsable(tipo: str = "") -> tuple:
    """El respaldo de arriba, salvo en un amparo directo de ÚNICA instancia
    (30-sep-2026): ahí «la Sala» es justo el error que David marcó en el AD
    323/2025 —un juzgado de oralidad mercantil llamado Sala—, así que se nombra
    al órgano por lo que es (`sujetos_de`, que lo lee del origen) y, de
    respaldo, con una fórmula que no afirma alzada. Hoy ningún prompt lee la
    tupla; quien la necesite, que pase por aquí."""
    if _ta_r.unica_instancia(tipo):
        return tuple(_sujetos(tipo)["organo"]) + ("el órgano de origen",)
    return SUJETOS_RESPONSABLE


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


def instrucciones_resumen_acto(tipo_asunto: str = "", objetivo: int = 0) -> str:
    # NI SIQUIERA RECIBÍA EL BOOLEANO. Es el prompt que fija cómo se llama al
    # órgano —«la responsable»— y no sabía nada del asunto, así que en una
    # queja ordenaba llamar «responsable» al Juzgado de Distrito, que es el
    # órgano de control cuya decisión se recurre, no una parte.
    _sj = _sujetos(tipo_asunto)["organo"]
    # CONTRA QUÉ VA LO QUE SE RESUME. Decía «agravios» en los cuatro tipos, que
    # en amparo directo es el nombre de lo que se alega en la APELACIÓN; en un
    # juicio de única instancia no hubo agravios de nadie (30-sep-2026, AD
    # 323/2025). Sólo en esa variante se dice lo que es; fuera de ella, igual.
    _contra = ("agravios" if not _ta_r.unica_instancia(tipo_asunto)
               else _ta_r.vocabulario_de(tipo_asunto)["combate"])
    # LO QUE TUVO POR ACREDITADO (2-oct-2026, bandera «preguntas_al_secretario»;
    # David: lo que afirma la parte «es justo lo que se debe verificar a la luz
    # de lo acreditado en el juicio, que por regla general viene dicho en la
    # sentencia reclamada o recurrida»). La propuesta y el estudio leen este
    # resumen; si calla los hechos probados, no hay contra qué contrastar.
    _acreditado = ""
    try:
        import contexto_taller as _ct_ra
        if _ct_ra.rediseno("preguntas_al_secretario"):
            _acreditado = (
                "\n- LO QUE TUVO POR ACREDITADO Y CÓMO VALORÓ LAS PRUEBAS, siempre, con su ancla:\n"
                "  qué hechos dio por probados y cuáles no, con qué pruebas y por qué les dio o\n"
                "  les negó valor. Con esto se contrasta después lo que afirma la parte; lo que\n"
                "  el resumen calle ya no se podrá verificar.")
    except Exception:
        _acreditado = ""
    return f"""RESUMEN DEL ACTO RECLAMADO O SENTENCIA RECURRIDA

Abre el estudio con esto. Cuenta qué resolvió la autoridad y con qué razones,
de modo que quien lea entienda la resolución impugnada sin tenerla enfrente.

- TIEMPO VERBAL: PRETÉRITO, sin excepción. {', '.join(VERBOS_RESPONSABLE[:6])}.
  Lo que la responsable hizo ya ocurrió y consta en autos.
- SUJETO: {', '.join(_sj[:4])}. Nunca su nombre propio.
- NO LA CALIFIQUES TODAVÍA. Aquí sólo se reconstruye su razonamiento con
  fidelidad; el juicio viene después, en el estudio.
- COMPLETO, NO ESCOGIDO: TODAS las consideraciones de fondo, cada una con su
  razón —qué decidió, por qué, con qué precepto— y en el orden en que la
  responsable las expuso. Si resolvió cinco cuestiones, aquí hay cinco. Un
  resumen que se queda con dos deja al estudio sin poder contestar los
  {_contra} contra las otras tres, y eso se llama incongruencia.{_acreditado}
- LAS TESIS Y JURISPRUDENCIAS EN QUE SE APOYÓ SE NOMBRAN, con su clave o su
  registro tal como las cita: «…con apoyo en la jurisprudencia 2a./J. 13/2015
  (registro 2008474)». No se transcriben; se nombran, porque los {_contra}
  suelen ir contra ellas y el estudio tiene que saber cuáles son.
- CADA AFIRMACIÓN ANCLADA a su origen, para que el secretario coteje sin
  releer. Se marca con [[p.7 §3]] al final de la frase —página y párrafo— y NO
  entre paréntesis: el ensamblador convierte esas marcas en NOTAS AL PIE con la
  forma «Cfr. página 7, párrafo 3», que es como se cita en una sentencia. Un
  «(p. 7)» en mitad del texto ensucia la prosa y hay que borrarlo a mano.
- EXTENSIÓN: alrededor de {objetivo or PALABRAS_RESUMEN_ACTO} palabras. Es una
  medida de la resolución, no un tope: si hacen falta más para que cada
  consideración tenga su frase, se escriben."""


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
                                    tipo_asunto: str = "",
                                    objetivo: int = 0) -> str:
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
- CADA APARTADO, unos {PARRAFOS_POR_CONCEPTO} párrafos COMO MÍNIMO, y dentro
  UN PÁRRAFO POR CADA ARGUMENTO DISTINTO que contenga. Un {sing} «único» de
  sesenta páginas trae diez razones, y las diez se resumen, cada una con su
  aspecto técnico: qué precepto se dice violado, por qué, y qué consecuencia
  se pide. Resumir un {sing} largo en tres párrafos es dejar sin contestar lo
  que no se resumió. Y NO EMPIECES POR EL FINAL: el cierre del escrito
  —«por lo expuesto procede revocar»— es la petición, no el argumento.
- LAS TESIS Y JURISPRUDENCIAS QUE INVOCA {parte} SE NOMBRAN, en el párrafo del
  argumento que apoyan y con la clave o el registro tal como las cita:
  «…e invoca la jurisprudencia 2a./J. 60/2007 (registro 172239)». No se
  transcriben; se nombran, porque el estudio tiene que hacerse cargo de ellas
  y de lo que se les atribuye.
- LO QUE NO ES {sing} NO SE RESUME: el ofrecimiento de pruebas, la designación
  de delegados o autorizados, el domicilio para notificaciones y las
  peticiones de trámite.
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
- CADA APARTADO ANCLADO A SU ORIGEN, igual que el resumen del acto reclamado.
  Se marca con [[p.7 §3]] al final de la frase —página y párrafo del escrito de
  quien promueve— y NO entre paréntesis: el ensamblador convierte esas marcas
  en NOTAS AL PIE con la forma «Cfr. página 7, párrafo 3».
  Es lo que permite al secretario cotejar que la síntesis dice lo que el
  escrito dice, sin releerlo entero. Un {sing} mal resumido se contesta mal, y
  eso no se ve en el proyecto: se ve en el amparo que vuelve.
  UNA MARCA POR APARTADO, al final del primer párrafo de cada uno, que es
  donde se dice de qué va. No marques cada frase: la nota repetida no aporta y
  ensucia el pie.
  Si no puedes ubicar la página, NO INVENTES el número: deja el apartado sin
  marca. Una nota al pie que manda a la página equivocada es peor que ninguna.
- EXTENSIÓN: alrededor de {objetivo or PALABRAS_RESUMEN_CONCEPTOS} palabras en
  total. Es una medida del escrito, no un tope: si hacen falta más para que
  cada argumento tenga su párrafo, se escriben."""


def instrucciones_problemas(global_primero: bool = True, tipo_asunto: str = "") -> str:
    """Los problemas jurídicos, del contraste entre los dos resúmenes.

    David planteó que un PROBLEMA GLOBAL puede ser más práctico que varios
    sueltos, y tiene razón en los asuntos de una sola cuestión toral: el
    estudio se ordena mejor alrededor de una pregunta que de cinco.

    `tipo_asunto` (30-sep-2026) sólo cuenta en un amparo directo de ÚNICA
    instancia: el ejemplo de la dependencia preguntaba «¿Debía la Sala…?», y
    una pregunta de ejemplo se copia con su sujeto; ahí el sujeto es el órgano
    que dictó lo reclamado (`sujetos_de`). Sin tipo, o sin esa variante, igual.
    """
    _ej_org = "la Sala"
    if _ta_r.unica_instancia(tipo_asunto):
        _ej_org = _sujetos(tipo_asunto)["organo"][0]
    base = f"""PROBLEMAS JURÍDICOS

Salen del CONTRASTE entre los dos resúmenes: lo que la responsable resolvió
frente a lo que se combate. No de la demanda sola ni del acto solo.

- Cada uno se redacta COMO PREGUNTA, que es como se resuelve. Y LA PREGUNTA
  CABE EN UNA LÍNEA: máximo 25 palabras, una sola cuestión, sin alternativas
  «o…» dentro. Medidas en un proyecto real, las cuatro preguntas salieron de
  353, 380, 406 y 419 caracteres, con la disyuntiva metida dentro:

    MAL: «¿La audiencia se limitó a admitir la reconvención y diferir la
          diligencia, o durante ella se recibieron y proveyeron diversas
          actuaciones, se adoptaron determinaciones y se permitió que la
          quejosa ejerciera su defensa sin asesor jurídico, de modo que deba
          determinarse el alcance de la afectación procesal alegada?»
    BIEN: «¿La quejosa compareció a la audiencia sin asesor jurídico?»

  Una pregunta de cuatrocientos caracteres no fija la cuestión: la reformula
  entera, y quien la lee sigue sin saber qué se va a decidir. Si de verdad hay
  dos cuestiones, son DOS problemas, no uno con una «o» en medio.
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
- «depende_de» NO ES DE ORDEN, ES DE PREMISA. Un problema depende de otro
  cuando su respuesta PRESUPONE la de aquél: si el principal cae, éste cae
  con él, y si prospera, éste queda sin materia. «¿Debía {_ej_org} estudiar
  los alegatos contra el crédito?» depende de «¿debió admitirse la ampliación
  que metía el crédito en la litis?»: sin ampliación no hay crédito en la
  litis ni alegatos que estudiar. Que dos temas se parezcan no los hace
  dependientes; que uno viva de la premisa del otro, sí.

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

# SIN ALZADA (30-sep-2026). Los verbos y los arranques de arriba se midieron en
# un corpus donde lo reclamado venía casi siempre de una apelación: «interpuso»,
# «confirmó», «revocó» e «Inconforme con esa resolución» son la cadena de la
# segunda instancia, y dados a un juicio oral mercantil (AD 323/2025) el motor
# la narraba aunque no hubiera existido. Éstos son los del trámite de la
# primera y única instancia. NO ESTÁN MEDIDOS como los de arriba: cuando haya
# corpus de amparos directos sin alzada, hay que medirlos.
VERBOS_ANTECEDENTES_UNICA = ("presentó", "admitió", "emplazó", "desahogó",
                             "dictó", "resolvió")
ARRANQUES_ANTECEDENTES_UNICA = ("Por auto de", "En proveído de", "En auto de",
                                "Radicada la demanda", "Seguido el juicio")


# ═══ ANTECEDENTES EN PROSA, SIN ENUMERAR (2-oct-2026) ════════════════════════
# David: «debemos quitar la enumeración de antecedentes», y en la misma petición
# que el proyecto «sigue siendo a veces muy extenso». Detrás de la bandera
# `antecedentes_en_prosa`: el considerando se conserva, pero en pocos párrafos
# de prosa encadenada —uno por etapa— y más corto. Las medidas de arriba (17
# párrafos, 645 palabras) son las del corpus y se quedan para la bandera
# apagada; éstas son la meta nueva, NO medidas todavía en engroses reales.
import re as _re_ant  # (aquí y no arriba: el resto del módulo no lo usa)

PARRAFOS_ANTECEDENTES_PROSA = (3, 6)
PALABRAS_ANTECEDENTES_PROSA = (350, 450)

# EL NÚMERO QUE PONE EL MODELO. Estrecha a propósito: uno o dos dígitos con
# punto o paréntesis y un espacio, sólo al principio del párrafo. No toca
# «15 de marzo…» (tras la cifra no hay punto ni paréntesis) ni nada que no
# abra el párrafo.
_RX_NUMERO_ANTECEDENTE = _re_ant.compile(r"^\s*\d{1,2}[.)]\s+")

# LA FÓRMULA DE ENTRADA, en cualquiera de sus variantes del corpus, más la del
# compositor («Previo al análisis de los planteamientos…») y la de la plantilla
# («Para una mejor comprensión del asunto…»).
_RX_ENTRADA_ANTECEDENTES = _re_ant.compile(
    r"^\s*(?:Para\s+contextualizar|Previo\s+al\s+an[áa]lisis|A\s+efecto\s+de\s+dar\s+claridad"
    r"|Para\s+una\s+mejor\s+comprensi[óo]n)\b", _re_ant.I)
# Una entrada SOLA es una frase de presentación, sin hechos: corta y sin fecha.
# Si el modelo la fundió con el primer hecho («Para contextualizar…, conviene
# precisar que por escrito presentado el quince de marzo de dos mil…»), el
# párrafo ya es un hecho y no se borra.
_MAX_PALABRAS_ENTRADA = 30
_RX_FECHA_EN_AUTOS = _re_ant.compile(r"\bde\s+dos\s+mil\b|\b(?:19|20)\d\d\b", _re_ant.I)


# LA LISTA TRANSCRITA NO ES LA NUMERACIÓN DEL MODELO (revisión adversarial,
# 3-oct-2026). En la sentencia dictada en cumplimiento el prompt pide contar
# «qué ordenó la ejecutoria, con sus efectos», un párrafo por renglón; si el
# modelo los transcribe uno por renglón («1. Deje insubsistente…», «2. Dicte
# otra…»), quitarles el número alteraba la transcripción. Un párrafo que
# termina en dos puntos o anuncia lo que sigue («para los efectos
# siguientes», «en los términos siguientes») abre una lista: los párrafos
# numerados que lo siguen sin interrupción conservan su número. Se decide
# sólo con el texto que queda, así que aplicarla dos veces (la fuente común y
# luego el documento) da lo mismo que una. NO abre lista la fórmula de entrada
# («…relatar los siguientes antecedentes:») ni lo que anuncia los antecedentes
# o los hechos: lo que sigue ahí es la enumeración del propio modelo, la que
# David pidió quitar.
_RX_ANUNCIA_LISTA = _re_ant.compile(
    r"(?::|\b(?:efectos|t[ée]rminos|lineamientos|puntos|resolutivos|directrices|consideraciones)"
    r"\s+siguientes\s*[.:]?|\bsiguientes\s+(?:efectos|t[ée]rminos|lineamientos|puntos"
    r"|resolutivos|directrices)\s*[.:]?)\s*[»\"”']?\s*$", _re_ant.I)


_RX_ANUNCIA_ANTECEDENTES = _re_ant.compile(r"\b(?:antecedentes|hechos)\b[^.;]{0,40}$", _re_ant.I)


def _abre_lista(parrafo: str) -> bool:
    t = str(parrafo or "").strip()
    return (bool(_RX_ANUNCIA_LISTA.search(t))
            and not _RX_ENTRADA_ANTECEDENTES.match(t)
            and not _RX_ANUNCIA_ANTECEDENTES.search(t))


def sin_numeracion(parrafos) -> list[str]:
    """Los párrafos de los antecedentes sin el «1. » o «2) » que traiga el
    modelo, salvo los de una lista transcrita (los efectos de una ejecutoria,
    unos resolutivos): ésos lo conservan."""
    salida = []
    en_lista = False
    for p in (parrafos or []):
        crudo = str(p or "").strip()
        if not crudo:
            continue
        numerado = bool(_RX_NUMERO_ANTECEDENTE.match(crudo))
        if not numerado:
            en_lista = False
        elif not en_lista and salida and _abre_lista(salida[-1]):
            en_lista = True
        t = crudo if en_lista else _RX_NUMERO_ANTECEDENTE.sub("", crudo, count=1).strip()
        if t:
            salida.append(t)
    return salida


def es_solo_entrada(parrafo: str) -> bool:
    """¿El párrafo es sólo la fórmula de entrada, sin ningún hecho?"""
    t = str(parrafo or "").strip()
    return (bool(_RX_ENTRADA_ANTECEDENTES.match(t))
            and len(t.split()) <= _MAX_PALABRAS_ENTRADA
            and not _RX_FECHA_EN_AUTOS.search(t))


def antecedentes_en_prosa(parrafos) -> tuple[list[str], bool]:
    """Prepara los antecedentes para el documento con la bandera encendida.

    Devuelve (párrafos, trae_entrada): sin números y sin el párrafo que sólo es
    la fórmula de entrada —ésa la pone el documento, una vez—; `trae_entrada`
    es verdadero cuando la fórmula del modelo viene FUNDIDA con el primer hecho:
    entonces el documento no escribe la suya, para que no salgan dos seguidas.
    """
    ps = sin_numeracion(parrafos)
    if ps and es_solo_entrada(ps[0]):
        ps = ps[1:]
    trae = bool(ps) and bool(_RX_ENTRADA_ANTECEDENTES.match(ps[0]))
    return ps, trae


def instrucciones_antecedentes(tipo_asunto: str = "") -> str:
    # EL ÚNICO DE LOS CUATRO QUE NO RECIBÍA EL TIPO, y el que más caro sale:
    # los antecedentes se escriben en el apartado «Antecedentes» del .docx, así
    # que aquí el disparate no se queda en pantalla, se FIRMA.
    import tipos_asunto as _ta
    _verbos = _ta.verbos_del_recurrido(tipo_asunto)
    # ═══ DE DÓNDE VIENE LO RECLAMADO (30-sep-2026) ═════════════════════════
    # David: «no siempre hay una sala […] la autoridad responsable era el
    # propio juez de oralidad mercantil […] la sentencia había sido dictada en
    # cumplimiento». Dos variantes, cada una sólo si consta (`tipos_asunto`):
    #   · ÚNICA INSTANCIA: ni apelación ni toca; el último antecedente es la
    #     sentencia del juez (o de la Junta, o de la Sala del TFJA) que se
    #     reclama, y la responsable no es «una Sala o un tribunal ordinario».
    #   · EN CUMPLIMIENTO: los antecedentes cuentan el amparo anterior —número,
    #     tribunal, qué ordenó— y la sentencia nueva que lo acató. Sin ese hilo
    #     el estudio no puede separar lo vinculado de lo libre.
    # Sin ninguna de las dos, el prompt es el de siempre, letra por letra.
    _unica = _ta.unica_instancia(tipo_asunto)
    _cumpl = _ta.cumplimiento_de_amparo(tipo_asunto)
    _v_tramite = VERBOS_ANTECEDENTES_UNICA if _unica else VERBOS_ANTECEDENTES
    _arranques = ARRANQUES_ANTECEDENTES_UNICA if _unica else ARRANQUES_ANTECEDENTES
    _quien = "una Sala o un tribunal ordinario"
    _extra = ""
    # ═══ EN PROSA Y SIN ENUMERAR (2-oct-2026) ══════════════════════════════
    # David: «debemos quitar la enumeración de antecedentes». La lista nacía
    # AQUÍ, no en el número: el prompt pedía 17 párrafos de «un hecho procesal
    # por párrafo, nada de encadenar», y el documento les ponía 1…17. Quitar
    # sólo el número dejaba los mismos 17 renglones sueltos. Con la bandera se
    # piden pocos párrafos, uno por etapa, con los hechos enlazados, y más
    # cortos (también se quejó de que el proyecto es extenso). La fórmula de
    # entrada se la queda el documento: pedírsela también al modelo daba dos
    # seguidas. El rótulo pierde el ordinal, que el documento calcula. El
    # cierre «en qué paró» NO se toca: lo leen `fase_rama`, `ficha_procesal`
    # y el resolutivo de la revisión. Sin la bandera, letra por letra como antes.
    import contexto_taller as _ct_ant
    _prosa = _ct_ant.rediseno("antecedentes_en_prosa")
    if _prosa:
        _rotulo = "ANTECEDENTES"
        _forma = f"""- SIN FÓRMULA DE ENTRADA: el documento ya pone la suya. Empieza directamente
  por el primer hecho del juicio de origen.
- EN PROSA ENCADENADA Y SIN ENUMERAR: entre {PARRAFOS_ANTECEDENTES_PROSA[0]} y {PARRAFOS_ANTECEDENTES_PROSA[1]} párrafos, uno por etapa
  del asunto —el juicio de origen hasta su sentencia; el recurso o la instancia
  que siguió, si la hubo; la resolución que aquí se reclama o se recurre—, y
  alrededor de {PALABRAS_ANTECEDENTES_PROSA[0]} a {PALABRAS_ANTECEDENTES_PROSA[1]} palabras en total. Dentro de cada párrafo los
  hechos se enlazan unos con otros; nada de un hecho por renglón. Ni números
  de orden, ni viñetas, ni incisos.
- SÓLO LOS HECHOS QUE HACEN FALTA para entender la solución. Los autos de mero
  trámite que no inciden en lo que se va a resolver (turnos, vistas,
  certificaciones, prórrogas) se omiten.
"""
        _como_empiezan = "ASÍ SE ENLAZAN los hechos en los engroses reales"
        _orden_cumpl = "en su orden y encadenado"
    else:
        _rotulo = "QUINTO. ANTECEDENTES"
        _forma = f"""- ARRANCA con una de estas fórmulas: «{ENTRADAS_ANTECEDENTES[0]}…» o
  «{ENTRADAS_ANTECEDENTES[3]}…».
- PÁRRAFOS CORTOS: mediana de 37 palabras, unos {PARRAFOS_ANTECEDENTES} en
  total, alrededor de {PALABRAS_ANTECEDENTES} palabras. Un hecho procesal por
  párrafo, nada de encadenar.
"""
        _como_empiezan = "ASÍ EMPIEZAN los párrafos en los engroses reales"
        _orden_cumpl = "en su orden, un hecho por párrafo"
    if _unica:
        _org = _ta.sujetos_de(tipo_asunto)["organo"][0]
        _quien = (f"quien dictó la sentencia reclamada —aquí, {_org}, que "
                  f"resolvió el juicio en única instancia—")
        _extra += f"""- EL JUICIO SE RESOLVIÓ EN ÚNICA INSTANCIA. No hubo segunda instancia: no
  narres un recurso de apelación, un toca ni una Sala de alzada que no
  existieron. El último párrafo es la sentencia que se reclama: la que dictó
  {_org}, con su fecha y lo que resolvió, con su verbo.
"""
    if _cumpl:
        _extra += f"""- LA SENTENCIA RECLAMADA SE DICTÓ EN CUMPLIMIENTO de la ejecutoria del
  {_cumpl.get("ejecutoria") or "amparo anterior"}. Cuéntalo {_orden_cumpl}: la
  sentencia que se combatió primero; el amparo que se promovió contra ella
  —su número con cifras y tal como aparece en autos (nunca en letra), el
  tribunal que lo resolvió y qué ordenó la ejecutoria, con sus efectos
  transcritos entre comillas si el documento los trae—; y, al final,
  la sentencia nueva que la responsable dictó para cumplirla, con su fecha y
  lo que resolvió. Sin ese hilo no se entiende qué quedó vinculado por la
  ejecutoria y qué se resolvió con libertad de jurisdicción. Si el documento
  no dice el número del amparo o el tribunal, no lo inventes: di que consta
  en autos.
"""
    return f"""{_rotulo}

Lo que PASÓ en el juicio de origen, en orden cronológico. NO es el resumen de
lo que la responsable resolvió —eso va aparte y después—: aquí sólo se cuenta
el trámite, para que quien lea entienda de dónde viene el asunto.

{_forma}- PRETÉRITO y verbos de TRÁMITE: {', '.join(_v_tramite[:6])}.
- {_como_empiezan}:
  {'; '.join(f'«{a}…»' for a in _arranques[:5])}.
- CADA FECHA EN LETRA, como en todo documento judicial.
- Los puntos resolutivos de las sentencias de origen se TRANSCRIBEN entre
  comillas cuando importan al asunto.
- Y EL ÚLTIMO PÁRRAFO DICE EN QUÉ PARÓ. Si el asunto viene de un juicio ya
  resuelto —una revisión, una queja contra la sentencia—, los antecedentes se
  cierran diciendo QUÉ RESOLVIÓ el órgano de origen, con su verbo:
  el verbo que corresponde a ESTE tipo de asunto —{_verbos}— y el precepto en
  que se apoyó.

  NO LE ATRIBUYAS UN DESENLACE QUE NO ES SUYO. En amparo directo la autoridad
  responsable es {_quien}: resuelve el juicio de
  origen, NO resuelve amparos, así que no puede conceder ni negar el amparo.
  Quien concede o niega es el Tribunal Colegiado, y eso va en el resolutivo de
  esta sentencia, no en los antecedentes.

  NO ES OPINAR NI ADELANTAR EL ESTUDIO: es el último hecho procesal de la
  cadena, y sin él los antecedentes cuentan cómo empezó todo y no cómo acabó.
  Medido: en un proyecto real los antecedentes narraron siete autos del juicio
  de nulidad y nunca dijeron que el Juzgado había sobreseído, así que el
  resolutivo del recurso salió con un hueco donde debía ir el verbo.
{_extra}- NO opines, NO califiques y NO adelantes el estudio."""
