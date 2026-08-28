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


def instrucciones_resumen_acto() -> str:
    return f"""RESUMEN DEL ACTO RECLAMADO O SENTENCIA RECURRIDA

Abre el estudio con esto. Cuenta qué resolvió la autoridad y con qué razones,
de modo que quien lea entienda la resolución impugnada sin tenerla enfrente.

- TIEMPO VERBAL: PRETÉRITO, sin excepción. {', '.join(VERBOS_RESPONSABLE[:6])}.
  Lo que la responsable hizo ya ocurrió y consta en autos.
- SUJETO: {', '.join(SUJETOS_RESPONSABLE[:4])}. Nunca su nombre propio.
- NO LA CALIFIQUES TODAVÍA. Aquí sólo se reconstruye su razonamiento con
  fidelidad; el juicio viene después, en el estudio.
- CADA AFIRMACIÓN ANCLADA a la página del documento de origen, para que el
  secretario coteje sin releer.
- EXTENSIÓN: alrededor de {PALABRAS_RESUMEN_ACTO} palabras, que es la mediana
  medida en los engroses reales."""


def instrucciones_resumen_conceptos(es_recurso: bool = False) -> str:
    q = "agravios" if es_recurso else "conceptos de violación"
    return f"""RESUMEN DE LOS {q.upper()}

Va inmediatamente después del resumen del acto reclamado.

- TIEMPO VERBAL: PRESENTE, sin excepción. {', '.join(VERBOS_PARTE[:6])}.
  Lo que la parte reclama se está diciendo ahora ante el tribunal.
- SUJETO: {', '.join(SUJETOS_PARTE[:3])}.
- UN PÁRRAFO POR {q[:-1].upper()}, en el orden en que se plantearon, sin
  fundirlos ni reordenarlos.
- NO LOS CALIFIQUES. Aquí sólo se expone lo que se alega.
- EXTENSIÓN: alrededor de {PALABRAS_RESUMEN_CONCEPTOS} palabras."""


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
- Señala, cuando lo adviertas, el IMPEDIMENTO TÉCNICO que llevaría a
  inoperancia: que el planteamiento no combata la razón toral, que sea
  novedoso, que verse sobre cuestión firme, o que no haya concepto.
- No propongas el sentido. El sentido lo pone el secretario."""
    if global_primero:
        base += """
- FORMULA PRIMERO UN PROBLEMA GLOBAL: la cuestión toral de la que dependen
  las demás. Si el asunto se resuelve con ella, el estudio se ordena alrededor
  de esa sola pregunta y se gana claridad. Sólo desglosa en problemas
  particulares cuando de verdad sean independientes entre sí."""
    return base
