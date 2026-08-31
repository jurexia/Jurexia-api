"""Fase 5 — LA PROPUESTA DE SOLUCIÓN.

Entre consultar el acervo y dictar el criterio faltaba un escalón. Hasta ahora
el secretario veía 17 tesis y 20 normas y tenía que decidir el sentido con eso
delante; si no dictaba criterio, el pipeline seguía igual y el proyecto salía
con la calificación de la plantilla. Así nació la incongruencia del ADC
380/2025: consideraciones para conceder, efectos redactados, y un resolutivo
que negaba.

Esto propone. NO decide. David, 30-ago-2026: «proponiendo declarar fundados,
infundados, inoperantes los agravios o conceptos de violación por determinadas
razones que pudieran resumírsele al secretario. O bien si lo desea él
introducir el criterio».

TRES REGLAS QUE LA HACEN ÚTIL EN VEZ DE DECORATIVA:

1. UNA PROPUESTA SIN APOYO EN EL ACERVO NO ES UNA PROPUESTA. Cada sentido
   propuesto cita los registros en que se apoya. Si el material no da para
   sostener ninguno, se dice —«no alcanza»— y no se propone: un sentido
   inventado con aire de fundado es peor que ninguno, porque se firma.

2. SE PROPONE POR PROBLEMA, NO POR SENTENCIA. Un asunto puede tener un
   concepto fundado y dos inoperantes, y esa mezcla es la que determina el
   resolutivo y sus efectos.

3. EL MODELO ES INTERCAMBIABLE. `MODELO_PROPUESTA` se lee del entorno. La
   pregunta de David —«¿cómo asegurar un buen criterio? ¿subiendo a un modelo
   más inteligente?»— no se contesta opinando: se contesta midiendo la
   propuesta contra los engroses en que él ya resolvió. Por eso el modelo se
   cambia sin tocar código.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

# El mismo motor que el estudio, salvo que se diga otra cosa. Se separa para
# poder subirlo sólo aquí: proponer el sentido es la decisión más cara de
# equivocar y la más barata de calcular —son unos cientos de palabras—.
MODELO_PROPUESTA = os.getenv("MODELO_PROPUESTA",
                             os.getenv("MODELO_ESTUDIO", "gpt-5.6-luna"))
ESFUERZO_PROPUESTA = os.getenv("ESFUERZO_PROPUESTA", "high")

# Cuántas tesis se le enseñan por problema. Más no ayuda: con el acervo entero
# delante el modelo elige la que suena, no la que aplica.
# EL RAZONAMIENTO CONSUME DEL MISMO PRESUPUESTO QUE LA RESPUESTA. La primera
# versión pedía 4,000 tokens con esfuerzo alto y volvió VACÍA: el modelo gastó
# el presupuesto pensando y no le quedó para escribir el JSON. La salida útil
# son doscientas palabras, pero el sitio para pensarlas hay que dárselo.
MAX_TOKENS_PROPUESTA = int(os.getenv("MAX_TOKENS_PROPUESTA", "16000"))

MAX_TESIS_PROPUESTA = 8
TESIS_CARACTERES = 1200

SENTIDOS = ("fundado", "infundado", "inoperante", "ineficaz")

# Lo que el secretario lee de un vistazo. Medido sobre sus propios estudios: la
# razón que él escribe para calificar cabe en tres o cuatro renglones antes de
# desarrollarse. Más largo aquí no se lee y se acaba ignorando la propuesta.
PALABRAS_RAZON = 60


@dataclass
class Propuesta:
    """Lo que el motor sugiere para UN problema jurídico."""
    problema: str
    sentido: str = ""                 # fundado | infundado | inoperante | ineficaz
    razon: str = ""                   # el porqué, en tres o cuatro renglones
    apoyos: list = field(default_factory=list)   # registros del acervo
    confianza: str = ""               # alta | media | baja
    alcanza: bool = True              # False = el material no da para proponer

    def bloque(self) -> str:
        if not self.alcanza:
            return (f"· {self.problema[:120]}\n"
                    f"    SIN PROPUESTA — el acervo no alcanza para sostener un "
                    f"sentido. {self.razon}")
        ap = ", ".join(str(a) for a in self.apoyos) or "sin apoyo"
        return (f"· {self.problema[:120]}\n"
                f"    PROPUESTA: {self.sentido.upper()} ({self.confianza})\n"
                f"    {self.razon}\n"
                f"    Se apoya en: {ap}")


def _tesis_del_material(material, limite: int = MAX_TESIS_PROPUESTA) -> list:
    """Las tesis que se le enseñan, con la obligatoria delante."""
    tesis = list(getattr(material, "tesis", []) or [])
    tesis.sort(key=lambda t: not t.get("obligatoria"))
    return tesis[:limite]


def _bloque_tesis(tesis: list) -> str:
    fuera = []
    for t in tesis:
        fuera.append(
            f"[registro {t.get('registro','')}] "
            f"{'OBLIGATORIA' if t.get('obligatoria') else 'orientadora'} · "
            f"{t.get('instancia','')}\n"
            f"  {t.get('rubro','')}\n"
            f"  {(t.get('texto','') or '')[:TESIS_CARACTERES]}")
    return "\n\n".join(fuera)


# EL PRECEPTO ENTERO, Y CON EL NOMBRE DE SU LEY. Dos fallos de una línea cada
# uno, y entre los dos decidieron un asunto.
#
# En el ADL 382/2024 el motor propuso INFUNDADO cinco veces razonando sobre si
# las incapacidades se habían entregado a tiempo, y no discutió ni una vez lo
# único que decide: que el artículo 47, fracción X, de la Ley Federal del
# Trabajo exige que las faltas sean «SIN CAUSA JUSTIFICADA». Fui a ver por qué y
# no era el razonamiento: NUNCA LO LEYÓ. Ese artículo mide 4,351 caracteres, la
# fracción X empieza en el 2,348 y aquí se recortaba en el 400. El modelo veía
# el encabezado y la fracción I.
#
# Y el nombre de la ley salía vacío en las diez normas, siempre, porque se leía
# de `fuente` y el material las trae en `cuerpo_legal`. El motor recibía
# «· art. 47: …» sin saber de qué ley, en un asunto donde el artículo 47 existe
# en sesenta y dos versiones distintas del acervo federal.
#
# Un precepto recortado antes de su fracción operativa no es una premisa: es un
# encabezado. Y el coste de traerlo entero son unos miles de caracteres en un
# prompt que ya pasa de treinta mil.
NORMA_CARACTERES = 4000


def _bloque_normas(material, limite: int = 10) -> str:
    fuera = []
    for n in list(getattr(material, "normas", []) or [])[:limite]:
        ley = n.get("cuerpo_legal") or n.get("fuente") or ""
        fuera.append(f"· {ley} — artículo {n.get('articulo','')}: "
                     f"{(n.get('texto','') or '')[:NORMA_CARACTERES]}")
    return "\n".join(fuera)


def _bloque_contexto(contexto: str) -> str:
    """Lo que el secretario aportó porque el acervo no lo tenía.

    El motor dice qué le falta —«falta el texto contractual y el resultado del
    cotejo», «el acervo no contiene la cláusula 64»— y hasta ahora eso era un
    callejón sin salida: el secretario leía el diagnóstico y no podía hacer
    nada con él. Ahora sube el contrato, el convenio o el acta y el motor
    propone con eso delante.
    """
    c = (contexto or "").strip()
    if not c:
        return ""
    return f"""

═══════════════════════════════════════════════════════════════════════
DOCUMENTO APORTADO POR EL SECRETARIO
═══════════════════════════════════════════════════════════════════════
Esto NO estaba en el acervo: lo aporta quien tiene el expediente delante
porque tú dijiste que te faltaba. Vale como material: cítalo por lo que dice,
identificándolo como el documento aportado, y NO lo confundas con
jurisprudencia ni le inventes un registro.

{c[:20000]}
"""



# ═══════════════════════════════════════════════════════════════════════════
# CÓMO RESOLVIÓ EL ACERVO — en el sitio donde de verdad se decide
# ═══════════════════════════════════════════════════════════════════════════
# Yair, 31-ago-2026: el motor le propuso INFUNDADO en un asunto donde los
# precedentes se inclinan a fundado. Tenía razón y el fallo era de diseño mío:
# construí el sondeo del acervo para que «se le enseñe al redactor ANTES de que
# fije el sentido» y luego se lo enseñé al que ESCRIBE el estudio, que corre
# cuando el sentido ya está fijado. El que propone —éste— decidía a ciegas.
#
# El sondeo viaja dentro del `material`, así que no hace falta cambiar ninguna
# firma: estaba aquí desde el principio, sin que nadie lo mirara.
#
# LAS DOS ESCALAS NO SON LA MISMA, y hay que traducir: el acervo clasifica
# SENTENCIAS —concede, niega, confirma— y aquí se califican CONCEPTOS —fundado,
# infundado, inoperante—. Que el 70% de los precedentes conceda no significa que
# todos los conceptos sean fundados: significa que al menos uno lo fue.

# LA ENTIDAD NO SE ESCRIBE A MANO. Este prompt decía literalmente «LA LEY QUE
# RIGE ES LA DEL ESTADO DE QUERÉTARO», y esto lo usan secretarios de toda la
# república: a uno de Yucatán se le estaba ordenando aplicar el código de otro
# estado, y en un amparo laboral federal la afirmación es sencillamente falsa.
# El prompt del estudio ya lo decía bien —«el código que rige es el de la
# entidad»— y éste se había quedado atrás.
def _regla_de_ley(material) -> str:
    ent = str(getattr(material, "entidad", "") or "").strip()
    mat = str(getattr(material, "materia", "") or "").strip().lower()
    if mat == "laboral":
        return ("LA LEY QUE RIGE ES LA FEDERAL DEL TRABAJO, no la de ninguna "
                "entidad. Si el asunto es de un trabajador al servicio del "
                "Estado, comprueba en el material cuál de las dos leyes le "
                "aplica antes de invocarla: confundirlas cambia el resultado.")
    if mat == "penal":
        return ("LA LEY QUE RIGE es el Código Nacional de Procedimientos "
                "Penales y el código penal que corresponda al fuero del "
                "asunto. No mezcles fuero común y federal.")
    if ent:
        return (f"LA LEY QUE RIGE ES LA DEL ESTADO DE {ent.upper()}. No "
                f"propongas aplicar la ley de otra entidad. La jurisprudencia "
                f"que interpreta legislación de otra entidad SÍ vale, y se "
                f"invoca por el principio que fija, sin excusarse.")
    return ("LA LEY QUE RIGE ES LA DE LA ENTIDAD DEL ASUNTO, y es la que está "
            "en el material. No propongas aplicar la ley de otra entidad. La "
            "jurisprudencia que interpreta legislación de otra entidad SÍ "
            "vale, y se invoca por el principio que fija, sin excusarse.")


# ═══════════════════════════════════════════════════════════════════════════
# LA SUPLENCIA, EN LA FASE QUE DECIDE
# ═══════════════════════════════════════════════════════════════════════════
# El proyecto 382/2024 —un trabajador despedido por el IMSS— salió declarando
# inoperantes conceptos del obrero. El prompt del ESTUDIO ya prohibía eso desde
# hace semanas; lo que faltaba era que lo supiera la fase que fija el sentido.
# Esta. Mientras la propuesta no conozca el artículo 79, fracción V, el estudio
# recibe un «inoperante» ya decidido y lo único que puede hacer es escribirlo.
#
# Y OJO CON EL MATIZ, que es donde se equivocan los dos extremos: la suplencia
# NO obliga a dar la razón al trabajador, y tampoco borra la inoperancia del
# catálogo. Cura la DEFICIENCIA del argumento, no su falta de pertinencia. Un
# concepto mal expuesto se suple y se estudia; un concepto que, ya suplido y
# entendido en su mejor versión, sigue sin atacar la razón que sostiene el
# laudo, puede declararse inoperante y hay que decir por qué.

_SUPLENCIA = {
    "laboral": ("el TRABAJADOR", "artículo 79, fracción V, de la Ley de Amparo"),
    "penal": ("el REO o el imputado", "artículo 79, fracción III, de la Ley de Amparo"),
}


def _bloque_suplencia(material) -> str:
    mat = str(getattr(material, "materia", "") or "").strip().lower()
    if mat not in _SUPLENCIA:
        return ""
    quien, precepto = _SUPLENCIA[mat]
    return f"""
═══════════════════════════════════════════════════════════════════════
ANTES DE CALIFICAR NADA: LA SUPLENCIA DE LA QUEJA
═══════════════════════════════════════════════════════════════════════
Éste es un asunto de materia {mat}. Si quien promueve es {quien}, la suplencia
de la deficiencia de la queja prevista en el {precepto} es ABSOLUTA: opera aun
ante la AUSENCIA TOTAL de conceptos de violación, y obliga al Tribunal a
examinar el expediente y a reparar la violación que encuentre.

QUÉ SIGNIFICA PARA TU CALIFICACIÓN:

· NO PUEDES proponer INOPERANTE por deficiencia de la impugnación. «No precisó
  qué prueba se omitió», «no combatió la razón toral», «no expresó argumento
  contra tal consideración»: eso es exactamente lo que la suplencia repara. Un
  argumento mal expuesto se SUPLE, se reconstruye en su mejor versión y se
  estudia en el fondo.
· TAMPOCO significa dar la razón. Suplir es examinar, no conceder. Si el
  planteamiento, ya suplido y entendido en su mejor versión, es contrario a
  derecho, es INFUNDADO y se dice por qué.
· La inoperancia sólo queda para lo que la suplencia no cura: un argumento que,
  ya reconstruido, no se dirige contra ninguna razón del acto. Si propones
  inoperante, escribe qué versión suplida examinaste y por qué ni siquiera así
  toca el fallo.
· Y ANTES DE ESO, MIRA EL EXPEDIENTE. La suplencia obliga a buscar la violación
  aunque nadie la haya alegado: si en lo que tienes delante aparece un vicio que
  beneficia a {quien} y no está en ningún concepto, propónlo igual y dilo.

Si por la suplencia te apartas de lo que literalmente pidió la parte, no es un
exceso: es el mandato del precepto.
"""


def _bloque_acervo_sentidos(material) -> str:
    s = getattr(material, "sondeo", None)
    if s is None or not getattr(s, "distribucion", None):
        return ""
    total = sum(s.distribucion.values())
    if total < 5:
        return ""
    favorables = sum(n for k, n in s.distribucion.items()
                     if k in ("concede", "parcialmente_concede", "ampara", "revoca"))
    L = ["", "═" * 71,
         "CÓMO RESOLVIERON OTROS COLEGIADOS ESTE MISMO PROBLEMA",
         "═" * 71,
         f"Se buscaron en el acervo las sentencias sobre este tema. De {total}:"]
    for k, n in sorted(s.distribucion.items(), key=lambda x: -x[1])[:6]:
        L.append(f"   · {k}: {n}  ({100*n//total}%)")
    L += ["",
          f"Es decir: el {100*favorables//total}% dio la razón —total o "
          f"parcialmente— a quien promovió.",
          "",
          "OJO CON LA ESCALA, QUE NO ES LA MISMA. El acervo clasifica SENTENCIAS",
          "y tú calificas CONCEPTOS. Que la mayoría conceda no vuelve fundados",
          "todos los conceptos: significa que al menos UNO lo fue. Y al revés,",
          "que la mayoría niegue no obliga a declararlos todos infundados.",
          "",
          "PERO SÍ TE OBLIGA A ESTO: si vas a proponer que NINGÚN concepto es",
          "fundado en un tema donde la mayoría de los tribunales concede —o al",
          "contrario—, escribe en tu razón por qué este caso no cae en esa",
          "corriente. Apartarse es legítimo; apartarse sin enterarse, no.",
          ""]
    if s.fundamentos:
        L.append("LOS FUNDAMENTOS QUE SE REPITEN en las sentencias del tema:")
        for f in s.fundamentos[:6]:
            L.append(f"   · {f['fundamento']}  ({f['veces']})")
        L.append("")
    if getattr(s, "concordantes", None):
        L += ["Y ASÍ RAZONARON LOS MÁS CERCANOS AL TUYO —no son fuente que",
              "obligue: un colegiado no obliga a otro. Son cómo se ha resuelto:", ""]
        for c in s.concordantes[:4]:
            L.append(f"   [{c.get('sentido')}] {str(c.get('holding') or '')[:420]}")
            L.append("")
    return "\n".join(L)


def prompt_propuesta(problemas: list, material, resumen_acto: str,
                     resumen_conceptos: str, es_recurso: bool = False,
                     contexto: str = "") -> str:
    q = "agravios" if es_recurso else "conceptos de violación"
    tesis = _tesis_del_material(material)
    lista = "\n".join(
        f"{i}. {p.get('pregunta','') if isinstance(p, dict) else str(p)}"
        + (f"\n   La responsable resolvió: {p.get('resolvio','')}"
           if isinstance(p, dict) and p.get("resolvio") else "")
        + (f"\n   Se combate diciendo: {p.get('combate','')}"
           if isinstance(p, dict) and p.get("combate") else "")
        for i, p in enumerate(problemas, 1))

    return f"""Eres el secretario de un Tribunal Colegiado preparando la propuesta
de solución de un amparo. NO escribes la sentencia: propones cómo debe
calificarse cada uno de los {q} y por qué, para que quien firma lo apruebe,
lo corrija o lo sustituya por su criterio.

LOS PROBLEMAS JURÍDICOS DEL ASUNTO
{lista}

LO QUE RESOLVIÓ LA RESPONSABLE
{resumen_acto[:3000]}

LO QUE SE COMBATE
{resumen_conceptos[:3000]}

JURISPRUDENCIA DEL ACERVO — es TODO lo que puedes invocar
{_bloque_tesis(tesis)}

NORMAS DEL ACERVO
{_bloque_normas(material)}
{_bloque_suplencia(material)}
{_bloque_acervo_sentidos(material)}
{_bloque_contexto(contexto)}

CÓMO SE CALIFICA, y no son sinónimos:
- FUNDADO: el planteamiento combate la razón de la responsable y tiene razón.
- INFUNDADO: la combate y no tiene razón.
- INOPERANTE: NO combate la razón toral —ataca algo que no sostiene el fallo,
  repite lo dicho en la instancia, o parte de una premisa falsa—. La
  inoperancia se razona: hay que decir POR QUÉ no combate.
- INEFICAZ: se dirige contra consideraciones que ya no rigen el sentido.

REGLAS QUE NO SE ROMPEN:
1. SÓLO TE APOYAS EN LOS REGISTROS DE ARRIBA. No cites de memoria: tus datos
   son viejos y falsos, y una cita inventada descalifica el proyecto entero.
2. SI EL ACERVO NO DA PARA SOSTENER UN SENTIDO, DILO. Pon alcanza=false y
   explica qué falta. Un sentido inventado con aire de fundado se firma, y ese
   es el daño que este paso existe para evitar.
3. {_regla_de_ley(material)}
4. NO SUPONGAS LO QUE NO CONSTA. Si el material no permite afirmar un hecho,
   di que no está acreditado; no escribas «si fuera cierto que…».
5. LA RAZÓN, EN {PALABRAS_RAZON} PALABRAS. Es lo que el secretario lee antes de
   decidir: tiene que caber en tres o cuatro renglones y decir la razón toral,
   no el desarrollo.

Devuelve SÓLO un JSON, sin texto alrededor, con esta forma exacta:
{{"propuestas": [
  {{"problema": "<la pregunta, tal cual>",
    "sentido": "fundado|infundado|inoperante|ineficaz",
    "razon": "<la razón toral, {PALABRAS_RAZON} palabras>",
    "apoyos": ["<registro>", "..."],
    "confianza": "alta|media|baja",
    "alcanza": true}}
]}}"""


_RX_JSON = re.compile(r"\{.*\}", re.S)


def _leer(crudo: str) -> list:
    """El JSON del modelo, tolerante a que lo envuelva en explicaciones."""
    m = _RX_JSON.search(crudo or "")
    if not m:
        return []
    try:
        datos = json.loads(m.group(0))
    except Exception:
        return []
    return datos.get("propuestas") or []


# El modelo escribe el apoyo como lo diría una sentencia —«registro 2007719»,
# «art. 296 del Código Civil del Estado de Querétaro»— y comparar esa cadena
# contra los registros pelados cantaba invención donde no la había. Se compara
# la CIFRA, y lo que no trae cifra de registro es una norma: se deja pasar,
# porque una norma también es apoyo legítimo.
_RX_CIFRA_REGISTRO = re.compile(r"\b(\d{6,7})\b")


def revisar(propuestas: list, material) -> list:
    """Lo comprobable sin modelo. Ninguna de estas es opinión."""
    avisos = []
    validos = {str(t.get("registro", "")) for t in getattr(material, "tesis", []) or []}
    for p in propuestas:
        if not p.alcanza:
            continue
        if p.sentido not in SENTIDOS:
            avisos.append(f"Sentido no reconocido: «{p.sentido}».")
        inventados = []
        for a in p.apoyos:
            m = _RX_CIFRA_REGISTRO.search(str(a))
            if m and m.group(1) not in validos:
                inventados.append(str(a))
        if inventados:
            avisos.append(
                f"La propuesta se apoya en registros que NO están en el acervo: "
                f"{inventados}. No se citan hasta comprobarlos en el Semanario.")
        if not p.apoyos:
            avisos.append(
                f"«{p.sentido}» se propone SIN APOYO del acervo. Una propuesta "
                f"sin fundamento es una opinión: compruébala antes de aceptarla.")
    return avisos


async def proponer(cliente, problemas: list, material, resumen_acto: str = "",
                   resumen_conceptos: str = "", es_recurso: bool = False,
                   contexto: str = "") -> tuple[list, list]:
    """Devuelve (propuestas, avisos). No decide nada: propone."""
    if not problemas:
        return [], []
    # EL SENTIDO NO SE SORTEA. Medido: con material fijo esta llamada ya daba
    # el mismo resultado cinco de cinco veces, así que esto no arregla nada
    # hoy; lo que hace es impedir que mañana empiece a variar por un cambio de
    # modelo o de proveedor. La decisión de un tribunal no puede depender del
    # muestreo.
    kw = dict(model=MODELO_PROPUESTA,
              temperature=0, seed=20260831,
              max_completion_tokens=MAX_TOKENS_PROPUESTA,
              messages=[{"role": "user", "content": prompt_propuesta(
                  problemas, material, resumen_acto, resumen_conceptos,
                  es_recurso, contexto)}])
    if ESFUERZO_PROPUESTA:
        kw["reasoning_effort"] = ESFUERZO_PROPUESTA
    r = await cliente.chat.completions.create(**kw)
    crudo = (r.choices[0].message.content or "").strip()

    # SI NO VUELVE NADA, HAY QUE PODER SABER POR QUÉ. Una lista vacía puede
    # ser «el modelo no respondió» o «respondió algo que no supe leer», y son
    # dos averías distintas. Se distinguen aquí y no adivinando en los logs.
    leidas = _leer(crudo)
    if not leidas:
        motivo = ("el modelo no devolvió texto —probablemente agotó el "
                  "presupuesto razonando—" if not crudo.strip()
                  else f"la respuesta no traía el JSON esperado: «{crudo[:200]}»")
        print(f"   ⚖️ PROPUESTA sin resultado ({MODELO_PROPUESTA}): {motivo}")
        return [], [f"El motor no propuso ningún sentido: {motivo}. "
                    f"Dicta tu criterio con la mecánica de siempre."]

    fuera = []
    for d in leidas:
        fuera.append(Propuesta(
            problema=str(d.get("problema", ""))[:400],
            sentido=str(d.get("sentido", "")).strip().lower(),
            razon=str(d.get("razon", ""))[:900],
            apoyos=[str(a) for a in (d.get("apoyos") or [])][:6],
            confianza=str(d.get("confianza", "")).strip().lower(),
            alcanza=bool(d.get("alcanza", True))))

    # Si el modelo devolvió menos propuestas que problemas, faltan: se dice.
    avisos = revisar(fuera, material)
    if len(fuera) < len(problemas):
        avisos.append(
            f"Se propusieron {len(fuera)} sentidos para {len(problemas)} "
            f"problemas. Los que faltan quedan sin propuesta.")
    return fuera, avisos


def resumen(propuestas: list) -> str:
    """Lo que se le enseña al secretario, de un vistazo."""
    if not propuestas:
        return "Sin propuesta: el acervo no alcanzó para sugerir un sentido."
    return "\n\n".join(p.bloque() for p in propuestas)


def calificaciones_de(propuestas: list) -> list:
    """Los sentidos, en orden, para el resolutivo. Sólo los que alcanzan."""
    return [p.sentido for p in propuestas if p.alcanza and p.sentido in SENTIDOS]
