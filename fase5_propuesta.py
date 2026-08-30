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


def _bloque_normas(material, limite: int = 10) -> str:
    fuera = []
    for n in list(getattr(material, "normas", []) or [])[:limite]:
        fuera.append(f"· {n.get('fuente','')} art. {n.get('articulo','')}: "
                     f"{(n.get('texto','') or '')[:400]}")
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
3. LA LEY QUE RIGE ES LA DEL ESTADO DE QUERÉTARO. No propongas aplicar la ley
   de otra entidad. La jurisprudencia que interpreta legislación de otra
   entidad SÍ vale, y se invoca por el principio que fija, sin excusarse.
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
    kw = dict(model=MODELO_PROPUESTA,
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
