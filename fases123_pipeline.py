"""FASES 1-3 — de los PDF a los problemas jurídicos.

Las tres fases que David dijo que NO necesitan al secretario:

    1. resumen del acto reclamado o sentencia recurrida   (438 palabras, PASADO)
    2. resumen de los conceptos de violación o agravios   (472 palabras, PRESENTE)
    3. los problemas jurídicos, del contraste de los dos

La especificación de estilo está medida en `fases123_resumenes.py` sobre 40
estudios firmados. Aquí se ejecuta.

LECTURA DE LOS PDF: se reutiliza `_extract_text_from_upload` de main.py, que ya
hace lo correcto y barato — extracción nativa con PyMuPDF, gratis, y OCR con
Gemini SÓLO si el PDF viene escaneado. No se paga OCR de lo que ya trae texto.

EL RECORTE, que es donde se va el dinero: una sentencia de treinta páginas no
cabe entera en el prompt sin costar una fortuna, y tampoco hace falta. Del acto
reclamado interesan las CONSIDERACIONES y los RESOLUTIVOS —no el proemio ni la
relatoría de constancias—, y de los conceptos interesa el apartado de conceptos.
`recortar_acto` y `recortar_conceptos` buscan esas marcas y, si no las
encuentran, se quedan con la cola del documento, que es donde vive el
razonamiento.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

from fases123_resumenes import (
    PALABRAS_ANTECEDENTES,
    PALABRAS_RESUMEN_ACTO,
    PALABRAS_RESUMEN_CONCEPTOS,
    instrucciones_antecedentes,
    instrucciones_problemas,
    instrucciones_resumen_acto,
    instrucciones_resumen_conceptos,
)

# ═══════════════════════════════════════════════════════════════════════════
# Recorte — dónde empieza lo que importa
# ═══════════════════════════════════════════════════════════════════════════

# Se busca el ESTUDIO DE FONDO, no el primer «CONSIDERANDO».
#
# Arrancar en el considerando primero mete la COMPETENCIA en el resumen, y el
# resultado narra el trámite en vez de la ratio: probado, la primera versión
# empezaba «declaró su competencia para resolver el recurso…», que es
# exactamente lo que a nadie le importa del acto reclamado.
_MARCAS_ESTUDIO = re.compile(
    r"^\s*(?:PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|S[ÉE]PTIMO)\.\s*"
    r"(?:Estudio|An[áa]lisis|Fondo|Consideraciones de fondo)|"
    r"ESTUDIO\s+DE\s+FONDO|es\s+fundado\s+el\s+agravio|"
    r"son\s+(?:fundados|infundados|inoperantes)",
    re.I | re.M,
)
_MARCAS_CONSIDERANDO = re.compile(
    r"C\s?O\s?N\s?S\s?I\s?D\s?E\s?R\s?A\s?N\s?D\s?O|CONSIDERACIONES",
    re.I | re.M,
)
_MARCAS_CONCEPTOS = re.compile(
    r"CONCEPTOS?\s+DE\s+VIOLACI[ÓO]N|A\s?G\s?R\s?A\s?V\s?I\s?O\s?S|"
    r"(?:PRIMER|ÚNICO)\s+(?:CONCEPTO|AGRAVIO)",
    re.I,
)

# MEDIDO, NO SUPUESTO. Decía «de sobra para una sentencia larga» y no lo era:
# la sentencia recurrida del ARC 25/2026 son 90,126 caracteres de texto nativo
# —37 folios de un PDF digital, sin OCR de por medio— y se cortaba a 60,000 a
# media frase. El proyecto salió diciendo, DENTRO del considerando quinto:
#
#     «El texto proporcionado se interrumpió antes de que la autoridad
#      responsable desarrollara las razones concretas…»
#
# El modelo no alucinó: describió con exactitud lo que le habíamos hecho. Se
# sube a cubrir lo medido y se corta por párrafo, nunca a mitad de oración.
TOPE_CARACTERES = 100_000         # ~25k tokens; cubre los 90k del ARC 25/2026


# El rótulo del bloque resolutivo, con y sin espaciado judicial.
_RX_RESUELVE = re.compile(
    r"\n\s*R\s*E\s*S\s*U\s*E\s*L\s*V\s*E\s*[:.]?\s*\n|"
    r"\bpor\s+lo\s+(?:expuesto|anteriormente\s+expuesto)[^.]{0,90}se\s+resuelve",
    re.I)


def _cortar_bien(cuerpo: str, tope: int) -> str:
    """Recorta por párrafo, y si hay que sacrificar, sacrifica el MEDIO.

    UN CORTE A MITAD DE FRASE SE NOTA, y lo que el modelo hace al notarlo es
    contarlo. Se corta en el último salto de párrafo antes del tope, y si no
    hay ninguno, en el último punto.

    Y CUANDO NO CABE, NO SE TIRA LA COLA. La versión anterior se quedaba con
    los primeros N caracteres desde el estudio, de modo que en una sentencia
    larga los RESOLUTIVOS —que están al final y son lo que se confirma o se
    revoca— caían fuera. Se conservan las tres cuartas partes de la cabeza y la
    última cuarta parte del final, que es donde vive el «R E S U E L V E».
    """
    if len(cuerpo) <= tope:
        return cuerpo

    def _hasta(x: str, n: int) -> str:
        t = x[:n]
        corte = t.rfind("\n\n")
        if corte < n * 0.5:
            corte = t.rfind(". ")
        return t[:corte + 1] if corte > 0 else t

    # LA COLA DE ESTOS PDF NO SON LOS RESOLUTIVOS: ES EL SELLO. Quedarse con
    # los últimos N caracteres traía «Identificador de la respuesta TSP:
    # 85450738 / Datos estampillados: 5soe2uW/axw52+2VvzE…», que es la cadena
    # de la FIEL. Se BUSCA el bloque resolutivo por su rótulo; si no aparece,
    # no se inventa una cola: se entrega sólo la cabeza, bien cortada.
    m = _RX_RESUELVE.search(cuerpo)
    if m:
        resolutivos = cuerpo[m.start():][:int(tope * 0.25)]
        cabeza = _hasta(cuerpo[:m.start()], tope - len(resolutivos))
        # Sin marca de omisión en medio: un marcador es una invitación a
        # comentarlo, y comentar el estado del documento DENTRO de la sentencia
        # es justo lo que no puede pasar. El rótulo del resuelve separa solo.
        return cabeza + "\n\n" + resolutivos
    return _hasta(cuerpo, tope)


def recortar_acto(texto: str, tope: int = TOPE_CARACTERES) -> str:
    """Del acto reclamado, su ESTUDIO DE FONDO y los resolutivos.

    Se prefiere la marca del estudio; sólo si no aparece se cae al primer
    considerando, y en último caso a la cola del documento.
    """
    m = _MARCAS_ESTUDIO.search(texto) or _MARCAS_CONSIDERANDO.search(texto)
    cuerpo = texto[m.start():] if m else texto[-tope:]
    return _cortar_bien(cuerpo, tope)


def recortar_conceptos(texto: str, tope: int = TOPE_CARACTERES) -> str:
    """Del escrito de la parte, el apartado de conceptos o agravios."""
    m = _MARCAS_CONCEPTOS.search(texto)
    cuerpo = texto[m.start():] if m else texto[-tope:]
    return _cortar_bien(cuerpo, tope)


# ═══════════════════════════════════════════════════════════════════════════
# Los prompts
# ═══════════════════════════════════════════════════════════════════════════

_NUCLEO = """Eres el secretario de un Tribunal Colegiado de Circuito preparando
el adelanto de una sentencia. Escribes en el registro judicial mexicano.

REGLAS QUE NO SE NEGOCIAN:
- NO INVENTES NADA. Si un dato no está en el documento, no existe. Cero
  tolerancia: un hecho inventado en una sentencia es un desastre, no un error.
- No califiques ni resuelvas. Aquí sólo se expone.
- Prosa corrida, sin viñetas ni esquemas. Frase larga y subordinada: la mediana
  de los engroses reales es de 35 palabras por oración.
- Sin Markdown.
- NUNCA ESCRIBAS SOBRE EL ESTADO DEL DOCUMENTO QUE LEES. Ni «el texto
  proporcionado se interrumpió», ni «no se alcanza a leer», ni «el fragmento
  disponible», ni «el documento parece incompleto», ni «hasta donde se
  transcribe». Quien lee esto es un magistrado leyendo una SENTENCIA, no un
  informe sobre un archivo: una frase así dentro del fallo no se corrige, se
  borra, y deja al secretario preguntándose qué más se inventó.
  Si algo no consta, simplemente no lo digas y sigue con lo que sí consta. Lo
  que falte se avisa por otro camino, fuera del documento."""


def prompt_resumen_acto(texto_acto: str, es_recurso: bool = False,
                        tipo_asunto: str = "") -> str:
    # «la sentencia reclamada» / «recurrida» / «el auto recurrido» / «la
    # sentencia impugnada»: cuatro, y el booleano sólo distinguía dos.
    import tipos_asunto as _tap
    _t = tipo_asunto or ("amparo_revision" if es_recurso else "amparo_directo")
    que = _tap.vocabulario_de(_t)["recurrido"]
    # EL EJEMPLO NOMBRA AL ÓRGANO, y un ejemplo se copia entero: decía «La Sala
    # consideró… contrario a lo resuelto por la jueza» en los cuatro tipos, así
    # que en una queja el modelo aprendía a llamar Sala al Juzgado de Distrito
    # antes de leer una sola instrucción. Quinta vez en este proyecto.
    # `.capitalize()` no: baja el resto y deja «La sala», «El juzgado de
    # distrito». Sólo la inicial.
    _ej_org = _tap.sujetos_de(_t)["organo"][0]
    _ej_org = _ej_org[:1].upper() + _ej_org[1:]
    return f"""{_NUCLEO}

{instrucciones_resumen_acto(_t)}

LO QUE NO VA EN ESTE RESUMEN, y es donde se equivoca siempre quien lo hace por
primera vez:
- NADA de competencia, personalidad, oportunidad ni trámite. Eso ya está en
  otros considerandos y aquí sobra.
- NADA de crónica cronológica del procedimiento.
- Se entra DIRECTO a lo que la autoridad decidió sobre el fondo y por qué.

UNA DECISIÓN POR FRASE. Así escribe el secretario: «{_ej_org} consideró fundado
el agravio respecto a la carga de la prueba. Determinó que, contrario a lo
resuelto por el inferior, cuando una mujer argumenta que se dedicó al hogar,
existe una presunción de que necesita alimentos.» Dos frases, dos decisiones.
No una sola oración de doscientas palabras encadenando gerundios.

Se trata de la {que}. Éste es su texto:

──────────────────────────────────────────
{recortar_acto(texto_acto)}
──────────────────────────────────────────

Escribe el resumen. Sólo el resumen, sin preámbulo ni rótulo."""


def prompt_resumen_conceptos(texto_conceptos: str, es_recurso: bool = False,
                             tipo_asunto: str = "") -> str:
    import tipos_asunto as _tap
    _t = tipo_asunto or ("amparo_revision" if es_recurso else "amparo_directo")
    q = _tap.vocabulario_de(_t)["combate"]
    return f"""{_NUCLEO}

{instrucciones_resumen_conceptos(es_recurso, _t)}

Éste es el escrito de la parte:

──────────────────────────────────────────
{recortar_conceptos(texto_conceptos)}
──────────────────────────────────────────

Escribe el resumen de los {q}, un párrafo por cada uno. Sólo el resumen."""


def prompt_antecedentes(texto_acto: str) -> str:
    """Los antecedentes se leen del documento ENTERO, no del recorte.

    El recorte del resumen se queda con el estudio de fondo, y ahí no está el
    trámite: la presentación, la admisión y el emplazamiento viven al principio
    del documento, en la parte que el otro recorte descarta.
    """
    return f"""{_NUCLEO}

{instrucciones_antecedentes()}

Éste es el documento:

──────────────────────────────────────────
{_cortar_bien(texto_acto, TOPE_CARACTERES)}
──────────────────────────────────────────

Escribe el apartado de antecedentes, un párrafo por línea. Sólo el apartado."""


def prompt_problemas(resumen_acto: str, resumen_conceptos: str,
                     es_recurso: bool = False, tipo_asunto: str = "") -> str:
    # EL RÓTULO ENSEÑA EN MAYÚSCULAS cómo llamar al órgano, que es la forma más
    # imitable que hay. Decía «LO QUE RESOLVIÓ LA RESPONSABLE» en los cuatro, y
    # en una queja lo que resolvió fue el Juzgado de Distrito.
    import tipos_asunto as _tap
    _t = tipo_asunto or ("amparo_revision" if es_recurso else "amparo_directo")
    _organo = _tap.sujetos_de(_t)["organo"][0].upper()
    return f"""{_NUCLEO}

{instrucciones_problemas(global_primero=True)}

LO QUE RESOLVIÓ {_organo}:
{resumen_acto}

LO QUE SE COMBATE:
{resumen_conceptos}

Devuelve JSON y nada más:
{{
  "problema_global": "la cuestión toral EN FORMA DE PREGUNTA, empezando por ¿ y terminando en ?",
  "problemas": [
    {{"pregunta": "...",
      "resolvio": "qué resolvió la responsable sobre este punto",
      "combate": "qué lo combate",
      "impedimento": null}}
  ]
}}
Si adviertes un impedimento técnico que llevaría a inoperancia, ponlo en
"impedimento" como {{"motivo": "inoperancia", "explicacion": "..."}}."""


# ═══════════════════════════════════════════════════════════════════════════
# El motor
# ═══════════════════════════════════════════════════════════════════════════

# `gpt-5.6-luna` POR LA API DE OPENAI, no por OpenRouter. Decisión de David
# (28-ago-2026): la cuenta de OpenAI ya está pagada y el modelo sale más barato
# que por el intermediario —$0.200/$1.200 por millón frente a $0.375/$1.875 de
# gemini-3.7-flash—, así que no hay razón para dar el rodeo.
#
# Es el mismo motor que ya corre en Redacción Pro y Platinum, con el mismo
# cliente (`chat_client` de main.py). Ver [[motores-iurexia]].
MODELO_FASES = os.getenv("MODELO_FASES", "gpt-5.6-luna")
# Un número cualquiera, pero SIEMPRE el mismo: es lo que hace que dos lecturas
# del mismo expediente den lo mismo.
SEMILLA = int(os.getenv("SEMILLA_FASES", "20260831"))


def _sin_repetidos(problemas: list) -> list:
    """Dos problemas que preguntan lo mismo son uno.

    Medido en las corridas de Kingston: una lectura del ADC 642/2024 sacó CINCO
    problemas y dos eran «¿La Sala responsable vulneró los derechos de
    legalidad, tutela judicial efectiva e impartición de justicia…?» palabra por
    palabra; otra lectura del RQC 233/2025 sacó cuatro y los dos primeros
    preguntaban ambos por el artículo 86 de la Ley de Instituciones de Crédito.
    El motor los resuelve por separado y el estudio contesta dos veces lo mismo:
    engorda el proyecto y en sesión se nota.

    Se comparan por VOCABULARIO, no por cadena: la duplicación real no es
    literal —el modelo reformula— y comparar textos exactos no habría cazado
    ninguno de los dos casos.
    """
    import re as _re
    import unicodedata as _ud

    def _vocab(x: str) -> set:
        y = _ud.normalize("NFKD", str(x or "").lower())
        y = "".join(c for c in y if not _ud.combining(c))
        vacias = {"que", "los", "las", "del", "para", "con", "por", "una", "sus",
                  "responsable", "quejosa", "quejoso", "sala", "debia", "podia"}
        return {w for w in _re.findall(r"[a-z]{4,}", y) if w not in vacias}

    fuera, vistos = [], []
    for p in problemas:
        q = p.get("pregunta", "") if isinstance(p, dict) else str(p)
        v = _vocab(q)
        if len(v) < 4:
            fuera.append(p)
            continue
        if any(len(v & w) / max(1, len(v | w)) > 0.60 for w in vistos):
            continue                       # ya se preguntó esto
        vistos.append(v)
        fuera.append(p)
    return fuera

# SIN RAZONAMIENTO, y lo decidió David: «es un proceso de resumen y recolección
# de información para ser plasmados en el docx». No hay nada que deducir — lo
# que se pide ya está escrito en el documento; hay que encontrarlo y contarlo
# en el registro correcto.
#
# Y no es sólo cuestión de coste: razonando, la fase de problemas devolvió
# respuesta VACÍA en uno de los dos casos de prueba, porque el razonamiento se
# comió el presupuesto de salida. Sin razonar salió a la primera. El
# razonamiento aquí no es que sobre: estorba.
#
# La familia 5.6 acepta none/low/medium/high/xhigh. `max` NO existe.
ESFUERZO_FASES = os.getenv("ESFUERZO_FASES", "none")


async def _pedir(cliente, prompt: str, tope: int = 2500, json_estricto: bool = False) -> str:
    """Una llamada al motor. `json_estricto` obliga al modelo a devolver JSON.

    Sin ese modo, extraer el objeto con un regex falla de vez en cuando —el
    modelo antepone una frase, o parte el objeto— y la fase de problemas se
    queda vacía con un error críptico. Pasó en el ADC 274-2025.
    """
    # ═══════════════════════════════════════════════════════════════════
    # LEER UN EXPEDIENTE NO ES REDACTAR: SE HACE IGUAL TODAS LAS VECES
    # ═══════════════════════════════════════════════════════════════════
    # David pidió estabilidad del sentido, y la medición dijo dónde estaba el
    # problema. La fase que DECIDE es estable: cinco propuestas con el mismo
    # material dieron el mismo sentido las cinco veces. La que varía es ÉSTA,
    # la que lee. Tres lecturas del mismo escrito de agravios dieron:
    #
    #   lectura 1 → 3 problemas
    #   lectura 2 → 3 problemas, redactados distinto
    #   lectura 3 → 4 problemas, dos de ellos casi duplicados
    #
    # Y con problemas distintos se le hacen preguntas distintas al motor, así
    # que el sentido cambia sin que nadie haya cambiado nada. Un secretario que
    # genera el mismo asunto dos veces obtenía resoluciones opuestas.
    #
    # Extraer los antecedentes, los conceptos y los problemas jurídicos de un
    # documento es identificar lo que YA ESTÁ ESCRITO: no admite creatividad, y
    # el muestreo por defecto sólo puede estropearlo. Se fija.
    #
    # La `seed` no garantiza nada por contrato —el proveedor la respeta «en la
    # medida de lo posible»— pero con temperature 0 el margen es mínimo, y no
    # ponerla no ayuda.
    kw = dict(model=MODELO_FASES,
              messages=[{"role": "user", "content": prompt}],
              max_completion_tokens=tope,
              temperature=0, seed=SEMILLA)
    if ESFUERZO_FASES:
        kw["reasoning_effort"] = ESFUERZO_FASES
    if json_estricto:
        kw["response_format"] = {"type": "json_object"}
    r = await cliente.chat.completions.create(**kw)
    txt = (r.choices[0].message.content or "").strip()
    if not txt:
        # RESPUESTA VACÍA: el razonamiento se comió el presupuesto de salida.
        # Es el mismo fallo que main.py ya documenta para los modelos 5.6, y
        # aquí lo delataba un «Expecting value: line 1 column 1» que parecía un
        # problema de JSON y no lo era. Se reintenta con el doble de tope y sin
        # razonamiento: para leer un documento no hace falta.
        kw["max_completion_tokens"] = tope * 2
        kw.pop("reasoning_effort", None)
        r = await cliente.chat.completions.create(**kw)
        txt = (r.choices[0].message.content or "").strip()
    return txt


async def correr(cliente, texto_acto: str, texto_conceptos: str,
                 es_recurso: bool = False,
                 tipo_asunto: str = "") -> "Fases123":
    """Las tres fases, en orden. Los dos resúmenes van EN PARALELO —son
    independientes— y los problemas esperan a los dos, porque salen de su
    contraste."""
    import asyncio
    import json as _json

    an, ra, rc = await asyncio.gather(
        _pedir(cliente, prompt_antecedentes(texto_acto), 3000),
        _pedir(cliente, prompt_resumen_acto(texto_acto, es_recurso, tipo_asunto)),
        _pedir(cliente, prompt_resumen_conceptos(texto_conceptos, es_recurso,
                                                 tipo_asunto)),
    )
    # EL MODELO NO HABLA DEL ARCHIVO DENTRO DE LA SENTENCIA. Aquí, en el
    # embudo por donde pasan los tres textos antes de existir como resumen: si
    # se le quita después, en el compositor, ya ha viajado a los problemas
    # jurídicos y al estudio, que se apoyan en estos resúmenes.
    _quitadas: list = []
    try:
        import meta_lenguaje as _ml
        an, _q1 = _ml.limpiar(an)
        ra, _q2 = _ml.limpiar(ra)
        rc, _q3 = _ml.limpiar(rc)
        _quitadas = _q1 + _q2 + _q3
    except Exception:
        pass
    f = Fases123(antecedentes=an, resumen_acto=ra, resumen_conceptos=rc)
    for _f in _quitadas:
        f.avisos.append(
            f"SE QUITÓ UNA FRASE QUE HABLABA DEL ARCHIVO, NO DEL ASUNTO: "
            f"«{_f[:220]}». Va entera aquí por si el filtro se equivocó y hay "
            f"que devolverla; pero una frase así dentro del fallo deja al que "
            f"firma preguntándose qué más se inventó.")
    try:
        crudo = await _pedir(cliente, prompt_problemas(ra, rc, es_recurso,
                                                       tipo_asunto),
                             3500, json_estricto=True)
        m = re.search(r"\{.*\}", crudo, re.S)
        j = _json.loads(m.group(0) if m else crudo)
        f.problema_global = j.get("problema_global", "")
        f.problemas = _sin_repetidos(j.get("problemas", []) or [])
    except Exception as e:
        f.avisos.append(f"No se pudieron derivar los problemas jurídicos: {e}")
    f.avisos.extend(revisar(f))
    return f


# ═══════════════════════════════════════════════════════════════════════════
# El resultado
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Fases123:
    antecedentes: str = ""
    resumen_acto: str = ""
    resumen_conceptos: str = ""
    problema_global: str = ""
    problemas: list[dict] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)
    # LAS CONSTANCIAS DEL EXPEDIENTE, si el secretario las subió. Van aquí y no
    # en el Resultado porque las fases son lo único que se serializa entero al
    # guardar la sesión: con dos workers de gunicorn, lo que no viaja en ese
    # estado no existe para la petición siguiente.
    autos: str = ""
    # Los textos del acto y del recurso, para el detector de contaminación.
    fuentes: list = field(default_factory=list)

    def parrafos_antecedentes(self) -> list[str]:
        """Sin el encabezado que el modelo se pone a sí mismo.

        El prompt se titula «QUINTO. ANTECEDENTES» y el modelo lo reproduce como
        primera línea. La plantilla ya trae el suyo —«QUINTO. Antecedentes. Para
        una mejor comprensión del asunto…»— y el documento salía con los dos
        seguidos. Mismo caso que el estudio de fondo, misma cura.
        """
        ps = [p.strip() for p in self.antecedentes.split("\n") if p.strip()]
        if ps and re.match(r"^(?:QUINTO|CUARTO|SEXTO)\.?\s*ANTECEDENTES\s*\.?$",
                           ps[0], re.I):
            ps = ps[1:]
        return ps

    def parrafos_acto(self) -> list[str]:
        return [p.strip() for p in self.resumen_acto.split("\n") if p.strip()]

    def parrafos_conceptos(self) -> list[str]:
        return [p.strip() for p in self.resumen_conceptos.split("\n") if p.strip()]

    def parrafos_problemas(self) -> list[str]:
        fuera = []
        if self.problema_global:
            fuera.append(f"El problema jurídico a resolver consiste en determinar "
                         f"{self.problema_global[0].lower()}{self.problema_global[1:]}")
        for p in self.problemas:
            fuera.append(p.get("pregunta", ""))
        return [x for x in fuera if x]


# ═══════════════════════════════════════════════════════════════════════════
# Comprobaciones deterministas — antes de dar por bueno un resumen
# ═══════════════════════════════════════════════════════════════════════════

_PRESENTE = re.compile(r"\b(argumenta|alega|aduce|sostiene|señala|refiere|manifiesta)\b", re.I)
_PASADO = re.compile(r"\b(consideró|concluyó|determinó|resolvió|precisó|señaló|sostuvo|estimó)\b", re.I)


def revisar(f: Fases123) -> list[str]:
    """Lo que se puede comprobar sin modelo. Devuelve avisos, no excepciones.

    El TIEMPO VERBAL es el que delata un resumen mal hecho: lo que hizo la
    responsable va en pretérito y lo que reclama la parte, en presente. Está
    medido sobre 40 engroses y es lo primero que se nota al leer.
    """
    avisos = []
    na, nc = len(f.resumen_acto.split()), len(f.resumen_conceptos.split())
    if f.resumen_acto and not _PASADO.search(f.resumen_acto):
        avisos.append("El resumen del acto no usa pretérito: no suena a engrose.")
    if f.resumen_conceptos and not _PRESENTE.search(f.resumen_conceptos):
        avisos.append("El resumen de los conceptos no usa presente.")
    if f.resumen_acto and _PRESENTE.search(f.resumen_acto[:400]):
        avisos.append("El resumen del acto arranca en presente; debe ir en pretérito.")
    for etiqueta, n, objetivo in (("del acto", na, PALABRAS_RESUMEN_ACTO),
                                  ("de conceptos", nc, PALABRAS_RESUMEN_CONCEPTOS)):
        if n and not (0.5 * objetivo <= n <= 1.8 * objetivo):
            avisos.append(f"El resumen {etiqueta} tiene {n} palabras; la mediana "
                          f"de los engroses es {objetivo}.")
    # OJO CON ESTA COMPROBACIÓN, que ya dio un falso positivo.
    #
    # «La Sala consideró fundado el agravio» NO es el resumidor calificando:
    # es reportar lo que la responsable calificó, y así lo escribe David
    # palabra por palabra. Lo que sí está prohibido es que el resumen califique
    # POR SU CUENTA — «los conceptos de violación son fundados»—, que es
    # adelantar el estudio.
    #
    # Se distingue por la atribución: si la calificación viene precedida de un
    # verbo de la responsable, es cita; si el sujeto son los conceptos o los
    # agravios, es juicio propio.
    _CALIFICA_SOLO = re.compile(
        r"\b(?:los\s+)?(?:conceptos(?:\s+de\s+violaci[óo]n)?|agravios)\s+"
        r"(?:son|resultan?|devienen)\s+(?:esencialmente\s+)?"
        r"(?:fundad|infundad|inoperant|ineficac)", re.I)
    _ATRIBUYE = re.compile(
        r"(?:consider[óo]|determin[óo]|estim[óo]|resolvi[óo]|conclu[yi][óo]|"
        # `\s*$` y no `\s+`: el texto previo llega ya sin espacios finales
        # (se le aplica rstrip), así que exigir espacio tras «que» hacía que
        # NUNCA casara y toda cita atribuida se marcaba como juicio propio.
        r"precis[óo]|se[ñn]al[óo]|sostuvo)\s+(?:que)?\s*$", re.I)
    for m in _CALIFICA_SOLO.finditer(f.resumen_acto or ""):
        # Si en los 70 caracteres previos hay un verbo de la responsable, la
        # calificación es SUYA y el resumen sólo la reporta.
        antes = f.resumen_acto[max(0, m.start() - 70):m.start()]
        if not _ATRIBUYE.search(antes.rstrip()):
            avisos.append("El resumen del acto CALIFICA por su cuenta. Ahí sólo se expone.")
            break
    if "**" in f.resumen_acto or "**" in f.resumen_conceptos:
        avisos.append("Se coló Markdown.")
    return avisos
