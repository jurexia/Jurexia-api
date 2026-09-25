"""
El esfuerzo de redacción (25-sep-2026).

David: «que no sea necesario consultar o redactar, sino que baste con que
quien introduzca "redacta" en el prompt active estas funciones». El
interruptor Buscar/Redactar del compositor desaparece; en su lugar hay un
solo desplegable —Esfuerzo: Básico, Pro o Platinum— que el frontend manda
en CADA consulta como campo `esfuerzo`. Aquí se decide:

  1. si el mensaje pide un escrito (`pide_escrito`) o ajusta el escrito que
     se acaba de entregar (`es_ajuste_de_escrito` + `parece_escrito`); sólo
     entonces el esfuerzo cuenta. Una pregunta sigue siendo una consulta,
     con el esfuerzo que sea;
  2. qué escalón le toca de verdad según su plan (`esfuerzo_permitido`): el
     desplegable se puede manipular, el plan no.

Los marcadores de siempre (`[MODO_REDACCION]`, `_PRO`, `_PLATINUM`) se
siguen respetando tal cual: los usan los flujos de trabajo, la tarjeta
«Escrito legal», «Desarrollar a partir de este fundamento» y las apps viejas.

Por qué las preguntas quedan fuera: antes, con el interruptor apagado, frases
como «cómo impugnar…» o «qué alego…» también encendían la redacción y el
abogado recibía un escrito en prosa donde había pedido una explicación. Con
el interruptor fuera, este detector ES el interruptor, y confundir una
pregunta con un encargo ahora cuesta el motor del escalón elegido.

CÓMO PROBARLO
    python test_esfuerzo_redaccion.py
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Optional

ESFUERZOS = ("basico", "pro", "platinum")

PLANES_PRO = {
    "pro_monthly", "pro_annual",
    "platinum_monthly", "platinum_annual", "ultra_secretarios",
}
PLANES_PLATINUM = {"platinum_monthly", "platinum_annual", "ultra_secretarios"}


def _plano(texto: str, lineas: bool = False) -> str:
    """Minúsculas, sin acentos y con un solo espacio (conservando los saltos
    de línea si `lineas`)."""
    t = unicodedata.normalize("NFD", texto or "")
    t = "".join(c for c in t if unicodedata.category(c) != "Mn").lower()
    if lineas:
        return "\n".join(re.sub(r"[ \t]+", " ", l).strip() for l in t.splitlines())
    return re.sub(r"\s+", " ", t).strip()


# Lo que se antepone por cortesía y no cambia el encargo.
_CORTESIA = re.compile(
    r"^(?:[¿¡\"'«(\[\-*•]+\s*)*"
    r"(?:(?:por favor|porfa|porfavor|hola|oye|buen(?:os|as)? (?:dias?|tardes?|noches?)|"
    r"buenas|iurexia|licenciad[oa]|ok|okay|muy bien|bien|perfecto|excelente|gracias|"
    r"ahora si|ahora|entonces|bueno|claro|listo|va|si|y)"
    r"[\s,.:;!]+)*"
)

# «¿Cómo se redacta…?» pregunta por el oficio; no lo encarga.
_PREGUNTA = re.compile(
    r"^(?:como|que|cual|cuales|cuando|donde|por que|porque|quien|quienes|cuanto|"
    r"cuantos|cuanta|cuantas|es|son|existe|existen|hay|procede|proceden|se puede|"
    r"puedo|debo|debe|conviene|sirve|vale)\b"
)

_DOC = (
    r"(?:demandas?|escritos?|amparos?|recursos?|apelacion|revision|queja|"
    r"reclamacion|agravios?|conceptos? de violacion|contestacion|reconvencion|"
    r"denuncias?|querellas?|promocion(?:es)?|oficios?|peticion(?:es)?|contratos?|"
    r"convenios?|acuerdos?|alegatos?|incidentes?|proyecto de|considerandos?|"
    r"estudio de fondo|sentencias?|resolucion|dictamen|opinion juridica|"
    r"carta poder|cartas?|poder notarial|desistimiento|ofrecimiento de pruebas|"
    r"solicitud|argumentos?|argumentacion|memorial|informe justificado|"
    r"excepciones|inconformidad|juicio de nulidad|requerimiento|minuta|"
    r"clausulas?|acta|testamento|mandato|pagare|interpelacion|notificacion|"
    r"contrarreplica|replica|duplica|vista|desahogo|interrogatorio|"
    r"pliego de posiciones|posiciones|comparecencia|manifestacion)"
)
_DOCUMENTO = re.compile(r"\b" + _DOC + r"\b")
_ARTICULO = r"(?:(?:un|una|el|la|los|las|unos|unas|mi|mis|tu|su|dicho|dicha|otro|otra)\s+)?"
_ENVOLTURA = r"(?:(?:nuevo|nueva|modelo de|borrador de|formato de|proyecto de|machote de)\s+)?"
_OBJETO_ESTRICTO = re.compile(r"^" + _ARTICULO + _ENVOLTURA + _DOC + r"\b")

# «¿Me puedes redactar…?» siempre es encargo; «¿puedes hacer…?» sólo si lo que
# se pide hacer es un documento («¿puedes hacer un análisis?» es consulta).
_PUEDES = r"^(?:me |nos )?(?:puedes|podrias|pudieras|puede|podria|podra|podras) (?:me |nos )?"
_PREGUNTA_REDACTAR = re.compile(_PUEDES + r"(?:redact|reescrib)")
_PREGUNTA_ENCARGO = re.compile(
    _PUEDES + r"(?:elabor|prepar|hac|escrib|formul|proyect|gener|arm|constru|realiz)\w*"
)

# En cualquier parte del mensaje, salvo cuando «redactar» no es lo que se
# pide sino de lo que se habla: «¿qué contrato DEBO redactar?», «la lista de
# datos que faltan PARA redactar el contrato», «cómo SE redacta», «el cuadro
# donde el suscrito TE redacta». Medido sobre 1,000 mensajes reales del 18 al
# 25-sep-2026.
_VERBO_REDACTAR = re.compile(
    r"(?<!\bse )(?<!\bte )(?<!\ble )(?<!\bdebo )(?<!\bdebe )(?<!\bdeberia )"
    r"(?<!\bpuedo )(?<!\bcomo )(?<!\bpara )(?<!\bestoy )(?<!\bestamos )"
    r"\bredact(?:a|ame|anos|alo|ala|alos|alas|ale|es|en|e|ar|arme|arnos|emos)\b"
)
# «Necesito ayuda para redactar…» sí es un encargo, aunque lleve «para».
_AYUDA_REDACTAR = re.compile(r"\bayuda (?:para|a|con) (?:la )?redact")

# Verbos que por sí solos ya son encargo de un texto jurídico.
_VERBO_FUERTE = re.compile(
    r"^(?:redact\w*|proyect(?:a|ame|anos)|formul(?:a|ame|anos))\b"
)

# Verbos que piden algo, pero no necesariamente un escrito: «escribe el
# artículo 17» es una consulta, «escribe la demanda» es un encargo. Van con un
# documento justo después.
_VERBO_AMBIGUO = re.compile(
    r"^(?:elabor(?:a|ame|anos|ar)|prepar(?:a|ame|anos|ar)|haz|hazme|hasme|asme|"
    r"hagame|haganme|hacer|escrib(?:e|eme|enos|ir)|gener(?:a|ame|anos|ar)|"
    r"arm(?:a|ame|anos|ar)|constru(?:ye|yeme|yenos|ir)|desarroll(?:a|ame|anos)|"
    r"realiz(?:a|ame|anos|ar)|ayudame con|"
    r"(?:ayudame|ayudanos|vamos) a (?:preparar|hacer|escribir|elaborar|armar|"
    r"construir|generar|formular|proyectar|desarrollar|realizar))\b"
)
# «Dame el escrito sin explicaciones» sí; «dame los requisitos de la demanda»
# no: con estos verbos el documento tiene que ser lo que se entrega.
_VERBO_ENTREGA = re.compile(
    r"^(?:dame|damelo|damela|pasame|mandame|entregame|me das|me pasas|me mandas)\b"
)

# «necesito una demanda» sí; «necesito saber los requisitos de una demanda» no.
_NECESIDAD = re.compile(
    r"^(?:necesito|nesecito|nececito|nesesito|necesitamos|quiero|queremos|ocupo|"
    r"requiero|me urge) "
    r"(?:(?:que )?(?:me |nos )?(?:redact|elabor|prepar|hag|escrib|formul|proyect|gener|arm|constru|realic)\w*"
    r"|(?=(?:un|una|el|la|los|las|unos|unas|mi|mis|otro|otra) ))"
)

# El encargo a media frase: «…ahora lee las páginas y ELABORA LA QUEJA»,
# «…necesito que me HAGAS UN CONTRATO». El imperativo es idéntico a la tercera
# persona («el juez elabora el acta»), así que sólo cuenta al empezar una
# oración o tras «y», «ahora», «entonces»…
_ENCARGO_DENTRO = re.compile(
    r"(?:(?:^|[.;:!?]\s*|,\s*|\by\s+|\bahora\s+(?:si\s+)?|\bentonces\s+|\bpor favor\s+|"
    r"\bluego\s+|\bdespues\s+)"
    r"(?:elabora(?:me)?|haz(?:me)?|hasme|prepara(?:me)?|formula(?:me)?|realiza(?:me)?|"
    r"escribe(?:me)?|genera(?:me)?|arma(?:me)?|proyecta(?:me)?)\s+"
    + _ARTICULO + _ENVOLTURA + _DOC + r"\b"
    r"|\bque (?:me |nos )?(?:hagas|elabores|prepares|formules|escribas|generes|"
    r"proyectes|realices|armes)\s+" + _ARTICULO + _ENVOLTURA + _DOC + r"\b)"
)

# «Formato de contestación de amparo…», «modelo de demanda de alimentos».
_FORMATO_INICIAL = re.compile(
    r"^(?:formato|modelo|machote|plantilla|ejemplo|borrador) (?:de |para |del )?"
    + _ARTICULO + _DOC + r"\b"
)
# «¿Cómo sería el escrito para presentarlo al juez?» pide verlo escrito.
_COMO_SERIA = re.compile(r"^como (?:seria|quedaria|queda) " + _ARTICULO + _DOC + r"\b")

# Lo que se le dice a un escrito recién entregado para retocarlo.
_AJUSTE = re.compile(
    r"^(?:(?:me |nos )?(?:puedes|podrias|pudieras) (?:me |nos )?(?:volver a|agreg|anad|"
    r"inclu|ampli|corregi|cambi|modific|ajust|quit|elimin|reescrib|reformul|mejor|"
    r"desarroll|profundiz|continu|pon|dar|pasar)\w*|"
    r"agreg\w*|anad\w*|inclu\w*|integr\w*|amplia\w*|amplie\w*|extiend\w*|"
    r"extend\w*|desarroll\w*|profundiz\w*|continu\w*|sigue|seguir|termina\w*|"
    r"complet\w*|corrig\w*|correg\w*|cambi\w*|modific\w*|ajust\w*|quit\w*|"
    r"elimin\w*|suprim\w*|borr\w*|sustitu\w*|reemplaz\w*|reescrib\w*|"
    r"reformul\w*|rehaz\w*|rehac\w*|mejor\w*|pul\w*|perfeccion\w*|fortalec\w*|"
    r"reforz\w*|refuerz\w*|adapt\w*|convierte\w*|convertir\w*|vuelve a|"
    r"hazl[oa]s?|haz(?:me)? (?:mas|menos|que)|pon\w*|redact\w*|resum\w*|"
    r"acort\w*|recort\w*|enumer\w*|numer\w*|orden\w*|reorden\w*|"
    r"traslad\w*|mueve\w*|usa|utiliza|cita\w*|mas (?:largo|extenso|formal|breve|corto)|"
    r"otra version|otra vez)\b"
)

# Rótulos que tiene un escrito y no tiene una respuesta de consulta.
_ROTULOS = (
    "hechos", "puntos petitorios", "petitorios", "conceptos de violacion",
    "concepto de violacion", "primer agravio", "segundo agravio", "agravios",
    "prestaciones", "protesto lo necesario", "protesto", "derecho",
    "pruebas", "estudio", "considerando", "resolutivos", "resuelve",
    "proemio", "c. juez", "h. tribunal", "presente", "antecedentes",
    "preceptos constitucionales violados", "autoridades responsables",
    "clausulas", "declaraciones",
)
_ORDINAL_INICIAL = re.compile(
    r"^\W*(?:primero|segundo|tercero|cuarto|quinto|sexto|septimo|octavo|noveno|"
    r"decimo)\.\s"
)


def pide_escrito(mensaje: str) -> bool:
    """¿El mensaje ENCARGA un texto jurídico, o pregunta algo?"""
    t = _plano(mensaje)
    if not t:
        return False
    t = _CORTESIA.sub("", t).strip()
    if not t:
        return False
    if t.startswith("redaccion de") or t.startswith("redaccion del"):
        return True
    if _COMO_SERIA.match(t):
        return True
    if _PREGUNTA.match(t):
        return False
    if _PREGUNTA_REDACTAR.match(t):
        return True
    m = _PREGUNTA_ENCARGO.match(t)
    if m and _objeto_es_documento(t[m.end():]):
        return True
    if _VERBO_REDACTAR.search(t) or _AYUDA_REDACTAR.search(t):
        return True
    if _VERBO_FUERTE.match(t) or _FORMATO_INICIAL.match(t):
        return True
    m = _VERBO_ENTREGA.match(t)
    if m and _objeto_es_documento(t[m.end():], estricto=True):
        return True
    m = _NECESIDAD.match(t) or _VERBO_AMBIGUO.match(t)
    if m and _objeto_es_documento(t[m.end():]):
        return True
    return bool(_ENCARGO_DENTRO.search(t))


def _objeto_es_documento(resto: str, estricto: bool = False) -> bool:
    """Lo que sigue al verbo: ¿es un documento? «hazme UNA DEMANDA» sí;
    «escribe el artículo 17 de la Ley de Amparo» no, aunque diga «amparo»
    más adelante. Se miran las cuatro palabras siguientes; en `estricto`, el
    documento tiene que ser el complemento mismo."""
    palabras = [p for p in resto.split() if p not in ("me", "nos", "que", "por", "favor")]
    if estricto:
        return bool(_OBJETO_ESTRICTO.match(" ".join(palabras[:6])))
    return bool(_DOCUMENTO.search(" ".join(palabras[:4])))


def parece_escrito(texto: str) -> bool:
    """¿La respuesta anterior fue un escrito (y no una consulta)?"""
    if not texto:
        return False
    t = _plano(texto, lineas=True)
    if _ORDINAL_INICIAL.match(t.lstrip()):
        return True
    # Rótulos a principio de línea, con o sin negritas: «**HECHOS**», «PRUEBAS.».
    lineas = [re.sub(r"^[\W_]+|[\W_]+$", "", l) for l in t.splitlines()]
    hallados = {r for l in lineas if l for r in _ROTULOS if l == r or l.startswith(r + " ")}
    return len(hallados) >= 2


def es_ajuste_de_escrito(mensaje: str) -> bool:
    """«Agrega un concepto de violación», «hazlo más extenso»…

    Una pregunta nunca es ajuste: «¿por qué citaste el 17?» se contesta.
    """
    crudo = (mensaje or "").strip()
    if not crudo or crudo.startswith("¿") or crudo.endswith("?"):
        return False
    t = _CORTESIA.sub("", _plano(crudo)).strip()
    return bool(t) and bool(_AJUSTE.match(t))


# «¿Desea que redacte la demanda?» — «Sí, por favor». La consulta ofreció el
# escrito y el abogado lo acepta: eso es un encargo, aunque no lo diga.
_OFRECE_ESCRITO = re.compile(
    r"\b(?:redact|elabor|prepar|formul|proyect)\w*\b[^?]{0,80}\b" + _DOC + r"\b[^?]{0,80}\?"
)
_AFIRMATIVO = re.compile(
    r"^(?:si|claro|adelante|hazlo|hazla|dale|va|ok|okay|de acuerdo|por favor|"
    r"procede|correcto|perfecto|me parece bien|esta bien|andale)\b"
)


def acepta_oferta(mensaje: str, anterior: str) -> bool:
    """El abogado dice «sí» a la oferta de redactar con que cerró la consulta."""
    t = _plano(mensaje)
    if not t or len(t) > 80 or t.endswith("?"):
        return False
    return bool(_AFIRMATIVO.match(t.lstrip("¡!¿ "))) and bool(
        _OFRECE_ESCRITO.search(_plano(anterior)[-600:]))


def _campo(m: Any, nombre: str) -> str:
    if isinstance(m, dict):
        return m.get(nombre) or ""
    return getattr(m, nombre, "") or ""


def detectar_redaccion(mensajes: Iterable[Any]) -> str:
    """'' si es consulta; 'pide' si encarga un escrito; 'ajuste' si retoca
    el escrito que se acaba de entregar; 'acepta' si dice que sí a la oferta
    de redactar con que cerró la respuesta anterior.

    `mensajes` es el historial del request (objetos con role/content o
    dicts). Los `system` —la carpeta, el flujo— no cuentan.
    """
    lista = [m for m in mensajes if _campo(m, "role") in ("user", "assistant")]
    if not lista or _campo(lista[-1], "role") != "user":
        return ""
    ultimo = _campo(lista[-1], "content")
    if pide_escrito(ultimo):
        return "pide"
    anterior = next((_campo(m, "content") for m in reversed(lista[:-1])
                     if _campo(m, "role") == "assistant"), "")
    if anterior and es_ajuste_de_escrito(ultimo) and parece_escrito(anterior):
        return "ajuste"
    if anterior and acepta_oferta(ultimo, anterior):
        return "acepta"
    return ""


def normalizar_esfuerzo(valor: Optional[str]) -> Optional[str]:
    v = _plano(valor or "")
    if v in ("platino", "platinum"):
        return "platinum"
    if v == "pro":
        return "pro"
    if v in ("basico", "base", "profesional", "normal"):
        return "basico"
    return None


def esfuerzo_permitido(pedido: Optional[str], plan: Optional[str], es_admin: bool = False) -> str:
    """El escalón que se sirve: el pedido, pero nunca por encima del plan."""
    p = normalizar_esfuerzo(pedido) or "basico"
    if es_admin or plan in PLANES_PLATINUM:
        return p
    if plan in PLANES_PRO:
        return "pro" if p == "platinum" else p
    return "basico"


# ── El acabado Platinum ──────────────────────────────────────────────────
# David: «si es pro dejarlo como está, pero si es platinum darle la fuerza de
# Terra con más tokens; ahí sí veríamos un cambio notable». El motor pone la
# capacidad; esto le dice en qué gastarla. Va al FINAL del prompt de
# redacción, donde mejor lo respetan los modelos de razonamiento, y sólo en
# Platinum: Básico y Pro reciben exactamente el prompt de siempre.
ACABADO_PLATINUM = """

════════════════════════════════════════════════════════════════
   ESCALÓN PLATINUM — ACABADO DE DESPACHO DE PRIMER NIVEL
════════════════════════════════════════════════════════════════

Este escrito lo firma un abogado que pagó el escalón más alto. Todo lo
anterior sigue rigiendo; esto sube el listón:

- EXTENSIÓN: desarrolla cada apartado hasta agotarlo. Una demanda o un
  recurso completos no bajan de 3,500 palabras; un estudio de fondo, de
  3,000. Si el material del contexto da para más, escribe más. Nunca
  cierres un apartado con una oración que podría haber sido un párrafo.
- FUNDAMENTACIÓN: integra al menos diez fuentes distintas del contexto
  cuando existan —Constitución, tratados, ley federal, ley local y
  jurisprudencia—, cada una tejida en el argumento que sostiene, con su
  transcripción pertinente y su [Doc ID].
- ARGUMENTACIÓN EN CAPAS: por cada pretensión o concepto, el argumento
  principal, el subsidiario («aun en el supuesto de que…») y la refutación
  anticipada de lo que opondrá la contraparte o sostendrá la autoridad.
- COHERENCIA CRUZADA: cada hecho relevante reaparece en el derecho que lo
  califica, cada prueba se relaciona con el hecho que acredita y cada punto
  petitorio responde a una prestación o concepto desarrollado. Nada queda
  suelto.
- DATOS QUE FALTAN: no inventes nombres, fechas, expedientes ni domicilios.
  Donde falte un dato del caso, deja el hueco visible entre corchetes
  —[NOMBRE DEL QUEJOSO], [FECHA DE NOTIFICACIÓN]— y sigue escribiendo.
- ANTES DE ENTREGAR, relee en silencio: que el esqueleto esté completo, que
  ninguna cita carezca de fuente en el contexto y que el tono sea el de un
  escrito que se presenta hoy.
"""
