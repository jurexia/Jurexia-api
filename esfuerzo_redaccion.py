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


_PENSAR_ABRE = "<!--THINKING_START-->"
_PENSAR_CIERRA = "<!--THINKING_END-->"


def _sin_marcadores(texto: str) -> str:
    """La respuesta anterior tal como la LEYÓ el abogado.

    El historial vuelve del cliente con lo que la pantalla guarda junto al
    texto: el mapa de fuentes al final (`<!-- CITATION_META:{…} -->`, con el
    texto de cada fuente, decenas de miles de caracteres), `REGISTROS_FUERA` y
    el razonamiento entre `THINKING_START` y `THINKING_END` al principio. Con
    el mapa detrás, «¿Desea que redacte la demanda?» quedaba fuera de los
    últimos 600 caracteres y el «sí» del abogado se atendía como consulta
    (27-sep-2026): las pruebas pasaban porque sus respuestas no traían mapa.

    Lineal a propósito, con str.find y sin expresiones: el texto llega del
    cliente. Un marcador sin cierre se lleva lo que queda, que es un corte y
    no prosa.
    """
    if not texto or "<!--" not in texto:
        return texto or ""
    partes, i = [], 0
    while True:
        a = texto.find("<!--", i)
        if a < 0:
            partes.append(texto[i:])
            break
        partes.append(texto[i:a])
        if texto.startswith(_PENSAR_ABRE, a):
            b = texto.find(_PENSAR_CIERRA, a + len(_PENSAR_ABRE))
            if b < 0:
                break
            i = b + len(_PENSAR_CIERRA)
        else:
            b = texto.find("-->", a + 4)
            if b < 0:
                break
            i = b + 3
    return "".join(partes)


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
    r"solicitud|argumentos?|argumentacion|esqueleto argumentativo|memorial|informe justificado|"
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
# «¿Podrías contestar la demanda?» y «¿puedes pasarlo a formato de demanda?»
# también encargan, pero con el documento como complemento mismo —«¿puedes
# contestar la PREGUNTA del amparo?» es consulta— o como destino de la
# conversión, nunca como su origen: «¿puedes convertir la sentencia en un
# resumen?» no pide un escrito.
_PREGUNTA_CONTESTAR = re.compile(_PUEDES + r"contestar\s+")

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
# «Quiero que me ayudes con la demanda de divorcio» es el mismo encargo que
# «ayúdame con la demanda». Sólo con «con» y el documento detrás, o con «a» y
# un verbo de hacer: «que me ayudes a ENTENDER la demanda» sigue siendo
# consulta, y la pregunta «¿me ayudas con…?» también —si era un encargo, la
# respuesta cierra ofreciendo el escrito y el «sí» lo pide—.
_AYUDA_CON = re.compile(
    r"^(?:(?:necesito|nesecito|nececito|nesesito|necesitamos|quiero|queremos|ocupo|"
    r"requiero|me urge|quisiera|me gustaria)\s+)?(?:que\s+)?(?:me\s+|nos\s+)?"
    r"ayud(?:es|en|e|ar|ame|anos)\s+(?:con|a\s+(?:preparar|hacer|escribir|elaborar|"
    r"armar|construir|generar|formular|proyectar|desarrollar|realizar|contestar))\s+"
)
_REDACCION_DE = re.compile(r"^(?:la\s+)?redaccion\s+(?:de|del)\b")
# «Hazlo en formato de demanda», «pásalo a escrito», «conviértelo en un
# amparo»: la respuesta anterior se pide convertida en escrito. Es encargo
# aunque lo anterior fuera una consulta. Sin preposición sólo con artículo
# detrás —«hazlo un escrito»—, para que «hazlo un poco más largo» no cuente.
_DESTINO = (
    r"(?:(?:en|como|a|al)\s+|(?=(?:un|una)\s))" + _ARTICULO
    + r"(?:(?:formato|forma|manera|estilo|modelo|machote|version)\s+(?:de\s+)?"
    + _ARTICULO + r")?" + _DOC + r"\b"
)
_CONVERTIR = re.compile(
    r"^(?:haz|pon|pasa|convierte|transforma|vuelve|escribe|presenta|dame|arma|deja)"
    r"(?:l[oa]s?|mel[oa]s?|nosl[oa]s?)\s+" + _DESTINO
)
_PREGUNTA_CONVERTIR = re.compile(
    _PUEDES + r"(?:convertir|transformar|pasar|poner|hacer|dejar|escribir|presentar)"
    r"(?:l[oa]s?|mel[oa]s?|nosl[oa]s?)\s+" + _DESTINO
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

# «Contesta la demanda que te pasé» es un encargo; pero «contesta» es también
# la tercera persona —«si contesta la demanda fuera de plazo, ¿qué pasa?»—.
# Por eso va aparte y sólo cuenta en imperativo: sin signo de pregunta y sin
# el «si» condicional delante (el «sí, contesta…» lleva coma).
_CONTESTA = re.compile(r"^contest(?:a|ame|anos)\b")
_CONTESTA_DENTRO = re.compile(
    r"(?:[.;:!]\s*|,\s*|\by\s+|\bahora\s+(?:si\s+)?|\bentonces\s+|\bpor favor\s+|"
    r"\bluego\s+|\bdespues\s+)contesta(?:me)?\s+" + _ARTICULO + _ENVOLTURA + _DOC + r"\b"
    r"|\bque (?:me |nos )?contestes\s+" + _ARTICULO + _ENVOLTURA + _DOC + r"\b"
)
_SI_CONDICIONAL = re.compile(r"\bsi\s+(?:no\s+|ya\s+|solo\s+)?contest")

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
    llano = _plano(mensaje)
    if not llano:
        return False
    t = _CORTESIA.sub("", llano).strip()
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
    m = _PREGUNTA_CONTESTAR.match(t)
    if (m and _objeto_es_documento(t[m.end():], estricto=True)) or _PREGUNTA_CONVERTIR.match(t):
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
    # El documento tiene que ser el complemento mismo: «que me ayudes con EL
    # PLAZO del amparo» es consulta. Y sin pregunta: «ayúdame con la demanda,
    # ¿qué requisitos lleva?» pide los requisitos, no el escrito.
    m = _AYUDA_CON.match(t)
    if m and "?" not in t and (_REDACCION_DE.match(t[m.end():])
                               or _objeto_es_documento(t[m.end():], estricto=True)):
        return True
    if _CONVERTIR.match(t):
        return True
    # El «si» condicional se busca ANTES de quitar la cortesía, que se lo come
    # como si fuera el «sí» de asentir. A media frase sólo en el arranque: un
    # borrador pegado para revisarlo («…el demandado fue emplazado y contesta
    # la demanda…») no es un encargo.
    if "?" not in t and not _SI_CONDICIONAL.search(llano[:60]):
        # Con el documento como complemento: «contesta la PREGUNTA del
        # amparo» es consulta.
        m = _CONTESTA.match(t)
        if (m and _objeto_es_documento(t[m.end():], estricto=True)) or _CONTESTA_DENTRO.search(t[:600]):
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
    texto = _sin_marcadores(texto)
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


# ── El retoque, en su lugar (28-sep-2026) ────────────────────────────────
# Un retoque —«agrega un concepto…», «corrige el nombre del quejoso»— llegaba
# como respuesta nueva y la hoja la anexaba DEBAJO del escrito anterior: el
# abogado acababa con dos escritos casi iguales y borraba a mano el viejo. Y
# el modelo retocaba SU versión, no la de la hoja, así que lo que el abogado
# había corregido a mano volvía a salir mal.
#
# Ahora se distinguen dos retoques. El que MODIFICA entrega el escrito
# completo, ya con el cambio, y la pantalla lo pone EN LUGAR del anterior
# (la respuesta lleva MARCA_REEMPLAZA); la pantalla, además, manda como
# respuesta anterior el escrito tal como está en la hoja, con las
# correcciones del abogado. El que CONTINÚA —«continúa», «sigue»,
# «termina»— se escribe desde donde se quedó y se anexa, como siempre.
MARCA_REEMPLAZA = "<!--REEMPLAZA_ESCRITO-->"

# «Sigue el mismo formato, pero agrega un agravio» no es seguir: es cambiar.
_CONTINUAR = re.compile(
    r"^(?:(?:me |nos )?(?:puedes|podrias|pudieras) (?:me |nos )?)?"
    r"(?:continu\w*|sigue\b|seguir\b|sigamos\b|prosigue\b|prosigamos\b|termina\w*)"
    r"(?! (?:con |usando |igual )?(?:el|la|los|las|ese|esa|esos|esas) (?:mism|formato|estilo|estructura))"
)


def tipo_de_ajuste(mensaje: str) -> str:
    """'continuar' si el retoque pide seguir el escrito donde se quedó;
    'modificar' si pide cambiarlo (y entonces sustituye al anterior)."""
    t = _CORTESIA.sub("", _plano(mensaje)).strip()
    return "continuar" if _CONTINUAR.match(t) else "modificar"


INSTRUCCION_MODIFICAR = """EL ABOGADO PIDE UN CAMBIO AL ESCRITO QUE YA TIENE.
Tu respuesta anterior es ese escrito tal como está ahora en su hoja, con lo
que él mismo corrigió a mano. Entrega el escrito COMPLETO, de la primera a la
última línea, con el cambio aplicado: sustituirá al anterior en su hoja. Lo
que el cambio no toca se conserva palabra por palabra —sus correcciones, sus
datos, los [DATO PENDIENTE: …] que siguen pendientes y las citas [Doc ID]—:
no lo resumas, no lo reordenes, no lo mejores de paso. No anuncies el cambio
dentro del escrito; si hay algo que advertir, va en la nota para el abogado."""

INSTRUCCION_CONTINUAR = """EL ABOGADO PIDE QUE SIGAS EL ESCRITO DONDE SE QUEDÓ.
Tu respuesta anterior es lo ya escrito: no lo repitas ni lo resumas. Empieza
exactamente donde terminó —a media sección, si ahí se cortó— y sigue hasta el
cierre del escrito."""


# «¿Desea que redacte la demanda?» — «Sí, por favor». La consulta ofreció el
# escrito y el abogado lo acepta: eso es un encargo, aunque no lo diga.
_OFRECE_ESCRITO = re.compile(
    r"\b(?:redact|elabor|prepar|formul|proyect)\w*\b[^?]{0,80}\b" + _DOC + r"\b[^?]{0,80}\?"
)
_AFIRMATIVO = re.compile(
    r"^(?:si|claro|adelante|hazlo|hazla|dale|va|ok|okay|de acuerdo|por favor|"
    r"procede|correcto|perfecto|me parece bien|esta bien|andale)\b"
)

# La despedida con que la revisión de sentencia ofrece redactar. Vive aquí, y
# main.py la inserta en su prompt, para que la oferta y el detector que la
# reconoce no vuelvan a separarse: la anterior mandaba elegir un Genio y
# encender «Redacción Especializada» —las dos cosas salieron de la pantalla el
# 25-sep-2026— y el «ok» que pedía no encendía la redacción (27-sep-2026).
OFERTA_TRAS_REVISION = (
    "¿Quieres que redacte los argumentos para fortalecer el proyecto o para "
    "cambiar su sentido? Respóndeme «sí» y los redacto."
)


def acepta_oferta(mensaje: str, anterior: str) -> bool:
    """El abogado dice «sí» a la oferta de redactar con que cerró la consulta."""
    t = _plano(mensaje)
    if not t or len(t) > 80 or t.endswith("?"):
        return False
    return bool(_AFIRMATIVO.match(t.lstrip("¡!¿ "))) and bool(
        _OFRECE_ESCRITO.search(_plano(_sin_marcadores(anterior))[-600:]))


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
    # Una sola limpieza para las dos lecturas: el mapa de fuentes puede medir
    # más que la respuesta misma.
    anterior = _sin_marcadores(next((_campo(m, "content") for m in reversed(lista[:-1])
                                     if _campo(m, "role") == "assistant"), ""))
    if anterior and es_ajuste_de_escrito(ultimo) and parece_escrito(anterior):
        return "ajuste"
    if anterior and acepta_oferta(ultimo, anterior):
        return "acepta"
    return ""


def normalizar_intencion(valor: Optional[str]) -> Optional[str]:
    """La decisión explícita de la etiqueta del compositor (27-sep-2026).

    El abogado ve antes de enviar si su mensaje se redactará como escrito o se
    contestará como consulta, y la cambia con un clic. Si la manda, manda ella:
    lo que la pantalla anunció es lo que ocurre. Cualquier otro valor cuenta
    como ausente y decide el detector."""
    v = _plano(valor or "")
    if v in ("redactar", "escrito", "redaccion"):
        return "redactar"
    if v in ("consultar", "consulta"):
        return "consultar"
    return None


def decidir_redaccion(mensajes: Iterable[Any], intencion: Optional[str] = None) -> str:
    """Lo que decide /chat cuando no hay marcador: '' si es consulta; si no,
    por qué se redacta —'pide', 'ajuste', 'acepta' o 'etiqueta' (lo eligió
    el abogado aunque el detector no lo viera)—. La etiqueta manda en los dos
    sentidos."""
    i = normalizar_intencion(intencion)
    if i == "consultar":
        return ""
    motivo = detectar_redaccion(mensajes)
    return motivo or ("etiqueta" if i == "redactar" else "")


def intencion_del_mensaje(mensaje: str, anterior: str = "") -> str:
    """Lo que `detectar_redaccion` diría de `mensaje` tras la respuesta
    `anterior` (vacía si no hay): '' consulta; 'pide', 'ajuste' o 'acepta'.
    Es lo que pregunta la etiqueta mientras el abogado escribe."""
    mensajes = [{"role": "assistant", "content": anterior}] if anterior else []
    mensajes.append({"role": "user", "content": mensaje or ""})
    return detectar_redaccion(mensajes)


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


# ── La nota para el abogado (28-sep-2026) ───────────────────────────────
# El escrito se entrega limpio, listo para firmarse. Lo que el abogado debe
# saber y no se presenta —qué decidió el modelo por él, qué verificar antes
# de presentar, el punto débil— o iba DENTRO del escrito, y acababa impreso
# en el Word, o no iba en ningún sitio, porque el núcleo prohibía toda nota.
# Ahora va DESPUÉS del escrito, bajo un rótulo fijo que la pantalla reconoce:
# la hoja y el Word se quedan con el escrito y la nota se enseña en el chat.
# El rótulo vive aquí para que el prompt que lo pide y la pantalla que lo
# corta (`separarNota`, en `lib/documento/marcado.ts` del frontend) no se
# separen.
ROTULO_NOTA = "NOTA PARA EL ABOGADO"

NOTA_PARA_EL_ABOGADO = f"""
────────────────────────────────────────────────────────────────
 AL TERMINAR — LA NOTA PARA EL ABOGADO (VA FUERA DEL ESCRITO)
────────────────────────────────────────────────────────────────
Cuando el escrito esté completo —después de la firma, del último resolutivo
o del último párrafo del estudio—, deja una línea en blanco y escribe este
rótulo, solo en su línea y tal cual —es la única almohadilla que admite la
respuesta—:

## {ROTULO_NOTA}

Debajo, en viñetas breves —no más de 150 palabras en total—, sólo lo que el
abogado tiene que saber antes de firmar y que no puede ir en el escrito:
  · lo que decidiste por él: la vía, el tipo de amparo o de recurso, la
    autoridad señalada, un hecho que diste por supuesto;
  · lo que conviene verificar antes de presentar: el plazo que corre, los
    anexos y las copias de traslado, la prueba que falta recabar;
  · el punto más débil del escrito y cómo reforzarlo.
La pantalla enseña la nota aparte: la hoja y el Word se quedan sólo con el
escrito. Por eso nada de la nota se anuncia dentro del escrito, la nota no
lleva citas [Doc ID] y no repite los [DATO PENDIENTE: …], que ya están donde
van. Si no hay nada que advertir, omite la nota entera.
"""


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
- DATOS QUE FALTAN: la regla de los tres registros, sin excepción: con más
  extensión hay más sitios donde un nombre o una fecha se cuelan inventados.
  Cada hueco con su forma —[DATO PENDIENTE: fecha de notificación]— y sigue
  escribiendo.
- ANTES DE ENTREGAR, relee en silencio: que el esqueleto esté completo, que
  ninguna cita carezca de fuente en el contexto y que el tono sea el de un
  escrito que se presenta hoy.
- LA NOTA PARA EL ABOGADO no crece con el escrito: sigue breve y fuera de él.
"""
