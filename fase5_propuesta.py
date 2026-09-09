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

# EL TOPE, Y POR QUÉ SUBE. Con ocho, un asunto de siete problemas veía como
# mucho una tesis por problema, y en la práctica ninguna: se ordenaban las
# obligatorias delante y las ocho primeras podían ser todas del problema 1.
# Medido sobre cinco asuntos: 191 recuperadas, 10 invocadas.
#
# Sube a veinticuatro, pero lo que arregla el hueco no es el número: es el
# REPARTO. Se toman por turnos, una de cada problema, para que ninguno se
# quede sin material que invocar.
MAX_TESIS_PROPUESTA = 24
TESIS_CARACTERES = 1200

# DEL CATÁLOGO, NO DE UNA TUPLA SUELTA. Añadir una calificación era tocar esta
# línea, la de la pantalla, dos diccionarios de plural y siete `startswith`.
import tipos_asunto as _ta_s
SENTIDOS = _ta_s.SENTIDOS_OFRECIDOS

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
            return (f"· {_entero(self.problema)}\n"
                    f"    SIN PROPUESTA — el acervo no alcanza para sostener un "
                    f"sentido. {self.razon}")
        ap = ", ".join(str(a) for a in self.apoyos) or "sin apoyo"
        return (f"· {_entero(self.problema)}\n"
                f"    PROPUESTA: {self.sentido.upper()} ({self.confianza})\n"
                f"    {self.razon}\n"
                f"    Se apoya en: {ap}")


def _entero(pregunta: str, tope: int = 260) -> str:
    """El problema jurídico, legible.

    Estaba cortado a 120 caracteres a pelo, y una pregunta jurídica rara vez
    cabe ahí: en el 91/2025 el secretario leía «…del artículo 38, fracción V,
    del Código Fiscal d» y ahí se acababa. Justo la frase sobre la que tiene
    que decidir el sentido.

    Se deja entera; y si de verdad se dispara, se corta por la última palabra
    completa, que al menos no miente sobre dónde acaba.
    """
    t = " ".join(str(pregunta or "").split())
    if len(t) <= tope:
        return t
    corte = t[:tope].rsplit(" ", 1)[0]
    return corte + "…"


@dataclass
class Global:
    """LA PROPUESTA DEL ASUNTO ENTERO, no la de un problema.

    Faltaba. El motor proponía un sentido por problema y la pantalla enseñaba
    el del problema principal haciéndolo pasar por la solución global: con tres
    problemas, el secretario veía el sentido de uno y la etiqueta «global».

    No es lo mismo, y la diferencia es justo lo que él tiene que decidir. La
    solución global es qué pasa con el ASUNTO —de qué problema cuelga, qué
    arrastra consigo, y qué queda sin materia si ese prospera—. Eso no se
    deduce sumando calificaciones: hay que razonarlo con todos los problemas
    delante, que es lo que se le pide ahora al modelo en la MISMA llamada.
    """
    sentido: str = ""                 # fundado | infundado | inoperante | ineficaz
    razon: str = ""                   # por qué el asunto se resuelve así
    problema_que_decide: str = ""     # de qué problema cuelga el resultado
    efecto: str = ""                  # qué les pasa a los demás problemas
    apoyos: list = field(default_factory=list)
    confianza: str = ""               # alta | media | baja
    # LA OBJECIÓN, EN SUS PROPIAS PALABRAS. El secretario no necesita que le
    # den la razón: necesita ver por dónde se cae la propuesta para poder
    # razonarla. Una propuesta sin su contra se acepta por inercia, y es él
    # quien firma.
    en_contra: str = ""
    alcanza: bool = True

    # EL CONTEXTO, EN PROSA. Es lo primero que ve el secretario y sustituye al
    # volcado del acervo: cuatro párrafos —los hechos, lo que resolvió el
    # órgano, lo que se dice en contra, y cuál es el tema principal— que se
    # leen en un minuto y con los que ya se puede formar criterio. Antes se le
    # enseñaban ocho tesis y treinta preceptos y se perdía ahí.
    # {hechos, resolvio, combate, tema_principal}
    contexto: dict = field(default_factory=dict)

    # LA VÍA CONTRARIA, YA ESCRITA. Si el secretario no está de acuerdo con la
    # propuesta, marca lo contrario y la resolución alternativa aparece en el
    # acto: no espera a otra llamada. Se pide en la MISMA respuesta porque
    # cuesta unas decenas de palabras y ahorra una llamada entera con todo el
    # material otra vez.
    #
    # No es la propuesta negada: es cómo se SOSTENDRÍA la solución opuesta, con
    # su propia razón y su propio efecto sobre los accesorios.
    # {sentido, razon, efecto, apoyos}
    alternativa: dict = field(default_factory=dict)

    # LA LISTA DE COMPROBACIÓN. Para que no se quede un tema sin contestar: cada
    # uno con su suerte en las DOS vías. La exhaustividad es de las cosas que se
    # revisan de oficio, y un tema olvidado es un amparo de vuelta.
    # [{tema, papel, con_propuesta, con_alternativa, tema_distinto}]
    checklist: list = field(default_factory=list)

    def bloque(self) -> str:
        if not self.alcanza:
            return ("SIN PROPUESTA GLOBAL — el acervo no alcanza. "
                    + self.razon)
        return (f"EL ASUNTO: {self.sentido.upper()} ({self.confianza})\n"
                f"  Cuelga de: {self.problema_que_decide}\n"
                f"  {self.razon}\n"
                f"  Con los demás: {self.efecto}\n"
                f"  En contra: {self.en_contra}")


def _norm_problema(x: str) -> str:
    """Para emparejar propuesta y problema POR TEXTO, no por posición."""
    import unicodedata
    x = unicodedata.normalize("NFKD", str(x or "").lower())
    x = "".join(c for c in x if not unicodedata.combining(c))
    return " ".join("".join(c if c.isalnum() else " " for c in x).split())


# Palabras que aparecen en cualquier frase jurídica y no distinguen un tema de
# otro. Si entran en la comparación, todo se parece a todo.
_VACIAS = {
    "para", "por", "que", "los", "las", "del", "con", "una", "uno", "sus", "esta",
    "este", "como", "sobre", "ante", "pese", "haber", "hacia", "desde", "entre",
    "cuando", "porque", "sino", "pero", "aunque", "segun", "cual", "cuales",
    "debia", "debio", "podia", "pudo", "fue", "sido", "ser", "son", "era",
}


def _huella(x: str) -> set:
    """Las palabras que de verdad identifican un tema, recortadas a su raíz.

    A CINCO LETRAS, y no es capricho: el modelo escribe «suplencia» donde la
    fase escribió «suplir», «valoración» donde decía «valorar», «promoción»
    donde decía «promovido». Comparar palabras enteras las da por distintas;
    cinco letras las junta sin fundir cosas que no lo son —«sentencia» queda en
    «sente» y «sentido» en «senti»—.
    """
    return {w[:5] for w in _norm_problema(x).split()
            if len(w) >= 4 and w not in _VACIAS}


def _mismo_tema(a: str, b: str) -> bool:
    """¿Hablan del mismo tema, aunque estén escritos distinto?

    POR QUÉ NO BASTA COMPARAR EL PRINCIPIO. Es lo que hacía antes, y en el
    primer asunto real —la revisión 410/2026— falló en los tres temas. Las
    fases escriben el problema como PREGUNTA: «¿La quejosa podía reclamar la
    falta de emplazamiento pese a haber promovido el juicio de nulidad?». El
    modelo escribe el mismo tema como TÍTULO: «Falta de emplazamiento pese a la
    promoción del juicio de nulidad». No empiezan igual ni de lejos, y sin
    embargo son el mismo.

    El resultado fue peor que no comprobar nada: la lista de comprobación
    duplicó los tres temas y marcó como «SIN DETERMINAR» tres que el modelo sí
    había resuelto. Una comprobación que acusa a lo que está bien es la que
    está mal —van trece veces en este proyecto—.

    Se compara el CONTENIDO: qué proporción de las palabras con carga del más
    corto aparece en el otro.
    """
    ha, hb = _huella(a), _huella(b)
    if not ha or not hb:
        return False
    comunes = len(ha & hb)
    # MEDIDO, NO ELEGIDO. En la revisión 410/2026 el mismo tema, titulado por el
    # modelo frente a la pregunta de la fase, coincidió en 3 de 6 palabras con
    # carga: exactamente 0.50. Con el umbral en 0.55 se daba por no cubierto y
    # la lista lo duplicaba. Se baja a lo medido.
    #
    # Y no aflojar más: esto es sólo la RESERVA —lo que manda es el `numero` que
    # ahora se le pide al modelo—, y un umbral bajo aquí da por contestado un
    # tema que falta, que es el defecto que la lista existe para evitar.
    return comunes / min(len(ha), len(hb)) >= 0.50


def emparejar(problemas: list, propuestas: list) -> list:
    """La propuesta de CADA problema, en el orden de los problemas.

    POR QUÉ EXISTE ESTO. La pantalla hacía `propuestas[i]` para el problema i.
    Pero las dos listas no son la misma lista: el orden lo pone el modelo, y
    unas líneas más abajo este mismo fichero avisa de que puede devolver MENOS
    propuestas que problemas —hay un aviso escrito para ese caso—. Con tres
    problemas y dos propuestas, el secretario veía el sentido del problema B
    pegado al problema C.

    `Propuesta.problema` trae el texto de la pregunta. Ésa es la clave, y no se
    estaba usando. Se empareja por texto normalizado; lo que no case queda en
    None, que es honesto: mejor un hueco que un sentido ajeno.
    """
    libres = list(propuestas)
    fuera = []
    for p in problemas:
        preg = p.get("pregunta", "") if isinstance(p, dict) else str(p)
        clave = _norm_problema(preg)
        elegida = None
        if clave:
            for c in libres:
                if _norm_problema(c.problema) == clave:
                    elegida = c
                    break
            if elegida is None:                      # el modelo suele recortar
                for c in libres:                     # o reformular la pregunta
                    if _mismo_tema(c.problema, preg):
                        elegida = c
                        break
        if elegida is not None:
            libres.remove(elegida)
        fuera.append(elegida)
    return fuera


def _recorte_limpio(x: str, tope: int) -> str:
    """El último corte crudo del taller. Mismo criterio que los otros tres:
    por frontera de párrafo, nunca a mitad de frase, porque un corte a media
    oración es lo que llevó al modelo a escribir «el texto proporcionado se
    interrumpió» dentro de un considerando."""
    try:
        from fases123_pipeline import _cortar_bien
        return _cortar_bien(x or "", tope)
    except Exception:
        return (x or "")[:tope]


def _tesis_del_material(material, limite: int = MAX_TESIS_PROPUESTA) -> list:
    """Las tesis que se le enseñan, repartidas entre los problemas.

    Antes se ordenaba por obligatoriedad y se cortaba: las ocho primeras podían
    ser todas del primer problema, y los demás llegaban a la propuesta sin nada
    que invocar. De ahí salían los seis problemas sin apoyo del ADA 47/2025.

    Ahora se toman POR TURNOS —una del problema 1, una del 2, una del 3, y otra
    vuelta— con las obligatorias delante DENTRO de cada problema. Así el corte
    quita profundidad a todos por igual en vez de dejar a alguno en cero.
    """
    tesis = list(getattr(material, "tesis", []) or [])
    if not tesis:
        return []

    grupos: dict = {}
    for t in tesis:
        # Sin procedencia —material de una versión anterior, o de un solo
        # problema— todo cae en el mismo cajón y esto se comporta como antes.
        for n in (t.get("para") or [0]):
            grupos.setdefault(n, []).append(t)
    for g in grupos.values():
        g.sort(key=lambda t: not t.get("obligatoria"))

    fuera, vistos = [], set()
    vuelta = 0
    while len(fuera) < limite:
        metidas = 0
        for n in sorted(grupos):
            g = grupos[n]
            if vuelta < len(g):
                t = g[vuelta]
                if t.get("registro") not in vistos:
                    vistos.add(t.get("registro"))
                    fuera.append(t)
                    if len(fuera) >= limite:
                        break
                metidas += 1
        if not metidas:
            break
        vuelta += 1
    return fuera


def _indice_por_problema(tesis: list, cuantos: int) -> str:
    """Qué registros hay para cada problema, de un vistazo.

    El bloque de tesis va seguido y largo; con siete problemas, saber cuáles le
    tocan a cada uno exige releerlo entero. Este índice lo pone delante, y hace
    visible el caso que más importa: el problema que se quedó sin nada.
    """
    if not tesis or cuantos <= 1:
        return ""
    de: dict = {}
    for t in tesis:
        for n in (t.get("para") or []):
            de.setdefault(n, []).append(str(t.get("registro", "")))
    if not de:
        return ""
    filas = []
    for n in range(1, cuantos + 1):
        regs = de.get(n) or []
        filas.append(f"  Problema {n}: "
                     + (", ".join(regs) if regs
                        else "SIN CRITERIO EN EL ACERVO — dilo en su razón"))
    return "Qué hay para cada problema:\n" + "\n".join(filas) + "\n"


def _bloque_tesis(tesis: list) -> str:
    fuera = []
    for t in tesis:
        fuera.append(
            f"[registro {t.get('registro','')}] "
            f"{'OBLIGATORIA' if t.get('obligatoria') else 'orientadora'} · "
            f"{t.get('instancia','')}"
            # PARA QUÉ PROBLEMA SE BUSCÓ. Sin esto el modelo recibe un montón
            # indistinto y se queda con las dos primeras que le suenan.
            + (f" · responde al problema "
               + ", ".join(str(x) for x in t["para"])
               if t.get("para") else "")
            + "\n"
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

{_recorte_limpio(c, 20000)}
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
    # EL TIPO VIAJA CON EL MATERIAL, igual que en el estudio: son dos módulos
    # que reciben el mismo objeto y así no hay un parámetro que se olvide.
    import tipos_asunto as _ta_p
    _t5 = getattr(material, "tipo_asunto", "") or (
        "amparo_revision" if es_recurso else "amparo_directo")
    _voc5 = _ta_p.vocabulario_de(_t5)
    q = _voc5["combate"]
    # «LO QUE RESOLVIÓ LA RESPONSABLE» en mayúsculas, en los cuatro tipos. Esta
    # fase fija el SENTIDO, así que la etiqueta viaja de aquí a la razón toral
    # y de ahí al estudio: es de los sitios donde más caro sale.
    _org5 = _ta_p.sujetos_de(_t5)["organo"][0].upper()
    tesis = _tesis_del_material(material)
    lista = "\n".join(
        f"{i}. {p.get('pregunta','') if isinstance(p, dict) else str(p)}"
        # «La responsable resolvió» rotulaba cada problema jurídico en los
        # cuatro tipos, y esta fase es la que fija el SENTIDO: la etiqueta viaja
        # después a la razón toral y de ahí al estudio.
        + (f"\n   El órgano recurrido resolvió: {p.get('resolvio','')}"
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

LO QUE RESOLVIÓ {_org5}
{resumen_acto[:3000]}

LO QUE SE COMBATE
{resumen_conceptos[:3000]}

JURISPRUDENCIA DEL ACERVO — es TODO lo que puedes invocar
{_indice_por_problema(tesis, len(problemas))}
{_bloque_tesis(tesis)}

NORMAS DEL ACERVO
{_bloque_normas(material)}
{_bloque_suplencia(material)}
{_bloque_acervo_sentidos(material)}
{_bloque_contexto(contexto)}

CADA PROBLEMA SE RESUELVE CON LO QUE SE BUSCÓ PARA ÉL. Arriba, cada tesis dice
a qué problema responde, porque se buscó problema por problema. Antes de dar un
problema por resuelto, mira qué hay listado para él e INVÓCALO en sus «apoyos».
Medido sobre cinco asuntos reales: se recuperaron 191 tesis y sólo se invocaron
10, y hubo un asunto donde seis de siete problemas se calificaron sin un solo
criterio. Un problema resuelto sin apoyo es una opinión.

Y NO SE INVENTA LO QUE NO HAY. Si para un problema no aparece nada útil en la
lista, dilo en su razón con esas palabras —«el acervo no ofrece criterio para
esto»— y deja «apoyos» vacío. Eso es información para quien firma; rellenarlo
con un registro que trata de otra cosa es peor que dejarlo en blanco, porque
esconde el hueco en vez de enseñarlo.

CÓMO SE CALIFICA, y no son sinónimos:
- FUNDADO: el planteamiento combate la razón de la responsable y tiene razón.
- INFUNDADO: la combate y no tiene razón.
- ESENCIALMENTE FUNDADO: combate la razón toral y tiene razón EN LO
  SUSTANCIAL, aunque no en todos sus términos —se equivoca en un dato, en un
  precepto o en el alcance que pide—. Prospera igual que el fundado: lo que
  cambia es que el proyecto acota en qué medida. Es el 23% de los agravios en
  las revisiones que revocan de este circuito, medido sobre su acervo.
- SUSTANCIALMENTE FUNDADO: tiene razón en lo esencial de su planteamiento y
  eso basta. Prospera. Medido: aparece en asuntos favorables el 97% de las
  veces, más que el propio «fundado».
- PARCIALMENTE FUNDADO: tiene razón en una parte de lo que plantea y no en
  otra. Prospera en esa parte, y el proyecto acota cuál. Medido: 141 de sus
  365 apariciones están en asuntos que conceden PARCIALMENTE.
- FUNDADO PERO INSUFICIENTE: tiene razón Y AUN ASÍ NO ALCANZA, porque
  subsisten otras consideraciones que sostienen el sentido. NO PROSPERA: en el
  acervo aparece en asuntos favorables el 12% de las veces, igual que el
  infundado. Es la calificación honesta cuando el planteamiento acierta y el
  resultado no cambia; usarla en lugar de «infundado» reconoce el acierto sin
  mover el fallo.
- INATENDIBLE: no puede atenderse por CÓMO o CUÁNDO se plantea —es oscuro, no
  se entiende qué combate, o llega fuera del momento procesal—, no por lo que
  dice. Se distingue del inoperante: el inoperante SE ENTIENDE y no combate la
  razón toral; el inatendible ni siquiera puede examinarse.
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

6. EL CONTEXTO, EN PROSA Y EN CUATRO PÁRRAFOS. Es lo PRIMERO que lee el
   secretario y con eso forma su criterio, sin volver al expediente. Escribe:
   `hechos` (de qué va el asunto, qué pasó); `resolvio` (ABRE CON EL VERBO DEL
   DESENLACE —«sobreseyó», «negó el amparo», «concedió el amparo», «desechó»—,
   que es el dato que decide el resolutivo de este proyecto, y sólo DESPUÉS la
   razón. Salió esto: «tuvo por acreditado que la moral promovió el juicio
   mediante representante… concluyó que era parte actora», que cuenta el
   razonamiento y no dice en qué paró, y el resolutivo quedó en hueco); `combate` (qué dice en su contra el inconforme); y
   `tema_principal` (cuál es LA cuestión de la que depende el resultado, y por
   qué es ésa y no otra). Párrafos de verdad, en prosa llana, sin viñetas y sin
   tecnicismos de adorno. No repitas el expediente: sintetiza.
7. EL SENTIDO DEL ASUNTO ENTERO. Es una decisión distinta, no la suma de las
   anteriores: di de QUÉ problema cuelga el resultado y qué les pasa a los
   demás. LA REGLA: cuando el tema principal se resuelve, los accesorios
   SIGUEN SU SUERTE —si el principal prospera, quedan sin materia—, SALVO que
   sean temas DISTINTOS que exijan estudio propio, o que alguno pueda dar MÁS
   de lo que da el principal. Esa salvedad la marcas tú, tema por tema.
8. LA PROPUESTA LLEVA SU PROPIA OBJECIÓN. En `en_contra`, di en un renglón por
   dónde se cae: el mejor argumento de quien resolvería al revés. No es un
   formalismo. Quien lee esto es quien firma, y una propuesta sin su contra se
   acepta por inercia. Si de verdad no ves ninguna objeción seria, dilo así.
9. Y ESCRIBE TAMBIÉN LA VÍA CONTRARIA, en `alternativa`. Si el secretario no
   está de acuerdo con tu propuesta, marcará lo contrario y tiene que
   encontrarla ya escrita. NO es tu propuesta negada ni una advertencia de que
   te parece peor: es cómo se SOSTENDRÍA de verdad la solución opuesta —con su
   razón toral, sus apoyos del acervo, y qué les pasa a los accesorios en ESA
   vía, que casi nunca es lo mismo—. Escríbela como si la defendieras.
10. LA LISTA DE COMPROBACIÓN, en `checklist`: TODOS los temas del asunto, el
   principal y los accesorios, cada uno con su suerte en las dos vías. Existe
   para que no se quede ninguno sin contestar: un tema olvidado es un amparo de
   vuelta. No omitas ninguno de los problemas de arriba.
11. CADA ENTRADA DE LA LISTA LLEVA EL `numero` DEL PROBLEMA al que corresponde
   —1, 2, 3… tal como van numerados en LOS PROBLEMAS JURÍDICOS DEL ASUNTO—.
   Puedes titular el tema como quieras; el número es lo que permite saber a
   cuál te refieres sin adivinarlo por el texto.

Devuelve SÓLO un JSON, sin texto alrededor, con esta forma exacta:
{{"propuestas": [
  {{"problema": "<la pregunta, tal cual>",
    "sentido": "fundado|infundado|inoperante|ineficaz",
    "razon": "<la razón toral, {PALABRAS_RAZON} palabras>",
    "apoyos": ["<registro>", "..."],
    "confianza": "alta|media|baja",
    "alcanza": true}}
 ],
 "global": {{"sentido": "fundado|infundado|inoperante|ineficaz",
   "razon": "<por qué el ASUNTO se resuelve así, {PALABRAS_RAZON} palabras>",
   "problema_que_decide": "<la pregunta del problema del que cuelga>",
   "efecto": "<qué les pasa a los demás problemas, un renglón>",
   "apoyos": ["<registro>", "..."],
   "confianza": "alta|media|baja",
   "en_contra": "<el mejor argumento en contra, un renglón>",
   "alcanza": true,
   "contexto": {{
     "hechos": "<un párrafo>",
     "resolvio": "<empieza por el verbo: sobreseyó | negó el amparo | concedió el amparo | desechó; y luego la razón>",
     "combate": "<un párrafo>",
     "tema_principal": "<un párrafo: cuál es y por qué ése>"}},
   "alternativa": {{
     "sentido": "<el contrario al de arriba>",
     "razon": "<cómo se sostendría, {PALABRAS_RAZON} palabras>",
     "efecto": "<qué les pasa a los accesorios en ESTA vía>",
     "apoyos": ["<registro>", "..."]}},
   "checklist": [
     {{"numero": <el numero del problema en la lista de arriba: 1, 2, 3...>,
       "tema": "<el tema, en una línea>",
       "papel": "principal|accesorio",
       "con_propuesta": "<su suerte si se sigue la propuesta>",
       "con_alternativa": "<su suerte si se sigue la alternativa>",
       "tema_distinto": false}}
   ]}}}}"""


_RX_JSON = re.compile(r"\{.*\}", re.S)


def _leer(crudo: str) -> tuple:
    """El JSON del modelo, tolerante a que lo envuelva en explicaciones.

    Devuelve (propuestas, global). El global puede venir vacío: los modelos
    omiten campos, y un asunto sin propuesta global se atiende —el secretario
    fija el sentido a mano, como siempre—. Lo que NO se hace es fabricarlo
    tomando el de un problema: eso es exactamente lo que se vino a corregir.
    """
    m = _RX_JSON.search(crudo or "")
    if not m:
        return [], {}
    try:
        datos = json.loads(m.group(0))
    except Exception:
        return [], {}
    g = datos.get("global")
    return (datos.get("propuestas") or []), (g if isinstance(g, dict) else {})


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

    # ── CUÁNTOS CRITERIOS DISTINTOS SOSTIENEN TODO ESTO ───────────────────
    # Medido sobre cinco asuntos reales: el 410-2026 sale con cinco criterios
    # distintos para cuatro problemas, y el 650-2025 con UNO para cinco. Nada
    # lo advertía.
    #
    # Y NO ES UN REPROCHE. En suspensión, los artículos 128 y 147 de la Ley de
    # Amparo SON el fundamento, y puede que no haya más criterio aplicable.
    # Por eso esto informa y no acusa: David lo dijo —«una sentencia se sostiene
    # por su argumentación y por los criterios vinculantes que invoca»—, así
    # que el dato tiene que estar a la vista de quien firma, sin decirle que
    # está mal.
    _vivas = [p for p in propuestas if p.alcanza]
    if len(_vivas) >= 3:
        _distintos = set()
        for p in _vivas:
            for a in p.apoyos:
                m = _RX_CIFRA_REGISTRO.search(str(a))
                if m:
                    _distintos.add(m.group(1))
        if len(_distintos) <= 1:
            avisos.append(
                f"Los {len(_vivas)} problemas se apoyan en "
                + (f"UN SOLO criterio ({next(iter(_distintos))})"
                   if _distintos else "NINGÚN criterio")
                + ", y el resto es ley. Puede estar bien —hay materias donde la "
                  "ley es el fundamento— pero conviene mirarlo antes de firmar.")
    return avisos


def completar_checklist(checklist: list, problemas: list) -> tuple[list, list]:
    """Que no falte ningún tema. Comprobado, no pedido.

    La lista de comprobación existe para garantizar exhaustividad, y una
    garantía que depende de que el modelo se acuerde de todos los temas no es
    una garantía. Aquí se compara contra los problemas que las fases
    extrajeron del expediente: lo que el modelo omitió se añade con el hueco
    declarado, y se avisa.

    Vale la pena decir por qué importa tanto: la exhaustividad se revisa de
    oficio, y un tema sin contestar es un amparo de vuelta. Es de los pocos
    defectos que no se ven leyendo el proyecto —lo que falta no se lee—.
    """
    # EL NÚMERO MANDA, EL TEXTO ES LA RESERVA. Comparar títulos con preguntas
    # es adivinar: en la revisión 410/2026 el modelo tituló el mismo tema
    # «Suplencia de la queja y valoración de la sentencia del amparo 323/2023»
    # frente a la pregunta «¿El Juzgado de Distrito debía suplir la deficiencia
    # de la queja y valorar la sentencia ofrecida como prueba?», y la
    # coincidencia se quedó en el 50%, justo debajo del umbral. Subir el umbral
    # arregla ese caso y rompe el siguiente.
    #
    # No hace falta adivinar: los problemas van NUMERADOS en el prompt y el
    # modelo sólo tiene que devolver el número. Se le pide, y el texto queda
    # para cuando lo omita.
    por_numero = set()
    for c in checklist:
        try:
            n_ = int(c.get("numero"))
        except (TypeError, ValueError):
            continue
        if 1 <= n_ <= len(problemas):
            por_numero.add(n_)

    vistos = [str(c.get("tema", "")) for c in checklist if c]
    faltan, fuera = [], list(checklist)
    for i, q in enumerate(problemas, 1):
        preg = q.get("pregunta", "") if isinstance(q, dict) else str(q)
        clave = _norm_problema(preg)
        if not clave:
            continue
        if i in por_numero:
            continue
        # Sin número, se cae al texto: mismo tema aunque esté escrito distinto.
        if any(_mismo_tema(v, preg) for v in vistos):
            continue
        faltan.append(preg)
        fuera.append({"numero": i, "tema": preg, "papel": "accesorio",
                      "con_propuesta": "SIN DETERMINAR — el motor no lo incluyó.",
                      "con_alternativa": "SIN DETERMINAR — el motor no lo incluyó.",
                      "tema_distinto": False})
    avisos = []
    if faltan:
        avisos.append(
            f"La lista de comprobación omitía {len(faltan)} tema(s): "
            f"{'; '.join(t[:80] for t in faltan)}. Se añadieron sin suerte "
            f"determinada: decídela antes de generar, o el proyecto saldrá sin "
            f"contestarlos.")
    return fuera, avisos


def revisar_global(glob, material) -> list:
    """Los apoyos de la propuesta del ASUNTO, comprobados como los demás.

    `revisar()` recorría las propuestas POR PROBLEMA y dejaba fuera la global y
    su alternativa —eran nuevas y nadie las metió en el recorrido—. Es
    justamente donde más caro sale: la global es la que el secretario acepta de
    un botón, y sus registros son los que acaban en el proyecto.

    Medido en la revisión 410/2026: el registro que citó la global (2020441) sí
    estaba en el acervo. Que saliera bien no significa que estuviera
    comprobado, y esa distinción es la que este proyecto lleva toda la semana
    aprendiendo.
    """
    validos = {str(t.get("registro", "")) for t in getattr(material, "tesis", []) or []}
    avisos = []
    for etiqueta, apoyos in (("La propuesta del asunto", glob.apoyos),
                             ("La vía alternativa", (glob.alternativa or {}).get("apoyos") or [])):
        inventados = []
        for a in apoyos:
            m = _RX_CIFRA_REGISTRO.search(str(a))
            if m and m.group(1) not in validos:
                inventados.append(str(a))
        if inventados:
            avisos.append(
                f"{etiqueta} se apoya en registros que NO están en el acervo: "
                f"{inventados}. No se citan hasta comprobarlos en el Semanario.")
    return avisos


async def proponer(cliente, problemas: list, material, resumen_acto: str = "",
                   resumen_conceptos: str = "", es_recurso: bool = False,
                   contexto: str = "") -> tuple[list, object, list]:
    """Devuelve (propuestas, global, avisos). No decide nada: propone.

    El GLOBAL es la propuesta del asunto entero y sale de la MISMA llamada: es
    una decisión distinta de las de cada problema, pero pedirla aparte sería
    pagar dos veces por el mismo material.
    """
    if not problemas:
        return [], Global(alcanza=False, razon="No hay problemas que resolver."), []
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
    import llamada_modelo as _lm
    r = await _lm.crear(cliente, **kw)
    crudo = (r.choices[0].message.content or "").strip()

    # SI NO VUELVE NADA, HAY QUE PODER SABER POR QUÉ. Una lista vacía puede
    # ser «el modelo no respondió» o «respondió algo que no supe leer», y son
    # dos averías distintas. Se distinguen aquí y no adivinando en los logs.
    leidas, crudo_global = _leer(crudo)
    if not leidas:
        motivo = ("el modelo no devolvió texto —probablemente agotó el "
                  "presupuesto razonando—" if not crudo.strip()
                  else f"la respuesta no traía el JSON esperado: «{crudo[:200]}»")
        print(f"   ⚖️ PROPUESTA sin resultado ({MODELO_PROPUESTA}): {motivo}")
        return [], Global(alcanza=False, razon=motivo), [
            f"El motor no propuso ningún sentido: {motivo}. "
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

    # LA PROPUESTA DEL ASUNTO. Si el modelo la omitió, se queda sin ella y se
    # avisa: NO se rellena con la del problema principal, que es el defecto que
    # esta ronda vino a corregir. Un hueco declarado es honesto; un sentido
    # ajeno con etiqueta de global, no.
    g = crudo_global or {}
    glob = Global(
        sentido=str(g.get("sentido", "")).strip().lower(),
        razon=str(g.get("razon", ""))[:900],
        problema_que_decide=str(g.get("problema_que_decide", ""))[:400],
        efecto=str(g.get("efecto", ""))[:400],
        apoyos=[str(a) for a in (g.get("apoyos") or [])][:6],
        confianza=str(g.get("confianza", "")).strip().lower(),
        en_contra=str(g.get("en_contra", ""))[:400],
        alcanza=bool(g.get("alcanza", True)) and bool(g.get("sentido")),
        contexto={k: str((g.get("contexto") or {}).get(k, ""))[:1800]
                  for k in ("hechos", "resolvio", "combate", "tema_principal")},
        alternativa={
            "sentido": str((g.get("alternativa") or {}).get("sentido", "")).strip().lower(),
            "razon": str((g.get("alternativa") or {}).get("razon", ""))[:900],
            "efecto": str((g.get("alternativa") or {}).get("efecto", ""))[:400],
            "apoyos": [str(a) for a in ((g.get("alternativa") or {}).get("apoyos") or [])][:6],
        })

    # LA LISTA SE COMPLETA CONTRA LOS PROBLEMAS REALES, no contra la memoria
    # del modelo. Si omitió un tema, se añade con la suerte sin determinar.
    glob.checklist, _av_lista = completar_checklist(
        [c for c in (g.get("checklist") or []) if isinstance(c, dict)], problemas)

    # Si el modelo devolvió menos propuestas que problemas, faltan: se dice.
    avisos = revisar(fuera, material)
    if len(fuera) < len(problemas):
        avisos.append(
            f"Se propusieron {len(fuera)} sentidos para {len(problemas)} "
            f"problemas. Los que faltan quedan sin propuesta.")
    avisos.extend(_av_lista)
    if glob.alcanza:
        avisos.extend(revisar_global(glob, material))
    if not glob.alcanza:
        avisos.append(
            "El motor no propuso una solución para el asunto entero: sólo por "
            "problema. Fija tú el sentido global.")
    else:
        if glob.sentido not in SENTIDOS:
            avisos.append(f"Sentido global no reconocido: «{glob.sentido}».")
        if not any((glob.contexto or {}).values()):
            avisos.append(
                "El motor no escribió el contexto del asunto. Lo tienes en el "
                "adelanto que ya generaste.")
        _alt = glob.alternativa or {}
        if not _alt.get("razon"):
            avisos.append(
                "No hay vía alternativa escrita: si no estás de acuerdo con la "
                "propuesta, tendrás que razonar el sentido contrario tú.")
        elif _alt.get("sentido") == glob.sentido:
            avisos.append(
                f"La «alternativa» vino con el MISMO sentido que la propuesta "
                f"(«{glob.sentido}»): no es una vía contraria. Ignórala.")
    return fuera, glob, avisos


def resumen(propuestas: list) -> str:
    """Lo que se le enseña al secretario, de un vistazo."""
    if not propuestas:
        return "Sin propuesta: el acervo no alcanzó para sugerir un sentido."
    return "\n\n".join(p.bloque() for p in propuestas)


def calificaciones_de(propuestas: list) -> list:
    """Los sentidos, en orden, para el resolutivo. Sólo los que alcanzan."""
    return [p.sentido for p in propuestas if p.alcanza and p.sentido in SENTIDOS]
