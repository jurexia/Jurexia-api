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
# ── SIN OBSTÁCULOS A LA LECTURA ──────────────────────────────────────────
# David: «yo jamás puse límites. Al contrario, se trata de una labor seria. La
# lectura debe ser ilimitada. No pongas obstáculos a esta lectura porque el
# proyecto será deficiente.»
#
# Tiene razón, y los topes que había no venían de una restricción real: venían
# de haber medido un caso concreto y haber puesto la pared justo detrás. Cada
# vez que llegó un expediente mayor, la pared amputó.
#
# Medido contra el motor de verdad (gpt-5.6-luna): 153.256 caracteres = 38.314
# tokens, respondió sin despeinarse. Estos valores están un orden de magnitud
# por encima de cualquier expediente que haya pasado por aquí, y existen sólo
# como freno contra un fichero corrupto de gigabytes, no como criterio.
TOPE_CARACTERES = 600_000
# EL ESCRITO DE LA PARTE VA APARTE, Y MÁS ANCHO. De él dependen la congruencia
# y la exhaustividad: un concepto que no entra es un concepto que no se
# contesta, y eso es un vicio de la sentencia. Medido contra el motor real
# (gpt-5.6-luna) con el escrito entero del ADC 536/2025: 153.256 caracteres,
# 38.314 tokens, respondió sin despeinarse. El tope de 100.000 no protegía de
# nada; sólo amputaba.
TOPE_CONCEPTOS = 600_000


# El rótulo del bloque resolutivo, con y sin espaciado judicial.
_RX_RESUELVE = re.compile(
    r"\n\s*R\s*E\s*S\s*U\s*E\s*L\s*V\s*E\s*[:.]?\s*\n|"
    r"\bpor\s+lo\s+(?:expuesto|anteriormente\s+expuesto)[^.]{0,90}se\s+resuelve",
    re.I)


# LO QUE SE TIRÓ, CONTADO. El recorte se perdía en silencio: medido sobre mil
# páginas de OCR —3.082.990 caracteres— llegaban 100.000, o sea el 3,2 % del
# expediente, y nada lo decía. Ni un aviso, ni una cabecera, ni una línea en el
# registro. Y el silencio era deliberado por partida doble: el prompt PROHÍBE
# al modelo mencionar que le falta material, para que no escriba «no cuento con
# elementos suficientes» en mitad de una sentencia.
#
# Esa prohibición se queda: un documento no debe comentarse a sí mismo. Pero el
# secretario tiene que saberlo, así que el dato viaja POR FUERA, en los avisos.
# Lo apunta aquí quien corta, que es el único que sabe cuánto.
DESCARTADO: dict = {}


def _cortar_bien(cuerpo: str, tope: int, que: str = "",
                 guardar_resolutivos: bool = True) -> str:
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
    if que:
        _ya = DESCARTADO.get(que)
        # Se conserva el TOTAL más grande: si la cola ya recortó, el documento
        # de verdad era el de antes, no la rebanada que llega aquí.
        DESCARTADO[que] = (max(len(cuerpo), _ya[0] if _ya else 0), tope)

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
    # ── LA REGLA DEL RESUELVE ES DE SENTENCIAS, NO DE ESCRITOS ───────────
    #
    # Reservar el último cuarto para el bloque resolutivo es correcto cuando lo
    # que se recorta es una SENTENCIA: ahí, el final es lo que se confirma o se
    # revoca. Aplicado al escrito de la parte es un desastre, y explica con
    # aritmética exacta por qué el proyecto contestaba «hasta el sexto»:
    #
    # una demanda de amparo directo TRANSCRIBE la sentencia reclamada, así que
    # el rótulo «R E S U E L V E» aparece EN MEDIO del escrito. Todo lo que va
    # después —que es justamente el apartado de conceptos de violación— se
    # quedaba con int(tope*0.25) = 25.000 caracteres, unas catorce páginas. Los
    # conceptos del final se tiraban por el agujero de en medio, sin marca y
    # sin aviso.
    m = _RX_RESUELVE.search(cuerpo) if guardar_resolutivos else None
    if m:
        resolutivos = cuerpo[m.start():][:int(tope * 0.25)]
        cabeza = _hasta(cuerpo[:m.start()], tope - len(resolutivos))
        # Sin marca de omisión en medio: un marcador es una invitación a
        # comentarlo, y comentar el estado del documento DENTRO de la sentencia
        # es justo lo que no puede pasar. El rótulo del resuelve separa solo.
        return cabeza + "\n\n" + resolutivos
    return _hasta(cuerpo, tope)


def texto_entero(texto: str, tope: int, que: str = "") -> str:
    """EL DOCUMENTO ENTERO, que es lo que lee el modelo desde el 17-sep-2026.

    David: «El modelo debe leer todo, ya hemos establecido que no hay límites
    en lectura.» Hasta hoy los resúmenes recibían el documento desde su
    marcador —«ESTUDIO DE FONDO», «CONCEPTOS DE VIOLACIÓN»— y la cabeza se
    tiraba; en un asunto de un tester entraron 79.953 de 100.220 caracteres
    del acto y 43.174 de 49.566 del escrito, con el aviso «NO SE LEYÓ …
    ENTERO» arriba del todo. Medido: 153.256 caracteres son 38.314 tokens y
    el motor responde sin despeinarse. Sólo queda el freno contra un fichero
    corrupto (`tope`), que es el único caso en que se avisa.
    """
    return _cortar_bien(texto or "", tope, que)


def recortar_acto(texto: str, tope: int = TOPE_CARACTERES) -> str:
    """Del acto reclamado, su ESTUDIO DE FONDO y los resolutivos.

    YA NO ES LO QUE LEE EL MODELO —ver `texto_entero`—: es el ENFOQUE que usan
    las cuentas (cuántas palabras pedir, qué tesis de la responsable hay que
    nombrar), que sí deben mirar el estudio y no la transcripción de los
    escritos de las partes en los resultandos.

    Se prefiere la marca del estudio; sólo si no aparece se cae al primer
    considerando, y en último caso a la cola del documento.
    """
    m = _MARCAS_ESTUDIO.search(texto) or _MARCAS_CONSIDERANDO.search(texto)
    # AQUÍ SE PIERDE LA MAYOR PARTE, no en `_cortar_bien`. Cuando no aparece la
    # marca del estudio se coge la COLA del documento, y esa rebanada ya viene
    # del tamaño del tope: `_cortar_bien` la ve y no tiene nada que recortar,
    # así que la primera versión de este contador no vio nada. Lo que se tiró
    # se cuenta donde se tira.
    cuerpo = texto[m.start():] if m else texto[-tope:]
    return _cortar_bien(cuerpo, tope)


def recortar_conceptos(texto: str, tope: int = TOPE_CONCEPTOS) -> str:
    """Del escrito de la parte, el apartado de conceptos o agravios.

    NO se le aplica la regla del bloque resolutivo. Aquí lo valioso está en la
    COLA —los conceptos se numeran hasta el final— y reservar un cuarto del
    presupuesto para un «RESUELVE» que en este documento es una CITA de la
    sentencia transcrita era lo que amputaba los últimos conceptos.
    """
    m = _MARCAS_CONCEPTOS.search(texto)
    cuerpo = texto[m.start():] if m else texto[-tope:]
    return _cortar_bien(cuerpo, tope, guardar_resolutivos=False)




# ═══════════════════════════════════════════════════════════════════════════
# NINGÚN CONCEPTO SE QUEDA SIN RESUMIR
#
# David, con un proyecto delante: «no me resumió ni me contestó los conceptos
# de violación posteriores al SEXTO. Había más de 6. Es inaceptable que se
# limite o fragmente un documento.»
#
# Tenía razón y el vicio es de los graves: un proyecto que no contesta todos
# los conceptos incurre en falta de CONGRUENCIA y EXHAUSTIVIDAD. No es un
# defecto de calidad, es un vicio de la sentencia.
#
# Y el mecanismo tenía DOS filos que se sumaban:
#   1. `recortar_conceptos` cortaba el escrito a TOPE_CARACTERES, así que los
#      conceptos del final no llegaban al resumidor.
#   2. `_NUCLEO` le PROHÍBE al modelo decir que el documento está incompleto
#      —con razón: un «el texto se interrumpió» dentro de una sentencia es un
#      desastre—. Sumadas, el modelo resumía lo que veía y callaba lo que
#      faltaba. Silencio por diseño, encima de una amputación.
#
# La solución no es subir el tope: eso sólo mueve la pared, y ya se movió una
# vez —de 60.000 a 100.000 para cubrir un caso de 90.126—. Se cuentan los
# conceptos que el escrito TIENE, se comprueban contra los que el resumen
# TRAJO, y lo que falte se pide en otra pasada. El tope deja de decidir cuántos
# conceptos existen.
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# CONTAR LOS PLANTEAMIENTOS: SE MUDÓ A SU PROPIO MÓDULO
# ═══════════════════════════════════════════════════════════════════════════
# Aquí vivían `_ORDINALES`, `_RX_RUBRICA`, `rubricas_de_conceptos` y
# `conceptos_sin_resumir`. Se retiran enteros, medidos: sobre los SIETE
# escritos que el taller guarda completos hay 43 conceptos de violación o
# agravios, y ese contador veía 13 —el 30%—. Cuatro de los siete devolvían
# CERO, y son justo los que rotulan con una cabecera de sección y debajo
# ordinales a secas, que es como escribe media judicatura.
#
# Dos fallos y el segundo era peor que el primero:
#   · exigía la palabra CONCEPTO o AGRAVIO pegada al ordinal;
#   · y sus ordinales acababan en DUODÉCIMO, así que «DÉCIMO SÉPTIMO CONCEPTO»
#     se leía «SÉPTIMO CONCEPTO», el conjunto de vistos lo tragaba como
#     repetido, y después la comprobación lo daba por resumido porque
#     «SÉPTIMO» sí estaba en el resumen. No es que no lo viera: daba un VISTO
#     BUENO FALSO.
# Consecuencia medida: el aviso «NO SE RESUMIERON TODOS LOS PLANTEAMIENTOS»
# no ha saltado ni una vez en 101 sesiones.
#
# El contador nuevo no busca un rótulo: busca una CADENA —ordinales 1,2,3…N en
# orden, dentro de la sección, descartando las enumeraciones transcritas y el
# petitorio—. 43 de 43 sobre el mismo corpus. Y cuando no puede contar lo dice:
# «no_contado» no es «no falta nada».
from contador_planteamientos import (  # noqa: E402
    aviso as aviso_de_cobertura,
    planteamientos,
    sin_resumir,
    sospecha_de_amputacion,
)


def rubricas_de_conceptos(texto: str) -> list:
    """Los rótulos que el escrito usa para separar un planteamiento de otro."""
    return list(planteamientos(texto).get("rubricas") or [])


def conceptos_sin_resumir(texto_fuente: str, resumen: str) -> list:
    """Los que el escrito trae y el resumen no menciona. Sólo los DUROS.

    Los «sin_rotular» —que el contador sospecha pero no puede nombrar— salen
    por el aviso, nunca por el reintento: pedirle al modelo que resuma algo que
    no se sabe nombrar gasta tres pasadas y no añade nada.
    """
    return list(sin_resumir(texto_fuente, resumen).get("faltan") or [])


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


def objetivo_acto(texto_acto: str) -> int:
    """Cuántas palabras pide la resolución: la mediana del corpus como suelo,
    un tope, y entre los dos proporcional a su estudio de fondo."""
    n = len(" ".join(recortar_acto(texto_acto or "").split()))
    return max(PALABRAS_RESUMEN_ACTO,
               min(PALABRAS_CONCEPTOS_TOPE, n // CARACTERES_POR_PALABRA_RESUMEN))


def prompt_acto_a_fondo(texto_acto: str, resumen_actual: str, faltan_citas: list,
                        objetivo: int, es_recurso: bool = False,
                        tipo_asunto: str = "") -> str:
    """La segunda pasada del resumen del acto: corto, o sin las tesis en que la
    responsable se apoyó. Devuelve el resumen ENTERO reescrito."""
    import tipos_asunto as _tap
    _t = tipo_asunto or ("amparo_revision" if es_recurso else "amparo_directo")
    que = _tap.vocabulario_de(_t)["recurrido"]
    _citas = ("\n".join(f"  - {c}" for c in faltan_citas)
              if faltan_citas else "  (ninguna pendiente)")
    return f"""{_NUCLEO}

{instrucciones_resumen_acto(_t, objetivo)}

Se trata de la {que}. Éste es su texto:
──────────────────────────────────────────
{texto_entero(texto_acto, TOPE_CARACTERES, "el acto reclamado")}
──────────────────────────────────────────

Y éste es el resumen que se escribió, que SE QUEDÓ CORTO: tiene
{len((resumen_actual or '').split())} palabras y la resolución pide alrededor de
{objetivo}; deja fuera consideraciones de fondo, y no nombra estas tesis en
que la responsable se apoyó:
{_citas}

──────────────────────────────────────────
{resumen_actual}
──────────────────────────────────────────

Reescribe el resumen ENTERO y completo: todas las consideraciones de fondo,
una decisión por frase, en pretérito, con sus marcas [[p.N §M]] donde puedas
ubicar la página, y cada tesis en que se apoyó nombrada donde la usó.
Devuelve sólo el resumen."""



def _bloque_tesis_a_nombrar(citas, quien: str) -> str:
    """La lista de tesis que el resumen tiene que nombrar, contada ANTES.

    POR QUÉ EN LA PRIMERA VUELTA. Revisión fiscal 2/2026, 17-sep-2026: el
    primer resumen de los agravios salió con 1,375 palabras y nombraba una de
    las seis tesis que invoca la recurrente, así que se reescribió entero —120
    segundos de los 240 del adelanto— sólo para añadir las cinco que faltaban.
    Esas tesis se cuentan con un regex, sin modelo: dárselas desde el principio
    es pedir lo mismo que pide la segunda vuelta, una vuelta antes. La segunda
    vuelta se queda como red: si aun así faltan, reescribe como siempre.

    Es una LISTA DE COMPROBACIÓN de este asunto, no un ejemplo: no hay texto
    que copiar, sólo claves que tienen que aparecer.
    """
    cs = sorted(citas or [])
    if not cs:
        return ""
    return ("\nLAS TESIS QUE " + quien + " INVOCA —contadas en el documento; cada una "
            "tiene que quedar NOMBRADA en el párrafo del argumento que apoya, con su "
            "clave o su registro tal como aparece:\n"
            + "\n".join(f"  - {c}" for c in cs) + "\n")

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

{instrucciones_resumen_acto(_t, objetivo_acto(texto_acto))}

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
{texto_entero(texto_acto, TOPE_CARACTERES, "el acto reclamado")}
──────────────────────────────────────────
{_bloque_tesis_a_nombrar(citas_invocadas(recortar_acto(texto_acto)), "LA AUTORIDAD")}
Escribe el resumen. Sólo el resumen, sin preámbulo ni rótulo."""


# ═══════════════════════════════════════════════════════════════════════════
# LA SÍNTESIS A FONDO — revisión fiscal 61/2025 (David, 15-sep-2026)
# ═══════════════════════════════════════════════════════════════════════════
# El escrito de la autoridad tenía 68.776 caracteres en un agravio ÚNICO y
# ocho jurisprudencias invocadas. El resumen salió de 147 palabras: tres
# párrafos sobre la ÚLTIMA página —la petición de revocar, las pruebas
# ofrecidas y los delegados— y ni una de las ocho tesis. El contador de
# planteamientos lo daba por bien resumido: había un agravio y había un
# apartado. Es el hueco del contador —mide que estén, no que estén ENTEROS—.
#
# David: «el resumen de los agravios fue excesivamente corto, no se
# parafraseó ni sintetizó todo el aspecto técnico que hizo valer la
# autoridad, ni las jurisprudencias que citó».
#
# Tres piezas: el objetivo de extensión sale del ESCRITO y no de la mediana
# del corpus; las tesis que la parte invoca se cuentan y se comprueban en el
# resumen; y si falta hondura o faltan tesis, una segunda pasada reescribe
# el resumen entero con el escrito delante.
PALABRAS_CONCEPTOS_TOPE = 1600
CARACTERES_POR_PALABRA_RESUMEN = 55     # 68.776 caracteres → ~1.250 palabras


def objetivo_conceptos(texto_conceptos: str) -> int:
    """Cuántas palabras pide el escrito: la mediana del corpus como suelo y
    un tope, y entre los dos, proporcional a lo que la parte escribió."""
    n = len(" ".join((texto_conceptos or "").split()))
    return max(PALABRAS_RESUMEN_CONCEPTOS,
               min(PALABRAS_CONCEPTOS_TOPE, n // CARACTERES_POR_PALABRA_RESUMEN))


_RX_CLAVE_TESIS = re.compile(
    r"\b(?:[1-2]a\.|P\.|[IVX]+\.\d*[oa]?\.[A-Z]?\.?)\s*/?\s*J\.?\s*\d{1,4}/\d{4}"
    r"|\b(?:[1-2]a\.|P\.)\s*[A-Z]{1,6}/\d{4}", re.I)
_RX_REGISTRO_TESIS = re.compile(r"\bregistro\b[^0-9]{0,25}(\d{6,7})", re.I)


def citas_invocadas(texto: str) -> set:
    """Las tesis que un escrito invoca, por clave («2a./J. 60/2007») y por
    registro («registro 172239»). Es lo que el resumen tiene que nombrar."""
    t = " ".join((texto or "").split())
    claves = {" ".join(m.group(0).split()).replace(" /", "/").replace("/ ", "/")
              for m in _RX_CLAVE_TESIS.finditer(t)}
    regs = {m.group(1) for m in _RX_REGISTRO_TESIS.finditer(t)}
    return {c.upper() for c in claves} | regs


def citas_sin_nombrar(texto_conceptos: str, resumen: str) -> tuple:
    """(las del escrito, las que el resumen no nombra)."""
    todas = citas_invocadas(texto_conceptos)
    r = " ".join((resumen or "").split()).upper()
    faltan = set()
    for c in todas:
        if c.isdigit():
            if c not in r:
                faltan.add(c)
        else:
            nucleo = re.sub(r"\s+", "", c)
            if nucleo not in re.sub(r"\s+", "", r):
                faltan.add(c)
    return todas, faltan


def prompt_resumen_conceptos(texto_conceptos: str, es_recurso: bool = False,
                             tipo_asunto: str = "") -> str:
    import tipos_asunto as _tap
    _t = tipo_asunto or ("amparo_revision" if es_recurso else "amparo_directo")
    q = _tap.vocabulario_de(_t)["combate"]
    return f"""{_NUCLEO}

{instrucciones_resumen_conceptos(es_recurso, _t, objetivo_conceptos(texto_conceptos))}

Éste es el escrito de la parte:

──────────────────────────────────────────
{texto_entero(texto_conceptos, TOPE_CONCEPTOS, "el escrito de la parte")}
──────────────────────────────────────────
{_bloque_tesis_a_nombrar(citas_invocadas(texto_conceptos), "LA PARTE")}
Escribe el resumen de los {q}: un apartado por cada uno y, dentro de cada
apartado, un párrafo por cada argumento distinto. Sólo el resumen."""


def prompt_conceptos_a_fondo(texto_conceptos: str, resumen_actual: str,
                             faltan_citas: list, objetivo: int,
                             es_recurso: bool = False, tipo_asunto: str = "",
                             tramos: list = None) -> str:
    """La segunda pasada por HONDURA: el resumen existe pero es corto o no
    nombra las tesis invocadas. Se devuelve el resumen ENTERO reescrito, no un
    apéndice: los argumentos van en el apartado que les toca."""
    import tipos_asunto as _tap
    _t = tipo_asunto or ("amparo_revision" if es_recurso else "amparo_directo")
    q = _tap.vocabulario_de(_t)["combate"]
    _cuerpo = texto_conceptos
    if tramos:
        try:
            _cuerpo = "\n\n[…]\n\n".join(
                texto_conceptos[a:b] for a, b in tramos if b > a)
        except Exception:
            _cuerpo = texto_conceptos
    _citas = ("\n".join(f"  - {c}" for c in faltan_citas)
              if faltan_citas else "  (ninguna pendiente)")
    return f"""{_NUCLEO}

{instrucciones_resumen_conceptos(es_recurso, _t, objetivo)}

Éste es el escrito de la parte:
──────────────────────────────────────────
{_cortar_bien(_cuerpo, TOPE_CONCEPTOS, "el escrito de la parte", guardar_resolutivos=False)}
──────────────────────────────────────────

Y éste es el resumen que se escribió, que SE QUEDÓ CORTO: tiene
{len((resumen_actual or '').split())} palabras y el escrito pide alrededor de
{objetivo}; deja fuera argumentos técnicos, y no nombra estas tesis que la
parte invoca:
{_citas}

──────────────────────────────────────────
{resumen_actual}
──────────────────────────────────────────

Reescribe el resumen de los {q} ENTERO, con la misma estructura —la bisagra,
un apartado por {q[:-1] if q.endswith('s') else q}, en su orden— pero completo:
cada argumento distinto con su párrafo y su aspecto técnico, y cada tesis
invocada nombrada en el párrafo del argumento que apoya. Conserva las marcas
[[p.N §M]] donde ya estaban y añádelas a los apartados nuevos si puedes ubicar
la página. Devuelve sólo el resumen."""


def prompt_conceptos_que_faltan(texto_conceptos: str, faltan: list,
                                es_recurso: bool = False,
                                tipo_asunto: str = "",
                                tramos: list = None) -> str:
    """La segunda pasada: sólo los planteamientos que el resumen dejó fuera.

    Se le da el escrito ENTERO —sin el recorte que causó el problema— y se le
    nombran uno por uno los que tiene que resumir. Nombrarlos importa: pedirle
    «los que falten» le deja decidir cuáles, y ya sabemos qué decide.
    """
    import tipos_asunto as _tap
    _t = tipo_asunto or ("amparo_revision" if es_recurso else "amparo_directo")
    q = _tap.vocabulario_de(_t)["combate"]
    _lista = "\n".join(f"  - {x}" for x in faltan)
    # SÓLO LOS TROZOS DONDE VIVEN LOS QUE FALTAN. Mandar el escrito entero en
    # la segunda pasada cuesta lo mismo que en la primera y no añade nada: el
    # contador ya sabe dónde empieza y acaba cada planteamiento. Medido en el
    # 393/2025: 43.796 caracteres en vez de 178.842, el 25%.
    _cuerpo = texto_conceptos
    if tramos:
        try:
            _cuerpo = "\n\n[…]\n\n".join(
                texto_conceptos[a:b] for a, b in tramos if b > a)
        except Exception:
            _cuerpo = texto_conceptos
    return f"""{_NUCLEO}

{instrucciones_resumen_conceptos(es_recurso, _t)}

Éste es el escrito de la parte:
──────────────────────────────────────────
{_cuerpo}
──────────────────────────────────────────

De todos los {q} que contiene, resume ÚNICAMENTE estos, que quedaron fuera de
un resumen anterior:
{_lista}

Un párrafo por cada uno, en el mismo registro que el resto del resumen, y en el
orden en que aparecen en el escrito. No repitas los que no están en esa lista.
No expliques que se trata de un complemento: escribe los párrafos y nada más."""


def prompt_antecedentes(texto_acto: str, tipo_asunto: str = "") -> str:
    """Los antecedentes se leen del documento ENTERO, no del recorte.

    El recorte del resumen se queda con el estudio de fondo, y ahí no está el
    trámite: la presentación, la admisión y el emplazamiento viven al principio
    del documento, en la parte que el otro recorte descarta.
    """
    return f"""{_NUCLEO}

{instrucciones_antecedentes(tipo_asunto)}

Éste es el documento:

──────────────────────────────────────────
{_cortar_bien(texto_acto, TOPE_CARACTERES)}
──────────────────────────────────────────

Escribe el apartado de antecedentes, un párrafo por línea. Sólo el apartado."""


def prompt_relato(antecedentes: str, resumen_acto: str, resumen_conceptos: str,
                  es_recurso: bool = False, tipo_asunto: str = "",
                  quejoso: str = "", responsable: str = "",
                  lo_resuelto: str = "") -> str:
    """De qué va el asunto, contado al secretario de corrido.

    David (17-sep-2026): «me gustaría una tarjeta más grande en la que al
    secretario se le explique de qué va el caso (…) mi redacción es más amena
    y trata de dar a entender el asunto de una forma más sencilla, con una
    sola tarjeta». Los tres resúmenes ya estaban en pantalla, pero como tres
    pliegues técnicos: antecedentes, qué resolvió, qué alega. Esto los cuenta
    como se cuenta un asunto a un compañero: quién hizo qué, quién se quejó,
    qué le contestaron y por qué, y quién viene ahora y con qué.

    NO INVENTA: todo sale de los resúmenes que recibe, que a su vez salieron
    de los documentos. Y NO LISTA los problemas jurídicos: la pantalla los
    pone debajo, numerados, tal como los calculó el reparto, para que la litis
    que lee el secretario sea exactamente la que se va a estudiar.
    """
    import tipos_asunto as _tap
    _t = tipo_asunto or ("amparo_revision" if es_recurso else "amparo_directo")
    _v = _tap.vocabulario_de(_t)
    _organo = _tap.sujetos_de(_t)["organo"][0]
    _nombre = _v.get("nombre", "amparo directo")
    _promovente = _v.get("promovente", "quejoso")
    _combate = _v.get("combate", "conceptos de violación")
    # QUÉ RESOLVIÓ LA RESPONSABLE, con su nombre. Iba escrito a mano —«esa
    # demanda»— y una Sala de apelación no resuelve una demanda: resuelve el
    # recurso. Lo lee `fase_origen.lo_resuelto`; si hay duda, devuelve vacío
    # y aquí se usa una fórmula que no afirma de dónde viene.
    _resuelto = " ".join(str(lo_resuelto or "").split())
    _pregunta = (f"¿Cómo resolvió {_organo} {_resuelto}?" if _resuelto
                 else f"¿Qué resolvió {_organo}?")
    _ficha = ""
    if (quejoso or "").strip():
        _ficha += f"Quien promueve, según la ficha: {quejoso.strip()}.\n"
    if (responsable or "").strip():
        _ficha += f"Órgano que dictó lo que se combate, según la ficha: {responsable.strip()}.\n"
    if es_recurso:
        _hilo = f"""1. EL ORIGEN. Qué autoridad hizo qué y a quién: el acto de donde arranca todo
   (una baja, una multa, una determinación, un crédito…), con los nombres.
2. AQUÍ EMPIEZA EL PROBLEMA. Quién lo combatió en el juicio de origen y qué
   alegó ahí; destaca lo principal, en una o dos frases.
3. CÓMO RESOLVIÓ {_organo.upper()}. Qué decidió y, en breve, las razones técnicas
   por las que lo decidió.
4. QUIÉN VIENE AHORA. Inconforme, quién interpone la {_nombre} y qué alega
   en sus {_combate}, entre otras cosas: la cuestión principal."""
    else:
        _hilo = f"""1. EL ORIGEN. De qué juicio o procedimiento viene el asunto: quién demandó a
   quién y qué pedía, con los nombres.
2. AQUÍ EMPIEZA EL PROBLEMA. Qué se discutió y qué resolvió cada instancia
   hasta llegar a lo que se reclama; lo principal, en una o dos frases.
3. CÓMO RESOLVIÓ {_organo.upper()}. Qué decidió en lo que se reclama y, en
   breve, las razones técnicas por las que lo decidió.
4. QUIÉN VIENE AHORA. Quién promueve el {_nombre} y qué alega en sus
   {_combate}, entre otras cosas: la cuestión principal."""
    return f"""Eres secretario proyectista de un Tribunal Colegiado y le cuentas a un
compañero de qué va un asunto que le acaban de turnar. Escribe en español,
en segunda persona («mira», «tendrás»), de corrido y sin tecnicismos de más:
que se entienda a la primera. Pero todo lo que digas tiene que estar en los
resúmenes de abajo; si un dato no consta —un nombre, una fecha, un monto—,
no lo inventes: di que no consta o sáltalo.

EL HILO, en este orden y en cinco párrafos cortos (320 a 480 palabras en
total):
{_hilo}
5. DE QUÉ PENDE. Cierra diciendo cuál es la cuestión técnica de la que
   depende el resultado y por qué: la figura jurídica que está en juego —la
   preclusión, la caducidad, la competencia, la firma autógrafa, la
   litisconsorcio, la valoración de una prueba—, el precepto o el plazo del
   que cuelga, y qué pasaría si se resuelve de un lado o del otro. Sin
   proponer el sentido: se dice qué se decide, no cómo debería decidirse.

Cuenta el origen con «Mira: el asunto tuvo su origen en que…»; marca el giro
con «Aquí empieza el problema, porque…»; abre el tercer párrafo con la
pregunta «{_pregunta}» —tal cual, sin cambiarla— y contéstala con «Pues…»; y
el cuarto con «Inconforme con esa determinación, …». Una pregunta y su
respuesta valen más que un párrafo de considerandos.

LLANO, PERO SIN PERDER LO TÉCNICO. Son las dos cosas a la vez y no se
negocia ninguna:
  · LLANO es el REGISTRO: frases cortas, orden natural, voz activa, sin
    latinajos («a quo», «sub júdice», «ad causam»), sin fórmulas de sentencia
    («en las relatadas consideraciones», «resulta dable estimar»), sin
    gerundios encadenados y sin perífrasis —«no valoró la prueba», no «omitió
    realizar la debida valoración probatoria»—. Se lee en voz alta y se
    entiende a la primera.
  · TÉCNICO es el CONTENIDO: la figura jurídica que decide el asunto se
    NOMBRA por su nombre —ampliación de demanda, preclusión, litis,
    violación procesal, competencia, caducidad— y se explica en la misma
    frase con seis o siete palabras llanas: «la preclusión, que es cuando se
    pierde el derecho a hacer algo por dejar pasar el plazo». Los datos que
    deciden —el plazo, la fecha, el precepto, el monto, la vía— van con su
    número. Simplificar hasta borrar la cuestión jurídica es peor que el
    tecnicismo: quien lee esto tiene que poder formar criterio.

LO QUE NO VA: no enumeres los problemas jurídicos ni digas «tendrás que
resolver» —eso lo pone la pantalla debajo de tu relato, calculado aparte—;
no pongas títulos, viñetas, negritas ni ningún formato; no cites tesis ni
transcribas artículos; no adelantes cómo debería resolverse.

{_ficha}ANTECEDENTES:
{antecedentes}

LO QUE RESOLVIÓ {_organo.upper()}:
{resumen_acto}

LO QUE SE COMBATE ({_combate.upper()} DE QUIEN PROMUEVE, {_promovente.upper()}):
{resumen_conceptos}

Devuelve sólo el relato, sin encabezado."""


def prompt_problemas(resumen_acto: str, resumen_conceptos: str,
                     es_recurso: bool = False, tipo_asunto: str = "",
                     n_planteamientos: int = 0, faltan: list = None) -> str:
    # EL RÓTULO ENSEÑA EN MAYÚSCULAS cómo llamar al órgano, que es la forma más
    # imitable que hay. Decía «LO QUE RESOLVIÓ LA RESPONSABLE» en los cuatro, y
    # en una queja lo que resolvió fue el Juzgado de Distrito.
    import tipos_asunto as _tap
    _t = tipo_asunto or ("amparo_revision" if es_recurso else "amparo_directo")
    _organo = _tap.sujetos_de(_t)["organo"][0].upper()
    # ═══════════════════════════════════════════════════════════════════════
    # EL REPARTO: NINGÚN PLANTEAMIENTO SE QUEDA FUERA
    # ═══════════════════════════════════════════════════════════════════════
    # Medido el 11-sep-2026 sobre el ADC 393/2025: de los MISMOS documentos
    # salieron 11 problemas por una corrida y 5 por otra. La causa de fondo es
    # que el motor rechaza `temperature` —«only the default (1) value is
    # supported»—, así que esta lectura corre con muestreo pese a que el módulo
    # dice fijarla; sobre un mismo resumen, cuatro tiradas dieron 7, 7, 7 y 8.
    #
    # Perseguir el número es perseguir lo que no se puede fijar. Y el número no
    # es el daño: agrupar dos planteamientos sobre la misma omisión en un
    # problema es buen criterio. El daño es que UNO se quede fuera de todos,
    # porque entonces no se decide, no se estudia, y la sentencia sale
    # incongruente sin que nadie lo note.
    #
    # Así que se deja de pedir un número y se pide un REPARTO comprobable: se
    # le dice cuántos planteamientos hay —el contador los cuenta, 43 de 43 en
    # el acervo— y cada problema declara cuáles cubre. La unión se comprueba
    # después con aritmética, no con confianza. Medido: 3 de 3 tiradas cubren
    # 17 de 17 sin huecos ni repeticiones, con el número variando entre 7 y 8.
    _reparto = ""
    if n_planteamientos >= 2:
        _reparto = (
            f"""
CUÁNTOS PLANTEAMIENTOS HAY QUE CUBRIR: el escrito trae {n_planteamientos},
numerados del 1 al {n_planteamientos} en el orden en que aparecen en el
resumen de arriba.

NO SE QUEDA NINGUNO FUERA. Agrupa los que compartan la misma cuestión jurídica
—dos planteamientos sobre la misma omisión son UN problema— pero cada uno de
los {n_planteamientos} tiene que quedar dentro de algún problema. Añade a cada
problema el campo "cubre": la lista de los números de planteamiento, enteros
entre 1 y {n_planteamientos}, que ese problema responde. La unión de todos los
"cubre" tiene que dar exactamente 1..{n_planteamientos}, sin huecos y sin
repetir un número en dos problemas.

Un planteamiento que no entra en ningún problema no se estudia y la sentencia
sale incongruente: por eso este reparto no es un adorno.
""")
        if faltan:
            _reparto += (
                f"""
Y ATENCIÓN, PORQUE ES UN SEGUNDO INTENTO: en el anterior se quedaron fuera los
planteamientos {", ".join(str(x) for x in faltan)}. Esta vez tienen que estar
dentro de algún problema, cada uno en el que le corresponda por su materia.
""")
    # ═══ «cubre» EN LA PLANTILLA: UN TIPO, NO UN VALOR (26-sep-2026) ═══════
    # La plantilla decía `"cubre": [1, 2]` SIEMPRE, también cuando no se pedía
    # el reparto. En los asuntos de un solo concepto el modelo lo copiaba tal
    # cual: 13 sesiones de concepto único con «cubre» lleno y fuera de rango
    # (L11 del diagnóstico), y el estudio leía «CUBRE: conceptos primero y
    # segundo» donde sólo había uno. Es la lección que ya se midió tres veces:
    # un ejemplo escrito en el prompt se copia literal. Ahora el campo se
    # describe por su tipo, sin números, y sólo se pide cuando hay reparto
    # que comprobar (dos o más planteamientos contados).
    _campo_cubre = (
        f'\n      "cubre": [<número de planteamiento, entero de 1 a {n_planteamientos}>, …],'
        if n_planteamientos >= 2 else "")
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
    {{"pregunta": "...",{_campo_cubre}
      "jerarquia": "principal|accesorio",
      "clase": "fondo|procesal|procedencia",
      "resolvio": "qué resolvió el órgano recurrido sobre este punto",
      "combate": "qué lo combate",
      "depende_de": "el número del problema del que depende, o null",
      "impedimento": null,
      "apoyo": null}}
  ]
}}
"clase": "procesal" cuando lo que se combate es una actuación del
procedimiento —una ampliación de demanda desechada o precluida, una prueba no
admitida o no desahogada, un emplazamiento, un recurso ordinario resuelto—;
"procedencia" cuando es una causa de improcedencia o sobreseimiento; "fondo"
en lo demás. En la violación procesal, `resolvio` es lo que decidió la
actuación combatida y, si la hubo, la resolución del recurso ordinario que la
confirmó; no lo que la sentencia definitiva dijo de pasada.
{_reparto}Si adviertes un impedimento técnico que llevaría a inoperancia, ponlo en
"impedimento" como {{"motivo": "inoperancia", "explicacion": "..."}}.
Y si adviertes lo contrario —algo que sostenga el planteamiento— ponlo en
"apoyo" como {{"motivo": "razón toral|jurisprudencia|constancia",
"explicacion": "..."}}. Los dos campos son opcionales y ninguno obliga al
otro; lo que no vale es rellenar siempre uno y nunca el otro."""


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

    AL DESCARTAR, SE FUSIONA (26-sep-2026). El repetido se tiraba entero, y
    con él su «cubre»: los planteamientos que sólo él respondía quedaban
    huérfanos —el reparto los daba por no recogidos y se gastaba una segunda
    lectura— o, peor, desaparecían del CUBRE del estudio (F6 del diagnóstico).
    Ahora el que se queda hereda el «cubre» del descartado, y también su
    jerarquía si el descartado era el principal. Y como la fase 3 numera los
    problemas y `depende_de` apunta a esos números, se renumeran: con un
    problema menos, «depende del 3» ya no señalaba al mismo.
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
    destino = {}                           # número original → número que queda
    for i, p in enumerate(problemas or [], 1):
        q = p.get("pregunta", "") if isinstance(p, dict) else str(p)
        v = _vocab(q)
        if len(v) < 4:
            fuera.append(p)
            destino[i] = len(fuera)
            continue
        # El MÁS parecido de los ya vistos: se descarta en los mismos casos que
        # antes (alguno pasa de 0.60), pero la fusión va al que de verdad es.
        parecido, donde = max(((len(v & w) / max(1, len(v | w)), k) for w, k in vistos),
                              default=(0.0, -1))
        if parecido > 0.60:
            queda = fuera[donde]
            if isinstance(queda, dict) and isinstance(p, dict):
                queda = dict(queda)
                if "cubre" in queda or "cubre" in p:
                    queda["cubre"] = sorted(set(cubre_de(queda)) | set(cubre_de(p)))
                if str(p.get("jerarquia") or "").strip().lower() == "principal":
                    queda["jerarquia"] = "principal"
                fuera[donde] = queda
            destino[i] = donde + 1
            continue                       # ya se preguntó esto
        vistos.append((v, len(fuera)))
        fuera.append(p)
        destino[i] = len(fuera)
    if len(fuera) == len(problemas or []):
        return fuera
    for k, p in enumerate(fuera, 1):
        if not isinstance(p, dict) or p.get("depende_de") in (None, "", "null"):
            continue
        try:
            d = int(p.get("depende_de"))
        except (TypeError, ValueError):
            continue
        if d in destino:
            p = dict(p)
            p["depende_de"] = None if destino[d] == k else destino[d]
            fuera[k - 1] = p
    return fuera


def cubre_de(p) -> list:
    """Los números de planteamiento de «cubre», como enteros ordenados.

    El modelo los devuelve casi siempre como lista de enteros, pero no hay
    contrato que lo asegure: una cadena «1, 2» recorrida letra a letra daba
    '1', ',', ' ', '2' y los consumidores (`formato_sentencia.cubre_de`, el
    reparto) la leían mal o la tiraban."""
    import re as _re
    x = p.get("cubre") if isinstance(p, dict) else None
    if x is None or isinstance(x, bool):
        return []
    if isinstance(x, (int, float)):
        x = [x]
    elif isinstance(x, str):
        x = _re.findall(r"\d+", x)
    fuera = []
    for y in (x if isinstance(x, (list, tuple)) else []):
        try:
            fuera.append(int(y))
        except (TypeError, ValueError):
            pass
    return sorted(set(fuera))


def _con_cubre_en_rango(p, n: int):
    """El problema con su «cubre» como enteros y dentro de 1..n (n = los
    planteamientos contados). Sin conteo, o sin «cubre», no se toca."""
    if not isinstance(p, dict) or n < 1 or "cubre" not in p:
        return p
    q = dict(p)
    q["cubre"] = [x for x in cubre_de(p) if 1 <= x <= n]
    return q

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


# EL CENTINELA DEL CORTE. Viaja pegado al texto porque `_pedir` devuelve una
# cadena y media docena de sitios la consumen: cambiar la firma obligaba a
# tocarlos todos y a que ninguno se olvidara de mirar la bandera. Una marca en
# el texto no se puede ignorar por descuido —se ve— y las fases la retiran
# convirtiéndola en aviso antes de que llegue a ningún documento.
MARCA_CORTE = "\n\u2702 LECTURA INCOMPLETA POR CUPO"


def texto_cortado(x: str) -> bool:
    return MARCA_CORTE.strip() in (x or "")


def sin_marca(x: str) -> str:
    return (x or "").replace(MARCA_CORTE, "").rstrip()


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
    import llamada_modelo as _lm

    def _cortado(resp) -> bool:
        """¿El modelo se quedó sin cupo a mitad de frase?"""
        try:
            return (resp.choices[0].finish_reason or "").lower() == "length"
        except Exception:
            return False

    r = await _lm.crear(cliente, **kw)
    txt = (r.choices[0].message.content or "").strip()
    if not txt:
        # RESPUESTA VACÍA: el razonamiento se comió el presupuesto de salida.
        # Es el mismo fallo que main.py ya documenta para los modelos 5.6, y
        # aquí lo delataba un «Expecting value: line 1 column 1» que parecía un
        # problema de JSON y no lo era. Se reintenta con el doble de tope y sin
        # razonamiento: para leer un documento no hace falta.
        kw["max_completion_tokens"] = tope * 2
        kw.pop("reasoning_effort", None)
        r = await _lm.crear(cliente, **kw)
        txt = (r.choices[0].message.content or "").strip()
    # ═══════════════════════════════════════════════════════════════════════
    # UNA RESPUESTA CORTADA NO ES UNA RESPUESTA
    # ═══════════════════════════════════════════════════════════════════════
    # Sólo se reintentaba con la respuesta VACÍA. Una respuesta TRUNCADA —que
    # llega llena, legible y a media frase— pasaba como buena y nadie se
    # enteraba: `finish_reason` no se lee en ninguna fase del taller.
    #
    # Medido en el amparo directo 393/2025 del 11-sep-2026: la demanda traía
    # DIECISIETE conceptos de violación, el resumen se pidió con el tope por
    # omisión de 2.500 tokens y volvió con 2.497 —el 99,88% del cupo, tres
    # tokens de margen— terminando dentro de una marca de ancla sin cerrar,
    # «[[». Cada concepto cuesta 191 tokens: (2500-20)/191 = 12,98, y el cupo
    # predice TRECE. Salieron trece. Los conceptos 14 a 17 no se escribieron
    # nunca, los problemas jurídicos se derivan del resumen —no de la fuente—,
    # y la sentencia salió sin ellos. El secretario no pudo subsanarlo porque
    # el sistema jamás le dijo que faltaban.
    #
    # Se reintenta doblando el cupo, hasta tres veces (2.500 → 5.000 → 10.000 →
    # 20.000), y si aun así vuelve cortada se devuelve lo que haya: media
    # lectura vale más que ninguna, pero quien llame tiene que poder saberlo.
    # Por eso el corte se marca en el propio texto con un centinela que las
    # fases de arriba reconocen y convierten en aviso.
    _vueltas = 0
    while _cortado(r) and _vueltas < 3:
        _vueltas += 1
        kw["max_completion_tokens"] = tope * (2 ** _vueltas)
        kw.pop("reasoning_effort", None)
        print(f"   ✂️ respuesta cortada por cupo ({len(txt)} caracteres); "
              f"se repite con {kw['max_completion_tokens']} tokens")
        r = await _lm.crear(cliente, **kw)
        _n = (r.choices[0].message.content or "").strip()
        if _n:
            txt = _n
    if _cortado(r):
        print(f"   ⚠️ SIGUE CORTADA tras {_vueltas} intentos: "
              f"{len(txt)} caracteres con {kw.get('max_completion_tokens')} tokens")
        txt += MARCA_CORTE
    return txt


async def correr(cliente, texto_acto: str, texto_conceptos: str,
                 es_recurso: bool = False,
                 tipo_asunto: str = "",
                 quejoso: str = "", responsable: str = "") -> "Fases123":
    """Las tres fases, en orden. Los dos resúmenes van EN PARALELO —son
    independientes— y los problemas esperan a los dos, porque salen de su
    contraste."""
    import asyncio
    import json as _json

    # ── EL CUPO SE MIDE POR LO QUE HAY QUE LEER ──────────────────────────
    #
    # Los tres resúmenes salían con el mismo tope por omisión, 2.500 tokens,
    # tanto si el escrito tenía tres páginas como setenta y tres. En el amparo
    # directo 393/2025 —174.702 caracteres, diecisiete conceptos— ese tope
    # alcanzaba para trece y la lectura se cortaba a media frase.
    #
    # Un concepto resumido cuesta unos 190 tokens, medido sobre los trece que
    # sí salieron de ese asunto. No se sabe cuántos conceptos hay antes de
    # leerlos —contar rótulos es justamente lo que no funciona—, así que el
    # cupo se estima por el tamaño de lo que entra: un escrito de 175.000
    # caracteres pide del orden de 3.500 tokens de resumen. Es una cota
    # holgada y barata: el modelo escribe lo que necesita y no se le cobra por
    # el cupo que no gasta. Y si aun así se corta, `_pedir` lo detecta por
    # `finish_reason` y dobla, que es la red de verdad.
    # EL CUPO NO PUEDE SER MENOR QUE LO QUE SE PIDE ESCRIBIR. Con un mínimo de
    # 2,500 tokens, el resumen del acto y el de los agravios —que el propio
    # prompt pide de 1,000 a 1,600 palabras— se cortaban SIEMPRE: 35 recortes en
    # los registros de producción del 16 y 17 de septiembre, por pares, uno por
    # resumen y por adelanto (~10,000 y ~11,000 caracteres). Cada corte tira lo
    # generado y repite la llamada con 5,000–6,222 tokens, que siempre alcanza.
    # Medido en local sobre la revisión fiscal 2/2026: 123 s el primer paso, con
    # los dos resúmenes generados dos veces. Se empieza con lo que alcanza. El
    # tope no cuesta: se paga lo que se escribe, y ahora no se escribe dos veces.
    def _cupo(texto: str, minimo: int = 6000, maximo: int = 12000) -> int:
        return max(minimo, min(maximo, 900 + len(texto or "") // 50))

    an, ra, rc = await asyncio.gather(
        _pedir(cliente, prompt_antecedentes(texto_acto, tipo_asunto), 3000),
        _pedir(cliente, prompt_resumen_acto(texto_acto, es_recurso, tipo_asunto),
               _cupo(texto_acto)),
        _pedir(cliente, prompt_resumen_conceptos(texto_conceptos, es_recurso,
                                                 tipo_asunto),
               _cupo(texto_conceptos)),
    )
    # ── Y SI AUN ASÍ SE CORTÓ, QUE SE SEPA ───────────────────────────────
    # El centinela llega pegado al texto. Se retira aquí —para que no viaje a
    # ningún documento— y se convierte en el aviso que el secretario sí puede
    # leer: es exactamente lo que faltó en el 393/2025, donde el corte fue
    # invisible en los tres sitios donde podía verse.
    _cortes = []
    for _et, _tx in (("los antecedentes", an), ("lo que resolvió la responsable", ra),
                     ("el resumen de los planteamientos", rc)):
        if texto_cortado(_tx):
            _cortes.append(_et)
    an, ra, rc = sin_marca(an), sin_marca(ra), sin_marca(rc)
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
    # ── LOS QUE FALTEN, SE PIDEN ─────────────────────────────────────────
    #
    # Se comprueba contra el escrito ENTERO, no contra el recorte: es
    # justamente el número que el recorte falsea. Si el resumen dejó fuera el
    # séptimo concepto, se pide otra pasada SÓLO de los que faltan y se añade.
    #
    # Sin esto, un escrito con ocho conceptos producía un resumen de seis y
    # nadie se enteraba —el prompt le prohíbe al modelo decir que le falta
    # documento—, y el proyecto salía incongruente e inexhaustivo.
    # ═══ EL ACTO A FONDO, EN PARALELO CON TODO LO DE LOS AGRAVIOS ═══════════
    # Iba DETRÁS de la cobertura y de la reescritura de los agravios, aunque no
    # depende de ninguna: sólo necesita el texto del acto y su propio primer
    # resumen, que ya existen aquí. Medido el 17-sep-2026 en la revisión fiscal
    # 2/2026: una reescritura a fondo tarda unos 120 s, y cuando hacían falta
    # las dos se pagaban en serie. Mismas instrucciones, mismas comprobaciones,
    # mismo resultado: sólo cambia cuándo empieza.
    async def _acto_a_fondo(ra_):
        try:
            _obj_a = objetivo_acto(texto_acto)
            _todas_a, _sin_a = citas_sin_nombrar(recortar_acto(texto_acto or ""), ra_)
            _vfa = 0
            # La misma red que los agravios: una tesis sin nombrar basta.
            while ((len(ra_.split()) < 0.75 * _obj_a
                    or (len(_todas_a) >= 1 and len(_sin_a) >= 1))
                   and _vfa < 2):
                _vfa += 1
                print(f"   🧩 resumen del acto a fondo (vuelta {_vfa}): {len(ra_.split())} palabras "
                      f"de ~{_obj_a} · tesis de la responsable {len(_todas_a)}, sin nombrar {len(_sin_a)}")
                _ra2 = await _pedir(cliente, prompt_acto_a_fondo(
                    texto_acto, ra_, sorted(_sin_a), _obj_a, es_recurso, tipo_asunto),
                    _cupo(texto_acto))
                _ra2 = sin_marca(_ra2 or "").strip()
                if len(_ra2.split()) <= len(ra_.split()):
                    break
                ra_ = _ra2
                _todas_a, _sin_a = citas_sin_nombrar(recortar_acto(texto_acto or ""), ra_)
            return ra_, dict(objetivo_acto=_obj_a, vueltas_fondo_acto=_vfa,
                             citas_acto=len(_todas_a),
                             citas_acto_sin_nombrar=sorted(_sin_a))
        except Exception as _exa:
            print(f"   ⚠️ no se pudo profundizar el resumen del acto: {_exa}")
            return ra_, {}

    _tarea_acto = asyncio.ensure_future(_acto_a_fondo(ra))
    _conteo, _avisos_cob = {}, []
    try:
        # ANTES QUE NADA, ¿LLEGÓ EL ESCRITO ENTERO? Si vino mutilado, la
        # cobertura se mide sobre un texto cortado y dirá que no falta ninguno
        # aunque falten: es una comprobación que miente en la dirección
        # peligrosa.
        _amp = sospecha_de_amputacion(texto_acto, texto_conceptos)
        if _amp:
            _avisos_cob.append(_amp)

        _conteo = planteamientos(texto_conceptos, es_recurso) or {}
        _cob = sin_resumir(texto_conceptos, rc, _conteo) or {}
        _faltan = list(_cob.get("faltan") or [])
        print(f"   🔢 planteamientos: {_conteo.get('n')} "
              f"({_conteo.get('estado')}, vía {_conteo.get('via')}) · "
              f"sin resumir: {_faltan or 'ninguno'}")
        _vuelta = 0
        while _faltan and _vuelta < 3:
            _vuelta += 1
            print(f"   🧩 faltan por resumir: {_faltan} (vuelta {_vuelta})")
            _extra = await _pedir(cliente, prompt_conceptos_que_faltan(
                texto_conceptos, _faltan, es_recurso, tipo_asunto,
                tramos=_cob.get("tramos_que_faltan")), 3000)
            if not _extra.strip():
                break
            rc = (rc.rstrip() + "\n\n" + _extra.strip())
            _cob = sin_resumir(texto_conceptos, rc, _conteo) or {}
            _nuevos = list(_cob.get("faltan") or [])
            if len(_nuevos) >= len(_faltan):
                # No avanzó: se para y se dice, en vez de dar vueltas.
                _faltan = _nuevos
                break
            _faltan = _nuevos
        _conteo = dict(_conteo, faltan=_faltan, vueltas=_vuelta,
                       apartados_resumen=_cob.get("apartados_resumen"),
                       sin_rotular=_cob.get("sin_rotular"))
        # ── LA HONDURA: corto, o sin las tesis invocadas → se reescribe ──
        _obj = objetivo_conceptos(texto_conceptos)
        _todas, _sin = citas_sin_nombrar(texto_conceptos, rc)
        _vf = 0
        # LA RED NO PUEDE TENER AGUJEROS DE MEDIA TESIS. Decía «reescribe si
        # falta la MITAD de las tesis o si no llega al 60 %». Mientras el primer
        # resumen nombraba una de seis, saltaba siempre y el final salía
        # completo. En cuanto la primera vuelta mejoró —se le da la lista de
        # tesis— nombró cuatro de seis, pasó por debajo de la red y el proyecto
        # de la revisión fiscal 2/2026 (V8, 17-sep-2026) salió con dos tesis de
        # la recurrente sin nombrar y 1,116 palabras de las 1,596 que pide el
        # escrito. David lo había fijado: «los agravios deben estar resumidos
        # de forma completa, sobre todo cuando citan múltiples tesis».
        # Ahora: una sola tesis sin nombrar, o menos del 75 %, y se reescribe.
        def _corto(txt):
            return len(txt.split()) < 0.75 * _obj
        def _mudo(sin):
            return len(_todas) >= 1 and len(sin) >= 1
        while (_corto(rc) or _mudo(_sin)) and _vf < 2:
            _vf += 1
            print(f"   🧩 síntesis a fondo (vuelta {_vf}): {len(rc.split())} palabras "
                  f"de ~{_obj} · tesis invocadas {len(_todas)}, sin nombrar {len(_sin)}")
            _rc2 = await _pedir(cliente, prompt_conceptos_a_fondo(
                texto_conceptos, rc, sorted(_sin), _obj, es_recurso, tipo_asunto,
                tramos=_conteo.get("tramos")), _cupo(texto_conceptos))
            _rc2 = sin_marca(_rc2 or "").strip()
            if len(_rc2.split()) <= len(rc.split()):
                break                      # no mejoró: se queda lo que había
            rc = _rc2
            _todas, _sin = citas_sin_nombrar(texto_conceptos, rc)
        _conteo = dict(_conteo, objetivo_palabras=_obj, vueltas_fondo=_vf,
                       citas_invocadas=len(_todas), citas_sin_nombrar=sorted(_sin))
        # ── Y LA SENTENCIA, IGUAL: corta o sin las tesis en que se apoyó ──
        # David: «los agravios y la sentencia deben estar resumidos de forma
        # completa, sobre todo cuando formulan planteamientos de fondo y
        # citan múltiples tesis». Un resumen del acto que escoge dos de cinco
        # consideraciones deja sin contestar los agravios contra las otras.
        # EL ACTO, YA EN CURSO DESDE ARRIBA: aquí sólo se recoge.
        ra, _extra_acto = await _tarea_acto
        _conteo = dict(_conteo, **_extra_acto)
        if _vf:
            print(f"   🧩 síntesis a fondo: quedó en {len(rc.split())} palabras · "
                  f"sin nombrar {sorted(_sin) or 'ninguna'}")
        _av = aviso_de_cobertura(texto_conceptos, rc, _conteo)
        if _av:
            _avisos_cob.append(_av)
    except Exception as _ex:
        # Si lo de los agravios revienta, el acto se recoge igual: no se pierde
        # el trabajo que ya estaba haciendo en paralelo.
        try:
            ra, _ = await _tarea_acto
        except Exception:
            pass
        # QUE EL FALLO DEL CONTADOR NO SE PAREZCA A «NO FALTA NADA». Callarse
        # aquí es reproducir el defecto que esto viene a cerrar.
        print(f"   ⚠️ no se pudo comprobar la cobertura de conceptos: {_ex}")
        _faltan = []
        _conteo = {"estado": "no_contado", "error": str(_ex)[:200]}
        _avisos_cob.append(
            "NO SE PUDO COMPROBAR que estén resumidos todos los "
            "planteamientos. No es que no falte ninguno: es que no se sabe. "
            "Cuéntalos tú en el escrito antes de firmar.")

    # EL RELATO VA EN PARALELO CON LOS PROBLEMAS. Los dos parten de los mismos
    # resúmenes ya definitivos y ninguno necesita al otro. Lanzarlo aquí y
    # recogerlo al final cuesta lo que tarde de más que la fase de problemas,
    # que suele ser nada: no se le añade un escalón al adelanto.
    async def _relato():
        try:
            import fase_origen as _fo_r
            _txt = await _pedir(cliente, prompt_relato(
                an, ra, rc, es_recurso, tipo_asunto, quejoso, responsable,
                # De dónde viene la sentencia reclamada: el recurso de
                # apelación, el juicio oral mercantil, el juicio de nulidad…
                lo_resuelto=_fo_r.lo_resuelto(f"{an}\n{ra}", tipo_asunto)), 2200)
            _txt = sin_marca(_txt or "").strip()
            try:
                import meta_lenguaje as _ml_r
                _txt, _ = _ml_r.limpiar(_txt)
            except Exception:
                pass
            return _txt
        except Exception as _exr:
            print(f"   ⚠️ no se pudo contar el asunto: {_exr}")
            return ""
    _tarea_relato = asyncio.ensure_future(_relato())

    f = Fases123(antecedentes=an, resumen_acto=ra, resumen_conceptos=rc,
                 conteo=_conteo)
    for _a in _avisos_cob:
        f.avisos.append(_a)
    for _f in _quitadas:
        f.avisos.append(
            f"SE QUITÓ UNA FRASE QUE HABLABA DEL ARCHIVO, NO DEL ASUNTO: "
            f"«{_f[:220]}». Va entera aquí por si el filtro se equivocó y hay "
            f"que devolverla; pero una frase así dentro del fallo deja al que "
            f"firma preguntándose qué más se inventó.")
    try:
        # ── EL REPARTO SE COMPRUEBA CON ARITMÉTICA, NO CON CONFIANZA ────
        # El número de problemas no se puede fijar —el motor rechaza
        # `temperature` y corre con muestreo—, pero el REPARTO sí se comprueba:
        # la unión de los «cubre» tiene que dar 1..N. Si falta alguno se pide
        # otra vez nombrándolo, y si aun así falta, se dice en un aviso con su
        # número. Un planteamiento huérfano no se decide ni se estudia.
        _n_plant = int((_conteo or {}).get("n") or 0) \
            if str((_conteo or {}).get("estado")) == "contado" else 0
        _faltan_rep, _vuelta_rep = [], 0
        while True:
            crudo = await _pedir(cliente, prompt_problemas(
                ra, rc, es_recurso, tipo_asunto, _n_plant, _faltan_rep),
                3500 if not _n_plant else 5000, json_estricto=True)
            m = re.search(r"\{.*\}", crudo, re.S)
            j = _json.loads(m.group(0) if m else crudo)
            _probs = _sin_repetidos(j.get("problemas", []) or [])
            # «cubre» SE CIÑE A LO QUE SE CONTÓ (26-sep-2026). Un número fuera
            # de 1..N es un planteamiento que no existe —el [1, 2] copiado de
            # la plantilla en un asunto de un solo concepto— y el estudio lo
            # escribía como «conceptos primero y segundo». Se quita. Sin
            # conteo no hay rango que comprobar y se deja como venga.
            _probs = [_con_cubre_en_rango(_p, _n_plant) for _p in _probs]
            # EL REPARTO SÓLO SE COMPRUEBA CUANDO SE PIDIÓ. Con un planteamiento
            # el prompt no pide «cubre» y comprobarlo daba por huérfano al 1:
            # una segunda lectura pagada y un aviso falso.
            if _n_plant < 2:
                break
            _cub = []
            for _p in _probs:
                _cub.extend(cubre_de(_p))
            _faltan_rep = sorted(set(range(1, _n_plant + 1)) - set(_cub))
            print(f"   🧮 reparto: {len(_probs)} problemas cubren "
                  f"{len(set(_cub))}/{_n_plant}"
                  + (f" · huérfanos: {_faltan_rep}" if _faltan_rep else ""))
            if not _faltan_rep or _vuelta_rep >= 1:
                break
            _vuelta_rep += 1
        f.problema_global = j.get("problema_global", "")
        f.problemas = _probs
        if _n_plant >= 2:
            _conteo = dict(_conteo or {}, reparto={
                "planteamientos": _n_plant, "problemas": len(_probs),
                "huerfanos": _faltan_rep, "vueltas": _vuelta_rep})
            f.conteo = _conteo
        if _faltan_rep:
            f.avisos.insert(0, (
                f"HAY PLANTEAMIENTOS QUE NINGÚN PROBLEMA JURÍDICO RECOGE: el "
                f"{', el '.join(str(x) for x in _faltan_rep[:8])} de los "
                f"{_n_plant} que trae el escrito. Lo que no entra en un problema "
                f"no se califica ni se estudia, y la sentencia sale "
                f"incongruente. Míralos en el resumen y, si hacen falta, "
                f"añádelos a mano al criterio antes de generar."))
    except Exception as e:
        # ESTE AVISO NO BASTABA, Y COSTÓ SEMANAS. Cuando el motor dejó de
        # aceptar `temperature`, esta captura apuntó el aviso y siguió: el
        # adelanto devolvía 200, el documento salía bien de forma, y los
        # PROBLEMAS JURÍDICOS iban vacíos. Sin ellos no hay consulta al acervo
        # que apuntar, ni propuesta que calificar, ni estudio que ordenar: el
        # corazón del taller estaba apagado y el único síntoma era una línea
        # entre catorce avisos.
        #
        # Sigue sin tumbar el adelanto —el documento formal vale por sí solo—
        # pero lo dice en mayúsculas, lo primero, y lo deja en los registros.
        print(f"   ❌ PROBLEMAS JURÍDICOS: no se derivaron — {e}")
        f.avisos.insert(0, (
            "NO SE DERIVARON LOS PROBLEMAS JURÍDICOS. El adelanto sale, pero "
            "SIN ELLOS no hay nada que consultar al acervo, nada que calificar "
            "y nada que ordenar en el estudio: el fondo saldría vacío. "
            f"Causa: {e}"))
    # ── EL CORTE POR CUPO, LO PRIMERO Y EN MAYÚSCULAS ────────────────────
    # No es un defecto de estilo: es que el sistema no leyó el documento
    # entero, y todo lo que venga después —los problemas jurídicos, la
    # búsqueda del acervo, el estudio— se construye sobre una lectura a medias.
    # En el 393/2025 se cayeron cuatro conceptos de diecisiete y el secretario
    # no pudo subsanarlo porque nunca supo que existían.
    if _cortes:
        f.avisos.insert(0, (
            "LA LECTURA SE CORTÓ POR FALTA DE CUPO y no se pudo completar ni "
            f"repitiéndola: {', '.join(_cortes)}. Lo que no se leyó NO está en "
            "los problemas jurídicos ni en el estudio, y no se ve que falta. "
            "Sube el documento por partes o reduce lo que no sea el escrito de "
            "la parte, y vuelve a generar el adelanto."))
    f.avisos.extend(revisar(f))
    try:
        f.relato = await _tarea_relato
    except Exception as _exr2:
        print(f"   ⚠️ el relato no llegó: {_exr2}")
    if f.relato:
        print(f"   📖 relato del asunto: {len(f.relato.split())} palabras")
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
    # ═══════════════════════════════════════════════════════════════════════
    # LA CUENTA DE PLANTEAMIENTOS, SIEMPRE, TAMBIÉN CUANDO NO SE PUDO CONTAR
    # ═══════════════════════════════════════════════════════════════════════
    # Sin esto no hay manera de distinguir «no faltaba ninguno» de «no se
    # comprobó», y ésa es exactamente la razón por la que el aviso estuvo
    # apagado 101 sesiones sin que nadie lo notara. Guardarlo permite la
    # consulta que prueba que la capa está encendida:
    #   SELECT estado->'fases'->'conteo'->>'estado', count(*)
    #     FROM taller_sesiones GROUP BY 1;
    conteo: dict = field(default_factory=dict)
    # ═══════════════════════════════════════════════════════════════════════
    # LO QUE RESOLVIÓ EL JUZGADO, LEÍDO DEL PAPEL Y NO PREGUNTADO
    # ═══════════════════════════════════════════════════════════════════════
    # De este dato depende el punto resolutivo entero —si se confirma, se
    # revoca o se modifica—, y se venía decidiendo con la PROSA DEL MODELO: el
    # resumen de los antecedentes, o la frase con la que describió el asunto al
    # proponer. Cuando esa prosa no lo decía, el resultando salía con «no
    # consta el sentido de la sentencia recurrida» —literalmente, en la
    # revisión 650/2025 de David— aunque el resolutivo del juzgado estuviera a
    # dos páginas diciendo «La Justicia de la Unión ampara y protege».
    #
    # Está en el papel y se lee. Se lee UNA VEZ, en el adelanto, que es donde
    # el texto crudo existe, y viaja en el estado de la sesión: `fuentes` no
    # viajaba —los 240 kB no caben— y por eso el worker que resolvía se
    # quedaba sin el texto y volvía a depender de la prosa.
    resolvio_a_quo: str = ""
    # El resolutivo del juzgado, listo para reproducirse, con la cola apuntando
    # a la sentencia recurrida. Vacío si no se pudo leer con seguridad.
    resolutivo_recurrida: str = ""
    # El expediente y la fecha de la sentencia de la Sala, leídos del PDF. El
    # resolutivo de la revisión fiscal los nombra —«Se confirma la sentencia de
    # {fecha}, dictada en el expediente {expediente}»— y hasta ahora se
    # buscaban en la prosa del proyecto, donde a menudo no están.
    expediente_origen: str = ""
    fecha_origen: str = ""
    # EL RELATO: de qué va el asunto, contado al secretario de corrido. Ver
    # `prompt_relato`. Sale de los tres resúmenes y va a la pantalla, no al
    # documento.
    relato: str = ""

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
    # LA BANDA SE ESCALA CON CUÁNTOS PLANTEAMIENTOS HAY QUE RESUMIR.
    #
    # La mediana de 472 palabras salió de engroses con cuatro o cinco
    # conceptos. En el ADC 393/2025, con DIECISIETE, el único aviso que saltó
    # dijo que el resumen se había pasado de largo —1.818 palabras contra
    # 472— cuando lo que había ocurrido es justo lo contrario: lo habían
    # CORTADO en el decimotercero. El aviso mandaba al secretario en dirección
    # contraria al problema, que es peor que no avisar.
    #
    # Con la cuenta delante, el objetivo es por planteamiento: la mediana
    # dividida entre los cuatro que la produjeron, por los que haya.
    _n_plant = int((getattr(f, "conteo", {}) or {}).get("n") or 0)
    _obj_conceptos = (PALABRAS_RESUMEN_CONCEPTOS if _n_plant < 2
                      else max(PALABRAS_RESUMEN_CONCEPTOS,
                               (PALABRAS_RESUMEN_CONCEPTOS // 4) * _n_plant))
    # EL OBJETIVO LO FIJA EL ESCRITO cuando se pudo medir (61/2025: 68 mil
    # caracteres piden ~1.250 palabras, no 472).
    _obj_conceptos = max(_obj_conceptos,
                         int((getattr(f, "conteo", {}) or {}).get("objetivo_palabras") or 0))
    _sin_nombrar = list((getattr(f, "conteo", {}) or {}).get("citas_sin_nombrar") or [])
    _n_citas = int((getattr(f, "conteo", {}) or {}).get("citas_invocadas") or 0)
    if _sin_nombrar:
        avisos.append(f"EL RESUMEN DE LOS PLANTEAMIENTOS NO NOMBRA {len(_sin_nombrar)} de las "
                      f"{_n_citas} tesis que invoca la parte: {', '.join(_sin_nombrar[:6])}"
                      f"{'…' if len(_sin_nombrar) > 6 else ''}. El estudio tiene que "
                      f"hacerse cargo de ellas; compruébalo.")
    _obj_acto = max(PALABRAS_RESUMEN_ACTO,
                    int((getattr(f, "conteo", {}) or {}).get("objetivo_acto") or 0))
    _sin_acto = list((getattr(f, "conteo", {}) or {}).get("citas_acto_sin_nombrar") or [])
    if _sin_acto:
        avisos.append(f"EL RESUMEN DE LA RESOLUCIÓN NO NOMBRA {len(_sin_acto)} de las "
                      f"{int((getattr(f, 'conteo', {}) or {}).get('citas_acto') or 0)} tesis en que "
                      f"se apoyó la responsable: {', '.join(_sin_acto[:6])}"
                      f"{'…' if len(_sin_acto) > 6 else ''}. Los agravios suelen ir contra "
                      f"ellas; compruébalo.")
    for etiqueta, n, objetivo in (("del acto", na, _obj_acto),
                                  ("de conceptos", nc, _obj_conceptos)):
        if n and not (0.5 * objetivo <= n <= 1.8 * objetivo):
            _coletilla = (f" con {_n_plant} planteamientos" if
                          etiqueta == "de conceptos" and _n_plant >= 2 else "")
            avisos.append(f"El resumen {etiqueta} tiene {n} palabras; lo "
                          f"esperable{_coletilla} son unas {objetivo}.")
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


def descartado() -> list:
    """[(qué, cuánto entró, cuánto había)] de lo recortado en esta corrida."""
    return [(q, tope, total) for q, (total, tope) in sorted(DESCARTADO.items())]


def olvidar_descartes() -> None:
    """Se limpia al empezar cada asunto: el diccionario es de módulo."""
    DESCARTADO.clear()
