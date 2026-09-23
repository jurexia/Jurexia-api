"""EL BARRIDO FINAL: ¿EXISTEN LOS ARTÍCULOS QUE EL PROYECTO CITA?

POR QUÉ EXISTE. David (23-sep-2026): «me gustaría agregar un barrido
inteligente a todo el proyecto al terminar para verificar que no está
inventando artículos, con una consulta sumamente rápida por internet».

QUÉ HUECO TAPA. El pipeline ya comprueba mucho, pero por tres puertas
distintas y ninguna cubre el documento entero:

  · `fase6_rag.resolver_articulo` verifica contra el ACERVO — y el acervo no
    lo tiene todo: la Ley de Amparo no está en Qdrant, y la Constitución y la
    Ley de Amparo están además en la lista de «notorias» que
    `preceptos_fuera` SALTA a propósito, porque rigen el juicio pase lo que
    pase. Un «artículo 274 de la Ley de Amparo» inventado pasaba entero.
  · `busqueda_web.texto_de_articulo` sí va a internet, pero sólo para los que
    el resolver no encontró y mientras se escribe, no sobre lo firmado.
  · `documento_generado.cotejar_articulo` compara la cabecera con el cuerpo
    transcrito: caza el texto equivocado, no el artículo inexistente.

Este barrido corre AL FINAL, sobre el .docx ya compuesto, y pregunta una sola
cosa por artículo: ¿existe? Una llamada para todos —no una por artículo—, con
tope de tiempo, y nunca lanza: si no se puede comprobar, se dice que no se
comprobó, que no es lo mismo que decir que está bien.

CONSERVADOR A PROPÓSITO. Acusar a un proyecto bueno es peor que no acusar: se
leen veinte avisos y los falsos entierran a los verdaderos. Sólo se acusa lo
que el buscador niega EXPRESAMENTE. Lo que no se pudo comprobar se cuenta
aparte, sin nombre de culpable.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Optional

# El motor. `perplexity/sonar` es el mismo de `busqueda_web`: busca siempre
# —es su única función—, devuelve las citas y está medido en 4.9 s. Se deja en
# una variable de entorno para poder probar otro sin desplegar.
BARRIDO_MODELO = os.getenv("BARRIDO_PRECEPTOS_MODELO", "perplexity/sonar")
# CUÁNTOS SE PREGUNTAN Y CÓMO. Medido el 23-sep-2026 sobre la v6 del ADC
# 93/2026: el documento cita 56 artículos distintos. Con un tope de 12 y una
# sola llamada, una cita inventada metida al final NO se cazaba —el tope la
# dejaba fuera, en silencio, que es la peor forma de no comprobar—. Se
# pregunta TODO, en lotes que corren a la vez: tres llamadas de quince tardan
# lo que la más lenta, no la suma.
BARRIDO_TOPE = int(os.getenv("BARRIDO_PRECEPTOS_TOPE", "60"))
BARRIDO_LOTE = int(os.getenv("BARRIDO_PRECEPTOS_LOTE", "15"))
BARRIDO_SEGUNDOS = float(os.getenv("BARRIDO_PRECEPTOS_SEGUNDOS", "25"))
# Interruptor propio: se apaga desde Render si sale caro o ruidoso.
BARRIDO_ACTIVO = os.getenv("BARRIDO_PRECEPTOS", "1").lower() in ("1", "true", "si", "sí")

# Leyes cuyo articulado NO se pregunta: las que el acervo ya verificó entran
# por otra puerta, y las de numeración larguísima y estable —los códigos
# civiles— sólo añadirían ruido si el buscador falla. Vacío por ahora: se
# llena con lo que la medición diga.
_SIN_PREGUNTAR: tuple = ()

_RX_JSON = re.compile(r"\{.*\}", re.S)

# ── EL NOMBRE DE LA LEY TERMINA DONDE EMPIEZA EL VERBO ──────────────────────
# `litis_normativa.citas` devuelve la ley con lo que venía detrás pegado —«…
# Contencioso Administrativo establecía la», «Código Fiscal de la Federación
# regulaban el recurso»— porque a ella le basta para saber de qué fuero es.
# Aquí el nombre se manda a un buscador, así que tiene que ser el nombre.
# Ninguna de estas palabras aparece DENTRO del título de un ordenamiento
# mexicano; la primera que asome, corta.
_RX_FIN_DE_LEY = re.compile(
    r"\s+(?:establec\w*|dispon\w*|regulan?|regulab\w*|prev[ée]\w*|se[ñn]al\w*|"
    r"exig\w*|orden\w*|permit\w*|indic\w*|refier\w*|dice|dicen|contempl\w*|"
    r"sostien\w*|oblig\w*|facult\w*|autoriz\w*|aplicable\w*|vigente\w*|"
    r"invocad\w*|citad\w*|mencionad\w*|transcrit\w*|en\s+relaci[óo]n|"
    r"deriv\w*|result\w*|impon\w*|confier\w*|reconoc\w*|garantiz\w*|proteg\w*|"
    # «y» NO corta por sí sola: hay títulos que la llevan dentro —Ley de
    # Ciencia y Tecnología, Ley de Adquisiciones, Arrendamientos y Servicios—.
    # Corta cuando lo que sigue no puede ser parte de un título.
    r"en\s+cuanto\b|y\s+(?:a\s+la|a\s+los|mediante|un\b|una\b|el\s+criterio|"
    r"decret\w*|declar\w*|con\s+ello)|"
    r"as[íi]\s+como|que\b|cuyo\b|cuya\b|porque\b|pues\b|para\s+el|"
    r"vigente|abrogad\w*|reformad\w*)\b.*$", re.I)


def limpia_ley(x: str) -> str:
    """El nombre del ordenamiento, sin lo que la frase le pegó detrás."""
    t = " ".join(str(x or "").split()).strip(" .,;:")
    t = _RX_FIN_DE_LEY.sub("", t).strip(" .,;:")
    return t


def _es_de_rubro(nombre: str) -> bool:
    """¿La cita vive dentro del rubro de una tesis?

    Los rubros van en MAYÚSCULAS y suelen nombrar un artículo —«…EL TÉRMINO
    DE VEINTE DÍAS QUE ESTABLECE EL ARTÍCULO 210 DEL CÓDIGO FISCAL DE LA
    FEDERACIÓN PARA AMPLIARLA»—. Ese artículo NO lo está citando el proyecto:
    lo nombra el criterio que transcribe, muchas veces de una ley abrogada.
    Preguntarlo es preguntar por algo que nadie afirmó, y el buscador
    contestó que el Código Fiscal «termina en el 83»: una acusación falsa
    sobre una cita que ni siquiera es del proyecto.
    """
    letras = [c for c in nombre if c.isalpha()]
    return bool(letras) and sum(c.isupper() for c in letras) >= 0.9 * len(letras)


def _norm_ley(x: str) -> str:
    return limpia_ley(x).lower()


def pares_citados(texto: str) -> list:
    """[(ley, número)] que el documento cita, en orden y sin repetir.

    Se apoya en `litis_normativa.citas`, que ya resuelve las anáforas —«el
    artículo 55 de la propia ley»— a la última ley nombrada antes. Sin eso,
    la mitad de las citas de un proyecto quedan sin ordenamiento.
    """
    try:
        import litis_normativa as _ln
    except Exception:
        return []
    fuera, vistos = [], set()
    for h in _ln.citas(texto or ""):
        ley = limpia_ley(h.get("ley"))
        if len(ley) < 10 or _es_de_rubro(ley):
            continue
        if any(s in ley.lower() for s in _SIN_PREGUNTAR):
            continue
        for n in (h.get("nums") or []):
            clave = (_norm_ley(ley), str(n))
            if clave in vistos:
                continue
            vistos.add(clave)
            fuera.append((ley, str(n)))
    return fuera


def _ya_verificados(material) -> set:
    """Los (ley, número) que el acervo o la fuente oficial ya confirmaron.

    `material.normas` son los preceptos traídos del acervo; los que llegaron
    de internet viajan ahí también, con su `url`. Los dos están comprobados
    contra su fuente: preguntarlos otra vez es gastar por gusto.
    """
    fuera = set()
    for n in (getattr(material, "normas", None) or []):
        if not isinstance(n, dict):
            continue
        if not str(n.get("texto") or "").strip():
            continue
        fuera.add((_norm_ley(n.get("cuerpo_legal")), str(n.get("articulo") or "")))
    return fuera


def por_preguntar(texto: str, material=None, tope: int = 0) -> list:
    """Lo que hay que preguntar: lo citado menos lo ya verificado, y PRIMERO
    lo que ninguna otra capa mira.

    Un proyecto cita sesenta artículos y no se van a preguntar sesenta. El
    orden no es caprichoso: arriba van los de una ley de la que el acervo no
    confirmó NI UNO —ahí no hay nada detrás comprobando, y es donde cabe la
    cita inventada—; después, el resto, por orden de aparición.
    """
    ya = _ya_verificados(material) if material is not None else set()
    leyes_vistas = {ley for ley, _ in ya}
    fuera = [(ley, num) for ley, num in pares_citados(texto)
             if (_norm_ley(ley), num) not in ya]
    fuera.sort(key=lambda x: 0 if _norm_ley(x[0]) not in leyes_vistas else 1)
    return fuera[:(tope or BARRIDO_TOPE)]


def _fuera_de_rango(num, ultimo) -> bool:
    """¿El número citado pasa del último artículo del ordenamiento?

    CON MARGEN, y generoso a propósito: los ordenamientos numeran «115 Bis» y
    «58-6», se reforman y añaden artículos, y el último que devuelva el
    buscador puede estar desactualizado. Un 10% más diez sobra para eso y
    sigue cazando lo que de verdad está inventado, que nunca falla por poco
    —el 884 de una ley de 271—.
    """
    try:
        tope = int(ultimo)
        pedido = int(re.match(r"\d+", str(num)).group(0))
    except (TypeError, ValueError, AttributeError):
        return False
    return tope >= 20 and pedido > tope * 1.1 + 10


def _prompt(pares: list) -> str:
    lista = "\n".join(f"{i}. Artículo {num} de la {ley}"
                      for i, (ley, num) in enumerate(pares, 1))
    return f"""Eres un verificador de citas legales mexicanas. Para CADA artículo de la
lista, busca en fuentes oficiales de México —diputados.gob.mx, dof.gob.mx,
scjn.gob.mx, los congresos estatales, ordenjuridico.gob.mx— y responde SÓLO
dos cosas:

1. `existe`: true si ESE ordenamiento tiene ESE número de artículo vigente o
   derogado; false SÓLO si compruebas que el ordenamiento no llega a ese
   número o que ese artículo no existe en él. Si no lo encuentras pero
   tampoco puedes negarlo, pon `existe: null`. NO adivines: `null` es una
   respuesta correcta y `false` es una acusación.
2. `materia`: en UNA línea de diez palabras, qué regula. Cadena vacía si no
   lo sabes. Sirve para que una persona lo coteje, no para acusar.

LOS ARTÍCULOS:
{lista}

Devuelve SÓLO un JSON, sin texto alrededor:
{{"resultados": [{{"n": 1, "existe": true, "materia": "…", "fuente": "dominio.gob.mx"}}]}}"""


async def _preguntar(pares: list, segundos: float) -> list:
    """Una sola llamada para todos. [] si no se pudo (nunca lanza)."""
    clave = os.getenv("OPENROUTER_API_KEY", "")
    if not clave:
        print("   🌐 BARRIDO: falta OPENROUTER_API_KEY — no se comprueba")
        return []
    import httpx
    try:
        async with httpx.AsyncClient(timeout=segundos) as cli:
            r = await cli.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {clave}"},
                json={"model": BARRIDO_MODELO, "temperature": 0,
                      "messages": [{"role": "user", "content": _prompt(pares)}]})
            r.raise_for_status()
            crudo = (r.json()["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:
        print(f"   🌐 BARRIDO: la consulta falló ({type(e).__name__}) — no se comprueba")
        return []
    m = _RX_JSON.search(crudo)
    if not m:
        print(f"   🌐 BARRIDO: sin JSON («{crudo[:90]}»)")
        return []
    try:
        datos = json.loads(m.group(0))
    except Exception:
        return []
    return [d for d in (datos.get("resultados") or []) if isinstance(d, dict)]


# ═══ NUNCA SE ACUSA CON UNA SOLA RESPUESTA ═════════════════════════════════
# Medido el 23-sep-2026 sobre la v6 del ADC 93/2026, con lotes de quince: el
# buscador marcó como inexistentes el artículo 17 de la LFPCA, el 17 de la
# Constitución, el 183 de la Ley de Amparo y el 145 del Código Fiscal —los
# cuatro existen y los cuatro se citan bien— y además acusó a UNOS en una
# corrida y a OTROS en la siguiente. Es la veleidad que `busqueda_web` ya
# tenía documentada: la abstención no es del buscador, es de la formulación,
# y en lote reparte la atención entre quince preguntas.
#
# Así que un `false` del barrido no acusa a nadie: abre una SEGUNDA pregunta,
# a solas y pidiendo la primera línea del artículo, que es lo que obliga a
# mirarlo. Sólo lo que las dos niegan se escribe. Cuesta una llamada por
# sospechoso —y son pocos—, corren en paralelo y lo no confirmado baja a «no
# se pudo comprobar», que es lo honrado.
def _prompt_uno(ley: str, num: str) -> str:
    return f"""¿Existe el artículo {num} de la {ley} (México)? Busca el texto vigente en
su fuente oficial —diputados.gob.mx, dof.gob.mx, el congreso estatal que
corresponda, ordenjuridico.gob.mx—.

Contesta SÓLO con un JSON:
{{"existe": true, "primera_linea": "<las primeras diez palabras del artículo,
tal cual, si existe>", "cuantos_articulos": <cuántos tiene ese ordenamiento, o
null>, "fuente": "<dominio>"}}

REGLAS:
- `true` si lo encuentras, aunque esté derogado o reformado.
- `false` SÓLO si compruebas que ese ordenamiento NO llega a ese número o que
  ese artículo no existe en él. Es una acusación de cita inventada: no la
  hagas por no haberlo encontrado a la primera.
- `null` si no lo encuentras y tampoco puedes negarlo. Es una respuesta
  correcta y preferible a equivocarse."""


async def _confirmar(pares: list, segundos: float) -> set:
    """Los (ley, num) que una SEGUNDA pregunta, a solas, vuelve a negar."""
    clave = os.getenv("OPENROUTER_API_KEY", "")
    if not clave or not pares:
        return set()
    import httpx

    async def _uno(ley: str, num: str):
        try:
            async with httpx.AsyncClient(timeout=segundos) as cli:
                r = await cli.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {clave}"},
                    json={"model": BARRIDO_MODELO, "temperature": 0,
                          "messages": [{"role": "user",
                                        "content": _prompt_uno(ley, num)}]})
                r.raise_for_status()
                crudo = (r.json()["choices"][0]["message"]["content"] or "")
            m = _RX_JSON.search(crudo)
            d = json.loads(m.group(0)) if m else {}
            if d.get("existe") is False:
                return (ley, num)
            # ── LA ARITMÉTICA, QUE NO TIENE VELEIDADES ───────────────────
            # «¿Existe el 884 de la Ley de Amparo?» se contesta unas veces
            # que sí y otras que no; «¿cuántos artículos tiene la Ley de
            # Amparo?» se contesta 271 siempre, porque es un dato, no un
            # juicio. Si el número citado pasa del último, la cita no puede
            # existir, lo diga el buscador o no. Medido: es lo que sube la
            # detección del 50% al 100% sin acusar a nadie de más.
            try:
                total = int(d.get("cuantos_articulos"))
                pedido = int(re.match(r"\d+", str(num)).group(0))
            except (TypeError, ValueError, AttributeError):
                return None
            if total >= 20 and pedido > total:
                print(f"   🌐 BARRIDO: art. {num} > {total} artículos de «{ley[:40]}»")
                return (ley, num)
            return None
        except Exception:
            return None            # sin confirmación no hay acusación

    try:
        res = await asyncio.wait_for(
            asyncio.gather(*[_uno(l, n) for l, n in pares],
                           return_exceptions=True),
            timeout=segundos + 5)
    except Exception:
        return set()
    return {x for x in res if isinstance(x, tuple)}


async def barrer(texto: str, material=None, tope: int = 0,
                 segundos: float = 0, preguntar=None, confirmar=None) -> dict:
    """{inexistentes, sin_comprobar, comprobados, avisos} del documento entero.

    `preguntar` se inyecta en las pruebas; en producción es la llamada real.
    """
    pares = por_preguntar(texto, material, tope)
    salida = {"inexistentes": [], "sin_comprobar": [], "comprobados": 0,
              "preguntados": len(pares), "avisos": []}
    if not pares:
        return salida
    if not BARRIDO_ACTIVO and preguntar is None:
        return salida
    fn = preguntar or _preguntar
    tope_s = segundos or BARRIDO_SEGUNDOS
    # EN LOTES, Y A LA VEZ. Cada lote numera del 1 al n; al recoger se
    # devuelve el número al índice que tiene en la lista entera.
    lotes = [pares[i:i + BARRIDO_LOTE] for i in range(0, len(pares), BARRIDO_LOTE)]
    try:
        tandas = await asyncio.wait_for(
            asyncio.gather(*[fn(l, tope_s) for l in lotes], return_exceptions=True),
            timeout=tope_s + 5)
    except asyncio.TimeoutError:
        print("   🌐 BARRIDO: se pasó del tope de tiempo — no se comprueba")
        tandas = []
    except Exception as e:
        print(f"   🌐 BARRIDO: {type(e).__name__} — no se comprueba")
        tandas = []

    por_n = {}
    for k, tanda in enumerate(tandas):
        if isinstance(tanda, BaseException) or not tanda:
            continue
        base = k * BARRIDO_LOTE
        for d in tanda:
            if not isinstance(d, dict):
                continue
            try:
                por_n[base + int(d.get("n"))] = d
            except (TypeError, ValueError):
                continue
    sospechosos = []
    for i, (ley, num) in enumerate(pares, 1):
        d = por_n.get(i) or {}
        existe = d.get("existe")
        # AQUÍ NO ENTRA LA ARITMÉTICA, y se probó: pedirle al buscador el
        # último artículo del ordenamiento para descartar por rango contestó
        # que el Código Fiscal termina en el 83 y que la Ley de Amparo
        # termina en el 107 —tienen cerca de 300 y 271—, y con eso acusó de
        # inventados al 237 del Código Fiscal y a los artículos 171 y 172 de
        # la Ley de Amparo, que se citan bien. El dato tampoco era un dato.
        # La comprobación por rango se queda SÓLO en la segunda pregunta, a
        # solas, donde se midió que no acusa de más.
        if existe is False:
            sospechosos.append((ley, num))
        elif existe is True:
            salida["comprobados"] += 1
        else:
            salida["sin_comprobar"].append({"ley": ley, "articulo": num})

    confirmados = set()
    if sospechosos:
        try:
            confirmados = await (confirmar or _confirmar)(sospechosos, tope_s)
        except Exception as e:
            print(f"   🌐 BARRIDO: la confirmación falló ({type(e).__name__})")
            confirmados = set()
        print(f"   🌐 BARRIDO: {len(sospechosos)} sospechoso(s) · "
              f"{len(confirmados)} confirmado(s) como inexistentes")
    for ley, num in sospechosos:
        if (ley, num) in confirmados:
            salida["inexistentes"].append({"ley": ley, "articulo": num, "fuente": ""})
        else:
            salida["sin_comprobar"].append({"ley": ley, "articulo": num})

    if salida["inexistentes"]:
        salida["avisos"].append(
            "ARTÍCULO(S) QUE EL BARRIDO NO ENCUENTRA EN SU ORDENAMIENTO: "
            + " · ".join(f"artículo {x['articulo']} de la {x['ley']}"
                         for x in salida["inexistentes"][:6])
            + ". Se preguntó DOS VECES a su fuente oficial en línea —la segunda "
              "artículo por artículo— y las dos veces contestó que ese "
              "ordenamiento no tiene ese artículo. COMPRUÉBALO antes de firmar: "
              "si de verdad no existe, la cita está inventada y hay que "
              "quitarla o corregirla.")
    if salida["sin_comprobar"] and salida["preguntados"]:
        salida["avisos"].append(
            f"{len(salida['sin_comprobar'])} de {salida['preguntados']} "
            f"artículo(s) citados NO SE PUDIERON COMPROBAR en línea "
            f"({'; '.join(f'art. ' + x['articulo'] + ' — ' + x['ley'][:48] for x in salida['sin_comprobar'][:4])}). "
            f"No es que estén mal: es que el barrido no pudo confirmarlos. Los "
            f"que sí están en el acervo no se preguntan, porque ya se "
            f"verificaron contra él.")
    print(f"   🌐 BARRIDO: {salida['preguntados']} preguntados · "
          f"{salida['comprobados']} confirmados · "
          f"{len(salida['inexistentes'])} inexistentes · "
          f"{len(salida['sin_comprobar'])} sin comprobar")
    return salida
