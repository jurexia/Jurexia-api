"""FASE 6 — el estudio de fondo, con el criterio del secretario.

Aquí es donde el proyecto entero cobra sentido, y donde se sostiene la regla
que David fijó desde el principio:

    su CRITERIO manda el sentido
    el CORPUS manda la forma
    la LEY manda el fundamento

La máquina no decide el fallo. Construye la mejor demostración posible del
fallo que el secretario ya decidió, y si encuentra un obstáculo serio —una
jurisprudencia obligatoria en contra, una causal de improcedencia— lo SEÑALA
en un apartado de advertencias en lugar de cambiar el sentido por su cuenta.

EL REGISTRO ESTÁ MEDIDO, no inventado. Sobre 40 estudios firmados del corpus:

    largo ............ mediana 3,454 palabras
    párrafo .......... 49 palabras (p90 101)
    frase ............ 35 palabras (p90 69)
    conectores ....... «Lo anterior» 26/40, «En ese sentido» 23, «Por tanto» 22,
                       «En consecuencia» 22, «No obstante» 18, «En efecto» 16
    calificación ..... fundado 103 · infundado 83 · inoperante 67 · ineficaz 15
    la autoridad ..... «la responsable» 27/40, «la autoridad responsable» 25/40
    el órgano ........ «este Tribunal Colegiado» 12/40, y voz impersonal
                       («se estima», «se considera») 31

Y dos recursos que el corpus usa y funcionan, aunque sean minoritarios: la
calificación anunciada en las primeras líneas (40%) y la cuestión jurídica
planteada como pregunta explícita y respondida acto seguido (18%).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

MODELO_ESTUDIO = os.getenv("MODELO_ESTUDIO", "gpt-5.6-luna")

# El estudio SÍ razona: es el único paso del pipeline donde hay que construir
# una demostración, no extraer lo que ya está escrito. Los resúmenes van sin
# razonamiento porque son lectura; esto es argumentación.
ESFUERZO_ESTUDIO = os.getenv("ESFUERZO_ESTUDIO", "high")

PALABRAS_ESTUDIO = 3454

CONECTORES = ("Lo anterior", "En ese sentido", "Por tanto", "En consecuencia",
              "No obstante", "En efecto", "Ahora bien")

# El sentido se dicta en singular («ineficaz») pero se escribe concordando con
# «los conceptos»: «son ineficaces». Sin esto sale «son ineficaz» en la primera
# línea de la sentencia, que es donde más se nota.
_PLURAL = {"fundado": "fundados", "infundado": "infundados",
           "inoperante": "inoperantes", "ineficaz": "ineficaces",
           "fundados": "fundados", "infundados": "infundados",
           "inoperantes": "inoperantes", "ineficaces": "ineficaces"}


def _calificacion(criterios: list["Criterio"]) -> str:
    """La frase de apertura, concordada y en el orden en que se estudian.

    Con sentidos distintos el corpus no dice «fundados e inoperantes» a secas,
    sino que anuncia el resultado mixto: «en parte fundados y en parte
    inoperantes».
    """
    vistos: list[str] = []
    for c in criterios:
        pl = _PLURAL.get(c.sentido.strip().lower(), c.sentido.strip().lower())
        if pl not in vistos:
            vistos.append(pl)
    if not vistos:
        return "el que corresponda"
    if len(vistos) == 1:
        return vistos[0]
    return " y ".join("en parte " + v for v in vistos)


@dataclass
class Criterio:
    """Lo que el secretario decidió para un problema jurídico."""
    problema: str
    sentido: str                      # fundado | infundado | inoperante | ineficaz
    razonamiento: str = ""            # el porqué, que es lo que de verdad alinea


@dataclass
class Material:
    """Lo que el RAG encontró para un problema. Sólo entra lo VERIFICADO."""
    tesis: list[dict] = field(default_factory=list)      # registro, rubro, texto
    normas: list[dict] = field(default_factory=list)     # cuerpo_legal, articulo, texto
    convencional: list[dict] = field(default_factory=list)
    # Los estudios de fondo van APARTE y son molde de FORMA, nunca fundamento.
    moldes: list[dict] = field(default_factory=list)


def _bloque_criterio(criterios: list[Criterio]) -> str:
    if not criterios:
        return ""
    lineas = ["", "═" * 71,
              "EL CRITERIO DEL SECRETARIO — DIRECTIVA INNEGOCIABLE",
              "═" * 71,
              "Tu papel NO es decidir el fallo: es CONSTRUIR la mejor demostración",
              "jurídica posible del sentido que él ya fijó. Elige los argumentos, las",
              "tesis y el orden de estudio que lo sostengan con el mayor rigor.", ""]
    for i, c in enumerate(criterios, 1):
        lineas.append(f"{i}. {c.problema}")
        lineas.append(f"   SENTIDO: {c.sentido.upper()}")
        if c.razonamiento:
            lineas.append(f"   RAZÓN DEL SECRETARIO: {c.razonamiento}")
        else:
            # Sin razón escrita, el modelo tiene que suponerla, y ahí es donde
            # el proyecto deja de parecerse a lo que el secretario pensaba.
            lineas.append("   (sin razón escrita: constrúyela con el material y "
                          "señálalo en las advertencias)")
        lineas.append("")
    lineas += [
        "SI ENCUENTRAS UN OBSTÁCULO SERIO para ese sentido —una jurisprudencia",
        "obligatoria en contra, una causal de improcedencia—, NO cambies el",
        "sentido: dilo en el apartado ADVERTENCIAS para que él lo valore.",
    ]
    return "\n".join(lineas)


def _bloque_material(m: Material) -> str:
    p = ["", "═" * 71, "MATERIAL PARA FUNDAR", "═" * 71]
    if m.tesis:
        p.append("\nTESIS Y JURISPRUDENCIA (existen: salen del acervo, no de tu memoria).")
        p.append("  La OBLIGATORIA vincula a este Tribunal y se invoca como razón que")
        p.append("  decide; la ORIENTADORA sólo ilustra y se cita como apoyo. Tratarlas")
        p.append("  igual es un error de fondo, no de estilo.")
        for t in m.tesis:
            fuerza = "JURISPRUDENCIA OBLIGATORIA" if t.get("obligatoria") else "tesis orientadora"
            p.append(f"\n  · [{fuerza}] Registro {t.get('registro','')} — {t.get('instancia','')}")
            p.append(f"    {t.get('rubro','')}")
            if t.get("localizacion"):
                p.append(f"    {t['localizacion']}")
            p.append(f"    {(t.get('texto') or '')[:900]}")
    if m.normas:
        p.append("\n\nPRECEPTOS:")
        for n in m.normas:
            p.append(f"\n  · {n.get('cuerpo_legal','')} — Art. {n.get('articulo','')}")
            p.append(f"    {(n.get('texto') or '')[:700]}")
    if m.convencional:
        p.append("\n\nCONVENCIONAL:")
        for c in m.convencional:
            p.append(f"\n  · {c.get('rubro','')}\n    {(c.get('texto') or '')[:600]}")
    if m.moldes:
        p.append("\n\nESTUDIOS DE FONDO — SÓLO COMO MODELO DE REDACCIÓN:")
        p.append("  Imita su prosa y su orden. PROHIBIDO citarlos como fundamento:")
        p.append("  no son jurisprudencia. Para fundar, usa las tesis y los preceptos.")
        for e in m.moldes:
            p.append(f"\n  · {e.get('tribunal','')} · {e.get('expediente','')}")
            p.append(f"    {(e.get('holding') or '')[:800]}")
    return "\n".join(p)


def prompt_estudio(resumen_acto: str, resumen_conceptos: str,
                   criterios: list[Criterio], material: Material,
                   es_recurso: bool = False) -> str:
    q = "agravios" if es_recurso else "conceptos de violación"
    calif = _calificacion(criterios)
    return f"""Eres el secretario de un Tribunal Colegiado de Circuito redactando el
estudio de fondo de una sentencia de amparo. Escribes mejor que la media del
oficio: con más orden, más precisión y menos relleno, pero en su mismo registro.

FORMA — medida sobre 40 engroses firmados, no inventada:
- ABRE con el encabezado ordinal y la CALIFICACIÓN: «SEXTO. Estudio. Los {q}
  son {calif}.» Anunciar el resultado y luego demostrarlo es el orden que mejor
  se lee, y el que sigue el 40% de los engroses reales.
- PLANTEA LA CUESTIÓN COMO PREGUNTA y respóndela acto seguido. Lo hace el 18%
  y ordena el estudio entero.
- FRASE de unas 35 palabras, SUBORDINADA; PÁRRAFO de unas 49, es decir UNA O
  DOS FRASES POR PÁRRAFO. Es la medida real del corpus y no es un capricho: la
  prosa judicial encadena la premisa y su consecuencia dentro de la misma
  oración —«toda vez que», «en tanto que», «sin que obste»— en vez de apilar
  cinco frases cortas bajo un mismo párrafo, que es como escribe un informe.
- CONECTORES, por orden de uso real: {', '.join(f'«{c}»' for c in CONECTORES)}.
  No repitas el mismo dos veces seguidas.
- LA AUTORIDAD es «la responsable» o «la autoridad responsable»; el órgano se
  nombra «este Tribunal Colegiado» y usa voz impersonal («se estima», «se
  considera»). Nunca primera persona del singular.
- EXTENSIÓN: alrededor de {PALABRAS_ESTUDIO} palabras. No cortes por brevedad:
  si crees que terminas, desarrolla los efectos de la concesión, los argumentos
  reforzadores y las objeciones previsibles con su refutación.
- Sin Markdown, sin viñetas, sin esquemas.

FUNDAMENTO — la regla que no se rompe:
- Sólo se cita lo que está en el MATERIAL. NUNCA inventes un registro digital
  ni un número de tesis: tus datos de entrenamiento son viejos y falsos.
- Al citar una tesis, transcribe su rubro y su registro tal como vienen.
- La INOPERANCIA se razona: hay que decir POR QUÉ el planteamiento no combate
  la razón toral, no basta con declararla.
{_bloque_criterio(criterios)}
{_bloque_material(material)}

═══════════════════════════════════════════════════════════════════════
LO QUE RESOLVIÓ LA RESPONSABLE
═══════════════════════════════════════════════════════════════════════
{resumen_acto}

═══════════════════════════════════════════════════════════════════════
LO QUE SE COMBATE
═══════════════════════════════════════════════════════════════════════
{resumen_conceptos}

Escribe el estudio de fondo. Si hay obstáculos al sentido fijado, añade al
final un apartado «ADVERTENCIAS» —fuera del cuerpo de la sentencia— con lo que
el secretario debe valorar. Nada más."""


# ═══════════════════════════════════════════════════════════════════════════
# Verificación antes de entregar
# ═══════════════════════════════════════════════════════════════════════════

_RX_REGISTRO = re.compile(r"\b(\d{6,7})\b")
# «ineficaz» lleva z y «ineficaces» c: la raíz «ineficac» sola NUNCA casaba
# la forma singular, y el verificador acusaba de no calificar a un estudio
# que calificaba en su primera línea.
_RX_CALIF = re.compile(r"\b(fundad|infundad|inoperant|inefica[cz])\w*", re.I)

# El nombre de la ley se lee en lo que SIGUE al número, no con un patrón que
# intente prever cómo se escribe. El primer intento exigía «de la|el|los|las» y
# no casaba «del Código Civil», que es la forma más común: cero hallazgos y un
# verificador que parecía funcionar porque nunca decía nada.
_RX_ARTICULO = re.compile(r"art[íi]culos?\s+(\d{1,4})\s*(?:bis|ter)?\.?\s*([^.;:]{0,80})", re.I)

_VACIAS = {"de", "del", "la", "el", "los", "las", "y", "en", "que", "propio",
           "citado", "mencionado", "invocado", "referido", "aludido", "a", "su"}

_NOTORIAS = ("ley de amparo", "constitución", "constitucion",
             "ley orgánica del poder judicial", "ley organica del poder judicial")


def revisar(estudio: str, criterios: list[Criterio], material: Material) -> list[str]:
    """Lo comprobable sin modelo. Ninguna de estas es opinión."""
    avisos: list[str] = []

    # 1. Registros inventados — el fallo que descalifica.
    validos = {str(t.get("registro", "")) for t in material.tesis}
    citados = set(_RX_REGISTRO.findall(estudio))
    inventados = {r for r in citados if r not in validos}
    if inventados:
        avisos.append(f"REGISTROS QUE NO ESTÁN EN EL MATERIAL: {sorted(inventados)}. "
                      "No se citan hasta comprobarlos en el Semanario.")

    # 2. El sentido dictado tiene que aparecer.
    for c in criterios:
        raiz = c.sentido[:7].lower()
        if raiz and raiz not in estudio.lower():
            avisos.append(f"El criterio pedía «{c.sentido}» y esa calificación "
                          f"no aparece en el estudio.")

    # 3. Largo.
    n = len(estudio.split())
    if n < 0.45 * PALABRAS_ESTUDIO:
        avisos.append(f"El estudio tiene {n} palabras; la mediana de los "
                      f"engroses es {PALABRAS_ESTUDIO}. Se quedó corto.")

    # 4. Higiene.
    if "**" in estudio or "##" in estudio:
        avisos.append("Se coló Markdown.")
    # 5. Preceptos citados que no salieron del acervo. Un artículo inventado de
    #    un código sustantivo es tan grave como un registro inventado, y hasta
    #    ahora sólo se vigilaban los registros.
    en_material = {(str(n.get("cuerpo_legal", "")).lower(), str(n.get("articulo", "")))
                   for n in material.normas}
    leyes_material = {c for c, _ in en_material}
    def _voces(x: str) -> set[str]:
        return {w for w in re.findall(r"[\wáéíóúñ]+", x.lower())
                if w not in _VACIAS and len(w) > 2}

    fuera: set[str] = set()
    for art, cola in _RX_ARTICULO.findall(estudio):
        cola_n = " ".join(cola.split()).lower()
        if any(n in cola_n for n in _NOTORIAS):
            continue
        vc = _voces(cola_n)
        for cuerpo in leyes_material:
            vl = _voces(cuerpo)
            # La ley se reconoce por sus voces propias («código», «civil»,
            # «querétaro»), no por igualdad de cadena: el estudio la nombra de
            # mil maneras y ninguna coincide carácter a carácter.
            if vl and len(vc & vl) >= max(2, len(vl) // 2):
                if (cuerpo, art) not in en_material:
                    fuera.add(f"art. {art} — {cuerpo}")
                break
    if fuera:
        avisos.append(f"PRECEPTOS CITADOS QUE NO ESTÁN EN EL MATERIAL: "
                      f"{sorted(fuera)}. Compruébalos antes de firmar.")

    # 6. Largo por exceso. El corpus tiene una medida y pasarse al doble no es
    #    rigor: es repetir el argumento con otras palabras.
    if n > 1.6 * PALABRAS_ESTUDIO:
        avisos.append(f"El estudio tiene {n} palabras frente a las "
                      f"{PALABRAS_ESTUDIO} de mediana. Revisa si hay repetición.")

    # 7. La medida de la prosa. El modelo tiende a apilar frases cortas dentro
    #    de párrafos largos —informe—; el corpus hace lo contrario: párrafo
    #    compacto con la frase subordinada larga. Se avisa, no se corrige solo.
    ps = [x for x in estudio.split("\n") if len(x.split()) > 4]
    if ps:
        med_p = sorted(len(x.split()) for x in ps)[len(ps) // 2]
        if med_p > 75:
            avisos.append(f"Párrafos de {med_p} palabras de mediana frente a las "
                          f"49 del corpus: se lee como informe, no como engrose.")

    if not _RX_CALIF.search(" ".join(estudio.split()[:80])):
        avisos.append("No califica en las primeras líneas: se pierde el orden "
                      "de anunciar y demostrar.")
    return avisos


def parrafos(estudio: str) -> list[str]:
    """El estudio, listo para el ensamblador.

    Se le quita el encabezado «SEXTO. Estudio.» que el modelo escribe, porque el
    ensamblador ya pone el suyo desde la plantilla, y dos encabezados seguidos
    delatan el documento. La CALIFICACIÓN que va pegada a él —«Los conceptos son
    ineficaces»— SE CONSERVA: es la frase que abre el estudio.
    """
    t = re.sub(r"^\s*(?:SEXTO|S[ÉE]PTIMO|QUINTO)\.\s*Estudio(?:\s+de\s+fondo)?\.?\s*",
               "", estudio.strip(), flags=re.I)
    return [p.strip() for p in t.split("\n") if p.strip()]


def separar_advertencias(estudio: str) -> tuple[str, str]:
    """Aparta el apartado de ADVERTENCIAS: no es parte de la sentencia.

    Se le enseña al secretario en pantalla, pero NO entra en el .docx: una
    sentencia no lleva notas del redactor a su lector.
    """
    m = re.search(r"\n\s*ADVERTENCIAS?\s*[:\n]", estudio, re.I)
    if not m:
        return estudio.strip(), ""
    return estudio[:m.start()].strip(), estudio[m.end():].strip()


async def redactar(cliente, resumen_acto: str, resumen_conceptos: str,
                   criterios: list[Criterio], material: Material,
                   es_recurso: bool = False) -> tuple[str, str, list[str]]:
    """Devuelve (estudio, advertencias, avisos)."""
    kw = dict(model=MODELO_ESTUDIO, max_completion_tokens=16000,
              messages=[{"role": "user", "content": prompt_estudio(
                  resumen_acto, resumen_conceptos, criterios, material, es_recurso)}])
    if ESFUERZO_ESTUDIO:
        kw["reasoning_effort"] = ESFUERZO_ESTUDIO
    r = await cliente.chat.completions.create(**kw)
    crudo = (r.choices[0].message.content or "").strip()
    estudio, advertencias = separar_advertencias(crudo)
    return estudio, advertencias, revisar(estudio, criterios, material)
