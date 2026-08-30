"""El circuito completo del adelanto: de dos PDF a un .docx en la mano.

Encadena lo que ya existe por separado:

    Fase 0   fase0_oportunidad.py   ficha y cómputo — SIN modelo, es aritmética
    Fase 1-3 fases123_pipeline.py   los dos resúmenes y los problemas jurídicos
    Fase 7   ensamblar_adelanto.py  relleno de la plantilla REAL del secretario

Y se detiene donde tiene que detenerse: el sentido del fallo y el criterio no
los pone la máquina. El .docx sale con esos huecos marcados, igual que el
adelanto de papel, y `huecos_pendientes()` los enumera para que nadie firme un
documento con un `*****` dentro.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
from dataclasses import dataclass, field
from typing import Optional

import ensamblar_adelanto as ens
import fase0_oportunidad as f0
import fase_partes as fpartes
import fase6_estudio as f6
import fase6_rag as f6rag
import marco_juridico as mjur
import fases123_pipeline as f123


@dataclass
class Encargo:
    """Lo que el secretario aporta. Todo esto lo sabe de memoria o lo lee de un
    sello; nada de esto justifica pagar el OCR de un expediente."""
    numero: str                       # «512/2026»
    encabezado: str                   # «AMPARO DIRECTO ADMINISTRATIVO: 512/2026»
    quejoso: str
    magistrado: str
    secretario: str
    notificacion: _dt.date
    presentacion: _dt.date
    regla_surtimiento: str = "tja_qro_boletin"
    plazo: int = 15
    responsable: Optional[str] = None
    es_recurso: bool = False
    plantilla: str = ""               # el .docx del propio tribunal
    coleccion_estatal: str = ""       # «leyes_queretaro», para el RAG del fondo


@dataclass
class Resultado:
    ruta: str
    computo: f0.Computo
    fases: f123.Fases123
    huecos: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)
    # Se guardan para poder REENSAMBLAR cuando llegue el criterio, sin releer
    # los PDF ni volver a pagar los resúmenes.
    encargo: Optional["Encargo"] = None
    estudio: str = ""
    advertencias: str = ""
    partes: Optional["fpartes.Partes"] = None

    @property
    def listo_para_el_secretario(self) -> bool:
        """El documento está completo HASTA donde puede estarlo sin criterio."""
        return not self.avisos


async def generar(cliente, e: Encargo, texto_acto: str, texto_conceptos: str,
                  ruta_salida: str) -> Resultado:
    """El circuito entero. `cliente` es el AsyncOpenAI de main.py."""
    avisos: list[str] = []

    # ── Fase 0 — aritmética, sin modelo ──────────────────────────────────
    c = f0.computar(e.notificacion, e.presentacion, e.regla_surtimiento,
                    e.plazo, e.responsable)
    avisos.extend(c.avisos)
    if c.oportuna is False:
        avisos.append("EL CÓMPUTO DA EXTEMPORÁNEA. Compruébalo antes de seguir: "
                      "si es correcto, el asunto no se resuelve en el fondo.")

    # ── Fases 1-3 — lectura ──────────────────────────────────────────────
    # La ficha de partes se hace AQUÍ, con los documentos delante, y viaja al
    # estudio. Sin ella el redactor resuelve los sujetos por proximidad, y en un
    # juicio con reconvención y tercero interesado la proximidad miente.
    f, partes = await asyncio.gather(
        f123.correr(cliente, texto_acto, texto_conceptos, e.es_recurso),
        fpartes.fichar(cliente, texto_acto, texto_conceptos))
    avisos.extend(f.avisos)
    avisos.extend(partes.avisos)

    # ── Fase 7 — el documento ────────────────────────────────────────────
    relleno = ens.Relleno(
        encabezado=e.encabezado, numero_asunto=e.numero, quejoso=e.quejoso,
        magistrado=e.magistrado, secretario=e.secretario,
        oportunidad=f0.parrafo_oportunidad(c),
        antecedentes=f.parrafos_antecedentes(),
        resumen_acto=f.parrafos_acto(),
        resumen_conceptos=f.parrafos_conceptos(),
        problemas=f.parrafos_problemas(),
        es_recurso=e.es_recurso,
    )
    ruta = ens.ensamblar(e.plantilla, relleno, ruta_salida)
    # El documento se lee antes de entregarlo: lo que quedó de la plantilla no
    # se ve leyendo por encima, se ve contándolo.
    avisos.extend(ens.residuo_de_plantilla(ruta, e.numero, e.plantilla))

    return Resultado(ruta=ruta, computo=c, fases=f, encargo=e, partes=partes,
                     huecos=ens.huecos_pendientes(ruta), avisos=avisos)


# ═══════════════════════════════════════════════════════════════════════════
# La segunda mitad: el criterio del secretario entra AQUÍ y sólo aquí
# ═══════════════════════════════════════════════════════════════════════════
#
# El proceso se parte en dos a propósito, porque así es como David lo describió:
# la máquina lee y ordena, él decide, la máquina redacta la demostración. Entre
# `generar()` y `resolver()` hay una persona, y ese es el punto del diseño.
#
#   generar()   →  adelanto con los problemas jurídicos planteados
#   consultar() →  lo que el acervo tiene sobre esos problemas, para que decida
#   resolver()  →  la sentencia, con su criterio dentro


async def consultar(qdrant, embed_juris, embed_leyes,
                    r: Resultado) -> f6.Material:
    """Lo que el acervo dice sobre los problemas del caso.

    Se le enseña ANTES de pedirle el criterio: decidir el sentido sin ver la
    jurisprudencia obligatoria del tema es exactamente el error que este
    utillaje existe para evitar.
    """
    # Los problemas de la Fase 3 son DICCIONARIOS —pregunta, resolvió, combate,
    # impedimento—, no cadenas. Con datos de prueba sintéticos nunca se notó;
    # con el primer caso real, `'dict' object has no attribute 'strip'`.
    problemas = ([r.fases.problema_global] if r.fases.problema_global else [])
    for p in (r.fases.problemas or []):
        pregunta = p.get("pregunta", "") if isinstance(p, dict) else str(p)
        if pregunta:
            problemas.append(pregunta)
        # El impedimento técnico ES una cuestión jurídica que hay que fundar:
        # la inoperancia se razona con tesis, no se declara.
        imp = p.get("impedimento") if isinstance(p, dict) else None
        if isinstance(imp, dict) and imp.get("explicacion"):
            problemas.append(f"¿{imp.get('motivo','inoperancia').capitalize()}: "
                             f"{imp['explicacion']}?")
    coleccion = (r.encargo.coleccion_estatal if r.encargo else "") or None
    return await f6rag.material_del_caso(qdrant, embed_juris, embed_leyes,
                                         problemas, coleccion)


async def resolver(cliente, r: Resultado, criterios: list[f6.Criterio],
                   material: f6.Material, ruta_salida: str,
                   marco: str = "") -> Resultado:
    """La sentencia: el mismo documento, ahora con el estudio de fondo dentro.

    Se REENSAMBLA desde la plantilla en vez de editar el adelanto, porque el
    ensamblador trabaja sobre los formatos de la plantilla original y aplicarlo
    dos veces sobre su propia salida duplica bloques.
    """
    if r.encargo is None:
        raise ValueError("El resultado no trae el encargo: no se puede reensamblar.")
    e = r.encargo

    estudio, advertencias, avisos = await f6.redactar(
        cliente, r.fases.resumen_acto, r.fases.resumen_conceptos,
        criterios, material, e.es_recurso, r.partes, marco)

    relleno = ens.Relleno(
        encabezado=e.encabezado, numero_asunto=e.numero, quejoso=e.quejoso,
        magistrado=e.magistrado, secretario=e.secretario,
        oportunidad=f0.parrafo_oportunidad(r.computo),
        antecedentes=r.fases.parrafos_antecedentes(),
        resumen_acto=r.fases.parrafos_acto(),
        resumen_conceptos=r.fases.parrafos_conceptos(),
        problemas=r.fases.parrafos_problemas(),
        estudio=f6.parrafos(estudio),
        tesis=material.tesis,
        calificaciones=[c.sentido for c in criterios],
        es_recurso=e.es_recurso,
    )
    ruta = ens.ensamblar(e.plantilla, relleno, ruta_salida)
    _, aviso_efectos = ens.formula_resolutivo(relleno.calificaciones)
    if aviso_efectos:
        avisos.append(aviso_efectos)
    # Deduplicado: el aviso del nombre se dispara una vez por párrafo donde
    # aparece, y el secretario no necesita leer tres veces lo mismo.
    for a in ens.avisos_ensamblado:
        if a not in avisos:
            avisos.append(a)
    for a in ens.residuo_de_plantilla(ruta, e.numero, e.plantilla):
        if a not in avisos:
            avisos.append(a)

    return Resultado(ruta=ruta, computo=r.computo, fases=r.fases, encargo=e,
                     partes=r.partes, estudio=estudio, advertencias=advertencias,
                     huecos=ens.huecos_pendientes(ruta),
                     avisos=list(r.avisos) + avisos)


# ── Los ANTECEDENTES ya se generan ────────────────────────────────────────
#
# Medidos sobre 199 apartados reales antes de escribir una línea de prompt:
# 645 palabras en 17 párrafos de 37 —crónica de trámite, frases cortas—, con
# verbos de procedimiento (dictó 186, admitió 112, interpuso 82) y arranques
# fijos («Por auto de», «En proveído de», «Seguido el juicio»).
#
# NO son el resumen del acto, aunque salgan de la misma sentencia: el resumen
# cuenta lo que la responsable RESOLVIÓ y por qué; los antecedentes, lo que
# PASÓ en el juicio de origen. Uno es razonamiento, el otro es crónica. Por eso
# se piden con otro prompt y sobre el documento ENTERO —el recorte del resumen
# se queda con el estudio de fondo, donde el trámite ya no está—.


def resumen_legible(r: Resultado) -> str:
    """Lo que se le enseña al secretario cuando termina el proceso."""
    lineas = [
        f"Adelanto generado: {r.ruta.split('/')[-1]}",
        f"Oportunidad: {'en tiempo' if r.computo.oportuna else 'EXTEMPORÁNEA'} "
        f"· plazo del {r.computo.inicio} al {r.computo.vencimiento} "
        f"({r.computo.plazo} días hábiles)",
        f"Antecedentes: {len(r.fases.antecedentes.split())} palabras "
        f"en {len(r.fases.parrafos_antecedentes())} párrafos",
        f"Resumen del acto: {len(r.fases.resumen_acto.split())} palabras",
        f"Resumen de conceptos: {len(r.fases.resumen_conceptos.split())} palabras",
        f"Problemas jurídicos: {len(r.fases.problemas)}"
        + (" (+ el global)" if r.fases.problema_global else ""),
    ]
    if r.avisos:
        lineas.append("\nAVISOS:")
        lineas += [f"  · {a}" for a in r.avisos]
    if r.huecos:
        lineas.append("\nPENDIENTE DE TU CRITERIO:")
        lineas += [f"  · {h[:100]}" for h in r.huecos]
    return "\n".join(lineas)
