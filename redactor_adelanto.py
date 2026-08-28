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


@dataclass
class Resultado:
    ruta: str
    computo: f0.Computo
    fases: f123.Fases123
    huecos: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)

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
    f = await f123.correr(cliente, texto_acto, texto_conceptos, e.es_recurso)
    avisos.extend(f.avisos)

    # ── Fase 7 — el documento ────────────────────────────────────────────
    relleno = ens.Relleno(
        encabezado=e.encabezado, numero_asunto=e.numero, quejoso=e.quejoso,
        magistrado=e.magistrado, secretario=e.secretario,
        oportunidad=f0.parrafo_oportunidad(c),
        antecedentes=[],                       # se llenan aparte; ver nota abajo
        resumen_acto=f.parrafos_acto(),
        resumen_conceptos=f.parrafos_conceptos(),
        problemas=f.parrafos_problemas(),
        es_recurso=e.es_recurso,
    )
    ruta = ens.ensamblar(e.plantilla, relleno, ruta_salida)

    return Resultado(ruta=ruta, computo=c, fases=f,
                     huecos=ens.huecos_pendientes(ruta), avisos=avisos)


# ── Nota sobre los ANTECEDENTES ───────────────────────────────────────────
#
# El QUINTO. Antecedentes se queda vacío a propósito en esta versión. Salen de
# la sentencia reclamada, igual que el resumen del acto, pero son otra cosa: el
# resumen cuenta lo que la responsable RESOLVIÓ, y los antecedentes cuentan lo
# que PASÓ en el juicio de origen —escrito inicial, admisión, sentencia de
# primera instancia, apelación—.
#
# Se pueden generar con una cuarta llamada sobre el mismo texto, pero antes hay
# que medirlos en el corpus como se midieron los resúmenes. Hacerlo a ojo sería
# volver a inventar el estilo, que es justo lo que este proyecto no hace.


def resumen_legible(r: Resultado) -> str:
    """Lo que se le enseña al secretario cuando termina el proceso."""
    lineas = [
        f"Adelanto generado: {r.ruta.split('/')[-1]}",
        f"Oportunidad: {'en tiempo' if r.computo.oportuna else 'EXTEMPORÁNEA'} "
        f"· plazo del {r.computo.inicio} al {r.computo.vencimiento} "
        f"({r.computo.plazo} días hábiles)",
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
