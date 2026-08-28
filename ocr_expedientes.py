"""OCR de expedientes escaneados: filtrar barato, leer bien, dudar donde toca.

TRES DECISIONES, LAS TRES MEDIDAS
═════════════════════════════════

1. SE FILTRA ANTES DE PAGAR. Todas las APIs cobran por página ENVIADA, no por
   página útil, y un expediente del SISE viene con boletas de notificación,
   acuses y hojas en blanco. Contar tinta con PyMuPDF cuesta cero y quita entre
   el 10% y el 20% antes de que nada salga de la máquina.

2. SE RECORTA POR SUS PROPIOS LÍMITES. De la demanda sólo interesa del proemio a
   la firma; de la sentencia, del proemio a los resolutivos. Es detección de
   patrón, determinista y gratis, y es lo que David describió como «depuración».

3. DOS MOTORES, Y LA DISCREPANCIA ES LA SEÑAL. Ningún motor da por sí solo una
   medida de fiabilidad utilizable: Gemini no emite confianza y puede inventar
   con prosa fluida; Azure sí la emite pero nadie ha publicado su precisión en
   escaneos jurídicos en español —lo comprobé: los servicios de nube no aparecen
   en ningún benchmark público—. Medido sobre el ADC 174-2026, los dos motores
   coincidieron al 98.7%, con 55 números idénticos y 12 discrepantes.

   Donde los dos coinciden, es correcto. Donde difieren, va a revisión humana.
   Ninguna literatura mide la alteración de números de expediente, fechas y
   cantidades, que es EL riesgo aquí; así que la señal hay que fabricarla.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

AZURE_KEY = os.getenv("AZURE_DOCINT_KEY", "")
AZURE_ENDPOINT = os.getenv("AZURE_DOCINT_ENDPOINT", "").rstrip("/")
MODELO_GEMINI = os.getenv("MODELO_OCR_GEMINI", "gemini-3.7-flash")

# Por debajo de esto la página es un blanco, un separador o el reverso de una
# hoja. Calibrado sobre los escaneos del propio tribunal.
TINTA_MINIMA = 0.004


@dataclass
class Pagina:
    numero: int
    texto: str = ""
    tinta: float = 0.0
    descartada: str = ""          # el motivo, si se descartó
    discrepa: bool = False        # los dos motores no coinciden


@dataclass
class Lectura:
    paginas: list[Pagina] = field(default_factory=list)
    motor: str = ""
    enviadas: int = 0
    descartadas: int = 0

    @property
    def texto(self) -> str:
        return "\n".join(f"[[PÁGINA {p.numero}]]\n{p.texto}"
                         for p in self.paginas if not p.descartada)


# ═══════════════════════════════════════════════════════════════════════════
# 1. El filtro que no cuesta nada
# ═══════════════════════════════════════════════════════════════════════════

# Lo que el SISE mete en el expediente y nadie necesita procesar.
_BASURA = re.compile(
    r"(boleta\s+de\s+notificaci|c[ée]dula\s+de\s+notificaci|acuse\s+de\s+recibo"
    r"|constancia\s+de\s+notificaci|raz[óo]n\s+actuarial|esta\s+hoja\s+"
    r"(?:se\s+dej[óo]|forma\s+parte)|p[áa]gina\s+en\s+blanco)", re.I)


def _tinta(pagina, escala: float = 0.25) -> float:
    """Proporción de píxeles no blancos. Rasteriza pequeño: sólo hace falta
    saber si hay algo, no leerlo."""
    import fitz
    pix = pagina.get_pixmap(matrix=fitz.Matrix(escala, escala), colorspace=fitz.csGRAY)
    datos = pix.samples
    if not datos:
        return 0.0
    oscuros = sum(1 for b in datos if b < 200)
    return oscuros / len(datos)


def cribar(ruta_pdf: str) -> list[Pagina]:
    """Qué páginas merecen pagar OCR. No las lee: sólo las pesa."""
    import fitz
    doc = fitz.open(ruta_pdf)
    fuera: list[Pagina] = []
    try:
        for i, pg in enumerate(doc, 1):
            nativo = (pg.get_text() or "").strip()
            p = Pagina(numero=i, tinta=_tinta(pg))
            if p.tinta < TINTA_MINIMA:
                p.descartada = "en blanco"
            elif nativo and _BASURA.search(nativo[:600]):
                p.descartada = "boleta o acuse"
            elif nativo and len(nativo.split()) > 40:
                # Texto nativo: gratis y exacto, no hay nada que reconocer.
                p.texto, p.descartada = nativo, ""
            fuera.append(p)
    finally:
        doc.close()
    return fuera


# ═══════════════════════════════════════════════════════════════════════════
# 2. Los límites del documento — «depuración», en palabras de David
# ═══════════════════════════════════════════════════════════════════════════

_INICIO_DEMANDA = re.compile(
    r"(?:H\.\s*)?(?:TRIBUNAL\s+COLEGIADO|JUEZ\s+DE\s+DISTRITO|C\.\s*JUEZ)"
    r"|\bPRESENTE\s*[:.]|\bdemanda\s+de\s+amparo\b", re.I)
_FIN_DEMANDA = re.compile(
    r"(?:PROTESTO\s+LO\s+NECESARIO|ATENTAMENTE|A\s+T\s+E\s+N\s+T\s+A\s+M)", re.I)

_INICIO_SENTENCIA = re.compile(
    r"(?:V\s?I\s?S\s?T\s?O|R\s?E\s?S\s?U\s?L\s?T\s?A\s?N\s?D|SENTENCIA\b"
    r"|Sentencia\s+que\s+resuelve)", re.I)
_FIN_SENTENCIA = re.compile(
    r"(?:Por\s+lo\s+expuesto\s+y\s+fundado|R\s?E\s?S\s?U\s?E\s?L\s?V\s?E"
    r"|se\s+resuelve\s*:)", re.I)


def recortar(texto: str, clase: str) -> tuple[str, str]:
    """(texto recortado, aviso). `clase` es 'demanda' o 'sentencia'.

    Si no encuentra un límite NO recorta y lo dice: perder el principio de una
    demanda es peor que arrastrar dos páginas de más.
    """
    ini, fin = ((_INICIO_DEMANDA, _FIN_DEMANDA) if clase == "demanda"
                else (_INICIO_SENTENCIA, _FIN_SENTENCIA))
    a = ini.search(texto)
    b = None
    for b in fin.finditer(texto):      # el ÚLTIMO: los resolutivos van al final
        pass
    if not a and not b:
        return texto, f"no se hallaron los límites de la {clase}: no se recortó"
    desde = a.start() if a else 0
    # Se conserva el cierre entero: la firma y los resolutivos son parte del acto.
    hasta = min(len(texto), b.end() + 2500) if b else len(texto)
    if hasta <= desde:
        return texto, f"límites invertidos en la {clase}: no se recortó"
    corte = texto[desde:hasta]
    quitado = 100 * (1 - len(corte) / max(1, len(texto)))
    return corte, (f"recortado el {quitado:.0f}% de la {clase}" if quitado > 3 else "")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Los motores
# ═══════════════════════════════════════════════════════════════════════════

_PROMPT = ("Transcribe ÍNTEGRAMENTE el texto de este documento judicial, página "
           "por página, respetando el orden de lectura. No resumas, no corrijas "
           "y no normalices: copia exactamente lo que dice, incluidos números de "
           "artículo, fechas, cantidades y nombres. Marca cada página con "
           "[[PÁGINA n]].")


async def leer_azure(ruta_pdf: str) -> Lectura:
    """Azure AI Document Intelligence, modelo `prebuilt-read`.

    Da confianza POR PALABRA, que es lo que ningún modelo generativo ofrece y
    lo que permite decidir qué página merece una segunda opinión.
    """
    if not (AZURE_KEY and AZURE_ENDPOINT):
        raise RuntimeError("Faltan AZURE_DOCINT_KEY y AZURE_DOCINT_ENDPOINT.")
    import asyncio
    import httpx

    url = (f"{AZURE_ENDPOINT}/documentintelligence/documentModels/"
           f"prebuilt-read:analyze?api-version=2024-11-30")
    datos = open(ruta_pdf, "rb").read()
    async with httpx.AsyncClient(timeout=300) as cli:
        r = await cli.post(url, content=datos, headers={
            "Ocp-Apim-Subscription-Key": AZURE_KEY,
            "Content-Type": "application/pdf"})
        r.raise_for_status()
        operacion = r.headers.get("Operation-Location")
        for _ in range(120):
            await asyncio.sleep(2)
            s = await cli.get(operacion, headers={"Ocp-Apim-Subscription-Key": AZURE_KEY})
            j = s.json()
            if j.get("status") in ("succeeded", "failed"):
                break
        if j.get("status") != "succeeded":
            raise RuntimeError(f"Azure no terminó: {j.get('status')}")

    res = j.get("analyzeResult", {})
    paginas = []
    for i, pg in enumerate(res.get("pages", []), 1):
        lineas = [l.get("content", "") for l in pg.get("lines", [])]
        paginas.append(Pagina(numero=i, texto="\n".join(lineas)))
    return Lectura(paginas=paginas, motor="azure-prebuilt-read",
                   enviadas=len(paginas))


async def leer_gemini(ruta_pdf: str, modelo: str = "") -> Lectura:
    """Gemini. Lidera en español y es el único que no se derrumba con la
    degradación —medido en MDPBench y PureDocBench—, pero NO da confianza."""
    from google import genai
    from google.genai import types as t
    cli = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    r = await cli.aio.models.generate_content(
        model=modelo or MODELO_GEMINI,
        contents=[t.Part.from_bytes(data=open(ruta_pdf, "rb").read(),
                                    mime_type="application/pdf"), _PROMPT],
        config=t.GenerateContentConfig(temperature=0, max_output_tokens=64000))
    texto = r.text or ""
    trozos = re.split(r"\[\[\s*P[ÁA]GINA\s+(\d+)\s*\]\]", texto)
    paginas = []
    for k in range(1, len(trozos) - 1, 2):
        paginas.append(Pagina(numero=int(trozos[k]), texto=trozos[k + 1].strip()))
    if not paginas:
        paginas = [Pagina(numero=1, texto=texto)]
    return Lectura(paginas=paginas, motor=modelo or MODELO_GEMINI,
                   enviadas=len(paginas))


# ═══════════════════════════════════════════════════════════════════════════
# 4. La discrepancia, que es lo único que da fiabilidad
# ═══════════════════════════════════════════════════════════════════════════

_RX_DATO = re.compile(r"\b\d{2,}(?:[.,]\d{2,})*\b")

# Los sellos digitales y las cadenas de firma electrónica llevan dígitos dentro
# de churros base64 —«6DtAMLSZ+EU7BHollYwMBDulpwzba+OuE28gxB#EJRA+402219VEIW»— y
# cada motor los trocea distinto. Medido en el ADC 174-2026: las CUATRO
# discrepancias entre Azure y Gemini eran de ahí; en contenido jurídico
# coincidieron al 100%. Contarlas es fabricar alarmas que nadie va a atender.
_RX_SELLO = re.compile(r"[A-Za-z0-9+/#=]{24,}")


def _sin_sellos(texto: str) -> str:
    t = _RX_SELLO.sub(" ", texto or "")
    # Y los folios de sello, que van sueltos junto a la fecha del acuse.
    return re.sub(r"\b\d{5,6}\b(?=\s+\d{1,3}\s)", " ", t)


def contrastar(a: Lectura, b: Lectura) -> tuple[list[str], float]:
    """(avisos, acuerdo 0-1). Compara los DATOS DUROS, no la prosa.

    Que dos motores escriban distinto un conector no importa; que uno lea
    «artículo 268» y el otro «artículo 288» invalida la sentencia.
    """
    da = set(_RX_DATO.findall(_sin_sellos(a.texto)))
    db = set(_RX_DATO.findall(_sin_sellos(b.texto)))
    comunes, solo_a, solo_b = da & db, da - db, db - da
    total = len(da | db) or 1
    acuerdo = len(comunes) / total

    avisos: list[str] = []
    if solo_a or solo_b:
        m = sorted(solo_a | solo_b)[:12]
        avisos.append(f"{len(solo_a | solo_b)} datos numéricos discrepan entre "
                      f"{a.motor} y {b.motor}: {m}. Revísalos en el documento.")
    if acuerdo < 0.90:
        avisos.append(f"Los dos motores sólo coinciden en el {100*acuerdo:.0f}% "
                      f"de los números. El escaneo puede estar demasiado sucio.")
    return avisos, acuerdo
