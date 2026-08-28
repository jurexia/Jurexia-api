"""Prueba las Fases 1-3 sobre casos reales del corpus, con luna por la API de
OpenAI —el mismo motor y el mismo cliente que Redacción Pro—.

Mide lo que se puede medir sin opinar:
  · DATOS INVENTADOS: fechas y cantidades que están en el resumen y NO en el
    documento de origen. Cero tolerancia; es el único fallo que descalifica.
  · cobertura de lo que el propio secretario escribió del mismo asunto.
  · tiempo verbal, largo y limpieza, con las comprobaciones deterministas.
"""
import asyncio, json, os, pathlib, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openai import AsyncOpenAI
import fases123_pipeline as fp

CASOS = "/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/redactor-sentencias/corpus/casos"
RX_FECHA = re.compile(r"\b(?:uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|dieciséis|diecisiete|dieciocho|diecinueve|veinte|veintiuno|veintidós|veintitrés|veinticuatro|veinticinco|veintiséis|veintisiete|veintiocho|veintinueve|treinta|treinta y uno)\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)", re.I)
RX_CIFRA = re.compile(r"\$\s?[\d,]+(?:\.\d{2})?|\b\d{1,3}(?:,\d{3})+(?:\.\d{2})?\b")

from fase0_oportunidad import _UNIDADES
_LETRA_A_CIFRA = {p: str(i) for i, p in enumerate(_UNIDADES) if p}

def normalizar(s):
    """Minúsculas, espacios colapsados y FECHAS EN LETRA PASADAS A CIFRA.

    Sin esto el detector grita invención en falso: el documento escribe «21 de
    febrero» y el resumen —correctamente, porque así se escribe en una
    sentencia— pone «veintiuno de febrero». Pasó en el ADA 103-2025 y me hizo
    acusar al modelo de inventar un dato que estaba en el expediente.
    """
    s = re.sub(r"\s+", " ", s.lower())
    for letra, cifra in sorted(_LETRA_A_CIFRA.items(), key=lambda x: -len(x[0])):
        s = s.replace(f"{letra} de ", f"{cifra} de ")
    return s

def inventados(resumen, fuente):
    f = normalizar(fuente)
    out = []
    for rx in (RX_FECHA, RX_CIFRA):
        for m in rx.finditer(resumen):
            t = normalizar(m.group(0))
            if t not in f:
                out.append(m.group(0))
    return sorted(set(out))

def cobertura(resumen, referencia):
    stop = set("de la el los las y o del en que se a por para su con al es lo un una".split())
    A = {w for w in re.findall(r"[a-záéíóúñ]{5,}", normalizar(resumen)) if w not in stop}
    B = {w for w in re.findall(r"[a-záéíóúñ]{5,}", normalizar(referencia)) if w not in stop}
    return len(A & B) / max(1, len(B))

async def main():
    cliente = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    casos = []
    for p in sorted(pathlib.Path(CASOS).glob("*.json")):
        d = json.load(open(p))
        if not isinstance(d, dict): continue
        pz = d.get("piezas", {})
        if pz.get("acto", {}).get("texto") and pz.get("demanda", {}).get("texto") and d.get("oro"):
            casos.append(d)
    print(f"motor: {fp.MODELO_FASES} (esfuerzo {fp.ESFUERZO_FASES}) · casos disponibles: {len(casos)}\n")

    for c in casos[:int(os.environ.get("N_CASOS", "2"))]:
        acto, conc, oro = c["piezas"]["acto"]["texto"], c["piezas"]["demanda"]["texto"], c["oro"]
        t0 = time.perf_counter()
        f = await fp.correr(cliente, acto, conc)
        dt = time.perf_counter() - t0
        inv_a = inventados(f.resumen_acto, acto)
        inv_c = inventados(f.resumen_conceptos, conc)
        print(f"══ {c['asunto']}  ({dt:.0f}s)")
        print(f"   resumen del acto ..... {len(f.resumen_acto.split()):4d} palabras · cobertura del texto de David {cobertura(f.resumen_acto, oro[:4000]):.0%}")
        print(f"   resumen de conceptos . {len(f.resumen_conceptos.split()):4d} palabras")
        print(f"   DATOS INVENTADOS ..... acto {len(inv_a)} · conceptos {len(inv_c)}"
              + (f"  ⚠ {(inv_a+inv_c)[:4]}" if inv_a or inv_c else "  ✔"))
        print(f"   problema global ...... {(f.problema_global or '(no salió)')[:110]}")
        print(f"   problemas .............{len(f.problemas)}")
        print(f"   avisos ............... {f.avisos or 'ninguno ✔'}")
        print(f"   arranque: {' '.join(f.resumen_acto.split()[:26])}…\n")

asyncio.run(main())
