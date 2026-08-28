"""Genera los dos resúmenes de un caso real y los compara con los de David."""
import asyncio, json, os, pathlib, re, sys, time
sys.path.insert(0, "/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/jurexia-api-git")
from openai import AsyncOpenAI
import fases123_pipeline as fp

OR = AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")
MODELO = os.environ.get("MODELO_FASES", "qwen/qwen3.7-flash")

CASOS = "/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/redactor-sentencias/corpus/casos"

async def pedir(prompt, tope=2500):
    r = await OR.chat.completions.create(
        model=MODELO, max_tokens=tope,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"reasoning": {"enabled": False, "exclude": True}})
    return (r.choices[0].message.content or "").strip()

async def main():
    # Un caso con acto, conceptos y estudio escrito
    casos = []
    for p in sorted(pathlib.Path(CASOS).glob("*.json")):
        d = json.load(open(p))
        if not isinstance(d, dict): continue
        pz = d.get("piezas", {})
        if pz.get("acto", {}).get("texto") and pz.get("demanda", {}).get("texto") and d.get("oro"):
            casos.append(d)
    print(f"casos con las tres piezas: {len(casos)}")
    c = casos[0]
    print(f"caso: {c['asunto']}  ·  motor: {MODELO}\n")

    t0 = time.perf_counter()
    ra = await pedir(fp.prompt_resumen_acto(c["piezas"]["acto"]["texto"]))
    rc = await pedir(fp.prompt_resumen_conceptos(c["piezas"]["demanda"]["texto"]))
    t1 = time.perf_counter()
    pr = await pedir(fp.prompt_problemas(ra, rc), tope=1500)
    t2 = time.perf_counter()

    f = fp.Fases123(resumen_acto=ra, resumen_conceptos=rc)
    try:
        j = json.loads(re.search(r"\{.*\}", pr, re.S).group(0))
        f.problema_global = j.get("problema_global", "")
        f.problemas = j.get("problemas", [])
    except Exception as e:
        f.avisos.append(f"JSON de problemas ilegible: {e}")

    print(f"── RESUMEN DEL ACTO ({len(ra.split())} palabras, {t1-t0:.0f}s los dos)")
    print("   " + ra[:520].replace("\n", "\n   "))
    print(f"\n── RESUMEN DE CONCEPTOS ({len(rc.split())} palabras)")
    print("   " + rc[:520].replace("\n", "\n   "))
    print(f"\n── PROBLEMA GLOBAL ({t2-t1:.0f}s)")
    print("   " + (f.problema_global or "(no salió)")[:300])
    for i, p in enumerate(f.problemas[:3], 1):
        print(f"   {i}. {p.get('pregunta','')[:150]}")
        if p.get("impedimento"): print(f"      ⚠ {p['impedimento'].get('motivo')}")

    print("\n── COMPROBACIONES DETERMINISTAS ──")
    av = fp.revisar(f) + f.avisos
    print("   " + ("\n   ".join(f"⚠ {a}" for a in av) if av else "sin avisos ✔"))

    print("\n── LO QUE ESCRIBIÓ DAVID en el mismo asunto (arranque del estudio) ──")
    print("   " + " ".join(c["oro"].split())[:520])

asyncio.run(main())
