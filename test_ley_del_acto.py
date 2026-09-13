# -*- coding: utf-8 -*-
"""El acto reclamado se juzga con la ley de la responsable, no con la de amparo.

Nació del amparo en revisión 322/2025: una medida provisional de restricción y
restitución del domicilio, dictada por un juez de San Juan del Río por violencia
familiar. El estudio declaró infundados los agravios fundándose en los artículos
124, 128 y 147 de la Ley de Amparo —el capítulo de la SUSPENSIÓN DEL JUICIO DE
AMPARO— y no citó ni un artículo del Código de Procedimientos Civiles del Estado.

NO FUE UNA CONFUSIÓN DEL MODELO: citó lo único que se le puso delante. De las 24
normas que el pipeline le entregó, TRECE eran de la Ley de Amparo y NINGUNA de
Querétaro. Dos tramos de código lo producían, los dos medidos:

  · `fase6_rag.py`, el silo SUSTITUÍA a la colección estatal. Con materia
    declarada las colecciones quedaban en `["leyes_civil"]` —5,812 puntos, nueve
    ordenamientos federales, 270 trozos de la Ley de Amparo y cero ley estatal—;
    sin materia ni entidad, en `["leyes_federales"]`. Por las dos ramas, lo
    único que el acervo podía ofrecer sobre «medidas cautelares» era la
    suspensión del amparo.
  · Y la búsqueda de la ley estatal iba con la PREGUNTA del recurso, que lleva
    el andamio del amparo. Con ella, `leyes_queretaro` devuelve la Ley de
    Justicia para Adolescentes; con el HECHO —lo que el problema ya trae en
    `combate` y `resolvio`— devuelve primero el artículo 202 del Código de
    Procedimientos Civiles del Estado, que es literalmente el de la restitución
    del domicilio y la restricción de acercamiento.

Se corre sola y necesita red: pega contra Qdrant y contra el embebedor, porque
lo que se vigila es una recuperación y ésa se mueve al reingerir.

    python3 test_ley_del_acto.py
"""
import asyncio
import json
import os
import sys
import urllib.request

sys.path.insert(0, ".")

import fase6_rag as f6r
import tipos_asunto as ta

FALLOS = []


def ok(cond, que):
    print(("  OK   " if cond else "  FALLA ") + que)
    if not cond:
        FALLOS.append(que)


def _env():
    d = {}
    if not os.path.exists(".env"):
        return d
    for l in open(".env", encoding="utf-8"):
        if "=" in l and not l.strip().startswith("#"):
            k, v = l.split("=", 1)
            d[k.strip()] = v.strip().strip('"').strip("'")
    return d


ENV = _env()

# El problema real del 322/2025, tal como lo dejó la fase 3 en la sesión.
PROBLEMA = {
    "pregunta": "¿La medida provisional de restricción y restitución del domicilio "
                "podía subsistir con la fundamentación y motivación expresadas?",
    "combate": "La parte recurrente sostiene que la orden de restricción vulnera sus "
               "derechos y los de su menor hijo, porque le impide acercarse o ingresar "
               "al domicilio en el que afirma residir con éste, y solicita que se "
               "revoque la resolución recurrida y se conceda la suspensión de plano.",
    "resolvio": "Sostuvo que la restitución del domicilio, la salida del cónyuge y la "
                "custodia materna encontraban respaldo provisional en las "
                "manifestaciones de las partes, en que los menores ya vivían con su "
                "madre y en las carpetas de investigación relacionadas con violencia "
                "familiar.",
}


# ═══════════════════════════════════════════════════════════════════════════
# 1 · LA BARANDILLA, QUE NO NECESITA RED
# ═══════════════════════════════════════════════════════════════════════════
print("── 1 · LA REGLA CAMBIA DE FORMA SEGÚN LO QUE EL SISTEMA SEPA ──")
import re
# LA REGLA EN NEGATIVO PRODUCE LO QUE QUIERE EVITAR, y costó un proyecto verlo.
# Decía «los preceptos de la Ley de Amparo que regulan la suspensión rigen la
# suspensión del amparo y nada más», y el estudio del 322/2025 salió con dos
# párrafos explicando exactamente eso. Correcto como derecho, y no es lo que se
# quiere: lo que no se nombra no se escribe.
for tipo in ("amparo_directo", "amparo_revision", "queja", "revision_fiscal"):
    t = ta.ley_de_la_via(tipo, "ordinaria", "principal")
    ok(not re.search(r"suspensi", t, re.I),
       f"«{tipo}» con sede ordinaria: la regla NO nombra la suspensión")
    ok("LA LEY QUE RIGE EL ACTO RECLAMADO" in t,
       f"«{tipo}» con sede ordinaria: se dice en POSITIVO cuál es la ley")

    t2 = ta.ley_de_la_via(tipo, "amparo", "principal")
    ok("SÍ RIGE EL ACTO" in t2,
       f"«{tipo}» con acto de órgano de amparo: esa ley SÍ funda el estudio")
    t3 = ta.ley_de_la_via(tipo, "", "incidental")
    ok("SÍ RIGE EL ACTO" in t3,
       f"«{tipo}» con cuaderno incidental: esa ley SÍ funda el estudio")

    # Sin datos derivados queda la regla general, que sigue siendo necesaria.
    t4 = ta.ley_de_la_via(tipo, "", "")
    ok("NO SE RIGE POR LA LEY DE AMPARO" in t4,
       f"«{tipo}» sin derivar: queda la regla general")

# NI UN NÚMERO QUE NO SEA DE LA LEY DE AMPARO, en ninguna de las variantes. Un
# ejemplo dentro de un prompt acaba copiado literal en la sentencia firmada.
for nombre, regla in (("ordinario", ta._ACTO_ORDINARIO),
                      ("de amparo", ta._ACTO_DE_AMPARO),
                      ("general", ta._ACTO_NO_ES_AMPARO)):
    nums = re.findall(r"\b\d{1,3}\b", regla)
    ok(not nums, f"la regla «{nombre}» no trae números copiables (salió: {nums})")
    for palabra in ("Querétaro", "Código de Procedimientos", "Código Civil"):
        ok(palabra.lower() not in regla.lower(),
           f"la regla «{nombre}» no nombra «{palabra}»")

ok(ta.ley_de_la_via("inventado") == "",
   "un tipo que no existe no recibe la regla a medias")


# ═══════════════════════════════════════════════════════════════════════════
# 1 bis · LOS DOS DATOS QUE SE DERIVAN DEL EXPEDIENTE
# ═══════════════════════════════════════════════════════════════════════════
# David: «no quiero que le impongas al modelo que invoque esa ley, sino que
# modifiques la ARQUITECTURA para que lo entienda». Eso son dos datos, y los dos
# SE LEEN: quién dictó el acto reclamado del amparo indirecto, y de qué cuaderno
# viene la sentencia recurrida.
print("\n── 1 bis · LA SEDE DEL ACTO Y EL CUADERNO ──")
import fase_rama as fr

_RECURRIDA_ORDINARIA = (
    "En la audiencia constitucional celebrada en el juicio de amparo 398/2025, "
    "promovido por Andrés A. M., por derecho propio y en representación de sus "
    "hijos menores de edad, contra actos del Juez Primero de Primera Instancia "
    "Civil de San Juan del Río, Querétaro y del actuario de su adscripción, que "
    "hizo consistir en la orden de restricción dictada como medida provisional.")
sede, quien = fr.sede_del_acto(_RECURRIDA_ORDINARIA)
ok(sede == "ordinaria", f"el acto de un juez de primera instancia es sede ORDINARIA (salió «{sede}»)")
ok("Primera Instancia" in quien, f"y se nombra a quien lo dictó: {quien[:60]}")
cuad, _p = fr.cuaderno_recurrido(_RECURRIDA_ORDINARIA)
ok(cuad == "principal",
   f"la sentencia de la audiencia constitucional viene del cuaderno PRINCIPAL "
   f"(salió «{cuad}»)")

_RECURRIDA_AMPARO = (
    "Interlocutoria dictada en el incidente de suspensión derivado del juicio de "
    "amparo 512/2025, en la que se concedió la suspensión definitiva contra "
    "actos del Juez Cuarto de Distrito en el Estado de Querétaro.")
sede2, _q2 = fr.sede_del_acto(_RECURRIDA_AMPARO)
ok(sede2 == "amparo",
   f"el acto de un juez de DISTRITO es sede de amparo: ahí la Ley de Amparo SÍ "
   f"rige el acto (salió «{sede2}»)")
cuad2, _p2 = fr.cuaderno_recurrido(_RECURRIDA_AMPARO)
ok(cuad2 == "incidental",
   f"la interlocutoria del incidente viene del cuaderno INCIDENTAL (salió «{cuad2}»)")

# ── CALIBRACIÓN: ni se inventa ni se estrecha ─────────────────────────────
# El fallo que se arregló: `responsable_originaria` devolvía «su informe
# justificado» porque el patrón del rótulo hacía los dos puntos opcionales y
# `re.I` anulaba el ancla de mayúscula. Eso viajaba al SEGUNDO punto resolutivo.
ok(fr.responsable_originaria(
    "admitió a trámite la demanda de amparo, solicitó a las autoridades "
    "responsables su informe justificado; dio al Agente del Ministerio "
    "Público la intervención que le compete.") == "",
   "la prosa «a las autoridades responsables su informe justificado» ya no "
   "pasa por nombre de autoridad")
ok(not fr._sirve("Usted y otras autoridades"),
   "«Usted y otras autoridades» no es una autoridad")
# Y NO SE ESTRECHA: un director de ingresos es responsable de las más corrientes
# en amparo administrativo. Un filtro que sólo admitiera órganos que juzgan
# habría tirado la materia entera — lo cazó test_taller_hallazgos.
ok(fr._sirve("Director de Ingresos del Municipio de Querétaro"),
   "un Director de Ingresos SÍ es autoridad: el filtro no puede exigir que "
   "quien dictó el acto sea un órgano jurisdiccional")
ok(fr.sede_del_acto("no dice nada de nadie")[0] == "",
   "sin autoridad legible se calla, que es el fallo correcto")


# ═══════════════════════════════════════════════════════════════════════════
# 2 · LA RECUPERACIÓN, CONTRA EL ACERVO DE VERDAD
# ═══════════════════════════════════════════════════════════════════════════
async def _contra_qdrant():
    from qdrant_client import AsyncQdrantClient

    def _emb(t, modelo):
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=json.dumps({"model": modelo, "input": t}).encode(),
            headers={"Authorization": "Bearer " + ENV["OPENAI_API_KEY"],
                     "Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=60))["data"][0]["embedding"]

    async def embed_leyes(t):
        return await asyncio.to_thread(_emb, t, "text-embedding-3-small")

    async def embed_juris(t):
        return await asyncio.to_thread(_emb, t, "text-embedding-3-large")

    q = AsyncQdrantClient(url=ENV["QDRANT_URL"], api_key=ENV["QDRANT_API_KEY"],
                          timeout=120)
    try:
        def _de_queretaro(ns):
            return [n for n in ns
                    if "quer" in str(n.get("cuerpo_legal") or "").lower()]

        def _de_amparo(ns):
            return [n for n in ns
                    if "amparo" in str(n.get("cuerpo_legal") or "").lower()]

        # ── CON la entidad: la ley del acto tiene que entrar ──────────────
        con = await f6r.material_del_caso(
            q, embed_juris, embed_leyes, [PROBLEMA],
            coleccion_estatal="leyes_queretaro", materia="civil", cliente=None)
        qro = _de_queretaro(con.normas)
        print(f"\n  con entidad: {len(con.normas)} normas · "
              f"{len(qro)} de Querétaro · {len(_de_amparo(con.normas))} de la Ley de Amparo")
        for n in qro[:5]:
            print(f"     art. {n.get('articulo')} · {str(n.get('cuerpo_legal'))[:56]}")
        ok(len(qro) >= 1,
           f"con la entidad declarada entra la ley del acto ({len(qro)} normas "
           f"de Querétaro)")
        ok(any(str(n.get("articulo")).strip() == "202"
               and "procedimientos civiles" in str(n.get("cuerpo_legal") or "").lower()
               for n in qro),
           "entra el artículo 202 del Código de Procedimientos Civiles del "
           "Estado, que es el de la restitución del domicilio y la restricción")

        # ── SIN la entidad: se comporta como antes, ni mejor ni peor ──────
        sin = await f6r.material_del_caso(
            q, embed_juris, embed_leyes, [PROBLEMA],
            coleccion_estatal=None, materia="civil", cliente=None)
        print(f"  sin entidad: {len(sin.normas)} normas · "
              f"{len(_de_queretaro(sin.normas))} de Querétaro")
        ok(len(_de_queretaro(sin.normas)) == 0,
           "sin entidad no se cuela ley estatal por la puerta de atrás: el silo "
           "sigue sustituyendo al corpus general, que es la guarda de siempre")

        # ── CALIBRACIÓN: la cesta del acto SE SUMA, no compite ────────────
        # Si el arreglo hubiera quitado plazas al material de siempre, el
        # estudio saldría peor fundado en todo lo demás. El cupo es propio.
        ok(len(con.normas) >= len(sin.normas),
           f"la cesta del acto SE SUMA: con entidad {len(con.normas)} normas "
           f"frente a {len(sin.normas)} sin ella; nunca menos")
        _base = {(str(n.get("cuerpo_legal")), str(n.get("articulo")))
                 for n in sin.normas}
        _ahora = {(str(n.get("cuerpo_legal")), str(n.get("articulo")))
                  for n in con.normas}
        _perdidas = _base - _ahora
        # Se admite alguna rotación por el corte final, pero no un vaciado.
        ok(len(_perdidas) <= len(_base) // 2,
           f"no se tira lo que antes entraba: se perdieron {len(_perdidas)} de "
           f"{len(_base)} normas del camino sin entidad")
    finally:
        await q.close()


print("\n── 2 · LA LEY DEL ACTO LLEGA AL MATERIAL ──")
if not (ENV.get("QDRANT_URL") and ENV.get("OPENAI_API_KEY")):
    print("  (sin credenciales en .env: esta parte no se corrió)")
    FALLOS.append("no se pudo correr contra el acervo")
else:
    try:
        asyncio.run(_contra_qdrant())
    except Exception as e:
        print(f"  FALLA  no se pudo correr contra el acervo: {e}")
        FALLOS.append(f"acervo: {e}")

print()
if FALLOS:
    print("FALLAS:")
    for f in FALLOS:
        print("  ✗", f)
    raise SystemExit(1)
print("Todo en orden.")
