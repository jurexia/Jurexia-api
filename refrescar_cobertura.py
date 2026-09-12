"""
Mide cuántos artículos de cada ley hay REALMENTE en el acervo.

POR QUÉ EXISTE (12-sep-2026)
----------------------------
«Cita mal las leyes» es el motivo escrito en las bajas de septiembre. No era
del modelo ni del prompt: siete códigos civiles estatales casi no están
indexados —Estado de México 10 artículos, Guanajuato 15, Nuevo León 21,
Michoacán 2, Oaxaca 0— y sin nada que recuperar, el modelo responde de memoria.
Y la memoria da números del Código Civil Federal con etiqueta estatal: folio
1946-02, diez artículos atribuidos al código de Puebla, ninguno en ese código,
los diez en el federal.

Las colecciones NO están vacías. Nuevo León tiene 23.530 puntos y Guanajuato
15.070 — de leyes de egresos, códigos electorales y presupuestos. Lo que falta
es justo el código que un litigante usa a diario.

CÓMO SE USA
-----------
    python refrescar_cobertura.py            # mide y escribe en Supabase
    python refrescar_cobertura.py --seco     # sólo imprime

Correrlo DESPUÉS DE CADA INGESTA. Una cobertura vieja es peor que ninguna: haría
callar a la plataforma sobre una ley que ya tiene, o peor, la dejaría hablar de
una que sigue sin tener.

LA COBERTURA ES APROXIMADA y no se disimula: se calcula como
`artículos / último_artículo`, lo que supone numeración continua desde 1. Es lo
normal en los códigos mexicanos, pero hay leyes con numeración salteada y ahí
la cifra queda baja sin que falte nada. Por eso el umbral de aviso es holgado:
sirve para decidir si podemos responder, no para presumir de exactitud.
"""
import os, sys, re, collections
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from supabase import create_client

load_dotenv('.env')
SECO = '--seco' in sys.argv

qd = QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ['QDRANT_API_KEY'], timeout=300)
sb = None if SECO else create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_SERVICE_KEY'])

# Sólo leyes con numeración de artículos. Se ignoran las colecciones de
# jurisprudencia y sentencias, que no se miden así.
cols = sorted(c.name for c in qd.get_collections().collections if c.name.startswith('leyes'))

filas = []
for col in cols:
    porley = collections.defaultdict(set)
    cur = None
    while True:
        pts, cur = qd.scroll(collection_name=col, limit=4000, offset=cur,
                             with_payload=['ley', 'articulo_num'], with_vectors=False)
        if not pts:
            break
        for p in pts:
            pl = p.payload or {}
            ley = (pl.get('ley') or '').strip()
            if not ley:
                continue
            try:
                porley[ley].add(int(pl.get('articulo_num')))
            except (TypeError, ValueError):
                continue          # artículos sin número (transitorios, anexos)
        if cur is None:
            break

    for ley, nums in porley.items():
        if not nums:
            continue
        lo, hi = min(nums), max(nums)
        cob = round(100.0 * len(nums) / hi, 2) if hi else None
        filas.append({'coleccion': col, 'ley': ley[:300], 'articulos': len(nums),
                      'primer_articulo': lo, 'ultimo_articulo': hi, 'cobertura': cob})
    print(f"  {col:<28} {len(porley):>4} leyes")

print(f"\n  {len(filas):,} leyes medidas en {len(cols)} colecciones")
flojas = sorted([f for f in filas if (f['cobertura'] or 100) < 50 and f['ultimo_articulo'] > 200],
                key=lambda f: f['cobertura'] or 0)
print(f"  {len(flojas)} por debajo del 50% con más de 200 artículos:")
for f in flojas[:14]:
    print(f"    {f['coleccion'].replace('leyes_',''):<20} {f['ley'][:46]:<48} "
          f"{f['articulos']:>5}/{f['ultimo_articulo']:<6} {f['cobertura']:>6.1f}%")

if not SECO:
    sb.table('acervo_cobertura').delete().neq('coleccion', '').execute()
    for i in range(0, len(filas), 500):
        sb.table('acervo_cobertura').insert(filas[i:i+500]).execute()
    print(f"\n  ✅ {len(filas):,} filas escritas en acervo_cobertura")
