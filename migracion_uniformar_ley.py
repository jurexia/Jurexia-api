"""
Uniforma el campo `ley` en las colecciones de leyes.

POR QUÉ
-------
De 38 colecciones, 24 traen `ley` y 14 no. Las 14 son las más pobladas
—Guerrero, CDMX, Nuevo León, Querétaro, Puebla…— unos 230.000 puntos, y son
invisibles para cualquier código que filtre por ese campo.

Medido el 12-sep-2026 sobre siete días de respuestas: las citas a Sonora y
Baja California (que sí tienen `ley`) se verifican 10/10; las de Querétaro,
Nuevo León, Veracruz y Jalisco, 0/10 — teniendo sus artículos dentro.

QUÉ HACE
--------
Copia `cuerpo_legal_oficial` (o `origen` si no lo hay) al campo `ley`.

POR QUÉ ES SEGURO
-----------------
· `set_payload` AÑADE, no reemplaza: ningún campo existente se toca.
· Sólo escribe donde `ley` falta. Si se corre dos veces, la segunda no hace nada.
· En seco por defecto. Escribe sólo con --aplicar.
· Se agrupa por valor: una llamada por nombre de ley y no por punto.
"""
import os, sys, collections
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, IsEmptyCondition, PayloadField

load_dotenv('.env')
cli = QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ['QDRANT_API_KEY'], timeout=120)

APLICAR = '--aplicar' in sys.argv
SOLO = next((a.split('=')[1] for a in sys.argv if a.startswith('--solo=')), None)

SIN_LEY = ['leyes_estatales', 'leyes_michoacan', 'leyes_chihuahua', 'leyes_edomex',
           'leyes_guanajuato', 'leyes_puebla', 'leyes_jalisco', 'leyes_queretaro',
           'leyes_sinaloa', 'leyes_veracruz', 'leyes_morelos', 'leyes_nuevo_leon',
           'leyes_cdmx', 'leyes_guerrero']
objetivo = [SOLO] if SOLO else SIN_LEY

print(f"{'MODO: ESCRITURA' if APLICAR else 'MODO: EN SECO (nada se escribe)'}\n")
tot_p = tot_e = 0
for col in objetivo:
    porvalor = collections.defaultdict(list)
    sin_fuente = 0
    cursor = None
    vistos = 0
    while True:
        pts, cursor = cli.scroll(
            collection_name=col, limit=2000, offset=cursor,
            with_payload=['ley', 'cuerpo_legal_oficial', 'origen'], with_vectors=False)
        if not pts:
            break
        for p in pts:
            pl = p.payload or {}
            vistos += 1
            if pl.get('ley'):
                continue                      # ya lo tiene: idempotente
            nombre = (pl.get('cuerpo_legal_oficial') or pl.get('origen') or '').strip()
            if not nombre:
                sin_fuente += 1
                continue
            porvalor[nombre].append(p.id)
        if cursor is None:
            break

    pendientes = sum(len(v) for v in porvalor.values())
    tot_p += pendientes
    print(f"  {col:<24} puntos={vistos:>7,}  a escribir={pendientes:>7,}  "
          f"leyes distintas={len(porvalor):>4}  sin fuente={sin_fuente}")
    if porvalor:
        ej = sorted(porvalor.items(), key=lambda kv: -len(kv[1]))[:2]
        for nombre, ids in ej:
            print(f"       · «{nombre[:62]}» → {len(ids):,} puntos")

    if APLICAR:
        escritos = 0
        for nombre, ids in porvalor.items():
            for i in range(0, len(ids), 500):
                cli.set_payload(collection_name=col, payload={'ley': nombre},
                                points=ids[i:i+500], wait=False)
                escritos += len(ids[i:i+500])
        tot_e += escritos
        print(f"       ✅ escritos {escritos:,}")

print(f"\n  TOTAL a escribir: {tot_p:,}" + (f"   ·   escritos: {tot_e:,}" if APLICAR else ""))

# ─────────────────────────────────────────────────────────────────────────
# BITÁCORA
#
# 12-sep-2026 · corrido sobre las 14 colecciones. Censo previo:
#
#   con `ley` ................ 24 colecciones
#   con `cuerpo_legal_oficial` 12  (chihuahua, edomex, guanajuato, guerrero,
#                                   jalisco, michoacan, morelos, nuevo_leon,
#                                   puebla, queretaro, sinaloa, veracruz)
#   sólo con `origen` ......... 2  (cdmx, estatales)
#
# Las 14 sin `ley` suman ~230.000 puntos y son las más pobladas del acervo:
# Guerrero 29.938, CDMX 27.196, Nuevo León 23.530, Morelos 16.288, Veracruz
# 16.116, Sinaloa 16.109, Jalisco 15.956, Querétaro 15.858, Guanajuato 15.070,
# Puebla 14.700, Edomex 14.409, Chihuahua 10.211, Michoacán 8.778, estatales
# 5.698.
#
# Piloto en `leyes_michoacan` antes del resto: 8.778 puntos, 152 leyes, cero
# sin fuente. Comprobado después contra producción —el artículo 240 de la Ley
# de Movilidad de Michoacán se encuentra y devuelve su texto— y verificado que
# `cuerpo_legal_oficial`, `articulo_num` y `texto` siguen intactos: set_payload
# añade, no reemplaza.
#
# LO QUE ESTO NO ARREGLA. Que el acervo tenga dos nombres para lo mismo sigue
# siendo cierto: ahora conviven `ley` y `cuerpo_legal_oficial` con el mismo
# valor. Lo correcto a medio plazo es que la INGESTA escriba siempre el mismo
# campo; esto sólo pone al día lo ya ingerido. Mientras las dos convivan, un
# script nuevo puede volver a elegir el campo equivocado.
