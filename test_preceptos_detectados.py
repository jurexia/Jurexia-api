#!/usr/bin/env python3
"""LA CITA DEL ESCRITO TIENE QUE LLEGAR CON SU LEY.

Por qué existe esta prueba (16-sep-2026)
----------------------------------------
El paso «Revisar fundamentos» le dijo a un abogado, en producción y luego en un
anuncio de la campaña, que no podía validar los artículos 940 y 942 del Código
de Procedimientos Civiles para el Distrito Federal. Los teníamos: están en
`leyes_cdmx` con su texto. Lo que pasó es que el patrón que extrae las citas
perdía el NOMBRE DE LA LEY —ocho de cada nueve veces, en cualquier ley— porque
el `\\s*` del grupo de fracción se comía el espacio que el `\\s+` del grupo de la
ley exigía, y como ese grupo era opcional no fallaba: devolvía None.

Sin nombre de ley, la resolución se queda con el primer artículo con ese número
que devuelva la base. Para el 942 de la Ciudad de México, el del Código Civil.

ESTE ES EL FALLO QUE MUERE EN SILENCIO: nada peta, nada se registra, la
respuesta sale igual de larga y sólo está mal. Por eso la prueba no mide el
resultado final —que depende del modelo— sino la pieza determinista: que de
cada cita salgan sus números Y su ley.

    IUREXIA-MAC/jurexia-api-git/.venv/bin/python3 test_preceptos_detectados.py

Devuelve 0 si todo pasa y 1 si algo falla, así que sirve de puerta.
"""
import os
import re
import sys

RAIZ = os.path.dirname(os.path.abspath(__file__))


def _cargar_de_main():
    """Ejecuta SÓLO los tramos de main.py que hacen falta.

    Importar main.py entero levanta clientes, lee variables de entorno y tarda.
    Aquí se recortan los tramos por sus anclas y se ejecutan en un espacio
    propio: la prueba corre en menos de un segundo y sin red, y aun así mide el
    código de verdad, no una copia que se queda vieja.
    """
    src = open(os.path.join(RAIZ, 'main.py'), encoding='utf-8').read()
    espacio = {'re': re}

    tramos = [
        # el recortador de nombres de ley, con sus conectores
        ('_CONECTORES_LEY = {', "# «federal», «local» y sus hermanas NO son ruido"),
        # el detector nuevo
        ('# ── EL DETECTOR DE PRECEPTOS CITADOS', 'def _extract_legal_citations(text: str) -> dict:'),
    ]
    for ini, fin in tramos:
        if ini not in src or fin not in src:
            sys.exit(f'ABORTA: no encuentro el tramo «{ini[:40]}» en main.py')
        exec(compile(src[src.index(ini):src.index(fin)], 'main.py', 'exec'), espacio)
    return espacio


E = _cargar_de_main()
detectar_ley = E['_ley_de_la_cola']
numeros = E['_numeros_citados']
RX = E['_RX_CITA']


def citas(texto: str) -> list:
    """[(numeros, ley), …] tal y como las vería `_extract_legal_citations`."""
    salida, ultima = [], ''
    for m in RX.finditer(texto):
        nums = numeros(m.group('nums'))
        if not nums:
            continue
        ley = detectar_ley(m.group('cola'), ultima)
        if ley:
            ultima = ley
        salida.append((nums, ley))
    return salida


CPC = 'Código de Procedimientos Civiles para el Distrito Federal'
CPEUM = 'Constitución Política de los Estados Unidos Mexicanos'

# El caso que costó el anuncio va primero, y con su nombre.
ACIERTOS = [
    ('el caso del v44', 'artículos 940 y 942 del ' + CPC, ['940', '942'], CPC),
    ('fracción con comas', 'artículo 940, fracción II, del ' + CPC, ['940'], CPC),
    ('fracción en minúscula', 'artículo 940, fracción ii, del ' + CPC, ['940'], CPC),
    ('fracción sin comas', 'artículo 940 fracción II del ' + CPC, ['940'], CPC),
    ('abreviado y en plural', 'arts. 1194 y 1195 del Código de Comercio', ['1194', '1195'], 'Código de Comercio'),
    ('ordinal', 'Artículo 2o. de la Ley de Amparo', ['2'], 'Ley de Amparo'),
    ('numeral', 'el numeral 940 del ' + CPC, ['940'], CPC),
    ('sigla', 'de conformidad con el artículo 940 del CPCDF', ['940'], CPC),
    ('sigla pegada', 'artículo 17 CPEUM', ['17'], CPEUM),
    ('apartado', 'artículo 123, apartado A, de la ' + CPEUM, ['123'], CPEUM),
    ('fracción e inciso', 'el artículo 107, fracción III, inciso a), de la ' + CPEUM, ['107'], CPEUM),
    ('sufijo BIS', 'artículo 941 BIS del ' + CPC, ['941'], CPC),
    ('rango', 'artículos 14 al 16 de la ' + CPEUM, ['14', '15', '16'], CPEUM),
    ('anáfora con nombre', 'artículo 940 del citado Código de Procedimientos Civiles', ['940'], 'Código de Procedimientos Civiles'),
    ('todo en mayúsculas', 'ARTÍCULOS 255 Y 260 DEL CÓDIGO DE PROCEDIMIENTOS CIVILES PARA EL DISTRITO FEDERAL',
     ['255', '260'], 'CÓDIGO DE PROCEDIMIENTOS CIVILES PARA EL DISTRITO FEDERAL'),
    ('civil local', 'artículo 2489 del Código Civil para el Distrito Federal', ['2489'], 'Código Civil para el Distrito Federal'),
    ('cortada por el verbo', 'artículos 255, 260 y 271 del ' + CPC + ', que establece la forma', ['255', '260', '271'], CPC),
    ('federal', 'artículo 123 de la Ley Federal del Trabajo', ['123'], 'Ley Federal del Trabajo'),
    ('el apartado DERECHO real',
     'Son aplicables los artículos 303, 308, 309 y 311 del Código Civil para el Distrito Federal, así como los demás relativos.',
     ['303', '308', '309', '311'], 'Código Civil para el Distrito Federal'),
    ('amparo con comas', 'artículo 5, fracción II, de la Ley de Amparo', ['5'], 'Ley de Amparo'),
]

# Números que NO son artículos. Un falso positivo aquí gasta una búsqueda y, lo
# que es peor, mete en el contexto una ley que nadie citó.
RECHAZOS = [
    ('fecha', 'el 2 de febrero de 2024 se presentó la demanda'),
    ('artículo de prensa', 'según el artículo periodístico publicado'),
    ('cláusula', 'la cláusula 940 del contrato de arrendamiento'),
    ('foja', 'a fojas 940 del expediente'),
    ('importe', 'importe de 942 pesos'),
]

CADENA = ('Resulta aplicable el artículo 940 del ' + CPC + '. El artículo 942 del mismo '
          'ordenamiento establece el procedimiento. Y el artículo 941 del citado código lo confirma.')


def main() -> int:
    fallos = 0

    print('LA CITA LLEGA CON SU LEY')
    for nombre, texto, nums_q, ley_q in ACIERTOS:
        r = citas(texto)
        ok = bool(r) and r[0][0] == nums_q and r[0][1] == ley_q
        if not ok:
            fallos += 1
            print(f'  ✗ {nombre}')
            print(f'      esperaba  {nums_q} · «{ley_q}»')
            print(f'      obtuvo    {r[0][0] if r else None} · «{r[0][1] if r else None}»')
    print(f'  {len(ACIERTOS) - fallos} de {len(ACIERTOS)}')

    print('LO QUE NO ES UN ARTÍCULO NO SE TOMA POR UNO')
    rf = 0
    for nombre, texto in RECHAZOS:
        r = citas(texto)
        if r:
            rf += 1
            print(f'  ✗ {nombre}: «{texto}» → {r}')
    print(f'  {len(RECHAZOS) - rf} de {len(RECHAZOS)}')

    print('LA ANÁFORA HEREDA LA LEY')
    cadena = citas(CADENA)
    mal = [c for c in cadena if c[1] != CPC]
    if len(cadena) != 3 or mal:
        fallos += 1
        print(f'  ✗ las tres citas deberían caer en el CPC del Distrito Federal: {cadena}')
    else:
        print('  3 de 3')

    total = fallos + rf
    print()
    print('TODO EN ORDEN' if not total else f'{total} FALLO(S) — la revisión volvería a citar la ley equivocada')
    return 1 if total else 0


if __name__ == '__main__':
    sys.exit(main())
