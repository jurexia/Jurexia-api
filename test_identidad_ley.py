#!/usr/bin/env python3
"""DOS LEYES QUE SE LLAMAN PARECIDO NO SON LA MISMA LEY.

Por qué existe esta prueba (16-sep-2026)
----------------------------------------
Hay 33 Códigos Civiles en México y la única palabra que los distingue suele ser
una: «Federal», «para el Estado de Sonora», «para el Distrito Federal». Si esa
palabra se lee mal, el verificador encuentra el artículo en el código de otro
fuero y da por buena una cita que no lo es — el fallo que los abogados más
corrigen, según las incidencias.

Y se leía mal: `_MARCA_FEDERAL` casaba la palabra «Federal» DENTRO de «Distrito
Federal», así que los códigos de la Ciudad de México se clasificaban como leyes
federales y `_misma_ley` los confundía con sus homónimos del fuero federal.

    IUREXIA-MAC/jurexia-api-git/.venv/bin/python3 test_identidad_ley.py
"""
import os
import re
import sys
import unicodedata

RAIZ = os.path.dirname(os.path.abspath(__file__))


def _cargar():
    """Ejecuta el tramo de main.py que decide identidad y ámbito de una ley."""
    src = open(os.path.join(RAIZ, 'main.py'), encoding='utf-8').read()
    ini = '_RE_RUIDO_LEY'
    fin = '@app.post("/acervo/articulos")'
    if ini not in src or fin not in src:
        sys.exit('ABORTA: no encuentro el tramo de identidad de leyes en main.py')
    espacio = {'re': re, 'unicodedata': unicodedata}
    exec(compile(src[src.index(ini):src.index(fin)], 'main.py', 'exec'), espacio)
    return espacio


E = _cargar()
ambito, misma = E['_ambito'], E['_misma_ley']

CPC_DF = 'Código de Procedimientos Civiles para el Distrito Federal'
CPC_DF_MAY = 'CÓDIGO DE PROCEDIMIENTOS CIVILES PARA EL DISTRITO FEDERAL'
CC_DF = 'Código Civil para el Distrito Federal'

AMBITOS = [
    (CPC_DF, 'local'),
    (CC_DF, 'local'),
    ('Código Penal para el Distrito Federal', 'local'),
    ('Código de Procedimientos Civiles de la Ciudad de México', 'local'),
    ('Código Civil para el Estado de Sonora', 'local'),
    ('Código Civil local', 'local'),
    ('Código Civil Federal', 'federal'),
    ('Código Federal de Procedimientos Civiles', 'federal'),
    ('Ley Federal del Trabajo', 'federal'),
    ('Código Nacional de Procedimientos Penales', 'federal'),
    ('Constitución Política de los Estados Unidos Mexicanos', 'federal'),
    ('Código de Comercio', 'indefinido'),
]

# (citada, real, ¿son la misma?)  — los cinco primeros son los que fallaban.
PAREJAS = [
    (CPC_DF, 'Código Federal de Procedimientos Civiles', False),
    (CC_DF, 'Código Civil Federal', False),
    ('Código Penal para el Distrito Federal', 'Código Penal Federal', False),
    ('Código de Procedimientos Civiles de la Ciudad de México', CPC_DF, True),
    (CPC_DF, CPC_DF_MAY, True),
    ('Código Civil para el Estado de Sonora', 'Código Civil para el Estado de Jalisco', False),
    ('Código Fiscal de la Federación', 'Código Fiscal del Estado de Querétaro', False),
    ('Código Civil local', 'Código Civil Federal', False),
    ('Ley Federal del Trabajo', 'Ley Federal del Trabajo', True),
    ('Ley de Amparo', 'Ley de Amparo', True),
    ('Constitución Política de los Estados Unidos Mexicanos',
     'Constitución Política de los Estados Unidos Mexicanos', True),
    ('Código de Comercio', 'Código de Comercio', True),
]


def main() -> int:
    fallos = 0

    print('EL FUERO DE CADA LEY')
    for nombre, esperado in AMBITOS:
        real = ambito(nombre)
        if real != esperado:
            fallos += 1
            print(f'  ✗ {nombre[:58]:60s} esperaba {esperado}, dice {real}')
    print(f'  {len(AMBITOS) - fallos} de {len(AMBITOS)}')

    print('¿SON LA MISMA LEY?')
    f2 = 0
    for citada, real, esperado in PAREJAS:
        obtenido = misma(citada, real)
        if obtenido != esperado:
            f2 += 1
            print(f'  ✗ «{citada[:42]}» ‖ «{real[:42]}»')
            print(f'      esperaba {esperado}, dice {obtenido}')
    print(f'  {len(PAREJAS) - f2} de {len(PAREJAS)}')

    total = fallos + f2
    print()
    print('TODO EN ORDEN' if not total
          else f'{total} FALLO(S) — se podría dar por buena la cita de otro fuero')
    return 1 if total else 0


if __name__ == '__main__':
    sys.exit(main())
