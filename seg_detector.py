#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
El detector de novedad.

DE QUÉ SE ENCARGA. De decidir, para cada acuerdo que devuelve el portal, si es
nuevo, si ya lo habíamos visto, o si es uno viejo que el juzgado reeditó. De
esa decisión depende la regla 2 —«sólo se escribe cuando hay actuación
nueva»—, así que equivocarse tiene dos costes distintos y muy asimétricos:

  · Falso negativo: no avisar de algo que pasó. Es el peor, porque el silencio
    de Iurexia significa «no hubo movimiento» y el abogado se fía.
  · Falso positivo: avisar de algo que ya sabía. Molesta, y a la tercera vez
    deja de leer los correos, que acaba siendo el mismo falso negativo por
    otra vía.

POR QUÉ NO BASTA COMPARAR TEXTO. El portal reimprime espacios, renumera la
columna «No.» cuando el juzgado intercala un acuerdo atrasado, mueve la fecha
de publicación, y reedita el resumen para corregir un nombre mal escrito.
Comparar el texto crudo daría decenas de falsos positivos al mes.

CÓMO SE RESUELVE. Con dos huellas y un desempate, que viven en seg_lector_pjf:

  · huella_clave  — la identidad: órgano, número, NEUN, cuaderno, fecha del
                    auto y los primeros 300 caracteres del resumen.
  · huella_texto  — el contenido completo normalizado.
  · parecido      — Jaccard sobre trigramas del resumen, para cuando la
                    reedición toca justo esos 300 caracteres y la identidad
                    cambia sin que el acuerdo sea otro. El simhash se guarda en
                    la tabla pero no decide: con resúmenes cortos es inestable,
                    y aquí el candidato ya viene acotado al mismo cuaderno y
                    día, así que comparar el texto sale barato y es exacto.

Este módulo no toca la red ni la base de datos: recibe una lectura y lo que ya
se conocía, y devuelve un dictamen. Así se puede probar con casos duros
inventados, que es exactamente lo que hace `--probar`.

    ./seg_detector.py --probar
"""
import sys

try:
    from seg_lector_pjf import normalizar
except ImportError:                                    # ejecutado desde otra ruta
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from seg_lector_pjf import normalizar

# Parecido por encima del cual dos acuerdos del mismo cuaderno y la misma fecha
# se consideran el mismo, reeditado.
#
# POR QUÉ JACCARD Y NO EL SIMHASH. El simhash está pensado para descartar
# rápido entre millones de documentos largos. Aquí el candidato ya viene
# acotado a los acuerdos del mismo cuaderno y el mismo día —dos o tres, rara
# vez más—, así que comparar el texto de verdad sale barato y es exacto. Y con
# textos cortos el simhash es directamente malo: un resumen de veinte
# trigramas cambia media firma porque se corrija una letra, que es justo el
# caso que hay que cazar. Se midió: corregir «Villalejo» por «Villalejos» daba
# distancia de Hamming muy por encima del umbral, y la reedición se colaba
# como acuerdo nuevo.
#
# El margen es cómodo: dos autos distintos del mismo día rondan 0.02 de
# parecido y una reedición pasa de 0.75. El umbral va en medio, más cerca de
# la reedición para no esconder acuerdos nuevos de verdad.
UMBRAL_PARECIDO = 0.62


def _trigramas(t):
    p = normalizar(t).split()
    if len(p) < 3:
        return {' '.join(p)} if p else set()
    return {' '.join(p[i:i + 3]) for i in range(len(p) - 2)}


def parecido(a, b):
    """Jaccard sobre trigramas de palabra. 1.0 es idéntico, 0.0 nada en común."""
    ta, tb = _trigramas(a), _trigramas(b)
    if not ta or not tb:
        return 1.0 if ta == tb else 0.0
    return len(ta & tb) / len(ta | tb)


def comparar(acuerdos, conocidas, es_alta=False):
    """Dictamina qué hacer con cada acuerdo leído.

    `acuerdos`  — los que devolvió el portal, ya con sus tres huellas.
    `conocidas` — las actuaciones que ya teníamos de ESTE seguimiento, cada una
                  con huella_clave, huella_texto, simhash, cuaderno, fecha_auto
                  y version.
    `es_alta`   — en el alta todo entra como línea base y NADA genera correo.
                  Sin esto, el primer día el abogado recibe tres años de
                  acuerdos de golpe y se da de baja.

    Devuelve {'nuevas': [...], 'reediciones': [...], 'ya_vistas': int}.
    Cada elemento de `reediciones` lleva `reemplaza_a` y la `version` que toca.
    """
    por_clave = {}
    for c in conocidas:
        k = c['huella_clave']
        # Puede haber varias versiones de la misma identidad: interesa la última.
        if k not in por_clave or (c.get('version') or 1) > (por_clave[k].get('version') or 1):
            por_clave[k] = c

    # Índice para el desempate: mismo cuaderno y misma fecha del auto.
    por_dia = {}
    for c in conocidas:
        por_dia.setdefault((normalizar(c.get('cuaderno') or ''), c.get('fecha_auto')), []).append(c)

    nuevas, reediciones, ya_vistas = [], [], 0

    for a in acuerdos:
        previa = por_clave.get(a['huella_clave'])

        if previa is not None:
            if previa['huella_texto'] == a['huella_texto']:
                ya_vistas += 1
                continue
            # Misma identidad, distinto contenido: el juzgado lo corrigió.
            reediciones.append({**a,
                                'reemplaza_a': previa.get('id'),
                                'version': (previa.get('version') or 1) + 1,
                                'motivo': 'texto_cambiado'})
            continue

        # La identidad no existe. Antes de declararla nueva, el desempate: una
        # reedición que toque los primeros 300 caracteres cambia la huella_clave
        # y, sin esto, se colaría como acuerdo nuevo.
        candidatas = por_dia.get(
            (normalizar(a.get('cuaderno') or ''), a.get('fecha_auto')), [])
        gemela, mejor = None, 0.0
        for c in candidatas:
            p = parecido(c.get('resumen') or '', a.get('resumen') or '')
            if p > mejor:
                gemela, mejor = c, p

        if gemela is not None and mejor >= UMBRAL_PARECIDO:
            reediciones.append({**a,
                                'reemplaza_a': gemela.get('id'),
                                'version': (gemela.get('version') or 1) + 1,
                                'motivo': f'cabecera_reeditada(parecido={mejor:.2f})',
                                # Se conserva la identidad ANTIGUA: si no, la
                                # siguiente lectura volvería a verla como nueva.
                                'huella_clave': gemela['huella_clave'],
                                'huella_clave_nueva': a['huella_clave']})
            continue

        nuevas.append({**a, 'version': 1})

    if es_alta:
        # Todo el histórico entra marcado y callado.
        for x in nuevas + reediciones:
            x['es_linea_base'] = True
        return {'nuevas': [], 'reediciones': [],
                'linea_base': nuevas + reediciones, 'ya_vistas': ya_vistas}

    return {'nuevas': nuevas, 'reediciones': reediciones,
            'linea_base': [], 'ya_vistas': ya_vistas}


def hay_que_avisar(dictamen, tope=10):
    """¿Se manda correo, y de qué?

    El tope es una válvula: si un solo expediente genera más de diez
    actuaciones nuevas en un día, casi siempre es que el detector se ha
    equivocado y no que el juzgado tuvo una mañana movida. Antes que mandar un
    correo con cuarenta acuerdos, se calla y se escala.
    """
    n = len(dictamen['nuevas']) + len(dictamen['reediciones'])
    if n == 0:
        return {'avisar': False, 'motivo': 'sin_novedad'}
    if len(dictamen['nuevas']) > tope:
        return {'avisar': False, 'motivo': 'demasiadas', 'n': n,
                'escalar': True}
    return {'avisar': True, 'motivo': 'novedad', 'n': n}


# ── Pruebas ───────────────────────────────────────────────────────────

def _probar():
    """Los casos que de verdad rompen un detector ingenuo."""
    from seg_lector_pjf import huella_clave, huella_texto, simhash64

    def acuerdo(resumen, fecha='2026-01-19', cuaderno='Principal', orden=1):
        return {
            'orden_en_lista': orden, 'fecha_auto': fecha, 'cuaderno': cuaderno,
            'resumen': resumen,
            'huella_clave': huella_clave('PJF', '293', '71/2026', '40911643',
                                         cuaderno, fecha, resumen),
            'huella_texto': huella_texto(resumen),
            'simhash': simhash64(resumen),
        }

    def conocida(a, id_='a1', version=1):
        return {**a, 'id': id_, 'version': version}

    fallos = []

    def cierto(titulo, condicion, detalle=''):
        print(f'  {"✓" if condicion else "✗"} {titulo}'
              + (f'   {detalle}' if detalle and not condicion else ''))
        if not condicion:
            fallos.append(titulo)

    base = 'Demanda. Vista la demanda de amparo promovida por María Luisa Rodríguez Villalejo, contra actos del Titular del Juzgado Cuarto de lo Civil.'

    print('\nCASOS QUE ROMPEN UN DETECTOR INGENUO\n')

    a = acuerdo(base)
    d = comparar([a], [conocida(a)])
    cierto('el mismo acuerdo dos veces no es novedad',
           d['ya_vistas'] == 1 and not d['nuevas'])

    # El juzgado intercala un acuerdo atrasado y renumera la columna «No.».
    movido = {**a, 'orden_en_lista': 7}
    d = comparar([movido], [conocida(a)])
    cierto('renumerar la lista no lo convierte en nuevo',
           d['ya_vistas'] == 1 and not d['nuevas'])

    # Cambia la fecha de publicación, no la del auto.
    otra_pub = {**a, 'fecha_publicacion': '2026-02-01'}
    d = comparar([otra_pub], [conocida(a)])
    cierto('mover la fecha de publicación no lo convierte en nuevo',
           d['ya_vistas'] == 1 and not d['nuevas'])

    # Reedición que toca el final: misma identidad, distinto texto.
    largo = base + ' Se admite a trámite y se pide informe justificado.'
    r = acuerdo(largo)
    d = comparar([r], [conocida(acuerdo(base))])
    cierto('reeditar el final es reedición, no acuerdo nuevo',
           len(d['reediciones']) == 1 and not d['nuevas'],
           f"nuevas={len(d['nuevas'])} reed={len(d['reediciones'])}")

    # El caso difícil: corrigen un nombre DENTRO de los primeros 300
    # caracteres, así que la huella de identidad cambia. Sin el desempate por
    # simhash, esto se colaría como acuerdo nuevo y mandaría un correo falso.
    corregido = base.replace('María Luisa Rodríguez Villalejo',
                             'María Luisa Rodríguez Villalejos')
    c = acuerdo(corregido)
    prev = conocida(acuerdo(base))
    cierto('corregir un nombre cambia la huella de identidad',
           c['huella_clave'] != prev['huella_clave'])
    d = comparar([c], [prev])
    cierto('…y aun así el desempate lo caza como reedición',
           len(d['reediciones']) == 1 and not d['nuevas'],
           f"nuevas={len(d['nuevas'])} reed={len(d['reediciones'])}")
    if d['reediciones']:
        cierto('…conservando la identidad antigua, para no repetirlo mañana',
               d['reediciones'][0]['huella_clave'] == prev['huella_clave'])

    # Dos autos distintos el mismo día en el mismo cuaderno SÍ son dos.
    otro = acuerdo('Se tiene por recibido el informe justificado de la autoridad '
                   'responsable y se señala fecha para la audiencia constitucional.')
    d = comparar([acuerdo(base), otro], [])
    cierto('dos autos distintos del mismo día son dos actuaciones',
           len(d['nuevas']) == 2, f"nuevas={len(d['nuevas'])}")

    # El principal y el incidente de suspensión, mismo día, no se pisan.
    inc = acuerdo(base, cuaderno='Incidente de suspensión')
    d = comparar([acuerdo(base), inc], [])
    cierto('el principal y el incidente no se pisan',
           len(d['nuevas']) == 2, f"nuevas={len(d['nuevas'])}")

    # El alta: todo entra callado.
    d = comparar([acuerdo(base), otro], [], es_alta=True)
    cierto('en el alta nada genera correo',
           not d['nuevas'] and len(d['linea_base']) == 2)

    # La válvula de las diez.
    muchos = [acuerdo(f'Acuerdo número {i} con texto bien distinto del resto '
                      f'para que no se parezcan entre sí en absoluto.', orden=i)
              for i in range(1, 13)]
    d = comparar(muchos, [])
    v = hay_que_avisar(d)
    cierto('doce actuacionesnuevas en un día no se envían: se escalan',
           v['avisar'] is False and v.get('escalar') is True,
           str(v))

    # Y el caso normal sí avisa.
    d = comparar([acuerdo(base)], [])
    cierto('una actuación nueva sí se avisa', hay_que_avisar(d)['avisar'] is True)

    print(f'\n{"TODO CORRECTO" if not fallos else f"FALLAN {len(fallos)}: {fallos}"}')
    return 1 if fallos else 0


if __name__ == '__main__':
    sys.exit(_probar() if '--probar' in sys.argv else print(__doc__.strip()) or 0)
