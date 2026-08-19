#!/usr/bin/env python3
"""Poda los corpus de los genios de cola: fiscal, mercantil y CIDH.

POR QUÉ
-------
Crear una caché se cobra como tokens de entrada: el corpus entero, cada vez.
Almacenarla tres minutos cuesta seis veces menos que crearla. Así que la
palanca de coste NO es el TTL —bajarlo multiplica las creaciones— sino el
TAMAÑO del corpus, que se paga en cada una de ellas.

Fiscal, mercantil y CIDH suman el 42% de todo el corpus para el 11% de las
consultas. Ahí es donde hay dinero.

QUÉ SE PUEDE CORTAR, Y POR QUÉ SE PUEDE
---------------------------------------
La regla que hace esto seguro se comprobó contra Qdrant antes de escribir una
línea: TODO lo que se quita de aquí sigue estando en `leyes_federales`, con su
texto íntegro y su artículo. Sacar un título del corpus del genio no lo borra
del sistema — deja de venir precargado en memoria, y pasa a llegar por
búsqueda cuando la consulta lo pide. Para un título que casi nadie consulta,
ese es exactamente el trato correcto.

Comprobado en producción (fragmentos en `leyes_federales`):
    CFF 351 · LISR 331 · IVA 69 · LFPCA 89 · C.Comercio 762
    LGTOC 449 · LGSM 275 · L.Contrato Seguro 204 · LISF 550

LO QUE NO SE TOCA, Y POR QUÉ
----------------------------
La Convención Americana, el PIDCP, el PIDESC y la Convención contra la Tortura
NO ESTÁN en el RAG. Existen únicamente en el corpus de los genios. Quitarlas
las haría desaparecer del sistema entero. Por eso el corpus de CIDH sólo se
poda en lo que sí está duplicado, y los tratados se quedan enteros.

LA NOTA AL PIE, QUE NO ES DECORACIÓN
------------------------------------
Cada archivo podado termina diciendo qué se le quitó. Sin esa nota el genio lee
una LISR que acaba en el Título IV y concluye que el Título V no existe: se
inventaría que no hay régimen para residentes en el extranjero. Con la nota
sabe que está ahí y que tiene que buscarlo. Truncar en silencio es como se
fabrica una alucinación.

    python3 podar_corpus.py            # enseña el plan, no toca nada
    python3 podar_corpus.py --aplicar  # poda, con respaldo en _source/
"""
import os
import re
import shutil
import sys

RAIZ = os.path.dirname(os.path.abspath(__file__))
os.chdir(RAIZ)

# Encabezado de título, con o sin almohadillas de Markdown.
H = re.compile(r'^#{0,4}\s*(LIBRO|T[IÍ]TULO)\s+(.+?)\s*$', re.I)

# ── El plan, título por título, con el motivo de cada corte ──────────────────
# Un corte sólo entra aquí si (a) está en el RAG y (b) hay una razón de
# práctica jurídica, no de tamaño. El orden es el del archivo.
PODA = {
 'cache_corpus_fiscal/02_ley_impuesto_renta.txt': [
   ('TÍTULO III', 'Personas morales con fines no lucrativos — donatarias, '
                  'asociaciones y sindicatos. Es materia de cumplimiento '
                  'corporativo, no de litigio fiscal.'),
   ('TÍTULO V',   'Residentes en el extranjero — establecimiento permanente y '
                  'retenciones. Es fiscal internacional, una especialidad '
                  'aparte del contencioso que atiende este genio.'),
 ],
 # El CFF NO se toca. Se intentó podar su Título Sexto dándolo por el juicio
 # contencioso derogado; el simulacro enseñó que es «De la Revelación de
 # Esquemas Reportables», derecho vigente desde 2020 y obligación viva de los
 # asesores fiscales. Se retiró el corte: el supuesto era falso.
 'cache_corpus_mercantil/01_codigo_comercio.txt': [
   ('TITULO CUARTO', 'Del arbitraje comercial — jurisdicción privada, materia '
                     'de despachos especializados; no es el juicio mercantil '
                     'que se litiga ante los juzgados.'),
   ('TITULO DECIMO', 'De los transportes por vías terrestres o fluviales — en '
                     'la práctica lo desplazaron las leyes de caminos y de '
                     'navegación.'),
 ],
}

# Archivos que salen enteros. El motivo pesa más que los kilobytes.
FUERA = {
 'cache_corpus_mercantil/05_ley_instituciones_seguros_fianzas.txt':
   'Ley de supervisión de aseguradoras por la Comisión Nacional de Seguros y '
   'Fianzas: regula a la institución, no al contrato. Quien litiga un siniestro '
   'cita la Ley sobre el Contrato de Seguro, que se queda completa en este '
   'corpus. 550 fragmentos disponibles en el RAG.',
}

# Archivos que se SUSTITUYEN por otro mejor. No es poda: es reparación, y el
# corpus queda más pequeño de propina.
REEMPLAZO = {
 'cache_corpus_cidh/01_cpeum_ddhh_resumida.txt': (
   'cache_corpus/CPEUM.txt',
   'Se llamaba «CPEUM resumida» pero no era la Constitución: de sus 272 KB, el '
   '82% eran transitorios y aparato de decretos de reforma, y sólo el 0.5% '
   'texto de artículos —siete artículos distintos en todo el archivo—. El '
   'genio de derechos humanos llevaba en memoria disposiciones transitorias en '
   'lugar de la Constitución. Se sustituye por el fragmento limpio que ya usa '
   'el genio constitucional: artículos 1–30 y 103–107, los de derechos '
   'humanos, debido proceso y amparo. Menos peso y, sobre todo, el texto que '
   'de verdad hacía falta.'),
}

NOTA = (
 '\n\n---\n'
 '## AVISO SOBRE ESTE TEXTO\n\n'
 'De este ordenamiento se omitieron aquí las partes que se enumeran abajo, por '
 'ser de consulta poco frecuente en el litigio. **NO están derogadas y NO son '
 'inaplicables**: su texto íntegro está disponible en la base documental de '
 'Iurexia y se recupera por búsqueda.\n\n'
 'Si la consulta del usuario toca alguna de ellas, NO afirmes que no existe ni '
 'improvises su contenido: indica que debe consultarse el texto vigente y '
 'apóyate en los fragmentos que la búsqueda te entregue.\n\n{detalle}\n')


def bloques(texto):
    """Devuelve [(inicio, fin, encabezado)] de cada título del archivo."""
    ls = texto.split('\n')
    marcas, pos = [], 0
    for l in ls:
        m = H.match(l.strip())
        if m:
            marcas.append((pos, f"{m.group(1).upper()} {m.group(2)}".strip()))
        pos += len(l) + 1
    return [(p, marcas[i+1][0] if i+1 < len(marcas) else len(texto), t)
            for i, (p, t) in enumerate(marcas)]


def norm(s):
    return re.sub(r'[^A-Z ]', '', s.upper().replace('Í', 'I')).strip()


def podar(ruta, cortes, aplicar):
    texto = open(ruta, encoding='utf-8', errors='replace').read()
    orig = len(texto)
    bs = bloques(texto)
    fuera, detalle = [], []
    for etiqueta, motivo in cortes:
        obj = norm(etiqueta)
        # el título buscado es el bloque cuyo encabezado empieza igual
        cand = [b for b in bs if norm(b[2]).startswith(obj)]
        if not cand:
            print(f'   ⚠ {os.path.basename(ruta)}: no encontré «{etiqueta}» — '
                  f'no se corta')
            continue
        # si el encabezado se repite, se poda el bloque más grande
        b = max(cand, key=lambda x: x[1] - x[0])
        fuera.append(b)
        nombre = next((l.strip() for l in texto[b[0]:b[1]].split('\n')[1:5]
                       if l.strip() and not l.startswith('#')), '')
        detalle.append(f'- **{etiqueta}** — {nombre[:70]}\n  *{motivo}*')
        print(f'   − {(b[1]-b[0])/1024:6.0f} KB  {etiqueta:16} {nombre[:44]}')
    if not fuera:
        return 0
    for b in sorted(fuera, key=lambda x: -x[0]):
        texto = texto[:b[0]] + texto[b[1]:]
    texto = texto.rstrip() + NOTA.format(detalle='\n'.join(detalle))
    if aplicar:
        resp = os.path.join(os.path.dirname(ruta), '_source')
        os.makedirs(resp, exist_ok=True)
        dest = os.path.join(resp, os.path.basename(ruta))
        if not os.path.exists(dest):          # el respaldo es el original, no
            shutil.copy2(ruta, dest)          # una copia de la poda anterior
        open(ruta, 'w', encoding='utf-8').write(texto)
    return orig - len(texto)


def main():
    aplicar = '--aplicar' in sys.argv
    print('  PODA DE CORPUS' + ('' if aplicar else '   (simulacro — no toca nada)'))
    ahorro = {}
    for ruta, cortes in PODA.items():
        print(f'\n  {ruta}')
        ahorro[ruta] = podar(ruta, cortes, aplicar)
    for ruta, motivo in FUERA.items():
        n = os.path.getsize(ruta) if os.path.exists(ruta) else 0
        print(f'\n  {ruta}\n   − {n/1024:6.0f} KB  ARCHIVO COMPLETO')
        print(f'     {motivo[:100]}…')
        if aplicar and n:
            resp = os.path.join(os.path.dirname(ruta), '_source')
            os.makedirs(resp, exist_ok=True)
            shutil.move(ruta, os.path.join(resp, os.path.basename(ruta)))
        ahorro[ruta] = n

    for ruta, (fuente, motivo) in REEMPLAZO.items():
        viejo = os.path.getsize(ruta) if os.path.exists(ruta) else 0
        nuevo = os.path.getsize(fuente)
        print(f'\n  {ruta}\n   ⇄ {viejo/1024:6.0f} KB → {nuevo/1024:.0f} KB  '
              f'SUSTITUIDO por {fuente}')
        print(f'     {motivo[:100]}…')
        if aplicar and viejo:
            resp = os.path.join(os.path.dirname(ruta), '_source')
            os.makedirs(resp, exist_ok=True)
            dest = os.path.join(resp, os.path.basename(ruta))
            if not os.path.exists(dest):
                shutil.move(ruta, dest)
            else:
                os.remove(ruta)
            shutil.copy2(fuente, ruta)
        ahorro[ruta] = viejo - nuevo

    print('\n  ── RESULTADO POR GENIO ──')
    tot_a = tot_d = 0
    for d in ('cache_corpus_fiscal', 'cache_corpus_mercantil', 'cache_corpus_cidh'):
        quit = sum(v for k, v in ahorro.items() if k.startswith(d))
        act = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)
                  if f.endswith('.txt'))
        # en simulacro el disco tiene el original; ya aplicado, tiene el podado
        antes = act + quit if aplicar else act
        print(f'   {d.replace("cache_corpus_",""):11} {antes/1024:7.0f} KB → '
              f'{(antes-quit)/1024:7.0f} KB   (−{100*quit/antes:.0f}%)')
        tot_a += antes; tot_d += antes - quit
    print(f'   {"TOTAL":11} {tot_a/1024:7.0f} KB → {tot_d/1024:7.0f} KB   '
          f'(−{100*(tot_a-tot_d)/tot_a:.0f}%)')
    if not aplicar:
        print('\n  Para aplicarlo:  python3 podar_corpus.py --aplicar')


if __name__ == '__main__':
    main()
