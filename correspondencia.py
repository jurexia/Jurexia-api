#!/usr/bin/env python3
"""SELLO DE CORRESPONDENCIA — ¿la cita dice lo que el documento dice?

QUÉ MIDE, Y POR QUÉ NO BASTABA LO QUE HABÍA
-------------------------------------------
El validador que ya existía comprueba UNA cosa: que el identificador citado
esté entre los documentos recuperados. Nunca compara la afirmación con el
documento. Por eso un Platinum de Sonora recibió el artículo 371 de la Ley
Federal del Trabajo —que habla de sindicatos— presentado como procedimiento
civil de Sonora, y el sello lo dio por verificado: el documento existía.

MEDIDO antes de escribir una línea de esto, sobre 959 citas reales de 400
respuestas de producción:

    98.2%  corresponden
     1.1%  el identificador apunta al artículo de al lado (438 → 439)
     0.2%  ley distinta (el 107 constitucional atado al 73 de la Ley de Amparo)
     0.4%  identificador que no se resuelve en ninguna colección

Una de cada sesenta citas apunta a otro documento. En una respuesta con ocho
citas, eso es una probabilidad de uno entre ocho de llevar una mala.

EL PRINCIPIO QUE GOBIERNA ESTE ARCHIVO
--------------------------------------
Un validador que marca citas buenas es PEOR que no tener validador. Si esto
borra una cita correcta, el abogado pierde un fundamento que sí tenía; si deja
pasar una mala, estamos como antes. Los dos errores NO cuestan lo mismo.

Por eso la regla es asimétrica: **sólo se marca cuando hay prueba positiva de
que son instrumentos distintos.** Ante la duda, se calla. No basta con que los
nombres no se parezcan: «CNPP» y «Código Nacional de Procedimientos Penales»
no se parecen en absoluto y son la misma ley.

Ya me costó una medición: en la primera pasada, 7 de 9 «discrepancias» eran
siglas contra nombre completo. Eso es exactamente el autoengaño que hay que
evitar.
"""
import re
import unicodedata

# ── Siglas que los abogados usan a diario ───────────────────────────────────
# No es capricho ni adorno: sin esta tabla, cada cita escrita en sigla —que son
# la mayoría— se marcaría como discrepante.
SIGLAS = {
    'cpeum': 'constitucion politica de los estados unidos mexicanos',
    'cn': 'constitucion politica de los estados unidos mexicanos',
    'lft': 'ley federal del trabajo',
    'lfpdppp': 'ley federal de proteccion de datos personales en posesion de los particulares',
    'lgsm': 'ley general de sociedades mercantiles',
    'lgtoc': 'ley general de titulos y operaciones de credito',
    'cnpp': 'codigo nacional de procedimientos penales',
    'cnpcyf': 'codigo nacional de procedimientos civiles y familiares',
    'cnpcf': 'codigo nacional de procedimientos civiles y familiares',
    'ccf': 'codigo civil federal',
    'ccom': 'codigo de comercio',
    'cff': 'codigo fiscal de la federacion',
    'ccdf': 'codigo civil para el distrito federal',
    'cpcdf': 'codigo de procedimientos civiles para el distrito federal',
    'lfpca': 'ley federal de procedimiento contencioso administrativo',
    'lfpa': 'ley federal de procedimiento administrativo',
    'lss': 'ley del seguro social',
    'lisr': 'ley del impuesto sobre la renta',
    'liva': 'ley del impuesto al valor agregado',
    'lgeepa': 'ley general del equilibrio ecologico y la proteccion al ambiente',
    'cadh': 'convencion americana sobre derechos humanos',
}

# ── Las 32 entidades, para detectar el cruce de jurisdicción ────────────────
ENTIDADES = [
    'aguascalientes', 'baja california sur', 'baja california', 'campeche',
    'chiapas', 'chihuahua', 'ciudad de mexico', 'distrito federal', 'coahuila',
    'colima', 'durango', 'estado de mexico', 'guanajuato', 'guerrero',
    'hidalgo', 'jalisco', 'michoacan', 'morelos', 'nayarit', 'nuevo leon',
    'oaxaca', 'puebla', 'queretaro', 'quintana roo', 'san luis potosi',
    'sinaloa', 'sonora', 'tabasco', 'tamaulipas', 'tlaxcala', 'veracruz',
    'yucatan', 'zacatecas',
]
# CDMX y Distrito Federal son la MISMA entidad con dos nombres: tratarlas como
# distintas marcaría como error media legislación capitalina.
MISMA_ENTIDAD = {'distrito federal': 'ciudad de mexico'}


def pelar(s: str) -> str:
    s = unicodedata.normalize('NFD', (s or '').lower())
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', s)).strip()


def expandir(nombre: str) -> str:
    """Sustituye CUALQUIER sigla que aparezca dentro del nombre.

    La primera versión sólo expandía la sigla cuando ocupaba toda la cadena, y
    eso fallaba en el caso más común: «Artículo 137, fracción VI, CPCDF» deja
    como nombre «fracción VI, CPCDF», que no es ninguna sigla suelta. El
    resultado era marcar como discrepante media legislación citada en sigla.
    Se busca palabra por palabra.
    """
    n = pelar(nombre)
    partes = [SIGLAS.get(w, w) for w in n.split()]
    return ' '.join(partes)


def sigla_de(largo: str) -> str:
    """Las iniciales de un nombre largo, para reconocer siglas que no están en
    la tabla. «Ley del Instituto de Seguridad y Servicios Sociales de los
    Trabajadores del Estado de Nuevo León» da ISSSTENL, que es lo bastante
    parecido a ISSSTELEON como para no marcarlo."""
    return ''.join(w[0] for w in pelar(largo).split() if len(w) > 2)


def entidad_de(nombre: str):
    n = pelar(nombre)
    for e in ENTIDADES:
        if e in n:
            return MISMA_ENTIDAD.get(e, e)
    return None


VACIAS = {'de', 'del', 'la', 'el', 'los', 'las', 'para', 'en', 'y', 'estado',
          'libre', 'soberano', 'republica', 'mexicanos', 'unidos', 'nacional',
          'general', 'federal', 'articulo', 'art', 'fraccion', 'parrafo',
          'inciso', 'bis', 'ter', 'vigente', 'aplicable'}
# «tipo» = qué clase de instrumento es. Un Código no es una Ley.
TIPOS = {'codigo', 'ley', 'constitucion', 'reglamento', 'convencion',
         'tratado', 'decreto', 'acuerdo', 'pacto'}


def nucleo(nombre: str) -> set:
    return {w for w in expandir(nombre).split()
            if w not in VACIAS and w not in TIPOS and len(w) > 3}


def tipos_de(nombre: str) -> set:
    return {w for w in expandir(nombre).split() if w in TIPOS}


def _parecidas(a: str, b: str) -> bool:
    """Dos siglas se parecen si comparten al menos cuatro iniciales seguidas."""
    return len(a) >= 4 and len(b) >= 4 and (a[:4] in b or b[:4] in a)


def veredicto(ley_citada: str, art_citado: str, doc_origen: str, doc_ref: str,
              doc_texto: str = ''):
    """Devuelve (estado, motivo).

    estado ∈ {'ok', 'ley_distinta', 'entidad_distinta', 'articulo_distinto'}
    Sólo se aparta de 'ok' con prueba positiva.
    """
    if not doc_origen:
        return 'ok', 'el documento no declara su origen: no hay con qué comparar'

    # 1. ENTIDAD — la prueba más limpia y la que más duele cuando falla.
    e_cita, e_doc = entidad_de(ley_citada), entidad_de(doc_origen)
    if e_cita and e_doc and e_cita != e_doc:
        return 'entidad_distinta', f'se citó {e_cita} y el documento es de {e_doc}'

    # 2. TIPO DE INSTRUMENTO — un Código no puede citarse como Ley.
    t_cita, t_doc = tipos_de(ley_citada), tipos_de(doc_origen)
    if t_cita and t_doc and not (t_cita & t_doc):
        # salvo que compartan el núcleo, que delata el mismo instrumento
        if not (nucleo(ley_citada) & nucleo(doc_origen)):
            return 'ley_distinta', f'{sorted(t_cita)} contra {sorted(t_doc)}, y sin núcleo común'

    # 3. NÚCLEO — sólo se marca si AMBOS tienen palabras propias y NINGUNA coincide.
    n_cita, n_doc = nucleo(ley_citada), nucleo(doc_origen)
    if n_cita and n_doc and not (n_cita & n_doc):
        # Antes de marcar: ¿alguna palabra de la cita es una sigla del nombre
        # del documento? ISSSTELEON contra «Instituto de Seguridad y Servicios
        # Sociales… Nuevo León» no comparte una sola palabra y es la misma ley.
        ini = sigla_de(doc_origen)
        for w in n_cita:
            if len(w) >= 4 and (w in ini or ini.startswith(w[:5])
                                or _parecidas(w, ini)):
                return 'ok', f'«{w}» parece la sigla de «{doc_origen[:40]}»'
        return 'ley_distinta', f'{sorted(n_cita)[:3]} contra {sorted(n_doc)[:3]}'

    # 4. NÚMERO DE ARTÍCULO — el documento manda.
    # El `ref` nombra UN artículo, pero un fragmento puede contener varios: el
    # de Morelos etiquetado «Art. 151» lleva dentro del 151 al 155. Comparar
    # sólo contra la etiqueta marcaba como error citas correctas, que es
    # justo lo que este archivo existe para no hacer.
    n_cita_num = re.sub(r'\D', '', art_citado or '')
    if n_cita_num:
        if doc_texto and re.search(rf'[Aa]rt[íi]culo\s+{n_cita_num}\b', doc_texto):
            return 'ok', ''
        nums = set(re.findall(r'\d+', doc_ref or ''))
        if nums and n_cita_num not in nums:
            if doc_texto:
                return 'articulo_distinto', f'se citó el {n_cita_num}; el documento es {doc_ref} y su texto no lo contiene'
            return 'articulo_dudoso', f'se citó el {n_cita_num} y la etiqueta dice {doc_ref} (sin texto para comprobar)'

    return 'ok', ''
