#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
El barrido diario: donde se cumplen las tres reglas.

  1. Una revisión al día, a las 9:10 hora de la Ciudad de México.
  2. Correo sólo cuando hay actuación nueva.
  3. Si no se pudo revisar, se avisa.

La tercera es la que da forma a todo lo demás. Obliga a que cada expediente
deje fila en `seg_revisiones` CADA día —se pueda leer o no— y a que el código
prefiera declararse ciego antes que declarar «sin novedad» con dudas. Un falso
«sin novedad» es indistinguible del silencio bueno, y el silencio bueno es la
promesa del producto.

    ./seg_barrido.py --alta 293 71/2026 --alias "Amparo de prueba" --correo x@y.com
    ./seg_barrido.py --barrer                # pase 1
    ./seg_barrido.py --barrer --pase 4       # cierre: correos de "no se pudo"
    ./seg_barrido.py --estado
"""
import argparse
import sys
import time
import urllib.error
from datetime import datetime, timedelta, timezone

import seg_almacen as db
import seg_correo as correo
import seg_lector_cdmx as cdmx
import seg_lector_pjf as pjf
from seg_detector import comparar, hay_que_avisar

MEXICO = timezone(timedelta(hours=-6))     # sin horario de verano desde 2022
PAUSA_ENTRE_PETICIONES = 2.0
ESPERAS_REINTENTO = [30, 120, 480]
BASE_URL = 'https://iurexia.com'


def hoy_mexico():
    return datetime.now(MEXICO).date().isoformat()


def ahora_mexico():
    return datetime.now(MEXICO).strftime('%H:%M')


def guardia_de_huso():
    """México suprimió el horario de verano en 2022, así que 9:10 local es
       15:10 UTC y punto. Pero si el país volviera a cambiar de huso, el cron
       de Vercel dispararía a otra hora local y los correos dirían una hora
       falsa. Antes que eso, el barrido se niega y grita."""
    h = datetime.now(MEXICO).hour
    return 6 <= h <= 14, h


# ── Una revisión ──────────────────────────────────────────────────────

def revisar(seg, corrida_id, intento=1, fecha=None):
    """Revisa un expediente y deja constancia. Devuelve el dictamen o None."""
    fecha = fecha or hoy_mexico()
    organo = seg.get('organo') or {}
    t0 = time.time()
    base = {'corrida_id': corrida_id, 'seguimiento_id': seg['id'],
            'user_id': seg['user_id'], 'fecha_local': fecha, 'intento': intento}

    conocidas = db.actuaciones_de(seg['id'])

    try:
        lectura = pjf.leer(organo.get('clave_externa'), seg['numero'],
                           tipo_asunto=seg.get('tipo_asunto_clave') or 1,
                           tipo_procedimiento=seg.get('tipo_procedimiento_clave') or 0,
                           minimo_esperado=len(conocidas))
    except pjf.ErrorNoEncontrado as e:
        db.registrar_revision({**base, 'resultado': 'fallo_no_encontrado',
                               'detalle': str(e)[:400],
                               'duracion_ms': int((time.time() - t0) * 1000)})
        return None
    except pjf.ErrorFormato as e:
        db.registrar_revision({**base, 'resultado': 'fallo_formato',
                               'detalle': str(e)[:400],
                               'duracion_ms': int((time.time() - t0) * 1000)})
        return None
    except urllib.error.HTTPError as e:
        db.registrar_revision({**base, 'resultado': 'fallo_http',
                               'http_status': e.code, 'detalle': str(e)[:400],
                               'duracion_ms': int((time.time() - t0) * 1000)})
        return None
    except Exception as e:
        db.registrar_revision({**base, 'resultado': 'fallo_red',
                               'detalle': f'{type(e).__name__}: {e}'[:400],
                               'duracion_ms': int((time.time() - t0) * 1000)})
        return None

    dictamen = comparar(lectura['acuerdos'], conocidas)
    nuevas = dictamen['nuevas'] + dictamen['reediciones']

    if nuevas:
        db.guardar_actuaciones([{
            'seguimiento_id': seg['id'], 'user_id': seg['user_id'],
            'huella_clave': a['huella_clave'], 'huella_texto': a['huella_texto'],
            'simhash': a.get('simhash'), 'cuaderno': a.get('cuaderno'),
            'fecha_auto': a.get('fecha_auto'),
            'fecha_publicacion': a.get('fecha_publicacion'),
            'orden_en_lista': a.get('orden_en_lista'),
            'resumen': a.get('resumen') or '', 'url_fuente': lectura['url'],
            'origen': 'pjf_vercaptura', 'version': a.get('version', 1),
            'reemplaza_a': a.get('reemplaza_a'), 'es_linea_base': False,
        } for a in nuevas])

    db.registrar_revision({
        **base,
        'resultado': 'ok_con_novedad' if nuevas else 'ok_sin_novedad',
        'http_status': lectura['http_status'], 'bytes': lectura['bytes'],
        'hash_respuesta': lectura['hash_respuesta'],
        'n_actuaciones_vistas': len(lectura['acuerdos']),
        'terminada_en': datetime.now(timezone.utc).isoformat(),
        'duracion_ms': int((time.time() - t0) * 1000),
    })
    db.actualizar_seguimiento(seg['id'], {
        'ultima_revision_ok': datetime.now(timezone.utc).isoformat(),
        'fallos_consecutivos': 0,
        'neun': (lectura['caratula'].get('neun') or seg.get('neun')),
    })

    dictamen['url'] = lectura['url']
    dictamen['caratula'] = lectura['caratula']
    return dictamen


def avisar_novedad(seg, dictamen, fecha=None):
    """Manda el correo A. Registra el aviso ANTES de enviarlo, con clave
       determinista, para que un reinicio no lo duplique."""
    fecha = fecha or hoy_mexico()
    pendientes = db.pendientes_de_aviso(seg['id'])
    if not pendientes:
        return None

    v = hay_que_avisar({'nuevas': pendientes, 'reediciones': []})
    if not v['avisar']:
        if v.get('escalar'):
            db.actualizar_seguimiento(seg['id'], {'estado': 'requiere_atencion'})
        return v

    organo = seg.get('organo') or {}
    seg_correo = {**seg, 'fecha_local': fecha, 'url_fuente': dictamen.get('url')}
    asunto, cuerpo = correo.correo_actuacion(
        seg_correo, organo, pendientes, ahora_mexico(), BASE_URL)

    clave = f"actuacion:{seg['id']}:{fecha}:{len(pendientes)}"
    fila = db.registrar_aviso({
        'user_id': seg['user_id'], 'seguimiento_id': seg['id'],
        'tipo': 'actuacion', 'fecha_local': fecha, 'clave_idem': clave,
        'destinatario': seg['correo_aviso'], 'asunto': asunto})
    if fila is None:
        return {'avisar': False, 'motivo': 'ya_avisado'}

    rid = correo.enviar(seg['correo_aviso'], asunto, cuerpo)
    db.marcar_aviso_enviado(fila['id'], rid)
    db.marcar_avisadas([p['id'] for p in pendientes])
    db.actualizar_seguimiento(seg['id'], {
        'ultima_actuacion_en': max((p.get('fecha_auto') or '') for p in pendientes) or None})
    return {'avisar': True, 'n': len(pendientes), 'resend': rid, 'asunto': asunto}


# ── El pase ───────────────────────────────────────────────────────────

def barrer(pase=1, fecha=None, solo=None, verboso=True, forzar=False):
    fecha = fecha or hoy_mexico()
    ok, hora = guardia_de_huso()
    if not ok and not forzar:
        # La guardia existe para el cron: si México volviera a cambiar de huso,
        # dispararía a otra hora local y los correos dirían una hora falsa.
        # Una corrida a mano es otra cosa y puede saltarla a propósito.
        print(f'✗ el barrido se disparó a las {hora}h local: fuera de ventana. '
              'No ejecuto. (--forzar para una corrida a mano)')
        return {'ok': False, 'motivo': 'fuera_de_ventana'}
    if not ok and verboso:
        print(f'· fuera de ventana ({hora}h local), forzado a mano\n')

    corrida = db.abrir_corrida(fecha, pase, 'manual')
    seguimientos = db.seguimientos_activos('automatico', jurisdiccion='PJF')
    if solo:
        seguimientos = [s for s in seguimientos if s['id'] == solo]

    n = {'total': len(seguimientos), 'ok': 0, 'novedad': 0, 'fallo': 0, 'inhabil': 0}
    if verboso:
        print(f'Corrida {fecha} pase {pase} · {len(seguimientos)} expedientes\n')

    for i, seg in enumerate(seguimientos):
        organo = seg.get('organo') or {}
        inhabil = db.es_inhabil(organo.get('jurisdiccion') or 'PJF', fecha)
        if inhabil:
            db.registrar_revision({
                'corrida_id': corrida['id'], 'seguimiento_id': seg['id'],
                'user_id': seg['user_id'], 'fecha_local': fecha, 'intento': 1,
                'resultado': 'inhabil', 'detalle': inhabil.get('motivo')})
            n['inhabil'] += 1
            continue

        if i:
            time.sleep(PAUSA_ENTRE_PETICIONES)

        dictamen = revisar(seg, corrida['id'], intento=pase, fecha=fecha)
        if dictamen is None:
            n['fallo'] += 1
            db.actualizar_seguimiento(
                seg['id'], {'fallos_consecutivos': (seg.get('fallos_consecutivos') or 0) + 1})
            if verboso:
                print(f"  ✗ {seg['numero']:14} {seg['alias'][:34]:36} no se pudo leer")
            continue

        n['ok'] += 1
        cuantas = len(dictamen['nuevas']) + len(dictamen['reediciones'])
        if cuantas:
            n['novedad'] += 1
            r = avisar_novedad(seg, dictamen, fecha)
            if verboso:
                print(f"  ● {seg['numero']:14} {seg['alias'][:34]:36} "
                      f"{cuantas} nueva(s) · correo {r.get('resend','—') if r else '—'}")
        elif verboso:
            print(f"  · {seg['numero']:14} {seg['alias'][:34]:36} sin novedad")

    # Ciudad de México va aparte: una sola descarga del boletín resuelve toda
    # su cartera, así que no tiene sentido meterla en el bucle por expediente.
    c = barrer_cdmx(corrida['id'], fecha, verboso)
    for k in ('total', 'ok', 'novedad', 'fallo'):
        n[k] = n.get(k, 0) + c.get(k, 0)

    db.cerrar_corrida(corrida['id'], n_total=n['total'], n_ok=n['ok'],
                      n_novedad=n['novedad'], n_fallo=n['fallo'])
    if verboso:
        print(f"\n{n['ok']} leídos · {n['novedad']} con novedad · "
              f"{n['fallo']} sin lectura · {n['inhabil']} inhábiles")
    return {'ok': True, **n}


# ── Ciudad de México: por boletín ─────────────────────────────────────

def _acuerdos_de_entradas(seg, organo, entradas, fecha_boletin):
    """Convierte las entradas del boletín en acuerdos con sus huellas.

    La fecha del acuerdo sale del rótulo «ACUERDOS DEL 1 DE SEPTIEMBRE DEL
    2026» de la propia sección; si no se pudo leer, se usa la del boletín, que
    es un día después. Se prefiere la del rótulo por lo mismo que en el PJF: es
    la fecha del juez y no se mueve.
    """
    salida = []
    for e in entradas:
        fecha = _fecha_de_rotulo(e.get('acuerdos_del')) or fecha_boletin
        texto = e['texto']
        salida.append({
            'orden_en_lista': None, 'fecha_auto': fecha,
            'fecha_publicacion': fecha_boletin,
            'cuaderno': f"Secretaría {e.get('secretaria')}" if e.get('secretaria') else None,
            'resumen': texto,
            'huella_clave': pjf.huella_clave(
                'CDMX', organo.get('clave_externa') or '', seg['numero'], None,
                e.get('secretaria') or '', fecha, texto),
            'huella_texto': pjf.huella_texto(texto),
            'simhash': pjf.simhash64(texto),
            'pagina': e.get('pagina'),
        })
    return salida


_MESES_TXT = {'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5,
              'junio': 6, 'julio': 7, 'agosto': 8, 'septiembre': 9,
              'octubre': 10, 'noviembre': 11, 'diciembre': 12}


def _fecha_de_rotulo(rotulo):
    """«1 DE SEPTIEMBRE DEL 2026» → «2026-09-01»."""
    import re
    if not rotulo:
        return None
    m = re.search(r'(\d{1,2})\s+DE\s+([A-ZÁÉÍÓÚa-záéíóú]+)\s+DEL?\s+(\d{4})',
                  rotulo, re.I)
    if not m:
        return None
    mes = _MESES_TXT.get(m.group(2).lower())
    return f'{m.group(3)}-{mes:02d}-{int(m.group(1)):02d}' if mes else None


def barrer_cdmx(corrida_id, fecha=None, verboso=True):
    """Una descarga del boletín resuelve TODA la cartera de CDMX.

    Es la ventaja de indexar por día en vez de por expediente: da igual que
    haya diez expedientes o mil, el coste de red es el mismo.
    """
    fecha = fecha or hoy_mexico()
    seguimientos = db.seguimientos_activos('automatico', jurisdiccion='CDMX')
    if not seguimientos:
        return {'total': 0}

    base_com = {'corrida_id': corrida_id, 'fecha_local': fecha, 'intento': 1}
    try:
        boletin = cdmx.del_dia(fecha) or cdmx.indice()[0]
        ruta = f'/tmp/boletin_{boletin["fecha"]}.pdf'
        import os
        if not os.path.exists(ruta):
            cdmx.descargar(boletin['url'], ruta)
        idx = cdmx.indexar(ruta)
    except Exception as e:
        # Sin boletín no se puede afirmar nada de NINGÚN expediente de CDMX.
        for seg in seguimientos:
            db.registrar_revision({**base_com, 'seguimiento_id': seg['id'],
                                   'user_id': seg['user_id'],
                                   'resultado': 'fallo_red',
                                   'detalle': f'boletín: {e}'[:400]})
            db.actualizar_seguimiento(seg['id'], {
                'fallos_consecutivos': (seg.get('fallos_consecutivos') or 0) + 1})
        if verboso:
            print(f'  ✗ CDMX: no se pudo traer el boletín ({e})')
        return {'total': len(seguimientos), 'fallo': len(seguimientos)}

    if verboso:
        print(f"  boletín del {boletin['fecha']} · {idx['paginas']} páginas · "
              f"{sum(len(v) for v in idx['entradas'].values()):,} entradas")

    n = {'total': len(seguimientos), 'ok': 0, 'novedad': 0, 'fallo': 0}
    for seg in seguimientos:
        organo = seg.get('organo') or {}
        entradas = cdmx.buscar(idx, organo.get('nombre') or '', seg['numero'])
        acuerdos = _acuerdos_de_entradas(seg, organo, entradas, boletin['fecha'])
        conocidas = db.actuaciones_de(seg['id'])
        dictamen = comparar(acuerdos, conocidas)
        nuevas = dictamen['nuevas'] + dictamen['reediciones']

        if nuevas:
            db.guardar_actuaciones([{
                'seguimiento_id': seg['id'], 'user_id': seg['user_id'],
                'huella_clave': a['huella_clave'], 'huella_texto': a['huella_texto'],
                'simhash': a.get('simhash'), 'cuaderno': a.get('cuaderno'),
                'fecha_auto': a.get('fecha_auto'),
                'fecha_publicacion': a.get('fecha_publicacion'),
                'resumen': a['resumen'], 'url_fuente': boletin['url'],
                'origen': 'cdmx_boletin', 'version': a.get('version', 1),
                'reemplaza_a': a.get('reemplaza_a'), 'es_linea_base': False,
            } for a in nuevas])

        db.registrar_revision({
            **base_com, 'seguimiento_id': seg['id'], 'user_id': seg['user_id'],
            'resultado': 'ok_con_novedad' if nuevas else 'ok_sin_novedad',
            'n_actuaciones_vistas': len(acuerdos),
            'terminada_en': datetime.now(timezone.utc).isoformat()})
        db.actualizar_seguimiento(seg['id'], {
            'ultima_revision_ok': datetime.now(timezone.utc).isoformat(),
            'fallos_consecutivos': 0})
        n['ok'] += 1

        if nuevas:
            n['novedad'] += 1
            r = avisar_novedad(seg, {'url': boletin['url']}, fecha)
            if verboso:
                print(f"  ● {seg['numero']:14} {seg['alias'][:34]:36} "
                      f"{len(nuevas)} nueva(s) · correo {r.get('resend','—') if r else '—'}")
        elif verboso:
            print(f"  · {seg['numero']:14} {seg['alias'][:34]:36} sin novedad")
    return n


def alta_cdmx(nombre_juzgado, numero, alias, correo_aviso, user_id):
    """Alta de un expediente de CDMX: se busca en el boletín más reciente y se
       guarda como línea base lo que salga."""
    organo = db._pedir(
        'GET', f'/seg_organos?jurisdiccion=eq.CDMX'
               f'&clave_externa=eq.{urllib_quote(cdmx.llano(nombre_juzgado))}&select=*')
    if not organo:
        raise SystemExit(f'no está el juzgado «{nombre_juzgado}» en el catálogo')
    organo = organo[0]

    boletin = cdmx.indice()[0]
    ruta = f'/tmp/boletin_{boletin["fecha"]}.pdf'
    import os
    if not os.path.exists(ruta):
        cdmx.descargar(boletin['url'], ruta)
    idx = cdmx.indexar(ruta)
    entradas = cdmx.buscar(idx, organo['nombre'], numero)

    print(f"\n{organo['nombre']}")
    print(f"  expediente {numero} · {len(entradas)} entrada(s) en el boletín "
          f"del {boletin['fecha']}")
    for e in entradas:
        print(f"  pág {e['pagina']} · {e['texto'][:96]}")

    seg = db.crear_seguimiento({
        'user_id': user_id, 'organo_id': organo['id'], 'numero': numero,
        'anio': int(numero.split('/')[-1]) if '/' in numero else None,
        'alias': alias, 'modo': 'automatico', 'correo_aviso': correo_aviso,
        'linea_base_en': datetime.now(timezone.utc).isoformat()})

    ahora = datetime.now(timezone.utc).isoformat()
    acuerdos = _acuerdos_de_entradas(seg, organo, entradas, boletin['fecha'])
    db.guardar_actuaciones([{
        'seguimiento_id': seg['id'], 'user_id': user_id,
        'huella_clave': a['huella_clave'], 'huella_texto': a['huella_texto'],
        'simhash': a.get('simhash'), 'cuaderno': a.get('cuaderno'),
        'fecha_auto': a.get('fecha_auto'), 'fecha_publicacion': a.get('fecha_publicacion'),
        'resumen': a['resumen'], 'url_fuente': boletin['url'],
        'origen': 'cdmx_boletin', 'version': 1,
        'es_linea_base': True, 'avisada_en': ahora,
    } for a in acuerdos])
    print(f"\n  Alta hecha. {len(acuerdos)} actuaciones como línea base.")
    return seg


def urllib_quote(t):
    from urllib.parse import quote
    return quote(t)


# ── El cierre: la regla 3 ─────────────────────────────────────────────

def cerrar(fecha=None, verboso=True):
    """El pase de las 11:00. No consulta portales; cierra el día.

    Para cada expediente cuyo MEJOR resultado del día sea un fallo, manda el
    correo B —o el último aviso, si ya van tres días— y escala a David. Es lo
    que impide que un fallo se convierta en silencio, que para el abogado es
    indistinguible de «no pasó nada».
    """
    fecha = fecha or hoy_mexico()
    corrida = db.abrir_corrida(fecha, 4, 'manual')
    seguimientos = db.seguimientos_activos('automatico')
    avisados, escalados = [], []

    for seg in seguimientos:
        filas = db._pedir('GET',
            f"/seg_revisiones?seguimiento_id=eq.{seg['id']}&fecha_local=eq.{fecha}"
            '&select=resultado,intento,iniciada_en&order=intento.asc') or []
        if not filas:
            continue
        # Si CUALQUIER intento del día salió bien, el día está resuelto.
        if any(f['resultado'].startswith('ok_') or f['resultado'] == 'inhabil'
               for f in filas):
            continue

        organo = seg.get('organo') or {}
        url_manual = pjf.url_de(organo.get('clave_externa'), seg['numero'],
                                seg.get('tipo_asunto_clave') or 1,
                                seg.get('tipo_procedimiento_clave') or 0)
        horas = [(f.get('iniciada_en') or '')[11:16] for f in filas if f.get('iniciada_en')]
        dias = (seg.get('fallos_consecutivos') or 0)
        seg_c = {**seg, 'fecha_local': fecha}

        if dias >= 3:
            # Al tercero se deja de escribir a diario y se escala. Insistir
            # cada mañana con un problema que no es suyo sólo consigue que
            # deje de leer los correos de Iurexia.
            asunto, cuerpo = correo.correo_ultimo_aviso(seg_c, organo, url_manual, BASE_URL)
            tipo, clave = 'ultimo_aviso', f"ultimo:{seg['id']}:{dias}"
            escalados.append(seg)
        else:
            asunto, cuerpo = correo.correo_no_se_pudo(
                seg_c, organo, horas, dias, url_manual, BASE_URL)
            tipo, clave = 'no_se_pudo', f"nopude:{seg['id']}:{fecha}"

        fila = db.registrar_aviso({
            'user_id': seg['user_id'], 'seguimiento_id': seg['id'], 'tipo': tipo,
            'fecha_local': fecha, 'clave_idem': clave,
            'destinatario': seg['correo_aviso'], 'asunto': asunto})
        if fila is None:
            continue
        rid = correo.enviar(seg['correo_aviso'], asunto, cuerpo)
        db.marcar_aviso_enviado(fila['id'], rid)
        avisados.append((seg['numero'], tipo, rid))
        if verboso:
            print(f"  ✉ {seg['numero']:14} {tipo:14} {rid}")

    if escalados:
        _escalar_a_david(escalados, fecha)

    db.cerrar_corrida(corrida['id'], n_total=len(seguimientos),
                      n_fallo=len(avisados), nota='cierre')
    if verboso:
        print(f"\nCierre {fecha}: {len(avisados)} avisos de «no se pudo», "
              f"{len(escalados)} escalados")
    return {'avisados': avisados, 'escalados': len(escalados)}


def _escalar_a_david(seguimientos, fecha):
    """Una sola vez por incidencia, no una por expediente."""
    lineas = [f'Corrida del {fecha}, pase de cierre.', '',
              f'{len(seguimientos)} expedientes llevan 3 días o más sin lectura correcta.', '']
    for s in seguimientos:
        o = s.get('organo') or {}
        lineas.append(f"  {s['numero']:14} {o.get('nombre','')[:52]}")
        lineas.append(f"  {'':14} fallos consecutivos: {s.get('fallos_consecutivos')}")
    lineas += ['', 'Los abogados afectados recibieron hoy el último aviso y se les',
               'silenció el correo diario.', '',
               f'Bitácora: {BASE_URL}/admin/seguimiento/corridas/{fecha}']
    asunto = f'[Seguimiento] 3 días sin lectura: {len(seguimientos)} expedientes'
    try:
        correo.enviar('jdm.juridico@gmail.com', asunto, '\n'.join(lineas))
    except Exception as e:
        print(f'  ✗ no se pudo escalar a David: {e}')


# ── Alta ──────────────────────────────────────────────────────────────

def alta(organismo, numero, alias, correo_aviso, user_id, tipo_asunto=1,
         tipo_procedimiento=0, jurisdiccion='PJF'):
    """Da de alta un expediente y guarda su histórico CALLADO.

    El histórico entra con `es_linea_base = true` y `avisada_en` puesto: sin
    esto, el primer barrido mandaría un correo con años de acuerdos.
    """
    organo = db.organo_por_clave(jurisdiccion, str(organismo))
    if not organo:
        raise SystemExit(f'no está el órgano {organismo} en el catálogo')

    lectura = pjf.leer(organismo, numero, tipo_asunto, tipo_procedimiento)
    c = lectura['caratula']
    print(f"\n{c['organo']}")
    print(f"  expediente {c['expediente']} · NEUN {c['neun']} · "
          f"{len(lectura['acuerdos'])} acuerdos en el histórico")
    if lectura['acuerdos']:
        u = lectura['acuerdos'][-1]
        print(f"  último: {u['fecha_auto']} · {(u['resumen'] or '')[:66]}")

    seg = db.crear_seguimiento({
        'user_id': user_id, 'organo_id': organo['id'], 'numero': numero,
        'anio': int(numero.split('/')[-1]) if '/' in numero else None,
        'tipo_asunto_clave': str(tipo_asunto),
        'tipo_procedimiento_clave': str(tipo_procedimiento),
        'neun': c.get('neun'), 'alias': alias, 'modo': 'automatico',
        'correo_aviso': correo_aviso,
        'linea_base_en': datetime.now(timezone.utc).isoformat(),
    })

    ahora = datetime.now(timezone.utc).isoformat()
    db.guardar_actuaciones([{
        'seguimiento_id': seg['id'], 'user_id': user_id,
        'huella_clave': a['huella_clave'], 'huella_texto': a['huella_texto'],
        'simhash': a.get('simhash'), 'cuaderno': a.get('cuaderno'),
        'fecha_auto': a.get('fecha_auto'),
        'fecha_publicacion': a.get('fecha_publicacion'),
        'orden_en_lista': a.get('orden_en_lista'),
        'resumen': a.get('resumen') or '', 'url_fuente': lectura['url'],
        'origen': 'pjf_vercaptura', 'version': 1,
        'es_linea_base': True, 'avisada_en': ahora,
    } for a in lectura['acuerdos']])

    print(f"\n  Alta hecha. {len(lectura['acuerdos'])} actuaciones guardadas como "
          'línea base (nadie recibe correo por ellas).')
    print(f"  Se revisa cada día a las 9:10, hora de la Ciudad de México.")
    return seg


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--alta', nargs=2, metavar=('ORGANISMO', 'EXPEDIENTE'))
    p.add_argument('--alta-cdmx', nargs=2, metavar=('JUZGADO', 'EXPEDIENTE'))
    p.add_argument('--alias', default='')
    p.add_argument('--correo', default='')
    p.add_argument('--tipoasunto', type=int, default=1)
    p.add_argument('--tipoprocedimiento', type=int, default=0)
    p.add_argument('--barrer', action='store_true')
    p.add_argument('--cerrar', action='store_true',
                   help='pase de cierre: correos de «no se pudo» y escalado')
    p.add_argument('--pase', type=int, default=1)
    p.add_argument('--estado', action='store_true')
    p.add_argument('--forzar', action='store_true',
                   help='corre aunque sea fuera de la ventana de las 9:10')
    a = p.parse_args()

    if a.alta:
        u = db.usuario_por_correo(a.correo)
        if not u:
            raise SystemExit(f'no hay usuario con el correo {a.correo}')
        alta(a.alta[0], a.alta[1], a.alias or a.alta[1], a.correo, u['id'],
             a.tipoasunto, a.tipoprocedimiento)
    elif a.alta_cdmx:
        u = db.usuario_por_correo(a.correo)
        if not u:
            raise SystemExit(f'no hay usuario con el correo {a.correo}')
        alta_cdmx(a.alta_cdmx[0], a.alta_cdmx[1], a.alias or a.alta_cdmx[1],
                  a.correo, u['id'])
    elif a.barrer:
        barrer(pase=a.pase, forzar=a.forzar)
    elif a.cerrar:
        cerrar()
    elif a.estado:
        for s in db.seguimientos_activos('automatico'):
            o = s.get('organo') or {}
            print(f"  {s['numero']:14} {s['alias'][:30]:32} {o.get('nombre','')[:44]}")
            print(f"  {'':14} última lectura {s.get('ultima_revision_ok') or '—'} · "
                  f"fallos {s.get('fallos_consecutivos')}")
    else:
        p.print_help()


if __name__ == '__main__':
    main()
