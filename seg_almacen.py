#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persistencia del seguimiento de expedientes.

Habla con Supabase por PostgREST con la clave de servicio, que omite RLS por
diseño: el barrido escribe en nombre de todos los abogados. La RLS sigue
protegiendo al frontend, que usa la anon key con la sesión de cada uno.

LO QUE ESTE MÓDULO GARANTIZA, Y POR QUÉ IMPORTA:

  · Cada revisión deja fila, se pueda o no leer el expediente. Es lo que hace
    que «hoy no pudimos revisar» sea un hecho consultable y no una opinión.
  · Los avisos se registran con clave determinista antes de mandarse, así que
    un reinicio a mitad del envío no manda el correo dos veces.
  · El alta guarda el histórico marcado como línea base, callado.
"""
import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request

AQUI = os.path.dirname(os.path.abspath(__file__))


def _entorno():
    env = dict(os.environ)
    ruta = os.path.join(AQUI, '.env')
    if os.path.exists(ruta):
        for l in io.open(ruta, encoding='utf-8'):
            l = l.strip()
            if l and not l.startswith('#') and '=' in l:
                k, v = l.split('=', 1)
                env.setdefault(k, v.strip().strip('"').strip("'"))
    return env


_ENV = _entorno()
URL = (_ENV.get('SUPABASE_URL') or _ENV.get('NEXT_PUBLIC_SUPABASE_URL', '')).rstrip('/')
CLAVE = _ENV.get('SUPABASE_SERVICE_KEY') or _ENV.get('SUPABASE_SERVICE_ROLE_KEY', '')


def _pedir(metodo, ruta, cuerpo=None, prefer=None, timeout=60):
    cabeceras = {
        'apikey': CLAVE, 'Authorization': f'Bearer {CLAVE}',
        'Content-Type': 'application/json', 'Accept': 'application/json',
    }
    if prefer:
        cabeceras['Prefer'] = prefer
    datos = json.dumps(cuerpo, ensure_ascii=False).encode() if cuerpo is not None else None
    req = urllib.request.Request(f'{URL}/rest/v1{ruta}', data=datos,
                                 headers=cabeceras, method=metodo)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            crudo = r.read()
            return json.loads(crudo) if crudo else None
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'{metodo} {ruta} → {e.code}: {e.read().decode()[:400]}') from None


# ── Lecturas ──────────────────────────────────────────────────────────

def seguimientos_activos(modo='automatico', jurisdiccion=None):
    """Los que le tocan al barrido, ordenados por órgano para agrupar peticiones."""
    q = ('/seg_expedientes_seguidos?estado=eq.activo'
         f'&modo=eq.{modo}&order=organo_id.asc'
         '&select=*,organo:seg_organos(id,jurisdiccion,clave_externa,nombre,entidad)')
    filas = _pedir('GET', q) or []
    if jurisdiccion:
        filas = [f for f in filas
                 if (f.get('organo') or {}).get('jurisdiccion') == jurisdiccion]
    return filas


def actuaciones_de(seguimiento_id):
    return _pedir('GET',
                  f'/seg_actuaciones?seguimiento_id=eq.{seguimiento_id}'
                  '&select=id,huella_clave,huella_texto,simhash,cuaderno,'
                  'fecha_auto,resumen,version&order=version.asc') or []


def pendientes_de_aviso(seguimiento_id):
    return _pedir('GET',
                  f'/seg_actuaciones?seguimiento_id=eq.{seguimiento_id}'
                  '&avisada_en=is.null&es_linea_base=is.false'
                  '&order=fecha_auto.asc') or []


def usuario_por_correo(correo):
    """El uuid de auth.users. Se consulta por RPC no: se usa user_profiles."""
    f = _pedir('GET', '/user_profiles?select=id,email,full_name'
                      f'&email=eq.{urllib.parse.quote(correo)}')
    return (f or [None])[0]


def organo_por_clave(jurisdiccion, clave_externa):
    f = _pedir('GET', f'/seg_organos?jurisdiccion=eq.{jurisdiccion}'
                      f'&clave_externa=eq.{clave_externa}&select=*')
    return (f or [None])[0]


# ── Escrituras ────────────────────────────────────────────────────────

def abrir_corrida(fecha_local, pase, disparo='manual'):
    """Idempotente por el índice único (fecha_local, pase): si ya existe la de
       hoy, la devuelve en vez de crear otra."""
    try:
        f = _pedir('POST', '/seg_corridas',
                   {'fecha_local': fecha_local, 'pase': pase, 'disparo': disparo},
                   prefer='return=representation')
        return f[0]
    except RuntimeError as e:
        if '23505' in str(e) or 'duplicate' in str(e).lower():
            f = _pedir('GET', f'/seg_corridas?fecha_local=eq.{fecha_local}&pase=eq.{pase}')
            return f[0]
        raise


def cerrar_corrida(corrida_id, **contadores):
    from datetime import datetime, timezone
    _pedir('PATCH', f'/seg_corridas?id=eq.{corrida_id}',
           {'terminada_en': datetime.now(timezone.utc).isoformat(), **contadores})


def crear_seguimiento(fila):
    return _pedir('POST', '/seg_expedientes_seguidos', fila,
                  prefer='return=representation')[0]


def guardar_actuaciones(filas):
    """Inserta ignorando las que ya existan por (seguimiento, huella, versión).
       Esa colisión es la última red bajo el detector: si el código se
       equivocara, la base de datos no deja entrar el duplicado."""
    if not filas:
        return 0
    _pedir('POST', '/seg_actuaciones?on_conflict=seguimiento_id,huella_clave,version',
           filas, prefer='resolution=ignore-duplicates,return=minimal')
    return len(filas)


def marcar_avisadas(ids):
    from datetime import datetime, timezone
    if not ids:
        return
    lista = ','.join(f'"{i}"' for i in ids)
    _pedir('PATCH', f'/seg_actuaciones?id=in.({lista})',
           {'avisada_en': datetime.now(timezone.utc).isoformat()})


def registrar_revision(fila):
    """Idempotente por (seguimiento, fecha_local, intento)."""
    _pedir('POST', '/seg_revisiones?on_conflict=seguimiento_id,fecha_local,intento',
           fila, prefer='resolution=merge-duplicates,return=minimal')


def actualizar_seguimiento(seguimiento_id, campos):
    _pedir('PATCH', f'/seg_expedientes_seguidos?id=eq.{seguimiento_id}', campos)


def aviso_ya_enviado(clave_idem):
    f = _pedir('GET', f'/seg_avisos?clave_idem=eq.{urllib.parse.quote(clave_idem)}'
                      '&select=id,estado')
    return bool(f)


def registrar_aviso(fila):
    """Se escribe ANTES de mandar el correo. Si el proceso muere entre esta
       fila y el envío, el correo no sale; si muriera después de mandarlo pero
       antes de escribir, saldría dos veces. Se prefiere perder un aviso a
       duplicarlo: el pase de cierre recoge lo que quedó sin avisar."""
    try:
        return _pedir('POST', '/seg_avisos', fila, prefer='return=representation')[0]
    except RuntimeError as e:
        if '23505' in str(e) or 'duplicate' in str(e).lower():
            return None                       # ya estaba: no se manda otra vez
        raise


def marcar_aviso_enviado(aviso_id, resend_id):
    from datetime import datetime, timezone
    _pedir('PATCH', f'/seg_avisos?id=eq.{aviso_id}',
           {'estado': 'enviado', 'resend_id': resend_id,
            'enviado_en': datetime.now(timezone.utc).isoformat()})


def es_inhabil(jurisdiccion, fecha_local):
    f = _pedir('GET', f'/seg_dias_inhabiles?jurisdiccion=eq.{jurisdiccion}'
                      f'&fecha=eq.{fecha_local}&select=motivo')
    return (f or [None])[0]
