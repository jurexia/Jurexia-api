#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Los dos correos del seguimiento, y el de escalado.

POR QUÉ SON DOS Y NO UNO. Porque el silencio de Iurexia tiene que significar
UNA sola cosa. Si sólo existiera el correo de «hay novedad», un día sin correo
sería a la vez «no pasó nada» y «no pudimos mirar», y el abogado no podría
distinguirlos. El correo B existe para que el silencio siga siendo fiable.

SIN NOMBRES DE PARTES EN EL ASUNTO. El asunto viaja por servidores ajenos y se
lee en la pantalla de bloqueo del móvil. Van el número y el órgano, que son
públicos; el nombre del quejoso, no.

EL DESCARGO VA ÍNTEGRO. Lo impone el propio Consejo de la Judicatura y además
protege al producto: esto es informativo y no sustituye la consulta del
expediente.
"""
import io
import json
import os
import urllib.error
import urllib.request
from datetime import datetime

AQUI = os.path.dirname(os.path.abspath(__file__))

REMITENTE = 'Iurexia <avisos@iurexia.com>'
RESPONDER_A = 'soporte@iurexia.com'

DESCARGO = (
    'El Consejo de la Judicatura Federal advierte que esta información «es '
    'únicamente de carácter informativo y que si bien es la misma que se '
    'encuentra en los Estrados de los Juzgados y Tribunales Federales, no se '
    'debe tomar como oficial. Por lo tanto, no será válida para ser utilizada '
    'en ningún tipo de proceso jurídico.» Iurexia no sustituye la consulta del '
    'expediente.'
)

MESES = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio',
         'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']


def _clave_resend():
    env = dict(os.environ)
    ruta = os.path.join(AQUI, '.env')
    if os.path.exists(ruta):
        for l in io.open(ruta, encoding='utf-8'):
            l = l.strip()
            if l and not l.startswith('#') and '=' in l:
                k, v = l.split('=', 1)
                env.setdefault(k, v.strip().strip('"').strip("'"))
    return env.get('RESEND_API_KEY', '')


def en_letra(iso):
    """«2026-01-19» → «19 de enero de 2026». Sin `datetime.strptime` sobre la
       zona: la fecha ya viene como día natural de México."""
    try:
        a, m, d = iso.split('-')
        return f'{int(d)} de {MESES[int(m) - 1]} de {a}'
    except Exception:
        return iso or ''


def _pie(url_baja):
    return (
        '—\n'
        'Iurexia le escribe sólo cuando hay algo nuevo. Si un día no le escribimos,\n'
        'es que no hubo actuación; y si no pudimos revisar, también se lo decimos.\n\n'
        + DESCARGO + '\n\n'
        f'Dejar de seguir este expediente: {url_baja}\n'
    )


# ── Correo A · actuación nueva ────────────────────────────────────────

def correo_actuacion(seg, organo, actuaciones, revisado_a_las, base_url='https://iurexia.com'):
    n = len(actuaciones)
    numero = seg['numero']
    corto = (organo or {}).get('nombre', '')
    asunto = (f'Movimiento en el {numero} — {corto}' if n == 1
              else f'{n} movimientos en el {numero} — {corto}')

    lineas = [f"Licenciado {seg.get('tratamiento') or ''}".rstrip() + ':', '']
    lineas.append('Hay una actuación nueva en un expediente que sigue con Iurexia.'
                  if n == 1 else
                  f'Hay {n} actuaciones nuevas en un expediente que sigue con Iurexia.')
    lineas.append('')
    lineas.append(f"  Expediente     {numero}"
                  + (f" · {seg['tipo_asunto_nombre']}" if seg.get('tipo_asunto_nombre') else ''))
    lineas.append(f"  Órgano         {corto}")
    lineas.append(f"  Su referencia  «{seg.get('alias','')}»")
    lineas.append('')

    for a in actuaciones:
        rotulo = f"  Acuerdo del {en_letra(a.get('fecha_auto'))}"
        if a.get('cuaderno'):
            rotulo += f" · cuaderno {a['cuaderno']}"
        if (a.get('version') or 1) > 1:
            rotulo += ' · CORREGIDO por el juzgado'
        lineas.append(rotulo)
        if a.get('fecha_publicacion'):
            lineas.append(f"  Publicado en estrados el {en_letra(a['fecha_publicacion'])}")
        lineas.append('')
        texto = (a.get('resumen') or '').strip()
        recorte = texto[:420] + ('...' if len(texto) > 420 else '')
        for trozo in _envolver(recorte, 72):
            lineas.append(f'   {trozo}')
        if (a.get('version') or 1) > 1:
            lineas.append('')
            lineas.append('   El juzgado modificó este acuerdo después de publicarlo.')
        lineas.append('')

    lineas.append(f"  → Verlo en Iurexia")
    lineas.append(f"    {base_url}/carpetas/seguimiento/{seg['id']}")
    lineas.append('')
    if seg.get('url_fuente'):
        lineas.append('  → Verlo en el portal del Consejo de la Judicatura Federal')
        lineas.append(f"    {seg['url_fuente']}")
        lineas.append('')
    lineas.append(f'Revisado hoy, {en_letra(seg.get("fecha_local"))}, a las {revisado_a_las},')
    lineas.append('hora de la Ciudad de México.')
    lineas.append('')
    lineas.append(_pie(f"{base_url}/carpetas/seguimiento/{seg['id']}/baja"))

    return asunto, '\n'.join(lineas)


# ── Correo B · no se pudo revisar ─────────────────────────────────────

def correo_no_se_pudo(seg, organo, intentos, dia_consecutivo, url_manual,
                      base_url='https://iurexia.com'):
    numero = seg['numero']
    asunto = f'Hoy no pudimos revisar el {numero} (y no sabemos si hubo movimiento)'
    if dia_consecutivo == 2:
        asunto = f'No pudimos revisar el {numero} — segundo día'

    horas = ', a las '.join(intentos) if intentos else ''
    lineas = [f"Licenciado {seg.get('tratamiento') or ''}".rstrip() + ':', '']
    lineas.append(f'Hoy, {en_letra(seg.get("fecha_local"))}, no conseguimos consultar '
                  'este expediente:')
    lineas.append('')
    lineas.append(f"  Expediente {numero} · {(organo or {}).get('nombre','')}")
    lineas.append(f"  Su referencia «{seg.get('alias','')}»")
    lineas.append('')
    if intentos:
        lineas.append(f'Lo intentamos {len(intentos)} '
                      + ('vez' if len(intentos) == 1 else 'veces')
                      + f': a las {horas}, hora de la Ciudad de México.')
    lineas.append('El portal no respondió como esperábamos.')
    lineas.append('')
    lineas.append('Le escribimos porque nuestro silencio significa siempre «no hubo')
    lineas.append('movimiento», y hoy no podemos afirmarlo. Puede que no haya pasado')
    lineas.append('nada; puede que sí.')
    lineas.append('')
    lineas.append('  → Consultarlo usted mismo ahora, con el formulario ya preparado')
    lineas.append(f'    {url_manual}')
    lineas.append('')
    lineas.append('Mañana a las 9:10 volvemos a intentarlo, y si el portal se recupera')
    lineas.append('le avisaremos de cualquier actuación que aparezca con fecha de estos')
    lineas.append('días: no se pierde nada, sólo se retrasa.')
    lineas.append('')
    lineas.append('—')
    lineas.append('Iurexia')
    return asunto, '\n'.join(lineas)


def correo_ultimo_aviso(seg, organo, url_manual, base_url='https://iurexia.com'):
    numero = seg['numero']
    asunto = f'Seguimos sin poder revisar el {numero}. Dejamos de escribirle a diario.'
    cuerpo = f"""Licenciado {seg.get('tratamiento') or ''}:

Llevamos tres días sin poder consultar el {numero}. El problema es nuestro o
del portal, no suyo, y ya está en manos de nuestro equipo técnico.

Para no darle la lata cada mañana, dejamos de escribirle a diario sobre esto.
Seguimos intentándolo todos los días a las 9:10, y le escribiremos en cuanto
volvamos a tener lectura del expediente — con todo lo que haya salido entre
tanto. El estado, siempre al día, en {base_url}/carpetas

Mientras, conviene que lo consulte usted directamente:
{url_manual}

—
Iurexia
"""
    return asunto, cuerpo.replace('Licenciado :', 'Licenciado:')


# ── Envío ─────────────────────────────────────────────────────────────

def _envolver(texto, ancho):
    palabras, linea, salida = texto.split(), '', []
    for p in palabras:
        if len(linea) + len(p) + 1 > ancho:
            salida.append(linea)
            linea = p
        else:
            linea = f'{linea} {p}'.strip()
    if linea:
        salida.append(linea)
    return salida or ['']


def _html_de(texto):
    """El cuerpo va en texto plano dentro de un <pre>: es un aviso operativo,
       no una pieza de marketing, y la alineación de la ficha del expediente
       es lo que lo hace legible de un vistazo."""
    from html import escape
    return (
        '<div style="background:#f6f5f1;padding:28px 16px;font-family:'
        'ui-monospace,SFMono-Regular,Menlo,monospace">'
        '<div style="max-width:640px;margin:0 auto;background:#fffefb;'
        'border:1px solid #e2ded4;border-radius:10px;padding:26px 28px">'
        '<div style="font:600 11px/1 ui-monospace,monospace;letter-spacing:.16em;'
        'text-transform:uppercase;color:#9a7526;margin-bottom:18px">Iurexia · '
        'Seguimiento de expedientes</div>'
        f'<pre style="margin:0;white-space:pre-wrap;word-wrap:break-word;'
        f'font:13px/1.6 ui-monospace,monospace;color:#17181c">{escape(texto)}</pre>'
        '</div></div>'
    )


def enviar(destinatario, asunto, cuerpo, responder_a=RESPONDER_A):
    """Devuelve el id de Resend. Levanta RuntimeError si el envío falla."""
    clave = _clave_resend()
    if not clave:
        raise RuntimeError('falta RESEND_API_KEY')
    datos = json.dumps({
        'from': REMITENTE, 'to': [destinatario], 'subject': asunto,
        'text': cuerpo, 'html': _html_de(cuerpo), 'reply_to': responder_a,
    }, ensure_ascii=False).encode()
    req = urllib.request.Request(
        'https://api.resend.com/emails', data=datos, method='POST',
        headers={'Authorization': f'Bearer {clave}',
                 'Content-Type': 'application/json',
                 # Cloudflare bloquea el User-Agent por defecto de urllib.
                 'User-Agent': 'iurexia-seguimiento/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            return json.load(r).get('id')
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'Resend {e.code}: {e.read().decode()[:300]}') from None
