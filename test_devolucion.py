#!/usr/bin/env python3
"""Cuándo se devuelve la consulta y cuándo no.

La regla es una sola: si el búfer salió vacío, el abogado no recibió nada y
no hay nada que cobrar. Lo que hace delicada la regla no es el caso feliz,
sino los bordes — que un corte voluntario NO devuelva, y que una respuesta
truncada tampoco.

    python3 test_devolucion.py
"""
import asyncio
import sys


class _RPC:
    """Supabase de mentira: apunta las llamadas en vez de tocar la red."""
    def __init__(self): self.devoluciones = []
    def rpc(self, nombre, params):
        self.devoluciones.append((nombre, params['p_user_id']))
        return self
    def execute(self):
        return type('R', (), {'data': {'devuelta': True, 'used': 41, 'limit': 140}})()


async def simular(excepcion, contenido, user_id='abogada-1'):
    """Reproduce el manejador: misma condición, mismo orden."""
    sb = _RPC()
    salida = []
    try:
        if excepcion:
            raise excepcion
    except Exception as e:                      # ← igual que en main.py
        if not contenido.strip() and user_id and sb:
            sb.rpc('devolver_consulta', {'p_user_id': user_id}).execute()
        if contenido.strip():
            salida.append('truncada')
        else:
            salida.append('no pudimos completar')
        return sb.devoluciones, salida, str(e)
    except BaseException:
        # CancelledError / GeneratorExit no deben llegar aquí desde `except Exception`
        return sb.devoluciones, ['CORTE'], ''
    return sb.devoluciones, ['ok'], ''


CASOS = [
    ('fallo del modelo, sin texto',      RuntimeError('BadRequestError'), '',            True,  'no pudimos completar'),
    ('fallo tras responder',             RuntimeError('stream cut'),      'Artículo 252…', False, 'truncada'),
    ('fallo con texto en blancos',       RuntimeError('x'),               '   \n  ',      True,  'no pudimos completar'),
    ('todo bien, sin excepción',         None,                            'respuesta',    False, 'ok'),
]

fallos = []
print()
for etiqueta, exc, contenido, espera_dev, espera_msg in CASOS:
    devs, salida, err = asyncio.run(simular(exc, contenido))
    hubo = len(devs) == 1
    ok = (hubo == espera_dev) and (salida[0] == espera_msg)
    print(f"  {'ok  ' if ok else 'MAL '} {etiqueta:32s} devuelve={str(hubo):5s} → {salida[0]}")
    if not ok:
        fallos.append(etiqueta)

# El corte del usuario: CancelledError hereda de BaseException.
print(f"\n  {'ok  ' if not issubclass(asyncio.CancelledError, Exception) else 'MAL '} "
      f"cortar desde el navegador NO entra por `except Exception` "
      f"(CancelledError hereda de BaseException)")
if issubclass(asyncio.CancelledError, Exception):
    fallos.append('CancelledError sería capturado')

# Nunca se devuelve sin usuario identificado.
devs, _, _ = asyncio.run(simular(RuntimeError('x'), '', user_id=None))
ok = len(devs) == 0
print(f"  {'ok  ' if ok else 'MAL '} sin user_id no se llama a devolver_consulta")
if not ok:
    fallos.append('devolvió sin user_id')

print()
if fallos:
    print('FALLOS: ' + '; '.join(fallos)); sys.exit(1)
print('La devolución sólo ocurre cuando el abogado no recibió nada.')
