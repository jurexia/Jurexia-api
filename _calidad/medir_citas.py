#!/usr/bin/env python3
"""
Banco de calidad de citas de Iurexia.

Mide, contra producción, tres cosas que hasta ahora nadie miraba de forma
sistemática:

  1. TRAZABILIDAD — cuántas citas [Doc ID:] corresponden a documentos que de
     verdad se recuperaron del acervo. Lo calcula el backend; aquí se recoge.

  2. TESIS REALES — cuántos «Registro digital: NNNNNNN» citados en la prosa
     existen de verdad en el Semanario Judicial de la Federación. Esto es lo
     que ninguna comprobación cubría: el validador del backend sólo mira los
     UUID del acervo, así que un registro inventado pasaba entero.

  3. ACIERTO DE RECUPERACIÓN (opcional) — si la pregunta trae `esperado`,
     cuántos de esos trozos aparecen en alguna fuente recuperada.

Lo importante del diseño: **1 y 2 no necesitan respuesta correcta conocida**.
Se pueden correr sobre cualquier pregunta y dan un número comparable entre
despliegues. Eso convierte «creo que mejoró» en «la tasa de registros
inexistentes pasó de X a Y».

Uso:
    python3 _calidad/medir_citas.py                    # todas las preguntas
    python3 _calidad/medir_citas.py --limite 3         # sólo las 3 primeras
    python3 _calidad/medir_citas.py --etiqueta antes   # nombra la corrida

Los resultados se guardan en _calidad/corridas/ para poder comparar.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
API = os.getenv("IUREXIA_API", "https://jurexia-api.onrender.com")
SJF = "https://sjf2.scjn.gob.mx/services/sjftesismicroservice/api/public/tesis"

# Un usuario con plan Pro o superior: si no, el backend recorta funciones
# (búsqueda web) y la medición no representa lo que ve un abogado de pago.
USUARIO = os.getenv("IUREXIA_USER_ID", "a5dad17e-0635-4fc7-9119-cabcca2d3813")

# Mismo patrón que usa el sello del frontend (SelloCitas.tsx). Si cambia uno,
# cambia el otro: medir con un criterio distinto del que se le enseña al
# abogado sería medir otra cosa.
PATRON_REGISTRO = re.compile(
    r"[Rr]egistro(?:\s+digital)?\s*(?:n[úu]m(?:ero)?\.?)?\s*[:.]?\s*(\d{6,8})"
)


def preguntar(texto: str, estado: str, tiempo_max: int = 180) -> str:
    """Una consulta real contra producción. Devuelve el stream completo."""
    cuerpo = json.dumps({
        "messages": [{"role": "user", "content": texto}],
        "estado": estado,
        "top_k": 30,
        "enable_reasoning": False,
        "genio_ids": [],
        "user_id": USUARIO,
    }).encode()
    req = urllib.request.Request(
        f"{API}/chat", data=cuerpo,
        headers={"Content-Type": "application/json", "X-Razonamiento-Vivo": "1"},
    )
    r = urllib.request.urlopen(req, timeout=tiempo_max)
    buf, salida = b"", ""
    while True:
        trozo = r.read(8192)
        if not trozo:
            break
        buf += trozo
        try:
            salida += buf.decode()
            buf = b""
        except UnicodeDecodeError:
            continue  # carácter partido entre trozos
    return salida


def existe_en_semanario(registro: str) -> str:
    """'existe' | 'no_existe' | 'sin_comprobar'.

    Sólo un 404 prueba que no existe. Cualquier otro fallo es del servidor de
    la Corte, y contarlo como invento inflaría la tasa de alucinación con
    caídas ajenas.
    """
    url = f"{SJF}/{registro}?isSemanal=false&hostName=https://sjf2.scjn.gob.mx"
    req = urllib.request.Request(url, headers={
        "Referer": f"https://sjf2.scjn.gob.mx/detalle/tesis/{registro}",
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"),
        "Accept": "application/json",
    })
    try:
        d = json.load(urllib.request.urlopen(req, timeout=25))
        return "existe" if d.get("ius") else "no_existe"
    except urllib.error.HTTPError as e:
        return "no_existe" if e.code == 404 else "sin_comprobar"
    except Exception:
        return "sin_comprobar"


def medir_una(p: dict) -> dict:
    t0 = time.time()
    try:
        texto = preguntar(p["texto"], p.get("estado", ""))
    except Exception as e:
        return {"id": p["id"], "error": f"{type(e).__name__}: {e}"}
    segundos = round(time.time() - t0, 1)

    meta = {}
    m = re.search(r"<!-- CITATION_META:(\{.*?\}) -->", texto, re.S)
    if m:
        try:
            meta = json.loads(m.group(1))
        except Exception:
            meta = {}

    # Los registros se buscan en la prosa, no en el JSON del metadato: es lo
    # que el abogado lee y lo que puede estar inventado.
    prosa = re.sub(r"<!-- CITATION_META:\{.*?\} -->", "", texto, flags=re.S)
    prosa = re.sub(r"<!-- PRECEDENTES_META:\[.*?\] -->", "", prosa, flags=re.S)
    registros = sorted(set(PATRON_REGISTRO.findall(prosa)))

    estados = {r: existe_en_semanario(r) for r in registros}

    fila = {
        "id": p["id"],
        "estado": p.get("estado"),
        "materia": p.get("materia"),
        "segundos": segundos,
        "caracteres": len(prosa),
        "citas_trazadas": meta.get("valid", 0),
        "citas_sin_trazar": meta.get("invalid", 0),
        "citas_total": meta.get("total", 0),
        "fuentes_recuperadas": len(meta.get("sources", {}) or {}),
        "registros": registros,
        "registros_existen": sum(1 for v in estados.values() if v == "existe"),
        "registros_inexistentes": [r for r, v in estados.items() if v == "no_existe"],
        "registros_sin_comprobar": [r for r, v in estados.items() if v == "sin_comprobar"],
    }

    esperado = p.get("esperado") or []
    if esperado:
        fuentes = " ".join(
            (s.get("texto", "") + " " + s.get("ref", ""))
            for s in (meta.get("sources", {}) or {}).values()
        ).lower()
        aciertos = [e for e in esperado if e.lower() in fuentes]
        fila["esperado_total"] = len(esperado)
        fila["esperado_encontrado"] = len(aciertos)
        fila["esperado_faltante"] = [e for e in esperado if e not in aciertos]

    return fila


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=0)
    ap.add_argument("--etiqueta", default="")
    args = ap.parse_args()

    banco = json.loads((RAIZ / "preguntas.json").read_text(encoding="utf-8"))
    preguntas = banco["preguntas"]
    if args.limite:
        preguntas = preguntas[:args.limite]

    print(f"Banco de calidad de citas · {len(preguntas)} preguntas · {API}\n")
    filas = []
    for i, p in enumerate(preguntas, 1):
        print(f"[{i}/{len(preguntas)}] {p['id']} … ", end="", flush=True)
        f = medir_una(p)
        filas.append(f)
        if f.get("error"):
            print(f"ERROR {f['error']}")
            continue
        aviso = ""
        if f["citas_sin_trazar"]:
            aviso += f" ⚠️ {f['citas_sin_trazar']} sin trazar"
        if f["registros_inexistentes"]:
            aviso += f" ❌ inexistentes: {','.join(f['registros_inexistentes'])}"
        print(f"{f['segundos']}s · {f['citas_trazadas']} citas · "
              f"{len(f['registros'])} tesis{aviso}")

    ok = [f for f in filas if not f.get("error")]
    if not ok:
        print("\nNinguna consulta completó.")
        return 1

    total_citas = sum(f["citas_total"] for f in ok)
    sin_trazar = sum(f["citas_sin_trazar"] for f in ok)
    total_reg = sum(len(f["registros"]) for f in ok)
    reg_malos = sum(len(f["registros_inexistentes"]) for f in ok)
    reg_dudosos = sum(len(f["registros_sin_comprobar"]) for f in ok)
    resp_con_problema = sum(
        1 for f in ok if f["citas_sin_trazar"] or f["registros_inexistentes"])

    resumen = {
        "cuando": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "etiqueta": args.etiqueta,
        "api": API,
        "consultas": len(ok),
        "citas_total": total_citas,
        "citas_sin_trazar": sin_trazar,
        "tasa_citas_sin_trazar": round(sin_trazar / total_citas * 100, 2) if total_citas else 0.0,
        "registros_citados": total_reg,
        "registros_inexistentes": reg_malos,
        "tasa_registros_inexistentes": round(reg_malos / total_reg * 100, 2) if total_reg else 0.0,
        "registros_sin_comprobar": reg_dudosos,
        "respuestas_con_algun_problema": resp_con_problema,
        "tasa_respuestas_con_problema": round(resp_con_problema / len(ok) * 100, 2),
        "segundos_medios": round(sum(f["segundos"] for f in ok) / len(ok), 1),
    }

    print("\n" + "─" * 62)
    print(f"  Consultas medidas ......... {resumen['consultas']}")
    print(f"  Citas al acervo ........... {total_citas} "
          f"({sin_trazar} sin trazar = {resumen['tasa_citas_sin_trazar']}%)")
    print(f"  Tesis citadas ............. {total_reg} "
          f"({reg_malos} inexistentes = {resumen['tasa_registros_inexistentes']}%"
          + (f", {reg_dudosos} sin comprobar" if reg_dudosos else "") + ")")
    print(f"  Respuestas con problema ... {resp_con_problema}/{len(ok)} "
          f"= {resumen['tasa_respuestas_con_problema']}%")
    print(f"  Tiempo medio .............. {resumen['segundos_medios']}s")
    print("─" * 62)

    destino = RAIZ / "corridas"
    destino.mkdir(exist_ok=True)
    sello = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    nombre = f"{sello}{'-' + args.etiqueta if args.etiqueta else ''}.json"
    (destino / nombre).write_text(
        json.dumps({"resumen": resumen, "filas": filas}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\nGuardado en _calidad/corridas/{nombre}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
