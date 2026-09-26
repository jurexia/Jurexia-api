#!/usr/bin/env python3
"""LA CACHÉ DEL SJF PARA EL ÍNDICE DE VIGENCIA → <dir>/{registro}.json   (26-sep-2026)

POR QUÉ EXISTE
--------------
`jurisprudencia_nacional_v3` guarda `precedentes` cortado a 2,500 caracteres
(ingesta v3_semanario_2026-08), y las notas de vigencia van AL FINAL: 1,370
tesis las perdieron así (la 2024159 perdió la suya: «La presente tesis
abandona… P. IX/2015 y P. X/2015»). scripts/vigencia_tesis_generar.py las lee
completas de esta caché. Sin ella el índice sale peor —555 tesis en vez de
558: 197896, 2016903 y 2026329 vuelven a pasar por vigentes y 23 entradas
cambian—, y por eso el generador ya no corre sin `--sjf-cache` (o sin un
`--sin-sjf` explícito).

La caché se armó el 25-sep-2026 con dos guiones de una sesión
(pilar/sjf.py y pilar/completar_truncadas.py) que vivían en un scratchpad
efímero: en otra máquina, o al día siguiente, el índice no se podía rehacer
igual. Éste es esos dos guiones, juntos y en el repo.

QUÉ HACE (y qué no)
-------------------
  · Sólo GET a la API pública del Semanario (sjf2.scjn.gob.mx), la misma que
    usa su página de detalle. Nada de escribir, nada de modelos ni embeddings.
  · Como mucho UNA petición por segundo (PAUSA = 1.1 s entre peticiones).
    1,351 tesis tardaron ~35 min el 25-sep-2026, sin un solo error.
  · Lo que ya está en la caché no se vuelve a pedir; un error NO se guarda
    (se reintenta en la próxima corrida). Sale con código 1 si hubo errores.
  · Qué pide: por omisión, las tesis de la v3 cuyo `precedentes` mide ≥ 2,500
    caracteres, leídas de Qdrant (sólo lectura, scroll sin vectores) o de
    `--tesis-cache`, el mismo volcado jsonl que usa el generador. Con registros
    como argumentos, sólo ésos (para cotejar una tesis a mano).

LAS CABECERAS (medido el 26-sep-2026): la API responde 403 a un User-Agent
que se identifica como guion («Iurexia-vigencia/1.0»); con el de un navegador
y el Referer de la página de detalle responde. Se mandan las mismas que usó la
caché de la que salió el índice versionado; `--user-agent` las cambia.

USO
---
    .venv/bin/python scripts/sjf_cache_descargar.py \\
        --sjf-cache <dir> [--env ../../../.env] [--tesis-cache tesis.jsonl] [--max N]
    .venv/bin/python scripts/sjf_cache_descargar.py --sjf-cache <dir> 164500 183349

y después:

    .venv/bin/python scripts/vigencia_tesis_generar.py --sjf-cache <dir> …
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
PAUSA = 1.1          # segundos entre peticiones: ≤ 1 por segundo
URL = ("https://sjf2.scjn.gob.mx/services/sjftesismicroservice/api/public/tesis/{registro}"
       "?isSemanal=false&hostName=https://sjf2.scjn.gob.mx")
REFERER = "https://sjf2.scjn.gob.mx/detalle/tesis/{registro}"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")


def _generador():
    """scripts/vigencia_tesis_generar.py, para leer las tesis igual que él
    (`cargar`) y con el mismo tope de corte (`TOPE_QDRANT`)."""
    spec = importlib.util.spec_from_file_location("vigencia_tesis_generar", AQUI / "vigencia_tesis_generar.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Descargador:
    """GET a la API del SJF con caché en disco y una pausa mínima entre
    peticiones. `abrir`, `dormir` y `reloj` se inyectan en las pruebas."""

    def __init__(self, cache: Path, pausa: float = PAUSA, user_agent: str = USER_AGENT,
                 abrir: Optional[Callable] = None, dormir: Callable[[float], None] = time.sleep,
                 reloj: Callable[[], float] = time.monotonic):
        self.cache = Path(cache)
        self.pausa = max(1.0, float(pausa))      # nunca más de una por segundo
        self.user_agent = user_agent
        self.abrir = abrir or urllib.request.urlopen
        self.dormir, self.reloj = dormir, reloj
        self._ultima: Optional[float] = None
        self.pedidas = 0

    def ruta(self, registro: str) -> Path:
        return self.cache / f"{registro}.json"

    def tesis(self, registro) -> Dict:
        """La ficha del SJF de ese registro (de la caché si ya está), o
        {"_error": …} si falló; los errores no se guardan."""
        registro = str(registro).strip()
        if not registro.isdigit():
            return {"_error": f"registro inválido: {registro!r}"}
        p = self.ruta(registro)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass                              # una copia rota se vuelve a pedir
        if self._ultima is not None:
            espera = self.pausa - (self.reloj() - self._ultima)
            if espera > 0:
                self.dormir(espera)
        req = urllib.request.Request(URL.format(registro=registro), method="GET", headers={
            "Referer": REFERER.format(registro=registro),
            "User-Agent": self.user_agent,
            "Accept": "application/json"})
        try:
            with self.abrir(req, timeout=30) as r:
                d = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            d = {"_error": repr(e)[:300]}
        finally:
            self._ultima = self.reloj()
            self.pedidas += 1
        if "_error" not in d:
            self.cache.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        return d


def registros_cortados(tesis: Iterable[dict], tope: int) -> List[str]:
    """Los registros cuyo `precedentes` llegó cortado (≥ tope), sin repetir."""
    return list(dict.fromkeys(str(t.get("registro")) for t in tesis
                              if t.get("registro") and len(t.get("precedentes") or "") >= tope))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("registros", nargs="*", help="registros concretos (si no, las tesis con `precedentes` cortado)")
    ap.add_argument("--sjf-cache", required=True, help="carpeta de la caché: {registro}.json")
    ap.add_argument("--env", default=str(RAIZ / ".env"),
                    help="archivo .env con QDRANT_URL y QDRANT_API_KEY (sólo se leen)")
    ap.add_argument("--tesis-cache", default=None,
                    help="volcado jsonl de la v3 (el de vigencia_tesis_generar.py): se reutiliza si existe")
    ap.add_argument("--max", type=int, default=None, help="pedir como mucho N tesis en esta corrida")
    ap.add_argument("--user-agent", default=USER_AGENT)
    a = ap.parse_args(argv)

    d = Descargador(Path(a.sjf_cache), user_agent=a.user_agent)
    if a.registros:
        regs = [str(r) for r in a.registros]
    else:
        gen = _generador()
        tesis = gen.cargar(Path(a.env), Path(a.tesis_cache) if a.tesis_cache else None)
        regs = registros_cortados(tesis, gen.TOPE_QDRANT)
    faltan = [r for r in regs if not d.ruta(r).exists()]
    if a.max is not None:
        faltan = faltan[:max(0, a.max)]
    print(f"registros {len(regs)} · faltan en la caché {len(faltan)} · ~{len(faltan) * PAUSA / 60:.0f} min", flush=True)
    errores = 0
    for i, r in enumerate(faltan):
        if "_error" in d.tesis(r):
            errores += 1
        if i % 100 == 0:
            print(f"  {i} {r} errores {errores} {time.strftime('%H:%M:%S')}", flush=True)
    print(f"fin: {len(faltan) - errores} bajadas, {errores} con error (se reintentan en la próxima corrida)")
    return 1 if errores else 0


if __name__ == "__main__":
    sys.exit(main())
