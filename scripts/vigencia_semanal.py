#!/usr/bin/env python3
"""LA REGENERACIÓN SEMANAL DEL ÍNDICE DE VIGENCIA.   (26-sep-2026)

POR QUÉ EXISTE
--------------
datos/vigencia_tesis.json se generó una vez (25/26-sep-2026) con las notas que
el Semanario tenía entonces. El SJF publica tesis cada semana y, al hacerlo,
AÑADE la nota de pérdida a la tesis vieja («Esta tesis fue abandonada por…»):
la 2019978 y la 2029850 se abandonaron el 12-ago-2026. Un índice que no se
rehace envejece en silencio y el chat vuelve a dar por vigente lo abandonado.
No puede rehacerse en Render ni en Vercel —Incapsula bloquea sus IP—; desde la
Mac de David el SJF contesta. Por eso corre aquí, el domingo de madrugada,
lanzado por launchd: scripts/launchd/com.iurexia.vigencia-semanal.plist →
scripts/vigencia_semanal.sh (candado, worktree dedicado en origin/main) → esto.

QUÉ HACE, EN ORDEN
------------------
 1. Vuelca la v3 de Qdrant (sólo lectura, sin vectores) a <dir>/tesis_v3.jsonl.
 2. Pone al día la caché del SJF (<dir>/sjf_cache) sin bajar 71 mil tesis:
    a. TESIS NUEVAS: desde el último registro hallado (o el mayor de la caché y
       del acervo), registro a registro hasta 60 seguidos sin ficha. Luego unos
       saltos largos (+100 … +5,000) por si la numeración brincó. Los huecos
       (sin ficha entre dos que sí) se vuelven a mirar durante 8 semanas. Cada
       registro se pide a la Gaceta y, si no lo tiene, al Semanario semanal
       (`isSemanal=true`): la Gaceta va semanas atrás —el 26-sep-2026 acababa en
       la 2032443, del 10 de julio— y lo publicado después sólo está en el
       Semanario. Sin eso, la «última» tesis habría sido la del 10 de julio.
    b. SUS AFECTADAS: se pasan las reglas del generador a las nuevas («[ABANDONO
       …]», «interrumpe…», «Tesis sustituida», «ya no se considera de aplicación
       obligatoria la diversa…»); las claves se resuelven con el índice de claves
       del acervo que arma el propio generador (normalizado: una clave citada
       rara vez es idéntica, byte a byte, al campo `clave_tesis` de Qdrant) y cada
       afectada se REBAJA del SJF, que es donde aparece su nota nueva.
    c. REPASO ROTATIVO acotado (1,500 por semana): 300 del carril «caliente» —las
       del índice, sus reemplazos y la capa curada; dan la vuelta en ~3 semanas— y
       el resto del carril general: jurisprudencias primero y aisladas después,
       de la más nueva a la más vieja (la vuelta entera, ~1 año). Atrapa las notas
       que se añaden a tesis viejas sin que ninguna tesis nueva lo anuncie.
    Todo con ≥ 1 s entre peticiones, reintentos acotados y sin guardar errores
    como si fueran tesis. El cursor vive en <dir>/estado.json.
 3. Regenera con scripts/vigencia_tesis_generar.py (en él, el SJF manda y las
    tesis nuevas cuentan).
 4. Compara POR REGISTRO con datos/vigencia_tesis.json. Sin cambios: termina,
    sin tocar el archivo.
 5. Cordura: no pueden irse más del 2 % de las entradas ni cambiar más de 50 de
    golpe —se detiene (código 3), avisa y deja el diff en <dir>/ultimo_diff.json—;
    el JSON tiene que cargar con vigencia_tesis.py y pasar test_vigencia_tesis.py
    y test_vigencia_semanal.py.
 6. Commit («vigencia semanal: +N −M entradas; …») y push a main por
    fast-forward; si main avanzó, rebase, pruebas otra vez y un reintento.
    Nunca --force, y sólo desde el worktree dedicado, parado en origin/main.

SI EL SJF NOS BLOQUEA: un registro que no existe NO da 404 —da 200 con la
página de Incapsula, la misma del bloqueo—. Antes de dar por buena una racha
sin fichas se pide de nuevo una tesis que sí existe (el «testigo»); si tampoco
llega, la corrida para con código 4 y no escribe nada del índice.

EL PRIMER ENSAYO, EN SECO (26-sep-2026): el acervo acababa en la 2032214
(29-may-2026). La Gaceta dio 229 tesis nuevas (2032215–2032443, hasta el
10-jul) y el Semanario semanal otras 244 (2032444–2032687, del 7-ago al
25-sep), sin huecos. El repaso de 1,500 halló 6 tesis con notas añadidas
después de la ingesta; una, la 2030612 (XXI.2o.C.T. J/1 L (11a.)), interrumpida
por la 2032611, obligatoria desde el 7-sep-2026, que el índice no tenía. 2,178
peticiones, cero errores; ~49 min la corrida completa, ~9 la de sólo nuevas.

CÓDIGOS DE SALIDA: 0 bien (con o sin commit) · 1 fallo · 3 la cordura detuvo
· 4 el SJF no contesta · 5 git (commit, rebase o push) · 6 las pruebas fallan
· 75 ya hay otra corrida.

USO
---
    # lo que corre launchd (vía scripts/vigencia_semanal.sh):
    .venv/bin/python scripts/vigencia_semanal.py --worktree ../wt-vigencia-semanal --env ../jurexia-api-git/.env
    # ensayo en seco (todo menos commit y push; la caché y el estado SÍ se ponen al día):
    .venv/bin/python scripts/vigencia_semanal.py --seco --env ../../../.env
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import fcntl
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
MAC = Path.home() / "Documents" / "IUREXIA-MAC"
DIR_VIGENCIA = MAC / "reingesta" / "vigencia"
WT_DEDICADO = MAC / "wt-vigencia-semanal"
ENV = MAC / "jurexia-api-git" / ".env"
LOG = Path.home() / "Library" / "Logs" / "iurexia" / "vigencia-semanal.log"
PRUEBAS = ("test_vigencia_tesis.py", "test_vigencia_semanal.py")

FALLOS_SEGUIDOS = 60          # registros seguidos sin ficha para dar por terminada la numeración
SALTOS = (100, 250, 500, 1000, 2500, 5000)
TOPE_NUEVAS = 3000            # peticiones para buscar tesis nuevas en una corrida
TOPE_AFECTADAS = 300
REPASO = 1500
REPASO_CALIENTES = 300
SEMANAS_HUECO = 8
TOPE_HUECOS = 200
ERRORES_SEGUIDOS = 8          # errores seguidos antes de preguntar al testigo si nos bloquearon
UMBRAL_BAJAS = 0.02
UMBRAL_CAMBIOS = 50
CO_AUTOR = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
SIN_FICHA = ("no_existe", "incapsula")
# Lo que el generador anota en sin_resolver y NO es una afectada que rebajar.
NO_AFECTADA = {"autorreferencia", "voto_no_perdida", "norma_reformada_no_perdida"}


def _modulo(nombre: str, ruta: Path):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _modulo("vigencia_tesis_generar", AQUI / "vigencia_tesis_generar.py")
dl = _modulo("sjf_cache_descargar", AQUI / "sjf_cache_descargar.py")


class Bloqueado(RuntimeError):
    """El SJF no da ni una tesis que sabemos que existe (Incapsula, red, caída)."""


class Detenido(RuntimeError):
    """La cordura detuvo la corrida: el índice nuevo no se escribe."""


class FalloGit(RuntimeError):
    pass


class FalloPruebas(RuntimeError):
    pass


class Ocupado(RuntimeError):
    pass


# ═══════════════════════════════════════════════════════════════ registro (log)
_LOG_ARCHIVO: Optional[Path] = None


def log(msg: str = "") -> None:
    linea = f"[{_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}" if msg else ""
    print(linea, flush=True)
    if _LOG_ARCHIVO:
        try:
            _LOG_ARCHIVO.parent.mkdir(parents=True, exist_ok=True)
            with _LOG_ARCHIVO.open("a", encoding="utf-8") as fh:
                fh.write(linea + "\n")
        except Exception:
            pass


@contextlib.contextmanager
def candado(ruta: Path):
    """Que no corran dos a la vez sobre la misma caché y el mismo estado. Es un
    flock: si el proceso muere, el sistema lo suelta solo."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fh = open(ruta, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        raise Ocupado(f"otra corrida tiene el candado {ruta}")
    try:
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()} {_dt.datetime.now().isoformat(timespec='seconds')}\n")
        fh.flush()
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


# ═══════════════════════════════════════════════════════════════ estado
def estado_vacio() -> dict:
    return {"version": 1,
            "ultimo_secuencial": None,     # la última tesis hallada al recorrer registro a registro
            "afectadas_hasta": None,       # las nuevas hasta aquí ya tienen sus afectadas rebajadas
            "huecos": {},                  # registro → primer día sin ficha (entre dos que sí)
            "pendientes": [],              # refrescos que fallaron: se reintentan la próxima vez
            "repaso": {"calientes": {"vuelta": 1, "hechos": []}, "general": {"vuelta": 1, "hechos": []}},
            "ultima_corrida": None}


def leer_estado(ruta: Path) -> dict:
    """El estado, o uno vacío si no hay. Si está roto se lanza: empezar de cero
    a ciegas volvería a recorrer la numeración y el repaso desde el principio."""
    e = estado_vacio()
    if ruta.exists():
        e.update(json.loads(ruta.read_text(encoding="utf-8")))
    e.setdefault("repaso", {})
    for k in ("calientes", "general"):
        e["repaso"].setdefault(k, {"vuelta": 1, "hechos": []})
    return e


def guardar_estado(ruta: Path, e: dict) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_name(ruta.name + ".tmp")
    tmp.write_text(json.dumps(e, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(ruta)


# ═══════════════════════════════════════════════════════════════ la caché del SJF
def clase(d: dict) -> str:
    return "ok" if "_error" not in d else (d.get("_tipo") or "error")


def registros_en_cache(cache: Path) -> Set[int]:
    if not cache.is_dir():
        return set()
    return {int(n[:-5]) for n in os.listdir(cache) if re.fullmatch(r"\d+\.json", n)}


def sjf_contesta(D, testigos: Sequence) -> bool:
    """¿Da el SJF una tesis que sabemos que existe? Se pide de nuevo, sin caché."""
    for r in list(dict.fromkeys(str(x) for x in testigos if x))[:3]:
        if clase(D.tesis(r, refrescar=True)) == "ok":
            return True
    return False


def _tesis(D, r, semanal_primero: bool) -> dict:
    return D.tesis(r, semanal_primero=semanal_primero)


def buscar_nuevas(D, desde: int, conocido_max: int, testigos: Sequence, fallos: int = FALLOS_SEGUIDOS,
                  saltos: Sequence[int] = SALTOS, tope: int = TOPE_NUEVAS) -> dict:
    """Registro a registro desde `desde` hasta `fallos` seguidos sin ficha (y,
    en todo caso, hasta pasar `conocido_max`). Al cerrarse la racha: el testigo
    (¿nos bloquearon?) y unos saltos largos (¿brincó la numeración?); si un
    salto da ficha, se sigue registro a registro hasta más allá de él."""
    r, ultimo = desde, None
    faltan: List[int] = []
    halladas: List[int] = []
    huecos: List[int] = []
    errores: List[int] = []
    saltos_hallados: List[int] = []
    seguidos = err_seguidos = 0
    # Más allá de la Gaceta todo sale del Semanario semanal: tras una ficha de
    # ahí, se le pregunta primero a él (media petición menos por tesis nueva).
    pref_semanal = False
    forzar_hasta = conocido_max
    p0 = D.pedidas
    cortado = False
    while True:
        if D.pedidas - p0 >= tope:
            cortado = True
            break
        d = _tesis(D, r, pref_semanal)
        c = clase(d)
        if c == "ok":
            pref_semanal = d.get("semanal") == 1
            halladas.append(r)
            huecos.extend(faltan)
            faltan, ultimo, seguidos, err_seguidos = [], r, 0, 0
        else:
            faltan.append(r)
            seguidos += 1
            if c in SIN_FICHA:
                err_seguidos = 0
            else:
                errores.append(r)
                err_seguidos += 1
                if err_seguidos >= ERRORES_SEGUIDOS:
                    if not sjf_contesta(D, testigos):
                        raise Bloqueado(f"{err_seguidos} errores seguidos buscando tesis nuevas (hasta el {r}) "
                                        f"y el testigo tampoco llega")
                    err_seguidos = 0
        r += 1
        if seguidos >= fallos and r > forzar_hasta:
            if not sjf_contesta(D, testigos):
                raise Bloqueado(f"{fallos} registros seguidos sin ficha desde el {r - fallos} y el testigo "
                                f"tampoco llega: puede ser un bloqueo, no el final de la numeración")
            base = ultimo if ultimo is not None else desde - 1
            hit = None
            for s in saltos:
                x = base + s
                if x < r:
                    continue
                if clase(_tesis(D, x, pref_semanal)) == "ok":
                    hit = x
                    break
            if hit is None:
                break
            log(f"   ⚠️ la numeración brincó: hay ficha en el {hit} (la última seguida era el {base}); "
                f"se recorre el tramo")
            saltos_hallados.append(hit)
            forzar_hasta = hit
    return {"halladas": halladas, "huecos": huecos, "errores": errores, "ultimo": ultimo,
            "saltos": saltos_hallados, "cortado": cortado, "hasta": r - 1, "peticiones": D.pedidas - p0}


def repasar_huecos(D, huecos: Dict[str, str], hoy: _dt.date, semanas: int = SEMANAS_HUECO,
                   tope: int = TOPE_HUECOS) -> Tuple[List[int], Dict[str, str]]:
    """Los huecos se vuelven a pedir unas semanas (una tesis puede publicarse
    con un registro que se reservó antes). -> (halladas, huecos que siguen)."""
    halladas: List[int] = []
    siguen: Dict[str, str] = {}
    pedidos = 0
    for r, visto in sorted(huecos.items(), key=lambda kv: int(kv[0])):
        try:
            edad = (hoy - _dt.date.fromisoformat(visto)).days
        except ValueError:
            edad = 0
        if edad > semanas * 7:
            continue                                   # se da por perdido
        if edad < 1 or pedidos >= tope:
            siguen[r] = visto                          # el de hoy ya se pidió hoy
            continue
        pedidos += 1
        if clase(D.tesis(r)) == "ok":
            halladas.append(int(r))
        else:
            siguen[r] = visto
    return halladas, siguen


def afectadas_de(tesis: List[dict], nuevas: List[dict], a_procesar: Iterable) -> Tuple[List[str], List[dict]]:
    """Las tesis del acervo que las nuevas declaran perdidas, con las reglas del
    generador. Además de las que resolvió, las claves que no pudo resolver
    (ambiguas, de otra época, rubro distinto) aportan todos sus candidatos del
    acervo: rebajar cinco de más cuesta cinco segundos; no rebajar la buena, una
    semana con la nota vieja. -> (registros, detalle)."""
    A = gen.Acervo(tesis, nuevas)
    E = gen.Extractor(A)
    quiero = {str(r) for r in a_procesar}
    for t in nuevas:
        if str(t["registro"]) in quiero and str(t["registro"]) not in A.en_acervo:
            E.procesar(t)
    regs: List[str] = []
    detalle: List[dict] = []

    def poner(r, de, como):
        r = str(r)
        if r in A.en_acervo and r not in regs:
            regs.append(r)
            detalle.append({"afectada": r, "segun": de, "via": como})

    for rel in E.rel:
        poner(rel["afectado"], (rel.get("por") or {}).get("registro") or rel.get("citada_en"), rel.get("patron"))
    for s in E.sin_resolver:
        if s.get("motivo") in NO_AFECTADA:
            continue
        de = s.get("por_registro") or s.get("citada_en")
        if s.get("afectada_registro"):
            poner(s["afectada_registro"], de, s.get("motivo"))
        if s.get("afectada_clave"):
            b, _ = gen.norm_clave(s["afectada_clave"])
            for _, r in A.por_base.get(b, [])[:5]:
                poner(r, de, f"clave_{s.get('motivo')}")
    return regs, detalle


def refrescar(D, registros: Sequence[str], previo: Callable[[str], Tuple[Optional[str], bool]],
              testigos: Sequence) -> dict:
    """Rebaja del SJF cada registro (aunque ya esté en la caché). Un fallo deja
    la copia anterior; 8 seguidos consultan al testigo y, si tampoco llega, se
    para con Bloqueado. -> {"ok", "errores", "cambiadas"}: «cambiada» es que su
    `precedentes` ya no es el que teníamos (de la caché o, si no, de Qdrant)."""
    ok: List[str] = []
    errores: List[str] = []
    cambiadas: List[str] = []
    seguidos = 0
    for r in registros:
        antes, cortado = previo(r)
        d = D.tesis(r, refrescar=True)
        if clase(d) == "ok":
            ok.append(r)
            seguidos = 0
            ahora = gen.texto_sjf(d.get("precedentes"))
            if antes is not None and ((not ahora.startswith(antes)) if cortado else ahora != antes):
                cambiadas.append(r)
        else:
            errores.append(r)
            seguidos += 1
            if seguidos >= ERRORES_SEGUIDOS:
                if not sjf_contesta(D, testigos):
                    raise Bloqueado(f"{seguidos} errores seguidos al rebajar (el último, {r}) y el testigo "
                                    f"tampoco llega")
                seguidos = 0
    return {"ok": ok, "errores": errores, "cambiadas": cambiadas}


def calientes(indice: dict, curada: dict, en_acervo: Set[str]) -> List[str]:
    """Las del índice y de la capa curada, con sus reemplazos: de la más nueva a la más vieja."""
    regs: Set[str] = set()
    for fuente in (indice, curada):
        tesis = fuente.get("tesis") if isinstance(fuente.get("tesis"), dict) else {}
        for r, v in tesis.items():
            regs.add(str(r))
            if isinstance(v, dict) and v.get("por_registro"):
                regs.add(str(v["por_registro"]))
    return sorted((r for r in regs if r in en_acervo), key=lambda r: -(gen.orden_registro(r) or 0))


def orden_general(tesis: List[dict]) -> List[str]:
    """Jurisprudencias primero y aisladas después; en cada grupo, de la más nueva a la más vieja."""
    def clave(t):
        r = str(t.get("registro"))
        es_j = str(t.get("tipo") or "").upper().startswith("JURISPRUDENCIA")
        return (0 if es_j else 1, -(gen.orden_registro(r) or 0))
    return [str(t.get("registro")) for t in sorted(tesis, key=clave)]


def elegir(orden: Sequence[str], carril: dict, n: int, excluir: Iterable[str] = ()) -> List[str]:
    """Los siguientes `n` del carril que no se han repasado en esta vuelta. Si la
    vuelta se acaba, empieza otra (y se completa con el principio del orden)."""
    if n <= 0 or not orden:
        return []
    fuera = set(excluir)
    hechos = set(carril.get("hechos") or [])
    toma = [r for r in orden if r not in hechos and r not in fuera][:n]
    if len(toma) < n:
        carril["vuelta"] = int(carril.get("vuelta") or 1) + 1
        carril["hechos"] = []
        ya = set(toma)
        toma += [r for r in orden if r not in ya and r not in fuera][:n - len(toma)]
    return toma


def actualizar_cache(D, tesis: List[dict], estado: dict, cache: Path, indice: dict, curada: dict,
                     hoy: _dt.date, guardar: Callable[[dict], None] = lambda e: None,
                     fallos: int = FALLOS_SEGUIDOS, saltos: Sequence[int] = SALTOS, tope_nuevas: int = TOPE_NUEVAS,
                     tope_afectadas: int = TOPE_AFECTADAS, repaso: int = REPASO,
                     repaso_calientes: int = REPASO_CALIENTES, esperas_red: Sequence[float] = (60, 120),
                     aviso: Callable[[str], None] = lambda s: None) -> dict:
    """Los pasos 2a-2c. Guarda el estado al cerrar cada paso: si algo truena a
    media corrida, la próxima no repite lo ya hecho. -> resumen."""
    gen.SJF_CACHE = str(cache)
    en_acervo = {str(t.get("registro")) for t in tesis}
    max_acervo = max((int(r) for r in en_acervo if r.isdigit()), default=0)
    antes = registros_en_cache(cache)
    conocido_max = max(antes | {max_acervo})
    res: dict = {"peticiones_0": D.pedidas}
    semanal_0 = getattr(D, "del_semanal", 0)
    qdrant = {str(t.get("registro")): t.get("precedentes") or "" for t in tesis}

    def previo(r: str) -> Tuple[Optional[str], bool]:
        f = gen.ficha_sjf(r)
        if f and isinstance(f.get("precedentes"), str):
            return gen.texto_sjf(f["precedentes"]), False
        q = qdrant.get(r)
        return (q, len(q) >= gen.TOPE_QDRANT) if q is not None else (None, False)

    # 0. ¿contesta el SJF? Si no, nada: ni una petición más.
    testigos = [str(estado.get("ultimo_secuencial") or ""), str(max_acervo), "2024159"]
    for espera in tuple(esperas_red) + (None,):
        if sjf_contesta(D, testigos):
            break
        if espera is None:
            raise Bloqueado("el SJF no da ni la tesis testigo: Incapsula, la red o el propio SJF")
        D.dormir(espera)                        # al despertar, la red tarda en volver
    hechos_hoy: Set[str] = set()

    # 1. lo que falló la vez pasada
    pend = [str(r) for r in estado.get("pendientes") or []]
    rp = refrescar(D, pend, previo, testigos) if pend else {"ok": [], "errores": [], "cambiadas": []}
    estado["pendientes"] = rp["errores"]
    hechos_hoy.update(rp["ok"])
    res["pendientes"] = {"pedidas": len(pend), "ok": len(rp["ok"]), "cambiadas": rp["cambiadas"]}
    guardar(estado)

    # 2a. tesis nuevas
    frontera = estado.get("ultimo_secuencial")
    frontera = max(int(frontera) if frontera else conocido_max, max_acervo)
    bn = buscar_nuevas(D, frontera + 1, conocido_max, testigos, fallos=fallos, saltos=saltos, tope=tope_nuevas)
    if bn["ultimo"] is not None:
        estado["ultimo_secuencial"] = bn["ultimo"]
    else:
        estado["ultimo_secuencial"] = frontera
    huecos = dict(estado.get("huecos") or {})
    for h in bn["huecos"]:
        huecos.setdefault(str(h), hoy.isoformat())
    hh, huecos = repasar_huecos(D, huecos, hoy)
    estado["huecos"] = huecos
    bajadas = sorted(set(bn["halladas"] + hh) - antes)
    res["nuevas"] = {"desde": frontera + 1, "hasta": bn["hasta"], "halladas": len(bn["halladas"]) + len(hh),
                     "bajadas": len(bajadas), "primera": bajadas[0] if bajadas else None,
                     "ultima": bajadas[-1] if bajadas else None, "huecos_nuevos": len(bn["huecos"]),
                     "huecos_llenados": hh, "huecos_vigentes": len(huecos), "errores": bn["errores"],
                     "saltos": bn["saltos"], "cortado": bn["cortado"],
                     "del_semanal": getattr(D, "del_semanal", 0) - semanal_0}
    guardar(estado)
    n = res["nuevas"]
    aviso(f"   nuevas: {n['bajadas']} bajadas del {n['desde']} al {n['hasta']} ({n['primera']}–{n['ultima']}); "
          f"{n['del_semanal']} sólo en el Semanario semanal (aún no en la Gaceta); "
          f"huecos nuevos {n['huecos_nuevos']}, llenados {len(hh)}, vigentes {len(huecos)}; saltos {n['saltos']}; "
          f"errores {len(n['errores'])}{' · CORTADO por el tope' if n['cortado'] else ''}")

    # 2b. las afectadas de las nuevas
    nuevas = gen.cargar_nuevas_sjf(str(cache), en_acervo)
    desde_af = estado.get("afectadas_hasta")
    desde_af = int(desde_af) if desde_af else max_acervo
    a_procesar = [t["registro"] for t in nuevas if int(t["registro"]) > desde_af]
    af, detalle = afectadas_de(tesis, nuevas, a_procesar) if a_procesar else ([], [])
    af = [r for r in af if r not in hechos_hoy]
    sobra = af[tope_afectadas:]
    ra = refrescar(D, af[:tope_afectadas], previo, testigos)
    hechos_hoy.update(ra["ok"])
    estado["pendientes"] = list(dict.fromkeys(estado["pendientes"] + ra["errores"] + sobra))
    if nuevas:
        estado["afectadas_hasta"] = max(desde_af, max(int(t["registro"]) for t in nuevas))
    res["afectadas"] = {"nuevas_leidas": len(a_procesar), "afectadas": len(af), "rebajadas": len(ra["ok"]),
                        "cambiadas": ra["cambiadas"], "errores": ra["errores"], "aplazadas": len(sobra),
                        "detalle": detalle[:50]}
    guardar(estado)
    aviso(f"   afectadas: {len(af)} de {len(a_procesar)} nuevas; rebajadas {len(ra['ok'])}, con nota distinta "
          f"{ra['cambiadas']}, errores {len(ra['errores'])}, aplazadas {len(sobra)}")

    # 2c. el repaso rotativo
    cal = calientes(indice, curada, en_acervo)
    cal_set = set(cal)
    carril_c, carril_g = estado["repaso"]["calientes"], estado["repaso"]["general"]
    toma_c = elegir(cal, carril_c, min(repaso_calientes, repaso), excluir=hechos_hoy)
    toma_g = elegir([r for r in orden_general(tesis) if r not in cal_set], carril_g,
                    repaso - len(toma_c), excluir=hechos_hoy | set(toma_c))
    rr = {"ok": [], "errores": [], "cambiadas": []}
    for carril, toma in ((carril_c, toma_c), (carril_g, toma_g)):
        r1 = refrescar(D, toma, previo, testigos)
        carril["hechos"] = list(dict.fromkeys(list(carril.get("hechos") or []) + r1["ok"]))
        for k in rr:
            rr[k] += r1[k]
        guardar(estado)
    res["repaso"] = {"calientes": len(toma_c), "general": len(toma_g), "ok": len(rr["ok"]),
                     "cambiadas": rr["cambiadas"], "errores": rr["errores"],
                     "vuelta_calientes": carril_c["vuelta"], "hechos_calientes": len(carril_c["hechos"]),
                     "total_calientes": len(cal), "vuelta_general": carril_g["vuelta"],
                     "hechos_general": len(carril_g["hechos"]), "total_general": len(tesis) - len(cal)}
    res["peticiones"] = D.pedidas - res.pop("peticiones_0")
    rp = res["repaso"]
    aviso(f"   repaso: {rp['calientes']} calientes + {rp['general']} generales; distintas {rp['cambiadas'][:30]}; "
          f"errores {len(rp['errores'])}; vuelta caliente {rp['vuelta_calientes']} "
          f"({rp['hechos_calientes']:,}/{rp['total_calientes']:,}), general {rp['vuelta_general']} "
          f"({rp['hechos_general']:,}/{rp['total_general']:,}); {res['peticiones']:,} peticiones")
    return res


# ═══════════════════════════════════════════════════════════════ el índice: diff y cordura
def comparar(viejo: Dict[str, dict], nuevo: Dict[str, dict]) -> dict:
    altas = sorted(set(nuevo) - set(viejo), key=int)
    bajas = sorted(set(viejo) - set(nuevo), key=int)
    cambios = sorted((r for r in set(viejo) & set(nuevo) if viejo[r] != nuevo[r]), key=int)
    campos = {r: sorted(k for k in set(viejo[r]) | set(nuevo[r]) if viejo[r].get(k) != nuevo[r].get(k))
              for r in cambios}
    return {"altas": altas, "bajas": bajas, "cambios": cambios, "campos": campos,
            "total": len(altas) + len(bajas) + len(cambios)}


def cordura(n_viejo: int, diff: dict, umbral_bajas: float = UMBRAL_BAJAS,
            umbral_cambios: int = UMBRAL_CAMBIOS) -> List[str]:
    """Lo que un índice nuevo no puede hacer de golpe sin que lo mire una persona."""
    problemas = []
    tope_bajas = int(n_viejo * umbral_bajas)
    if len(diff["bajas"]) > tope_bajas:
        problemas.append(f"desaparecen {len(diff['bajas'])} de {n_viejo} entradas "
                         f"(más del {umbral_bajas:.0%}: tope {tope_bajas})")
    if diff["total"] > umbral_cambios:
        problemas.append(f"cambian {diff['total']} entradas de golpe (+{len(diff['altas'])} "
                         f"−{len(diff['bajas'])} ~{len(diff['cambios'])}; tope {umbral_cambios})")
    return problemas


def carga_con_vigencia_tesis(raiz: Path, ruta: Path) -> int:
    """Cuántas entradas lee vigencia_tesis.py de ese archivo (0 si no lo lee)."""
    vt = _modulo("vigencia_tesis_semanal", raiz / "vigencia_tesis.py")
    return len(vt._leer(ruta, "índice semanal"))


def _etiqueta(raiz: Path) -> Callable[[dict], str]:
    try:
        vt = _modulo("vigencia_tesis_etiqueta", raiz / "vigencia_tesis.py")
        return vt.etiqueta
    except Exception:
        return lambda v: str(v.get("estado"))


def mensaje_commit(diff: dict, viejo: Dict[str, dict], nuevo: Dict[str, dict], res: Optional[dict],
                   raiz: Path) -> str:
    et = _etiqueta(raiz)
    n = res.get("nuevas", {}) if res else {}
    a = res.get("afectadas", {}) if res else {}
    rp = res.get("repaso", {}) if res else {}
    cab = f"vigencia semanal: +{len(diff['altas'])} −{len(diff['bajas'])} entradas; {len(diff['cambios'])} cambiadas"
    lineas = [cab, ""]
    if res:
        por_que = []
        if n.get("bajadas"):
            por_que.append(f"el SJF publicó {n['bajadas']} tesis nuevas ({n.get('primera')}–{n.get('ultima')})")
        cambiadas = len(a.get("cambiadas") or []) + len(rp.get("cambiadas") or [])
        if cambiadas:
            por_que.append(f"{cambiadas} tesis ya tenían en el SJF un `precedentes` distinto del que teníamos "
                           f"(notas añadidas después de la ingesta)")
        lineas.append("Por qué: " + ("; ".join(por_que) if por_que else "cambió lo que el Semanario dice de ellas")
                      + ". El índice se rehízo con esas notas; el SJF manda sobre Qdrant.")
        lineas.append("")
    if diff["altas"]:
        lineas.append("Altas (perdieron vigencia según el Semanario):")
        for r in diff["altas"]:
            v = nuevo[r]
            lineas.append(f"  + {r}: {et(v)} [{v.get('fuente')}]")
        lineas.append("")
    if diff["bajas"]:
        lineas.append("Bajas (el índice ya no las marca):")
        for r in diff["bajas"]:
            lineas.append(f"  − {r}: era «{et(viejo[r])}»")
        lineas.append("")
    if diff["cambios"]:
        lineas.append("Cambios:")
        for r in diff["cambios"]:
            lineas.append(f"  ~ {r} ({', '.join(diff['campos'][r])}): {et(nuevo[r])}")
        lineas.append("")
    if res:
        lineas.append(f"Caché del SJF: {n.get('bajadas', 0)} tesis nuevas bajadas, {a.get('rebajadas', 0)} afectadas "
                      f"rebajadas, {rp.get('ok', 0)} del repaso rotativo; {res.get('peticiones', 0)} peticiones.")
    lineas.append(f"Cordura: {len(diff['bajas'])} bajas (tope 2 %) y {diff['total']} cambios (tope 50); "
                  f"vigencia_tesis.py lo carga; test_vigencia_tesis.py y test_vigencia_semanal.py pasan.")
    lineas.append("Lo generó scripts/vigencia_semanal.py (desatendido, en la Mac).")
    lineas += ["", CO_AUTOR]
    return "\n".join(lineas) + "\n"


# ═══════════════════════════════════════════════════════════════ git
def git(raiz: Path, *args: str, check: bool = False) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", "-C", str(raiz), *args], capture_output=True, text=True, timeout=600)
    if check and r.returncode != 0:
        raise FalloGit(f"git {' '.join(args)}: {(r.stderr or r.stdout).strip()[:400]}")
    return r


def comprobar_worktree(raiz: Path, esperado: Path) -> None:
    """Commit y push SÓLO desde el worktree dedicado, en origin/main y con el
    índice limpio: nunca desde el checkout principal ni desde el de otra sesión."""
    top = Path(git(raiz, "rev-parse", "--show-toplevel", check=True).stdout.strip()).resolve()
    gd = Path(git(raiz, "rev-parse", "--absolute-git-dir", check=True).stdout.strip()).resolve()
    gcd = git(raiz, "rev-parse", "--git-common-dir", check=True).stdout.strip()
    gcd = (Path(gcd) if os.path.isabs(gcd) else (raiz / gcd)).resolve()
    if top != Path(esperado).resolve():
        raise FalloGit(f"no es el worktree dedicado: {top} (se esperaba {esperado})")
    if gd == gcd:
        raise FalloGit(f"{top} es el checkout principal, no un worktree")
    head = git(raiz, "rev-parse", "HEAD", check=True).stdout.strip()
    om = git(raiz, "rev-parse", "origin/main", check=True).stdout.strip()
    if head != om:
        raise FalloGit(f"el worktree no está en origin/main (HEAD {head[:9]}, origin/main {om[:9]})")
    if git(raiz, "status", "--porcelain", "--", "datos/vigencia_tesis.json", check=True).stdout.strip():
        raise FalloGit("datos/vigencia_tesis.json ya tenía cambios antes de empezar")


def correr_pruebas(raiz: Path, pruebas: Sequence[str]) -> None:
    for p in pruebas:
        t0 = time.time()
        r = subprocess.run([sys.executable, p], cwd=str(raiz), capture_output=True, text=True, timeout=900,
                           env={**os.environ, "VIGENCIA_SEMANAL_DENTRO": "1"})
        if r.returncode != 0:
            fallas = [l for l in (r.stdout + r.stderr).splitlines() if "FALLA" in l or "Error" in l][:12]
            raise FalloPruebas(f"{p} falla (rc={r.returncode}): " + " | ".join(fallas))
        log(f"   ✓ {p} pasa ({time.time() - t0:.0f} s)")


def empujar(raiz: Path, reprobar: Callable[[], None]) -> str:
    """Push por fast-forward. Si main avanzó: rebase, pruebas otra vez y UN reintento. Nunca --force."""
    r = git(raiz, "push", "-q", "origin", "HEAD:main")
    if r.returncode == 0:
        return "push ok"
    log(f"   push rechazado ({(r.stderr or '').strip()[:200]}); rebase sobre origin/main y otro intento")
    git(raiz, "fetch", "-q", "origin", check=True)
    rb = git(raiz, "rebase", "-q", "origin/main")
    if rb.returncode != 0:
        git(raiz, "rebase", "--abort")
        raise FalloGit(f"el rebase choca: {(rb.stderr or rb.stdout).strip()[:300]}")
    reprobar()
    r = git(raiz, "push", "-q", "origin", "HEAD:main")
    if r.returncode != 0:
        raise FalloGit(f"el push falló otra vez: {(r.stderr or '').strip()[:300]}")
    return "push ok tras rebase"


# ═══════════════════════════════════════════════════════════════ la corrida
def resumen_linea(res: Optional[dict], diff: Optional[dict], final: str, t0: float) -> str:
    partes = []
    if res:
        n, a, rp = res.get("nuevas", {}), res.get("afectadas", {}), res.get("repaso", {})
        rango = f" ({n.get('primera')}–{n.get('ultima')})" if n.get("bajadas") else ""
        partes.append(f"nuevas bajadas {n.get('bajadas', 0)}{rango}, huecos {n.get('huecos_vigentes', 0)}")
        partes.append(f"afectadas {a.get('afectadas', 0)} (rebajadas {a.get('rebajadas', 0)}, "
                      f"con nota distinta {len(a.get('cambiadas') or [])})")
        partes.append(f"repaso {rp.get('ok', 0)} (distintas {len(rp.get('cambiadas') or [])})")
        errores = len(n.get("errores") or []) + len(a.get("errores") or []) + len(rp.get("errores") or [])
        partes.append(f"errores {errores}, peticiones {res.get('peticiones', 0)}")
    if diff:
        partes.append(f"índice +{len(diff['altas'])} −{len(diff['bajas'])} ~{len(diff['cambios'])}")
    partes.append(final)
    partes.append(f"{(time.time() - t0) / 60:.1f} min")
    return "RESUMEN · " + " · ".join(partes)


def principal(argv: Optional[List[str]] = None) -> int:
    global _LOG_ARCHIVO
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default=str(DIR_VIGENCIA), help="caché del SJF, estado, volcado e informes")
    ap.add_argument("--env", default=str(ENV), help=".env con QDRANT_URL y QDRANT_API_KEY (sólo se leen)")
    ap.add_argument("--indice", default=None, help="el índice a comparar y reemplazar (datos/vigencia_tesis.json)")
    ap.add_argument("--worktree", default=str(WT_DEDICADO), help="el único checkout desde el que se hace commit")
    ap.add_argument("--seco", action="store_true", help="todo menos commit y push")
    ap.add_argument("--sin-actualizar", action="store_true", help="no pedir nada al SJF: regenerar con la caché")
    ap.add_argument("--reusar-volcado", action="store_true", help="no volver a leer Qdrant si ya hay volcado")
    ap.add_argument("--forzar", action="store_true", help="saltarse los topes de cordura (a mano, tras revisar)")
    ap.add_argument("--repaso", type=int, default=REPASO)
    ap.add_argument("--repaso-calientes", type=int, default=REPASO_CALIENTES)
    ap.add_argument("--fallos-seguidos", type=int, default=FALLOS_SEGUIDOS)
    ap.add_argument("--pruebas", nargs="*", default=list(PRUEBAS))
    ap.add_argument("--log", default=str(LOG))
    ap.add_argument("--sin-log", action="store_true", help="sólo stdout (el .sh ya lo manda al registro)")
    a = ap.parse_args(argv)
    _LOG_ARCHIVO = None if a.sin_log else Path(a.log)

    d = Path(a.dir).expanduser()
    cache, estado_ruta = d / "sjf_cache", d / "estado.json"
    volcado, propuesta, informes = d / "tesis_v3.jsonl", d / "propuesta" / "vigencia_tesis.json", d / "informes"
    indice = Path(a.indice) if a.indice else RAIZ / "datos" / "vigencia_tesis.json"
    t0 = time.time()
    res: Optional[dict] = None
    diff: Optional[dict] = None
    try:
        with candado(d / "vigencia-semanal.lock"):
            log(f"── vigencia semanal · {RAIZ} · {'SECO' if a.seco else 'con commit y push'} ──")
            if not a.seco:
                comprobar_worktree(RAIZ, Path(a.worktree))
            estado = leer_estado(estado_ruta)
            if volcado.exists() and not a.reusar_volcado:
                volcado.unlink()                   # cada semana, Qdrant de nuevo
            tesis = gen.cargar(Path(a.env), volcado)
            if len(tesis) < 1000:
                raise RuntimeError(f"el volcado de Qdrant trae {len(tesis)} tesis: no se sigue")
            log(f"   acervo: {len(tesis):,} tesis · caché del SJF: {len(registros_en_cache(cache)):,} fichas")
            if not a.sin_actualizar:
                D = dl.Descargador(cache, validar=True, reintentos=2, aligerar=True, semanal_si_falta=True)
                vt_indice = json.loads(indice.read_text(encoding="utf-8")) if indice.exists() else {}
                curada_ruta = RAIZ / "datos" / "vigencia_curada.json"
                curada = json.loads(curada_ruta.read_text(encoding="utf-8")) if curada_ruta.exists() else {}
                res = actualizar_cache(D, tesis, estado, cache, vt_indice, curada, _dt.date.today(),
                                       guardar=lambda e: guardar_estado(estado_ruta, e),
                                       fallos=a.fallos_seguidos, repaso=a.repaso,
                                       repaso_calientes=a.repaso_calientes, aviso=log)
                estado["ultima_corrida"] = {"fecha": _dt.datetime.now().isoformat(timespec="seconds"),
                                            "resumen": {k: v for k, v in res.items() if k != "afectadas"}}
                guardar_estado(estado_ruta, estado)

            # 3. regenerar, fuera del repo
            propuesta.parent.mkdir(parents=True, exist_ok=True)
            g = subprocess.run([sys.executable, str(AQUI / "vigencia_tesis_generar.py"), "--env", str(a.env),
                                "--sjf-cache", str(cache), "--tesis-cache", str(volcado), "--salida", str(propuesta),
                                "--informes", str(informes)], capture_output=True, text=True, timeout=3600)
            if g.returncode != 0:
                raise RuntimeError(f"el generador falló (rc={g.returncode}): {(g.stderr or g.stdout).strip()[-600:]}")
            log("   " + (g.stdout.strip().splitlines() or ["(sin salida)"])[-1])
            with contextlib.suppress(Exception):
                meta = json.loads((informes / "vigencia_meta.json").read_text(encoding="utf-8"))
                log(f"   generador: {meta.get('afectadas')} afectadas · tesis nuevas del SJF "
                    f"{meta.get('tesis_nuevas_del_sjf')} · reemplazo fuera del acervo "
                    f"{meta.get('reemplazo_fuera_del_acervo')} · precedentes {meta.get('precedentes_origen')}")

            # 4. comparar por registro
            viejo_bytes = indice.read_bytes() if indice.exists() else b'{"tesis": {}}'
            viejo = json.loads(viejo_bytes.decode("utf-8")).get("tesis") or {}
            nuevo_json = json.loads(propuesta.read_text(encoding="utf-8"))
            nuevo = nuevo_json.get("tesis") or {}
            diff = comparar(viejo, nuevo)
            (d / "ultimo_diff.json").write_text(json.dumps(
                {"fecha": _dt.datetime.now().isoformat(timespec="seconds"), "diff": diff,
                 "altas": {r: nuevo[r] for r in diff["altas"]}, "bajas": {r: viejo[r] for r in diff["bajas"]},
                 "cambios": {r: {"antes": viejo[r], "ahora": nuevo[r]} for r in diff["cambios"]}},
                ensure_ascii=False, indent=1), encoding="utf-8")
            if diff["total"] == 0:
                log(resumen_linea(res, diff, "sin cambios en el índice: nada que commitear", t0))
                return 0
            for r in diff["altas"][:60]:
                log(f"   + {r} {nuevo[r].get('estado')} por {nuevo[r].get('por_registro')} [{nuevo[r].get('fuente')}]")
            for r in diff["bajas"][:60]:
                log(f"   − {r} {viejo[r].get('estado')} por {viejo[r].get('por_registro')}")
            for r in diff["cambios"][:60]:
                log(f"   ~ {r}: {', '.join(diff['campos'][r])}")

            # 5. cordura
            problemas = cordura(len(viejo), diff)
            if problemas and not a.forzar:
                raise Detenido("; ".join(problemas) + f". El diff está en {d / 'ultimo_diff.json'}; si es "
                               f"correcto, córrelo a mano con --forzar")
            n_carga = carga_con_vigencia_tesis(RAIZ, propuesta)
            if n_carga != len(nuevo) or n_carga != nuevo_json.get("n") or n_carga == 0:
                raise Detenido(f"vigencia_tesis.py lee {n_carga} entradas de la propuesta, que dice tener "
                               f"{nuevo_json.get('n')} ({len(nuevo)} en «tesis»)")
            indice.write_bytes(propuesta.read_bytes())
            try:
                correr_pruebas(RAIZ, a.pruebas)
            except Exception:
                indice.write_bytes(viejo_bytes)    # el índice queda como estaba
                raise
            if a.seco:
                log(resumen_linea(res, diff, f"SECO: {indice} queda modificado, sin commit ni push", t0))
                return 0

            # 6. commit y push
            msg_ruta = d / "propuesta" / "mensaje_commit.txt"
            msg_ruta.write_text(mensaje_commit(diff, viejo, nuevo, res, RAIZ), encoding="utf-8")
            git(RAIZ, "add", "--", "datos/vigencia_tesis.json", check=True)
            git(RAIZ, "commit", "-q", "-F", str(msg_ruta), check=True)
            sha = git(RAIZ, "rev-parse", "--short", "HEAD", check=True).stdout.strip()

            def reprobar():
                if carga_con_vigencia_tesis(RAIZ, indice) != len(nuevo):
                    raise Detenido("tras el rebase, vigencia_tesis.py ya no lee el índice entero")
                correr_pruebas(RAIZ, a.pruebas)
            estado_push = empujar(RAIZ, reprobar)
            sha = git(RAIZ, "rev-parse", "--short", "HEAD", check=True).stdout.strip()
            log(resumen_linea(res, diff, f"commit {sha} · {estado_push}", t0))
            return 0
    except Ocupado as e:
        log(f"✗ {e}")
        return 75
    except Bloqueado as e:
        log(f"✗ EL SJF NO CONTESTA: {e}")
        log(resumen_linea(res, diff, "detenido: SJF", t0))
        return 4
    except Detenido as e:
        log(f"✗ CORDURA: {e}")
        log(resumen_linea(res, diff, "detenido por cordura", t0))
        return 3
    except FalloPruebas as e:
        log(f"✗ PRUEBAS: {e}")
        log(resumen_linea(res, diff, "detenido: pruebas", t0))
        return 6
    except FalloGit as e:
        log(f"✗ GIT: {e}")
        log(resumen_linea(res, diff, "detenido: git", t0))
        return 5
    except SystemExit as e:                        # gen.cargar sale así si falta QDRANT_URL
        log(f"✗ {e}")
        return 1
    except Exception as e:
        log(f"✗ {type(e).__name__}: {e}")
        log(resumen_linea(res, diff, "detenido: error", t0))
        return 1


if __name__ == "__main__":
    sys.exit(principal())
