#!/usr/bin/env python3
"""LA LÍNEA DEL CONTROL DE CONVENCIONALIDAD COMO PILAR EN QDRANT → `lineas`.   (25-sep-2026)

POR QUÉ EXISTE
--------------
David preguntó si los Tribunales Colegiados pueden hacer control difuso sobre
las normas del juicio de origen y el chat le dio como vigentes la P. IX/2015 y
la P. X/2015, abandonadas desde feb-2022 por la P./J. 2/2022 (reg. 2024159).
La pregunta no decía «convencionalidad» y la línea no abrió; y por similitud
la sustituta salía en el puesto 38. Las palabras de `linea_coidh.pregunta_por_
linea` cubren lo que ya se vio; para lo demás está la SONDA
(`linea_coidh.sondear`): el vector de la pregunta contra esta colección, que
guarda la línea entera, verificada, como puntos pequeños que se parecen a una
pregunta de abogado.

QUÉ ES UN PUNTO
---------------
  · una entrada de la `cronologia` de datos/lineas_coidh.json (tipo_ficha
    «hito»): las que sólo referencian un hito de la Corte IDH (`ref_hito`)
    toman del hito su URL, página, extracto y aporte;
  · un hito interamericano de `hitos` que la cronología no referencie (hoy
    ninguno: la cronología los referencia todos; se deja por si se añade un
    hito sin su entrada);
  · un tramo (tipo_ficha «tramo»): título, periodo y resumen verificado;
  · el mapa (tipo_ficha «mapa»): la línea entera en una ficha, con sus seis
    tramos y los cortes de cada fuente.

El texto que se embebe es corto y en el orden en que pregunta un abogado:
«fecha — órgano — título — aporte — extracto — vigencia/postura». El payload
lleva todo lo demás (linea, tramo, orden, tipo_ficha, id, vigencia, postura,
fuerza, registro, llave, url_oficial, en_acervo…), así que la sonda devuelve
ids que `linea_coidh.seleccionar` entiende sin otra consulta.

EL ID Y LO QUE NUNCA HACE
-------------------------
id = uuid5(NAMESPACE_URL, "iurexia-lineas|{linea}|{tipo_ficha}|{id}"):
determinista, reescribir sobrescribe y no duplica; y BORRA de `lineas_p1` lo
que ya no está en el JSON (una entrada quitada o renombrada seguía abriendo la
línea por la sonda; 26-sep-2026). Nunca toca otra colección
que `lineas_p1` (alias `lineas`); `--revertir` borra sólo el alias (si apunta
a `lineas_p1`) y esa colección. En seco no abre una sola conexión ni paga un
embedding: arma los puntos, cuenta los tokens con tiktoken (cl100k_base, el de
text-embedding-3-small) y estima el costo.

    python scripts/lineas_qdrant.py                        # = --seco
    python scripts/lineas_qdrant.py --seco --salida DIR
    python scripts/lineas_qdrant.py --escribir --confirmar  # PREPARADO, sin correr (paga embeddings, escribe en Qdrant)
    python scripts/lineas_qdrant.py --revertir --confirmar  # PREPARADO, sin correr

Sin --confirmar, --escribir y --revertir abortan antes de conectarse.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

RAIZ = Path(__file__).resolve().parents[1]
RUTA_LINEAS = RAIZ / "datos" / "lineas_coidh.json"
COLECCION = "lineas_p1"            # física
ALIAS = "lineas"                   # la que consulta linea_coidh.sondear
MARCA = "lineas-p1-2026-09"        # `ingesta`
MODELO_EMBED = "text-embedding-3-small"
DIM = 1536
PRECIO_MTOK = 0.02                 # USD por millón de tokens de text-embedding-3-small (el de ingesta_coidh.py)
LOTE_EMBED, LOTE_UPSERT, PAUSA = 64, 32, 0.25
TOPE_TEXTO = 1600                  # caracteres: una ficha, no un documento

# Qué fases abre cada tramo (las mismas de linea_coidh._FASES_TRAMO).
FASES_TRAMO = {"t1": ["origen"], "t2": ["evolucion"], "t3": ["evolucion", "mexico"], "t4": ["evolucion", "mexico"],
               "t5": ["mexico", "tema:restricciones_constitucionales"], "t6": ["actual", "mexico"]}
INDICES = [("linea", "keyword"), ("tramo", "keyword"), ("tipo_ficha", "keyword"), ("id", "keyword"),
           ("tipo", "keyword"), ("vigencia", "keyword"), ("postura", "keyword"), ("fuerza", "keyword"),
           ("registro", "keyword"), ("llave", "keyword"), ("temas", "keyword"), ("orden", "integer"),
           ("ingesta", "keyword")]


def _pid(linea: str, tipo_ficha: str, ident: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"iurexia-lineas|{linea}|{tipo_ficha}|{ident}"))


def _texto_entrada(e: Dict[str, Any]) -> str:
    fecha = e.get("fecha_resolucion") or e.get("fecha_publicacion") or ""
    ident = " ".join(x for x in (e.get("clave"), f"reg. {e['registro']}" if e.get("registro") else None) if x)
    partes = [fecha, e.get("organo") or "", (e.get("titulo") or "") + (f" ({ident})" if ident else ""),
              e.get("aporte") or "", f"«{e['extracto']}»" if e.get("extracto") else "",
              "/".join(x for x in (e.get("vigencia"), e.get("postura"), e.get("sentido")) if x)]
    return " — ".join(p for p in partes if p)[:TOPE_TEXTO]


def _resuelta(e: Dict[str, Any], hitos: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """La entrada con los datos de su hito si es una referencia (`ref_hito`)."""
    if not e.get("ref_hito"):
        return e
    h = hitos.get(e["ref_hito"]) or {}
    out = dict(e)
    for k in ("url_oficial", "pagina", "parrafo", "extracto", "aporte"):
        if out.get(k) in (None, "") and h.get(k) not in (None, ""):
            out[k] = str(h[k]) if k in ("pagina", "parrafo") else h[k]
    if not out.get("titulo"):
        out["titulo"] = h.get("caso")
    return out


def armar_puntos(ruta: Path = RUTA_LINEAS) -> List[Dict[str, Any]]:
    """Todos los puntos que SE ESCRIBIRÍAN, sin vectores: {id, texto, payload}."""
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    puntos: List[Dict[str, Any]] = []
    for fid, f in (datos.get("figuras") or {}).items():
        hitos = {h["llave"]: h for h in f.get("hitos") or []}
        for t in (f.get("temas_mx") or {}).values():
            for x in t.get("extra") or []:
                hitos.setdefault(x["llave"], x)
        temas_de: Dict[str, List[str]] = collections.defaultdict(list)
        for tid, t in (f.get("temas_mx") or {}).items():
            for x in list(t.get("hitos") or []) + list(t.get("cronologia") or []):
                temas_de[str(x)].append(tid)
            for r in t.get("registros") or []:
                temas_de[f"reg:{r}"].append(tid)
        crono = f.get("cronologia") or []
        referidos = {e.get("ref_hito") for e in crono if e.get("ref_hito")}
        for e in crono:
            r = _resuelta(e, hitos)
            temas = sorted(set(temas_de.get(e["id"], []) + temas_de.get(f"reg:{e.get('registro')}", [])))
            payload = dict(
                linea=fid, tramo=e.get("tramo"), orden=e.get("orden"), tipo_ficha="hito", id=e["id"],
                tipo=e.get("tipo"), organo=r.get("organo"), titulo=r.get("titulo"),
                fecha=r.get("fecha_resolucion") or r.get("fecha_publicacion"),
                fecha_resolucion=r.get("fecha_resolucion"), fecha_publicacion=r.get("fecha_publicacion"),
                clave=r.get("clave"), registro=r.get("registro"), llave=r.get("llave") or r.get("ref_hito"),
                url_oficial=r.get("url_oficial"), pagina=r.get("pagina"), parrafo=r.get("parrafo"),
                aporte=r.get("aporte"), extracto=r.get("extracto"), vigencia=r.get("vigencia"),
                vigencia_nota=r.get("vigencia_nota"), reemplazo=r.get("reemplazo"), sustituye=r.get("sustituye"),
                postura=r.get("postura"), sentido=r.get("sentido"), fuerza=r.get("fuerza"),
                en_acervo=r.get("en_acervo"), verificacion=(r.get("verificacion") or {}).get("estado"),
                fases=FASES_TRAMO.get(e.get("tramo") or "", []), temas=temas, ingesta=MARCA)
            texto = _texto_entrada(r)
            puntos.append(dict(id=_pid(fid, "hito", e["id"]), texto=texto, payload=dict(payload, texto=texto)))
        for ll, h in hitos.items():
            if ll in referidos:
                continue
            e = dict(id=ll, llave=ll, tipo="corte_idh", organo="Corte IDH", titulo=h.get("caso"),
                     fecha_resolucion=h.get("fecha"), aporte=h.get("aporte"), extracto=h.get("extracto"),
                     url_oficial=h.get("url_oficial"))
            texto = _texto_entrada(e)
            puntos.append(dict(id=_pid(fid, "hito", ll), texto=texto, payload=dict(
                linea=fid, tramo=None, orden=None, tipo_ficha="hito", id=ll, llave=ll, tipo="corte_idh",
                titulo=h.get("caso"), fecha=h.get("fecha"), url_oficial=h.get("url_oficial"),
                aporte=h.get("aporte"), extracto=h.get("extracto"), fases=[], temas=temas_de.get(ll, []),
                ingesta=MARCA, texto=texto)))
        for tid, t in (f.get("tramos") or {}).items():
            texto = f"{f.get('nombre')}: {t.get('titulo')} ({t.get('periodo')}) — {t.get('resumen')}"[:4000]
            puntos.append(dict(id=_pid(fid, "tramo", tid), texto=texto, payload=dict(
                linea=fid, tramo=tid, orden=None, tipo_ficha="tramo", id=tid, titulo=t.get("titulo"),
                periodo=t.get("periodo"), ids=t.get("ids") or [], fases=FASES_TRAMO.get(tid, []), temas=[],
                ingesta=MARCA, texto=texto)))
        cortes = "; ".join(f"{k}: {v.get('fecha')}" for k, v in (f.get("cortes") or {}).items())
        tramos_txt = "; ".join(f"{t.get('titulo')} ({t.get('periodo')})" for t in (f.get("tramos") or {}).values())
        texto = (f"Línea jurisprudencial del {(f.get('nombre') or fid).lower()} en el sistema interamericano y en "
                 f"México, del origen a la postura actual de la SCJN: {tramos_txt}. Cortes: {cortes}.")
        puntos.append(dict(id=_pid(fid, "mapa", fid), texto=texto, payload=dict(
            linea=fid, tramo=None, orden=None, tipo_ficha="mapa", id=f"mapa:{fid}",
            fases=["evolucion", "mexico"], temas=[], ingesta=MARCA, texto=texto)))
    return puntos


def informe(puntos: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Conteo de puntos y tokens (cl100k_base) y el costo estimado de embeberlos."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        contar = lambda s: len(enc.encode(s))  # noqa: E731
    except ImportError:                          # sin tiktoken: ~4 caracteres por token
        contar = lambda s: max(1, len(s) // 4)  # noqa: E731
    toks = [contar(p["texto"]) for p in puntos]
    ids = [p["id"] for p in puntos]
    por_tipo = collections.Counter(p["payload"]["tipo_ficha"] for p in puntos)
    por_tramo = collections.Counter(p["payload"].get("tramo") or "-" for p in puntos)
    return dict(
        coleccion=COLECCION, alias=ALIAS, modelo=MODELO_EMBED, dimension=DIM, puntos=len(puntos),
        ids_unicos=len(set(ids)), por_tipo_ficha=dict(por_tipo), por_tramo=dict(sorted(por_tramo.items())),
        tokens=sum(toks), tokens_max=max(toks) if toks else 0, tokens_medio=round(sum(toks) / max(1, len(toks)), 1),
        costo_usd=round(sum(toks) * PRECIO_MTOK / 1e6, 6), precio_mtok_usd=PRECIO_MTOK,
        nota="Estimado con tiktoken cl100k_base; OpenAI cobra por tokens de entrada del embedding.")


def seco(salida: Path, ruta: Path = RUTA_LINEAS) -> Dict[str, Any]:
    salida.mkdir(parents=True, exist_ok=True)
    puntos = armar_puntos(ruta)
    with (salida / "lineas_puntos_seco.jsonl").open("w", encoding="utf-8") as fh:
        for p in puntos:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    inf = informe(puntos)
    (salida / "lineas_informe.json").write_text(json.dumps(inf, ensure_ascii=False, indent=1), encoding="utf-8")
    return inf


def _env(ruta: Path) -> Dict[str, str]:
    env = {}
    for linea in ruta.read_text().splitlines():
        if "=" in linea and not linea.startswith("#"):
            k, v = linea.split("=", 1)
            env[k.strip()] = v.strip().strip('"')
    return env


def escribir(env_ruta: Path, ruta: Path = RUTA_LINEAS) -> int:
    """Crea `lineas_p1` (vectores, payload y HNSW en disco, como `coidh`) con sus
    índices, embebe, sube, borra los huérfanos y crea el alias `lineas`. Si el
    alias ya apunta a otra colección, no lo mueve (azul-verde es manual).
    Devuelve 1 si al final la colección no tiene exactamente los puntos del
    JSON."""
    from openai import OpenAI
    from qdrant_client import QdrantClient, models

    env = _env(env_ruta)
    q = QdrantClient(url=env["QDRANT_URL"], api_key=env["QDRANT_API_KEY"], timeout=120)
    oa = OpenAI(api_key=env["OPENAI_API_KEY"])
    puntos = armar_puntos(ruta)
    if not q.collection_exists(COLECCION):
        q.create_collection(COLECCION,
                            vectors_config={"dense": models.VectorParams(size=DIM, distance=models.Distance.COSINE,
                                                                         on_disk=True)},
                            on_disk_payload=True, hnsw_config=models.HnswConfigDiff(on_disk=True))
        for campo, tipo in INDICES:
            esquema = {"keyword": models.KeywordIndexParams(type="keyword", on_disk=True),
                       "integer": models.IntegerIndexParams(type="integer", on_disk=True, lookup=True, range=True)}[tipo]
            q.create_payload_index(COLECCION, field_name=campo, field_schema=esquema, wait=True)
        print(f"colección {COLECCION} creada ({len(INDICES)} índices)")
    gastados = 0
    for i in range(0, len(puntos), LOTE_EMBED):
        lote = puntos[i:i + LOTE_EMBED]
        e = oa.embeddings.create(model=MODELO_EMBED, input=[p["texto"] for p in lote])
        gastados += e.usage.total_tokens
        est = [models.PointStruct(id=p["id"], vector={"dense": v.embedding}, payload=p["payload"])
               for p, v in zip(lote, e.data)]
        for j in range(0, len(est), LOTE_UPSERT):
            q.upsert(COLECCION, points=est[j:j + LOTE_UPSERT], wait=True)
            time.sleep(PAUSA)
    # LOS HUÉRFANOS (revisión adversarial, 26-sep-2026): el upsert sólo pisa
    # los ids que siguen en el JSON. Una entrada quitada o renombrada (p. ej.
    # por mal verificada) se quedaba en `lineas_p1` y la sonda la devolvía: la
    # línea se abría (~6 mil tokens) por algo que ya no existe y, si su registro
    # seguía en recepcion_mx, lo forzaba a entrar. La colección es sólo de este
    # guion: se borra todo lo que no sea un punto de esta corrida.
    vivos = [p["id"] for p in puntos]
    antes = q.count(COLECCION, exact=True).count
    q.delete(COLECCION, points_selector=models.FilterSelector(filter=models.Filter(
        must_not=[models.HasIdCondition(has_id=vivos)])), wait=True)
    n = q.count(COLECCION, exact=True).count
    if antes > n:
        print(f"{antes - n} puntos huérfanos borrados (ya no están en {ruta.name})")
    alias = {a.alias_name: a.collection_name for a in q.get_aliases().aliases}
    if ALIAS not in alias:
        q.update_collection_aliases(change_aliases_operations=[
            models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name=COLECCION, alias_name=ALIAS))])
        print(f"alias {ALIAS} → {COLECCION}")
    elif alias[ALIAS] != COLECCION:
        print(f"AVISO: el alias {ALIAS} apunta a {alias[ALIAS]}; no se mueve")
    n = q.count(COLECCION, exact=True).count
    print(f"{len(puntos)} puntos subidos; en Qdrant {n}; tokens {gastados:,} ≈ {gastados * PRECIO_MTOK / 1e6:.6f} USD")
    if n != len(puntos):
        print(f"AVISO: el conteo no cuadra ({n} en {COLECCION} frente a {len(puntos)} del JSON): revisa antes de "
              "darla por buena")
        return 1
    return 0


def revertir(env_ruta: Path) -> int:
    """Borra SÓLO lo que crea este guion: el alias `lineas` si apunta a
    `lineas_p1`, y la colección `lineas_p1`."""
    from qdrant_client import QdrantClient, models
    env = _env(env_ruta)
    q = QdrantClient(url=env["QDRANT_URL"], api_key=env["QDRANT_API_KEY"], timeout=120)
    alias = {a.alias_name: a.collection_name for a in q.get_aliases().aliases}
    if alias.get(ALIAS) == COLECCION:
        q.update_collection_aliases(change_aliases_operations=[
            models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=ALIAS))])
        print(f"alias {ALIAS} borrado")
    elif ALIAS in alias:
        print(f"el alias {ALIAS} apunta a {alias[ALIAS]}, no a {COLECCION}: no se toca")
    if q.collection_exists(COLECCION):
        q.delete_collection(COLECCION)
        print(f"colección {COLECCION} borrada")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seco", action="store_true", help="arma los puntos e informa; no se conecta (por omisión)")
    ap.add_argument("--escribir", action="store_true", help="crea lineas_p1, embebe, sube y crea el alias")
    ap.add_argument("--revertir", action="store_true", help="borra el alias lineas y la colección lineas_p1")
    ap.add_argument("--confirmar", action="store_true",
                    help="sin esto --escribir y --revertir abortan: escribir en Qdrant y pagar embeddings es con permiso de David")
    ap.add_argument("--lineas", default=str(RUTA_LINEAS))
    ap.add_argument("--salida", default=str(Path(tempfile.gettempdir()) / "lineas_seco"))
    ap.add_argument("--env", default=str(RAIZ / ".env"))
    a = ap.parse_args(argv)
    if sum([a.seco, a.escribir, a.revertir]) > 1:
        print("elige uno: --seco, --escribir o --revertir")
        return 1
    if (a.escribir or a.revertir) and not a.confirmar:
        print("ABORTA: --escribir y --revertir tocan Qdrant (y --escribir paga embeddings); añade --confirmar "
              "sólo con el permiso de David")
        return 1
    if a.revertir:
        return revertir(Path(a.env))
    if a.escribir:
        return escribir(Path(a.env), Path(a.lineas))
    inf = seco(Path(a.salida), Path(a.lineas))
    print(json.dumps(inf, ensure_ascii=False, indent=1))
    print(f"puntos en {Path(a.salida) / 'lineas_puntos_seco.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
