#!/usr/bin/env python3
"""SENTENCIAS DE LA CORTE IDH POR PÁRRAFO, CON LA PÁGINA DEL PDF OFICIAL → `coidh`.   (25-sep-2026)

POR QUÉ EXISTE
--------------
Iurexia citaba el ¶124 de Almonacid con la etiqueta de Trabajadores Cesados:
el texto venía de un cuadernillo cuyas etiquetas están corridas un caso (1,408
de 1,453 trozos comprobables) y el visor abría otra edición. Aquí se ingiere la
sentencia misma, párrafo por párrafo, con la página del PDF que publica
corteidh.or.cr, para que la cita diga caso, párrafo y página y el visor abra el
documento oficial sin copia nuestra. Diseño: reingesta/coidh/plan_ingesta_coidh.md,
secciones 3.3 y 3.4, y los puntos A.5, B.13 y C.15 de su revisión escéptica.

QUÉ ES UNA UNIDAD
-----------------
El párrafo numerado dentro de su segmento: el cuerpo (sentencia u opinión
consultiva), el visto y los considerandos (supervisión), los puntos
resolutivos (reinician en 1) y cada voto.
  · Resolutivos: unas sentencias numeran corrido bajo DECLARA y DISPONE
    (Tzompaxtle 1-15: llave `C-470|r|8`); otras vuelven a 1 en cada verbo
    (El Mozote, Río Negro, Santo Domingo, la OC-21). Ahí «resolutivo 8» es
    ambiguo y cada bloque lleva su verbo: `C-252|r:dispone|8`.
  · Los anexos (listas de víctimas numeradas) se cortan: en Río Negro la
    lista de sobrevivientes seguía la cadena del ¶324 al «383» sin aviso.
  · La numeración sale de la cadena consecutiva más larga, nunca de una regex
    suelta: la ingenua halló 34 de 602 párrafos en Campo Algodonero. Ante un
    empate gana la primera aparición: en la OC-11 las preguntas citadas («1.
    ¿Se aplica…», «2. En caso…») van DESPUÉS de los ¶1 y ¶2 verdaderos.
  · Los huecos se buscan antes de darlos por perdidos: el ¶216 de la C-141 y el
    ¶51 de la C-8 están impresos sin punto («216» a secas).
  · El autor de cada voto sale del CATÁLOGO (`datos/coidh_catalogo.json`), no
    del título que se lee en el PDF (revisión B.13): si mañana cambia el
    troceador, el id del punto no cambia. El título sólo sirve para casar.
    Un voto del catálogo que no aparece se busca otra vez con sus apellidos:
    el de Martínez Gálvez en Myrna Mack tiene título en minúsculas a media
    página y quedaba dentro del voto de Abreu Burelli. Desde 2024 el mismo
    voto viene en portugués y en español: el español lleva el slug y el otro
    «slug@pt», y no se vectoriza.
  · Notas al pie fuera del vector (`notas[]`); las llamadas voladitas se quitan
    del cuerpo. Las citas en bloque en letra chica SE QUEDAN en el párrafo: el
    prototipo base mandaba a las notas todo renglón chico, y eso eran cientos
    de renglones citados por documento (172 en la C-158, 259 en la C-154).
    Aquí la zona de notas empieza en la primera nota numerada del pie y lo
    chico que queda arriba es cuerpo.
  · Página = la del PDF en base 1 (lo que entiende `#page=`), nunca el folio
    impreso: en el voto de Ferrer en Cabrera, la pág. 106 del PDF dice «5».
  · Más de 1,200 tokens → se parte por subnumeración («82.1», «82.2») o en
    ventanas de ~500 con solape de 60, y cada parte lleva la página donde
    empieza lo suyo. Hay párrafos así de verdad: el ¶47 de la OC-18 resume
    86 páginas de observaciones (47 mil tokens); el ¶127 de Myrna Mack, 22.
  · Votos sin numerar → ventanas de ~400 tokens con parrafo=null.
  · Lo que no está en español se marca y no se vectoriza.

LOS CONTROLES (plan 3.3)
------------------------
Mandan el documento a CUARENTENA (no se escribe):
  · la cadena principal no empieza en 1, o tiene más huecos que max(1, 1 %)
    de N — con N=42 (OC-11) el 1 % literal pondría en cuarentena un solo
    número que el propio PDF no imprime; se tolera uno y se avisa. La misma
    regla vale para los votos y resolutivos numerados (revisión: el ¶139 de
    Vio Grossi en la OC-24 y el ¶4 de Medina en la C-141 se pegaban al
    anterior sin dejar rastro). OJO: un hueco tolerado NO prueba que el PDF no
    imprima el número; en Tibi el «229» estaba impreso y se perdía (arreglado);
  · la página retrocede dentro de un segmento;
  · la portada no dice el nombre del catálogo o su PRIMERA fecha no es la del
    catálogo. La portada NO trae el número de Serie (revisado en 64 PDF
    oficiales el 25-sep-2026: ninguno lo imprime), así que se coteja fecha +
    nombre (+ «OC-N» en las opiniones); atrapa el cruce 384/385 del sitio y,
    desde la revisión, el cruce entre resoluciones del mismo caso (fondo,
    reparaciones, interpretación), cuya portada cita la fecha de las otras;
  · más de 5 % de páginas sin texto (C-529, págs. 65-80);
  · un voto de 20 o más párrafos con más de 10 por página (el plan decía 6:
    Salgado Pesantes en la OC-18 da 7.3 y es legítimo; el 1-440 falso de la
    C-252, 25.9), o más de 14 «párrafos» en una página del último cuarto;
  · una unidad final de más de 8,191 tokens (el tope del modelo);
  · un voto del PDF que no casa con ninguno del catálogo (sin autor de
    catálogo no hay id estable: revisión B.13), o el mismo voto dos veces en
    la misma lengua;
  · el tamaño local no es el que dio el servidor (si se guardó su cabecera);
  · un ancla de oro que no cae en su página.
Sólo AVISAN: votos del catálogo que no están en el PDF después de buscarlos
(muchos se publican aparte, en .doc), resolución de llamadas a nota < 90 %,
huecos tolerados, resolutivos que reinician, anexos omitidos, segmentos en
otra lengua, y una copia local que no es el archivo exacto de url_oficial
(`--escribir` la rechaza).

EL ID Y LO QUE NUNCA HACE
-------------------------
id = uuid(md5("coidh|{doc_id}|{seg}|{autor_slug}|{parrafo}|{sub}")), sin marca
de ingesta ni versión del troceador: reprocesar sobrescribe, no duplica
(patrón de scripts/ingesta_leyes_federales.py). Como una versión nueva puede
partir en menos ventanas, `--escribir` borra después del upsert los ids del
mismo `doc_id` que ya no existen. Nunca toca otra colección que
`coidh_parrafos_p1` (alias `coidh`), nunca borra un documento porque falte en
el catálogo, y en seco no abre una sola conexión.

    python scripts/ingesta_coidh.py --seco --grupo linea mexico --pdfs DIR [DIR…]
    python scripts/ingesta_coidh.py --seco --docs C-154 C-470 --pdfs DIR
    python scripts/ingesta_coidh.py --seco --docs C-8 A-11 --etiqueta control=C-8,A-11 --pdfs DIR
    python scripts/ingesta_coidh.py --escribir --confirmar --docs C-154 --pdfs DIR   # PREPARADO, sin correr (F1)
    python scripts/ingesta_coidh.py --revertir --confirmar [--docs C-154]           # PREPARADO, sin correr (F1)

Sin --confirmar, --escribir y --revertir abortan antes de conectarse.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import statistics
import sys
import time
import unicodedata
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
COLECCION = "coidh_parrafos_p1"      # física; el chat consulta el alias
ALIAS = "coidh"                       # = silo: _parse_results usa el nombre consultado
MARCA = "coidh-p1-2026-09"            # `ingesta`, para --revertir
PARSER = "coidh-troceo-2026-09-25"    # `parser_version`; NO entra en el id
MODELO_EMBED = "text-embedding-3-small"
DIM = 1536
PRECIO_MTOK = 0.02                    # USD por millón de tokens (el del plan; no reverificado hoy)
TECHO_RAM_GIB = 3.20                  # el de _calidad/ingestar_doctrina.py
LOTE_EMBED, LOTE_UPSERT, PAUSA = 64, 20, 0.35

TOPE_UNIDAD = 1200                    # por encima se parte
VENTANA, SOLAPE = 500, 60
VENTANA_VOTO = 400
TOPE_MODELO = 8191                    # text-embedding-3-small
HUECOS_MAX = 0.01
SIN_TEXTO_MAX = 0.05
VOTO_PARR_POR_PAG = 10   # el plan decía 6; el voto de Salgado Pesantes en la OC-18 da 7.3 y es legítimo; el falso 1-440 de El Mozote, 25.9
DENSIDAD_MAX = 14        # «párrafos» por página en el último cuarto del documento (listas de víctimas sin rótulo)
NOTAS_MIN = 0.90

# Los documentos de la línea verificada (tabla de la sección 1 del plan), más
# C-239 y C-252, que no están en la tabla pero tienen anclas de oro.
LINEA = ["C-101", "C-114", "C-141", "C-154", "C-155", "C-158", "C-169", "C-209", "C-217", "C-218",
         "C-219", "C-220", "C-221", "C-250", "C-253", "C-259", "SS:gelman_20_03_13.pdf", "C-276",
         "A-21", "C-282", "C-330", "A-24", "C-373", "C-406", "C-450", "C-470", "C-481", "C-482"]
ORO_DOCS_EXTRA = ["C-239", "C-252"]

# Las 50 anclas de oro de SCR/plan/evaluar.py: (doc, autor del voto o "", párrafo, página del PDF).
# Sin autor = cuerpo (sentencia u OC) o considerandos (supervisión).
_G = "SS:gelman_20_03_13.pdf"
ORO = [("C-154", "", 124, 53), ("C-101", "garcia-ramirez", 27, 165), ("C-114", "garcia-ramirez", 3, 115),
       ("C-141", "garcia-ramirez", 30, 83), ("C-155", "garcia-ramirez", 6, 50), ("C-155", "garcia-ramirez", 12, 52),
       ("C-158", "", 128, 47), ("C-158", "garcia-ramirez", 13, 68), ("C-169", "", 78, 22), ("C-209", "", 339, 92),
       ("C-209", "", 340, 93), ("C-219", "", 176, 66), ("C-219", "", 49, 20), ("C-220", "", 225, 86),
       ("C-220", "ferrer-mac-gregor-poisot", 21, 109), ("C-220", "ferrer-mac-gregor-poisot", 41, 116),
       ("C-221", "", 193, 57), ("C-221", "", 239, 69), ("C-253", "", 330, 118), ("C-259", "", 142, 42),
       ("C-259", "", 144, 43), ("C-276", "", 124, 38), ("C-276", "", 151, 43), ("C-282", "", 311, 109),
       ("C-282", "", 471, 159), ("C-330", "", 100, 33), ("C-373", "", 128, 32), ("C-373", "", 75, 20),
       ("C-406", "", 108, 42), ("C-406", "", 107, 41), ("C-470", "", 118, 31), ("C-470", "", 219, 54),
       ("C-481", "", 175, 52), ("C-481", "", 146, 45), ("C-482", "", 303, 79), ("C-482", "", 176, 48),
       ("C-450", "", 202, 47), ("C-250", "", 262, 92), ("C-218", "", 287, 90), ("C-217", "", 202, 64),
       ("C-239", "", 282, 82), ("C-252", "", 318, 124), ("A-21", "", 31, 13), ("A-24", "", 26, 14),
       ("A-24", "", 171, 71), ("A-24", "vio-grossi", 61, 104), (_G, "", 67, 19), (_G, "", 88, 25),
       (_G, "ferrer-mac-gregor-poisot", 43, 50), (_G, "ferrer-mac-gregor-poisot", 100, 73)]
# Preguntas de oro 14 y 15 del plan: fuera de las 50, se informan aparte.
ORO_EXTRA = [("C-158", "v:garcia-ramirez", 12, 68), ("C-470", "r", 8, 62)]


# ══════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════
def _contador():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")   # el de text-embedding-3-small
        return (lambda t: len(enc.encode(t, disallowed_special=()))), "tiktoken cl100k_base"
    except Exception:
        return (lambda t: max(1, len(t) // 4)), "caracteres/4 (sin tiktoken)"


TOKENS, METODO_TOKENS = _contador()


def _plegar(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _mayus(t: str) -> bool:
    letras = [c for c in t if c.isalpha()]
    return len(letras) >= 4 and sum(c.isupper() for c in letras) / len(letras) > 0.85


def _sha1(ruta: Path) -> str:
    h = hashlib.sha1()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


# ══════════════════════════════════════════════════════════════════════
# CATÁLOGO — el de coidh_catalogo.py; aquí sólo se lee
# ══════════════════════════════════════════════════════════════════════
def cargar_catalogo(ruta: Path) -> dict:
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    docs = datos["documentos"]
    por_archivo = {}
    for d in docs.values():
        if d.get("url_oficial"):
            por_archivo[d["url_oficial"].rsplit("/", 1)[1].lower()] = d["doc_id"]
    return dict(docs=docs, por_archivo=por_archivo, autores=datos.get("autores", {}))


def resolver_doc_id(cat: dict, clave: str) -> str | None:
    """«C-154» tal cual; «SS:gelman_20_03_13.pdf» por el nombre del archivo
    (los ids de supervisión los pone el catálogo y no conviene fijarlos aquí)."""
    if clave.startswith("SS:"):
        return cat["por_archivo"].get(clave[3:].lower())
    return clave if clave in cat["docs"] else None


def nombre_caso(d: dict) -> str:
    """Mismo formato que coidh_catalogo._nombre_caso: el resolvedor y el payload
    deben nombrar igual el mismo documento."""
    if d["clase"] == "OC":
        return f"{d['oc']} · {d['nombre']}" if d.get("oc") else d["nombre"]
    return f"{d['nombre']} Vs. {d['estado']}" if d.get("estado") else d["nombre"]


def llave_de(doc_id: str, seg: str, autor: str | None, n) -> str:
    """Mismo formato que coidh_catalogo._llave: `C-154|s|124`, `C-101|v:garcia-ramirez|27`,
    `C-470|r|8`. El visto de las supervisiones es `vi`; las ventanas sin número, `w0`, `w1`…"""
    s = {"sentencia": "s", "resolutivos": "r", "considerandos": "c", "visto": "vi"}.get(seg, seg)
    if seg == "voto":
        s = f"v:{autor or '?'}"
    return f"{doc_id}|{s}|{n}"


# ══════════════════════════════════════════════════════════════════════
# DEL PDF A RENGLONES — cuerpo, notas y llamadas
# ══════════════════════════════════════════════════════════════════════
RX_NUM = re.compile(r"^\s*(\d{1,4})\.(?!\d)\s*")          # «124.» / «124. La Corte»; no «82.1»
RX_FOLIO = re.compile(r"^[\s\-–—]*\d{1,4}[\s\-–—]*$")
RX_INDICE = re.compile(r"\.{6,}\s*\d{1,4}\s*$")             # «XIV PUNTOS RESOLUTIVOS ....... 169»
RX_DIGITOS = re.compile(r"\s*\d{1,4}\s*")


def leer_renglones(ruta: Path) -> dict:
    """Renglones del cuerpo en orden de lectura, notas al pie por página y
    estadísticas de la página. Nada de esto depende del catálogo."""
    import fitz  # PyMuPDF

    doc = fitz.open(str(ruta))
    tallas = collections.Counter()
    crudo, sin_texto = [], []
    for pi, pag in enumerate(doc):
        alto = pag.rect.height
        if len(pag.get_text("text").strip()) < 20:
            sin_texto.append(pi + 1)
        for b in pag.get_text("dict", sort=True)["blocks"]:
            for l in b.get("lines", []):
                sp = [s for s in l["spans"] if s["text"].strip()]
                if not sp:
                    continue
                for s in sp:
                    tallas[round(s["size"], 1)] += len(s["text"].strip())
                crudo.append(dict(p=pi + 1, y=l["bbox"][1] / alto, x=l["bbox"][0], spans=l["spans"], sp=sp))
    total = sum(tallas.values()) or 1
    # La talla del cuerpo: la mayor con ≥20 % de los caracteres (C-529 mezcla 10 y 9.7).
    cuerpo = max((s for s, v in tallas.items() if v >= 0.2 * total), default=10.0)

    for r in crudo:
        sp = r["sp"]
        no_dig = [s for s in sp if not RX_DIGITOS.fullmatch(s["text"])]
        tam = max(s["size"] for s in (no_dig or sp))
        # Llamada a nota: dígitos voladitos o más chicos que su renglón. La
        # talla es RELATIVA al renglón para que también salgan de una cita en
        # bloque de 8 pt (su llamada va en 5). Si el renglón no tiene letras
        # (sólo cifras y puntuación), la referencia no puede pasar de la talla
        # del cuerpo: en Tibi (C-114, pág. 92) el «229» va en 10 pt y su punto
        # en 12; comparado con el punto, el número pasaba por llamada, se
        # borraba, y el ¶229 quedaba pegado al ¶228 (revisión, 25-sep-2026).
        ref = tam if any(re.search(r"[^\W\d_]", s["text"]) for s in sp) else min(tam, cuerpo)
        llam = [s for s in sp if no_dig and RX_DIGITOS.fullmatch(s["text"])
                and ((s["flags"] & 1) or s["size"] < ref - 1.5)]
        ids_llam = {id(s) for s in llam}
        r["tam"] = round(tam, 1)
        r["chica"] = tam < cuerpo - 1.0
        r["llamadas"] = [int(s["text"]) for s in llam]
        r["texto"] = re.sub(r"\s+", " ", "".join(s["text"] for s in r["spans"] if id(s) not in ids_llam)).strip()
        r["todo"] = re.sub(r"\s+", " ", "".join(s["text"] for s in r["spans"])).strip()
        txt_sp = [s for s in sp if id(s) not in ids_llam]
        r["negrita"] = bool(txt_sp) and all(s["flags"] & 16 for s in txt_sp)
        r["cursiva"] = bool(txt_sp) and all(s["flags"] & 2 for s in txt_sp)
        r["primero_dig"] = int(sp[0]["text"]) if RX_DIGITOS.fullmatch(sp[0]["text"]) else None
        r["primero_chico"] = sp[0]["size"] < tam - 1.0 or bool(sp[0]["flags"] & 1)

    # Margen izquierdo del cuerpo: las continuaciones de nota van pegadas a él;
    # las citas en bloque, sangradas.
    xs = sorted(r["x"] for r in crudo if not r["chica"] and len(r["texto"]) > 40)
    margen = xs[len(xs) // 10] if xs else 72.0

    # Encabezados/pies repetidos (la franja de arriba y abajo), aprendidos del documento.
    npag = doc.page_count
    repet = collections.Counter()
    for p in range(1, npag + 1):
        repet.update({re.sub(r"\d+", "#", _plegar(r["todo"])) for r in crudo
                      if r["p"] == p and (r["y"] < 0.08 or r["y"] > 0.92)})
    umbral_rep = max(3, int(npag * 0.25))

    por_pag = collections.defaultdict(list)
    for r in crudo:
        por_pag[r["p"]].append(r)

    notas = collections.defaultdict(dict)      # página → {número: texto}
    abierta = None                              # (página, número) de la última nota
    ult_nota = 0
    cuerpo_ls = []
    for p in range(1, npag + 1):
        ls = por_pag.get(p, [])
        orden_y = sorted(ls, key=lambda r: r["y"])

        def es_inicio(r):
            if not r["chica"] or r["y"] <= 0.40:
                return None
            if r["primero_dig"] is not None and (r["primero_chico"] or len(r["sp"]) > 1 or r["tam"] < cuerpo - 1):
                return r["primero_dig"]
            m = re.match(r"^(\d{1,4})\s+\S", r["todo"])
            if m and int(m.group(1)) == ult_nota + 1:    # número sin formato propio: sólo en secuencia
                return int(m.group(1))
            return None

        zona_y = None
        for r in orden_y:
            if es_inicio(r) is None:
                continue
            # Inválido si debajo sigue cuerpo sustancioso: sería una cita que empieza con cifra.
            if any(not o["chica"] and o["y"] > r["y"] + 0.002 and len(o["texto"]) > 40
                   and not RX_FOLIO.match(o["todo"]) for o in ls):
                continue
            zona_y = r["y"]
            break
        if abierta:
            # Continuación de la nota de la página anterior: renglones chicos al
            # pie, después del último renglón de cuerpo y pegados al margen.
            ult_cuerpo = max((o["y"] for o in ls if not o["chica"] and len(o["texto"]) > 40), default=0.0)
            cont = [o for o in ls if o["chica"] and o["y"] > max(0.40, ult_cuerpo) and o["x"] <= margen + 4
                    and (zona_y is None or o["y"] < zona_y)]
            if cont:
                zona_y = min(o["y"] for o in cont) if zona_y is None else min(zona_y, min(o["y"] for o in cont))

        hay_notas = False
        for r in ls:
            en_zona = zona_y is not None and r["y"] >= zona_y - 0.002 and (r["chica"] or len(r["texto"]) <= 40)
            if en_zona:
                if RX_FOLIO.match(r["todo"]) and not r["chica"]:
                    continue
                n = es_inicio(r)
                if n is not None:
                    ult_nota, abierta, hay_notas = n, (p, n), True
                    resto = r["todo"][len(str(n)):].strip() if r["todo"].startswith(str(n)) else r["todo"]
                    notas[p][n] = resto
                elif abierta:
                    notas[abierta[0]][abierta[1]] = (notas[abierta[0]][abierta[1]] + " " + r["todo"]).strip()
                continue
            # Folio: número solo en la franja alta o baja (la OC-11 lo pone a 71 pt, y=0.09).
            if RX_FOLIO.match(r["todo"]) and (r["y"] < 0.12 or r["y"] > 0.90):
                continue
            # Encabezado o pie repetido. Nunca un «8.» al tope de la página: su clave
            # («#») es la misma que la de los folios y se perdían 9 párrafos de la C-154.
            clave = re.sub(r"\d+", "#", _plegar(r["todo"]))
            if ((r["y"] < 0.08 or r["y"] > 0.92) and clave != "#" and not RX_NUM.match(r["todo"])
                    and repet[clave] >= umbral_rep):
                continue
            if RX_INDICE.search(r["todo"]):
                continue
            cuerpo_ls.append(dict(p=p, y=r["y"], x=r["x"], tam=r["tam"], chica=r["chica"], texto=r["texto"],
                                  llamadas=r["llamadas"], negrita=r["negrita"], cursiva=r["cursiva"]))
        if not hay_notas and zona_y is None:
            abierta = None
    return dict(lineas=[l for l in cuerpo_ls if l["texto"]], notas=notas, cuerpo=cuerpo, paginas=npag,
                sin_texto=sin_texto, margen=margen)


# ══════════════════════════════════════════════════════════════════════
# SEGMENTOS — cuerpo, visto, considerandos, resolutivos y votos
# ══════════════════════════════════════════════════════════════════════
# Los votos en inglés también: en la OC-17 el de Jackman («DISSENTING OPINION OF
# JUDGE JACKMAN», págs. 89-90) no se reconocía, se perdía tras el «Comuníquese»
# y el aviso decía que no estaba en el PDF. Detrás de otro voto se habría
# pegado al último párrafo de ése, con su autor (revisión, 25-sep-2026).
RX_VOTO = re.compile(r"^\s*(?:VOTO\s+(?:[A-ZÁÉÍÓÚÑ]+\s*){1,8}"
                     r"|OPINI[ÓO]N\s+(?:SEPARADA|DISIDENTE|CONCURRENTE|INDIVIDUAL|PARCIALMENTE|DISCREPANTE)"
                     r"|(?:(?:PARTIALLY|PARTLY)\s+)?(?:DISSENTING|CONCURRING|SEPARATE|INDIVIDUAL|JOINT)\s+(?:(?:AND|\w+ING)\s+)*OPINION\b)")
RX_TIPO_VOTO = re.compile(r"^\s*((?:VOTO|OPINI[ÓO]N)\b.*?)\s+(?:DE\s+LOS|DE\s+LAS|DE\s+LA|DEL|DE)\s", re.I)
RX_RESOL_TITULO = re.compile(r"^\s*(?:[IVXLC]+\.?\s*)?PUNTOS\s+RESOLUTIVOS")
RX_RESOL_VERBO = re.compile(r"\b(DECLARA|DISPONE|RESUELVE|DECIDE|ES\s+DE\s+OPINI[ÓO]N|OPINA)\b")
RX_VISTO = re.compile(r"^\s*VISTOS?\b\s*:?")
RX_CONSIDERANDO = re.compile(r"^\s*CONSIDERANDO(?:\s+QUE)?\s*:?\s*$")
RX_ENUM = re.compile(r"^\s*(?:[IVXLC]+\.?\s|[A-Za-z]\.\d+(?:\.\d+)*\.?\s|[A-Za-z][\.\)]\s*[A-ZÁÉÍÓÚ]|\d+\.\d+(?:\.\d+)*\.?\s)")
RX_ROMANO = re.compile(r"^\s*[IVXLC]{1,6}\.?\s*$")
RX_FORMULA = re.compile(r"^\s*(?:Y\s+)?(?:DECLARA|DISPONE|RESUELVE|DECIDE)\b.{0,40}$"
                        r"|^\s*(?:Por|por)\s+(?:unanimidad|mayor[íi]a|\w+\s+votos?\s+contra\s+\w+)[^.]{0,60}?(?:,\s*que)?\s*:?\s*$")
RX_CIERRE = re.compile(r"^\s*(?:Redactad[ao]|Hech[ao]|Dad[ao]|Emitid[ao]|Le[íi]d[ao])\s+en\s+(?:español|espa|ingl|San\s+Jos|la\s+ciudad|castellano)"
                       r"|Comun[íi]quese\s+y\s+ej[ée]c[úu]tese", re.I)


def es_titulo(l: dict) -> bool:
    t = l["texto"]
    if RX_NUM.match(t):
        return False
    if RX_ROMANO.match(t):
        return True
    if _mayus(t) and len(t) < 160:
        return True
    return bool(l["negrita"] and RX_ENUM.match(t) and len(t) < 200)


def es_epigrafe_suelto(l: dict) -> bool:
    """«Valoración de Prueba Documental» en cursiva, justo antes de un párrafo
    numerado: sin esto se pegaba al final del ¶127 de Myrna Mack."""
    t = l["texto"]
    return ((l["cursiva"] or l["negrita"]) and len(t) < 120 and not RX_NUM.match(t)
            and not re.search(r"[.,;:]\s*$", t) and t[:1].isupper())


def detectar_votos(B: list, extra: list | None = None) -> list:
    """Índices donde empieza cada voto: título en mayúsculas en la parte alta de la
    página, más los que halló `buscar_votos_faltantes` (extra)."""
    inicios = []
    for i, l in enumerate(B):
        t = l["texto"]
        if RX_VOTO.match(t) and _mayus(t) and len(t) < 200 and l["y"] < 0.40 and not RX_NUM.match(t):
            if inicios and i - inicios[-1][0] < 3:
                continue
            tit = [t]
            for x in B[i + 1:i + 4]:
                if es_titulo(x) and not RX_VOTO.match(x["texto"]):
                    tit.append(x["texto"])
                else:
                    break
            # PyMuPDF ordena por bloques y a veces deja ANTES del «VOTO…» el resto del
            # título («GO MUDROVITSCH E» / «RICARDO C. PÉREZ MANRIQUE», C-532): para casar
            # con el catálogo se suman los renglones en mayúsculas de la misma altura.
            antes = [x["texto"] for x in B[max(0, i - 3):i]
                     if x["p"] == l["p"] and abs(x["y"] - l["y"]) < 0.06 and es_titulo(x)]
            inicios.append((i, " ".join(tit + antes), len(tit)))
    for e in extra or []:
        if all(abs(e[0] - x[0]) >= 3 for x in inicios):
            inicios.append(e)
    return sorted(inicios)


RX_TIT_VOTO = re.compile(r"^\s*(?:voto|opini[oó]n|(?:(?:partially|partly)\s+)?(?:dissenting|concurring|separate|individual|joint)\s+opinion)\b", re.I)


def buscar_votos_faltantes(B: list, faltan: list, desde: int) -> list:
    """Segunda pasada, guiada por el catálogo, para los votos cuyo título no va
    en mayúsculas o no está en la parte alta de la página. En Myrna Mack el voto
    del juez ad hoc Martínez Gálvez empieza a media pág. 186 con «Voto Razonado
    y Parcialmente Disidente del / Juez Arturo Martínez Gálvez» y, sin esto, su
    texto quedaba dentro del voto de Abreu Burelli: atribuido a otro juez. Sólo
    busca después del cuerpo principal y exige, en el título o los dos renglones
    siguientes, «Juez/Jueza» y los apellidos de ESE voto del catálogo."""
    hallados = []
    for v in faltan:
        for i in range(max(1, desde), len(B)):
            t = B[i]["texto"]
            if not RX_TIT_VOTO.match(t) or len(t) > 160:
                continue
            previo = B[i - 1]
            if not (previo["p"] != B[i]["p"] or re.search(r"[.:;]\s*$", previo["texto"]) or es_titulo(previo)
                    or len(previo["texto"]) < 60):
                continue          # «…en su Voto Razonado el Juez X…» a media frase no es un título
            ctx = B[i:i + 3]
            pl = " " + _plegar(" ".join(x["texto"] for x in ctx)) + " "
            if not re.search(r" (?:jue(?:z|za|ces|zas)|judges?) ", pl):
                continue
            if all(_casa_autor(a["nombre"], pl) for a in v["autores"]):
                ntit = next((k + 1 for k, x in enumerate(ctx)
                             if any(_casa_autor(a["nombre"], " " + _plegar(x["texto"]) + " ") for a in v["autores"])), 1)
                hallados.append((i, " ".join(x["texto"] for x in ctx[:ntit]), ntit))
                break
    return hallados


def _casa_autor(nombre: str, pl: str) -> int:
    """Tokens del apellido del catálogo presentes en el título; 0 si no casa.
    Todos, salvo el último cuando son tres o más («Ferrer Mac-Gregor» sin
    «Poisot»). Con la mitad bastaba y «Pérez» casaba a Pérez Goldberg con el
    voto de Pérez Manrique (C-563)."""
    toks = [t for t in _plegar(nombre).split() if len(t) >= 3 and t not in ("del", "las", "los")]
    if not toks:
        return 0
    hay = [f" {t} " in pl for t in toks]
    if all(hay) or (len(toks) >= 2 and all(hay[:-1])):   # «CECILIA MEDINA» por Medina Quiroga (C-113)
        return sum(hay)
    return 0


def casar_votos(detectados: list, votos_cat: list) -> tuple[list, list, list]:
    """Empareja cada título del PDF con un voto del catálogo por apellidos.
    Un título que casa con un voto YA usado es su traducción: desde 2024 la
    Corte publica en el mismo PDF el voto en portugués y en español
    (Mudrovitsch en C-532, C-563 y C-567). Devuelve
    (asignaciones [(i, titulo, voto, es_traduccion)], no_casados_pdf, no_encontrados_catalogo)."""
    usados, asign, sin = set(), [], []
    for (i, titulo, _n) in detectados:
        pl = " " + _plegar(titulo) + " "
        puntaje = []
        for v in votos_cat:
            p = [_casa_autor(a["nombre"], pl) for a in v["autores"]]
            if p and all(p):
                puntaje.append((sum(p), v))
        # Gana el que casa más apellidos; si ése ya se usó, éste es su traducción.
        # Así «PÉREZ MANRIQUE» (2 de 2) no se va con Pérez Goldberg (1 de 2).
        if puntaje:
            mejor = max(p for p, _ in puntaje)
            cand = [v for p, v in puntaje if p == mejor]
            libres = [v for v in cand if v["n"] not in usados]
            if libres:
                usados.add(libres[0]["n"])
                asign.append((i, titulo, libres[0], False))
            else:
                asign.append((i, titulo, cand[0], True))
        else:
            asign.append((i, titulo, None, False))
            sin.append(titulo[:120])
    faltan = [v["autor"] for v in votos_cat if v["n"] not in usados]
    return asign, sin, faltan


def tipo_voto(titulo: str) -> str:
    m = RX_TIPO_VOTO.match(titulo)
    t = (m.group(1) if m else titulo.split(" DEL ")[0]).lower()
    return re.sub(r"\s+", " ", t).strip()[:60]


def cadena(cands: list) -> list:
    """Subsecuencia más larga de números consecutivos (huecos de hasta 4).
    Empate: gana la primera aparición del número."""
    if not cands:
        return []
    best, last = [None] * len(cands), {}
    for k, (n, _) in enumerate(cands):
        b = (1, None)
        for gap in (1, 2, 3, 4):
            j = last.get(n - gap)
            if j is not None and best[j][0] + 1 > b[0]:
                b = (best[j][0] + 1, j)
        best[k] = b
        if n not in last or best[last[n]][0] < b[0]:
            last[n] = k
    k = max(range(len(cands)), key=lambda z: (best[z][0], -z))
    out = []
    while k is not None:
        out.append(cands[k])
        k = best[k][1]
    return out[::-1]


RX_PARRS_INDICE = re.compile(r"^\s*p[áa]rrs?\.\s*\d{1,4}(?:\s*[-–]\s*\d{1,4})?\s*$")


def _es_entrada_de_indice(ls: list, i: int) -> bool:
    """«1. Falta de agotamiento…» seguido, a los pocos renglones, de un «párrs. 29-36»
    suelto: es el índice, no un párrafo. En Vélez Loor (C-218) los «1.»-«4.» del
    índice (págs. 2-3) ganaban el empate a los ¶1-4 de verdad (págs. 4-5), que
    quedaban pegados dentro de un «¶4» de 1,201 tokens (revisión, 25-sep-2026)."""
    return len(ls[i]["texto"]) < 120 and any(RX_PARRS_INDICE.match(x["texto"]) for x in ls[i + 1:i + 5])


def candidatos(ls: list, cuerpo: float, desde: int = 0, hasta: int | None = None) -> list:
    hasta = len(ls) if hasta is None else hasta
    out = []
    for i in range(desde, hasta):
        m = RX_NUM.match(ls[i]["texto"])
        if m and ls[i]["tam"] >= cuerpo - 0.6 and not _es_entrada_de_indice(ls, i):
            out.append((int(m.group(1)), i))
    return out


def recuperar_huecos(ls: list, cad: list, cuerpo: float) -> tuple[list, list]:
    """Busca cada número que falta entre sus vecinos, aceptando que venga sin
    punto («216» a secas, C-141) o un punto más chico que el cuerpo (los ¶191,
    193 y 230 de la OC-23 van en 9 pt con cuerpo de 10) — sólo ahí, nunca en
    otro lugar: la cadena exige la talla del cuerpo para que no se cuelen las
    listas numeradas de las citas en bloque."""
    if len(cad) < 2:
        return cad, []
    pos = {n: i for n, i in cad}
    recuperados = []
    nuevos = list(cad)
    for (a, ia), (b, ib) in zip(cad, cad[1:]):
        for g in range(a + 1, b):
            rx = re.compile(rf"^\s*{g}(?:\.(?!\d)|\s|$)")
            for j in range(ia + 1, ib):
                if ls[j]["tam"] >= cuerpo - 1.0 and rx.match(ls[j]["texto"]) and g not in pos:
                    nuevos.append((g, j))
                    pos[g] = j
                    recuperados.append(g)
                    break
    nuevos.sort(key=lambda t: t[1])
    return nuevos, recuperados


VERBOS = {"DECIDE": "decide", "DECLARA": "declara", "DISPONE": "dispone", "RESUELVE": "resuelve", "OPINA": "opina"}


def _verbo(t: str):
    m = RX_RESOL_VERBO.search(t)
    if not m:
        return None
    v = m.group(1)
    return "opina" if v.startswith("ES") else VERBOS.get(v)


def corridas_resolutivas(cands: list, B: list, r0: int) -> list:
    """Bloques de puntos resolutivos. Unas sentencias numeran corrido bajo
    DECLARA y DISPONE (Tzompaxtle: 1-5 y 6-15); otras vuelven a 1 en cada verbo
    (El Mozote: DECIDE 1, DECLARA 1-8, DISPONE 1-16; la OC-21: DECIDE 1, ES DE
    OPINIÓN 1-14). Un bloque vale si entre el anterior y su primer punto hay un
    verbo en mayúsculas: así cae el «1. Por tanto,» de Radilla, que va antes
    del DECIDE y es un error de numeración del PDF. Un número que no sigue a
    la corrida actual puede seguir a una anterior (un «1.» citado dentro del
    punto 7 no corta el 8). Devuelve [(verbo, [(n, i)…])]."""
    corridas = []
    for n, i in cands:
        ext = next((c for c in reversed(corridas) if 0 < n - c[-1][0] <= 2), None)
        if ext is not None:
            ext.append((n, i))
        elif n == 1:
            corridas.append([(n, i)])
    buenas, frontera = [], r0
    for c in corridas:
        verbos = [_verbo(B[j]["texto"]) for j in range(frontera, c[0][1])
                  if _mayus(B[j]["texto"][:40]) and _verbo(B[j]["texto"])]
        if verbos:
            buenas.append((verbos[-1], c))
            frontera = c[-1][1] + 1
    if not buenas and corridas:        # resolutivos sin verbo en mayúsculas (sentencias viejas)
        buenas = [(None, max(corridas, key=len))]
    return buenas


RX_ANEXO = re.compile(r"^\s*ANEXOS?\b")


def _es_anexo(l: dict) -> bool:
    return bool(RX_ANEXO.match(l["texto"])) and _mayus(l["texto"][:60]) and len(l["texto"]) < 200


def _es_marca_resolutiva(l: dict) -> bool:
    t = l["texto"]
    return bool(RX_RESOL_TITULO.match(t) or (RX_RESOL_VERBO.search(t) and (_mayus(t) or len(t) < 90)))


def cortar_en_resolutivos(B: list, cad: list, cuerpo: float, b: int) -> tuple[list, dict | None]:
    """La cadena más larga puede seguir de los considerandos a los resolutivos:
    en la supervisión de García Rodríguez (26-nov-2024) hay 4 considerandos y 8
    puntos resolutivos, y la cadena salía 1-4 (considerandos) + 5-8 (resolutivos):
    los resolutivos 1-4 quedaban dentro del considerando 4 y los 5-8 se
    llamaban «considerando». Se corta en una marca resolutiva en mayúsculas
    («RESUELVE:», «DECLARA:»…) que caiga ENTRE dos eslabones, si detrás viene un
    «1.» del cuerpo antes del eslabón siguiente y la numeración que arranca ahí
    cubre al menos lo que la cadena tenía después de la marca (así una cita de
    resolutivos ajenos a media sentencia no la corta). Revisión, 25-sep-2026."""
    for k in range(max(1, len(cad) // 2), len(cad)):     # sólo en la segunda mitad: el índice también dice «PUNTOS RESOLUTIVOS»
        lo, hi = cad[k - 1][1], cad[k][1]
        m = next((j for j in range(lo + 1, hi)
                  if _es_marca_resolutiva(B[j]) and _mayus(B[j]["texto"][:40])), None)
        if m is None or not any(n == 1 for n, _ in candidatos(B, cuerpo, m + 1, hi)):
            continue
        sigue = 1
        for n, _ in candidatos(B, cuerpo, m + 1, b):
            if n == sigue:
                sigue += 1
        if sigue - 1 >= len(cad) - k:
            return cad[:k], dict(pag=B[m]["p"], marca=B[m]["texto"][:60], quitados=[n for n, _ in cad[k:]])
    return cad, None


def segmentar(B: list, clase: str, cuerpo: float, extra_votos: list | None = None) -> tuple[list, dict]:
    """Parte los renglones del cuerpo en segmentos con su cadena de párrafos.
    Cada segmento: tipo, desde, hasta, inicios=[(n, índice)], titulo (votos), verbo (resolutivos)."""
    info = {"huecos_recuperados": [], "resolutivos_bloques": 0, "anexos_omitidos": []}
    votos = detectar_votos(B, extra_votos)
    fin_principal = votos[0][0] if votos else len(B)

    # Los anexos (listas de víctimas numeradas 1…N) NO son párrafos: en Río Negro
    # (C-250) la lista de sobrevivientes siguió la cadena del ¶324 al «383» sin que
    # nada avisara. Se corta en el primer «ANEXO» en mayúsculas que venga después
    # de una marca resolutiva situada en el último 60 % del cuerpo (el índice del
    # principio también dice «PUNTOS RESOLUTIVOS» y «ANEXO»).
    desde_marca = next((j for j in range(int(0.4 * fin_principal), fin_principal) if _es_marca_resolutiva(B[j])), None)
    if desde_marca is not None:
        ax = next((j for j in range(desde_marca, fin_principal) if _es_anexo(B[j])), None)
        if ax is not None:
            info["anexos_omitidos"].append(dict(desde_pag=B[ax]["p"], hasta_pag=B[fin_principal - 1]["p"],
                                                titulo=B[ax]["texto"][:80]))
            fin_principal = ax
    segs = []

    # 1) El cuerpo principal (y, en supervisiones, visto + considerandos).
    subs = [("sentencia", 0, fin_principal)]
    if clase == "SS":
        iv = next((i for i in range(fin_principal) if RX_VISTO.match(B[i]["texto"]) and _mayus(B[i]["texto"][:12])), None)
        ic = next((i for i in range(fin_principal) if RX_CONSIDERANDO.match(B[i]["texto"])), None)
        if ic is not None:
            subs = ([("visto", iv, ic)] if iv is not None and iv < ic else []) + [("considerandos", ic, fin_principal)]
    for tipo, a, b in subs:
        cad = cadena(candidatos(B, cuerpo, a, b))
        cad, rec = recuperar_huecos(B, cad, cuerpo)
        info["huecos_recuperados"] += rec
        if cad and tipo in ("sentencia", "considerandos"):
            cad, cortada = cortar_en_resolutivos(B, cad, cuerpo, b)
            if cortada:
                info["cadena_cortada"] = cortada
        fin = b
        bloques = []
        if cad and tipo in ("sentencia", "considerandos"):
            ult = cad[-1][1]
            r0 = next((j for j in range(ult + 1, b) if _es_marca_resolutiva(B[j])), None)
            cands_r = candidatos(B, cuerpo, (r0 if r0 is not None else ult + 1), b)
            if r0 is None:
                r0 = next((i for n, i in cands_r if n == 1), None)
                cands_r = [c for c in cands_r if r0 is not None and c[1] >= r0]
            if r0 is not None and cands_r:
                bloques = corridas_resolutivas(cands_r, B, r0)
                if bloques:
                    fin = r0
        segs.append(dict(tipo=tipo, desde=a, hasta=fin, inicios=cad, autor=None, titulo=None))
        info["resolutivos_bloques"] = max(info["resolutivos_bloques"], len(bloques))
        for k, (verbo, c) in enumerate(bloques):
            hasta = bloques[k + 1][1][0][1] if k + 1 < len(bloques) else b
            segs.append(dict(tipo="resolutivos", desde=(fin if k == 0 else c[0][1]), hasta=hasta, inicios=c,
                             autor=None, titulo=None, verbo=verbo if len(bloques) > 1 else None))

    # 2) Los votos; un anexo dentro de un voto también se corta (Vio Grossi en El Mozote: anexos A-E).
    for k, (i, titulo, ntit) in enumerate(votos):
        j = votos[k + 1][0] if k + 1 < len(votos) else len(B)
        a = i + ntit
        ax = next((x for x in range(a + 3, j) if _es_anexo(B[x])), None)
        if ax is not None:
            info["anexos_omitidos"].append(dict(desde_pag=B[ax]["p"], hasta_pag=B[j - 1]["p"], titulo=B[ax]["texto"][:80]))
            j = ax
        cad = cadena(candidatos(B, cuerpo, a, j))
        cad, rec = recuperar_huecos(B, cad, cuerpo)
        segs.append(dict(tipo="voto", desde=a, hasta=j, inicios=cad, autor=None, titulo=titulo))
    return segs, info


# ══════════════════════════════════════════════════════════════════════
# UNIDADES — párrafo, partes y ventanas
# ══════════════════════════════════════════════════════════════════════
def _unir(renglones: list) -> str:
    t = " ".join(r["texto"] for r in renglones)
    t = re.sub(r"(\w)- (\w)", r"\1\2", t)          # guion de corte de renglón
    return re.sub(r"\s+", " ", t).strip()


def _ventanas(renglones: list, objetivo: int, solape: int) -> list:
    """Ventanas de renglones de ~objetivo tokens que se enciman ~solape.
    Devuelve [(renglones, índice del primer renglón propio)]."""
    toks = [TOKENS(r["texto"]) + 1 for r in renglones]
    out, ini = [], 0
    while ini < len(renglones):
        fin, acum = ini, 0
        while fin < len(renglones) and (acum < objetivo or fin == ini):
            acum += toks[fin]
            fin += 1
        propio = ini
        if out:
            # el solape: retroceder desde `ini` hasta juntar ~solape tokens
            k, s = ini, 0
            while k > out[-1][2] and s < solape:
                k -= 1
                s += toks[k]
            out.append((k, fin, ini))
        else:
            out.append((ini, fin, ini))
        if fin >= len(renglones):
            break
        ini = fin
    return [(renglones[a:b], c - a) for a, b, c in out]


def construir_unidades(B: list, segs: list, cuerpo: float) -> tuple[list, dict]:
    """Del segmento a unidades crudas (sin partir): párrafo con sus renglones."""
    unidades, cob = [], {"chars_total": sum(len(l["texto"]) for l in B), "chars_en_unidades": 0}
    for s in segs:
        ini = [i for _, i in s["inicios"]]
        nums = [n for n, _ in s["inicios"]]
        seccion = None
        # Texto antes del primer número: en votos es su introducción (o todo el voto
        # si no está numerado); en el cuerpo es portada, índice y composición.
        if s["tipo"] == "voto":
            chars_seg = sum(len(B[i]["texto"]) for i in range(s["desde"], s["hasta"]))
            chars_num = sum(len(B[i]["texto"]) for i in range(ini[0], s["hasta"])) if ini else 0
            numerado = len(ini) >= 3 and chars_num >= 0.5 * max(1, chars_seg)
            pre_hasta = ini[0] if numerado else s["hasta"]
            pre = [B[i] for i in range(s["desde"], pre_hasta) if not es_titulo(B[i]) and not RX_CIERRE.search(B[i]["texto"])]
            if len(_unir(pre)) >= 200:
                unidades.append(dict(seg=s, parrafo=None, renglones=pre, seccion=None))
            if not numerado:
                continue
        if not ini:
            continue
        cortes = ini + [s["hasta"]]
        for k, n in enumerate(nums):
            a, b = cortes[k], cortes[k + 1]
            renglones, cerrado = [], False
            for j in range(a, b):
                l = B[j]
                if j > a and RX_CIERRE.search(l["texto"]):
                    cerrado = True       # «Redactada en español…», firmas: fuera
                if cerrado:
                    continue
                if j > a and es_titulo(l):
                    seccion = l["texto"] if not RX_ROMANO.match(l["texto"]) else seccion
                    continue
                if s["tipo"] == "resolutivos" and j > a and RX_FORMULA.match(l["texto"]):
                    continue
                renglones.append(l)
            # epígrafes sueltos al final: son del párrafo siguiente
            while len(renglones) > 1 and es_epigrafe_suelto(renglones[-1]):
                seccion = renglones.pop()["texto"]
            if renglones:
                primero = dict(renglones[0])
                primero["texto"] = RX_NUM.sub("", primero["texto"], count=1) if RX_NUM.match(primero["texto"]) \
                    else re.sub(rf"^\s*{n}\s*", "", primero["texto"], count=1)
                renglones = [primero] + renglones[1:]
            unidades.append(dict(seg=s, parrafo=n, renglones=renglones, seccion=seccion_previa(B, a, s) or seccion))
    for u in unidades:
        cob["chars_en_unidades"] += sum(len(r["texto"]) for r in u["renglones"])
    return unidades, cob


def seccion_previa(B: list, a: int, s: dict):
    """El epígrafe inmediatamente anterior (hasta 4 renglones arriba)."""
    for h in range(a - 1, max(s["desde"] - 1, a - 5), -1):
        if es_titulo(B[h]) and not RX_ROMANO.match(B[h]["texto"]):
            return B[h]["texto"][:160]
        if es_epigrafe_suelto(B[h]):
            return B[h]["texto"][:160]
    return None


def partir(u: dict) -> list:
    """Una unidad de más de TOPE_UNIDAD tokens se parte: primero por su
    subnumeración («82.1»), después en ventanas. Devuelve [(renglones, propio, etiqueta)]."""
    texto = _unir(u["renglones"])
    if u["parrafo"] is None:
        return [(r, p, None) for r, p in _ventanas(u["renglones"], VENTANA_VOTO, SOLAPE)] \
            if TOKENS(texto) > VENTANA_VOTO * 1.3 else [(u["renglones"], 0, None)]
    if TOKENS(texto) <= TOPE_UNIDAD:
        return [(u["renglones"], 0, None)]
    rx = re.compile(rf"^\s*{u['parrafo']}\.(\d{{1,3}})\.?\s")
    cortes = [i for i, r in enumerate(u["renglones"]) if rx.match(r["texto"])]
    piezas = []
    if len(cortes) >= 2:
        bordes = ([0] if cortes[0] > 0 else []) + cortes + [len(u["renglones"])]
        for a, b in zip(bordes, bordes[1:]):
            m = rx.match(u["renglones"][a]["texto"])
            piezas.append((u["renglones"][a:b], f"{u['parrafo']}.{m.group(1)}" if m else None))
    else:
        piezas = [(u["renglones"], None)]
    out = []
    for rs, et in piezas:
        if TOKENS(_unir(rs)) > TOPE_UNIDAD:
            out += [(v, p, et) for v, p in _ventanas(rs, VENTANA, SOLAPE)]
        else:
            out.append((rs, 0, et))
    return out


# ══════════════════════════════════════════════════════════════════════
# IDIOMA, NOTAS Y ARISTAS
# ══════════════════════════════════════════════════════════════════════
# Sólo palabras que la otra lengua no usa: «como», «para», «sobre», «entre» son
# de las dos y hacían pasar por español el voto en portugués de la C-567.
_ES = set("el los del las la una y al con su sus pero muy también".split())
_PT = set("não são uma pelo pela do da dos das em ao às com foi isso também seu sua os as e".split())
_EN = set("the of and which that is this with shall be by from".split())


def idioma(texto: str) -> str:
    """Español salvo que otra lengua gane con claridad (≥8 marcas y el doble que
    el español): el voto en portugués de la C-529 no debe vectorizarse."""
    w = re.findall(r"[a-záéíóúñãõçêâ]+", texto.lower())
    c = {"es": sum(x in _ES for x in w), "pt": sum(x in _PT for x in w), "en": sum(x in _EN for x in w)}
    mejor = max(c, key=c.get)
    return mejor if mejor != "es" and c[mejor] >= 8 and c[mejor] >= 1.5 * max(1, c["es"]) else "es"


def notas_de(u_renglones: list, notas: dict) -> tuple[list, list]:
    """Texto de cada nota llamada en esos renglones; se busca en sus páginas y en la siguiente."""
    out, sin = [], []
    pags = sorted({r["p"] for r in u_renglones})
    for r in u_renglones:
        for n in r["llamadas"]:
            txt = None
            for p in pags + [pags[-1] + 1]:
                if n in notas.get(p, {}):
                    txt = notas[p][n]
                    break
            if txt is None:
                sin.append(n)
            elif not any(x["n"] == n for x in out):
                out.append({"n": n, "texto": txt.strip()})
    return out, sin


RX_PARR = re.compile(r"p[áa]rrs?\.\s*(\d[\d\s,y\-–a]*)")
RX_SIMILAR = re.compile(r"en\s+(?:similar|el\s+mismo)\s+sentido", re.I)


def _parrafos(s: str) -> list:
    out = []
    for a, b in re.findall(r"(\d{1,4})\s*(?:[-–]|a)\s*(\d{1,4})", s):
        a, b = int(a), int(b)
        if 0 < b - a <= 10:
            out += list(range(a, b + 1))
    out += [int(x) for x in re.findall(r"\d{1,4}", re.sub(r"\d{1,4}\s*(?:[-–]|a)\s*\d{1,4}", " ", s))]
    return sorted(set(out))


RX_REF = re.compile(r"Series?\s+([CA])\s+No\.?\s*(\d{1,3})(?!\d)|OC-(\d{1,2})/\d{2,4}")


def aristas_de_notas(notas: list) -> tuple[list, list]:
    """Llaves citadas en las notas: «Serie C No. 154, párr. 124» → C-154|s|124.
    Sin LLM y sin resolver nombres: sólo lo que la nota dice con su número
    (plan 3.5; el grafo completo con «supra nota N» es de aristas.py). Cada
    referencia se lleva el «párr.» que tenga ANTES de la referencia siguiente:
    con una por trozo se perdía el «OC-21/14, párr. 31» que cierra la nota 32
    de la OC-24, detrás de «Serie C No. 260, párr. 221»."""
    cita, similar = set(), set()
    for nt in notas:
        for pieza in re.split(r";|\.\s+(?=Cfr\.|V[ée]ase|En\s)", nt["texto"]):
            refs = list(RX_REF.finditer(pieza))
            for k, m in enumerate(refs):
                dest = f"{m.group(1)}-{int(m.group(2))}" if m.group(1) else f"A-{int(m.group(3))}"
                tope = refs[k + 1].start() if k + 1 < len(refs) else len(pieza)
                mp = RX_PARR.search(pieza[m.end():min(tope, m.end() + 80)])
                if not mp:
                    continue
                for n in _parrafos(mp.group(1)):
                    (similar if RX_SIMILAR.search(pieza) else cita).add(f"{dest}|s|{n}")
    return sorted(cita), sorted(similar - cita)


# ══════════════════════════════════════════════════════════════════════
# PORTADA
# ══════════════════════════════════════════════════════════════════════
MESES = {m: i + 1 for i, m in enumerate("enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split())}
MESES["setiembre"] = 9


def cotejar_portada(ruta: Path, d: dict) -> dict:
    """La portada contra el catálogo: fecha, nombre y, en OC, el «OC-N»."""
    import fitz
    doc = fitz.open(str(ruta))
    t = " ".join(doc[i].get_text() for i in range(min(2, doc.page_count)))
    pl = " " + _plegar(t) + " "
    fechas = []
    for dd, mm, aa in re.findall(r"(\d{1,2})(?:o|er)?\s+de\s+([a-z]+)\s+de\s+(\d{4})", _plegar(t)):
        if mm in MESES:
            fechas.append(f"{aa}-{MESES[mm]:02d}-{int(dd):02d}")
    # La fecha del catálogo tiene que ser la PRIMERA de la portada. Con «alguna
    # de las cuatro primeras» pasaban los hermanos del mismo caso: la portada de
    # una interpretación o de unas reparaciones cita la fecha del fondo, y el PDF
    # de C-8 se aceptaba como C-5, el de C-175 como C-163, el de C-91 como C-70
    # (9 de 9 probados). En los 79 PDF locales la propia fecha va primero
    # (revisión, 25-sep-2026).
    ok_fecha = fechas[:1] == [d.get("fecha")]
    toks = [x for x in _plegar(d.get("nombre", "")).split()
            if len(x) >= 4 and x not in ("caso", "otros", "otras", "familiares", "miembros", "comunidad")]
    toks = toks[:8]
    ok_nombre = (not toks) or sum(f" {x} " in pl for x in toks) >= max(1, len(toks) // 2)
    ok_oc = True
    if d.get("clase") == "OC" and d.get("oc"):
        n = d["oc"].split("-")[1].split("/")[0]
        ok_oc = bool(re.search(rf"\boc\s*{n}\b", pl))
    return dict(ok=ok_fecha and ok_nombre and ok_oc, fecha=ok_fecha, nombre=ok_nombre, oc=ok_oc,
                fechas_vistas=fechas[:4])


# ══════════════════════════════════════════════════════════════════════
# UN DOCUMENTO → PUNTOS (sin vectores) + CONTROLES
# ══════════════════════════════════════════════════════════════════════
def _cabecera(d: dict, seg: str, autor_nombre: str | None, tvoto: str | None, parrafo, etiqueta, k, n,
              verbo: str | None = None) -> str:
    fecha = "-".join(reversed(d["fecha"].split("-"))) if d.get("fecha") else ""
    if d["clase"] == "OC":
        caso, serie = f"Opinión Consultiva {d.get('oc') or ''}".strip(), f"Serie A {d['serie_num']}"
    elif d["clase"] == "SS":
        caso, serie = f"Caso {nombre_caso(d)}", "Supervisión de cumplimiento"
    else:
        caso, serie = f"Caso {nombre_caso(d)}", f"Serie C {d['serie_num']}"
    partes = ["Corte IDH", caso, fecha, serie]
    if seg == "voto":
        partes.append(f"{(tvoto or 'voto').capitalize()} de {autor_nombre}")
    num = etiqueta or parrafo
    if num is not None:
        partes.append({"resolutivos": f"punto resolutivo {num}" + (f" ({verbo})" if verbo else ""),
                       "considerandos": f"considerando {num}",
                       "visto": f"visto {num}"}.get(seg, f"párr. {num}"))
    if n > 1:
        partes.append(f"fragmento {k + 1} de {n}")
    return "[" + " | ".join(p for p in partes if p) + "]"


def _cita_canonica(d: dict, seg: str, autor_nombre: str | None, num, verbo: str | None = None) -> str:
    """Formato de coidh_catalogo._cita_canonica (más el verbo si los resolutivos reinician)."""
    base = d["cita"].rstrip(". ")
    if seg == "voto" and autor_nombre:
        base += f". Voto del juez {autor_nombre}"
    if num is None:
        return base + "."
    et = {"resolutivos": "punto resolutivo", "considerandos": "considerando", "visto": "visto"}.get(seg, "párr.")
    return f"{base}, {et} {num}" + (f" ({verbo})." if verbo else ".")


def id_de(doc_id: str, seg: str, autor_slug: str | None, parrafo, sub: int) -> str:
    base = f"coidh|{doc_id}|{seg}|{autor_slug or ''}|{'' if parrafo is None else parrafo}|{sub}"
    return str(uuid.UUID(hashlib.md5(base.encode()).hexdigest()))


def _cabeceras_http(ruta: Path, url_oficial: str) -> dict:
    """ETag y tamaño que dio el servidor al bajarlo (`<pdf>.hdr.json`), si se guardaron.
    El monitor de F4 usará ETag + total de Content-Range, no sha1 (revisión C.15):
    el sha1 exige bajar el archivo entero; el ETag sale de un GET de un byte.
    Sólo vale si la cabecera es de ESE archivo: la de «seriec_221_esp.pdf» no
    dice nada de «seriec_221_esp1.pdf», que es el que enlaza el catálogo."""
    lado = ruta.with_name(ruta.name + ".hdr.json")
    if lado.exists():
        try:
            h = json.loads(lado.read_text())
            if (h.get("url") or "").rsplit("/", 1)[-1].lower() == url_oficial.rsplit("/", 1)[-1].lower():
                return {"etag": h.get("etag"), "content_length": h.get("content_length"),
                        "last_modified": h.get("last_modified")}
        except Exception:
            pass
    return {}


def procesar_documento(d: dict, ruta: Path, cat: dict) -> dict:
    t0 = time.time()
    R = leer_renglones(ruta)
    B, cuerpo = R["lineas"], R["cuerpo"]
    segs, info = segmentar(B, d["clase"], cuerpo)
    cuarentena, avisos = [], []

    # Idioma por segmento (antes de los autores: decide cuál de dos versiones es la canónica).
    for s in segs:
        s["idioma"] = idioma(" ".join(B[i]["texto"] for i in range(s["desde"], min(s["hasta"], s["desde"] + 400))))

    # Votos: el autor sale del catálogo. Si un mismo voto viene en dos lenguas,
    # la versión en español lleva el slug a secas y la otra «slug@pt» (no se
    # vectoriza); dos versiones en español del mismo voto es un error: cuarentena.
    votos_cat = d.get("votos") or []
    vsegs = [s for s in segs if s["tipo"] == "voto"]
    asign, sin_casar, faltan = casar_votos([(s["desde"], s["titulo"], 0) for s in vsegs], votos_cat)
    if faltan:
        # Se busca desde los resolutivos (o desde el 60 % del documento si no los hay).
        fin_cuerpo = next((s["desde"] for s in segs if s["tipo"] == "resolutivos"), int(0.6 * len(B)))
        extra = buscar_votos_faltantes(B, [v for v in votos_cat if v["autor"] in faltan], fin_cuerpo)
        if extra:
            segs, info = segmentar(B, d["clase"], cuerpo, extra)
            for s in segs:
                s["idioma"] = idioma(" ".join(B[i]["texto"] for i in range(s["desde"], min(s["hasta"], s["desde"] + 400))))
            vsegs = [s for s in segs if s["tipo"] == "voto"]
            asign, sin_casar, faltan = casar_votos([(s["desde"], s["titulo"], 0) for s in vsegs], votos_cat)
            avisos.append(f"votos con título fuera de formato hallados por el catálogo: {[t[:70] for _, t, _ in extra]}")
    por_voto = collections.defaultdict(list)
    for s, (_i, _t, v, _trad) in zip(vsegs, asign):
        s["tipo_voto"] = tipo_voto(s["titulo"] or "")
        s["autor"], s["autor_nombre"] = None, None
        if v:
            por_voto[v["n"]].append((s, v))
    repetidos, slug_de = [], {}
    for n, lista in por_voto.items():
        canon = next((x for x in lista if x[0]["idioma"] == "es"), lista[0])
        for s, v in lista:
            s["autor_nombre"] = " y ".join(a["nombre"] for a in v["autores"])
            if s is canon[0]:
                # Dos votos distintos del mismo juez (Piza Escalante en A-101,
                # Orihuela Iberico en C-13): el segundo lleva «~n» del catálogo.
                s["autor"] = v["slug"] if slug_de.setdefault(v["slug"], n) == n else f"{v['slug']}~{n}"
            elif s["idioma"] != canon[0]["idioma"]:
                s["autor"] = f"{v['slug']}@{s['idioma']}"
            else:
                s["autor"] = f"{v['slug']}~{len(repetidos) + 2}"
                repetidos.append(v["autor"])
    if sin_casar:
        cuarentena.append(f"voto del PDF sin autor en el catálogo: {sin_casar}")
    if repetidos:
        cuarentena.append(f"el mismo voto aparece dos veces en la misma lengua: {repetidos}")
    if faltan:
        avisos.append(f"votos del catálogo que no están en el PDF: {faltan}")

    unidades, cob = construir_unidades(B, segs, cuerpo)

    # Partes, payload e ids.
    sha1 = _sha1(ruta)
    nbytes = ruta.stat().st_size
    http = _cabeceras_http(ruta, d["url_oficial"])
    if http.get("content_length") and int(http["content_length"]) != nbytes:
        cuarentena.append(f"el tamaño local ({nbytes}) no es el que dio el servidor ({http['content_length']})")
    if d.get("_archivo_alterno"):
        avisos.append(f"copia local «{d['_archivo_alterno']}», no «{d['url_oficial'].rsplit('/', 1)[1]}» del catálogo: "
                      "sha1, bytes y páginas sin comprobar contra url_oficial (bajarlo antes de F1)")
    puntos, orden = [], 0
    notas_llamadas = notas_resueltas = 0
    for u in unidades:
        s = u["seg"]
        seg = s["tipo"]
        autor = s.get("autor") if seg == "voto" else None
        partes = partir(u)
        verbo = s.get("verbo") if seg == "resolutivos" else None
        # Si los resolutivos vuelven a 1 en cada verbo, «resolutivo 8» es ambiguo:
        # cada bloque lleva su verbo en la llave (C-252|r:dispone|8) y en el id.
        seg_id = f"{seg}:{verbo}" if verbo else seg
        for k, (rs, propio, etiqueta) in enumerate(partes):
            texto_raw = _unir(rs)
            if not texto_raw:
                continue
            if u["parrafo"] is None:
                llave = llave_de(d["doc_id"], seg, autor, f"w{k}")
            elif verbo:
                llave = f"{d['doc_id']}|r:{verbo}|{u['parrafo']}"
            else:
                llave = llave_de(d["doc_id"], seg, autor, u["parrafo"])
            propios = rs[propio:] or rs
            nts, _ = notas_de(rs, R["notas"])
            _, sin_propias = notas_de(propios, R["notas"])     # el solape no se cuenta dos veces
            llamadas_propias = sum(len(r["llamadas"]) for r in propios)
            notas_llamadas += llamadas_propias
            notas_resueltas += llamadas_propias - len(sin_propias)
            sin = sin_propias
            cita_a, cita_sim = aristas_de_notas(nts)
            cab = _cabecera(d, seg, s.get("autor_nombre"), s.get("tipo_voto"), u["parrafo"], etiqueta, k, len(partes), verbo)
            texto = f"{cab}\n{texto_raw}"
            tipo = ("voto_coidh" if seg == "voto" else "resolutivo_coidh" if seg == "resolutivos"
                    else {"OC": "oc_coidh", "SS": "supervision_coidh"}.get(d["clase"], "sentencia_coidh"))
            num_ref = etiqueta or u["parrafo"]
            ref = {"resolutivos": f"Punto resolutivo {num_ref}" + (f" ({verbo})" if verbo else ""),
                   "considerandos": f"Considerando {num_ref}",
                   "visto": f"Visto {num_ref}"}.get(seg, f"Párr. {num_ref}" if num_ref is not None else "Sin numerar")
            if seg == "voto":
                ref = f"Voto de {s.get('autor_nombre') or '?'}, " + (ref[0].lower() + ref[1:])
            pl = {
                # identidad
                "doc_id": d["doc_id"], "serie": d["serie"], "serie_num": d.get("serie_num"),
                "caso_id": d.get("caso_id") or (d.get("oc") or d["doc_id"]).lower(),
                "caso": nombre_caso(d), "estado": d.get("estado"), "acto": d.get("acto"),
                "acto_principal": bool(d.get("acto_principal", d["clase"] == "OC")),
                "fecha": d.get("fecha"), "fecha_int": int(d["fecha"].replace("-", "")) if d.get("fecha") else None,
                "catalogo_id": d.get("catalogo_id"),
                # posición
                "seg": seg, "voto_autor": autor, "tipo_voto": s.get("tipo_voto") if seg == "voto" else None,
                "parrafo": u["parrafo"], "sub": k, "orden": orden, "llave": llave,
                "pagina": propios[0]["p"], "pagina_fin": rs[-1]["p"],
                "ancla": " ".join(_unir(propios).split()[:15]),
                "seccion": (verbo.upper() if verbo else (u.get("seccion") or None) and u["seccion"][:160]),
                # texto
                "texto_raw": texto_raw, "texto": texto, "notas": nts, "n_tokens": TOKENS(texto),
                "idioma": s["idioma"],
                # grafo
                "cita_a": cita_a, "cita_similar": cita_sim, "citado_por_n": 0, "en_cuadernillos": [],
                "lineas": [], "fase_linea": None,
                # fuente
                "url_oficial": d["url_oficial"], "pdf_url": d["url_oficial"], "pdf_sha1": sha1, "pdf_bytes": nbytes,
                "pdf_etag": http.get("etag"),
                "cita_canonica": _cita_canonica(d, seg, s.get("autor_nombre"), etiqueta or u["parrafo"], verbo),
                # operación
                "silo": ALIAS, "tipo": tipo, "ref": ref, "ingesta": MARCA, "parser_version": PARSER,
                "qc": {k2: v2 for k2, v2 in {"parte": f"{k + 1}/{len(partes)}" if len(partes) > 1 else None,
                                              "subnumero": etiqueta,
                                              "notas_sin_resolver": sin or None,
                                              "hueco_recuperado": (u["parrafo"] in info["huecos_recuperados"]
                                                                   and seg in ("sentencia", "considerandos")) or None,
                                              "vectorizar": s["idioma"] == "es"}.items() if v2 is not None},
            }
            puntos.append({"id": id_de(d["doc_id"], seg_id, autor, u["parrafo"], k), "payload": pl})
            orden += 1

    # ── controles ────────────────────────────────────────────────────
    ids = [p["id"] for p in puntos]
    if len(ids) != len(set(ids)):
        rep = [i for i, c in collections.Counter(ids).items() if c > 1]
        dup_llaves = sorted({p["payload"]["llave"] for p in puntos if p["id"] in rep})[:8]
        cuarentena.append(f"ids repetidos ({len(rep)}): {dup_llaves}")
    principal = next((s for s in segs if s["tipo"] in ("sentencia", "considerandos") and s["inicios"]), None)
    resumen_segs = []
    for s in segs:
        nums = [n for n, _ in s["inicios"]]
        huecos = sorted(set(range(nums[0], nums[-1] + 1)) - set(nums)) if nums else []
        pags = [B[i]["p"] for _, i in s["inicios"]]
        retro = [(nums[i], pags[i], nums[i + 1], pags[i + 1]) for i in range(len(pags) - 1) if pags[i + 1] < pags[i]]
        pag_ini = B[s["desde"]]["p"] if s["desde"] < len(B) else None
        pag_fin = B[s["hasta"] - 1]["p"] if s["hasta"] - 1 < len(B) and s["hasta"] > s["desde"] else pag_ini
        resumen_segs.append(dict(tipo=s["tipo"], autor=s.get("autor"), titulo=(s.get("titulo") or "")[:100] or None,
                                 n=len(nums), rango=[nums[0], nums[-1]] if nums else None, huecos=huecos[:20],
                                 n_huecos=len(huecos), paginas=[pag_ini, pag_fin], idioma=s.get("idioma"),
                                 verbo=s.get("verbo")))
        if retro:
            cuarentena.append(f"la página retrocede en {s['tipo']} {s.get('autor') or ''}: {retro[:3]}")
        # Los huecos de los votos y resolutivos se miden igual que los del cuerpo:
        # en el voto de Vio Grossi (OC-24) el «139.» va en 6.5 pt y el ¶139 quedó
        # dentro del ¶138; en el de Medina Quiroga (C-141) el «4.» va en 8 pt. Antes
        # sólo contaba la cadena principal y esto no dejaba rastro (revisión, 25-sep-2026).
        # Sólo si la cadena es la numeración del segmento (empieza en 1 y tiene 3 o más):
        # un voto sin numerar con dos «26.» y «29.» sueltos (Pérez Goldberg, OC-32) va
        # en ventanas y sus «huecos» no significan nada.
        if s is not principal and huecos and nums[0] == 1 and len(nums) >= 3:
            tope = max(1, int(HUECOS_MAX * nums[-1]))
            (cuarentena if len(huecos) > tope else avisos).append(
                f"{len(huecos)} huecos en {s['tipo']} {s.get('autor') or s.get('verbo') or ''} 1..{nums[-1]} (tope {tope}): {huecos[:15]}")
        if s["tipo"] == "voto" and len(nums) >= 20 and pag_ini and pag_fin:
            # Sólo con 20 o más: el voto de Salgado Pesantes en Tibi tiene 7 párrafos
            # cortos en una página y es legítimo; el 1-440 falso de El Mozote no.
            # (El Mozote ya no llega aquí: sus anexos A-E se cortan en `segmentar`.)
            por_pag = len(nums) / max(1, pag_fin - pag_ini + 1)
            if por_pag > VOTO_PARR_POR_PAG:
                cuarentena.append(f"voto {s.get('autor')} con {por_pag:.1f} párrafos por página (¿anexo numerado?)")
        # Una lista numerada que se colara (víctimas, anexos sin rótulo) mete muchos
        # «párrafos» de un renglón en la misma página. Sólo en el último cuarto: la
        # OC-23 numera 30 observaciones escritas en su pág. 8 y son párrafos de verdad.
        densas = [p for p, c in collections.Counter(pags).items() if c > DENSIDAD_MAX and p > 0.75 * R["paginas"]]
        if densas:
            cuarentena.append(f"{s['tipo']} {s.get('autor') or ''} con más de 14 párrafos en las págs. {densas[:6]} (¿lista o anexo?)")
    if principal is None:
        cuarentena.append("sin cadena de párrafos en el cuerpo")
    else:
        nums = [n for n, _ in principal["inicios"]]
        huecos = sorted(set(range(1, nums[-1] + 1)) - set(nums))
        tolerados = max(1, int(HUECOS_MAX * nums[-1]))
        if nums[0] != 1:
            cuarentena.append(f"la cadena principal empieza en {nums[0]}, no en 1")
        if len(huecos) > tolerados:
            cuarentena.append(f"{len(huecos)} huecos en 1..{nums[-1]} (tope {tolerados}): {huecos[:15]}")
        elif huecos:
            avisos.append(f"huecos tolerados en 1..{nums[-1]}: {huecos}")
    if info["resolutivos_bloques"] > 1:
        verbos = [s.get("verbo") for s in segs if s["tipo"] == "resolutivos"]
        avisos.append(f"los resolutivos vuelven a 1 en cada verbo: {verbos} (llaves r:<verbo>|n)")
    if info.get("cadena_cortada"):
        c = info["cadena_cortada"]
        avisos.append(f"la cadena seguía en los resolutivos: se cortó en «{c['marca']}» (pág. {c['pag']}); "
                      f"salen del cuerpo los números {c['quitados']}")
    for ax in info["anexos_omitidos"]:
        avisos.append(f"anexo omitido, págs. {ax['desde_pag']}-{ax['hasta_pag']}: «{ax['titulo']}»")
    if not any(s["tipo"] == "resolutivos" for s in segs):
        avisos.append("sin resolutivos separados (quedan dentro del último párrafo)")
    if len(R["sin_texto"]) > SIN_TEXTO_MAX * R["paginas"]:
        cuarentena.append(f"{len(R['sin_texto'])} de {R['paginas']} páginas sin texto: {R['sin_texto'][:12]}")
    grandes = [p["payload"]["llave"] for p in puntos if p["payload"]["n_tokens"] > TOPE_MODELO]
    if grandes:
        cuarentena.append(f"unidades de más de {TOPE_MODELO} tokens: {grandes[:5]}")
    portada = cotejar_portada(ruta, d)
    if not portada["ok"]:
        cuarentena.append(f"la portada no coincide con el catálogo: {portada}")
    tasa = notas_resueltas / notas_llamadas if notas_llamadas else 1.0
    if tasa < NOTAS_MIN:
        avisos.append(f"llamadas a nota resueltas {tasa:.1%} ({notas_resueltas}/{notas_llamadas})")
    no_es = [s["tipo"] + ":" + str(s.get("autor")) + ":" + s["idioma"] for s in segs if s.get("idioma") != "es"]
    if no_es:
        avisos.append(f"segmentos que no están en español (no se vectorizan): {no_es}")

    tok = [p["payload"]["n_tokens"] for p in puntos]
    mayor = max(puntos, key=lambda p: p["payload"]["n_tokens"]) if puntos else None
    return dict(
        doc_id=d["doc_id"], archivo=ruta.name, archivo_alterno=d.get("_archivo_alterno"),
        url_oficial=d["url_oficial"], pdf_sha1=sha1, pdf_bytes=nbytes,
        pdf_etag=http.get("etag"), paginas=R["paginas"], paginas_sin_texto=len(R["sin_texto"]), cuerpo_pt=cuerpo,
        segmentos=resumen_segs, votos_catalogo=len(votos_cat), votos_pdf=len(vsegs),
        unidades=len(puntos), unidades_vectorizables=sum(1 for p in puntos if p["payload"]["qc"].get("vectorizar")),
        tokens=sum(tok), tokens_vectorizables=sum(p["payload"]["n_tokens"] for p in puntos if p["payload"]["qc"].get("vectorizar")),
        unidad_mas_larga=(dict(llave=mayor["payload"]["llave"], sub=mayor["payload"]["sub"],
                               tokens=mayor["payload"]["n_tokens"]) if mayor else None),
        partidas=sum(1 for p in puntos if p["payload"]["sub"] > 0),
        huecos_recuperados=info["huecos_recuperados"],
        notas=dict(llamadas=notas_llamadas, resueltas=notas_resueltas, tasa=round(tasa, 3)),
        cobertura_texto=round(cob["chars_en_unidades"] / max(1, cob["chars_total"]), 3),
        portada=portada, cuarentena=cuarentena, avisos=avisos, segundos=round(time.time() - t0, 2),
        _puntos=puntos)


# ══════════════════════════════════════════════════════════════════════
# ANCLAS DE ORO
# ══════════════════════════════════════════════════════════════════════
def evaluar_oro(resultados: dict, cat: dict) -> dict:
    ok, fallos = 0, []
    for clave, autor, n, pag in ORO:
        did = resolver_doc_id(cat, clave)
        r = resultados.get(did)
        if not r:
            fallos.append(dict(doc=clave, autor=autor, parrafo=n, esperada=pag, motivo="documento no procesado"))
            continue
        cand = [p["payload"] for p in r["_puntos"] if p["payload"]["parrafo"] == n and p["payload"]["sub"] == 0
                and ((not autor and p["payload"]["seg"] in ("sentencia", "considerandos"))
                     or (autor and p["payload"]["seg"] == "voto" and (p["payload"]["voto_autor"] or "").startswith(autor)))]
        if cand and cand[0]["pagina"] == pag:
            ok += 1
        else:
            fallos.append(dict(doc=did, autor=autor, parrafo=n, esperada=pag,
                               obtenida=[(c["seg"], c["voto_autor"], c["pagina"]) for c in cand]))
            if did in resultados:
                resultados[did]["cuarentena"].append(f"ancla de oro ¶{n} {autor} no cae en la pág. {pag}")
    extra = []
    for did, seg, n, pag in ORO_EXTRA:
        r = resultados.get(did)
        c = [p["payload"] for p in (r["_puntos"] if r else []) if p["payload"]["llave"] == f"{did}|{seg}|{n}" and p["payload"]["sub"] == 0]
        extra.append(dict(llave=f"{did}|{seg}|{n}", esperada=pag, obtenida=c[0]["pagina"] if c else None,
                          ok=bool(c and c[0]["pagina"] == pag)))
    return dict(total=len(ORO), aciertos=ok, fallos=fallos, extra=extra)


# ══════════════════════════════════════════════════════════════════════
# ESCRITURA Y REVERSIÓN — PREPARADAS PARA F1, NO SE HAN EJECUTADO
# ══════════════════════════════════════════════════════════════════════
INDICES = [("doc_id", "keyword"), ("serie", "keyword"), ("serie_num", "integer"), ("caso_id", "keyword"),
           ("estado", "keyword"), ("acto", "keyword"), ("acto_principal", "bool"), ("fecha_int", "integer"),
           ("seg", "keyword"), ("voto_autor", "keyword"), ("parrafo", "integer"), ("orden", "integer"),
           ("llave", "keyword"), ("pagina", "integer"), ("n_tokens", "integer"), ("idioma", "keyword"),
           ("cita_a", "keyword"), ("cita_similar", "keyword"), ("citado_por_n", "integer"),
           ("en_cuadernillos", "integer"), ("lineas", "keyword"), ("tipo", "keyword"), ("ingesta", "keyword")]


def _env(ruta: Path) -> dict:
    env = {}
    for l in ruta.read_text().splitlines():
        if "=" in l and not l.startswith("#"):
            k, v = l.split("=", 1)
            env[k.strip()] = v.strip().strip('"')
    return env


def escribir(resultados: dict, env_ruta: Path, carpeta_respaldo: Path) -> int:
    """Crea la colección y el alias si no existen, embebe y sube documento por
    documento, y borra los ids viejos del mismo doc_id. Sólo documentos sin
    cuarentena y sólo unidades en español."""
    import requests
    from openai import OpenAI
    from qdrant_client import QdrantClient, models

    env = _env(env_ruta)
    q = QdrantClient(url=env["QDRANT_URL"], api_key=env["QDRANT_API_KEY"], timeout=120)
    oa = OpenAI(api_key=env["OPENAI_API_KEY"])
    carpeta_respaldo.mkdir(parents=True, exist_ok=True)

    def ram_gib() -> float:
        r = requests.get(env["QDRANT_URL"].rstrip("/") + "/telemetry?details_level=1",
                         headers={"api-key": env["QDRANT_API_KEY"]}, timeout=60)
        return r.json()["result"]["memory"]["resident_bytes"] / 2 ** 30

    def reintentar(fn, *a, **kw):
        for i in range(5):
            try:
                return fn(*a, **kw)
            except Exception:
                if i == 4:
                    raise
                time.sleep(2 * (i + 1))

    if ram_gib() >= TECHO_RAM_GIB:
        print(f"ABORTA: Qdrant en {ram_gib():.2f} GiB (techo {TECHO_RAM_GIB})")
        return 1
    if not q.collection_exists(COLECCION):
        q.create_collection(COLECCION,
                            vectors_config={"dense": models.VectorParams(size=DIM, distance=models.Distance.COSINE, on_disk=True)},
                            on_disk_payload=True, hnsw_config=models.HnswConfigDiff(on_disk=True))
        for campo, tipo in INDICES:
            esquema = {"keyword": models.KeywordIndexParams(type="keyword", on_disk=True),
                       "integer": models.IntegerIndexParams(type="integer", on_disk=True, lookup=True, range=True),
                       "bool": models.PayloadSchemaType.BOOL}[tipo]
            q.create_payload_index(COLECCION, field_name=campo, field_schema=esquema, wait=True)
        print(f"colección {COLECCION} creada (vectores, payload y HNSW en disco; {len(INDICES)} índices)")
    alias = {a.alias_name: a.collection_name for a in q.get_aliases().aliases}
    if ALIAS not in alias:
        q.update_collection_aliases(change_aliases_operations=[
            models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name=COLECCION, alias_name=ALIAS))])
        print(f"alias {ALIAS} → {COLECCION}")
    elif alias[ALIAS] != COLECCION:
        print(f"AVISO: el alias {ALIAS} apunta a {alias[ALIAS]}; no se mueve (azul-verde es manual)")

    gastados = 0
    for did, r in resultados.items():
        if r["cuarentena"]:
            print(f"SALTA {did}: cuarentena {r['cuarentena'][:2]}")
            continue
        if r.get("archivo_alterno"):
            print(f"SALTA {did}: la copia local no es el archivo de url_oficial")
            continue
        if ram_gib() >= TECHO_RAM_GIB:
            print(f"ALTO antes de {did}: RAM en el techo")
            return 2
        pts = [p for p in r["_puntos"] if p["payload"]["qc"].get("vectorizar")]
        ids = []
        for i in range(0, len(pts), LOTE_EMBED):
            lote = pts[i:i + LOTE_EMBED]
            e = reintentar(oa.embeddings.create, model=MODELO_EMBED, input=[p["payload"]["texto"] for p in lote])
            gastados += e.usage.total_tokens
            estructuras = [models.PointStruct(id=p["id"], vector={"dense": v.embedding}, payload=p["payload"])
                           for p, v in zip(lote, e.data)]
            for j in range(0, len(estructuras), LOTE_UPSERT):
                reintentar(q.upsert, COLECCION, points=estructuras[j:j + LOTE_UPSERT], wait=True)
                time.sleep(PAUSA)
            ids += [p["id"] for p in lote]
            (carpeta_respaldo / f"{did}.ids.json").write_text(json.dumps(ids))
        # Lo que quedó de una corrida anterior con otro troceo del mismo archivo.
        viejos, sig = [], None
        filtro = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=did))])
        while True:
            pag, sig = q.scroll(COLECCION, scroll_filter=filtro, limit=500, offset=sig, with_payload=False)
            viejos += [str(p.id) for p in pag]
            if sig is None:
                break
        sobran = sorted(set(viejos) - set(ids))
        if sobran:
            reintentar(q.delete, COLECCION, points_selector=models.PointIdsList(points=sobran), wait=True)
        n = q.count(COLECCION, count_filter=filtro, exact=True).count
        print(f"{did}: {len(ids)} puntos, {len(sobran)} viejos borrados, en Qdrant {n}"
              + ("" if n == len(ids) else "  ← NO CUADRA"))
    print(f"tokens embebidos {gastados:,} ≈ {gastados * PRECIO_MTOK / 1e6:.4f} USD")
    return 0


def revertir(docs: list, env_ruta: Path) -> int:
    """Borra por filtro lo que subió ESTA ingesta (`ingesta` = MARCA), de los
    documentos pedidos o de todos. La colección no se toca: se borra a mano."""
    from qdrant_client import QdrantClient, models
    env = _env(env_ruta)
    q = QdrantClient(url=env["QDRANT_URL"], api_key=env["QDRANT_API_KEY"], timeout=120)
    must = [models.FieldCondition(key="ingesta", match=models.MatchValue(value=MARCA))]
    if docs:
        must.append(models.FieldCondition(key="doc_id", match=models.MatchAny(any=docs)))
    f = models.Filter(must=must)
    antes = q.count(COLECCION, count_filter=f, exact=True).count
    q.delete(COLECCION, points_selector=models.FilterSelector(filter=f), wait=True)
    print(f"REVERTIDO: {antes} puntos borrados; quedan {q.count(COLECCION, count_filter=f, exact=True).count}")
    return 0


# ══════════════════════════════════════════════════════════════════════
# PROGRAMA
# ══════════════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--catalogo", default=str(RAIZ / "datos" / "coidh_catalogo.json"),
                    help="datos/coidh_catalogo.json (lo construye coidh_catalogo.py --construir)")
    ap.add_argument("--pdfs", nargs="+", default=[], help="carpetas con los PDF oficiales ya bajados")
    ap.add_argument("--docs", nargs="*", default=[], help="doc_id del catálogo (C-154, A-24, SS-…) o SS:<archivo.pdf>")
    ap.add_argument("--grupo", nargs="*", default=[], choices=["linea", "mexico"],
                    help="linea = tabla de la sección 1 + C-239 y C-252 (anclas); mexico = toda sentencia Vs. México del catálogo")
    ap.add_argument("--etiqueta", nargs="*", default=[], help="grupo=doc_id,doc_id… para rotular documentos en el informe")
    ap.add_argument("--seco", action="store_true", help="trocea y controla; no abre una sola conexión")
    ap.add_argument("--escribir", action="store_true", help="F1: crea la colección y sube (requiere permiso)")
    ap.add_argument("--confirmar", action="store_true",
                    help="sin esto --escribir sólo trocea e informa: escribir en Qdrant y pagar embeddings es F1, con permiso de David")
    ap.add_argument("--revertir", action="store_true", help="borra lo subido con la marca de esta ingesta")
    ap.add_argument("--aceptar-alterno", action="store_true",
                    help="enlaza la copia «_esp» verificada (URL oficial propia) cuando el catálogo pide «_esp1/_esp2»")
    ap.add_argument("--env", default=str(RAIZ / ".env"))
    ap.add_argument("--salida", default="coidh_troceo_informe.json")
    ap.add_argument("--puntos", default="coidh_puntos_seco.jsonl", help="JSONL con lo que SE ESCRIBIRÍA, sin vectores")
    a = ap.parse_args()

    if sum([a.seco, a.escribir, a.revertir]) != 1:
        print("elige exactamente uno: --seco, --escribir o --revertir")
        return 1
    if (a.escribir or a.revertir) and not a.confirmar:
        print("ABORTA: --escribir y --revertir tocan Qdrant (y --escribir paga embeddings); añade --confirmar "
              "sólo con el permiso de F1")
        return 1
    if a.revertir:
        return revertir(a.docs, Path(a.env))

    cat = cargar_catalogo(Path(a.catalogo))
    grupos = collections.defaultdict(set)
    pedidos = []
    for clave in a.docs:
        pedidos.append(clave)
        if not a.etiqueta:
            grupos["docs"].add(clave)
    if "linea" in a.grupo:
        for clave in LINEA + ORO_DOCS_EXTRA:
            pedidos.append(clave)
            grupos["linea" if clave in LINEA else "anclas"].add(clave)
    if "mexico" in a.grupo:
        mx = sorted((d for d in cat["docs"].values() if d["clase"] == "CC" and d.get("estado") == "México"),
                    key=lambda d: d["serie_num"])
        for d in mx:
            pedidos.append(d["doc_id"])
            grupos["mexico"].add(d["doc_id"])
    for et in a.etiqueta:
        nombre, _, lista = et.partition("=")
        for clave in lista.split(","):
            if clave:
                grupos[nombre].add(clave)
    faltan_cat, sin_pdf, elegidos = [], [], {}
    for clave in pedidos:
        did = resolver_doc_id(cat, clave)
        if not did:
            faltan_cat.append(clave)
            continue
        d = cat["docs"][did]
        nombre = d["url_oficial"].rsplit("/", 1)[1].lower()
        ruta = next((Path(c) / nombre for c in a.pdfs if (Path(c) / nombre).exists()), None)
        if ruta is None:
            # Las copias de trabajo de C-217, C-218, C-221 y C-253 se bajaron como
            # «seriec_N_esp.pdf», pero el catálogo enlaza «…_esp1.pdf»/«…_esp2.pdf».
            # En seco se usan, marcadas: su sha1/bytes/páginas NO están comprobados
            # contra url_oficial. --escribir las rechaza (ver abajo).
            alterno = re.sub(r"_esp\d\.pdf$", "_esp.pdf", nombre)
            ruta = next((Path(c) / alterno for c in a.pdfs if alterno != nombre and (Path(c) / alterno).exists()), None)
            if ruta is not None:
                if a.aceptar_alterno:
                    # 25-sep-2026: Cloudflare de la Corte ya reta toda descarga
                    # automática, así que el «_esp1» del catálogo no se puede
                    # bajar. El «_esp» local también es un PDF oficial —se bajó
                    # de su propia URL en la Corte— y es el archivo contra el
                    # que se verificaron las páginas: se enlaza ESE, para que el
                    # visor abra justo el PDF cuyas páginas se mapearon. La URL
                    # del listado queda en url_catalogo.
                    d = dict(d, url_catalogo=d["url_oficial"],
                             url_oficial=d["url_oficial"].rsplit("/", 1)[0] + "/" + alterno)
                else:
                    d = dict(d, _archivo_alterno=alterno)
        if ruta is None:
            sin_pdf.append(did)
            continue
        elegidos[did] = (d, ruta)
    if faltan_cat:
        print("no están en el catálogo:", faltan_cat)

    resultados = {}
    for did, (d, ruta) in elegidos.items():
        try:
            resultados[did] = procesar_documento(d, ruta, cat)
        except Exception as e:     # un PDF roto no tumba la corrida: cuarentena
            resultados[did] = dict(doc_id=did, archivo=ruta.name, cuarentena=[f"error al trocear: {e!r}"],
                                   avisos=[], unidades=0, tokens=0, _puntos=[])
        r = resultados[did]
        print(f"{did:24s} {r.get('paginas', 0):4d} págs {r['unidades']:5d} u {r['tokens']:8d} tok"
              f"  {'CUARENTENA ' + '; '.join(r['cuarentena'])[:150] if r['cuarentena'] else 'ok'}", flush=True)

    # citado_por_n dentro de lo procesado (se recalcula en cada corrida, plan 3.5)
    entrantes = collections.Counter()
    for r in resultados.values():
        for p in r["_puntos"]:
            for ll in set(p["payload"]["cita_a"]):
                if not ll.startswith(p["payload"]["doc_id"] + "|"):
                    entrantes[ll] += 1
    for r in resultados.values():
        for p in r["_puntos"]:
            if p["payload"]["sub"] == 0:
                p["payload"]["citado_por_n"] = entrantes.get(p["payload"]["llave"], 0)

    oro = evaluar_oro(resultados, cat)

    def de_grupo(g):
        ids = {resolver_doc_id(cat, c) for c in grupos[g]}
        return [resultados[i] for i in ids if i in resultados]

    def resumen(rs):
        ok = [r for r in rs if not r["cuarentena"]]
        todas = [p["payload"] for r in rs for p in r["_puntos"]]
        mayor = max(todas, key=lambda p: p["n_tokens"]) if todas else None
        return dict(documentos=len(rs), en_cuarentena=len(rs) - len(ok),
                    unidades=sum(r["unidades"] for r in rs), unidades_escribibles=sum(r.get("unidades_vectorizables", 0) for r in ok),
                    tokens=sum(r["tokens"] for r in rs), tokens_escribibles=sum(r.get("tokens_vectorizables", 0) for r in ok),
                    unidad_mas_larga=(dict(llave=mayor["llave"], sub=mayor["sub"], tokens=mayor["n_tokens"]) if mayor else None))

    todos = list(resultados.values())
    tot = resumen(todos)
    tok_doc = [r["tokens"] for r in todos if r["tokens"]]
    informe = dict(
        cuando=time.strftime("%Y-%m-%d %H:%M"), parser_version=PARSER, metodo_tokens=METODO_TOKENS,
        coleccion=COLECCION, alias=ALIAS, marca=MARCA, catalogo=str(a.catalogo),
        pedidos=len(set(pedidos)), no_en_catalogo=faltan_cat, sin_pdf_local=sorted(set(sin_pdf)),
        total=tot, por_grupo={g: resumen(de_grupo(g)) for g in sorted(grupos)},
        tokens_por_documento=dict(media=round(statistics.mean(tok_doc)) if tok_doc else 0,
                                  mediana=round(statistics.median(tok_doc)) if tok_doc else 0),
        costo_embeddings_usd=round(tot["tokens_escribibles"] * PRECIO_MTOK / 1e6, 4),
        oro=oro,
        cuarentena={r["doc_id"]: r["cuarentena"] for r in todos if r["cuarentena"]},
        documentos=[{k: v for k, v in r.items() if k != "_puntos"} for r in todos],
    )
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    Path(a.salida).write_text(json.dumps(informe, ensure_ascii=False, indent=1))
    n_escritos = 0
    with open(a.puntos, "w", encoding="utf-8") as f:
        for r in todos:
            if r["cuarentena"]:
                continue
            for p in r["_puntos"]:
                if p["payload"]["qc"].get("vectorizar"):
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
                    n_escritos += 1
    print(f"\nTOTAL {tot['documentos']} docs ({tot['en_cuarentena']} en cuarentena), {tot['unidades']:,} unidades, "
          f"{tot['tokens']:,} tokens ({METODO_TOKENS}); escribibles {n_escritos:,} puntos, "
          f"{tot['tokens_escribibles']:,} tokens ≈ {informe['costo_embeddings_usd']} USD")
    print(f"ORO {oro['aciertos']}/{oro['total']}; extra {[(e['llave'], e['ok']) for e in oro['extra']]}")
    if sin_pdf:
        print("sin PDF local:", sorted(set(sin_pdf)))
    if a.seco:
        return 0
    return escribir({k: v for k, v in resultados.items()}, Path(a.env), Path("respaldos_ingesta_coidh"))


if __name__ == "__main__":
    sys.exit(main())
