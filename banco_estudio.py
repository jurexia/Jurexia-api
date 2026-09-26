# -*- coding: utf-8 -*-
"""EL BANCO DEL ESTUDIO — variantes del prompt, generadas en producción.

POR QUÉ EXISTE. David, 26-sep-2026, al aprobar la propuesta del estudio de
fondo: «Si adelante autorizo» (la evaluación) y «Si acepto» (el plan). La
propuesta (w2_final §6) pone una condición antes de tocar el prompt: medir, con
el taller DE VERDAD, cuánto cambia el estudio entre una variante y otra, y
cuánto cambia entre dos corridas de la misma —no hay semilla: `llamada_modelo`
quita `seed` y `temperature` (L8)—. Sin esa banda de ruido, cualquier mejora es
una corrida con suerte.

QUÉ HACE. Pide el proyecto a `/taller/resolver/stream` —el mismo camino que la
pantalla, con FormData— para cada caso, variante y corrida, y guarda en
`bancos/estudio/<etiqueta>/<caso>/<variante>_<k>.json`:
  · el texto del .docx (y el .docx al lado, para la lectura ciega de David);
  · el estudio crudo reconstruido de los eventos «texto»;
  · avisos, huecos, versión, tiempos, y la variante y el commit que devuelva
    el evento «listo» (los añade el Paso 0 del servidor; si no vienen, la
    corrida de una variante que no sea «prod» se DESCARTA: no se sabe qué
    prompt corrió).

LAS REGLAS, cada una con su porqué:
  · SÓLO CUENTAS DE CASA. `variante_estudio` sólo la acepta el servidor de
    cuentas de casa, y un banco no gasta la cuota ni el historial de un
    secretario. La lista está escrita aquí y no se amplía por parámetro.
  · EN SECUENCIA DENTRO DE UN EXPEDIENTE, EN PARALELO ENTRE EXPEDIENTES. Las
    corridas de un mismo asunto escriben la misma fila de `taller_sesiones`
    (leer-modificar-escribir del `estado` entero): dos a la vez se pisan.
  · LAS VARIANTES SE INTERCALAN (v1#1, v2#1, v1#2, v2#2…). Si a media tanda
    cambia algo en producción —un despliegue, la hora pico del proveedor—
    afecta a las dos por igual en vez de caerle entera a la segunda.
  · REANUDABLE. Lo que ya está bien guardado no se repite; lo que falló o se
    descartó, sí.
  · 900 s DE TOPE por corrida: el estudio con razonamiento alto tarda 70-150 s
    y la recomposición otros 30; lo que pasa de quince minutos está colgado.
  · NUNCA SE PIDE /taller/proponer. En modo `acervo` el criterio sale de la
    propuesta ya guardada en la sesión; volver a proponer cambiaría el
    criterio entre variantes y ya no se compararía el prompt.

Uso (nunca sin --dry-run hasta que el servidor acepte `variante_estudio`):
    .venv/bin/python banco_estudio.py --variantes v1,v2 --corridas 3 --dry-run
    .venv/bin/python banco_estudio.py --variantes v1,v2 --corridas 3 \\
        --formato estandar --casos 103/2025,93/2026 --paralelo 3
    Variantes: «prod» = no se manda el campo (lo que haya en producción);
    «v1@r» = se pide v1 y se guarda aparte, para la banda de ruido v1 contra v1'.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import datetime as dt
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

BASE = "https://jurexia-api.onrender.com"
RUTA_STREAM = "/taller/resolver/stream"
TIMEOUT_S = 900
AQUI = Path(__file__).parent / "bancos" / "estudio"

# ═══ LAS CUENTAS DE CASA — la única lista. No se amplía por parámetro. ═══
ADMIN = "administracion@iurexia.com"
SOPORTE = "soporte@iurexia.com"
CASA = frozenset({ADMIN, SOPORTE})

CASOS_KINGSTON = Path("/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/"
                      "redactor-sentencias/corpus/casos")
DIR93 = Path("/private/tmp/claude-501/-Users-josedavidalcantarmendoza-Documents-"
             "Viaje-a-Europa/5f71a5c8-bc09-427e-90e2-108495d0f272/scratchpad/93")


class CuentaAjena(ValueError):
    """Se intentó correr el banco con una cuenta que no es de casa."""


def es_de_casa(correo: str) -> bool:
    return (correo or "").strip().lower() in CASA


def exigir_casa(correo: str) -> str:
    c = (correo or "").strip().lower()
    if c not in CASA:
        raise CuentaAjena(f"«{correo}» no es una cuenta de casa: el banco sólo corre "
                          f"con {', '.join(sorted(CASA))}.")
    return c


# ═══════════════════════════════════════════════════════════════════════════
# LOS CASOS
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class Caso:
    numero: str
    correo: str
    modo: str                       # acervo | por_problema
    kingston: str = ""              # archivo del corpus: escrito (y oro)
    criterios: str = ""             # por_problema: criterios_json
    global_: str = ""               # por_problema: global_json
    contexto: str = ""              # por_problema: contexto
    escrito: str = ""               # si el escrito no es el del corpus
    nota: str = ""
    # El «oro» del corpus para el 93/2026 es el mismo texto que entregó el
    # taller, no una corrección de David (w2_evaluacion, hallazgo 8): no es
    # referencia humana y no se compara contra él.
    con_oro: bool = True

    @property
    def slug(self) -> str:
        return self.numero.replace("/", "-")


# Sesiones reales de las cuentas de casa (w2_evaluacion §2.2). Las de
# administración salieron de `banco_kingston.py` con el escrito del corpus, así
# que su escrito es `piezas.demanda.texto`; el 93/2026 es el de soporte, con el
# criterio que fijó el secretario (crit_v6) y la demanda tal como entró.
CASOS = [
    Caso("103/2025", ADMIN, "acervo", "ADA_103_2025_159e66.json",
         nota="reiteración entre conceptos 2, 3 y 4"),
    Caso("642/2024", ADMIN, "acervo",
         "ADC_642_2024_ORD_CIVIL_SOBRE_REIVINDICACIO_N_8eff8c.json",
         nota="mismo tema con razón distinta; el oro no es el techo"),
    Caso("174/2026", ADMIN, "acervo", "ADC_174_2026_aabe46.json",
         nota="concepto único con dato propio"),
    Caso("722/2025", ADMIN, "acervo", "ADC_722_2025_con_adhesivo_26052f.json",
         nota="adhesivo y procesal"),
    Caso("192/2025", ADMIN, "acervo", "ADC_192_2025_EN_CUMPLIMIENTO__8c5489.json",
         nota="en cumplimiento, n=0"),
    Caso("263/2025", ADMIN, "acervo", "ADA_263_2025_AGRARIO_dee9f5.json",
         nota="suplencia; el engrose que más remite"),
    Caso("640/2024", ADMIN, "acervo",
         "ADC_640_2024_RECONOCIMIENTO_DE_PATERNIDAD_3030ef.json",
         nota="familiar, menores, adhesivo"),
    Caso("526/2024", ADMIN, "acervo", "ADC_526_2024_ORD_CIVIL_DIVORCIO_ffa173.json",
         nota="demanda en varias piezas"),
    Caso("481/2024", ADMIN, "acervo", "ADC_481_2024_352823.json",
         nota="valoración de pruebas, niega"),
    Caso("43/2025", ADMIN, "acervo", "ADC_43_2025_120576.json",
         nota="concede, efectos"),
    Caso("274/2025", ADMIN, "acervo", "2_ADC_274_2025_ac6a60.json",
         nota="control: corto"),
    Caso("702/2022", ADMIN, "acervo", "ADA_702_2022_INFRACCIO_N_4e4f48.json",
         nota="control: concepto único, 9 tesis"),
    Caso("810/2025", ADMIN, "acervo", "ADC_810_2025_PRESC_POSITIVA_38c016.json",
         nota="muy largo: truncamiento"),
    Caso("93/2026", SOPORTE, "por_problema", "ADA_93_2026_115abc.json",
         criterios=str(DIR93 / "crit_v6.json"), global_=str(DIR93 / "global2.json"),
         contexto=str(DIR93 / "contexto_v6.txt"), escrito=str(DIR93 / "fuente1.txt"),
         nota="violación procesal, n=0, criterio real del secretario", con_oro=False),
]
POR_NUMERO = {c.numero: c for c in CASOS}


def normalizar_numero(x: str) -> str:
    """«103-2025», «103/2025», «ADA 103-2025» → «103/2025»."""
    m = re.search(r"(\d{1,4})\s*[-/]\s*(20\d{2})", x or "")
    return f"{m.group(1)}/{m.group(2)}" if m else (x or "").strip()


def elegir_casos(lista: str) -> list:
    if not (lista or "").strip():
        return list(CASOS)
    fuera = []
    for x in lista.split(","):
        n = normalizar_numero(x)
        if n not in POR_NUMERO:
            raise SystemExit(f"Caso desconocido: «{x}». Los del banco: "
                             f"{', '.join(POR_NUMERO)}")
        fuera.append(POR_NUMERO[n])
    return fuera


# ── LAS ENTRADAS DEL 93 SE CONGELAN ───────────────────────────────────────
# Viven en el scratchpad de la sesión del diagnóstico, que el sistema puede
# borrar. La primera vez que se usan se copian a `bancos/estudio/_entradas/`
# y desde entonces se leen de ahí: el criterio del secretario no puede
# desaparecer a media evaluación.
def entrada(caso: Caso, ruta: str, raiz: Path = AQUI) -> Path:
    if not ruta:
        return Path("")
    src = Path(ruta)
    dst = raiz / "_entradas" / caso.slug / src.name
    if not dst.exists() and src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return dst if dst.exists() else src


def escrito_de(caso: Caso, raiz: Path = AQUI) -> str:
    """El texto de la demanda con que se mide la cobertura."""
    if caso.escrito:
        p = entrada(caso, caso.escrito, raiz)
        if p.exists():
            return p.read_text(encoding="utf-8")
    if caso.kingston:
        p = CASOS_KINGSTON / caso.kingston
        if p.exists():
            c = json.loads(p.read_text(encoding="utf-8"))
            return ((c.get("piezas") or {}).get("demanda") or {}).get("texto", "")
    return ""


def oro_de(caso: Caso) -> str:
    if caso.kingston and caso.con_oro:
        p = CASOS_KINGSTON / caso.kingston
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8")).get("oro") or ""
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# VARIANTES
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Variante:
    etiqueta: str               # con la que se guarda: «v1», «v1@r», «prod»
    servidor: str | None        # la que se pide: «v1»; None = no se manda


def variantes(lista: str) -> list:
    fuera, vistas = [], set()
    for x in (lista or "").split(","):
        x = x.strip()
        if not x:
            continue
        if not re.fullmatch(r"[A-Za-z0-9_.-]+(?:@[A-Za-z0-9_.-]+)?", x):
            raise SystemExit(f"Variante inválida: «{x}»")
        if x in vistas:
            continue
        vistas.add(x)
        if x == "prod":
            fuera.append(Variante("prod", None))
        else:
            fuera.append(Variante(x, x.split("@", 1)[0]))
    if not fuera:
        raise SystemExit("Hace falta al menos una variante (--variantes v1,v2).")
    return fuera


# ═══════════════════════════════════════════════════════════════════════════
# EL FORMULARIO
# ═══════════════════════════════════════════════════════════════════════════
def formulario(caso: Caso, formato: str, variante: Variante, raiz: Path = AQUI) -> dict:
    """Los campos de la FormData, como los manda la pantalla (o
    `scratchpad/93/generar_formatos.sh` para el 93)."""
    correo = exigir_casa(caso.correo)
    f = {"numero": caso.numero, "user_email": correo, "formato": formato,
         "modo_decision": caso.modo}
    if variante.servidor is not None:
        f["variante_estudio"] = variante.servidor
    if caso.modo == "por_problema":
        crit = entrada(caso, caso.criterios, raiz)
        glob_ = entrada(caso, caso.global_, raiz)
        ctx = entrada(caso, caso.contexto, raiz)
        faltan = [str(p) for p in (crit, glob_, ctx) if not p.exists()]
        if faltan:
            raise FileNotFoundError(f"{caso.numero}: faltan {', '.join(faltan)}")
        g_txt = glob_.read_text(encoding="utf-8")
        f["criterios_json"] = crit.read_text(encoding="utf-8")
        f["global_json"] = g_txt
        f["contexto"] = ctx.read_text(encoding="utf-8")
        # Igual que generar_formatos.sh: lo que resolvió el órgano, del global.
        f["resolvio_declarado"] = str(
            ((json.loads(g_txt) or {}).get("contexto") or {}).get("resolvio") or "")
    return f


# ═══════════════════════════════════════════════════════════════════════════
# EL FLUJO SSE
# ═══════════════════════════════════════════════════════════════════════════
class LectorSSE:
    """Líneas de `text/event-stream` → eventos (dict), de una en una.

    Los «: latido» se saltan; un evento puede traer varias líneas `data:` y se
    juntan; la línea en blanco lo cierra. Lo usan la corrida en vivo y las
    pruebas, para que las dos lean el flujo igual."""

    def __init__(self):
        self.datos = []

    def linea(self, ln: str):
        ln = (ln or "").rstrip("\r\n")
        if ln == "":
            return self.cerrar()
        if ln.startswith("data:"):
            self.datos.append(ln[5:].lstrip(" "))
        return None

    def cerrar(self):
        if not self.datos:
            return None
        e, self.datos = _evento("\n".join(self.datos)), []
        return e


def eventos_sse(lineas) -> list:
    lector, fuera = LectorSSE(), []
    for ln in lineas:
        e = lector.linea(ln)
        if e is not None:
            fuera.append(e)
    e = lector.cerrar()
    return fuera + ([e] if e is not None else [])


def _evento(crudo: str) -> dict:
    try:
        e = json.loads(crudo)
        return e if isinstance(e, dict) else {"tipo": "?", "crudo": crudo[:300]}
    except Exception:
        return {"tipo": "?", "crudo": crudo[:300]}


class Acumulador:
    """Lo que se va sabiendo de una corrida mientras llegan los eventos."""

    def __init__(self, t0: float = None):
        self.t0 = time.time() if t0 is None else t0
        self.trozos = []
        self.tipos = Counter()
        self.listo = None
        self.error = None
        self.t_primer_texto = None
        self.t_componiendo = None
        self.t_listo = None

    def meter(self, e: dict, ahora: float = None) -> None:
        ahora = time.time() if ahora is None else ahora
        t = e.get("tipo")
        self.tipos[t] += 1
        if t == "texto":
            if self.t_primer_texto is None:
                self.t_primer_texto = round(ahora - self.t0, 1)
            self.trozos.append(str(e.get("dato") or ""))
        elif t == "componiendo":
            self.t_componiendo = round(ahora - self.t0, 1)
        elif t == "listo":
            self.listo = e
            self.t_listo = round(ahora - self.t0, 1)
        elif t == "error":
            self.error = str(e.get("mensaje") or "error sin mensaje")[:500]

    @property
    def estudio_crudo(self) -> str:
        return "".join(self.trozos)


def texto_docx(datos: bytes) -> str:
    """El texto del .docx como lo sacaba el diagnóstico (`gen_*.txt`): los
    párrafos no vacíos, uno por línea. Así las cifras de ayer y las de hoy
    salen de la misma extracción."""
    import docx
    d = docx.Document(io.BytesIO(datos))
    return "\n".join(p.text for p in d.paragraphs if p.text.strip())


# Los campos del «listo» que no son el documento se guardan tal cual: el
# Paso 0 del servidor añade variante, commit, tokens y finish_reason, y lo que
# añada después también debe quedar sin tocar este archivo.
_NO_GUARDAR = {"docx_b64"}


def resultado(acc: Acumulador, caso: Caso, var: Variante, k: int, formato: str,
              etiqueta: str, http_status: int = None) -> tuple:
    """(fila para el .json, bytes del .docx o None)."""
    fila = {
        "caso": caso.numero, "correo": caso.correo, "modo": caso.modo,
        "variante": var.etiqueta, "variante_pedida": var.servidor,
        "formato": formato, "k": k, "etiqueta": etiqueta,
        "http": http_status,
        "t_primer_texto": acc.t_primer_texto, "t_componiendo": acc.t_componiendo,
        "t_listo": acc.t_listo, "t_total": round(time.time() - acc.t0, 1),
        "eventos": dict(acc.tipos),
        "estudio_crudo": acc.estudio_crudo,
        "ok": False, "error": acc.error, "descartada": None,
    }
    docx_bytes = None
    if acc.listo is not None:
        ev = acc.listo
        fila["listo"] = {k2: v for k2, v in ev.items() if k2 not in _NO_GUARDAR}
        for campo in ("avisos", "huecos", "palabras", "version", "tiempos",
                      "advertencias", "nombre"):
            fila[campo] = ev.get(campo)
        fila["variante_servidor"] = ev.get("variante") or ev.get("variante_estudio")
        fila["commit"] = ev.get("commit")
        b64 = ev.get("docx_b64") or ""
        if b64:
            try:
                docx_bytes = base64.b64decode(b64)
                fila["texto"] = texto_docx(docx_bytes)
            except Exception as ex:
                fila["error"] = f"el .docx no se pudo leer: {type(ex).__name__}: {ex}"[:300]
        else:
            fila["error"] = fila["error"] or "el evento «listo» no trajo el .docx"
        if not fila["error"]:
            fila["ok"] = True
        fila["descartada"] = motivo_de_descarte(var, fila.get("variante_servidor"))
    elif not fila["error"]:
        fila["error"] = "el flujo terminó sin evento «listo»"
    return fila, docx_bytes


def motivo_de_descarte(var: Variante, devuelta) -> str | None:
    """Una corrida cuya variante no se puede comprobar no cuenta.

    `prod` no pide nada y acepta lo que haya. Cualquier otra exige que el
    servidor diga cuál corrió: si calla, es un servidor sin el Paso 0 y lo que
    corrió fue el prompt de producción con la etiqueta de otra variante.
    """
    if var.servidor is None:
        return None
    if not devuelta:
        return "el servidor no devolvió la variante: no se sabe qué prompt corrió"
    if str(devuelta).strip().lower() != var.servidor.strip().lower():
        return f"se pidió «{var.servidor}» y corrió «{devuelta}»"
    return None


# ═══════════════════════════════════════════════════════════════════════════
# DÓNDE SE GUARDA, Y QUÉ FALTA
# ═══════════════════════════════════════════════════════════════════════════
def ruta_corrida(raiz: Path, etiqueta: str, caso: Caso, var: Variante, k: int) -> Path:
    return raiz / etiqueta / caso.slug / f"{var.etiqueta}_{k}.json"


def hecha(ruta: Path) -> bool:
    """Bien guardada y no descartada. Lo demás se vuelve a correr."""
    if not ruta.exists():
        return False
    try:
        f = json.loads(ruta.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(f.get("ok")) and not f.get("descartada")


def guardar(ruta: Path, fila: dict, docx_bytes: bytes = None) -> None:
    """Escritura atómica: un corte a medias no deja un .json que parezca bueno."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(fila, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, ruta)
    if docx_bytes:
        ruta.with_suffix(".docx").write_bytes(docx_bytes)


def plan(casos: list, vars_: list, corridas: int, etiqueta: str, raiz: Path = AQUI) -> dict:
    """{caso: [(variante, k, ruta, hecha)]}, en el orden en que se correrán:
    por corrida y, dentro de cada una, variante tras variante (intercaladas)."""
    fuera = {}
    for c in casos:
        exigir_casa(c.correo)
        fila = []
        for k in range(1, corridas + 1):
            for v in vars_:
                r = ruta_corrida(raiz, etiqueta, c, v, k)
                fila.append((v, k, r, hecha(r)))
        fuera[c.numero] = fila
    return fuera


# ═══════════════════════════════════════════════════════════════════════════
# LA HUELLA DE LA SESIÓN (opcional: --congelar)
# ═══════════════════════════════════════════════════════════════════════════
# El criterio de una corrida en modo `acervo` sale de lo que la sesión guardó:
# problemas, resúmenes, conteo y propuestas. Si alguien vuelve a proponer a
# media tanda, la variante B se escribe con otro criterio que la A y la
# comparación deja de ser del prompt. Se guarda sólo el HASH —no el estado,
# que lleva el expediente— y se comprueba antes de cada corrida.
def huella_de(fases: dict, propuestas) -> str:
    f = fases or {}
    base = {"problemas": f.get("problemas"), "problema_global": f.get("problema_global"),
            "resumen_acto": f.get("resumen_acto"),
            "resumen_conceptos": f.get("resumen_conceptos"), "conteo": f.get("conteo"),
            "propuestas": propuestas}
    return hashlib.sha256(json.dumps(base, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()).hexdigest()[:16]


async def huella_sesion(cx, correo: str, numero: str) -> str | None:
    url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_KEY") or ""
    if not (url and key):
        return None
    r = await cx.get(f"{url}/rest/v1/taller_sesiones",
                     params={"select": "fases:estado->fases,propuestas",
                             "email": f"eq.{exigir_casa(correo)}",
                             "expediente": f"eq.{numero}", "limit": "1"},
                     headers={"apikey": key, "Authorization": f"Bearer {key}"},
                     timeout=30)
    r.raise_for_status()
    filas = r.json() or []
    if not filas:
        return None
    return huella_de(filas[0].get("fases"), filas[0].get("propuestas"))


# ═══════════════════════════════════════════════════════════════════════════
# CORRER
# ═══════════════════════════════════════════════════════════════════════════
async def correr_una(cx, base: str, caso: Caso, var: Variante, k: int, formato: str,
                     etiqueta: str, raiz: Path = AQUI, timeout: float = TIMEOUT_S) -> tuple:
    """Una corrida: (fila para el .json, bytes del .docx o None). Nunca lanza:
    cualquier fallo queda escrito en la fila como error."""
    acc = Acumulador()
    status = None
    try:
        campos = formulario(caso, formato, var, raiz)
    except Exception as ex:
        acc.error = f"formulario: {type(ex).__name__}: {str(ex)[:300]}"
        return resultado(acc, caso, var, k, formato, etiqueta, None)

    async def _flujo():
        nonlocal status
        # FormData de verdad (multipart), como la pantalla: los tres campos
        # largos del 93 pasan de 2 KB y así viajan igual que en producción.
        partes = [(k2, (None, v)) for k2, v in campos.items()]
        async with cx.stream("POST", base.rstrip("/") + RUTA_STREAM, files=partes) as r:
            status = r.status_code
            if r.status_code != 200:
                cuerpo = (await r.aread()).decode("utf-8", "replace")
                acc.error = f"HTTP {r.status_code}: {cuerpo[:300]}"
                return
            lector = LectorSSE()
            async for ln in r.aiter_lines():
                e = lector.linea(ln)
                if e is not None:
                    acc.meter(e)
            e = lector.cerrar()
            if e is not None:
                acc.meter(e)

    try:
        await asyncio.wait_for(_flujo(), timeout=timeout)
    except asyncio.TimeoutError:
        acc.error = acc.error or f"sin evento «listo» a los {int(timeout)} s"
    except Exception as ex:
        acc.error = acc.error or f"{type(ex).__name__}: {str(ex)[:300]}"
    fila, docx_bytes = resultado(acc, caso, var, k, formato, etiqueta, status)
    return fila, docx_bytes


def _linea(fila: dict) -> str:
    if fila.get("ok") and not fila.get("descartada"):
        marca = "✓"
    elif fila.get("descartada"):
        marca = "≠"
    else:
        marca = "✗"
    extra = (fila.get("descartada") or fila.get("error") or "")[:90]
    return (f"  {marca} {fila['caso']:<9} {fila['variante']:<7} #{fila['k']} "
            f"{fila.get('t_total', 0):>6.0f}s  {fila.get('palabras') or '—':>5} pal "
            f"v{fila.get('version') or '—'} {(fila.get('commit') or '')[:8]} {extra}")


async def correr(casos: list, vars_: list, corridas: int, formato: str, etiqueta: str,
                 paralelo: int = 3, base: str = BASE, raiz: Path = AQUI,
                 congelar: bool = False, cliente=None, timeout: float = TIMEOUT_S,
                 imprimir=print) -> list:
    import httpx
    p = plan(casos, vars_, corridas, etiqueta, raiz)
    pend = sum(1 for xs in p.values() for x in xs if not x[3])
    imprimir(f"═══ banco del estudio «{etiqueta}» · {formato} · {len(casos)} casos · "
             f"{', '.join(v.etiqueta for v in vars_)} × {corridas} · {pend} por correr · "
             f"{paralelo} en paralelo ═══")
    _manifiesto(raiz, etiqueta, casos, vars_, corridas, formato, base)
    if congelar and not (os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_KEY")):
        imprimir("  ! --congelar sin SUPABASE_URL / SUPABASE_SERVICE_KEY: la sesión NO se "
                 "comprueba entre corridas")
    sem = asyncio.Semaphore(max(1, paralelo))
    hechas = []
    propio = cliente is None
    cx = cliente or httpx.AsyncClient(timeout=httpx.Timeout(30, read=timeout, write=60))

    async def _caso(c: Caso):
        # UN EXPEDIENTE, UNA CORRIDA A LA VEZ: este bucle es secuencial y el
        # semáforo sólo reparte turnos entre expedientes.
        huella0 = None
        f_huella = raiz / etiqueta / c.slug / "huella.txt"
        if f_huella.exists():
            huella0 = f_huella.read_text(encoding="utf-8").strip()
        for v, k, ruta, ya in p[c.numero]:
            if ya:
                continue
            async with sem:
                if congelar:
                    try:
                        h = await huella_sesion(cx, c.correo, c.numero)
                    except Exception as ex:
                        h = None
                        imprimir(f"  ! {c.numero}: no se pudo leer la sesión "
                                 f"({type(ex).__name__}); se corre sin comprobar")
                    if h and huella0 and h != huella0:
                        imprimir(f"  ✗ {c.numero}: LA SESIÓN CAMBIÓ desde la primera "
                                 f"corrida de «{etiqueta}» ({huella0} → {h}). El criterio "
                                 f"ya no es el mismo; usa otra etiqueta. Se para este caso.")
                        return
                    if h and not huella0:
                        f_huella.parent.mkdir(parents=True, exist_ok=True)
                        f_huella.write_text(h, encoding="utf-8")
                        huella0 = h
                else:
                    h = None
                fila, docx_bytes = await correr_una(cx, base, c, v, k, formato, etiqueta,
                                                    raiz, timeout)
            fila["huella_sesion"] = h
            fila["t_inicio"] = dt.datetime.now().isoformat(timespec="seconds")
            guardar(ruta, fila, docx_bytes)
            hechas.append(fila)
            imprimir(_linea(fila))
            # Un 404/409/422 es la sesión, no la suerte: las demás corridas de
            # este expediente fallarían igual.
            if fila.get("http") and 400 <= int(fila["http"]) < 500:
                imprimir(f"  ✗ {c.numero}: HTTP {fila['http']}: se para este caso.")
                return

    try:
        await asyncio.gather(*(_caso(c) for c in casos))
    finally:
        if propio:
            await cx.aclose()
    buenas = sum(1 for f in hechas if f.get("ok") and not f.get("descartada"))
    imprimir(f"═══ {buenas} buenas · {sum(1 for f in hechas if f.get('descartada'))} "
             f"descartadas · {sum(1 for f in hechas if not f.get('ok'))} con error ═══")
    return hechas


def _manifiesto(raiz: Path, etiqueta: str, casos, vars_, corridas, formato, base) -> None:
    ruta = raiz / etiqueta / "manifiesto.json"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    try:
        m = json.loads(ruta.read_text(encoding="utf-8"))
    except Exception:
        m = {"invocaciones": []}
    m["formato"] = formato
    m["invocaciones"].append({
        "fecha": dt.datetime.now().isoformat(timespec="seconds"), "base": base,
        "casos": [c.numero for c in casos], "variantes": [v.etiqueta for v in vars_],
        "corridas": corridas})
    ruta.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# EN SECO
# ═══════════════════════════════════════════════════════════════════════════
# Lo que tarda una corrida de punta a punta (estudio 70-150 s + recomposición
# ~30 s, medido en las del 93/2026 del 25-sep). Sólo para avisar en seco.
SEGUNDOS_POR_CORRIDA = 200


def en_seco(casos: list, vars_: list, corridas: int, formato: str, etiqueta: str,
            raiz: Path = AQUI, base: str = BASE, paralelo: int = 3,
            imprimir=print) -> dict:
    """Lo que se correría, sin tocar la red. Comprueba cuentas, entradas y
    escritos, y enseña los campos del formulario (claves y tamaños, no el
    contenido: el criterio y el contexto son del expediente)."""
    p = plan(casos, vars_, corridas, etiqueta, raiz)
    imprimir(f"═══ EN SECO · {base}{RUTA_STREAM} · «{etiqueta}» · {formato} ═══")
    problemas = []
    for c in casos:
        filas = p[c.numero]
        pend = [f"{v.etiqueta}#{k}" for v, k, _, ya in filas if not ya]
        esc = escrito_de(c, raiz)
        try:
            campos = formulario(c, formato, vars_[0], raiz)
            txt = ", ".join(f"{k2}({len(v)})" if len(v) > 40 else f"{k2}={v}"
                            for k2, v in campos.items())
        except Exception as ex:
            txt = f"FORMULARIO IMPOSIBLE: {ex}"
            problemas.append(f"{c.numero}: {ex}")
        if not esc:
            problemas.append(f"{c.numero}: sin escrito para medir la cobertura")
        imprimir(f"  {c.numero:<9} {c.correo:<27} {c.modo:<12} escrito "
                 f"{len(esc.split()):>6} pal · {len(filas) - len(pend)}/{len(filas)} hechas · "
                 f"por correr: {' '.join(pend) or '—'}")
        imprimir(f"            {txt}")
    total = sum(1 for xs in p.values() for x in xs if not x[3])
    mas_larga = max([sum(1 for x in xs if not x[3]) for xs in p.values()] or [0])
    minutos = max(total / max(1, paralelo), mas_larga) * SEGUNDOS_POR_CORRIDA / 60
    imprimir(f"═══ {total} corridas por hacer · ≈ {minutos:.0f} min con {paralelo} en "
             f"paralelo (un expediente nunca corre dos a la vez) ═══")
    for x in problemas:
        imprimir(f"  ! {x}")
    return {"plan": p, "problemas": problemas, "pendientes": total}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Banco del estudio: variantes del prompt "
                                             "generadas en producción.")
    ap.add_argument("--variantes", default="v1,v2")
    ap.add_argument("--corridas", type=int, default=3)
    ap.add_argument("--formato", choices=("estandar", "moderna"), default="estandar")
    ap.add_argument("--casos", default="")
    ap.add_argument("--paralelo", type=int, default=3)
    ap.add_argument("--etiqueta", default="",
                    help="carpeta dentro de bancos/estudio (por omisión, el formato)")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--congelar", action="store_true",
                    help="comprueba con la huella de la sesión (Supabase) que el "
                         "criterio no cambió entre corridas")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    casos = elegir_casos(a.casos)
    vars_ = variantes(a.variantes)
    etiqueta = a.etiqueta or a.formato
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", etiqueta):
        raise SystemExit(f"Etiqueta inválida: «{etiqueta}»")
    if a.corridas < 1:
        raise SystemExit("--corridas tiene que ser 1 o más")
    if a.dry_run:
        r = en_seco(casos, vars_, a.corridas, a.formato, etiqueta, AQUI, a.base,
                    a.paralelo)
        return 1 if r["problemas"] else 0
    asyncio.run(correr(casos, vars_, a.corridas, a.formato, etiqueta, a.paralelo,
                       a.base, AQUI, a.congelar))
    return 0


if __name__ == "__main__":
    sys.exit(main())
