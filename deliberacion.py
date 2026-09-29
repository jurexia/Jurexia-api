# -*- coding: utf-8 -*-
"""LA DELIBERACIÓN DEL PROBLEMA PRINCIPAL — detrás de bandera, APAGADA.

POR QUÉ EXISTE. David, 28-sep-2026, al pedir la tarjeta «El problema principal y
su solución»: «Aquí el punto es acercarnos a la solución certera. ¿Cómo le damos
esa potencia de inteligencia para resolver un problema jurídico? ¿Cómo la
maximizamos?».

LO MEDIDO QUE MANDA EL DISEÑO (lector2_potencia.md, 28-sep-2026):
  · Redactar con el criterio dado ya funciona: 9 de 10 lentes a ciegas.
  · DECIDIR es el cuello de botella. Kingston-24 (engroses reales de amparo
    directo): 50, 48, 42 y 46 % en cuatro corridas, contra el 54 % de contestar
    siempre «se niega». El error es ESTABLE —15 asuntos salen igual en tres
    corridas o más y sólo 7 aciertan—, así que votar entre corridas no lo
    arregla; y va sesgado a conceder en el fondo. La «confianza alta» que
    escribe el modelo no predice el acierto (4 de 8, 2 de 6).
  · La búsqueda usa la pregunta del agravio, no la figura que decide. AR
    631/2025: el motor se apoyó en dos jurisprudencias genéricas de cosa
    juzgada porque buscó «¿alteró la cosa juzgada?», no la causahabiencia o
    sustitución procesal del adquirente en la ejecución.
  · `obligatoria = bool(vincula)` (fase6_rag._tesis_de) rotula OBLIGATORIA a
    unas 5,900 jurisprudencias de colegiados y de Plenos de Circuito; el art.
    217, párrafo tercero, LA dice que la de colegiados NO obliga a otro
    colegiado. Un juez que aplique «lo que obliga manda» con ese rótulo
    decidiría mal: aquí la fuerza la calcula el código (`fuerza_para_colegiado`).

LA INTELIGENCIA NO SE MAXIMIZA CON UN MODELO MÁS CARO NI CORRIENDO MÁS VECES. Se
maximiza cambiando QUÉ se pregunta (la pregunta decisiva, no la del agravio),
CON QUÉ se decide (lo que obliga, primero y bien rotulado) y CÓMO (dos vías
argumentadas por separado y un juez con reglas de carga). Seis etapas:

  A · LA PREGUNTA DECISIVA. Una llamada barata, APARTE del contraste —y no una
      extensión de su prompt, que era la otra opción—: el contraste corre
      siempre, para todos, al terminar el adelanto, y alimenta la propuesta
      que ya está en producción; tocar su prompt cambiaría lo que ve todo el
      mundo aunque la bandera esté apagada. Una llamada propia sólo existe
      cuando la bandera está encendida y cuesta unas milésimas de dólar.
      Devuelve la figura, la pregunta que un rubro contestaría, la
      proposición toral con su cita LITERAL (se comprueba palabra por palabra
      contra el acto; la que no está se descarta) y los hechos que deciden.
  B · LA BÚSQUEDA EN ESCALERA sobre la pregunta decisiva (la búsqueda de
      siempre —conceptual, co-citación, rerank— reapuntada), más lo que ya
      trae el material del principal, más el refuerzo dirigido por vía, más lo
      de internet que el acervo confirmó. Una lectura barata dice de cada
      candidata si RESUELVE el punto, si es DISTINGUIBLE o si es ajena. Se
      ordena en escalones con cupo propio: 1) lo que OBLIGA a un colegiado,
      2) lo que orienta, 3) el propio tribunal como PRECEDENTE PROPIO (nunca
      voto), 4) las normas enteras. Catálogo CERRADO de identificadores (T, O,
      N), el patrón de `toulmin.catalogo`/`resolver`. NO se filtra por
      `vincula` ni por `instancia` en Qdrant: sin índice de payload esos
      filtros pueden ser lentos o rechazarse, y no se crean índices en
      producción; se busca sin filtro y el escalón lo pone el código.
  C · DOS ABOGADOS INDEPENDIENTES, EN PARALELO: uno argumenta la vía en que el
      planteamiento principal prospera y otro la vía en que no. Ninguno ve al
      otro ni sabe qué prefirió el motor. Regla en abstracto, hechos con cita,
      subsunción, conclusión, la mejor objeción de la otra vía y su respuesta,
      la autoridad contraria y cómo se distingue, lo que propone aplicar (sólo
      identificadores) y la suerte de cada secundario con los campos de la
      lista de comprobación, para que el árbol la verifique. Así la vía
      contraria deja de ser «el caso del perdedor escrito por el ganador»
      (regla 9 de la propuesta): se cierra por construcción la causa 1 del
      retroceso del 631, la razón que argumentaba la vía opuesta.
      Modelo de la propuesta con esfuerzo ALTO: David rechazó el medio.
  D · UN JUEZ CIEGO, DOS PASADAS CON EL ORDEN INVERTIDO para anular el sesgo de
      posición. Orden fijo de carga: 1) lo que obliga manda; 2) la vía que
      prospera combate la proposición toral con un hecho acreditado (cita
      verificada); 3) presunción de legalidad y estricto derecho, salvo
      suplencia —es el contrapeso explícito del sesgo medido a conceder—;
      4) precedente propio (apartarse exige razón); 5) tasa base sólo como
      desempate declarado. Sin porcentajes.
  E · VERIFICACIÓN DETERMINISTA antes de enseñar nada: los identificadores se
      traducen a registro, rubro, vigencia y fuerza por código; toda cifra de
      registro en el texto libre que no esté en el catálogo se QUITA (hoy
      `fase5_propuesta.revisar` sólo avisa); los hechos citados se buscan como
      subcadena en el acto, el escrito o las constancias; lo abandonado o
      sustituido no entra en «propongo aplicar». El estado «claro» lo decide el
      código, no el juez: las dos pasadas coinciden Y el escalón que invocaron
      se comprueba (lo que «obliga» es de verdad obligatorio y vigente; el
      hecho está verificado).
  F · LA CONSECUENCIA DE CADA VÍA LA CALCULA EL CÓDIGO (`tipos_asunto` y la
      rama del amparo en revisión, con el arreglo 577c700: revocar una
      concesión niega LO QUE ELLA CONCEDIÓ); y los SECUNDARIOS los resuelve
      `arbol_decision` en las dos vías, con la suerte que escribió cada abogado
      y todas sus garantías (presupuesto verificado, guarda procesal de los
      arts. 74-V, 174 y 189, tema distinto, mayor beneficio). Nadie pide una
      razón por argumento: la Decisión 6 está retirada (plan-6).

LO QUE NO HACE, y cada cosa tiene su porqué en lector2_potencia.md §riesgos:
no inventa registros ni cita de memoria; no decide por el secretario (se
proyecta en la tarjeta, que nunca genera nada sin su clic y deja «Resolver con
mi criterio» siempre a la vista); no enseña porcentajes sin calibrar; no vota
(espejo, sondeo y OAJ se leen como precedentes); no usa la tasa base como regla;
no le dice al juez qué vía prefirió el motor; no llama «obligatoria» a la
jurisprudencia de otro colegiado; no baja el esfuerzo a medio; no reintroduce la
razón por argumento secundario.

BANDERA. `DELIBERACION_ACTIVA` (apagada por omisión) Y la cuenta en
`DELIBERACION_CUENTAS` (vacía por omisión): las dos. Con la lista vacía no corre
para nadie aunque alguien encienda la bandera por error; «*» la abre a todas y
sólo se pone cuando el banco pase sus compuertas (banco_deliberacion.py) y David
lo apruebe. Con la bandera apagada, este módulo no hace una sola llamada.

DUDAS PARA DAVID (escritas aquí para que no se decidan en silencio):
  1. Los Plenos de Circuito (extintos desde 2021): su jurisprudencia sale con
     rótulo propio, «Pleno de Circuito», sin afirmar si obliga.
  2. La región del Pleno Regional de cada circuito: desde la integración
     (28-sep-2026) sale de la tabla MEDIDA de la tarjeta
     (`tarjeta_decision.REGION_DEL_CIRCUITO`: el Vigésimo Segundo, 17 de 17
     tesis del volcado del Semanario, Centro-Norte), y `DELIBERACION_REGIONES`
     («22:CS», p. ej.) la corrige si el Consejo mueve un circuito. Falta el
     otro lado: la región del PLENO sale de su clave («PR.A.C.CN. J/7 K»), y el
     payload del taller (`fase6_rag._tesis_de`) no guarda `numero_tesis`; la
     localización que sí guarda («[J]; 10a. Época; Pleno; Gaceta S.J.F.…»,
     sesión 462) no la trae. Hasta que `_tesis_de` conserve la clave, todo
     Pleno Regional sale «orienta, por confirmar» (medido sobre la sesión real
     del 631, sin red): afirmar que obliga sin saberlo es el mismo error que el
     rótulo de hoy, en la dirección que más cuesta.
  3. El precedente obligatorio de la Corte (arts. 222 y 223 LA) se reconoce
     porque se publica como jurisprudencia o como «precedente»; una aislada de
     la Corte orienta. Si hubiera precedentes obligatorios publicados como
     aislados, habría que marcarlos a mano.
  4. La tesis emitida por el propio tribunal (clave propia, p. ej.
     «XXII.3o.A.C.») se trata como precedente propio (art. 228), no como
     jurisprudencia que obligue. Se reconoce por la clave: la misma falta que
     en el punto 2. La designación del tribunal se lee de su nombre
     (`tarjeta_decision.designacion_de`) o de `DELIBERACION_CLAVE_PROPIA`.
  5. Nada de esto cambia todavía el rótulo del ESTUDIO (fase6_estudio): es
     sólo para la tarjeta y la deliberación, hasta que David lo decida.

Todo aquí es puro salvo las llamadas al modelo y a la búsqueda, que se
INYECTAN: las pruebas corren con clientes falsos y sin red
(test_deliberacion.py). El gancho en segundo plano vive en main.py
(`_taller_lanzar_deliberacion`) y la marca «deliberacion» va a la fila con su
huella, como la propuesta y el contraste (gunicorn -w 2: nada en memoria).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import unicodedata
from typing import Any, Awaitable, Callable, Optional

FORMATO = 1
VERSION = "deliberacion-1"

# ═══ LA BANDERA ══════════════════════════════════════════════════════════════
# Se leen en cada llamada, no al importar: así se enciende y se apaga sin
# reiniciar el proceso, y las pruebas pueden cambiarlas.


def _activa_global() -> bool:
    return (os.getenv("DELIBERACION_ACTIVA", "0") or "0").strip().lower() in (
        "1", "on", "si", "sí", "true", "activa")


def cuentas() -> set:
    return {c.strip().lower() for c in (os.getenv("DELIBERACION_CUENTAS", "") or "").split(",")
            if c.strip()}


def activa_para(email: str) -> bool:
    """¿Corre la deliberación para esta cuenta? Hacen falta las DOS cosas: la
    bandera encendida y la cuenta en la lista («*» = todas). Por omisión, no
    corre para nadie."""
    if not _activa_global():
        return False
    c = (email or "").strip().lower()
    lista = cuentas()
    if "*" in lista or "todas" in lista:
        return bool(c)
    return bool(c) and c in lista


def programar(email: str, lanzar: Callable[[], Any]) -> bool:
    """LA ÚNICA PUERTA DEL GANCHO. `lanzar` crea la tarea en segundo plano y
    SÓLO se llama si la bandera y la cuenta lo permiten: con la bandera
    apagada no se crea nada, no se lee nada y no se gasta nada."""
    if not activa_para(email):
        return False
    lanzar()
    return True


# ═══ LOS MODELOS ═════════════════════════════════════════════════════════════
# Abogados y juez: el modelo de la propuesta con esfuerzo ALTO. Se probó el
# medio en la propuesta (17-sep-2026, 151 s → 57 s) y David lo rechazó el mismo
# día: baja la calidad de los argumentos. No bajarlo aquí tampoco.
def _modelo() -> str:
    try:
        import fase5_propuesta as _f5
        base = _f5.MODELO_PROPUESTA
    except Exception:                                   # pragma: no cover
        base = "gpt-5.6-luna"
    return os.getenv("MODELO_DELIBERACION", "") or base


ESFUERZO_DELIBERACION = os.getenv("ESFUERZO_DELIBERACION", "high")
# La pregunta decisiva y la lectura de candidatas son de las baratas: leer y
# clasificar, no decidir. Mismo modelo y esfuerzo que el rerank de la búsqueda.
MODELO_LECTURA = os.getenv("MODELO_LECTURA_DELIBERACION",
                           os.getenv("MODELO_CONSULTA", "gpt-5.6-luna"))
ESFUERZO_LECTURA = "low"

# EL PRESUPUESTO ES COMPARTIDO CON EL RAZONAMIENTO (lección de la propuesta:
# con 4,000 y esfuerzo alto volvió vacía). Los abogados escriben más que la
# propuesta global de un problema; el juez, menos.
TOKENS_ABOGADO = int(os.getenv("DELIBERACION_TOKENS_ABOGADO", "16000"))
TOKENS_JUEZ = int(os.getenv("DELIBERACION_TOKENS_JUEZ", "12000"))
TOKENS_LECTURA = 6000

# Precios de gpt-5.6-luna por la API de OpenAI, USD por millón: los mismos que
# `inventario_escrito`. Sólo para decir cuánto costó cada deliberación.
PRECIO_ENTRADA = float(os.getenv("DELIBERACION_PRECIO_ENTRADA", "0.20"))
PRECIO_SALIDA = float(os.getenv("DELIBERACION_PRECIO_SALIDA", "1.20"))

# ═══ LOS CUPOS DE LA ESCALERA ════════════════════════════════════════════════
# Cada escalón con cupo propio, para que lo que orienta no le quite sitio a lo
# que obliga. Unos 12 T + 4 O + normas enteras (lector2_potencia §2).
CUPO_OBLIGA = 8
CUPO_ORIENTA = 6
CUPO_PROPIO = 4
CUPO_NORMAS = 8
CANDIDATAS_LECTURA = 40
TESIS_CARACTERES_ABOGADO = 1200
TESIS_CARACTERES_JUEZ = 600
# EL PRECEPTO ENTERO: un artículo recortado antes de su fracción operativa es
# un encabezado (ADL 382/2024, la fracción X del art. 47 LFT). Mismo tope que
# la propuesta.
NORMA_CARACTERES = 4000
# Ventanas del texto literal para los abogados: de ahí copian las citas.
VENTANA_ACTO = 40000
VENTANA_ESCRITO = 40000
VENTANA_CONSTANCIAS = 12000

CODIGOS_FUERZA = ("obliga", "orienta", "pleno_circuito", "precedente_propio")


# ═══ LA FUERZA DE CADA CRITERIO PARA ESTE TRIBUNAL (un colegiado) ═══════════

def _plano_mayus(x: Any) -> str:
    t = unicodedata.normalize("NFKD", str(x or ""))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.upper().split())


# UNA SOLA DEFINICIÓN DE LA FUERZA (integración del 28-sep-2026, AR 631/2025).
# La tarjeta (C1) y esta pieza (C2) escribieron cada una su
# `fuerza_para_colegiado` en paralelo; la de aquí prefería la de la tarjeta «si
# existía». Ahora hay una, en `tarjeta_decision` —módulo puro, sin `main` ni
# base—, y aquí se importa con la clave, la región de la clave y la región del
# circuito. Lo que sólo tenía ésta (la clave leída de la localización, el Pleno
# «en materia», la región y la clave propia explícitas, `DELIBERACION_REGIONES`
# y `por_confirmar`) pasó allá. La región del circuito ya no depende sólo de la
# variable de entorno: la tarjeta la MIDIÓ en el volcado del Semanario (el
# Vigésimo Segundo, 17 de 17 tesis, Centro-Norte) y la variable la corrige.
from tarjeta_decision import (clave_de_tesis, fuerza_para_colegiado,  # noqa: E402
                              region_de_clave, region_del_circuito)


def _fuerza(tesis: dict, *, tribunal: str = "", region: Optional[str] = None,
            clave_propia: str = "") -> dict:
    """La fuerza de la regla única, normalizada al contrato {fuerza,
    fuerza_texto, por_confirmar?}. `tribunal` da el circuito, su región y la
    designación del propio tribunal; `region` y `clave_propia` los fuerzan."""
    r = fuerza_para_colegiado(tesis, tribunal, region=region, clave_propia=clave_propia)
    return {"fuerza": r["fuerza"], "fuerza_texto": str(r.get("fuerza_texto") or r["fuerza"]),
            **({"por_confirmar": True} if r.get("por_confirmar") else {})}


# ═══ LA VIGENCIA ═════════════════════════════════════════════════════════════

_CORRECCION = ("aclarada", "texto_sustituido")


def _vigencia_de(t: dict) -> Optional[dict]:
    v = t.get("vigencia")
    if isinstance(v, dict) and v:
        return v
    try:
        import vigencia_tesis as _vig
        return _vig.de(t.get("registro"))
    except Exception:
        return None


def pierde_vigencia(v: Optional[dict]) -> bool:
    """¿Perdió vigencia ENTERA? La aclaración y la republicación corregida no
    cambian el criterio (se citan corregidas); la pérdida parcial se enseña con
    su sello pero no se tira: la parte vigente sigue siéndolo."""
    if not isinstance(v, dict) or not v:
        return False
    return str(v.get("estado") or "") not in _CORRECCION and not v.get("parcial")


def _etiqueta_vigencia(v: Optional[dict]) -> Optional[str]:
    if not v:
        return None
    if v.get("etiqueta"):
        return str(v["etiqueta"])
    try:
        import vigencia_tesis as _vig
        return _vig.etiqueta(v) or None
    except Exception:
        return None


# ═══ EL CATÁLOGO CERRADO ═════════════════════════════════════════════════════

def _tipo_corto(t: dict) -> str:
    tipo = _plano_mayus(t.get("tipo"))
    if "JURISPRUDENCIA" in tipo:
        return "jurisprudencia"
    if "PRECEDENTE" in tipo:
        return "precedente"
    return "aislada"


def _ley_de_norma(n: dict) -> str:
    return str(n.get("cuerpo_legal") or n.get("fuente") or n.get("ley") or "").strip()


def indice_principal(problemas: list) -> int:
    """El principal por la jerarquía de la fase 3 (o la que corrigió el
    secretario); sin marca, el primero, como el árbol."""
    for i, p in enumerate(problemas or []):
        if isinstance(p, dict) and str(p.get("jerarquia") or "").strip().lower() == "principal":
            return i
    return 0


def _pregunta(p) -> str:
    return str(p.get("pregunta") or "") if isinstance(p, dict) else str(p or "")


def construir_catalogo(candidatas: list, normas: list = None, filas_propias: list = None,
                       lecturas: dict = None, *, region: Optional[str] = None,
                       clave_propia: str = "", tribunal: str = "") -> tuple:
    """(catálogo, avisos). Escalones con cupo propio; identificadores T (tesis),
    O (precedentes del propio tribunal) y N (normas). Las candidatas llegan en
    el orden de la búsqueda; `lecturas` = {registro: {"es", "a_favor"}} de la
    lectura barata. Lo ajeno sale; lo que perdió vigencia entera NO entra al
    catálogo (se avisa): así nadie puede proponer aplicarlo."""
    avisos: list = []
    lecturas = lecturas or {}
    vistas, tesis = set(), []
    for t in candidatas or []:
        if not isinstance(t, dict):
            continue
        reg = str(t.get("registro") or "").strip()
        if not reg or reg in vistas or not str(t.get("rubro") or "").strip():
            continue
        vistas.add(reg)
        tesis.append(t)
    fuera_vig, ajenas = [], 0
    filas = []
    for pos, t in enumerate(tesis):
        reg = str(t.get("registro"))
        lec = lecturas.get(reg) or {}
        if lec.get("es") == "ajena":
            ajenas += 1
            continue
        v = _vigencia_de(t)
        if pierde_vigencia(v):
            fuera_vig.append(f"{reg} ({_etiqueta_vigencia(v) or 'sin vigencia'})")
            continue
        f = _fuerza(t, tribunal=tribunal, region=region, clave_propia=clave_propia)
        filas.append((pos, t, v, f, lec))
    # EL ORDEN DENTRO DE CADA ESCALÓN: primero lo que la lectura dice que
    # RESUELVE el punto, después lo distinguible, después lo no leído; a
    # igualdad, el orden de la búsqueda (que ya trae el rerank y la jerarquía
    # por instancia que decidió David).
    _peso = {"resuelve": 0, "distinguible": 1}
    filas.sort(key=lambda x: (_peso.get(x[4].get("es"), 2), x[0]))
    obliga = [x for x in filas if x[3]["fuerza"] == "obliga"][:CUPO_OBLIGA]
    propias_t = [x for x in filas if x[3]["fuerza"] == "precedente_propio"][:CUPO_PROPIO]
    orienta = [x for x in filas if x[3]["fuerza"] in ("orienta", "pleno_circuito")][:CUPO_ORIENTA]
    cat: dict = {}
    i = 0
    for escalon, grupo in ((1, obliga), (2, orienta), (3, propias_t)):
        for _pos, t, v, f, lec in grupo:
            i += 1
            k = f"T{i}"
            cat[k] = {
                "id": k, "clase": "tesis", "escalon": escalon,
                "registro": str(t.get("registro")),
                "rubro": " ".join(str(t.get("rubro") or "").split()),
                "instancia": str(t.get("instancia") or ""),
                "tipo": _tipo_corto(t),
                "clave": clave_de_tesis(t),
                "texto": str(t.get("texto") or ""),
                "fuerza": f["fuerza"], "fuerza_texto": f["fuerza_texto"],
                **({"por_confirmar": True} if f.get("por_confirmar") else {}),
                "vigencia": _etiqueta_vigencia(v),
                "de_internet": bool(t.get("de_internet")),
                "lectura": lec.get("es") or None,
                "a_favor": lec.get("a_favor") or None,
            }
    j = 0
    for fila in (filas_propias or [])[:CUPO_PROPIO]:
        if not isinstance(fila, dict) or not str(fila.get("expediente") or "").strip():
            continue
        j += 1
        k = f"O{j}"
        cat[k] = {"id": k, "clase": "propio", "escalon": 3,
                  "fuerza": "precedente_propio",
                  "fuerza_texto": "precedente propio: apartarse exige razón (art. 228); no es voto",
                  **{c: fila.get(c) for c in ("expediente", "fecha", "sentido", "tipo_asunto",
                                              "tema", "calificacion", "razon", "similitud",
                                              "nivel", "neun", "pdf_url")
                     if fila.get(c) not in (None, "")}}
    vistas_n, n = set(), 0
    for nm in normas or []:
        if not isinstance(nm, dict):
            continue
        ley, art = _ley_de_norma(nm), str(nm.get("articulo") or "").strip()
        clave_n = (ley.lower(), art.lower())
        if not ley or not art or clave_n in vistas_n:
            continue
        vistas_n.add(clave_n)
        n += 1
        if n > CUPO_NORMAS:
            break
        k = f"N{n}"
        cat[k] = {"id": k, "clase": "norma", "escalon": 4, "ley": ley, "articulo": art,
                  "texto": str(nm.get("texto") or "")[:NORMA_CARACTERES]}
    if fuera_vig:
        avisos.append("Fuera del catálogo por haber perdido vigencia: " + "; ".join(fuera_vig[:6])
                      + ". No pueden proponerse como apoyo.")
    if ajenas:
        avisos.append(f"{ajenas} candidata(s) no tratan la figura que decide y no entraron.")
    return cat, avisos


def bloque_catalogo(cat: dict, caracteres: int = TESIS_CARACTERES_ABOGADO) -> str:
    """El catálogo para el prompt: lo que el modelo necesita para razonar con
    cada fuente, con su fuerza y su vigencia calculadas por el código."""
    def _t(e):
        extra = []
        if e.get("vigencia"):
            extra.append(f"⚠️ {e['vigencia']}")
        if e.get("de_internet"):
            extra.append("línea de la Corte hallada en internet y confirmada en el acervo")
        if e.get("lectura"):
            extra.append({"resuelve": "según la lectura previa, contesta la pregunta",
                          "distinguible": "según la lectura previa, trata la figura en "
                                          "circunstancias distintas"}.get(e["lectura"], ""))
        cab = f"[{e['id']}] {e['tipo'].upper()} · {e['instancia']} · FUERZA: {e['fuerza_texto']}"
        return (cab + ("\n    " + " · ".join(x for x in extra if x) if extra else "")
                + f"\n    {e['rubro']}\n    {e['texto'][:caracteres]}")
    L = []
    grupos = (
        ("LO QUE OBLIGA A ESTE TRIBUNAL", [e for e in cat.values() if e["clase"] == "tesis" and e["escalon"] == 1]),
        ("LO QUE ORIENTA (no obliga a un tribunal colegiado)", [e for e in cat.values() if e["clase"] == "tesis" and e["escalon"] == 2]),
        ("TESIS DEL PROPIO TRIBUNAL (precedente propio)", [e for e in cat.values() if e["clase"] == "tesis" and e["escalon"] == 3]),
    )
    for titulo, xs in grupos:
        if xs:
            L.append(titulo)
            L.extend(_t(e) for e in xs)
            L.append("")
    propios = [e for e in cat.values() if e["clase"] == "propio"]
    if propios:
        L.append("PRECEDENTES DEL PROPIO TRIBUNAL (cómo resolvió él mismo; no son voto ni se cuentan)")
        for e in propios:
            L.append(f"[{e['id']}] {e.get('tipo_asunto', '')} {e['expediente']} · {e.get('fecha', '')} · "
                     f"sentido: {e.get('sentido', '') or 'no consta'}"
                     + (f" · calificó: {e['calificacion']}" if e.get("calificacion") else "")
                     + (f"\n    razón: {str(e['razon'])[:500]}" if e.get("razon") else "")
                     + (f"\n    tema: {e['tema']}" if e.get("tema") else ""))
        L.append("")
    normas = [e for e in cat.values() if e["clase"] == "norma"]
    if normas:
        L.append("NORMAS")
        for e in normas:
            L.append(f"[{e['id']}] {e['ley']} — artículo {e['articulo']}: {e['texto']}")
    return "\n".join(L).strip() or "(el catálogo está vacío: no hay fuentes que citar)"


# ═══ LA VERIFICACIÓN (E) ═════════════════════════════════════════════════════

_RX_ID = re.compile(r"\[\s*([TNO]\d{1,3})\s*\]")
_RX_ID_LISTA = re.compile(r"\b([TNO]\d{1,3})\b")
_RX_ID_SUELTO = re.compile(r"(?<![\w\[])([TNO]\d{1,3})(?![\w\]])")
# «registro digital 2015688», «reg. 168958», «(registro número 2007402)»
_RX_REGISTRO_CON_PALABRA = re.compile(
    r"\(?\s*(?:registros?(?:\s+digital(?:es)?)?|reg\.)\s*(?:n[úu]m(?:ero)?\.?\s*)?:?\s*(\d{6,7})\s*\)?",
    re.I)
# La cifra suelta de seis o siete dígitos (`fase5_propuesta._RX_CIFRA_REGISTRO`),
# salvo una cantidad con signo de pesos o una parte de un número mayor.
_RX_CIFRA = re.compile(r"(?<![\$\d.,/])\b(\d{6,7})\b(?![\d/])")


def _de_ley(ley: str) -> str:
    try:
        import toulmin as _tl
        return _tl._de_ley(ley)
    except Exception:
        return f"de {ley}"


def _cita_legible(e: dict) -> str:
    if e["clase"] == "tesis":
        return f"(registro {e['registro']})"
    if e["clase"] == "norma":
        return f"(artículo {e['articulo']} {_de_ley(e['ley'])})"
    return f"({e.get('tipo_asunto', '')} {e.get('expediente', '')} de este tribunal)".replace("( ", "(")


def limpiar_texto(texto: Any, cat: dict, quitadas: Optional[list] = None) -> str:
    """El texto libre, sin nada que no esté en el catálogo.

    1. [T3] → «(registro 2015688)»; [N2] → «(artículo … de …)»; un
       identificador que no existe se BORRA.
    2. Toda cifra de registro que no pertenezca al catálogo se BORRA, con la
       palabra «registro» que la acompañe. Hoy `revisar` sólo avisa y el
       secretario lee el registro inventado igual; aquí no llega a verse.
    Lo quitado se anota en `quitadas` —para contarlo y medirlo—, y NO se
    repite en ningún aviso: un aviso que dice «se quitó el 2099999» vuelve a
    poner el registro inventado delante de quien firma."""
    t = str(texto or "")
    if not t.strip():
        return ""
    validos = {e["registro"] for e in cat.values() if e.get("clase") == "tesis"}
    quitados: list = []

    def _id(m):
        k = m.group(1)
        if k in cat:
            return _cita_legible(cat[k])
        quitados.append(k)
        return ""
    t = _RX_ID.sub(_id, t)
    # EL IDENTIFICADOR SIN CORCHETES («según T3») es la misma referencia: se
    # traduce o se borra igual; si no, un «T99» llegaría a la pantalla.
    t = _RX_ID_SUELTO.sub(_id, t)

    def _reg(m):
        if m.group(1) in validos:
            return m.group(0)
        quitados.append(m.group(1))
        return " "

    def _cifra(m):
        # UNA CANTIDAD NO ES UN REGISTRO: «$ 250000» conserva su cifra.
        if "$" in m.string[max(0, m.start() - 3):m.start()]:
            return m.group(0)
        return _reg(m)
    t = _RX_REGISTRO_CON_PALABRA.sub(_reg, t)
    t = _RX_CIFRA.sub(_cifra, t)
    t = re.sub(r"\(\s*[;,]?\s*\)", "", t)
    t = re.sub(r"[ \t]+([.,;:)])", r"\1", t)
    # Al borrar una referencia queda «… y .»: fuera la conjunción huérfana
    # (el mismo remate que `toulmin._redactar`).
    t = re.sub(r"\s+(?:y|e|o|u|así como)\s*([.,;:])", r"\1", t)
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"[ \t]{2,}", " ", t).strip()
    if quitados and quitadas is not None:
        quitadas.extend(quitados)
    return t


def ids_de(x: Any) -> list:
    """Los identificadores de una lista o de un texto («T3», «[N1]», "T3, O1")."""
    if isinstance(x, (list, tuple)):
        fuente = " ".join(str(y) for y in x)
    else:
        fuente = str(x or "")
    vistos, out = set(), []
    for k in _RX_ID_LISTA.findall(fuente):
        if k not in vistos:
            vistos.add(k)
            out.append(k)
    return out


def apoyo_de(e: dict) -> dict:
    """Una entrada del catálogo como APOYO del contrato de la tarjeta."""
    if e["clase"] == "norma":
        return {"norma": f"artículo {e['articulo']} {_de_ley(e['ley'])}", "id": e["id"],
                "registro": None, "rubro": None, "fuerza": None, "fuerza_texto": None,
                "vigencia": None, "de_internet": False, "en_acervo": True}
    if e["clase"] == "propio":
        return {"precedente_propio": f"{e.get('tipo_asunto', '')} {e.get('expediente', '')}".strip(),
                "id": e["id"], "registro": None, "rubro": None,
                "fuerza": "precedente_propio", "fuerza_texto": e["fuerza_texto"],
                "vigencia": None, "de_internet": False, "en_acervo": True, "norma": None}
    return {"id": e["id"], "registro": e["registro"], "rubro": e["rubro"],
            "instancia": e["instancia"], "tipo": e["tipo"], "fuerza": e["fuerza"],
            "fuerza_texto": e["fuerza_texto"], "vigencia": e.get("vigencia"),
            "de_internet": bool(e.get("de_internet")), "en_acervo": True, "norma": None,
            **({"por_confirmar": True} if e.get("por_confirmar") else {})}


def resolver_ids(ids: Any, cat: dict, quitadas: Optional[list] = None) -> list:
    """Los APOYOS de una lista de identificadores. Lo que no está en el catálogo
    se descarta (y se anota en `quitadas`); el catálogo ya no contiene lo
    abandonado, así que tampoco puede proponerse. Un registro escrito en lugar
    del identificador tampoco pasa: sólo se aceptan identificadores."""
    out, fuera = [], []
    for k in ids_de(ids):
        if k in cat:
            out.append(apoyo_de(cat[k]))
        else:
            fuera.append(k)
    crudo = " ".join(str(y) for y in ids) if isinstance(ids, (list, tuple)) else str(ids or "")
    fuera += re.findall(r"\b\d{6,7}\b", crudo)
    if fuera and quitadas is not None:
        quitadas.extend(fuera)
    return out


class _Textos:
    """Los textos literales (acto, escrito, constancias) listos para buscar
    citas palabra por palabra, con el mismo `Texto` del plan del estudio (dos
    lecturas: la cruda y la de renglones repuestos). Sin él, una comparación
    normalizada sencilla."""

    def __init__(self, textos: Optional[dict]):
        self.crudos = {k: str(v or "") for k, v in (textos or {}).items() if str(v or "").strip()}
        self.objs: dict = {}
        try:
            import plan_estudio as _pe
            for k, v in self.crudos.items():
                self.objs[k] = _pe.Texto(v)
            self._pe = _pe
        except Exception:                               # pragma: no cover
            self._pe = None

    @staticmethod
    def _plano(t: str) -> str:
        t = unicodedata.normalize("NFD", str(t or "").lower())
        t = "".join(c for c in t if unicodedata.category(c) != "Mn")
        return " " + " ".join(re.findall(r"\w+", t)) + " "

    def buscar(self, cita: str, fuente: str = "") -> tuple:
        """(cita tal como consta —recortada de sus bordes si hizo falta—, fuente
        donde está) o («», «»)."""
        cita = " ".join(str(cita or "").strip().strip("«»“”\"'").split())
        if len(cita.split()) < 5:
            return "", ""
        orden = ([fuente] if fuente in self.crudos else []) + [k for k in self.crudos if k != fuente]
        for k in orden:
            if self._pe is not None and k in self.objs:
                if self.objs[k].contiene(cita):
                    return cita, k
                c, i = self._pe._recorte_literal(cita, [self.objs[k]])
                if c:
                    return c, k
            elif self._plano(cita) in self._plano(self.crudos[k]):
                return cita, k
        return "", ""


def verificar_hechos(hechos: Any, textos: _Textos, cat: dict, quitadas: list) -> list:
    """Cada hecho con su cita buscada como subcadena. El que no se encuentra
    conserva lo que afirma, pierde la cita y queda «no acreditado»: el juez lo
    lee así (escalón 2) y la pantalla también."""
    out = []
    for h in hechos if isinstance(hechos, list) else []:
        if not isinstance(h, dict):
            continue
        afirma = limpiar_texto(h.get("afirma"), cat, quitadas)
        if not afirma:
            continue
        fuente = str(h.get("fuente") or "").strip().lower()
        fuente = {"constancias": "constancia", "autos": "constancia"}.get(fuente, fuente)
        cita, donde = textos.buscar(str(h.get("cita") or ""), fuente)
        # El número del hecho es el que lee el juez («1.h2»): se numera sobre
        # los que quedan, para que no haya huecos entre lo que ve y lo que cita.
        out.append({"id": f"h{len(out) + 1}", "afirma": afirma, "cita": cita,
                    "fuente": donde or (fuente if fuente in ("acto", "escrito", "constancia") else None),
                    "verificada": bool(cita)})
    return out


# ═══ LA CONSECUENCIA DE CADA VÍA (F) — por código, nunca por el modelo ═══════

def consecuencia_de(sentido: str, tipo_asunto: str = "", resolvio_a_quo: str = "",
                    resolutivo_recurrida: str = "", quejoso: str = "", responsable: str = "",
                    tenemos_conceptos: Optional[bool] = None, *, quien_recurre: str = "",
                    sobresee_ademas: bool = False) -> dict:
    """{"rama", "desenlace": [puntos], "desenlace_nota", "conceptos_omitidos"}
    de una vía, calculados por código.

    AR 631/2025 (28-sep-2026): revocar una concesión NIEGA lo que ella concedió
    —`puntos_revoca_concesion`, el arreglo 577c700—, pero no de inmediato: si
    recurre el tercero interesado o la autoridad y los agravios son fundados,
    el tribunal estudia los conceptos de violación que el juez no estudió (art.
    93, fr. VI, LA) y sólo si también caen se niega. «Revoca y niega» sin eso es
    prematuro, y la tarjeta tiene que decirlo antes de que el secretario elija
    la vía.

    UNA SOLA REGLA PARA LOS TRES (integración del 28-sep-2026): aquí había una
    tercera versión de la consecuencia, que ponía «no ampara ni protege» en el
    punto del amparo de una concesión revocada y decía «hacen falta los
    conceptos» aunque recurriera la propia quejosa. Ahora los puntos y la nota
    salen de `tarjeta_decision.desenlace_de` —los mismos puntos del documento:
    `tipos_asunto.puntos_reasuncion`, con el verbo del amparo en hueco hasta
    que se estudien los conceptos— y los conceptos omitidos de
    `fase_rama.conceptos_omitidos`, la función de SPEC B. `quejoso` y
    `responsable` sólo nombran a las partes fuera de un recurso: en la revisión
    los nombra el resolutivo del juzgado (577c700)."""
    import tipos_asunto as _ta
    import tarjeta_decision as _td
    t = _ta.normalizar(tipo_asunto) or "amparo_directo"
    s = str(sentido or "").strip().lower().replace(" ", "_")
    if not s:
        return {"rama": "", "desenlace": [], "desenlace_nota": None, "conceptos_omitidos": None}
    pros = _ta.prospera(s)
    if t == "amparo_revision":
        a = str(resolvio_a_quo or "").strip().lower()
        rama = _ta.rama_revision(a, s)
        puntos, nota = _td.desenlace_de(t, a, s, quien_recurre=quien_recurre,
                                        sobresee_ademas=sobresee_ademas,
                                        resolutivo_recurrida=resolutivo_recurrida)
        import fase_rama as _fr
        omitidos = _fr.conceptos_omitidos({"tipo_asunto": t, "que_hizo": a,
                                          "quien_recurre": quien_recurre,
                                          "sobresee_ademas": sobresee_ademas,
                                          "conceptos_violacion": ""}, s, None)
        if isinstance(omitidos, dict) and tenemos_conceptos is not None:
            # La deliberación sólo sabe si el secretario los aportó; dónde
            # buscarlos en el expediente lo hace la tarjeta con las fases.
            omitidos = dict(omitidos, tenemos=bool(tenemos_conceptos),
                            donde="secretario" if tenemos_conceptos else omitidos.get("donde", ""))
        if rama.startswith("revoca_sobreseimiento"):
            rama = "revoca_sobreseimiento"
        return {"rama": rama, "desenlace": list(puntos), "desenlace_nota": nota,
                "conceptos_omitidos": omitidos}
    c = _ta.cierre_de(t)
    frase = c["positivo"] if pros else c["negativo"]
    return {"rama": ("concede" if pros else "niega") if t == "amparo_directo"
            else ("prospera" if pros else "no_prospera"),
            "desenlace": [frase[:1].upper() + frase[1:] + "."], "desenlace_nota": None,
            "conceptos_omitidos": None}


# ═══ LOS SECUNDARIOS POR EL ÁRBOL, EN LAS DOS VÍAS (F) ═══════════════════════

def _de_contrato(de: str) -> str:
    """El `de` del árbol al vocabulario del contrato de la tarjeta."""
    return {"principal": "principal", "tuya": "secretario", "": "arbol"}.get(de or "", "arbol")


def _relacion_contrato(c: dict, rel_lista: str) -> str:
    r = str(c.get("relacion") or "")
    if r == "presupone":
        return "presupone"
    if r in ("autonoma", "mixta"):
        return "autonoma"
    if str(c.get("de") or "") == "distinto" or rel_lista == "distinto":
        return "distinto"
    return rel_lista or "depende"


def lista_de_comprobacion(problemas: list, pi: int, sec_a: list, sec_b: list) -> list:
    """La suerte que escribió cada abogado, con los campos de la lista de la
    propuesta (`relacion`, `si_prospera`, `si_no_prospera`, `presupone`), para
    que `arbol_decision` la verifique igual que la del motor."""
    por_a = {int(x.get("numero")): x for x in (sec_a or []) if isinstance(x, dict) and str(x.get("numero", "")).isdigit()}
    por_b = {int(x.get("numero")): x for x in (sec_b or []) if isinstance(x, dict) and str(x.get("numero", "")).isdigit()}
    lista = [{"numero": pi + 1, "tema": _pregunta(problemas[pi]), "papel": "principal"}]
    for i, p in enumerate(problemas):
        if i == pi:
            continue
        n = i + 1
        a, b = por_a.get(n) or {}, por_b.get(n) or {}
        e = {"numero": n, "tema": _pregunta(p), "papel": "accesorio"}
        rel = str(a.get("relacion") or b.get("relacion") or "").strip().lower()
        if rel in ("depende", "distinto"):
            e["relacion"] = rel
        if isinstance(a.get("suerte"), dict) and a["suerte"].get("sentido"):
            e["si_prospera"] = {"sentido": str(a["suerte"].get("sentido") or ""),
                                "razon": str(a["suerte"].get("razon") or "")}
        if isinstance(b.get("suerte"), dict) and b["suerte"].get("sentido"):
            e["si_no_prospera"] = {"sentido": str(b["suerte"].get("sentido") or ""),
                                   "razon": str(b["suerte"].get("razon") or "")}
        if isinstance(b.get("presupone"), dict):
            e["presupone"] = b["presupone"]
        lista.append(e)
    return lista


def secundarios_por_arbol(problemas: list, pi: int, sentido_a: str, sentido_b: str,
                          lista: list, tipo_asunto: str = "") -> tuple:
    """(secundarios, avisos_A, avisos_B). `arbol_decision.reparto_para_pantalla`
    corre DOS veces —el principal en la vía A y en la vía B— con la lista que
    escribieron los abogados. En cada corrida el «sentido del motor» es el de
    esa misma vía: el principal nunca va «al revés» y nada se tumba para
    recalificar (la suerte de esa vía ya la escribió su abogado). Sin modelo."""
    import arbol_decision as _ad
    probs = []
    for i, p in enumerate(problemas):
        d = dict(p) if isinstance(p, dict) else {"pregunta": str(p)}
        d["jerarquia"] = "principal" if i == pi else "accesorio"
        probs.append(d)
    ptxt = _pregunta(probs[pi])

    def _correr(sentido: str) -> tuple:
        if not sentido:
            return {}, []
        crit = [{"problema": ptxt, "sentido": sentido, "razonamiento": "",
                 "jerarquia": "principal", "tocado": True}]
        crit += [{"problema": _pregunta(p), "sentido": "", "razonamiento": "",
                  "jerarquia": "accesorio", "tocado": False}
                 for i, p in enumerate(probs) if i != pi]
        out = _ad.reparto_para_pantalla(
            probs, crit, lista, [{"problema": ptxt, "sentido": sentido, "alcanza": True}],
            sentido_motor=sentido, tipo_asunto=tipo_asunto)
        return {str(c.get("problema")): c for c in out["criterios"]}, list(out.get("avisos") or [])

    en_a, av_a = _correr(sentido_a)
    en_b, av_b = _correr(sentido_b)
    rel_de = {int(e["numero"]): str(e.get("relacion") or "") for e in lista}
    secs = []
    for i, p in enumerate(probs):
        if i == pi:
            continue
        txt = _pregunta(p)
        rel_l = rel_de.get(i + 1, "")
        if not rel_l:
            try:
                rel_l = "depende" if int(p.get("depende_de")) == pi + 1 else ""
            except (TypeError, ValueError):
                rel_l = ""

        def _suerte(c: Optional[dict]) -> Optional[dict]:
            if not c:
                return None
            return {"sentido": str(c.get("sentido") or ""),
                    "razon": str(c.get("razonamiento") or ""),
                    "de": _de_contrato(str(c.get("de") or "")),
                    "de_arbol": str(c.get("de") or ""),
                    "por_que": str(c.get("por_que") or ""),
                    "relacion": _relacion_contrato(c, rel_l),
                    "guarda": (str(c.get("guarda")) or None) if c.get("guarda") else None,
                    "recalificar": bool(c.get("recalificar")),
                    "previsto": False}
        secs.append({"numero": i + 1, "pregunta": txt, "clase": str(p.get("clase") or ""),
                     "relacion": rel_l or "autonoma",
                     "en_A": _suerte(en_a.get(txt)), "en_B": _suerte(en_b.get(txt))})
    return secs, av_a, av_b


def efecto_de(secs: list, clave: str) -> str:
    """Una línea con lo que les pasa a los secundarios en esa vía, contada por
    código a partir del árbol (no la escribe el modelo)."""
    if not secs:
        return "No hay otros planteamientos."
    n_sin = n_cae = n_est = 0
    for s in secs:
        x = s.get(clave) or {}
        sen = str(x.get("sentido") or "")
        if sen in ("innecesario", "sin_materia"):
            n_sin += 1
        elif x.get("relacion") == "presupone":
            n_cae += 1
        else:
            n_est += 1
    partes = []
    if n_sin:
        partes.append(f"{n_sin} queda(n) sin materia")
    if n_cae:
        partes.append(f"{n_cae} cae(n) con lo desestimado")
    if n_est:
        partes.append(f"{n_est} se estudia(n) con su calificación")
    return "Los demás planteamientos: " + ", ".join(partes) + "."


# ═══ LAS LLAMADAS ════════════════════════════════════════════════════════════

class _Uso:
    def __init__(self):
        self.llamadas = 0
        self.entrada = 0
        self.salida = 0

    def anotar(self, r) -> None:
        self.llamadas += 1
        try:
            u = getattr(r, "usage", None)
            self.entrada += int(getattr(u, "prompt_tokens", 0) or 0)
            self.salida += int(getattr(u, "completion_tokens", 0) or 0)
        except Exception:
            pass

    def doc(self) -> dict:
        return {"llamadas": self.llamadas, "entrada": self.entrada, "salida": self.salida,
                "coste_usd": round((self.entrada * PRECIO_ENTRADA + self.salida * PRECIO_SALIDA) / 1e6, 4)}


_RX_JSON = re.compile(r"\{.*\}", re.S)


def leer_json(crudo: str) -> dict:
    m = _RX_JSON.search(crudo or "")
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


async def _pedir(cliente, prompt: str, *, modelo: str, esfuerzo: str, tope: int,
                 semilla: int, uso: _Uso) -> dict:
    """Una llamada que devuelve JSON. VACÍA = se agotó razonando: se repite UNA
    vez con el doble de sitio y EL MISMO esfuerzo (bajarlo es lo que David
    rechazó). Nunca lanza: {} si no hay JSON."""
    import llamada_modelo as _lm
    kw = dict(model=modelo, temperature=0, seed=semilla, max_completion_tokens=tope,
              messages=[{"role": "user", "content": prompt}])
    if esfuerzo:
        kw["reasoning_effort"] = esfuerzo
    try:
        r = await _lm.crear(cliente, **kw)
        uso.anotar(r)
        crudo = (r.choices[0].message.content or "").strip()
        if not crudo:
            r = await _lm.crear(cliente, **dict(kw, max_completion_tokens=tope * 2))
            uso.anotar(r)
            crudo = (r.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"   ⚖️ DELIBERACIÓN: la llamada falló ({type(e).__name__}: {str(e)[:120]})")
        return {}
    return leer_json(crudo)


# ═══ LOS PROMPTS ═════════════════════════════════════════════════════════════
# Descripciones de cada campo; NUNCA una frase modelo que copiar (lección del
# proyecto: el modelo copia el ejemplo). Ningún caso real va de ejemplo.

def _vocab(tipo_asunto: str, es_recurso: bool) -> tuple:
    try:
        import tipos_asunto as _ta
        t = tipo_asunto or ("amparo_revision" if es_recurso else "amparo_directo")
        return _ta.vocabulario_de(t)["combate"], _ta.sujetos_de(t)["organo"][0]
    except Exception:                                   # pragma: no cover
        return ("agravios" if es_recurso else "conceptos de violación"), "la autoridad"


def _ventana(texto: str, tope: int, centro: str = "", desde_el_final: bool = False) -> str:
    """Un tramo del texto literal. Si se conoce la cita de la proposición
    toral, alrededor de ella; si no, el final (en una sentencia recurrida el
    estudio está al final) o el principio (en el escrito)."""
    t = str(texto or "")
    if len(t) <= tope:
        return t
    if centro:
        pl = " ".join(centro.split()[:8])
        i = t.find(pl) if pl else -1
        if i >= 0:
            a = max(0, i - tope // 2)
            return t[a:a + tope]
    return t[-tope:] if desde_el_final else t[:tope]


def _bloque_ficha(ficha: str) -> str:
    """La ficha procesal del asunto como DATOS (SPEC_E2, 28-sep-2026;
    `ficha_procesal.bloque`): quién promovió, quién recurre y con qué
    carácter, qué resolvió el juzgado por acto, qué es materia de la revisión
    y qué quedó firme, la fracción del art. 93 y el desenlace de cada vía por
    código. En el AR 631/2025 los dos abogados y el juez no sabían que la
    recurrente era la tercera interesada. «» si no hay ficha."""
    b = str(ficha or "").strip()
    return (b + "\n\n") if b else ""


def prompt_pregunta_decisiva(pral: dict, contraste: Optional[dict], resumen_acto: str,
                             resumen_conceptos: str, texto_acto: str,
                             tipo_asunto: str = "", es_recurso: bool = False,
                             ficha: str = "") -> str:
    q, org = _vocab(tipo_asunto, es_recurso)
    c = contraste or {}
    _c = (f"\nEl contraste previo identificó como razón toral: {c.get('razon_toral')}\n"
          if c.get("razon_toral") else "")
    return f"""TAREA: LA PREGUNTA DECISIVA DEL PLANTEAMIENTO PRINCIPAL
Eres secretario de un Tribunal Colegiado. Antes de buscar criterio y antes de
decidir, identificas QUÉ decide el planteamiento principal. No lo resuelves.

{_bloque_ficha(ficha)}EL PLANTEAMIENTO PRINCIPAL
{_pregunta(pral)}
Lo que resolvió {org}: {pral.get('resolvio') or '(no consta)'}
Lo que sostienen los {q}: {pral.get('combate') or '(no consta)'}
{_c}
LO QUE RESOLVIÓ {org.upper()}, resumido:
{str(resumen_acto or '')[:15000]}

LO QUE SE COMBATE, resumido:
{str(resumen_conceptos or '')[:15000]}

EL TEXTO LITERAL DE LA RESOLUCIÓN (de aquí, y sólo de aquí, sale la cita):
{texto_acto or '(no se tiene el texto literal)'}

QUÉ DEVUELVES:
1. `figura`: la institución jurídica que gobierna el punto, nombrada como la
   nombraría el rubro de una tesis: la figura, no los hechos ni las partes.
2. `pregunta_decisiva`: la pregunta de derecho que, contestada, decide el
   planteamiento. Se formula sobre la figura, con las notas del caso que la
   distinguen (calidad de quien actúa, etapa del procedimiento, acto de que se
   trata) y sin nombres propios. No repite la pregunta tal como la planteó la
   parte o la resolución: es la que contestaría un criterio del Semanario.
3. `proposicion_toral`: la consideración de la resolución que sostiene el
   fallo en ese punto. `dice`: en una frase tuya. `cita`: las palabras
   LITERALES de la resolución, copiadas del texto de arriba, seguidas, sin
   cortes ni puntos suspensivos, de ocho a sesenta palabras. Si no la
   encuentras escrita, deja `cita` vacía: el taller la busca palabra por
   palabra y la que no está se descarta.
4. `hechos_que_deciden`: los hechos del expediente de los que depende la
   respuesta, como mucho cinco, uno por elemento, sin calificarlos.
5. `busquedas`: de dos a cuatro consultas para buscar en el Semanario el
   criterio que contesta la pregunta decisiva, escritas en el lenguaje de los
   RUBROS: sintagmas nominales en versales, la figura primero y sus notas
   después, sin verbos conjugados, sin hechos ni partes. Al menos una nombra
   la figura con las notas del caso; otra, el GÉNERO al que pertenece, para
   alcanzar por analogía si no hay criterio sobre el caso concreto.
6. `interpretacion_conforme`: si la respuesta depende del alcance de un
   precepto que admite más de una lectura y una de ellas es la conforme con la
   Constitución o la más favorable a la persona, `precepto` (el artículo y su
   ley, tal como constan arriba) y `por_que` (en un renglón, qué lecturas
   admite y de qué depende elegir). Si no depende de eso, null.

No cites tesis, registros ni preceptos que no estén arriba. No decidas.

Devuelve SÓLO un JSON, sin texto alrededor:
{{"figura": "<la institución>",
  "pregunta_decisiva": "<la pregunta de derecho>",
  "proposicion_toral": {{"dice": "<una frase>", "cita": "<literal o vacía>"}},
  "hechos_que_deciden": ["<un hecho>"],
  "busquedas": ["<consulta en lenguaje de rubro>"],
  "interpretacion_conforme": {{"precepto": "<artículo y ley>", "por_que": "<un renglón>"}} o null}}"""


def prompt_lectura(decisiva: dict, candidatas: list) -> str:
    filas = []
    for i, t in enumerate(candidatas, 1):
        filas.append(f"{i}. [{_tipo_corto(t).upper()} · {t.get('instancia', '')}] "
                     f"{' '.join(str(t.get('rubro') or '').split())[:300]}\n"
                     f"   {' '.join(str(t.get('texto') or '').split())[:500]}")
    return f"""TAREA: LECTURA DE CANDIDATOS FRENTE A LA PREGUNTA DECISIVA
Eres secretario de un Tribunal Colegiado. Abajo va la pregunta que decide el
planteamiento principal y una lista numerada de criterios candidatos. Para
CADA uno dices:
- `es`: "resuelve" si contesta ESA pregunta, sobre la misma figura y en
  circunstancias que este caso comparte; "distinguible" si trata la figura en
  circunstancias que el caso no comparte, o contesta una pregunta cercana pero
  distinta; "ajena" si no trata la figura.
- `a_favor`: la vía que sostendría si se aplicara: "prospera" (lo que plantea
  quien combate), "no_prospera" (lo resuelto se sostiene) o "ninguna".
Juzgas lo que dice el criterio, no quién lo emitió.

PREGUNTA DECISIVA: {decisiva.get('pregunta_decisiva') or '(no consta)'}
FIGURA: {decisiva.get('figura') or '(no consta)'}
PROPOSICIÓN TORAL DE LA RESOLUCIÓN: {(decisiva.get('proposicion_toral') or {}).get('dice') or '(no consta)'}

CANDIDATOS:
{chr(10).join(filas)}

Devuelve SÓLO un JSON:
{{"lectura": [{{"n": <número>, "es": "resuelve|distinguible|ajena", "a_favor": "prospera|no_prospera|ninguna"}}]}}"""


_VIA_SENTIDOS = {
    "A": ("fundado", "esencialmente_fundado", "sustancialmente_fundado", "parcialmente_fundado"),
    "B": ("infundado", "inoperante", "fundado_insuficiente", "ineficaz", "inatendible"),
}

_GLOSARIO = """CÓMO SE CALIFICA (no son sinónimos):
- fundado: combate la razón de lo resuelto y tiene razón.
- esencialmente fundado: combate la razón toral y tiene razón en lo sustancial,
  aunque no en todos sus términos; prospera y el proyecto acota en qué medida.
- sustancialmente fundado: tiene razón en lo esencial de lo que plantea, y basta.
- parcialmente fundado: tiene razón en una parte; prospera en esa parte.
- fundado pero insuficiente: tiene razón y aun así no alcanza, porque otra
  consideración que no combate sostiene lo resuelto; no prospera.
- infundado: combate la razón y no tiene razón.
- inoperante: no combate la razón toral (ataca lo que no sostiene el fallo,
  repite lo dicho en la instancia o parte de una premisa falsa); se razona.
- inatendible: no puede examinarse por cómo o cuándo se plantea.
- ineficaz: se dirige contra consideraciones que ya no rigen el sentido."""


def _lista_secundarios(problemas: list, pi: int, q: str, org: str) -> str:
    L = []
    for i, p in enumerate(problemas):
        if i == pi:
            continue
        L.append(f"{i + 1}. {_pregunta(p)}"
                 + (f"\n   Lo que resolvió {org}: {p.get('resolvio')}" if p.get("resolvio") else "")
                 + (f"\n   Lo que se combate: {p.get('combate')}" if p.get("combate") else "")
                 + (f"\n   Clase: {p.get('clase')}" if p.get("clase") else ""))
    return "\n".join(L) or "(no hay otros planteamientos)"


def prompt_abogado(via: str, *, pral: dict, pi: int, problemas: list, decisiva: dict,
                   contraste: Optional[dict], resumen_acto: str, resumen_conceptos: str,
                   textos: dict, cat: dict, tipo_asunto: str = "", es_recurso: bool = False,
                   marco: str = "", metodo: str = "", suplencia: str = "",
                   direccion: str = "", regla_ley: str = "",
                   hay_procesal_ad: bool = False, ficha: str = "") -> str:
    q, org = _vocab(tipo_asunto, es_recurso)
    prospera = via == "A"
    nombre = "PROSPERA" if prospera else "NO PROSPERA"
    toral = decisiva.get("proposicion_toral") or {}
    c = contraste or {}
    _c = ""
    if c:
        _c = (f"\nEL CONTRASTE PREVIO (orienta, no decide): razón toral: {c.get('razon_toral') or 'no consta'}"
              f" · ¿la combate? {'sí' if c.get('la_combate') else 'no'}"
              + (f" · ¿sobrevive el fallo? {'sí' if c.get('sobrevive') else 'no'}" if c.get('la_combate') else "")
              + f"\n  {c.get('por_que') or ''}\n")
    if prospera:
        _suerte = ("`suerte`: qué le pasa a ese planteamiento si el principal PROSPERA: "
                   "\"innecesario\" si queda sin materia o lo absorbe lo que se resuelve en el "
                   "principal; su propia calificación, con su razón, si es un tema distinto o si "
                   "pide más de lo que da el principal.")
        _presupone = ""
        _pres_json = ""
    else:
        _suerte = ("`suerte`: qué le pasa a ese planteamiento si el principal NO prospera. "
                   "Quedar sin materia cuando el principal prospera no lo hace caer cuando no "
                   "prospera: si ataca otra consideración o denuncia un vicio propio, lleva la "
                   "calificación que merece por lo que él plantea, con su razón.")
        _presupone = ("\n    `presupone`: null, salvo que TODO su argumento dé por cierta la "
                      "premisa del principal que esta vía desestima. Entonces: `premisa` (esa "
                      "premisa, en una frase), `cita` (un pasaje LITERAL breve, de seis a unas "
                      "cuarenta palabras seguidas, de lo que se combate en ESE planteamiento, "
                      "donde la da por cierta; nunca el planteamiento entero) y `causa_propia` "
                      "(lo que plantea además por su cuenta, en una frase, o null). El taller "
                      "busca el pasaje y, si no está, el planteamiento se estudia.")
        _pres_json = (',\n      "presupone": null | {"premisa": "<…>", "cita": "<literal>", '
                      '"causa_propia": null | "<…>"}')
    _proc = ("\n    Las VIOLACIONES PROCESALES de un amparo directo no quedan sin materia ni caen "
             "con el principal (arts. 74, fr. V, y 174 LA): se deciden por lo que ellas plantean; "
             "la única excepción es un principal de fondo que prospere con mayor beneficio que "
             "la reposición (art. 189)." if hay_procesal_ad else "")
    sentidos = " | ".join(_VIA_SENTIDOS[via])
    ventana_acto = _ventana(textos.get("acto", ""), VENTANA_ACTO, toral.get("cita", ""), True)
    ventana_escrito = _ventana(textos.get("escrito", ""), VENTANA_ESCRITO)
    ventana_const = _ventana(textos.get("constancia", ""), VENTANA_CONSTANCIAS)
    return f"""TAREA: ABOGADO DE LA VÍA EN QUE EL PLANTEAMIENTO PRINCIPAL {nombre}
Eres secretario de un Tribunal Colegiado y en este encargo ARGUMENTAS UNA SOLA
VÍA: aquella en que los {q} del planteamiento principal {nombre.lower()}. Otra
persona argumenta la vía contraria por separado, y un tercero que no sabe quién
escribió cada una las compara después. Tu trabajo es la mejor versión HONESTA
de esta vía con lo que consta; no decides cuál es mejor.
{direccion}
{_bloque_ficha(ficha)}EL PLANTEAMIENTO PRINCIPAL
{_pregunta(pral)}
Lo que resolvió {org}: {pral.get('resolvio') or '(no consta)'}
Lo que sostienen los {q}: {pral.get('combate') or '(no consta)'}

LA PREGUNTA QUE LO DECIDE: {decisiva.get('pregunta_decisiva') or '(no se formuló)'}
LA FIGURA: {decisiva.get('figura') or '(no consta)'}
LA PROPOSICIÓN TORAL DE LA RESOLUCIÓN: {toral.get('dice') or '(no consta)'}
  cita literal {'VERIFICADA' if toral.get('verificada') else 'NO verificada'}: {toral.get('cita') or '(sin cita)'}
{_c}
LOS DEMÁS PLANTEAMIENTOS DEL ASUNTO
{_lista_secundarios(problemas, pi, q, org)}

LO QUE RESOLVIÓ {org.upper()}, resumido:
{str(resumen_acto or '')[:15000]}

LO QUE SE COMBATE, resumido:
{str(resumen_conceptos or '')[:15000]}

TEXTO LITERAL DE LA RESOLUCIÓN (fuente «acto»):
{ventana_acto or '(no se tiene)'}

TEXTO LITERAL DEL ESCRITO (fuente «escrito»):
{ventana_escrito or '(no se tiene)'}

CONSTANCIAS DE AUTOS (fuente «constancia»):
{ventana_const or '(no hay)'}

FUENTES DISPONIBLES — CATÁLOGO CERRADO: sólo existen éstas y se citan por su
identificador entre corchetes. La FUERZA y la vigencia las calculó el taller;
no las cambies.
{bloque_catalogo(cat)}
{marco}
{metodo}
{suplencia}
{_GLOSARIO}

LO QUE ESCRIBES, EN ESTE ORDEN:
1. `regla`: la regla que contesta la pregunta decisiva, EN ABSTRACTO —sin los
   hechos del caso—, con las fuentes del catálogo que la establecen por su
   identificador entre corchetes. Si algo que OBLIGA a este tribunal apunta a
   la vía contraria, no lo calles: distínguelo en `autoridad_contraria`.
2. `hechos`: los hechos de los que depende aplicarla. Cada uno con `afirma`
   (el hecho, en una frase), `cita` (sus palabras LITERALES, copiadas de uno de
   los tres textos literales de arriba, de cinco a cuarenta palabras seguidas)
   y `fuente` ("acto", "escrito" o "constancia"). El hecho cuya cita no esté
   palabra por palabra en esa fuente se tiene por NO acreditado.
3. `subsuncion`: cómo esos hechos caen bajo la regla.
4. `conclusion`: la calificación que resulta y por qué, en un párrafo.
5. `sentido`: la calificación de esta vía: {sentidos}.
6. `razon`: la razón toral de esta vía en unas sesenta palabras: es lo que
   quien firma lee antes de elegir.
7. `interpretacion`: qué precepto y qué lectura, o qué criterio, sostienen
   esta vía, en dos renglones y con sus identificadores.
8. `objecion`: `de_la_otra_via` (el mejor argumento de la vía contraria, en un
   renglón) y `respuesta` (por qué no la derriba).
9. `autoridad_contraria`: los criterios del catálogo que apoyarían la otra vía,
   cada uno con `id` y `distincion` (por qué no rige este caso). Lista vacía si
   no hay.
10. `propongo_aplicar`: los identificadores del catálogo que aplicarías,
   tesis y normas, del más fuerte al más débil. Sólo identificadores.
11. `precedente_propio`: por cada fuente O que toque el punto, `id`, `trato`
   ("sigue", "distingue" o "se_aparta") y `por_que`. Apartarse de lo que este
   tribunal ya resolvió exige decir por qué.
12. `secundarios`: por cada uno de LOS DEMÁS PLANTEAMIENTOS, `numero`,
   `relacion` ("depende" si, al prosperar el principal, queda sin materia o lo
   absorbe; "distinto" si se sostiene y se resuelve solo) y
    {_suerte}{_presupone}{_proc}
13. `sostenible`: false si con estas fuentes y estos hechos esta vía no se
   puede sostener; entonces lo dices en `razon` y no la fuerzas.

REGLAS QUE NO SE ROMPEN:
- Sólo existen las fuentes del catálogo. No escribas números de registro,
  claves de tesis ni artículos que no estén en él: cita por identificador. Lo
  que no está en el catálogo se borra antes de que nadie lo lea.
- No supongas lo que no consta: un hecho sin cita literal no está acreditado.
- {regla_ley or 'La ley que rige es la del asunto, que es la que está en el catálogo.'}
- No sabes qué propuso nadie antes ni qué escribió la otra vía, y no importa.

Devuelve SÓLO un JSON, sin texto alrededor:
{{"sentido": "<{sentidos}>",
  "razon": "<unas sesenta palabras>",
  "interpretacion": "<dos renglones con identificadores>",
  "regla": "<la regla en abstracto, con identificadores>",
  "hechos": [{{"afirma": "<el hecho>", "cita": "<literal>", "fuente": "acto|escrito|constancia"}}],
  "subsuncion": "<…>",
  "conclusion": "<…>",
  "objecion": {{"de_la_otra_via": "<un renglón>", "respuesta": "<…>"}},
  "autoridad_contraria": [{{"id": "<T…>", "distincion": "<…>"}}],
  "propongo_aplicar": ["<T… o N…>"],
  "precedente_propio": [{{"id": "<O…>", "trato": "sigue|distingue|se_aparta", "por_que": "<…>"}}],
  "secundarios": [{{"numero": <n>, "relacion": "depende|distinto",
      "suerte": {{"sentido": "<calificación o innecesario>", "razon": "<una frase>"}}{_pres_json}}}],
  "sostenible": true}}"""


def _bloque_via_para_juez(etq: str, v: dict, cat: dict) -> str:
    hechos = "\n".join(
        f"    {h['id']}. {h['afirma']}"
        + (f"\n        cita {h['fuente']} VERIFICADA: «{h['cita']}»" if h.get("verificada")
           else "\n        SIN cita verificada: NO acreditado")
        for h in (v.get("cadena") or {}).get("hechos") or [])
    ap = ", ".join(f"{a.get('id')} ({a.get('fuerza_texto') or a.get('norma') or ''})"
                   for a in v.get("apoyos") or []) or "ninguna fuente del catálogo"
    contra = "\n".join(f"    {x['id']}: {x['distincion']}" for x in v.get("autoridad_contraria") or []) \
        or "    (no declara)"
    prop = "\n".join(f"    {x['id']}: {x['trato']} — {x['por_que']}" for x in v.get("precedente_propio") or []) \
        or "    (no declara)"
    cad = v.get("cadena") or {}
    ob = v.get("objecion") or {}
    return f"""{etq} — sentido: {str(v.get('sentido') or '').replace('_', ' ')} · el planteamiento {'PROSPERA' if v.get('prospera') else 'NO prospera'}{'' if v.get('sostenible', True) else ' · quien la argumentó dice que NO se sostiene'}
  regla: {cad.get('regla') or '(no la formuló)'}
  hechos:
{hechos or '    (ninguno)'}
  subsunción: {cad.get('subsuncion') or '(no la hizo)'}
  conclusión: {cad.get('conclusion') or '(no la dio)'}
  objeción de la otra vía: {ob.get('de_la_otra_via') or '(no la dijo)'}
  respuesta: {ob.get('respuesta') or '(no la dio)'}
  autoridad contraria y cómo la distingue:
{contra}
  propone aplicar: {ap}
  precedente propio:
{prop}"""


def prompt_juez(orden: tuple, vias: dict, *, decisiva: dict, cat: dict,
                constancias_faltantes: list, suplencia: str = "", tasa_base: str = "",
                ficha: str = "") -> str:
    toral = decisiva.get("proposicion_toral") or {}
    faltan = "\n".join(f"  · {c.get('que')}" + (f" — para: {c.get('para_que')}" if c.get("para_que") else "")
                       for c in constancias_faltantes or [] if isinstance(c, dict) and c.get("que")) \
        or "  (no se señaló ninguna)"
    _sup = ("\n   Aquí OPERA LA SUPLENCIA DE LA QUEJA (abajo): cura la deficiencia del "
            "argumento, no obliga a darle la razón." if suplencia.strip() else "")
    return f"""TAREA: JUEZ DE LAS DOS VÍAS
Eres el magistrado ponente de un Tribunal Colegiado. Tienes delante dos vías
argumentadas por separado, rotuladas «Vía 1» y «Vía 2». No sabes quién
escribió cada una ni cuál prefirió nadie, y el orden en que aparecen no
significa nada. Decides cuál se sostiene con UN ORDEN FIJO: cada escalón se
consulta sólo si el anterior no decide.

1. LO QUE OBLIGA MANDA. ¿Hay en el catálogo un criterio con fuerza «obliga» a
   este tribunal, vigente, que conteste la pregunta decisiva? La vía que lo
   contradiga sin distinguirlo pierde. Lo que «orienta» no obliga: la
   jurisprudencia de otro colegiado no obliga a un colegiado (art. 217, párrafo
   tercero, de la Ley de Amparo).
2. EL HECHO ACREDITADO. ¿La vía en que el planteamiento prospera combate la
   proposición toral con un hecho acreditado? Sólo está acreditado el hecho con
   cita VERIFICADA; uno sin cita verificada no está acreditado. No supongas lo
   que no consta.
3. PRESUNCIÓN DE LEGALIDAD Y ESTRICTO DERECHO. A la vía que prospera le toca
   demostrar el error de lo resuelto; en empate, no prospera.{_sup}
4. EL PRECEDENTE PROPIO. Si este tribunal ya resolvió el punto (fuentes O), la
   vía que se aparte tiene que decir por qué; apartarse sin razón pesa en su
   contra. No se cuenta como voto.
5. LA TASA BASE, sólo para desempatar y diciéndolo: {tasa_base or '(no se tiene)'}

{_bloque_ficha(ficha)}LA PREGUNTA DECISIVA: {decisiva.get('pregunta_decisiva') or '(no se formuló)'}
LA FIGURA: {decisiva.get('figura') or '(no consta)'}
LA PROPOSICIÓN TORAL DE LO RESUELTO: {toral.get('dice') or '(no consta)'}
  cita {'VERIFICADA' if toral.get('verificada') else 'NO verificada'}: {toral.get('cita') or '(sin cita)'}

CONSTANCIAS QUE HARÍA FALTA VER Y NO ESTÁN:
{faltan}

FUENTES (catálogo cerrado; la fuerza y la vigencia las calculó el taller):
{bloque_catalogo(cat, TESIS_CARACTERES_JUEZ)}
{suplencia}

{_bloque_via_para_juez('VÍA 1', vias[orden[0]], cat)}

{_bloque_via_para_juez('VÍA 2', vias[orden[1]], cat)}

DEVUELVES:
- `recomendada`: "1", "2" o "ninguna" (ninguna sólo si ninguna de las dos se
  sostiene con lo que hay).
- `escalon`: el número del escalón que decidió (1 a 5).
- `fuentes`: los identificadores del catálogo que decidieron ese escalón.
- `hechos`: los hechos que decidieron, como «1.h2» (número de la vía, punto,
  hecho).
- `por_que`: tres renglones, cada uno una frase.
- `crux`: `que` (el dato o la regla de la que depende la decisión),
  `si_cambia` (qué pasaría si ese dato fuera otro) y `constancia` (la
  constancia que lo acreditaría, si falta; si no, null).
- `debilidad`: la mayor debilidad de la vía recomendada, en un renglón.
- `otra_sostenible`: true si la otra vía también se sostiene con apoyo del
  catálogo.
- `estado`: "claro" si decidieron los escalones 1 o 2; "reñido" si hubo que
  llegar al 3, al 4 o al 5; "no_alcanza" si ninguna vía tiene apoyo en el
  catálogo.
- `precedente_propio`: `se_aparta` (true, false o null) y `por_que`.
Sin porcentajes ni grados de confianza. No escribas registros: identificadores.

Devuelve SÓLO un JSON:
{{"recomendada": "1|2|ninguna", "escalon": <1-5>, "fuentes": ["<id>"], "hechos": ["<1.h1>"],
  "por_que": ["<…>", "<…>", "<…>"],
  "crux": {{"que": "<…>", "si_cambia": "<…>", "constancia": null | "<…>"}},
  "debilidad": "<…>", "otra_sostenible": true,
  "estado": "claro|reñido|no_alcanza",
  "precedente_propio": {{"se_aparta": null, "por_que": "<…>"}}}}"""


# ═══ LAS ETAPAS ══════════════════════════════════════════════════════════════

async def etapa_a(cliente, pral: dict, contraste: Optional[dict], resumen_acto: str,
                  resumen_conceptos: str, textos: _Textos, tipo_asunto: str,
                  es_recurso: bool, uso: _Uso, avisos: list, ficha: str = "") -> dict:
    """La pregunta decisiva, con la cita de la proposición toral VERIFICADA
    contra el acto. Si la llamada falla, se sigue con la pregunta de la fase 3
    y se dice: peor búsqueda, pero búsqueda."""
    acto = textos.crudos.get("acto", "")
    d = await _pedir(cliente, prompt_pregunta_decisiva(
        pral, contraste, resumen_acto, resumen_conceptos,
        _ventana(acto, 60000, desde_el_final=True), tipo_asunto, es_recurso, ficha=ficha),
        modelo=MODELO_LECTURA, esfuerzo=ESFUERZO_LECTURA, tope=TOKENS_LECTURA,
        semilla=20260928, uso=uso)
    if not d.get("pregunta_decisiva"):
        avisos.append("No se formuló la pregunta decisiva: se buscó con la pregunta del "
                      "planteamiento principal.")
    tor = d.get("proposicion_toral") if isinstance(d.get("proposicion_toral"), dict) else {}
    cita, donde = textos.buscar(str(tor.get("cita") or ""), "acto")
    if str(tor.get("cita") or "").strip() and not (cita and donde == "acto"):
        avisos.append("La cita de la proposición toral no está palabra por palabra en la "
                      "resolución: se descartó.")
    if donde != "acto":
        cita = ""
    return {"pregunta_decisiva": " ".join(str(d.get("pregunta_decisiva") or _pregunta(pral)).split()),
            "figura": " ".join(str(d.get("figura") or "").split()),
            "proposicion_toral": {"dice": " ".join(str(tor.get("dice") or "").split()),
                                  "cita": cita, "verificada": bool(cita)},
            "hechos_que_deciden": [" ".join(str(h).split()) for h in (d.get("hechos_que_deciden") or [])
                                   if str(h).strip()][:5],
            # LA PREGUNTA TAL COMO LLEGÓ (SPEC E3, AR 631/2025): la recurrida
            # la planteó como «¿alteró la cosa juzgada?» y lo que decide es la
            # figura. Las dos viajan: la decisiva manda la búsqueda y el
            # razonamiento; la recurrida es el marco en que se contesta.
            "pregunta_recurrida": " ".join(_pregunta(pral).split()),
            "busquedas": _busquedas_de(d.get("busquedas")),
            "interpretacion_conforme": _interpretacion_de(d.get("interpretacion_conforme")),
            "formulada": bool(d.get("pregunta_decisiva"))}


def _busquedas_de(x: Any) -> list:
    """Las consultas en lenguaje de rubro, sin repetir y con tope: cada una es
    un embedding y una consulta a Qdrant. Se quitan los signos de pregunta: con
    prosa interrogativa el vector `rubro` es lo peor medido (fase6_rag)."""
    out, vistos = [], set()
    for q in x if isinstance(x, list) else []:
        t = " ".join(str(q or "").replace("¿", " ").replace("?", " ").split())[:220]
        k = t.lower()
        if len(t) >= 8 and k not in vistos:
            vistos.add(k)
            out.append(t)
    return out[:4]


def _interpretacion_de(x: Any) -> Optional[dict]:
    """{precepto, por_que} o None. Sin precepto no hay interpretación conforme
    que buscar: un «por qué» suelto no dice qué norma leer."""
    if not isinstance(x, dict):
        return None
    pre = " ".join(str(x.get("precepto") or "").split())[:200]
    if not pre:
        return None
    return {"precepto": pre, "por_que": " ".join(str(x.get("por_que") or "").split())[:400]}


def _tesis_de_resultado(x: Any) -> tuple:
    """(tesis, normas) de lo que devuelva el buscador: lista, dict o Material."""
    if x is None:
        return [], []
    if isinstance(x, list):
        return [t for t in x if isinstance(t, dict)], []
    if isinstance(x, dict):
        return list(x.get("tesis") or []), list(x.get("normas") or [])
    return list(getattr(x, "tesis", []) or []), list(getattr(x, "normas", []) or [])


async def etapa_b(cliente, decisiva: dict, pral: dict, pi: int, material, *,
                  buscar=None, reforzar=None, internet=None, filas_propias=None,
                  region=None, clave_propia: str = "", tribunal: str = "",
                  uso: _Uso, avisos: list) -> dict:
    """La escalera: lo que ya trae el material del principal + la búsqueda sobre
    la pregunta decisiva + el refuerzo por vía + la línea de internet que el
    acervo confirmó. Se suma; no sustituye (lector2_potencia §2)."""
    base = [t for t in (getattr(material, "tesis", []) or []) if isinstance(t, dict)
            and not t.get("metodo")]
    con_para = [t for t in base if t.get("para")]
    if con_para:
        base = [t for t in base if (pi + 1) in (t.get("para") or []) or t.get("de_internet")]
    normas = [n for n in (getattr(material, "normas", []) or []) if isinstance(n, dict)]
    nuevas: list = []
    tareas = []
    pd = decisiva.get("pregunta_decisiva") or _pregunta(pral)
    if buscar is not None:
        tareas.append(("busqueda", buscar(pd, decisiva.get("figura") or "")))
    if reforzar is not None:
        tareas.append(("refuerzo_A", reforzar(pd, "", str(pral.get("combate") or ""), list(base))))
        tareas.append(("refuerzo_B", reforzar(pd, str(pral.get("resolvio") or ""), "", list(base))))
    if internet is not None:
        tareas.append(("internet", internet(pd)))
    if tareas:
        res = await asyncio.gather(*[t for _, t in tareas], return_exceptions=True)
        for (nombre, _), r in zip(tareas, res):
            if isinstance(r, BaseException):
                avisos.append(f"La {nombre.replace('_', ' ')} de la escalera falló "
                              f"({type(r).__name__}); se sigue con lo demás.")
                continue
            ts, ns = _tesis_de_resultado(r)
            if nombre == "internet":
                # SÓLO LO QUE EL ACERVO CONFIRMÓ (fase_internet): la pista no
                # es cita y aquí ni siquiera entra.
                ts = [dict(t, de_internet=True) for t in ts if t.get("registro") and t.get("rubro")]
            nuevas.extend(ts)
            normas.extend(ns)
    # LA BÚSQUEDA SOBRE LA PREGUNTA DECISIVA VA PRIMERO: es la que llega al
    # anaquel de la figura; lo del material viene de la pregunta del agravio.
    candidatas, vistos = [], set()
    for t in nuevas + base:
        reg = str(t.get("registro") or "")
        if reg and reg not in vistos:
            vistos.add(reg)
            candidatas.append(t)
    lecturas: dict = {}
    if candidatas and cliente is not None:
        muestra = candidatas[:CANDIDATAS_LECTURA]
        d = await _pedir(cliente, prompt_lectura(decisiva, muestra), modelo=MODELO_LECTURA,
                         esfuerzo=ESFUERZO_LECTURA, tope=TOKENS_LECTURA, semilla=20260928, uso=uso)
        for x in d.get("lectura") or []:
            try:
                n = int(x.get("n"))
            except (TypeError, ValueError, AttributeError):
                continue
            if 1 <= n <= len(muestra):
                es = str(x.get("es") or "").strip().lower()
                af = str(x.get("a_favor") or "").strip().lower()
                lecturas[str(muestra[n - 1].get("registro"))] = {
                    "es": es if es in ("resuelve", "distinguible", "ajena") else None,
                    "a_favor": af if af in ("prospera", "no_prospera", "ninguna") else None}
    cat, av = construir_catalogo(candidatas, normas, filas_propias, lecturas,
                                 region=region, clave_propia=clave_propia, tribunal=tribunal)
    avisos.extend(av)
    return cat


def verificar_via(d: dict, via: str, cat: dict, textos: _Textos, avisos: list,
                  quitadas: Optional[list] = None) -> dict:
    """La salida de un abogado después de E: identificadores resueltos por
    código, cifras de registro sueltas fuera, hechos buscados en el texto
    literal, y el sentido obligado a ser el de SU vía (un abogado de la vía que
    prospera no puede devolver «infundado»)."""
    import tipos_asunto as _ta
    av_via: list = []
    q = quitadas if quitadas is not None else []
    s = str(d.get("sentido") or "").strip().lower().replace(" ", "_")
    quiere = via == "A"
    if not s or s not in _VIA_SENTIDOS[via] or _ta.prospera(s) != quiere:
        if s:
            av_via.append(f"volvió con «{s}», que no es de su vía: se tomó "
                          f"«{_VIA_SENTIDOS[via][0]}».")
        # AUNQUE EL ABOGADO NO RESPONDA, LA VÍA EXISTE: su consecuencia y la
        # suerte de los secundarios las calcula el código; lo que falta es el
        # argumento, y eso se dice (`respondio`).
        s = _VIA_SENTIDOS[via][0]
    L = lambda x: limpiar_texto(x, cat, q)                       # noqa: E731
    hechos = verificar_hechos(d.get("hechos"), textos, cat, q)
    propuestos = resolver_ids(d.get("propongo_aplicar"), cat, q)
    contra = []
    for x in d.get("autoridad_contraria") or []:
        if isinstance(x, dict) and x.get("id") in cat:
            contra.append({"id": x["id"], **({"registro": cat[x["id"]].get("registro"),
                                              "rubro": cat[x["id"]].get("rubro")}
                                             if cat[x["id"]]["clase"] == "tesis" else {}),
                           "distincion": L(x.get("distincion"))})
    propio = []
    for x in d.get("precedente_propio") or []:
        if isinstance(x, dict) and x.get("id") in cat and cat[x["id"]]["clase"] == "propio":
            tr = str(x.get("trato") or "").strip().lower()
            propio.append({"id": x["id"], "expediente": cat[x["id"]].get("expediente"),
                           "trato": tr if tr in ("sigue", "distingue", "se_aparta") else "sigue",
                           "por_que": L(x.get("por_que"))})
    ob = d.get("objecion") if isinstance(d.get("objecion"), dict) else {}
    secs = []
    for x in d.get("secundarios") or []:
        if not isinstance(x, dict):
            continue
        try:
            n = int(x.get("numero"))
        except (TypeError, ValueError):
            continue
        su = x.get("suerte") if isinstance(x.get("suerte"), dict) else {}
        e = {"numero": n, "relacion": str(x.get("relacion") or "").strip().lower(),
             "suerte": {"sentido": str(su.get("sentido") or "").strip().lower().replace(" ", "_"),
                        "razon": L(su.get("razon"))}}
        if via == "B" and isinstance(x.get("presupone"), dict):
            e["presupone"] = {"premisa": L(x["presupone"].get("premisa")),
                              "cita": str(x["presupone"].get("cita") or ""),
                              "causa_propia": x["presupone"].get("causa_propia")}
        secs.append(e)
    fuera = {
        "via": via, "sentido": s, "prospera": quiere,
        "razon": L(d.get("razon")), "interpretacion": L(d.get("interpretacion")) or None,
        "cadena": {"regla": L(d.get("regla")), "hechos": hechos,
                   "subsuncion": L(d.get("subsuncion")), "conclusion": L(d.get("conclusion"))},
        "objecion": {"de_la_otra_via": L(ob.get("de_la_otra_via")), "respuesta": L(ob.get("respuesta"))},
        "autoridad_contraria": contra, "apoyos": propuestos, "precedente_propio": propio,
        "secundarios_escritos": secs,
        "sostenible": bool(d.get("sostenible", True)) if d else False,
        "respondio": bool(d),
    }
    for a in av_via:
        _a = f"Vía {via}: {a}"
        if _a not in avisos:
            avisos.append(_a)
    return fuera


def tiene_apoyo(v: dict) -> bool:
    """¿La vía se apoya en algo del catálogo verificado (tesis vigente o
    norma)? Sin eso, es una opinión."""
    return bool(v.get("apoyos"))


async def etapa_d(cliente, orden: tuple, vias: dict, *, decisiva: dict, cat: dict,
                  constancias_faltantes: list, suplencia: str, tasa_base: str,
                  uso: _Uso, semilla: int, ficha: str = "") -> dict:
    d = await _pedir(cliente, prompt_juez(orden, vias, decisiva=decisiva, cat=cat,
                                         constancias_faltantes=constancias_faltantes,
                                         suplencia=suplencia, tasa_base=tasa_base,
                                         ficha=ficha),
                     modelo=_modelo(), esfuerzo=ESFUERZO_DELIBERACION, tope=TOKENS_JUEZ,
                     semilla=semilla, uso=uso)
    rec = str(d.get("recomendada") or "").strip().lower()
    rec_via = {"1": orden[0], "2": orden[1]}.get(rec[:1] if rec else "", None)
    try:
        esc = int(d.get("escalon"))
    except (TypeError, ValueError):
        esc = 0
    hechos = []
    for h in d.get("hechos") or []:
        m = re.match(r"\s*([12])\s*\.\s*h\s*(\d+)", str(h), re.I)
        if m:
            hechos.append((orden[int(m.group(1)) - 1], f"h{m.group(2)}"))
    est = str(d.get("estado") or "").strip().lower().replace("renido", "reñido")
    pq = d.get("por_que")
    pq = pq if isinstance(pq, list) else [pq]
    return {"orden": list(orden), "respondio": bool(d), "recomendada": rec_via,
            "escalon": esc if 1 <= esc <= 5 else 0, "fuentes": ids_de(d.get("fuentes")),
            "hechos": hechos, "estado": est if est in ("claro", "reñido", "no_alcanza") else "",
            "por_que": [str(x) for x in pq if str(x or "").strip()][:3],
            "crux": d.get("crux") if isinstance(d.get("crux"), dict) else None,
            "debilidad": str(d.get("debilidad") or ""),
            "otra_sostenible": d.get("otra_sostenible") if isinstance(d.get("otra_sostenible"), bool) else None,
            "precedente_propio": d.get("precedente_propio") if isinstance(d.get("precedente_propio"), dict) else None}


def _escalon_comprobado(p: dict, vias: dict, cat: dict) -> tuple:
    """(escalón que se sostiene, motivo si se rebajó). El 1 exige un criterio
    del catálogo que de verdad obligue (fuerza calculada por el código, no el
    rótulo del acervo) y esté vigente; el 2, un hecho con cita verificada en la
    vía que prospera cuando es ella la recomendada."""
    esc, rec = p.get("escalon") or 0, p.get("recomendada")
    if esc == 1:
        ids = set(p.get("fuentes") or []) | {a.get("id") for a in (vias.get(rec) or {}).get("apoyos") or []}
        if not any(cat.get(k, {}).get("fuerza") == "obliga" for k in ids if k):
            return 3, ("el juez invocó «lo que obliga» sin un criterio que obligue a este "
                       "tribunal en el catálogo verificado")
    if esc == 2 and rec and (vias.get(rec) or {}).get("prospera"):
        hechos = (vias[rec].get("cadena") or {}).get("hechos") or []
        verif = {h["id"] for h in hechos if h.get("verificada")}
        citados = {h for v, h in p.get("hechos") or [] if v == rec}
        if not verif or (citados and not (citados & verif)):
            return 3, ("el juez decidió por un hecho acreditado y la vía que prospera no tiene "
                       "ese hecho con cita verificada")
    return esc, ""


def combinar(p1: dict, p2: dict, vias: dict, cat: dict) -> dict:
    """El estado lo decide el CÓDIGO con lo que dijeron las dos pasadas:
      · «no_alcanza» si ninguna vía tiene apoyo verificado, o las dos pasadas
        lo dicen;
      · «claro» sólo si las dos pasadas recomiendan la MISMA vía, esa vía
        tiene apoyo verificado y el escalón que invocaron (1 o 2) se comprueba;
      · «reñido» en todo lo demás.
    `recomendada` sólo con «claro»; `inclinacion` guarda la vía en que
    coincidieron aunque sea reñido (para el orden de la pantalla, no como
    recomendación)."""
    razones: list = []
    apA, apB = tiene_apoyo(vias["A"]), tiene_apoyo(vias["B"])
    respondieron = [p for p in (p1, p2) if p.get("respondio")]
    if not apA and not apB:
        return {"estado": "no_alcanza", "recomendada": None, "inclinacion": None,
                "estado_por_que": ["Ninguna de las dos vías se apoya en un criterio o una norma "
                                   "del catálogo verificado."]}
    if respondieron and all(p.get("estado") == "no_alcanza" for p in respondieron) \
            and len(respondieron) == 2:
        return {"estado": "no_alcanza", "recomendada": None, "inclinacion": None,
                "estado_por_que": ["Las dos pasadas del juez concluyen que lo que hay no alcanza "
                                   "para sostener ninguna vía."]}
    if len(respondieron) < 2:
        razones.append("Una de las dos pasadas del juez no respondió: falta la contraprueba "
                       "del orden invertido.")
    r1, r2 = p1.get("recomendada"), p2.get("recomendada")
    coinciden = bool(r1 and r1 == r2)
    if len(respondieron) == 2 and not r1 and not r2:
        razones.append("El juez no recomendó ninguna vía en ninguna de las dos pasadas.")
    elif len(respondieron) == 2 and not coinciden:
        razones.append("Las dos pasadas del juez, con el orden invertido, no recomiendan la "
                       "misma vía.")
    if coinciden:
        inclinacion = r1
    elif len(respondieron) == 1:
        inclinacion = respondieron[0].get("recomendada")
    else:
        inclinacion = None
    claro = coinciden and len(respondieron) == 2
    if claro:
        rebajado = False
        for p in (p1, p2):
            esc, mot = _escalon_comprobado(p, vias, cat)
            if mot:
                rebajado = True
                if mot[:1].upper() + mot[1:] + "." not in razones:
                    razones.append(mot[:1].upper() + mot[1:] + ".")
            if esc not in (1, 2):
                claro = False
        if not tiene_apoyo(vias[r1]):
            claro = False
            razones.append("La vía en que coincidieron no tiene apoyo verificado en el catálogo.")
        if claro:
            e = min(p1["escalon"], p2["escalon"])
            razones.append(f"Las dos pasadas coinciden y deciden en el escalón {e} "
                           + ("(lo que obliga a este tribunal)." if e == 1
                              else "(el hecho acreditado frente a la proposición toral)."))
        elif not rebajado and tiene_apoyo(vias[r1]):
            razones.append("Las dos pasadas coinciden, pero hubo que llegar a la presunción de "
                           "legalidad, al precedente propio o a la tasa base.")
    return {"estado": "claro" if claro else "reñido",
            "recomendada": r1 if claro else None,
            "inclinacion": inclinacion, "estado_por_que": razones}


# ═══ LA DELIBERACIÓN ENTERA ══════════════════════════════════════════════════

def clave_de(huella: str, contexto: str = "", registros: list = None) -> str:
    """Qué hace a esta deliberación ESTA: el adelanto (huella), lo que aportó el
    secretario y el acervo sobre el que se delibera. No entra la propuesta del
    motor: el juez no la ve."""
    base = json.dumps([VERSION, huella or "", " ".join(str(contexto or "").split()),
                       sorted(str(r) for r in (registros or []))], ensure_ascii=False)
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:20]


def _contraste_de(contraste: Optional[list], numero: int) -> Optional[dict]:
    for c in contraste or []:
        try:
            if isinstance(c, dict) and int(c.get("numero")) == numero:
                return c
        except (TypeError, ValueError):
            continue
    return None


async def deliberar(cliente, *, problemas: list, material=None,
                    resumen_acto: str = "", resumen_conceptos: str = "",
                    textos: Optional[dict] = None, tipo_asunto: str = "",
                    es_recurso: bool = False, recurrente: str = "",
                    contraste: Optional[list] = None,
                    constancias_faltantes: Optional[list] = None,
                    resolvio_a_quo: str = "", resolutivo_recurrida: str = "",
                    quejoso: str = "", responsable: str = "",
                    tenemos_conceptos: Optional[bool] = None, marco: str = "",
                    buscar: Optional[Callable[..., Awaitable[Any]]] = None,
                    reforzar: Optional[Callable[..., Awaitable[Any]]] = None,
                    internet: Optional[Callable[..., Awaitable[Any]]] = None,
                    filas_propias: Optional[list] = None, region: Optional[str] = None,
                    clave_propia: str = "", tasa_base: str = "", tribunal: str = "",
                    quien_recurre: str = "", sobresee_ademas: bool = False,
                    ficha: str = "",
                    decisiva_previa: Optional[dict] = None) -> dict:
    """La deliberación del problema principal. Devuelve el documento que va a la
    marca «deliberacion» (JSON puro). Las búsquedas se inyectan:
      buscar(pregunta, figura)                     → tesis (lista, dict o Material)
      reforzar(pregunta, resolvio, combate, ya)    → tesis nuevas
      internet(pregunta)                           → tesis que el acervo confirmó
    Nunca lanza por un fallo del modelo: lo que falte se dice en `avisos`."""
    t0 = time.perf_counter()
    uso, avisos = _Uso(), []
    probs = [dict(p) if isinstance(p, dict) else {"pregunta": str(p)} for p in (problemas or [])]
    if not probs:
        return {"formato": FORMATO, "version": VERSION, "origen": "deliberacion",
                "estado": "no_alcanza", "recomendada": None, "inclinacion": None,
                "estado_por_que": ["No hay planteamientos que deliberar."], "vias": {},
                "secundarios": [], "avisos": [], "uso": uso.doc()}
    pi = indice_principal(probs)
    pral = probs[pi]
    tx = _Textos(textos)
    c_pral = _contraste_de(contraste, pi + 1)

    # A · la pregunta decisiva. SI YA SE FORMULÓ PARA TODOS (SPEC E3,
    # `pregunta_decisiva.py`, marca «decisiva» del mismo adelanto) y es de este
    # principal, se reutiliza: la misma llamada, ya pagada, y la búsqueda de la
    # figura ya está en el material.
    _prev = decisiva_previa if isinstance(decisiva_previa, dict) else {}
    if (_prev.get("formulada") and _prev.get("pregunta_decisiva")
            and " ".join(str(_prev.get("pregunta_recurrida") or "").split())
            == " ".join(_pregunta(pral).split())):
        decisiva = {k: _prev.get(k) for k in (
            "pregunta_decisiva", "figura", "proposicion_toral", "hechos_que_deciden",
            "pregunta_recurrida", "busquedas", "interpretacion_conforme", "formulada")}
    else:
        decisiva = await etapa_a(cliente, pral, c_pral, resumen_acto, resumen_conceptos, tx,
                                 tipo_asunto, es_recurso, uso, avisos, ficha=ficha)

    # B · la escalera
    cat = await etapa_b(cliente, decisiva, pral, pi, material, buscar=buscar, reforzar=reforzar,
                        internet=internet, filas_propias=filas_propias, region=region,
                        clave_propia=clave_propia, tribunal=tribunal, uso=uso, avisos=avisos)

    # C · los dos abogados, en paralelo y sin verse
    import fase5_propuesta as _f5
    try:
        import dialogo_constitucional as _dc
    except Exception:                                   # pragma: no cover
        _dc = None
    try:
        import violacion_procesal as _vp
        hay_proc = bool(_vp.guarda_aplica(tipo_asunto)) and any(
            _vp.clase_de(p) == "procesal" for i, p in enumerate(probs) if i != pi)
    except Exception:
        hay_proc = False
    suplencia = _f5._bloque_suplencia(material) if material is not None else ""

    def _pa(via: str) -> str:
        s_rep = _VIA_SENTIDOS[via][0]
        metodo = ""
        if _dc is not None and material is not None:
            try:
                metodo = _dc.bloque_metodo(material, "razon", _dc.favorece_a_la_persona(
                    s_rep, tipo_asunto, recurrente, es_recurso))
            except Exception:
                metodo = ""
        return prompt_abogado(
            via, pral=pral, pi=pi, problemas=probs, decisiva=decisiva, contraste=c_pral,
            resumen_acto=resumen_acto, resumen_conceptos=resumen_conceptos, textos=tx.crudos,
            cat=cat, tipo_asunto=tipo_asunto, es_recurso=es_recurso, marco=marco,
            metodo=metodo, suplencia=suplencia,
            # LA CALIFICACIÓN ES DEL AGRAVIO, NO DE LA PREGUNTA (AR 631/2025): con
            # la pregunta al lado, el modelo contestaba «sí» y razonaba a favor
            # del juez con «fundado» al pie. Se le dice quién gana en su vía.
            direccion=_f5.bloque_direccion(s_rep, tipo_asunto, es_recurso,
                                           str(pral.get("combate") or ""),
                                           str(pral.get("resolvio") or "")),
            regla_ley=_f5._regla_de_ley(material) if material is not None else "",
            hay_procesal_ad=hay_proc, ficha=ficha)

    crudoA, crudoB = await asyncio.gather(
        _pedir(cliente, _pa("A"), modelo=_modelo(), esfuerzo=ESFUERZO_DELIBERACION,
               tope=TOKENS_ABOGADO, semilla=20260928, uso=uso),
        _pedir(cliente, _pa("B"), modelo=_modelo(), esfuerzo=ESFUERZO_DELIBERACION,
               tope=TOKENS_ABOGADO, semilla=20260929, uso=uso))
    # E · antes de que el juez lea nada
    quitadas: list = []
    vias = {"A": verificar_via(crudoA, "A", cat, tx, avisos, quitadas),
            "B": verificar_via(crudoB, "B", cat, tx, avisos, quitadas)}
    for k in ("A", "B"):
        if not vias[k]["respondio"]:
            avisos.append(f"El abogado de la vía {k} no respondió: esa vía queda sin argumentar.")

    # D · el juez ciego, dos pasadas con el orden invertido
    p1 = p2 = {"respondio": False, "orden": [], "recomendada": None, "escalon": 0,
               "fuentes": [], "hechos": [], "estado": "", "por_que": [], "crux": None,
               "debilidad": "", "otra_sostenible": None, "precedente_propio": None}
    # Sin apoyo verificado en ninguna de las dos no hay nada que juzgar: el
    # estado es «no_alcanza» por código y se ahorran las dos pasadas.
    if (vias["A"]["respondio"] or vias["B"]["respondio"]) and (
            tiene_apoyo(vias["A"]) or tiene_apoyo(vias["B"])):
        p1, p2 = await asyncio.gather(
            etapa_d(cliente, ("A", "B"), vias, decisiva=decisiva, cat=cat,
                    constancias_faltantes=constancias_faltantes or [], suplencia=suplencia,
                    tasa_base=tasa_base, uso=uso, semilla=20260930, ficha=ficha),
            etapa_d(cliente, ("B", "A"), vias, decisiva=decisiva, cat=cat,
                    constancias_faltantes=constancias_faltantes or [], suplencia=suplencia,
                    tasa_base=tasa_base, uso=uso, semilla=20260930, ficha=ficha))
    comb = combinar(p1, p2, vias, cat)

    # E · lo del juez, limpio. Habla la pasada que recomendó la vía en que
    # coincidieron; si no coincidieron, la primera que respondió.
    guia = next((p for p in (p1, p2) if p.get("respondio") and comb.get("inclinacion")
                 and p.get("recomendada") == comb["inclinacion"]),
                next((p for p in (p1, p2) if p.get("respondio")), p1))
    L = lambda x: limpiar_texto(x, cat, quitadas)                # noqa: E731
    crux = None
    if isinstance(guia.get("crux"), dict) and any(guia["crux"].values()):
        crux = {"que": L(guia["crux"].get("que")), "si_cambia": L(guia["crux"].get("si_cambia")),
                "constancia": (L(guia["crux"].get("constancia")) or None)}

    # F · la consecuencia de cada vía y los secundarios, por código
    for k in ("A", "B"):
        v = vias[k]
        v.update(consecuencia_de(v["sentido"], tipo_asunto, resolvio_a_quo, resolutivo_recurrida,
                                 quejoso, responsable, tenemos_conceptos,
                                 quien_recurre=quien_recurre, sobresee_ademas=sobresee_ademas))
    lista = lista_de_comprobacion(probs, pi, vias["A"]["secundarios_escritos"],
                                  vias["B"]["secundarios_escritos"])
    secs, av_a, av_b = secundarios_por_arbol(probs, pi, vias["A"]["sentido"], vias["B"]["sentido"],
                                             lista, tipo_asunto)
    vias["A"]["efecto"] = efecto_de(secs, "en_A")
    vias["B"]["efecto"] = efecto_de(secs, "en_B")
    vias["A"]["avisos_arbol"], vias["B"]["avisos_arbol"] = av_a, av_b
    por_que = [y for y in (L(x) for x in guia.get("por_que") or []) if y]
    debilidad = L(guia.get("debilidad")) or None
    # LO QUE SE QUITÓ, CONTADO Y NO REPETIDO: el aviso dice cuántas y no
    # cuáles (repetir el registro inventado es volver a enseñarlo).
    _q = sorted(set(quitadas))
    if _q:
        avisos.append(f"La verificación quitó {len(_q)} referencia(s) que el modelo escribió "
                      f"y no están en el catálogo verificado: no se enseñan.")
    _no_acr = sum(1 for k in ("A", "B") for h in (vias[k]["cadena"]["hechos"] or [])
                  if not h.get("verificada"))
    seg = round(time.perf_counter() - t0, 1)
    doc = {
        "formato": FORMATO, "version": VERSION, "origen": "deliberacion",
        "principal": {"numero": pi + 1, "pregunta": _pregunta(pral), **decisiva},
        "catalogo": {k: {c: v for c, v in e.items() if c != "texto"} for k, e in cat.items()},
        "recomendada": comb["recomendada"], "inclinacion": comb["inclinacion"],
        "estado": comb["estado"], "estado_por_que": comb["estado_por_que"],
        "por_que": por_que, "crux": crux, "debilidad": debilidad,
        "otra_sostenible": guia.get("otra_sostenible"),
        # SÓLO CUÁNTAS, NUNCA CUÁLES: el documento lo lee la tarjeta y un
        # registro inventado no se guarda donde alguien pueda enseñarlo.
        "verificacion": {"referencias_quitadas": len(_q), "hechos_no_acreditados": _no_acr},
        "vias": vias, "secundarios": secs,
        # LA LISTA QUE ESCRIBIERON LOS ABOGADOS, con los campos de la de la
        # propuesta: la tarjeta la pasa al árbol en las dos vías (integración
        # del 28-sep-2026), igual que aquí `secundarios_por_arbol`.
        "checklist": lista,
        "juez": {"pasadas": [{c: p.get(c) for c in ("orden", "respondio", "recomendada", "escalon",
                                                    "estado")} for p in (p1, p2)],
                 "coinciden": bool(p1.get("recomendada") and p1.get("recomendada") == p2.get("recomendada")),
                 "precedente_propio": guia.get("precedente_propio")},
        "avisos": avisos, "uso": uso.doc(), "segundos": seg,
    }
    print(f"   ⚖️ DELIBERACIÓN: {doc['estado']} · recomendada {doc['recomendada'] or '—'} · "
          f"{len(cat)} fuentes · {uso.llamadas} llamadas · {doc['uso']['coste_usd']} USD · {seg} s")
    return doc


# ═══ LA PROYECCIÓN SOBRE LA TARJETA ══════════════════════════════════════════
# Vive en UN solo sitio: `tarjeta_decision.armar(…, deliberacion=<marca>)`.
# Hasta la integración (28-sep-2026) había aquí una segunda proyección,
# `para_tarjeta`, que nadie llamaba desde el servidor y que no volvía a
# verificar los apoyos, ni corría el árbol con la propuesta del motor, ni
# aplicaba los frenos duros del estado (constancia indispensable, ningún apoyo
# verificado). La tarjeta lee de este documento: `vias` (A prospera, B no), su
# `cadena`, `objecion` y `apoyos`; `secundarios_escritos` de cada abogado;
# `checklist`; `catalogo` (para no tirar como «fuera del acervo» lo que la
# escalera trajo); `recomendada`, `inclinacion`, `estado`, `crux` y
# `principal` con la pregunta decisiva.
