# -*- coding: utf-8 -*-
"""LAS PREGUNTAS AL SECRETARIO, EN VEZ DE PEDIRLE CONSTANCIAS (David, 2-oct-2026).

«El motor siempre pide constancias, creo que pudiéramos simplificarlo a
preguntas como ¿El emplazamiento carece de cercioramiento? o ¿En el contrato
base la cláusula sexta qué dice su texto expreso? … simplificar el tema de
constancias y no ser tan exigentes.»

«Nunca dar por hecho que lo que se dice en los recursos o conceptos de
violación es cierto (eso es justo lo que se debe verificar a la luz de lo
acreditado en el juicio, que por regla general viene dicho en la sentencia
reclamada o recurrida); sólo si no se dice se le pregunta al abogado, pero no
se genera ninguna propuesta hasta que no se respondan esas preguntas, sólo si
se consideran indispensables.»

DE DÓNDE SALEN. El análisis neutral de la litis (`analisis_litis`), que ya lee
el acto, el escrito y los autos ENTEROS, entrega con la bandera
«preguntas_al_secretario» una lista de PREMISAS: cada afirmación fáctica
decisiva de la parte, contrastada con lo que la resolución tuvo por cierto,
desestimó o calló. Sólo cuando calla (y la cita del acto no se halla) nace una
pregunta cerrada al secretario. Este módulo es PURO —sin modelo, sin red, sin
base—: convierte esas premisas en preguntas, decide cuáles son
INDISPENSABLES por código y arma el bloque con que las respuestas llegan a la
propuesta y al estudio.

POR QUÉ «INDISPENSABLE» LO DECIDE EL CÓDIGO Y NO EL MODELO. La regla 13 de la
propuesta dejaba al modelo marcar sus constancias como indispensables, y
`constancias.normalizar` ponía indispensable=True por omisión: así nació el
«siempre pide constancias». Aquí una pregunta sólo bloquea la propuesta si:
  1. es del problema PRINCIPAL (o del que decide el asunto),
  2. su respuesta CAMBIA el desenlace —`probabilidad_sentido.prospera` de la
     calificación con «sí» distinta de la de «no»—, y
  3. la resolución no se pronunció sobre el hecho.
Como mucho 3 indispensables y 5 en total: preguntar de más es la queja.

EL CONTRATO (B, compartido con el front y los otros paquetes):
  {"id", "pregunta" (cerrada, ≤220, con «?»), "tipo": "si_no"|"texto",
   "para_que", "problema" (1-based), "indispensable", "afirma_la_parte",
   "cita_escrito", "el_acto": "no_se_pronuncia", "si_si", "si_no",
   "si_no_contesta", "respuesta": None|"si"|"no"|"no_consta"|texto}
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

MAX_INDISPENSABLES = 3
MAX_PREGUNTAS = 5
MAX_PREGUNTA = 220
MAX_RESPUESTA_TEXTO = 3000   # cinco respuestas caben enteras en el tope de 20 mil del contexto

# Dónde viven en la fila: `taller_sesiones.estado.respuestas` (un dato del
# secretario, como `origen.manual`; no una marca calculada del manifiesto).
CLAVE_MARCA = "respuestas"

CABECERA_BLOQUE = "RESPUESTAS DEL SECRETARIO"
FIN_BLOQUE = "FIN DE LAS RESPUESTAS DEL SECRETARIO"

# Quién carga con la prueba, ya normalizado por `analisis_litis`.
CARGAS = ("la_parte", "contraparte", "autoridad")
_TXT_CARGA = {"la_parte": "a la parte que lo afirma", "contraparte": "a la contraparte",
              "autoridad": "a la autoridad"}


def activa() -> bool:
    """¿Rige la bandera «preguntas_al_secretario» en esta petición?"""
    try:
        import contexto_taller as _ct
        return bool(_ct.rediseno("preguntas_al_secretario"))
    except Exception:
        return False


def _txt(x, n: int = 0) -> str:
    s = " ".join(str(x or "").split())
    return s[:n] if n else s


def _plano(x) -> str:
    s = unicodedata.normalize("NFD", str(x or "").lower())
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return " ".join(re.sub(r"[^a-z0-9ñ ]+", " ", s).split())


def _prospera(calificacion):
    try:
        import probabilidad_sentido as _ps
        return _ps.prospera(calificacion)
    except Exception:
        return None


# ═══ LA FORMA DE UNA PREGUNTA ════════════════════════════════════════════════

def pregunta_cerrada(x) -> str:
    """La pregunta con sus signos y dentro del tope; «» si no hay texto.

    Se corta por palabra y se cierra con «…?» antes que perderla: una pregunta
    indispensable que se tira por larga es una premisa que nadie verifica."""
    t = _txt(x).strip(" .;:")
    if not t:
        return ""
    t = t.lstrip("¿").rstrip("?").strip()
    if not t:
        return ""
    t = "¿" + t[0].upper() + t[1:] + "?"
    if len(t) > MAX_PREGUNTA:
        corte = t[:MAX_PREGUNTA - 2].rsplit(" ", 1)[0].rstrip(" ,;:")
        t = corte + "…?"
    return t


def id_de(premisa: dict) -> str:
    """Un id ESTABLE para la misma premisa: no depende del orden en que salga ni
    de cuántas preguntas haya, sólo de lo que se pregunta. Así una respuesta
    guardada sigue apuntando a su pregunta aunque cambie la lista."""
    base = "|".join(_plano((premisa or {}).get(k)) for k in ("segmento", "afirma_la_parte", "pregunta"))
    return "P" + hashlib.sha1(base.encode("utf-8")).hexdigest()[:6]


def normalizar_respuesta(x, tipo: str = "si_no"):
    """None | "si" | "no" | "no_consta" | texto. Lo vacío es «sin contestar»."""
    if x is None:
        return None
    if isinstance(x, bool):
        return "si" if x else "no"
    s = _txt(x)
    if not s:
        return None
    p = _plano(s)
    if p in ("si", "s", "yes", "true", "1", "afirmativo", "cierto"):
        return "si"
    if p in ("no", "n", "false", "0", "negativo", "falso"):
        return "no"
    if p in ("no consta", "no_consta", "noconsta", "nc", "no obra", "no se sabe", "no hay constancia",
             "no consta en autos", "se desconoce", "desconozco", "no lo se"):
        return "no_consta"
    return s[:MAX_RESPUESTA_TEXTO]


# LA RESPUESTA SIGUE A SU PREGUNTA AUNQUE EL ANÁLISIS SE REHAGA (3-oct-2026,
# revisión adversarial). El id de una pregunta hashea el texto del modelo; si el
# secretario reescribe un problema, el análisis se rehace y los ids cambian, y
# sus respuestas quedaban huérfanas: las preguntas volvían sin contestar. Cada
# respuesta guardada lleva además dos ALIAS —la pregunta y la afirmación de la
# parte, normalizadas— con que se reengancha a la misma pregunta con otro id. Los
# alias empiezan por «~» y no entran en la firma (no cambian la clave de las
# propuestas ya guardadas).
_ALIAS = "~"


def _alias_pregunta(texto) -> str:
    t = _plano(pregunta_cerrada(texto))
    return f"{_ALIAS}p:{t}" if t else ""


def _alias_premisa(afirma) -> str:
    t = _plano(afirma)
    return f"{_ALIAS}a:{t}" if t else ""


def respuestas_de(guardadas) -> dict:
    """{id: respuesta} desde lo que se guardó (`estado.respuestas`: {"items": [...]},
    o una lista de {id, respuesta}, o ya un dict). De una lista salen también
    los alias de cada respuesta (ver arriba); el id manda sobre el alias."""
    if isinstance(guardadas, dict) and "items" in guardadas:
        guardadas = guardadas.get("items")
    if isinstance(guardadas, dict):
        return {str(k): v for k, v in guardadas.items() if v not in (None, "")}
    out, alias = {}, {}
    for x in guardadas or []:
        if isinstance(x, dict) and x.get("id"):
            v = normalizar_respuesta(x.get("respuesta"), x.get("tipo") or "si_no")
            if v is not None:
                out[str(x["id"])] = v
                for k in (_alias_pregunta(x.get("pregunta")), _alias_premisa(x.get("afirma_la_parte"))):
                    if k:
                        alias[k] = v
    for k, v in alias.items():
        out.setdefault(k, v)
    return out


def _respuesta_de(respuestas: dict, pid: str, pregunta: str, afirma: str):
    """La respuesta de una pregunta: por su id, o por sus alias."""
    if pid in respuestas:
        return respuestas[pid]
    for k in (_alias_pregunta(pregunta), _alias_premisa(afirma)):
        if k and k in respuestas:
            return respuestas[k]
    return None


# ═══ DE LAS PREMISAS A LAS PREGUNTAS ═════════════════════════════════════════

def indice_principal(problemas: list) -> int:
    """El principal (1-based) por la jerarquía de la fase 3 o la que corrigió el
    secretario; sin marca, el primero (la misma regla que el árbol y la
    deliberación)."""
    for i, p in enumerate(problemas or [], 1):
        if isinstance(p, dict) and str(p.get("jerarquia") or "").strip().lower() == "principal":
            return i
    return 1


def _problema_de(premisa: dict, analisis: dict, n_problemas: int) -> int:
    """El problema de la premisa; si el modelo no lo dijo, el de las razones
    que combate su segmento."""
    try:
        k = int(str(premisa.get("problema") or "").strip())
        if 1 <= k <= max(1, n_problemas):
            return k
    except (TypeError, ValueError):
        pass
    seg = str(premisa.get("segmento") or "")
    if seg and isinstance(analisis, dict):
        comb = set()
        for a in analisis.get("argumentos") or []:
            if isinstance(a, dict) and str(a.get("segmento")) == seg:
                comb |= set(a.get("combate") or [])
        for r_ in analisis.get("razones") or []:
            if isinstance(r_, dict) and r_.get("id") in comb:
                for k in r_.get("problemas") or []:
                    if isinstance(k, int) and 1 <= k <= max(1, n_problemas):
                        return k
    return 0


def _si_no_contesta(premisa: dict) -> str:
    """Lo que supone el motor si nadie contesta: decide la carga de la prueba.
    La pregunta se formula de modo que «sí» confirme lo que afirma la parte.

    UN HECHO DEL PROCEDIMIENTO NO SE DECIDE POR CARGA (3-oct-2026, revisión
    adversarial): «que hizo valer la prescripción» lo verifica el tribunal en
    autos; sin respuesta queda pendiente, no «no demostrado». La carga sólo
    rige para lo que la parte debía probar en el juicio de origen."""
    if premisa.get("se_verifica_en") == "autos":
        return ("Sin respuesta, queda pendiente de verificar en autos: el motor razona con lo que consta y "
                "lo dice; no lo decide la carga de la prueba.")
    carga = str(premisa.get("carga_de") or "la_parte")
    if carga == "la_parte":
        calif = _txt(premisa.get("si_no"))
        return ("Sin respuesta, lo que afirma la parte no se tiene por demostrado: la carga de probarlo "
                "es suya" + (f" ({calif})." if calif else "."))
    calif = _txt(premisa.get("si_si"))
    return (f"Sin respuesta, la carga de la prueba corresponde {_TXT_CARGA.get(carga, 'a la contraparte')}, "
            "que no demostró lo contrario" + (f" ({calif})." if calif else "."))


def preguntas_de(analisis, problemas: list, respuestas: dict | None = None,
                 decide: int | None = None) -> list:
    """Las preguntas (contrato B) de las premisas del análisis. Sólo las de
    premisas que la resolución NO se pronunció; indispensables primero, como
    mucho 3, y como mucho 5 en total. `decide`: número (1-based) del problema
    que decide el asunto, si se sabe; si no, el principal."""
    if not isinstance(analisis, dict):
        return []
    respuestas = respuestas or {}
    n = len(problemas or [])
    pral = indice_principal(problemas)
    deciden = {pral} | ({int(decide)} if isinstance(decide, int) and decide > 0 else set())
    todas, vistas = [], set()
    for prem in analisis.get("premisas") or []:
        if not isinstance(prem, dict):
            continue
        if prem.get("el_acto") != "no_se_pronuncia":
            continue
        # Una omisión de la propia resolución se comprueba leyéndola: no se le
        # pregunta al secretario (3-oct-2026).
        if prem.get("se_verifica_en") == "resolucion":
            continue
        preg = pregunta_cerrada(prem.get("pregunta"))
        if not preg:
            continue
        pid = id_de({**prem, "pregunta": preg})
        if pid in vistas:
            continue
        vistas.add(pid)
        k = _problema_de(prem, analisis, n)
        si_si, si_no = _txt(prem.get("si_si"), 120), _txt(prem.get("si_no"), 120)
        a, b = _prospera(si_si), _prospera(si_no)
        cambia = a is not None and b is not None and a != b
        tipo = "texto" if str(prem.get("tipo") or "") == "texto" else "si_no"
        _r0 = _respuesta_de(respuestas, pid, preg, prem.get("afirma_la_parte"))
        resp = normalizar_respuesta(_r0, tipo) if _r0 is not None else None
        todas.append({
            "id": pid, "pregunta": preg, "tipo": tipo,
            "para_que": _txt(prem.get("para_que"), 300),
            "problema": k,
            "indispensable": bool(k in deciden and cambia),
            "afirma_la_parte": _txt(prem.get("afirma_la_parte"), 400),
            "cita_escrito": _txt(prem.get("cita_escrito"), 400),
            "el_acto": "no_se_pronuncia",
            "si_si": si_si, "si_no": si_no,
            "si_no_contesta": _si_no_contesta(prem),
            "respuesta": resp,
            "premisa": str(prem.get("id") or ""),
        })
    # Indispensables primero, en el orden del análisis; después las demás.
    ind = [p for p in todas if p["indispensable"]]
    for p in ind[MAX_INDISPENSABLES:]:
        p["indispensable"] = False
    ind = ind[:MAX_INDISPENSABLES]
    resto = [p for p in todas if not p["indispensable"]]
    return (ind + resto)[:MAX_PREGUNTAS]


def contestada(p: dict) -> bool:
    return isinstance(p, dict) and p.get("respuesta") not in (None, "")


def pendientes(preguntas: list, respuestas: dict | None = None) -> list:
    """Las INDISPENSABLES sin contestar. Sólo éstas detienen la propuesta."""
    respuestas = respuestas or {}
    out = []
    for p in preguntas or []:
        if not (isinstance(p, dict) and p.get("indispensable")):
            continue
        r_ = p.get("respuesta")
        if r_ in (None, "") and p.get("id") in respuestas:
            r_ = normalizar_respuesta(respuestas[p["id"]], p.get("tipo") or "si_no")
        if r_ in (None, ""):
            out.append(p)
    return out


def con_respuestas(analisis, preguntas: list):
    """Una COPIA del análisis con la respuesta del secretario en su premisa,
    para que `analisis_litis.bloque_propuesta` la imprima junto a ella."""
    if not isinstance(analisis, dict):
        return analisis
    por_prem = {p.get("premisa"): p for p in preguntas or [] if isinstance(p, dict) and p.get("premisa")}
    out = dict(analisis)
    prems = []
    for prem in analisis.get("premisas") or []:
        if isinstance(prem, dict) and prem.get("id") in por_prem:
            q = por_prem[prem["id"]]
            prem = dict(prem, pregunta_id=q["id"], respuesta=q.get("respuesta"),
                        indispensable=q.get("indispensable", False))
        prems.append(prem)
    out["premisas"] = prems
    return out


# ═══ LAS RESPUESTAS, PARA LA PROPUESTA Y EL ESTUDIO ══════════════════════════

_TXT_RESP = {"si": "SÍ", "no": "NO",
             "no_consta": "NO CONSTA EN AUTOS (se lee como no acreditado)"}


def bloque_respuestas(preguntas: list, respuestas: dict | None = None) -> str:
    """Las preguntas CONTESTADAS, como dato de autos según quien firma. «» si no
    hay ninguna. Va delante de las constancias y sin recorte (contrato C)."""
    respuestas = respuestas or {}
    filas = []
    for p in preguntas or []:
        if not isinstance(p, dict) or not p.get("pregunta"):
            continue
        r_ = p.get("respuesta")
        if r_ in (None, "") and p.get("id") in respuestas:
            r_ = normalizar_respuesta(respuestas[p["id"]], p.get("tipo") or "si_no")
        if r_ in (None, ""):
            continue
        filas.append((p, r_))
    if not filas:
        return ""
    L = [CABECERA_BLOQUE + " — constan en autos según quien firma el proyecto:"]
    for p, r_ in filas:
        L.append(f"  · {p['pregunta']}" + (f" (planteamiento {p['problema']})" if p.get("problema") else ""))
        L.append(f"    Respuesta: {_TXT_RESP.get(r_, '«' + str(r_) + '»')}")
    L.append("  Lo contestado aquí es lo que obra en el expediente: se afirma como hecho de autos, sin "
             "decir quién lo informó. «No consta» se lee como NO ACREDITADO y decide la carga de la "
             "prueba.")
    L.append(FIN_BLOQUE)
    return "\n".join(L) + "\n"


def separar_bloque(contexto: str) -> tuple:
    """(bloque de respuestas, el resto del contexto). Para que el estudio lo
    reciba ENTERO aunque lo demás se recorte."""
    c = contexto or ""
    i = c.find(CABECERA_BLOQUE)
    if i < 0:
        return "", c
    j = c.find(FIN_BLOQUE, i)
    if j < 0:
        return "", c
    j += len(FIN_BLOQUE)
    return c[i:j].strip() + "\n", (c[:i] + c[j:]).strip()


def firma(respuestas) -> str:
    """La firma de las respuestas: entra en la clave de la propuesta guardada,
    para no servir la que se calculó antes de contestar."""
    d = {k: v for k, v in respuestas_de(respuestas).items() if not str(k).startswith(_ALIAS)}
    if not d:
        return ""
    base = "|".join(f"{k}={_plano(v)}" for k, v in sorted(d.items()))
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]


def items_de_marca(marca, huella: str) -> list:
    """Las respuestas guardadas en `estado.respuestas` si son de ESTE adelanto
    (misma huella); [] si no. Una respuesta a la pregunta de otro adelanto no
    se aplica a éste."""
    if not isinstance(marca, dict) or not huella or marca.get("huella") != huella:
        return []
    return [x for x in (marca.get("items") or []) if isinstance(x, dict) and x.get("id")]


def firma_de_marca(marca, huella: str) -> str:
    return firma(items_de_marca(marca, huella))


def marca_respuestas(huella: str, previas: list, nuevas: list, preguntas: list, ahora: float = 0.0) -> tuple:
    """(marca para `estado.respuestas`, ids ignorados). Mezcla las respuestas
    nuevas [{id, respuesta}] con las previas de este adelanto; sólo se aceptan
    ids de preguntas que existen, y se guarda el texto de la pregunta para que
    el bloque de respuestas no dependa de volver a leer el análisis. Una
    respuesta vacía BORRA la anterior (el secretario se equivocó)."""
    por_id = {p["id"]: p for p in preguntas or [] if isinstance(p, dict) and p.get("id")}
    items = {x["id"]: dict(x) for x in previas or [] if isinstance(x, dict) and x.get("id")}
    ignorados = []
    for x in nuevas or []:
        if not isinstance(x, dict):
            continue
        pid = str(x.get("id") or "")
        q = por_id.get(pid)
        if q is None:
            if pid:
                ignorados.append(pid)
            continue
        v = normalizar_respuesta(x.get("respuesta"), q.get("tipo") or "si_no")
        # LA RESPUESTA ANTERIOR A LA MISMA PREGUNTA CON OTRO ID (se reenganchó
        # por alias tras rehacerse el análisis) se sustituye, no se acumula: si
        # no, el bloque de respuestas diría «sí» y «no» a la misma pregunta.
        _al = {k for k in (_alias_pregunta(q.get("pregunta")), _alias_premisa(q.get("afirma_la_parte"))) if k}
        for _old in [k for k, it in items.items() if k != pid and _al & {
                _alias_pregunta(it.get("pregunta")), _alias_premisa(it.get("afirma_la_parte"))}]:
            items.pop(_old, None)
        if v is None:
            items.pop(pid, None)
            continue
        items[pid] = {"id": pid, "pregunta": q.get("pregunta", ""), "tipo": q.get("tipo", "si_no"),
                      "problema": q.get("problema", 0), "para_que": q.get("para_que", ""),
                      "afirma_la_parte": q.get("afirma_la_parte", ""),
                      "respuesta": v, "en": ahora}
    return {"huella": huella, "items": list(items.values()), "firma": firma(list(items.values()))}, ignorados


def resellar_tras_corregir(estado: dict, huella_antes: str, huella_nueva: str,
                           huella_analisis_nueva: str = "") -> list:
    """CORREGIR EL PROBLEMA NO TIRA LAS RESPUESTAS (3-oct-2026, revisión
    adversarial). /taller/problema reescribe `estado.huella` (la jerarquía y la
    pregunta entran en `huella_contraste`) y, con la huella vieja, las
    respuestas dejaban de leerse —las preguntas volvían sin contestar y el
    estudio se escribía sin lo que el secretario confirmó— y el análisis se
    volvía a pagar aunque lo que lee no hubiera cambiado.

    Sobre `estado` (la fila, que el llamador escribe después), en su sitio:
      · `respuestas`: si eran de ESTE adelanto (huella vieja), pasan a la
        nueva. Son hechos de autos: no dependen de cómo se formula el problema;
      · `analisis`: sólo si lo que lee no cambió (`huella_analisis` igual a la
        recalculada; la jerarquía no entra en ella). Si se reescribió la
        pregunta, el análisis se rehace y las respuestas se reenganchan por
        alias (ver `respuestas_de`).
    Devuelve las claves reselladas. Puro; nunca lanza."""
    hechas = []
    try:
        if not isinstance(estado, dict) or not huella_antes or not huella_nueva or huella_antes == huella_nueva:
            return hechas
        m = estado.get(CLAVE_MARCA)
        if isinstance(m, dict) and m.get("huella") == huella_antes:
            m["huella"] = huella_nueva
            hechas.append(CLAVE_MARCA)
        a = estado.get("analisis")
        if isinstance(a, dict) and a.get("huella") == huella_antes and huella_analisis_nueva \
                and a.get("huella_analisis") == huella_analisis_nueva and a.get("estado") == "listo":
            a["huella"] = huella_nueva
            hechas.append("analisis")
    except Exception:
        pass
    return hechas


def respuesta_preguntas(numero: str, preguntas: list, modelo: str = "", avisos: list | None = None,
                        origen_firma: str = "", firma_respuestas: str = "") -> dict:
    """La respuesta de /taller/proponer cuando faltan respuestas indispensables
    (contrato A, estado «preguntas»): sin propuestas, sin global y SIN llamar
    al modelo de la propuesta."""
    pend = pendientes(preguntas)
    av = list(avisos or [])
    av.insert(0, (f"Antes de proponer, el motor necesita {len(pend)} respuesta"
                  f"{'s' if len(pend) != 1 else ''} tuya{'s' if len(pend) != 1 else ''}: "
                  "la resolución no dice nada sobre "
                  f"{'esos hechos' if len(pend) != 1 else 'ese hecho'} y de "
                  f"{'ellos' if len(pend) != 1 else 'él'} depende el sentido."))
    return {
        "expediente": numero,
        "modelo": modelo,
        "estado": "preguntas",
        "formato": 3,
        "preguntas": list(preguntas or []),
        "pendientes": len(pend),
        "propuestas": [],
        "global": None,
        "resumen": "",
        "contraste": [],
        "necesita_conceptos": False,
        "conceptos_omitidos": None,
        "avisos": av,
        "internet": None,
        "origen_firma": origen_firma,
        "respuestas_firma": firma_respuestas,
        "criterios_json": "[]",
    }


# ═══ LAS FRASES DE HERRAMIENTA ═══════════════════════════════════════════════
# El tribunal TIENE el expediente: una sentencia no dice que algo «no obra en el
# material proporcionado» ni habla de «los insumos» o de «las constancias
# remitidas para este estudio». Eso es la voz del sistema que preparó el
# proyecto y se cuela en la prosa que se firma. Se escribe «de la resolución
# recurrida no se advierte que…» o «la recurrente no demuestra que…».
#
# CALIBRADO PARA NO ACUSAR A UNA SENTENCIA BUENA: «la información proporcionada
# por la autoridad», «lo aportado por la quejosa como prueba» o «los insumos de
# producción» (revisión fiscal) son legítimos y no se cuentan.
#
# 3-oct-2026 (revisión adversarial: «el aviso acusa frases buenas»): tampoco lo
# que lleva DESTINO o FUENTE PROCESAL —«de lo aportado al juicio no se
# advierte», «con lo aportado en autos se acredita», «la información
# proporcionada en respuesta al requerimiento», «la documentación proporcionada
# a la autoridad fiscalizadora», «el expediente proporcionado al perito»—: son
# giros normales, sobre todo en la revisión fiscal. La voz de la herramienta es
# el término suelto o referido a este estudio («el material proporcionado», «la
# información proporcionada no permite…», «…para este estudio»). La voz del
# secretario como fuente («aportado por el secretario») la caza su propia
# expresión, aparte.
_DESTINO_PROCESAL = (r"(?!\s+(?:por|a|al|ante|como|durante|dentro|mediante|desde"
                     r"|en\s+(?:respuesta|autos|juicio|el\s+juicio|la\s+audiencia|el\s+procedimiento"
                     r"|el\s+expediente|la\s+instancia|primera|segunda|la\s+visita|la\s+revisi[oó]n"
                     r"|el\s+requerimiento|cumplimiento|la\s+diligencia|el\s+sumario|la\s+contestaci[oó]n"
                     r"|la\s+demanda|el\s+recurso|la\s+etapa|el\s+periodo|el\s+per[ií]odo|el\s+t[eé]rmino)"
                     r"|en\s+(?:la|el|las|los)\s+(?:quejos|recurrent|tercer|actor|actora|demandad|autoridad"
                     r"|parte|oferente|trabajador|patr[oó]n|inconforme|promovente))\b)")
_FRASES_HERRAMIENTA = [
    r"\bmaterial(?:es)?\s+(?:que\s+(?:se\s+)?(?:me\s+)?(?:ha\s+sido\s+|fue\s+)?)?(?:proporcionad|entregad|"
    r"disponibl|remitid|recibid|facilitad|suministrad)\w*\b" + _DESTINO_PROCESAL,
    r"\b(?:no\s+)?obra\s+en\s+(?:el|lo|los)\s+(?:material|proporcionad\w*|insumos|documentos?\s+proporcionad\w*)",
    r"\bconstancias\s+(?:remitidas|proporcionadas|entregadas|aportadas|recibidas|disponibles)\s+"
    r"(?:para|a)\s+(?:este|el\s+presente)\s+(?:estudio|an[aá]lisis|proyecto)",
    r"\bconstancias\s+(?:proporcionadas|entregadas|disponibles)\b" + _DESTINO_PROCESAL,
    r"\b(?:en|entre|de|con)\s+los\s+insumos(?=\s*[\.,;:)]|\s*$|\s+(?:no\b|proporcionad|entregad|disponibl|"
    r"recibid|del\s+(?:presente\s+)?(?:estudio|proyecto|asunto)|con\s+que\s+se\s+cuenta))",
    r"\blos\s+insumos\s+(?:proporcionad|entregad|disponibl|recibid)\w*",
    r"\blo\s+aportado\b" + _DESTINO_PROCESAL,
    r"\b(?:informaci[oó]n|documentaci[oó]n|texto|textos|documentos?|expediente)\s+(?:que\s+se\s+(?:me\s+)?"
    r"(?:ha\s+|han\s+)?)?(?:proporcionad|facilitad|suministrad)\w*\b" + _DESTINO_PROCESAL,
    r"\b(?:aportad|proporcionad|remitid|informad)[oa]s?\s+por\s+(?:el|la)\s+secretari[oa]\b",
    r"\b(?:seg[uú]n|conforme\s+a\s+lo\s+que)\s+(?:informa|indica|señala|dice|refiere)\s+(?:el|la)\s+secretari[oa]\b",
    r"\bseg[uú]n\s+(?:informa\s+)?quien\s+firma\b",
    r"\bmodelo\s+de\s+lenguaje\b",
]
_RX_HERRAMIENTA = re.compile("|".join(f"(?:{x})" for x in _FRASES_HERRAMIENTA), re.I)


def frases_de_herramienta(texto: str) -> list:
    """Las frases con voz de herramienta que aparecen en `texto`, sin repetir y
    en orden de aparición. Determinista: sirve de hallazgo al supervisor y de
    aviso al verificar el estudio."""
    out, vistas = [], set()
    for m in _RX_HERRAMIENTA.finditer(texto or ""):
        f = " ".join(m.group(0).split())
        k = f.lower()
        if k not in vistas:
            vistas.add(k)
            out.append(f)
    return out
