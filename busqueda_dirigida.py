"""El acervo como HERRAMIENTA: el modelo busca, lee lo que salió y vuelve a buscar.

POR QUÉ EXISTE. Hoy el acervo entra por PRE-CARGA: `fase6_rag.material_para`
formula UNA consulta por problema, trae lo que trae, y eso es todo lo que el
redactor verá. Si la formulación falla, el estudio se escribe sin la tesis que
decidía el punto y nadie se entera —la sentencia sale «bien» y sin autoridad—.

Y sabemos que la formulación falla. Medido sobre el banco de 404 tesis reales
citadas en engroses (`redactor-sentencias/rag/`, 14-sep-2026):

    techo (buscar con el rubro de la propia tesis) ....... 99 % en top-10
    producción (una formulación, sin ver la tesis) ....... 19 % en top-10

Ochenta puntos de distancia, y la nota de aquel día ya decía dónde estaba el
hueco: «el hueco es cómo se formula la consulta, no materia ni precepto». Un
solo tiro a ciegas contra un acervo de dos millones de fragmentos.

LO QUE CAMBIA AQUÍ es quién formula y cuántas veces. El modelo pide una
búsqueda, RECIBE LOS RUBROS, y con eso delante decide si ya tiene lo que
necesita o si su pregunta iba desviada y la reformula. Es la diferencia entre
adivinar la palabra y mirar el índice de la biblioteca.

(OpenAI publicó el 17-sep-2026 la misma idea para el derecho estadounidense
—modelo + índice jurídico como herramienta— y midió el salto: 38.7 % → 54.0 %
de corrección. Su índice es de EE. UU. y no nos sirve; el patrón sí, y el
índice de México es nuestro.)

TRES REGLAS QUE NO SE NEGOCIAN:

1. NADA ENTRA SIN PASAR POR EL ACERVO. La herramienta sólo devuelve lo que
   Qdrant tiene, con su registro. El modelo no puede «recordar» una tesis por
   esta vía: si no está, la respuesta es que no está. Es la misma doctrina de
   [[regla8-denuncia]] y del sello de citas, ahora del lado de la entrada.

2. LO QUE ENCUENTRA SE SUMA AL MATERIAL. Si el modelo cita una tesis que trajo
   la herramienta y el material no la tiene, nuestros propios verificadores la
   acusarían como inventada («REGISTROS QUE NO ESTÁN EN EL MATERIAL»). Por eso
   `Hallazgos.al_material()` la mete donde el resto del pipeline la ve.

3. CON TOPE. Cada vuelta es una llamada al modelo y una a Qdrant. El tope de
   llamadas y el de resultados son argumentos, no adornos: sin ellos, un
   modelo indeciso convierte un estudio de cuatro minutos en uno de quince.

LA RESTRICCIÓN DEL PROVEEDOR, probada contra la API el 20-sep-2026, que
decide la forma de todo esto. En `/v1/chat/completions`:

    tools + reasoning_effort=low/high .... 400 «Function tools with
                                           reasoning_effort are not supported»
    tools SIN reasoning_effort ........... 400 EL MISMO ERROR: luna razona por
                                           omisión, así que hay que pedir
                                           explícitamente el «none»
    tools + reasoning_effort="none" ...... FUNCIONA
    gpt-6-astra + effort="none" .......... 400: ese modelo no admite «none»,
                                           o sea que con él esta vía no existe

Y en `/v1/responses`: **herramientas CON razonamiento alto, funciona**
(probado: devuelve `reasoning` y `function_call` en la misma respuesta).

De ahí las dos reglas. Mientras BUSCA, sin razonamiento —buscar es mirar el
índice, no argumentar— y la vuelta final, ya sin herramientas, recupera el
esfuerzo que pidió quien llamó, que es donde se redacta. Y si algún día el
modelo debe buscar EN MITAD de un estudio con razonamiento alto —que es a
donde esto apunta—, esa llamada se muda a `/v1/responses`; no cabe aquí. Eso
NO es sólo una limitación: es la razón por la que este módulo se estrena como
paso aparte y no metido en `fase6_estudio`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os

log = logging.getLogger("busqueda_dirigida")

# Cuántas veces puede el modelo pedir búsquedas en una misma conversación.
# Tres es lo medido como suficiente: la primera va a ciegas, la segunda corrige
# el tiro con los rubros delante, la tercera cubre un segundo tema.
TOPE_VUELTAS = int(os.getenv("BUSQUEDA_DIRIGIDA_VUELTAS", "3"))
# Cuántas tesis ve el modelo por búsqueda. Más no ayuda —está medido en el
# rerank: con el acervo entero delante elige la que suena, no la que aplica— y
# cada una son ~40 palabras de rubro en su contexto.
TOPE_RESULTADOS = int(os.getenv("BUSQUEDA_DIRIGIDA_RESULTADOS", "8"))


def herramientas() -> list:
    """El contrato que ve el modelo. Los nombres y las descripciones son parte
    del prompt: dicen QUÉ es cada cosa en el vocabulario del secretario."""
    return [
        {"type": "function", "function": {
            "name": "buscar_tesis",
            "description": (
                "Busca jurisprudencia y tesis aisladas mexicanas en el acervo "
                "verificado (Semanario Judicial de la Federación). Devuelve el "
                "registro, el rubro y si es obligatoria. Úsala cuando necesites "
                "la autoridad que resuelve un punto y no la tengas delante. "
                "Puedes llamarla varias veces: si lo que vuelve no es del tema, "
                "reformula y vuelve a buscar."),
            "parameters": {
                "type": "object",
                "properties": {
                    "consulta": {
                        "type": "string",
                        "description": (
                            "La cuestión jurídica REDACTADA COMO UN RUBRO, no "
                            "como pregunta: sujeto, figura y consecuencia, en "
                            "el vocabulario del Semanario. Ejemplo de la forma "
                            "(no del tema): «NOTIFICACIÓN POR ESTRADOS. "
                            "REQUISITOS DE VALIDEZ CUANDO EL DOMICILIO ESTÁ "
                            "CERRADO».")},
                    "materia": {
                        "type": "string",
                        "description": "administrativa, civil, laboral, penal o común. Opcional."},
                },
                "required": ["consulta"]}}},
        {"type": "function", "function": {
            "name": "leer_articulo",
            "description": (
                "Devuelve el TEXTO VIGENTE de un artículo de una ley mexicana, "
                "tal como está en el acervo. Úsala antes de transcribir o "
                "interpretar un precepto que no tengas delante; nunca lo cites "
                "de memoria."),
            "parameters": {
                "type": "object",
                "properties": {
                    "ley": {"type": "string",
                            "description": "Nombre completo del ordenamiento."},
                    "articulo": {"type": "string",
                                 "description": "Número del artículo, sin la palabra «artículo»."},
                },
                "required": ["ley", "articulo"]}}},
        {"type": "function", "function": {
            "name": "comprobar_registro",
            "description": (
                "Comprueba si existe una tesis con ese número de registro y "
                "devuelve su rubro. Úsala SIEMPRE que vayas a citar un registro "
                "que recuerdes: si no existe, no lo cites."),
            "parameters": {
                "type": "object",
                "properties": {
                    "registro": {"type": "string",
                                 "description": "El número de registro digital."},
                },
                "required": ["registro"]}}},
    ]


class Hallazgos:
    """Lo que las búsquedas trajeron, para que el resto del pipeline lo vea."""

    def __init__(self):
        self.tesis: list[dict] = []
        self.normas: list[dict] = []
        self.consultas: list[str] = []
        self.vueltas: int = 0

    def registros(self) -> list:
        return [str(t.get("registro") or "") for t in self.tesis]

    def al_material(self, material) -> tuple:
        """Suma al material lo nuevo. Devuelve (tesis nuevas, normas nuevas)."""
        if material is None:
            return 0, 0
        # LA MISMA TESIS NO ENTRA DOS VECES, y la trampa no era el material sino
        # ESTA lista: el modelo encuentra un registro con `buscar_tesis` y
        # luego lo pasa por `comprobar_registro`, así que aparece dos veces en
        # `self.tesis` y las dos pasaban el filtro contra el material. Visto en
        # el 2/2026: «3 tesis nuevas · 2018365, 174219, 174219».
        _ya = {str(t.get("registro") or "") for t in (material.tesis or [])}
        nuevas = []
        for t in self.tesis:
            reg = str(t.get("registro") or "")
            if not reg or reg in _ya:
                continue
            _ya.add(reg)
            nuevas.append(t)
        if nuevas:
            material.tesis = list(material.tesis or []) + nuevas
        _yan = {(str(n.get("cuerpo_legal", "")).lower(), str(n.get("articulo", "")))
                for n in (material.normas or [])}
        nn = []
        for n in self.normas:
            k = (str(n.get("cuerpo_legal", "")).lower(), str(n.get("articulo", "")))
            if k in _yan:
                continue
            _yan.add(k)
            nn.append(n)
        if nn:
            material.normas = list(material.normas or []) + nn
        return len(nuevas), len(nn)


async def _ejecutar(qdrant, embed_juris, nombre: str, args: dict,
                    hallazgos: Hallazgos, materia: str = "",
                    coleccion_estatal: str = "") -> str:
    """Corre una herramienta y devuelve lo que el modelo leerá. SIEMPRE texto:
    un error aquí no puede tumbar el estudio, sólo decir que no hubo nada."""
    import fase6_rag as _f6r
    try:
        if nombre == "buscar_tesis":
            consulta = " ".join(str(args.get("consulta") or "").split())
            if not consulta:
                return "Sin consulta: no se buscó nada."
            hallazgos.consultas.append(consulta)
            v = await embed_juris(consulta)
            crudas = await _f6r._buscar(qdrant, _f6r.COLECCION_JURIS, "rubro", v,
                                        TOPE_RESULTADOS * 3)
            vistos, salida = set(hallazgos.registros()), []
            for p in crudas:
                t = _f6r._tesis_de(p)
                if not t["registro"] or t["registro"] in vistos:
                    continue
                vistos.add(t["registro"])
                salida.append(t)
                if len(salida) >= TOPE_RESULTADOS:
                    break
            hallazgos.tesis.extend(salida)
            if not salida:
                return "El acervo no devolvió nada para esa consulta. Reformúlala."
            return json.dumps([
                {"registro": t["registro"], "rubro": t["rubro"],
                 "instancia": t["instancia"],
                 "obligatoria": t["obligatoria"]} for t in salida], ensure_ascii=False)

        if nombre == "leer_articulo":
            ley = str(args.get("ley") or "").strip()
            art = str(args.get("articulo") or "").strip()
            if not (ley and art):
                return "Falta la ley o el artículo."
            d = await _f6r.resolver_articulo(qdrant, coleccion_estatal or "", ley, art)
            if not d:
                return (f"El acervo no tiene el artículo {art} de «{ley}». "
                        f"NO lo transcribas de memoria.")
            hallazgos.normas.append(d)
            return json.dumps({"cuerpo_legal": d.get("cuerpo_legal") or ley,
                               "articulo": d.get("articulo") or art,
                               "texto": (d.get("texto") or "")[:4000]}, ensure_ascii=False)

        if nombre == "comprobar_registro":
            reg = str(args.get("registro") or "").strip()
            if not reg:
                return "Falta el registro."
            fichas = await _f6r.tesis_por_registro(qdrant, [reg])
            if not fichas:
                return (f"NO EXISTE ninguna tesis con el registro {reg} en el "
                        f"acervo. No la cites.")
            hallazgos.tesis.extend(fichas)
            f = fichas[0]
            return json.dumps({"registro": f.get("registro"), "rubro": f.get("rubro"),
                               "instancia": f.get("instancia"),
                               "obligatoria": f.get("obligatoria")}, ensure_ascii=False)
    except Exception as e:
        log.error("herramienta %s falló: %s: %s", nombre, type(e).__name__, e)
        return "La búsqueda falló. Sigue con lo que tengas."
    return f"No existe la herramienta «{nombre}»."


async def con_acervo(cliente, qdrant, embed_juris, kw: dict,
                     materia: str = "", coleccion_estatal: str = "",
                     tope: int = TOPE_VUELTAS) -> tuple:
    """Una llamada al modelo que PUEDE buscar en el acervo. Devuelve
    (respuesta final, Hallazgos).

    `kw` es lo que se le pasaría a `llamada_modelo.crear`; aquí se le añaden
    las herramientas y se atienden sus peticiones hasta `tope` vueltas. Si el
    modelo no pide ninguna, esto cuesta exactamente lo mismo que antes.
    """
    import llamada_modelo as _lm
    hallazgos = Hallazgos()
    kw = dict(kw)
    esfuerzo = kw.get("reasoning_effort")     # el de quien llamó: se devuelve al final
    kw["tools"] = herramientas()
    # Ver la cabecera: con herramientas hay que pedir el «none» EXPRESAMENTE;
    # omitirlo da el mismo 400, porque este modelo razona por omisión.
    kw["reasoning_effort"] = "none"
    mensajes = list(kw.get("messages") or [])
    for vuelta in range(tope + 1):
        kw["messages"] = mensajes
        if vuelta == tope:
            # Última: sin herramientas y CON el razonamiento de quien llamó,
            # porque es la vuelta en la que se escribe.
            kw.pop("tools", None)
            if esfuerzo:
                kw["reasoning_effort"] = esfuerzo
        r = await _lm.crear(cliente, **kw)
        msg = r.choices[0].message
        llamadas = list(getattr(msg, "tool_calls", None) or [])
        # EN LA VUELTA FINAL YA NO SE ATIENDE NADA. Si el modelo insiste en
        # pedir con las herramientas retiradas, se queda con lo que tiene: el
        # tope es un tope, no una sugerencia.
        if not llamadas or "tools" not in kw:
            return r, hallazgos
        hallazgos.vueltas += 1
        mensajes.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.function.name,
                                         "arguments": c.function.arguments}}
                           for c in llamadas]})
        resultados = await asyncio.gather(*[
            _ejecutar(qdrant, embed_juris, c.function.name,
                      _args(c.function.arguments), hallazgos, materia,
                      coleccion_estatal) for c in llamadas])
        for c, res in zip(llamadas, resultados):
            mensajes.append({"role": "tool", "tool_call_id": c.id, "content": res})
        print(f"   🔎 acervo: {len(llamadas)} búsqueda(s) del modelo · "
              f"{' · '.join(_resumir(c) for c in llamadas)}")
    return r, hallazgos


# ═══════════════════════════════════════════════════════════════════════════
# EL PASO DE PRODUCCIÓN: rellenar lo que la pre-carga no encontró
# ═══════════════════════════════════════════════════════════════════════════

_PROMPT_REFUERZO = """Eres secretario proyectista de un Tribunal Colegiado mexicano.
Vas a resolver este planteamiento del asunto:

    {problema}

{contexto}LO QUE YA TIENES para ese punto, traído del acervo:
{teniendo}

TU TAREA NO ES RESOLVERLO: es comprobar si te falta autoridad y traerla.
Mira los rubros de arriba. Si entre ellos NO está la tesis que de verdad
resuelve el punto, búscala con la herramienta; mira lo que vuelva y, si va
desviado, reformula y vuelve a buscar. Si necesitas el texto de un artículo
que no tienes, léelo con la herramienta.

Si lo que ya tienes basta, no busques nada: dilo y ya.

Termina con una línea, sin adornos:
    FALTABA: <registros nuevos separados por coma, o «nada»>"""


async def reforzar(cliente, qdrant, embed_juris, problemas: list, material,
                   materia: str = "", coleccion_estatal: str = "",
                   tope_problemas: int = 6) -> dict:
    """Busca lo que la pre-carga no encontró y lo suma al material.

    Corre DESPUÉS de `redactor_adelanto.consultar` y ANTES de proponer, que es
    cuando el material ya está y todavía no se ha decidido nada con él. Un
    planteamiento por llamada y todos a la vez: son segundos, no minutos.

    Medido sobre el banco de 404 tesis reales (20-sep-2026), buscando la tesis
    que el engrose de verdad citó: el camino de producción —RRF + rerank— la
    trae en el 28 % de los casos; sumando lo que encuentra esto, sube. Los
    números finales, en [[busqueda-dirigida]].
    """
    if material is None or not problemas:
        return {}
    _ya = {str(t.get("registro") or "") for t in (material.tesis or [])}

    async def _uno(p) -> Hallazgos | None:
        preg = (p.get("pregunta") if isinstance(p, dict) else str(p)) or ""
        if not preg.strip():
            return None
        # Lo que ya tiene, para que busque lo que FALTA y no lo mismo.
        rubros = [f"  · {t.get('rubro') or ''}" for t in (material.tesis or [])][:14]
        _ctx = ""
        if isinstance(p, dict) and (p.get("resolvio") or p.get("combate")):
            _ctx = (f"Lo que resolvió el órgano: {p.get('resolvio') or '(no consta)'}\n"
                    f"Lo que se combate: {p.get('combate') or '(no consta)'}\n\n")
        kw = dict(model=os.getenv("MODELO_REFUERZO",
                                  os.getenv("MODELO_FASES", "gpt-5.6-luna")),
                  temperature=0, seed=20260920, max_completion_tokens=4000,
                  messages=[{"role": "user", "content": _PROMPT_REFUERZO.format(
                      problema=preg.strip(), contexto=_ctx,
                      teniendo="\n".join(rubros) or "  (nada)")}])
        try:
            r, hall = await con_acervo(cliente, qdrant, embed_juris, kw,
                                       materia, coleccion_estatal)
            # SÓLO LO QUE EL MODELO SE QUEDÓ. Sus búsquedas devuelven ocho
            # rubros cada una y quedarse con todos dobla el material: medido en
            # el 2/2026, 37 tesis nuevas de golpe. Y está medido también lo que
            # eso hace —el rerank existe por esto—: con el acervo entero
            # delante, el modelo elige la que suena, no la que aplica. Se
            # conserva la unión en `hall` para el registro, pero al material
            # entra lo que declaró en su línea FALTABA. Con REFUERZO_TODO=1
            # entra todo, que en el banco recupera unos puntos más a costa de
            # precisión.
            if os.getenv("REFUERZO_TODO", "0") != "1":
                quedadas = _faltaba((r.choices[0].message.content or ""))
                hall.tesis = [t for t in hall.tesis
                              if str(t.get("registro") or "") in quedadas]
            return hall
        except Exception as e:
            log.error("refuerzo de «%s…» falló: %s: %s", preg[:40], type(e).__name__, e)
            return None

    halls = await asyncio.gather(*[_uno(p) for p in problemas[:tope_problemas]])
    tesis_nuevas = normas_nuevas = consultas = 0
    for h in halls:
        if h is None:
            continue
        t, n = h.al_material(material)
        tesis_nuevas += t
        normas_nuevas += n
        consultas += len(h.consultas)
    nuevas_reg = [str(t.get("registro") or "") for t in (material.tesis or [])
                  if str(t.get("registro") or "") not in _ya]
    print(f"   🔎 REFUERZO del acervo: {consultas} búsqueda(s) en "
          f"{len(halls)} planteamiento(s) · {tesis_nuevas} tesis y "
          f"{normas_nuevas} precepto(s) nuevos"
          + (f" · {', '.join(nuevas_reg[:6])}" if nuevas_reg else ""))
    return {"tesis": tesis_nuevas, "normas": normas_nuevas,
            "consultas": consultas, "registros": nuevas_reg}


def _faltaba(texto: str) -> set:
    """Los registros de la línea «FALTABA: …». Vacío si no la escribió o si
    dijo «nada»: en la duda no se mete nada al material."""
    import re
    m = re.search(r"FALTABA\s*:\s*([^\n]*)", texto or "", re.I)
    if not m or "nada" in m.group(1).lower():
        return set()
    return {x for x in re.findall(r"\d{4,9}", m.group(1))}


def _args(crudo) -> dict:
    try:
        d = json.loads(crudo or "{}")
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _resumir(c) -> str:
    a = _args(c.function.arguments)
    if c.function.name == "buscar_tesis":
        return f"«{str(a.get('consulta') or '')[:60]}»"
    if c.function.name == "leer_articulo":
        return f"art. {a.get('articulo')} de {str(a.get('ley') or '')[:40]}"
    return f"registro {a.get('registro')}"
