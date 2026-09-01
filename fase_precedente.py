"""EL PASO DE PRECEDENTE — cómo resolvieron otros el mismo problema.

David, 30-ago-2026: «hay una cosa que hemos pasado por alto y que desde mi punto
de vista vale oro. En Iurexia tenemos precedentes de tribunales colegiados de
circuito… miles de sentencias almacenadas… la idea es superarlos».

Tenía razón en que valía oro y en dónde estaba. Lo que faltaba era saber QUÉ
sacar de ahí, y eso se midió: cuatro agentes leyeron 966 sentencias de calidad
alta y 980 de calidad media, en laboral, administrativa, civil y penal, y
recuperaron el estudio de fondo completo de las mejores. El acervo se puntuó a
sí mismo; nosotros sólo medimos la diferencia.

LO QUE SALIÓ, Y QUE NO ES LO QUE YO HABRÍA SUPUESTO:

1. El resumen del holding NO SIRVE PARA MEDIR CALIDAD. Sus metadatos son casi
   idénticos entre una sentencia de calidad 5 y una de 3, y su `tesis_registros`
   dice 3.14 contra 2.70 cuando en el texto real son 6.33 contra 3.46. El
   holding sirve para RECUPERAR. La señal entera está en el estudio de fondo.

2. LA DIFERENCIA MÁS GRANDE NO ES CITAR MÁS: es escribir la operación que
   normalmente se queda en la cabeza. Derivar la regla en abstracto antes de
   aplicarla aparece en el 33% de las mejores civiles y en el 3.3% de las
   medias. Es la mayor diferencia relativa de todo el análisis.

3. EL PASO QUE HOY NO EXISTÍA EN NINGUNA PARTE: responder la mejor objeción del
   que pierde. 93% de las laborales de calidad máxima contra 58% de las medias.
   Y ese argumento contrario no hay que inventarlo: está en el acervo, en las
   sentencias que resolvieron lo mismo al revés. De ahí este módulo.

LA TRAMPA DE LA MATERIA, y cuesta el 6% de las mejores: el campo tiene DOS
grafías. Hay 45,466 sentencias con `materia: "civil"` y otras 1,135 con
`"Civil"`; 44,938 con `"laboral"` y 1,533 con `"Trabajo"`. Filtrar sólo por la
minúscula —que es lo que hace cualquiera que mire un ejemplo— tira ese resto por
la ventana. Se filtra con `any` de las dos, siempre.

LA OTRA TRAMPA, y ésta la traía mal hasta la propia especificación: el enlace
entre el resumen y el estudio NO EXISTE como campo indexado. `sentencias_holdings`
no tiene `holding_id` —lo comprobé pidiéndolo y Qdrant contestó que no hay índice
ni campo— y en las colecciones `ef` el `neun` que sí los emparejaría no está
indexado. Unir las dos colecciones exigiría crear índices sobre 1.26 millones de
puntos, que es tocar producción.

EL ENLACE SÍ EXISTE, PERO NO ES UN CAMPO: es el propio identificador del punto.
El `holding_id` que las colecciones `ef` traen indexado en su payload resulta ser
el id del punto correspondiente en `sentencias_holdings`. Lo comprobé sobre tres
sentencias de calidad 5 de circuitos distintos y las tres enlazan. Buscarlo como
campo devolvía «no hay índice» y parecía que la unión era imposible; estaba
delante, un piso más arriba.

Con eso el molde sale barato: la búsqueda de precedentes ya filtra por calidad
—que en `holdings` sí está indexada— y los estudios se recuperan por el id de
esos mismos puntos. Ni una búsqueda vectorial de más, ni una escritura.

Y una tercera, pequeña y silenciosa: `circuito` está indexado como TEXTO. Filtrar
con el número 3 devuelve un 400 que, si nadie lo mira, parece «no hay
precedentes». Se pasa como cadena.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from dataclasses import dataclass, field

COL_HOLDINGS = "sentencias_holdings"
COL_JURIS = "jurisprudencia_nacional_v3"

# Sólo estos circuitos tienen el estudio de fondo troceado en Qdrant.
CIRCUITOS_CON_ESTUDIO = {1, 2, 3, 4, 22}

# Las dos grafías de cada materia. Medidas, no supuestas.
GRAFIAS = {
    "civil": ["civil", "Civil"],
    "laboral": ["laboral", "Trabajo"],
    "administrativa": ["administrativa", "Administrativa"],
    "penal": ["penal", "Penal"],
}

# Cuántos precedentes se leen. Cuarenta por tiro es lo que midió el análisis;
# más allá el sondeo tarda más de lo que aporta.
TOP_SONDEO = 40
# CUÁNTOS MOLDES SE QUIEREN Y CUÁNTOS SE INTENTAN. No todo estudio bien
# puntuado sirve de molde: hace falta que tenga una fórmula de derivación
# —«De dicho numeral se advierte que…»— y que el tramo hasta el primer anclaje
# al caso ajeno dé para algo. Midiendo salió que aproximadamente la mitad de los
# candidatos no la tienen, así que se piden el doble y se conservan los buenos.
# Cuesta cuatro scrolls más, que corren en paralelo y no se notan.
MAX_MOLDES = 3          # los que se entregan al redactor
CANDIDATOS_MOLDE = 8    # los que se leen para conseguirlos
MAX_OBJECION = 6        # sentencias que resolvieron al revés


@dataclass
class Sondeo:
    """Cómo resolvió el acervo este mismo problema."""
    distribucion: dict = field(default_factory=dict)   # sentido -> cuántas
    fundamentos: list = field(default_factory=list)    # los recurrentes, por frecuencia
    concordantes: list = field(default_factory=list)   # los holdings más cercanos
    claves: list = field(default_factory=list)         # tesis que ellos citaron
    # TODOS los precedentes con su sentido y su razón, para poder sacar la
    # objeción DESPUÉS. El sondeo corre antes de que exista sentido propuesto
    # —ése es justamente su motivo: enseñarle al redactor cómo se resuelve esto
    # antes de que decida— así que en ese momento no se sabe todavía qué es «lo
    # contrario». Se guarda el material y se filtra cuando el sentido ya existe.
    razonados: list = field(default_factory=list)
    objeciones: list = field(default_factory=list)     # cómo razonó el que resolvió al revés
    moldes: list = field(default_factory=list)         # tramos de REGLA, molde de forma
    avisos: list = field(default_factory=list)
    # La predicción de CADA problema: [{"problema", "prediccion"}]. El sondeo
    # se hace uno por problema y el más rico se queda como `sondeo` del
    # material —que es lo que lee el estudio—; esto lleva los demás a la
    # pantalla, que es donde el secretario califica.
    por_problema: list = field(default_factory=list)

    def objeciones_contra(self, sentido: str) -> list:
        """Cómo razonó el que resolvió al revés, ya sabido el sentido."""
        return _objeciones_de(self.razonados, sentido)

    def concuerda(self, sentido: str) -> tuple:
        """(¿va con la corriente?, cuánta corriente hay). Para exigir justificación.

        No se bloquea nada: un tribunal puede apartarse del criterio mayoritario
        —para eso existe la contradicción de tesis— pero tiene que decir que lo
        hace y por qué. Lo que no puede es apartarse sin enterarse.
        """
        total = sum(self.distribucion.values())
        if total < 5:
            return True, 0.0
        mio = self.distribucion.get(_norm_sentido(sentido), 0)
        return (mio / total) >= 0.20, mio / total


def _norm_sentido(s: str) -> str:
    s = (s or "").strip().lower().replace(" ", "_")
    if "ampar" in s and "conced" in s:
        return "concede"
    return s


def _filtro_materia(materia: str) -> list:
    """El `any` de las dos grafías. Sin esto se pierde el 6% de las buenas."""
    from qdrant_client.models import FieldCondition, MatchAny
    g = GRAFIAS.get((materia or "").strip().lower())
    if not g:
        return []
    return [FieldCondition(key="materia", match=MatchAny(any=g))]


async def _esperar(r):
    return await r if inspect.isawaitable(r) else r


async def _buscar(qdrant, vector, debe, top: int) -> list:
    """[(id-del-punto, payload)]. El id es el `holding_id` del estudio de fondo."""
    from qdrant_client.models import Filter
    try:
        r = await _esperar(qdrant.query_points(
            collection_name=COL_HOLDINGS, query=vector, using="dense",
            query_filter=Filter(must=debe) if debe else None,
            limit=top, with_payload=True))
        return [(str(p.id), p.payload) for p in (getattr(r, "points", None) or [])]
    except Exception as e:
        print(f"   ⚠️ sondeo de precedente: {e}")
        return []


# ── EL ESTUDIO DE FONDO AJENO, LIMPIO ─────────────────────────────────────────
# Los trozos se solapan hasta un 30% y arrastran la firma electrónica, el sello
# «PJF - Versión Pública» y el bloque de evidencia criptográfica. Eso no es
# prosa judicial: es basura de PDF, y si entra al prompt el modelo la imita.

_RX_BASURA = re.compile(
    r"(PJF\s*[-–]\s*Versi[óo]n\s*P[úu]blica"
    r"|EVIDENCIA\s+CRIPTOGR[ÁA]FICA"
    r"|FIRMANTE|Nombre:\s*\w+.*?Certificado"
    r"|Cadena\s+de\s+firma|Sello\s+digital"
    r"|[A-Za-z0-9+/]{60,}={0,2})",           # el bloque base64 de la firma
    re.I)

# DE DÓNDE ARRANCA UN TRAMO DE REGLA. La primera versión disparaba con
# cualquier «El artículo N», y el primero de una sentencia está SIEMPRE en el
# considerando de competencia: los artículos 103 y 107 constitucionales, el 34,
# 170 y 181 de la Ley de Amparo. El molde salía siendo la cadena competencial,
# que es lo más rutinario del documento y justo lo que no hace falta enseñar.
#
# Se ata a las FÓRMULAS DE DERIVACIÓN —«De dicho numeral se advierte que…»,
# «tiene por objeto»— que son exactamente el rasgo que midió el análisis: la
# frase que extrae la regla en abstracto después de transcribir la fuente. Donde
# aparece una de éstas hay razonamiento; donde aparece «el artículo 103
# constitucional» sólo hay trámite.
_RX_REGLA = re.compile(
    r"(De\s+dicho\s+numeral|Del\s+precepto\s+(?:transcrito|citado)"
    r"|Del\s+numeral\s+transcrito|De\s+la\s+norma\s+transcrita"
    r"|Del\s+art[íi]culo\s+(?:transcrito|citado)"
    r"|tiene\s+por\s+objeto|tiene\s+como\s+finalidad|la\s+finalidad\s+de\s+(?:la|dicha)\s+norma"
    r"|De\s+lo\s+(?:anterior|transcrito)\s+se\s+(?:advierte|desprende|colige|sigue)"
    r"|por\s+regla\s+general|la\s+regla\s+general\s+es)", re.I)

# Y dónde termina: el primer anclaje al caso ajeno. Ahí se corta, porque los
# hechos de ESE expediente no son los de éste y no deben cruzar.
_RX_CASO = re.compile(
    r"(En\s+el\s+caso\s+concreto|En\s+la\s+especie|En\s+el\s+caso\s+a\s+estudio"
    r"|En\s+el\s+asunto\s+que\s+nos\s+ocupa|el\s+quejoso|la\s+quejosa"
    r"|el\s+recurrente|la\s+recurrente|la\s+responsable\s+resolvi)", re.I)

# Y cuánto se recoge HACIA ATRÁS: la derivación no se entiende sin la
# transcripción que la precede, que es el primer eslabón de la cadena.
ANTES_DE_LA_REGLA = 900

def _limpiar(trozos: list) -> str:
    """Los trozos en orden, sin línea repetida y sin basura de PDF."""
    trozos = sorted(trozos, key=lambda p: int(p.get("chunk_index") or 0))
    lineas, vistas = [], set()
    for p in trozos:
        for ln in str(p.get("chunk_text") or "").splitlines():
            ln = " ".join(ln.split())
            if len(ln) < 12 or _RX_BASURA.search(ln):
                continue
            clave = ln.lower()[:110]
            if clave in vistas:
                continue
            vistas.add(clave)
            lineas.append(ln)
    return "\n".join(lineas)


def _tramo_de_regla(texto: str) -> str:
    """La transcripción, su derivación y el límite. Hasta que empiece el caso ajeno.

    Es molde de FORMA: cómo se transcribe un precepto, cómo se deriva la regla,
    cómo se enuncia su frontera. Nunca fundamento, y nunca hechos de otro: el
    tramo se corta en el primer anclaje al expediente ajeno.

    SE BUSCA A PARTIR DE LA MITAD del documento. El estudio de fondo va después
    de la competencia, la oportunidad y la legitimación, y esos apartados traen
    sus propias fórmulas: «por regla general» aparece en la procedencia tan bien
    como en el fondo. Empezar por el medio no es exacto, pero deja fuera casi
    todo el trámite y no cuesta nada.
    """
    texto = texto or ""
    if len(texto) < 2000:
        return ""
    m = _RX_REGLA.search(texto, len(texto) // 2) or _RX_REGLA.search(texto)
    if not m:
        return ""
    ini = max(0, m.start() - ANTES_DE_LA_REGLA)
    resto = texto[ini:]
    # El corte se busca DESPUÉS de la derivación, no antes: si no, un «el
    # quejoso» que aparezca dentro de la transcripción recogida hacia atrás
    # dejaría el tramo en nada.
    desde = m.start() - ini
    c = _RX_CASO.search(resto, desde)
    tramo = resto[:c.start()] if c else resto[:2600]
    return tramo.strip()[:2600]


# ── EL SONDEO ─────────────────────────────────────────────────────────────────

_RX_CLAVE = re.compile(
    r"\b((?:P|1a|2a|PC)\.?\s*/?\s*J\.?\s*\d+/\d{2,4}"
    r"|[IVXLC]+\.\d*[A-Za-z]?\.\d*[A-Za-z]?\.?\s*J?/?\s*\d+/\d{2,4})", re.I)


def _claves_de(payloads: list) -> list:
    """Las tesis que ellos citaron, por frecuencia.

    Se leen de `tesis_registros` del holding sabiendo que SUBESTIMA: el análisis
    midió 3.14 declaradas contra 6.33 realmente citadas en el texto. Vale como
    punto de partida —dice qué criterios orbitan el tema— pero el conjunto
    autorizado se cierra después, contra el corpus de tesis.
    """
    cuenta = {}
    for p in payloads:
        for t in (p.get("tesis_registros") or p.get("tesis_citadas") or []):
            t = str(t).strip()
            if _RX_CLAVE.match(t) or t.isdigit():
                cuenta[t] = cuenta.get(t, 0) + 1
    return [k for k, _ in sorted(cuenta.items(), key=lambda x: -x[1])][:20]


def _fundamentos(payloads: list) -> list:
    cuenta = {}
    for p in payloads:
        for campo in ("tema_juridico", "fundamento_central"):
            v = str(p.get(campo) or "").strip()
            if len(v) > 6:
                cuenta[v] = cuenta.get(v, 0) + 1
    return [{"fundamento": k, "veces": n}
            for k, n in sorted(cuenta.items(), key=lambda x: -x[1])[:8]]


async def sondear(qdrant, embed, problema: str, materia: str,
                  circuito=None, sentido_propuesto: str = "") -> Sondeo:
    """Cómo resolvió el acervo este mismo problema, antes de fijar el sentido.

    EL VECTOR SE CONSTRUYE DEL PROBLEMA, NO DEL ESCRITO. Es la lección ya medida
    de HyDE: el fraseo de la demanda envenena la búsqueda porque arrastra el
    vocabulario de una de las partes. Aquí entra el enunciado abstracto que la
    fase 3 ya normalizó por concepto.
    """
    s = Sondeo()
    if not (problema or "").strip():
        return s
    mat = _filtro_materia(materia)
    if not mat:
        s.avisos.append(f"Materia «{materia}» sin grafías conocidas: el sondeo "
                        f"de precedente corrió sin filtro de materia.")
    try:
        vector = await embed(problema)
    except Exception as e:
        s.avisos.append(f"No se pudo sondear el acervo de precedentes: {e}")
        return s

    from qdrant_client.models import FieldCondition, Range

    # DOS TIROS. El primero busca sólo entre las buenas —de ahí sale el molde de
    # forma—; el segundo, sin filtro de calidad, para no perder cobertura del
    # tema: la calidad 5 es el 0.06% del acervo y filtrar sólo por ella deja
    # temas enteros sin un solo precedente.
    buenas, todas = await asyncio.gather(
        _buscar(qdrant, vector,
                mat + [FieldCondition(key="calidad_argumentativa_v2",
                                      range=Range(gte=4))], TOP_SONDEO),
        _buscar(qdrant, vector, mat, TOP_SONDEO))

    if not todas and not buenas:
        s.avisos.append("El acervo no devolvió precedentes para este problema.")
        return s

    # LA DISTRIBUCIÓN SE MIDE SOBRE TODAS, no sólo sobre las buenas: la pregunta
    # es cómo se resuelve esto de ordinario, no cómo lo resuelve el 0.06%.
    pl_todas = [pl for _, pl in todas]
    pl_buenas = [pl for _, pl in buenas]

    for p in pl_todas:
        v = _norm_sentido(p.get("sentido"))
        if v:
            s.distribucion[v] = s.distribucion.get(v, 0) + 1

    s.fundamentos = _fundamentos(pl_buenas + pl_todas)
    s.claves = _claves_de(pl_buenas + pl_todas)
    s.concordantes = [{
        "tema": p.get("tema_juridico"), "sentido": p.get("sentido"),
        "holding": str(p.get("holding") or "")[:600],
        "tribunal": p.get("tribunal_completo"), "circuito": p.get("circuito"),
        "calidad": p.get("calidad_argumentativa_v2"),
        "pdf_url": p.get("pdf_url"),
    } for p in (pl_buenas or pl_todas)[:5]]

    s.moldes = await _moldes(qdrant, buenas, s)
    s.razonados = [{"sentido": _norm_sentido(p.get("sentido")),
                    "razon": str(p.get("holding") or p.get("resumen") or "")[:900],
                    "tribunal": p.get("tribunal_completo"),
                    "tema": p.get("tema_juridico")}
                   for p in pl_todas
                   if len(str(p.get("holding") or p.get("resumen") or "")) >= 120]
    if sentido_propuesto:
        s.objeciones = s.objeciones_contra(sentido_propuesto)
    return s


async def _moldes(qdrant, buenos: list, s: Sondeo) -> list:
    """Tramos de REGLA de estudios bien escritos sobre este mismo problema.

    Se lee el estudio de fondo, no el resumen, porque toda la señal de calidad
    está ahí: el análisis midió que los metadatos del holding son casi idénticos
    entre una sentencia de calidad 5 y una de 3.

    Es molde de FORMA. Se entrega diciendo que los hechos de esos asuntos NO son
    los de este expediente, y el tramo se corta en el primer anclaje al caso
    ajeno para que no haya manera de que crucen.
    """
    # Sólo los circuitos cuyo estudio está en Qdrant. Para el 6 y el 16 hay
    # resumen pero no texto: pedirlo devuelve vacío y parecería que no hay
    # precedentes buenos, cuando lo que no hay es de dónde leerlos.
    con_texto = [(pid, pl) for pid, pl in buenos
                 if str(pl.get("circuito") or "").strip() in
                 {str(c) for c in CIRCUITOS_CON_ESTUDIO}][:CANDIDATOS_MOLDE]
    if not con_texto:
        if buenos:
            s.avisos.append(
                "Hay precedentes bien argumentados sobre el tema, pero de "
                "circuitos cuyo estudio de fondo no está en el acervo (sólo "
                "1, 2, 3, 4 y 22 lo tienen). El redactor trabaja sin molde.")
        return []

    estudios = await asyncio.gather(*[
        _hermanos(qdrant, pl.get("circuito"), pid) for pid, pl in con_texto])

    fuera = []
    for (pid, pl), trozos in zip(con_texto, estudios):
        if not trozos:
            continue
        tramo = _tramo_de_regla(_limpiar(trozos))
        if len(tramo) < 300:
            continue
        fuera.append({
            "tramo": tramo,
            "tribunal": pl.get("tribunal_completo") or pl.get("tribunal"),
            "expediente": pl.get("expediente"),
            "calidad": pl.get("calidad_argumentativa_v2"),
            "pdf_url": pl.get("pdf_url"),
        })
        if len(fuera) >= MAX_MOLDES:
            break
    return fuera


async def _hermanos(qdrant, circuito, holding_id) -> list:
    """Los demás trozos de la misma sentencia. `holding_id` sí está indexado aquí."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    try:
        c = int(str(circuito).strip())
    except Exception:
        return []
    if c not in CIRCUITOS_CON_ESTUDIO or not holding_id:
        return []
    try:
        r = await _esperar(qdrant.scroll(
            collection_name=f"sentencias_ef_c{c}",
            scroll_filter=Filter(must=[FieldCondition(
                key="holding_id", match=MatchValue(value=str(holding_id)))]),
            limit=80, with_payload=True))
        pts = r[0] if isinstance(r, tuple) else r
        return [p.payload for p in (pts or [])]
    except Exception:
        return []


def _objeciones_de(razonados: list, sentido_propuesto: str) -> list:
    """Cómo razonó el que resolvió al revés.

    ES EL PASO QUE SEPARA ARRIBA DE ABAJO y el que no existía en ningún sitio:
    93% de las laborales de calidad máxima responden el mejor argumento del que
    pierde, contra el 58% de las medias. Y ese argumento no hay que inventarlo
    —inventarlo es construir un espantapájaros y tumbarlo—: está escrito en las
    sentencias del acervo que resolvieron lo mismo en sentido contrario.
    """
    mio = _norm_sentido(sentido_propuesto)
    if not mio:
        return []
    # «Concede» y «parcialmente concede» no son contrarios: quien concede en
    # parte le da la razón al quejoso en lo que aquí importa. Tomarlo por
    # objeción daría un contraargumento que no combate nada.
    afines = {"concede", "parcialmente_concede", "ampara"}
    contrario = (lambda x: x not in afines) if mio in afines else (lambda x: x in afines)
    fuera = []
    for p in razonados:
        if p.get("sentido") and contrario(p["sentido"]):
            fuera.append(p)
        if len(fuera) >= MAX_OBJECION:
            break
    return fuera


# ── LO QUE VE EL REDACTOR ─────────────────────────────────────────────────────

def bloque(s: Sondeo, sentido_propuesto: str = "") -> str:
    """El sondeo, escrito para que entre al prompt del estudio.

    Va SEPARADO del material que funda. Un precedente de otro colegiado no es
    fuente que obligue —salvo contradicción resuelta— y confundir las dos cosas
    es exactamente el error que este proyecto existe para evitar. Aquí se dice
    para qué sirve cada pieza: la distribución, para saber si uno se aparta de
    la corriente; el molde, para la forma; la objeción, para responderla.
    """
    if not s or (not s.distribucion and not s.moldes and not s.razonados):
        return ""
    L = ["", "═" * 71, "CÓMO RESOLVIERON OTROS TRIBUNALES ESTE MISMO PROBLEMA",
         "═" * 71,
         "Esto NO es fuente que funde. Es el acervo de sentencias de colegiados",
         "de circuito, y sirve para tres cosas concretas:", ""]

    if s.distribucion:
        total = sum(s.distribucion.values())
        L.append(f"1. EL SENTIDO DE ORDINARIO (sobre {total} sentencias del tema):")
        for k, n in sorted(s.distribucion.items(), key=lambda x: -x[1])[:6]:
            L.append(f"   · {k}: {n}  ({100*n//max(total,1)}%)")
        if sentido_propuesto:
            va, cuanto = s.concuerda(sentido_propuesto)
            if not va:
                L += ["",
                      f"   ATENCIÓN: el sentido que vas a proponer «{sentido_propuesto}»",
                      f"   sólo aparece en el {cuanto:.0%} de los precedentes del tema.",
                      "   Apartarse del criterio mayoritario es legítimo —para eso existe",
                      "   la contradicción de tesis— pero ESCRÍBELO: di que la corriente",
                      "   es otra y explica por qué este caso no cae en ella. Lo que no",
                      "   puedes es apartarte sin enterarte."]
        L.append("")

    if s.fundamentos:
        L.append("2. LOS FUNDAMENTOS QUE SE REPITEN en las sentencias del tema:")
        for f in s.fundamentos[:6]:
            L.append(f"   · {f['fundamento']}  ({f['veces']})")
        L.append("")

    if s.moldes:
        L += ["3. MOLDE DE FORMA — así construye la regla un tribunal que escribe bien.",
              "   COPIA LA FORMA, NUNCA EL CONTENIDO: los hechos de estos asuntos NO",
              "   son los de tu expediente y no puedes traerlos. Fíjate en cómo",
              "   transcriben el precepto, cómo derivan la regla en abstracto y cómo",
              "   enuncian su límite:", ""]
        for m in s.moldes[:3]:
            L.append(f"   ── {m.get('tribunal') or 'colegiado'} "
                     f"(calidad {m.get('calidad')}) ──")
            for ln in str(m["tramo"])[:1400].splitlines():
                L.append(f"   {ln}")
            L.append("")

    objeciones = s.objeciones or (s.objeciones_contra(sentido_propuesto)
                                  if sentido_propuesto else [])
    if objeciones:
        L += ["4. LA MEJOR OBJECIÓN — otros tribunales resolvieron esto AL REVÉS,",
              "   y así lo razonaron. Tienes que responder este argumento con razón",
              "   propia, no ignorarlo ni sustituirlo por uno más fácil de tumbar:", ""]
        for o in objeciones[:3]:
            L.append(f"   · [{o['sentido']}] {o['razon'][:500]}")
        L.append("")
    return "\n".join(L)


# ── DE DÓNDE SALEN LA MATERIA Y EL CIRCUITO ───────────────────────────────────
# No se preguntan. Vienen escritos en el encabezado del asunto —«AMPARO DIRECTO
# ADMINISTRATIVO: 512/2026»— y en el nombre del tribunal —«Tercer Tribunal
# Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito»—.
# Es la misma lección que la autoridad responsable: si el dato está en el
# expediente, pedírselo al secretario es hacerle teclear lo que ya nos dijo.

_MATERIA_EN_TEXTO = [
    ("laboral", r"laboral|trabajo|junta\s+(?:especial|federal|local)"),
    ("penal", r"\bpenal\b|ejecuci[óo]n\s+de\s+penas"),
    ("administrativa", r"administrativ|fiscal|tribunal\s+de\s+justicia\s+administrativa"),
    ("civil", r"\bcivil\b|familiar|mercantil"),
]

_ORDINALES = {
    "primer": 1, "segundo": 2, "tercer": 3, "cuarto": 4, "quinto": 5,
    "sexto": 6, "séptimo": 7, "septimo": 7, "octavo": 8, "noveno": 9,
    "décimo": 10, "decimo": 10, "undécimo": 11, "undecimo": 11,
    "duodécimo": 12, "duodecimo": 12, "vigésimo": 20, "vigesimo": 20,
    "trigésimo": 30, "trigesimo": 30,
}


def materia_de(encabezado: str = "", tribunal: str = "",
               tipo_asunto: str = "") -> str:
    """La materia del asunto, leída de lo que ya está escrito.

    Se mira primero el NOMBRE DEL TRIBUNAL, que es el dato duro —un colegiado
    en materia de trabajo no ve otra cosa—, y sólo después el encabezado. Al
    revés fallaba: «AMPARO DIRECTO CIVIL» tramitado ante un tribunal mixto en
    materias administrativa y civil daba las dos, y ganaba la primera de la
    lista en vez de la del asunto.
    """
    for fuente in (tribunal, encabezado, tipo_asunto):
        texto = str(fuente or "")
        if not texto.strip():
            continue
        hallados = [m for m, rx in _MATERIA_EN_TEXTO
                    if re.search(rx, texto, re.I)]
        # Un tribunal MIXTO nombra dos materias; ahí el nombre no decide y hay
        # que seguir mirando el encabezado, que sí dice de qué es este asunto.
        if len(hallados) == 1:
            return hallados[0]
    for fuente in (encabezado, tipo_asunto, tribunal):
        for m, rx in _MATERIA_EN_TEXTO:
            if re.search(rx, str(fuente or ""), re.I):
                return m
    return ""


def circuito_de(tribunal: str) -> str:
    """El número de circuito, del nombre del tribunal. Como TEXTO.

    En Qdrant `circuito` está indexado como cadena: pasarlo como número devuelve
    un 400 que, si nadie lo mira, se lee como «no hay precedentes».
    """
    m = re.search(r"del\s+((?:[A-Za-zÁÉÍÓÚáéíóú]+\s+){1,2}?)circuito",
                  str(tribunal or ""), re.I)
    if not m:
        return ""
    palabras = [p.lower() for p in m.group(1).split() if p.strip()]
    total = 0
    for p in palabras:
        total += _ORDINALES.get(p.rstrip("aos"), 0) or _ORDINALES.get(p, 0)
    return str(total) if total else ""


# ═══════════════════════════════════════════════════════════════════════════
# LA PREDICCIÓN, POR PROBLEMA
# ═══════════════════════════════════════════════════════════════════════════
# David: «además de los holdings hemos desperdiciado una herramienta que es
# parte de Iurexia: Jurimetría. Esta herramienta da la predicción del resultado
# del juicio en función de su contexto, permitiéndole calibrar el proyecto con
# un clic».
#
# Y la desperdiciábamos dos veces. El endpoint /api/jurimetria existe desde
# hace meses y el taller no lo llamaba nunca; y el sondeo de precedentes, que
# calcula exactamente lo mismo —la distribución de sentidos del acervo sobre
# este tema—, corría SÓLO SOBRE EL PRIMER PROBLEMA y su resultado se usaba de
# material de lectura, no de predicción.
#
# NO HACE FALTA UNA SEGUNDA BÚSQUEDA. La jurimetría del endpoint agrega
# `sentencias_holdings` por sentido; el sondeo consulta esa misma colección con
# el vector del problema, que además está mejor apuntado —el endpoint busca por
# la descripción del asunto entero—. Sondeando cada problema se obtiene la
# predicción de cada uno con una llamada que ya se estaba haciendo para el
# primero, y las N corren en paralelo.
#
# LO QUE ESTO NO ES: no es un pronóstico de lo que ESTE tribunal hará. Es la
# distribución de lo que hicieron otros sobre el mismo tema, y se presenta así.
# Un secretario que lea «82%» y crea que es la probabilidad de su asunto está
# leyendo mal, y por eso el texto dice cuántas sentencias hay detrás.
# LAS DOS HERRAMIENTAS DECÍAN LO CONTRARIO SOBRE LA MISMA BÚSQUEDA, y ésta es
# justamente la cifra que va a la pantalla al lado de cada problema.
#
# La jurimetría del endpoint agrupa los sentidos en tres cubos —«confirma» y
# «niega» e «infundado» son lo mismo desde el punto de vista de quien promueve—
# y esta función tomaba el MÁXIMO sobre las etiquetas crudas. Con una
# distribución {concede: 30, niega: 20, confirma: 15} el taller decía
# «Conceder (46%)» y la jurimetría, «NIEGA 54%». Sentidos opuestos, la misma
# consulta a la misma colección.
#
# Se agrupa igual que el endpoint, que es el mapa correcto: lo que importa no
# es la palabra que usó cada tribunal en su resolutivo sino si le dio o no la
# razón a quien promovía.
_CUBO = {
    "concede": "CONCEDE", "parcialmente_concede": "CONCEDE",
    "fundado": "CONCEDE", "revoca": "CONCEDE",
    "niega": "NIEGA", "infundado": "NIEGA", "confirma": "NIEGA",
    "inoperante": "NIEGA", "ineficaz": "NIEGA",
    "sobresee": "SOBRESEE", "desecha": "SOBRESEE", "sin_materia": "SOBRESEE",
    "incompetencia": "SOBRESEE", "modifica": "SOBRESEE",
}
_NOMBRE_CUBO = {"CONCEDE": "Conceder", "NIEGA": "Negar",
                "SOBRESEE": "Sobreseer"}


def prediccion(s: "Sondeo") -> dict:
    """{'sentido', 'porcentaje', 'n', 'confianza', 'frase'} o {} si no alcanza."""
    if s is None:
        return {}
    from collections import Counter
    cubos: Counter = Counter()
    for k, v in (s.distribucion or {}).items():
        cubos[_CUBO.get(str(k).strip().lower(), "OTRO")] += v
    cubos.pop("OTRO", None)
    total = sum(cubos.values())
    if not total:
        return {}
    sentido, cuantas = max(cubos.items(), key=lambda kv: kv[1])
    pct = cuantas / total
    # LA CONFIANZA ES DEL TAMAÑO DE LA BASE, no del porcentaje: un 100% sobre
    # tres sentencias no dice nada y un 60% sobre ochenta dice mucho.
    #
    # Y LOS UMBRALES SON LOS DE ESTA BÚSQUEDA, no los del endpoint. Allí se
    # piden 80 resultados y «alta» exige 40; aquí se piden 40, así que exigir
    # 40 con sentido significaba que TODOS lo trajeran: «alta» no salía nunca
    # y la pantalla enseñaba «baja» en una base perfectamente sólida. Se
    # calibran a la mitad, que es la misma proporción.
    confianza = "alta" if total >= 20 else "media" if total >= 8 else "baja"
    return {
        "sentido": sentido,
        "porcentaje": round(pct, 3),
        "n": total,
        "confianza": confianza,
        "frase": (f"{_NOMBRE_CUBO.get(sentido, sentido)} "
                  f"({round(pct * 100)}% de {total} sentencias del acervo"
                  f"{'' if confianza != 'baja' else ', base corta'})"),
        "distribucion": dict(cubos),
        "detalle": dict(s.distribucion),
    }


# Cómo se dice cada sentido cuando se le enseña al secretario. El acervo los
# guarda como los guarda; la pantalla no es sitio para su jerga.
_ETIQUETA = {
    "concede": "Conceder", "niega": "Negar", "sobresee": "Sobreseer",
    "confirma": "Confirmar", "revoca": "Revocar", "modifica": "Modificar",
    "fundado": "Fundado", "infundado": "Infundado", "inoperante": "Inoperante",
    "desecha": "Desechar",
}
