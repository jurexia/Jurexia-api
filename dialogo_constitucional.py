"""EL DIÁLOGO CONSTITUCIONAL: el método de interpretación, en la razón y en el estudio.

═══ POR QUÉ (24-sep-2026) ═════════════════════════════════════════════════
David: «en muchos razonamientos puede partirse de una interpretación más
favorable a la luz del principio pro persona, o de la interpretación conforme
—cuando se decida por fundado y ello requiera de una interpretación favorable
de la ley— en favor del reconocimiento de un derecho, del acceso a la justicia,
a un recurso… ¿cómo ampliamos esta calidad de diálogo constitucional al
interpretar las leyes locales?».

Medido sobre los 151 asuntos del taller antes de tocar nada:

  · LA RAZÓN SE DECIDÍA SIN CONSTITUCIÓN. El marco jurídico —artículos
    constitucionales, tratados, Corte Interamericana, precepto local— sólo se
    armaba al redactar el proyecto; la propuesta y la razón del sentido nunca
    lo veían. El diálogo entraba como capa sobre una decisión ya razonada sin
    él. En sus prompts no aparecía ni una vez «pro persona», «interpretación
    conforme» ni «acceso a la justicia».
  · EL 17 NO ENTRABA POR LA PUERTA. 104 de 151 asuntos giran sobre una puerta
    procesal —desechamiento, extemporaneidad, legitimación, sobreseimiento,
    prevención— y en 84 (81%) el marco no traía el artículo 17: se encendía
    con «acceso a la justicia», palabras que un problema redactado como
    «extemporaneidad» no usa.
  · EL MÉTODO ESTABA EN EL ACERVO Y NADIE LO TRAÍA. Las tesis de abajo se
    comprobaron una por una contra `jurisprudencia_nacional_v3` el 24-sep-2026.
    Dos de Radilla (160525, 160589) están en el acervo SIN clave: buscadas por
    clave no aparecen. Por eso aquí todo va por REGISTRO.

═══ SÓLO A FAVOR DE LA PERSONA ════════════════════════════════════════════
La primera versión recorría la escalera «en las dos direcciones», y el 711/2025
—revisión de la UIF contra el amparo concedido a una sociedad cuyas cuentas se
bloquearon— salió diciendo que el artículo 115 de la Ley de Instituciones de
Crédito se interpretaba «de manera compatible con el artículo 16» PARA SOSTENER
EL BLOQUEO. David lo vio en el acto: «ahí no aplicaría, ya que se está dando la
razón a la autoridad y se valida la restricción de un derecho… sólo operan ese
tipo de interpretaciones en favor de la persona y en supuestos de alternativas
para mayor acceso. Si cambio de alternativa, señalar cuándo sería posible una
interpretación conforme o pro persona».

Así que la dirección se CALCULA (`favorece_a_la_persona`): quién combate —la
persona o una autoridad— y si la calificación prospera. En la vía que valida
una restricción el pro persona y la interpretación conforme NO se invocan; en
la que reconoce el derecho o da mayor acceso, la razón se construye desde la
lectura protectora, con los límites que la propia Corte le puso —no exime de
los requisitos de procedencia, no obliga a resolver como pide la parte, cede
ante una restricción constitucional expresa—. Y la propuesta dice, en
`via_protectora`, cuál es la vía que favorece a la persona y si en ella cabe
esa lectura, para que el secretario lo vea al cambiar de alternativa.

Y la arquitectura de la materia administrativa ya lo había medido por su lado:
invocar la Constitución «para subir de nivel» es MÁS frecuente en los proyectos
medios que en los buenos. Lo que distingue es usarla para ELEGIR la lectura, no
para adornarla. Por eso la escalera se recorre sólo cuando la calificación
depende del sentido de una norma, y si no, se calla.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

# ═══ EL CANON DEL MÉTODO, POR REGISTRO ═════════════════════════════════════
# (registro, para qué sirve). Pocas y exactas: cada una responde a un peldaño.
CANON_BASE: Tuple[Tuple[str, str], ...] = (
    ("160525", "los pasos del control ex officio: interpretación conforme en "
               "sentido amplio, en sentido estricto y, sólo al final, inaplicación"),
    ("2014332", "qué es la interpretación conforme y cómo se ordena con el "
                "principio pro persona"),
    ("2002000", "el pro persona como criterio para elegir la norma o la lectura "
                "más protectora"),
    ("2006224", "el límite: la restricción constitucional expresa prevalece sobre "
                "la lectura más favorable"),
    ("2004748", "el límite: del pro persona no se sigue resolver como pide la parte"),
)

# Cuando el asunto tiene PUERTA PROCESAL: el acceso a la justicia se lee con su
# razón de ser, y con sus límites al lado.
CANON_PUERTA: Tuple[Tuple[str, str], ...] = (
    ("2007064", "los requisitos procesales se leen por su finalidad, para evitar "
                "formalismos que impidan resolver el fondo"),
    ("2007621", "el acceso a la justicia frente a los presupuestos procesales"),
    ("2005717", "el límite: el pro persona no exime de los requisitos de "
                "procedencia de un medio de defensa"),
    ("2005917", "el límite: un requisito formal razonable no viola el derecho a un "
                "recurso efectivo"),
)

REGISTROS_METODO = frozenset(r for r, _ in CANON_BASE + CANON_PUERTA)


# ═══ LA PUERTA PROCESAL ════════════════════════════════════════════════════
# La clase ya la trae cada problema desde que se formula (`fases123_pipeline`):
# «procesal» —una actuación del procedimiento— y «procedencia» —causa de
# improcedencia o sobreseimiento—. No hace falta otra llamada. Las raíces son el
# respaldo para los problemas guardados antes de que existiera la clase.
_RX_PUERTA = re.compile(
    r"desech|improceden|extempor|oportunidad|sobresei|legitimaci|personer|"
    r"prevenci|preclu|caducidad|admisi|requisitos?\s+de\s+procedencia|"
    r"plazo\s+para\s+(?:promover|interponer|presentar)", re.I)


def hay_puerta(problemas: Iterable) -> bool:
    """¿Algún problema del asunto se decide en la puerta del proceso?"""
    for p in problemas or []:
        if isinstance(p, dict):
            if str(p.get("clase") or "").lower() in ("procesal", "procedencia"):
                return True
            texto = " ".join(str(p.get(k) or "") for k in ("pregunta", "resolvio", "combate"))
        else:
            texto = str(p or "")
        if _RX_PUERTA.search(texto):
            return True
    return False


# ═══ A FAVOR DE QUIÉN ══════════════════════════════════════════════════════
PROSPERA = frozenset({"fundado", "esencialmente_fundado", "sustancialmente_fundado",
                      "parcialmente_fundado"})
NO_PROSPERA = frozenset({"infundado", "inoperante", "ineficaz", "inatendible",
                         "fundado_insuficiente", "fundado_pero_insuficiente"})


def _sentido(s) -> str:
    return "_".join(str(s or "").strip().lower().replace("-", " ").split())


def impugna_la_autoridad(tipo_asunto: str = "", recurrente: str = "",
                         es_recurso: bool = False) -> Optional[bool]:
    """¿Quien combate en ESTE asunto es una autoridad? None si no consta.

    En el amparo directo combate el quejoso. En la revisión fiscal, siempre la
    autoridad. En la revisión y en la queja depende de quién recurrió, y eso se
    lee del nombre con el mismo detector que ya separa los papeles del asunto
    (`redactor_adelanto._parece_autoridad`).
    """
    t = (tipo_asunto or "").strip().lower()
    if t == "revision_fiscal":
        return True
    if t in ("amparo_revision", "queja") or es_recurso:
        if not (recurrente or "").strip():
            return None
        from redactor_adelanto import _parece_autoridad
        return bool(_parece_autoridad(recurrente))
    return False


def favorece_a_la_persona(sentido: str, tipo_asunto: str = "", recurrente: str = "",
                          es_recurso: bool = False) -> Optional[bool]:
    """¿Esta calificación le da la razón a quien reclama el derecho?

    True: le reconoce el derecho o le da mayor acceso —ahí caben el pro persona y
    la interpretación conforme—. False: valida una restricción o le niega lo que
    pide —ahí no se invocan—. None: no se puede saber, y el prompt lo pregunta.
    """
    s = _sentido(sentido)
    if s in PROSPERA:
        prospera = True
    elif s in NO_PROSPERA:
        prospera = False
    else:
        return None
    autoridad = impugna_la_autoridad(tipo_asunto, recurrente, es_recurso)
    if autoridad is None:
        return None
    return prospera != autoridad


def quien_combate(tipo_asunto: str = "", recurrente: str = "", es_recurso: bool = False) -> str:
    """Una línea para la propuesta: cuál de las dos vías favorece a la persona."""
    a = impugna_la_autoridad(tipo_asunto, recurrente, es_recurso)
    if a is True:
        quien = f" ({' '.join(recurrente.split())[:90]})" if (recurrente or "").strip() else ""
        return ("QUIEN COMBATE AQUÍ ES UNA AUTORIDAD" + quien + ": la vía que favorece a quien "
                "reclama el derecho es la que NO prospera —infundado, inoperante—.")
    if a is False:
        return ("QUIEN COMBATE AQUÍ ES QUIEN RECLAMA EL DERECHO: la vía que lo favorece es la "
                "que prospera —fundado—.")
    return ("NO CONSTA si quien combate es una autoridad o un particular: decide tú cuál de "
            "las dos vías favorece a quien reclama el derecho.")


# ═══ TRAERLO AL MATERIAL ═══════════════════════════════════════════════════
def _para_que(registro: str) -> str:
    for r, que in CANON_BASE + CANON_PUERTA:
        if r == registro:
            return que
    return ""


async def canon(qdrant, puerta: bool) -> List[dict]:
    """Las tesis del método, tal como están en el acervo, marcadas."""
    import fase6_rag as _rag
    regs = [r for r, _ in CANON_BASE] + ([r for r, _ in CANON_PUERTA] if puerta else [])
    tesis = await _rag.tesis_por_registro(qdrant, regs)
    orden = {r: i for i, r in enumerate(regs)}
    fuera = []
    for t in sorted(tesis, key=lambda t: orden.get(str(t.get("registro")), 99)):
        t = dict(t)
        t["metodo"] = True
        # EXENTAS DEL RECORTE, como las de la técnica: responden a otra
        # pregunta —cómo se interpreta— y no compiten con las del caso.
        t["tecnica"] = True
        t["metodo_para"] = _para_que(str(t.get("registro")))
        fuera.append(t)
    return fuera


async def inyectar(qdrant, material, problemas) -> int:
    """Pone el método en el material. Devuelve cuántas tesis añadió.

    VAN DELANTE, no al final: la sesión se guarda con un tope de ochenta tesis y
    lo que va al final es lo primero que se pierde. Delante no estorban: las de
    la técnica se separan antes del corte y la propuesta las lee aparte.
    """
    if material is None or qdrant is None:
        return 0
    puerta = hay_puerta(problemas)
    ya = {str(t.get("registro")) for t in (getattr(material, "tesis", None) or [])
          if t.get("metodo")}
    necesarias = {r for r, _ in CANON_BASE} | ({r for r, _ in CANON_PUERTA} if puerta else set())
    if necesarias <= ya:
        return 0
    try:
        nuevas = await canon(qdrant, puerta)
    except Exception as e:
        print(f"   ⚠️ no se pudo traer el método de interpretación: {e}")
        return 0
    nuevas = [t for t in nuevas if str(t.get("registro")) not in ya]
    if not nuevas:
        return 0
    # Si alguna ya estaba como tesis del caso, se queda UNA vez: la del método.
    regs = {str(t.get("registro")) for t in nuevas}
    resto = [t for t in (material.tesis or []) if str(t.get("registro")) not in regs]
    material.tesis = nuevas + resto
    print(f"   ⚖️ método de interpretación al material: {len(nuevas)} tesis"
          f"{' (con puerta procesal)' if puerta else ''}")
    return len(nuevas)


def tesis_de_metodo(material) -> List[dict]:
    return [t for t in (getattr(material, "tesis", None) or []) if t.get("metodo")]


def con_puerta(material) -> bool:
    return any(str(t.get("registro")) in {r for r, _ in CANON_PUERTA}
               for t in tesis_de_metodo(material))


# ═══ LA ESCALERA ═══════════════════════════════════════════════════════════
def _escalera(puerta: bool) -> str:
    s = """1. EL DERECHO EN JUEGO y su fuente: el artículo constitucional y, si lo hay,
   el del tratado —van en el parámetro de este asunto—.
2. LAS LECTURAS POSIBLES del precepto que se discute —dos, a veces tres— y en
   qué se apoya cada una: su letra, su lugar en el sistema, su finalidad.
3. INTERPRETACIÓN CONFORME: de las lecturas válidas se elige la que mejor
   protege el derecho (registros 160525 y 2014332); el pro persona es el
   criterio para elegir entre normas o entre lecturas (registro 2002000).
4. LOS LÍMITES, dichos: una restricción constitucional expresa prevalece
   (registro 2006224) y del pro persona no se sigue resolver como pide la parte
   (registro 2004748).
5. INAPLICAR es el último peldaño: sólo si ninguna lectura es compatible, y se
   advierte a quien firma."""
    if puerta:
        s += """

PUERTA PROCESAL. En los problemas de procedencia o de procedimiento el
parámetro es el artículo 17 constitucional —incluido su tercer párrafo:
privilegiar la solución del conflicto sobre los formalismos, siempre que no se
afecte la igualdad de las partes ni el debido proceso— y los artículos 8.1 y 25
de la Convención Americana. Un requisito se lee por su razón de ser: si su
finalidad se cumplió, exigir la forma es formalismo (registros 2007064 y
2007621). Pero un requisito razonable no viola el acceso a la justicia
(registro 2005917) y el pro persona no lo suprime (registro 2005717)."""
    return s


def _lista(material) -> str:
    fuera = []
    for t in tesis_de_metodo(material):
        fuera.append(f"  [registro {t.get('registro','')}] {t.get('instancia','')} — "
                     f"sirve para: {t.get('metodo_para','')}\n"
                     f"    {t.get('rubro','')}\n"
                     f"    {(t.get('texto') or '')[:700]}")
    return "\n".join(fuera)


_NO_ENTRA = """
═══════════════════════════════════════════════════════════════════════
SIN PRO PERSONA NI INTERPRETACIÓN CONFORME EN ESTA CALIFICACIÓN
═══════════════════════════════════════════════════════════════════════
Esta calificación NO favorece a quien reclama el derecho: valida una
restricción o le niega lo que pide. El principio pro persona y la
interpretación conforme operan SÓLO a favor de la persona —para reconocerle un
derecho o darle mayor acceso—, así que NO los invoques para sostenerla: no la
fortalecen, la hacen vulnerable. Sostén la calificación con su fundamento
legal y con la jurisprudencia que la valida. Si la parte planteó esos
principios, contéstalos en lo que alegó, sin hacerlos tuyos.
"""


def bloque_metodo(material, modo: str = "razon", favorece: Optional[bool] = None,
                  quien: str = "") -> str:
    """Para la propuesta y la razón. Vacío si el método no está en el material.

    modo «razon»: la calificación ya la decidió quien firma. Si NO favorece a
    la persona, el bloque dice que no se invocan (sin criterios del método); si
    la favorece, la razón se construye desde la lectura protectora; si no se
    sabe, el modelo lo resuelve con la regla por delante.
    modo «propuesta»: el motor aún propone; la lectura protectora sólo cabe en
    la vía que favorece a la persona, y la señala en `via_protectora`.
    """
    if not tesis_de_metodo(material):
        return ""
    if modo != "propuesta" and favorece is False:
        return _NO_ENTRA
    puerta = con_puerta(material)
    if modo == "propuesta":
        uso = f"""SÓLO A FAVOR DE LA PERSONA. El pro persona y la interpretación conforme operan
únicamente en la vía que le reconoce un derecho o le da mayor acceso; en la
que valida una restricción NO se invocan. {quien}
- En la razón de la vía que favorece a la persona —sea tu propuesta o tu
  alternativa—, si la calificación depende de cómo se lee un precepto,
  construye desde la lectura protectora.
- En la vía contraria no la menciones.
- Y di en `via_protectora` si esa lectura existe en ESTE asunto: qué precepto,
  qué lectura y con qué apoyo, y qué la haría inviable. Si ningún precepto
  admite una lectura más favorable, dilo así. No la fuerces."""
    elif favorece is True:
        uso = """ESTA CALIFICACIÓN FAVORECE A QUIEN RECLAMA EL DERECHO. Si depende de cómo se
lee un precepto, la razón se construye DESDE la lectura protectora: es su
premisa, no un adorno al final. Si ningún precepto admite una lectura más
favorable, no la fuerces: un pro persona sin lecturas en disputa es retórica.
Cita estos registros sólo en el peldaño donde deciden algo."""
    else:
        uso = """SÓLO SI LA CALIFICACIÓN DE ARRIBA FAVORECE A QUIEN RECLAMA EL DERECHO —le
reconoce el derecho o le da mayor acceso— recorre la escalera y construye desde
la lectura protectora. Si valida una restricción o le niega lo que pide, NO
invoques el pro persona ni la interpretación conforme: operan sólo a favor de
la persona. Cita estos registros sólo en el peldaño donde deciden algo."""
    return f"""
═══════════════════════════════════════════════════════════════════════
EL DIÁLOGO CONSTITUCIONAL — cuando la calificación depende de cómo se lee una norma
═══════════════════════════════════════════════════════════════════════
Si lo que decide es el sentido de un precepto, el razonamiento recorre esta
escalera, en este orden y en pocas líneas:

{_escalera(puerta)}

{uso}

CRITERIOS DEL MÉTODO — dicen CÓMO interpretar; los del caso dicen QUÉ se resuelve:
{_lista(material)}
"""


def cierre_estudio(material, favorece: Optional[bool] = None) -> str:
    """Lo último que lee el redactor antes de escribir, si el método está."""
    if not tesis_de_metodo(material):
        return ""
    if favorece is False:
        return """
Y SIN PRO PERSONA NI INTERPRETACIÓN CONFORME: esta resolución no favorece a
quien reclama el derecho —valida una restricción o le niega lo que pide— y esos
principios operan sólo a su favor. No los invoques para sostenerla. Si la parte
los planteó, se contestan en lo que alegó, sin hacerlos propios.
"""
    puerta = con_puerta(material)
    _p = ("\n- EN LA PUERTA PROCESAL el peldaño se escribe con el artículo 17 —y su tercer\n"
          "  párrafo— y con los artículos 8.1 y 25 de la Convención Americana: el\n"
          "  requisito se lee por su finalidad, y si se exige, se dice qué finalidad\n"
          "  protege." if puerta else "")
    _cuando = ("La resolución favorece a quien reclama el derecho: si la calificación\n"
               "depende de cómo se lee una norma —sobre todo una ley local o un requisito\n"
               "procesal—, el estudio se construye DESDE la lectura protectora."
               if favorece is True else
               "SÓLO si la resolución favorece a quien reclama el derecho —le reconoce el\n"
               "derecho o le da mayor acceso—. Si valida una restricción, este párrafo no\n"
               "se escribe: el pro persona y la interpretación conforme operan sólo a favor\n"
               "de la persona.")
    return f"""
Y EL DIÁLOGO CONSTITUCIONAL. {_cuando} Va DENTRO del problema que decide,
después del marco y antes del caso concreto, en dos a cuatro párrafos: el
derecho en juego y su fuente; las lecturas posibles del precepto y en qué se
apoya cada una; cuál es la conforme y por qué; y los límites que la Corte fijó.{_p}
- Los CRITERIOS DE MÉTODO del material se citan con su registro sólo en ese
  peldaño, y NO cuentan entre los tres a seis criterios del caso: dicen cómo
  interpretar, no qué se resuelve.
- Si ninguna calificación depende del sentido de una norma, este párrafo no se
  escribe. Un pro persona sin lecturas en disputa es retórica.
"""
