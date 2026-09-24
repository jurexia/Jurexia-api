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

═══ LO QUE NO ES ══════════════════════════════════════════════════════════
No es un sesgo a favor de quien pide. La propia Corte le puso límites al pro
persona —no exime de los requisitos de procedencia, no obliga a resolver como
pide la parte, cede ante una restricción constitucional expresa— y un proyecto
que lo usa como llave maestra es el que se revoca. El método se recorre en LAS
DOS DIRECCIONES: construye desde la lectura protectora cuando la calificación
la adopta, y explica por qué no está disponible cuando la rechaza. Ese diálogo
—no la palabra «convencionalidad»— es lo que hace resistente un proyecto.

Y la arquitectura de la materia administrativa ya lo había medido por su lado:
invocar la Constitución «para subir de nivel» es MÁS frecuente en los proyectos
medios que en los buenos. Lo que distingue es usarla para ELEGIR la lectura, no
para adornarla. Por eso la escalera se recorre sólo cuando la calificación
depende del sentido de una norma, y si no, se calla.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Tuple

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


def bloque_metodo(material, modo: str = "razon") -> str:
    """Para la propuesta y la razón. Vacío si el método no está en el material.

    modo «razon»: la calificación ya la decidió quien firma y la escalera se
    pone a su servicio. Modo «propuesta»: el motor aún propone, y la lectura
    protectora es una de las que tiene que sopesar.
    """
    if not tesis_de_metodo(material):
        return ""
    puerta = con_puerta(material)
    if modo == "propuesta":
        uso = """CUÁNDO: si la calificación de un problema depende de CÓMO se lee una norma
—sobre todo una ley local o un requisito procesal—, antes de proponer sopesa la
lectura más protectora del derecho en juego. Si la propones, dilo en la razón
(«interpretado conforme al artículo 17…»); si la descartas, di en una línea por
qué no está disponible. Y en «alternativa» escribe la vía de la otra lectura.
Si el problema no depende del sentido de una norma, no la fuerces."""
    else:
        uso = """CÓMO SE USA CON LA CALIFICACIÓN YA DECIDIDA. La escalera está al servicio de
la calificación de arriba, no la discute.
- Si la calificación FAVORECE a quien reclama el derecho, la razón se
  construye DESDE la lectura protectora: ésa es su premisa, no un adorno final.
- Si NO le favorece, recorre la escalera para mostrar POR QUÉ la lectura
  protectora no está disponible —la letra no la admite, hay una restricción
  expresa, el requisito tiene una razón que lo sostiene—. Ese diálogo es el que
  hace resistente el proyecto.
- Si la calificación no depende del sentido de una norma, NO la recorras: un
  pro persona invocado sin lecturas en disputa es retórica, y se nota.
- Cita estos registros sólo en el peldaño donde deciden algo."""
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


def cierre_estudio(material) -> str:
    """Lo último que lee el redactor antes de escribir, si el método está."""
    if not tesis_de_metodo(material):
        return ""
    puerta = con_puerta(material)
    _p = ("\n- EN LA PUERTA PROCESAL el peldaño se escribe con el artículo 17 —y su tercer\n"
          "  párrafo— y con los artículos 8.1 y 25 de la Convención Americana: el\n"
          "  requisito se lee por su finalidad, y si se exige, se dice qué finalidad\n"
          "  protege." if puerta else "")
    return f"""
Y EL DIÁLOGO CONSTITUCIONAL, donde la calificación depende de cómo se lee una
norma —sobre todo una ley local o un requisito procesal—. Va DENTRO del
problema que decide, después del marco y antes del caso concreto, en dos a
cuatro párrafos: el derecho en juego y su fuente; las lecturas posibles del
precepto y en qué se apoya cada una; cuál es la conforme y por qué; y los
límites que la Corte fijó.
- Si la calificación adopta la lectura protectora, el razonamiento se
  construye DESDE ella: es la premisa, no una coda.
- Si la rechaza, se dice por qué no está disponible —la letra no la admite,
  hay una restricción expresa, el requisito tiene una razón que lo sostiene—:
  contestar el pro persona que se planteó, y no ignorarlo, es lo que sostiene
  el proyecto en revisión.{_p}
- Los CRITERIOS DE MÉTODO del material se citan con su registro sólo en ese
  peldaño, y NO cuentan entre los tres a seis criterios del caso: dicen cómo
  interpretar, no qué se resuelve.
- Si ninguna calificación depende del sentido de una norma, este párrafo no se
  escribe. Un pro persona sin lecturas en disputa es retórica.
"""
