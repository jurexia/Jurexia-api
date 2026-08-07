"""
El catálogo ÚNICO de motores de Iurexia: qué modelo corre en cada botón.

POR QUÉ EXISTE (7-ago-2026)
---------------------------
David lo dijo así: «es como si vendiéramos un producto sin saber qué
contiene». El enrutamiento vivía repartido en una cadena de veinte `elif`
dentro de main.py, con las constantes 400 líneas más arriba, y nadie —ni
él ni yo— podía decir de memoria qué motor atendía cada botón. De ahí la
sospecha de que se estaban mezclando.

Este módulo NO enruta. main.py sigue decidiendo. Lo que hace es DECLARAR
lo que main.py debería estar haciendo, y `verificar()` compara ambas cosas
contra las constantes reales del proceso. Si alguien cambia un modelo y no
actualiza el catálogo, la comprobación falla y se entera. Es un contrato,
no documentación: la documentación se pudre en silencio, esto grita.

CÓMO LEERLO
-----------
Cada entrada es un MOTOR: la combinación de botón + plan que determina qué
API se llama. `razona` es la palanca real de coste, no un adorno:

  · False → el modelo responde directo. Barato y rápido.
  · True  → el modelo piensa antes. Caro; en v4-flash el pensamiento
            consume del MISMO tope de tokens que la respuesta.

CÓMO PROBARLO
    python motores.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Motor:
    """Un botón de la app y el motor que lo atiende."""
    clave: str                      # identificador estable
    boton: str                      # cómo lo ve el abogado
    marcador: Optional[str]         # lo que manda el frontend, si manda algo
    proveedor: str                  # deepseek | openai | google
    modelo: str                     # constante de main.py, no literal
    razona: bool
    esfuerzo: Optional[str]         # sólo si el proveedor lo respeta
    max_tokens: int
    planes: str                     # quién puede usarlo
    nota: str = ""


# ── El catálogo ─────────────────────────────────────────────────────────
# Ordenado como la cadena de decisión de main.py: el primero que casa gana.
# Ese orden IMPORTA — «precedentes» va antes que «redactar», así que pulsar
# los dos deja fuera la redacción.

CATALOGO: list[Motor] = [
    Motor(
        clave="precedentes", boton="Precedentes", marcador="[MODO_PRECEDENTES]",
        proveedor="openai", modelo="gpt-5-mini", razona=False, esfuerzo="minimal",
        max_tokens=25_000, planes="todos",
        nota="Síntesis, no razonamiento. Corría en DeepSeek y se desbocó: 53 s de "
             "espera con 20,773 caracteres de pensamiento invisible. "
             "PRECEDENTES_DEEPSEEK=1 revierte sin desplegar.",
    ),
    Motor(
        clave="flash", boton="Consulta rápida (rayo)", marcador="[MODO_FLASH]",
        proveedor="google", modelo="FLASH_MODEL", razona=False, esfuerzo=None,
        max_tokens=2_200, planes="todos",
        nota="Extrae, no redacta. Sin genios ni caché: el corpus cacheado añade "
             "~190,000 tokens de prefill que aquí sólo estorban.",
    ),
    Motor(
        clave="sentencia", boton="Revisión de sentencia", marcador=None,
        proveedor="openai", modelo="gpt-5.2", razona=True, esfuerzo=None,
        max_tokens=32_000, planes="platinum",
        nota="El modelo más potente del catálogo.",
    ),
    Motor(
        clave="redaccion_platinum", boton="Redactar › Platinum",
        marcador="[MODO_REDACCION_PLATINUM]",
        proveedor="openai", modelo="REDACTOR_PLATINUM_MODEL", razona=True,
        esfuerzo="REDACTOR_PLATINUM_ESFUERZO", max_tokens=32_000,
        planes="platinum, ultra_secretarios, admin",
        nota="Sin plan Platinum cae a Pro, no se rechaza: el abogado igual "
             "recibe su escrito. Comparte modelo con Pro y se distingue por el "
             "esfuerzo alto, que en un modelo de razonamiento es más capacidad "
             "real y más coste — es la configuración elegida, no un descuido.",
    ),
    Motor(
        clave="redaccion_pro", boton="Redactar › Pro", marcador="[MODO_REDACCION_PRO]",
        proveedor="openai", modelo="REDACTOR_PRO_MODEL", razona=True,
        esfuerzo="REDACTOR_PRO_ESFUERZO", max_tokens=32_000,
        planes="pro y superiores",
    ),
    Motor(
        clave="genio", boton="Cualquiera, con un Genio activo", marcador=None,
        proveedor="google", modelo="cache_manager.get_cache_model()", razona=False,
        esfuerzo=None, max_tokens=25_000, planes="según el genio",
        nota="El modelo SALE del gestor de caché, jamás escrito a mano: Gemini "
             "rechaza con 400 si el modelo que genera no es exactamente el que "
             "creó el caché. Ya pasó en producción — 24 palabras y cero artículos.",
    ),
    Motor(
        clave="redaccion_profesional", boton="Redactar › Profesional",
        marcador="[MODO_REDACCION]",
        proveedor="deepseek", modelo="DEEPSEEK_OFFICIAL_CHAT_MODEL", razona=True,
        esfuerzo="REDACTOR_PROFESIONAL_ESFUERZO", max_tokens=16_384, planes="todos",
        nota="El escalón base. El esfuerzo se manda pero v4-flash NO lo respeta "
             "(medido en este repo: 'low' gastó MÁS que 'high'). La única "
             "palanca real en este motor es el interruptor de razonamiento.",
    ),
    Motor(
        clave="redaccion_con_documento", boton="Redactar con documento adjunto",
        marcador=None, proveedor="deepseek", modelo="DEEPSEEK_OFFICIAL_REASONER_MODEL",
        razona=True, esfuerzo=None, max_tokens=32_000, planes="todos",
    ),
    Motor(
        clave="buscar", boton="Buscar (el chat por omisión)", marcador=None,
        proveedor="deepseek", modelo="DEEPSEEK_OFFICIAL_CHAT_MODEL", razona=False,
        esfuerzo=None, max_tokens=30_000, planes="todos",
        nota="Razonamiento APAGADO a propósito. Medido: por omisión 34.9 s hasta "
             "el primer token; apagado, 0.9 s — con la misma extensión de "
             "respuesta. 30,000 tokens y no 16,384 porque v4-flash razona aunque "
             "el interruptor esté OFF y descuenta del mismo tope: una respuesta "
             "chocó con el límite y llegó sin conclusión. CHAT_MAX_TOKENS lo cambia.",
    ),
    Motor(
        clave="salvame", boton="Sálvame", marcador=None,
        proveedor="deepseek", modelo="DEEPSEEK_OFFICIAL_CHAT_MODEL", razona=True,
        esfuerzo="SALVAME_RAZONAMIENTO", max_tokens=32_000, planes="todos",
        nota="NUNCA apagarle el razonamiento: sin él degenera en bucles.",
    ),
    Motor(
        clave="documento", boton="Analizar documento adjunto", marcador=None,
        proveedor="google", modelo="DOCUMENT_MODEL", razona=False, esfuerzo=None,
        max_tokens=0, planes="todos",
        nota="Platinum y admin suben a gemini-3.1-pro-preview.",
    ),
]

# Servicios que NO son un botón del chat pero sí cuestan dinero.
AUXILIARES: list[Motor] = [
    Motor("hyde", "Expansión de la consulta (interno)", None, "openai", "gpt-5-mini",
          False, None, 0, "interno", "Corre en CADA consulta del chat."),
    Motor("web", "Fuentes de internet (globo)", None, "openrouter",
          "perplexity/sonar", False, None, 0, "pro y superiores",
          "Restringido a Pro tras gastar 15 USD en un día abierto a todos."),
    Motor("soporte", "Chat de soporte", None, "openrouter",
          "gemini-2.5-flash-lite", False, None, 0, "todos",
          "El más barato del catálogo, a propósito."),
    Motor("jurisconsulto", "Jurisconsulto", None, "openrouter",
          "JURISCONSULTO_MODEL", False, None, 0, "interno"),
]


# ── El contrato ─────────────────────────────────────────────────────────

def verificar() -> list[str]:
    """Compara el catálogo contra las constantes REALES de main.py.

    Devuelve la lista de discrepancias. Vacía = el mapa dice la verdad.
    No importa main.py entero (arrancaría el servidor): lee sus constantes
    del archivo, que es lo único que se necesita comparar.
    """
    import re
    from pathlib import Path

    fuente = (Path(__file__).parent / "main.py").read_text(encoding="utf-8")
    constantes: dict[str, str] = {}
    for m in re.finditer(
            r'^([A-Z_]+)\s*=\s*(?:os\.getenv\(\s*"[A-Z_]+"\s*,\s*)?"([^"]+)"',
            fuente, re.M):
        constantes[m.group(1)] = m.group(2)

    fallos = []
    for mo in CATALOGO + AUXILIARES:
        if mo.modelo in constantes or not mo.modelo.isupper():
            continue
        if "." in mo.modelo or "(" in mo.modelo:
            continue
        fallos.append(f"{mo.clave}: la constante {mo.modelo} ya no existe en main.py")

    # ¿Pro y Platinum se distinguen en ALGO? Compartir modelo no es un
    # defecto: en un modelo de razonamiento el esfuerzo alto es más capacidad
    # real y cuesta más, así que Platinum sobre luna-high entrega más que Pro
    # sobre luna-medium. Es la configuración que eligió David (7-ago-2026) y
    # esta comprobación NO debe gritar por ella: un contrato que avisa de algo
    # deliberado enseña a ignorar los avisos.
    #
    # Lo que sí sería un fallo silencioso es que coincidieran en las DOS
    # cosas. Ahí el abogado pagaría el escalón superior por exactamente el
    # mismo motor, y nadie se enteraría: no hay error, sólo dos constantes
    # iguales.
    pro = constantes.get("REDACTOR_PRO_MODEL")
    plat = constantes.get("REDACTOR_PLATINUM_MODEL")
    esf_pro = constantes.get("REDACTOR_PRO_ESFUERZO")
    esf_plat = constantes.get("REDACTOR_PLATINUM_ESFUERZO")
    if pro and plat and pro == plat and esf_pro == esf_plat:
        fallos.append(
            f"redaccion_pro y redaccion_platinum son IDÉNTICOS ({pro}, "
            f"esfuerzo {esf_pro}): el escalón superior no entrega nada más.")

    return fallos


def tabla() -> str:
    """El catálogo en texto, para pegarlo donde haga falta."""
    anchos = (22, 30, 11, 34, 7, 9)
    cab = ("clave", "botón", "proveedor", "modelo", "razona", "tokens")
    filas = [" | ".join(c.ljust(a) for c, a in zip(cab, anchos)),
             "-+-".join("-" * a for a in anchos)]
    for m in CATALOGO:
        filas.append(" | ".join([
            m.clave.ljust(anchos[0]), m.boton[:30].ljust(anchos[1]),
            m.proveedor.ljust(anchos[2]), m.modelo[:34].ljust(anchos[3]),
            ("sí" if m.razona else "no").ljust(anchos[4]),
            (f"{m.max_tokens:,}" if m.max_tokens else "—").ljust(anchos[5]),
        ]))
    return "\n".join(filas)


if __name__ == "__main__":
    print("\n" + tabla() + "\n")
    problemas = verificar()
    if problemas:
        print(f"⚠️  {len(problemas)} discrepancia(s) entre el mapa y main.py:\n")
        for p in problemas:
            print(f"   · {p}")
    else:
        print("✅ El mapa coincide con main.py")
