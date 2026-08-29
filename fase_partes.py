"""LA FICHA DE PARTES: quién es quién, y quién aportó cada prueba.

EL DEFECTO QUE ESTE MÓDULO EXISTE PARA CERRAR
═════════════════════════════════════════════
En el ADC 199-2025 el estudio generado atribuyó al QUEJOSO la aportación del
terreno y el trabajo en el hotel, que eran de la TERCERA INTERESADA. Y en un
pasaje se delató escribiendo «la tercera interesada o de la parte contraria».

Un texto que confunde a las partes no se corrige: se reescribe. Y es la peor
falla posible en un tribunal, porque el resultado es plausible —se lee bien, cita
bien, razona bien— y está mal en lo único que no puede estar mal.

LA CAUSA es que el estudio se redacta desde dos resúmenes en prosa donde los
sujetos ya vienen difuminados («la actora», «el apelante», «la parte»), sin una
tabla que fije quién es cada quien en ESTE expediente. Sin esa tabla el modelo
resuelve el sujeto por proximidad gramatical, y en un juicio con reconvención y
tercero interesado la proximidad miente.

LA REGLA: si el sujeto de una atribución no se puede resolver contra la ficha,
el párrafo se MARCA y no se entrega. Nunca una disyuntiva.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

MODELO_PARTES = os.getenv("MODELO_PARTES", "gpt-5.6-luna")
ESFUERZO_PARTES = os.getenv("ESFUERZO_PARTES", "none")


@dataclass
class Partes:
    quejoso: str = ""
    tercero_interesado: str = ""
    autoridad_responsable: str = ""
    actor_origen: str = ""
    demandado_origen: str = ""
    # [{"hecho": "...", "de_quien": "...", "papel": "ofreció|confesó|aportó"}]
    atribuciones: list[dict] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)

    def nombres(self) -> dict[str, str]:
        """nombre → papel, para poder cotejar una frase contra la ficha."""
        m = {}
        for papel, quien in (("quejoso", self.quejoso),
                             ("tercero interesado", self.tercero_interesado),
                             ("actor en el origen", self.actor_origen),
                             ("demandado en el origen", self.demandado_origen)):
            if (quien or "").strip():
                m[quien.strip()] = papel
        return m

    def bloque(self) -> str:
        """Lo que se le pone delante al redactor, antes de nada más."""
        if not (self.quejoso or self.tercero_interesado):
            return ""
        p = ["", "═" * 71, "QUIÉN ES QUIÉN EN ESTE EXPEDIENTE — SE COTEJA, NO SE SUPONE",
             "═" * 71]
        for papel, quien in (("PARTE QUEJOSA", self.quejoso),
                             ("TERCERO INTERESADO", self.tercero_interesado),
                             ("AUTORIDAD RESPONSABLE", self.autoridad_responsable),
                             ("ACTOR en el juicio de origen", self.actor_origen),
                             ("DEMANDADO en el juicio de origen", self.demandado_origen)):
            if (quien or "").strip():
                p.append(f"  {papel}: {quien}")
        if self.atribuciones:
            p.append("\n  QUIÉN APORTÓ QUÉ:")
            for a in self.atribuciones[:20]:
                p.append(f"    · {a.get('de_quien','?')} {a.get('papel','aportó')}: "
                         f"{a.get('hecho','')[:120]}")
        p += ["",
              "  ANTES DE ESCRIBIR CADA FRASE QUE ATRIBUYA UN ACTO, UNA PRUEBA O UNA",
              "  APORTACIÓN, COMPRUÉBALA CONTRA ESTA TABLA. Atribuir a la parte",
              "  equivocada produce un texto que se lee bien y está mal en lo único",
              "  que no puede estar mal.",
              "  Y NUNCA escribas una disyuntiva —«la tercera interesada o la parte",
              "  contraria»—: si no puedes resolver el sujeto, dilo y no lo atribuyas."]
        return "\n".join(p)


_PROMPT = """Eres un secretario de tribunal fichando un expediente de amparo directo.

De los dos documentos que siguen, extrae ÚNICAMENTE quién es quién y quién
aportó qué. No resumas, no opines, no interpretes.

REGLAS:
- Los NOMBRES tal como aparecen, completos. Si alguien promueve por otro
  (mandatario, apoderado), la parte es el REPRESENTADO, no el representante.
- El ACTOR y el DEMANDADO del juicio de ORIGEN no siempre coinciden con el
  quejoso: en amparo directo la quejosa puede ser la que perdió la apelación,
  sea cual fuera su papel abajo. Léelo, no lo supongas.
- En ATRIBUCIONES pon los hechos probatorios relevantes y de QUIÉN son:
  quién ofreció una prueba, quién confesó algo, quién aportó dinero o trabajo,
  quién adquirió un bien. Es lo que evita que después se le adjudiquen al
  contrario.
- Si un dato no consta, cadena vacía. NUNCA lo inventes.

Devuelve JSON y nada más:
{{"quejoso":"", "tercero_interesado":"", "autoridad_responsable":"",
  "actor_origen":"", "demandado_origen":"",
  "atribuciones":[{{"hecho":"", "de_quien":"", "papel":"ofreció|confesó|aportó|adquirió"}}]}}

═══ SENTENCIA RECLAMADA ═══
{acto}

═══ DEMANDA DE AMPARO ═══
{conceptos}"""


async def fichar(cliente, texto_acto: str, texto_conceptos: str) -> Partes:
    kw = dict(model=MODELO_PARTES, max_completion_tokens=4000,
              response_format={"type": "json_object"},
              messages=[{"role": "user", "content": _PROMPT.format(
                  acto=(texto_acto or "")[:22000],
                  conceptos=(texto_conceptos or "")[:14000])}])
    if ESFUERZO_PARTES:
        kw["reasoning_effort"] = ESFUERZO_PARTES
    r = await cliente.chat.completions.create(**kw)
    try:
        j = json.loads((r.choices[0].message.content or "{}").strip())
    except Exception as e:
        return Partes(avisos=[f"No se pudo fichar a las partes: {e}"])
    p = Partes(
        quejoso=j.get("quejoso", ""), tercero_interesado=j.get("tercero_interesado", ""),
        autoridad_responsable=j.get("autoridad_responsable", ""),
        actor_origen=j.get("actor_origen", ""), demandado_origen=j.get("demandado_origen", ""),
        atribuciones=[a for a in (j.get("atribuciones") or []) if isinstance(a, dict)])
    if not p.quejoso:
        p.avisos.append("No se identificó a la parte quejosa: el redactor va a "
                        "resolver los sujetos por proximidad y puede equivocarse.")
    return p


# ═══════════════════════════════════════════════════════════════════════════
# La comprobación, que es lo que de verdad cierra el hueco
# ═══════════════════════════════════════════════════════════════════════════

# «la tercera interesada o de la parte contraria» — la disyuntiva es la señal
# de que el modelo no supo de quién hablaba y lo dejó abierto.
_RX_DISYUNTIVA = re.compile(
    r"\b(la\s+(?:parte\s+)?(?:quejosa|actora|demandada|recurrente)|el\s+quejoso"
    r"|la\s+tercera?\s+interesada?)\s+o\s+(?:la|el)\s+"
    r"(?:parte\s+)?(?:contraria|quejosa|actora|demandada|tercera?\s+interesada?)", re.I)


def revisar_partes(estudio: str, p: Partes) -> list[str]:
    """Lo comprobable sin modelo sobre la atribución de sujetos."""
    avisos: list[str] = []

    dis = _RX_DISYUNTIVA.findall(estudio or "")
    if dis:
        avisos.append(f"DISYUNTIVAS DE SUJETO ({len(dis)}): el texto no sabe de "
                      f"quién habla — {sorted(set(dis))[:3]}. Resuélvelo antes de "
                      f"entregar: no se firma una sentencia que duda de quién es quién.")

    # Un nombre de la ficha usado con el papel del contrario.
    for nombre, papel in p.nombres().items():
        corto = nombre.split()[0] if nombre.split() else ""
        if len(corto) < 4:
            continue
        for m in re.finditer(re.escape(corto), estudio or ""):
            ventana = estudio[max(0, m.start() - 90):m.start()].lower()
            otros = {"quejoso": ("tercero interesado", "tercera interesada"),
                     "tercero interesado": ("quejosa", "quejoso")}.get(papel, ())
            if any(o in ventana for o in otros):
                avisos.append(f"«{nombre}» ({papel}) aparece descrito con el papel "
                              f"contrario. Cotéjalo contra la ficha de partes.")
                break
    return avisos
