# -*- coding: utf-8 -*-
"""EL CONTRASTE — el paso 1 para automatizar el sentido.

David, 14-sep-2026: «adelante, empieza por el paso 1 y mídelo sobre los 72».

Lo que se comprueba aquí es la MECÁNICA, sin red y sin modelo: que el prompt se
arme con lo que hay, que la respuesta se lea con tolerancia, que los veredictos
se normalicen, y —lo que importa— que el bloque que entra a la propuesta lleve
LA REGLA: un inoperante no puede volverse fundado sin decir por qué el contraste
se equivoca. La medida de si acierta está en banco_kingston.py, contra los 24
engroses reales; esto sólo garantiza que el paso exista y no se rompa mudo.

CALIBRADO EN LAS DOS DIRECCIONES: un contraste vacío no bloquea nada, y un
contraste con todo «a examinar» tampoco. Si la regla se colara donde no toca,
sería la verificación la que estaría mal.
"""
import asyncio, json, sys
import fase5_propuesta as f5

fallos = []
def ok(cond, nota):
    print(f"  {'OK ' if cond else 'MAL'} {nota}")
    if not cond: fallos.append(nota)

PROBLEMAS = [
    {"pregunta": "¿La Sala valoró bien la pericial?",
     "resolvio": "La Sala tuvo la pericial por desahogada conforme al art. 346.",
     "combate": "Que la pericial no se desahogó porque el perito no ratificó."},
    {"pregunta": "¿Debió suplirse la queja?",
     "resolvio": "No se pronunció.",
     "combate": "Que debió suplirse por tratarse de un menor."},
]

# ── el prompt ──────────────────────────────────────────────────────────────
pr = f5.prompt_contraste(PROBLEMAS, "La Sala confirmó.", "Se alega indebida valoración.", False)
ok("razon_toral" in pr and "la_combate" in pr and "sobrevive" in pr, "el prompt pide las tres preguntas")
ok("concepto de violación" in pr and "agravio" not in pr.split("EL CONTRASTE")[0], "vocabulario de amparo directo")
ok("agravio" in f5.prompt_contraste(PROBLEMAS, "", "", True), "vocabulario de recurso")
ok("1. ¿La Sala valoró bien la pericial?" in pr and "Resolvió:" in pr, "los planteamientos van numerados con lo que resolvió")

# ── un cliente falso: devuelve lo que se le diga ───────────────────────────
class _R:
    def __init__(self, txt): self.choices = [type("c", (), {"message": type("m", (), {"content": txt})()})()]
class Falso:
    def __init__(self, txt): self.txt = txt; self.chat = self
    @property
    def completions(self): return self
    async def create(self, **kw): return _R(self.txt)

def corre(txt):
    return asyncio.run(f5.contrastar(Falso(txt), PROBLEMAS, "", "", False))

# ── lectura tolerante y normalización ──────────────────────────────────────
bueno = json.dumps({"contraste": [
    {"numero": 1, "razon_toral": "El perito sí ratificó según constancia de f. 40",
     "la_combate": False, "sobrevive": False, "veredicto_previo": "Inoperante",
     "por_que": "Parte de una premisa falsa: la ratificación consta."},
    {"numero": 2, "razon_toral": "no consta", "la_combate": True, "sobrevive": False,
     "veredicto_previo": "a examinar", "por_que": "Hay que decidirlo."}]})
c = corre("Aquí va el JSON:\n" + bueno + "\nfin.")
ok(len(c) == 2, "lee dos entradas aunque el JSON venga envuelto en texto")
ok(c[0]["veredicto_previo"] == "inoperante", "normaliza «Inoperante» → inoperante")
ok(c[1]["veredicto_previo"] == "a_examinar", "normaliza «a examinar» → a_examinar")
ok(corre("no hay json") == [], "sin JSON devuelve vacío, no revienta")
ok(corre(json.dumps({"contraste": [{"numero": 1, "veredicto_previo": "cualquier cosa"}]}))[0]["veredicto_previo"] == "a_examinar",
   "un veredicto desconocido cae a «a examinar», nunca a inoperante")

# ── el bloque y su regla ───────────────────────────────────────────────────
b = f5.bloque_contraste(c)
ok("Problema 1: INOPERANTE" in b, "el bloque marca el inoperante")
ok("el contraste se equivoca porque" in b, "lleva la cláusula de desmarque, con esas palabras")
# LA REGLA DEL GLOBAL CAMBIÓ tras medir: la primera versión ataba el asunto al
# pre-veredicto del contraste y en el ADC 296-2025 el único problema quedó
# fundado y el global salió infundado — el prompt se contradecía a sí mismo.
# Ahora el global sale de las calificaciones del propio modelo.
ok("EL ASUNTO ENTERO prospera si prospera al menos un problema" in b, "lleva la regla del global: sigue a las calificaciones")
ok("Sólo los problemas A EXAMINAR pueden prosperar" not in b, "y ya NO ata el global al pre-veredicto")
ok("(a) la resolución enuncia OTRA consideración" in f5.prompt_contraste(PROBLEMAS, "", "", False),
   "«sobrevive» exige nombrar una consideración independiente")
ok(f5.bloque_contraste([]) == "", "sin contraste, sin bloque: no bloquea nada")

# ── entra al prompt de la propuesta, antes de las calificaciones ───────────
class _M: tipo_asunto = "amparo_directo"; sondeo = None; tesis = []; normas = []; espejo = None
try:
    pp = f5.prompt_propuesta(PROBLEMAS, _M(), "acto", "conceptos", False, "", bloque_contraste := b)
    ok(pp.index("EL CONTRASTE, HECHO ANTES DE CALIFICAR") < pp.index("CÓMO SE CALIFICA"),
       "el contraste va ANTES de las definiciones de calificación")
    ok("EL CONTRASTE" not in f5.prompt_propuesta(PROBLEMAS, _M(), "acto", "conceptos", False, "", ""),
       "sin bloque, el prompt de la propuesta es el de siempre")
except Exception as e:
    ok(False, f"prompt_propuesta con contraste: {type(e).__name__}: {str(e)[:100]}")

print()
if fallos:
    print(f"FALLAN {len(fallos)}: " + " · ".join(fallos)); sys.exit(1)
print("TODAS LAS COMPROBACIONES PASAN")
