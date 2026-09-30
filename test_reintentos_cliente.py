# -*- coding: utf-8 -*-
"""El reintento del chat, contado (30-sep-2026): la línea de registro y dónde se engancha.

    .venv/bin/python test_reintentos_cliente.py
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import reintentos_cliente as rc

FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


print("\n1 · LA LÍNEA")
l = rc.linea({"intento": 1, "tipo": "red", "status": 0, "espera_ms": 2000, "error": "Load failed"})
ok(l == "🔁 REINTENTO_CHAT intento=1 tipo=red status=0 espera_ms=2000 error='Load failed'", f"la línea completa ({l})")
ok(rc.MARCA in l, "lleva la marca que se busca en Render")
l = rc.linea({"intento": "99", "tipo": "Ocupado\n<script>", "status": 5030, "error": "x\ny" * 100})
ok("\n" not in l and "intento=10" in l and "status=999" in l and "tipo=ocupadoscript" in l,
   "lo que manda el navegador va acotado y sin saltos de línea")
ok(len(l) < 200, "y corta")
ok(rc.linea(None) == "" and rc.linea("texto") == "" and rc.linea([1]) == "", "lo que no es un dict no se apunta")
ok("tipo=desconocido" in rc.linea({}), "sin tipo, «desconocido»")

print("\n2 · DÓNDE SE ENGANCHA")
F = Path("main.py").read_text(encoding="utf-8")
CR = F[F.index("class ChatRequest(BaseModel):"):]
CR = CR[:CR.index("\nclass ")]
ok("reintento: Optional[Dict[str, Any]] = Field(" in CR, "ChatRequest acepta `reintento` (opcional: los clientes viejos no lo mandan)")
CE = F[F.index("async def chat_endpoint("):]
i_val = CE.index('detail="Se requiere al menos un mensaje"')
i_rei = CE.index("_rc.linea(request.reintento)")
i_san = CE.index("from input_sanitizer import sanitize_input")
ok(i_val < i_rei < i_san, "se apunta al entrar, antes de todo lo demás (también si luego falla)")

print()
if FALLOS:
    print("FALLAS:\n  " + "\n  ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
