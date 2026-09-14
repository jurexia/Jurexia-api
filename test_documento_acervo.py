# -*- coding: utf-8 -*-
"""EL ACERVO ENTRA AL DOCUMENTO ADJUNTO — la mecánica, sin red.

David, 14-sep-2026: «Empieza por darle acervo a la vía con documento adjunto».

Lo que se comprueba aquí es lo que se puede comprobar sin Qdrant ni modelo:
que el prompt SIN acervo sea exactamente el de siempre (la ruta de respaldo
no puede cambiar por accidente), que el prompt CON acervo traiga las reglas
que importan y no la frase «no tienes acervo», y que la consulta al acervo se
arme con la instrucción, los ordenamientos del documento y su arranque, sin
pasarse de tamaño. Si el acervo devuelve lo bueno o lo malo se mide en
producción, sobre la cuenta de demostración.
"""
import sys
import documento_acervo as da

fallos = []


def ok(cond, nota):
    print(f"  {'OK ' if cond else 'MAL'} {nota}")
    if not cond:
        fallos.append(nota)


# ── 1. Los dos prompts ───────────────────────────────────────────────────────
sin = da.prompt_documento(con_acervo=False)
con = da.prompt_documento(con_acervo=True)

ok(sin.startswith("Eres Iurexia"), "el prompt sin acervo arranca como siempre")
ok("NO CITES JURISPRUDENCIA NI TESIS" in sin, "sin acervo: sigue prohibiendo tesis")
ok("NO tienes acervo que consultar" in sin, "sin acervo: sigue diciendo que no hay acervo")
ok("folio 1946-02" in sin, "sin acervo: conserva el caso de Puebla que enseña la regla")
ok(sin.rstrip().endswith("trabajo jurídico."), "sin acervo: conserva el cierre")

ok(con.startswith("Eres Iurexia"), "con acervo: mismas reglas 1-6 al frente")
ok("[Doc ID: uuid]" in con, "con acervo: exige Doc ID en cada cita")
ok("PROHIBIDO MUDAR UN ARTÍCULO DE UNA LEY A OTRA" in con, "con acervo: prohíbe mudar artículos entre leyes")
ok("Transcribe el texto del artículo LITERAL" in con, "con acervo: exige transcripción literal")
ok("NO tienes acervo" not in con and "no tengo acervo" not in con, "con acervo: NO dice que no hay acervo")
ok("NO CITES JURISPRUDENCIA NI TESIS. NUNCA" not in con, "con acervo: ya no prohíbe toda jurisprudencia")
ok("sin Doc ID, para ti, no existe" in con, "con acervo: tesis sin Doc ID no existe")
ok(con.rstrip().endswith("trabajo jurídico."), "con acervo: conserva el cierre")
ok(sin[:sin.index("7. **")] == con[:con.index("7. **")], "las reglas 1-6 son idénticas en ambos")

# ── 2. Los ordenamientos que nombra un documento ────────────────────────────
demanda = (
    "JUICIO ORDINARIO CIVIL. Con fundamento en los artículos 1, 2 y 3 del Código de "
    "Procedimientos Civiles del Estado de Querétaro, y en los artículos 2398 y 2483 del "
    "Código Civil del Estado de Querétaro, vengo a demandar. Invoco además la Ley de "
    "Amparo y la Constitución Política de los Estados Unidos Mexicanos. El Código Civil "
    "del Estado de Querétaro dispone en su artículo 2483 que el arrendamiento..."
)
leyes = da.leyes_mencionadas(demanda)
ok("Código Civil del Estado de Querétaro" in leyes, f"encuentra el Código Civil de Querétaro: {leyes}")
ok("Código de Procedimientos Civiles del Estado de Querétaro" in leyes, "encuentra el procesal de Querétaro")
ok("Ley de Amparo" in leyes, "encuentra la Ley de Amparo")
ok(any(l.startswith("Constitución Política") for l in leyes), "encuentra la Constitución")
ok(leyes[0] == "Código Civil del Estado de Querétaro", "el más nombrado va primero")
ok(len(leyes) == len({l.lower() for l in leyes}), "sin duplicados")
ok(da.leyes_mencionadas("Este contrato no cita ley alguna.") == [], "un texto sin ordenamientos da lista vacía")
ok(da.leyes_mencionadas("La ley que rige el caso es clara.") == [], "«la ley que» no es un ordenamiento")

# ── 3. La consulta al acervo ────────────────────────────────────────────────
q = da.consulta_para_acervo("Revisa la cláusula de tácita reconducción y fundaméntala", demanda, "demanda.pdf")
ok(q.startswith("Revisa la cláusula de tácita reconducción"), "la instrucción del abogado va primero")
ok("Ordenamientos que invoca el documento: Código Civil del Estado de Querétaro" in q, "los ordenamientos van después, el más nombrado primero")
ok("Arranque del documento: JUICIO ORDINARIO CIVIL." in q, "el arranque del documento cierra la consulta")
ok(len(q) <= 2000, f"cabe en 2.000 caracteres ({len(q)})")

largo = "x" * 5000 + " Código Civil Federal " + "y" * 200000
q2 = da.consulta_para_acervo("p" * 3000, largo)
ok(len(q2) <= 2000, f"con instrucción y documento enormes sigue cabiendo ({len(q2)})")
ok(q2.startswith("p" * 600 + "\n"), "la instrucción se recorta a 600")

q3 = da.consulta_para_acervo("", "", "solo_nombre.pdf")
ok(q3 == "Documento: solo_nombre.pdf", "sin instrucción ni texto queda el nombre del archivo")

q4 = da.consulta_para_acervo("  varios   espacios\n\ny saltos ", "  texto\tcon\ttabs  ")
ok(q4 == "varios espacios y saltos\nArranque del documento: texto con tabs", "colapsa espacios y saltos")

print()
if fallos:
    print(f"{len(fallos)} FALLO(S):")
    for f in fallos:
        print("  -", f)
    sys.exit(1)
print("todo en orden")
