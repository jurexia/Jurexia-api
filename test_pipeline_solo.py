"""El modelo lee entero, y la propuesta corre sola tras el adelanto — 17-sep-2026.

David: «El modelo debe leer todo, ya hemos establecido que no hay límites en
lectura» (los resúmenes recibían el documento desde su marcador y la cabeza se
tiraba: «NO SE LEYÓ … ENTERO: 79.953 de 100.220»). Y: «cuando entrega el
asunto en corto debería ya estarse buscando la solución jurídica y contar con
una propuesta global y por puntos que el secretario pueda cambiar».

    .venv/bin/python test_pipeline_solo.py
"""
import fases123_pipeline as f123

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


print("\n1 · EL MODELO LEE EL DOCUMENTO ENTERO")
CABEZA = "CABECERA-DEL-DOCUMENTO Resultando primero: la demanda se presentó. " * 40
ESTUDIO = "\n\nC O N S I D E R A N D O\n\nQUINTO. Estudio de fondo. La Sala razona que… " * 40
acto = CABEZA + ESTUDIO
f123.olvidar_descartes()
pa = f123.prompt_resumen_acto(acto, True, "revision_fiscal")
ok("CABECERA-DEL-DOCUMENTO" in pa, "el resumen del acto recibe la cabeza (los resultandos), no sólo desde el considerando")
ok(f123.descartado() == [], "y no se apunta ningún descarte: no hay aviso «NO SE LEYÓ … ENTERO»")
pf = f123.prompt_acto_a_fondo(acto, "resumen", [], 1000, True, "revision_fiscal")
ok("CABECERA-DEL-DOCUMENTO" in pf, "la segunda vuelta del acto también lee entero")
PROEMIO = "PROEMIO-DEL-ESCRITO Vengo a promover juicio de amparo directo. " * 40
CONCEPTOS = "\n\nCONCEPTOS DE VIOLACIÓN\n\nPRIMERO. La sentencia viola… " * 40
esc = PROEMIO + CONCEPTOS
pc = f123.prompt_resumen_conceptos(esc, False, "amparo_directo")
ok("PROEMIO-DEL-ESCRITO" in pc, "el resumen del escrito recibe el proemio, no sólo desde «CONCEPTOS DE VIOLACIÓN»")
ok(f123.descartado() == [], "sin descarte tampoco aquí")
enfoque = f123.recortar_acto(acto)
ok("CABECERA-DEL-DOCUMENTO" not in enfoque and "Estudio de fondo" in enfoque,
   "el ENFOQUE (para contar palabras y tesis de la responsable) sigue mirando el estudio")
ok(f123.objetivo_acto(acto) == f123.objetivo_acto(ESTUDIO),
   "y el objetivo de palabras no crece por leer los resultandos")
grande = "x" * (f123.TOPE_CARACTERES + 5000)
f123.olvidar_descartes()
_ = f123.texto_entero(grande, f123.TOPE_CARACTERES, "el acto reclamado")
ok(f123.descartado() and f123.descartado()[0][0] == "el acto reclamado",
   "sólo un fichero por encima del freno (600.000) deja descarte y aviso")
f123.olvidar_descartes()

print("\n2 · LA PROPUESTA CORRE SOLA Y EL BOTÓN LA SIRVE")
m = open("main.py", encoding="utf-8").read()
i_nuc = m.find("async def _taller_proponer_nucleo(")
i_dec = m.find('@app.post("/taller/proponer")')
i_str = m.find('@app.post("/taller/resolver/stream")')
ok(0 < i_nuc < i_dec, "el núcleo de la propuesta está fuera del endpoint")
nucleo = m[i_nuc:i_dec]
ok('_taller_registrar_uso(user_email, numero, "propuesta")' not in nucleo
   and "await _f5.proponer(" in nucleo and 'ses["global"] = glob' in nucleo,
   "el núcleo propone y guarda, pero no registra uso (eso lo hace quien lo sirve)")
ok('huella=_te.huella_contraste(r))' in nucleo, "y el material que completa lo guarda con la huella")
handler = m[i_dec:i_str]
ok("await _taller_proponer_nucleo(user_email, numero, ses, contexto)" in handler
   and handler.count('_taller_registrar_uso(user_email, numero, "propuesta")') == 2,
   "el endpoint llama al núcleo y registra el uso en las dos salidas")
ok('_taller_leer_marca(user_email, numero, "propuesta")' in handler
   and 'if not (contexto or "").strip():' in handler and 'return _previa' in handler,
   "sin contexto sirve la propuesta calculada sola; con contexto, calcula")
i_pre = m.find("async def _taller_preconsultar(")
i_pp = m.find("async def _taller_preproponer(")
ok(0 < i_pre < m.find("await _taller_preproponer(email, numero, r)", i_pre) < i_pp,
   "la consulta sola encadena la propuesta sola")
pp = m[i_pp:m.find("\n\n\ndef _taller_avance(")]
ok('"estado": "en_curso"' in pp and '"respuesta": resp' in pp and '"estado": "fallo"' in pp
   and "await _taller_proponer_nucleo(email, numero, ses, \"\")" in pp,
   "la propuesta sola marca en curso, guarda la respuesta entera al terminar y marca el fallo")
ok('ses.get("resultado") is not r or ses.get("material") is None' in pp,
   "y no corre si la sesión ya no es la de este adelanto o no hay acervo")
i_ctx = m.find('@app.get("/taller/contexto-del-asunto")')
ok(0 < i_ctx < m.find('"avance": _taller_avance(user_email, numero),', i_ctx) < m.find('@app.post("/taller/consultar")'),
   "/taller/contexto-del-asunto dice cómo va lo que corre solo")
ok('.select("estado->consulta, estado->contraste, estado->propuesta")' in m,
   "y lo lee por sus ramas, no la fila entera")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
