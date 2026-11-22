# -*- coding: utf-8 -*-
"""La jurimetría que no miente (jurimetria_v2.py), sin red. 22-nov-2026.

Lo que fallaba en vivo y se comprueba aquí que ya no pasa:
  1 · con 0 resultados contestaba «NIEGA»;
  2 · una sentencia con cinco puntos votaba cinco veces;
  3 · amparo directo, revisión y queja se mezclaban en un solo porcentaje;
  4 · «modifica» contaba como sobreseer;
  5 · el endpoint filtraba por `acto_reclamado_tipo` (sin índice: 400 del modo estricto → 500) y por la materia
      cruda («administrativo» contra lo guardado «administrativa»: 0 resultados).
"""
import io, re, sys
import jurimetria_v2 as jv

fallos = []


def ok(cond, msg):
    if not cond:
        fallos.append(msg)


def pt(sid, score, tipo="amparo_directo", prospera="no", circ="22", periodo="anterior", **kw):
    pl = {"j_sentencia": sid, "j_tipo": tipo, "j_prospera": prospera, "circuito": circ, "j_periodo": periodo}
    pl.update(kw)
    return {"id": sid, "score": score, "payload": pl}


# 1 · sin base no hay sentido
e = jv.agregar([], [], circuito="5")
ok(e["sentido_probable"] == jv.SIN_BASE, f"sin resultados dijo {e['sentido_probable']!r}")
ok(e["n_base"] == 0 and e["probabilidades"] == {}, "sin resultados hay probabilidades")
ok(e["confianza"] == "baja", "sin resultados la confianza no es baja")

# circuito sin sentencias pero con nacionales: NO se pinta lo nacional como del circuito
nac = [pt(f"n{i}", 0.8, prospera="si" if i % 3 else "no", circ="1") for i in range(30)]
e = jv.agregar([], nac, circuito="5")
ok(e["sentido_probable"] == jv.SIN_BASE, "circuito vacío tomó el sentido nacional")
ok(e["nacional"]["n"] == 30, "lo nacional no se informó aparte")

# 2 · una sentencia, un voto
loc = [pt("A", 0.9, prospera="si")] * 5 + [pt(f"B{i}", 0.8, prospera="no") for i in range(12)]
e = jv.agregar(loc, loc)
ok(e["n_base"] == 13, f"una sentencia con 5 puntos votó varias veces (n={e['n_base']})")
ok(e["sentido_probable"] == "NIEGA", "12 no y 1 sí no dio NIEGA")

# la base es el mismo problema; si no alcanza, los parecidos y se dice
mismo = [pt(f"m{i}", 0.80, prospera="si") for i in range(15)]
lejos = [pt(f"l{i}", 0.50, prospera="no") for i in range(40)]
e = jv.agregar(mismo + lejos, mismo + lejos)
ok(e["base"] == "mismo_problema" and e["n_base"] == 15, f"no tomó sólo el mismo problema ({e['base']}, {e['n_base']})")
ok(e["sentido_probable"] == "CONCEDE", "el mismo problema prospera y salió otra cosa")
e = jv.agregar(lejos[:20], lejos[:20])
ok(e["base"] == "parecidos" and "más parecidas" in e["aviso"], "sin el mismo problema no avisó que son parecidas")

# 3 · por tipo de asunto
mix = [pt(f"d{i}", 0.8, "amparo_directo", "si") for i in range(14)] + \
      [pt(f"q{i}", 0.8, "queja", "no") for i in range(30)]
e = jv.agregar(mix, mix)
ok(e["tipo_asunto"] == "queja" and e["n_base"] == 30, "sin tipo pedido no tomó el dominante")
e = jv.agregar(mix, mix, tipo="amparo_directo")
ok(e["tipo_asunto"] == "amparo_directo" and e["n_base"] == 14 and e["sentido_probable"] == "CONCEDE",
   "el tipo pedido no separó la base")
ok(e["por_tipo"].get("queja") == 30, "no informa el reparto por tipo")

# 4 · «parcial» (modifica) prospera en parte: en la pantalla va con CONCEDE, nunca con SOBRESEE
e = jv.agregar([pt(f"p{i}", 0.8, "amparo_revision", "parcial") for i in range(13)], None)
ok(e["probabilidades"].get("CONCEDE") == 1.0 and "SOBRESEE" not in e["probabilidades"], "parcial no fue a CONCEDE")

# el circuito con base corta se apoya en lo nacional, y lo dice
loc = [pt(f"c{i}", 0.8, prospera="si") for i in range(4)]
nac = loc + [pt(f"x{i}", 0.8, prospera="no", circ="1") for i in range(96)]
e = jv.agregar(loc, nac, circuito="22")
ok(e["prestado_de_nacional"] and 0.2 < e["prospera"].get("si", 0) < 0.5,
   f"el encogimiento no movió el 100% local hacia lo nacional ({e['prospera']})")
ok("se completa con la tendencia nacional" in e["aviso"], "no avisó del préstamo nacional")
ok(e["confianza"] == "baja", "4 sentencias propias no son confianza baja")
ok(set(e["por_circuito"]) == {"22", "1"}, "por_circuito no sale de la base nacional")

# intervalo de Wilson
iv = jv.wilson(30, 40)
ok(iv and 0.58 < iv[0] < 0.62 and 0.84 < iv[1] < 0.88, f"Wilson 30/40 raro: {iv}")
e = jv.agregar([pt(f"w{i}", 0.8, prospera="si" if i < 30 else "no") for i in range(40)], None)
ok(e["confianza"] == "alta" and e["intervalo_95"], "40 sentencias claras no dieron confianza alta")

# periodo de la nueva integración y quejoso
e = jv.agregar([pt(f"r{i}", 0.8, prospera="no", periodo="nueva_integracion" if i < 5 else "anterior",
                   j_quejoso="gana") for i in range(13)], None)
ok(e["por_periodo"].get("nueva_integracion", {}).get("n") == 5, "no separa la nueva integración")
ok(e["quejoso"].get("gana") == 1.0 and e["quejoso_n"] == 13, "no informa cómo quedó el quejoso")
ok("no prospera" in jv.frase(e), "la frase no habla desde quien promueve")

# 5 · el endpoint: ni `acto_reclamado_tipo` ni la materia cruda como filtro; la estadística es la de v2
src = io.open("main.py", encoding="utf-8").read()
i = src.find("async def jurimetria_endpoint(")
cuerpo = src[i:src.find("\n\n\n# ====", i)]
ok('key="acto_reclamado_tipo"' not in cuerpo, "el endpoint sigue filtrando por acto_reclamado_tipo (sin índice)")
ok('key="materia"' not in cuerpo, "el endpoint sigue filtrando por la materia cruda")
ok("_jc.materia_de_consulta(materia)" in cuerpo and 'key="j_materia"' in cuerpo, "la materia no pasa por el canon")
ok("_jv2.agregar(" in cuerpo and "_aggregate_jurimetria" not in src, "la agregación vieja sigue viva")
ok("HTTPException(500" not in cuerpo, "un fallo de Qdrant sigue saliendo como 500 con el error crudo")
ok(cuerpo.count("permitir_ocr=_ocr_jur") == 2, "se perdió el freno del OCR en los adjuntos")

if fallos:
    print("FALLOS:")
    for f in fallos:
        print("  ·", f)
    sys.exit(1)
print("OK · jurimetría v2: sin base no hay sentido, una sentencia un voto, por tipo, encogimiento, Wilson, filtros")
