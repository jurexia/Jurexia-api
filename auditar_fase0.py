"""¿La Fase 0 reproduce los cómputos que ya están firmados?

Lee el párrafo de oportunidad de los adelantos y engroses reales, le extrae las
fechas que el secretario escribió, y las compara con lo que calcula el módulo.
"""
import re, glob, sys, datetime as dt
import docx
sys.path.insert(0, "/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/jurexia-api-git")
import fase0_oportunidad as f0

DIA = {p: i for i, p in enumerate(f0._UNIDADES) if p}
MES = {m: i for i, m in enumerate(f0._MESES) if m}
ANIOS = {f0._anio_en_letra(y): y for y in range(2015, 2031)}
RX_DIA = "|".join(sorted(DIA, key=len, reverse=True))
RX_MES = "|".join(MES)
RX_ANIO = "|".join(sorted(ANIOS, key=len, reverse=True))
# día de mes [de año]  — el año puede faltar y se hereda
RX = re.compile(rf"\b({RX_DIA})\s+de\s+({RX_MES})(?:\s+de\s+({RX_ANIO}))?", re.I)

def fechas(txt, anio_defecto=None):
    out = []
    for m in RX.finditer(txt):
        d, mes, a = DIA[m.group(1).lower()], MES[m.group(2).lower()], m.group(3)
        anio = ANIOS[a.lower()] if a else anio_defecto
        if anio:
            try: out.append(dt.date(anio, mes, d))
            except ValueError: pass
    return out

def una(txt, anio=None):
    f = fechas(txt, anio); return f[0] if f else None

def analizar(p):
    todas = fechas(p)
    anio = todas[0].year if todas else None
    tr = {}
    if "se notificó" in p:
        tr["notif"] = una(p.split("se notificó")[1].split("por lo que")[0], anio)
    if "es decir," in p:
        tr["surtio"] = una(p.split("es decir,")[1][:90], anio)
    if "fue del" in p:
        resto = p.split("fue del")[1]
        # «del X de MES al Y de MES de AÑO» — el año va al final y se hereda
        cola = resto.split(" al ", 1)
        if len(cola) == 2:
            anio_fin = (fechas(cola[1][:90]) or [None])[0]
            anio2 = anio_fin.year if anio_fin else anio
            tr["inicio"] = una(cola[0][:80], anio2)
            tr["vence"] = anio_fin
    if "se presentó el" in p:
        tr["pres"] = una(p.split("se presentó el")[1][:90], anio)
    return tr

archivos = [a for a in sorted(glob.glob("/Volumes/KINGSTON/**/*.docx", recursive=True)) if "~$" not in a]
casos = []
for ruta in archivos:
    if len(casos) >= 60: break
    try: ps = [x.text.strip() for x in docx.Document(ruta).paragraphs]
    except Exception: continue
    for p in ps:
        if "surtió efectos" in p and "el plazo" in p:
            casos.append((ruta.split("/")[-1], p)); break

print(f"engroses con párrafo de oportunidad: {len(casos)}\n")
ok = mal = incompleto = 0
detalles = []
for nombre, p in casos:
    tr = analizar(p)
    if not all(tr.get(k) for k in ("notif", "inicio", "vence")):
        incompleto += 1; continue
    dias_nat = (tr["vence"] - tr["inicio"]).days
    plazo = 15 if dias_nat > 16 else 10
    regla = ("tja_qro_boletin" if re.search(r"bolet[íi]n", p, re.I)
             else "lista" if "por lista" in p else "personal")
    c = f0.computar(tr["notif"], tr.get("pres"), regla, plazo)
    coincide = c.inicio == tr["inicio"] and c.vencimiento == tr["vence"]
    ok += coincide; mal += (not coincide)
    if not coincide:
        detalles.append((nombre, tr, c, plazo, regla))

print(f"  reproduce exacto : {ok}")
print(f"  discrepa         : {mal}")
print(f"  no legible       : {incompleto}")
if ok + mal: print(f"  concordancia     : {ok/(ok+mal):.0%}\n")
import collections
por_anio = collections.Counter(tr["inicio"].year for _, tr, _, _, _ in detalles)
por_mes  = collections.Counter(f"{tr['inicio'].year}-{tr['inicio'].month:02d}" for _, tr, _, _, _ in detalles)
desfase  = collections.Counter((tr["vence"] - c.vencimiento).days for _, tr, c, _, _ in detalles)
print("  discrepancias por AÑO del plazo:")
for a, n in sorted(por_anio.items()): print(f"     {a}  {n:3d}  {'#'*n}")
print("\n  meses donde más falla (top 8):")
for m, n in por_mes.most_common(8): print(f"     {m}  {n}")
print("\n  desfase del vencimiento (días naturales, firmado − calculado):")
for d, n in sorted(desfase.items()): print(f"     {d:+4d} días  {n:3d} casos")
