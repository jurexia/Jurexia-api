import io, os, json, collections, time
os.chdir("/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/jurexia-api-git")
for l in io.open(".env",encoding="utf8",errors="ignore"):
    if "=" in l and not l.strip().startswith("#"):
        k,v=l.split("=",1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
from qdrant_client import QdrantClient
S="/private/tmp/claude-501/-Users-josedavidalcantarmendoza-Documents-Viaje-a-Europa/37f7b791-1b1a-4b02-93eb-f9b5fdcf0854/scratchpad"
q=QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"], timeout=300)
t0=time.time(); off=None; leidos=0
c22=[]
while True:
    pts,off=q.scroll(collection_name="sentencias_holdings", limit=2000,
                     with_payload=True, with_vectors=False, offset=off)
    if not pts: break
    leidos+=len(pts)
    for p in pts:
        d=p.payload or {}
        if d.get("tipo_punto")!="holding": continue
        if str(d.get("circuito")) != "22": continue
        c22.append({k: d.get(k) for k in
                    ("expediente","tribunal","materia","tipo_asunto","sentido",
                     "tema_juridico","calidad_argumentativa_v2","agravios_resueltos",
                     "cuestiones_juridicas","principios_juridicos","holding")})
    if leidos % 20000 == 0:
        print(f"  {leidos:,} leídos · {len(c22):,} del 22 · {time.time()-t0:.0f}s", flush=True)
    if not off: break
io.open(f"{S}/c22.json","w",encoding="utf8").write(json.dumps(c22, ensure_ascii=False))
print(f"\n  ✓ {leidos:,} holdings leídos · {len(c22):,} del circuito 22 · {time.time()-t0:.0f}s")
print(f"  guardado en c22.json")
