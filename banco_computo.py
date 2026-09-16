"""EL BANCO DEL CÓMPUTO — las sesiones reales contra las que se mide fase 0.

POR QUÉ EXISTE. `test_computo.py` abría `/tmp/audit_f0/sesiones.json`, un
volcado que alguien hizo a mano una vez. macOS vacía `/tmp` y el 16-sep-2026
la prueba dejó de correr con un FileNotFoundError: la comprobación que vigila
que ningún cambio mueva un plazo ya contado se apagó sola y en silencio. Una
prueba que desaparece sin avisar es peor que no tenerla, porque el verde de
las demás se lee como si estuviera.

Aquí se reconstruye desde `taller_sesiones`: son los encargos de verdad, con
su notificación, su presentación, su regla y sus inhábiles declarados. Se
cachea en el sitio de siempre para no pedirlos en cada corrida.
"""
import json
import os

RUTA = "/tmp/audit_f0/sesiones.json"


def _de_supabase():
    from dotenv import load_dotenv
    load_dotenv()
    from supabase import create_client
    cli = create_client(os.environ["SUPABASE_URL"],
                        os.environ["SUPABASE_SERVICE_KEY"])
    filas = cli.table("taller_sesiones").select("expediente,estado") \
               .limit(400).execute().data or []
    fuera = []
    for f in filas:
        e = ((f.get("estado") or {}).get("encargo")) or {}
        if not (e.get("notificacion") and e.get("presentacion")):
            continue
        fuera.append({
            "exp": f.get("expediente") or e.get("numero") or "",
            "notif": e["notificacion"], "pres": e["presentacion"],
            "regla": e.get("regla_surtimiento") or "personal",
            "plazo": e.get("plazo") or 15,
            "resp": e.get("responsable") or "",
            "extra": e.get("dias_inhabiles_extra") or [],
            "tipo": e.get("tipo_asunto") or "amparo_directo",
        })
    return fuera


def sesiones():
    """Las sesiones del banco. Del caché si está; de la base si no."""
    if os.path.exists(RUTA):
        try:
            return json.load(open(RUTA))
        except Exception:
            pass
    fuera = _de_supabase()
    os.makedirs(os.path.dirname(RUTA), exist_ok=True)
    json.dump(fuera, open(RUTA, "w"), ensure_ascii=False)
    print(f"   (banco reconstruido desde taller_sesiones: {len(fuera)} sesiones)")
    return fuera


if __name__ == "__main__":
    print(f"{len(sesiones())} sesiones en el banco")
