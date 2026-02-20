"""
Scraper de Juzgados de Distrito del CJF
Fuente: https://www.cjf.gob.mx/directorios/ojintcirc.aspx?cir=X

Extrae denominación, dirección, teléfonos y materia de todos los
Juzgados de Distrito de los 32 circuitos judiciales de México.
Inserta los resultados en la tabla `juzgados_distrito` de Supabase.
"""

import re
import os
import time
import json
import requests
from bs4 import BeautifulSoup

# ─── Supabase config ────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# ─── Circuito → Estado mapping ──────────────────────────────────────
CIRCUITO_ESTADO = {
    1: "Ciudad de México",
    2: "Estado de México",
    3: "Jalisco",
    4: "Nuevo León",
    5: "Sonora",
    6: "Puebla",
    7: "Veracruz",
    8: "Coahuila",
    9: "San Luis Potosí",
    10: "Tabasco",
    11: "Michoacán",
    12: "Sinaloa",
    13: "Oaxaca",
    14: "Chihuahua",
    15: "Baja California",
    16: "Guanajuato",
    17: "Chiapas",
    18: "Morelos",
    19: "Tamaulipas",
    20: "Chiapas",       # 20 is also Chiapas (Tuxtla)
    21: "Guerrero",
    22: "Querétaro",
    23: "Zacatecas",
    24: "Nayarit",
    25: "Durango",
    26: "Baja California Sur",
    27: "Quintana Roo",
    28: "Tlaxcala",
    29: "Hidalgo",
    30: "Aguascalientes",
    31: "Campeche",
    32: "Colima",
}

CJF_BASE = "https://www.cjf.gob.mx/directorios/ojintcirc.aspx?cir={}"

def detect_materia(name: str) -> str:
    """Detect the materia (subject) from the court name."""
    name_upper = name.upper()
    if "ADMINISTRATIVA" in name_upper:
        return "Administrativa"
    elif "PENAL" in name_upper:
        return "Penal"
    elif "CIVIL" in name_upper:
        return "Civil"
    elif "TRABAJO" in name_upper or "LABORAL" in name_upper:
        return "Trabajo"
    elif "AMPARO" in name_upper:
        return "Amparo"
    elif "MERCANTIL" in name_upper:
        return "Mercantil"
    elif "PROCESOS PENALES FEDERALES" in name_upper:
        return "Procesos Penales Federales"
    elif "EJECUCIÓN" in name_upper or "EJECUCION" in name_upper:
        return "Ejecución"
    else:
        return "Mixto"

def detect_ciudad(name: str, estado: str) -> str:
    """Extract city from the court name."""
    # Pattern: "EN LA CIUDAD DE MÉXICO", "EN ZAPOPAN", "EN MONTERREY", etc.
    match = re.search(r'(?:EN|DE)\s+(?:LA\s+CIUDAD\s+DE\s+)?([A-ZÁÉÍÓÚÑÜ][A-ZÁÉÍÓÚÑÜ\s,]+?)(?:\)|$)', name.upper())
    if match:
        city = match.group(1).strip().rstrip(',')
        # Title case
        return city.title()
    return estado

def scrape_circuit(cir: int, estado: str) -> list:
    """Scrape all Juzgados de Distrito from a single circuit."""
    url = CJF_BASE.format(cir)
    print(f"\n📡 Circuito {cir} ({estado}) — {url}")

    try:
        resp = requests.get(url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; JurexiaScraper/1.0)'
        })
        resp.encoding = 'utf-8'
        if resp.status_code != 200:
            print(f"   ❌ HTTP {resp.status_code}")
            return []
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
        return []

    soup = BeautifulSoup(resp.text, 'html.parser')

    courts = []
    current = None

    # Iterate through all alert-oj* elements in order
    elements = soup.select('.alert-ojarea, .alert-ojubica, .alert-ojregistro')

    for el in elements:
        classes = el.get('class', [])
        text = el.get_text(strip=True)

        if 'alert-ojarea' in classes:
            # Check if previous was a juzgado — save it
            if current and 'JUZGADO' in current['denominacion'].upper() and 'DISTRITO' in current['denominacion'].upper():
                courts.append(current)

            current = {
                'denominacion': text,
                'materia': detect_materia(text),
                'circuito': cir,
                'estado': estado,
                'ciudad': detect_ciudad(text, estado),
                'direccion': '',
                'telefono': '',
                'titular': '',
                'cv_x': 0,
            }

        elif current:
            if 'alert-ojubica' in classes:
                current['direccion'] = text
            elif 'alert-ojregistro' in classes:
                # Extract phone numbers from the contact info
                # The contact section has multiple rows with Name | Cargo | Phone
                # We want to collect all phone numbers
                phone_matches = re.findall(r'\(?\s*\d{2,4}\s*\)?\s*\d{3,5}(?:\s*(?:EXT|Ext|ext)\.?\s*\d+)?', text)
                if phone_matches:
                    phones = ', '.join(phone_matches)
                    if current['telefono']:
                        current['telefono'] += '; ' + phones
                    else:
                        current['telefono'] = phones

    # Don't forget the last one
    if current and 'JUZGADO' in current['denominacion'].upper() and 'DISTRITO' in current['denominacion'].upper():
        courts.append(current)

    print(f"   ✅ Found {len(courts)} Juzgados de Distrito")
    return courts


def insert_to_supabase(courts: list):
    """Insert courts into Supabase."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("\n⚠️  No Supabase credentials — saving to JSON instead")
        with open('juzgados_distrito.json', 'w', encoding='utf-8') as f:
            json.dump(courts, f, ensure_ascii=False, indent=2)
        print(f"   Saved {len(courts)} records to juzgados_distrito.json")
        return

    headers = {
        'apikey': SUPABASE_SERVICE_KEY,
        'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
        'Content-Type': 'application/json',
        'Prefer': 'return=minimal',
    }

    # Insert in batches of 50
    batch_size = 50
    for i in range(0, len(courts), batch_size):
        batch = courts[i:i+batch_size]
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/juzgados_distrito",
            headers=headers,
            json=batch,
            timeout=30,
        )
        if resp.status_code in (200, 201):
            print(f"   ✅ Inserted batch {i//batch_size + 1} ({len(batch)} records)")
        else:
            print(f"   ❌ Batch {i//batch_size + 1} failed: {resp.status_code} — {resp.text[:200]}")


def main():
    print("=" * 60)
    print("🏛️  SCRAPER DE JUZGADOS DE DISTRITO — CJF")
    print("=" * 60)

    all_courts = []
    for cir in range(1, 33):
        estado = CIRCUITO_ESTADO.get(cir, f"Circuito {cir}")
        courts = scrape_circuit(cir, estado)
        all_courts.extend(courts)
        time.sleep(1)  # Be polite

    print(f"\n{'=' * 60}")
    print(f"📊 RESUMEN: {len(all_courts)} Juzgados de Distrito en 32 circuitos")
    print(f"{'=' * 60}")

    # Stats by estado
    from collections import Counter
    by_estado = Counter(c['estado'] for c in all_courts)
    for estado, count in sorted(by_estado.items()):
        print(f"   {estado}: {count}")

    # Stats by materia
    by_materia = Counter(c['materia'] for c in all_courts)
    print(f"\n   Por materia:")
    for materia, count in sorted(by_materia.items()):
        print(f"   {materia}: {count}")

    # Insert
    insert_to_supabase(all_courts)

    print(f"\n✅ Done!")


if __name__ == "__main__":
    main()
