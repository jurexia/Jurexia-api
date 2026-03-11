"""
Scrape Veracruz catalog from legisver.gob.mx
Leyes: Inicio.php?p=le (incl. Constitución)
Códigos: Inicio.php?p=co  
PDF links use javascript:PDF('path') — extract path and build full URL
"""
import re, json
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE = "https://www.legisver.gob.mx/"
PAGES = {
    "ley":    f"{BASE}Inicio.php?p=le",
    "codigo": f"{BASE}Inicio.php?p=co",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def scrape_page(url: str, cat: str) -> list[dict]:
    """Scrape legislation from legisver.gob.mx — extract javascript:PDF() links."""
    print(f"  Fetching {url} ...")
    r = httpx.get(url, timeout=30, follow_redirects=True, verify=False, headers=HEADERS)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    seen_urls = set()

    # Find all links with javascript:PDF('...')
    js_links = soup.find_all('a', href=re.compile(r"javascript:\s*PDF\s*\(", re.I))
    print(f"  Found {len(js_links)} javascript:PDF() links")

    for link in js_links:
        href = link.get('href', '')
        # Extract path from javascript:PDF('...')
        m = re.search(r"PDF\s*\(\s*['\"]([^'\"]+)['\"]", href)
        if not m:
            continue
        
        pdf_path = m.group(1)
        full_url = urljoin(url, pdf_path)

        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        # Get name from link text
        name = link.get_text(strip=True)
        if not name or len(name) < 5:
            # Try parent element
            parent = link.find_parent(['li', 'tr', 'div'])
            if parent:
                name = parent.get_text(strip=True)
        
        if not name or len(name) < 5:
            continue

        # Clean name
        name = re.sub(r'\s+', ' ', name).strip()

        # Detect constitution
        actual_cat = cat
        name_u = name.upper()
        if "CONSTITUCIÓN POLÍTICA" in name_u and "ESTADO" in name_u and "VERACRUZ" in name_u:
            actual_cat = "constitucion"

        results.append({
            "name": name,
            "url": full_url,
            "cat": actual_cat,
        })

    # Also look for regular <a href="*.pdf"> links
    pdf_links = soup.find_all('a', href=re.compile(r'\.pdf', re.I))
    for link in pdf_links:
        href = link.get('href', '')
        if 'javascript' in href.lower():
            continue
        full_url = urljoin(url, href)
        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)
        
        name = link.get_text(strip=True)
        if not name or len(name) < 5:
            continue
        name = re.sub(r'\s+', ' ', name).strip()
        
        actual_cat = cat
        name_u = name.upper()
        if "CONSTITUCIÓN POLÍTICA" in name_u and "ESTADO" in name_u:
            actual_cat = "constitucion"
        
        results.append({"name": name, "url": full_url, "cat": actual_cat})

    return results


def main():
    all_laws = []

    for cat, url in PAGES.items():
        laws = scrape_page(url, cat)
        all_laws.extend(laws)
        print(f"  ✅ {cat}: {len(laws)}")

    # Deduplicate
    seen = set()
    unique = []
    for law in all_laws:
        key = law["url"]
        if key not in seen:
            seen.add(key)
            unique.append(law)

    # Save
    out = r"c:\Users\jdmju\.gemini\antigravity\playground\obsidian-expanse\jurexia-api-git\scripts\veracruz_laws_catalog.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)

    # Stats
    cats = {}
    for law in unique:
        cats[law['cat']] = cats.get(law['cat'], 0) + 1

    print(f"\n📄 Total: {len(unique)} entries")
    for c, n in sorted(cats.items()):
        print(f"   {c}: {n}")

    # Show samples
    print("\n--- Constitución ---")
    for x in unique:
        if x['cat'] == 'constitucion':
            print(f"  {x['name'][:80]}")
            print(f"    {x['url']}")

    print("\n--- First 5 leyes ---")
    count = 0
    for x in unique:
        if x['cat'] == 'ley' and count < 5:
            print(f"  {x['name'][:80]}")
            count += 1

    print("\n--- First 5 códigos ---")
    count = 0
    for x in unique:
        if x['cat'] == 'codigo' and count < 5:
            print(f"  {x['name'][:80]}")
            count += 1

    # Test one URL
    if unique:
        test = unique[0]
        print(f"\n🔍 Test: {test['name'][:50]}")
        try:
            tr = httpx.get(test['url'], timeout=15, follow_redirects=True, verify=False, headers=HEADERS)
            is_pdf = tr.content[:5] == b'%PDF-'
            print(f"   Status: {tr.status_code}, Size: {len(tr.content)}, PDF: {is_pdf}")
        except Exception as e:
            print(f"   ❌ {type(e).__name__}: {str(e)[:100]}")


if __name__ == "__main__":
    main()
