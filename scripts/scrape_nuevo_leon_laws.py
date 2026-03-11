"""
Scrape Nuevo León catalog from hcnl.gob.mx
Pages: index.php (leyes + constitución), codigos.php, paquete_fiscal.php
Output: scripts/nuevo_leon_laws_catalog.json
"""
import re, json
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE = "https://www.hcnl.gob.mx/trabajo_legislativo/leyes/"

PAGES = {
    "ley":        f"{BASE}index.php",
    "codigo":     f"{BASE}codigos.php",
    "fiscal":     f"{BASE}paquete_fiscal.php",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def scrape_page(url: str, cat: str) -> list[dict]:
    """Scrape a legislation page from hcnl.gob.mx — extract PDF links."""
    print(f"  Fetching {url} ...")
    r = httpx.get(url, timeout=30, follow_redirects=True, verify=False, headers=HEADERS)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    seen_urls = set()

    # Find all PDF links
    pdf_links = soup.find_all('a', href=re.compile(r'\.pdf', re.I))

    for link in pdf_links:
        href = link.get('href', '')
        if not href:
            continue

        full_url = urljoin(url, href)

        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        # Get name from link text or parent tr
        name = link.get_text(strip=True)
        if not name or len(name) < 5:
            parent = link.find_parent('tr')
            if parent:
                tds = parent.find_all('td')
                if tds:
                    name = tds[0].get_text(strip=True)
        
        # If still no good name, try extracting from URL
        if not name or len(name) < 5:
            # Get filename from URL, decode %20, remove extension
            from urllib.parse import unquote
            fname = unquote(full_url.split('/')[-1].split('?')[0])
            if fname.lower().endswith('.pdf'):
                fname = fname[:-4]
            name = fname.replace('-', ' ').replace('_', ' ')

        # Clean name
        name = re.sub(r'\s+', ' ', name).strip()
        if name.lower() in ('descargar', 'pdf', 'ver', 'download', '') or len(name) < 5:
            continue

        # Determine if it's the constitution
        actual_cat = cat
        if "CONSTITUCI" in name.upper() and "ESTADO" in name.upper() and "NUEVO" in name.upper():
            actual_cat = "constitucion"

        results.append({
            "name": name,
            "url": full_url,
            "cat": actual_cat,
        })

    return results


def main():
    all_laws = []

    for cat, url in PAGES.items():
        laws = scrape_page(url, cat)
        all_laws.extend(laws)
        print(f"✅ {cat}: {len(laws)}")

    # Deduplicate
    seen = set()
    unique = []
    for law in all_laws:
        key = law["url"].split("?")[0]  # Remove date param for dedup
        if key not in seen:
            seen.add(key)
            unique.append(law)

    # Save
    out = r"c:\Users\jdmju\.gemini\antigravity\playground\obsidian-expanse\jurexia-api-git\scripts\nuevo_leon_laws_catalog.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)

    # Stats
    cats = {}
    for law in unique:
        cats[law['cat']] = cats.get(law['cat'], 0) + 1

    print(f"\n📄 Total: {len(unique)} entries (from {len(all_laws)} raw)")
    for c, n in sorted(cats.items()):
        print(f"   {c}: {n}")

    # Show first and last
    print("\n--- First 5 ---")
    for x in unique[:5]:
        print(f"  [{x['cat']}] {x['name'][:70]}")
    print("\n--- Last 5 ---")
    for x in unique[-5:]:
        print(f"  [{x['cat']}] {x['name'][:70]}")

    # Test one URL
    if unique:
        test = unique[0]
        print(f"\n🔍 Test: {test['name'][:50]}")
        try:
            tr = httpx.get(test['url'], timeout=15, follow_redirects=True, verify=False, headers=HEADERS)
            is_pdf = tr.content[:5] == b'%PDF-'
            print(f"   Status: {tr.status_code}, Size: {len(tr.content)}, Is PDF: {is_pdf}")
        except Exception as e:
            print(f"   ❌ {type(e).__name__}: {str(e)[:100]}")


if __name__ == "__main__":
    main()
