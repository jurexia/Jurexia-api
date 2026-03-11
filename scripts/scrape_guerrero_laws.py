"""
Scrape Guerrero catalog from congresogro.gob.mx
Pages: constitución (direct PDF), codigos.php, leyes-ordinarias.php, leyes-organicas.php, leyes-ingresos-2025.php
Output: scripts/guerrero_laws_catalog.json
"""
import re, json
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE = "https://congresogro.gob.mx/legislacion/"
PAGES = {
    "codigo":    f"{BASE}codigos.php",
    "ley":       f"{BASE}leyes-ordinarias.php",
    "organica":  f"{BASE}leyes-organicas.php",
    "ingresos":  f"{BASE}leyes-ingresos-2025.php",
}

def scrape_page(url: str, cat: str) -> list[dict]:
    """Scrape a legislation page — extract PDF links with law names."""
    print(f"  Fetching {url} ...")
    r = httpx.get(url, timeout=30, follow_redirects=True, verify=False,
                  headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
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
        
        # Resolve relative URL
        full_url = urljoin(url, href)
        
        # Skip duplicates
        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)
        
        # Get name from link text or parent context
        name = link.get_text(strip=True)
        if not name or len(name) < 5:
            # Try parent row or list item
            parent = link.find_parent(['tr', 'li', 'div', 'p'])
            if parent:
                name = parent.get_text(strip=True)[:200]
        
        # Clean name
        if name:
            name = re.sub(r'\s+', ' ', name).strip()
            # Remove PDF extension from name if it's just the filename
            if name.lower().endswith('.pdf'):
                name = name[:-4]
            # Skip if it's just "Descargar" or too short
            if name.lower() in ('descargar', 'pdf', 'ver', 'download') or len(name) < 5:
                continue
        
        results.append({
            "name": name,
            "url": full_url,
            "cat": cat,
        })
    
    return results

def main():
    all_laws = []
    
    # 1. Constitución (direct PDF link)
    all_laws.append({
        "name": "Constitución Política del Estado Libre y Soberano de Guerrero",
        "url": f"{BASE}CONSTITUCION-GUERRERO-15-06-2022.pdf",
        "cat": "constitucion"
    })
    print("✅ Constitución: 1")
    
    # 2. Each category page
    for cat, url in PAGES.items():
        laws = scrape_page(url, cat)
        all_laws.extend(laws)
        print(f"✅ {cat}: {len(laws)}")
    
    # Save
    out = r"c:\Users\jdmju\.gemini\antigravity\playground\obsidian-expanse\jurexia-api-git\scripts\guerrero_laws_catalog.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_laws, f, ensure_ascii=False, indent=2)
    
    # Stats
    cats = {}
    for law in all_laws:
        cats[law['cat']] = cats.get(law['cat'], 0) + 1
    
    print(f"\n📄 Total: {len(all_laws)} entries")
    for c, n in sorted(cats.items()):
        print(f"   {c}: {n}")
    
    wpdf = sum(1 for x in all_laws if x['url'])
    print(f"   With PDF: {wpdf}")
    
    # Show first and last entries
    print("\n--- First 5 ---")
    for x in all_laws[:5]:
        print(f"  [{x['cat']}] {x['name'][:70]}")
    
    print("\n--- Last 5 ---")
    for x in all_laws[-5:]:
        print(f"  [{x['cat']}] {x['name'][:70]}")
    
    # Test one URL
    test = all_laws[1]  # First código
    print(f"\n🔍 Testing: {test['name'][:50]}")
    try:
        r = httpx.get(test['url'], timeout=15, follow_redirects=True, verify=False,
                      headers={"User-Agent": "Mozilla/5.0"})
        is_pdf = r.content[:5] == b'%PDF-'
        print(f"   Status: {r.status_code}, Size: {len(r.content)}, Is PDF: {is_pdf}")
    except Exception as e:
        print(f"   ❌ {type(e).__name__}: {str(e)[:100]}")

if __name__ == "__main__":
    main()
