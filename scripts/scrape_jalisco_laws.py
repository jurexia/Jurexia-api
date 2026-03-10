"""
Re-scrape Jalisco catalog with correct URL resolution.
The page URL is: https://congresoweb.congresojal.gob.mx/BibliotecaVirtual/busquedasleyes/Listado'2.cfm
Relative links are relative to this base, so urljoin handles them correctly.
"""
import re, json
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote, unquote

PAGE_URL = "https://congresoweb.congresojal.gob.mx/BibliotecaVirtual/busquedasleyes/Listado'2.cfm"

def main():
    print(f"Fetching {PAGE_URL} ...")
    r = httpx.get(PAGE_URL, timeout=30, follow_redirects=True,
                  headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                  verify=False)
    r.raise_for_status()
    
    soup = BeautifulSoup(r.text, "html.parser")
    
    all_laws = []
    current_section = None
    
    rows = soup.find_all('tr')
    print(f"Found {len(rows)} table rows")
    
    for row in rows:
        # Check section headers (td with colspan)
        td_colspan = row.find('td', colspan=True)
        if td_colspan:
            text = td_colspan.get_text(strip=True).lower()
            if 'constituc' in text:
                current_section = 'constitucion'
            elif 'digo' in text:
                current_section = 'codigo'
            elif 'ley' in text and 'ingreso' not in text:
                current_section = 'ley'
            elif 'ingreso' in text:
                current_section = 'ingresos'
            elif 'reglamento' in text:
                current_section = 'reglamentos'
            continue
        
        if current_section not in ('constitucion', 'codigo', 'ley'):
            continue
        
        tds = row.find_all('td')
        if len(tds) < 2:
            continue
        
        name = tds[0].get_text(strip=True)
        if not name or len(name) < 5:
            continue
        if name.lower() in ('#', 'nombre', 'no.', 'núm', 'num'):
            continue
        
        # Clean leading numbers
        name = re.sub(r'^\d+[\.\)\-\s]+', '', name).strip()
        
        # Find PDF link using urljoin for correct resolution
        pdf_url = ""
        for td in tds:
            pdf_link = td.find('a', href=re.compile(r'\.pdf', re.I))
            if pdf_link:
                href = pdf_link.get('href', '')
                # Use urljoin to correctly resolve relative URLs
                pdf_url = urljoin(PAGE_URL, href)
                break
        
        if name and pdf_url:
            all_laws.append({
                "name": name,
                "url": pdf_url,
                "cat": current_section,
            })
    
    # Save
    out = r"c:\Users\jdmju\.gemini\antigravity\playground\obsidian-expanse\jurexia-api-git\scripts\jalisco_laws_catalog.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_laws, f, ensure_ascii=False, indent=2)
    
    cats = {}
    for law in all_laws:
        cats[law['cat']] = cats.get(law['cat'], 0) + 1
    
    print(f"\n📄 Total: {len(all_laws)} entries saved")
    for c, n in sorted(cats.items()):
        print(f"   {c}: {n}")
    
    # Test first URL
    test_url = all_laws[0]["url"]
    print(f"\n🔍 Testing: {all_laws[0]['name'][:50]}")
    print(f"   URL: {test_url[:120]}")
    try:
        r2 = httpx.get(test_url, timeout=15, follow_redirects=True,
                       headers={"User-Agent": "Mozilla/5.0"},
                       verify=False)
        is_pdf = r2.content[:5] == b'%PDF-'
        print(f"   Status: {r2.status_code}, Size: {len(r2.content)}, Is PDF: {is_pdf}")
    except Exception as e:
        print(f"   ❌ {type(e).__name__}: {str(e)[:100]}")

if __name__ == "__main__":
    main()
