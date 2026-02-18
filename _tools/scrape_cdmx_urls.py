#!/usr/bin/env python3
"""
scrape_cdmx_urls.py — Scrape current PDF URLs from the CDMX legal portal.

Fetches all pages from:
  - https://data.consejeria.cdmx.gob.mx/index.php/leyes/leyes
  - https://data.consejeria.cdmx.gob.mx/index.php/leyes/codigos
  - https://data.consejeria.cdmx.gob.mx/index.php/leyes/reglamentos
  - https://data.consejeria.cdmx.gob.mx/index.php/leyes/constitucion

Handles pagination and extracts all <a href="...pdf"> links.
Outputs: cdmx_urls_current.json
"""

import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

BASE = "https://data.consejeria.cdmx.gob.mx"

SECTIONS = {
    "constitucion": f"{BASE}/index.php/leyes/constitucion",
    "leyes": f"{BASE}/index.php/leyes/leyes",
    "codigos": f"{BASE}/index.php/leyes/codigos",
    "reglamentos": f"{BASE}/index.php/leyes/reglamentos",
}

# Items per page on the CDMX portal (observed default)
PAGE_SIZE = 27  # The portal shows ~27 items per page for leyes


def scrape_section(client: httpx.Client, section_name: str, base_url: str) -> list[dict]:
    """Scrape all pages of a section, extracting PDF links."""
    all_entries = []
    start = 0
    page_num = 1

    while True:
        url = f"{base_url}?start={start}" if start > 0 else base_url
        print(f"  📄 {section_name} page {page_num} (start={start})...")

        try:
            resp = client.get(url, timeout=30)
            if resp.status_code != 200:
                print(f"     ❌ HTTP {resp.status_code}")
                break
        except Exception as e:
            print(f"     ❌ Error: {e}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        # Find all PDF links
        pdf_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".pdf"):
                full_url = urljoin(BASE, href)
                # Try to extract the law name from context
                # The link is usually an image inside a container with law name
                name = extract_law_name(a, soup)
                pdf_links.append({
                    "nombre": name,
                    "url": full_url,
                    "categoria": section_name,
                    "filename": href.split("/")[-1],
                })

        if not pdf_links:
            # Also look for links in <img> parents and onclick handlers
            for img in soup.find_all("img", src=True):
                src = img["src"]
                if src.lower().endswith(".pdf"):
                    full_url = urljoin(BASE, src)
                    parent_a = img.find_parent("a")
                    if parent_a and parent_a.get("href", "").lower().endswith(".pdf"):
                        full_url = urljoin(BASE, parent_a["href"])
                    name = extract_law_name(img, soup)
                    pdf_links.append({
                        "nombre": name,
                        "url": full_url,
                        "categoria": section_name,
                        "filename": full_url.split("/")[-1],
                    })

        new_count = 0
        for entry in pdf_links:
            # Deduplicate by URL
            if not any(e["url"] == entry["url"] for e in all_entries):
                all_entries.append(entry)
                new_count += 1

        print(f"     Found {len(pdf_links)} PDF links ({new_count} new)")

        # Check for next page
        has_next = False
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True).lower()
            if text in ("siguiente", "next", "›", "»"):
                href = a["href"]
                # Extract the start parameter
                match = re.search(r'start=(\d+)', href)
                if match:
                    next_start = int(match.group(1))
                    if next_start > start:
                        start = next_start
                        has_next = True
                        break

        if not has_next:
            break

        page_num += 1
        time.sleep(0.5)  # Rate limit

    return all_entries


def extract_law_name(element, soup) -> str:
    """Try to extract the law name from the element's context."""
    # Look for the nearest h2, h3, or strong element
    parent = element.parent
    for _ in range(5):
        if parent is None:
            break
        # Check for a heading
        heading = parent.find(["h2", "h3", "h4"])
        if heading:
            return heading.get_text(strip=True)
        # Check for strong/b
        strong = parent.find(["strong", "b"])
        if strong:
            text = strong.get_text(strip=True)
            if len(text) > 10:
                return text
        parent = parent.parent

    # Fallback: infer from filename
    href = element.get("href", "") or element.get("src", "")
    filename = href.split("/")[-1].replace(".pdf", "").replace("_", " ")
    return filename


def compare_with_existing(scraped: list[dict]) -> dict:
    """Compare scraped URLs against the current ingest_cdmx.py URLs."""
    # Load existing URLs from ingest_cdmx.py
    ingest_file = Path(__file__).parent / "ingest_cdmx.py"
    if not ingest_file.exists():
        return {"error": "ingest_cdmx.py not found"}

    content = ingest_file.read_text(encoding="utf-8")

    # Extract all URLs from the ingest script
    existing_urls = re.findall(r'https?://[^\s"\']+\.pdf', content)
    existing_filenames = {url.split("/")[-1]: url for url in existing_urls}

    # Build comparison
    scraped_filenames = {e["filename"]: e["url"] for e in scraped}

    # Find matches and differences
    matched = []
    updated = []
    only_in_portal = []
    only_in_script = []

    for sf, surl in scraped_filenames.items():
        # Check for exact match
        if sf in existing_filenames:
            if existing_filenames[sf] == surl:
                matched.append(sf)
            else:
                updated.append({"filename": sf, "old_url": existing_filenames[sf], "new_url": surl})
        else:
            # Check for base name match (version changed)
            base = re.sub(r'_[\d.]+\.pdf$', '', sf)
            found = False
            for ef in existing_filenames:
                ebase = re.sub(r'_[\d.]+\.pdf$', '', ef)
                if base == ebase:
                    updated.append({"filename": sf, "old_filename": ef,
                                    "old_url": existing_filenames[ef], "new_url": surl})
                    found = True
                    break
            if not found:
                only_in_portal.append({"filename": sf, "url": surl})

    for ef, eurl in existing_filenames.items():
        base = re.sub(r'_[\d.]+\.pdf$', '', ef)
        found = False
        for sf in scraped_filenames:
            sbase = re.sub(r'_[\d.]+\.pdf$', '', sf)
            if base == sbase or ef == sf:
                found = True
                break
        if not found:
            only_in_script.append({"filename": ef, "url": eurl})

    return {
        "matched": len(matched),
        "updated": updated,
        "only_in_portal": only_in_portal,
        "only_in_script": only_in_script,
    }


def main():
    print("═══════════════════════════════════════════════════════════════")
    print("  SCRAPING CDMX LEGAL PORTAL — Current PDF URLs")
    print("═══════════════════════════════════════════════════════════════\n")

    all_entries = []

    with httpx.Client(follow_redirects=True, verify=False) as client:
        for section_name, url in SECTIONS.items():
            print(f"\n🔍 Section: {section_name}")
            entries = scrape_section(client, section_name, url)
            all_entries.extend(entries)
            print(f"   Total for {section_name}: {len(entries)}")
            time.sleep(1)

    # Deduplicate by URL
    seen = set()
    unique = []
    for e in all_entries:
        if e["url"] not in seen:
            seen.add(e["url"])
            unique.append(e)

    print(f"\n═══════════════════════════════════════════════════════════════")
    print(f"  TOTAL UNIQUE PDF URLs: {len(unique)}")
    print(f"═══════════════════════════════════════════════════════════════\n")

    # Save to JSON
    output = Path(__file__).parent / "cdmx_urls_current.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved to {output}\n")

    # Compare with existing
    print("═══════════════════════════════════════════════════════════════")
    print("  COMPARING WITH ingest_cdmx.py")
    print("═══════════════════════════════════════════════════════════════\n")

    comparison = compare_with_existing(unique)

    if "error" in comparison:
        print(f"❌ {comparison['error']}")
        return

    print(f"  ✅ Exact matches: {comparison['matched']}")
    print(f"  🔄 Updated URLs: {len(comparison['updated'])}")
    print(f"  🆕 New (only in portal): {len(comparison['only_in_portal'])}")
    print(f"  ❓ Missing from portal: {len(comparison['only_in_script'])}")

    if comparison["updated"]:
        print(f"\n  Updated URLs (version changes):")
        for u in comparison["updated"][:10]:
            old_fn = u.get("old_filename", u["filename"])
            print(f"    {old_fn} → {u['filename']}")
            print(f"      NEW: {u['new_url']}")

    if comparison["only_in_portal"]:
        print(f"\n  New laws found on portal (not in script):")
        for p in comparison["only_in_portal"][:10]:
            print(f"    {p['filename']}")

    if comparison["only_in_script"]:
        print(f"\n  In script but NOT found on portal:")
        for s in comparison["only_in_script"][:10]:
            print(f"    {s['filename']}")

    # Save comparison
    comp_output = Path(__file__).parent / "cdmx_comparison.json"
    with open(comp_output, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Comparison saved to {comp_output}")


if __name__ == "__main__":
    main()
