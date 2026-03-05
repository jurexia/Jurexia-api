/**
 * extract.js — PDF → Clean TXT for Genio Mercantil corpus
 * Uses pdfjs-dist with Y-position line reconstruction and aggressive cleaning
 */

const fs = require('fs');
const path = require('path');

const LAWS = [
    { pdf: 'CCom.pdf', output: '01_codigo_comercio.txt', title: 'CÓDIGO DE COMERCIO' },
    { pdf: 'LGTOC.pdf', output: '02_ley_titulos_credito.txt', title: 'LEY GENERAL DE TÍTULOS Y OPERACIONES DE CRÉDITO' },
    { pdf: 'LGSM.pdf', output: '03_ley_sociedades_mercantiles.txt', title: 'LEY GENERAL DE SOCIEDADES MERCANTILES' },
    { pdf: '211.pdf', output: '04_ley_contrato_seguro.txt', title: 'LEY SOBRE EL CONTRATO DE SEGURO' },
];

// ── Extract text with proper line breaks ──────────────────────────────
async function extractPdfText(pdfPath) {
    const pdfjsLib = await import('pdfjs-dist/legacy/build/pdf.mjs');
    const data = new Uint8Array(fs.readFileSync(pdfPath));
    const doc = await pdfjsLib.getDocument({ data }).promise;

    let fullText = '';
    for (let i = 1; i <= doc.numPages; i++) {
        const page = await doc.getPage(i);
        const content = await page.getTextContent();

        let lastY = null;
        let lineText = '';
        for (const item of content.items) {
            if (item.str === undefined) continue;
            const y = item.transform ? item.transform[5] : null;
            if (lastY !== null && y !== null && Math.abs(y - lastY) > 2) {
                fullText += lineText.trim() + '\n';
                lineText = item.str;
            } else {
                if (lineText && !lineText.endsWith(' ') && !item.str.startsWith(' ')) lineText += ' ';
                lineText += item.str;
            }
            lastY = y;
        }
        if (lineText.trim()) fullText += lineText.trim() + '\n';
        fullText += '\n';
    }
    return { text: fullText, numPages: doc.numPages };
}

// ── Should this line be removed? ──────────────────────────────────────
function shouldRemoveLine(s) {
    // Page numbers (standalone)
    if (/^\d{1,4}$/.test(s)) return true;
    if (/^\d{1,3}\s+de\s+\d{1,3}$/.test(s)) return true;

    // Repeating page headers (PDF has these on every page)
    if (/^C\s*[ÁA]\s*MARA\s+DE\s+D\s*IPUTADOS/i.test(s)) return true;
    if (/^H\.?\s*C\s*ONGRESO\s+DE\s+LA\s+U\s*NI[ÓO]N/i.test(s)) return true;
    if (/^Secretar[ií]a\s+(General|de\s+Servicios)/i.test(s)) return true;
    if (/^Direcci[óo]n\s+General\s+de\s+Servicios/i.test(s)) return true;
    if (/^[ÚU]ltima\s+[Rr]eforma\s+(DOF|publicada)/i.test(s)) return true;
    if (/^Nueva\s+Ley\s+publicada/i.test(s)) return true;
    if (/^Ley\s+publicada\s+en\s+el/i.test(s)) return true;
    if (/^Diario\s+Oficial\s+de\s+la\s+Federaci[oó]n/i.test(s)) return true;
    if (/^TEXTO\s+VIGENTE$/i.test(s)) return true;
    if (/^Cantidades\s+actualizadas\s+por/i.test(s)) return true;

    // Repeating law title at top of every page
    if (/^C[ÓO]DIGO\s+DE\s+COMERCIO$/i.test(s)) return true;
    if (/^LEY\s+GENERAL\s+DE\s+T[ÍI]TULOS\s+Y\s+OPERACIONES\s+DE\s+CR[ÉE]DITO$/i.test(s)) return true;
    if (/^LEY\s+GENERAL\s+DE\s+SOCIEDADES\s+MERCANTILES$/i.test(s)) return true;
    if (/^LEY\s+SOBRE\s+EL\s+CONTRATO\s+DE\s+SEGURO$/i.test(s)) return true;

    // Full-line reform/addition/derogation annotations
    if (/^\(?(Reformad[oa]|Adicionad[oa]|Derogad[oa]),?\s.*D\.?\s*O\.?\s*F\.?/i.test(s)) return true;
    if (/^Art[íi]culo\s+(reformado|adicionado|derogado)\s+DOF/i.test(s)) return true;
    if (/^Fracci[oó]n\s+(reformada|adicionada|derogada|recorrida)\s+DOF/i.test(s)) return true;
    if (/^P[áa]rrafo\s+(reformado|adicionado|derogado)\s+DOF/i.test(s)) return true;
    if (/^(Inciso|Secci[óo]n|Cap[íi]tulo|T[íi]tulo|Libro|Denominaci[oó]n|Disposici[oó]n|N[oó]ta|Nota)\s+(reformad|adicionad|derogad|derogatoria)/i.test(s)) return true;
    if (/^Nuevo\s+C[oó]digo\s+publicado/i.test(s)) return true;

    // Historical preface text
    if (/^El\s+Presidente\s+de\s+la\s+Rep[úu]blica\s+se\s+ha\s+servido/i.test(s)) return true;
    if (/^PORFIRIO\s+DIAZ/i.test(s)) return true;
    if (/^Que\s+en\s+virtud\s+de\s+la\s+autorizaci[oó]n\s+concedida/i.test(s)) return true;
    if (/^he\s+tenido\s+[aá]\s+bien\s+expedir/i.test(s)) return true;
    if (/^sabed\s*:$/i.test(s)) return true;

    // Derogated articles (Se deroga) — entire line
    if (/^Art[íi]culo\s+\d+[^.]*[-–—.]?\s*\(?\s*Se\s+deroga\s*\)?\s*\.?\s*$/i.test(s)) return true;

    return false;
}

// ── Cleaning function ─────────────────────────────────────────────────
function cleanLawText(rawText, title) {
    const lines = rawText.split('\n');
    const cleanedLines = [];
    let inTransitorios = false;

    for (let i = 0; i < lines.length; i++) {
        const stripped = lines[i].trim();

        if (!stripped) {
            if (cleanedLines.length > 0 && cleanedLines[cleanedLines.length - 1].trim() !== '') {
                cleanedLines.push('');
            }
            continue;
        }

        // TRANSITORIOS: stop processing
        if (/^(TRANSITORIOS?|ART[IÍ]CULOS?\s+TRANSITORIOS?)$/i.test(stripped)) {
            inTransitorios = true;
            continue;
        }
        if (inTransitorios) {
            if (/^(LIBRO\s+\w|T[IÍ]TULO\s+\w|CAP[IÍ]TULO\s+\w)/i.test(stripped) &&
                !/TRANSITORIO/i.test(stripped)) {
                inTransitorios = false;
            } else {
                continue;
            }
        }

        // Remove noise lines
        if (shouldRemoveLine(stripped)) continue;

        // Strip inline reform annotations
        let cleanedLine = stripped.replace(
            /\s*\((Reformad[oa]|Adicionad[oa]|Derogad[oa]|Fe de erratas|Nota del editor)[^)]*\)/gi, ''
        );
        if (!cleanedLine.trim()) continue;

        cleanedLines.push(cleanedLine);
    }

    // Collapse multiple blank lines
    const result = [];
    let prevBlank = false;
    for (const line of cleanedLines) {
        if (!line.trim()) {
            if (!prevBlank) { result.push(''); prevBlank = true; }
        } else {
            result.push(line);
            prevBlank = false;
        }
    }

    return `${'='.repeat(60)}\n${title}\n${'='.repeat(60)}\n` + result.join('\n').trim();
}

// ── Main ──────────────────────────────────────────────────────────────
async function main() {
    console.log('='.repeat(60));
    console.log('  IUREXIA — Mercantil Corpus Extractor');
    console.log('='.repeat(60));

    const dir = __dirname;
    let totalChars = 0;

    for (const law of LAWS) {
        const pdfPath = path.join(dir, law.pdf);
        if (!fs.existsSync(pdfPath)) { console.log(`\nX Not found: ${law.pdf}`); continue; }

        console.log(`\n${'─'.repeat(50)}`);
        console.log(`> ${law.title}`);

        const { text: rawText, numPages } = await extractPdfText(pdfPath);
        console.log(`  Raw: ${rawText.length.toLocaleString()} chars, ${numPages} pages`);

        const cleaned = cleanLawText(rawText, law.title);
        const tokensEst = Math.floor(cleaned.length / 4);
        console.log(`  Clean: ${cleaned.length.toLocaleString()} chars (~${tokensEst.toLocaleString()} tokens)`);
        console.log(`  Reduction: ${((1 - cleaned.length / rawText.length) * 100).toFixed(1)}%`);

        fs.writeFileSync(path.join(dir, law.output), cleaned, 'utf-8');
        console.log(`  OK -> ${law.output}`);
        totalChars += cleaned.length;
    }

    console.log(`\n${'='.repeat(60)}`);
    console.log('  SUMMARY');
    const txtFiles = fs.readdirSync(dir).filter(f => /^\d/.test(f) && f.endsWith('.txt')).sort();
    for (const f of txtFiles) {
        const content = fs.readFileSync(path.join(dir, f), 'utf-8');
        console.log(`  ${f}: ${content.length.toLocaleString()} chars (~${Math.floor(content.length / 4).toLocaleString()} tokens)`);
    }
    const totalTokens = Math.floor(totalChars / 4);
    console.log(`\n  Total: ${totalChars.toLocaleString()} chars (~${totalTokens.toLocaleString()} tokens)`);
    console.log(`  Cache cost: ~$${(totalTokens / 1_000_000).toFixed(3)}/hour`);
}

main().catch(err => { console.error('Error:', err); process.exit(1); });
