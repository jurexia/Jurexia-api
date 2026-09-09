/* ═══════════════════════════════════════════════════════════════════════════
 * CAPTURAR EL PDF DE LA RESPUESTA REAL, CON EL DEPURADOR
 * ═══════════════════════════════════════════════════════════════════════════
 * David: «si acepto lo de arrastrar el archivo, estoy casi en el mismo lugar en
 * el que estoy actualmente. No hay una automatización auténtica».
 *
 * Tiene razón, y las dos vías que intenté antes eran callejones:
 *
 *   · FABRICAR la petición con fetch → SISE la rechaza. El filtro de IIS
 *     respondió «request filtering is configured to deny double escape
 *     sequences»: una petición de guion no se parece a una navegación.
 *   · DEJAR que se descargue y leer el fichero → Chrome protege el disco, y
 *     aunque se conceda la casilla, el trabajador de fondo no lo lee.
 *
 * La vía correcta es la tercera y no la había usado: el propio depurador de
 * Chrome. Se engancha a la pestaña, intercepta la RESPUESTA del clic —el clic
 * de verdad, el que SISE acepta sin rechistar— y devuelve el cuerpo. Ni se
 * fabrica la petición ni se toca el disco.
 *
 * El precio: Chrome enseña un aviso de «se está depurando este navegador»
 * mientras dura. Es visible y es honesto: la extensión está mirando el tráfico
 * de esa pestaña, y el secretario debe saberlo.
 */
const PROTOCOLO = "1.3";
const capturando = new Map();   // tabId → {clave, resolver, rechazar, reloj}

async function enviar(tabId, metodo, params) {
  return chrome.debugger.sendCommand({ tabId }, metodo, params || {});
}

async function engancharSiHaceFalta(tabId) {
  const enganchadas = await chrome.debugger.getTargets();
  const ya = enganchadas.some((t) => t.tabId === tabId && t.attached);
  if (!ya) {
    await chrome.debugger.attach({ tabId }, PROTOCOLO);
    await enviar(tabId, "Fetch.enable", {
      patterns: [{ urlPattern: "*sise.cjf.gob.mx*", requestStage: "Response" }],
    });
  }
}

chrome.debugger.onEvent.addListener(async (fuente, metodo, params) => {
  if (metodo !== "Fetch.requestPaused") return;
  const tabId = fuente.tabId;
  const pendiente = capturando.get(tabId);
  const tipo = (params.responseHeaders || [])
    .find((h) => /^content-type$/i.test(h.name))?.value || "";
  const disposicion = (params.responseHeaders || [])
    .find((h) => /^content-disposition$/i.test(h.name))?.value || "";
  const esPDF = /application\/pdf|application\/octet-stream/i.test(tipo)
             || /\.pdf/i.test(disposicion);

  if (pendiente && esPDF) {
    try {
      const r = await enviar(tabId, "Fetch.getResponseBody",
                             { requestId: params.requestId });
      // El depurador devuelve el cuerpo en base64 cuando es binario.
      const bytes = r.base64Encoded
        ? Uint8Array.from(atob(r.body), (c) => c.charCodeAt(0))
        : new TextEncoder().encode(r.body);
      const cab = String.fromCharCode(...bytes.slice(0, 5));
      if (!cab.startsWith("%PDF")) throw new Error("la respuesta no es un PDF");
      clearTimeout(pendiente.reloj);
      capturando.delete(tabId);
      pendiente.resolver({ bytes: Array.from(bytes) });
    } catch (e) {
      clearTimeout(pendiente.reloj);
      capturando.delete(tabId);
      pendiente.rechazar(String(e.message || e));
    }
    // FALLAR LA PETICIÓN A PROPÓSITO: ya tenemos el cuerpo, y dejarla seguir
    // descargaría el fichero otra vez a la carpeta del secretario. Nadie
    // quiere quince PDF sueltos por cada expediente.
    try { await enviar(tabId, "Fetch.failRequest",
                       { requestId: params.requestId, errorReason: "Aborted" }); }
    catch (_) { /* ya se cerró */ }
    return;
  }
  try { await enviar(tabId, "Fetch.continueRequest", { requestId: params.requestId }); }
  catch (_) { /* la petición ya no está */ }
});

chrome.runtime.onMessage.addListener((msg, remitente, responder) => {
  const tabId = remitente?.tab?.id;
  (async () => {
    try {
      if (msg?.que === "preparar-captura") {
        await engancharSiHaceFalta(tabId);
        responder({ ok: true });
      } else if (msg?.que === "capturar") {
        await engancharSiHaceFalta(tabId);
        const p = new Promise((resolver, rechazar) => {
          const reloj = setTimeout(() => {
            capturando.delete(tabId);
            rechazar("no llegó ningún PDF en 45 s");
          }, 45000);
          capturando.set(tabId, { clave: msg.clave, resolver, rechazar, reloj });
        });
        responder({ ok: true });           // el guion ya puede pulsar
        const r = await p.catch((e) => ({ error: String(e) }));
        await chrome.tabs.sendMessage(tabId, { que: "capturado", clave: msg.clave, ...r });
      } else if (msg?.que === "soltar") {
        try { await chrome.debugger.detach({ tabId }); } catch (_) {}
        responder({ ok: true });
      }
    } catch (e) {
      responder({ ok: false, error: String(e.message || e) });
    }
  })();
  return true;
});

// Si la pestaña se va, se suelta el depurador: dejar el aviso puesto sin
// motivo es la clase de detalle que hace que alguien desinstale la extensión.
chrome.tabs.onRemoved.addListener((tabId) => {
  chrome.debugger.detach({ tabId }).catch(() => {});
  capturando.delete(tabId);
});
