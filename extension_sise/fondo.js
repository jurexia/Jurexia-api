/* ═══════════════════════════════════════════════════════════════════════════
 * RECOGER LO QUE EL NAVEGADOR DESCARGA
 * ═══════════════════════════════════════════════════════════════════════════
 * David: «la manera real y efectiva es que solo hagas click en los archiveros
 * y te devuelven el pdf».
 *
 * Tenía razón desde el principio. Fabricar la petición falló cuatro veces —la
 * última con la página de error de IIS: «request filtering is configured to
 * deny double escape sequences»—, porque una petición hecha por un guion no se
 * parece a una navegación por más cabeceras que se le copien.
 *
 * Así que no se fabrica nada: el guion PULSA el archivero, el navegador hace
 * exactamente lo que haría con el dedo del secretario, y esto recoge el
 * fichero que cae. Más piezas, pero deja de depender de que yo acierte cómo
 * espera SISE cada byte.
 */
const enEspera = new Map();   // id de descarga → cómo mandarla

chrome.runtime.onMessage.addListener((msg, remitente, responder) => {
  if (msg?.que === "esperar-descarga") {
    // DE QUÉ PESTAÑA VIENE. Sin esto, `chrome.tabs.sendMessage(undefined, …)`
    // revienta y el PDF descargado se queda en el disco sin llegar a nadie:
    // el fichero caía, la barra no decía nada y el taller no recibía. Es el
    // dato que el propio evento trae y yo no estaba mirando.
    enEspera.set("siguiente", { ...msg.datos, tabId: remitente?.tab?.id });
    responder({ ok: true });
  }
  return true;
});

chrome.downloads.onCreated.addListener((d) => {
  const pendiente = enEspera.get("siguiente");
  if (!pendiente) return;
  enEspera.delete("siguiente");
  enEspera.set(d.id, pendiente);
});

chrome.downloads.onChanged.addListener(async (cambio) => {
  if (cambio.state?.current !== "complete") return;
  const pendiente = enEspera.get(cambio.id);
  if (!pendiente) return;
  enEspera.delete(cambio.id);
  try {
    const [d] = await chrome.downloads.search({ id: cambio.id });
    if (!d?.filename) throw new Error("la descarga no dejó ruta");
    // LEER EL FICHERO EXIGE «Permitir acceso a URL de archivo» en la
    // extensión. Es una casilla, una vez, y sin ella esto no puede funcionar:
    // Chrome no deja a una extensión leer el disco por defecto, y hace bien.
    const r = await fetch("file://" + d.filename.replace(/\\/g, "/"));
    const buf = await r.arrayBuffer();
    const cab = String.fromCharCode(...new Uint8Array(buf.slice(0, 5)));
    if (!cab.startsWith("%PDF")) throw new Error("lo descargado no es un PDF");
    await chrome.tabs.sendMessage(pendiente.tabId, {
      que: "descarga-lista", clave: pendiente.clave,
      bytes: Array.from(new Uint8Array(buf)), nombre: d.filename.split(/[\\/\\\\]/).pop(),
    });
  } catch (e) {
    try {
      await chrome.tabs.sendMessage(pendiente.tabId, {
        que: "descarga-falló", clave: pendiente.clave, error: String(e.message || e),
      });
    } catch (_) { /* la pestaña ya no está */ }
  }
});
