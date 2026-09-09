/* ═══════════════════════════════════════════════════════════════════════════
 * LA CAPTURA VIVE AQUÍ, PORQUE LA PÁGINA SE RECARGA EN CADA CLIC
 * ═══════════════════════════════════════════════════════════════════════════
 * El rastro de la v0.9 lo enseñó entero:
 *
 *     [iurexia] pulsado
 *     [iurexia] arrancar · enPromociones = true
 *     [iurexia] pulsando promocion
 *     [iurexia] guion cargado en …/PanelPromociones.aspx   ← la página SE RECARGÓ
 *
 * Pulsar el archivero recarga la página, y con ella muere el guion que estaba
 * esperando los bytes. Poner la captura y el envío en el guion de la página fue
 * un error de arquitectura mío: nada que dependa de sobrevivir a un clic puede
 * vivir ahí.
 *
 * Aquí no. El trabajador de fondo aguanta las recargas, así que lleva él la
 * máquina de estados: qué documentos faltan, cuáles ya tiene, y cuándo mandar
 * todo a Iurexia. El guion de la página queda reducido a lo único que sólo él
 * puede hacer —pulsar— y a preguntar «¿qué toca ahora?» cada vez que carga.
 */
const PROTOCOLO = "1.3";
const API = "https://jurexia-api.onrender.com";
const sesiones = new Map();   // tabId → estado de la captura

const log = (...a) => console.log("[iurexia·fondo]", ...a);

async function cmd(tabId, metodo, params) {
  return chrome.debugger.sendCommand({ tabId }, metodo, params || {});
}

async function enganchar(tabId) {
  const objetivos = await chrome.debugger.getTargets();
  if (objetivos.some((t) => t.tabId === tabId && t.attached)) return;
  await chrome.debugger.attach({ tabId }, PROTOCOLO);
  await cmd(tabId, "Fetch.enable", {
    patterns: [{ urlPattern: "*sise.cjf.gob.mx*", requestStage: "Response" }],
  });
  log("enganchado a", tabId);
}

async function soltar(tabId) {
  try { await chrome.debugger.detach({ tabId }); } catch (_) {}
}

chrome.debugger.onEvent.addListener(async (fuente, metodo, params) => {
  if (metodo !== "Fetch.requestPaused") return;
  const tabId = fuente.tabId;
  const s = sesiones.get(tabId);
  const cab = (n) => (params.responseHeaders || [])
    .find((h) => h.name.toLowerCase() === n)?.value || "";
  const esPDF = /application\/pdf|octet-stream/i.test(cab("content-type"))
             || /\.pdf/i.test(cab("content-disposition"));

  if (s && s.esperando && esPDF) {
    const clave = s.esperando;
    s.esperando = null;
    try {
      const r = await cmd(tabId, "Fetch.getResponseBody", { requestId: params.requestId });
      const bytes = r.base64Encoded
        ? Uint8Array.from(atob(r.body), (c) => c.charCodeAt(0))
        : new TextEncoder().encode(r.body);
      if (String.fromCharCode(...bytes.slice(0, 5)) !== "%PDF") {
        throw new Error("la respuesta no era un PDF");
      }
      s.capturados[clave] = Array.from(bytes);
      log("capturado", clave, Math.round(bytes.length / 1024), "KB");
    } catch (e) {
      s.errores.push(`${clave}: ${e.message || e}`);
      log("falló", clave, e);
    }
    // ABORTAR LA PETICIÓN: ya tenemos el cuerpo. Dejarla seguir descargaría
    // además el fichero a la carpeta del secretario, uno por constancia.
    try { await cmd(tabId, "Fetch.failRequest",
                    { requestId: params.requestId, errorReason: "Aborted" }); } catch (_) {}
    return;
  }
  try { await cmd(tabId, "Fetch.continueRequest", { requestId: params.requestId }); } catch (_) {}
});

async function enviarAIurexia(tabId) {
  const s = sesiones.get(tabId);
  if (!s) return { error: "sin sesión" };
  const d = new FormData();
  d.append("user_email", s.correo);
  d.append("numero", s.ficha.numero || "");
  d.append("expediente_unico", s.ficha.unico || "");
  d.append("tipo_sise", s.ficha.tipo || "");
  d.append("organo", s.ficha.organo || "");
  if (s.presentacion) d.append("presentacion_sise", s.presentacion);
  d.append("actuaciones_json", JSON.stringify(s.actuaciones || []));
  const uno = (k) => new Blob([new Uint8Array(s.capturados[k])], { type: "application/pdf" });
  if (!s.capturados.promocion) return { error: "no se capturó el escaneo principal" };
  d.append("promocion", uno("promocion"), "promocion.pdf");
  for (const k of Object.keys(s.capturados)) {
    if (k !== "promocion") d.append("acuerdos", uno(k), `${k}.pdf`);
  }
  try {
    const r = await fetch(`${API}/taller/desde-sise`, { method: "POST", body: d });
    const j = await r.json().catch(() => ({}));
    return r.ok ? { ok: true, ...j } : { error: j.detail || `el taller respondió ${r.status}` };
  } catch (e) {
    return { error: String(e.message || e) };
  }
}

chrome.runtime.onMessage.addListener((msg, remitente, responder) => {
  const tabId = remitente?.tab?.id;
  (async () => {
    try {
      if (msg?.que === "iniciar") {
        await enganchar(tabId);
        sesiones.set(tabId, {
          ficha: msg.ficha, actuaciones: msg.actuaciones || [], correo: msg.correo,
          presentacion: "", pendientes: [], capturados: {}, errores: [], esperando: null,
        });
        responder({ ok: true });
      } else if (msg?.que === "estado") {
        const s = sesiones.get(tabId);
        responder({ ok: true, hay: !!s, capturados: s ? Object.keys(s.capturados) : [],
                    errores: s ? s.errores : [], ficha: s ? s.ficha : null });
      } else if (msg?.que === "voy-a-pulsar") {
        const s = sesiones.get(tabId);
        if (!s) return responder({ ok: false, error: "no hay captura en curso" });
        await enganchar(tabId);
        if (msg.presentacion) s.presentacion = msg.presentacion;
        s.esperando = msg.clave;
        responder({ ok: true });
      } else if (msg?.que === "enviar") {
        const r = await enviarAIurexia(tabId);
        await soltar(tabId);
        sesiones.delete(tabId);
        responder(r);
      } else if (msg?.que === "cancelar") {
        await soltar(tabId);
        sesiones.delete(tabId);
        responder({ ok: true });
      }
    } catch (e) {
      responder({ ok: false, error: String(e.message || e) });
    }
  })();
  return true;
});

chrome.tabs.onRemoved.addListener((tabId) => { soltar(tabId); sesiones.delete(tabId); });
