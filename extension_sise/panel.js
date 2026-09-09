/* ═══════════════════════════════════════════════════════════════════════════
 * EL TALLER, DESDE SISE — pulsando, no fabricando peticiones
 * ═══════════════════════════════════════════════════════════════════════════
 * David: «la manera real y efectiva es que solo hagas click en los archiveros
 * y te devuelven el pdf».
 *
 * Tenía razón desde el principio y yo insistí cuatro veces en imitar la
 * petición por detrás. La última intentona se estrelló contra el filtro de
 * IIS —«request filtering is configured to deny double escape sequences»—:
 * una petición hecha por un guion no se parece a una navegación por más
 * cabeceras que se le copien.
 *
 * Aquí no se fabrica nada. Se PULSA el archivero, el navegador hace lo que
 * haría con el dedo del secretario, y el trabajador de fondo recoge el
 * fichero que cae.
 *
 * DOS FASES, PORQUE EL CLIC NAVEGA:
 *   · en el Panel Central se guarda la ficha del expediente y se pulsa
 *     «Promoción», que lleva al Panel de Promociones;
 *   · allí se pulsan los archiveros, se recogen los PDF y se manda todo.
 * La ficha viaja entre las dos en `chrome.storage.local`, porque el número
 * de expediente NO consta en el segundo panel —comprobado— y sin él no se
 * sabe de qué asunto son las constancias.
 */
(() => {
  "use strict";
  if (document.getElementById("iurexia-barra")) return;

  /* ═══════════════════════════════════════════════════════════════════════
   * QUE NO PUEDA FALLAR EN SILENCIO
   * ═══════════════════════════════════════════════════════════════════════
   * Tres vueltas perdidas con el mismo síntoma —«no pasa nada al pulsar»—
   * porque el guion moría entre dibujar el recuadro y enganchar el botón, y
   * ahí no había quién lo contara. El recuadro quedaba dibujado y muerto,
   * indistinguible de uno vivo.
   *
   * Ahora: rastro en la consola con prefijo, y cualquier error va a parar al
   * propio recuadro. Un instrumento que se calla es peor que no tenerlo.
   */
  const LOG = (...a) => console.log("[iurexia]", ...a);
  const pintarError = (e) => {
    const c = document.getElementById("iurexia-estado");
    const t = (e && e.message) || String(e);
    LOG("ERROR", t, e);
    if (c) c.innerHTML += `<div class="mal">${t}</div>`;
  };
  window.addEventListener("error", (ev) => {
    if (/panel\.js/.test(ev.filename || "")) pintarError(ev.error || ev.message);
  });
  LOG("guion cargado en", location.pathname);

  const API = "https://jurexia-api.onrender.com";
  const VERSION = "v0.8";
  const enPromociones = /PanelPromociones/i.test(location.pathname);

  const txt = (n) => (n ? n.textContent.replace(/\s+/g, " ").trim() : "");

  function fichaDelExpediente() {
    const t = document.body.innerText;
    const uno = (rx) => (t.match(rx) || [, ""])[1].trim();
    return {
      unico: uno(/Número de Expediente Único Nacional:\s*(\d+)/i),
      numero: uno(/Número de Expediente Asignado:\s*([\d]+\/[\d]{4})/i),
      tipo: uno(/Tipo de asunto:\s*([^\n]+)/i),
      organo: (t.match(/^(.*Tribunal Colegiado[^\n]*)$/mi) || [, ""])[1].trim(),
    };
  }

  function actuaciones() {
    const filas = [];
    for (const tr of document.querySelectorAll("table tr")) {
      const botones = tr.querySelectorAll('input[type=image][name*="grvPanelCentral"]');
      if (!botones.length) continue;
      const celdas = [...tr.querySelectorAll("td")].map(txt);
      filas.push({
        acuerdo: celdas[0] || "", publicacion: celdas[1] || "",
        promocion: celdas[2] || "", determinacion: celdas[4] || "",
      });
    }
    return filas;
  }

  /** Pulsa de verdad y espera a que el trabajador de fondo traiga el fichero. */
  function pulsarYRecoger(boton, clave, segundos = 45) {
    return new Promise((resolve, reject) => {
      const oyente = (msg) => {
        if (msg?.clave !== clave) return;
        chrome.runtime.onMessage.removeListener(oyente);
        clearTimeout(reloj);
        if (msg.que === "descarga-lista") {
          resolve(new Blob([new Uint8Array(msg.bytes)], { type: "application/pdf" }));
        } else {
          reject(new Error(msg.error || "la descarga falló"));
        }
      };
      const reloj = setTimeout(() => {
        chrome.runtime.onMessage.removeListener(oyente);
        // SI NO CAE NADA, casi siempre es que falta la casilla de acceso a
        // ficheros. Decirlo aquí ahorra media hora de buscar en otro sitio.
        reject(new Error("no llegó ningún fichero en " + segundos + "s. "
          + "Comprueba «Permitir acceso a URL de archivo» en chrome://extensions"));
      }, segundos * 1000);
      chrome.runtime.onMessage.addListener(oyente);
      chrome.runtime.sendMessage({ que: "esperar-descarga", datos: { clave } })
        .then(() => boton.click())
        .catch(reject);
    });
  }

  const barra = document.createElement("div");
  barra.id = "iurexia-barra";
  barra.innerHTML = `
    <h4>Taller de sentencias · Iurexia
        <span style="opacity:.45;font-weight:400">${VERSION}</span></h4>
    <p>Trae las constancias de este expediente sin que teclees nada.
       Tu contraseña de SISE no sale de aquí.</p>
    <button type="button" id="iurexia-ir">Traer las constancias</button>
    <div id="iurexia-estado"></div>`;
  document.body.appendChild(barra);

  const estado = barra.querySelector("#iurexia-estado");
  const boton = barra.querySelector("#iurexia-ir");
  // SE ENGANCHA AQUÍ, lo primero. Si algo revienta más abajo, el botón ya
  // responde y puede contar qué pasó, en vez de quedarse mudo.
  boton.addEventListener("click", (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    LOG("pulsado");
    try {
      arrancar();
    } catch (e) {
      pintarError(e);
    }
  });
  const di = (h) => { estado.innerHTML = h; };
  const suma = (h) => { estado.innerHTML += h; };

  /* ── FASE 1 · el Panel Central ─────────────────────────────────────────── */
  async function faseCentral() {
    const ficha = fichaDelExpediente();
    if (!ficha.numero) {
      throw new Error("No reconozco esta pantalla. Ábrela desde el Panel "
                    + "Central de Consultas de un expediente.");
    }
    const filas = actuaciones();
    const promo = [...document.querySelectorAll('input[type=image][name$="imgPromocion"]')][0];
    if (!promo) throw new Error("Este cuaderno no tiene ninguna promoción.");
    await chrome.storage.local.set({
      sise_ficha: ficha, sise_actuaciones: filas, sise_en_curso: true,
    });
    di(`<div>Expediente <b>${ficha.numero}</b> · voy al panel de promociones…</div>`);
    // Un clic de verdad: navega, y la fase 2 sigue sola al cargar.
    promo.click();
  }

  /* ── FASE 2 · el Panel de Promociones ──────────────────────────────────── */
  async function fasePromociones() {
    const g = await chrome.storage.local.get(
      ["sise_ficha", "sise_actuaciones", "sise_en_curso"]);
    const ficha = g.sise_ficha;
    if (!ficha?.numero) {
      throw new Error("Vengo sin la ficha del expediente. Empieza desde el "
                    + "Panel Central: aquí no consta de qué asunto es esto.");
    }
    await chrome.storage.local.set({ sise_en_curso: false });

    // La fecha de presentación está en la tabla, no dentro de un PDF.
    let presentacion = "";
    const grid = document.querySelector('[id*="grvPanelCentral"]');
    if (grid && grid.rows.length > 1) {
      const cab = [...grid.rows[0].cells].map((c) => c.innerText.trim());
      const i = cab.findIndex((h) => /fecha de presentaci/i.test(h));
      if (i >= 0) presentacion = (grid.rows[1].cells[i] || {}).innerText?.trim() || "";
    }
    di(`<div>Expediente <b>${ficha.numero}</b>`
       + (presentacion ? ` · presentado el <b>${presentacion}</b>` : "") + "</div>");

    const archivos = {};
    const aPulsar = [
      ["promocion", document.querySelector('input[type=image][name$="imgArchivo"]'),
       "el escaneo con las constancias"],
      ["acuerdo_asociado", document.querySelector('input[type=image][name$="imgArchivoDJ"]'),
       "la determinación asociada"],
    ].filter(([, b]) => b);

    for (const [clave, b, comoSeLlama] of aPulsar) {
      suma(`<div class="doc"><span>${comoSeLlama}</span><span>…</span></div>`);
      try {
        archivos[clave] = await pulsarYRecoger(b, clave);
        estado.lastElementChild.lastElementChild.innerHTML =
          `<span class="bien">${Math.round(archivos[clave].size / 1024)} KB</span>`;
      } catch (e) {
        estado.lastElementChild.lastElementChild.innerHTML =
          `<span class="mal">${e.message}</span>`;
      }
    }
    if (!archivos.promocion) {
      throw new Error("Sin el escaneo no hay nada que proyectar.");
    }

    const correo = (await chrome.storage.local.get("correo")).correo || "";
    if (!correo) {
      throw new Error("Falta tu correo de Iurexia: guárdalo en las opciones "
                    + "de la extensión, una sola vez.");
    }
    const d = new FormData();
    d.append("user_email", correo);
    d.append("numero", ficha.numero);
    d.append("expediente_unico", ficha.unico || "");
    d.append("tipo_sise", ficha.tipo || "");
    d.append("organo", ficha.organo || "");
    if (presentacion) d.append("presentacion_sise", presentacion);
    d.append("actuaciones_json", JSON.stringify(g.sise_actuaciones || []));
    d.append("promocion", archivos.promocion, "promocion.pdf");
    if (archivos.acuerdo_asociado) {
      d.append("acuerdos", archivos.acuerdo_asociado, "acuerdo_asociado.pdf");
    }
    suma("<div>Enviando al taller…</div>");
    const r = await fetch(`${API}/taller/desde-sise`, { method: "POST", body: d });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.detail || `El taller respondió ${r.status}`);
    suma(`<div class="bien">Listo. El expediente ${ficha.numero} está en el
          taller con sus constancias.</div>`);
    if (Array.isArray(j.avisos)) {
      for (const a of j.avisos) suma(`<div class="mal">${a}</div>`);
    }
  }

  async function arrancar() {
    LOG("arrancar · enPromociones =", enPromociones);
    boton.disabled = true;
    try {
      await (enPromociones ? fasePromociones() : faseCentral());
    } catch (e) {
      pintarError(e);
    } finally {
      boton.disabled = false;
    }
  }

  // (el botón se enganchó arriba, antes de que nada pudiera reventar)

  // SIGUE SOLA. Si venimos del Panel Central, la fase 2 arranca al cargar:
  // para el secretario es un solo clic aunque por dentro sean dos pantallas.
  if (enPromociones) {
    chrome.storage.local.get("sise_en_curso").then((g) => {
      if (g.sise_en_curso) arrancar();
    });
  }
})();
