/* ═══════════════════════════════════════════════════════════════════════════
 * EL TALLER, DESDE SISE — el guion sólo pulsa; el resto vive en el fondo
 * ═══════════════════════════════════════════════════════════════════════════
 * Pulsar un archivero RECARGA la página, y con ella muere este guion. Por eso
 * aquí no hay promesas que esperen bytes ni estado que sobrevivir: cada vez
 * que la página carga, esto pregunta al trabajador de fondo qué toca, lo pulsa
 * y se deja morir. El fondo lleva la cuenta y manda todo a Iurexia al final.
 *
 * Costó ocho versiones llegar aquí, y todas se estrellaron contra la misma
 * piedra por sitios distintos: nada que dependa de sobrevivir a un clic puede
 * vivir en la página.
 */
(() => {
  "use strict";
  if (document.getElementById("iurexia-barra")) return;

  const VERSION = "v1.0.2";
  const LOG = (...a) => console.log("[iurexia]", ...a);
  const enPromociones = /PanelPromociones/i.test(location.pathname);

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
  const di = (h) => { estado.innerHTML = h; };
  const suma = (h) => { estado.innerHTML += h; };
  const mal = (t) => { LOG("ERROR", t); suma(`<div class="mal">${t}</div>`); };

  window.addEventListener("error", (ev) => {
    if (/panel\.js/.test(ev.filename || "")) mal(ev.message);
  });
  LOG("guion cargado en", location.pathname);

  const txt = (n) => (n ? n.textContent.replace(/\s+/g, " ").trim() : "");
  const pregunta = (m) => chrome.runtime.sendMessage(m);

  function ficha() {
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
      if (!tr.querySelector('input[type=image][name*="grvPanelCentral"]')) continue;
      const c = [...tr.querySelectorAll("td")].map(txt);
      filas.push({ acuerdo: c[0] || "", publicacion: c[1] || "",
                   promocion: c[2] || "", determinacion: c[4] || "" });
    }
    return filas;
  }

  function fechaDePresentacion() {
    const g = document.querySelector('[id*="grvPanelCentral"]');
    if (!g || g.rows.length < 2) return "";
    const cab = [...g.rows[0].cells].map((c) => c.innerText.trim());
    const i = cab.findIndex((h) => /fecha de presentaci/i.test(h));
    return i >= 0 ? (g.rows[1].cells[i] || {}).innerText?.trim() || "" : "";
  }

  /** Pulsa y se deja morir: la página se recarga y el ciclo sigue al cargar. */
  async function pulsar(b, clave, comoSeLlama) {
    suma(`<div class="doc"><span>${comoSeLlama}</span><span>…</span></div>`);
    const r = await pregunta({ que: "voy-a-pulsar", clave,
                               presentacion: fechaDePresentacion() });
    if (!r?.ok) {
      // SI EL FONDO DICE BASTA, SE PARA. No se pulsa «por si acaso»: cada
      // pulsación deja un fichero en la carpeta del secretario.
      mal(r?.error || "no se pudo preparar la captura");
      return false;
    }
    LOG("pulsando", clave);
    b.click();
    return true;
  }

  /** En el Panel de Promociones: qué falta por traer. */
  async function seguir() {
    const s = await pregunta({ que: "estado" });
    if (!s?.hay) return false;
    const ya = s.capturados || [];
    di(`<div>Expediente <b>${s.ficha?.numero || "?"}</b> · `
       + `${ya.length} constancia(s) recogida(s)</div>`);
    for (const e of (s.errores || [])) mal(e);

    const archivo = document.querySelector('input[type=image][name$="imgArchivo"]');
    const dj = document.querySelector('input[type=image][name$="imgArchivoDJ"]');
    if (archivo && !ya.includes("promocion")) {
      if (await pulsar(archivo, "promocion", "el escaneo con las constancias")) return true;
      return true;   // agotado: el fondo ya lo dijo y no se insiste
    }
    if (dj && !ya.includes("acuerdo_asociado")) {
      if (await pulsar(dj, "acuerdo_asociado", "la determinación asociada")) return true;
      return true;
    }
    suma("<div>Enviando al taller…</div>");
    const r = await pregunta({ que: "enviar" });
    if (r?.error) return mal(r.error), true;
    suma(`<div class="bien">Listo. El expediente ${s.ficha?.numero} está en el
          taller con ${Object.keys(r.documentos || {}).length || (r.documentos || []).length}
          constancia(s).</div>`);
    for (const a of (r.avisos || [])) mal(a);
    return true;
  }

  async function arrancar() {
    boton.disabled = true;
    try {
      if (enPromociones) {
        if (!(await seguir())) {
          mal("Vengo sin la ficha del expediente. Empieza desde el Panel "
            + "Central: aquí no consta de qué asunto es esto.");
        }
        return;
      }
      const f = ficha();
      if (!f.numero) {
        return mal("No reconozco esta pantalla. Ábrela desde el Panel Central "
                 + "de Consultas de un expediente.");
      }
      const correo = (await chrome.storage.local.get("correo")).correo || "";
      if (!correo) {
        return mal("Falta tu correo de Iurexia: guárdalo en las opciones de la "
                 + "extensión, una sola vez.");
      }
      const promo = document.querySelector('input[type=image][name$="imgPromocion"]');
      if (!promo) return mal("Este cuaderno no tiene ninguna promoción.");
      const r = await pregunta({ que: "iniciar", ficha: f,
                                 actuaciones: actuaciones(), correo });
      if (!r?.ok) return mal(r?.error || "no se pudo iniciar la captura");
      di(`<div>Expediente <b>${f.numero}</b> · voy al panel de promociones…</div>`);
      promo.click();
    } catch (e) {
      mal(e.message || String(e));
    } finally {
      boton.disabled = false;
    }
  }

  boton.addEventListener("click", (ev) => {
    ev.preventDefault(); ev.stopPropagation();
    LOG("pulsado"); arrancar();
  });

  // SIGUE SOLA. Si hay una captura en curso, esta carga es un paso más del
  // ciclo y no hay que pulsar nada: para el secretario fue un solo clic.
  if (enPromociones) {
    pregunta({ que: "estado" }).then((s) => { if (s?.hay) seguir(); }).catch(() => {});
  }
})();
