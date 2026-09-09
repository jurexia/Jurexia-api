/* ═══════════════════════════════════════════════════════════════════════════
 * EL TALLER, DESDE SISE — sin salir de la sesión del secretario
 * ═══════════════════════════════════════════════════════════════════════════
 * David eligió la opción (A): «la del secretario, en su máquina». Su sesión,
 * su contraseña, que Iurexia no ve nunca. Este guion se ejecuta DENTRO de la
 * página del Panel Central de Consultas, así que las peticiones que hace van
 * con la cookie de sesión que ya existe —es `HttpOnly`, y por eso esto no se
 * puede hacer desde nuestro servidor ni aunque quisiéramos—.
 *
 * QUÉ HACE. Lee la tabla de actuaciones, se trae los PDF que el taller
 * necesita y los manda al adelanto junto con el número y el tipo de asunto,
 * que también están en la página. Con eso desaparece el formulario.
 *
 * LO QUE NO HACE, Y ES DELIBERADO: no adivina las fechas de notificación y de
 * presentación. Las manda como PISTA, con el PDF del que salen, para que el
 * pipeline las lea del documento y el secretario las confirme. De esas dos
 * fechas depende el cómputo entero, y una equivocada es exactamente lo que
 * dejó un proyecto en extemporáneo sin que nadie se enterara.
 */
(() => {
  "use strict";
  if (document.getElementById("iurexia-barra")) return;

  const API = "https://jurexia-api.onrender.com";

  // Los nombres de los controles son estables en SISE: la fila va en `ctlNN` y
  // el documento en el sufijo. Explorado el 9-sep-2026 sobre el 91/2025.
  const COLUMNAS = {
    promocion: "imgPromocion",
    determinacion: "imgDetJud",
    acuse: "imgAcuDetJud",
    notificacion: "imgNotConsCon",
  };

  const txt = (n) => (n ? n.textContent.replace(/\s+/g, " ").trim() : "");

  /** El encabezado de la página: expediente, tipo de asunto y número único. */
  function fichaDelExpediente() {
    const t = document.body.innerText;
    const uno = (rx) => (t.match(rx) || [, ""])[1].trim();
    return {
      unico: uno(/Número de Expediente Único Nacional:\s*(\d+)/i),
      numero: uno(/Número de Expediente Asignado:\s*([\d]+\/[\d]{4})/i),
      tipo: uno(/Tipo de asunto:\s*([^\n]+?)\s{2,}|Tipo de asunto:\s*([^\n]+)/i),
      organo: (t.match(/^(.*Tribunal Colegiado[^\n]*)$/mi) || [, ""])[1].trim(),
    };
  }

  /** Una fila por actuación, con sus fechas y qué documentos tiene. */
  function actuaciones() {
    const filas = [];
    for (const tr of document.querySelectorAll("table tr")) {
      const botones = tr.querySelectorAll('input[type=image][name*="grvPanelCentral"]');
      if (!botones.length) continue;
      const celdas = [...tr.querySelectorAll("td")].map(txt);
      const ctl = (botones[0].name.match(/\$(ctl\d+)\$/) || [, ""])[1];
      const docs = {};
      for (const b of botones) {
        for (const [clave, suf] of Object.entries(COLUMNAS)) {
          if (b.name.endsWith(suf)) docs[clave] = b.name;
        }
      }
      filas.push({
        ctl,
        fechaAcuerdo: celdas[0] || "",
        fechaPublicacion: celdas[1] || "",
        contenidoPromocion: celdas[2] || "",
        contenidoDeterminacion: celdas[4] || "",
        docs,
      });
    }
    return filas;
  }

  /** Los campos ocultos de un formulario, que es lo que SISE exige devolver. */
  function ocultosDe(raiz) {
    const c = new FormData();
    for (const el of raiz.querySelectorAll("input[type=hidden]")) {
      if (el.name) c.append(el.name, el.value);
    }
    return c;
  }

  /** Pulsa un control de SISE y devuelve la respuesta cruda. */
  async function pulsar(accion, cuerpo, nombreControl) {
    // Un input[type=image] viaja como `nombre.x` / `nombre.y`: sin eso, el
    // servidor no sabe qué botón se pulsó y devuelve la misma página.
    cuerpo.append(nombreControl + ".x", "8");
    cuerpo.append(nombreControl + ".y", "8");
    return fetch(accion, { method: "POST", body: cuerpo, credentials: "include" });
  }

  async function aPDF(r, deQue) {
    const buf = await r.arrayBuffer();
    const cab = String.fromCharCode(...new Uint8Array(buf.slice(0, 5)));
    if (!cab.startsWith("%PDF")) {
      throw new Error(`SISE no devolvió un PDF de ${deQue} (¿caducó la sesión?)`);
    }
    return new Blob([buf], { type: "application/pdf" });
  }

  /** Un documento que se descarga de un solo clic (acuerdos, notificaciones). */
  async function traerPDF(nombreControl) {
    const f = document.forms[0];
    return aPDF(await pulsar(f.action, ocultosDe(f), nombreControl), "esa constancia");
  }

  /* ═══════════════════════════════════════════════════════════════════════
   * LA PROMOCIÓN VA EN DOS PASOS, Y ÉSE ERA EL FALLO
   * ═══════════════════════════════════════════════════════════════════════
   * David: «te faltó cliquear el primer botón para que te mandara al panel de
   * promociones (la principal); allí jalas el escaneo principal que tiene
   * todas las constancias».
   *
   * El icono de «Promoción» del Panel Central NO devuelve un PDF: NAVEGA al
   * Panel de Promociones. Allí está el escaneo de verdad —`imgArchivo`— y,
   * de regalo, la FECHA DE PRESENTACIÓN en la propia tabla, que es uno de los
   * datos que el secretario teclea hoy.
   *
   * Mi primera versión pulsaba el primer icono y esperaba un PDF. Recibía el
   * HTML del panel intermedio, y la comprobación de `%PDF` lo rechazaba: la
   * salvaguarda funcionó, el camino estaba mal.
   */
  async function traerPromocion(nombreControl) {
    const f = document.forms[0];
    const r1 = await pulsar(f.action, ocultosDe(f), nombreControl);
    const html = await r1.text();
    const doc = new DOMParser().parseFromString(html, "text/html");
    const grid = doc.querySelector('[id*="grvPanelCentral"]');
    const archivo = doc.querySelector('input[type=image][name$="imgArchivo"]');
    if (!archivo) {
      throw new Error("El panel de promociones no trae el escaneo principal.");
    }
    // La fecha de presentación, de la tabla y no de un PDF.
    let presentacion = "";
    if (grid) {
      const cabeceras = [...grid.rows[0].cells].map((c) => c.innerText.trim());
      const iFecha = cabeceras.findIndex((h) => /fecha de presentaci/i.test(h));
      if (iFecha >= 0 && grid.rows[1]) {
        presentacion = (grid.rows[1].cells[iFecha] || {}).innerText?.trim() || "";
      }
    }
    const r2 = await pulsar(r1.url, ocultosDe(doc), archivo.getAttribute("name"));
    const pdf = await aPDF(r2, "la promoción");

    // LA «DETERMINACIÓN JUDICIAL ASOCIADA» de este panel es el acuerdo que
    // recayó a ESTA promoción: en un recurso, el auto que lo admite. Está aquí
    // y no en el Panel Central, así que si no se toma ahora hay que volver.
    let asociada = null;
    const dj = doc.querySelector('input[type=image][name$="imgArchivoDJ"]');
    if (dj) {
      try {
        const r3 = await pulsar(r1.url, ocultosDe(doc), dj.getAttribute("name"));
        asociada = await aPDF(r3, "la determinación asociada");
      } catch (e) { /* si no está, se sigue: los acuerdos del panel la cubren */ }
    }
    return { pdf, presentacion, asociada };
  }

  // EN QUÉ PANTALLA ESTAMOS. El Panel de Promociones no lleva el número de
  // expediente en ninguna parte —comprobado leyendo sus campos ocultos—, así
  // que desde ahí no se puede saber de qué asunto son las constancias.
  const enPromociones = /PanelPromociones/i.test(location.pathname);

  const barra = document.createElement("div");
  barra.id = "iurexia-barra";
  barra.innerHTML = `
    <h4>Taller de sentencias · Iurexia <span style="opacity:.45;font-weight:400">v0.2</span></h4>
    <p>Trae las constancias de este expediente sin que teclees nada.
       Tu contraseña de SISE no sale de aquí.</p>
    <button id="iurexia-ir">${enPromociones
        ? "Vuelve al Panel Central" : "Traer las constancias"}</button>
    <div id="iurexia-estado">${enPromociones
        ? "Estás en el panel de promociones, y aquí no consta de qué expediente "
          + "son estas constancias. Pulsa «Regresar» y dale al botón allí: desde "
          + "el Panel Central se trae todo, este panel incluido."
        : ""}</div>`;
  document.body.appendChild(barra);

  const estado = barra.querySelector("#iurexia-estado");
  const boton = barra.querySelector("#iurexia-ir");
  const di = (h) => { estado.innerHTML = h; };

  boton.addEventListener("click", async () => {
    if (enPromociones) {
      // Volver es un clic y se hace solo: el botón «Regresar» está ahí.
      const r = [...document.querySelectorAll('input[type=image],input[type=submit],a')]
        .find((e) => /regresar/i.test(e.value || e.alt || e.textContent || ""));
      if (r) r.click();
      return;
    }
    boton.disabled = true;
    try {
      const ficha = fichaDelExpediente();
      const filas = actuaciones();
      if (!ficha.numero || !filas.length) {
        throw new Error("No reconozco esta pantalla. Ábrela desde el Panel "
                      + "Central de Consultas de un expediente.");
      }
      di(`<div>Expediente <b>${ficha.numero}</b> · ${filas.length} actuaciones</div>`);

      // QUÉ SE TRAE, Y POR QUÉ TODOS LOS ACUERDOS.
      //
      // David: «también está el auto de admisión y el auto de turno, que son
      // indispensables para verificar datos como la presentación, los terceros
      // interesados, el magistrado ponente».
      //
      // La admisión y el turno son ACTUACIONES DISTINTAS: traer sólo el
      // acuerdo de la primera fila deja fuera el otro. Se traen todos los
      // acuerdos del cuaderno —son pocos, dos o tres al principio del
      // asunto— y ya los clasifica el pipeline leyéndolos. Adivinar cuál es
      // cuál por su posición en la tabla es la clase de suposición que aquí
      // sale cara.
      //
      // Y LA PROMOCIÓN de la primera fila que la tenga: es el escrito que abre
      // el asunto. MEDIDO en la revisión fiscal 91/2025: son 78 páginas y NO
      // traen la sentencia recurrida —ni su expediente, ni su «VISTOS los
      // autos», ni su «RESUELVE»—. Así que «el primero trae todo» es
      // «regularmente», no siempre, y el pipeline tiene que decirlo cuando
      // falte en vez de proyectar sin ella.
      const conPromocion = filas.find((f) => f.docs.promocion);
      const base = conPromocion || filas[0];
      const aTraer = [
        ["promocion", base.docs.promocion, "el escrito que abre el asunto"],
      ].filter(([, n]) => n);
      const TOPE_ACUERDOS = 4;
      filas.slice(0, TOPE_ACUERDOS).forEach((f, i) => {
        if (f.docs.determinacion) {
          aTraer.push([`acuerdo_${i + 1}`, f.docs.determinacion,
                       `acuerdo de ${f.fechaAcuerdo || "fecha desconocida"}`]);
        }
      });
      const conNotificacion = filas.find((f) => f.docs.notificacion);
      if (conNotificacion) {
        aTraer.push(["notificacion", conNotificacion.docs.notificacion,
                     "la notificación"]);
      }

      const archivos = {};
      let presentacion = "";
      for (const [clave, control, comoSeLlama] of aTraer) {
        di(estado.innerHTML + `<div class="doc"><span>${comoSeLlama}</span><span>…</span></div>`);
        try {
          if (clave === "promocion") {
            const r = await traerPromocion(control);
            archivos[clave] = r.pdf;
            presentacion = r.presentacion || "";
            if (r.asociada) archivos["acuerdo_asociado"] = r.asociada;
          } else {
            archivos[clave] = await traerPDF(control);
          }
          estado.lastElementChild.lastElementChild.innerHTML =
            `<span class="bien">${Math.round(archivos[clave].size / 1024)} KB</span>`;
        } catch (e) {
          estado.lastElementChild.lastElementChild.innerHTML =
            `<span class="mal">${e.message}</span>`;
        }
      }
      if (!archivos.promocion) {
        throw new Error("Sin el escrito que abre el asunto no hay nada que proyectar.");
      }

      const correo = (await chrome.storage.local.get("correo")).correo || "";
      if (!correo) {
        throw new Error("Falta tu correo de Iurexia. Ábrelo en las opciones de "
                      + "la extensión y guárdalo una vez.");
      }

      const d = new FormData();
      d.append("user_email", correo);
      d.append("numero", ficha.numero);
      d.append("expediente_unico", ficha.unico);
      d.append("tipo_sise", ficha.tipo);
      d.append("organo", ficha.organo);
      // LA FECHA DE PRESENTACIÓN, leída de la tabla del panel de promociones.
      // Sigue siendo una PISTA: el pipeline la confirma contra el acuse.
      if (presentacion) d.append("presentacion_sise", presentacion);
      d.append("actuaciones_json", JSON.stringify(filas.map((f) => ({
        ctl: f.ctl, acuerdo: f.fechaAcuerdo, publicacion: f.fechaPublicacion,
        promocion: f.contenidoPromocion, determinacion: f.contenidoDeterminacion,
      }))));
      for (const [k, b] of Object.entries(archivos)) {
        // Los acuerdos viajan todos bajo el mismo campo: el pipeline los
        // clasifica leyéndolos, no por el nombre que les pongamos aquí.
        d.append(k.startsWith("acuerdo_") ? "acuerdos" : k, b, `${k}.pdf`);
      }

      di(estado.innerHTML + "<div>Enviando al taller…</div>");
      const r = await fetch(`${API}/taller/desde-sise`, { method: "POST", body: d });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.detail || `El taller respondió ${r.status}`);
      di(`<div class="bien">Listo. Abre el taller y el expediente
          ${ficha.numero} estará esperándote con sus constancias.</div>`);
    } catch (e) {
      di(`<div class="mal">${e.message}</div>`);
    } finally {
      boton.disabled = false;
    }
  });
})();
