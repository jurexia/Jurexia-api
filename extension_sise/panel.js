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

  /** Un PDF, pidiéndoselo a SISE como se lo pide un clic. */
  async function traerPDF(nombreControl) {
    const f = document.forms[0];
    const cuerpo = new FormData();
    for (const el of f.querySelectorAll("input[type=hidden]")) {
      if (el.name) cuerpo.append(el.name, el.value);
    }
    // Un input[type=image] viaja como `nombre.x` / `nombre.y`: sin eso, el
    // servidor no sabe qué botón se pulsó y devuelve la misma página.
    cuerpo.append(nombreControl + ".x", "8");
    cuerpo.append(nombreControl + ".y", "8");
    const r = await fetch(f.action, {
      method: "POST", body: cuerpo, credentials: "include",
    });
    const buf = await r.arrayBuffer();
    const cabecera = String.fromCharCode(...new Uint8Array(buf.slice(0, 5)));
    if (!cabecera.startsWith("%PDF")) {
      // SISE devuelve la página entera cuando el envío no le cuadra —el
      // ViewState caducó, o la sesión—. Devolver eso como si fuera un PDF
      // haría que el taller proyectara sobre una página de error.
      throw new Error("SISE no devolvió un PDF (¿caducó la sesión?)");
    }
    return new Blob([buf], { type: "application/pdf" });
  }

  const barra = document.createElement("div");
  barra.id = "iurexia-barra";
  barra.innerHTML = `
    <h4>Taller de sentencias · Iurexia</h4>
    <p>Trae las constancias de este expediente sin que teclees nada.
       Tu contraseña de SISE no sale de aquí.</p>
    <button id="iurexia-ir">Traer las constancias</button>
    <div id="iurexia-estado"></div>`;
  document.body.appendChild(barra);

  const estado = barra.querySelector("#iurexia-estado");
  const boton = barra.querySelector("#iurexia-ir");
  const di = (h) => { estado.innerHTML = h; };

  boton.addEventListener("click", async () => {
    boton.disabled = true;
    try {
      const ficha = fichaDelExpediente();
      const filas = actuaciones();
      if (!ficha.numero || !filas.length) {
        throw new Error("No reconozco esta pantalla. Ábrela desde el Panel "
                      + "Central de Consultas de un expediente.");
      }
      di(`<div>Expediente <b>${ficha.numero}</b> · ${filas.length} actuaciones</div>`);

      // QUÉ SE TRAE. La promoción de la PRIMERA fila que la tenga —es el
      // escrito que abre el asunto y donde viene lo recurrido—, y de esa misma
      // fila el acuerdo y la notificación. Las demás filas viajan como lista,
      // sin descargar, para que el secretario elija si hace falta más.
      const conPromocion = filas.find((f) => f.docs.promocion);
      const base = conPromocion || filas[0];
      const aTraer = [
        ["promocion", base.docs.promocion, "el escrito que abre el asunto"],
        ["determinacion", base.docs.determinacion, "el acuerdo de admisión"],
        ["notificacion", base.docs.notificacion, "la notificación"],
      ].filter(([, n]) => n);

      const archivos = {};
      for (const [clave, control, comoSeLlama] of aTraer) {
        di(estado.innerHTML + `<div class="doc"><span>${comoSeLlama}</span><span>…</span></div>`);
        try {
          archivos[clave] = await traerPDF(control);
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
      d.append("actuaciones_json", JSON.stringify(filas.map((f) => ({
        ctl: f.ctl, acuerdo: f.fechaAcuerdo, publicacion: f.fechaPublicacion,
        promocion: f.contenidoPromocion, determinacion: f.contenidoDeterminacion,
      }))));
      for (const [k, b] of Object.entries(archivos)) d.append(k, b, `${k}.pdf`);

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
