// ═══════════════════════════════════════════════════════════════════════════
// EL PANEL, DENTRO DEL EXPEDIENTE ELECTRÓNICO
//
// La versión anterior forcejeaba con el Panel Central de SISE: ViewState de
// 50 kB, iconos que navegan en vez de devolver un PDF, recargas que mataban al
// script a media captura. Diez versiones, y una de ellas dejó 48 ficheros en
// las descargas de David.
//
// Esta no hace nada de eso. El visor «Vista Expediente Electrónico» habla con
// una API de verdad, y esta extensión le habla igual:
//
//   POST /wsebook/api/Index/GetIndexDetail  → el índice del expediente
//   POST /wsebook/api/File/Dowload          → un documento, en base64
//                                             (sí, «Dowload»: así se llama)
//
// TRES REGLAS QUE NO SE ROMPEN:
//
//  1. NADA SE DESCARGA A LA MÁQUINA. No se usa chrome.downloads. Los PDF van
//     de la memoria de esta pestaña al servidor de Iurexia y de ahí a nadie
//     más. La carpeta de descargas del secretario no se toca.
//
//  2. EL TOKEN NO SALE DEL NAVEGADOR. La sesión del CJF vive en el
//     sessionStorage de esta pestaña. Se lee ahí y se usa ahí, en la cabecera
//     Authorization hacia el propio CJF. No se copia, no se guarda, no se
//     manda a Iurexia, no se escribe en ningún registro.
//
//  3. NADA SE ENVÍA SIN QUE SE VEA ANTES. El índice se enseña con casillas.
//     El secretario mira, desmarca lo que sobra y pulsa. Sin ese clic no viaja
//     un solo byte.
// ═══════════════════════════════════════════════════════════════════════════

(function () {
  'use strict';

  if (window.__iurexiaPanel) return;   // no duplicar si se reinyecta
  window.__iurexiaPanel = true;

  const API = 'https://jurexia-api.onrender.com';
  const WS  = 'https://serviciosvistaee.cjf.gob.mx/wsebook/api';

  // TOPES DUROS. La versión que descargó 48 ficheros no tenía ninguno. Un
  // instrumento sin freno no es un instrumento: es un accidente esperando.
  const TOPE_DOCS = 40;          // documentos por envío
  const TOPE_BYTES = 80 * 1024 * 1024;

  // Qué es cada `tipo` del índice, y si entra por omisión. La regla es la que
  // dijo David: el escrito que abre el asunto trae casi todo, y los acuerdos
  // de admisión y turno son indispensables. Las notificaciones no aportan al
  // proyecto, así que se enseñan pero llegan desmarcadas.
  const TIPOS = {
    0: { nombre: 'Carátula',      marcado: false },
    1: { nombre: 'Acuerdo',       marcado: true  },
    2: { nombre: 'Promoción',     marcado: true  },
    3: { nombre: 'Notificación',  marcado: false },
  };

  const $ = (t, cls, txt) => {
    const e = document.createElement(t);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;   // textContent SIEMPRE: el índice
    return e;                               // trae <b> y <em> del servidor
  };

  const limpio = (s) => String(s || '').replace(/<[^>]*>/g, ' ')
                                       .replace(/\s+/g, ' ').trim();

  const ddmmaaaa = (iso) => {
    const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(iso || ''));
    if (!m || m[1] === '0001') return '';
    return `${m[3]}/${m[2]}/${m[1]}`;
  };

  // ── LA SESIÓN DEL VISOR ───────────────────────────────────────────────────
  // Vive en el sessionStorage de esta pestaña. Si no está, es que el visor no
  // ha abierto ningún expediente todavía, y hay que decirlo con esas palabras
  // en vez de fallar en silencio.
  function sesion() {
    let P = null, N = null;
    try { P = JSON.parse(sessionStorage.getItem('EbookParamsData') || 'null'); } catch (e) {}
    try {
      N = JSON.parse(sessionStorage.getItem('EbookNeunData')
                     || localStorage.getItem('EbookNeunData') || 'null');
    } catch (e) {}
    if (!P || !P.Token || !P.Neun) return null;
    return {
      neun: P.Neun,
      usuario: P.Usuario,
      sistema: P.Sistema,
      token: P.Token,
      organismo: (N && N.catOrganismoId) || null,
      numero: (N && N.asuntoAlias) || '',
      tipoAsunto: (N && N.tipoAsunto) || '',
      organo: limpio(N && N.organo),
      ingreso: ddmmaaaa(N && N.fechaIngreso),
    };
  }

  const cabeceras = (s) => ({
    'Content-Type': 'application/json',
    'Accept': 'application/json, text/plain, */*',
    'Authorization': 'Bearer ' + s.token,
  });

  async function indice(s) {
    const r = await fetch(`${WS}/Index/GetIndexDetail`, {
      method: 'POST', headers: cabeceras(s),
      body: JSON.stringify({
        Neun: s.neun, Usuario: s.usuario,
        sistema: s.sistema, catOrganismoId: s.organismo,
      }),
    });
    if (r.status === 401)
      throw new Error('El visor no aceptó la sesión (401). Suele significar '
                    + 'que caducó: recarga esta página y vuelve a entrar.');
    if (!r.ok) throw new Error(`El índice respondió ${r.status}.`);
    const j = await r.json();
    if (!Array.isArray(j)) throw new Error('El índice no vino como lista.');
    return j.filter(e => e && e.nombreArchivo);   // la carátula no tiene fichero
  }

  // UN DOCUMENTO. La respuesta trae {base64File: [partes], size, parts}: los
  // ficheros grandes vienen troceados, y quedarse con la primera parte daría
  // un PDF truncado que abre y engaña. Se concatenan todas.
  async function documento(s, e) {
    const r = await fetch(`${WS}/File/Dowload`, {
      method: 'POST', headers: cabeceras(s),
      body: JSON.stringify({
        sistema: s.sistema, Usuario: s.usuario, Neun: s.neun,
        TipoArchivo: e.tipo, ID: '', nombre: e.nombreArchivo,
        catOrganismoId: e.catOrganismoId, extesionId: e.extesionId,
      }),
    });
    if (!r.ok) throw new Error(`el servidor respondió ${r.status}`);
    const j = await r.json();
    const partes = Array.isArray(j.base64File) ? j.base64File
                 : (j.base64File ? [j.base64File] : []);
    const b64 = partes.join('');
    if (!b64) throw new Error('vino vacío');
    // UN PDF EMPIEZA POR %PDF, que en base64 es JVBER. Si el CJF devolvió una
    // página de error, esto lo caza aquí y no en el taller tres pasos después.
    if (!b64.startsWith('JVBER'))
      throw new Error('lo que llegó no es un PDF');
    const bin = atob(b64);
    const u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return new Blob([u8], { type: 'application/pdf' });
  }

  // ── EL CORREO ─────────────────────────────────────────────────────────────
  // Se guarda en chrome.storage.local, que sobrevive a recargas y a reinicios.
  // La página de opciones sigue existiendo, pero preguntarlo aquí evita el
  // viaje que la vez pasada acabó en «no deja guardar el correo».
  const leerCorreo = () => new Promise(res => {
    try { chrome.storage.local.get(['correo'], d => res((d && d.correo) || '')); }
    catch (e) { res(''); }
  });
  const guardarCorreo = (c) => new Promise(res => {
    try { chrome.storage.local.set({ correo: c }, () => res(true)); }
    catch (e) { res(false); }
  });

  // ── LA PANTALLA ───────────────────────────────────────────────────────────
  const caja = $('div', 'iux-caja');
  const boton = $('button', 'iux-boton', 'Iurexia');
  boton.type = 'button';
  const panel = $('div', 'iux-panel');
  panel.hidden = true;
  caja.append(boton, panel);
  document.documentElement.appendChild(caja);

  // TRES BANDERAS, Y VIVEN EN EL MÓDULO. Sin ellas el panel es reentrante: la
  // revisión adversarial reconstruyó la secuencia entera. El secretario pulsa
  // «Enviar», se impacienta viendo «Trayendo 3 de 12», cierra el panel y lo
  // reabre. Cerrar no detenía nada, así que la primera pasada seguía bajando a
  // ciegas; y como al repintar nacía un botón «Enviar» NUEVO —habilitado,
  // porque el `disabled` se puso en el botón anterior, ya desprendido del
  // árbol— volvía a pulsar. Dos pasadas, dos POST, y el upsert por
  // email+numero deja la última: si la segunda traía menos constancias, la
  // buena se perdía SIN UN SOLO MENSAJE DE ERROR.
  //
  // No se aborta el envío en vuelo a propósito. Si el aborto cayera dentro del
  // POST a Iurexia, el servidor podría haber guardado ya y la pantalla diría
  // «cancelado» sobre algo que sí quedó: cambiar un fallo mudo por otro peor.
  // Lo que se impide es EMPEZAR el segundo.
  let abierto  = false;
  let pintando = false;
  let enviando = false;

  boton.addEventListener('click', () => {
    abierto = !abierto;
    panel.hidden = !abierto;
    // Con un envío vivo, reabrir enseña ESE panel y su progreso. Repintar
    // borraría el «Trayendo N de M» —y también el acuse de un envío que ya
    // terminó bien, que es justo lo que empujaba a reenviar.
    if (abierto && !enviando && !pintando) pintar();
  });

  const nota = (txt, clase) => {
    const p = $('p', 'iux-nota ' + (clase || ''), txt);
    return p;
  };

  async function pintar() {
    if (pintando) return;
    pintando = true;
    try {
      await _pintar();
    } finally {
      pintando = false;
    }
  }

  async function _pintar() {
    panel.textContent = '';
    const s = sesion();
    if (!s) {
      panel.append(
        $('h3', 'iux-tit', 'Iurexia'),
        nota('Abre un expediente en el visor y vuelve a pulsar. Todavía no hay '
           + 'ninguno cargado en esta pestaña.', 'iux-aviso'));
      return;
    }

    panel.append($('h3', 'iux-tit', `${s.numero || 'Expediente'} · ${s.tipoAsunto || ''}`));
    if (s.organo) panel.append($('p', 'iux-sub', s.organo));
    if (s.ingreso) panel.append($('p', 'iux-sub', 'Ingreso: ' + s.ingreso));

    const cargando = nota('Leyendo el índice del expediente…');
    panel.append(cargando);

    let lista;
    try {
      lista = await indice(s);
    } catch (e) {
      cargando.remove();
      panel.append(nota(String(e.message || e), 'iux-error'));
      return;
    }
    cargando.remove();

    if (!lista.length) {
      panel.append(nota('El índice no trae ningún documento descargable.', 'iux-aviso'));
      return;
    }

    // El correo: si no está, se pide aquí mismo.
    const correoGuardado = await leerCorreo();
    let entradaCorreo = null;
    if (!correoGuardado) {
      const fila = $('div', 'iux-correo');
      entradaCorreo = document.createElement('input');
      entradaCorreo.type = 'email';
      entradaCorreo.placeholder = 'nombre@ejemplo.mx';
      fila.append($('label', null, 'El correo de TU CUENTA de Iurexia'), entradaCorreo);
      fila.append($('span', 'iux-pista',
        'Tiene que ser exactamente el mismo con el que entras a Iurexia. Si '
      + 'cambias una letra, las constancias llegan pero no aparecen en tu taller.'));
      panel.append(fila);
    }

    const ul = $('div', 'iux-lista');
    const filas = lista.map((e, i) => {
      const t = TIPOS[e.tipo] || { nombre: 'Documento ' + e.tipo, marcado: false };
      const fila = $('label', 'iux-fila');
      const chk = document.createElement('input');
      chk.type = 'checkbox';
      chk.checked = t.marcado;
      const txt = $('span', 'iux-txt');
      txt.append($('span', 'iux-tipo', t.nombre));
      txt.append($('span', 'iux-desc', limpio(e.description) || e.nombreArchivo));
      const f = ddmmaaaa(e.fechaAuto);
      if (f) txt.append($('span', 'iux-fecha', f));
      fila.append(chk, txt);
      ul.append(fila);
      return { chk, entrada: e };
    });
    panel.append(ul);

    // A QUÉ CUENTA VAN. Un correo escrito a mano con una letra cambiada manda
    // las constancias a un sitio donde nadie las busca, y el envío dice
    // «Listo» igual: pasó con jmd en vez de jdm, y se perdió media tarde
    // buscando el fallo en el sitio equivocado. Verlo antes de pulsar cuesta
    // una línea.
    if (correoGuardado) {
      const fila = $('div', 'iux-cuenta');
      fila.append($('span', null, 'Van a la cuenta '),
                  $('strong', null, correoGuardado));
      const cambiar = $('button', 'iux-cambiar', 'cambiar');
      cambiar.type = 'button';
      cambiar.addEventListener('click', async () => {
        await guardarCorreo('');
        pintar();
      });
      fila.append(cambiar);
      panel.append(fila);
    }

    const estado = nota('');
    const enviar = $('button', 'iux-enviar', 'Enviar a Iurexia');
    enviar.type = 'button';
    panel.append(enviar, estado);

    enviar.addEventListener('click', async () => {
      if (enviando) return;
      const correo = (correoGuardado || (entradaCorreo && entradaCorreo.value) || '')
                       .trim().toLowerCase();
      if (!correo || correo.indexOf('@') < 0) {
        estado.className = 'iux-nota iux-error';
        estado.textContent = 'Falta el correo de tu cuenta de Iurexia.';
        return;
      }
      if (!correoGuardado) await guardarCorreo(correo);

      const elegidos = filas.filter(f => f.chk.checked).map(f => f.entrada);
      if (!elegidos.length) {
        estado.className = 'iux-nota iux-error';
        estado.textContent = 'No has marcado ninguna constancia.';
        return;
      }
      if (elegidos.length > TOPE_DOCS) {
        estado.className = 'iux-nota iux-error';
        estado.textContent = `Son ${elegidos.length} documentos y el tope es `
                           + `${TOPE_DOCS}. Marca menos.`;
        return;
      }

      enviando = true;
      enviar.disabled = true;
      estado.className = 'iux-nota';
      try {

      // UNA PASADA, UN INTENTO POR DOCUMENTO. Sin reintentos y sin bucle: lo
      // que falle se dice al final por su nombre, y el secretario decide.
      const bajados = [];
      const fallos = [];
      let bytes = 0;
      for (let i = 0; i < elegidos.length; i++) {
        const e = elegidos[i];
        const rotulo = limpio(e.description) || e.nombreArchivo;
        estado.textContent = `Trayendo ${i + 1} de ${elegidos.length}: ${rotulo}`;
        try {
          const blob = await documento(s, e);
          bytes += blob.size;
          if (bytes > TOPE_BYTES) {
            fallos.push(`${rotulo} — se llegó al tope de tamaño del envío`);
            break;
          }
          bajados.push({ entrada: e, blob, rotulo });
        } catch (err) {
          fallos.push(`${rotulo} — ${err.message || err}`);
        }
      }

      // La promoción es lo que abre el asunto: sin ella el taller no tiene qué
      // resolver, y el servidor la exige. Si no hay ninguna marcada, se manda
      // como promoción el primer documento elegido y el servidor —que lee el
      // texto— dirá qué es en realidad.
      const iPro = bajados.findIndex(b => b.entrada.tipo === 2);
      if (!bajados.length) {
        estado.className = 'iux-nota iux-error';
        estado.textContent = 'No se pudo traer ningún documento. '
                           + fallos.join(' · ');
        return;
      }
      const principal = bajados[iPro >= 0 ? iPro : 0];
      const resto = bajados.filter(b => b !== principal);

      estado.textContent = `Enviando ${bajados.length} constancias a Iurexia…`;

      const fd = new FormData();
      fd.append('user_email', correo);
      fd.append('numero', s.numero || String(s.neun));
      fd.append('tipo_sise', s.tipoAsunto || '');
      fd.append('organo', s.organo || '');
      fd.append('presentacion_sise', s.ingreso || '');
      fd.append('actuaciones_json', JSON.stringify(lista.map(e => ({
        tipo: (TIPOS[e.tipo] || {}).nombre || e.tipo,
        descripcion: limpio(e.description),
        fecha: ddmmaaaa(e.fechaAuto),
        parte: limpio(e.nombreParte),
        archivo: e.nombreArchivo,
        elegido: elegidos.indexOf(e) >= 0,
      }))));
      fd.append('promocion', principal.blob, principal.entrada.nombreArchivo);
      for (const b of resto) fd.append('acuerdos', b.blob, b.entrada.nombreArchivo);

      try {
        const r = await fetch(`${API}/taller/desde-sise`, { method: 'POST', body: fd });
        const j = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(j.detail || `el servidor respondió ${r.status}`);
        estado.className = 'iux-nota iux-bien';
        panel.querySelectorAll('.iux-resultado').forEach(n => n.remove());
        const res = $('div', 'iux-resultado');
        res.append($('p', 'iux-bien',
          `Listo: ${bajados.length} constancias del ${s.numero}, enviadas a `
          + `${correo}. Abre el taller —si ya lo tenías abierto, vuelve a esa `
          + 'pestaña— y te estarán esperando.'));
        for (const d of (j.inventario || []))
          res.append($('p', 'iux-item',
            `${d.que}: ${d.tipo}${d.caracteres ? ` (${d.caracteres} caracteres)` : ''}`));
        for (const a of (j.avisos || []))
          res.append($('p', 'iux-aviso', a));
        for (const f of fallos)
          res.append($('p', 'iux-error', 'No se pudo traer ' + f));
        estado.textContent = '';
        panel.append(res);
      } catch (err) {
        estado.className = 'iux-nota iux-error';
        estado.textContent = 'No se pudo enviar: ' + (err.message || err);
      }
      } finally {
        // TODA salida pasa por aquí, incluidos los `return` tempranos de las
        // ramas de error. Una bandera que se queda colgada deja el panel
        // inservible hasta recargar, y eso es peor que el fallo que evita.
        enviando = false;
        enviar.disabled = false;
      }
    });
  }
})();
