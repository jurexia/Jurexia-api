// ═══════════════════════════════════════════════════════════════════════════
// QUIÉN ERES, SIN PREGUNTÁRTELO
//
// David: «no sería posible omitir este paso del correo para conectar? bastaría
// un botón adicional de "Mandar constancias seleccionadas al taller"».
//
// Tenía razón. Si ya estás dentro de Iurexia en este mismo navegador, pedirte
// el correo sobra — y además se teclea mal: dos letras cambiadas mandaron sus
// constancias a un sitio donde el taller no las busca, y el envío dijo «Listo»
// igual.
//
// Este guion corre SÓLO en iurexia.com. Lee la sesión que la propia web ya
// guardó ahí y se queda con dos cosas: tu correo, para poder enseñártelo, y el
// token, para que el servidor compruebe por sí mismo de quién son las
// constancias en vez de fiarse de un texto.
//
// NO SE LEE NADA MÁS. Ni el expediente, ni lo que escribes, ni tu contraseña
// —que nunca está aquí: Supabase guarda un token, no la clave—. Y no sale del
// navegador más que hacia el propio Iurexia, que es quien lo emitió.
// ═══════════════════════════════════════════════════════════════════════════

(function () {
  'use strict';

  const CLAVE = 'iurexia-auth';   // el storageKey que fija la web

  function leerSesion() {
    let crudo = null;
    try { crudo = localStorage.getItem(CLAVE); } catch (e) { return null; }
    if (!crudo) return null;
    // Supabase guarda a veces el JSON en claro y a veces prefijado en base64.
    // Aguantar las dos formas cuesta tres líneas y evita que un cambio de
    // versión de su biblioteca deje esto mudo sin que nadie se entere.
    if (crudo.startsWith('base64-')) {
      try { crudo = atob(crudo.slice(7)); } catch (e) { return null; }
    }
    let j;
    try { j = JSON.parse(crudo); } catch (e) { return null; }
    const s = j && (j.currentSession || j.session || j);
    const correo = s && s.user && s.user.email;
    const token = s && s.access_token;
    if (!correo || !token) return null;
    return { correo: String(correo).toLowerCase(), token: String(token),
             caduca: Number(s.expires_at || 0) };
  }

  function guardar() {
    const s = leerSesion();
    try {
      if (s) chrome.storage.local.set({ sesion: s });
      // Si no hay sesión, se BORRA lo guardado: dejar la de antes haría que el
      // complemento siguiera enviando a nombre de quien cerró la sesión.
      else chrome.storage.local.remove('sesion');
    } catch (e) { /* la pestaña se está cerrando */ }
  }

  guardar();
  // La sesión se renueva sola cada hora; mirar de vez en cuando mientras la
  // pestaña esté abierta evita que el complemento se quede con una caducada.
  setInterval(guardar, 60000);
  window.addEventListener('focus', guardar);
})();
