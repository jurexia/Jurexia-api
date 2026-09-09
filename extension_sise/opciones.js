// EN FICHERO APARTE, no en un <script> dentro del HTML: la política de las
// extensiones MV3 bloquea el código en línea, y esa fue justo la razón por la
// que la vez pasada «no dejaba guardar el correo».
const campo = document.getElementById('correo');
const dicho = document.getElementById('dicho');

chrome.storage.local.get(['correo'], (d) => { campo.value = (d && d.correo) || ''; });

document.getElementById('guardar').addEventListener('click', () => {
  const c = (campo.value || '').trim().toLowerCase();
  if (!c || c.indexOf('@') < 0) {
    dicho.style.color = '#a11';
    dicho.textContent = 'Escribe un correo válido.';
    return;
  }
  chrome.storage.local.set({ correo: c }, () => {
    dicho.style.color = '#1b6b3a';
    dicho.textContent = chrome.runtime.lastError
      ? 'No se pudo guardar: ' + chrome.runtime.lastError.message
      : 'Guardado.';
  });
});
