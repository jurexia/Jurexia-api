/* Las páginas de una extensión NO permiten guiones dentro del HTML: la
 * política de seguridad de Manifest V3 los bloquea, y sin un solo mensaje.
 * Por eso el botón de guardar no hacía nada. Va en su propio fichero. */
const c = document.getElementById("correo");
const ok = document.getElementById("ok");

chrome.storage.local.get("correo").then((v) => { c.value = v.correo || ""; });

document.getElementById("guardar").addEventListener("click", async () => {
  const correo = (c.value || "").trim();
  if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(correo)) {
    ok.textContent = "Ese correo no tiene forma de correo.";
    ok.style.color = "#b3261e";
    return;
  }
  await chrome.storage.local.set({ correo });
  ok.textContent = "Guardado.";
  ok.style.color = "#2e7d4f";
});
