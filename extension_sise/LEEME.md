# Iurexia · Taller desde el Expediente Electrónico

Trae al taller las constancias del expediente que tienes abierto, sin que
teclees el número, ni el tipo de asunto, ni la fecha.

## Instalar (una vez)

1. Chrome → `chrome://extensions`
2. Enciende **Modo de desarrollador** (arriba a la derecha).
3. **Cargar descomprimida** → elige esta carpeta, `extension_sise`.
4. Entra a **iurexia.com** y accede con tu cuenta. El complemento la recoge
   solo: no hay que escribir ningún correo.

## Usar

1. Entra en SISE como siempre y abre tu expediente.
2. Pulsa el icono **Vista Expediente Electrónico**. Se abre el visor.
3. Abajo a la derecha aparece el botón **Iurexia**. Púlsalo.
4. Verás la lista de actuaciones. Vienen marcados los **acuerdos** y las
   **promociones**; las **notificaciones**, no. Desmarca lo que sobre.
5. **Mandar constancias seleccionadas al taller.** Al terminar te dice qué reconoció en cada documento.
6. Abre el taller: el expediente está esperando.

## Qué hace con tus datos

- **No guarda ni envía tu usuario, tu contraseña ni tu sesión del CJF.** Usa la
  sesión que ya tienes abierta, dentro de tu navegador, y sólo para pedirle al
  propio CJF los documentos que hayas marcado.
- **No descarga nada a tu disco.** Los PDF van de la memoria de esa pestaña al
  servidor de Iurexia. Tu carpeta de descargas no se toca.
- A Iurexia viajan los PDF que marcaste y el índice del expediente. Nada más.

## Si algo va mal

| Dice | Qué pasa |
|---|---|
| «Abre un expediente en el visor…» | Estás en el visor pero sin expediente cargado. Ábrelo desde SISE. |
| «El visor no aceptó la sesión (401)» | Caducó. Recarga la página y vuelve a entrar en SISE. |
| «Lo que llegó no es un PDF» | El CJF devolvió una página en vez del documento. Reintenta ese documento. |
| «Hace falta tu sesión de Iurexia» | Abre iurexia.com, entra, y vuelve a pulsar. |

## Topes

40 documentos y 80 MB por envío, y **un solo intento por documento**. Lo que
falle se dice por su nombre al final. Sin reintentos automáticos: una versión
anterior los tenía y dejó 48 ficheros en la carpeta de descargas.
