# Taller desde SISE · cómo instalarla

Una vez por equipo. No pide, no guarda y no envía tu contraseña de SISE.

1. Abre Chrome y ve a `chrome://extensions`.
2. Activa **Modo de desarrollador** (arriba a la derecha).
3. Pulsa **Cargar descomprimida** y elige esta carpeta:
   `IUREXIA-MAC/jurexia-api-git/extension_sise`
4. En la extensión, pulsa **Detalles** y activa **«Permitir acceso a URL de
   archivo»**. Sin esa casilla no puede leer el PDF que SISE descarga, y no
   hay manera de saltársela: Chrome no deja a una extensión leer el disco por
   defecto, y hace bien.
5. En la extensión, entra a **Opciones** y guarda tu correo de Iurexia.
   Es lo único que se le pide: sirve para saber a qué cuenta mandar el asunto.

## Por qué pulsa en vez de pedir los ficheros por detrás

Se intentó cuatro veces imitar la petición que hace el clic, y SISE la rechazó
—la última con el filtro de IIS: «request filtering is configured to deny
double escape sequences»—. Una petición hecha por un guion no se parece a una
navegación por más cabeceras que se le copien. Así que la extensión **pulsa el
archivero de verdad** y recoge lo que el navegador descarga. Verás los PDF
aparecer en tu carpeta de descargas: es normal, y es lo que los hace fiables.

## Cómo se usa

1. Entra a SISE como siempre, con tu usuario.
2. Expediente Electrónico → Consultar expediente electrónico.
3. Teclea el número, elige el tipo de asunto, Buscar, y entra al cuaderno.
4. En el **Panel Central de Consultas** aparece abajo a la derecha el recuadro
   de Iurexia. Pulsa **Traer las constancias**.
5. Abre el taller: el expediente estará esperando con sus documentos.

## Qué se trae, y qué no

Se trae, de la primera actuación que tenga promoción:

- el **escrito que abre el asunto** (la promoción) — de ahí sale lo recurrido;
- el **acuerdo** de esa actuación — el auto de admisión, con su fecha;
- la **notificación** — el documento del que sale la fecha que manda el cómputo.

Las fechas de la tabla viajan como **pista**, no como dato: la de notificación
y la de presentación se leen del PDF y las confirmas tú. De esas dos depende el
cómputo entero, y una equivocada deja el proyecto en extemporáneo sin avisar.

## Al actualizar la extensión, RECARGA LA PÁGINA

Chrome no reinyecta el guion en las pestañas que ya estaban abiertas: la barra
que se ve sigue siendo la vieja y su botón ya no responde. Después de pulsar
recargar en `chrome://extensions`, **recarga también la pestaña de SISE** (F5).
La barra dice su versión al lado del título; si no coincide con la que
instalaste, es la vieja.

## Si algo falla

- **«SISE no devolvió un PDF»** — casi siempre es que caducó la sesión de SISE.
  Vuelve a entrar y pulsa otra vez.
- **«Falta tu correo de Iurexia»** — Opciones de la extensión, guárdalo una vez.
- **No aparece el recuadro** — sólo sale en el Panel Central de Consultas de un
  expediente, no en el buscador ni en la portada.
