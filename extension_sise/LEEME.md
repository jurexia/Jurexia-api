# Taller desde SISE · cómo instalarla

Una vez por equipo. No pide, no guarda y no envía tu contraseña de SISE.

1. Abre Chrome y ve a `chrome://extensions`.
2. Activa **Modo de desarrollador** (arriba a la derecha).
3. Pulsa **Cargar descomprimida** y elige esta carpeta:
   `IUREXIA-MAC/jurexia-api-git/extension_sise`
4. En la extensión, entra a **Opciones** y guarda tu correo de Iurexia.
   Es lo único que se le pide: sirve para saber a qué cuenta mandar el asunto.

## Chrome avisará de que «se está depurando este navegador»

Es esperado y dura sólo mientras trae las constancias. La extensión usa el
depurador de Chrome para leer el PDF de la respuesta del clic, y ése es el
aviso que Chrome pone —con razón— cuando una extensión mira el tráfico de una
pestaña. Al terminar lo suelta sola.

## Por qué por ahí y no de otra manera

Se intentó cuatro veces imitar la petición del clic y SISE la rechazó: el
filtro de IIS responde «request filtering is configured to deny double escape
sequences», porque una petición de guion no se parece a una navegación. Y
dejar que el fichero se descargue para leerlo del disco tampoco sirve: Chrome
protege el disco.

El clic real es el único que SISE acepta, y el depurador es la única forma de
leer su respuesta. De paso, los PDF ya NO se acumulan en tu carpeta de
descargas: se capturan antes.

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
