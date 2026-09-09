# SISE · el mapa, explorado el 9 de septiembre de 2026

Explorado con la sesión de David abierta por él. **No se tecleó ninguna
contraseña ni se descargó ningún fichero**: sólo se navegó y se leyó la
estructura de la página.

## El camino, de la portada a los PDF

    Expediente Electrónico  →  Consultar expediente electrónico
      https://sise.cjf.gob.mx/Sise/ExpedienteElectronico/DefaultConsulta.aspx

Ahí, «Búsqueda de Expedientes por Número de Expediente Asignado». **El circuito,
el tipo de órgano, la materia y el órgano vienen ya fijados por el usuario que
entró**: para el usuario de David, Vigésimo Segundo Circuito · Tribunal
Colegiado · Administrativa y Civil · Tercer Tribunal Colegiado.

Sólo quedan **dos campos**, que son exactamente los dos que él quiere pedir:

    Número de Expediente   (con máscara 00/0000 — no acepta valor programático,
                            hay que teclearlo)
    Tipo de Asuntos        (desplegable)

### El desplegable, con sus valores

    10  Amparo Directo                          → amparo_directo
    11  Amparo en revisión                      → amparo_revision
    12  Conflictos Competenciales
    13  Impedimento
    14  Revisión Contenciosa Administrativa
    15  Queja                                   → queja
    16  Revisión Fiscal                         → revision_fiscal
    17  Conflictos de Acumulación
    20  Reclamaciones
    23  Incidentes de Inejecución
    24  Repetición del Acto Reclamado
    25  Inconformidades
    26  Reconocimiento de Inocencia
    27  Varios

Los cuatro que el taller sabe proyectar están ahí, y el mapeo es directo.

## Lo que devuelve

Buscando «91/2025» con tipo «Revisión Fiscal»:

    Número de Expediente Único Nacional: 40531343
    Número de Expediente Asignado:       91/2025
    Selección de cuaderno → (un cuaderno: «Revisión Fiscal»)

Y al entrar al cuaderno, el **Panel Central de Consultas**
(`/Sise/ExpedienteElectronico/PanelCentralDeConsultas/PanelCentralDeConsultas.aspx`),
que es una tabla con una fila por actuación:

| Fecha de Acuerdo | Fecha Publicación | Contenido Promoción | **Promoción** | Contenido Determinación | **Determinación Judicial** | Acuse | **Notificación o Constancia de Conocimiento** | Video | Constancia FGR |

En el 91/2025:

    21/11/2025 · 24/11/2025 · «Revisión fiscal,» · [PDF] · Acuerdo · [PDF] · [acuse] · [PDF]
    16/01/2026 · 19/01/2026 ·                              Acuerdo · [PDF] · [acuse] · [PDF]

**Esto es lo que elimina el formulario.** De aquí salen, sin teclear nada:

- la **promoción** de la primera fila = el recurso (y, según David, la
  recurrida suele venir en el mismo PDF: hay que partirlo);
- la **determinación judicial** de la primera fila = el auto de admisión, con
  su fecha (21/11/2025);
- la **notificación** = la fecha que hoy se teclea a mano y de la que depende
  todo el cómputo. Es donde Erika se equivocó.

## Cómo se descargan, y por qué esto decide la arquitectura

Los iconos NO son enlaces. Son `input type="image"` de ASP.NET, y cada
descarga es un envío del formulario entero.

Los controles están nombrados con un patrón estable —eso es la buena noticia—:

    ctl00$MainContentPlaceHolder$grvPanelCentral$ctl02$imgPromocion
    ctl00$MainContentPlaceHolder$grvPanelCentral$ctl02$imgDetJud
    ctl00$MainContentPlaceHolder$grvPanelCentral$ctl02$imgAcuDetJud
    ctl00$MainContentPlaceHolder$grvPanelCentral$ctl02$imgNotConsCon
    …$ctl03$…  ← la segunda fila, y así

Las filas van `ctl02`, `ctl03`, … y los cuatro tipos de documento tienen
nombre fijo. Con eso se puede pedir cualquier PDF de cualquier fila.

**La mala noticia**: el formulario lleva un `__VIEWSTATE` de **50,544
caracteres** que hay que devolver íntegro en cada envío. Reconstruir eso a mano
desde nuestro servidor es posible y es frágil: se rompe con cualquier cambio de
pantalla, y aquí un fallo silencioso significa proyectar sobre el expediente
equivocado.

**Lo robusto es manejar un navegador de verdad**, que hace ese baile solo.

## La decisión que falta, y no es técnica

¿De quién es el navegador y de quién la sesión?

- **(A) La del secretario, en su máquina.** Entra a SISE como entra siempre y
  el taller toma de ahí los PDF. Su sesión, su contraseña, que nosotros no
  vemos nunca. Requiere instalar algo en su equipo.
- **(B) La nuestra, en el servidor.** Pedimos usuario y contraseña en el
  recuadro, y un navegador sin ventana entra por él. Es lo que David describió,
  y funciona — pero significa que Iurexia custodia credenciales del Consejo de
  la Judicatura de usuarios nombrados.

Con (B) hay que decidir además si la contraseña se guarda —para no pedirla en
cada asunto— o vive sólo en memoria durante la sesión. Guardarla es cómodo y es
la que carga con la responsabilidad.

---

# LA PUERTA ANCHA · «Vista Expediente Electrónico» (9-sep, tarde)

David, después de que la extensión le descargara 48 ficheros: «no sé si es la
ruta correcta la que estés tomando o haya una más sencilla».

**La había, y estaba a un clic.** El icono morado de la barra del Panel Central
—«Vista Expediente Electrónico»— abre una pestaña en OTRO dominio:

    https://ejusticia.cjf.gob.mx/vistaee/#/ebook/0     «Expediente Electrónico v1.0»

No es el ASP.NET de 2009: es una aplicación web moderna, y trae **el expediente
completo indexado por actuación**. En el 91/2025:

    Revisión Fiscal  (cuaderno)
      Acuerdo del 21/11/2025
        Promoción 1. Revisión Fiscal
        Notificación por oficio 1. RECURRENTE
        Notificación por oficio 2. AUTORIDAD EMISORA
        Notificación por lista 3. OPOSITOR
      Acuerdo del 16/01/2026
        Notificación por lista 1. OPOSITOR
        Notificación por lista 2. AUTORIDAD EMISORA
        Notificación por lista 3. RECURRENTE

## Y habla con una API REST de verdad

Sacadas de `performance.getEntriesByType("resource")`, que guarda todo lo que la
página pidió aunque no estuvieras mirando:

    serviciosvistaee.cjf.gob.mx/wsebook/api/Index/GetIndexDetail      ← el índice
    serviciosvistaee.cjf.gob.mx/wsebook/api/Index/File/{id}           ← un documento
    serviciosvistaee.cjf.gob.mx/wsebook/api/Index/TypeNotebook/{tipo} ← 16 = Revisión Fiscal
    serviciosvistaee.cjf.gob.mx/wsFileManager/api/SingleFile/UserInfo/{usuario}/{sistema}/{organo}
    serviciosinterconexion.pjf.gob.mx/wsDocumentoAPI/api/document/

El `{tipo}` es el MISMO catálogo del desplegable de SISE: 16 = Revisión Fiscal,
10 = Amparo Directo, 11 = Amparo en revisión, 15 = Queja.

## La autenticación es por token, y vive en el navegador del secretario

En `localStorage` y `sessionStorage` de esa pestaña:

    credentialUser   → {issuerName: "CJF_DGETD_IM_01_PROD", token: "eyJ…"}   (JWT)
    currentUser      → {userName, name, id: 92005, OrganId: "2422"}
    EbookParamsData  → {Neun: 40531343, Usuario: 92005, Sistema: 5, Token: "eyJ…"}
    EbookNeunData    → {asuntoNeunId, catMateriaId, catOrganismoId: 2422, catTipoAsunto…}

**El token es una credencial del secretario y no se copia a ningún sitio.** Se
lee donde está —su navegador— y se usa desde ahí. No se manda a Iurexia, no se
guarda, no se registra. Lo que viaja son los PDF, igual que hasta ahora.

## Lo que esto significa para lo construido

Todo el forcejeo con el Panel Central sobra: los paneles de ASP.NET, el
ViewState de 50 kB, la cadena de dos pasos, el depurador de Chrome y el tope de
intentos que hubo que poner después de las 48 descargas. Nada de eso hace falta
si se pide el índice y luego cada fichero por su identificador.

Lo que SÍ se conserva y no fue en balde: el lado de Iurexia —la tabla
`sise_pendientes`, los dos endpoints, el clasificador de documentos por su
texto, el aviso de que falta la recurrida— está probado y no depende de por
dónde lleguen los PDF.

**Diez versiones de extensión para descubrir que había un icono al lado.** La
lección no es sobre SISE: cuando una vía se resiste una y otra vez por sitios
distintos, conviene mirar si el sistema ofrece otra puerta, en vez de forzar
más la que no cede.
