# Evolución del taller de sentencias

**Para qué es este documento.** Para no perderme. El taller tiene muchas piezas
y yo he llegado a cambiar una creyendo que arreglaba otra. Aquí está lo que el
taller hace HOY después de generar el adelanto, comprobado leyendo el código, y
debajo el registro de cada cambio con su estado: **comprobado** o **pendiente**.

**Regla de uso: leer esto ANTES de tocar nada del taller. Anotar cada cambio
DESPUÉS de comprobarlo, no después de escribirlo.** Un cambio escrito y no
comprobado se anota como pendiente, con esa palabra.

Última revisión del mapa: 6 de septiembre de 2026.

---

## 1. El recorrido, tal como está hoy

Después del adelanto la pantalla pasa por cuatro estados
(`Paso` en `taller/page.tsx:44`): `ficha` → `adelanto` → **`acervo`** →
`criterio` → `proyecto`. Lo que sigue es de `acervo` en adelante.

### Paso 1 · Buscar solución jurídica  → `POST /taller/consultar`

`main.py:27550`. Botón rojo pulsante en `taller/page.tsx:409`.

Llama a `redactor_adelanto.consultar()`, que busca en Qdrant la jurisprudencia y
las normas de los problemas del asunto, más un **sondeo** del acervo de
sentencias que produce la jurimetría (`prediccion`: «Conceder, 82% de 50
sentencias»). Todo eso se guarda en la sesión como `material` y se devuelve a la
pantalla: tesis, normas, los problemas enteros y los avisos.

La entidad estatal viene del encargo. Si va vacía **no se consulta ley estatal y
se avisa** — deliberado: inventar una entidad era peor.

### Paso 2 · Proponer solución  → `POST /taller/proponer`

`main.py:27723` → `fase5_propuesta.py`.

Motor: **`gpt-5.6-luna`**, esfuerzo `high` (`fase5_propuesta.py:43`). Se le pasa
el material del RAG, el resumen del acto y de los conceptos, y el contexto que
haya escrito el secretario o salga de las constancias.

Devuelve **una propuesta por cada problema**, con `sentido`, `razon`, `apoyos`
(registros del acervo), `confianza` y `alcanza`. Más:

- `revisar()` comprueba sin modelo que los registros citados existan en el
  acervo, y avisa si la propuesta se apoya en algo inventado o en nada.
- `criterios_json`: lo que la pantalla devuelve al resolver para aceptar la
  propuesta tal cual. Lleva jerarquía y predicción sembradas antes.

### Paso 3 · Decidir  → `SolucionDelAsunto.tsx`  *(reestructurado 6-sep)*

**Esta es la pantalla donde el secretario se perdía.** Antes se le volcaba el
acervo entero. Ahora se lee de arriba abajo:

1. **El asunto en cuatro párrafos** — hechos, qué se resolvió, qué se dice en
   contra, y el tema del que cuelga todo.
2. **Las dos vías, lado a lado** — la propuesta y «resolver al revés». La
   contraria viene ya escrita del motor: marcarla es instantáneo.
3. **«Por dónde se cae»**, en ámbar, bajo la propuesta.
4. **La razón, editable**, con «volver a la del motor».
5. **La lista de comprobación** — cada tema con su suerte en la vía elegida.

El acervo pasa a un pliegue, «En qué se apoya»: sigue siendo lo que impide
citar de memoria, pero va detrás de la decisión.

Cambiar de vía **sustituye la razón** aunque estuviera editada, y se avisa
antes: son resoluciones opuestas y quedarse con la razón de la otra es la
incongruencia que costó el engrose del ADC 380/2025.

El modo «por problema» sigue existiendo para quien lo quiera.

### Paso 4 · Generar el proyecto  → `POST /taller/resolver` o `/resolver/stream`

`main.py:28034` y `main.py:27837`. Corre el estudio de fondo (`fase6_estudio`) y
compone el .docx. `GET /taller/descargar` lo entrega.

---

## 2. Defectos verificados

### D-1 · No existe propuesta de solución GLOBAL  · grave · **ARREGLADO 6-sep**

El modelo propone **por problema**. No hay en ninguna parte una propuesta del
sentido del asunto entero. Lo que hoy se ofrece como «solución global» es la
propuesta del problema principal reutilizada:
`VentanaCriterio.tsx:101` — `propuesta?.propuestas?.[iPrincipal]`.

Si el asunto tiene tres problemas, inserta el sentido de uno solo y lo llama
global. **Es el hueco que David señaló.**

### D-2 · Las propuestas se emparejan con los problemas por POSICIÓN  · grave · **ARREGLADO 6-sep**

`page.tsx:210` (`p.propuestas[i]`) y `VentanaCriterio.tsx:101`
(`propuestas[iPrincipal]`).

`fase5_propuesta.py:499` **admite explícitamente** que el modelo puede devolver
menos propuestas que problemas — hay un aviso escrito para ese caso. Y el orden
lo pone el modelo, no el código. Así que la posición *i* de una lista no es la
posición *i* de la otra.

Consecuencia: con tres problemas y dos propuestas devueltas, el secretario puede
ver el sentido del problema B pegado al problema C. La clase `Propuesta` **tiene
un campo `problema`** con el texto: esa es la clave correcta, y no se usa.

Anterior a los cambios de esta semana.

---

## 3. Registro de cambios

| # | Fecha | Qué | Estado |
|---|-------|-----|--------|
| 1 | 2-sep-2026 | Botón del acervo renombrado a «Buscar solución jurídica», rojo pulsante (`page.tsx:409`, `globals.css` `@keyframes latido`) | **Comprobado** — desplegado, `tsc` y build limpios |
| 2 | 2-sep-2026 | `BloqueGlobal`: problema principal + predicción del acervo, pastillas de sentido, consecuencia por problema, razón editable | **Comprobado a medias** — se ve, pero se apoya en D-1 y D-2 |
| 3 | 2-sep-2026 | Botón «Insertar la solución propuesta» | **No cumple lo pedido** — inserta la propuesta del principal, no una global (D-1) |
| 4 | 6-sep-2026 | Sello de citas: `rubroCorresponde` se importa, no sólo se reexporta (`SelloCitas.tsx`) | **Comprobado** — desplegado `fb66162`; antes lanzaba `ReferenceError` que el `catch` convertía en «sin comprobar» |
| 5 | 6-sep-2026 | **D-1** · El modelo propone la solución del **asunto entero** en la misma llamada: sentido, razón, de qué problema cuelga, qué les pasa a los demás, apoyos y `en_contra` (`fase5_propuesta.Global`, `main.py` devuelve `global`) | **Comprobado en lógica, PENDIENTE en producción** — ver abajo |
| 6 | 6-sep-2026 | **D-2** · `emparejar()` casa propuesta y problema por **texto**, y el servidor devuelve la lista ya alineada con huecos declarados | **Comprobado** — prueba determinista: con 3 problemas y 2 propuestas desordenadas, antes 2 de 3 mostraban el sentido de otro problema; ahora ninguno |
| 7 | 6-sep-2026 | Panel «Lo que propone el motor» con el recuadro ámbar **«Por dónde se cae»** (`VentanaCriterio.tsx`) | **Comprobado** que compila y despliega; **pendiente** verlo con un asunto real |

---

## 4. Lo que falta comprobar

**El campo `global` con el modelo de verdad.** La lógica está probada con un
JSON fabricado por mí: `_leer` lo lee, `Global` lo arma, la pantalla lo pinta.
Lo que **no** está probado es que `gpt-5.6-luna` devuelva ese campo cuando se le
pide de verdad, con un expediente real y el prompt entero.

Puede pasar que lo omita —el prompt ya es largo y esto va al final, que es donde
se pierden las instrucciones; ya ocurrió con la pregunta expresa del estudio—.
Si lo omite, no se rompe nada: sale el aviso «El motor no propuso una solución
para el asunto entero» y el secretario fija el sentido a mano, como hasta ayer.

**Cómo comprobarlo:** correr un asunto real hasta la propuesta y mirar si
aparece el panel «Lo que propone el motor» con su recuadro ámbar. Si no
aparece, el arreglo es subir ese bloque del prompt más arriba, antes de las
reglas — no volver a escribirlo.

Anotar aquí el resultado.
| 8 | 6-sep-2026 | **Reestructura.** El motor devuelve contexto en prosa, la vía contraria ya escrita y la lista de comprobación (`fase5_propuesta.Global`) | **Comprobado en lógica**; pendiente con el modelo real |
| 9 | 6-sep-2026 | `completar_checklist()`: la exhaustividad se **comprueba** contra los problemas de las fases, no se le pide al modelo | **Comprobado** — con 3 problemas y lista de 2, completa el tercero y avisa por su nombre |
| 10 | 6-sep-2026 | `SolucionDelAsunto.tsx` sustituye al volcado; el acervo pasa a un pliegue | **Comprobado en navegador** — al marcar la vía contraria cambian a la vez sentido, razón y las 3 líneas de la lista |
| 11 | 6-sep-2026 | Al llegar la propuesta se entra directo a la pantalla de decisión con el sentido y la razón puestos | **Comprobado** que compila; pendiente verlo con un asunto real |

---

## 5. Primera prueba con un asunto real — revisión 410/2026 · 6-sep

Corrida completa contra producción con los dos PDF: adelanto → consultar (41
tesis, 37 normas, 3 problemas) → proponer (95 s, `gpt-5.6-luna`).

**Funcionó:**

- El **contexto** salió con sus cuatro párrafos, específicos y legibles.
- La **vía contraria** salió con sentido DISTINTO —`fundado` frente a
  `infundado`— y argumentada de verdad (representación fraudulenta, revocar el
  sobreseimiento), no como negación de la propuesta.
- Las tres **propuestas por problema** se emparejaron correctamente.

**Falló la lista de comprobación, y era mío.** Devolvió **6 temas para 3
problemas** y marcó tres como «SIN DETERMINAR» que el modelo sí había resuelto.

Causa: las fases escriben el problema como **pregunta** («¿La quejosa podía
reclamar la falta de emplazamiento pese a haber promovido el juicio de
nulidad?») y el modelo escribe el mismo tema como **título** («Falta de
emplazamiento pese a la promoción del juicio de nulidad»). Yo comparaba los
primeros 60 caracteres: no coincidió ninguno.

Arreglado comparando **contenido**: palabras con carga, recortadas a cinco
letras, con umbral del 55%. Calibrado en los dos sentidos contra el texto real.

**Y un hueco que salió al mirar el resultado:** `revisar()` comprobaba los
registros de las propuestas por problema y dejaba fuera **la global y su
alternativa** — que es donde más caro sale, porque la global se acepta de un
botón. `revisar_global()` los comprueba desde ahora.

---

## 6. Lo pendiente

Lo mismo que en el punto 4, ampliado: **hay que correr un asunto real**. La
lógica está probada con datos que fabriqué yo. Lo que no está probado es que
`gpt-5.6-luna` devuelva, en una sola respuesta y con el prompt entero:

- `contexto` (los cuatro párrafos)
- `alternativa` con un sentido **distinto** al de la propuesta
- `checklist` con todos los temas

El prompt creció. Cuanto más se le pide en una respuesta, más fácil es que
omita lo último. Cada omisión está cubierta con su aviso y su degradación
—sin contexto se sigue decidiendo; sin alternativa el botón sale apagado; sin
checklist completo se avisa por nombre— pero **cubierto no es lo mismo que
funcionando**.

Si en la prueba real falta algo, el arreglo NO es reescribir la instrucción:
es partir la llamada en dos o subir ese bloque antes de las reglas. Anotarlo
aquí.
| 12 | 6-sep-2026 | `_mismo_tema()`: los temas se comparan por contenido, no por prefijo | **Comprobado con el texto real del 410/2026** en los dos sentidos: 3 temas sin duplicar, caza el omitido, no funde los parecidos |
| 13 | 6-sep-2026 | `revisar_global()`: los apoyos de la propuesta global y de la alternativa se comprueban contra el acervo | **Comprobado** — 0 avisos con los apoyos reales, 2 con registros inventados |
| 14 | 6-sep-2026 | El verbo del resolutivo se rellena con lo que el motor declaró en el contexto (`resolvio_a_quo(declarado=…)`, campo `resolvio_declarado` de vuelta desde la pantalla) | **Comprobado en el 410/2026 real** — huecos de 4 a 2; salió «Se confirma la sentencia impugnada» y «Se sobresee» |

---

## 7. Proyecto completo del 410/2026 · 6-sep

Generado con sentido global `INFUNDADO` y criterios infundado / infundado /
inoperante. **3,639 palabras, 0 advertencias, 2 huecos** (sólo las dos fechas
de sesión, que van así por diseño).

Estructura completa: V I S T O, cinco resultandos, siete considerandos
—Competencia, Existencia del acto reclamado, Legitimación y oportunidad,
Resolución recurrida y agravios, Antecedentes, Materia de la revisión con las
tres preguntas, Estudio— resolutivos y firmas. Los tres temas aparecen
enumerados en la Materia y contestados en el Estudio.

Congruente: el estudio cierra «se confirma la sentencia recurrida» y los
resolutivos dicen confirmar y sobreseer.

**Lo que salió mal y se arregló en el acto:** el verbo del resolutivo estaba en
hueco. No fallaba el detector —los antecedentes narraban el juicio de nulidad y
nunca decían en qué paró el amparo, así que devolver vacío era honesto—. El
dato estaba en el contexto que escribe el motor y no llegaba a la composición.
Ahora llega. Ver cambio 14.

**Pendiente menor, sin arreglar:** el resolutivo dice «sentencia IMPUGNADA» y
el estudio «sentencia RECURRIDA». Es la misma cosa con dos nombres dentro del
mismo proyecto.

**Y un hallazgo que no es un defecto sino una propiedad:** en tres corridas del
mismo asunto con el mismo material, el sentido global cambió. Corridas 1 y 2:
`infundado`. Corrida 3: `fundado`, y además movió el problema principal del
primero al tercero. Ocurre con `temperature=0` y semilla fija, así que no es el
muestreo: la pregunta admite las dos lecturas.

Esto **confirma que el diseño de dos vías es el correcto**: el motor no es una
autoridad sobre el sentido, es un generador de las dos hipótesis, y quien elige
es el secretario. Lo reproducible es lo demás —la estructura, la suerte de los
accesorios, los apoyos comprobados contra el acervo—.
| 15 | 6-sep-2026 | `_bloque_global`: el estudio ve la objeción, el efecto y de qué cuelga el resultado | **Comprobado en el 410/2026** — párrafos que abordan la objeción: 6 → 13, con una refutación real («No obsta a lo anterior que la recurrente afirme que la representación pudo ser fraudulenta…») |
| 16 | 6-sep-2026 | Cierre duplicado: `_sin_remate_duplicado` + instrucción | **Comprobado** — un solo cierre; la recapitulación con sustancia se conserva |
| 17 | 6-sep-2026 | Fracciones citadas en los dos órdenes y en plural, abarcando el tramo | **Comprobado en local** (VI y VII → las dos; IV a VII → las cuatro); **NO verificado en producción** |
| 18 | 6-sep-2026 | La fórmula contradictoria de «prelación lógica» sale del prompt | **Sin verificar** |
| 19 | 6-sep-2026 | Los dos resolver pasan `jerarquia` y `prediccion` | **Comprobado en código**; sin verificar en salida |

---

## 8. RESUELTO · la respuesta se perdía con dos workers · 7-sep

**Medido, no supuesto.** Dos resoluciones seguidas del 410/2026 murieron en el
cliente con `RemoteDisconnected`, a los **1,449 s** y a los **3,639 s**.

Lo que dicen los registros de Render:

- Arranque real: `gunicorn -w 2 -k uvicorn.workers.UvicornWorker --timeout 240`.
- **El servidor TERMINÓ el trabajo**: `POST /taller/resolver 200`, 4,031
  palabras, 14 avisos.
- **Ni un `WORKER TIMEOUT`, ni una traza, ni un `SIGKILL`.**
- `GET /taller/descargar` responde **404 «No hay documento generado para ese
  expediente en este proceso»**: el .docx quedó en el disco de un worker y la
  petición cayó en el otro.

**Conclusión:** el trabajo se hace, se paga, y no llega. La pantalla le diría al
secretario que falló mientras el proyecto existe en el servidor y es
inalcanzable.

**Lo que NO se pudo determinar:** quién corta la conexión. No fue gunicorn —no
hay `WORKER TIMEOUT`—. Queda por descartar el proxy de Render.

**Dos hipótesis mías que la medición desmintió**, anotadas para no repetirlas:
que era el tope de 240 s (aguantó 1,449 s y 3,639 s) y que era un retroceso
catastrófico en mi propia expresión regular (resuelve 32 KB en 1.8 ms). El
prompt tampoco desborda: 54,774 caracteres, y mi bloque añade 1,100.

**Lo que apunta al arreglo:** `/taller/resolver/stream` existe en el servidor y
**la pantalla no lo llama nunca** (`api.ts:452` va al bloqueante). Al emitir sin
parar, un flujo no deja el hueco largo en que se pierde la respuesta. Y
`/taller/descargar` debería leer del almacén duradero y no del proceso.

**RESUELTO el 7-sep.** David pidió las dos cosas:

1. **La pantalla resuelve por flujo** (`resolverEnVivo` → `/taller/resolver/stream`),
   los dos caminos. El documento viaja DENTRO del flujo, así que no depende de
   que sobreviva una respuesta larga. Y el secretario ve el estudio
   escribirse: primer texto a los 60-88 s en vez de cuatro minutos de pantalla
   quieta.
2. **El .docx se sube al cubo `expedientes`** nada más generarlo, y
   `/taller/descargar` lo busca ahí cuando no está en el disco de este proceso.
   Una ruta por secretario y expediente, que se sobrescribe; el correo cifrado
   en la ruta. Si el almacén falla se avisa y se sigue.

Medido dos veces por el flujo: **153 s y 123 s**, documento entregado (≈80,000
caracteres base64), ninguna respuesta perdida.
| 20 | 7-sep-2026 | La pantalla resuelve por `/taller/resolver/stream`, con el estudio a la vista | **Comprobado 2 veces** — 153 s y 123 s, documento entregado en el flujo |
| 21 | 7-sep-2026 | El proyecto se guarda en el cubo `expedientes`; `/taller/descargar` lo busca ahí | **Comprobado en código y guardián**; el camino de recuperación sin verificar en vivo |
| 22 | 7-sep-2026 | La fracción se busca en TODO el estudio, no en el párrafo que anuncia el precepto | **Comprobado en producción** — el art. 79 imprime ya las fracciones VI y VII, que son sobre las que gira el razonamiento |

---

## 9. Pendientes menores, medidos y sin arreglar

- **«Medios de Impugnación» pegado al final del artículo 79.** El acervo guarda
  rótulos de capítulo dentro del texto del precepto —el código ya conoce la
  migaja *delante* («[Ley de Amparo | CAPÍTULO X …] Artículo 79…») y la limpia,
  pero no la de detrás—. Es cosmético.
- **El «sin materia» no se escribe.** Se le pide en el bloque global y no
  aparece ni una vez en tres corridas.
- **El resolutivo dice «sentencia impugnada» y el estudio «sentencia
  recurrida».** La misma cosa con dos nombres en el mismo proyecto.
- **El sentido no es reproducible entre corridas.** Tres corridas del mismo
  asunto: dos `infundado`, una `fundado`. No es un defecto que arreglar — es la
  razón de ser de las dos vías.
| 23 | 7-sep-2026 | El resolutivo decía «impugnada» donde el estudio dice «recurrida» (rama `confirma_sobresee`) | **Comprobado** — unificado; «impugnada» se queda sólo en revisión fiscal |
| 24 | 7-sep-2026 | Los documentos se piden con el nombre del tipo: «Sentencia recurrida» / «Agravios» | **Comprobado en producción** — visto en el navegador con un amparo en revisión |
| 25 | 7-sep-2026 | Dos caminos explicados en vez de tres rotulados; advertencia de responsabilidad antes de elegir | **Comprobado en producción** |
| 26 | 7-sep-2026 | `obtenerTipos` pedía el catálogo con `cache: 'force-cache'` | **Comprobado** — la ficha de una revisión pedía «Quejoso» y «Autoridad responsable» porque servía un catálogo anterior a que existiera `caratula` |
| 27 | 7-sep-2026 | La ficha pintaba el marcador literal `{QUEJOSO_A} Y RECURRENTE` | **Comprobado en producción** — ahora «PARTE QUEJOSA Y RECURRENTE» |

---

## 10. Recorrido completo por la interfaz · 7-sep

Conducido a mano en producción, con los dos PDF reales del 410/2026.

Ficha → adelanto → «Buscar solución jurídica» (rojo, pulsando) → propuesta →
decisión global → sentencia. **Funcionó de punta a punta.**

Lo verificado en pantalla, no en el código:

- Los documentos se piden como **Sentencia recurrida** y **Agravios**, y el
  aviso de faltantes usa esas mismas palabras.
- La carátula pide **PARTE QUEJOSA Y RECURRENTE** y **RECURRENTE ADHESIVO**.
- El acervo va plegado en «En qué se apoya» (11 obligatorias · 29 en total).
- Los **dos caminos** con su explicación, y la advertencia en ámbar antes de
  elegir.
- **El contexto en cuatro párrafos**, específico del asunto.
- **Las dos vías**: propuesta `INFUNDADO` y contraria `FUNDADO`, con «por dónde
  se cae» y la suerte de los demás temas.
- **El estudio se ve escribirse**: contador subiendo hasta 3,980 palabras y
  luego «… componiendo el documento».
- Cierre: 3,976 palabras · 11 avisos · 3 huecos, con el aviso de borrador.

**Dos defectos encontrados EN LA PRUEBA**, no leyendo código: la caché eterna
del catálogo y el marcador literal. Los dos arreglados en el momento. Ninguno
se habría visto sin conducir la interfaz.
| 28 | 7-sep-2026 | Los preceptos bajan a nota al pie, como las tesis (`notas_de_articulos`, que estaba escrita y muerta) | **Comprobado** — 3 bloques y 818 palabras en el cuerpo → 0; 7 artículos al pie |
| 29 | 7-sep-2026 | La extensión se reparte: principal a fondo, inoperante y accesorio en 2-3 párrafos | **Comprobado a medias** — el principal ya es el apartado más largo, pero por poco |
| 30 | 7-sep-2026 | Cuatro reglas de diálogo jurídico sacadas de la edición a mano de David | **Comprobado** — huérfanos 4→0, frases del prompt 2→0, tesis descartada 1→0; conector 11%→17% frente al 21% suyo |

---

## 11. Lo que enseñó comparar con la edición a mano de David · 7-sep

Comparado el proyecto generado con `410-2026 PROYECTO FINAL (ajustes dialogo
juridico)`: 13 párrafos tocados, cuatro patrones.

1. **El artículo quedaba sin verbo** —«El artículo 76 de la Ley de Amparo. Por
   ello…»— cuatro veces. **Regresión mía**: al bajar el precepto al pie, el
   recorte que quitaba la transcripción dejó la frase descabezada, y el prompt
   seguía mandando transcribir.
2. **El prompt se escribía a sí mismo**: «en su versión más favorable», «la
   mejor objeción a esta conclusión es». Sexta vez medida.
3. **Los párrafos no se enlazaban**: David repuso nueve conectores.
4. **Se citaba una tesis para decir que no aplica.**

**LA LECCIÓN QUE MÁS VALE, y ya va dos veces medida:** una regla puesta en la
lista de estilo (carácter ~1,100 del prompt) NO se obedece; la misma regla
dentro de la ARQUITECTURA (~7,000) sí. Pasó con la extensión y volvió a pasar
con los conectores. **Antes de dar una instrucción por escrita, comprobar dónde
cae.**

**Y un remiendo puede ser peor que el defecto:** mi primer arreglo del huérfano
insertaba «dispone lo siguiente», que promete una transcripción que no llega.
Ahora funde las dos frases como las escribió David, y si no encaja en el molde
NO TOCA NADA: un huérfano se ve y se corrige; una frase inventada se firma sin
mirarla.

---

## 12. El considerando de los conceptos, la portada y el formato de impresión · 8-sep

Tres encargos de David sobre el proyecto que sale del pipeline. Los tres
comprobados sobre una generación real, no sobre texto inventado.

### 12.1 El considerando de los conceptos de violación, con su ordinal

El prompt ya pedía «un apartado nuevo» y el modelo lo abría —pero como
subtítulo en negrita DENTRO del Estudio, porque ahí es donde el compositor
mete todo lo que devuelve. Un subtítulo no es un considerando: no lleva
ordinal, y en un engrose el ordinal es lo que dice que ahí empieza otra cosa
que se resolvió.

`partir_conceptos` corta por donde el propio modelo puso el rótulo y mete el
resto en la lista de apartados, que es donde se calculan los ordinales. **No se
le pide al modelo que numere**: esa regla ya estaba y es la buena.

Salvaguardas medidas: no parte si detrás del rótulo hay menos de dos párrafos
—más vale un subtítulo suelto que un considerando hueco con su ordinal
gastado—, no parte con menciones en prosa («el juzgado sobreseyó sin estudiar
los conceptos de violación planteados») y **no partió el proyecto real de 125
párrafos**, que era la regresión a evitar.

**Un efecto de segundo orden que sí apareció**: el cierre del estudio se
escribía al final del apartado de los agravios, o sea ANTES de estudiar los
conceptos de los que depende el desenlace. El documento decía «procede conceder
el amparo» y acto seguido se ponía a examinar si los conceptos eran fundados.
Es la incongruencia del resolutivo que negaba lo que el estudio concedía,
entrando por otra puerta. El cierre se mueve al final del último considerando.

### 12.2 La portada con la síntesis (`fase_sintesis.py`)

**Medido antes de escribir nada: 1,360 documentos de la carpeta del taller.**

Hay DOS formas de síntesis en el corpus, y la primera lectura me dio la
equivocada:

- «CONTEXTO / PROPUESTA / JUSTIFICACIÓN» — 4 documentos.
- **«título en versales + Hechos / Criterio jurídico / Justificación» — 81
  documentos, y 23 de ellos AL FINAL (por encima del 85% de su extensión),
  siempre con el título delante.** Es la que David describió, nombrando los
  cuatro campos del Semanario.

Extensiones medianas de esos 23, que son los parámetros del prompt: título 236
· hechos 815 · criterio 571 · justificación 1,148 caracteres.

**El registro es de proyecto, no de tesis publicada**: en el ADC 821/2025 el
criterio empieza «Se propone determinar que…», no «Se determina que…». Un
proyecto propone; lo que se publica ya resolvió. Esa palabra distingue un
documento que va a sesión de uno que salió de ella, y se corrige en código si
el modelo la escribe en presente.

Se pide **con el estudio ya redactado**, no con los datos del asunto: una
síntesis escrita antes resumiría lo que se pensaba resolver. Sustituye a las
dos firmas del pie —los nombres siguen en la carátula, que es donde se
buscan—, y si el modelo falla, el documento sale con las firmas de siempre.

### 12.3 El formato de impresión: dónde NO estaba el problema

David: «cuando hay saltos de página, como se imprime por ambas caras, la
sangría cambia».

**Lo primero que miré fueron los márgenes, y ahí no estaba.** Medidos los 30
engroses de la carpeta: el generado YA coincidía con el suyo —21.59 × 34.04,
izquierda 5, derecha 2, superior e inferior 3— y **ninguno de los treinta usa
márgenes en espejo**. La caja era la misma. Haber «arreglado» los márgenes
habría estropeado lo único que ya estaba bien.

La diferencia estaba en el encabezado:

    evenAndOddHeaders ....... 29 de 30 engroses (97%)
    primera página distinta .. 29 de 30 (97%)
    pie con número de página . 27 de 30 (90%)
    ese número, centrado ..... 56 de 58 pies

Word alterna el encabezado entre página par e impar, y por eso «la sangría
cambia» al pasar la hoja. El generado ponía el MISMO encabezado en las tres y
no numeraba: impreso por ambas caras no cuadraba con nada.

**La alineación del par no la decidió la mayoría.** Emparejando dentro de cada
documento: 10 veces derecha→izquierda, 10 derecha→centro, 5 derecha→derecha, 4
centro→centro. Empate. Se eligió derecha→izquierda porque **es la única de las
cuatro que alterna**, y alternar es lo que él pide.

Detalle que costaría una corrida: el interruptor vive en `settings.xml` y
python-docx no lo expone. Sin él, Word ignora el encabezado de página par por
mucho que esté escrito en el fichero.

### 12.4 El predicado que se equivocaba en cuatro de ocho

Al conectar el recuadro de los conceptos apareció un fallo vivo. El aviso de
«faltan los conceptos de violación» colgaba de
`str(glob.sentido).startswith("fundad")`, escrito antes de que existieran las
calificaciones con matiz. Medido sobre las ocho, **se equivoca en cuatro**: no
reconoce esencialmente / parcialmente / sustancialmente fundado —con
cualquiera de los tres el aviso no salía— y en cambio cuenta como próspero
`fundado_insuficiente`, que en el circuito confirma el 88% de las veces.

`prospera()` existe justamente para que añadir una calificación no obligue a
acordarse de catorce sitios. **Cada vez que se añade una calificación hay que
buscar los `startswith` que quedaron sueltos.**

### 12.5 Y una lección de método, otra vez

Copié `~/Downloads/*.pdf` al directorio de trabajo para preparar la prueba y
**sobrescribí los agravios de la 410/2026 con los de otro asunto**. La corrida
salió con la sentencia recurrida correcta y los agravios de un caso de
suspensión: sirvió para comprobar el formato y la portada, no el contenido.
Se detectó porque el contexto propuesto hablaba de un exhorto que no venía a
cuento. **Un comodín en un `cp` sobre el directorio de trabajo es una carga de
datos, no una copia de conveniencia.**

---

## 13. El proyecto que volvía a salir extemporáneo · 8-sep

**El caso.** Erika, la tester, generó un adelanto con el plazo mal tecleado: el
proyecto salió extemporáneo, que era lo correcto. Rehízo el adelanto con el
plazo bueno, el adelanto salió bien —y el proyecto **volvió a salir vacío por
extemporaneidad**.

**La base tenía razón desde el principio.** La sesión 437/2025 guardaba plazo
10, notificación 18-jun-2025, presentación 2-jul-2025 y `oportuna=true`.

**Los registros lo enseñan al minuto:**

    18:30  worker prrxc · adelanto con el plazo mal → memoria de prrxc
    18:47  worker fm77h · adelanto CORREGIDO        → memoria de fm77h + base
    18:51  worker prrxc · resolver                  → memoria RANCIA de prrxc
    18:54  worker prrxc · proyecto vacío

`_taller_recuperar_sesion` devolvía la copia en memoria sin comprobar nada. Con
dos workers, que el adelanto corregido y el resolver caigan en el mismo es una
moneda al aire.

**Ya había un parche de esta misma familia** —el del «Consulta primero el
acervo»—, y arreglaba UN campo: releía `consultado` de la base y dejaba rancio
todo lo demás. Cualquier dato del adelanto podía quedar viejo igual: las
fechas, las partes, la materia, el tipo de asunto. **Arreglar el síntoma campo
por campo deja la puerta abierta por los otros catorce.**

### Las dos trampas del arreglo

**1 · `creado_en` no sirve como sello.** Su default `now()` sólo corre en el
INSERT: un upsert que ACTUALIZA deja la fecha del primer adelanto. Comprobado
en la fila real: `creado_en` 18:30 con el contenido de las 18:47. Una
comprobación de frescura basada en ella no habría detectado nada nunca.

**2 · `now()` no se mueve dentro de una transacción.** Es la hora de inicio de
transacción. Mi primera prueba del disparador —insertar y actualizar en una
sola sentencia— dio «el sello no cambió», y eso ocurría tanto si el disparador
funcionaba como si no: **una prueba que no puede distinguir las dos cosas no
prueba nada.** Con `clock_timestamp()` la diferencia es de 64 ms y la prueba sí
distingue.

**Y un tercer detalle, de coste:** el disparador NO debe correr al marcar
`consultado`. Si corriera, el worker que acaba de consultar tiraría su propia
memoria buena en la llamada siguiente y volvería a recuperar el acervo. Se
acota con `WHEN (OLD.estado IS DISTINCT FROM NEW.estado …)`: lo que invalida la
memoria es que cambie el adelanto, no que se marque una casilla. Comprobado por
separado: marcar consultado deja el sello quieto, un adelanto nuevo lo mueve.

### La prueba

Se reprodujo la avería a propósito: cuatro adelantos con el plazo mal para
dejar la sesión rancia en los dos workers, uno con el plazo bueno, y luego seis
consultas y dos propuestas.

    ♻️ memoria rancia detectada y releída ... 2 veces (una por worker)
    ⚖️ «cómputo extemporáneo» tras corregir . 0

**Las dos cifras importan.** El cero es el arreglo. Y que el reciclaje salte
exactamente dos veces —una por worker— y no en cada llamada es lo que prueba
que la comparación del sello funciona: si el formato del sello no coincidiera
entre el guardado y la lectura, saltaría siempre y el taller estaría releyendo
la base sin parar.

**Si la base no contesta, se sigue con la memoria.** Peor que un dato viejo es
no poder trabajar: el secretario tiene el adelanto delante y lo que quiere es
su proyecto.

---

## 14. El resolutivo que negaba el amparo que confirmaba · 8-sep

**El caso.** Revisión 650/2025. El proyecto calificó los agravios de
infundados, confirmó la sentencia recurrida y acto seguido NEGÓ el amparo que
esa misma sentencia había concedido.

**La causa no era la tabla de ramas.** `confirma_concede` existía y
`rama_revision` la elegía bien. Lo que fallaba era el dato de entrada: de qué
resolvió el Juzgado de Distrito depende el resolutivo entero, y se decidía con
**prosa del modelo** —el resumen de antecedentes o la frase con la que describió
el asunto al proponer—.

`fase_rama.resolvio_a_quo` sí lee bien el PDF: probado sobre la recurrida
devuelve «concede» en las tres formas de invocarlo. **El texto crudo no le
llegaba.** Vive en `fases.fuentes` y `fuentes` NO viaja en el estado de la
sesión —son 240 kB—, así que el worker que resuelve se queda sin él. Misma
familia que el apartado 13, y el propio comentario de `Fases123` ya enunciaba
la regla que se incumplía: «lo que no viaja en ese estado no existe para la
petición siguiente».

**Se ve en las corridas de comprobación**, y esto es lo que convence: cuatro
generaciones del mismo asunto, con el mismo PDF, y el modelo dijo del a quo
«Concedió el amparo», luego «negó el amparo», luego «concedió», luego
«concedió». La segunda es falsa. **El resolutivo dependía de eso.**

Ahora se lee UNA VEZ, en el adelanto, y viajan dos cadenas cortas:
`resolvio_a_quo` y `resolutivo_recurrida`.

### El fallo que cometí al arreglarlo, y cómo salió

Leí el dato de `relleno` —el ensamblado de la plantilla— en vez de `fases`.
`getattr(relleno, "resolvio_a_quo", "")` devuelve cadena vacía **en silencio**,
así que el arreglo no hacía nada y el resolutivo seguía saliendo de la prosa.

No lo cazó una lectura del código: lo cazó la comprobación de extremo a
extremo, en la que el modelo declaró «negó el amparo» y su versión ganó igual.
**Un `getattr` con valor por omisión sobre el objeto equivocado no falla, y no
avisa.** Por eso ahora, cuando el dato determinista falta, el documento lo dice
en un aviso: leer el papel y no leerlo producían documentos indistinguibles.

### Lo que pidió David, y lo que dijo el corpus

**Reproducir el resolutivo del juzgado.** «La Justicia de la Unión ampara y
protege a X en términos del último considerando de la resolución recurrida» no
dice contra qué acto ni para qué efectos. Se reproduce el del juzgado con la
cola reescrita para que apunte a la recurrida —«…en el diverso séptimo DE LA
SENTENCIA RECURRIDA»—. Es el detalle que se pierde al copiar y pegar: **en su
propio proyecto corregido a mano la cola quedó sin cambiar**, y por eso lo pidió
por escrito. El precedente del tribunal, ARA 361/2025, hace lo mismo. Si el
resolutivo del juzgado tiene más de un punto, no se reproduce: se escribe la
fórmula genérica, que dice menos pero no dice de más.

**«En la materia de la revisión» NO va siempre.** Medido: 0 de los 381
resolutivos legibles de su carpeta la usan, y el 27% de los engroses habla de
aspectos no combatidos sin usarla. Se escribe sólo cuando el estudio de ESTE
proyecto dice que algo quedó fuera. Calibrado: acusa su proyecto corregido —por
«no fueron combatidos por la parte recurrente»— y no acusa ninguno de los tres
proyectos normales probados.

### Dos hallazgos de la comparación con su versión corregida

**El órgano emisor iba en un renglón suyo.** Debajo de cada rubro salía
«SEGUNDA SALA.» y nada más. Él lo borró en las cuatro tesis. De los 1,358
documentos de la carpeta, **UNO** lo escribe así. Y el anuncio de la cita ya
nombraba al órgano en prosa: era pura duplicación. Se va a la nota, con la
localización y el registro.

**El encabezado NO se toca.** Él quitó la materia y los dos puntos —«AMPARO EN
REVISIÓN 650/2025»—, pero medidos sus 472 encabezados, el **90%** lleva materia
Y dos puntos, que es lo que ya generamos. Su edición es la variante que no
aparece ninguna vez. *Una corrección a mano no es siempre una regla: a veces es
una prisa.*

### Lo que queda abierto

`revoca_fondo_niega if a == "concede"` supone que quien recurre es la autoridad.
Cuando recurre la QUEJOSA que ya ganó y quiere más —el caso 650/2025—, un
agravio fundado no puede terminar negándole el amparo. Visto en una corrida: con
sentido «fundado», el proyecto revocó y negó el amparo a quien lo había
obtenido. **Falta meter en la ecuación quién recurre.**
