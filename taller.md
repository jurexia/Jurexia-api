# El taller de sentencias · dónde vamos

Estado a **8 de septiembre de 2026**. Este fichero dice qué hace hoy el taller,
qué se ha implementado y comprobado, y **con qué regla se hace cada mejora**.

`Evoluciontaller.md` es el diario: cada cambio con lo que se midió antes y
después. Éste es el mapa. Se leen los dos antes de tocar nada.

---

## LA REGLA · cómo se prueba cada mejora

Ésta es la parte que no se salta. Nació de los errores de este trabajo, no de
una idea previa de cómo trabajar, y cada punto tiene detrás una vez que costó
una corrida o algo peor.

### 1 · Se mide antes de tocar

Nada se cambia porque parezca que está mal. Primero se mide sobre el corpus
real —los engroses del tribunal, los proyectos, los 33,036 holdings del
circuito 22— y el número decide.

**Lo que esto ha evitado:** iba a «arreglar» los márgenes del documento porque
David dijo que la impresión no le servía. Medidos los 30 engroses, los márgenes
del generado YA coincidían con los suyos y ninguno usa márgenes en espejo.
Tocarlos habría roto lo único que estaba bien; el problema estaba en el
encabezado par/impar, que la medición sí encontró (29 de 30).

### 2 · Un remiendo que puede dañar texto correcto es peor que el defecto

Si el arreglo no se puede acotar, no se hace. El primer parche del artículo
huérfano insertaba «dispone lo siguiente» y producía «dispone que el órgano
jurisdiccional dispone lo siguiente»: peor que el defecto. Se retiró entero y
se arregló en la raíz.

Corolario: **un defecto se ve y se corrige; una frase inventada se firma sin
mirarla.**

### 3 · Si una comprobación acusa al trabajo bueno, la que está mal es la
comprobación

Medido catorce veces en este proyecto. Antes de creer que algo está roto, se
pasa la comprobación por sentencias que se sabe que están bien. Dos de cada
cuatro comprobaciones se cayeron en esa prueba.

### 4 · Se calibra sobre el texto real, nunca sobre uno inventado

Tres veces medí el artefacto equivocado: tomé el documento compuesto por
salida del modelo, tomé un recorte de pantalla de 600 caracteres por el texto
del acervo, y declaré «cero huérfanos» con un detector que no conocía «Al
respecto,».

En esta última ronda, un remate duplicado que aparecía en mi texto de prueba
**no aparecía en la generación real**. No se parcheó.

### 5 · Se comprueba el camino completo, no cada pieza

Dos arreglos correctos por separado produjeron un desastre juntos. La prueba
válida es la que va de los PDF al `.docx`.

### 6 · La posición de una instrucción en el prompt decide si se obedece

Medido dos veces: una regla en la lista de estilo (carácter ~1,100) **no** se
obedece; la misma regla dentro de la arquitectura (~7,000) sí. Antes de dar por
escrita una instrucción, comprobar dónde cae.

### 7 · Lo que se puede componer no se pide

Los ordinales, las fórmulas de resolutivo, el proemio, el cierre del tipo: son
determinismo. Pedírselos al modelo es pagar por que a veces salgan mal.

### 8 · Se verifica el commit vivo, no `/health`

Tres veces medí código viejo por fiarme de `/health`. Se consulta la API de
Render y se comprueba que el commit desplegado es el que se acaba de empujar.

### 9 · El guardián se ejecuta antes de empujar

`comprobar_antes_de_empujar.sh`. Y si un fallo nuevo pertenece a una clase que
el guardián no podía ver, **se arregla el guardián**: su `liga()` recorría el
módulo entero con `ast.walk` y contaba las variables locales de las funciones
como nombres del módulo, así que jamás habría cazado el `_rama` fuera de
ámbito que dejó mudo el generador. Se reintrodujo el fallo a propósito para
comprobar que ahora sí lo caza.

### 10 · Cada cambio comprobado se anota en `Evoluciontaller.md`

Y se lee antes del siguiente. Es lo que impide repetir un camino ya andado.

---

## EL PIPELINE, DE PRINCIPIO A FIN

Nueve rutas. El secretario nunca queda sin saber dónde mirar: la pantalla se
desplaza sola a `recorrido`, `criterio`, `estudio` y `proyecto` según avanza.

| Ruta | Qué hace |
|---|---|
| `/taller/tipos` | El catálogo de tipos de asunto y su vocabulario |
| `/taller/adelanto` | De los PDF al adelanto: partes, fechas, cómputo, resúmenes y problemas jurídicos |
| `/taller/consultar` | El acervo: tesis y normas verificadas para esos problemas |
| `/taller/contexto` | El secretario aporta contexto que no está en los papeles |
| `/taller/proponer` | El motor propone el sentido, con su vía contraria y la lista de comprobación |
| `/taller/resolver/stream` | La sentencia, viéndose escribir |
| `/taller/resolver` | Lo mismo, sin streaming |
| `/taller/descargar` | El `.docx` |
| `/taller/estado` | El cupo del piloto |

### Las fases

- **`fase0_oportunidad`** — el cómputo de días hábiles. Aritmética, sin modelo.
  Si da extemporánea, **el pipeline no entra al fondo**: escribe la
  improcedencia. Antes declaraba la extemporaneidad y luego concedía.
- **`fases123_pipeline` / `fases123_resumenes`** — la ratio del acto recurrido
  y la síntesis de los conceptos o agravios, con nota al pie al párrafo de
  origen. **Estos dos resúmenes se quedan en el proyecto**: son útiles.
- **`fase_partes`, `fase_autoridad`, `fase_origen`, `fase_procedencia_rf`** —
  quién es quién, y de dónde viene el asunto.
- **`fase_normas`** — los preceptos, del acervo, no de la memoria del modelo.
- **`fase5_propuesta`** — la propuesta de sentido, global y por problema, con
  la **objeción de quien resolvería al revés**.
- **`fase6_rag`** — **la palanca mayor de calidad.** La pregunta se traduce a
  consulta conceptual (precisa y amplia) antes de buscar contra el vector
  `rubro`. Medido: prosa 0.612 / 0.596 / 0.623 frente a precisa 0.729 / 0.776 /
  0.736 y amplia 0.674 / 0.725 / 0.820. Y no es sólo puntuación: con la
  pregunta cruda salían seis tesis de nulidad y ninguna de tercero extraño por
  equiparación, que era la figura que decidía. La consulta «amplia» es la que
  trae **la tesis análoga cuando no hay una en el punto**.
- **`fase_precedente`** — el acervo de colegiados: sondeo de sentido, molde de
  forma y la objeción del que resolvió al revés.
- **`fase6_estudio`** — el estudio, con los cuatro pasos, la técnica del
  escenario, la propuesta global y el bloque del circuito.
- **`fase_rama`** — qué resolvió el a quo, y de ahí qué se puede hacer.
- **`fase_sintesis`** — la portada con la síntesis, con el estudio ya escrito.
- **`documento_generado`** — compone el `.docx`. Ordinales, resolutivos,
  preceptos al pie, encabezados y formato de impresión.
- **`tipos_asunto`** — **la única fuente de verdad**: calificaciones,
  resolutivos, rótulos, técnica de resolución.
- **`tabla_circuito`** — lo que de verdad hace el circuito 22, medido.

---

## LO QUE YA ESTÁ, Y COMPROBADO

### El catálogo (`tipos_asunto.py`)

**10 calificaciones**, con su recuento medido sobre 65,282 agravios del
circuito 22. `prospera(sentido)` es **el predicado único**: existe para que
añadir una calificación no obligue a acordarse de catorce sitios.

Dos que costaron medirlas:
- `fundado_insuficiente` **NO prospera**: medido, confirma el 88% de las veces,
  como el infundado. Mi primera clasificación lo daba por próspero y habría
  revocado sentencias que se confirman.
- `sin_materia` no se conjuga con «ser»: no se es sin materia, se **queda** sin
  materia. «Los agravios quedaron sin materia».

**La tabla de la técnica de resolución**, cinco escenarios:
`revision_levanta_sobreseimiento` · `revision_no_es_materia` ·
`revision_firmeza` · `recurso_sin_materia` · `directo_orden_de_estudio`.

Con la distinción que pidió David y que no es la misma figura:
- **«No es materia del recurso»** — la consideración que favorece a quien no
  acudió a la revisión.
- **«Firmeza»** — la consideración que perjudica a quien sí acudió y no la
  impugnó.

### Lo que el circuito hace de verdad (`tabla_circuito.py`)

Medido sobre 33,036 holdings, 12,272 expedientes y 65,282 agravios:

    revisión CONFIRMA 60.0% → infundado 53% · inoperante 33% · fundado 1%
    revisión REVOCA   10.6% → fundado 52% · esencialmente fundado 23%

### El documento

- **Sin reenvío en revisión**: se levanta el sobreseimiento y el tribunal asume
  jurisdicción (artículo 93, fracción I).
- **El considerando de los conceptos de violación, con su ordinal.** Se corta
  por donde el modelo pone el rótulo y entra en la lista de apartados, que es
  donde se calculan los ordinales. No parte estudios normales —comprobado
  contra el proyecto real de 125 párrafos— ni con menciones en prosa.
- **El cierre va al final del último considerando**, no antes de estudiar
  aquello de lo que depende el desenlace.
- **Los preceptos van al pie**, citados en prosa arriba, como las tesis.
- **La portada con la síntesis** sustituye a las dos firmas: título en
  versales, Hechos, Criterio jurídico, Justificación. Medido sobre 1,360
  documentos; el registro es «Se propone determinar que», porque un proyecto
  propone.
- **El formato de impresión a doble cara**: `evenAndOddHeaders`, primera página
  distinta, pie con número centrado, encabezado a la derecha en impar y a la
  izquierda en par.
- **El considerando de «Materia del recurso» se retiró**: los problemas
  jurídicos guían el estudio, no se imprimen.
- **Se produce un proyecto completo**: si falta un dato se remite a las
  constancias del expediente —«en la fecha que se advierte de las constancias»—
  y nunca se escribe que el dato no consta. Las excepciones son las fechas de
  sesión y de lista, que llevan hueco visible.
- **El resolutivo va justificado** en los cuatro tipos.
- **La extensión es proporcional**: los temas accesorios y la inoperancia se
  resuelven en 3 a 7 párrafos.

### Dos deslindes que no se pueden confundir

**En revisión se REVOCA; sólo en amparo se deja INSUBSISTENTE.** No son dos
maneras de decir lo mismo:

- En un **recurso**, el tribunal es órgano revisor de esa misma sentencia y la
  revoca: con eso deja de existir, y no hay a quién ordenarle que la deje
  insubsistente. Si hay reenvío, lo que se ordena es **dictar otra**.
- En **amparo**, el tribunal no revoca el acto reclamado —no es superior
  jerárquico de la responsable—: concede la protección y le ordena dejarlo
  insubsistente. Ahí la fórmula existe, y por eso se confunde.

**En revisión de amparo NO hay reenvío; en revisión fiscal SÍ.** En el amparo en
revisión el colegiado levanta el sobreseimiento y **asume jurisdicción**
(artículo 93, fracción I). En la revisión fiscal no puede: el estudio de los
conceptos de anulación corresponde en primera instancia a la Sala, y sustituirla
dejaría al particular sin amparo contra ese estudio. Apoyos verificados en el
acervo: registros 188742, 193181, 196875 y 2000895; el deslinde —cuando el vicio
formal no trasciende y el colegiado corrige él mismo— en el 185493.

### El cuerpo no transcribe

El texto íntegro de preceptos y tesis largas lo baja el documento a la **nota al
pie**. Por eso el cuerpo NUNCA anuncia una transcripción —«establece lo
siguiente»— ni remite a ella —«del precepto transcrito»—: son frases que
apuntan a algo que el lector no va a encontrar.

Esto estuvo roto mucho tiempo porque **la arquitectura del prompt ordenaba lo
contrario**, y la arquitectura es lo que se obedece. Medido antes de
corregirlo: 55 remisiones falsas repartidas por los 50 proyectos generados.

### El acceso

`_taller_puerta()`: Platinum, o bien el campo **`can_access_sentencia`** en
`user_profiles` —que es la puerta para testers—, más el cupo del piloto
(`TALLER_PILOTO_CUPO`, hoy 10) y las cuotas de `_taller_cuota()`. Quien ya está
dentro sigue dentro aunque el cupo se cierre.

---

## LO QUE QUEDA

- **Decisión de David**: si el considerando de «Materia del recurso» debe
  volver en algún tipo concreto.
- **Decisión de David**: la alineación del encabezado en página par. La
  medición quedó en empate (10 derecha→izquierda, 10 derecha→centro); se eligió
  izquierda porque es la única que alterna.
- Fijar el mapa del «Recorrido del asunto» al desplazarse, y plegar la ficha
  después del adelanto.
- El sentido propuesto **no es reproducible entre corridas** cuando el asunto
  está reñido. La forma sí quedó estable en seis vueltas; el sentido no. Es el
  límite conocido y es donde entra el criterio del secretario.
