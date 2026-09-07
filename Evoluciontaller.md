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
