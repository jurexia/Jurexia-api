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

### Paso 3 · Fijar el criterio  → `VentanaCriterio.tsx`

El secretario ve la propuesta y decide. Dos modos:

- **Global**: un solo sentido para el asunto. El principal decide; si prospera,
  los accesorios quedan sin materia.
- **Por problema**: un sentido para cada uno.

### Paso 4 · Generar el proyecto  → `POST /taller/resolver` o `/resolver/stream`

`main.py:28034` y `main.py:27837`. Corre el estudio de fondo (`fase6_estudio`) y
compone el .docx. `GET /taller/descargar` lo entrega.

---

## 2. Defectos verificados, sin arreglar

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
