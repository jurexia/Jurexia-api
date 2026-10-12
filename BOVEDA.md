# La Bóveda de Iurexia — diseño (12-oct-2026)

Equivalente al Vault de Harvey: por carpeta se suben de 50 a 500 documentos (PDF y Word), se indexan por
usuario y carpeta, y se consultan juntos de dos maneras: el **chat** con la Bóveda como fuente (cita
documento y página) y la **tabla de revisión** (filas = documentos, columnas = preguntas del abogado,
cada celda con su cita literal verificada). **Sólo Platinum** (y Ultra y administradores, como en todo).

## Lo medido (60 contratos ficticios propios: 190 páginas, 19 escaneadas, 12 en Word)

| Concepto | Costo real medido | Nota |
|---|---|---|
| OCR (Azure `prebuilt-read`) | **0.0015 USD por página escaneada** | Sólo las páginas sin texto van a Azure (parámetro `pages`); las que tienen capa de texto son gratis |
| Embeddings (text-embedding-3-small) | **0.000014 USD por página** | 2.3 fragmentos por página |
| Tabla de revisión (gpt-6-luna, low) | **0.0001 USD por celda** (0.00057 por fila de 6 columnas) | Una llamada por FILA: el documento se lee una vez para todas las columnas |
| Búsqueda en la Bóveda | 295 ms de mediana | Híbrida (densa + BM25 en español), con el embedding de la pregunta incluido |
| Precisión de la tabla | 360/360 celdas correctas, **100% de citas verificadas** | Contratos sintéticos y sencillos; los reales serán más difíciles |
| Tiempo | 60 documentos: ~1 min de ingesta; tabla de 60 filas: 68 s | 500 documentos escaneados: 10–20 min |

Un documento largo (hasta ~64 páginas) se manda entero; más allá, sólo las páginas que la búsqueda dentro
de ESE documento señala para cada pregunta. El peor caso de una fila es ~0.0045 USD.

**Lo que vale una consulta:** Platinum mensual 599/560 = 1.07 MXN (≈0.058 USD); anual 0.89 MXN (≈0.048 USD), a ~18.5 MXN/USD.

## Regla de cobro propuesta (constantes configurables en Render)

| Constante | Propuesta | Costo por consulta cobrada | % de lo que vale la consulta (anual) |
|---|---|---|---|
| `BOVEDA_PAGINAS_OCR_POR_CONSULTA` | **10** páginas escaneadas | 0.015 USD | 31% |
| `BOVEDA_PAGINAS_TEXTO_POR_CONSULTA` | **100** páginas con texto | 0.0014 USD | 3% |
| `BOVEDA_CELDAS_POR_CONSULTA` | **10** celdas | 0.001 USD típico · 0.045 peor caso | 2% típico · 94% peor caso |

`consultas = ⌈ páginas_escaneadas/10 + páginas_con_texto/100 ⌉` por tanda, y `⌈ celdas/10 ⌉` por corrida de tabla.
Ejemplos: 50 contratos nativos de 15 páginas → 8 consultas; un expediente escaneado de 400 páginas → 40;
una tabla de 50 documentos × 6 columnas → 30. Con 560 al mes, un Platinum puede ingerir ~5,600 páginas
escaneadas al mes. Si se quiere más holgura: 20 páginas escaneadas por consulta deja el OCR en 63%.

**Cómo se ve:** antes de ejecutar, «47 documentos · 312 páginas (40 escaneadas) → 8 consultas. Te quedan
520. ¿Indexar?». Si no alcanzan, no se ejecuta. Se cobra del MISMO contador del chat (`queries_used`,
luego `consultas_recargadas`) con `consumir_consultas` (todo o nada). Lo que falla se devuelve al cerrar la tanda.

## Dónde vive cada cosa

- **Archivos:** Supabase Storage, cubo privado nuevo `boveda` (50 MB por archivo, sólo PDF y .docx), en
  `<usuario>/<carpeta>/<documento>`. No GCS: Supabase ya guarda las carpetas, la sesión del usuario ya
  firma ahí, el visor ya tiene a Supabase en su lista blanca, y hoy se usan 1.4 GB de los 100 incluidos.
  El navegador sube directo con URL firmada que da la API tras revisar plan y cuotas (no pasa por Render).
  El texto de cada página se guarda junto al archivo (`….paginas.json.gz`) para la tabla y el visor.
- **Índice:** una colección Qdrant `boveda`, multiinquilino: payload `user_id` (índice de inquilino),
  `carpeta_id`, `doc_id`, `pagina`. Vectores, HNSW (por inquilino: m=0, payload_m=16), cuantización int8 y
  payload, todo en disco: **no toma RAM del acervo** (Qdrant usa hoy 2.1 de 3.4 GB). ~23 KB de disco por página.
- **OCR:** Azure, sólo páginas escaneadas; si el mismo archivo ya está en la Bóveda del usuario, se reutiliza su texto.
- **Cola:** tablas en Supabase y un trabajador en cada proceso de uvicorn que toma trabajo con
  `FOR UPDATE SKIP LOCKED`. Sobrevive a despliegues (lo huérfano se retoma a los 15 min) y reparte entre
  usuarios: quinientos documentos de uno no dejan esperando los cinco de otro.

## Cuotas propuestas por usuario Platinum

500 documentos por carpeta · 2,000 documentos y **10,000 páginas** en total · 50 MB por archivo ·
1,500 páginas por documento · 5 GB de archivos. Con las 46 cuentas Platinum al tope serían ~11 GB en
Qdrant (hay ~25 libres). El cobro ya limita el ritmo; el tope protege el disco.

## Privacidad

Borrar un documento, una carpeta o toda la Bóveda (o la cuenta) borra la fila y un disparador deja en
`boveda_purgas` lo de fuera de Postgres: el trabajador borra los vectores (filtro por `doc_id` y
`user_id`) y los archivos. Toda lectura de Qdrant filtra por `user_id` del TOKEN, nunca del cuerpo de la
petición. `/cita/{id}` (sin sesión) no conoce la colección `boveda`.

## El chat sobre la Bóveda

Dentro de una carpeta con Bóveda, «Fuentes» suma la fila **Bóveda**. La API busca 12 fragmentos de esa
carpeta y los suma a las fuentes en el mismo punto donde entran doctrina y la Corte IDH; el modelo cita
`[Doc ID: …]` como siempre y el clic abre el PDF en esa página con el pasaje resaltado (`VisorArticulo`).
Cuesta lo mismo que cualquier consulta (1). Hoy `/chat` confía en el `user_id` del cuerpo: para la
Bóveda se exige el token y que la carpeta sea suya.

## La tabla de revisión

Columnas sugeridas (partes, fecha, vigencia, contraprestación, pena convencional, jurisdicción) y las que
el abogado escriba. Cada celda: `{valor, encontrado, cita, pagina, verificada}`. La cita se busca en el
texto real: si no está, la celda dice «cita no verificada». Clic en la celda → el PDF en esa página.
Exporta a Excel (openpyxl, dependencia nueva) y Word (python-docx, ya está).

## Compartido con la Cronología

`cobro_consultas.py` (`cobrar_consultas`, `devolver_consultas`, `consultas_disponibles`,
`exigir_platinum`) y `migration_cobro_consultas.sql` (`consumir_consultas`, `devolver_consultas`,
`consultas_disponibles`, tabla `consultas_movimientos`).
