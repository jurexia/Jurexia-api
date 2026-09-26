-- ═══════════════════════════════════════════════════════════════════════════
-- EL PLAN DEL ESTUDIO, EN SU PROPIA COLUMNA (Paso 2, 26-sep-2026)
-- ═══════════════════════════════════════════════════════════════════════════
-- SIN APLICAR. La aplica el integrador de la entrega del Paso 2 (propuesta del
-- estudio aprobada por David, diag/w2_final.md §3.6 y §4.5).
--
-- POR QUÉ UNA COLUMNA Y NO UNA RAMA DE `estado`:
--
-- 1. `estado` lleva `fuentes` —el acto y el escrito, hasta 2 × 600.000
--    caracteres— y se reescribe entero con lectura-cambio-escritura. Un plan
--    que se guardara ahí tendría que releer y reescribir ese megabyte en cada
--    latido, y una escritura cruzada podría pisar la pila `proyectos` o la
--    propuesta guardada.
-- 2. El sello de la sesión (`actualizado_en`, migration_taller_sesiones_sello)
--    sólo se mueve cuando cambia `estado` o la plantilla. Escribir el plan en
--    `estado` invalidaría en cada latido la copia en memoria del otro worker y
--    le haría releer el acervo; en esta columna, no.
-- 3. El compare-and-set es sobre `plan->>rev`: un update dirigido que sólo
--    escribe si la fila sigue en el turno que se leyó (main.py
--    `_taller_plan_cas`). Así un plan viejo que termina último no pisa al nuevo.
--
-- LA FORMA (la escribe y la lee sólo el servidor, con el rol de servicio):
--   {rev, version, huella, corridas, pedido_clave,
--    planes: {<clave>: {estado: en_curso|listo|fallo, desde, latido,
--                       plan, avisos, segundos, hecho}}}
-- `huella` es la del adelanto: si el secretario rehace el adelanto, el tope de
-- cuatro corridas y los planes empiezan de cero. Como mucho seis casillas.
--
-- REVERSA: la columna puede quedarse; sin ella el servidor sigue funcionando
-- (calcula el plan dentro de la petición, sin reutilizarlo) y con
-- ESTUDIO_PROMPT distinto de v4 y sin cuentas de casa pidiéndola, nadie la lee.
-- Para retirarla del todo:  ALTER TABLE public.taller_sesiones DROP COLUMN plan;

ALTER TABLE public.taller_sesiones
    ADD COLUMN IF NOT EXISTS plan jsonb;

COMMENT ON COLUMN public.taller_sesiones.plan IS
    'Plan del estudio (Paso 2): estado por clave, contador de corridas y turno '
    '(rev) para el compare-and-set. Lo escribe sólo el servidor.';
