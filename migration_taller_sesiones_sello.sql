-- ═══════════════════════════════════════════════════════════════════════════
-- EL SELLO DE LA SESIÓN DEL TALLER
-- ═══════════════════════════════════════════════════════════════════════════
-- Aplicada en producción el 8 de septiembre de 2026.
--
-- POR QUÉ. La sesión del taller se cachea en la memoria de cada worker y la
-- memoria se queda rancia. El 8 de septiembre una tester rehízo el adelanto con
-- el plazo corregido y el proyecto volvió a salir extemporáneo: el adelanto
-- bueno lo atendió un worker y el resolver cayó en el otro, que seguía con el
-- cómputo viejo. Con dos workers eso es una moneda al aire.
--
-- `creado_en` no servía para detectarlo: su default `now()` sólo corre en el
-- INSERT, así que un upsert que ACTUALIZA la fila deja la fecha del primer
-- adelanto. Medido en la sesión 437/2025: creado_en 18:30 con el contenido de
-- las 18:47.
--
-- DOS DETALLES QUE COSTARON UNA VUELTA CADA UNO:
--
-- 1. `now()` es la hora de INICIO DE TRANSACCIÓN y no se mueve dentro de ella.
--    La primera prueba —insertar y actualizar en una sola transacción— daba que
--    el sello no cambiaba, tanto si el disparador funcionaba como si no. Con
--    `clock_timestamp()` la prueba distingue: 64 ms de diferencia.
--
-- 2. El disparador NO debe correr al marcar `consultado`. Si corriera, el
--    worker que acaba de consultar tiraría su propia memoria buena en la
--    llamada siguiente y volvería a recuperar el acervo: un arreglo de
--    corrección pagado en una consulta de más por asunto. La condición de
--    invalidación es que cambie el ADELANTO (`estado`) o la plantilla.

ALTER TABLE public.taller_sesiones
    ADD COLUMN IF NOT EXISTS actualizado_en timestamptz NOT NULL DEFAULT now();

CREATE OR REPLACE FUNCTION public.taller_sesiones_sella_actualizacion()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
    NEW.actualizado_en := clock_timestamp();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_taller_sesiones_actualizado_en ON public.taller_sesiones;
DROP TRIGGER IF EXISTS trg_taller_sesiones_sello_alta   ON public.taller_sesiones;
DROP TRIGGER IF EXISTS trg_taller_sesiones_sello_cambio ON public.taller_sesiones;

CREATE TRIGGER trg_taller_sesiones_sello_alta
    BEFORE INSERT ON public.taller_sesiones
    FOR EACH ROW
    EXECUTE FUNCTION public.taller_sesiones_sella_actualizacion();

CREATE TRIGGER trg_taller_sesiones_sello_cambio
    BEFORE UPDATE ON public.taller_sesiones
    FOR EACH ROW
    WHEN (OLD.estado IS DISTINCT FROM NEW.estado
          OR OLD.plantilla IS DISTINCT FROM NEW.plantilla)
    EXECUTE FUNCTION public.taller_sesiones_sella_actualizacion();
