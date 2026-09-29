-- ESCRITURAS ATÓMICAS DEL ESTADO DEL TALLER (rediseño, etapa 2, prerrequisito; 29-sep-2026).
--
-- Hoy cada marca del taller —consulta, contraste, propuesta, deliberación, decisiva,
-- material— LEE el `estado` entero, cambia una rama y lo REESCRIBE entero, sin
-- comparar-y-escribir, con hasta cuatro tareas latiendo cada 45 s sobre la misma
-- fila: la última que escribe borra lo que otra acababa de guardar. Esta función
-- mezcla un PARCHE de claves de primer nivel en una sola sentencia
-- (`estado || parche`), con la guarda de la huella del adelanto en el mismo WHERE.
--
-- Seguridad (auditoría de seguridad, sep-2026): SECURITY DEFINER con search_path
-- fijo, y EXECUTE sólo para service_role; nada para public, anon ni authenticated.
-- Vuelta atrás: DROP FUNCTION public.taller_estado_parche(text, text, jsonb, text);
-- el código cae solo a la escritura de antes si la función no existe.

create or replace function public.taller_estado_parche(
    p_email text, p_expediente text, p_parche jsonb, p_huella text default null)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
    n integer;
begin
    if p_parche is null or jsonb_typeof(p_parche) <> 'object' then
        return false;
    end if;
    update public.taller_sesiones
       set estado = coalesce(estado, '{}'::jsonb) || p_parche
     where email = p_email
       and expediente = p_expediente
       and (p_huella is null or p_huella = '' or estado->>'huella' = p_huella);
    get diagnostics n = row_count;
    return n > 0;
end;
$$;

revoke all on function public.taller_estado_parche(text, text, jsonb, text) from public;
revoke all on function public.taller_estado_parche(text, text, jsonb, text) from anon;
revoke all on function public.taller_estado_parche(text, text, jsonb, text) from authenticated;
grant execute on function public.taller_estado_parche(text, text, jsonb, text) to service_role;
