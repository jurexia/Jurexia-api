-- Deja constancia de las veces que la plataforma tuvo que recuperarse sola.
--
-- El 4-sep-2026 se añadió un reintento sin caché para que el rechazo de Google
-- dejara de dejar a un abogado sin respuesta. Eso arregló el síntoma y creó un
-- punto ciego: la consulta ahora se responde, no se escribe ningún error, y la
-- alarma —que cuenta consultas sin respuesta— vería cero mientras el genio
-- contesta sin su corpus precargado. Una avería ruidosa se había convertido en
-- una degradación silenciosa.
--
-- Aquí se anota cada recuperación, para poder medir lo que ya no se ve.
-- Aplicada en producción el 4-sep-2026.
create table if not exists public.incidencias_modelo (
    id           bigserial   primary key,
    tipo         text        not null,
    genio        text,
    user_id      uuid,
    detalle      jsonb       not null default '{}'::jsonb,
    ocurrido_at  timestamptz not null default now()
);

comment on table public.incidencias_modelo is
    'Recuperaciones automáticas del modelo. Telemetría: sólo la escribe el servidor y sólo la lee la alarma.';
comment on column public.incidencias_modelo.tipo is
    'cache_rechazada = Google devolvió 400 con cached_content y se repitió sin caché.';

-- La alarma siempre pregunta lo mismo: cuántas de este tipo en la última hora.
create index if not exists incidencias_modelo_tipo_fecha
    on public.incidencias_modelo (tipo, ocurrido_at desc);

-- Sin políticas: es telemetría interna. El service_role se salta RLS, y nadie
-- más tiene por qué leerla ni escribirla.
alter table public.incidencias_modelo enable row level security;
